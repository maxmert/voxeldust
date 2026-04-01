use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, RwLock};
use tokio_util::sync::CancellationToken;
use tracing::info;

use voxeldust_core::shard_types::{ShardId, ShardInfo, ShardState, ShardType};

use voxeldust_core::client_message::{PlayerInputData, ServerMsg};
use voxeldust_core::shard_message::ShardMsg;

use crate::client_listener::{self, ClientConnectEvent, ClientRegistry};
use crate::heartbeat_sender;
use crate::healthz;
use crate::peer_registry::PeerShardRegistry;
use crate::quic_transport::QuicTransport;
use crate::shutdown;
use crate::tick_loop::TickLoop;

/// Configuration for a shard harness.
#[derive(Debug, Clone)]
pub struct ShardHarnessConfig {
    pub shard_id: ShardId,
    pub shard_type: ShardType,
    pub tcp_addr: SocketAddr,
    pub udp_addr: SocketAddr,
    pub quic_addr: SocketAddr,
    pub orchestrator_url: String,
    /// Supports DNS names (e.g. "orchestrator.voxeldust.svc.cluster.local:9090").
    pub orchestrator_heartbeat_addr: String,
    pub healthz_addr: SocketAddr,
    /// For planet shards: the planet seed.
    pub planet_seed: Option<u64>,
    /// For system shards: the system seed.
    pub system_seed: Option<u64>,
    /// For ship shards: the ship entity id.
    pub ship_id: Option<u64>,
    /// For galaxy shards: the galaxy seed.
    pub galaxy_seed: Option<u64>,
    /// For ship shards: the shard managing this ship's exterior.
    pub host_shard_id: Option<ShardId>,
    /// Host to advertise to the orchestrator. If set, endpoints use this
    /// instead of the bind address. Needed in K8s (advertise 127.0.0.1
    /// for hostNetwork so clients can reach the shard via k3d port mapping).
    pub advertise_host: Option<String>,
}

/// The reusable shard skeleton. Every shard type embeds this.
pub struct ShardHarness {
    pub config: ShardHarnessConfig,
    pub tick_loop: TickLoop,
    pub peer_registry: Arc<RwLock<PeerShardRegistry>>,
    pub connect_rx: mpsc::UnboundedReceiver<ClientConnectEvent>,
    /// Client registry for tracking TCP + UDP connections.
    pub client_registry: Arc<RwLock<ClientRegistry>>,
    /// Incoming PlayerInput from UDP clients.
    pub input_rx: mpsc::UnboundedReceiver<(SocketAddr, PlayerInputData)>,
    /// Incoming inter-shard messages from QUIC.
    pub quic_msg_rx: mpsc::UnboundedReceiver<ShardMsg>,
    /// Channel to send WorldState for UDP broadcast (bounded for backpressure).
    pub broadcast_tx: mpsc::Sender<ServerMsg>,
    /// Channel to send QUIC messages to other shards (bounded for backpressure).
    /// Each message is (target_shard_id, target_quic_addr, message).
    pub quic_send_tx: mpsc::Sender<(ShardId, std::net::SocketAddr, ShardMsg)>,
    broadcast_rx: Option<mpsc::Receiver<ServerMsg>>,
    quic_send_rx: Option<mpsc::Receiver<(ShardId, std::net::SocketAddr, ShardMsg)>>,
    connect_tx: mpsc::UnboundedSender<ClientConnectEvent>,
    input_tx: mpsc::UnboundedSender<(SocketAddr, PlayerInputData)>,
    quic_msg_tx: mpsc::UnboundedSender<ShardMsg>,
    cancel: CancellationToken,
}

impl ShardHarness {
    pub fn new(config: ShardHarnessConfig) -> Self {
        let (connect_tx, connect_rx) = mpsc::unbounded_channel();
        let (input_tx, input_rx) = mpsc::unbounded_channel();
        let (quic_msg_tx, quic_msg_rx) = mpsc::unbounded_channel();
        let (broadcast_tx, broadcast_rx) = mpsc::channel(64);
        let (quic_send_tx, quic_send_rx) = mpsc::channel(256);

        Self {
            config,
            tick_loop: TickLoop::new(),
            peer_registry: Arc::new(RwLock::new(PeerShardRegistry::new())),
            connect_rx,
            client_registry: Arc::new(RwLock::new(ClientRegistry::new())),
            input_rx,
            quic_msg_rx,
            broadcast_tx,
            quic_send_tx,
            broadcast_rx: Some(broadcast_rx),
            quic_send_rx: Some(quic_send_rx),
            connect_tx,
            input_tx,
            quic_msg_tx,
            cancel: CancellationToken::new(),
        }
    }

    /// Register a system to run each tick.
    pub fn add_system(&mut self, name: impl Into<String>, func: impl FnMut() + Send + 'static) {
        self.tick_loop.add_system(name, func);
    }

    /// Get the cancellation token (for external shutdown triggers).
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    /// Run the harness: starts background tasks, enters tick loop.
    /// Returns when shutdown signal is received.
    pub async fn run(mut self) {
        let shard_id = self.config.shard_id;
        info!(shard_id = %shard_id, shard_type = %self.config.shard_type, "shard harness starting");

        // Register with orchestrator.
        self.register_with_orchestrator().await;

        // Start background tasks.
        let cancel = self.cancel.clone();

        // Healthz server.
        let healthz_cancel = cancel.clone();
        let healthz_addr = self.config.healthz_addr;
        tokio::spawn(async move {
            healthz::run_healthz_server(healthz_addr, healthz_cancel).await;
        });

        // TCP client listener.
        let tcp_cancel = cancel.clone();
        let tcp_addr = self.config.tcp_addr;
        let connect_tx = self.connect_tx.clone();
        tokio::spawn(async move {
            client_listener::run_tcp_listener(tcp_addr, connect_tx, tcp_cancel).await;
        });

        // Heartbeat sender.
        let hb_cancel = cancel.clone();
        let hb_addr = self.config.orchestrator_heartbeat_addr.clone();
        let tick_loop_ref = &self.tick_loop as *const TickLoop;
        // Safety: tick_loop lives as long as the harness, and the metrics_fn
        // only reads tick_ms/p99 which are updated on the same task.
        // For a proper impl we'd use Arc<Mutex<>>, but for now send static metrics.
        let shard_id_for_hb = shard_id;
        tokio::spawn(async move {
            heartbeat_sender::run_heartbeat_sender(
                shard_id_for_hb,
                hb_addr,
                Box::new(move || {
                    // Static metrics for now — will be wired to tick loop later
                    (0.0, 0.0, 0, 0)
                }),
                hb_cancel,
            )
            .await;
        });

        // Shutdown signal listener.
        let shutdown_cancel = cancel.clone();
        tokio::spawn(async move {
            shutdown::wait_for_shutdown_signal(shutdown_cancel).await;
        });

        // Peer registry refresh.
        let peer_registry = self.peer_registry.clone();
        let orchestrator_url = self.config.orchestrator_url.clone();
        let refresh_cancel = cancel.clone();
        tokio::spawn(async move {
            refresh_peer_registry(peer_registry, &orchestrator_url, refresh_cancel).await;
        });

        // UDP socket for client WorldState broadcast + PlayerInput receive.
        let udp_socket = Arc::new(
            tokio::net::UdpSocket::bind(self.config.udp_addr)
                .await
                .expect("failed to bind UDP socket"),
        );
        info!(addr = %self.config.udp_addr, "UDP socket bound");

        // WorldState broadcast drainer: reads from broadcast_tx channel, sends UDP.
        let broadcast_cancel = cancel.clone();
        let broadcast_socket = udp_socket.clone();
        let broadcast_registry = self.client_registry.clone();
        let mut broadcast_rx = self.broadcast_rx.take().expect("broadcast_rx already taken");
        tokio::spawn(async move {
            info!("broadcast drainer task started");
            loop {
                tokio::select! {
                    _ = broadcast_cancel.cancelled() => return,
                    msg = broadcast_rx.recv() => {
                        if let Some(msg) = msg {
                            client_listener::broadcast_world_state_udp(
                                &broadcast_socket, &broadcast_registry, &msg,
                            ).await;
                        } else {
                            return;
                        }
                    }
                }
            }
        });

        // UDP receiver task (PlayerInput + client discovery).
        let udp_recv_cancel = cancel.clone();
        let udp_recv_socket = udp_socket.clone();
        let udp_registry = self.client_registry.clone();
        let input_tx = self.input_tx.clone();
        tokio::spawn(async move {
            client_listener::run_udp_receiver(udp_recv_socket, udp_registry, input_tx, udp_recv_cancel).await;
        });

        // QUIC accept loop for inter-shard messages.
        let quic_cancel = cancel.clone();
        let quic_addr = self.config.quic_addr;
        let quic_msg_tx = self.quic_msg_tx.clone();
        tokio::spawn(async move {
            match QuicTransport::bind(quic_addr).await {
                Ok(transport) => {
                    info!(%quic_addr, "QUIC transport bound for inter-shard");
                    loop {
                        tokio::select! {
                            _ = quic_cancel.cancelled() => return,
                            conn = transport.accept() => {
                                if let Some(conn) = conn {
                                    let tx = quic_msg_tx.clone();
                                    tokio::spawn(async move {
                                        loop {
                                            match conn.recv().await {
                                                Ok(msg) => { let _ = tx.send(msg); }
                                                Err(_) => return,
                                            }
                                        }
                                    });
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(%e, "failed to bind QUIC transport");
                }
            }
        });

        // QUIC send drainer: reads from quic_send_tx channel, sends via QUIC transport.
        let quic_send_cancel = cancel.clone();
        let mut quic_send_rx = self.quic_send_rx.take().expect("quic_send_rx already taken");
        tokio::spawn(async move {
            // Create a dedicated QUIC transport for sending (separate from the accept transport).
            let send_transport = match QuicTransport::bind("0.0.0.0:0".parse().unwrap()).await {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(%e, "failed to bind QUIC send transport");
                    return;
                }
            };
            info!("QUIC send drainer task started");
            loop {
                tokio::select! {
                    _ = quic_send_cancel.cancelled() => return,
                    msg = quic_send_rx.recv() => {
                        if let Some((peer_id, peer_addr, shard_msg)) = msg {
                            if let Err(e) = send_transport.send(peer_id, peer_addr, &shard_msg).await {
                                tracing::debug!(%peer_id, %e, "failed to send QUIC message");
                            }
                        } else {
                            return;
                        }
                    }
                }
            }
        });

        // Main tick loop.
        info!(shard_id = %shard_id, "entering tick loop (20Hz)");
        let mut interval = tokio::time::interval(self.tick_loop.interval());

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    info!(shard_id = %shard_id, "shard shutting down");
                    break;
                }
                _ = interval.tick() => {
                    self.tick_loop.tick();
                }
            }
        }

        // Deregister from orchestrator.
        self.deregister_from_orchestrator().await;
        info!(shard_id = %shard_id, "shard stopped");
    }

    fn build_endpoint(&self) -> voxeldust_core::shard_types::ShardEndpoint {
        if let Some(ref host) = self.config.advertise_host {
            // Replace the bind address host with the advertise host, keeping ports.
            let make_addr = |addr: SocketAddr| -> SocketAddr {
                format!("{host}:{}", addr.port()).parse().unwrap_or(addr)
            };
            voxeldust_core::shard_types::ShardEndpoint {
                tcp_addr: make_addr(self.config.tcp_addr),
                udp_addr: make_addr(self.config.udp_addr),
                quic_addr: make_addr(self.config.quic_addr),
            }
        } else {
            voxeldust_core::shard_types::ShardEndpoint {
                tcp_addr: self.config.tcp_addr,
                udp_addr: self.config.udp_addr,
                quic_addr: self.config.quic_addr,
            }
        }
    }

    async fn register_with_orchestrator(&self) {
        let info = ShardInfo {
            id: self.config.shard_id,
            shard_type: self.config.shard_type,
            state: ShardState::Starting,
            endpoint: self.build_endpoint(),
            planet_seed: self.config.planet_seed,
            sectors: None,
            system_seed: self.config.system_seed,
            ship_id: self.config.ship_id,
            galaxy_seed: self.config.galaxy_seed,
            host_shard_id: self.config.host_shard_id,
            launch_args: vec![],
        };

        let url = format!("{}/register", self.config.orchestrator_url);
        match reqwest::Client::new().post(&url).json(&info).send().await {
            Ok(resp) if resp.status().is_success() => {
                info!(shard_id = %self.config.shard_id, "registered with orchestrator");
            }
            Ok(resp) => {
                tracing::warn!(
                    shard_id = %self.config.shard_id,
                    status = %resp.status(),
                    "orchestrator registration returned non-success"
                );
            }
            Err(e) => {
                tracing::warn!(
                    shard_id = %self.config.shard_id,
                    %e,
                    "failed to register with orchestrator (will retry via heartbeat)"
                );
            }
        }
    }

    async fn deregister_from_orchestrator(&self) {
        let info = ShardInfo {
            id: self.config.shard_id,
            shard_type: self.config.shard_type,
            state: ShardState::Stopped,
            endpoint: self.build_endpoint(),
            planet_seed: self.config.planet_seed,
            sectors: None,
            system_seed: self.config.system_seed,
            ship_id: self.config.ship_id,
            galaxy_seed: self.config.galaxy_seed,
            host_shard_id: self.config.host_shard_id,
            launch_args: vec![],
        };

        let url = format!("{}/register", self.config.orchestrator_url);
        let _ = reqwest::Client::new().post(&url).json(&info).send().await;
    }
}

/// Periodically refreshes the peer shard registry from the orchestrator.
async fn refresh_peer_registry(
    registry: Arc<RwLock<PeerShardRegistry>>,
    orchestrator_url: &str,
    cancel: CancellationToken,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(10));
    let client = reqwest::Client::new();
    let url = format!("{orchestrator_url}/shards");

    loop {
        tokio::select! {
            _ = cancel.cancelled() => return,
            _ = interval.tick() => {
                match client.get(&url).send().await {
                    Ok(resp) => {
                        #[derive(serde::Deserialize)]
                        struct ShardsResponse {
                            shards: Vec<ShardInfo>,
                        }

                        if let Ok(body) = resp.json::<ShardsResponse>().await {
                            let mut reg = registry.write().await;
                            reg.update(body.shards);
                        }
                    }
                    Err(_) => {
                        // Orchestrator might not be available yet — that's fine.
                    }
                }
            }
        }
    }
}
