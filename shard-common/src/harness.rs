use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use bevy_app::App;
use bevy_ecs::prelude::*;
use tokio::sync::{mpsc, RwLock};
use tokio_util::sync::CancellationToken;
use tracing::info;

use voxeldust_core::shard_types::{ShardId, ShardInfo, ShardState, ShardType};

use voxeldust_core::client_message::{BlockEditData, PlayerInputData, ServerMsg};
use voxeldust_core::shard_message::ShardMsg;

/// QUIC message with verified source shard identification.
/// The sender writes its ShardId as a header on each QUIC stream.
/// The receiver reads it and attaches to every message. This is
/// immune to ephemeral port mismatches and race conditions.
pub struct QueuedShardMsg {
    /// The shard that sent this message (from the stream identity header).
    pub source_shard_id: ShardId,
    /// The deserialized shard message.
    pub msg: ShardMsg,
}

use crate::client_listener::{self, ClientConnectEvent, ClientRegistry};
use crate::heartbeat_sender;
use crate::healthz;
use crate::peer_registry::PeerShardRegistry;
use crate::quic_transport::QuicTransport;
use crate::shutdown;

/// ECS resource wrapping all async networking channels.
///
/// Bridge systems drain the receivers at the start of each tick, converting
/// messages into ECS events. Send channels are used by broadcast/QUIC systems.
#[derive(Resource)]
pub struct NetworkBridge {
    /// Incoming TCP client connections.
    pub connect_rx: mpsc::UnboundedReceiver<ClientConnectEvent>,
    /// Incoming player input from UDP.
    pub input_rx: mpsc::UnboundedReceiver<(SocketAddr, PlayerInputData)>,
    /// Incoming block edit requests from UDP.
    pub block_edit_rx: mpsc::UnboundedReceiver<BlockEditData>,
    /// Incoming inter-shard messages from QUIC.
    pub quic_msg_rx: mpsc::UnboundedReceiver<QueuedShardMsg>,
    /// Send WorldState for UDP broadcast.
    pub broadcast_tx: mpsc::Sender<ServerMsg>,
    /// Send QUIC messages to other shards.
    pub quic_send_tx: mpsc::Sender<(ShardId, SocketAddr, ShardMsg)>,
    /// Client registry (TCP + UDP connections).
    pub client_registry: Arc<RwLock<ClientRegistry>>,
    /// Peer shard registry (discovery).
    pub peer_registry: Arc<RwLock<PeerShardRegistry>>,
    /// Universe epoch in Unix milliseconds (for deterministic celestial_time).
    pub universe_epoch_ms: Arc<AtomicU64>,
}

/// Compute deterministic celestial time from the universal epoch.
/// All shards derive the same value from wall clock — no sync messages needed.
/// `epoch` is the shared universe epoch Arc (from `harness.epoch_arc()`).
pub fn celestial_time_from_epoch(epoch: &AtomicU64, time_scale: f64) -> f64 {
    let epoch_ms = epoch.load(Ordering::Relaxed);
    let now_ms = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let elapsed_ms = now_ms.saturating_sub(epoch_ms);
    (elapsed_ms as f64 / 1000.0) * time_scale
}

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
    pub peer_registry: Arc<RwLock<PeerShardRegistry>>,
    /// Universe epoch in Unix milliseconds (from orchestrator). Shared with tick
    /// closures via Arc so they can compute deterministic celestial_time.
    pub universe_epoch_ms: Arc<AtomicU64>,
    pub connect_rx: mpsc::UnboundedReceiver<ClientConnectEvent>,
    /// Client registry for tracking TCP + UDP connections.
    pub client_registry: Arc<RwLock<ClientRegistry>>,
    /// Incoming PlayerInput from UDP clients.
    pub input_rx: mpsc::UnboundedReceiver<(SocketAddr, PlayerInputData)>,
    /// Incoming BlockEditRequest from UDP clients.
    pub block_edit_rx: mpsc::UnboundedReceiver<BlockEditData>,
    /// Incoming inter-shard messages from QUIC (with source peer address).
    pub quic_msg_rx: mpsc::UnboundedReceiver<QueuedShardMsg>,
    /// Channel to send WorldState for UDP broadcast (bounded for backpressure).
    pub broadcast_tx: mpsc::Sender<ServerMsg>,
    /// Channel to send QUIC messages to other shards (bounded for backpressure).
    /// Each message is (target_shard_id, target_quic_addr, message).
    pub quic_send_tx: mpsc::Sender<(ShardId, std::net::SocketAddr, ShardMsg)>,
    broadcast_rx: Option<mpsc::Receiver<ServerMsg>>,
    quic_send_rx: Option<mpsc::Receiver<(ShardId, std::net::SocketAddr, ShardMsg)>>,
    connect_tx: mpsc::UnboundedSender<ClientConnectEvent>,
    input_tx: mpsc::UnboundedSender<(SocketAddr, PlayerInputData)>,
    block_edit_tx: mpsc::UnboundedSender<BlockEditData>,
    quic_msg_tx: mpsc::UnboundedSender<QueuedShardMsg>,
    cancel: CancellationToken,
}

impl ShardHarness {
    pub fn new(config: ShardHarnessConfig) -> Self {
        let (connect_tx, connect_rx) = mpsc::unbounded_channel();
        let (input_tx, input_rx) = mpsc::unbounded_channel();
        let (block_edit_tx, block_edit_rx) = mpsc::unbounded_channel();
        let (quic_msg_tx, quic_msg_rx) = mpsc::unbounded_channel();
        let (broadcast_tx, broadcast_rx) = mpsc::channel(64);
        let (quic_send_tx, quic_send_rx) = mpsc::channel(256);

        // Default epoch: current wall clock (celestial_time starts at 0).
        // Overwritten with the orchestrator's epoch during registration.
        let now_unix_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            config,
            peer_registry: Arc::new(RwLock::new(PeerShardRegistry::new())),
            universe_epoch_ms: Arc::new(AtomicU64::new(now_unix_ms)),
            connect_rx,
            client_registry: Arc::new(RwLock::new(ClientRegistry::new())),
            input_rx,
            block_edit_rx,
            quic_msg_rx,
            broadcast_tx,
            quic_send_tx,
            broadcast_rx: Some(broadcast_rx),
            quic_send_rx: Some(quic_send_rx),
            connect_tx,
            input_tx,
            block_edit_tx,
            quic_msg_tx,
            cancel: CancellationToken::new(),
        }
    }

    /// Get the shared universe epoch Arc (for tick closures to capture).
    pub fn epoch_arc(&self) -> Arc<AtomicU64> {
        self.universe_epoch_ms.clone()
    }

    /// Get the cancellation token (for external shutdown triggers).
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    /// Spawn all background networking tasks (TCP, UDP, QUIC, healthz, heartbeat, etc.).
    /// Called by `run_ecs()` before entering the tick loop.
    async fn start_networking(&mut self) {
        let shard_id = self.config.shard_id;

        // Register with orchestrator.
        self.register_with_orchestrator().await;

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
            let mut packet_buf = Vec::with_capacity(8192);
            loop {
                tokio::select! {
                    _ = broadcast_cancel.cancelled() => return,
                    msg = broadcast_rx.recv() => {
                        if let Some(msg) = msg {
                            client_listener::broadcast_world_state_udp(
                                &broadcast_socket, &broadcast_registry, &msg,
                                &mut packet_buf,
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
        let block_edit_tx = self.block_edit_tx.clone();
        tokio::spawn(async move {
            client_listener::run_udp_receiver(udp_recv_socket, udp_registry, input_tx, block_edit_tx, udp_recv_cancel).await;
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
                                        let _ = conn.recv_loop_sourced(tx).await;
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

        // QUIC send drainer: per-peer isolated async tasks (AAA MMO pattern).
        // Each peer owns its QUIC connection directly — no shared mutex. A dead peer's
        // connection timeout only blocks its own task. The dispatcher routes messages
        // to the correct per-peer channel; no shared I/O state.
        let quic_send_cancel = cancel.clone();
        let mut quic_send_rx = self.quic_send_rx.take().expect("quic_send_rx already taken");
        let local_shard_id = self.config.shard_id;
        tokio::spawn(async move {
            // Shared QUIC endpoint — thread-safe, connect() takes &self.
            let send_transport = match QuicTransport::bind("0.0.0.0:0".parse().unwrap()).await {
                Ok(t) => Arc::new(t),
                Err(e) => {
                    tracing::warn!(%e, "failed to bind QUIC send transport");
                    return;
                }
            };
            info!("QUIC send dispatcher started (per-peer connection ownership)");

            let mut peer_senders: std::collections::HashMap<
                ShardId,
                mpsc::Sender<(SocketAddr, ShardMsg)>,
            > = std::collections::HashMap::new();

            loop {
                tokio::select! {
                    _ = quic_send_cancel.cancelled() => return,
                    msg = quic_send_rx.recv() => {
                        if let Some((peer_id, peer_addr, shard_msg)) = msg {
                            let tx = peer_senders.entry(peer_id).or_insert_with(|| {
                                let (tx, mut rx) = mpsc::channel::<(SocketAddr, ShardMsg)>(32);
                                let endpoint = send_transport.endpoint().clone();
                                let cancel = quic_send_cancel.clone();

                                // Per-peer task: owns its own QUIC connection lifecycle.
                                tokio::spawn(async move {
                                    let mut conn: Option<quinn::Connection> = None;
                                    let mut send_stream: Option<quinn::SendStream> = None;
                                    let mut breaker = crate::circuit_breaker::CircuitBreaker::new();
                                    let timeout_dur = std::time::Duration::from_secs(2);

                                    loop {
                                        tokio::select! {
                                            _ = cancel.cancelled() => return,
                                            msg = rx.recv() => {
                                                let (addr, shard_msg) = match msg {
                                                    Some(m) => m,
                                                    None => return,
                                                };

                                                // Circuit breaker with half-open recovery.
                                                if !breaker.allow_request() {
                                                    continue;
                                                }

                                                // Get or create connection.
                                                let needs_connect = match &conn {
                                                    Some(c) => c.close_reason().is_some(),
                                                    None => true,
                                                };
                                                if needs_connect {
                                                    conn = None;
                                                    send_stream = None;
                                                    match tokio::time::timeout(timeout_dur, async {
                                                        let connecting = endpoint.connect(addr, "voxeldust-shard")
                                                            .map_err(|e| format!("{e}"))?;
                                                        connecting.await.map_err(|e| format!("{e}"))
                                                    }).await {
                                                        Ok(Ok(c)) => {
                                                            if breaker.consecutive_failures() > 0 {
                                                                tracing::info!(%peer_id,
                                                                    "QUIC connection recovered after {} failures",
                                                                    breaker.consecutive_failures());
                                                            }
                                                            conn = Some(c);
                                                            breaker.record_success();
                                                        }
                                                        Ok(Err(e)) => {
                                                            breaker.record_failure();
                                                            if breaker.consecutive_failures() == 5 {
                                                                tracing::warn!(%peer_id, %e,
                                                                    "circuit breaker opened for peer");
                                                            }
                                                            continue;
                                                        }
                                                        Err(_) => {
                                                            breaker.record_failure();
                                                            if breaker.consecutive_failures() == 5 {
                                                                tracing::warn!(%peer_id,
                                                                    "circuit breaker opened for peer (timeout)");
                                                            }
                                                            continue;
                                                        }
                                                    }
                                                }

                                                // Send message.
                                                let c = conn.as_ref().unwrap();
                                                let data = shard_msg.serialize();

                                                // Open stream if needed. Write shard ID header
                                                // so the receiver knows who sent these messages.
                                                if send_stream.is_none() {
                                                    match c.open_uni().await {
                                                        Ok(mut s) => {
                                                            let id_bytes = local_shard_id.0.to_be_bytes();
                                                            if s.write_all(&id_bytes).await.is_err() {
                                                                conn = None;
                                                                breaker.record_failure();
                                                                continue;
                                                            }
                                                            send_stream = Some(s);
                                                        }
                                                        Err(e) => {
                                                            tracing::debug!(%peer_id, %e, "failed to open uni stream");
                                                            conn = None;
                                                            send_stream = None;
                                                            breaker.record_failure();
                                                            continue;
                                                        }
                                                    }
                                                }

                                                let s = send_stream.as_mut().unwrap();
                                                let len_bytes = (data.len() as u32).to_be_bytes();
                                                if s.write_all(&len_bytes).await.is_err()
                                                    || s.write_all(&data).await.is_err()
                                                {
                                                    // Stream broken — discard, reconnect next time.
                                                    send_stream = None;
                                                    conn = None;
                                                    breaker.record_failure();
                                                } else {
                                                    breaker.record_success();
                                                }
                                            }
                                        }
                                    }
                                });
                                tx
                            });
                            let _ = tx.try_send((peer_addr, shard_msg));
                        } else {
                            return;
                        }
                    }
                }
            }
        });
    }

    /// Run the harness with a bevy_ecs App.
    ///
    /// The shard builds its `App` with systems and resources, then passes it here.
    /// This method:
    /// 1. Spawns all networking background tasks
    /// 2. Inserts a `NetworkBridge` resource into the App with all mpsc channels
    /// 3. Drives `app.update()` at 20Hz (50ms tick interval) with per-tick diagnostics
    ///
    /// The shard's bevy systems drain the `NetworkBridge` channels at the start
    /// of each tick, converting async messages into ECS events.
    pub async fn run_ecs(mut self, mut app: App) {
        let shard_id = self.config.shard_id;
        info!(shard_id = %shard_id, shard_type = %self.config.shard_type, "shard harness starting");

        self.start_networking().await;

        // Capture deregistration info before moving receivers out of self.
        let deregister_info = ShardInfo {
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
        let deregister_url = format!("{}/register", self.config.orchestrator_url);

        // Move networking channels into the App as a Resource.
        // After this, self's receivers are consumed (they were UnboundedReceivers,
        // which don't impl Clone — ownership transfers to the ECS world).
        let bridge = NetworkBridge {
            connect_rx: self.connect_rx,
            input_rx: self.input_rx,
            block_edit_rx: self.block_edit_rx,
            quic_msg_rx: self.quic_msg_rx,
            broadcast_tx: self.broadcast_tx.clone(),
            quic_send_tx: self.quic_send_tx.clone(),
            client_registry: self.client_registry.clone(),
            peer_registry: self.peer_registry.clone(),
            universe_epoch_ms: self.universe_epoch_ms.clone(),
        };
        app.insert_resource(bridge);

        // Main tick loop — drives bevy App at 20Hz with per-tick timing diagnostics.
        let cancel = self.cancel.clone();
        let tick_interval = Duration::from_millis(50); // 20Hz
        let budget_ms = tick_interval.as_secs_f32() * 1000.0;
        info!(shard_id = %shard_id, "entering tick loop (20Hz)");
        let mut interval = tokio::time::interval(tick_interval);
        let mut tick_count: u64 = 0;
        let mut tick_history: std::collections::VecDeque<f32> =
            std::collections::VecDeque::with_capacity(100);

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    info!(shard_id = %shard_id, "shard shutting down");
                    break;
                }
                _ = interval.tick() => {
                    let tick_start = std::time::Instant::now();
                    app.update();
                    let elapsed_ms = tick_start.elapsed().as_secs_f32() * 1000.0;
                    tick_count += 1;

                    // Rolling p99 tracker (last 100 ticks).
                    if tick_history.len() >= 100 {
                        tick_history.pop_front();
                    }
                    tick_history.push_back(elapsed_ms);

                    if elapsed_ms > budget_ms {
                        tracing::warn!(
                            tick = tick_count,
                            elapsed_ms = format!("{:.2}", elapsed_ms),
                            budget_ms = format!("{:.2}", budget_ms),
                            "tick exceeded budget"
                        );
                    }

                    tracing::debug!(
                        tick = tick_count,
                        elapsed_us = tick_start.elapsed().as_micros(),
                        "tick"
                    );
                }
            }
        }

        // Deregister from orchestrator.
        let _ = reqwest::Client::new().post(&deregister_url).json(&deregister_info).send().await;
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
                // Parse universe epoch from orchestrator response.
                if let Ok(body) = resp.json::<serde_json::Value>().await {
                    if let Some(epoch_ms) = body.get("universe_epoch_ms").and_then(|v| v.as_u64()) {
                        self.universe_epoch_ms.store(epoch_ms, Ordering::Relaxed);
                        info!(shard_id = %self.config.shard_id, universe_epoch_ms = epoch_ms,
                            "registered with orchestrator (epoch synced)");
                    } else {
                        info!(shard_id = %self.config.shard_id, "registered with orchestrator");
                    }
                } else {
                    info!(shard_id = %self.config.shard_id, "registered with orchestrator");
                }
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
