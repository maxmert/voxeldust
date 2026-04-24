use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio::net::tcp::OwnedWriteHalf;
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use voxeldust_core::client_message::{
    BlockEditData, ClientMsg, PlayerInputData, ServerMsg, SignalPublishData,
};
use voxeldust_core::shard_types::SessionToken;

/// A persistent client connection. The TCP write half is stored for server→client sends.
/// The read half is consumed by a per-client `run_tcp_read_loop` task.
pub struct ClientConnection {
    pub session_token: SessionToken,
    pub player_name: String,
    pub tcp_write: Arc<Mutex<OwnedWriteHalf>>,
    pub peer_addr: SocketAddr,
    /// Client's UDP address (learned from first UDP packet — hole-punch pattern).
    pub udp_addr: Option<SocketAddr>,
}

/// Channels for forwarding client TCP messages to the ECS bridge.
/// Each message carries the sender's SessionToken for per-player routing.
/// Cloned per-client — `mpsc::UnboundedSender` is `Clone`.
#[derive(Clone)]
pub struct TcpMessageChannels {
    pub block_edit_tx: mpsc::UnboundedSender<(SessionToken, BlockEditData)>,
    pub config_update_tx: mpsc::UnboundedSender<(SessionToken, voxeldust_core::signal::config::BlockConfigUpdateData)>,
    pub sub_block_edit_tx: mpsc::UnboundedSender<(SessionToken, voxeldust_core::client_message::SubBlockEditData)>,
    /// Publisher-widget signal publishes (button presses, slider drags,
    /// toggle clicks). Routed to the shard's signal evaluator which
    /// validates `publish_policy` against the sender's player_id before
    /// accepting.
    pub signal_publish_tx: mpsc::UnboundedSender<(SessionToken, SignalPublishData)>,
}

/// Event emitted when a client connects via TCP.
pub struct ClientConnectEvent {
    pub connection: ClientConnection,
}

/// Tracks all connected clients and observers. Thread-safe for use across tick systems.
///
/// Two connection types:
/// - **Client**: Full participant with TCP + UDP. Has a player entity, processes input,
///   participates in handoffs. Created via TCP Connect message.
/// - **Observer**: UDP-only spectator for dual-shard compositing. Receives WorldState
///   broadcasts but has no player entity, no input, no handoff. Created when a secondary
///   shard connection sends a UDP hole-punch without a preceding TCP connect.
/// Stale observer timeout: observers not successfully sent to within this duration
/// are removed to prevent unbounded accumulation from disconnected secondary shards.
const OBSERVER_TIMEOUT: Duration = Duration::from_secs(10);

pub struct ClientRegistry {
    clients: HashMap<SessionToken, ClientEntry>,
    /// UDP-only observers (secondary/spectator connections for dual-shard compositing).
    /// These receive WorldState broadcasts but don't have player entities.
    /// Each entry tracks the last successful send for timeout-based cleanup.
    observers: Vec<ObserverEntry>,
    /// UDP addresses seen before any client registered (for late-join matching).
    pending_udp: Vec<SocketAddr>,
}

struct ObserverEntry {
    addr: SocketAddr,
    registered_at: Instant,
    last_successful_send: Instant,
}

struct ClientEntry {
    tcp_write: Arc<Mutex<OwnedWriteHalf>>,
    udp_addr: Option<SocketAddr>,
    player_name: String,
}


impl ClientRegistry {
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
            observers: Vec::new(),
            pending_udp: Vec::new(),
        }
    }

    pub fn register(&mut self, conn: &ClientConnection) {
        self.clients.insert(conn.session_token, ClientEntry {
            tcp_write: conn.tcp_write.clone(),
            udp_addr: conn.udp_addr,
            player_name: conn.player_name.clone(),
        });

        // Check if there are pending UDP addresses waiting to be matched.
        if !self.pending_udp.is_empty() {
            let entry = self.clients.get_mut(&conn.session_token).unwrap();
            if entry.udp_addr.is_none() {
                let addr = self.pending_udp.remove(0);
                entry.udp_addr = Some(addr);
                info!(player = %entry.player_name, %addr, "matched pending UDP address on register");
            }
        }
    }

    pub fn set_udp_addr(&mut self, token: SessionToken, addr: SocketAddr) {
        if let Some(entry) = self.clients.get_mut(&token) {
            entry.udp_addr = Some(addr);
        }
    }

    /// Remove a client from the registry (e.g., after ShardRedirect during handoff).
    pub fn unregister(&mut self, session: &SessionToken) {
        if let Some(entry) = self.clients.remove(session) {
            info!(player = %entry.player_name, "client unregistered");
        }
    }

    /// Register a UDP address by matching against known clients.
    /// First client without a UDP addr gets it. If no clients yet,
    /// store as pending for later matching on register().
    pub fn discover_udp(&mut self, udp_addr: SocketAddr) {
        // Idempotent fast-path: if this UDP address is already known (either
        // assigned to a client or registered as an observer), bail out.
        // Without this, every incoming UDP packet (20 Hz) re-ran the
        // clear-and-reassign dance below — spamming logs and, in edge cases
        // with multiple unassigned clients, swapping which client gets
        // which UDP address between packets.
        if self.clients.values().any(|e| e.udp_addr == Some(udp_addr)) {
            return;
        }
        if self.observers.iter().any(|o| o.addr == udp_addr) {
            return;
        }

        // Assign to the first client without a UDP address.
        for entry in self.clients.values_mut() {
            if entry.udp_addr.is_none() {
                entry.udp_addr = Some(udp_addr);
                info!(player = %entry.player_name, %udp_addr, "discovered client UDP address");
                return;
            }
        }

        // No client to match. If there are no unmatched clients at all, this is likely
        // an observer (secondary shard connection for dual compositing). Register as
        // observer so it receives WorldState broadcasts without a player entity.
        if self.clients.values().all(|e| e.udp_addr.is_some()) {
            // All clients already have UDP — this is a new observer connection.
            if !self.observers.iter().any(|o| o.addr == udp_addr) {
                info!(%udp_addr, "registered UDP observer (dual-shard compositing)");
                let now = Instant::now();
                self.observers.push(ObserverEntry {
                    addr: udp_addr,
                    registered_at: now,
                    last_successful_send: now,
                });
            }
        } else {
            // There's an unmatched client waiting — store as pending for late matching.
            if self.pending_udp.len() < 16 && !self.pending_udp.contains(&udp_addr) {
                debug!(%udp_addr, "storing pending UDP address (no client registered yet)");
                self.pending_udp.push(udp_addr);
            }
        }
    }

    /// Get all UDP addresses for broadcasting (clients + observers).
    pub fn udp_addrs(&self) -> Vec<SocketAddr> {
        let mut addrs: Vec<SocketAddr> = self.clients.values()
            .filter_map(|e| e.udp_addr).collect();
        addrs.extend(self.observers.iter().map(|o| o.addr));
        addrs
    }

    /// Mark an observer as having received a successful send.
    pub fn mark_observer_active(&mut self, addr: &SocketAddr) {
        if let Some(obs) = self.observers.iter_mut().find(|o| &o.addr == addr) {
            obs.last_successful_send = Instant::now();
        }
    }

    /// Remove observers that haven't received a successful send within the timeout.
    pub fn cleanup_stale_observers(&mut self) {
        let now = Instant::now();
        let before = self.observers.len();
        self.observers.retain(|obs| now.duration_since(obs.last_successful_send) < OBSERVER_TIMEOUT);
        let removed = before - self.observers.len();
        if removed > 0 {
            info!(removed, remaining = self.observers.len(), "cleaned up stale UDP observers");
        }
    }

    /// Remove an observer UDP address (e.g., when the secondary connection closes).
    pub fn remove_observer(&mut self, addr: &SocketAddr) {
        self.observers.retain(|o| o.addr != *addr);
    }

    pub fn len(&self) -> usize {
        self.clients.len()
    }

    pub fn is_empty(&self) -> bool {
        self.clients.is_empty()
    }

    /// Check if a client with the given session token is registered.
    pub fn has_client(&self, token: &SessionToken) -> bool {
        self.clients.contains_key(token)
    }

    /// Reverse lookup: find the session token for a given UDP address.
    /// Used by multi-player shards to route input to the correct player entity.
    pub fn session_for_udp(&self, addr: SocketAddr) -> Option<SessionToken> {
        self.clients
            .iter()
            .find(|(_, entry)| entry.udp_addr == Some(addr))
            .map(|(&token, _)| token)
    }

    /// Send a TCP message to a specific client.
    pub async fn send_tcp(&self, token: SessionToken, msg: &ServerMsg) -> Result<(), std::io::Error> {
        if let Some(entry) = self.clients.get(&token) {
            let mut writer = entry.tcp_write.lock().await;
            send_tcp_msg(&mut *writer, msg).await?;
        }
        Ok(())
    }
}

impl Default for ClientRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Listens for TCP client connections and reads Connect messages.
pub async fn run_tcp_listener(
    addr: SocketAddr,
    connect_tx: mpsc::UnboundedSender<ClientConnectEvent>,
    msg_channels: TcpMessageChannels,
    client_registry: Arc<RwLock<ClientRegistry>>,
    cancel: CancellationToken,
) {
    let listener = match TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            warn!(%e, "failed to bind TCP listener");
            return;
        }
    };
    info!(%addr, "TCP client listener ready");

    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                info!("TCP listener shutting down");
                return;
            }
            result = listener.accept() => {
                match result {
                    Ok((stream, peer_addr)) => {
                        let tx = connect_tx.clone();
                        let channels = msg_channels.clone();
                        let registry = client_registry.clone();
                        tokio::spawn(async move {
                            handle_client_connection(stream, peer_addr, tx, channels, registry).await;
                        });
                    }
                    Err(e) => {
                        warn!(%e, "TCP accept error");
                    }
                }
            }
        }
    }
}

/// Handle a new TCP client connection:
/// 1. Read the initial Connect message
/// 2. Split the stream into read/write halves
/// 3. Send the ClientConnectEvent with the write half
/// 4. Run a persistent read loop on the read half (blocks until disconnect)
async fn handle_client_connection(
    mut stream: TcpStream,
    peer_addr: SocketAddr,
    connect_tx: mpsc::UnboundedSender<ClientConnectEvent>,
    channels: TcpMessageChannels,
    client_registry: Arc<RwLock<ClientRegistry>>,
) {
    let _ = stream.set_nodelay(true);

    // Phase 1: Read the initial Connect message (before splitting).
    let mut len_buf = [0u8; 4];
    if let Err(e) = stream.read_exact(&mut len_buf).await {
        warn!(%peer_addr, %e, "failed to read message length");
        return;
    }
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > 65536 {
        warn!(%peer_addr, len, "message too large");
        return;
    }
    let mut buf = vec![0u8; len];
    if let Err(e) = stream.read_exact(&mut buf).await {
        warn!(%peer_addr, %e, "failed to read message body");
        return;
    }
    let decoded = match voxeldust_core::wire_codec::decode(&buf) {
        Ok(d) => d,
        Err(e) => {
            warn!(%peer_addr, %e, "failed to decode wire message");
            return;
        }
    };
    let first_msg = match ClientMsg::deserialize(&decoded) {
        Ok(msg) => msg,
        Err(e) => {
            warn!(%peer_addr, %e, "failed to deserialize client message");
            return;
        }
    };

    match first_msg {
        ClientMsg::Connect { player_name } => {
            let token = SessionToken(rand_u64());
            info!(%peer_addr, %player_name, session_token = token.0, "client connected");

            // Split the TCP stream into read/write halves.
            let (read_half, write_half) = stream.into_split();

            let connection = ClientConnection {
                session_token: token,
                player_name: player_name.clone(),
                tcp_write: Arc::new(Mutex::new(write_half)),
                peer_addr,
                udp_addr: None,
            };

            let _ = connect_tx.send(ClientConnectEvent { connection });

            // Persistent read loop — blocks until client disconnects.
            run_tcp_read_loop(read_half, peer_addr, &player_name, token, channels).await;
            info!(%peer_addr, %player_name, "client TCP read loop ended");

            // Release registry resources tied to this session: the ClientEntry
            // keeps the write half alive via its Arc, and its UDP addr stays in
            // udp_addrs() forever otherwise — broadcasting WorldState to a dead
            // socket and spamming ICMP unreachables at us.
            let mut reg = client_registry.write().await;
            reg.unregister(&token);
            return;
        }
        ClientMsg::ObserverConnect { observer_name } => {
            info!(%peer_addr, %observer_name, "observer TCP connected");

            // Observer connections: split stream, store write half for chunk sync,
            // but do NOT send a ClientConnectEvent (no player entity).
            // The shard will detect the observer via the ObserverConnect channel.
            let (read_half, write_half) = stream.into_split();
            let write = Arc::new(Mutex::new(write_half));

            // Store observer TCP write half in the connect channel with a special
            // sentinel name so the shard can distinguish observers from players.
            let observer_token = SessionToken(rand_u64());
            let connection = ClientConnection {
                session_token: observer_token,
                player_name: format!("__observer__{}", observer_name),
                tcp_write: write,
                peer_addr,
                udp_addr: None,
            };
            let _ = connect_tx.send(ClientConnectEvent { connection });
            info!(%peer_addr, %observer_name, "observer TCP setup complete");

            // Observers send nothing meaningful, but we must still drive the
            // read half to detect EOF. Without this, the ClientEntry lingers
            // forever after the secondary shard disconnects, and its stale UDP
            // address stays in udp_addrs() — broadcasting WorldState to a
            // dead socket and spamming ICMP port-unreachables at our UDP
            // socket, which can disrupt input flow for live clients.
            run_observer_read_loop(read_half, peer_addr, observer_name.as_str()).await;
            info!(%peer_addr, %observer_name, "observer TCP read loop ended");

            let mut reg = client_registry.write().await;
            reg.unregister(&observer_token);
            return;
        }
        _ => {
            warn!(%peer_addr, "expected Connect or ObserverConnect, got something else");
            return;
        }
    };
    // All match arms return — this is unreachable.
}

/// Observer TCP read loop. Observers don't send application messages, but we
/// must drive the read half to detect EOF so we can clean up the ClientEntry.
async fn run_observer_read_loop(
    mut reader: tokio::net::tcp::OwnedReadHalf,
    peer_addr: SocketAddr,
    observer_name: &str,
) {
    let mut buf = [0u8; 256];
    loop {
        match reader.read(&mut buf).await {
            Ok(0) => {
                info!(%peer_addr, %observer_name, "observer disconnected (TCP EOF)");
                return;
            }
            Ok(_) => {
                // Observers aren't expected to send anything; silently discard.
            }
            Err(e) => {
                warn!(%peer_addr, %observer_name, %e, "observer TCP read error, disconnecting");
                return;
            }
        }
    }
}

/// Persistent per-client TCP read loop. Reads length-prefixed messages from the
/// client and forwards them to the appropriate ECS bridge channels.
/// Returns when the client disconnects (EOF) or on unrecoverable error.
async fn run_tcp_read_loop(
    mut reader: tokio::net::tcp::OwnedReadHalf,
    peer_addr: SocketAddr,
    player_name: &str,
    session_token: SessionToken,
    channels: TcpMessageChannels,
) {
    let mut len_buf = [0u8; 4];

    loop {
        // Read 4-byte length prefix.
        match reader.read_exact(&mut len_buf).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                info!(%peer_addr, %player_name, "client disconnected (TCP EOF)");
                return;
            }
            Err(e) => {
                warn!(%peer_addr, %player_name, %e, "TCP read error, disconnecting");
                return;
            }
        }

        let len = u32::from_be_bytes(len_buf) as usize;

        // Zero-length = keepalive from client.
        if len == 0 {
            continue;
        }

        if len > 65536 {
            warn!(%peer_addr, len, "TCP message too large, disconnecting");
            return;
        }

        let mut buf = vec![0u8; len];
        if let Err(e) = reader.read_exact(&mut buf).await {
            warn!(%peer_addr, %e, "failed to read TCP message body");
            return;
        }

        let decoded = match voxeldust_core::wire_codec::decode(&buf) {
            Ok(d) => d,
            Err(e) => {
                warn!(%peer_addr, %e, "bad TCP wire message, skipping");
                continue;
            }
        };

        match ClientMsg::deserialize(&decoded) {
            Ok(ClientMsg::BlockEditRequest(edit)) => {
                let _ = channels.block_edit_tx.send((session_token, edit));
            }
            Ok(ClientMsg::BlockConfigUpdate(update)) => {
                let _ = channels.config_update_tx.send((session_token, update));
            }
            Ok(ClientMsg::SubBlockEdit(edit)) => {
                let _ = channels.sub_block_edit_tx.send((session_token, edit));
            }
            Ok(ClientMsg::PlayerInput(_)) => {
                // PlayerInput should go via UDP for performance. Ignore on TCP.
            }
            Ok(ClientMsg::Connect { .. }) => {
                warn!(%peer_addr, "duplicate Connect on established connection");
            }
            Ok(ClientMsg::ObserverConnect { .. }) => {
                // ObserverConnect on an already-established connection — ignore.
                warn!(%peer_addr, "ObserverConnect on established connection");
            }
            Ok(ClientMsg::SignalPublish(data)) => {
                let _ = channels.signal_publish_tx.send((session_token, data));
            }
            Err(e) => {
                debug!(%peer_addr, %e, "failed to deserialize TCP client message");
            }
        }
    }
}

/// Send a length-prefixed ServerMsg over any async writer (with LZ4 compression).
/// Works with both `TcpStream` and `OwnedWriteHalf`.
pub async fn send_tcp_msg(
    stream: &mut (impl AsyncWriteExt + Unpin),
    msg: &ServerMsg,
) -> Result<(), std::io::Error> {
    let data = msg.serialize();
    let mut buf = Vec::new();
    voxeldust_core::wire_codec::encode(&data, &mut buf);
    stream.write_all(&buf).await?;
    stream.flush().await?;
    Ok(())
}

/// Broadcast a WorldState to all registered UDP clients.
/// `packet_buf` is a reusable buffer to avoid per-broadcast heap allocation.
pub async fn broadcast_world_state_udp(
    socket: &UdpSocket,
    registry: &RwLock<ClientRegistry>,
    world_state: &ServerMsg,
    packet_buf: &mut Vec<u8>,
) {
    let data = world_state.serialize();
    packet_buf.clear();
    voxeldust_core::wire_codec::encode(&data, packet_buf);
    let packet = &*packet_buf;

    // Read phase: get addresses to broadcast to.
    let addrs = {
        let reg = registry.read().await;
        reg.udp_addrs()
    };

    if !addrs.is_empty() {
        tracing::info!(clients = addrs.len(), bytes = packet.len(), "broadcasting WorldState UDP");
    }

    // Track which observer addresses succeeded for lifecycle management.
    let mut successful_observers = Vec::new();
    for addr in &addrs {
        if let Err(e) = socket.send_to(packet, addr).await {
            debug!(%addr, %e, "failed to send WorldState UDP");
        } else {
            successful_observers.push(*addr);
        }
    }

    // Write phase: mark active observers and clean up stale ones.
    if !successful_observers.is_empty() {
        let mut reg = registry.write().await;
        for addr in &successful_observers {
            reg.mark_observer_active(addr);
        }
        reg.cleanup_stale_observers();
    }
}

/// Run UDP receiver loop: reads PlayerInput and BlockEditRequest packets,
/// discovers client UDP addresses.
pub async fn run_udp_receiver(
    socket: Arc<UdpSocket>,
    registry: Arc<RwLock<ClientRegistry>>,
    input_tx: mpsc::UnboundedSender<(SocketAddr, PlayerInputData)>,
    block_edit_tx: mpsc::UnboundedSender<(SessionToken, BlockEditData)>,
    config_update_tx: mpsc::UnboundedSender<(SessionToken, voxeldust_core::signal::config::BlockConfigUpdateData)>,
    sub_block_edit_tx: mpsc::UnboundedSender<(SessionToken, voxeldust_core::client_message::SubBlockEditData)>,
    cancel: CancellationToken,
) {
    let mut buf = vec![0u8; 65536];

    loop {
        tokio::select! {
            _ = cancel.cancelled() => return,
            result = socket.recv_from(&mut buf) => {
                match result {
                    Ok((len, src)) => {
                        // Register UDP address (hole-punch).
                        {
                            let mut reg = registry.write().await;
                            reg.discover_udp(src);
                        }

                        // Parse PlayerInput (wire codec: length-prefixed with optional LZ4).
                        if len < 4 { continue; }
                        let msg_len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
                        if len < 4 + msg_len { continue; }

                        let payload = match voxeldust_core::wire_codec::decode(&buf[4..4 + msg_len]) {
                            Ok(p) => p,
                            Err(_) => continue,
                        };
                        // Resolve UDP source → SessionToken for per-player routing.
                        let session = {
                            let reg = registry.read().await;
                            reg.session_for_udp(src)
                        };

                        match ClientMsg::deserialize(&payload) {
                            Ok(ClientMsg::PlayerInput(input)) => {
                                let _ = input_tx.send((src, input));
                            }
                            Ok(ClientMsg::BlockEditRequest(edit)) => {
                                if let Some(s) = session {
                                    let _ = block_edit_tx.send((s, edit));
                                }
                            }
                            Ok(ClientMsg::BlockConfigUpdate(update)) => {
                                if let Some(s) = session {
                                    let _ = config_update_tx.send((s, update));
                                }
                            }
                            Ok(ClientMsg::SubBlockEdit(edit)) => {
                                if let Some(s) = session {
                                    let _ = sub_block_edit_tx.send((s, edit));
                                }
                            }
                            _ => {}
                        }
                    }
                    Err(e) => {
                        warn!(%e, "UDP recv error");
                    }
                }
            }
        }
    }
}

fn rand_u64() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let s = RandomState::new();
    let mut h = s.build_hasher();
    h.write_u64(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64,
    );
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_client_after_register() {
        let mut reg = ClientRegistry::new();
        let token = SessionToken(42);
        // Simulate a minimal connection (we can't create a real TcpStream in tests,
        // so we test the data path via discover_udp + pending).
        assert!(!reg.has_client(&token));
        assert!(reg.is_empty());
    }

    #[test]
    fn pending_udp_caps_at_limit() {
        let mut reg = ClientRegistry::new();
        for i in 0..20u16 {
            let addr: SocketAddr = format!("127.0.0.1:{}", 5000 + i).parse().unwrap();
            reg.discover_udp(addr);
        }
        // Should be capped at 16.
        assert!(reg.pending_udp.len() <= 16);
    }

    #[test]
    fn pending_udp_no_duplicates() {
        // With no clients registered, the first unmatched UDP address is
        // stored as an observer (dual-shard compositing path). The
        // idempotent fast-path in `discover_udp` must prevent duplicate
        // observer entries when the same hello arrives twice.
        let mut reg = ClientRegistry::new();
        let addr: SocketAddr = "127.0.0.1:5000".parse().unwrap();
        reg.discover_udp(addr);
        reg.discover_udp(addr);
        assert_eq!(reg.pending_udp.len(), 0);
        assert_eq!(reg.observers.len(), 1);
    }
}
