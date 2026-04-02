use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use voxeldust_core::client_message::{ClientMsg, PlayerInputData, ServerMsg};
use voxeldust_core::shard_types::SessionToken;

/// A persistent client connection (TCP stream kept alive).
pub struct ClientConnection {
    pub session_token: SessionToken,
    pub player_name: String,
    pub tcp_stream: Arc<Mutex<TcpStream>>,
    pub peer_addr: SocketAddr,
    /// Client's UDP address (learned from first UDP packet — hole-punch pattern).
    pub udp_addr: Option<SocketAddr>,
}

/// Event emitted when a client connects via TCP.
pub struct ClientConnectEvent {
    pub connection: ClientConnection,
}

/// Tracks all connected clients. Thread-safe for use across tick systems.
pub struct ClientRegistry {
    clients: HashMap<SessionToken, ClientEntry>,
    /// UDP addresses seen before any client registered (for late-join matching).
    pending_udp: Vec<SocketAddr>,
}

struct ClientEntry {
    tcp_stream: Arc<Mutex<TcpStream>>,
    udp_addr: Option<SocketAddr>,
    player_name: String,
}


impl ClientRegistry {
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
            pending_udp: Vec::new(),
        }
    }

    pub fn register(&mut self, conn: &ClientConnection) {
        self.clients.insert(conn.session_token, ClientEntry {
            tcp_stream: conn.tcp_stream.clone(),
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
        // If a stale entry already has this address (reconnect from same endpoint),
        // clear it so the new client can claim it.
        for entry in self.clients.values_mut() {
            if entry.udp_addr == Some(udp_addr) {
                entry.udp_addr = None;
            }
        }

        // Assign to the first client without a UDP address.
        for entry in self.clients.values_mut() {
            if entry.udp_addr.is_none() {
                entry.udp_addr = Some(udp_addr);
                info!(player = %entry.player_name, %udp_addr, "discovered client UDP address");
                return;
            }
        }

        // No client to match — store as pending (cap at 16 to prevent unbounded growth).
        if self.pending_udp.len() < 16 && !self.pending_udp.contains(&udp_addr) {
            debug!(%udp_addr, "storing pending UDP address (no client registered yet)");
            self.pending_udp.push(udp_addr);
        }
    }

    /// Get all UDP addresses for broadcasting.
    pub fn udp_addrs(&self) -> Vec<SocketAddr> {
        self.clients.values().filter_map(|e| e.udp_addr).collect()
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

    /// Send a TCP message to a specific client.
    pub async fn send_tcp(&self, token: SessionToken, msg: &ServerMsg) -> Result<(), std::io::Error> {
        if let Some(entry) = self.clients.get(&token) {
            let mut stream = entry.tcp_stream.lock().await;
            send_tcp_msg(&mut stream, msg).await?;
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
                        tokio::spawn(async move {
                            handle_client_connection(stream, peer_addr, tx).await;
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

async fn handle_client_connection(
    mut stream: TcpStream,
    peer_addr: SocketAddr,
    connect_tx: mpsc::UnboundedSender<ClientConnectEvent>,
) {
    // Disable Nagle's algorithm for low-latency TCP messaging.
    let _ = stream.set_nodelay(true);

    // Read 4-byte length prefix.
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
    match ClientMsg::deserialize(&decoded) {
        Ok(ClientMsg::Connect { player_name }) => {
            let token = SessionToken(rand_u64());
            info!(%peer_addr, %player_name, session_token = token.0, "client connected");

            // Keep the TCP stream alive — wrap in Arc<Mutex<>> for shared access.
            let connection = ClientConnection {
                session_token: token,
                player_name,
                tcp_stream: Arc::new(Mutex::new(stream)),
                peer_addr,
                udp_addr: None,
            };

            let _ = connect_tx.send(ClientConnectEvent { connection });
        }
        Ok(_) => {
            warn!(%peer_addr, "expected Connect message, got something else");
        }
        Err(e) => {
            warn!(%peer_addr, %e, "failed to deserialize client message");
        }
    }
}

/// Send a length-prefixed ServerMsg over a TCP stream (with LZ4 compression).
pub async fn send_tcp_msg(
    stream: &mut TcpStream,
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

    let reg = registry.read().await;
    let addrs = reg.udp_addrs();
    if !addrs.is_empty() {
        tracing::info!(clients = addrs.len(), bytes = packet.len(), "broadcasting WorldState UDP");
    }
    for addr in addrs {
        if let Err(e) = socket.send_to(&packet, addr).await {
            debug!(%addr, %e, "failed to send WorldState UDP");
        }
    }
}

/// Run UDP receiver loop: reads PlayerInput packets and discovers client UDP addresses.
pub async fn run_udp_receiver(
    socket: Arc<UdpSocket>,
    registry: Arc<RwLock<ClientRegistry>>,
    input_tx: mpsc::UnboundedSender<(SocketAddr, PlayerInputData)>,
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
                        match ClientMsg::deserialize(&payload) {
                            Ok(ClientMsg::PlayerInput(input)) => {
                                let _ = input_tx.send((src, input));
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
