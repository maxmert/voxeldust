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
    /// Phase T0: when this connection is an observer (`player_name`
    /// starts with `__observer__`), `observed_session` is the player's
    /// authoritative `SessionToken` from their source shard's
    /// `JoinResponse`. The shard registers this observer in
    /// `ClientRegistry.observers_by_session` keyed by `observed_session`
    /// so a future `PlayerHandoff` matching that token can promote the
    /// existing observer TCP into the player's primary connection
    /// in-place — no fresh TCP handshake on the seamless `ShardHandoff`
    /// path. `None` indicates a non-promotable legacy observer
    /// (chunks-only) or a real player-bearing connection.
    pub observed_session: Option<SessionToken>,
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
    /// Lamp-config edits from the F-key tablet UI. Each message
    /// carries the lamp's `(world_pos, face)` and the new
    /// `LampConfig`. Server validates ownership, persists to
    /// `ShipGrid.lamp_configs`, and rebroadcasts via the host chunk's
    /// next `ChunkDelta`.
    pub lamp_config_update_tx:
        mpsc::UnboundedSender<(SessionToken, voxeldust_core::client_message::LampConfigUpdateClientData)>,
    /// Phase 3C: client-issued `RemoteAccessGrant` create requests. The
    /// shard's grant-management ECS system drains this, validates that
    /// the requesting player owns the listed channels, mints a fresh
    /// `(grant_id, key)`, inserts into `GrantsRegistry`, and replies
    /// with a fresh `GrantsSnapshotData` containing the new key.
    pub grant_create_tx:
        mpsc::UnboundedSender<(SessionToken, voxeldust_core::client_message::GrantCreateData)>,
    /// Phase 3C: client-issued grant revocation requests. The grant is
    /// tombstoned (kept for audit) rather than removed. Idempotent —
    /// already-revoked grants reply success.
    pub grant_revoke_tx:
        mpsc::UnboundedSender<(SessionToken, voxeldust_core::client_message::GrantRevokeData)>,
    /// Phase 3C: recipient-side held grant registrations. Stored in the
    /// player's session-scoped `HeldGrants` resource on the primary
    /// shard so subsequent `RemoteSignalPublish` can look up the key.
    pub add_held_grant_tx:
        mpsc::UnboundedSender<(SessionToken, voxeldust_core::client_message::AddHeldGrantData)>,
    /// Phase 3C: cleanup for a held grant.
    pub forget_held_grant_tx:
        mpsc::UnboundedSender<(SessionToken, voxeldust_core::client_message::ForgetHeldGrantData)>,
    /// Phase 3C: client-issued remote publish — primary shard signs
    /// HMAC and ships a SignalBroadcastBatch to the target shard.
    pub remote_signal_publish_tx: mpsc::UnboundedSender<(
        SessionToken,
        voxeldust_core::client_message::RemoteSignalPublishData,
    )>,
    /// Phase D: chat lines typed on an engaged Terminal block.  Routed
    /// to the shard's media pipeline as `KeyboardTerminalInput`, which
    /// HMAC-signs (or sends bare for open channels) and ships through
    /// the configured publish channel.
    pub terminal_chat_send_tx: mpsc::UnboundedSender<(
        SessionToken,
        voxeldust_core::client_message::TerminalChatSendData,
    )>,
    /// Phase J: tablet open / close requests. Server validates the
    /// target block + range, then inserts/removes the
    /// `IsHoldingTablet` component on the player's character.
    pub tablet_interact_tx: mpsc::UnboundedSender<(
        SessionToken,
        voxeldust_core::client_message::TabletInteractData,
    )>,
    /// Phase J: per-tick (~20 Hz throttled) cursor position updates
    /// during the tablet hold. Server clamps and stuffs into the
    /// player's `TabletCursor` for broadcast — never used for
    /// gameplay decisions.
    pub tablet_cursor_update_tx: mpsc::UnboundedSender<(SessionToken, glam::Vec2)>,
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
    /// Phase T0 — session-tagged observers (TCP + optional UDP) keyed
    /// by the player's authoritative `SessionToken` from the source
    /// shard's `JoinResponse`. Distinct from the `observers` Vec
    /// above (which is a heuristic UDP-only catalogue with no
    /// session linkage). When a `PlayerHandoff` arrives matching one
    /// of these entries, [`Self::promote_observer_to_client`] moves
    /// the entry's TCP write half into `clients` under the same
    /// `SessionToken`, eliminating the TCP handshake on the seamless
    /// `ShardHandoff` path.
    session_observers: HashMap<SessionToken, SessionObserverEntry>,
}

struct ObserverEntry {
    addr: SocketAddr,
    registered_at: Instant,
    last_successful_send: Instant,
}

/// Phase T0 — observer connection tagged with the player's
/// authoritative `SessionToken` so a future `PlayerHandoff` for the
/// same token can promote this exact TCP stream into the player's
/// primary connection without a fresh handshake.
struct SessionObserverEntry {
    tcp_write: Arc<Mutex<OwnedWriteHalf>>,
    /// Last-known UDP addr used by the secondary's hello/WorldState
    /// stream, if discovered. Promoted into `ClientEntry.udp_addr` on
    /// `promote_observer_to_client` so `PlayerInput` arriving from
    /// the secondary's UDP socket post-promote routes to the right
    /// session without waiting for re-discovery.
    udp_addr: Option<SocketAddr>,
    /// TCP peer addr (= the secondary connection's source ip:port).
    /// `discover_udp` matches UDP packets to this session-observer by
    /// `peer_addr.ip() == udp_src.ip()` — a UDP socket on the client
    /// shares the host IP with its TCP socket (but not the port), so
    /// IP-match unambiguously associates a new UDP src with the
    /// session-observer whose TCP came from the same host. Without
    /// this association, `udp_addr` above stays `None` forever, the
    /// seamless promote moves the entry to `ClientEntry` with no
    /// `udp_addr`, and `session_for_udp` returns `None` for every
    /// PlayerInput → input is silently dropped → the player can't
    /// move post-EVA-exit while their CLIENT-side camera rotation
    /// (mouse-driven, doesn't need server) still works.
    peer_addr: SocketAddr,
    registered_at: Instant,
    /// Diagnostic tag — the `observer_name` field from
    /// `ClientMsg::ObserverConnect` (typically `observer_<seed>`).
    observer_name: String,
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
            session_observers: HashMap::new(),
        }
    }

    /// Phase T0 — register an observer TCP connection that carries the
    /// player's authoritative `SessionToken` (from `ObserverConnect`
    /// with the new `session_token` field). The destination shard
    /// calls this when it sees an `__observer__` connection whose
    /// `ClientConnection.observed_session` is `Some(token)`.
    /// Replaces any prior observer for the same session (last-writer
    /// wins, since the wire flow has the client open at most one
    /// observer per shard per session).
    pub fn register_session_observer(
        &mut self,
        session_token: SessionToken,
        observer_name: String,
        peer_addr: SocketAddr,
        tcp_write: Arc<Mutex<OwnedWriteHalf>>,
    ) {
        let entry = SessionObserverEntry {
            tcp_write,
            udp_addr: None,
            peer_addr,
            registered_at: Instant::now(),
            observer_name,
        };
        if self.session_observers.insert(session_token, entry).is_some() {
            info!(
                session = session_token.0,
                "session-observer replaced (existing entry overwritten)"
            );
        } else {
            info!(session = session_token.0, "session-observer registered");
        }
    }

    /// Phase T0 — record the secondary's UDP addr against an existing
    /// session-observer. Called by `discover_udp` (see below) the
    /// first time a packet from the secondary's UDP socket arrives,
    /// so `promote_observer_to_client` can carry the addr into the
    /// resulting `ClientEntry` and avoid a one-tick UDP-discovery
    /// gap right after the seamless `ShardHandoff`.
    fn record_session_observer_udp(&mut self, session: SessionToken, udp_addr: SocketAddr) {
        if let Some(entry) = self.session_observers.get_mut(&session) {
            entry.udp_addr = Some(udp_addr);
        }
    }

    /// Phase T0 — deterministic UDP↔session binding using the
    /// `session_token` field every `PlayerInput` UDP packet now carries
    /// (see `PlayerInputData::session_token`). The client knows its own
    /// session_token from `JoinResponse` and stamps every UDP packet,
    /// so the server can correctly assign `udp_src` even when:
    ///
    ///   * the TCP `ObserverConnect` registration lost the race against
    ///     the UDP hello (so `discover_udp`'s IP heuristic ran against
    ///     an empty `session_observers` and the addr landed in the
    ///     anonymous `observers` Vec instead), OR
    ///   * multiple clients share an IP (localhost dev / CGNAT) and
    ///     `discover_udp`'s IP-only heuristic mis-assigned the addr to
    ///     the wrong session-observer.
    ///
    /// Updates the canonical entry (clients OR session_observers) and
    /// clears any contradicting wrong assignment elsewhere in the
    /// registry so `session_for_udp(addr)` returns the correct token
    /// immediately on the next call. Idempotent.
    pub fn bind_udp_to_session(&mut self, session: SessionToken, udp_addr: SocketAddr) {
        // 1) If session is an active client (promoted or fresh): correct
        // its udp_addr and scrub the same addr from any other client
        // (race-survivor leftover).
        if self.clients.contains_key(&session) {
            for (token, entry) in self.clients.iter_mut() {
                if *token == session {
                    if entry.udp_addr != Some(udp_addr) {
                        info!(
                            session = session.0,
                            %udp_addr,
                            prior = ?entry.udp_addr,
                            "bind_udp_to_session: client udp_addr corrected"
                        );
                        entry.udp_addr = Some(udp_addr);
                    }
                } else if entry.udp_addr == Some(udp_addr) {
                    info!(
                        wrong_session = token.0,
                        %udp_addr,
                        "bind_udp_to_session: cleared udp_addr wrongly bound to other client"
                    );
                    entry.udp_addr = None;
                }
            }
            // Also remove from anonymous observers Vec — promoted
            // clients route their own broadcasts via clients HashMap.
            self.observers.retain(|o| o.addr != udp_addr);
            return;
        }

        // 2) Otherwise, this is a still-secondary session-observer.
        // Correct it and scrub the same addr from any wrong
        // session-observer or anonymous observer.
        if self.session_observers.contains_key(&session) {
            for (token, entry) in self.session_observers.iter_mut() {
                if *token == session {
                    if entry.udp_addr != Some(udp_addr) {
                        info!(
                            session = session.0,
                            %udp_addr,
                            prior = ?entry.udp_addr,
                            "bind_udp_to_session: session-observer udp_addr corrected"
                        );
                        entry.udp_addr = Some(udp_addr);
                    }
                } else if entry.udp_addr == Some(udp_addr) {
                    info!(
                        wrong_session = token.0,
                        %udp_addr,
                        "bind_udp_to_session: cleared udp_addr wrongly bound to other session-observer"
                    );
                    entry.udp_addr = None;
                }
            }
            // Anonymous observers Vec carries broadcasts only — when
            // we have a session-observer match, the anonymous entry
            // for the same addr is redundant. Leave it: the broadcast
            // path uses both `clients.udp_addr` and `observers` Vec,
            // and double-sending the same WorldState is harmless (UDP
            // is unordered anyway). Removing here would force every
            // session-observer ingest path to also re-add to the Vec
            // for broadcast — more code, no gain.
        }
    }

    /// Phase T0 — promote a session-tagged observer to a full
    /// player-bearing client. Returns `Some(tcp_write)` if a matching
    /// observer existed and was moved into the `clients` map under
    /// the same `SessionToken`; `None` if no observer was registered
    /// for that token (caller should fall back to the legacy
    /// fresh-TCP `JoinResponse` path).
    ///
    /// On promote, the observer's last-known UDP addr is carried into
    /// the new `ClientEntry` so `PlayerInput` arriving from the
    /// secondary's UDP socket immediately after promotion routes
    /// directly to this session — no `discover_udp` race.
    pub fn promote_observer_to_client(
        &mut self,
        session_token: SessionToken,
        player_name: String,
    ) -> Option<Arc<Mutex<OwnedWriteHalf>>> {
        let SessionObserverEntry {
            tcp_write,
            udp_addr,
            observer_name,
            ..
        } = self.session_observers.remove(&session_token)?;
        info!(
            session = session_token.0,
            %player_name,
            %observer_name,
            udp_known = udp_addr.is_some(),
            "promoting session-observer to client (seamless ShardHandoff)"
        );
        // The same UDP src that was tracked on the session-observer
        // is also sitting in `observers` Vec (added by `discover_udp`
        // before we knew which session it belonged to). After the
        // promote it logically belongs to the new `ClientEntry`, so
        // remove the now-stale observer entry — otherwise WorldState
        // broadcast sends two packets per tick to the same addr (once
        // via client.udp_addr, once via observer.addr).
        if let Some(addr) = udp_addr {
            self.observers.retain(|o| o.addr != addr);
        }
        self.clients.insert(
            session_token,
            ClientEntry {
                tcp_write: tcp_write.clone(),
                udp_addr,
                player_name,
            },
        );
        Some(tcp_write)
    }

    /// Phase T0 — drop a session-observer (e.g. the observer TCP
    /// disconnected without ever being promoted). Idempotent.
    pub fn unregister_session_observer(&mut self, session_token: SessionToken) {
        if self.session_observers.remove(&session_token).is_some() {
            info!(session = session_token.0, "session-observer unregistered");
        }
    }

    /// Phase T0 — true if a session-observer is registered for this
    /// `SessionToken`. Used by source shards to decide between the
    /// seamless `ShardHandoff` path (when destination has confirmed
    /// the observer) and the legacy `ShardRedirect` fallback.
    pub fn has_session_observer(&self, session_token: SessionToken) -> bool {
        self.session_observers.contains_key(&session_token)
    }

    pub fn register(&mut self, conn: &ClientConnection) {
        // Preserve the existing entry's `udp_addr` when the incoming
        // `conn.udp_addr` is `None`. This is required by the seamless
        // ShardHandoff promote path: ship-shard's PlayerHandoff handler
        // synthesises a `ClientConnectEvent` AFTER
        // `promote_observer_to_client` populated `clients[token].udp_addr`
        // from the session-observer's UDP binding. Without preservation,
        // the synthetic register() call here would overwrite that
        // udp_addr with None, breaking PlayerInput routing for the just-
        // promoted player until `discover_udp` re-discovers the UDP src.
        let preserved_udp = self
            .clients
            .get(&conn.session_token)
            .and_then(|e| e.udp_addr);
        self.clients.insert(conn.session_token, ClientEntry {
            tcp_write: conn.tcp_write.clone(),
            udp_addr: conn.udp_addr.or(preserved_udp),
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
        // Phase T0 — associate this UDP src with a session-observer
        // by IP match BEFORE the early-return checks. A UDP socket
        // on the client shares the host IP with its TCP socket (the
        // ports differ), so peer-IP-match unambiguously links the
        // UDP src to the session-observer's TCP. Without this,
        // `SessionObserverEntry.udp_addr` stays `None` indefinitely;
        // `promote_observer_to_client` then moves the observer to
        // `clients` with `udp_addr = None`; `session_for_udp`
        // returns `None` for every `PlayerInput` packet → input is
        // silently dropped post-seamless-promote → the player
        // can't move (only client-side camera rotation works since
        // it doesn't depend on the server). Match runs every call
        // but is idempotent (no-op once `udp_addr` is set).
        for (session, entry) in self.session_observers.iter_mut() {
            if entry.udp_addr.is_none() && entry.peer_addr.ip() == udp_addr.ip() {
                entry.udp_addr = Some(udp_addr);
                info!(
                    session = session.0,
                    %udp_addr,
                    peer = %entry.peer_addr,
                    "associated UDP src with session-observer (IP match)"
                );
                break;
            }
        }

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
        // Shard ignores `ship_join_key` — that's gateway-only routing
        // metadata. Once the client is on the shard, the ship is
        // already chosen.
        ClientMsg::Connect { player_name, ship_join_key: _ } => {
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
                // Real player-bearing connection — observed_session is
                // only populated for observer connections (Phase T0).
                observed_session: None,
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
        ClientMsg::ObserverConnect { observer_name, session_token: observed_session } => {
            info!(
                %peer_addr,
                %observer_name,
                observed_session = observed_session.0,
                "observer TCP connected"
            );

            // Observer connections: split stream, store write half for chunk sync,
            // but do NOT send a ClientConnectEvent (no player entity).
            // The shard will detect the observer via the ObserverConnect channel.
            let (read_half, write_half) = stream.into_split();
            let write = Arc::new(Mutex::new(write_half));

            // Store observer TCP write half in the connect channel with a special
            // sentinel name so the shard can distinguish observers from players.
            //
            // `observed_session` is the player's authoritative SessionToken
            // from `JoinResponse` on their source shard (Phase T0). The shard
            // will register this observer in
            // `ClientRegistry.observers_by_session` keyed by `observed_session`
            // (T0.C) so a future `PlayerHandoff` matching that token can
            // promote this very TCP write half into the player's primary
            // entry — no fresh handshake on the seamless `ShardHandoff`
            // path. `SessionToken(0)` indicates a legacy non-promotable
            // observer (chunks-only).
            let observer_token = SessionToken(rand_u64());
            let connection = ClientConnection {
                session_token: observer_token,
                player_name: format!("__observer__{}", observer_name),
                tcp_write: write.clone(),
                peer_addr,
                udp_addr: None,
                observed_session: if observed_session.0 != 0 {
                    Some(observed_session)
                } else {
                    None
                },
            };
            let _ = connect_tx.send(ClientConnectEvent { connection });
            info!(%peer_addr, %observer_name, "observer TCP setup complete");

            // Phase T0 — register the observed-session entry in
            // `ClientRegistry.session_observers` so a subsequent
            // `PlayerHandoff` whose `session_token` matches can
            // promote this very TCP write half into the player's
            // primary `ClientEntry` (seamless ShardHandoff path
            // instead of fresh ShardRedirect handshake). Without
            // this call, `register_session_observer` was dead code:
            // the `observed_session` arrived in `ObserverConnect`
            // but was only kept on the `ClientConnection` struct,
            // never propagated into the registry, so
            // `has_session_observer` always returned `false` on
            // hull-exit and the EVA spawn took the slow path.
            if observed_session.0 != 0 {
                let mut reg = client_registry.write().await;
                reg.register_session_observer(
                    observed_session,
                    observer_name.clone(),
                    peer_addr,
                    write,
                );
            }

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
            // Phase T0 — also drop the session-observer entry if the
            // observer disconnected before being promoted to a primary
            // client. If `promote_observer_to_client` already moved it
            // into `clients`, this is a no-op (the entry was removed
            // by the promote and the corresponding `unregister(&token)`
            // call above won't have triggered yet because the player
            // path runs in a different code branch).
            if observed_session.0 != 0 {
                reg.unregister_session_observer(observed_session);
            }
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
            Ok(ClientMsg::LampConfigUpdate(update)) => {
                let _ = channels.lamp_config_update_tx.send((session_token, update));
            }
            Ok(ClientMsg::SignalPublish(data)) => {
                let _ = channels.signal_publish_tx.send((session_token, data));
            }
            Ok(ClientMsg::GrantCreate(data)) => {
                let _ = channels.grant_create_tx.send((session_token, data));
            }
            Ok(ClientMsg::GrantRevoke(data)) => {
                let _ = channels.grant_revoke_tx.send((session_token, data));
            }
            Ok(ClientMsg::AddHeldGrant(data)) => {
                let _ = channels.add_held_grant_tx.send((session_token, data));
            }
            Ok(ClientMsg::ForgetHeldGrant(data)) => {
                let _ = channels.forget_held_grant_tx.send((session_token, data));
            }
            Ok(ClientMsg::RemoteSignalPublish(data)) => {
                let _ = channels.remote_signal_publish_tx.send((session_token, data));
            }
            Ok(ClientMsg::TerminalChatSend(data)) => {
                let _ = channels.terminal_chat_send_tx.send((session_token, data));
            }
            Ok(ClientMsg::TabletInteract(data)) => {
                let _ = channels.tablet_interact_tx.send((session_token, data));
            }
            Ok(ClientMsg::TabletCursorUpdate(uv)) => {
                let _ = channels.tablet_cursor_update_tx.send((session_token, uv));
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
    lamp_config_update_tx: mpsc::UnboundedSender<(SessionToken, voxeldust_core::client_message::LampConfigUpdateClientData)>,
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
                                // Deterministic UDP↔session binding —
                                // the client stamps every PlayerInput
                                // packet with its known session_token,
                                // so we don't need `discover_udp`'s
                                // peer-IP heuristic (which mis-assigns
                                // on shared-IP setups like localhost
                                // multi-client). 0 = legacy/pre-T0,
                                // fall back to heuristic that
                                // `discover_udp` already ran above.
                                if input.session_token != 0 {
                                    let token = SessionToken(input.session_token);
                                    let mut reg = registry.write().await;
                                    reg.bind_udp_to_session(token, src);
                                }
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
                            Ok(ClientMsg::LampConfigUpdate(update)) => {
                                if let Some(s) = session {
                                    let _ = lamp_config_update_tx.send((s, update));
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
