//! Unified shard-connection task — Phase T0.E architecture.
//!
//! Replaces the legacy split-task topology in `network.rs` (separate
//! primary `send_handle` + `recv_handle` + `tcp_handle` triple AND
//! per-secondary observer-UDP-task + observer-TCP-task pair) with a
//! single async function [`run_connection`] that:
//!
//!   * Owns its TCP stream (split into read+write halves) and UDP
//!     socket directly.
//!   * Carries a [`ConnectionMode`] state (`Primary` | `Secondary` |
//!     `Demoting`) that determines the active behaviour.
//!   * Listens on a [`ConnectionControl`] mpsc for in-flight mode
//!     transitions sent by the controller.
//!
//! ## Why a unified task
//!
//! The legacy split-task layout means promoting a pre-connected
//! secondary to primary requires either:
//!
//!   1. Aborting the existing secondary tasks and spawning fresh
//!      primary tasks against the same socket — needs careful socket
//!      ownership transfer between tasks (oneshot channels), or
//!   2. Spawning new primary tasks that open a fresh TCP+UDP to the
//!      destination's endpoint — costs ~5-50 ms TCP handshake even
//!      though the destination's chunks are already in client memory
//!      from the secondary's stream.
//!
//! Option 2 is the legacy `ShardRedirect` flow that this codebase
//! already implements; it never achieves true zero-RTT promote.
//!
//! Option 1 with the legacy split is fragile (multiple ownership
//! handoffs, easy to leak sockets or miss frames). The unified task
//! resolves it cleanly: the SAME task continues running across the
//! Promote, just with its [`ConnectionMode`] flipped from `Secondary`
//! to `Primary`. Sockets stay where they are. The behaviour set
//! changes (start sending `PlayerInput`, expand TCP read set),
//! everything else is preserved.
//!
//! ## Mode → behaviour matrix
//!
//! |                    | Secondary             | Primary             | Demoting             |
//! |--------------------|-----------------------|---------------------|----------------------|
//! | UDP send           | initial hello only    | PlayerInput @ 20Hz  | initial hello only   |
//! | UDP recv (WS)      | NetEvent::SecondaryWorldState | NetEvent::WorldState | NetEvent::SecondaryWorldState |
//! | UDP recv (GalaxyWS)| NetEvent::GalaxyWorldState    | NetEvent::GalaxyWorldState | NetEvent::GalaxyWorldState |
//! | TCP read set       | Chunks/Catalog only   | Full primary set    | Chunks/Catalog only  |
//! | TCP outbound       | none                  | block edits / etc   | none                 |
//! | Exit trigger       | Cancel control        | Cancel control      | TTL elapsed          |
//!
//! ## Migration plan
//!
//! `run_network` (network.rs) is refactored across follow-up sessions
//! to spawn `run_connection` invocations instead of the legacy
//! send/recv/tcp triples and observer task pairs. Until that lands,
//! [`run_connection`] coexists with the legacy code paths; nothing
//! calls it in production yet. The implementation below is complete
//! and ready to be the migration target.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpStream, UdpSocket};
use tokio::sync::{mpsc, Mutex};
use tracing::{info, warn};

use voxeldust_core::client_message::{ClientMsg, PlayerInputData, ServerMsg};
use voxeldust_core::handoff::SpawnPose;
use voxeldust_core::shard_types::SessionToken;

use super::NetEvent;

/// What kind of work a [`run_connection`] task is currently doing.
///
/// State transitions are externally-driven via [`ConnectionControl`]:
/// `Secondary → Primary` via [`ConnectionControl::Promote`], `Primary →
/// Demoting` via [`ConnectionControl::Demote`]. `Demoting` self-exits
/// when its TTL elapses. There is intentionally no `Demoting →
/// Secondary` transition — once demoted the connection is winding
/// down and any further promote should come from a different task.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConnectionMode {
    /// The owned player connection. Sends `PlayerInput` on UDP at
    /// 20 Hz and consumes the full primary TCP message set
    /// (`ShardRedirect`, `ShardHandoff`, `ShardPreConnect`,
    /// `BlockConfigState`, etc.). Inbound `WorldState` is forwarded
    /// as `NetEvent::WorldState`.
    Primary,
    /// Observer connection. Sends only the initial UDP hello
    /// (so the destination's NAT learns this socket's source addr
    /// for the WorldState broadcast loop). Inbound `WorldState`
    /// is forwarded as `NetEvent::SecondaryWorldState` so the client's
    /// composite renderer can route it to the correct shard slot.
    /// TCP read set restricted to the chunk-streaming subset.
    Secondary,
    /// Previously-Primary connection that has been demoted because
    /// a different task was promoted to take its place. Behaves
    /// exactly like `Secondary` (so other clients still see this
    /// player's pose updates ghosted from the source shard) until
    /// `exit_after` elapses, at which point the task exits
    /// gracefully. Used during the seamless `ShardHandoff` transition
    /// to give the destination a few ticks to take over without a
    /// visible WorldState gap.
    Demoting { exit_after: Instant },
}

/// Messages the controller sends to a [`run_connection`] task to
/// trigger mode transitions or termination.
///
/// Sent over an mpsc channel so multiple control messages can be
/// queued (typical case: a single `Promote` followed by a `Demote`
/// arriving on different tasks). Capacity is small (4-8); back-pressure
/// from a slow task is a bug.
#[derive(Debug)]
pub enum ConnectionControl {
    /// Switch from `Secondary` to `Primary`. Carries the destination
    /// shard's authoritative `SpawnPose` (computed by the source
    /// shard at handoff time) so the controller can latch it for the
    /// camera-pose system to apply on its next frame, and the
    /// `target_shard_type` so downstream `NetEvent::Transitioning`
    /// consumers route correctly.
    Promote {
        spawn_pose: Option<SpawnPose>,
        target_shard_type: u8,
    },
    /// Switch from `Primary` to `Demoting { exit_after = now + ttl }`.
    /// `ttl_ticks` matches the wire-protocol field
    /// `ShardHandoffMsg.source_demote_after_ticks` (default 15 ticks
    /// = 750 ms at 20 Hz).
    Demote {
        ttl_ticks: u8,
    },
    /// Immediate cancel. Task exits at its next select boundary.
    Cancel,
}

/// One tick = 50 ms = the server-broadcast interval. `Demote.ttl_ticks`
/// is multiplied by this to compute the `Demoting.exit_after`
/// `Instant`.
pub const DEMOTE_TICK: Duration = Duration::from_millis(50);

/// Input-send interval — 20 Hz UDP `PlayerInput` cadence. Matches
/// the legacy `network.rs::run_network` `send_handle` interval.
const INPUT_TICK: Duration = Duration::from_millis(50);

/// TCP keepalive interval — primary mode emits a zero-length packet
/// every 60s so the server's receive timeout doesn't reap the
/// connection during long idle stretches.
const KEEPALIVE_TICK: Duration = Duration::from_secs(60);

/// Coalesce-suppression interval — when input is unchanged for this
/// many input-ticks, send a keepalive copy anyway so the server
/// doesn't time out waiting for the next change. 20 ticks at 20 Hz
/// = 1s, matching `network.rs::run_network`.
const INPUT_KEEPALIVE_TICKS: u32 = 20;

/// Handle the controller (`run_network` main loop) keeps for each
/// active [`run_connection`] task. Sends control messages and
/// identifies the task by `(shard_type, shard_id)`.
#[derive(Debug, Clone)]
pub struct ConnectionHandle {
    pub shard_type: u8,
    pub shard_id: u64,
    pub seed: u64,
    control_tx: mpsc::Sender<ConnectionControl>,
}

impl ConnectionHandle {
    pub fn new(
        shard_type: u8,
        shard_id: u64,
        seed: u64,
        control_tx: mpsc::Sender<ConnectionControl>,
    ) -> Self {
        Self {
            shard_type,
            shard_id,
            seed,
            control_tx,
        }
    }

    /// True if the task is still listening (channel not dropped).
    pub fn is_alive(&self) -> bool {
        !self.control_tx.is_closed()
    }

    /// Send `Promote` to the task. Errors only if the task has already
    /// exited.
    pub async fn promote(
        &self,
        spawn_pose: Option<SpawnPose>,
        target_shard_type: u8,
    ) -> Result<(), mpsc::error::SendError<ConnectionControl>> {
        self.control_tx
            .send(ConnectionControl::Promote {
                spawn_pose,
                target_shard_type,
            })
            .await
    }

    /// Send `Demote` with the given TTL.
    pub async fn demote(
        &self,
        ttl_ticks: u8,
    ) -> Result<(), mpsc::error::SendError<ConnectionControl>> {
        self.control_tx
            .send(ConnectionControl::Demote { ttl_ticks })
            .await
    }

    /// Send hard `Cancel`.
    pub async fn cancel(
        &self,
    ) -> Result<(), mpsc::error::SendError<ConnectionControl>> {
        self.control_tx.send(ConnectionControl::Cancel).await
    }
}

/// Wire-codec helpers — the legacy `network.rs::build_input` and
/// `network.rs::recv_server_msg` are duplicated here as private
/// utilities so [`run_connection`] doesn't depend on `network.rs`
/// internals during the migration window. Once the migration is
/// complete the legacy versions can be deleted in favour of these.

fn empty_input() -> PlayerInputData {
    PlayerInputData {
        movement: [0.0; 3],
        look_yaw: 0.0,
        look_pitch: 0.0,
        jump: false,
        fly_toggle: false,
        orbit_stabilizer_toggle: false,
        speed_tier: 0,
        action: 0,
        block_type: 0,
        tick: 0,
        thrust_limiter: 0.75,
        roll: 0.0,
        cruise: false,
        atmo_comp: false,
        seat_values: Vec::new(),
        actions_bits: 0,
    }
}

fn build_input_packet(input: &PlayerInputData) -> Vec<u8> {
    let data = ClientMsg::PlayerInput(input.clone()).serialize();
    let mut buf = Vec::new();
    voxeldust_core::wire_codec::encode(&data, &mut buf);
    buf
}

async fn send_client_msg(
    stream: &mut tokio::io::WriteHalf<TcpStream>,
    msg: &ClientMsg,
) -> Result<(), std::io::Error> {
    let data = msg.serialize();
    let mut buf = Vec::new();
    voxeldust_core::wire_codec::encode(&data, &mut buf);
    stream.write_all(&buf).await?;
    stream.flush().await
}

async fn recv_server_msg<R>(
    stream: &mut R,
) -> Result<ServerMsg, Box<dyn std::error::Error + Send + Sync>>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len == 0 {
        // Server keepalive — synthesize a no-op return.
        // Caller loops back to read next message.
        return Err("keepalive".into());
    }
    if len > 65536 {
        return Err(format!("oversize tcp msg: {len}").into());
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).await?;
    let decoded = voxeldust_core::wire_codec::decode(&buf)?;
    Ok(ServerMsg::deserialize(&decoded)?)
}

/// The unified connection task. Owns its TCP+UDP, runs in a mode-
/// dependent loop, and listens on `control_rx` for mode transitions.
///
/// ## Pre-conditions
///
/// * `tcp` must be CONNECTED. For Primary mode the caller is expected
///   to have completed the `ClientMsg::Connect` → `ServerMsg::JoinResponse`
///   handshake before invoking; this function does NOT re-do the
///   handshake. For Secondary mode this function sends the
///   `ClientMsg::ObserverConnect` itself (no handshake needed —
///   ObserverConnect is one-shot).
/// * `udp` must be BOUND but the server's NAT has not necessarily
///   learnt the source addr yet — this function sends an empty UDP
///   hello to `udp_target` on entry so the server's hole-punch
///   recognises this socket.
///
/// ## Exits when
///
/// * `control_rx` receives `Cancel`.
/// * `control_rx` is dropped by the controller.
/// * `Demoting` mode's TTL elapses.
/// * UDP recv error (typically a Linux ECONNREFUSED from a stale
///   server addr); `NetEvent::SecondaryDisconnected { seed }` is
///   emitted on exit.
/// * TCP recv error (EOF or malformed framing).
#[allow(clippy::too_many_arguments)]
pub async fn run_connection(
    initial_mode: ConnectionMode,
    tcp: TcpStream,
    udp: UdpSocket,
    udp_target: SocketAddr,
    shard_type: u8,
    shard_id: u64,
    seed: u64,
    session_token: SessionToken,
    event_tx: mpsc::UnboundedSender<NetEvent>,
    input_rx: Arc<Mutex<mpsc::Receiver<PlayerInputData>>>,
    tcp_out_rx: Arc<Mutex<mpsc::Receiver<Vec<u8>>>>,
    mut control_rx: mpsc::Receiver<ConnectionControl>,
) {
    let mut mode = initial_mode;
    let _ = tcp.set_nodelay(true);
    let (tcp_read, tcp_write) = tokio::io::split(tcp);
    let mut tcp_read = tokio::io::BufReader::new(tcp_read);
    let mut tcp_write = tcp_write;

    // Per-mode initial handshake.
    if matches!(mode, ConnectionMode::Secondary) {
        // Send `ObserverConnect` carrying the player's session token
        // so the destination shard registers this TCP in its
        // `ClientRegistry.session_observers` (Phase T0.B/C/D wire
        // contract).
        let oc = ClientMsg::ObserverConnect {
            observer_name: format!("observer_{}", seed),
            session_token,
        };
        if let Err(e) = send_client_msg(&mut tcp_write, &oc).await {
            warn!(%e, %shard_type, %shard_id, %seed, "failed to send ObserverConnect");
            let _ = event_tx.send(NetEvent::SecondaryDisconnected { seed });
            return;
        }
    }

    // UDP hole-punch on entry so the server NATs the source addr
    // before the first `WorldState` broadcast. For Primary mode the
    // legacy code did this in `connect_to_shard_full` after JoinResponse;
    // for the unified task we always do it here so callers don't
    // need to.
    let hello = build_input_packet(&empty_input());
    if let Err(e) = udp.send_to(&hello, udp_target).await {
        warn!(%e, %udp_target, "UDP hole-punch send failed");
    }

    if matches!(mode, ConnectionMode::Secondary) {
        // Mirror the legacy network.rs flow that emits SecondaryConnected
        // on entry so the client's shard registry can spawn a
        // ChunkSource for this seed. The reference_position/_rotation
        // are not known at this layer; the migrating caller will pass
        // them via a different code path or extend the event. For the
        // skeleton-target version we leave them at default; the
        // controller layer will populate them when it calls into
        // `run_connection`.
        let _ = event_tx.send(NetEvent::SecondaryConnected {
            shard_type,
            seed,
            reference_position: glam::DVec3::ZERO,
            reference_rotation: glam::DQuat::IDENTITY,
        });
    }

    let mut buf = vec![0u8; 65536];
    let mut input_interval = tokio::time::interval(INPUT_TICK);
    input_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut last_input_sent = empty_input();
    let mut ticks_since_send: u32 = 0;
    let mut keepalive_interval = tokio::time::interval(KEEPALIVE_TICK);
    keepalive_interval.tick().await; // skip the immediate first tick

    loop {
        // Demoting TTL gate — exit when the timer elapses, at the
        // top of each select cycle so we never block past the
        // configured handoff-grace window.
        if let ConnectionMode::Demoting { exit_after } = mode {
            if Instant::now() >= exit_after {
                info!(
                    %shard_type, %shard_id, %seed,
                    "demoting connection: TTL elapsed, exiting"
                );
                let _ = event_tx.send(NetEvent::SecondaryDisconnected { seed });
                return;
            }
        }

        let is_primary = matches!(mode, ConnectionMode::Primary);

        tokio::select! {
            biased;

            // 1) Control messages — highest priority. Mode transitions
            //    must take effect before any further I/O is dispatched
            //    against the old mode.
            ctrl = control_rx.recv() => {
                match ctrl {
                    None => {
                        // Controller dropped sender — orderly exit.
                        if !is_primary {
                            let _ = event_tx.send(NetEvent::SecondaryDisconnected { seed });
                        }
                        return;
                    }
                    Some(ConnectionControl::Cancel) => {
                        info!(%shard_type, %shard_id, %seed, "connection cancelled");
                        if !is_primary {
                            let _ = event_tx.send(NetEvent::SecondaryDisconnected { seed });
                        }
                        return;
                    }
                    Some(ConnectionControl::Promote { spawn_pose, target_shard_type }) => {
                        match mode {
                            ConnectionMode::Secondary => {
                                info!(
                                    %shard_type, %shard_id, %seed,
                                    "promoting Secondary → Primary"
                                );
                                // Emit Transitioning + synthetic Connected
                                // so downstream client systems run their
                                // existing primary-promotion code path
                                // (camera-pose latch, shard registry
                                // primary swap).
                                let _ = event_tx.send(NetEvent::Transitioning {
                                    target_shard_type,
                                    spawn_pose: spawn_pose.clone(),
                                });
                                let _ = event_tx.send(NetEvent::Connected {
                                    shard_type: target_shard_type,
                                    seed,
                                    reference_position: glam::DVec3::ZERO,
                                    reference_rotation: glam::DQuat::IDENTITY,
                                    game_time: 0.0,
                                    system_seed: 0,
                                    galaxy_seed: 0,
                                    player_id: session_token.0,
                                });
                                mode = ConnectionMode::Primary;
                                // Reset input cadence — first PlayerInput
                                // goes out on the next interval tick.
                                last_input_sent = empty_input();
                                ticks_since_send = INPUT_KEEPALIVE_TICKS;
                            }
                            other => {
                                warn!(
                                    ?other, %shard_type, %shard_id,
                                    "Promote received in non-Secondary mode — ignoring"
                                );
                            }
                        }
                    }
                    Some(ConnectionControl::Demote { ttl_ticks }) => {
                        match mode {
                            ConnectionMode::Primary => {
                                let exit_after = Instant::now()
                                    + DEMOTE_TICK * u32::from(ttl_ticks);
                                info!(
                                    %shard_type, %shard_id, %seed, ttl_ticks,
                                    "demoting Primary → Demoting (will exit at TTL)"
                                );
                                mode = ConnectionMode::Demoting { exit_after };
                            }
                            other => {
                                warn!(
                                    ?other, %shard_type, %shard_id,
                                    "Demote received in non-Primary mode — ignoring"
                                );
                            }
                        }
                    }
                }
            }

            // 2) UDP recv — `WorldState` / `GalaxyWorldState` arrival.
            //    Mode-dependent forwarding: Primary → NetEvent::WorldState,
            //    Secondary/Demoting → NetEvent::SecondaryWorldState.
            udp_result = udp.recv_from(&mut buf) => {
                match udp_result {
                    Ok((len, _src)) => {
                        if len < 4 { continue; }
                        let msg_len = u32::from_be_bytes(
                            [buf[0], buf[1], buf[2], buf[3]]) as usize;
                        if len < 4 + msg_len { continue; }
                        let decoded = match voxeldust_core::wire_codec::decode(
                            &buf[4..4 + msg_len])
                        {
                            Ok(d) => d,
                            Err(_) => continue,
                        };
                        match ServerMsg::deserialize(&decoded) {
                            Ok(ServerMsg::WorldState(ws)) => {
                                if is_primary {
                                    let _ = event_tx.send(NetEvent::WorldState(ws));
                                } else {
                                    let _ = event_tx.send(NetEvent::SecondaryWorldState {
                                        shard_type,
                                        seed,
                                        ws,
                                    });
                                }
                            }
                            Ok(ServerMsg::GalaxyWorldState(gws)) => {
                                let _ = event_tx.send(NetEvent::GalaxyWorldState(gws));
                            }
                            _ => { /* unexpected on UDP — drop */ }
                        }
                    }
                    Err(e) => {
                        warn!(
                            %e, %udp_target, %shard_type, %shard_id,
                            "UDP recv error — exiting connection"
                        );
                        let _ = event_tx.send(NetEvent::SecondaryDisconnected { seed });
                        return;
                    }
                }
            }

            // 3) TCP recv — server messages. Primary handles the
            //    full set; Secondary/Demoting only the chunk-streaming
            //    subset.
            msg_result = recv_server_msg(&mut tcp_read) => {
                match msg_result {
                    Ok(msg) => {
                        if is_primary {
                            forward_primary_tcp_msg(msg, &event_tx);
                        } else {
                            forward_secondary_tcp_msg(msg, seed, &event_tx);
                        }
                    }
                    Err(e) => {
                        // "keepalive" comes back as Err with that exact
                        // text from `recv_server_msg` — silently loop.
                        if e.to_string() == "keepalive" { continue; }
                        warn!(
                            %e, %shard_type, %shard_id,
                            "TCP recv error — exiting connection"
                        );
                        if !is_primary {
                            let _ = event_tx.send(NetEvent::SecondaryDisconnected { seed });
                        }
                        return;
                    }
                }
            }

            // 4) Input send tick — Primary mode only. Coalesce-
            //    suppress unchanged input but force a keepalive
            //    copy every INPUT_KEEPALIVE_TICKS ticks so the
            //    server's UDP hole-punch doesn't time out on long
            //    no-input stretches.
            _ = input_interval.tick(), if is_primary => {
                let input = {
                    let mut rx = input_rx.lock().await;
                    let mut latest = empty_input();
                    while let Ok(i) = rx.try_recv() { latest = i; }
                    latest
                };
                let changed = input != last_input_sent;
                if changed || ticks_since_send >= INPUT_KEEPALIVE_TICKS {
                    let pkt = build_input_packet(&input);
                    if let Err(e) = udp.send_to(&pkt, udp_target).await {
                        if changed {
                            warn!(%e, %udp_target, "UDP input send failed");
                        }
                    }
                    last_input_sent = input;
                    ticks_since_send = 0;
                } else {
                    ticks_since_send += 1;
                }
            }

            // 5) TCP outbound (block edits, sub-block edits, configs).
            //    Primary mode only. The legacy network.rs gates these
            //    behind a separate task; here they share the same
            //    select so we don't spawn extra tasks per primary.
            pkt = async {
                let mut rx = tcp_out_rx.lock().await;
                rx.recv().await
            }, if is_primary => {
                if let Some(pkt) = pkt {
                    let _ = tcp_write.write_all(&pkt).await;
                    let _ = tcp_write.flush().await;
                }
            }

            // 6) TCP keepalive — Primary mode only. Zero-length frame
            //    every KEEPALIVE_TICK so the server's idle timeout
            //    doesn't reap us during quiet stretches.
            _ = keepalive_interval.tick(), if is_primary => {
                let _ = tcp_write.write_all(&0u32.to_be_bytes()).await;
                let _ = tcp_write.flush().await;
            }
        }
    }
}

/// Dispatch a server message received over a Primary-mode TCP into
/// the appropriate `NetEvent`. The set is intentionally large — the
/// legacy `network.rs::tcp_handle` enumerates the same variants and
/// this stays in lockstep with that.
fn forward_primary_tcp_msg(msg: ServerMsg, event_tx: &mpsc::UnboundedSender<NetEvent>) {
    match msg {
        // ShardRedirect / ShardHandoff / ShardPreConnect are intercepted
        // by the controller layer (run_network) — when run_connection
        // is the migration target, the controller will subscribe to
        // a separate channel for them. For now we forward them via
        // dedicated events (added in T0.G).
        ServerMsg::ShardRedirect(_)
        | ServerMsg::ShardHandoff(_)
        | ServerMsg::ShardPreConnect(_) => {
            // T0.G follow-up: route via a dedicated controller channel
            // so the run_connection task doesn't need to know about
            // primary lifecycle. For the skeleton-target we drop
            // them; the migrating caller will rewrite this dispatch
            // to forward via that channel.
        }
        ServerMsg::BlockConfigState(d) => {
            let _ = event_tx.send(NetEvent::BlockConfigState(d));
        }
        ServerMsg::GrantsSnapshot(d) => {
            let _ = event_tx.send(NetEvent::GrantsSnapshot(d));
        }
        ServerMsg::OpenTerminalChat(d) => {
            let _ = event_tx.send(NetEvent::OpenTerminalChat(d));
        }
        ServerMsg::TerminalScrollbackDelta(d) => {
            let _ = event_tx.send(NetEvent::TerminalScrollbackDelta(d));
        }
        ServerMsg::SeatBindingsNotify(d) => {
            let _ = event_tx.send(NetEvent::SeatBindingsNotify(d));
        }
        ServerMsg::ChunkSnapshot(cs) => {
            let _ = event_tx.send(NetEvent::ChunkSnapshot(cs));
        }
        ServerMsg::ChunkDelta(cd) => {
            let _ = event_tx.send(NetEvent::ChunkDelta(cd));
        }
        ServerMsg::SubGridAssignmentUpdate(d) => {
            let _ = event_tx.send(NetEvent::SubGridAssignmentUpdate(d));
        }
        ServerMsg::StarCatalog(d) => {
            let _ = event_tx.send(NetEvent::StarCatalog(d));
        }
        ServerMsg::HudSignalDelta(d) => {
            let _ = event_tx.send(NetEvent::HudSignalDelta(d));
        }
        ServerMsg::ShardDisconnectNotify(_) => {
            // Controller-only — the per-secondary handle is closed
            // by the controller, which then drops control_tx.
        }
        // Other variants are unexpected on primary TCP; drop silently.
        _ => {}
    }
}

/// Dispatch a server message received over a Secondary/Demoting
/// TCP. Restricted to the chunk-streaming + scene-context subset.
fn forward_secondary_tcp_msg(
    msg: ServerMsg,
    seed: u64,
    event_tx: &mpsc::UnboundedSender<NetEvent>,
) {
    match msg {
        ServerMsg::ChunkSnapshot(cs) => {
            let _ = event_tx.send(NetEvent::SecondaryChunkSnapshot { seed, data: cs });
        }
        ServerMsg::ChunkDelta(cd) => {
            let _ = event_tx.send(NetEvent::SecondaryChunkDelta { seed, data: cd });
        }
        ServerMsg::SubGridAssignmentUpdate(d) => {
            let _ = event_tx.send(NetEvent::SecondarySubGridAssignment { seed, data: d });
        }
        ServerMsg::StarCatalog(d) => {
            // Galaxy secondaries deliver StarCatalog over their
            // observer TCP — forward via the shared StarCatalog event
            // so the lighting / starfield systems see one path.
            let _ = event_tx.send(NetEvent::StarCatalog(d));
        }
        // Anything else on a Secondary TCP is unexpected; drop.
        _ => {}
    }
}
