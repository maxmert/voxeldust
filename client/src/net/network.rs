//! Client networking — Phase T0 controller architecture.
//!
//! [`run_network`] is the high-level controller. It owns the
//! gateway-resolution → primary-handshake flow and dispatches
//! [`ControllerEvent`]s arriving from the active primary's TCP read
//! loop. All actual TCP/UDP I/O lives in
//! [`super::connection::run_connection`]; the legacy split-task
//! topology (separate primary `send_handle` + `recv_handle` +
//! `tcp_handle` triple AND per-secondary observer-UDP-task +
//! observer-TCP-task pair) has been removed in favour of the
//! unified per-connection task.
//!
//! ## Controller responsibilities
//!
//! 1. Initial gateway resolution → first shard address pair (or
//!    `--direct` override).
//! 2. Per-shard handshake: send `ClientMsg::Connect`, receive
//!    `ServerMsg::JoinResponse`, bind UDP, then spawn
//!    [`run_connection`] in `Primary` mode.
//! 3. Listen on `controller_rx` for `ControllerEvent`s the primary
//!    forwards from its TCP read loop:
//!    - `ShardRedirect` → cancel current primary, cancel non-scene
//!      secondaries, fresh handshake to new shard (legacy non-
//!      seamless path).
//!    - `ShardHandoff` → look up destination secondary's handle,
//!      send `Promote` to it, send `Demote` to old primary
//!      (seamless T0 path — no fresh handshake, no socket churn).
//!    - `ShardPreConnect` → handshake-less open of a Secondary
//!      `run_connection` (sends `ObserverConnect` carrying the
//!      session token internally).
//!    - `ShardDisconnect` → `Cancel` the matching secondary.
//! 4. Detect primary task exit (TCP EOF, UDP error) and emit
//!    `NetEvent::Disconnected`.
//!
//! ## Invariants preserved from the legacy implementation
//!
//! * Scene-context secondaries (`shard_type` 1=System, 3=Galaxy)
//!   survive every `ShardRedirect` so star catalogue / system bodies
//!   never blip off mid-transition.
//! * `MAX_SECONDARIES` = 4 cap on simultaneous Ship/Planet
//!   secondaries (eviction is FIFO of the existing entries when a
//!   new pre-connect would exceed the cap).
//! * Per-key replacement: a `ShardPreConnect` for an already-known
//!   `(shard_type, shard_id)` cancels the prior secondary first.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use glam::{DQuat, DVec3};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpStream, UdpSocket};
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tracing::{info, warn};

use voxeldust_core::client_message::{
    BlockEditData, ChunkDeltaData, ChunkSnapshotData, ClientMsg, PlayerInputData, ServerMsg,
    WorldStateData,
};
use voxeldust_core::handoff::{ShardPreConnect, ShardRedirect, SpawnPose};
use voxeldust_core::shard_types::SessionToken;

use super::connection::{
    run_connection, ConnectionControl, ConnectionHandle, ConnectionMode, ControllerEvent,
};

/// Events from the network thread to the render thread.
pub enum NetEvent {
    Connected {
        shard_type: u8,
        seed: u64,
        reference_position: DVec3,
        reference_rotation: DQuat,
        game_time: f64,
        system_seed: u64,
        galaxy_seed: u64,
        /// Our player's id on this shard. Client must filter
        /// `WorldState.players[]` by this to avoid picking up someone
        /// else's position (system-shard EVA broadcasts include all
        /// players, not just the recipient).
        player_id: u64,
    },
    WorldState(WorldStateData),
    /// A secondary shard has been pre-connected for rendering.
    SecondaryConnected {
        shard_type: u8,
        seed: u64,
        reference_position: DVec3,
        reference_rotation: DQuat,
    },
    /// A secondary shard's connection has ended (for any reason: server-sent
    /// `ShardDisconnectNotify`, replacement by a new pre-connect for the
    /// same key, primary-transition cancel, session end, or UDP error).
    /// The main thread must release the chunk source associated with `seed`
    /// so its GPU buffers are freed — otherwise chunks accumulate across
    /// every ship transition.
    SecondaryDisconnected {
        seed: u64,
    },
    /// WorldState from a secondary shard (for composite rendering).
    /// `shard_type` identifies which secondary the WS came from —
    /// necessary because multiple secondaries (SYSTEM, PLANET) can be
    /// connected simultaneously and their WorldStates interleave. The
    /// client uses this to route the WS to the correct per-type slot
    /// so Transitioning's `ws.secondary.take()` picks the secondary
    /// matching the redirect target, not whichever sent last.
    ///
    /// `seed` is the secondary's authoritative wire seed (= the
    /// `seed` carried by `NetEvent::SecondaryConnected` and used as
    /// the `ShardKey.seed` in `Secondaries.runtimes` /
    /// `SourceIndex.by_shard`). Without this, multiple secondaries
    /// of the same type (e.g. several pre-connected SHIPs) collapse
    /// into one slot in `SecondaryWorldStates.by_shard_type` and
    /// downstream lookups have to guess the seed — leading to
    /// player visuals being parented under the wrong ship's
    /// `ChunkSource` (or, if the guess misses, never spawning).
    SecondaryWorldState {
        shard_type: u8,
        seed: u64,
        ws: WorldStateData,
    },
    /// Galaxy world state from secondary UDP (warp travel position for star parallax).
    GalaxyWorldState(voxeldust_core::client_message::GalaxyWorldStateData),
    /// Block signal config state from server (config UI).
    BlockConfigState(voxeldust_core::signal::config::BlockSignalConfig),
    /// Phase D: server snapshot of the player's owned grants. Sent in
    /// response to `GrantCreate` / `GrantRevoke` and as a one-shot after
    /// `AddHeldGrant` so the grants panel always reflects authoritative
    /// state.
    GrantsSnapshot(voxeldust_core::client_message::GrantsSnapshotData),
    /// Phase D: server response to E (INTERACT) on a Terminal block.
    /// Carries the engaged block's position, configured channels, and
    /// the current scrollback so the client can paint the in-world
    /// terminal screen with content right away.
    OpenTerminalChat(voxeldust_core::client_message::OpenTerminalChatData),
    /// Phase D: real-time scrollback append while the player is engaged
    /// with a terminal.
    TerminalScrollbackDelta(voxeldust_core::client_message::TerminalScrollbackDeltaData),
    /// Seat bindings from server (when player enters a seat).
    SeatBindingsNotify(voxeldust_core::client_message::SeatBindingsNotifyData),
    /// Sub-grid block assignments from server (mechanical mount membership).
    SubGridAssignmentUpdate(voxeldust_core::client_message::SubGridAssignmentData),
    /// Full chunk snapshot received (initial sync or resync).
    ChunkSnapshot(ChunkSnapshotData),
    /// Incremental block changes to a chunk.
    ChunkDelta(ChunkDeltaData),
    /// Chunk snapshot from a secondary (observer) connection.
    SecondaryChunkSnapshot {
        seed: u64,
        data: ChunkSnapshotData,
    },
    /// Chunk delta from a secondary (observer) connection.
    SecondaryChunkDelta {
        seed: u64,
        data: ChunkDeltaData,
    },
    /// Sub-grid assignments from a secondary (observer) connection.
    SecondarySubGridAssignment {
        seed: u64,
        data: voxeldust_core::client_message::SubGridAssignmentData,
    },
    /// Primary shard is changing (ShardRedirect received).
    /// `target_shard_type` disambiguates which open secondary to promote when
    /// more than one is live (e.g., Ship + Galaxy secondaries). `255` = legacy
    /// redirect with no type hint (falls back to last-connected secondary).
    ///
    /// `spawn_pose` (when present) carries the authoritative
    /// post-transition camera pose, computed server-side. The client uses
    /// this directly for its first rendered frame after the primary switch,
    /// so there is zero client-side prediction of position/rotation across
    /// the handoff.
    Transitioning {
        target_shard_type: u8,
        spawn_pose: Option<voxeldust_core::handoff::SpawnPose>,
    },
    /// Full star catalogue for the galaxy — sent by the galaxy
    /// shard once per connect. Authoritative; no per-tick update.
    StarCatalog(voxeldust_core::client_message::StarCatalogData),
    /// Phase 4.4: server-pushed HUD signal delta. Resolved against
    /// the client's per-session `InboundDict` and applied to the
    /// `SignalRegistry`. Coexists with the legacy
    /// `WorldStateData.hud_signals` UDP path until that field is
    /// removed.
    HudSignalDelta(voxeldust_core::client_message::HudSignalDeltaData),
    Disconnected(String),
}

/// Maximum simultaneous Ship/Planet secondaries (the cap excludes
/// scene-context secondaries which are always-on for the session).
/// Mirrors the legacy `network.rs::run_network` constant.
const MAX_SECONDARIES: usize = 4;

/// Returns true for the always-on scene-context shard types
/// (1 = System, 3 = Galaxy). These survive every `ShardRedirect`
/// so star catalogue / system bodies / warp parallax never blip
/// off mid-transition.
fn is_scene_context(shard_type: u8) -> bool {
    shard_type == 1 || shard_type == 3
}

/// Tracks one running connection task: the handle to send control
/// messages, the JoinHandle for exit detection, and the connection's
/// reference pose so a later `Promote` (Secondary→Primary transition
/// via `ShardHandoff`) can emit the correct `NetEvent::Connected`
/// for the registry's primary-promotion code path.
struct ActiveConn {
    handle: ConnectionHandle,
    join: JoinHandle<()>,
    /// Reference position of the shard's local frame, captured at
    /// handshake time (`JoinResponse.reference_position` for
    /// primary; `ShardPreConnect.reference_position` for secondary).
    reference_position: DVec3,
    /// Reference rotation of the shard's local frame.
    reference_rotation: DQuat,
}

/// Run the network controller. Owns the lifecycle of the player's
/// primary shard connection plus all pre-connected secondaries.
/// Spawns one [`run_connection`] task per shard (primary or
/// secondary) and dispatches [`ControllerEvent`]s from the active
/// primary to drive transitions.
///
/// `ship_join_key` is the optional `--ship-join` CLI value. When
/// non-empty, the gateway hashes it (instead of `player_name`) to
/// pick the ship_id, so two clients sharing a join key land on the
/// same ship. Sessions remain keyed on `player_name` either way.
pub async fn run_network(
    gateway_addr: SocketAddr,
    player_name: String,
    ship_join_key: String,
    event_tx: mpsc::UnboundedSender<NetEvent>,
    input_rx: Arc<Mutex<mpsc::UnboundedReceiver<PlayerInputData>>>,
    block_edit_rx: Arc<Mutex<mpsc::UnboundedReceiver<BlockEditData>>>,
    tcp_out_rx: Arc<Mutex<mpsc::UnboundedReceiver<Vec<u8>>>>,
    direct: Option<String>,
) {
    // Suppress unused-warning on the kept-for-future-fallback channel —
    // BlockEditData is currently routed via the TCP-out channel
    // (handed to run_connection); the dedicated UDP-fallback
    // pipeline that block_edit_rx was provisioned for never landed.
    let _ = block_edit_rx;

    // 1) Initial address resolution.
    let Some((mut next_tcp_addr, mut next_udp_addr)) =
        resolve_initial_addrs(gateway_addr, &player_name, &ship_join_key, direct, &event_tx)
            .await
    else {
        return; // resolve_initial_addrs already emitted Disconnected.
    };

    let (controller_tx, mut controller_rx) = mpsc::channel::<ControllerEvent>(64);

    let mut primary: Option<ActiveConn> = None;
    let mut secondaries: HashMap<(u8, u64), ActiveConn> = HashMap::new();
    /// Demoting connections (former primaries waiting for TTL self-
    /// exit, plus cancelled secondaries draining out). We must keep
    /// the FULL `ActiveConn` here — NOT just the `JoinHandle` —
    /// because `ConnectionHandle` owns the only `control_tx` for the
    /// connection task; dropping the handle immediately closes the
    /// channel, which makes the task's `control_rx.recv()` return
    /// `None` on its NEXT poll, causing an immediate task exit
    /// regardless of the demote TTL the task was supposed to honour.
    /// That premature exit drops the TCP socket → server sees EOF →
    /// `SecondaryDisconnected` may not propagate cleanly (the
    /// `if !is_primary` check in connection.rs's TCP-error path
    /// silently swallows the event when the task hasn't yet
    /// processed the `Demote` control message). Net result: the
    /// graced ChunkSource has no live feed and downstream visuals
    /// age out.
    let mut demoting: Vec<ActiveConn> = Vec::new();
    /// The player's authoritative SessionToken — captured on the
    /// FIRST primary handshake and reused for every subsequent
    /// secondary's `ObserverConnect` so the destination shard can
    /// associate observers with the player's session for seamless
    /// promote (Phase T0). `SessionToken` is documented as
    /// "stable across handoffs"; we capture once and treat as
    /// constant for the rest of the session.
    let mut session_token: Option<SessionToken> = None;

    // 2) Main controller loop.
    loop {
        // Ensure primary is connected — handshake if not.
        if primary.is_none() {
            info!(
                %next_tcp_addr,
                %next_udp_addr,
                "controller: handshaking primary"
            );
            match handshake_and_spawn_primary(
                next_tcp_addr,
                next_udp_addr,
                &player_name,
                &ship_join_key,
                event_tx.clone(),
                input_rx.clone(),
                tcp_out_rx.clone(),
                controller_tx.clone(),
            )
            .await
            {
                Ok((active, st)) => {
                    primary = Some(active);
                    if session_token.is_none() {
                        session_token = Some(st);
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        %next_tcp_addr,
                        %e,
                        "primary handshake failed — disconnecting"
                    );
                    let _ = event_tx.send(NetEvent::Disconnected(format!("handshake: {e}")));
                    return;
                }
            }
        }

        // Prune any demoting tasks that have completed.
        demoting.retain(|a| !a.join.is_finished());

        let primary_join: &mut JoinHandle<()> = &mut primary
            .as_mut()
            .expect("primary must be Some after handshake")
            .join;

        tokio::select! {
            // Primary-task-exit detection — TCP EOF / UDP error / etc.
            join_result = primary_join => {
                match join_result {
                    Ok(_) => warn!("primary task exited cleanly (no redirect / handoff seen)"),
                    Err(e) => warn!(%e, "primary task panicked"),
                }
                let _ = event_tx.send(NetEvent::Disconnected(
                    "primary connection lost".into()));
                return;
            }
            // Controller events forwarded from the primary's TCP loop.
            event = controller_rx.recv() => {
                let Some(event) = event else {
                    warn!("controller_rx closed — all senders dropped");
                    let _ = event_tx.send(NetEvent::Disconnected(
                        "controller channel closed".into()));
                    return;
                };
                match event {
                    ControllerEvent::ShardRedirect(r) => {
                        info!(
                            target_tcp = %r.target_tcp_addr,
                            target_shard_type = r.target_shard_type,
                            "controller: ShardRedirect — fresh handshake to new primary"
                        );
                        // Emit Transitioning BEFORE teardown so
                        // downstream camera-pose latch can apply
                        // spawn_pose on its next frame.
                        let _ = event_tx.send(NetEvent::Transitioning {
                            target_shard_type: r.target_shard_type,
                            spawn_pose: r.spawn_pose.clone(),
                        });
                        // Cancel current primary and await its exit.
                        if let Some(ActiveConn { handle, join, .. }) = primary.take() {
                            let _ = handle.cancel().await;
                            let _ = join.await;
                        }
                        // Cancel non-scene-context secondaries.
                        // Scene-context (System/Galaxy) survive the
                        // primary swap so their chunks don't blip off.
                        let dropped: Vec<(u8, u64)> = secondaries
                            .keys()
                            .filter(|(st, _)| !is_scene_context(*st))
                            .copied()
                            .collect();
                        for key in dropped {
                            if let Some(c) = secondaries.remove(&key) {
                                let _ = c.handle.cancel().await;
                                // Don't await — secondaries exit
                                // promptly on cancel; track full
                                // ActiveConn for GC (keeps the
                                // handle's control_tx alive so the
                                // Cancel control message can be
                                // processed BEFORE the channel closes).
                                demoting.push(c);
                            }
                        }
                        // Update next_addr; loop will re-handshake.
                        match (r.target_tcp_addr.parse(), r.target_udp_addr.parse()) {
                            (Ok(tcp), Ok(udp)) => {
                                next_tcp_addr = tcp;
                                next_udp_addr = udp;
                            }
                            _ => {
                                let _ = event_tx.send(NetEvent::Disconnected(
                                    "bad redirect addrs".into()));
                                return;
                            }
                        }
                    }
                    ControllerEvent::ShardHandoff(h) => {
                        info!(
                            promote_shard_type = h.promote_shard_type,
                            promote_shard_id = h.promote_shard_id.0,
                            "controller: ShardHandoff (seamless promote)"
                        );
                        let dest_key = (h.promote_shard_type, h.promote_shard_id.0);
                        let spawn_pose = Some(SpawnPose {
                            position: h.handoff_position,
                            rotation: h.handoff_rotation,
                            velocity: h.handoff_velocity,
                        });
                        if let Some(dest) = secondaries.remove(&dest_key) {
                            // Emit Transitioning so downstream camera-
                            // pose latch sees the spawn_pose.
                            let _ = event_tx.send(NetEvent::Transitioning {
                                target_shard_type: h.promote_shard_type,
                                spawn_pose: spawn_pose.clone(),
                            });
                            // Demote old primary; it will self-exit at
                            // the configured TTL. Move the FULL ActiveConn
                            // into the demoting Vec — we MUST keep
                            // `handle` alive so its `control_tx` stays
                            // open, otherwise the task's
                            // `control_rx.recv()` returns `None` on the
                            // next poll and the task exits IMMEDIATELY
                            // instead of honouring the TTL (the
                            // documented "self-exit at TTL" only works
                            // if the channel stays open).
                            if let Some(active) = primary.take() {
                                let _ = active.handle.demote(h.source_demote_after_ticks).await;
                                demoting.push(active);
                            }
                            // Promote dest in place — sockets reused,
                            // no fresh TCP handshake.
                            if let Err(e) =
                                dest.handle.promote(spawn_pose, h.promote_shard_type).await
                            {
                                warn!(%e, "promote send to dest secondary failed");
                            }
                            // Emit Connected for the new primary so the
                            // shard registry's promote-in-place code
                            // path runs (registry.rs::handle_connected
                            // moves the existing ChunkSource entity from
                            // secondaries.runtimes into the primary
                            // slot — same entity, same chunks, no
                            // re-stream). reference_position/_rotation
                            // are the secondary's values from
                            // ShardPreConnect, captured in ActiveConn.
                            let session_id =
                                session_token.map(|s| s.0).unwrap_or(0);
                            let _ = event_tx.send(NetEvent::Connected {
                                shard_type: h.promote_shard_type,
                                seed: dest.handle.seed,
                                reference_position: dest.reference_position,
                                reference_rotation: dest.reference_rotation,
                                game_time: 0.0,
                                system_seed: 0,
                                galaxy_seed: 0,
                                player_id: session_id,
                            });
                            primary = Some(dest);
                        } else {
                            // Fallback: dest not pre-connected. Treat
                            // as ShardRedirect.
                            warn!(
                                promote_shard_type = h.promote_shard_type,
                                promote_shard_id = h.promote_shard_id.0,
                                "ShardHandoff for unknown secondary — falling back to fresh handshake"
                            );
                            let _ = event_tx.send(NetEvent::Transitioning {
                                target_shard_type: h.promote_shard_type,
                                spawn_pose,
                            });
                            if let Some(ActiveConn { handle, join, .. }) = primary.take() {
                                let _ = handle.cancel().await;
                                let _ = join.await;
                            }
                            match (h.target_tcp_addr.parse(), h.target_udp_addr.parse()) {
                                (Ok(tcp), Ok(udp)) => {
                                    next_tcp_addr = tcp;
                                    next_udp_addr = udp;
                                }
                                _ => {
                                    let _ = event_tx.send(NetEvent::Disconnected(
                                        "bad handoff addrs".into()));
                                    return;
                                }
                            }
                        }
                    }
                    ControllerEvent::ShardPreConnect(pc) => {
                        let st = match session_token {
                            Some(s) => s,
                            None => {
                                warn!("ShardPreConnect arrived before any session_token captured — ignoring");
                                continue;
                            }
                        };
                        handle_preconnect(
                            pc,
                            &mut secondaries,
                            &mut demoting,
                            &event_tx,
                            input_rx.clone(),
                            tcp_out_rx.clone(),
                            controller_tx.clone(),
                            st,
                        )
                        .await;
                    }
                    ControllerEvent::ShardDisconnect { shard_type, seed } => {
                        let key = (shard_type, seed);
                        if let Some(c) = secondaries.remove(&key) {
                            info!(
                                shard_type, seed,
                                "controller: ShardDisconnect — cancelling secondary"
                            );
                            let _ = c.handle.cancel().await;
                            demoting.push(c);
                        }
                    }
                }
            }
        }
    }
}

/// Resolve the first shard's address pair via gateway redirect or
/// the explicit `--direct` override. Returns `None` after emitting
/// `NetEvent::Disconnected` on any failure.
async fn resolve_initial_addrs(
    gateway_addr: SocketAddr,
    player_name: &str,
    ship_join_key: &str,
    direct: Option<String>,
    event_tx: &mpsc::UnboundedSender<NetEvent>,
) -> Option<(SocketAddr, SocketAddr)> {
    if let Some(direct_str) = direct {
        let parts: Vec<&str> = direct_str.split(',').collect();
        let tcp: SocketAddr = parts[0].parse().expect("bad direct tcp addr");
        let udp: SocketAddr = if parts.len() > 1 {
            parts[1].parse().expect("bad direct udp addr")
        } else {
            SocketAddr::new(tcp.ip(), tcp.port() + 1)
        };
        info!(%tcp, %udp, "direct shard connection");
        Some((tcp, udp))
    } else {
        info!(%gateway_addr, "connecting to gateway");
        let redirect = match connect_to_gateway(gateway_addr, player_name, ship_join_key).await
        {
            Ok(r) => r,
            Err(e) => {
                let _ = event_tx.send(NetEvent::Disconnected(format!("gateway error: {e}")));
                return None;
            }
        };
        info!(
            tcp = %redirect.target_tcp_addr,
            udp = %redirect.target_udp_addr,
            "received gateway redirect"
        );
        match (
            redirect.target_tcp_addr.parse(),
            redirect.target_udp_addr.parse(),
        ) {
            (Ok(tcp), Ok(udp)) => Some((tcp, udp)),
            _ => {
                let _ = event_tx
                    .send(NetEvent::Disconnected("bad gateway redirect addrs".into()));
                None
            }
        }
    }
}

/// Do the `Connect` → `JoinResponse` TCP handshake, bind UDP, then
/// spawn `run_connection` in `Primary` mode. Emits
/// `NetEvent::Connected` so the shard registry can spawn a
/// ChunkSource for this shard.
#[allow(clippy::too_many_arguments)]
async fn handshake_and_spawn_primary(
    tcp_addr: SocketAddr,
    udp_addr: SocketAddr,
    player_name: &str,
    ship_join_key: &str,
    event_tx: mpsc::UnboundedSender<NetEvent>,
    input_rx: Arc<Mutex<mpsc::UnboundedReceiver<PlayerInputData>>>,
    tcp_out_rx: Arc<Mutex<mpsc::UnboundedReceiver<Vec<u8>>>>,
    controller_tx: mpsc::Sender<ControllerEvent>,
) -> Result<(ActiveConn, SessionToken), Box<dyn std::error::Error + Send + Sync>> {
    let (tcp, jr) = connect_to_shard_full(tcp_addr, player_name, ship_join_key).await?;
    info!(shard_type = jr.shard_type, "joined shard");
    let _ = event_tx.send(NetEvent::Connected {
        shard_type: jr.shard_type,
        seed: jr.seed,
        reference_position: jr.reference_position,
        reference_rotation: jr.reference_rotation,
        game_time: jr.game_time,
        system_seed: jr.system_seed,
        galaxy_seed: jr.galaxy_seed,
        player_id: jr.player_id,
    });

    let udp = UdpSocket::bind("0.0.0.0:0").await?;

    let (control_tx, control_rx) = mpsc::channel::<ConnectionControl>(8);
    // For the PRIMARY connection, JoinResponse carries `seed` only,
    // not a separate `shard_id`. Use seed for both (downstream
    // registry keys ChunkSource by ShardKey { shard_type, seed }
    // anyway).
    let handle = ConnectionHandle::new(jr.shard_type, jr.seed, jr.seed, control_tx);

    let join = tokio::spawn(run_connection(
        ConnectionMode::Primary,
        tcp,
        udp,
        udp_addr,
        jr.shard_type,
        jr.seed,
        jr.seed,
        jr.session_token,
        event_tx,
        input_rx,
        tcp_out_rx,
        control_rx,
        Some(controller_tx),
    ));

    Ok((
        ActiveConn {
            handle,
            join,
            reference_position: jr.reference_position,
            reference_rotation: jr.reference_rotation,
        },
        jr.session_token,
    ))
}

/// Open a new secondary connection in response to `ShardPreConnect`.
/// Replaces any existing secondary at the same `(shard_type,
/// shard_id)` key, evicts the oldest non-scene secondary if
/// `MAX_SECONDARIES` would be exceeded, then opens TCP+UDP and
/// spawns `run_connection` in `Secondary` mode.
#[allow(clippy::too_many_arguments)]
async fn handle_preconnect(
    pc: ShardPreConnect,
    secondaries: &mut HashMap<(u8, u64), ActiveConn>,
    demoting: &mut Vec<ActiveConn>,
    event_tx: &mpsc::UnboundedSender<NetEvent>,
    input_rx: Arc<Mutex<mpsc::UnboundedReceiver<PlayerInputData>>>,
    tcp_out_rx: Arc<Mutex<mpsc::UnboundedReceiver<Vec<u8>>>>,
    controller_tx: mpsc::Sender<ControllerEvent>,
    session_token: SessionToken,
) {
    let key = (pc.shard_type, pc.shard_id);
    info!(
        shard_type = pc.shard_type,
        shard_id = pc.shard_id,
        seed = pc.seed,
        tcp = %pc.tcp_addr,
        udp = %pc.udp_addr,
        "controller: ShardPreConnect — opening secondary"
    );

    // Replace existing secondary at same key.
    if let Some(existing) = secondaries.remove(&key) {
        info!(
            shard_type = key.0,
            shard_id = key.1,
            "ShardPreConnect: replacing existing secondary at same key"
        );
        let _ = existing.handle.cancel().await;
        demoting.push(existing);
    }

    // Evict oldest non-scene secondary if over MAX_SECONDARIES.
    if !is_scene_context(pc.shard_type) {
        loop {
            let count = secondaries
                .keys()
                .filter(|(st, _)| !is_scene_context(*st))
                .count();
            if count < MAX_SECONDARIES {
                break;
            }
            let evict_key = secondaries
                .keys()
                .find(|(st, _)| !is_scene_context(*st))
                .copied();
            let Some(k) = evict_key else { break };
            if let Some(c) = secondaries.remove(&k) {
                info!(
                    shard_type = k.0,
                    shard_id = k.1,
                    "evicting secondary — MAX_SECONDARIES reached"
                );
                let _ = c.handle.cancel().await;
                demoting.push(c);
            }
        }
    }

    // Open destination TCP + UDP.
    let tcp_addr: SocketAddr = match pc.tcp_addr.parse() {
        Ok(a) => a,
        Err(e) => {
            warn!(%e, "bad ShardPreConnect tcp addr — skipping secondary");
            return;
        }
    };
    let udp_target: SocketAddr = match pc.udp_addr.parse() {
        Ok(a) => a,
        Err(e) => {
            warn!(%e, "bad ShardPreConnect udp addr — skipping secondary");
            return;
        }
    };

    let tcp = match TcpStream::connect(tcp_addr).await {
        Ok(s) => s,
        Err(e) => {
            warn!(%e, %tcp_addr, "secondary TCP connect failed");
            return;
        }
    };
    let _ = tcp.set_nodelay(true);

    let udp = match UdpSocket::bind("0.0.0.0:0").await {
        Ok(s) => s,
        Err(e) => {
            warn!(%e, "secondary UDP bind failed");
            return;
        }
    };

    // Emit SecondaryConnected with the AUTHORITATIVE
    // reference_position / reference_rotation from ShardPreConnect
    // BEFORE spawning the task. The shard registry uses these to
    // initialise the ChunkSource's ShardOrigin so chunks render at
    // the right system-space coordinates from the very first
    // WorldState tick.
    let _ = event_tx.send(NetEvent::SecondaryConnected {
        shard_type: pc.shard_type,
        seed: pc.seed,
        reference_position: pc.reference_position,
        reference_rotation: pc.reference_rotation,
    });

    let (control_tx, control_rx) = mpsc::channel::<ConnectionControl>(8);
    let handle = ConnectionHandle::new(pc.shard_type, pc.shard_id, pc.seed, control_tx);

    // Spawn run_connection in Secondary mode. controller_tx is
    // Some so this secondary can be promoted to Primary later via
    // ShardHandoff (Phase T0 seamless transition).
    let join = tokio::spawn(run_connection(
        ConnectionMode::Secondary,
        tcp,
        udp,
        udp_target,
        pc.shard_type,
        pc.shard_id,
        pc.seed,
        session_token,
        event_tx.clone(),
        input_rx,
        tcp_out_rx,
        control_rx,
        Some(controller_tx),
    ));

    secondaries.insert(
        key,
        ActiveConn {
            handle,
            join,
            reference_position: pc.reference_position,
            reference_rotation: pc.reference_rotation,
        },
    );
}

async fn connect_to_gateway(
    addr: SocketAddr,
    player_name: &str,
    ship_join_key: &str,
) -> Result<ShardRedirect, Box<dyn std::error::Error + Send + Sync>> {
    let mut stream = TcpStream::connect(addr).await?;
    send_msg(
        &mut stream,
        &ClientMsg::Connect {
            player_name: player_name.to_string(),
            ship_join_key: ship_join_key.to_string(),
        },
    )
    .await?;
    let response = recv_server_msg(&mut stream).await?;
    match response {
        ServerMsg::ShardRedirect(r) => Ok(r),
        other => Err(format!(
            "expected ShardRedirect, got {:?}",
            std::mem::discriminant(&other)
        )
        .into()),
    }
}

/// Connect to a shard, returning both the TCP stream (handed off
/// to the spawned `run_connection` task) and the JoinResponse so
/// the controller can emit `NetEvent::Connected` and capture the
/// session_token.
async fn connect_to_shard_full(
    addr: SocketAddr,
    player_name: &str,
    ship_join_key: &str,
) -> Result<
    (TcpStream, voxeldust_core::client_message::JoinResponseData),
    Box<dyn std::error::Error + Send + Sync>,
> {
    let mut stream = TcpStream::connect(addr).await?;
    let _ = stream.set_nodelay(true);

    send_msg(
        &mut stream,
        &ClientMsg::Connect {
            player_name: player_name.to_string(),
            ship_join_key: ship_join_key.to_string(),
        },
    )
    .await?;
    let response = recv_server_msg(&mut stream).await?;
    match response {
        ServerMsg::JoinResponse(jr) => Ok((stream, jr)),
        other => Err(format!(
            "expected JoinResponse, got {:?}",
            std::mem::discriminant(&other)
        )
        .into()),
    }
}

async fn send_msg(stream: &mut TcpStream, msg: &ClientMsg) -> Result<(), std::io::Error> {
    let data = msg.serialize();
    let mut buf = Vec::new();
    voxeldust_core::wire_codec::encode(&data, &mut buf);
    stream.write_all(&buf).await?;
    stream.flush().await?;
    Ok(())
}

async fn recv_server_msg(
    stream: &mut (impl tokio::io::AsyncRead + Unpin),
) -> Result<ServerMsg, Box<dyn std::error::Error + Send + Sync>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).await?;
    let decoded =
        voxeldust_core::wire_codec::decode(&buf).map_err(|e| format!("wire decode: {e}"))?;
    Ok(ServerMsg::deserialize(&decoded)?)
}
