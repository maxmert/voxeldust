//! Gateway M0: ONE subscription, ONE authority, NO transfer
//! (`docs/design/connection_plane.md` §M0). The client holds exactly one logical
//! connection; everything server-side routes by in-frame `SessionId` + `Fence`,
//! NEVER by source address (R2).
//!
//! Binding shapes that exist NOW because P2 cannot retrofit them:
//! - `RouteSnapshot { authority, fence, cut }` behind `ArcSwap`; `route.store` is
//!   the gateway's SOLE route-mutation primitive (P2's `CommitAuthority` drives it).
//! - The 20 Hz hot paths ([`route_input`], [`forward_frame`]) touch ONLY the
//!   `ArcSwap` load, one `AtomicU64`, and a byte-level header re-tag — no lock any
//!   control path takes (SPIKE-2a benches exactly these functions).
//! - Every forwarded frame's fence is compared against the route's accepted fence;
//!   stale frames are dropped and counted (fence rule 5 — inert with one authority,
//!   load-bearing the moment P2 swaps routes).
//! - The gateway is the SOLE ticket validator; the session mint COMMITS at the
//!   orchestrator's directory insert (the gateway only proposes entropy).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arc_swap::ArcSwap;
use bevy_ecs::prelude::{IntoScheduleConfigs, Res, ResMut, Resource, Schedule, World};
use vd_core::rng::SplitMix64;
use vd_core::{AccountId, EntityId, Fence, NodeId, SessionId};
use vd_sim::io::{Inbound, MsgClass};
use vd_sim::runtime::{ClockSample, InboundBox, NodeIdentity, OutboundBox};
use vd_wire::channels::{ClientControlMsg, ServerControlMsg, SubId};
use vd_wire::intershard::InterShardFlow;
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey, DirectoryOp, DirectoryReply};
use vd_wire::session_flow::{GatewayToShard, ShardToGateway, peek_input_seq, retag_snapshot_sub};
use vd_wire::version::ProtoVersion;

use crate::tickets;

/// Gateway operational parameters — ONE reviewed struct, no inline literals.
#[derive(Clone, Copy, Debug)]
pub struct TransportTuning {
    /// Hard cap on concurrent sessions (beyond it, logins are refused loudly).
    pub max_sessions: usize,
}

/// Gateway configuration (composer-provided).
#[derive(Resource, Clone, Copy, Debug)]
pub struct GatewayConfig {
    pub orchestrator: NodeId,
    /// P1: the single stub shard every session lands on.
    pub shard: NodeId,
    /// The auth service's Ed25519 verifying key (login validation).
    pub auth_verifying_key: [u8; tickets::ED25519_KEY_BYTES],
    /// Seed for session-id proposal entropy (the directory insert is the mint).
    pub session_seed: u64,
    /// The cluster's universe-tick rate (Hz), relayed to clients via
    /// `ServerControlMsg::UniverseRate` so they drive the render cursor at the
    /// server's rate. The SAME value the node feeds its `TickPacer` (VD_TICK_HZ).
    pub tick_hz: u32,
    pub tuning: TransportTuning,
}

/// The route a session's input follows and the fence its frames are accepted at.
/// P2's transfer commit swaps BOTH atomically via one `route.store`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteSnapshot {
    pub authority: NodeId,
    /// The accepted frame fence (frames below it are dropped).
    pub fence: Fence,
    /// The input seq cut during a transfer — ALWAYS `None` in P1.
    pub cut: Option<SeqCut>,
}

/// Transfer-time input partition (P2; the shape exists so the route never reshapes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeqCut {
    pub marker_seq: u64,
    pub dest: NodeId,
}

/// The lock-free per-session hot state shared with the (future, threaded) 20 Hz
/// forwarding path. The cold session record owns an `Arc` of this.
#[derive(Debug)]
pub struct SessionHot {
    pub route: ArcSwap<RouteSnapshot>,
    pub last_input_seq: AtomicU64,
}

/// Where one session is in its login lifecycle.
#[derive(Debug)]
enum SessionPhase {
    /// Session-key grant sent; awaiting the directory head (retried every tick —
    /// idempotent by fence).
    AwaitingDirectory,
    /// Directory granted; attach sent to the shard (retried until attached).
    AwaitingAttach,
    /// Live: input routes shard-ward, frames flow client-ward.
    Active { sub: SubId, entity: EntityId },
}

/// One session's cold record (the hot part is the shared `Arc<SessionHot>`).
#[derive(Debug)]
struct Session {
    client: NodeId,
    account: AccountId,
    fence: Fence,
    phase: SessionPhase,
    next_sub: u32,
    /// The negotiated proto minor for this connection (the sender-gates-variants
    /// rule): minor-1+ variants like `UniverseRate` are emitted only when `>= 1`.
    negotiated_minor: u16,
    hot: Arc<SessionHot>,
}

/// The session table: by session id (authoritative) and by client connection
/// (in-process: the client's NodeId IS the connection).
#[derive(Resource, Debug, Default)]
pub struct GatewaySessions {
    by_session: BTreeMap<SessionId, Session>,
    by_client: BTreeMap<NodeId, SessionId>,
}

impl GatewaySessions {
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_session.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_session.is_empty()
    }
    /// The session ids currently active (for tests/oracles).
    pub fn sessions(&self) -> impl Iterator<Item = SessionId> + '_ {
        self.by_session.keys().copied()
    }
    /// The avatar entity a session renders authoritatively (None until Active).
    /// P2's transfer coordinator keys its sagas on this.
    #[must_use]
    pub fn entity_of(&self, session: SessionId) -> Option<EntityId> {
        self.by_session.get(&session).and_then(|s| match s.phase {
            SessionPhase::Active { entity, .. } => Some(entity),
            SessionPhase::AwaitingDirectory | SessionPhase::AwaitingAttach => None,
        })
    }
}

/// Session-id proposal entropy (the directory insert is the authoritative mint).
#[derive(Resource, Debug)]
struct SessionMint(SplitMix64);

/// Honesty counters: everything tolerated-but-rejected is counted, never silent.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GatewayStats {
    pub logins_rejected: u64,
    pub version_rejected: u64,
    pub sessions_refused_capacity: u64,
    pub session_mints_refused: u64,
    pub resumes_refused: u64,
    pub inputs_deduped: u64,
    pub inputs_unroutable: u64,
    pub inputs_malformed: u64,
    pub stale_frames_dropped: u64,
    pub undecodable: u64,
    /// TransferControl commands received before the consumer exists (Slice 1c.2 wires it).
    /// Counted, never mis-applied — a transient honesty surface that should be 0 once 1c.2
    /// lands (the consumer replaces this arm).
    pub transfer_control_unhandled: u64,
}

/// Install the gateway systems (composed by the harness/bin for `NodeKind::Gateway`).
pub fn register_gateway(world: &mut World, schedule: &mut Schedule, config: GatewayConfig) {
    world.insert_resource(config);
    world.insert_resource(GatewaySessions::default());
    world.insert_resource(SessionMint(SplitMix64::new(config.session_seed)));
    world.insert_resource(GatewayStats::default());
    schedule.add_systems((process_gateway_inbound, drive_pending_sessions).chain());
}

/// ---------------------------------------------------------------------------
/// THE HOT PATHS (pure, lock-free; SPIKE-2a benches these exact functions)
/// ---------------------------------------------------------------------------
///
/// Route one client input datagram: dedup by the leading seq varint, then read
/// the route via one `ArcSwap` load. No decode of the payload, no locks.
#[must_use]
pub fn route_input(hot: &SessionHot, input_bytes: &[u8]) -> InputRouting {
    let Ok(seq) = peek_input_seq(input_bytes) else {
        return InputRouting::Malformed;
    };
    // Latest-wins dedup: monotonic high-water mark on one atomic. `fetch_max` is a
    // SINGLE atomic RMW — it advances the mark to `max(prev, seq)` and returns the
    // PRIOR value, so the load+compare+store is indivisible. A non-atomic load-then-
    // store (FG-2) would let two concurrent forwarder threads both read the same
    // `last`, both pass, and both forward the same datagram (or store out of order).
    // With `fetch_max` exactly one observes `seq > prev` for any given seq, and the
    // mark never moves backward regardless of arrival interleaving. Still wait-free
    // (one instruction; no lock, no retry loop) — the SPIKE-2a hot-path budget holds.
    let prev = hot.last_input_seq.fetch_max(seq, Ordering::Relaxed);
    if seq <= prev {
        return InputRouting::Deduped;
    }
    let route = hot.route.load();
    // P1: `cut` is always None; the P2 cut partition slots in right here.
    InputRouting::Forward {
        to: route.authority,
    }
}

/// What the input hot path decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputRouting {
    Forward { to: NodeId },
    Deduped,
    Malformed,
}

/// Does this session ACCEPT a frame at `frame_fence`? (Stale-fence frames from a
/// demoted old owner are dropped — fence rule 5.) The cheap per-session half of
/// the fan-out; the expensive re-tag is shared per sub-id (SCALE-1).
#[must_use]
pub fn frame_passes_fence(hot: &SessionHot, frame_fence: Fence) -> bool {
    !frame_fence.is_stale_against(hot.route.load().fence)
}

/// Forward one shard frame to one session: fence-compare then byte-level sub re-tag.
/// Returns the client-bound bytes, or `None` if the frame is stale (counted by the
/// caller). Retained for the SPIKE-2a microbench (single-session hot path).
#[must_use]
pub fn forward_frame(
    hot: &SessionHot,
    sub: SubId,
    frame_fence: Fence,
    snapshot_bytes: &[u8],
) -> Option<Vec<u8>> {
    if !frame_passes_fence(hot, frame_fence) {
        return None;
    }
    retag_snapshot_sub(snapshot_bytes, sub).ok()
}

/// ---------------------------------------------------------------------------
/// Control-plane systems (cold paths)
/// ---------------------------------------------------------------------------
#[allow(clippy::too_many_arguments)] // bevy system: each resource is one parameter
fn process_gateway_inbound(
    config: Res<GatewayConfig>,
    identity: Res<NodeIdentity>,
    clock: Res<ClockSample>,
    inbox: Res<InboundBox>,
    mut sessions: ResMut<GatewaySessions>,
    mut mint: ResMut<SessionMint>,
    mut stats: ResMut<GatewayStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    for msg in &inbox.0 {
        let Inbound::Wire { from, class, bytes } = msg else {
            continue;
        };
        let from = *from;
        if from == config.shard {
            match class {
                MsgClass::Control => {
                    on_shard_control(bytes, &mut sessions, &mut stats, &mut outbox)
                }
                MsgClass::Snapshot => {
                    on_shard_frame(bytes, &sessions, &mut stats, &mut outbox);
                }
                _ => stats.undecodable += 1,
            }
        } else if from == config.orchestrator {
            match class {
                // The orchestrator→gateway Saga class carries an InterShardFlow envelope:
                // a DirectoryReply (session-grant head) OR a Saga(TransferControl) command.
                // Decode ONCE and dispatch by variant (the dispatch split that lets them
                // coexist on one class without mis-decoding into each other).
                MsgClass::Saga => match postcard::from_bytes::<InterShardFlow>(bytes) {
                    Ok(InterShardFlow::DirectoryReply(reply)) => on_directory_reply(
                        reply,
                        &config,
                        &identity,
                        &clock,
                        &mut sessions,
                        &mut stats,
                        &mut outbox,
                    ),
                    // The TransferControl consumer (the 7-command saga vocabulary) lands in
                    // Slice 1c.2; until then a command is counted, not mis-applied.
                    Ok(InterShardFlow::Saga(_)) => stats.transfer_control_unhandled += 1,
                    Ok(_) | Err(_) => stats.undecodable += 1,
                },
                // Membership (clock sync) is consumed by the follower system.
                MsgClass::Membership => {}
                _ => stats.undecodable += 1,
            }
        } else {
            // A client connection.
            match class {
                MsgClass::Control => on_client_control(
                    bytes,
                    from,
                    &config,
                    &identity,
                    &mut sessions,
                    &mut mint,
                    &mut stats,
                    &mut outbox,
                ),
                MsgClass::Input => {
                    on_client_input(bytes, from, &sessions, &mut stats, &mut outbox);
                }
                _ => stats.undecodable += 1,
            }
        }
    }
}

fn push_control(outbox: &mut OutboundBox, to: NodeId, msg: &ServerControlMsg) {
    let bytes = postcard::to_allocvec(msg).expect("closed wire enums serialize infallibly");
    outbox
        .0
        .push((to, MsgClass::Control, vd_sim::io::bytes(bytes)));
}

fn push_to_shard(outbox: &mut OutboundBox, to: NodeId, class: MsgClass, msg: &GatewayToShard) {
    let bytes = postcard::to_allocvec(msg).expect("closed wire enums serialize infallibly");
    outbox.0.push((to, class, vd_sim::io::bytes(bytes)));
}

fn push_directory(outbox: &mut OutboundBox, to: NodeId, op: DirectoryOp) {
    let bytes = postcard::to_allocvec(&InterShardFlow::Directory(op))
        .expect("closed wire enums serialize infallibly");
    outbox
        .0
        .push((to, MsgClass::Saga, vd_sim::io::bytes(bytes)));
}

/// Handle one client control message. The gateway is the SOLE ticket validator;
/// shards never see a credential.
#[allow(clippy::too_many_arguments)]
fn on_client_control(
    bytes: &[u8],
    client: NodeId,
    config: &GatewayConfig,
    identity: &NodeIdentity,
    sessions: &mut GatewaySessions,
    mint: &mut SessionMint,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let Ok(msg) = postcard::from_bytes::<ClientControlMsg>(bytes) else {
        stats.undecodable += 1;
        return;
    };
    match msg {
        ClientControlMsg::Hello { version, login } => {
            let Some(negotiated) = ProtoVersion::negotiate(ProtoVersion::CURRENT, version) else {
                stats.version_rejected += 1;
                push_control(
                    outbox,
                    client,
                    &ServerControlMsg::Close {
                        reason: "incompatible protocol major version".to_owned(),
                    },
                );
                return;
            };
            if tickets::validate_login(&config.auth_verifying_key, &login).is_err() {
                stats.logins_rejected += 1;
                push_control(
                    outbox,
                    client,
                    &ServerControlMsg::Close {
                        reason: "login ticket rejected".to_owned(),
                    },
                );
                return;
            }
            if sessions.by_session.len() >= config.tuning.max_sessions {
                stats.sessions_refused_capacity += 1;
                push_control(
                    outbox,
                    client,
                    &ServerControlMsg::Close {
                        reason: "gateway at session capacity".to_owned(),
                    },
                );
                return;
            }
            if sessions.by_client.contains_key(&client) {
                // A duplicate Hello on a live connection: idempotent no-op (the
                // pending/active session keeps progressing).
                return;
            }
            // Propose a session id; the DIRECTORY INSERT is the authoritative mint.
            let proposed =
                SessionId((u128::from(mint.0.next_u64()) << 64) | u128::from(mint.0.next_u64()));
            let fence = Fence::GENESIS.next();
            sessions.by_session.insert(
                proposed,
                Session {
                    client,
                    account: login.account,
                    fence,
                    phase: SessionPhase::AwaitingDirectory,
                    next_sub: 0,
                    negotiated_minor: negotiated.minor,
                    hot: Arc::new(SessionHot {
                        route: ArcSwap::from_pointee(RouteSnapshot {
                            authority: config.shard,
                            fence: Fence::GENESIS,
                            cut: None,
                        }),
                        last_input_seq: AtomicU64::new(0),
                    }),
                },
            );
            sessions.by_client.insert(client, proposed);
            push_directory(
                outbox,
                config.orchestrator,
                DirectoryOp::LeaseGrant {
                    key: DirectoryKey::Session(proposed),
                    owner: AuthorityRef::Gateway(identity.node_id),
                    fence,
                },
            );
        }
        ClientControlMsg::Resume { .. } => {
            // Resume/adoption lands in P3; refusing is honest, not silent.
            stats.resumes_refused += 1;
            push_control(
                outbox,
                client,
                &ServerControlMsg::Close {
                    reason: "resume is not available yet".to_owned(),
                },
            );
        }
        ClientControlMsg::Bye => {
            let Some(session_id) = sessions.by_client.remove(&client) else {
                return;
            };
            let session = sessions
                .by_session
                .remove(&session_id)
                .expect("session maps are kept in sync");
            push_to_shard(
                outbox,
                config.shard,
                MsgClass::Control,
                &GatewayToShard::DetachSession {
                    session: session_id,
                    fence: session.fence,
                },
            );
            push_directory(
                outbox,
                config.orchestrator,
                DirectoryOp::LeaseRevoke {
                    key: DirectoryKey::Session(session_id),
                    fence: session.fence,
                },
            );
        }
        // No transfers in P1: a cut confirmation has nothing to bind to.
        // Pongs are liveness echoes; the P1 gateway sends no pings.
        ClientControlMsg::CutEmitted { .. } | ClientControlMsg::Pong { .. } => {}
    }
}

/// Route one client input datagram (hot path + bookkeeping).
fn on_client_input(
    bytes: &[u8],
    client: NodeId,
    sessions: &GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let Some(session_id) = sessions.by_client.get(&client) else {
        stats.inputs_unroutable += 1;
        return;
    };
    // Guarded lookup, never `[]`: the two session maps are kept in sync by construction,
    // but a desync must DROP-and-count on this 20Hz hot path, never panic the gateway
    // (R2 lineage — a routing-map slip is a counted unroutable, not a crash).
    let Some(session) = sessions.by_session.get(session_id) else {
        stats.inputs_unroutable += 1;
        return;
    };
    if !matches!(session.phase, SessionPhase::Active { .. }) {
        stats.inputs_unroutable += 1;
        return;
    }
    match route_input(&session.hot, bytes) {
        InputRouting::Forward { to } => {
            push_to_shard(
                outbox,
                to,
                MsgClass::Input,
                &GatewayToShard::SessionInput {
                    session: *session_id,
                    fence: session.fence,
                    input_bytes: bytes.to_vec(),
                },
            );
        }
        InputRouting::Deduped => stats.inputs_deduped += 1,
        InputRouting::Malformed => stats.inputs_malformed += 1,
    }
}

/// Handle a shard control reply (attach/detach lifecycle).
fn on_shard_control(
    bytes: &[u8],
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let Ok(msg) = postcard::from_bytes::<ShardToGateway>(bytes) else {
        stats.undecodable += 1;
        return;
    };
    match msg {
        ShardToGateway::SessionAttached {
            session: session_id,
            entity,
            frame,
            realm_fence,
        } => {
            let Some(session) = sessions.by_session.get_mut(&session_id) else {
                // Attach reply for a session that left meanwhile: ignore (the
                // detach path already ran).
                return;
            };
            if matches!(session.phase, SessionPhase::Active { .. }) {
                return; // duplicate attach reply (at-least-once): idempotent
            }
            // sub ids come from the per-session monotonic allocator — NEVER reused.
            let sub = SubId(session.next_sub);
            session.next_sub += 1;
            // THE route mutation primitive: one atomic store (P2 drives this same
            // call from CommitAuthority).
            let authority = session.hot.route.load().authority;
            session.hot.route.store(Arc::new(RouteSnapshot {
                authority,
                fence: realm_fence,
                cut: None,
            }));
            session.phase = SessionPhase::Active { sub, entity };
            // X1: SubscriptionOpened strictly precedes any data for the sub.
            push_control(
                outbox,
                session.client,
                &ServerControlMsg::SubscriptionOpened { sub, frame },
            );
            push_control(
                outbox,
                session.client,
                &ServerControlMsg::AuthorityChanged { entity, sub },
            );
        }
        ShardToGateway::SessionDetached { .. } => {
            // The session was already removed on Bye; the confirmation closes the loop.
        }
        ShardToGateway::Frame { .. } => {
            // Frames ride the Snapshot class; one on Control is a peer bug.
            stats.undecodable += 1;
        }
    }
}

/// Fan one shard frame out to every ACTIVE session (fence-checked, re-tagged).
fn on_shard_frame(
    bytes: &[u8],
    sessions: &GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let Ok(ShardToGateway::Frame {
        realm_fence,
        source_tick: _,
        snapshot_bytes,
    }) = postcard::from_bytes::<ShardToGateway>(bytes)
    else {
        stats.undecodable += 1;
        return;
    };
    // SCALE-1: re-tag the snapshot body ONCE per distinct sub-id into a SHARED
    // `Arc`; the per-session fan-out is then a cheap fence check + refcount bump,
    // never an O(entities) re-allocation per subscriber. The gateway therefore
    // holds ONE body per sub-id regardless of how many sessions subscribe to it.
    let mut retagged: BTreeMap<SubId, vd_sim::io::Bytes> = BTreeMap::new();
    for session in sessions.by_session.values() {
        let SessionPhase::Active { sub, .. } = session.phase else {
            continue;
        };
        if !frame_passes_fence(&session.hot, realm_fence) {
            stats.stale_frames_dropped += 1;
            continue;
        }
        let body = match retagged.entry(sub) {
            std::collections::btree_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::btree_map::Entry::Vacant(e) => {
                match retag_snapshot_sub(&snapshot_bytes, sub) {
                    Ok(b) => e.insert(vd_sim::io::bytes(b)),
                    Err(_) => {
                        // A corrupt snapshot body re-tags for no sub: count once and
                        // abandon the whole frame (every sub would fail identically).
                        stats.undecodable += 1;
                        return;
                    }
                }
            }
        };
        outbox
            .0
            .push((session.client, MsgClass::Snapshot, body.clone()));
    }
}

/// Handle a directory reply: the Session-key head confirms (or denies) the mint.
fn on_directory_reply(
    reply: DirectoryReply,
    config: &GatewayConfig,
    identity: &NodeIdentity,
    clock: &ClockSample,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let DirectoryReply::Head {
        key: DirectoryKey::Session(session_id),
        record,
    } = reply
    else {
        return; // entity/realm heads carry no gateway obligation in P1
    };
    let Some(session) = sessions.by_session.get_mut(&session_id) else {
        return; // session left while the reply was in flight
    };
    if !matches!(session.phase, SessionPhase::AwaitingDirectory) {
        return; // duplicate head (at-least-once): already progressed
    }
    let granted = record.is_some_and(|r| {
        r.authority == AuthorityRef::Gateway(identity.node_id) && r.fence == session.fence
    });
    if granted {
        // The mint is committed. Welcome the client; attach to the shard.
        session.phase = SessionPhase::AwaitingAttach;
        push_control(
            outbox,
            session.client,
            &ServerControlMsg::Welcome {
                version: ProtoVersion::CURRENT,
                session: session_id,
                session_fence: session.fence,
                epoch: clock.epoch,
            },
        );
        // Relay the cluster tick rate to a minor-1+ client (sender-gates-variants),
        // so it drives its render cursor at the server's rate, not a guess.
        if session.negotiated_minor >= 1 {
            push_control(
                outbox,
                session.client,
                &ServerControlMsg::UniverseRate {
                    tick_hz: config.tick_hz,
                },
            );
        }
        push_to_shard(
            outbox,
            config.shard,
            MsgClass::Control,
            &GatewayToShard::AttachSession {
                session: session_id,
                fence: session.fence,
                account: session.account,
            },
        );
    } else {
        // Mint refused (id collision or foreign holder): close loudly; the client
        // retries login with a fresh Hello.
        stats.session_mints_refused += 1;
        let client = session.client;
        sessions.by_session.remove(&session_id);
        sessions.by_client.remove(&client);
        push_control(
            outbox,
            client,
            &ServerControlMsg::Close {
                reason: "session mint refused by the directory".to_owned(),
            },
        );
    }
}

/// Per-tick retry driver: pending directory grants and shard attaches are
/// re-sent until answered (all idempotent — at-least-once over a lossy fabric).
fn drive_pending_sessions(
    config: Res<GatewayConfig>,
    identity: Res<NodeIdentity>,
    sessions: Res<GatewaySessions>,
    mut outbox: ResMut<OutboundBox>,
) {
    for (session_id, session) in &sessions.by_session {
        match session.phase {
            SessionPhase::AwaitingDirectory => {
                push_directory(
                    &mut outbox,
                    config.orchestrator,
                    DirectoryOp::LeaseGrant {
                        key: DirectoryKey::Session(*session_id),
                        owner: AuthorityRef::Gateway(identity.node_id),
                        fence: session.fence,
                    },
                );
            }
            SessionPhase::AwaitingAttach => {
                push_to_shard(
                    &mut outbox,
                    config.shard,
                    MsgClass::Control,
                    &GatewayToShard::AttachSession {
                        session: *session_id,
                        fence: session.fence,
                        account: session.account,
                    },
                );
            }
            SessionPhase::Active { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use vd_core::pose::FrameRef;
    use vd_core::{EpochId, TickId, UniverseTick};
    use vd_sim::capability::NodeKind;
    use vd_wire::channels::{EntitySnap, InputDatagram, SnapshotDatagram};
    use vd_wire::seams::directory::OwnerRecord;

    const GW: NodeId = NodeId(1);
    const SHARD: NodeId = NodeId(2);
    const ORCH: NodeId = NodeId(3);
    const CLIENT: NodeId = NodeId(100);
    const SIGNING_KEY: [u8; 32] = [0x42; 32];

    fn verifying_key() -> [u8; 32] {
        SigningKey::from_bytes(&SIGNING_KEY)
            .verifying_key()
            .to_bytes()
    }

    fn config() -> GatewayConfig {
        GatewayConfig {
            orchestrator: ORCH,
            shard: SHARD,
            auth_verifying_key: verifying_key(),
            session_seed: 7,
            tick_hz: 50,
            tuning: TransportTuning { max_sessions: 4 },
        }
    }

    /// The per-tick sends captured across a login drive (one Vec of `(to, class,
    /// bytes)` per tick).
    type LoginSends = Vec<Vec<(NodeId, MsgClass, Vec<u8>)>>;

    struct Rig {
        world: World,
        schedule: Schedule,
    }

    impl Rig {
        fn new() -> Rig {
            let mut world = World::new();
            world.insert_resource(InboundBox::default());
            world.insert_resource(OutboundBox::default());
            world.insert_resource(NodeIdentity {
                node_id: GW,
                kind: NodeKind::Gateway,
            });
            world.insert_resource(ClockSample {
                local_tick: TickId(1),
                universe_tick: UniverseTick(50),
                epoch: EpochId(9),
            });
            let mut schedule = Schedule::default();
            register_gateway(&mut world, &mut schedule, config());
            Rig { world, schedule }
        }

        fn tick(&mut self, inbound: Vec<Inbound>) -> Vec<(NodeId, MsgClass, Vec<u8>)> {
            self.world.resource_mut::<InboundBox>().0 = inbound;
            self.schedule.run(&mut self.world);
            std::mem::take(&mut self.world.resource_mut::<OutboundBox>().0)
                .into_iter()
                .map(|(to, class, bytes)| (to, class, bytes.to_vec()))
                .collect()
        }

        fn stats(&self) -> GatewayStats {
            *self.world.resource::<GatewayStats>()
        }

        /// Hello → directory grant → attach reply; returns (session_id, all sends).
        #[allow(clippy::type_complexity)] // test helper: ticks of raw sends
        fn login(&mut self) -> (SessionId, LoginSends) {
            self.login_with(&hello_msg())
        }

        fn login_with(&mut self, hello: &ClientControlMsg) -> (SessionId, LoginSends) {
            let hello = self.tick(vec![wire(CLIENT, MsgClass::Control, hello)]);
            let session_id = *self
                .world
                .resource::<GatewaySessions>()
                .sessions()
                .collect::<Vec<_>>()
                .first()
                .expect("session pending");
            let granted = self.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(session_id))]);
            let attached = self.tick(vec![wire(
                SHARD,
                MsgClass::Control,
                &ShardToGateway::SessionAttached {
                    session: session_id,
                    entity: EntityId(77),
                    frame: FrameRef::SystemSpace { system_seed: 7 },
                    realm_fence: Fence(1),
                },
            )]);
            (session_id, vec![hello, granted, attached])
        }
    }

    fn wire<T: serde::Serialize>(from: NodeId, class: MsgClass, msg: &T) -> Inbound {
        Inbound::Wire {
            from,
            class,
            bytes: postcard::to_allocvec(msg).expect("encode").into(),
        }
    }

    fn hello_msg() -> ClientControlMsg {
        ClientControlMsg::Hello {
            version: ProtoVersion::CURRENT,
            login: tickets::mint_login(&SIGNING_KEY, AccountId(5), EpochId(9), 1),
        }
    }

    fn hello_msg_minor0() -> ClientControlMsg {
        ClientControlMsg::Hello {
            version: ProtoVersion { major: 1, minor: 0 },
            login: tickets::mint_login(&SIGNING_KEY, AccountId(5), EpochId(9), 1),
        }
    }

    #[test]
    fn a_minor_0_client_is_welcomed_without_the_universe_rate_variant() {
        // Sender-gates-variants: a peer that negotiated minor 0 must NOT be sent the
        // minor-1 UniverseRate (it would desync an old decoder). Welcome only.
        let mut rig = Rig::new();
        let (session_id, sends) = rig.login_with(&hello_msg_minor0());
        let welcomes = decode_controls(&sends[1], CLIENT);
        assert_eq!(
            welcomes,
            vec![ServerControlMsg::Welcome {
                version: ProtoVersion::CURRENT,
                session: session_id,
                session_fence: Fence(1),
                epoch: EpochId(9),
            }],
            "a minor-0 client gets Welcome but NOT UniverseRate"
        );
    }

    /// A session-grant head reply as the orchestrator now sends it — wrapped in the
    /// InterShardFlow::DirectoryReply envelope (the dispatch split).
    fn granted_head(session: SessionId) -> InterShardFlow {
        InterShardFlow::DirectoryReply(DirectoryReply::Head {
            key: DirectoryKey::Session(session),
            record: Some(OwnerRecord {
                authority: AuthorityRef::Gateway(GW),
                fence: Fence(1),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        })
    }

    fn decode_controls(sent: &[(NodeId, MsgClass, Vec<u8>)], to: NodeId) -> Vec<ServerControlMsg> {
        sent.iter()
            .filter(|(node, class, _)| (*node == to) & (*class == MsgClass::Control))
            .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("decode"))
            .collect()
    }

    fn input_bytes(seq: u64) -> Vec<u8> {
        postcard::to_allocvec(&InputDatagram {
            seq,
            is_cut_marker: false,
            client_tick: TickId(1),
            movement: [1.0, 0.0, 0.0],
            look: [0.0, 0.0],
            action_bits: 0,
        })
        .expect("encode")
    }

    fn frame_msg(fence: Fence, frame_id: u64) -> ShardToGateway {
        let snapshot = SnapshotDatagram {
            sub: SubId(0),
            frame_id,
            source_tick: TickId(5),
            universe_tick: UniverseTick(50),
            entities: vec![EntitySnap {
                entity: EntityId(77),
                pose: vd_core::pose::StampedPose::at_rest(
                    FrameRef::SystemSpace { system_seed: 7 },
                    vd_core::glam::DVec3::ZERO,
                    UniverseTick(50),
                ),
            }],
        };
        ShardToGateway::Frame {
            realm_fence: fence,
            source_tick: TickId(5),
            snapshot_bytes: postcard::to_allocvec(&snapshot).expect("encode"),
        }
    }

    #[test]
    fn the_full_login_flow_reaches_active_with_welcome_then_subscription() {
        let mut rig = Rig::new();
        let (session_id, sends) = rig.login();

        // Hello tick: exactly one directory grant went out (plus the retry driver's
        // duplicate — idempotent by fence).
        let to_orch: Vec<NodeId> = sends[0].iter().map(|(to, _, _)| *to).collect();
        assert!(
            to_orch.iter().all(|to| *to == ORCH),
            "only directory traffic"
        );

        // Grant tick: Welcome to the client (with the directory-committed id,
        // fence, epoch) and an attach toward the shard.
        // Welcome, then UniverseRate (the client negotiated minor 1, so the gateway
        // relays the cluster tick rate right after the Welcome).
        let welcomes = decode_controls(&sends[1], CLIENT);
        assert_eq!(
            welcomes,
            vec![
                ServerControlMsg::Welcome {
                    version: ProtoVersion::CURRENT,
                    session: session_id,
                    session_fence: Fence(1),
                    epoch: EpochId(9),
                },
                ServerControlMsg::UniverseRate { tick_hz: 50 },
            ]
        );
        // Attach tick: SubscriptionOpened STRICTLY BEFORE AuthorityChanged (X1),
        // sub allocated from the monotonic allocator.
        let controls = decode_controls(&sends[2], CLIENT);
        assert_eq!(
            controls,
            vec![
                ServerControlMsg::SubscriptionOpened {
                    sub: SubId(0),
                    frame: FrameRef::SystemSpace { system_seed: 7 },
                },
                ServerControlMsg::AuthorityChanged {
                    entity: EntityId(77),
                    sub: SubId(0),
                },
            ]
        );
        assert_eq!(
            rig.stats(),
            GatewayStats::default(),
            "clean run, zero rejects"
        );
    }

    #[test]
    fn entity_of_tracks_the_session_lifecycle() {
        let mut rig = Rig::new();
        let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        let session_id = rig
            .world
            .resource::<GatewaySessions>()
            .sessions()
            .next()
            .expect("pending");
        // Pending phases expose no entity; unknown sessions expose none either.
        assert_eq!(
            rig.world
                .resource::<GatewaySessions>()
                .entity_of(session_id),
            None
        );
        assert_eq!(
            rig.world
                .resource::<GatewaySessions>()
                .entity_of(SessionId(0xDEAD)),
            None
        );
        let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(session_id))]);
        assert_eq!(
            rig.world
                .resource::<GatewaySessions>()
                .entity_of(session_id),
            None,
            "still awaiting attach"
        );
        let _ = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::SessionAttached {
                session: session_id,
                entity: EntityId(77),
                frame: FrameRef::SystemSpace { system_seed: 7 },
                realm_fence: Fence(1),
            },
        )]);
        assert_eq!(
            rig.world
                .resource::<GatewaySessions>()
                .entity_of(session_id),
            Some(EntityId(77)),
            "active sessions expose their avatar (P2's saga key)"
        );
    }

    #[test]
    fn version_mismatch_and_bad_tickets_close_with_reasons() {
        let mut rig = Rig::new();
        let wrong_version = ClientControlMsg::Hello {
            version: ProtoVersion {
                major: ProtoVersion::CURRENT.major + 1,
                minor: 0,
            },
            login: tickets::mint_login(&SIGNING_KEY, AccountId(5), EpochId(9), 1),
        };
        let sent = rig.tick(vec![wire(CLIENT, MsgClass::Control, &wrong_version)]);
        let controls = decode_controls(&sent, CLIENT);
        assert_eq!(controls.len(), 1);
        assert_eq!(
            controls[0],
            ServerControlMsg::Close {
                reason: "incompatible protocol major version".to_owned()
            }
        );
        // A foreign signer's ticket is rejected by the SOLE validator.
        let forged = ClientControlMsg::Hello {
            version: ProtoVersion::CURRENT,
            login: tickets::mint_login(&[0x66; 32], AccountId(5), EpochId(9), 1),
        };
        let sent = rig.tick(vec![wire(CLIENT, MsgClass::Control, &forged)]);
        let controls = decode_controls(&sent, CLIENT);
        assert_eq!(
            controls[0],
            ServerControlMsg::Close {
                reason: "login ticket rejected".to_owned()
            }
        );
        assert_eq!(rig.stats().version_rejected, 1);
        assert_eq!(rig.stats().logins_rejected, 1);
        assert!(rig.world.resource::<GatewaySessions>().is_empty());
    }

    #[test]
    fn capacity_duplicate_hello_and_resume_paths() {
        let mut rig = Rig::new();
        // Fill to capacity with distinct clients.
        for n in 0..4u64 {
            let _ = rig.tick(vec![wire(NodeId(100 + n), MsgClass::Control, &hello_msg())]);
        }
        assert_eq!(rig.world.resource::<GatewaySessions>().len(), 4);
        // One more is refused loudly.
        let sent = rig.tick(vec![wire(NodeId(199), MsgClass::Control, &hello_msg())]);
        assert_eq!(
            decode_controls(&sent, NodeId(199))[0],
            ServerControlMsg::Close {
                reason: "gateway at session capacity".to_owned()
            }
        );
        assert_eq!(rig.stats().sessions_refused_capacity, 1);
        // Resume is refused honestly until P3.
        let resume = ClientControlMsg::Resume {
            version: ProtoVersion::CURRENT,
            ticket: dummy_resume(),
        };
        let sent = rig.tick(vec![wire(NodeId(198), MsgClass::Control, &resume)]);
        assert_eq!(
            decode_controls(&sent, NodeId(198))[0],
            ServerControlMsg::Close {
                reason: "resume is not available yet".to_owned()
            }
        );
        assert_eq!(rig.stats().resumes_refused, 1);
    }

    fn dummy_resume() -> vd_wire::seams::tickets::ResumeTicket {
        vd_wire::seams::tickets::ResumeTicket {
            claims: vd_wire::seams::tickets::SessionClaims {
                session: SessionId(1),
                account: AccountId(1),
                epoch: EpochId(1),
                validity_epoch: 0,
                expires: UniverseTick(0),
                key_id: 0,
            },
            session_fence: Fence(0),
            resume_nonce: 0,
            hmac: [0; 32],
        }
    }

    #[test]
    fn mint_refusal_closes_the_client_and_clears_the_session() {
        let mut rig = Rig::new();
        let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        let session_id = rig
            .world
            .resource::<GatewaySessions>()
            .sessions()
            .next()
            .expect("pending session");
        // The directory says someone ELSE holds the session key.
        let refused = InterShardFlow::DirectoryReply(DirectoryReply::Head {
            key: DirectoryKey::Session(session_id),
            record: Some(OwnerRecord {
                authority: AuthorityRef::Gateway(NodeId(55)),
                fence: Fence(3),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        });
        let sent = rig.tick(vec![wire(ORCH, MsgClass::Saga, &refused)]);
        assert_eq!(
            decode_controls(&sent, CLIENT)[0],
            ServerControlMsg::Close {
                reason: "session mint refused by the directory".to_owned()
            }
        );
        assert!(rig.world.resource::<GatewaySessions>().is_empty());
        assert_eq!(rig.stats().session_mints_refused, 1);
    }

    #[test]
    fn input_routes_dedups_and_counts_every_failure_mode() {
        let mut rig = Rig::new();
        let (session_id, _) = rig.login();

        // A fresh input forwards to the shard wrapped with (session, fence).
        let sent = rig.tick(vec![Inbound::Wire {
            from: CLIENT,
            class: MsgClass::Input,
            bytes: input_bytes(1).into(),
        }]);
        let inputs: Vec<&(NodeId, MsgClass, Vec<u8>)> = sent
            .iter()
            .filter(|(to, class, _)| (*to == SHARD) & (*class == MsgClass::Input))
            .collect();
        assert_eq!(inputs.len(), 1);
        let fwd: GatewayToShard = postcard::from_bytes(&inputs[0].2).expect("decode");
        assert_eq!(
            fwd,
            GatewayToShard::SessionInput {
                session: session_id,
                fence: Fence(1),
                input_bytes: input_bytes(1),
            }
        );

        // Same seq again: deduped. Garbage: malformed. Unknown client: unroutable.
        let _ = rig.tick(vec![
            Inbound::Wire {
                from: CLIENT,
                class: MsgClass::Input,
                bytes: input_bytes(1).into(),
            },
            Inbound::Wire {
                from: CLIENT,
                class: MsgClass::Input,
                bytes: vec![0x80].into(),
            },
            Inbound::Wire {
                from: NodeId(177),
                class: MsgClass::Input,
                bytes: input_bytes(2).into(),
            },
        ]);
        let stats = rig.stats();
        assert_eq!(stats.inputs_deduped, 1);
        assert_eq!(stats.inputs_malformed, 1);
        assert_eq!(stats.inputs_unroutable, 1);
    }

    #[test]
    fn input_for_a_desynced_session_map_is_dropped_and_counted_never_panics() {
        // WB-1: `by_client` and `by_session` are kept in sync by construction, but a slip
        // must DROP-and-count on the 20Hz input path, never panic the gateway. Proven by
        // an artificially desynced table (present in `by_client`, absent from `by_session`).
        let mut sessions = GatewaySessions::default();
        sessions.by_client.insert(CLIENT, SessionId(7));
        let mut stats = GatewayStats::default();
        let mut outbox = OutboundBox::default();
        on_client_input(&input_bytes(1), CLIENT, &sessions, &mut stats, &mut outbox);
        assert_eq!(
            stats.inputs_unroutable, 1,
            "the desync is counted, not crashed"
        );
        assert!(
            outbox.0.is_empty(),
            "nothing forwarded for a desynced session"
        );
    }

    #[test]
    fn duplicate_hello_below_capacity_is_an_idempotent_noop() {
        let mut rig = Rig::new();
        let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        assert_eq!(rig.world.resource::<GatewaySessions>().len(), 1);
        let sent = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        assert_eq!(
            rig.world.resource::<GatewaySessions>().len(),
            1,
            "no second session"
        );
        // Only the pending-grant retry went out — no Close, no new mint.
        assert!(sent.iter().all(|(to, _, _)| *to == ORCH));
        assert_eq!(rig.stats(), GatewayStats::default());
    }

    #[test]
    fn input_before_active_is_unroutable() {
        let mut rig = Rig::new();
        let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        let _ = rig.tick(vec![Inbound::Wire {
            from: CLIENT,
            class: MsgClass::Input,
            bytes: input_bytes(1).into(),
        }]);
        assert_eq!(rig.stats().inputs_unroutable, 1);
    }

    #[test]
    fn frames_fan_out_retagged_and_stale_fences_drop() {
        let mut rig = Rig::new();
        let (_, _) = rig.login();

        // A current-fence frame reaches the client re-tagged to ITS sub.
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Snapshot,
            &frame_msg(Fence(1), 9),
        )]);
        let snaps: Vec<&(NodeId, MsgClass, Vec<u8>)> = sent
            .iter()
            .filter(|(to, class, _)| (*to == CLIENT) & (*class == MsgClass::Snapshot))
            .collect();
        assert_eq!(snaps.len(), 1);
        let snap: SnapshotDatagram = postcard::from_bytes(&snaps[0].2).expect("decode");
        assert_eq!(snap.sub, SubId(0));
        assert_eq!(snap.frame_id, 9);

        // A HIGHER fence is current (not stale) — still forwarded.
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Snapshot,
            &frame_msg(Fence(2), 10),
        )]);
        assert_eq!(sent.len(), 1);

        // A stale fence is dropped and counted (the P2 old-owner guard).
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Snapshot,
            &frame_msg(Fence::GENESIS, 11),
        )]);
        assert_eq!(sent.len(), 0);
        assert_eq!(rig.stats().stale_frames_dropped, 1);
    }

    #[test]
    fn two_active_sessions_share_one_retagged_body() {
        // SCALE-1: a second session with the SAME sub re-uses the ONE retagged body
        // (the Occupied map arm) — the gateway never re-encodes per subscriber.
        let mut rig = Rig::new();
        let (_, _) = rig.login();
        // A second client logs in fully (distinct session, same sub 0).
        let before: std::collections::BTreeSet<SessionId> =
            rig.world.resource::<GatewaySessions>().sessions().collect();
        let _ = rig.tick(vec![Inbound::Wire {
            from: NodeId(101),
            class: MsgClass::Control,
            bytes: postcard::to_allocvec(&hello_msg()).expect("encode").into(),
        }]);
        let session2 = rig
            .world
            .resource::<GatewaySessions>()
            .sessions()
            .find(|s| !before.contains(s))
            .expect("second session pending");
        let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(session2))]);
        let _ = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::SessionAttached {
                session: session2,
                entity: EntityId(88),
                frame: FrameRef::SystemSpace { system_seed: 7 },
                realm_fence: Fence(1),
            },
        )]);

        // One frame: BOTH clients receive a sub-0 snapshot from the shared body.
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Snapshot,
            &frame_msg(Fence(1), 9),
        )]);
        let mut recipients: Vec<NodeId> = sent
            .iter()
            .filter(|(_, class, _)| *class == MsgClass::Snapshot)
            .map(|(to, _, _)| *to)
            .collect();
        recipients.sort_unstable();
        assert_eq!(
            recipients,
            vec![CLIENT, NodeId(101)],
            "both subscribers fed"
        );
        for (_, _, bytes) in sent.iter().filter(|(_, c, _)| *c == MsgClass::Snapshot) {
            let snap: SnapshotDatagram = postcard::from_bytes(bytes).expect("decode");
            assert_eq!(snap.sub, SubId(0));
        }
        assert_eq!(rig.stats().undecodable, 0);
    }

    #[test]
    fn a_corrupt_snapshot_body_is_counted_once_and_abandons_the_frame() {
        // The Err arm of the per-sub retag: a body that can't re-tag (bad varint)
        // would fail identically for every sub, so the whole frame is abandoned.
        let mut rig = Rig::new();
        let (_, _) = rig.login();
        let corrupt = ShardToGateway::Frame {
            realm_fence: Fence(1),
            source_tick: TickId(5),
            snapshot_bytes: vec![0x80], // truncated varint: retag fails
        };
        let sent = rig.tick(vec![wire(SHARD, MsgClass::Snapshot, &corrupt)]);
        // The corrupt body re-tags for no sub, so the whole frame is abandoned: the
        // active session receives NOTHING and the failure is counted exactly once.
        assert_eq!(sent.len(), 0, "no output from a corrupt body");
        assert_eq!(rig.stats().undecodable, 1, "counted exactly once");
    }

    #[test]
    fn the_orchestrator_saga_class_dispatch_splits_reply_from_command_from_garbage() {
        use vd_wire::seams::transfer_control::TransferControl;
        let mut rig = Rig::new();

        // A TransferControl command (the saga driving the gateway) is COUNTED as
        // not-yet-handled — never mis-decoded as a directory reply (1c.2 wires the consumer).
        let cmd = InterShardFlow::Saga(TransferControl::PrepareSubscribe {
            transfer: vd_core::TransferId(1),
            session: SessionId(7),
            dest: SHARD,
        });
        let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &cmd)]);
        assert_eq!(rig.stats().transfer_control_unhandled, 1);
        assert_eq!(rig.stats().undecodable, 0, "a command is NOT undecodable");

        // A non-reply / non-command Saga-class arm (a misdirected Ghost) → undecodable.
        let ghost = InterShardFlow::Ghost(vd_wire::intershard::GhostFlow::Despawn {
            entity: EntityId::pack(vd_core::entity_kind::EntityKind::Player, 1, 7, 3),
            source_fence: Fence(1),
        });
        let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &ghost)]);
        assert_eq!(
            rig.stats().undecodable,
            1,
            "a non-dispatchable arm is undecodable"
        );

        // Raw garbage on the orchestrator Saga path → also undecodable (the Err arm).
        let _ = rig.tick(vec![Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes: vec![0xFF, 0xFF].into(),
        }]);
        assert_eq!(rig.stats().undecodable, 2);
        // The command count did not move (the split is clean in both directions).
        assert_eq!(rig.stats().transfer_control_unhandled, 1);

        // A well-formed reply that is NOT a Session head (a CAS outcome carries no gateway
        // obligation) decodes + dispatches to on_directory_reply, which returns without
        // effect — it is NOT undecodable (valid arm), just no-op for the gateway.
        let cas = InterShardFlow::DirectoryReply(DirectoryReply::CasResult {
            key: DirectoryKey::Session(SessionId(7)),
            outcome: vd_wire::seams::directory::CasOutcome::Won {
                new_fence: Fence(2),
            },
        });
        let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &cas)]);
        assert_eq!(
            rig.stats().undecodable,
            2,
            "a valid non-Head reply is not undecodable"
        );
    }

    #[test]
    fn frames_skip_sessions_that_are_not_active_yet() {
        let mut rig = Rig::new();
        let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Snapshot,
            &frame_msg(Fence(1), 1),
        )]);
        let to_client = sent.iter().filter(|(to, _, _)| *to == CLIENT).count();
        assert_eq!(to_client, 0, "pending sessions receive nothing");
    }

    #[test]
    fn bye_detaches_revokes_and_clears() {
        let mut rig = Rig::new();
        let (session_id, _) = rig.login();
        let sent = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        assert!(rig.world.resource::<GatewaySessions>().is_empty());
        let detach: GatewayToShard = postcard::from_bytes(
            &sent
                .iter()
                .find(|(to, _, _)| *to == SHARD)
                .expect("detach sent")
                .2,
        )
        .expect("decode");
        assert_eq!(
            detach,
            GatewayToShard::DetachSession {
                session: session_id,
                fence: Fence(1),
            }
        );
        let revoke: InterShardFlow = postcard::from_bytes(
            &sent
                .iter()
                .find(|(to, _, _)| *to == ORCH)
                .expect("revoke sent")
                .2,
        )
        .expect("decode");
        assert_eq!(
            revoke,
            InterShardFlow::Directory(DirectoryOp::LeaseRevoke {
                key: DirectoryKey::Session(session_id),
                fence: Fence(1),
            })
        );
        // Bye from a connection with no session is a no-op.
        let sent = rig.tick(vec![wire(
            NodeId(177),
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        assert_eq!(sent.len(), 0);
        // The shard detach confirmation closes the loop silently.
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::SessionDetached {
                session: session_id,
            },
        )]);
        assert_eq!(sent.len(), 0);
    }

    #[test]
    fn pending_phases_retry_every_tick_until_answered() {
        let mut rig = Rig::new();
        let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        // AwaitingDirectory: a grant retry goes out on an idle tick.
        let sent = rig.tick(vec![]);
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].0, ORCH);
        let session_id = rig
            .world
            .resource::<GatewaySessions>()
            .sessions()
            .next()
            .expect("pending");
        let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(session_id))]);
        // AwaitingAttach: an attach retry goes out on an idle tick.
        let sent = rig.tick(vec![]);
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].0, SHARD);
        let retry: GatewayToShard = postcard::from_bytes(&sent[0].2).expect("decode");
        assert_eq!(
            retry,
            GatewayToShard::AttachSession {
                session: session_id,
                fence: Fence(1),
                account: AccountId(5),
            }
        );
    }

    #[test]
    fn duplicate_and_late_replies_are_idempotent() {
        let mut rig = Rig::new();
        let (session_id, _) = rig.login();
        // A duplicate directory head after activation: no-op.
        let sent = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(session_id))]);
        assert_eq!(sent.len(), 0);
        // A duplicate attach reply: no second SubscriptionOpened.
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::SessionAttached {
                session: session_id,
                entity: EntityId(77),
                frame: FrameRef::SystemSpace { system_seed: 7 },
                realm_fence: Fence(1),
            },
        )]);
        assert_eq!(decode_controls(&sent, CLIENT).len(), 0);
        // Replies for a session that's gone: ignored.
        let _ = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        let sent = rig.tick(vec![
            wire(ORCH, MsgClass::Saga, &granted_head(session_id)),
            wire(
                SHARD,
                MsgClass::Control,
                &ShardToGateway::SessionAttached {
                    session: session_id,
                    entity: EntityId(77),
                    frame: FrameRef::SystemSpace { system_seed: 7 },
                    realm_fence: Fence(1),
                },
            ),
        ]);
        assert_eq!(sent.len(), 0);
    }

    #[test]
    fn garbage_wrong_classes_and_notices_are_counted_or_skipped() {
        let mut rig = Rig::new();
        let _ = rig.tick(vec![
            // Undecodable from every peer family.
            Inbound::Wire {
                from: CLIENT,
                class: MsgClass::Control,
                bytes: vec![0xFF].into(),
            },
            Inbound::Wire {
                from: SHARD,
                class: MsgClass::Control,
                bytes: vec![0xFF].into(),
            },
            Inbound::Wire {
                from: SHARD,
                class: MsgClass::Snapshot,
                bytes: vec![0xFF].into(),
            },
            Inbound::Wire {
                from: ORCH,
                class: MsgClass::Saga,
                bytes: vec![0xFF].into(),
            },
            // Wrong classes.
            Inbound::Wire {
                from: SHARD,
                class: MsgClass::Saga,
                bytes: vec![1].into(),
            },
            Inbound::Wire {
                from: ORCH,
                class: MsgClass::Control,
                bytes: vec![1].into(),
            },
            Inbound::Wire {
                from: CLIENT,
                class: MsgClass::Snapshot,
                bytes: vec![1].into(),
            },
            // Clock sync is the follower system's business: skipped here.
            Inbound::Wire {
                from: ORCH,
                class: MsgClass::Membership,
                bytes: vec![1].into(),
            },
            // Transport notices are skipped by the dispatcher.
            Inbound::NodeUnreachable {
                to: SHARD,
                class: MsgClass::Input,
                undelivered: vd_core::MsgId(0),
            },
        ]);
        assert_eq!(rig.stats().undecodable, 7);
        // CutEmitted/Pong are accepted no-ops (nothing to bind to in P1).
        let _ = rig.tick(vec![
            wire(
                CLIENT,
                MsgClass::Control,
                &ClientControlMsg::CutEmitted {
                    transfer: vd_core::TransferId(1),
                    marker_seq: 5,
                },
            ),
            wire(
                CLIENT,
                MsgClass::Control,
                &ClientControlMsg::Pong { nonce: 2 },
            ),
        ]);
        // Entity heads carry no gateway obligation.
        let _ = rig.tick(vec![wire(
            ORCH,
            MsgClass::Saga,
            &DirectoryReply::Head {
                key: DirectoryKey::Entity(EntityId(9)),
                record: None,
            },
        )]);
        // A Frame on the Control class is a peer bug, counted.
        let before = rig.stats().undecodable;
        let _ = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &frame_msg(Fence(1), 1),
        )]);
        assert_eq!(rig.stats().undecodable, before + 1);
    }

    /// SPIKE-2a: the 20 Hz hot path is lock-free by construction (ArcSwap load +
    /// one atomic + a byte splice) and fast enough that 50k route+retag rounds
    /// finish far inside any tick budget even in a debug build. The bound is a
    /// generous PROPERTY gate (catches contention collapse / accidental O(n²)),
    /// not a microbenchmark number.
    #[test]
    // The seam ban targets PRODUCTION reaching for wall-clock; a latency microbench
    // measuring elapsed time is exactly what Instant is for (justified exemption).
    #[allow(clippy::disallowed_methods)]
    fn spike_2a_hot_path_volume_bound() {
        let hot = SessionHot {
            route: ArcSwap::from_pointee(RouteSnapshot {
                authority: SHARD,
                fence: Fence(1),
                cut: None,
            }),
            last_input_seq: AtomicU64::new(0),
        };
        let snapshot_bytes = frame_msg(Fence(1), 1)
            .into_snapshot_bytes()
            .expect("frame_msg builds a Frame");
        let started = std::time::Instant::now();
        let mut forwarded = 0u64;
        for seq in 1..=50_000u64 {
            let input = input_bytes(seq);
            assert_eq!(
                route_input(&hot, &input),
                InputRouting::Forward { to: SHARD },
                "monotonic seqs always forward"
            );
            forwarded += 1;
            let out = forward_frame(&hot, SubId(0), Fence(1), &snapshot_bytes)
                .expect("current fence forwards");
            assert!(!out.is_empty());
        }
        assert_eq!(forwarded, 50_000);
        let elapsed = started.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "hot path collapsed: 50k rounds took {elapsed:?}"
        );
    }

    /// SPIKE-2a (the route-swap hot-path GATE — formally blocks the P2 route-swap design):
    /// proves the gateway route swap is WAIT-FREE and TORN-READ-FREE under a CONCURRENT
    /// `route.store` publisher (modeling a P2 `CommitAuthority` swap under load). The ONE
    /// `ArcSwap` swap means any `route.load()` yields exactly ONE published `RouteSnapshot` —
    /// never a field-mix — so authority + fence + cut (incl. `cut: Some(SeqCut)`, the field
    /// P2 adds) move together atomically; the read path is one `ArcSwap::load` (+ for
    /// `route_input` one relaxed atomic), no `Mutex` anywhere. No `CommitAuthority`-driven
    /// `route.store` lands until this is green.
    ///
    /// SCOPE (honest): this gates the SWAP MECHANIC (atomicity + the wait-free read latency).
    /// It does NOT prove the cut-PARTITIONING read logic — `route_input`'s future
    /// `seq <= marker → source / > marker → dest` branch (the slot at line ~190) is a
    /// CORRECTNESS property that gets its own test in Slice 1c, not a latency gate. Two
    /// timed bands are measured separately: the isolated ROUTE DECISION (`frame_passes_fence`
    /// = one `route.load` + fence compare, no alloc — the thing the swap actually contends),
    /// and the end-to-end per-sub FORWARD (`forward_frame` = load + retag, where production
    /// amortizes the retag once-per-SubId via SCALE-1, so this is a fan-out figure, not the
    /// route decision). The tight ROUTE-DECISION budget is what catches a contended-load
    /// regression that the alloc-dominated forward number would hide.
    ///
    /// HAND-ROLLED (no bench crate): no library expresses a concurrent-contention p99
    /// HARD-FAIL gate — criterion/divan are report-only steady-state harnesses with no p99
    /// and no fail-threshold (investigated, 2026); we would hand-compute p99 + the assert
    /// regardless. The latency ASSERTS are RELEASE-ONLY: debug + coverage instrumentation
    /// make a tail meaningless, so a debug/coverage run still exercises the concurrency plus
    /// the torn-read invariant (fast, small N) while a release run (`just spike2a`,
    /// `--test-threads=1` so siblings don't oversubscribe) enforces the timing. `Instant` is
    /// the justified seam exemption (a latency microbench is exactly what wall-clock is for).
    #[test]
    #[allow(clippy::disallowed_methods)]
    fn spike_2a_route_swap_is_wait_free_and_torn_read_free() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        use std::time::{Duration, Instant};

        // The publisher's small FIXED set of known-good routes (distinct authority + fence),
        // INCLUDING a `cut: Some(SeqCut)` member — the exact field P2's CommitAuthority adds —
        // so a Some-cut genuinely crosses the swap under contention and the torn-read
        // membership check covers the SeqCut bytes (not just the always-None P1 shape).
        // A frame at Fence(9) is never stale against any fence here, so `forward_frame`
        // always reaches the full retag (the worst-case forward).
        let routes = [
            RouteSnapshot {
                authority: SHARD,
                fence: Fence(1),
                cut: None,
            },
            RouteSnapshot {
                authority: ORCH,
                fence: Fence(2),
                cut: Some(SeqCut {
                    marker_seq: 7,
                    dest: SHARD,
                }),
            },
            RouteSnapshot {
                authority: SHARD,
                fence: Fence(3),
                cut: None,
            },
        ];
        let hot = Arc::new(SessionHot {
            route: ArcSwap::from_pointee(routes[0]),
            last_input_seq: AtomicU64::new(0),
        });
        let frame = frame_msg(Fence(9), 1)
            .into_snapshot_bytes()
            .expect("frame_msg builds a Frame");

        // 4 readers vs 1 publisher: a CONSERVATIVE-on-dev-hardware contention figure (a 4-core
        // box oversubscribes 5:N) — NOT a model of cloud core counts. Production reads a
        // session's hot state from ~one forwarder; 4 concurrent loaders is strictly HARDER, so
        // a pass here is a safe upper bound, not a scale claim.
        const READERS: usize = 4;
        // Small N under debug/coverage (instrumented — keep it quick); large N in release for
        // a stable tail. `cfg!` folds at compile time → no runtime branch (no coverage hole).
        const SAMPLES_PER_READER: usize = if cfg!(debug_assertions) {
            2_000
        } else {
            200_000
        };

        let stop = Arc::new(AtomicBool::new(false));
        let misses = Arc::new(AtomicU64::new(0));

        // Each reader returns TWO sample bands: (isolated route-decision, end-to-end forward).
        type Bands = (Vec<Duration>, Vec<Duration>);
        let bands: Vec<Bands> = std::thread::scope(|s| {
            // Publisher: swap the route as fast as it can (CommitAuthority under contention).
            let pub_hot = Arc::clone(&hot);
            let pub_stop = Arc::clone(&stop);
            s.spawn(move || {
                let mut i = 0usize;
                while !pub_stop.load(Ordering::Relaxed) {
                    pub_hot.route.store(Arc::new(routes[i % routes.len()]));
                    i = i.wrapping_add(1);
                }
            });
            let handles: Vec<_> = (0..READERS)
                .map(|_| {
                    let hot = Arc::clone(&hot);
                    let misses = Arc::clone(&misses);
                    let frame = frame.clone();
                    s.spawn(move || {
                        let mut route_read = Vec::with_capacity(SAMPLES_PER_READER);
                        let mut forward = Vec::with_capacity(SAMPLES_PER_READER);
                        let mut local = 0u64;
                        for _ in 0..SAMPLES_PER_READER {
                            // Band 1 — the ISOLATED route decision the swap contends: one
                            // `route.load` + fence compare, NO alloc, so a contended-load
                            // regression can't hide under the retag's heap-alloc noise.
                            let t = Instant::now();
                            let pass = frame_passes_fence(&hot, Fence(9));
                            route_read.push(t.elapsed());
                            // Band 2 — the end-to-end per-sub forward (load + retag alloc).
                            let t = Instant::now();
                            let out = forward_frame(&hot, SubId(0), Fence(9), &frame);
                            forward.push(t.elapsed());
                            // Invariants — accumulated via `+= u64::from(..)` (NOT an `if`), so
                            // each never-taken failure case stays a COVERED region, not a hole.
                            // A high-fence frame always passes + forwards (proves the reads
                            // RAN); a DIRECT load is a COMPLETE member of the published set.
                            // This last check is a STRUCTURAL-INVARIANT CANARY: ArcSwap cannot
                            // tear a single Arc today, so it guards a FUTURE regression where
                            // authority/fence/cut stop sharing one Arc (e.g. the P2 temptation
                            // to bolt marker_seq onto a separate atomic) — then a mixed load
                            // would be a non-member and fire here.
                            local += u64::from(!pass);
                            local += u64::from(out.is_none());
                            let loaded: RouteSnapshot = **hot.route.load();
                            local += u64::from(!routes.contains(&loaded));
                        }
                        misses.fetch_add(local, Ordering::Relaxed);
                        (route_read, forward)
                    })
                })
                .collect();
            let bands = handles
                .into_iter()
                .map(|h| h.join().expect("reader thread"))
                .collect();
            stop.store(true, Ordering::Relaxed); // let the publisher exit before scope-join
            bands
        });

        // Invariants checked in EVERY build (incl. debug/coverage): no torn read AND every
        // high-fence read passed + forwarded (misses counts all failure modes → exactly 0).
        assert_eq!(
            misses.load(Ordering::Relaxed),
            0,
            "a route.load() was not a complete member of the published set (torn read), or a \
             high-fence frame failed to pass/forward"
        );
        let route_read: Vec<Duration> = bands.iter().flat_map(|(r, _)| r.iter().copied()).collect();
        let forward: Vec<Duration> = bands.iter().flat_map(|(_, f)| f.iter().copied()).collect();
        assert_eq!(route_read.len(), READERS * SAMPLES_PER_READER);
        assert_eq!(forward.len(), READERS * SAMPLES_PER_READER);

        let route_p99 = percentile_unstable(route_read, 99);
        let forward_p99 = percentile_unstable(forward, 99);
        // The hard latency GATES are release-only (a debug/coverage tail is meaningless).
        #[cfg(not(debug_assertions))]
        {
            // The route DECISION (one ArcSwap load + fence compare) must be lost in the noise
            // of a 50 ms (20 Hz) tick — this is ~10,000x under. The budget guards the property
            // that actually matters: WAIT-FREE (no lock). Observed p99 ~625 ns under a
            // hammering publisher; a Mutex/lock in this read would be ≥20 µs under the same
            // contention, so 5 µs (~8x over observed) cleanly catches that regression while
            // staying robust on a throttled CI-less dev box. THE number that blocks P2.
            const ROUTE_DECISION_P99_BUDGET: Duration = Duration::from_micros(5);
            // The end-to-end forward includes the retag alloc production amortizes per-SubId
            // (SCALE-1) — a looser fan-out ceiling, not the route decision (observed ~600 ns).
            const FORWARD_FAN_OUT_P99_BUDGET: Duration = Duration::from_micros(50);
            eprintln!(
                "SPIKE-2a: route-decision p99 = {route_p99:?} (budget {ROUTE_DECISION_P99_BUDGET:?}); \
                 forward-fan-out p99 = {forward_p99:?} (budget {FORWARD_FAN_OUT_P99_BUDGET:?}); \
                 {} samples/band across {READERS} readers, 0 torn reads",
                READERS * SAMPLES_PER_READER
            );
            assert!(
                route_p99 < ROUTE_DECISION_P99_BUDGET,
                "route-decision p99 {route_p99:?} exceeded {ROUTE_DECISION_P99_BUDGET:?} under a \
                 concurrent route.store publisher (a contended-load regression)"
            );
            assert!(
                forward_p99 < FORWARD_FAN_OUT_P99_BUDGET,
                "forward-fan-out p99 {forward_p99:?} exceeded {FORWARD_FAN_OUT_P99_BUDGET:?}"
            );
        }
        #[cfg(debug_assertions)]
        let _ = (route_p99, forward_p99);
    }

    /// The p99-style tail of a latency sample set (sort + nearest-rank index). TOTAL — an
    /// empty set is `Duration::ZERO` (no panic), since this is earmarked for extraction to a
    /// shared harness helper for the 2nd hard latency gate (SPIKE-3a, P3) whose caller may not
    /// guarantee non-empty. Hand-rolled (no bench crate — see the spike doc).
    fn percentile_unstable(
        mut samples: Vec<std::time::Duration>,
        pct: usize,
    ) -> std::time::Duration {
        if samples.is_empty() {
            return std::time::Duration::ZERO;
        }
        samples.sort_unstable();
        let rank = samples.len().saturating_mul(pct) / 100;
        samples[rank.min(samples.len() - 1)]
    }

    #[test]
    fn percentile_unstable_total_over_empty_single_and_edges() {
        use std::time::Duration;
        let d = Duration::from_nanos;
        // Empty → ZERO (the total-ness the future harness reuse relies on; no panic).
        assert_eq!(percentile_unstable(Vec::new(), 99), Duration::ZERO);
        // Single element → itself at any percentile.
        assert_eq!(percentile_unstable(vec![d(5)], 99), d(5));
        assert_eq!(percentile_unstable(vec![d(5)], 0), d(5));
        // Nearest-rank over a known set; p100 clamps to the max (no out-of-bounds).
        let s = vec![d(10), d(40), d(20), d(30), d(50)]; // sorts to 10,20,30,40,50
        assert_eq!(percentile_unstable(s.clone(), 99), d(50)); // rank 4
        assert_eq!(percentile_unstable(s.clone(), 100), d(50)); // rank 5 → clamp to 4
        assert_eq!(percentile_unstable(s, 50), d(30)); // rank 2
    }

    #[test]
    fn hot_path_unit_outcomes() {
        let hot = SessionHot {
            route: ArcSwap::from_pointee(RouteSnapshot {
                authority: SHARD,
                fence: Fence(2),
                cut: None,
            }),
            last_input_seq: AtomicU64::new(10),
        };
        assert_eq!(route_input(&hot, &input_bytes(10)), InputRouting::Deduped);
        assert_eq!(route_input(&hot, &input_bytes(5)), InputRouting::Deduped);
        assert_eq!(
            route_input(&hot, &input_bytes(11)),
            InputRouting::Forward { to: SHARD }
        );
        assert_eq!(route_input(&hot, &[0x80]), InputRouting::Malformed);
        // Stale frame fence → None; corrupt snapshot header → None.
        assert_eq!(forward_frame(&hot, SubId(0), Fence(1), &[0]), None);
        assert_eq!(forward_frame(&hot, SubId(0), Fence(2), &[0x80; 6]), None);
        // A multi-byte sub id re-tags exactly (the varint continuation path).
        let snapshot_bytes = frame_msg(Fence(2), 3)
            .into_snapshot_bytes()
            .expect("frame_msg builds a Frame");
        let big = forward_frame(&hot, SubId(40_000), Fence(2), &snapshot_bytes)
            .expect("current fence forwards");
        let decoded: SnapshotDatagram = postcard::from_bytes(&big).expect("decode");
        assert_eq!(decoded.sub, SubId(40_000));
    }

    #[test]
    fn concurrent_inputs_at_one_seq_forward_exactly_once() {
        // FG-2: the dedup is a SINGLE atomic `fetch_max`, so many threads racing the
        // SAME seq yield exactly ONE Forward — every other thread dedups. The prior
        // non-atomic load-then-store could let several threads observe the same stale
        // high-water mark, all pass, and all forward a duplicate input. A barrier
        // maximizes the contention window.
        use std::sync::Barrier;
        use std::sync::atomic::AtomicUsize;

        const THREADS: usize = 32;
        let hot = SessionHot {
            route: ArcSwap::from_pointee(RouteSnapshot {
                authority: SHARD,
                fence: Fence(2),
                cut: None,
            }),
            last_input_seq: AtomicU64::new(0),
        };
        let forwards = AtomicUsize::new(0);
        let barrier = Barrier::new(THREADS);
        std::thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    barrier.wait();
                    if route_input(&hot, &input_bytes(7)) == (InputRouting::Forward { to: SHARD }) {
                        forwards.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
        });
        assert_eq!(
            forwards.load(Ordering::Relaxed),
            1,
            "exactly one thread forwards seq 7; the rest dedup"
        );
        assert_eq!(
            hot.last_input_seq.load(Ordering::Relaxed),
            7,
            "the high-water mark advanced to seq 7 exactly once"
        );
    }
}
