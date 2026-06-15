//! StubShard simulation systems: points in empty space (P1–P3 proving ground;
//! `docs/design/sealed_shards.md` §stub, roadmap P1). NO voxels, NO physics, NO
//! rendering — the point of a stub shard is that the CONNECTION and (later)
//! TRANSFER machinery around it is real.
//!
//! Binding behaviors encoded here:
//! - The shard NEVER sees a ticket: it acts on `(SessionId, Fence)` handed over the
//!   gateway↔shard flow, and drops stale-fence input with a logged reason — the
//!   stale-gateway drop branch exists (and is covered) from day one even though it
//!   cannot fire with a single gateway.
//! - Every avatar `EntityId` is minted HERE (`EntityId::pack`, seed-derived entropy,
//!   never time-derived — R7).
//! - Every applied or discarded input lands in [`InputLog`] — the ground truth the
//!   INPUT-CONSERVATION oracle audits against what the fabric delivered.
//! - Frames are emitted only while the shard HOLDS its realm authority (fence
//!   granted via the Directory seam), stamped with that fence.

use std::collections::{BTreeMap, VecDeque};

use bevy_ecs::prelude::{IntoScheduleConfigs, Res, ResMut, Resource, Schedule, World};
use vd_core::entity_kind::EntityKind;
use vd_core::glam::DVec3;
use vd_core::kinematics;
use vd_core::pose::{FrameRef, RealmId, StampedPose};
use vd_core::rng::SplitMix64;
use vd_core::{AccountId, EntityId, Fence, NodeId, SessionId};
use vd_wire::channels::{EntitySnap, InputDatagram, SnapshotDatagram, SubId, partition_entities};
use vd_wire::intershard::InterShardFlow;
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey, DirectoryOp, DirectoryReply};
use vd_wire::session_flow::{GatewayToShard, ShardToGateway};

use crate::io::{Inbound, MsgClass};
use crate::runtime::{ClockSample, InboundBox, NodeIdentity, OutboundBox};

/// Stub-shard configuration (composer-provided; world params seed-derived, no
/// inline literals in systems).
#[derive(Resource, Clone, Copy, Debug)]
pub struct StubConfig {
    pub realm: RealmId,
    pub frame: FrameRef,
    /// Dot walk speed, meters per second.
    pub move_speed_mps: f64,
    /// Simulation tick length, seconds.
    pub tick_dt_s: f64,
    /// The orchestrator's node id (directory seam peer).
    pub orchestrator: NodeId,
    /// Seed for the entity-mint entropy tail (NEVER wall-clock — R7).
    pub mint_seed: u64,
    /// Bounded window for the input-conservation log (SCALE-3): production sets a
    /// small ring; the harness sets a large one so the oracle sees a whole run.
    pub input_log_capacity: usize,
    /// How often (ticks) a shard re-reads its realm head to OBSERVE a lost lease
    /// (fence rule 4 self-fence — FENCE-1/5/8). 0 disables (single-shard P1 tests).
    pub realm_recheck_interval: u64,
    /// Per-datagram byte budget for snapshot partitioning (audit GW-1): a full-world
    /// snapshot is split into chunks each encoding under this, so none exceeds the
    /// QUIC datagram MTU. Operational param (never an inline literal in systems).
    pub snapshot_datagram_budget: usize,
}

/// One connected avatar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dot {
    pub entity: EntityId,
    pub account: AccountId,
    /// The Session-key fence the owning gateway holds; stale input is dropped.
    pub session_fence: Fence,
    /// The gateway this session arrived through (reply route — NEVER an address).
    pub gateway: NodeId,
    /// Authority is DERIVED FROM THE DIRECTORY (fence rule 2): a dot is held,
    /// visible, and attachable only after its entity grant is recorded. Until
    /// then it is provisional — it renders nowhere and applies nothing.
    pub granted: bool,
    /// A TRANSFER-DESTINATION input slot may APPLY input before its full directory grant
    /// (1c.5): the gateway sends `OpenInputSlot` only AFTER the directory committed authority
    /// to this shard (the saga's `commit_cas`), so this is the gateway's commit-time
    /// attestation that the shard owns the session's INPUT. It is NARROWER than `granted`:
    /// an `input_active` dot applies input (so the post-marker cut buffer drains here, input-
    /// conservation) but is NOT rendered and holds NO directory record — the real per-entity
    /// `Authority` attach + render + ghost is 1d (D-27), which sets `granted`. A regular
    /// attach sets `granted` (and `input_active` stays false; `granted` alone permits input).
    pub input_active: bool,
    /// This dot is a TRANSFER-DESTINATION ADOPT (1c.8, set by `OpenInputSlot` carrying the
    /// transfer subject): its `entity` is already the SUBJECT id (not a fresh mint), and it
    /// ADOPTS the existing directory record (a `HeadRead`, never a `LeaseGrant` which the CAS
    /// fence would Refuse) rather than minting a new entity authority. Cleared on the grant
    /// flip. An adopt grant sets `granted` (authority-held) WITHOUT `render_ready` (no pose is
    /// carried in 1c.8 — the dest renders NOTHING; 1d flips render_ready with the real pose) and
    /// WITHOUT a `SessionAttached` push (the source still owns the client connection — R2).
    pub adopting: bool,
    /// AUTHORITY-held (`granted`) is split from RENDER-ready (1c.8, D-2/D-27 user_decision): a
    /// login attach sets BOTH (it carries a real pose); a transfer-dest ADOPT sets `granted`
    /// (so the oracle/directory-record-owner sees it held) but leaves `render_ready=false` — the
    /// dot renders NOTHING (no origin-teleport, and the single-shard gateway never sees an
    /// unroutable dest frame). `emit_frames` gates on THIS, never on `granted`. 1d flips it.
    pub render_ready: bool,
    /// The mirror on release: a detached dot stays HELD (authoritative) until
    /// the directory confirms its revoke — authority is released AT the
    /// directory, never by local despawn.
    pub departing: bool,
    /// The directory-RECORDED authority fence for this entity (FENCE-1/5/8): every
    /// grant/revoke uses THIS, never a hardcoded literal, so a transfer that
    /// advanced the fence past genesis cannot wedge a logout forever.
    pub entity_fence: Fence,
    pub pose: StampedPose,
    pub yaw: f64,
    pub pitch: f64,
    pub last_applied_seq: Option<u64>,
}

/// All avatars on this shard, in deterministic session order. The key set IS the
/// shard's held-set for the AUTHORITY-UNIQUE oracle.
#[derive(Resource, Debug, Default)]
pub struct Dots(pub BTreeMap<SessionId, Dot>);

/// The realm authority this shard holds (None until the directory grants it; no
/// frames are emitted unowned).
#[derive(Resource, Debug, Default)]
pub struct RealmAuthority(pub Option<Fence>);

/// Entity minting state: a per-shard monotonic sequence + seed-derived entropy.
#[derive(Resource, Debug)]
pub struct EntityMint {
    seq: u64,
    rng: SplitMix64,
}

/// Why an input was not applied (typed — never a stringly warn-and-drop).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiscardReason {
    /// Carried a fence below the session's highest-seen (stale gateway — R2 guard).
    StaleFence,
    /// seq at or below the last applied (duplicate or reordered; latest-wins).
    DuplicateSeq,
    /// No avatar for that session on this shard.
    UnknownSession,
    /// The avatar exists but its directory grant is not yet confirmed — it has
    /// no authority to consume input (fence rule 2).
    PendingAuthority,
    /// The avatar is being released (detach received, revoke in flight) — it no
    /// longer consumes input.
    Departing,
    /// The payload failed to decode.
    MalformedInput,
    /// The payload decoded but carried a non-finite (NaN/Inf) movement/look component —
    /// a forged or corrupt client. Integrating it would PERMANENTLY poison the
    /// authoritative pose (NaN sticks through every later tick), so it is discarded
    /// and counted at the ingress gate (`InputDatagram::is_finite`), mirroring the
    /// client's delivered-pose `sanitized()` chokepoint.
    NonFiniteInput,
}

/// The INPUT-CONSERVATION ground truth: every delivered input lands here, applied or
/// discarded-with-reason. BOUNDED (SCALE-3): a day-long shard run cannot grow this
/// without limit — the windows hold the most recent `capacity` entries and the
/// `*_total` counters are EXACT for metrics. The harness sets a large window so the
/// oracle still sees a whole short run; production sets a small one.
#[derive(Resource, Debug)]
pub struct InputLog {
    applied: VecDeque<(SessionId, u64)>,
    discarded: VecDeque<(SessionId, Option<u64>, DiscardReason)>,
    capacity: usize,
    /// Exact lifetime totals (never lossy — the honest metric).
    pub applied_total: u64,
    pub discarded_total: u64,
    /// Entries evicted from the windows because the consumer fell behind the
    /// `capacity` window (in production nobody drains; counted, never an OOM).
    pub window_evictions: u64,
}

impl InputLog {
    #[must_use]
    pub fn new(capacity: usize) -> InputLog {
        InputLog {
            applied: VecDeque::new(),
            discarded: VecDeque::new(),
            capacity: capacity.max(1),
            applied_total: 0,
            discarded_total: 0,
            window_evictions: 0,
        }
    }

    fn record_applied(&mut self, session: SessionId, seq: u64) {
        self.applied_total += 1;
        if self.applied.len() >= self.capacity {
            self.applied.pop_front();
            self.window_evictions += 1;
        }
        self.applied.push_back((session, seq));
    }

    fn record_discarded(&mut self, session: SessionId, seq: Option<u64>, reason: DiscardReason) {
        self.discarded_total += 1;
        if self.discarded.len() >= self.capacity {
            self.discarded.pop_front();
            self.window_evictions += 1;
        }
        self.discarded.push_back((session, seq, reason));
    }

    /// The applied window, oldest→newest (the oracle's ground truth).
    #[must_use]
    pub fn applied(&self) -> Vec<(SessionId, u64)> {
        self.applied.iter().copied().collect()
    }

    /// The discarded window, oldest→newest.
    #[must_use]
    pub fn discarded(&self) -> Vec<(SessionId, Option<u64>, DiscardReason)> {
        self.discarded.iter().copied().collect()
    }
}

/// Per-shard monotonic snapshot frame counter.
#[derive(Resource, Debug, Default)]
pub struct FrameCounter(pub u64);

/// Counters for conditions that are tolerated but must never be silent.
#[derive(Resource, Debug, Default, PartialEq, Eq)]
pub struct StubStats {
    /// Attach requests that arrived before the realm lease was granted; the
    /// gateway retries attach until it sees `SessionAttached` (at-least-once).
    pub attaches_deferred: u64,
    /// Inter-shard frames that failed to decode at ingress (a malformed/garbage
    /// gateway→shard or directory-reply payload). Tolerated — the frame is dropped,
    /// never mis-applied — but counted so a decode regression is observable rather
    /// than log-only (ROB-E2E-1; mirrors the gateway's `undecodable`). 0 in any
    /// healthy run.
    pub undecodable: u64,
    /// `OpenInputSlot` arrived before this shard holds its realm lease — dropped + counted.
    /// **1c.5 has NO re-drive**: the gateway emits `OpenInputSlot` exactly once (at
    /// `apply_commit`) and the saga is forward-only past `Committed`, so a deferred slot means
    /// the post-marker buffer the gateway already take-drained is PERMANENTLY lost — the
    /// re-drive / durable-backstop producer is owed 1d/P3 (DEFERRED D-8 + the 1c.7 conservation
    /// gate). 0 in a healthy single-process run where the dest realm lease precedes the commit.
    pub input_slots_deferred: u64,
    /// `OpenInputSlot` carrying a fence BELOW the dot's session fence — a replay or a
    /// partitioned old gateway; dropped + counted, input never re-armed (the day-one
    /// stale-gateway-drop rule, `wire::session_flow`). 0 in a healthy run.
    pub input_slots_stale: u64,
    /// `OpenInputSlot` carrying a NON-Entity transfer subject (1c.8): the dest only ADOPTS an
    /// `Entity` transfer, so a Realm/Session/Ship subject is a counted no-op (no extraction
    /// panic, no adopt). 0 in a healthy Entity-transfer run.
    pub input_slots_malformed: u64,
    /// The `resume_from_seq` of the most recent HONORED `OpenInputSlot` — the watermark this
    /// shard last opened a transfer-dest input slot at. Observability latch (NOT a counter): it
    /// makes the gateway-EMITTED resume value visible to the cross-cut conservation gate (the
    /// emitted value must equal the client's own CUT_MARKER seq, threaded through the real saga —
    /// D-28), and is the operational answer to "what seq did this shard resume a handed-off
    /// session at". `None` until the first honored slot.
    pub last_input_slot_resume: Option<u64>,
}

/// Install the stub-shard systems and resources onto a node's world + schedule.
/// Called by the node composer for `NodeKind::StubShard` (never by feature code).
pub fn register_stub_shard(world: &mut World, schedule: &mut Schedule, config: StubConfig) {
    world.insert_resource(config);
    world.insert_resource(Dots::default());
    world.insert_resource(RealmAuthority::default());
    world.insert_resource(EntityMint {
        seq: 0,
        rng: SplitMix64::new(config.mint_seed),
    });
    world.insert_resource(InputLog::new(config.input_log_capacity));
    world.insert_resource(FrameCounter::default());
    world.insert_resource(StubStats::default());
    schedule.add_systems((request_pending_grants, process_inbound, emit_frames).chain());
}

/// Until the directory has granted this shard its realm — and every provisional
/// dot its entity — keep requesting (grants are idempotent by fence; a lost
/// reply costs one tick).
fn request_pending_grants(
    config: Res<StubConfig>,
    identity: Res<NodeIdentity>,
    clock: Res<ClockSample>,
    authority: Res<RealmAuthority>,
    dots: Res<Dots>,
    mut outbox: ResMut<OutboundBox>,
) {
    match authority.0 {
        None => {
            let op = DirectoryOp::LeaseGrant {
                key: DirectoryKey::Realm(config.realm),
                owner: AuthorityRef::Shard(identity.node_id),
                fence: Fence::GENESIS.next(),
            };
            outbox.push_flow(
                config.orchestrator,
                MsgClass::Saga,
                &InterShardFlow::Directory(op),
            );
        }
        Some(_) if config.realm_recheck_interval > 0 => {
            // Periodically re-read the realm head: the reply reveals a lost lease so
            // the shard self-fences (the loss-reaction is otherwise unreachable).
            if clock
                .local_tick
                .0
                .is_multiple_of(config.realm_recheck_interval)
            {
                let op = DirectoryOp::HeadRead {
                    key: DirectoryKey::Realm(config.realm),
                };
                outbox.push_flow(
                    config.orchestrator,
                    MsgClass::Saga,
                    &InterShardFlow::Directory(op),
                );
            }
        }
        Some(_) => {}
    }
    // The SOURCE granted-key poll fires on the SAME cadence as the realm recheck (1c.8): a
    // granted dot periodically HeadReads its Entity key so a foreign takeover (a transfer that
    // moved the entity to the dest) is OBSERVED and self-fenced. Computed once, monomorphically.
    let poll_granted_keys =
        granted_key_poll_tick(config.realm_recheck_interval, clock.local_tick.0);
    for dot in dots.0.values() {
        if let Some(op) = pending_grant_op(dot, identity.node_id, poll_granted_keys) {
            outbox.push_flow(
                config.orchestrator,
                MsgClass::Saga,
                &InterShardFlow::Directory(op),
            );
        }
    }
}

/// Whether this is a SOURCE granted-key poll tick (1c.8): a disabled interval (0) never polls;
/// otherwise the poll fires on the same `is_multiple_of(interval)` cadence as the realm recheck.
/// Monomorphic so the `&&`-short-circuit lives in ONE covered helper (HR5), not the system body.
///
/// ⚠️ INTERIM, OWED-FOR-REMOVAL (DEFERRED D-2): this poll is how the SOURCE DISCOVERS a foreign
/// takeover to self-demote — a cooperative, best-effort directory poll, NOT the binding fence-first
/// saga-pushed `Demote`. 1d/the band-ghost slice RIPS IT OUT (see `self_fence_foreign_entity`).
#[must_use]
fn granted_key_poll_tick(interval: u64, local_tick: u64) -> bool {
    interval > 0 && local_tick.is_multiple_of(interval)
}

/// The per-dot directory op `request_pending_grants` should (re)issue, as a monomorphic 3-way
/// (+ the source granted-key poll) so the system loop stays a branchless shim (HR5):
/// - adopting + !granted → `HeadRead{Entity}` (1c.8): ADOPT the record the transfer CAS moved
///   here; a `LeaseGrant` at `GENESIS.next` would be Refused (the record is past genesis).
/// - !adopting + !granted → `LeaseGrant{Entity}` at `GENESIS.next` (the login fresh-mint path).
/// - departing → `LeaseRevoke{Entity}` at the RECORDED fence (FENCE-1/5/8; a literal would be
///   Refused once a transfer advanced the fence, stranding the logout).
/// - granted, non-departing, non-adopting, AND it is a poll tick → `HeadRead{Entity}` (1c.8 the
///   SOURCE granted-key poll): the source observes a foreign takeover of its held entity and
///   self-fences (the Entity-key analogue of the realm recheck).
#[must_use]
fn pending_grant_op(dot: &Dot, node: NodeId, poll_granted_keys: bool) -> Option<DirectoryOp> {
    if !dot.granted {
        if dot.adopting {
            Some(DirectoryOp::HeadRead {
                key: DirectoryKey::Entity(dot.entity),
            })
        } else {
            Some(DirectoryOp::LeaseGrant {
                key: DirectoryKey::Entity(dot.entity),
                owner: AuthorityRef::Shard(node),
                fence: Fence::GENESIS.next(),
            })
        }
    } else if dot.departing {
        Some(DirectoryOp::LeaseRevoke {
            key: DirectoryKey::Entity(dot.entity),
            fence: dot.entity_fence,
        })
    } else if poll_granted_keys {
        Some(DirectoryOp::HeadRead {
            key: DirectoryKey::Entity(dot.entity),
        })
    } else {
        None
    }
}

/// Drain and dispatch everything delivered this tick.
#[allow(clippy::too_many_arguments)]
fn process_inbound(
    config: Res<StubConfig>,
    identity: Res<NodeIdentity>,
    clock: Res<ClockSample>,
    inbox: Res<InboundBox>,
    mut dots: ResMut<Dots>,
    mut authority: ResMut<RealmAuthority>,
    mut mint: ResMut<EntityMint>,
    mut log: ResMut<InputLog>,
    mut stats: ResMut<StubStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    for msg in &inbox.0 {
        let Inbound::Wire { from, class, bytes } = msg else {
            // Unreachability notices are observed by the node shell (TickReport);
            // the stub has no retry obligations in P1.
            continue;
        };
        match class {
            MsgClass::Control | MsgClass::Input => {
                let ctx = GatewayMsgCtx {
                    config: &config,
                    identity: &identity,
                    clock: &clock,
                    realm_fence: authority.0,
                };
                on_gateway_msg(
                    bytes,
                    *from,
                    &ctx,
                    &mut dots,
                    &mut mint,
                    &mut log,
                    &mut stats,
                    &mut outbox,
                );
            }
            MsgClass::Saga => on_directory_reply(
                bytes,
                &identity,
                &config,
                &mut authority,
                &mut dots,
                &mut stats,
                &mut outbox,
            ),
            // Membership (clock sync) is consumed by the node-level follower system;
            // Snapshot never targets a shard.
            MsgClass::Membership | MsgClass::Snapshot => {}
        }
    }
}

/// Read-only context for one gateway-message dispatch.
struct GatewayMsgCtx<'a> {
    config: &'a StubConfig,
    identity: &'a NodeIdentity,
    clock: &'a ClockSample,
    realm_fence: Option<Fence>,
}

/// Handle one gateway→shard message.
#[allow(clippy::too_many_arguments)]
fn on_gateway_msg(
    bytes: &[u8],
    from: NodeId,
    ctx: &GatewayMsgCtx<'_>,
    dots: &mut Dots,
    mint: &mut EntityMint,
    log: &mut InputLog,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    let Ok(msg) = postcard::from_bytes::<GatewayToShard>(bytes) else {
        stats.undecodable += 1;
        tracing::error!("undecodable gateway->shard message");
        return;
    };
    match msg {
        GatewayToShard::AttachSession {
            session,
            fence,
            account,
        } => {
            // An avatar cannot exist on a shard that doesn't own its realm yet:
            // defer (counted); the gateway retries attach until it sees the reply.
            let Some(realm_fence) = ctx.realm_fence else {
                stats.attaches_deferred += 1;
                return;
            };
            let dot = dots.0.entry(session).or_insert_with(|| {
                let entity = mint_entity(mint, ctx.identity.node_id);
                Dot {
                    entity,
                    account,
                    session_fence: fence,
                    gateway: from,
                    granted: false,
                    input_active: false,
                    adopting: false,
                    render_ready: false,
                    departing: false,
                    entity_fence: Fence::GENESIS,
                    pose: StampedPose::at_rest(
                        ctx.config.frame,
                        DVec3::ZERO,
                        ctx.clock.universe_tick,
                    ),
                    yaw: 0.0,
                    pitch: 0.0,
                    last_applied_seq: None,
                }
            });
            // Idempotent re-attach: refresh the fence if the gateway's advanced.
            if fence > dot.session_fence {
                dot.session_fence = fence;
            }
            if dot.granted {
                // Authority recorded: confirm (idempotently, on every retry).
                let reply = ShardToGateway::SessionAttached {
                    session,
                    entity: dot.entity,
                    frame: ctx.config.frame,
                    realm_fence,
                };
                push_session_reply(outbox, from, &reply);
            } else {
                // The avatar exists ONLY provisionally until the directory
                // records its grant (requested by `request_pending_grants`,
                // retried every tick). No reply yet — the gateway retries attach.
                let op = DirectoryOp::LeaseGrant {
                    key: DirectoryKey::Entity(dot.entity),
                    owner: AuthorityRef::Shard(ctx.identity.node_id),
                    fence: Fence::GENESIS.next(),
                };
                outbox.push_flow(
                    ctx.config.orchestrator,
                    MsgClass::Saga,
                    &InterShardFlow::Directory(op),
                );
            }
        }
        GatewayToShard::SessionInput {
            session,
            fence,
            input_bytes,
        } => {
            apply_input(
                ctx.config,
                ctx.clock,
                dots,
                log,
                session,
                fence,
                &input_bytes,
            );
        }
        GatewayToShard::DetachSession { session, fence } => {
            match dots.0.get_mut(&session) {
                Some(dot) if fence.is_stale_against(dot.session_fence) => {
                    log.record_discarded(session, None, DiscardReason::StaleFence);
                }
                Some(dot) if dot.granted => {
                    // Two-phase release: the dot stays HELD until the directory
                    // confirms the revoke (authority is released AT the
                    // directory, mirroring the grant). The revoke is (re)sent by
                    // `request_pending_grants`; the SessionDetached reply waits
                    // for the confirmation.
                    dot.departing = true;
                }
                Some(_) => {
                    // A provisional dot has no directory record: safe to drop now.
                    dots.0.remove(&session);
                    push_session_reply(outbox, from, &ShardToGateway::SessionDetached { session });
                }
                None => {
                    // Idempotent: detaching an unknown session still confirms.
                    push_session_reply(outbox, from, &ShardToGateway::SessionDetached { session });
                }
            }
        }
        GatewayToShard::OpenInputSlot {
            session,
            fence,
            account,
            resume_from_seq,
            subject,
        } => {
            // A transfer-DESTINATION input slot. The gateway sends this only AFTER the
            // directory committed authority to this shard (the saga's `commit_cas`), so the
            // shard may APPLY this session's input even before its own per-entity grant
            // records: the post-marker cut buffer the gateway held drains here. The dot is
            // `input_active` and ADOPTS the transfer subject (1c.8): its entity becomes the
            // SUBJECT id (the record the CAS moved here), so the adopt HeadRead lands on that
            // record and flips `granted` (authority-held). It is NOT `render_ready` — it renders
            // nowhere (no pose carried; 1d) — and gets no `SessionAttached` reply (the source
            // still owns the client connection — R2).
            let Some(_realm_fence) = ctx.realm_fence else {
                // No realm lease yet: drop + count. NO re-drive in 1c.5 (the gateway already
                // take-drained the buffer; OpenInputSlot is emitted once, the saga is
                // forward-only past Committed) — the recovery producer is owed 1d/P3 (D-8).
                stats.input_slots_deferred += 1;
                return;
            };
            // Extract the SUBJECT EntityId to ADOPT. A non-Entity subject (e.g. a Realm-subject
            // saga, which the FSM proptests drive through CommitAuthority) is a COUNTED no-op —
            // never an extraction panic: the dest only adopts an Entity transfer.
            let Some(subject_entity) = subject_entity(subject) else {
                stats.input_slots_malformed += 1;
                tracing::warn!(
                    ?subject,
                    "OpenInputSlot carried a non-Entity subject — no adopt (counted no-op)"
                );
                return;
            };
            let dot = dots.0.entry(session).or_insert_with(|| Dot {
                entity: subject_entity, // 1c.8 ADOPT: the transferred subject id, not a fresh mint
                account,
                session_fence: fence,
                gateway: from,
                granted: false,
                input_active: false,
                adopting: true,
                render_ready: false,
                departing: false,
                entity_fence: Fence::GENESIS, // the adopt HeadRead fills the real CAS fence
                pose: StampedPose::at_rest(ctx.config.frame, DVec3::ZERO, ctx.clock.universe_tick),
                yaw: 0.0,
                pitch: 0.0,
                last_applied_seq: None,
            });
            // STALE-GATEWAY-DROP (the binding day-one rule, `wire::session_flow`): a slot whose
            // fence is BELOW the dot's session fence is a replay or a partitioned old gateway —
            // drop it, never re-arm input (1d adds departing/revoking states this guards). A
            // fresh mint set `session_fence := fence`, so it is never stale against itself.
            if fence.is_stale_against(dot.session_fence) {
                stats.input_slots_stale += 1;
                return;
            }
            // SECURITY / HR1: only activate + seed a PROVISIONAL slot, and only from the
            // gateway that owns the session — a shard must never apply input for a session it
            // was not legitimately routed, and a granted entity's input stream is owned by its
            // own applied watermark.
            if !dot.granted && dot.gateway == from {
                dot.input_active = true;
                // OBSERVABILITY: record the gateway-EMITTED resume watermark (unmutated) so the
                // cross-cut conservation gate can assert it equals the client's CUT_MARKER seq
                // (D-28). Distinct from `dot.last_applied_seq`, which advances as the drained
                // batch applies — this latch holds the AS-RECEIVED value.
                stats.last_input_slot_resume = Some(resume_from_seq);
                // SEED the dedup watermark to `resume_from_seq` (= marker_seq): the drained
                // resume batch (marker+1..) applies in order; a `seq <= marker` replay is
                // rejected (the source already applied it). MAX-merge — never LOWER it.
                dot.last_applied_seq = Some(
                    dot.last_applied_seq
                        .map_or(resume_from_seq, |s| s.max(resume_from_seq)),
                );
                if fence > dot.session_fence {
                    dot.session_fence = fence;
                }
            }
        }
    }
}

/// The transfer SUBJECT's `EntityId` for an ADOPT, or `None` for a non-Entity subject (the
/// counted no-op). Monomorphic so the `OpenInputSlot` arm + the directory reply arms stay
/// branchless shims (HR5): the ONE `DirectoryKey → EntityId` extraction in the dest.
#[must_use]
fn subject_entity(subject: DirectoryKey) -> Option<EntityId> {
    match subject {
        DirectoryKey::Entity(entity) => Some(entity),
        DirectoryKey::Session(_) | DirectoryKey::Realm(_) | DirectoryKey::Ship(_) => None,
    }
}

/// Mint a fresh avatar id: `{kind, mint_shard, seq, rand24}` — monotonic sequence
/// plus seed-derived entropy; NEVER time-derived (R7).
fn mint_entity(mint: &mut EntityMint, node: NodeId) -> EntityId {
    let seq = mint.seq;
    mint.seq += 1;
    let rand24 = (mint.rng.next_u64() & 0x00FF_FFFF) as u32;
    // NodeIds fit u32 in every real topology; the mask documents the packing.
    let mint_shard = (node.0 & u64::from(u32::MAX)) as u32;
    EntityId::pack(EntityKind::Player, mint_shard, seq, rand24)
}

fn push_session_reply(outbox: &mut OutboundBox, to: NodeId, reply: &ShardToGateway) {
    let bytes = postcard::to_allocvec(reply).expect("closed wire enums serialize infallibly");
    outbox
        .0
        .push((to, MsgClass::Control, crate::io::bytes(bytes)));
}

/// The input path: fence gate → decode → seq gate → integrate. Every outcome lands
/// in the [`InputLog`].
fn apply_input(
    config: &StubConfig,
    clock: &ClockSample,
    dots: &mut Dots,
    log: &mut InputLog,
    session: SessionId,
    fence: Fence,
    input_bytes: &[u8],
) {
    let Some(dot) = dots.0.get_mut(&session) else {
        log.record_discarded(session, None, DiscardReason::UnknownSession);
        return;
    };
    // A dot may apply input once it is the AUTHORITY for the session's input: either a fully
    // granted attach (`granted`) OR a committed transfer-destination input slot
    // (`input_active`, 1c.5 — the gateway opened it only post-directory-commit). A purely
    // provisional dot (neither) drops input as PendingAuthority.
    if !dot.granted && !dot.input_active {
        log.record_discarded(session, None, DiscardReason::PendingAuthority);
        return;
    }
    if dot.departing {
        log.record_discarded(session, None, DiscardReason::Departing);
        return;
    }
    if fence.is_stale_against(dot.session_fence) {
        log.record_discarded(session, None, DiscardReason::StaleFence);
        return;
    }
    // A higher fence means the gateway re-granted (P3 adoption); track it.
    if fence > dot.session_fence {
        dot.session_fence = fence;
    }
    let Ok(input) = postcard::from_bytes::<InputDatagram>(input_bytes) else {
        log.record_discarded(session, None, DiscardReason::MalformedInput);
        return;
    };
    // The authoritative-ingress finite gate (never trust network input): a NaN/Inf
    // component would integrate into the pose and STICK. Discarded + counted; the seq
    // does NOT advance (the input was never applied — INPUT-CONSERVATION holds).
    if !input.is_finite() {
        log.record_discarded(session, Some(input.seq), DiscardReason::NonFiniteInput);
        return;
    }
    if dot.last_applied_seq.is_some_and(|last| input.seq <= last) {
        log.record_discarded(session, Some(input.seq), DiscardReason::DuplicateSeq);
        return;
    }
    dot.last_applied_seq = Some(input.seq);
    integrate(dot, &input, config, clock);
    log.record_applied(session, input.seq);
}

/// Kinematic point integration: axes are clamped to [-1, 1], displacement is
/// speed·dt in the dot's yaw-rotated heading. Pure f64 closed-form per tick.
fn integrate(dot: &mut Dot, input: &InputDatagram, config: &StubConfig, clock: &ClockSample) {
    dot.yaw += f64::from(input.look[0]);
    // Pitch is CLAMPED to the valid look range (WB-1): unbounded accumulation would wrap
    // past the ±π/2 gimbal pole and silently corrupt authoritative orientation. Yaw wraps
    // freely (no pole). ONE shared bound (`vd_core::kinematics::PITCH_LIMIT`).
    dot.pitch = kinematics::clamp_pitch(dot.pitch + f64::from(input.look[1]));
    dot.orient_from_angles();
    // The movement-axis map is the ONE shared input convention (vd_core::kinematics) —
    // the client's nav/camera invert the SAME definition (no hand-re-encoded drift).
    let axes = kinematics::local_axes_from_movement(input.movement);
    let step = dot.pose.orient * axes * (config.move_speed_mps * config.tick_dt_s);
    dot.pose.pos += step;
    dot.pose.vel = step / config.tick_dt_s;
    dot.pose.universe_tick = clock.universe_tick;
}

impl Dot {
    fn orient_from_angles(&mut self) {
        self.pose.orient = kinematics::orient_from_yaw_pitch(self.yaw, self.pitch);
    }
}

/// A `SessionAttached` reply plus the gateway it routes to (the grant-flip's optional egress).
struct AttachEgress {
    gateway: NodeId,
    reply: ShardToGateway,
}

/// Flip the matching provisional dot to `granted` at the recorded fence, returning a
/// `SessionAttached` egress IFF this is a login attach (NOT a transfer-dest adopt). Monomorphic
/// (the loop + the adopt branching live here) so the reply arm stays a branchless shim (HR5).
///
/// - login (`!adopting`): granted := true, render_ready := true, push SessionAttached.
/// - adopt (`adopting`, 1c.8): granted := true, render_ready := false, adopting cleared, NO
///   SessionAttached (R2 — the source owns the client; 1d re-homes), so returns `None`.
///
/// Guarded by `!dot.granted` so a duplicate grant head is idempotent (no second flip/reply).
fn flip_grant(
    dots: &mut BTreeMap<SessionId, Dot>,
    entity: EntityId,
    fence: Fence,
    config: &StubConfig,
    realm_fence: Fence,
) -> Option<AttachEgress> {
    for (session, dot) in dots.iter_mut() {
        if !dot_grant_target(dot, entity) {
            continue;
        }
        dot.granted = true;
        dot.entity_fence = fence; // the recorded authority fence (== the CAS new_fence)
        if dot.adopting {
            // Transfer-dest adopt: held but not rendered, no re-home (1c.8). Clear the adopt
            // marker so the source granted-key poll engages on this now-granted dot at the dest.
            dot.adopting = false;
            return None;
        }
        dot.render_ready = true;
        return Some(AttachEgress {
            gateway: dot.gateway,
            reply: ShardToGateway::SessionAttached {
                session: *session,
                entity,
                frame: config.frame,
                realm_fence,
            },
        });
    }
    None
}

/// Whether a dot is the (single) ungranted holder of `entity` awaiting its grant flip.
/// Monomorphic predicate (the `&&` short-circuit is covered once here, not in the loop body).
#[must_use]
fn dot_grant_target(dot: &Dot, entity: EntityId) -> bool {
    (dot.entity == entity) & !dot.granted
}

/// The SOURCE self-fence (1c.8): a directory head naming a FOREIGN owner of an entity this shard
/// holds means the transfer CAS moved the record to the dest — DROP the local granted,
/// non-departing dot for it (NO directory write; the record is the dest's now). Idempotent: a
/// second foreign-owner reply for an already-dropped entity finds no matching dot. Monomorphic
/// (the find + remove is hoisted out of the reply arm — HR5 branchless shim).
///
/// ⚠️ INTERIM, OWED-FOR-REMOVAL (DEFERRED D-2): this is a COOPERATIVE IN-MEMORY drop discovered by
/// the granted-key poll — NOT the binding ordered, FENCE-enforced demote-before-promote
/// (`transfer_protocol.md` §2.4: "the freeze is enforced by the fence, not by cooperative in-memory
/// state"). The dest already promoted (`flip_grant`) off its own post-CAS HeadRead, so there is a
/// transient two-holder window (masked only by `render_ready=false` + the single-shard gateway),
/// and a crash before the next poll strands a stale source grant. 1d/the band-ghost slice RIPS THIS
/// OUT: a saga-pushed `Demote`/`DemoteAck` flips the source to a RETAINED ghost-as-collider via
/// `authority.rs` `Owned→Frozen→Ghost` demote-BEFORE-promote — a tear-out, not a predicate swap.
fn self_fence_foreign_entity(
    dots: &mut BTreeMap<SessionId, Dot>,
    entity: EntityId,
    foreign: AuthorityRef,
) {
    let Some(session) = dots
        .iter()
        .find(|(_, d)| foreign_takeover_target(d, entity))
        .map(|(session, _)| *session)
    else {
        return; // no matching dot: already dropped (or never held here) — a clean no-op
    };
    tracing::warn!(
        "entity {entity} now held by {foreign:?} — self-demoting the transferred dot (local drop)"
    );
    dots.remove(&session);
}

/// Whether a dot is the local granted, non-departing holder of `entity` (the self-fence target).
/// Monomorphic predicate so the chained `&&`s are covered in one helper, not the reply arm.
#[must_use]
fn foreign_takeover_target(dot: &Dot, entity: EntityId) -> bool {
    (dot.entity == entity) & dot.granted & !dot.departing
}

/// Handle a directory reply: realm-lease and entity-grant confirmations.
#[allow(clippy::too_many_arguments)]
fn on_directory_reply(
    bytes: &[u8],
    identity: &NodeIdentity,
    config: &StubConfig,
    authority: &mut RealmAuthority,
    dots: &mut Dots,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    // The orchestrator wraps every reply in InterShardFlow::DirectoryReply (the dispatch
    // split); decode the envelope once and route the inner reply. Any other Saga-class arm
    // to a stub (none today; the Transfer arm to a dest shard lands at 1d) is ignored.
    let reply = match postcard::from_bytes::<InterShardFlow>(bytes) {
        Ok(InterShardFlow::DirectoryReply(reply)) => reply,
        Ok(_) => return,
        Err(_) => {
            stats.undecodable += 1;
            tracing::error!("undecodable saga-class message");
            return;
        }
    };
    match reply {
        DirectoryReply::Head {
            key: DirectoryKey::Realm(_),
            record: Some(record),
        } => {
            if record.authority == AuthorityRef::Shard(identity.node_id) {
                authority.0 = Some(record.fence);
            } else {
                // The realm was taken over (P2 transfer / reassignment): SELF-FENCE
                // immediately (fence rule 4) — drop authority and stop emitting
                // frames so a stale old owner cannot affect clients.
                tracing::warn!(
                    "realm lease now held by {:?}, not this shard — self-fencing",
                    record.authority
                );
                authority.0 = None;
            }
        }
        DirectoryReply::Head {
            key: DirectoryKey::Realm(_),
            record: None,
        } => {
            // The realm record is gone (revoked): self-fence (frames stop).
            authority.0 = None;
        }
        DirectoryReply::Head {
            key: DirectoryKey::Entity(entity),
            record: Some(record),
        } => {
            // The avatar's authority is now RECORDED: it becomes held, visible,
            // and attachable (fence rule 2 — authority derives from the directory).
            let Some(realm_fence) = authority.0 else {
                return; // grant raced ahead of the realm lease: retry resolves it
            };
            if record.authority == AuthorityRef::Shard(identity.node_id) {
                // OURS: flip granted + stamp the recorded fence. A login attach pushes
                // SessionAttached and becomes render-ready; a transfer-dest ADOPT (1c.8) suppresses
                // the attach (R2 — source owns the client) and stays NOT render-ready (no pose
                // carried; 1d). The login-vs-adopt branching is hoisted into `flip_grant`, which
                // returns the optional `SessionAttached` egress (Some for a login, None for an adopt).
                if let Some(reply) =
                    flip_grant(&mut dots.0, entity, record.fence, config, realm_fence)
                {
                    push_session_reply(outbox, reply.gateway, &reply.reply);
                }
            } else {
                // FOREIGN owner of an entity this shard holds (1c.8 the SOURCE self-fence, the
                // Entity-key analogue of the realm self-fence): the transfer CAS moved the record
                // to the dest, so the source DROPS its dot LOCALLY — NO LeaseRevoke (a revoke at the
                // stale entity_fence is Refused; a revoke at new_fence would delete the dest's
                // record), NO `departing`. Idempotent: no matching dot → no-op.
                self_fence_foreign_entity(&mut dots.0, entity, record.authority);
            }
        }
        DirectoryReply::Head {
            key: DirectoryKey::Entity(entity),
            record: None,
        } => {
            // The revoke is RECORDED (no record remains): finish the release —
            // despawn and confirm to the gateway.
            let departed: Vec<SessionId> = dots
                .0
                .iter()
                .filter(|(_, d)| (d.entity == entity) & d.departing)
                .map(|(s, _)| *s)
                .collect();
            for session in departed {
                let dot = dots.0.remove(&session).expect("just found");
                push_session_reply(
                    outbox,
                    dot.gateway,
                    &ShardToGateway::SessionDetached { session },
                );
            }
        }
        // Headless realm reads, CAS results, clock answers: no obligation in P1.
        _ => {}
    }
}

/// Emit one fence-stamped frame per tick to every gateway with an attached session.
/// No realm authority ⇒ no frames (an unowned shard is silent, never speculative).
fn emit_frames(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    authority: Res<RealmAuthority>,
    dots: Res<Dots>,
    mut counter: ResMut<FrameCounter>,
    mut outbox: ResMut<OutboundBox>,
) {
    let Some(realm_fence) = authority.0 else {
        return;
    };
    // Gate on RENDER-ready, not authority-held (1c.8 split): a transfer-dest ADOPT is granted
    // (authority-held, oracle-visible) but renders NOTHING until 1d carries a pose + flips
    // render_ready — so it never emits an origin-teleport frame, and the single-shard gateway
    // never sees an unroutable dest frame.
    let mut gateways: Vec<NodeId> = dots
        .0
        .values()
        .filter(|d| d.render_ready)
        .map(|d| d.gateway)
        .collect();
    gateways.sort_unstable();
    gateways.dedup();
    if gateways.is_empty() {
        return;
    }
    let entities: Vec<EntitySnap> = dots
        .0
        .values()
        .filter(|d| d.render_ready)
        .map(|d| EntitySnap {
            entity: d.entity,
            pose: d.pose,
        })
        .collect();
    // Partition BY CONTENT so no datagram exceeds the MTU budget (audit GW-1): a
    // full-world snapshot ships as several independent self-contained frames. Per
    // connection_plane.md §6.3 EVERY chunk of one tick carries the SAME frame_id +
    // celestial_tick (each chunk self-contained, latest-wins) — the client merges
    // them and treats only a STRICTLY older frame_id as stale, so a reordered
    // sibling chunk of the same tick is never dropped. The counter advances once per
    // tick. The shared partitioner is the ONE place every shard type does this.
    let frame_id = counter.0;
    counter.0 += 1;
    for chunk in partition_entities(&entities, config.snapshot_datagram_budget) {
        let snapshot = SnapshotDatagram {
            // The shard always stamps sub 0; the gateway re-tags per session.
            sub: SubId(0),
            frame_id,
            source_tick: clock.local_tick,
            universe_tick: clock.universe_tick,
            entities: chunk,
        };
        let snapshot_bytes =
            postcard::to_allocvec(&snapshot).expect("closed wire enums serialize infallibly");
        let frame = ShardToGateway::Frame {
            realm_fence,
            source_tick: clock.local_tick,
            snapshot_bytes,
        };
        // ONE shared body per chunk, cloned (refcount bump) to every subscribing
        // gateway — never an O(entities) copy per gateway (SCALE-1).
        let bytes = crate::io::bytes(
            postcard::to_allocvec(&frame).expect("closed wire enums serialize infallibly"),
        );
        for &gateway in &gateways {
            outbox.0.push((gateway, MsgClass::Snapshot, bytes.clone()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::NodeKind;
    use vd_core::{MsgId, UniverseTick};

    const SHARD: NodeId = NodeId(10);
    const GATEWAY: NodeId = NodeId(20);
    const ORCH: NodeId = NodeId(30);
    const SESSION: SessionId = SessionId(0xAA);

    fn config() -> StubConfig {
        StubConfig {
            realm: RealmId::System(7),
            frame: FrameRef::SystemSpace { system_seed: 7 },
            move_speed_mps: 2.0,
            tick_dt_s: 0.05,
            orchestrator: ORCH,
            mint_seed: 99,
            input_log_capacity: 1024,
            realm_recheck_interval: 0,
            snapshot_datagram_budget: 1100,
        }
    }

    struct Rig {
        world: World,
        schedule: Schedule,
    }

    impl Rig {
        fn new() -> Rig {
            Rig::with_config(config())
        }

        fn with_config(cfg: StubConfig) -> Rig {
            let mut world = World::new();
            world.insert_resource(InboundBox::default());
            world.insert_resource(OutboundBox::default());
            world.insert_resource(NodeIdentity {
                node_id: SHARD,
                kind: NodeKind::StubShard,
            });
            world.insert_resource(ClockSample {
                local_tick: vd_core::TickId(1),
                universe_tick: UniverseTick(100),
                epoch: vd_core::EpochId(1),
            });
            let mut schedule = Schedule::default();
            register_stub_shard(&mut world, &mut schedule, cfg);
            Rig { world, schedule }
        }

        /// Set the rig's local tick (the test rig runs the schedule directly, so it
        /// must drive the clock the node shell would normally advance).
        fn set_local_tick(&mut self, tick: u64) {
            self.world.resource_mut::<ClockSample>().local_tick = vd_core::TickId(tick);
        }

        /// Run one tick with the given inbound; returns everything sent.
        fn tick(&mut self, inbound: Vec<Inbound>) -> Vec<(NodeId, MsgClass, Vec<u8>)> {
            self.world.resource_mut::<InboundBox>().0 = inbound;
            self.schedule.run(&mut self.world);
            std::mem::take(&mut self.world.resource_mut::<OutboundBox>().0)
                .into_iter()
                .map(|(to, class, bytes)| (to, class, bytes.to_vec()))
                .collect()
        }

        fn grant_realm(&mut self) {
            let reply = DirectoryReply::Head {
                key: DirectoryKey::Realm(config().realm),
                record: Some(vd_wire::seams::directory::OwnerRecord {
                    authority: AuthorityRef::Shard(SHARD),
                    fence: Fence(1),
                    lease_expires: UniverseTick(1_000),
                    in_transfer: None,
                }),
            };
            let bytes = crate::io::bytes(
                postcard::to_allocvec(&InterShardFlow::DirectoryReply(reply)).expect("encode"),
            );
            let _ = self.tick(vec![Inbound::Wire {
                from: ORCH,
                class: MsgClass::Saga,
                bytes,
            }]);
        }

        /// Send the attach request only (the dot stays provisional).
        fn attach_request(
            &mut self,
            session: SessionId,
            gateway: NodeId,
        ) -> Vec<(NodeId, MsgClass, Vec<u8>)> {
            let msg = GatewayToShard::AttachSession {
                session,
                fence: Fence(1),
                account: AccountId(5),
            };
            self.tick(vec![wire_msg(gateway, MsgClass::Control, &msg)])
        }

        /// Deliver the directory's entity-grant confirmation for `session`'s dot.
        fn confirm_entity_grant(&mut self, session: SessionId) -> Vec<(NodeId, MsgClass, Vec<u8>)> {
            let entity = self.world.resource::<Dots>().0[&session].entity;
            let reply = DirectoryReply::Head {
                key: DirectoryKey::Entity(entity),
                record: Some(vd_wire::seams::directory::OwnerRecord {
                    authority: AuthorityRef::Shard(SHARD),
                    fence: Fence(1),
                    lease_expires: UniverseTick(1_000),
                    in_transfer: None,
                }),
            };
            self.tick(vec![wire_msg(
                ORCH,
                MsgClass::Saga,
                &InterShardFlow::DirectoryReply(reply),
            )])
        }

        /// The full attach flow: request, then grant confirmation.
        fn attach(&mut self) -> Vec<(NodeId, MsgClass, Vec<u8>)> {
            let _ = self.attach_request(SESSION, GATEWAY);
            self.confirm_entity_grant(SESSION)
        }
    }

    fn wire_msg<T: serde::Serialize>(from: NodeId, class: MsgClass, msg: &T) -> Inbound {
        Inbound::Wire {
            from,
            class,
            bytes: postcard::to_allocvec(msg).expect("encode").into(),
        }
    }

    fn input_msg(seq: u64, fence: Fence, movement: [f32; 3], look: [f32; 2]) -> Inbound {
        let input = InputDatagram {
            seq,
            is_cut_marker: false,
            client_tick: vd_core::TickId(2),
            movement,
            look,
            action_bits: 0,
        };
        let msg = GatewayToShard::SessionInput {
            session: SESSION,
            fence,
            input_bytes: postcard::to_allocvec(&input).expect("encode input"),
        };
        wire_msg(GATEWAY, MsgClass::Input, &msg)
    }

    fn decode_frames(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<SnapshotDatagram> {
        sent.iter()
            .filter(|(_, class, _)| *class == MsgClass::Snapshot)
            .map(|(_, _, bytes)| {
                let frame: ShardToGateway = postcard::from_bytes(bytes).expect("frame");
                let snapshot_bytes = frame
                    .into_snapshot_bytes()
                    .expect("snapshot class carries Frame");
                postcard::from_bytes::<SnapshotDatagram>(&snapshot_bytes).expect("snapshot")
            })
            .collect()
    }

    #[test]
    fn boot_requests_the_realm_lease_until_granted_then_stops() {
        let mut rig = Rig::new();
        let expected_request =
            postcard::to_allocvec(&InterShardFlow::Directory(DirectoryOp::LeaseGrant {
                key: DirectoryKey::Realm(config().realm),
                owner: AuthorityRef::Shard(SHARD),
                fence: Fence(1),
            }))
            .expect("encode");
        // Two unanswered ticks: two identical idempotent requests.
        for _ in 0..2 {
            let sent = rig.tick(vec![]);
            assert_eq!(sent, vec![(ORCH, MsgClass::Saga, expected_request.clone())]);
        }
        rig.grant_realm();
        assert_eq!(rig.world.resource::<RealmAuthority>().0, Some(Fence(1)));
        // Granted: no more requests (and no frames — no sessions yet).
        assert_eq!(rig.tick(vec![]), vec![]);
    }

    #[test]
    fn foreign_realm_grant_is_rejected_loudly_and_authority_stays_none() {
        let mut rig = Rig::new();
        let reply = DirectoryReply::Head {
            key: DirectoryKey::Realm(config().realm),
            record: Some(vd_wire::seams::directory::OwnerRecord {
                authority: AuthorityRef::Shard(NodeId(99)),
                fence: Fence(1),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        };
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(reply),
        )]);
        assert_eq!(rig.world.resource::<RealmAuthority>().0, None);
    }

    #[test]
    fn a_lost_realm_lease_self_fences_the_shard() {
        // FENCE-1/5/8: once granted, a realm head showing a FOREIGN owner (a P2
        // takeover) or NO record makes the shard drop authority and stop frames —
        // a stale old owner cannot affect clients (fence rule 4).
        let mut rig = Rig::new();
        rig.grant_realm();
        assert_eq!(rig.world.resource::<RealmAuthority>().0, Some(Fence(1)));
        // A foreign owner head: self-fence.
        let foreign = DirectoryReply::Head {
            key: DirectoryKey::Realm(config().realm),
            record: Some(vd_wire::seams::directory::OwnerRecord {
                authority: AuthorityRef::Shard(NodeId(99)),
                fence: Fence(2),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        };
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(foreign),
        )]);
        assert_eq!(
            rig.world.resource::<RealmAuthority>().0,
            None,
            "self-fenced"
        );

        // Re-grant, then a headless realm read (record gone): also self-fence.
        rig.grant_realm();
        assert_eq!(rig.world.resource::<RealmAuthority>().0, Some(Fence(1)));
        let gone = DirectoryReply::Head {
            key: DirectoryKey::Realm(config().realm),
            record: None,
        };
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(gone),
        )]);
        assert_eq!(rig.world.resource::<RealmAuthority>().0, None);
    }

    #[test]
    fn a_granted_shard_periodically_re_reads_its_realm_head() {
        // With a re-check interval, a granted shard sends a HeadRead so a revoked
        // lease is OBSERVED (the self-fence reaction is otherwise unreachable).
        let mut rig = Rig::with_config(StubConfig {
            realm_recheck_interval: 2,
            ..config()
        });
        rig.grant_realm();
        let is_head_read = |sent: &[(NodeId, MsgClass, Vec<u8>)]| {
            sent.iter()
                .filter(|(to, _, _)| *to == ORCH)
                .any(|(_, _, bytes)| {
                    let flow: InterShardFlow =
                        postcard::from_bytes(bytes).expect("directory flow decodes");
                    flow == InterShardFlow::Directory(DirectoryOp::HeadRead {
                        key: DirectoryKey::Realm(config().realm),
                    })
                })
        };
        // An EVEN tick (local_tick % 2 == 0) re-reads the realm head.
        rig.set_local_tick(2);
        assert!(is_head_read(&rig.tick(vec![])), "even tick re-reads");
        // An ODD tick does not (the interval gate's other branch).
        rig.set_local_tick(3);
        assert!(!is_head_read(&rig.tick(vec![])), "odd tick is quiet");
    }

    #[test]
    fn logout_revokes_at_the_recorded_entity_fence_not_a_literal() {
        // FENCE-1/5/8: a dot granted at a NON-genesis fence revokes at THAT fence on
        // logout — a hardcoded literal would be Refused and strand the logout.
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach_request(SESSION, GATEWAY);
        let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
        // The directory granted the entity at fence 5 (a transfer advanced it).
        let granted_at_5 = DirectoryReply::Head {
            key: DirectoryKey::Entity(entity),
            record: Some(vd_wire::seams::directory::OwnerRecord {
                authority: AuthorityRef::Shard(SHARD),
                fence: Fence(5),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        };
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(granted_at_5),
        )]);
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].entity_fence,
            Fence(5)
        );
        // Detach, then the retry driver revokes at the RECORDED fence 5.
        let detach = GatewayToShard::DetachSession {
            session: SESSION,
            fence: Fence(1),
        };
        let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &detach)]);
        let sent = rig.tick(vec![]);
        let expected_revoke = InterShardFlow::Directory(DirectoryOp::LeaseRevoke {
            key: DirectoryKey::Entity(entity),
            fence: Fence(5),
        });
        let to_orch: Vec<InterShardFlow> = sent
            .iter()
            .filter(|(to, _, _)| *to == ORCH)
            .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("flow decodes"))
            .collect();
        assert!(
            to_orch.contains(&expected_revoke),
            "revoke at the recorded fence 5, not a literal: {to_orch:?}"
        );
    }

    #[test]
    fn non_head_directory_replies_are_ignored() {
        // CAS results / clock answers carry no shard obligation (the catch-all arm).
        let mut rig = Rig::new();
        rig.grant_realm();
        let cas = DirectoryReply::CasResult {
            key: DirectoryKey::Realm(config().realm),
            outcome: vd_wire::seams::directory::CasOutcome::Won {
                new_fence: Fence(9),
            },
        };
        let clock = DirectoryReply::ClockNow {
            universe_tick: UniverseTick(5),
            epoch: vd_core::EpochId(1),
        };
        // A non-reply InterShardFlow arm misdirected to the stub on Saga (a SagaAck — the
        // gateway→saga ack) is also ignored: the stub handles ONLY DirectoryReply, every
        // other arm is a no-op (the dispatch `Ok(_) => return`), never a panic or a decode
        // error. (The real Transfer arm to a dest shard lands at Slice 1d.)
        let stray = InterShardFlow::SagaAck(
            vd_wire::seams::transfer_control::TransferControlAck::Committed {
                transfer: vd_core::TransferId(9),
            },
        );
        let _ = rig.tick(vec![
            wire_msg(ORCH, MsgClass::Saga, &InterShardFlow::DirectoryReply(cas)),
            wire_msg(ORCH, MsgClass::Saga, &InterShardFlow::DirectoryReply(clock)),
            wire_msg(ORCH, MsgClass::Saga, &stray),
        ]);
        // Authority unaffected by non-Head replies and the stray arm.
        assert_eq!(rig.world.resource::<RealmAuthority>().0, Some(Fence(1)));
    }

    #[test]
    fn unrelated_directory_replies_are_ignored() {
        let mut rig = Rig::new();
        // A headless realm read and an entity head: neither grants authority.
        let none_head = DirectoryReply::Head {
            key: DirectoryKey::Realm(config().realm),
            record: None,
        };
        let entity_head = DirectoryReply::Head {
            key: DirectoryKey::Entity(EntityId(1)),
            record: None,
        };
        let _ = rig.tick(vec![
            wire_msg(
                ORCH,
                MsgClass::Saga,
                &InterShardFlow::DirectoryReply(none_head),
            ),
            wire_msg(
                ORCH,
                MsgClass::Saga,
                &InterShardFlow::DirectoryReply(entity_head),
            ),
        ]);
        assert_eq!(rig.world.resource::<RealmAuthority>().0, None);
    }

    #[test]
    fn undecodable_messages_are_survived() {
        let mut rig = Rig::new();
        let garbage = vec![0xFF, 0x00, 0x13, 0x37];
        let _ = rig.tick(vec![
            Inbound::Wire {
                from: ORCH,
                class: MsgClass::Saga,
                bytes: garbage.clone().into(),
            },
            Inbound::Wire {
                from: GATEWAY,
                class: MsgClass::Control,
                bytes: garbage.into(),
            },
            // Non-wire inbound is skipped by the dispatcher.
            Inbound::NodeUnreachable {
                to: GATEWAY,
                class: MsgClass::Snapshot,
                undelivered: MsgId(1),
            },
            // Snapshot/Membership classes carry nothing for the stub dispatcher.
            Inbound::Wire {
                from: GATEWAY,
                class: MsgClass::Snapshot,
                bytes: vec![1].into(),
            },
            Inbound::Wire {
                from: ORCH,
                class: MsgClass::Membership,
                bytes: vec![2].into(),
            },
        ]);
        assert_eq!(rig.world.resource::<Dots>().0.len(), 0);
        // Both decode failures (Saga + Control) are COUNTED, never silent (ROB-E2E-1);
        // the Snapshot/Membership garbage is not decoded by the stub, so it adds nothing.
        assert_eq!(rig.world.resource::<StubStats>().undecodable, 2);
    }

    #[test]
    fn attach_before_realm_grant_is_deferred_and_counted() {
        let mut rig = Rig::new();
        let sent = rig.attach_request(SESSION, GATEWAY);
        // Only the lease re-request went out — no attach reply.
        assert_eq!(sent.len(), 1);
        assert_eq!(rig.world.resource::<StubStats>().attaches_deferred, 1);
        assert_eq!(rig.world.resource::<Dots>().0.len(), 0);
    }

    #[test]
    fn attach_is_two_phase_authority_derives_from_the_directory() {
        let mut rig = Rig::new();
        rig.grant_realm();

        // Phase 1: the attach request spawns a PROVISIONAL dot and asks the
        // directory for its entity grant — no attach reply, no frames yet.
        let sent = rig.attach_request(SESSION, GATEWAY);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert!(!dot.granted, "provisional until the directory records it");
        assert_eq!(dot.account, AccountId(5));
        assert_eq!(dot.gateway, GATEWAY);
        assert_eq!(dot.entity.kind_tag(), EntityKind::Player as u8);
        assert_eq!(dot.entity.mint_shard(), 10, "minted by THIS shard");
        let grant: InterShardFlow = postcard::from_bytes(&sent[0].2).expect("decode");
        assert_eq!(
            grant,
            InterShardFlow::Directory(DirectoryOp::LeaseGrant {
                key: DirectoryKey::Entity(dot.entity),
                owner: AuthorityRef::Shard(SHARD),
                fence: Fence(1),
            })
        );
        assert!(sent.iter().all(|(to, _, _)| *to == ORCH), "directory only");
        assert_eq!(
            decode_frames(&sent).len(),
            0,
            "provisional dots are invisible"
        );
        // The grant is retried every tick until confirmed (idempotent by fence).
        let sent = rig.tick(vec![]);
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].0, ORCH);

        // Phase 2: the grant confirmation makes the dot HELD: SessionAttached
        // (with the REAL realm fence) and the first frame flow the same tick.
        // (The retry system also fires one last pre-grant request that tick.)
        let sent = rig.confirm_entity_grant(SESSION);
        assert!(rig.world.resource::<Dots>().0[&SESSION].granted);
        let to_gateway: Vec<ShardToGateway> = sent
            .iter()
            .filter(|(to, class, _)| (*to == GATEWAY) & (*class == MsgClass::Control))
            .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("decode"))
            .collect();
        assert_eq!(
            to_gateway,
            vec![ShardToGateway::SessionAttached {
                session: SESSION,
                entity: dot.entity,
                frame: config().frame,
                realm_fence: Fence(1),
            }]
        );
        assert_eq!(decode_frames(&sent).len(), 1, "held dots render");
        // A duplicate grant head is idempotent (no second attach reply).
        let sent = rig.confirm_entity_grant(SESSION);
        let attach_replies = sent
            .iter()
            .filter(|(_, class, _)| *class == MsgClass::Control)
            .count();
        assert_eq!(attach_replies, 0);
    }

    #[test]
    fn foreign_entity_grants_and_pregrant_races_are_survived() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach_request(SESSION, GATEWAY);
        let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
        // The directory says ANOTHER shard owns the entity: loud, not granted.
        let foreign = DirectoryReply::Head {
            key: DirectoryKey::Entity(entity),
            record: Some(vd_wire::seams::directory::OwnerRecord {
                authority: AuthorityRef::Shard(NodeId(99)),
                fence: Fence(1),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        };
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(foreign),
        )]);
        assert!(!rig.world.resource::<Dots>().0[&SESSION].granted);
    }

    #[test]
    fn entity_grant_racing_ahead_of_the_realm_lease_waits_for_retry() {
        // The realm lease is NOT granted yet; an entity head arriving anyway
        // cannot activate the dot (no realm fence to stamp) — the per-tick retry
        // resolves it once the realm lease lands.
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach_request(SESSION, GATEWAY);
        let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
        rig.world.resource_mut::<RealmAuthority>().0 = None;
        let head = DirectoryReply::Head {
            key: DirectoryKey::Entity(entity),
            record: Some(vd_wire::seams::directory::OwnerRecord {
                authority: AuthorityRef::Shard(SHARD),
                fence: Fence(1),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        };
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(head),
        )]);
        assert!(!rig.world.resource::<Dots>().0[&SESSION].granted);
    }

    #[test]
    fn reattach_is_idempotent_and_a_higher_fence_upgrades() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        let first_entity = rig.world.resource::<Dots>().0[&SESSION].entity;
        // Same-fence re-attach: same entity, still one dot.
        let _ = rig.attach();
        assert_eq!(rig.world.resource::<Dots>().0.len(), 1);
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].entity,
            first_entity
        );
        // Higher-fence re-attach upgrades the stored fence.
        let msg = GatewayToShard::AttachSession {
            session: SESSION,
            fence: Fence(3),
            account: AccountId(5),
        };
        let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &msg)]);
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].session_fence,
            Fence(3)
        );
    }

    #[test]
    fn applied_input_moves_the_dot_and_is_logged() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        // Forward input, yaw 0: heading is -Z.
        let _ = rig.tick(vec![input_msg(1, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        let expected_step = 2.0 * 0.05; // speed * dt
        assert!((dot.pose.pos.z + expected_step).abs() < 1e-12, "moved -Z");
        assert_eq!(dot.pose.pos.x, 0.0);
        assert_eq!(dot.last_applied_seq, Some(1));
        assert_eq!(
            rig.world.resource::<InputLog>().applied(),
            vec![(SESSION, 1)]
        );
        // Velocity is displacement over dt.
        assert!((dot.pose.vel.z + 2.0).abs() < 1e-12);
    }

    #[test]
    fn yaw_rotates_the_heading() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        // Look 90° left (yaw = +π/2), then walk forward: heading becomes -X.
        let half_pi = std::f32::consts::FRAC_PI_2;
        let _ = rig.tick(vec![input_msg(
            1,
            Fence(1),
            [1.0, 0.0, 0.0],
            [half_pi, 0.0],
        )]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        let expected_step = 2.0 * 0.05;
        assert!((dot.pose.pos.x + expected_step).abs() < 1e-6, "moved -X");
        assert!(dot.pose.pos.z.abs() < 1e-6);
    }

    #[test]
    fn pitch_is_clamped_at_the_gimbal_pole_never_wraps_past_vertical() {
        // WB-1: a huge look-up delta must NOT accumulate past ±π/2 (which would flip the
        // authoritative orientation). Two big up-pitches in a row stay clamped at the limit.
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        let _ = rig.tick(vec![input_msg(1, Fence(1), [0.0, 0.0, 0.0], [0.0, 3.0])]);
        let _ = rig.tick(vec![input_msg(2, Fence(1), [0.0, 0.0, 0.0], [0.0, 3.0])]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert_eq!(
            dot.pitch,
            vd_core::kinematics::PITCH_LIMIT,
            "accumulated pitch is held at the limit, never wrapped past vertical"
        );
    }

    #[test]
    fn strafe_and_vertical_axes_integrate() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        // Strafe right + up, no forward; clamp catches the out-of-range axis.
        let _ = rig.tick(vec![input_msg(1, Fence(1), [0.0, 2.0, 1.0], [0.0, 0.0])]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        let expected_step = 2.0 * 0.05;
        assert!(
            (dot.pose.pos.x - expected_step).abs() < 1e-12,
            "clamped strafe"
        );
        assert!((dot.pose.pos.y - expected_step).abs() < 1e-12, "vertical");
    }

    #[test]
    fn every_discard_reason_is_logged() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        let _ = rig.tick(vec![input_msg(5, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);

        // DuplicateSeq: same seq again.
        let _ = rig.tick(vec![input_msg(5, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
        // StaleFence: fence below the session's.
        let _ = rig.tick(vec![input_msg(
            6,
            Fence::GENESIS,
            [1.0, 0.0, 0.0],
            [0.0, 0.0],
        )]);
        // MalformedInput: undecodable payload.
        let bad = GatewayToShard::SessionInput {
            session: SESSION,
            fence: Fence(1),
            input_bytes: vec![0xFF],
        };
        let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Input, &bad)]);
        // UnknownSession.
        let unknown = GatewayToShard::SessionInput {
            session: SessionId(0xBB),
            fence: Fence(1),
            input_bytes: vec![],
        };
        let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Input, &unknown)]);
        // PendingAuthority: input for a provisional (ungranted) dot.
        let _ = rig.attach_request(SessionId(0xCC), GATEWAY);
        let pending = GatewayToShard::SessionInput {
            session: SessionId(0xCC),
            fence: Fence(1),
            input_bytes: vec![],
        };
        let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Input, &pending)]);
        // NonFiniteInput: a forged NaN component is caught by the finite gate.
        let _ = rig.tick(vec![input_msg(
            6,
            Fence(1),
            [f32::NAN, 0.0, 0.0],
            [0.0, 0.0],
        )]);

        let log = rig.world.resource::<InputLog>();
        assert_eq!(log.applied(), vec![(SESSION, 5)]);
        assert_eq!(
            log.discarded(),
            vec![
                (SESSION, Some(5), DiscardReason::DuplicateSeq),
                (SESSION, None, DiscardReason::StaleFence),
                (SESSION, None, DiscardReason::MalformedInput),
                (SessionId(0xBB), None, DiscardReason::UnknownSession),
                (SessionId(0xCC), None, DiscardReason::PendingAuthority),
                (SESSION, Some(6), DiscardReason::NonFiniteInput),
            ]
        );
    }

    #[test]
    fn non_finite_input_is_discarded_and_never_poisons_the_pose() {
        // ROB-1 (whole-codebase audit): a forged/corrupt NaN or Inf input must NEVER
        // integrate — NaN sticks in the authoritative pose forever and fans out to every
        // observer. The finite gate discards + counts it, the pose stays untouched, and
        // the seq does NOT advance (the input was never applied), so a subsequent FINITE
        // datagram at the same seq applies normally — the session is not wedged.
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        let _ = rig.tick(vec![input_msg(
            1,
            Fence(1),
            [f32::NAN, 0.0, 0.0],
            [f32::INFINITY, 0.0],
        )]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert_eq!(dot.pose.pos, vd_core::glam::DVec3::ZERO, "pose untouched");
        // Split asserts (no `&&` short-circuit branch — the HR5 coverage discipline).
        assert_eq!(dot.yaw, 0.0, "yaw untouched");
        assert_eq!(dot.pitch, 0.0, "pitch untouched");

        // The same seq, now finite: applies (the poisoned datagram never consumed it).
        let _ = rig.tick(vec![input_msg(1, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
        let log = rig.world.resource::<InputLog>();
        assert_eq!(log.applied(), vec![(SESSION, 1)]);
        assert_eq!(
            log.discarded(),
            vec![(SESSION, Some(1), DiscardReason::NonFiniteInput)]
        );
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert!(dot.pose.pos.is_finite(), "authoritative pose finite");
        assert!(
            dot.pose.pos.z < 0.0,
            "the finite input integrated (moved -Z)"
        );
    }

    #[test]
    fn a_higher_input_fence_upgrades_the_session() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        let _ = rig.tick(vec![input_msg(1, Fence(4), [0.0, 0.0, 0.0], [0.0, 0.0])]);
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].session_fence,
            Fence(4)
        );
        assert_eq!(
            rig.world.resource::<InputLog>().applied(),
            vec![(SESSION, 1)]
        );
    }

    #[test]
    fn detach_is_two_phase_release_via_the_directory() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
        let detach = GatewayToShard::DetachSession {
            session: SESSION,
            fence: Fence(1),
        };
        // Phase 1: the dot stays HELD (departing); the retry driver sends the
        // revoke on the following tick (and every tick until confirmed).
        let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &detach)]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert!(dot.departing, "held until the directory releases it");
        let sent = rig.tick(vec![]);
        let revokes: Vec<InterShardFlow> = sent
            .iter()
            .filter(|(to, _, _)| *to == ORCH)
            .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("decode"))
            .collect();
        assert!(
            revokes.contains(&InterShardFlow::Directory(DirectoryOp::LeaseRevoke {
                key: DirectoryKey::Entity(entity),
                fence: Fence(1),
            })),
            "the entity revoke is on its way"
        );
        // Departing dots no longer consume input.
        let _ = rig.tick(vec![input_msg(9, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
        assert_eq!(
            rig.world.resource::<InputLog>().discarded().last().copied(),
            Some((SESSION, None, DiscardReason::Departing))
        );
        // Phase 2: the headless entity head confirms the revoke — despawn + reply.
        let gone = DirectoryReply::Head {
            key: DirectoryKey::Entity(entity),
            record: None,
        };
        let sent = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(gone),
        )]);
        assert_eq!(rig.world.resource::<Dots>().0.len(), 0);
        let confirms: Vec<ShardToGateway> = sent
            .iter()
            .filter(|(to, class, _)| (*to == GATEWAY) & (*class == MsgClass::Control))
            .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("decode"))
            .collect();
        assert_eq!(
            confirms,
            vec![ShardToGateway::SessionDetached { session: SESSION }]
        );
        // Unknown-session detach still confirms immediately (idempotent).
        let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &detach)]);
        let confirm: ShardToGateway = postcard::from_bytes(&sent[0].2).expect("decode");
        assert_eq!(
            confirm,
            ShardToGateway::SessionDetached { session: SESSION }
        );
        assert_eq!(
            confirm.into_snapshot_bytes(),
            None,
            "only frames carry snapshot bytes"
        );
        // A provisional (ungranted) dot detaches immediately — no record exists.
        let _ = rig.attach_request(SessionId(0xDD), GATEWAY);
        let detach_pending = GatewayToShard::DetachSession {
            session: SessionId(0xDD),
            fence: Fence(1),
        };
        let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &detach_pending)]);
        assert!(
            !rig.world
                .resource::<Dots>()
                .0
                .contains_key(&SessionId(0xDD))
        );
        let confirms = sent
            .iter()
            .filter(|(to, class, _)| (*to == GATEWAY) & (*class == MsgClass::Control))
            .count();
        assert_eq!(confirms, 1);
    }

    #[test]
    fn stale_fence_detach_is_discarded_with_reason() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        // Upgrade the session fence, then detach with the old one.
        let _ = rig.tick(vec![input_msg(1, Fence(4), [0.0, 0.0, 0.0], [0.0, 0.0])]);
        let stale = GatewayToShard::DetachSession {
            session: SESSION,
            fence: Fence(1),
        };
        let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &stale)]);
        assert_eq!(rig.world.resource::<Dots>().0.len(), 1, "dot survives");
        let log = rig.world.resource::<InputLog>();
        assert_eq!(
            log.discarded().last().copied(),
            Some((SESSION, None, DiscardReason::StaleFence))
        );
    }

    #[test]
    fn frames_carry_all_dots_and_count_monotonically() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        // A second session through the same gateway (full grant flow).
        let _ = rig.attach_request(SessionId(0xBB), GATEWAY);
        let _ = rig.confirm_entity_grant(SessionId(0xBB));
        let sent = rig.tick(vec![]);
        let frames = decode_frames(&sent);
        assert_eq!(frames.len(), 1, "one gateway, one frame");
        let snap = &frames[0];
        assert_eq!(snap.sub, SubId(0));
        assert_eq!(snap.entities.len(), 2, "both dots present");
        assert_eq!(snap.source_tick, vd_core::TickId(1));
        assert_eq!(snap.universe_tick, UniverseTick(100));
        // Frame ids increment.
        let next = decode_frames(&rig.tick(vec![]));
        assert_eq!(next[0].frame_id, snap.frame_id + 1);
    }

    #[test]
    fn a_large_world_partitions_into_multiple_under_budget_frames() {
        // GW-1: many dots exceed the datagram budget, so the snapshot ships as
        // several same-frame_id sibling chunks, each encoding under the budget,
        // together carrying EVERY entity (no silent MTU drop).
        let mut rig = Rig::with_config(StubConfig {
            snapshot_datagram_budget: 300,
            ..config()
        });
        rig.grant_realm();
        // Insert 12 granted dots directly (bypassing the attach handshake).
        {
            let mut dots = rig.world.resource_mut::<Dots>();
            for n in 0..12u64 {
                dots.0.insert(
                    SessionId(u128::from(n) + 1),
                    Dot {
                        entity: EntityId::pack(EntityKind::Player, 10, n, n as u32),
                        account: AccountId(n as u128),
                        session_fence: Fence(1),
                        gateway: GATEWAY,
                        granted: true,
                        input_active: false,
                        adopting: false,
                        render_ready: true,
                        departing: false,
                        entity_fence: Fence(1),
                        pose: StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(100)),
                        yaw: 0.0,
                        pitch: 0.0,
                        last_applied_seq: None,
                    },
                );
            }
        }
        let frames = decode_frames(&rig.tick(vec![]));
        assert!(
            frames.len() > 1,
            "12 dots must partition into multiple frames"
        );
        // Every chunk is under budget and the union is all 12 entities.
        let mut all_entities = std::collections::BTreeSet::new();
        for f in &frames {
            let encoded = postcard::to_allocvec(f).expect("encode").len();
            assert!(encoded <= 300, "chunk encodes to {encoded} > 300");
            for e in &f.entities {
                all_entities.insert(e.entity);
            }
        }
        assert_eq!(all_entities.len(), 12, "no entity lost across chunks");
        // §6.3: every chunk of ONE tick shares the SAME frame_id (each self-contained
        // latest-wins) so a reordered sibling chunk is never dropped as stale.
        let ids: std::collections::BTreeSet<u64> = frames.iter().map(|f| f.frame_id).collect();
        assert_eq!(ids.len(), 1, "all chunks of one tick share a frame_id");
        let tick0_id = *ids.iter().next().expect("at least one chunk");
        // The next tick's chunks all share a STRICTLY GREATER frame_id: the counter
        // advances exactly once per tick (monotonic between ticks, stable within one).
        let next = decode_frames(&rig.tick(vec![]));
        assert!(next.len() > 1, "still partitioned the next tick");
        let next_ids: std::collections::BTreeSet<u64> = next.iter().map(|f| f.frame_id).collect();
        assert_eq!(
            next_ids.len(),
            1,
            "next tick's chunks also share one frame_id"
        );
        assert_eq!(
            *next_ids.iter().next().expect("chunk"),
            tick0_id + 1,
            "frame_id advances exactly once per tick"
        );
    }

    #[test]
    fn two_gateways_each_get_the_frame() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        let other_gateway = NodeId(21);
        let _ = rig.attach_request(SessionId(0xBB), other_gateway);
        let _ = rig.confirm_entity_grant(SessionId(0xBB));
        let sent = rig.tick(vec![]);
        let snapshot_targets: Vec<NodeId> = sent
            .iter()
            .filter(|(_, class, _)| *class == MsgClass::Snapshot)
            .map(|(to, _, _)| *to)
            .collect();
        assert_eq!(snapshot_targets, vec![GATEWAY, other_gateway]);
    }

    #[test]
    fn minted_entities_are_unique_and_structured() {
        let mut mint = EntityMint {
            seq: 0,
            rng: SplitMix64::new(1),
        };
        let a = mint_entity(&mut mint, SHARD);
        let b = mint_entity(&mut mint, SHARD);
        assert_ne!(a, b);
        assert_eq!(a.seq(), 0);
        assert_eq!(b.seq(), 1);
        assert_eq!(a.kind_tag(), EntityKind::Player as u8);
        assert_eq!(a.mint_shard(), 10);
    }

    #[test]
    fn input_log_is_a_bounded_window_with_exact_totals() {
        // SCALE-3: a small window holds only the NEWEST entries (no unbounded
        // growth), while the totals are EXACT and evictions are counted.
        let mut log = InputLog::new(3);
        for seq in 0..10u64 {
            log.record_applied(SessionId(1), seq);
        }
        for seq in 0..4u64 {
            log.record_discarded(SessionId(2), Some(seq), DiscardReason::DuplicateSeq);
        }
        // Window holds the last 3 of each; totals count everything.
        assert_eq!(log.applied().len(), 3);
        assert_eq!(
            log.applied(),
            vec![(SessionId(1), 7), (SessionId(1), 8), (SessionId(1), 9)]
        );
        assert_eq!(log.discarded().len(), 3);
        assert_eq!(log.applied_total, 10);
        assert_eq!(log.discarded_total, 4);
        // 7 applied + 1 discarded evicted from the windows.
        assert_eq!(log.window_evictions, 8);
        // Capacity floors at 1.
        let mut tiny = InputLog::new(0);
        tiny.record_applied(SessionId(9), 1);
        tiny.record_applied(SessionId(9), 2);
        assert_eq!(tiny.applied(), vec![(SessionId(9), 2)]);
    }

    // ---- Slice 1c.5: the dest-side OpenInputSlot (transfer-destination input slot) ------

    /// The transfer subject the OpenInputSlot test helpers carry (the avatar the dest adopts).
    const SUBJECT: EntityId = EntityId(0xBEEF);

    fn open_input_slot_subj(
        session: SessionId,
        gateway: NodeId,
        resume_from_seq: u64,
        fence: Fence,
        subject: DirectoryKey,
    ) -> Inbound {
        wire_msg(
            gateway,
            MsgClass::Control,
            &GatewayToShard::OpenInputSlot {
                session,
                fence,
                account: AccountId(5),
                resume_from_seq,
                subject,
            },
        )
    }

    fn open_input_slot_f(
        session: SessionId,
        gateway: NodeId,
        resume_from_seq: u64,
        fence: Fence,
    ) -> Inbound {
        open_input_slot_subj(
            session,
            gateway,
            resume_from_seq,
            fence,
            DirectoryKey::Entity(SUBJECT),
        )
    }

    fn open_input_slot(session: SessionId, gateway: NodeId, resume_from_seq: u64) -> Inbound {
        open_input_slot_f(session, gateway, resume_from_seq, Fence(1))
    }

    fn input_for(session: SessionId, seq: u64, gateway: NodeId) -> Inbound {
        wire_msg(
            gateway,
            MsgClass::Input,
            &GatewayToShard::SessionInput {
                session,
                fence: Fence(1),
                input_bytes: postcard::to_allocvec(&InputDatagram {
                    seq,
                    is_cut_marker: false,
                    client_tick: vd_core::TickId(2),
                    movement: [1.0, 0.0, 0.0],
                    look: [0.0, 0.0],
                    action_bits: 0,
                })
                .expect("encode"),
            },
        )
    }

    #[test]
    fn open_input_slot_adopts_the_subject_input_active_without_attaching() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let sent = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert!(dot.input_active, "the slot is input-active");
        // 1c.8: the dot ADOPTS the subject — its entity IS the subject id (not a fresh mint),
        // it is `adopting`, NOT granted (the adopt HeadRead flips that), and NOT render-ready.
        assert_eq!(dot.entity, SUBJECT, "the dot adopted the transfer subject");
        assert!(dot.adopting, "the dot is a transfer-dest adopt");
        assert!(
            !dot.granted,
            "the adopt HeadRead has not flipped granted yet"
        );
        assert!(
            !dot.render_ready,
            "the adopt carries no pose — renders nothing"
        );
        assert_eq!(
            dot.entity_fence,
            Fence::GENESIS,
            "the adopt HeadRead fills the real fence"
        );
        assert_eq!(
            dot.last_applied_seq,
            Some(5),
            "seeded to the resume watermark"
        );
        assert_eq!(
            rig.world.resource::<StubStats>().last_input_slot_resume,
            Some(5),
            "the as-received resume watermark is latched for the conservation gate"
        );
        // The slot is a SILENT inbound state change: it emits NOTHING — no SessionAttached,
        // no re-home (the source still owns the client connection — R2), and the provisional
        // dot is not render-ready so it renders no snapshot frame either.
        assert!(sent.is_empty(), "the input slot emits nothing back");
    }

    #[test]
    fn an_adopting_dot_head_reads_the_record_never_lease_grants() {
        // 1c.8 HR5: the request_pending_grants 3-way ADOPT arm — an adopting !granted dot emits
        // HeadRead{Entity} (to adopt the record the CAS moved here), NEVER a LeaseGrant (which
        // the directory would Refuse at the post-genesis CAS fence).
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
        let sent = rig.tick(vec![]);
        let to_orch: Vec<InterShardFlow> = sent
            .iter()
            .filter(|(to, _, _)| *to == ORCH)
            .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("flow decodes"))
            .collect();
        assert!(
            to_orch.contains(&InterShardFlow::Directory(DirectoryOp::HeadRead {
                key: DirectoryKey::Entity(SUBJECT),
            })),
            "the adopting dot HeadReads its subject: {to_orch:?}"
        );
        // Value-compare (NOT matches!, whose match-success arm would be an uncoverable region):
        // the EXACT LeaseGrant an adopting dot must NEVER send (it adopts via HeadRead instead).
        assert!(
            !to_orch.contains(&InterShardFlow::Directory(DirectoryOp::LeaseGrant {
                key: DirectoryKey::Entity(SUBJECT),
                owner: AuthorityRef::Shard(SHARD),
                fence: Fence::GENESIS.next(),
            })),
            "an adopting dot NEVER LeaseGrants its entity: {to_orch:?}"
        );
    }

    #[test]
    fn the_adopt_grant_flip_holds_authority_without_render_or_attach() {
        // 1c.8: the grant-flip on the adopted record sets granted (authority-held) AND stamps
        // entity_fence = the recorded CAS fence, but leaves render_ready FALSE and pushes NO
        // SessionAttached (R2 — the source owns the client). adopting is cleared on the flip.
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
        // The directory record the CAS moved here: SUBJECT @ this shard @ a NON-genesis fence.
        let adopted = DirectoryReply::Head {
            key: DirectoryKey::Entity(SUBJECT),
            record: Some(vd_wire::seams::directory::OwnerRecord {
                authority: AuthorityRef::Shard(SHARD),
                fence: Fence(2),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        };
        let sent = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(adopted),
        )]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        // Split asserts (each &&-short-circuit false arm is uncoverable — HR5).
        assert!(dot.granted, "the adopt flips granted (authority-held)");
        assert_eq!(dot.entity_fence, Fence(2), "stamped the recorded CAS fence");
        assert!(!dot.render_ready, "an adopt renders NOTHING (no pose; 1d)");
        assert!(!dot.adopting, "adopting is cleared on the flip");
        // No SessionAttached push (suppressed when adopting) and no frame (not render-ready).
        let attach_replies = sent
            .iter()
            .filter(|(_, class, _)| *class == MsgClass::Control)
            .count();
        assert_eq!(attach_replies, 0, "no SessionAttached on an adopt (R2)");
        assert_eq!(
            decode_frames(&sent).len(),
            0,
            "an adopted dot renders nothing"
        );
    }

    #[test]
    fn a_non_entity_subject_open_input_slot_is_a_counted_noop() {
        // 1c.8 HR5: the subject_entity else arm — a non-Entity subject (e.g. a Realm saga driven
        // through CommitAuthority) does NOT adopt; it is a counted no-op, never an extraction panic.
        let mut rig = Rig::new();
        rig.grant_realm();
        let sent = rig.tick(vec![open_input_slot_subj(
            SESSION,
            GATEWAY,
            5,
            Fence(1),
            DirectoryKey::Realm(RealmId::System(99)),
        )]);
        assert!(
            !rig.world.resource::<Dots>().0.contains_key(&SESSION),
            "a non-Entity subject mints no dot"
        );
        assert_eq!(rig.world.resource::<StubStats>().input_slots_malformed, 1);
        assert!(sent.is_empty(), "the no-op emits nothing");
    }

    #[test]
    fn the_source_self_fences_a_foreign_owned_entity_without_revoking() {
        // 1c.8 SOURCE self-fence: a granted dot whose directory head names a FOREIGN owner (the
        // transfer CAS moved it to the dest) is DROPPED LOCALLY — NO LeaseRevoke (which would be
        // Refused at the stale fence, or delete the dest's record at the new fence).
        //
        // ⚠️ EXISTS-TO-BE-FLIPPED (DEFERRED D-2 ordering inversion): this asserts the INTERIM
        // local DROP (`dots.remove`). The proper fence-first, saga-pushed demote (post-1d
        // band/ghost slice) instead flips the source to a RETAINED ghost-as-collider
        // (`authority.rs` Owned→Frozen→Ghost) demote-BEFORE-promote — so when that lands the dot
        // must NOT vanish (it becomes a ghost), and this assertion + `self_fence_foreign_entity`
        // + the granted-key poll are RIPPED OUT (a tear-out, not a predicate body-swap).
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach(); // a granted, locally-owned dot
        let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
        let foreign = DirectoryReply::Head {
            key: DirectoryKey::Entity(entity),
            record: Some(vd_wire::seams::directory::OwnerRecord {
                authority: AuthorityRef::Shard(NodeId(99)), // the transfer DEST
                fence: Fence(2),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        };
        let sent = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(foreign),
        )]);
        assert!(
            !rig.world.resource::<Dots>().0.contains_key(&SESSION),
            "the source self-demotes (drops) the transferred dot"
        );
        // The drop is LOCAL and IMMEDIATE: the dot is gone in the SAME tick as the foreign-owner
        // reply (asserted above). A revoke-based removal (the logout two-phase path) would instead
        // set `departing` and KEEP the dot until the directory confirms a LeaseRevoke a round-trip
        // later — so single-tick removal IS the proof that NO directory write (no LeaseRevoke) was
        // issued; revoking the record (now the dest's) would orphan the player.
        assert!(
            sent.is_empty(),
            "the self-fence drop is purely local — no directory write at all: {sent:?}"
        );
        // Idempotent: a second foreign-owner reply finds no matching dot — a clean no-op.
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(foreign),
        )]);
        assert!(!rig.world.resource::<Dots>().0.contains_key(&SESSION));
    }

    #[test]
    fn a_granted_source_polls_its_entity_keys_on_the_recheck_cadence() {
        // 1c.8 SOURCE granted-key poll: with a non-zero recheck interval, a granted dot emits a
        // HeadRead{Entity} on the cadence so a foreign takeover is OBSERVED (otherwise the source
        // self-fence is unreachable). The OFF arm (non-multiple tick) sends no entity HeadRead.
        let mut rig = Rig::with_config(StubConfig {
            realm_recheck_interval: 2,
            ..config()
        });
        rig.grant_realm();
        let _ = rig.attach();
        let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
        let entity_head_read = |sent: &[(NodeId, MsgClass, Vec<u8>)]| {
            sent.iter()
                .filter(|(to, _, _)| *to == ORCH)
                .any(|(_, _, bytes)| {
                    postcard::from_bytes::<InterShardFlow>(bytes).expect("decode")
                        == InterShardFlow::Directory(DirectoryOp::HeadRead {
                            key: DirectoryKey::Entity(entity),
                        })
                })
        };
        rig.set_local_tick(2);
        assert!(
            entity_head_read(&rig.tick(vec![])),
            "an on-cadence tick polls the entity key"
        );
        rig.set_local_tick(3);
        assert!(
            !entity_head_read(&rig.tick(vec![])),
            "an off-cadence tick does not"
        );
    }

    #[test]
    fn the_dest_applies_post_marker_input_and_rejects_replays_at_the_marker() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 10)]); // watermark = 10
        // marker+1, marker+2 apply in order; a seq <= marker is a counted DuplicateSeq.
        let _ = rig.tick(vec![
            input_for(SESSION, 11, GATEWAY),
            input_for(SESSION, 12, GATEWAY),
            input_for(SESSION, 9, GATEWAY),
        ]);
        let log = rig.world.resource::<InputLog>();
        assert_eq!(log.applied(), vec![(SESSION, 11), (SESSION, 12)]);
        assert_eq!(
            log.discarded(),
            vec![(SESSION, Some(9), DiscardReason::DuplicateSeq)],
            "a seq <= marker was already applied at the source"
        );
    }

    #[test]
    fn open_input_slot_before_the_realm_lease_is_deferred_and_counted() {
        let mut rig = Rig::new(); // NO grant_realm
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
        assert!(
            !rig.world.resource::<Dots>().0.contains_key(&SESSION),
            "no slot yet"
        );
        assert_eq!(rig.world.resource::<StubStats>().input_slots_deferred, 1);
    }

    #[test]
    fn open_input_slot_max_merges_the_watermark_and_guards_a_granted_dot() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 20)]);
        // A re-sent slot with a LOWER watermark never lowers it (max-merge).
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].last_applied_seq,
            Some(20)
        );
        // A slot from a NON-owning gateway does not touch the dot (security guard false arm).
        let other_gateway = NodeId(999);
        let _ = rig.tick(vec![open_input_slot(SESSION, other_gateway, 100)]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert_eq!(
            dot.last_applied_seq,
            Some(20),
            "a foreign gateway cannot move the watermark"
        );
        assert_eq!(dot.gateway, GATEWAY, "ownership unchanged");
    }

    #[test]
    fn open_input_slot_bumps_the_session_fence_then_is_inert_on_a_granted_dot() {
        let mut rig = Rig::new();
        rig.grant_realm();
        // Mint the provisional slot at fence 1, watermark 5.
        let _ = rig.tick(vec![open_input_slot_f(SESSION, GATEWAY, 5, Fence(1))]);
        // A slot at a HIGHER fence (as 1d/P3 will re-issue under a fresher realm lease) bumps
        // the session fence and MAX-merges the watermark.
        let _ = rig.tick(vec![open_input_slot_f(SESSION, GATEWAY, 7, Fence(4))]);
        {
            let dot = rig.world.resource::<Dots>().0[&SESSION];
            assert_eq!(dot.session_fence, Fence(4), "bumped to the fresher lease");
            assert_eq!(dot.last_applied_seq, Some(7), "watermark advanced");
        }
        // A STALE slot (fence below the dot's session fence — a replay or partitioned old
        // gateway) is dropped + counted, never re-arming input (the day-one stale-gateway rule).
        let _ = rig.tick(vec![open_input_slot_f(SESSION, GATEWAY, 999, Fence(1))]);
        {
            let dot = rig.world.resource::<Dots>().0[&SESSION];
            assert_eq!(
                dot.session_fence,
                Fence(4),
                "stale slot does not touch the fence"
            );
            assert_eq!(
                dot.last_applied_seq,
                Some(7),
                "stale slot does not move the watermark"
            );
            assert_eq!(rig.world.resource::<StubStats>().input_slots_stale, 1);
        }
        // Once the dot is GRANTED (the 1d promotion), a stray OpenInputSlot is inert —
        // the granted entity owns its own input watermark (the `!granted` guard false arm).
        rig.world
            .resource_mut::<Dots>()
            .0
            .get_mut(&SESSION)
            .expect("the provisional dot was minted above")
            .granted = true;
        let _ = rig.tick(vec![open_input_slot_f(SESSION, GATEWAY, 999, Fence(9))]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert_eq!(dot.session_fence, Fence(4), "granted dot's fence untouched");
        assert_eq!(
            dot.last_applied_seq,
            Some(7),
            "granted dot's watermark untouched"
        );
    }
}
