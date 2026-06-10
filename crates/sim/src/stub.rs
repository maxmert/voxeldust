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
    for dot in dots.0.values() {
        if !dot.granted {
            // A provisional entity is always requested at the genesis-next fence (a
            // never-recorded entity); the RECORDED fence (entity_fence) is what the
            // revoke uses, which is the FENCE-1/5/8 fix.
            let op = DirectoryOp::LeaseGrant {
                key: DirectoryKey::Entity(dot.entity),
                owner: AuthorityRef::Shard(identity.node_id),
                fence: Fence::GENESIS.next(),
            };
            outbox.push_flow(
                config.orchestrator,
                MsgClass::Saga,
                &InterShardFlow::Directory(op),
            );
        } else if dot.departing {
            // Revoke at the RECORDED fence (FENCE-1/5/8): a hardcoded literal would
            // be Refused once any transfer advanced the entity's fence, stranding the
            // logout forever.
            let op = DirectoryOp::LeaseRevoke {
                key: DirectoryKey::Entity(dot.entity),
                fence: dot.entity_fence,
            };
            outbox.push_flow(
                config.orchestrator,
                MsgClass::Saga,
                &InterShardFlow::Directory(op),
            );
        }
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
    if !dot.granted {
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
    dot.pitch += f64::from(input.look[1]);
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

/// Handle a directory reply: realm-lease and entity-grant confirmations.
fn on_directory_reply(
    bytes: &[u8],
    identity: &NodeIdentity,
    config: &StubConfig,
    authority: &mut RealmAuthority,
    dots: &mut Dots,
    outbox: &mut OutboundBox,
) {
    let Ok(reply) = postcard::from_bytes::<DirectoryReply>(bytes) else {
        tracing::error!("undecodable directory reply");
        return;
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
            let ours = record.authority == AuthorityRef::Shard(identity.node_id);
            if !ours {
                tracing::error!(
                    "entity {entity} granted to {:?}, not this shard",
                    record.authority
                );
                return;
            }
            for (session, dot) in &mut dots.0 {
                if dot.entity == entity && !dot.granted {
                    dot.granted = true;
                    dot.entity_fence = record.fence; // the recorded authority fence
                    let reply = ShardToGateway::SessionAttached {
                        session: *session,
                        entity,
                        frame: config.frame,
                        realm_fence,
                    };
                    push_session_reply(outbox, dot.gateway, &reply);
                }
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
    let mut gateways: Vec<NodeId> = dots
        .0
        .values()
        .filter(|d| d.granted)
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
        .filter(|d| d.granted)
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
            let bytes = crate::io::bytes(postcard::to_allocvec(&reply).expect("encode"));
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
            self.tick(vec![wire_msg(ORCH, MsgClass::Saga, &reply)])
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
        let _ = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &reply)]);
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
        let _ = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &foreign)]);
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
        let _ = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &gone)]);
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
        let _ = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &granted_at_5)]);
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
        let _ = rig.tick(vec![
            wire_msg(ORCH, MsgClass::Saga, &cas),
            wire_msg(ORCH, MsgClass::Saga, &clock),
        ]);
        // Authority unaffected by non-Head replies.
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
            wire_msg(ORCH, MsgClass::Saga, &none_head),
            wire_msg(ORCH, MsgClass::Saga, &entity_head),
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
        let _ = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &foreign)]);
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
        let _ = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &head)]);
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
        let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &gone)]);
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
}
