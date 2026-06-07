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

use std::collections::BTreeMap;

use bevy_ecs::prelude::{IntoScheduleConfigs, Res, ResMut, Resource, Schedule, World};
use vd_core::entity_kind::EntityKind;
use vd_core::glam::{DQuat, DVec3, EulerRot};
use vd_core::pose::{FrameRef, RealmId, StampedPose};
use vd_core::rng::SplitMix64;
use vd_core::{AccountId, EntityId, Fence, NodeId, SessionId};
use vd_wire::channels::{EntitySnap, InputDatagram, SnapshotDatagram, SubId};
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
}

/// The INPUT-CONSERVATION ground truth: every delivered input is in exactly one of
/// these logs. The oracle audits this against fabric delivery accounting.
#[derive(Resource, Debug, Default)]
pub struct InputLog {
    /// (session, seq) of every applied input, in application order.
    pub applied: Vec<(SessionId, u64)>,
    /// (session, seq-if-decodable, reason) of every discarded input.
    pub discarded: Vec<(SessionId, Option<u64>, DiscardReason)>,
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
    world.insert_resource(InputLog::default());
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
    authority: Res<RealmAuthority>,
    dots: Res<Dots>,
    mut outbox: ResMut<OutboundBox>,
) {
    if authority.0.is_none() {
        let op = DirectoryOp::LeaseGrant {
            key: DirectoryKey::Realm(config.realm),
            owner: AuthorityRef::Shard(identity.node_id),
            fence: Fence::GENESIS.next(),
        };
        push_flow(
            &mut outbox,
            config.orchestrator,
            &InterShardFlow::Directory(op),
        );
    }
    for dot in dots.0.values() {
        if !dot.granted {
            let op = DirectoryOp::LeaseGrant {
                key: DirectoryKey::Entity(dot.entity),
                owner: AuthorityRef::Shard(identity.node_id),
                fence: Fence::GENESIS.next(),
            };
            push_flow(
                &mut outbox,
                config.orchestrator,
                &InterShardFlow::Directory(op),
            );
        } else if dot.departing {
            let op = DirectoryOp::LeaseRevoke {
                key: DirectoryKey::Entity(dot.entity),
                fence: Fence::GENESIS.next(),
            };
            push_flow(
                &mut outbox,
                config.orchestrator,
                &InterShardFlow::Directory(op),
            );
        }
    }
}

/// Encode one inter-shard flow into the outbox (the InterShardFlow encoder is the
/// SOLE producer of shard-bound bytes — HR1).
fn push_flow(outbox: &mut OutboundBox, to: NodeId, flow: &InterShardFlow) {
    let bytes = postcard::to_allocvec(flow).expect("closed wire enums serialize infallibly");
    outbox.0.push((to, MsgClass::Saga, crate::io::bytes(bytes)));
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
                push_flow(
                    outbox,
                    ctx.config.orchestrator,
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
                    log.discarded
                        .push((session, None, DiscardReason::StaleFence));
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
    outbox.0.push((to, MsgClass::Control, crate::io::bytes(bytes)));
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
        log.discarded
            .push((session, None, DiscardReason::UnknownSession));
        return;
    };
    if !dot.granted {
        log.discarded
            .push((session, None, DiscardReason::PendingAuthority));
        return;
    }
    if dot.departing {
        log.discarded
            .push((session, None, DiscardReason::Departing));
        return;
    }
    if fence.is_stale_against(dot.session_fence) {
        log.discarded
            .push((session, None, DiscardReason::StaleFence));
        return;
    }
    // A higher fence means the gateway re-granted (P3 adoption); track it.
    if fence > dot.session_fence {
        dot.session_fence = fence;
    }
    let Ok(input) = postcard::from_bytes::<InputDatagram>(input_bytes) else {
        log.discarded
            .push((session, None, DiscardReason::MalformedInput));
        return;
    };
    if dot.last_applied_seq.is_some_and(|last| input.seq <= last) {
        log.discarded
            .push((session, Some(input.seq), DiscardReason::DuplicateSeq));
        return;
    }
    dot.last_applied_seq = Some(input.seq);
    integrate(dot, &input, config, clock);
    log.applied.push((session, input.seq));
}

/// Kinematic point integration: axes are clamped to [-1, 1], displacement is
/// speed·dt in the dot's yaw-rotated heading. Pure f64 closed-form per tick.
fn integrate(dot: &mut Dot, input: &InputDatagram, config: &StubConfig, clock: &ClockSample) {
    dot.yaw += f64::from(input.look[0]);
    dot.pitch += f64::from(input.look[1]);
    dot.orient_from_angles();
    let axes = DVec3::new(
        f64::from(input.movement[1].clamp(-1.0, 1.0)),
        f64::from(input.movement[2].clamp(-1.0, 1.0)),
        // Forward is -Z in the dot's local frame (right-handed, Y-up).
        -f64::from(input.movement[0].clamp(-1.0, 1.0)),
    );
    let step = dot.pose.orient * axes * (config.move_speed_mps * config.tick_dt_s);
    dot.pose.pos += step;
    dot.pose.vel = step / config.tick_dt_s;
    dot.pose.universe_tick = clock.universe_tick;
}

impl Dot {
    fn orient_from_angles(&mut self) {
        self.pose.orient = DQuat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0);
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
                tracing::error!(
                    "realm lease held by {:?}, not this shard — misconfigured topology",
                    record.authority
                );
            }
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
    let snapshot = SnapshotDatagram {
        // The shard always stamps sub 0; the gateway re-tags per session.
        sub: SubId(0),
        frame_id: counter.0,
        source_tick: clock.local_tick,
        universe_tick: clock.universe_tick,
        entities,
    };
    counter.0 += 1;
    let snapshot_bytes =
        postcard::to_allocvec(&snapshot).expect("closed wire enums serialize infallibly");
    let frame = ShardToGateway::Frame {
        realm_fence,
        source_tick: clock.local_tick,
        snapshot_bytes,
    };
    // ONE shared body, cloned (refcount bump) to every subscribing gateway — never
    // an O(entities) copy per gateway (SCALE-1).
    let bytes = crate::io::bytes(
        postcard::to_allocvec(&frame).expect("closed wire enums serialize infallibly"),
    );
    for gateway in gateways {
        outbox.0.push((gateway, MsgClass::Snapshot, bytes.clone()));
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
        }
    }

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
                node_id: SHARD,
                kind: NodeKind::StubShard,
            });
            world.insert_resource(ClockSample {
                local_tick: vd_core::TickId(1),
                universe_tick: UniverseTick(100),
                epoch: vd_core::EpochId(1),
            });
            let mut schedule = Schedule::default();
            register_stub_shard(&mut world, &mut schedule, config());
            Rig { world, schedule }
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
        assert_eq!(rig.world.resource::<InputLog>().applied, vec![(SESSION, 1)]);
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

        let log = rig.world.resource::<InputLog>();
        assert_eq!(log.applied, vec![(SESSION, 5)]);
        assert_eq!(
            log.discarded,
            vec![
                (SESSION, Some(5), DiscardReason::DuplicateSeq),
                (SESSION, None, DiscardReason::StaleFence),
                (SESSION, None, DiscardReason::MalformedInput),
                (SessionId(0xBB), None, DiscardReason::UnknownSession),
                (SessionId(0xCC), None, DiscardReason::PendingAuthority),
            ]
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
        assert_eq!(rig.world.resource::<InputLog>().applied, vec![(SESSION, 1)]);
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
            rig.world.resource::<InputLog>().discarded.last(),
            Some(&(SESSION, None, DiscardReason::Departing))
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
            log.discarded.last(),
            Some(&(SESSION, None, DiscardReason::StaleFence))
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
}
