//! THE STUB SHARD'S UNIT TIER — the assertions that were the tail of `stub.rs` until the file
//! was split, moved VERBATIM and named identically (every `stub::tests::…` citation in `docs/`
//! still resolves, because the module path did not change: this is still `stub::tests`).
//!
//! Owns: the fixtures and the assertions for every lane in the `stub` tree. It reads `super::*`, so
//! it sees the module's whole re-exported surface exactly as a caller outside the crate does, plus
//! the crate-private items the lanes share.
//!
//! Does NOT own: any production behaviour. None of this is compiled into a shipped shard, and no
//! fixture here may become the only place a rule is stated.

// ===== THE PER-SUBJECT MODULES (slice S10) ==================================================
// This file was ONE 16,139-line module. It now holds the SHARED FIXTURES ONLY — every helper, rig,
// constant and world the assertions are built from — and the assertions themselves live one level
// down, grouped by what they are about.
//
// NOTHING MOVED BUT THE TESTS. No helper was relocated, so no judgement was made about which fixture
// belongs to which subject — the one call that would have broken the build (a helper reported local,
// used in three other subjects) cannot arise when helpers do not move. Test bodies are VERBATIM: the
// cut was proved lossless by reassembling the original file from the pieces before writing anything.
//
// Test NAMES are unchanged, so every `cargo test <name>` filter still selects what it always did.
// The PATHS gain one level (`stub::tests::<subject>::<name>`); the six doc citations that named a
// full path are updated with this change.
/// RLM Step 2, the demand-driven realm lifecycle (banner 11218): the hysteresis state machine that decides SpinUp/KeepAlive/drop per observer, the scalar primitives the demand fold reduces to (live and predictive distance, the cross-observer union, seed-lineage coord recovery, the counted coord-less ship exclusion), the evaluate_realm_aoi fold end to end through the rig, the downward interest byte of the look horizon (fail-closed admission, down-proxy, rising/beat/falling edges, self-proxy exclusion), the SL7 occupied-child liveness bit as a demand input, and the whole loop's latches, G-IDENTICAL spin-up, recheck cadence, parent/child directory head-reads, node caches, and the ChildLive bit shipped upward with its hand-off hold budget.
mod aoi_demand;
/// The realm lease and session admission, merged per reader 1's recommendation: boot requests the lease until granted, a foreign grant is refused, a lost lease self-fences, the holder re-reads and renews realm and entity leases on cadence, logout revokes at the recorded fence; then how a session's dot is admitted — spawn-pose resolution and the two-phase attach handshake through the directory, races, wrong-realm routing, idempotent re-attach.
mod authority;
/// The CONTAINMENT re-home trigger section (banner 6655, running to 11217): the shell/box region forest fixtures, the ancestor chain a shard folds over its own roster, the child index proving it decides what a full scan decides without touching every child (SL9); then the deepest-container scan (rootless forest, dense crowd, first-class Station/Area and co-hosted realms, banner 9143), the shape- and motion-agnostic G-IDENTICAL fixtures, and the crossing lifecycle (banner 9817: D-38 discharge, ledger eviction, cell-carry shape, grant/abort demuxes, ttl re-drive, exhaustion backoff, demote terminal, dock/undock, boot validation).
mod containment;
/// The crossing lane itself: Slice 1d.
mod crossing;
/// The ownership hand-off machinery end to end (banner 3581): the dest-side OpenInputSlot adopt, the applied-steps idempotency journal, the ordered saga Demote/Promote/ReHome consumers including source==dest re-own and the retained-ghost collision, the source-ghost lifecycle (take-over proof, hold closure, band-exit Despawn), then the remaining saga-Promote no-op/defer arms, the deferred promote re-driven after the crossing lands, post-marker input at the resume watermark, the idle re-stamp and the slot's fence/watermark guards.
mod handoff;
/// Occupant input integration and THE SPEED LAW (banner 2831) plus the client-session lane: walk/yaw/pitch-clamp, per-axis clamp and diagonal normalization, the governed ceiling and its ramp, the ceiling applied to a transient, the realm time multiplier; then input-discard reasons, the finite gate, action_bits inertness, session-fence upgrade and stale-fence detach, per-tick snapshot frames and budget partitioning, entity minting and the bounded InputLog.
mod movement;
/// The ONE WRITER: a parent authoring its direct children's placements and every SL1 conversion built on it.
mod placement;
/// The placement-carry derivation (how far a shard may extrapolate a moving placement) and the pose-stamp invariant, plus the RLM 5f RG-1 reactive greeting: a shard announces itself to every booked peer after a silence interval, suppressed per-peer by inbound contact, identical across shard kinds.
mod presence;
/// The whole D-7 transient/debris batched-crossing machinery: status tiers, re-advance of held items, emit/adopt/release/promote/complete, discard/abandon poisoning, re-drive and re-solicit, and the inbound dispatch arms.
mod transients;
/// The Q2 window-relay lane and THE WINDOW LANE Slice A (banner 15096): sealed batches held verbatim with fail-closed admission, forwarded send-on-change and TTL-pruned, byte-identical across two hops, egress and union over-draw measured, the structural proof that production never opens a seal; the ship half (a shard sealing its own self-authored statements up to its resolved parent on resolve/change/cadence) and the read-only accessors; what a realm states about itself on a window (SL3 self-look) and the membership verdict's anti-flicker grace hold; then one frame per tick per open window, the pre-inverted hop row, self-look and point-of-light marker bodies, the send-on-change beat, the derived TTL, the Child-scope guards, the Q1 one-level fence and the membership verdict fold.
mod window_lane;

/// Flatten a lattice position to metres at FINE (every fixture frame here is FINE) — the one
/// test-side reduction, so no assert reads `.offset()` as if it were a position (the habit the
/// activation makes unwritable in production).
fn fm(p: vd_core::pose::LatticePos) -> DVec3 {
    p.delta_m(vd_core::pose::LatticePos::ORIGIN, vd_core::pose::Tier::Fine)
}

/// The same flatten, in a STATED unit — for a position that is not counted in millimetres.
///
/// ★ WHY THIS EXISTS (slice S9). `fm` reads every position at the FINE rung, which was every position
/// there was. A pose handed UP into the galaxy's frame is counted in two-metre steps, and reading it in
/// millimetres is not a small error — it is a factor of 2048, and it looks like a plausible number.
fn fm_at(p: vd_core::pose::LatticePos, tier: vd_core::pose::Tier) -> DVec3 {
    p.delta_m(vd_core::pose::LatticePos::ORIGIN, tier)
}
use super::*;
use crate::authority::Authority;
use crate::capability::NodeKind;
use crate::io::{Durability, Inbound, MsgClass};
use crate::runtime::{ClockSample, InboundBox, NodeIdentity, OutboundBox};
use bevy_ecs::prelude::{Schedule, World};
use glam::I64Vec3;
use std::collections::{BTreeMap, BTreeSet};
use vd_core::entity_kind::{DurabilityClass, EntityKind};
use vd_core::flight::{self};
use vd_core::frame::{FrameError, FramePlacement, transfer_frame};
use vd_core::glam::{DQuat, DVec3};
use vd_core::kinematics::secs_since_epoch;
use vd_core::placement::{MotionFn, PlacementBook, PlacementLedger};
use vd_core::pose::{FrameRef, RealmId, StampedPose, Tier};
use vd_core::realm_coord::RealmCoord;
use vd_core::realm_path::{RealmLevel, RealmPath};
use vd_core::rng::SplitMix64;
use vd_core::worldgen::level_of;
use vd_core::{AccountId, EntityId, Fence, NodeId, SessionId, TickId, TransferId};
use vd_wire::channels::{InputDatagram, RealmShape, RealmSnap, SnapshotDatagram, SubId};
use vd_wire::intershard::{
    CrossingAborted, CrossingRequest, DemandVerb, DemoteCmd, FlushSource, GhostFlow,
    InterShardFlow, PROMOTE_STEP, PromoteCmd, RE_HOME_STEP, ReHomeCmd, ReHomeState, RealmDemand,
    TRANSFER_SCHEMA_VERSION, TRANSIENT_ABANDON_STEP, TRANSIENT_BATCH_STEP, TRANSIENT_COMPLETE_STEP,
    TRANSIENT_DISCARD_STEP, TRANSIENT_DROP_STEP, TRANSIENT_RELEASE_STEP, TransferAck,
    TransferEnvelope, TransientCrossingGrant, TransientCrossingRequest, TransientHandoff,
    TransientItem, TransitionPayload, crossing_transfer_id,
};
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey, DirectoryOp, DirectoryReply};
use vd_wire::seams::transfer_control::TransferControlAck;
use vd_wire::session_flow::{BodyStmt, GatewayToShard, ShardToGateway, WindowId, WindowScope}; // only the tests name a cell anchor directly (prod poses ride at cell ZERO)
// DEV-ONLY motion (SL4): fixtures plant real Kepler movers through the SAME opaque seam the boot
// injects; the shipped crate cannot name any of this (vd-physics is a dev-dependency).
use vd_core::pose::LatticePos; // only the tests construct a LatticePos directly; prod uses .map_offset/.offset
use vd_core::{MsgId, UniverseTick};
use vd_physics::celestial::{OrbitalElements, orbital_state};
use vd_physics::motion::kepler_motion_fns;
use vd_wire::intershard::{DEMOTE_STEP, FLUSH_SOURCE_STEP, RE_SOLICIT_STEP, STUB_CROSSING_STEP};

const SHARD: NodeId = NodeId(10);
const GATEWAY: NodeId = NodeId(20);
const ORCH: NodeId = NodeId(30);
const SESSION: SessionId = SessionId(0xAA);

/// HR5 — install a TRACE-level SINK subscriber once per test binary, so every tracing macro's
/// lazy field closure EVALUATES on the paths the tests drive. Without a subscriber, tracing
/// short-circuits at the callsite and the field expressions (the Stage-A log points' whole
/// value) are dead regions no test can reach. The output goes to a sink: the fields are
/// exercised, the terminal stays quiet.
fn init_test_tracing() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::level_filters::LevelFilter::TRACE)
            .with_writer(std::io::sink)
            .finish();
        let _ = tracing::subscriber::set_global_default(subscriber);
    });
}

fn config() -> StubConfig {
    init_test_tracing(); // every test builds a config, so every test gets the TRACE sink
    StubConfig {
        realm: RealmId::System(7),
        held_realms: StubConfig::single_realm(RealmId::System(7)),
        frame: FrameRef::SystemSpace { system_seed: 7 },
        move_speed_mps: 2.0,
        tick_dt_s: 0.05,
        time_multiplier: 1.0,
        orchestrator: ORCH,
        mint_seed: 99,
        input_log_capacity: 1024,
        realm_recheck_interval: 0,
        lease_renew_interval_ticks: 0,
        self_fence_grace_ticks: 0,
        snapshot_datagram_budget: 1100,
        boundary: vd_core::geometry::BoundaryTuning::DEFAULT,
        request_ttl_ticks: 0,
        crossing_redrive_budget: 0,
        handoff_hold_ttl_ticks: 0,
        own_coord: StubConfig::root_coord(RealmId::System(7)),
        boot_ticks_p99: 0,
        // 5f-3b: no stored spawn poses ⇒ every login births origin-at-rest (byte-identical).
        spawn_poses: BTreeMap::new(),
    }
}

// ---- RLM 5f RG-1: the reactive greeting ------------------------------------------------------
/// A booked ANCESTOR peer (parent realm) — distinct from the orchestrator + gateway.
const ANCESTOR: NodeId = NodeId(40);

/// A greeting policy over `peers` with silence threshold `interval`.
fn presence(peers: &[NodeId], interval: u64) -> PresenceAnnounce {
    PresenceAnnounce::new(peers.iter().copied().collect(), interval).expect("valid interval")
}

/// The `(target → greeted-at tick)` of every `ShardPresence` a tick emitted (ignoring the other
/// flows a shard also sends). The `_ => None` arm is covered by those other flows (e.g. the boot
/// `LeaseGrant`), the `Some` arm by any greeting.
fn presence_ticks(
    sent: &[(NodeId, MsgClass, InterShardFlow, Durability)],
) -> BTreeMap<NodeId, TickId> {
    sent.iter()
        .filter_map(|(to, _, flow, _)| match flow {
            InterShardFlow::ShardPresence(p) => Some((*to, p.local_tick)),
            _ => None,
        })
        .collect()
}

/// A well-formed but INERT inbound FROM `from` (a head-read reply for a realm this shard does not
/// own ⇒ `process_inbound` no-ops) — used only to register CONTACT for the silence heuristic.
fn benign_contact(from: NodeId) -> Inbound {
    let reply = DirectoryReply::Head {
        key: DirectoryKey::Realm(RealmId::System(999)),
        record: None,
    };
    Inbound::Wire {
        from,
        class: MsgClass::Saga,
        bytes: crate::io::bytes(
            postcard::to_allocvec(&InterShardFlow::DirectoryReply(reply)).expect("encode"),
        ),
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
        Rig::with_config_and_kind(cfg, NodeKind::StubShard)
    }

    /// As [`with_config`](Self::with_config) but with an explicit [`NodeKind`] — the RLM Step-5a
    /// capability-inertness gate parameterizes the shard's carried profile over this.
    fn with_config_and_kind(cfg: StubConfig, kind: NodeKind) -> Rig {
        let mut world = World::new();
        world.insert_resource(InboundBox::default());
        world.insert_resource(OutboundBox::default());
        world.insert_resource(NodeIdentity {
            node_id: SHARD,
            kind,
        });
        world.insert_resource(ClockSample {
            local_tick: vd_core::TickId(1),
            universe_tick: UniverseTick(100),
            epoch: vd_core::EpochId(1),
            // RLM Step 2 (M-1 rig sweep): the shared rig is SYNCED so the newly-gated authors
            // (`evaluate_realm_boundaries`/`emit_realm_frames`/`evaluate_realm_aoi`) actually run —
            // else they silently no-op and every author assertion goes false-green.
            synced: true,
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
            .map(|(to, class, bytes, _)| (to, class, bytes.to_vec()))
            .collect()
    }

    /// Like [`tick`](Self::tick) but PRESERVES each frame's [`Durability`] — for the R-6d marker
    /// conformance (verifying a producer-less one-shot's push site carries `Retained`).
    fn tick_raw(
        &mut self,
        inbound: Vec<Inbound>,
    ) -> Vec<(NodeId, MsgClass, InterShardFlow, Durability)> {
        self.world.resource_mut::<InboundBox>().0 = inbound;
        self.schedule.run(&mut self.world);
        std::mem::take(&mut self.world.resource_mut::<OutboundBox>().0)
            .into_iter()
            .filter_map(|(to, class, bytes, dur)| {
                postcard::from_bytes::<InterShardFlow>(&bytes)
                    .ok()
                    .map(|flow| (to, class, flow, dur))
            })
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
        self.attach_request_with(session, gateway, None)
    }

    /// The same attach, carrying the spawn pose the gateway measured against THIS realm.
    fn attach_request_with(
        &mut self,
        session: SessionId,
        gateway: NodeId,
        spawn: Option<StampedPose>,
    ) -> Vec<(NodeId, MsgClass, Vec<u8>)> {
        let msg = GatewayToShard::AttachSession {
            session,
            fence: Fence(1),
            account: AccountId(5),
            spawn,
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

// ---- D-7 (transient/debris transfer) -------------------------------------------------------

const DEST_NODE: NodeId = NodeId(40);

fn transient_pose() -> StampedPose {
    StampedPose::at_rest(config().frame, DVec3::new(1.0, 2.0, 3.0), UniverseTick(5))
}

/// Decode an `OutboundBox` into `(target, InterShardFlow)` pairs (the Saga-class egress).
fn decode_flows(outbox: &mut OutboundBox) -> Vec<(NodeId, InterShardFlow)> {
    std::mem::take(&mut outbox.0)
        .into_iter()
        .map(|(to, _class, bytes, _)| {
            (
                to,
                postcard::from_bytes::<InterShardFlow>(&bytes).expect("decode"),
            )
        })
        .collect()
}

fn input_msg(seq: u64, fence: Fence, movement: [f32; 3], look: [f32; 2]) -> Inbound {
    // Every existing caller keeps the inert action_bits:0 default via this delegation (DRY — ONE
    // InputDatagram construction site); only the action_bits tripwire below varies the bits.
    input_msg_bits(seq, fence, movement, look, 0)
}

/// Like [`input_msg`] but with an EXPLICIT `action_bits` — the D-41/D-39.1 tripwire is the sole
/// caller that needs to vary the one field `input_msg` otherwise hardcodes to 0.
fn input_msg_bits(
    seq: u64,
    fence: Fence,
    movement: [f32; 3],
    look: [f32; 2],
    action_bits: u32,
) -> Inbound {
    let input = InputDatagram {
        seq,
        is_cut_marker: false,
        client_tick: vd_core::TickId(2),
        movement,
        look,
        action_bits,
    };
    let msg = GatewayToShard::SessionInput {
        session: SESSION,
        fence,
        input_bytes: postcard::to_allocvec(&input).expect("encode input"),
    };
    wire_msg(GATEWAY, MsgClass::Input, &msg)
}

/// The EntityIds this tick's snapshot frames carried to the gateway — the emit-set probe.
fn entity_rows_to_gateway(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<EntityId> {
    decode_frames(sent)
        .iter()
        .flat_map(|s| s.entities.iter().map(|e| e.entity))
        .collect()
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

/// A frame no shard in these rigs is: stands in for "measured somewhere else". It used to be the
/// UNIVERSE-ROOT frame every stored spawn pose was written in — which, after the fold's removal, is a
/// frame nobody may measure in at all, which is exactly why a pose wearing it must be refused.
fn foreign_frame() -> FrameRef {
    FrameRef::SystemSpace { system_seed: 0 }
}

/// A position's TOTAL displacement from its frame origin, in metres — whole-number part and leftover
/// combined. The integrator now folds the leftover into the whole number every tick, so reading the
/// leftover alone (which these assertions used to do) reports a sub-millimetre remainder rather than
/// where the subject actually is.
fn total_m(p: &vd_core::pose::LatticePos) -> DVec3 {
    p.delta_m(vd_core::pose::LatticePos::ORIGIN, vd_core::pose::Tier::Fine)
}

// ---- THE SPEED LAW (S3): the governed ceiling, the ramp, and the inertness measurement ----

/// A clamped-scale forest in the shard's own frame: every extent under the break-even
/// (`v_foot·T/2 = 2·180/2 = 180 m` at the fixture's 2 m/s foot), so every ceiling clamps to
/// the foot speed — THE world's in-system regime, at the fixture's scale.
fn clamped_forest() -> RealmRegions {
    RealmRegions::new(vec![
        region(ROOT_REALM, None, DVec3::ZERO, 150.0),
        region(OWN_REALM, Some(ROOT_REALM), DVec3::ZERO, 100.0),
    ])
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

/// Whether the orchestrator-bound egress carries a specific `SagaAck` (value-compare; the
/// `*to == ORCH` filter restricts decode to ack/directory traffic — frames go to gateways).
fn saga_ack_to_orch(sent: &[(NodeId, MsgClass, Vec<u8>)], ack: TransferControlAck) -> bool {
    sent.iter()
        .filter(|(to, _, _)| *to == ORCH)
        .any(|(_, _, bytes)| {
            postcard::from_bytes::<InterShardFlow>(bytes).ok() == Some(InterShardFlow::SagaAck(ack))
        })
}

fn orbit() -> OrbitalElements {
    OrbitalElements {
        sma: 1.5e11,
        ecc: 0.1,
        inclination: 0.4,
        raan: 0.3,
        arg_periapsis: 0.9,
        mean_anomaly_epoch: 0.2,
        central_mass: 1.989e30,
    }
}

/// The own frame the client-facing emit restates every row against, for a shard whose own realm
/// is `OWN_REALM` (each row selects its book from the ledger at its own pose stamp).
fn emit_context(regions: &RealmRegions) -> FrameRef {
    regions.own_frame(OWN_REALM)
}

/// The empty-forest shard's ledger: what `author_placements` publishes over `RealmRegions::default()`
/// — an EMPTY head book per held anchor (anchored on the `GalaxySpace` fallback), so an arrival is
/// refused on the missing FRAME, not on a missing book.
// Test twin of the ONE writer — the same stated exemption from the publish ban.
#[allow(clippy::disallowed_methods)]
fn bare_ledger(cfg: &StubConfig) -> PlacementLedger {
    let regions = RealmRegions::default();
    let mut ledger = PlacementLedger::new(64);
    for anchor in placement_anchors(&regions, cfg) {
        ledger.publish(
            anchor,
            regions.author_book(anchor, STORY_TICK_HZ, UniverseTick(0)),
        );
    }
    ledger
}

/// The ledger a shard holds at `clock`'s universe tick — one head book per anchor the writer
/// covers, exactly as `author_placements` publishes them.
// Test twin of the ONE writer — the same stated exemption from the publish ban.
#[allow(clippy::disallowed_methods)]
fn obs_ledger(regions: &RealmRegions, cfg: &StubConfig, clock: &ClockSample) -> PlacementLedger {
    let mut ledger = PlacementLedger::new(64);
    for anchor in placement_anchors(regions, cfg) {
        ledger.publish(
            anchor,
            regions.author_book(anchor, 1.0 / cfg.tick_dt_s, clock.universe_tick),
        );
    }
    ledger
}

/// A test ledger holding what `author_placements` would have published for `OWN_REALM` at `tick`
/// (a generous window so multi-instant fixtures stay inside it).
// Test twin of the ONE writer — the same stated exemption from the publish ban.
#[allow(clippy::disallowed_methods)]
fn ledger_at(regions: &RealmRegions, tick_hz: f64, tick: UniverseTick) -> PlacementLedger {
    let mut ledger = PlacementLedger::new(64);
    ledger.publish(OWN_REALM, regions.author_book(OWN_REALM, tick_hz, tick));
    ledger
}

/// One emitting dot at local (1,2,3) stamped at tick 100 — the shared fixture.
fn slice6_dot(entity: EntityId, frame: FrameRef, authority: Authority) -> Dot {
    Dot {
        entity,
        account: AccountId(1),
        session_fence: Fence(1),
        gateway: GATEWAY,
        granted: true,
        input_active: false,
        adopting: false,
        authority,
        departing: false,
        entity_fence: Fence(1),
        pose: StampedPose::at_rest(frame, DVec3::new(1.0, 2.0, 3.0), UniverseTick(100)),
        yaw: 0.0,
        pitch: 0.0,
        last_applied_seq: None,
        prev_offset: LatticePos::ORIGIN,
    }
}

// ---- Slice 1d.1: the pose-only entity-state crossing (source flush + dest adopt) ---------

/// Where the occupant came FROM: the neighbouring star system it left.
const FROM_REALM: RealmId = RealmId::System(8);
/// Where the crossing is going TO — the realm THIS RIG HOSTS (`config().realm`), because a crossing
/// addressed to this shard is by definition addressed to a realm it holds.
///
/// These two were the other way round, naming this shard as the SOURCE and its neighbour as the
/// destination, which no real crossing into this shard can be. It went unnoticed while the receiver
/// measured every arrival against its own realm and never read the destination the crossing named; now
/// that it reads it, a crossing addressed to a realm this shard does not hold is refused — which is the
/// behaviour that stops an occupant being placed in a space nobody here can measure.
const TO_REALM: RealmId = RealmId::System(7);

/// A non-origin crossing pose (so a test can tell an applied crossing from the origin-adopt),
/// stamped in the frame of the realm the RIG HOSTS (`config().realm` is `System(7)`).
///
/// That is the DOWNWARD hand-off: the parent has already expressed the occupant in this realm's frame,
/// so the receiver accepts it verbatim and does no arithmetic. It used to be stamped `SystemSpace{8}`
/// — a frame this shard hosts nothing in and was never told the position of. That only worked because
/// the ingress swallowed the resulting error and kept the number under the new label; now such an
/// arrival is refused and counted, which is the whole point of the change.
fn crossing_pose() -> StampedPose {
    StampedPose::at_rest(
        FrameRef::SystemSpace { system_seed: 7 },
        DVec3::new(4.0, -5.0, 6.0),
        UniverseTick(200),
    )
}

/// The directory head that flips the adopted dot to `granted` at `fence` (dest-owned record).
fn adopted_head(fence: Fence) -> Inbound {
    wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::DirectoryReply(DirectoryReply::Head {
            key: DirectoryKey::Entity(SUBJECT),
            record: Some(vd_wire::seams::directory::OwnerRecord {
                authority: AuthorityRef::Shard(SHARD),
                fence,
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        }),
    )
}

/// A `StubCrossing` envelope for `SUBJECT` at `fence` carrying `pose`.
fn crossing_msg(transfer: TransferId, fence: Fence, pose: StampedPose) -> Inbound {
    wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::Transfer(TransferEnvelope {
            transfer_id: transfer,
            universe_epoch: vd_core::EpochId(1),
            schema_version: vd_wire::intershard::TRANSFER_SCHEMA_VERSION,
            fence,
            step_id: STUB_CROSSING_STEP,
            class: vd_core::entity_kind::DurabilityClass::Durable,
            payload: TransitionPayload::StubCrossing {
                entity: SUBJECT,
                from_realm: FROM_REALM,
                to_realm: TO_REALM,
                pose,
                state: vec![],
            },
        }),
    )
}

fn flush_msg(entity: EntityId) -> Inbound {
    wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::FlushSource(FlushSource {
            transfer: TransferId(7),
            subject: DirectoryKey::Entity(entity),
            step_id: FLUSH_SOURCE_STEP,
            to_realm: RealmId::System(0),
            to_parent: None,
        }),
    )
}

/// The flows the shard sent to the orchestrator this tick.
fn to_orch(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<InterShardFlow> {
    sent.iter()
        .filter(|(to, _, _)| *to == ORCH)
        .map(|(_, _, b)| postcard::from_bytes(b).expect("flow decodes"))
        .collect()
}

/// Whether the shard acked a crossing step (value-compare, never `matches!` — HR5).
fn acked(sent: &[(NodeId, MsgClass, Vec<u8>)], transfer: TransferId) -> bool {
    to_orch(sent).contains(&InterShardFlow::TransferAck(TransferAck::Accepted {
        transfer_id: transfer,
        step_id: STUB_CROSSING_STEP,
    }))
}

/// A saga `Promote` for `SUBJECT` at `new_fence`, naming `source` as the ghost-host (1d.5b.3b).
fn promote_msg(new_fence: Fence, source: NodeId) -> Inbound {
    wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::Promote(PromoteCmd {
            transfer: TransferId(7),
            subject: DirectoryKey::Entity(SUBJECT),
            new_fence,
            step_id: PROMOTE_STEP,
            source,
        }),
    )
}

fn re_home_msg(new_fence: Fence, source: NodeId, subject: DirectoryKey) -> Inbound {
    wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::ReHome(ReHomeCmd {
            transfer: TransferId(7),
            universe_epoch: vd_core::EpochId(1),
            subject,
            new_fence,
            step_id: RE_HOME_STEP,
            state: ReHomeState::PoseOnly(crossing_pose()),
            source,
        }),
    )
}

/// The `ShardToGateway` Control replies this shard sent to `GATEWAY` this tick (value-compare).
fn gw_replies(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<ShardToGateway> {
    sent.iter()
        .filter(|(to, class, _)| (*to == GATEWAY) & (*class == MsgClass::Control))
        .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("decode"))
        .collect()
}

/// The `InterShardFlow`s this shard sent to `target` this tick (value-compare; `.ok()` drops the
/// rare non-flow / decode failure, of which there are none on a ghost/orchestrator peer).
fn flows_to(sent: &[(NodeId, MsgClass, Vec<u8>)], target: NodeId) -> Vec<InterShardFlow> {
    sent.iter()
        .filter(|(to, _, _)| *to == target)
        .filter_map(|(_, _, bytes)| postcard::from_bytes::<InterShardFlow>(bytes).ok())
        .collect()
}

/// One `GhostFlow::Delta` as the SOURCE ghost-host receives it (for the consumer tests).
fn ghost_delta(entity: EntityId, pose: StampedPose, source_fence: Fence, seq: u64) -> Inbound {
    Inbound::Wire {
        from: DEST_OWNER,
        class: MsgClass::GhostDelta,
        bytes: crate::io::bytes(
            postcard::to_allocvec(&InterShardFlow::Ghost(GhostFlow::Delta {
                entity,
                pose,
                source_fence,
                source_tick: vd_core::TickId(0),
                seq,
            }))
            .expect("encode"),
        ),
    }
}

/// One `GhostFlow::Spawn` / `Despawn` as the SOURCE ghost-host receives it (reliable carrier).
fn ghost_lifecycle(flow: GhostFlow) -> Inbound {
    Inbound::Wire {
        from: DEST_OWNER,
        class: MsgClass::GhostReliable,
        bytes: crate::io::bytes(
            postcard::to_allocvec(&InterShardFlow::Ghost(flow)).expect("encode"),
        ),
    }
}

/// Demote the SESSION dot to a RETAINED source Ghost at `new_owner_fence` (the consumer/emit
/// fixture): a granted Owned dot → the saga `Demote` → `Ghost`. Returns the dot's entity.
/// ARMS the hand-off budget first (slice F: the retained ghost emits only while its Source
/// hold is open, and a hold only opens on an armed budget — the shipped posture everywhere).
fn make_retained_ghost(rig: &mut Rig, new_owner_fence: Fence) -> EntityId {
    rig.world
        .resource_mut::<StubConfig>()
        .handoff_hold_ttl_ticks = 1_000;
    rig.grant_realm();
    let _ = rig.attach();
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
    let demote = InterShardFlow::Demote(DemoteCmd {
        transfer: TransferId(7),
        subject: DirectoryKey::Entity(entity),
        new_owner_fence,
        step_id: vd_wire::intershard::DEMOTE_STEP,
    });
    let _ = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &demote)]);
    entity
}

/// The DEST owner node that feeds this shard's hosted ghosts in the consumer tests.
const DEST_OWNER: NodeId = NodeId(99);

/// The remove messages a tick pushed, as `(gateway, entity, at)` — the shard half of the
/// D-4(a) lane, read off the wire.
fn entity_removals(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<(NodeId, EntityId, UniverseTick)> {
    sent.iter()
        .filter(|(_, class, _)| *class == MsgClass::Control)
        .filter_map(
            |(to, _, bytes)| match postcard::from_bytes::<ShardToGateway>(bytes) {
                Ok(ShardToGateway::EntityRemoved { entity, at, .. }) => Some((*to, entity, at)),
                _ => None,
            },
        )
        .collect()
}

// ---- Slice 5 → Step 5 slice E: the cross-realm ENTITY lane is DEAD -------------------------
// The unit tests of the deleted relay (origin-tag loop-freedom, lift-on-arrival, staleness,
// mis-route, TTL age-out, from-above duties) died with the machinery. THREE tests that lived in
// this section pinned SURVIVING machinery and are RESTORED below (the first sweep took them too
// — coverage and the adversarial review both caught it): the up-lane wire dispatch, the
// observed-interior fan/up-recursion, and the unplaceable-observer refusal. What remains to pin
// beyond those: a tombstoned frame on the carrier is COUNTED, and the client-edge emit ships
// OWN rows only (the emit tests above). SL2's steady state — no occupant pose crossing a realm
// boundary — is measured at the scenario tier (`frame_conversion_e2e`), where the chain runs.

/// The routing key of the child a test wants to address, built the way the shard itself builds it.
fn with_child_coord(rig: &mut Rig, child: RealmId) -> RealmCoord {
    let config = rig.world.resource::<StubConfig>().clone();
    let regions = rig.world.resource::<RealmRegions>();
    let region = regions
        .direct_children(config.realm)
        .find(|r| r.realm == child)
        .expect("the fixture registered that child");
    config
        .own_coord
        .child(region_level(region).expect("a seed-lineage child region"))
}

// ---- Slice 3d/3e/4b + C-3 (the CONTAINMENT re-home trigger) --------------------------------

use vd_core::geometry::{Boundary, BoundaryTuning, ContainmentBand, RealmRegion};
use vd_core::pose::frame_for_realm;

/// The shard's OWN realm (mirrors `config().realm`); a re-home fires when the container differs.
const OWN_REALM: RealmId = RealmId::System(7);
/// The DEEPER child region a dot docks INTO (its container becomes `OTHER_REALM` ⇒ re-home here).
const OTHER_REALM: RealmId = RealmId::Planet(42);
/// The ambient ROOT realm (`parent: None`) — the `container` fold identity. A large Shell covering
/// everything, so an entity is ALWAYS in ≥1 realm (the "no way we will not be in any realm" mandate).
const ROOT_REALM: RealmId = RealmId::System(0);
/// The PARENT an OWN-realm child undocks OUTWARD to (distinct from `OWN_REALM` and `OTHER_REALM`, so
/// the undock destination a test reads is unambiguous). In the escape forest `OWN_REALM` nests under it.
const PARENT_REALM: RealmId = RealmId::System(77);
const TRIG_SESSION: SessionId = SessionId(0xBB);

/// Build a velocity-safe containment band (slow v_rel so it is inset/outset-sized, not widened). The
/// generous inset/outset (metres) means a dot placed clearly inside/outside a region's shell resolves
/// membership without any per-tick flap.
fn band() -> ContainmentBand {
    ContainmentBand::for_containment_velocity_safe(50.0, 100.0, 1.0, 0.05, 1.0)
        .expect("valid containment band")
}

/// The frame `realm`'s region is expressed in. System/Planet always resolve; at walk scale every
/// placement is the identity, so positions are frame-invariant regardless. (`IdentityFrames` itself
/// is DELETED — D-PLACE-1.)
fn frame_of(realm: RealmId) -> FrameRef {
    frame_for_realm(realm, None).expect("System/Planet realm always resolves a frame")
}

/// One `RealmRegion` shell at `center` of radius `r`, in `realm`'s own frame, nested under `parent`.
fn region(realm: RealmId, parent: Option<RealmId>, center: DVec3, r: f64) -> RealmRegion {
    RealmRegion {
        realm,
        center: vd_core::geometry::ParentCentre::authored(LatticePos::from_metres(
            center,
            vd_core::pose::Tier::Fine,
        )),
        frame: frame_of(realm),
        shape: Boundary::Shell { r },
        look: Some(Boundary::Shell { r }),
        band: band(),
        aoi: vd_core::geometry::AoiConfig::inert(),
        parent,
        interior_band: vd_core::geometry::AoiConfig::inert(),
    }
}

/// One `RealmRegion` axis-aligned BOX at `center` with per-axis `half`, nested under `parent`.
fn region_box(realm: RealmId, parent: Option<RealmId>, center: DVec3, half: DVec3) -> RealmRegion {
    RealmRegion {
        realm,
        center: vd_core::geometry::ParentCentre::authored(LatticePos::from_metres(
            center,
            vd_core::pose::Tier::Fine,
        )),
        frame: frame_of(realm),
        shape: Boundary::Aabb { half },
        look: Some(Boundary::Aabb { half }),
        band: band(),
        aoi: vd_core::geometry::AoiConfig::inert(),
        parent,
        interior_band: vd_core::geometry::AoiConfig::inert(),
    }
}

/// The AMBIENT-ROOT region: a huge shell at the origin covering all placements a test uses, so the
/// container fold is total (every point is at least in the root). `parent: None`.
fn root_region() -> RealmRegion {
    region(ROOT_REALM, None, DVec3::ZERO, 1.0e9)
}

/// The shard's OWN-realm region (`System(7)`): a large shell at the origin nested under the root. A dot
/// inside only this (and the root) has container == `OWN_REALM` ⇒ NO re-home.
fn own_region() -> RealmRegion {
    region(OWN_REALM, Some(ROOT_REALM), DVec3::ZERO, 100_000.0)
}

/// The DEEPER child region (`OTHER_REALM` = `Planet(42)`): a small shell at the origin nested under
/// `OWN_REALM`. A dot clearly inside it ⇒ container == `OTHER_REALM` ⇒ re-home INWARD to `OTHER_REALM`.
fn child_region() -> RealmRegion {
    region(OTHER_REALM, Some(OWN_REALM), DVec3::ZERO, 1000.0)
}

/// Plant the STANDARD 3-level dock forest (root ⊃ own ⊃ child) into the world's `RealmRegions`. A dot
/// inside the child re-homes to `OTHER_REALM`; a dot only inside own (outside child) stays.
fn plant_dock_regions(rig: &mut Rig) {
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), own_region(), child_region()]);
}

/// Plant the ESCAPE forest: root(`System(0)`) ⊃ parent(`PARENT_REALM`) ⊃ own(`OWN_REALM`, a SMALL
/// shell). The shard's config realm is `OWN_REALM`. A dot INSIDE the small own-shell has container ==
/// `OWN_REALM` (no re-home); a dot OUTSIDE it but still inside the parent shell has container ==
/// `PARENT_REALM` ⇒ re-home OUTWARD to `PARENT_REALM` (the undock). Symmetric with the dock — the same
/// containment machinery, no "direction".
fn plant_escape_regions(rig: &mut Rig) {
    let parent = region(PARENT_REALM, Some(ROOT_REALM), DVec3::ZERO, 100_000.0);
    let own_small = region(OWN_REALM, Some(PARENT_REALM), DVec3::ZERO, 1000.0);
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), parent, own_small]);
}

/// Insert an OWNED dot (a durable Player by default) at frame-local `offset`, its `prev_offset`
/// seeded to the SAME offset (a fresh spawn — the first evaluated segment is degenerate).
fn insert_owned_dot(rig: &mut Rig, session: SessionId, entity: EntityId, offset: DVec3) {
    rig.world.resource_mut::<Dots>().0.insert(
        session,
        Dot {
            entity,
            account: AccountId(1),
            session_fence: Fence(1),
            gateway: GATEWAY,
            granted: true,
            input_active: false,
            adopting: false,
            authority: Authority::Owned { fence: Fence(1) },
            departing: false,
            entity_fence: Fence(1),
            pose: StampedPose {
                pos: seated_pos(offset),
                ..StampedPose::at_rest(config().frame, offset, UniverseTick(100))
            },
            yaw: 0.0,
            pitch: 0.0,
            last_applied_seq: None,
            prev_offset: LatticePos::from_metres(offset, vd_core::pose::Tier::Fine),
        },
    );
}

/// Move an owned dot's frame-local offset (its render/trigger position) without touching
/// `prev_offset` (the trigger writes that itself each tick).
fn move_dot(rig: &mut Rig, session: SessionId, offset: DVec3) {
    let pos = seated_pos(offset);
    rig.world
        .resource_mut::<Dots>()
        .0
        .get_mut(&session)
        .expect("the owned dot")
        .pose
        .pos = pos;
}

/// A position stored THE WAY THE INTEGRATOR STORES IT — whole-number part folded out, sub-cell
/// remainder left behind (`normalize`, exactly what `integrate` does every tick).
///
/// WHY THE TEST HELPERS GO THROUGH THIS. Every fixture used to seat its dot at rest, which leaves the
/// whole-number part at zero and makes the remainder equal the whole position. That is a state no
/// moving player has been in since the movement fold, so no test in this file has ever exercised what
/// the game actually produces — which is why a consumer reading the remainder as if it were the
/// position went unnoticed. Seating dots the real way is what turns those consumers from arguable
/// into measurable.
fn seated_pos(offset: DVec3) -> LatticePos {
    LatticePos::from_metres(offset, vd_core::pose::Tier::Fine)
}

/// Every `CrossingRequest` in the outbox (decoded), so a test asserts the exact count.
fn crossing_requests(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<CrossingRequest> {
    sent.iter()
        .filter_map(
            |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                Ok(InterShardFlow::CrossingRequest(r)) => Some(r),
                _ => None,
            },
        )
        .collect()
}

fn realm_demands(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<RealmDemand> {
    sent.iter()
        .filter_map(
            |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                Ok(InterShardFlow::RealmDemand(d)) => Some(d),
                _ => None,
            },
        )
        .collect()
}

/// Every `CrossingAbortedAck` in the outbox (decoded) — the source's latch-clear confirm (3f-D).
fn crossing_aborted_acks(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<CrossingAborted> {
    sent.iter()
        .filter_map(
            |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                Ok(InterShardFlow::CrossingAbortedAck(a)) => Some(a),
                _ => None,
            },
        )
        .collect()
}

/// Every `TransientCrossingRequest` in the outbox (decoded).
fn transient_crossing_requests(
    sent: &[(NodeId, MsgClass, Vec<u8>)],
) -> Vec<TransientCrossingRequest> {
    sent.iter()
        .filter_map(
            |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                Ok(InterShardFlow::TransientCrossingRequest(r)) => Some(r),
                _ => None,
            },
        )
        .collect()
}

// ==== THE WORKED EXAMPLE — the negative gate on the ground rule ==================================
//
// THE GROUND RULE, in the owner's words: only the parent knows where the children are; a child has no
// idea about its own position; we do not leak unnecessary data from one realm to another.
//
// THE STORY these three tests re-tell. A galaxy holds two star systems; the neighbour sits 12031 m
// out. Inside it a planet is authored 145 m from its star. A player flies 3 m above that planet.
//   - the planet knows only "an occupant at 3 from my centre";
//   - the system knows only "I put that planet at 145";
//   - the galaxy knows only "I put that system at 12031".
// GOING UP: the SYSTEM adds 145 → 148, then the GALAXY adds 12031 → 12179. Three separate contexts,
// three separate additions, each by the one party that holds that number. GOING DOWN the galaxy
// subtracts to 148, the system subtracts to 3, and the planet ACCEPTS 3 and does no arithmetic at all.
//
// WHAT THIS REPLACES, and why the negative test is the valuable one. A realm used to obtain its own
// absolute position by folding its whole chain of ancestors from the universe root. That works, and it
// is the leak: it makes every realm know where it itself is, and it throws away the precision that
// lets a planet's surface deal in metres however far the planet is from anything else. Removing the
// fold is not enough on its own — the fold can come back under another name at any layer. What stops
// it is that a shard asking where it sits in its own parent's frame gets a TYPED REFUSAL, and that is
// asserted here, on every `coverage-fast` loop.
//
// The world is built by the PRODUCTION generator (`WorldView::generated`, the same
// `f(seed, UniverseConfig)` `boot_regions_and_movers` reaches through) and scoped by the PRODUCTION
// neighbourhood filter, so the fixture cannot describe a world no shard would boot. The same story is
// planted at the scenario tier in `vd-tests`' `frame_fixture` with the same three constants; vd-sim
// cannot depend on vd-tests (vd-tests depends on vd-sim), so the constants are stated in both places
// and each side asserts them against what the generator actually planted.

/// The occupant's offset from the planet, metres — the one hand-picked story number.
const STORY_OCCUPANT_FROM_PLANET_M: f64 = 3.0;
/// The planet's orbital period is REAL Kepler now (the story's star mass is drawn); the
/// story reads distances off the generated world instead of tuning them (the in-system
/// true-size re-solve deleted the compression/synthetic-mass knobs the old story turned).
/// The story cluster's tick rate (the authored-book cadence).
const STORY_TICK_HZ: f64 = 20.0;
/// How much bigger than the thing it must contain each shell in the story world is.
const STORY_SHELL_HEADROOM: f64 = 4.0;

/// The three levels of the story, as a production-generated world plus the realms that play the parts.
struct Story {
    world: vd_physics::worldgen::WorldView,
    config: vd_physics::worldgen::UniverseConfig,
    universe: RealmId,
    galaxy: RealmId,
    system: RealmId,
    planet: RealmId,
    sibling_system: RealmId,
    sibling_planet: RealmId,
    elements: OrbitalElements,
}

impl Story {
    /// Generate the story world through the production generator, then read the parts out of the
    /// forest rather than naming their seeds — a system's seed is a `child_seed` avalanche of
    /// (galaxy, salt, index), so writing one down would be copying a hash into a test.
    fn new() -> Story {
        let mut config = vd_physics::worldgen::UniverseConfig::walk_scale();
        // Exactly two stars; the second is the SIBLING the negative gates demand refusals for.
        config.galaxy.system_count_lo = 2;
        config.galaxy.system_count_hi = 2;
        // The in-system true-size re-solve SOLVES each system's shell (no config radius
        // exists), so the placement radius and the ambient shells derive from the solve's
        // own reserved bound — disjoint siblings, nesting ambients, no tuned number.
        config.stellar.system_ring_r_m =
            STORY_SHELL_HEADROOM * vd_physics::worldgen::target_system_bound_max_m();
        config.planet.n_planets = 2;
        // Circular and in-plane, so the planet's distance from its star is its semi-major
        // axis at EVERY tick — a fact about the orbit, not about one instant. The axis
        // itself is the √L-anchored ladder's rung 0, READ from the generated elements below.
        config.planet.ecc_sigma = 0.0;
        config.planet.incl_sigma = 0.0;
        config.scale.galaxy_r_m = (config.stellar.system_ring_r_m
            + vd_physics::worldgen::target_system_bound_max_m())
            * STORY_SHELL_HEADROOM;
        config.scale.universe_r_m = config.scale.galaxy_r_m * STORY_SHELL_HEADROOM;

        let world = vd_physics::worldgen::WorldView::generated(0, &config);
        let universe = world
            .regions()
            .iter()
            .find(|r| r.parent.is_none())
            .expect("a generated world has one ambient root")
            .realm;
        let galaxy = Story::children(&world, universe)[0];
        let systems = Story::children(&world, galaxy);
        // The story's system is the one that is actually SOMEWHERE: a hop of zero would prove nothing
        // about a parent adding its child's placement.
        let system = *systems
            .iter()
            .find(|r| Story::centre(&world, **r) != DVec3::ZERO)
            .expect("a two-star galaxy puts one system off its own centre");
        let sibling_system = *systems
            .iter()
            .find(|r| **r != system)
            .expect("a two-star galaxy has a second system");
        let planets = Story::children(&world, system);
        // The story planet's elements: the config's semi-major axis, eccentricity, inclination and
        // central mass, with the three PHASE angles pinned to zero. That chooses WHERE ON ITS CIRCLE
        // the planet sits at tick 0 and nothing else — without it the planet sits at a seed-drawn
        // angle and the second addition becomes `12031 + 145·(some direction)`, a true statement about
        // a rotated triangle and a useless one to read.
        let mut elements = vd_physics::worldgen::moving_children_for_config(0, &config, system)
            .into_iter()
            .find(|(r, _)| *r == planets[0])
            .map(|(_, e)| e)
            .expect("a generated planet is an orbital child of its star");
        elements.raan = 0.0;
        elements.arg_periapsis = 0.0;
        elements.mean_anomaly_epoch = 0.0;
        Story {
            world,
            config,
            universe,
            galaxy,
            system,
            planet: planets[0],
            sibling_planet: planets[1],
            sibling_system,
            elements,
        }
    }

    fn children(world: &vd_physics::worldgen::WorldView, realm: RealmId) -> Vec<RealmId> {
        world
            .regions()
            .iter()
            .filter(|r| r.parent == Some(realm))
            .map(|r| r.realm)
            .collect()
    }

    fn centre(world: &vd_physics::worldgen::WorldView, realm: RealmId) -> DVec3 {
        // Flatten the NORMALIZED centre (the generator is a lattice producer since the cell
        // activation) — reading `.offset()` here would read the sub-cell residual and place
        // every realm at the origin (the H-21 class this arc cures).
        //
        // ★ AT THE PARENT'S RUNG (slice S9). A region's `center` is its position in its PARENT's
        // frame, while `frame` is its own — two different units since the ladder landed, so reading
        // the parent's number with the child's ruler is wrong by the ratio between them. This read
        // the CHILD's, which put a star system 2048× further out than the galaxy had placed it.
        let regions = world.regions();
        let r = regions
            .iter()
            .find(|r| r.realm == realm)
            .expect("the story only names realms the generator produced");
        // The lookup, and the refusal when the parent is absent, are `ParentCentre`'s own now — the
        // `map_or_else(child's own tier)` fallback this replaces WAS the 2048× defect, written as a
        // default.
        r.centre_m(regions)
            .expect("the story's forest holds every named realm's parent")
    }

    fn frame(&self, realm: RealmId) -> FrameRef {
        self.world
            .regions()
            .iter()
            .find(|r| r.realm == realm)
            .expect("the story only names realms the generator produced")
            .frame
    }

    /// The `RealmRegions` the shard hosting `held` boots with: the PRODUCTION neighbourhood scope
    /// (own realm, ancestors, direct children — never a sibling) through the PRODUCTION builders.
    fn regions(&self, held: RealmId) -> RealmRegions {
        let moving = vd_physics::worldgen::moving_children_for_config(0, &self.config, held)
            .into_iter()
            .map(|(realm, e)| {
                if realm == self.planet {
                    (realm, self.elements)
                } else {
                    (realm, e)
                }
            })
            .collect();
        RealmRegions::new(
            self.world
                .neighbourhood(&std::collections::BTreeSet::from([held])),
        )
        .with_moving_children(kepler_motion_fns(moving))
    }

    /// The authored book that shard converts through, at tick 0 — the production `author_book`.
    fn ctx(&self, held: RealmId) -> vd_core::placement::PlacementBook {
        self.regions(held)
            .author_book(held, STORY_TICK_HZ, UniverseTick(0))
    }

    /// The placement ledger that shard holds at tick 0 — one head book per anchor the writer
    /// covers, exactly as `author_placements` would have published them.
    // Test twin of the ONE writer — the same stated exemption from the publish ban.
    #[allow(clippy::disallowed_methods)]
    fn ledger(&self, held: RealmId) -> PlacementLedger {
        let regions = self.regions(held);
        let cfg = story_config(self, held);
        let mut ledger = PlacementLedger::new(64);
        for anchor in placement_anchors(&regions, &cfg) {
            ledger.publish(
                anchor,
                regions.author_book(anchor, STORY_TICK_HZ, UniverseTick(0)),
            );
        }
        ledger
    }

    /// The planet's orbital radius (its semi-major axis — circular by construction), read
    /// off the generated elements: the √L ladder's rung 0 for the drawn star.
    fn planet_orbit_m(&self) -> f64 {
        self.elements.sma
    }

    /// The occupant's distance from the STAR while 3 m up the planet's own +x — the
    /// story's "145 + 3", derived (the up-conversion adds exactly this, bit-for-bit).
    fn up_1_m(&self) -> f64 {
        self.planet_orbit_m() + STORY_OCCUPANT_FROM_PLANET_M
    }

    /// A departed occupant: just past the planet's own RELEASE EDGE, read off the roster.
    ///
    /// ★ THE MARGIN IS DERIVED FROM THE BAND, not a literal. It used to be "the shell plus five
    /// metres", which was outside the release edge only while every band in the universe was the same
    /// three metres wide. Once bands are sized from the bodies they wrap, a planet's release edge sits
    /// kilometres out and five metres past the shell is still firmly INSIDE it — so the fixture stopped
    /// describing a departure while still calling itself one.
    fn departed_m(&self) -> f64 {
        let region = self
            .world
            .regions()
            .iter()
            .find(|r| r.realm == self.planet)
            .expect("the story planet is rostered");
        // One extra band's width past the release edge, so the fixture is unambiguously outside at
        // every size rather than by a margin that shrinks as the band grows.
        region.shape.finite_extent()
            + region.band.outset()
            + (region.band.inset() + region.band.outset())
    }

    /// A pose `x` metres along `+x` in `frame`, at rest at tick 0.
    fn pose_in(&self, frame: FrameRef, x: f64) -> StampedPose {
        StampedPose::at_rest(frame, DVec3::new(x, 0.0, 0.0), UniverseTick(0))
    }
}

/// The `StubConfig` the shard hosting `held` in the story world boots with. The FULL root-rooted
/// coord matters: `own_coord.parent()` is how a shard learns whose child it is, and both pose
/// ingresses read it to name their own frame.
fn story_config(story: &Story, held: RealmId) -> StubConfig {
    let regions = story
        .world
        .neighbourhood(&std::collections::BTreeSet::from([held]));
    StubConfig {
        realm: held,
        held_realms: StubConfig::single_realm(held),
        frame: story.frame(held),
        own_coord: vd_core::worldgen::coord_of_realm(&regions, held)
            .expect("a story realm has a seed lineage back to the root"),
        ..config()
    }
}

// ---- the hand-off hold ledger (inert until armed) ---------------------------------------------

fn hold_key(seed: u64) -> (EntityId, HoldRole) {
    (
        EntityId::pack(EntityKind::Player, 1, seed, 3),
        HoldRole::Source,
    )
}

// ---- task #133: the first-class Station/Area realm re-home GATE (kind-agnostic detector) --------

/// The System-7 seed neighbourhood the LIVE `shard.rs` boots — now including the first-class Station 7
/// child (task #133). Planting THIS (not a hand-authored fixture) proves the Station region is one the
/// production bins actually load.
const STATION_A: RealmId = RealmId::Station(7);

/// Grant `realm` to THIS shard (a parameterized [`Rig::grant_realm`], which hardcodes `config().realm`)
/// so a non-default-realm rig (e.g. a Station-owning shard) can take authority. Mirrors `grant_realm`'s
/// round-trip directory confirmation, keyed on the passed realm.
fn grant_realm_for(rig: &mut Rig, realm: RealmId) {
    grant_realm_for_at(rig, realm, Fence(1));
}

/// Like [`grant_realm_for`] but at an explicit `fence` — for asserting a RE-affirm refreshes the
/// co-hosted child's recorded fence (the `else if ours` refresh arm of `affirm_realm_head`).
fn grant_realm_for_at(rig: &mut Rig, realm: RealmId, fence: Fence) {
    let reply = DirectoryReply::Head {
        key: DirectoryKey::Realm(realm),
        record: Some(vd_wire::seams::directory::OwnerRecord {
            authority: AuthorityRef::Shard(SHARD),
            fence,
            lease_expires: UniverseTick(1_000),
            in_transfer: None,
        }),
    };
    let bytes = crate::io::bytes(
        postcard::to_allocvec(&InterShardFlow::DirectoryReply(reply)).expect("encode"),
    );
    let _ = rig.tick(vec![Inbound::Wire {
        from: ORCH,
        class: MsgClass::Saga,
        bytes,
    }]);
}

/// Affirm a realm head that is NO LONGER this shard's — a REVOKED record (`None`, the reaper dropped
/// the lease). For a CO-HOSTED child this drives the FOREIGN/None `else` arm of `affirm_realm_head`
/// (`cohosted.0.remove(&realm)`) — the child dropped from `CoHostedAuthority`.
fn revoke_realm_for(rig: &mut Rig, realm: RealmId) {
    let reply = DirectoryReply::Head {
        key: DirectoryKey::Realm(realm),
        record: None,
    };
    let bytes = crate::io::bytes(
        postcard::to_allocvec(&InterShardFlow::DirectoryReply(reply)).expect("encode"),
    );
    let _ = rig.tick(vec![Inbound::Wire {
        from: ORCH,
        class: MsgClass::Saga,
        bytes,
    }]);
}

/// Insert an OWNED durable dot at frame-local `offset` expressed in `frame` (the frame-aware sibling of
/// [`insert_owned_dot`], which pins the pose frame to `config().frame`). Needed for a Station-owning
/// shard, whose dots live in the StationLocal frame. At walk scale the frame is inert for the
/// container decision, but carrying the OWNING realm's frame keeps the fixture honest.
fn insert_owned_dot_framed(
    rig: &mut Rig,
    session: SessionId,
    entity: EntityId,
    frame: FrameRef,
    offset: DVec3,
) {
    rig.world.resource_mut::<Dots>().0.insert(
        session,
        Dot {
            entity,
            account: AccountId(1),
            session_fence: Fence(1),
            gateway: GATEWAY,
            granted: true,
            input_active: false,
            adopting: false,
            authority: Authority::Owned { fence: Fence(1) },
            departing: false,
            entity_fence: Fence(1),
            pose: StampedPose::at_rest(frame, offset, UniverseTick(100)),
            yaw: 0.0,
            pitch: 0.0,
            last_applied_seq: None,
            prev_offset: LatticePos::from_metres(offset, vd_core::pose::Tier::Fine),
        },
    );
}

/// Co-hosting config: a shard hosting System 7 that ALSO CO-HOSTS Planet 7 (the un-hosted-child cure).
/// Its `held_realms` is `{System(7), Planet(7)}` — the primary realm plus the co-hosted child. The
/// region union (`realm_neighbourhood_for_held`) gives it BOTH neighbourhoods so the detector can
/// evaluate Planet 7's SOI from the System-7 authority.
fn cohost_planet_config() -> StubConfig {
    StubConfig {
        held_realms: BTreeSet::from([RealmId::System(7), RealmId::Planet(7)]),
        ..config()
    }
}

/// Grant the PRIMARY realm (System 7) AND affirm the CO-HOSTED child (Planet 7), populating both
/// `RealmAuthority` and `CoHostedAuthority` — the co-hosting boot state. Plant the UNION region set.
fn boot_cohost_planet(rig: &mut Rig) {
    grant_realm_for(rig, RealmId::System(7)); // primary → RealmAuthority
    grant_realm_for(rig, RealmId::Planet(7)); // co-hosted child → CoHostedAuthority
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vd_physics::worldgen::realm_neighbourhood_for_held(
            0,
            &BTreeSet::from([RealmId::System(7), RealmId::Planet(7)]),
        ));
}

// ---- Slice 4a (task #133): D-38 discharge + ledger-eviction + cell-carry SHAPE -----------------

/// The DEEPER CHILD region as a Spherical Shell (SOI descent) — the shape the G-IDENTICAL fixture
/// varies. Radius 1000, at the origin, nested under `OWN_REALM`, realm `OTHER_REALM`.
fn child_region_shell() -> RealmRegion {
    child_region()
}

/// The DEEPER CHILD region as a Cartesian Aabb (station volume) — mirrors `child_region_shell`
/// arg-for-arg (same realm/parent/center/band; the ONLY difference is the SHAPE). Half-extents 200 m.
fn child_region_aabb() -> RealmRegion {
    region_box(
        OTHER_REALM,
        Some(OWN_REALM),
        DVec3::ZERO,
        DVec3::new(200.0, 200.0, 200.0),
    )
}

/// The ONE crossing-feature fixture, driven by a CHILD-REGION-CONSTRUCTOR closure so its BODY is
/// written exactly once and run over two region shapes (§9's "same fixture on a Shell now, box at
/// P4"). Plants root ⊃ own ⊃ CHILD(make_child()) in `RealmRegions`, inserts an OWNED DURABLE dot
/// clearly OUTSIDE the child (signed_distance > 0 — no vacuous "already inside" pass), then drives it
/// INWARD across the child's acquire edge on a DIAGONAL segment (a velocity with ≥2 non-zero axis
/// components — so the Aabb run genuinely exercises `box_signed_distance`'s corner metric, which a
/// pure axis-aligned approach would skip; a shell's `|p|` distance handles the same diagonal unchanged,
/// so the body is identical). Returns the emitted `CrossingRequest`s + the dot's start/end SIGNED
/// DISTANCE to the child (the anti-vacuity guarantee — the region actually GATED the re-home, it was
/// not a no-op that fired on an already-inside dot) + the child shape.
fn drive_inward_crossing_feature(
    make_child: impl Fn() -> RealmRegion,
) -> (Vec<CrossingRequest>, f64, f64, Boundary) {
    let mut rig = Rig::new();
    rig.grant_realm();
    let child = make_child();
    let to_realm = child.realm; // OTHER_REALM
    let shape = child.shape;
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), own_region(), child]);
    // A distinct durable Player, placed on a DIAGONAL well OUTSIDE the child region.
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 0x4A);
    let outside = DVec3::new(1000.0, 1000.0, 0.0);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, outside);
    // Signed distance of the START pose to the child (proves it began OUTSIDE — positive distance).
    let start_sd = child_signed_distance(&child, outside);
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    // Approach inward along the diagonal, then hold deep inside (the band IS the dwell — one member
    // tick suffices, but hold a few so the run mirrors a real approach).
    let waypoints = [
        DVec3::new(700.0, 700.0, 0.0),
        DVec3::new(400.0, 400.0, 0.0),
        DVec3::new(50.0, 50.0, 0.0),
        DVec3::new(50.0, 50.0, 0.0),
        DVec3::new(50.0, 50.0, 0.0),
        DVec3::new(50.0, 50.0, 0.0),
    ];
    for (i, wp) in waypoints.iter().enumerate() {
        rig.set_local_tick(2 + i as u64);
        move_dot(&mut rig, TRIG_SESSION, *wp);
        all.extend(rig.tick(vec![]));
    }
    let inside = DVec3::new(50.0, 50.0, 0.0);
    let end_sd = child_signed_distance(&child, inside);
    // Only requests whose destination is THIS child's realm — filtering here proves the collection is
    // this feature's, not incidental noise.
    let reqs: Vec<CrossingRequest> = crossing_requests(&all)
        .into_iter()
        .filter(|r| r.to_realm == to_realm)
        .collect();
    (reqs, start_sd, end_sd, shape)
}

/// The signed distance from a frame-local `offset` to a region's surface (the exact scalar the
/// containment band consumes). The book places the region's frame at the identity — the P3 walk
/// shape — so the reframe is a no-op and this is the same value the detector reads.
fn child_signed_distance(region: &RealmRegion, offset: DVec3) -> f64 {
    use vd_core::geometry::region_signed_distance;
    let pose = StampedPose::at_rest(config().frame, offset, UniverseTick(100));
    let book = PlacementBook::new(
        pose.frame,
        pose.universe_tick,
        vec![(region.frame, FramePlacement::identity())],
    );
    region_signed_distance(&pose, region, &book).expect("identity reframe never errors")
}

/// S6 of the placement arc — SL4's own acceptance line, MEASURED: *"a ship, a station, a moon and
/// a rock cross by identical code because that code cannot tell them apart."* ONE fixture parks an
/// occupant while its realm's MOVING child sweeps over it. The fixture takes the child's motion as
/// the OPAQUE injected seam closure and CANNOT branch on what is inside — `MotionFn` has no arms
/// to match and this crate has no edge to the crate that could name them. The parked dot is Stage
/// B4's case: its stamp re-advances every tick, so the sweeping world is measured against NOW.
/// Returns the filtered re-home requests plus the child's distance to the dot at the sweep's start
/// and end (read THROUGH the seam — anti-vacuity without motion knowledge).
fn drive_swept_crossing_feature(
    motion: MotionFn,
    kind: NodeKind,
) -> (Vec<CrossingRequest>, f64, f64) {
    let mut rig = Rig::with_config_and_kind(config(), kind);
    rig.grant_realm();
    let child = region(OTHER_REALM, Some(OWN_REALM), DVec3::ZERO, 100.0);
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), own_region(), child])
            .with_moving_children(BTreeMap::from([(OTHER_REALM, motion.clone())]));
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 0x53);
    let parked = DVec3::new(1000.0, 0.0, 0.0);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, parked);
    let tick_hz = 20.0;
    let dist_at = |tick: u64| {
        ((motion.0)(tick as f64 / tick_hz)
            .anchor()
            .delta_m(vd_core::pose::LatticePos::ORIGIN, vd_core::pose::Tier::Fine)
            - parked)
            .length()
    };
    let (start, end) = (100u64, 160u64);
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in start..=end {
        rig.set_local_tick(t);
        rig.world.resource_mut::<ClockSample>().universe_tick = UniverseTick(t);
        all.extend(rig.tick(vec![]));
    }
    let reqs: Vec<CrossingRequest> = crossing_requests(&all)
        .into_iter()
        .filter(|r| r.to_realm == OTHER_REALM)
        .collect();
    (reqs, dist_at(start), dist_at(end))
}

/// 3f-D4 config: the shared trigger fixture with a NON-ZERO `request_ttl_ticks` so
/// `redrive_stranded_crossings` is armed (every other rig uses the disarmed `0`). The re-drive
/// budget rides THE production derivation (D-WORLD-2) so the re-drive tests run the shipped
/// patience ratio; the exhaustion tests pick smaller windows via [`config_with_ttl_and_budget`].
fn config_with_ttl(ttl: u32) -> StubConfig {
    config_with_ttl_and_budget(
        ttl,
        crate::saga::derive_crossing_redrive_budget(&crate::saga::SagaTuning::default()),
    )
}

/// D-WORLD-2 config: ttl AND re-drive budget explicit, for the exhaustion/backoff arcs.
fn config_with_ttl_and_budget(ttl: u32, budget: u32) -> StubConfig {
    StubConfig {
        request_ttl_ticks: ttl,
        crossing_redrive_budget: budget,
        handoff_hold_ttl_ticks: 0,
        ..config()
    }
}

/// D-WORLD-2 arc fixture: ttl 3, budget 2, `k_dwell` = `BoundaryTuning::DEFAULT.k_dwell` (5).
/// Timeline for a dot latched at tick 2 whose dest NEVER resolves (no orchestrator in the rig):
/// re-drives at 5 and 8, EXHAUSTION abort at 11, cooldown until 15, re-fire (attempt 1) at 16.
const XTTL: u32 = 3;
const XBUDGET: u32 = 2;

/// Drive an entity OUTWARD across a shell in ONE move (prev inside → cur far outside ⇒ swept
/// Outward, which `should_commit` commits IMMEDIATELY, no dwell). Returns the outbound frames of the
/// exit tick. Shared by the nested-undock and top-level-degrade tests so they differ ONLY in the
/// planted boundary's `parent`.
fn drive_outward_exit(rig: &mut Rig, entity: EntityId) -> Vec<(NodeId, MsgClass, Vec<u8>)> {
    // ESCAPE forest: root ⊃ parent(PARENT_REALM) ⊃ own(OWN_REALM, small shell r=1000). The shard owns
    // OWN_REALM; a dot leaving the small own-shell (but inside the parent) undocks OUTWARD to the parent.
    plant_escape_regions(rig);
    // Start INSIDE the own-shell (sd = 500 - 1000 ≪ -inset ⇒ container == OWN_REALM, no re-home yet).
    insert_owned_dot(rig, TRIG_SESSION, entity, DVec3::new(500.0, 0.0, 0.0));
    // Prime membership with one in-band tick.
    rig.set_local_tick(2);
    let _ = rig.tick(vec![]);
    // Step OUTSIDE the own-shell (sd = 5000 - 1000 ≫ +outset ⇒ own membership releases) but still
    // deep inside the PARENT shell ⇒ container flips to PARENT_REALM ⇒ re-home OUTWARD to the parent.
    move_dot(rig, TRIG_SESSION, DVec3::new(5000.0, 0.0, 0.0));
    rig.set_local_tick(3);
    rig.tick(vec![])
}

// ===== RLM Step 2 — demand-driven realm lifecycle (AoI) ==========================================

/// A live AoI band (base 1000 ⇒ spin_up 1000 m, tear_down 2000 m, `grace` ticks). The velocity-safe
/// ctor is the only way to build a live band; `v_rel = 0` here (the tests place occupants by geometry).
fn aoi_band(grace: u32) -> vd_core::geometry::AoiConfig {
    vd_core::geometry::AoiConfig::for_velocity_safe(1000.0, 1.0, 2.0, 0.0, 0.05, grace, 0.0)
        .expect("0 < spin_up < tear_down ⇒ a valid live band")
}

/// A Shell region with an EXPLICIT frame — `region`/`frame_of` resolves via `frame_for_realm(realm,
/// None)`, which returns `None` for an `Area` (Area needs its `Planet` parent); the HR4 fixture
/// supplies the frame here.
fn region_framed(
    realm: RealmId,
    parent: Option<RealmId>,
    center: DVec3,
    r: f64,
    frame: FrameRef,
) -> RealmRegion {
    RealmRegion {
        realm,
        center: vd_core::geometry::ParentCentre::authored(LatticePos::from_metres(
            center,
            vd_core::pose::Tier::Fine,
        )),
        frame,
        shape: Boundary::Shell { r },
        look: Some(Boundary::Shell { r }),
        band: band(),
        aoi: vd_core::geometry::AoiConfig::inert(),
        parent,
        interior_band: vd_core::geometry::AoiConfig::inert(),
    }
}

/// A small direct-child region (containment shell radius `r`) carrying a LIVE AoI band — frame via
/// `frame_of`. An occupant OUTSIDE the small containment shell but INSIDE the 1000 m AoI is the
/// "reaches before it enters" case the whole feature exists for.
fn aoi_child(realm: RealmId, parent: RealmId, r: f64, grace: u32) -> RealmRegion {
    RealmRegion {
        aoi: aoi_band(grace),
        ..region(realm, Some(parent), DVec3::ZERO, r)
    }
}

/// Like [`aoi_child`] but with an EXPLICIT frame (for the `Area` child `frame_of` cannot resolve).
fn aoi_child_framed(
    realm: RealmId,
    parent: Option<RealmId>,
    frame: FrameRef,
    grace: u32,
) -> RealmRegion {
    RealmRegion {
        aoi: aoi_band(grace),
        ..region_framed(realm, parent, DVec3::ZERO, 100.0, frame)
    }
}

fn plant_aoi(rig: &mut Rig, regions: Vec<RealmRegion>) {
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(regions);
}

/// Every `RealmDemand` in the outbox (decoded). The `_ => None` arm is exercised too — an AoI tick also
/// ships entity snapshots (non-`InterShardFlow` bytes) alongside the demand.
fn demands(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<RealmDemand> {
    sent.iter()
        .filter_map(
            |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                Ok(InterShardFlow::RealmDemand(d)) => Some(d),
                _ => None,
            },
        )
        .collect()
}

/// The child coord the loop names for `own_realm`'s child `child_realm` — `own_coord.child(level)`.
fn child_coord_of(own_realm: RealmId, child_realm: RealmId) -> RealmCoord {
    StubConfig::root_coord(own_realm).child(level_of(child_realm).expect("seed-lineage child"))
}

fn player(tag: u32) -> EntityId {
    EntityId::pack(EntityKind::Player, 10, 1, tag)
}

/// Every `RealmInterest` in the outbox (decoded, with its destination — the routing assert).
fn interests(
    sent: &[(NodeId, MsgClass, Vec<u8>)],
) -> Vec<(NodeId, vd_wire::intershard::RealmInterest)> {
    sent.iter()
        .filter_map(
            |(to, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                Ok(InterShardFlow::RealmInterest(ri)) => Some((*to, ri)),
                _ => None,
            },
        )
        .collect()
}

/// Every `WindowRelay` a tick shipped, with its destination — the Q2 relay's up-probe.
fn window_relays(
    sent: &[(NodeId, MsgClass, Vec<u8>)],
) -> Vec<(NodeId, vd_wire::intershard::WindowRelay)> {
    sent.iter()
        .filter_map(
            |(to, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                Ok(InterShardFlow::WindowRelay(r)) => Some((*to, r)),
                _ => None,
            },
        )
        .collect()
}

// ---- Step 5 slice B — FOLD an occupied child's bit into the parent's cull (the payoff) -----

/// A root System(7) PARENT shard with its realm granted and TWO armed Planet children — Planet(42) at the
/// origin and Planet(43) far away at `sibling_center` — the S2b-iii proxy-fold fixture.
fn parent_with_two_planet_children(sibling_center: DVec3) -> Rig {
    let cfg = StubConfig {
        realm: OWN_REALM,
        held_realms: StubConfig::single_realm(OWN_REALM),
        frame: frame_of(OWN_REALM),
        own_coord: StubConfig::root_coord(OWN_REALM),
        ..config()
    };
    let mut rig = Rig::with_config(cfg);
    grant_realm_for(&mut rig, OWN_REALM);
    let planet_a = aoi_child(RealmId::Planet(42), OWN_REALM, 1000.0, 0); // AoI band spin-up 1000, at origin
    let planet_b = RealmRegion {
        aoi: aoi_band(0),
        ..region(RealmId::Planet(43), Some(OWN_REALM), sibling_center, 1000.0)
    };
    plant_aoi(
        &mut rig,
        vec![root_region(), own_region(), planet_a, planet_b],
    );
    rig
}

/// The heartbeat-sender (home shard) an injected bit carries — the down-reflect return address.
const HOME_SHARD: NodeId = NodeId(70);

/// Inject a FRESH child bit (last_seen = the rig's current tick, home = [`HOME_SHARD`]) directly
/// into the parent store — the fold input, bypassing the separately-tested receive path.
fn inject_bit(rig: &mut Rig, child: RealmId) {
    let now = rig.world.resource::<ClockSample>().local_tick;
    rig.world.resource_mut::<ChildLiveness>().0.insert(
        child,
        ChildLiveEntry {
            home: HOME_SHARD,
            fence: Fence(1),
            at: UniverseTick(100),
            last_seen: now,
        },
    );
}

// ---- Step 5 slice C — the parent reflects the sibling scene DOWN to each live child ----------

// ---- Step 5 slice C — the child holds the from-above set and folds it into its client scenes --

/// A public `RealmShape` for `realm` (a Planet, so `frame_of` resolves) — PURE
/// SELF-DESCRIPTION (no position field exists). Only tombstoned-lane fixtures still build
/// one: since Slice C2 no message a realm receives carries another realm's outline.
fn render_shape(realm: RealmId) -> RealmShape {
    let r = region(realm, Some(ROOT_REALM), DVec3::ZERO, 1000.0);
    RealmShape {
        realm: r.realm,
        frame: r.frame,
        shape: r.shape,
        parent: r.parent,
    }
}

// ---- Slice 4 — the SHAPE lane descends the chain, one subtraction per level ------------------

/// Drive ONE AoI tick for a shard hosting `own_realm` with a live `child` region, an in-range occupant,
/// and a granted lease — returning every demand. The HR4 fixture runs this on two realm kinds.
fn drive_aoi_spinup(
    own_realm: RealmId,
    own_frame: FrameRef,
    child: RealmRegion,
) -> Vec<RealmDemand> {
    let cfg = StubConfig {
        realm: own_realm,
        held_realms: StubConfig::single_realm(own_realm),
        frame: own_frame,
        own_coord: StubConfig::root_coord(own_realm),
        ..config()
    };
    let mut rig = Rig::with_config(cfg);
    grant_realm_for(&mut rig, own_realm);
    let root = region(ROOT_REALM, None, DVec3::ZERO, 1.0e9);
    let own = region_framed(
        own_realm,
        Some(ROOT_REALM),
        DVec3::ZERO,
        100_000.0,
        own_frame,
    );
    plant_aoi(&mut rig, vec![root, own, child]);
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(500.0, 0.0, 0.0),
    );
    demands(&rig.tick(vec![]))
}

// ---- VU AoI S2a-2b-i — the parent-realm resolve/cache half ---------------------------------

/// Every parent-realm `HeadRead` directory key emitted this tick. The `_ => None` arm is exercised by
/// the demands + entity snapshots that ride the same tick.
fn headreads(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<DirectoryKey> {
    sent.iter()
        .filter_map(
            |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                Ok(InterShardFlow::Directory(DirectoryOp::HeadRead { key })) => Some(key),
                _ => None,
            },
        )
        .collect()
}

// ---- Step 5 — the SL7 ChildLive bit, upward (the deleted up-relay's replacement) -----------

/// Deliver a parent-realm Head reply naming `node` as the parent's authority — resolving
/// `ParentRealmNode` to it (the parent resolve every up-lane depends on). `node` is distinct from this
/// shard so `affirm_realm_head` takes its foreign no-op arm (never a spurious co-host insert).
fn resolve_parent_head(rig: &mut Rig, parent_realm: RealmId, node: NodeId) {
    let reply = DirectoryReply::Head {
        key: DirectoryKey::Realm(parent_realm),
        record: Some(vd_wire::seams::directory::OwnerRecord {
            authority: AuthorityRef::Shard(node),
            fence: Fence(1),
            lease_expires: UniverseTick(1_000),
            in_transfer: None,
        }),
    };
    let bytes = crate::io::bytes(
        postcard::to_allocvec(&InterShardFlow::DirectoryReply(reply)).expect("encode"),
    );
    let _ = rig.tick(vec![Inbound::Wire {
        from: ORCH,
        class: MsgClass::Saga,
        bytes,
    }]);
}

/// A parented PLANET shard (own Planet(42) under System(7)) with its OWN realm granted and an ARMED
/// child band planted — the up-lane fixture. `recheck` arms the parent-resolve cadence.
fn parented_aoi_rig(recheck: u64) -> Rig {
    parented_aoi_rig_holding(recheck, 0)
}

/// The same fixture with the HAND-OFF BUDGET armed to `hold_ttl` local ticks — the shard keeps
/// speaking for a subject it has handed away for that long, or until the take-over lands.
fn parented_aoi_rig_holding(recheck: u64, hold_ttl: u32) -> Rig {
    let cfg = StubConfig {
        realm: OTHER_REALM,
        held_realms: StubConfig::single_realm(OTHER_REALM),
        frame: frame_of(OTHER_REALM),
        own_coord: child_coord_of(OWN_REALM, OTHER_REALM),
        realm_recheck_interval: recheck,
        handoff_hold_ttl_ticks: hold_ttl,
        ..config()
    };
    let mut rig = Rig::with_config(cfg);
    grant_realm_for(&mut rig, OTHER_REALM);
    plant_aoi(
        &mut rig,
        vec![
            region(ROOT_REALM, None, DVec3::ZERO, 1.0e9),
            region_framed(
                OTHER_REALM,
                Some(ROOT_REALM),
                DVec3::ZERO,
                100_000.0,
                frame_of(OTHER_REALM),
            ),
            aoi_child_framed(
                RealmId::Area(99),
                Some(OTHER_REALM),
                FrameRef::AreaLocal {
                    planet_seed: 42,
                    area_seed: 99,
                },
                0,
            ),
        ],
    );
    rig
}

/// Every SL7 occupancy bit this tick shipped, with its destination — the up-liveness probe.
fn child_live_bits(
    sent: &[(NodeId, MsgClass, Vec<u8>)],
) -> Vec<(NodeId, vd_wire::intershard::ChildLive)> {
    sent.iter()
        .filter_map(
            |(to, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                Ok(InterShardFlow::ChildLive(cl)) => Some((*to, cl)),
                _ => None,
            },
        )
        .collect()
}

/// The ordered `Demote` for `entity` at the take-over fence — the message that starts a hand-off at
/// the source and, with a budget armed, opens the hold.
fn demote_msg(entity: EntityId, new_owner_fence: Fence) -> Inbound {
    wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::Demote(DemoteCmd {
            transfer: TransferId(7),
            subject: DirectoryKey::Entity(entity),
            new_owner_fence,
            step_id: DEMOTE_STEP,
        }),
    )
}

// ===== THE WINDOW LANE, Slice A (docs/design/window_lane.md §2.9/§4) ======================

/// Every `ShardToGateway::WindowFrame` in the outbox, decoded — `(to, window, at, hop, rows)`.
/// The `_ => None` arm is exercised by the entity/realm frames + demands riding the same tick.
#[allow(clippy::type_complexity)]
fn window_frames(
    sent: &[(NodeId, MsgClass, Vec<u8>)],
) -> Vec<(
    NodeId,
    WindowId,
    UniverseTick,
    Option<vd_wire::session_flow::HopRow>,
    Vec<RealmSnap>,
)> {
    sent.iter()
        .filter_map(
            |(node, _, b)| match postcard::from_bytes::<ShardToGateway>(b) {
                Ok(ShardToGateway::WindowFrame {
                    window,
                    at,
                    hop,
                    rows,
                    ..
                }) => Some((*node, window, at, hop.map(|h| *h), rows)),
                _ => None,
            },
        )
        .collect()
}

/// Every `ShardToGateway::WindowBody` in the outbox, decoded — `(to, window, subject, stmt)`.
/// Every `WindowStaticRows` in the outbox (slice S10) — the reliable lane's static roster.
fn window_static_rows(
    sent: &[(NodeId, MsgClass, Vec<u8>)],
) -> Vec<(NodeId, WindowId, Vec<vd_wire::channels::RealmSnap>)> {
    sent.iter()
        .filter_map(
            |(node, _, b)| match postcard::from_bytes::<ShardToGateway>(b) {
                Ok(ShardToGateway::WindowStaticRows { window, rows, .. }) => {
                    Some((*node, window, rows))
                }
                _ => None,
            },
        )
        .collect()
}

fn window_bodies(
    sent: &[(NodeId, MsgClass, Vec<u8>)],
) -> Vec<(NodeId, WindowId, RealmId, BodyStmt)> {
    sent.iter()
        .filter_map(
            |(node, _, b)| match postcard::from_bytes::<ShardToGateway>(b) {
                Ok(ShardToGateway::WindowBody {
                    window,
                    subject,
                    stmt,
                    ..
                }) => Some((*node, window, subject, stmt)),
                _ => None,
            },
        )
        .collect()
}

/// Every `ShardToGateway::WindowMembership` in the outbox, decoded.
fn window_memberships(
    sent: &[(NodeId, MsgClass, Vec<u8>)],
) -> Vec<(NodeId, WindowId, Vec<RealmId>, Vec<RealmId>)> {
    sent.iter()
        .filter_map(
            |(node, _, b)| match postcard::from_bytes::<ShardToGateway>(b) {
                Ok(ShardToGateway::WindowMembership {
                    window,
                    added,
                    removed,
                }) => Some((*node, window, added, removed)),
                _ => None,
            },
        )
        .collect()
}

/// Where the window fixtures place the armed child — off-axis and non-zero on every
/// component, so a dropped or transposed coordinate cannot pass as a lucky zero.
const WINDOW_CHILD_CENTER: DVec3 = DVec3::new(500.0, -20.0, 3.0);

/// One System(7) shard with an ARMED direct child at [`WINDOW_CHILD_CENTER`], a second STATIC
/// child (no luma, inert band — the marker-absent / verdict-absent arms), a marker bag for the
/// armed child, and one in-band occupant. The window fixtures' shared base.
fn window_rig() -> Rig {
    let mut rig = Rig::new();
    rig.grant_realm();
    let armed = RealmRegion {
        aoi: aoi_band(3),
        ..region(OTHER_REALM, Some(OWN_REALM), WINDOW_CHILD_CENTER, 100.0)
    };
    let quiet = region(
        RealmId::Planet(43),
        Some(OWN_REALM),
        DVec3::new(-40_000.0, 0.0, 0.0),
        100.0,
    );
    plant_aoi(&mut rig, vec![root_region(), own_region(), armed, quiet]);
    rig.world
        .resource_mut::<ChildLuma>()
        .0
        .insert(OTHER_REALM, (6, 0.25));
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    rig
}

/// Drive ONE window-emission tick for a shard of `kind` hosting `own_realm` with an armed
/// `child` at [`WINDOW_CHILD_CENTER`], one in-band occupant, the SAME marker bag, and BOTH
/// window scopes open — returning (frames, bodies, memberships). The HR4 fixture runs this on
/// two shard kinds (G-IDENTICAL).
#[allow(clippy::type_complexity)]
fn drive_window_emit(
    kind: NodeKind,
    own_realm: RealmId,
    own_frame: FrameRef,
    child: RealmRegion,
) -> (
    Vec<(
        NodeId,
        WindowId,
        UniverseTick,
        Option<vd_wire::session_flow::HopRow>,
        Vec<RealmSnap>,
    )>,
    Vec<(NodeId, WindowId, RealmId, BodyStmt)>,
    Vec<(NodeId, WindowId, Vec<RealmId>, Vec<RealmId>)>,
    // ★ S10: the STATIC roster, which is where a non-moving child's row rides now.
    Vec<(NodeId, WindowId, Vec<RealmSnap>)>,
) {
    let cfg = StubConfig {
        realm: own_realm,
        held_realms: StubConfig::single_realm(own_realm),
        frame: own_frame,
        own_coord: StubConfig::root_coord(own_realm),
        ..config()
    };
    let child_realm = child.realm;
    let mut rig = Rig::with_config_and_kind(cfg, kind);
    grant_realm_for(&mut rig, own_realm);
    let root = region(ROOT_REALM, None, DVec3::ZERO, 1.0e9);
    let own = region_framed(
        own_realm,
        Some(ROOT_REALM),
        DVec3::ZERO,
        100_000.0,
        own_frame,
    );
    plant_aoi(&mut rig, vec![root, own, child]);
    rig.world
        .resource_mut::<ChildLuma>()
        .0
        .insert(child_realm, (6, 0.25));
    insert_owned_dot_framed(
        &mut rig,
        TRIG_SESSION,
        player(7),
        own_frame,
        DVec3::new(500.0, 0.0, 0.0),
    );
    let sent = rig.tick(vec![
        wire_msg(
            GATEWAY,
            MsgClass::Control,
            &GatewayToShard::WindowOpen {
                window: WindowId(1),
                scope: WindowScope::Occupants,
            },
        ),
        wire_msg(
            GATEWAY,
            MsgClass::Control,
            &GatewayToShard::WindowOpen {
                window: WindowId(2),
                scope: WindowScope::Child(child_realm),
            },
        ),
    ]);
    (
        window_frames(&sent),
        window_bodies(&sent),
        window_memberships(&sent),
        window_static_rows(&sent),
    )
}
