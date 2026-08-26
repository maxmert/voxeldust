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

#[test]
fn the_placement_carry_cap_is_derived_from_the_tick_rate_and_never_zero() {
    // A NAMED derivation, not a literal: the cap is the configured wall-clock budget expressed in
    // whatever ticks this cluster runs at. It rounds UP and floors at one, because a cap of zero would
    // refuse every moving placement that arrived even a single tick late — i.e. all of them.
    //
    // It used to live on the GATEWAY's `TransportTuning`, which is a router's transport parameter; how
    // far a simulation may extrapolate along a velocity it authored is not the router's to decide, and
    // the router no longer applies a placement at all.
    let ms = PlacementCarry::BUDGET_MS;
    assert_eq!(
        PlacementCarry::skew_ticks_for(50),
        50 * ms / 1_000,
        "at 50 Hz the cap is the budget in ticks"
    );
    assert_eq!(PlacementCarry::skew_ticks_for(20), 20 * ms / 1_000);
    // A tick rate so slow that the budget is under one tick still yields one, not zero.
    assert_eq!(PlacementCarry::skew_ticks_for(1), 1);
    // A zero tick rate is a misconfiguration, not a division by zero.
    assert_eq!(PlacementCarry::skew_ticks_for(0), 1);
    // The struct constructor and the scalar are the SAME derivation, so a caller cannot get two answers.
    assert_eq!(
        PlacementCarry::for_tick_rate(50),
        PlacementCarry {
            max_skew_ticks: PlacementCarry::skew_ticks_for(50)
        }
    );
}

/// The detector's stamp invariant, all three arms by name (HR5): a LATCHED dot is exempt even
/// with a frozen stamp; a non-latched dot holding the clock's stamp passes.
#[test]
fn the_stamp_invariant_exempts_a_latched_dot_and_accepts_a_current_stamp() {
    let entity = EntityId::pack(EntityKind::Debris, 1, 7, 1);
    let mut latched = BTreeMap::new();
    latched.insert(entity, TransferId(1));
    debug_assert_nonlatched_stamp_is_current(&latched, entity, UniverseTick(1), UniverseTick(9));
    debug_assert_nonlatched_stamp_is_current(
        &BTreeMap::new(),
        entity,
        UniverseTick(9),
        UniverseTick(9),
    );
}

/// …and the panic arm: a non-latched dot whose stamp trails the clock is the `readvance_dots`
/// ordering guarantee broken — the invariant fails loudly rather than scanning a stale world.
#[test]
#[should_panic(expected = "a non-latched simulating dot's stamp must equal the clock")]
fn the_stamp_invariant_panics_on_a_non_latched_stale_stamp() {
    let entity = EntityId::pack(EntityKind::Debris, 1, 7, 1);
    debug_assert_nonlatched_stamp_is_current(
        &BTreeMap::new(),
        entity,
        UniverseTick(1),
        UniverseTick(9),
    );
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

#[test]
fn presence_absent_sends_no_greeting() {
    // No PresenceAnnounce ⇒ announce_presence no-ops ⇒ byte-identical (a static shard).
    let mut rig = Rig::new();
    assert!(presence_ticks(&rig.tick_raw(vec![])).is_empty());
}

#[test]
fn greets_every_booked_peer_while_silent() {
    let mut rig = Rig::new();
    rig.world
        .insert_resource(presence(&[GATEWAY, ANCESTOR, ORCH], 50));
    let ticks = presence_ticks(&rig.tick_raw(vec![]));
    assert_eq!(
        ticks.keys().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([GATEWAY, ANCESTOR, ORCH]),
    );
    // The greeting carries the shard's local tick (1 in a fresh rig).
    assert_eq!(ticks[&GATEWAY], TickId(1));
}

#[test]
fn greeting_rides_the_reliable_saga_class() {
    // Learning fires ONLY on a reliable frame; a datagram would teach nothing.
    let mut rig = Rig::new();
    rig.world.insert_resource(presence(&[GATEWAY], 50));
    let sent = rig.tick_raw(vec![]);
    let greeting = sent
        .iter()
        .find(|(_, _, flow, _)| matches!(flow, InterShardFlow::ShardPresence(_)))
        .expect("a greeting");
    assert_eq!(greeting.1, MsgClass::Saga);
}

#[test]
fn stays_silent_within_the_interval_after_greeting() {
    let mut rig = Rig::new();
    rig.world.insert_resource(presence(&[GATEWAY], 50));
    rig.tick_raw(vec![]); // tick 1: greet ⇒ last_contact[GATEWAY] = 1
    rig.set_local_tick(40); // 40 − 1 = 39 < 50
    assert!(presence_ticks(&rig.tick_raw(vec![])).is_empty());
}

#[test]
fn re_greets_after_an_interval_of_silence() {
    let mut rig = Rig::new();
    rig.world.insert_resource(presence(&[GATEWAY], 50));
    rig.tick_raw(vec![]); // tick 1: greet
    rig.set_local_tick(51); // 51 − 1 = 50 ≥ 50
    let ticks = presence_ticks(&rig.tick_raw(vec![]));
    assert_eq!(
        ticks.keys().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([GATEWAY]),
    );
    assert_eq!(ticks[&GATEWAY], TickId(51));
}

#[test]
fn inbound_from_a_peer_suppresses_only_that_peers_greeting() {
    let mut rig = Rig::new();
    rig.world
        .insert_resource(presence(&[GATEWAY, ANCESTOR], 50));
    // A frame from GATEWAY this tick = contact ⇒ GATEWAY silent; ANCESTOR (unheard) still greeted.
    let ticks = presence_ticks(&rig.tick_raw(vec![benign_contact(GATEWAY)]));
    assert_eq!(
        ticks.keys().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([ANCESTOR]),
    );
}

#[test]
fn a_non_wire_notice_is_not_contact() {
    let mut rig = Rig::new();
    rig.world.insert_resource(presence(&[GATEWAY], 50));
    // A NodeUnreachable notice is not an inbound frame ⇒ does NOT reset silence ⇒ GATEWAY greeted.
    let notice = Inbound::NodeUnreachable {
        to: GATEWAY,
        class: MsgClass::Saga,
        undelivered: MsgId(0),
    };
    assert!(presence_ticks(&rig.tick_raw(vec![notice])).contains_key(&GATEWAY));
}

#[test]
fn presence_new_rejects_a_zero_interval() {
    assert!(PresenceAnnounce::new(BTreeSet::from([GATEWAY]), 0).is_err());
    assert_eq!(
        PresenceAnnounce::new(BTreeSet::from([GATEWAY]), 1)
            .expect("ok")
            .interval_ticks,
        1,
    );
}

#[test]
fn greeting_is_g_identical_across_shard_kinds() {
    // HR4: the SAME greeting fixture on a System-realm shard and a Planet-realm shard emits the
    // IDENTICAL ShardPresence set — the greeting is shard-kind-blind (reachability, not gameplay).
    let mut system = Rig::with_config(config());
    let mut planet = {
        let mut cfg = config();
        cfg.realm = RealmId::Planet(7);
        cfg.held_realms = StubConfig::single_realm(RealmId::Planet(7));
        cfg.own_coord = StubConfig::root_coord(RealmId::Planet(7));
        Rig::with_config(cfg)
    };
    system
        .world
        .insert_resource(presence(&[GATEWAY, ANCESTOR], 50));
    planet
        .world
        .insert_resource(presence(&[GATEWAY, ANCESTOR], 50));
    assert_eq!(
        presence_ticks(&system.tick_raw(vec![])),
        presence_ticks(&planet.tick_raw(vec![])),
    );
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

#[test]
fn shard_profile_swap_is_capability_inert_for_every_profile_kind() {
    // RLM Step 5a ACCEPTANCE GATE. The shard boot swaps `NodeKind::StubShard` →
    // `NodeKind::Shard(profile_for(coord.profile_kind()))`. That swap MUST be CAPABILITY-INERT at P1–P3:
    // no system reads the carried `ShardProfile` caps to decide behavior yet, so a shard carrying ANY
    // profile emits BYTE-IDENTICAL authored frames + grants to the zero-cap StubShard shard, at the same
    // synced tick, with identical inputs. PROVEN here (not asserted) over EVERY `ProfileKind` — incl.
    // `Galaxy` (the one that carries `signal_relay`, which will uncorner P9 Signals). A future
    // `match kind` / capability gate that changes authored output would break this guard.
    use vd_core::taxonomy::ProfileKind;
    let baseline = Rig::with_config_and_kind(config(), NodeKind::StubShard).tick(vec![]);
    assert!(
        !baseline.is_empty(),
        "a synced shard authors + self-grants on an empty tick — a non-vacuous inertness baseline"
    );
    for k in ProfileKind::ALL {
        let profile =
            crate::capability::profile_for(k).expect("every ProfileKind yields a profile");
        let out = Rig::with_config_and_kind(config(), NodeKind::Shard(profile)).tick(vec![]);
        assert_eq!(
            out, baseline,
            "Shard({k:?}) authored output diverged from StubShard — the profile swap is NOT inert"
        );
    }
}

#[test]
fn transient_status_is_held_excludes_arriving_and_departing() {
    // Held + the pre-emit Crossing are authoritatively held (counted); Arriving (dest mid-flight)
    // AND Departing (source released, retained) are the two UNCOUNTED tiers (D-7b).
    assert!(TransientStatus::Held { outbound: None }.is_held());
    assert!(
        TransientStatus::Held {
            outbound: Some(TransferId(1))
        }
        .is_held()
    );
    assert!(
        TransientStatus::Crossing {
            dest: DEST_NODE,
            to_realm: RealmId::System(8),
            dst_realm_fence: Fence(2),
            batch: TransferId(1),
            to_parent: None,
        }
        .is_held()
    );
    assert!(
        !TransientStatus::Arriving {
            batch: TransferId(1)
        }
        .is_held()
    );
    assert!(
        !TransientStatus::Departing {
            batch: TransferId(1)
        }
        .is_held()
    );
}

#[test]
fn realm_lease_lapsed_requires_held_armed_channel_and_stale() {
    // The shared proactive self-fence predicate (HR5 branchless shim). Self-fence iff ALL hold: the
    // realm is HELD, the timer is ARMED (grace != 0), the confirmation CHANNEL exists (recheck != 0),
    // and the last confirmation is STALER than the grace. Each guard alone vetoes; the boundary
    // (`== grace`) is NOT lapsed (a strict `>`), so the holder keeps authority for the whole window.
    use crate::directory::lease_self_fence_due;
    let now = TickId(100);
    assert!(lease_self_fence_due(true, 5, 2, now, TickId(94))); // 6 > 5 ⇒ lapsed
    assert!(!lease_self_fence_due(false, 5, 2, now, TickId(94))); // not held
    assert!(!lease_self_fence_due(true, 0, 2, now, TickId(94))); // timer disarmed (pre-D-3 default)
    assert!(!lease_self_fence_due(true, 5, 0, now, TickId(94))); // no confirmation channel
    assert!(!lease_self_fence_due(true, 5, 2, now, TickId(95))); // 100-95 = 5 == grace ⇒ within, holds
}

#[test]
fn a_partitioned_holder_proactively_self_fences_then_re_arms_on_re_grant() {
    // D-3 Slice 5: the realm is granted (a round-trip confirmation at tick 1), then NO further
    // realm-head reply arrives (a partition from the orchestrator). Once `local_tick - confirmed`
    // exceeds the grace, the holder hard-stops its OWN authority and drops its held transients —
    // before the orchestrator's reassign window opens — so a stale owner can never affect clients.
    // A later re-grant re-arms the deadline, so a re-granted realm never inherits the stale one.
    let mut rig = Rig::with_config(StubConfig {
        realm_recheck_interval: 2, // the round-trip confirmation channel is active
        self_fence_grace_ticks: 5, // rig-local; ttl < grace <= ttl + max is validated orch-side
        ..config()
    });
    rig.grant_realm(); // confirm at local_tick 1 ⇒ RealmConfirmedAt(1), authority Some(Fence(1))
    let debris = EntityId::pack(EntityKind::Debris, 1, 7, 1);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        debris,
        Transient {
            pose: StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(98)),
            anchor_fence: Fence(1),
            status: TransientStatus::Held { outbound: None },
            prev_offset: LatticePos::ORIGIN,
        },
    );

    // Within the grace (local 6 - confirmed 1 = 5, NOT > 5): the holder KEEPS authority.
    rig.set_local_tick(6);
    let _ = rig.tick(vec![]);
    assert_eq!(rig.world.resource::<RealmAuthority>().0, Some(Fence(1)));
    assert_eq!(
        rig.world.resource::<StubStats>().realm_self_fenced_lapsed,
        0
    );
    assert!(
        rig.world
            .resource::<OwnedTransients>()
            .0
            .contains_key(&debris)
    );

    // Past the grace (local 7 - confirmed 1 = 6 > 5): SELF-FENCE — authority dropped, transient lost.
    rig.set_local_tick(7);
    let _ = rig.tick(vec![]);
    assert_eq!(
        rig.world.resource::<RealmAuthority>().0,
        None,
        "authority hard-stopped"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().realm_self_fenced_lapsed,
        1
    );
    assert_eq!(
        rig.world.resource::<StubStats>().transients_dropped,
        1,
        "the held transient anchored to the lost lease is a counted loss"
    );
    assert!(rig.world.resource::<OwnedTransients>().0.is_empty());

    // A re-grant at local 7 re-confirms (RealmConfirmedAt ⇒ 7); at local 11 (11-7 = 4 <= 5) the
    // holder is STILL authoritative — proof the re-granted realm did NOT inherit the stale deadline
    // (had `confirmed` stayed 1, 11-1 = 10 > 5 would have re-fenced it immediately).
    rig.grant_realm();
    rig.set_local_tick(11);
    let _ = rig.tick(vec![]);
    assert_eq!(
        rig.world.resource::<RealmAuthority>().0,
        Some(Fence(1)),
        "re-grant re-armed the timer"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().realm_self_fenced_lapsed,
        1,
        "no second self-fence"
    );
}

#[test]
fn readvance_advances_held_transients_and_skips_the_uncounted_tiers() {
    // D-7b: an AUTHORITATIVELY-held debris re-advances by its closed-form ballistic motion each
    // tick (vel·dt); the uncounted Arriving/Departing tiers are SKIPPED (not rendered, caught up
    // at promote). The Rig clock is universe_tick=100; a pose stamped at tick 98 advances dt =
    // (100-98)·tick_dt_s(0.05) = 0.1s.
    let mut rig = Rig::new();
    rig.grant_realm();
    let held = EntityId::pack(EntityKind::Debris, 1, 7, 1);
    let arriving = EntityId::pack(EntityKind::Debris, 1, 7, 2);
    let departing = EntityId::pack(EntityKind::Debris, 1, 7, 3);
    let moving = StampedPose {
        vel: DVec3::new(10.0, 0.0, 0.0),
        ..StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(98))
    };
    {
        let mut owned = rig.world.resource_mut::<OwnedTransients>();
        owned.0.insert(
            held,
            Transient {
                pose: moving,
                anchor_fence: Fence(1),
                status: TransientStatus::Held { outbound: None },
                prev_offset: LatticePos::ORIGIN,
            },
        );
        owned.0.insert(
            arriving,
            Transient {
                pose: moving,
                anchor_fence: Fence(1),
                status: TransientStatus::Arriving {
                    batch: TransferId(1),
                },
                prev_offset: LatticePos::ORIGIN,
            },
        );
        owned.0.insert(
            departing,
            Transient {
                pose: moving,
                anchor_fence: Fence(1),
                status: TransientStatus::Departing {
                    batch: TransferId(1),
                },
                prev_offset: LatticePos::ORIGIN,
            },
        );
    }
    let _ = rig.tick(vec![]);
    let owned = rig.world.resource::<OwnedTransients>();
    assert_eq!(
        fm(owned.0[&held].pose.pos),
        DVec3::new(1.0, 0.0, 0.0),
        "the held debris advanced by vel·dt (10 · 0.1)"
    );
    assert_eq!(
        owned.0[&held].pose.universe_tick,
        UniverseTick(100),
        "re-stamped to now"
    );
    assert_eq!(
        fm(owned.0[&arriving].pose.pos),
        DVec3::ZERO,
        "the uncounted Arriving tier is NOT advanced"
    );
    assert_eq!(
        fm(owned.0[&departing].pose.pos),
        DVec3::ZERO,
        "the uncounted Departing tier is NOT advanced"
    );
}

#[test]
fn emit_transient_batch_ships_one_envelope_and_marks_outbound() {
    let mut rig = Rig::new();
    rig.grant_realm(); // authority.0 = Some(Fence(1))
    let entity = EntityId::pack(EntityKind::Debris, 1, 7, 1);
    let batch = TransferId(0xB3);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        entity,
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: TransientStatus::Crossing {
                dest: DEST_NODE,
                to_realm: RealmId::System(8),
                dst_realm_fence: Fence(2),
                batch,
                to_parent: None,
            },
            prev_offset: LatticePos::ORIGIN,
        },
    );
    let sent = rig.tick(vec![]);
    // Exactly ONE TransientBatch envelope to the dest (G-TIER: one per batch), at the dst fence.
    let to_dest: Vec<_> = sent
        .iter()
        .filter(|(to, _, _)| *to == DEST_NODE)
        .map(|(_, _, bytes)| postcard::from_bytes::<InterShardFlow>(bytes).expect("decode"))
        .collect();
    assert_eq!(
        to_dest,
        vec![InterShardFlow::Transfer(TransferEnvelope {
            transfer_id: batch,
            universe_epoch: vd_core::EpochId(1),
            schema_version: TRANSFER_SCHEMA_VERSION,
            fence: Fence(2),
            step_id: TRANSIENT_BATCH_STEP,
            class: DurabilityClass::Transient,
            payload: TransitionPayload::TransientBatch {
                from_realm: config().realm,
                to_realm: RealmId::System(8),
                src_realm_fence: Fence(1),
                dst_realm_fence: Fence(2),
                source_tick: vd_core::TickId(1),
                items: vec![TransientItem {
                    entity,
                    // D-7b: `readvance_transients` ran BEFORE emit (Crossing is_held), re-stamping
                    // the pose to the source's current universe-tick (100); a rest pose's position
                    // is unchanged (vel ZERO), only the stamp moves.
                    //
                    // The pose then ships VERBATIM in THIS shard's own frame ({7}), because this rig
                    // plants no region forest and so `System(8)` is not one of its direct children —
                    // this shard has not been told where that realm is and has nothing to subtract.
                    // It used to ship stamped `{8}`: the frame flipped, the number did not move, and
                    // the receiver then read a source-frame position as if it were its own.
                    pose: StampedPose {
                        universe_tick: UniverseTick(100),
                        ..transient_pose()
                    },
                    state: vec![],
                }],
            },
        })],
    );
    // The emitted item is now Held{outbound} (still authoritative — adopt-before-drop).
    assert_eq!(
        rig.world.resource::<OwnedTransients>().0[&entity].status,
        TransientStatus::Held {
            outbound: Some(batch)
        }
    );
    assert_eq!(rig.world.resource::<StubStats>().transients_emitted, 1);
}

#[test]
fn a_transient_whose_flush_refuses_returns_to_held_and_ships_nothing() {
    // Stage B1 reaches the TRANSIENT lane too: a crossing item whose re-read pose is still held
    // by this shard's own band (the departure is no longer true) is NOT shipped — it returns to
    // plain `Held`, this shard keeps authority, and the scan re-decides. Shipping anyway would
    // hand a peer a position nobody vouches for; dropping while marked sent would strand it.
    let mut rig = Rig::new();
    rig.grant_realm();
    // A roster whose OWN region still HOLDS the transient's pose (5 m inside a 1000 m shell).
    plant_aoi(&mut rig, vec![root_region(), own_region()]);
    let entity = EntityId::pack(EntityKind::Debris, 1, 7, 1);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        entity,
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: TransientStatus::Crossing {
                dest: DEST_NODE,
                to_realm: RealmId::System(8),
                dst_realm_fence: Fence(2),
                batch: TransferId(0xB3),
                to_parent: None,
            },
            prev_offset: LatticePos::ORIGIN,
        },
    );
    let sent = rig.tick(vec![]);
    assert!(
        !sent.iter().any(|(to, _, _)| *to == DEST_NODE),
        "the refused item ships nothing: {sent:?}"
    );
    assert_eq!(
        rig.world.resource::<OwnedTransients>().0[&entity].status,
        TransientStatus::Held { outbound: None },
        "the item is back to plain Held — this shard is still its authority"
    );
    assert_eq!(rig.world.resource::<StubStats>().transients_emitted, 0);
}

#[test]
fn emit_transient_batch_ships_nothing_without_a_realm_lease() {
    let mut rig = Rig::new(); // NO grant_realm → authority.0 = None
    let entity = EntityId::pack(EntityKind::Debris, 1, 7, 1);
    let crossing = TransientStatus::Crossing {
        dest: DEST_NODE,
        to_realm: RealmId::System(8),
        dst_realm_fence: Fence(2),
        batch: TransferId(0xB3),
        to_parent: None,
    };
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        entity,
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: crossing,
            prev_offset: LatticePos::ORIGIN,
        },
    );
    let sent = rig.tick(vec![]);
    assert!(
        sent.iter().all(|(to, _, _)| *to != DEST_NODE),
        "a shard without its realm lease ships no transient batch"
    );
    // The Crossing item is UNCHANGED (it retries when the lease arrives).
    assert_eq!(
        rig.world.resource::<OwnedTransients>().0[&entity].status,
        crossing
    );
    assert_eq!(rig.world.resource::<StubStats>().transients_emitted, 0);
}

#[test]
fn adopt_transient_batch_adopts_arriving_acks_and_dedups() {
    let batch = TransferId(0xB1);
    let entity = EntityId::pack(EntityKind::Debris, 1, 7, 0);
    let cfg = config();
    let regions = RealmRegions::default();
    let placements = PlacementLedger::default();
    let mut owned = OwnedTransients::default();
    let mut applied = AppliedSteps::default();
    let mut stats = StubStats::default();
    let mut outbox = OutboundBox::default();
    let items = vec![TransientItem {
        entity,
        pose: transient_pose(),
        state: vec![],
    }];

    // FIRST delivery: adopt as Arriving (uncounted) anchored to the dst fence + ack BatchAdopted.
    // The pose is in the dest realm's OWN frame, so the receiver conversion's verbatim arm runs.
    adopt_transient_batch(
        batch,
        cfg.realm,
        Fence(5),
        items.clone(),
        &cfg,
        &regions,
        &placements,
        &mut owned,
        &mut applied,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(owned.0[&entity].status, TransientStatus::Arriving { batch });
    assert_eq!(owned.0[&entity].anchor_fence, Fence(5));
    assert_eq!(stats.transients_adopted, 1);
    assert_eq!(stats.transient_arrivals_unplaceable, 0);
    assert_eq!(stats.transients_adopt_redelivered, 0);
    assert_eq!(
        decode_flows(&mut outbox),
        vec![(
            ORCH,
            InterShardFlow::TransferAck(TransferAck::BatchAdopted {
                transfer_id: batch,
                step_id: TRANSIENT_BATCH_STEP,
            })
        )]
    );

    // REDELIVERY: no re-adopt, re-ack only (at-least-once — the ack may have been lost).
    adopt_transient_batch(
        batch,
        cfg.realm,
        Fence(5),
        items,
        &cfg,
        &regions,
        &placements,
        &mut owned,
        &mut applied,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.transients_adopted, 1, "not re-adopted");
    assert_eq!(stats.transients_adopt_redelivered, 1);
    assert_eq!(decode_flows(&mut outbox).len(), 1, "re-acked exactly once");
}

#[test]
fn adopt_transient_batch_refuses_an_unplaceable_item_counted_still_acks() {
    // THE RECEIVER-SIDE FRAME GUARD on the transient tier (audit :105/:374/:384 — the adopt used
    // to store the wire pose VERBATIM): an item whose pose this shard cannot measure — here a
    // frame it holds no placement book for — is REFUSED + counted, never inserted, while the
    // batch itself still journals + acks (adopt-before-drop proceeds; the loss is per-item and
    // accounted, the Transient class budget discipline).
    let batch = TransferId(0xB4);
    let entity = EntityId::pack(EntityKind::Debris, 1, 7, 9);
    let cfg = config();
    let regions = RealmRegions::default();
    let placements = PlacementLedger::default();
    let mut owned = OwnedTransients::default();
    let mut applied = AppliedSteps::default();
    let mut stats = StubStats::default();
    let mut outbox = OutboundBox::default();
    let misframed = StampedPose::at_rest(
        FrameRef::SystemSpace { system_seed: 99 }, // a sibling frame — nobody told us its placement
        DVec3::new(3.0, 0.0, 0.0),
        UniverseTick(5),
    );
    adopt_transient_batch(
        batch,
        cfg.realm,
        Fence(5),
        vec![TransientItem {
            entity,
            pose: misframed,
            state: vec![],
        }],
        &cfg,
        &regions,
        &placements,
        &mut owned,
        &mut applied,
        &mut stats,
        &mut outbox,
    );
    assert!(
        owned.0.is_empty(),
        "the mis-framed item is NOT adopted — nothing stored verbatim"
    );
    assert_eq!(
        stats.transient_arrivals_unplaceable, 1,
        "the refusal is counted"
    );
    assert_eq!(stats.transients_adopted, 0);
    assert_eq!(
        decode_flows(&mut outbox),
        vec![(
            ORCH,
            InterShardFlow::TransferAck(TransferAck::BatchAdopted {
                transfer_id: batch,
                step_id: TRANSIENT_BATCH_STEP,
            })
        )],
        "the batch still acks — the refusal is per-item, the choreography completes"
    );
}

#[test]
fn transient_release_promote_complete_lifecycle_and_dedup() {
    // The full D-7b source/dest handler lifecycle on ONE mixed owned set (the handlers walk by
    // status; source-vs-dest is just which statuses are present in production): RELEASE flips
    // `Held{Some}→Departing` (uncounted), PROMOTE flips `Arriving→Held` (re-anchored), COMPLETE
    // retires `Departing` — each journaled/state idempotent (redelivery = counted no-op + re-ack).
    let this = TransferId(0xB2);
    let other = TransferId(0xB9);
    let held_this = EntityId::pack(EntityKind::Debris, 1, 7, 1);
    let held_other = EntityId::pack(EntityKind::Debris, 1, 7, 2);
    let held_settled = EntityId::pack(EntityKind::Debris, 1, 7, 3);
    let arriving_this = EntityId::pack(EntityKind::Debris, 1, 7, 4);
    let arriving_other = EntityId::pack(EntityKind::Debris, 1, 7, 5);
    let crossing = EntityId::pack(EntityKind::Debris, 1, 7, 6);
    let departing_other = EntityId::pack(EntityKind::Debris, 1, 7, 7);
    let mk = |status| Transient {
        pose: transient_pose(),
        anchor_fence: Fence(2),
        status,
        prev_offset: LatticePos::ORIGIN,
    };
    let mut owned = OwnedTransients::default();
    owned.0.insert(
        held_this,
        mk(TransientStatus::Held {
            outbound: Some(this),
        }),
    );
    owned.0.insert(
        held_other,
        mk(TransientStatus::Held {
            outbound: Some(other),
        }),
    );
    owned
        .0
        .insert(held_settled, mk(TransientStatus::Held { outbound: None }));
    owned
        .0
        .insert(arriving_this, mk(TransientStatus::Arriving { batch: this }));
    owned.0.insert(
        arriving_other,
        mk(TransientStatus::Arriving { batch: other }),
    );
    owned.0.insert(
        crossing,
        mk(TransientStatus::Crossing {
            dest: DEST_NODE,
            to_realm: RealmId::System(8),
            dst_realm_fence: Fence(9),
            batch: other,
            to_parent: None,
        }),
    );
    owned.0.insert(
        departing_other,
        mk(TransientStatus::Departing { batch: other }),
    );
    let mut applied = AppliedSteps::default();
    let mut stats = StubStats::default();
    let cfg = config();
    let mut outbox = OutboundBox::default();

    // RELEASE (SOURCE): `Held{Some(this)}` → uncounted `Departing`; `Held{Some(other)}` and every
    // other status untouched; ack `DropApplied`(RELEASE_STEP) to the orchestrator.
    let rel = TransientHandoff {
        transfer: this,
        step_id: TRANSIENT_RELEASE_STEP,
        fence: Fence(5),
    };
    on_transient_release(rel, &mut owned, &mut applied, &mut stats, &cfg, &mut outbox);
    assert_eq!(
        owned.0[&held_this].status,
        TransientStatus::Departing { batch: this },
        "the source's Held item for THIS batch is released to the uncounted Departing tier"
    );
    assert_eq!(
        owned.0[&held_other].status,
        TransientStatus::Held {
            outbound: Some(other)
        },
        "a Held item for ANOTHER batch is untouched"
    );
    assert_eq!(
        owned.0[&held_settled].status,
        TransientStatus::Held { outbound: None }
    );
    assert_eq!(stats.transients_handed_off, 1);
    assert_eq!(
        decode_flows(&mut outbox),
        vec![(
            ORCH,
            InterShardFlow::TransferAck(TransferAck::DropApplied {
                transfer_id: this,
                step_id: TRANSIENT_RELEASE_STEP,
            })
        )]
    );
    // RELEASE REDELIVERY: re-ack, no re-flip.
    on_transient_release(rel, &mut owned, &mut applied, &mut stats, &cfg, &mut outbox);
    assert_eq!(stats.transient_release_noop, 1);
    assert_eq!(stats.transients_handed_off, 1, "not re-released");
    assert_eq!(decode_flows(&mut outbox).len(), 1, "re-acked exactly once");

    // PROMOTE (DEST): `Arriving{this}` → `Held{None}` re-anchored to the commit fence;
    // `Arriving{other}` untouched; ack `DropApplied`(DROP_STEP).
    let promote = TransientHandoff {
        transfer: this,
        step_id: TRANSIENT_DROP_STEP,
        fence: Fence(5),
    };
    on_transient_promote(
        promote,
        &mut owned,
        &mut applied,
        &mut stats,
        &cfg,
        &mut outbox,
    );
    assert_eq!(
        owned.0[&arriving_this].status,
        TransientStatus::Held { outbound: None },
        "THIS batch's Arriving is promoted to authoritative Held"
    );
    assert_eq!(
        owned.0[&arriving_this].anchor_fence,
        Fence(5),
        "the promoted item re-anchors to the batch's commit fence"
    );
    assert_eq!(
        owned.0[&arriving_other].status,
        TransientStatus::Arriving { batch: other },
        "another batch's Arriving is untouched"
    );
    assert!(
        owned.0.contains_key(&crossing),
        "a pending Crossing is untouched"
    );
    assert_eq!(stats.transients_promoted, 1);
    assert_eq!(
        decode_flows(&mut outbox),
        vec![(
            ORCH,
            InterShardFlow::TransferAck(TransferAck::DropApplied {
                transfer_id: this,
                step_id: TRANSIENT_DROP_STEP,
            })
        )]
    );
    // PROMOTE REDELIVERY: re-ack, no re-flip.
    on_transient_promote(
        promote,
        &mut owned,
        &mut applied,
        &mut stats,
        &cfg,
        &mut outbox,
    );
    assert_eq!(stats.transient_drop_noop, 1);
    assert_eq!(stats.transients_promoted, 1, "not re-promoted");
    assert_eq!(decode_flows(&mut outbox).len(), 1, "re-acked exactly once");

    // COMPLETE (SOURCE): retire THIS batch's `Departing` copy (held_this); a `Departing` for
    // ANOTHER batch (departing_other) is untouched; D-7d: ALWAYS ack DropApplied(COMPLETE_STEP) —
    // the `SourceRetired` signal that drives the saga's BatchHandoff tail to Done.
    let rc = TransientHandoff {
        transfer: this,
        step_id: TRANSIENT_RELEASE_STEP,
        fence: Fence(5),
    };
    on_release_complete(rc, &mut owned, &mut stats, &cfg, &mut outbox);
    assert!(
        !owned.0.contains_key(&held_this),
        "the retained Departing copy for THIS batch is retired"
    );
    assert_eq!(
        owned.0[&departing_other].status,
        TransientStatus::Departing { batch: other },
        "a Departing copy for ANOTHER batch is untouched"
    );
    assert_eq!(
        decode_flows(&mut outbox),
        vec![(
            ORCH,
            InterShardFlow::TransferAck(TransferAck::DropApplied {
                transfer_id: this,
                step_id: TRANSIENT_COMPLETE_STEP,
            })
        )],
        "ReleaseComplete acks the retire-complete so the saga tail reaches Done (SourceRetired)"
    );
    // COMPLETE REDELIVERY: no Departing for THIS batch → counted no-op, but STILL acks
    // (at-least-once — the orchestrator's tombstoned saga absorbs the duplicate).
    on_release_complete(rc, &mut owned, &mut stats, &cfg, &mut outbox);
    assert_eq!(
        stats.transient_release_noop, 2,
        "the redelivered complete is a counted no-op"
    );
    assert_eq!(
        decode_flows(&mut outbox).len(),
        1,
        "the redelivery still acks (at-least-once)"
    );
}

#[test]
fn self_fence_drops_held_transients_as_a_counted_loss() {
    // A realm takeover (the lease now held by someone else) self-fences the shard AND drops its
    // transients (anchored to the now-lost lease) as a counted LOSS — durable dots are retained.
    let mut rig = Rig::new();
    rig.grant_realm();
    let entity = EntityId::pack(EntityKind::Debris, 1, 7, 1);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        entity,
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: TransientStatus::Held { outbound: None },
            prev_offset: LatticePos::ORIGIN,
        },
    );
    let takeover = DirectoryReply::Head {
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
        &InterShardFlow::DirectoryReply(takeover),
    )]);
    assert!(
        rig.world.resource::<OwnedTransients>().0.is_empty(),
        "transients dropped on self-fence"
    );
    assert_eq!(rig.world.resource::<StubStats>().transients_dropped, 1);
    assert!(
        rig.world.resource::<RealmAuthority>().0.is_none(),
        "the shard self-fenced its realm"
    );
}

#[test]
fn self_fence_drops_held_transients_on_realm_revoke() {
    // The revoked arm (the realm record is GONE) also self-fences + drops transients — the second
    // `self_fence_drop_transients` call site (a revoke vs a takeover).
    let mut rig = Rig::new();
    rig.grant_realm();
    let entity = EntityId::pack(EntityKind::Debris, 1, 7, 2);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        entity,
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: TransientStatus::Held { outbound: None },
            prev_offset: LatticePos::ORIGIN,
        },
    );
    let revoked = DirectoryReply::Head {
        key: DirectoryKey::Realm(config().realm),
        record: None,
    };
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::DirectoryReply(revoked),
    )]);
    assert!(rig.world.resource::<OwnedTransients>().0.is_empty());
    assert_eq!(rig.world.resource::<StubStats>().transients_dropped, 1);
    assert!(rig.world.resource::<RealmAuthority>().0.is_none());
}

#[test]
fn transient_status_is_in_handover_excludes_only_settled_held() {
    // D-7b.3: every tier EXCEPT a settled `Held{outbound: None}` is a handover loss if dropped.
    assert!(
        !TransientStatus::Held { outbound: None }.is_in_handover(),
        "a settled Held is a resident eviction, not a handover loss"
    );
    assert!(
        TransientStatus::Held {
            outbound: Some(TransferId(1))
        }
        .is_in_handover()
    );
    assert!(
        TransientStatus::Crossing {
            dest: DEST_NODE,
            to_realm: RealmId::System(8),
            dst_realm_fence: Fence(2),
            batch: TransferId(1),
            to_parent: None,
        }
        .is_in_handover()
    );
    assert!(
        TransientStatus::Arriving {
            batch: TransferId(1)
        }
        .is_in_handover()
    );
    assert!(
        TransientStatus::Departing {
            batch: TransferId(1)
        }
        .is_in_handover()
    );
}

#[test]
fn self_fence_buckets_handover_loss_per_kind_and_excludes_resident_and_corrupt() {
    // D-7b.3: a realm self-fence buckets ONLY handover-status items into the per-kind
    // handover-loss counter (the budget gate's input), keyed by kind; a settled `Held{None}` is a
    // resident eviction (NOT bucketed); a corrupt kind tag is counted GROSS but NOT bucketed (the
    // `from_tag` Err arm). The gross `transients_dropped` still counts EVERY tier.
    let t = TransferId(0xC0);
    let mk = |status| Transient {
        pose: transient_pose(),
        anchor_fence: Fence(2),
        status,
        prev_offset: LatticePos::ORIGIN,
    };
    let mut owned = OwnedTransients::default();
    // Four Debris in handover (one per tier).
    owned.0.insert(
        EntityId::pack(EntityKind::Debris, 1, 1, 0),
        mk(TransientStatus::Held { outbound: Some(t) }),
    );
    owned.0.insert(
        EntityId::pack(EntityKind::Debris, 1, 2, 0),
        mk(TransientStatus::Crossing {
            dest: DEST_NODE,
            to_realm: RealmId::System(8),
            dst_realm_fence: Fence(2),
            batch: t,
            to_parent: None,
        }),
    );
    owned.0.insert(
        EntityId::pack(EntityKind::Debris, 1, 3, 0),
        mk(TransientStatus::Arriving { batch: t }),
    );
    owned.0.insert(
        EntityId::pack(EntityKind::Debris, 1, 4, 0),
        mk(TransientStatus::Departing { batch: t }),
    );
    // A Projectile in handover (proves per-kind disentangling).
    owned.0.insert(
        EntityId::pack(EntityKind::Projectile, 1, 5, 0),
        mk(TransientStatus::Held { outbound: Some(t) }),
    );
    // A SETTLED Debris (resident eviction — NOT a handover loss).
    owned.0.insert(
        EntityId::pack(EntityKind::Debris, 1, 6, 0),
        mk(TransientStatus::Held { outbound: None }),
    );
    // A corrupt kind tag (99) in handover — counted gross, NOT bucketed (the from_tag Err arm).
    owned.0.insert(
        EntityId(99u128 << 120),
        mk(TransientStatus::Departing { batch: t }),
    );
    let mut stats = StubStats::default();

    self_fence_drop_transients(&mut owned, &mut stats);
    assert!(
        owned.0.is_empty(),
        "every transient is dropped on self-fence"
    );
    assert_eq!(stats.transients_dropped, 7, "gross counts EVERY tier");
    assert_eq!(
        stats.transients_lost_in_handover.get(&EntityKind::Debris),
        Some(&4),
        "4 Debris in handover bucketed (the settled one + the corrupt one excluded)"
    );
    assert_eq!(
        stats
            .transients_lost_in_handover
            .get(&EntityKind::Projectile),
        Some(&1),
        "the Projectile is bucketed under its OWN kind (per-kind disentangling)"
    );
}

#[test]
fn on_transient_abandon_drops_the_batch_as_accounted_loss_and_is_idempotent() {
    // D-7d dead-DEST resolution: abandon drops THIS batch's retained items — both `Departing{this}`
    // (already released) AND `Held{Some(this)}` (the dest died before release) — bucketing each into
    // the per-kind loss budget + counting departure_cancelled. A `Held{None}` resident, a
    // `Departing{other}` batch, and a `Held{Some(other)}` batch are UNTOUCHED. A corrupt kind tag is
    // removed but NOT bucketed (the from_tag Err arm). A redelivery is a journaled no-op.
    let this = TransferId(0xD7D);
    let other = TransferId(0xBEEF);
    let mk = |status| Transient {
        pose: transient_pose(),
        anchor_fence: Fence(3),
        status,
        prev_offset: LatticePos::ORIGIN,
    };
    let mut owned = OwnedTransients::default();
    let dep_this = EntityId::pack(EntityKind::Debris, 1, 1, 0);
    let held_this = EntityId::pack(EntityKind::Debris, 1, 2, 0);
    let resident = EntityId::pack(EntityKind::Debris, 1, 3, 0);
    let dep_other = EntityId::pack(EntityKind::Debris, 1, 4, 0);
    let held_other = EntityId::pack(EntityKind::Debris, 1, 5, 0);
    let corrupt = EntityId(99u128 << 120); // tag 99 → from_tag Err (removed, not bucketed)
    owned
        .0
        .insert(dep_this, mk(TransientStatus::Departing { batch: this }));
    owned.0.insert(
        held_this,
        mk(TransientStatus::Held {
            outbound: Some(this),
        }),
    );
    owned
        .0
        .insert(resident, mk(TransientStatus::Held { outbound: None }));
    owned
        .0
        .insert(dep_other, mk(TransientStatus::Departing { batch: other }));
    owned.0.insert(
        held_other,
        mk(TransientStatus::Held {
            outbound: Some(other),
        }),
    );
    owned
        .0
        .insert(corrupt, mk(TransientStatus::Departing { batch: this }));
    let mut applied = AppliedSteps::default();
    let mut stats = StubStats::default();
    let abandon = TransientHandoff {
        transfer: this,
        step_id: TRANSIENT_ABANDON_STEP,
        fence: Fence(3),
    };

    on_transient_abandon(abandon, &mut owned, &mut applied, &mut stats);
    // THIS batch's Departing + Held{Some} + the corrupt one are dropped.
    assert!(!owned.0.contains_key(&dep_this));
    assert!(!owned.0.contains_key(&held_this));
    assert!(!owned.0.contains_key(&corrupt));
    // The resident + the OTHER batch's copies survive (not this batch).
    assert!(owned.0.contains_key(&resident));
    assert!(owned.0.contains_key(&dep_other));
    assert!(owned.0.contains_key(&held_other));
    assert_eq!(
        stats.transients_departure_cancelled, 3,
        "3 items abandoned (2 Debris + 1 corrupt)"
    );
    assert_eq!(
        stats.transients_lost_in_handover.get(&EntityKind::Debris),
        Some(&2),
        "only the 2 Debris are bucketed (the corrupt kind is removed but not attributable)"
    );

    // REDELIVERY: the journal short-circuits — no double-count, the survivors are untouched.
    on_transient_abandon(abandon, &mut owned, &mut applied, &mut stats);
    assert_eq!(
        stats.transients_departure_cancelled, 3,
        "the redelivery did not re-count"
    );
    assert_eq!(
        stats.transient_release_noop, 1,
        "the redelivery is a counted journal no-op"
    );
}

#[test]
fn on_transient_discard_removes_arriving_poisons_adopt_and_is_idempotent() {
    // R-6d3c NEVER-restart resolution (DEST role): the source died in AwaitAdopt pre-adopt, so the
    // discard REMOVES this batch's `Arriving{this}` items as an accounted loss — a decodable Debris
    // is bucketed, a corrupt kind tag is removed but NOT bucketed (the from_tag Err arm). An
    // `Arriving{other}` (batch mismatch), a `Held{None}` resident, and a `Departing{this}` (non-
    // Arriving) are UNTOUCHED — the false arms. A redelivery is a journaled no-op (loss counts once).
    let this = TransferId(0x6D3C);
    let other = TransferId(0xBEEF);
    let mk = |status| Transient {
        pose: transient_pose(),
        anchor_fence: Fence(4),
        status,
        prev_offset: LatticePos::ORIGIN,
    };
    let mut owned = OwnedTransients::default();
    let arr_this = EntityId::pack(EntityKind::Debris, 1, 1, 0);
    let corrupt = EntityId(99u128 << 120); // tag 99 → from_tag Err (removed, not bucketed)
    let arr_other = EntityId::pack(EntityKind::Debris, 1, 2, 0);
    let resident = EntityId::pack(EntityKind::Debris, 1, 3, 0);
    let dep_this = EntityId::pack(EntityKind::Debris, 1, 4, 0);
    owned
        .0
        .insert(arr_this, mk(TransientStatus::Arriving { batch: this }));
    owned
        .0
        .insert(corrupt, mk(TransientStatus::Arriving { batch: this }));
    owned
        .0
        .insert(arr_other, mk(TransientStatus::Arriving { batch: other }));
    owned
        .0
        .insert(resident, mk(TransientStatus::Held { outbound: None }));
    owned
        .0
        .insert(dep_this, mk(TransientStatus::Departing { batch: this }));
    let mut applied = AppliedSteps::default();
    let mut stats = StubStats::default();
    let discard = TransientHandoff {
        transfer: this,
        step_id: TRANSIENT_DISCARD_STEP,
        fence: Fence(4),
    };

    on_transient_discard(discard, &mut owned, &mut applied, &mut stats);
    // THIS batch's Arriving items (the decodable + the corrupt) are removed.
    assert!(!owned.0.contains_key(&arr_this));
    assert!(!owned.0.contains_key(&corrupt));
    // The OTHER batch's Arriving, the settled resident, and a Departing item survive (false arms).
    assert!(owned.0.contains_key(&arr_other));
    assert!(owned.0.contains_key(&resident));
    assert!(owned.0.contains_key(&dep_this));
    assert_eq!(
        stats.transients_discarded_source_crash, 2,
        "2 Arriving items discarded (1 Debris + 1 corrupt)"
    );
    assert_eq!(
        stats.transients_lost_in_handover.get(&EntityKind::Debris),
        Some(&1),
        "only the decodable Debris is bucketed (the corrupt kind is removed but not attributable)"
    );

    // REDELIVERY: the journal short-circuits — no double-count, the survivors are untouched.
    on_transient_discard(discard, &mut owned, &mut applied, &mut stats);
    assert_eq!(
        stats.transients_discarded_source_crash, 2,
        "the redelivery did not re-count"
    );
    assert_eq!(
        stats.transient_release_noop, 1,
        "the redelivery is a counted journal no-op"
    );
    assert_eq!(
        owned.0.len(),
        3,
        "the survivors are untouched by the redelivery"
    );
}

#[test]
fn discard_before_adopt_poisons_so_a_late_replay_never_orphans() {
    // THE target interleave (Defect A closed): the discard fires FIRST on an EMPTY owned set (the
    // dest never received the batch — the source died pre-adopt) → it removes nothing but POISONS
    // `(transfer, TRANSIENT_BATCH_STEP)`. A LATE outbox replay of the batch then adopts as
    // `AlreadyApplied` — inserting NOTHING — so no `Arriving` orphan is ever stranded.
    let this = TransferId(0x6D3C);
    let entity = EntityId::pack(EntityKind::Debris, 1, 7, 0);
    let mut owned = OwnedTransients::default();
    let mut applied = AppliedSteps::default();
    let mut stats = StubStats::default();
    let mut outbox = OutboundBox::default();

    let discard = TransientHandoff {
        transfer: this,
        step_id: TRANSIENT_DISCARD_STEP,
        fence: Fence(4),
    };
    on_transient_discard(discard, &mut owned, &mut applied, &mut stats);
    assert_eq!(
        stats.transients_discarded_source_crash, 0,
        "nothing to remove on an empty dest — the discard only poisons the adopt"
    );

    // The LATE batch replay: the adopt hits its `AlreadyApplied` arm (poisoned) — no insert.
    let cfg = config();
    adopt_transient_batch(
        this,
        cfg.realm,
        Fence(5),
        vec![TransientItem {
            entity,
            pose: transient_pose(),
            state: vec![],
        }],
        &cfg,
        &RealmRegions::default(),
        &PlacementLedger::default(),
        &mut owned,
        &mut applied,
        &mut stats,
        &mut outbox,
    );
    assert!(
        owned.0.is_empty(),
        "the poisoned adopt inserts nothing — no Arriving orphan"
    );
    assert_eq!(
        stats.transients_adopt_redelivered, 1,
        "the adopt short-circuited on the poisoned step"
    );
    assert_eq!(stats.transients_adopted, 0, "no item was ever adopted");
}

#[test]
fn transient_discard_flows_through_the_inbound_dispatch() {
    // Covers the `on_directory_reply` DISPATCH arm for `TransientDiscard` (the direct-call tests
    // above cover the handler itself): a DEST holding an `Arriving` copy receives the discard, drops
    // it as an accounted loss, and emits NOTHING (ack-FREE — the resolving saga is terminal).
    let mut rig = Rig::new();
    rig.grant_realm(); // authority.0 = Some(Fence(1))
    let debris = EntityId::pack(EntityKind::Debris, 1, 7, 0);
    let batch = TransferId(0xB7);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        debris,
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: TransientStatus::Arriving { batch },
            prev_offset: LatticePos::ORIGIN,
        },
    );
    let discard = TransientHandoff {
        transfer: batch,
        step_id: TRANSIENT_DISCARD_STEP,
        fence: Fence(1),
    };
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::TransientDiscard(discard),
    )]);
    assert!(
        !rig.world
            .resource::<OwnedTransients>()
            .0
            .contains_key(&debris),
        "the Arriving copy was discarded"
    );
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .transients_discarded_source_crash,
        1
    );
    assert_eq!(sent, vec![], "the discard is ack-free (terminal saga)");
}

#[test]
fn redrive_pending_adoptions_re_emits_one_ack_per_distinct_arriving_batch() {
    // CA-1 S3/S4 LIVENESS: `redrive_pending_adoptions` re-emits `BatchAdopted` every tick for each
    // DISTINCT `Arriving` batch (dedup — a batch of N items yields ONE ack, the MMO-scale discipline),
    // and SKIPS settled `Held` items. Seed the tiers DIRECTLY + tick with an empty inbox so the ONLY
    // egress is the re-drive (adopt itself is not exercised here).
    let mut rig = Rig::new();
    rig.grant_realm();
    let batch = TransferId(0xB7);
    let other = TransferId(0xB8); // 0xB7 < 0xB8 → deterministic BTreeSet emit order
    let arriving = |b| Transient {
        pose: transient_pose(),
        anchor_fence: Fence(1),
        status: TransientStatus::Arriving { batch: b },
        prev_offset: LatticePos::ORIGIN,
    };
    {
        let mut owned = rig.world.resource_mut::<OwnedTransients>();
        owned
            .0
            .insert(EntityId::pack(EntityKind::Debris, 1, 7, 0), arriving(batch));
        // Second item, SAME batch → still ONE ack for `batch` (distinct-batch dedup).
        owned
            .0
            .insert(EntityId::pack(EntityKind::Debris, 1, 7, 1), arriving(batch));
        owned
            .0
            .insert(EntityId::pack(EntityKind::Debris, 1, 7, 2), arriving(other));
        // A SETTLED resident (Held) is NOT re-driven — the `if let Arriving` false arm.
        owned.0.insert(
            EntityId::pack(EntityKind::Debris, 1, 7, 3),
            Transient {
                pose: transient_pose(),
                anchor_fence: Fence(1),
                status: TransientStatus::Held { outbound: None },
                prev_offset: LatticePos::ORIGIN,
            },
        );
    }
    let sent = rig.tick(vec![]);
    let ack = |t| {
        postcard::to_allocvec(&InterShardFlow::TransferAck(TransferAck::BatchAdopted {
            transfer_id: t,
            step_id: TRANSIENT_BATCH_STEP,
        }))
        .expect("encode")
    };
    // Exactly ONE ack per DISTINCT batch, in ascending BTreeSet order.
    assert_eq!(
        sent,
        vec![
            (ORCH, MsgClass::Saga, ack(batch)),
            (ORCH, MsgClass::Saga, ack(other)),
        ]
    );
    assert_eq!(rig.world.resource::<StubStats>().batch_adopts_redriven, 2);
}

#[test]
fn redrive_pending_adoptions_is_a_noop_when_nothing_is_arriving() {
    // The empty-`Arriving` path (the `for batch in batches` empty loop + `if let` all-false): a shard
    // holding only a settled `Held` transient re-drives nothing and emits no egress.
    let mut rig = Rig::new();
    rig.grant_realm();
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        EntityId::pack(EntityKind::Debris, 1, 7, 0),
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: TransientStatus::Held { outbound: None },
            prev_offset: LatticePos::ORIGIN,
        },
    );
    let sent = rig.tick(vec![]);
    assert_eq!(sent, vec![], "nothing Arriving → no re-drive");
    assert_eq!(rig.world.resource::<StubStats>().batch_adopts_redriven, 0);
}

#[test]
fn re_solicit_batch_is_a_counted_noop_at_the_source() {
    // CA-1 S3: the orchestrator's AwaitAdopt liveness PROBE arriving at a (live) SOURCE is a counted
    // no-op — no state change, no egress (the probe's signal is its SEND outcome at the orchestrator,
    // not this handler). No `Arriving` items, so the re-drive adds nothing to `sent`.
    let mut rig = Rig::new();
    rig.grant_realm();
    let probe = TransientHandoff {
        transfer: TransferId(0xB7),
        step_id: RE_SOLICIT_STEP,
        fence: Fence(1),
    };
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::ReSolicitBatch(probe),
    )]);
    assert_eq!(rig.world.resource::<StubStats>().re_solicits_received, 1);
    assert_eq!(
        sent,
        vec![],
        "the probe is a pure no-op — no reply, no state change"
    );
}

#[test]
fn transient_batch_and_drop_flow_through_the_inbound_dispatch() {
    // Covers the inbound DISPATCH into the transient handlers (the direct-call tests above cover
    // the handlers themselves): on_directory_reply → on_transfer_envelope's `TransientBatch` arm
    // (adopt + ack) AND on_directory_reply's `TransientDrop` arm (promote). The DEST adopts a
    // batch, acks BatchAdopted, then promotes Arriving→Held on the drop.
    let mut rig = Rig::new();
    rig.grant_realm(); // authority.0 = Some(Fence(1))
    let debris = EntityId::pack(EntityKind::Debris, 1, 7, 0);
    let batch = TransferId(0xB7);
    let env = TransferEnvelope {
        transfer_id: batch,
        universe_epoch: vd_core::EpochId(1),
        schema_version: TRANSFER_SCHEMA_VERSION,
        fence: Fence(1),
        step_id: TRANSIENT_BATCH_STEP,
        class: DurabilityClass::Transient,
        payload: TransitionPayload::TransientBatch {
            from_realm: RealmId::System(8),
            to_realm: config().realm,
            src_realm_fence: Fence(1),
            dst_realm_fence: Fence(1),
            source_tick: vd_core::TickId(1),
            items: vec![TransientItem {
                entity: debris,
                pose: transient_pose(),
                state: vec![],
            }],
        },
    };
    let sent = rig.tick(vec![wire_msg(
        NodeId(50),
        MsgClass::Saga,
        &InterShardFlow::Transfer(env),
    )]);
    // Adopted as the uncounted Arriving tier, anchored to the envelope's dst realm fence.
    assert_eq!(
        rig.world.resource::<OwnedTransients>().0[&debris].status,
        TransientStatus::Arriving { batch }
    );
    assert_eq!(
        rig.world.resource::<OwnedTransients>().0[&debris].anchor_fence,
        Fence(1)
    );
    // The egress is the BatchAdopted ack TWICE (exact-vec equality — no filter/any closure with an
    // uncoverable short-circuit arm, the HR5 test discipline): the adopt handler acks it once (in
    // `process_inbound`), then CA-1 S3/S4's `redrive_pending_adoptions` re-emits it the SAME tick (the
    // item is now `Arriving`) — the liveness re-drive. Both are byte-identical; the orchestrator absorbs
    // the duplicate (idempotent). Order is adopt-ack THEN re-drive (chain order).
    let expected_ack =
        postcard::to_allocvec(&InterShardFlow::TransferAck(TransferAck::BatchAdopted {
            transfer_id: batch,
            step_id: TRANSIENT_BATCH_STEP,
        }))
        .expect("encode");
    assert_eq!(
        sent,
        vec![
            (ORCH, MsgClass::Saga, expected_ack.clone()),
            (ORCH, MsgClass::Saga, expected_ack),
        ]
    );

    // The TransientDrop dispatch arm PROMOTES the Arriving item → Held + acks DropApplied(DROP).
    let promote = TransientHandoff {
        transfer: batch,
        step_id: TRANSIENT_DROP_STEP,
        fence: Fence(1),
    };
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::TransientDrop(promote),
    )]);
    assert_eq!(
        rig.world.resource::<OwnedTransients>().0[&debris].status,
        TransientStatus::Held { outbound: None }
    );
    assert_eq!(rig.world.resource::<StubStats>().transients_promoted, 1);
    let promote_ack =
        postcard::to_allocvec(&InterShardFlow::TransferAck(TransferAck::DropApplied {
            transfer_id: batch,
            step_id: TRANSIENT_DROP_STEP,
        }))
        .expect("encode");
    assert_eq!(sent, vec![(ORCH, MsgClass::Saga, promote_ack)]);

    // The TransientRelease + ReleaseComplete dispatch arms (SOURCE role): seed a Held source item
    // for a 2nd batch, release it → Departing + DropApplied(RELEASE) ack, then complete → retired.
    let src_batch = TransferId(0xB8);
    let src_item = EntityId::pack(EntityKind::Debris, 1, 7, 9);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        src_item,
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: TransientStatus::Held {
                outbound: Some(src_batch),
            },
            prev_offset: LatticePos::ORIGIN,
        },
    );
    let rel = TransientHandoff {
        transfer: src_batch,
        step_id: TRANSIENT_RELEASE_STEP,
        fence: Fence(1),
    };
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::TransientRelease(rel),
    )]);
    assert_eq!(
        rig.world.resource::<OwnedTransients>().0[&src_item].status,
        TransientStatus::Departing { batch: src_batch }
    );
    let release_ack =
        postcard::to_allocvec(&InterShardFlow::TransferAck(TransferAck::DropApplied {
            transfer_id: src_batch,
            step_id: TRANSIENT_RELEASE_STEP,
        }))
        .expect("encode");
    assert_eq!(sent, vec![(ORCH, MsgClass::Saga, release_ack)]);

    let rc = TransientHandoff {
        transfer: src_batch,
        step_id: TRANSIENT_RELEASE_STEP,
        fence: Fence(1),
    };
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::ReleaseComplete(rc),
    )]);
    assert!(
        !rig.world
            .resource::<OwnedTransients>()
            .0
            .contains_key(&src_item),
        "ReleaseComplete retired the Departing copy"
    );
    let complete_ack =
        postcard::to_allocvec(&InterShardFlow::TransferAck(TransferAck::DropApplied {
            transfer_id: src_batch,
            step_id: TRANSIENT_COMPLETE_STEP,
        }))
        .expect("encode");
    assert_eq!(
        sent,
        vec![(ORCH, MsgClass::Saga, complete_ack)],
        "D-7d: ReleaseComplete acks the retire-complete (SourceRetired drives the saga tail to Done)"
    );

    // The D-7d TransientAbandon dispatch arm (SOURCE role): seed a fresh Departing item for a 3rd
    // batch (the dead-dest case), abandon it → dropped + bucketed as an accounted loss, NO ack (the
    // resolving saga is already terminal).
    let abandon_batch = TransferId(0xB9);
    let abandon_item = EntityId::pack(EntityKind::Debris, 1, 8, 9);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        abandon_item,
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: TransientStatus::Departing {
                batch: abandon_batch,
            },
            prev_offset: LatticePos::ORIGIN,
        },
    );
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::TransientAbandon(TransientHandoff {
            transfer: abandon_batch,
            step_id: TRANSIENT_ABANDON_STEP,
            fence: Fence(1),
        }),
    )]);
    assert!(
        !rig.world
            .resource::<OwnedTransients>()
            .0
            .contains_key(&abandon_item),
        "TransientAbandon dropped the retained Departing copy"
    );
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .transients_lost_in_handover
            .get(&EntityKind::Debris),
        Some(&1),
        "the abandoned Debris is bucketed as an accounted loss"
    );
    assert!(sent.is_empty(), "TransientAbandon is terminal — no ack");
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
fn a_granted_shard_renews_its_realm_and_granted_entities_on_cadence() {
    // D-3 heartbeat: on the renew cadence a granted shard re-sends LeaseRenew for its Realm AND
    // every GRANTED, NON-DEPARTING Entity — never a non-granted (still-granting) or departing
    // (logging-out) dot. INERT off-cadence + when the interval is 0.
    let mut rig = Rig::with_config(StubConfig {
        lease_renew_interval_ticks: 4,
        ..config()
    });
    rig.grant_realm();
    let mk = |entity: EntityId, granted: bool, departing: bool| Dot {
        entity,
        account: AccountId(1),
        session_fence: Fence(1),
        gateway: GATEWAY,
        granted,
        input_active: false,
        adopting: false,
        authority: Authority::Owned { fence: Fence(1) },
        departing,
        entity_fence: Fence(1),
        pose: StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(0)),
        yaw: 0.0,
        pitch: 0.0,
        last_applied_seq: None,
        prev_offset: LatticePos::ORIGIN,
    };
    let granted_e = EntityId::pack(EntityKind::Player, 10, 1, 1);
    let provisional_e = EntityId::pack(EntityKind::Player, 10, 2, 2);
    let departing_e = EntityId::pack(EntityKind::Player, 10, 3, 3);
    {
        let mut dots = rig.world.resource_mut::<Dots>();
        dots.0.insert(SessionId(1), mk(granted_e, true, false));
        dots.0.insert(SessionId(2), mk(provisional_e, false, false)); // emits LeaseGrant
        dots.0.insert(SessionId(3), mk(departing_e, true, true)); // emits LeaseRevoke
    }
    let renew_keys = |sent: &[(NodeId, MsgClass, Vec<u8>)]| -> Vec<DirectoryKey> {
        sent.iter()
            .filter(|(to, _, _)| *to == ORCH)
            .filter_map(
                |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                    Ok(InterShardFlow::Directory(DirectoryOp::LeaseRenew { key, fence })) => {
                        assert_eq!(fence, Fence(1), "renews at the held fence");
                        Some(key)
                    }
                    _ => None,
                },
            )
            .collect()
    };
    // A multiple tick renews the Realm + the granted, non-departing entity ONLY.
    rig.set_local_tick(4);
    let on = renew_keys(&rig.tick(vec![]));
    assert!(
        on.contains(&DirectoryKey::Realm(config().realm)),
        "the realm lease is renewed"
    );
    assert!(
        on.contains(&DirectoryKey::Entity(granted_e)),
        "a granted, non-departing entity lease is renewed"
    );
    assert!(
        !on.contains(&DirectoryKey::Entity(provisional_e)),
        "a non-granted (still-granting) entity is NOT renewed"
    );
    assert!(
        !on.contains(&DirectoryKey::Entity(departing_e)),
        "a departing (logging-out) entity is NOT renewed"
    );
    // Off-cadence: no heartbeat at all (the interval gate's modulo branch). The inert branch
    // (interval == 0) is covered by every other granted-shard test (all run config() at interval 0).
    rig.set_local_tick(5);
    assert!(
        renew_keys(&rig.tick(vec![])).is_empty(),
        "an off-cadence tick emits no LeaseRenew"
    );
}

#[test]
fn a_cohosting_shard_renews_its_child_realm_lease_on_cadence() {
    // D-3 heartbeat, co-hosting (task #149): a MULTI-realm shard renews its PRIMARY realm AND every
    // CO-HOSTED CHILD realm it holds (the `cohosted.0` chain in the renewal set). Drives the co-host
    // renewal closure that a single-realm shard never reaches (`cohosted.0` empty). Boot System 7
    // (primary) + Planet 7 (co-hosted child), then a cadence tick renews BOTH realm keys.
    let mut rig = Rig::with_config(StubConfig {
        lease_renew_interval_ticks: 4,
        ..cohost_planet_config()
    });
    boot_cohost_planet(&mut rig); // primary System 7 on RealmAuthority + child Planet 7 on CoHostedAuthority
    assert_eq!(
        rig.world
            .resource::<CoHostedAuthority>()
            .0
            .get(&RealmId::Planet(7)),
        Some(&Fence(1)),
        "precondition: the co-hosted Planet-7 head is held (so the renewal chain has a child to emit)",
    );
    // A still-granting (provisional) dot so the cadence tick ALSO emits a `LeaseGrant` to ORCH — a
    // NON-`LeaseRenew` op that exercises the extractor's fall-through arm (no uncoverable `_ => None`).
    rig.world.resource_mut::<Dots>().0.insert(
        SessionId(1),
        Dot {
            entity: EntityId::pack(EntityKind::Player, 7, 1, 1),
            account: AccountId(1),
            session_fence: Fence(1),
            gateway: GATEWAY,
            granted: false, // provisional ⇒ emits LeaseGrant, NOT LeaseRenew
            input_active: false,
            adopting: false,
            authority: Authority::Owned { fence: Fence(1) },
            departing: false,
            entity_fence: Fence(1),
            pose: StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(0)),
            yaw: 0.0,
            pitch: 0.0,
            last_applied_seq: None,
            prev_offset: LatticePos::ORIGIN,
        },
    );
    let renew_keys = |sent: &[(NodeId, MsgClass, Vec<u8>)]| -> Vec<DirectoryKey> {
        sent.iter()
            .filter(|(to, _, _)| *to == ORCH)
            .filter_map(
                |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                    Ok(InterShardFlow::Directory(DirectoryOp::LeaseRenew { key, .. })) => Some(key),
                    _ => None,
                },
            )
            .collect()
    };
    rig.set_local_tick(4); // on the renew cadence
    let on = renew_keys(&rig.tick(vec![]));
    assert!(
        on.contains(&DirectoryKey::Realm(RealmId::System(7))),
        "the PRIMARY realm lease is renewed: {on:?}",
    );
    assert!(
        on.contains(&DirectoryKey::Realm(RealmId::Planet(7))),
        "the CO-HOSTED CHILD realm lease is ALSO renewed (the co-host chain): {on:?}",
    );
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

/// A frame no shard in these rigs is: stands in for "measured somewhere else". It used to be the
/// UNIVERSE-ROOT frame every stored spawn pose was written in — which, after the fold's removal, is a
/// frame nobody may measure in at all, which is exactly why a pose wearing it must be refused.
fn foreign_frame() -> FrameRef {
    FrameRef::SystemSpace { system_seed: 0 }
}

#[test]
fn a_spawn_pose_offered_in_this_realms_frame_is_taken_verbatim() {
    // The gateway did the conversion — it holds the whole forest and stepped down it subtracting each
    // realm's placement — so the pose arrives already measured from THIS realm's centre. The shard's
    // job is to check the frame and use the number, doing no arithmetic of its own. It cannot do any:
    // it does not know where it itself sits.
    let cfg = config();
    let mut stats = StubStats::default();
    let offered = StampedPose::at_rest(cfg.frame, DVec3::new(3.0, 0.0, 0.0), UniverseTick(0));
    let got = resolve_spawn_pose(
        &cfg,
        AccountId(0x5F3B),
        Some(offered),
        UniverseTick(77),
        &mut stats,
    );
    assert_eq!(got.frame, cfg.frame);
    assert_eq!(fm(got.pos), DVec3::new(3.0, 0.0, 0.0));
    // Re-stamped to the current clock tick; at rest (the offered pose was at rest).
    assert_eq!(got.universe_tick, UniverseTick(77));
    assert_eq!(got.vel, DVec3::ZERO);
    assert_eq!(stats.spawn_poses_refused, 0);
}

#[test]
fn a_stored_spawn_pose_in_the_wrong_frame_is_refused_not_relabelled() {
    // THE FOOT-GUN THIS CLOSES. A pose measured in a frame this realm is not used to be handed to
    // `rebind_pose_to_dest` with an identity context, which renamed the frame and moved no number — so
    // a player stored 3 m above a planet 145 m from its star was planted 3 m from the STAR. There is
    // now no conversion to attempt (only the party holding the whole forest can convert), so the pose
    // is REFUSED, counted, and the avatar births at the origin.
    let mut cfg = config();
    let account = AccountId(0x5F3B);
    let wrong = StampedPose::at_rest(
        foreign_frame(),
        DVec3::new(100.0, 200.0, 300.0),
        UniverseTick(0),
    );
    cfg.spawn_poses.insert(account, wrong);
    let mut stats = StubStats::default();
    let got = resolve_spawn_pose(&cfg, account, None, UniverseTick(77), &mut stats);
    assert_eq!(
        got,
        StampedPose::at_rest(cfg.frame, DVec3::ZERO, UniverseTick(77)),
        "a pose in a frame this realm is not must never be worn — the origin is the honest fallback"
    );
    assert_eq!(stats.spawn_poses_refused, 1);
    // The OFFERED half of the same rule: a wire pose in the wrong frame is refused identically, so a
    // mis-routed login cannot plant an avatar by coming in over the wire instead of the store.
    let mut stats = StubStats::default();
    let got = resolve_spawn_pose(
        &cfg,
        AccountId(0xAB),
        Some(wrong),
        UniverseTick(77),
        &mut stats,
    );
    assert_eq!(
        got,
        StampedPose::at_rest(cfg.frame, DVec3::ZERO, UniverseTick(77))
    );
    assert_eq!(stats.spawn_poses_refused, 1);
}

#[test]
fn the_offered_pose_wins_over_this_realms_own_store() {
    // Order matters and is stated: the gateway's answer is about THIS login, the store is about this
    // account's last known place. A login that carries a position uses it.
    let mut cfg = config();
    let account = AccountId(7);
    cfg.spawn_poses.insert(
        account,
        StampedPose::at_rest(cfg.frame, DVec3::new(9.0, 9.0, 9.0), UniverseTick(0)),
    );
    let mut stats = StubStats::default();
    let offered = StampedPose::at_rest(cfg.frame, DVec3::new(1.0, 0.0, 0.0), UniverseTick(0));
    let got = resolve_spawn_pose(&cfg, account, Some(offered), UniverseTick(5), &mut stats);
    assert_eq!(fm(got.pos), DVec3::new(1.0, 0.0, 0.0));
    assert_eq!(stats.spawn_poses_refused, 0);
}

#[test]
fn resolve_spawn_pose_without_an_entry_is_origin_at_rest_byte_identical() {
    // The `None`/`None` arm: no offered pose and nothing stored ⇒ origin-at-rest in this shard's frame
    // — BYTE-IDENTICAL to the admit literal every rig has always taken.
    let cfg = config(); // empty spawn_poses
    let mut stats = StubStats::default();
    let got = resolve_spawn_pose(&cfg, AccountId(0xAB), None, UniverseTick(42), &mut stats);
    assert_eq!(
        got,
        StampedPose::at_rest(cfg.frame, DVec3::ZERO, UniverseTick(42))
    );
    assert_eq!(stats.spawn_poses_refused, 0);
}

#[test]
fn login_admits_the_dot_at_its_stored_spawn_pose_through_the_one_admit_path() {
    // The admit-path proof (the wiring, not just the helper): a real AttachSession for an account WITH
    // a realm-local stored pose births its dot there — same Ghost-birth admit path, only the pose value
    // differs. `attach_request` attaches AccountId(5), so key the stand-in on it.
    let stored_pos = DVec3::new(11.0, -22.0, 33.0);
    let cfg = config();
    let stored = StampedPose::at_rest(cfg.frame, stored_pos, UniverseTick(0));
    let mut rig = Rig::with_config(StubConfig {
        spawn_poses: BTreeMap::from([(AccountId(5), stored)]),
        ..config()
    });
    rig.grant_realm();
    let _ = rig.attach_request(SESSION, GATEWAY);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    // Born at the STORED pose, re-stamped to the rig clock (universe_tick 100) — NOT origin-at-rest.
    assert_eq!(dot.pose.frame, FrameRef::SystemSpace { system_seed: 7 });
    assert_eq!(fm(dot.pose.pos), stored_pos);
    assert_eq!(dot.pose.universe_tick, UniverseTick(100));
    // `prev_offset` is seeded from the SAME stored offset (not the old hardcoded ZERO).
    assert_eq!(
        dot.prev_offset
            .delta_m(LatticePos::ORIGIN, vd_core::pose::Tier::Fine),
        stored_pos
    );
}

#[test]
fn a_login_is_admitted_at_the_pose_the_gateway_measured_for_it() {
    // THE WIRING, end to end through the real attach message: the gateway put a pose measured from
    // THIS realm's centre in `AttachSession`, and the avatar is born there. This is the path that used
    // to run through a cluster-wide map of universe-absolute positions that the shard relabelled.
    let cfg = config();
    let offered = StampedPose::at_rest(cfg.frame, DVec3::new(3.0, 0.0, 0.0), UniverseTick(0));
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach_request_with(SESSION, GATEWAY, Some(offered));
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(fm(dot.pose.pos), DVec3::new(3.0, 0.0, 0.0));
    assert_eq!(dot.pose.frame, cfg.frame);
    assert_eq!(rig.world.resource::<StubStats>().spawn_poses_refused, 0);
}

#[test]
fn a_login_routed_to_the_wrong_realm_is_admitted_at_the_origin_and_counted() {
    // The loud arm of the same wiring: a static cluster attaches every login to one fixed shard, which
    // in general is NOT the realm the account's position was measured in. The shard cannot convert
    // (only the party holding the whole forest can), so it says so and births at its own origin —
    // rather than wearing a number that means something else here.
    let elsewhere =
        StampedPose::at_rest(foreign_frame(), DVec3::new(3.0, 0.0, 0.0), UniverseTick(0));
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach_request_with(SESSION, GATEWAY, Some(elsewhere));
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(fm(dot.pose.pos), DVec3::ZERO);
    assert_eq!(dot.pose.frame, config().frame);
    assert_eq!(rig.world.resource::<StubStats>().spawn_poses_refused, 1);
}

#[test]
fn login_without_a_stored_pose_births_at_the_origin_unchanged() {
    // The admit-path `None` proof: an account with no stand-in entry births origin-at-rest — the
    // byte-identical pre-5f-3b behaviour through the real attach path.
    let mut rig = Rig::new(); // config() ⇒ empty spawn_poses
    rig.grant_realm();
    let _ = rig.attach_request(SESSION, GATEWAY);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(
        dot.pose,
        StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(100))
    );
    assert_eq!(dot.prev_offset, LatticePos::ORIGIN);
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
        spawn: None,
    };
    let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &msg)]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].session_fence,
        Fence(3)
    );
}

/// A position's TOTAL displacement from its frame origin, in metres — whole-number part and leftover
/// combined. The integrator now folds the leftover into the whole number every tick, so reading the
/// leftover alone (which these assertions used to do) reports a sub-millimetre remainder rather than
/// where the subject actually is.
fn total_m(p: &vd_core::pose::LatticePos) -> DVec3 {
    p.delta_m(vd_core::pose::LatticePos::ORIGIN, vd_core::pose::Tier::Fine)
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
    assert!(
        (total_m(&dot.pose.pos).z + expected_step).abs() < 1e-12,
        "moved -Z"
    );
    assert_eq!(total_m(&dot.pose.pos).x, 0.0);
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
    assert!(
        (total_m(&dot.pose.pos).x + expected_step).abs() < 1e-6,
        "moved -X"
    );
    assert!(total_m(&dot.pose.pos).z.abs() < 1e-6);
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
    // Strafe right + up, no forward; the per-axis clamp catches the out-of-range axis, and
    // THE SPEED LAW (S3) normalizes the over-unit diagonal to the realm's ceiling: the
    // pre-law arithmetic commanded √2·move_speed on this stick — ABOVE the one ceiling every
    // realm now states, which the law no longer permits (`vd_core::flight::
    // throttle_axes_scale`'s over-unit arm — the one measured behaviour change at human
    // scale, and it is a cure). Per axis: (move_speed·dt)/√2; the TOTAL step is exactly the
    // ceiling's move_speed·dt.
    let _ = rig.tick(vec![input_msg(1, Fence(1), [0.0, 2.0, 1.0], [0.0, 0.0])]);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    let expected_step = 2.0 * 0.05 / 2.0_f64.sqrt();
    assert!(
        (total_m(&dot.pose.pos).x - expected_step).abs() < 1e-12,
        "clamped strafe, ceiling-normalized"
    );
    assert!(
        (total_m(&dot.pose.pos).y - expected_step).abs() < 1e-12,
        "vertical, ceiling-normalized"
    );
    assert!(
        (total_m(&dot.pose.pos).length() - 2.0 * 0.05).abs() < 1e-12,
        "the diagonal's TOTAL step is the ceiling exactly — never √2× it"
    );
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

/// ★THE THROWAWAY OVERDRIVE INSTRUMENT, both arms. `set_cruise_overdrive` is the only way the
/// `VD_TEST_OVERDRIVE` knob reaches the sim, and its whole contract is one clamp: the instrument may
/// make a cruise FASTER and may never make it slower, so anything under the lawful `1.0` is refused
/// back to it and the caller is TOLD what was planted (the returned value, not the argument).
///
/// Covered here because it is reachable production code with no other caller inside this crate — the
/// shard binary is its only consumer, and a binary is outside the coverage domain. Without this the
/// clamp could invert and every gate would still be green while a test instrument silently GOVERNED
/// a flight it was only ever allowed to speed up.
#[test]
fn the_cruise_overdrive_instrument_may_only_ever_go_faster() {
    let mut regions = clamped_forest();
    // The default IS the law — an unset knob is inert.
    assert_eq!(regions.cruise_overdrive, 1.0);
    // The lawful arm: a factor above one is planted verbatim and reported back.
    assert_eq!(regions.set_cruise_overdrive(4.0), 4.0);
    assert_eq!(regions.cruise_overdrive, 4.0);
    // The refused arm: below the law it is clamped back to the law, not to the argument.
    assert_eq!(regions.set_cruise_overdrive(0.25), 1.0);
    assert_eq!(regions.cruise_overdrive, 1.0);
    // And the boundary itself is lawful rather than refused.
    assert_eq!(regions.set_cruise_overdrive(1.0), 1.0);
}

#[test]
fn the_governed_ceiling_is_the_realm_cap_lowered_by_the_child_arm() {
    // A wide own realm (cap 2·150 000/180 ≈ 1 667 m/s at foot 2) with one small child at the
    // origin: the governor is the own cap far away, the child arm as the subject nears, and
    // exactly the child's own cap AT the bound.
    let t = flight_tuning(&config());
    let own = region(OWN_REALM, Some(ROOT_REALM), DVec3::ZERO, 150_000.0);
    let child = region(OTHER_REALM, Some(OWN_REALM), DVec3::ZERO, 100.0);
    let regions = RealmRegions::new(vec![root_region(), own, child]);
    let book = regions.author_book(OWN_REALM, 20.0, UniverseTick(10));
    let at = |x: f64| LatticePos::from_metres(DVec3::new(x, 0.0, 0.0), Tier::Fine);
    let own_cap = flight::realm_speed_cap_mps(150_000.0, 2.0, t.traverse_s);
    let child_cap = flight::realm_speed_cap_mps(100.0, 2.0, t.traverse_s);
    assert_eq!(child_cap, 2.0, "a 100 m child clamps to the foot");
    // Far from the child (149 000 m out): the child arm is way above the own cap — the own
    // cap binds.
    assert_eq!(
        governed_ceiling_in_book(&regions, OWN_REALM, &book, at(149_000.0), &t),
        Some(own_cap),
    );
    // Nearing the child, the arm binds: child_cap + (dist − extent)/τ.
    assert_eq!(
        governed_ceiling_in_book(&regions, OWN_REALM, &book, at(600.0), &t),
        Some(flight::approach_ceiling_mps(
            child_cap,
            600.0 - 100.0,
            t.tau_s
        )),
    );
    // AT (and inside) the child's bound: exactly the child's own ceiling — you arrive at ITS
    // speed, never through it.
    assert_eq!(
        governed_ceiling_in_book(&regions, OWN_REALM, &book, at(100.0), &t),
        Some(child_cap),
    );
    assert_eq!(
        governed_ceiling_in_book(&regions, OWN_REALM, &book, at(50.0), &t),
        Some(child_cap),
    );
    // A childless forest: the own cap alone (the loop's empty arm).
    let bare = RealmRegions::new(vec![root_region(), own]);
    let bare_book = bare.author_book(OWN_REALM, 20.0, UniverseTick(10));
    assert_eq!(
        governed_ceiling_in_book(&bare, OWN_REALM, &bare_book, at(0.0), &t),
        Some(own_cap),
    );
    // A realm absent from the forest: None — no region, no law.
    assert_eq!(
        governed_ceiling_in_book(&bare, OTHER_REALM, &bare_book, at(0.0), &t),
        None,
    );
}

#[test]
fn the_governed_ceiling_resolve_answers_none_off_the_law() {
    // The outer resolve's three no-law arms: an empty forest (unhosted frame), a hosted frame
    // with NO authored book yet (the pre-sync ledger), and the full Some path once the one
    // writer has run.
    let t = flight_tuning(&config());
    let pos = LatticePos::ORIGIN;
    let mut rig = Rig::new();
    // Empty forest (the default resource): the frame resolves to no realm.
    {
        let regions = rig.world.resource::<RealmRegions>();
        let placements = rig.world.resource::<Placements>();
        assert_eq!(
            governed_ceiling_for_frame(regions, &placements.0, config().frame, pos, &t),
            None,
        );
    }
    // Planted forest, but the writer has not run (no tick yet): still None — never outrun
    // your feet before the realm has authored its world.
    rig.world.insert_resource(clamped_forest());
    {
        let regions = rig.world.resource::<RealmRegions>();
        let placements = rig.world.resource::<Placements>();
        assert_eq!(
            governed_ceiling_for_frame(regions, &placements.0, config().frame, pos, &t),
            None,
        );
    }
    // One synced tick authors the book: the ceiling exists, and at this clamped scale it IS
    // the foot speed exactly.
    let _ = rig.tick(vec![]);
    let regions = rig.world.resource::<RealmRegions>();
    let placements = rig.world.resource::<Placements>();
    assert_eq!(
        governed_ceiling_for_frame(regions, &placements.0, config().frame, pos, &t),
        Some(2.0),
    );
}

#[test]
fn the_speed_law_is_bit_inert_wherever_the_ceiling_clamps() {
    // THE S3 INERTNESS MEASUREMENT, unit tier: the SAME input flight on a forestless rig (the
    // pre-law posture) and on a rig with a planted CLAMPED forest (every ceiling at the foot)
    // lands the dot at BIT-IDENTICAL poses — position, cell, velocity. This is the measured
    // form of "every sub-45 km realm runs at exactly today's speeds"; the process battery is
    // its world-scale twin.
    let fly = |plant: bool| {
        let mut rig = Rig::new();
        if plant {
            rig.world.insert_resource(clamped_forest());
        }
        rig.grant_realm();
        let _ = rig.attach();
        // A fractional stick, a full stick, a diagonal and a look-turn — the input shapes the
        // landed battery flies.
        let _ = rig.tick(vec![input_msg(1, Fence(1), [0.6, 0.0, 0.0], [0.3, 0.1])]);
        let _ = rig.tick(vec![input_msg(2, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
        let _ = rig.tick(vec![input_msg(3, Fence(1), [0.0, 0.5, 0.5], [0.0, 0.0])]);
        rig.world.resource::<Dots>().0[&SESSION].pose
    };
    let (bare, planted) = (fly(false), fly(true));
    assert_eq!(bare.pos.cell(), planted.pos.cell());
    assert_eq!(bare.pos.offset(), planted.pos.offset());
    assert_eq!(bare.vel, planted.vel);
    assert_eq!(bare.orient, planted.orient);
}

#[test]
fn full_throttle_rides_the_proportional_ramp_up_to_the_realm_ceiling() {
    // A wide own realm (cap 2·1.8e6/180 = 20 000 m/s at foot 2): holding full throttle from
    // rest compounds the speed by e^(dt/τ) per tick from the foot floor — the §4.2(c) ramp —
    // and parks AT the ceiling, never above it.
    let mut rig = Rig::new();
    rig.world.insert_resource(RealmRegions::new(vec![
        root_region(),
        region(OWN_REALM, Some(ROOT_REALM), DVec3::ZERO, 1.8e6),
    ]));
    rig.grant_realm();
    let _ = rig.attach();
    let t = flight_tuning(&config());
    let g = (t.tick_dt_s / t.tau_s).exp();
    let cap = flight::realm_speed_cap_mps(1.8e6, 2.0, t.traverse_s);
    assert_eq!(cap, 20_000.0);
    let mut expected = 2.0; // the foot floor the ramp compounds from
    for seq in 1..=40u64 {
        let _ = rig.tick(vec![input_msg(seq, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
        expected = (expected * g).min(cap);
        let v = rig.world.resource::<Dots>().0[&SESSION].pose.vel.length();
        assert!(
            (v - expected).abs() <= expected * 1e-9,
            "tick {seq}: v {v} vs ramp {expected}",
        );
    }
    // Releasing the stick stops DEAD — deceleration is instant (P3 "stopped means stopped");
    // the gradual arrival slow-down is the governor's falling ceiling, never a coast.
    let _ = rig.tick(vec![input_msg(41, Fence(1), [0.0, 0.0, 0.0], [0.0, 0.0])]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].pose.vel,
        DVec3::ZERO,
    );
}

#[test]
fn a_transient_faster_than_the_governed_ceiling_is_clamped_on_the_crossing_path() {
    // ★OQ-2 (owner-ruled): the realm's ceiling governs everything it contains, piloted or
    // not. On a clamped-scale forest (ceiling = foot = 2 m/s) a 10 m/s debris is cut to the
    // ceiling before its advance; a slower one is untouched BIT-FOR-BIT. (The forestless
    // no-clamp arm is `readvance_advances_held_transients_and_skips_the_uncounted_tiers`.)
    let mut rig = Rig::new();
    rig.world.insert_resource(clamped_forest());
    rig.grant_realm();
    let fast = EntityId::pack(EntityKind::Debris, 1, 7, 1);
    let slow = EntityId::pack(EntityKind::Debris, 1, 7, 2);
    let pose_with = |vel: DVec3| StampedPose {
        vel,
        ..StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(99))
    };
    {
        let mut owned = rig.world.resource_mut::<OwnedTransients>();
        for (id, vel) in [
            (fast, DVec3::new(10.0, 0.0, 0.0)),
            (slow, DVec3::new(0.0, 1.5, 0.0)),
        ] {
            owned.0.insert(
                id,
                Transient {
                    pose: pose_with(vel),
                    anchor_fence: Fence(1),
                    status: TransientStatus::Held { outbound: None },
                    prev_offset: LatticePos::ORIGIN,
                },
            );
        }
    }
    let _ = rig.tick(vec![]);
    let owned = rig.world.resource::<OwnedTransients>();
    assert_eq!(
        owned.0[&fast].pose.vel,
        DVec3::new(2.0, 0.0, 0.0),
        "the 10 m/s debris is governed down to the realm ceiling (OQ-2: no subject kind is \
         exempt)",
    );
    assert_eq!(
        owned.0[&fast].pose.universe_tick,
        UniverseTick(100),
        "clamped AND advanced — the governor never freezes a subject",
    );
    assert_eq!(
        owned.0[&slow].pose.vel,
        DVec3::new(0.0, 1.5, 0.0),
        "a sub-ceiling transient is untouched bit-for-bit",
    );
}

#[test]
fn occupant_movement_dilates_with_the_realm_time_multiplier() {
    // A realm's SUBJECTIVE time factor scales OCCUPANT movement: a slow-time realm (0.5) advances the
    // dot HALF as far per tick; the default (1.0) is byte-identical to the un-scaled `speed·dt` step.
    let step_len_at = |mult: f64| {
        let mut rig = Rig::with_config(StubConfig {
            time_multiplier: mult,
            ..config()
        });
        rig.grant_realm();
        let _ = rig.attach();
        // Pure forward input (movement = [forward, strafe, up]).
        let _ = rig.tick(vec![input_msg(1, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
        total_m(&rig.world.resource::<Dots>().0[&SESSION].pose.pos).length()
    };
    // Default 1.0 = the un-multiplied step (`move_speed 2.0 · dt 0.05` = 0.1) — byte-identical.
    assert!((step_len_at(1.0) - 2.0 * 0.05).abs() < 1e-12);
    // The 0.5-multiplier realm moved EXACTLY half as far (time dilation).
    assert!((step_len_at(0.5) - step_len_at(1.0) * 0.5).abs() < 1e-12);
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
    assert_eq!(
        total_m(&dot.pose.pos),
        vd_core::glam::DVec3::ZERO,
        "pose untouched"
    );
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
    assert!(
        total_m(&dot.pose.pos).is_finite(),
        "authoritative pose finite"
    );
    assert!(
        total_m(&dot.pose.pos).z < 0.0,
        "the finite input integrated (moved -Z)"
    );
}

#[test]
fn action_bits_are_inert_two_datagrams_differing_only_in_action_bits_integrate_identically() {
    // D-41 / D-39.1 MECHANICAL-GUARD TRIPWIRE (exists-to-be-flipped). `action_bits` is INERT in the
    // current integrator — `integrate` reads ONLY `look` + `movement`, never `action_bits` — so two
    // inputs identical EXCEPT for `action_bits` MUST integrate to the identical authoritative pose
    // today. This flips RED the day the reliable client→shard discrete-action arm makes the sim
    // consume `action_bits` (its named consumers: P6 block-edit-forward + P11 PvP fire-registration,
    // DEFERRED D-39.1) — a hard guard that world-mutating actions (esp. PvP fire-reg) can NEVER be
    // silently gated onto the lossy UNRELIABLE input datagram that `action_bits` rides.
    let mut inert = Rig::new();
    inert.grant_realm();
    let _ = inert.attach();
    let mut set = Rig::new();
    set.grant_realm();
    let _ = set.attach();

    // A non-trivial input (movement + look both non-zero) so the pose actually MOVES — proving the
    // two agree on a REAL integration, not on a shared do-nothing origin.
    let movement = [1.0f32, 0.5, -0.25];
    let look = [0.3f32, 0.1];
    let _ = inert.tick(vec![input_msg_bits(1, Fence(1), movement, look, 0)]);
    let _ = set.tick(vec![input_msg_bits(1, Fence(1), movement, look, u32::MAX)]);

    let dot_inert = inert.world.resource::<Dots>().0[&SESSION];
    let dot_set = set.world.resource::<Dots>().0[&SESSION];
    // Dot is Copy + PartialEq (pose + yaw + pitch + vel): ONE equality assert is the strongest,
    // HR5-coverage-safe identical-pose check (no `matches!` false-arm, no `&&` short-circuit).
    assert_eq!(
        dot_inert, dot_set,
        "action_bits is inert: 0 vs u32::MAX must not change the integrated pose"
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
fn window_control_registers_refreshes_and_closes_the_registry_idempotently() {
    // THE WINDOW LANE, Slice A (rebased from the Slice-0 fail-closed drop — the registry this
    // control lane was owed now EXISTS): an open REGISTERS under (opener, id); a duplicate
    // open of the same scope is the keep-alive (refreshes the TTL, counted apart); a re-used
    // id under a NEW scope replaces the window whole; a close removes it; a close of an
    // unknown id is a COUNTED no-op (the polite fast path racing the TTL backstop). Never
    // `undecodable`, never a reply, never a dot.
    let mut rig = Rig::new();
    rig.grant_realm();
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    let sent = rig.tick(vec![
        wire_msg(GATEWAY, MsgClass::Control, &open),
        wire_msg(GATEWAY, MsgClass::Control, &open), // the keep-alive re-assert
    ]);
    // A registered Occupants window on a region-less rig: the emitter runs (empty roster ⇒
    // an empty-rows WindowFrame each tick — the leaf's per-tick stamp) — nothing else.
    let control_replies = sent
        .iter()
        .filter(|(to, class, _)| (*to == GATEWAY) & (*class == MsgClass::Control))
        .count();
    assert_eq!(control_replies, 0, "no Control reply from a window open");
    assert_eq!(rig.world.resource::<Dots>().0.len(), 0, "no dot minted");
    {
        let stats = rig.world.resource::<StubStats>();
        assert_eq!(stats.windows_opened, 1, "one window registered");
        assert_eq!(stats.window_reasserted, 1, "the duplicate refreshed it");
        assert_eq!(stats.windows_open, 1, "the gauge reads the registry");
        assert_eq!(stats.undecodable, 0, "window control is NOT garbage");
    }
    let held = rig.world.resource::<OpenWindows>();
    assert_eq!(held.0.len(), 1);
    assert_eq!(
        held.0.get(&(GATEWAY, WindowId(1))).map(|w| w.scope),
        Some(WindowScope::Occupants)
    );
    // A re-used id under a NEW scope replaces the window (fresh baselines, counted as opened).
    let rescope = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Child(RealmId::Planet(9)),
    };
    let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &rescope)]);
    assert_eq!(rig.world.resource::<StubStats>().windows_opened, 2);
    assert_eq!(
        rig.world
            .resource::<OpenWindows>()
            .0
            .get(&(GATEWAY, WindowId(1)))
            .map(|w| w.scope),
        Some(WindowScope::Child(RealmId::Planet(9)))
    );
    // Close removes it; a second close of the now-unknown id is a counted no-op.
    let close = GatewayToShard::WindowClose {
        window: WindowId(1),
    };
    let _ = rig.tick(vec![
        wire_msg(GATEWAY, MsgClass::Control, &close),
        wire_msg(GATEWAY, MsgClass::Control, &close),
    ]);
    assert_eq!(rig.world.resource::<OpenWindows>().0.len(), 0);
    {
        let stats = rig.world.resource::<StubStats>();
        assert_eq!(stats.window_close_unknown, 1, "the second close counted");
        assert_eq!(stats.windows_open, 0, "zero subscribers ⇒ zero windows");
    }
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
                    authority: Authority::Owned { fence: Fence(1) },
                    departing: false,
                    entity_fence: Fence(1),
                    pose: StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(100)),
                    yaw: 0.0,
                    pitch: 0.0,
                    last_applied_seq: None,
                    prev_offset: LatticePos::ORIGIN,
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
    // it is `adopting`, NOT granted (the adopt HeadRead flips that), and a Ghost (no simulate).
    assert_eq!(dot.entity, SUBJECT, "the dot adopted the transfer subject");
    assert!(dot.adopting, "the dot is a transfer-dest adopt");
    assert!(
        !dot.granted,
        "the adopt HeadRead has not flipped granted yet"
    );
    assert!(
        !dot.authority.simulates(),
        "the adopt is a Ghost (the frozen mirror) — does not simulate, renders nothing"
    );
    assert_eq!(
        dot.authority,
        Authority::Ghost {
            source_fence: Fence::GENESIS,
            since_tick: TickId(1),
        },
        "the dest adopt is born the GENESIS frozen ghost mirror (Promoted to Owned by the crossing)"
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
    // no re-home (the source still owns the client connection — R2), and the Ghost adopt dot
    // does not simulate so it renders no snapshot frame either.
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
fn the_adopt_grant_flip_holds_authority_without_announcing_the_sub_until_promote() {
    // 1d.5b.3b: the adopt grant-flip sets granted + stamps entity_fence + STAYS Ghost, but NO
    // LONGER announces the dest sub — `SubscriptionReady` RELOCATED to on_saga_promote (announced
    // only at the genuine Ghost→Owned promote). So the client stays on the SOURCE sub until then,
    // and demote-before-promote is strict. It still pushes NO SessionAttached (R2).
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let sent = rig.tick(vec![adopted_head(Fence(2))]);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    // Split asserts (each &&-short-circuit false arm is uncoverable — HR5).
    assert!(dot.granted, "the adopt flips granted (authority-held)");
    assert_eq!(dot.entity_fence, Fence(2), "stamped the recorded CAS fence");
    assert!(
        !dot.authority.simulates(),
        "an adopt STAYS a Ghost (renders nothing) until the saga Promote flips it Owned"
    );
    assert!(!dot.adopting, "adopting is cleared on the flip");
    // NO SubscriptionReady at the adopt (it moved to the promote) and NO frame rendered.
    assert!(
        gw_replies(&sent).is_empty(),
        "the adopt announces NO gateway reply (the sub moved to the promote): {sent:?}"
    );
    assert_eq!(
        decode_frames(&sent).len(),
        0,
        "an adopted Ghost dot renders nothing (simulates()==false)"
    );

    // After the crossing lands, the saga Promote DOES announce the dest sub (the relocated one).
    let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]);
    let sent = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]);
    assert_eq!(
        gw_replies(&sent),
        vec![ShardToGateway::SubscriptionReady {
            session: SESSION,
            entity: SUBJECT,
            frame: FrameRef::SystemSpace { system_seed: 7 },
            realm_fence: Fence(1), // the DEST realm fence (grant_realm set Fence(1))
        }],
        "the Promote announces exactly one SubscriptionReady, never a SessionAttached"
    );
    assert!(
        rig.world.resource::<Dots>().0[&SESSION]
            .authority
            .simulates(),
        "the Promote flips the dest Ghost→Owned"
    );
}

#[test]
fn a_non_entity_subject_open_input_slot_is_a_counted_noop() {
    // 1c.8 HR5: the DirectoryKey::transfer_subject_entity None arm — a non-Entity subject (e.g.
    // a Realm saga driven through CommitAuthority) does NOT adopt; counted no-op, never a panic.
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
fn applied_step_redelivery_is_idempotent() {
    // 1d.0 PERMANENT GATE: the dest applied_steps journal dedups by the frozen
    // IdempotencyKey::TransferStep (transfer, step_id). The FIRST apply records + returns
    // FirstApply; every redelivery of the SAME key returns AlreadyApplied with no re-effect;
    // a distinct step_id OR transfer is independent.
    let mut steps = AppliedSteps::default();
    let t = TransferId(1);
    assert_eq!(steps.journal_step(t, 0), StepOutcome::FirstApply);
    assert_eq!(
        steps.journal_step(t, 0),
        StepOutcome::AlreadyApplied,
        "a redelivered (transfer, step_id) is a no-op — never a second effect"
    );
    assert_eq!(
        steps.journal_step(t, 1),
        StepOutcome::FirstApply,
        "a distinct step_id is journaled independently"
    );
    assert_eq!(
        steps.journal_step(TransferId(2), 0),
        StepOutcome::FirstApply,
        "a distinct transfer is journaled independently"
    );
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

#[test]
fn the_saga_demote_flips_the_source_to_ghost_and_acks_unconditionally() {
    // 1d.5b.2 SOURCE consumer of the ordered Demote (the SOLE source-demote driver): drive
    // Owned→Frozen→Ghost (REUSING self_fence_foreign_entity) at the new owner fence, then ack
    // DemoteAck. A redelivered Demote finds an already-Ghost dot → the !simulates() counted no-op
    // — but STILL acks (the unconditional ack: never wedge the saga in Demoting).
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach(); // a granted, locally-owned dot at Fence(1)
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
    let demote = InterShardFlow::Demote(DemoteCmd {
        transfer: TransferId(7),
        subject: DirectoryKey::Entity(entity),
        new_owner_fence: Fence(2),
        step_id: DEMOTE_STEP,
    });
    let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &demote)]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].authority,
        Authority::Ghost {
            source_fence: Fence(2),
            since_tick: TickId(1),
        },
        "the ordered Demote drives Owned{{1}}→Frozen→Ghost at the new owner fence (2)"
    );
    // The demote keys `authority.fence()` to the new owner (2) but does NOT touch `entity_fence`
    // — it stays the dot's OWN old grant fence (1); the intended divergence on a retained Ghost.
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].entity_fence,
        Fence(1),
        "the demote leaves entity_fence at the dot's own grant fence (authority.fence() diverges)"
    );
    assert!(
        saga_ack_to_orch(
            &sent,
            TransferControlAck::DemoteAck {
                transfer: TransferId(7)
            }
        ),
        "DemoteAck is sent to the orchestrator: {sent:?}"
    );
    // The self-fence is purely LOCAL — no directory write (no LeaseRevoke at the stale fence,
    // no delete of the dest's record): the saga demote's ONLY orch-bound emission is the
    // DemoteAck. (Preserves the deleted poll-era test's no-directory-write guard.)
    assert_eq!(
        sent.iter().filter(|(to, _, _)| *to == ORCH).count(),
        1,
        "the saga demote writes nothing to the directory — only the DemoteAck: {sent:?}"
    );
    // Redelivery on the already-Ghost dot: counted skip, but STILL acks.
    let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &demote)]);
    assert_eq!(
        rig.world.resource::<StubStats>().self_fence_skipped,
        1,
        "the already-Ghost redelivery is the !simulates() counted no-op"
    );
    assert!(
        saga_ack_to_orch(
            &sent,
            TransferControlAck::DemoteAck {
                transfer: TransferId(7)
            }
        ),
        "the redelivery STILL acks DemoteAck (never wedge the saga)"
    );
}

#[test]
fn the_saga_demote_on_an_unheld_entity_is_a_clean_noop_but_acks() {
    // The no-match arm of self_fence_foreign_entity (`foreign_takeover_target` finds no dot) —
    // now reachable ONLY via on_saga_demote, since the poll that used to drive it is torn out
    // (1d.5b.2). A Demote for an Entity this shard does not hold is a clean no-op (no flip, no
    // panic) but STILL acks DemoteAck (never wedge the saga in Demoting).
    let mut rig = Rig::new();
    rig.grant_realm(); // realm held, but NO dot attached → this shard holds no entity
    let unheld = EntityId::pack(EntityKind::Player, 99, 99, 99);
    let demote = InterShardFlow::Demote(DemoteCmd {
        transfer: TransferId(7),
        subject: DirectoryKey::Entity(unheld),
        new_owner_fence: Fence(2),
        step_id: DEMOTE_STEP,
    });
    let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &demote)]);
    assert!(
        rig.world.resource::<Dots>().0.is_empty(),
        "an unheld-entity Demote creates or mutates no dot"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().self_fence_skipped,
        0,
        "the no-match arm is distinct from the already-Ghost skip arm"
    );
    assert!(
        saga_ack_to_orch(
            &sent,
            TransferControlAck::DemoteAck {
                transfer: TransferId(7)
            }
        ),
        "an unheld-entity Demote STILL acks DemoteAck: {sent:?}"
    );
}

#[test]
fn the_saga_demote_on_a_non_entity_subject_is_a_counted_noop_but_acks() {
    // The None arm of subject→entity: a Realm/Session/Ship subject has no local Entity dot to
    // demote — counted (`saga_demote_no_entity`), no flip, but the DemoteAck is STILL sent.
    let mut rig = Rig::new();
    rig.grant_realm();
    let demote = InterShardFlow::Demote(DemoteCmd {
        transfer: TransferId(7),
        subject: DirectoryKey::Realm(config().realm),
        new_owner_fence: Fence(2),
        step_id: DEMOTE_STEP,
    });
    let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &demote)]);
    assert_eq!(rig.world.resource::<StubStats>().saga_demote_no_entity, 1);
    assert!(
        saga_ack_to_orch(
            &sent,
            TransferControlAck::DemoteAck {
                transfer: TransferId(7)
            }
        ),
        "a non-Entity Demote still acks: {sent:?}"
    );
}

#[test]
fn the_saga_promote_flips_owned_announces_the_sub_and_spawns_the_ghost() {
    // 1d.5b.3b: on_saga_promote is the REAL promoter. Set up a transfer-dest Ghost dot (adopt +
    // crossing stores the pose, the dot STAYS Ghost), then the saga Promote: flips Ghost→Owned,
    // announces the dest read-sub (RELOCATED from adopt), registers the source ghost-neighbor,
    // and Spawns the source ghost (the dest DRIVES the feed). Redelivery re-acks only.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]); // adopting dot for SUBJECT
    let _ = rig.tick(vec![adopted_head(Fence(2))]); // flip → granted Ghost
    let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]);
    assert!(
        !rig.world.resource::<Dots>().0[&SESSION]
            .authority
            .simulates(),
        "still Ghost after the crossing (the autonomous promote is gone)"
    );

    rig.set_local_tick(5); // pins the Spawn's since_tick deterministically
    let source = NodeId(99);
    let sent = rig.tick(vec![promote_msg(Fence(2), source)]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].authority,
        Authority::Owned { fence: Fence(2) },
        "the Promote flips the dest Ghost→Owned at the new fence"
    );
    assert_eq!(rig.world.resource::<StubStats>().promotes_confirmed, 1);
    assert!(
        saga_ack_to_orch(
            &sent,
            TransferControlAck::PromoteAck {
                transfer: TransferId(7)
            }
        ),
        "PromoteAck is sent: {sent:?}"
    );
    // The dest read-sub is announced NOW (relocated from the adopt flip).
    assert_eq!(
        gw_replies(&sent),
        vec![ShardToGateway::SubscriptionReady {
            session: SESSION,
            entity: SUBJECT,
            frame: FrameRef::SystemSpace { system_seed: 7 },
            realm_fence: Fence(1),
        }],
        "the Promote announces exactly one SubscriptionReady"
    );
    // The source ghost-neighbor is registered (the feed pass has already advanced `seq` once this
    // tick — it runs after process_inbound), and the source ghost is Spawned (the feed started).
    assert_eq!(
        rig.world
            .resource::<GhostColliderRegistration>()
            .0
            .get(&SUBJECT)
            .map(|n| n.source),
        Some(source),
        "the source ghost-neighbor is registered"
    );
    assert!(
        flows_to(&sent, source).contains(&InterShardFlow::Ghost(GhostFlow::SpawnV2 {
            entity: SUBJECT,
            source_fence: Fence(2),
        })),
        "the take-over proof is sent — pose-free (slice F): {sent:?}"
    );

    // Redelivery: re-ack only, NO re-flip / re-register / re-proof — the journal returns
    // AlreadyApplied, so `promote_apply` (which holds the flip + register + proof) is NOT
    // entered. `promotes_confirmed` staying 1 proves it.
    let sent = rig.tick(vec![promote_msg(Fence(2), source)]);
    assert_eq!(rig.world.resource::<StubStats>().promotes_redelivered, 1);
    assert_eq!(
        rig.world.resource::<StubStats>().promotes_confirmed,
        1,
        "no re-flip / re-Spawn on redelivery (promote_apply gated on FirstApply)"
    );
    assert!(saga_ack_to_orch(
        &sent,
        TransferControlAck::PromoteAck {
            transfer: TransferId(7)
        }
    ));
}

#[test]
fn the_saga_promote_re_owns_a_source_equals_dest_ghost_at_the_exact_cas_fence() {
    // The SOURCE==DEST re-own arm of `promote_apply` (task #149): a co-hosted-child crossing whose
    // `head(Realm(dest))` resolves to THIS node reaches `on_saga_promote` with `cmd.source == self_node`
    // (= the Rig's own SHARD id). Unlike the cross-node case (a GENESIS-fenced Ghost, strictly older than
    // the CAS fence, promoted via `AuthorityCmd::Promote`), the ordered same-node `Demote` already
    // self-fenced THIS dot to `Ghost{source_fence: cmd.new_fence}`, so a strict-newer Promote at the SAME
    // fence would `StaleFence`. The `if cmd.source == self_node` arm RE-OWNS the dot directly at that exact
    // fence — the idempotent route-swap the degenerate saga is. Same Ghost-dot fixture as the cross-node
    // test but with `source = SHARD`, asserting `Owned { fence: <the promote's new_fence> }`.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]); // adopting dot for SUBJECT
    let _ = rig.tick(vec![adopted_head(Fence(2))]); // flip → granted Ghost
    let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]); // lands STUB_CROSSING_STEP
    assert!(
        !rig.world.resource::<Dots>().0[&SESSION]
            .authority
            .simulates(),
        "still Ghost after the crossing (source==dest re-own has not run yet)"
    );
    rig.set_local_tick(5);
    // `source == SHARD` (the Rig's own node id) ⇒ `cmd.source == self_node` ⇒ the direct re-own arm.
    let sent = rig.tick(vec![promote_msg(Fence(2), SHARD)]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].authority,
        Authority::Owned { fence: Fence(2) },
        "the source==dest Promote RE-OWNS the dot at the exact CAS fence (no strict-newer StaleFence)"
    );
    assert_eq!(rig.world.resource::<StubStats>().promotes_confirmed, 1);
    assert!(
        saga_ack_to_orch(
            &sent,
            TransferControlAck::PromoteAck {
                transfer: TransferId(7)
            }
        ),
        "the source==dest Promote still acks PromoteAck: {sent:?}"
    );
    // The self-ghost tail is skipped on source==dest (no self-feed loop) — mirrors the re-home guard.
    assert_eq!(
        rig.world
            .resource::<GhostColliderRegistration>()
            .0
            .get(&SUBJECT),
        None,
        "a source==dest promote registers NO self-ghost neighbor",
    );
    assert!(
        flows_to(&sent, SHARD).is_empty(),
        "no GhostFlow::Spawn is sent to self on a source==dest promote: {sent:?}",
    );
}

#[test]
fn on_re_home_creates_an_owned_dot_from_the_pose_acks_and_spawns_the_ghost() {
    // D-37 CELL 2 adopt: the FRESH target receives a `ReHome` and CREATES an Owned dot from the pose
    // (no pre-existing ghost to flip, unlike Promote). Acks PromoteAck, registers + Spawns the source
    // ghost, but emits NO SubscriptionReady (clientless until the session re-homes — D-37/D-36).
    let mut rig = Rig::new();
    rig.grant_realm();
    rig.set_local_tick(5); // pins the Spawn's since_tick deterministically
    let source = NodeId(99);
    let session = SessionId(SUBJECT.0); // the deterministic clientless session key
    let sent = rig.tick(vec![re_home_msg(
        Fence(2),
        source,
        DirectoryKey::Entity(SUBJECT),
    )]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&session].authority,
        Authority::Owned { fence: Fence(2) },
        "the re-home CREATES an Owned dot born at the new fence"
    );
    assert_eq!(rig.world.resource::<Dots>().0[&session].entity, SUBJECT);
    assert_eq!(rig.world.resource::<StubStats>().re_home_adopted, 1);
    assert!(
        saga_ack_to_orch(
            &sent,
            TransferControlAck::PromoteAck {
                transfer: TransferId(7)
            }
        ),
        "PromoteAck is sent: {sent:?}"
    );
    assert!(
        gw_replies(&sent).is_empty(),
        "a clientless re-home announces NO SubscriptionReady"
    );
    assert_eq!(
        rig.world
            .resource::<GhostColliderRegistration>()
            .0
            .get(&SUBJECT)
            .map(|n| n.source),
        Some(source),
        "the source ghost-neighbor is registered"
    );
    assert!(
        flows_to(&sent, source).contains(&InterShardFlow::Ghost(GhostFlow::SpawnV2 {
            entity: SUBJECT,
            source_fence: Fence(2),
        })),
        "the take-over proof is sent — pose-free (slice F): {sent:?}"
    );

    // Redelivery: re-ack only, NO re-adopt (journal AlreadyApplied ⇒ re_home_apply not entered).
    let sent = rig.tick(vec![re_home_msg(
        Fence(2),
        source,
        DirectoryKey::Entity(SUBJECT),
    )]);
    assert_eq!(rig.world.resource::<StubStats>().re_home_redelivered, 1);
    assert_eq!(
        rig.world.resource::<StubStats>().re_home_adopted,
        1,
        "no re-adopt on redelivery (re_home_apply gated on FirstApply)"
    );
    assert!(saga_ack_to_orch(
        &sent,
        TransferControlAck::PromoteAck {
            transfer: TransferId(7)
        }
    ));
}

#[test]
fn a_same_node_re_home_registers_no_self_ghost_and_spawns_none() {
    // GHOST-FEED GUARD (task #149): a SOURCE==DEST re-home (the co-hosted-child crossing whose
    // `head(Realm(dest))` resolves to THIS node) reaches `re_home_apply` with `cmd.source == self_node`.
    // `register_and_spawn_source_ghost` must SKIP the self-ghost registration + the `GhostFlow::Spawn`
    // (else a self-feed loop: the owner would `Delta` its own retained copy). The dot is still adopted
    // Owned + acked; ONLY the ghost tail is skipped. `source == SHARD` (the Rig's own node id).
    let mut rig = Rig::new();
    rig.grant_realm();
    rig.set_local_tick(5);
    let session = SessionId(SUBJECT.0);
    let sent = rig.tick(vec![re_home_msg(
        Fence(2),
        SHARD, // source == this node's id (the source==dest degenerate saga)
        DirectoryKey::Entity(SUBJECT),
    )]);
    // The dot is still adopted Owned + PromoteAck'd (the re-home body ran) …
    assert_eq!(
        rig.world.resource::<Dots>().0[&session].authority,
        Authority::Owned { fence: Fence(2) },
        "a same-node re-home still adopts the dot Owned",
    );
    assert!(saga_ack_to_orch(
        &sent,
        TransferControlAck::PromoteAck {
            transfer: TransferId(7)
        }
    ));
    // … but NO self-ghost was registered and NO Spawn was emitted to self.
    assert_eq!(
        rig.world
            .resource::<GhostColliderRegistration>()
            .0
            .get(&SUBJECT),
        None,
        "a source==dest re-home registers NO self-ghost neighbor",
    );
    assert!(
        flows_to(&sent, SHARD).is_empty(),
        "no GhostFlow::Spawn is sent to self on a same-node re-home: {sent:?}",
    );
}

#[test]
fn a_re_home_colliding_with_the_retained_ghost_flips_it_and_never_mints_a_second_dot() {
    // THE COLLISION (batch review, MAJOR — re_home_apply had no same-entity guard): an upward
    // hand-off demotes this shard's dot to a RETAINED GHOST; the committed dest dies; the
    // forward re-home resolves its target from the flushed pose's frame realm — which on an
    // upward hand-off IS this source realm — so the `ReHome` lands exactly where the ghost
    // still lives. The re-home must FLIP that held dot (same session key, Owned at the CAS
    // fence), never insert a rival under the synthetic `SessionId(entity.0)` key: the orphan
    // ghost's hold would run to TTL and the expiry fan would broadcast `EntityRemoved` for an
    // entity this shard now OWNS and emits.
    let mut rig = Rig::new();
    rig.grant_realm();
    rig.set_local_tick(5);
    insert_owned_dot(&mut rig, SESSION, SUBJECT, DVec3::new(3.0, 0.0, 0.0));
    // The outward demote: the dot survives as the retained Ghost under its ORIGINAL session key
    // (0xAA — which also sorts BELOW the synthetic 0xBEEF key, the order that made the doubled
    // state pick the ghost first).
    let _ = rig.tick(vec![demote_msg(SUBJECT, Fence(2))]);
    {
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert!(
            is_retained_ghost(&dot),
            "precondition: the demote left the retained Ghost in place: {dot:?}"
        );
    }

    // The colliding re-home: source == SHARD (this very node — the CELL-2 shape that resolves
    // the target back onto the source).
    let sent = rig.tick(vec![re_home_msg(
        Fence(3),
        SHARD,
        DirectoryKey::Entity(SUBJECT),
    )]);

    // ONE dot for the entity — flipped in place, never doubled.
    let dots = rig.world.resource::<Dots>();
    assert_eq!(
        dots.0.values().filter(|d| d.entity == SUBJECT).count(),
        1,
        "exactly one dot holds the entity after the colliding re-home: {:?}",
        dots.0,
    );
    assert!(
        !dots.0.contains_key(&SessionId(SUBJECT.0)),
        "no rival dot under the synthetic clientless key — the held session key is reused",
    );
    let dot = dots.0[&SESSION];
    assert_eq!(
        dot.authority,
        Authority::Owned { fence: Fence(3) },
        "the held ghost re-owns at the exact CAS fence"
    );
    assert_eq!(dot.entity_fence, Fence(3));
    assert!(dot.granted & !dot.departing & !dot.adopting);
    assert!(
        !is_retained_ghost(&dot),
        "the flipped dot is the live owner — the expiry fan has no retained ghost to evict"
    );
    assert_eq!(
        (dot.account, dot.gateway),
        (AccountId(1), GATEWAY),
        "the flip keeps the client linkage the ghost retained (a fresh build would be clientless)"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().re_home_flipped,
        1,
        "the collision shape is counted apart from the fresh-build adopts"
    );
    assert_eq!(rig.world.resource::<StubStats>().re_home_adopted, 1);
    // The saga still gets its landing proof, and no self-ghost is registered (source == self).
    assert!(saga_ack_to_orch(
        &sent,
        TransferControlAck::PromoteAck {
            transfer: TransferId(7)
        }
    ));
    assert_eq!(
        rig.world
            .resource::<GhostColliderRegistration>()
            .0
            .get(&SUBJECT),
        None,
        "a source==dest re-home registers NO self-ghost neighbor",
    );
}

#[test]
fn on_re_home_no_ops_for_a_non_entity_subject_or_a_target_without_its_realm() {
    // re_home_apply BAILS on a non-Entity subject; the dispatch BAILS (no adopt) when the target does
    // not hold its realm — both counted degrade-never-panic no-ops.
    let mut rig = Rig::new();
    rig.grant_realm();
    let sent = rig.tick(vec![re_home_msg(
        Fence(2),
        NodeId(99),
        DirectoryKey::Realm(config().realm),
    )]);
    assert_eq!(rig.world.resource::<StubStats>().re_home_no_entity, 1);
    assert_eq!(rig.world.resource::<StubStats>().re_home_adopted, 0);
    // AND IT ACKS NOTHING. The ack is the claim "the entity is here now", and nothing landed: there
    // was no entity in the command to land. This assertion used to read the other way, on the
    // reasoning that acking unconditionally "never wedges the saga" — but the saga is not the thing
    // being protected. An ack for an adopt that refused makes the orchestrator commit authority to a
    // shard that holds nothing, and the source, told the hand-off succeeded, lets go: the entity then
    // exists nowhere at all, which is strictly worse than a stalled saga. Unacked, the saga times out
    // and aborts, and whoever held the entity still holds it.
    assert!(
        !saga_ack_to_orch(
            &sent,
            TransferControlAck::PromoteAck {
                transfer: TransferId(7)
            }
        ),
        "a re-home that adopted NOTHING must not claim it landed: {sent:?}"
    );

    // A re-home delivered while the target does NOT hold its realm (no grant_realm) → no adopt.
    let mut rig2 = Rig::new();
    let _ = rig2.tick(vec![re_home_msg(
        Fence(2),
        NodeId(99),
        DirectoryKey::Entity(SUBJECT),
    )]);
    assert_eq!(rig2.world.resource::<StubStats>().re_home_without_realm, 1);
    assert_eq!(rig2.world.resource::<StubStats>().re_home_adopted, 0);
}

/// Slice F: the dest-side pass STREAMS NOTHING — no pose ever crosses back to the source (the
/// Delta feed is dead; the registration only drives the band-exit Despawn). A registration with
/// no Owned dot is still a counted skip.
#[test]
fn the_dest_sweep_streams_no_poses_and_skips_unowned_registrations() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]);
    let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]);
    let source = NodeId(99);
    let _ = rig.tick(vec![promote_msg(Fence(2), source)]); // Owned + registered
    // The next ticks send the source NOTHING while the entity stays in-band: no Delta exists
    // to send, and Despawn waits for band-exit.
    rig.set_local_tick(6);
    let sent = rig.tick(vec![]);
    assert!(
        flows_to(&sent, source).is_empty(),
        "the sweep streams no poses to the ghost host: {sent:?}"
    );
    // A registration whose entity is NOT owned here (no dot) is a counted no-op.
    rig.world
        .resource_mut::<GhostColliderRegistration>()
        .0
        .insert(
            EntityId(0xABCD),
            GhostNeighbor {
                source,
                anchor: LatticePos::ORIGIN,
            },
        );
    let skipped_before = rig.world.resource::<StubStats>().ghost_feed_skipped;
    let _ = rig.tick(vec![]);
    assert_eq!(
        rig.world.resource::<StubStats>().ghost_feed_skipped,
        skipped_before + 1,
        "a registration with no Owned dot is skipped"
    );
}

#[test]
fn the_dest_feed_pass_skips_a_ghost_dot_registration() {
    // The simulates()-gate negative arm: a registration whose entity dot is a Ghost (not Owned)
    // is skipped — only an OWNED entity's live pose is fed.
    let mut rig = Rig::new();
    let entity = make_retained_ghost(&mut rig, Fence(2)); // a granted Ghost dot for `entity`
    rig.world
        .resource_mut::<GhostColliderRegistration>()
        .0
        .insert(
            entity,
            GhostNeighbor {
                source: NodeId(88),
                anchor: LatticePos::ORIGIN,
            },
        );
    let sent = rig.tick(vec![]);
    assert!(
        flows_to(&sent, NodeId(88)).is_empty(),
        "no feed for a Ghost (non-Owned) dot: {sent:?}"
    );
    assert!(rig.world.resource::<StubStats>().ghost_feed_skipped >= 1);
}

/// Slice F, THE CORE: the pose feed is dead and the take-over proof is pose-free. A tombstoned
/// `Spawn`/`Delta` frame is a counted no-op whose pose NEVER lands (the §4u poison is
/// structurally impossible — its writer is deleted); `SpawnV2` closes the source hold at the
/// exact fence, stops the retained ghost's emit, and evicts the bystanders' figures (the remove
/// message, retimed to hold closure); a stale-fence proof closes nothing; Despawn still tears
/// the dot out; a malformed body is counted, never mis-applied.
#[test]
fn the_take_over_proof_closes_the_hold_stops_the_emit_and_evicts() {
    const BYSTANDER: SessionId = SessionId(0xBB);
    let mut rig = Rig::new();
    let entity = make_retained_ghost(&mut rig, Fence(2)); // retained source Ghost{Fence(2)}
    let demote_pose = rig.world.resource::<Dots>().0[&SESSION].pose;
    insert_owned_dot(&mut rig, BYSTANDER, player(9), DVec3::new(1.0, 0.0, 0.0));
    // The demote opened the SOURCE hold, so the retained ghost EMITS its own-frame demote pose
    // (the fill), alongside the bystander.
    assert!(
        rig.world
            .resource::<HandoffHolds>()
            .0
            .contains_key(&(entity, HoldRole::Source)),
        "DEBUG: the demote opened the Source hold"
    );
    let sent = rig.tick(vec![]);
    let emitted: Vec<EntityId> = entity_rows_to_gateway(&sent);
    assert!(
        emitted.contains(&entity),
        "the retained ghost fills the hand-off window: {emitted:?}"
    );

    // A TOMBSTONED Spawn (the old pose-carrying proof) is a counted no-op: the pose does not
    // land and the hold does not close.
    let foreign = StampedPose::at_rest(
        FrameRef::PlanetCentered { planet_seed: 42 },
        DVec3::new(1.0, 2.0, 3.0),
        UniverseTick(2),
    );
    let undec_before = rig.world.resource::<StubStats>().undecodable;
    let _ = rig.tick(vec![ghost_lifecycle(GhostFlow::Spawn {
        entity,
        pose: foreign,
        source_fence: Fence(2),
        since_tick: vd_core::TickId(0),
    })]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].pose,
        demote_pose,
        "a tombstoned Spawn's pose NEVER lands — the poison's writer is deleted"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().undecodable,
        undec_before + 1
    );
    // ...and so is a tombstoned Delta.
    let _ = rig.tick(vec![ghost_delta(entity, foreign, Fence(2), 1)]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].pose,
        demote_pose,
        "a tombstoned Delta's pose NEVER lands"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().undecodable,
        undec_before + 2
    );
    assert!(
        rig.world
            .resource::<HandoffHolds>()
            .0
            .contains_key(&(entity, HoldRole::Source)),
        "a tombstoned frame closes no hold"
    );

    // A STALE-fence proof (a replayed take-over from a superseded crossing) closes nothing.
    let _ = rig.tick(vec![ghost_lifecycle(GhostFlow::SpawnV2 {
        entity,
        source_fence: Fence(1),
    })]);
    assert!(
        rig.world
            .resource::<HandoffHolds>()
            .0
            .contains_key(&(entity, HoldRole::Source)),
        "a stale-fence proof closes no hold"
    );

    // THE PROOF at the exact fence: the hold closes, the removal fans to the bystander's
    // gateway at the shard's exact tick, and the ghost STOPS emitting — the leaver vanishes at
    // hold closure, never freezing at the boundary.
    let sent = rig.tick(vec![ghost_lifecycle(GhostFlow::SpawnV2 {
        entity,
        source_fence: Fence(2),
    })]);
    assert!(
        !rig.world
            .resource::<HandoffHolds>()
            .0
            .contains_key(&(entity, HoldRole::Source)),
        "the exact-fence proof closes the hold"
    );
    assert_eq!(
        entity_removals(&sent),
        vec![(GATEWAY, entity, UniverseTick(100))],
        "the bystanders' eviction fans at hold closure"
    );
    let emitted: Vec<EntityId> = entity_rows_to_gateway(&sent);
    assert!(
        !emitted.contains(&entity),
        "the retained ghost stopped emitting at hold closure: {emitted:?}"
    );
    assert!(
        rig.world.resource::<Dots>().0.contains_key(&SESSION),
        "the retained DOT stays (the return-crossing target) until band-exit"
    );

    // A REPLAYED proof after the close is a clean no-op (no hold to close, no second eviction).
    let sent = rig.tick(vec![ghost_lifecycle(GhostFlow::SpawnV2 {
        entity,
        source_fence: Fence(2),
    })]);
    assert!(entity_removals(&sent).is_empty(), "a replay evicts nobody");

    // Despawn still TEARS the dot out at band-exit (idempotent; no-host counted).
    let _ = rig.tick(vec![ghost_lifecycle(GhostFlow::Despawn {
        entity,
        source_fence: Fence(2),
    })]);
    assert!(
        !rig.world.resource::<Dots>().0.contains_key(&SESSION),
        "Despawn removes the retained ghost dot (the lifecycle ends)"
    );
    assert_eq!(rig.world.resource::<StubStats>().ghost_despawns, 1);
    let no_host_before = rig.world.resource::<StubStats>().ghost_despawn_no_host;
    let _ = rig.tick(vec![ghost_lifecycle(GhostFlow::Despawn {
        entity: EntityId(0x12345),
        source_fence: Fence(2),
    })]);
    assert_eq!(
        rig.world.resource::<StubStats>().ghost_despawn_no_host,
        no_host_before + 1,
        "a Despawn for an unhosted entity is a counted no-op"
    );

    // A malformed ghost body is counted undecodable, never mis-applied.
    let undec_before = rig.world.resource::<StubStats>().undecodable;
    let _ = rig.tick(vec![Inbound::Wire {
        from: DEST_OWNER,
        class: MsgClass::GhostDelta,
        bytes: crate::io::bytes(vec![0xFF, 0xFF, 0xFF]),
    }]);
    assert_eq!(
        rig.world.resource::<StubStats>().undecodable,
        undec_before + 1,
        "a malformed ghost body is counted undecodable"
    );
}

#[test]
fn the_dest_feed_despawns_on_band_exit_and_deregisters() {
    // 1d.5b.3c → slice F: the dest (owner) drives the source-ghost lifecycle END. While the
    // owned entity is IN the overlap band (anchored at its crossing) the sweep sends NOTHING
    // (the pose feed is dead); once it walks PAST the band's destroy edge the dest emits
    // GhostFlow::Despawn (reliable) + DEREGISTERS.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]);
    let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]);
    let source = NodeId(99);
    // Owned + registered; the anchor is the crossed pose position (the boundary it entered through).
    let _ = rig.tick(vec![promote_msg(Fence(2), source)]);

    // IN-BAND (the dot is at the anchor, distance 0): nothing streams and the registration is
    // KEPT — the false arm of the band-exit decision.
    rig.set_local_tick(6);
    let sent = rig.tick(vec![]);
    assert!(
        flows_to(&sent, source).is_empty(),
        "in-band: the sweep streams nothing (no Delta exists; Despawn waits for band-exit): {sent:?}"
    );
    assert!(
        rig.world
            .resource::<GhostColliderRegistration>()
            .0
            .contains_key(&SUBJECT),
        "in-band: the registration is kept"
    );
    let exits_before = rig.world.resource::<StubStats>().ghost_band_exits;

    // BAND-EXIT: move the owned dot well past the destroy edge from the crossing anchor. The band
    // is `for_motion(move_speed*dt)` = `for_motion(0.1)`, destroy_above = 20*0.1 = 2.0 m; +3 m exits.
    let exit_pos = fm(crossing_pose().pos) + DVec3::new(3.0, 0.0, 0.0);
    rig.world
        .resource_mut::<Dots>()
        .0
        .get_mut(&SESSION)
        .expect("the owned dot")
        .pose
        .pos = LatticePos::from_metres(exit_pos, vd_core::pose::Tier::Fine);
    let sent = rig.tick(vec![]);
    assert!(
        flows_to(&sent, source).contains(&InterShardFlow::Ghost(GhostFlow::Despawn {
            entity: SUBJECT,
            source_fence: Fence(2),
        })),
        "band-exit: the dest Despawns the ghost on the reliable carrier: {sent:?}"
    );
    assert!(
        !rig.world
            .resource::<GhostColliderRegistration>()
            .0
            .contains_key(&SUBJECT),
        "band-exit: the feed is deregistered"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().ghost_band_exits,
        exits_before + 1
    );

    // ...and the feed truly STOPS: a further tick sends nothing to the source (no Delta, no re-Despawn).
    let sent = rig.tick(vec![]);
    assert!(
        flows_to(&sent, source).is_empty(),
        "after deregistration the feed is silent: {sent:?}"
    );
}

#[test]
fn a_band_exit_despawn_refuses_to_remove_a_reowned_owned_dot() {
    // 1d.5b.3c structural refusal — the shard-LOCAL stand-in for the orchestrator in-transfer gate
    // (a `vd-sim` shard cannot see the live-saga set). A stale Despawn for an entity the source has
    // since RE-OWNED removes nothing: only a retained Ghost is torn down, never a live `Owned` dot.
    // Covers `remove_retained_ghost`'s `matches!(Ghost{..})` FALSE arm + the `no_host` counter.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach(); // SESSION's dot is granted + Owned (a re-acquisition would land here)
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
    assert!(
        rig.world.resource::<Dots>().0[&SESSION]
            .authority
            .simulates(),
        "precondition: the dot is Owned"
    );

    let despawns_before = rig.world.resource::<StubStats>().ghost_despawns;
    let _ = rig.tick(vec![ghost_lifecycle(GhostFlow::Despawn {
        entity,
        source_fence: Fence(1),
    })]);
    assert!(
        rig.world.resource::<Dots>().0[&SESSION]
            .authority
            .simulates(),
        "the re-owned Owned dot is structurally refused (kept, still simulating)"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().ghost_despawn_no_host,
        1,
        "the stale Despawn is a counted no-op"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().ghost_despawns,
        despawns_before,
        "...and tears nothing down"
    );
}

#[test]
fn a_retained_ghost_self_emits_but_an_unfed_genesis_ghost_does_not() {
    // 1d.5b.3b → slice F: a RETAINED source Ghost SELF-EMITS its own-frame demote pose while
    // its Source HOLD is open (the demote→take-over fill); a pre-promote DEST-adopt Ghost
    // (GENESIS, no real pose) emits NOTHING.
    // (a) retained source ghost, hold open (make_retained_ghost arms the budget) → emits.
    let mut rig = Rig::new();
    let _entity = make_retained_ghost(&mut rig, Fence(2));
    let sent = rig.tick(vec![]);
    assert_eq!(
        decode_frames(&sent).len(),
        1,
        "the retained source Ghost self-emits its last-Owned pose (no vanish): {sent:?}"
    );

    // (b) a pre-promote DEST-adopt Ghost (GENESIS) → emits nothing.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let sent = rig.tick(vec![adopted_head(Fence(2))]); // granted Ghost{GENESIS}, no pose
    assert_eq!(
        decode_frames(&sent).len(),
        0,
        "an unfed GENESIS dest-adopt Ghost renders nothing until the Promote"
    );
}

#[test]
fn emit_frames_renders_an_emitting_dot_and_filters_a_silent_one_in_the_same_tick() {
    // Covers the entity-collect filter's FALSE arm: a tick with BOTH a retained source Ghost
    // (emits) AND a pre-grant GENESIS adopt Ghost (silent). emit_frames does NOT early-return
    // (the emitter makes gateways non-empty) AND the entity-collect filter EXCLUDES the silent
    // dot — exactly the emitting one renders.
    let mut rig = Rig::new();
    let emitter = make_retained_ghost(&mut rig, Fence(2)); // SESSION: retained Ghost, emits
    let _ = rig.tick(vec![open_input_slot(SessionId(0xBB), GATEWAY, 5)]); // a pre-grant GENESIS adopt Ghost, silent
    let sent = rig.tick(vec![]);
    let entities: Vec<EntityId> = decode_frames(&sent)
        .into_iter()
        .flat_map(|f| f.entities)
        .map(|e| e.entity)
        .collect();
    assert_eq!(
        entities,
        vec![emitter],
        "only the emitting retained Ghost renders; the silent GENESIS adopt Ghost is filtered out"
    );
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

#[test]
fn authored_realm_snaps_computes_a_moving_child_pose_in_the_own_frame_and_ships_static_children() {
    // FA-2c: a moving child's authored `RealmSnap` is its ephemeris pose (position + velocity) at the
    // tick, stamped in the shard's OWN (ambient-root) frame — and the feed ships EVERY direct child,
    // static and moving alike (owner Q3 / D-FO-7: "movers only" was the last rival has-orbit test).
    let elements = orbit();
    let mut moving = BTreeMap::new();
    moving.insert(OTHER_REALM, elements);
    let regions = RealmRegions::new(vec![root_region(), own_region(), child_region()])
        .with_moving_children(kepler_motion_fns(moving));
    let (tick_hz, tick) = (20.0, UniverseTick(1_000));
    let snaps =
        regions.authored_realm_snaps(OWN_REALM, &regions.author_book(OWN_REALM, tick_hz, tick));
    let state = orbital_state(&elements, secs_since_epoch(tick.0, tick_hz));
    assert_eq!(snaps.len(), 1);
    assert_eq!(snaps[0].realm, OTHER_REALM);
    assert_eq!(
        snaps[0].pose.frame,
        frame_of(OWN_REALM),
        "authored in the SHARD'S OWN frame — the only frame a parent can author its children in, and \
         the one its own occupants are measured in. This asserted the ROOT's frame while calling it \
         the own frame; the two were the same numbers only because every fixture realm sat at the \
         origin, and the mismatch put the realm boxes and the things standing in them in different \
         spaces the moment a shard sat anywhere else."
    );
    assert_eq!(fm(snaps[0].pose.pos), state.position);
    assert_eq!(snaps[0].pose.vel, state.velocity);
    assert_eq!(snaps[0].pose.universe_tick, tick);
    // D-FO-7: a STATIC direct child ships a row too — its stored placement, zero velocity, in the
    // same own frame. This is the arm a re-introduced movers-only filter turns RED (batch review:
    // the old form asserted `is_empty()` over a CHILDLESS forest and called it "the static case",
    // so deleting the filter changed nothing here and re-adding it would not have either).
    let static_forest = RealmRegions::new(vec![root_region(), own_region(), child_region()]);
    let static_snaps = static_forest.authored_realm_snaps(
        OWN_REALM,
        &static_forest.author_book(OWN_REALM, tick_hz, tick),
    );
    assert_eq!(
        static_snaps.len(),
        1,
        "a static direct child ships a row — the feed is every direct child, never movers-only"
    );
    assert_eq!(static_snaps[0].realm, OTHER_REALM);
    assert_eq!(static_snaps[0].pose.frame, frame_of(OWN_REALM));
    assert_eq!(static_snaps[0].pose.vel, DVec3::ZERO);
    // A CHILDLESS forest authors NO rows — empty means "no direct children", never "no movers".
    let bare = RealmRegions::new(vec![root_region(), own_region()]);
    assert!(
        bare.authored_realm_snaps(OWN_REALM, &bare.author_book(OWN_REALM, tick_hz, tick))
            .is_empty()
    );
}

#[test]
fn authored_realm_snaps_ships_the_child_frame_as_the_edge_head() {
    // proto_minor 8: each row is a complete placement EDGE — head (the child's own frame), tail
    // (this shard's own frame, on the pose) and value. A receiver reading it needs no join against
    // the reliable shape lane and no ordering guarantee between the two lanes.
    //
    // The head cannot be recovered from `realm`: `frame_for_realm` needs the PARENT to build an
    // `AreaLocal`, and `FrameRef::realm` throws that parent away going the other way. Deriving it
    // receiver-side is exactly the hierarchy lookup this field exists to remove.
    let mut moving = BTreeMap::new();
    moving.insert(OTHER_REALM, orbit());
    let regions = RealmRegions::new(vec![root_region(), own_region(), child_region()])
        .with_moving_children(kepler_motion_fns(moving));
    let snaps = regions.authored_realm_snaps(
        OWN_REALM,
        &regions.author_book(OWN_REALM, 20.0, UniverseTick(1_000)),
    );
    assert_eq!(snaps.len(), 1);
    assert_eq!(
        snaps[0].frame,
        frame_of(OTHER_REALM),
        "the HEAD is the CHILD's own frame — the same frame `frame_context` registers this child \
         under, so the two sides of the ephemeris cannot disagree about what is being placed"
    );
    assert_eq!(
        snaps[0].pose.frame,
        frame_of(OWN_REALM),
        "the TAIL is THIS shard's own frame — the only frame a parent can author a child in"
    );
    assert_ne!(
        snaps[0].frame, snaps[0].pose.frame,
        "head and tail differ on every real row; equal would mean a realm placed inside itself, \
         and a receiver would compose that placement twice"
    );
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

#[test]
fn an_occupant_standing_in_a_child_realm_is_emitted_from_this_shards_own_centre() {
    // SYMPTOM 2, as a unit. An occupant standing inside a realm this shard hosts as a CHILD wears that
    // child's frame — and the realm lane, in the same tick, states where that child sits in THIS
    // shard's frame. Shipping the occupant's row unrestated put one shard's two feeds in two spaces one
    // level apart, so the box and the player standing at its centre drew a whole child-placement
    // apart. Measured at walk scale here: the child sits 300 m out, the occupant stands 7 m from its
    // centre, and the row that leaves says 307 — in this shard's frame, the one it speaks in.
    let child_at = 300.0;
    let inside_child = 7.0;
    let regions = RealmRegions::new(vec![
        root_region(),
        own_region(),
        region(
            OTHER_REALM,
            Some(OWN_REALM),
            DVec3::new(child_at, 0.0, 0.0),
            1000.0,
        ),
    ]);
    let own_frame = emit_context(&regions);
    let entity = EntityId::pack(EntityKind::Player, 10, 11, 11);
    let mut dot = slice6_dot(entity, own_frame, Authority::Owned { fence: Fence(1) });
    dot.pose = StampedPose::at_rest(
        frame_of(OTHER_REALM),
        DVec3::new(inside_child, 0.0, 0.0),
        UniverseTick(100),
    );
    let mut dots = Dots::default();
    dots.0.insert(SessionId(11), dot);

    let mut stats = StubStats::default();
    let entities = emitted_entities(
        &dots,
        &HandoffHolds::default(),
        own_frame,
        &ledger_at(&regions, 20.0, UniverseTick(100)),
        OWN_REALM,
        &mut stats,
    );
    assert_eq!(entities.len(), 1);
    assert_eq!(
        entities[0].pose.frame, own_frame,
        "the row leaves in the ONE frame this shard speaks in",
    );
    assert_eq!(
        fm(entities[0].pose.pos),
        DVec3::new(child_at + inside_child, 0.0, 0.0),
        "this shard added where it put its own child: {child_at} + {inside_child}",
    );
    assert_eq!(
        stats.entity_rows_foreign_labelled, 0,
        "a row this shard CAN place is not a degrade",
    );
}

#[test]
fn a_fed_ghost_ships_with_the_neighbours_frame_label_and_is_never_re_measured_here() {
    // THE FED-GHOST RULE, and it is the one this shard is most likely to get wrong. A fed ghost's pose
    // was authored by the NEIGHBOUR that owns the entity, measured from the NEIGHBOUR's centre and
    // labelled with the NEIGHBOUR's frame. This shard has no idea where that neighbour sits relative to
    // itself unless the neighbour is one of its DIRECT CHILDREN — so it must not touch the number, and
    // it must not quietly restamp the label to its own, which would claim the value is measured from
    // here.
    //
    // It used to be FOLDED here, against a per-tick table of universe-root absolutes keyed on the
    // ghost's own frame. That worked only because the shard believed it knew where every frame was in
    // the universe. With that belief removed, the honest answer is to forward exactly what arrived and
    // let the gateway — which holds both realms' placements — relate them.
    let local_frame = frame_of(OWN_REALM);
    let ghost_frame = frame_of(OTHER_REALM);
    let entity = EntityId::pack(EntityKind::Player, 10, 9, 9);
    let ghost_pose =
        StampedPose::at_rest(ghost_frame, DVec3::new(0.0, 0.0, 7.0), UniverseTick(100));
    let mut dot = slice6_dot(entity, local_frame, Authority::Owned { fence: Fence(1) });
    dot.pose = ghost_pose;
    let mut dots = Dots::default();
    dots.0.insert(SessionId(9), dot);

    // The forest holds this shard's own realm and NOT the ghost's, so the ghost's frame is one this
    // shard has genuinely never been told the position of — the case the rule is about.
    let regions = RealmRegions::new(vec![root_region(), own_region()]);
    let own_frame = emit_context(&regions);
    let mut stats = StubStats::default();
    let entities = emitted_entities(
        &dots,
        &HandoffHolds::default(),
        own_frame,
        &ledger_at(&regions, 20.0, UniverseTick(100)),
        OWN_REALM,
        &mut stats,
    );
    assert_eq!(entities.len(), 1);
    assert_eq!(
        entities[0].pose, ghost_pose,
        "the neighbour's pose ships BIT-IDENTICAL — same value, same neighbour frame label"
    );
    assert_eq!(
        stats.entity_rows_foreign_labelled, 1,
        "and the degrade is COUNTED, not silent",
    );
    assert_eq!(
        entities[0].pose.frame, ghost_frame,
        "the label stays the NEIGHBOUR's; restamping it local would claim the number is measured \
         from here, which is exactly the lie the fold shipped"
    );
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

#[test]
fn emitted_entities_ships_the_emitting_union_bit_identical_and_drops_the_silent() {
    // The EMIT UNION, slice F's shape: an Owned dot emits (the authority truth); a RETAINED
    // source ghost emits ONLY while its hand-off hold is open (the demote→take-over fill —
    // hold closed means the leaver already vanished from bystanders' screens). A pre-grant
    // provisional Ghost, a Frozen dot, and a hold-CLOSED retained ghost emit NOTHING. (The
    // fed-ghost category died with the pose feed.)
    //
    // And every surviving row's pose is BIT-IDENTICAL to the dot's own — value, velocity, tick
    // AND frame label. There is no compose, no re-stamp, no feed to overwrite it. A shard ships
    // what it holds.
    let frame = frame_of(OWN_REALM);
    let owned = EntityId::pack(EntityKind::Player, 10, 1, 1);
    let retained = EntityId::pack(EntityKind::Player, 10, 3, 3);
    let lapsed = EntityId::pack(EntityKind::Player, 10, 2, 2);
    let provisional = EntityId::pack(EntityKind::Player, 10, 4, 4);
    let frozen = EntityId::pack(EntityKind::Player, 10, 5, 5);

    let mut dots = Dots::default();
    dots.0.insert(
        SessionId(1),
        slice6_dot(owned, frame, Authority::Owned { fence: Fence(1) }),
    );
    // A RETAINED source ghost with an OPEN hold: granted, not departing, post-GENESIS fence.
    dots.0.insert(
        SessionId(3),
        slice6_dot(
            retained,
            frame,
            Authority::Ghost {
                source_fence: Fence(1),
                since_tick: TickId(1),
            },
        ),
    );
    // A RETAINED source ghost whose hold CLOSED (no HandoffHolds entry) — silent: the leaver
    // vanished at hold closure and must not reappear frozen.
    dots.0.insert(
        SessionId(2),
        slice6_dot(
            lapsed,
            frame,
            Authority::Ghost {
                source_fence: Fence(2),
                since_tick: TickId(1),
            },
        ),
    );
    // A PRE-GRANT provisional Ghost holding GENESIS — silent (nothing owns it yet).
    let mut provisional_dot = slice6_dot(
        provisional,
        frame,
        Authority::Ghost {
            source_fence: Fence::GENESIS,
            since_tick: TickId(1),
        },
    );
    provisional_dot.granted = false;
    dots.0.insert(SessionId(4), provisional_dot);
    dots.0.insert(
        SessionId(5),
        slice6_dot(
            frozen,
            frame,
            Authority::Frozen {
                transfer: TransferId(1),
                fence: Fence(1),
            },
        ),
    );

    let mut holds = HandoffHolds::default();
    holds.0.insert(
        (retained, HoldRole::Source),
        HandoffHold {
            opened_at: TickId(1),
            takeover_fence: Fence(1),
        },
    );

    // Every dot here already stands in this shard's OWN realm, so the restatement is the identity and
    // "bit-identical" below is asserted through the same-frame short-circuit, not through a degrade.
    let regions = RealmRegions::new(vec![root_region(), own_region()]);
    let own_frame = emit_context(&regions);
    let mut stats = StubStats::default();
    let entities = emitted_entities(
        &dots,
        &holds,
        own_frame,
        &ledger_at(&regions, 20.0, UniverseTick(100)),
        OWN_REALM,
        &mut stats,
    );
    assert_eq!(
        stats.entity_rows_foreign_labelled, 0,
        "an occupant in this shard's own realm is never a degrade",
    );
    let mut got: Vec<EntityId> = entities.iter().map(|e| e.entity).collect();
    got.sort_unstable();
    let mut want = vec![owned, retained];
    want.sort_unstable();
    assert_eq!(
        got, want,
        "Owned | retained-with-open-hold emit; a closed-hold ghost is silent"
    );

    for snap in &entities {
        let dot = dots
            .0
            .values()
            .find(|d| d.entity == snap.entity)
            .expect("every emitted row came from a dot");
        assert_eq!(
            snap.pose, dot.pose,
            "the emitted pose is BIT-IDENTICAL to the held one — frame label included"
        );
    }
}

#[test]
fn emit_realm_frames_states_its_children_only_to_an_open_window_with_authority() {
    // The window lane's two gates, both arms each: a realm states its children's placements
    // when (a) it holds its realm lease and (b) somebody holds a window on it. The old
    // "is a player standing here" gate is GONE with the direct realm datagram (Slice C2):
    // WHO is watching is the subscriber's business, and a childless or unwatched realm simply
    // states nothing.
    let plant = |rig: &mut Rig| {
        let mut moving = BTreeMap::new();
        moving.insert(OTHER_REALM, orbit());
        *rig.world.resource_mut::<RealmRegions>() =
            RealmRegions::new(vec![root_region(), own_region(), child_region()])
                .with_moving_children(kepler_motion_fns(moving));
    };
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    let realms = |sent: &[(NodeId, MsgClass, Vec<u8>)]| -> Vec<RealmId> {
        window_frames(sent)
            .into_iter()
            .flat_map(|(_, _, _, _, rows)| rows)
            .map(|s| s.realm)
            .collect()
    };

    // (a) HAPPY: granted + a child + an open window ⇒ the child's placement is stated.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant(&mut rig);
    assert_eq!(
        realms(&rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)])),
        vec![OTHER_REALM]
    );

    // (b) NO WINDOW: granted + a child but nobody subscribed ⇒ silent.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant(&mut rig);
    assert!(realms(&rig.tick(vec![])).is_empty());

    // (c) NO CHILD: granted + an open window but an EMPTY roster ⇒ a frame with no rows (the
    // per-tick stamp a leaf still owes its subscriber), so no realm is named.
    let mut rig = Rig::new();
    rig.grant_realm();
    assert!(realms(&rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)])).is_empty());

    // (d) NO AUTHORITY: an ungranted shard is silent even with a child and a window.
    let mut rig = Rig::new();
    plant(&mut rig);
    assert!(realms(&rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)])).is_empty());
}

#[test]
fn the_saga_promote_on_an_unheld_or_non_entity_subject_is_a_counted_noop_but_acks() {
    // on_saga_promote's no-dot arms: a Promote for an entity not held here, and for a non-Entity
    // (Realm) subject, each flip nothing (counted `promote_no_dot`) but STILL ack PromoteAck.
    let mut rig = Rig::new();
    rig.grant_realm();
    // (a) unheld Entity subject.
    let sent = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]); // SUBJECT not held here
    assert_eq!(rig.world.resource::<StubStats>().promote_no_dot, 1);
    assert!(saga_ack_to_orch(
        &sent,
        TransferControlAck::PromoteAck {
            transfer: TransferId(7)
        }
    ));
    // (b) non-Entity (Realm) subject.
    let realm_promote = InterShardFlow::Promote(PromoteCmd {
        transfer: TransferId(8),
        subject: DirectoryKey::Realm(RealmId::System(9)),
        new_fence: Fence(2),
        step_id: PROMOTE_STEP,
        source: NodeId(99),
    });
    let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &realm_promote)]);
    assert_eq!(rig.world.resource::<StubStats>().promote_no_dot, 2);
    assert!(saga_ack_to_orch(
        &sent,
        TransferControlAck::PromoteAck {
            transfer: TransferId(8)
        }
    ));
}

#[test]
fn the_saga_promote_before_the_crossing_lands_defers_the_flip_but_acks() {
    // pose-before-promote: a Promote arriving BEFORE the crossing journaled does NOT flip (no
    // poseless origin frame); counted `promote_before_crossing`, still acks. The dot stays Ghost.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]); // granted Ghost, NO crossing yet
    let sent = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]);
    assert!(
        !rig.world.resource::<Dots>().0[&SESSION]
            .authority
            .simulates(),
        "the Promote does NOT flip a dot whose crossing pose has not landed (stays Ghost)"
    );
    assert_eq!(rig.world.resource::<StubStats>().promote_before_crossing, 1);
    assert!(saga_ack_to_orch(
        &sent,
        TransferControlAck::PromoteAck {
            transfer: TransferId(7)
        }
    ));
}

#[test]
fn a_deferred_promote_flips_when_re_driven_after_the_crossing_lands() {
    // FREEZE FIX (the return-crossing total-freeze): a Promote arriving BEFORE its crossing pose DEFERS
    // WITHOUT journaling PROMOTE_STEP, so when the crossing later lands and the saga RE-DRIVES the Promote,
    // the flip completes (Ghost→Owned). The old code journaled PROMOTE_STEP on the deferred delivery, so the
    // re-drive hit AlreadyApplied and the flip NEVER happened — the entity stayed a silent non-emitting
    // Ghost (no feed, input dropped as PendingAuthority), the exact freeze a RETURN to a re-adopting shard
    // triggers when the ordered Promote outruns the crossing pose.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]); // granted Ghost, NO crossing yet
    // Promote arrives BEFORE the crossing ⇒ DEFER (stays Ghost; PROMOTE_STEP NOT journaled).
    let _ = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]);
    assert!(
        !rig.world.resource::<Dots>().0[&SESSION]
            .authority
            .simulates(),
        "deferred: stays Ghost"
    );
    assert_eq!(rig.world.resource::<StubStats>().promotes_confirmed, 0);
    // The crossing pose LANDS (STUB_CROSSING_STEP journaled for the transfer).
    let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]);
    // The saga RE-DRIVES the Promote (a redelivery of the same step). Pre-fix this hit AlreadyApplied and
    // never re-ran the flip; with the fix PROMOTE_STEP was withheld on the defer, so the re-run COMPLETES it.
    let sent = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]);
    assert!(
        rig.world.resource::<Dots>().0[&SESSION]
            .authority
            .simulates(),
        "the re-driven Promote flips Ghost→Owned once the crossing has landed (no permanent freeze)"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().promotes_confirmed,
        1,
        "exactly one confirmed flip"
    );
    // The flip announces the dest read sub (so the client's feed re-opens) and acks.
    assert!(saga_ack_to_orch(
        &sent,
        TransferControlAck::PromoteAck {
            transfer: TransferId(7)
        }
    ));
}

#[test]
fn the_saga_promote_without_a_realm_is_a_counted_noop_never_a_panic() {
    // The realm-owner guard's None arm (1d.5b.3b audit hardening): a Promote arriving while this
    // shard does NOT hold its realm is DROPPED as a counted no-op (DEGRADE, never panic), no ack
    // — the saga re-drives. Mirrors every sibling handler's degrade-not-crash discipline.
    let mut rig = Rig::new(); // NO grant_realm → the shard holds no realm (authority.0 == None)
    let sent = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]);
    assert_eq!(rig.world.resource::<StubStats>().promote_without_realm, 1);
    assert!(
        !saga_ack_to_orch(
            &sent,
            TransferControlAck::PromoteAck {
                transfer: TransferId(7)
            }
        ),
        "a realm-less Promote is dropped (no ack) — the saga re-drives: {sent:?}"
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

/// Stage B4 (§4u refutation 8): an input-idle durable pose must follow the clock. The scan and
/// the flush resolve MOVING-child placements at the POSE's stamp, so a stamp frozen at the last
/// input measured a parked occupant against a frozen world — a planet could sweep through a
/// parked ship with no crossing ever firing. The re-stamp is position-preserving (`Frozen`
/// continuity: stopped means stopped).
#[test]
fn an_input_idle_dot_is_restamped_to_the_current_tick_each_tick() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]);
    let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]);
    let _ = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]); // Owned — the dot simulates
    let before = rig.world.resource::<Dots>().0[&SESSION].pose;
    rig.world.resource_mut::<ClockSample>().universe_tick =
        UniverseTick(before.universe_tick.0 + 50);
    let _ = rig.tick(vec![]);
    let after = rig.world.resource::<Dots>().0[&SESSION].pose;
    assert_eq!(
        after.universe_tick,
        UniverseTick(before.universe_tick.0 + 50),
        "the idle stamp follows the shard clock",
    );
    assert_eq!(
        after.pos, before.pos,
        "the re-stamp never moves a stopped player"
    );
    assert_eq!(
        after.frame, before.frame,
        "the re-stamp never touches the frame"
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
    // Stage B2: the early slot is BUFFERED, not dropped — the lease affirm drains it through the
    // identical adopt path, so the adopt completes LATE instead of NEVER (the fence-7 strand,
    // rehome_one_mechanism §4v fact 3). The drained dot is the ordinary adopt-Ghost: input armed,
    // watermark seeded, awaiting its HeadRead grant and the saga Promote.
    rig.grant_realm();
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(
        stats.input_slots_drained, 1,
        "the buffered slot drained at the lease affirm"
    );
    let dot = rig
        .world
        .resource::<Dots>()
        .0
        .get(&SESSION)
        .copied()
        .expect("the lease affirm minted the adopt dot from the buffered slot");
    assert!(
        dot.adopting,
        "the drained slot is a transfer-destination adopt"
    );
    assert!(
        dot.input_active,
        "the drained slot armed input from its owning gateway"
    );
    assert_eq!(
        dot.last_applied_seq,
        Some(5),
        "the drained slot seeded the resume watermark it was buffered with",
    );
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

/// The three ingress refusals, driven end to end through the wire (never direct calls): a
/// CROSSING whose pose this shard cannot measure is not adopted (and not acked — an ack would
/// commit authority to a shard holding nothing); a RE-HOME with the same fault reconstructs
/// nothing; a FLUSH for an entity this shard does not hold ships nothing. Each is the loud
/// counted degrade the Stage-A log points instrument.
#[test]
fn unplaceable_ingresses_are_refused_counted_and_never_acked() {
    // (1) The crossing adopt refusal.
    let mut rig = Rig::new();
    rig.grant_realm();
    let foreign = StampedPose::at_rest(
        FrameRef::PlanetCentered { planet_seed: 999 },
        DVec3::new(1.0, 2.0, 3.0),
        UniverseTick(200),
    );
    let sent = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), foreign)]);
    assert_eq!(rig.world.resource::<StubStats>().arrivals_unplaceable, 1);
    assert!(
        !rig.world
            .resource::<Dots>()
            .0
            .values()
            .any(|d| d.entity == SUBJECT),
        "the unplaceable crossing adopted nothing"
    );
    assert!(
        !saga_ack_to_orch(
            &sent,
            TransferControlAck::PromoteAck {
                transfer: TransferId(7)
            }
        ),
        "an adopt that refused must not claim it landed: {sent:?}"
    );
    // (2) The re-home reconstruction refusal — the same one rule, at the second ingress.
    let mut rig = Rig::new();
    rig.grant_realm();
    let cmd = InterShardFlow::ReHome(ReHomeCmd {
        transfer: TransferId(7),
        universe_epoch: vd_core::EpochId(1),
        subject: DirectoryKey::Entity(SUBJECT),
        new_fence: Fence(2),
        step_id: RE_HOME_STEP,
        state: ReHomeState::PoseOnly(foreign),
        source: NodeId(99),
    });
    let _ = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &cmd)]);
    assert_eq!(rig.world.resource::<StubStats>().arrivals_unplaceable, 1);
    assert_eq!(rig.world.resource::<StubStats>().re_home_adopted, 0);
    // (3) The flush for an entity nobody here holds: nothing ships, loudly.
    let mut rig = Rig::new();
    rig.grant_realm();
    let sent = rig.tick(vec![flush_msg(EntityId(0xDEAD))]);
    assert_eq!(
        to_orch(&sent),
        vec![],
        "no SourceFlushed (nothing at all) for an entity this shard does not hold"
    );
    // (4) A flush whose Stage-B1 re-validation refuses (the dot is still HELD by this realm's
    // own band) ships nothing either — the saga aborts pre-commit and this shard keeps
    // authority. The refusal itself is unit-covered on `flush_pose_for_dest`; this drives the
    // `on_flush_source` arm that consumes it.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(&mut rig, vec![root_region(), own_region()]);
    let _ = rig.attach();
    let held = rig.world.resource::<Dots>().0[&SESSION].entity;
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::FlushSource(FlushSource {
            transfer: TransferId(7),
            subject: DirectoryKey::Entity(held),
            step_id: FLUSH_SOURCE_STEP,
            to_realm: RealmId::System(9),
            to_parent: None,
        }),
    )]);
    assert_eq!(
        to_orch(&sent),
        vec![],
        "a departure that is no longer true ships no SourceFlushed"
    );
    assert_eq!(rig.world.resource::<StubStats>().flush_stale_exit, 1);
    // (5) A crossing into a realm this shard neither is nor hosts has no frame to measure the
    // arrival in — refused counted (the `arrival_frame` None arm).
    let mut rig = Rig::new();
    rig.grant_realm();
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::Transfer(TransferEnvelope {
            transfer_id: TransferId(8),
            universe_epoch: vd_core::EpochId(1),
            schema_version: vd_wire::intershard::TRANSFER_SCHEMA_VERSION,
            fence: Fence(2),
            step_id: STUB_CROSSING_STEP,
            class: vd_core::entity_kind::DurabilityClass::Durable,
            payload: TransitionPayload::StubCrossing {
                entity: SUBJECT,
                from_realm: FROM_REALM,
                to_realm: RealmId::System(9),
                pose: crossing_pose(),
                state: vec![],
            },
        }),
    )]);
    assert_eq!(rig.world.resource::<StubStats>().arrivals_unplaceable, 1);
    assert_eq!(
        to_orch(&sent),
        vec![],
        "an arrival with no nameable frame is never acked"
    );
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

/// THE REMOVE MESSAGE at the band-exit teardown (D-4(a)): the Despawn that tears out the
/// retained ghost — the last emitter of a leaver here — tells every remaining dot's gateway to
/// evict the figure, ONE message per DISTINCT gateway (two bystanders behind one gateway share
/// one; a second gateway gets its own), each stamped with the realm fence and EXACTLY this
/// shard's universe tick (the resurrect guard's whole input — a wrong stamp mis-gates every
/// client refusal). A stale Despawn that removes nothing tells nobody, and a shard whose lease
/// lapsed (self-fenced) withholds the removal LOUDLY — counted, never silent.
#[test]
fn a_despawned_leavers_removal_is_told_once_per_bystander_gateway_and_stamped() {
    const BYSTANDER_A: SessionId = SessionId(0xBB);
    const BYSTANDER_B: SessionId = SessionId(0xBC);
    const BYSTANDER_FAR: SessionId = SessionId(0xBD);
    const OTHER_GATEWAY: NodeId = NodeId(21);
    let mut rig = Rig::new();
    let leaver = make_retained_ghost(&mut rig, Fence(2));
    // THREE bystanders: two behind the default GATEWAY (the dedup half), one behind a second
    // gateway (the multi-gateway half).
    insert_owned_dot(&mut rig, BYSTANDER_A, player(9), DVec3::new(1.0, 0.0, 0.0));
    insert_owned_dot(&mut rig, BYSTANDER_B, player(10), DVec3::new(2.0, 0.0, 0.0));
    insert_owned_dot(
        &mut rig,
        BYSTANDER_FAR,
        player(11),
        DVec3::new(3.0, 0.0, 0.0),
    );
    rig.world
        .resource_mut::<Dots>()
        .0
        .get_mut(&BYSTANDER_FAR)
        .expect("just inserted")
        .gateway = OTHER_GATEWAY;
    let sent = rig.tick(vec![ghost_lifecycle(GhostFlow::Despawn {
        entity: leaver,
        source_fence: Fence(2),
    })]);
    let removals = entity_removals(&sent);
    // The stamp is THE shard clock, exactly (the Rig clock reads universe_tick 100).
    assert_eq!(
        removals,
        vec![
            (GATEWAY, leaver, UniverseTick(100)),
            (OTHER_GATEWAY, leaver, UniverseTick(100)),
        ],
        "one removal per DISTINCT gateway, stamped with the shard's exact universe tick"
    );
    // A second Despawn removes nothing (already gone) ⇒ NO second removal fans out.
    let sent = rig.tick(vec![ghost_lifecycle(GhostFlow::Despawn {
        entity: leaver,
        source_fence: Fence(2),
    })]);
    assert!(
        entity_removals(&sent).is_empty(),
        "a no-op Despawn tells nobody"
    );
}

/// The despawn emit's NO-LEASE arm: a self-fenced shard (lease lapsed mid-teardown) withholds
/// the removal — an unowned shard is silent — but COUNTS the suppression, because each one is
/// a bystander who may keep a frozen figure until the sub machinery catches up.
#[test]
fn a_despawn_on_a_leaseless_shard_suppresses_the_removal_loudly() {
    const BYSTANDER: SessionId = SessionId(0xBB);
    let mut rig = Rig::new();
    let leaver = make_retained_ghost(&mut rig, Fence(2));
    insert_owned_dot(&mut rig, BYSTANDER, player(9), DVec3::new(1.0, 0.0, 0.0));
    rig.world.resource_mut::<RealmAuthority>().0 = None; // the self-fenced shape
    let sent = rig.tick(vec![ghost_lifecycle(GhostFlow::Despawn {
        entity: leaver,
        source_fence: Fence(2),
    })]);
    assert!(
        entity_removals(&sent).is_empty(),
        "an unowned shard is silent"
    );
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .entity_removals_suppressed_no_lease,
        1,
        "…but never silently: the suppression is counted"
    );
    // The SAME suppression at the take-over proof: a fresh rig, lease dropped before the
    // exact-fence SpawnV2 lands — the hold closes, the eviction is withheld + counted.
    let mut rig = Rig::new();
    let leaver = make_retained_ghost(&mut rig, Fence(2));
    insert_owned_dot(&mut rig, BYSTANDER, player(9), DVec3::new(1.0, 0.0, 0.0));
    rig.world.resource_mut::<RealmAuthority>().0 = None;
    let sent = rig.tick(vec![ghost_lifecycle(GhostFlow::SpawnV2 {
        entity: leaver,
        source_fence: Fence(2),
    })]);
    assert!(
        !rig.world
            .resource::<HandoffHolds>()
            .0
            .contains_key(&(leaver, HoldRole::Source)),
        "the proof still closes the hold"
    );
    assert!(
        entity_removals(&sent).is_empty(),
        "an unowned shard is silent at the proof too"
    );
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .entity_removals_suppressed_no_lease,
        1,
        "…and counted"
    );
}

/// THE REMOVE MESSAGE at detach completion (D-4(a)): the directory confirming a logout's
/// revoke is the permanent stop of that avatar here — the remaining bystander's gateway is
/// told to evict it, stamped with EXACTLY the shard's universe tick. The PROVISIONAL drop
/// tells nobody — driven here, not asserted in prose: an ungranted dot's detach removes it
/// with no removal fanned (it never emitted, so no client holds its figure). A lease lapse
/// at the completion instant withholds the removal loudly (counted).
#[test]
fn a_detached_dots_removal_is_told_to_the_bystanders_gateway() {
    const BYSTANDER: SessionId = SessionId(0xBB);
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach();
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
    insert_owned_dot(&mut rig, BYSTANDER, player(9), DVec3::new(1.0, 0.0, 0.0));
    // Phase 1: detach → departing (held); no removal yet (the dot still emits).
    let detach = GatewayToShard::DetachSession {
        session: SESSION,
        fence: Fence(1),
    };
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &detach)]);
    assert!(
        entity_removals(&sent).is_empty(),
        "a departing dot still emits — nobody is told yet"
    );
    // Phase 2: the headless entity head confirms the revoke — despawn + THE removal, stamped
    // with the shard clock exactly (the Rig clock reads universe_tick 100).
    let gone = DirectoryReply::Head {
        key: DirectoryKey::Entity(entity),
        record: None,
    };
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::DirectoryReply(gone),
    )]);
    assert_eq!(
        entity_removals(&sent),
        vec![(GATEWAY, entity, UniverseTick(100))],
        "the bystander's gateway is told exactly once, at the shard's exact tick"
    );
}

/// The detach path's two NEGATIVE arms, driven: (a) a PROVISIONAL (ungranted) dot's detach
/// drops it with NO removal — it never emitted, so no client holds a figure to evict;
/// (b) a detach completing on a LEASELESS shard withholds the removal loudly (counted).
#[test]
fn a_provisional_drop_and_a_leaseless_completion_fan_no_removal() {
    const BYSTANDER: SessionId = SessionId(0xBB);
    // (a) the provisional drop: attach_request WITHOUT the grant confirm — an ungranted dot.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach_request(SESSION, GATEWAY);
    insert_owned_dot(&mut rig, BYSTANDER, player(9), DVec3::new(1.0, 0.0, 0.0));
    let detach = GatewayToShard::DetachSession {
        session: SESSION,
        fence: Fence(1),
    };
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &detach)]);
    assert!(
        !rig.world.resource::<Dots>().0.contains_key(&SESSION),
        "the provisional dot is dropped immediately (no directory round-trip)"
    );
    assert!(
        entity_removals(&sent).is_empty(),
        "…and NO removal fans — a pre-grant dot never emitted"
    );
    // (b) the completion on a leaseless shard: granted dot, detach, lease lapses, confirm.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach();
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
    insert_owned_dot(&mut rig, BYSTANDER, player(9), DVec3::new(1.0, 0.0, 0.0));
    let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &detach)]);
    rig.world.resource_mut::<RealmAuthority>().0 = None;
    let gone = DirectoryReply::Head {
        key: DirectoryKey::Entity(entity),
        record: None,
    };
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::DirectoryReply(gone),
    )]);
    assert!(
        entity_removals(&sent).is_empty(),
        "an unowned shard is silent"
    );
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .entity_removals_suppressed_no_lease,
        1,
        "…but the suppression is counted"
    );
}

#[test]
fn flush_source_ships_the_held_dots_pose() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach(); // a granted login dot for SESSION
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;

    // (a) BEFORE any input: the dot is at origin with NO applied seq → drained_seq defaults 0.
    let dot0 = rig.world.resource::<Dots>().0[&SESSION];
    let sent0 = rig.tick(vec![flush_msg(entity)]);
    assert_eq!(
        to_orch(&sent0),
        vec![InterShardFlow::TransferAck(TransferAck::SourceFlushed {
            transfer_id: TransferId(7),
            step_id: FLUSH_SOURCE_STEP,
            pose: dot0.pose,
            drained_seq: 0,
        })],
        "ships the held dot's pose; an unset watermark defaults to 0"
    );

    // (b) AFTER applying seq 1: the pose moved and the watermark is Some(1).
    let _ = rig.tick(vec![input_for(SESSION, 1, GATEWAY)]);
    let dot1 = rig.world.resource::<Dots>().0[&SESSION];
    assert_ne!(fm(dot1.pose.pos), DVec3::ZERO, "the dot moved on input");
    let sent1 = rig.tick(vec![flush_msg(entity)]);
    assert_eq!(
        to_orch(&sent1),
        vec![InterShardFlow::TransferAck(TransferAck::SourceFlushed {
            transfer_id: TransferId(7),
            step_id: FLUSH_SOURCE_STEP,
            pose: dot1.pose,
            drained_seq: 1,
        })],
        "ships the moved pose + the real drain watermark"
    );
}

#[test]
fn flush_source_for_an_unheld_or_non_entity_subject_ships_nothing() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach();
    // An entity this shard does not hold → no ship (counted no-op).
    let sent = rig.tick(vec![flush_msg(EntityId(0xDEAD))]);
    assert!(to_orch(&sent).is_empty(), "no pose for an unheld entity");
    // A non-Entity subject → no ship.
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::FlushSource(FlushSource {
            transfer: TransferId(7),
            subject: DirectoryKey::Realm(RealmId::System(9)),
            step_id: FLUSH_SOURCE_STEP,
            to_realm: RealmId::System(0),
            to_parent: None,
        }),
    )]);
    assert!(
        to_orch(&sent).is_empty(),
        "no pose for a non-Entity subject"
    );
}

#[test]
fn an_arriving_crossing_reseeds_the_swept_prior_at_the_arrival_point() {
    // THE STALE PRIOR AFTER A HAND-OFF. A Ghost is minted with its prior at the realm ORIGIN, because
    // a fresh dot has no history. Once membership tests the tick's whole motion segment, leaving it
    // there makes the first scan after arrival sweep a realm-wide line from the destination's centre
    // to wherever the subject actually landed — through every region on that line, none of which it
    // visited — and feed the result straight into a re-home decision. An arrival is a discontinuity,
    // not a movement.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]);
    let pose = crossing_pose();
    // The Ghost starts with the prior at the origin, and the arrival pose is somewhere else — so the
    // fixture can distinguish "reseeded" from "happened to already be right".
    let before = rig.world.resource::<Dots>().0[&SESSION].prev_offset;
    assert_eq!(before, LatticePos::ORIGIN);
    assert_ne!(
        pose.sanitized().pos.cell(),
        LatticePos::ORIGIN.cell(),
        "the arrival must be somewhere other than the origin, or this proves nothing"
    );

    let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), pose)]);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(
        dot.prev_offset, dot.pose.pos,
        "the arriving dot's swept prior must BE the arrival point"
    );
}

#[test]
fn a_crossing_to_an_adopted_dot_stores_the_pose_stays_ghost_then_promote_flips_owned() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]); // adopting dot
    let _ = rig.tick(vec![adopted_head(Fence(2))]); // flip → granted, still a Ghost (not simulating)
    let pose = crossing_pose();

    // 1d.5b.3b: the crossing STORES the pose but leaves the dot a GHOST — the Ghost→Owned promote
    // RELOCATED to on_saga_promote (strict demote-before-promote). No autonomous flip here.
    let sent = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), pose)]);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(dot.pose, pose.sanitized(), "the crossed pose stored");
    assert!(
        !dot.authority.simulates(),
        "the dot STAYS Ghost after the crossing (the autonomous promote is gone)"
    );
    assert_eq!(rig.world.resource::<StubStats>().crossings_applied, 1);
    assert!(acked(&sent, TransferId(7)), "the crossing step is acked");

    // The saga Promote flips it Ghost→Owned (pose-before-promote satisfied — the crossing landed).
    let _ = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].authority,
        Authority::Owned { fence: Fence(2) },
        "the Promote flips the dest Ghost→Owned at the recorded CAS fence"
    );

    // A crossing redelivery AFTER the flip still matches `crossing_target` (now Owned, still
    // granted+non-departing) → the journal returns `AlreadyApplied`: re-ack WITHOUT re-applying,
    // NOT re-buffered (no strand).
    let buffered_before = rig.world.resource::<StubStats>().crossings_buffered;
    let sent = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), pose)]);
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_applied,
        1,
        "no re-apply on a post-flip redelivery"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_buffered,
        buffered_before,
        "a post-flip redelivery re-acks via the journal — it is NOT re-buffered (no strand)"
    );
    assert!(
        acked(&sent, TransferId(7)),
        "a post-flip redelivery still re-acks"
    );
}

#[test]
fn a_crossing_before_adopt_is_buffered_then_applied_on_the_grant_flip() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]); // adopting dot, NOT granted
    let pose = crossing_pose();

    // The crossing arrives BEFORE the adopt flip → BUFFERED (no ack, dot unchanged).
    let sent = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), pose)]);
    assert!(
        !acked(&sent, TransferId(7)),
        "a buffered crossing is not acked yet"
    );
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(
        fm(dot.pose.pos),
        DVec3::ZERO,
        "buffered, not applied (still adopting)"
    );
    assert_eq!(rig.world.resource::<StubStats>().crossings_buffered, 1);
    assert_eq!(rig.world.resource::<StubStats>().crossings_applied, 0);

    // A redelivery WHILE STILL BUFFERED (the saga re-emits at-least-once) overwrites the same
    // key and must NOT inflate the buffered count (audit F-3 — the counter is per-crossing, not
    // per-redelivery).
    let sent = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), pose)]);
    assert!(
        !acked(&sent, TransferId(7)),
        "still buffered, still not acked"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_buffered,
        1,
        "a still-buffered redelivery does not inflate the buffered count"
    );

    // The adopt grant-flip DRAINS the buffer → applies the pose + acks, but the dot STAYS Ghost
    // (1d.5b.3b — the drained path shares `apply_crossing`, whose autonomous promote is gone).
    let sent = rig.tick(vec![adopted_head(Fence(2))]);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(
        dot.pose,
        pose.sanitized(),
        "the buffered crossing applied on the flip"
    );
    assert!(
        !dot.authority.simulates(),
        "the drained crossing leaves the dot a Ghost (the promote relocated to on_saga_promote)"
    );
    assert_eq!(rig.world.resource::<StubStats>().crossings_applied, 1);
    assert!(acked(&sent, TransferId(7)), "the drained crossing is acked");

    // The saga Promote then flips it Ghost→Owned (the buffered-drain path also satisfies
    // pose-before-promote — the crossing journaled on the drain).
    let _ = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].authority,
        Authority::Owned { fence: Fence(2) },
        "the Promote flips Owned on the buffered-drain path"
    );

    // A redelivery AFTER the drained crossing was journaled hits the IMMEDIATE path (now Owned,
    // still matched by `crossing_target`) and the journal dedups it across the boundary — re-ack
    // only, NO second apply, NO re-buffer.
    let buffered_before = rig.world.resource::<StubStats>().crossings_buffered;
    let sent = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), pose)]);
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_applied,
        1,
        "a redelivery after a DRAINED crossing does not re-apply (journal spans the boundary)"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_buffered,
        buffered_before,
        "a post-drain redelivery re-acks via the journal — it is NOT re-buffered",
    );
    assert!(
        acked(&sent, TransferId(7)),
        "the post-drain redelivery still re-acks"
    );
}

#[test]
fn a_stale_fence_crossing_is_dropped() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]); // adopted at Fence(2)
    // A crossing at Fence(1) is BELOW the dot's recorded authority fence → stale (fence rule 1).
    let sent = rig.tick(vec![crossing_msg(TransferId(7), Fence(1), crossing_pose())]);
    assert_eq!(rig.world.resource::<StubStats>().crossings_stale, 1);
    assert_eq!(rig.world.resource::<StubStats>().crossings_applied, 0);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(
        fm(dot.pose.pos),
        DVec3::ZERO,
        "a stale crossing does not move the dot"
    );
    assert!(
        !acked(&sent, TransferId(7)),
        "a stale crossing is not acked"
    );
}

#[test]
fn a_non_crossing_transfer_payload_is_a_counted_noop() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]);
    // An InitialSpawn payload is not a 1d.1 dest concern → counted no-op (no apply, no ack).
    let env = InterShardFlow::Transfer(TransferEnvelope {
        transfer_id: TransferId(7),
        universe_epoch: vd_core::EpochId(1),
        schema_version: vd_wire::intershard::TRANSFER_SCHEMA_VERSION,
        fence: Fence(2),
        step_id: STUB_CROSSING_STEP,
        class: vd_core::entity_kind::DurabilityClass::Durable,
        payload: TransitionPayload::InitialSpawn {
            entity: SUBJECT,
            to_realm: TO_REALM,
            pose: crossing_pose(),
            state: vec![],
        },
    });
    let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &env)]);
    assert_eq!(rig.world.resource::<StubStats>().crossings_unhandled, 1);
    assert_eq!(rig.world.resource::<StubStats>().crossings_applied, 0);
    assert!(
        !acked(&sent, TransferId(7)),
        "an unhandled payload is not acked"
    );
}

#[test]
fn a_stale_epoch_crossing_is_refused() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]);
    // A crossing minted under epoch 2 while this shard's clock epoch is 1 (the rig default) — a
    // delayed redelivery across a re-genesis, or a clock-desync bug. Refused at the ingress
    // BEFORE the payload match and BEFORE find-dot: counted, never applied, never buffered,
    // never acked (transfer_protocol §3.3 fail-safe — no entity placed at a stale celestial
    // position). This exercises the mismatch arm of the epoch gate; every other crossing test
    // exercises the match arm (rig clock epoch == envelope epoch == EpochId(1)).
    let env = InterShardFlow::Transfer(TransferEnvelope {
        transfer_id: TransferId(7),
        universe_epoch: vd_core::EpochId(2),
        schema_version: vd_wire::intershard::TRANSFER_SCHEMA_VERSION,
        fence: Fence(2),
        step_id: STUB_CROSSING_STEP,
        class: vd_core::entity_kind::DurabilityClass::Durable,
        payload: TransitionPayload::StubCrossing {
            entity: SUBJECT,
            from_realm: FROM_REALM,
            to_realm: TO_REALM,
            pose: crossing_pose(),
            state: vec![],
        },
    });
    let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &env)]);
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_epoch_mismatch,
        1
    );
    assert_eq!(rig.world.resource::<StubStats>().crossings_applied, 0);
    assert_eq!(rig.world.resource::<StubStats>().crossings_buffered, 0);
    assert_eq!(
        fm(rig.world.resource::<Dots>().0[&SESSION].pose.pos),
        DVec3::ZERO,
        "a stale-epoch crossing does not move the dot"
    );
    assert!(
        !acked(&sent, TransferId(7)),
        "a stale-epoch crossing is not acked"
    );
}

#[test]
fn a_stale_epoch_re_home_is_refused() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let session = SessionId(SUBJECT.0); // the deterministic clientless re-home session key
    // A re-home minted under epoch 2 while this shard's clock epoch is 1 (the rig default) — a delayed
    // redelivery across a re-genesis. Refused at the ingress BEFORE journal/adopt/ack: counted, no dot
    // created, no PromoteAck. The §3.3 fail-safe, UNIFORM with the crossing arm
    // (a_stale_epoch_crossing_is_refused). Exercises the mismatch arm of the re-home epoch gate; every
    // other re-home test exercises the match arm (rig clock epoch == cmd epoch == EpochId(1)).
    let env = InterShardFlow::ReHome(ReHomeCmd {
        transfer: TransferId(7),
        universe_epoch: vd_core::EpochId(2),
        subject: DirectoryKey::Entity(SUBJECT),
        new_fence: Fence(2),
        step_id: RE_HOME_STEP,
        state: ReHomeState::PoseOnly(crossing_pose()),
        source: NodeId(99),
    });
    let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &env)]);
    assert_eq!(rig.world.resource::<StubStats>().re_home_epoch_mismatch, 1);
    assert_eq!(rig.world.resource::<StubStats>().re_home_adopted, 0);
    assert!(
        !rig.world.resource::<Dots>().0.contains_key(&session),
        "a stale-epoch re-home creates no dot"
    );
    assert!(
        !saga_ack_to_orch(
            &sent,
            TransferControlAck::PromoteAck {
                transfer: TransferId(7)
            }
        ),
        "a stale-epoch re-home is not acked"
    );
}

#[test]
fn a_duplicate_entity_grant_head_is_an_idempotent_noop() {
    // GrantFlip::NoOp: a SECOND grant head for an already-granted dot flips nothing.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]); // first flip → granted
    let before = rig.world.resource::<Dots>().0[&SESSION];
    let sent = rig.tick(vec![adopted_head(Fence(2))]); // duplicate → NoOp
    let after = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(after, before, "a duplicate grant head changes nothing");
    assert!(!acked(&sent, TransferId(7)), "no ack on a duplicate grant");
}

#[test]
fn producer_less_reliable_flows_push_with_the_retained_marker() {
    // R-6d §7 CONFORMANCE: the three `FlowDurabilityClass::ProducerLessReliable` flows — the
    // source-shard `TransientBatch` emit (D-6 #1), the band-exit `Ghost::Despawn`, and slice F's
    // `Ghost::SpawnV2` take-over proof — have NO scan_deadlines re-driver, so their push MUST
    // carry `Durability::Retained` (the R-6d durable outbox mirrors + replays them across a
    // source crash). The `send`/`push_flow` default is Ephemeral, so THIS test is the guarantee
    // that these sites opted into durability; a future producer-less flow (compile-forced-
    // classified by `durability_class`, R-6d2a) whose author forgets the marker trips this.
    use vd_wire::intershard::FlowDurabilityClass;
    let producer_less = |sent: &[(NodeId, MsgClass, InterShardFlow, Durability)]| {
        sent.iter()
            .find(|(_, _, f, _)| f.durability_class() == FlowDurabilityClass::ProducerLessReliable)
            .map(|(_, _, _, dur)| *dur)
    };

    // --- (a) TransientBatch (the emit_transient_batch producer-less one-shot) ---
    let mut rig = Rig::new();
    rig.grant_realm();
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        EntityId::pack(EntityKind::Debris, 1, 7, 1),
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: TransientStatus::Crossing {
                dest: DEST_NODE,
                to_realm: RealmId::System(8),
                dst_realm_fence: Fence(2),
                batch: TransferId(0xB3),
                to_parent: None,
            },
            prev_offset: LatticePos::ORIGIN,
        },
    );
    assert_eq!(
        producer_less(&rig.tick_raw(vec![])),
        Some(Durability::Retained),
        "the TransientBatch emit MUST push Durability::Retained (no re-driver, D-6 #1)"
    );

    // --- (b) band-exit Ghost::Despawn (reuses the_dest_feed_despawns...'s setup) ---
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]);
    let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]);
    // (c, folded into (b)'s rig) the PROMOTE tick itself pushes the SpawnV2 proof — read it
    // off the raw outbox before stepping further.
    let promote_sends = rig.tick_raw(vec![promote_msg(Fence(2), NodeId(99))]);
    assert_eq!(
        producer_less(&promote_sends),
        Some(Durability::Retained),
        "the take-over proof Ghost::SpawnV2 MUST push Durability::Retained (no re-driver)"
    );
    rig.set_local_tick(6);
    let _ = rig.tick(vec![]); // in-band: the sweep streams nothing, keeps the registration
    let exit_pos = fm(crossing_pose().pos) + DVec3::new(3.0, 0.0, 0.0);
    rig.world
        .resource_mut::<Dots>()
        .0
        .get_mut(&SESSION)
        .expect("the owned dot")
        .pose
        .pos = LatticePos::from_metres(exit_pos, vd_core::pose::Tier::Fine);
    assert_eq!(
        producer_less(&rig.tick_raw(vec![])),
        Some(Durability::Retained),
        "the band-exit Ghost::Despawn MUST push Durability::Retained (no re-driver)"
    );
}

/// Slice F's TTL BACKSTOP: a Source hold that EXPIRES (the take-over proof never came — a
/// logout mid-crossing kills the dest before it can send one) still evicts the bystanders'
/// figures: the retained ghost stops emitting the tick the hold dies, and the removal fans at
/// the shard's exact tick. The lane's loss-mode ends in a vanish, never a frozen phantom.
#[test]
fn an_expired_hold_evicts_the_bystanders_like_the_proof_that_never_came() {
    const BYSTANDER: SessionId = SessionId(0xBB);
    let mut rig = Rig::new();
    let entity = make_retained_ghost(&mut rig, Fence(2)); // arms the budget (1_000 ticks)
    insert_owned_dot(&mut rig, BYSTANDER, player(9), DVec3::new(1.0, 0.0, 0.0));
    let sent = rig.tick(vec![]);
    assert!(
        entity_rows_to_gateway(&sent).contains(&entity),
        "the fill emits while the hold lives"
    );
    // Jump past the budget: the prune drops the hold and the eviction fans in the same pass.
    let opened = rig.world.resource::<HandoffHolds>().0[&(entity, HoldRole::Source)].opened_at;
    rig.set_local_tick(opened.0 + 1_001);
    let sent = rig.tick(vec![]);
    assert_eq!(
        entity_removals(&sent),
        vec![(GATEWAY, entity, UniverseTick(100))],
        "the expiry fans the eviction — the TTL backstop of the take-over proof"
    );
    assert!(
        !entity_rows_to_gateway(&sent).contains(&entity),
        "…and the ghost stopped emitting the same tick"
    );

    // The LEASELESS expiry (the suppression arm at the prune): same setup, lease gone before
    // the budget runs out — the removal is withheld, counted, never silent.
    let mut rig = Rig::new();
    let entity = make_retained_ghost(&mut rig, Fence(2));
    insert_owned_dot(&mut rig, BYSTANDER, player(9), DVec3::new(1.0, 0.0, 0.0));
    rig.world.resource_mut::<RealmAuthority>().0 = None;
    let opened = rig.world.resource::<HandoffHolds>().0[&(entity, HoldRole::Source)].opened_at;
    rig.set_local_tick(opened.0 + 1_001);
    let sent = rig.tick(vec![]);
    assert!(
        entity_removals(&sent).is_empty(),
        "an unowned shard is silent"
    );
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .entity_removals_suppressed_no_lease,
        1,
        "…but the suppression is counted"
    );

    // A DEST-role hold expiring fans nothing (the false arm): only a SOURCE hold guards a
    // retained ghost's emit.
    let mut rig = Rig::new();
    rig.grant_realm();
    rig.world
        .resource_mut::<StubConfig>()
        .handoff_hold_ttl_ticks = 4;
    let stranger = player(7);
    open_hold(
        &mut rig.world.resource_mut::<HandoffHolds>(),
        (stranger, HoldRole::Dest),
        Fence(2),
        TickId(1),
        4,
    );
    rig.set_local_tick(20);
    let sent = rig.tick(vec![]);
    assert!(
        entity_removals(&sent).is_empty(),
        "an expired Dest hold is bookkeeping, never an eviction"
    );
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

#[test]
fn the_living_up_lanes_dispatch_and_every_dead_scenery_lane_counts_undecodable() {
    // The dispatch arms land in their stores through the REAL inbound path — the receive
    // helpers are covered directly elsewhere; this pins the wiring, and it pins the OTHER
    // half too: after the Slice-C2 deletion the ONLY things a realm still receives about the
    // world are the SL7 occupancy bit and a child's SEALED self-statements. The three old
    // scenery lanes are tombstoned, so a frame of any of them must fall through to
    // `undecodable` ON ITS OWN CARRIER — never a silent apply, never a panic.
    let mut rig = Rig::new();
    rig.grant_realm();
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), own_region(), child_region()]);
    // The up-lanes admit only the directory-attested child node (findings 0/43) — seed the map
    // the realm-Head reply arm fills.
    rig.world
        .resource_mut::<ChildRealmNodes>()
        .0
        .insert(OTHER_REALM, NodeId(70));
    let child = with_child_coord(&mut rig, OTHER_REALM);
    // LIVING: the SL7 occupancy bit (SignalDelta) and the Q2 relay (reliable Saga).
    let bit = InterShardFlow::ChildLive(vd_wire::intershard::ChildLive {
        child: child.clone(),
        fence: Fence(1),
        at: UniverseTick(5),
    });
    let relay = InterShardFlow::WindowRelay(vd_wire::intershard::WindowRelay {
        child: child.clone(),
        realm_fence: Fence(1),
        own: vd_wire::session_flow::seal_relay_statements(&[]),
        interior: Vec::new(),
    });
    // ★TOMBSTONED (minors 17/19): the up-observation ship, the interim shape lane and the
    // down-cascade all rode SignalDelta; the down-reflect rode the reliable Saga carrier.
    let rows = InterShardFlow::RealmObservation(vd_wire::intershard::RealmObservation {
        child: child.clone(),
        realm_snapshot_bytes: Vec::new(),
    });
    let shapes =
        InterShardFlow::RealmShapeObservation(vd_wire::intershard::RealmShapeObservation {
            child: child.clone(),
            shapes: vec![render_shape(RealmId::Planet(52))],
        });
    let cascade = InterShardFlow::RealmCascade(vd_wire::intershard::RealmCascade {
        child: rig.world.resource::<StubConfig>().own_coord.clone(),
        realm_snapshot_bytes: Vec::new(),
    });
    let scene_set = InterShardFlow::ChildSceneSet(vd_wire::intershard::ChildSceneSet {
        child: rig.world.resource::<StubConfig>().own_coord.clone(),
        realms: vec![render_shape(RealmId::Planet(53))],
    });
    let signal_msg = |flow: &InterShardFlow| Inbound::Wire {
        from: NodeId(70),
        class: MsgClass::SignalDelta,
        bytes: crate::io::bytes(postcard::to_allocvec(flow).expect("encode")),
    };
    let saga_msg = |flow: &InterShardFlow| Inbound::Wire {
        from: NodeId(70),
        class: MsgClass::Saga,
        bytes: crate::io::bytes(postcard::to_allocvec(flow).expect("encode")),
    };
    let _ = rig.tick(vec![
        signal_msg(&bit),
        signal_msg(&rows),
        signal_msg(&shapes),
        signal_msg(&cascade),
        saga_msg(&relay),
        saga_msg(&scene_set),
    ]);
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(stats.child_live_received, 1, "the bit arm dispatched");
    assert_eq!(
        stats.window_relays_received, 1,
        "the relay arm dispatched on the Saga carrier"
    );
    assert_eq!(
        stats.undecodable, 4,
        "all FOUR tombstoned scenery frames counted undecodable on their own carriers, \
         and none of them reached a store"
    );
    assert_eq!(
        rig.world.resource::<RelayHeld>().0.len(),
        1,
        "and the sealed batch is held"
    );
}

/// A TOMBSTONED entity-lane frame (either leg) arriving on the SignalDelta carrier is counted
/// undecodable, never applied and never a panic — the discriminants are reserved forever, their
/// meaning is gone (Step 5 slice E). Measured per carrier: both legs rode SignalDelta, whose
/// closed fall-through is the counting arm (the slice D lesson — never assert a tombstone's
/// receiver behaviour without driving its actual dispatch).
#[test]
fn a_tombstoned_entity_lane_frame_is_counted_undecodable_on_both_legs() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let relay = vd_wire::intershard::EntityRelay {
        realm: StubConfig::root_coord(OWN_REALM),
        frame: config().frame,
        frame_id: 1,
        universe_tick: UniverseTick(1),
        entities: vec![],
    };
    let _ = rig.tick(vec![Inbound::Wire {
        from: NodeId(9),
        class: MsgClass::SignalDelta,
        bytes: crate::io::bytes(
            postcard::to_allocvec(&InterShardFlow::EntityInterest(relay.clone())).expect("encode"),
        ),
    }]);
    assert_eq!(rig.world.resource::<StubStats>().undecodable, 1);
    let _ = rig.tick(vec![Inbound::Wire {
        from: NodeId(9),
        class: MsgClass::SignalDelta,
        bytes: crate::io::bytes(
            postcard::to_allocvec(&InterShardFlow::EntityCascade(relay)).expect("encode"),
        ),
    }]);
    assert_eq!(rig.world.resource::<StubStats>().undecodable, 2);
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

/// A LOOK-LESS CHILD STATES NOTHING (the bound/look split's structural half): the ambient
/// Universe/Galaxy rows carry no look, so a parent authoring its markers must SKIP them —
/// a marker sized by a containment bound would draw an authority promise as an object,
/// which the owner's law forbids. Measured as an absence: the look-bearing sibling is
/// stated, the look-less one is not, and the realm's own look-less arm states nothing
/// either.
#[test]
fn a_look_less_realm_states_no_body_and_its_look_less_children_get_no_marker() {
    let mut lit = region(OTHER_REALM, Some(OWN_REALM), DVec3::ZERO, 1000.0);
    lit.look = Some(Boundary::Shell { r: 250.0 });
    let mut dark = region(RealmId::Station(99), Some(OWN_REALM), DVec3::ZERO, 1000.0);
    dark.look = None;
    let mut own = own_region();
    own.look = None;
    let regions = RealmRegions::new(vec![root_region(), own, lit, dark]);
    let luma = BTreeMap::new();
    let stated: Vec<RealmId> = current_bodies(&config(), &regions, &luma)
        .into_iter()
        .map(|(realm, _)| realm)
        .collect();
    assert_eq!(stated, vec![OTHER_REALM]);
}

#[test]
fn ancestor_chain_names_self_and_every_ancestor_and_nothing_else() {
    // THE DERIVED HYSTERESIS PRIOR (task #177): being authoritatively in a realm makes you a member of
    // that realm AND of every realm containing it — and of NOTHING else. A sibling must never be
    // implied, or a subject would arrive already "inside" a realm it has never been in.
    //
    // AND THE CEILING IS GONE (SL9, 2026-08-24). This half used to assert the OPPOSITE: with 70
    // children the 70th was indexed past the `u64`'s 64 bits and its own bit was DROPPED, so its chain
    // came back as the root alone. That silent truncation is what capped a parent's child count. The
    // chain names REALMS now, so the seventieth child names itself exactly like the first — and a
    // galaxy's hundred-and-fifty-thousandth star system does too.
    let mut wide: Vec<RealmRegion> = vec![root_region()];
    for i in 0..70u64 {
        wide.push(region(
            RealmId::Station(1000 + i),
            Some(ROOT_REALM),
            DVec3::new(1.0e7 + 1.0e5 * i as f64, 0.0, 0.0),
            10.0,
        ));
    }
    let wide_rr = RealmRegions::new(wide);
    assert_eq!(
        *wide_rr.ancestor_chain_for(RealmId::Station(1069)),
        BTreeSet::from([ROOT_REALM, RealmId::Station(1069)]),
        "the seventieth child names ITSELF and its root — no index, so no width to fall off"
    );
    let regions = vec![root_region(), own_region(), child_region()];
    let rr = RealmRegions::new(regions);

    assert_eq!(
        *rr.ancestor_chain_for(ROOT_REALM),
        BTreeSet::from([ROOT_REALM])
    );
    assert_eq!(
        *rr.ancestor_chain_for(OWN_REALM),
        BTreeSet::from([ROOT_REALM, OWN_REALM])
    );
    assert_eq!(
        *rr.ancestor_chain_for(OTHER_REALM),
        BTreeSet::from([ROOT_REALM, OWN_REALM, OTHER_REALM])
    );
}

#[test]
fn ancestor_chain_excludes_a_sibling_branch() {
    // The anti-vacuity twin of the test above: with TWO children under one parent, each child's chain
    // must contain itself + the chain up, and must NOT contain the other child. A chain built by "every
    // region at or below my depth" (a plausible wrong implementation) would fail exactly here.
    let sibling = region(RealmId::Planet(43), Some(OWN_REALM), DVec3::ZERO, 1000.0);
    let rr = RealmRegions::new(vec![root_region(), own_region(), child_region(), sibling]);

    assert_eq!(
        *rr.ancestor_chain_for(OTHER_REALM),
        BTreeSet::from([ROOT_REALM, OWN_REALM, OTHER_REALM])
    );
    assert_eq!(
        *rr.ancestor_chain_for(RealmId::Planet(43)),
        BTreeSet::from([ROOT_REALM, OWN_REALM, RealmId::Planet(43)])
    );
}

#[test]
fn ancestor_chain_is_empty_for_a_realm_this_shard_does_not_host() {
    // The safe-degrade arm: a pose naming an unhosted realm resolves to NO realms, which reduces to
    // today's blank-prior behaviour rather than inventing membership. Callers treat this as a loud
    // condition, not a normal one — an unhosted realm in a pose means a rebind degraded upstream.
    let rr = RealmRegions::new(vec![root_region(), own_region()]);
    assert_eq!(
        *rr.ancestor_chain_for(RealmId::Planet(9999)),
        BTreeSet::new()
    );
    // And on a forest with no regions at all (the inert default), every lookup is empty.
    assert_eq!(
        *RealmRegions::new(vec![]).ancestor_chain_for(OWN_REALM),
        BTreeSet::new()
    );
}

#[test]
fn ancestor_chain_terminates_on_a_dangling_parent() {
    // `region_depth` stops at a dangling parent rather than hanging; the chain walk mirrors it exactly,
    // so the two caches can never disagree about the forest's shape. The boot guard rejects such a
    // forest — this only proves the walk is safe if one ever slips through.
    let orphan = region(
        RealmId::Planet(77),
        Some(RealmId::Planet(404)),
        DVec3::ZERO,
        10.0,
    );
    let rr = RealmRegions::new(vec![root_region(), orphan]);
    assert_eq!(
        *rr.ancestor_chain_for(RealmId::Planet(77)),
        BTreeSet::from([RealmId::Planet(77)]),
        "the walk stops at the dangling parent, naming only what it reached"
    );
}

#[test]
fn the_child_index_decides_exactly_what_the_full_scan_decides() {
    // THE DIFFERENTIAL PROOF for SL9's second half. The fold now asks a LOOKUP which children are worth
    // evaluating instead of walking all of them, and the only thing that makes that safe is that the
    // two decide identically. This drives a spread of subject positions through the SAME shard, once
    // with the index built (the shipped path) and once with it empty (the full scan it replaced), and
    // requires the emitted crossing destinations to match position for position.
    //
    // Anti-vacuity: the scan half is not a re-run of the same code. An EMPTY index answers for nothing,
    // so `worth_asking` returns true for every region and every child is evaluated — which is exactly
    // the pre-slice behaviour, reached through the shipped code.
    let forest = || {
        let sibling = region(
            RealmId::Planet(43),
            Some(OWN_REALM),
            DVec3::new(4000.0, 0.0, 0.0),
            500.0,
        );
        vec![root_region(), own_region(), child_region(), sibling]
    };
    // The subject positions: inside the child, inside the sibling, in the gap between them, well
    // outside everything, and exactly on each boundary — the places a verdict can differ.
    let probes = [
        DVec3::ZERO,
        DVec3::new(999.0, 0.0, 0.0),
        DVec3::new(1001.0, 0.0, 0.0),
        DVec3::new(2500.0, 0.0, 0.0),
        DVec3::new(3600.0, 0.0, 0.0),
        DVec3::new(4000.0, 0.0, 0.0),
        DVec3::new(4499.0, 0.0, 0.0),
        DVec3::new(4501.0, 0.0, 0.0),
        DVec3::new(50_000.0, 0.0, 0.0),
        DVec3::new(0.0, 900.0, 0.0),
        DVec3::new(0.0, 0.0, 1100.0),
    ];
    let run = |indexed: bool, at: DVec3| -> Vec<RealmId> {
        let mut rig = Rig::new();
        // WITHOUT THIS THE FOLD NEVER RUNS. `evaluate_realm_boundaries` gates on the realm lease, so an
        // ungranted rig returns before the scan and BOTH halves below come back empty — a comparison
        // that could not have failed. Caught by the coverage gate reporting the skip arm as never
        // taken, which is exactly what a vacuous differential looks like from the outside.
        rig.grant_realm();
        let regions = RealmRegions::new(forest());
        *rig.world.resource_mut::<RealmRegions>() = if indexed {
            regions.with_own_realm(OWN_REALM)
        } else {
            regions
        };
        let entity = EntityId(0x5100_0001);
        insert_owned_dot_framed(&mut rig, TRIG_SESSION, entity, frame_of(OWN_REALM), at);
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..6 {
            rig.set_local_tick(t);
            all.extend(rig.tick(vec![]));
        }
        crossing_requests(&all).iter().map(|r| r.to_realm).collect()
    };
    let mut crossings_seen = 0usize;
    for at in probes {
        let indexed = run(true, at);
        crossings_seen += indexed.len();
        assert_eq!(
            indexed,
            run(false, at),
            "the lookup and the full scan disagreed for a subject at {at:?}"
        );
    }
    // NOT VACUOUS: at least one probe must actually have produced a crossing, or the loop above is
    // comparing empty against empty. This assert is here because it already caught exactly that — the
    // rig's realm lease was ungranted, the fold returned before the scan, and every comparison passed
    // while proving nothing.
    assert!(
        crossings_seen > 0,
        "no probe produced a crossing — the differential proved nothing"
    );
    // AND THE INDEX IS NOT VACUOUS: it must actually hold the two static children, or the loop above
    // would be comparing the scan with itself.
    let built = RealmRegions::new(forest()).with_own_realm(OWN_REALM);
    assert_eq!(built.child_index().indexed_len(), 2);
    assert!(built.child_index().answers_for(OTHER_REALM));
    assert!(!built.child_index().answers_for(ROOT_REALM));
    // AND THE SKIP ACTUALLY FIRES — the measurement that stops the loop above from proving nothing. At
    // the far probe the lookup names NEITHER child while answering for both, which is precisely the
    // state in which the fold declines to evaluate them. If the lookup ever answered with everything,
    // the differential loop would still pass and would mean nothing; this line fails instead.
    assert_eq!(
        built.child_index().candidates(
            LatticePos::from_metres(DVec3::new(50_000.0, 0.0, 0.0), vd_core::pose::Tier::Fine),
            vd_core::pose::Tier::Fine,
        ),
        &[] as &[RealmId],
        "a subject far outside every child must be told about none of them"
    );
    // And the near probe NEVER OMITS the child that actually holds the point. It may name more — this
    // forest's two children are comparable in size to the derived cell, so they share one, and the
    // answer here is both. That is the stated degradation, not a defect: the lookup is a SUPERSET and
    // its only hard duty is to never drop the holder. Asserting a singleton here would be asserting a
    // grid-tuning detail, and it would fail the day the band widens for a reason unrelated to this.
    assert!(
        built
            .child_index()
            .candidates(
                LatticePos::from_metres(DVec3::new(4000.0, 0.0, 0.0), vd_core::pose::Tier::Fine),
                vd_core::pose::Tier::Fine,
            )
            .contains(&RealmId::Planet(43)),
        "the lookup dropped the child that holds the point"
    );
}

#[test]
fn a_subject_this_shard_cannot_place_in_its_own_frame_still_gets_the_full_scan() {
    // THE SAFE-DEGRADE ARM of the candidate lookup. The index lives in this shard's own frame, so a
    // subject whose pose names a frame the shard's placement book cannot reach — an ancestor's frame,
    // here the ambient root's — yields NO candidates. That must NOT be read as "near nothing": the
    // fold's skip only ever drops a realm the index positively answers for, and with an unplaceable
    // point every child falls back to being evaluated. Proven the only honest way — by comparing with
    // the same subject on a shard whose index was never built.
    let forest = || vec![root_region(), own_region(), child_region()];
    let run = |indexed: bool| -> Vec<RealmId> {
        let mut rig = Rig::new();
        rig.grant_realm(); // the fold is lease-gated — see the sibling test's note
        let regions = RealmRegions::new(forest());
        *rig.world.resource_mut::<RealmRegions>() = if indexed {
            regions.with_own_realm(OWN_REALM)
        } else {
            regions
        };
        let entity = EntityId(0x5100_0002);
        insert_owned_dot_framed(
            &mut rig,
            TRIG_SESSION,
            entity,
            frame_of(ROOT_REALM),
            DVec3::ZERO,
        );
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..6 {
            rig.set_local_tick(t);
            all.extend(rig.tick(vec![]));
        }
        crossing_requests(&all).iter().map(|r| r.to_realm).collect()
    };
    assert_eq!(
        run(true),
        run(false),
        "an unplaceable subject must be decided exactly as the full scan decides it"
    );
}

#[test]
fn worth_asking_evaluates_everything_the_lookup_cannot_speak_for() {
    // THE FOUR ARMS OF THE SKIP DECISION, each on its own (HR5), because this is the one function that
    // can silently lose a crossing: every arm that returns TRUE is a region the fold still evaluates,
    // and the single FALSE at the end is the only place work is ever dropped.
    use super::worth_asking;
    use vd_core::child_index::{ChildIndex, IndexedChild};

    let indexed = RealmId::Planet(1);
    let unindexed = RealmId::Planet(2);
    let ix = ChildIndex::build(
        &[IndexedChild {
            realm: indexed,
            centre: LatticePos::from_metres(DVec3::ZERO, vd_core::pose::Tier::Fine),
            radius_m: 10.0,
        }],
        vd_core::pose::Tier::Fine,
    );
    let empty_chain = BTreeSet::new();
    let no_memory = RegionMembership::default();

    // ARM 0 — THE LOOKUP ABSTAINED (slice S5). A segment wider than the index can enumerate answers
    // nothing at all, and that is IGNORANCE, not a miss. Evaluate everything, which is conservative
    // and therefore always correct. Without this term the swept verdict would silently stop being
    // asked for exactly the fast subjects it exists for.
    assert!(worth_asking(
        false,
        &ix,
        &[],
        &no_memory,
        &empty_chain,
        indexed
    ));

    // ARM 1 — NOT INDEXED. An ancestor, this realm itself, or anything that moves. The index has no
    // opinion, so the fold must ask. This is the arm that keeps the whole thing conservative.
    assert!(worth_asking(
        true,
        &ix,
        &[],
        &no_memory,
        &empty_chain,
        unindexed
    ));

    // ARM 2 — THE LOOKUP NAMED IT.
    assert!(worth_asking(
        true,
        &ix,
        &[indexed],
        &no_memory,
        &empty_chain,
        indexed
    ));

    // ARM 3 — ALREADY A MEMBER. Hysteresis: a subject inside a region must be re-asked every tick to
    // decide RELEASE, however far the lookup now thinks it is. Skipping here would strand it inside.
    let mut remembered = RegionMembership::default();
    remembered.set(indexed, true);
    assert!(worth_asking(
        true,
        &ix,
        &[],
        &remembered,
        &empty_chain,
        indexed
    ));

    // ARM 4 — ON THE DERIVED CHAIN. Same reason, for the membership nobody stored.
    let chain = BTreeSet::from([indexed]);
    assert!(worth_asking(true, &ix, &[], &no_memory, &chain, indexed));

    // THE ONLY SKIP: indexed, not named, not remembered, not on the chain.
    assert!(!worth_asking(
        true,
        &ix,
        &[],
        &no_memory,
        &empty_chain,
        indexed
    ));
}

#[test]
fn a_moving_child_is_left_out_of_the_index_and_stays_a_candidate() {
    // SL4's clause, made structural: a child that MOVES has no fixed centre to index, so it is omitted
    // and evaluated unconditionally — conservative, and the index never learns that anything moves. The
    // order of the two builders must not matter either, which is why both are asserted.
    // An opaque motion closure — its CONTENTS are irrelevant here and unreadable by design (SL4). The
    // index reads only the KEYS of the moving roster: whether to recompute, never what anyone reads.
    let motion = MotionFn(std::sync::Arc::new(|_| FramePlacement::identity()));
    let moving: BTreeMap<RealmId, MotionFn> = BTreeMap::from([(OTHER_REALM, motion)]);
    let forest = vec![root_region(), own_region(), child_region()];
    let a = RealmRegions::new(forest.clone())
        .with_own_realm(OWN_REALM)
        .with_moving_children(moving.clone());
    let b = RealmRegions::new(forest)
        .with_moving_children(moving)
        .with_own_realm(OWN_REALM);
    for rr in [&a, &b] {
        assert_eq!(rr.child_index().indexed_len(), 0);
        assert!(!rr.child_index().answers_for(OTHER_REALM));
    }
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

#[test]
fn author_book_places_the_anchor_a_child_and_refuses_a_parent() {
    // (a) EMPTY forest → the anchor defaults to the AMBIENT ROOT (the detector short-circuits on
    // `is_empty` before ever authoring, but the default must still be well-formed): the root frame
    // resolves to the identity via the anchor arm, and NO other frame has a row.
    //
    // ★ RE-BASED IN S9: that default is `UniverseSpace`, not `GalaxySpace`. It was a galaxy frame
    // because that was the nearest thing to "outermost" available while the universe had no frame of
    // its own — a stand-in whose meaning had to be remembered. It names the thing it always meant now.
    let empty = RealmRegions::new(vec![]);
    let ebook = empty.author_book(OWN_REALM, 20.0, UniverseTick(0));
    assert_eq!(
        ebook.of(FrameRef::UniverseSpace),
        Some(FramePlacement::identity()),
        "empty-forest anchor defaults to the ambient root ⇒ identity",
    );
    // …and the frame it used to default to is now just another unplaced frame, which is what makes
    // the line above a statement about the root rather than about `GalaxySpace` in particular.
    assert_eq!(ebook.of(FrameRef::GalaxySpace { galaxy_seed: 0 }), None);
    assert_eq!(
        ebook.of(frame_of(OWN_REALM)),
        None,
        "an unplaced frame in an empty forest resolves to None",
    );
    // (b) THE SHIPPED MODEL, stated as three separate facts rather than one sweeping one. This used to
    // assert that EVERY region — root, own and child alike — sat at the identity, which was the retired
    // model: it held only because every fixture realm was parked at the origin, and it is the same
    // assumption that made an arriving traveller find a neighbouring star system empty.
    let child_at = DVec3::new(300.0, -40.0, 7.5);
    let placed_child = region(OTHER_REALM, Some(OWN_REALM), child_at, 1000.0);
    let regions = RealmRegions::new(vec![root_region(), own_region(), placed_child]);
    for tick in [UniverseTick(0), UniverseTick(1_000_000)] {
        let book = regions.author_book(OWN_REALM, 20.0, tick);
        assert_eq!(book.at(), tick, "the instant is a property of the table");
        // The shard's OWN realm: the identity, always. It IS its own origin.
        assert_eq!(
            book.of(own_region().frame),
            Some(FramePlacement::identity()),
            "the shard's own realm is its own origin at tick {}",
            tick.0,
        );
        // A STATIC DIRECT CHILD: at the centre this shard authors for it — NOT the identity. Nothing
        // else green asserts this, which is exactly why registering children at the origin survived so
        // long unnoticed.
        assert_eq!(
            book.of(placed_child.frame)
                .map(|p| (p.origin_cell, p.origin)),
            Some((
                placed_child.center.in_parents_frame().cell(),
                placed_child.center.in_parents_frame().offset()
            )),
            "a static direct child rides the placement its parent authored, at tick {}",
            tick.0,
        );
        // …and that placement's VALUE is exactly the authored centre (normalized carry).
        assert_eq!(
            book.of(placed_child.frame).map(|p| fm(p.anchor())),
            Some(child_at),
        );
        // The PARENT: unknown, deliberately. Nobody has told this shard where its parent is, and under
        // the ground rule nobody ever will — so a conversion involving it must fail loudly rather than
        // quietly assume the identity and answer confidently from the wrong numbers.
        assert_eq!(
            book.of(root_region().frame),
            None,
            "a shard is never told where its parent is, at tick {}",
            tick.0,
        );
    }
}

#[test]
fn author_book_writes_a_registered_moving_childs_row_from_its_ephemeris() {
    // FA-2b, restated on the one writer: a region in the MOVING roster gets its row solved from its
    // `OrbitalElements` at the BOOK's instant; a region ABSENT from the roster stays static at its
    // stored centre (the byte-identity arm). Both arms write the SAME kind of row — downstream
    // cannot tell which ran (SL4). The moving child is `child_region` (`OTHER_REALM` = Planet 42).
    let elements = OrbitalElements {
        sma: 1.5e11,
        ecc: 0.1,
        inclination: 0.4,
        raan: 0.3,
        arg_periapsis: 0.9,
        mean_anomaly_epoch: 0.2,
        central_mass: 1.989e30,
    };
    let mut moving = BTreeMap::new();
    moving.insert(OTHER_REALM, elements);
    let regions = RealmRegions::new(vec![root_region(), own_region(), child_region()])
        .with_moving_children(kepler_motion_fns(moving));
    let tick_hz = 20.0;
    let tick = UniverseTick(1_000);
    let book = regions.author_book(OWN_REALM, tick_hz, tick);
    // The MOVING child's row: the ephemeris solved at the book's instant (the mover arm).
    let state = orbital_state(&elements, secs_since_epoch(tick.0, tick_hz));
    assert_eq!(
        book.of(frame_of(OTHER_REALM)),
        Some(FramePlacement::moving(state.position, state.velocity)),
        "a registered moving child's row is authored live from its orbit, not the static identity",
    );
    // A region ABSENT from the roster (`own`): the anchor identity — the byte-identity arm.
    assert_eq!(
        book.of(frame_of(OWN_REALM)),
        Some(FramePlacement::identity()),
        "a non-roster region stays at the identity placement (byte-identical to FA-1)",
    );
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

/// THE MOST VALUABLE TEST IN THIS ARC, and the reason it is planted before anything moves: it is what
/// stops a realm learning its own address again under some other name. At EVERY level of the story the
/// shard's authored book knows its own realm and its direct children and NOTHING ELSE — asking it to
/// place its own parent, or a sibling, is `None`, and asking `transfer_frame` to re-express a pose into
/// an ancestor's frame is a TYPED REFUSAL rather than a number.
///
/// The parent is not merely missing from the world: for the galaxy and the system it is a REGION the
/// shard holds and evaluates containment against. It is deliberately absent from the authored book
/// anyway, because containment is a question a realm answers about its own volume and placement is a
/// question only its parent can answer.
#[test]
fn no_shard_can_place_its_own_parent_or_a_sibling() {
    use vd_core::frame::{FrameError, transfer_frame};
    let story = Story::new();
    for (own, parent, sibling) in [
        // The galaxy: its parent is the ambient universe; the deepest realm in the world stands in
        // for "somebody else's realm" (the galaxy has no sibling — the universe holds one galaxy).
        (story.galaxy, story.universe, story.planet),
        (story.system, story.galaxy, story.sibling_system),
        (story.planet, story.system, story.sibling_planet),
    ] {
        let book = story.ctx(own);
        assert_eq!(
            book.of(story.frame(own)),
            Some(FramePlacement::identity()),
            "{own:?} IS its own origin",
        );
        assert_eq!(
            book.of(story.frame(parent)),
            None,
            "{own:?} is never told where its parent {parent:?} is",
        );
        assert_eq!(
            book.of(story.frame(sibling)),
            None,
            "{own:?} is never told where {sibling:?} is either",
        );
    }
    // A planet asking for its own galaxy-frame position must be a typed refusal, not a number.
    assert_eq!(
        transfer_frame(
            &story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M),
            story.frame(story.galaxy),
            &story.ctx(story.planet),
        )
        .expect_err("a planet must not be able to answer where it is in its galaxy"),
        FrameError::UnknownDestFrame,
    );
    // And a frame no region in this world carries is refused the same way — the refusal is a property
    // of "I was not told", not of "I recognised the name and declined".
    assert_eq!(
        transfer_frame(
            &story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M),
            FrameRef::GalaxySpace { galaxy_seed: 0 },
            &story.ctx(story.planet),
        )
        .expect_err("an unheard-of frame is a refusal, never a guess"),
        FrameError::UnknownDestFrame,
    );
}

/// GOING UP: three separate contexts, three separate additions, each made by the one party that holds
/// that number. The planet ships "3, in my frame"; the SYSTEM adds 145 → 148; the system ships "148,
/// in my frame"; the GALAXY adds 12031 → 12179. Nobody ever learns its own address on the way.
#[test]
fn the_parent_adds_its_childs_placement_going_up() {
    use vd_core::frame::transfer_frame;
    let story = Story::new();

    // The planet's own view: an occupant 3 m from its centre. This is ALL it knows.
    let at_planet = story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M);

    // The SYSTEM converts, because the system is the only party that holds "I put that planet at 145".
    let at_system = transfer_frame(
        &at_planet,
        story.frame(story.system),
        &story.ctx(story.system),
    )
    .expect("a star can place its own planet");
    assert_eq!(at_system.frame, story.frame(story.system));
    assert_eq!(
        fm(at_system.pos),
        DVec3::new(story.up_1_m(), 0.0, 0.0),
        "the system adds its child's placement: 145 + 3",
    );

    // The GALAXY converts the result, because only the galaxy holds where it put that system.
    // Since the 3-D seeded placement law (owner Q-B) the system's authored centre is a seeded
    // 3-D direction at the story radius — the galaxy's addition is the same VECTOR add the
    // collinear story spelt on one axis: placement + (148, 0, 0).
    let at_galaxy = transfer_frame(
        &at_system,
        story.frame(story.galaxy),
        &story.ctx(story.galaxy),
    )
    .expect("a galaxy can place its own star system");
    assert_eq!(at_galaxy.frame, story.frame(story.galaxy));
    // ★ READ IN THE GALAXY'S OWN UNIT (slice S9). This pose is now counted in two-metre steps, because
    // that is what the frame it was handed into counts in — reading it in millimetres would be out by
    // 2048× and would still look like a plausible distance.
    assert_eq!(
        fm_at(at_galaxy.pos, story.frame(story.galaxy).tier()),
        Story::centre(&story.world, story.system) + DVec3::new(story.up_1_m(), 0.0, 0.0),
        "the galaxy adds its child's placement (the seeded 3-D vector + 148 along x)",
    );
}

/// GOING DOWN: the galaxy computes 12179 − 12031 = 148 and hands it to the system; the system computes
/// 148 − 145 = 3 and hands it to the planet; the planet ACCEPTS 3 and does no arithmetic at all. The
/// last step is asserted BIT-IDENTICAL, because "the child does nothing" is the half of the ground rule
/// that an approximate assertion would let slide.
#[test]
fn the_parent_subtracts_going_down_and_the_child_does_nothing() {
    use vd_core::frame::transfer_frame;
    let story = Story::new();

    // The pose the galaxy holds: the occupant 148 m up the system's own +x, expressed in the
    // galaxy frame through the galaxy's own book (the seeded 3-D placement + the offset — the
    // collinear story's "12179 on one axis" generalized to the vector it always was).
    let at_galaxy = transfer_frame(
        &story.pose_in(story.frame(story.system), story.up_1_m()),
        story.frame(story.galaxy),
        &story.ctx(story.galaxy),
    )
    .expect("a galaxy can place its own star system");

    // The GALAXY subtracts, because it is the party that knows where it put the system —
    // BIT-EXACT: the integer anchors cancel and the residual subtraction is of equal halves.
    let at_system = transfer_frame(
        &at_galaxy,
        story.frame(story.system),
        &story.ctx(story.galaxy),
    )
    .expect("a galaxy can place its own star system");
    assert_eq!(
        fm(at_system.pos),
        DVec3::new(story.up_1_m(), 0.0, 0.0),
        "the galaxy subtracts its child's placement exactly",
    );

    // The SYSTEM subtracts, because it is the party that knows where it put the planet.
    let at_planet = transfer_frame(
        &at_system,
        story.frame(story.planet),
        &story.ctx(story.system),
    )
    .expect("a star can place its own planet");
    assert_eq!(
        fm(at_planet.pos),
        DVec3::new(STORY_OCCUPANT_FROM_PLANET_M, 0.0, 0.0),
        "the system subtracts its child's placement: 148 − 145",
    );

    // The PLANET accepts and does nothing. Same frame in, same frame out, BIT-IDENTICAL — the child
    // never re-derives a number its parent has already measured for it.
    let accepted = transfer_frame(
        &at_planet,
        story.frame(story.planet),
        &story.ctx(story.planet),
    )
    .expect("a realm always knows its own frame");
    assert_eq!(
        accepted, at_planet,
        "the child accepts and does no arithmetic"
    );
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

/// THE DOWNWARD ARRIVAL: my parent already expressed this in my frame, so I accept it and do NOTHING.
/// Asserted BIT-IDENTICAL — "the child does no arithmetic" is the half of the ground rule that an
/// approximate assertion would let slide.
#[test]
fn an_arrival_already_in_my_own_frame_is_accepted_bit_identical() {
    let mut stats = StubStats::default();
    let story = Story::new();
    let cfg = story_config(&story, story.planet);
    let arriving = story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M);
    assert_eq!(
        place_arriving_pose(
            arriving,
            story.planet,
            &cfg,
            &story.regions(story.planet),
            &story.ledger(story.planet),
            &mut stats,
        )
        .expect("a realm always knows its own frame"),
        arriving,
        "the child accepts and does no arithmetic",
    );
}

/// THE UPWARD ARRIVAL, which is the entire point of the change: the planet ships "3, in my frame" and
/// the SYSTEM — the only party that holds "I put that planet at 145" — adds it and gets 148.
#[test]
fn an_arrival_from_a_direct_child_gets_that_childs_placement_added() {
    let mut stats = StubStats::default();
    let story = Story::new();
    let cfg = story_config(&story, story.system);
    let placed = place_arriving_pose(
        story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M),
        story.system,
        &cfg,
        &story.regions(story.system),
        &story.ledger(story.system),
        &mut stats,
    )
    .expect("a star can place its own planet");
    assert_eq!(placed.frame, story.frame(story.system));
    assert_eq!(
        fm(placed.pos),
        DVec3::new(story.up_1_m(), 0.0, 0.0),
        "the system adds its child's placement: 145 + 3",
    );
}

/// A SIBLING HAND-OFF IS A REFUSAL, not a relabel. Measured before the change: a pose handed
/// planet-11 → planet-22 landed as `3.0 @ PlanetCentered{22}` with `Ok` at all three hops — no error,
/// no counter, no log anywhere, and an occupant silently teleported by the distance between the two
/// planets. Nobody told this planet where its sibling is, so there is nothing it could add.
#[test]
fn an_arrival_from_a_sibling_is_refused_never_relabelled() {
    let mut stats = StubStats::default();
    let story = Story::new();
    let cfg = story_config(&story, story.planet);
    assert_eq!(
        place_arriving_pose(
            story.pose_in(
                story.frame(story.sibling_planet),
                STORY_OCCUPANT_FROM_PLANET_M
            ),
            story.planet,
            &cfg,
            &story.regions(story.planet),
            &story.ledger(story.planet),
            &mut stats,
        )
        .expect_err("a planet must not be able to place its sibling's occupants"),
        UnplaceableArrival::ForeignFrame(FrameError::UnknownSourceFrame),
    );
}

/// A shard that booted with NO region forest knows where nothing is — including its own realm — so
/// every arrival it cannot pass through verbatim is refused. Both sides of the conversion are
/// exercised: a child-framed pose fails on the SOURCE side (that child has no placement here), and a
/// pose that happens to arrive in the context's fallback frame fails on the DESTINATION side (this
/// shard's own realm has no placement either). The shipped `shard.rs` always plants a neighbourhood;
/// this is the degenerate boot the guard must survive without inventing a position.
#[test]
fn an_arrival_at_a_shard_with_no_forest_is_refused_on_whichever_side_is_missing() {
    let mut stats = StubStats::default();
    let story = Story::new();
    let cfg = story_config(&story, story.system);
    assert_eq!(
        place_arriving_pose(
            story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M),
            story.system,
            &cfg,
            &RealmRegions::default(),
            &bare_ledger(&cfg),
            &mut stats,
        )
        .expect_err("no forest ⇒ no child placements ⇒ nothing to add"),
        UnplaceableArrival::ForeignFrame(FrameError::UnknownSourceFrame),
    );
    // An empty forest has no region to read a frame from, so the context anchors its identity on the
    // ambient-root fallback — which is NOT the frame this shard's realm is named in. So the source
    // resolves and the DESTINATION does not.
    //
    // ★ RE-BASED IN S9: that fallback is `UniverseSpace`. It was `GalaxySpace` only because the
    // universe had no frame to fall back to, and a pose arriving in a galaxy frame no longer lands on
    // the anchor — it is simply a frame this bare shard has never been told about, which fails on the
    // SOURCE side and would have tested nothing new.
    assert_eq!(
        place_arriving_pose(
            story.pose_in(FrameRef::UniverseSpace, STORY_OCCUPANT_FROM_PLANET_M),
            story.system,
            &cfg,
            &RealmRegions::default(),
            &bare_ledger(&cfg),
            &mut stats,
        )
        .expect_err("a shard with no forest cannot place its own realm either"),
        UnplaceableArrival::ForeignFrame(FrameError::UnknownDestFrame),
    );
}

/// An `Area` shard booted WITHOUT its enclosing planet cannot even NAME its own frame (an area frame
/// carries its planet's seed as well as its own), so it has no space to measure an arrival in.
#[test]
fn an_arrival_at_a_shard_that_cannot_name_its_own_frame_is_refused() {
    let mut stats = StubStats::default();
    let story = Story::new();
    let cfg = StubConfig {
        realm: RealmId::Area(9),
        held_realms: StubConfig::single_realm(RealmId::Area(9)),
        own_coord: StubConfig::root_coord(RealmId::Area(9)), // no parent ⇒ no planet seed
        ..config()
    };
    assert_eq!(
        place_arriving_pose(
            story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M),
            RealmId::Area(9),
            &cfg,
            &RealmRegions::default(),
            &bare_ledger(&cfg),
            &mut stats,
        )
        .expect_err("an area with no planet has no frame of its own"),
        UnplaceableArrival::UnnameableOwnFrame,
    );
}

/// THE SOURCE'S DOWNWARD HALF: handing an occupant to one of MY OWN children is arithmetic I hold, so
/// I do it — 148 in my frame becomes 3 in my planet's.
#[test]
fn a_flush_into_my_own_child_subtracts_that_childs_placement() {
    let story = Story::new();
    let cfg = story_config(&story, story.system);
    let mut stats = StubStats::default();
    let shipped = flush_pose_for_dest(
        story.pose_in(story.frame(story.system), story.up_1_m()),
        story.planet,
        &cfg,
        &story.regions(story.system),
        &story.ledger(story.system),
        UniverseTick(0),
        &mut stats,
    )
    .expect("a star can place its own planet");
    assert_eq!(shipped.frame, story.frame(story.planet));
    assert_eq!(
        fm(shipped.pos),
        DVec3::new(STORY_OCCUPANT_FROM_PLANET_M, 0.0, 0.0),
        "the system subtracts its child's placement: 148 − 145",
    );
    assert_eq!(stats.flush_unplaceable_child, 0);
}

/// THE SOURCE'S UPWARD HALF: handing an occupant to my PARENT ships the pose VERBATIM, in my own
/// frame, because I have never been told where my parent is. Bit-identical — the arm builds no
/// context and does no arithmetic at all.
///
/// This used to reach the same answer down the FAILURE path: one unconditional conversion whose
/// upward arm worked only because it errored and the degrade handed the pose back untouched. Right
/// answer, no way to tell it from a genuine fault.
#[test]
fn a_flush_up_to_my_parent_ships_the_pose_verbatim() {
    let story = Story::new();
    let cfg = story_config(&story, story.planet);
    let mut stats = StubStats::default();
    // OUTSIDE the release edge: under Stage B1 the departure must still be TRUE at flush time —
    // an occupant still inside the shell is a refusal (its own test below), not a verbatim ship.
    let held = story.pose_in(story.frame(story.planet), story.departed_m());
    assert_eq!(
        flush_pose_for_dest(
            held,
            story.system,
            &cfg,
            &story.regions(story.planet),
            &story.ledger(story.planet),
            UniverseTick(0),
            &mut stats,
        )
        .expect("shipping upward never fails — there is nothing to compute"),
        held,
        "the child ships what it holds, tagged with its own frame",
    );
    assert_eq!(
        stats.flush_unplaceable_child, 0,
        "an upward hand-off is not an error and must never be counted as one",
    );
    assert_eq!(
        stats.flush_stale_exit, 0,
        "a departure that is still true at flush time is never refused",
    );
}

/// EVERY LEDGER-MISS ARM, exercised loudly (the placement arc S2 — HR5: a fallible selection's
/// refusal is a region, and an unexercised refusal is a belief). Each case hands the pure function
/// a ledger that genuinely lacks the book it asks for and asserts the LOUD half: the typed refusal
/// or the counted degrade, never a silently substituted instant.
#[test]
fn every_ledger_miss_arm_refuses_loudly() {
    let story = Story::new();
    let t0 = UniverseTick(0);

    // (a) THE FLUSH'S HEAD MISS: no head book for this shard's own realm ⇒ the flush refuses
    // outright (no `SourceFlushed` ⇒ the saga aborts ⇒ this shard keeps authority).
    let cfg = story_config(&story, story.planet);
    let mut stats = StubStats::default();
    assert_eq!(
        flush_pose_for_dest(
            story.pose_in(story.frame(story.planet), story.departed_m()),
            story.system,
            &cfg,
            &story.regions(story.planet),
            &PlacementLedger::new(8),
            t0,
            &mut stats,
        ),
        None,
    );
    assert_eq!(stats.placement_book_miss, 1);

    // A CO-HOSTING shard (galaxy + system) — the multi-link paths live here.
    let held = std::collections::BTreeSet::from([story.galaxy, story.system]);
    let co_regions = RealmRegions::new(story.world.neighbourhood(&held));
    let co_cfg = StubConfig {
        realm: story.galaxy,
        held_realms: held.clone(),
        frame: story.frame(story.galaxy),
        own_coord: vd_core::worldgen::coord_of_realm(
            &story.world.neighbourhood(&held),
            story.galaxy,
        )
        .expect("the galaxy has a lineage"),
        ..config()
    };
    // Test twin of the ONE writer — the same stated exemption from the publish ban.
    #[allow(clippy::disallowed_methods)]
    let ledger_with = |anchors: &[RealmId]| -> PlacementLedger {
        let mut ledger = PlacementLedger::new(8);
        for &anchor in anchors {
            ledger.publish(anchor, co_regions.author_book(anchor, STORY_TICK_HZ, t0));
        }
        ledger
    };

    // (b) A PER-LINK MISS: descending galaxy → system → planet with the SYSTEM's book absent ⇒
    // the second link refuses the flush.
    //
    // ★ WHERE THE POSE STARTS, RE-DERIVED IN S9 — and the comment it replaces was wrong before S9
    // ever touched it. It read "far outside the galaxy so the departure re-validation lets it leave",
    // and 1.0e6 m was never outside a galaxy whose own extent is 4.5e16 m. It was a thousand
    // kilometres from the galaxy's CENTRE, which is a different thing entirely.
    //
    // What the climb exposed is that the real constraint runs the other way. This descent ends in a
    // PLANET's frame, which counts in millimetres, and a millimetre lattice reaches 2.25e15 m — about
    // a quarter of a light year. The galaxy's centre is 0.7 light years from this star system, so a
    // pose parked there has NO millimetre count relative to anything inside the system, and the flush
    // is refused (`BeyondReach`) before any book is consulted. That is the coordinate system telling
    // the truth, not a fault.
    //
    // So the pose starts AT THE SYSTEM — read from the galaxy's own authored placement for it, never
    // typed — which is where an occupant descending into that system would actually be.
    let mut stats = StubStats::default();
    let outside = StampedPose::at_rest(
        story.frame(story.galaxy),
        Story::centre(&story.world, story.system),
        UniverseTick(0),
    );
    assert_eq!(
        flush_pose_for_dest(
            outside,
            story.planet,
            &co_cfg,
            &co_regions,
            &ledger_with(&[story.galaxy]),
            t0,
            &mut stats,
        ),
        None,
    );
    assert_eq!(stats.placement_book_miss, 1);

    // (c) THE ENTRY RE-VALIDATION'S MISS: a pure ASCENT (planet-labelled pose, dest = the galaxy
    // itself) converts through system and galaxy, then re-validates the entry in the galaxy's own
    // PARENT's book (the universe) — absent ⇒ refused. Links hit; only the entry book is missing.
    let mut stats = StubStats::default();
    assert_eq!(
        flush_pose_for_dest(
            story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M),
            story.galaxy,
            &co_cfg,
            &co_regions,
            &ledger_with(&[story.galaxy, story.system]),
            t0,
            &mut stats,
        ),
        None,
    );
    assert_eq!(stats.placement_book_miss, 1);

    // (d) THE ENTITY FEED'S MISS: a row whose stamp has no book ships VERBATIM under its own
    // label, counted — the same counted degrade a foreign-labelled row takes.
    let mut stats = StubStats::default();
    let entity = EntityId::pack(EntityKind::Player, 10, 9, 9);
    let own_frame = story.frame(story.system);
    let mut dots = Dots::default();
    dots.0.insert(
        SessionId(77),
        slice6_dot(entity, own_frame, Authority::Owned { fence: Fence(1) }),
    );
    let rows = emitted_entities(
        &dots,
        &HandoffHolds::default(),
        own_frame,
        &PlacementLedger::new(8),
        story.system,
        &mut stats,
    );
    assert_eq!(rows.len(), 1, "the row still ships");
    assert_eq!(
        rows[0].pose,
        dots.0[&SessionId(77)].pose,
        "verbatim, never re-spaced"
    );
    assert_eq!(stats.placement_book_miss, 1);
}

/// The DETECTOR's two ledger selections, refused loudly through the FULL schedule: a durable dot
/// standing in a leaf child (an anchor the writer never authors — head miss) and a held transient
/// whose stamp fell behind the retained window (an at() miss). Each is COUNTED and the subject is
/// simply not evaluated that tick — no crossing is invented from a book nobody authored.
#[test]
fn the_detector_counts_a_missing_book_and_skips_the_subject() {
    // (a) Dot in a leaf child's frame: `book_anchor` resolves to the child, which has no children
    // and is not held ⇒ never authored ⇒ head miss.
    let mut rig = Rig::new();
    rig.grant_realm();
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), own_region(), child_region()]);
    let entity = EntityId::pack(EntityKind::Player, 10, 2, 0x60);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(1.0, 0.0, 0.0));
    rig.world
        .resource_mut::<Dots>()
        .0
        .get_mut(&TRIG_SESSION)
        .expect("just inserted")
        .pose
        .frame = frame_of(OTHER_REALM);
    let sent = rig.tick(vec![]);
    assert_eq!(
        rig.world.resource::<StubStats>().placement_book_miss,
        1,
        "the head miss is counted"
    );
    assert!(
        crossing_requests(&sent).is_empty(),
        "no crossing is invented from a missing book"
    );

    // (b) A held transient whose stamp is older than the retained window.
    let mut rig = Rig::new();
    rig.grant_realm();
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), own_region()]);
    let stale = StampedPose::at_rest(
        frame_of(OWN_REALM),
        DVec3::new(1.0, 0.0, 0.0),
        UniverseTick(10),
    );
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        EntityId::pack(EntityKind::Debris, 10, 2, 0x61),
        Transient {
            pose: stale,
            anchor_fence: Fence(1),
            status: TransientStatus::Held { outbound: None },
            prev_offset: stale.pos,
        },
    );
    rig.tick(vec![]);
    assert_eq!(
        rig.world.resource::<StubStats>().placement_book_miss,
        1,
        "the stale-stamp miss is counted (rig clock 100, window far narrower than 90 ticks)"
    );
}

/// The co-hosted OUTWARD destination: an occupant leaving a realm this shard co-hosts (not its
/// primary) falls to THAT realm's roster parent — one level up from where it stood, never from
/// where the shard is named.
#[test]
fn outward_dest_of_a_cohosted_realm_is_that_realms_own_parent() {
    let story = Story::new();
    let held = std::collections::BTreeSet::from([story.system, story.planet]);
    let regions = story.world.neighbourhood(&held);
    let cfg = story_config(&story, story.system);
    assert_eq!(
        outward_dest(&cfg, &regions, story.universe, story.planet),
        story.system,
        "leaving the co-hosted planet lands in the SYSTEM (its parent), not the shard's own parent"
    );
}

/// A receiver with NO AUTHORED BOOK AT ALL (its first synced tick has not run) is the ONE case
/// the arrival still refuses — typed, counted, and retried by the saga until the writer's first
/// pass lands: nothing can be placed against a world nobody has authored yet.
#[test]
fn an_arrival_before_the_first_authored_book_is_refused_typed() {
    let story = Story::new();
    let cfg = story_config(&story, story.system);
    let mut stats = StubStats::default();
    let err = place_arriving_pose(
        story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M),
        story.system,
        &cfg,
        &story.regions(story.system),
        &PlacementLedger::new(8),
        &mut stats,
    )
    .expect_err("no head book ⇒ nothing to place against");
    assert_eq!(
        err,
        UnplaceableArrival::BookMiss(vd_core::placement::PlacementMiss {
            anchor: story.system,
            wanted: UniverseTick(0),
            head: None,
            span: 8,
        })
    );
    assert_eq!(stats.placement_book_miss, 1);
    assert_eq!(stats.placement_skew_clamped, 0);
}

/// FORWARD CLOCK SKEW is CLAMPED and MEASURED, never a refusal and never silent (the process-tier
/// wedge's cure): a message-carried instant AHEAD of this shard's head reads the head book with the
/// instant clamped to the head's own; behind-the-window stays the loud miss; an exact hit stays
/// exact. The arrival lane then accepts the hand-off it used to refuse.
#[test]
fn a_forward_skewed_instant_clamps_to_the_head_counted_and_measured() {
    let story = Story::new();
    let ledger = story.ledger(story.system); // heads at tick 0
    let mut stats = StubStats::default();
    // THE ARRIVAL LANE ACCEPTS THE SKEWED HAND-OFF (the wedge's cure), re-stamped to the
    // receiver's own instant: an upward planet→system hand-off whose pose is stamped one tick
    // ahead of everything the system has authored.
    let cfg = story_config(&story, story.system);
    let mut ahead = story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M);
    ahead.universe_tick = UniverseTick(1);
    let placed = place_arriving_pose(
        ahead,
        story.system,
        &cfg,
        &story.regions(story.system),
        &ledger,
        &mut stats,
    )
    .expect("a skewed but healthy hand-off is accepted, never wedged");
    assert_eq!(placed.frame, story.frame(story.system));
    assert_eq!(
        placed.universe_tick,
        UniverseTick(0),
        "the pose speaks at the receiver's own instant after the clamp"
    );
    assert_eq!(
        fm(placed.pos),
        DVec3::new(story.up_1_m(), 0.0, 0.0),
        "the system adds its child's placement exactly as an un-skewed arrival: 145 + 3"
    );
    assert_eq!(stats.placement_skew_clamped, 1);
    assert_eq!(
        stats.placement_skew_ahead_max_ticks, 1,
        "the arrival's 1-tick lead rides the ahead gauge"
    );
    assert_eq!(
        stats.placement_skew_behind_max_ticks, 0,
        "a FORWARD clamp writes the ahead gauge only — the two directions are separate \
         measurements (batch review: one |Δ| magnitude let redelivery staleness drown the \
         span_ahead bound)"
    );

    // …and a RETRIED hand-off whose stamp aged BEHIND the window (the process-tier wedge: an
    // immutably-stamped envelope redelivered while the head advanced) is ALSO accepted at the
    // receiver's now — never refused forever.
    #[allow(clippy::disallowed_methods)] // test twin of the ONE writer
    let aged = {
        let mut ledger = PlacementLedger::new(2);
        for t in [100u64, 101, 102] {
            ledger.publish(
                story.system,
                story.regions(story.system).author_book(
                    story.system,
                    STORY_TICK_HZ,
                    UniverseTick(t),
                ),
            );
        }
        ledger
    };
    let mut stats = StubStats::default();
    let mut stale = story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M);
    stale.universe_tick = UniverseTick(10); // 90 ticks behind the head, window 2
    let placed = place_arriving_pose(
        stale,
        story.system,
        &cfg,
        &story.regions(story.system),
        &aged,
        &mut stats,
    )
    .expect("an aged but healthy hand-off is accepted at the receiver's now, never wedged");
    assert_eq!(placed.universe_tick, UniverseTick(102));
    assert_eq!(stats.placement_skew_clamped, 1);
    assert_eq!(
        stats.placement_skew_behind_max_ticks, 92,
        "a BACKWARD (redelivery-staleness) clamp writes the behind gauge only"
    );
    assert_eq!(stats.placement_skew_ahead_max_ticks, 0);
}

/// STAGE B1's mirrored half (§4v cure 1): the departure was decided, and by flush time the occupant
/// is back INSIDE its own realm (the run-1 measurement: 2.37 m against a 4.16 m boundary). The flush
/// refuses — no `SourceFlushed`, the saga aborts pre-commit, this shard keeps authority.
#[test]
fn a_departure_that_is_no_longer_true_is_refused_at_the_flush() {
    let story = Story::new();
    let cfg = story_config(&story, story.planet);
    let mut stats = StubStats::default();
    assert_eq!(
        flush_pose_for_dest(
            story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M),
            story.system,
            &cfg,
            &story.regions(story.planet),
            &story.ledger(story.planet),
            UniverseTick(0),
            &mut stats,
        ),
        None,
        "an occupant still inside its own realm must not be handed to the parent",
    );
    assert_eq!(stats.flush_stale_exit, 1, "the stale departure is counted");
    assert_eq!(
        stats.flush_unplaceable_child, 0,
        "a stale departure is a refusal, never an unplaceable-child fault",
    );
}

/// STAGE B1's first half (§4v cure 1): the entry was decided, and by flush time the occupant has
/// left the destination (the run-1 measurement: landed 12.15 / 23.00 / 20.72 m outside a 4.16 m
/// boundary, each an instant bounce-back). The flush refuses instead of committing a mislanding.
#[test]
fn an_entry_that_is_no_longer_true_is_refused_at_the_flush() {
    let story = Story::new();
    let cfg = story_config(&story, story.system);
    let mut stats = StubStats::default();
    // The system's own centre is one full orbital radius (145 m) from the planet — far past the
    // shell (36.25 m) plus the release edge, so the destination would never hold it.
    assert_eq!(
        flush_pose_for_dest(
            story.pose_in(story.frame(story.system), 0.0),
            story.planet,
            &cfg,
            &story.regions(story.system),
            &story.ledger(story.system),
            UniverseTick(0),
            &mut stats,
        ),
        None,
        "a pose the destination would not hold must not be shipped into it",
    );
    assert_eq!(stats.flush_stale_entry, 1, "the stale entry is counted");
    assert_eq!(
        stats.flush_unplaceable_child, 0,
        "a stale entry is a refusal, never an unplaceable-child fault",
    );
}

/// The self-crossing (`to_realm == config.realm`) skips the departure re-validation: "still inside
/// myself" is not a stale departure, and the pose ships verbatim exactly as before Stage B1.
#[test]
fn a_self_crossing_flush_skips_the_departure_check_and_ships_verbatim() {
    let story = Story::new();
    let cfg = story_config(&story, story.planet);
    let mut stats = StubStats::default();
    let held = story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M);
    assert_eq!(
        flush_pose_for_dest(
            held,
            story.planet,
            &cfg,
            &story.regions(story.planet),
            &story.ledger(story.planet),
            UniverseTick(0),
            &mut stats,
        ),
        Some(held),
        "a self-crossing ships what it holds — the launder path stays open",
    );
    assert_eq!(
        stats.flush_stale_exit, 0,
        "a self-crossing is never counted as a stale departure",
    );
}

/// A pose still LABELLED with an ancestor's frame (a realm this shard rosters but does not
/// author) may not anchor the hand-off: the anchor warn counts, the conversion path refuses the
/// ancestor as a converting parent (authorship constraint), and the pose ships VERBATIM for the
/// true author to place.
#[test]
fn a_pose_labelled_with_an_ancestors_frame_warns_and_ships_verbatim() {
    let story = Story::new();
    let cfg = story_config(&story, story.system);
    let mut stats = StubStats::default();
    // Labelled with the GALAXY's frame — rostered here (the ambient chain) but never authored.
    let held = story.pose_in(
        story.frame(story.galaxy),
        story.config.stellar.system_ring_r_m + story.up_1_m(),
    );
    assert_eq!(
        flush_pose_for_dest(
            held,
            story.planet,
            &cfg,
            &story.regions(story.system),
            &story.ledger(story.system),
            UniverseTick(0),
            &mut stats,
        ),
        Some(held),
        "an ancestor-labelled pose ships un-converted — the arithmetic is not this shard's",
    );
    assert_eq!(
        stats.flush_anchor_not_own, 1,
        "the mis-anchored label is counted where it is refused"
    );
}

/// A pose whose frame NOBODY in this roster registers cannot descend: the first converting link
/// refuses (a typed frame error, never a guess) and the flush declines the hand-off with the
/// unplaceable-child count — the one arm that serves both directions of the step list.
#[test]
fn a_descent_from_an_unregistered_frame_is_refused_and_counted() {
    let story = Story::new();
    let cfg = story_config(&story, story.system);
    let mut stats = StubStats::default();
    // A frame no region of this roster carries: the label falls back to the own realm, so the
    // path is a pure descent — whose first conversion cannot place the pose's actual frame.
    let phantom = story.pose_in(FrameRef::PlanetCentered { planet_seed: 999 }, 3.0);
    assert_eq!(
        flush_pose_for_dest(
            phantom,
            story.planet,
            &cfg,
            &story.regions(story.system),
            &story.ledger(story.system),
            UniverseTick(0),
            &mut stats,
        ),
        None,
        "a pose nobody here can measure is refused, never shipped under a guess",
    );
    assert_eq!(stats.flush_unplaceable_child, 1);
}

/// A roster that holds NO region for this shard's own realm cannot re-validate a departure, and the
/// guard steps aside: the pose ships verbatim exactly as before Stage B1 (the receiver stays the
/// judge). Never normal on the seed path — the roster always carries self — but the guard must not
/// invent a refusal from an absence.
#[test]
fn a_departure_check_without_an_own_region_ships_verbatim() {
    let story = Story::new();
    let cfg = story_config(&story, story.planet);
    let mut stats = StubStats::default();
    // A roster holding ONLY the ambient root: no own region, and the dest is not a direct child.
    let roster = RealmRegions::new(vec![
        story
            .regions(story.planet)
            .regions
            .iter()
            .find(|r| r.parent.is_none())
            .copied()
            .expect("every neighbourhood carries the ambient root"),
    ]);
    let held = story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M);
    let clock0 = ClockSample {
        local_tick: vd_core::TickId(0),
        universe_tick: UniverseTick(0),
        epoch: vd_core::EpochId(1),
        synced: true,
    };
    assert_eq!(
        flush_pose_for_dest(
            held,
            story.system,
            &cfg,
            &roster,
            &obs_ledger(&roster, &cfg, &clock0),
            UniverseTick(0),
            &mut stats
        ),
        Some(held),
        "with no own region to measure against, the flush ships as it always did",
    );
    assert_eq!(
        stats.flush_stale_exit, 0,
        "an absent own region is never counted as a stale departure",
    );
}

/// The roster says "my direct child" and the ephemeris still cannot express the outgoing pose in it —
/// here because the dot carries a frame from a realm this shard was never told the position of. That
/// is a REAL internal contradiction: counted, logged, and the hand-off REFUSED, so this shard keeps
/// authority and the saga aborts rather than shipping a number nobody computed.
#[test]
fn a_flush_into_my_child_that_cannot_be_computed_is_refused_and_counted() {
    let story = Story::new();
    let cfg = story_config(&story, story.system);
    let mut stats = StubStats::default();
    assert_eq!(
        flush_pose_for_dest(
            story.pose_in(
                story.frame(story.sibling_system),
                STORY_OCCUPANT_FROM_PLANET_M
            ),
            story.planet,
            &cfg,
            &story.regions(story.system),
            &story.ledger(story.system),
            UniverseTick(0),
            &mut stats,
        ),
        None,
        "a pose this shard cannot measure is not shipped",
    );
    assert_eq!(stats.flush_unplaceable_child, 1);
}

#[test]
fn an_unnameable_pose_frame_safe_degrades_to_non_member_never_a_spurious_container() {
    // FA-1 safe-degrade: a subject whose pose frame the shard cannot NAME (not among its regions) has
    // `region_signed_distance` → `Err` → `f64::MAX` for EVERY region, so it is a member of NONE and its
    // deepest container folds to the ambient ROOT — never a spurious INNER container, never a panic. The
    // discriminator: at the origin under the OWN (registered) frame the deepest container is the child
    // `OTHER_REALM` (a re-home INWARD); under an un-nameable frame the child must NOT be entered.
    let mut rig = Rig::new(); // owns System 7 (config().realm == OWN_REALM)
    rig.grant_realm();
    plant_dock_regions(&mut rig); // root(1e9) ⊃ own(1e5) ⊃ child(1000), all shells at the origin
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 77);
    // A Station frame the dock forest NEVER planted — un-nameable to any dock region.
    insert_owned_dot_framed(
        &mut rig,
        TRIG_SESSION,
        entity,
        FrameRef::StationLocal { station_seed: 999 },
        DVec3::ZERO,
    );
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..6 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    assert!(
        crossing_requests(&all)
            .iter()
            .all(|r| r.to_realm != OTHER_REALM),
        "an un-nameable pose frame safe-degrades to non-member ⇒ NEVER re-homes into the inner child",
    );
}

#[test]
fn the_crossing_decision_does_not_depend_on_whether_this_shard_remembers_the_subject() {
    // THE INVARIANT THAT FIXES THE BOUNDARY RE-FIRE (task #177), stated as a property rather than a
    // scenario. The stored membership bitset is per-shard and RAM-only: the source shard has one, an
    // arriving shard (or any shard after a restart) has a BLANK one. Before the derived prior those two
    // states produced DIFFERENT answers for the same subject at the same place — which is exactly why a
    // subject resting between the acquire and release edges ping-ponged between two shards forever.
    //
    // So: run the identical setup twice, differing ONLY in whether the shard already remembers the
    // subject as a member of its owning realm, and require the emitted crossings to be IDENTICAL.
    //
    // This is the RED control for the fix: with the stored-only prior the blank run acquires the realm
    // from scratch and the seeded run does not, so the two disagree and this fails. It cannot pass
    // vacuously either — `plant_dock_regions` gives a real nested forest and the dot sits where the
    // hysteresis band is genuinely ambiguous, so the prior is load-bearing for the outcome.
    let run = |seed_membership: bool| -> Vec<RealmId> {
        let mut rig = Rig::new(); // owns System 7 == OWN_REALM
        rig.grant_realm();
        plant_dock_regions(&mut rig);
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 78);
        // THE POSITION THAT MATTERS: just inside the inner child's shell (r=1000) but NOT far enough
        // in to ACQUIRE it from scratch — the band's ambiguous zone, where the prior alone decides
        // membership. This is exactly where a player who stops on a boundary ends up.
        //
        // The subject is OWNED BY the child (its pose frame names Planet 42), i.e. it has just been
        // handed off inward. A correct shard therefore keeps it there and emits NO crossing. With the
        // stored-only prior, a shard with no memory of it fails to acquire the child, folds its
        // container out to the parent, and immediately re-homes it BACK OUT — the flap.
        insert_owned_dot_framed(
            &mut rig,
            TRIG_SESSION,
            entity,
            FrameRef::PlanetCentered { planet_seed: 42 },
            DVec3::new(990.0, 0.0, 0.0),
        );
        if seed_membership {
            // The SOURCE shard's state: it already remembers this subject inside the child.
            let mut bits = RegionMembership::default();
            bits.set(ROOT_REALM, true);
            bits.set(OWN_REALM, true);
            // The child — the membership an arriving shard does NOT have. Named by REALM now, not by a
            // position in a bitset, so this seeding says what it means and cannot drift with an index.
            bits.set(OTHER_REALM, true);
            rig.world
                .resource_mut::<ContainmentProgress>()
                .0
                .insert(entity, bits);
        }
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..8 {
            rig.set_local_tick(t);
            all.extend(rig.tick(vec![]));
        }
        crossing_requests(&all).iter().map(|r| r.to_realm).collect()
    };

    let remembered = run(true);
    let blank = run(false);
    assert_eq!(
        blank, remembered,
        "a shard that has never seen this subject must decide EXACTLY as one that remembers it — \
         otherwise the two sides of a hand-off disagree and the subject flaps at the boundary"
    );
    // And the correct shared answer is "no crossing at all": the dot is already in its owning realm.
    assert_eq!(remembered, Vec::<RealmId>::new());
}

#[test]
fn owning_realm_reads_the_pose_frame_when_nameable() {
    // The owning realm is the pose FRAME's realm. There is no longer a fallback to ignore — every frame
    // names one — so what this covers is that each KIND of frame names the right thing.
    assert_eq!(
        super::owning_realm(FrameRef::SystemSpace { system_seed: 7 }),
        RealmId::System(7),
        "a System frame owns System(system_seed), not the config fallback",
    );
    assert_eq!(
        super::owning_realm(FrameRef::PlanetCentered { planet_seed: 3 }),
        RealmId::Planet(3),
        "a Planet frame owns Planet(planet_seed)",
    );
    assert_eq!(
        super::owning_realm(FrameRef::AreaLocal {
            planet_seed: 3,
            area_seed: 8,
        }),
        RealmId::Area(8),
        "an Area frame owns Area(area_seed) — the very frame the to_parent fix makes form",
    );
}

#[test]
fn owning_realm_names_the_galaxy_and_the_universe_too() {
    // ★ REPLACES `owning_realm_falls_back_to_config_realm_for_an_unnameable_frame` (slice S9). That test
    // drove a fallback whose ONLY reachable input was galaxy space, and its own comment said so:
    // "otherwise UNCOVERABLE — no live shard uses GalaxySpace". The fallback is gone because the reason
    // for it is: a galaxy names its realm now, so there is nothing left to fall back FROM.
    //
    // What is asserted instead is the property that replaced it — the two frames that used to have no
    // answer now give one, and it is their own.
    assert_eq!(
        super::owning_realm(FrameRef::GalaxySpace { galaxy_seed: 5 }),
        RealmId::Galaxy(5),
        "a galaxy frame owns the galaxy it names",
    );
    assert_eq!(
        super::owning_realm(FrameRef::UniverseSpace),
        RealmId::Universe,
        "the universe frame owns the universe — there is exactly one",
    );
}

#[test]
fn durable_dot_dwelling_in_band_triggers_exactly_one_crossing_request() {
    let mut rig = Rig::new();
    rig.grant_realm();
    // The dock forest (root ⊃ own(System(7)) ⊃ child(Planet(42))).
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 1);
    // Deep inside the CHILD region's shell (|100| ≪ 1000 - inset): container == OTHER_REALM ⇒ re-home.
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));

    // Under containment the band IS the dwell: the re-home fires as soon as the container differs;
    // the RequestInFlight latch then suppresses every later tick → exactly ONE request across the dwell.
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..8 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    let reqs = crossing_requests(&all);
    assert_eq!(
        reqs.len(),
        1,
        "exactly ONE CrossingRequest across the dwell"
    );
    assert_eq!(reqs[0].subject, DirectoryKey::Entity(entity));
    assert_eq!(reqs[0].from_realm, config().realm);
    assert_eq!(reqs[0].to_realm, OTHER_REALM);
    assert_eq!(reqs[0].subject_fence, Fence(1));
    // Slice 3f: the source threads the dot's session (its `Dots` map key) so the orchestrator's
    // saga can `PrepareSubscribe` to the client's gateway.
    assert_eq!(reqs[0].session, TRIG_SESSION);
    // 3f-D: the FIRST crossing uses attempt 0.
    assert_eq!(reqs[0].attempt, 0);
    // The latch is set to the deterministic id BOTH ends derive (attempt 0).
    assert_eq!(
        rig.world.resource::<RequestInFlight>().0.get(&entity),
        Some(&crossing_transfer_id(
            DirectoryKey::Entity(entity),
            Fence(1),
            0
        )),
    );
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(stats.crossings_requested, 1);
    // The cooldown was armed (`last_commit_tick` set) on the emit.
    assert!(
        rig.world
            .resource::<CrossingProgress>()
            .0
            .get(&entity)
            .expect("progress")
            .last_commit_tick
            .is_some(),
        "the commit armed the cooldown",
    );
}

#[test]
fn the_in_flight_latch_suppresses_a_second_request() {
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 2);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    // Deep inside the child ⇒ the FIRST re-home fires (sets the latch + arms the cooldown).
    for t in 2..8 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    assert_eq!(crossing_requests(&all).len(), 1, "the first crossing fired");
    assert!(
        rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity)
    );
    // WITHOUT sending the Demote terminal (so the latch STAYS set), drive a SECOND container change:
    // leave the child region (container reverts to own ⇒ no re-home) long enough for the cooldown to
    // lapse, then re-enter the child (container flips back to OTHER_REALM). This second re-home
    // decision reaches `fan_out_crossing` — but the latch is still held, so it takes the SUPPRESS arm.
    for t in 8..12 {
        rig.set_local_tick(t);
        // Outside the child shell (sd 1000 > outset) but still inside own ⇒ container == own.
        move_dot(&mut rig, TRIG_SESSION, DVec3::new(2000.0, 0.0, 0.0));
        all.extend(rig.tick(vec![]));
    }
    for t in 12..18 {
        rig.set_local_tick(t);
        move_dot(&mut rig, TRIG_SESSION, DVec3::new(100.0, 0.0, 0.0)); // back inside the child
        all.extend(rig.tick(vec![]));
    }
    assert_eq!(
        crossing_requests(&all).len(),
        1,
        "the RequestInFlight latch holds it to ONE request across the second container change",
    );
    assert!(
        rig.world
            .resource::<StubStats>()
            .crossings_suppressed_in_flight
            > 0,
        "the second container change hit the in-flight SUPPRESS arm",
    );
}

#[test]
fn a_transient_crossing_emits_a_transient_crossing_request() {
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    // A Debris entity is Transient → the transient fan-out arm.
    let entity = EntityId::pack(EntityKind::Debris, 10, 1, 3);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        entity,
        Transient {
            pose: StampedPose::at_rest(
                config().frame,
                DVec3::new(100.0, 0.0, 0.0),
                UniverseTick(100),
            ),
            anchor_fence: Fence(1),
            status: TransientStatus::Held { outbound: None },
            prev_offset: LatticePos::from_metres(
                DVec3::new(100.0, 0.0, 0.0),
                vd_core::pose::Tier::Fine,
            ),
        },
    );
    // A transient carries NO per-entity latch (its batch journal dedups instead), so the ONLY
    // anti-thrash is the symmetric `k_dwell` cooldown (§2.7). Run WITHIN the cooldown window
    // (commit at tick 2, k_dwell = 5) so the container-differs re-home fires exactly once.
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..7 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    let reqs = transient_crossing_requests(&all);
    assert_eq!(reqs.len(), 1, "exactly ONE TransientCrossingRequest");
    assert_eq!(reqs[0].subject, DirectoryKey::Entity(entity));
    assert_eq!(reqs[0].to_realm, OTHER_REALM);
    assert_eq!(reqs[0].src_realm_fence, Fence(1));
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .transient_crossings_requested,
        1
    );
    // Transients are NOT latched in RequestInFlight (the batch journal dedups instead).
    assert!(rig.world.resource::<RequestInFlight>().0.is_empty());
}

#[test]
fn a_durable_tagged_transient_degrades_instead_of_panicking() {
    // REGRESSION (goal-audit L4): a Durable-TAGGED entity in the held-transient set (a kind/loop
    // mismatch — e.g. a mis-tagged batch item) reaches the durable crossing arm from the transient
    // loop, which passes `None` for the session. The arm must DEGRADE (count + emit nothing), NOT
    // panic on `subject_session.expect`, and must leave NO orphan latch.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    // A Player entity is DURABLE, but we place it (wrongly) in the transient set.
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 9);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        entity,
        Transient {
            pose: StampedPose::at_rest(
                config().frame,
                DVec3::new(100.0, 0.0, 0.0),
                UniverseTick(100),
            ),
            anchor_fence: Fence(1),
            status: TransientStatus::Held { outbound: None },
            prev_offset: LatticePos::from_metres(
                DVec3::new(100.0, 0.0, 0.0),
                vd_core::pose::Tier::Fine,
            ),
        },
    );
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..8 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![])); // must not panic
    }
    assert!(
        crossing_requests(&all).is_empty(),
        "a session-less durable subject emits NO CrossingRequest",
    );
    assert!(
        rig.world.resource::<RequestInFlight>().0.is_empty(),
        "the degradation leaves no orphan latch",
    );
    assert!(
        rig.world
            .resource::<StubStats>()
            .crossing_durable_no_session
            > 0,
        "the kind/loop mismatch is counted, not panicked",
    );
}

#[test]
fn a_dot_inside_only_its_own_realm_triggers_no_re_home() {
    // The "no re-home" case (§4): a dot inside the OWN region (System(7)) but OUTSIDE the deeper child
    // (Planet(42)) has container == OWN_REALM == the realm the shard owns it in ⇒ `should_rehome`
    // returns None. No CrossingRequest, no latch — symmetric with the empty-registry inert path.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 4);
    // Outside the child shell (sd 3000 > outset) but well inside own ⇒ container == OWN_REALM.
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(4000.0, 0.0, 0.0));
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..8 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    // Split the two emptiness checks (HR5(d): a single `assert!(a && b)` leaves the short-circuit
    // false-branch of `a` uncovered).
    assert!(
        crossing_requests(&all).is_empty(),
        "container == own ⇒ NO durable CrossingRequest",
    );
    assert!(
        transient_crossing_requests(&all).is_empty(),
        "container == own ⇒ NO TransientCrossingRequest",
    );
    assert!(
        rig.world.resource::<RequestInFlight>().0.is_empty(),
        "no re-home ⇒ no latch",
    );
}

// ---- the hand-off hold ledger (inert until armed) ---------------------------------------------

fn hold_key(seed: u64) -> (EntityId, HoldRole) {
    (
        EntityId::pack(EntityKind::Player, 1, seed, 3),
        HoldRole::Source,
    )
}

#[test]
fn open_hold_is_inert_at_a_zero_ttl() {
    // THE DISARMED ARM. A fully populated call at a zero budget leaves the ledger empty, so a shard
    // that was never given a budget behaves exactly as it did before the ledger existed.
    let mut holds = HandoffHolds::default();
    assert_eq!(
        open_hold(&mut holds, hold_key(7), Fence(2), TickId(10), 0),
        HoldOutcome::Inert
    );
    assert!(holds.0.is_empty());
}

#[test]
fn open_hold_supersedes_a_newer_fence_refuses_an_older_one_and_never_refreshes_the_budget() {
    let mut holds = HandoffHolds::default();
    let k = hold_key(7);
    assert_eq!(
        open_hold(&mut holds, k, Fence(2), TickId(10), 8),
        HoldOutcome::Opened
    );

    // A GENUINELY newer crossing replaces the hold outright — and re-anchors the budget, because it
    // is a different hand-off, not a continuation of the old one.
    assert_eq!(
        open_hold(&mut holds, k, Fence(3), TickId(14), 8),
        HoldOutcome::Superseded
    );
    assert_eq!(holds.0[&k].opened_at, TickId(14));

    // THE IMMORTALITY GUARD: a same-fence re-open (a saga re-driving its demote every tick) leaves the
    // budget anchored where it was. Without this, a wedged hand-off would keep a realm alive forever by
    // heartbeat alone — the one failure the budget exists to make impossible.
    assert_eq!(
        open_hold(&mut holds, k, Fence(3), TickId(17), 8),
        HoldOutcome::Refreshed
    );
    assert_eq!(holds.0[&k].opened_at, TickId(14));

    // A DELAYED REDELIVERY of the FIRST crossing, arriving after the second opened its hold. On an
    // at-least-once mesh this is a real delivery, not a hypothetical, and honouring it would rewind
    // the ledger to a hand-off that has already been superseded.
    assert_eq!(
        open_hold(&mut holds, k, Fence(2), TickId(19), 8),
        HoldOutcome::RefusedStale
    );
    assert_eq!(holds.0[&k].takeover_fence, Fence(3));
    assert_eq!(holds.0[&k].opened_at, TickId(14));
}

#[test]
fn close_hold_at_fence_accepts_the_matching_proof_and_refuses_a_stale_one() {
    let mut holds = HandoffHolds::default();
    let k = hold_key(7);
    // Nothing to close.
    assert!(!close_hold_at_fence(&mut holds, k, Fence(3)));

    open_hold(&mut holds, k, Fence(3), TickId(1), 8);
    // A REPLAYED take-over from the superseded crossing must not close the live hold.
    assert!(!close_hold_at_fence(&mut holds, k, Fence(2)));
    assert!(holds.0.contains_key(&k));
    // The matching proof does.
    assert!(close_hold_at_fence(&mut holds, k, Fence(3)));
    assert!(holds.0.is_empty());
}

#[test]
fn close_hold_is_unconditional_and_reports_whether_there_was_one() {
    // The terminals where the subject is simply gone have no take-over to prove.
    let mut holds = HandoffHolds::default();
    let k = hold_key(7);
    assert!(!close_hold(&mut holds, k));
    open_hold(&mut holds, k, Fence(3), TickId(1), 8);
    assert!(close_hold(&mut holds, k));
    assert!(holds.0.is_empty());
}

#[test]
fn both_ends_of_a_same_node_rehome_are_held_at_once() {
    // Why the key carries the ROLE: on a co-hosting shard a re-home hands the subject from one realm
    // to another WITHOUT leaving the machine, so the same shard is both ends. A bare entity key would
    // silently collapse the two and one end would be lost.
    let mut holds = HandoffHolds::default();
    let e = EntityId::pack(EntityKind::Player, 1, 7, 3);
    open_hold(&mut holds, (e, HoldRole::Source), Fence(2), TickId(1), 8);
    open_hold(&mut holds, (e, HoldRole::Dest), Fence(2), TickId(1), 8);
    assert_eq!(holds.0.len(), 2);
    assert!(close_hold(&mut holds, (e, HoldRole::Source)));
    assert_eq!(
        holds.0.len(),
        1,
        "closing one end leaves the other standing"
    );
}

#[test]
fn handing_over_reads_the_source_end_only_and_expires_with_the_budget() {
    // The consumer-side question, asked of the two ways it can be false — no hold at all, and a hold
    // that has aged past its budget — plus the role split (a DEST hold is not this shard handing away).
    let mut holds = HandoffHolds::default();
    let e = EntityId::pack(EntityKind::Player, 1, 7, 3);
    assert!(
        !handing_over(&holds, e, TickId(5), 4),
        "no hold, no hand-off"
    );
    open_hold(&mut holds, (e, HoldRole::Dest), Fence(2), TickId(5), 4);
    assert!(
        !handing_over(&holds, e, TickId(5), 4),
        "a DEST hold is the other end — this shard is not handing anything away"
    );
    open_hold(&mut holds, (e, HoldRole::Source), Fence(2), TickId(5), 4);
    assert!(handing_over(&holds, e, TickId(5), 4));
    assert!(
        handing_over(&holds, e, TickId(8), 4),
        "age 3 of 4 is still live"
    );
    assert!(
        !handing_over(&holds, e, TickId(9), 4),
        "past the budget the hold stops answering even before the prune sweeps it"
    );
}

#[test]
fn hold_live_covers_both_edges_and_a_backwards_clock() {
    let h = HandoffHold {
        takeover_fence: Fence(2),
        opened_at: TickId(10),
    };
    assert!(hold_live(&h, TickId(10), 4), "age 0 is live");
    assert!(
        hold_live(&h, TickId(13), 4),
        "age 3 against a budget of 4 is live"
    );
    assert!(
        !hold_live(&h, TickId(14), 4),
        "age 4 is EXPIRED — the budget is exclusive"
    );
    assert!(!hold_live(&h, TickId(99), 4));
    // A BACKWARDS clock reads as age zero rather than wrapping to a colossal age and expiring the
    // whole ledger at once.
    assert!(hold_live(&h, TickId(1), 4));
}

#[test]
fn prune_drops_exactly_the_expired_holds_and_reports_how_many() {
    let mut holds = HandoffHolds::default();
    open_hold(&mut holds, hold_key(1), Fence(2), TickId(0), 4);
    open_hold(&mut holds, hold_key(2), Fence(2), TickId(6), 4);
    // The DROPPED KEYS come back (slice F: an expired Source hold is a leaver-vanish moment
    // the caller must fan the eviction for).
    assert_eq!(prune_holds(&mut holds, TickId(8), 4), vec![hold_key(1)]);
    assert_eq!(holds.0.len(), 1);
    assert!(holds.0.contains_key(&hold_key(2)));
    // A second prune with nothing due reports nothing — the arm that proves it is not clearing.
    assert_eq!(prune_holds(&mut holds, TickId(8), 4), Vec::new());
}

#[test]
fn a_rootless_region_forest_is_a_safe_no_op() {
    // DEFENSIVE (HR5): a NON-EMPTY forest with NO ambient root (every region has a parent — a
    // malformed set that `guard_regions_nest` rejects at boot in C-5) leaves `root_realm == None`, so
    // `evaluate_one_subject` cannot seed the `container` fold and returns EARLY — no re-home, never a
    // panic. Covers the `let Some(root_realm) = ctx.root_realm else { return cur }` guard.
    let mut rig = Rig::new();
    rig.grant_realm();
    // A single region whose `parent` is `Some(..)` ⇒ NO `parent: None` root ⇒ `root_realm == None`.
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(vec![own_region()]);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 5);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::ZERO);
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..6 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    assert!(
        crossing_requests(&all).is_empty(),
        "a rootless forest cannot seed the container fold ⇒ NO re-home",
    );
}

#[test]
fn the_live_containment_scan_holds_a_dense_crowd_in_one_realm_without_a_spurious_re_home() {
    // C-6c SCALE — the "hundreds in ONE location" mandate at the DETECTOR tier. The O(subjects × regions)
    // container fold runs over the WHOLE owned crowd EVERY tick; this proves it stays bounded + CORRECT
    // at crowd scale with the REAL production neighbourhood planted. (The e2e density gate
    // `p1_volume_dense_hundreds_walk_under_invariants` runs the detector INERT — empty `RealmRegions`,
    // early-return — so the live full-scan is only exercised at N ≥ 128 HERE.)
    const CROWD: usize = 128; // the "hundreds in one location" floor (N ≥ 128)
    let mut rig = Rig::new();
    rig.grant_realm();
    // The REAL seed neighbourhood `shard.rs` boots for System 7 — {Universe, Galaxy, System 7, Planet 7,
    // Station 7} (Station 7 is System 7's first-class child, task #133), a 5-region fold per subject — NOT
    // a hand-authored fixture, so this exercises the scan the bins run. The crowd clears the Station box.
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(
        vd_physics::worldgen::realm_neighbourhood_for(0, RealmId::System(7)),
    );
    // Pack N dots into a tight ~3.5 m cube at the star (origin): every dot is well inside System 7 (r=40)
    // and ≥ ~16 m from Planet 7's centre (20,0,0) ⇒ its deepest container is System 7 == its owning realm
    // (no re-home). Overlapping positions are fine — a crowd IS hundreds in one place; the entities differ.
    for i in 0..CROWD {
        let offset = DVec3::new(
            (i % 5) as f64 - 2.0,
            ((i / 5) % 5) as f64 - 2.0,
            ((i / 25) % 5) as f64 - 2.0,
        );
        insert_owned_dot(
            &mut rig,
            SessionId(i as u128),
            EntityId::pack(EntityKind::Player, i as u32, 1, i as u32),
            offset,
        );
    }
    // Re-scan the whole crowd for several ticks.
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 0..5 {
        rig.set_local_tick(2 + t);
        all.extend(rig.tick(vec![]));
    }
    // The scan TOUCHED every dot (membership computed for all N) — proof the full-scan RAN at scale, not
    // an inert early-return (an empty `RealmRegions` leaves this map empty).
    assert_eq!(
        rig.world.resource::<ContainmentProgress>().0.len(),
        CROWD,
        "the live containment scan evaluated all {CROWD} dots (not an inert early-return)",
    );
    // And it stayed CORRECT at scale: the whole crowd is contained in System 7 ⇒ ZERO re-homes fire.
    let spurious = crossing_requests(&all).len();
    assert_eq!(
        spurious, 0,
        "a dense crowd inside one realm triggers NO spurious re-home under the O(N×regions) scan",
    );
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

#[test]
fn a_dot_moving_into_the_station_box_re_homes_into_the_first_class_station_realm() {
    // THE task #133 headline: the SAME kind-agnostic containment detector re-homes a dot into a
    // first-class STATION realm with ZERO station-specific code (HR3). The shard OWNS System 7 and boots
    // the REAL seed neighbourhood (now {Universe, Galaxy, System 7, Planet 7, STATION 7}). A dot starts
    // at the origin (container == System 7 == owning ⇒ NO re-home), then walks into the Station BOX at
    // (-25,0,0) (a Cartesian `Aabb`, not an SOI shell) — its deepest container flips to Station 7 ≠ the
    // owning System 7, so ONE re-home fires whose `to_realm` is the Station. The box `signed_distance`
    // feeds the identical `ContainmentBand` the shells use — the Station is detected by geometry alone.
    let mut rig = Rig::new(); // owns System 7 (config().realm)
    rig.grant_realm();
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(
        vd_physics::worldgen::realm_neighbourhood_for(0, RealmId::System(7)),
    );
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 33);
    // Origin: well inside System 7 (r=40), clear of the Station box (x∈[-30,-20]) ⇒ container == System 7.
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::ZERO);
    rig.set_local_tick(2);
    let at_origin = crossing_requests(&rig.tick(vec![]));
    assert_eq!(
        at_origin.len(),
        0,
        "a dot at the origin is contained in System 7 (== owning) ⇒ NO re-home",
    );
    // Walk INTO the Station box centre — the deepest container becomes Station 7.
    let mut into: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 3..8 {
        rig.set_local_tick(t);
        move_dot(&mut rig, TRIG_SESSION, DVec3::new(-25.0, 0.0, 0.0)); // the Station box centre
        into.extend(rig.tick(vec![]));
    }
    let reqs = crossing_requests(&into);
    assert_eq!(
        reqs.len(),
        1,
        "exactly ONE re-home into the Station (latched thereafter)"
    );
    assert_eq!(
        reqs[0].to_realm, STATION_A,
        "the kind-agnostic detector re-homes into the first-class Station realm",
    );
    assert_eq!(
        reqs[0].from_realm,
        config().realm,
        "leaving the owning System 7"
    );
    assert_eq!(reqs[0].subject, DirectoryKey::Entity(entity));
}

#[test]
fn a_station_owning_shard_re_homes_a_dot_that_leaves_the_station_back_to_system_7() {
    // The RETURN leg of the round-trip, proven as a GENUINE emission (symmetric detector, no direction):
    // a shard that OWNS Station 7 boots Station 7's seed neighbourhood — its own realm + its ANCESTOR
    // chain {System 7, Galaxy, Universe}. A dot INSIDE the Station box is contained in Station 7 (==
    // owning ⇒ NO re-home); when it LEAVES the box (to the origin, still inside System 7's r=40 SOI) its
    // deepest container becomes System 7 ≠ the owning Station 7, so ONE re-home fires whose `to_realm` is
    // System 7. This is the same machinery as the inbound gate above, run from the Station's authority —
    // the "both ways" proof that a Station is a first-class realm on the identical kind-agnostic path.
    let station_cfg = StubConfig {
        realm: STATION_A,
        held_realms: StubConfig::single_realm(STATION_A),
        frame: FrameRef::StationLocal { station_seed: 7 },
        // A REAL lineage: this shard hosts Station 7 INSIDE System 7. It used to inherit a ROOT coord
        // while declaring a Station realm — realm and lineage disagreeing — which was harmless only
        // while leaving was a search through the ancestors. Now that leaving hands UP, a shard's own
        // lineage is how it knows who to hand to, so a stub lineage means it hands to nobody.
        own_coord: child_coord_of(RealmId::System(7), STATION_A),
        ..config()
    };
    let station_frame = station_cfg.frame; // FrameRef is Copy — capture before the config move
    let mut rig = Rig::with_config(station_cfg);
    grant_realm_for(&mut rig, STATION_A);
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vd_physics::worldgen::realm_neighbourhood_for(0, STATION_A));
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 34);
    // Inside the Station box, at its CENTRE — which in the Station's OWN frame is zero, because every
    // realm is centred on itself. This used to read -25: the station's position in SYSTEM 7's frame,
    // written on a pose stamped in the STATION's frame. The two spellings agreed on nothing except
    // while the containment maths subtracted the station's position a second time and cancelled the
    // error. An occupant of a station is a few metres from its middle, whatever the station's address.
    insert_owned_dot_framed(&mut rig, TRIG_SESSION, entity, station_frame, DVec3::ZERO);
    rig.set_local_tick(2);
    let inside = crossing_requests(&rig.tick(vec![]));
    assert_eq!(
        inside.len(),
        0,
        "a dot inside the Station box is contained in Station 7 (== owning) ⇒ NO re-home",
    );
    // LEAVE the Station box, heading for the star. The station sits 25 m along -X of System 7's centre,
    // so System 7's centre is +25 in the STATION's own frame — well outside the box's ±5, still deep
    // inside System 7's 40 m reach. The dot is now inside nothing this shard holds ⇒ it has left, and
    // the destination is the shard's parent.
    let mut out: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 3..8 {
        rig.set_local_tick(t);
        move_dot(&mut rig, TRIG_SESSION, DVec3::new(25.0, 0.0, 0.0)); // out of the box, toward the star
        out.extend(rig.tick(vec![]));
    }
    let reqs = crossing_requests(&out);
    assert_eq!(
        reqs.len(),
        1,
        "exactly ONE re-home back to System 7 (latched thereafter)"
    );
    assert_eq!(
        reqs[0].to_realm,
        RealmId::System(7),
        "leaving the Station re-homes back to the enclosing System 7 (symmetric, same detector)",
    );
    assert_eq!(
        reqs[0].from_realm, STATION_A,
        "leaving the owning Station 7"
    );
    assert_eq!(reqs[0].subject, DirectoryKey::Entity(entity));
}

#[test]
fn a_dot_moving_into_the_area_box_re_homes_into_the_first_class_area_realm() {
    // The AREA analog of the Station inbound gate (task #133), proving the SAME kind-agnostic detector
    // re-homes into a first-class AREA realm — the DEEPEST region in the seed forest (depth 4). A shard
    // that OWNS Planet 7 boots Planet 7's seed neighbourhood (its own realm + ancestors {System 7,
    // Galaxy, Universe} + its child AREA 7). A dot starts at Planet 7's centre (20,0,0) (container ==
    // Planet 7 == owning ⇒ NO re-home — this is ALSO the escape-SOI probe point, which must NOT resolve
    // to the Area), then walks into the Area BOX at (25,0,0) — its deepest container flips to Area 7 ≠
    // the owning Planet 7, so ONE re-home fires whose `to_realm` is the Area. Zero area-specific code.
    let planet_cfg = StubConfig {
        realm: RealmId::Planet(7),
        held_realms: StubConfig::single_realm(RealmId::Planet(7)),
        frame: FrameRef::PlanetCentered { planet_seed: 7 },
        ..config()
    };
    let planet_frame = planet_cfg.frame; // FrameRef is Copy — capture before the config move
    let mut rig = Rig::with_config(planet_cfg);
    grant_realm_for(&mut rig, RealmId::Planet(7));
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(
        vd_physics::worldgen::realm_neighbourhood_for(0, RealmId::Planet(7)),
    );
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 35);
    // Planet 7's centre, which in the PLANET'S OWN frame is zero — every realm is centred on itself.
    // Inside Planet 7 (r=10) and OUTSIDE the Area box (which sits at +5 from the planet, half 3, so
    // x∈[2,8]) ⇒ container == Planet 7 == owning ⇒ NO re-home. This used to read (20,0,0): the planet's
    // position in SYSTEM 7's frame, written on a pose stamped in the planet's own frame.
    insert_owned_dot_framed(&mut rig, TRIG_SESSION, entity, planet_frame, DVec3::ZERO);
    rig.set_local_tick(2);
    let at_centre = crossing_requests(&rig.tick(vec![]));
    assert_eq!(
        at_centre.len(),
        0,
        "the dot at Planet 7's centre is contained in Planet 7 (== owning) ⇒ NO re-home",
    );
    // Walk INTO the Area box centre — +5 along X in the PLANET's own frame, still well inside the
    // planet's 10 m sphere and squarely in the box ⇒ the deepest container becomes Area 7.
    let mut into: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 3..8 {
        rig.set_local_tick(t);
        move_dot(&mut rig, TRIG_SESSION, DVec3::new(5.0, 0.0, 0.0)); // the Area box centre
        into.extend(rig.tick(vec![]));
    }
    let reqs = crossing_requests(&into);
    assert_eq!(
        reqs.len(),
        1,
        "exactly ONE re-home into the Area (latched thereafter)"
    );
    assert_eq!(
        reqs[0].to_realm,
        RealmId::Area(7),
        "the kind-agnostic detector re-homes into the first-class Area realm (the deepest region)",
    );
    assert_eq!(
        reqs[0].from_realm,
        RealmId::Planet(7),
        "leaving the owning Planet 7"
    );
    assert_eq!(reqs[0].subject, DirectoryKey::Entity(entity));
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

#[test]
fn a_cohosting_shard_affirms_the_child_realm_head_into_its_own_authority_map() {
    // The multi-realm AFFIRM path (co-hosting): a shard co-hosting Planet 7 must land the Planet-7
    // realm-head reply in `CoHostedAuthority` (independent of the primary `RealmAuthority`), so the
    // short-circuit's `held_here` answers `Some` for the child. This is the grant/affirm half of the cure.
    let mut rig = Rig::with_config(cohost_planet_config());
    grant_realm_for(&mut rig, RealmId::System(7));
    assert_eq!(
        rig.world.resource::<RealmAuthority>().0,
        Some(Fence(1)),
        "the primary System-7 realm is held on RealmAuthority",
    );
    assert!(
        !rig.world
            .resource::<CoHostedAuthority>()
            .0
            .contains_key(&RealmId::Planet(7)),
        "the child is NOT held until its own head affirms",
    );
    grant_realm_for(&mut rig, RealmId::Planet(7));
    assert_eq!(
        rig.world
            .resource::<CoHostedAuthority>()
            .0
            .get(&RealmId::Planet(7)),
        Some(&Fence(1)),
        "the co-hosted Planet-7 head lands in CoHostedAuthority (not RealmAuthority)",
    );
    assert_eq!(
        rig.world.resource::<RealmAuthority>().0,
        Some(Fence(1)),
        "the child affirm leaves the primary realm untouched",
    );
    // RE-AFFIRM the SAME child at a NEWER fence: the child is ALREADY in `CoHostedAuthority`, so this hits
    // the `else if ours` REFRESH arm (not the first insert). The stored fence must update to the refreshed
    // value — the periodic round-trip re-arming a still-held co-hosted child head.
    grant_realm_for_at(&mut rig, RealmId::Planet(7), Fence(2));
    assert_eq!(
        rig.world
            .resource::<CoHostedAuthority>()
            .0
            .get(&RealmId::Planet(7)),
        Some(&Fence(2)),
        "the re-affirm REFRESHES the co-hosted Planet-7 fence to the newer value",
    );
    assert_eq!(
        rig.world.resource::<RealmAuthority>().0,
        Some(Fence(1)),
        "the child re-affirm still leaves the primary realm untouched",
    );
}

#[test]
fn a_revoked_cohosted_child_head_is_dropped_from_the_cohost_authority_map() {
    // The FOREIGN/None `else` arm of `affirm_realm_head` (`cohosted.0.remove(&realm)`): a co-hosted CHILD
    // realm this shard held is TAKEN OVER or REVOKED (here `record: None` — the reaper dropped the lease),
    // so its `CoHostedAuthority` entry is DROPPED (no transient loss — a child never anchors this shard's
    // transients; those ride the PRIMARY `RealmAuthority`). This is distinct from the primary self-fence
    // (which drops `RealmAuthority` + declares transients lost). Boot with Planet 7 HELD, then revoke it.
    let mut rig = Rig::with_config(cohost_planet_config());
    boot_cohost_planet(&mut rig); // primary System 7 + co-hosted child Planet 7 both held
    assert_eq!(
        rig.world
            .resource::<CoHostedAuthority>()
            .0
            .get(&RealmId::Planet(7)),
        Some(&Fence(1)),
        "precondition: the co-hosted Planet-7 head is held",
    );
    revoke_realm_for(&mut rig, RealmId::Planet(7)); // record None ⇒ not ours ⇒ the remove arm
    assert!(
        !rig.world
            .resource::<CoHostedAuthority>()
            .0
            .contains_key(&RealmId::Planet(7)),
        "the revoked co-hosted child is DROPPED from CoHostedAuthority",
    );
    // The PRIMARY realm is UNTOUCHED — the child-revoke path never self-fences the primary lease.
    assert_eq!(
        rig.world.resource::<RealmAuthority>().0,
        Some(Fence(1)),
        "revoking the co-hosted child leaves the primary System-7 realm held",
    );
}

#[test]
fn a_dot_re_homing_into_a_cohosted_child_emits_a_crossing_request_with_its_parent() {
    // THE UNIVERSAL re-home assertion (task #149, re-baselined from the old relabel test): a durable dot
    // that walks from System 7 into CO-HOSTED Planet 7's SOI emits the SAME `CrossingRequest` as a
    // foreign crossing — there is NO local short-circuit. `head(Realm(Planet 7))` resolves to THIS node
    // (source==dest), which the ONE orchestrator saga handles as the degenerate case. The request carries
    // Planet 7 as `to_realm` and its enclosing System 7 as `to_parent` — a WIRE-SHAPE pin only: the field
    // is DEAD (its consumer `rebind_pose_to_dest` is deleted, D-PLACE-1/D-WIRE-1; the dest forms the child
    // frame from its own ROSTER via `arrival_frame`). The pose is NOT rewritten in place here (the
    // detector only requests); the frame flips at the dest's adopt (same node on a co-hosted re-home).
    let mut rig = Rig::with_config(cohost_planet_config());
    boot_cohost_planet(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 51);
    // Start at the origin — inside System 7 (r=40), OUTSIDE Planet 7 (centre 20, r=10) ⇒ container ==
    // System 7 == owning ⇒ NO re-home.
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::ZERO);
    rig.set_local_tick(2);
    let at_origin = crossing_requests(&rig.tick(vec![]));
    assert_eq!(
        at_origin.len(),
        0,
        "at the origin the dot is in System 7 (== owning)"
    );
    // Walk INTO Planet 7's centre (20,0,0) — its deepest container flips to Planet 7, a realm THIS
    // shard CO-HOSTS ⇒ ONE CrossingRequest (the uniform saga; a co-hosted dest is source==dest).
    let mut out: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 3..8 {
        rig.set_local_tick(t);
        move_dot(&mut rig, TRIG_SESSION, DVec3::new(20.0, 0.0, 0.0));
        out.extend(rig.tick(vec![]));
    }
    let reqs = crossing_requests(&out);
    assert_eq!(
        reqs.len(),
        1,
        "a re-home into a CO-HOSTED child emits ONE CrossingRequest (source==dest — the uniform saga)",
    );
    assert_eq!(
        reqs[0].to_realm,
        RealmId::Planet(7),
        "the crossing targets the co-hosted Planet 7 realm",
    );
    assert_eq!(
        reqs[0].to_parent,
        Some(RealmId::System(7)),
        "the request carries Planet 7's enclosing System 7 as to_parent (so the dest's Area/child frame forms)",
    );
}

#[test]
fn a_dot_re_homing_into_a_non_cohosted_child_emits_a_crossing_request() {
    // The SAME co-hosting shard, a dot walking into Station 7 — a child it does NOT co-host (`held_realms`
    // is `{System(7), Planet(7)}`, no Station). Post-task-#149 this is IDENTICAL to the co-hosted case:
    // ONE `CrossingRequest` (the node-placement branch is deleted — held-here vs foreign no longer
    // matters, both are the uniform saga). Kept as a second geometry to prove the request fires for any
    // container change, co-hosted or not.
    let mut rig = Rig::with_config(cohost_planet_config());
    boot_cohost_planet(&mut rig);
    // The region union for {System 7, Planet 7} DOES include Station 7 (a child of the held System 7).
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 52);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::ZERO);
    let mut out: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 3..9 {
        rig.set_local_tick(t);
        move_dot(&mut rig, TRIG_SESSION, DVec3::new(-25.0, 0.0, 0.0)); // Station 7 box centre
        out.extend(rig.tick(vec![]));
    }
    let reqs = crossing_requests(&out);
    assert_eq!(
        reqs.len(),
        1,
        "a re-home into a NON-co-hosted child emits ONE CrossingRequest (the uniform saga)",
    );
    assert_eq!(
        reqs[0].to_realm,
        RealmId::Station(7),
        "the crossing targets the Station 7 realm",
    );
    assert_eq!(
        reqs[0].to_parent,
        Some(RealmId::System(7)),
        "the request carries Station 7's enclosing System 7 as to_parent",
    );
}

#[test]
fn a_held_transient_re_homing_into_a_cohosted_child_emits_a_transient_request_with_its_parent() {
    // The TRANSIENT twin (HR2 — the SAME machinery, no per-kind fork, no local short-circuit): a held
    // Debris transient inside CO-HOSTED Planet 7 emits ONE `TransientCrossingRequest` carrying Planet 7's
    // enclosing System 7 as `to_parent` — a WIRE-SHAPE pin only (the field is DEAD: its consumer is
    // deleted, D-PLACE-1/D-WIRE-1; the dest places the pose from its own roster at adopt). The
    // pose is not rewritten in place; the frame flips at the dest's adopt.
    let mut rig = Rig::with_config(cohost_planet_config());
    boot_cohost_planet(&mut rig);
    let entity = EntityId::pack(EntityKind::Debris, 10, 1, 61);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        entity,
        Transient {
            // Inside Planet 7's SOI (centre x=20, r=10) — container == Planet 7 (co-hosted).
            pose: StampedPose::at_rest(
                config().frame,
                DVec3::new(20.0, 0.0, 0.0),
                UniverseTick(100),
            ),
            anchor_fence: Fence(1),
            status: TransientStatus::Held { outbound: None },
            prev_offset: LatticePos::from_metres(
                DVec3::new(20.0, 0.0, 0.0),
                vd_core::pose::Tier::Fine,
            ),
        },
    );
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..7 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    let reqs = transient_crossing_requests(&all);
    assert_eq!(
        reqs.len(),
        1,
        "a transient re-home into a CO-HOSTED child emits ONE TransientCrossingRequest (the uniform saga)",
    );
    assert_eq!(reqs[0].to_realm, RealmId::Planet(7));
    assert_eq!(
        reqs[0].to_parent,
        Some(RealmId::System(7)),
        "the transient request carries Planet 7's enclosing System 7 as to_parent",
    );
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .transient_crossings_requested,
        1,
    );
}

#[test]
fn an_aborted_re_home_re_fires_while_the_dot_is_still_in_the_region() {
    // POSITIVE proof of the abort RE-FIRE (`on_crossing_aborted` resets `last_commit_tick=None`): a dot
    // whose crossing ABORTED is STILL geometrically in the deeper region, so `container != owning` and
    // `should_rehome` fires AGAIN with a FRESH id — the re-home self-heals without a physical re-cross.
    // A green tree that DISARMS the detector during the abort (the e2e tests) cannot catch a broken
    // re-fire; this drives the abort WITH the regions still armed and asserts the attempt-1 re-emit.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 9);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::ZERO); // inside the child ⇒ re-homes
    rig.set_local_tick(2);
    let first = crossing_requests(&rig.tick(vec![]));
    assert_eq!(first.len(), 1, "the in-region dot re-homes once");
    assert_eq!(first[0].attempt, 0, "the first re-home is attempt 0");
    // The MATCHING abort clears the latch + resets the cooldown so the re-home can re-fire.
    let transfer = crossing_transfer_id(DirectoryKey::Entity(entity), Fence(1), 0);
    let mut after: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    rig.set_local_tick(3);
    after.extend(rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::CrossingAborted(CrossingAborted {
            subject: DirectoryKey::Entity(entity),
            transfer,
        }),
    )]));
    rig.set_local_tick(4);
    after.extend(rig.tick(vec![])); // still in the region ⇒ re-fires with a fresh id
    let refired = crossing_requests(&after).iter().any(|r| r.attempt == 1);
    assert!(
        refired,
        "after the abort the still-in-region dot RE-FIRES with attempt==1 (self-heal)",
    );
}

#[test]
fn a_dot_jittering_across_a_region_surface_within_the_band_never_re_homes() {
    // The BAND (not the latch) proves anti-flap. A dot whose signed distance to the child region
    // oscillates ACROSS the surface (sd ∈ [-40, +40]) but stays inside the acquire-hysteresis dead-zone
    // (acquire only at sd ≤ -inset = -50) NEVER acquires membership, so `container` stays the OWN realm
    // and ZERO re-homes fire — latch-INDEPENDENT (no re-home ⇒ no latch to mask a flap). A broken band
    // (naive point membership `sd ≤ 0`) would acquire the child on every sd<0 tick and re-home.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig); // child (Planet 42) at origin, r=1000, band inset=50/outset=100
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 11);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(1040.0, 0.0, 0.0)); // sd_child = +40
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    // Jitter x ∈ [960, 1040] ⇒ sd_child ∈ [-40, +40], crossing the surface but never reaching -inset,
    // for well over `k_dwell` ticks.
    for (i, &x) in [960.0, 1040.0, 970.0, 1030.0, 965.0, 1035.0, 962.0, 1038.0]
        .iter()
        .enumerate()
    {
        rig.set_local_tick(2 + i as u64);
        move_dot(&mut rig, TRIG_SESSION, DVec3::new(x, 0.0, 0.0));
        all.extend(rig.tick(vec![]));
    }
    assert!(
        crossing_requests(&all).is_empty(),
        "the acquire-hysteresis dead-zone keeps the dot a non-member ⇒ ZERO re-homes (band, not latch)",
    );
}

#[test]
fn an_empty_registry_or_no_realm_triggers_nothing() {
    // (a) EMPTY region registry (the production-inert path): no candidates, no emit, no state.
    let mut rig = Rig::new();
    rig.grant_realm();
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 5);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..8 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    assert!(crossing_requests(&all).is_empty());
    assert!(rig.world.resource::<RequestInFlight>().0.is_empty());
    assert!(rig.world.resource::<CrossingProgress>().0.is_empty());

    // (b) NO realm authority (never granted): the authority gate short-circuits BEFORE any work —
    // even with a populated registry + an in-band dot. A NON-ZERO `request_ttl_ticks` here so the
    // 3f-D4 `redrive_stranded_crossings` scan ALSO reaches (and covers) its authority-gate `else`
    // arm (past its own `ttl==0` early-return), mirroring the trigger's gate.
    let mut rig2 = Rig::with_config(config_with_ttl(3));
    plant_dock_regions(&mut rig2);
    // A dot cannot normally exist without a realm, but the trigger's gate must be authority-first:
    // force one in and confirm the early-return fires.
    insert_owned_dot(&mut rig2, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
    assert_eq!(rig2.world.resource::<RealmAuthority>().0, None);
    let mut all2: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..8 {
        rig2.set_local_tick(t);
        all2.extend(rig2.tick(vec![]));
    }
    assert!(
        crossing_requests(&all2).is_empty(),
        "no realm authority ⇒ the trigger + the ttl re-drive both early-return",
    );
    assert!(rig2.world.resource::<CrossingProgress>().0.is_empty());
    assert_eq!(
        rig2.world.resource::<StubStats>().crossings_redriven,
        0,
        "no realm authority ⇒ the ttl scan never re-drives",
    );
}

#[test]
fn eviction_drops_state_for_a_departed_subject() {
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 6);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
    // Deep inside the child ⇒ a re-home fires, so ALL FOUR per-entity ledgers hold a row (the
    // ContainmentProgress bitset is written by the full-scan every tick a subject is evaluated).
    for t in 2..8 {
        rig.set_local_tick(t);
        let _ = rig.tick(vec![]);
    }
    assert!(
        rig.world
            .resource::<CrossingProgress>()
            .0
            .contains_key(&entity)
    );
    assert!(
        rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity)
    );
    assert!(
        rig.world
            .resource::<ContainmentProgress>()
            .0
            .contains_key(&entity),
        "the per-region membership bitset holds a row for the evaluated subject",
    );
    // The dot logs out (removed): next evaluation tick evicts its per-entity state (the DRY retain).
    rig.world.resource_mut::<Dots>().0.remove(&TRIG_SESSION);
    rig.set_local_tick(8);
    let _ = rig.tick(vec![]);
    assert!(
        !rig.world
            .resource::<CrossingProgress>()
            .0
            .contains_key(&entity)
    );
    assert!(
        !rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity)
    );
    assert!(
        !rig.world
            .resource::<ContainmentProgress>()
            .0
            .contains_key(&entity)
    );
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

/// ★ SLICE S5's ACCEPTANCE LINE, AT THE SCAN. A subject that clears a child's WHOLE DIAMETER inside
/// one tick still acquires it and still re-homes into it. Before the verdict tested the tick's motion
/// this was impossible in principle, not merely unlikely: acquisition needed a SAMPLE landing at least
/// one inset INSIDE the surface, so no widening of any band could ever buy it.
///
/// The anti-vacuity guarantee is stated as a measurement rather than assumed: BOTH endpoint poses are
/// asserted to sit outside the child, so the point rule cannot have decided this.
#[test]
fn a_subject_that_clears_a_whole_child_in_one_tick_still_re_homes_into_it() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let child = child_region(); // a 1000 m shell at the origin
    let to_realm = child.realm;
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), own_region(), child]);
    let entity = EntityId::pack(EntityKind::Player, 11, 1, 0x4B);

    // Tick 1 — well outside on one side. This is the tick that RECORDS the prior.
    let before = DVec3::new(-9_000.0, 0.0, 0.0);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, before);
    rig.set_local_tick(2);
    let _ = rig.tick(vec![]);

    // Tick 2 — well outside on the OTHER side, 18 km later: nine child diameters in one tick.
    let after = DVec3::new(9_000.0, 0.0, 0.0);
    rig.set_local_tick(3);
    move_dot(&mut rig, TRIG_SESSION, after);
    let sent = rig.tick(vec![]);

    // ANTI-VACUITY: neither sample is inside the child, so a point rule decides "never a member".
    assert!(
        child_signed_distance(&child, before) > 0.0,
        "the prior sample must be outside the child"
    );
    assert!(
        child_signed_distance(&child, after) > 0.0,
        "the current sample must be outside the child"
    );

    let reqs: Vec<CrossingRequest> = crossing_requests(&sent)
        .into_iter()
        .filter(|r| r.to_realm == to_realm)
        .collect();
    assert_eq!(
        reqs.len(),
        1,
        "the tick's motion crossed the child, so exactly one re-home into it is owed"
    );
}

/// AND IT IS NOT A BLANKET YES. The same jump, offset so the path misses the child, must produce
/// nothing — otherwise the arm above would pass for a verdict that simply said "member" at speed.
#[test]
fn the_same_jump_one_child_radius_off_the_path_re_homes_nowhere() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let child = child_region();
    let to_realm = child.realm;
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), own_region(), child]);
    let entity = EntityId::pack(EntityKind::Player, 12, 1, 0x4C);
    // Same 18 km jump, displaced 2 km off-axis — outside the 1000 m child at every point.
    let before = DVec3::new(-9_000.0, 2_000.0, 0.0);
    let after = DVec3::new(9_000.0, 2_000.0, 0.0);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, before);
    rig.set_local_tick(2);
    let _ = rig.tick(vec![]);
    rig.set_local_tick(3);
    move_dot(&mut rig, TRIG_SESSION, after);
    let sent = rig.tick(vec![]);
    let reqs: Vec<CrossingRequest> = crossing_requests(&sent)
        .into_iter()
        .filter(|r| r.to_realm == to_realm)
        .collect();
    assert!(
        reqs.is_empty(),
        "a path that misses the child must acquire nothing: {reqs:?}"
    );
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

/// The HR4/G-IDENTICAL extension proof (the placement arc S6): the IDENTICAL swept-crossing
/// fixture, run with a KEPLER child and an INTEGRATED (ballistic burn) child, on TWO shard kinds —
/// each run commits exactly ONE re-home with identical subject/source/destination/attempt. The
/// motions are built in the motion crate (a dev-dependency: fixtures may plant real motion; the
/// shipped crate cannot name one) and enter ONLY as opaque closures, so the fixture is measurably
/// blind to HOW the child moves and to WHAT KIND of shard runs it.
#[test]
fn a_kepler_child_and_a_thrusting_child_cross_by_identical_code() {
    use vd_physics::motion::{Motion, motion_fn};
    let tick_hz = 20.0;
    // KEPLER: a circular 1000 m orbit phased so the child sits far off the dot at tick 100
    // (√2·1000 m away) and EXACTLY on it at tick 160 (M(8 s) ≡ 0 ⇒ position (1000, 0, 0)).
    let n = std::f64::consts::FRAC_PI_2 / 3.0; // rad/s across the 3 s sweep window
    let kepler = motion_fn(Motion::Kepler(OrbitalElements {
        sma: 1000.0,
        ecc: 0.0,
        inclination: 0.0,
        raan: 0.0,
        arg_periapsis: 0.0,
        mean_anomaly_epoch: -n * (160.0 / tick_hz),
        central_mass: n * n * 1000.0_f64.powi(3) / vd_physics::celestial::G,
    }));
    // INTEGRATED: a ballistic burn along -Y that carries the child from 1200 m off the dot at
    // tick 100 onto it at tick 160 (y(t) = 1600 + 120·t − 40·t²: y(5) = 1200, y(8) = 0).
    let burn = motion_fn(Motion::Integrated {
        placement: FramePlacement::moving(
            DVec3::new(1000.0, 1600.0, 0.0),
            DVec3::new(0.0, 120.0, 0.0),
        ),
        acceleration: DVec3::new(0.0, -80.0, 0.0),
    });

    let (k_reqs, k_start, k_end) = drive_swept_crossing_feature(kepler, NodeKind::StubShard);
    let planet = crate::capability::profiles::planet().expect("planet profile");
    let (b_reqs, b_start, b_end) = drive_swept_crossing_feature(burn, NodeKind::Shard(planet));

    // Anti-vacuity: both children genuinely swept from far OUTSIDE the shell onto the dot.
    assert!(k_start > 200.0);
    assert!(b_start > 200.0);
    assert!(k_end < 1.0);
    assert!(b_end < 1.0);
    // Exactly ONE re-home each — the latch suppressed every later tick of the dwell.
    assert_eq!(k_reqs.len(), 1, "the Kepler sweep commits one re-home");
    assert_eq!(b_reqs.len(), 1, "the burn sweep commits one re-home");
    // IDENTICAL crossing, field by field: the code cannot tell an orbit from a burn, and cannot
    // tell a stub shard from a planet-profile shard (HR4's G-IDENTICAL, both axes at once).
    assert_eq!(k_reqs[0].subject, b_reqs[0].subject);
    assert_eq!(k_reqs[0].from_realm, b_reqs[0].from_realm);
    assert_eq!(k_reqs[0].to_realm, b_reqs[0].to_realm);
    assert_eq!(k_reqs[0].attempt, b_reqs[0].attempt);
    assert_eq!(k_reqs[0].session, b_reqs[0].session);
}

/// DISCHARGES D-38: the G-IDENTICAL assert_feature_anywhere — ONE containment re-home fixture,
/// identical whether the deeper child region is a Spherical (Shell) or a Cartesian (Aabb) volume.
#[test]
fn assert_feature_anywhere() {
    // Run (a): a Spherical PLANET profile ⇔ a Shell child region (SOI descent). The dot descends the
    // diagonal across the shell's acquire edge and commits exactly ONE re-home to OTHER_REALM.
    let planet = crate::capability::profiles::planet().expect("planet profile");
    assert_eq!(
        planet.voxel(),
        Some(crate::capability::VoxelGeometry::Spherical),
        "the Shell run is tied to the Spherical profile",
    );
    let (shell_reqs, shell_start_sd, shell_end_sd, shell_shape) =
        drive_inward_crossing_feature(child_region_shell);
    // Spherical ⇔ Shell: the run's region IS a Shell (compared by equality, not `matches!`, so there
    // is no uncoverable false arm — HR5(d)). `child_region` uses r 1000.
    assert_eq!(shell_shape, Boundary::Shell { r: 1000.0 });
    // Run (b): a Cartesian STATION profile ⇔ an Aabb child region (station volume). The IDENTICAL body
    // drives the SAME diagonal descent across the box's acquire edge → exactly ONE re-home.
    let station = crate::capability::profiles::station().expect("station profile");
    assert_eq!(
        station.voxel(),
        Some(crate::capability::VoxelGeometry::Cartesian),
        "the Aabb run is tied to the Cartesian profile",
    );
    let (aabb_reqs, aabb_start_sd, aabb_end_sd, aabb_shape) =
        drive_inward_crossing_feature(child_region_aabb);
    // Cartesian ⇔ Aabb: the run's region IS an Aabb (equality, no `matches!` false arm — HR5(d)).
    assert_eq!(
        aabb_shape,
        Boundary::Aabb {
            half: DVec3::new(200.0, 200.0, 200.0),
        }
    );

    // (i) EXACTLY ONE re-home CrossingRequest per run, whose destination is the child's realm — the
    // box/shell actually GATED the re-home (not a bare count of "something fired").
    assert_eq!(
        shell_reqs.len(),
        1,
        "the Shell run emits exactly one re-home"
    );
    assert_eq!(aabb_reqs.len(), 1, "the Aabb run emits exactly one re-home");
    assert_eq!(
        shell_reqs[0].to_realm, OTHER_REALM,
        "the Shell re-home gated to OTHER_REALM"
    );
    assert_eq!(
        aabb_reqs[0].to_realm, OTHER_REALM,
        "the Aabb re-home gated to OTHER_REALM"
    );
    // IDENTICAL feature behavior across the two profiles: same subject, same source realm, same
    // destination, same attempt. The two runs differ ONLY in region shape, never in the feature.
    assert_eq!(shell_reqs[0].subject, aabb_reqs[0].subject);
    assert_eq!(shell_reqs[0].from_realm, aabb_reqs[0].from_realm);
    assert_eq!(shell_reqs[0].to_realm, aabb_reqs[0].to_realm);
    assert_eq!(shell_reqs[0].attempt, aabb_reqs[0].attempt);
    assert_eq!(shell_reqs[0].session, aabb_reqs[0].session);

    // (ii) Anti-vacuity: the signed distance ACTUALLY crossed the acquire edge in BOTH runs — the dot
    // started OUTSIDE the surface (sd > 0) and ended at least `inset` (50 m) INSIDE (sd <= -inset), a
    // real membership acquire, never a no-op that fired on an already-inside dot.
    assert!(
        shell_start_sd > 0.0,
        "Shell start OUTSIDE the surface (sd {shell_start_sd})"
    );
    assert!(
        shell_end_sd <= -50.0,
        "Shell end at least the inset INSIDE (sd {shell_end_sd})"
    );
    assert!(
        aabb_start_sd > 0.0,
        "Aabb start OUTSIDE the surface (sd {aabb_start_sd})"
    );
    assert!(
        aabb_end_sd <= -50.0,
        "Aabb end at least the inset INSIDE (sd {aabb_end_sd})"
    );
}

/// Slice 4a: the ledger-eviction-on-leave gate (the InputLog-leak class). The `retain_live` machinery
/// already exists (`stub.rs` `evaluate_realm_boundaries`); this PROVES all THREE per-entity ledgers
/// evicted by the detector (CrossingProgress + RequestInFlight + ContainmentProgress — the C-3 bitset
/// is the 4th `retain_live` monomorphization that REPLACES the deleted `InterestZones`) are evicted
/// when the subject leaves. Uses `assert_eq!(.get(), None)` (not `assert!(matches!)`) so the None
/// equality is the covered arm.
#[test]
fn slice4a_all_three_ledgers_evict_when_the_subject_leaves() {
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 0x4B);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
    // Deep inside the child ⇒ a durable re-home commits, so CrossingProgress (cooldown), RequestInFlight
    // (latch) AND ContainmentProgress (bitset) all hold a row for the subject.
    for t in 2..8 {
        rig.set_local_tick(t);
        let _ = rig.tick(vec![]);
    }
    // All three ledgers now hold an entry (setup pre-conditions; the load-bearing asserts are the
    // three `None` equalities after the subject leaves).
    assert!(
        rig.world
            .resource::<CrossingProgress>()
            .0
            .contains_key(&entity)
    );
    assert!(
        rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity)
    );
    assert!(
        rig.world
            .resource::<ContainmentProgress>()
            .0
            .contains_key(&entity)
    );
    // The dot logs out (removed from Dots). Not in OwnedTransients either, so `live` excludes it.
    rig.world.resource_mut::<Dots>().0.remove(&TRIG_SESSION);
    rig.set_local_tick(8);
    let _ = rig.tick(vec![]);
    // All THREE per-entity ledgers no longer contain the subject — split into three `is_none()`
    // asserts (HR5: never a collapsed `assert!(a && b)`), equality-form (not `matches!`).
    assert_eq!(
        rig.world.resource::<CrossingProgress>().0.get(&entity),
        None
    );
    assert_eq!(rig.world.resource::<RequestInFlight>().0.get(&entity), None);
    assert_eq!(
        rig.world.resource::<ContainmentProgress>().0.get(&entity),
        None
    );
}

/// Slice 4a (adversary MEDIUM-3 — the cell-carry SHAPE assertion, NOT an integer-gate flip). Proves
/// the dot's `LatticePos.cell` (a DISTINCT NON-ZERO cell) is CARRIED UNCHANGED through the trigger
/// AND the offset-based crossing still fires. This proves the cell is PRESERVED (shape only). It does
/// NOT prove the "integer in/out DECISION flips across a cell boundary": `should_commit` /
/// `evaluate_one_subject` read ONLY `LatticePos.offset()` today (`stub.rs`), never comparing the cell,
/// so a cross-cell in/out DECISION test is BLOCKED on the P4/P5 rebase math (D-41) and is deliberately
/// NOT written here.
#[test]
fn slice4a_nonzero_cell_is_carried_unchanged_and_the_offset_crossing_still_fires() {
    use vd_core::glam::I64Vec3;
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 0x4C);
    // A DISTINCT non-zero cell anchor. The trigger reads only `offset()`, so the crossing decision is
    // driven by the (in-band) offset exactly as if the cell were zero.
    let cell = I64Vec3::new(5, -7, 11);
    let offset_inside = DVec3::new(100.0, 0.0, 0.0); // sd = 100 - 1000 ≪ -inset ⇒ inside the child
    rig.world.resource_mut::<Dots>().0.insert(
        TRIG_SESSION,
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
                // Plant the non-zero cell anchor on an otherwise-rest pose (avoids naming DQuat).
                pos: LatticePos::at(cell, offset_inside),
                ..StampedPose::at_rest(config().frame, offset_inside, UniverseTick(100))
            },
            yaw: 0.0,
            pitch: 0.0,
            last_applied_seq: None,
            prev_offset: LatticePos::from_metres(offset_inside, vd_core::pose::Tier::Fine),
        },
    );
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..8 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    // The OFFSET-based crossing still fires: exactly one inward CrossingRequest.
    assert_eq!(
        crossing_requests(&all).len(),
        1,
        "the offset-based crossing fires regardless of the (non-zero) cell anchor",
    );
    // The non-zero cell is CARRIED UNCHANGED through the trigger — PROVING the cell is PRESERVED
    // (shape). (It is NOT compared in the in/out decision — that cell-aware gate is P4/P5 D-41 math.)
    assert_eq!(
        rig.world
            .resource::<Dots>()
            .0
            .get(&TRIG_SESSION)
            .expect("the owned dot")
            .pose
            .pos
            .cell(),
        cell,
        "the non-zero LatticePos.cell is carried through the trigger unchanged (SHAPE preserved)",
    );
}

#[test]
fn the_grant_demux_flips_a_source_transient_to_crossing() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let entity = EntityId::pack(EntityKind::Debris, 10, 1, 7);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        entity,
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: TransientStatus::Held { outbound: None },
            prev_offset: transient_pose().pos,
        },
    );
    let grant = TransientCrossingGrant {
        subject: DirectoryKey::Entity(entity),
        dest: DEST_NODE,
        to_realm: OTHER_REALM,
        dst_realm_fence: Fence(3),
        batch: TransferId(77),
        to_parent: None,
    };
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::TransientCrossingGrant(grant),
    )]);
    // The grant flipped Held → Crossing; `emit_transient_batch` (later in the SAME schedule tick)
    // drained the Crossing into ONE batch and left it Held{outbound: Some(batch)}. The flip is
    // proven by BOTH the emit having fired and the resulting outbound tag.
    assert_eq!(
        rig.world.resource::<StubStats>().transient_grants_applied,
        1
    );
    assert_eq!(rig.world.resource::<StubStats>().transients_emitted, 1);
    assert_eq!(
        rig.world.resource::<OwnedTransients>().0[&entity].status,
        TransientStatus::Held {
            outbound: Some(TransferId(77)),
        },
        "the flipped Crossing was emitted, leaving Held{{outbound: Some(batch)}}",
    );
    // A REDELIVERED grant now finds Held{outbound: Some} (not a settled Held{None}) — a counted
    // no-op, so no re-flip + re-emit.
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::TransientCrossingGrant(grant),
    )]);
    assert_eq!(rig.world.resource::<StubStats>().transient_grant_noop, 1);
    assert_eq!(
        rig.world.resource::<StubStats>().transients_emitted,
        1,
        "the redelivered grant did NOT re-emit",
    );
}

#[test]
fn the_crossing_aborted_demux_clears_the_latch_on_an_id_match_only() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 8);
    // A live owned dot for the entity so the trigger's eviction retain keeps its latch (the empty
    // registry means no crossing is triggered; the dot only keeps the subject alive).
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(0.0, 0.0, 0.0));
    let transfer = crossing_transfer_id(DirectoryKey::Entity(entity), Fence(1), 0);
    rig.world
        .resource_mut::<RequestInFlight>()
        .0
        .insert(entity, transfer);
    // A STALE abort (wrong id) preserves the latch (a superseded / re-latched crossing) — but STILL acks
    // (3f-D: the orchestrator keeps its durable entry until the ack, so every delivery must re-ack).
    let stale_out = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::CrossingAborted(CrossingAborted {
            subject: DirectoryKey::Entity(entity),
            transfer: TransferId(999),
        }),
    )]);
    assert!(
        rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity),
        "a mismatched abort does NOT clear the latch",
    );
    assert_eq!(rig.world.resource::<StubStats>().crossing_abort_stale, 1);
    assert_eq!(
        crossing_aborted_acks(&stale_out).len(),
        1,
        "even a stale abort is acked (all paths ack — else the orch entry leaks)",
    );
    // The MATCHING abort clears the latch, acks, and RE-ARMS (bumps the attempt + resets the dwell).
    let match_out = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::CrossingAborted(CrossingAborted {
            subject: DirectoryKey::Entity(entity),
            transfer,
        }),
    )]);
    assert!(
        !rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity)
    );
    assert_eq!(
        rig.world.resource::<StubStats>().crossing_latches_cleared,
        1
    );
    assert_eq!(
        crossing_aborted_acks(&match_out).len(),
        1,
        "the match acks too"
    );
    // The re-arm bumped the attempt (0 -> 1), so a re-cross mints a FRESH id (H2).
    assert_eq!(
        rig.world
            .resource::<CrossingProgress>()
            .0
            .get(&entity)
            .map(|st| st.crossing_attempt),
        Some(1),
    );
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

#[test]
fn a_stranded_durable_latch_redrives_after_the_ttl() {
    // 3f-D4: a delivered-but-unresolved dest leaves the durable latch STANDING with no rising edge
    // (the dot dwells statically in-band; `evaluate_realm_boundaries`'s `Occupied` arm only
    // suppresses). The per-tick latch-scan MUST re-emit the SAME `CrossingRequest` once the ttl
    // elapses — the C1 fix (the `Entry::Occupied` re-drive the DEFERRED text prescribed is
    // unreachable for a static dweller).
    const TTL: u32 = 3;
    let mut rig = Rig::with_config(config_with_ttl(TTL));
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 42);
    // Deep inside the child from spawn (sd = 100 - 1000 ≪ -inset) so the first commit latches it; there
    // is no orchestrator in this rig, so the latch is never cleared → it STRANDS (the tested condition).
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));

    // Inside the child from tick 2 ⇒ the re-home commits on the FIRST evaluated tick (the band IS the
    // dwell) → the durable `Vacant` arm emits ONE request + arms `last_commit_tick = 2`.
    let mut first: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..=4 {
        rig.set_local_tick(t);
        first.extend(rig.tick(vec![]));
    }
    let first_reqs = crossing_requests(&first);
    assert_eq!(
        first_reqs.len(),
        1,
        "exactly ONE request from the rising edge"
    );
    assert_eq!(
        first_reqs[0].attempt, 0,
        "the first crossing uses attempt 0"
    );
    let latched_id = crossing_transfer_id(DirectoryKey::Entity(entity), Fence(1), 0);
    assert_eq!(
        rig.world.resource::<RequestInFlight>().0.get(&entity),
        Some(&latched_id),
        "the latch is held (no orchestrator ever cleared it)",
    );
    // The commit armed `last_commit_tick = 2`. Ticks 3/4 (already ticked in the loop above) were still
    // < TTL, so no re-drive fired there — proven by `first_reqs.len() == 1`.
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_redriven,
        0,
        "the ttl window has NOT elapsed yet through tick 4 (the `>= ttl` false arm)",
    );

    // Tick 5: local_tick - 2 == 3 >= TTL → the scan re-emits the SAME request (the `>= ttl` TRUE arm).
    rig.set_local_tick(5);
    let redrive_out = rig.tick(vec![]);
    let redrive_reqs = crossing_requests(&redrive_out);
    assert_eq!(
        redrive_reqs.len(),
        1,
        "the ttl re-drive re-emitted exactly one request"
    );
    // The re-drive is BYTE-IDENTICAL to the standing latch: same subject/to_realm/fence/session/attempt.
    assert_eq!(redrive_reqs[0].subject, DirectoryKey::Entity(entity));
    assert_eq!(redrive_reqs[0].to_realm, OTHER_REALM);
    assert_eq!(redrive_reqs[0].from_realm, config().realm);
    assert_eq!(redrive_reqs[0].subject_fence, Fence(1));
    assert_eq!(redrive_reqs[0].session, TRIG_SESSION);
    assert_eq!(redrive_reqs[0].attempt, 0, "same attempt → same id");
    assert_eq!(
        crossing_transfer_id(
            redrive_reqs[0].subject,
            redrive_reqs[0].subject_fence,
            redrive_reqs[0].attempt,
        ),
        latched_id,
        "the re-emitted id equals the standing latch id (idempotent at the orchestrator)",
    );
    assert!(
        rig.world.resource::<StubStats>().crossings_redriven >= 1,
        "the re-drive counter bumped",
    );
    assert_eq!(
        rig.world.resource::<RequestInFlight>().0.get(&entity),
        Some(&latched_id),
        "the latch is STILL held after the re-drive (only a saga terminal clears it)",
    );

    // Tick 6: the re-drive re-armed `last_commit_tick = 5`, so 6 - 5 == 1 < TTL → NO third emit yet
    // (the `>= ttl` false arm again, proving the timer re-arms and does not storm every tick).
    rig.set_local_tick(6);
    let after = rig.tick(vec![]);
    assert_eq!(
        crossing_requests(&after).len(),
        0,
        "the re-drive re-armed the ttl timer — no storm before the next window",
    );
}

/// D-WORLD-2 arc fixture: ttl 3, budget 2, `k_dwell` = `BoundaryTuning::DEFAULT.k_dwell` (5).
/// Timeline for a dot latched at tick 2 whose dest NEVER resolves (no orchestrator in the rig):
/// re-drives at 5 and 8, EXHAUSTION abort at 11, cooldown until 15, re-fire (attempt 1) at 16.
const XTTL: u32 = 3;
const XBUDGET: u32 = 2;

#[test]
fn an_exhausted_latch_aborts_locally_and_the_next_crossing_fires() {
    // THE D-WORLD-2 headline arc: unresolved-dest request → bounded ttl re-drives (count measured)
    // → exhaustion → the LOCAL pre-CAS abort (latch cleared, attempt bumped, entity stays simulated
    // at the source) → a LATER crossing of the SAME entity fires with a FRESH id. Before the cure
    // this latch stood FOREVER (the permanent strand the walk gate hit live on an unhosted planet).
    let mut rig = Rig::with_config(config_with_ttl_and_budget(XTTL, XBUDGET));
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 44);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));

    // Ticks 2..=10: the rising-edge request (t2) + exactly XBUDGET re-drives (t5, t8), all attempt 0.
    let mut before_exhaust: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..=10 {
        rig.set_local_tick(t);
        before_exhaust.extend(rig.tick(vec![]));
    }
    let reqs = crossing_requests(&before_exhaust);
    assert_eq!(
        reqs.len(),
        1 + XBUDGET as usize,
        "one rising edge + exactly the budgeted re-drives before exhaustion",
    );
    assert!(
        reqs.iter().all(|r| r.attempt == 0),
        "every pre-exhaustion emit is the SAME attempt (byte-identical id)"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_redriven,
        u64::from(XBUDGET)
    );
    assert_eq!(rig.world.resource::<StubStats>().crossings_exhausted, 0);

    // Tick 11: the third expiry finds the budget spent → the LOCAL exhaustion abort.
    rig.set_local_tick(11);
    let exhaust_out = rig.tick(vec![]);
    assert_eq!(
        crossing_requests(&exhaust_out).len(),
        0,
        "exhaustion emits NO further request — it aborts instead",
    );
    assert_eq!(rig.world.resource::<StubStats>().crossings_exhausted, 1);
    assert_eq!(
        rig.world.resource::<StubStats>().crossing_latches_cleared,
        1,
        "the exhaustion abort rides the ONE pre-CAS latch clear",
    );
    assert_eq!(
        rig.world.resource::<RequestInFlight>().0.get(&entity),
        None,
        "THE LATCH CLEARS — the strand is over",
    );
    let st = *rig
        .world
        .resource::<CrossingProgress>()
        .0
        .get(&entity)
        .expect("the crossing state survives the abort");
    assert_eq!(st.crossing_attempt, 1, "the abort bumped the attempt (H2)");
    assert_eq!(st.latched_crossing, None, "the re-emit payload is dropped");
    assert_eq!(st.redrives_spent, 0, "the budget resets for the next latch");
    assert_eq!(
        st.last_commit_tick,
        Some(TickId(11)),
        "the cooldown arms AT the abort tick — the sit-inside backoff rides k_dwell",
    );
    // THAW: the entity stays simulated at the source (no saga ever started; nothing was frozen).
    assert!(
        rig.world
            .resource::<Dots>()
            .0
            .get(&TRIG_SESSION)
            .expect("the dot is still here")
            .authority
            .simulates(),
        "the entity stays simulated at the source after the local abort",
    );

    // Ticks 12..=15: since_commit 1..4 < k_dwell(5) → the EXISTING dwell suppresses the re-fire.
    let mut cooldown_out: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 12..=15 {
        rig.set_local_tick(t);
        cooldown_out.extend(rig.tick(vec![]));
    }
    assert_eq!(
        crossing_requests(&cooldown_out).len(),
        0,
        "the k_dwell cooldown bounds the refire — no per-tick abort/latch storm",
    );

    // Tick 16: the cooldown elapsed and the dot is STILL in-band → the SAME entity's crossing
    // FIRES AGAIN, re-latched under a FRESH id (attempt 1) — the strand cure's whole point.
    rig.set_local_tick(16);
    let refire_out = rig.tick(vec![]);
    let refire = crossing_requests(&refire_out);
    assert_eq!(refire.len(), 1, "a later crossing of the SAME entity fires");
    assert_eq!(refire[0].attempt, 1, "the re-fire mints the bumped attempt");
    assert_eq!(
        rig.world.resource::<RequestInFlight>().0.get(&entity),
        Some(&crossing_transfer_id(
            DirectoryKey::Entity(entity),
            Fence(1),
            1
        )),
        "the re-latch id is FRESH (attempt-stamped) — a stale abort can never wrong-clear it",
    );
}

#[test]
fn a_sit_inside_dweller_is_bounded_to_the_derived_refire_cycle() {
    // D-WORLD-2 sit-inside backoff: a dot PARKED inside an unresolvable region refires scan →
    // re-drives → abort FOREVER — but at a BOUNDED cadence derived from the existing machinery:
    // one full cycle is `(budget+1)·ttl` (the latch's re-drive life) + `k_dwell` (the containment
    // cooldown the abort arms), and each cycle emits exactly `budget+1` requests. No sleep
    // literal anywhere: the bound is computed from the same config the systems read.
    let cfg = config_with_ttl_and_budget(XTTL, XBUDGET);
    let cycle = (u64::from(XBUDGET) + 1) * u64::from(XTTL) + u64::from(cfg.boundary.k_dwell);
    let per_cycle = u64::from(XBUDGET) + 1;
    let mut rig = Rig::with_config(cfg);
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 45);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));

    // Two full cycles, starting at the first evaluated tick (2): 2 ..= 2 + 2·cycle − 1.
    let window = 2 * cycle;
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..(2 + window) {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    let total = crossing_requests(&all).len() as u64;
    // The derived bound, computed (never a literal): ⌈window/cycle⌉ cycles × (budget+1) requests.
    let bound = window.div_ceil(cycle) * per_cycle;
    assert!(
        total <= bound,
        "refire count over {window} ticks must stay within the derived bound {bound}, got {total}",
    );
    // Deterministic rig ⇒ the exact count is the bound itself (2 cycles × 3 requests): t2/t5/t8
    // then (post-abort at 11, cooldown to 15) t16/t19/t22 — pinning the cadence, not just the cap.
    assert_eq!(total, 2 * per_cycle);
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_exhausted,
        2,
        "one exhaustion abort per cycle",
    );
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_redriven,
        2 * u64::from(XBUDGET),
        "the budgeted re-drives per cycle, twice",
    );
}

#[test]
fn a_graze_continues_unharmed_after_the_exhaustion_abort() {
    // D-WORLD-2 graze: the dot PASSES THROUGH the unresolvable region (the walk gate's live
    // failure: a fly-in grazing an unhosted planet's 3.95 m SOI) and is long gone by exhaustion.
    // The abort clears the latch; the container then MATCHES the owning realm, so nothing
    // re-fires — the graze simply continues, and the entity's later crossings are free again.
    let mut rig = Rig::with_config(config_with_ttl_and_budget(XTTL, XBUDGET));
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 46);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
    // Tick 2: in-band → the rising-edge request latches (the graze's entry).
    rig.set_local_tick(2);
    let first = rig.tick(vec![]);
    assert_eq!(crossing_requests(&first).len(), 1);
    // The graze leaves: well past the child's destroy edge (r 1000 + outset), still inside OWN.
    move_dot(&mut rig, TRIG_SESSION, DVec3::new(5000.0, 0.0, 0.0));

    // Through the re-drives (5, 8 — the scan re-emits the LATCHED payload regardless of where the
    // dot now is) and the exhaustion (11), then a long quiet tail.
    let mut rest: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 3..=30 {
        rig.set_local_tick(t);
        rest.extend(rig.tick(vec![]));
    }
    assert_eq!(
        crossing_requests(&rest).len(),
        XBUDGET as usize,
        "only the budgeted re-drives — after the abort the departed graze NEVER re-fires",
    );
    assert_eq!(rig.world.resource::<StubStats>().crossings_exhausted, 1);
    assert_eq!(
        rig.world.resource::<RequestInFlight>().0.get(&entity),
        None,
        "the latch is gone — the entity's future crossings are unsuppressed",
    );
    let dot = rig
        .world
        .resource::<Dots>()
        .0
        .get(&TRIG_SESSION)
        .expect("the dot is still here");
    assert!(
        dot.authority.simulates(),
        "the graze continues unharmed — still owned and simulated at the source",
    );
}

#[test]
fn an_armed_shard_keepalive_demands_an_inward_crossing_dest() {
    // The PRODUCER gate of the Symptom-B fix, on the LAWFUL shape (finding 37): on an ARMED
    // (demand-scale) shard, every tick a durable crossing latch stands,
    // `redrive_stranded_crossings` emits a KeepAlive `RealmDemand` for an INWARD dest — a DIRECT
    // CHILD, the one crossing shape SL7 lets a shard demand (`ancestor_close` pulls the chain
    // above it at the orchestrator). The child's own AoI band is INERT and only the root's is
    // armed, so the keep-alive is the SOLE demand under test (no AoI-union collision);
    // `request_ttl_ticks == 0` keeps the ttl re-drive inert too.
    let armed_root = RealmRegion {
        aoi: aoi_band(1), // arms `aoi_live` without adding any direct-child AoI demand
        ..region(ROOT_REALM, None, DVec3::ZERO, 1.0e9)
    };
    let forest = RealmRegions::new(vec![
        armed_root,
        region(OWN_REALM, Some(ROOT_REALM), DVec3::ZERO, 100_000.0),
        region(OTHER_REALM, Some(OWN_REALM), DVec3::ZERO, 1000.0),
    ]);
    // `own_coord` comes from the SAME forest the demand coord will (exactly the production boot:
    // one lineage source), or the parent-of-child compare would be measuring two derivations.
    let own_coord = forest.coord_of(OWN_REALM).expect("own realm is rostered");
    let want = forest
        .coord_of(OTHER_REALM)
        .expect("the child is a seed-lineage realm");
    let mut rig = Rig::with_config(StubConfig {
        own_coord,
        ..config()
    });
    grant_realm_for(&mut rig, OWN_REALM);
    *rig.world.resource_mut::<RealmRegions>() = forest;
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 99);
    // INSIDE the child (r=1000, off-centre) ⇒ container == OTHER_REALM ⇒ cross IN, latching the
    // crossing (no orchestrator in the rig ⇒ the latch STRANDS and keeps emitting).
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(500.0, 0.0, 0.0));
    rig.set_local_tick(2);
    let _ = rig.tick(vec![]); // cross IN + latch
    rig.set_local_tick(3);
    let out = rig.tick(vec![]); // the latch stands ⇒ the keep-alive fires this tick
    let dest_keepalives = realm_demands(&out)
        .into_iter()
        .filter(|d| d.child == want)
        .filter(|d| d.verb == DemandVerb::KeepAlive)
        .count();
    assert!(dest_keepalives >= 1);
    // The lawful shape passed the structural gate uncounted; the ttl re-drive stayed inert.
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .demand_refused_not_own_or_child,
        0
    );
    assert_eq!(rig.world.resource::<StubStats>().crossings_redriven, 0);
}

#[test]
fn an_outward_crossing_emits_no_demand_and_counts_the_refusal() {
    // The OTHER half of the finding-37 cure, measured: an OUTWARD crossing's dest is this shard's
    // PARENT, which a shard may never demand (SL7 — a realm speaks about itself or a direct child,
    // never upward). `push_demand`'s structural gate refuses it, counted; the latch stands
    // untouched. The parent needs no upward demand: the latch keeps the departing occupant in this
    // shard's observer fold (`speaks_for`), so this realm never reports Empty, arm B of
    // `desired_alive` holds it, and `ancestor_close` pulls the chain — the deterministic twin is
    // `rlm::tests::an_outward_crossings_parent_stays_alive_without_an_upward_demand`, the
    // process-tier experiment the return-crossing gate.
    let mut rig = Rig::new();
    rig.grant_realm();
    let armed = |realm, parent, r| RealmRegion {
        realm,
        center: vd_core::geometry::ParentCentre::authored(LatticePos::ORIGIN),
        frame: frame_of(realm),
        shape: Boundary::Shell { r },
        look: Some(Boundary::Shell { r }),
        band: band(),
        aoi: aoi_band(1),
        parent,
        interior_band: vd_core::geometry::AoiConfig::inert(),
    };
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(vec![
        armed(ROOT_REALM, None, 1.0e9),
        armed(PARENT_REALM, Some(ROOT_REALM), 100_000.0),
        armed(OWN_REALM, Some(PARENT_REALM), 1000.0),
    ]);
    let parent_coord = rig
        .world
        .resource::<RealmRegions>()
        .coord_of(PARENT_REALM)
        .expect("the parent is a seed-lineage realm");
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 99);
    // OUTSIDE own_small (r=1000) but inside the parent (r=100_000) ⇒ container == PARENT_REALM ⇒
    // cross OUT, latching the crossing (no orchestrator ⇒ the latch STRANDS).
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(5000.0, 0.0, 0.0));
    rig.set_local_tick(2);
    let _ = rig.tick(vec![]); // cross OUT + latch
    rig.set_local_tick(3);
    let out = rig.tick(vec![]); // the latch stands ⇒ the refusal fires this tick
    // The WHOLE demand list is empty — the only candidate emitter this tick was the outward
    // keep-alive (the own realm has no AoI children here), so this is the strongest claim, by
    // equality (a filtering closure would carry an uncoverable never-matching region, HR5).
    assert_eq!(
        realm_demands(&out),
        vec![],
        "no demand at all leaves the shard — the upward emit toward {parent_coord:?} is \
         structurally gone"
    );
    assert!(
        rig.world
            .resource::<StubStats>()
            .demand_refused_not_own_or_child
            >= 1,
        "the refusal is counted where the shape law lives"
    );
    assert!(
        rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity),
        "the crossing latch stands regardless — only a saga terminal clears it"
    );
    assert_eq!(rig.world.resource::<StubStats>().crossings_redriven, 0);
}

#[test]
fn the_ttl_redrive_is_inert_when_request_ttl_ticks_is_zero() {
    // 3f-D4: with the DEFAULT `request_ttl_ticks == 0` the scan EARLY-RETURNS (the inert default of
    // every current rig) — a held latch is NEVER re-driven, even past many ticks.
    let mut rig = Rig::new(); // request_ttl_ticks == 0
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 43);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));

    // Commit on the first in-band tick, then advance well past any plausible ttl window.
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..=40 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    assert_eq!(
        crossing_requests(&all).len(),
        1,
        "exactly the ONE rising-edge request — the ttl scan is inert (never re-drives)",
    );
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_redriven,
        0,
        "the ttl==0 early-return means a held latch is never re-driven",
    );
    assert!(
        rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity),
        "the latch stands (no positive clear), but is never re-emitted",
    );
}

#[test]
fn a_crossing_aborted_for_a_non_entity_subject_still_acks() {
    // The THIRD unconditional-ack path (3f-D): a `CrossingAborted` whose subject is not an Entity (a
    // malformed/realm subject) has no latch to clear, but MUST still ack — else the orchestrator's
    // durable `pending_abort_replies` entry would leak.
    let mut rig = Rig::new();
    rig.grant_realm();
    let out = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::CrossingAborted(CrossingAborted {
            subject: DirectoryKey::Realm(RealmId::System(7)),
            transfer: TransferId(5),
        }),
    )]);
    assert_eq!(
        rig.world.resource::<StubStats>().crossing_abort_no_entity,
        1
    );
    assert_eq!(
        crossing_aborted_acks(&out).len(),
        1,
        "the no-entity path acks too",
    );
}

#[test]
fn a_saga_demote_clears_the_durable_crossing_latch() {
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 9);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
    for t in 2..8 {
        rig.set_local_tick(t);
        let _ = rig.tick(vec![]);
    }
    assert!(
        rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity)
    );
    // The durable COMMIT terminal at the source (the saga-pushed Demote) POSITIVELY clears it.
    let demote = DemoteCmd {
        transfer: crossing_transfer_id(DirectoryKey::Entity(entity), Fence(1), 0),
        subject: DirectoryKey::Entity(entity),
        new_owner_fence: Fence(2),
        step_id: DEMOTE_STEP,
    };
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::Demote(demote),
    )]);
    assert!(
        !rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity),
        "the durable Demote terminal cleared the crossing latch",
    );
    assert!(rig.world.resource::<StubStats>().crossing_latches_cleared >= 1);
}

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

#[test]
fn slice4b_an_undock_re_homes_outward_to_the_parent_realm() {
    // Slice 4b (retargeted to CONTAINMENT) — the UNDOCK leg: the shard OWNS a CHILD realm (OWN_REALM,
    // a small shell nested under PARENT_REALM). An entity leaving the child region (container flips to
    // the parent) re-homes OUTWARD to PARENT_REALM. Symmetric with the dock — the SAME machinery; there
    // is no "direction", only "the container changed".
    let mut rig = Rig::new();
    rig.grant_realm();
    let entity = EntityId::pack(EntityKind::Player, 10, 2, 1);
    let out = drive_outward_exit(&mut rig, entity);
    let reqs = crossing_requests(&out);
    assert_eq!(
        reqs.len(),
        1,
        "leaving the owned child region re-homes exactly once",
    );
    // THE LOAD-BEARING ASSERT: the undock re-homes to the PARENT (the deepest region still containing
    // the dot after it left the child). NON-VACUOUS because PARENT_REALM != OWN_REALM != OTHER_REALM.
    assert_eq!(
        reqs[0].to_realm, PARENT_REALM,
        "the undock re-homes OUTWARD to the parent realm (the new deepest container)",
    );
    assert_ne!(
        reqs[0].to_realm, OTHER_REALM,
        "the undock destination is NOT the inward child interior",
    );
    assert_eq!(
        reqs[0].from_realm, OWN_REALM,
        "the re-home leaves the realm the shard owns the subject in",
    );
}

#[test]
fn slice4b_a_dock_and_undock_resolve_contrasting_destinations() {
    // Slice 4b (retargeted to CONTAINMENT) — DOCK vs UNDOCK contrast: entering a DEEPER child region
    // re-homes to the child interior (OTHER_REALM); leaving an OWNED child region re-homes to the
    // parent (PARENT_REALM). Both fall out of `container()`; the two destinations DIFFER (no accidental
    // alias) — proving the symmetric rule distinguishes the two container changes without a direction.

    // DOCK (into a deeper child) — the standard dock forest; the dot inside the child re-homes to it.
    let mut dock = Rig::new();
    dock.grant_realm();
    plant_dock_regions(&mut dock);
    let dock_entity = EntityId::pack(EntityKind::Player, 10, 2, 2);
    // Deep inside the child from spawn (sd ≪ -inset) → container == OTHER_REALM ≠ own ⇒ re-home INWARD.
    insert_owned_dot(
        &mut dock,
        TRIG_SESSION,
        dock_entity,
        DVec3::new(100.0, 0.0, 0.0),
    );
    let mut dock_out: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..8 {
        dock.set_local_tick(t);
        dock_out.extend(dock.tick(vec![]));
    }
    let dock_reqs = crossing_requests(&dock_out);
    assert_eq!(dock_reqs.len(), 1, "the dock commits exactly one re-home");
    assert_eq!(
        dock_reqs[0].to_realm, OTHER_REALM,
        "the dock docks INWARD to the deeper child region's realm",
    );

    // UNDOCK (out of an owned child) — the escape forest; leaving the own-shell undocks to the parent.
    let mut undock = Rig::new();
    undock.grant_realm();
    let undock_entity = EntityId::pack(EntityKind::Player, 10, 2, 3);
    let undock_out = drive_outward_exit(&mut undock, undock_entity);
    let undock_reqs = crossing_requests(&undock_out);
    assert_eq!(
        undock_reqs.len(),
        1,
        "the undock commits exactly one re-home"
    );
    assert_eq!(
        undock_reqs[0].to_realm, PARENT_REALM,
        "the undock re-homes OUTWARD to the parent realm",
    );

    // THE CONTRAST: the two container changes resolve DISTINCT destinations.
    assert_ne!(
        dock_reqs[0].to_realm, undock_reqs[0].to_realm,
        "a dock and an undock resolve DIFFERENT destinations (deeper child vs parent)",
    );
}

#[test]
fn a_nested_inner_boundary_wins_over_its_parent() {
    // Retargeted to CONTAINMENT (the `deepest_wins` property): a dot inside BOTH the own region
    // (System(7)) AND a coincident DEEPER child (Planet(42), nested under own) has container == the
    // INNER child (depth beats the parent), so the re-home targets the inner realm — exercising the
    // boot-cached `region_depth` argmax.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig); // root ⊃ own(System 7) ⊃ child(Planet 42), all origin-coincident
    let entity = EntityId::pack(EntityKind::Player, 10, 2, 5);
    // At |100| the dot is inside own (100000) AND the child (1000): the deeper child wins.
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..8 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    let reqs = crossing_requests(&all);
    assert_eq!(reqs.len(), 1);
    assert_eq!(
        reqs[0].to_realm, OTHER_REALM,
        "the INNER (deeper) region wins the depth argmax",
    );
}

#[test]
fn a_boundary_hovering_dot_does_not_flap() {
    // Retargeted to CONTAINMENT (`hysteresis_no_flap`): a dot jittering INSIDE the child region's band
    // for many ticks stays a member (its bit never releases), so container is stable and the latch caps
    // the emits at ≤ 1 across the whole window (no per-tick flap).
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 2, 2);
    // Hover deep inside the child (sd stays ≪ -inset even with the jitter): a member every tick.
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(200.0, 0.0, 0.0));
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..15 {
        rig.set_local_tick(t);
        // Jitter within the child region each tick (still a member: sd stays deeply negative).
        let jitter = if t % 2 == 0 { 210.0 } else { 190.0 };
        move_dot(&mut rig, TRIG_SESSION, DVec3::new(jitter, 0.0, 0.0));
        all.extend(rig.tick(vec![]));
    }
    assert!(
        crossing_requests(&all).len() <= 1,
        "a region-hovering dot emits at most one request (no flap)",
    );
}

#[test]
fn a_non_entity_grant_and_abort_are_counted_no_ops() {
    // The `transfer_subject_entity() == None` arms of the two demuxes (a Realm subject).
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::TransientCrossingGrant(TransientCrossingGrant {
            subject: DirectoryKey::Realm(RealmId::System(7)),
            dest: DEST_NODE,
            to_realm: OTHER_REALM,
            dst_realm_fence: Fence(3),
            batch: TransferId(1),
            to_parent: None,
        }),
    )]);
    assert_eq!(
        rig.world.resource::<StubStats>().transient_grant_no_entity,
        1
    );
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::CrossingAborted(CrossingAborted {
            subject: DirectoryKey::Realm(RealmId::System(7)),
            transfer: TransferId(1),
        }),
    )]);
    assert_eq!(
        rig.world.resource::<StubStats>().crossing_abort_no_entity,
        1
    );
}

#[test]
fn a_grant_for_an_unknown_transient_is_a_counted_no_op() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let entity = EntityId::pack(EntityKind::Debris, 10, 2, 3);
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::TransientCrossingGrant(TransientCrossingGrant {
            subject: DirectoryKey::Entity(entity),
            dest: DEST_NODE,
            to_realm: OTHER_REALM,
            dst_realm_fence: Fence(3),
            batch: TransferId(1),
            to_parent: None,
        }),
    )]);
    assert_eq!(rig.world.resource::<StubStats>().transient_grant_noop, 1);
}

#[test]
fn a_dot_outside_the_deeper_child_but_inside_own_stays() {
    // Retargeted from the portal "StaysOutside is not a candidate": a dot that is OUTSIDE the deeper
    // child region (its bit never acquires) but inside the OWN region has container == OWN_REALM — the
    // realm the shard owns it in — so NO re-home fires and no crossing latch is created.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 2, 4);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(9000.0, 0.0, 0.0));
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..8 {
        rig.set_local_tick(t);
        // Move around, but always well outside the child region (and inside own).
        move_dot(
            &mut rig,
            TRIG_SESSION,
            DVec3::new(9000.0 + t as f64, 0.0, 0.0),
        );
        all.extend(rig.tick(vec![]));
    }
    assert!(crossing_requests(&all).is_empty());
    assert!(rig.world.resource::<RequestInFlight>().0.is_empty());
    // The subject is evaluated every tick (so it carries a ContainmentProgress bit + a CrossingProgress
    // cooldown row), but since container == own NO re-home ever COMMITTED — the cooldown was never
    // armed (the `last_commit_tick` is None).
    assert_eq!(
        rig.world
            .resource::<CrossingProgress>()
            .0
            .get(&entity)
            .and_then(|st| st.last_commit_tick),
        None,
        "no re-home committed ⇒ the cooldown was never armed",
    );
}

#[test]
#[should_panic(expected = "valid BoundaryTuning")]
fn a_zero_dwell_boundary_tuning_fails_loud_at_boot() {
    let bad = StubConfig {
        boundary: BoundaryTuning {
            k_dwell: 0,
            ..BoundaryTuning::DEFAULT
        },
        ..config()
    };
    // The fail-loud validation in `register_stub_shard` (mirrors the tick-pair guard). `k_dwell` is the
    // post-commit cooldown `should_rehome` still reads (§2.7); zero fails `BoundaryTuning::validate`.
    let _ = Rig::with_config(bad);
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

#[test]
fn aoi_transition_covers_every_hysteresis_arm() {
    let g = 3;
    // (false,true) ACQUIRE ⇒ SpinUp, grace armed.
    assert_eq!(
        aoi_transition(AoiState::default(), true, g),
        (
            Some(DemandVerb::SpinUp),
            Some(AoiState {
                was_in: true,
                grace_remaining: g
            })
        ),
    );
    // (true,true) HOLD-IN ⇒ KeepAlive, grace re-armed.
    assert_eq!(
        aoi_transition(
            AoiState {
                was_in: true,
                grace_remaining: 1
            },
            true,
            g
        ),
        (
            Some(DemandVerb::KeepAlive),
            Some(AoiState {
                was_in: true,
                grace_remaining: g
            })
        ),
    );
    // (true,false) grace > 0 ⇒ KeepAlive, grace decrements.
    assert_eq!(
        aoi_transition(
            AoiState {
                was_in: true,
                grace_remaining: 2
            },
            false,
            g
        ),
        (
            Some(DemandVerb::KeepAlive),
            Some(AoiState {
                was_in: true,
                grace_remaining: 1
            })
        ),
    );
    // (true,false) grace == 0 ⇒ DROP the key, NO demand (never a TearDown — M-1).
    assert_eq!(
        aoi_transition(
            AoiState {
                was_in: true,
                grace_remaining: 0
            },
            false,
            g
        ),
        (None, None),
    );
    // (false,false) never-in ⇒ nothing.
    assert_eq!(aoi_transition(AoiState::default(), false, g), (None, None));
}

#[test]
fn arriving_at_a_star_system_lands_on_its_edge_and_shows_its_planets() {
    // THE GATE for "every realm counts from its own centre", driven through the PRODUCTION path — the
    // same rebase the source shard performs at flush, with the same frame context the shard builds.
    // (An earlier draft called the rebase with the identity stub and so proved nothing about the
    // system; measuring a stand-in is not a measurement.)
    //
    // Two assertions, one arrival, because they are two faces of one defect:
    //   (1) you land ON the boundary you crossed, carrying a position no larger than the realm itself;
    //   (2) standing there, that system's planets are in range — the empty-neighbour-system bug.
    let cfg = vd_physics::worldgen::UniverseConfig::visual_demand(500.0, 0.02);
    let world = vd_physics::worldgen::WorldView::generated(0, &cfg);
    let system = world
        .regions()
        .iter()
        .find(|r| {
            // Flatten the NORMALIZED centre — at the seeded placement radius every component is
            // ulp-coarser than one cell, so the residual `.offset()` is exactly ZERO and only
            // the integer half carries the position.
            // ★ THE FIFTEENTH SITE (slice S9): this read the SYSTEM's own step to flatten a centre
            // the GALAXY authored. It picks "a system away from the origin", and every candidate
            // looked 2048× further out than it is — which happened not to change WHICH system it
            // picked, so nothing ever failed.
            matches!(r.realm, RealmId::System(_))
                && r.centre_m(world.regions())
                    .is_some_and(|c| c.length() > 0.0)
        })
        .copied()
        .expect("a galaxy of several stars has one away from the origin");
    let parent = world
        .regions()
        .iter()
        .find(|r| Some(r.realm) == system.parent)
        .copied()
        .expect("its parent is in the same world");
    let held = BTreeSet::from([system.realm]);
    let regions =
        RealmRegions::new(world.neighbourhood(&held)).with_moving_children(kepler_motion_fns(
            vd_physics::worldgen::moving_children_for_config(0, &cfg, system.realm)
                .into_iter()
                .collect(),
        ));
    // THE SOURCE shard — the one the occupant is leaving, which holds the PARENT realm. It is the only
    // side that can do this conversion, and under the ground rule it is the side that must: it authored
    // where this system sits, so it knows; the system itself does not and never will.
    let src_held = BTreeSet::from([parent.realm]);
    let src_regions =
        RealmRegions::new(world.neighbourhood(&src_held)).with_moving_children(kepler_motion_fns(
            vd_physics::worldgen::moving_children_for_config(0, &cfg, parent.realm)
                .into_iter()
                .collect(),
        ));
    let tick_hz = 50.0;
    let tick = UniverseTick(0);
    let extent = system.shape.circumscribed_extent();
    // Approach along -X and cross at the near face, expressed in the PARENT's frame — where an
    // occupant about to cross in genuinely is.
    // Build the approach position in LATTICE space (centre TRANSLATED by −extent), not as
    // an f64 sum: at the 2.25e15 m galaxy magnitude a flattened `centre − extent` rounds to
    // the 0.25 m grid BEFORE normalization and the exact-relabel claim below would be
    // measuring representability, not the relabel (the H-21 class the activation cures).
    let at_the_face_lattice = system
        .center
        .in_parents_frame()
        .translated(DVec3::new(-extent, 0.0, 0.0), parent.frame.tier());
    let approaching = StampedPose {
        frame: parent.frame,
        pos: at_the_face_lattice,
        vel: DVec3::ZERO,
        orient: glam::DQuat::IDENTITY,
        universe_tick: tick,
    };
    // The PARENT'S authored book does the rebase. An earlier draft built this from the DESTINATION
    // and so asked the arriving realm to place itself — the one thing it cannot do. It answered with
    // the pose unchanged, still measured from the galaxy, which is exactly the 12 km error the owner
    // flew into.
    let src_book = src_regions.author_book(parent.realm, tick_hz, tick);
    let arrived = vd_core::frame::transfer_frame(&approaching, system.frame, &src_book)
        .expect("a galaxy can place its own star system");
    // …and the DESTINATION re-runs the receiver's conversion on receipt, which must be a no-op: the
    // pose already arrives in its frame ("the child does nothing" — the same-frame accept arm).
    let frames = regions.author_book(system.realm, tick_hz, tick);
    assert_eq!(
        vd_core::frame::transfer_frame(&arrived, system.frame, &frames)
            .expect("a same-frame transfer is the identity")
            .pos,
        arrived.pos,
        "the receiver's re-run must be a no-op downward — converting a pose into the frame it is \
         already in cannot move it"
    );

    // (1) ON the edge, on the side we came from. Asserted as an EXACT position, not merely "inside":
    // a bound alone would be satisfied by relocating the occupant to the centre, which is precisely the
    // failure the owner called out — arriving must not move you, only rename where you are. Approaching
    // the near face along -X, that is exactly one extent out on -X in the realm's own frame.
    assert_eq!(
        fm(arrived.pos),
        DVec3::new(-extent, 0.0, 0.0),
        "arriving must RELABEL the occupant, not move it: entering at the near face must read exactly \
         one extent out on the side it came from"
    );
    // …and the crossing is a pure relabel, so nothing about its motion changed either.
    assert_eq!(arrived.vel, approaching.vel);
    // (1b) CONTAINMENT must agree: having arrived on the edge, the occupant is INSIDE this realm.
    // Arriving somewhere the containment engine then denies you are is how a crossing flaps.
    let sd = vd_core::geometry::region_signed_distance(&arrived, &system, &frames)
        .expect("the shard's own frame resolves");
    assert!(
        sd <= 0.0,
        "arrived on the edge but containment says OUTSIDE — the occupant's frame and the \
         boundary's own position disagree"
    );

    // (2) …and from there its planets are visible. This is the owner's live finding: warp to a
    // neighbouring star, arrive, and find the system empty. It works at the HOME system only because
    // that one sits at the origin, where "measured from my star" and "measured from the universe" are
    // the same numbers — which is why every existing proof of streaming planets flies there.
    let placements = regions.child_placements(system.realm, tick_hz, tick);
    assert!(
        !placements.is_empty(),
        "the hosting shard holds this system's planets at all"
    );
    let visible = placements
        .iter()
        .filter(|(region, pose)| {
            let dist =
                occupant_child_dist(arrived.pos, DVec3::ZERO, pose.pos, system.frame.tier(), 0.0);
            region.aoi.in_range(false, dist)
        })
        .count();
    assert!(
        visible > 0,
        "standing on this system's edge, none of its planets are in range"
    );
}

#[test]
fn occupant_child_dist_takes_the_lesser_of_live_and_predictive() {
    let t = vd_core::pose::Tier::Fine;
    let child = LatticePos::ORIGIN;
    // A STATIC occupant (vel 0): pred == live == its distance.
    assert_eq!(
        occupant_child_dist(
            seated_pos(DVec3::new(500.0, 0.0, 0.0)),
            DVec3::ZERO,
            child,
            t,
            1.0
        ),
        500.0
    );
    // A MOVING occupant closing in: pred (300) beats live (1500) — the F7 predictive term.
    assert_eq!(
        occupant_child_dist(
            seated_pos(DVec3::new(1500.0, 0.0, 0.0)),
            DVec3::new(-1200.0, 0.0, 0.0),
            child,
            t,
            1.0
        ),
        300.0
    );
    // THE REGRESSION THIS FUNCTION EXISTS TO SURVIVE: the occupant seated the way the integrator
    // stores a position (whole-number part folded out) must measure the SAME distance as one seated
    // at rest. Reading either endpoint's leftover alone answered ~0 here, which is how every moving
    // player came to be measured as standing at their realm's origin.
    assert_eq!(
        occupant_child_dist(
            seated_pos(DVec3::new(500.0, 0.0, 0.0)),
            DVec3::ZERO,
            child,
            t,
            1.0
        ),
        occupant_child_dist(
            LatticePos::from_metres(DVec3::new(500.0, 0.0, 0.0), vd_core::pose::Tier::Fine),
            DVec3::ZERO,
            child,
            t,
            1.0
        ),
        "where the whole-number part sits cannot change the distance",
    );
    // (The cross-observer MIN/union now lives in `aoi_decide`/`union_verb`, covered by
    // `evaluate_realm_aoi_demand_order_is_stable` + the two-observer tests below.)
}

#[test]
fn union_verb_is_spinup_on_first_demand_keepalive_while_sustained() {
    // The child-level demand = the observer union: SpinUp the tick a child FIRST becomes demanded by
    // anyone, KeepAlive while any observer sustains it, nothing when none want it. (One observer ⇒
    // exactly the per-observer SpinUp→KeepAlive sequence, so the single-observer byte-shape is kept.)
    assert_eq!(union_verb(false, true), Some(DemandVerb::SpinUp)); // newly demanded by someone
    assert_eq!(union_verb(true, true), Some(DemandVerb::KeepAlive)); // still demanded
    assert_eq!(union_verb(true, false), None); // the last observer left ⇒ drop (never a TearDown)
    assert_eq!(union_verb(false, false), None); // never demanded
}

#[test]
fn region_level_recovers_seed_lineage_kinds() {
    use vd_core::realm_path::{RealmKindTag, RealmLevel};
    assert_eq!(
        region_level(&region(
            RealmId::Planet(42),
            Some(OWN_REALM),
            DVec3::ZERO,
            1.0
        )),
        Some(RealmLevel::new(RealmKindTag::Planet, 42))
    );
    assert_eq!(
        region_level(&region(
            RealmId::System(7),
            Some(ROOT_REALM),
            DVec3::ZERO,
            1.0
        )),
        Some(RealmLevel::new(RealmKindTag::System, 7))
    );
    // An entity-backed Ship realm has no lineage tag until P8 (D-SHIP-1): `None`, the typed
    // graceful exclusion — never a panic (audit :713).
    assert_eq!(
        region_level(&region(
            RealmId::Ship(EntityId::pack(EntityKind::Ship, 1, 7, 3)),
            Some(OWN_REALM),
            DVec3::ZERO,
            1.0
        )),
        None
    );
}

#[test]
fn a_ship_child_region_is_excluded_from_every_coord_lane_counted_never_a_panic() {
    // Audit :713 — `region_level` used to `expect` on an entity-backed Ship realm, so the FIRST
    // hosted ship region would abort the whole shard the moment any lane touched it. Until P8
    // gives ships a lineage coordinate (D-SHIP-1), every coord-needing lane must EXCLUDE it:
    // typed, counted, never a panic. Since Slice C2 there are exactly TWO such lanes left (the
    // cascade targeting, the interior fan and the scene reflect died with their messages): the
    // AoI/demand fold (`aoi_decide`) and a `Child`-scope window's hop row
    // (`emit_window_frames`). The ship still counts where no coord is needed — its live bit is
    // a child OBSERVER, so an occupied ship keeps its parent warm (SL7).
    const SHIP_HOME: NodeId = NodeId(77);
    let ship_realm = RealmId::Ship(EntityId::pack(EntityKind::Ship, 1, 7, 3));
    let mut rig = Rig::new();
    rig.grant_realm();
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(vec![
        root_region(),
        own_region(),
        child_region(),
        // The hosted ship: a direct child placed far from the dot (never a containment member).
        region(
            ship_realm,
            Some(OWN_REALM),
            DVec3::new(50_000.0, 0.0, 0.0),
            10.0,
        ),
    ]);
    // An occupant (the realm is not Empty), inside OWN only — outside the child and the ship.
    insert_owned_dot(
        &mut rig,
        SessionId(1),
        player(7),
        DVec3::new(5_000.0, 0.0, 0.0),
    );
    // The ship is LIVE (a fresh occupancy bit — it counts as a child observer).
    let now = rig.world.resource::<ClockSample>().local_tick;
    rig.world.resource_mut::<ChildLiveness>().0.insert(
        ship_realm,
        ChildLiveEntry {
            home: SHIP_HOME,
            fence: Fence(1),
            at: UniverseTick(1),
            last_seen: now,
        },
    );
    // …and a subscriber asks for a window scoped to the ship itself — the hop-row lane.
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Child(ship_realm),
    };
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .ship_child_regions_excluded,
        2,
        "each coord-needing lane skipped the ship exactly once, counted"
    );
    // Nothing was addressed TO the unaddressable child, and the ship-scoped window got no frame.
    assert!(
        sent.iter().all(|(to, _, _)| *to != SHIP_HOME),
        "no lane targeted the ship's home"
    );
    assert!(
        window_frames(&sent).is_empty(),
        "a window scoped to a coord-less ship is served no frame, never a guessed hop"
    );
}

#[test]
fn child_placements_unifies_movers_and_static() {
    // A STATIC direct child of OWN (not in the moving roster) rides its region center — place_child's
    // `None` arm — and is_direct_child accepts ONLY the OWN child (root/own excluded).
    let regions = RealmRegions::new(vec![
        root_region(),
        own_region(),
        region(
            OTHER_REALM,
            Some(OWN_REALM),
            DVec3::new(10.0, 0.0, 0.0),
            100.0,
        ),
    ]);
    let placements = regions.child_placements(OWN_REALM, 20.0, UniverseTick(5));
    assert_eq!(placements.len(), 1);
    assert_eq!(placements[0].0.realm, OTHER_REALM);
    assert_eq!(fm(placements[0].1.pos), DVec3::new(10.0, 0.0, 0.0));
    assert_eq!(placements[0].1.vel, DVec3::ZERO);
    // NORMALIZED since the cell activation: 10 m rides the integer half (10 × 1024 cells).
    assert_eq!(placements[0].1.pos.cell(), I64Vec3::new(10_240, 0, 0));
    assert_eq!(placements[0].1.pos.offset(), DVec3::ZERO);
}

#[test]
fn a_static_child_placement_carries_its_whole_cell_anchored_center() {
    // place_child's static arm used to ship `r.center.in_parents_frame().offset()` — the f64 remainder only — so a child
    // authored at a real integer cell anchor was placed as if the anchor were zero. Every region in
    // the forest today sits at cell ZERO, so nothing caught it; the row this produces is the realm
    // lane the gateway keys its placement table on, and one dropped anchor there becomes every
    // occupant in that realm drawn a cell-block away.
    let mut cell_anchored = region(
        OTHER_REALM,
        Some(OWN_REALM),
        DVec3::new(10.0, 0.0, 0.0),
        100.0,
    );
    cell_anchored.center = vd_core::geometry::ParentCentre::authored(LatticePos::at(
        I64Vec3::new(4096, 0, 0),
        DVec3::new(10.0, 0.0, 0.0),
    ));
    let regions = RealmRegions::new(vec![root_region(), own_region(), cell_anchored]);
    let placements = regions.child_placements(OWN_REALM, 20.0, UniverseTick(5));
    // BOTH halves ride the row bit-for-bit (the anchored centre is carried, never re-derived);
    // the VALUE is the total 4096 cells (4 m) + 10 m of raw offset = 14 m.
    assert_eq!(placements[0].1.pos.cell(), I64Vec3::new(4096, 0, 0));
    assert_eq!(placements[0].1.pos.offset(), DVec3::new(10.0, 0.0, 0.0));
    assert_eq!(fm(placements[0].1.pos), DVec3::new(14.0, 0.0, 0.0));
}

#[test]
fn evaluate_realm_aoi_inert_without_authority() {
    // No realm lease ⇒ the authority `else` short-circuits ⇒ no demand (even with a live child + dot).
    let mut rig = Rig::new();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
        ],
    );
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(500.0, 0.0, 0.0),
    );
    assert!(demands(&rig.tick(vec![])).is_empty());
}

#[test]
fn evaluate_realm_aoi_inert_empty_regions() {
    // Granted, but NO regions ⇒ the `is_empty` guard short-circuits ⇒ no demand.
    let mut rig = Rig::new();
    rig.grant_realm();
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(500.0, 0.0, 0.0),
    );
    assert!(demands(&rig.tick(vec![])).is_empty());
}

#[test]
fn evaluate_realm_aoi_unsynced_authors_nothing() {
    // D-Finding-1: a shard whose clock is NOT yet synced authors nothing (`has_synced` skips the
    // system) — no pre-sync demand even with authority + a live child + an in-range occupant.
    let mut rig = Rig::new();
    rig.grant_realm();
    rig.world.resource_mut::<ClockSample>().synced = false;
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
        ],
    );
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(500.0, 0.0, 0.0),
    );
    assert!(demands(&rig.tick(vec![])).is_empty());
}

#[test]
fn evaluate_realm_aoi_empty_self_report() {
    // Zero occupants ⇒ the CHILD shard self-reports its OWN realm holds nobody: exactly one
    // `Empty { child = own_coord }` (Step-3's occupancy authority — a sealed parent cannot see inside).
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
        ],
    );
    assert_eq!(
        demands(&rig.tick(vec![])),
        vec![RealmDemand {
            child: StubConfig::root_coord(OWN_REALM),
            parent_fence: Fence(1),
            verb: DemandVerb::Empty,
            universe_tick: UniverseTick(100),
        }]
    );
}

#[test]
fn evaluate_realm_aoi_spinup_then_keepalive() {
    // AOI-2: an occupant reaching the child emits exactly one SpinUp keyed on `child.path()`; the next
    // tick (still in range) emits KeepAlive. The FULL demand is asserted (child, fence, verb, tick).
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
        ],
    );
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(500.0, 0.0, 0.0),
    );
    let want = child_coord_of(OWN_REALM, OTHER_REALM);
    assert_eq!(
        demands(&rig.tick(vec![])),
        vec![RealmDemand {
            child: want.clone(),
            parent_fence: Fence(1),
            verb: DemandVerb::SpinUp,
            universe_tick: UniverseTick(100),
        }]
    );
    assert_eq!(
        demands(&rig.tick(vec![])),
        vec![RealmDemand {
            child: want,
            parent_fence: Fence(1),
            verb: DemandVerb::KeepAlive,
            universe_tick: UniverseTick(100),
        }]
    );
}

#[test]
fn evaluate_realm_aoi_grace_then_drop() {
    // An occupant LEAVES: while grace remains the child stays demanded (KeepAlive), then its key drops
    // and it stops being demanded — and NO parent TearDown is EVER emitted (M-1 locks REVISION-1 R2).
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 2),
        ],
    );
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(500.0, 0.0, 0.0),
    );
    let mut seen = Vec::new();
    seen.extend(demands(&rig.tick(vec![]))); // SpinUp (grace armed to 2)
    move_dot(&mut rig, TRIG_SESSION, DVec3::new(3000.0, 0.0, 0.0)); // OUT of the 2000 m tear-down
    seen.extend(demands(&rig.tick(vec![]))); // KeepAlive (grace 2 → 1)
    seen.extend(demands(&rig.tick(vec![]))); // KeepAlive (grace 1 → 0)
    let after_grace = demands(&rig.tick(vec![])); // grace 0 ⇒ drop, silent
    assert!(
        after_grace.is_empty(),
        "grace expired ⇒ the child stops being demanded"
    );
    assert_eq!(
        seen.iter().map(|d| d.verb).collect::<Vec<_>>(),
        vec![
            DemandVerb::SpinUp,
            DemandVerb::KeepAlive,
            DemandVerb::KeepAlive
        ]
    );
    assert!(
        !seen.iter().any(|d| d.verb == DemandVerb::TearDown),
        "Step 2 never emits a parent TearDown"
    );
}

#[test]
fn evaluate_realm_aoi_predictive_spinup() {
    // F7: an occupant OUTSIDE the spin-up radius but whose `pos + vel·horizon` lands inside ⇒ SpinUp
    // (boot latency masked); a STATIC occupant at the same pos ⇒ NO demand.
    let horizon = StubConfig {
        boot_ticks_p99: 20,
        ..config()
    }; // horizon_s = 20 · 0.05 = 1.0
    let mut rig = Rig::with_config(horizon);
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
        ],
    );
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(1500.0, 0.0, 0.0),
    );
    rig.world
        .resource_mut::<Dots>()
        .0
        .get_mut(&TRIG_SESSION)
        .expect("the dot")
        .pose
        .vel = DVec3::new(-1000.0, 0.0, 0.0); // pred: 1500 − 1000·1.0 = 500 < 1000
    assert_eq!(
        demands(&rig.tick(vec![]))
            .iter()
            .map(|d| d.verb)
            .collect::<Vec<_>>(),
        vec![DemandVerb::SpinUp]
    );

    // STATIC occupant (vel 0) at the same 1500 m ⇒ live == pred == 1500 > 1000 ⇒ silent.
    let mut still = Rig::with_config(StubConfig {
        boot_ticks_p99: 20,
        ..config()
    });
    still.grant_realm();
    plant_aoi(
        &mut still,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
        ],
    );
    insert_owned_dot(
        &mut still,
        TRIG_SESSION,
        player(7),
        DVec3::new(1500.0, 0.0, 0.0),
    );
    assert!(demands(&still.tick(vec![])).is_empty());
}

#[test]
fn evaluate_realm_aoi_demand_order_is_stable() {
    // H-2: two children + two occupants — the emitted `Vec<RealmDemand>` is IDENTICAL regardless of the
    // occupants' `Dots` insertion order (occupants reduce to a scalar min BEFORE any emit).
    let build = |sessions: &[(SessionId, DVec3)]| -> Vec<RealmDemand> {
        let mut rig = Rig::new();
        rig.grant_realm();
        plant_aoi(
            &mut rig,
            vec![
                root_region(),
                own_region(),
                aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
                aoi_child(RealmId::Planet(43), OWN_REALM, 100.0, 3),
            ],
        );
        for (i, (s, pos)) in sessions.iter().enumerate() {
            insert_owned_dot(&mut rig, *s, player(i as u32), *pos);
        }
        demands(&rig.tick(vec![]))
    };
    let a = build(&[
        (SessionId(1), DVec3::new(500.0, 0.0, 0.0)),
        (SessionId(2), DVec3::new(700.0, 0.0, 0.0)),
    ]);
    let b = build(&[
        (SessionId(2), DVec3::new(700.0, 0.0, 0.0)),
        (SessionId(1), DVec3::new(500.0, 0.0, 0.0)),
    ]);
    assert_eq!(a, b);
    assert_eq!(
        a.len(),
        2,
        "both children reached ⇒ two SpinUps in a stable order"
    );
}

#[test]
fn evaluate_realm_aoi_evicts_a_departed_child() {
    // The AoI ledger is lazily evicted to the current roster: a child removed from the forest drops its
    // membership entry (retain_live, keyed by RealmPath) — no leak.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
        ],
    );
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(500.0, 0.0, 0.0),
    );
    let _ = rig.tick(vec![]); // SpinUp ⇒ the child's AoI state is recorded
    assert_eq!(rig.world.resource::<AoiMembership>().0.len(), 1);
    // The child leaves the roster (region removed); its state is evicted next tick.
    plant_aoi(&mut rig, vec![root_region(), own_region()]);
    let _ = rig.tick(vec![]);
    assert!(
        rig.world.resource::<AoiMembership>().0.is_empty(),
        "a departed child's AoI state is evicted"
    );
}

#[test]
fn evaluate_realm_aoi_ships_a_membership_verdict_on_acquire_and_release() {
    // The per-observer BAND MEMBERSHIP that drives the lifecycle demand is ALSO the parent's
    // SL7 VERDICT — each tick it is diffed against what the window was already told. A child
    // entering the band is ADDED; leaving is REMOVED; an UNCHANGED verdict ships nothing.
    // Ids only: since Slice C2 a parent never states what a child LOOKS like (SL3).
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 0), // grace 0 ⇒ leaving is an immediate release
        ],
    );
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(500.0, 0.0, 0.0),
    );
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    // Tick 1: ACQUIRE ⇒ exactly one verdict ADDING the entered child, to the window's opener.
    let v = window_memberships(&rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]));
    assert_eq!(v.len(), 1, "one verdict on acquire");
    assert_eq!(v[0].0, GATEWAY, "routed to the window's opener");
    assert_eq!(v[0].1, WindowId(1));
    assert_eq!(v[0].2, vec![OTHER_REALM], "the entered child is added");
    assert!(v[0].3.is_empty(), "nothing removed on entry");
    // Tick 2: STILL in range (sustained) ⇒ NOTHING (only acquire/release change the verdict).
    assert!(
        window_memberships(&rig.tick(vec![])).is_empty(),
        "a sustained verdict states nothing"
    );
    // Move OUT (grace 0 ⇒ immediate release) ⇒ exactly one verdict REMOVING the departed child.
    move_dot(&mut rig, TRIG_SESSION, DVec3::new(3000.0, 0.0, 0.0));
    let v = window_memberships(&rig.tick(vec![]));
    assert_eq!(v.len(), 1, "one verdict on release");
    assert!(v[0].2.is_empty(), "nothing added on exit");
    assert_eq!(v[0].3, vec![OTHER_REALM], "the departed child is removed");
}

/// A TOMBSTONED-lane frame (the deleted per-occupant `OccupantInterest`) arriving on the
/// SignalDelta carrier is counted undecodable, never retained and never a panic — the
/// discriminant is reserved forever, its meaning is gone.
#[test]
fn a_tombstoned_occupant_interest_frame_is_counted_undecodable() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let interest = InterShardFlow::OccupantInterest(vd_wire::intershard::OccupantInterest {
        observer: AccountId(5),
        to_realm: StubConfig::root_coord(OWN_REALM),
        occupant: StampedPose::at_rest(
            config().frame,
            DVec3::new(10.0, 0.0, 0.0),
            UniverseTick(100),
        ),
        coarsen_level: 0,
    });
    let _ = rig.tick(vec![Inbound::Wire {
        from: NodeId(9),
        class: MsgClass::SignalDelta,
        bytes: crate::io::bytes(postcard::to_allocvec(&interest).expect("encode")),
    }]);
    assert_eq!(rig.world.resource::<StubStats>().undecodable, 1);
    // Garbage bytes take the same arm — counted, never a panic.
    let _ = rig.tick(vec![Inbound::Wire {
        from: NodeId(9),
        class: MsgClass::SignalDelta,
        bytes: crate::io::bytes(vec![0xFF, 0xFF, 0xFF]),
    }]);
    assert_eq!(rig.world.resource::<StubStats>().undecodable, 2);
}

/// The OTHER tombstoned lane, on its OWN carrier: the deleted per-occupant `ProxySceneSet` rode
/// the Saga class, whose dispatch DECODES it fine (the discriminant is reserved) — so the `Err`
/// fall-through can never count it. The explicit tombstone arm must, or the frame vanishes
/// silently. Pins the wire contract's "a received frame counts `undecodable`" for BOTH halves.
#[test]
fn a_tombstoned_proxy_scene_set_frame_is_counted_undecodable() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let reflect = InterShardFlow::ProxySceneSet(vd_wire::intershard::ProxySceneSet {
        observer: AccountId(5),
        realms: vec![],
    });
    let _ = rig.tick(vec![Inbound::Wire {
        from: NodeId(9),
        class: MsgClass::Saga,
        bytes: crate::io::bytes(postcard::to_allocvec(&reflect).expect("encode")),
    }]);
    assert_eq!(rig.world.resource::<StubStats>().undecodable, 1);
}

/// Step 5 slice A — the SL7 bit's receive arm: attested upsert, mis-route drop, stale-(fence,at)
/// drop, and the lane cure's ADMISSION (findings 0/43): a sender the directory head does not name
/// is refused fail-closed — counted, a head re-read armed, the stored route untouched — until the
/// head is re-read, at which point the re-homed child's new node is believed and last-wins
/// overwrites the home (the down-lanes follow it).
#[test]
fn retain_child_live_upserts_misroutes_and_rejects_stale() {
    use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};
    let cfg = config(); // own realm System(7), own_coord [System(7)]
    let mut store = ChildLiveness::default();
    let mut stats = StubStats::default();
    let mut outbox = OutboundBox::default();
    // The admission map: the directory head names NodeId(41) for Planet(42).
    let mut attested = ChildRealmNodes(BTreeMap::from([(RealmId::Planet(42), NodeId(41))]));
    // The armed-head-read probe: EVERYTHING the receive pushed, decoded — asserted by equality
    // (never a filtering match: its catch-all arm would be an uncoverable region, HR5).
    let flows = |outbox: &OutboundBox| -> Vec<InterShardFlow> {
        outbox
            .0
            .iter()
            .map(|(_, _, b, _)| {
                postcard::from_bytes::<InterShardFlow>(b).expect("a pushed flow decodes")
            })
            .collect()
    };
    let head_read_for = |realm: RealmId| {
        InterShardFlow::Directory(DirectoryOp::HeadRead {
            key: DirectoryKey::Realm(realm),
        })
    };
    let child_coord = vd_core::realm_coord::RealmCoord::from_path(RealmPath::from_levels(vec![
        RealmLevel::new(RealmKindTag::System, 7),
        RealmLevel::new(RealmKindTag::Planet, 42),
    ]))
    .expect("two-level path");
    let bit = |fence: u64, at: u64| vd_wire::intershard::ChildLive {
        child: child_coord.clone(),
        fence: Fence(fence),
        at: UniverseTick(at),
    };
    // On-target + attested (the head names the sender) ⇒ upserted, home = the sender.
    retain_child_live(
        &mut store,
        &cfg,
        bit(1, 10),
        vd_core::TickId(5),
        NodeId(41),
        &attested,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.child_live_received, 1);
    let held = store.0[&RealmId::Planet(42)];
    assert_eq!(
        (held.home, held.fence, held.at),
        (NodeId(41), Fence(1), UniverseTick(10))
    );
    // STALE by (fence, at): an older heartbeat from the attested node never regresses the entry.
    retain_child_live(
        &mut store,
        &cfg,
        bit(1, 9),
        vd_core::TickId(6),
        NodeId(41),
        &attested,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.child_live_stale, 1);
    assert_eq!(
        store.0[&RealmId::Planet(42)].home,
        NodeId(41),
        "the stale bit changed nothing"
    );
    assert_eq!(flows(&outbox), vec![], "no refusal ⇒ no re-read armed");
    // UNATTESTED (the re-homed child's FIRST bit from its new node, before the head re-read):
    // refused fail-closed — counted, the route untouched, ONE head re-read armed for that child.
    retain_child_live(
        &mut store,
        &cfg,
        bit(2, 9),
        vd_core::TickId(7),
        NodeId(44),
        &attested,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.child_live_unattested, 1);
    assert_eq!(
        store.0[&RealmId::Planet(42)].home,
        NodeId(41),
        "an unattested bit never refreshes the route — the zombie window is bounded here"
    );
    assert_eq!(
        flows(&outbox),
        vec![head_read_for(RealmId::Planet(42))],
        "the refusal armed the lazy head re-read (D-RLM-6 mechanism C)"
    );
    // THE HEAD RE-READ LANDS (the same reply arm the parent resolve rides): the new node is now
    // the admission answer, and the SAME bit is believed — last-wins overwrites the home.
    attested.0.insert(RealmId::Planet(42), NodeId(44));
    retain_child_live(
        &mut store,
        &cfg,
        bit(2, 9),
        vd_core::TickId(7),
        NodeId(44),
        &attested,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(store.0[&RealmId::Planet(42)].home, NodeId(44));
    // NO HEAD AT ALL (fail closed): a child the directory has no record for is refused too.
    attested.0.remove(&RealmId::Planet(42));
    retain_child_live(
        &mut store,
        &cfg,
        bit(3, 11),
        vd_core::TickId(8),
        NodeId(44),
        &attested,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.child_live_unattested, 2);
    assert_eq!(
        store.0[&RealmId::Planet(42)].home,
        NodeId(44),
        "the no-head refusal stored nothing new"
    );
    // MIS-ROUTE: a child whose parent is NOT this realm is dropped + counted (before admission —
    // WHICH REALM precedes WHO SENT).
    let foreign = vd_core::realm_coord::RealmCoord::from_path(RealmPath::from_levels(vec![
        RealmLevel::new(RealmKindTag::System, 9),
        RealmLevel::new(RealmKindTag::Planet, 42),
    ]))
    .expect("two-level path");
    retain_child_live(
        &mut store,
        &cfg,
        vd_wire::intershard::ChildLive {
            child: foreign,
            fence: Fence(9),
            at: UniverseTick(99),
        },
        vd_core::TickId(8),
        NodeId(45),
        &attested,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.child_live_misrouted, 1);
    assert_eq!(store.0.len(), 1, "the mis-route stored nothing");
}

/// Slice C1, the Q2-relay receive (the tombstoned shape lane's successor — owner-approved
/// 2026-08-16, owner_decisions_2026-08-15.md addendum + window_lane.md §5 RULINGS): a live
/// child's sealed batch is HELD VERBATIM (byte-equal, unopened — the parent's whole lawful
/// vocabulary is forward-or-drop), a receipt is a last-wins replace, and every admission arm
/// refuses fail-closed + counted: mis-route, unattested sender, a deposed incarnation's stale
/// fence.
#[test]
fn on_window_relay_holds_the_sealed_batch_verbatim_and_admits_fail_closed() {
    use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};
    let story = Story::new();
    let cfg = story_config(&story, story.system);
    let clock = ClockSample {
        local_tick: vd_core::TickId(4),
        universe_tick: UniverseTick(0),
        epoch: vd_core::EpochId(1),
        synced: true,
    };
    let mut held = RelayHeld::default();
    let mut stats = StubStats::default();
    let seal = vd_wire::session_flow::seal_relay_statements(&[
        vd_wire::session_flow::RelayedStatement::Body {
            subject: story.planet,
            stmt: vd_wire::session_flow::BodyStmt::SelfLook {
                bag: vd_core::look::look_bag(&Boundary::Shell { r: 1.0 }),
            },
            authored_at: UniverseTick(9),
        },
    ]);
    let child_coord = vd_core::realm_coord::RealmCoord::from_path(RealmPath::from_levels(vec![
        level_of(story.system).expect("system level"),
        level_of(story.planet).expect("planet level"),
    ]))
    .expect("two-level path");
    let attested = ChildRealmNodes(BTreeMap::from([(story.planet, NodeId(70))]));
    on_window_relay(
        vd_wire::intershard::WindowRelay {
            child: child_coord.clone(),
            realm_fence: Fence(3),
            own: seal.clone(),
            interior: Vec::new(),
        },
        NodeId(70),
        &cfg,
        &clock,
        &attested,
        &mut held,
        &mut stats,
    );
    assert_eq!(stats.window_relays_received, 1);
    let entry = &held.0[&story.planet];
    assert_eq!(entry.own, seal, "held VERBATIM — byte-equal, unopened");
    assert_eq!(
        entry.digest,
        relay_entry_digest(Fence(3), &seal, &[]),
        "the §5.4 baseline digest is computed once, at receive"
    );
    assert_eq!(entry.fence, Fence(3), "the child's own fence rides intact");
    // LAST-WINS REPLACE: a same-or-newer fence's batch replaces the held one whole.
    let newer = vd_wire::session_flow::seal_relay_statements(&[
        vd_wire::session_flow::RelayedStatement::Level {
            at: UniverseTick(10),
            rows: Vec::new(),
        },
    ]);
    on_window_relay(
        vd_wire::intershard::WindowRelay {
            child: child_coord.clone(),
            realm_fence: Fence(4),
            own: newer.clone(),
            interior: Vec::new(),
        },
        NodeId(70),
        &cfg,
        &clock,
        &attested,
        &mut held,
        &mut stats,
    );
    assert_eq!(held.0[&story.planet].own, newer, "replaced, not merged");
    // STALE FENCE: a deposed incarnation (fence below the held one) is refused counted.
    on_window_relay(
        vd_wire::intershard::WindowRelay {
            child: child_coord.clone(),
            realm_fence: Fence(3),
            own: seal.clone(),
            interior: Vec::new(),
        },
        NodeId(70),
        &cfg,
        &clock,
        &attested,
        &mut held,
        &mut stats,
    );
    assert_eq!(stats.window_relay_stale, 1);
    assert_eq!(
        held.0[&story.planet].fence,
        Fence(4),
        "the zombie replaced nothing"
    );
    // UNATTESTED: a sender the head does not name is refused fail-closed, counted.
    on_window_relay(
        vd_wire::intershard::WindowRelay {
            child: child_coord,
            realm_fence: Fence(9),
            own: seal.clone(),
            interior: Vec::new(),
        },
        NodeId(71),
        &cfg,
        &clock,
        &attested,
        &mut held,
        &mut stats,
    );
    assert_eq!(stats.window_relay_unattested, 1);
    assert_eq!(
        held.0[&story.planet].fence,
        Fence(4),
        "nothing new believed"
    );
    // MIS-ROUTE: a sender whose parent link is not this realm drops counted, stores nothing.
    let foreign = vd_core::realm_coord::RealmCoord::from_path(RealmPath::from_levels(vec![
        RealmLevel::new(RealmKindTag::System, 999),
        RealmLevel::new(RealmKindTag::Planet, 1),
    ]))
    .expect("two-level path");
    on_window_relay(
        vd_wire::intershard::WindowRelay {
            child: foreign,
            realm_fence: Fence(1),
            own: seal,
            interior: Vec::new(),
        },
        NodeId(70),
        &cfg,
        &clock,
        &attested,
        &mut held,
        &mut stats,
    );
    assert_eq!(stats.window_relay_misrouted, 1);
    assert_eq!(held.0.len(), 1);
}

/// Slice C1, the Q2-relay forward: a held batch reaches every open window ONCE (send-on-change
/// per (window, child) — a fresh window is served everything currently held), the forwarded
/// bytes are the held bytes VERBATIM with the child's fence intact, a held entry outliving the
/// derived retain TTL is pruned (and its per-window baseline cleared so a re-held child
/// re-ships), and zero windows forward nothing while the prune still runs.
#[test]
fn emit_window_relays_forwards_verbatim_send_on_change_and_prunes_by_the_ttl() {
    let cfg = config();
    let clock = ClockSample {
        local_tick: vd_core::TickId(10),
        universe_tick: UniverseTick(10),
        epoch: vd_core::EpochId(1),
        synced: true,
    };
    let mut held = RelayHeld::default();
    let mut windows = OpenWindows::default();
    let mut stats = StubStats::default();
    let mut outbox = OutboundBox::default();
    let seal = vd_wire::session_flow::seal_relay_statements(&[
        vd_wire::session_flow::RelayedStatement::Body {
            subject: OTHER_REALM,
            stmt: vd_wire::session_flow::BodyStmt::SelfLook {
                bag: vd_core::look::look_bag(&Boundary::Shell { r: 2.0 }),
            },
            authored_at: UniverseTick(9),
        },
    ]);
    held.0.insert(
        OTHER_REALM,
        RelayHeldEntry {
            seen: clock.local_tick,
            fence: Fence(3),
            own: seal.clone(),
            interior: Vec::new(),
            digest: relay_entry_digest(Fence(3), &seal, &[]),
        },
    );
    // ZERO WINDOWS: nothing forwards, nothing panics, the holder survives (still fresh).
    emit_window_relays(
        &cfg,
        &clock,
        Fence(1),
        &mut held,
        &mut windows,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.window_relays_forwarded, 0);
    assert_eq!(held.0.len(), 1);
    // ONE WINDOW: the held batch forwards once, VERBATIM, then send-on-change holds its tongue.
    windows.0.insert(
        (GATEWAY, WindowId(1)),
        OpenWindow::opened(WindowScope::Occupants, clock.local_tick),
    );
    emit_window_relays(
        &cfg,
        &clock,
        Fence(1),
        &mut held,
        &mut windows,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.window_relays_forwarded, 1);
    // ONE constructed-expected equality (HR5: no decode-match with an unreachable
    // wildcard): the forwarded bytes ARE the envelope below — the forwarder's OWN fence,
    // the child's fence INTACT, the sealed statements byte-for-byte, unopened.
    let expected = postcard::to_allocvec(&ShardToGateway::WindowRelayed {
        realm_fence: Fence(1),
        window: WindowId(1),
        child: OTHER_REALM,
        child_fence: Fence(3),
        statements: seal.clone(),
        interior: Vec::new(),
    })
    .expect("encodes");
    assert_eq!(outbox.0.len(), 1);
    let (to, _, bytes, _) = &outbox.0[0];
    assert_eq!(*to, GATEWAY);
    assert_eq!(
        &bytes[..],
        &expected[..],
        "forwarded verbatim: own envelope fence, intact child fence, unopened seal"
    );
    emit_window_relays(
        &cfg,
        &clock,
        Fence(1),
        &mut held,
        &mut windows,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(
        stats.window_relays_forwarded, 1,
        "unchanged — send-on-change holds its tongue"
    );
    // TTL: an entry past the derived retain TTL is pruned; the baseline clears with it, so a
    // re-held child re-ships to the same window.
    let later = ClockSample {
        local_tick: vd_core::TickId(10 + retain_ttl_ticks(&cfg) + 1),
        ..clock
    };
    emit_window_relays(
        &cfg,
        &later,
        Fence(1),
        &mut held,
        &mut windows,
        &mut stats,
        &mut outbox,
    );
    assert!(held.0.is_empty(), "pruned by the derived TTL");
    held.0.insert(
        OTHER_REALM,
        RelayHeldEntry {
            seen: later.local_tick,
            fence: Fence(3),
            own: seal.clone(),
            interior: Vec::new(),
            digest: relay_entry_digest(Fence(3), &seal, &[]),
        },
    );
    emit_window_relays(
        &cfg,
        &later,
        Fence(1),
        &mut held,
        &mut windows,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(
        stats.window_relays_forwarded, 2,
        "a re-held child re-ships — the vanished entry cleared its baseline"
    );
}

/// G-VERBATIM (look_horizon.md §6 slice 3; owner-approved 2026-08-17 — RULINGS + §2 ASK A):
/// the bytes a GRANDPARENT forwards are BYTE-IDENTICAL to what the author sealed, across
/// both hops — a measurement that could fail. The full path, on the story world's real
/// lineage: the planet seals its own batch → the SYSTEM holds it and selects it into its
/// up-relay's interior (membership-gated, §3.4.5 — the out-of-verdict arm is driven too) →
/// the GALAXY holds the pair and forwards to a window subscriber — where the encoded
/// `interior[0].own` must be the planet's sealed bytes exactly, its fence intact. Plus the
/// §5.4 third digest component as a measurement: a grandchild-only change (same system own
/// batch) re-ships past send-on-change.
#[test]
fn g_verbatim_a_grandchilds_sealed_bytes_survive_both_hops_byte_identical() {
    let story = Story::new();
    let clock = ClockSample {
        local_tick: vd_core::TickId(4),
        universe_tick: UniverseTick(0),
        epoch: vd_core::EpochId(1),
        synced: true,
    };
    // THE AUTHOR: the planet's own sealed batch (its own look about ITSELF — SL3).
    let sealed_p = vd_wire::session_flow::seal_relay_statements(&[
        vd_wire::session_flow::RelayedStatement::Body {
            subject: story.planet,
            stmt: vd_wire::session_flow::BodyStmt::SelfLook {
                bag: vd_core::look::look_bag(&Boundary::Shell { r: 1.0 }),
            },
            authored_at: UniverseTick(9),
        },
    ]);
    // HOP 1 — the SYSTEM holds the planet's batch sealed.
    let cfg_s = story_config(&story, story.system);
    let p_coord = cfg_s
        .own_coord
        .child(level_of(story.planet).expect("planet level"));
    let mut s_held = RelayHeld::default();
    let mut stats = StubStats::default();
    on_window_relay(
        vd_wire::intershard::WindowRelay {
            child: p_coord,
            realm_fence: Fence(3),
            own: sealed_p.clone(),
            interior: Vec::new(),
        },
        NodeId(70),
        &cfg_s,
        &clock,
        &ChildRealmNodes(BTreeMap::from([(story.planet, NodeId(70))])),
        &mut s_held,
        &mut stats,
    );
    // The §3.4.5 forward gate, BOTH arms: out of the verdict ⇒ the batch stays home;
    // in the verdict ⇒ it joins the interior, fence + bytes VERBATIM.
    assert!(
        build_relay_interior(&s_held, &BTreeSet::new()).is_empty(),
        "a child outside the realm's own in-band verdict is not forwarded (§3.4.5)"
    );
    let interior = build_relay_interior(&s_held, &BTreeSet::from([story.planet]));
    assert_eq!(interior.len(), 1);
    assert_eq!(interior[0].child, story.planet);
    assert_eq!(
        interior[0].child_fence,
        Fence(3),
        "the author's OWN fence, intact"
    );
    assert_eq!(interior[0].own, sealed_p, "hop 1: byte-identical");
    // HOP 2 — the GALAXY holds the system's (own + interior) pair sealed...
    let sealed_s = vd_wire::session_flow::seal_relay_statements(&[
        vd_wire::session_flow::RelayedStatement::Level {
            at: UniverseTick(9),
            rows: Vec::new(),
        },
    ]);
    let cfg_g = story_config(&story, story.galaxy);
    let s_coord = cfg_g
        .own_coord
        .child(level_of(story.system).expect("system level"));
    let mut g_held = RelayHeld::default();
    let g_attested = ChildRealmNodes(BTreeMap::from([(story.system, NodeId(71))]));
    on_window_relay(
        vd_wire::intershard::WindowRelay {
            child: s_coord.clone(),
            realm_fence: Fence(5),
            own: sealed_s.clone(),
            interior: interior.clone(),
        },
        NodeId(71),
        &cfg_g,
        &clock,
        &g_attested,
        &mut g_held,
        &mut stats,
    );
    // ...and forwards it to a window subscriber. ONE constructed-expected byte equality:
    // the wire bytes ARE the planet's sealed bytes riding inside, unopened at every hop.
    let mut windows = OpenWindows::default();
    windows.0.insert(
        (GATEWAY, WindowId(1)),
        OpenWindow::opened(WindowScope::Occupants, clock.local_tick),
    );
    let mut outbox = OutboundBox::default();
    emit_window_relays(
        &cfg_g,
        &clock,
        Fence(6),
        &mut g_held,
        &mut windows,
        &mut stats,
        &mut outbox,
    );
    let expected = postcard::to_allocvec(&ShardToGateway::WindowRelayed {
        realm_fence: Fence(6),
        window: WindowId(1),
        child: story.system,
        child_fence: Fence(5),
        statements: sealed_s.clone(),
        interior: interior.clone(),
    })
    .expect("encodes");
    assert_eq!(outbox.0.len(), 1);
    let (to, _, bytes, _) = &outbox.0[0];
    assert_eq!(*to, GATEWAY);
    assert_eq!(
        &bytes[..],
        &expected[..],
        "G-VERBATIM: what the grandparent forwards is what the author sealed, byte for byte"
    );
    // THE §5.4 THIRD COMPONENT, measured: the planet's picture changes while the system's
    // OWN statements stay identical — send-on-change must re-ship (the re-keyed digest), or
    // a grandchild's change would be invisible on the load-bearing draw path.
    let sealed_p2 = vd_wire::session_flow::seal_relay_statements(&[
        vd_wire::session_flow::RelayedStatement::Body {
            subject: story.planet,
            stmt: vd_wire::session_flow::BodyStmt::SelfLook {
                bag: vd_core::look::look_bag(&Boundary::Shell { r: 2.0 }),
            },
            authored_at: UniverseTick(10),
        },
    ]);
    on_window_relay(
        vd_wire::intershard::WindowRelay {
            child: s_coord,
            realm_fence: Fence(5),
            own: sealed_s.clone(), // the system's own half UNCHANGED
            interior: vec![vd_wire::intershard::InteriorRelay {
                child: story.planet,
                child_fence: Fence(3),
                own: sealed_p2.clone(),
            }],
        },
        NodeId(71),
        &cfg_g,
        &clock,
        &g_attested,
        &mut g_held,
        &mut stats,
    );
    emit_window_relays(
        &cfg_g,
        &clock,
        Fence(6),
        &mut g_held,
        &mut windows,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(
        stats.window_relays_forwarded, 2,
        "a grandchild-only change re-keys the digest and re-ships (§5.4)"
    );
    // The read-only gate views hand back BYTES, never values (the same seal discipline):
    // the held own half and the held interior half, verbatim.
    assert_eq!(g_held.statements_for(story.system), Some(sealed_s));
    assert_eq!(
        g_held.interior_for(story.system),
        Some(vec![vd_wire::intershard::InteriorRelay {
            child: story.planet,
            child_fence: Fence(3),
            own: sealed_p2,
        }])
    );
}

/// look_horizon.md §6 SLICE 6 — RELAY EGRESS, measured against §5.2's formula
/// `W × C × blob × rate`, WITH the gateway multiplier MEASURED rather than assumed: the same
/// held children re-ship the same number of times under one open window and under two, and
/// the two-window egress must be EXACTLY double; with a second held child the egress must be
/// exactly the per-child blob sum times W times the re-ship count. The blob is the
/// DEPARTURE-FIXTURE class, built through the REAL codecs at THE world's own numbers (the
/// home system's sealed batch: its 5-row level + its own look + 5 markers, plus the 5
/// slice-3 interior batches), and its size is PINNED — §5.4's law that this design does not
/// change rows per fold has its bytes-per-relay counterpart pinned here.
#[test]
fn slice6_relay_egress_measured_equals_w_times_c_times_blob_times_rate() {
    let cfg_w = vd_physics::worldgen::UniverseConfig::world(15.0, 0.05);
    let world = vd_physics::worldgen::WorldView::generated(0, &cfg_w);
    let galaxy = vd_core::worldgen::GALAXY;
    let systems: Vec<RealmId> = world
        .regions()
        .iter()
        .filter(|r| r.parent == Some(galaxy))
        .map(|r| r.realm)
        .collect();
    let home = vd_core::worldgen::default_home_realm(world.regions()).expect("home");
    let sibling = *systems
        .iter()
        .find(|s| **s != home)
        .expect("a ring sibling");
    let luma: BTreeMap<RealmId, (u8, f64)> =
        vd_physics::worldgen::system_photometrics_for_config(0, &cfg_w)
            .into_iter()
            .map(|(r, p)| (r, vd_physics::worldgen::marker_datum(&p)))
            .collect();
    let shape_of = |realm: RealmId| {
        world
            .regions()
            .iter()
            .find(|r| r.realm == realm)
            .expect("rostered")
            .shape
    };
    // ONE system's departure-fixture batch at tick `t`, through the real codecs: the level
    // rows carry the planets' true closed-form positions, so consecutive ticks genuinely
    // change the bytes (the re-ship trigger is the fingerprint over row VALUES).
    let batch_at =
        |system: RealmId, t: u64| -> (Vec<u8>, Vec<vd_wire::intershard::InteriorRelay>) {
            let at = UniverseTick(t);
            let movers = vd_physics::worldgen::moving_children_for_config(0, &cfg_w, system);
            assert_eq!(
                movers.len(),
                9,
                "THE world's systems orbit the derived 9 planets"
            );
            let sys_frame = vd_core::pose::frame_for_realm(system, Some(galaxy)).expect("frame");
            let rows: Vec<vd_wire::channels::RealmSnap> = movers
                .iter()
                .map(|(p, e)| {
                    let st = vd_physics::motion::Motion::Kepler(*e)
                        .state_at(t as f64 * cfg_w.interest.tick_dt_s);
                    vd_wire::channels::RealmSnap {
                        realm: *p,
                        frame: vd_core::pose::frame_for_realm(*p, Some(system)).expect("frame"),
                        pose: vd_core::pose::StampedPose {
                            frame: sys_frame,
                            pos: st.anchor(),
                            vel: st.velocity,
                            orient: vd_core::glam::DQuat::IDENTITY,
                            universe_tick: at,
                        },
                    }
                })
                .collect();
            let mut stmts = vec![
                vd_wire::session_flow::RelayedStatement::Level { at, rows },
                vd_wire::session_flow::RelayedStatement::Body {
                    subject: system,
                    stmt: vd_wire::session_flow::BodyStmt::SelfLook {
                        bag: vd_core::look::look_bag(&shape_of(system)),
                    },
                    authored_at: at,
                },
            ];
            for (p, _) in &movers {
                stmts.push(vd_wire::session_flow::RelayedStatement::Body {
                    subject: *p,
                    stmt: vd_wire::session_flow::BodyStmt::Marker {
                        luma: vd_core::look::marker_bag(
                            luma.get(p).copied(),
                            shape_of(*p).circumscribed_extent(),
                        ),
                    },
                    authored_at: at,
                });
            }
            let own = vd_wire::session_flow::seal_relay_statements(&stmts);
            let interior = movers
                .iter()
                .map(|(p, _)| vd_wire::intershard::InteriorRelay {
                    child: *p,
                    child_fence: Fence(3),
                    own: vd_wire::session_flow::seal_relay_statements(&[
                        vd_wire::session_flow::RelayedStatement::Level {
                            at,
                            rows: Vec::new(),
                        },
                        vd_wire::session_flow::RelayedStatement::Body {
                            subject: *p,
                            stmt: vd_wire::session_flow::BodyStmt::SelfLook {
                                bag: vd_core::look::look_bag(&shape_of(*p)),
                            },
                            authored_at: at,
                        },
                    ]),
                })
                .collect();
            (own, interior)
        };
    // One measured run: `w` open windows, `children` held, `n` digest-changing re-ships.
    const N_RESHIPS: u64 = 3;
    let run = |w: u64, children: &[RealmId]| -> (u64, BTreeMap<RealmId, usize>, u64) {
        let cfg = config();
        let clock = ClockSample {
            local_tick: vd_core::TickId(4),
            universe_tick: UniverseTick(0),
            epoch: vd_core::EpochId(1),
            synced: true,
        };
        let mut held = RelayHeld::default();
        let mut windows = OpenWindows::default();
        for i in 0..w {
            windows.0.insert(
                (NodeId(400 + i), WindowId(1)),
                OpenWindow::opened(WindowScope::Occupants, clock.local_tick),
            );
        }
        let mut stats = StubStats::default();
        let mut outbox = OutboundBox::default();
        for t in 0..N_RESHIPS {
            for c in children {
                let (own, interior) = batch_at(*c, t);
                let digest = relay_entry_digest(Fence(5), &own, &interior);
                held.0.insert(
                    *c,
                    RelayHeldEntry {
                        seen: clock.local_tick,
                        fence: Fence(5),
                        own,
                        interior,
                        digest,
                    },
                );
            }
            emit_window_relays(
                &cfg,
                &clock,
                Fence(6),
                &mut held,
                &mut windows,
                &mut stats,
                &mut outbox,
            );
        }
        // THE WIRE, matched byte-for-byte against a CONSTRUCTED expectation (g_verbatim's
        // move): the emit order is deterministic (BTreeMap: windows by node, then held
        // children by realm), so every message's full bytes are re-derivable — no decode,
        // no refutable match, and per-child blob sizes fall out of the zip. One child's
        // blob size must be constant across re-ships (values change, layout does not).
        let mut kids: Vec<RealmId> = children.to_vec();
        kids.sort_unstable();
        let mut blobs: BTreeMap<RealmId, usize> = BTreeMap::new();
        let mut total = 0u64;
        let mut i = 0usize;
        for t in 0..N_RESHIPS {
            for wi in 0..w {
                for c in &kids {
                    let (own, interior) = batch_at(*c, t);
                    let expected = postcard::to_allocvec(&ShardToGateway::WindowRelayed {
                        realm_fence: Fence(6),
                        window: WindowId(1),
                        child: *c,
                        child_fence: Fence(5),
                        statements: own,
                        interior,
                    })
                    .expect("encodes");
                    let (to, _, bytes, _) = &outbox.0[i];
                    assert_eq!(*to, NodeId(400 + wi), "the emit order is the derived one");
                    assert_eq!(
                        &bytes[..],
                        &expected[..],
                        "each relay is byte-identical to its constructed expectation"
                    );
                    total += bytes.len() as u64;
                    let prior = blobs.insert(*c, bytes.len());
                    assert_eq!(
                        prior.unwrap_or(bytes.len()),
                        bytes.len(),
                        "one child's blob size is constant across re-ships"
                    );
                    i += 1;
                }
            }
        }
        assert_eq!(i, outbox.0.len(), "every emitted message was matched");
        (total, blobs, stats.window_relays_forwarded)
    };
    // W = 1, C = 1 — the base measurement.
    let (total_w1, blobs_w1, fwd_w1) = run(1, &[home]);
    let blob = *blobs_w1.get(&home).expect("the home blob was measured");
    assert_eq!(
        fwd_w1, N_RESHIPS,
        "every tick re-ships (the digest changed)"
    );
    assert_eq!(
        total_w1,
        blob as u64 * N_RESHIPS,
        "W=1, C=1: egress == blob × rate"
    );
    // THE GATEWAY MULTIPLIER, measured: two windows double the egress exactly.
    let (total_w2, _, fwd_w2) = run(2, &[home]);
    assert_eq!(fwd_w2, 2 * N_RESHIPS);
    assert_eq!(
        total_w2,
        2 * total_w1,
        "W=2 doubles the egress: the W term is real, not assumed"
    );
    // THE CHILD MULTIPLIER, measured: a second held child (the ring sibling, same 5-planet
    // class) adds exactly its own per-child blob, per window, per re-ship.
    let (total_c2, blobs_c2, fwd_c2) = run(2, &[home, sibling]);
    let blob_sib = *blobs_c2
        .get(&sibling)
        .expect("the sibling blob was measured");
    assert_eq!(fwd_c2, 2 * 2 * N_RESHIPS);
    assert_eq!(
        total_c2,
        2 * N_RESHIPS * (blob as u64 + blob_sib as u64),
        "W=2, C=2: egress == W × Σ_child blob × rate",
    );
    // THE PIN (slice 6): the departure-fixture blob in bytes — §5.2's ~1 045 B estimate,
    // measured. A wire or codec change that moves this number flips it loudly.
    // RE-BASELINED 1118 → 1142 B at the cell activation (real-scale addendum §A6.4, the ONE
    // measured wire cost): every pose now rides normalized, so the postcard varints carry the
    // integer cells (+24 B here). Schema unchanged, PROTO_MINOR unmoved.
    assert_eq!(
        blob, 2124,
        "the pinned departure-fixture relay blob (bytes)"
    );
    eprintln!(
        "[slice6] RELAY EGRESS measured == formula: blob {blob} B (sibling {blob_sib} B; \
         §5.2 modelled ~1045 B) · W=1 egress {total_w1} B / {N_RESHIPS} re-ships · W=2 \
         doubles exactly · W=2,C=2 {total_c2} B == 2 × {N_RESHIPS} × ({blob}+{blob_sib}). \
         At the §5.1 reference model (W=50, C=3, 50 Hz re-ship): {:.1} MB/s per shard",
        50.0 * 3.0 * blob as f64 * 50.0 / 1.0e6,
    );
}

/// look_horizon.md §6 SLICE 6 — THE UNION OVER-DRAW, measured at interim scale (D-LOOK-2):
/// the shared per-realm verdict is the UNION over every observer's in-band set (the owner's
/// one-verdict-per-realm ruling), so rows are drawn for one observer because ANOTHER
/// observer's verdict is more generous. Two occupants of THE world's galaxy — one parked
/// near the home star, one 300 m short of a ring sibling — and the measurement: the fold's
/// published verdict equals the union of the two out-of-band singleton sets, and each
/// observer over-draws exactly the OTHER's child (its own look + its 5-planet interior = 6
/// drawn realm rows per fold, at the departure-fixture row model). The number the D-LOOK-2
/// ledger row records; UNMEASURED at near-real scale, by design.
#[test]
fn slice6_the_union_over_draw_is_measured_at_interim_scale() {
    let cfg_w = vd_physics::worldgen::UniverseConfig::world(15.0, 0.05);
    let world = vd_physics::worldgen::WorldView::generated(0, &cfg_w);
    let galaxy = vd_core::worldgen::GALAXY;
    let scope = world.neighbourhood(&BTreeSet::from([galaxy]));
    let galaxy_row = scope
        .iter()
        .find(|r| r.realm == galaxy)
        .expect("the galaxy is rostered");
    let home = vd_core::worldgen::default_home_realm(world.regions()).expect("home");
    // ★ AT THE GALAXY'S RUNG, NOT THE CHILD'S (slice S9). Every region read here is a direct child of
    // the galaxy, so its `center` is stated in the GALAXY's frame — and reading it with the child's
    // own ruler was out by 2048×. What made this one instructive is that the ORACLE and the fixture's
    // observer positions both used the wrong ruler, so they agreed with each other perfectly; only
    // the REAL fold, which reads the centre correctly, disagreed. Two wrongs that agree look exactly
    // like a right answer until something honest shows up.
    // The step is read off the galaxy ROW, which is the parent of every region this closure sees —
    // `metres_in` takes the parent itself, so a child's step cannot be substituted for it.
    let centre_of = |r: &vd_core::geometry::RealmRegion| r.center.metres_in(galaxy_row);
    let systems: Vec<(RealmId, DVec3)> = scope
        .iter()
        .filter(|r| r.parent == Some(galaxy))
        .map(|r| (r.realm, centre_of(r)))
        .collect();
    assert_eq!(systems.len(), 3, "THE galaxy holds 3 systems");
    let (sib, sib_centre) = *systems
        .iter()
        .find(|(s, _)| *s != home)
        .expect("a ring sibling");
    // Observer A: 300 m from the home star (deep inside its 11 458 m visibility band).
    // Observer B: 300 m short of the ring sibling, on the line toward the galaxy origin.
    let pos_a = DVec3::new(0.0, 0.0, -300.0);
    let pos_b = sib_centre - sib_centre.normalize() * 300.0;
    // THE OUT-OF-BAND SINGLETON SETS (the oracle): which children each observer's own
    // position puts in band — the per-observer verdict a per-session fold WOULD have
    // computed. Asserted disjoint singletons so the union measurement has teeth.
    let in_band_of = |pos: DVec3| -> BTreeSet<RealmId> {
        scope
            .iter()
            .filter(|r| r.parent == Some(galaxy))
            .filter(|r| r.aoi.in_range(false, (centre_of(r) - pos).length()))
            .map(|r| r.realm)
            .collect()
    };
    let set_a = in_band_of(pos_a);
    let set_b = in_band_of(pos_b);
    assert_eq!(
        set_a,
        BTreeSet::from([home]),
        "A sees exactly the home star"
    );
    assert_eq!(set_b, BTreeSet::from([sib]), "B sees exactly its sibling");
    // THE REAL FOLD, both observers held: the galaxy shard's own AoI tick publishes the
    // SHARED verdict.
    let cfg = StubConfig {
        realm: galaxy,
        held_realms: StubConfig::single_realm(galaxy),
        frame: galaxy_row.frame,
        own_coord: vd_core::worldgen::coord_of_realm(&scope, galaxy)
            .expect("the galaxy has a lineage"),
        ..config()
    };
    let mut rig = Rig::with_config(cfg);
    // The grant, keyed on THE GALAXY (the shared `grant_realm` helper grants the default
    // rig realm, which this fixture is not).
    {
        let reply = DirectoryReply::Head {
            key: DirectoryKey::Realm(galaxy),
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
        let _ = rig.tick(vec![Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes,
        }]);
    }
    plant_aoi(&mut rig, scope.clone());
    let own_frame = galaxy_row.frame;
    for (i, pos) in [(1u32, pos_a), (2, pos_b)] {
        let entity = EntityId::pack(EntityKind::Player, 60, u64::from(i), i);
        let mut dot = slice6_dot(entity, own_frame, Authority::Owned { fence: Fence(1) });
        dot.pose = vd_core::pose::StampedPose::at_rest(own_frame, pos, UniverseTick(100));
        rig.world
            .resource_mut::<Dots>()
            .0
            .insert(SessionId(u128::from(i)), dot);
    }
    let _ = rig.tick(vec![]);
    let verdict = rig.world.resource::<InBandVerdict>().0.clone();
    let union: BTreeSet<RealmId> = set_a.union(&set_b).copied().collect();
    assert_eq!(
        verdict, union,
        "the shared verdict IS the union of the observers' own sets"
    );
    // THE OVER-DRAW, recorded (D-LOOK-2's interim-scale measurement): each observer's fold
    // carries the OTHER observer's child — its own look plus its interior's 5 planets = 6
    // drawn realm rows per fold that this observer's own position never asked for.
    let interior_rows_of =
        |s: RealmId| 1 + vd_physics::worldgen::moving_children_for_config(0, &cfg_w, s).len();
    let overdraw_a: usize = union.difference(&set_a).map(|s| interior_rows_of(*s)).sum();
    let overdraw_b: usize = union.difference(&set_b).map(|s| interior_rows_of(*s)).sum();
    assert_eq!(
        (overdraw_a, overdraw_b),
        (10, 10),
        "each observer over-draws exactly the other's 1-look + 9-planet subtree"
    );
    eprintln!(
        "[slice6] UNION OVER-DRAW at interim scale (D-LOOK-2): union verdict {} children \
         (A's own {}, B's own {}); over-draw {} rows per fold per observer (1 own look + 5 \
         interior rows for the child only the OTHER observer needs) — measured at interim \
         scale ONLY; the near-real-scale number stays owed on the ledger row",
        union.len(),
        set_a.len(),
        set_b.len(),
        overdraw_a,
    );
}

/// G-STRUCTURAL-SEAL (look_horizon.md §6 slice 3): production code in `vd-sim` contains NO
/// call to any relay-open function — the HR1 guarantee across two hops as a TEST, not a
/// claim. The scan strips line comments and stops at each file's `#[cfg(test)]` boundary
/// (tests may lawfully open a seal to assert its content; a realm never does). Two positive
/// controls keep it honest: the needle exists in the raw sources (so the scanner hunts a
/// real name), and the crate's own files were actually walked.
#[test]
fn g_structural_seal_vd_sim_production_code_calls_no_open_function() {
    let needle = ["open_", "relay"].concat(); // split so this file's own scan cannot self-trip
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    let mut pending = vec![src];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).expect("vd-sim src dir reads") {
            let path = entry.expect("dir entry reads").path();
            if path.is_dir() {
                pending.push(path);
            } else {
                // EVERY file — a stray non-source file reads lossily and scans inert,
                // which is cheaper than an uncoverable extension-filter arm (HR5).
                files.push(path);
            }
        }
    }
    assert!(
        files.iter().any(|p| p.ends_with("stub.rs")),
        "anti-vacuity: the holder's own file is in the walk"
    );
    let mut raw_hits = 0usize;
    for path in &files {
        let content =
            String::from_utf8_lossy(&std::fs::read(path).expect("source reads")).into_owned();
        raw_hits += content.matches(&needle).count();
        let production = content
            .split_once("#[cfg(test)]")
            .map_or(content.as_str(), |(prod, _)| prod);
        for (n, line) in production.lines().enumerate() {
            let code = line.split("//").next().expect("split yields a first part");
            let clean = !code.contains(&needle);
            let at = format!("{}:{}", path.display(), n + 1);
            assert!(
                clean,
                "G-STRUCTURAL-SEAL: {at} calls a relay-open function in production code — \
                 a realm may hold and forward sealed bytes, never read them (HR1; the Q2 \
                 ruling; look_horizon.md §2 ASK A)"
            );
        }
    }
    assert!(
        raw_hits > 0,
        "anti-vacuity: the needle never appears anywhere — the scanner hunts a dead name"
    );
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

/// THE INTEREST BYTE's admission (look horizon slice 4, §2 ASK B's fail-closed shape),
/// every refusal arm driven by name: mis-route, unattested sender (which arms the lazy
/// PARENT head re-read — the re-home backstop mirroring `retain_child_live`), an unlawful
/// value, a deposed incarnation's stale byte — and the lawful path holding the LATEST value
/// whole, `0` included, so the fence ordering survives an explicit switch-off.
#[test]
fn the_interest_byte_admits_fail_closed_and_holds_the_latest_lawful_value() {
    let story = Story::new();
    let cfg = story_config(&story, story.system);
    let clock = ClockSample {
        local_tick: vd_core::TickId(4),
        universe_tick: UniverseTick(0),
        epoch: vd_core::EpochId(1),
        synced: true,
    };
    let parent = ParentRealmNode(Some(NodeId(40)));
    let mut held = InterestHeld::default();
    let mut stats = StubStats::default();
    let mut outbox = OutboundBox::default();
    let ri =
        |child: RealmCoord, fence: u64, at: u64, look: u8| vd_wire::intershard::RealmInterest {
            child,
            parent_fence: Fence(fence),
            at: UniverseTick(at),
            look_inside: look,
        };
    // MIS-ROUTE: a coord lowering to somebody else (the system's own planet) drops counted.
    let planet_coord = cfg
        .own_coord
        .child(level_of(story.planet).expect("planet level"));
    on_realm_interest(
        ri(planet_coord, 5, 1, 1),
        NodeId(40),
        &cfg,
        &clock,
        &parent,
        &mut held,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.realm_interest_misrouted, 1);
    assert_eq!(held.0, None, "nothing believed");
    // UNATTESTED: right coord, wrong sender — refused + the lazy parent head re-read armed.
    on_realm_interest(
        ri(cfg.own_coord.clone(), 5, 1, 1),
        NodeId(41),
        &cfg,
        &clock,
        &parent,
        &mut held,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.realm_interest_unattested, 1);
    assert_eq!(held.0, None);
    let parent_realm = cfg
        .own_coord
        .parent()
        .expect("the story system has a parent")
        .lowered();
    assert_eq!(outbox.0.len(), 1, "the unattested arm arms ONE re-read");
    let (to, _, bytes, _) = &outbox.0[0];
    assert_eq!(*to, cfg.orchestrator);
    assert_eq!(
        postcard::from_bytes::<InterShardFlow>(bytes).expect("the re-read decodes"),
        InterShardFlow::Directory(DirectoryOp::HeadRead {
            key: DirectoryKey::Realm(parent_realm),
        }),
        "…and it is the PARENT's head read (the re-home backstop)"
    );
    // UNATTESTED AT A ROOT: a realm with no parent coord refuses the same way but has no
    // parent head to re-read — the re-read arm is skipped, nothing armed, nothing believed.
    let root_cfg = config(); // own_coord = root_coord(System(7)) — parent() is None
    let outbox_before = outbox.0.len();
    on_realm_interest(
        ri(root_cfg.own_coord.clone(), 5, 1, 1),
        NodeId(41),
        &root_cfg,
        &clock,
        &parent,
        &mut held,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.realm_interest_unattested, 2);
    assert_eq!(
        outbox.0.len(),
        outbox_before,
        "a rootward refusal arms no re-read — there is no parent to resolve"
    );
    // UNLAWFUL VALUE: 1 or 0, nothing else.
    on_realm_interest(
        ri(cfg.own_coord.clone(), 5, 1, 2),
        NodeId(40),
        &cfg,
        &clock,
        &parent,
        &mut held,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.realm_interest_unlawful, 1);
    assert_eq!(held.0, None);
    // LAWFUL 1: held whole.
    on_realm_interest(
        ri(cfg.own_coord.clone(), 5, 2, 1),
        NodeId(40),
        &cfg,
        &clock,
        &parent,
        &mut held,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.realm_interest_received, 1);
    assert_eq!(
        held.0,
        Some(InterestEntry {
            fence: Fence(5),
            at: UniverseTick(2),
            seen: clock.local_tick,
            look_inside: 1,
        })
    );
    // STALE: a deposed incarnation (fence below the held one) never regresses the entry.
    on_realm_interest(
        ri(cfg.own_coord.clone(), 4, 9, 0),
        NodeId(40),
        &cfg,
        &clock,
        &parent,
        &mut held,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.realm_interest_stale, 1);
    assert_eq!(
        held.0.map(|e| e.look_inside),
        Some(1),
        "the zombie changed nothing"
    );
    // LAWFUL 0 at a fresher stamp: stored WHOLE — the ordering survives the switch-off.
    on_realm_interest(
        ri(cfg.own_coord.clone(), 5, 3, 0),
        NodeId(40),
        &cfg,
        &clock,
        &parent,
        &mut held,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.realm_interest_received, 2);
    assert_eq!(
        held.0,
        Some(InterestEntry {
            fence: Fence(5),
            at: UniverseTick(3),
            seen: clock.local_tick,
            look_inside: 0,
        }),
        "an explicit 0 is held, not erased — the fence high-water survives"
    );
}

/// THE DOWN-PROXY + THE STRUCTURAL CASCADE CAP (look horizon slice 4, §3.4.3; the G-NO-CASCADE
/// structural half): a VACATED realm holding a live interest byte inserts ONE synthetic
/// observer at its own centre and its existing fold demands its in-band interior — while the
/// OCCUPANCY truths stay exactly what they were: it still self-reports Empty, ships NO
/// `ChildLive` bit, and — with a live interior band on its child, a resolved route, and the
/// synthetic observer standing in band, so the origin flag is the ONLY thing left to stop it
/// — emits NO interest of its own downward. The byte then DECAYS on the derived TTL and the
/// wake ends: "nobody is watching".
#[test]
fn a_live_interest_byte_wakes_the_interior_reports_empty_and_never_cascades() {
    let mut rig = Rig::new();
    rig.grant_realm();
    // The child carries a LIVE interior band + a resolved head route: everything an unlawful
    // cascade would need is present, on purpose.
    let child = RealmRegion {
        interior_band: aoi_band(0),
        ..aoi_child(OTHER_REALM, OWN_REALM, 100.0, 0)
    };
    plant_aoi(&mut rig, vec![root_region(), own_region(), child]);
    rig.world
        .resource_mut::<ChildRealmNodes>()
        .0
        .insert(OTHER_REALM, NodeId(70));
    rig.world.resource_mut::<ParentRealmNode>().0 = Some(ANCESTOR);
    // The byte arrives through the REAL dispatch (the reliable peer carrier the relay
    // rides), from the resolved parent head — the same admission production runs.
    let byte = InterShardFlow::RealmInterest(vd_wire::intershard::RealmInterest {
        child: rig.world.resource::<StubConfig>().own_coord.clone(),
        parent_fence: Fence(5),
        at: UniverseTick(50),
        look_inside: 1,
    });
    let sent = rig.tick(vec![Inbound::Wire {
        from: ANCESTOR,
        class: MsgClass::Saga,
        bytes: crate::io::bytes(postcard::to_allocvec(&byte).expect("encode")),
    }]);
    let d = demands(&sent);
    let child_coord = child_coord_of(OWN_REALM, OTHER_REALM);
    assert!(
        d.iter()
            .any(|d| (d.child == child_coord) & (d.verb == DemandVerb::SpinUp)),
        "the down-proxy woke the interior: {d:?}"
    );
    assert!(
        d.iter()
            .any(|d| (d.child.lowered() == OWN_REALM) & (d.verb == DemandVerb::Empty)),
        "…while the realm STILL truthfully reports Empty (occupancy is occupancy): {d:?}"
    );
    let bits = sent
        .iter()
        .filter(|(_, _, b)| {
            matches!(
                postcard::from_bytes::<InterShardFlow>(b),
                Ok(InterShardFlow::ChildLive(_))
            )
        })
        .count();
    assert_eq!(
        bits, 0,
        "a byte from outside must never manufacture an occupancy bit"
    );
    assert!(
        interests(&sent).is_empty(),
        "THE CAP, structurally: route + live band + in-band synthetic observer, and still \
         no interest goes down — only occupancy-derived observers produce interest"
    );
    assert!(
        rig.world
            .resource::<InBandVerdict>()
            .0
            .contains(&OTHER_REALM),
        "the shared verdict names the interior (§3.4.5 — the outside watcher's forward gate)"
    );
    // AN EXPLICIT 0 (still fresh, held whole): powers NO proxy — the fold runs empty.
    let off = InterShardFlow::RealmInterest(vd_wire::intershard::RealmInterest {
        child: rig.world.resource::<StubConfig>().own_coord.clone(),
        parent_fence: Fence(5),
        at: UniverseTick(51),
        look_inside: 0,
    });
    rig.set_local_tick(2);
    let sent = rig.tick(vec![Inbound::Wire {
        from: ANCESTOR,
        class: MsgClass::Saga,
        bytes: crate::io::bytes(postcard::to_allocvec(&off).expect("encode")),
    }]);
    assert_eq!(
        rig.world
            .resource::<InterestHeld>()
            .0
            .map(|e| e.look_inside),
        Some(0),
        "the 0 was admitted through the dispatch and held whole"
    );
    assert!(
        demands(&sent)
            .iter()
            .all(|d| d.child != child_coord_of(OWN_REALM, OTHER_REALM)),
        "a held 0 wakes nothing — do-not-assume is as lawful as assume"
    );
    // DECAY: the lane goes silent past the derived TTL ⇒ the byte is gone, the wake ends
    // (the TTL base is the 0-entry's own `seen = 2`).
    rig.set_local_tick(2 + retain_ttl_ticks(&config()) + 1);
    let sent = rig.tick(vec![]);
    assert_eq!(
        rig.world.resource::<InterestHeld>().0,
        None,
        "silence decays to nobody-is-watching"
    );
    let d = demands(&sent);
    assert!(
        d.iter().all(|d| d.child != child_coord),
        "the expired byte demands nothing: {d:?}"
    );
}

/// THE INTEREST EMISSION's edges (look horizon slice 4, §2 ASK B / G-INTEREST-BAND's unit
/// half): a `1` fires the tick the band is first entered (the transition itself — the
/// finding-41 doctrine), re-asserts on the AoI beat and ONLY the beat while the band holds
/// (the hysteresis hold zone included), and ONE explicit `0` fires on the falling edge; the
/// dead zone between spin-up and tear-down admits nobody fresh (the band-edge gate: it
/// spins up none).
#[test]
fn the_interest_emission_rises_on_entry_beats_and_falls_to_zero_at_the_band() {
    let mut rig = Rig::new();
    rig.grant_realm();
    // A hand-derived interior band: spin-up 400 m, tear-down 400 + 2·0.05·2.5 = 400.25 m.
    let band = vd_core::geometry::AoiConfig::for_velocity_safe(400.0, 1.0, 1.0, 2.0, 0.05, 0, 0.5)
        .expect("a live interior band");
    let child = RealmRegion {
        interior_band: band,
        ..aoi_child(OTHER_REALM, OWN_REALM, 100.0, 0)
    };
    plant_aoi(&mut rig, vec![root_region(), own_region(), child]);
    rig.world
        .resource_mut::<ChildRealmNodes>()
        .0
        .insert(OTHER_REALM, NodeId(70));
    let child_coord = child_coord_of(OWN_REALM, OTHER_REALM);
    // NO ROUTE YET: the occupant stands in band before the child's head is resolved —
    // nothing is sent, the latch stays untouched, and the NEXT pass (route in hand) still
    // fires the rising edge (fail-closed, self-healing — never through the orchestrator).
    rig.world.resource_mut::<ChildRealmNodes>().0.clear();
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(350.0, 0.0, 0.0),
    );
    rig.set_local_tick(10);
    assert!(
        interests(&rig.tick(vec![])).is_empty(),
        "an unresolved head sends nothing — and must not consume the edge"
    );
    rig.world
        .resource_mut::<ChildRealmNodes>()
        .0
        .insert(OTHER_REALM, NodeId(70));
    // RISING EDGE, off-beat: an occupant enters the band ⇒ the 1 ships at once, direct to
    // the child's attested head.
    rig.set_local_tick(11);
    let ints = interests(&rig.tick(vec![]));
    assert_eq!(ints.len(), 1, "the rising edge ships immediately");
    assert_eq!(
        ints[0].0,
        NodeId(70),
        "direct to the child's head, nowhere else"
    );
    assert_eq!(ints[0].1.look_inside, 1);
    assert_eq!(ints[0].1.child, child_coord);
    // Between beats: send-on-beat holds its tongue.
    rig.set_local_tick(12);
    assert!(
        interests(&rig.tick(vec![])).is_empty(),
        "no beat, no edge, no byte"
    );
    // On the beat: re-asserted (the receiver's TTL is sized against this).
    rig.set_local_tick(20);
    let ints = interests(&rig.tick(vec![]));
    assert_eq!(ints.len(), 1);
    assert_eq!(ints[0].1.look_inside, 1);
    // THE HOLD ZONE: between spin-up and tear-down an acquired child is kept…
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(400.1, 0.0, 0.0),
    );
    rig.set_local_tick(30);
    let ints = interests(&rig.tick(vec![]));
    assert_eq!(ints.len(), 1, "held inside the dead zone");
    assert_eq!(ints[0].1.look_inside, 1);
    // FALLING EDGE, off-beat: past tear-down the explicit 0 ships once.
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(401.0, 0.0, 0.0),
    );
    rig.set_local_tick(31);
    let ints = interests(&rig.tick(vec![]));
    assert_eq!(
        ints.len(),
        1,
        "the falling edge ships the explicit 0 at once"
    );
    assert_eq!(ints[0].1.look_inside, 0);
    // OUT, on a beat: silent (nothing to re-assert).
    rig.set_local_tick(40);
    assert!(interests(&rig.tick(vec![])).is_empty());
    // THE BAND EDGE from outside: the dead zone admits nobody fresh — it spins up none.
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(400.1, 0.0, 0.0),
    );
    rig.set_local_tick(50);
    assert!(
        interests(&rig.tick(vec![])).is_empty(),
        "between spin-up and tear-down, a FRESH approach is not yet interest"
    );
}

/// THE SELF-PROXY EXCLUSION (the warp gate's red of 2026-08-17, root-caused to this arm):
/// a child's OWN occupied-child proxy never produces interest TO that child — the byte
/// exists to make "something OUTSIDE may be looking in" representable (Q1's exact
/// sentence), and a child's own occupants are inside it, already held, already waking its
/// interior. Without the exclusion the universe flags its occupied galaxy, the galaxy's
/// down-proxy reaches its own full extent, and EVERY system wakes — §2 ASK B's priced and
/// rejected alternative, measured live. The SL7 sibling-warming STAYS: the same proxy
/// still produces interest to a SIBLING inside its band.
#[test]
fn a_childs_own_proxy_never_flags_it_interested_but_still_warms_its_sibling() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let band = vd_core::geometry::AoiConfig::for_velocity_safe(400.0, 1.0, 1.0, 2.0, 0.05, 0, 0.5)
        .expect("a live interior band");
    let child = RealmRegion {
        interior_band: band,
        ..aoi_child(OTHER_REALM, OWN_REALM, 100.0, 0)
    };
    let sibling = RealmId::Planet(43);
    let sibling_region = RealmRegion {
        interior_band: band,
        ..region(sibling, Some(OWN_REALM), DVec3::new(200.0, 0.0, 0.0), 100.0)
    };
    plant_aoi(
        &mut rig,
        vec![root_region(), own_region(), child, sibling_region],
    );
    {
        let mut nodes = rig.world.resource_mut::<ChildRealmNodes>();
        nodes.0.insert(OTHER_REALM, NodeId(70));
        nodes.0.insert(sibling, NodeId(71));
    }
    // The child's fresh occupancy bit makes it an OCCUPIED-CHILD observer standing at its
    // own placement (the origin), reaching its own 100 m extent.
    rig.world.resource_mut::<ChildLiveness>().0.insert(
        OTHER_REALM,
        ChildLiveEntry {
            home: NodeId(70),
            fence: Fence(1),
            at: UniverseTick(1),
            last_seen: vd_core::TickId(9),
        },
    );
    rig.set_local_tick(10); // a beat
    let ints = interests(&rig.tick(vec![]));
    let child_coord = child_coord_of(OWN_REALM, OTHER_REALM);
    let sibling_coord = child_coord_of(OWN_REALM, sibling);
    assert!(
        ints.iter().all(|(_, ri)| ri.child != child_coord),
        "a child's own proxy must never flag the child itself: {ints:?}"
    );
    let to_sibling: Vec<_> = ints
        .iter()
        .filter(|(_, ri)| ri.child == sibling_coord)
        .collect();
    assert_eq!(
        to_sibling.len(),
        1,
        "…while the SAME proxy still warms the sibling inside its band: {ints:?}"
    );
    assert_eq!(to_sibling[0].0, NodeId(71));
    assert_eq!(to_sibling[0].1.look_inside, 1);
}

/// A GRANDCHILD's claim on the scanned point (a region whose parent is another child, not the
/// owning realm): the crossing still moves ONE hop — into the DIRECT child — because travel is
/// always through the parent chain, level by level; the grandchild is the NEXT shard's decision.
/// (Measured: the first draft of this test expected the deepest region to win the request and
/// the machinery correctly refused — the one-hop law is the assertion now.) The grandchild's
/// claim still runs the containment fold's non-direct-child branch, which is the log site this
/// pins.
#[test]
fn a_grandchilds_claim_still_crosses_one_hop_into_the_direct_child() {
    let grandchild = RealmId::Station(77);
    let mut rig = Rig::new();
    rig.grant_realm();
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(vec![
        root_region(),
        own_region(),
        region(OTHER_REALM, Some(OWN_REALM), DVec3::ZERO, 1000.0),
        region(grandchild, Some(OTHER_REALM), DVec3::ZERO, 100.0),
    ]);
    insert_owned_dot(&mut rig, TRIG_SESSION, player(7), DVec3::ZERO);
    let sent = rig.tick(vec![]);
    let decided: Vec<RealmId> = crossing_requests(&sent)
        .iter()
        .map(|r| r.to_realm)
        .collect();
    assert_eq!(
        decided,
        vec![OTHER_REALM],
        "one hop, into the DIRECT child — the grandchild is the next shard's decision"
    );
}

/// The conversion path's hop cap: a parent CYCLE (boot-rejected on any real forest) never
/// reaches a root, so there is NO chain — the path refuses and the flush falls to the verbatim
/// arm, never a hang and never an LCA over a chain that lies.
#[test]
fn a_cyclic_roster_yields_no_conversion_path() {
    let cyclic = RealmRegions::new(vec![
        region(
            RealmId::Planet(1),
            Some(RealmId::Planet(2)),
            DVec3::ZERO,
            10.0,
        ),
        region(
            RealmId::Planet(2),
            Some(RealmId::Planet(1)),
            DVec3::new(100.0, 0.0, 0.0),
            10.0,
        ),
    ]);
    assert_eq!(
        conversion_path(
            &cyclic,
            RealmId::Planet(1),
            RealmId::Planet(1),
            RealmId::Planet(2)
        ),
        None,
        "a cycle has no chain, so it has no path"
    );
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

/// Slice C1, the Q2-relay ship (the up-shape ship's rewritten successor — owner-approved
/// 2026-08-16, owner_decisions_2026-08-15.md addendum + window_lane.md §5 RULINGS): with a
/// resolved parent this shard ships its VERBATIM self-authored statements — sealed — on the
/// parent's resolve (the relay-subscription moment), holds its tongue while nothing changed
/// off the cadence, re-asserts on the AoI cadence (a restarted parent holds relays only in
/// RAM), ships nothing with no parent, and a shard with no self-look and no markers ships
/// nothing at all (absence of data, never an empty batch).
#[test]
fn the_relay_ship_sends_sealed_statements_on_resolve_change_and_cadence() {
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 0),
        ],
    );
    // Parent resolves at an OFF-cadence tick: the memo never matches a fresh parent, so the
    // resolve itself ships (the relay-subscription moment) — no waiting for a beat.
    rig.world.resource_mut::<ParentRealmNode>().0 = Some(ANCESTOR);
    rig.set_local_tick(31);
    let ships = window_relays(&rig.tick(vec![]));
    assert_eq!(ships.len(), 1, "the parent's resolve ships immediately");
    assert_eq!(ships[0].0, ANCESTOR, "addressed to the resolved parent");
    assert_eq!(
        ships[0].1.child.lowered(),
        OWN_REALM,
        "the routing key is this shard's own coord"
    );
    // The seal opens (HERE, in the test — the PARENT never opens it) to the child's verbatim
    // statements. ONE constructed-expected equality (HR5: no let-else destructure whose
    // refusal arm a green test can never take): the authored interior level FIRST — built
    // from the same placement head the producer read — then the realm's own look about
    // ITSELF (SL3: the boot extent, nothing else).
    let opened = vd_wire::session_flow::open_relay_statements(&ships[0].1.own)
        .expect("the child's own seal opens");
    let (expected_rows, at) = {
        let cfg_realm = rig.world.resource::<StubConfig>().realm;
        let at = rig.world.resource::<ClockSample>().universe_tick;
        let placements = rig.world.resource::<Placements>();
        let head = placements.0.head(cfg_realm).expect("the writer authored");
        let regions = rig.world.resource::<RealmRegions>();
        (regions.authored_realm_snaps(cfg_realm, head), at)
    };
    assert_eq!(
        expected_rows.iter().map(|r| r.realm).collect::<Vec<_>>(),
        vec![OTHER_REALM],
        "the authored interior — one row per direct child, nothing deeper"
    );
    assert_eq!(
        opened,
        vec![
            vd_wire::session_flow::RelayedStatement::Level {
                at,
                rows: expected_rows,
            },
            vd_wire::session_flow::RelayedStatement::Body {
                subject: OWN_REALM,
                stmt: vd_wire::session_flow::BodyStmt::SelfLook {
                    bag: vd_core::look::look_bag(&own_region().shape),
                },
                authored_at: at,
            },
            vd_wire::session_flow::RelayedStatement::Body {
                subject: OTHER_REALM,
                stmt: vd_wire::session_flow::BodyStmt::Marker {
                    // The presence floor (look_horizon.md slice 1): the non-glowing child's
                    // point of light — one radius, nothing else.
                    luma: vd_core::look::marker_bag(None, 100.0),
                },
                authored_at: at,
            },
        ],
        "verbatim: the level leads, the self-look follows, one marker per direct child, \
         nothing deeper rides"
    );
    // Unchanged + off-cadence: send-on-change holds its tongue.
    rig.set_local_tick(32);
    assert!(
        window_relays(&rig.tick(vec![])).is_empty(),
        "nothing changed, no beat — no ship"
    );
    // The AoI cadence re-asserts (tick 40, cadence 10): a restarted parent re-learns the seal.
    rig.set_local_tick(40);
    assert_eq!(
        window_relays(&rig.tick(vec![])).len(),
        1,
        "the cadence re-assert ships"
    );
    // Parent unresolved (the root; a boot race): nothing ships, even on the cadence.
    rig.world.resource_mut::<ParentRealmNode>().0 = None;
    rig.set_local_tick(50);
    assert!(
        window_relays(&rig.tick(vec![])).is_empty(),
        "no resolved parent, no ship"
    );
    // A shard whose realm is absent from its forest states no look, and with no marker roster
    // it has NO bodies: parent resolved and on the cadence, it still ships nothing.
    rig.world.resource_mut::<ParentRealmNode>().0 = Some(ANCESTOR);
    plant_aoi(&mut rig, vec![root_region()]);
    rig.set_local_tick(60);
    assert!(
        window_relays(&rig.tick(vec![])).is_empty(),
        "no bodies to state — an empty batch is never sent"
    );
}

#[test]
fn the_two_read_only_diagnosis_accessors_answer_both_ways() {
    // `RelayHeld::statements_for` and `AoiState::in_band` are the ONLY windows a harness or a
    // gate has onto two private stores, so both arms of each are driven here rather than only
    // from the crates that read them (HR5: a crate covers its own surface).
    let mut held = RelayHeld::default();
    assert_eq!(
        held.statements_for(OTHER_REALM),
        None,
        "nothing held for a child that never spoke"
    );
    held.0.insert(
        OTHER_REALM,
        RelayHeldEntry {
            seen: vd_core::TickId(1),
            fence: Fence(2),
            own: vec![7, 7, 7],
            interior: Vec::new(),
            digest: relay_entry_digest(Fence(2), &[7, 7, 7], &[]),
        },
    );
    assert_eq!(
        held.statements_for(OTHER_REALM),
        Some(vec![7, 7, 7]),
        "the SEALED bytes come back verbatim — this hands them over, it never opens them"
    );
    assert!(
        !AoiState::default().in_band(),
        "a fresh latch is out of band"
    );
    let (_, entered) = aoi_transition(AoiState::default(), true, 0);
    assert!(
        entered.expect("entering yields a state").in_band(),
        "and a latch that entered says so"
    );
}

#[test]
fn ttl_alive_and_retain_ttl_are_derived_and_bridge_one_loss() {
    // TTL is DERIVED, never a magic number: the widest of the 1 s loiter constant (round(1.0 /
    // 0.05) = 20 ticks) and TWO beats of the bit's own cadence plus one tick of slack (finding 41
    // — the bit beats per cadence now, so the TTL must bridge one LOST BEAT, not one lost tick).
    // Here the disarmed recheck derives cadence hz/2 = 10 ⇒ the beats term (21) edges out the
    // loiter term (20).
    let normal = StubConfig {
        tick_dt_s: 0.05,
        ..config()
    };
    assert_eq!(aoi_recheck_cadence(&normal), 10);
    assert_eq!(retain_ttl_ticks(&normal), 21);
    // The SHIPPED dev profile (50 Hz, recheck 25): loiter 50 vs beats 51 — the old exactly-two-
    // beats-zero-slack coincidence is now a derivation with slack.
    let shipped = StubConfig {
        tick_dt_s: 0.02,
        realm_recheck_interval: 25,
        ..config()
    };
    assert_eq!(retain_ttl_ticks(&shipped), 51);
    // A degenerate dt: the cadence term dominates everything (an absurd dt derives an absurd
    // cadence; every REAL profile has dt > 0, where the two derived terms above decide).
    let degenerate = StubConfig {
        tick_dt_s: 0.0,
        ..config()
    };
    assert_eq!(
        retain_ttl_ticks(&degenerate),
        RETAIN_TTL_CADENCE_BEATS * aoi_recheck_cadence(&degenerate) + 1
    );
    // Alive predicate: age 0 and age == ttl are alive; age == ttl + 1 has expired.
    assert!(ttl_alive(vd_core::TickId(10), vd_core::TickId(10), 3));
    assert!(ttl_alive(vd_core::TickId(10), vd_core::TickId(13), 3));
    assert!(!ttl_alive(vd_core::TickId(10), vd_core::TickId(14), 3));
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

#[test]
fn an_occupied_child_bit_warms_the_sibling_before_the_empty_gate() {
    // SL7's occupied-child proxy (Step 5 slice B): a parent with ZERO local dots still warms an
    // occupied child's NEIGHBOURHOOD — the child observer stands at the placement THIS parent
    // authors, reaching as far as the child's own extent, and folds into `observers` BEFORE the
    // emptiness gate. A live bit on Planet(42) (at the origin, extent 1000) warms the NEARBY
    // sibling Planet(43) at 500 m — and keeps ITSELF demanded, which is what lets the parent's
    // KeepAlive shadow the child's own liveness through a hand-off.
    let near = DVec3::new(500.0, 0.0, 0.0);
    let mut rig = parent_with_two_planet_children(near);
    inject_bit(&mut rig, RealmId::Planet(42));
    let verbs: BTreeMap<RealmId, DemandVerb> = demands(&rig.tick(vec![]))
        .iter()
        .map(|d| (d.child.lowered(), d.verb))
        .collect();
    assert_eq!(
        verbs.get(&RealmId::Planet(43)),
        Some(&DemandVerb::SpinUp),
        "the near sibling warms off the occupied child's bit alone"
    );
    assert_eq!(
        verbs.get(&RealmId::Planet(42)),
        Some(&DemandVerb::SpinUp),
        "the occupied child keeps itself demanded (dist 0, reach its own extent)"
    );
    // ...and the child observer carries its OWN per-child hysteresis latch on the sibling.
    let sib = child_coord_of(OWN_REALM, RealmId::Planet(43));
    assert!(
        rig.world
            .resource::<AoiMembership>()
            .0
            .contains_key(&(ObserverId::Child(RealmId::Planet(42)), sib.path().clone())),
        "the child observer has its own latch"
    );
    // A FAR sibling (10 km against a 1000 m band + 1000 m reach) is NOT warmed.
    let mut far_rig = parent_with_two_planet_children(DVec3::new(10_000.0, 0.0, 0.0));
    inject_bit(&mut far_rig, RealmId::Planet(42));
    let far_verbs: BTreeMap<RealmId, DemandVerb> = demands(&far_rig.tick(vec![]))
        .iter()
        .map(|d| (d.child.lowered(), d.verb))
        .collect();
    assert!(
        !far_verbs.contains_key(&RealmId::Planet(43)),
        "a sibling beyond the band + reach is not warmed: {far_verbs:?}"
    );
}

#[test]
fn a_live_child_bit_keeps_the_parent_non_empty() {
    let sibling = DVec3::new(500.0, 0.0, 0.0);
    // No dots, no bit ⇒ the parent self-reports Empty for its own realm.
    let mut rig = parent_with_two_planet_children(sibling);
    assert!(
        demands(&rig.tick(vec![]))
            .iter()
            .any(|d| d.verb == DemandVerb::Empty),
        "a truly empty parent self-reports Empty"
    );
    // A fresh bit ⇒ NOT Empty (SL7: a live child keeps its parent alive); the sibling is demanded.
    let mut rig = parent_with_two_planet_children(sibling);
    inject_bit(&mut rig, RealmId::Planet(42));
    let verbs: BTreeMap<RealmId, DemandVerb> = demands(&rig.tick(vec![]))
        .iter()
        .map(|d| (d.child.lowered(), d.verb))
        .collect();
    assert!(
        !verbs.values().any(|v| *v == DemandVerb::Empty),
        "a live child bit makes the parent non-empty"
    );
    assert_eq!(verbs.get(&RealmId::Planet(43)), Some(&DemandVerb::SpinUp));
    // An EXPIRED bit stops counting: past the TTL the parent is empty again (and the bit is gone).
    let now = rig.world.resource::<ClockSample>().local_tick.0 + retain_ttl_ticks(&config()) + 5;
    rig.set_local_tick(now);
    assert!(
        demands(&rig.tick(vec![]))
            .iter()
            .any(|d| d.verb == DemandVerb::Empty),
        "an expired bit no longer holds the parent open"
    );
    assert!(
        rig.world.resource::<ChildLiveness>().0.is_empty(),
        "the expired bit was pruned"
    );
}

#[test]
fn a_child_observer_never_lands_in_an_occupants_verdict_but_a_dot_does() {
    // The `Occupants` verdict is folded ONLY over DOT observers — an occupied CHILD runs the
    // same band math (it warms its siblings, SL7) but the opener holds no client for it, so it
    // contributes nothing here. Its own band rides a `Child`-scope window instead.
    let sibling = DVec3::new(10_000.0, 0.0, 0.0);
    let mut rig = parent_with_two_planet_children(sibling);
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    // A live bit in range of itself ⇒ the Occupants verdict stays EMPTY (never a render route).
    inject_bit(&mut rig, RealmId::Planet(42));
    let opened = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    assert!(
        window_memberships(&opened)
            .iter()
            .all(|(_, _, added, _)| added.is_empty()),
        "an occupied-child observer never enters an Occupants verdict"
    );
    // But a real local DOT with Planet(42) in its band gets it added.
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::ZERO); // at the origin ⇒ in range
    let added: Vec<RealmId> = window_memberships(&rig.tick(vec![]))
        .into_iter()
        .flat_map(|(_, _, added, _)| added)
        .collect();
    assert!(
        added.contains(&RealmId::Planet(42)),
        "a dot with Planet(42) in its band gets it added"
    );
}

#[test]
fn a_realm_states_its_own_look_with_no_child_and_no_parent() {
    // THE ROOM, STATED BY THE PARTY THAT OWNS IT. The box a player is standing INSIDE used to
    // be authored by the router out of its own copy of the seed forest. It is stated here
    // instead, by the shard that owns the realm, out of the one geometric fact a realm holds
    // about ITSELF: its own boundary. Nothing about where it sits is involved, so the ground
    // rule is untouched — and since Slice C2 nobody else can state it at all (SL3 reached).
    //
    // The fixture is deliberately BARE: no AoI child anywhere, no parent, nothing arriving.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(&mut rig, vec![root_region(), own_region()]);
    insert_owned_dot(&mut rig, TRIG_SESSION, player(7), DVec3::ZERO);
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    let bodies = window_bodies(&rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]));
    assert_eq!(bodies.len(), 1, "exactly one body: this realm's own look");
    let (to, window, subject, stmt) = &bodies[0];
    assert_eq!(*to, GATEWAY, "routed to the window's opener");
    assert_eq!(*window, WindowId(1));
    assert_eq!(
        *subject, OWN_REALM,
        "a realm states a look only about itself"
    );
    // Compared as a WHOLE constructed value (HR5: a `let…else { panic! }` leaves an
    // uncoverable false arm): the statement IS a self-look carrying this realm's own boundary,
    // verbatim — and no position, because the type has no such field.
    assert_eq!(
        *stmt,
        BodyStmt::SelfLook {
            bag: vd_core::look::look_bag(&own_region().shape)
        },
        "a realm's own body is a SelfLook of its own boundary, never a marker"
    );
    // Send-on-change: an unchanged look is not re-stated next tick.
    assert!(
        window_bodies(&rig.tick(vec![])).is_empty(),
        "the room does not re-state itself every tick"
    );
}

#[test]
fn a_shard_whose_own_realm_is_absent_from_its_forest_states_no_look() {
    // The refusal arm of `own_shape`. A shard planted with a forest its own realm is not in
    // cannot state a boundary for itself — and the honest answer is to say nothing, not to
    // invent one. (`own_frame` degrades the same way, to the ambient root's frame.) With no
    // child carrying a marker either, the window is served no body at all.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(&mut rig, vec![root_region()]); // no `own_region()`
    insert_owned_dot(&mut rig, TRIG_SESSION, player(7), DVec3::ZERO);
    assert_eq!(
        rig.world
            .resource::<RealmRegions>()
            .own_shape(OWN_REALM)
            .map(|s| s.realm),
        None,
        "a realm absent from the planted forest has no outline to state"
    );
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    assert!(
        window_bodies(&rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)])).is_empty(),
        "no look invented for a realm this shard was never told the shape of"
    );
}

#[test]
fn the_membership_verdict_holds_a_passed_realm_across_grace() {
    // No flicker: a realm the dot has PASSED stays in the verdict for the whole grace window
    // (the grace latch holds `next_in` true), so NOTHING ships until the grace lapses — then
    // exactly one removal. The verdict's anti-blink guarantee, the direct twin of the demand
    // grace hold, on the SAME transition (HR3 — no second AoI derivation exists).
    let grace = 3;
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, grace),
        ],
    );
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(500.0, 0.0, 0.0),
    );
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    // Tick 1: acquire ⇒ the realm is ADDED.
    let v = window_memberships(&rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]));
    assert_eq!(v.len(), 1, "one add on acquire");
    assert_eq!(v[0].2, vec![OTHER_REALM], "the child it acquired");
    // Move OUT of the AoI band ⇒ grace begins; the realm is HELD (still `next_in`) ⇒ nothing
    // ships for `grace` ticks.
    move_dot(&mut rig, TRIG_SESSION, DVec3::new(3000.0, 0.0, 0.0));
    for _ in 0..grace {
        assert!(
            window_memberships(&rig.tick(vec![])).is_empty(),
            "a passed realm is held across the grace window — no flicker"
        );
    }
    // Grace lapses ⇒ the realm finally leaves the verdict (exactly one removal).
    let v = window_memberships(&rig.tick(vec![]));
    assert_eq!(v.len(), 1, "one removal once grace lapses");
    assert!(v[0].2.is_empty(), "nothing added on the final release");
    assert_eq!(v[0].3, vec![OTHER_REALM], "the passed realm is removed");
}

#[test]
fn a_child_bit_survives_a_lost_heartbeat_then_expires() {
    // Step 5 slice B — the anti-blink property, transferred from the pose lane to the bit: the
    // TTL bridges a lost `Unreliable` heartbeat, so one missed datagram never blinks a warmed
    // sibling; past the TTL the bit prunes and stops warming.
    let near = DVec3::new(500.0, 0.0, 0.0);
    let mut rig = parent_with_two_planet_children(near);
    inject_bit(&mut rig, RealmId::Planet(42));
    let sib_verb = |sent: &[(NodeId, MsgClass, Vec<u8>)]| -> Option<DemandVerb> {
        demands(sent)
            .iter()
            .find(|d| d.child.lowered() == RealmId::Planet(43))
            .map(|d| d.verb)
    };
    // Tick 1 (local_tick 1): the sibling spins up off the occupied child.
    assert_eq!(sib_verb(&rig.tick(vec![])), Some(DemandVerb::SpinUp));
    // A LOST heartbeat: advance one tick WITHOUT refreshing the bit. It is still alive (age 1 ≤
    // TTL) ⇒ the sibling STAYS demanded (KeepAlive) — no blink.
    rig.set_local_tick(2);
    assert_eq!(
        sib_verb(&rig.tick(vec![])),
        Some(DemandVerb::KeepAlive),
        "one lost heartbeat never blinks the warmed sibling"
    );
    // After the TTL lapses ⇒ the bit is pruned ⇒ it no longer warms the sibling.
    let ttl = retain_ttl_ticks(&config());
    rig.set_local_tick(2 + ttl + 5);
    assert_eq!(
        sib_verb(&rig.tick(vec![])),
        None,
        "after the TTL the pruned bit stops warming the sibling"
    );
    assert!(
        rig.world.resource::<ChildLiveness>().0.is_empty(),
        "the bit is pruned once its TTL lapses"
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

#[test]
fn evaluate_realm_aoi_two_observers_latch_independently_and_union() {
    // VU S0: each observer carries its OWN acquire/grace latch; the child-level demand is their UNION.
    // A hands the child off to B (A leaves as B arrives) WITHOUT a thrash — the child stays KeepAlive
    // across the swap (never a re-SpinUp, never a drop), and both latches are tracked independently.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 2),
        ],
    );
    let a = SessionId(1);
    let b = SessionId(2);
    insert_owned_dot(&mut rig, a, player(1), DVec3::new(500.0, 0.0, 0.0)); // A in range (< spin-up)
    insert_owned_dot(&mut rig, b, player(2), DVec3::new(3000.0, 0.0, 0.0)); // B out (past tear-down)
    // Tick 1: A acquires ⇒ exactly ONE SpinUp (the union); ONLY A holds a latch (B is out of range).
    assert_eq!(
        demands(&rig.tick(vec![]))
            .iter()
            .map(|d| d.verb)
            .collect::<Vec<_>>(),
        vec![DemandVerb::SpinUp]
    );
    assert_eq!(
        rig.world.resource::<AoiMembership>().0.len(),
        1,
        "only the in-range observer holds a latch"
    );
    // A leaves (into grace) as B arrives: the child stays demanded via the union — ONE KeepAlive, no
    // re-SpinUp, no drop — and NOW both observers hold a latch (A grace-holding, B acquired).
    move_dot(&mut rig, a, DVec3::new(3000.0, 0.0, 0.0));
    move_dot(&mut rig, b, DVec3::new(500.0, 0.0, 0.0));
    assert_eq!(
        demands(&rig.tick(vec![]))
            .iter()
            .map(|d| d.verb)
            .collect::<Vec<_>>(),
        vec![DemandVerb::KeepAlive]
    );
    assert_eq!(
        rig.world.resource::<AoiMembership>().0.len(),
        2,
        "both observers hold independent latches (A in grace, B acquired)"
    );
}

#[test]
fn evaluate_realm_aoi_per_observer_hysteresis_is_stricter_than_the_old_global_min() {
    // VU S0 correctness: per-observer latches are STRICTER (more correct) than the old global-min.
    // Observer A acquires then leaves entirely; observer B loiters in the HYSTERESIS band (past spin-up,
    // inside tear-down) but NEVER acquired. Per-observer: once A is gone the child STOPS being demanded
    // (B's release-band distance can't hold a latch it never acquired). The old global-min WOULD have
    // kept it alive (B is within tear-down of the SHARED latch A set) — the flaw this per-observer fixes.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 0), // grace 0 ⇒ A drops the instant it is out
        ],
    );
    let a = SessionId(1);
    let b = SessionId(2);
    insert_owned_dot(&mut rig, a, player(1), DVec3::new(500.0, 0.0, 0.0)); // A acquires (< spin-up 1000)
    insert_owned_dot(&mut rig, b, player(2), DVec3::new(1500.0, 0.0, 0.0)); // B loiters in the band
    // Tick 1: A acquires, B (past spin-up, never acquired) does not ⇒ exactly one SpinUp.
    assert_eq!(
        demands(&rig.tick(vec![]))
            .iter()
            .map(|d| d.verb)
            .collect::<Vec<_>>(),
        vec![DemandVerb::SpinUp]
    );
    // A leaves entirely (past tear-down 2000); B stays loitering in the hysteresis band.
    move_dot(&mut rig, a, DVec3::new(3000.0, 0.0, 0.0));
    assert!(
        demands(&rig.tick(vec![])).is_empty(),
        "B never acquired ⇒ once A is gone the child stops being demanded (global-min would wrongly hold)"
    );
}

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

#[test]
fn assert_realm_aoi_feature_anywhere() {
    // HR4 G-IDENTICAL: the IDENTICAL AoI feature (an occupant reaching a child ⇒ exactly ONE SpinUp
    // keyed on the child's coord) fires byte-identically on a SYSTEM shard (child = Planet) AND a
    // PLANET shard (child = Area). The loop is kind-BLIND — `own_coord.child(level_of(child))`, no
    // match-on-realm-kind — so the two runs differ ONLY in the child realm named.
    let a = drive_aoi_spinup(
        OWN_REALM,
        FrameRef::SystemSpace { system_seed: 7 },
        aoi_child_framed(
            OTHER_REALM,
            Some(OWN_REALM),
            FrameRef::PlanetCentered { planet_seed: 42 },
            3,
        ),
    );
    let b = drive_aoi_spinup(
        RealmId::Planet(42),
        FrameRef::PlanetCentered { planet_seed: 42 },
        aoi_child_framed(
            RealmId::Area(99),
            Some(RealmId::Planet(42)),
            FrameRef::AreaLocal {
                planet_seed: 42,
                area_seed: 99,
            },
            3,
        ),
    );
    // Count the SpinUp feature under test specifically: on the Planet run the (500,0,0) dot ALSO sits
    // in a child it re-homes into, so `redrive_stranded_crossings` now (correctly) emits a KeepAlive to
    // hold that crossing DEST alive — a SEPARATE, universally-correct behaviour (every crossing keeps
    // its dest alive). The keep-alive fires ONLY for a real held latch, so filtering to SpinUp isolates
    // the AoI feature this G-IDENTICAL test asserts.
    let a_spinup: Vec<_> = a.iter().filter(|d| d.verb == DemandVerb::SpinUp).collect();
    let b_spinup: Vec<_> = b.iter().filter(|d| d.verb == DemandVerb::SpinUp).collect();
    assert_eq!(
        a_spinup.len(),
        1,
        "the System shard emits exactly one SpinUp"
    );
    assert_eq!(
        b_spinup.len(),
        1,
        "the Planet shard emits exactly one SpinUp"
    );
    // IDENTICAL structure — same verb, same emitter fence, same tick; only the child KIND differs.
    assert_eq!(a_spinup[0].parent_fence, b_spinup[0].parent_fence);
    assert_eq!(a_spinup[0].universe_tick, b_spinup[0].universe_tick);
    // Each names its OWN child through the coord machinery (System→Planet, Planet→Area).
    assert_eq!(a_spinup[0].child, child_coord_of(OWN_REALM, OTHER_REALM));
    assert_eq!(
        b_spinup[0].child,
        child_coord_of(RealmId::Planet(42), RealmId::Area(99))
    );
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

#[test]
fn aoi_emits_the_parent_headread_when_armed_and_on_cadence() {
    // A PLANET shard (own realm Planet(42)) nested under System(7): its `own_coord` carries the full
    // lineage, so `parent()` is the System. With an ARMED child band + the recheck cadence live, the AoI
    // pass resolves the parent by HeadRead-ing its directory record — its node lands in `ParentRealmNode`,
    // the send target of the UP-lanes (the ChildLive bit + the up-observation rows/outlines; the
    // per-occupant interest up-relay is DELETED, Step 5 slice D).
    let cfg = StubConfig {
        realm: OTHER_REALM,
        held_realms: StubConfig::single_realm(OTHER_REALM),
        frame: frame_of(OTHER_REALM),
        own_coord: child_coord_of(OWN_REALM, OTHER_REALM),
        realm_recheck_interval: 2,
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
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    rig.set_local_tick(4); // 4 % 2 == 0 ⇒ due
    let sent = rig.tick(vec![]);
    let parent_key = DirectoryKey::Realm(StubConfig::root_coord(OWN_REALM).lowered());
    assert!(
        headreads(&sent).contains(&parent_key),
        "the AoI pass HeadReads the PARENT realm on the recheck cadence"
    );
}

#[test]
fn aoi_recheck_cadence_uses_the_armed_recheck_else_a_tick_derived_demand_cadence() {
    // ARMED (self-fence recheck > 0): that IS the AoI cadence (one HeadRead round-trip serves both, HR3).
    let armed = StubConfig {
        realm_recheck_interval: 4,
        tick_dt_s: 0.05,
        ..config()
    };
    assert_eq!(aoi_recheck_cadence(&armed), 4);
    // DISARMED (recheck 0 — the DevTest profile turns the self-fence off): fall back to a tick-DERIVED
    // demand cadence = hz/2 = (1/0.05)/2 = 10 — the up-relay/cascade must still run in demand mode.
    let disarmed = StubConfig {
        realm_recheck_interval: 0,
        tick_dt_s: 0.05,
        ..config()
    };
    assert_eq!(aoi_recheck_cadence(&disarmed), 10);
    // DISARMED at a very slow tick (hz < 2 ⇒ hz/2 rounds to 0) clamps to 1, never a 0-divisor in
    // `due_this_tick`.
    let slow = StubConfig {
        realm_recheck_interval: 0,
        tick_dt_s: 1.0,
        ..config()
    };
    assert_eq!(aoi_recheck_cadence(&slow), 1);
}

#[test]
fn parent_headread_due_resolves_only_when_parented_armed_and_on_cadence() {
    let clock = ClockSample {
        local_tick: vd_core::TickId(4),
        universe_tick: UniverseTick(100),
        epoch: vd_core::EpochId(1),
        synced: true,
    };
    let armed = RealmRegions::new(vec![aoi_child(RealmId::Planet(99), ROOT_REALM, 1000.0, 0)]);
    let inert = RealmRegions::new(vec![region(
        RealmId::Planet(99),
        Some(ROOT_REALM),
        DVec3::ZERO,
        1000.0,
    )]);
    let parented = || StubConfig {
        own_coord: child_coord_of(OWN_REALM, OTHER_REALM),
        realm_recheck_interval: 2,
        ..config()
    };
    // parented + armed + on cadence (4 % 2 == 0) ⇒ resolve the parent.
    assert_eq!(
        parent_headread_due(&parented(), &armed, &clock),
        Some(StubConfig::root_coord(OWN_REALM)),
    );
    // parented + INERT bands ⇒ `aoi_live()` false ⇒ nothing (the walk/static byte-identity guard).
    assert_eq!(parent_headread_due(&parented(), &inert, &clock), None);
    // parented + armed but OFF cadence (4 % 3 != 0) ⇒ nothing.
    let off = StubConfig {
        own_coord: child_coord_of(OWN_REALM, OTHER_REALM),
        realm_recheck_interval: 3,
        ..config()
    };
    assert_eq!(parent_headread_due(&off, &armed, &clock), None);
    // a ROOT shard (its `own_coord` has no parent) ⇒ nothing, ever.
    let root = StubConfig {
        own_coord: StubConfig::root_coord(OWN_REALM),
        realm_recheck_interval: 2,
        ..config()
    };
    assert_eq!(parent_headread_due(&root, &armed, &clock), None);
}

#[test]
fn update_parent_node_caches_the_parent_shard_overwrites_revokes_and_ignores_non_parents() {
    let cfg = StubConfig {
        own_coord: child_coord_of(OWN_REALM, OTHER_REALM),
        ..config()
    };
    let parent = StubConfig::root_coord(OWN_REALM).lowered();
    let rec = |auth: AuthorityRef| vd_wire::seams::directory::OwnerRecord {
        authority: auth,
        fence: Fence(1),
        lease_expires: UniverseTick(1_000),
        in_transfer: None,
    };
    let mut pn = ParentRealmNode::default();
    // A parent Head with a Shard authority ⇒ the node is cached.
    update_parent_node(
        parent,
        Some(&rec(AuthorityRef::Shard(NodeId(77)))),
        &cfg,
        &mut pn,
    );
    assert_eq!(pn.0, Some(NodeId(77)));
    // Parent RE-HOME (a later reply naming a new node) ⇒ overwrite.
    update_parent_node(
        parent,
        Some(&rec(AuthorityRef::Shard(NodeId(88)))),
        &cfg,
        &mut pn,
    );
    assert_eq!(pn.0, Some(NodeId(88)));
    // A reply for a NON-parent realm — here the shard's OWN realm (Planet(42) ≠ the System(7) parent),
    // i.e. its own-realm recheck reply — ⇒ the cache is untouched (the guard's false arm).
    update_parent_node(
        OTHER_REALM,
        Some(&rec(AuthorityRef::Shard(NodeId(99)))),
        &cfg,
        &mut pn,
    );
    assert_eq!(
        pn.0,
        Some(NodeId(88)),
        "a non-parent Head never touches the parent cache"
    );
    // A parent record held by a GATEWAY (not a shard) ⇒ cleared (no shard to relay to).
    update_parent_node(
        parent,
        Some(&rec(AuthorityRef::Gateway(NodeId(5)))),
        &cfg,
        &mut pn,
    );
    assert_eq!(pn.0, None);
    // A parent REVOKE (record gone) ⇒ cleared.
    update_parent_node(
        parent,
        Some(&rec(AuthorityRef::Shard(NodeId(88)))),
        &cfg,
        &mut pn,
    );
    update_parent_node(parent, None, &cfg, &mut pn);
    assert_eq!(pn.0, None);
}

#[test]
fn update_child_node_caches_a_rostered_childs_shard_revokes_and_ignores_non_children() {
    // The downward twin (findings 0/43): a realm-Head reply for a ROSTERED direct child caches its
    // node as the up-lanes' admission answer; a re-home overwrites; a non-Shard or absent record
    // REMOVES (fail closed); a reply for anything not on the roster is a no-op.
    let cfg = config(); // own realm System(7)
    let regions = RealmRegions::new(vec![root_region(), own_region(), child_region()]);
    let rec = |auth: AuthorityRef| vd_wire::seams::directory::OwnerRecord {
        authority: auth,
        fence: Fence(1),
        lease_expires: UniverseTick(1_000),
        in_transfer: None,
    };
    let mut cn = ChildRealmNodes::default();
    // A rostered child's Head with a Shard authority ⇒ cached.
    update_child_node(
        OTHER_REALM,
        Some(&rec(AuthorityRef::Shard(NodeId(61)))),
        &cfg,
        &regions,
        &mut cn,
    );
    assert_eq!(cn.0.get(&OTHER_REALM), Some(&NodeId(61)));
    // A child RE-HOME (a later reply naming a new node) ⇒ overwrite.
    update_child_node(
        OTHER_REALM,
        Some(&rec(AuthorityRef::Shard(NodeId(62)))),
        &cfg,
        &regions,
        &mut cn,
    );
    assert_eq!(cn.0.get(&OTHER_REALM), Some(&NodeId(62)));
    // A NON-child realm (here the shard's own) ⇒ the map is untouched (the roster guard's false arm).
    update_child_node(
        OWN_REALM,
        Some(&rec(AuthorityRef::Shard(NodeId(63)))),
        &cfg,
        &regions,
        &mut cn,
    );
    assert_eq!(cn.0.len(), 1, "a non-child Head never touches the map");
    // A child record held by a GATEWAY (not a shard) ⇒ REMOVED — the up-lanes fail closed.
    update_child_node(
        OTHER_REALM,
        Some(&rec(AuthorityRef::Gateway(NodeId(5)))),
        &cfg,
        &regions,
        &mut cn,
    );
    assert!(cn.0.is_empty());
    // A child REVOKE (record gone) ⇒ removed too.
    update_child_node(
        OTHER_REALM,
        Some(&rec(AuthorityRef::Shard(NodeId(62)))),
        &cfg,
        &regions,
        &mut cn,
    );
    update_child_node(OTHER_REALM, None, &cfg, &regions, &mut cn);
    assert!(cn.0.is_empty());
}

#[test]
fn admission_head_reads_ride_the_aoi_cadence_for_demanded_children() {
    // Findings 0/43, the EAGER pre-resolve: on the same cadence (and behind the same `aoi_live`
    // gate) as the parent head-read, the shard reads the directory head of every child it is
    // currently demanding or holds a live bit for — so the admission answer is in hand before the
    // child's first bit ever arrives (zero added spin-up-to-visible latency). Off-cadence ticks
    // read nothing.
    let mut rig = Rig::with_config(StubConfig {
        realm_recheck_interval: 2,
        ..config()
    });
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 1000.0, 0),
        ],
    );
    // An occupant inside the child's band ⇒ the child is demanded every tick.
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    rig.set_local_tick(4); // on the cadence
    assert!(
        headreads(&rig.tick(vec![])).contains(&DirectoryKey::Realm(OTHER_REALM)),
        "a demanded child's head is read on the cadence beat"
    );
    rig.set_local_tick(5); // off the cadence
    assert!(
        !headreads(&rig.tick(vec![])).contains(&DirectoryKey::Realm(OTHER_REALM)),
        "off-cadence ticks read nothing — the count is bounded by beats, not ticks"
    );
}

#[test]
#[should_panic(expected = "self-fence")]
fn register_stub_shard_rejects_a_parent_that_aliases_its_own_realm_id() {
    use vd_core::realm_path::RealmKindTag;
    // A realm whose PARENT lowers to its own id is unrepresentable, and the boot guard refuses it —
    // otherwise the shard's own parent-Head reply drives the PRIMARY-FOREIGN self-fence branch.
    //
    // ★ RE-BASED IN S9, AND THE WAY IN CHANGED. This used to build `Galaxy(5) → System(1)`, because
    // a galaxy LOWERED to `System(1)` whatever its seed: a perfectly ordinary system, seed 1, hosted
    // under a perfectly ordinary galaxy, collided with its own parent. That was the realistic way to
    // hit this, and S9 closed it — a galaxy keeps its seed now, so `Galaxy(5)` lowers to `Galaxy(5)`
    // and nothing collides.
    //
    // The guard is NOT dead, so it is still driven: with lossless lowering a self-alias needs the
    // same KIND and the same SEED on both levels, which is a nonsense lineage rather than an
    // accident of naming. That is the improvement — the failure went from something a real world
    // could produce to something only a malformed path can.
    let own_coord = RealmCoord::from_path(RealmPath::from_levels(vec![
        RealmLevel::new(RealmKindTag::System, 1),
        RealmLevel::new(RealmKindTag::System, 1),
    ]))
    .expect("a two-level path has a leaf");
    let bad = StubConfig {
        realm: RealmId::System(1),
        held_realms: StubConfig::single_realm(RealmId::System(1)),
        frame: frame_of(RealmId::System(1)),
        own_coord,
        ..config()
    };
    let _ = Rig::with_config(bad);
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

#[test]
fn a_demoted_dot_goes_silent_to_the_parent_on_the_very_tick_it_is_handed_over() {
    // THE SILENCE the hand-off ledger exists to end — MEASURED, not argued. What crosses upward
    // now is the ONE occupancy bit (slice D deleted the pose relay), and the property is the
    // same: a source shard that has applied the ordered Demote still holds the dot (as a
    // retained Ghost), but the observer fold filters on `speaks_for` — so at a ZERO budget the
    // realm counts nobody, the bit stops on the demote tick, and the parent loses the only
    // liveness that was standing in for the traveller mid-crossing. This is the BEFORE reading;
    // the armed budget below keeps the bit beating through the window.
    const PARENT_NODE: NodeId = NodeId(55);
    let mut rig = parented_aoi_rig(2);
    resolve_parent_head(
        &mut rig,
        StubConfig::root_coord(OWN_REALM).lowered(),
        PARENT_NODE,
    );
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    assert_eq!(
        child_live_bits(&rig.tick(vec![])).len(),
        1,
        "while OWNED the realm's bit ships (the occupancy edge here; cadence beats follow)"
    );
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::Demote(DemoteCmd {
            transfer: TransferId(7),
            subject: DirectoryKey::Entity(entity),
            new_owner_fence: Fence(2),
            step_id: DEMOTE_STEP,
        }),
    )]);
    assert_eq!(
        child_live_bits(&sent).len(),
        0,
        "the demote tick is the LAST beat the parent hears — at a zero budget the realm goes silent"
    );
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

#[test]
fn an_armed_shard_keeps_telling_its_parent_where_a_departing_occupant_is() {
    // THE ARMED READING of the silence measured above. Same fixture, same demote, a budget of 8 —
    // and the parent keeps hearing about the traveller for the whole window instead of losing them
    // at the worst possible moment.
    const PARENT_NODE: NodeId = NodeId(55);
    // recheck 1 ⇒ the bit's cadence is every tick (finding 41): this test measures the HOLD
    // budget tick by tick, so it pins the densest beat; the cadence itself has its own gate
    // (`the_bit_beats_on_the_cadence_plus_the_occupancy_edge`).
    let mut rig = parented_aoi_rig_holding(1, 8);
    resolve_parent_head(
        &mut rig,
        StubConfig::root_coord(OWN_REALM).lowered(),
        PARENT_NODE,
    );
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    let owned = child_live_bits(&rig.tick(vec![]));
    assert_eq!(owned.len(), 1, "while OWNED, the bit beats");
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;

    let handed = child_live_bits(&rig.tick(vec![demote_msg(entity, Fence(2))]));
    assert_eq!(
        handed.len(),
        1,
        "the realm keeps counting a subject it has handed away — the bit beats on"
    );
    assert_eq!(handed[0].0, PARENT_NODE);
    // …and it is doing so WITHOUT owning the dot. The hold is what carries the liveness, not
    // ownership.
    assert!(
        !rig.world.resource::<Dots>().0[&SESSION]
            .authority
            .simulates()
    );

    // Past the budget the shard falls silent on its own, so a hand-off that WEDGES rather than
    // completes cannot hold the parent's attention — or, below, its own realm — open forever.
    let opened = rig.world.resource::<HandoffHolds>().0[&(entity, HoldRole::Source)].opened_at;
    rig.set_local_tick(opened.0 + 8);
    assert!(
        child_live_bits(&rig.tick(vec![])).is_empty(),
        "the budget is a real cap, not a formality"
    );
}

#[test]
fn the_take_over_landing_stops_the_relay_before_the_budget_does() {
    // THE NORMAL TERMINAL. The destination's `GhostFlow::Spawn` is its proof that it has taken over —
    // and from that moment IT relays the occupant, so this shard must stop. The budget is only the
    // backstop for a hand-off that never gets here.
    const PARENT_NODE: NodeId = NodeId(55);
    // recheck 1 ⇒ the bit beats every tick (see the hold-budget test above for why).
    let mut rig = parented_aoi_rig_holding(1, 8);
    resolve_parent_head(
        &mut rig,
        StubConfig::root_coord(OWN_REALM).lowered(),
        PARENT_NODE,
    );
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
    let _ = rig.tick(vec![demote_msg(entity, Fence(2))]);

    // A REPLAYED take-over from a superseded crossing (an older fence) proves nothing and leaves the
    // hold — and therefore the relay — standing.
    let sent = rig.tick(vec![ghost_lifecycle(GhostFlow::SpawnV2 {
        entity,
        source_fence: Fence(1),
    })]);
    assert_eq!(
        child_live_bits(&sent).len(),
        1,
        "a stale take-over proof does not end this shard's hand-off — the bit beats on"
    );

    // The matching proof does, well inside the budget.
    let sent = rig.tick(vec![ghost_lifecycle(GhostFlow::SpawnV2 {
        entity,
        source_fence: Fence(2),
    })]);
    assert!(
        child_live_bits(&sent).is_empty(),
        "once the destination has taken over, the source stops — no lingering double beat"
    );
}

#[test]
fn a_realm_does_not_call_itself_empty_while_somebody_is_still_leaving_it() {
    // THE SECOND CONSUMER, and the behaviour-changing half: emptiness is a claim about occupancy, and
    // a subject mid-hand-off has not finished leaving. Without this a realm can report itself empty in
    // the very window its own hand-off is running — the source-side twin of the arrival race.
    let mut rig = parented_aoi_rig_holding(2, 8);
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
    assert!(
        !demands(&rig.tick(vec![]))
            .iter()
            .any(|d| d.verb == DemandVerb::Empty),
        "an occupied realm never reports Empty"
    );

    assert!(
        !demands(&rig.tick(vec![demote_msg(entity, Fence(2))]))
            .iter()
            .any(|d| d.verb == DemandVerb::Empty),
        "nor does it the moment it hands that occupant away"
    );

    // And when the hand-off is over — here by running out of budget, the wedged case — the realm goes
    // back to telling the truth, so an abandoned crossing cannot keep a realm alive indefinitely.
    let opened = rig.world.resource::<HandoffHolds>().0[&(entity, HoldRole::Source)].opened_at;
    rig.set_local_tick(opened.0 + 8);
    assert!(
        demands(&rig.tick(vec![]))
            .iter()
            .any(|d| d.verb == DemandVerb::Empty),
        "past the budget the realm reports the truth again"
    );
}

#[test]
fn aoi_up_relays_nothing_from_a_root_shard() {
    // A ROOT shard (galaxy free-fly: `own_coord` has no parent) is the TOP of the chain — it relays
    // interest to no one. Armed + occupied, but `own_coord.parent()` is None ⇒ the relay never fires.
    let mut rig = Rig::with_config(StubConfig {
        realm: OWN_REALM,
        held_realms: StubConfig::single_realm(OWN_REALM),
        frame: frame_of(OWN_REALM),
        own_coord: StubConfig::root_coord(OWN_REALM),
        realm_recheck_interval: 2,
        ..config()
    });
    grant_realm_for(&mut rig, OWN_REALM);
    plant_aoi(
        &mut rig,
        vec![
            region(ROOT_REALM, None, DVec3::ZERO, 1.0e9),
            aoi_child(OTHER_REALM, ROOT_REALM, 1000.0, 0),
        ],
    );
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    rig.set_local_tick(4);
    assert!(
        child_live_bits(&rig.tick(vec![])).is_empty(),
        "a root shard has nobody above it to report liveness to"
    );
}

#[test]
fn aoi_resolves_the_parent_but_up_relays_nothing_until_the_node_is_known() {
    // Parented + armed + on cadence, but the parent Head has NOT come back yet: the resolve HeadRead
    // fires (so the node WILL arrive), but NO ChildLive bit until it does (the inner-unresolved arm).
    let mut rig = parented_aoi_rig(2);
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    rig.set_local_tick(4);
    let sent = rig.tick(vec![]);
    let parent_key = DirectoryKey::Realm(StubConfig::root_coord(OWN_REALM).lowered());
    assert!(
        headreads(&sent).contains(&parent_key),
        "the parent HeadRead fires (resolving the node)"
    );
    assert!(
        child_live_bits(&sent).is_empty(),
        "no bit until the parent node is resolved — the next tick catches up"
    );
}

#[test]
fn the_bit_beats_on_the_cadence_plus_the_occupancy_edge() {
    // Lane cure, finding 41 — the bit's contract-stated rate, measured: on the AoI cadence, plus
    // immediately when the realm becomes occupied (the adopt edge, derived from the occupancy
    // transition itself — no hook in any adopt path), and re-armed by going empty.
    const PARENT_NODE: NodeId = NodeId(55);
    let mut rig = parented_aoi_rig(4); // cadence 4 — beats land on multiples of 4
    resolve_parent_head(
        &mut rig,
        StubConfig::root_coord(OWN_REALM).lowered(),
        PARENT_NODE,
    );
    // OCCUPANCY EDGE: the first occupied tick ships the bit at once, OFF the cadence.
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    rig.set_local_tick(5);
    assert_eq!(
        child_live_bits(&rig.tick(vec![])).len(),
        1,
        "the occupancy transition ships immediately, off-cadence"
    );
    // Sustained occupancy off the cadence: silent — the parent has been told, the TTL holds it.
    rig.set_local_tick(6);
    assert!(
        child_live_bits(&rig.tick(vec![])).is_empty(),
        "off-cadence, already told ⇒ no beat"
    );
    // The cadence beat re-asserts the level.
    rig.set_local_tick(8);
    assert_eq!(
        child_live_bits(&rig.tick(vec![])).len(),
        1,
        "the cadence beat ships"
    );
    // GOING EMPTY re-arms the edge: the next occupant's first tick ships at once again.
    rig.world.resource_mut::<Dots>().0.clear();
    rig.set_local_tick(9);
    assert!(
        child_live_bits(&rig.tick(vec![])).is_empty(),
        "an empty realm ships no bit (it self-reports Empty instead)"
    );
    insert_owned_dot(&mut rig, SESSION, player(8), DVec3::new(500.0, 0.0, 0.0));
    rig.set_local_tick(10);
    assert_eq!(
        child_live_bits(&rig.tick(vec![])).len(),
        1,
        "re-occupied ⇒ the edge fires again, off-cadence"
    );
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

#[test]
fn an_occupants_window_ships_rows_bodies_and_membership_send_on_change() {
    // §2.9 steps 1/3/4/5 on an Occupants window: ONE WindowFrame per tick (hop: None, the
    // authored roster verbatim, stamped at the one universe tick), the self-look + the armed
    // child's marker ON OPEN, the SL7 verdict as one added-batch — then a SECOND tick ships
    // the per-tick frame again and NOTHING else (send-on-change holds its tongue).
    let mut rig = window_rig();
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    let frames = window_frames(&sent);
    assert_eq!(frames.len(), 1, "ONE frame per tick per open window");
    let (to, window, at, hop, rows) = &frames[0];
    assert_eq!(*to, GATEWAY);
    assert_eq!(*window, WindowId(1));
    assert_eq!(*at, UniverseTick(100), "stamped at the one universe tick");
    assert_eq!(*hop, None, "an occupant already stands in this frame");
    // The FULL direct-child roster, static and armed alike, in the sender's own frame.
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].realm, OTHER_REALM);
    assert_eq!(rows[0].frame, frame_of(OTHER_REALM));
    assert_eq!(rows[0].pose.frame, frame_of(OWN_REALM));
    assert_eq!(fm(rows[0].pose.pos), WINDOW_CHILD_CENTER);
    assert_eq!(rows[1].realm, RealmId::Planet(43));
    // The bodies: the realm's OWN look (its boot extent, SL3) + one point-of-light marker
    // per direct child, GLOWING OR NOT (look_horizon.md slice 1's presence floor): the armed
    // child's datum + extent, the quiet child's extent ALONE (the one-radius law).
    let bodies = window_bodies(&sent);
    assert_eq!(bodies.len(), 3);
    assert_eq!(bodies[0].2, OWN_REALM);
    assert_eq!(
        bodies[0].3,
        BodyStmt::SelfLook {
            bag: vd_core::look::look_bag(&Boundary::Shell { r: 100_000.0 })
        },
        "the look IS the realm's own boot extent, framed by the one shared codec"
    );
    assert_eq!(
        bodies[1],
        (
            GATEWAY,
            WindowId(1),
            OTHER_REALM,
            BodyStmt::Marker {
                luma: vd_core::look::marker_bag(Some((6, 0.25)), 100.0)
            }
        )
    );
    assert_eq!(
        bodies[2],
        (
            GATEWAY,
            WindowId(1),
            RealmId::Planet(43),
            BodyStmt::Marker {
                luma: vd_core::look::marker_bag(None, 100.0)
            }
        ),
        "the non-glowing child still states a correctly-sized point of light"
    );
    // The verdict: the armed child is in the dot's band; the quiet child (inert band) is not.
    assert_eq!(
        window_memberships(&sent),
        vec![(GATEWAY, WindowId(1), vec![OTHER_REALM], vec![])]
    );
    // Second tick: the per-tick frame repeats; bodies + membership are send-on-change quiet.
    rig.set_local_tick(2);
    let sent = rig.tick(vec![]);
    assert_eq!(window_frames(&sent).len(), 1);
    assert_eq!(window_bodies(&sent), vec![]);
    assert_eq!(window_memberships(&sent), vec![]);
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(stats.window_frames_sent, 2);
    assert_eq!(stats.window_frame_rows_sent, 4);
    assert_eq!(stats.window_bodies_sent, 3);
    assert_eq!(stats.window_memberships_sent, 1);
    assert_eq!(stats.windows_open, 1);
}

/// THE PRESENCE FLOOR's producer half (look_horizon.md slice 1, Q2 APPROVED 2026-08-17 — the
/// STATION-LAPSE gate's root cause, DECLARED RED BEFORE THE SLICE): EVERY direct child states
/// a point-of-light marker — a glowing child's carries its photometric datum plus its
/// circumscribed extent, a non-glowing child's the extent alone. Before the floor, a
/// non-glowing child (a station, a city, a ship) had NO lawful bag content at all, so the
/// moment its own picture lapsed it VANISHED instead of degrading to a correctly-sized point.
#[test]
fn every_direct_child_states_a_point_of_light_marker_glowing_or_not() {
    let mut rig = window_rig();
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    let bodies = window_bodies(&sent);
    assert_eq!(
        bodies.len(),
        3,
        "the realm's own look + a marker for BOTH direct children (the quiet, non-glowing \
         child included — the presence floor): {bodies:?}",
    );
    let marker_of = |realm: RealmId| {
        bodies
            .iter()
            .find(|(_, _, subject, _)| *subject == realm)
            .map(|(_, _, _, stmt)| stmt.clone())
            .unwrap_or_else(|| panic!("{realm:?} states no marker — the presence floor hole"))
    };
    // The armed (glowing) child: its photometric datum AND its circumscribed extent, in the
    // ONE marker bag.
    assert_eq!(
        marker_of(OTHER_REALM),
        BodyStmt::Marker {
            luma: vd_core::look::marker_bag(Some((6, 0.25)), 100.0)
        },
    );
    // The quiet (non-glowing) child: the extent ALONE — one radius, nothing else (the
    // one-radius law: a bound is a promise about space; a look is a statement about
    // appearance).
    assert_eq!(
        marker_of(RealmId::Planet(43)),
        BodyStmt::Marker {
            luma: vd_core::look::marker_bag(None, 100.0)
        },
    );
}

#[test]
fn the_keep_alive_re_assert_re_serves_the_whole_set_so_silence_means_a_dead_realm() {
    // WINDOW LANE SLICE D — the beat the gateway's roster-loss window is derived from. A
    // send-on-change lane that only ever speaks on CHANGE cannot distinguish "nothing changed"
    // from "the realm behind this statement stopped speaking", so a keep-alive `WindowOpen`
    // (which the subscriber already sends on its own derived cadence) clears the baselines and
    // re-serves the whole body set and the whole membership verdict. Without this, a departed
    // system's last look would sit in the composer forever and never shrink back to a dot.
    let mut rig = window_rig();
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    let first_bodies = window_bodies(&sent);
    let first_membership = window_memberships(&sent);
    assert_eq!(first_bodies.len(), 3, "the full set on open");
    assert_eq!(first_membership.len(), 1);
    // A quiet tick is still quiet — the re-assert is the BEAT, not every tick.
    rig.set_local_tick(2);
    assert_eq!(window_bodies(&rig.tick(vec![])), vec![]);
    // The keep-alive: the SAME window, the SAME scope. Byte-for-byte the same statements come
    // back out (the realm is still stating exactly what it stated), counted as a re-assert.
    rig.set_local_tick(3);
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    assert_eq!(window_bodies(&sent), first_bodies);
    assert_eq!(window_memberships(&sent), first_membership);
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(stats.window_reasserted, 1);
    assert_eq!(stats.windows_opened, 1, "a keep-alive is not a new window");
    assert_eq!(stats.window_bodies_sent, 6, "three served twice");
    assert_eq!(stats.window_memberships_sent, 2);
}

#[test]
fn a_child_window_ships_the_hop_row_pre_inverted_at_the_author() {
    // §2.2 R1: the hop row is the author's own frame expressed in the child's frame, the ONE
    // inversion made at the author. Identity orientation + zero spin (every placement of THE
    // world today) ⇒ the inversion is exactly the negated placement — asserted EXACTLY.
    let mut rig = window_rig();
    let open = GatewayToShard::WindowOpen {
        window: WindowId(2),
        scope: WindowScope::Child(OTHER_REALM),
    };
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    let frames = window_frames(&sent);
    assert_eq!(frames.len(), 1);
    let (_, window, _, hop, rows) = &frames[0];
    assert_eq!(*window, WindowId(2));
    assert_eq!(rows.len(), 2, "the same authored roster rides every scope");
    let hop = hop.as_ref().expect("a Child window carries the hop row");
    assert_eq!(hop.child, OTHER_REALM);
    // The inverted row is NORMALIZED (transfer_frame's output invariant since the cell
    // activation): the negated placement's value rides the integer anchor.
    assert_eq!(fm(hop.inv.anchor()), -WINDOW_CHILD_CENTER);
    assert_eq!(hop.inv.origin, DVec3::ZERO, "sub-cell residual only");
    assert_eq!(hop.inv.velocity, DVec3::ZERO);
    assert_eq!(hop.inv.orientation, DQuat::IDENTITY);
    assert_eq!(hop.inv.angular_velocity, DVec3::ZERO);
}

#[test]
fn the_hop_inversion_carries_velocity_and_spin_through_the_frame_core() {
    // A MOVING, SPINNING child (identity orientation, cell zero — invertible today): the hop
    // row's velocity is the frame core's own answer (the parent origin as seen from the
    // rotating child: q⁻¹(ω×o − v)) and the angular term is the child's spin reversed and
    // re-axed (−(q⁻¹·ω)) — exact numbers, no epsilon.
    let mut rig = Rig::new();
    rig.grant_realm();
    let mover = RealmId::Planet(45);
    let spin = RealmRegion {
        aoi: aoi_band(0),
        ..region(mover, Some(OWN_REALM), DVec3::ZERO, 100.0)
    };
    let placed = FramePlacement {
        origin_cell: vd_core::glam::I64Vec3::ZERO,
        origin: DVec3::new(7.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 1.0, 0.0),
        orientation: DQuat::IDENTITY,
        angular_velocity: DVec3::new(0.0, 0.0, 0.5),
    };
    let motions: BTreeMap<RealmId, MotionFn> =
        [(mover, MotionFn(std::sync::Arc::new(move |_| placed)))]
            .into_iter()
            .collect();
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), own_region(), spin]).with_moving_children(motions);
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    let open = GatewayToShard::WindowOpen {
        window: WindowId(3),
        scope: WindowScope::Child(mover),
    };
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    let frames = window_frames(&sent);
    assert_eq!(frames.len(), 1);
    let hop = frames[0].3.as_ref().expect("hop row");
    assert_eq!(fm(hop.inv.anchor()), DVec3::new(-7.0, 0.0, 0.0));
    // ω×o − v = (0,0,0.5)×(7,0,0) − (0,1,0) = (0,3.5,0) − (0,1,0) = (0,2.5,0).
    assert_eq!(hop.inv.velocity, DVec3::new(0.0, 2.5, 0.0));
    assert_eq!(hop.inv.angular_velocity, DVec3::new(0.0, 0.0, -0.5));
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
    )
}

#[test]
fn assert_window_emission_feature_anywhere() {
    // HR4 G-IDENTICAL: the IDENTICAL window-emission feature — one frame per tick per window,
    // the pre-inverted hop at the author, the look/marker bodies, the SL7 verdict — on a
    // SYSTEM shard (child = Planet) AND a PLANET shard (child = Area), under their REAL
    // capability profiles. The emitter is kind-BLIND, so the two runs differ ONLY in the
    // realm ids they name: every VALUE (poses, hop numbers, bags, stamps) is byte-equal.
    let a = drive_window_emit(
        NodeKind::Shard(crate::capability::profiles::system().expect("system profile")),
        OWN_REALM,
        FrameRef::SystemSpace { system_seed: 7 },
        RealmRegion {
            aoi: aoi_band(3),
            ..region_framed(
                OTHER_REALM,
                Some(OWN_REALM),
                WINDOW_CHILD_CENTER,
                100.0,
                FrameRef::PlanetCentered { planet_seed: 42 },
            )
        },
    );
    let b = drive_window_emit(
        NodeKind::Shard(crate::capability::profiles::planet().expect("planet profile")),
        RealmId::Planet(42),
        FrameRef::PlanetCentered { planet_seed: 42 },
        RealmRegion {
            aoi: aoi_band(3),
            ..region_framed(
                RealmId::Area(99),
                Some(RealmId::Planet(42)),
                WINDOW_CHILD_CENTER,
                100.0,
                FrameRef::AreaLocal {
                    planet_seed: 42,
                    area_seed: 99,
                },
            )
        },
    );
    let (a_frames, a_bodies, a_members) = a;
    let (b_frames, b_bodies, b_members) = b;
    assert_eq!(a_frames.len(), 2, "one frame per open window (System run)");
    assert_eq!(b_frames.len(), 2, "one frame per open window (Planet run)");
    for ((_, aw, aat, ahop, arows), (_, bw, bat, bhop, brows)) in
        a_frames.iter().zip(b_frames.iter())
    {
        assert_eq!(aw, bw, "same window order");
        assert_eq!(aat, bat, "same universe stamp");
        assert_eq!(arows.len(), 1);
        assert_eq!(brows.len(), 1);
        // The VALUES are identical — only the realm names differ (kind-blind emission).
        assert_eq!(arows[0].pose.pos, brows[0].pose.pos);
        assert_eq!(arows[0].pose.vel, brows[0].pose.vel);
        assert_eq!(arows[0].realm, OTHER_REALM);
        assert_eq!(brows[0].realm, RealmId::Area(99));
        // Option equality on the INV alone (HR5: plain equality, no match with an
        // uncoverable divergence arm): both absent on the Occupants frame, both the SAME
        // pre-inverted numbers on the Child frame — the shape and the values in one compare.
        assert_eq!(
            ahop.as_ref().map(|h| h.inv),
            bhop.as_ref().map(|h| h.inv),
            "the hop inversion is shape- and value-identical across kinds"
        );
        if let Some(h) = ahop {
            assert_eq!(h.child, OTHER_REALM);
        }
        if let Some(h) = bhop {
            assert_eq!(h.child, RealmId::Area(99));
        }
    }
    // The bodies, as WHOLE expected sets (HR5: plain equality, never matches!): the look and
    // marker BAGS are the same byte strings in both runs — the same boot extent through the
    // one codec, the same planted datum — and only the subject ids differ per run.
    let look = vd_core::look::look_bag(&Boundary::Shell { r: 100_000.0 });
    // The presence floor (look_horizon.md slice 1): the planted datum + the child's
    // circumscribed extent, one bag through the one codec, identical bytes on every profile.
    let luma = vd_core::look::marker_bag(Some((6, 0.25)), 100.0);
    let expected = |own: RealmId, child: RealmId| {
        vec![
            (
                GATEWAY,
                WindowId(1),
                own,
                BodyStmt::SelfLook { bag: look.clone() },
            ),
            (
                GATEWAY,
                WindowId(1),
                child,
                BodyStmt::Marker { luma: luma.clone() },
            ),
            (
                GATEWAY,
                WindowId(2),
                own,
                BodyStmt::SelfLook { bag: look.clone() },
            ),
            (
                GATEWAY,
                WindowId(2),
                child,
                BodyStmt::Marker { luma: luma.clone() },
            ),
        ]
    };
    assert_eq!(a_bodies, expected(OWN_REALM, OTHER_REALM));
    assert_eq!(b_bodies, expected(RealmId::Planet(42), RealmId::Area(99)));
    // The verdicts: the Occupants window ships the dot's in-band child; the Child window's
    // occupied-child proxy verdict is EMPTY here (no live bit) ⇒ exactly ONE message per run.
    assert_eq!(a_members.len(), 1);
    assert_eq!(b_members.len(), 1);
    assert_eq!(a_members[0].1, WindowId(1));
    assert_eq!(b_members[0].1, WindowId(1));
    assert_eq!(a_members[0].2, vec![OTHER_REALM]);
    assert_eq!(b_members[0].2, vec![RealmId::Area(99)]);
}

#[test]
fn a_dead_gateways_windows_die_by_the_derived_ttl_with_zero_leaked_emissions() {
    // THE SLICE-A CHAOS GATE (window_lane.md §4 Slice A "TTL-expiry chaos"), in-proc form of
    // the process-tier gateway kill (`rlm_demand_login`'s `kill_and_reap`): a gateway opens
    // windows, then DIES — its keep-alives stop. The shard's windows must expire on the
    // DERIVED TTL (2 beats + 1, owner law 3(a)) and not one emission may leak past expiry.
    let mut rig = window_rig();
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
                scope: WindowScope::Child(OTHER_REALM),
            },
        ),
    ]);
    assert_eq!(window_frames(&sent).len(), 2, "both windows serve");
    let ttl = window_ttl_ticks(&config());
    assert_eq!(
        ttl,
        RETAIN_TTL_CADENCE_BEATS * aoi_recheck_cadence(&config()) + 1,
        "the TTL is the derived 2-beats+1, never a literal"
    );
    // The LAST tick inside the window: a whole TTL of silence is still served (one lost
    // keep-alive never blinks a live subscriber).
    rig.set_local_tick(1 + ttl);
    let sent = rig.tick(vec![]);
    assert_eq!(window_frames(&sent).len(), 2, "alive at the TTL edge");
    assert_eq!(rig.world.resource::<StubStats>().windows_open, 2);
    // One past: BOTH windows die BEFORE anything ships — zero leaked emissions after expiry.
    rig.set_local_tick(2 + ttl);
    let sent = rig.tick(vec![]);
    assert_eq!(window_frames(&sent), vec![], "no frame leaks past the TTL");
    assert_eq!(window_bodies(&sent), vec![], "no body leaks past the TTL");
    assert_eq!(
        window_memberships(&sent),
        vec![],
        "no verdict leaks past the TTL"
    );
    assert_eq!(rig.world.resource::<OpenWindows>().0.len(), 0);
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(stats.window_ttl_expired, 2, "each expiry counted");
    assert_eq!(stats.windows_open, 0, "the gauge reads the empty registry");
    // And a keep-alive RESETS the clock: a re-opened window re-asserted at the edge survives
    // the next whole TTL from THAT refresh.
    let open = GatewayToShard::WindowOpen {
        window: WindowId(9),
        scope: WindowScope::Occupants,
    };
    rig.set_local_tick(100);
    let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    rig.set_local_tick(100 + ttl);
    let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]); // the re-assert
    rig.set_local_tick(100 + ttl + ttl);
    let sent = rig.tick(vec![]);
    assert_eq!(
        window_frames(&sent).len(),
        1,
        "the refreshed window outlives the original TTL horizon"
    );
}

#[test]
fn inv_body_at_origin_and_the_rotated_hop_inertness_are_pinned_on_the_world() {
    // TWO NAMED PINS ON THE WORLD (window_lane.md §2.12; SL5 — the one seed universe, at the
    // shipped posture the boot fences enforce):
    // 1. INV-BODY-AT-ORIGIN — a realm's own body sits at its own frame origin, so the origin
    //    of ITS frame, re-expressed through its parent's authored book, IS the parent's
    //    placement row, exactly; and the pre-inverted hop row round-trips back to the origin
    //    exactly.
    // 2. ROTATED-CROSS-CELL INERTNESS — the hop inversion cannot be refused on THE world
    //    today (a MEASUREMENT that fails the day the world grows a spinning realm a
    //    cell-block out before the P10 cell math lands — never an argument).
    let cfg = vd_physics::worldgen::UniverseConfig::world(
        vd_physics::worldgen::VISUAL_OCCUPANT_V_MAX_MPS,
        vd_physics::worldgen::AOI_TICK_DT_S,
    );
    let forest = vd_physics::worldgen::realm_regions_for_config(0, &cfg);
    let anchors: BTreeSet<RealmId> = forest.iter().filter_map(|r| r.parent).collect();
    let mut movers: BTreeMap<RealmId, vd_physics::celestial::OrbitalElements> = BTreeMap::new();
    for &anchor in &anchors {
        movers.extend(vd_physics::worldgen::moving_children_for_config(
            0, &cfg, anchor,
        ));
    }
    let regions = RealmRegions::new(forest).with_moving_children(kepler_motion_fns(movers));
    let tick_hz = 1.0 / vd_physics::worldgen::AOI_TICK_DT_S;
    let mut rows_pinned = 0usize;
    let mut moved_since_epoch = 0usize;
    let mut cross_rung_hops = 0usize;
    for &t in &[UniverseTick(0), UniverseTick(50_000)] {
        for &anchor in &anchors {
            let own = regions.own_frame(anchor);
            let book = regions.author_book(anchor, tick_hz, t);
            for (region, pose) in regions.child_rows(anchor, &book) {
                rows_pinned += 1;
                // (1a) The child's own body — the ORIGIN of its own frame — maps through the
                // parent's book onto EXACTLY the authored placement row.
                let body = transfer_frame(
                    &StampedPose::at_rest(region.frame, DVec3::ZERO, t),
                    own,
                    &book,
                )
                .expect("every direct child of THE world resolves through its parent's book");
                assert_eq!(
                    body.pos, pose.pos,
                    "{anchor} -> {}: body at origin",
                    region.realm
                );
                assert_eq!(body.vel, pose.vel);
                // (2) THE INVERSION, AND THE LIMIT IT HAS NOW REACHED (slice S9).
                //
                // ★ THIS PIN WAS BUILT TO FAIL ONE DAY, AND THAT DAY IS TODAY — which is the test
                // working, not breaking. It asserted the hop inversion is NEVER refused on THE world.
                // The inversion states the PARENT's body in the CHILD's frame, so it must count the
                // parent's origin in the child's unit. A star system counts in millimetres and its
                // galaxy's centre is now 0.7 LIGHT YEARS away, against a millimetre lattice that
                // reaches a quarter of one. There is no such count, and there never will be: this is
                // not a bug to fix but a fact about units, and the honest thing is to state it.
                //
                // So the pin splits by whether the two frames count in the SAME unit, and both arms
                // are asserted — the working one exactly, the refused one by name and with its
                // numbers, so it stays a measurement rather than becoming a silence.
                let inv = match invert_hop_placement(own, region.frame, &book) {
                    Ok(inv) => inv,
                    Err(refused) => {
                        // ★ THE LIMIT, NAMED. It is about REACH and not about the units differing:
                        // a galaxy sitting at the universe's origin inverts perfectly well across two
                        // rungs, because zero has a count in every unit. What has no count is a
                        // DISTANCE too large for the finer one. Asserted, so a rotation failure or an
                        // unknown frame arriving here would not be quietly absorbed as "expected".
                        assert!(
                            matches!(
                                refused,
                                vd_core::frame::FrameError::CrossTierCrossing(
                                    vd_core::pose::TierConversionError::BeyondReach { .. }
                                )
                            ),
                            "{anchor} -> {}: the only lawful refusal here is reach: {refused:?}",
                            region.realm
                        );
                        // …and it is only ever the coarse-to-fine direction that can run out.
                        assert!(
                            own.tier().step_exponent() > region.frame.tier().step_exponent(),
                            "a hop into a COARSER unit can always be counted"
                        );
                        cross_rung_hops += 1;
                        continue;
                    }
                };
                // (1b) ...and it round-trips: the parent's body, as the hop states it in the
                // child's frame, maps back to the parent's own origin EXACTLY (identity
                // orientations everywhere today ⇒ bit-exact, no epsilon).
                let back = transfer_frame(
                    &StampedPose {
                        frame: region.frame,
                        pos: LatticePos::at(inv.origin_cell, inv.origin),
                        vel: inv.velocity,
                        orient: inv.orientation,
                        universe_tick: t,
                    },
                    own,
                    &book,
                )
                .expect("the inverse rides the same book");
                assert_eq!(back.pos.cell(), vd_core::glam::I64Vec3::ZERO);
                assert_eq!(fm_at(back.pos, own.tier()), DVec3::ZERO);
                assert_eq!(back.vel, DVec3::ZERO);
                // ★ THE SIXTEENTH SITE, AND THE MOST INSTRUCTIVE (slice S9). Both sides of this
                // comparison were read at the CHILD's step, when both quantities are stated in the
                // ANCHOR's: the authored pose comes out of the anchor's own book, and the centre is
                // a position in the anchor's frame. Two wrongs of the same size cancel, so the
                // comparison gave the right answer for the wrong reason and nothing ever failed.
                //
                // Both now read `own`, the anchor's frame, which is the one unit they share.
                if t == UniverseTick(50_000)
                    && pose
                        .pos
                        .delta_m(vd_core::pose::LatticePos::ORIGIN, own.tier())
                        != region
                            .center
                            .in_parents_frame()
                            .delta_m(vd_core::pose::LatticePos::ORIGIN, own.tier())
                {
                    moved_since_epoch += 1;
                }
            }
        }
    }
    // NON-VACUITY, pinned: THE world's parent-authored rows — the galaxy under the universe,
    // three systems under the galaxy, five planets under each system — at BOTH instants.
    assert_eq!(
        rows_pinned,
        2 * (1 + 3 + 27 + 3 + 6),
        "THE world's full child-row set (galaxy + systems + planets + the T2 stars + the \
         T3 census moons)"
    );
    // ...and the movers actually MOVED between the two instants (the pin measured a live
    // world, not a static fixture): every planet is off its zeroed region center at t=50000.
    assert_eq!(
        moved_since_epoch, 33,
        "all twenty-seven planets and six census moons author live placements"
    );
    // ★ AND THE REFUSAL ARM IS NOT VACUOUS (slice S9). Pinned as a count, because "some hops are out
    // of reach" is worthless without knowing how many: if this silently went to zero the arm above
    // would stop being exercised and the pin would quietly narrow to the easy cases.
    //
    // FOUR: the two RING star systems, at both instants. Not six — the HOME system is anchored at
    // the galaxy's own origin, and zero has a count in every unit, so its hop inverts perfectly well
    // across the two rungs. The ring siblings sit roughly 0.7 light years out, and a system counts in
    // millimetres, which reach a quarter of a light year.
    //
    // That the home system is the exception is worth more than the number: it means the galaxy's
    // ORIGIN is the only place in it a millimetre-counting realm can name its parent from.
    //
    // Every other row in THE world — planets, stars, moons under their own parents, and the galaxy
    // under the universe — either shares its parent's unit or sits close enough to count.
    assert_eq!(
        cross_rung_hops, 4,
        "the two RING systems, at both instants, cannot count their galaxy's centre in millimetres"
    );
}

#[test]
fn the_live_siblings_interior_is_one_level_out_and_never_deeper_q1() {
    // THE Q1 FENCE PIN (owner ruling 2026-08-16, window_lane.md §5 RULINGS + §2.12): one
    // level into ANY live realm you are next to — generic, never a planet-specific case —
    // and NEVER deeper. Structurally: a parent's window statements name AT MOST its direct
    // children (its held interior outlines for a live child do NOT enter the window lane —
    // the parent's per-child message stays a placement and nothing else, SL3); the ONE level
    // of interior comes from the live realm's OWN window lane. Chained, an observer's
    // windows reach exactly two levels — inside the two-level visibility guarantee the boot
    // fence (`guard_visibility_climb_bounded`, look_horizon slice 2 — the MEASURED climb
    // against the look carrier's arity) proves on THE world.
    let near = DVec3::new(500.0, 0.0, 0.0);
    let mut rig = parent_with_two_planet_children(near);
    inject_bit(&mut rig, RealmId::Planet(42));
    // (Since Slice C2 there is no store a parent could hold a child's interior in AT ALL —
    // the up-shape lane and its holding died together, so the leak this pin guards against is
    // now unrepresentable as well as untaken. HALF 1 below still measures it.)
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
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
                scope: WindowScope::Child(RealmId::Planet(42)),
            },
        ),
    ]);
    // HALF 1 — the PARENT's lane: every realm the window statements name is a DIRECT child
    // (or the parent itself, as a body subject). The held grandchild outline is NOWHERE.
    let direct: BTreeSet<RealmId> = [RealmId::Planet(42), RealmId::Planet(43)]
        .into_iter()
        .collect();
    let frames = window_frames(&sent);
    assert_eq!(frames.len(), 2, "both windows serve (non-vacuous)");
    for (_, _, _, _, rows) in &frames {
        assert!(
            !rows.is_empty(),
            "the roster rides every frame (non-vacuous)"
        );
        for row in rows {
            assert!(
                direct.contains(&row.realm),
                "a window frame row names a non-direct-child: {}",
                row.realm
            );
        }
    }
    for (_, _, subject, _) in &window_bodies(&sent) {
        // Bitwise `|` (both operands pure, HR5 — a short-circuit RHS is an uncoverable region).
        assert!(
            (*subject == OWN_REALM) | direct.contains(subject),
            "a body subject beyond one level: {subject}"
        );
    }
    for (_, _, added, removed) in &window_memberships(&sent) {
        for id in added.iter().chain(removed.iter()) {
            assert!(direct.contains(id), "a verdict id beyond one level: {id}");
        }
    }
    // HALF 2 — the LIVE realm's OWN lane states the one level of interior (Q1 = YES), on a
    // DIFFERENT realm kind (a Planet shard with an Area child — the G-IDENTICAL discipline):
    // its window frame names ITS direct children and nothing deeper exists to leak.
    let planet = RealmId::Planet(42);
    let cfg = StubConfig {
        realm: planet,
        held_realms: StubConfig::single_realm(planet),
        frame: frame_of(planet),
        own_coord: StubConfig::root_coord(planet),
        ..config()
    };
    let mut inner = Rig::with_config(cfg);
    grant_realm_for(&mut inner, planet);
    plant_aoi(
        &mut inner,
        vec![
            region_framed(planet, None, DVec3::ZERO, 100_000.0, frame_of(planet)),
            region_framed(
                RealmId::Area(99),
                Some(planet),
                DVec3::new(1.0, 2.0, 3.0),
                5.0,
                FrameRef::AreaLocal {
                    planet_seed: 42,
                    area_seed: 99,
                },
            ),
        ],
    );
    insert_owned_dot_framed(
        &mut inner,
        TRIG_SESSION,
        player(8),
        frame_of(planet),
        DVec3::new(50.0, 0.0, 0.0),
    );
    let sent = inner.tick(vec![wire_msg(
        GATEWAY,
        MsgClass::Control,
        &GatewayToShard::WindowOpen {
            window: WindowId(1),
            scope: WindowScope::Occupants,
        },
    )]);
    let frames = window_frames(&sent);
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0].4.iter().map(|r| r.realm).collect::<Vec<_>>(),
        vec![RealmId::Area(99)],
        "one level INTO the live realm — stated by that realm itself"
    );
}

#[test]
fn a_child_window_guards_unrostered_ship_and_rotated_hops_counted() {
    // The Child-scope guards, each dropped + counted, never a guess and never a panic:
    // a stranger realm (`window_child_unrostered`), a Ship child (D-SHIP-1 — no lineage
    // coord until P8, the SAME counter every coord lane uses), and a ROTATED placement BEYOND
    // THE EXACT ROTATION REACH (the frame core's own restated refusal — real-scale addendum
    // §A4.6: an in-reach rotated hop now FOLDS exactly, so the guard fires only past
    // 2⁴² m ≈ 29.4 AU — `window_hop_refused`, R2's P10 trigger). The Occupants window beside
    // them keeps serving: one bad hop never mutes the lane.
    let ship_realm = RealmId::Ship(EntityId::pack(EntityKind::Ship, 1, 7, 3));
    let rotated = RealmId::Planet(44);
    let mut rig = Rig::new();
    rig.grant_realm();
    let placed = FramePlacement {
        // 2⁵³ cells = 2× the FINE rotation reach: the one rotated shape the restated law
        // still refuses.
        origin_cell: vd_core::glam::I64Vec3::new(1_i64 << 53, 0, 0),
        origin: DVec3::new(5.0, 0.0, 0.0),
        velocity: DVec3::ZERO,
        orientation: DQuat::from_rotation_z(0.3),
        angular_velocity: DVec3::ZERO,
    };
    let motions: BTreeMap<RealmId, MotionFn> =
        [(rotated, MotionFn(std::sync::Arc::new(move |_| placed)))]
            .into_iter()
            .collect();
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(vec![
        root_region(),
        own_region(),
        region(rotated, Some(OWN_REALM), DVec3::ZERO, 100.0),
        region(
            ship_realm,
            Some(OWN_REALM),
            DVec3::new(50_000.0, 0.0, 0.0),
            10.0,
        ),
    ])
    .with_moving_children(motions);
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(5_000.0, 0.0, 0.0));
    let opens: Vec<Inbound> = [
        (WindowId(1), WindowScope::Occupants),
        (WindowId(2), WindowScope::Child(RealmId::Planet(555))), // rostered NOWHERE
        (WindowId(3), WindowScope::Child(ship_realm)),           // D-SHIP-1
        (WindowId(4), WindowScope::Child(rotated)),              // the P10-owed refusal
    ]
    .into_iter()
    .map(|(window, scope)| {
        wire_msg(
            GATEWAY,
            MsgClass::Control,
            &GatewayToShard::WindowOpen { window, scope },
        )
    })
    .collect();
    let sent = rig.tick(opens);
    let frames = window_frames(&sent);
    assert_eq!(
        frames.iter().map(|f| f.1).collect::<Vec<_>>(),
        vec![WindowId(1)],
        "only the Occupants window serves — every guarded Child window is withheld"
    );
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(stats.window_child_unrostered, 1);
    // The ship exclusion counts once in the AoI/demand fold and once on the window lane —
    // the SAME per-lane-pass counting the other D-SHIP-1 guards use.
    assert_eq!(stats.ship_child_regions_excluded, 2);
    assert_eq!(stats.window_hop_refused, 1);
    assert_eq!(
        stats.windows_open, 4,
        "guarded windows stay open — only their frames are withheld"
    );
    // The refusal is the frame core's own (one rule, inherited): asserted directly, plus the
    // unknown-frame arm the emitter can never reach (the roster resolves first).
    let book = rig
        .world
        .resource::<Placements>()
        .0
        .head(OWN_REALM)
        .expect("authored")
        .clone();
    assert_eq!(
        invert_hop_placement(frame_of(OWN_REALM), frame_of(rotated), &book),
        Err(FrameError::RotationBeyondExactReach)
    );
    assert_eq!(
        invert_hop_placement(
            frame_of(OWN_REALM),
            FrameRef::PlanetCentered { planet_seed: 777 },
            &book
        ),
        Err(FrameError::UnknownDestFrame)
    );
}

#[test]
fn window_membership_rides_the_one_fold_and_clears_with_the_last_observer() {
    // §2.9 step 5: the verdict is THE existing aoi_decide fold — per-dot for Occupants
    // (scoped to the OPENER's own dots: a second gateway's window holds an EMPTY verdict),
    // the occupied-child proxy set for Child scopes — and when the LAST observer leaves, the
    // emptiness pass clears every window's verdict (the removals ship once).
    let near = DVec3::new(500.0, 0.0, 0.0);
    let mut rig = parent_with_two_planet_children(near);
    inject_bit(&mut rig, RealmId::Planet(42));
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    let sent = rig.tick(vec![
        wire_msg(
            GATEWAY,
            MsgClass::Control,
            &GatewayToShard::WindowOpen {
                window: WindowId(1),
                scope: WindowScope::Occupants,
            },
        ),
        // A SECOND subscriber (another gateway) with no dots here: its per-dot verdict is
        // empty — the fold is scoped per opener, never a global union.
        wire_msg(
            ANCESTOR,
            MsgClass::Control,
            &GatewayToShard::WindowOpen {
                window: WindowId(1),
                scope: WindowScope::Occupants,
            },
        ),
        wire_msg(
            GATEWAY,
            MsgClass::Control,
            &GatewayToShard::WindowClose {
                window: WindowId(7), // unknown — the counted no-op rides the same tick
            },
        ),
        wire_msg(
            GATEWAY,
            MsgClass::Control,
            &GatewayToShard::WindowOpen {
                window: WindowId(2),
                scope: WindowScope::Child(RealmId::Planet(42)),
            },
        ),
    ]);
    let members = window_memberships(&sent);
    // GATEWAY's dot stands in BOTH armed bands (42 at the origin, 43 at 500) ⇒ its Occupants
    // verdict is both; the occupied child 42 (reach = its own 1000 m extent) also reaches
    // both ⇒ the Child(42) verdict is both; ANCESTOR's window ships NOTHING (empty verdict).
    let both = vec![RealmId::Planet(42), RealmId::Planet(43)];
    assert_eq!(
        members,
        vec![
            (GATEWAY, WindowId(1), both.clone(), vec![]),
            (GATEWAY, WindowId(2), both.clone(), vec![]),
        ]
    );
    // The LAST observer leaves (the dot detaches, the child bit expires): the emptiness pass
    // ships the removals — every window's verdict returns to the empty set, exactly once.
    rig.world.resource_mut::<Dots>().0.clear();
    rig.world.resource_mut::<ChildLiveness>().0.clear();
    rig.set_local_tick(2);
    let sent = rig.tick(vec![]);
    assert_eq!(
        window_memberships(&sent),
        vec![
            (GATEWAY, WindowId(1), vec![], both.clone()),
            (GATEWAY, WindowId(2), vec![], both),
        ]
    );
    assert_eq!(
        rig.world.resource::<StubStats>().window_close_unknown,
        1,
        "the unknown close counted"
    );
}
