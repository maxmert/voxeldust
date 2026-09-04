//! THE ARRIVAL RACE — a player is mid-crossing INTO a realm and the reconciler reaps that realm out
//! from under them.
//!
//! WHAT THIS FILE PROVES TODAY (slice 8a, the reproduction): a durable crossing that has passed the
//! commit CAS but whose `Promote` never lands leaves the destination realm completely unprotected. The
//! source shard's keep-alive for the destination is tied to its in-flight crossing latch
//! (`vd_sim` `redrive_stranded_crossings` scans only HELD latches), and that latch is cleared when the
//! source applies the ordered `Demote` (`vd_sim` `on_saga_demote`) — one step AFTER the commit and one
//! step BEFORE the arrival. So the destination's protection ends before the player gets there. From
//! that point the destination honestly reports itself empty (an arriving player is a Ghost until the
//! promote, and only SIMULATED dots count as occupants), the demand TTL expires, the drain elapses, and
//! the orchestrator kills a realm the directory says is the committed owner of a live player.
//!
//! THE BELIEF THIS FILE RETIRES: that the directory's `in_transfer` field shields a landing realm. It
//! does not. `in_transfer` is set on the SUBJECT's key (the player), never on the realm's, and the
//! commit CAS clears even that. `the_dest_realm_head_is_never_in_transfer_during_a_crossing` asserts the
//! realm head's `in_transfer` is `None` on EVERY tick of a real crossing — so the teardown clause that
//! reads it can never be false, and shields nothing.
//!
//! WHAT IS REAL HERE: the orchestrator's whole per-tick chain (clock → demand ingest → directory service
//! → saga drive → reconcile → commit barrier), the transfer FSM, the directory CAS, and the lifecycle
//! reconciler. What is MODELLED is only the peer shards' outgoing wire frames — the same contract
//! `realm_lifecycle_e2e.rs` documents. Nothing the test asserts about is stubbed.

use bevy_ecs::prelude::{Schedule, World};

use vd_node::orchestrator::{DirectoryRes, OrchestratorConfig, register_orchestrator_with_store};
use vd_node::rlm_runtime::RlmReconcilerRes;
use vd_node::saga_runtime::SagaRuntimeRes;
use vd_sim::directory::DirectoryTuning;
use vd_sim::io::mem::{MemHub, MemSpawner, MemStore};
use vd_sim::io::{Inbound, MsgClass};
use vd_sim::rlm::RlmTuning;
use vd_sim::runtime::{ClockSample, InboundBox, OutboundBox};
use vd_sim::saga::{LivenessTuning, SagaCtx, SagaTuning};
use vd_wire::intershard::{
    DemandVerb, FLUSH_SOURCE_STEP, InterShardFlow, RealmDemand, TransferAck,
};
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey, OwnerRecord};
use vd_wire::seams::transfer_control::{PrepareResult, TransferControlAck};

use vd_core::entity_kind::{DurabilityClass, EntityKind};
use vd_core::glam::DVec3;
use vd_core::pose::{FrameRef, RealmId, StampedPose};
use vd_core::realm_coord::RealmCoord;
use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};
use vd_core::{EntityId, EpochId, Fence, NodeId, SessionId, TransferId, UniverseTick};

// ---- the cast -----------------------------------------------------------------------------------

/// The shard the player is leaving (also the shard that authors the star system itself).
const SOURCE: NodeId = NodeId(30);
/// The shard hosting the planet the player is arriving at.
const DEST: NodeId = NodeId(40);
/// The gateway the transfer control acks ride back from.
const GATEWAY: NodeId = NodeId(20);
const XFER: TransferId = TransferId(7701);
const SESSION: SessionId = SessionId(55);

/// The lineage: a star system with one planet inside it. The player crosses INWARD, system → planet,
/// which is the exact shape of the in-game round trip that produced the two prior lifecycle bugs.
const SYSTEM_SEED: u64 = 1;
const PLANET_SEED: u64 = 2;
const FROM_REALM: RealmId = RealmId::System(SYSTEM_SEED);
const TO_REALM: RealmId = RealmId::Planet(PLANET_SEED);

fn planet_coord() -> RealmCoord {
    RealmCoord::from_path(RealmPath::from_levels(vec![
        RealmLevel::new(RealmKindTag::System, SYSTEM_SEED),
        RealmLevel::new(RealmKindTag::Planet, PLANET_SEED),
    ]))
    .expect("a two-level path has a leaf")
}

fn subject_eid() -> EntityId {
    EntityId::pack(EntityKind::Player, 1, 7, 3)
}

fn subject() -> DirectoryKey {
    DirectoryKey::Entity(subject_eid())
}

/// The crossing context a source shard's boundary detector produces for this lineage.
fn crossing_ctx() -> SagaCtx {
    SagaCtx {
        exterior: false,
        transfer: XFER,
        session: SESSION,
        subject: subject(),
        expected_fence: Fence(1),
        source: SOURCE,
        dest: DEST,
        class: DurabilityClass::Durable,
        needs_provision: false,
        from_realm: FROM_REALM,
        to_realm: TO_REALM,
        to_parent: Some(FROM_REALM),
    }
}

// ---- the rig ------------------------------------------------------------------------------------

/// A real orchestrator World + schedule, with the lifecycle reconciler ARMED — the production
/// configuration of a demand cluster.
struct Rig {
    world: World,
    schedule: Schedule,
}

fn orch_cfg(saga: SagaTuning) -> OrchestratorConfig {
    OrchestratorConfig {
        epoch: EpochId(1),
        reserve_chunk: 1024,
        clock_peers: vec![],
        directory: DirectoryTuning::default(),
        saga,
        liveness: LivenessTuning::default(),
        roster: std::collections::BTreeMap::new(),
        // ARMED: the cloud budget at 20 Hz — the same tuning a `--demand` orchestrator boots with.
        rlm: RlmTuning::cloud(20),
    }
}

impl Rig {
    /// `saga` is the deployment's transfer budget. It matters here because the arrival shield's leak
    /// bound is DERIVED from it at boot — so shrinking it is a production-legal way to give the shield
    /// a budget too small to cover the reclaim window, which is exactly the mutation control.
    fn with_saga(saga: SagaTuning) -> Rig {
        let mut world = World::new();
        world.insert_resource(InboundBox::default());
        world.insert_resource(OutboundBox::default());
        world.insert_resource(ClockSample::default());
        let mut schedule = Schedule::default();
        register_orchestrator_with_store(
            &mut world,
            &mut schedule,
            &orch_cfg(saga),
            Box::new(MemStore::new()),
            Box::new(MemSpawner::new(MemHub::new(), NodeId(1_000_000), 8)),
            vd_node::rlm_runtime::LaunchSeed::new(),
        );
        Rig { world, schedule }
    }

    fn now(&self) -> UniverseTick {
        self.world.resource::<ClockSample>().universe_tick
    }

    fn rlm(&self) -> &RlmReconcilerRes {
        self.world.resource::<RlmReconcilerRes>()
    }

    /// The directory record for a realm — `None` once its head is revoked (i.e. once it is reaped).
    fn realm_head(&self, realm: RealmId) -> Option<OwnerRecord> {
        self.world
            .resource::<DirectoryRes>()
            .0
            .head(DirectoryKey::Realm(realm))
    }

    /// The live sagas' states, as the admin snapshot reports them.
    fn saga_states(&self) -> Vec<String> {
        self.world
            .resource::<SagaRuntimeRes>()
            .views()
            .into_iter()
            .map(|v| v.state)
            .collect()
    }

    /// Model a shard REGISTERING a key it owns (self-granting a realm head / an avatar record) — the
    /// exact directory state a booted shard reaches. The launch itself is elsewhere-proven.
    fn grant(&mut self, key: DirectoryKey, owner: NodeId, fence: Fence) {
        let now = self.now();
        self.world.resource_mut::<DirectoryRes>().0.grant(
            key,
            AuthorityRef::Shard(owner),
            fence,
            now,
        );
    }

    /// Arm the crossing through the real entry point a resolved `CrossingRequest` uses.
    fn start_crossing(&mut self) {
        self.world
            .resource_mut::<SagaRuntimeRes>()
            .start_transfer(crossing_ctx(), GATEWAY);
    }

    /// Run ONE real orchestrator tick with `frames` delivered first.
    fn tick(&mut self, frames: Vec<Inbound>) {
        self.world.resource_mut::<InboundBox>().0 = frames;
        self.schedule.run(&mut self.world);
        // The realm head must never be `in_transfer` — the belief this file retires. Checked on EVERY
        // tick rather than once at the end, so no phase of the crossing can hide a transient lock.
        assert_eq!(
            self.realm_head(TO_REALM).and_then(|r| r.in_transfer),
            None,
            "the DESTINATION REALM's head is never locked by a crossing — the transfer lock lives on \
             the player's key, so a teardown clause reading the realm's lock can never be false"
        );
    }
}

// ---- the modelled peer frames -------------------------------------------------------------------

fn wire(from: NodeId, flow: &InterShardFlow) -> Inbound {
    Inbound::Wire {
        from,
        class: MsgClass::Saga,
        bytes: postcard::to_allocvec(flow).expect("encode").into(),
    }
}

/// A realm demand exactly as a shard's AoI/crossing machinery emits it.
fn demand(coord: &RealmCoord, verb: DemandVerb, tick: UniverseTick) -> Inbound {
    wire(
        SOURCE,
        &InterShardFlow::RealmDemand(RealmDemand {
            child: coord.clone(),
            parent_fence: Fence(1),
            verb,
            universe_tick: tick,
        }),
    )
}

/// One transfer-control ack, as the gateway ships it back.
fn ack(a: TransferControlAck) -> Inbound {
    wire(GATEWAY, &InterShardFlow::SagaAck(a))
}

/// The source shard shipping its authoritative pose (the second half of the pose-before-promote gate).
fn source_flush() -> Inbound {
    wire(
        SOURCE,
        &InterShardFlow::TransferAck(TransferAck::SourceFlushed {
            transfer_id: XFER,
            step_id: FLUSH_SOURCE_STEP,
            pose: StampedPose::at_rest(
                FrameRef::SystemSpace {
                    system_seed: SYSTEM_SEED,
                },
                DVec3::new(1.0, 2.0, 3.0),
                UniverseTick(5),
            ),
            drained_seq: 42,
            state: vec![],
        }),
    )
}

// ---- the scenario -------------------------------------------------------------------------------

/// Drive a REAL durable crossing from `start_transfer` through the commit CAS and the demote, then
/// STOP — the `Promote` is withheld, so the saga parks post-commit exactly as a lost or refused promote
/// leaves it. Returns the rig with the player committed to `DEST` and the saga live in `Promoting`.
///
/// The keep-alive discipline mirrors the source shard exactly: it demands the destination every tick
/// while its crossing latch is held, and the latch is cleared when the source applies the ordered
/// `Demote` — so the last keep-alive is the one emitted on the demote tick, and it never resumes. The
/// fixture is deliberately GENEROUS about where that boundary sits (it keeps demanding through the
/// demote-ack tick), because over-stating the protection can only make the race harder to reproduce.
/// The realistic durable crossing: the source shard IS demanding the destination while its latch is
/// held, so the destination also carries the tail of that demand's own freshness window.
fn crossing_parked_after_commit() -> Rig {
    crossing_parked_after_commit_with(SagaTuning::default(), true)
}

/// THE DOOR THE SHIELD EXISTS FOR: a destination that gets NO keep-alive from the departing side at
/// all. Three real configurations reach this — the departing shard losing its lease mid-crossing (it
/// stops demanding immediately), a batch of non-persistent things crossing over (which never had a
/// per-entity latch to demand from), and a cluster running the reclaimer with the shards' area-of-
/// interest switched off. In every one of them the destination is naked, and the shield is the only
/// thing between an arriving subject and a reaped destination.
fn crossing_parked_with_no_cover() -> Rig {
    crossing_parked_after_commit_with(SagaTuning::default(), false)
}

fn crossing_parked_after_commit_with(saga: SagaTuning, source_keeps_dest_alive: bool) -> Rig {
    let mut rig = Rig::with_saga(saga);
    rig.tick(vec![]); // tick 1 — the clock starts

    // The shards register what they own: the source holds the player, and both realms have live heads.
    rig.grant(subject(), SOURCE, Fence(1));
    rig.grant(DirectoryKey::Realm(FROM_REALM), SOURCE, Fence(1));
    rig.grant(DirectoryKey::Realm(TO_REALM), DEST, Fence(1));

    // SETTLE THE DESTINATION FIRST. A realm cannot be reclaimed inside its own boot-and-settle window,
    // so a brand-new planet is protected by that floor and the race cannot even be posed against it.
    // The realm a player actually crosses into has normally been alive for a while — somebody was there
    // a minute ago — so the fixture runs the planet as a demanded, occupied realm past that floor
    // before the crossing starts. Without this the test would pass for the wrong reason.
    // The realm is held through the settle by the rule that a LIVE realm only leaves the wanted set by
    // saying it is empty — so no demand is needed, and in the uncovered case none is sent. That matters:
    // a demand here would leave its own freshness window covering the destination for long after, and
    // the test would then be measuring that window rather than the shield.
    let settle = RlmTuning::cloud(20).min_dwell_ticks() + 5;
    for _ in 0..settle {
        let frames = if source_keeps_dest_alive {
            vec![demand(&planet_coord(), DemandVerb::KeepAlive, rig.now())]
        } else {
            vec![]
        };
        rig.tick(frames);
    }

    // The boundary detector fires: the crossing arms, and from this tick the source keeps the
    // destination demand-alive every tick (its in-flight latch is held).
    rig.start_crossing();
    let ka = |rig: &Rig| {
        if source_keeps_dest_alive {
            vec![demand(&planet_coord(), DemandVerb::KeepAlive, rig.now())]
        } else {
            vec![]
        }
    };

    // One tick per step of the real hand-off, each carrying the source's keep-alive alongside the ack
    // it would really have been emitted with.
    let step = |rig: &mut Rig, frame: Option<Inbound>| {
        let mut frames = ka(rig);
        frames.extend(frame);
        rig.tick(frames);
    };

    step(&mut rig, None); // the start is processed → PrepareSubscribe
    step(
        &mut rig,
        Some(ack(TransferControlAck::Prepared {
            transfer: XFER,
            result: PrepareResult::Ready,
        })),
    );
    step(
        &mut rig,
        Some(ack(TransferControlAck::CutConfirmed {
            transfer: XFER,
            marker_seq: 42,
        })),
    );
    step(
        &mut rig,
        Some(ack(TransferControlAck::SourceFrozen {
            transfer: XFER,
            drained_seq: 42,
        })),
    );
    // THE COMMIT: both gate halves land, the CAS wins in-process, authority flips to DEST.
    step(&mut rig, Some(source_flush()));
    assert_eq!(
        rig.world
            .resource::<DirectoryRes>()
            .0
            .head(subject())
            .expect("the player has a directory record")
            .authority,
        AuthorityRef::Shard(DEST),
        "the CAS committed — the directory now says the player belongs to the planet's shard"
    );

    // The route swap acks and the saga pushes the ordered Demote to the source — the tick the source's
    // crossing latch dies, taking the destination's keep-alive with it. The fixture keeps demanding
    // through here and one tick past (the generous reading); after that, nothing keeps the planet alive.
    step(
        &mut rig,
        Some(ack(TransferControlAck::Committed { transfer: XFER })),
    );
    step(
        &mut rig,
        Some(ack(TransferControlAck::DemoteAck { transfer: XFER })),
    );
    assert_eq!(
        rig.saga_states().len(),
        1,
        "the crossing is still live — it is waiting for the promote"
    );
    assert!(
        rig.saga_states()[0].starts_with("Promoting"),
        "parked post-commit awaiting the promote, got {:?}",
        rig.saga_states()
    );
    rig
}

#[test]
fn the_arrival_shield_is_armed_exactly_when_the_reconciler_is() {
    // The shield and the thing it protects against are armed by the same switch, so there is no
    // configuration in which realms are being reclaimed while arrivals go unprotected. Note what this
    // replaces: the destination used to be covered only by a keep-alive the SOURCE SHARD sent, which
    // meant a cluster with the reclaimer on and the shards' area-of-interest off ran the reclaimer with
    // no cover at all. The shield is derived by the orchestrator from its own hand-offs, so that
    // configuration cannot exist any more.
    let armed = Rig::with_saga(SagaTuning::default());
    assert!(
        armed.rlm().tuning().arrival_shield_ticks > 0,
        "an armed reconciler always boots with a derived shield budget"
    );

    let mut world = World::new();
    world.insert_resource(InboundBox::default());
    world.insert_resource(OutboundBox::default());
    world.insert_resource(ClockSample::default());
    let mut schedule = Schedule::default();
    register_orchestrator_with_store(
        &mut world,
        &mut schedule,
        &OrchestratorConfig {
            rlm: RlmTuning::default(), // the reconciler never sweeps
            ..orch_cfg(SagaTuning::default())
        },
        Box::new(MemStore::new()),
        Box::new(MemSpawner::new(MemHub::new(), NodeId(2_000_000), 8)),
        vd_node::rlm_runtime::LaunchSeed::new(),
    );
    assert_eq!(
        world
            .resource::<RlmReconcilerRes>()
            .tuning()
            .arrival_shield_ticks,
        0,
        "and a reconciler that never reclaims anything carries no shield budget — zero, not a guess"
    );
}

/// Run the parked crossing forward, with the destination's shard truthfully reporting itself empty
/// every tick (the arriving player is held frozen and uncounted until the promote, which never comes)
/// and nobody else demanding it. Returns whether the destination was reaped.
fn run_until_reap_or(rig: &mut Rig, ticks: u32) -> bool {
    for _ in 0..ticks {
        let e = demand(&planet_coord(), DemandVerb::Empty, rig.now());
        rig.tick(vec![e]);
        if rig.rlm().teardowns_reaped == 1 {
            return true;
        }
    }
    false
}

/// The window in which an unshielded destination is reclaimed: the drain plus the teardown cooldown,
/// with room for the empty report to be confirmed. Derived from the same tuning the orchestrator runs.
fn reclaim_window_ticks() -> u32 {
    let t = RlmTuning::cloud(20);
    (t.teardown_drain_ticks + t.teardown_cooldown_ticks + t.empty_grace_ticks + 5) as u32
}

#[test]
fn a_stalled_promote_shields_the_dest_realm_from_the_reap() {
    // SLICE 9a — the inversion of the 8a reproduction. The identical fixture that reaped the planet out
    // from under an arriving player now holds it, because the orchestrator's own list of in-progress
    // hand-offs says somebody is on their way there.
    let mut rig = crossing_parked_with_no_cover();
    // STATE THE MARGIN, DON'T ASSUME IT. The previous protection for a landing realm was a timing
    // margin nobody had written down, which is how it went unnoticed that it had stopped covering the
    // last step. So the relationship this test depends on — the shield's budget outlasting the window
    // that would otherwise reclaim the realm — is asserted here rather than left implicit.
    assert!(
        u64::from(reclaim_window_ticks()) < rig.rlm().tuning().arrival_shield_ticks,
        "the fixture only means anything if the shield outlasts the reclaim window: window {}, shield {}",
        reclaim_window_ticks(),
        rig.rlm().tuning().arrival_shield_ticks
    );
    assert!(
        !run_until_reap_or(&mut rig, reclaim_window_ticks()),
        "the planet a committed player is arriving at survives the whole window that used to reclaim it"
    );
    assert!(
        rig.realm_head(TO_REALM).is_some(),
        "its head is intact — there is somewhere to land"
    );
    assert!(
        rig.realm_head(FROM_REALM).is_some(),
        "and so is its PARENT's: the arrival is a reason to WANT the realm, and a want flows up the \
         whole chain — otherwise the player lands on a planet whose star system just shut down"
    );
    assert_eq!(
        rig.saga_states().len(),
        1,
        "the crossing is still live and still waiting — the shield holds while it does"
    );
    assert!(
        rig.rlm().arrival_shield_vetoes > 0,
        "and the shield is COUNTED BY NAME while it is the only thing holding the realm up"
    );
    assert_eq!(
        rig.rlm().arrivals_shield_expired,
        0,
        "nothing has been abandoned — the hand-off is still inside its budget"
    );
}

#[test]
fn the_mutation_control_the_reap_reappears_when_the_shield_cannot_cover_the_window() {
    // THE CONTROL, and the slice is invalid without it. Same fixture, same order of events, ONE input
    // changed: a transfer budget so small that the shield's derived leak bound cannot span the reclaim
    // window. If the reap does NOT come back, the test above is passing for some reason other than the
    // shield and proves nothing. This arc has already produced one vacuous test and inherited another.
    let mut rig = crossing_parked_after_commit_with(
        SagaTuning {
            // The DESTRUCTIVE deadline is left healthy — shrinking that aborts the crossing before it
            // ever commits, which would change the scenario instead of controlling it. Only the cheap
            // re-drive budget is shrunk, and the shield's bound is derived from it.
            redrive_deadline_ticks: 1,
            abort_deadline_ticks: 24,
        },
        false,
    );
    assert!(
        run_until_reap_or(&mut rig, reclaim_window_ticks() * 4),
        "with a shield too short to cover the window the destination is reaped again — so the shield \
         is what saved it in the test above"
    );
    assert_eq!(
        rig.realm_head(TO_REALM),
        None,
        "the head is revoked, exactly as it was before the fix"
    );
    assert_eq!(
        rig.rlm().arrivals_shield_expired,
        1,
        "and the abandonment is LOUD and counted — exactly once for the one wedged hand-off, not once \
         per sweep forever"
    );
}

#[test]
fn the_shield_is_a_leak_bound_not_a_licence_to_pin_a_realm_forever() {
    // The deliberate failure mode: a hand-off that is wedged rather than slow eventually loses its
    // protection, the realm is reclaimed, and the subject falls back to the orphan recovery that
    // already exists. The alternative — pin the realm forever — leaks one live realm per wedged
    // hand-off, which does not survive contact with a hundred thousand players.
    let mut rig = crossing_parked_with_no_cover();
    assert!(
        run_until_reap_or(&mut rig, 2_000),
        "the shield is bounded: a permanently stuck hand-off cannot hold a realm indefinitely"
    );
    assert_eq!(rig.rlm().arrivals_shield_expired, 1);
}

#[test]
fn the_dest_realm_head_is_never_in_transfer_during_a_crossing() {
    // The executable retirement of "the mid-transfer clause already shields a landing realm". `Rig::tick`
    // asserts it on every tick of the crossing; this test names the belief and drives the phases that
    // would have to set it. The lock is taken on the PLAYER's key and cleared by the commit CAS.
    let rig = crossing_parked_after_commit();
    assert_eq!(
        rig.realm_head(TO_REALM)
            .expect("the planet is still alive at this point")
            .in_transfer,
        None,
        "no phase of a real crossing ever locks the destination realm's directory key"
    );
    assert_eq!(
        rig.world
            .resource::<DirectoryRes>()
            .0
            .head(subject())
            .expect("the player has a record")
            .in_transfer,
        None,
        "even the PLAYER's lock is cleared at the commit CAS — so nothing is locked post-commit, which \
         is exactly the window the arrival race lives in"
    );
}

#[test]
fn a_healthy_crossing_keeps_the_planet_alive_all_the_way_to_done() {
    // The control: the SAME fixture with the promote delivered settles cleanly and never reaps. This is
    // what makes the failing test above a race rather than a broken fixture.
    let mut rig = crossing_parked_after_commit();
    rig.tick(vec![ack(TransferControlAck::PromoteAck { transfer: XFER })]);
    rig.tick(vec![ack(TransferControlAck::DeliveredToObservers {
        transfer: XFER,
    })]);
    rig.tick(vec![ack(TransferControlAck::Released { transfer: XFER })]);
    assert_eq!(
        rig.saga_states(),
        Vec::<String>::new(),
        "the crossing completed and tombstoned"
    );
    // The player now lives on the planet, so its shard reports the planet OCCUPIED (a KeepAlive, which
    // is what an occupied realm's own AoI emits) — and it is never reaped.
    for _ in 0..200 {
        let d = demand(&planet_coord(), DemandVerb::KeepAlive, rig.now());
        rig.tick(vec![d]);
    }
    assert_eq!(
        rig.rlm().teardowns_reaped,
        0,
        "an arrived-and-settled player keeps their planet alive"
    );
    assert!(
        rig.realm_head(TO_REALM).is_some(),
        "the planet is still running under the arrived player"
    );
}
