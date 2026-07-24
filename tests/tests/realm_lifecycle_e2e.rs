//! RLM Step 3f — the adversarial END-TO-END capstone for the realm-lifecycle reconciler.
//!
//! Drives a REAL orchestrator through its REAL per-tick chain (`advance_and_broadcast_clock →
//! record_realm_demands → serve_directory → drive_sagas_core → reconcile_realm_lifecycle →
//! commit_barrier`) with a LIVE `RlmTuning`, feeding it the EXACT `InterShardFlow::RealmDemand` wire
//! frames a Step-2 shard emits (the production payload — Step 2 already proves `evaluate_realm_aoi`
//! produces these bytes at 100%). The spawner is an inspectable `MemSpawner`; the durable store is retained
//! so the orchestrator-crash cell rebuilds from its committed WAL.
//!
//! A spawned realm's OWN shard (which self-grants its head + emits its own `Empty`) does not exist in this
//! test — the REAL child-shard launch is RLM Step 5. Its registration is modelled by granting the realm
//! head directly (the exact directory state a booted shard reaches); its emptiness is fed as the real
//! `Empty` wire frame that shard would send. Neither is a stub of the code under test — the orchestrator
//! chain + the reconciler are 100% real; only the not-yet-built peer shards are modelled at their wire seam.

use bevy_ecs::prelude::{Schedule, World};

use vd_node::orchestrator::{DirectoryRes, OrchestratorConfig, register_orchestrator_with_store};
use vd_node::rlm_runtime::RlmReconcilerRes;
use vd_sim::directory::DirectoryTuning;
use vd_sim::io::mem::{MemHub, MemSpawner, MemStore};
use vd_sim::io::{Inbound, MsgClass, RealmSpawner};
use vd_sim::rlm::RlmTuning;
use vd_sim::runtime::{ClockSample, InboundBox, OutboundBox};
use vd_sim::saga::{LivenessTuning, SagaTuning};
use vd_wire::intershard::{DemandVerb, InterShardFlow, RealmDemand};
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey};

use vd_core::pose::RealmId;
use vd_core::realm_coord::RealmCoord;
use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};
use vd_core::{EpochId, Fence, NodeId, UniverseTick};

const STUB: NodeId = NodeId(10); // the (real, elsewhere-proven) demand-emitting shard's id
const CHILD_SHARD: NodeId = NodeId(50); // the id a spawned child shard self-grants under (modelled)

/// A real orchestrator World + its schedule + an inspectable spawner + the retained durable store.
struct Orch {
    world: World,
    schedule: Schedule,
    spawner: MemSpawner,
    store: MemStore,
}

fn orch_cfg(rlm: RlmTuning) -> OrchestratorConfig {
    OrchestratorConfig {
        epoch: EpochId(1),
        reserve_chunk: 1024,
        clock_peers: vec![],
        directory: DirectoryTuning::default(),
        saga: SagaTuning::default(),
        liveness: LivenessTuning::default(),
        roster: std::collections::BTreeMap::new(),
        rlm,
    }
}

fn assemble(rlm: RlmTuning, spawner: MemSpawner, store: MemStore) -> Orch {
    let mut world = World::new();
    world.insert_resource(InboundBox::default());
    world.insert_resource(OutboundBox::default());
    world.insert_resource(ClockSample::default());
    let mut schedule = Schedule::default();
    register_orchestrator_with_store(
        &mut world,
        &mut schedule,
        &orch_cfg(rlm),
        Box::new(store.clone()),
    );
    // Overwrite the default (internal-spawner) reconciler with one holding OUR inspectable spawner.
    world.insert_resource(RlmReconcilerRes::new(rlm, Box::new(spawner.clone())));
    Orch {
        world,
        schedule,
        spawner,
        store,
    }
}

fn build() -> Orch {
    assemble(
        RlmTuning::cloud(20),
        MemSpawner::new(MemHub::new(), NodeId(1_000_000), 8),
        MemStore::new(),
    )
}

impl Orch {
    /// The orchestrator's current universe tick (advanced each `run`).
    fn now(&self) -> UniverseTick {
        self.world.resource::<ClockSample>().universe_tick
    }

    fn rlm(&self) -> &RlmReconcilerRes {
        self.world.resource::<RlmReconcilerRes>()
    }

    /// The realm heads currently in the directory (what is "running").
    fn head(&self, rid: RealmId) -> bool {
        self.world
            .resource::<DirectoryRes>()
            .0
            .head(DirectoryKey::Realm(rid))
            .is_some()
    }

    /// Run ONE real orchestrator tick, delivering `demands` as real Saga-class wire frames first.
    fn tick(&mut self, demands: &[RealmDemand]) {
        let inbox: Vec<Inbound> = demands
            .iter()
            .map(|d| Inbound::Wire {
                from: STUB,
                class: MsgClass::Saga,
                bytes: postcard::to_allocvec(&InterShardFlow::RealmDemand(d.clone()))
                    .expect("encode")
                    .into(),
            })
            .collect();
        self.world.resource_mut::<InboundBox>().0 = inbox;
        self.schedule.run(&mut self.world);
    }

    /// Model a spawned child shard REGISTERING (self-granting its realm head) — the directory state a
    /// booted shard reaches (the real launch is Step 5). `node` is "alive" (never latched dead).
    fn register_shard(&mut self, rid: RealmId, node: NodeId) {
        let now = self.now();
        self.world.resource_mut::<DirectoryRes>().0.grant(
            DirectoryKey::Realm(rid),
            AuthorityRef::Shard(node),
            Fence(1),
            now,
        );
    }

    /// Rebuild the orchestrator from its retained WAL (a kill-9): a FRESH world, the SAME store.
    fn crash_rebuild(self) -> Orch {
        assemble(
            RlmTuning::cloud(20),
            MemSpawner::new(MemHub::new(), NodeId(2_000_000), 8),
            self.store,
        )
    }
}

// ---- coords ------------------------------------------------------------------------------------

fn system(seed: u64) -> RealmCoord {
    RealmCoord::from_path(RealmPath::from_levels(vec![RealmLevel::new(
        RealmKindTag::System,
        seed,
    )]))
    .expect("one-level path")
}

/// A demand for `coord` at the orchestrator's current tick (a Step-2 shard stamps its own `now`).
fn demand(orch: &Orch, coord: &RealmCoord, verb: DemandVerb) -> RealmDemand {
    RealmDemand {
        child: coord.clone(),
        parent_fence: Fence(1),
        verb,
        universe_tick: orch.now(),
    }
}

// ---- the scenarios -----------------------------------------------------------------------------

#[test]
fn a_demand_spins_up_the_realm_then_holds_while_launching() {
    let mut orch = build();
    let c = system(7);
    orch.tick(&[]); // advance the genesis tick so `now` is live
    // A real SpinUp wire frame ⇒ the reconciler spawns the realm exactly once.
    let d = demand(&orch, &c, DemandVerb::SpinUp);
    orch.tick(&[d]);
    assert_eq!(
        orch.rlm().spins_requested,
        1,
        "the demand spun up the realm"
    );
    assert_eq!(orch.spawner.live_nodes().len(), 1, "one pod launched");
    // Re-asserted while still launching (no head yet) ⇒ NO double-spawn (BUG-B).
    let d2 = demand(&orch, &c, DemandVerb::KeepAlive);
    orch.tick(&[d2]);
    assert_eq!(
        orch.rlm().spins_requested,
        1,
        "no double-spawn while launching"
    );
}

#[test]
fn a_registered_realm_stays_running_under_keepalive() {
    let mut orch = build();
    let c = system(7);
    orch.tick(&[]);
    orch.tick(&[demand(&orch, &c, DemandVerb::SpinUp)]);
    orch.register_shard(RealmId::System(7), CHILD_SHARD); // the child shard boots + self-grants
    // KeepAlive on a running realm ⇒ no re-spawn, no kill; it is counted running.
    for _ in 0..5 {
        let d = demand(&orch, &c, DemandVerb::KeepAlive);
        orch.tick(&[d]);
    }
    assert_eq!(orch.rlm().spins_requested, 1);
    assert_eq!(orch.rlm().teardowns_reaped, 0);
    assert!(orch.head(RealmId::System(7)), "the realm is still running");
}

#[test]
fn no_strand_a_running_realm_with_no_demand_and_no_empty_is_never_killed() {
    // EDGE #1 (arm-B NO-STRAND): the parent partitions (stops demanding) and the realm never affirmatively
    // reports Empty (a player is inside) ⇒ it is NEVER reaped, however long the silence lasts.
    let mut orch = build();
    orch.tick(&[]);
    orch.register_shard(RealmId::System(7), CHILD_SHARD);
    for _ in 0..200 {
        orch.tick(&[]); // no demands at all — total silence
    }
    assert_eq!(
        orch.rlm().teardowns_reaped,
        0,
        "a live realm is never killed without an affirmative Empty"
    );
    assert!(
        orch.head(RealmId::System(7)),
        "the realm stayed alive through the silence"
    );
}

#[test]
fn an_empty_realm_out_of_aoi_is_reclaimed_after_the_windows() {
    let mut orch = build();
    let c = system(7);
    orch.tick(&[]);
    orch.register_shard(RealmId::System(7), CHILD_SHARD);
    // The realm affirmatively goes empty (its own shard's real `Empty` frame, re-asserted every tick) and
    // the parent never demands it ⇒ once past the boot-dwell + teardown-cooldown + drain windows it is
    // reaped exactly once (the windows are ~80+20+10 ticks, so this needs a long run — the designed
    // conservatism: a realm survives its settle window before it can be reclaimed).
    let mut reaped = false;
    for _ in 0..200 {
        let e = demand(&orch, &c, DemandVerb::Empty);
        orch.tick(&[e]);
        if orch.rlm().teardowns_reaped == 1 {
            reaped = true;
            break;
        }
    }
    assert!(reaped, "a genuinely-empty out-of-AoI realm is reclaimed");
    assert!(
        !orch.head(RealmId::System(7)),
        "its head was revoked on the reap"
    );
}

#[test]
fn a_demand_in_the_drain_window_rescues_the_realm() {
    // EDGE #2 (BUG-C rescue): a realm goes empty and enters the drain, but a demand (a player warping in)
    // lands before the drain elapses ⇒ the teardown ABORTS; the realm is NEVER killed-then-respawned.
    let mut orch = build();
    let c = system(7);
    orch.tick(&[]);
    orch.register_shard(RealmId::System(7), CHILD_SHARD);
    // Empty for a couple ticks to open the drain, then a rescuing SpinUp every tick for a good while.
    orch.tick(&[demand(&orch, &c, DemandVerb::Empty)]);
    orch.tick(&[demand(&orch, &c, DemandVerb::Empty)]);
    for _ in 0..40 {
        let d = demand(&orch, &c, DemandVerb::SpinUp);
        orch.tick(&[d]);
    }
    assert_eq!(
        orch.rlm().teardowns_reaped,
        0,
        "the rescuing demand aborted the teardown"
    );
    assert_eq!(
        orch.rlm().spins_requested,
        0,
        "the realm was never killed, so never re-spawned"
    );
    assert!(
        orch.head(RealmId::System(7)),
        "the rescued realm is still running"
    );
}

#[test]
fn an_orchestrator_crash_reaps_nothing_on_the_first_reboot_tick() {
    // STRESS #7: kill-9 the orchestrator. The RAM demand ledger is lost but the durable realm head survives.
    // The rebuilt orchestrator (empty ledger) must NOT mass-reap the surviving realm before demands
    // re-accrue (arm B keeps a running realm alive; a fresh Empty is required to kill).
    let mut orch = build();
    orch.tick(&[]);
    orch.register_shard(RealmId::System(7), CHILD_SHARD);
    orch.tick(&[]); // commit the head durably
    assert!(orch.head(RealmId::System(7)));

    let mut reborn = orch.crash_rebuild();
    for _ in 0..50 {
        reborn.tick(&[]); // no demands yet (shards re-accrue over time)
    }
    assert_eq!(
        reborn.rlm().teardowns_reaped,
        0,
        "a rebuilt orchestrator never mass-reaps a surviving realm"
    );
    assert!(
        reborn.head(RealmId::System(7)),
        "the durable realm survived the crash"
    );
}

#[test]
fn the_reconciler_is_deterministic_across_two_runs() {
    // The SAME demand sequence through two fresh orchestrators ⇒ identical spawn/reap/gauge state.
    let run = || {
        let mut orch = build();
        let c = system(7);
        orch.tick(&[]);
        orch.tick(&[demand(&orch, &c, DemandVerb::SpinUp)]);
        orch.register_shard(RealmId::System(7), CHILD_SHARD);
        for _ in 0..10 {
            let d = demand(&orch, &c, DemandVerb::KeepAlive);
            orch.tick(&[d]);
        }
        let r = orch.rlm();
        (
            r.spins_requested,
            r.teardowns_reaped,
            r.force_reaps,
            r.running_gauge,
        )
    };
    assert_eq!(run(), run());
}
