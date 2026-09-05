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

/// `clock_peers` is EMPTY for every hand-fed cell (nothing follows this clock); the RLM 5f-3d ARMED cell
/// passes the real gateway so it receives the per-tick `ClockSync` broadcast through the production path.
fn orch_cfg(rlm: RlmTuning, clock_peers: Vec<NodeId>) -> OrchestratorConfig {
    OrchestratorConfig {
        epoch: EpochId(1),
        reserve_chunk: 1024,
        clock_peers,
        directory: DirectoryTuning::default(),
        saga: SagaTuning::default(),
        liveness: LivenessTuning::default(),
        roster: std::collections::BTreeMap::new(),
        rlm,
    }
}

fn assemble(rlm: RlmTuning, spawner: MemSpawner, store: MemStore) -> Orch {
    assemble_with_peers(rlm, spawner, store, vec![])
}

fn assemble_with_peers(
    rlm: RlmTuning,
    spawner: MemSpawner,
    store: MemStore,
    clock_peers: Vec<NodeId>,
) -> Orch {
    let mut world = World::new();
    world.insert_resource(InboundBox::default());
    world.insert_resource(OutboundBox::default());
    world.insert_resource(ClockSample::default());
    let mut schedule = Schedule::default();
    // RLM Step 4a — the spawner is INJECTED through the REAL boot path (no post-boot overwrite), so the
    // orchestrator's own `arm_quiesce`-on-recover fires: on a crash-rebuild from a non-empty store the
    // reconciler comes up with the crash-recovery freeze ALREADY armed, exactly as production would.
    register_orchestrator_with_store(
        &mut world,
        &mut schedule,
        &orch_cfg(rlm, clock_peers),
        Box::new(store.clone()),
        Box::new(spawner.clone()),
        // RLM 5e-3b: the crash-recovery launch seed — EMPTY here (the injected spawner IS the recovery
        // subject; this E2E drives the freeze-on-recover, not a launch-ledger seed). Byte-identical.
        vd_node::rlm_runtime::LaunchSeed::new(),
    );
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
        self.tick_with(demands, Vec::new());
    }

    /// [`Orch::tick`] plus `extra` raw inbound frames — the seam the RLM 5f-3d ARMED cell pumps a REAL
    /// gateway's outbox through, so the login's demand/head-read reach this orchestrator as the exact wire
    /// bytes the gateway emitted (never hand-built). Empty for every pre-5f-3d cell ⇒ behaviour-identical.
    fn tick_with(&mut self, demands: &[RealmDemand], extra: Vec<Inbound>) {
        let mut inbox: Vec<Inbound> = demands
            .iter()
            .map(|d| Inbound::Wire {
                from: STUB,
                class: MsgClass::Saga,
                bytes: postcard::to_allocvec(&InterShardFlow::RealmDemand(d.clone()))
                    .expect("encode")
                    .into(),
            })
            .collect();
        inbox.extend(extra);
        self.world.resource_mut::<InboundBox>().0 = inbox;
        self.schedule.run(&mut self.world);
    }

    /// Take everything this orchestrator's systems staged for the wire this tick (the in-process stand-in
    /// for its flush phase — the ARMED cell hands the GATEWAY-bound frames to the gateway).
    fn drain_outbox(&mut self) -> Vec<(NodeId, MsgClass, Vec<u8>)> {
        std::mem::take(&mut self.world.resource_mut::<OutboundBox>().0)
            .into_iter()
            .map(|(to, class, bytes, _)| (to, class, bytes.to_vec()))
            .collect()
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
    // RLM Step 4a — the crash-recovery freeze is ARMED on recover (before 4a it was hard-set to 0 and the
    // clause was DEAD: `arm_quiesce` had no boot caller). Resume the clock one tick so `now` is meaningful,
    // then assert teardown is frozen strictly PAST the resume point.
    reborn.tick(&[]);
    assert!(
        reborn.rlm().rlm_quiesced_until() > reborn.now(),
        "the crash-recovery freeze is armed past the resume point ({:?} > {:?})",
        reborn.rlm().rlm_quiesced_until(),
        reborn.now(),
    );
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
fn a_genesis_orchestrator_has_no_crash_freeze() {
    // The freeze exists ONLY to protect a RECOVERED realm's demand ledger while it re-accrues. A GENESIS
    // orchestrator never ran a realm, so there is nothing to freeze — `rlm_quiesced_until` stays unarmed
    // (`0`). This is the before/after contrast that pins the 4a fix: genesis == 0, recover > now.
    let orch = build();
    assert_eq!(
        orch.rlm().rlm_quiesced_until(),
        UniverseTick(0),
        "a genesis orchestrator's crash freeze is unarmed",
    );
}

#[test]
fn the_crash_freeze_bridges_an_empty_report_until_a_demand_reaccrues() {
    // THE STRAND the 4a freeze exists to close (vet wf_c09198b3). A demand-kept-alive STAGING realm (empty
    // of direct occupants, but a parent's `KeepAlive` holds it desired because an occupant's AoI reaches it)
    // survives a kill-9. Post-crash the RAM demand ledger is EMPTY. The surviving child shard keeps
    // self-reporting `Empty`; its parent's `KeepAlive` has not re-accrued yet. WITHOUT the freeze the FRESH
    // ledger cell (`spawn_watermark == 0`) bypasses `min_dwell` AND arm-B fires (`empty_confirmed`, not
    // demanded) ⇒ the realm is reaped despite a demand being milliseconds from re-landing. The freeze holds
    // teardown across that gap.
    let mut orch = build();
    let c = system(7);
    orch.tick(&[]);
    orch.tick(&[demand(&orch, &c, DemandVerb::SpinUp)]);
    orch.register_shard(RealmId::System(7), CHILD_SHARD);
    orch.tick(&[demand(&orch, &c, DemandVerb::KeepAlive)]);
    assert!(
        orch.head(RealmId::System(7)),
        "the staging realm is running"
    );

    let mut reborn = orch.crash_rebuild();
    reborn.tick(&[]); // resume the clock
    assert!(
        reborn.rlm().rlm_quiesced_until() > reborn.now(),
        "the freeze is armed",
    );
    // The child shard survived and keeps reporting Empty; NO demand re-accrues yet. Past `empty_grace` the
    // realm is confirmed-empty AND undesired — reaped in an instant if the freeze were dead.
    for _ in 0..15 {
        let empty = demand(&reborn, &c, DemandVerb::Empty);
        reborn.tick(&[empty]);
    }
    assert_eq!(
        reborn.rlm().teardowns_reaped,
        0,
        "the freeze bridges the empty report — no reap while frozen",
    );
    assert!(
        reborn.head(RealmId::System(7)),
        "the realm survives the freeze window",
    );
    // The parent's `KeepAlive` re-accrues (the occupant's AoI re-lands) — now permanently desired.
    for _ in 0..15 {
        let ka = demand(&reborn, &c, DemandVerb::KeepAlive);
        reborn.tick(&[ka]);
    }
    assert_eq!(
        reborn.rlm().teardowns_reaped,
        0,
        "a re-accrued demand keeps the realm alive past the freeze",
    );
    assert!(
        reborn.head(RealmId::System(7)),
        "the realm stays running once its demand re-accrues",
    );
}

#[test]
fn the_crash_freeze_releases_and_a_truly_abandoned_realm_is_reaped() {
    // The freeze is a TEMPORARY bridge, not a permanent no-reap. If a realm is genuinely abandoned (the warp
    // was cancelled: the child shard keeps reporting `Empty` and NO demand ever re-accrues), it MUST still be
    // reaped once the freeze releases — otherwise a crash would leak a realm shard forever (the 100K-scale
    // cost the reconciler exists to reclaim). This proves the 4a freeze RELEASES.
    let mut orch = build();
    let c = system(7);
    orch.tick(&[]);
    orch.tick(&[demand(&orch, &c, DemandVerb::SpinUp)]);
    orch.register_shard(RealmId::System(7), CHILD_SHARD);
    orch.tick(&[demand(&orch, &c, DemandVerb::KeepAlive)]);
    assert!(orch.head(RealmId::System(7)));

    let mut reborn = orch.crash_rebuild();
    // Empty-only forever, no demand ever re-accrues. Run well past the freeze (`recovery_grace` = one full
    // demand TTL) plus the two-phase drain + cooldown; the abandoned realm is reaped.
    let mut reaped_tick = None;
    for _ in 0..400 {
        let empty = demand(&reborn, &c, DemandVerb::Empty);
        reborn.tick(&[empty]);
        if reborn.rlm().teardowns_reaped >= 1 {
            reaped_tick = Some(reborn.now());
            break;
        }
    }
    assert!(
        reaped_tick.is_some(),
        "the freeze released and the abandoned realm was reaped",
    );
    assert!(
        !reborn.head(RealmId::System(7)),
        "the abandoned realm's head is revoked after the reap",
    );
    // The reap happened AFTER the freeze window (never during it).
    assert!(
        reaped_tick.expect("reaped") >= reborn.rlm().rlm_quiesced_until(),
        "the reap fired only after the freeze released",
    );
}

// ===== RLM 5f-3d — the ARMED dynamic-home LOGIN, end to end =====================================
//
// Every cell above hand-feeds the reconciler. THIS section drives it from a REAL gateway: a signed `Hello`
// → the real directory-committed `Session` lease → the gateway's SERVER-DERIVED home demand → the real
// reconciler spawn of the whole home lineage → the (modelled) child shard taking its realm head → the
// gateway's OWN `HeadRead` reply → its `AttachSession` landing on the SPAWNED node.
//
// Both nodes are 100% real code on their real per-tick schedules (`register_gateway` +
// `register_orchestrator_with_store`), talking through their real wire bytes. Only the never-built peer is
// modelled at its wire seam, exactly as the cells above model it: the spawned shard's self-grant of its
// realm head, its `Empty` self-report, and its `SessionAttached` reply.

// The cluster's node ids + auth key come from `vd_tests` (never re-invented). `SHARD` is aliased
// `LOGIN_SHARD`: it is the gateway's FROZEN `config.shard`, and it does NOT exist in this cluster — that is
// the point, since on an armed gateway nothing may ever be routed there.
use vd_tests::{AUTH_SIGNING_KEY, GATEWAY, ORCH, SHARD as LOGIN_SHARD, auth_verifying_key};

use vd_connection_plane::gateway::{
    GatewayConfig, GatewaySessions, GatewayStats, SeedInjectorConfig, TransportTuning,
    register_gateway,
};
use vd_connection_plane::tickets;
use vd_core::glam::DVec3;
use vd_core::pose::FrameRef;
use vd_core::taxonomy::ProfileKind;
use vd_core::{AccountId, EntityId, SessionId, TickId};
use vd_node::follower::register_clock_follower;
use vd_physics::worldgen::{UniverseConfig, WorldView};
use vd_sim::capability::NodeKind;
use vd_sim::runtime::NodeIdentity;
use vd_wire::channels::ClientControlMsg;
use vd_wire::seams::directory::DirectoryOp;
use vd_wire::session_flow::{GatewayToShard, ShardToGateway};
use vd_wire::version::ProtoVersion;

/// The client connection (vd-tests reserves 100+ for clients).
const CLIENT: NodeId = NodeId(100);
/// The login account, and its STORED absolute spawn pose (the 5f-3b pose-store stand-in): the Area-A box at
/// x=25, whose deepest containing realm is the 5-level [Universe, Galaxy, System(7), Planet(7), Area(7)] home.
const LOGIN_ACCOUNT: AccountId = AccountId(5);
/// THE LOGIN'S HOME REALM: the walk-scale world's deepest realm — an area inside a planet inside a star
/// system — so demanding it spins up a four-level chain, which is what this suite watches.
///
/// It used to be said as a POSITION, 25 metres along `+x` from the ambient root, and the router descended
/// the forest to work out which realm that fell in. It fell in exactly this one, at that area's own centre.
/// Naming it says what the test always meant, and takes the descent out of the test as well as the code.
const LOGIN_HOME_REALM: RealmId = RealmId::Area(7);
/// The cluster tick rate EVERY budget in this section derives from (the shipped cloud rate).
const TICK_HZ: u32 = 20;
/// The avatar entity the modelled home shard reports on attach.
const AVATAR: EntityId = EntityId(77);

/// The login's SERVER-derived home lineage (what the gateway's injector resolves), and the `RealmId` the
/// directory keys it by.
fn home_lineage() -> RealmCoord {
    // Asked of the SAME world the cluster's gateway holds, so the test and the code can never describe
    // different universes — which is exactly how this drifted the last time.
    vd_core::worldgen::coord_of_realm(
        WorldView::hand_placed(&UniverseConfig::walk_scale()).regions(),
        LOGIN_HOME_REALM,
    )
    .expect("the login's home realm belongs to the world the cluster holds")
}

/// The cluster's ONE RLM budget: the orchestrator reconciles with it AND the gateway derives its
/// demand-re-drive cadence + bootstrap TTL from it (HR3 — `resolve_rlm_tuning` is the single derivation, so
/// the gateway's hold can never be tighter than the boot the reconciler itself allows).
fn armed_tuning() -> RlmTuning {
    vd_node::rlm_runtime::resolve_rlm_tuning(true, TICK_HZ, 0, 0)
}

/// A real gateway node: its own World + schedule + local tick counter (the node shell's per-tick
/// `local_tick` advance, mirrored here since this rig has no transport).
struct Gw {
    world: World,
    schedule: Schedule,
    tick: TickId,
}

impl Gw {
    fn new(rlm: &RlmTuning) -> Gw {
        let mut world = World::new();
        world.insert_resource(InboundBox::default());
        world.insert_resource(OutboundBox::default());
        // synced = false: the REAL clock follower arms it off the orchestrator's broadcast, so the MF3
        // pre-sync hold is exercised by the boot itself rather than hand-set.
        world.insert_resource(ClockSample::default());
        world.insert_resource(NodeIdentity {
            node_id: GATEWAY,
            kind: NodeKind::Gateway,
        });
        let mut schedule = Schedule::default();
        register_clock_follower(&mut world, &mut schedule);
        // The ARMED injector — the ONLY difference from the default cluster gateway config: the
        // `GatewayConfig` FIELD SET is untouched, just its `seed_injector`. Both windows come from the SAME
        // `resolve_rlm_tuning` the orchestrator above reconciles with.
        let seed_injector = SeedInjectorConfig {
            armed: true,
            // THE HAND-PLACED WORLD — the one with a station to stand in and an area to spin up. The
            // generator emits neither (players build them), so a test that needs a deep home places
            // it — LOWERED, because the gateway only ever receives a region forest (SL4).
            world: WorldView::hand_placed(&UniverseConfig::walk_scale()).lowered(),
            // The account lives in the deep home, at that realm's own centre. A home is a NAME plus a pose
            // already measured from the named realm — there is nothing here for anybody to descend.
            homes: {
                let world = WorldView::hand_placed(&UniverseConfig::walk_scale());
                let home = vd_core::home::StoredHome::in_realm(
                    world.regions(),
                    LOGIN_HOME_REALM,
                    DVec3::ZERO,
                )
                .expect("the login's home realm belongs to the world the cluster holds");
                vd_core::home::HomeRegistry::new(home)
            },
            demand_ttl_ticks: rlm.demand_ttl_ticks,
            bootstrap_ttl_ticks: SeedInjectorConfig::bootstrap_ttl_from_rlm(
                rlm.launch_ttl_ticks,
                rlm.demand_ttl_ticks,
            ),
        };
        seed_injector
            .validate()
            .expect("the derived armed budget is valid (the bin fails loud otherwise)");
        register_gateway(
            &mut world,
            &mut schedule,
            GatewayConfig {
                orchestrator: ORCH,
                // No world booted in a fixture, so the gateway states no sky (S11).
                sky: Vec::new(),
                sky_generation: 0,
                sky_frame: None,
                shard: LOGIN_SHARD,
                known_shards: std::collections::BTreeSet::from([LOGIN_SHARD]),
                auth_verifying_key: auth_verifying_key(),
                session_seed: 23,
                tick_hz: TICK_HZ,
                trace_realm_kind: None,
                // MF2: the lease-liveness heartbeat is LIVE (a derived cadence, not a literal) — a login held
                // across a whole pod boot must keep its committed lease renewed.
                lease_renew_interval_ticks: rlm.demand_ttl_ticks
                    / SeedInjectorConfig::REDRIVE_DIVISOR,
                session_recheck_interval: 0,
                self_fence_grace_ticks: 0,
                reject_next_prepare: None,
                seed_injector,
                tuning: TransportTuning {
                    max_sessions: 4,
                    max_buffered_inputs: TransportTuning::DEFAULT_MAX_BUFFERED_INPUTS,
                },
            },
        );
        Gw {
            world,
            schedule,
            tick: TickId(0),
        }
    }

    /// ONE real gateway tick: advance `local_tick`, deliver `inbound`, run the schedule, take the outbox.
    fn step(&mut self, inbound: Vec<Inbound>) -> Vec<(NodeId, MsgClass, Vec<u8>)> {
        self.tick = self.tick.next();
        self.world.resource_mut::<ClockSample>().local_tick = self.tick;
        self.world.resource_mut::<InboundBox>().0 = inbound;
        self.schedule.run(&mut self.world);
        std::mem::take(&mut self.world.resource_mut::<OutboundBox>().0)
            .into_iter()
            .map(|(to, class, bytes, _)| (to, class, bytes.to_vec()))
            .collect()
    }

    fn sessions(&self) -> &GatewaySessions {
        self.world.resource::<GatewaySessions>()
    }

    fn stats(&self) -> &GatewayStats {
        self.world.resource::<GatewayStats>()
    }
}

/// The two-real-node cluster: a gateway and an orchestrator, one wire hop apart.
struct Armed {
    orch: Orch,
    gw: Gw,
    to_orch: Vec<Inbound>,
    to_gw: Vec<Inbound>,
    /// Everything the gateway sent SHARD-ward (never the orchestrator, never the client), in order.
    shard_bound: Vec<(NodeId, MsgClass, Vec<u8>)>,
    /// Everything the gateway sent the ORCHESTRATOR (also forwarded) — the MF2 renewal observable.
    orch_bound: Vec<Vec<u8>>,
}

impl Armed {
    fn new() -> Armed {
        let rlm = armed_tuning();
        Armed {
            orch: assemble_with_peers(
                rlm,
                MemSpawner::new(MemHub::new(), NodeId(1_000_000), 8),
                MemStore::new(),
                vec![GATEWAY],
            ),
            gw: Gw::new(&rlm),
            to_orch: Vec::new(),
            to_gw: Vec::new(),
            shard_bound: Vec::new(),
            orch_bound: Vec::new(),
        }
    }

    /// ONE cluster step: the gateway runs (seeing last step's orchestrator replies plus `client_frames`),
    /// then the orchestrator runs (seeing the gateway's frames plus the modelled child shard's `demands`).
    /// Replies land on the gateway next step — one wire hop each way.
    fn step(&mut self, client_frames: Vec<Inbound>, demands: &[RealmDemand]) {
        let mut gw_in = std::mem::take(&mut self.to_gw);
        gw_in.extend(client_frames);
        for (to, class, bytes) in self.gw.step(gw_in) {
            if to == ORCH {
                self.orch_bound.push(bytes.clone());
                self.to_orch.push(Inbound::Wire {
                    from: GATEWAY,
                    class,
                    bytes: bytes.into(),
                });
            } else if to != CLIENT {
                // SHARD-bound (the client-visible control stream is the unit cells' subject; this section's
                // is the ROUTE — WHICH node the login lands on).
                self.shard_bound.push((to, class, bytes));
            }
        }
        let inbound = std::mem::take(&mut self.to_orch);
        self.orch.tick_with(demands, inbound);
        for (to, class, bytes) in self.orch.drain_outbox() {
            if to == GATEWAY {
                self.to_gw.push(Inbound::Wire {
                    from: ORCH,
                    class,
                    bytes: bytes.into(),
                });
            }
        }
    }

    /// A real signed `Hello` from the client (the auth service's key — never a bypass).
    fn hello() -> Vec<Inbound> {
        let hello = ClientControlMsg::Hello {
            version: ProtoVersion::CURRENT,
            login: tickets::mint_login(&AUTH_SIGNING_KEY, LOGIN_ACCOUNT, EpochId(1), 1),
        };
        vec![Inbound::Wire {
            from: CLIENT,
            class: MsgClass::Control,
            bytes: postcard::to_allocvec(&hello).expect("encode").into(),
        }]
    }

    /// The modelled home shard confirming the attach (the wire frame a booted shard sends).
    fn attached(session: SessionId, node: NodeId) -> Vec<Inbound> {
        let msg = ShardToGateway::SessionAttached {
            session,
            entity: AVATAR,
            frame: FrameRef::SystemSpace { system_seed: 7 },
            realm_fence: Fence(1),
        };
        vec![Inbound::Wire {
            from: node,
            class: MsgClass::Control,
            bytes: postcard::to_allocvec(&msg).expect("encode").into(),
        }]
    }

    fn session(&self) -> SessionId {
        self.gw
            .sessions()
            .sessions()
            .next()
            .expect("the login minted a session")
    }

    /// The node the reconciler LAUNCHED for the home leaf: the only live pod with the `Area` profile (its
    /// four ancestors launch as Galaxy/System/Planet profiles), so this never guesses a mint order.
    fn home_node(&self) -> Option<NodeId> {
        let area: Vec<NodeId> = self
            .orch
            .spawner
            .live()
            .into_iter()
            .filter(|(_, (kind, _))| *kind == ProfileKind::Area)
            .map(|(node, _)| node)
            .collect();
        assert!(area.len() <= 1, "one Area realm ⇒ at most one Area pod");
        area.first().copied()
    }

    /// How many `AttachSession` frames for `session` the gateway sent to `node`.
    fn attaches_to(&self, node: NodeId, session: SessionId) -> usize {
        self.shard_bound
            .iter()
            .filter(|(to, class, bytes)| {
                (*to == node)
                    & (*class == MsgClass::Control)
                    & matches!(
                        postcard::from_bytes::<GatewayToShard>(bytes),
                        Ok(GatewayToShard::AttachSession { session: s, .. }) if s == session
                    )
            })
            .count()
    }

    /// How many `LeaseRenew`s for this session's key the gateway sent the orchestrator (MF2).
    fn session_renewals(&self, session: SessionId) -> usize {
        self.orch_bound
            .iter()
            .filter(|bytes| {
                matches!(
                    postcard::from_bytes::<InterShardFlow>(bytes),
                    Ok(InterShardFlow::Directory(DirectoryOp::LeaseRenew {
                        key: DirectoryKey::Session(s),
                        ..
                    })) if s == session
                )
            })
            .count()
    }

    /// Drive the cluster until `home_shard_of` resolves (feeding nothing), at most `max` steps.
    fn run_until_home_resolved(&mut self, session: SessionId, max: usize) {
        for _ in 0..max {
            if self.gw.sessions().home_shard_of(session).is_some() {
                return;
            }
            self.step(vec![], &[]);
        }
    }
}

#[test]
fn an_armed_login_spins_its_home_realm_and_attaches_on_the_spawned_node() {
    // RLM 5f-3d, end to end. NOTHING is hand-fed to the reconciler here: the login itself produces the
    // demand, and the gateway routes to whatever node the directory says owns the spawned realm.
    let mut cluster = Armed::new();
    let lineage = home_lineage();
    let home_rid = lineage.lowered();
    // Boot: two ticks so the gateway's REAL follower sees the orchestrator's `ClockSync` broadcast.
    cluster.step(vec![], &[]);
    cluster.step(vec![], &[]);
    assert!(
        cluster.gw.world.resource::<ClockSample>().synced,
        "the gateway followed the cluster clock through the production path"
    );
    // The login: Hello → LeaseGrant → the real committed lease → the SERVER-derived home demand.
    cluster.step(Armed::hello(), &[]);
    let sid = cluster.session();
    for _ in 0..3 {
        cluster.step(vec![], &[]);
    }
    assert_eq!(
        cluster.gw.sessions().home_realm_of(sid),
        Some(home_rid),
        "the gateway derived this account's home realm from its STORED pose"
    );
    assert_eq!(
        cluster.gw.stats().logins_held_pre_sync,
        0,
        "the clock was synced before the lease committed, so nothing was held (MF3)"
    );
    assert_eq!(
        cluster.orch.rlm().spins_requested,
        5,
        "the login's ONE leaf demand spun up the WHOLE 5-level home lineage (the 5f-3a ride)"
    );
    let home = cluster.home_node().expect("the home realm's pod launched");
    assert_eq!(
        cluster.gw.sessions().home_shard_of(sid),
        None,
        "…and nothing is routable yet: the pod is still booting"
    );
    assert_eq!(
        cluster.attaches_to(LOGIN_SHARD, sid),
        0,
        "the held login is NEVER attached to the static login shard"
    );
    // The spawned shard boots and takes its realm lease (modelled — the real launch is Step 5).
    cluster.orch.register_shard(home_rid, home);
    // The gateway's own re-driven `HeadRead` now names it.
    cluster.run_until_home_resolved(sid, 200);
    assert_eq!(
        cluster.gw.sessions().home_shard_of(sid),
        Some(home),
        "the gateway routed the session to the SPAWNED node the directory named"
    );
    assert!(
        cluster.attaches_to(home, sid) >= 1,
        "and sent its AttachSession THERE"
    );
    assert_eq!(
        cluster.attaches_to(LOGIN_SHARD, sid),
        0,
        "never to config.shard — the whole point of the dynamic route"
    );
    // The spawned shard confirms: the session goes Active on ITS attach, off the RUNTIME routable roster.
    cluster.step(Armed::attached(sid, home), &[]);
    assert_eq!(
        cluster.gw.sessions().entity_of(sid),
        Some(AVATAR),
        "the login completed on the demand-spawned home shard"
    );
    assert_eq!(cluster.gw.stats().home_bootstrap_timeouts, 0);
    assert_eq!(cluster.gw.stats().home_wait_desync, 0);
    assert_eq!(
        cluster.orch.rlm().teardowns_reaped,
        0,
        "and nothing was reaped along the way"
    );
}

#[test]
fn the_login_re_seed_keeps_the_reconciler_from_reaping_the_home_it_is_attaching_to() {
    // MF1 DEFECT A, END TO END — the load-bearing invariant of this whole slice, against the REAL
    // reconciler. The home realm boots and takes its lease, but its `SessionAttached` never comes back (the
    // 5f-4 dial-a-fresh-pod race), and being unoccupied it self-reports `Empty` every tick — so the
    // reconciler's arm-B (`running_live & !empty_confirmed`) is FALSE. The ONLY thing keeping the realm
    // desired is the gateway's re-seeded demand. Before MF1 the re-seed stopped at the head resolve, arm-A
    // lapsed one `demand_ttl` later, and the reconciler KILLED the realm the login was attaching to — tens of
    // ticks BEFORE the gateway's bounded bootstrap TTL would have noticed anything.
    let rlm = armed_tuning();
    let mut cluster = Armed::new();
    let lineage = home_lineage();
    let home_rid = lineage.lowered();
    cluster.step(vec![], &[]);
    cluster.step(vec![], &[]);
    cluster.step(Armed::hello(), &[]);
    let sid = cluster.session();
    for _ in 0..3 {
        cluster.step(vec![], &[]);
    }
    let login_tick = cluster.gw.tick.0;
    let home = cluster.home_node().expect("the home realm's pod launched");
    cluster.orch.register_shard(home_rid, home);
    cluster.run_until_home_resolved(sid, 200);
    assert_eq!(cluster.gw.sessions().home_shard_of(sid), Some(home));
    // HOLD past the point the reconciler WOULD have reaped: one whole demand TTL after the last demand the
    // pre-MF1 re-drive would have sent, plus the drain + cooldown windows a Kill rides.
    let hold_until =
        login_tick + rlm.demand_ttl_ticks + rlm.teardown_drain_ticks + rlm.teardown_cooldown_ticks;
    // …and the hold must stay INSIDE the gateway's bounded bootstrap TTL, so a Close can never be mistaken
    // for a reap. `login_tick` is measured a few ticks AFTER the lease committed, so this is the pessimistic
    // side of the real deadline; the empirical proof is `home_bootstrap_timeouts == 0` below.
    let bootstrap_deadline = login_tick
        + SeedInjectorConfig::bootstrap_ttl_from_rlm(rlm.launch_ttl_ticks, rlm.demand_ttl_ticks);
    assert!(
        hold_until < bootstrap_deadline,
        "the reap window ({hold_until}) must fall strictly inside the bootstrap window \
         ({bootstrap_deadline})"
    );
    while cluster.gw.tick.0 < hold_until {
        // The booted-but-unoccupied shard's own `Empty` self-report, every tick.
        let empty = demand(&cluster.orch, &lineage, DemandVerb::Empty);
        cluster.step(vec![], &[empty]);
        assert_eq!(
            cluster.orch.rlm().teardowns_reaped,
            0,
            "the realm this login is attaching to must NEVER be reaped mid-bootstrap (MF1-A)"
        );
    }
    assert_eq!(
        cluster.orch.rlm().force_reaps,
        0,
        "and no zombie sweep either"
    );
    assert!(
        cluster.orch.head(home_rid),
        "its realm lease is still recorded"
    );
    assert!(
        cluster.orch.spawner.live_nodes().contains(&home),
        "and its pod is still alive"
    );
    assert_eq!(
        cluster.gw.stats().home_bootstrap_timeouts,
        0,
        "the login is still held, still inside its bounded window"
    );
    assert!(
        cluster.session_renewals(sid) >= 1,
        "and its COMMITTED lease was renewed while pre-Active (MF2)"
    );
    // The lost attach finally lands: the login completes on the realm that was never reaped.
    cluster.step(Armed::attached(sid, home), &[]);
    assert_eq!(
        cluster.gw.sessions().entity_of(sid),
        Some(AVATAR),
        "the session reached Active on its ORIGINAL home shard — nothing was killed under it"
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
