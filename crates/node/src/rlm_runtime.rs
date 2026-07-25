//! RLM Step 3 — the orchestrator-side RUNTIME for the realm-lifecycle reconciler (spec
//! `scripts/rlm_step3_reconciler_spec.md` §2.4/§3.3). The PURE decision kernel lives in
//! [`vd_sim::rlm`]; this module is the Bevy glue that (3c) ingests re-asserted `RealmDemand`s into the
//! kernel's ledger, and (3d) EXECUTES the kernel's decisions against a [`RealmSpawner`] — the mem twin
//! in-process, a real cluster launcher in production. The DECIDE (pure, `vd_sim::rlm::reconcile`) and
//! EXECUTE (this file's [`RlmReconcilerRes::drive`]) halves are separate so the identical decision logic
//! drives both the deterministic harness and the async cluster.
//!
//! INERT by default: with `RlmTuning::reconcile_interval_ticks == 0` the reconcile sweep never runs, so an
//! orchestrator that does not opt in is byte-identical (the walk/canonical gate).

use std::collections::BTreeMap;

use bevy_ecs::prelude::{Res, ResMut, Resource};
use vd_sim::directory::{DirectoryCore, RevokeOutcome};
use vd_sim::io::{Inbound, MsgClass, RealmSpawner};
use vd_sim::rlm::{
    DemandLedger, LifecycleAction, RlmTuning, desired_alive, reconcile, running_live,
};
use vd_sim::runtime::InboundBox;
use vd_wire::intershard::InterShardFlow;
use vd_wire::seams::directory::DirectoryKey;

use vd_core::realm_coord::RealmCoord;
use vd_core::realm_path::RealmPath;
use vd_core::{Fence, NodeId, UniverseTick};

/// Exponential-backoff cap: a launch-failure streak past this stops widening the retry window (`2^10 ×`
/// the spin-up cooldown) — the retry never hard-stops (a transient cluster-full self-heals) but never
/// hammers a permanently-unschedulable realm either.
const BACKOFF_CAP: u32 = 10;

/// The launch-set shape (RLM Step 5e): `path → { minted node → mint tick }` — both [`LaunchLedger::minted`]
/// and the crash-recovery seed a rebuilt orchestrator feeds [`RlmReconcilerRes::with_launch_seed`] (the
/// spawner's `launch_ledger_seed()` projection). ONE name for the shape (DRY), keyed by the lineage
/// [`RealmPath`] so it is galaxy-unique + deterministic.
pub type LaunchSeed = BTreeMap<RealmPath, BTreeMap<NodeId, UniverseTick>>;

/// The EXECUTE-side launch bookkeeping (BUG-B): every node the reconciler minted per realm coord, so a
/// re-issue never orphans a prior node, plus the launch-failure backoff streak. Keyed by the lineage
/// [`RealmPath`] (`Ord`, deterministic). Drained by the launch-reconcile step when a head appears or a
/// minted node ages out of the spawner's live set (slice 3d).
#[derive(Default, Debug)]
pub struct LaunchLedger {
    /// `path → { minted node → mint tick }` — the SET of live launches per coord (never one overwriting
    /// ticket, so re-issuing a spawn cannot lose the prior node).
    pub minted: BTreeMap<RealmPath, BTreeMap<NodeId, UniverseTick>>,
    /// `path → consecutive launch failures` — the exponential-backoff streak (slice 3d).
    pub fail_streak: BTreeMap<RealmPath, u32>,
}

impl LaunchLedger {
    /// The live-launch view the kernel's [`vd_sim::rlm::reconcile`] reads as the "launching" fact (BUG-B):
    /// `path → { node → mint tick }`. Borrowed directly — it IS the source of truth for minted-not-leased.
    #[must_use]
    pub fn live(&self) -> &BTreeMap<RealmPath, BTreeMap<NodeId, UniverseTick>> {
        &self.minted
    }
}

/// The orchestrator's realm-lifecycle reconciler state (RLM Step 3). Holds the RAM demand ledger, the
/// launch bookkeeping, the timing budget, the [`RealmSpawner`] EXECUTE seam (`Box<dyn>` mirroring
/// `StoreRes`), the crash-quiesce watermark, and honesty counters surfaced in the admin snapshot.
#[derive(Resource)]
pub struct RlmReconcilerRes {
    ledger: DemandLedger,
    launches: LaunchLedger,
    tuning: RlmTuning,
    /// THE execute seam: the mem twin in-process, a real k8s launcher in prod. Object-safe (no
    /// per-monomorphization coverage debt — `io/mod.rs:444`), exactly like `StoreRes`.
    spawner: Box<dyn RealmSpawner + Send + Sync>,
    /// Crash-recovery freeze: teardown is blocked until `now >= rlm_quiesced_until` so a rebuilt
    /// orchestrator (empty ledger) never mass-reaps before demands re-accrue (mirror
    /// `liveness_quiesced_until`).
    rlm_quiesced_until: UniverseTick,

    // ---- honesty counters (0 on a quiescent run; surfaced in the admin snapshot) ----
    /// `SpinUp` intents executed.
    pub spins_requested: u64,
    /// `spawn_realm` refusals (drives the backoff + the stuck alarm).
    pub spins_failed: u64,
    /// `Kill`s executed (a reclaimed realm).
    pub teardowns_reaped: u64,
    /// `ForceReap`s executed (a zombie cleaned — BUG-A).
    pub force_reaps: u64,
    /// Saga-class frames that did not decode to a `RealmDemand` payload (ROB honesty; 0 in a healthy run).
    pub undecodable_demands: u64,
    /// Last sweep's desired-realm count (gauge).
    pub desired_gauge: u64,
    /// Last sweep's running-realm count (gauge).
    pub running_gauge: u64,
}

impl RlmReconcilerRes {
    /// Build the reconciler around a validated tuning + an execute-seam spawner. `rlm_quiesced_until` is
    /// `0` (no freeze) at a cold boot; the crash-restart path re-arms it (slice 3e). Genesis/inert boot →
    /// an EMPTY launch seed (byte-identical); the crash-restart path uses [`with_launch_seed`].
    #[must_use]
    pub fn new(
        tuning: RlmTuning,
        spawner: Box<dyn RealmSpawner + Send + Sync>,
    ) -> RlmReconcilerRes {
        RlmReconcilerRes::with_launch_seed(tuning, spawner, BTreeMap::new())
    }

    /// Build the reconciler with a PRE-SEEDED launch ledger (RLM Step 5e crash-recovery entry point).
    /// `launch_seed` (`path → {node → mint tick}`) is the spawner's `launch_ledger_seed()` projection of the
    /// shards a rebuilt orchestrator recovered from its durable launch ledger — so the FIRST reconcile
    /// sweep sees them as already-launched (`launch_present`) and does NOT re-spawn a survivor. It is the
    /// crash-recovery half of the double-spawn guard: the RAM-only demand ledger comes up EMPTY, so this
    /// seed is the only thing suppressing a spurious re-spawn until demands re-accrue (bridged meanwhile by
    /// the Step-4a quiesce freeze). SUPERSET of the pre-crash `minted` (vet D5) — a head-up survivor whose
    /// `minted[path]` was empty pre-crash is re-seeded here, and drained on sweep-1 once the head is seen.
    #[must_use]
    pub fn with_launch_seed(
        tuning: RlmTuning,
        spawner: Box<dyn RealmSpawner + Send + Sync>,
        launch_seed: LaunchSeed,
    ) -> RlmReconcilerRes {
        RlmReconcilerRes {
            ledger: DemandLedger::default(),
            launches: LaunchLedger {
                minted: launch_seed,
                fail_streak: BTreeMap::new(),
            },
            tuning,
            spawner,
            rlm_quiesced_until: UniverseTick(0),
            spins_requested: 0,
            spins_failed: 0,
            teardowns_reaped: 0,
            force_reaps: 0,
            undecodable_demands: 0,
            desired_gauge: 0,
            running_gauge: 0,
        }
    }

    /// Read the demand ledger (the reconcile system + tests).
    #[must_use]
    pub fn ledger(&self) -> &DemandLedger {
        &self.ledger
    }

    /// This reconciler's timing budget.
    #[must_use]
    pub fn tuning(&self) -> &RlmTuning {
        &self.tuning
    }

    /// Set the crash-recovery quiesce watermark (slice 3e boot path): teardown is blocked until `now`
    /// reaches it, so a rebuilt orchestrator (empty ledger) never mass-reaps before demands re-accrue.
    pub fn arm_quiesce(&mut self, until: UniverseTick) {
        self.rlm_quiesced_until = until;
    }

    /// The crash-recovery freeze watermark: teardown is blocked while `now < rlm_quiesced_until`. `0` when
    /// unarmed (genesis boot / inert). Read by the boot path's post-recover assertion + the E2E freeze proof.
    #[must_use]
    pub fn rlm_quiesced_until(&self) -> UniverseTick {
        self.rlm_quiesced_until
    }

    /// Slice 3d — the ONLY impure step: run the pure kernel decision, apply its ledger delta, then EXECUTE
    /// each action against the [`RealmSpawner`] (spawn / kill / force-reap), reconcile the launch
    /// bookkeeping, and refresh the gauges. `dir` is the live directory (read for heads, staged for revoke
    /// — the caller commits it in the group-commit barrier, slice 3e); `liveness_dead` is the
    /// orchestrator's `is_latched_dead`. Deterministic given identical inputs (the pure kernel decides;
    /// this only executes).
    pub fn reconcile_and_drive(
        &mut self,
        dir: &mut DirectoryCore,
        liveness_dead: &dyn Fn(NodeId) -> bool,
        now: UniverseTick,
    ) {
        let (actions, delta) = reconcile(
            &self.ledger,
            dir,
            liveness_dead,
            self.launches.live(),
            &self.tuning,
            now,
            self.rlm_quiesced_until,
        );
        self.ledger.apply_delta(&delta);
        for action in actions {
            match action {
                LifecycleAction::SpinUp { coord } => self.exec_spinup(&coord, now),
                LifecycleAction::Kill { path, node, fence } => {
                    self.exec_kill(dir, &path, node, fence, now);
                }
                LifecycleAction::ForceReap { path, node, fence } => {
                    self.exec_force_reap(dir, &path, node, fence);
                }
            }
        }
        self.reconcile_launches(dir, now);
        self.update_gauges(dir, liveness_dead, now);
    }

    /// Execute a `SpinUp` — with an exponential backoff on the per-coord failure streak so a transient
    /// cluster-full self-heals when capacity returns but a permanently-unschedulable realm never hammers
    /// the launcher. The attempt (success OR failure) anchors `spawn_watermark` (min-dwell + backoff clock).
    fn exec_spinup(&mut self, coord: &RealmCoord, now: UniverseTick) {
        let path = coord.path().clone();
        let streak = self.launches.fail_streak.get(&path).copied().unwrap_or(0);
        if streak > 0 {
            let backoff = self
                .tuning
                .spinup_cooldown_ticks
                .saturating_mul(1u64 << streak.min(BACKOFF_CAP));
            let since = self
                .ledger
                .get(&path)
                .map_or(u64::MAX, |c| now.0.saturating_sub(c.spawn_watermark.0));
            if since < backoff {
                return; // still backing off after a launch failure — re-derived next sweep
            }
        }
        self.spins_requested += 1;
        self.ledger.mark_spawned(coord, now);
        match self.spawner.spawn_realm(coord, now) {
            Ok(node) => {
                self.launches
                    .minted
                    .entry(path.clone())
                    .or_default()
                    .insert(node, now);
                self.launches.fail_streak.remove(&path);
            }
            Err(_) => {
                self.spins_failed += 1;
                *self.launches.fail_streak.entry(path).or_default() += 1;
            }
        }
    }

    /// Execute a `Kill` — tear down the pod (idempotent: any `SpawnError` converges) then revoke the head.
    /// A REFUSED revoke (a fence race — a newer owner grabbed the realm) leaves the head + cell so the next
    /// sweep re-decides rather than desyncing (critique H-1).
    fn exec_kill(
        &mut self,
        dir: &mut DirectoryCore,
        path: &RealmPath,
        node: NodeId,
        fence: Fence,
        now: UniverseTick,
    ) {
        let _ = self.spawner.kill_realm(node);
        if apply_revoke(dir, path, fence) {
            self.ledger.mark_reaped(path, now);
            self.launches.minted.remove(path);
            self.teardowns_reaped += 1;
        }
    }

    /// Execute a `ForceReap` (BUG-A zombie) — kill the corpse (idempotent) + force-revoke the stale head so
    /// it stops reading "running". The ledger cell is kept: if still demanded the next sweep re-spawns a
    /// FRESH incarnation (monotone ids never resurrect the dead one); else it retires.
    fn exec_force_reap(
        &mut self,
        dir: &mut DirectoryCore,
        path: &RealmPath,
        node: NodeId,
        fence: Fence,
    ) {
        self.force_reaps += 1;
        let _ = self.spawner.kill_realm(node);
        let _ = apply_revoke(dir, path, fence);
        self.launches.minted.remove(path);
    }

    /// Drain the launch bookkeeping (BUG-B). A minted node is kept ONLY while its realm has no head yet AND
    /// it is EITHER still live per the spawner OR within the launch-TTL grace (a real pod may still be
    /// registering — the mem twin registers instantly, but a k8s pod takes a moment). It is dropped once
    /// the head appears (launch succeeded) or it has neither shown up nor stayed live past the TTL (a
    /// silent async failure) — so a `SpinUp` re-fires for a genuinely-dead launch but never double-spawns a
    /// still-pending one. Bitwise operators (no short-circuit false-arm, HR5).
    fn reconcile_launches(&mut self, dir: &DirectoryCore, now: UniverseTick) {
        let live = self.spawner.live_nodes();
        let ttl = self.tuning.launch_ttl_ticks;
        for (path, nodes) in &mut self.launches.minted {
            let rid = path
                .realm_id()
                .expect("a lifecycle realm path is non-empty");
            let head_up = dir.head(DirectoryKey::Realm(rid)).is_some();
            nodes.retain(|node, minted_at| {
                let within_ttl = now.0.saturating_sub(minted_at.0) < ttl;
                !head_up & (live.contains(node) | within_ttl)
            });
        }
        self.launches.minted.retain(|_, nodes| !nodes.is_empty());
    }

    /// Refresh the observability gauges (desired + running realm counts among tracked cells) using the
    /// public kernel predicates — not a second decision. Branchless (`u64::from(bool)`, HR5).
    fn update_gauges(
        &mut self,
        dir: &DirectoryCore,
        liveness_dead: &dyn Fn(NodeId) -> bool,
        now: UniverseTick,
    ) {
        let mut desired = 0u64;
        let mut running = 0u64;
        for (_path, cell) in self.ledger.iter() {
            let head = dir.head(DirectoryKey::Realm(cell.coord.lowered()));
            let rl = running_live(head, liveness_dead);
            running += u64::from(rl);
            desired += u64::from(desired_alive(cell, now, &self.tuning, rl));
        }
        self.desired_gauge = desired;
        self.running_gauge = running;
    }
}

/// Revoke a realm head, returning `true` iff the head is now gone (revoked or already absent) and `false`
/// on a REFUSED revoke (a stale fence / a concurrent re-home lock — leave the head, retry next sweep). All
/// three [`RevokeOutcome`] arms are handled exhaustively (critique H-1: a refused revoke self-heals rather
/// than desyncing).
fn apply_revoke(dir: &mut DirectoryCore, path: &RealmPath, fence: Fence) -> bool {
    let rid = path
        .realm_id()
        .expect("a lifecycle realm path is non-empty");
    match dir.revoke(DirectoryKey::Realm(rid), fence) {
        RevokeOutcome::Revoked | RevokeOutcome::UnknownKey => true,
        RevokeOutcome::Refused { .. } => false,
    }
}

/// Slice 3c — INGEST: fold every re-asserted `RealmDemand` on the `Saga` class into the kernel ledger
/// (the SOURCE-AGNOSTIC ingress — an AoI demand and a player-spawn demand are indistinguishable). Mirrors
/// `serve_directory`'s decode; a Saga frame that is not a `RealmDemand` is skipped (it is another
/// service's — `Directory`/`SagaAck`), and an undecodable one is counted (honesty, parity with
/// `orchestrator.rs`).
pub fn record_realm_demands(inbox: Res<InboundBox>, mut rlm: ResMut<RlmReconcilerRes>) {
    for msg in &inbox.0 {
        let Inbound::Wire { class, bytes, .. } = msg else {
            continue;
        };
        if *class != MsgClass::Saga {
            continue;
        }
        let flow = match postcard::from_bytes::<InterShardFlow>(bytes) {
            Ok(flow) => flow,
            Err(_) => {
                rlm.undecodable_demands += 1;
                tracing::error!("undecodable saga-class message (RLM demand ingest)");
                continue;
            }
        };
        let InterShardFlow::RealmDemand(demand) = flow else {
            continue;
        };
        rlm.ledger.record_demand(
            &demand.child,
            demand.verb,
            demand.universe_tick,
            demand.parent_fence,
        );
    }
}

/// Slice 3e — the orchestrator's realm-lifecycle reconcile SWEEP, chained BETWEEN `drive_sagas_core` (the
/// reaper + rehome-arm) and `commit_barrier` (the ONE fsync). Reads this tick's post-CAS directory heads +
/// the D-3 liveness latch, runs the pure kernel decision, and executes it — staging any grant/revoke into
/// the SAME `dir.dirty` set the barrier drains (persist-before-effect). INERT by default
/// (`reconcile_interval_ticks == 0` ⇒ returns immediately ⇒ byte-identical to a build without RLM); when
/// active it sweeps every `reconcile_interval_ticks` universe ticks (the scale cadence — a knob to bound
/// the orchestrator's per-tick reconcile cost under heavy realm churn).
pub fn reconcile_realm_lifecycle(
    clock: Res<vd_sim::runtime::ClockSample>,
    mut dir: ResMut<crate::orchestrator::DirectoryRes>,
    runtime: Res<crate::saga_runtime::SagaRuntimeRes>,
    mut rlm: ResMut<RlmReconcilerRes>,
) {
    let interval = rlm.tuning().reconcile_interval_ticks;
    if interval == 0 {
        return; // INERT: the byte-identical default (no reconciler wired).
    }
    let now = clock.universe_tick;
    if !now.0.is_multiple_of(interval) {
        return; // scale cadence: sweep only every `interval` ticks.
    }
    let dead = |node: NodeId| runtime.is_node_latched_dead(node);
    rlm.reconcile_and_drive(&mut dir.0, &dead, now);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::prelude::{Schedule, World};
    use vd_sim::io::mem::{MemHub, MemSpawner};
    use vd_wire::intershard::{DemandVerb, RealmDemand};

    use vd_core::realm_coord::RealmCoord;
    use vd_core::realm_path::{RealmKindTag, RealmLevel};
    use vd_core::{Fence, MsgId};

    fn sys(seed: u64) -> RealmCoord {
        RealmCoord::from_path(RealmPath::from_levels(vec![RealmLevel::new(
            RealmKindTag::System,
            seed,
        )]))
        .expect("one-level path has a leaf")
    }

    fn spawner() -> Box<dyn RealmSpawner + Send + Sync> {
        Box::new(MemSpawner::new(MemHub::new(), NodeId(1000), 8))
    }

    fn demand_frame(coord: &RealmCoord, verb: DemandVerb, tick: u64, fence: u64) -> Inbound {
        let bytes = postcard::to_allocvec(&InterShardFlow::RealmDemand(RealmDemand {
            child: coord.clone(),
            parent_fence: Fence(fence),
            verb,
            universe_tick: UniverseTick(tick),
        }))
        .expect("encode");
        Inbound::Wire {
            from: NodeId(7),
            class: MsgClass::Saga,
            bytes: bytes.into(),
        }
    }

    fn run(inbound: Vec<Inbound>) -> RlmReconcilerRes {
        let mut world = World::new();
        world.insert_resource(InboundBox(inbound));
        world.insert_resource(RlmReconcilerRes::new(RlmTuning::cloud(20), spawner()));
        let mut schedule = Schedule::default();
        schedule.add_systems(record_realm_demands);
        schedule.run(&mut world);
        world.remove_resource::<RlmReconcilerRes>().expect("res")
    }

    #[test]
    fn ingest_folds_each_verb_into_the_ledger() {
        let rlm = run(vec![
            demand_frame(&sys(7), DemandVerb::SpinUp, 100, 1),
            demand_frame(&sys(8), DemandVerb::Empty, 100, 1),
            demand_frame(&sys(7), DemandVerb::KeepAlive, 104, 1),
        ]);
        // sys(7): SpinUp then KeepAlive → last_demand_tick advanced.
        assert_eq!(
            rlm.ledger()
                .get(sys(7).path())
                .expect("cell present")
                .last_demand_tick,
            UniverseTick(104)
        );
        // sys(8): Empty → last_empty latched, never demanded.
        assert_eq!(
            rlm.ledger()
                .get(sys(8).path())
                .expect("cell present")
                .last_empty_tick,
            Some(UniverseTick(100))
        );
        assert_eq!(rlm.ledger().len(), 2);
        assert_eq!(rlm.undecodable_demands, 0);
    }

    #[test]
    fn ingest_counts_an_undecodable_saga_frame() {
        let garbage = Inbound::Wire {
            from: NodeId(7),
            class: MsgClass::Saga,
            bytes: vec![0xFF, 0x00].into(),
        };
        let rlm = run(vec![garbage]);
        assert_eq!(rlm.undecodable_demands, 1);
        assert!(rlm.ledger().is_empty());
    }

    #[test]
    fn ingest_skips_wrong_class_and_non_demand_saga_and_non_wire() {
        // A wrong-class frame (Snapshot), a Saga frame that is a Directory op (another service's), and a
        // non-Wire inbound are ALL skipped — none touches the ledger or the counter.
        let directory_saga = {
            let bytes = postcard::to_allocvec(&InterShardFlow::Directory(
                vd_wire::seams::directory::DirectoryOp::HeadRead {
                    key: vd_wire::seams::directory::DirectoryKey::Realm(
                        vd_core::pose::RealmId::System(7),
                    ),
                },
            ))
            .expect("encode");
            Inbound::Wire {
                from: NodeId(7),
                class: MsgClass::Saga,
                bytes: bytes.into(),
            }
        };
        let wrong_class = Inbound::Wire {
            from: NodeId(7),
            class: MsgClass::Snapshot,
            bytes: vec![0x00].into(),
        };
        let non_wire = Inbound::NodeUnreachable {
            to: NodeId(9),
            class: MsgClass::Saga,
            undelivered: MsgId(0),
        };
        let rlm = run(vec![directory_saga, wrong_class, non_wire]);
        assert!(rlm.ledger().is_empty());
        assert_eq!(
            rlm.undecodable_demands, 0,
            "a valid non-demand frame is not undecodable"
        );
    }

    #[test]
    fn reconciler_res_new_is_inert_gauges_zero() {
        let rlm = RlmReconcilerRes::new(RlmTuning::default(), spawner());
        assert_eq!(rlm.tuning().reconcile_interval_ticks, 0);
        assert_eq!(rlm.spins_requested, 0);
        assert_eq!(rlm.desired_gauge, 0);
        assert!(rlm.ledger().is_empty());
        assert!(rlm.launches.live().is_empty());
    }

    // ---- slice 3d — reconcile_and_drive (execute) -----------------------------------------------

    use vd_sim::directory::{DirectoryCore, DirectoryTuning};
    use vd_sim::io::SpawnError;
    use vd_sim::rlm::LedgerDelta;
    use vd_wire::seams::directory::{AuthorityRef, DirectoryKey};

    use vd_core::pose::RealmId;

    fn dir() -> DirectoryCore {
        DirectoryCore::new(DirectoryTuning::default())
    }

    fn grant(dir: &mut DirectoryCore, rid: RealmId, node: u64, fence: u64) {
        dir.grant(
            DirectoryKey::Realm(rid),
            AuthorityRef::Shard(NodeId(node)),
            Fence(fence),
            UniverseTick(0),
        );
    }

    fn head_present(dir: &DirectoryCore, rid: RealmId) -> bool {
        dir.head(DirectoryKey::Realm(rid)).is_some()
    }

    /// A spawner that always refuses — for the launch-failure backoff test.
    struct FailingSpawner;
    impl RealmSpawner for FailingSpawner {
        fn spawn_realm(&self, _c: &RealmCoord, _t: UniverseTick) -> Result<NodeId, SpawnError> {
            Err(SpawnError::UnknownNode(NodeId(0)))
        }
        fn kill_realm(&self, _n: NodeId) -> Result<(), SpawnError> {
            Ok(())
        }
        fn live_nodes(&self) -> std::collections::BTreeSet<NodeId> {
            std::collections::BTreeSet::new()
        }
    }

    /// A spawner that reports a spawn Ok but whose node NEVER shows up live (a silent async failure — a
    /// crash-looped pod). Models the BUG-B "spawned but vanished" case the mem twin can't produce.
    struct GhostSpawner;
    impl RealmSpawner for GhostSpawner {
        fn spawn_realm(&self, _c: &RealmCoord, _t: UniverseTick) -> Result<NodeId, SpawnError> {
            Ok(NodeId(999))
        }
        fn kill_realm(&self, _n: NodeId) -> Result<(), SpawnError> {
            Ok(())
        }
        fn live_nodes(&self) -> std::collections::BTreeSet<NodeId> {
            std::collections::BTreeSet::new()
        }
    }

    #[test]
    fn drive_spawns_a_demanded_realm_once_then_holds_while_launching() {
        let mut rlm = RlmReconcilerRes::new(RlmTuning::cloud(20), spawner());
        let mut d = dir();
        rlm.ledger
            .record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(100));
        assert_eq!(rlm.spins_requested, 1);
        assert_eq!(
            rlm.launches.minted.get(sys(7).path()).map(BTreeMap::len),
            Some(1)
        );
        // Next sweep: launching (minted, no head) ⇒ NO second spawn (BUG-B).
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(101));
        assert_eq!(rlm.spins_requested, 1, "no double-spawn while launching");
        // RLM 5d C-1 idempotency-by-coord.path: a genuinely RE-ISSUED SpinUp on the still-launching coord
        // is ALSO a no-op. The suppression is keyed on the minted node under `sys(7).path()` (the lineage
        // path), NOT on the incarnation — the allocator WOULD mint a fresh NodeId/cookie, but `launch_present`
        // for the path blocks the emit. So a re-issued demand can never orphan a second incarnation.
        rlm.ledger
            .record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(102), Fence(2));
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(102));
        assert_eq!(
            rlm.spins_requested, 1,
            "a re-issued SpinUp on a live/launching coord is idempotent (path-keyed, not incarnation-keyed)"
        );
        assert_eq!(
            rlm.launches.minted.get(sys(7).path()).map(BTreeMap::len),
            Some(1),
            "still exactly one minted node for the path — no second incarnation"
        );
    }

    #[test]
    fn a_recovered_launch_seed_suppresses_the_respawn_on_the_first_sweep() {
        // RLM 5e-3b (D10): the crash-recovery half of idempotency-by-coord.path. A rebuilt orchestrator
        // seeded (from the durable launch ledger) with a survivor for `sys(7)` does NOT re-spawn it on the
        // first sweep even under a re-asserted SpinUp — `launch_present` is true from the SEED (read by the
        // pure kernel before any drain). The empty-seed control arm proves the seed is the differentiator.
        let seed: LaunchSeed = [(
            sys(7).path().clone(),
            [(NodeId(50), UniverseTick(90))].into_iter().collect(),
        )]
        .into_iter()
        .collect();
        let mut seeded = RlmReconcilerRes::with_launch_seed(RlmTuning::cloud(20), spawner(), seed);
        let mut d = dir();
        seeded
            .ledger
            .record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        seeded.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(100));
        assert_eq!(
            seeded.spins_requested, 0,
            "the recovered launch seed suppresses the re-spawn (path-keyed launch_present)"
        );

        // Control: the SAME demand with an EMPTY (genesis) seed DOES spawn — so it is the seed, not some
        // other guard, that suppressed the re-spawn above.
        let mut cold = RlmReconcilerRes::new(RlmTuning::cloud(20), spawner());
        let mut d2 = dir();
        cold.ledger
            .record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        cold.reconcile_and_drive(&mut d2, &|_n| false, UniverseTick(100));
        assert_eq!(
            cold.spins_requested, 1,
            "an unseeded (genesis) reconciler spawns the demanded realm"
        );
    }

    #[test]
    fn drive_launch_reconcile_drops_a_minted_node_once_its_head_appears() {
        let mut rlm = RlmReconcilerRes::new(RlmTuning::cloud(20), spawner());
        let mut d = dir();
        rlm.ledger
            .record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(100));
        assert_eq!(
            rlm.launches.minted.get(sys(7).path()).map(BTreeMap::len),
            Some(1)
        );
        // The shard self-grants its head ⇒ the launch reconcile drains the minted node.
        grant(&mut d, RealmId::System(7), 50, 1);
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(102));
        assert!(
            !rlm.launches.minted.contains_key(sys(7).path()),
            "head appeared ⇒ launch drained (BUG-B)"
        );
    }

    #[test]
    fn drive_kills_a_teardown_ready_realm_and_revokes_its_head() {
        let mut rlm = RlmReconcilerRes::new(RlmTuning::cloud(20), spawner());
        let mut d = dir();
        grant(&mut d, RealmId::System(7), 50, 3);
        let now = UniverseTick(500);
        rlm.ledger.record_demand(
            &sys(7),
            DemandVerb::Empty,
            UniverseTick(now.0 - 2),
            Fence(1),
        );
        let mut open = LedgerDelta::default();
        open.set_draining.insert(
            sys(7).path().clone(),
            Some(UniverseTick(now.0 - rlm.tuning.teardown_drain_ticks)),
        );
        rlm.ledger.apply_delta(&open);
        rlm.reconcile_and_drive(&mut d, &|_n| false, now);
        assert_eq!(rlm.teardowns_reaped, 1);
        assert!(
            !head_present(&d, RealmId::System(7)),
            "head revoked on kill"
        );
        // Post-kill gauges: nothing desired, nothing running.
        assert_eq!(rlm.desired_gauge, 0);
        assert_eq!(rlm.running_gauge, 0);
    }

    #[test]
    fn drive_force_reaps_a_zombie_then_respawns_next_sweep() {
        let mut rlm = RlmReconcilerRes::new(RlmTuning::cloud(20), spawner());
        let mut d = dir();
        grant(&mut d, RealmId::System(7), 50, 3);
        rlm.ledger
            .record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        // node 50 DEAD ⇒ zombie ⇒ ForceReap (force-revoke the stale head).
        rlm.reconcile_and_drive(&mut d, &|n| n == NodeId(50), UniverseTick(100));
        assert_eq!(rlm.force_reaps, 1);
        assert!(
            !head_present(&d, RealmId::System(7)),
            "zombie head force-revoked"
        );
        assert_eq!(
            rlm.spins_requested, 0,
            "no spawn the same sweep as the reap"
        );
        // Next sweep: head gone, still demanded ⇒ re-spawn (self-heal).
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(101));
        assert_eq!(rlm.spins_requested, 1);
    }

    #[test]
    fn drive_backs_off_after_a_launch_failure_then_retries() {
        let t = RlmTuning::cloud(20);
        let cd = t.spinup_cooldown_ticks;
        let mut rlm = RlmReconcilerRes::new(t, Box::new(FailingSpawner));
        let mut d = dir();
        rlm.ledger
            .record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        // Sweep 1: spawn fails ⇒ streak 1.
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(100));
        assert_eq!(rlm.spins_requested, 1);
        assert_eq!(rlm.spins_failed, 1);
        assert_eq!(
            rlm.launches.fail_streak.get(sys(7).path()).copied(),
            Some(1)
        );
        // Sweep 2 at now = 100 + cooldown: eligible per reconcile, but within the 2×cooldown backoff ⇒ skip.
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(100 + cd));
        assert_eq!(rlm.spins_requested, 1, "still backing off ⇒ no 2nd attempt");
        // Sweep 3 at now = 100 + 2×cooldown: past the backoff ⇒ retry (fails again).
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(100 + 2 * cd));
        assert_eq!(rlm.spins_requested, 2, "past backoff ⇒ retry");
        assert_eq!(rlm.spins_failed, 2);
        // The failing spawner still kills + lists cleanly (a launcher that can't create can still reap).
        assert!(FailingSpawner.kill_realm(NodeId(0)).is_ok());
        assert!(FailingSpawner.live_nodes().is_empty());
    }

    #[test]
    fn drive_launch_reconcile_holds_a_pending_launch_then_drops_it_past_ttl() {
        let t = RlmTuning::cloud(20);
        let ttl = t.launch_ttl_ticks;
        let mut rlm = RlmReconcilerRes::new(t, Box::new(GhostSpawner));
        let mut d = dir();
        rlm.ledger
            .record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        // Sweep 1: spawns (minted 999), NOT live yet, but within the launch TTL ⇒ HELD (a real pod may
        // still be registering — never drop a just-launched node).
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(100));
        assert!(
            rlm.launches.minted.contains_key(sys(7).path()),
            "a fresh launch is held through its TTL grace"
        );
        // Sweep past the TTL, still no head + never live ⇒ a silent-async-failure is dropped so a fresh
        // SpinUp re-fires next sweep (BUG-B silent-failure self-heal).
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(100 + ttl));
        assert!(
            !rlm.launches.minted.contains_key(sys(7).path()),
            "a launch that never went live is dropped past its TTL"
        );
        assert!(GhostSpawner.kill_realm(NodeId(999)).is_ok());
    }

    #[test]
    fn drive_gauges_reflect_desired_and_running() {
        let mut rlm = RlmReconcilerRes::new(RlmTuning::cloud(20), spawner());
        let mut d = dir();
        grant(&mut d, RealmId::System(7), 50, 1); // sys7 has a live head
        rlm.ledger
            .record_demand(&sys(7), DemandVerb::KeepAlive, UniverseTick(100), Fence(1));
        rlm.ledger
            .record_demand(&sys(8), DemandVerb::SpinUp, UniverseTick(100), Fence(1)); // desired, not running
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(100));
        assert_eq!(rlm.running_gauge, 1, "only sys7 has a live head");
        assert_eq!(rlm.desired_gauge, 2, "both are demanded");
    }

    #[test]
    fn drive_quiesce_blocks_teardown() {
        let mut rlm = RlmReconcilerRes::new(RlmTuning::cloud(20), spawner());
        rlm.arm_quiesce(UniverseTick(10_000));
        assert_eq!(
            rlm.rlm_quiesced_until(),
            UniverseTick(10_000),
            "the freeze watermark reads back what was armed"
        );
        let mut d = dir();
        grant(&mut d, RealmId::System(7), 50, 3);
        let now = UniverseTick(500);
        rlm.ledger.record_demand(
            &sys(7),
            DemandVerb::Empty,
            UniverseTick(now.0 - 2),
            Fence(1),
        );
        rlm.reconcile_and_drive(&mut d, &|_n| false, now);
        assert_eq!(rlm.teardowns_reaped, 0, "quiesce blocks the reap");
        assert!(head_present(&d, RealmId::System(7)));
    }

    #[test]
    fn exec_kill_leaves_the_head_and_cell_on_a_refused_revoke() {
        // A Kill whose revoke is REFUSED (a fence race — a newer owner grabbed the realm between the
        // decision and the drive) must LEAVE the head + the cell so the next sweep re-decides, and must NOT
        // count a reap (critique H-1: a refused revoke self-heals rather than desyncing). The synchronous
        // integration path can't produce this race, so the defensive branch is covered by driving
        // `exec_kill` directly with a stale fence.
        let mut rlm = RlmReconcilerRes::new(RlmTuning::cloud(20), spawner());
        let mut d = dir();
        grant(&mut d, RealmId::System(7), 50, 5); // head now at fence 5
        rlm.ledger
            .record_demand(&sys(7), DemandVerb::Empty, UniverseTick(100), Fence(1));
        rlm.exec_kill(
            &mut d,
            sys(7).path(),
            NodeId(50),
            Fence(3),
            UniverseTick(200),
        ); // stale fence 3
        assert_eq!(rlm.teardowns_reaped, 0, "a refused revoke counts no reap");
        assert!(
            head_present(&d, RealmId::System(7)),
            "the newer owner's head stands"
        );
        assert!(
            rlm.ledger().get(sys(7).path()).is_some(),
            "the cell is retained for a retry"
        );
    }

    #[test]
    fn apply_revoke_covers_revoked_refused_and_unknown() {
        let mut d = dir();
        grant(&mut d, RealmId::System(7), 1, 5);
        // Refused (stale fence) ⇒ false, head stands.
        assert!(!apply_revoke(&mut d, sys(7).path(), Fence(3)));
        assert!(
            head_present(&d, RealmId::System(7)),
            "a refused revoke leaves the head"
        );
        // Revoked (matching fence) ⇒ true, head gone.
        assert!(apply_revoke(&mut d, sys(7).path(), Fence(5)));
        assert!(!head_present(&d, RealmId::System(7)));
        // UnknownKey (already gone) ⇒ true.
        assert!(apply_revoke(&mut d, sys(7).path(), Fence(5)));
    }

    // ---- slice 3e — the reconcile_realm_lifecycle system (chain wiring) --------------------------

    fn lifecycle_world(rlm: RlmReconcilerRes, tick: u64) -> World {
        let mut world = World::new();
        let cs = vd_sim::runtime::ClockSample {
            universe_tick: UniverseTick(tick),
            ..vd_sim::runtime::ClockSample::default()
        };
        world.insert_resource(cs);
        world.insert_resource(crate::orchestrator::DirectoryRes(DirectoryCore::new(
            DirectoryTuning::default(),
        )));
        world.insert_resource(crate::saga_runtime::SagaRuntimeRes::with_tuning(
            vd_sim::saga::SagaTuning::default(),
        ));
        world.insert_resource(rlm);
        world
    }

    fn run_lifecycle(world: &mut World) -> u64 {
        let mut sched = Schedule::default();
        sched.add_systems(reconcile_realm_lifecycle);
        sched.run(world);
        world.resource::<RlmReconcilerRes>().spins_requested
    }

    #[test]
    fn reconcile_system_is_inert_when_interval_zero() {
        let mut rlm = RlmReconcilerRes::new(RlmTuning::default(), spawner()); // interval 0
        rlm.ledger
            .record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        let mut world = lifecycle_world(rlm, 100);
        assert_eq!(run_lifecycle(&mut world), 0, "inert tuning ⇒ no sweep");
    }

    #[test]
    fn reconcile_system_sweeps_on_the_cadence() {
        // interval 1 (cloud) ⇒ every tick: a demanded realm with no head spins up.
        let mut rlm = RlmReconcilerRes::new(RlmTuning::cloud(20), spawner());
        rlm.ledger
            .record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        let mut world = lifecycle_world(rlm, 100);
        assert_eq!(run_lifecycle(&mut world), 1);
    }

    #[test]
    fn reconcile_system_skips_off_cadence_ticks() {
        let t = RlmTuning {
            reconcile_interval_ticks: 4,
            ..RlmTuning::cloud(20)
        };
        // now = 101 ⇒ 101 % 4 != 0 ⇒ skip.
        let mut off = RlmReconcilerRes::new(t, spawner());
        off.ledger
            .record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        let mut w_off = lifecycle_world(off, 101);
        assert_eq!(run_lifecycle(&mut w_off), 0, "off-cadence tick ⇒ skip");
        // now = 104 ⇒ 104 % 4 == 0 ⇒ sweep.
        let mut on = RlmReconcilerRes::new(t, spawner());
        on.ledger
            .record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        let mut w_on = lifecycle_world(on, 104);
        assert_eq!(run_lifecycle(&mut w_on), 1, "on-cadence tick ⇒ sweep");
    }

    #[test]
    fn reconcile_system_consults_liveness_for_a_running_realm() {
        // A realm WITH a live head ⇒ the sweep evaluates `running_live(Some, dead)`, invoking the liveness
        // closure that bridges to `SagaRuntimeRes::is_node_latched_dead`.
        let mut rlm = RlmReconcilerRes::new(RlmTuning::cloud(20), spawner());
        rlm.ledger
            .record_demand(&sys(7), DemandVerb::KeepAlive, UniverseTick(100), Fence(1));
        let mut world = lifecycle_world(rlm, 100);
        world
            .resource_mut::<crate::orchestrator::DirectoryRes>()
            .0
            .grant(
                DirectoryKey::Realm(RealmId::System(7)),
                AuthorityRef::Shard(NodeId(50)),
                Fence(1),
                UniverseTick(0),
            );
        run_lifecycle(&mut world);
        assert_eq!(
            world.resource::<RlmReconcilerRes>().running_gauge,
            1,
            "the live realm is counted ⇒ the liveness closure was consulted"
        );
    }
}
