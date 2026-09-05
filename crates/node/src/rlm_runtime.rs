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

use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::prelude::{Res, ResMut, Resource};
use vd_sim::directory::{DirectoryCore, RevokeOutcome};
use vd_sim::io::{Inbound, MsgClass, RealmSpawner};
use vd_sim::rlm::{
    DemandLedger, LifecycleAction, RlmTuning, desired_alive, reconcile, running_live,
};
use vd_sim::runtime::InboundBox;
use vd_wire::intershard::InterShardFlow;
use vd_wire::seams::directory::DirectoryKey;

use vd_core::pose::RealmId;
use vd_core::realm_coord::RealmCoord;
use vd_core::realm_path::RealmPath;
use vd_core::{Fence, NodeId, UniverseTick};

/// Exponential-backoff cap: a launch-failure streak past this stops widening the retry window (`2^10 ×`
/// the spin-up cooldown) — the retry never hard-stops (a transient cluster-full self-heals) but never
/// hammers a permanently-unschedulable realm either.
const BACKOFF_CAP: u32 = 10;

/// RLM 5f — the ONE arming decision (Tier-A, both arms covered): resolve the orchestrator's [`RlmTuning`]
/// from the launcher's `--demand` flag. When `demand` is set the reconciler is ARMED with a live budget
/// whose launch-TTL is floored by the MEASURED process boot latency (so a slow real fork is never re-spun
/// mid-boot); when unset it is the fully-INERT default (`reconcile_interval_ticks == 0` ⇒ never sweeps ⇒
/// boot byte-identical). NOT a per-kind fork (HR3) — a value selection on ONE struct. The caller must
/// [`RlmTuning::validate`] the result and fail loud (mirrors the saga-budget validate at boot).
#[must_use]
pub fn resolve_rlm_tuning(
    demand: bool,
    tick_hz: u32,
    boot_ticks_p99: u64,
    settle: u64,
) -> RlmTuning {
    if demand {
        RlmTuning::cloud_with_boot(tick_hz, boot_ticks_p99, settle)
    } else {
        RlmTuning::default()
    }
}

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

    /// The ruler switch, slice 4 — exterior crossings whose cell and launch record moved house.
    pub reparents_applied: u64,
    /// The ruler switch, slice 4 — reparents this reconciler could not resolve (counted, not assumed).
    pub reparents_unresolved: u64,
    /// `ExteriorMoved` statements pushed to session gateways (one per gateway per applied reparent).
    pub exterior_moves_told: u64,
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
    /// Demands whose SENDER the directory head does not show holding the demanded realm's PARENT or the
    /// realm ITSELF (own-coord keep-alive) — including demands whose realm resolves to no head at all.
    /// MEASURE-ONLY (owner ruling 2026-08-15, docs/design/owner_decisions_2026-08-15.md item 11): counted
    /// and warned, and the demand is processed UNCHANGED — the warm-ahead path must never eat a blind
    /// window. Refusal is deferred to cloud, where mTLS names the sender.
    pub demand_sender_mismatch: u64,
    /// Last sweep's desired-realm count (gauge).
    pub desired_gauge: u64,
    /// Last sweep's running-realm count (gauge).
    pub running_gauge: u64,
    /// RLM 5f-4: the MAX observed launch→head-up latency (universe ticks), monotone. The measured pod boot
    /// the launch-TTL is tuned from (`VD_BOOT_TICKS_P99`, 5f-4j); 0 until a demand-spawned head first
    /// appears. Surfaced in the admin snapshot (RlmView).
    boot_ticks_observed_max: u64,
    /// MONOTONE total of realm-sweeps where a pending arrival was the SOLE reason a realm stayed alive.
    /// ZERO on every healthy run — a healthy crossing's destination is already held up by the source
    /// shard's own keep-alive, so this only moves when that cover has lapsed and the arrival is genuinely
    /// the last thing standing between a player and a reaped destination.
    pub arrival_shield_vetoes: u64,
    /// How many realms the arrival shield is holding up RIGHT NOW (last sweep's gauge).
    pub arrival_shield_gauge: u64,
    /// MONOTONE count of hand-offs whose arrival shield ran out of budget and was lifted. `> 0` means a
    /// hand-off is wedged and a player may have been dropped onto the orphan-recovery path — the loudest
    /// number in this struct, and 0 in every healthy run.
    pub arrivals_shield_expired: u64,
    /// Which hand-offs have already been alarmed about, so a wedged one alarms once rather than every
    /// sweep forever. Pruned each sweep to the still-expiring set — bounded by the live saga count.
    arrivals_shield_expired_seen: std::collections::BTreeSet<vd_core::TransferId>,
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
            reparents_applied: 0,
            reparents_unresolved: 0,
            exterior_moves_told: 0,
            spins_requested: 0,
            spins_failed: 0,
            teardowns_reaped: 0,
            force_reaps: 0,
            undecodable_demands: 0,
            demand_sender_mismatch: 0,
            desired_gauge: 0,
            running_gauge: 0,
            boot_ticks_observed_max: 0,
            arrival_shield_vetoes: 0,
            arrival_shield_gauge: 0,
            arrivals_shield_expired: 0,
            arrivals_shield_expired_seen: std::collections::BTreeSet::new(),
        }
    }

    /// The MAX observed launch→head-up latency (universe ticks) — the measured pod boot (RLM 5f-4). Read by
    /// the admin snapshot; 5f-4j feeds it back into `VD_BOOT_TICKS_P99` to tune the launch-TTL.
    #[must_use]
    pub fn boot_ticks_observed_max(&self) -> u64 {
        self.boot_ticks_observed_max
    }

    /// Read the demand ledger (the reconcile system + tests).
    #[must_use]
    pub fn ledger(&self) -> &DemandLedger {
        &self.ledger
    }

    /// LIFT THE SHIELD LOUDLY. A hand-off that has held its destination past the whole derived budget is
    /// not slow, it is wedged; its protection is withdrawn so the realm can be reclaimed, and the player
    /// falls back to the orphan-recovery path that already exists for someone whose owner is gone. This
    /// is the failure mode chosen deliberately over pinning one realm forever per wedged hand-off.
    ///
    /// Alarms ONCE per hand-off, not once per sweep: a wedged saga re-reports every tick forever, and an
    /// error line per tick is how a real alarm gets muted by the people who need to see it. The set of
    /// already-alarmed transfers is pruned to the ones still expiring, so it cannot grow without bound.
    pub fn report_expired_arrivals(&mut self, expired: &[crate::saga_runtime::ExpiredArrival]) {
        for e in expired {
            if self.arrivals_shield_expired_seen.insert(e.transfer) {
                self.arrivals_shield_expired += 1;
                tracing::error!(
                    transfer = ?e.transfer,
                    realm = %e.realm,
                    state = %e.state,
                    age_ticks = e.age_ticks,
                    "ARRIVAL SHIELD EXPIRED — a hand-off has held its destination realm alive past its \
                     whole budget and is wedged, not slow. The shield is lifting: the realm may now be \
                     reclaimed and the subject falls back to orphan recovery.",
                );
            }
        }
        let still: std::collections::BTreeSet<vd_core::TransferId> =
            expired.iter().map(|e| e.transfer).collect();
        self.arrivals_shield_expired_seen
            .retain(|t| still.contains(t));
    }

    /// ★ A CHILD MOVED HOUSE (the ruler switch, slice 4): an exterior crossing committed, `child` is
    /// now `to_realm`'s child. Its demand cell is re-keyed under its new path (the coord the new
    /// parent's own cell gives, plus the child's level) and its launch record rewritten, so a restart
    /// rehydrates it under its true parent and the old cell is not desired for ever. Counted rather
    /// than assumed when the new parent has no cell here or the child has no live node.
    pub fn reparent(
        &mut self,
        child: RealmId,
        to_realm: RealmId,
        dir: &DirectoryCore,
    ) -> Option<RealmCoord> {
        let old = self.ledger.path_of_realm(child);
        let parent = self.ledger.coord_of_realm(to_realm);
        let node = dir
            .head(DirectoryKey::Realm(child))
            .map(|r| r.authority.node());
        let (Some(old), Some(parent), Some(node)) = (old, parent, node) else {
            self.reparents_unresolved += 1;
            tracing::warn!(
                ?child,
                ?to_realm,
                "REPARENT UNRESOLVED: no cell for the child, no cell for the new parent, or no live node",
            );
            return None;
        };
        let coord = parent.child(vd_core::worldgen::level_of(child));
        self.ledger.rekey(&old, &coord);
        if let Err(e) = self.spawner.reparent_realm(node, &coord) {
            tracing::warn!(
                ?child,
                ?node,
                ?e,
                "REPARENT: the launch record could not be rewritten"
            );
        }
        self.reparents_applied += 1;
        tracing::info!(
            ?child,
            ?to_realm,
            ?node,
            "REPARENTED: the child's cell and launch record moved house"
        );
        Some(coord)
    }

    /// ★ THE PEER BOOK: where a node this reconciler's spawner launched listens (`None` for a stranger).
    #[must_use]
    pub fn addr_of(&self, node: NodeId) -> Option<([u8; 16], u16)> {
        self.spawner.addr_of(node)
    }

    /// This reconciler's timing budget.
    #[must_use]
    pub fn tuning(&self) -> &RlmTuning {
        &self.tuning
    }

    /// The current live demand-spawned node roster (RLM 5f-4c). Republished into
    /// [`DynamicClockPeers`](crate::orchestrator::DynamicClockPeers) each sweep so a spawned shard receives
    /// the `ClockSync` broadcast (and therefore syncs, then authors + demands its own children).
    #[must_use]
    pub fn live_nodes(&self) -> std::collections::BTreeSet<NodeId> {
        self.spawner.live_nodes()
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
        arriving: &BTreeSet<RealmId>,
    ) {
        let (actions, delta) = reconcile(
            &self.ledger,
            dir,
            liveness_dead,
            self.launches.live(),
            &self.tuning,
            now,
            self.rlm_quiesced_until,
            arriving,
        );
        // Count the shield BY NAME every sweep it is load-bearing — the cure for this subsystem's
        // history of guarantees that held silently and then stopped holding just as silently. The
        // gauge is this sweep's count; the total is monotone, so a wedged hand-off is a climbing
        // number an operator can see rather than an absence nobody notices.
        self.arrival_shield_gauge = delta.arrival_shielded.len() as u64;
        self.arrival_shield_vetoes += self.arrival_shield_gauge;
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
        self.update_gauges(dir, liveness_dead, now, arriving);
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
                // ★ THE STREAK IS NOT CLEARED HERE, AND THAT IS THE FIX (owner ruling 2026-08-24 —
                // slice S1's third part).
                //
                // It used to be. A successful fork+exec was treated as a successful LAUNCH — but all a
                // successful fork proves is that a process started, not that a realm came up. A child
                // that starts and then exits at once (it refuses its durable store, its port is taken,
                // its config is wrong) therefore never engaged the backoff, and the reconciler respawned
                // it at the base cooldown FOREVER. That failure mode arrives the moment a store can
                // refuse — which is exactly what the rest of this slice built.
                //
                // A launch is proven by the realm's HEAD appearing in the directory, so the streak is
                // cleared THERE (see `reconcile_launches`), and a child that dies without ever getting
                // a head is counted as the failure it is.
            }
            Err(e) => {
                self.spins_failed += 1;
                let streak = self.launches.fail_streak.entry(path.clone()).or_default();
                *streak += 1;
                // LOUD, not silent. This arm used to be `Err(_)` — it bumped the counter and threw the
                // reason away, so a demand spawn that failed left NO trace of WHY. `SpawnError`
                // carries a full `LaunchFailed { reason }` (the launcher already names the child's log
                // and the likely cause), and discarding it is what made the standing 1-in-8 fly-out
                // spawn failure undiagnosable: the gate could say a spawn failed but nothing could say
                // what happened. Nothing in this file is worth knowing silently.
                tracing::warn!(
                    realm = ?path,
                    error = %e,
                    fail_streak = *streak,
                    spins_failed = self.spins_failed,
                    "demand spawn FAILED — the reconciler backs off and retries",
                );
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
        let mut observed_boot = self.boot_ticks_observed_max;
        // Split the borrow so the streak can be written while the minted set is walked — the two live in
        // one struct and this loop must now touch both.
        let LaunchLedger {
            minted,
            fail_streak,
            ..
        } = &mut self.launches;
        for (path, nodes) in &mut *minted {
            let rid = path
                .realm_id()
                .expect("a lifecycle realm path is non-empty");
            let head_up = dir.head(DirectoryKey::Realm(rid)).is_some();
            if head_up {
                // THE ONLY PROOF A LAUNCH WORKED: the realm registered. Clearing the streak here rather
                // than at the fork is what stops a child that starts and dies from looking like a
                // success (see `exec_spinup`).
                fail_streak.remove(path);
                // RLM 5f-4: the head appeared — the launch→head-up latency (now - mint tick) is the MEASURED
                // pod boot. Fold the max over the draining minted nodes into the monotone gauge so 5f-4j can
                // tune the launch-TTL from real cluster boots. (These nodes drain in the retain below.)
                for minted_at in nodes.values() {
                    observed_boot = observed_boot.max(now.0.saturating_sub(minted_at.0));
                }
            }
            nodes.retain(|node, minted_at| {
                let within_ttl = now.0.saturating_sub(minted_at.0) < ttl;
                let alive = live.contains(node);
                // A CHILD THAT DIED WITHOUT EVER REGISTERING IS A FAILED LAUNCH, and it must be counted
                // as one or nothing ever backs off. Distinguished from the two innocent cases by the
                // three facts already in hand: no head appeared, the process is gone, and it has had
                // longer than the launch window to get there.
                let died_before_registering = !head_up & !alive & !within_ttl;
                if died_before_registering {
                    *fail_streak.entry(path.clone()).or_default() += 1;
                }
                !head_up & (alive | within_ttl)
            });
        }
        self.boot_ticks_observed_max = observed_boot;
        minted.retain(|_, nodes| !nodes.is_empty());
    }

    /// Refresh the observability gauges (desired + running realm counts among tracked cells) using the
    /// public kernel predicates — not a second decision. Branchless (`u64::from(bool)`, HR5).
    fn update_gauges(
        &mut self,
        dir: &DirectoryCore,
        liveness_dead: &dyn Fn(NodeId) -> bool,
        now: UniverseTick,
        arriving: &BTreeSet<RealmId>,
    ) {
        let mut desired = 0u64;
        let mut running = 0u64;
        for (_path, cell) in self.ledger.iter() {
            let head = dir.head(DirectoryKey::Realm(cell.coord.lowered()));
            let rl = running_live(head, liveness_dead);
            running += u64::from(rl);
            // The same three arms the decision uses — a gauge that disagreed with the decision would
            // be worse than no gauge.
            let arrival = arriving.contains(&cell.coord.lowered());
            desired += u64::from(desired_alive(cell, now, &self.tuning, rl, arrival));
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

/// Demand PROVENANCE (owner ruling 2026-08-15, docs/design/owner_decisions_2026-08-15.md item 11): when
/// a demand names realm X, the LAWFUL senders are the process the directory head shows holding X's
/// PARENT, or the process holding X ITSELF (the own-coord keep-alive). Anything else — including a
/// realm whose heads are unresolvable — is a mismatch the caller counts. Pure head-reads, no realm-kind
/// branch (a system's demand and an area's resolve identically).
fn demand_sender_is_lawful(dir: &DirectoryCore, child: &RealmCoord, from: NodeId) -> bool {
    let own = dir.head(DirectoryKey::Realm(child.lowered()));
    let parent = child
        .path()
        .parent_realm()
        .and_then(|rid| dir.head(DirectoryKey::Realm(rid)));
    own.is_some_and(|record| record.authority.node() == from)
        || parent.is_some_and(|record| record.authority.node() == from)
}

/// Slice 3c — INGEST: fold every re-asserted `RealmDemand` on the `Saga` class into the kernel ledger
/// (the SOURCE-AGNOSTIC ingress — an AoI demand and a player-spawn demand are indistinguishable). Mirrors
/// `serve_directory`'s decode; a Saga frame that is not a `RealmDemand` is skipped (it is another
/// service's — `Directory`/`SagaAck`), and an undecodable one is counted (honesty, parity with
/// `orchestrator.rs`). Sender provenance is MEASURED per demand ([`demand_sender_is_lawful`]) and never
/// gates: a mismatch counts + warns and the demand is honored unchanged (owner ruling 2026-08-15 —
/// observe mode only; enforcement waits for cloud mTLS).
pub fn record_realm_demands(
    inbox: Res<InboundBox>,
    dir: Res<crate::orchestrator::DirectoryRes>,
    mut rlm: ResMut<RlmReconcilerRes>,
) {
    for msg in &inbox.0 {
        let Inbound::Wire { from, class, bytes } = msg else {
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
        if !demand_sender_is_lawful(&dir.0, &demand.child, *from) {
            rlm.demand_sender_mismatch += 1;
            tracing::warn!(
                from = from.0,
                child = ?demand.child.path(),
                "realm demand from a sender the directory shows holding neither the realm's \
                 parent nor the realm itself — measured, demand honored (observe mode)"
            );
        }
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
    mut runtime: ResMut<crate::saga_runtime::SagaRuntimeRes>,
    mut rlm: ResMut<RlmReconcilerRes>,
    mut dynamic_peers: ResMut<crate::orchestrator::DynamicClockPeers>,
    mut outbox: ResMut<vd_sim::runtime::OutboundBox>,
) {
    // The ruler switch, slice 4: committed exterior crossings move house every tick, ahead of the
    // sweep cadence, so a cell is never swept under a stale path.
    for (child, to_realm, parent_node) in std::mem::take(&mut runtime.pending_reparents) {
        let Some(coord) = rlm.reparent(child, to_realm, &dir.0) else {
            continue; // unresolved: counted inside, and nothing to tell
        };
        // ★ THE SKY FOLLOWS THE HULL (slice 5): every session gateway hears the child's new coord, so
        // a session aboard re-derives its chain. Addressed from the record (the gateways holding
        // sessions, like the shard roster); a producer-less one-shot, so `Retained` — the durable
        // outbox replays it across a crash of this process.
        let flow = InterShardFlow::ExteriorMoved(vd_wire::intershard::ExteriorMoved {
            child: coord,
            parent_node,
            at: clock.universe_tick,
        });
        for gateway in dir.0.session_gateways() {
            outbox.push_flow_durable(
                gateway,
                MsgClass::Saga,
                &flow,
                vd_sim::io::Durability::Retained,
            );
            rlm.exterior_moves_told += 1;
        }
    }
    let interval = rlm.tuning().reconcile_interval_ticks;
    if interval == 0 {
        return; // INERT: the byte-identical default (no reconciler wired) — DynamicClockPeers stays empty.
    }
    let now = clock.universe_tick;
    if !now.0.is_multiple_of(interval) {
        return; // scale cadence: sweep only every `interval` ticks.
    }
    let dead = |node: NodeId| runtime.is_node_latched_dead(node);
    // THE ARRIVAL SET, read from the same saga runtime this system already borrows, on the same tick,
    // right after `drive_sagas_core` produced it. No message to lose, no countdown to beat.
    let (arriving, expired) = runtime.arriving_dest_realms(now, rlm.tuning().arrival_shield_ticks);
    rlm.report_expired_arrivals(&expired);
    rlm.reconcile_and_drive(&mut dir.0, &dead, now, &arriving);
    // RLM 5f-4c: republish the live spawned-node roster so the clock broadcast (next tick) reaches every
    // demand-spawned shard — a node spawned in THIS sweep is included immediately, a reaped one drops next
    // sweep. Assigned (not merged) so the set can only shrink when the roster does (no unbounded growth).
    dynamic_peers.0 = rlm.live_nodes();
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

    /// One expired arrival, as the shield reports it when a hand-off runs out of budget.
    fn expired(transfer: u128) -> crate::saga_runtime::ExpiredArrival {
        crate::saga_runtime::ExpiredArrival {
            transfer: vd_core::TransferId(transfer),
            realm: RealmId::Planet(7),
            state: "Promoting".to_owned(),
            age_ticks: 999,
        }
    }

    #[test]
    fn a_committed_exterior_crossing_rekeys_the_childs_cell_under_its_new_parent() {
        // The ruler switch, slice 4: the hull's cell moves from under System 41 to under System 42;
        // its watermarks ride along; an unresolvable reparent is counted, never assumed.
        let mut rlm = RlmReconcilerRes::new(RlmTuning::default(), spawner());
        let mut dir = DirectoryCore::new(DirectoryTuning::default());
        let hull_entity = vd_core::EntityId::pack(vd_core::entity_kind::EntityKind::Ship, 1, 1, 0);
        let hull = RealmId::Ship(hull_entity);
        let old = sys(41).child(vd_core::worldgen::level_of(hull));
        rlm.ledger
            .record_demand(&sys(41), DemandVerb::SpinUp, UniverseTick(1), Fence(1));
        rlm.ledger
            .record_demand(&sys(42), DemandVerb::SpinUp, UniverseTick(1), Fence(1));
        rlm.ledger
            .record_demand(&old, DemandVerb::SpinUp, UniverseTick(5), Fence(1));
        let _ = dir.grant(
            DirectoryKey::Realm(hull),
            vd_wire::seams::directory::AuthorityRef::Shard(NodeId(1_007)),
            Fence(1),
            UniverseTick(1),
        );
        rlm.reparent(hull, sys(42).lowered(), &dir);
        let new = sys(42).child(vd_core::worldgen::level_of(hull));
        assert!(rlm.ledger.get(old.path()).is_none(), "the old cell is gone");
        let cell = rlm.ledger.get(new.path()).expect("the cell moved");
        assert_eq!(cell.coord, new);
        assert_eq!(
            cell.last_demand_tick,
            UniverseTick(5),
            "its watermarks rode along"
        );
        assert_eq!(rlm.reparents_applied, 1);
        // No cell for the new parent: unresolved, counted, nothing moved.
        rlm.reparent(hull, sys(43).lowered(), &dir);
        assert_eq!(rlm.reparents_unresolved, 1);
        assert!(rlm.ledger.get(new.path()).is_some());
    }

    #[test]
    fn a_reparent_the_spawner_refuses_to_record_still_moves_the_childs_cell() {
        // TWO records say who a hull's parent is: the reconciler's own cell, which the running world
        // reads every tick, and the launch record, which only a restart reads. The move applies to the
        // cell at once. A spawner that cannot rewrite the launch record is WARNED about, never allowed
        // to hold the move back — the live world must agree with the directory the same tick the
        // hand-over commits.
        //
        // Example: the hull's exterior crosses from System 41 to System 42. The reconciler re-keys the
        // hull's cell under System 42 and counts the move applied, even though the launcher refused to
        // rewrite the hull's launch record.
        let mut rlm = RlmReconcilerRes::new(RlmTuning::default(), Box::new(FailingSpawner));
        let mut dir = DirectoryCore::new(DirectoryTuning::default());
        let hull_entity = vd_core::EntityId::pack(vd_core::entity_kind::EntityKind::Ship, 1, 1, 0);
        let hull = RealmId::Ship(hull_entity);
        let old = sys(41).child(vd_core::worldgen::level_of(hull));
        rlm.ledger
            .record_demand(&sys(42), DemandVerb::SpinUp, UniverseTick(1), Fence(1));
        rlm.ledger
            .record_demand(&old, DemandVerb::SpinUp, UniverseTick(5), Fence(1));
        let _ = dir.grant(
            DirectoryKey::Realm(hull),
            vd_wire::seams::directory::AuthorityRef::Shard(NodeId(1_007)),
            Fence(1),
            UniverseTick(1),
        );
        let new = rlm
            .reparent(hull, sys(42).lowered(), &dir)
            .expect("the cell moves house");
        assert_eq!(new, sys(42).child(vd_core::worldgen::level_of(hull)));
        assert_eq!((rlm.reparents_applied, rlm.reparents_unresolved), (1, 0));
        assert!(rlm.ledger.get(old.path()).is_none(), "the old cell is gone");
        assert!(
            rlm.ledger.get(new.path()).is_some(),
            "the new cell is there"
        );
    }

    #[test]
    fn a_child_that_starts_and_dies_without_registering_is_counted_as_a_failed_launch() {
        // THE RESPAWN LOOP, CLOSED (owner ruling 2026-08-24, slice S1).
        //
        // A successful fork used to count as a successful LAUNCH and clear the failure streak. But a
        // fork only proves a process started — not that a realm came up. A child that starts and exits
        // at once (it refuses its durable store, its port is taken, its config is wrong) therefore never
        // engaged the backoff, and the reconciler respawned it at the base cooldown forever.
        //
        // That failure mode is not hypothetical: the rest of this slice built a store that REFUSES, and
        // a refusing shard is exactly a child that starts and dies.
        //
        // RED BEFORE THE FIX: with the streak cleared at the fork, the assertion below reads zero.
        let mut rlm = RlmReconcilerRes::new(RlmTuning::default(), spawner());
        let dir = DirectoryCore::new(DirectoryTuning::default());
        let coord = sys(41);
        let path = coord.path().clone();

        // A launch that forks fine…
        rlm.exec_spinup(&coord, UniverseTick(1));
        let child = *rlm
            .launches
            .minted
            .get(&path)
            .and_then(|n| n.keys().next())
            .expect("the fork minted a child");
        assert_eq!(
            rlm.launches.fail_streak.get(&path).copied(),
            None,
            "a fork that succeeded is not yet a failure — nothing is known about it yet"
        );
        assert!(
            !rlm.launches.minted.is_empty(),
            "the child is being watched"
        );

        // …and then the child DIES, and no realm head ever appeared. This is the shape of a shard that
        // starts and immediately refuses its durable store — which is precisely what the rest of this
        // slice built, and why this loop had to be closed before that refusal shipped.
        rlm.spawner.kill_realm(child).expect("the child dies");
        let past_ttl = UniverseTick(1 + rlm.tuning.launch_ttl_ticks + 1);
        rlm.reconcile_launches(&dir, past_ttl);

        assert_eq!(
            rlm.launches.fail_streak.get(&path).copied(),
            Some(1),
            "a child that died without ever registering must count as a failed launch, or nothing ever \
             backs off and the reconciler respawns it forever"
        );
        assert!(
            rlm.launches.minted.is_empty(),
            "and it stops being watched — it is gone"
        );
    }

    #[test]
    fn a_launch_is_proven_by_the_realm_registering_and_that_is_what_clears_the_streak() {
        // THE OTHER HALF, and the reason the streak moved: a launch is successful when the REALM comes
        // up, not when the process does. Without this the streak would only ever grow and a realm that
        // recovered would stay in backoff forever — which would be a different bug of the same family.
        let mut rlm = RlmReconcilerRes::new(RlmTuning::default(), spawner());
        let mut dir = DirectoryCore::new(DirectoryTuning::default());
        let coord = sys(42);
        let path = coord.path().clone();

        // Start from a realm that has already failed twice.
        rlm.launches.fail_streak.insert(path.clone(), 2);
        rlm.exec_spinup(&coord, UniverseTick(1));
        assert_eq!(
            rlm.launches.fail_streak.get(&path).copied(),
            Some(2),
            "forking does not clear the streak"
        );

        // Now the realm registers — the only thing that proves the launch worked.
        dir.grant(
            DirectoryKey::Realm(coord.lowered()),
            AuthorityRef::Shard(NodeId(1000)),
            Fence(1),
            UniverseTick(2),
        );
        rlm.reconcile_launches(&dir, UniverseTick(2));

        assert_eq!(
            rlm.launches.fail_streak.get(&path).copied(),
            None,
            "the head appearing is what clears the streak"
        );
    }

    #[test]
    fn a_wedged_handoff_alarms_once_and_can_alarm_again_after_it_clears() {
        // AN ALARM NOBODY TESTED IS AN ALARM THAT FAILS QUIETLY. This is the operator-facing end of the
        // arrival shield: it fires when a hand-off has held its destination realm alive past its ENTIRE
        // budget, which means wedged rather than slow.
        //
        // ONCE PER HAND-OFF, not once per sweep — and that is the whole subtlety. A wedged saga re-reports
        // every single tick, forever. An error line per tick at 20Hz is not an alarm; it is how a real
        // alarm gets muted by the people who most need to see it.
        let mut rlm = RlmReconcilerRes::new(RlmTuning::default(), spawner());
        assert_eq!(rlm.arrivals_shield_expired, 0);

        rlm.report_expired_arrivals(&[expired(1)]);
        assert_eq!(rlm.arrivals_shield_expired, 1, "the first sighting alarms");

        // The same wedged hand-off, still wedged, sweep after sweep: counted ONCE.
        rlm.report_expired_arrivals(&[expired(1)]);
        rlm.report_expired_arrivals(&[expired(1)]);
        assert_eq!(
            rlm.arrivals_shield_expired, 1,
            "a hand-off that stays wedged does not re-alarm every tick"
        );

        // A DIFFERENT wedged hand-off is its own alarm — the dedup is per hand-off, not a global latch
        // that would swallow every later failure.
        rlm.report_expired_arrivals(&[expired(1), expired(2)]);
        assert_eq!(rlm.arrivals_shield_expired, 2);

        // THE PRUNE, which is what stops the memory of alarmed hand-offs growing without bound — and what
        // makes the alarm re-armable. Once a transfer stops being reported it is forgotten…
        rlm.report_expired_arrivals(&[]);
        // …so if it ever wedges again, that is a NEW event and it alarms again. Without the prune this
        // would stay silent forever after the first occurrence, which is the failure mode that looks
        // exactly like "the problem went away".
        rlm.report_expired_arrivals(&[expired(1)]);
        assert_eq!(
            rlm.arrivals_shield_expired, 3,
            "a hand-off that wedges, clears, and wedges again alarms both times"
        );
    }

    #[test]
    fn resolve_rlm_tuning_arms_only_under_demand() {
        // --demand OFF => the fully-INERT default (never sweeps; boot byte-identical); validate vacuous.
        let inert = resolve_rlm_tuning(false, 20, 100, 10);
        assert_eq!(inert, RlmTuning::default());
        assert_eq!(inert.reconcile_interval_ticks, 0);
        assert_eq!(inert.validate(), Ok(()));
        // --demand ON => the boot-floored live budget; validate-clean; min_dwell covers the measured boot.
        let armed = resolve_rlm_tuning(true, 20, 100, 10);
        assert_eq!(armed, RlmTuning::cloud_with_boot(20, 100, 10));
        assert_eq!(armed.reconcile_interval_ticks, 1);
        assert_eq!(armed.validate(), Ok(()));
        assert!(armed.min_dwell_ticks() >= 100 + 10);
    }

    fn demand_frame(coord: &RealmCoord, verb: DemandVerb, tick: u64, fence: u64) -> Inbound {
        demand_frame_from(coord, verb, tick, fence, NodeId(7))
    }

    /// [`demand_frame`] with a caller-chosen SENDER — the provenance tests name who is demanding.
    fn demand_frame_from(
        coord: &RealmCoord,
        verb: DemandVerb,
        tick: u64,
        fence: u64,
        from: NodeId,
    ) -> Inbound {
        let bytes = postcard::to_allocvec(&InterShardFlow::RealmDemand(RealmDemand {
            child: coord.clone(),
            parent_fence: Fence(fence),
            verb,
            universe_tick: UniverseTick(tick),
        }))
        .expect("encode");
        Inbound::Wire {
            from,
            class: MsgClass::Saga,
            bytes: bytes.into(),
        }
    }

    fn run(inbound: Vec<Inbound>) -> RlmReconcilerRes {
        // No directory heads: every sender is UNRESOLVABLE — measured, never gating (the demand
        // still folds; the provenance tests below pin all four sender classes explicitly).
        run_with_dir(DirectoryCore::new(DirectoryTuning::default()), inbound)
    }

    /// [`run`] with caller-built directory heads — the provenance tests grant who holds what.
    fn run_with_dir(dir: DirectoryCore, inbound: Vec<Inbound>) -> RlmReconcilerRes {
        crate::init_test_tracing();
        let mut world = World::new();
        world.insert_resource(InboundBox(inbound));
        world.insert_resource(crate::orchestrator::DirectoryRes(dir));
        world.insert_resource(RlmReconcilerRes::new(RlmTuning::cloud(20), spawner()));
        let mut schedule = Schedule::default();
        schedule.add_systems(record_realm_demands);
        schedule.run(&mut world);
        world.remove_resource::<RlmReconcilerRes>().expect("res")
    }

    /// Ingest `inbound` into an EXISTING reconciler through the UNCHANGED `record_realm_demands` system
    /// (mirrors [`run`] but folds into a res that already carries state), returning it for the next drive.
    fn ingest_into(rlm: RlmReconcilerRes, inbound: Vec<Inbound>) -> RlmReconcilerRes {
        crate::init_test_tracing();
        let mut world = World::new();
        world.insert_resource(InboundBox(inbound));
        world.insert_resource(crate::orchestrator::DirectoryRes(DirectoryCore::new(
            DirectoryTuning::default(),
        )));
        world.insert_resource(rlm);
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

    // ---- demand provenance (owner ruling 2026-08-15, item 11: measure-only) ----------------------

    /// A two-level coord (System 1 → Planet `seed`) so the PARENT arm of the provenance check has a
    /// resolvable head to name.
    fn planet_in_sys1(seed: u64) -> RealmCoord {
        RealmCoord::from_path(RealmPath::from_levels(vec![
            RealmLevel::new(RealmKindTag::System, 1),
            RealmLevel::new(RealmKindTag::Planet, seed),
        ]))
        .expect("two-level path has a leaf")
    }

    #[test]
    fn a_demand_from_the_parents_holder_or_the_realms_own_holder_is_lawful() {
        // OWN-holder arm: node 7 holds System(7) itself (the own-coord keep-alive) — no mismatch.
        let mut own = dir();
        grant(&mut own, RealmId::System(7), 7, 1);
        let rlm = run_with_dir(
            own,
            vec![demand_frame_from(
                &sys(7),
                DemandVerb::KeepAlive,
                100,
                1,
                NodeId(7),
            )],
        );
        assert_eq!(
            rlm.demand_sender_mismatch, 0,
            "the realm's own holder is lawful"
        );
        assert_eq!(rlm.ledger().len(), 1);

        // PARENT-holder arm: node 7 holds System(1); it demands the planet UNDER it — no mismatch
        // (the child's own head does not exist yet; that is exactly the warm-ahead spin-up shape).
        let mut parent = dir();
        grant(&mut parent, RealmId::System(1), 7, 1);
        let rlm = run_with_dir(
            parent,
            vec![demand_frame_from(
                &planet_in_sys1(4),
                DemandVerb::SpinUp,
                100,
                1,
                NodeId(7),
            )],
        );
        assert_eq!(
            rlm.demand_sender_mismatch, 0,
            "the parent's holder is lawful"
        );
        assert_eq!(rlm.ledger().len(), 1);
    }

    #[test]
    fn a_mismatched_sender_is_counted_and_the_demand_is_still_honored() {
        // BOTH heads resolve — to node 8 — and node 7 demands: a mismatch, counted + warned, and the
        // demand folds into the ledger UNCHANGED (observe mode: the warm-ahead path never eats a
        // blind window; enforcement waits for cloud mTLS).
        let mut d = dir();
        grant(&mut d, RealmId::System(1), 8, 1);
        grant(&mut d, RealmId::Planet(4), 8, 1);
        let rlm = run_with_dir(
            d,
            vec![demand_frame_from(
                &planet_in_sys1(4),
                DemandVerb::SpinUp,
                100,
                1,
                NodeId(7),
            )],
        );
        assert_eq!(rlm.demand_sender_mismatch, 1, "the stranger is measured");
        assert_eq!(
            rlm.ledger()
                .get(planet_in_sys1(4).path())
                .expect("the demand was honored despite the mismatch")
                .last_demand_tick,
            UniverseTick(100)
        );
    }

    #[test]
    fn an_unresolvable_head_is_counted_and_the_demand_is_still_honored() {
        // NO head resolves (empty directory — e.g. a rebuilt orchestrator before re-grants): counted
        // as unresolvable, and the demand is still honored (the same observe-only rule).
        let rlm = run(vec![demand_frame(&sys(7), DemandVerb::SpinUp, 100, 1)]);
        assert_eq!(
            rlm.demand_sender_mismatch, 1,
            "unresolvable is measured too"
        );
        assert_eq!(
            rlm.ledger()
                .get(sys(7).path())
                .expect("the demand was honored despite no resolvable head")
                .last_demand_tick,
            UniverseTick(100)
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

    /// A spawner that always refuses — for the launch-failure backoff test and for the reparent whose
    /// launch record cannot be rewritten.
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
        fn reparent_realm(&self, _n: NodeId, _c: &RealmCoord) -> Result<(), SpawnError> {
            Err(SpawnError::LaunchFailed {
                reason: "a refusing spawner keeps no launch record".into(),
            })
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
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(100), &BTreeSet::new());
        assert_eq!(rlm.spins_requested, 1);
        assert_eq!(
            rlm.launches.minted.get(sys(7).path()).map(BTreeMap::len),
            Some(1)
        );
        // Next sweep: launching (minted, no head) ⇒ NO second spawn (BUG-B).
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(101), &BTreeSet::new());
        assert_eq!(rlm.spins_requested, 1, "no double-spawn while launching");
        // RLM 5d C-1 idempotency-by-coord.path: a genuinely RE-ISSUED SpinUp on the still-launching coord
        // is ALSO a no-op. The suppression is keyed on the minted node under `sys(7).path()` (the lineage
        // path), NOT on the incarnation — the allocator WOULD mint a fresh NodeId/cookie, but `launch_present`
        // for the path blocks the emit. So a re-issued demand can never orphan a second incarnation.
        rlm.ledger
            .record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(102), Fence(2));
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(102), &BTreeSet::new());
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
    fn ingest_reconcile_drive_5f3a_bootstrap_ride_spins_the_whole_home_lineage() {
        // RLM 5f-3a — the OQ-4 "ride" contract (Layer-2, ingest→reconcile→drive). ONE synthetic RealmDemand
        // for the player's DEEPEST home realm — the Area-A box at x=25, resolved by the REAL 5f-2 resolver
        // `container_coord_at` to the 5-level lineage [Universe(0), Galaxy(1), System(7), Planet(7),
        // Area(7)] — injected through the UNCHANGED source-agnostic `record_realm_demands` ingress, must
        // spin up the WHOLE ancestor chain in ONE reconcile sweep, STRUCTURALLY via
        // ledger→reconcile→ancestor_close→spawn. ANTI-BYPASS: this test ONLY ever pushes a Wire{Saga} demand
        // frame (`demand_frame` → `record_realm_demands`); it NEVER calls `spawn_realm` / a direct spawn
        // hook, so passing proves the seed rode the ONE machinery.
        use RealmKindTag::{Area, Galaxy, Planet, System, Universe};
        use vd_core::pose::RealmId;
        use vd_core::worldgen::coord_of_realm;
        use vd_physics::worldgen::realm_regions_for;

        // NAMED in the FIXTURE forest: the deepest realm there is an AREA, which players build and the
        // generator never makes. Naming a generated realm instead would give a 3-level chain and the test
        // would claim to prove a 5-level one. The realm used to be said as the position (25,0,0) and
        // descended for; the descent is gone, so the test names what it always meant.
        let deepest = coord_of_realm(&realm_regions_for(0), RealmId::Area(7))
            .expect("the fixture forest holds the deep area home");
        let lin = |levels: &[(RealmKindTag, u64)]| -> RealmPath {
            RealmPath::from_levels(levels.iter().map(|&(k, s)| RealmLevel::new(k, s)).collect())
        };
        let expected: std::collections::BTreeSet<RealmPath> = [
            lin(&[(Universe, 0)]),
            lin(&[(Universe, 0), (Galaxy, 1)]),
            lin(&[(Universe, 0), (Galaxy, 1), (System, 7)]),
            lin(&[(Universe, 0), (Galaxy, 1), (System, 7), (Planet, 7)]),
            lin(&[
                (Universe, 0),
                (Galaxy, 1),
                (System, 7),
                (Planet, 7),
                (Area, 7),
            ]),
        ]
        .into_iter()
        .collect();

        // Ingest ONE Wire{Saga} demand for the deepest home (build the World exactly like `run`), then
        // reconcile+drive at tick 100 (NONZERO — a post-ClockSync tick).
        let mut rlm = run(vec![demand_frame(&deepest, DemandVerb::SpinUp, 100, 1)]);
        let mut d = dir();
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(100), &BTreeSet::new());
        assert_eq!(
            rlm.spins_requested, 5,
            "one leaf demand spins the whole 5-level home lineage in one sweep"
        );
        assert_eq!(rlm.undecodable_demands, 0);
        let minted: std::collections::BTreeSet<RealmPath> =
            rlm.launches.minted.keys().cloned().collect();
        assert_eq!(
            minted, expected,
            "minted EXACTLY the whole Universe→Area ancestor chain — nothing missing, nothing extra"
        );

        // A SECOND sweep at the next tick is a NO-OP: the chain is launching (minted, no head) ⇒ nothing
        // re-spawns (idempotent, launching-not-double-spawned).
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(101), &BTreeSet::new());
        assert_eq!(
            rlm.spins_requested, 5,
            "second sweep is idempotent — a launching chain is not re-spawned"
        );

        // Control — the fence is CARRIED, not AUTHORISING: a re-injected demand for the SAME home with a
        // DIFFERENT `parent_fence` yields the byte-identical spawn set (reconcile never reads `last_fence`
        // for a decision, rlm.rs ~206-208). spins_requested stays 5.
        let mut rlm = ingest_into(
            rlm,
            vec![demand_frame(&deepest, DemandVerb::SpinUp, 102, 999)],
        );
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(102), &BTreeSet::new());
        assert_eq!(
            rlm.spins_requested, 5,
            "a different parent_fence changes nothing — the fence is carried, never authorising"
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
        seeded.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(100), &BTreeSet::new());
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
        cold.reconcile_and_drive(&mut d2, &|_n| false, UniverseTick(100), &BTreeSet::new());
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
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(100), &BTreeSet::new());
        assert_eq!(
            rlm.launches.minted.get(sys(7).path()).map(BTreeMap::len),
            Some(1)
        );
        // The shard self-grants its head ⇒ the launch reconcile drains the minted node.
        grant(&mut d, RealmId::System(7), 50, 1);
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(102), &BTreeSet::new());
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
        rlm.reconcile_and_drive(&mut d, &|_n| false, now, &BTreeSet::new());
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
        rlm.reconcile_and_drive(
            &mut d,
            &|n| n == NodeId(50),
            UniverseTick(100),
            &BTreeSet::new(),
        );
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
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(101), &BTreeSet::new());
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
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(100), &BTreeSet::new());
        assert_eq!(rlm.spins_requested, 1);
        assert_eq!(rlm.spins_failed, 1);
        assert_eq!(
            rlm.launches.fail_streak.get(sys(7).path()).copied(),
            Some(1)
        );
        // Sweep 2 at now = 100 + cooldown: eligible per reconcile, but within the 2×cooldown backoff ⇒ skip.
        rlm.reconcile_and_drive(
            &mut d,
            &|_n| false,
            UniverseTick(100 + cd),
            &BTreeSet::new(),
        );
        assert_eq!(rlm.spins_requested, 1, "still backing off ⇒ no 2nd attempt");
        // Sweep 3 at now = 100 + 2×cooldown: past the backoff ⇒ retry (fails again).
        rlm.reconcile_and_drive(
            &mut d,
            &|_n| false,
            UniverseTick(100 + 2 * cd),
            &BTreeSet::new(),
        );
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
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(100), &BTreeSet::new());
        assert!(
            rlm.launches.minted.contains_key(sys(7).path()),
            "a fresh launch is held through its TTL grace"
        );
        // Sweep past the TTL, still no head + never live ⇒ a silent-async-failure is dropped so a fresh
        // SpinUp re-fires next sweep (BUG-B silent-failure self-heal).
        rlm.reconcile_and_drive(
            &mut d,
            &|_n| false,
            UniverseTick(100 + ttl),
            &BTreeSet::new(),
        );
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
        rlm.reconcile_and_drive(&mut d, &|_n| false, UniverseTick(100), &BTreeSet::new());
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
        rlm.reconcile_and_drive(&mut d, &|_n| false, now, &BTreeSet::new());
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
        world.insert_resource(crate::orchestrator::DynamicClockPeers::default());
        world.insert_resource(vd_sim::runtime::OutboundBox::default());
        world.insert_resource(rlm);
        world
    }

    #[test]
    fn a_committed_move_is_told_to_every_session_gateway_and_an_unresolved_one_is_not() {
        // The ruler switch, slice 5: the hull's cell moves under System 42 and every gateway holding a
        // session hears the hull's new coord; a move the ledger cannot resolve tells nobody.
        let mut rlm = RlmReconcilerRes::new(RlmTuning::default(), spawner());
        let hull_entity = vd_core::EntityId::pack(vd_core::entity_kind::EntityKind::Ship, 1, 1, 0);
        let hull = RealmId::Ship(hull_entity);
        let old = sys(41).child(vd_core::worldgen::level_of(hull));
        rlm.ledger
            .record_demand(&sys(41), DemandVerb::SpinUp, UniverseTick(1), Fence(1));
        rlm.ledger
            .record_demand(&sys(42), DemandVerb::SpinUp, UniverseTick(1), Fence(1));
        rlm.ledger
            .record_demand(&old, DemandVerb::SpinUp, UniverseTick(5), Fence(1));
        let mut world = lifecycle_world(rlm, 9);
        {
            let dir = &mut world.resource_mut::<crate::orchestrator::DirectoryRes>().0;
            let _ = dir.grant(
                DirectoryKey::Realm(hull),
                vd_wire::seams::directory::AuthorityRef::Shard(NodeId(1_007)),
                Fence(1),
                UniverseTick(1),
            );
            let _ = dir.grant(
                DirectoryKey::Session(vd_core::SessionId(1)),
                vd_wire::seams::directory::AuthorityRef::Gateway(NodeId(5)),
                Fence(1),
                UniverseTick(1),
            );
        }
        world
            .resource_mut::<crate::saga_runtime::SagaRuntimeRes>()
            .pending_reparents
            .extend([
                (hull, sys(42).lowered(), NodeId(1_042)),
                (hull, sys(43).lowered(), NodeId(1_043)),
            ]);
        let _ = run_lifecycle(&mut world);
        let out = &world.resource::<vd_sim::runtime::OutboundBox>().0;
        assert_eq!(
            out.len(),
            1,
            "one statement, to the one gateway holding a session"
        );
        let (to, class, bytes, durability) = &out[0];
        assert_eq!(*to, NodeId(5));
        assert_eq!(*class, MsgClass::Saga);
        assert_eq!(*durability, vd_sim::io::Durability::Retained);
        assert_eq!(
            postcard::from_bytes::<InterShardFlow>(bytes),
            Ok(InterShardFlow::ExteriorMoved(
                vd_wire::intershard::ExteriorMoved {
                    child: sys(42).child(vd_core::worldgen::level_of(hull)),
                    parent_node: NodeId(1_042),
                    at: UniverseTick(9),
                }
            ))
        );
        let rlm = world.resource::<RlmReconcilerRes>();
        assert_eq!(rlm.exterior_moves_told, 1);
        assert_eq!(rlm.reparents_applied, 1);
        assert_eq!(
            rlm.reparents_unresolved, 1,
            "System 43 has no cell: nothing told"
        );
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
        // RLM 5f-4c: the inert path returns BEFORE the publish ⇒ DynamicClockPeers stays empty
        // (byte-identical — the clock broadcast reaches only the static peers).
        assert!(
            world
                .resource::<crate::orchestrator::DynamicClockPeers>()
                .0
                .is_empty(),
            "inert reconciler publishes no dynamic clock peers"
        );
    }

    #[test]
    fn reconcile_system_republishes_the_live_spawned_roster_as_clock_peers() {
        // RLM 5f-4c (the blocker fix): an armed sweep that spins a realm up must publish the spawned node
        // into DynamicClockPeers, so the next clock broadcast reaches it — without ClockSync a spawned shard
        // gates off all authoring and would be mute. Published SET == the spawner's live roster EXACTLY
        // (assignment, not merge ⇒ a reaped node drops next sweep; no unbounded growth at 100K).
        let mut rlm = RlmReconcilerRes::new(RlmTuning::cloud(20), spawner());
        rlm.ledger
            .record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        let mut world = lifecycle_world(rlm, 100);
        assert!(
            world
                .resource::<crate::orchestrator::DynamicClockPeers>()
                .0
                .is_empty(),
            "empty before the first sweep"
        );
        assert_eq!(run_lifecycle(&mut world), 1, "the demanded realm spins up");
        let published = world
            .resource::<crate::orchestrator::DynamicClockPeers>()
            .0
            .clone();
        let live = world.resource::<RlmReconcilerRes>().live_nodes();
        assert_eq!(
            published, live,
            "DynamicClockPeers mirrors the live roster exactly"
        );
        assert!(
            !published.is_empty(),
            "the spawned node is now a clock peer"
        );
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
    fn boot_ticks_observed_max_records_the_launch_to_head_latency_monotonically() {
        use vd_wire::seams::directory::AuthorityRef;
        let mut rlm = RlmReconcilerRes::new(RlmTuning::cloud(20), spawner());
        let mut dir = DirectoryCore::new(DirectoryTuning::default());
        let dead = |_n: NodeId| false;
        // Demand + reconcile at tick 100 ⇒ sys(7) spins up (a node minted AT 100), no head yet ⇒ boot 0.
        rlm.ledger
            .record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        rlm.reconcile_and_drive(&mut dir, &dead, UniverseTick(100), &BTreeSet::new());
        assert_eq!(
            rlm.boot_ticks_observed_max(),
            0,
            "no head up yet ⇒ no boot measured"
        );
        // The realm head appears (the shard booted); reconcile at 137 ⇒ launch(100)→head(137) = 37 ticks.
        let _ = dir.grant(
            DirectoryKey::Realm(RealmId::System(7)),
            AuthorityRef::Shard(NodeId(50)),
            Fence(1),
            UniverseTick(137),
        );
        rlm.reconcile_and_drive(&mut dir, &dead, UniverseTick(137), &BTreeSet::new());
        assert_eq!(
            rlm.boot_ticks_observed_max(),
            37,
            "launch(100)→head(137) = 37 ticks"
        );
        // A LATER, SHORTER boot does NOT lower the monotone max (the `.max()` keep arm).
        rlm.ledger
            .record_demand(&sys(8), DemandVerb::SpinUp, UniverseTick(200), Fence(1));
        rlm.reconcile_and_drive(&mut dir, &dead, UniverseTick(200), &BTreeSet::new());
        let _ = dir.grant(
            DirectoryKey::Realm(RealmId::System(8)),
            AuthorityRef::Shard(NodeId(51)),
            Fence(1),
            UniverseTick(203),
        );
        rlm.reconcile_and_drive(&mut dir, &dead, UniverseTick(203), &BTreeSet::new());
        assert_eq!(
            rlm.boot_ticks_observed_max(),
            37,
            "a 3-tick boot does not lower the 37-tick max (monotone)"
        );
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
