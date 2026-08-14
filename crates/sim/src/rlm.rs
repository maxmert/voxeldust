//! RLM Step 3 — the PURE orchestrator realm-lifecycle decision kernel (`docs/design` RLM; spec
//! `scripts/rlm_step3_reconciler_spec.md`). NO ECS, NO I/O, NO wall clock, NO `RealmSpawner` — every
//! function here is a deterministic `f(ledger, resolved facts, tuning, tick)`, unit-testable without a
//! `World`, so the whole decision surface is per-monomorphization coverable in this crate (HR5).
//!
//! The reconciler is the SOLE realm spin-up/tear-down authority (RLM Step 2's shard-side loop NEVER emits
//! a kill — `stub.rs::aoi_transition`). It is LEVEL-TRIGGERED: [`reconcile`] re-derives the full desired
//! set every tick from the re-asserted (`ReDriven`) demands + the durable directory, so a dropped demand
//! or a missed teardown self-heals on the next sweep. The decision is a straight-line shim; ALL branching
//! lives in the monomorphic predicate helpers below, each using bitwise `&`/`|` (no short-circuit
//! false-arm to leave uncovered — mirrors `saga_runtime::should_reap`).
//!
//! Slice 3a (this file's first half) is the pure kernel: tuning, the demand ledger + fold, and the leaf
//! predicates. They take RESOLVED facts (a looked-up [`OwnerRecord`], booleans) as inputs; the runtime
//! (`vd-node::rlm_runtime`, slice 3c/3d) resolves those from the real `DirectoryCore` + spawner + liveness
//! and feeds them in. Slice 3b adds [`reconcile`] + ancestor-closure on top.

use std::collections::{BTreeMap, BTreeSet};

use vd_core::pose::RealmId;
use vd_core::realm_coord::RealmCoord;
use vd_core::realm_path::RealmPath;
use vd_core::{Fence, NodeId, UniverseTick};
use vd_wire::intershard::DemandVerb;
use vd_wire::seams::directory::{DirectoryKey, OwnerRecord};

use crate::directory::DirectoryCore;

// ===== 1. TUNING (§2.1) ===========================================================================

/// The RLM reconciler's timing budget (operational params, NEVER inline literals). DEFAULT is fully
/// INERT (all zero): with `reconcile_interval_ticks == 0` the runtime never sweeps, so a shard/harness
/// that does not opt in is byte-identical (the walk/canonical gate). [`RlmTuning::cloud`] derives a live
/// budget from the tick rate; [`RlmTuning::validate`] fails LOUD on a mis-ordered budget at boot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RlmTuning {
    /// A realm stays demand-desired for this many ticks after its last SpinUp/KeepAlive/spawn demand.
    pub demand_ttl_ticks: u64,
    /// How recent a realm's `Empty` self-report must be to count as confirmed-empty (tolerates a dropped
    /// `Empty` datagram — the realm may skip one report without being mistaken for crashed).
    pub empty_grace_ticks: u64,
    /// A LIVE realm must remain continuously teardown-eligible for this cooldown before a kill can fire.
    pub teardown_cooldown_ticks: u64,
    /// BUG-C veto window: once teardown-ready, a realm enters `Draining` for this many ticks during which
    /// ANY demand aborts the kill. The kill is NEVER synchronous with the decision.
    pub teardown_drain_ticks: u64,
    /// A just-spun realm cannot be re-spun for this cooldown (spin-up hysteresis / backoff base).
    pub spinup_cooldown_ticks: u64,
    /// A minted-but-unleased launch is considered live for this long before `SpinUp` re-fires.
    pub launch_ttl_ticks: u64,
    /// How often (ticks) the reconciler sweeps. `0` = INERT (never sweeps; the byte-identical default).
    pub reconcile_interval_ticks: u64,
    /// RLM Step 4a — the crash-recovery FREEZE window: after a kill-9 rebuild (empty RAM ledger), teardown
    /// is blocked for this many ticks past the recovered ceiling so a re-asserted demand can re-accrue
    /// before a realm can be reaped. Sized from the DEMAND re-accrue cadence (≈ `demand_ttl_ticks` — a realm
    /// must survive at least one full demand-TTL of post-restart silence so a parent's `KeepAlive` breaks
    /// it). Its OWN knob (NOT the liveness/directory grace, which is sized for the dead-peer renewal
    /// cadence, a different clock). `0` in the inert default (nothing to freeze).
    pub recovery_grace_ticks: u64,
    /// THE LEAK BOUND on the arrival shield: how long ONE pending hand-off may hold its destination
    /// realm alive before the shield lifts LOUDLY and the realm becomes reapable again. `0` = the shield
    /// is DISARMED (the inert default, and the honest reading of a budget nobody derived).
    ///
    /// This is NOT the safety mechanism — membership in the live hand-off set is, and that is zero ticks
    /// wide by construction. The cap exists only so that a hand-off which is wedged rather than slow can
    /// never pin a realm alive forever, which in a system meant for 100k players is one permanently
    /// leaked realm per wedged hand-off. Derived, never a literal: see [`derive_arrival_shield_ticks`].
    /// Left `0` by [`RlmTuning::cloud`] because the derivation needs the deployment's SAGA budget too;
    /// the orchestrator fills it in at boot, where both budgets are in the same room.
    pub arrival_shield_ticks: u64,
}

/// The number of post-commit phases a hand-off can DWELL in on its way to terminal, each bounded by one
/// `redrive_deadline_ticks`. Durable: `Swapping → Demoting → Promoting → Releasing`. Transient: the four
/// `BatchHandoff` phases. Pinned by `post_commit_dwell_count_matches_the_state_machine` in `saga.rs`, so
/// adding a post-commit phase fails that test until this constant is re-decided.
pub const POST_COMMIT_STEPS: u64 = 4;

/// THE ARRIVAL-SHIELD BUDGET, derived from the two budgets that actually bound a hand-off — never a
/// literal. It must comfortably outlast the worst HEALTHY post-commit completion, which is one
/// destructive-abort budget plus one cheap re-drive per post-commit phase; and it must be longer than the
/// reclaim window it is protecting against, or it could expire inside the very drain it exists to
/// survive. `+ 1` makes that strict.
#[must_use]
pub fn derive_arrival_shield_ticks(rlm: &RlmTuning, saga: &crate::saga::SagaTuning) -> u64 {
    let saga_budget = saga
        .abort_deadline_ticks
        .saturating_add(POST_COMMIT_STEPS.saturating_mul(saga.redrive_deadline_ticks));
    let reclaim_floor = rlm
        .teardown_drain_ticks
        .saturating_add(rlm.teardown_cooldown_ticks)
        .saturating_add(1);
    saga_budget.max(reclaim_floor)
}

impl Default for RlmTuning {
    /// Fully INERT — every field zero. `reconcile_interval_ticks == 0` ⇒ the runtime never sweeps ⇒
    /// behaviour-identical to a build with no reconciler.
    fn default() -> RlmTuning {
        RlmTuning {
            demand_ttl_ticks: 0,
            empty_grace_ticks: 0,
            teardown_cooldown_ticks: 0,
            teardown_drain_ticks: 0,
            spinup_cooldown_ticks: 0,
            launch_ttl_ticks: 0,
            reconcile_interval_ticks: 0,
            recovery_grace_ticks: 0,
            arrival_shield_ticks: 0,
        }
    }
}

impl RlmTuning {
    /// A live cloud budget derived from the tick rate (mirrors `LivenessTuning::cloud` / `DirectoryTuning`
    /// deriving from `tick_hz`, no magic literals in systems). The ordering invariant
    /// (`demand_ttl > drain + cooldown`, `empty_grace >= 1`, all non-zero) holds by construction — asserted
    /// by [`validate`](RlmTuning::validate) at boot. `hz` = ticks per second; the windows are seconds-scaled.
    /// The boot-AGNOSTIC form (launch-TTL defaults to ~3s); DRY — it is [`cloud_with_boot`] with an
    /// unmeasured boot (`boot_ticks_p99 = settle = 0`), so `max(hz*3, 0) = hz*3` (byte-identical to before).
    #[must_use]
    pub fn cloud(tick_hz: u32) -> RlmTuning {
        RlmTuning::cloud_with_boot(tick_hz, 0, 0)
    }

    /// A live cloud budget whose launch-TTL is FLOORED by the MEASURED process boot latency (RLM 5f): a
    /// minted-but-unleased launch counts as "still booting" for at least `boot_ticks_p99 + settle`, so a
    /// slow real fork (a ~3s CI boot) is NEVER re-spun mid-boot — which would double-spawn = thrash. The
    /// floor is `max(hz*3, boot_ticks_p99 + settle)`, so the boot-agnostic ~3s default still applies when
    /// boot is unmeasured (0). All other windows are seconds-scaled exactly as the boot-agnostic budget; the
    /// ordering invariant holds by construction (validate-clean). Because `min_dwell = spinup_cooldown +
    /// launch_ttl`, flooring launch_ttl AUTOMATICALLY makes `min_dwell >= boot_ticks_p99 + settle` — a realm
    /// is never teardown-eligible before it finishes booting (the 5f coupling). `hz` = ticks per second.
    #[must_use]
    pub fn cloud_with_boot(tick_hz: u32, boot_ticks_p99: u64, settle: u64) -> RlmTuning {
        let hz = u64::from(tick_hz).max(1);
        // ~1s demand freshness, ~0.5s empty-report tolerance, a drain + cooldown that together stay well
        // inside the demand TTL so a re-asserted demand always aborts a drain before a kill fires.
        let drain = (hz / 2).max(1);
        let cooldown = hz.max(1);
        let spinup_cooldown = hz.max(1);
        // The boot floor: at least ~3s OR the measured boot+settle, whichever is larger (the DERIVED-from-
        // measured-boot fix — a fixed hz*3 would re-fire SpinUp mid-boot for a slower real fork).
        let launch_ttl = (hz * 3).max(boot_ticks_p99.saturating_add(settle));
        let demand_ttl = (hz * 4).max(drain + cooldown + 1);
        RlmTuning {
            demand_ttl_ticks: demand_ttl,
            empty_grace_ticks: (hz / 2).max(1),
            teardown_cooldown_ticks: cooldown,
            teardown_drain_ticks: drain,
            spinup_cooldown_ticks: spinup_cooldown,
            launch_ttl_ticks: launch_ttl,
            reconcile_interval_ticks: 1,
            // The post-crash freeze spans one full demand cadence: a rebuilt orchestrator with an empty RAM
            // ledger must let a parent's `KeepAlive` (or a login demand) re-accrue before any realm can be
            // reaped, so we hold teardown for exactly one demand TTL past the recovered ceiling.
            recovery_grace_ticks: demand_ttl,
            // Filled in at boot by `derive_arrival_shield_ticks`, which needs the SAGA budget too. `0`
            // here means the shield is disarmed, which is the honest state of a budget nobody derived.
            arrival_shield_ticks: 0,
        }
    }

    /// The composite spin-up floor: a just-spun realm cannot be reaped inside its own boot+settle window
    /// (`spinup_cooldown + launch_ttl`) even if it reports `Empty` immediately — a booted-empty realm must
    /// survive at least one demand cadence so a warping-in occupant's demand can land (thrash-heed).
    #[must_use]
    pub fn min_dwell_ticks(&self) -> u64 {
        self.spinup_cooldown_ticks
            .saturating_add(self.launch_ttl_ticks)
    }

    /// Reject a mis-tuned budget at boot (fail-loud like `DirectoryTuning::validate`). ALL checks are gated
    /// on the reconciler being ACTIVE (`reconcile_interval_ticks != 0`): when INERT the budget is never
    /// read, so a default-zero build is vacuously valid. Bitwise `&` keeps both operands covered (HR5).
    ///
    /// # Errors
    /// [`RlmTuningError`] when active AND any window is zero, or the demand TTL does not strictly outlast
    /// the drain+cooldown (a demand must be able to abort a drain before the kill window opens — the
    /// thrash-heed inequality the co-hosting limit-cycle needed).
    pub fn validate(&self) -> Result<(), RlmTuningError> {
        let active = self.reconcile_interval_ticks != 0;
        let any_zero = (self.demand_ttl_ticks == 0)
            | (self.empty_grace_ticks == 0)
            | (self.teardown_cooldown_ticks == 0)
            | (self.teardown_drain_ticks == 0)
            | (self.spinup_cooldown_ticks == 0)
            | (self.launch_ttl_ticks == 0)
            | (self.recovery_grace_ticks == 0);
        if active & any_zero {
            return Err(RlmTuningError::ZeroWindowWhileActive);
        }
        let reclaim_floor = self
            .teardown_drain_ticks
            .saturating_add(self.teardown_cooldown_ticks);
        if active & (self.demand_ttl_ticks <= reclaim_floor) {
            return Err(RlmTuningError::DemandTtlRacesReclaim {
                demand_ttl: self.demand_ttl_ticks,
                reclaim_floor,
            });
        }
        Ok(())
    }
}

/// A mis-ordered [`RlmTuning`] — rejected LOUD at boot so a deployment never runs a thrash-prone budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RlmTuningError {
    /// An ACTIVE reconciler with a zero window (any of the seven timing fields) — every window must be > 0.
    #[error("active RlmTuning has a zero timing window")]
    ZeroWindowWhileActive,
    /// The demand TTL does not strictly outlast the drain+cooldown reclaim floor, so a re-asserted demand
    /// could arrive too late to abort a drain — the thrash-heed inequality.
    #[error(
        "demand_ttl {demand_ttl} must strictly exceed the drain+cooldown reclaim floor {reclaim_floor}"
    )]
    DemandTtlRacesReclaim { demand_ttl: u64, reclaim_floor: u64 },
}

// ===== 2. THE DEMAND LEDGER (§2.2) ================================================================

/// One realm's demand-lifecycle bookkeeping, keyed in [`DemandLedger`] by the collision-free lineage
/// [`RealmPath`] (NEVER the lossy `lowered()` `RealmId`). All ticks are `UniverseTick`; a zero
/// `last_demand_tick` is the "never demanded" sentinel (demands arrive post-`ClockSync`, tick > 0).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LedgerCell {
    /// The realm's FULL lineage coord — for spawn (`profile_kind`) and the ancestor walk.
    pub coord: RealmCoord,
    /// Freshest SpinUp | KeepAlive | player-spawn demand tick (`Empty` does NOT refresh it). The `max`
    /// of every non-Empty demand seen ⇒ ORDER-INDEPENDENT (a reordered inbox yields the same value).
    pub last_demand_tick: UniverseTick,
    /// Most recent `Empty` self-report (the `max` of every `Empty` tick ⇒ ORDER-INDEPENDENT). Drives BOTH
    /// clauses of [`empty_confirmed`]: freshness (`now - last_empty <= grace`) AND the arm-C
    /// no-demand-since (`last_demand < last_empty`). Keying arm-C on this `max` (not a streak-start) is
    /// what makes the reap decision deterministic when a parent's `KeepAlive` and the child's `Empty` land
    /// on the SAME coord in the SAME tick from different sources — their arrival order must NOT change
    /// whether the realm is reaped (the crash-replay proptest's INV-FOLD-ORDER caught the streak-based bug).
    pub last_empty_tick: Option<UniverseTick>,
    /// Last tick a `SpinUp` intent was emitted for this realm (min-dwell + spin-up cooldown anchor).
    pub spawn_watermark: UniverseTick,
    /// Last tick a `Kill` executed for this realm (teardown cooldown anchor).
    pub teardown_watermark: UniverseTick,
    /// BUG-C: the tick teardown-readiness first became continuous. `None` = not draining.
    pub draining_since: Option<UniverseTick>,
    /// The parent_fence of the lexicographically-freshest `(tick, fence)` demand (audit; Step-4 auth). NOT
    /// read by any Step-3 decision — the `Kill`/revoke uses the HEAD's fence, never this.
    pub last_fence: Fence,
}

impl LedgerCell {
    /// A fresh cell for `coord` with no demands yet (all ticks zero, no empty report).
    #[must_use]
    fn empty(coord: RealmCoord) -> LedgerCell {
        LedgerCell {
            coord,
            last_demand_tick: UniverseTick(0),
            last_empty_tick: None,
            spawn_watermark: UniverseTick(0),
            teardown_watermark: UniverseTick(0),
            draining_since: None,
            last_fence: Fence(0),
        }
    }
}

/// The per-realm demand ledger (RAM-only; self-heals on restart via re-asserted (`ReDriven`, every tick)
/// demands within ~1 demand round-trip). The durable `StoreKey::Rlm` snapshot is DEFERRED (`DEFERRED.md`
/// D-RLM-2): the crash-replay proptest (`mod crash_replay_proptest`) + the armed Step-4a crash-recovery
/// freeze (`RlmTuning::recovery_grace_ticks`) are the machine-checked evidence the RAM self-heal + freeze
/// suffice (INV-SNAPSHOT-EQUIVALENCE: the RAM-heal and durable-snapshot arms reach identical reap-sets).
/// Keyed by the lineage [`RealmPath`] (`Ord`, collision-free) for determinism.
#[derive(Default, Debug)]
pub struct DemandLedger {
    cells: BTreeMap<RealmPath, LedgerCell>,
}

impl DemandLedger {
    /// Read a realm's cell (the pure predicates + the runtime read this).
    #[must_use]
    pub fn get(&self, path: &RealmPath) -> Option<&LedgerCell> {
        self.cells.get(path)
    }

    /// The cells, in `RealmPath` order (the reconciler's deterministic iteration source).
    pub fn iter(&self) -> impl Iterator<Item = (&RealmPath, &LedgerCell)> {
        self.cells.iter()
    }

    /// How many realms are tracked (bounded by `gc` — slice 3b).
    #[must_use]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// True when no realm is tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Fold ONE decoded demand into the ledger (the SOURCE-AGNOSTIC ingress — an AoI demand and a
    /// player-spawn demand are indistinguishable here). All branching is in the monomorphic helpers
    /// [`refresh_demand`]/[`latch_empty`]/[`update_fence`]. `TearDown` is a no-op (Step 2 never sends it;
    /// the reconciler is the sole kill authority).
    pub fn record_demand(
        &mut self,
        coord: &RealmCoord,
        verb: DemandVerb,
        tick: UniverseTick,
        fence: Fence,
    ) {
        let cell = self
            .cells
            .entry(coord.path().clone())
            .or_insert_with(|| LedgerCell::empty(coord.clone()));
        update_fence(cell, tick, fence);
        match verb {
            DemandVerb::SpinUp | DemandVerb::KeepAlive => refresh_demand(cell, tick),
            DemandVerb::Empty => latch_empty(cell, tick),
            DemandVerb::TearDown => {}
        }
    }

    /// Ensure a cell exists for `coord` and stamp its `spawn_watermark` (called by the EXECUTE step on a
    /// successful spawn — slice 3d). Creates an ancestor's cell the demand fold never touched.
    pub fn mark_spawned(&mut self, coord: &RealmCoord, tick: UniverseTick) {
        self.cells
            .entry(coord.path().clone())
            .or_insert_with(|| LedgerCell::empty(coord.clone()))
            .spawn_watermark = tick;
    }

    /// Stamp a realm's `teardown_watermark` on a Kill (slice 3d). No-op if the cell is gone.
    pub fn mark_reaped(&mut self, path: &RealmPath, tick: UniverseTick) {
        if let Some(cell) = self.cells.get_mut(path) {
            cell.teardown_watermark = tick;
        }
    }

    /// Apply the [`LedgerDelta`] `reconcile` returned (the `draining_since` mutations + retirements). Kept
    /// separate from the pure decision so `reconcile` stays immutable-in / testable (run-twice determinism).
    pub fn apply_delta(&mut self, delta: &LedgerDelta) {
        for (path, v) in &delta.set_draining {
            if let Some(cell) = self.cells.get_mut(path) {
                cell.draining_since = *v;
            }
        }
        for path in &delta.retire {
            self.cells.remove(path);
        }
    }
}

/// Refresh a non-Empty demand: bump `last_demand_tick` to the `max` seen. ORDER-INDEPENDENT — a stale
/// reordered SpinUp simply loses the `max` and never lowers it (no streak-clearing needed; `empty_confirmed`
/// compares `last_demand < last_empty`, both maxes, so a stale demand can't un-empty a fresher `Empty`).
fn refresh_demand(cell: &mut LedgerCell, tick: UniverseTick) {
    cell.last_demand_tick = cell.last_demand_tick.max(tick);
}

/// Latch an `Empty`: bump `last_empty_tick` to the `max` seen (ORDER-INDEPENDENT). A continuously-empty
/// realm re-reporting `Empty` keeps this fresh; one that stops lets it age past `grace` (⇒ the zombie /
/// not-confirmed path — the crashed-vs-empty disambiguation).
fn latch_empty(cell: &mut LedgerCell, tick: UniverseTick) {
    cell.last_empty_tick = Some(cell.last_empty_tick.map_or(tick, |t| t.max(tick)));
}

/// Keep the fence of the lexicographically-freshest `(tick, fence)` demand (audit only; order-independent
/// so the ledger state is deterministic regardless of inbox order). Monomorphic.
fn update_fence(cell: &mut LedgerCell, tick: UniverseTick, fence: Fence) {
    let seen = cell
        .last_demand_tick
        .max(cell.last_empty_tick.unwrap_or(UniverseTick(0)));
    if (tick, fence) >= (seen, cell.last_fence) {
        cell.last_fence = fence;
    }
}

// ===== 3. THE DECISION OUTPUT (§2.3) =============================================================

/// The pure decision output — a lifecycle intent the EXECUTE step (slice 3d) turns into a real spawn/kill.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LifecycleAction {
    /// Bring `coord`'s realm alive (a real cluster pod in prod; the mem twin in tests).
    SpinUp { coord: RealmCoord },
    /// Reclaim a LIVE realm's lease + pod (out of AoI, affirmatively empty, past drain).
    Kill {
        path: RealmPath,
        node: NodeId,
        fence: Fence,
    },
    /// BUG-A: a dead-but-recorded realm — force-revoke the stale head + kill the corpse (idempotent),
    /// then re-spawn next sweep if still demanded.
    ForceReap {
        path: RealmPath,
        node: NodeId,
        fence: Fence,
    },
}

/// The ledger mutations [`reconcile`] decides, applied by [`DemandLedger::apply_delta`] (kept out of the
/// pure decision so `reconcile` is immutable-in / run-twice-deterministic).
#[derive(Default, Debug, PartialEq, Eq)]
pub struct LedgerDelta {
    /// `path → new draining_since` (`Some(t)` = start draining at `t`; `None` = clear/rescue).
    pub set_draining: BTreeMap<RealmPath, Option<UniverseTick>>,
    /// Retired paths — a dead, undesired, un-launching cell to drop (keeps the ledger bounded).
    pub retire: BTreeSet<RealmPath>,
    /// REPORTED, NOT APPLIED (`apply_delta` ignores it): the realms a pending arrival is the SOLE
    /// reason to keep alive this sweep — nothing else demanded them and they are not otherwise live-
    /// and-occupied. Counting the shield BY NAME every time it is load-bearing is the answer to this
    /// subsystem's history of silent guarantees: a wedged hand-off shows up as a climbing number
    /// instead of as nothing at all. ZERO on every healthy sweep, including during a healthy crossing
    /// (the source's own keep-alive already covers the destination while its latch is held).
    pub arrival_shielded: BTreeSet<RealmPath>,
}

// ===== 4. THE LEAF PREDICATES (§1.4–1.6) — monomorphic, bitwise, take RESOLVED facts ==============

/// The resolved "actual" facts for one realm (looked up from the real `DirectoryCore`/spawner/liveness by
/// the runtime, slice 3d) — passed into [`teardown_ready`] so the predicate is pure + directly testable.
#[derive(Clone, Copy, Debug)]
pub struct TeardownFacts {
    /// A LIVE lease exists (head present AND owner not latched-dead).
    pub running_live: bool,
    /// This realm is in the ancestor-closed desired set (kept alive as a demand target OR an ancestor).
    pub desired_in_closure: bool,
    /// Some desired descendant keeps this realm alive as its frame-authority ancestor.
    pub has_desired_descendant: bool,
    /// The orchestrator-crash quiesce window has elapsed (`now >= rlm_quiesced_until`).
    pub quiesced: bool,
    // RETIRED (slice 9c): there used to be an `in_transfer` clause here, meaning "do not shut this
    // realm down, something is being handed over". Nothing in the running system ever set that flag on
    // a REALM — the transfer lock is taken on the SUBJECT's key, and the commit CAS clears even that —
    // so the clause was permanently true and shielded nothing, while the unit test that proved it
    // load-bearing hand-built a value the real system cannot produce. Protecting a landing realm is now
    // the arrival shield's job (`desired_alive`'s third arm), which is driven by the orchestrator's own
    // list of in-progress hand-offs. See DEFERRED D-RLM-15 for the realm-keyed lock that would need a
    // producer before anything like the old clause could return.
}

/// A realm has a LIVE lease: head present AND its owner is NOT latched-dead (BUG-A: a dead owner's lingering
/// head is a `zombie`, not `running_live`). `dead` is `is_latched_dead` passed as a trait object (one
/// monomorphization — no per-closure coverage debt).
#[must_use]
pub fn running_live(head: Option<OwnerRecord>, dead: &dyn Fn(NodeId) -> bool) -> bool {
    match head {
        Some(r) => !dead(r.authority.node()),
        None => false,
    }
}

/// A dead-but-recorded realm: head present AND its owner IS latched-dead (the crashed-shard-leak case).
#[must_use]
pub fn zombie(head: Option<OwnerRecord>, dead: &dyn Fn(NodeId) -> bool) -> bool {
    match head {
        Some(r) => dead(r.authority.node()),
        None => false,
    }
}

/// A minted-but-not-yet-leased realm: no head yet, but a live launch exists (BUG-B bridge — closes the
/// "spawned but head not self-granted" gap so the reconciler does not double-spawn).
#[must_use]
pub fn launching(head: Option<OwnerRecord>, launch_present: bool) -> bool {
    head.is_none() & launch_present
}

/// `(A)` of the desired-alive predicate — a REAL demand (SpinUp/KeepAlive/spawn) landed within the TTL.
/// `Empty` never refreshes `last_demand_tick`, so an empty realm ages out of arm A. The `0` sentinel (a
/// realm that has ONLY ever self-reported `Empty`, never been demanded) is NOT "demanded" — the raw
/// `now - 0 <= ttl` would be spuriously true at every early universe tick (`now <= ttl`), keeping a
/// never-demanded empty realm undead near genesis. Bitwise `&` (both operands covered, HR5).
#[must_use]
pub fn demanded_recently(cell: &LedgerCell, now: UniverseTick, ttl: u64) -> bool {
    (cell.last_demand_tick != UniverseTick(0))
        & (now.0.saturating_sub(cell.last_demand_tick.0) <= ttl)
}

/// The realm has AFFIRMATIVELY gone empty: a RECENT `Empty` report (`now - last_empty <= grace` — so a
/// continuously-empty realm re-asserting `Empty` every tick STAYS confirmed, while grace tolerates a single
/// dropped datagram) AND the realm's LATEST signal is that `Empty`, not a demand (arm C:
/// `last_demand_tick < last_empty_tick`). Both terms key on `max`-tracked ticks, so the result is
/// ORDER-INDEPENDENT: a parent's `KeepAlive` and the child's `Empty` on the SAME coord in the SAME tick from
/// different sources resolve to the SAME reap decision regardless of arrival order (a same-tick pair leaves
/// `last_demand == last_empty` ⇒ `<` is false ⇒ NOT confirmed ⇒ the realm is kept alive — the safe tie-break;
/// a stale reordered demand keeps `last_demand < last_empty` ⇒ still confirmed). A crashed shard STOPS
/// reporting ⇒ `last_empty_tick` ages past grace ⇒ NOT confirmed-empty (it takes the `zombie` path instead —
/// the dead-vs-empty disambiguation). Bitwise `&` (both operands covered, HR5).
#[must_use]
pub fn empty_confirmed(cell: &LedgerCell, now: UniverseTick, grace: u64) -> bool {
    match cell.last_empty_tick {
        Some(last) => {
            let fresh = now.0.saturating_sub(last.0) <= grace;
            let no_demand_since = cell.last_demand_tick < last;
            fresh & no_demand_since
        }
        None => false,
    }
}

/// THE desired-alive predicate (§1.5). Arm A = demand-driven (AoI OR player-spawn — source-agnostic). Arm
/// B = LIVE and not affirmatively-empty (the NO-STRAND core: a live realm only LEAVES desired by saying
/// `Empty`; demand-absence alone never un-desires it — a partitioned parent stops `KeepAlive`-ing but arm
/// B holds). Arm C = SOMEBODY IS ON THEIR WAY HERE (slice 9a): a hand-off the orchestrator itself is
/// driving names this realm as its destination.
///
/// Arm C is deliberately a reason to WANT the realm alive rather than a veto on tearing it down, and that
/// is the whole point: a want flows through the ancestor closure, so the destination's entire parent chain
/// is held up with it. A veto would have left the player landing on a planet whose star system had just
/// been shut down. It follows — argued explicitly, not hidden — that a pending arrival can also START a
/// realm whose shard has vanished, through the same fenced minting path as any other demand. That is the
/// correct self-heal for "your destination died while you were crossing", and it is a real action, not a
/// passive refusal.
///
/// Bitwise (no short-circuit false-arm, HR5). `running_live` and `arrival_pending` are resolved by caller.
#[must_use]
pub fn desired_alive(
    cell: &LedgerCell,
    now: UniverseTick,
    tuning: &RlmTuning,
    running_live: bool,
    arrival_pending: bool,
) -> bool {
    demanded_recently(cell, now, tuning.demand_ttl_ticks)
        | (running_live & !empty_confirmed(cell, now, tuning.empty_grace_ticks))
        | arrival_pending
}

/// THE teardown-ready predicate (§1.6) — necessary-but-not-sufficient (the kill fires only after the drain,
/// slice 3b). Every clause is a resolved boolean or a monomorphic arithmetic compare; bitwise `&` (HR5).
/// A realm is reap-eligible ONLY when it is live, out of the whole AoI closure, affirmatively empty, past
/// its boot+settle floor and its teardown cooldown, orphans no live child, and the
/// crash-quiesce window has elapsed.
#[must_use]
pub fn teardown_ready(
    cell: &LedgerCell,
    now: UniverseTick,
    tuning: &RlmTuning,
    facts: &TeardownFacts,
) -> bool {
    let past_dwell = now.0.saturating_sub(cell.spawn_watermark.0) >= tuning.min_dwell_ticks();
    let past_cooldown =
        now.0.saturating_sub(cell.teardown_watermark.0) >= tuning.teardown_cooldown_ticks;
    facts.running_live
        & !facts.desired_in_closure
        & empty_confirmed(cell, now, tuning.empty_grace_ticks)
        & past_dwell
        & past_cooldown
        & !facts.has_desired_descendant
        & facts.quiesced
}

// ===== 5. RECONCILE + ANCESTOR-CLOSURE + TWO-PHASE DRAIN (§3.1/§3.2/§1.6) =========================

/// The ancestor-closure of a desired set: every desired realm pulls its whole `Universe→…→P` parent chain
/// alive (each parent is its children's frame authority — no apartment without its building/planet/system).
/// Bounded by lineage depth (`parent()` → `None` at the root, `realm_coord.rs`); no cycle. Recomputed EVERY
/// tick from the FULL desired set (including arm-B running-occupied-but-undemanded realms), so a live child
/// can never strand its parent (critique #1).
#[must_use]
pub fn ancestor_close(
    desired: &BTreeMap<RealmPath, RealmCoord>,
) -> BTreeMap<RealmPath, RealmCoord> {
    let mut closed = desired.clone();
    for coord in desired.values() {
        let mut cur = coord.clone();
        while let Some(parent) = cur.parent() {
            closed
                .entry(parent.path().clone())
                .or_insert_with(|| parent.clone());
            cur = parent;
        }
    }
    closed
}

/// True iff some path in `closed` is a STRICT descendant of `path` (a longer path with `path` as a prefix)
/// — the teardown guard that never orphans a live child. SCALE (100k users): `closed` is a `BTreeMap`
/// sorted by lineage path, so `path`'s descendants form a contiguous range and its SMALLEST strict
/// successor is either the first descendant (if any) or a diverging sibling — so ONE `O(log n)` range probe
/// answers this, keeping the per-sweep teardown scan `O(realms · log realms)` instead of `O(realms²)`. (A
/// non-descendant can never sort between `path` and its first descendant: every `path`-prefixed key sorts
/// before any key that diverges earlier.)
#[must_use]
pub fn has_desired_descendant(path: &RealmPath, closed: &BTreeMap<RealmPath, RealmCoord>) -> bool {
    use std::ops::Bound;
    closed
        .range::<RealmPath, _>((Bound::Excluded(path.clone()), Bound::Unbounded))
        .next()
        .is_some_and(|(cand, _)| is_strict_descendant(cand, path))
}

fn is_strict_descendant(candidate: &RealmPath, ancestor: &RealmPath) -> bool {
    (candidate.levels().len() > ancestor.levels().len())
        & candidate.levels().starts_with(ancestor.levels())
}

/// A realm may be (re-)spun only past its spin-up cooldown. A realm that has never been spawn-ATTEMPTED is
/// eligible immediately: either it has no cell yet (an ancestor the demand fold never touched) OR its cell's
/// `spawn_watermark` is still the `0` sentinel (created by the demand fold, no spawn stamped). The sentinel
/// is load-bearing at EARLY universe ticks — `now - 0 >= cooldown` is FALSE while `now < cooldown`, which
/// would wrongly block a realm's very first spawn near genesis (a real orchestrator boots at tick ~1, not
/// `≫ cooldown`; the unit tests masked this by using `now=100`). Monomorphic bitwise (both arms covered).
fn spinup_eligible(cell: Option<&LedgerCell>, now: UniverseTick, tuning: &RlmTuning) -> bool {
    match cell {
        None => true,
        Some(c) => {
            (c.spawn_watermark == UniverseTick(0))
                | (now.0.saturating_sub(c.spawn_watermark.0) >= tuning.spinup_cooldown_ticks)
        }
    }
}

/// The two-phase teardown transition (BUG-C): a realm that becomes teardown-ready enters `Draining` for
/// `teardown_drain_ticks` — the veto window — and is killed ONLY if it stays ready through it. Returns
/// `(emit_kill, next_draining_since)`. The kill is NEVER synchronous with the decision, so any demand that
/// clears readiness within the window aborts it. Monomorphic (every arm a covered region, HR5).
fn drive_drain(
    cell: &LedgerCell,
    now: UniverseTick,
    tuning: &RlmTuning,
    ready: bool,
) -> (bool, Option<UniverseTick>) {
    match (ready, cell.draining_since) {
        (false, _) => (false, None), // not ready ⇒ not draining (clears a prior drain: the rescue)
        (true, None) => (false, Some(now)), // just became ready ⇒ open the drain window, NO kill
        (true, Some(since)) => {
            if now.0.saturating_sub(since.0) >= tuning.teardown_drain_ticks {
                (true, Some(since)) // drain elapsed + still ready ⇒ KILL
            } else {
                (false, Some(since)) // still draining
            }
        }
    }
}

/// THE reconcile decision (§3.1) — pure, deterministic, NO I/O. Reads the immutable ledger + the real
/// directory heads + a liveness oracle + the spawner's live-launch map, and returns the lifecycle actions
/// to execute plus the ledger delta to apply. Level-triggered: the full desired set is re-derived every
/// sweep, so a dropped demand or missed teardown self-heals. The whole body is a straight-line shim —
/// ALL branching is in the monomorphic predicates above.
///
/// Action order is deterministic: `ForceReap` then `SpinUp` (both ancestor-first, `RealmPath`-ascending)
/// then `Kill` (child-first, `RealmPath`-descending) — a building boots before its apartment; an apartment
/// is reaped before its building.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn reconcile(
    ledger: &DemandLedger,
    dir: &DirectoryCore,
    liveness_dead: &dyn Fn(NodeId) -> bool,
    launch_live: &BTreeMap<RealmPath, BTreeMap<NodeId, UniverseTick>>,
    tuning: &RlmTuning,
    now: UniverseTick,
    quiesced_until: UniverseTick,
    arriving: &BTreeSet<RealmId>,
) -> (Vec<LifecycleAction>, LedgerDelta) {
    // 1. The demand-driven / arm-B desired set (§1.5), then its ancestor-closure (§3.2).
    let mut delta = LedgerDelta::default();
    let mut desired: BTreeMap<RealmPath, RealmCoord> = BTreeMap::new();
    for (path, cell) in ledger.iter() {
        let head = dir.head(DirectoryKey::Realm(cell.coord.lowered()));
        let live = running_live(head, liveness_dead);
        let arrival = arriving.contains(&cell.coord.lowered());
        // Report the shield ONLY when the arrival is the SOLE reason this realm is wanted, so the
        // counter's healthy baseline is zero and a non-zero reading always means something.
        let otherwise_wanted = desired_alive(cell, now, tuning, live, false);
        if arrival & !otherwise_wanted {
            delta.arrival_shielded.insert(path.clone());
        }
        if desired_alive(cell, now, tuning, live, arrival) {
            desired.insert(path.clone(), cell.coord.clone());
        }
    }
    let closed = ancestor_close(&desired);

    let mut force_reaps: Vec<LifecycleAction> = Vec::new();
    let mut spinups: Vec<LifecycleAction> = Vec::new();
    let mut kills: Vec<LifecycleAction> = Vec::new();

    // 2. ForceReap every zombie (BUG-A) — a dead-but-recorded head — regardless of desire. The `_` arm
    // covers "no head" AND "live head", so the guarded arm is the only reap (no uncoverable region).
    for (path, cell) in ledger.iter() {
        match dir.head(DirectoryKey::Realm(cell.coord.lowered())) {
            Some(r) if liveness_dead(r.authority.node()) => {
                force_reaps.push(LifecycleAction::ForceReap {
                    path: path.clone(),
                    node: r.authority.node(),
                    fence: r.fence,
                });
            }
            _ => {}
        }
    }

    // 3. SpinUp every desired-closed realm that is neither running (a live head) nor launching, past its
    // spin-up cooldown. A zombie head is "present" here (not absent), so ForceReap runs first, then the
    // next sweep spins it up (self-heal).
    for (path, coord) in &closed {
        let head = dir.head(DirectoryKey::Realm(coord.lowered()));
        let launch_present = launch_live.get(path).is_some_and(|s| !s.is_empty());
        let absent = head.is_none() & !launch_present;
        if absent & spinup_eligible(ledger.get(path), now, tuning) {
            spinups.push(LifecycleAction::SpinUp {
                coord: coord.clone(),
            });
        }
    }

    // 4. Two-phase teardown of every LIVE realm + retire dead undesired cells (bounded ledger).
    for (path, cell) in ledger.iter() {
        let head = dir.head(DirectoryKey::Realm(cell.coord.lowered()));
        let launch_present = launch_live.get(path).is_some_and(|s| !s.is_empty());
        match head {
            Some(r) => {
                let facts = TeardownFacts {
                    running_live: !liveness_dead(r.authority.node()),
                    desired_in_closure: closed.contains_key(path),
                    has_desired_descendant: has_desired_descendant(path, &closed),
                    quiesced: now >= quiesced_until,
                };
                let ready = teardown_ready(cell, now, tuning, &facts);
                let (kill, next_draining) = drive_drain(cell, now, tuning, ready);
                if next_draining != cell.draining_since {
                    delta.set_draining.insert(path.clone(), next_draining);
                }
                if kill {
                    kills.push(LifecycleAction::Kill {
                        path: path.clone(),
                        node: r.authority.node(),
                        fence: r.fence,
                    });
                }
            }
            None => {
                // No head ⇒ not running ⇒ not teardown-ready ⇒ any open drain clears (cleanup/rescue).
                let (_kill, next_draining) = drive_drain(cell, now, tuning, false);
                if next_draining != cell.draining_since {
                    delta.set_draining.insert(path.clone(), next_draining);
                }
                // Retire a dead, undesired, un-launching cell so the ledger never grows without bound.
                if !launch_present & !closed.contains_key(path) {
                    delta.retire.insert(path.clone());
                }
            }
        }
    }

    // Deterministic assembly: ForceReap + SpinUp ancestor-first (path-ascending, as collected), Kill
    // child-first (reverse the path-ascending collection order).
    kills.reverse();
    let mut actions = force_reaps;
    actions.append(&mut spinups);
    actions.append(&mut kills);
    (actions, delta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_core::realm_path::{RealmKindTag, RealmLevel};
    use vd_wire::seams::directory::AuthorityRef;

    // ---- fixtures --------------------------------------------------------------------------------

    fn coord(kind: RealmKindTag, seed: u64) -> RealmCoord {
        RealmCoord::from_path(RealmPath::from_levels(vec![RealmLevel::new(kind, seed)]))
            .expect("one-level path has a leaf")
    }

    fn sys(seed: u64) -> RealmCoord {
        coord(RealmKindTag::System, seed)
    }

    /// A live cloud tuning at 20 Hz — the reference the timing tests use.
    fn cloud() -> RlmTuning {
        RlmTuning::cloud(20)
    }

    fn head(node: u64, fence: u64, in_transfer: bool) -> OwnerRecord {
        OwnerRecord {
            authority: AuthorityRef::Shard(NodeId(node)),
            fence: Fence(fence),
            lease_expires: UniverseTick(1_000_000),
            in_transfer: in_transfer.then_some(vd_core::TransferId(1)),
        }
    }

    fn facts(running_live: bool) -> TeardownFacts {
        TeardownFacts {
            running_live,
            desired_in_closure: false,
            has_desired_descendant: false,
            quiesced: true,
        }
    }

    // ---- 1. TUNING -------------------------------------------------------------------------------

    #[test]
    fn rlm_tuning_default_is_inert() {
        let t = RlmTuning::default();
        assert_eq!(t.reconcile_interval_ticks, 0);
        assert_eq!(t.demand_ttl_ticks, 0);
        // An inert tuning validates vacuously (never read).
        assert_eq!(t.validate(), Ok(()));
    }

    #[test]
    fn rlm_tuning_cloud_is_valid_and_ordered() {
        let t = cloud();
        assert_eq!(t.validate(), Ok(()));
        assert_eq!(t.reconcile_interval_ticks, 1);
        // demand TTL strictly outlasts the drain+cooldown reclaim floor (thrash-heed).
        assert!(t.demand_ttl_ticks > t.teardown_drain_ticks + t.teardown_cooldown_ticks);
        assert_eq!(
            t.min_dwell_ticks(),
            t.spinup_cooldown_ticks + t.launch_ttl_ticks
        );
    }

    #[test]
    fn rlm_tuning_cloud_floors_at_one_hz() {
        // hz saturates to 1 — every window is still non-zero + ordered.
        assert_eq!(RlmTuning::cloud(0).validate(), Ok(()));
        assert_eq!(RlmTuning::cloud(1).validate(), Ok(()));
    }

    #[test]
    fn rlm_tuning_cloud_delegates_to_boot_agnostic() {
        // DRY: the boot-agnostic `cloud` IS `cloud_with_boot` with an unmeasured boot — byte-identical, so
        // every existing cloud() caller is unaffected by the 5f boot-aware path.
        assert_eq!(RlmTuning::cloud(20), RlmTuning::cloud_with_boot(20, 0, 0));
        assert_eq!(RlmTuning::cloud(1), RlmTuning::cloud_with_boot(1, 0, 0));
    }

    #[test]
    fn rlm_tuning_cloud_with_boot_floors_launch_ttl_on_measured_boot() {
        // At 20 Hz the boot-agnostic launch-TTL is hz*3 = 60 ticks (~3s). A MEASURED boot+settle ABOVE that
        // raises the floor (so a slow real fork is never re-spun mid-boot); one BELOW leaves the ~3s default.
        let settle = 10;
        for p99 in [0u64, 30, 60, 100, 200] {
            let t = RlmTuning::cloud_with_boot(20, p99, settle);
            assert_eq!(
                t.validate(),
                Ok(()),
                "boot-floored budget stays valid (p99={p99})"
            );
            // launch_ttl is the MAX of the ~3s default and the measured boot+settle.
            assert_eq!(t.launch_ttl_ticks, (20 * 3).max(p99 + settle), "p99={p99}");
            // The 5f coupling: min_dwell (spinup_cooldown + launch_ttl) never falls below boot+settle, so a
            // realm is never teardown-eligible before it finishes booting. The condition evaluates both
            // sides every pass (covered); the message is STATIC so the never-taken failure path carries no
            // uncoverable method-call region (HR5).
            assert!(
                t.min_dwell_ticks() >= p99 + settle,
                "min_dwell must cover the measured boot+settle floor"
            );
        }
        // The floor DOMINATES the default when boot is large (200+10 > 60).
        assert_eq!(
            RlmTuning::cloud_with_boot(20, 200, 10).launch_ttl_ticks,
            210
        );
        // The default DOMINATES when boot is small (40+5 < 60).
        assert_eq!(RlmTuning::cloud_with_boot(20, 40, 5).launch_ttl_ticks, 60);
    }

    #[test]
    fn rlm_tuning_active_zero_window_is_rejected() {
        let bad = RlmTuning {
            reconcile_interval_ticks: 1,
            ..RlmTuning::default()
        };
        assert_eq!(bad.validate(), Err(RlmTuningError::ZeroWindowWhileActive));
    }

    #[test]
    fn rlm_tuning_demand_ttl_racing_reclaim_is_rejected() {
        let bad = RlmTuning {
            reconcile_interval_ticks: 1,
            demand_ttl_ticks: 5,
            empty_grace_ticks: 2,
            teardown_cooldown_ticks: 3,
            teardown_drain_ticks: 2, // floor = 5, demand_ttl 5 !> 5
            spinup_cooldown_ticks: 1,
            launch_ttl_ticks: 1,
            recovery_grace_ticks: 1, // non-zero so `any_zero` passes and the reclaim-race check is reached
            arrival_shield_ticks: 0,
        };
        assert_eq!(
            bad.validate(),
            Err(RlmTuningError::DemandTtlRacesReclaim {
                demand_ttl: 5,
                reclaim_floor: 5,
            })
        );
    }

    // ---- 2. LEDGER FOLD --------------------------------------------------------------------------

    #[test]
    fn record_spinup_then_keepalive_refreshes_demand() {
        let mut l = DemandLedger::default();
        l.record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(10), Fence(1));
        assert_eq!(
            l.get(sys(7).path()).expect("cell present").last_demand_tick,
            UniverseTick(10)
        );
        l.record_demand(&sys(7), DemandVerb::KeepAlive, UniverseTick(14), Fence(1));
        assert_eq!(
            l.get(sys(7).path()).expect("cell present").last_demand_tick,
            UniverseTick(14)
        );
        assert_eq!(l.len(), 1);
        assert!(!l.is_empty());
    }

    #[test]
    fn record_empty_advances_last_empty_to_the_max_order_independently() {
        let mut l = DemandLedger::default();
        l.record_demand(&sys(7), DemandVerb::Empty, UniverseTick(20), Fence(2));
        assert_eq!(
            l.get(sys(7).path()).expect("cell present").last_empty_tick,
            Some(UniverseTick(20))
        );
        // A fresher re-report advances the max; a stale (reordered) one is ignored by the max.
        l.record_demand(&sys(7), DemandVerb::Empty, UniverseTick(25), Fence(2));
        l.record_demand(&sys(7), DemandVerb::Empty, UniverseTick(15), Fence(2));
        assert_eq!(
            l.get(sys(7).path()).expect("cell present").last_empty_tick,
            Some(UniverseTick(25)),
            "last_empty tracks the max Empty tick — order-independent"
        );
    }

    #[test]
    fn empty_confirmed_is_order_independent_for_a_same_tick_demand_and_empty() {
        // The determinism bug the crash-replay proptest caught (INV-FOLD-ORDER): a parent's `KeepAlive` and
        // the child's `Empty` on the SAME coord in the SAME tick, from different sources, must reap the SAME
        // way regardless of arrival order. Keying arm-C on `last_demand < last_empty` (both maxes) fixes it.
        let t = cloud();
        let fold = |order: &[(DemandVerb, u64)]| -> LedgerCell {
            let mut l = DemandLedger::default();
            for (v, tk) in order {
                l.record_demand(&sys(7), *v, UniverseTick(*tk), Fence(1));
            }
            l.get(sys(7).path()).expect("cell present").clone()
        };
        let ka_first = fold(&[
            (DemandVerb::KeepAlive, 10),
            (DemandVerb::Empty, 10),
            (DemandVerb::Empty, 11),
        ]);
        let empty_first = fold(&[
            (DemandVerb::Empty, 11),
            (DemandVerb::Empty, 10),
            (DemandVerb::KeepAlive, 10),
        ]);
        for now in [11u64, 12, 15, 100] {
            assert_eq!(
                empty_confirmed(&ka_first, UniverseTick(now), t.empty_grace_ticks),
                empty_confirmed(&empty_first, UniverseTick(now), t.empty_grace_ticks),
                "empty_confirmed diverged across fold order at now={now}"
            );
        }
        // A stale (earlier) reordered demand does NOT un-empty a fresher Empty.
        let stale = fold(&[(DemandVerb::Empty, 20), (DemandVerb::SpinUp, 15)]);
        assert!(
            empty_confirmed(&stale, UniverseTick(20), t.empty_grace_ticks),
            "a stale demand cannot un-empty a fresher Empty"
        );
        // A demand at-or-after the latest Empty DOES un-empty it.
        let fresh_demand = fold(&[(DemandVerb::Empty, 20), (DemandVerb::SpinUp, 22)]);
        assert!(
            !empty_confirmed(&fresh_demand, UniverseTick(22), t.empty_grace_ticks),
            "a demand at-or-after the latest Empty un-empties the realm"
        );
    }

    #[test]
    fn teardown_verb_is_a_noop_the_reconciler_is_the_sole_kill_authority() {
        let mut l = DemandLedger::default();
        l.record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(10), Fence(1));
        l.record_demand(&sys(7), DemandVerb::TearDown, UniverseTick(30), Fence(1));
        // TearDown neither refreshed the demand nor latched empty.
        let c = l.get(sys(7).path()).expect("cell present");
        assert_eq!(c.last_demand_tick, UniverseTick(10));
        assert_eq!(c.last_empty_tick, None);
    }

    #[test]
    fn last_fence_tracks_the_lexicographically_freshest_demand_order_independently() {
        // Same demand set, two inbox orders → identical last_fence (determinism).
        let build = |pairs: &[(u64, u64)]| -> Fence {
            let mut l = DemandLedger::default();
            for (tick, fence) in pairs {
                l.record_demand(
                    &sys(7),
                    DemandVerb::SpinUp,
                    UniverseTick(*tick),
                    Fence(*fence),
                );
            }
            l.get(sys(7).path()).expect("cell present").last_fence
        };
        // The max-(tick,fence) is (5,2); a later lower-tick demand does not displace it.
        assert_eq!(build(&[(5, 2), (3, 9)]), Fence(2));
        assert_eq!(build(&[(3, 9), (5, 2)]), Fence(2));
        // A tie on tick → the greater fence wins, both orders.
        assert_eq!(build(&[(5, 2), (5, 9)]), Fence(9));
        assert_eq!(build(&[(5, 9), (5, 2)]), Fence(9));
    }

    #[test]
    fn mark_spawned_and_reaped_stamp_watermarks() {
        let mut l = DemandLedger::default();
        // mark_spawned creates the cell for an ancestor the demand fold never touched.
        l.mark_spawned(&sys(7), UniverseTick(40));
        assert_eq!(
            l.get(sys(7).path()).expect("cell present").spawn_watermark,
            UniverseTick(40)
        );
        l.mark_reaped(sys(7).path(), UniverseTick(50));
        assert_eq!(
            l.get(sys(7).path())
                .expect("cell present")
                .teardown_watermark,
            UniverseTick(50)
        );
        // mark_reaped on an absent cell is a no-op (no panic).
        l.mark_reaped(sys(8).path(), UniverseTick(60));
        assert!(l.get(sys(8).path()).is_none());
    }

    #[test]
    fn apply_delta_sets_draining_and_ignores_absent_cells() {
        let mut l = DemandLedger::default();
        l.record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(10), Fence(1));
        let mut d = LedgerDelta::default();
        d.set_draining
            .insert(sys(7).path().clone(), Some(UniverseTick(12)));
        d.set_draining
            .insert(sys(8).path().clone(), Some(UniverseTick(12))); // absent → ignored
        l.apply_delta(&d);
        assert_eq!(
            l.get(sys(7).path()).expect("cell present").draining_since,
            Some(UniverseTick(12))
        );
        assert!(l.get(sys(8).path()).is_none());
        // Clearing (rescue).
        let mut clear = LedgerDelta::default();
        clear.set_draining.insert(sys(7).path().clone(), None);
        l.apply_delta(&clear);
        assert_eq!(
            l.get(sys(7).path()).expect("cell present").draining_since,
            None
        );
    }

    // ---- 3. ACTIONS + DELTA (derives) ------------------------------------------------------------

    #[test]
    fn lifecycle_action_and_delta_equality() {
        let a = LifecycleAction::SpinUp { coord: sys(7) };
        assert_eq!(a, LifecycleAction::SpinUp { coord: sys(7) });
        assert_ne!(a, LifecycleAction::SpinUp { coord: sys(8) });
        let k = LifecycleAction::Kill {
            path: sys(7).path().clone(),
            node: NodeId(3),
            fence: Fence(1),
        };
        assert_ne!(a, k);
        let f = LifecycleAction::ForceReap {
            path: sys(7).path().clone(),
            node: NodeId(3),
            fence: Fence(1),
        };
        assert_ne!(k, f);
        assert_eq!(LedgerDelta::default(), LedgerDelta::default());
    }

    // ---- 4. RESOLVERS (from head) ----------------------------------------------------------------

    #[test]
    fn running_live_zombie_launching_partition_the_actual_states() {
        let alive = &|_n: NodeId| false;
        let dead = &|_n: NodeId| true;
        // A live head.
        assert!(running_live(Some(head(9, 1, false)), alive));
        assert!(!zombie(Some(head(9, 1, false)), alive));
        // A dead head → zombie, NOT running_live (BUG-A).
        assert!(!running_live(Some(head(9, 1, false)), dead));
        assert!(zombie(Some(head(9, 1, false)), dead));
        // No head → neither.
        assert!(!running_live(None, alive));
        assert!(!zombie(None, alive));
        // Launching = no head but a live launch (BUG-B).
        assert!(launching(None, true));
        assert!(!launching(None, false));
        assert!(!launching(Some(head(9, 1, false)), true));
    }

    // ---- 5. DESIRED / EMPTY / TEARDOWN predicates ------------------------------------------------

    #[test]
    fn desired_alive_arm_a_demand_driven() {
        let mut l = DemandLedger::default();
        l.record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        let c = l.get(sys(7).path()).expect("cell present");
        let t = cloud();
        // Within TTL ⇒ desired even when NOT running (arm A).
        assert!(desired_alive(c, UniverseTick(100), &t, false, false));
        assert!(desired_alive(
            c,
            UniverseTick(100 + t.demand_ttl_ticks),
            &t,
            false,
            false
        ));
        // Past TTL and not running ⇒ not desired.
        assert!(!desired_alive(
            c,
            UniverseTick(101 + t.demand_ttl_ticks),
            &t,
            false,
            false
        ));
        // ...unless somebody is on their way here (arm C): a pending hand-off wants it alive on its
        // own, which is exactly what lets an arrival re-start a destination whose shard has vanished.
        assert!(desired_alive(
            c,
            UniverseTick(101 + t.demand_ttl_ticks),
            &t,
            false,
            true
        ));
    }

    #[test]
    fn desired_alive_arm_b_live_and_not_empty_no_strand() {
        let mut l = DemandLedger::default();
        // A realm demanded long ago (arm A now stale) that is LIVE and never said Empty.
        l.record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(1), Fence(1));
        let c = l.get(sys(7).path()).expect("cell present");
        let t = cloud();
        let now = UniverseTick(10_000); // arm A long expired
        assert!(!demanded_recently(c, now, t.demand_ttl_ticks));
        // Arm B keeps it alive because it is running_live and never confirmed empty (NO-STRAND).
        assert!(desired_alive(c, now, &t, true, false));
        // If it is NOT running, arm B is off ⇒ not desired.
        assert!(!desired_alive(c, now, &t, false, false));
    }

    #[test]
    fn an_outward_crossings_parent_stays_alive_without_an_upward_demand() {
        // Lane cure, finding 37 — the deterministic twin of the shard-side refusal gate
        // (`stub`'s `an_outward_crossing_emits_no_demand_and_counts_the_refusal`): an occupant
        // crossing OUT of a system toward its parent produces NO demand naming the parent — and none
        // is needed. While the hand-off latch stands the source still speaks for the crosser
        // (`speaks_for`), so it never reports Empty: arm B holds the SOURCE desired, and
        // `ancestor_close` pulls its whole parent chain — the parent is alive through CLOSURE, never
        // through the upward demand SL7 forbids. (The process-tier experiment is the return-crossing
        // gate `a_planet_to_system_return_commits_both_rehomes_and_the_player_rides`.)
        let mut l = DemandLedger::default();
        // The source was demanded once, long ago (arm A stale by `now`), and never reported Empty
        // (its mid-crossing observer fold keeps it non-empty for the whole hand-off window).
        l.record_demand(&u_g_s(), DemandVerb::SpinUp, UniverseTick(1), Fence(1));
        let c = l.get(u_g_s().path()).expect("cell present");
        let t = cloud();
        let now = UniverseTick(10_000);
        assert!(
            !demanded_recently(c, now, t.demand_ttl_ticks),
            "no demand from anyone — the upward keep-alive is deleted and arm A is stale"
        );
        assert!(
            desired_alive(c, now, &t, true, false),
            "arm B alone holds the source: running and not affirmatively empty"
        );
        // The closure pulls the parent chain: every ancestor of the desired source stays alive.
        let mut desired = BTreeMap::new();
        desired.insert(u_g_s().path().clone(), u_g_s());
        let closed = ancestor_close(&desired);
        assert!(
            closed.contains_key(u_g().path()),
            "the crossing dest (the parent) is alive through the closure"
        );
        assert!(
            closed.contains_key(u_root().path()),
            "…and so is the whole chain above it"
        );
    }

    #[test]
    fn demanded_recently_ignores_the_never_demanded_sentinel() {
        let t = cloud();
        let mut l = DemandLedger::default();
        // A realm that has ONLY reported Empty keeps the `0` sentinel ⇒ NOT demanded, even at an early
        // tick where `now <= ttl` would spuriously pass a raw `now - 0 <= ttl`.
        l.record_demand(&sys(7), DemandVerb::Empty, UniverseTick(3), Fence(1));
        let c = l.get(sys(7).path()).expect("cell present");
        assert_eq!(c.last_demand_tick, UniverseTick(0));
        assert!(!demanded_recently(c, UniverseTick(5), t.demand_ttl_ticks));
        // A real SpinUp ⇒ demanded within the TTL.
        l.record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(6), Fence(1));
        assert!(demanded_recently(
            l.get(sys(7).path()).expect("cell present"),
            UniverseTick(7),
            t.demand_ttl_ticks
        ));
    }

    #[test]
    fn empty_confirmed_requires_fresh_report_and_no_demand_since_streak() {
        let t = cloud();
        let mut l = DemandLedger::default();
        l.record_demand(&sys(7), DemandVerb::Empty, UniverseTick(100), Fence(1));
        let c = l.get(sys(7).path()).expect("cell present").clone();
        // Fresh + no demand since streak ⇒ confirmed.
        assert!(empty_confirmed(&c, UniverseTick(100), t.empty_grace_ticks));
        assert!(empty_confirmed(
            &c,
            UniverseTick(100 + t.empty_grace_ticks),
            t.empty_grace_ticks
        ));
        // Stale report (past grace from the streak start) ⇒ NOT confirmed (crashed-shard disambiguation).
        assert!(!empty_confirmed(
            &c,
            UniverseTick(101 + t.empty_grace_ticks),
            t.empty_grace_ticks
        ));
    }

    #[test]
    fn empty_confirmed_stays_true_while_empty_is_re_asserted_beyond_grace() {
        // A realm that stays empty and re-reports `Empty` every tick must REMAIN confirmed-empty far past
        // the grace window (freshness is from the LAST report, not the streak start) — else a long-empty
        // realm could never be reaped. A crashed realm (stops reporting) DOES age out.
        let t = cloud();
        let mut l = DemandLedger::default();
        for tick in 100..100 + 5 * t.empty_grace_ticks {
            l.record_demand(&sys(7), DemandVerb::Empty, UniverseTick(tick), Fence(1));
            assert!(
                empty_confirmed(
                    l.get(sys(7).path()).expect("cell present"),
                    UniverseTick(tick),
                    t.empty_grace_ticks
                ),
                "a continuously-empty realm stays confirmed at tick {tick}"
            );
        }
        // It STOPS reporting (crash) ⇒ ages out of confirmed-empty past grace.
        let last = 100 + 5 * t.empty_grace_ticks - 1;
        assert!(!empty_confirmed(
            l.get(sys(7).path()).expect("cell present"),
            UniverseTick(last + t.empty_grace_ticks + 1),
            t.empty_grace_ticks
        ));
    }

    #[test]
    fn empty_confirmed_false_when_never_empty_or_demand_after_streak() {
        let t = cloud();
        let mut l = DemandLedger::default();
        // Never said Empty ⇒ false.
        l.record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        assert!(!empty_confirmed(
            l.get(sys(7).path()).expect("cell present"),
            UniverseTick(100),
            t.empty_grace_ticks
        ));
        // Empty streak, then a SpinUp AT-OR-AFTER it processed FIRST (reorder): last_demand >= streak ⇒
        // NOT confirmed regardless of order (arm C).
        let mut r = DemandLedger::default();
        r.record_demand(&sys(8), DemandVerb::SpinUp, UniverseTick(30), Fence(1));
        r.record_demand(&sys(8), DemandVerb::Empty, UniverseTick(30), Fence(1));
        assert!(!empty_confirmed(
            r.get(sys(8).path()).expect("cell present"),
            UniverseTick(30),
            t.empty_grace_ticks
        ));
    }

    #[test]
    fn teardown_ready_requires_every_clause() {
        let t = cloud();
        let mut l = DemandLedger::default();
        // A realm that is live, FRESHLY empty (confirmed), past dwell + cooldown, orphans nothing.
        // `now` is well past min_dwell (spawn_watermark 0); the Empty streak is within grace.
        let now = UniverseTick(200);
        l.record_demand(&sys(7), DemandVerb::Empty, UniverseTick(195), Fence(1));
        let base = l.get(sys(7).path()).expect("cell present").clone();
        assert!(teardown_ready(&base, now, &t, &facts(true)));

        // Each single clause flip makes it NOT ready:
        assert!(!teardown_ready(&base, now, &t, &facts(false))); // not running_live
        let mut in_closure = facts(true);
        in_closure.desired_in_closure = true;
        assert!(!teardown_ready(&base, now, &t, &in_closure)); // still demanded/desired
        let mut has_desc = facts(true);
        has_desc.has_desired_descendant = true;
        assert!(!teardown_ready(&base, now, &t, &has_desc)); // a live child
        let mut not_quiesced = facts(true);
        not_quiesced.quiesced = false;
        assert!(!teardown_ready(&base, now, &t, &not_quiesced)); // crash quiesce

        // Not empty-confirmed ⇒ not ready.
        let mut never_empty = DemandLedger::default();
        never_empty.record_demand(&sys(9), DemandVerb::SpinUp, UniverseTick(1), Fence(1));
        assert!(!teardown_ready(
            never_empty.get(sys(9).path()).expect("cell present"),
            now,
            &t,
            &facts(true)
        ));

        // Inside the min-dwell window (just spawned) ⇒ not ready even if empty.
        let mut fresh = base.clone();
        fresh.spawn_watermark = UniverseTick(now.0 - 1);
        assert!(!teardown_ready(&fresh, now, &t, &facts(true)));
        // Inside the teardown cooldown (just reaped) ⇒ not ready.
        let mut cooling = base.clone();
        cooling.teardown_watermark = UniverseTick(now.0 - 1);
        assert!(!teardown_ready(&cooling, now, &t, &facts(true)));
    }

    // ---- 6. RECONCILE + CLOSURE + DRAIN (slice 3b) ----------------------------------------------

    use crate::directory::DirectoryTuning;
    use vd_core::pose::RealmId;

    /// A multi-level lineage coord (Universe→…→leaf).
    fn deep(levels: &[(RealmKindTag, u64)]) -> RealmCoord {
        RealmCoord::from_path(RealmPath::from_levels(
            levels
                .iter()
                .map(|(k, s)| RealmLevel::new(*k, *s))
                .collect(),
        ))
        .expect("non-empty lineage")
    }

    /// A directory with the given realm heads granted `(rid, owner node, fence)`.
    fn dir(heads: &[(RealmId, u64, u64)]) -> DirectoryCore {
        let mut d = DirectoryCore::new(DirectoryTuning::default());
        for (rid, node, fence) in heads {
            d.grant(
                DirectoryKey::Realm(*rid),
                AuthorityRef::Shard(NodeId(*node)),
                Fence(*fence),
                UniverseTick(0),
            );
        }
        d
    }

    fn no_launch() -> BTreeMap<RealmPath, BTreeMap<NodeId, UniverseTick>> {
        BTreeMap::new()
    }

    /// No hand-off is delivering anyone anywhere — the state of a cluster with no crossing in flight,
    /// which is what every pre-existing lifecycle scenario assumed and still gets.
    fn no_arrivals() -> BTreeSet<RealmId> {
        BTreeSet::new()
    }

    // Lineage helpers for the ancestor tests: a Planet nested [U,G,System(7),Planet(3)].
    fn u_root() -> RealmCoord {
        deep(&[(RealmKindTag::Universe, 0)])
    }
    fn u_g() -> RealmCoord {
        deep(&[(RealmKindTag::Universe, 0), (RealmKindTag::Galaxy, 1)])
    }
    fn u_g_s() -> RealmCoord {
        deep(&[
            (RealmKindTag::Universe, 0),
            (RealmKindTag::Galaxy, 1),
            (RealmKindTag::System, 7),
        ])
    }
    fn u_g_s_p() -> RealmCoord {
        deep(&[
            (RealmKindTag::Universe, 0),
            (RealmKindTag::Galaxy, 1),
            (RealmKindTag::System, 7),
            (RealmKindTag::Planet, 3),
        ])
    }

    #[test]
    fn reconcile_spins_up_a_demanded_realm_not_running() {
        let t = cloud();
        let mut l = DemandLedger::default();
        l.record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        let d = dir(&[]); // no head ⇒ not running
        let (actions, delta) = reconcile(
            &l,
            &d,
            &|_n: NodeId| false,
            &no_launch(),
            &t,
            UniverseTick(100),
            UniverseTick(0),
            &no_arrivals(),
        );
        assert_eq!(actions, vec![LifecycleAction::SpinUp { coord: sys(7) }]);
        assert!(delta.set_draining.is_empty());
        assert!(delta.retire.is_empty());
    }

    #[test]
    fn reconcile_ancestor_closes_and_spins_up_ancestor_first() {
        let t = cloud();
        let mut l = DemandLedger::default();
        l.record_demand(&u_g_s_p(), DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        let d = dir(&[]);
        let (actions, _) = reconcile(
            &l,
            &d,
            &|_n: NodeId| false,
            &no_launch(),
            &t,
            UniverseTick(100),
            UniverseTick(0),
            &no_arrivals(),
        );
        // The whole chain spins up, ancestor-first (a shorter path prefix sorts before its extension).
        assert_eq!(
            actions,
            vec![
                LifecycleAction::SpinUp { coord: u_root() },
                LifecycleAction::SpinUp { coord: u_g() },
                LifecycleAction::SpinUp { coord: u_g_s() },
                LifecycleAction::SpinUp { coord: u_g_s_p() },
            ]
        );
    }

    /// RLM 5f-3a — the OQ-4 "ride" contract (Layer-1, pure kernel). The bootstrap-ride proof: ONE synthetic
    /// leaf demand for a player's DEEPEST home realm spins up the WHOLE ancestor chain in ONE reconcile
    /// sweep. DISTINCT from `reconcile_ancestor_closes_and_spins_up_ancestor_first` above (which demands a
    /// hand-built 4-level coord `u_g_s_p` = [Universe(0), Galaxy(1), System(7), Planet(3)]): here the
    /// demanded leaf is resolved by the REAL 5f-2 resolver `container_coord_at` at the Area-A box (x=25) —
    /// proven by 5f-2 to be the DEEPER full 5-level lineage [Universe(0), Galaxy(1), System(7), Planet(7),
    /// Area(7)] — so this exercises the real resolver → reconcile seam end-to-end on the deepest chain.
    /// The four ancestors have NO demand cell of their own (only the Area leaf was recorded); they are
    /// desired PURELY by `ancestor_close`, so passing proves a leaf demand pulls its whole chain with NO
    /// per-level descent.
    #[test]
    fn reconcile_5f3a_bootstrap_ride_spins_the_whole_area_lineage_from_one_leaf_demand() {
        use RealmKindTag::{Area, Galaxy, Planet, System, Universe};
        use vd_core::pose::RealmId;
        use vd_core::worldgen::coord_of_realm;
        use vd_physics::worldgen::realm_regions_for;

        let t = cloud();
        // The deepest home lineage from the REAL forest (NOT hand-built) — the Area-A box. FIXTURE forest:
        // the deepest realm there is a player-built AREA, which the generator never makes, so naming a
        // generated realm would spin a 3-level chain while the test asserted it proved a 5-level one. This
        // used to be said as the position (25,0,0) and descended for; the descent is gone.
        let deepest = coord_of_realm(&realm_regions_for(0), RealmId::Area(7))
            .expect("the fixture forest holds the deep area home");
        let mut l = DemandLedger::default();
        // tick NONZERO (a post-ClockSync tick) — a tick-0 seed is inert by `demanded_recently`.
        l.record_demand(&deepest, DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        let d = dir(&[]); // EMPTY directory — no heads, nothing running.
        let (actions, delta) = reconcile(
            &l,
            &d,
            &|_n: NodeId| false,
            &no_launch(),
            &t,
            UniverseTick(100),
            UniverseTick(0),
            &no_arrivals(),
        );
        // EXACTLY 5 SpinUps, ancestor-first, coord-for-coord — the whole Universe→Area chain from ONE leaf.
        // (All-SpinUp ⇒ no ForceReap/Kill.) The comparison is over the FULL `.path()` per level (each `deep`
        // builds the lineage), proving `container_coord_at`'s leaf drove the exact ancestor prefix chain.
        assert_eq!(
            actions,
            vec![
                LifecycleAction::SpinUp {
                    coord: deep(&[(Universe, 0)])
                },
                LifecycleAction::SpinUp {
                    coord: deep(&[(Universe, 0), (Galaxy, 1)])
                },
                LifecycleAction::SpinUp {
                    coord: deep(&[(Universe, 0), (Galaxy, 1), (System, 7)])
                },
                LifecycleAction::SpinUp {
                    coord: deep(&[(Universe, 0), (Galaxy, 1), (System, 7), (Planet, 7)])
                },
                LifecycleAction::SpinUp {
                    coord: deep(&[
                        (Universe, 0),
                        (Galaxy, 1),
                        (System, 7),
                        (Planet, 7),
                        (Area, 7)
                    ])
                },
            ]
        );
        // Pure spin-up bootstrap: no teardown/draining side-effects.
        assert!(delta.set_draining.is_empty());
        assert!(delta.retire.is_empty());
    }

    #[test]
    fn reconcile_does_not_respawn_a_launching_realm() {
        let t = cloud();
        let mut l = DemandLedger::default();
        l.record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        let d = dir(&[]); // head not yet self-granted
        let mut launch = BTreeMap::new();
        let mut nodes = BTreeMap::new();
        nodes.insert(NodeId(50), UniverseTick(99));
        launch.insert(sys(7).path().clone(), nodes);
        let (actions, _) = reconcile(
            &l,
            &d,
            &|_n: NodeId| false,
            &launch,
            &t,
            UniverseTick(100),
            UniverseTick(0),
            &no_arrivals(),
        );
        assert!(
            actions.is_empty(),
            "a live launch ⇒ no double-spawn (BUG-B)"
        );
    }

    #[test]
    fn reconcile_respects_the_spinup_cooldown() {
        let t = cloud();
        let mut l = DemandLedger::default();
        l.record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        l.mark_spawned(&sys(7), UniverseTick(100)); // just spawned; head not yet up, no live launch
        let d = dir(&[]);
        // within the spin-up cooldown ⇒ no re-spawn.
        let (early, _) = reconcile(
            &l,
            &d,
            &|_n: NodeId| false,
            &no_launch(),
            &t,
            UniverseTick(105),
            UniverseTick(0),
            &no_arrivals(),
        );
        assert!(early.is_empty());
        // past the cooldown (a silent-failed launch) ⇒ re-spawn (self-heal).
        let (late, _) = reconcile(
            &l,
            &d,
            &|_n: NodeId| false,
            &no_launch(),
            &t,
            UniverseTick(100 + t.spinup_cooldown_ticks),
            UniverseTick(0),
            &no_arrivals(),
        );
        assert_eq!(late, vec![LifecycleAction::SpinUp { coord: sys(7) }]);
    }

    #[test]
    fn reconcile_force_reaps_a_zombie_but_not_a_live_head() {
        let t = cloud();
        let mut l = DemandLedger::default();
        l.record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(100), Fence(2));
        let d = dir(&[(RealmId::System(7), 9, 5)]); // head owned by node 9
        // node 9 DEAD ⇒ zombie ⇒ ForceReap (revoke the stale head + kill the corpse).
        let (reap, _) = reconcile(
            &l,
            &d,
            &|n: NodeId| n == NodeId(9),
            &no_launch(),
            &t,
            UniverseTick(100),
            UniverseTick(0),
            &no_arrivals(),
        );
        assert_eq!(
            reap,
            vec![LifecycleAction::ForceReap {
                path: sys(7).path().clone(),
                node: NodeId(9),
                fence: Fence(5),
            }]
        );
        // node 9 ALIVE ⇒ running_live ⇒ no ForceReap; and it is desired+running ⇒ no SpinUp/Kill.
        let (quiet, _) = reconcile(
            &l,
            &d,
            &|_n: NodeId| false,
            &no_launch(),
            &t,
            UniverseTick(100),
            UniverseTick(0),
            &no_arrivals(),
        );
        assert!(quiet.is_empty());
    }

    #[test]
    fn reconcile_two_phase_teardown_opens_a_drain_then_a_demand_rescues() {
        let t = cloud();
        let d = dir(&[(RealmId::System(7), 9, 3)]);
        let now = UniverseTick(500);
        let mut l = DemandLedger::default();
        l.record_demand(
            &sys(7),
            DemandVerb::Empty,
            UniverseTick(now.0 - 2),
            Fence(1),
        ); // fresh empty
        // Tick A: becomes teardown-ready ⇒ open the drain, NO kill.
        let (a, da) = reconcile(
            &l,
            &d,
            &|_n: NodeId| false,
            &no_launch(),
            &t,
            now,
            UniverseTick(0),
            &no_arrivals(),
        );
        assert!(
            a.is_empty(),
            "the first ready tick opens the drain, never kills"
        );
        assert_eq!(da.set_draining.get(sys(7).path()), Some(&Some(now)));
        l.apply_delta(&da);
        // Tick B: a demand lands INSIDE the drain window ⇒ rescue (drain cleared, no kill).
        l.record_demand(
            &sys(7),
            DemandVerb::SpinUp,
            UniverseTick(now.0 + 1),
            Fence(1),
        );
        let (b, db) = reconcile(
            &l,
            &d,
            &|_n: NodeId| false,
            &no_launch(),
            &t,
            UniverseTick(now.0 + 1),
            UniverseTick(0),
            &no_arrivals(),
        );
        assert!(b.is_empty(), "a rescuing demand aborts the kill (BUG-C)");
        assert_eq!(
            db.set_draining.get(sys(7).path()),
            Some(&None),
            "drain cleared"
        );
        l.apply_delta(&db);
        assert_eq!(
            l.get(sys(7).path()).expect("cell present").draining_since,
            None
        );
    }

    #[test]
    fn reconcile_kills_after_the_drain_window_elapses() {
        let t = cloud();
        let d = dir(&[(RealmId::System(7), 9, 3)]);
        let now = UniverseTick(500);
        let mut l = DemandLedger::default();
        l.record_demand(
            &sys(7),
            DemandVerb::Empty,
            UniverseTick(now.0 - 2),
            Fence(1),
        );
        // Force it into Draining, started exactly `drain` ticks ago ⇒ the window has elapsed.
        let mut open = LedgerDelta::default();
        open.set_draining.insert(
            sys(7).path().clone(),
            Some(UniverseTick(now.0 - t.teardown_drain_ticks)),
        );
        l.apply_delta(&open);
        let (actions, _) = reconcile(
            &l,
            &d,
            &|_n: NodeId| false,
            &no_launch(),
            &t,
            now,
            UniverseTick(0),
            &no_arrivals(),
        );
        assert_eq!(
            actions,
            vec![LifecycleAction::Kill {
                path: sys(7).path().clone(),
                node: NodeId(9),
                fence: Fence(3),
            }]
        );
    }

    /// The exact fixture of [`reconcile_kills_after_the_drain_window_elapses`]: a live realm that has
    /// gone affirmatively empty, is out of everyone's area of interest, and has sat out its drain.
    fn drained_empty_realm(t: &RlmTuning, now: UniverseTick) -> DemandLedger {
        let mut l = DemandLedger::default();
        l.record_demand(
            &sys(7),
            DemandVerb::Empty,
            UniverseTick(now.0 - 2),
            Fence(1),
        );
        let mut open = LedgerDelta::default();
        open.set_draining.insert(
            sys(7).path().clone(),
            Some(UniverseTick(now.0 - t.teardown_drain_ticks)),
        );
        l.apply_delta(&open);
        l
    }

    /// Whether a hand-off in this phase is PAST the commit point and still short of terminal — the
    /// phases whose dwell the arrival shield's budget has to cover. EXHAUSTIVE with no wildcard, so a
    /// new phase does not compile until someone decides which side of the commit it falls on.
    fn is_post_commit_dwell(s: &crate::saga::SagaState) -> bool {
        use crate::saga::SagaState as S;
        match s {
            S::Swapping { .. } | S::Demoting { .. } | S::Promoting { .. } | S::Releasing { .. } => {
                true
            }
            S::AwaitProvision
            | S::Preparing
            | S::Cutting
            | S::Freezing { .. }
            | S::CommittingCas { .. }
            | S::BatchCommitting { .. }
            // The transient tail is counted through its OWN phase enum below, not here.
            | S::BatchHandoff { .. }
            // A re-home never shields, so its dwell is not the shield's to cover.
            | S::ReHoming { .. }
            | S::Done { .. }
            | S::Aborting { .. }
            | S::Aborted { .. } => false,
        }
    }

    #[test]
    fn post_commit_steps_is_recounted_from_the_state_machine() {
        use crate::saga::{BatchHandoffPhase as P, SagaState as S};
        // The constant the shield's budget is sized from is RECOUNTED here from the machine itself,
        // rather than trusted. Adding a phase breaks the exhaustive match above and this array, so the
        // budget cannot silently stop covering a hand-off that grew a step.
        let all = [
            S::AwaitProvision,
            S::Preparing,
            S::Cutting,
            S::Freezing {
                marker_seq: 1,
                frozen_drained: None,
                flushed: false,
            },
            S::CommittingCas {
                marker_seq: 1,
                drained_seq: 1,
            },
            S::BatchCommitting { step_id: 1 },
            S::BatchHandoff {
                phase: P::AwaitAdopt,
                new_fence: Fence(2),
            },
            S::Swapping {
                new_fence: Fence(2),
            },
            S::Demoting {
                new_fence: Fence(2),
                dest_delivered: false,
            },
            S::Promoting {
                new_fence: Fence(2),
                promote_acked: false,
                dest_delivered: false,
                rehome_target: None,
            },
            S::Releasing {
                new_fence: Fence(2),
            },
            S::ReHoming {
                target: NodeId(1),
                prev_fence: Fence(1),
            },
            S::Done {
                new_fence: Fence(2),
            },
            S::Aborting {
                reason: crate::saga::AbortReason::CasLost,
                awaiting_thaw: false,
                awaiting_abort_ack: false,
            },
            S::Aborted {
                reason: crate::saga::AbortReason::CasLost,
            },
        ];
        let durable = all.iter().filter(|s| is_post_commit_dwell(s)).count() as u64;
        assert_eq!(durable, POST_COMMIT_STEPS, "the durable post-commit tail");

        // The transient tail's four phases live inside ONE state, so they are counted separately.
        let transient = [
            P::AwaitAdopt,
            P::AwaitRelease,
            P::AwaitPromote,
            P::AwaitComplete,
        ];
        assert_eq!(
            transient.len() as u64,
            POST_COMMIT_STEPS,
            "the transient batch tail is the same length — one constant covers both (HR2)"
        );
    }

    #[test]
    fn the_arrival_shield_budget_outlasts_the_window_it_protects_against() {
        // The invariant that makes the shield useful rather than decorative: its budget must span the
        // whole window in which an unshielded realm would be reclaimed, or it could expire inside the
        // very drain it exists to survive.
        let rlm = cloud();
        let saga = crate::saga::SagaTuning::default();
        let cap = derive_arrival_shield_ticks(&rlm, &saga);
        assert!(
            cap > rlm.teardown_drain_ticks + rlm.teardown_cooldown_ticks,
            "shield budget {cap} must outlast drain {} + cooldown {}",
            rlm.teardown_drain_ticks,
            rlm.teardown_cooldown_ticks
        );
        // And it must cover the worst HEALTHY completion: one destructive-abort budget plus a cheap
        // re-drive for every post-commit phase.
        assert!(cap >= saga.abort_deadline_ticks + POST_COMMIT_STEPS * saga.redrive_deadline_ticks);
        // A deployment whose transfer budget is tiny still cannot get a shield shorter than the
        // reclaim floor — the `max` is what stops a misconfigured saga budget disarming the shield.
        let tiny = crate::saga::SagaTuning {
            redrive_deadline_ticks: 1,
            abort_deadline_ticks: 1,
        };
        assert_eq!(
            derive_arrival_shield_ticks(&rlm, &tiny),
            rlm.teardown_drain_ticks + rlm.teardown_cooldown_ticks + 1
        );
    }

    #[test]
    fn the_inert_default_leaves_the_arrival_shield_disarmed() {
        // A reconciler that never sweeps carries no shield budget — zero, not a guessed number.
        assert_eq!(RlmTuning::default().arrival_shield_ticks, 0);
        assert_eq!(cloud().arrival_shield_ticks, 0, "derived at boot, not here");
    }

    #[test]
    fn an_arrival_holds_the_realm_the_same_fixture_would_have_killed() {
        // THE FLIP, at the kernel. The same drained, empty, out-of-everyone's-interest realm that gets
        // killed in the test above is NOT killed once a hand-off is delivering somebody into it — and
        // it is reported by name, because the arrival is the sole reason it is still standing.
        let t = cloud();
        let d = dir(&[(RealmId::System(7), 9, 3)]);
        let now = UniverseTick(500);
        let l = drained_empty_realm(&t, now);
        let (actions, delta) = reconcile(
            &l,
            &d,
            &|_n: NodeId| false,
            &no_launch(),
            &t,
            now,
            UniverseTick(0),
            &BTreeSet::from([RealmId::System(7)]),
        );
        assert_eq!(actions, vec![], "no kill: somebody is on their way here");
        assert_eq!(
            delta.arrival_shielded,
            BTreeSet::from([sys(7).path().clone()]),
            "and the shield is named, because nothing else was keeping this realm alive"
        );
    }

    #[test]
    fn an_arrival_holds_the_destinations_whole_parent_chain() {
        // Why the arrival is a REASON TO WANT rather than a veto on tearing down: a want flows through
        // the ancestor closure, so the destination's parents are held up with it. A veto would have let
        // the player land on a planet whose star system had just been shut down.
        let t = cloud();
        // Both the planet and its parent system are live, drained and empty — reapable on their own.
        let d = dir(&[(RealmId::System(7), 9, 3), (RealmId::Planet(3), 11, 3)]);
        let now = UniverseTick(500);
        let mut l = DemandLedger::default();
        for c in [u_g_s(), u_g_s_p()] {
            l.record_demand(&c, DemandVerb::Empty, UniverseTick(now.0 - 2), Fence(1));
            let mut open = LedgerDelta::default();
            open.set_draining.insert(
                c.path().clone(),
                Some(UniverseTick(now.0 - t.teardown_drain_ticks)),
            );
            l.apply_delta(&open);
        }
        let (actions, _delta) = reconcile(
            &l,
            &d,
            &|_n: NodeId| false,
            &no_launch(),
            &t,
            now,
            UniverseTick(0),
            &BTreeSet::from([RealmId::Planet(3)]),
        );
        assert_eq!(
            actions
                .iter()
                .filter(|a| matches!(a, LifecycleAction::Kill { .. }))
                .count(),
            0,
            "neither the planet being arrived at NOR the star system holding it is reaped"
        );
        // The other half of "a want, not a veto", and worth naming: the arrival pulls the destination's
        // WHOLE lineage into the desired set, so the ancestors above it that are not running get spun
        // up rather than merely spared. That is a real action taken because of a hand-off, accepted
        // deliberately — a player must have an entire chain of places to arrive into, not just a leaf.
        assert_eq!(
            actions,
            vec![
                LifecycleAction::SpinUp { coord: u_root() },
                LifecycleAction::SpinUp { coord: u_g() },
            ],
            "the missing ancestors above the destination are started, ancestor-first"
        );
    }

    #[test]
    fn an_arrival_at_a_realm_someone_else_already_wants_is_not_reported() {
        // The counter's healthy baseline is ZERO. During a healthy crossing the source shard's own
        // keep-alive already holds the destination up, so the shield is not load-bearing and says
        // nothing — which is what makes a NON-zero reading meaningful rather than routine.
        let t = cloud();
        let d = dir(&[(RealmId::System(7), 9, 3)]);
        let now = UniverseTick(500);
        let mut l = DemandLedger::default();
        l.record_demand(&sys(7), DemandVerb::KeepAlive, now, Fence(1));
        let (_actions, delta) = reconcile(
            &l,
            &d,
            &|_n: NodeId| false,
            &no_launch(),
            &t,
            now,
            UniverseTick(0),
            &BTreeSet::from([RealmId::System(7)]),
        );
        assert_eq!(delta.arrival_shielded, BTreeSet::new());
    }

    #[test]
    fn reconcile_is_run_twice_identical_with_an_arriving_set() {
        // The decision stays a pure function of its inputs: same ledger, same directory, same arrivals
        // ⇒ same actions and the same report, every time.
        let t = cloud();
        let d = dir(&[(RealmId::System(7), 9, 3)]);
        let now = UniverseTick(500);
        let l = drained_empty_realm(&t, now);
        let arriving = BTreeSet::from([RealmId::System(7)]);
        let run = || {
            reconcile(
                &l,
                &d,
                &|_n: NodeId| false,
                &no_launch(),
                &t,
                now,
                UniverseTick(0),
                &arriving,
            )
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn a_keepalive_demand_on_the_crossing_dest_blocks_the_reap_that_freezes_the_return() {
        // CONSUMER gate for the Symptom-B fix: reconcile reaps a drained, empty, out-of-closure realm — and
        // for a RETURN crossing's dest System (mid-crossing) that reap FREEZES THE WHOLE GAME — UNLESS a
        // KeepAlive demand keeps it desired. This proves the source shard's per-tick keep-alive
        // (`redrive_stranded_crossings`) is SUFFICIENT to hold the dest alive. Mirrors
        // `reconcile_kills_after_the_drain_window_elapses` above, adding the keep-alive variant.
        let t = cloud();
        let d = dir(&[(RealmId::System(7), 9, 3)]);
        let now = UniverseTick(500);
        let drained_empty_ledger = || {
            let mut l = DemandLedger::default();
            l.record_demand(
                &sys(7),
                DemandVerb::Empty,
                UniverseTick(now.0 - 2),
                Fence(1),
            );
            let mut open = LedgerDelta::default();
            open.set_draining.insert(
                sys(7).path().clone(),
                Some(UniverseTick(now.0 - t.teardown_drain_ticks)),
            );
            l.apply_delta(&open);
            l
        };
        let kill = LifecycleAction::Kill {
            path: sys(7).path().clone(),
            node: NodeId(9),
            fence: Fence(3),
        };
        // WITHOUT a keep-alive → the reap fires (the return-freeze root cause).
        let (killed, _) = reconcile(
            &drained_empty_ledger(),
            &d,
            &|_n: NodeId| false,
            &no_launch(),
            &t,
            now,
            UniverseTick(0),
            &no_arrivals(),
        );
        assert!(killed.contains(&kill));
        // WITH a KeepAlive demand for the dest at `now` → it is desired → ancestor_close keeps its whole
        // chain → teardown_ready is false → NO reap. The freeze cannot happen while a player is crossing in.
        let mut l = drained_empty_ledger();
        l.record_demand(&sys(7), DemandVerb::KeepAlive, now, Fence(3));
        let (protected, _) = reconcile(
            &l,
            &d,
            &|_n: NodeId| false,
            &no_launch(),
            &t,
            now,
            UniverseTick(0),
            &no_arrivals(),
        );
        assert!(!protected.contains(&kill));
    }

    #[test]
    fn reconcile_reaps_child_before_parent() {
        let t = cloud();
        let d = dir(&[(RealmId::System(7), 9, 3), (RealmId::Planet(3), 8, 4)]);
        let now = UniverseTick(500);
        let mut l = DemandLedger::default();
        l.record_demand(
            &u_g_s(),
            DemandVerb::Empty,
            UniverseTick(now.0 - 2),
            Fence(1),
        );
        l.record_demand(
            &u_g_s_p(),
            DemandVerb::Empty,
            UniverseTick(now.0 - 2),
            Fence(1),
        );
        let mut open = LedgerDelta::default();
        let started = Some(UniverseTick(now.0 - t.teardown_drain_ticks));
        open.set_draining.insert(u_g_s().path().clone(), started);
        open.set_draining.insert(u_g_s_p().path().clone(), started);
        l.apply_delta(&open);
        let (actions, _) = reconcile(
            &l,
            &d,
            &|_n: NodeId| false,
            &no_launch(),
            &t,
            now,
            UniverseTick(0),
            &no_arrivals(),
        );
        // The child (longer path) is reaped BEFORE its parent — never orphan a live child.
        assert_eq!(
            actions,
            vec![
                LifecycleAction::Kill {
                    path: u_g_s_p().path().clone(),
                    node: NodeId(8),
                    fence: Fence(4),
                },
                LifecycleAction::Kill {
                    path: u_g_s().path().clone(),
                    node: NodeId(9),
                    fence: Fence(3),
                },
            ]
        );
    }

    #[test]
    fn reconcile_keeps_a_parent_alive_while_a_descendant_is_desired() {
        let t = cloud();
        // System live + empty, but its Planet child is DEMANDED ⇒ the System is an ancestor ⇒ not reaped.
        let d = dir(&[(RealmId::System(7), 9, 3)]);
        let now = UniverseTick(500);
        let mut l = DemandLedger::default();
        l.record_demand(
            &u_g_s(),
            DemandVerb::Empty,
            UniverseTick(now.0 - 2),
            Fence(1),
        );
        l.record_demand(
            &u_g_s_p(),
            DemandVerb::SpinUp,
            UniverseTick(now.0),
            Fence(1),
        );
        let (actions, delta) = reconcile(
            &l,
            &d,
            &|_n: NodeId| false,
            &no_launch(),
            &t,
            now,
            UniverseTick(0),
            &no_arrivals(),
        );
        // The System is NOT reaped (has a desired descendant); the Planet + its intermediate ancestors
        // that are not yet running get spun up. Crucially: no Kill for the System.
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, LifecycleAction::Kill { .. })),
            "a live child keeps its ancestor un-reaped"
        );
        assert!(!delta.set_draining.contains_key(u_g_s().path()));
    }

    #[test]
    fn reconcile_clears_draining_and_retires_a_dead_undesired_cell() {
        let t = cloud();
        let mut l = DemandLedger::default();
        l.record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(1), Fence(1));
        // Pretend it was draining, then its shard vanished (no head).
        let mut open = LedgerDelta::default();
        open.set_draining
            .insert(sys(7).path().clone(), Some(UniverseTick(2)));
        l.apply_delta(&open);
        let d = dir(&[]); // no head
        let (actions, delta) = reconcile(
            &l,
            &d,
            &|_n: NodeId| false,
            &no_launch(),
            &t,
            UniverseTick(10_000),
            UniverseTick(0),
            &no_arrivals(),
        );
        assert!(actions.is_empty());
        assert_eq!(
            delta.set_draining.get(sys(7).path()),
            Some(&None),
            "a vanished head clears the drain"
        );
        assert!(delta.retire.contains(sys(7).path()));
        l.apply_delta(&delta);
        assert!(l.is_empty(), "the retired cell is dropped");
    }

    #[test]
    fn reconcile_quiesce_blocks_teardown_after_a_restart() {
        let t = cloud();
        let d = dir(&[(RealmId::System(7), 9, 3)]);
        let now = UniverseTick(500);
        let mut l = DemandLedger::default();
        l.record_demand(
            &sys(7),
            DemandVerb::Empty,
            UniverseTick(now.0 - 2),
            Fence(1),
        );
        // The crash-quiesce window has NOT elapsed ⇒ no teardown even though otherwise ready (arm B keeps
        // running realms alive while demands re-accrue).
        let (actions, delta) = reconcile(
            &l,
            &d,
            &|_n: NodeId| false,
            &no_launch(),
            &t,
            now,
            UniverseTick(now.0 + 100),
            &no_arrivals(),
        );
        assert!(actions.is_empty());
        assert!(
            delta.set_draining.is_empty(),
            "quiesce blocks even opening the drain"
        );
    }

    #[test]
    fn reconcile_is_deterministic_run_twice() {
        let t = cloud();
        let mut l = DemandLedger::default();
        l.record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(100), Fence(1));
        l.record_demand(&sys(8), DemandVerb::Empty, UniverseTick(498), Fence(1));
        let d = dir(&[(RealmId::System(8), 9, 3)]);
        let run = || {
            reconcile(
                &l,
                &d,
                &|_n: NodeId| false,
                &no_launch(),
                &t,
                UniverseTick(500),
                UniverseTick(0),
                &no_arrivals(),
            )
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn drive_drain_covers_all_four_transitions() {
        let t = cloud();
        let mut c = LedgerCell::empty(sys(7));
        // (false, None) ⇒ not draining.
        assert_eq!(drive_drain(&c, UniverseTick(10), &t, false), (false, None));
        // (true, None) ⇒ open the drain, no kill.
        assert_eq!(
            drive_drain(&c, UniverseTick(10), &t, true),
            (false, Some(UniverseTick(10)))
        );
        // (true, Some, not elapsed) ⇒ still draining.
        c.draining_since = Some(UniverseTick(10));
        assert_eq!(
            drive_drain(&c, UniverseTick(10 + t.teardown_drain_ticks - 1), &t, true),
            (false, Some(UniverseTick(10)))
        );
        // (true, Some, elapsed) ⇒ KILL.
        assert_eq!(
            drive_drain(&c, UniverseTick(10 + t.teardown_drain_ticks), &t, true),
            (true, Some(UniverseTick(10)))
        );
        // (false, Some) ⇒ clear (rescue).
        assert_eq!(drive_drain(&c, UniverseTick(30), &t, false), (false, None));
    }

    #[test]
    fn ancestor_close_pulls_the_chain_and_terminates_at_root() {
        let mut desired = BTreeMap::new();
        desired.insert(u_g_s_p().path().clone(), u_g_s_p());
        let closed = ancestor_close(&desired);
        assert_eq!(closed.len(), 4); // U, G, S, P
        assert!(closed.contains_key(u_root().path()));
        // A root-only desired terminates immediately (parent None).
        let mut root = BTreeMap::new();
        root.insert(u_root().path().clone(), u_root());
        assert_eq!(ancestor_close(&root).len(), 1);
    }

    #[test]
    fn descendant_predicates_cover_every_arm() {
        let mut closed = BTreeMap::new();
        closed.insert(u_g_s_p().path().clone(), u_g_s_p());
        // A strict descendant present.
        assert!(has_desired_descendant(u_g_s().path(), &closed));
        // A leaf has no descendant in the set.
        assert!(!has_desired_descendant(u_g_s_p().path(), &closed));
        // A deeper SIBLING is not a descendant (starts_with false arm).
        let sibling = deep(&[
            (RealmKindTag::Universe, 0),
            (RealmKindTag::Galaxy, 1),
            (RealmKindTag::System, 8),
            (RealmKindTag::Planet, 3),
        ]);
        let mut c2 = BTreeMap::new();
        c2.insert(sibling.path().clone(), sibling);
        assert!(!has_desired_descendant(u_g_s().path(), &c2));
        // Equal length is NOT a strict descendant (the length false arm).
        assert!(!is_strict_descendant(u_g_s().path(), u_g_s().path()));
    }

    #[test]
    fn spinup_eligible_covers_none_and_cooldown() {
        let t = cloud();
        // Never spawned (no cell) ⇒ eligible.
        assert!(spinup_eligible(None, UniverseTick(0), &t));
        // Cell with the `0` sentinel (demanded, never spawn-attempted) ⇒ eligible even at an EARLY tick
        // inside the cooldown window (the genesis-boot case the E2E exposed).
        let fresh = LedgerCell::empty(sys(7));
        assert_eq!(fresh.spawn_watermark, UniverseTick(0));
        assert!(spinup_eligible(Some(&fresh), UniverseTick(2), &t));
        let mut c = LedgerCell::empty(sys(7));
        c.spawn_watermark = UniverseTick(100);
        assert!(!spinup_eligible(Some(&c), UniverseTick(105), &t)); // within cooldown
        assert!(spinup_eligible(
            Some(&c),
            UniverseTick(100 + t.spinup_cooldown_ticks),
            &t
        )); // past cooldown
    }

    // ==============================================================================================
    // RLM Step 4b — the crash-replay determinism + crash-robustness PROPTEST (mirror R-6d4-A,
    // `crates/io-prod/src/outbox.rs`). Generalizes the single 3f E2E (`realm_lifecycle_e2e.rs`) into a
    // machine-checked proof that `reconcile` + the demand fold + `drive_drain` are DETERMINISTIC and
    // CRASH-ROBUST under ARBITRARY crash / reorder / dup interleavings. It also carries the DIVERGENCE
    // ORACLE that DECIDES whether the deferred durable snapshot (4c / D-RLM-2) is ever needed:
    // INV-SNAPSHOT-SAFETY — the RAM-heal arm (empty ledger on crash, the shipped 4a-armed behaviour) must
    // NEVER reap a realm the durable-snapshot arm (ledger survives the crash) would not. A violation IS the
    // "restart-window strand the CAP-freeze + arm-B don't cover" and the trigger to promote the snapshot.
    //
    // Everything here drives the REAL production kernel (`reconcile`/`DemandLedger`/`apply_delta`) against a
    // REAL `DirectoryCore` for the durable heads; the RAM/durable crash boundary is modelled as (durable
    // heads kept) vs (RAM ledger + liveness + quiesce cleared). The reference oracle (`empty_open`,
    // `model_draining`, `ever_demanded`) is maintained BY HAND from the op history — never by calling the
    // kernel — so the invariants are a genuine cross-check, not a tautology.
    mod crash_replay_proptest {
        use crate::directory::{DirectoryCore, DirectoryTuning};
        use crate::rlm::*;
        use proptest::prelude::*;
        use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};
        use std::cell::Cell;
        use std::collections::{BTreeMap, BTreeSet};
        use vd_core::pose::RealmId;
        use vd_core::realm_coord::RealmCoord;
        use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};
        use vd_core::{Fence, NodeId, UniverseTick};
        use vd_wire::intershard::DemandVerb;
        use vd_wire::seams::directory::{AuthorityRef, DirectoryKey};

        /// SMALL crossable windows so a `vec(op, 0..=48)` run actually crosses `demand_ttl`/`empty_grace`/
        /// `drain`/`cooldown`/`min_dwell`/`recovery_grace` — the reconcile LOGIC is scale-invariant, so a
        /// 1 Hz budget exercises the identical arms a cloud budget does, just reachably.
        fn tuning() -> RlmTuning {
            RlmTuning::cloud(1)
        }

        /// The three model realms: two siblings + a CHILD of `System(7)` so ancestor-closure +
        /// `has_desired_descendant` fire; a run that never demands `System(8)` leaves it never-demanded.
        fn realm_coord(sel: u8) -> RealmCoord {
            match sel % 3 {
                0 => one(RealmKindTag::System, 7),
                1 => one(RealmKindTag::System, 8),
                _ => RealmCoord::from_path(RealmPath::from_levels(vec![
                    RealmLevel::new(RealmKindTag::System, 7),
                    RealmLevel::new(RealmKindTag::Planet, 9),
                ]))
                .expect("two-level path"),
            }
        }
        fn one(kind: RealmKindTag, seed: u64) -> RealmCoord {
            RealmCoord::from_path(RealmPath::from_levels(vec![RealmLevel::new(kind, seed)]))
                .expect("one-level path")
        }
        fn decode_verb(v: u8) -> DemandVerb {
            match v % 4 {
                0 => DemandVerb::SpinUp,
                1 => DemandVerb::KeepAlive,
                2 => DemandVerb::Empty,
                _ => DemandVerb::TearDown,
            }
        }

        #[derive(Clone, Debug)]
        enum Op {
            Demand { realm: u8, verb: u8, fence: u8 },
            Sweep,
            IdleTick { by: u8 },
            GrantHead { realm: u8, node: u8 },
            RevokeHead { realm: u8 },
            MarkDead { node: u8 },
            Crash,
            Reboot,
            ReorderDup { realm: u8, verb: u8, count: u8 },
            // RLM 5e-5 (D9) — the launch-ledger arm. `SpawnConfirm` = the v2 pid:Some intent landed durably
            // (the realm joins the crash-recovery seed); `PartialCrash` = rehydrate DROPS every launch-live
            // entry that never confirmed (the pid:None drop), preserving the F2 cursor so a re-drive mints
            // STRICTLY higher — the ledger-level image of the forked-before-v2 window (D-RLM-5).
            SpawnConfirm { realm: u8 },
            PartialCrash,
        }

        fn op_strategy() -> impl Strategy<Value = Op> {
            prop_oneof![
                4 => (0u8..3, 0u8..4, 0u8..3).prop_map(|(realm, verb, fence)| Op::Demand { realm, verb, fence }),
                4 => Just(Op::Sweep),
                3 => (0u8..8).prop_map(|by| Op::IdleTick { by }),
                2 => (0u8..3, 0u8..3).prop_map(|(realm, node)| Op::GrantHead { realm, node }),
                1 => (0u8..3).prop_map(|realm| Op::RevokeHead { realm }),
                1 => (0u8..3).prop_map(|node| Op::MarkDead { node }),
                1 => Just(Op::Crash),
                2 => Just(Op::Reboot),
                1 => (0u8..3, 0u8..4, 0u8..5).prop_map(|(realm, verb, count)| Op::ReorderDup { realm, verb, count }),
                2 => (0u8..3).prop_map(|realm| Op::SpawnConfirm { realm }),
                1 => Just(Op::PartialCrash),
            ]
        }

        /// One reconcile sweep's observable outcome (node-id-AGNOSTIC — the kernel-determinism scope: the
        /// ledger/head/desired-set + action COUNTS are invariant across a rebuild, raw `NodeId` bytes are
        /// not; here every head node is supplied by an explicit `GrantHead` op, so even node ids are
        /// deterministic within one arm and `SweepRec` compares fully).
        #[derive(Clone, Debug, PartialEq, Eq)]
        struct SweepRec {
            now: u64,
            kills: BTreeSet<RealmPath>,
            force_reaps: usize,
            actions_len: usize,
            in_freeze: bool,
        }

        struct Run {
            sweeps: Vec<SweepRec>,
            killed_ever: BTreeSet<RealmPath>,
            /// RLM 5e-5 (D9): how many `SpinUp` actions were EXECUTED (minted a launch id) per path across
            /// the whole run — the double-spawn observable. A confirmed survivor that is re-minted after a
            /// crash reads `2` here; the D6 control (unseeded recovery) is exactly the arm that produces it.
            minted_per_path: BTreeMap<RealmPath, usize>,
        }

        /// Drive the op history through the REAL kernel. `lose_ram=true` is the shipped RAM-heal (4a) arm
        /// (a `Crash` clears the demand ledger); `lose_ram=false` is the durable-snapshot dry-run arm (the
        /// ledger survives). `seed_survives=true` is the shipped launch recovery (a reboot RESEEDS the live
        /// launch set from the durable `confirmed` rows — the `launch_ledger_seed`→`with_launch_seed` path);
        /// `seed_survives=false` is the RLM 5e-5 **D6 CONTROL** (the `water_only` rebuild — no reseed), which
        /// re-mints a confirmed survivor = the double-spawn the seed prevents. The hand oracle
        /// (`empty_open`/`model_draining`/`ever_demanded`) asserts the per-arm invariants after every op.
        fn run_arm(
            ops: &[Op],
            lose_ram: bool,
            seed_survives: bool,
            cov: Option<&Cell<[bool; 9]>>,
        ) -> Run {
            let t = tuning();
            let grace = t.recovery_grace_ticks;

            let mut ledger = DemandLedger::default();
            let mut dir = DirectoryCore::new(DirectoryTuning::default());
            let mut dead: BTreeSet<NodeId> = BTreeSet::new();
            let mut empty_open: BTreeMap<RealmPath, bool> = BTreeMap::new();
            let mut model_draining: BTreeSet<RealmPath> = BTreeSet::new();
            let mut ever_demanded: BTreeSet<RealmPath> = BTreeSet::new();
            let mut fence_ctr: BTreeMap<RealmId, u64> = BTreeMap::new();
            let mut now: u64 = 1;
            let mut quiesced_until: u64 = 0;
            let mut running = true;

            // RLM 5e-5 launch-ledger hand-model. `launch_live` = the RAM "minted" set fed to `reconcile`'s
            // no-double-spawn guard (drains on head-up, mirroring `reconcile_launches`); it is LOST on a
            // crash. `confirmed` = the DURABLE launch rows (pid:Some) — the crash-recovery seed source; it
            // survives a crash and drains only on Kill/ForceReap. `id_cursor` is the F2 monotone allocator,
            // NEVER reset across a crash (so a re-drive mints strictly higher — no id-reuse).
            let mut launch_live: BTreeMap<RealmPath, BTreeMap<NodeId, UniverseTick>> =
                BTreeMap::new();
            let mut confirmed: BTreeMap<RealmPath, BTreeMap<NodeId, UniverseTick>> =
                BTreeMap::new();
            let mut id_cursor: u64 = 1_000;
            let mut minted_ids: BTreeSet<NodeId> = BTreeSet::new();
            let mut pre_crash_launch: BTreeMap<RealmPath, BTreeMap<NodeId, UniverseTick>> =
                BTreeMap::new();
            let mut minted_per_path: BTreeMap<RealmPath, usize> = BTreeMap::new();

            let mut sweeps = Vec::new();
            let mut killed_ever: BTreeSet<RealmPath> = BTreeSet::new();
            let set_cov = |i: usize| {
                if let Some(c) = cov {
                    let mut f = c.get();
                    f[i] = true;
                    c.set(f);
                }
            };

            for op in ops {
                match op {
                    Op::Demand { realm, verb, fence } => {
                        let coord = realm_coord(*realm);
                        let v = decode_verb(*verb);
                        ledger.record_demand(
                            &coord,
                            v,
                            UniverseTick(now),
                            Fence(u64::from(*fence)),
                        );
                        ever_demanded.insert(coord.path().clone());
                        match v {
                            DemandVerb::Empty => {
                                empty_open.insert(coord.path().clone(), true);
                            }
                            DemandVerb::SpinUp | DemandVerb::KeepAlive => {
                                empty_open.insert(coord.path().clone(), false);
                            }
                            DemandVerb::TearDown => {}
                        }
                    }
                    Op::IdleTick { by } => now = now.saturating_add(u64::from(*by)),
                    Op::GrantHead { realm, node } => {
                        let rid = realm_coord(*realm).lowered();
                        let f = fence_ctr.entry(rid).or_insert(0);
                        *f += 1;
                        dir.grant(
                            DirectoryKey::Realm(rid),
                            AuthorityRef::Shard(NodeId(100 + u64::from(*node))),
                            Fence(*f),
                            UniverseTick(now),
                        );
                    }
                    Op::RevokeHead { realm } => {
                        let rid = realm_coord(*realm).lowered();
                        if let Some(rec) = dir.head(DirectoryKey::Realm(rid)) {
                            let _ = dir.revoke(DirectoryKey::Realm(rid), rec.fence);
                        }
                    }
                    Op::MarkDead { node } => {
                        dead.insert(NodeId(100 + u64::from(*node)));
                    }
                    Op::Crash => {
                        if !ledger.is_empty() {
                            set_cov(0);
                        }
                        dead.clear();
                        if lose_ram {
                            ledger = DemandLedger::default();
                            empty_open.clear();
                            model_draining.clear();
                        }
                        // The RAM launch-live set is LOST on a crash; the durable `confirmed` rows survive.
                        // Capture the pre-crash live set for the SUPERSET check the reboot reseed makes.
                        pre_crash_launch = launch_live.clone();
                        launch_live.clear();
                        running = false;
                    }
                    Op::Reboot => {
                        running = true;
                        quiesced_until = now.saturating_add(grace);
                        now = now.saturating_add(1);
                        // The shipped launch recovery: RESEED the live set from the durable `confirmed` rows
                        // (`launch_ledger_seed` → `with_launch_seed`). The D6 CONTROL (`seed_survives=false`,
                        // the `water_only` rebuild) SKIPS this, leaving `launch_live` empty so a survivor is
                        // re-minted = the double-spawn the seed prevents.
                        if seed_survives {
                            launch_live = confirmed.clone();
                            // INV-LAUNCH-SNAPSHOT-SUPERSET: every confirmed survivor that was live pre-crash
                            // is recovered into the reseeded set (the seed never LOSES a durable survivor).
                            for path in pre_crash_launch.keys() {
                                if confirmed.contains_key(path) {
                                    assert!(
                                        launch_live.contains_key(path),
                                        "INV-LAUNCH-SNAPSHOT-SUPERSET: reseed dropped a confirmed survivor"
                                    );
                                    set_cov(8);
                                }
                            }
                        }
                    }
                    // ReorderDup drives the standalone fold-order check (in `model_check`); a world no-op.
                    Op::ReorderDup { .. } => {}
                    Op::SpawnConfirm { realm } => {
                        // v2 pid:Some landed durably: the realm's live launch row joins the durable seed. A
                        // confirm for a realm with no in-flight launch is a benign no-op (the None arm). Every
                        // `launch_live` entry is non-empty by construction (minted with a node, drained as a
                        // whole entry), so `get(..).is_some()` alone is the launch-present test — no
                        // `!is_empty()` sub-check (its false arm would be an uncoverable region, HR5).
                        let path = realm_coord(*realm).path().clone();
                        if let Some(nodes) = launch_live.get(&path) {
                            confirmed.insert(path, nodes.clone());
                            set_cov(6);
                        }
                    }
                    Op::PartialCrash => {
                        // rehydrate's pid:None DROP, modeled directly: every launch-live entry that never
                        // confirmed is removed (the forked-before-v2 window), while `id_cursor` is preserved
                        // so a subsequent re-drive mints STRICTLY higher — the ledger image of D-RLM-5.
                        let before = launch_live.len();
                        launch_live.retain(|p, _| confirmed.contains_key(p));
                        if launch_live.len() < before {
                            set_cov(7);
                        }
                    }
                    Op::Sweep => {
                        if !running {
                            continue; // the orchestrator is down between Crash and Reboot.
                        }
                        let dead_ref = &dead;
                        let is_dead = |n: NodeId| dead_ref.contains(&n);
                        let (actions, delta) = reconcile(
                            &ledger,
                            &dir,
                            &is_dead,
                            &launch_live,
                            &t,
                            UniverseTick(now),
                            UniverseTick(quiesced_until),
                            &BTreeSet::new(),
                        );
                        let in_freeze = now < quiesced_until;
                        let mut kills: BTreeSet<RealmPath> = BTreeSet::new();
                        let mut force_reaps = 0usize;
                        for a in &actions {
                            match a {
                                LifecycleAction::Kill { path, .. } => {
                                    // INV-CRASH-NO-REAP: the ARMED freeze blocks every kill in its window.
                                    assert!(
                                        !in_freeze,
                                        "INV-CRASH-NO-REAP: a Kill fired inside the freeze (now={now} < quiesce={quiesced_until})"
                                    );
                                    // INV-NO-STRAND: a kill only ever targets a realm the INDEPENDENT oracle
                                    // saw affirmatively report Empty (no wrongful reap of a live realm).
                                    assert_eq!(
                                        empty_open.get(path),
                                        Some(&true),
                                        "INV-NO-STRAND: killed a realm that never reported Empty in the oracle"
                                    );
                                    // INV-DELTA-CONSISTENT: two-phase — a kill implies a drain opened on a
                                    // PRIOR sweep (never a first-sweep synchronous kill; the BUG-C veto).
                                    assert!(
                                        model_draining.contains(path),
                                        "INV-DELTA: a Kill fired without a prior-opened drain window"
                                    );
                                    kills.insert(path.clone());
                                }
                                LifecycleAction::ForceReap { .. } => force_reaps += 1,
                                LifecycleAction::SpinUp { coord } => {
                                    // EXECUTE the launch (the runtime's mint step): the launch-model image of
                                    // `reconcile_launches` recording a fresh minted id.
                                    let path = coord.path().clone();
                                    let rid = coord.lowered();
                                    // INV-NO-DOUBLE-SPAWN: the kernel guard (rlm.rs:612-614) held — a SpinUp is
                                    // emitted ONLY for a realm with no live head AND no in-flight launch (the
                                    // model never leaves a Some-empty entry, so `contains_key` == launch_present).
                                    assert!(
                                        dir.head(DirectoryKey::Realm(rid)).is_none(),
                                        "INV-NO-DOUBLE-SPAWN: SpinUp for a realm that already has a head"
                                    );
                                    assert!(
                                        !launch_live.contains_key(&path),
                                        "INV-NO-DOUBLE-SPAWN: SpinUp for a realm already launching"
                                    );
                                    // F2 mint: strictly monotone, NEVER reset across a crash — INV-NO-ID-REUSE
                                    // (`insert` returns false on a reused id, tripping the assert).
                                    let n = NodeId(id_cursor);
                                    id_cursor += 1;
                                    assert!(
                                        minted_ids.insert(n),
                                        "INV-NO-ID-REUSE: a mint reused an F2 id"
                                    );
                                    launch_live
                                        .entry(path.clone())
                                        .or_default()
                                        .insert(n, UniverseTick(now));
                                    *minted_per_path.entry(path).or_insert(0) += 1;
                                }
                            }
                        }
                        // cov[1]: the freeze demonstrably FIRED — a live-head, oracle-empty realm swept inside
                        // the window and was NOT reaped (the exact strand 4a closes).
                        let mut live_empty_open = false;
                        for (path, cell) in ledger.iter() {
                            let head = dir.head(DirectoryKey::Realm(cell.coord.lowered()));
                            if running_live(head, &is_dead) & (empty_open.get(path) == Some(&true))
                            {
                                live_empty_open = true;
                            }
                        }
                        if in_freeze & live_empty_open & kills.is_empty() {
                            set_cov(1);
                        }
                        if !kills.is_empty() {
                            set_cov(2);
                        }
                        if force_reaps > 0 {
                            set_cov(3);
                        }
                        // Apply the delta + actions to advance the world (the runtime's EXECUTE step).
                        ledger.apply_delta(&delta);
                        for (path, v) in &delta.set_draining {
                            if v.is_some() {
                                model_draining.insert(path.clone());
                            } else {
                                model_draining.remove(path);
                            }
                        }
                        for path in &delta.retire {
                            model_draining.remove(path);
                        }
                        for a in &actions {
                            match a {
                                LifecycleAction::Kill { path, fence, .. } => {
                                    let rid =
                                        path.realm_id().expect("a lifecycle path is non-empty");
                                    let _ = dir.revoke(DirectoryKey::Realm(rid), *fence);
                                    ledger.mark_reaped(path, UniverseTick(now));
                                    killed_ever.insert(path.clone());
                                    // The reaped realm's durable launch row is deleted (kill_realm) — so a
                                    // later re-demand mints a FRESH id, never resurrecting the retired one.
                                    launch_live.remove(path);
                                    confirmed.remove(path);
                                }
                                LifecycleAction::ForceReap { path, fence, .. } => {
                                    let rid =
                                        path.realm_id().expect("a lifecycle path is non-empty");
                                    let _ = dir.revoke(DirectoryKey::Realm(rid), *fence);
                                    // A force-reaped zombie's row is dropped too, so the self-heal SpinUp on
                                    // the next sweep mints a fresh id (the retired zombie id is never reused).
                                    launch_live.remove(path);
                                    confirmed.remove(path);
                                }
                                LifecycleAction::SpinUp { .. } => {}
                            }
                        }
                        // Drain the launch bookkeeping (the launch-model image of `reconcile_launches`): a
                        // minted node whose realm now has a head is no longer "launching", so drop the whole
                        // entry — the guard then reads head-present (not launch-present) and never re-spins it.
                        launch_live.retain(|path, _| {
                            let rid = path.realm_id().expect("a lifecycle path is non-empty");
                            dir.head(DirectoryKey::Realm(rid)).is_none()
                        });
                        // INV-BOUNDED: the ledger never exceeds the realms ever demanded (the retire path
                        // keeps the RAM self-heal from leaking — the property that matters at 100K scale).
                        // Both `len()`s live in the CONDITION (always evaluated ⇒ covered); a STATIC message
                        // keeps the failure path free of uncoverable method-call regions (HR5).
                        assert!(
                            ledger.len() <= ever_demanded.len(),
                            "INV-BOUNDED: the ledger grew past the distinct-demanded realm count"
                        );
                        sweeps.push(SweepRec {
                            now,
                            kills,
                            force_reaps,
                            actions_len: actions.len(),
                            in_freeze,
                        });
                        now = now.saturating_add(1);
                    }
                }
            }
            Run {
                sweeps,
                killed_ever,
                minted_per_path,
            }
        }

        /// `record_demand`'s fold is order- AND dup-independent — every field it writes is a `max`/latch
        /// (`last_demand_tick`, `last_empty_tick`, `last_fence`), so the whole cell (and thus every predicate)
        /// is identical across fold orders. This is the invariant whose VIOLATION this proptest first caught:
        /// the original streak-based `empty_confirmed` (`last_demand < first_empty_tick`) was order-SENSITIVE
        /// for a same-tick `KeepAlive`+`Empty`, so a parent's demand and a child's empty racing on one coord
        /// could flip a reap decision. Fold a mixed multiset forward vs reversed+duplicated and assert every
        /// probed predicate agrees — generalizing the arm-C reorder unit tests over the fuzzer's multisets.
        fn assert_fold_order_independent(realm: u8, verb: u8, count: u8) {
            let coord = realm_coord(realm);
            let n = u64::from(count % 6) + 1;
            let mut demands: Vec<(DemandVerb, u64, u64)> = Vec::new();
            for k in 0..n {
                demands.push((decode_verb(verb), 10 + 2 * k, k + 1));
                demands.push((DemandVerb::Empty, 11 + 2 * k, 1));
                demands.push((DemandVerb::KeepAlive, 10 + 2 * k, 2));
            }
            let fold = |order: &[(DemandVerb, u64, u64)]| -> LedgerCell {
                let mut l = DemandLedger::default();
                for (v, tk, f) in order {
                    l.record_demand(&coord, *v, UniverseTick(*tk), Fence(*f));
                }
                l.get(coord.path()).cloned().expect("folded cell exists")
            };
            let forward = fold(&demands);
            let mut shuffled = demands.clone();
            shuffled.reverse();
            // Re-push the FIRST demand (`k=0`: verb @ tick 10, fence 1) verbatim ⇒ the fold sees a duplicate,
            // proving idempotence. Constructed (not `demands.first()`) so there is no never-taken empty-vec
            // arm — `demands` always holds ≥3 tuples (`n >= 1`), so an `Option`/index here is dead (HR5).
            shuffled.push((decode_verb(verb), 10, 1));
            let reversed = fold(&shuffled);
            // The order-INVARIANT projections (max/max/lex-max).
            assert_eq!(forward.last_demand_tick, reversed.last_demand_tick);
            assert_eq!(forward.last_empty_tick, reversed.last_empty_tick);
            assert_eq!(forward.last_fence, reversed.last_fence);
            // The DECISION agrees at probes spanning the whole demand window.
            let t = tuning();
            for probe in [0u64, 12, 14, 16, 20, 40] {
                let now = UniverseTick(probe);
                assert_eq!(
                    empty_confirmed(&forward, now, t.empty_grace_ticks),
                    empty_confirmed(&reversed, now, t.empty_grace_ticks),
                    "INV-FOLD-ORDER: empty_confirmed disagrees across fold order at {probe}"
                );
                assert_eq!(
                    demanded_recently(&forward, now, t.demand_ttl_ticks),
                    demanded_recently(&reversed, now, t.demand_ttl_ticks),
                    "INV-FOLD-ORDER: demanded_recently disagrees across fold order at {probe}"
                );
            }
        }

        /// The whole battery for one op history. INV-DET (run-twice), INV-SNAPSHOT-SAFETY (the 4c decider),
        /// INV-FOLD-ORDER, and the per-arm invariants asserted inside `run_arm`.
        fn model_check(ops: &[Op], cov: &Cell<[bool; 9]>) {
            // INV-DET — `reconcile` is a pure deterministic `f(...)`; the whole crash-interleaved history
            // replays byte-identically (BTreeMap keys, no wall clock / default hasher). Generalizes
            // `reconcile_is_deterministic_run_twice` over arbitrary crash histories. `seed_survives=true`
            // is the shipped launch recovery (the D6 CONTROL — no reseed — is exercised by its own witness).
            let a1 = run_arm(ops, true, true, Some(cov));
            let a2 = run_arm(ops, true, true, None);
            assert_eq!(
                a1.sweeps, a2.sweeps,
                "INV-DET: sweep trace diverged across replay"
            );
            assert_eq!(
                a1.killed_ever, a2.killed_ever,
                "INV-DET: whole-run reap set diverged across replay"
            );
            // INV-SNAPSHOT-EQUIVALENCE / the DIVERGENCE ORACLE (the DECIDER for 4c): run the shipped RAM-heal
            // arm (empty ledger on crash) and the durable-snapshot dry-run arm (ledger survives) over the SAME
            // history plus a fixed SETTLING TAIL (age stale demands out, resolve open drains — no new demands).
            // Because `recovery_grace == demand_ttl` (RlmTuning::cloud), any pre-crash demand the snapshot arm
            // clings to has AGED OUT exactly when the freeze lifts, so the two arms REACH THE SAME REAP-SET.
            // Equal ⇒ the durable snapshot changes no teardown decision the armed freeze + arm-B don't already
            // achieve ⇒ D-RLM-2 stays deferred (the expected outcome, §0). A DIFFERENCE is precisely "the
            // restart-window strand the CAP-freeze + arm-B don't cover" — the machine-checked trigger to
            // promote §3. A raw per-sweep subset would false-fail (the snapshot arm reaps up to one demand-TTL
            // LATER), so the invariant is equality AFTER settling, not a mid-flight subset.
            let mut settled: Vec<Op> = ops.to_vec();
            settled.push(Op::Reboot); // ensure the reconciler is UP (a run may end mid-crash) before settling.
            for _ in 0..6 {
                settled.push(Op::IdleTick { by: 50 });
                settled.push(Op::Sweep);
            }
            let ram = run_arm(&settled, true, true, None);
            let snap = run_arm(&settled, false, true, None);
            assert_eq!(
                ram.killed_ever, snap.killed_ever,
                "INV-SNAPSHOT-EQUIVALENCE: RAM-heal and durable-snapshot reap-sets DIFFER after settling — promote D-RLM-2 (4c)"
            );
            // INV-FOLD-ORDER + the op-derived cov floor.
            for (i, op) in ops.iter().enumerate() {
                if let Op::ReorderDup { realm, verb, count } = op {
                    assert_fold_order_independent(*realm, *verb, *count);
                    if *count >= 2 {
                        let mut f = cov.get();
                        f[4] = true;
                        cov.set(f);
                    }
                }
                // cov[5]: a Sweep in the tick immediately after Reboot — the off-tick-race position where the
                // freeze MUST hold (a Crash landing just before the reconcile scan).
                if matches!(op, Op::Reboot)
                    && ops.get(i + 1).is_some_and(|n| matches!(n, Op::Sweep))
                {
                    let mut f = cov.get();
                    f[5] = true;
                    cov.set(f);
                }
            }
        }

        #[test]
        fn rlm_reconcile_survives_arbitrary_crash_interleavings_deterministically() {
            let cov = std::rc::Rc::new(Cell::new([false; 9]));
            let cov_run = std::rc::Rc::clone(&cov);
            let mut runner = TestRunner::new_with_rng(
                Config {
                    cases: 1024,
                    ..Config::default()
                },
                TestRng::from_seed(RngAlgorithm::ChaCha, &[0x72u8; 32]),
            );
            // `.expect` (not `if let Err { panic! }`): the never-taken failure branch's panic lives INSIDE
            // `Result::expect` (std, uninstrumented for this crate) rather than as an uncoverable region here,
            // and `expect` prints the `Err`'s Debug (the shrunk minimal case) automatically (HR5-clean).
            runner
                .run(&prop::collection::vec(op_strategy(), 0..=48), move |ops| {
                    model_check(&ops, &cov_run);
                    Ok(())
                })
                .expect("crash-replay proptest failed (shrunk minimal case follows)");
            let f = cov.get();
            // The RELIABLY-random anti-vacuity floor (these arms hit across the 1024 fixed-seed cases). The
            // TIMING-SPECIFIC arms (cov1 freeze-fired, cov2 a kill, cov3 a zombie force-reap) need a tight
            // Empty→Sweep→Sweep / MarkDead cadence that random ops seldom line up, so each is PINNED by a
            // deterministic named witness below (which asserts its own flag) — the union is the full floor.
            assert!(f[0], "cov0: a Crash lost a non-empty ledger");
            assert!(f[4], "cov4: a ReorderDup folded a ≥2 multiset");
            assert!(f[5], "cov5: a Sweep landed in the tick right after Reboot");
        }

        // ---- deterministic named witnesses (regressions kept alongside the fuzzer) -------------------

        #[test]
        fn witness_crash_before_sweep_reaps_nothing_in_grace() {
            let cov = Cell::new([false; 9]);
            // A live, confirmed-empty, past-window realm crashes then a Sweep lands right after reboot: the
            // ARMED freeze blocks the reap that would otherwise fire.
            model_check(
                &[
                    Op::Demand {
                        realm: 0,
                        verb: 2,
                        fence: 1,
                    }, // Empty
                    Op::GrantHead { realm: 0, node: 0 },
                    Op::IdleTick { by: 6 },
                    Op::Crash,
                    Op::Reboot,
                    Op::Demand {
                        realm: 0,
                        verb: 2,
                        fence: 1,
                    }, // re-report Empty post-reboot
                    Op::Sweep, // inside the freeze ⇒ no kill
                ],
                &cov,
            );
            // The freeze fired: a live, oracle-empty realm swept inside the post-reboot window was not reaped
            // (cov1). A post-reboot `Empty` MUST precede the sweep for the realm to have a cell to protect, so
            // this witness cannot also be the strict Reboot→Sweep adjacency (cov5) — the fuzzer pins that.
            assert!(
                cov.get()[1],
                "the freeze fired on a would-be-reapable realm"
            );
        }

        #[test]
        fn witness_a_confirmed_empty_realm_is_reaped_after_the_drain() {
            let cov = Cell::new([false; 9]);
            // No crash ⇒ the freeze is unarmed (quiesce 0). A live, out-of-AoI, confirmed-empty realm past its
            // dwell drains one window then is KILLED on the next sweep — the reap path (cov2), exercised in a
            // DEBUG test so `coverage-fast` covers the kill/revoke/mark_reaped arms (the soak is release-only).
            model_check(
                &[
                    Op::IdleTick { by: 6 },              // now → 7 (past min_dwell 4)
                    Op::GrantHead { realm: 0, node: 0 }, // a live head
                    Op::Demand {
                        realm: 0,
                        verb: 2,
                        fence: 1,
                    }, // Empty (undesired, confirmed)
                    Op::Sweep,                           // opens the drain (quiesced ⇒ ready)
                    Op::Demand {
                        realm: 0,
                        verb: 2,
                        fence: 1,
                    }, // keep Empty fresh through the drain
                    Op::Sweep,                           // drain elapsed ⇒ KILL
                ],
                &cov,
            );
            assert!(cov.get()[2], "a Kill was emitted (the reap path)");
        }

        #[test]
        fn witness_snapshot_and_ramheal_agree_on_a_simple_trace() {
            let cov = Cell::new([false; 9]);
            // A plain spin/keepalive trace: both arms decide identically (no divergence) — the expected 4c
            // outcome. `model_check`'s INV-SNAPSHOT-EQUIVALENCE asserts the settled reap-sets are equal.
            model_check(
                &[
                    Op::Demand {
                        realm: 0,
                        verb: 0,
                        fence: 1,
                    }, // SpinUp
                    Op::Sweep,
                    Op::GrantHead { realm: 0, node: 0 },
                    Op::Demand {
                        realm: 0,
                        verb: 1,
                        fence: 1,
                    }, // KeepAlive
                    Op::Sweep,
                ],
                &cov,
            );
        }

        #[test]
        fn witness_zombie_after_reboot_force_reaps() {
            let cov = Cell::new([false; 9]);
            // A demanded realm with a live head whose owner latches dead ⇒ the sweep force-reaps the zombie.
            model_check(
                &[
                    Op::Demand {
                        realm: 0,
                        verb: 0,
                        fence: 1,
                    },
                    Op::GrantHead { realm: 0, node: 0 },
                    Op::MarkDead { node: 0 },
                    Op::Sweep,
                ],
                &cov,
            );
            assert!(cov.get()[3], "the zombie force-reap path fired");
        }

        #[test]
        fn witness_reorder_dup_folds_identically() {
            let cov = Cell::new([false; 9]);
            model_check(
                &[Op::ReorderDup {
                    realm: 2,
                    verb: 2,
                    count: 4,
                }],
                &cov,
            );
            assert!(cov.get()[4], "a ≥2 reorder/dup multiset was folded");
        }

        #[test]
        fn witness_crash_mid_drain_reopens_not_double_kills() {
            let cov = Cell::new([false; 9]);
            // A realm draining toward a kill crashes mid-drain: the post-reboot sweep re-opens the drain
            // under the fresh freeze rather than double-killing (INV-DELTA + INV-CRASH-NO-REAP).
            model_check(
                &[
                    Op::Demand {
                        realm: 0,
                        verb: 2,
                        fence: 1,
                    }, // Empty
                    Op::GrantHead { realm: 0, node: 0 },
                    Op::IdleTick { by: 6 },
                    Op::Demand {
                        realm: 0,
                        verb: 2,
                        fence: 1,
                    },
                    Op::Sweep, // opens the drain
                    Op::Crash, // mid-drain crash
                    Op::Reboot,
                    Op::Demand {
                        realm: 0,
                        verb: 2,
                        fence: 1,
                    },
                    Op::Sweep, // re-opens under the freeze; never a double-kill
                ],
                &cov,
            );
        }

        #[test]
        fn witness_ancestor_closure_survives_crash() {
            let cov = Cell::new([false; 9]);
            // Demand the CHILD (realm 2 = Planet under System(7)); ancestor-closure pulls System(7) alive.
            // A crash + re-demand re-derives the same closure (the level-triggered self-heal).
            model_check(
                &[
                    Op::Demand {
                        realm: 2,
                        verb: 0,
                        fence: 1,
                    },
                    Op::Sweep, // spins up the child + its ancestor
                    Op::Crash,
                    Op::Reboot,
                    Op::Demand {
                        realm: 2,
                        verb: 0,
                        fence: 1,
                    },
                    Op::Sweep,
                ],
                &cov,
            );
        }

        #[test]
        fn witness_a_drain_opened_then_a_demand_rescues_it() {
            let cov = Cell::new([false; 9]);
            // A realm opens its teardown drain (confirmed-empty, out of AoI) then a KeepAlive RE-ACCRUES
            // before the drain elapses ⇒ the next sweep CLEARS `draining_since` (the BUG-C veto — a
            // login-into-draining rescue). Exercises the drain-CLEAR delta arm (`set_draining = None`), which
            // the crash/kill witnesses never take (they crash mid-drain or kill through it, never rescue).
            model_check(
                &[
                    Op::IdleTick { by: 6 },
                    Op::GrantHead { realm: 0, node: 0 },
                    Op::Demand {
                        realm: 0,
                        verb: 2,
                        fence: 1,
                    }, // Empty ⇒ confirmed
                    Op::Sweep, // teardown-ready ⇒ opens the drain (draining_since = Some)
                    Op::Demand {
                        realm: 0,
                        verb: 1,
                        fence: 1,
                    }, // KeepAlive rescues within the drain window
                    Op::Sweep, // ready=false ⇒ the drain CLEARS (draining_since = None)
                ],
                &cov,
            );
        }

        #[test]
        fn witness_crash_with_empty_ledger_then_a_sweep_while_down() {
            let cov = Cell::new([false; 9]);
            // The two boundary arms the demand-first witnesses never take: a `Crash` with NOTHING demanded yet
            // (the empty-ledger side of the `!is_empty()` cov0 check) IMMEDIATELY followed by a `Sweep` while
            // the orchestrator is DOWN (running=false ⇒ the sweep is skipped, the `continue`).
            model_check(&[Op::Crash, Op::Sweep, Op::Reboot, Op::Sweep], &cov);
        }

        // ---- RLM 5e-5 (D9) launch-ledger witnesses -------------------------------------------------

        #[test]
        fn witness_spawnconfirm_then_crash_survives_in_the_seed() {
            let cov = Cell::new([false; 9]);
            // A realm is spun up (minted), CONFIRMED (v2 pid:Some durable), then the orchestrator crashes. The
            // reboot RESEEDS the live launch set from the durable `confirmed` rows, so the survivor is
            // recovered (INV-LAUNCH-SNAPSHOT-SUPERSET) — the crash-recovery seed that suppresses a re-spawn.
            model_check(
                &[
                    Op::IdleTick { by: 6 }, // now → 7 (past min_dwell) so the SpinUp is eligible
                    Op::Demand {
                        realm: 0,
                        verb: 0,
                        fence: 1,
                    }, // SpinUp
                    Op::Sweep,              // mints the launch into launch_live
                    Op::SpawnConfirm { realm: 0 }, // v2 durable ⇒ joins the seed (cov6)
                    Op::Crash,              // RAM launch lost; confirmed survives
                    Op::Reboot,             // reseed from confirmed ⇒ SUPERSET recovers it (cov8)
                ],
                &cov,
            );
            assert!(cov.get()[6], "a SpawnConfirm recorded a durable launch row");
            assert!(
                cov.get()[8],
                "the reboot reseed recovered a confirmed survivor"
            );
        }

        #[test]
        fn witness_partialcrash_drops_unconfirmed_keeps_confirmed_and_redrives() {
            let cov = Cell::new([false; 9]);
            // Two realms are launched; ONE is confirmed (v2 durable), the other is NOT. A PartialCrash (the
            // rehydrate pid:None drop) removes ONLY the unconfirmed one; the confirmed one survives. The
            // dropped realm re-drives on the next sweep and — because the F2 cursor is preserved — mints a
            // STRICTLY higher id (INV-NO-ID-REUSE, asserted inside run_arm). Exercises BOTH retain arms.
            model_check(
                &[
                    Op::IdleTick { by: 6 },
                    Op::Demand {
                        realm: 0,
                        verb: 0,
                        fence: 1,
                    }, // SpinUp realm 0
                    Op::Demand {
                        realm: 1,
                        verb: 0,
                        fence: 1,
                    }, // SpinUp realm 1
                    Op::Sweep,                     // mint both
                    Op::SpawnConfirm { realm: 0 }, // confirm ONLY realm 0 (kept across PartialCrash)
                    Op::PartialCrash,              // drop realm 1 (unconfirmed) — cov7
                    Op::Demand {
                        realm: 1,
                        verb: 0,
                        fence: 1,
                    }, // re-desire realm 1
                    Op::Sweep,                     // re-mint realm 1 at a strictly higher id
                ],
                &cov,
            );
            assert!(cov.get()[7], "a PartialCrash dropped an unconfirmed launch");
        }

        #[test]
        fn witness_unseeded_reconciler_double_spawns_a_survivor() {
            // The D6 CONTROL (the anti-vacuity twin): the SAME crash history proves the launch-recovery seed
            // is LOAD-BEARING. WITH the seed (a reboot reseeds launch_live from `confirmed` — the shipped
            // `launch_ledger_seed`→`with_launch_seed` path) the confirmed survivor is recognized ⇒ minted
            // ONCE. WITHOUT it (`seed_survives=false` — the `water_only` rebuild) the reconciler re-spins the
            // survivor ⇒ minted TWICE = the double-spawn the seed prevents. `minted_per_path` is the observable.
            let ops = [
                Op::IdleTick { by: 6 },
                Op::Demand {
                    realm: 0,
                    verb: 0,
                    fence: 1,
                }, // SpinUp
                Op::Sweep,                     // mint #1 into launch_live
                Op::SpawnConfirm { realm: 0 }, // v2 durable
                Op::Crash,                     // RAM launch lost; confirmed (durable) survives
                Op::Reboot,
                Op::Demand {
                    realm: 0,
                    verb: 0,
                    fence: 1,
                }, // re-desire (the RAM demand ledger was cleared by the crash)
                Op::Sweep, // no head + (no seed ⇒ re-mint | seed ⇒ suppressed)
            ];
            let path = realm_coord(0).path().clone();
            let seeded = run_arm(&ops, true, true, None);
            let unseeded = run_arm(&ops, true, false, None);
            assert_eq!(
                seeded.minted_per_path.get(&path).copied(),
                Some(1),
                "WITH the recovery seed the confirmed survivor is minted exactly once"
            );
            assert_eq!(
                unseeded.minted_per_path.get(&path).copied(),
                Some(2),
                "WITHOUT the seed the survivor is re-spun — the double-spawn the seed prevents"
            );
        }

        /// The SOAK (release-gated, mirror the SPIKE-3a `just <name> release` latency-gate pattern): one very
        /// long op stream with periodic crashes + continuous re-accrue, asserting the full battery
        /// throughout. A SUSTAINED INV-SNAPSHOT-SAFETY / INV-NO-STRAND / leak violation is the concrete
        /// trigger to promote the deferred durable snapshot (§3 / D-RLM-2). `just rlm-soak` runs it.
        #[test]
        #[cfg(not(debug_assertions))]
        fn rlm_soak_long_crash_stream_stays_consistent() {
            // A fixed-seed xorshift (no wall clock / no rand dep — deterministic repro) drives ~30k ops.
            let mut state: u64 = 0x7272_7272_7272_7272;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let mut ops: Vec<Op> = Vec::with_capacity(30_000);
            for _ in 0..30_000 {
                let r = next();
                let realm = (r & 0x3) as u8 % 3;
                let node = ((r >> 2) & 0x3) as u8 % 3;
                let verb = ((r >> 4) & 0x3) as u8;
                ops.push(match (r >> 8) % 18 {
                    0 | 1 | 2 | 3 => Op::Demand {
                        realm,
                        verb,
                        fence: 1,
                    },
                    4 | 5 | 6 | 7 => Op::Sweep,
                    8 | 9 => Op::IdleTick {
                        by: ((r >> 12) & 0x7) as u8,
                    },
                    10 | 11 => Op::GrantHead { realm, node },
                    12 => Op::RevokeHead { realm },
                    13 => Op::MarkDead { node },
                    14 => Op::Crash,
                    15 => Op::SpawnConfirm { realm },
                    16 => Op::PartialCrash,
                    _ => Op::Reboot,
                });
            }
            let cov = Cell::new([false; 9]);
            model_check(&ops, &cov);
        }
    }
}
