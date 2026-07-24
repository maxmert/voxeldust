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
        }
    }
}

impl RlmTuning {
    /// A live cloud budget derived from the tick rate (mirrors `LivenessTuning::cloud` / `DirectoryTuning`
    /// deriving from `tick_hz`, no magic literals in systems). The ordering invariant
    /// (`demand_ttl > drain + cooldown`, `empty_grace >= 1`, all non-zero) holds by construction — asserted
    /// by [`validate`](RlmTuning::validate) at boot. `hz` = ticks per second; the windows are seconds-scaled.
    #[must_use]
    pub fn cloud(tick_hz: u32) -> RlmTuning {
        let hz = u64::from(tick_hz).max(1);
        // ~1s demand freshness, ~0.5s empty-report tolerance, a drain + cooldown that together stay well
        // inside the demand TTL so a re-asserted demand always aborts a drain before a kill fires.
        let drain = (hz / 2).max(1);
        let cooldown = hz.max(1);
        let spinup_cooldown = hz.max(1);
        let launch_ttl = (hz * 3).max(1);
        RlmTuning {
            demand_ttl_ticks: (hz * 4).max(drain + cooldown + 1),
            empty_grace_ticks: (hz / 2).max(1),
            teardown_cooldown_ticks: cooldown,
            teardown_drain_ticks: drain,
            spinup_cooldown_ticks: spinup_cooldown,
            launch_ttl_ticks: launch_ttl,
            reconcile_interval_ticks: 1,
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
            | (self.launch_ttl_ticks == 0);
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
    /// An ACTIVE reconciler with a zero window (any of the six timing fields) — every window must be > 0.
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
    /// Freshest SpinUp | KeepAlive | player-spawn demand tick (`Empty` does NOT refresh it).
    pub last_demand_tick: UniverseTick,
    /// Most recent `Empty` self-report (retained for audit; the streak below drives `empty_confirmed`).
    pub last_empty_tick: Option<UniverseTick>,
    /// Start of the CURRENT empty streak (the grace/`arm-C` anchor). Cleared by a non-Empty demand
    /// at-or-after it; a RE-asserted `Empty` does NOT advance it.
    pub first_empty_tick: Option<UniverseTick>,
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
    /// A fresh cell for `coord` with no demands yet (all ticks zero, no empty streak).
    #[must_use]
    fn empty(coord: RealmCoord) -> LedgerCell {
        LedgerCell {
            coord,
            last_demand_tick: UniverseTick(0),
            last_empty_tick: None,
            first_empty_tick: None,
            spawn_watermark: UniverseTick(0),
            teardown_watermark: UniverseTick(0),
            draining_since: None,
            last_fence: Fence(0),
        }
    }
}

/// The per-realm demand ledger (RAM-only in Step 3; self-heals on restart via re-asserted demands — a
/// durable snapshot is the reserved Step-4 `StoreKey::Rlm`). Keyed by the lineage [`RealmPath`] (`Ord`,
/// collision-free) for determinism.
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

/// Refresh a non-Empty demand: bump `last_demand_tick`, and CLEAR the empty streak iff this demand is
/// at-or-after the streak start (a stale reordered SpinUp must NOT clear a fresher `Empty` — determinism
/// #8). Monomorphic (all branching here, HR5).
fn refresh_demand(cell: &mut LedgerCell, tick: UniverseTick) {
    cell.last_demand_tick = cell.last_demand_tick.max(tick);
    // Clear the empty streak iff one is open AND this demand is at-or-after its start.
    if cell.first_empty_tick.is_some_and(|start| tick >= start) {
        cell.first_empty_tick = None;
    }
}

/// Latch an `Empty`: bump `last_empty_tick`, and START the streak if none is open (a re-asserted `Empty`
/// does NOT advance the streak anchor, so grace ages from the streak start). Monomorphic.
fn latch_empty(cell: &mut LedgerCell, tick: UniverseTick) {
    cell.last_empty_tick = Some(cell.last_empty_tick.map_or(tick, |t| t.max(tick)));
    if cell.first_empty_tick.is_none() {
        cell.first_empty_tick = Some(tick);
    }
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
    /// The head is mid re-home saga (`in_transfer.is_some()`) — never reap a landing realm.
    pub in_transfer: bool,
    /// The orchestrator-crash quiesce window has elapsed (`now >= rlm_quiesced_until`).
    pub quiesced: bool,
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

/// The realm has AFFIRMATIVELY gone empty: a RECENT `Empty` report (within grace of the LAST report — so a
/// continuously-empty realm re-asserting `Empty` every tick STAYS confirmed, while grace tolerates a single
/// dropped datagram) AND no non-Empty demand at-or-after the empty STREAK start (arm C — order-independent:
/// the `last_demand_tick < first_empty_tick` term resolves a same-tick / reordered `Empty`+`SpinUp`
/// regardless of inbox order). A crashed shard STOPS reporting ⇒ `last_empty_tick` ages past grace ⇒ NOT
/// confirmed-empty (it takes the `zombie` path instead — the dead-vs-empty disambiguation). NOTE: freshness
/// is measured from `last_empty_tick`, NOT the streak start — an empty realm that outlives `grace` must
/// remain reap-eligible, which measuring from the streak start would wrongly prevent. Bitwise `&` (HR5).
#[must_use]
pub fn empty_confirmed(cell: &LedgerCell, now: UniverseTick, grace: u64) -> bool {
    match (cell.last_empty_tick, cell.first_empty_tick) {
        (Some(last), Some(streak)) => {
            let fresh = now.0.saturating_sub(last.0) <= grace;
            let no_demand_since = cell.last_demand_tick < streak;
            fresh & no_demand_since
        }
        _ => false,
    }
}

/// THE desired-alive predicate (§1.5). Arm A = demand-driven (AoI OR player-spawn — source-agnostic). Arm
/// B = LIVE and not affirmatively-empty (the NO-STRAND core: a live realm only LEAVES desired by saying
/// `Empty`; demand-absence alone never un-desires it — a partitioned parent stops `KeepAlive`-ing but arm
/// B holds). Bitwise (no short-circuit false-arm, HR5). `running_live` is resolved by the caller.
#[must_use]
pub fn desired_alive(
    cell: &LedgerCell,
    now: UniverseTick,
    tuning: &RlmTuning,
    running_live: bool,
) -> bool {
    demanded_recently(cell, now, tuning.demand_ttl_ticks)
        | (running_live & !empty_confirmed(cell, now, tuning.empty_grace_ticks))
}

/// THE teardown-ready predicate (§1.6) — necessary-but-not-sufficient (the kill fires only after the drain,
/// slice 3b). Every clause is a resolved boolean or a monomorphic arithmetic compare; bitwise `&` (HR5).
/// A realm is reap-eligible ONLY when it is live, out of the whole AoI closure, affirmatively empty, past
/// its boot+settle floor and its teardown cooldown, orphans no live child, is not mid-transfer, and the
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
        & !facts.in_transfer
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
) -> (Vec<LifecycleAction>, LedgerDelta) {
    // 1. The demand-driven / arm-B desired set (§1.5), then its ancestor-closure (§3.2).
    let mut desired: BTreeMap<RealmPath, RealmCoord> = BTreeMap::new();
    for (path, cell) in ledger.iter() {
        let head = dir.head(DirectoryKey::Realm(cell.coord.lowered()));
        if desired_alive(cell, now, tuning, running_live(head, liveness_dead)) {
            desired.insert(path.clone(), cell.coord.clone());
        }
    }
    let closed = ancestor_close(&desired);

    let mut delta = LedgerDelta::default();
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
                    in_transfer: r.in_transfer.is_some(),
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
            in_transfer: false,
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
            l.get(sys(7).path()).unwrap().last_demand_tick,
            UniverseTick(10)
        );
        l.record_demand(&sys(7), DemandVerb::KeepAlive, UniverseTick(14), Fence(1));
        assert_eq!(
            l.get(sys(7).path()).unwrap().last_demand_tick,
            UniverseTick(14)
        );
        assert_eq!(l.len(), 1);
        assert!(!l.is_empty());
    }

    #[test]
    fn record_empty_starts_a_streak_that_a_reassert_does_not_advance() {
        let mut l = DemandLedger::default();
        l.record_demand(&sys(7), DemandVerb::Empty, UniverseTick(20), Fence(2));
        let c = l.get(sys(7).path()).unwrap();
        assert_eq!(c.first_empty_tick, Some(UniverseTick(20)));
        assert_eq!(c.last_empty_tick, Some(UniverseTick(20)));
        l.record_demand(&sys(7), DemandVerb::Empty, UniverseTick(25), Fence(2));
        let c = l.get(sys(7).path()).unwrap();
        assert_eq!(
            c.first_empty_tick,
            Some(UniverseTick(20)),
            "streak anchor unchanged"
        );
        assert_eq!(
            c.last_empty_tick,
            Some(UniverseTick(25)),
            "last report advances"
        );
    }

    #[test]
    fn a_demand_at_or_after_the_streak_clears_it_but_a_stale_one_does_not() {
        let mut l = DemandLedger::default();
        l.record_demand(&sys(7), DemandVerb::Empty, UniverseTick(20), Fence(1));
        // A stale (reordered) SpinUp BEFORE the streak start does NOT clear it.
        l.record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(15), Fence(1));
        assert_eq!(
            l.get(sys(7).path()).unwrap().first_empty_tick,
            Some(UniverseTick(20)),
            "a stale demand cannot clear a fresher empty streak"
        );
        // A SpinUp at-or-after the streak start clears it.
        l.record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(22), Fence(1));
        assert_eq!(l.get(sys(7).path()).unwrap().first_empty_tick, None);
    }

    #[test]
    fn teardown_verb_is_a_noop_the_reconciler_is_the_sole_kill_authority() {
        let mut l = DemandLedger::default();
        l.record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(10), Fence(1));
        l.record_demand(&sys(7), DemandVerb::TearDown, UniverseTick(30), Fence(1));
        // TearDown neither refreshed the demand nor latched empty.
        let c = l.get(sys(7).path()).unwrap();
        assert_eq!(c.last_demand_tick, UniverseTick(10));
        assert_eq!(c.first_empty_tick, None);
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
            l.get(sys(7).path()).unwrap().last_fence
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
            l.get(sys(7).path()).unwrap().spawn_watermark,
            UniverseTick(40)
        );
        l.mark_reaped(sys(7).path(), UniverseTick(50));
        assert_eq!(
            l.get(sys(7).path()).unwrap().teardown_watermark,
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
            l.get(sys(7).path()).unwrap().draining_since,
            Some(UniverseTick(12))
        );
        assert!(l.get(sys(8).path()).is_none());
        // Clearing (rescue).
        let mut clear = LedgerDelta::default();
        clear.set_draining.insert(sys(7).path().clone(), None);
        l.apply_delta(&clear);
        assert_eq!(l.get(sys(7).path()).unwrap().draining_since, None);
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
        let c = l.get(sys(7).path()).unwrap();
        let t = cloud();
        // Within TTL ⇒ desired even when NOT running (arm A).
        assert!(desired_alive(c, UniverseTick(100), &t, false));
        assert!(desired_alive(
            c,
            UniverseTick(100 + t.demand_ttl_ticks),
            &t,
            false
        ));
        // Past TTL and not running ⇒ not desired.
        assert!(!desired_alive(
            c,
            UniverseTick(101 + t.demand_ttl_ticks),
            &t,
            false
        ));
    }

    #[test]
    fn desired_alive_arm_b_live_and_not_empty_no_strand() {
        let mut l = DemandLedger::default();
        // A realm demanded long ago (arm A now stale) that is LIVE and never said Empty.
        l.record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(1), Fence(1));
        let c = l.get(sys(7).path()).unwrap();
        let t = cloud();
        let now = UniverseTick(10_000); // arm A long expired
        assert!(!demanded_recently(c, now, t.demand_ttl_ticks));
        // Arm B keeps it alive because it is running_live and never confirmed empty (NO-STRAND).
        assert!(desired_alive(c, now, &t, true));
        // If it is NOT running, arm B is off ⇒ not desired.
        assert!(!desired_alive(c, now, &t, false));
    }

    #[test]
    fn demanded_recently_ignores_the_never_demanded_sentinel() {
        let t = cloud();
        let mut l = DemandLedger::default();
        // A realm that has ONLY reported Empty keeps the `0` sentinel ⇒ NOT demanded, even at an early
        // tick where `now <= ttl` would spuriously pass a raw `now - 0 <= ttl`.
        l.record_demand(&sys(7), DemandVerb::Empty, UniverseTick(3), Fence(1));
        let c = l.get(sys(7).path()).unwrap();
        assert_eq!(c.last_demand_tick, UniverseTick(0));
        assert!(!demanded_recently(c, UniverseTick(5), t.demand_ttl_ticks));
        // A real SpinUp ⇒ demanded within the TTL.
        l.record_demand(&sys(7), DemandVerb::SpinUp, UniverseTick(6), Fence(1));
        assert!(demanded_recently(
            l.get(sys(7).path()).unwrap(),
            UniverseTick(7),
            t.demand_ttl_ticks
        ));
    }

    #[test]
    fn empty_confirmed_requires_fresh_report_and_no_demand_since_streak() {
        let t = cloud();
        let mut l = DemandLedger::default();
        l.record_demand(&sys(7), DemandVerb::Empty, UniverseTick(100), Fence(1));
        let c = l.get(sys(7).path()).unwrap().clone();
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
                    l.get(sys(7).path()).unwrap(),
                    UniverseTick(tick),
                    t.empty_grace_ticks
                ),
                "a continuously-empty realm stays confirmed at tick {tick}"
            );
        }
        // It STOPS reporting (crash) ⇒ ages out of confirmed-empty past grace.
        let last = 100 + 5 * t.empty_grace_ticks - 1;
        assert!(!empty_confirmed(
            l.get(sys(7).path()).unwrap(),
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
            l.get(sys(7).path()).unwrap(),
            UniverseTick(100),
            t.empty_grace_ticks
        ));
        // Empty streak, then a SpinUp AT-OR-AFTER it processed FIRST (reorder): last_demand >= streak ⇒
        // NOT confirmed regardless of order (arm C).
        let mut r = DemandLedger::default();
        r.record_demand(&sys(8), DemandVerb::SpinUp, UniverseTick(30), Fence(1));
        r.record_demand(&sys(8), DemandVerb::Empty, UniverseTick(30), Fence(1));
        assert!(!empty_confirmed(
            r.get(sys(8).path()).unwrap(),
            UniverseTick(30),
            t.empty_grace_ticks
        ));
    }

    #[test]
    fn teardown_ready_requires_every_clause() {
        let t = cloud();
        let mut l = DemandLedger::default();
        // A realm that is live, FRESHLY empty (confirmed), past dwell + cooldown, orphans nothing, not in
        // transfer. `now` is well past min_dwell (spawn_watermark 0); the Empty streak is within grace.
        let now = UniverseTick(200);
        l.record_demand(&sys(7), DemandVerb::Empty, UniverseTick(195), Fence(1));
        let base = l.get(sys(7).path()).unwrap().clone();
        assert!(teardown_ready(&base, now, &t, &facts(true)));

        // Each single clause flip makes it NOT ready:
        assert!(!teardown_ready(&base, now, &t, &facts(false))); // not running_live
        let mut in_closure = facts(true);
        in_closure.desired_in_closure = true;
        assert!(!teardown_ready(&base, now, &t, &in_closure)); // still demanded/desired
        let mut has_desc = facts(true);
        has_desc.has_desired_descendant = true;
        assert!(!teardown_ready(&base, now, &t, &has_desc)); // a live child
        let mut transferring = facts(true);
        transferring.in_transfer = true;
        assert!(!teardown_ready(&base, now, &t, &transferring)); // mid re-home
        let mut not_quiesced = facts(true);
        not_quiesced.quiesced = false;
        assert!(!teardown_ready(&base, now, &t, &not_quiesced)); // crash quiesce

        // Not empty-confirmed ⇒ not ready.
        let mut never_empty = DemandLedger::default();
        never_empty.record_demand(&sys(9), DemandVerb::SpinUp, UniverseTick(1), Fence(1));
        assert!(!teardown_ready(
            never_empty.get(sys(9).path()).unwrap(),
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
        );
        assert!(b.is_empty(), "a rescuing demand aborts the kill (BUG-C)");
        assert_eq!(
            db.set_draining.get(sys(7).path()),
            Some(&None),
            "drain cleared"
        );
        l.apply_delta(&db);
        assert_eq!(l.get(sys(7).path()).unwrap().draining_since, None);
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
        );
        // The System is NOT reaped (has a desired descendant); the Planet + its intermediate ancestors
        // that are not yet running get spun up. Crucially: no Kill for the System.
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, LifecycleAction::Kill { .. })),
            "a live child keeps its ancestor un-reaped"
        );
        assert!(delta.set_draining.get(u_g_s().path()).is_none());
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
}
