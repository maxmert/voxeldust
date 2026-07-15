I now have complete verification. This confirms the HIGH finding's reframing: the module doc itself states that `abort_deadline_ticks` bounds the "worst-case healthy pre-freeze round-trip" — i.e. the abort budget's real purpose is to give a slow-but-alive saga time, NOT to bound transport confirm latency. The sequential ordering is explicit.

Producing the final vetted design.

---

# R-4c FINAL DESIGN — LivenessTuning geometric cross-config invariant + transport env-plumb

## Verdict: SOUND TO IMPLEMENT — with three mandatory corrections folded in

The two SOUND_TO_IMPLEMENT lenses (crate-dependency/HR5/Duration-overflow, and integration/DRY-env-plumb) stand. The formula-correctness lens is correctly FIX_BEFORE_IMPL: **check (b) sums the wrong series segment (CRITICAL)**, **check (a)'s premise is unsound (HIGH)**, and **the runtime file:line references are wrong (MEDIUM)**. All three are real and I verified each against the code. I fold them in below. The three integration MEDIUMs (extra call site, stale doc, Option-A reframing) are also real and folded in.

---

## 0. Verified load-bearing facts (re-pointed to the REAL tree)

The runtime that owns window/consecutive/abort lives in **`crates/node/src/saga_runtime.rs`** (NOT vd-sim — vd-sim has no `saga_runtime.rs`; the design's `saga_runtime.rs:*` cites were against a phantom path). The tuning structs live in **`crates/sim/src/saga.rs`**. Corrected anchors:

- **Backoff growth** (`peer_writer`, `mesh.rs:1033-1157`): `backoff` starts `= backoff_min` (`mesh.rs:1038`), reset to `backoff_min` on any success (`:1091`/`:1143`). The initial `WriteFail::Down` (`:1097`) arms the timer at `now + backoff` with **un-doubled `backoff_min`** (`:1104`), sets `counting=true`, does **NOT bump the counter** (drop fan-out is `handle_connection_drop`→`on_write_error`, `:1244-1246`, the C2 cure), THEN doubles (`:1106`). Each later timer fire that still fails re-arms at the current `backoff` then doubles (`:1147-1149`), capped at `backoff_max`.
- **Counter + bounce** (`on_replay_failed` `mesh.rs:487-489` bumps `consecutive_failures`; `confirm_and_maybe_bounce` `:1263` bounces when `consecutive_failures >= confirm_unreachable_after_retries`). Re-bounces on EVERY subsequent fire (counter stays `>= N`, `:1250-1252` "re-bounces past the threshold").
- **The fire timeline (decisive):** interval before fire `i` is `min(backoff_min·2^(i-1), backoff_max)`. The counter reaches `k` on fire `k`, so **the first `NodeUnreachable` fires on timer-fire N** (`N = confirm_unreachable_after_retries`), and subsequent notices fire on **fires N, N+1, N+2, …** — one per fire.
- **Saga observation** (`LivenessTracker`, `saga_runtime.rs:256-289`): `record_unreachable` (`:256`) EXTENDS the run or RESETS to `consecutive=1, first=now` when `now − first > unreachable_window_ticks` (`:261-269`); `is_confirmed_dead` (`:281`) is `(consecutive >= n) & (now − first <= window)` (`:285-288`).
- **Abort anchor is SEQUENTIAL, downstream of confirm** (`rehome_event_for`, `saga_runtime.rs:1241-1317`): `dead_observed_since` is `get_or_insert(now)` ONLY inside the `is_confirmed_dead(...)` == true branch (`:1260`, `:1299`); the destructive resolution fires at `now − observed >= abort_deadline_ticks` (`:1261`/`:1300`). The module doc (`:33-35`) confirms `abort_deadline_ticks` bounds "worst-case healthy pre-freeze round-trip", not transport confirm latency.
- **Tick rate is env-driven** `VD_TICK_HZ` (orchestrator reads it at `orchestrator.rs:275`; `TickPacer::new` uses `1s / tick_hz.max(1)`, `runtime.rs:181`). NOT in `core/src/ids.rs` (a dead pointer). 50Hz dev cluster (`bins/src/lib.rs:81`), 20Hz tests. **The conversion factor MUST be an argument.**
- **`MeshConfig::new` hardcodes all three reliability knobs** (`mesh.rs:169-188`); five call sites: `orchestrator.rs:51`, `gateway.rs:24`, `shard.rs:25`, `client.rs:88`, **`process_parity.rs:217`** (the 5th, missing from the original §3/§7). `DEFAULT_CONFIRM_UNREACHABLE_AFTER_RETRIES = 3` (`mesh.rs:87`).
- **`vd-sim` cannot see `MeshReliabilityTuning`** (it's in io-prod; the dependency rule forbids sim→io-prod). `runtime.rs` does NOT currently import `mesh` (both are in io-prod, so adding the import is legal — no cycle).
- **Stale doc** at `mesh.rs:92-96` still says `confirm_unreachable_after_retries` is "NOT YET CONSUMED" — false since R-4a; `mesh.rs:1263` consumes it. Fix within R-4c.

---

## 1. The geometric-backoff-sum-in-ticks formula (pure, HR5-faithful)

Two segments of the SAME geometric series are needed — a general helper that sums a **half-open index window** `[lo, hi)` of the series `interval(i) = min(backoff_min·2^(i-1), backoff_max)` (1-based `i`). This is what lets check (b) sum the correct LATE segment. Lives in `crates/sim/src/saga.rs`, raw args (no io-prod edge).

```rust
use std::time::Duration; // verify not already imported at top of saga.rs before adding

/// R-4c: sum interval indices `[lo, hi)` (1-based, half-open) of the peer_writer redial series
/// `interval(i) = min(backoff_min * 2^(i-1), backoff_max)` — traced against mesh.rs:1104/1147 (the initial
/// Down arms at the UN-doubled backoff_min = interval(1); each later fire doubles, capped at backoff_max).
/// Monomorphic, straight-line (HR5): the cap is `Duration::min` (no `if`); the growth is `saturating_mul`
/// (a >=~31-bit doubling saturates and is pinned by the same `min`, never a shift/overflow UB). `lo >= hi`
/// ⇒ ZERO (empty window). Advancing `step` for the skipped `[1, lo)` prefix is also cap-pinned, so a large
/// `lo` costs at most `lo` saturating doublings, never UB.
fn backoff_series_sum(lo: u32, hi: u32, backoff_min: Duration, backoff_max: Duration) -> Duration {
    let mut step = backoff_min;
    // advance step to interval(lo) without summing the prefix
    for _ in 1..lo {
        step = step.saturating_mul(2).min(backoff_max);
    }
    let mut sum = Duration::ZERO;
    for _ in lo..hi {
        sum = sum.saturating_add(step.min(backoff_max)); // step already <= max, min is defensive
        step = step.saturating_mul(2).min(backoff_max);
    }
    sum
}

/// R-4c: ceiling-divide a `Duration` into whole universe ticks at `tick_hz`. CEILING — a partial tick still
/// costs a full tick of budget (never truncate the real latency below what the saga must survive).
/// `tick_hz == 0` clamps to 1 (matches TickPacer::new's `tick_hz.max(1)`, runtime.rs:181 — one physical
/// convention, no div-by-zero). u128 headroom: 5s * 1e9 ns * 50 Hz fits with room to spare.
fn duration_to_ticks_ceil(d: Duration, tick_hz: u32) -> u64 {
    let hz = tick_hz.max(1) as u128;
    let ticks = d.as_nanos().saturating_mul(hz).div_ceil(1_000_000_000u128);
    u64::try_from(ticks).unwrap_or(u64::MAX)
}

/// R-4c: the transport WORST-CASE confirmed-dead latency in whole ticks — the FIRST bounce fires on
/// timer-fire N (`N = confirm_retries`), preceded by N intervals (indices 1..=N). = ceil(sum of the first N
/// intervals). Used by check (a)'s *diagnostic* number only (see §2 — check (a)'s ORDERING is re-derived).
#[must_use]
pub fn transport_confirm_latency_ticks(
    confirm_retries: u32,
    backoff_min: Duration,
    backoff_max: Duration,
    tick_hz: u32,
) -> u64 {
    duration_to_ticks_ceil(
        backoff_series_sum(1, confirm_retries.saturating_add(1), backoff_min, backoff_max),
        tick_hz,
    )
}

/// R-4c CRITICAL-FIX: the wall-clock spread, in ticks, between the FIRST and LAST `NodeUnreachable` of a
/// confirmation run the orchestrator actually observes. The transport fires the run's notices on timer-fires
/// N, N+1, …, N+n-1 (`N = confirm_retries`, `n = n_consecutive`), so the run spans the `(n-1)` inter-notice
/// gaps at intervals `N+1 .. N+n-1` — the LATE (near/at backoff_max) segment, NOT the early min/2min/4min.
/// = ceil(sum of series indices [N+1, N+n]). `n <= 1` ⇒ 0 (a single notice has no spread). This is the
/// segment `record_unreachable` (saga_runtime.rs:256) must NOT reset before, so the window must cover it.
#[must_use]
pub fn transport_run_spread_ticks(
    confirm_retries: u32,
    n_consecutive: u32,
    backoff_min: Duration,
    backoff_max: Duration,
    tick_hz: u32,
) -> u64 {
    // gaps between notices at fires N..N+n-1 = intervals with index (N+1)..=(N+n-1) = series[N+1, N+n).
    let lo = confirm_retries.saturating_add(1);
    let hi = confirm_retries.saturating_add(n_consecutive); // (N+1) + (n-1) = N+n, exclusive
    duration_to_ticks_ceil(backoff_series_sum(lo, hi, backoff_min, backoff_max), tick_hz)
}
```

**Off-by-one (verified, do NOT alter):** N intervals precede the N-th fire; interval 1 = un-doubled `backoff_min`. First notice on fire N; last on fire N+n-1. The spread is the `(n-1)` gaps at indices `N+1 .. N+n-1` — i.e. series window `[N+1, N+n)`. `div_ceil` rounds latency UP (conservative); abort is already integer ticks (no round owed).

**Worked numbers (min=50ms, max=5s, 20Hz):**
- `confirm_latency(N=3)` = 50+100+200 = 350ms → **7 ticks** (diagnostic only).
- `run_spread(N=3, n=3)` = gaps at indices 4,5 = 400+800 = 1200ms → **24 ticks** (the CRITICAL fix: the design's old 7 was a 3.4x under-estimate).
- `run_spread(N=8, n=3)` = indices 9,10 = 5000+5000 = 10000ms → **200 ticks** (cap reached — collapses to `(n-1)·backoff_max`).
- At 50Hz dev, `run_spread(N=3, n=3)` = 1200ms → **60 ticks** (vs default window 64 — a real ~4-tick margin the design's old check hid).

---

## 2. `validate_against` — signature, crate, the corrected checks

**Home: `LivenessTuning` method in `crates/sim/src/saga.rs`, raw transport args** (no sim→io-prod edge; the orchestrator bin alone holds both tunings and supplies the raw numbers). Tier-A unit-coverable beside the tuning.

```rust
/// R-4c cross-config error: the saga clock (ticks) and the transport clock (wall-clock backoff), validated
/// in isolation elsewhere, are mutually MIS-ORDERED. Fail-loud at orchestrator boot (the ONE process holding
/// both). Equality-comparable (HR5(d)): every field on the error so tests assert the whole struct.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LivenessCrossTuningError {
    #[error(
        "unreachable_window_ticks ({window}) must be >= the transport re-bounce SPREAD of a confirmation \
         run ({run_spread} ticks): the orchestrator observes the {n}-notice run on redial fires \
         confirm_retries+1..+{n} at {tick_hz} Hz — its inter-notice gaps are the LATE (near backoff_max) \
         backoff intervals, so a narrower window lets record_unreachable reset the run before the n-th \
         notice and is_confirmed_dead NEVER fires (a genuinely-dead peer is orphaned — the liveness hole)"
    )]
    WindowNarrowerThanRunSpread { window: u64, run_spread: u64, n: u32, tick_hz: u32 },

    #[error(
        "abort_deadline_ticks ({abort}) must be >= the worst-case time a genuinely-dead peer takes to \
         reach a confirmed-dead RUN ({recover_budget} ticks = confirm_retries backoff + the {n}-notice \
         run spread at {tick_hz} Hz): a tighter abort budget fires the DESTRUCTIVE re-home before a \
         recoverable blip could ever have cleared via record_ack — a false abandon of a slow-but-live saga"
    )]
    AbortTighterThanRecoverBudget { abort: u64, recover_budget: u64, n: u32, tick_hz: u32 },
}

impl LivenessTuning {
    /// R-4c: cross-validate this saga-side liveness/abort budget against the RAW transport redial backoff
    /// (io-prod MeshConfig), passed as primitives so vd-sim needs no io-prod dependency. Called at
    /// orchestrator boot AFTER both tunings are read — the ONE cross-config gate. Two ORDERED invariants,
    /// each a separate `if` (never `&&` — HR5), each a typed equality-comparable error:
    ///
    /// (a) `unreachable_window_ticks` must span the transport re-bounce SPREAD of the n-consecutive run
    ///     (the LATE-segment geometric gaps the orchestrator actually observes) — else record_unreachable
    ///     resets the run and the peer is never confirmed dead (the CRITICAL fix: the spread is the late
    ///     segment [N+1, N+n), NOT the early [1, n) the first design summed).
    /// (b) `abort_deadline_ticks` must be >= the worst-case DEAD-peer recover budget: the full confirm
    ///     latency PLUS the run spread — the longest a genuinely-slow peer can be transiently unreachable
    ///     yet still clear via record_ack before is_confirmed_dead. The abort clock (dead_observed_since,
    ///     saga_runtime.rs:1260/1299) starts DOWNSTREAM of confirmation, so this bounds a FALSE-ABANDON of a
    ///     recoverable saga — NOT the (unsound) "peer dead before transport confirms" story of the first
    ///     design (the two clocks are SEQUENTIAL, not concurrent — verified at saga_runtime.rs:33-35).
    ///
    /// # Errors
    /// [`LivenessCrossTuningError`] for either mis-ordering.
    pub fn validate_against(
        &self,
        saga: &SagaTuning,
        confirm_retries: u32,
        backoff_min: Duration,
        backoff_max: Duration,
        tick_hz: u32,
    ) -> Result<(), LivenessCrossTuningError> {
        // (a) window >= the LATE-segment run spread (the CRITICAL correction).
        let run_spread = transport_run_spread_ticks(
            confirm_retries,
            self.n_consecutive_unreachable,
            backoff_min,
            backoff_max,
            tick_hz,
        );
        if self.unreachable_window_ticks < run_spread {
            return Err(LivenessCrossTuningError::WindowNarrowerThanRunSpread {
                window: self.unreachable_window_ticks,
                run_spread,
                n: self.n_consecutive_unreachable,
                tick_hz,
            });
        }
        // (b) abort budget >= confirm latency + run spread (the recoverable-blip clearance budget).
        let confirm_latency =
            transport_confirm_latency_ticks(confirm_retries, backoff_min, backoff_max, tick_hz);
        let recover_budget = confirm_latency.saturating_add(run_spread);
        if saga.abort_deadline_ticks < recover_budget {
            return Err(LivenessCrossTuningError::AbortTighterThanRecoverBudget {
                abort: saga.abort_deadline_ticks,
                recover_budget,
                n: self.n_consecutive_unreachable,
                tick_hz,
            });
        }
        Ok(())
    }
}
```

**What changed from the first design and why (all three FIX_BEFORE_IMPL findings folded):**

1. **CRITICAL (check b→now (a)):** `run_spread` is `transport_run_spread_ticks(N, n, …)` summing series window `[N+1, N+n)` — the LATE gaps the orchestrator observes — NOT `transport_confirm_latency_ticks(n, …)` summing `[1, n)`. The formula now takes BOTH `confirm_retries` (the offset to the first bounce) and `n_consecutive_unreachable` (the run length). Verified against `confirm_and_maybe_bounce` (`mesh.rs:1263`, first bounce on fire N) + `record_unreachable` reset (`saga_runtime.rs:261`).

2. **HIGH (check a→now (b)):** re-derived. The old "confirm latency ≤ abort, else the peer is modeled dead before transport confirms" is **unsound** — `dead_observed_since` is `get_or_insert(now)` only after `is_confirmed_dead` is already true (`saga_runtime.rs:1260/1299`), so the two clocks are sequential; a tight abort never strands. The invariant worth asserting (per `saga_runtime.rs:33-35`) is that the DESTRUCTIVE abort budget exceeds the worst-case **recover budget** of a slow-but-live peer: `confirm_latency + run_spread`. This bounds a false-abandon, the real failure. The old error message's causal story ("strands in a Timeout re-drive that never reaches re-home") is deleted.

3. **MEDIUM (stale file:line):** every runtime cite re-pointed to `crates/node/src/saga_runtime.rs` (`record_unreachable :256`, `is_confirmed_dead :281`, `dead_observed_since :1260/:1299`, `rehome_event_for :1241`). The `core/src/ids.rs` tick-rate assumption is dropped (env `VD_TICK_HZ`). The doc-comment shift-vs-`saturating_mul` mismatch is fixed (the code always said `saturating_mul`; the doc now matches).

### Supersede-or-keep the linear `validate()` check — SHIP OPTION A (additive)

**Decision: KEEP `LivenessTuning::validate()` unchanged (`saga.rs:187-203`), ADD `validate_against` as a stricter io-prod-specific gate at the orchestrator only.** No harness regression. But reframe the rationale per the integration MEDIUM: `retry_delay_ticks_hint` is defined (`saga.rs:141-144`) as BOTH the harness `FaultFabric` retry delay AND io-prod's QUIC idle cadence — so `validate()`'s linear check is a **coarse always-run floor** approximating the same quantity `validate_against` computes exactly from the geometric backoff. It is NOT "a different transport." The correct framing: `validate()` = the always-run single-struct invariants (`ZeroConsecutive`) + a coarse linear window floor for rigs that never call `validate_against`; `validate_against` = the exact io-prod-geometric ceiling/floor layered on top at boot. Strictly additive, matches the existing `validate` + cross-check layering (mirrors `SagaTuning::validate` + `MeshReliabilityTuning::validate`).

---

## 3. Env-plumb the transport clock (DRY) — unchanged in shape, 5 call sites

**New env vars** (mirror `VD_LIVENESS_*`/`VD_SAGA_*`):
- `VD_MESH_CONFIRM_RETRIES` → `reliability.confirm_unreachable_after_retries` (default `DEFAULT_CONFIRM_UNREACHABLE_AFTER_RETRIES` = 3, `mesh.rs:87`)
- `VD_MESH_BACKOFF_MIN_MS` → `redial_backoff_min` (default 50)
- `VD_MESH_BACKOFF_MAX_MS` → `redial_backoff_max` (default 5000)

**DRY read home: `EnvConfig::mesh_reliability_overrides()` in `crates/io-prod/src/runtime.rs`** (Tier-B, the env-parse home, covered by `runtime.rs` tests rather than the coverage-excluded bins; present-but-bad value fails LOUD via `parse_or`). Returns a `MeshReliabilityOverrides`.

**Type home: `MeshReliabilityOverrides` in `crates/io-prod/src/mesh.rs`** (beside `MeshConfig`, which consumes it). `runtime.rs` must add `use crate::mesh::MeshReliabilityOverrides;` (verified: `runtime.rs` does not currently import `mesh`; both in io-prod, no cycle — legal).

**`MeshConfig::with_reliability_overrides` builder** (`mesh.rs`, after `MeshConfig::new` `:188`) so bins don't touch fields inline.

**Call sites (5, `.with_reliability_overrides(&env.mesh_reliability_overrides()?)` chained on `MeshConfig::new`):** `orchestrator.rs:51`, `gateway.rs:24`, `shard.rs:25`, `client.rs:88`, and **`process_parity.rs:217`** — documented-exempt from cross-validate (the parity clients hold no saga tuning) but MUST get the override chain (or, since it uses positional `MeshConfig::new(id, bind, book, 64, 0)` with no `EnvConfig`, explicitly document it inherits `new`'s defaults; **tripwire:** if any `VD_MESH_*` is ever set in DEV/`common_env`, this parity site silently diverges from the spawned bins — call it out in the test's comment). Recommendation: give `process_parity.rs:217` the same override read if it has an `EnvConfig` in scope; else leave defaulted with the tripwire comment.

The dev-cluster env home (`bins/src/lib.rs::common_env`) stays defaulted (50Hz/N=3 → safe); the vars exist for prod tuning. **Do NOT set `VD_MESH_*` in `common_env`** without also plumbing `process_parity.rs:217`, per the tripwire.

---

## 4. Orchestrator boot-assert wiring

In `crates/bins/src/bin/orchestrator.rs`: pull the mesh overrides into a **named local before `spawn_mesh`** (`:48-59`) so the cross-validated numbers are LITERALLY the ones the mesh runs with (no drift), chain `.with_reliability_overrides(&overrides)` in the `MeshConfig::new` call, and add the gate **after `liveness.validate()?` (`:116`)**:

```rust
// after saga.validate()? (:82), directory.validate()? (:100), liveness.validate()? (:116):
let overrides = env.mesh_reliability_overrides()?;         // the SAME struct fed to MeshConfig below
let tick_hz: u32 = env.parse("VD_TICK_HZ")?;               // the SAME rate the pacer uses (:275)
// R-4c: the orchestrator is the ONE process holding BOTH the transport reliability tuning and the saga
// liveness/abort budget — cross-validate them LOUD (nothing else asserts the two clocks are ordered).
liveness.validate_against(
    &saga,
    overrides.confirm_retries,
    overrides.backoff_min,
    overrides.backoff_max,
    tick_hz,
)?;
```

`?` propagates `LivenessCrossTuningError` as boxed `dyn Error` (it's `thiserror::Error`, same as the existing `.validate()?` sites). **Only the orchestrator boot-asserts** (it alone holds `SagaTuning`/`LivenessTuning`); the other four bins only get the §3 env-plumb.

Also within R-4c: **update the stale `mesh.rs:92-96` CONSUMER NOTE** to state `confirm_unreachable_after_retries` IS consumed by `confirm_and_maybe_bounce` (Nth-timer-fire bounce, `mesh.rs:1263`).

---

## 5. Tier-A test plan (`saga.rs` `#[cfg(test)]`, vd-sim, no tokio; HR5(d) equality/`expect_err`, split `&&`, no `matches!`)

**Pure series helper (`backoff_series_sum` / `duration_to_ticks_ceil`):**
1. `series_first_n_default` — `backoff_series_sum(1, 4, 50ms, 5s)` = `from_millis(350)`; `(1,9,…)` = `from_millis(11350)`.
2. `series_empty_window` — `lo >= hi` ⇒ `Duration::ZERO` (e.g. `(3,3,…)`, `(5,2,…)`).
3. `series_caps_at_max` — `(1, 11, 50ms, 5s)` pins indices 8+ at 5s; assert the exact sum.
4. `series_no_overflow` — `(1, 41, 50ms, 5s)` and `(35, 40, …)` (large `lo` prefix advance) do not panic; assert finite exact values (proves `saturating_mul` + cap, no shift UB).
5. `ticks_ceil_rounds_up` — `duration_to_ticks_ceil(from_millis(351), 20)` = `ceil(7.02)=8`; `from_millis(350)`→7; `ZERO`→0; `tick_hz=0`→ equals the 1Hz value (no div-by-zero).

**Segment-selection (the CRITICAL fix — the tests that would have caught the old bug):**
6. `run_spread_late_segment` — `transport_run_spread_ticks(3, 3, 50ms, 5s, 20)` = **24** (NOT 7); `(3,3,50ms,5s,50)` = **60**; `(8,3,…,20)` = **200** (cap). Assert exact — this is the regression guard against re-summing `[1,n)`.
7. `run_spread_single_notice` — `n=1` ⇒ 0 (no spread). `n=0` ⇒ 0 (defensive).
8. `confirm_latency_first_n` — `transport_confirm_latency_ticks(3, 50ms, 5s, 20)` = 7; `(…,50)` = 18; `(8,…,20)` = 227.

**`validate_against` arm tests:**
9. `cross_ok_at_prod_defaults` — `LivenessTuning{n=3, window=64, hint=2}` + `SagaTuning{abort=24, redrive=…}` @ 20Hz `(3,50ms,5s)`: window 64 ≥ run_spread 24 ✓; abort 24 ≥ recover_budget (7+24=31)? **NO — 24 < 31 ⇒ Err(b)!** → this test PROVES the prod default at 20Hz is a real mis-order the gate CATCHES (the razor margin the reviewer flagged). Adjust: assert `Err(AbortTighterThanRecoverBudget{ abort:24, recover_budget:31, n:3, tick_hz:20 })` — whole-struct equality. (This is a genuine finding: **the default `SagaTuning::abort_deadline_ticks=24` is too tight for `n=3` prod liveness at these backoffs** — the gate surfaces it at boot, which is exactly its job. Flag to the user in §6.)
10. `cross_ok_when_abort_widened` — same but `SagaTuning{abort=31}` → `Ok(())` (boundary: `abort == recover_budget`).
11. `cross_rejects_narrow_window` — `window=20` (< run_spread 24), abort wide → `Err(WindowNarrowerThanRunSpread{ window:20, run_spread:24, n:3, tick_hz:20 })`.
12. `cross_boundary_window_equals_spread` — `window == run_spread` → check (a) passes; `window == run_spread-1` → `Err(a)` (the `<` boundary, off-by-one guard).
13. `cross_boundary_abort_equals_budget` — `abort == recover_budget` → passes (b); `abort == recover_budget-1` → `Err(b)`.
14. `cross_check_a_precedes_b` — a config failing BOTH returns the `(a)` window error (deterministic arm ordering).
15. `cross_default_liveness_n1` — `LivenessTuning::default()` (n=1) + `SagaTuning::default()`: run_spread(N=3,n=1)=0, recover_budget=confirm_latency(7); window 64≥0 ✓, abort 24≥7 ✓ → `Ok(())` (the DEV/kill-only default is fine; only prod `n≥3` trips the razor margin).

**Regression + Display:**
16. `validate_still_green` — existing `LivenessTuning::validate` tests (`saga.rs:4340-4406` region) stay green unchanged (proves Option A is additive).
17. `cross_errors_display` — each `LivenessCrossTuningError` arm `.to_string()` contains its key fields (mirrors `runtime.rs`'s `errors_display`; the `thiserror` format strings are HR5 regions).

**io-prod side:**
18. `mesh.rs`: `MeshConfig::new(...).with_reliability_overrides(&o)` sets all three fields (equality on each). Near `mesh.rs:1895+`.
19. `runtime.rs`: `mesh_reliability_overrides()` defaults on absence + errors on a bad `VD_MESH_CONFIRM_RETRIES` (mirrors `parse_or_defaults_on_absence_but_errors_on_a_bad_value`). Near `runtime.rs:248`.

---

## 6. Decisions + flags for the user

1. **SHIP Option A** (keep `validate()` linear check, add `validate_against`) — additive, reframed as coarse-floor-vs-exact-gate (same quantity, not different transports).
2. **`validate_against` = `LivenessTuning` method in vd-sim, raw args** — no sim→io-prod edge, Tier-A coverable.
3. **`MeshReliabilityOverrides` in `mesh.rs`**; `runtime.rs` adds `use crate::mesh::MeshReliabilityOverrides` (verified legal, no cycle).
4. **Pass `VD_TICK_HZ`** (never hardcode 20/50 — dev is 50Hz, under-counting by 2.5x otherwise).
5. **⚠️ NEW FINDING (surface to user):** the current `SagaTuning::default().abort_deadline_ticks = 24` (`saga.rs:3524` in tests; verify the real default) is **too tight for prod `n_consecutive_unreachable = 3`** at the default backoffs (recover_budget = 31 ticks @ 20Hz, 91 @ 50Hz). The gate will REJECT such a deployment at boot — correct behavior, but it means prod configs must raise `VD_SAGA_ABORT_DEADLINE` (or the prod `LivenessTuning` must use a lower `n`/tighter backoff). This is the gate doing its job; confirm the intended prod abort budget so tests 9/10 assert the right target. If the intent is that defaults must always self-satisfy, either raise the default `abort_deadline_ticks` or document that prod MUST set it — a config-policy call for you.
6. **`process_parity.rs:217`** — the 5th call site; document-exempt from cross-validate, but plumb the override (or leave defaulted with the `VD_MESH_*`-in-DEV tripwire comment).

## 7. Files/lines to touch (implement-ready)
- `crates/sim/src/saga.rs`: add `use std::time::Duration` (verify absent first); `backoff_series_sum`, `duration_to_ticks_ceil`, `transport_confirm_latency_ticks`, `transport_run_spread_ticks` (free fns); `LivenessCrossTuningError`; `LivenessTuning::validate_against` (after `validate` at `:203`). Tests in the `#[cfg(test)]` mod near `:4340`.
- `crates/io-prod/src/mesh.rs`: `MeshReliabilityOverrides` + `MeshConfig::with_reliability_overrides` (after `:188`); test near `:1895`; **fix the stale CONSUMER NOTE `:92-96`**.
- `crates/io-prod/src/runtime.rs`: `use crate::mesh::MeshReliabilityOverrides`; `EnvConfig::mesh_reliability_overrides`; test near `:248`.
- `crates/io-prod/src/lib.rs`: `pub use` `MeshReliabilityOverrides` (+ `DEFAULT_CONFIRM_UNREACHABLE_AFTER_RETRIES` if not already re-exported).
- `crates/bins/src/bin/orchestrator.rs`: named `overrides` local before `spawn_mesh` (`:48`), chain `.with_reliability_overrides`, `validate_against` boot-assert after `:116` (reading `VD_TICK_HZ`, already read at `:275`).
- `crates/bins/src/bin/{gateway.rs:24, shard.rs:25, client.rs:88}`: chain `.with_reliability_overrides(&env.mesh_reliability_overrides()?)`.
- `crates/bins/tests/process_parity.rs:217`: chain the override or document-exempt with the tripwire comment.

**Bottom line: SOUND TO IMPLEMENT as corrected.** The formula is pure/tokio-free; vd-sim gains no io-prod dependency (raw args); check (a) now sums the correct LATE series segment `[N+1, N+n)` (CRITICAL fixed); check (b) asserts the sound sequential recover-budget ordering (HIGH fixed); all runtime cites re-pointed to `crates/node/src/saga_runtime.rs` (MEDIUM fixed); the 5th call site + stale doc + Option-A reframing folded in. One config-policy question flagged (default `abort_deadline_ticks` vs prod `n=3`) for the user.