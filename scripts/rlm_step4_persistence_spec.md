# RLM Step 4 — Determinism + Persistence: THE definitive vetted spec

Synthesized from 3 scope designs (proptest-only, durable-snapshot, hybrid-minimal) + their
adversarial critiques, re-grounded against the actual tree (HEAD on `worktree-new-system`,
2026-07-24). Every load-bearing claim below was verified in code; file:line anchors cite the
verified site, not a design's assertion.

RLM Steps 1–3 are COMPLETE + committed: `crates/sim/src/rlm.rs` (pure kernel), `crates/node/src/rlm_runtime.rs`
(runtime), the saga_runtime barrier split, all 100% Tier-A; the 3f E2E in
`tests/tests/realm_lifecycle_e2e.rs`. Step-3 spec: `scripts/rlm_step3_reconciler_spec.md`.

---

## 0. SCOPE VERDICT (what Step 4 lands NOW)

**Step 4 lands THREE things; the durable snapshot is NOT one of them.**

| # | Deliverable | Status | Why |
|---|---|---|---|
| **4a** | **Arm the RLM crash-quiesce freeze in prod** (rerouted into `rehydrate`) | **LAND NOW (mandatory)** | The freeze is plumbed but **DEAD in prod** — a real correctness defect independent of the snapshot. It is the SOLE guard for one concrete restart strand (§6-Risk-1). |
| **4b** | **The crash-replay determinism proptest** (mirrors R-6d4-A) | **LAND NOW (the deliverable)** | Generalizes the single 3f E2E into a machine-checked proof of determinism + crash-robustness under arbitrary crash/reorder/dup interleavings. Also the DECIDER for 4c. |
| **4c** | **Durable `StoreKey::Rlm` (tag 6) demand-ledger snapshot** | **DEFER — ready-to-land, soak-gated** | D-RLM-2's binding gate (`rlm_step3_reconciler_spec.md:773-774`): lands *"only if soak testing finds a restart-window strand the CAP-freeze + arm-B don't cover."* The traced prize is ≤1-tick re-spin latency (not a strand/leak/wrong-reap), and the snapshot can't even *replace* the freeze (§4). Full stage/restore/race design is §3, ready to promote if the proptest's divergence oracle or a soak fires. |

### The EVIDENCE — the concrete restart-strand scenario (and what actually fixes it)

Traced tick-by-tick against the verified predicate bodies (`rlm.rs:407-456`):

1. Realm `System(7)` is live + occupied. Its durable head is in the Directory (D-6). Orchestrator kill-9 at tick T. RAM `DemandLedger` lost; head survives.
2. Reboot: `rehydrate` resumes `now` at the recovered ceiling. **Today `arm_quiesce` has NO production caller** — verified: the only two callers are the definition (`rlm_runtime.rs:126`) and one unit test (`rlm_runtime.rs:696`); the boot path `register_orchestrator_with_store` inserts a fresh `RlmReconcilerRes::new(...)` (`orchestrator.rs:133-136`) which hard-sets `rlm_quiesced_until = UniverseTick(0)` (`rlm_runtime.rs:101`). So `quiesced = (now >= 0) = true` ALWAYS ⇒ the freeze clause in `teardown_ready` (`rlm.rs:455`) is inert.
3. Tick T+1: `System(7)`'s own shard (still running) re-asserts `Empty` — its realm genuinely emptied during the outage, OR its own `aoi_decide` reports Empty before the parent's `KeepAlive` re-accrues (the parent shard's demand cadence is independent). `record_demand(Empty)` starts a fresh streak `first_empty_tick=T+1`.
4. **Arm B does NOT protect here** (this is the crux). `desired_alive = demanded_recently | (running_live & !empty_confirmed)` (`rlm.rs:429-430`). With a fresh Empty, `empty_confirmed` is true (`rlm.rs:407-415`: `last_empty_tick` fresh + `last_demand_tick(=0) < streak`), so arm B = `true & !true` = **false**. Arm A false (never demanded, `0` sentinel). ⇒ `System(7)` leaves desired.
5. `teardown_ready` (`rlm.rs:439-456`) needs `empty_confirmed & past_dwell & past_cooldown & quiesced & …`. On a **fresh cell `spawn_watermark=0`** (`rlm.rs:186`), `past_dwell = now.saturating_sub(0) >= min_dwell` = **true** at any real `now`. `past_cooldown` true. `quiesced` true (inert). ⇒ `teardown_ready` fires, drain opens, and after `teardown_drain_ticks` the **occupied realm is KILLED**. A wrongful reap.

**What fixes it: arming the freeze (4a).** During the frozen window (`quiesced=false`), the parent's `KeepAlive` re-accrues (arm A), `refresh_demand` clears the empty streak, and `System(7)` is safe. The durable snapshot (4c) would *also* fix it (arm A holds immediately from the restored parent-demand cell), but it is **not necessary** — the freeze alone closes the window, and the freeze is cheaper and covers the case the snapshot's ≤1-batch off-tick loss cannot (§4).

**Is there a residual strand the ARMED freeze does NOT cover, that only the snapshot fixes?** Only if `recovery_grace_ticks` < the demand re-accrue latency. But demands are **`ReDriven` every tick** by every live realm shard (Step-3 contract; wire `InterShardFlow::RealmDemand`), so the RAM ledger repopulates within ~1 demand round-trip. A freeze sized from the demand cadence (§4, new `RlmTuning::recovery_grace_ticks`) trivially covers it. **No structural strand survives a correctly-sized armed freeze** ⇒ 4c is deferred with evidence, not guessed.

---

## 1. Exactly what lands, where (verified file:line anchors)

### 4a — arm the crash-quiesce freeze (MANDATORY), rerouted into `rehydrate`

The three critiques converge on a non-obvious correctness detail the naive "arm at the orchestrator insert site" plan gets WRONG:

- **Ceiling coherence (verified):** `rehydrate` computes the liveness freeze from the **persisted `ceiling`** at `saga_runtime.rs:1402-1403` (`ceiling.0 + recovery_grace_ticks`) — this runs AFTER `CeilingClock::recover` reserves `next = ceiling + reserve_chunk` and `confirm_ceiling(next)` advances the confirmed ceiling to `next` (`saga_runtime.rs:1332-1337`). So at the `orchestrator.rs:133` insert site, `clock.confirmed_ceiling()` returns `next` (= `ceiling + reserve_chunk`, `reserve_chunk=1024` in the 3f cfg), NOT the raw `ceiling`. A boot-site `clock`-read arm would be **off by `reserve_chunk`** and DIVERGE from the liveness sibling — a coherence/determinism bug. The RLM freeze MUST be computed where the raw `ceiling` is in scope.

- **The discriminant is lost by the insert site (verified):** `register_orchestrator_with_store`'s `match` collapses both `Some(r)` (RECOVER) and `None` (GENESIS) into `(clock, directory, runtime)` at `orchestrator.rs:104,113`; by the `RlmReconcilerRes::new` insert at `:133`, the recover-ness is gone. Arming only-on-recover at the insert site requires re-deriving the discriminant.

**Therefore, land the arm INSIDE `rehydrate`, beside `liveness_quiesced_until`, and surface it on `Rehydrated`:**

- `saga_runtime.rs:1304-1308` — add `pub rlm_quiesced_until: UniverseTick` to `Rehydrated`.
- `saga_runtime.rs:1402-1409` — compute `rlm_quiesced_until = UniverseTick(ceiling.0.saturating_add(rlm_recovery_grace))` beside the liveness arm (ONE formula, ONE place, raw `ceiling` in scope — no drift, no `reserve_chunk` skew). `rehydrate` gains an `rlm: RlmTuning` param (or reads the grace from a threaded `OrchestratorConfig.rlm`) so it knows the RLM grace (§4).
- `orchestrator.rs:96-136` — on RECOVER (`Some(r)`), after building `RlmReconcilerRes::new(cfg.rlm, spawner)`, call `rlm.arm_quiesce(r.rlm_quiesced_until)` before `world.insert_resource`. On GENESIS, leave `rlm_quiesced_until=0` (nothing to protect).
- `rlm_runtime.rs` — add `pub fn rlm_quiesced_until(&self) -> UniverseTick` getter (mirrors `ledger()`/`tuning()`), so the E2E can assert the freeze is ARMED, not incidentally covered by arm-B.

**THE HARNESS BLOCKER (CRITICAL — verified, must fix or 4a is untestable):** `tests/tests/realm_lifecycle_e2e.rs:71` — `assemble` calls `register_orchestrator_with_store` (which would arm the freeze on recover) and then **unconditionally overwrites** the resource: `world.insert_resource(RlmReconcilerRes::new(rlm, Box::new(spawner.clone())))` — a fresh `new()` with `rlm_quiesced_until=0`. And `crash_rebuild` (`:136-142`) rebuilds via `assemble`. So **any boot-path arm is silently discarded in the E2E**, and the existing crash test (`:287`) passes SOLELY on arm-B (empty-ledger-no-candidate) and would stay green even if the freeze regressed. Fix: make `assemble` arm the resource it inserts by mirroring the boot path — thread the recovered `rlm_quiesced_until` through and call `arm_quiesce` on it after the overwrite, OR (cleaner) inject the inspectable spawner INTO `register_orchestrator_with_store` (add a `spawner: Box<dyn RealmSpawner>` param) so the overwrite disappears and the E2E exercises the real production boot path. The injection route is preferred — it also removes a standing "the E2E tests a reconciler the boot path did not build" divergence.

### 4b — the crash-replay determinism proptest (THE deliverable)

- **WHERE:** new inline `mod crash_replay_proptest` inside `#[cfg(test)] mod tests` in `crates/sim/src/rlm.rs` (the pure-kernel crate — no ECS/`World`, so the whole model is per-monomorphization coverable in-crate, the reason the kernel was split pure). `proptest` is **already** a `crates/sim` dev-dep (verified `crates/sim/Cargo.toml:19`) — no dep add, no new-library decision.
- Full design in §2.

### 4c — durable `StoreKey::Rlm` snapshot (DEFERRED, ready-to-land)

Full stage/restore/race design in §3; the soak trigger in §3.5. Nothing lands in the tree now (HR5 forbids inert uncovered reserved code — the exact reason tag 6 was deferred in Step 3). D-RLM-2 stays reserved-prose-only.

---

## 2. The determinism + crash-robustness proptest (mirrors R-6d4-A)

Structural mirror of the vetted R-6d4-A both-ends-restart replay proptest (`crates/io-prod/src/outbox.rs:1459-end`): op-DSL + hand-maintained independent reference mutated in lockstep (never by calling the kernel) + per-op agreement asserts + `Cell<[bool;N]>` anti-vacuity coverage floor + fixed-seed runner + named deterministic witnesses.

**Name:** `rlm_reconcile_survives_arbitrary_crash_interleavings_deterministically`.

### 2.1 What it models — the RAM/durable crash boundary

Mirror `ModelOutboxSink`'s two-set split (`outbox.rs:1477`), scoped to the reconciler:

- **`committed` (survives crash)** = the durable Directory heads (D-6). A `BTreeMap<RealmPath, ModelHead{node, fence}>` mirroring `OwnerRecord`. Kept across `Crash`.
- **RAM (lost on crash)** = the live `DemandLedger` (`rlm.rs:198`) + `LaunchLedger.minted` (`rlm_runtime.rs:38-44`) + `rlm_quiesced_until`. Cleared on `Crash`.

The model drives the REAL production functions: `DemandLedger::record_demand`/`apply_delta` (`rlm.rs:230/267`) and the pure `reconcile` (`rlm.rs:553`), plus `drive_drain` (`rlm.rs:523`). It builds a real `DirectoryCore` for the head reads (the existing `dir(...)`/`grant`/`head` test fixtures in `rlm.rs`). **NB the running-set is modeled as the durable `committed` heads + a `live` node set** — the exact shape Step 5's `live_nodes()` will provide (§5), so the model needs zero change when Step 5 lands.

### 2.2 The op-DSL (mirror `outbox.rs:1624-1642`)

```
enum Op {
    Demand   { realm: u8, verb: DemandVerb, tick_delta: u8, fence: u8 },  // a re-asserted wire demand
    Sweep,                     // reconcile(...) at the current `now` + apply_delta
    IdleTick { by: u8 },       // advance the clock with no demand (age windows)
    GrantHead  { realm: u8, node: u8 },  // a spawned shard self-grants its head (durable)
    RevokeHead { realm: u8 },            // liveness/revoke removes the head (durable)
    MarkDead   { node: u8 },             // liveness latches a node dead (RAM oracle → zombie path)
    Crash,     // RAM lost (ledger + launches + quiesce cleared), durable heads kept; re-arm quiesce = now + grace
    Reboot,    // rebuild the reconciler from the durable half (empty ledger in ARM-A; snapshot-derived in ARM-B)
    ReorderDup { realm: u8, verb: DemandVerb, count: u8 },  // a multiset folded shuffled + duplicated
}
```

- `realm ∈ {7,8,9}` incl. a parent/child pair so ancestor-closure + `has_desired_descendant` (`rlm.rs:490`) fire; one realm stays never-demanded.
- `verb` covers all four `DemandVerb` arms (exercises `record_demand`'s full fold incl. the `Empty`-latch + arm-C streak).
- `tick_delta` bounded so `demand_ttl`/`empty_grace`/`drain`/`cooldown`/`min_dwell` windows are actually crossed within a `vec(op, 0..=64)` run. Fixed `RlmTuning::cloud(20)`.
- `now` advances **monotonically** (never rewinds — the clock never rewinds on recover, `saga_runtime.rs:1332`), so the model respects the real clock contract. **Crash can land between ANY two ops**, including immediately before a `Sweep` — the "off-tick writer races the reconcile scan" position (§3.4 proves the design cannot race there).

### 2.3 What it asserts

A hand-maintained independent reference (`desired_ref: BTreeSet<RealmPath>` + a projected ledger), mutated BY HAND from the op history under the documented `desired_alive`/`teardown_ready` law (never by calling the kernel — the anti-tautology oracle, mirror `outbox.rs:1674`). After EVERY op:

- **INV-DET (byte-identical replay — the determinism core).** Keep a full `Vec<Op>` history; after each op, run the WHOLE history from scratch through a FRESH `(ledger, dir, launches, quiesce)` and `assert_eq!` the resulting `(Vec<LifecycleAction>, LedgerDelta, ledger-state, head-state)` at the current point against the incrementally-evolved live state. Generalizes the run-twice test (`rlm.rs`, `reconcile_is_deterministic_run_twice`) over arbitrary crash-interleaved histories. **Scoped honestly (see the critique HIGH):** this is a PURE-KERNEL property — `reconcile`'s actions are byte-identical for identical `(ledger, dir, liveness, launch_live, tuning, now, quiesced_until)`. The full-stack cross-crash claim is weaker: minted `NodeId`s differ across a rebuild (the 3f harness uses base `1_000_000` vs `2_000_000`, `realm_lifecycle_e2e.rs:83,139`), so `Kill{node}`/`ForceReap{node}` PAYLOADS differ continuous-vs-crashed. **The spec's determinism claim is: identical up to node-id renaming — the ledger-state, head-state, desired-set, and action COUNTS are invariant; the raw `NodeId` bytes are not.** The full-stack E2E (§5-slice-4c) asserts counts + head-state, never raw node ids, for exactly this reason. INV-DET must pin `now` identically on both replay sides (both resume at the same recovered ceiling), else the `demanded_recently`/`empty_confirmed` windows diverge legitimately.
- **INV-CRASH-NO-REAP (the 3f property, generalized + the freeze exercised).** On ANY `Sweep` in `[Crash, Crash + grace)` where the pre-crash ledger had no fresh `Empty`, assert the actions contain ZERO `Kill` of a still-live head. Generalizes `an_orchestrator_crash_reaps_nothing...` (`realm_lifecycle_e2e.rs:287`) from ONE 50-tick no-demand run to arbitrary interleavings — AND exercises the ARMED freeze (2.2 `Crash` sets `rlm_quiesced_until`), which the E2E does NOT (its harness overwrite kills the arm, §1).
- **INV-NO-STRAND.** A realm with a live head that never folded a fresh `Empty` is in `desired`/`closed` on every `Sweep` (arm-B core, `rlm.rs:430`), independently computed by the hand-reference.
- **INV-FOLD-ORDER (reorder/dup determinism).** For every `ReorderDup`, fold the multiset into a scratch ledger in a SECOND independent order + with duplicates; `assert_eq!` the two `LedgerCell`s byte-equal. Proves `record_demand` commutes over reorder + dup, generalizing the existing `last_fence_tracks…_order_independently` (`rlm.rs:824`) and the arm-C reorder test (`rlm.rs:1038`). **Also add a reorder-ACROSS-CRASH property:** permute the post-`Reboot` demand suffix → identical cell (nearly free given the DSL; closes the critique LOW).
- **INV-BOUNDED (self-heal doesn't leak).** After a `Crash`+re-accrue, `ledger.len()` never exceeds the distinct realms ever demanded post-crash — the `retire` path (`rlm.rs`, `delta.retire.insert`) keeps it bounded. Mirror of R-6d4-A's accounting (`outbox.rs`).
- **INV-DELTA-CONSISTENT.** After `apply_delta`, each path's `draining_since` equals what the hand-reference predicts from `drive_drain` (`rlm.rs:523`) — a crash mid-drain drops `draining_since` and the post-crash sweep re-opens it deterministically, never double-killing.

### 2.4 The DIVERGENCE ORACLE (the DECIDER for 4c)

Run BOTH arms over the SAME op-vec (this is the single highest-value idea across the four designs):

- **ARM-A (no-snapshot — the current + 4a-armed prod behaviour):** `Reboot` restores an EMPTY ledger + arms quiesce = `now + grace`.
- **ARM-B (snapshot dry-run — a MODEL of §3, test-only, no tree code):** `Crash` snapshots the live ledger into a durable shadow; `Reboot` restores from it + arms quiesce.

**INV-SNAPSHOT-EQUIVALENCE:** compare ARM-A vs ARM-B action traces on the identical op-vec.
- **If byte-identical for all 1024 fixed-seed cases** ⇒ the durable snapshot changes nothing the RAM self-heal + armed freeze don't already achieve ⇒ **4c stays deferred** (D-RLM-2 unwritten). This is the expected outcome (§0 evidence + §8 prior).
- **If they DIVERGE** — ARM-A emits a spurious `Kill`/`SpinUp` in a post-crash window that ARM-B (with the restored demand set) does not — that divergence IS "the restart-window strand the CAP-freeze + arm-B don't cover" (D-RLM-2's exact trigger). The divergence op-vec becomes the regression witness for §3.

Note ARM-B here is a TEST-ONLY model of the snapshot decision — it lands NO product code, so it carries no HR5 obligation and does not violate the "no inert reserved code" rule.

### 2.5 Anti-vacuity coverage floor + witnesses (mirror `outbox.rs:1874-1878`)

`cov: Cell<[bool;6]>` recording which hard arms fired across the 1024-case run; assert all six after:
1. a `Crash` with a non-empty pre-crash ledger (RAM actually lost);
2. a `Sweep` inside the post-crash freeze window that would-otherwise-reap but did not (freeze fired);
3. a `Kill` actually emitted somewhere (the reap path exercised, not vacuously absent);
4. a `MarkDead`→`Sweep` emitting a `ForceReap` (zombie path);
5. a `ReorderDup` producing a non-trivial (≥2, reordered) multiset;
6. a `Crash` landing in the tick immediately before a `Sweep` (the off-tick-race position).

Fixed-seed runner: `TestRunner::new_with_rng(Config{cases:1024,..}, TestRng::from_seed(ChaCha, &[0x72;32]))` (deterministic repro, mirror `outbox.rs`). Named deterministic witnesses alongside (each asserting its cov flag, mirror `outbox.rs:1881+`): `witness_crash_before_sweep_reaps_nothing_in_grace`, `witness_snapshot_and_ramheal_agree_on_a_simple_trace`, `witness_zombie_after_reboot_force_reaps`, `witness_reorder_dup_folds_identically`, `witness_crash_mid_drain_reopens_not_double_kills`, `witness_ancestor_closure_survives_crash`.

### 2.6 Soak variant (the "long runs" — the concrete strand trigger)

One additional release-gated `#[test]` (mirror the SPIKE-3a latency-gate `just <name> release` pattern in `just gate`, task #127): a single very long op stream (≈100k ops) with periodic `Crash`es + continuous re-accrue, asserting INV-DET + INV-BOUNDED + INV-CRASH-NO-REAP + INV-SNAPSHOT-EQUIVALENCE throughout. **A sustained ARM-A≠ARM-B divergence, or an INV-NO-STRAND/leak violation, is the trigger to promote §3.** Add `just rlm-soak` to `just gate`.

---

## 3. Durable snapshot — READY-TO-LAND, DEFERRED (§3.5 = the trigger)

Kept OUT of the tree (HR5: no inert uncovered reserved code). Documented so promotion is a zero-migration add. The StoreKey byte format + postcard wire are append-only; every touch below is additive.

### 3.1 The key/wire (`saga_runtime.rs`)
- `saga_runtime.rs:88-95` — add `Rlm(RealmPath)` to `enum StoreKey`. Keyed by the FULL `RealmPath` (matching `DemandLedger.cells: BTreeMap<RealmPath, LedgerCell>`, `rlm.rs:199`), NOT the lossy `lowered()` `RealmId`.
- `saga_runtime.rs:98-103` — add `const RLM: u8 = 6;` (next free after `ABORT_REPLY=5`, verified `:103`; frozen append-only).
- `saga_runtime.rs:108-128` — add the `bytes()` arm mirroring `Directory` (`:119-122`): `k.push(Self::RLM); k.extend(postcard::to_allocvec(&p).expect("encode RealmPath store key"))` (one straight-line monomorphic `.expect`, HR5-clean per the codebase pattern doc at `:105-107`).
- **`StoreKey` loses `Copy`:** `RealmPath` holds a `Vec<RealmLevel>` (not `Copy`). Verified every `.bytes()` call takes `self` by value and all other uses pass owned/`*`-copied values (`directory_store_key` at `:139`, the group_commit sites at `:2442-2448`, the test sites) — so `#[derive(Clone)]` (dropping `Copy`) compiles. The enum is private (`enum StoreKey`, no `pub`), used only inside `saga_runtime.rs`. **Cost noted:** the `Rlm` arm's `.clone()` allocates per changed cell per commit — under the 100k-realm SCALE framing this is a per-tick allocation the RAM-only path avoids; bounded by "cells changed this tick" (§3.2), but quantify in the promotion PR.

### 3.2 The value record + stage (`saga_runtime.rs` + `rlm_runtime.rs`)
- Value: `RlmCellSnapshot { path: RealmPath, cell: LedgerCell }`, `derive(Clone, Debug, Serialize, Deserialize)` (mirror `DirSnapshot`), self-describing so rehydrate never parses a key back (`spec:83-84`). **Required change:** `LedgerCell` (`rlm.rs:155`) currently derives `Clone, Debug, PartialEq, Eq` — add `Serialize, Deserialize`. Verified every field is postcard-serializable (`RealmCoord`, `UniverseTick`, `Option<UniverseTick>`, `Fence`; `RealmCoord`/`RealmPath`/`RealmLevel` are serde-derived — they ride `RealmDemand` on the wire). One cross-crate touch in `vd-sim`; a store record, not a frozen wire frame.
- **`draining_since` is NOT resumed on restore.** Persist the field for audit but restore it as `None` (force a fresh post-reboot drain). **Rationale (a real hazard the durable-snapshot design got wrong):** a restored elapsed `draining_since` + a restored-live head + an EMPTY post-reboot liveness tracker (`saga_runtime.rs:1393` rebuilds it empty; everyone is `!is_latched_dead`) = an INSTANT kill the moment the freeze lifts, with zero post-reboot occupancy re-confirmation. Restoring `draining_since=None` forces the drain to re-open under fresh liveness. (INV-DELTA-CONSISTENT in §2 already asserts this shape.)
- **FULL-per-cell, incrementally-KEYED (verdict).** Each changed cell is written whole (8 fields), but only cells that CHANGED this tick are written — not a field-delta WAL (which adds an uncoverable delta-apply branch per field, HR5) and not a whole-ledger monolith (which is O(all realms) per tick, defeating the O(changes) barrier discipline and losing the splittable-prefix property). The ledger is BOUNDED by `reconcile`'s retire (`rlm.rs`), so full-cell-per-changed-cell is cheap.
- **Dirty tracking:** give `RlmReconcilerRes` a per-tick `dirty_cells: BTreeSet<RealmPath>` + `forget_cells: BTreeSet<RealmPath>`. `record_realm_demands` (`rlm_runtime.rs:301`) inserts touched paths into `dirty_cells` after the fold; `reconcile_and_drive` (`rlm_runtime.rs:136`) unions `delta.set_draining.keys()` + spawned/reaped coords into `dirty_cells` and `delta.retire` into `forget_cells`. A `stage_durable(&mut self, writes)` drains them: for `dirty_cells.difference(&forget_cells)` push `(StoreKey::Rlm(path).bytes(), Some(encode(&snap)))`; for `forget_cells` push `(…, None)` (durable DELETE, the tombstone). The `difference` filter guarantees delete-wins for a same-tick dirty-then-retire. **Tombstones derive from `delta.retire`, NOT from iterating the post-`apply_delta` ledger** (the retired cell is already gone by then — a real footgun the durable design's prose glossed).

### 3.3 Where it stages into the barrier
`stage_durable` pushes onto the EXISTING `SagaRuntimeRes.pending_writes: Vec<(Vec<u8>, Option<Bytes>)>` (`saga_runtime.rs:449`), drained by `group_commit`'s existing loop (`:2432-2437`) into the ONE `store.commit()` (`:2451`) — atomically with saga+directory+clock. **Wiring caveat (a real HIGH the "one param flip" summary undercounts):** `pending_writes` is a **PRIVATE** field of `SagaRuntimeRes` (verified `:449`, no `pub`), and `reconcile_realm_lifecycle` (`rlm_runtime.rs:336`) is in a different module. It CANNOT touch the field directly. Add a `pub(crate) fn stage_write(&mut self, key, val)` accessor on `SagaRuntimeRes` (or `pub(crate)` the field). `reconcile_realm_lifecycle` already reads `runtime: Res<SagaRuntimeRes>` for the liveness latch (`:339`) — flip it to `ResMut` and call `rlm.stage_durable(&mut runtime)` at the tail (`:351`), AFTER `reconcile_and_drive` mutated the ledger, BEFORE `commit_barrier`. `dirty_cells` is populated inside `reconcile_and_drive` (owns `rlm`) and drained outside (needs `runtime`) — a two-borrow dance across the same system, workable but NOT "one line". **No change to `group_commit`/`commit_barrier` bodies.**

### 3.4 Restore + boot wiring
- `saga_runtime.rs:1383-1388` — after the AbortReply loop, add a `scan(&[StoreKey::RLM])` block (mirror it exactly): decode each `RlmCellSnapshot` into `rlm_cells: BTreeMap<RealmPath, LedgerCell>`, restoring `draining_since=None`. Surface on `Rehydrated` (`pub rlm_cells`).
- `orchestrator.rs:133-136` — on RECOVER, build `RlmReconcilerRes::with_ledger(cfg.rlm, spawner, r.rlm_cells)` (a constructor = `new` + a `DemandLedger{cells}` assignment; `new` becomes `with_ledger(t,s,BTreeMap::new())` so there is ONE body — no uncovered duplicate). Genesis passes empty ⇒ byte-identical to `new()`.
- **`LaunchLedger.minted` is NOT restored** — it re-derives from `spawner.live_nodes()` on the first `reconcile_launches` sweep (`rlm_runtime.rs:245-259`, verified `live_nodes()` at `:246`). Persisting it would DUPLICATE Step 5's running-set bridge (§5).

### 3.5 Off-tick-writer race avoidance (D-6 precondition #2) — resolved BY CONSTRUCTION
The precondition warns "the off-tick writer RACES the reconcile scan." For the RLM family it **cannot**, verified:
1. **The reconciler NEVER `scan()`s the store.** `reconcile` (`rlm.rs:553`) reads `&DemandLedger` (RAM) + `&DirectoryCore` heads (RAM); `reconcile_and_drive` (`rlm_runtime.rs:136`) touches no store. The ONLY `scan(&[RLM])` is in `rehydrate` (§3.4), at BOOT, single-threaded, before any tick runs and before the writer processes any batch for this incarnation.
2. **This is precisely the D-alpha invariant, already retired store-wide** (verified `store.rs:782-784`, verbatim): *"D-alpha made the directory reconcile INCREMENTAL — a `dirty`-delta drain that never reads the store back — so the off-tick writer no longer races a reconcile read: the prior 'scan reads T-1' coupling is GONE."* The RLM family inherits this — it has no reconcile-scan at all, so it is strictly SAFER than the directory family. **No WAL-version / no tombstone table / no incremental-reconcile machinery needed** — the existing `Option<Bytes>` DELETE (None) in `pending_writes` IS the tombstone, drained atomically.
3. **Persist-before-effect + ≤1-batch bound preserved:** the barrier commits the RLM rows in the SAME fsync as the directory grant/revoke the sweep produced, so a crash can never leave a persisted `Kill`-revoke without the ledger state that justified it. The io-prod depth-1 block-on-prior (`store.rs:781`) bounds crash-loss to ≤1 batch ⇒ at most one tick of demand deltas lost, which re-accrue next tick (`ReDriven`). **This ≤1-batch loss is exactly why the snapshot cannot REPLACE the freeze** (§4).
- **Tripwire (plant on promotion):** a unit test `rlm_durable_reconcile_never_scans_the_store` runs a reconcile sweep against a `Store` whose `scan` panics; asserts the sweep completes. If a future change makes the reconciler read the store mid-tick (reintroducing the race), it goes RED. Mirror the outbox anti-drift hook shape.

---

## 4. Quiesce-window verdict: KEEP, ARM, RESIZE (do NOT drop)

**Verdict: the freeze is load-bearing but currently DEAD in prod; arm it (4a), give it its OWN tuning, and keep it even if 4c ever lands.**

- **Correct kernel, dead in prod (verified):** `teardown_ready` gates on `facts.quiesced = now >= quiesced_until` (`rlm.rs:455`, wired at the reconcile site `now >= quiesced_until`). But `arm_quiesce` has no prod caller and `new` hard-sets `rlm_quiesced_until=0` (§0 evidence), so `quiesced` is always true ⇒ the freeze is inert. The `new` doc's promise ("the crash-restart path re-arms it, slice 3e", `rlm_runtime.rs:89`) was never delivered. The sibling `liveness_quiesced_until` IS armed (`saga_runtime.rs:1402`). Arm the RLM freeze the same way, in the same place (§1-4a).
- **Load-bearing, NOT merely belt-and-suspenders:** §0's strand (fresh Empty during outage → arm-B OFF, `spawn_watermark=0` → dwell bypassed) has the freeze as its SOLE guard. So arming it is a correctness fix, not decoration.
- **Does the durable ledger let us DROP the freeze (warm restart)?** **No.** Even with a full durable ledger restoring the parent's demand cell (arm A holds tick T+1), the io-prod depth-1 writer can lose the LAST un-fsynced tick's demand deltas (§3.5 point 3), leaving a residual 1-tick gap that still needs a freeze. So the snapshot cannot eliminate the freeze — it only DEMOTES it from "the sole guard" to "a short grace" — which further shrinks 4c's payoff (§8).
- **Resize — give RLM its OWN grace (a real HR-no-magic-number fix):** the design's "reuse `DirectoryTuning::recovery_grace_ticks`" (verified live + validated at `saga_runtime.rs:1403`) is CONVENIENT but WRONG-homed: that field is sized for the LIVENESS re-accrue cadence (dead-peer renewal), not the RLM DEMAND re-accrue cadence (a parent shard's KeepAlive interval). Add `RlmTuning::recovery_grace_ticks`, derived in `RlmTuning::cloud(hz)` from the demand cadence (≈`demand_ttl_ticks` — a realm must survive at least one full demand-TTL of silence so a re-accrued demand can break it). Too short ⇒ the §0 strand before the parent's KeepAlive lands; too long ⇒ a genuinely-abandoned realm lingers `grace` extra ticks (benign — out of AoI, costs one pod). `rehydrate` reads this RLM grace (§1-4a).

---

## 5. Composition with Step 5 `live_nodes()` — no duplication (structural)

The durable ledger (if it lands) and `live_nodes()` are the two ORTHOGONAL halves of the level-triggered reconcile (`desired_alive` vs `running_live`), and CANNOT overlap:

| Axis | Question | Durable owner | Key domain | Feeds |
|---|---|---|---|---|
| **DESIRED (intent)** | which realms SHOULD exist / who demands them | `StoreKey::Rlm` (Step 4, deferred) | `RealmPath` (`rlm.rs:199`) | `desired_alive` arm-A/B (`rlm.rs:429-431`) |
| **ACTUAL (running)** | which shard pods physically exist | `RealmSpawner::live_nodes()` (Step 5, cluster-derived) | `NodeId` (`rlm_runtime.rs:517,532`) | `reconcile_launches` / `running_live` (`rlm.rs`) |

1. **Disjoint key spaces** — `RealmPath` (demand facts) vs `NodeId` (pod incarnations). Neither encodes the other. The snapshot stores demand facts, NEVER node liveness; `live_nodes()` returns a `BTreeSet<NodeId>`, NEVER demand.
2. **Different predicates** — verified `reconcile(&ledger, dir, liveness_dead, launch_live, tuning, now, quiesced_until)` (`rlm.rs:553-561`) already SEPARATES the inputs: `ledger` (Step-4 durable-source) vs `launch_live` (Step-5 running-source). Step 4 changes only the ledger's SOURCE (rehydrate-seeded vs empty); Step 5 changes only `live_nodes()`'s backing (cluster vs mem-twin). Same signature; they compose on the existing decide seam with ZERO migration.
3. **`minted` explicitly NOT persisted** (§3.4) — it re-derives from `live_nodes()` on the first post-reboot `reconcile_launches`. Persisting it WOULD be the duplication; not persisting it is the seam.
4. **No foreclosure** — arming the freeze (4a) blocks only TEARDOWN; spawn/launch-reconcile via `live_nodes()` runs freely during the freeze. A warm restart with the durable ledger + a Step-5 cluster `live_nodes()` yields the fully-correct first-sweep decision (right desired × right actual), the end-state both steps converge on. The proptest's `GrantHead`/`MarkDead`/`RevokeHead` ops (§2.1) already drive the `live_nodes()` seam as a durable post-crash input, so it catches any launch-set interaction with zero proptest change when Step 5 lands.

---

## 6. Risk register (each stress item: resolved-how / deferred-because)

| Stress | Resolution |
|---|---|
| **1. Restart-strand reality** | **RESOLVED by 4a (arm the freeze), NOT by 4c.** The RAM path never wrongly REAPS the demand set via arm-B for a still-demanded realm, never LEAKS a dead realm (ForceReap is head-driven), and does not STRAND the demand set — EXCEPT the one traced window (fresh Empty during outage, arm-B OFF, dwell bypassed by `spawn_watermark=0`) where the DEAD freeze is the sole missing guard. Arming it (§0/§4) closes it. Residual without the snapshot: ≤1-tick re-spin latency for the narrow "head-also-lost + demand-only-in-RAM" case, refilled by per-tick `ReDriven` demands. |
| **2. Off-tick-writer race (D-6 #2)** | **RESOLVED / does-not-exist for this family** (§3.5). The reconciler never `scan()`s the store; the only scan is boot-time single-threaded; D-alpha already retired the reconcile-scan-races-writer coupling store-wide (`store.rs:782-784`). No WAL/tombstone needed; the `Option<Bytes>` DELETE is the tombstone. Tripwire planted on promotion. |
| **3. Determinism across a crash** | **RESOLVED, honestly scoped** (§2.3 INV-DET). Byte-identical up to node-id renaming: ledger/head/desired/action-COUNTS invariant; raw `NodeId` payloads differ (rebuild mints from a different base). Full-stack E2E asserts counts+head-state only. INV-DET pins `now` identically on both replay sides. |
| **4. Reorder/dup/delayed demands** | **RESOLVED** (§2.3 INV-FOLD-ORDER). The fold is commutative/idempotent by construction (`max`/latch/`(tick,fence)`-max, verified `rlm.rs:283-305`); existing tests + the proptest's reorder-across-crash property nail it. Snapshot persists the FOLDED cell, adding no order-dependence. |
| **5. Quiesce-window sizing** | **RESOLVED: keep + arm + resize** (§4). Currently dead; arm inside `rehydrate` with the coherent persisted `ceiling`; give RLM its OWN `RlmTuning::recovery_grace_ticks` (not the Directory's). The snapshot CANNOT drop the freeze (≤1-batch off-tick loss leaves a residual gap). |
| **6. Foreclose/duplicate Step-5 `live_nodes()`** | **RESOLVED: structurally orthogonal** (§5). Disjoint key domains + disjoint predicates + `minted` not persisted. Neither re-implements the other; they compose on the existing `reconcile` signature with zero migration. |
| **7. HR5 / no inert reserved code** | **RESOLVED.** Tag 6 stays UNWRITTEN until 4c triggers (no inert uncovered arm). The proptest is test-only. 4a's new branch (recover-arms vs genesis-skips) is covered by the fixed E2E (recover) + all genesis boots; the getter is straight-line. 4c's HR5 landmine flagged: `dirty_cells` inserts are new regions in `DemandLedger` (generic, instantiated in vd-sim + vd-node + vd-tests binaries — the per-monomorphization × per-binary llvm gotcha); they must be driven or skipped WHOLESALE in all three (§7 sign-off). |
| **8. Worth-it / minimal scope** | **RESOLVED: DEFER 4c, SHIP 4a+4b** (§8). |

---

## 7. HR5 + determinism sign-off

- **HR5 (100% region+branch of new code):**
  - 4a: `Rehydrated.rlm_quiesced_until` compute (straight-line), the recover-arm branch (covered by the fixed crash E2E), the genesis-skip (covered by every genesis boot test), the getter (straight-line + unit test), the harness injection (exercised by all E2Es). No uncovered region.
  - 4b: test-only; drives the ALREADY-100% kernel (`reconcile`/`record_demand`/`apply_delta`/`drive_drain`/predicates) — verify `just coverage-fast` stays 100% (the proptest introduces no product branch). `Cell<[bool;6]>` floor + named witnesses guarantee every hard arm fires deterministically even if `cases` is ever reduced.
  - 4c (deferred): lands LIVE at 100% when triggered — the `bytes()` arm is CALLED (roundtrip test), the scan/restore mirrors the 100%-covered AbortReply block, `dirty_cells`/`difference`/PUT/DELETE arms enumerated in unit tests, `with_ledger`/`new` share ONE body. Explicit obligation: cover `DemandLedger`'s new `dirty` inserts in vd-sim AND drive-or-skip them in the vd-node runtime + vd-tests E2E binaries (per the CLAUDE.md generic-coverage gotcha). No `coverage(off)` needed. The reserved tag lands ONLY with real use — clearing the exact HR5 reason it was deferred in Step 3.
- **Determinism:** `reconcile` is pure `f(...)`; keys `BTreeMap<RealmPath>` (Ord, collision-free); no wall clock / no default hasher; run-twice + 3f tests exist. The proptest generalizes run-twice over crash-interleaved histories. Node-id renaming across a rebuild is the ONE non-determinism source; it is a Step-5-owned `NodeId` mint concern, scoped out of the kernel INV-DET and out of the counts-only full-stack assertion. INERT-by-default preserved: `RlmTuning::reconcile_interval_ticks==0` early-returns before reading `rlm_quiesced_until`, so 4a is byte-identical for non-opted-in builds (the walk/canonical gate is safe).

---

## 8. Drift check vs the end-goal + WORTH-IT verdict

**End-goal (SC + Minecraft: 100k users, seamless warp, cloud/k8s, signal-heavy cross-shard blocks):**
- **100k users / scale:** the demand ledger is bounded by `retire`; the proptest's INV-BOUNDED + soak prove the RAM self-heal doesn't leak over long runs — the property that matters at scale. The deferred snapshot's per-changed-cell `RealmPath.clone()` allocation (§3.1) is a SCALE cost the RAM path avoids, reinforcing defer-until-needed.
- **Seamless warp / dynamic realm lifecycle:** RLM is THE warp mechanism (demand-driven spin-up by AoI). Step 4 hardens its DETERMINISM + crash-robustness — a prerequisite for a warp that never wrongly reaps a realm you are flying toward. Arming the freeze (4a) directly prevents the "reboot mid-warp reaps the destination realm" corner.
- **Cloud/k8s + Step 5:** the composition proof (§5) guarantees Step 4 does not corner Step 5's `live_nodes()` bridge — the two restore halves (desired vs running) stay orthogonal. Deferring 4c also avoids landing a durable artifact on the WRONG axis before Step 5's running-set bridge exists.
- **No drift:** Step 4 changes the reconciler's ROBUSTNESS, not its shape; it does not paint any pillar into a corner.

**WORTH-IT verdict: DEFER the durable snapshot; the high-value Step-4 deliverable is the proptest + the freeze-arming fix.** Evidence: (1) D-RLM-2's binding gate already mandates "only if soak finds a strand," and no soak has run; (2) the only traced strand is closed by arming the DEAD freeze (4a, ~15 lines), not by the snapshot; (3) the snapshot cannot even REPLACE the freeze (≤1-batch off-tick loss) — it only demotes it; (4) demands are `ReDriven` every tick, so the RAM ledger repopulates in ~1 round-trip; (5) the snapshot's cost is real (StoreKey `Copy` drop + per-cell clone + private-field encapsulation break on the D-6 barrier + cross-crate `LedgerCell` serde + a 3-binary generic-coverage obligation + a `draining_since`-restore hazard) — all to save one tick. The proptest's ARM-A-vs-ARM-B divergence oracle (§2.4) + the soak (§2.6) turn "do we need durability?" from a judgment call into a machine-checked decision that will TRIGGER the ready-to-land §3 the moment (and only if) a real strand appears.

---

## Slice plan

| Slice | WHAT | WHERE | TESTS | HR5 |
|---|---|---|---|---|
| **4a** | Arm the RLM crash-quiesce INSIDE `rehydrate` (coherent persisted `ceiling`); add `RlmTuning::recovery_grace_ticks`; surface `rlm_quiesced_until` on `Rehydrated`; arm on RECOVER at boot; add `rlm_quiesced_until()` getter; **FIX the E2E harness overwrite (inject the spawner into `register_orchestrator_with_store`)** | `saga_runtime.rs:1304-1308,1402-1409`; `rlm_runtime.rs` (tuning+getter); `orchestrator.rs:96-136`; `realm_lifecycle_e2e.rs:64-78` | Extend `an_orchestrator_crash_reaps_nothing...` to assert `rlm_quiesced_until() > now` post-reboot (proves ARMED, not incidental); an E2E that the freeze RELEASES after grace (reaps a genuinely-empty realm after the window); a unit test that genesis leaves `rlm_quiesced_until==0` | recover-arm + genesis-skip branches covered by the fixed E2E + genesis boots; getter straight-line; INERT-when-cfg-zero covered by all non-RLM boots |
| **4b** | The crash-replay determinism proptest + divergence oracle + soak | `crates/sim/src/rlm.rs` (`mod crash_replay_proptest` in `mod tests`); `justfile` (`just rlm-soak`) | the proptest (1024 fixed-seed) + 6 named witnesses + the release-gated soak IS the test | test-only; drives already-100% kernel; verify `just coverage-fast` stays 100% |
| **4c (DEFERRED)** | Land `StoreKey::Rlm` tag 6 + `RlmCellSnapshot` + `LedgerCell` serde + dirty-drain + `stage_durable` (via a `pub(crate)` accessor) + `rehydrate` scan + `with_ledger` boot seed + `draining_since=None` restore + the anti-scan tripwire; add D-RLM-2🟩 to `docs/design/DEFERRED.md` | §3 anchors | roundtrip (key+cell); `stage_durable` PUT/DELETE/difference arms; rehydrate restore (empty + non-empty scan); a warm-restart E2E (the 4b divergence witness, now green); the anti-scan tripwire | lands LIVE at 100% (tag 6 CALLED); the 3-binary `dirty`-insert obligation enumerated |
| **4d** | If 4b does NOT diverge: record the deferral. Add D-RLM-2 to `docs/design/DEFERRED.md` as 🟨-deferred with the proptest+soak as the evidence it is not yet needed; update the `DemandLedger` doc (`rlm.rs:194`) to cite the proptest as the proof the RAM self-heal holds; keep tag 6 UNWRITTEN | `docs/design/DEFERRED.md`; `rlm.rs:194` | n/a (doc) | n/a |

**Ship order:** 4a → 4b → (4d if no divergence | 4c if divergence/soak-strand). 4a lands regardless — the dead freeze is a defect independent of the snapshot question.
