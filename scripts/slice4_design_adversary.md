# Slice 4 design + adversary (wf_76e783d9)

## DESIGN

Excellent. There's a critical reality I must surface: `retain_live` at lines 2399-2401 ALREADY evicts CrossingProgress/RequestInFlight/InterestZones for departed subjects — the ledger-eviction-on-leave gate (#7 in the task) is **already implemented**, so Slice 4's job is to add a *test* proving it, not the eviction itself. I have complete ground truth now. Writing the design.

Terse note on one more reality check — the parent mutation. The `RealmBoundary.parent` is a `Copy` field inside `RealmBoundaries(Vec<...>)`, a Bevy resource on the shard. The undock destination on an outward crossing needs `winner.parent`, and the dock destination needs `winner.to_realm`. This is a pure read of existing fields — the *mutation* the plan demands is re-parenting a nested boundary at commit. Let me produce the plan.

---

# Slice 4 — Station Nesting + Docking + Area Interest: implementation plan

**Absolute paths** (all relative to `…/worktrees/new-system/`): `crates/core/src/geometry.rs`, `crates/sim/src/stub.rs`, `crates/sim/src/saga.rs`, `crates/node/src/saga_runtime.rs`, `crates/wire/src/intershard.rs`, `crates/sim/src/capability.rs`, plus tests in `crates/sim/tests/`.

## CRITICAL reality/plan disagreements (read first — three change the plan)

- **R-A. `to_realm` is the winner's STATIC field; `_dir` is DISCARDED at `fan_out_crossing`.** (`stub.rs:2636` `let to_realm = winner.to_realm;`, `:2628` `_dir: Direction`.) The plan's "undock outward → to_realm=parent" is **not** expressible by "only the (from_realm,to_realm) tuple differs" *unless the destination is selected by `dir`*. Undock destination must be `winner.parent` (an `Option<RealmId>`), not `winner.to_realm`. Edit 1 makes `fan_out_crossing` branch the *destination* on `dir` — a `(from,to)`-tuple-only difference at the **wire** level (`CrossingRequest` still has no dir field), achieved by an in-fn destination pick. This is NOT a realm-kind match (HR3-safe): it branches on `Direction`, a total 2-valued enum.
- **R-B. Ledger-eviction-on-leave is ALREADY IMPLEMENTED.** `retain_live` at `stub.rs:2399-2401` already evicts `CrossingProgress`/`RequestInFlight`/`InterestZones` for non-resident subjects. Slice 4 owes only the **blocking test** (Edit 7), not the machinery. The task framing ("evicted on entity removal — the InputLog-leak class") is satisfied structurally; the test proves it.
- **R-C. "Parent MUTATION at post-CAS commit" has no persisted home.** `OwnerRecord` (`wire/src/seams/directory.rs:66-84`) has no parent field; `RealmBoundary.parent` is a `Copy` descriptor in the shard-local `RealmBoundaries(Vec)` resource, and the CAS commits on the **orchestrator**, while the boundary lives on the **ship-hosting shard**. The plan's "mutate `RealmBoundary.parent` at the saga post-CAS commit keyed by fence" is therefore a **fence-carried command from orchestrator→ship-shard**, applied when the ship shard learns of commit — i.e. it rides the existing `DemoteCmd` (already fence-bearing, already routed to the source/ship shard, already the latch-clear terminal at `on_saga_demote` `stub.rs:1597`). Edit 2 adds a `new_parent: Option<RealmId>` field to `DemoteCmd` and applies it there. This is the ONLY seam that already delivers a fence to the shard owning the boundary; inventing a new flow arm would violate "reuse one machinery." **Flag: for the near-term inert P3 window the mutation is armed-not-exercised (no real boundaries), so this rides as SHAPE.**

---

## Edit 1 — Destination-by-direction + wire parent/to_realm/effect live (dock vs undock)

**File `crates/sim/src/stub.rs`, `fan_out_crossing` (`:2622-2712`).**

- `:2628` change `_dir: Direction` → `dir: Direction` (consume it).
- `:2636` replace `let to_realm = winner.to_realm;` with a monomorphic destination pick:
```rust
// Dock (Inward) enters winner.to_realm; undock (Outward) re-homes to the parent one level up.
// ONLY the destination realm differs by direction — the wire request is dir-agnostic (HR3: a
// 2-valued Direction branch, never a realm-KIND match). A parentless outward crossing (top-level
// realm, parent==None) has no destination: count it and emit nothing (degrade, never panic).
let to_realm = match dir {
    Direction::Inward => Some(winner.to_realm),
    Direction::Outward => winner.parent, // Option<RealmId>
};
let Some(to_realm) = to_realm else {
    stats.crossing_outward_no_parent += 1;
    return;
};
```
Everything downstream (`:2641`, `:2676`, `:2704`) already uses the local `to_realm` binding — no further change. `from_realm` stays `ctx.config.realm`.

- **`effect` and `parent`** are already read live (`effect` at `:2637` match, `parent` in `boundary_depth`/here). Making them "go live in production" = the composer planting real boundaries — **deferred, out of Slice-4 scope** (INERT-through-P3 holds). Slice 4 makes the *logic* live; tests plant boundaries.

**HR5:** add `crossing_outward_no_parent: u64` to `StubStats` (near `crossing_durable_no_session` ~`stub.rs:714`). The new `match dir` has both arms + the `else` covered by Edit-5/Edit-6 fixtures (inward dock, outward undock-to-parent, outward-at-top-level = the no-parent degrade). Keep `fan_out_crossing` monomorphic (already is) so all three arms count once. No `matches!` — use the `let Some(..) = .. else` straight-line form.

**Three-deep exit = three single-level transfers:** no new code. Player→ship→station→planet is three *separate* outward crossings on three *separate* boundaries (player-in-ship-interior boundary with `parent=Station`; ship-in-station boundary with `parent=Planet`; each fires its own `CrossingRequest` through the ONE saga). The nesting is expressed purely by each boundary's `parent` link; `resolve_winner_ix` (Edit 4) picks innermost-first per tick so exactly one fires per tick. **HR2/HR3 clean — no per-level machinery.**

**Determinism:** `Direction` is a `Copy` enum from `should_commit`; no rng/clock. `winner.parent` read is pure.

---

## Edit 2 — Fence-keyed parent mutation at post-CAS commit (guards dock/undock interleave)

**Goal:** re-parent the *nested* boundary (e.g. the ship-in-station boundary's `parent`) atomically with the authority flip, keyed by the winning fence, on the shard that hosts the boundary. Per R-C this rides `DemoteCmd`.

**File `crates/wire/src/intershard.rs`, `DemoteCmd` (`:519-524`):** add
```rust
    /// The dock/undock re-parent to apply to the crossed subject's nested RealmBoundary, keyed by
    /// new_owner_fence (the winning CAS fence): Some(realm) on a dock/undock commit, None otherwise.
    /// Fence-keyed so a stale/racing re-parent is rejected (the dock/undock interleave guard). INERT
    /// through P3 (always None until real boundaries are planted).
    pub new_parent: Option<RealmId>,
```
Update the constructor at `saga_runtime.rs` `SagaAction::Demote` (~`:914-925`) and the test builders at `intershard.rs:1086/1309` to pass `new_parent: None` (default; the orchestrator has no boundary knowledge in P3). **Add `RealmId` import to `intershard.rs` if absent.**

**File `crates/sim/src/stub.rs`, `on_saga_demote` (`:1597-1634`):** after the latch-clear block (`:1621`), apply the re-parent under the fence:
```rust
// Slice 4: fence-keyed dock/undock re-parent — mutate the nested boundary's parent ONLY here (never
// speculatively at trigger time), keyed by new_owner_fence. Applied to the boundary whose to_realm
// (dock) or realm (undock) matches the crossed subject. Delegated to a monomorphic helper so the
// find/compare branches are covered once (HR5). INERT through P3 (new_parent is always None).
apply_reparent(&mut boundaries.0, cmd.new_parent, entity, cmd.new_owner_fence, stats);
```
Add `boundaries: &mut RealmBoundaries` and `stats` to `on_saga_demote`'s params and its call site (thread `ResMut<RealmBoundaries>` into whatever system drains saga-pushed `Demote`s — same system that owns `RequestInFlight`). New monomorphic helper `apply_reparent(bounds: &mut Vec<RealmBoundary>, new_parent: Option<RealmId>, entity, fence, stats)`:
```rust
fn apply_reparent(bounds: &mut [RealmBoundary], new_parent: Option<RealmId>, entity: EntityId,
                  fence: Fence, stats: &mut StubStats) {
    let Some(np) = new_parent else { return; };            // no re-parent this commit — common path
    // ... find the subject's nested boundary; set .parent = Some(np); stats.reparents_applied += 1;
}
```

**Why the fence guards the interleave:** the re-parent is delivered by the SAME `DemoteCmd` that carries `new_owner_fence` (the winning CAS fence). Two racing docks produce two CAS attempts; only the winner reaches `CasWon`→`Demote`, so only its `new_parent` is ever delivered. A superseded dock never emits a `Demote` (it CasLost→abort→`CrossingAborted`, which clears the latch, not re-parents). So the parent is written exactly once per successful commit, keyed by the fence that won — **never speculatively at trigger time** (`fan_out_crossing` writes NO parent; it only reads).

**HR5:** `new_parent` is `Option` — the `None` early-return is the hot P3 path (covered by every existing demote test that now passes `None`); the `Some` arm covered by Edit-5's dock E2E. `apply_reparent` monomorphic, straight-line `let Some(..) else`. **Determinism:** fence is integer, no clock/rng; the find is over a `Vec` in deterministic order.

**Flag (plan vs reality):** the plan says "mutated at the saga post-CAS commit *point*." Reality: the CAS commit point is on the *orchestrator* (`saga_runtime.rs:950`), which does not host the boundary. The mutation therefore *originates* fence-keyed at commit but *applies* at the ship-shard's `on_saga_demote` — the post-commit terminal that ALREADY carries the fence to that shard. Ledger this in DEFERRED as "re-parent origin=commit, apply=demote-terminal, same fence."

---

## Edit 3 — Area/Interest: EffectFree Tier-1, dispatched by CrossEffect (not realm-kind)

**Already 95% built** (`fan_out_crossing` `:2638-2643` Interest arm writes `InterestZones`, no saga/CAS/directory). Slice-4 work:

- **No dispatch change needed** — `evaluate_realm_boundaries`→`fan_out_crossing` already branches `match (winner.effect, durability_of(entity))` with `(CrossEffect::Interest, _)` as the first arm. This is the `CrossEffect` branch the plan wants, and it is **not** a realm-kind match (it keys on the boundary's `effect` DATA field). **HR3 clean.**
- **Confirm no directory record:** the Interest arm emits zero flow, so `transfer_subject_entity`/`DirectoryKey::Realm(Area)` are never touched — matches "Area is never a DirectoryKey" (Tier-1). Nothing to change.
- **Add the Area Tier-1 assertion to Edit-5's harness:** an `Area(seed)` boundary with `effect: CrossEffect::Interest` fires `InterestZones.insert(entity, Area(seed))` and emits **no** `CrossingRequest`/`TransientCrossingRequest` and arms **no** cooldown (`last_commit_tick` stays `None`). Assert `stats.crossings_requested == 0 && stats.crossing_interest_updates == 1`.

**DEFERRED ledger (Tier-2 @P6):** "an Area promoted to Authority flips `effect: Interest→Authority` + adds `DirectoryKey::Realm(Area)` — ADDITIVE, no new type; a pre-P6 signal-endpoint Area promotes early." Verbatim from plan line 13.

**HR5:** Interest arm already covered by existing 3e tests; Edit-5 adds the Area-named assertion. **Determinism:** `insert` into `BTreeMap`, idempotent, no clock.

---

## Edit 4 — Multi-boundary tie-break (MUST-FIX #6): ALREADY IMPLEMENTED — add the nesting proof

**Reality:** `resolve_winner_ix` + `candidate_beats_ix` (`geometry.rs:564-602`) already implement the strict total order (depth desc → RealmId asc → Direction asc → index asc) and are already permutation-invariant-proptested (`geometry.rs:1593-1652`). `boundary_depth` (`stub.rs:2597-2612`) already computes parent-chain depth. **The primitive is a Slice-1 deliverable already landed.** No `geometry.rs` change.

**Slice-4 owed:** the *nesting scenario* that exercises ≥2 boundaries — the docked-ship-in-station-in-planet case — proving exactly ONE re-home per tick. This is folded into Edit-5's E2E: plant three nested boundaries (player-in-ship `depth=2`, ship-in-station `depth=1`, station-in-planet `depth=0`), put one entity where two boundaries are simultaneously candidates, assert `resolve_winner_ix` picks the innermost (deepest) and exactly one `CrossingRequest` fires that tick. No new primitive; new *fixture*.

**HR5:** covered by Edit-5 fixture. **Determinism:** the index tiebreak makes it slice-order-independent (already proptested).

---

## Edit 5 — D-38 discharge: `assert_feature_anywhere` (identical crossing on Shell AND Aabb)

**New file `crates/sim/tests/assert_feature_anywhere.rs`.** This is the *first* genuine G-IDENTICAL feature test.

**The shared fixture (ONE closure, run over two `RealmBoundary` shapes):**
```rust
// The IDENTICAL crossing feature: one entity descends into a realm, dwells past n_entry, commits
// exactly one INWARD crossing, then ascends and commits exactly one OUTWARD. Parameterized ONLY by
// the boundary constructor — the shape/band differ; the assertions do NOT. This is genuinely
// assert_feature_anywhere (not two copies): the SAME drive_and_assert body runs both profiles.
fn drive_and_assert(make_boundary: impl Fn() -> RealmBoundary, profile: ShardProfile) { ... }

#[test]
fn assert_feature_anywhere() {
    // Spherical planet — Shell boundary + SOI velocity-safe band.
    drive_and_assert(
        || RealmBoundary::shell(RealmId::Planet(1), center, r_soi, cf, df, v, dt, pad, k, None, RealmId::Planet(1), CrossEffect::Authority),
        profiles::planet(),   // Spherical
    );
    // Cartesian station — Aabb boundary + normalized box band.
    drive_and_assert(
        || RealmBoundary::boxed(RealmId::Station(1), center, half, None, cf, df, None, RealmId::Station(1), CrossEffect::Authority).expect("valid box band"),
        profiles::station(),  // Cartesian, hull_host+signal_relay
    );
}
```

**Shared assertions (identical for both, per geometry-profiles MAP (d)):**
- After a full descend→dwell→ascend trajectory: exactly **one** inward `CrossingRequest` emitted, exactly **one** outward. Assert on `StubStats.crossings_requested == 2` (one dock + one undock) OR on the collected `OutboundBox` flows filtered to `CrossingRequest`, asserting `len()==2` and the inward `to_realm==Station/Planet(1)`, the outward `to_realm==parent` (None here → the top-level degrade, so for the E2E use a *nested* boundary with a real parent).
- No flap: `should_commit` fires inward once, outward once — asserted via the stat delta per phase, not `matches!`.

**Why it is genuinely `assert_feature_anywhere` and not two copies:** the `drive_and_assert` body — the tick loop, the pose trajectory, and every `assert_eq!` — is written **once**. The only per-profile input is the `make_boundary` closure and the `ShardProfile` (selected by capability DATA via `profiles::planet()`/`profiles::station()`, never a shard-kind `match`). The shape-agnostic tail (`should_commit`→`resolve_winner_ix`→`Direction`→`fan_out_crossing`) produces byte-identical `CrossingRequest`s modulo the destination realm. Per geometry-profiles MAP (d), `should_commit` takes only `(ShellCrossing, bool, u32, since_commit, tuning)` and `resolve_winner_ix` only `(depth, RealmId, Direction, ix)` — neither sees geometry, so the feature is provably shape-independent.

**The `reanchor()` requirement (D-38 doc, capability.rs:14):** `FrameSpace`/`reanchor()` are P4/P5 (SphericalSpace/CartesianSpace don't exist yet). **Flag: cannot force a real `reanchor()` in Slice 4.** Ledger in DEFERRED: "D-38 discharged for the crossing feature over Shell+Aabb; the `reanchor()`-forcing variant is re-owed at P5 when `FrameSpace` lands (block-edit fixture)." This partially discharges D-38 — flip its ledger entry to note "crossing half GREEN; reanchor half owed P5."

**HR5:** the box path currently has NO full-trajectory `should_commit` test (geometry-profiles MAP (d) caveat) — this test is the first to drive the Aabb through a trajectory, closing that gap. Use `.expect("valid box band")` straight-line (not `?`) in a test. Assertions are `assert_eq!` on stat counts, not `matches!`. **Determinism:** virtual clock via the harness `ClockSample`, integer ticks, no sockets.

---

## Edit 6 — Non-zero-cell FRAME-CHANGE fixture (MUST-FIX #5/#7): the DECISION must FLIP

**New test in `crates/sim/tests/` (e.g. `frame_change_nonzero_cell.rs`).** The non-vacuity condition (deferred-d38 MAP (b)): boundary `center` AND entity poses at **DISTINCT non-zero `LatticePos.cell`s** so the integer in/out gate *toggles*, not merely subtracts.

- Construct `RealmBoundary::boxed(..., center = LatticePos{ cell: <non-zero A>, offset: … }, …)`.
- Drive the entity from a pose at `cell=<non-zero B>` (B≠A, both ≠0) across the boundary so the swept classification flips `Outside→Inside` — i.e. the coarse cell comparison in the offset/`LatticePos` math changes value across the crossing, not just that a subtraction runs.
- Assert the inward `CrossingRequest` fires (`crossings_requested` increments) — proving the DECISION path (cell-comparison flip), not merely a line-covered subtraction.

**The armed-not-exercised ledger:** the live normalize/cross-cell rebase MATH is owed P4/P5 (`pose.rs:96-99`). Per plan line 83, mark in `DEFERRED.md`: "multi-level-exit rebase ARMED-NOT-EXERCISED through P3 (cell subtraction runs, live rebase math is a P4/P5 dependency); the non-zero-cell FRAME-CHANGE fixture proves the in/out DECISION flips so the docking E2E does not pass vacuously."

**Why this closes MUST-FIX #7's coverage-trap sibling:** a fixture with non-zero center but zero-cell poses would line-cover the subtraction while the integer gate stays constant (behaviour-inert). Distinct non-zero cells force the branch to be *exercised*. Assert with `assert_eq!` on the resulting `CrossingRequest` presence — the false arm (no flip → no request) must ALSO be asserted in a sibling case (poses that stay same-cell → no crossing) so both sides of the DECISION branch are covered (HR5: split, don't `assert!(a && b)`).

**HR5:** two named tests — flip-case (crossing fires) and no-flip-case (no crossing) — cover both branch arms. `.expect()` on the `boxed(...)` Result, straight-line. **Determinism:** integer `LatticePos.cell` values, virtual clock.

---

## Edit 7 — Ledger-eviction-on-leave test (the machinery already exists — Edit R-B)

**New test in `crates/sim/tests/` (e.g. `ledger_eviction_on_leave.rs`).** `retain_live` (`stub.rs:2399-2401`) already evicts `CrossingProgress`/`RequestInFlight`/`InterestZones` for non-resident subjects. This test **proves** it (the gate is a test, not machinery).

- Plant boundaries, drive one entity so all three ledgers get a row: `CrossingProgress` (via any tick — dwell state), `RequestInFlight` (via an inward durable commit — latch), `InterestZones` (via an Interest-effect boundary crossing).
- Assert each map `.contains_key(&entity)`.
- Remove the entity (despawn the dot / release the transient — remove from `Dots`/`OwnedTransients`) and step `evaluate_realm_boundaries` once.
- Assert `progress.0.get(&entity).is_none() && in_flight.0.get(&entity).is_none() && interest.0.get(&entity).is_none()` — three separate `assert_eq!`/`is_none()` asserts (split, HR5).
- **Nesting eviction:** also assert that a re-parented boundary's `parent` is untouched by entity removal (the boundary is realm state, not per-entity — it must NOT be evicted with the entity). This distinguishes per-entity ledgers (evicted) from realm descriptors (persistent).

**HR5:** three `is_none()` asserts, no `matches!`, no `&&`-collapsed assert. **Determinism:** `retain_live` iterates the `live` `BTreeSet` deterministically.

---

## Ordered edit list for a single implementer

1. **`intershard.rs`**: add `new_parent: Option<RealmId>` to `DemoteCmd` (`:519`); add `RealmId` import if needed; fix the two test builders (`:1086`,`:1309`) + the `saga_runtime.rs` Demote constructor to pass `None`. Rebuild wire.
2. **`stub.rs` `StubStats`**: add `crossing_outward_no_parent: u64` and `reparents_applied: u64` (near `:714`).
3. **`stub.rs` `fan_out_crossing`** (`:2628`,`:2636`): consume `dir`; destination-by-direction (Edit 1). Inward→`to_realm`, Outward→`parent`, `None`→count+return.
4. **`stub.rs` `on_saga_demote`** (`:1597`): thread `boundaries: &mut RealmBoundaries` + `stats`; call new `apply_reparent` after latch-clear (Edit 2). Add `apply_reparent` monomorphic helper. Thread `ResMut<RealmBoundaries>` into the demote-draining system.
5. **`crates/sim/tests/assert_feature_anywhere.rs`** (Edit 5, D-38): shared `drive_and_assert` over `shell(...)`+`boxed(...)`, `profiles::planet()`+`profiles::station()`. Fold the nesting/tie-break scenario (Edit 4) as a second case with three nested boundaries → exactly one re-home/tick.
6. **`crates/sim/tests/frame_change_nonzero_cell.rs`** (Edit 6): flip-case + no-flip-case at DISTINCT non-zero cells.
7. **`crates/sim/tests/ledger_eviction_on_leave.rs`** (Edit 7): three ledgers populated → entity removed → all three evicted; boundary `parent` untouched.
8. **`DEFERRED.md`**: (a) D-38 — crossing half GREEN over Shell+Aabb, `reanchor()` half re-owed P5; (b) D-41 — multi-level-exit rebase ARMED-NOT-EXERCISED, live math P4/P5; (c) Area Tier-2 (Authority+`DirectoryKey::Realm(Area)`) ADDITIVE @P6; (d) parent-mutation origin=commit/apply=demote-terminal note.
9. **`just gate`** (fmt + clippy `-D` + tests + `coverage-fast`): verify `crossing_outward_no_parent` no-parent arm, both `match dir` arms, `apply_reparent` `Some`/`None`, and the box full-trajectory path are all covered; run `cargo llvm-cov clean --workspace` first (stale-profdata gotcha).

## Cross-cutting guarantees
- **HR1:** the only new cross-shard byte is the `new_parent` field on the existing `DemoteCmd` arm — no new `InterShardFlow` variant. **HR2:** dock/undock/three-deep-exit all ride the ONE saga via ordinary `CrossingRequest`s; parent is policy data, not new machinery. **HR3:** every new branch keys on `Direction` (2-valued) or `CrossEffect` (2-valued) DATA, never a `RealmId`-kind `match`. **HR4:** Edit 5 IS the discharge. **HR5:** every new branch has both arms fixture-covered; helpers monomorphic + branchless-shim; `.expect()` straight-line; `assert_eq!`/`is_none()` over `matches!`; `&&` split. **Determinism:** integer fences/cells, `BTreeMap`/`BTreeSet`, virtual clock, no rng/sockets.
- **INERT-through-P3 preserved:** `new_parent` is always `None` until real boundaries are planted; `RealmBoundaries` defaults empty; `apply_reparent` `None`-early-returns. Production behaviour byte-identical; all new logic exercised only by tests.

---

## ADVERSARY (NO-GO)

I now have complete and decisive ground truth. The `process_inbound` system is already at Bevy's 16-param ceiling and has bundled resources into tuples specifically to stay under it. This is the exact system that must gain `RealmBoundaries` for Edit 2. I have everything needed for a rigorous verdict.

---

# HOSTILE REVIEW — #133 Slice-4 design

I verified every load-bearing claim against the actual code. The maps are largely accurate, but the design has **one CRITICAL correctness bug**, **two HIGH structural defects**, and several MEDIUM/LOW issues. Findings below, each with the exact failure scenario and a concrete fix.

---

## CRITICAL-1 — The Outward crossing goes through the durable latch and re-drive machinery, which Edit 1 breaks. Undock will strand or mis-home.

This is the design's central hole, and it is invisible in the maps because the maps never traced the Outward path through the latch.

**Ground truth.** `should_commit` (geometry.rs:638-644) returns `Some(Direction::Outward)` for a leaving entity. `evaluate_one_subject` (stub.rs:2527-2547) feeds that `dir` straight into `fan_out_crossing`, which — for an `Authority + Durable` subject — takes the `in_flight.entry(entity)` latch path **regardless of direction** (stub.rs:2660). So an **outward durable crossing latches into `RequestInFlight`, arms `last_commit_tick`, and stores a `LatchedCrossing`** (stub.rs:2669-2689). It is then re-driven by `redrive_stranded_crossings` (stub.rs:2751-2787) using `lc.to_realm` **captured at emit time**.

Now apply Edit 1. Edit 1 makes `to_realm = winner.parent` for the Outward branch. The captured `LatchedCrossing.to_realm` becomes the parent. Fine so far. But two things break:

1. **Undock and dock share ONE latch keyed by `entity`, not by direction.** `in_flight` is `BTreeMap<EntityId, TransferId>` (stub.rs:2630). A dock (Inward) leaves a latch until `on_saga_demote` clears it (stub.rs:1619). If the entity then immediately turns around and crosses **Outward** before the dock's Demote arrives (a real interleave — the design's own dock/undock race), `fan_out_crossing`'s `Entry::Occupied` arm **suppresses the undock** (stub.rs:2692 `crossings_suppressed_in_flight`). The entity is now committed inward to the station in the directory but geometrically outside it, **with no outward crossing ever emitted**. It is stranded in the parent realm's space while owned by the child realm. Edit 1 does nothing about this; it changes only the destination, not the latch semantics.

2. **The re-drive re-emits an Outward request as if it were a fresh crossing.** `redrive_stranded_crossings` re-mints `CrossingRequest { to_realm: lc.to_realm, ... }` (stub.rs:2772-2781). With Edit 1's parent-destination for outward, a stranded undock latch will be re-driven to the parent forever with no direction discriminator on the wire — and because `CrossingRequest` carries no direction (map (c) confirms), the orchestrator cannot tell dock-redrive from undock-redrive. If the parent realm is transiently absent (the exact stranded case the re-drive exists for), the undock re-drives, but the *dock* of a different entity to the same realm is byte-distinguished only by subject — acceptable — yet the **undock's `from_realm` is still `ctx.config.realm`** (the child), which for an outward crossing is semantically "I am leaving realm X to go to parent" but the wire shape is identical to "I am entering realm parent." The orchestrator's saga has no way to know the source authority is the child realm being vacated vs. a peer. The design explicitly claims (R-A) "a `(from,to)`-tuple-only difference at the wire level" makes this safe — **it does not**, because the latch/re-drive/abort machinery is direction-blind and the design never audited it.

**Severity: CRITICAL.** Undock is the headline Slice-4 deliverable and the interleave guard is the headline correctness claim. The design's fence-keying argument (Edit 2) is about the *parent-mutation* race and is sound for that; but it never addresses that **the crossing REQUEST itself is single-latched per entity across both directions**, so a dock-then-undock interleave suppresses the undock silently.

**Concrete fix.** The latch must be keyed so an inward and an outward crossing of the same entity are distinct in-flight operations, OR the Outward path must be explicitly excluded from the durable latch and given its own emit/clear discipline. Minimal correct option: change `RequestInFlight` to `BTreeMap<EntityId, (Direction, TransferId)>` (or a two-map split), make the `Occupied` suppression check `(entity, dir)`, and thread `dir` into `LatchedCrossing` so the re-drive and the `on_saga_demote` clear match on direction. This also forces `crossing_transfer_id` to include direction so a dock and an undock of one entity produce distinct transfer ids (today two crossings would collide on `(subject, fence, attempt)`). **This is real machinery the plan omits entirely** and cannot be waved away as "INERT through P3": the moment a real boundary is planted (the Slice-4 tests themselves plant one and drive descend→dwell→ascend), both directions latch on the same entity.

---

## HIGH-1 — Edit 2 threads `RealmBoundaries` into `process_inbound`, which is already at Bevy's 16-param ceiling. The design's "thread `ResMut<RealmBoundaries>` into the demote-draining system" is not a mechanical add.

**Ground truth.** `on_saga_demote` is called from the `MsgClass::Saga` arm inside `process_inbound` (stub.rs:3329-3331). `process_inbound` (stub.rs:1027-1049) **already bundles resources into tuples specifically because "bevy caps a system at 16 top-level params"** (stub.rs:1040-1041, 1045-1047) — Slice 3e's `RequestInFlight` and 3f-D's `CrossingProgress` were tuple-packed for exactly this reason. Edit 2 says "thread `ResMut<RealmBoundaries>` into whatever system drains saga-pushed `Demote`s" as if trivial. It is the most param-pressured system in the crate, and `RealmBoundaries` is currently only a `Res` in `evaluate_realm_boundaries`.

Worse: `evaluate_realm_boundaries` holds `Res<RealmBoundaries>` (read-only, stub.rs:2375) and `process_inbound` would now hold `ResMut<RealmBoundaries>`. If these two systems are unordered in the schedule, Bevy will serialize them (a `ResMut` conflicts with a `Res`), silently removing any parallelism between inbound processing and boundary evaluation — a **determinism-neutral but scale-relevant** coupling the design claims doesn't exist ("HR1: the only new cross-shard byte…"). At minimum it needs an explicit ordering edge and a note; the design provides neither.

**Severity: HIGH** (implementability + undisclosed coupling; not a correctness bug but the plan understates the work and the schedule impact). **Fix:** bundle `RealmBoundaries` into the existing `crossing` tuple param (making it `(ResMut<RequestInFlight>, ResMut<CrossingProgress>, ResMut<RealmBoundaries>)`), and explicitly document/enforce the ordering vs `evaluate_realm_boundaries` (or make the mutation happen in a dedicated post-`process_inbound` system that reads a queued re-parent command, avoiding the `ResMut`-in-inbound entirely — cleaner).

---

## HIGH-2 — `apply_reparent`'s "find the subject's nested boundary" has no defined key, and the map proves there is none. The parent mutation cannot be correctly targeted.

**Ground truth.** Edit 2's helper is `apply_reparent(bounds, new_parent, entity, fence, stats)` with the body `// ... find the subject's nested boundary; set .parent = Some(np)`. But a `RealmBoundary` is keyed by `realm: RealmId` and `to_realm: RealmId` — **there is no `EntityId` on a boundary** (geometry.rs:381-397, confirmed by boundary-trigger map (a)). The docked *ship* is `Ship(EntityId)` as a realm identity (plan-slice4: "a docked ship stays Ship(EntityId); containment is RealmBoundary.parent"), so the boundary to re-parent is the one whose `realm == Ship(entity)` — but `on_saga_demote` receives `cmd.subject` = the crossed subject's `DirectoryKey`, and `transfer_subject_entity()` yields the *crossing subject's* `EntityId`, which for a ship-dock **is the ship**, so `realm == Ship(entity)` is findable — but for a **player walking out of the ship** (the three-deep leg), the subject is the *player* entity, and the boundary to re-parent is **not** keyed by the player at all; there is nothing to re-parent (the player has no boundary). The design's helper signature (`entity`) implies a per-entity lookup that is meaningful for exactly one of the two Slice-4 crossing kinds and ill-defined for the other.

Concretely: on the "player walks out → re-homes to Station parent" leg (the E2E's own line 53), what boundary does `apply_reparent` mutate? None. But `new_parent` would have to be `None` for that Demote, and the design says `new_parent` is "always None through P3" anyway — so **the re-parent is never actually exercised even by the Slice-4 tests**, making Edit 2's `Some` arm uncoverable in Slice 4 and violating HR5 unless a test synthesizes a `Demote` with `new_parent: Some(...)` by hand (which then tests the mechanism against a boundary that no production path would target). The design admits "INERT through P3, armed-not-exercised" but simultaneously claims (Edit 2 HR5) "the `Some` arm covered by Edit-5's dock E2E" — **contradiction**: if `new_parent` is always `None` in P3, Edit-5's E2E cannot cover the `Some` arm.

**Severity: HIGH** (the mutation's target is underspecified AND the HR5 coverage claim is self-contradictory). **Fix:** decide explicitly — either (a) the re-parent targets the boundary whose `realm == RealmId::Ship(entity)` / `RealmId::Station(entity)` (make the key rule explicit and assert the subject is a container-realm entity), and cover the `Some` arm with a **dedicated unit test** that hand-builds a `DemoteCmd { new_parent: Some(..) }` plus a matching boundary (not the E2E), OR (b) drop the `Some` arm from Slice 4 entirely, ship `apply_reparent` as a `None`-only early-return stub with `#[cfg_attr(coverage_nightly, coverage(off))]` on the unreachable `Some` body, and ledger the whole re-parent as owed-at-P6 when real boundaries and the target-key rule land. Option (b) is more honest with "INERT through P3."

---

## MEDIUM-1 — The "no-parent outward degrade" silently drops a top-level exit, but the entity may have already had its dwell/latch state mutated. Possible stuck state.

Edit 1's `Direction::Outward` with `winner.parent == None` does `stats.crossing_outward_no_parent += 1; return;` **inside `fan_out_crossing`**. But by the time `fan_out_crossing` is called, `evaluate_one_subject` has already advanced `state.inward_ticks`, `state.was_member`, and (on the next line after the call) will write `state.was_member = now_member` (stub.rs:2550). The `last_commit_tick` cooldown is **not** armed on this degrade path (good), so the entity will re-attempt the outward commit every tick it stays outside — `should_commit` returns `Some(Outward)` on every tick with `Outward`/`ThroughAndBack` swept and no cooldown (geometry.rs:639, no cooldown gate passed for outward since `since_commit` is `None`). That means `crossing_outward_no_parent` **increments every tick forever** for any entity loitering just outside a top-level realm boundary. Not a crash, but an unbounded stat and a per-tick wasted evaluation — and it masks the real question: what *should* happen when you exit a top-level realm? (Answer: you enter the enclosing System/Galaxy realm — there is always a parent in the real hierarchy; a `None` parent is a *composer bug*, not a runtime-degrade case.) Treating it as a silent per-tick degrade hides an authoring error.

**Severity: MEDIUM.** **Fix:** make a `None` parent on an Outward-capable boundary a **loud one-time** condition (warn-once or a distinct "malformed boundary" counter that a test asserts stays 0 in valid fixtures), not a per-tick silent increment. Better: assert at boundary-plant time that any boundary reachable Outward has a parent, so the runtime path is unreachable and can be coverage-exempted.

---

## MEDIUM-2 — The D-38 "assert_feature_anywhere" fixture is at real risk of being VACUOUSLY identical, and the design's own text concedes the weak point.

The maps (geometry-profiles (d)) prove the *primitives* return the same types for Shell and Aabb, and that `should_commit`/`resolve_winner` are geometry-blind. That is exactly why the "identical assertions" can be **too weak to prove anything**: if the assertion is only `stats.crossings_requested == 2`, then a fixture that never actually enters the box (e.g. a normalized-Chebyshev band mis-sized so `now_member` never rises) would give `0 == 0`-style false confidence, or a Shell fixture and an Aabb fixture that both trivially cross would pass without the box geometry ever exercising `segment_aabb_crossing`'s slab test. Geometry-profiles map explicitly flags: **"The box equivalent of this full-trajectory commit test is NOT yet written… the box side is covered at the primitive level… but not yet driven through `should_commit` over a trajectory."** So the box trajectory is genuinely new surface, and the D-38 test is the *first* thing exercising it — which means a bug in the box band sizing would make the test pass vacuously (both sides report "one inward, one outward" only because neither side's box actually gates).

Additionally, the design's E2E-vs-anywhere conflation is a trap: Edit 5's `assert_feature_anywhere` uses `to_realm==None → top-level degrade` for the outward case (its own note: "for the E2E use a *nested* boundary with a real parent"). So the D-38 fixture as written asserts the outward crossing produces the **degrade** (no request), while claiming "exactly one outward `CrossingRequest`." **These cannot both be true.** The design mixes the top-level (degrade, no request) and nested (request to parent) cases in one fixture description.

**Severity: MEDIUM** (vacuity risk + internal inconsistency in the fixture spec). **Fix:** the D-38 fixture must (a) assert the box actually entered — e.g. assert the membership scalar crossed the create edge, or assert the inward `CrossingRequest.to_realm` is the box's `to_realm` — not merely a count; (b) use a **nested** boundary (real parent) for both Shell and Aabb so the outward case emits a real request to the parent, not the degrade; (c) assert the emitted request destinations differ appropriately between profiles so the two paths are provably distinct runs, not the same code path twice. And it must drive the Aabb through the *slab-test* corner case at least once, or the box's anti-tunneling path stays unexercised.

---

## MEDIUM-3 — The non-zero-cell fixture's non-vacuity depends on `should_commit` reading `LatticePos.cell`, but the whole trigger currently reads only `.offset()` (the f64 fine coordinate). The "integer in/out DECISION" the fixture must flip does not exist in the code path yet.

**Ground truth.** `evaluate_one_subject` computes `cur = dot.pose.pos.offset()` (stub.rs:2413) — the **f64 offset within the cell**, and every downstream call (`swept`, `membership_scalar`, `should_commit`) operates on `DVec3` offsets. The coarse `LatticePos.cell` is **never compared** in the crossing decision today — the D-41 memory confirms "cell is ALWAYS ZERO through P3, the normalize/cross-cell math is owed P4/P5." So the design's Edit 6 requires the fixture to make "the integer in/out DECISION actually flip" — **but there is no integer in/out decision in the current trigger**; `should_commit` sees only f64 offsets. If the boundary center is at `cell=A` and the entity at `cell=B≠A`, then `winner.center.offset()` (stub.rs:2499) and `dot.pose.pos.offset()` return the **within-cell offsets ignoring the cell entirely** — so `prev - center` and `cur - center` are computed as if both were in cell 0, and a distinct non-zero cell **changes nothing** about the crossing classification. The fixture as specified cannot flip any integer gate because the gate is not wired; it would pass or fail based purely on the f64 offsets, exactly the vacuity MUST-FIX #7 warns against, just relocated.

The design half-acknowledges this ("the live rebase MATH is a P4/P5 dependency") but then claims the fixture "proves the in/out DECISION flips." **It cannot prove that today** — there is no cell-comparison branch to exercise. The honest statement is: the fixture can only assert the *offset* math produces a crossing when poses at distinct cells are **first rebased to a common frame** — and that rebase (`map_offset` / cross-cell normalize) is precisely the P4/P5 gap. So Edit 6 either tests nothing new (offsets only) or tests a rebase that doesn't exist.

**Severity: MEDIUM** (the fixture's stated non-vacuity guarantee is unachievable against the current code; it risks being the very vacuous plant it's meant to prevent). **Fix:** be explicit that Slice 4 can only plant a **SHAPE** assertion — e.g. assert that `pose.pos.cell` values are carried/preserved through the trigger unchanged and that `offset()`-based crossing still fires — and ledger that the "integer decision flips across a cell boundary" test is **BLOCKED on the P4/P5 rebase** and cannot be written now. Do not claim it "proves the decision flips." Alternatively, if `should_commit` is to gain a cell-aware gate in Slice 4, that is new production logic the plan hasn't scoped and must be designed first.

---

## LOW-1 — HR5: `let Some(to_realm) = to_realm else { … return; }` inside the generic-adjacent `fan_out_crossing` adds a branch that must be covered per the (Interest, Authority-Durable, Authority-Transient) fan-out — but the no-parent return sits *before* the `match`, so it short-circuits all three arms.

Edit 1 places the `let Some(to_realm) = … else { return }` at the top of `fan_out_crossing`, before the `match (winner.effect, durability_of(entity))`. For an **Interest** boundary crossed Outward at top level, this now `return`s before the interest insert — so an outward interest-zone drop at a top-level Area silently does nothing and increments `crossing_outward_no_parent` instead of dropping the interest. That is a behavior change to the Interest path the design never mentions (Interest is supposed to be direction-agnostic per boundary-trigger map (e): "dispatched by `winner.effect == Interest` for ANY durability"). The no-parent degrade should apply **only to the Authority arms**, not Interest.

**Severity: LOW→MEDIUM** (a genuine unintended behavior change to the Area/Interest path). **Fix:** compute the outward-destination-or-degrade **inside** the two Authority arms, not before the `match`; leave Interest reading `winner.to_realm` directly (an interest zone's "entered realm" for an outward crossing is arguably the parent too, but that is a design decision to make explicitly, not a side effect of an early return).

---

## Correctly-handled claims (credit where due)

- **`resolve_winner_ix` permutation-invariance and the index tiebreak** are real, proptested (geometry.rs:589-602, 1593-1652), and the design correctly declines to touch them. No determinism hole there.
- **The `retain_live` ledger eviction** is genuinely already implemented (stub.rs:2399-2401) over a `BTreeSet` (deterministic), covering all three maps. Edit 7's "test-only" framing is correct.
- **The `match (effect, durability)` fan-out keys on `CrossEffect`/`DurabilityClass` DATA, not `RealmId` kind** (stub.rs:2637) — no HR3 violation there; the design's HR3 claim for the *existing* dispatch holds.
- **`Direction` and `CrossEffect` are 2-valued enums**, so branching on them is not a shard-kind match — the design's HR3 defense for the *new* `match dir` is valid **in isolation** (the problem is CRITICAL-1's latch, not an HR3 violation).
- **The parent-mutation-rides-`DemoteCmd` seam choice** is the right instinct (it's the only fence-bearing flow already routed to the boundary-hosting shard) — the flaw is the target-key (HIGH-2), not the seam.

---

## VERDICT: **NO-GO**

The design cannot be implemented as written. **CRITICAL-1** is disqualifying on its own: the durable crossing latch (`RequestInFlight`) and its re-drive/abort machinery are keyed by `entity` and are **direction-blind**, so a dock→undock interleave — the exact race the design claims to guard — silently suppresses the undock and strands the entity, and Edit 1's destination-swap does nothing to fix it. The design's entire "fence-keying is airtight" argument (Edit 2) addresses only the *parent-mutation* race, not the *crossing-request* race, and never audited the Outward path through the latch.

Additionally: **HIGH-2** exposes a self-contradiction (the `Some` arm is claimed both "always None in P3" and "covered by the dock E2E"), **HIGH-1** understates the `process_inbound` param-ceiling and schedule coupling, and **MEDIUM-3** shows the marquee anti-vacuity fixture (Edit 6) targets an integer gate that does not exist in the current code path, so it cannot deliver its promised guarantee.

**Required before GO:**
1. Redesign the durable crossing latch to distinguish inward/outward per entity (key on `(entity, direction)` or split maps; thread `dir` into `LatchedCrossing`, the re-drive, `on_saga_demote` clear, and `crossing_transfer_id`). This is real new machinery, not INERT shape.
2. Resolve the Edit-2 target-key rule for `apply_reparent` explicitly, and reconcile the `Some`-arm coverage claim honestly (dedicated unit test, or coverage-exempt the unreachable-in-P3 body).
3. Move the no-parent degrade inside the Authority arms only (LOW-1) and make a `None` outward parent loud/authoring-checked, not a per-tick silent increment (MEDIUM-1).
4. Re-spec the D-38 fixture to assert the box actually gated (not a bare count) and to use a nested parent for the outward leg — the current spec is internally inconsistent (MEDIUM-2).
5. Downgrade Edit 6's claim: it can only plant a cell-carry SHAPE assertion in Slice 4; the "integer decision flips" test is blocked on the P4/P5 rebase and must be ledgered as such, not claimed as delivered.

Files that must change beyond the design's list: `crates/sim/src/stub.rs` — `LatchedCrossing` (line 348), `RequestInFlight` type, `redrive_stranded_crossings` (2751-2787), `on_crossing_aborted` (1682-1710), and `crossing_transfer_id` — none of which the design touches, all of which CRITICAL-1 forces.