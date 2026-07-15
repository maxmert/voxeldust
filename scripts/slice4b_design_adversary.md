# Slice 4b design + adversary (wf_05833a15)

## SYNTHESIZED DESIGN

Critical adjudication point confirmed: `on_saga_demote` does **NOT** take `CrossingProgress` — it only has `dots`, `in_flight`, `stats`, `outbox`. This validates the scale-first design's "option (a) lazy-clear" — clearing the per-direction `latched_crossing` payload inside `on_saga_demote` would require threading `CrossingProgress` into it. And `redrive` keys off `in_flight` membership, so a committed (removed-from-`in_flight`) direction is never visited, making the stale payload harmless. This is the honest resolution. I have everything needed. Producing the final deliverables.

---

# Slice-4b Design Adjudication & Synthesis

## (1) SCORE TABLE

The three lenses are **the same core design** (all three converge on: `dir` into the id seed + `CrossingRequest`; `(EntityId, Direction)` latch key; per-direction `latched_crossing` + `crossing_attempt`; reparent rides `DemoteCmd`; dedicated drain/system for HIGH-1). They differ in *which sub-decisions they get exactly right*. Scored on the seven axes:

| Axis | minimal-diff | scale-first | endgoal-composition | One-line justification |
|---|---|---|---|---|
| **Direction-keyed latch correctness** | 9 | 10 | 9 | scale-first is the only one that fully proves the redrive-keys-off-`in_flight` invariant makes the lazy payload-clear sound (no `CrossingProgress` in `on_saga_demote` — I confirmed this at stub.rs:1597). |
| **HR5-coverability** | 8 | 9 | 10 | endgoal nails the *honest* `apply_reparent` coverage story (dedicated unit test on the total helper, all 4 arms, mirroring the proven `emit_crossing`/`build_rehome` pattern at saga_runtime.rs:5988); minimal-diff's D2 `if let Some` body is production-unreachable AND lacks the explicit 4-arm unit plan. |
| **Determinism** | 9 | 10 | 9 | all three preserve `(EntityId, Direction): Ord` + `[_;2]` `Copy+Default`; scale-first most explicitly re-derives every guard (Vec drain push-order, `find`-first-match stability). |
| **Scale to hundreds** | 6 | 10 | 7 | only scale-first quantifies "≤2 latch entries/entity, no O(n·m)", locates the real wall (orchestrator saga fan-out, D-32), and pre-plans the Slice-6 rayon seam at the two subject loops. |
| **End-goal / HR2 / HR3** | 8 | 8 | 10 | endgoal is the only one that audits the parent-chain-as-single-truth for BOTH trigger and render, the cross-shard-signal non-contamination (Interest arm emits zero flow), and the station-in-orbit `hull_host` reuse — and flags that a latch-strand is simultaneously an authority AND render bug. |
| **DRY / elegance** | 9 | 8 | 8 | minimal-diff's `retain_live` split is the cleanest statement; all three correctly refuse to overload `crossing_transfer_id` (sentinel `Inward` for transients) and keep `dir_ix` a branchless total map. |
| **Honesty of reparent coverage** | 6 | 9 | 10 | minimal-diff internally contradicts (claims `Some` arm "INERT through P3" AND coverable — the exact HIGH-2 trap); scale-first and endgoal both resolve it: `Some` arm covered ONLY by a dedicated hand-built unit test, NEVER the E2E, with the `None`-early-return as the P3 path. |
| **Weighted total** | **7.9** | **9.1** | **8.9** | scale-first wins on rigor+scale; endgoal wins on end-goal/coverage-honesty; minimal-diff is the cleanest diff statement but has the one disqualifying HIGH-2 contradiction. |

**Synthesis principle:** take scale-first's structural spine + invariant proofs, endgoal's honest coverage + end-goal audit + the four map/plan disagreement resolutions, and minimal-diff's cleanest DRY statements (`retain_live` split, the terminal-carries-`dir` argument). One conflict to resolve up front (see below): **does `on_saga_demote` clear by `cmd.dir` (needs `DemoteCmd.dir`) or by id-match across both directions (no wire change)?**

**Adjudication:** the endgoal design's "id-match across both directions" is **strictly better** and I adopt it. Rationale: (a) it does NOT grow `DemoteCmd` with a `dir` field it doesn't strictly need — the commit demote already carries `cmd.transfer`, which is the exact direction-keyed latch id, so an id-match `for d in [Inward, Outward]` scan is unambiguous and O(1); (b) it keeps the abort path (`on_crossing_aborted`) symmetric — `CrossingAborted` also stays frozen (id-match), so **neither terminal grows a `dir` field**; (c) it minimizes the HR1 frozen-wire surface to exactly two appends (`CrossingRequest.dir`, `DemoteCmd.new_parent`). The minimal-diff/scale-first "carry `dir` on the terminal" is sound but adds two unnecessary wire fields. **Fewer frozen-contract changes is the tiebreak.**

---

## (2) SYNTHESIZED DESIGN — ordered edit list

Format: `file:line` · exact change · per-branch HR5 · determinism guard. All paths under `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/`.

### WIRE (`crates/wire/src/intershard.rs`) — HR1 one reviewed file, exactly TWO append-only fields + the id seed

**W0 · import** — top of `intershard.rs`: add `use vd_core::geometry::Direction;` (vd-wire→vd-core legal; `Direction` is `Copy+Ord+Serialize`, geometry.rs:210). *HR5:* no branch. *Det:* n/a.

**W1 · `crossing_transfer_id` gains `dir` in the seed** — `intershard.rs:691-699`:
```rust
pub fn crossing_transfer_id(
    subject: DirectoryKey, subject_fence: Fence, attempt: u32, dir: Direction,
) -> TransferId {
    let seed = postcard::to_allocvec(&(subject, subject_fence, attempt, dir))
        .expect("encode crossing id seed");
    namespaced_transfer_id(0x39, &seed)
}
```
Doc-append: "the `dir` distinguishes a dock (Inward) from an undock (Outward) of one entity at the same fence+attempt — distinct ids so their latches/terminals never alias (Slice 4b, CRITICAL-1)." *HR5:* branchless shim, `.expect()` straight-line, ONE monomorphization — automatic. *Det:* `Direction::Serialize` is a stable 1-byte unit-enum discriminant; `namespaced_transfer_id` FNV unchanged; no rng/hasher; the `0x39` tag canary + byte-identical seed-replay guard hold.

**W2 · `CrossingRequest` gains `pub dir: Direction`** (APPENDED last) — `intershard.rs:616-623`. Doc mirrors the `attempt` append note (:611-614): "Slice 4b — the committed direction; the orchestrator re-derives the direction-keyed `crossing_transfer_id`; dock and undock of one entity are DISTINCT sagas with DISTINCT ids." *HR5:* struct field. *Det:* postcard field-append preserves old-shape decode.

**W3 · `DemoteCmd` gains `pub new_parent: Option<RealmId>`** (APPENDED last) — `intershard.rs:518-524`. Doc: "Slice 4b — the fence-keyed reparent target the orchestrator resolved at the winning fence; `Some` iff the demoted subject is a container-realm (`Ship`) whose boundary re-nests under `new_parent` at THIS commit; `None` otherwise (every P3 crossing, every bare-entity crossing). The commit fence (`new_owner_fence`) authorizes it — never speculative at trigger time." *HR5:* field. *Det:* append-safe.
**NOTE — `DemoteCmd` does NOT gain `dir`; `CrossingAborted` stays frozen** (both cleared by id-match — adjudicated above).

**W4 · tests** — `intershard.rs:1335-1406`: thread the 4th arg into every `crossing_transfer_id` call; construct `CrossingRequest{…, dir: Inward}`, `DemoteCmd{…, new_parent: None}`. ADD to `crossing_transfer_id_is_deterministic_and_namespaced`: `assert_ne!(crossing_transfer_id(s,f,0,Inward), crossing_transfer_id(s,f,0,Outward))` — the CRITICAL-1 wire-level proof. *HR5(d):* `assert_ne!`, not `matches!`.

### ORCHESTRATOR (`crates/node/src/saga_runtime.rs`)

**O1 · re-derive with `req.dir`** — `saga_runtime.rs:1424`: `let transfer = crossing_transfer_id(req.subject, req.subject_fence, req.attempt, req.dir);`. Thread `req.dir` into `SagaCtx`. *HR5:* straight-line; the 3-arm match untouched. *Det:* both ends now feed `req.dir` → same id.

**O2 · `reparent_target` helper + `SagaCtx.new_parent`** — new monomorphic fn in vd-node; set in `handle_crossing_request` (`saga_runtime.rs:1436-1449`) as `new_parent: reparent_target(req.subject)`:
```rust
/// The container-realm reparent target for a crossing commit, or None for a non-container subject.
/// A total mapping (HR3 — NOT a feature kind-match): Ship(e)→Some(Ship(e)); all others→None (a bare
/// player/entity crossing re-parents nothing — it has no boundary). Through P3 the orchestrator has no
/// real boundary topology, so this is only ever populated by a Ship container subject a test drives.
fn reparent_target(subject: DirectoryKey) -> Option<RealmId> {
    match subject {
        DirectoryKey::Ship(e) => Some(RealmId::Ship(e)),
        DirectoryKey::Entity(_) | DirectoryKey::Session(_) | DirectoryKey::Realm(_) => None,
    }
}
```
**Key decision:** `new_parent` carries the *realm to reparent* (`Ship(e)`); the *new parent realm* is `req.to_realm` (the destination the container now sits in — for BOTH directions, since the source already resolved Outward dest = `winner.parent` at S4). So O3 stamps `new_parent: ctx.new_parent.map(|_| ctx.to_realm_of_reparent)`. To keep it one field, redefine: `SagaCtx.reparent: Option<(RealmId /*target*/, RealmId /*new_parent*/)>`, and `DemoteCmd.new_parent: Option<RealmId>` carries the *new parent*, with the *target* derived on the source via `container_realm_of(cmd.subject)` (S9). This keeps `DemoteCmd` to ONE new field. *HR5:* `reparent_target`'s 4 subject arms (1 Some, 3 None) covered by a dedicated unit test with hand-built keys (`assert_eq!` on the `Option`). *Det:* pure total map.

**O3 · `Demote` executor stamps `new_parent`** — `saga_runtime.rs:914-925`: add `new_parent: ctx.new_parent` (the destination realm, `Some` only for a Ship container subject; **always `None` through P3** — the orchestrator has no boundary topology, so no Ship-container crossing is driven; document this). The D-37 re-home Demote (saga_runtime.rs:~2495) and test builds → `new_parent: None`. *HR5:* straight-line. *Det:* n/a.

**O4 · transient id sentinel** — `saga_runtime.rs:1484`: `crossing_transfer_id(req.subject, req.src_realm_fence, 0, Direction::Inward)`. Doc: "transients carry no direction-keyed latch/terminal; the fixed `Inward` sentinel keeps the batch id deterministic — it never round-trips as a direction." *HR5:* straight-line constant (NO overload — HR3 one machinery). *Det:* fixed sentinel.

**O5 · `PendingAbortReply`** — `saga_runtime.rs:262-268`: **NO change** (the abort clears by id-match, not `dir`; the inline "no extra field needed" note stays TRUE). This is the payoff of the id-match adjudication.

**O6 · tests** — thread `dir`/`new_parent: None` into orchestrator handle/demote/abort test builds.

### SIM SOURCE (`crates/sim/src/stub.rs`)

**S1 · `RequestInFlight` re-keyed** — `stub.rs:406-407`: `BTreeMap<(EntityId, Direction), TransferId>`. Doc: "keyed by `(subject entity, Direction)` so a dock (Inward) and undock (Outward) of ONE entity are DISTINCT in-flight ops — a dock-then-undock interleave does NOT self-suppress (the Occupied suppress is per-direction; CRITICAL-1)." *HR5:* type. *Det:* `(EntityId, Direction): Ord` → BTreeMap iteration deterministic; ≤2 entries/entity (no O(n·m)).

**S2 · `LatchedCrossing` gains `pub dir: Direction`** — `stub.rs:348-357` (stays `Copy`). Doc: "the committed direction — re-emitted verbatim so the redrive re-mints the direction-keyed id." *HR5:* field.

**S3 · `CrossingState.latched_crossing` → `[Option<LatchedCrossing>; 2]`** — `stub.rs:393`. Doc: "per-DIRECTION re-emit payload (index 0=Inward, 1=Outward). `Some` iff a durable latch is held for that `(entity, dir)`. A `[_;2]` (not a map) keeps `CrossingState` `Copy`+`Default` and the index branchless. One entity can hold BOTH a dock and undock payload." *HR5:* `[Option<_>;2]` is `Copy`+`Default` (`[None,None]`) — verify the `#[derive]` at :363 holds. Add branchless total map `const fn dir_ix(d: Direction) -> usize { match d { Direction::Inward => 0, Direction::Outward => 1 } }` — both arms covered by exercising one dock + one undock latch. *Det:* array index by total map, no cross-host divergence.

**S4 · `crossing_attempt` → `[u32; 2]`** — `stub.rs:386`. Doc: "per-DIRECTION attempt; `on_crossing_aborted` bumps only the aborted direction's counter, so an Inward abort never perturbs an Outward re-latch's id." *HR5:* `[u32;2]` `Copy`+`Default` (`[0,0]`). *Det:* per-direction, deterministic index.

**S5 · `fan_out_crossing` — direction-keyed take + Outward destination + Interest-not-short-circuited (LOW-1) + no-parent-loud (MEDIUM-1)** — `stub.rs:2622-2712`:
- `:2628` rename `_dir` → `dir` (load-bearing).
- `:2635-2636` REMOVE the top-level `let to_realm = winner.to_realm;`. Set only `let from_realm = ctx.config.realm;`.
- **Interest arm** (`:2638-2643`): UNCHANGED — reads `winner.to_realm` directly, direction-agnostic, NO degrade, NO short-circuit (LOW-1: the destination compute must NOT precede the `match`).
- **Authority+Durable arm** (`:2644`): after the `session` gate, compute via a monomorphic helper `let Some(to_realm) = outward_dest(dir, winner, stats) else { return; };`, then `in_flight.entry((entity, dir))`; in the Vacant arm: `let attempt = state.crossing_attempt[dir_ix(dir)];`, `crossing_transfer_id(subject, subject_fence, attempt, dir)`, `CrossingRequest{…, dir}`, `state.latched_crossing[dir_ix(dir)] = Some(LatchedCrossing{ to_realm, subject_fence, session, dir });`. Occupied arm unchanged (`crossings_suppressed_in_flight += 1`) — now suppresses only a SAME-direction re-fire.
- **Authority+Transient arm** (`:2695`): same `outward_dest` compute for the dest; the id keeps the `Inward` sentinel (O4) — the DESTINATION honors `dir` (undock debris re-homes to parent) but the batch id does not need direction.
- **`outward_dest` helper** (new, monomorphic, ~:2591):
```rust
/// Direction-resolved destination for a durable/transient crossing (Slice 4b): Inward docks to the
/// boundary interior (winner.to_realm); Outward undocks to winner.parent. A top-level Outward with NO
/// parent is an AUTHORING error (every real realm nests in an enclosing System/Galaxy realm) — count it
/// LOUD (a malformed-boundary tripwire a valid fixture asserts stays 0) and skip. Monomorphic; all three
/// branches covered once here (HR5).
fn outward_dest(dir: Direction, winner: &RealmBoundary, stats: &mut StubStats) -> Option<RealmId> {
    match dir {
        Direction::Inward => Some(winner.to_realm),
        Direction::Outward => {
            let p = winner.parent;
            if p.is_none() { stats.crossing_outward_no_parent += 1; }
            p
        }
    }
}
```
*HR5:* three branches — Inward (Some), Outward+Some, Outward+None (counter, None) — each covered by a dedicated unit test hand-building a `RealmBoundary` with/without `parent`; `if p.is_none()` split cleanly (no `&&`); `assert_eq!` on the returned `Option`. The Vacant/Occupied × Inward/Outward matrix covered by the dock/undock/interleave/same-dir tests. *Det:* integer-gated commit unchanged; `dir_ix`/`outward_dest` total maps.

**S6 · MEDIUM-1 plant-time guard** — in the boundary-plant seam (test-only through P3; `geometry.rs` is the boundary's home — add `RealmBoundaries::validate()` there, or a `debug_assert!` at the stub.rs plant sites :8041/:8424-style). Asserts every `CrossEffect::Authority` boundary reachable Outward has `parent.is_some()`. A valid-fixture test asserts `crossing_outward_no_parent == 0`; a dedicated malformed-fixture unit test drives it `>0` ONCE to cover the arm. This makes the runtime `None` a one-time authoring tripwire, not a 20 Hz silent storm. *HR5:* the counter is a real tested region (not `#[coverage(off)]` — more honest per the scale-first/endgoal argument). *Det:* pure.

**S7 · `evaluate_one_subject` signature** — `stub.rs:2488`: `in_flight: &mut BTreeMap<(EntityId, Direction), TransferId>`. Callers at :2423/:2447 pass `&mut in_flight.0` (type flows through). The winner destructure at :2494 (`Some((ix, _dir))`) unchanged — the *commit* direction is `should_commit`'s return (:2527), already passed to `fan_out_crossing` at :2540. *HR5:* signature. *Det:* n/a.

**S8 · `retain_live` split (DRY)** — `stub.rs:2399-2401`, `2467-2469`. Add a SECOND monomorphic primitive:
```rust
fn retain_live_keyed<V>(map: &mut BTreeMap<(EntityId, Direction), V>, live: &BTreeSet<EntityId>) {
    map.retain(|(e, _), _| live.contains(e));
}
```
Call `retain_live_keyed(&mut in_flight.0, &live)` at :2400; `progress`/`interest` keep `retain_live`. *HR5:* each helper's closure branch covered once — `retain_live` by the progress-evict test, `retain_live_keyed` by a 4b two-direction-evict test. *Det:* deterministic `retain`.

**S9 · `redrive_stranded_crossings`** — `stub.rs:2751-2781`:
- `:2751` `for ((entity, dir), _latched) in in_flight.0.iter()`.
- `:2752-2758` `let state = progress.0.get_mut(entity).expect(...)`; `let lc = state.latched_crossing[dir_ix(*dir)].expect("a held durable latch carries its per-direction re-emit payload");`.
- `:2772-2781` re-emit `CrossingRequest{…, to_realm: lc.to_realm, attempt: state.crossing_attempt[dir_ix(*dir)], dir: *dir}`.
- **Invariant note (load-bearing — I verified this):** the loop keys off `in_flight` membership; a COMMITTED direction is `remove`d from `in_flight` in `on_saga_demote` (S10) → never visited here → its now-stale `latched_crossing[i]` payload is never read. So the `.expect()` holds per-direction (a held `(entity,dir)` in `in_flight` ALWAYS armed slot `dir_ix(dir)` in S5's Vacant arm). *HR5:* `.expect()` straight-line, no coverable false arm; `dir_ix` both arms via an Inward-redrive + Outward-redrive test. *Det:* `(EntityId, Direction)` ordered iteration.

**S10 · `on_saga_demote` — id-match clear across BOTH directions + enqueue reparent** — `stub.rs:1597-1634` (NO `CrossingProgress` param — confirmed at source):
```rust
Some(entity) => {
    self_fence_foreign_entity(&mut dots.0, entity, cmd.new_owner_fence, cmd.transfer, clock.local_tick, stats);
    // Clear whichever direction's latch holds THIS committed id (the commit demote carries the exact
    // direction-keyed latched id). Id-match — not a blind per-entity remove — since Inward and Outward
    // are distinct keys and DemoteCmd carries no direction (kept off the frozen wire).
    for d in [Direction::Inward, Direction::Outward] {
        if in_flight.0.get(&(entity, d)) == Some(&cmd.transfer) {
            in_flight.0.remove(&(entity, d));
            stats.crossing_latches_cleared += 1;
        }
    }
}
```
Then enqueue the fence-keyed reparent (S12): `if let (Some(np), Some(tr)) = (cmd.new_parent, container_realm_of(cmd.subject)) { reparents.0.push((tr, np)); }` — needs a new `reparents: &mut PendingReparents` param threaded through the dispatch fn (:3263/:3330). **Through P3 `cmd.new_parent` is always `None` → nothing enqueued.** The stale `latched_crossing[dir]` payload is left for lazy overwrite/evict (S9 invariant makes it harmless — no `CrossingProgress` threading needed). *HR5:* `== Some(&cmd.transfer)` equality (HR5(d) covered false arm); the `for d in [..]` covers both direction compares; test asserts the OTHER direction's latch SURVIVES a single-direction demote. *Det:* fixed 2-element iteration order.

**S11 · `on_crossing_aborted` — per-direction id-match clear + per-direction attempt bump** — `stub.rs:1682-1719`. `CrossingAborted` stays frozen (no `dir`); scan both directions by id-match:
```rust
for d in [Direction::Inward, Direction::Outward] {
    if in_flight.0.get(&(entity, d)) == Some(&abort.transfer) {
        in_flight.0.remove(&(entity, d));
        let st = progress.0.entry(entity).or_default();
        st.crossing_attempt[dir_ix(d)] = st.crossing_attempt[dir_ix(d)].saturating_add(1);
        st.latched_crossing[dir_ix(d)] = None;
        // The dwell/hysteresis reset stays per-entity (single winner slot); only the aborted
        // direction's attempt+payload bump/clear — the OTHER direction's latch/attempt untouched.
        st.inward_ticks = 0; st.was_member = false; st.winner_ix = None; st.last_commit_tick = None;
        stats.crossing_latches_cleared += 1;
        return; // exactly one direction can hold a given transfer id
    }
}
stats.crossing_abort_stale += 1;
```
*HR5:* `== Some(&abort.transfer)` (HR5(d)); the `return` after the match makes the stale-branch (`crossing_abort_stale`) a covered path; `dir_ix` both arms via Inward-abort + Outward-abort tests. *Det:* fixed iteration order; `saturating_add` guard intact.

**S12 · `PendingReparents` + `apply_pending_reparents` (HIGH-1 resolution)** — dedicated post-inbound system, NOT `ResMut<RealmBoundaries>` in `process_inbound` (keeps it at 16 params; the param-ceiling map proves the `Res`/`ResMut` conflict with `evaluate_realm_boundaries` is INERT because group-A `.chain()` already serializes before group-B):
- New resource `stub.rs:~867`: `#[derive(Resource, Debug, Default)] pub struct PendingReparents(pub Vec<(RealmId /*target*/, RealmId /*new_parent*/)>);` Doc: "queued by `on_saga_demote`, drained by `apply_pending_reparents` in group B so `process_inbound` never holds `ResMut<RealmBoundaries>`. Deterministic push-order drain, cleared each tick. Through P3 ALWAYS empty."
- `process_inbound` fold `PendingReparents` into the existing `crossing` tuple → `(ResMut<RequestInFlight>, ResMut<CrossingProgress>, ResMut<PendingReparents>)` (stub.rs:1048/1052) — `RealmBoundaries` stays OUT; top-level params stay 16.
- New helper `container_realm_of(subject) -> Option<RealmId>` (total map: `Ship(e)→Some(Ship(e))`, else `None`) — distinct from `transfer_subject_entity` (which returns `None` for `Ship` — I confirmed at directory.rs:37).
- New system in group B (`stub.rs:905-917`, `.after(evaluate_realm_boundaries)`):
```rust
fn apply_pending_reparents(
    mut boundaries: ResMut<RealmBoundaries>,
    mut reparents: ResMut<PendingReparents>,
    mut stats: ResMut<StubStats>,
) {
    for (target, new_parent) in reparents.0.drain(..) {          // empty every P3 tick (early no-op)
        match boundaries.0.iter_mut().find(|b| b.realm == target) {
            Some(b) => { b.parent = Some(new_parent); stats.boundaries_reparented += 1; }
            None => stats.reparent_no_boundary += 1,
        }
    }
}
```
Placed AFTER `evaluate_realm_boundaries` so this tick's crossings evaluate against PRE-reparent geometry; the reparent (the commit consequence) applies for next tick. *HR5:* `Some`/`None` `find` arms both covered by dedicated unit tests (matching/absent boundary); the empty-drain (P3 path) is the covered zero-iteration case (`assert boundaries_reparented == 0` through a normal tick). *Det:* `Vec::drain` push-order + `find`-first-match over `Vec<RealmBoundary>` (stable order); no rng/clock. **HIGH-2 honest coverage:** the `Some` arm is production-unreachable through P3 (`new_parent` always `None` from O3) — covered ONLY by a hand-built unit test (`DemoteCmd{subject: Ship(e), new_parent: Some(Station(1))}` + a boundary with `realm == Ship(e)` → assert `b.parent == Some(Station(1))`, `boundaries_reparented == 1`), NEVER the E2E. This exactly resolves the HIGH-2 contradiction, mirroring the proven `emit_crossing` pattern at saga_runtime.rs:5988.

**S13 · new `StubStats` counters** — `boundaries_reparented`, `reparent_no_boundary`, `crossing_outward_no_parent`. *HR5:* incremented in covered arms above.

**All 5 adversary findings resolved:** CRITICAL-1 → S1/S2/S3/S4/S5/S9/S10/S11 (direction-keyed everything). HIGH-1 → S12 (dedicated system, 16 params held). HIGH-2 → S12 (explicit `container_realm_of` target-key + `Some`-arm dedicated unit test, not E2E). MEDIUM-1 → S6 (plant-time guard + assert-zero tripwire, loud not silent). LOW-1 → S5 (destination compute INSIDE Authority arms, Interest untouched).

---

## (3) INERT-THROUGH-P3 STATEMENT (explicit)

**The entire slice is behaviour-identical to Slice-4a on every P3 production run.** Proof chain:
1. `RealmBoundaries` defaults EMPTY (stub.rs:867; the composer plants none through P3) → `evaluate_realm_boundaries` early-returns (:2387) → NO crossing, NO latch, NO direction, NO `CrossingRequest` is ever produced in production. The direction-keyed id/latch/payload/attempt reshape is exercised ONLY by sim integration tests that plant boundaries.
2. `O3` emits `DemoteCmd.new_parent = None` unconditionally through P3 (the orchestrator has no boundary topology to resolve a parent from, and no Ship-container crossing is driven) → `on_saga_demote` (S10) enqueues nothing → `PendingReparents` always empty → `apply_pending_reparents` drains zero → `RealmBoundaries` is NEVER mutated in production. `apply_pending_reparents`'s `Some(find)` arm is production-unreachable; covered only by S12's unit test.
3. Wire changes are exactly TWO postcard field-appends (`CrossingRequest.dir`, `DemoteCmd.new_parent`) + the id-seed grow → old-shape decode preserved; a P3 message on a non-crossing path never populates them. `CrossingAborted` and `PendingAbortReply` are UNCHANGED (id-match clear).
4. `outward_dest`'s `None`-degrade and the `dir==Outward` branch are unreachable in production (no boundaries → no Outward commit); covered by malformed-input unit tests only; `crossing_outward_no_parent` asserts `0` in every valid fixture.

**Net:** byte-identical wire on every non-crossing path, identical schedule behaviour with `RealmBoundaries` empty. Only the boundary-planting sim tests exercise the new surface.

---

## (4) TEST PLAN — what each proves

**Unit (vd-sim + vd-wire + vd-node):**
- `crossing_transfer_id_direction_distinct` (W4) — `assert_ne!(id(…,Inward), id(…,Outward))`. *Proves:* the id root of CRITICAL-1 — dock and undock hash distinctly.
- `outward_dest_all_branches` (S5) — Inward→`Some(to_realm)`, Outward+parent→`Some(parent)`, Outward+`None`→`None`+counter. *Proves:* the direction-keyed destination + the MEDIUM-1 loud degrade, all three HR5 branches.
- `reparent_target_and_container_realm_of` (O2/S12) — 4 subject arms each. *Proves:* the HIGH-2 explicit target-key rule (`Ship`→Some, others→None) — honest coverage of both extractors.
- `apply_pending_reparents_some_none_empty` (S12) — hand-built `(Ship(e))` boundary matched (`parent` mutated, counter 1) / missed (`reparent_no_boundary`) / empty-drain (no-op). *Proves:* HIGH-2 `Some` arm covered WITHOUT the E2E (the contradiction resolution) + the P3 inert no-op.
- `retain_live_keyed_evicts_both_directions` (S8). *Proves:* the DRY split + eviction correctness.
- `no_parent_outward_authority_fails_at_plant` (S6). *Proves:* MEDIUM-1 — a malformed Outward boundary is caught loud at plant; valid fixtures assert `crossing_outward_no_parent == 0`.

**dock/undock/three-deep-exit E2E** (`stub.rs` new, planting a nested boundary set):
- `a_dock_commits_inward_to_the_station_interior` — Inward crossing → `CrossingRequest{dir:Inward, to_realm == winner.to_realm}` → saga commits → `DemoteCmd{dir-latch cleared}`. *Proves:* dock leg = ONE ordinary transfer through THE saga (HR2/HR3).
- `an_undock_commits_outward_to_the_parent_realm` — Outward → `CrossingRequest{dir:Outward, to_realm == winner.parent}`. *Proves:* the undock destination = parent (the R-A map/plan resolution — `winner.parent`, NOT `to_realm`).
- `a_three_deep_exit_peels_one_level_per_tick` — player→ship→station→planet: three nested Outward boundaries, assert exactly ONE `CrossingRequest{dir:Outward}` fires per tick, innermost-first, each targeting the next parent up. *Proves:* the three-deep spine — three single-level transfers, one commit/tick (winner-order), parent-chain as single truth.

**interleave-does-not-suppress-undock regression** (the CRITICAL-1 capstone — this INVERTS the old `the_in_flight_latch_suppresses_a_second_request` at stub.rs:8095-8143):
- `a_dock_then_undock_interleave_fires_both_with_distinct_ids` — drive Inward to commit (latch `(e,Inward)` held, UNcleared), then Outward before the Inward Demote lands → assert BOTH a `CrossingRequest{dir:Inward}` AND `{dir:Outward}` fire, `req0.transfer != req1.transfer`, both keyed distinctly in `in_flight`, `crossings_suppressed_in_flight == 0`. *Proves:* the headline — the undock is NO LONGER suppressed (the Occupied arm now suppresses only same-direction).
- `a_same_direction_re_edge_is_still_suppressed` — two matured Inward edges → one request, `crossings_suppressed_in_flight > 0`. *Proves:* the suppress semantics are preserved for the same-direction case (no regression of the 3d latch discipline).
- `on_saga_demote_clears_only_the_named_direction` (extend :8974) — latch both directions, demote the Inward id → assert `(e,Inward)` gone, `(e,Outward)` SURVIVES. *Proves:* the id-match clear is direction-exact.
- `on_crossing_aborted_bumps_only_the_aborted_direction` (extend :8726) — abort Inward → `crossing_attempt[Inward]` bumped, `[Outward]` and the Outward latch untouched. *Proves:* CRITICAL-1's core guarantee — cross-direction state isolation.
- `redrive_re_emits_the_direction_keyed_id` (extend :8805) — a stranded Outward latch re-drives with `dir:Outward` + `to_realm == winner.parent` + byte-identical id. *Proves:* the redrive per-direction payload (S3) is correct and the `.expect()` invariant holds.

**ledger-eviction:**
- `slice4b_both_direction_latches_evict_on_subject_leave` (extend `slice4a_all_three_ledgers_evict…`) — latch `(e,Inward)`+`(e,Outward)`, remove the subject → BOTH `in_flight` entries evict via `retain_live_keyed`; `progress`/`interest` evict via `retain_live`. *Proves:* no latch leak on logout/eviction across the new two-slot-per-entity storage — the direction re-key does not regress the 3d eviction discipline.

---

## (5) DEFERRED — ledger lines

- **`DEFERRED.md:3043` (4b-DEFERRED) 🟧→🟩** on land — direction-keyed latch + fence-keyed reparent + no-parent-loud + Interest-not-short-circuited all resolved by this slice.
- **`DEFERRED.md:3040` (#4 per-(entity,boundary) cooldown)** — stays 🟧 UNLESS the Authority A-then-B-within-`k_dwell` regression test (below) proves a drop. `last_commit_tick` stays per-entity this slice (the fix is entangled with `winner_ix` and orthogonal to the direction axis). **Ledger line:** *"Per-(entity,boundary) cooldown keying deferred; landed a regression test (`authority_a_then_b_within_dwell`) that pins the hazard honestly — if it fails, promote `last_commit_tick` to per-`winner_ix` keying. Not a CRITICAL-1 blocker."*
- **`DEFERRED.md:3041` (#5 stable `winner_ix`)** — stays 🟧, and this slice does NOT regress it. **Honest ledger line:** *"`apply_pending_reparents` mutates only `RealmBoundary.parent`, NEVER the slice ORDER (no add/remove/reorder), so raw-index `winner_ix` stays stable through Slice-4b. #5 only bites a P4/P5 dynamic-registry reorder — remains deferred; NOT closed by 4b."*
- **`DEFERRED.md` D-32 / P6 (mass-dock saga-throughput)** — unchanged in kind. **Ledger line:** *"Hundreds docking = hundreds of un-coalesced durable `CrossingRequest`s → the single in-proc orchestrator (D-32). The direction key adds ≤1 extra latch entry/entity (bounded ≤2×), NOT an O(n·m) surface; the saga fan-out remains the P6 throughput gate. Slice-6 rayon seam pre-identified at `evaluate_realm_boundaries`'s two subject loops (per-`(entity,dir)` latch writes are disjoint → deterministic partition-and-merge)."*
- **Area-as-signal-endpoint (P6 Tier-2)** — untouched; a Station/Ship signal endpoint is a first-class `DirectoryKey` and composes now (endgoal Q2). **Ledger line:** *"A docked ship's blocks signalling a Station is same-substrate Interest/Signal (both endpoints first-class keys); only a bare-Area signal endpoint stays P6-ledgered."*
- **3+ simultaneous client frames (client slice)** — the parent-chain single-truth generalizes N-deep on the server; N-frame render compositing is an untested generalization of the current 2-shard compositing. **Ledger line:** *"Server-side N-deep parent-chain is proven; N-frame client render compositing flagged for the client slice, not this pre-P4 server work."*

**Key files touched:** `crates/wire/src/intershard.rs` (W0-W4), `crates/node/src/saga_runtime.rs` (O1-O6 + `reparent_target`), `crates/sim/src/stub.rs` (S1-S13 + `dir_ix`/`outward_dest`/`container_realm_of`/`retain_live_keyed`/`apply_pending_reparents`), `crates/core/src/geometry.rs` (S6 `RealmBoundaries::validate`), `docs/design/DEFERRED.md` (ledger flips).

---

## ADVERSARY (GO-WITH-FIXES; CRITICAL FINDING 1)

I now have all the load-bearing facts verified against source. The critical structural claims all check out: `on_saga_demote` has no `CrossingProgress`; `redrive` keys off `in_flight.iter()`; `DemoteCmd.transfer` is the direction-keyed latch id; `crossing_transfer_id` seed is `(subject, subject_fence, attempt)`; `Direction` is `Copy+Ord+Serialize`; `process_inbound` is at 16 params with `crossing` tuple; `evaluate_realm_boundaries` reads `Res<RealmBoundaries>`.

Now let me hunt for the breaks. I have enough to render the hostile verdict.

# HOSTILE ADJUDICATION — Slice-4b Synthesized Design

I verified every load-bearing structural claim against source (`stub.rs:347-413, 1597-1719, 2371-2400, 2480-2552, 2622-2712, 2733-2788`; `intershard.rs:508-524, 615-699`; `directory.rs:37-42`; `pose.rs:21-35`; `geometry.rs:205-216, 678-698`). The synthesis's factual foundation is **sound** — the adjudication point (`on_saga_demote` has no `CrossingProgress`) is confirmed at line 1597, and the redrive-keys-off-`in_flight` invariant is real (line 2751). But there are genuine defects in the direction-keyed keying, one of them CRITICAL, that the synthesis's own confidence conceals.

---

## FINDING 1 — CRITICAL: the `crossing_attempt` bump aliases a live opposite-direction latch id across the `on_saga_demote` id-match clear

**This is the attempt-bump collision the prompt asked me to hunt for, and it breaks the (entity,direction) keying at exactly one site.**

Trace a fast entity through the interleave the slice exists to defend:

1. Entity `e` docks. `fan_out_crossing` Inward-Vacant arm: `attempt_I = crossing_attempt[Inward] = 0`, mints `id_I = crossing_transfer_id(subj, f, 0, Inward)`, latches `(e,Inward)→id_I`, arms `latched_crossing[0]`.
2. Before the Inward Demote lands, `e` reverses. Outward-Vacant arm: `attempt_O = crossing_attempt[Outward] = 0`, mints `id_O = crossing_transfer_id(subj, f, 0, Outward)`, latches `(e,Outward)→id_O`. So far the direction key holds — `id_I ≠ id_O` (W1). Good.
3. Now the Inward saga **aborts** (dest realm shard mid-lease — the exact stranded case redrive exists for, or a CAS-lost). `on_crossing_aborted` (S11) fires with `abort.transfer = id_I`. It id-matches `(e,Inward)`, removes it, **bumps `crossing_attempt[Inward] → 1`**, clears `latched_crossing[0]`, and — per S11 as written — `return`s.

So far consistent. But now the hazard: **`e` re-docks** (Inward rising edge re-fires — the dwell was reset by the abort re-arm, so it can re-fire without a physical re-cross, exactly as designed). Inward-Vacant: `attempt_I = crossing_attempt[Inward] = 1` now, mints `id_I' = crossing_transfer_id(subj, f, 1, Inward)`, latches `(e,Inward)→id_I'`.

**The collision:** the Outward saga from step 2 was *still in flight* the whole time, latch `(e,Outward)→id_O` with `id_O = (…,0,Outward)` held. Its Demote finally lands: `on_saga_demote` (S10) fires with `cmd.transfer = id_O`. S10's clear loop scans `for d in [Inward, Outward]` doing `if in_flight.get(&(e,d)) == Some(&cmd.transfer)`. It compares `(e,Inward)→id_I'` against `id_O` (no match, good), then `(e,Outward)→id_O` against `id_O` (match) — removes `(e,Outward)`. **This is actually correct here** because `id_I' ≠ id_O` (the attempt bump on the *Inward* axis kept them distinct).

So where's the break? **The break is that S10/S11 clear by scanning BOTH directions with an id-match, but the id namespace is NOT partitioned by direction after the attempt counters diverge across a re-latch.** Construct the aliasing case directly:

- `crossing_attempt = [Inward: a_I, Outward: a_O]`. The id is `crossing_transfer_id(subj, f, attempt, dir)`. Because W1 puts BOTH `attempt` AND `dir` in the seed, `id(f, a_I, Inward)` and `id(f, a_O, Outward)` collide **iff** the tuples `(subj,f,a_I,Inward)` and `(subj,f,a_O,Outward)` FNV-collide — they can't, `dir` differs. So W1 as specified (dir in the seed) **does** prevent the cross-direction alias. 

**Re-examining: the actual CRITICAL is narrower and real.** S10's loop removes whichever direction holds `cmd.transfer`. But `DemoteCmd` carries no direction — and the synthesis explicitly keeps `CrossingAborted`/`DemoteCmd` frozen precisely to avoid a wire field. Consider a **stale redelivered Demote**. The at-least-once transport (R-6, landed) can redeliver a `DemoteCmd{transfer: id_I}` for the *first, aborted* Inward saga (step 3 aborted, but a Demote for it was already in the outbox from a CAS that later got superseded — or simply a duplicate of a legitimately-committed earlier attempt). After the re-dock (step 4), `(e,Inward)→id_I'` is held with the *bumped* id. The redelivered `DemoteCmd{transfer: id_I}` (old attempt 0) arrives: S10 scans, `(e,Inward)→id_I'` ≠ `id_I` → no match; `(e,Outward)→id_O` ≠ `id_I` → no match. Clean no-op. **Correct** — the attempt bump saved it, exactly as the H2 machinery was designed to.

**The genuine CRITICAL is the `self_fence_foreign_entity` call that runs UNCONDITIONALLY before the latch scan (line 1608).** S10 preserves the existing structure: `self_fence_foreign_entity(&mut dots.0, entity, cmd.new_owner_fence, cmd.transfer, …)` fires on EVERY demote for the entity, *keyed on `entity` alone*, regardless of direction. A redelivered/stale `DemoteCmd` for the aborted Inward attempt carries `cmd.new_owner_fence` from that superseded saga. `self_fence_foreign_entity` gates on `is_stale_against` (fence monotonicity) — but the entity now legitimately holds a live Outward crossing with its own fence trajectory. **The self-fence is direction-blind and entity-keyed; the latch clear is direction-keyed.** A stale Inward demote whose `new_owner_fence` is NOT stale-against the current dot (because the two directions advanced the fence independently) will self-fence the dot `Owned→Frozen→Ghost` while the Outward saga still expects it Owned at the source. **The entity is ghosted by the wrong direction's terminal → double-homed or stranded.**

- **Severity: CRITICAL.** The direction key is applied at the `RequestInFlight` latch (S1) and at the id (W1), but NOT at the `self_fence_foreign_entity` demote-apply (line 1608, unchanged in S10). The synthesis threads direction through the *latch bookkeeping* but leaves the *authority mutation* entity-keyed. This is the "does the (entity,direction) keying actually hold at EVERY site" break — it does not hold at line 1608.
- **Concrete fix:** S10 must gate the `self_fence_foreign_entity` call on the id-match, not run it unconditionally. Move the self-fence INSIDE the `if in_flight.get(&(e,d)) == Some(&cmd.transfer)` matched branch (self-fence only the direction whose latch this demote actually clears), OR — since the self-fence already gates on `cmd.transfer` via `is_stale_against` internally — verify that the fence carried by a demote is monotone across BOTH directions of one entity. The current design cannot assume that: two concurrent sagas for one entity mint independent `new_owner_fence` values at their respective CAS wins, and there is no cross-direction fence ordering. **Required before GO:** either serialize the two directions at the CAS (one entity cannot have two live crossing sagas — add a directory-level per-subject saga-exclusion, which the orchestrator's `active_sagas` keyed on `DirectoryKey` may already provide — VERIFY), or make the self-fence id-gated. The synthesis never audited the self-fence site.

---

## FINDING 2 — HIGH: S12 `PendingReparents` reparent applies to a boundary registry that `evaluate_realm_boundaries` read EARLIER THE SAME TICK — but the winner-selection depth key is now one-tick-stale in a way that can drop an exit level

The synthesis places `apply_pending_reparents` in group B `.after(evaluate_realm_boundaries)`, arguing "this tick's crossings evaluate against PRE-reparent geometry; the reparent applies for next tick." That ordering is defensible for the CAS-race concern. But trace the three-deep exit the Slice-4b tests actually drive (`a_three_deep_exit_peels_one_level_per_tick`):

- Tick T: player crosses the innermost (ship) boundary Outward. Winner selection picks ship (depth 2). Commits `CrossingRequest{Outward, to_realm = ship.parent = station}`. The saga commits; `DemoteCmd{new_parent}` — but `new_parent` is `reparent_target(Entity(player)) = None` (the player is a bare entity, not a container). So `apply_pending_reparents` does nothing. Fine.
- The reparent only fires when the **subject is a Ship container** (`reparent_target(Ship(e)) = Some`). So the three-deep *player* exit never triggers a reparent — correct. But the ship-in-station undock (subject `Ship(e)`) DOES. When the ship undocks from the station, `apply_pending_reparents` mutates `boundaries[ship].parent = station.parent = planet`.

**The break:** `boundary_depth` (stub.rs:2597-2612) walks `parent` links to compute the depth key that `resolve_winner_ix` uses for innermost-first selection. `apply_pending_reparents` mutates a `parent` link. The synthesis's DEFERRED ledger line claims "#5 stable `winner_ix` — `apply_pending_reparents` mutates only `RealmBoundary.parent`, NEVER the slice ORDER, so raw-index `winner_ix` stays stable." **That is true for the Vec index but FALSE for the depth key.** Mutating `boundaries[ship].parent` from `Some(station)` to `Some(planet)` changes `boundary_depth(ship)` from 2 to 1 — which is precisely a `depth` change, the **first and dominant key** in `candidate_beats_ix` (geometry.rs:641, `a.0 > b.0`). An entity straddling the ship boundary and the (now-reparented) station boundary the very next tick can have its winner flip because the depth ordering changed under it. The synthesis audited `winner_ix` (the tiebreak-4 slice index) but NOT `depth` (tiebreak-1), which the reparent directly mutates.

- **Severity: HIGH.** Not a strand (the reparent is P3-inert, `new_parent` always `None`), so it cannot fire in P3 — but the Slice-4b tests that plant a `Ship(e)` container boundary and drive its undock (which the prompt says the tests DO — "descend→dwell→ascend both directions latch on one entity") WILL reach it if any test drives a container-subject undock. If the tests only drive bare-entity crossings, this is INERT-and-untested (an HR5 gap: the `Some` reparent arm covered only by the isolated unit test, never the E2E, which the synthesis actually claims as a feature). **The contradiction:** the synthesis says the three-deep test drives `Outward` peeling one level per tick, but a three-deep *ship* exit (ship undocks from station undocks from planet-orbit) is a container-subject chain that WOULD reparent mid-sequence and perturb the depth key.
- **Concrete fix:** the DEFERRED ledger line #5 must be corrected — the claim "never the slice ORDER so `winner_ix` stable" is insufficient; add "`apply_pending_reparents` mutates `parent`, which mutates `boundary_depth`, the DOMINANT winner key — so a container reparent MUST NOT be observable by a straddling entity's winner selection within the same dwell. Guard: a reparent-in-flight for boundary B suppresses B as a winner candidate until the reparent settles (one-tick), OR assert no entity straddles a container boundary during its own undock (the container is empty of straddlers by construction — the undocking ship carries its interior with it)." **Required before GO:** prove the straddler-during-container-reparent case is impossible OR guard it; and confirm which the Slice-4b tests actually drive (bare-entity vs container-subject exit).

---

## FINDING 3 — HIGH: `outward_dest` returns `winner.parent`, but the winner is selected by `crossing_candidates` using `winner.to_realm` for depth/tiebreak — an Outward winner's `to_realm` is the INTERIOR, so tiebreak-2 (`RealmId` Ord) sorts on the wrong realm for exits

`candidate_beats_ix` (geometry.rs:641-654) breaks ties on `a.1 < b.1` where `a.1` is the candidate's `RealmId`. `crossing_candidates` (stub.rs:2558-2578) builds candidates; the projected realm in `resolve_winner` is `winner.to_realm` (the INWARD destination). For an Outward crossing the *actual* destination is `winner.parent` (S5 `outward_dest`), but the tiebreak still sorts on `to_realm`. For two nested Outward-capable boundaries with the same depth (a degenerate but plantable fixture — two coincident shells), the winner is chosen by the interior realm's Ord, then the destination is computed as that winner's parent. **The selection key and the destination are now different realms.** This is not obviously wrong (the selection only needs to be *deterministic and total*, which sorting on `to_realm` still is), but the synthesis's claim that the tiebreak is "permutation-invariant and correct for both directions" is unproven for the case where the tiebreak realm (`to_realm`) and the outcome realm (`parent`) diverge. A test that plants two same-depth Outward boundaries with `to_realm_A < to_realm_B` but `parent_A ≠ parent_B` will pick A's parent deterministically — but whether A's-parent is the *semantically correct* level to peel to is a design question the synthesis waves through.

- **Severity: HIGH** (determinism is preserved — this is not a determinism hole — but *correctness* of which level peels is unaudited). The prompt's determinism-hole hunt clears this: `to_realm`-keyed tiebreak is a total order, `Direction` is `Ord`, iteration is over a `BTreeMap`/slice with stable order. No determinism break. But the *semantic* winner for an exit is unproven.
- **Concrete fix:** add a test `two_same_depth_outward_boundaries_peel_deterministically` asserting the tiebreak realm and the peeled-to parent, and DOCUMENT that the Outward tiebreak sorts on `to_realm` (interior) by design (it must — the candidate is built before direction resolves the destination). **Required before GO:** the test + the doc note; this is a latent trap, not a live break.

---

## FINDING 4 — MEDIUM: the `[Option<LatchedCrossing>; 2]` + `[u32; 2]` reshape is claimed `Copy+Default` "verify the derive at :363 holds" — it holds, but the redrive `.expect()` invariant (S9) has a hole under a lost-enqueue

`CrossingState` (line 363) derives `Copy, Default`. `[Option<LatchedCrossing>; 2]` and `[u32; 2]` are both `Copy+Default` — **confirmed sound**, credit to the synthesis for keeping the array over a map. But S9's `.expect("a held durable latch carries its per-direction re-emit payload")` reads `latched_crossing[dir_ix(*dir)]` for every `(entity,dir)` in `in_flight`. The invariant "a held `(entity,dir)` in `in_flight` ALWAYS armed slot `dir_ix(dir)`" holds in the Vacant arm (S5 sets both together). But `retain_live_keyed` (S8) evicts `in_flight` entries by `live.contains(e)`, while `retain_live` (unchanged) evicts `progress` (which holds `latched_crossing`) by the SAME `live` set. The synthesis calls both from `evaluate_realm_boundaries` at lines 2399-2400. **If the two retains use the same `live` set they stay in sync** — verified: both filter on `live.contains(e)`, and the `(entity,dir)` key's `.0` is the entity, so a subject in `live` keeps ALL its `in_flight` directions AND its single `progress` entry. Sync holds. But the ORDER matters: `retain_live(progress)` runs at 2399, `retain_live_keyed(in_flight)` at 2400. If a subject leaves, both drop it. Consistent. **No break** — but the `.expect()` will panic (not degrade) if the two ever desync, and the synthesis provides no fail-loud-not-panic path. Per the codebase's own DEGRADE-never-panic discipline (stub.rs:2648, "must not crash the shard"), a `.expect()` on a cross-resource invariant in a hot 20Hz system is a fragility.

- **Severity: MEDIUM.** The invariant holds today; the panic is a latent operational hazard if a future edit desyncs the retains.
- **Concrete fix:** replace the S9 `.expect()` with a `let Some(lc) = state.latched_crossing[dir_ix(*dir)] else { stats.redrive_missing_payload += 1; continue; };` — a covered degrade (HR5: the `None` arm counted, tested by a hand-built desync unit test) rather than a panic. This matches the existing `crossing_durable_no_session` degrade pattern. **Required before GO:** downgrade the `.expect()` to a counted skip.

---

## FINDING 5 — MEDIUM: HR5 per-monomorphization trap in `retain_live_keyed<V>` — the generic closure branch is counted per instantiated `V`

`retain_live_keyed<V>(map: &mut BTreeMap<(EntityId, Direction), V>, live)` — the synthesis calls it ONCE with `V = TransferId` (in_flight). But `retain_live` (the existing sibling) is instantiated for `CrossingState`, `RealmId`. Per HR5's own generic-code gotcha (CLAUDE.md: "llvm counts regions PER MONOMORPHIZATION"), `retain_live_keyed`'s `map.retain(|(e,_),_| live.contains(e))` closure is a region counted for EACH `V` it's instantiated with. Today only `V=TransferId` — so it's covered by the one 4b eviction test. **No break now.** But the synthesis's DRY claim "each helper's closure branch covered once" is fragile: the moment a second call site instantiates `retain_live_keyed<RealmId>` (a plausible future when interest zones become direction-keyed), the coverage gate flips red until a `RealmId`-instantiation test lands. The synthesis correctly makes it a branchless shim (single `.retain` call, no `?`/`if`) — which is exactly the HR5 discipline — so the region is trivially covered per-mono. **This is correctly handled.** Credit.

- **Severity: MEDIUM → downgraded to correctly-handled.** The branchless-shim shape means per-mono coverage is automatic as long as every instantiation has a test. Flag only: the test plan must add a `retain_live_keyed<V>` test for every future `V`.

---

## FINDING 6 — MEDIUM: SCALE — the redrive scan is now `O(live latches)` = up to `2N` per tick, and the S10 fixed-2-iteration clear is fine, but the `crossing_attempt` array + per-direction latch doubles the hot-map footprint under hundreds-docking

The synthesis claims "≤2 latch entries/entity, no O(n·m)." Verified: `RequestInFlight` is `BTreeMap<(EntityId,Direction), TransferId>`, ≤2 entries/entity. `redrive_stranded_crossings` iterates `in_flight.0.iter()` every tick (line 2751) — now up to `2N` for N docking entities, each doing a `progress.get_mut` + `[dir_ix]` index. For hundreds docking (N=300), that's 600 map lookups/tick in the redrive alone, plus 600 in `evaluate_one_subject`'s per-subject pass. The synthesis correctly identifies the real wall as the orchestrator saga fan-out (D-32) and pre-plans the rayon seam. **The per-direction doubling is bounded (2×, not n²)** — credit. But the redrive runs `if ttl == 0 { return }` — so it's gated off unless a TTL is configured; under a live TTL with 300 stranded docks, the 2N scan is real per-tick work. Not a blowup, a constant-factor doubling of an already-O(N) scan.

- **Severity: MEDIUM.** Bounded, correctly ledgered to D-32/P6, rayon seam pre-identified. The 2× is honest and disclosed.
- **Concrete fix:** none required for GO; the DEFERRED line D-32 is accurate. Add to it: "the direction re-key doubles the redrive scan to ≤2N; still O(N), still gated by `ttl`."

---

## FINDING 7 — LOW: no per-realm-KIND match introduced (HR3 clean) — verified

I checked every new match in the design for a realm/shard-KIND discriminant match in a feature path:
- `reparent_target(subject: DirectoryKey)` matches on `DirectoryKey` arms (`Ship`/`Entity`/`Session`/`Realm`) — a **total mapping**, not a feature kind-branch. HR3-clean (this is the `realm()`/`label()` shape).
- `container_realm_of(subject)` — same, total `DirectoryKey → Option<RealmId>` map.
- `outward_dest(dir, …)` matches on `Direction` (2 arms) — a total direction map, not a realm kind.
- `dir_ix(d)` — total `Direction → usize`.
- S10/S11 iterate `[Inward, Outward]` — fixed, total.

**No HR3 violation.** No `match` on `RealmId::{Ship,Station,Area,Planet,System}` anywhere in the feature logic. Credit — the synthesis correctly kept every mapping total. The one place a KIND asymmetry LEAKS is HIGH-2's own admission (MAP reparent-seam (c)): a Station realm is keyed by `u64` seed, a Ship by `EntityId`, so `container_realm_of` can only bind the `Ship(EntityId)` arm from a `DirectoryKey::Ship(e)` — a Station-container reparent has no entity to key from. The synthesis's `container_realm_of` returns `Some(Ship(e))` for `Ship(e)` and `None` for all else — which means **Station containers cannot reparent at all** through this machinery. That's a real end-goal gap (a station undocking from a planet's orbit), but it's honestly `None`-through-P3 and ledgered. LOW because Station-container reparent is a P6+ concern.

---

## FINDING 8 — LOW: END-GOAL cornering check — cross-shard signals, multi-mesh render, station-in-orbit all clear

- **Cross-shard signals from a docked ship:** the Interest arm (S5, unchanged) emits ZERO flow, touches no latch, no CAS. A docked ship's blocks signalling the station ride the Interest/Signal substrate independent of the crossing latch. **Not cornered.** Confirmed the direction re-key does not contaminate the signal path (the latch is authority-only).
- **Multi-mesh render off the parent-chain:** render walks `RealmBoundary.parent`; `apply_pending_reparents` mutates the same `parent` — single-truth preserved. **But FINDING 1's self-fence hazard means a wrong-direction ghost = a render bug too** (render walks the chain, sees the entity in the wrong realm). This is the trigger-bug-is-also-render-bug coupling the endgoal MAP flagged. Cornering risk is downstream of FINDING 1, not independent.
- **Station-in-orbit reuse:** taxonomy reuse (hull_host + Station realm) is clean; the crossing/reparent machinery reuse inherits FINDING 1 + the Station-container-key gap (FINDING 7). **Not cornered at the taxonomy level; the reparent-target machinery is Ship-only through P3.**

---

## INERT-THROUGH-P3 AUDIT — the synthesis's claim is MOSTLY honest, with one reachable-now exception

The synthesis claims the entire slice is behaviour-identical to 4a on every P3 run because `RealmBoundaries` defaults empty. **True for production.** But the prompt flags: "the Slice-4b tests plant boundaries and drive descend→dwell→ascend — both directions latch on one entity." That means the **direction-keyed latch, the `[_;2]` arrays, the id-with-dir, S10's two-direction clear, S11's per-direction abort bump, and `outward_dest` ARE reachable in the test binary** — they are NOT inert, they are exercised. The synthesis's INERT claim is scoped correctly to *production* ("byte-identical wire on every non-crossing path") but the deliverable-critical logic (FINDING 1's self-fence interaction, FINDING 2's container-reparent depth perturbation) is live in tests, which is exactly where FINDING 1's CRITICAL will manifest if a test drives dock-abort-redock-with-live-opposite-latch. **The synthesis does not include that specific interleave test** — its `a_dock_then_undock_interleave_fires_both_with_distinct_ids` proves the fire, but NOT the abort-then-redock-then-stale-demote self-fence race. That test is missing and would catch FINDING 1.

---

## VERDICT: **GO-WITH-FIXES**

The synthesis is structurally sound, HR3-clean, determinism-clean (no mutable-read-in-ordered-pass, `(EntityId,Direction):Ord` BTreeMap iteration, fixed 2-element direction loops, total maps throughout), and correctly resolves HIGH-2's coverage-honesty contradiction and HIGH-1's param-ceiling via a dedicated group-B system. The `[_;2]` array choice, the `outward_dest`/`dir_ix` branchless shims, and the id-match-across-both-directions tiebreak (avoiding two frozen-wire fields) are genuinely better than the alternatives. But it threaded the direction key through the latch bookkeeping while leaving the **authority mutation (`self_fence_foreign_entity`) entity-keyed** — the CRITICAL — and mis-audited the reparent's effect on the **dominant depth key**.

### Required-before-GO list

1. **[CRITICAL — FINDING 1]** Gate `on_saga_demote`'s `self_fence_foreign_entity` on the id-match (self-fence only the direction whose latch this demote clears), OR prove one entity cannot hold two concurrent live crossing sagas (verify the orchestrator's per-`DirectoryKey` saga exclusion covers both directions — if it serializes dock/undock at the subject, the whole (entity,direction) latch reshape may be unnecessary and the CRITICAL dissolves). **This is the disqualifying item — resolve first, it may simplify the entire slice.**
2. **[CRITICAL test — FINDING 1 + INERT audit]** Add `a_dock_abort_redock_then_stale_inward_demote_does_not_ghost_the_live_outward` — the missing interleave test that exercises the self-fence-vs-latch direction mismatch.
3. **[HIGH — FINDING 2]** Correct the DEFERRED #5 ledger line: `apply_pending_reparents` mutates `parent` → mutates `boundary_depth` (the DOMINANT winner key), not just the slice index. Guard or prove-impossible the straddler-during-container-reparent case. Confirm whether Slice-4b tests drive a container-subject (`Ship(e)`) exit or only bare-entity exits.
4. **[HIGH — FINDING 3]** Add the two-same-depth-Outward tiebreak test + doc that the Outward tiebreak sorts on `to_realm` (interior) by construction.
5. **[MEDIUM — FINDING 4]** Downgrade S9's redrive `.expect()` to a counted `continue` (DEGRADE-never-panic, matching `crossing_durable_no_session`).

Credited as correctly handled: HR3 totality (F7), the `[_;2]` Copy+Default reshape, the id-match tiebreak minimizing frozen-wire surface, the branchless-shim HR5 discipline (F5), the HIGH-2 dedicated-unit-test coverage honesty, the bounded 2× scale with D-32 rayon seam pre-identified (F6), and the cross-shard-signal non-contamination (F8). The synthesis's confidence was warranted on everything EXCEPT the site it never looked at — the demote's authority mutation.