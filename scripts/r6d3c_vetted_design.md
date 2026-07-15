# R-6d3c Vetted Design of Record (wf_9e5f126d)

## ADJUDICATED SYNTHESIS
All claims are verified. I confirmed reviewer 3's scope claim: `Arriving` and `Departing` are `!is_held()` (uncounted), and the dest-only-status assertion holds — the abandon handler removes `Departing`/`Held{Some}` (source statuses), so `on_transient_discard` correctly needs only the single `Arriving` arm. I also confirmed reviewer 3's NIT #4: the abandon counter doc at :575-577 already says "Each increment ALSO feeds transients_lost_in_handover" despite the corrupt-tag Err arm — so this imprecision is pre-existing and the design should not copy it verbatim.

I have everything needed. Here is the adjudicated design of record.

---

# R-6d3c — ADJUDICATED Design of Record (NEVER-restart closure of D-6 #1)

**VERDICT: SOUND_TO_IMPLEMENT** (with 4 folded must-fixes; all others confirmed sound or downgraded to NITs).

The DESIGN is code-accurate on every load-bearing claim. I verified each against the tree and folded the reviewers' real findings. No CRITICAL survives; no closed-set breach, silent-loss, resurrect-after-discard, or double-resolution hazard is introduced beyond one already-tracked CA-1 residual that the design honestly ledgers.

## ADJUDICATION of the reviews (dedup + confirm/refute against code)

**M1 — "budget-gate does NOT order the race; the anti-double-resolution safety net is the phase-structure + adopt-poison" (reviewer 1, MEDIUM). CONFIRMED — fold as a must-fix (narrative + one test), keep the budget-gate.**
Verified at `saga_runtime.rs:1676-1678` (`record_ack(*from)` fires on ANY `Inbound::Wire`) + `LivenessTracker::record_ack` (`:274-276`, removes the node's evidence). The restart re-drives source→dest `TransientBatch`; the dest acks `BatchAdopted` to the ORCHESTRATOR — that inbound is from the DEST, so it clears the DEST's evidence, NOT the source's. So a restarted source can advance the saga `AwaitAdopt→AwaitRelease` (`saga.rs:838-845`) while `is_confirmed_dead(source)` is still true. The HARD guarantees against double-resolution are: (i) once `BatchAdopted` moves the saga to a post-adopt phase, the `AwaitAdopt`-discard arm is structurally unreachable; (ii) if the discard already fired, the `TRANSIENT_BATCH_STEP` poison makes a late replayed `adopt_transient_batch` hit `AlreadyApplied` (`stub.rs:2163,2181`). The budget is a *widening* of the restart window (desirable, mirrors the dest-dead ladder at `:1274-1280`), NOT the correctness proof. **Fold:** correct the `CountBatchLostSourceCrash` doc + design narrative, and ADD the interleave test "discard fires → then a late `BatchAdopted` arrives at the now-`Done` saga → terminal-absorb no-op (`saga.rs:1153`) + (composed) the dest adopt is poisoned."

**M2 — "the `scan_deadlines` counter match has a `_ => {}` wildcard; the new counter MUST be added there or it is silently dropped with no compile error" (reviewer 1, MEDIUM). CONFIRMED — this is the single silent-omission risk in the slice; fold as a must-fix.**
Verified `saga_runtime.rs:1532-1536` is exactly `match event { SourceUnreachable => …, DestUnreachable => …, _ => {} }`. Every OTHER new `SagaEvent`/`SagaAction` forces a compile error (the FSM `step` is total-with-catch-all by design; the executor match at `:717-893` has NO wildcard — verified). This counter arm is the ONE place a missing arm compiles green. **Fold:** add `SagaEvent::SourceUnreachablePreAdopt => runtime.batch_lost_source_crash += 1` BEFORE the `_`, and require BOTH a `scan_deadlines`-level test (covers the counter) AND the direct-`deliver()` test (the direct test bypasses `scan_deadlines`, so it alone does NOT cover the counter).

**NIT-A — "the design's affected-tests list is WRONG: `:3806` is the Demoting arm, unaffected by the BatchHandoff-only gate" (reviewer 1, NIT). CONFIRMED — fold the correction.**
Verified `saga_runtime.rs:3806` injects `SagaState::Demoting {…}` (line 3820), driven by the D-37 CELL-1 arm at `:1290-1296` — which the design does NOT change. Only `:4293` (`SagaState::BatchHandoff { phase: AwaitPromote }`, line 4302-4303) is affected: it asserts source-dead fires on the FIRST scan with `n==1` and NO budget (line 4315-4317), which the added budget-gate breaks. **Fold:** the tick-bump list is `:4293` ONLY. Do NOT touch `:3806`.

**L1 (dest) / L1 (wire) / open-Q#3 — "a LOST discard + a late batch replay re-orphans; ReDriven is semantically stretched for a fire-once-then-tombstone arm" — these are the SAME root cause (no re-driver behind a terminal-saga egress). DEDUP to ONE. CONFIRMED as a REAL residual; DOWNGRADE severity from the design's implied "closed" to an explicitly-ledgered CA-1-gated gap. Not a blocker.**
Verified: the resolving saga tombstones (`saga.rs:922-925 → Done`), `deadline_for` returns `u64::MAX` for `Done`, `Tombstone` removes the saga — so a lost `TransientDiscard` has no re-driver, exactly as the shipped `TransientAbandon` (`saga.rs:926-928`, classified `ReDriven` at `:351`, pushed Ephemeral, ack-free at `stub.rs:2468-2474`). If the discard is lost AND the `TransientBatch` outbox replay lands later, `adopt_transient_batch` runs UNPOISONED → re-inserts `Arriving`. **This means the reachable-now dest cure closes the orphan only WHEN the discard is delivered.** `ReDriven` is nonetheless the CORRECT classification within the existing taxonomy (classifying it `ProducerLessReliable` would trip the push-`Ephemeral` debug_assert with no outbox re-emitter behind it AND break the `producer_less.len()==2` pin). **Fold:** keep `ReDriven` and fire-once ack-free (consistent with `TransientAbandon`); make the arm doc HONEST (not the inherited "scan_deadlines re-drives it" — say "orchestrator-EMITTED so it does not grow the outbox set; fire-once terminal, its lost-delivery residual is covered by the CA-1 re-solicit, NOT scan_deadlines"); and DOWNGRADE the DEFERRED.md phrasing from "never-restart CORRECTNESS = closed by R-6d3c" to "closed ON delivered-discard; the lost-discard residual folds into the SAME CA-1 reliability gate as the trigger."

**Coverage NIT (dest, reviewer 3) — "the `EntityKind::from_tag` Err arm must be exercised or vd-sim drops below 100% branch." CONFIRMED — fold as a required test case.** The abandon test seeds `EntityId(99u128 << 120)` (`stub.rs:3785`); the discard test must mirror it.

**Doc NITs (counter semantics, reviewer 3 #4; "15-arm"→"16-arm" header, reviewer 2). CONFIRMED — fold.** Reviewer 3 #4 is right that the abandon counter doc at `stub.rs:575-577` ALREADY over-claims "Each increment ALSO feeds transients_lost_in_handover" despite the Err arm — the new counter doc must NOT copy that; word it "each increment with a DECODABLE kind also feeds `transients_lost_in_handover`." Update `intershard_closed.rs:5` "15-arm" → "16-arm".

**Counter-conflation NIT (reviewer 3 #2) — "`AlreadyApplied` reuses `transient_release_noop`, weakening the idempotence assertion." CONFIRMED, accept option (a): keep the shared counter (DRY, matches abandon at `stub.rs:2347`); the idempotence test asserts on `transients_discarded_source_crash` unchanged + `owned.0` unchanged (the FirstApply-only counter IS discard-specific).**

**Confirmed-SOUND (no change): the saga phase-split correctness (reviewer 1 summary), the closed-set landing through all 3 gates (reviewer 2 summary), the single-arm `Arriving`-only match for the dest (reviewer 3 summary — verified: `Departing`/`Held{Some}` are source-side statuses set at emit/release, `stub.rs:2214`/`2211`; a dest only holds `Arriving` or `Held{None}` for a batch it destinations), the untouched nested tripwires (reviewer 2 #3 — `TransientDiscard` is an OUTER arm reusing the existing flat `TransientHandoff`, not a new `TransitionPayload`/`GhostFlow` variant).**

**No upgrades warranted.** The one residual with silent-loss FLAVOR (lost-discard re-orphan) is genuinely CA-1-gated (the same reliability gate as the trigger) and now explicitly ledgered rather than over-claimed — it is a narrowing, not a regression.

---

## THE CORRECTED DESIGN

### R-6d3c-1 — the wire arm (`vd-wire`, Tier-A 100%)

`crates/wire/src/intershard.rs`:
- Add const after `RE_HOME_STEP` (`:84`): `pub const TRANSIENT_DISCARD_STEP: u32 = 17;` (16 is `RE_HOME_STEP`; 17 verified free/disjoint from 0-6 route-swap, 7-16 state/transient/rehome).
- Append the arm AFTER `ReHome` (`:165`) to preserve every postcard discriminant:
```rust
    /// Orchestrator → DEST shard (R-6d3c): the source died in AwaitAdopt (PRE-adopt) so the batch is
    /// counted lost; REMOVE any Arriving{batch==transfer} item AND poison (transfer, TRANSIENT_BATCH_STEP)
    /// so a late outbox replay's adopt is AlreadyApplied (never re-inserts). Reuses `TransientHandoff`.
    /// Side-effecting, ack-FREE (the resolving saga is terminal); idempotent by (transfer,
    /// TRANSIENT_DISCARD_STEP). Classified ReDriven because it is orchestrator-EMITTED (it does NOT grow
    /// the producer-less outbox set / trip the push-Ephemeral debug_assert) — NOT because scan_deadlines
    /// re-drives it: it is a FIRE-ONCE terminal egress, its lost-delivery residual is covered by the
    /// CA-1/L5 re-solicit (§3), NOT by a re-driver. APPENDED (preserves every existing postcard discriminant).
    TransientDiscard(TransientHandoff),
```
- `effect_class` (`:288-296`): add `| InterShardFlow::TransientDiscard(h)` to the existing `TransientRelease | TransientDrop | ReleaseComplete | TransientAbandon` group → `SideEffecting{TransferStep{h.transfer, h.step_id}}`.
- `durability_class` (`:340-352`): add `| InterShardFlow::TransientDiscard(_)` to the big `ReDriven` group.

`crates/wire/tests/intershard_closed.rs`:
- `arm_tripwire` (`:226-242`): add `| InterShardFlow::TransientDiscard(_)`.
- `every_arm` (`:52-217`): add `TransientDiscard(TransientHandoff{transfer:TransferId(10), step_id:TRANSIENT_DISCARD_STEP, fence:Fence(6)})`.
- `durability_class_pins_the_producer_less_reliable_set` (`:301`): the discard falls into `_ => ReDriven` (`:320`); **assert `producer_less.len() == 2` UNCHANGED** (`:331-334`) — the load-bearing pin that the arm does NOT grow the outbox set.
- In-module `g_sealed_effect_classes`: add the discard to the shared classification loop.
- **Fold (NIT):** update `:5` header "15-arm" → "16-arm".

Gate: wire 100% region+branch; `every_arm_roundtrips_postcard_and_classifies_coherently` green.

### R-6d3c-2 — the dest handler (`vd-sim`, Tier-A 100%)

`crates/sim/src/stub.rs`:
- Import `TRANSIENT_DISCARD_STEP` (`:32-33` region).
- New counter in `StubStats` (~`:577`), **HONEST doc (fold NIT):**
```rust
    /// R-6d3c — transients DISCARDED at the dest by `on_transient_discard` (source died in AwaitAdopt
    /// pre-adopt). 0 on every happy path and on a RESTART recovery (which adopts normally); `> 0` is the
    /// never-restart cell's anti-vacuity proof. Each increment WITH A DECODABLE kind also feeds
    /// `transients_lost_in_handover`; a corrupt-tag item is removed + counted HERE but NOT attributed
    /// (the from_tag Err arm, HR2 never-decode-to-default).
    pub transients_discarded_source_crash: u64,
```
- `on_transient_discard` (model on `on_transient_abandon` `:2318`, but DEST-side, single `Arriving` arm, poison, DRY loss-bucket):
```rust
fn on_transient_discard(
    discard: TransientHandoff,
    owned: &mut OwnedTransients,
    applied: &mut AppliedSteps,
    stats: &mut StubStats,
) {
    match applied.journal_step(discard.transfer, TRANSIENT_DISCARD_STEP) {
        StepOutcome::FirstApply => {
            // POISON the adopt so a late outbox replay is AlreadyApplied (never re-inserts an orphan).
            let _ = applied.journal_step(discard.transfer, TRANSIENT_BATCH_STEP);
            let mut to_remove: Vec<EntityId> = Vec::new();
            for (entity, t) in owned.0.iter() {
                if let TransientStatus::Arriving { batch } = t.status
                    && batch == discard.transfer
                {
                    to_remove.push(*entity);
                }
            }
            for entity in to_remove {
                owned.0.remove(&entity);
                stats.transients_discarded_source_crash += 1;
                if let Ok(kind) = EntityKind::from_tag(entity.kind_tag()) {
                    *stats.transients_lost_in_handover.entry(kind).or_insert(0) += 1;
                }
            }
        }
        StepOutcome::AlreadyApplied => stats.transient_release_noop += 1,
    }
}
```
- Dispatch arm in `on_directory_reply` (`:2471` region, mirroring the ack-free `TransientAbandon`):
```rust
        Ok(InterShardFlow::TransientDiscard(discard)) => {
            on_transient_discard(discard, owned_transients, applied, stats);
            return;
        }
```
Tests (all `assert_eq!`/`expect_err`, split short-circuit `&&`, HR5(d)):
1. `on_transient_discard_removes_arriving_poisons_adopt_and_is_idempotent` — seed `Arriving{batch}`; discard removes + buckets + `transients_discarded_source_crash==1`; second discard `AlreadyApplied`, assert `owned.0` unchanged + `transients_discarded_source_crash` unchanged (the discard-specific anti-vacuity signal, per reviewer-3 fold).
2. `discard_before_adopt_poisons_so_a_late_replay_never_orphans` (the target interleave) — discard on EMPTY `owned` (poison only) → `adopt_transient_batch(same transfer)` → `owned.0.is_empty()` AND `transients_adopt_redelivered==1`. **The direct proof Defect A is closed.**
3. `on_transient_discard_on_a_settled_or_departing_item_leaves_it` — a `Held`/`Departing` item is NOT removed (the false `Arriving` arm, 100% branch).
4. **Fold (coverage):** a corrupt-tag `Arriving` item (`EntityId(99u128 << 120)`, mirroring `stub.rs:3785`) — removed + `transients_discarded_source_crash` incremented but NOT bucketed (the `from_tag` Err arm).

Gate: sim 100% region+branch.

### R-6d3c-3 — the saga FSM + producer (`vd-sim` FSM + `vd-node` producer, Tier-A 100%)

`crates/sim/src/saga.rs`:
- `SagaEvent::SourceUnreachablePreAdopt` (after `SourceUnreachable`, `:509`).
- `SagaAction::EmitTransientDiscard { fence: Fence }` + `CountBatchLostSourceCrash` (after `EmitTransientAbandon`, `:615`), **`CountBatchLostSourceCrash` doc corrected per M1:** "counted in `scan_deadlines` pre-delivery; the anti-double-resolution guarantee is the phase-structure (post-adopt phases cannot route here) + the dest adopt-poison, NOT this counter or the budget ordering."
- Split the wildcard at `:922-925` into (POST-adopt keeps self-promote; PRE-adopt discards):
```rust
        (
            S::BatchHandoff { phase: P::AwaitRelease | P::AwaitPromote | P::AwaitComplete, new_fence },
            E::SourceUnreachable,
        ) => (S::Done { new_fence }, vec![A::EmitTransientPromote { fence: new_fence }, A::Tombstone]),
        (
            S::BatchHandoff { phase: P::AwaitAdopt, new_fence },
            E::SourceUnreachablePreAdopt,
        ) => (
            S::Done { new_fence },
            vec![A::EmitTransientDiscard { fence: new_fence }, A::CountBatchLostSourceCrash, A::Tombstone],
        ),
        (S::BatchHandoff { new_fence, .. }, E::DestUnreachable) => (
            S::Done { new_fence },
            vec![A::EmitTransientAbandon { fence: new_fence }, A::Tombstone],
        ),
```
  Crossed events (`SourceUnreachable`@AwaitAdopt, `SourceUnreachablePreAdopt`@post-adopt) fall through the terminal-absorb catch-all (`:1154`) — total, at-least-once safe.
- MODIFY `batch_handoff_resolves_terminally_on_a_dead_participant` (`:1528`): AwaitAdopt uses `SourceUnreachablePreAdopt` (→ `[EmitTransientDiscard, CountBatchLostSourceCrash, Tombstone]`); the three post-adopt phases keep `SourceUnreachable` (→ `EmitTransientPromote`). Split the loop.
- ADD `await_adopt_source_unreachable_is_a_noop` + `post_adopt_source_unreachable_pre_adopt_is_a_noop` (the crossed-event catch-all arms — 100% branch AND they pin the producer contract per reviewer-1 L1).

`crates/node/src/saga_runtime.rs`:
- `rehome_event_for` BatchHandoff branch (`:1270-1285`) — phase-discrimination + budget-gate (the budget WIDENS the restart window; M1 narrative):
```rust
        SagaState::BatchHandoff { phase, .. } => {
            if liveness.is_confirmed_dead(ctx.source, now) {
                let observed = *dead_observed_since.get_or_insert(now);
                if now.0.saturating_sub(observed.0) >= tuning.abort_deadline_ticks {
                    *dead_observed_since = None;
                    match phase {
                        BatchHandoffPhase::AwaitAdopt => SagaEvent::SourceUnreachablePreAdopt,
                        _ => SagaEvent::SourceUnreachable,
                    }
                } else {
                    SagaEvent::Timeout
                }
            } else if liveness.is_confirmed_dead(ctx.dest, now) {
                // (unchanged dest-dead ladder, :1274-1280)
            } else {
                *dead_observed_since = None;
                SagaEvent::Timeout
            }
        }
```
- Executor arms (`:867` region, no wildcard so both forced):
```rust
                SagaAction::EmitTransientDiscard { fence } => {
                    outbox.push_flow(ctx.dest, MsgClass::Saga,
                        &InterShardFlow::TransientDiscard(TransientHandoff {
                            transfer: ctx.transfer, step_id: TRANSIENT_DISCARD_STEP, fence }));
                }
                SagaAction::CountBatchLostSourceCrash => {} // counted in scan_deadlines pre-delivery
```
- `batch_lost_source_crash: u64` field + accessor (mirror `source_unreachable_resolutions` `:526`).
- **Fold (M2 — the silent-omission fix):** in the `scan_deadlines` counter match (`:1532-1536`), add BEFORE the `_`:
```rust
            SagaEvent::SourceUnreachablePreAdopt => runtime.batch_lost_source_crash += 1,
```
- **Fold (NIT-A — corrected test list):** MODIFY `scan_deadlines_resolves_a_batch_handoff_with_a_dead_participant` (`:4293`) ONLY — advance `now` past `abort_deadline_ticks` (the source-dead path is now budget-gated; the current test asserts first-scan fire at `n==1`). **Do NOT touch `:3806`** (it is the `Demoting` arm at `:1290-1296`, unchanged). Add `batch_lost_source_crash()` accessor assertions.
- ADD `rehome_event_for_await_adopt_source_dead_emits_pre_adopt_past_budget_else_redrives` (mirror `:3933`): AwaitAdopt + dead source → `Timeout` before budget, `SourceUnreachablePreAdopt` past budget; **assert a post-adopt phase past budget yields `SourceUnreachable` and NOT `SourceUnreachablePreAdopt`** (closes the mis-pairing gap, reviewer-1 L1).
- ADD the direct-inject `deliver(runtime, .., SourceUnreachablePreAdopt)` into an `AwaitAdopt` saga → asserts `outbox` carries `TransientDiscard` to `ctx.dest` + `Done` (covers the `EmitTransientDiscard`/`CountBatchLostSourceCrash` executor arms; NO CA-1).
- **Fold (M1 — the anti-double-resolution proof):** ADD the interleave test "discard fires (AwaitAdopt+SourceUnreachablePreAdopt past budget) → then a late `BatchAdopted` arrives → the now-`Done` saga terminal-absorbs it (no re-transition, no second egress)"; and a composed handler assertion that after `on_transient_discard` a subsequent `adopt_transient_batch` is poisoned. This is the real anti-double-resolution proof, currently only implied.
- REQUIRE a `scan_deadlines`-level test asserting `batch_lost_source_crash()==1` after an AwaitAdopt+dead-source resolution past budget, `==0` for the post-adopt path (M2 — the direct-inject test bypasses `scan_deadlines`, so this is separately required).

Gate: sim + node 100% region+branch. Every new arm is monomorphic straight-line (`(state, vec![...])`) — HR5(a) trivially satisfied.

### CA-1/L5-GATED follow-up (NOT in R-6d3c; ledgered in DEFERRED.md)

The `AwaitAdopt` RE-SOLICIT egress (idempotent orch→source re-prompt so a dead source accrues `NodeUnreachable{source}` and `is_confirmed_dead(source)` can fire — REQUIRED because AwaitAdopt is entered with an EMPTY egress at `saga.rs:820-822`, and `confirm_and_maybe_bounce` at `mesh.rs:1449-1470` only bounces a lane with a non-empty retry) + the confirm-dead-toward-source TRIGGER + the real-QUIC `sigkill_source_in_await_adopt_never_restarts` process proof. Depends on CA-1 (`mesh.rs:2890` `#[ignore]` red-guard, reply-on-connection/addr-reread) + M3. **The lost-discard re-orphan residual (dedup'd L1) folds into THIS SAME reliability gate** (not a separate item).

## REACHABLE-NOW vs CA-1 — HONEST bottom line

- **RESTART case: 🟩 closed** (R-6d3b-2b, commit 2f140a4 — the durable outbox re-drives the batch on boot replay).
- **NEVER-restart CORRECTNESS: 🟩 closed by R-6d3c ON DELIVERED-DISCARD** — the dest can never strand an `Arriving` orphan (the discard removes it + poisons the adopt regardless of replay timing), and the saga resolves to a bounded terminal with a loud `batch_lost_source_crash`/`transients_discarded_source_crash` counter instead of self-promoting an empty dest or parking forever. The pure-logic half (dest handler + poison + phase-split + producer discrimination + budget-gate) lands complete NOW, testable with ZERO CA-1 (direct-inject + FSM `assert_eq!` + composed poison test).
- **NEVER-restart DETECTION-IN-PROD: 🟧 open-until-CA-1/L5** — the confirm-dead trigger + re-solicit egress that make the AwaitAdopt resolution AUTO-FIRE in prod. The lost-discard re-orphan residual folds into this same gate. This is a pre-existing HARD deploy precondition already tracked (DEFERRED.md:957, R-6 + CA-1).

R-6d3c is a genuine NARROWING: the silent-loss becomes an accounted-loss-on-detection; the only remaining gap is the detection trigger, already ledgered.

## FILES TOUCHED
- `crates/wire/src/intershard.rs` (const, arm, effect_class group, durability_class group, in-module g_sealed loop)
- `crates/wire/tests/intershard_closed.rs` (arm_tripwire, every_arm, golden pin assert-unchanged, "16-arm" header)
- `crates/sim/src/stub.rs` (on_transient_discard, StubStats counter, dispatch arm, import, 4 unit tests)
- `crates/sim/src/saga.rs` (SagaEvent, 2 SagaActions, phase-split arms, MODIFY 1 test + ADD 2 no-op tests)
- `crates/node/src/saga_runtime.rs` (rehome_event_for branch, 2 executor arms, counter+accessor, **scan_deadlines counter arm [M2]**, MODIFY `:4293` ONLY [NIT-A], ADD rehome-producer test + direct-inject deliver test + scan_deadlines-counter test + **the late-BatchAdopted interleave test [M1]**)
- `docs/design/DEFERRED.md` (D-6 #1: RESTART 🟩; NEVER-restart CORRECTNESS 🟩-on-delivered-discard; DETECTION + lost-discard residual 🟧-CA-1/L5)

---
## SUB-SLICES
- R-6d3c-1 (vd-wire, Tier-A): the InterShardFlow::TransientDiscard arm + TRANSIENT_DISCARD_STEP=17 (appended, preserves postcard discriminants); classify effect_class (SideEffecting/TransferStep, folded into the existing TransientRelease|TransientDrop|ReleaseComplete|TransientAbandon group) + durability_class (ReDriven, folded into the big group — NOT producer-less); land through arm_tripwire + every_arm + the golden durability_class pin (assert producer_less.len() STAYS 2). Gate: wire 100% region+branch, postcard roundtrip green.
- R-6d3c-2 (vd-sim, Tier-A): on_transient_discard dest handler (remove Arriving{batch==transfer}, POISON (transfer, TRANSIENT_BATCH_STEP) via journal_step, bucket loss into transients_lost_in_handover by kind + new transients_discarded_source_crash counter, idempotent by TRANSIENT_DISCARD_STEP) + the dispatch arm in on_directory_reply. Tests: discard-before-adopt (poison ⇒ late replay no-orphan, the target interleave), discard-after-adopt (remove), idempotent redelivery, the false Arriving arm. Gate: sim 100% region+branch.
- R-6d3c-3 (vd-sim FSM + vd-node producer, Tier-A): SagaEvent::SourceUnreachablePreAdopt + SagaAction::EmitTransientDiscard/CountBatchLostSourceCrash; split saga.rs:922 BatchHandoff arm by phase (AwaitAdopt→discard+count+Done; post-adopt→unchanged self-promote); rehome_event_for phase-discrimination + abort_deadline_ticks budget-gate so a restart-within-budget wins; batch_lost_source_crash counter + accessor; executor arms; modify the 3 budget-affected/phase-affected existing tests + add the assert_eq! split-arm FSM tests + the direct-inject deliver() runtime test (NO CA-1). Gate: sim + node 100% region+branch.
- CA-1/L5-GATED follow-up (NOT in R-6d3c, ledgered in DEFERRED.md): the AwaitAdopt re-solicit egress (idempotent orch→source re-prompt, 2d FSM-field template) that makes is_confirmed_dead(source) reachable in AwaitAdopt + the confirm-dead-toward-source trigger + the real-QUIC sigkill_source_in_await_adopt_never_restarts process proof. Depends on CA-1 (mesh.rs:2890 red-guard) + M3.

---
## REACHABLE-NOW vs CA-1
REACHABLE NOW (pure Tier-A logic, testable with zero CA-1): (1) the entire dest side — the InterShardFlow::TransientDiscard wire arm through all 3 closed-set gates + on_transient_discard which REMOVES Arriving{batch} AND POISONS (transfer,TRANSIENT_BATCH_STEP) so a late outbox replay adopts as AlreadyApplied and never re-inserts an orphan. This is the ACTUAL cure for the dest-side silent orphan (Defect A) and lands complete. (2) The saga phase-split — SourceUnreachablePreAdopt/EmitTransientDiscard/CountBatchLostSourceCrash + the AwaitAdopt-vs-post-adopt arms + the budget-gate in rehome_event_for. This makes a future confirm-dead do the RIGHT thing (discard-to-dest + accounted loss, NOT self-promote-an-empty-dest). Both are testable via a DIRECT deliver(runtime, .., SourceUnreachablePreAdopt) into an AwaitAdopt saga (bypassing the trigger) + a saga FSM assert_eq! unit test of the split + a composed on_transient_discard-then-adopt_transient_batch handler test proving zero orphan. NET EFFECT NOW: the never-restart loss becomes impossible-if-detected + accounted (loud batch_lost_source_crash / transients_discarded_source_crash counters into the SAME transients_lost_in_handover budget) — the dest can NEVER strand an Arriving orphan regardless of replay timing, and the saga resolves to a bounded terminal instead of parking or self-promoting empty. CA-1/L5-GATED (stays open): the AwaitAdopt RE-SOLICIT egress (an idempotent orch→source re-prompt so a dead source accrues NodeUnreachable{source} and is_confirmed_dead(source) can fire — REQUIRED because AwaitAdopt is entered with an empty egress at saga.rs:820-822 and confirm_and_maybe_bounce at mesh.rs:1449-1470 only bounces a lane with a non-empty retry) + the confirm-dead-toward-source TRIGGER. These need CA-1 (mesh.rs:2890 #[ignore] red-guard, reply-on-connection/addr-reread) + M3 (durable monotone incarnation). HONEST BOTTOM LINE: D-6 #1 RESTART case = 🟩 closed (R-6d3b-2b); NEVER-restart CORRECTNESS (discard-not-strand + accounted-loss-on-detection) = 🟩 closed by R-6d3c; NEVER-restart DETECTION-IN-PROD = 🟧 open-until-CA-1/L5 (the trigger that makes the AwaitAdopt resolution auto-fire in prod). R-6d3c is a genuine narrowing: the silent-loss becomes an accounted-loss-on-detection with the only remaining gap being a pre-existing HARD deploy precondition already tracked (DEFERRED.md:957).

---
## DESIGNER (original)

# R-6d3c — NEVER-restart closure of D-6 #1: Design of Record

## 0. The residual, stated precisely

R-6d3b-2b (commit 2f140a4) closed the D-6 #1 **RESTART** case: a source that crashes in `BatchHandoff::AwaitAdopt` after its `TransientBatch` frame reached the durable outbox but before it was acked will, on RESTART, re-drive the frame via boot replay → the dest adopts → the saga completes normally through `BatchAdopted → AwaitRelease → … → Done`. No FSM change was needed for that path.

The **NEVER-restart** residual is the interleave the outbox cannot reach: a source SIGKILLed in `AwaitAdopt` that never comes back. Two coupled defects make this a silent loss today, and neither is fully closable-in-prod without CA-1/L5. R-6d3c lands the **pure-logic** half now (so a future confirm-dead does the RIGHT thing) and honestly leaves the **trigger** half CA-1/L5-gated.

Two defects (both code-cited, both REAL — not roadmap items):

- **Defect A (dest-side silent orphan).** If the orchestrator ever resolves an `AwaitAdopt` saga as source-dead and self-promotes, but a late replay of the batch still reaches the dest, `adopt_transient_batch` (`crates/sim/src/stub.rs:2153-2191`) journals `(transfer, TRANSIENT_BATCH_STEP)=FirstApply` and inserts each item as `TransientStatus::Arriving{batch}` (`stub.rs:2175`). An `Arriving` item is removed ONLY by `on_transient_promote` on a `TransientDrop` (`stub.rs:2249-2252`). `on_transient_abandon` runs on the SOURCE and removes only `Departing`/`Held{Some}` (`stub.rs:2328-2334`); `EmitTransientAbandon` targets `ctx.source` in the executor (`crates/node/src/saga_runtime.rs:867-877`). **There is NO dest-side path that removes an `Arriving` item.** The item is stuck `Arriving` forever — uncounted (`TransientStatus::is_held()==false`, `stub.rs:273-278`), un-rendered, authoritative nowhere.

- **Defect B (saga self-promotes an empty dest AND cannot even fire).** The saga's `(S::BatchHandoff { new_fence, .. }, E::SourceUnreachable)` arm (`crates/sim/src/saga.rs:922-925`) uses the phase-WILDCARD `..` and emits `EmitTransientPromote` for EVERY phase — including `AwaitAdopt`, entered on `CasWon` with an empty egress (`saga.rs:815-823`). Self-promoting from `AwaitAdopt` promotes a dest that never received the batch. The saga.rs:912-921 comment already flags this as INACCURATE-for-AwaitAdopt. Worse, in the current posture the arm cannot even fire in `AwaitAdopt`: `rehome_event_for`'s `BatchHandoff` branch reads `liveness.is_confirmed_dead(ctx.source, now)` (`saga_runtime.rs:1270-1273`), fed only by `record_unreachable` (`saga_runtime.rs:256-270`) on `Inbound::NodeUnreachable{to:source}` (`saga_runtime.rs:1684-1686`), synthesized only by `confirm_and_maybe_bounce` (`crates/io-prod/src/mesh.rs:1449-1470`) which requires a lane toward the source with a non-empty retry (`lane.last_msg_id.is_some()`). In `AwaitAdopt` the orchestrator sends NOTHING to the source (`saga.rs:820-822` has an empty vec), so no such lane exists ⇒ no `NodeUnreachable{source}` ever accrues ⇒ `is_confirmed_dead(source)` is permanently false ⇒ the saga re-drives forever (`rehome_event_for` returns `Timeout` at `saga_runtime.rs:1281-1284`) — a leaked live saga.

## 1. THE DEST SIDE (CRITICAL-2) — new `InterShardFlow::TransientDiscard` arm + `on_transient_discard`

### 1.1 Wire arm (`crates/wire/src/intershard.rs`)

Add a const beside the transient step ids (after `RE_HOME_STEP`, `intershard.rs:84`):
```rust
/// D-6 #1 NEVER-restart closure (R-6d3c): the orchestrator→DEST discard command — the source died
/// in AwaitAdopt (pre-adopt) so the batch is being counted lost, but a late outbox replay could still
/// insert Arriving{batch} at the dest; this REMOVES any such item AND poisons TRANSIENT_BATCH_STEP so
/// a later replayed adopt is AlreadyApplied. Journaled idempotent by (transfer, TRANSIENT_DISCARD_STEP).
pub const TRANSIENT_DISCARD_STEP: u32 = 17;
```
(17 is the next free id — `RE_HOME_STEP` is 16. Verified disjoint from 0–6 route-swap + 7–16 state/transient/rehome steps.)

Add the arm to the enum, APPENDED after `ReHome` (`intershard.rs:165`) to preserve every existing postcard discriminant:
```rust
    /// Orchestrator → DEST shard (R-6d3c): the source died in AwaitAdopt (PRE-adopt) so the batch is
    /// counted lost; REMOVE any Arriving{batch==transfer} item AND poison (transfer, TRANSIENT_BATCH_STEP)
    /// so a late outbox replay's adopt is AlreadyApplied (never re-inserts). Reuses `TransientHandoff`
    /// (shares the shape + one classification arm with the other transient handoff commands, DRY).
    /// Side-effecting, ack-FREE (the resolving saga is terminal); idempotent by (transfer,
    /// TRANSIENT_DISCARD_STEP). APPENDED (preserves every existing postcard discriminant).
    TransientDiscard(TransientHandoff),
```

**Landing it through the frozen closed-set (all three gates):**
- **`effect_class`** (`intershard.rs:288-296`): add `| InterShardFlow::TransientDiscard(h)` to the existing 4-arm `TransientRelease | TransientDrop | ReleaseComplete | TransientAbandon` group → SideEffecting/`TransferStep{h.transfer, h.step_id}`. No new arm body.
- **`durability_class`** (`intershard.rs:340-352`): add `| InterShardFlow::TransientDiscard(_)` to the big `ReDriven` group. It is orchestrator/saga-driven (the resolution injects it, `scan_deadlines` re-drives it) → `ReDriven`, NOT producer-less. This is the load-bearing HR1/D-6 statement: the discard is NOT itself a producer-less flow, so it does NOT grow the outbox set.
- **`arm_tripwire`** (`crates/wire/tests/intershard_closed.rs:226-242`): add `| InterShardFlow::TransientDiscard(_)` to the wildcard-free match (forces representation in `every_arm`).
- **`every_arm`** (`intershard_closed.rs:52-217`): add a `TransientDiscard(TransientHandoff{transfer:TransferId(10), step_id:TRANSIENT_DISCARD_STEP, fence:Fence(6)})` entry. This routes it through the postcard roundtrip + effect-class coherence assertions automatically.
- **The golden `durability_class` pin** (`intershard_closed.rs:300-336`): the discard falls into the `_ => FlowDurabilityClass::ReDriven` arm (line 320). **`producer_less.len()` STAYS 2** — assert unchanged (`intershard_closed.rs:331-335`). This is the proof the new arm does NOT expand the producer-less-reliable set.
- The nested `payload_tripwire`/`ghost_tripwire` (`intershard_closed.rs:256-269`) are UNTOUCHED — `TransientDiscard` is an outer arm carrying the existing `TransientHandoff`, not a new `TransitionPayload`/`GhostFlow` variant, so those guards are correctly not involved.

### 1.2 Dest handler `on_transient_discard` (`crates/sim/src/stub.rs`)

Model on `on_transient_abandon` (`stub.rs:2318-2349`) but: (a) it runs on the DEST, (b) removes `Arriving{batch}` (not `Departing`/`Held{Some}`), (c) POISONS the adopt, (d) buckets loss by kind into the SAME `transients_lost_in_handover` budget (`stub.rs:571`) + a new named `transients_discarded_source_crash` counter (mirroring `transients_departure_cancelled`, `stub.rs:577`).

```rust
/// DEST — R-6d3c NEVER-restart closure: on `TransientDiscard` (the source died in AwaitAdopt, PRE-adopt,
/// so the batch is being counted lost) REMOVE any Arriving{batch==transfer} item as an ACCOUNTED loss AND
/// POISON (transfer, TRANSIENT_BATCH_STEP) — so a LATE outbox replay of the batch adopts as AlreadyApplied
/// and never re-inserts an orphan (the exact silent-loss interleave). Journaled idempotent by
/// (transfer, TRANSIENT_DISCARD_STEP): a redelivery short-circuits, so the loss counts EXACTLY once.
fn on_transient_discard(
    discard: TransientHandoff,
    owned: &mut OwnedTransients,
    applied: &mut AppliedSteps,
    stats: &mut StubStats,
) {
    match applied.journal_step(discard.transfer, TRANSIENT_DISCARD_STEP) {
        StepOutcome::FirstApply => {
            // POISON the adopt: record (transfer, TRANSIENT_BATCH_STEP) so a late replayed
            // adopt_transient_batch is AlreadyApplied (never re-inserts). The return value is ignored —
            // we only need the row present. (If the adopt already ran, this is a harmless no-op insert.)
            let _ = applied.journal_step(discard.transfer, TRANSIENT_BATCH_STEP);
            let mut to_remove: Vec<EntityId> = Vec::new();
            for (entity, t) in owned.0.iter() {
                if let TransientStatus::Arriving { batch } = t.status
                    && batch == discard.transfer
                {
                    to_remove.push(*entity);
                }
            }
            for entity in to_remove {
                owned.0.remove(&entity);
                stats.transients_discarded_source_crash += 1;
                if let Ok(kind) = EntityKind::from_tag(entity.kind_tag()) {
                    *stats.transients_lost_in_handover.entry(kind).or_insert(0) += 1;
                }
            }
        }
        StepOutcome::AlreadyApplied => stats.transient_release_noop += 1,
    }
}
```

**Idempotence + adopt-poison are the two properties that close the orphan** and are exhaustively covered:
1. discard-BEFORE-adopt (the target interleave): poisons `TRANSIENT_BATCH_STEP`; a following `adopt_transient_batch` hits `AlreadyApplied` (`stub.rs:2163,2181`) → increments `transients_adopt_redelivered`, inserts NOTHING. No orphan.
2. discard-AFTER-adopt: removes the extant `Arriving{batch}` items, buckets loss. No orphan.
3. discard redelivered: `AlreadyApplied` short-circuit (loss counts once).

**Dispatch** (`crates/sim/src/stub.rs:2471-2474`, in `on_directory_reply`): add an arm mirroring `TransientAbandon` — ack-free, since the resolving saga is terminal:
```rust
        Ok(InterShardFlow::TransientDiscard(discard)) => {
            on_transient_discard(discard, owned_transients, applied, stats);
            return;
        }
```
No `Ok(_) => return` change needed; the new arm is matched before the catch-all (`stub.rs:2525`).

### 1.3 New counter (`crates/sim/src/stub.rs` StubStats, ~line 577)
```rust
    /// R-6d3c — transients DISCARDED at the dest by `on_transient_discard` (the source died in AwaitAdopt
    /// pre-adopt, so a late-replayed Arriving copy is removed as an ACCOUNTED loss + the adopt is poisoned).
    /// 0 on every happy path and on a RESTART recovery (which adopts normally); `> 0` is the never-restart
    /// cell's anti-vacuity proof. Each increment ALSO feeds `transients_lost_in_handover` (the SAME budget).
    pub transients_discarded_source_crash: u64,
```

## 2. THE SAGA SIDE (CRITICAL-3, pure Tier-A FSM) — the BatchHandoff phase-split

### 2.1 New event + action (`crates/sim/src/saga.rs`)

Event (after `SourceUnreachable`, `saga.rs:509`):
```rust
    /// R-6d3c — the SOURCE is confirmed unreachable while the saga is in `BatchHandoff::AwaitAdopt`
    /// (PRE-adopt: the dest never received the batch). Distinct from `SourceUnreachable` (post-adopt,
    /// self-promote): this resolves as an ACCOUNTED LOSS — discard-to-dest + count — NEVER a self-promote
    /// of an empty dest. Terminal on first fire.
    SourceUnreachablePreAdopt,
```

Action (after `EmitTransientAbandon`, `saga.rs:615`):
```rust
    /// R-6d3c — tell the DEST to DISCARD any late-replayed Arriving copy of this batch + poison its adopt
    /// (the source died pre-adopt). Target `SagaCtx.dest` + step `TRANSIENT_DISCARD_STEP`. A PROPER new
    /// action (NOT the source-addressed `EmitTransientAbandon`, NOT `EmitTransientPromote` of an empty dest).
    EmitTransientDiscard { fence: Fence },
    /// R-6d3c — count the batch lost because the SOURCE crashed pre-adopt (distinct from the dest-dead
    /// `EmitTransientAbandon` loss counter so the two failure causes are never conflated in the ledger).
    CountBatchLostSourceCrash,
```

### 2.2 The phase-split (`crates/sim/src/saga.rs`, replacing the wildcard at 922-925)

Split the ONE wildcard arm into a post-adopt arm (unchanged behaviour) + a pre-adopt arm. Keep `DestUnreachable` unchanged. The pre-adopt resolution is terminal-on-first-fire (straight to `Done`, never back to `BatchHandoff` — structurally no forever-bounce, mirroring the existing comment at `saga.rs:908`):
```rust
        // POST-ADOPT phases (AwaitRelease/AwaitPromote/AwaitComplete): the dest provably holds the batch
        // and the go-token is the SOLE promote authority → self-promote (zero loss). UNCHANGED.
        (
            S::BatchHandoff {
                phase: P::AwaitRelease | P::AwaitPromote | P::AwaitComplete,
                new_fence,
            },
            E::SourceUnreachable,
        ) => (
            S::Done { new_fence },
            vec![A::EmitTransientPromote { fence: new_fence }, A::Tombstone],
        ),
        // PRE-ADOPT (AwaitAdopt): the dest never received the batch → NEVER self-promote an empty dest.
        // Resolve as an accounted loss: discard any late-replayed Arriving copy at the dest + count + done.
        (
            S::BatchHandoff {
                phase: P::AwaitAdopt,
                new_fence,
            },
            E::SourceUnreachablePreAdopt,
        ) => (
            S::Done { new_fence },
            vec![
                A::EmitTransientDiscard { fence: new_fence },
                A::CountBatchLostSourceCrash,
                A::Tombstone,
            ],
        ),
        (S::BatchHandoff { new_fence, .. }, E::DestUnreachable) => (
            S::Done { new_fence },
            vec![A::EmitTransientAbandon { fence: new_fence }, A::Tombstone],
        ),
```

Because `step` is a total match with an idempotent no-op catch-all, a stale `SourceUnreachable` reaching `AwaitAdopt` (or `SourceUnreachablePreAdopt` reaching a post-adopt phase) falls through to the no-op — no panic, at-least-once safe. The `E::SourceUnreachable` reaching `AwaitAdopt` is thus a documented no-op (the producer never emits it there — see §2.3).

### 2.3 Producer: `rehome_event_for` phase-discrimination (`crates/node/src/saga_runtime.rs:1270-1285`)

The `BatchHandoff` branch must emit `SourceUnreachablePreAdopt` for `AwaitAdopt` and `SourceUnreachable` for the post-adopt phases, and (CA-1/L5-gated, §3) must be BUDGET-gated so a restart-within-budget wins. The pure discrimination is landable now:
```rust
        SagaState::BatchHandoff { phase, .. } => {
            if liveness.is_confirmed_dead(ctx.source, now) {
                // budget-gate on abort_deadline_ticks so a source that RESTARTS within budget delivers via
                // its outbox replay FIRST (the two recoveries race; budget picks the restart winner).
                let observed = *dead_observed_since.get_or_insert(now);
                if now.0.saturating_sub(observed.0) >= tuning.abort_deadline_ticks {
                    *dead_observed_since = None;
                    match phase {
                        BatchHandoffPhase::AwaitAdopt => SagaEvent::SourceUnreachablePreAdopt,
                        _ => SagaEvent::SourceUnreachable,
                    }
                } else {
                    SagaEvent::Timeout // cheap re-drive while the restart-race budget accrues
                }
            } else if liveness.is_confirmed_dead(ctx.dest, now) {
                // (unchanged dest-dead ladder, saga_runtime.rs:1274-1280)
                ...
            } else {
                *dead_observed_since = None;
                SagaEvent::Timeout
            }
        }
```
Note: this ADDS a budget-gate to the source-dead path (today source-dead in BatchHandoff self-promotes immediately, `saga_runtime.rs:1271-1273`). Sound because the source-dead path in AwaitAdopt is now a DESTRUCTIVE accounted-loss (must give the restart a chance); the post-adopt source-dead path becomes budget-gated too, which is strictly safer (the existing `scan_deadlines_resolves_a_batch_handoff_with_a_dead_participant` test at `saga_runtime.rs:4293` and `scan_deadlines_self_promotes_a_demoting_saga_...` at 3806 will need their tick offsets bumped past `abort_deadline_ticks`, or the tuning set so the first fire is already past budget).

### 2.4 Executor + counter (`crates/node/src/saga_runtime.rs`)

Executor arm (in `run_to_quiescence`, beside `EmitTransientAbandon` at `saga_runtime.rs:867-877`):
```rust
                SagaAction::EmitTransientDiscard { fence } => {
                    outbox.push_flow(
                        ctx.dest,
                        MsgClass::Saga,
                        &InterShardFlow::TransientDiscard(TransientHandoff {
                            transfer: ctx.transfer,
                            step_id: TRANSIENT_DISCARD_STEP,
                            fence,
                        }),
                    );
                }
                SagaAction::CountBatchLostSourceCrash => {} // counted in scan_deadlines pre-delivery (below)
```
New runtime counter (`saga_runtime.rs:360-361` region) + accessor (mirroring `source_unreachable_resolutions`, `saga_runtime.rs:526`):
```rust
    batch_lost_source_crash: u64,
```
Counted in `scan_deadlines` before delivery (`saga_runtime.rs:1532-1536`), terminal-on-first-fire so it counts once:
```rust
            SagaEvent::SourceUnreachablePreAdopt => runtime.batch_lost_source_crash += 1,
```

## 3. REACHABLE-NOW vs CA-1/L5-GATED — the decisive scope resolution

**What CLOSES the never-restart silent-loss AS PURE LOGIC, landable NOW (R-6d3c-1 + -2):**
- The dest-side `TransientDiscard` arm + `on_transient_discard` handler (adopt-poison + Arriving removal + loss bucket). Directly unit-testable with NO transport: call `on_transient_discard` then `adopt_transient_batch` and assert zero orphan. **This is the actual cure for Defect A** and it lands complete now.
- The saga phase-split: `SourceUnreachablePreAdopt` event + `EmitTransientDiscard`/`CountBatchLostSourceCrash` actions + the AwaitAdopt-vs-post-adopt arms. **This makes a future confirm-dead do the RIGHT thing** (discard-to-dest, not self-promote-empty) — Defect B's *behaviour* is fixed now.
- The `rehome_event_for` phase-discrimination + budget-gate + the `batch_lost_source_crash` counter.

Testable NOW with zero CA-1 dependency: (a) a DIRECT `deliver(runtime, .., SourceUnreachablePreAdopt)` into an `AwaitAdopt` saga (bypassing the trigger) asserts the discard egress + counter + `Done`; (b) an FSM `assert_eq!` unit test of the split; (c) the composed handler test proving no orphan survives.

**What stays OPEN-until-CA-1/L5 (the TRIGGER that MAKES the arm reachable in prod):**
- The `AwaitAdopt` RE-SOLICIT egress — an idempotent orchestrator→source re-prompt so a dead source accrues `NodeUnreachable{source}` and `is_confirmed_dead(source)` can fire. This is REQUIRED (not redundant), because `AwaitAdopt` is entered with an empty egress (`saga.rs:820-822`) and `confirm_and_maybe_bounce` (`mesh.rs:1449-1470`) only bounces a lane with a non-empty retry. WITHOUT any orchestrator→source lane, `is_confirmed_dead(source)` is unreachable in `AwaitAdopt` (`saga_runtime.rs:1271`). Adding the re-solicit needs the 2d FSM-field template (DEFERRED.md:280-281) AND depends on CA-1 (reply-on-connection / addr-reread; `mesh.rs:2890-2909` `#[ignore]` red-guard) + M3 (durable monotone incarnation, already landing in R-6) so a source rescheduled to a new addr is re-dialed and its death confirmed.

**Honest verdict:** R-6d3c NOW makes the never-restart loss **impossible-if-detected** and **accounted** — the dest can never strand an `Arriving` orphan (the discard poisons the adopt regardless of replay timing), and the saga resolves to a bounded terminal with a loud counter instead of self-promoting an empty dest or parking forever. But the never-restart loss is not **auto-detected in prod** until the CA-1/L5-gated re-solicit egress + confirm-dead trigger land. So D-6 #1 is: **RESTART case = 🟩 closed (R-6d3b-2b); NEVER-restart case = the CORRECTNESS (discard-not-strand, accounted-loss) is 🟩 closed by R-6d3c, the DETECTION-in-prod stays 🟧 open-until-CA-1/L5.** This is a genuine narrowing — the silent-loss becomes an accounted-loss-on-detection, and the only gap is the detection trigger, which is a pre-existing HARD deploy precondition already tracked (DEFERRED.md:957, R-6 + CA-1).

## 4. TESTS + Tier-A 100% coverage plan

**Wire (`crates/wire/tests/intershard_closed.rs` + in-module `intershard.rs` tests):**
- `every_arm` gains the `TransientDiscard` entry → `every_arm_roundtrips_postcard_and_classifies_coherently` (`:272`) covers postcard roundtrip + effect-class coherence for free.
- `durability_class_pins_the_producer_less_reliable_set` (`:301`) — the `_ => ReDriven` arm covers the discard; assert `producer_less.len() == 2` UNCHANGED (the load-bearing pin).
- In-module `g_sealed_effect_classes` (`intershard.rs:683`): add the discard to the `TransientRelease|TransientDrop|ReleaseComplete` loop (`:889-924`) so the shared classification arm is exercised for it.
- `arm_tripwire` + `effect_class`/`durability_class` exhaustive matches: adding the arm without classifying fails to compile (the conformance guarantee).

**Dest (`crates/sim/src/stub.rs` unit tests, modeled on `on_transient_abandon_drops_the_batch_as_accounted_loss_and_is_idempotent` at `:3737`):**
- `on_transient_discard_removes_arriving_poisons_adopt_and_is_idempotent`: seed an `Arriving{batch}` item; discard removes it + buckets loss + increments `transients_discarded_source_crash`; a second discard is `AlreadyApplied` (loss counts once). Split every `assert!` into `assert_eq!` per HR5(d).
- `discard_before_adopt_poisons_so_a_late_replay_never_orphans` (the target interleave): discard on an EMPTY `OwnedTransients` (poison only) → then `adopt_transient_batch(same transfer)` → assert `owned.0.is_empty()` AND `transients_adopt_redelivered == 1` (adopt hit `AlreadyApplied`). This is the direct proof Defect A is closed.
- `on_transient_discard_on_a_settled_or_departing_item_leaves_it` (the false arm of the `Arriving` match is covered — a `Held`/`Departing` item is NOT removed; needed for 100% branch).

**Saga FSM (`crates/sim/src/saga.rs`, `assert_eq!` on `(SagaState, Vec<SagaAction>)` per HR5(d)):**
- MODIFY `batch_handoff_resolves_terminally_on_a_dead_participant` (`saga.rs:1528-1568`): the `AwaitAdopt` phase must now be exercised with `SourceUnreachablePreAdopt` (→ `Done` + `[EmitTransientDiscard, CountBatchLostSourceCrash, Tombstone]`), while `AwaitRelease/AwaitPromote/AwaitComplete` keep `SourceUnreachable` (→ `EmitTransientPromote`). Split the loop: AwaitAdopt uses the new event, the three post-adopt phases keep the old event.
- ADD `await_adopt_source_unreachable_is_a_noop` and `post_adopt_source_unreachable_pre_adopt_is_a_noop` (the crossed events fall through the catch-all) — covers the total-function no-op arms so no uncoverable region.
- Every new match arm is a straight-line `(state, vec![...])` — no `?`/closure inside a generic; HR5(a) trivially satisfied (the FSM is monomorphic).

**Saga runtime (`crates/node/src/saga_runtime.rs`):**
- `rehome_event_for_await_adopt_source_dead_emits_pre_adopt_past_budget_else_redrives` (mirror `rehome_event_for_promoting_dead_dest_rehomes_past_budget_else_redrives` at `:3933`): AwaitAdopt + dead source → `Timeout` before budget, `SourceUnreachablePreAdopt` past `abort_deadline_ticks`; a post-adopt phase past budget → `SourceUnreachable`.
- MODIFY `scan_deadlines_resolves_a_batch_handoff_with_a_dead_participant` (`:4293`) + `scan_deadlines_self_promotes_a_demoting_saga_with_a_confirmed_dead_source` (`:3806`): the source-dead BatchHandoff path is now budget-gated — advance `now` past `abort_deadline_ticks` (or the resolution won't fire on the first scan). Add a `batch_lost_source_crash()` accessor assertion (== 1 after the AwaitAdopt resolution; == 0 elsewhere).
- Executor arm `EmitTransientDiscard` covered by a composed `deliver(AwaitAdopt, SourceUnreachablePreAdopt)` test asserting `outbox` carries a `TransientDiscard` to `ctx.dest`; `CountBatchLostSourceCrash` executor arm is a straight-line no-op (covered by the same delivery).

**Coverage discipline (HR5):** every new arm is monomorphic straight-line; the `Arriving` match's false arm, the discard journal `AlreadyApplied` arm, the `EntityKind::from_tag` Ok/Err arms (mirror `on_transient_abandon`'s existing pattern at `stub.rs:2342`) all get an explicit case. `just coverage-fast` must stay 100% region+branch on vd-sim, vd-node; vd-wire 100%.

## 5. FILES/FUNCTIONS TOUCHED PER SUB-SLICE

**R-6d3c-1 — the wire arm (vd-wire, Tier-A 100%):**
- `crates/wire/src/intershard.rs`: `TRANSIENT_DISCARD_STEP` const; `InterShardFlow::TransientDiscard` variant (appended); `effect_class` group (`:288-296`); `durability_class` group (`:340-352`); in-module `g_sealed_effect_classes` loop.
- `crates/wire/tests/intershard_closed.rs`: `arm_tripwire` (`:226-242`); `every_arm` (`:52-217`); the `durability_class` golden pin (`:310-336`, assert `producer_less.len()==2`).

**R-6d3c-2 — the dest handler (vd-sim, Tier-A 100%):**
- `crates/sim/src/stub.rs`: `on_transient_discard` (new, modeled on `on_transient_abandon` `:2318`); `StubStats.transients_discarded_source_crash` (`~:577`); dispatch arm in `on_directory_reply` (`:2471` region); import `TRANSIENT_DISCARD_STEP` (`:32-33`); the 3 unit tests.

**R-6d3c-3 — the saga FSM + producer (vd-sim + vd-node, Tier-A 100%):**
- `crates/sim/src/saga.rs`: `SagaEvent::SourceUnreachablePreAdopt` (`:509`); `SagaAction::EmitTransientDiscard`/`CountBatchLostSourceCrash` (`:615`); the BatchHandoff phase-split arms (`:922-925`); MODIFY `batch_handoff_resolves_terminally_on_a_dead_participant` + add the two no-op tests.
- `crates/node/src/saga_runtime.rs`: `rehome_event_for` BatchHandoff branch (`:1270-1285`) — phase-discrimination + budget-gate; `EmitTransientDiscard`/`CountBatchLostSourceCrash` executor arms (`:867` region); `batch_lost_source_crash` field + accessor; `scan_deadlines` counter (`:1533`); MODIFY the two existing budget-affected tests; add the two new producer tests.

**CA-1/L5-GATED (NOT in R-6d3c — recorded in DEFERRED.md as the remaining open half of D-6 #1):**
- The `AwaitAdopt` re-solicit egress (a new idempotent orchestrator→source re-prompt on the `AwaitAdopt` Timeout, 2d FSM-field template) + the confirm-dead-toward-source trigger. Depends on CA-1 (`mesh.rs:2890` red-guard) + M3.
- The `sigkill_source_in_await_adopt_never_restarts` real-QUIC process proof (bins, extends `orchestrator_crash.rs` `store-test-hooks` sentinel + `mesh_under_loss.rs`) — the composed gate can only assert the FULL prod closure once the trigger is reachable; R-6d3c ships the direct-inject FSM/handler tests as the reachable-now proof.

## 6. API insufficiency flags

- **None blocking.** `AppliedSteps::journal_step` (`stub.rs:615`) is sufficient to poison `TRANSIENT_BATCH_STEP` (its insert-or-report semantics make the `let _ =` poison correct). `TransientHandoff` (`intershard.rs:449-454`) already carries `(transfer, step_id, fence)` — no new payload type needed (DRY: the discard shares it with the other 4 handoff arms).
- **Minor:** the `rehome_event_for` source-dead path in BatchHandoff currently self-promotes on the FIRST confirm (`saga_runtime.rs:1271-1273`, no budget). R-6d3c ADDS a budget-gate to the source-dead path (needed so a restart-within-budget wins the race). This is a behaviour change to the POST-adopt source-dead path too (now budget-gated) — strictly safer, but it requires bumping the tick offsets in two existing tests (`:3806`, `:4293`). Flagged as an intentional, covered change, not a regression.
