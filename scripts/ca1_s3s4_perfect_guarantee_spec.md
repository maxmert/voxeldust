# CA-1 S3/S4 — the achievable hard over-discard guarantee (SOUND_TO_IMPLEMENT)

Consolidates 4 design rounds + the perfect-guarantee adjudication (wf_0fea488d, NEEDS_ANOTHER_ROUND).
User choice: **implement the achievable hard cure now** (dest_adopted latch + dest re-drive + S3),
honestly documented, with the sustained-shed residual + the non-sheddable-ack transport refinement
ledgered for the deploy bundle.

## The problem (retires the R-6d4-F2 tripwire at saga.rs:955 / saga_runtime.rs:1347)

The `(BatchHandoff{AwaitAdopt}, SourceUnreachablePreAdopt)` arm (saga.rs:968) discards a transient batch
as an accounted loss when the SOURCE is confirmed dead PRE-adopt. Today it is INERT (no orch→source
egress in AwaitAdopt, so `is_confirmed_dead(source)` is never reachable → the discard never fires → the
saga would wedge if the source really died pre-adopt). Making it LIVE requires an orch→source probe (S3).
But a naive probe OVER-DISCARDS: a LIVE dest that DID adopt, whose `BatchAdopted` ack is lost/in-flight,
is still in AwaitAdopt and would be discarded even though it holds the batch.

## The cure decomposes into three orthogonal parts

- **SAFETY (no over-discard) — the `dest_adopted` latch.** A monotone `bool` on `LiveSaga`, latched TRUE
  the moment the orchestrator has ANY evidence the dest adopted (a `BatchAdopted` present in the inbox).
  The AwaitAdopt discard fires ONLY when `!dest_adopted`. So the discard is a genuine "the dest never got
  the batch" loss, never a "we haven't heard the ack yet" false loss.
- **LIVENESS (the evidence keeps flowing) — the dest re-drive.** The DEST re-emits `BatchAdopted` every
  tick for each `Arriving{batch}` item it holds (reusing the existing ack emit; NO new wire arm). Because the
  latch is monotone (needs one delivered ack EVER), once any ack has landed no later loss can strand it; the
  only uncovered corner (first-adopt-exactly-at-maturity with THAT ack lost) is a strict sub-case of the
  sustained-shed residual below.

**Post-impl-review fix (folded):** the first cut gated the discard on a bare `!dest_adopted`, which let
source-death PREEMPT the dest-death terminal and WEDGE the both-dead double-crash (dest adopted-then-crashed
+ source dead) into a perpetual `Timeout`→probe loop — a regression (pre-CA-1-S3 that case resolved via the
dest-dead branch). Fixed with a `defer_to_dest` gate in `rehome_event_for`: an adopted `AwaitAdopt` batch
makes the DEST the holder-of-record, so its death ladder (budget-gated `DestUnreachable` abandon) decides,
and the source-death resolutions apply only when `!defer_to_dest`.
- **DETECTION (the dead-source trigger) — S3 `ReSolicitBatch`.** A new orch→SOURCE arm emitted every
  AwaitAdopt Timeout. Its DELIVERY makes `is_confirmed_dead(source)` reachable: a dead source ⇒ the send
  fails ⇒ `NodeUnreachable` ⇒ confirm-dead ⇒ (past budget, `!dest_adopted`) the discard fires. A live
  source handles it as a counted no-op (its regular lease-renewal inbound clears any stale evidence).

## THE CRITICAL FIX (corrects the adjudicator): latch in a PRE-SCAN pass, not in `deliver()`

`drive_sagas` runs `scan_deadlines` (saga_runtime.rs:1765) BEFORE the inbox drain (1766+). The
adjudicator's "latch in `deliver()`" is INSUFFICIENT: `deliver()` runs inside the post-scan drain, so a
`BatchAdopted` arriving on the exact budget-maturity tick is latched too late — `scan_deadlines` has
already fired the discard + tombstoned the saga. The latch MUST be set in a **pre-scan pass over the
inbox** (before `scan_deadlines`), so the discard decision sees the adoption evidence that arrived THIS
tick. Combined with the dest re-drive (a `BatchAdopted` is in the inbox EVERY tick the dest holds
Arriving), the discard is suppressed on every such tick ⇒ **zero over-discard**.

Latch site is the pre-scan pass ONLY (single-site). The pre-scan and the drain iterate the SAME
`inbox.0`, so any `BatchAdopted` the drain would process was already seen by the pre-scan → the latch is
set before `deliver()` transitions the phase. No `deliver()`-site latch needed.

RAM-only (like `dead_observed_since`): re-derived `false` on rehydrate. Safe — on rehydrate the liveness
tracker is EMPTY (the D-6 freeze), so `is_confirmed_dead(source)` is false until fresh notices re-accrue;
the dest re-drive re-establishes the latch long before any discard can fire.

## The acknowledged residual (ledgered, NOT closed here)

If the dest→orch lane SHEDS every `BatchAdopted` for the WHOLE abort window (R-4b sustained overload),
no ack lands → the latch never sets → the discard fires. This is fundamental: the orch cannot prove
adoption without receiving one ack. In that corner the dest is anyway likely `is_confirmed_dead(dest)`
(a saturated lane) → the DEST-dead arm fires instead (a different, correct resolution). Absolute-zero
needs a non-sheddable ack (a transport priority/reservation lane) — a bigger transport change, ledgered
in DEFERRED.md for the deploy bundle.

## Implementation bundle

### wire (crates/wire/src/intershard.rs + tests/intershard_closed.rs)
1. `RE_SOLICIT_STEP: u32 = 18` (next free after TRANSIENT_DISCARD_STEP=17).
2. `ReSolicitBatch(TransientHandoff)` arm APPENDED after `TransientDiscard` (discriminant 16 → 17 arms).
   Doc: orch→SOURCE AwaitAdopt liveness probe; its delivery-failure drives `is_confirmed_dead(source)`;
   a live source handles it as a counted no-op.
3. `effect_class`: join the `TransientHandoff` group ⇒ `SideEffecting{TransferStep(transfer, step_id)}`.
4. `durability_class`: `ReDriven` (orch-emitted, scan re-drives it every AwaitAdopt Timeout — NOT
   producer-less; the golden pin's producer-less set stays exactly {Ghost::Despawn, TransientBatch}).
5. Conformance: add to `every_arm()` + `arm_tripwire`; bump the "16-arm" header to 17.

### sim FSM (crates/sim/src/saga.rs)
6. ADD arm `(BatchHandoff{AwaitAdopt, new_fence}, Timeout) => (state, vec![A::EmitReSolicit{fence}])`
   (an ADD before the terminal catch-all; the 906-909 comment about AwaitAdopt having no egress is
   updated — it now emits the S3 probe). New `SagaAction::EmitReSolicit{fence}`.
7. Retire the F2 tripwire prose at saga.rs:955 (the over-discard is now guarded by `dest_adopted`).

### sim stub (crates/sim/src/stub.rs)
8. `redrive_pending_adoptions` DEST system (added to `register_stub_shard` chain after `process_inbound`,
   beside `emit_transient_batch`): for each DISTINCT `Arriving{batch}` in `OwnedTransients`, re-emit
   `TransferAck::BatchAdopted{transfer_id: batch, step_id: TRANSIENT_BATCH_STEP}` to the orchestrator.
   Gated PURELY on Arriving-presence (adopt is lease-free — NO authority/lease gate). `StubStats`
   counter `batch_adopts_redriven`.
9. `ReSolicitBatch` source handler in the inbound dispatch: a counted no-op (`stats.re_solicits_received
   += 1`), replacing the silent `Ok(_) => return` for this arm.

### node runtime (crates/node/src/saga_runtime.rs)
10. `dest_adopted: bool` on `LiveSaga` (init `false` at every construction site; RAM-only, not in
    `SagaSnapshot`).
11. PRE-SCAN latch pass in `drive_sagas` BEFORE `scan_deadlines`: decode Saga-class inbound, for each
    `TransferAck::BatchAdopted{transfer_id}` set `sagas[transfer_id].dest_adopted = true`.
12. `EmitReSolicit` executor: push `ReSolicitBatch(TransientHandoff{ctx.transfer, RE_SOLICIT_STEP,
    new_fence})` toward `ctx.source`.
13. Gate `rehome_event_for`'s AwaitAdopt arm (saga_runtime.rs:1356): `SourceUnreachablePreAdopt` fires
    ONLY when `!dest_adopted` (pass the latch in); `dest_adopted ⇒ Timeout` (keep re-driving/probing).

### io-prod boot (crates/io-prod/src/runtime.rs) — the MEDIUM
14. Boot assertion: the production/cloud shard profile MUST set `self_fence_grace_ticks > 0` (so a
    permanently-orphaned `Arriving` item is eventually self-fenced → the dest re-drive terminates). Tests
    run finite ticks so grace==0 stays valid there. If no clean site, LEDGER instead.

## Coverage (HR5, Tier-A 100% region+branch)
- Direct-inject FSM test: `(AwaitAdopt, Timeout) => EmitReSolicit`.
- Runtime test: pre-scan latch suppresses the discard when a `BatchAdopted` is in the inbox on the
  maturity tick (the exact race); and the discard STILL fires when `!dest_adopted` (genuine pre-adopt
  source death).
- Stub test: `redrive_pending_adoptions` re-emits for an Arriving batch; no-op when none Arriving.
- Stub test: the `ReSolicitBatch` counted no-op handler.
- Wire: the conformance additions carry through the existing surface gate.
