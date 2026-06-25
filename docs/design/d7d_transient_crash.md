# D-7d Slice 1 — saga-owned transient handoff tail + dead-aware crash resolution

LOCKED design (workflow `wf_f82daa5e-56b`: ground → 2 designs [A saga-owned re-drive · B shard-local
detector] → 4 adversarial skeptics default-refute → synthesize). Builds on D-7a/b/c (HEAD `69cffc6`).
The synthesis took A's saga-owned spine (ONE recovery producer, no second machine — HR2/HR3) and grafted
B's observable counters + self-resolve framing. Every surviving BLOCKER/MAJOR is resolved in the steps.

## The gap this closes (the P3 kill-9 headline for debris)

Today the transient saga TOMBSTONES at the go-token commit (`saga.rs:483` — `(BatchCommitting, CasWon) →
Done{Tombstone}`), so the adopt-before-drop choreography (`TransientRelease → DropApplied →
TransientDrop/promote → DropApplied → ReleaseComplete`) runs with NO owner. A **permanent kill mid-handoff**
strands an item that no producer resolves until the whole realm self-fences (a soak/kill-9 storm surfaces
it as silent loss or a wedged item). D-7d makes every stranded transient land DETERMINISTICALLY in exactly
one of {held-at-the-survivor, bucketed-as-accounted-loss-within-budget}.

## Resolved open questions (recommendations locked, code-grounded + standing-rule-dictated)

1. **Dead-dest resolution = DROP-WITHIN-BUDGET** (not re-home) for ALL transient kinds. Unanimous (both
   designs + 4 skeptics); reuses the D-7b.3 loss-budget machinery; debris is invisible at gameplay scale;
   re-home is the durable-only answer (gated on D-3). Reversible: a future Durable-transient hybrid adds
   re-home then. NOT a one-way door.
2. **The redrive-deadline bound = DERIVED** in the config composer from the existing
   `max_fabric_delay_ticks + max_absent_ticks` (worst-case healthy round-trip `1 + 2·delay + 2·absent`,
   with a documented safety multiplier surfaced as a `TransferTuning` field) — no inline literal
   (no-magic-numbers). The `CrashResurrect` control cell guards the lower edge.
3. **Load posture** — Slice 1 lands the single-item cells + RE-VERIFIES the D-7c burst gates under the new
   `O(concurrent-batches)` live-saga lifetime (`live_sagas==0` only AFTER the handoff completes; count is
   `O(batches)` not `O(items)`). The NEW burst/over-budget cells (1000-item self-promote, cluster
   `LostOverBudget`) go to Slice 2.

## The mechanism (saga-owned post-commit tail, the transient twin of the durable Demoting/Promoting)

`(BatchCommitting, CasWon{new_fence})` no longer tombstones — it enters a **not-yet-tombstoned
`BatchHandoff{phase, source, dest, new_fence}` tail** (NO new commit; the go-token already committed at
this edge). The ONE existing `scan_deadlines` producer re-drives it. Phases + ack edges:

- `AwaitAdopt`  + `BatchAdopted`      → `AwaitRelease`  (emit `TransientRelease` to source)
- `AwaitRelease`+ `SourceDropApplied` → `AwaitPromote`  (emit `TransientDrop`/promote to dest)
- `AwaitPromote`+ `DestDropApplied`   → `AwaitComplete` (emit `ReleaseComplete` to source)
- `AwaitComplete`+`SourceRetired`     → `Done{new_fence}` + `[Tombstone, GcGoToken{batch}]`
- any phase + `Timeout`               → re-emit the current phase's pending egress idempotently (journaled)

Dead-resolution edges (terminal on FIRST fire — directly to `Done+Tombstone+GcGoToken`, never back to
`BatchHandoff`, structurally avoiding the D-37 forever-bounce):

- any `BatchHandoff` + `SourceUnreachable` → (phase≥AwaitAdopt: the dest already adopted) emit
  promote-direct-to-dest → `Done` → **`BatchCommittedAt{DEST}`, zero loss** (go-token is the SOLE promote
  authority — legitimate even with the source dead; no second commit, no forgery).
- any `BatchHandoff` + `DestUnreachable` → emit `TransientAbandon` to source → `Done` →
  **`BatchDroppedWithinBudget{kind}`, lost==1≤budget** (the retained `Departing` copy is what makes the
  loss COUNTABLE, not silent).

`SourceUnreachable`/`DestUnreachable` are injected by `scan_deadlines` ONLY when a `BatchHandoff`
participant is in `dead_participants` **AND** the redrive deadline elapsed (CRITIQUE-3 fix: a prod
`NodeUnreachable` means "one send failed", incl. a 20s idle reap of a live peer — NOT "dead"; the
deadline-AND-notice gate lets a blipped-then-redialed peer clear itself via its next ack before resolution
fires). Documented kill-only-today; D-3 lease-lapse is the real crash-vs-partition discriminator.

## Implementation deviations from the synthesis (correctness-improving, recorded)

Two synthesis details were changed during implementation because they conflicted with existing
invariants:

1. **NO `GcGoToken` in Slice 1.** The synthesis put `GcGoToken{batch}` on the terminal Done edges. But
   GC'ing the go-token at Done would (a) break the D-7c G-TIER gate (`batch_goes == [(batch, fence)]`
   asserted at quiescence — the saga reaches Done WITHIN the window) and (b) break
   `TRANSIENT-AUTHORITY-HELD` for the SOURCE-kill outcome (the dest's promoted item stays `Held` and the
   oracle cross-checks it against a live go-token). So ALL go-token GC stays deferred to D-7d Slice 2
   (the drop-completion-signal-gated GC, which the deferred-list already owns) — Slice 1 adds no
   `SagaAction::GcGoToken`. The unbounded-ledger concern is unchanged (already documented on `batch_goes`).
2. **Resolution counters live on the ORCHESTRATOR (`SagaRuntimeRes`), not `StubStats`.** The synthesis put
   `transients_self_promoted` on the dest's `StubStats`, but the dest CANNOT distinguish a self-promote
   from a normal promote (a `TransientDrop` is a `TransientDrop`). The resolution is DECIDED at the
   orchestrator, so the anti-vacuity observables (`source_unreachable_resolutions` /
   `dest_unreachable_resolutions`, incremented in `scan_deadlines`) live there. The source-side
   `transients_departure_cancelled` (on `StubStats`) is kept — the source DOES know it abandoned.

## Steps (S1–S8)

- **S1** `sim/saga.rs` — replace the e0 tombstone with the `BatchHandoff` tail; add `BatchHandoffPhase`,
  the choreography + dead-resolution + Timeout arms, `SagaAction::GcGoToken{batch}`. Durable path
  byte-untouched. All branching in monomorphic step arms (HR5). Update the module doc.
- **S2** `node/saga_runtime.rs` — route `BatchAdopted`/`Source|DestDropApplied`/`SourceRetired` through
  `deliver()` to advance the live saga (handle_batch_adopted/handle_drop_applied are read-only today);
  move egress into the matching `SagaAction` executors (preserve FF-1: one feedback event per ack).
  Distinguish source-vs-dest `DropApplied` by the existing `RELEASE_STEP`/`DROP_STEP` id on the ack.
  Unit: a 2nd `Timeout` on an already-resolved (removed) saga is an idempotent no-op.
- **S3** `node/saga_runtime.rs` — `dead_participants: BTreeSet<NodeId>` + an `Inbound::NodeUnreachable`
  arm inserting the dead `to` (no MsgId correlation — `app.rs:198` discards the flush MsgId). A
  successful ack clears the node. `scan_deadlines`: BatchHandoff due AND source/dest∈dead → inject
  Source/DestUnreachable; else re-emit phase egress. `deadline_for(BatchHandoff)` = the derived
  redrive deadline (Q2), NOT `u64::MAX`.
- **S4** `wire/intershard.rs` + `sim/stub.rs` — `InterShardFlow::TransientAbandon` (reuses the
  `TransientHandoff` struct shape; one-line `|` in the `effect_class` arm; a PROPER new typed action,
  NOT a repurposed `ReleaseComplete`). `TRANSIENT_ABANDON_STEP=14`. `on_transient_abandon` stub handler
  (sibling of `on_release_complete`): journal-gate, remove this batch's `Departing{b}`/`Held{Some(b)}`
  items, bucket each into `transients_lost_in_handover` by `EntityKind::from_tag` (the DETERMINISTIC
  route into the loss counter). New counter `transients_departure_cancelled`.
- **S5** `sim/stub.rs` — dest self-promote on a dead source: the `SourceUnreachable` resolution emits a
  promote DIRECTLY to the live dest, which runs the EXISTING `on_transient_promote` body gated through
  the `(transfer, DROP_STEP)` journal (a later genuine `TransientDrop` redelivery = no-op). New counter
  `transients_self_promoted` (anti-vacuity: 0 on happy path).
- **S6** `harness/oracle.rs` — `verify_transient_authority_held_excluding(reports, dead)` +
  `verify_transient_conservation_tick_excluding(reports, dead, tick)`; the no-dead fns DELEGATE with an
  empty set (DRY, mirroring `verify_authority_unique`). Land RED-then-GREEN against an explicit
  dead-source-corpse (`Held{Some(b)}`) + live-dest-`Held{None}` fixture, pinned to the exact variant.
  NO new `TransientViolation` arm (a transient has no directory row; `verify_transient_loss_budget` is
  the honest surfaced outcome — D-7d's job is to FEED it deterministically).
- **S7** `tests/src/lib.rs` — `run_transient_fault_scenario` + `assert_transient_end_state` (transient
  sibling of `run_fault_scenario`; the durable driver is directory-shaped to the bone, a transient writes
  ZERO directory rows → a shared fn would need a forbidden match-on-class). REUSE the ~60% shared helpers
  (p2_cluster, p1_client, warmup, seed_transient_crossing, fault_step_until, schedule_crash/kill/
  dead_nodes, Fault/CrashWhen verbatim). `EndState::BatchCommittedAt{node}` + `BatchDroppedWithinBudget
  {kind}`. The class-pick is ONE match at the cell-table call-site, never inside a driver.
- **S8** `tests/tests/p3_transient.rs` — the THREE cells (SOURCE-kill@AwaitRelease → BatchCommittedAt
  {DEST} zero loss; DEST-kill@AwaitPromote → BatchDroppedWithinBudget{Debris} lost==1; SOURCE-
  CrashResurrect@AwaitRelease control → BatchCommittedAt{DEST}, never enters dead_participants → no
  spurious abandon, proving the kill-only gate). RE-VERIFY the D-7c G-TIER/DURABLE-UNAFFECTED gates
  under the new live-saga lifetime (`live_sagas==0` only after e6; count `O(batches)`).

## Conservation argument (every cell — SINGLE-participant kill)

SCOPE (audit D7D-1): the guarantees below hold under a SINGLE-participant kill (the source OR the dest,
not both). The cardinal invariant — no item COUNTED at two LIVE shards (no double-hold, no forgery, no
second commit) — holds UNCONDITIONALLY (the dead-aware twin excludes every corpse). The dual-participant
caveat is below.

Under a single-participant kill the resolution NEVER counts an item at two LIVE shards and routes a
stranded item into exactly one of {held-at-survivor, bucketed-as-loss}; the promote is gated SOLELY on the
kill-only notice (can never fire against a partitioned-but-live source). SOURCE-kill: live counted set `{}`
at kill (S's `Held{Some}` corpse excluded by the dead-aware twin; D's `Arriving` uncounted) → go-token
authorizes `{}→{D}`, journaled so a stray `TransientDrop` is a no-op (promoted EXACTLY once), lost==0.
DEST-kill: D's `Arriving` uncounted then excluded; S `Departing` uncounted (the legal zero-held gap) →
`TransientAbandon` buckets EXACTLY 1 (a redelivered abandon finds nothing), lost==1≤budget, no double-hold.
CrashResurrect control: the kill-only notice never arrives → held-set self-heals `{S}→{}→{D}` via the
retained `Departing` + redelivered promote, lost==0. Across these: go-token = SOLE promote authority;
retained `Departing` makes loss countable; dead-aware twin excludes corpses; `TransientAbandon` = the ONE
deterministic counting event; terminal-on-first-fire + journal idempotency prevent re-introduction.

⚠️ DUAL-PARTICIPANT death (BOTH source AND dest killed mid-handoff — audit D7D-1, MEDIUM, owed): the
`scan_deadlines` gate is source-first, so both-dead selects `SourceUnreachable` → a promote aimed at the
DEAD dest (never applied). The item lands at NO live shard (the honest outcome — it COULD not land
anywhere live) AND is bucketed by NO loss path (`on_transient_abandon` is never reached), so the loss is
SILENT — uncounted by `verify_transient_loss_budget`. This is NOT an invariant break (no double-hold, no
corruption); it is a loss-ACCOUNTING gap under a simultaneous double-kill crash-storm (debris-only,
loss-tolerable). The proper fix (a COUNTED orchestrator-side drop on dual-death so the loss enters the
budget gate) is owed with Slice 2's loss-counting-model work + a dual-victim harness `Scenario` (the
current single-`victim` field cannot express it). See `DEFERRED.md` D-7.

## Deferred (HONEST, each dependency named)

- **D-7d Slice 2:** burst-scale SOURCE-kill (1000-item wholesale self-promote) + DEST-kill cluster
  over-budget (`lost(1000)>budget(4)` → `LostOverBudget{Debris}` honest RED via `expect_err`); happy-path
  `batch_goes` GC at the shard's terminal retire (the two-sided drop-completion signal — Slice 1 GCs only
  at the saga crash-path tombstone; keep the token while ANY item anchored to its fence `is_held()` so the
  D-7c quiescence count stays stable); burst soak-RSS re-verification.
- **D-6:** ORCHESTRATOR-kill cell (batch_goes + live-saga set are in-memory — loudly documented, not
  silently passed); the durable go-token WAL.
- **D-3:** lease-lapse liveness as the proper crash-vs-partition discriminator (replaces kill-only
  `NodeUnreachable`); a harness "flap" fault (NodeUnreachable for a LIVE peer then recover) to exercise
  the prod partition-as-false-kill path (the fabric cannot model a recoverable NodeUnreachable today).
- **Named owners / P11:** cap-split (`max_items_per_batch`→distinct go-token — lossy-if-triggered, unneeded
  for a reliable Saga-class batch); mixed-realm/multi-dest (3rd shard + `seed_transient_crossing_to`);
  O(K·M) shard-side handoff index (fold the strand predicate into the already-mandatory per-tick
  `readvance_transients` pass so detection adds ZERO new passes; the index is a P11-scale fix unreachable
  at K=1).

## Multithreading verdict (unchanged): SEQUENTIAL

No new parallelism in D-7d; the recovery producer is control-plane (one saga per batch, re-driven on a
deterministic deadline). The cross-shard fan-out IS the parallelism (HR1, per-node `step_tick`). rayon
remains un-adopted (user decision; flagged not taken).
