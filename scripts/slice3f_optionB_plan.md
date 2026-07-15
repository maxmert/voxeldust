All facts verified. H3 is confirmed: the current `commit_result` tombstone at line 1086 discards `sagas.remove` result — the plan's rewrite must keep the Saga-DELETE (lines 1090-1092) unconditional, and I now see `ctx` must be captured from the returned `LiveSaga`. The rehydrate scan pattern (lines 1188-1207, `since: UniverseTick(0)`) confirms the RAM-`since`-rearm graft. I don't need to read process_parity further — the adversary confirmed it has no crossing content, which is the load-bearing fact for C2/8d. I now have every fact to render the definitive plan.

---

# DEFINITIVE Option-B edit plan — crossing-abort durability (Mechanism Y + verified adversary folds)

## Scope reframe (adversary C2 — CONFIRMED, load-bearing)

This is **not** durability-hardening of a live acked protocol. Verified against the tree:
- **No production emitter of `CrossingAborted` exists.** `grep` across `crates/node` returns zero `push_flow(...CrossingAborted...)`. The only emit sites are three **test** sites (`stub.rs:8180/8197/8331`). The wire arm is documented **INERT** (`intershard.rs:213-214, 237-238`: "planted; the source consume route lands later — the demux wildcards catch it").
- **The source consumer DID land** (Slice 3e): `on_crossing_aborted` (`stub.rs:1624-1641`) is live and wired at the demux (`stub.rs:3155`). So the *inbound* half exists; the *orchestrator egress* half is Slice **3f** (owed).
- **No crossing saga is minted in production.** `start_transfer`'s only caller is the test harness `trigger()` (`saga_runtime.rs:2416`). `crossing_transfer_id` is used only in `stub.rs` (the source-side latch derivation + tests).

**Therefore this work is a PLANT, delivered alongside Slice 3f, not a standalone hardening slice.** Every new branch is reachable *only* by unit tests that synthesize states directly until 3f wires the orchestrator's `EmitCrossingAborted` executor. This must land as a **DEFERRED.md 🟥 entry**: *"abort-reply durability planted; orchestrator first-emit + full e2e SIGKILL crash-leg owed at Slice 3f."* The in-process `rehydrate` unit test (§8a-iv) is the **only reachable crash proof this slice**; the `process_parity.rs` SIGKILL cell (former §8d) **cannot be written yet** (confirmed: no crossing content in that file, no production abort to kill between) and moves to a 3f dependency.

Given C2, the honest sequencing is: **do this plant AS PART OF Slice 3f** (the emitter and its durability are one correctness unit), OR land the plant now with the DEFERRED entry and a synthesize-only test suite. Either is defensible; the plan below is written so the durability machinery composes with 3f's emitter regardless of ordering.

---

## Mechanism decision: Y stands (durable `pending_abort_replies` map, saga-FSM untouched)

The adversary agreed Y-over-X is sound (zero blast radius on the FSM + its proptests). The five verified folds below change *how* Y is built, not *whether* Y wins. Two of the adversary's HIGHs (H1 store-format, H2 same-fence aliasing) are real correctness holes the original plan missed and are now folded as first-class sub-slices.

---

## Verified corrections to the original plan (do NOT trust the superseded line numbers)

| # | Original claim | VERIFIED truth | Source |
|---|---|---|---|
| C1a | new `CROSSING_ABORT_ACK_STEP = 13 or 14` | **Both taken** (`TRANSIENT_RELEASE_STEP=13`, `TRANSIENT_COMPLETE_STEP=14`); highest is `RE_SOLICIT_STEP=18` | `intershard.rs:82,86,106` |
| C1b | mint a new step const for the ack | **Do NOT mint one.** The existing `CrossingAborted` arm keys on `TRANSIENT_BATCH_STEP`; the ack MUST key on the SAME `(transfer, TRANSIENT_BATCH_STEP)` so request+ack share the journal | `intershard.rs:408-412` |
| C1c(tag) | `StoreKey::ABORT_REPLY = 5` | **5 IS free** (SAGA=1, BATCH_GO=2, DIRECTORY=3, CLOCK=4) — the store-tag claim was correct | `saga_runtime.rs:93-96` |
| H1 | field on `SagaCtx` "threads through `SagaSnapshot` for free" | **That is a store-format BREAK**, not a benefit — `SagaSnapshot{ctx:SagaCtx}` is persisted postcard, non-self-describing; a pre-upgrade store panics on `postcard::from_bytes(...).expect("decode persisted saga snapshot")` | `saga_runtime.rs:138-144, 1189` |
| H2 | fence-inclusion closes double-free | **Only for a moved subject.** A same-fence re-cross re-derives the IDENTICAL `crossing_transfer_id` (`f(subject, subject_fence)`), so a redelivered abort can clear the fresh crossing's latch | `intershard.rs:635`, `stub.rs:1635` |
| H3 | tombstone rewrite | Current code discards `sagas.remove()`; the Saga-DELETE push (`:1090-1092`) MUST stay unconditional; capture `ctx` from the returned `LiveSaga` | `saga_runtime.rs:1086-1092` |
| M1 | `values_mut()` in re-emit | **Use `.iter_mut()`** — need the `TransferId` key; single-pass is fine (`outbox` is a disjoint `&mut` param, not `&runtime`) | `saga_runtime.rs` scan_deadlines sig |

---

## Ordered sub-slices (dependency order: wire → node → sim → tests)

All paths under `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/`.

### Sub-slice 0 — DEFERRED.md 🟥 entry + scope note (adversary C2)
`docs/design/DEFERRED.md`: add *"CROSSING-ABORT-DURABILITY-PLANT — WHAT: orchestrator re-emit + source ack round-trip for a pre-CAS crossing abort, crash-durable. WHERE: `saga_runtime.rs` `pending_abort_replies`, `intershard.rs` `CrossingAbortedAck`, `stub.rs on_crossing_aborted`. WHEN: machinery plants with this slice; the orchestrator FIRST-EMIT (`EmitCrossingAborted` executor) + the e2e SIGKILL-between-abort-and-ack crash leg land at Slice 3f."* Flips 🟩 when 3f wires the emitter and the process-parity crash cell passes.

### Sub-slice 1 — vd-wire: the `CrossingAbortedAck` arm (`crates/wire/src/intershard.rs`)
1a. **NO new step const.** Delete the original Step 1a entirely.
1b. **Append** after `CrossingAborted(CrossingAborted)` (`:241`): `CrossingAbortedAck(CrossingAborted)` — reuse the `{subject, transfer}` payload (`:620-624`). Append-only preserves every postcard discriminant.
1c. `effect_class` — add after the `CrossingAborted` arm (`:408-413`):
```rust
InterShardFlow::CrossingAbortedAck(a) => EffectClass::SideEffecting {
    idempotency: IdempotencyKey::TransferStep { transfer: a.transfer, step_id: TRANSIENT_BATCH_STEP },
},
```
(Same `(transfer, TRANSIENT_BATCH_STEP)` key as the request it answers — adversary C1b; the source's ack-journal and the orchestrator's request-journal are then on the same key.)
1d. `durability_class` — add `| InterShardFlow::CrossingAbortedAck(_)` to the `ReDriven` group (`:471-474`). Justification: source-re-driven (re-emitted whenever the orchestrator re-emits `CrossingAborted`); NOT producer-less → no `Retained` marker owed. Update the group doc comment (`:467-470`) to name the ack.
1e. `crates/wire/src/lib.rs:15` — extend the header arm list (`| CrossingAbortedAck`).
1f. Header seam-map comment (`intershard.rs:22-26`) — add the `CrossingAbortedAck` bullet: *"source→orch: latch clear-confirm; the orchestrator drops the durable pending-abort entry."*

### Sub-slice 2 — vd-wire marker test (`crates/wire/tests/intershard_closed.rs`)
2a. Add a `CrossingAbortedAck` sample to `every_arm` (mirror `CrossingAborted`).
2b. Add `| InterShardFlow::CrossingAbortedAck(_) => {}` to the exhaustive no-op match.
2c. `durability_class_pins_the_producer_less_reliable_set` — NO change (the ack is not producer-less). **Verified consistent** (adversary M3).
2d. Add an assertion that `CrossingAbortedAck`'s `effect_class` step == `CrossingAborted`'s step (both `TRANSIENT_BATCH_STEP`) — pins C1b so a future refactor can't split them.

### Sub-slice 3 — vd-sim: crossing-origin discriminator, done WITHOUT a persisted-struct break (adversary H1)
The original plan put `is_crossing_origin` on `SagaCtx`. **Rejected** — `SagaCtx` is inside the persisted `SagaSnapshot` (verified `:139`), and postcard is not self-describing, so appending breaks decode of any pre-upgrade store. Two options; **OPEN QUESTION Q1 decides**:

- **Option 3-A (preferred — no store-format risk):** derive crossing-origin from data the saga *already* carries. The crossing saga is minted with `ctx.transfer == crossing_transfer_id(subject, expected_fence)` (verified: the source and orchestrator independently derive the SAME id, `intershard.rs:626-628`). So the orchestrator can recompute `crossing_transfer_id(ctx.subject, ctx.expected_fence) == ctx.transfer` as a **branchless pure check** — no new field, no `SagaSnapshot` change, no migration. This is the honest discriminator (it's literally what "this is a crossing saga" *means*) and costs zero persisted bytes. **This supersedes both the realm-proxy AND the bool.**
- **Option 3-B (fallback, only if 3-A is unsound):** add `is_crossing_origin: bool` to `SagaCtx` **and** bump the store format — version `SagaSnapshot` behind an explicit `u8` version tag (postcard has no `#[serde(default)]`), documented as a genesis-wipe store-format bump (pre-prod, store wiped on genesis — acceptable but must be *stated*, not silent). Extend §8e to assert a pre-field snapshot is handled.

3a. If 3-A: add a `#[must_use] fn is_crossing_origin(ctx: &SagaCtx) -> bool` helper in `saga_runtime.rs` (NOT `saga.rs` — keeps the saga crate byte-for-byte untouched), body `crossing_transfer_id(ctx.subject, ctx.expected_fence) == ctx.transfer`. Import `crossing_transfer_id` from `vd_wire`.
3b. **No `saga.rs` edit at all** under 3-A. The terminal `Aborted` edge (`:1432-1439`) and both proptests stay verbatim. (Under 3-B, `saga.rs` gains one field + every `SagaCtx` constructor is compile-forced to set it — the `ctx_class` proptest helper picks a fixed value; a dedicated crossing-ctx test pins `true`.)

### Sub-slice 4 — vd-node: durable map + persistence (`crates/node/src/saga_runtime.rs`)
4a. `struct PendingAbortReply { source: NodeId, subject: DirectoryKey, since: UniverseTick }` (`since` RAM-only re-drive cadence).
4b. Field on `SagaRuntimeRes`: `pending_abort_replies: BTreeMap<TransferId, PendingAbortReply>`. Doc: *"bounded PROVIDED the source acks OR is confirmed dead (self-reap, §5) — one entry per in-flight aborted crossing"* (adversary L2 — the bound is only honest post-C3).
4c. `StoreKey::AbortReply(TransferId)` variant (`:86`), `const ABORT_REPLY: u8 = 5;` (**verified free**, `:93-96`), and its `bytes()` arm mirroring `Saga` (`:104-107`).
4d. Persisted record `AbortReplySnapshot { transfer, source, subject }` — self-describing (mirror `GoTokenSnapshot` `:149-155`). **Persist only `source`+`subject`, NOT `since`** — re-armed to `UniverseTick(0)` on rehydrate (mirror the saga `since: UniverseTick(0)` at `:1199`).
4e. Insert site in `commit_result` `if tombstone` (`:1085-1092`) — capture `ctx` from the returned `LiveSaga`, keep the Saga-DELETE unconditional (adversary H3):
```rust
if tombstone {
    if let Some(saga) = runtime.sagas.remove(&transfer) {
        if matches!(final_state, SagaState::Aborted { .. })
            && saga.ctx.class == DurabilityClass::Durable
            && is_crossing_origin(&saga.ctx)
        {
            let reply = PendingAbortReply { source: saga.ctx.source, subject: saga.ctx.subject, since: now };
            runtime.pending_abort_replies.insert(transfer, reply);
            runtime.pending_writes.push((
                StoreKey::AbortReply(transfer).bytes(),
                Some(encode(&AbortReplySnapshot { transfer, source: reply.source, subject: reply.subject })),
            ));
        }
    }
    // UNCONDITIONAL (unchanged): the tombstoned saga's snapshot is deleted even when removed==None
    // (a re-driven tombstone), and the AbortReply insert above is correctly skipped in that case.
    runtime.pending_writes.push((StoreKey::Saga(transfer).bytes(), None));
}
```
Split the three gate conditions so each false arm is a covered region (HR5(d) — do NOT `&&`-short-circuit them into one uncovered predicate; use nested `if` / separate boolean bindings asserted in tests). `now` is already a `commit_result` param (verified `:1057`).
4f. Rehydrate scan after the `BATCH_GO` scan (`:1210-1213`):
```rust
let mut pending_abort_replies = BTreeMap::new();
for (_k, v) in store.scan(&[StoreKey::ABORT_REPLY]) {
    let snap: AbortReplySnapshot = postcard::from_bytes(&v).expect("decode persisted abort reply");
    pending_abort_replies.insert(snap.transfer,
        PendingAbortReply { source: snap.source, subject: snap.subject, since: UniverseTick(0) });
}
```
then `runtime.pending_abort_replies = pending_abort_replies;` alongside `:1220-1222`.

### Sub-slice 5 — vd-node: `scan_deadlines` re-emit WITH dead-source self-reap (adversary C3 — the lost-ACK reclaim answer)
This is the sub-slice that **answers the prompt's critical question**: the ack-gated protocol must not merely relocate the strand. The re-emit loop is gated on `redrive_deadline_ticks` (deterministic cadence, adversary L1 guard) AND, on a **confirmed-dead source, self-reaps the entry instead of re-emitting forever** (adversary C3 — reuse the same dead-source signal the saga re-drive uses, `dead_observed_since` / `is_confirmed_dead`). D-37 has already removed the entity on a permanent source kill, so the abort is genuinely moot — DELETE the record and drop the entry.

Add after the existing `due` drain in `scan_deadlines`, **iterating `.iter_mut()`** (adversary M1 — need the key; `outbox` is a disjoint `&mut` param so single-pass compiles):
```rust
// PLANT (Slice 3f-prep): re-emit CrossingAborted per durable pending abort reply until the source
// acks OR is confirmed dead. DECOUPLED from the saga (it's tombstoned). Gated by redrive_deadline_ticks
// (determinism); `since` re-armed BEFORE the push (one fire per window). Straight-line — no branch inside
// the loop body other than the two decided arms (HR5); collect reaps, apply after the borrow ends.
debug_assert!(tuning.redrive_deadline_ticks > 0, "every-tick storm guard (L1)");
let mut reap: Vec<TransferId> = Vec::new();
let mut reemit: Vec<(TransferId, NodeId, DirectoryKey)> = Vec::new();
for (transfer, reply) in runtime.pending_abort_replies.iter_mut() {
    if runtime_is_confirmed_dead(reply.source) {          // same D-3 signal the saga re-drive uses
        reap.push(*transfer);                              // C3: entity already re-homed → abort moot
    } else if now.0.saturating_sub(reply.since.0) >= tuning.redrive_deadline_ticks {
        reply.since = now;
        reemit.push((*transfer, reply.source, reply.subject));
    }
}
for t in reap {
    runtime.pending_abort_replies.remove(&t);
    runtime.pending_writes.push((StoreKey::AbortReply(t).bytes(), None));  // DELETE — self-reaped
}
for (transfer, source, subject) in reemit {
    outbox.push_flow(source, MsgClass::Saga,
        &InterShardFlow::CrossingAborted(CrossingAborted { subject, transfer }));
}
```
**OPEN QUESTION Q2:** the exact confirmed-dead accessor. The saga loop reads `dead_observed_since` per-saga; a tombstoned saga has none. The reap must consult the **runtime-level liveness tracker** (`is_confirmed_dead(NodeId)` — the same `UnreachEvidence`-backed predicate `rehome_event_for` uses, `saga_runtime.rs:~1682`). Confirm that predicate is callable given `&runtime` here without a borrow clash against `pending_abort_replies.iter_mut()` — if it borrows `&runtime.liveness` it's a disjoint field (fine); if it takes `&mut self` it needs the two-pass collect form (already used above). **Read the `is_confirmed_dead` signature before coding.**

### Sub-slice 6 — vd-node: consume the ack (`drive_sagas` inbound match, `~:1990`)
Before the `_ => {}` catch-all:
```rust
Ok(InterShardFlow::CrossingAbortedAck(a)) => {
    if runtime.pending_abort_replies.remove(&a.transfer).is_some() {
        runtime.pending_writes.push((StoreKey::AbortReply(a.transfer).bytes(), None));
    }
}
```
Idempotent (the `is_some()` false arm = a redelivered ack for an already-removed/reaped entry, a no-op — cover it). Rides the same barrier. **OPEN QUESTION Q3:** confirm `MsgClass::Saga` from a source shard routes to `drive_sagas`, not `serve_directory` (the 1c.4 `DirectoryReply` dispatch-collision precedent) — read the demux at `saga_runtime.rs:~198`.

### Sub-slice 7 — vd-sim: source-side UNCONDITIONAL ack (`stub.rs:1624-1641` `on_crossing_aborted`, call site `:3155`)
The verified consumer clears the latch ONLY on an id-match (`:1635`) and else counts `crossing_abort_stale` / `crossing_abort_no_entity`. The ack must fire on **all three paths** — else a lost first ack whose re-emit finds the latch already cleared hits the stale arm and strands the durable entry forever (this is the lost-ACK-reclaim lynchpin, and it interlocks with §5's re-emit).
7a. Thread `outbox: &mut OutboundBox` + `config.orchestrator` into `on_crossing_aborted` (like `on_saga_promote`'s `config.orchestrator` at `:1684`); update the call site (`:3155`). **OPEN QUESTION Q4:** confirm `StubConfig` exposes `orchestrator` and that the `:3155` call site has `outbox` + `config` in scope. **Also — adversary L3 / H2 interlock:** ack the orchestrator that *sent* this `CrossingAborted`, not a static `config.orchestrator`, if the inbound carries a usable origin `NodeId`; if not (P3 single-orchestrator), use `config.orchestrator` and DEFER the multi-orch case with a note.
7b. Emit unconditionally — outside the id-match, after the stats bump, on every return path:
```rust
outbox.push_flow(config.orchestrator, MsgClass::Saga,
    &InterShardFlow::CrossingAbortedAck(CrossingAborted { subject: abort.subject, transfer: abort.transfer }));
```
Keep the latch-clear id-match-gated; the ack is unconditional (the no-entity early-return must ALSO ack before returning — restructure so the ack is unconditional).

### Sub-slice 8 — vd-sim H2 fix: same-fence re-cross id-aliasing (adversary H2 — the double-free the fence-inclusion does NOT close)
Verified: an abort → source clears latch → same-fence re-cross re-derives the **identical** `crossing_transfer_id` (`:635`). Then a still-pending `CrossingAborted{T}` redelivery arriving at the source finds the latch re-holding `T` (the *new* cross) and wrongly clears it + acks it away, un-tracking the fresh saga. The `{subject, transfer}` payload (`:620-624`) has no fence to disambiguate.

**Cure (minimal, no wire-payload grow):** make the source latch clear **one-shot per aborted id**. The source records the last aborted `transfer` id it has already cleared (a small bounded `BTreeSet<TransferId>` or the `RequestInFlight` gaining an `aborted_ids` shadow); `on_crossing_aborted` clears the latch only if `in_flight.0.get(&entity) == Some(&transfer)` **AND** this id has not already been cleared-by-abort. A re-cross at the same fence re-mints the same id but re-latches it *after* the abort — so the dedup set must be scoped to *"aborts I've consumed"*, and the set entry is dropped when the entity next successfully commits/re-latches a NEW crossing.
**OPEN QUESTION Q5 (needs a code read before coding — the HARDEST open item):** the exact dedup lifetime. A pure "seen this aborted id" set leaks/over-suppresses a *legitimate* same-fence re-abort (entity re-crosses same fence, that new saga ALSO aborts → same id → must clear again). The clean cure is a **fence-or-generation stamp on the latch** so abort matches `(entity, transfer, latch_generation)`, where `generation` increments each time the source re-latches. That requires `RequestInFlight` to store `(TransferId, generation)` and `CrossingAborted` to carry the generation — **a wire-payload grow** (`CrossingAborted { subject, transfer, generation }`), which is the fully-correct fix but touches the frozen arm. **Decision owed with the user / at 3f:** (a) bounded one-shot dedup set (no wire change, over-suppresses the rare same-fence-re-abort — acceptable if that path is genuinely rare and the entity just retries next tick), vs (b) generation-stamped latch + wire-payload grow (fully correct, touches the frozen `CrossingAborted`). **Recommend (b)** — it's the honest fix and `no_simplest_fix_v3` disfavors the dedup-set hack; but it's a wire grow that belongs to Slice 3f's design, not a bolt-on here. Flag it as the primary design decision 3f must resolve.

### Sub-slice 9 — tests (HR5 100% region+branch)
9a. **`saga_runtime.rs` unit** (module `~:2600`), all synthesizing states directly (C2 — no reachable prod path):
  - (i) crossing-origin Durable `Aborted` tombstone inserts a `pending_abort_replies` entry + stages the PUT (assert both the map entry AND the exact `StoreKey::AbortReply(t).bytes()` in `pending_writes`).
  - (ii) `scan_deadlines` re-emits `CrossingAborted` to `source` **exactly once per `redrive_deadline_ticks` window** (`assert_eq!` on the emit count, not `>=` — adversary L1).
  - (iii) `CrossingAbortedAck` inbound removes the entry + stages the DELETE + stops the re-emit.
  - (iv) **crash leg (the core Y payoff):** `rehydrate` restores a persisted entry (synthesize the `AbortReply` store record) and re-emits on the first post-restart `scan_deadlines` tick.
  - (v) **dead-source self-reap (adversary C3):** a `pending_abort_replies` entry whose `source` is confirmed-dead is REAPED (map entry removed + DELETE staged), NOT re-emitted — assert zero `CrossingAborted` emitted for it.
  - (vi) every false gate arm inserts NOTHING: non-crossing ctx (id ≠ `crossing_transfer_id(subject, expected_fence)` under 3-A), `Transient` class, non-`Aborted` tombstone (`Done`), tombstone-with-`removed==None` (re-driven tombstone — adversary H3), ack-for-unknown-transfer.
  Use `assert_eq!`/`expect_err`, split conditions (HR5(d)).
9b. **`stub.rs` unit** (near `:8180`): `on_crossing_aborted` emits `CrossingAbortedAck` on ALL THREE paths — id-match-clear, stale (`crossing_abort_stale`), and no-entity (`crossing_abort_no_entity`) — the unconditional-ack guarantee (§7); latch cleared exactly once. Plus the **H2 same-fence** test (§8): an abort for `T` after a same-fence re-cross re-latched `T` does NOT wrongly clear the fresh latch (under whichever Q5 cure lands).
9c. **`intershard_closed.rs`**: §2 covers it (marker + the step-equality assertion §2d).
9d. **e2e / process-parity crash leg** — **MOVED to Slice 3f** (adversary C2: cannot be written now; `process_parity.rs` has no crossing content and there is no production abort to SIGKILL between). At 3f: a SIGKILL-between-abort-and-ack cell asserting the source latch is eventually cleared post-restart via the persisted `AbortReply` family + RedbStore rehydrate. The §9a-iv in-process test is the reachable proof this slice; the process leg is belt-and-suspenders at 3f.
9e. **Retain/no-resurrect test** for the `AbortReply` family, analogous to `rehydrate_does_not_resurrect_a_revoked_directory_record`: a persisted-then-DELETEd `AbortReply` does not resurrect on rehydrate. **Plus** the rehydrate-then-immediately-ack race (adversary M4): record restored → ack same first tick → DELETE staged → committed, no re-emit. Note (verified `:2038-2048` region is Directory-only incremental-reconcile): `AbortReply` DELETE is a direct `pending_writes.push((key, None))` like `Saga`/`BatchGo`, so it owes its OWN retain test (the differential oracle doesn't cover it).

### HR5 coverage notes
- All new branching lives in **monomorphic** `saga_runtime.rs` / `stub.rs` sites and wire unit tests — **none inside a generic body**, none touching a proptest invariant (Y's whole point).
- Under **Option 3-A** there is no `SagaCtx` field, so no per-monomorphization both-values obligation on the generic `ctx_class` proptest helper (the exact HR5 gotcha the original bool would have imposed — adversary M2). This is a further reason to prefer 3-A.
- The §5 loop body has exactly two decided arms (reap / re-emit) plus the disjoint apply-passes — cover: confirmed-dead→reap, alive+window-elapsed→re-emit, alive+window-not-elapsed→no-op. The §4e gate: true + each false arm separately (§9a-vi). The §6 ack: remove-hit + remove-miss. All conventional monomorphic region/branch coverage.
- `encode`/`postcard::to_allocvec(...).expect(...)` at the new store-key/record sites are straight-line infallible (the panic body is stdlib, not a coverable caller branch — matches the existing `StoreKey::bytes` discipline, `:99-100`).

---

## Explicit resolution of the lost-ACK reclaim (the prompt's critical requirement)

The ack-gated protocol must NOT merely relocate the strand from source→orchestrator. The complete, verified liveness chain:
1. **Source alive:** orchestrator re-emits `CrossingAborted` every `redrive_deadline_ticks` (§5) → source re-acks **unconditionally** on every redelivery including the stale/no-entity paths (§7) → orchestrator removes the entry + DELETEs the record (§6). A lost first ack is recoverable because §5 keeps re-emitting and §7 always re-acks. ✅
2. **Source permanently dead:** §5 detects the confirmed-dead source (same D-3 signal the saga re-drive uses) and **self-reaps** the entry (drops it + DELETEs the record) — because D-37 has already re-homed the entity, so the abort is genuinely moot. This closes the "unbounded persisted re-emitting entry forever" hole (adversary C3) — the original plan asserted this reasoning but did NOT implement the reap, leaving the strand relocated to an unbounded durable map. §5 implements it. ✅

The bound in §4b is therefore **honest**: `pending_abort_replies` is bounded because every entry terminates in either an ack (source alive) or a self-reap (source dead) — never a permanent re-emit.

---

## OPEN QUESTIONS needing a code read before coding

- **Q1 (Option 3-A vs 3-B) — decide first.** Confirm the crossing saga's `ctx.transfer` is minted as `crossing_transfer_id(ctx.subject, ctx.expected_fence)` at the orchestrator start path (read Slice 3f's `start_transfer` crossing branch, not yet written, OR the design in the 3c/3f plan). If yes → **3-A** (recompute, zero persisted-struct change, avoids H1 entirely). If the orchestrator mints the id by another means → **3-B** (versioned `SagaSnapshot`, documented store-format bump).
- **Q2** — the `is_confirmed_dead(NodeId)` runtime accessor signature + borrow shape vs `pending_abort_replies.iter_mut()` (`saga_runtime.rs:~1682`, `rehome_event_for`).
- **Q3** — the source-shard `MsgClass::Saga` demux target (`drive_sagas` vs `serve_directory`, dispatch-collision check, `~:198`).
- **Q4** — `StubConfig.orchestrator` exists + `outbox`/`config` in scope at `stub.rs:3155`; and whether the inbound `CrossingAborted` can carry/derive the SENDING orchestrator id (adversary L3, and the H2 interlock).
- **Q5 (the hardest — H2 same-fence aliasing)** — the exact `RequestInFlight` shape (`stub.rs:374`) and whether the correct cure is a bounded one-shot dedup set (no wire change, over-suppresses the rare same-fence re-abort) or a **generation-stamped latch + a `CrossingAborted { …, generation }` wire-payload grow** (fully correct, touches the frozen arm). **Recommend the generation stamp**, but it is a wire-payload design decision that belongs to Slice 3f, not a bolt-on — flag it to the user as the primary correctness decision 3f must make.

## Net honest assessment
Mechanism Y is the right choice (verified: zero saga-FSM blast radius, `saga.rs` untouched under Option 3-A). Three adversary CRITICALs are **confirmed against the tree and folded**: C1 (wire step is 18-not-13/14, and the ack must reuse `TRANSIENT_BATCH_STEP` — no new const), C2 (this is an **inert 3f-prep PLANT**, not a hardening — re-scoped with a DEFERRED entry and synthesize-only tests; the process crash leg moves to 3f), C3 (the **dead-source self-reap** is the real lost-ACK-reclaim answer and is now implemented in §5, not merely asserted). Two adversary HIGHs are **real holes the original plan missed and are now first-class**: H1 (persisted-`SagaCtx` decode break → Option 3-A recompute-discriminator avoids it entirely) and H2 (same-fence re-cross id-aliasing → §8, with the generation-stamp cure flagged as the primary 3f design decision, Q5). The store-tag (5) and H3 (unconditional Saga-DELETE) were verified and folded. Five OPEN QUESTIONs remain, all requiring a specific code read before the first line is written; Q1 and Q5 are decisive and gate the design.