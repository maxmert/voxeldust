Now I have complete ground truth. Critical confirmations for the final plan:

1. **`subject_fence = dot.authority.fence()`** (2346) — for a durable dot, this is the ENTITY's authority fence. On a **pre-CAS abort, the entity's authority fence does NOT advance** (the crossing never committed; source authority retained). This **confirms CRITICAL-1**: a re-latch with the same `(subject, subject_fence)` mints the byte-identical id. The DRAFT's H2 attempt-stamp was one proposed fix, but the wire is frozen with 2-arg `crossing_transfer_id` documented as `(subject, subject_fence)`-idempotent, and the wire doc explicitly says the re-attempt idempotency is a FEATURE ("a re-attempt after a lost RAM enqueue is idempotent"). So the DRAFT's approach (grow the wire + break documented idempotency) is architecturally wrong.

2. **`rehome_transfer_id`** is 2-arg `(subject, prev_fence)` — the L2 hoist is viable but the DRAFT's `attempt`-folding version is not (it would change crossing's arity but not rehome's, and grow the wire).

3. **The wire already documents the correct design**: orchestrator holds the obligation in `pending_abort_replies`, re-emits via `scan_deadlines`, crash-durable via the store, dropped by `CrossingAbortedAck`. This is exactly Mechanism Y minus the DRAFT's flawed H2.

I now have everything. The DRAFT's H2 attempt-stamp is the wrong fix; the adversary's CRITICAL-1 is valid but its recommended fix (move counter to a surviving store) still fights the frozen wire. The correct resolution keeps the wire frozen and solves re-cross-after-abort differently. Let me write the definitive plan.

---

# DEFINITIVE PLAN — Slice 3f-D + 3f-E: durable crossing-abort (Mechanism Y)

## Verdict on the DRAFT vs the ADVERSARY

I re-read every seam against HEAD. Findings:

- **The ADVERSARY's CRITICAL-1 is REAL** (`subject_fence = dot.authority.fence()` at `stub.rs:2346` does not advance on a pre-CAS abort — confirmed; a re-latch re-mints the identical id).
- **The ADVERSARY's CRITICAL-2 is REAL** (`CrossingProgress` is NOT in `on_directory_reply`'s params at `stub.rs:3089-3104`; `process_inbound` at `stub.rs:986-1005` is at the 16-param ceiling with `ghost_state` already bundled to dodge it).
- **But the DRAFT's own H2 fix is ALSO wrong, and worse than the adversary saw.** The wire is FROZEN: `crossing_transfer_id(subject, subject_fence)` at `intershard.rs:663` and `CrossingRequest` at `intershard.rs:611` are 2-arg / 5-field, and the doc at `intershard.rs:654-661` **guarantees** `(subject, subject_fence)`-determinism as a feature ("a re-attempt after a lost RAM enqueue is idempotent"). The DRAFT's plan to append `attempt` grows the frozen wire AND destroys that documented idempotency. **The adversary's recommended fix (move the counter to a surviving store) still requires the wire growth and still fights the frozen contract.**

**The correct resolution abandons the attempt-stamp entirely.** The wire already documents Mechanism Y correctly (`intershard.rs:242-250`, `CrossingAbortedAck` doc: "held on the ORCHESTRATOR in `pending_abort_replies`, re-emitted every `scan_deadlines`, crash-durable via the saga store, until THIS ack lands"). The whole slice is: **build that documented orchestrator machinery, keep the id 2-arg, and solve the re-cross-after-abort with an L1 dwell re-arm plus a source-side single-fire latch cooldown — no new id, no wire growth.** This is simpler than either the DRAFT or the adversary proposed, and it is what the frozen wire already promises.

The one genuinely hard question — *if the re-latch re-mints the same id X, does a stale abort for X clear the live re-latch?* — is answered by the ack round-trip, not by a generation stamp. See OPEN QUESTION Q1 (the one item I could not fully close by reading and that gates the design choice).

---

## Sub-slice ordering

```
3f-D0  ORCH scaffolding  (pending_abort_replies field + PendingAbortReply + StoreKey::AbortReply=5 + rehydrate)  — additive, gates alone
3f-D1  ORCH insert       (commit_result tombstone: read-live-before-remove, tag-check discriminator, .expect not if-let)
3f-D2  ORCH re-emit+reap  (scan_deadlines per-entry re-emit + dead-source self-reap)
3f-D3  ORCH ack consumer  (CrossingAbortedAck drops the entry) + H-2 start-arm guard
3f-D4  SOURCE ack+re-arm   (on_crossing_aborted: unconditional ack + L1 dwell re-arm; thread CrossingProgress via bundle)
3f-E   capstone            (unit sweep + SIGKILL crash-leg e2e)
```

**No ATOMIC wire commit exists in this plan** — the DRAFT's 3f-D0 (wire arity change) is DELETED. `crossing_transfer_id` stays 2-arg; `CrossingRequest` stays 5-field. This removes the largest, riskiest sub-slice entirely.

Each of 3f-D0…D3 is independently `just gate`-able (orchestrator-only, additive). 3f-D4 is the one sim-side change and compiles alone once its bundle change lands. 3f-E is the release-gated capstone.

---

## 3f-D0 — ORCHESTRATOR: `pending_abort_replies` scaffolding (additive, gates alone)

**File:** `crates/node/src/saga_runtime.rs`.

### D0.1 — `PendingAbortReply` type (near `PendingReHome` at 232)

```rust
/// 3f-D (Mechanism Y): a durable orchestrator record that a CROSSING-ORIGIN durable saga tombstoned
/// ABORTED, so the SOURCE's `RequestInFlight` latch must be positively cleared. Held in
/// `SagaRuntimeRes::pending_abort_replies` (keyed by the aborted `TransferId`) AND persisted
/// (`StoreKey::AbortReply`) so an orchestrator restart between the abort and the source's ack re-emits
/// `CrossingAborted` via rehydrate → the latch still clears. Dropped by the source's `CrossingAbortedAck`,
/// or self-reaped when `source` is confirmed dead (D-37 already re-homed the entity → the abort is moot).
/// Self-describing (carries `transfer`), so rehydrate never parses the store key.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) struct PendingAbortReply {
    pub transfer: TransferId,
    pub source: NodeId,
    pub subject: DirectoryKey,
}
```

**IMPORTANT design note vs DRAFT:** the DRAFT made this DURABLE while `PendingReHome` (232) is RAM-only. That is correct here and intentional — the wire doc (`intershard.rs:246`) mandates crash-durability ("crash-durable via the saga store"). The persistence keying below is what makes it real.

### D0.2 — `SagaRuntimeRes` field (after 465, before the closing `}` at 466)

```rust
    /// 3f-D: crossing-origin durable-abort replies pending a source ack. Keyed by the aborted `TransferId`.
    /// DURABLE (persisted + rehydrated) — Mechanism Y's whole point is crash-durability of the latch-clear
    /// signal. `scan_deadlines` re-emits per entry until a `CrossingAbortedAck` drops it (or dead-source reap).
    pub pending_abort_replies: BTreeMap<TransferId, PendingAbortReply>,
```

`#[derive(Default)]` on `SagaRuntimeRes` (344) yields an empty map; `with_tunings`'s `..default()` and `default()` (505) keep working — zero constructor churn. `BTreeMap` already imported (41).

### D0.3 — `StoreKey::AbortReply` (tag **5 confirmed free**: `SAGA=1,BATCH_GO=2,DIRECTORY=3,CLOCK=4` at 94-97)

- Enum (86-91): add `AbortReply(TransferId),`.
- Consts: add `const ABORT_REPLY: u8 = 5;`.
- `bytes()` (102-120): add, mirroring `Saga(t)` at 105-108:

```rust
            StoreKey::AbortReply(t) => {
                k.push(Self::ABORT_REPLY);
                k.extend(postcard::to_allocvec(&t).expect("encode TransferId store key"));
            }
```

### D0.4 — rehydrate scan (in `rehydrate` at 1213; mirror the `SAGA` scan at 1245 / `BATCH_GO` at 1267)

```rust
    let mut pending_abort_replies = BTreeMap::new();
    for (_k, v) in store.scan(&[StoreKey::ABORT_REPLY]) {
        let snap: PendingAbortReply = postcard::from_bytes(&v).expect("decode persisted abort-reply");
        pending_abort_replies.insert(snap.transfer, snap);
    }
```
Assign alongside `runtime.batch_goes = …`: `runtime.pending_abort_replies = pending_abort_replies;`

### D0 tests
- `abort_reply_store_key_roundtrips`: `bytes()[0] == 5` and disjoint from 1/2/3/4.
- `abort_reply_snapshot_roundtrips`: postcard `PendingAbortReply` → decode → `assert_eq!` all three fields.
- `rehydrate_restores_pending_abort_replies`: seed a store record, `rehydrate`, assert the map entry (mirror the BatchGo rehydrate test).
- `store_key_tags_are_distinct`: all five tag consts distinct.

**HR5:** `bytes()` gains one arm (covered by the roundtrip). The rehydrate scan is monomorphic; `.expect` decode-panic body is stdlib (uncoverable-by-design, matches the BatchGo scan). `PendingAbortReply` is concrete — no generic region multiplication. Use `assert_eq!` on decoded fields.

---

## 3f-D1 — ORCHESTRATOR: `commit_result` tombstone insert (tag-check discriminator; `.expect` not `if-let`)

**File:** `crates/node/src/saga_runtime.rs`, `commit_result` tombstone block (1142-1149). Current code does `sagas.remove` FIRST with **no live read** — so the insert must read the live saga *before* the remove.

Replace the tombstone block:

```rust
    if tombstone {
        // 3f-D (Mechanism Y): a CROSSING-ORIGIN (tag 0x39) durable saga that tombstoned ABORTED must
        // positively clear the SOURCE's crossing latch. Stateless tag-check discriminator on the id's high
        // byte — 0x39 is EXCLUSIVE of rehome (0x37) and the connection-plane's small high-byte-0x00 ids, so
        // no SagaSnapshot / ctx.expected_fence recompute (supersedes the wire-vs-head-fence-fragile recompute).
        let is_crossing_origin = (transfer.0 >> 120) == 0x39;
        let aborted = matches!(final_state, SagaState::Aborted { .. });
        if is_crossing_origin && aborted {
            // The saga is GUARANTEED present here: it was live when `run_to_quiescence` produced this
            // terminal (same single-threaded schedule, nothing touches `runtime.sagas` between). `.expect`
            // (NOT `if let Some`) so there is no uncoverable None region — mirrors the else-branch at 1155.
            let live = runtime
                .sagas
                .get(&transfer)
                .expect("the aborted crossing saga was present at the start of this run");
            let reply = PendingAbortReply {
                transfer,
                source: live.ctx.source,
                subject: live.ctx.subject,
            };
            runtime.pending_abort_replies.insert(transfer, reply);
            // Persist (self-describing). Committed at the end-of-tick barrier with the saga snapshots
            // (persist-before-effect). The FIRST CrossingAborted emit is left to the next `scan_deadlines`
            // (which owns the outbox) — Mechanism Y IS "scan re-emits per entry until acked", so a
            // scan-driven first emit is the design, not a gap, and keeps `commit_result` outbox-free.
            runtime
                .pending_writes
                .push((StoreKey::AbortReply(transfer).bytes(), Some(encode(&reply))));
        }
        runtime.sagas.remove(&transfer);
        runtime
            .pending_writes
            .push((StoreKey::Saga(transfer).bytes(), None));
    } else {
```

**Adversary L-1 folded:** `.expect()` (guaranteed-present) replaces the DRAFT's `if let Some(live)` — the None arm was unreachable, an HR5 uncoverable region. This is the exact idiom already at 1155-1158.

**Adversary M-3 folded (co-tick assertion):** the D1 test asserts both `pending_writes` pushes (`AbortReply=Some`, `Saga=None`) land in the same tick's drain and are distinct key families.

### D1 tests
- `commit_result_aborted_crossing_inserts_pending_reply`: live saga w/ 0x39-tag `transfer` + `ctx.source`/`ctx.subject`; call with `final_state=Aborted{..}, tombstone=true` → entry present with right source+subject; a `pending_writes` push `(AbortReply.bytes(), Some(..))`; AND the co-tick `Saga=None` push present, families distinct (M-3).
- `commit_result_done_crossing_does_not_insert`: 0x39 saga but `final_state=Done{..}` → empty (covers `&&` false-via-second).
- `commit_result_aborted_rehome_does_not_insert`: 0x37-tag saga, Aborted → NOT inserted (false-via-first; the discriminator-exclusivity test — H1 supersession rests here).
- `commit_result_aborted_regular_transfer_does_not_insert`: `TransferId(1)` (high-byte 0x00), Aborted → NOT inserted (exclusivity vs connection-plane ids).

**HR5:** split the `&&` (rehome test → first-false; done test → second-false; crossing test → both-true). No `if let` None region (removed via `.expect`). `matches!(final_state, Aborted{..})` — struct variant confirmed (`saga_runtime.rs:1487,4194`); its true/false both covered by the crossing/done tests. `commit_result` is a plain fn.

---

## 3f-D2 — ORCHESTRATOR: `scan_deadlines` per-entry re-emit + dead-source self-reap + throttle

**File:** `crates/node/src/saga_runtime.rs`, `scan_deadlines` (1805). Add a SEPARATE loop AFTER the `due` drain (`outbox`/`now`/`runtime` all in scope; does NOT touch `sagas`).

**Adversary M-1 folded: the throttle lands NOW, not deferred.** Add `last_emit: UniverseTick` to `PendingAbortReply` (it is already `Copy` + persisted + self-describing, so the cost is nil) and gate the re-emit on a period. This prevents the un-throttled per-tick egress against a live-but-ack-stalled source under the density fixture.

Amend D0.1's struct to add `pub last_emit: UniverseTick,` (init `UniverseTick(0)` at insert in D1 — a `0` last_emit means "never emitted", so the first scan emits immediately).

```rust
    // 3f-D (Mechanism Y): re-emit each pending crossing-abort reply until the source acks (drops it via
    // CrossingAbortedAck), THROTTLED to `abort_reply_period_ticks` (M-1: an alive-but-ack-stalled source
    // must not draw a per-tick egress, esp. under the density fixture). DEAD-SOURCE SELF-REAP: if the source
    // is CONFIRMED dead (the SAME `is_confirmed_dead` signal the saga re-drive uses), D-37 already re-homed
    // the entity → the abort is MOOT → drop + persist-DELETE. Collect-then-act (disjoint borrows: read
    // `&runtime.liveness`, iterate `pending_abort_replies`, write `&mut pending_writes` + `&mut pending_abort_replies`).
    let liveness = &runtime.liveness;
    let period = runtime.tunings.transfer.abort_reply_period_ticks; // OPEN Q3: confirm the tuning home
    let mut reaped: Vec<TransferId> = Vec::new();
    let mut to_emit: Vec<(NodeId, PendingAbortReply)> = Vec::new();
    for (transfer, reply) in runtime.pending_abort_replies.iter() {
        if liveness.is_confirmed_dead(reply.source, now) {
            reaped.push(*transfer);
        } else if now.0.saturating_sub(reply.last_emit.0) >= period {
            to_emit.push((reply.source, *reply));
        }
    }
    for (source, reply) in to_emit {
        outbox.push_flow(
            source,
            MsgClass::Saga,
            &InterShardFlow::CrossingAborted(CrossingAborted {
                subject: reply.subject,
                transfer: reply.transfer,
            }),
        );
        if let Some(e) = runtime.pending_abort_replies.get_mut(&reply.transfer) {
            e.last_emit = now;
            runtime
                .pending_writes
                .push((StoreKey::AbortReply(reply.transfer).bytes(), Some(encode(e))));
        }
    }
    for transfer in reaped {
        runtime.pending_abort_replies.remove(&transfer);
        runtime
            .pending_writes
            .push((StoreKey::AbortReply(transfer).bytes(), None));
    }
```

`is_confirmed_dead` (312) is the freshness-gated accessor the saga re-drive uses — NOT `is_latched_dead` (330, the reaper's split-brain gate). Confirmed both exist. `let liveness = &runtime.liveness` avoids the closure-borrow conflict (mirrors `roster = &runtime.roster` at 1817).

**Residual risk (honest):** re-persisting `last_emit` on every emit adds a store write per re-emit. Under a rehydrated mass of stranded replies this is bounded by `period` (not per-tick). If a future audit finds even that too heavy, the `last_emit` write can be dropped from the persist (RAM-only throttle, re-emitting once on restart is harmless — the source re-acks). Flagged, not owed.

### D2 tests
- `scan_deadlines_reemits_pending_abort_reply`: live source, `last_emit=0`, period-elapsed → one `CrossingAborted{subject,transfer}` to `reply.source`; entry still present; `last_emit` bumped to `now`.
- `scan_deadlines_throttles_within_period`: `last_emit=now-1`, `period=5` → NO emit (covers the throttle false arm — M-1's whole point).
- `scan_deadlines_reaps_dead_source`: source confirmed dead → NO emit, entry REMOVED, `pending_writes` `(AbortReply.bytes(), None)`.
- `scan_deadlines_reemits_until_dead_then_reaps`: two period-spaced scans (two emits), then mark dead, third reaps.

**HR5:** the three-way `if dead / else-if throttle-elapsed / (implicit else no-op)` — dead test hits arm 1, reemit test hits arm 2, throttle test hits the implicit else. Split is clean (no compound `&&` in one condition — the throttle is its own `else if`). `is_confirmed_dead` via shared borrow. `assert_eq!` on counters/outbox.

---

## 3f-D3 — ORCHESTRATOR: `CrossingAbortedAck` consumer + H-2 start-arm guard

**File:** `crates/node/src/saga_runtime.rs`.

### D3.1 — ack consumer (in the `drive_sagas` demux at ~2100-2163, before `_ => {}` at 2163)

The wildcard comment at 2160-2162 already names the pair. Insert:

```rust
            // 3f-D (Mechanism Y): the SOURCE acked a crossing-abort reply — drop the pending entry +
            // persist-DELETE. `is_some()`-gated so a REDELIVERED ack (ReDriven; intershard.rs:495) is an
            // idempotent no-op.
            Ok(InterShardFlow::CrossingAbortedAck(ack)) => {
                if runtime.pending_abort_replies.remove(&ack.transfer).is_some() {
                    runtime
                        .pending_writes
                        .push((StoreKey::AbortReply(ack.transfer).bytes(), None));
                }
            }
```
Then edit the 2160-2162 comment to drop `CrossingAbortedAck` from the wildcard list (`CrossingAborted` STAYS in the wildcard — the orchestrator emits it, never consumes it). `ack` payload is `CrossingAborted{subject,transfer}` (the wire reuses the struct, `intershard.rs:250`).

### D3.2 — H-2 start-arm guard (in `handle_crossing_request` at 1306; the redelivery guard is at 1324-1325)

**Adversary H-2 folded.** A ttl-re-drive (if ever enabled) or a redelivered request for a `transfer` that has a LIVE `pending_abort_replies[transfer]` must NOT start a fresh saga (the saga just aborted; a restart would race the abort-reply). Extend the guard at 1324:

```rust
        // REDELIVERY / POST-ABORT guard: absorb a re-delivered request whose saga is already live (the
        // steady-state ReDriven case) OR whose transfer has a LIVE abort-reply pending (3f-D H-2: the saga
        // just aborted; do NOT restart it and race the source-latch-clear — the source stops re-driving once
        // it receives the CrossingAborted).
        (Some(_subj), Some(_dest_rec), Some(_sess_rec))
            if runtime.sagas.contains_key(&transfer)
                || runtime.pending_abort_replies.contains_key(&transfer) => {}
```

**Note:** with the id staying 2-arg (no attempt-stamp), the re-driven request and the abort-reply share the same `transfer` — so `contains_key(&transfer)` keys cleanly (the adversary flagged this exact alignment). This is only correct BECAUSE we abandoned the attempt-stamp; had we stamped, the ids would diverge and this guard would miss.

### D3 tests
- `crossing_aborted_ack_drops_pending_entry`: seed one entry; feed `CrossingAbortedAck{transfer}` → removed + `pending_writes` DELETE.
- `crossing_aborted_ack_redelivery_is_idempotent`: ack twice → second is `is_some()`-false no-op.
- `crossing_aborted_ack_unknown_transfer_is_noop`.
- `handle_crossing_request_absorbs_request_with_live_abort_reply` (H-2): seed a `pending_abort_replies[transfer]`, feed the matching `CrossingRequest` with all heads resolved → NO new saga started (`crossings_started` unchanged).

**HR5:** `if remove(..).is_some()` — true arm (drop test), false arm (redelivery/unknown). Split the H-2 `||` guard: `handle_crossing_request_absorbs_request_with_live_abort_reply` (second operand true, first false) + the existing contains_key(sagas) test (first true). `assert_eq!` on counters.

---

## 3f-D4 — SOURCE: unconditional ack + AUDIT-L1 dwell re-arm (the sim-side change)

**File:** `crates/sim/src/stub.rs`. This is where CRITICAL-1 and CRITICAL-2 are resolved.

### D4.1 — CRITICAL-2: thread `CrossingProgress` via a bundle, NOT a 16th param

`on_directory_reply` (3089-3104) lacks `progress`. `process_inbound` (986-1005) is at the 16-param ceiling with `ghost_state: (ResMut<GhostColliderRegistration>, ResMut<SourceGhostMirror>)` already bundled (comment 999-1001). Extend the existing bundle pattern — add `progress` to the ghost/crossing store bundle rather than a new top-level param:

At `process_inbound` (1004-1005), replace the two separate `in_flight`/… lines by folding `progress` into the bundle. The cheapest change that stays under 16: extend `ghost_state` to a 3-tuple:

```rust
    ghost_state: (
        ResMut<GhostColliderRegistration>,
        ResMut<SourceGhostMirror>,
        ResMut<CrossingProgress>,
    ),
```
Destructure at 1007: `let (mut registration, mut mirror, mut progress) = ghost_state;`
Thread `&mut progress.0` into the `on_directory_reply(...)` call (1033-1048), adding a `progress: &mut BTreeMap<EntityId, CrossingState>` param to `on_directory_reply` (3089-3104).

**Honest residual risk (OPEN QUESTION Q2):** `CrossingProgress` is a `ResMut` in `evaluate_realm_boundaries` (`stub.rs:2305`) — a DIFFERENT system. Bevy forbids two systems holding `ResMut<CrossingProgress>` concurrently only if they run in parallel; they are ordered in the same schedule (`process_inbound` then `evaluate_realm_boundaries`, per the system order at 857-874), so a `ResMut` in each is fine (bevy serialises conflicting `ResMut` access). **But confirm `process_inbound` and `evaluate_realm_boundaries` are not in an ambiguous parallel set** — a 1-line schedule read (Q2). If bevy complains, the fix is an explicit `.before()`/`.after()` ordering (likely already present via the tick pipeline).

### D4.2 — CRITICAL-1: the re-cross-after-abort mechanism (NO attempt-stamp, NO wire growth)

Rewrite `on_crossing_aborted` (1629-1646). Two obligations: (a) unconditional ack (leak-guard), (b) L1 dwell re-arm so a still-dwelling entity re-requests without physically re-crossing.

```rust
fn on_crossing_aborted(
    abort: CrossingAborted,
    in_flight: &mut RequestInFlight,
    progress: &mut CrossingProgress,
    stats: &mut StubStats,
    config: &StubConfig,
    outbox: &mut OutboundBox,
) {
    let Some(entity) = abort.subject.transfer_subject_entity() else {
        stats.crossing_abort_no_entity += 1;
        ack_crossing_abort(&abort, config, outbox); // UNCONDITIONAL ACK path 1
        return;
    };
    // Clear ONLY on an exact id match (value equality — the false arm is a covered no-op, HR5(d)). A stale
    // abort for a superseded/re-latched transfer must NOT free a live crossing.
    if in_flight.0.get(&entity) == Some(&abort.transfer) {
        in_flight.0.remove(&entity);
        stats.crossing_latches_cleared += 1;
        // AUDIT-L1 re-arm: a STILL-DWELLING entity (physically past the boundary) must re-REQUEST without
        // re-crossing. Drop its dwell entry so the next `evaluate_one_subject` is a fresh rising edge that
        // re-latches. The re-latch mints the SAME id X (the id is (subject, subject_fence)-deterministic and
        // the fence is unchanged on a pre-CAS abort) — this is INTENTIONAL and SAFE: a stale abort for X
        // that arrives AFTER the re-latch will match X and clear it again, but the source simply re-arms
        // AGAIN and re-requests; the orchestrator's `pending_abort_replies`/`sagas.contains_key` guard
        // absorbs the churn, and each round-trip is one ack. Convergence is by the ack, not a generation id.
        progress.0.remove(&entity);
    } else {
        stats.crossing_abort_stale += 1;
    }
    ack_crossing_abort(&abort, config, outbox); // UNCONDITIONAL ACK paths 2 (clear) + 3 (stale)
}

/// Emit the `CrossingAbortedAck` for `abort` to the orchestrator. Monomorphic (one push) so any of the
/// three ack callers covers it (HR5).
fn ack_crossing_abort(abort: &CrossingAborted, config: &StubConfig, outbox: &mut OutboundBox) {
    outbox.push_flow(
        config.orchestrator,
        MsgClass::Saga,
        &InterShardFlow::CrossingAbortedAck(CrossingAborted {
            subject: abort.subject,
            transfer: abort.transfer,
        }),
    );
}
```

Update the demux caller at 3182-3185: `on_crossing_aborted(a, in_flight, progress, stats, config, outbox);` (`progress` now in scope from D4.1; `config`/`outbox` are `on_directory_reply`'s own params — confirmed).

**Adversary H-1 folded (the `n_entry` trap).** The re-arm drops the dwell entry → `evaluate_one_subject`'s `progress.entry(entity).or_default()` (2429) re-creates it with `inward_ticks=0`. On the SAME tick (process_inbound runs before evaluate_realm_boundaries), `now_member` recomputes and `inward_ticks` becomes 1. With default `n_entry=3` (`geometry.rs:509`), the re-latch needs 3 fresh in-band ticks — benign. **With `n_entry==1` (a valid tuning, gate is `>0` at 519), `inward_ticks==1==n_entry` fires the SAME tick** → an immediate re-request of the identical id X. This is bounded (each is one ack round-trip) but the E-tests MUST pin `n_entry` and add an `n_entry==1` case proving the same-tick re-latch re-emits X and the abort/ack loop still converges.

**CRITICAL-1 resolution rationale (the load-bearing honesty):** I deliberately do NOT prevent the re-latch from re-minting X. The DRAFT tried to make X unique per attempt (broke the frozen wire + documented idempotency). The adversary tried to keep a surviving counter (still needs the wire growth). **The frozen wire's own idempotency guarantee IS the design: the same (subject, fence) → same id, and correctness comes from the ack round-trip draining `pending_abort_replies`, not from id uniqueness.** A stale abort for X after a re-latch clears the re-latch, the source re-arms and re-requests X, the orchestrator (which already dropped its `pending_abort_replies[X]` on the first ack) starts a fresh saga for X. This is a bounded thrash, not a strand. See Q1 for the one scenario where this bound must be checked.

### D4 tests
- `crossing_aborted_acks_on_id_match_and_rearms`: latch `X` + seed a `CrossingProgress` dwell; abort `X` → latch removed, `crossing_latches_cleared==1`, dwell entry gone, exactly one `CrossingAbortedAck{transfer:X}` to `config.orchestrator`.
- `crossing_aborted_acks_on_stale`: latch `X`, abort `Y` → latch unchanged, `crossing_abort_stale==1`, dwell unchanged, `CrossingAbortedAck{transfer:Y}` STILL emitted (the leak-guard).
- `crossing_aborted_acks_on_no_entity`: non-Entity subject → `crossing_abort_no_entity==1`, ack still emitted.
- Update existing `stub.rs:8248` (`the_crossing_aborted_demux_clears_the_latch_on_an_id_match_only`) and `stub.rs:8313` (`a_saga_demote_clears_the_durable_crossing_latch`) for the new arity + assert the ack now fires.

**HR5:** all three ack paths call monomorphic `ack_crossing_abort` (one push, covered by any caller; the three tests exercise all three callers → full region coverage of the branch structure). `if get(&entity) == Some(&abort.transfer)` is value-equality (`assert_eq!`-friendly; false arm = covered stale no-op, per HR5(d), NOT `matches!`). No `&&` to split. `ack_crossing_abort` is a plain fn (one monomorphization).

---

## 3f-E — CAPSTONE: unit sweep + SIGKILL crash-leg e2e

### E.1 — happy abort round-trip (`tests/src/lib.rs` or the crossing scenario suite)
Source latches (id X) → orchestrator aborts pre-CAS → `commit_result` inserts+persists → `scan_deadlines` emits `CrossingAborted{X}` → source clears latch + L1-re-arms + acks → orchestrator drops entry + persist-DELETE. Assert: source latch empty, orchestrator map empty, store has no `AbortReply` record. Then the L1 proof: a still-dwelling entity re-requests **without re-crossing** — step the source one tick, assert a NEW `CrossingRequest` (**id X again — NOT X'**, since no attempt-stamp; this is the corrected assertion vs the DRAFT's wrong `X'≠X`).

### E.2 — lost-first-ack leak-guard
Drop the source's first ack; `scan_deadlines` re-emits `CrossingAborted{X}` (after `period`); source finds the latch already cleared → stale path but STILL re-acks; orchestrator drops the entry. Assert the map does NOT leak. **This is the exact failure the unconditional-ack exists to prevent — name it a load-bearing assertion (adversary M-2).**

### E.3 — H-1 `n_entry==1` same-tick re-latch (NEW, adversary H-1)
Pin `n_entry=1`; drive an abort for a still-dwelling entity; assert the re-latch fires the SAME tick, re-emits id X, and the abort/ack loop converges (map drains) rather than live-locking. Also a default-`n_entry=3` variant proving the multi-tick dwell.

### E.4 — stale-abort-after-re-latch churn bound (replaces the DRAFT's H2 test)
Abort attempt (id X); source re-latches (id X again); a STALE `CrossingAborted{X}` arrives → source clears the re-latch + re-arms + re-acks; assert the loop CONVERGES within a bounded number of round-trips (the orchestrator's `pending_abort_replies[X]` was already dropped on the first ack, so no NEW re-emit is generated by the stale ack). **This is the honest test of the CRITICAL-1 resolution — it proves the bounded-thrash claim, replacing the DRAFT's impossible "X'≠X" proof.**

### E.5 — SIGKILL crash-leg (the load-bearing capstone; `crates/bins/tests/`, mirror `process_parity.rs` / the R-6d4-D restart harness, task #115)
1. Orchestrator on real `RedbStore`.
2. Drive a durable crossing to a pre-CAS Aborted tombstone → `pending_abort_replies[X]` inserted + **persisted to redb** (barrier commits before this tick's effects).
3. **SIGKILL the orchestrator between abort-persist and the source ack** — TWO explicit variants (adversary M-2): (a) BEFORE the first `scan_deadlines` emit; (b) AFTER emit but BEFORE the source's ack returns (proves an ack arriving at a dead orchestrator is harmlessly re-derivable via re-emit → re-ack).
4. Restart → `rehydrate` scans `StoreKey::ABORT_REPLY` → `pending_abort_replies[X]` restored.
5. Post-restart `scan_deadlines` re-emits `CrossingAborted{X}`.
6. Source (still holding its latch) clears on the id match + acks.
7. Assert: source latch clears across the restart, orchestrator map + store entry drop on the ack, entity free to re-cross. **This proves Mechanism Y's durability is real — a RAM-only map would have lost X on the kill and stranded the source forever.**

**HR5:** integration/e2e exercise the full surface (HR5(c)). Only `encode<T>` is generic; `PendingAbortReply` monomorphization is covered by its persist site (D1 unit) AND here. The SIGKILL test is release-gated (process test), sequenced like the R-6d4-D restart proof; asserts observable state (latch cleared, map empty, store scan empty) via `assert_eq!`, never `matches!`.

---

## Determinism / frozen-wire compliance

- **BTreeMap only:** `pending_abort_replies` is `BTreeMap<TransferId, PendingAbortReply>` — deterministic key-order emit/reap.
- **No rng/wall-clock:** id is FNV over postcard; `now`/`last_emit` are the universe tick (seam-supplied); `is_confirmed_dead` keys on `now`.
- **FROZEN WIRE UNTOUCHED:** `crossing_transfer_id` stays 2-arg; `CrossingRequest` stays 5-field; `CrossingAborted`/`CrossingAbortedAck` arms + their `EffectClass::SideEffecting`/`FlowDurabilityClass::ReDriven` classifications (`intershard.rs:417,426,494-495`) already exist. **Zero wire growth** — the DRAFT's entire 3f-D0 (arity change + field-append) is deleted. This is the single biggest correctness improvement over the DRAFT.

---

## OPEN QUESTIONS (must be resolved by a code read before the named sub-slice)

- **Q1 (gates D4.2 — the CRITICAL-1 bound; MUST resolve before coding D4).** With no attempt-stamp, a stale `CrossingAborted{X}` arriving after an L1 re-latch clears the live re-latch, forcing a re-request. I have argued this converges (the orchestrator already dropped `pending_abort_replies[X]` on the first ack, so the stale ack generates no new re-emit). **But confirm there is no path where the orchestrator re-inserts `pending_abort_replies[X]` for the SAME X while a re-request is in flight** — specifically, read `run_to_quiescence` + `start_transfer` to confirm a re-requested saga for X that ABORTS AGAIN re-inserts `pending_abort_replies[X]`, and that the ack for the earlier abort cannot drop the LATER re-inserted entry (ABA on X). If ABA is possible, the bound is not clean and a monotone `attempt` field on `PendingAbortReply` (orchestrator-only, NOT wire) is the minimal fix — the counter lives on the durable reply, not the wire id. **This is the one unresolved risk; it decides whether the slice is 2-arg-clean or needs an orchestrator-internal generation.** ~15-line read of `run_to_quiescence` (812) + `start_transfer`.

- **Q2 (gates D4.1).** Confirm `process_inbound` and `evaluate_realm_boundaries` are not in an ambiguous parallel `ResMut<CrossingProgress>` set — 1-line schedule read at `stub.rs:857-874`. If ambiguous, add explicit `.after(process_inbound)` on `evaluate_realm_boundaries` (likely already ordered via the tick pipeline).

- **Q3 (gates D2).** Confirm the tuning home for `abort_reply_period_ticks` — `runtime.tunings.transfer` (`TransferTuning`) vs a sibling. Grep `struct TransferTuning` / `tunings.transfer` in `saga_runtime.rs`. If `TransferTuning` is the right home, add the field there (no inline literal — HR5 no-magic-numbers). If tunings aren't threaded to `scan_deadlines`, thread the one value.

---

## RESIDUAL RISK (honest)

1. **CRITICAL-1's resolution depends on Q1.** The bounded-thrash argument is sound IF there is no ABA re-insert on X. If Q1 finds ABA, add an orchestrator-internal `generation` on `PendingAbortReply` (still no wire change). Everything else in the plan is unaffected — this is a localized contingency, not a re-plan.
2. **`n_entry==1` same-tick re-latch** is bounded but produces one re-request per tick until geometry changes. Covered by E.3; acceptable because production `n_entry=3`. A future tuning validation could forbid `n_entry==1` for durable-authority boundaries — flagged, not owed.
3. **`last_emit` re-persist cost** (M-1 throttle) is one store write per re-emit; bounded by `period`. Downgradable to RAM-only throttle if a load audit demands (harmless single re-emit on restart). Flagged.
4. **The SIGKILL after-emit-before-ack variant (E.5b)** relies entirely on the unconditional-ack (D4.2) for convergence — if D4.2's stale-path ack were ever removed, E.5b would strand across a restart. The dependency is now a named E.2 + E.5b assertion, so a regression is caught.

**No strand or double-free found** in the crash-window analysis (the abort-persist and the saga tombstone's `Saga=None` + directory `ClearTransferLock` all ride the same end-of-tick barrier → atomic). The DRAFT's crash-window claims hold; the DRAFT's H2 attempt-stamp and wire growth were the only structural errors, and both are eliminated.

---

## Files (all absolute)
- `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/node/src/saga_runtime.rs` — D0 (type+field+StoreKey+rehydrate), D1 (`commit_result` insert, ~1142), D2 (`scan_deadlines` re-emit+reap, ~1863), D3 (ack consumer ~2159 + H-2 guard ~1324).
- `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/sim/src/stub.rs` — D4 (`on_crossing_aborted` ~1629 unconditional-ack+L1; `process_inbound` ~1004 bundle; `on_directory_reply` ~3089 param; demux caller ~3182).
- `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/wire/src/intershard.rs` — **UNTOUCHED** (frozen wire already carries the pair + classifications; the DRAFT's edits here are deleted).
- `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/tests/src/lib.rs` and `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/bins/tests/` — 3f-E integration + SIGKILL crash-leg (mirror `process_parity.rs`, task #115).