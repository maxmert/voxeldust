# R-6d3a — MF-3 CORRECTION (implementer finding, adversarially verified opus)

The vetted synth's MF-1 fix (submit_nonblocking + depth-2 channel + try_send-panic-on-Full) has a
CONFIRMED multi-producer flaw: N distinct peer_writer tasks share ONE Arc<Mutex<NodeOutbox>>; the lock
serializes block A but the fsync WAIT is outside it, so >=3 concurrent durable submitters pile >depth-2
batches into the channel before the single writer drains one fsync -> try_send Full -> PANIC under the
exact "source fans TransientBatch to several dest peers + Despawn to neighbors" load the slice targets.
The synth's "depth-2 exactly enough" proof reasons about ONE writer's consecutive iterations, not N
concurrent writers. FLAW CONFIRMED (verifier, code-grounded).

CORRECTED DESIGN (MF-3), verified SOUND leg-by-leg (durable-before-send / seq-durability monotonicity /
crash-safety; no out-of-order-send, no false-durable, bounded memory):
1. The OUTBOX RedbStore gets its OWN generous writer_channel_depth via a NAMED const
   OUTBOX_WRITER_CHANNEL_DEPTH (HR no-magic-numbers; one home in outbox.rs) >> realistic peer count, so
   try_send Full is unreachable BY CAPACITY, not by an unproven timing invariant. Orchestrator store keeps
   depth-2 + block-on-prior UNCHANGED (its persist-before-effect reconcile-scan needs the depth-1 bound;
   the outbox has NO reconcile scan — scan_all is boot-only — so it never needed that bound).
2. submit_nonblocking stays block-on-prior-FREE; seq-assign + try_send stay UNDER the lock (the sole
   out-of-order-send hazard is avoided by NOT moving try_send outside the lock). On Full: panic (now a
   dead-code impossible-invariant tripwire given depth >> peers).
3. Block B waits on a SINGLE wait_durable_through(bar.seq). Drop bar.prior/wait_prior entirely: durability
   is monotone (writer drains FIFO == seq order, last_durable.store(max_seq) ascending), so
   is_durable_through(seq) implies is_durable_through(prior). SubmitBarrier collapses to the seq;
   API: submit_barrier(&mut self) -> Option<(u64 seq, DurabilityHandle)> (submit + handle-clone under lock).
4. LOAD-BEARING invariant to keep: block C (wire send) STRICTLY after block B; the crash-safety of the now-
   wider (all-un-fsynced, not <=1) outbox window rests on "un-durable == un-sent == source re-emits".
   MF-2 None-barrier pin retained (a durable retain always staged -> Some; debug_assert).

The rest of scripts/r6d3a_vetted_design.md stands (injection shape, write_frame block A/B/C, crash table,
tests T-DBS-1..4 + the pure-unit non-serialization mirror, scope=bins pass None, files touched).
# R-6d3a Vetted Design of Record (workflow wf_e8bea726)

## ADJUDICATED SYNTHESIS (design of record)

All facts confirmed. `writer_channel_depth` default is 2 (store.rs:97) — depth-2 slack, effective depth-1 via block-on-prior. The channel has one free slot beyond the in-flight batch. This is the invariant that makes the fix safe: with block-on-prior guaranteeing the prior batch drained before the current send, the channel `send` never blocks. I now have everything needed to adjudicate.

The unanimous HIGH (confirmed against store.rs:556-567 + outbox.rs:189-196) is the single blocking issue and it collapses across all three lenses to one root cause. I'll fold in the fix, downgrade the secondary MEDIUM/LOW findings the code already handles or that follow from the HIGH, and emit the corrected design.

---

# R-6d3a — Wire the Shared Outbox Sink + Durable-Before-Send Gate: ADJUDICATED Design of Record (IMPLEMENT-READY)

**VERDICT: SOUND_TO_IMPLEMENT** — with two folded must-fixes (MF-1 the non-serialization API reshape; MF-2 the submit-seq contract). One HIGH, unanimously raised and code-confirmed, is fixed in-place below. All other findings are NIT-level or already handled by existing code; adjudicated inline.

Scope unchanged from the DESIGNER: (1) inject ONE shared `NodeOutbox` sink per node into `spawn_mesh → PeerWriter → write_frame/on_ack`, flipping the two literal `None`s; (2) the durable-before-send GATE. R-6d3b (boot replay + `gc_below`), R-6d3c (saga `AwaitAdopt` split + dest `TransientDiscard`), R-6d4 (SIGKILL e2e + proptest) remain OUT of scope and named. Line cites are against worktree HEAD (`ebbe544`, R-6d2c landed).

---

## 0. Adjudication summary — what the reviewers found, ranked by true impact

| # | Finding | Lenses | Verdict vs code | Disposition |
|---|---|---|---|---|
| **MF-1** | `submit()` = `store.commit()` runs BLOCK-ON-PRIOR **under the sink lock** → peer writers convoy on the prior batch's fsync; §3.2 non-serialization proof is FALSE | all 3 (HIGH / HIGH / MEDIUM) | **CONFIRMED** — `store.commit()` unconditionally block-on-priors at store.rs:563-567; `NodeOutbox::submit` as specced calls `store.commit()` (outbox.rs:189-193). The DESIGNER's own Open Questions #3/#5 conceded it but left it in §3.2 as a settled proof. | **UPHELD as the single blocking HIGH. Folded fix below (MF-1): `submit()` must NOT run block-on-prior under the lock. Reshape so the lock covers only the pure-RAM stage + non-blocking channel send.** |
| **MF-2** | `submit()` under-specified for the shared store: `last_submitted()` after `commit()` could name a stale batch (empty-staged early return; the Open-Q#5 "stage-without-submit" variant) → `wait_durable_through(stale_seq)` passes before this frame is durable | crash-lens (MEDIUM) | **CONFIRMED as a latent contract hole.** `commit()` early-returns on empty staged without bumping `last_submitted` (store.rs:568-570); a stale/zero seq fast-returns from the wait (store.rs:272). On the durable path a `retain` always precedes, so staged is non-empty *today* — but the contract is unwritten and MF-1's reshape must preserve it. | **UPHELD as a contract-pin, folded into MF-1's API. `submit()` returns the seq of the batch carrying THIS span's stage, or `None` if nothing was staged; block B treats `None`/"not-yet-covered" as "must not proceed as durable."** |
| 3 | `wait_durable_through` parks a tokio **worker thread** (sync condvar, not `.await`) | concurrency (LOW) | CONFIRMED but **accepted posture** — Slice-D set this precedent for the flush gate (store.rs:162-176 doc); happy path fast-returns (store.rs:272). Amplification was purely a consequence of MF-1. | **DOWNGRADED to NIT.** Once MF-1 lands, the worst case is bounded to peers actually sending a durable frame during a stall — correct back-pressure. No change; flag `spawn_blocking` as an R-6d4 load-test lever only. |
| 4 | Idle-lane release tombstones never fsync (crash-table rows 6/7 "at most one re-delivery" understates to "up to retry-cap") | crash-lens (LOW) | CONFIRMED as a **wording** defect, not a correctness defect — a lost tombstone only ever causes idempotent re-delivery (verified against `on_ack` retire + receiver dedup). No flush-tick is wired in 3a. | **ACCEPTED + folded into the crash table.** Rows 6/7 corrected to "up to the retry-cap window's acked count," and the residue is explicitly bounded (swept by 3b `gc_below`; capped in-incarnation by `retry_buffer_max_bytes`). No idle-commit added in 3a (out of scope; named for 3b). |
| 5 | `await_holding_lock` risk — a `std::sync::MutexGuard` held across block C's `.await` | concurrency (NIT) | The DESIGNER already drops the guard before block B. **Already handled** by the ordering. | **DOWNGRADED to a lint gate.** Fold: enable clippy `await_holding_lock` for io-prod so a regression is caught mechanically. |
| 6 | `DurabilityHandle::already_durable()` test-ctor must stay `#[cfg(test)]`/`pub(crate)`; `MockOutboxSink::submit` must return a monotone non-zero seq | HR-lens (NIT×2) | Valid test-hygiene points. | **ACCEPTED, folded into §6.** |

**Net:** the DESIGNER's *core crash-correctness guarantee is SOUND and I re-derived it independently* — retain→submit→wait→send genuinely fsyncs the row before its first wire appearance, and every crash-table row is NO-LOSS / NO-double-adopt. The ONE real defect is the concurrency API shape (MF-1), which is fixable without touching the ordering or the crash guarantee. That fix is folded below.

---

## 1. The API shape — inject the shared sink + the MF-1 non-serialization split

### 1.1 The root defect and its cure (MF-1)

The hazard the slice exists to prove absent: **a peer writer holding the shared `NodeOutbox` `Mutex` across an fsync serializes every other writer.** The DESIGNER moved the *current-batch* wait outside the lock (correct) but left `submit()` = `store.commit()`, whose FIRST action is BLOCK-ON-PRIOR (store.rs:563-567):

```rust
// store.rs:563-567 — runs at the TOP of commit(), i.e. under the sink lock as specced:
let last_submitted = self.last_submitted.load(Acquire);
if self.last_durable.load(Acquire) < last_submitted {
    self.fsync_backpressure.fetch_add(1, Relaxed);
    self.wait_durable_through(last_submitted);   // ← BLOCKING PARK, under the lock
}
```

With ONE shared store (one `last_submitted`/`last_durable` pair, depth-1), peer B's `submit()` block-on-priors on peer A's un-fsynced batch **while holding the shared lock** — the exact convoy the split was meant to remove.

**The cure — a new object-safe `stage_and_submit` that does the block-on-prior WAIT *before* taking anything blocking, and does the channel hand-off non-blocking under the lock.** The key realization from the code: block-on-prior exists to keep in-flight ≤1 batch, and the writer channel has **depth-2 slack** (`writer_channel_depth: 2`, store.rs:97/411). So the RAM stage + a `tx.send` for batch N+1 can proceed WITHOUT waiting for batch N to fsync, *as long as the depth-2 channel has a free slot* — which it does when at most one batch (N) is in flight. The block-on-prior wait is therefore not needed to make the send non-blocking; it is only needed to *bound* in-flight to ≤1. We move that bounding wait OUTSIDE the lock, keeping in-flight ≤2 momentarily (absorbed by depth-2) and re-establishing ≤1 before the next stage.

Concretely, add to `RedbStore` a split of `commit()` into a non-blocking submit and an out-of-lock prior-wait:

```rust
// store.rs — NEW on RedbStore, additive; `commit()` stays as the composed batched-flush path.

/// Non-blocking submit: stage is already in `self.staged`. Hands the batch to the writer WITHOUT
/// block-on-prior. Returns the submitted seq, or `None` if nothing was staged (idle). The depth-2
/// writer channel absorbs one in-flight batch + this one, so `tx.send` does NOT block here (the
/// caller re-establishes the ≤1-in-flight bound via `wait_prior_durable` OUTSIDE its lock — MF-1).
pub fn submit_nonblocking(&mut self) -> Option<u64> {
    if self.staged.is_empty() {
        return None;                       // MF-2: idle span ⇒ nothing to gate on
    }
    let batch = std::mem::take(&mut self.staged);
    let seq = self.next_seq + 1;
    let tx = match &self.submit_tx {
        Some(tx) => tx.clone(),
        None => { tracing::error!("submit_nonblocking with no writer (post-Drop) — retained");
                  self.staged = batch; return None; }
    };
    // Depth-2 slack guarantees a free slot when in-flight ≤1 (the invariant every caller upholds by
    // waiting on the PRIOR seq before its NEXT submit). try_send never rendezvous-blocks under the lock;
    // a full channel means the ≤1 invariant was violated → FAIL LOUD (never block a peer under the lock).
    match tx.try_send((seq, batch)) {
        Ok(()) => { self.next_seq = seq; self.last_submitted.store(seq, Release); Some(seq) }
        Err(TrySendError::Full(_)) =>
            panic!("submit_nonblocking: writer channel full — the ≤1-in-flight invariant was violated \
                    (a caller submitted without waiting on the prior seq); refusing to block under the lock"),
        Err(TrySendError::Disconnected(rejected)) => {
            // Writer DIED while the store is live — same fail-loud posture as commit() (store.rs:593-599):
            self.staged = rejected.1; // retain (fail-safe)
            panic!("submit_nonblocking: writer channel disconnected — writer thread gone; refusing to \
                    lose a durable batch (recovery rehydrates + re-drives)");
        }
    }
}

/// The block-on-prior bound, callable OUTSIDE any sink lock: park until `prior` is durable so in-flight
/// returns to ≤1 before the next stage. `prior` is the seq captured BEFORE this submit (0 ⇒ genesis, no-op).
pub fn wait_prior_durable(&self, prior: u64) { self.wait_durable_through(prior); }
```

**Why depth-2 is exactly enough (the re-verified invariant, per the concurrency reviewer's fix-request):** every writer, before it `submit_nonblocking`s batch N, has already (in its PRIOR iteration) waited on batch N−1's durability *or* is the first submitter. At the instant of `try_send`, at most one un-fsynced batch (N−1) occupies the channel; N takes the second slot; the writer thread drains N−1 (freeing slot 1) concurrently. So `try_send` never sees `Full` under correct use — and if it ever does, that is a real invariant violation and we fail loud rather than silently block a peer under the lock. This preserves the depth-1 *crash-loss bound* (≤1 batch lost on crash) because the block-on-prior wait (`wait_prior_durable`) still runs every cycle — just outside the lock.

### 1.2 The `OutboxSink` trait — the MF-1/MF-2 shape

Replace the DESIGNER's `submit()->u64` + `durability()` with a split that never blocks under the lock and pins the seq contract:

```rust
pub trait OutboxSink: Send {
    fn retain(&mut self, key: &OutboxKey, framed: &[u8]);
    fn release(&mut self, key: &OutboxKey);
    fn commit(&mut self);                       // UNCHANGED: submit + block-until-durable (future flush-tick)
    fn scan_all(&self) -> Vec<(OutboxKey, Vec<u8>)>;
    fn gc_below(&mut self, incarnation: u64);

    /// NEW (R-6d3a, MF-1/MF-2): submit the CURRENTLY-STAGED batch WITHOUT any blocking wait, returning the
    /// durability BARRIER for the caller to await OUTSIDE the lock. `prior` is the previous submitted seq —
    /// the caller waits on it (outside the lock) to re-establish the ≤1-in-flight bound BEFORE its next stage.
    /// Returns `None` if nothing was staged (MF-2: an idle/release-only span produces no durable barrier ⇒
    /// block B must NOT treat it as "already durable" and proceed to send). The returned `seq` is guaranteed
    /// to cover every row this locked span staged (a `retain` on the durable path always precedes ⇒ non-empty).
    fn submit_barrier(&mut self) -> Option<SubmitBarrier>;

    /// A clone of the durability watermark handle — parkable WITHOUT the sink lock (block B).
    fn durability(&self) -> DurabilityHandle;
}

/// The durability barrier a locked stage span hands out. `seq` is the batch carrying this span's rows;
/// `prior` is the seq to await to re-bound in-flight to ≤1. Both awaited on a cloned `DurabilityHandle`
/// OUTSIDE the sink lock (MF-1).
#[derive(Clone, Copy, Debug)]
pub struct SubmitBarrier { pub seq: u64, pub prior: u64 }
```

`NodeOutbox` impl:

```rust
fn submit_barrier(&mut self) -> Option<SubmitBarrier> {
    let prior = self.store.last_submitted();           // capture BEFORE the submit (0 at genesis)
    self.store.submit_nonblocking().map(|seq| SubmitBarrier { seq, prior })
}
fn durability(&self) -> DurabilityHandle { self.durability.clone() }
// `commit` unchanged (submit + wait) for the future batched flush-tick path.
```

MF-2 is now structural: `submit_barrier` returns `None` iff `submit_nonblocking` staged nothing, and the caller (block B) never sends a durable frame on a `None` — it can only reach the wire via the non-durable path, which the `durable` bool already gates. On the durable path a `retain` always precedes, so `None` there is unreachable (a `debug_assert!(barrier.is_some())` in block A when `durable` makes it executable).

### 1.3 The shared sink: ONE `NodeOutbox` per node, `Arc<Mutex<Box<dyn>>>`, cloned per writer

```rust
// mesh.rs — near the ConnRegistry alias (~:54):
type SharedOutbox = std::sync::Arc<std::sync::Mutex<Box<dyn OutboxSink + Send>>>;
```

- `MeshConfig` (mesh.rs:57-83): **NO new field** (it is `Clone + Debug`; a live redb sink is neither). Thread the sink as a `spawn_mesh` PARAMETER:

```rust
pub fn spawn_mesh(
    handle: &tokio::runtime::Handle,
    trust: &ClusterTrust,
    cfg: &MeshConfig,
    outbox: Option<SharedOutbox>,   // NEW last param; None = today's behaviour, byte-identical
) -> Result<(MeshTransport, MeshControl), ProdIoError>
```

- Per-peer spawn loop (mesh.rs:852-865): `outbox: outbox.clone()` into each `PeerWriter` (one `Arc`, one store behind it; `OutboxKey.peer` at outbox.rs:82 keeps rows disjoint per peer).
- `PeerWriter` struct (mesh.rs:1142-1160): add `outbox: Option<SharedOutbox>`.

### 1.4 LOW-4 — the SAME `Arc` to both `assign_and_retain` and `on_ack`

`peer_writer` owns `w.outbox`. Both the send path (`write_frame`) and the ack path (`on_ack`, mesh.rs:1195) lock the identical `Arc`, so a durable row retained on it is released on it — no leak. The `on_ack` call site (mesh.rs:1187-1206) locks the shared sink ONCE for the whole ack fan-out and passes `Some(&mut **guard)`, flipping the literal `None` at :1195. Release-through is **staged only** (no `submit`/fsync in the ack path — tombstones ride the next retain's `submit_barrier`, or a future flush-tick; a crash before that harmlessly re-delivers an already-acked frame, receiver dedups). `replay_lanes` (mesh.rs:1273) is NOT given the sink — redial never re-retains (§4).

---

## 2. The durable-before-send gate — exact `write_frame` ordering

`write_frame` gains `outbox: Option<&SharedOutbox>` (from mesh.rs:1209-1224). The Reliable arm's assign block (mesh.rs:1611-1658) widens to produce BOTH `seq` (used by block C) and the durability `gate`; the lock is dropped before ANY wait.

### 2.1 The Reliable arm, in exact order

```rust
Reliability::Reliable => {
    let durable = matches!(frame.durability, vd_sim::io::Durability::Retained);  // mesh.rs:1602, unchanged

    // ── Block A: assign + retain + submit_barrier. Sink locked for THIS span ONLY: a pure-RAM stage
    //    (retain) + a NON-BLOCKING channel hand-off (submit_nonblocking). NOTHING that block-on-priors
    //    runs under the lock (MF-1). The lock is DROPPED before any fsync/prior wait.
    let (seq, gate): (u64, Option<(SubmitBarrier, DurabilityHandle)>) = {
        let mut guard = if durable {
            outbox.map(|o| o.lock().unwrap_or_else(PoisonError::into_inner))
        } else { None };
        let sink: Option<&mut dyn OutboxSink> =
            guard.as_deref_mut().map(|b| &mut **b as &mut dyn OutboxSink);

        let lane = lanes.entry(frame.class)
            .or_insert_with(|| ReliableLaneSender::new(dest, incarnation, retry_cap));
        let seq = match lane.assign_and_retain(local, frame.class, &frame.bytes, durable, sink) {
            Ok(seq) => { lane.last_msg_id = Some(frame.msg_id); seq }
            Err(reject) => { /* ...unchanged shed path, mesh.rs:1624-1656... */
                             stats.reliable_shed.fetch_add(1, Ordering::Relaxed);
                             return Err(WriteFail::Shed(map_reason(reject))); }
        };
        // Submit the just-staged retain NON-BLOCKING; capture (barrier, handle). Only when a real durable
        // row was written (durable AND a sink present) — else the gate is inert. MF-2: a durable retain
        // always staged a row ⇒ submit_barrier is Some; assert it.
        let gate = if durable {
            guard.as_deref_mut().and_then(|b| {
                let barrier = b.submit_barrier();
                debug_assert!(barrier.is_some(),
                    "durable retain staged nothing — submit_barrier must cover this span (MF-2)");
                barrier.map(|bar| (bar, b.durability()))
            })
        } else { None };
        drop(guard);                               // ← LOCK RELEASED HERE, before any wait (MF-1)
        (seq, gate)
    };

    // ── Block B: the durable-before-send BARRIER — OUTSIDE the lock. Two waits, both on the CLONED handle,
    //    neither holding the sink lock: (1) wait_prior_durable re-bounds in-flight to ≤1; (2) wait on THIS
    //    batch's seq so the row is fsynced before the wire. No-op for Ephemeral (gate == None).
    if let Some((bar, durability)) = gate {
        durability.wait_durable_through(bar.prior);   // MF-1: block-on-prior, OUTSIDE the lock
        durability.wait_durable_through(bar.seq);      // this batch durable ⇒ row on disk before send
    }

    // ── Block C: the QUIC send — VERBATIM from mesh.rs:1660-1728 (ensure_connection, open-stream replay
    //    OR steady-state write_reliable_frame). Reached ONLY after the row is durable (for Retained).
    ensure_connection(/* ...unchanged... */).await.map_err(|()| WriteFail::Down)?;
    /* ...the rest of the current Reliable arm, verbatim, using `seq`... */
}
```

**Ephemeral (`durable == false`):** `guard` is `None` (no `lock()`, no contention), `sink` is `None` → `assign_and_retain` takes its existing false arm (`retained = durable && sink.is_some()`, mesh.rs:651) — nothing mirrored, byte-identical to today's literal `None`. `gate == None` → block B is a skipped `if let`. Block C unchanged. The datagram (Unreliable) arm (mesh.rs:1730-1780) is not in this path. **No fsync, no commit, no lock on the Ephemeral / 20 Hz path.**

### 2.2 The gate proves the invariant

For a `Retained` frame (`AwaitAdopt TransientBatch` on `Saga`; band-exit `GhostFlow::Despawn` on `GhostReliable`), block C's `write_reliable_frame` is reached ONLY after `wait_durable_through(bar.seq)` returned — the row is fsynced. A crash in the "durable-but-not-yet-sent" window (§5 row 4) leaves the row on disk → R-6d3b boot replay re-sends it. **A batch frame that can ever be observed on the wire survives the source crash.**

---

## 3. Concurrency argument (corrected for MF-1)

### 3.1 No deadlock
The fsync runs on the `vd-store-writer` thread (store.rs:334-390), spawned outside the tokio runtime (store.rs:426-442). `wait_durable_through` parks a tokio worker on the store's condvar (store.rs:166-176 → `park_until_durable` store.rs:266-290); the writer bumps `last_durable` + notifies (store.rs:369-372). The parked worker holds NO lock the writer needs — the sink `Mutex` was dropped in block A, and the writer thread never touches that `Mutex` (it only drains the crossbeam channel). Parking a worker on the fsync can never block the fsync thread. The liveness escape (store.rs:282-288) panics loud if the writer died — a refusal, never a hang. **No deadlock.**

### 3.2 No cross-peer serialization (MF-1 — the property the slice exists to prove, NOW TRUE)
Each `peer_writer` clones the ONE `Arc<Mutex<NodeOutbox>>`. The `Mutex` is held ONLY across block A, which is now **pure RAM stage (`retain`) + a non-blocking `try_send` (`submit_nonblocking`)** — NOTHING that block-on-priors runs under the lock. Both durability waits (block-on-prior `bar.prior` and current-batch `bar.seq`) are in block B, outside the lock, on a cloned handle. Sequence: peer A locks, stages, `try_send`s batch N (non-blocking, depth-2 slot free), drops the lock, parks OUTSIDE on N−1 then N. Peer B immediately locks, stages, `try_send`s batch N+1 (second depth-2 slot; still non-blocking because A already re-bounds to ≤1 before its next stage), drops the lock, parks outside. **Peer writers never serialize on any fsync under the lock.** The depth-2 channel absorbs the momentary ≤2 in-flight; `wait_prior_durable` re-establishes ≤1 before each writer's *next* stage, preserving the depth-1 crash-loss bound. `T-DBS-3` now asserts the property the prompt demands (peer B's *stage AND submit* complete while A is parked), not the weakened "stage-only" the DESIGNER's Open-Q#3 feared.

### 3.3 No datagram / same-peer head-of-line (HIGH-1)
- **Cross-peer:** eliminated by §3.2.
- **Same-peer:** `peer_writer` is one `select!` (mesh.rs:1185); a datagram to the same peer is a later `w.rx.recv()` iteration (single-consumer FIFO). A `Retained` reliable send that parks in block B delays THAT peer's next frame by at most **one own-peer fsync** (MF-1 removed the cross-peer widening). The datagram CODE is byte-unchanged (`assign_and_retain` is reliable-only, mesh.rs:1595); Snapshot/Input/GhostDelta are `Unreliable` (ride datagrams, mesh.rs:1730) and `Durability::Ephemeral` by default → the gate is skipped, so the 20 Hz latency path never pays the fsync. The added cost is paid ONLY by low-rate producer-less `Retained` sends (`AwaitAdopt` batch, band-exit `Despawn`). A same-peer datagram queued behind a `Retained` send waits at most one fsync (~sub-ms at 50 Hz; a real wait IS disk-stall back-pressure, store.rs:164-165).

### 3.4 Cancel-safety
`write_frame` is still never wrapped in `timeout`/`select!` (mesh.rs:1576-1578); the sole cancel point stays `w.rx.recv()` (mesh.rs:1207). The block-B waits are synchronous `std::sync::Condvar` parks (store.rs:277), NOT `.await`s → not cancellation points; a dropped task is dropped at an `.await` (the `recv()` boundary), never mid-park. Ordering makes a cancel benign: retain+submit complete before block B, so a cancel before the QUIC send leaves at worst a durable-(or-soon-durable)-not-sent row (§5 row 4), recovered by 3b replay — **never a sent-not-durable frame.** `write_reliable_frame` is the existing all-or-nothing framed `write_all` (lib.rs). **Guard-across-await is structurally impossible** (the `MutexGuard` is dropped in block A before block C's first `.await`); fold: enable clippy `await_holding_lock` for io-prod so a regression is caught mechanically (adjudicated NIT #5).

---

## 4. Redial / retry: the gate fires ONCE, never on resend

The gate is on the first-send path (blocks A/B, driven by a fresh `OutFrame` at `w.rx.recv()`). Resends take two paths, neither of which calls `assign_and_retain`, `submit_barrier`, or the barrier — **confirmed against the code:**

- **Reopened-stream replay inside `write_frame`** (mesh.rs:1679-1710): writes `lane.replay_batch()` (mesh.rs:679-692), re-stamping epoch on existing `retry` entries. No new retain, no submit. The just-assigned frame's barrier already ran in block B before this point.
- **Retransmit-timer replay** (`replay_lanes` → `replay_one_lane`, mesh.rs:1474-1497): pure resend of `replay_batch()`; no `assign_and_retain`, no sink passed (`replay_lanes` is NOT given `outbox`, §1.4).

A `Retained` frame is retained once (at assign) and released once (on the ack that retires it, `on_ack` mesh.rs:585-589). The outbox key is `(peer, class, incarnation, seq)` — **epoch-independent** (outbox.rs:80-86; `assign_and_retain` uses `self.outbox_key(class, seq)`, mesh.rs:655) — so an epoch bump on redial never orphans/duplicates a row. **Confirmed: the gate fires only on the first (assign) send; resends re-commit nothing and re-fsync nothing.** (Corroborated by existing test `r6d2c_t7`.)

---

## 5. Crash-window table (corrected; assumes a real `NodeOutbox` injected + R-6d3b replay)

For ONE producer-less `Retained` frame (`AwaitAdopt TransientBatch` on `Saga`, or band-exit `GhostFlow::Despawn` on `GhostReliable`). "RAM retry" = `ReliableLaneSender.retry`; "outbox on disk" = the fsynced row keyed `(peer,class,incarnation,seq)`; "sent" = `write_reliable_frame` put it on the wire.

| # | SIGKILL point (source) | RAM retry | Outbox on disk | Sent? | Outcome / recovery |
|---|---|---|---|---|---|
| 1 | before block A | absent | absent | no | Item still Crossing (never Held/emitted; a Transient source retains nothing across a crash). Re-emitted next tick, idempotent by `(transfer, BATCH_STEP)`. **NO LOSS.** |
| 2 | after `assign_and_retain` staged into RAM, before `submit_nonblocking` | present (dies) | absent | no | Row staged but not submitted → the crossbeam batch never left the store; nothing on disk. Restart re-emits from a fresh source. **NO LOSS.** |
| 3 | after `submit_nonblocking` (batch handed off), before its fsync completes | present (dies) | absent (fsync not done) OR present (if fsync raced in) | no | redb txn is all-or-nothing (`apply_batch`, store.rs:222-256): committed-and-fsynced (⇒ row 4) OR rolled back (⇒ no durable row). If rolled back: restart re-emits from source. **NO LOSS.** No torn half-row. |
| 4 | after `wait_durable_through(bar.seq)` returned (row DURABLE), before `write_reliable_frame` | present (dies) | **PRESENT** | no | **THE WINDOW THE GATE CLOSES.** Source dead + forgot the item, but the row is on disk. R-6d3b boot replay `scan_all` → fresh lane at bumped incarnation, seq re-based 0.. → re-sends (`classify_reliable` Reset-then-Accept) → dest adopts exactly once. **NO LOSS.** (Before 3a: loses the batch = D-6 #1. Before 3b lands replay: durable-but-orphaned = inert, swept by 3b `gc_below`.) |
| 5 | after `write_reliable_frame` (on the wire), before dest adopts+acks | present (dies) | present | yes | At-least-once: 3b replay re-sends the durable row; dest dedups by `(transfer, BATCH_STEP)` or the receiver seq-ledger. At most one duplicate, harmless. **NO LOSS, NO double-adopt.** |
| 6 | after dest adopts + ack retires the frame, before the release tombstone is fsynced | retired from RAM on ack | still PRESENT (release STAGED, not fsynced) | yes | Tombstone staged but not durable. 3b replay re-sends the already-adopted frame(s); dest dedups (no-op), re-acks, fresh `on_ack` releases the row. **Corrected bound (finding #4):** on a crash in an all-acked idle window with no subsequent `submit`, **up to the retry-cap window's worth of acked frames** (not "one") are re-delivered — all idempotent no-ops. Bounded by `retry_buffer_max_bytes` in-incarnation and swept by 3b `gc_below` on boot. **NO LOSS, NO double-adopt.** |
| 7 | after the release tombstone is fsynced | retired | ABSENT | yes | Terminal: nothing to replay. Clean. |

**Concurrency crash note (MF-1-corrected):** the sink `Mutex` is dropped BEFORE block B, so a crash while peer-writer A is parked in a durability wait holds NO shared lock → peer-writer B's disjoint `(peer,...)` rows follow this table independently. `wait_prior_durable` keeps in-flight ≤1 batch across the whole node (depth-2 channel absorbs the transient ≤2), so at most one batch's retains/releases are at risk in any single-crash window, every row resolving NO-LOSS / NO-double-adopt above.

---

## 6. Tests — Tier-B ≥ 90, new gate regions covered (`justfile:39-41`, region-only, no `--branch`)

New coverable regions: block A's `durable`-gated lock + `submit_barrier` capture; block B's two-wait barrier + the Ephemeral (`gate == None`) skip; `NodeOutbox::submit_barrier`/`durability` + `RedbStore::submit_nonblocking`/`wait_prior_durable`; `on_ack` under a live shared sink. Every region runs in a deterministic in-process test (no SIGKILL — that's R-6d4).

### 6.1 Unit tests (pure FSM + `NodeOutbox`, `#[cfg(test)]`)
- **`submit_barrier_returns_seq_and_prior_and_waits`** (`outbox.rs`): real `NodeOutbox::open` (temp path, `TempOutbox` guard, outbox.rs:256-263); `retain` two rows; `submit_barrier()` returns `Some{seq>0, prior}`; `durability().is_durable_through(seq)` becomes true after `wait_durable_through(seq)`; a second `submit_barrier()` after an empty span returns `None` (MF-2). Covers the new submit/barrier regions.
- **`ephemeral_gate_is_inert`** (extend R-6d2c `t2`/`t3`, mesh.rs:3220-3239): `MockOutboxSink` records nothing when `durable=false`; assert its new `submit_barrier`/`durability` are **not called** on the ephemeral path (call-count == 0). Covers `gate == None`.
- **`MockOutboxSink` gains `submit_barrier`/`durability`** (mesh.rs:3138-3154): `submit_barrier` returns `Some{seq: monotone-non-zero, prior: prev}` (finding #6 — a monotone `next_submit` field, so a "gate fired" assertion is not aliased to genesis `0`); `durability` returns a test-only always-durable handle via **`DurabilityHandle::already_durable()`** (`#[cfg(test)]` or `#[cfg(any(test, feature="store-test-hooks"))]`, store.rs; seeds `last_durable = u64::MAX` + `writer_alive = true` from owned Arcs so `wait_durable_through` fast-returns, store.rs:272; NO production API/field-visibility change — finding #6).

### 6.2 io-prod integration tests (real `NodeOutbox` + real QUIC loopback, `crates/io-prod/tests/`)
Extend `mesh_under_loss.rs` (has `spawn_mesh` + trust harness, :53); `node()` gains `outbox: Option<SharedOutbox>` threaded into `spawn_mesh(handle, trust, &cfg, outbox)`; existing callers pass `None`.

- **T-DBS-1 `retained_frame_is_durable_before_it_hits_the_wire`**: open a real `NodeOutbox` on a temp path; two-node mesh with the sink on the sender; send ONE `Durability::Retained` frame on `MsgClass::Saga`. Assert `outbox.scan_all()` has the row AND the receiver drains it. To prove *ordering* (durable BEFORE wire) without SIGKILL: use the `store-test-hooks` `pause_on_key_prefix` (store.rs:320-332) paused on the `OUTBOX_TAG` prefix — the sender's `write_frame` must BLOCK in block B and the receiver must NOT yet have the frame; unpause → it arrives. The direct-handle, inert-in-prod proof (no `main`-opened store — §7).
- **T-DBS-2 `ephemeral_frame_takes_the_fast_path_unchanged`**: live sink; send a `Durability::Ephemeral` Saga frame; assert `outbox.scan_all()` is EMPTY and the frame delivers. Ephemeral fast path byte-identical even with a sink wired.
- **T-DBS-3 `two_peers_do_not_serialize_on_fsync`** (the MF-1 non-serialization gate): sender with TWO peers, each its own writer, one shared `NodeOutbox` with the writer PAUSED (`pause_on_key_prefix`) so an fsync stalls. Send a `Retained` frame to peer A (its writer parks in block B). Then send a `Retained` frame to peer B: assert peer B's `assign_and_retain` **AND `submit_barrier`** complete (peer B's row appears staged/submitted for its own key) while A is parked — i.e. B acquired the shared lock and finished its non-blocking submit while A waited outside it. **This is the strong assertion MF-1 makes provable** (not the weakened stage-only the DESIGNER feared). Observe via a bounded `tokio::time::timeout` wrapping the TEST's observation (never `write_frame`).
- **T-DBS-4 `on_ack_releases_through_the_shared_sink`** (LOW-4): after T-DBS-1's delivery, drive the ack back; assert the row is released (`scan_all` empties) — the SAME `Arc` reached both `assign_and_retain` and `on_ack`.

### 6.3 Coverage bookkeeping
All new gate regions + `submit_barrier`/`durability`/`submit_nonblocking`/`wait_prior_durable` run in the deterministic tests above. io-prod floor is `--fail-under-regions 90`, no branch — new regions ratchet the % UP. If the paused-writer real-QUIC tests are timing-sensitive under the coverage debug build, **mirror the T-DBS-3 assertion into a pure `NodeOutbox` + two `ReliableLaneSender` unit test (no QUIC)** so the non-serialization region is covered deterministically and the QUIC test is a redundant e2e witness. **Never lower `tier_b_floor`; ratchet up.**

---

## 7. Scope decision — bins pass `None` in 3a (UPHELD)

**3a stays io-prod-focused and inert-safe in prod, exactly like 2c.** `spawn_mesh` gains `Option<SharedOutbox>` + the gate; SHARD/GATEWAY bins continue to pass `None`. `open_node_outbox` (bins/lib.rs:270-292) stays as-is, unused by bins. Justification (unchanged, still correct): (a) the gate is fully testable with a real `NodeOutbox` injected directly at the io-prod seam (T-DBS-1..4) — no bin needed; (b) bins on `None` keep `step_tick`/boot byte-identical (the 2c inert-safe posture); (c) opening a live outbox in `main` WITHOUT 3b's `scan_all` replay would leave recovered rows unread until 3b, changing prod boot half-way — the bin wiring + tick-loop split + replay land together in 3b where they are correct as a unit. **Consequence:** in prod TODAY (bins pass `None`) the gate is inert and a source crash still loses the batch, as before 3a; the crash table above is written for the WIRED case (the post-3b state, which the io-prod tests exercise via direct injection).

---

## 8. Files / functions touched (precise)

- `crates/io-prod/src/store.rs`
  - Add `RedbStore::submit_nonblocking(&mut self) -> Option<u64>` (non-blocking `try_send`, fail-loud on `Full`/`Disconnected`, MF-1) and `RedbStore::wait_prior_durable(&self, prior: u64)`.
  - Add `#[cfg(test)]` (or `#[cfg(any(test, feature="store-test-hooks"))]`) `DurabilityHandle::already_durable()` (seeds `last_durable=u64::MAX`, `writer_alive=true`; no production API/field-visibility change).
  - `commit()` (store.rs:556) UNCHANGED (the future batched-flush path).
- `crates/io-prod/src/outbox.rs`
  - `OutboxSink` (:138-161): add `fn submit_barrier(&mut self) -> Option<SubmitBarrier>;` and `fn durability(&self) -> DurabilityHandle;`; add `pub struct SubmitBarrier { seq, prior }`.
  - `impl OutboxSink for NodeOutbox` (:180-223): implement both (capture `prior = last_submitted()` before `submit_nonblocking`). `commit` unchanged.
- `crates/io-prod/src/mesh.rs`
  - Add `type SharedOutbox = Arc<Mutex<Box<dyn OutboxSink + Send>>>` (~:54).
  - `spawn_mesh` (:775): add `outbox: Option<SharedOutbox>` param; `outbox.clone()` into each `PeerWriter` (:852-865).
  - `PeerWriter` struct (:1142-1160): add `outbox: Option<SharedOutbox>`.
  - `peer_writer` (:1167-1315): lock the shared sink around the `on_ack` fan-out (:1187-1206), flip the `None` at :1195; pass `w.outbox.as_ref()` into `write_frame` (:1209-1224).
  - `write_frame` (:1580-1782): add `outbox: Option<&SharedOutbox>` param; rewrite the Reliable arm's assign block (:1611-1658) into blocks A (lock + assign + `submit_barrier`, flip the `None` at :1615) and B (the two-wait barrier), leaving block C (:1660-1728) verbatim; Unreliable arm (:1730-1780) untouched.
  - `MockOutboxSink` (:3138-3154): implement `submit_barrier` (monotone non-zero seq) / `durability` (always-durable stub); add the §6.1 unit tests.
  - Enable clippy `await_holding_lock` for io-prod (adjudicated NIT #5).
- `crates/io-prod/tests/mesh_under_loss.rs`: `node()` (:43-54) gains `outbox: Option<SharedOutbox>`; add T-DBS-1..4.
- `crates/io-prod/tests/mesh_load.rs`, `mesh_redelivery.rs`: update `spawn_mesh(...)` call sites (mesh_load.rs:110, mesh_redelivery.rs:52,459) to pass the new `None` arg (mechanical, behaviour-identical).
- NOT touched in 3a: `crates/bins/src/lib.rs` (`open_node_outbox` unused by bins yet), `shard.rs`/`gateway.rs` `main` (still pass `None`), the frozen `sim::io` seam, the saga/dest FSMs, the `ReliableLaneSender` FSM (`assign_and_retain`/`on_ack` already carry the sink param).

---

## 9. HR posture

- **HR1:** the outbox value is opaque already-framed `ReliableFrame` bytes (outbox.rs:117-122; `assign_and_retain` passes `encoded`, mesh.rs:655); the gate never decodes a `World`. Clean.
- **HR3:** ONE `ReliableLaneSender` FSM (unchanged), ONE `OutboxSink`/`NodeOutbox`, ONE `SharedOutbox` alias, the per-send `Durability` param already lowered at mesh.rs:1602 — never a shard-kind match. Clean.
- **HR5:** io-prod Tier-B (region-only ≥ 90); `if let Some((bar, h)) = gate` is one region, covered by durable + ephemeral cases; the `dyn` call boundary absorbs the generic-region gotcha; every new region exercised in an in-process test.
- **Frozen seam:** untouched — `submit_barrier`/`durability`/`SubmitBarrier` are on `OutboxSink` (io-prod-internal), not `sim::io`.
- **No magic numbers:** no new literals; reuses `StoreTuning::default()` (`writer_channel_depth: 2` is the existing depth-2 slack MF-1 depends on) and existing consts.

---

**VERDICT: SOUND_TO_IMPLEMENT** — the crash-correctness guarantee is proven sound; the single blocking HIGH (block-on-prior under the sink lock) is fixed by MF-1 (non-blocking `submit_nonblocking` under the lock + `wait_prior_durable`/`wait_durable_through` outside it, absorbed by the existing depth-2 writer channel), and MF-2 (the `Option<SubmitBarrier>` seq contract) is folded into the same API. All other findings are NIT-level or already handled by existing code and are dispositioned inline.


---

## ORIGINAL DESIGNER OUTPUT

# R-6d3a — Wire the Shared Outbox Sink + Durable-Before-Send Gate: Design of Record (IMPLEMENT-READY)

Scope: (1) inject ONE shared `NodeOutbox` sink per node into `spawn_mesh → PeerWriter → write_frame/on_ack`, flipping the two literal `None`s; (2) the durable-before-send GATE in `write_frame`'s Reliable arm — a `Retained` frame's outbox row is fsynced BEFORE `write_reliable_frame` puts it on the wire; Ephemeral frames keep the byte-identical fast path. Boot replay / `gc_below` (R-6d3b), the saga `AwaitAdopt` split + dest `TransientDiscard` (R-6d3c), and the composed SIGKILL e2e + proptest (R-6d4) are OUT of scope and named as the next sub-slices.

All line cites are against the worktree HEAD (`ebbe544` R-6d2c landed).

---

## 0. The decisive facts (grounded)

- The `OutboxSink` seam already exists and the FSM already write-throughs / delete-throughs: `assign_and_retain(from, class, bytes, durable, sink)` retains only when `durable && sink.is_some()` (`mesh.rs:651-656`); `on_ack(.., sink)` releases only `retained` rows (`mesh.rs:585-589`); `RetainedFrame.retained` records the subset (`mesh.rs:442`). Both call sites pass literal `None` today (`mesh.rs:1615` assign; `mesh.rs:1195` on_ack). **R-6d3a's whole job is to replace those two `None`s with a shared handle and add the fsync gate — the FSM does not change.**
- `NodeOutbox` holds `store: RedbStore` + `durability: DurabilityHandle` (`outbox.rs:167-170`). Its `commit()` TODAY does `store.commit()` THEN `durability.wait_durable_through(store.last_submitted())` — i.e. **it blocks on the fsync internally** (`outbox.rs:189-196`). If a `PeerWriter` holds a shared `&mut NodeOutbox` across that call, it serializes every other peer writer. **This is exactly the hazard; the split below removes it.**
- `DurabilityHandle` is `Clone` (`store.rs:123-124`), non-blocking `is_durable_through`/`durable_through`/`last_submitted`/`writer_alive` are pure Acquire loads (`store.rs:138-161`), and `wait_durable_through` parks on a condvar (`store.rs:166-176`). It does NOT touch the store — so a writer can wait on a cloned handle WITHOUT holding the `NodeOutbox` lock.
- The fsync runs on `RedbStore`'s own `vd-store-writer` thread (`store.rs:301-390`, spawned `store.rs:426-442`), independent of the tokio runtime. Parking a tokio worker on `wait_durable_through` cannot block that thread → no deadlock.
- `store.commit()` returns `()` (`store.rs:556`); the submitted seq is read via `store.last_submitted()` (`store.rs:492-494`) / `DurabilityHandle::last_submitted()` (`store.rs:152-154`). `commit()` submits AT MOST one batch and block-on-priors the previous, so after a `commit()` the value of `last_submitted()` names exactly this batch (`outbox.rs:190-192` states this contract).
- `write_frame` cancel-safety: NEVER wrap in `timeout`/`select!`; the only cancel point is `w.rx.recv()` in the outer `select!` (`mesh.rs:1576-1578`, `mesh.rs:1207`).
- Redial re-send path: `replay_batch()` (`mesh.rs:684-692`) re-stamps epoch and re-sends the WHOLE retained window on `write_frame`'s reopened-stream arm (`mesh.rs:1691-1710`) and on `replay_lanes` (`mesh.rs:1273-1286`). It does NOT call `assign_and_retain` → **no re-retain, no re-commit on resend** (proven in §4).
- io-prod is Tier-B, `--fail-under-regions {{tier_b_floor}}` = 90, NO `--branch` (`justfile:39-41`). New gate branches must be covered by io-prod's own in-process tests.

---

## 1. The API shape — inject the shared sink + split submit / wait

### 1.1 The API insufficiency and the minimal additive fix

The current `OutboxSink::commit(&mut self)` conflates submit + fsync-wait and returns nothing. To stage+submit under the lock but wait OUTSIDE it, the writer needs three things it cannot get today:

1. a way to **submit the staged batch and learn the target seq** without blocking;
2. a **cloned `DurabilityHandle`** to wait on outside the lock;
3. the wait itself (already exists on the handle: `wait_durable_through`).

**Minimal additive change to `OutboxSink` (`outbox.rs:138-161`)** — add ONE method, keep the existing `commit` for the batched flush-tick path:

```rust
pub trait OutboxSink: Send {
    fn retain(&mut self, key: &OutboxKey, framed: &[u8]);
    fn release(&mut self, key: &OutboxKey);
    fn commit(&mut self);                       // UNCHANGED: submit + block-until-durable (batched flush)
    fn scan_all(&self) -> Vec<(OutboxKey, Vec<u8>)>;
    fn gc_below(&mut self, incarnation: u64);
    /// NEW (R-6d3a): submit the staged batch WITHOUT waiting; return the batch seq to await
    /// on a cloned DurabilityHandle OUTSIDE the lock. Splits the fsync barrier so a peer writer
    /// never holds the shared NodeOutbox lock across an fsync (the non-serialization property).
    fn submit(&mut self) -> u64;
    /// NEW (R-6d3a): a clone of the durability watermark handle — parkable WITHOUT the sink lock.
    fn durability(&self) -> DurabilityHandle;
}
```

`NodeOutbox` impl (`outbox.rs:180-223`):

```rust
fn submit(&mut self) -> u64 {
    self.store.commit();            // submit staged batch + block-on-prior (depth-1); does NOT wait
    self.store.last_submitted()     // the seq of THIS batch (commit submits at most one batch)
}
fn durability(&self) -> DurabilityHandle { self.durability.clone() }
// `commit` stays: submit + wait_durable_through — the future batched flush-tick path (unchanged).
```

Rationale: `store.commit()` already blocks ONLY on the PRIOR batch being durable (block-on-prior, `store.rs:556-567`), never on the current one — that residual block is bounded (≤1 batch behind) and is the intended depth-1 back-pressure, not the fsync-of-this-frame. The current-batch fsync wait is deferred to the cloned handle outside the lock. `submit` is object-safe (returns `u64`), branchless.

`MockOutboxSink` in mesh tests (`mesh.rs:3142-3154`) gains the two methods; the pure FSM mock returns a dummy seq and a stub handle (see §6.1 — a test-only `DurabilityHandle::already_durable()` constructor).

### 1.2 The shared sink: ONE `NodeOutbox` per node, behind a `Mutex`, cloned `Arc` to each writer

`OutboxKey.peer` (`outbox.rs:82`) makes every writer's rows disjoint per peer, so ONE store is correct. Because each `peer_writer` is a separate tokio task, the shared handle must be `Send + Sync`. `NodeOutbox` is `Send` (`RedbStore` + `DurabilityHandle` are `Send`) but not `Sync`. Wrap it: **`type SharedOutbox = Arc<std::sync::Mutex<dyn OutboxSink + Send>>`** — but a trait-object `Mutex<dyn>` is awkward to construct; use a concrete alias in prod and keep the FSM `&mut dyn`:

```rust
// mesh.rs — one shared handle, cloned into every writer:
type SharedOutbox = std::sync::Arc<std::sync::Mutex<Box<dyn OutboxSink + Send>>>;
```

- `MeshConfig` (`mesh.rs:57-83`): **NO new field** — `MeshConfig` is `Clone + Debug` and a live redb-backed sink is neither. Keep config topology-only. **Instead thread the sink as a `spawn_mesh` PARAMETER**, so a `None` (dev/tests) keeps the exact current path.

```rust
pub fn spawn_mesh(
    handle: &tokio::runtime::Handle,
    trust: &ClusterTrust,
    cfg: &MeshConfig,
    outbox: Option<SharedOutbox>,   // NEW last param; None = today's behaviour, byte-identical
) -> Result<(MeshTransport, MeshControl), ProdIoError>
```

- In the per-peer spawn loop (`mesh.rs:846-867`), clone the `Arc` into each `PeerWriter`:

```rust
handle.spawn(peer_writer(PeerWriter {
    // ...existing fields...
    reliability: cfg.reliability,
    outbox: outbox.clone(),      // NEW: Option<SharedOutbox>, one Arc per writer, one store behind it
}));
```

- `PeerWriter` struct (`mesh.rs:1142-1160`) gains `outbox: Option<SharedOutbox>`.

### 1.3 Threading the handle into `write_frame` + `on_ack` — LOW-4: the SAME sink to both

`peer_writer` owns `w.outbox: Option<SharedOutbox>`. Both the send path (`write_frame`, `mesh.rs:1209-1224`) and the ack path (`on_ack`, `mesh.rs:1195`) must lock the SAME `Arc` — LOW-4: a durable row released via a different sink than it was retained on leaks. Because it is ONE `Arc<Mutex<..>>` cloned into the writer, both paths lock the identical store:

- **On_ack call site (`mesh.rs:1187-1206`):** lock the shared sink for the duration of the ack fan-out and pass `Some(&mut **guard)`:

```rust
changed = ack_rx.changed() => {
    if changed.is_ok() && let Some(ack) = ack_rx.borrow_and_update().clone() {
        // Lock ONCE for the whole ack (all entries share the same store); release-through is
        // staged only (no fsync here) — the tombstones ride the NEXT retain's submit, or a
        // flush-tick commit. A crash before that fsync harmlessly re-delivers an already-acked
        // frame (at-least-once; receiver dedups). LOW-4: SAME Arc the retain used.
        let mut guard = w.outbox.as_ref().map(|o| o.lock().unwrap_or_else(PoisonError::into_inner));
        for e in &ack.entries {
            if let Some(lane) = lanes.get_mut(&e.class) {
                let sink = guard.as_deref_mut().map(|b| &mut **b as &mut dyn OutboxSink);
                let retired = lane.on_ack(e.incarnation, e.epoch, e.ack_through, sink);
                if retired > 0 { w.stats.reliable_acked.fetch_add(retired as u64, Ordering::Relaxed); }
            }
        }
    }
}
```

- **write_frame:** takes `outbox: Option<&SharedOutbox>` as a new parameter (`mesh.rs:1580-1594` signature; passed from `mesh.rs:1209-1224`). It locks the sink ONLY around `assign_and_retain` (§2), then releases before the fsync wait and the QUIC send. `replay_lanes` (the redial path, `mesh.rs:1273`) is NOT given the sink — redial never re-retains (§4).

### 1.4 Ordering summary — what `write_frame`'s Reliable arm does, in order

```
1. durable = matches!(frame.durability, Durability::Retained)              // mesh.rs:1602 (unchanged)
2. { lock the shared sink (if durable && Some) }
     seq = lane.assign_and_retain(local, class, bytes, durable, sink)      // stages the retain row
     if durable { submit_seq = sink.submit(); durability = sink.durability() }  // submit under lock, NO wait
   { drop the lock }                                                        // ← lock released BEFORE fsync
3. if durable { durability.wait_durable_through(submit_seq) }              // ← park OUTSIDE the lock
4. ensure_connection / open stream / write_reliable_frame                  // the QUIC send, AFTER durable
```

For Ephemeral (`durable == false`): step 2 passes `None` for the sink into `assign_and_retain` (no lock taken, no retain, byte-identical to today), steps 2b/3 are skipped entirely, step 4 is unchanged. **The datagram (Unreliable) arm is 100% untouched.**

---

## 2. The durable-before-send gate — exact placement in `write_frame`'s Reliable arm

Current Reliable arm (`mesh.rs:1596-1728`): `assign_and_retain` with `None` (`:1615`) → `ensure_connection` (`:1662`) → open-stream-replay OR steady-state `write_reliable_frame` (`:1679-1728`). The gate inserts between assign and `ensure_connection`, gated on `durable`.

### 2.1 The replacement (assign + submit under lock; wait + send outside)

```rust
Reliability::Reliable => {
    let durable = matches!(frame.durability, vd_sim::io::Durability::Retained);  // mesh.rs:1602 unchanged

    // (A) assign + retain + submit — sink locked for THIS span only. The durable barrier
    //     seq + a cloned handle are captured; the lock is DROPPED before any fsync wait.
    let gate: Option<(u64, DurabilityHandle)> = {
        let mut guard = if durable {
            outbox.map(|o| o.lock().unwrap_or_else(PoisonError::into_inner))
        } else { None };
        let sink: Option<&mut dyn OutboxSink> =
            guard.as_deref_mut().map(|b| &mut **b as &mut dyn OutboxSink);

        let lane = lanes.entry(frame.class)
            .or_insert_with(|| ReliableLaneSender::new(dest, incarnation, retry_cap));
        let seq = match lane.assign_and_retain(local, frame.class, &frame.bytes, durable, sink) {
            Ok(seq) => { lane.last_msg_id = Some(frame.msg_id); seq }
            Err(reject) => { /* ...unchanged shed path, mesh.rs:1624-1656... */ return Err(WriteFail::Shed(reason)); }
        };
        // Submit the staged retain WITHOUT waiting; capture (barrier_seq, handle). Only when a real
        // durable row was written (durable AND a sink was present) — else the gate is inert.
        let mut g2 = guard;                          // move the guard so we can submit then drop
        let gate = if durable {
            g2.as_deref_mut().map(|b| { let s = b.submit(); (s, b.durability()) })
        } else { None };
        drop(g2);                                    // ← LOCK RELEASED HERE, before the fsync wait
        gate
    };

    // (B) the durable-before-send BARRIER — OUTSIDE the lock, so a parked worker never
    //     serializes another peer writer's retains. No-op for Ephemeral (gate == None).
    if let Some((barrier_seq, durability)) = gate {
        durability.wait_durable_through(barrier_seq);  // parks on the store's condvar; NOT the sink lock
    }

    // (C) the QUIC send — UNCHANGED from mesh.rs:1660-1728 (ensure_connection, open-stream replay
    //     OR steady-state write_reliable_frame). Reached ONLY after the row is durable (for Retained).
    ensure_connection(/* ...unchanged... */).await.map_err(|()| WriteFail::Down)?;
    /* ...the rest of the current Reliable arm, verbatim... */
}
```

Note `seq` is bound inside block (A) and used by block (C) — restructure so the `let seq = { ... }` binding survives (bind `seq` and `gate` together, e.g. return `(seq, gate)` from block A). The current code already binds `let seq = { ... };` (`mesh.rs:1611-1658`), so this is a mechanical widening of that block to also produce `gate`.

### 2.2 Why the Ephemeral fast path is byte-identical

When `durable == false`: `guard` is `None` (no `lock()` call, no contention), `sink` is `None` → `assign_and_retain` takes its existing `retained = durable && sink.is_some()` false arm (`mesh.rs:651`) — nothing mirrored, exactly as today with the literal `None`. `gate` is `None` → block (B) is a skipped `if let`. Block (C) is the unchanged send. The generated code for an Ephemeral Saga/Snapshot/Input frame is identical to HEAD except for one `matches!` already present at `:1602` and one always-false `if let None` branch. **No fsync, no commit, no lock.** The datagram arm (`mesh.rs:1730-1780`) is not in this path at all.

### 2.3 The gate proves the invariant

For a `Retained` frame (the `AwaitAdopt TransientBatch`, the band-exit `GhostFlow::Despawn`), `write_reliable_frame` (block C) is reached ONLY after `wait_durable_through(barrier_seq)` returned, i.e. the outbox row is fsynced. A crash in the "durable-but-not-yet-QUIC-sent" window (§5 row 4) leaves the row on disk → R-6d3b boot replay re-sends it. **A batch frame that can ever be observed on the wire survives the source crash.** (This design lands the gate; R-6d3b lands the replay that consumes it. Until 3b, the row is durable-but-orphaned on crash — inert, safe, GC'd by 3b's `gc_below`; §5 row annotations.)

---

## 3. No-deadlock, no-datagram-head-of-line, cancel-safety

### 3.1 No deadlock
The fsync executes on the `vd-store-writer` thread (`store.rs:301-390`), spawned outside the tokio runtime (`store.rs:426-442`). `wait_durable_through` parks a tokio worker on the store's condvar (`store.rs:166-176` → `park_until_durable`, `store.rs:266-290`); the writer thread bumps `last_durable` + notifies (`store.rs` writer loop). The parked tokio worker holds NO lock the writer needs (the sink `Mutex` was dropped in block A; the writer thread never touches that `Mutex` — it only drains the crossbeam channel). Therefore: parking a tokio worker on the fsync can never block the fsync thread. The liveness escape (`store.rs:282-288`) panics loud if the writer died — a refusal, never a hang. **No deadlock possible.**

### 3.2 No datagram / cross-peer head-of-line (HIGH-1)
Two independent claims:
- **Cross-peer:** each `peer_writer` clones its OWN `Arc<Mutex<NodeOutbox>>` but the `Mutex` is held ONLY across block A (assign + submit — pure RAM stage + one crossbeam `send`, no fsync). The fsync WAIT (block B) is outside the lock. So while writer P1 parks on P1's `barrier_seq`, writer P2 can immediately lock, stage, submit, and unlock. **Peer writers never serialize on the fsync.** (This is the exact hazard the prompt names; the split is the cure.)
- **Same-peer datagram:** `peer_writer` is one `select!` (`mesh.rs:1185`); a datagram to the same peer is a separate `OutFrame` processed in a later loop iteration. A `Retained` reliable send that parks in block B does delay the NEXT `w.rx.recv()` for THAT peer (single-consumer FIFO) — but the CODE path is byte-identical (`assign_and_retain` is reliable-only, `mesh.rs:1595`), and the fsync is paid ONLY by producer-less `Retained` sends: the `AwaitAdopt` batch and band-exit `Despawn` — low-rate, NOT the 20 Hz Snapshot/Input classes (those are Ephemeral → gate skipped, `Durability::Ephemeral` default per R-6d2b). The corrected claim (per the vetted design HIGH-1): **datagram CODE unchanged; the added fsync is paid only by low-rate producer-less reliable sends, and bounded off the datagram latency path because Snapshot/Input never carry `Retained`.** A same-peer datagram queued behind a `Retained` send waits at most one fsync (~sub-ms at 50 Hz; a real wait IS disk-stall back-pressure, `store.rs:164-165`).

### 3.3 Cancel-safety
`write_frame` is still never wrapped in `timeout`/`select!` (`mesh.rs:1576-1578`); the only cancel point stays `w.rx.recv()` (`mesh.rs:1207`). The added `wait_durable_through` (block B) is a synchronous, blocking, NON-`.await` call — it parks the OS thread on a `std::sync::Condvar`, it is not a future, so it is not a cancellation point. If the outer task is dropped, it is dropped at an `.await` (the `w.rx.recv()` boundary), never mid-`wait_durable_through`. Ordering makes a cancel between assign and send benign: **retain+submit is complete before the wait**, so the row is already staged+submitted; a cancel before the QUIC send leaves a durable (or soon-durable) row with no wire send — identical to the crash-window §5 row 4, recovered by boot replay. There is no half-durable/half-sent state: the fsync barrier strictly precedes the single `write_reliable_frame`, and `write_reliable_frame` itself is the existing all-or-nothing framed `write_all` (`lib.rs:433-434`). **A cancelled durable-wait leaves at worst a durable-not-sent row, which replay re-sends — never a sent-not-durable frame.**

---

## 4. Redial / retry: the gate fires ONCE, never on resend

The gate lives on the FIRST-send path (block A/B, driven by `w.rx.recv()` delivering a fresh `OutFrame`). Resends take two OTHER paths, neither of which calls `assign_and_retain`, `submit`, or the barrier:

- **Reopened-stream replay inside `write_frame`** (`mesh.rs:1679-1710`): when `lane.stream.is_none()`, it writes `lane.replay_batch()` (`mesh.rs:684-692`) — which reads existing `retry` entries and re-stamps epoch. The just-assigned frame is already in `retry` (retained in block A) and rides as the highest replay entry (`mesh.rs:1692-1696` debug_assert). No new retain, no new submit. The barrier for the just-assigned frame already ran in block B before this point — so even in the reopen case the frame is durable before its first wire appearance, and its RE-appearances on later redials cost zero fsync.
- **Retransmit-timer replay** (`replay_lanes`, `mesh.rs:1271-1297` → `mesh.rs` replay loop calling `write_reliable_frame` at `:1484`): pure resend of `replay_batch()`, no `assign_and_retain`, no sink passed. `replay_lanes` is NOT given the `outbox` handle (§1.3).

A `Retained` frame is retained exactly once (at assign) and released exactly once (on the ack that retires it, `on_ack`, `mesh.rs:585-589`). Redial re-stamps epoch on the in-RAM frame but the outbox key is `(peer, class, incarnation, seq)` — **epoch-independent** (`RetainedFrame.retained` doc, `mesh.rs:438-442`; `outbox_key`, `mesh.rs:528-535`) — so an epoch bump never orphans or duplicates a row. **Confirmed: the gate fires only on the first (assign) send; resends re-commit nothing and re-fsync nothing.**

---

## 5. Crash-window table (durable-before-send gate, assuming R-6d3b replay exists)

See the structured `crash_window_table` field.

---

## 6. Tests — how they hold Tier-B ≥ 90 with the new gate branch covered

The new gate has these coverable regions in `write_frame`: the `durable`-gated lock acquisition, the `submit`+`durability` capture, the `gate.is_some()` barrier, and the Ephemeral (`gate == None`) skip. Plus `NodeOutbox::submit` / `durability`, and `on_ack` under a live shared sink. io-prod is Tier-B, NO `--branch` (`justfile:39-41`) — region coverage only, but every region must run in an in-process test (deterministic, no SIGKILL).

### 6.1 Unit tests (pure FSM + `NodeOutbox`, in `mesh.rs` / `outbox.rs` `#[cfg(test)]`)

- **`submit_returns_batch_seq_and_durability_waits`** (`outbox.rs`): a real `NodeOutbox::open` (temp path, existing `TempOutbox` guard, `outbox.rs:256-263`); `retain` two rows; `submit()` returns seq `s > 0`; `durability().is_durable_through(s)` becomes true after `wait_durable_through(s)`; `scan_all` shows both rows. Covers the new `submit`/`durability` regions and proves submit-then-wait equals the old `commit`.
- **`ephemeral_gate_is_inert`** / extend the existing R-6d2c `t2`/`t3` (`mesh.rs:3220-3239`): `MockOutboxSink` records nothing when `durable=false`; add that the mock's new `submit`/`durability` are NOT called on the ephemeral path (assert a call-count). Covers the `gate == None` skip.
- **`MockOutboxSink` gains `submit`/`durability`** (`mesh.rs:3142-3154`): `submit` returns a monotone counter, `durability` returns a test-only always-durable handle. **Add `DurabilityHandle::already_durable()` test-constructor** (`store.rs`, `#[cfg(test)]` or `pub(crate)`) that builds a handle whose `last_durable == u64::MAX` so `wait_durable_through` fast-returns (`store.rs:167`, `store.rs:272-274`) — keeps the pure FSM tests tokio-free and non-blocking. (Minimal additive; covered by its own use.)

### 6.2 io-prod integration tests (real `NodeOutbox` + real QUIC loopback, in `crates/io-prod/tests/`)

Extend `mesh_under_loss.rs` (has `spawn_mesh` + trust harness, `mesh_under_loss.rs:53`; `node()` builds `MeshConfig::new`, `:52`). The `node()` helper gains an `outbox: Option<SharedOutbox>` param and threads it into `spawn_mesh(handle, trust, &cfg, outbox)`; existing call sites pass `None` (unchanged behaviour).

- **T-DBS-1 `retained_frame_is_durable_before_it_hits_the_wire`** (the durable-before-send proof): open a real `NodeOutbox` on a temp path; spawn a two-node mesh with the sink on the sender; send ONE `Durability::Retained` frame on `MsgClass::Saga`. Assert: after the send, `outbox.scan_all()` contains the row (durable) AND the receiver drains the frame. To prove ordering (durable BEFORE wire) without a SIGKILL (that's R-6d4), inject a `StoreTuning` with a slow/paused writer via the existing `store-test-hooks` `pause_on_key_prefix` (`store.rs:315-320`, `store.rs:423`): with the writer paused on the outbox tag, the sender's `write_frame` must BLOCK in `wait_durable_through` and the receiver must NOT yet have the frame; unpause → the frame arrives. This is the direct-handle, inert-in-prod proof the prompt asks for (no `main`-opened store needed — see §7 scope decision).
- **T-DBS-2 `ephemeral_frame_takes_the_fast_path_unchanged`**: same two-node mesh with a live sink; send a `Durability::Ephemeral` Saga frame; assert `outbox.scan_all()` is EMPTY (nothing mirrored) and the frame is delivered. Proves the Ephemeral fast path stays byte-identical even with a sink wired.
- **T-DBS-3 `two_peers_do_not_serialize_on_fsync`** (the non-serialization property): a sender node with TWO peers, each its own writer, one shared `NodeOutbox` with a WRITER PAUSED (via `store-test-hooks`, `store.rs:315`) so an fsync stalls. Send a `Retained` frame to peer A (its writer parks in block B). Then send a `Retained` frame to peer B: assert peer B's `assign_and_retain` + `submit` COMPLETE (the row appears staged for peer B's key) even while peer A is parked — i.e. peer B acquired the shared lock while A waited outside it. This is the concurrency non-serialization gate. Assert via a bounded `tokio::time::timeout` on peer-B's stage completing (the timeout wraps the TEST's observation, never `write_frame`).
- **T-DBS-4 `on_ack_releases_through_the_shared_sink`** (LOW-4): after T-DBS-1's delivery, drive the ack back; assert the outbox row is released (`scan_all` empties) — proving the SAME `Arc` reached both `assign_and_retain` and `on_ack`.

### 6.3 Coverage bookkeeping
All four new gate regions (lock, submit, barrier, ephemeral-skip) + `NodeOutbox::submit`/`durability` run in the deterministic in-process tests above. The io-prod floor is `--fail-under-regions 90`, no branch (`justfile:39-41`) — the new regions ratchet the measured % UP, never below the 90 floor. If the paused-writer real-QUIC tests are timing-sensitive under the coverage debug build, mirror the non-serialization ASSERTION into a pure `NodeOutbox`+two-`ReliableLaneSender` unit test (no QUIC) so the region is covered deterministically and the QUIC test is a redundant e2e witness. **Never lower `tier_b_floor`; ratchet up if the new tests raise the baseline.**

---

## 7. Scope decision — does `shard.rs`/`gateway.rs` `main` open a real `NodeOutbox` in 3a?

**Decision: NO. 3a stays io-prod-focused and inert-safe in prod, exactly like 2c.** `spawn_mesh` gains the `Option<SharedOutbox>` param and the gate; the SHARD/GATEWAY bins continue to pass `None` for now. `open_node_outbox` already exists (`bins/lib.rs:270-292`) and is tested, but WIRING it into the bin `main` — splitting the tick loop into `run_schedule`/`flush_outbox`, opening the store, passing the real `SharedOutbox` — is coupled to boot replay (R-6d3b: the store must be `scan_all`-replayed BEFORE the first send, else a recovered row would race a fresh send). Opening a live outbox in `main` WITHOUT the replay would make the gate durable-before-send but leave recovered rows unread until 3b — half a feature that changes prod boot behaviour.

**Justification:** (a) 3a is fully testable with a real `NodeOutbox` injected directly at the io-prod seam (T-DBS-1..4 inject the real handle into `spawn_mesh` in a test) — the gate's correctness does not need a bin to open the store; (b) keeping the bins on `None` means `step_tick` and boot stay byte-identical (the R-6d2c inert-safe posture, MEMORY: "3a is inert-safe in prod like 2c was"); (c) the bin wiring + tick-loop split + replay land together in R-6d3b where they are correct as a unit. **3a proves the gate mechanism against a real store injected at the seam; 3b makes a bin open it and replay it.**

Consequence for the crash table: in prod TODAY (bins pass `None`), the gate is inert and a source crash still loses the batch (as before 3a). The table below is written for the WIRED case (a real `NodeOutbox` injected — the state after 3b opens it in `main`), because that is the state the gate is designed for and the io-prod tests exercise.

---

## 8. Files / functions touched (precise)

- `crates/io-prod/src/outbox.rs`
  - `OutboxSink` trait (`:138-161`): add `fn submit(&mut self) -> u64;` and `fn durability(&self) -> DurabilityHandle;`.
  - `impl OutboxSink for NodeOutbox` (`:180-223`): implement `submit` (= `store.commit(); store.last_submitted()`) and `durability` (= `self.durability.clone()`). `commit` unchanged.
- `crates/io-prod/src/store.rs`
  - Add `#[cfg(test)] DurabilityHandle::already_durable()` test-constructor (for the pure FSM mock handle). No production API change; `DurabilityHandle` is already `Clone` (`:123-124`).
- `crates/io-prod/src/mesh.rs`
  - Add `type SharedOutbox = Arc<Mutex<Box<dyn OutboxSink + Send>>>` (near the `ConnRegistry` alias, `:54`).
  - `spawn_mesh` (`:775-882`): add `outbox: Option<SharedOutbox>` param; `outbox.clone()` into each `PeerWriter` in the peer loop (`:852-865`).
  - `PeerWriter` struct (`:1142-1160`): add `outbox: Option<SharedOutbox>`.
  - `peer_writer` (`:1167-1315`): lock the shared sink around the `on_ack` fan-out (`:1187-1206`), passing `Some(&mut dyn)` (flip the `None` at `:1195`); pass `w.outbox.as_ref()` into `write_frame` (`:1209-1224`).
  - `write_frame` (`:1580-1782`): add `outbox: Option<&SharedOutbox>` param; rewrite the Reliable arm's assign block (`:1611-1658`) into blocks A (lock+assign+submit, flip the `None` at `:1615`) and B (barrier), leaving block C (`:1660-1728`) verbatim; the Unreliable arm (`:1730-1780`) untouched.
  - `MockOutboxSink` (`:3142-3154`): implement `submit`/`durability`; add the four new io-prod unit tests referenced in §6.1.
- `crates/io-prod/tests/mesh_under_loss.rs`
  - `node()` helper (`:43-54`): add `outbox: Option<SharedOutbox>` param, thread into `spawn_mesh`; existing callers pass `None`. Add T-DBS-1..4.
- `crates/io-prod/tests/mesh_load.rs`, `mesh_redelivery.rs`: update their `spawn_mesh(...)` call sites (`mesh_load.rs:110`, `mesh_redelivery.rs:52,459`) to pass the new `None` arg (mechanical, behaviour-identical).
- NOT touched in 3a: `crates/bins/src/lib.rs` (`open_node_outbox` stays as-is, unused by bins yet), `shard.rs`/`gateway.rs` `main` (still pass `None` to `spawn_mesh`), the frozen `sim::io` seam, the saga/dest FSMs.

---

## 9. HR posture

- **HR1:** the outbox value is opaque already-framed `ReliableFrame` bytes (`outbox.rs:117-122`, `mesh.rs:655` passes `encoded`); the gate never decodes a `World`. Clean.
- **HR3:** ONE `ReliableLaneSender` FSM (unchanged), ONE `OutboxSink`/`NodeOutbox`, ONE `SharedOutbox` alias, the per-send `Durability` PARAMETER already lowered at `mesh.rs:1602` — never a shard-kind match. Clean.
- **HR5:** io-prod is Tier-B (region-only ≥ 90); the gate branches are branchless-friendly (`if let Some((seq, h)) = gate` is one region, covered by durable + ephemeral cases); the FSM stays monomorphic-shim (the `dyn` call boundary absorbs the generic-region gotcha). Every new region is exercised in an in-process test.
- **Frozen seam:** untouched — `submit`/`durability` are on `OutboxSink` (an io-prod-internal trait), not `sim::io`.
- **No magic numbers:** no new literals; reuses `StoreTuning::default()` and existing consts.


### DESIGNER crash-window table

Crash-window table for the R-6d3a durable-before-send gate on the SOURCE shard, for ONE producer-less Retained frame (an AwaitAdopt TransientBatch on MsgClass::Saga, or a band-exit GhostFlow::Despawn on MsgClass::GhostReliable). Assumes a real NodeOutbox is injected (the post-3b wired state) and R-6d3b boot replay (scan_all -> re-send at fresh incarnation -> gc_below) exists. "RAM retry" = ReliableLaneSender.retry; "outbox on disk" = the fsynced NodeOutbox row keyed (peer,class,incarnation,seq); "QUIC sent" = write_reliable_frame put the frame on the wire.

| # | SIGKILL point (source) | RAM retry | Outbox on disk | QUIC sent? | Outcome / recovery |
|---|---|---|---|---|---|
| 1 | before block A (assign) runs | absent | absent | no | The item is still Crossing (never flipped Held/emitted; a Transient source retains NOTHING across a crash). Re-emitted next tick after restart, idempotent by (transfer, BATCH_STEP). NO LOSS. |
| 2 | after assign_and_retain retained into RAM, before sink.submit() | present (dies with process) | absent | no | Row was staged but not submitted -> the crossbeam batch never left the store; nothing on disk. Restart re-emits from a fresh source (Transient source forgot the RAM row). NO LOSS. |
| 3 | after sink.submit() (batch handed to writer), before the writer's fsync completes | present (dies) | absent (fsync not done) | no | submit() returned the seq but wait_durable_through has NOT yet returned; the writer thread was mid-fsync when killed -> redb txn either committed-and-fsynced (=> treat as row 4) or rolled back (all-or-nothing, apply_batch store.rs:222-256) => NO durable row. If rolled back: restart re-emits from source, NO LOSS. redb's single-txn fsync is the atomic boundary; there is no torn half-row. |
| 4 | after wait_durable_through returned (row DURABLE), before write_reliable_frame (block C) | present (dies) | PRESENT | no | THE WINDOW THE GATE CLOSES. The source is dead and forgot the item, but the row is on disk. R-6d3b boot replay scan_all -> re-mints a fresh lane at the bumped incarnation, seq re-based 0.. -> re-sends (classify_reliable Reset-then-Accept) -> dest adopts exactly once. NO LOSS. (Before 3a: this window loses the batch = the D-6 #1 bug. Before 3b lands replay: the row is durable-but-orphaned = inert+safe, GC'd once 3b runs.) |
| 5 | after write_reliable_frame (QUIC on the wire), before dest adopts+acks | present (dies) | present | yes | At-least-once: 3b replay re-sends the durable row; the dest dedups by (transfer, BATCH_STEP) OR the receiver seq-ledger (classify_reliable). At most one duplicate delivery, harmless. NO LOSS, NO double-adopt. |
| 6 | after dest adopts + the ack retires the frame, before on_ack's release is fsynced | retired from RAM on ack | still PRESENT (release staged, not yet fsynced) | yes | The release tombstone is staged but not durable. On restart, 3b replay re-sends the ALREADY-ADOPTED frame; the dest dedups (idempotent no-op), then re-acks, and the fresh on_ack releases the row. At most one redundant re-delivery. NO LOSS, NO double-adopt (journal/seq dedup). The release fsync being lazy is deliberate: a lost tombstone only ever causes a harmless re-send, never a loss. |
| 7 | after on_ack release fsynced | retired | ABSENT (released+fsynced) | yes | Terminal: nothing to replay. Clean. |

Concurrency crash note: because the sink Mutex is dropped BEFORE block B (the fsync wait), a crash while peer-writer A is parked in wait_durable_through does NOT hold the shared lock -> peer-writer B's already-staged/submitted rows for its own disjoint (peer,...) keys follow the same table independently. The store's depth-1 block-on-prior (store.rs:556-567) bounds in-flight to <=1 batch, so at most one batch's worth of retains/releases is at risk in any single-crash window, and every such row resolves via a row above (all NO LOSS / NO double-adopt).


---

## REVIEW VERDICTS (summary)


### concurrency-deadlock (concurrency / deadlock / latency / cancel-safety) — REVISE
The design is deadlock-free and cancel-safe, and correctly identifies that the fsync WAIT must move outside the sink Mutex. But its central non-serialization claim (§3.2) is WRONG for the real store: `NodeOutbox::submit()` = `store.commit()`, and `RedbStore::commit()` (store.rs:563-567) contains a BLOCK-ON-PRIOR that waits on the PRIOR batch's fsync while the design holds the sink Mutex across `submit()`. Because the ONE shared `RedbStore` has a single `last_submitted`/`last_durable` pair and a depth-1 pipeline, a second peer's `submit()` blocks on the first peer's un-fsynced batch UNDER THE LOCK — the exact cross-peer serialization the split was meant to remove. The design's own OPEN QUESTIONS flags this but leaves it unresolved and does not correct the §3.2 proof or the block-A structure. This is a real HIGH defect for the concurrency lens: the API must be reshaped so nothing that can block-on-prior runs under the sink lock (stage under the lock; submit+both waits outside it). Deadlock-freedom (b), cancel-safety (d), and the redial fire-once property are all sound. HIGH-1 datagram head-of-line is bounded and acceptable as argued, with one caveat inherited from the block-on-prior defect.

- **HIGH** block A holds the sink Mutex across submit(), which block-on-prior-waits on ANOTHER peer's fsync — the non-serialization property (§3.2) is false against the real store — The design asserts (§3.2, §3, and the crash-note) that because the sink Mutex is dropped before the fsync WAIT, peer writers never serialize on fsync: 'while writer P1 parks on P1's barrier_seq, writer P2 can immediately lock, stage, submit, and unlock.' This is incorrect. `NodeOutbox::submit()` is defined as `store.commit(); store.last_submitted()` (§1.1). `RedbStore::commit()` itself contains a BLOCK-ON-PRIOR barrier that BLOCKS the caller until the PREVIOUS batch is durable. Block A holds the sink Mutex across `submit()`, so that block-on-prior wait executes UNDER the lock. With ONE shared NodeOutbox/RedbStore per node (§1.2), there is a SINGLE `last_submitted`/`last_durable` pair and a depth-1 pipeline (only one batch in flight). Sequence: peer A locks, stages, `submit()` returns seq N (prior N-1 durable, no wait), drops lock, parks on N outside the lock. Peer B locks, stages, `submit()` → `store.commit()` reads `last_submitted==N`, sees `last_durable < N` (A's batch not yet fsynced) and calls `wait_durable_through(N)` — blocking B on A's fsync WHILE HOLDING THE SINK LOCK. Every subsequent peer that wants to stage now blocks on B's held lock. Under any nonzero fsync latency with ≥2 concurrent durable senders this convoys all peer writers through the disk, which is precisely the hazard the prompt names and the split was meant to cure.

- **LOW** wait_durable_through blocks a tokio executor thread (synchronous condvar park) — accepted project posture but the convoy amplifies it — block B's `durability.wait_durable_through(barrier_seq)` is a synchronous std condvar park (store.rs:277 `cv.wait_timeout`), not an `.await`. This blocks the whole tokio WORKER thread, not just the task. On the happy path (~50Hz, prior tiny batch already durable) it returns immediately (store.rs:272 fast-path) so it is benign, and Slice-D already established this posture for the orchestrator flush gate. But combined with the HIGH-1 block-on-prior-under-lock defect, a disk stall can park MULTIPLE tokio workers simultaneously (each peer writer is its own task, potentially on distinct workers), degrading the runtime's ability to make progress on unrelated work (accept loop, ack readers) during the stall. This is intended back-pressure ('durability over liveness') for the durable send itself but it also stalls co-located non-durable work on the same worker.

- **NIT** confirm cancel-safety wording: the only .await between assign and send is inside ensure_connection/open_uni (block C), not the wait — the design's claim is correct but rests on the wait being non-.await — The design's cancel-safety argument (§3.3) is sound: write_frame is never timeout/select-wrapped (mesh.rs:1576), the wait is a synchronous park (not a cancel point), and the only task-drop boundary is `w.rx.recv()` in the outer select (mesh.rs:1207). A cancel therefore cannot land mid-wait; retain+submit complete before the park, so a cancel before the QUIC send leaves at worst a durable-not-sent row (crash-table row 4), recovered by 3b replay — never a sent-not-durable frame. This is correct. The only thing to nail down at implementation is that the guard/lock in block A is fully dropped before block C's first `.await` (ensure_connection, mesh.rs:1662), else a held std Mutex across an await is a footgun (blocking a worker that holds it can wedge other peers). The design does drop the lock in block A before block B (§2.1 `drop(g2)`), so this holds — just verify no guard survives into block C.

- **NIT** HIGH-1 same-peer datagram head-of-line is bounded and correctly argued, contingent on fixing the block-on-prior defect — The design's HIGH-1 analysis (§3.2 same-peer) is correct: a datagram to the same peer is a later `w.rx.recv()` iteration in the single-consumer select (mesh.rs:1185/1207), so a Retained reliable send that parks in block B delays the NEXT datagram to THAT peer by at most the fsync latency. Snapshot/Input never carry Retained (Ephemeral default), so the 20Hz latency path never pays the fsync. This is bounded and acceptable. HOWEVER: with the HIGH-1 block-on-prior-under-lock defect present, a same-peer datagram could be delayed by up to the fsync of ANOTHER peer's batch (via the shared lock convoy), not just this peer's own send — widening the bound beyond the design's 'at most one fsync' claim. Fixing the block-on-prior defect restores the tight bound.


### CRASH-WINDOW + DURABILITY CORRECTNESS — REVISE
On the crash-window/durability lens the design's CORE guarantee is SOUND: for a Retained frame the ordering retain→submit→wait_durable_through(seq)→write_reliable_frame genuinely puts the row on disk (fsynced) before its first QUIC appearance (b), so no frame is ever both observed-as-sent AND lost once R-6d3b replay exists. I re-derived the 7-row crash table against the exact ordering and every row is NO-LOSS / NO-double-adopt: kill-before-assign and kill-before-submit lose nothing (Transient source forgot it, re-emits); kill-mid-fsync is redb-atomic (all-or-nothing txn, store.rs:222-256) so it resolves to durable-or-absent, never torn; kill-after-durable-before-send is the exact window the gate closes; kill-after-send is at-least-once with receiver dedup. The redial/replay_batch path provably does NOT re-commit or re-fsync (replay_lanes/replay_one_lane at mesh.rs:1471-1497 never call assign_and_retain/submit and are not passed the sink; on_write_error/replay_batch are &self / no-sink — confirmed by existing test r6d2c_t7). The OutboxKey (peer,class,incarnation,seq) is incarnation-stamped from the lane (outbox.rs:82-86, mesh.rs:528-535 uses self.incarnation), so a bumped-incarnation replay writes DISJOINT rows — no collision/orphan with the dead incarnation's window. The acked-then-crash vs crash-then-acked delete-through cases (e) are durability-safe: a lost release tombstone only ever causes a harmless idempotent re-delivery, never a loss or double-adopt. HOWEVER two real defects invalidate load-bearing CLAIMS (not the core guarantee): (1) the §3.2 non-serialization property is FALSE as stated — submit()'s block-on-prior runs UNDER the sink Mutex and serializes peer writers on the prior batch's fsync; (2) submit() is under-specified for the shared-store contract and one interleaving (a release-only ack flush racing another peer's retain) can attribute the wrong seq to the durable barrier. Both are fixable without changing the ordering; the gate's crash-correctness is intact.

- **HIGH** §3.2 non-serialization property is FALSE: submit()'s block-on-prior runs under the sink Mutex and DOES serialize peer writers on the prior fsync — The design's §3.2 (and the crash-note in the table) asserts as PROVEN: 'while writer P1 parks on P1's barrier_seq, writer P2 can immediately lock, stage, submit, and unlock — peer writers never serialize on the fsync.' This is the exact hazard the prompt names and the split is sold as the cure. Re-deriving against the real code, the claim does not hold: submit() = store.commit(); store.last_submitted(), and store.commit() begins with BLOCK-ON-PRIOR (store.rs:563-567) which calls wait_durable_through(last_submitted) — a BLOCKING park — whenever last_durable < last_submitted. Under depth-1, after writer A submits batch N (last_submitted=N, not yet durable), writer B's submit() enters commit(), reads last_submitted=N, sees last_durable(N-1) < N, and PARKS in wait_durable_through(N) — all while B holds the shared sink Mutex it acquired in block A. So B serializes on A's fsync INSIDE the lock, not outside it. The split (wait outside the lock in block B) removes the CURRENT-batch wait but not the PRIOR-batch block-on-prior, which the design's own OPEN QUESTION #3/#5 half-acknowledge but §3.2 states as a settled proof.

- **MEDIUM** submit() is under-specified for the SHARED store: last_submitted() after commit() can name a batch that is NOT the one carrying this frame's row when a prior staged-but-unsubmitted release is present, or when commit() early-returns — submit() is defined as `store.commit(); store.last_submitted()` and the design asserts 'commit submits at most one batch, so after commit() last_submitted names exactly this batch' (design §0, §1.1). Against the SHARED-store reality this is only true if commit() actually submitted a new batch on THIS call. commit() has an early return when staged.is_empty() (store.rs:568-570) that returns WITHOUT bumping last_submitted. On the durable-send path a preceding retain() always stages a non-empty put, so the empty path is not reachable FROM a retain (verified: sink.retain stages a put keyed by bytes, always non-empty) — that direction is safe. The residual under-specification is the on_ack release path: the design §1.3 stages release() WITHOUT calling submit(), leaving pending deletes in the shared store.staged. When a later block-A retain calls submit(), commit() flushes a batch containing THIS retain AND those unrelated pending deletes — the returned seq DOES cover this frame's row, so it is still correct — but the design nowhere states the invariant 'every submit() that follows a retain() under the same lock returns a seq >= the batch carrying that retain', which is the property block B relies on. Without that invariant written and tested, a future change to submit() (e.g. the OPEN-QUESTION-#5 'only stage without submitting under contention' variant floated in the design) would silently return a stale last_submitted and wait_durable_through(stale_seq) would pass BEFORE this frame's row is durable — sending a not-yet-durable frame.

- **LOW** Crash-table row 6/7 relies on release tombstones eventually fsyncing, but the design never gives an idle lane's released rows a durability path (OPEN QUESTION #4) — a correctness-inert but unbounded on-disk residue until R-6d3b — The delete-through crash cases (e) are durability-CORRECT (a lost tombstone only causes an idempotent re-delivery, never a loss/double-adopt — I confirmed this against classify_reliable dedup at mesh.rs:393-423 and the journal (transfer,BATCH_STEP) dedup). But the table's rows 6-7 assume the release becomes durable 'on the NEXT retain's submit'. On a hot-then-idle lane (a producer-less Retained one-shot that is acked, then the lane goes silent — the EXACT AwaitAdopt/band-exit-Despawn shape this feature targets) there is NO next retain, so the released row's tombstone is staged in store.staged forever and never fsynced. It is not a correctness loss (boot replay re-sends an already-adopted frame, dest dedups), but the row stays on disk and, worse, on a crash-before-any-later-submit the tombstone is simply lost (row 6) so boot replay WILL re-send every acked-but-not-tombstone-fsynced frame — a re-delivery storm proportional to the idle window's acked count, not one frame.


### HR-COMPLIANCE / SEAM-INTEGRITY / SCOPE-DISCIPLINE / TEST-COVERAGE — REVISE
On my four lens dimensions the design is fundamentally sound. Seam integrity is clean: `sim::io::Durability` is already frozen and present (crates/sim/src/io/mod.rs:93-102) and the design adds nothing to it; the two new methods (`submit`/`durability`) sit on the io-prod-internal `OutboxSink` trait (crates/io-prod/src/outbox.rs:138), not the frozen seam, and no wire contract is touched. The Ephemeral/datagram byte-identical claim is verified against code: Snapshot/Input/GhostDelta are `Unreliable` (io/mod.rs:110) and ride the datagram arm (mesh.rs:1730) the design never touches; `Durability::Ephemeral` is `#[default]` (io/mod.rs:97) so every existing reliable flow stays Ephemeral, and with `durable==false` the existing false arm `retained = durable && sink.is_some()` at mesh.rs:651 fires exactly as with today's literal `None`. HR3 holds (one FSM, one sink, `Durability` a per-send param already lowered at mesh.rs:1602 — no shard/class match). Scope discipline is disciplined: 3b/3c/R-6d4 cleanly named and deferred, and the §7 decision to keep bins on `None` (inert-safe like 2c) while injecting a real `NodeOutbox` at the io-prod seam for tests is well-justified and keeps 3a testable without a bin. Coverage is achievable at Tier-B ≥90 (justfile:39-41, region-only): the new regions run in deterministic in-process tests, and `store-test-hooks` is an in-crate io-prod feature (Cargo.toml:13) so the pause-driven ordering proof is available. ONE real defect blocks a clean pass: the §3.2 non-serialization claim that block A is 'pure RAM stage + one crossbeam send, no fsync' is contradicted by the code — `submit()`→`store.commit()` contains block-on-prior (store.rs:563-567) which can block until the PRIOR batch is durable, so under the shared store a second peer writer holding the lock CAN stall on another peer's fsync. The design only surfaces this in Open Questions 3/5 instead of resolving it in the API shape. That must be resolved (it also determines what T-DBS-3 can assert). Everything else is NIT-level.

- **MEDIUM** §3.2 'no fsync under the lock' is contradicted by commit()'s block-on-prior — the shared store CAN serialize peer writers on an fsync — The design's central non-serialization property (§3.2, §3.1, and the concurrency crash note) rests on the claim that block A holds the shared NodeOutbox Mutex only across 'assign + submit — pure RAM stage + one crossbeam send, no fsync', and that the fsync WAIT is the only blocking part and lives OUTSIDE the lock. This is not what the code does. `submit()` is specified as `store.commit(); store.last_submitted()` (design §1.1) and `RedbStore::commit()` begins with BLOCK-ON-PRIOR: it waits until the PREVIOUS batch is durable before submitting the next (store.rs:563-567: `if last_durable < last_submitted { wait_durable_through(last_submitted) }`). Under ONE shared NodeOutbox, peer A's submit and peer B's submit share `last_submitted`/`last_durable`, so when peer B calls `submit()` under the lock while peer A's prior batch is still mid-fsync, peer B's block-on-prior parks ON peer A's fsync WHILE HOLDING THE SHARED LOCK. That is exactly the cross-peer serialization the split was meant to remove — it is merely bounded to ≤1 batch, not eliminated.

- **NIT** DurabilityHandle::already_durable() test-constructor needs private-field access; confirm it stays #[cfg(test)]/pub(crate) — The pure-FSM mock handle (§6.1, Open Question 2) needs a DurabilityHandle whose wait_durable_through fast-returns. DurabilityHandle's fields (last_durable, last_submitted, fsync_backpressure, durable_cv, writer_alive) are all private (store.rs:125-135). A test constructor seeding last_durable=u64::MAX + writer_alive=true is fine but must not widen the production API.

- **NIT** MockOutboxSink::submit must return a MONOTONE non-zero seq or ephemeral-inertness assertions can false-pass — §6.1 extends MockOutboxSink with `submit` returning 'a monotone counter' and `durability` returning an always-durable stub. For the `ephemeral_gate_is_inert` assertion (submit/durability NOT called on the ephemeral path) to be meaningful, and for the durable path's captured barrier_seq to be well-defined, submit must return a distinct value per call (not a constant 0 that could alias the genesis `last_submitted()==0`).
