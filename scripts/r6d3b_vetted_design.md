# R-6d3b Vetted Design of Record (wf_ab97227a)

## ADJUDICATED SYNTHESIS
Confirmed: `peer_writer` tasks are spawned onto the runtime by `spawn_mesh` (line 874), independent of the `MeshTransport` handle. The transport holds only the `lane.tx` senders; `build_app` moving the `MeshTransport` does not stop the already-running writer tasks. Boot-replay sends issued before `build_app` will be drained. The reviewer's claim on that point is confirmed.

I now have full code-grounding to adjudicate every finding. Here is my synthesis.

---

# R-6d3b — ADJUDICATED, IMPLEMENT-READY Design of Record

**VERDICT: REVISE** — the F2 async-wait mechanism, the sub-slice ordering, and the crash-window table are SOUND_TO_IMPLEMENT as written; but three code-grounded defects (one HIGH the reviewers missed, one HIGH they found, one under-specified fence) must be folded before implementation. All are fixed below; nothing needs a redesign.

## Adjudication of every finding (confirmed against HEAD 4ae5907)

| # | Finding (source lens) | Verdict | Rank |
|---|---|---|---|
| **A** | **NEW (adjudicator): the bin CANNOT read `key.peer`/`key.class` (`pub(crate)`, outbox.rs:94-95) NOR name/decode `ReliableFrame` (`pub(crate)` in io-prod lib.rs:70, all-private fields).** The §4.1 bin loop `send_durable(key.peer, key.class, rf.bytes,…)` **does not compile.** | **CONFIRMED — HIGH** | 1 |
| **B** | Boot-replay `let _ = send_durable(…)` swallows `QueueFull` → silent loss when `key.peer` ∉ lane book, or when replaying N rows overflows the bounded per-peer mpsc (`outbound_capacity`, mesh.rs:873). (2 lenses, corroborated) | **CONFIRMED — HIGH** | 2 |
| **C** | Boot replay re-calls `resolve_process_incarnation` → **double-increments the durable BootCounter** in the prod `VD_BOOT_STATE_DIR` path (lib.rs:243 `increment_on_boot` mutates); frames stamped N, gc uses N+1. Masked in dev (`VD_PROCESS_INCARNATION` explicit, lib.rs:225 short-circuit + common_env:301). | **CONFIRMED — HIGH** | 3 |
| **D** | gc-fence (§4.3 / OPEN Q#1) under-specified: bin-side `commit()` fences nothing; `scan_all()`-count poll wedges if a row fails to re-frame (Unframable, mesh.rs:1721). (2 lenses) | **CONFIRMED — MEDIUM** | 4 |
| **E** | Async waiter lacks a prompt writer-death wake — `WriterExitSignal::Drop` (store.rs:321-327) notifies only the condvar, not the new `Notify`; async waiter eats the full 100 ms `WRITER_WAIT_POLL` before failing loud. | **CONFIRMED — NIT (upgraded to LOW: fold it, it's one line)** | 5 |
| **F** | Async liveness-panic coverage OPEN QUESTION over-worried — sync sibling `park_until_durable` panic arm (store.rs:299) has NO `coverage(off)`/exemption; the Tier-B floor 90 (justfile, below ~92% baseline) absorbs it. | **CONFIRMED — resolve as stated: do nothing special.** | — |
| **G** | F2 async `Notify` mechanism (reject spawn_blocking/more-workers); lost-wakeup-free via `notify_waiters()`+enroll-then-recheck; cancel-safety of the new block-B `.await`. | **CONFIRMED SOUND** (tokio 1.52.3 verified; peer_writer select mesh.rs:1255-1273 confirmed; block C strictly after block B). **No change.** | — |

**Finding A is the decisive one the reviewers all missed** — it makes the published §4.1 bin snippet non-compiling and forces the entire drain to be an **io-prod-internal method**, which also happens to be the cleanest home for the fixes to B, C, and D. The design's own §4.2 already leaned toward a `decode_payload` accessor "to keep `ReliableFrame` private"; A proves that is not a preference but a **hard requirement**, and extends it: the *whole replay loop* (not just decode) must live in io-prod because `OutboxKey`'s fields are `pub(crate)` too.

---

## 1. F2 — async block-B wait (SOUND; unchanged from the DoR, with the finding-E one-liner folded)

Mechanism **(a)** — an `Arc<tokio::sync::Notify>` on `DurabilityHandle`/`RedbStore` — is confirmed correct and adopted. (b)/(c) rejected as in the DoR (verified: N concurrent durable sends must cost 0 workers on the `worker_threads(2)` pool, shard.rs:18; spawn_blocking spends N blocking threads, more-workers spends N workers).

**store.rs additive changes:**
- Add field `durable_notify: Arc<tokio::sync::Notify>` to `DurabilityHandle` (store.rs:124-136), `RedbStore` (:204-233), and `WriterExitSignal` (:316-319) — Arc-shared, cloned in `RedbStore::open` (:427-464) exactly like `durable_cv`.
- `already_durable()` (:190-198) gains `durable_notify: Arc::new(Notify::new())` (never awaited; struct must be complete).
- **Writer success site (store.rs:384):** add `durable_notify.notify_waiters();` immediately after `last_durable.store(max_seq, Release)`, before the existing `cv.notify_all()`. `run_writer` gains a `durable_notify: Arc<Notify>` param.
- **FOLD FINDING E:** `WriterExitSignal::Drop` (store.rs:322-327) also fires `durable_notify.notify_waiters();` alongside `cv.notify_all()`, so an async waiter gets the SAME prompt death-wake the sync path already gets. `notify_waiters()` needs no runtime handle → sound from the std-thread Drop. (Correctness does not depend on this — the timeout backstop fails loud regardless — but it makes the two liveness escapes symmetric for one line.)

**New async method (sync `wait_durable_through` at :166-176 BYTE-UNCHANGED — orchestrator.rs:280 + `NodeOutbox::commit` outbox.rs:223 stay on it):**

```rust
/// R-6d3b F2: the ASYNC durable-before-send gate. Awaits until every batch through `seq` is durable
/// WITHOUT parking a thread — mesh block-B awaits this on a tokio worker, RELEASED at the `.await`
/// (no worker starvation, unlike the sync `wait_durable_through`). Additive; the sync variant stays THE
/// persist-before-effect gate for the orchestrator MAIN-thread caller. Fails LOUD (panic) if the writer
/// dies before `seq`, mirroring the sync park's escape.
pub async fn wait_durable_through_async(&self, seq: u64) {
    if self.last_durable.load(Ordering::Acquire) >= seq { return; }        // fast path: no await/enroll
    self.fsync_backpressure.fetch_add(1, Ordering::Relaxed);               // same back-pressure counter
    loop {
        let notified = self.durable_notify.notified();                     // ENROLL before re-read
        if self.last_durable.load(Ordering::Acquire) >= seq { return; }    // re-check after enroll
        match tokio::time::timeout(WRITER_WAIT_POLL, notified).await {     // reuse the ONE-home const
            Ok(()) => {}                                                    // writer woke us; loop re-checks
            Err(_elapsed) => {
                if self.last_durable.load(Ordering::Acquire) < seq
                    && !self.writer_alive.load(Ordering::Acquire)
                {
                    panic!("wait_durable_through_async: writer died before seq {seq} — refusing to hang \
                            the send (a refusal is never a loss; recovery rehydrates + the source re-drives)");
                }
            }
        }
    }
}
```

**Lost-wakeup-free** (verified against tokio-1.52.3 `notify.rs`): `notify_waiters()` (not `notify_one()`) + enroll-then-recheck. `Notified` captures the `notify_waiters` counter on creation; a `notify_waiters()` racing after enroll but before `.await` still wakes us. The `store(Release)`→`notify_waiters()` writer order paired with our `enroll`→`load(Acquire)` order makes every interleaving either observe `>= seq` on the recheck or catch the notify. `notify_one()` here would be a lost-wakeup bug (multiple block-B futures on different `seq`s). Confirmed sound.

**Block B (mesh.rs:1759-1761)** — the ONLY mesh functional change:
```rust
if let Some((batch_seq, durability)) = gate {
    durability.wait_durable_through_async(batch_seq).await;
}
```

**Cancel-safety (mesh.rs:1625-1627, 1758 comments UPDATED):** the new `.await` adds a cancel point. Confirmed safe: `write_frame` is `.await`ed as a plain arm body in the `w.rx.recv()` select branch (mesh.rs:1255-1273), never inside a nested `select!`/`timeout`, so it drops ONLY on whole-task abort. A cancel in block B is BEFORE block C (any stream write) → the load-bearing invariant ("never a torn stream write on a `Some` stream") holds. State at cancel: frame retained in `lane.retry` (dies with the aborted task = crash-equivalent, covered by the outbox) + row durable-or-rolled-back-but-NOT-sent = **crash-window row 4**, re-driven by boot replay. Never sent-not-durable. Update the comment to: *"block B is now an `.await`; a cancel there is safe — frame retained + row durable-or-rolled-back but NOT sent ⇒ boot replay re-drives (crash-row-4-equivalent); the load-bearing invariant (no torn stream write) holds because block C has not run."*

---

## 2. THE STRUCTURAL FIX (folds A + B + C + D): the drain lives in io-prod, driven by one bin-owned incarnation

Because `OutboxKey.peer/.class` and `ReliableFrame` are `pub(crate)` (finding A), the replay loop **cannot** live in the bin. Introduce ONE io-prod-internal method that owns decode + route + fence + gc, and give the bin a thin, correct call. This is also where B, C, D are fixed.

### 2.1 `NodeOutbox::replay_all` (io-prod-internal — the ONE replay home)

```rust
impl NodeOutbox {
    /// R-6d3b BOOT REPLAY: re-drive every retained row through `transport.send_durable` at the CALLER's
    /// already-resolved `new_incarnation` (ascending scan order ⇒ the fresh lane assigns seq 0,1,2.. ⇒
    /// classify_reliable Reset-then-Accept, no gap_drop), then FENCE on durability, then gc the prior
    /// window. Fails LOUD on any un-routable / un-enqueuable row (finding B) — a durable row is NEVER
    /// silently dropped on the recovery path. Decodes the payload internally (ReliableFrame stays private,
    /// finding A). Called by shard/gateway main BEFORE build_app consumes the transport.
    ///
    /// # Errors
    /// A row whose peer is not in the transport's lane book, an mpsc that will not accept the send after
    /// bounded retry, an undecodable row, or a writer that dies before the fresh window is durable.
    pub fn replay_all(
        &mut self,
        transport: &mut dyn Transport,       // the frozen seam — the ONLY thing io-prod calls on the bin's transport
        new_incarnation: u64,
    ) -> Result<usize, ReplayError> {
        let rows = self.scan_all();                        // ascending (peer,class,incarnation,seq)
        for (key, framed) in &rows {
            let payload = decode_value_payload(framed)     // strip envelope already done by scan_all; here
                .ok_or(ReplayError::Undecodable)?;         // decode_frame::<ReliableFrame>(framed).bytes
            // FINDING B: do NOT swallow. Retry a momentarily-full LIVE lane; FAIL LOUD on a route-book miss.
            send_durable_with_retry(transport, key.peer, key.class, payload)?;
        }
        // FINDING D: fence on the SHARED DurabilityHandle high-water, NOT a scan_all COUNT (a row that
        // fails to re-frame would break a count and wedge boot). last_submitted is bumped by the peer_writer
        // tasks as they re-mirror; wait_durable_through(last_submitted) fences ALL submitted fresh rows.
        self.wait_fresh_window_durable(rows.len())?;       // §2.3
        // FINDING C is fixed at the CALL SITE (one incarnation): new_incarnation is passed in, never re-resolved.
        self.gc_below(new_incarnation);
        self.commit();                                     // durable sweep of the prior window (HIGH-3 order)
        Ok(rows.len())
    }
}
```

- **Finding A fixed:** `key.peer`/`key.class` and `ReliableFrame` are read INSIDE io-prod where they are visible; the bin never names them. `send_durable` is called through the frozen `dyn Transport` seam (`send_durable` is a `Transport` method, mesh.rs:1888). No new pub surface on `OutboxKey`/`ReliableFrame`. `decode_value_payload` is a private io-prod fn doing `decode_frame::<ReliableFrame>(framed).map(|(rf,_)| rf.bytes.into())` — re-sends the **PAYLOAD**, not the frame (confirmed: the send path re-frames at `assign_and_retain` mesh.rs:631, stamping the fresh incarnation/epoch/seq; re-sending the stored frame would double-frame + carry stale incarnation).
- **Finding B fixed:** `send_durable_with_retry` (§2.2) — a route-book miss FAILS LOUD (returns `ReplayError::Unroutable{peer}`); a momentarily-full live lane retries bounded, then fails loud. No row is `let _ =`-dropped. gc runs only after every row was ACCEPTED (`?` bails first), so an un-routable row's data is never swept.
- **Finding D fixed:** the fence is `wait_durable_through(last_submitted())` on the shared handle, not a fragile `scan_all()` count (§2.3).
- **Finding C fixed:** `new_incarnation` is a parameter — resolved ONCE in main (§3), reused for both `MeshConfig` and gc. `replay_all` never calls `resolve_process_incarnation`.

### 2.2 `send_durable_with_retry` (finding B — no silent QueueFull swallow)

```rust
// io-prod-internal. A route-book miss is a permanent can't-route → FAIL LOUD (the row stays durable on disk;
// gc never runs because `?` bails before the fence). A momentarily-FULL live lane (the synchronous replay
// loop out-pacing the peer_writer drain on a small outbound_capacity) retries with a bounded backoff.
fn send_durable_with_retry(t: &mut dyn Transport, peer: NodeId, class: MsgClass, payload: Bytes)
    -> Result<(), ReplayError>
{
    let mut attempt = 0;
    loop {
        match t.send_durable(peer, class, payload.clone(), Durability::Retained) {
            Ok(_) => return Ok(()),
            Err(SendError::QueueFull(returned)) => {
                // QueueFull fires for BOTH a route-book miss (lane absent, mesh.rs:1895) AND a full live
                // lane (try_send Full, mesh.rs:1913). Disambiguate by asking the transport if the peer is
                // routable at all; a miss is permanent → loud, a full lane is transient → bounded retry.
                if !t.is_peer_routable(peer) {             // §2.4 — a tiny additive Transport-seam? NO — see note
                    return Err(ReplayError::Unroutable { peer });
                }
                attempt += 1;
                if attempt >= REPLAY_SEND_MAX_RETRIES { return Err(ReplayError::LaneStuck { peer }); }
                std::thread::sleep(REPLAY_SEND_RETRY_BACKOFF);   // bin is on its own thread pre-build_app
                let _ = returned;                                 // payload is `.clone()`d; retry re-clones
            }
        }
    }
}
```

**Seam note (important — keeps the frozen seam UNTOUCHED):** `SendError::QueueFull` already returns the peer-book-miss vs full-lane distinction *implicitly*, but the bin cannot tell them apart from `QueueFull` alone. **Do NOT add `is_peer_routable` to the frozen `Transport` seam.** Instead, `replay_all` takes the concrete route-book knowledge from the OUTBOX side: since `replay_all` is io-prod-internal and the caller passes `&cfg.peers` (or the bin pre-validates), the cleaner cut is — **the bin passes the peer set** (`&BTreeMap<NodeId,SocketAddr>` = `cfg.peers`, which the bin already holds, lib.rs peer_book) into `replay_all`, and `replay_all` FAILS LOUD *before* sending if `!peers.contains(key.peer)`. That removes the QueueFull ambiguity entirely: a pre-send membership check catches the route-book miss (loud), and any remaining `QueueFull` during the send is definitionally a transient full lane (bounded retry). This needs **no** new `Transport` method. Final `replay_all` signature: `replay_all(&mut self, transport: &mut dyn Transport, peers: &BTreeMap<NodeId, SocketAddr>, new_incarnation: u64)`.

### 2.3 `wait_fresh_window_durable` (finding D — fence on high-water, not a count)

```rust
// FINDING D: the fence. The peer_writer tasks re-mirror each replayed send at new_incarnation (block A
// retain → submit_barrier bumps last_submitted); wait_durable_through(last_submitted()) waits for EVERY
// submitted fresh row, robust to a row that fails to re-frame (it simply never bumps last_submitted for
// that row — which is correct, there is nothing to fence for a shed row). Bounded on WRITER_WAIT_POLL
// liveness; fails loud on a DEAD writer only, never on a count mismatch (which cannot wedge us here).
fn wait_fresh_window_durable(&self, _expected: usize) -> Result<(), ReplayError> {
    // Give the peer_writers a bounded window to have run block A for all fresh sends, THEN fence on the
    // resulting high-water. (last_submitted is monotone; once all live sends have hit block A it covers
    // them.) A stuck writer fails loud via the sync wait's own liveness escape.
    self.durability.wait_durable_through(self.durability_last_submitted());
    Ok(())
}
```

The subtlety the reviewers flagged is real: the fresh rows are submitted by the *async* peer_writers, so the bin's synchronous `replay_all` must not gc until they are on disk. Using `wait_durable_through(last_submitted())` is the robust fence — `last_submitted` is bumped under `submit_nonblocking` by each re-mirror, and `wait_durable_through` already fails loud on a dead writer (store.rs:173-175). This sidesteps the count-wedge entirely (a shed/Unframable row never bumps `last_submitted`, so it never needs fencing). One residual: `replay_all` must ensure the peer_writers have *reached* block A for all sends before reading `last_submitted` — enforce with a short bounded settle keyed off `expected` reached in `scan_all().count()` at `new_incarnation` OR simpler, since `send_durable` only *enqueues*, drive a bounded `wait_durable_through` loop that re-reads `last_submitted` until it stops advancing across two `WRITER_WAIT_POLL`s. Pin this exact mechanism with `boot_replay_gc_strictly_after_durable` (§6) — this closes crash-window B5 by construction. **This is the ONE remaining design detail to lock at implementation; both candidate mechanisms are sound and testable — prefer the high-water fence.**

---

## 3. Bin wiring — shard.rs + gateway.rs (finding C fixed at the call site)

Resolve incarnation ONCE; open the outbox; keep the `NodeOutbox` handle for the drain BEFORE wrapping into `SharedOutbox` for `spawn_mesh`.

```rust
// shard.rs / gateway.rs main — ONE incarnation (finding C: never resolve twice):
let new_incarnation = vd_bins::resolve_process_incarnation(&env)?;
let peers = env.peer_book("VD_PEERS")?;
let mut node_outbox: Option<NodeOutbox> = vd_bins::open_node_outbox(&env)?;    // hold the handle for the drain

// Wrap a CLONE-able SharedOutbox for spawn_mesh. Because NodeOutbox is not Clone, open ONE and share via Arc:
let shared: Option<SharedOutbox> = node_outbox.take().map(|ob|
    Arc::new(Mutex::new(Box::new(ob) as Box<dyn OutboxSink + Send>)));

let (mut transport, _control) = spawn_mesh(runtime.handle(), &trust,
    &MeshConfig::new(local, env.parse("VD_BIND")?, peers.clone(),
        env.parse("VD_OUTBOUND_CAP")?, new_incarnation),   // SAME new_incarnation
    shared.clone())?;                                        // replaces the `None` (shard.rs:36 / gateway.rs:35)

// BOOT REPLAY before build_app moves the transport (app.rs:87):
if let Some(sh) = shared.as_ref() {
    let mut g = sh.lock().unwrap_or_else(PoisonError::into_inner);
    // replay_all is io-prod-internal on NodeOutbox — but `sh` is a Box<dyn OutboxSink>. Add replay_all to
    // the OutboxSink trait (io-prod-internal seam) so the SharedOutbox can drive it. See note below.
    g.replay_all(&mut transport, &peers, new_incarnation)?;
}

let mut node = build_app(NodeConfig { node_id: local, kind: NodeKind::StubShard }, transport);
```

**One wrinkle:** after wrapping into `Arc<Mutex<Box<dyn OutboxSink>>>`, the bin can only call trait methods. So **add `replay_all` to the `OutboxSink` trait** (io-prod-internal, NOT the frozen seam) with a default `unimplemented!()`-free signature; `NodeOutbox` implements it, `MockOutboxSink` gets a trivial impl. Alternatively drain BEFORE wrapping (call `node_outbox.replay_all(...)` on the concrete `NodeOutbox` while it's still owned, then wrap) — **prefer this**: it keeps `replay_all` an inherent `NodeOutbox` method (finding A's tighter surface) and avoids growing the trait. Sequence: open `NodeOutbox` → `spawn_mesh` needs the sink though, so wrap first... The clean resolution: `spawn_mesh` takes `Option<SharedOutbox>`; drain must run after `spawn_mesh` (transport exists) but the concrete handle is inside the Arc. **Decision: add `replay_all` to the `OutboxSink` trait** (it's io-prod-internal, already the object-safe seam the FSM uses) — that is the coherent home and keeps ONE wrapping. `MockOutboxSink::replay_all` is a no-op returning `Ok(0)`.

- **HR3:** ONE `open_node_outbox`, ONE `SharedOutbox`, ONE `replay_all`, ONE gate. Shard + gateway wire byte-identically. Gateway opens for uniformity (empty outbox → `replay_all` scans 0 rows → no-op).
- **F3 boot-check (R-6d3a):** confirmed `<< 256` — a shard's book = `{ORCH, GATEWAY}` (2, lib.rs:362), a gateway's = `{ORCH, SHARD}` + dev clients (lib.rs:339-344), all far below `OUTBOX_WRITER_CHANNEL_DEPTH = 256` (outbox.rs:49). The tripwire never fires at real scale.
- **`SharedOutbox` must become `pub`** (mesh.rs:63, currently private `type`) so the bin can name it — io-prod-internal surface, not the frozen seam. Re-export `OutboxSink` (already `pub`, outbox.rs:150).
- **No magic numbers:** `REPLAY_SEND_MAX_RETRIES` + `REPLAY_SEND_RETRY_BACKOFF` are NEW named consts with ONE home (io-prod, next to `WRITER_WAIT_POLL`); the replay-fence poll bound reuses `WRITER_WAIT_POLL`.

---

## 4. Boot-replay crash-window table (adopted from the DoR; B5 now UNREACHABLE by construction)

The DoR's B0–B5 table is CONFIRMED correct. With finding-D's high-water fence, **B5 (gc before fresh durable) is unreachable**: `replay_all` calls `gc_below` only after `wait_fresh_window_durable` returns (fences on `last_submitted`, fails loud on a dead writer). B3 (all fresh durable, before gc) is loss-free (both copies survive). Finding-B's fix adds one row:

| # | SIGKILL / error point | Old rows | Fresh rows | Sent? | Outcome |
|---|---|---|---|---|---|
| B6 | replay hits an **un-routable peer** (finding B) | PRESENT | none | no | `replay_all` returns `Err(Unroutable)` BEFORE gc → boot FAILS LOUD. Row stays durable; a future boot with the peer re-seeded (or the R-6d3c orchestrator re-solicit) re-drives it. **NO SILENT LOSS.** |

Steady-state (non-replay) durable-before-send window unchanged (row 4: durable-but-not-sent ⇒ replayed). F2 cancel lands in row 4.

---

## 5. F1 — the deterministic durable-before-send pin (store-test-hooks; runs in R-6d4)

Adopted as the DoR §2.1 specifies. Add the `store-test-hooks`-only `pause_release: Option<Arc<AtomicBool>>` to `StoreTuning` (store.rs:73-92) and check it in `maybe_pause_before_fsync` (store.rs:344-346: `while !release.map_or(false, |r| r.load(Acquire)) { park_timeout(short) }`), so the test flips it to let the fsync proceed (deterministic unpause, no SIGKILL). Test `durable_row_absent_from_scan_all_until_block_b_has_waited`: writer parks pre-fsync on the sentinel → marker file appears → assert `scan_all()` lacks the row AND the receiver has NOT drained the frame → flip `pause_release` → fsync → `last_durable` bumps → `notify_waiters()` → block B returns → block C sends → assert `scan_all()` has it AND receiver drains. `#[cfg(feature = "store-test-hooks")]`, EXCLUDED from `just gate`.

---

## 6. Tests + Tier-B ≥ 90 (io-prod region-floor 90, no `--branch`, justfile)

**R-6d3b-1 (io-prod, DEFAULT gate — deterministic):**
- `async_wait_fast_returns_when_already_durable` (`#[tokio::test]`, `last_durable >= seq` ⇒ no enroll).
- `async_wait_wakes_on_writer_notify` (real `RedbStore`; task awaits, `put`+`submit_nonblocking`, assert completes after fsync — covers enroll→await→wake).
- `async_block_b_wait_returns_when_durable` (two-node mesh, `Retained` Saga frame ⇒ `scan_all` has the row AND receiver drains — end-to-end block-B `.await`).
- `two_peers_do_not_starve_workers` (peer A durable send parked via `pause_on_key_prefix`; assert peer B's ack/datagram to a third peer progresses within a bounded `tokio::time::timeout` — **the property F2 exists to prove**).
- `writer_death_wakes_async_waiter_promptly` (finding E: drop the store mid-wait; assert the async waiter fails loud without eating a full `WRITER_WAIT_POLL` — gated `store-test-hooks` if not deterministically triggerable in the default gate).
- MockOutboxSink `submit_barrier`→`already_durable()` async fast-returns (mock FSM tests unchanged); assert `gate == None` is a no-op.
- **F (coverage):** the async liveness-**panic** arm rides the Tier-B floor 90 (below ~92% baseline) exactly like the sync sibling (store.rs:299, no `coverage(off)`, no exemption). **Do NOT add an exemption** — close the OPEN QUESTION as "nothing special."

**R-6d3b-2 (bins, DEFAULT gate — mem/mock, deterministic):**
- `boot_replay_redelivers_outbox_rows` (retain+commit 2 `Retained` rows at inc N; reopen; `replay_all` against a recording MockTransport; assert ascending payload order + gc swept only `< new_incarnation`).
- `boot_replay_gc_strictly_after_durable` (finding D: drive the fence; assert gc issued only after `wait_fresh_window_durable`).
- `boot_replay_unroutable_peer_fails_loud` (finding B: a row whose peer ∉ `peers` ⇒ `replay_all` errors, row survives, NO `let _ =` drop).
- `boot_counter_advances_by_exactly_one_per_boot` (finding C: with an outbox wired + rows to replay, `VD_BOOT_STATE_DIR` counter advances by 1, not 2).
- `open_node_outbox_wraps_into_shared_outbox` (VD_OUTBOX_PATH temp + `VD_OUTBOX_EPHEMERAL_OK=1`; boot opens, wraps, `spawn_mesh` accepts; F3 check passes) — extend `dev_cluster_smoke`.

**F1** (`durable_row_absent…`) runs in R-6d4 store-test-hooks tier.

New coverable regions (async wait fast-path/backpressure/enroll-loop/timeout arms; the writer `notify_waiters()` line; block-B `.await`; `replay_all` decode/route/fence/gc; `send_durable_with_retry`) all run in DETERMINISTIC in-process tests → ratchet Tier-B UP. Never lower the floor.

---

## 7. Files touched (per sub-slice)

**R-6d3b-1 (io-prod ONLY — make block B production-safe FIRST; independently mergeable):**
- `crates/io-prod/src/store.rs`: `durable_notify: Arc<Notify>` on `DurabilityHandle`/`RedbStore`/`WriterExitSignal`; clone in `open`; `notify_waiters()` at :384 AND in `WriterExitSignal::Drop` (finding E); NEW `wait_durable_through_async` (sync `wait_durable_through` UNCHANGED); `already_durable()` gains the field; `run_writer` gains the param; `StoreTuning` + `maybe_pause_before_fsync` gain `pause_release` (store-test-hooks).
- `crates/io-prod/src/mesh.rs`: block B (:1759-1761) → `.await` the async variant; cancel-safety comments (:1625-1627, :1758) updated.
- `crates/io-prod/tests/`: F2 tests (§6) + F1 pin `durable_before_send_pin.rs` (store-test-hooks).

**R-6d3b-2 (bin wiring + boot replay — flips the gate LIVE, lands with its tests):**
- `crates/io-prod/src/mesh.rs`: `pub type SharedOutbox` (:63).
- `crates/io-prod/src/outbox.rs`: NEW private `decode_value_payload` (`decode_frame::<ReliableFrame>`→`.bytes`, keeps `ReliableFrame` private — finding A); NEW `send_durable_with_retry` + `wait_fresh_window_durable` helpers; NEW `OutboxSink::replay_all(&mut self, &mut dyn Transport, &BTreeMap<NodeId,SocketAddr>, u64) -> Result<usize, ReplayError>`; NEW `ReplayError` enum (`Unroutable{peer}` / `LaneStuck{peer}` / `Undecodable` / writer-dead via the fence's own panic); NEW consts `REPLAY_SEND_MAX_RETRIES` / `REPLAY_SEND_RETRY_BACKOFF`.
- `crates/bins/src/bin/shard.rs`: resolve incarnation ONCE (finding C); `open_node_outbox` → wrap `SharedOutbox`; pass to `spawn_mesh` (replace `None` :36); `replay_all(&mut transport, &peers, new_incarnation)?` before `build_app` (:46).
- `crates/bins/src/bin/gateway.rs`: identical wiring (replace `None` :35; drain before `build_app` :37).
- `crates/bins/tests/`: the boot-replay tests (§6); extend `dev_cluster_smoke`.
- `crates/bins/src/lib.rs`: `open_node_outbox` (:270-292) REUSED as-is.

---

## 8. Sub-slice ordering (CONFIRMED)

**R-6d3b-1 (io-prod only):** F2 async block-B wait + finding-E death-wake + F1 hook/pin. Makes block B production-safe; provable at the io-prod seam via direct `NodeOutbox` injection; no bin/boot change; ratchets Tier-B independently.

**R-6d3b-2 (bin wiring + boot replay):** `pub SharedOutbox`; `replay_all`/`decode_value_payload`/`send_durable_with_retry`/`wait_fresh_window_durable`; ONE-incarnation wiring in shard.rs + gateway.rs; drain before `build_app`; gc strictly after the durable fence. The "changes prod boot" step, landed as a unit with its bins tests (per R-6d3a §7).

**OUT OF SCOPE (named as later slices):** R-6d3c — saga AwaitAdopt `SourceUnreachable` phase-split + dest `InterShardFlow::TransientDiscard` + adopt-poisoning (the NEVER-restart closure that retires D-6 #1; also the home for multi-shard/dynamic-client route-book coverage flagged by finding B). R-6d4 — composed SIGKILL-source-in-AwaitAdopt e2e (real QUIC, extends `orchestrator_crash.rs`) + both-ends-restart replay proptest; where the F1 store-test-hooks pin runs in the process tier.

---

## HR / seam posture
- **Frozen `sim::io` + `vd-wire` UNTOUCHED:** `wait_durable_through_async`, `durable_notify`, `pub SharedOutbox`, `decode_value_payload`, `OutboxSink::replay_all`, `ReplayError` are io-prod-internal; replay drives the EXISTING `Transport::send_durable` through `&mut dyn Transport`; `vd-wire` framing unchanged. `ReliableFrame`/`OutboxKey` stay `pub(crate)` (finding A honored — the drain moved into io-prod rather than widening their surface).
- **No magic numbers:** `OUTBOX_WRITER_CHANNEL_DEPTH` (F3), `WRITER_WAIT_POLL` (async timeout + fence), `REPLAY_SEND_MAX_RETRIES`/`REPLAY_SEND_RETRY_BACKOFF` (new, ONE home each).
- **HR3:** ONE `spawn_mesh`, `open_node_outbox`, `SharedOutbox`, `replay_all`, gate — no shard-kind match.
- **HR5:** every new region covered by deterministic in-process tests; the one async liveness-panic arm rides the floor like its sync sibling.

**VERDICT: REVISE** — fold the three HIGHs (A: drain must be io-prod-internal because `OutboxKey`/`ReliableFrame` are `pub(crate)`; B: no silent `QueueFull` swallow — pre-send peer-membership check + bounded retry + fail-loud; C: resolve incarnation ONCE, never re-call `resolve_process_incarnation`) plus the MEDIUM fence (D: fence on `last_submitted` high-water, not a `scan_all` count) and the one-line NIT (E: `notify_waiters()` in `WriterExitSignal::Drop`); then SOUND_TO_IMPLEMENT. The F2 async-wait mechanism, sub-slice ordering, and crash-window table are correct as designed and need no change.

---
## SUB-SLICES (proposed)
- R-6d3b-1 (io-prod ONLY, makes block B production-safe FIRST): F2 async block-B wait — add Arc<tokio::sync::Notify> to DurabilityHandle+RedbStore (Arc-shared), writer notify_waiters() at store.rs:384 alongside cv.notify_all(), NEW async DurabilityHandle::wait_durable_through_async(seq) (sync wait_durable_through UNCHANGED for the orchestrator MAIN-thread gate), block B (mesh.rs:1759-1761) awaits the async variant, cancel-safety comment updated. PLUS F1: the store-test-hooks pause_release unpause hook + the deterministic durable_row_absent_from_scan_all_until_block_b_has_waited pin. Gate: F2 tests in the DEFAULT tier ratchet Tier-B up; F1 pin runs in store-test-hooks (R-6d4). Mergeable independently — no bin/boot change, provable at the io-prod seam via direct NodeOutbox injection.
- R-6d3b-2 (bin wiring + boot replay — flips the gate LIVE): make SharedOutbox pub + add NodeOutbox::decode_payload (+ optional replay_fence); in shard.rs + gateway.rs main open_node_outbox(&env)? -> wrap Arc<Mutex<Box<dyn OutboxSink+Send>>> -> pass to spawn_mesh (replace the two None literals). Boot replay BEFORE build_app: scan_all (ascending) -> decode_payload per row -> transport.send_durable(peer, class, payload, Retained); poll until all N fresh rows durable at new_incarnation; THEN gc_below(new_incarnation)+commit (STRICT HIGH-3 ordering). Confirm the F3 boot-check passes at real peer counts (<< 256). Gate: bins boot-replay tests (mem/mock, deterministic) + dev_cluster_smoke on the wired path. This is the 'changes prod boot' step, landed as a unit with its tests (per R-6d3a §7).
- OUT OF SCOPE (named as later slices): R-6d3c — the saga AwaitAdopt SourceUnreachable phase-split + the dest InterShardFlow::TransientDiscard + adopt-poisoning (the NEVER-restart closure that actually retires D-6 #1). R-6d4 — the composed SIGKILL-source-in-AwaitAdopt e2e (real QUIC, extends orchestrator_crash.rs) + the both-ends-restart replay proptest; this is where the F1 store-test-hooks pin runs in the process tier.

---
## CRASH-WINDOW TABLE
Boot-replay crash-window table (source shard; ONE producer-less Retained frame — AwaitAdopt TransientBatch on Saga, or band-exit GhostFlow::Despawn on GhostReliable). Assumes the R-6d3b-2 wired case (real NodeOutbox open + boot replay). "fresh rows" = re-sent rows at new_incarnation; "old rows" = the crashed process's incarnation rows.

| # | SIGKILL point (during BOOT REPLAY, source restart) | Old rows on disk | Fresh rows on disk | Fresh sent? | Outcome / recovery |
|---|---|---|---|---|---|
| B0 | before scan_all (post spawn_mesh, pre-drain) | PRESENT | absent | no | Next boot re-runs replay from the SAME old rows (nothing consumed yet). Idempotent — replay is a pure function of the old rows. NO LOSS. |
| B1 | mid-drain: some payloads send_durable-enqueued, none fsynced yet | PRESENT | absent (peer_writers not fsynced) | no | Old rows still on disk; enqueued OutFrames die with the process (RAM mpsc). Next boot re-drains the SAME old rows ⇒ re-enqueues all. NO LOSS, NO gc yet ran. |
| B2 | after a fresh row is durable (block B fenced it), before ALL fresh rows durable, before gc | PRESENT | PARTIAL (k of N) | some | Old rows intact ⇒ next boot re-drains ALL N old rows again. The k fresh rows are duplicates at new_incarnation (LWW by (peer,class,new_inc,seq) key — re-mirror is idempotent). Re-boot bumps incarnation AGAIN (new_incarnation'); the fresh rows re-based from old rows still key deterministically. NO LOSS (old rows are the source of truth until gc). Possible at-least-once double-send absorbed by receiver dedup / journal (transfer, BATCH_STEP). |
| B3 | ALL fresh rows durable, before gc_below | PRESENT | PRESENT (all N) | all (or in flight) | THE HIGH-3 WINDOW. Old + fresh BOTH on disk. Next boot re-drains BOTH old AND fresh rows (scan_all returns all) ⇒ re-sends duplicates, receiver dedups. NO LOSS. gc will sweep old on a clean boot. This is why gc runs STRICTLY here (after all fresh durable): a crash here loses nothing (both copies survive). |
| B4 | after gc_below staged + committed (old swept), fresh durable | ABSENT (swept) | PRESENT | all | Terminal-clean. Only the fresh (new_incarnation) rows remain; a subsequent crash replays THEM (row 4-equivalent of the steady-state table). NO LOSS. The invariant HIGH-3 protects: gc never precedes fresh durability, so "old gone + fresh not-durable" is UNREACHABLE. |
| B5 | gc_below issued BEFORE fresh rows durable (THE FORBIDDEN ordering — must be prevented) | ABSENT (swept) | absent/partial | no | BOTH LOST — the batch is gone with no durable copy. THIS IS WHY §4.3 fences gc on the fresh-window durability. The design makes B5 unreachable by construction (poll until all N fresh rows durable, then gc). Listed to name the hazard the ordering closes. |

Steady-state (non-replay) durable-before-send window is unchanged from R-6d3a §5 (row 4: durable-but-not-sent ⇒ replayed on next boot). F2's async block-B cancel (task abort mid-wait) lands in that same row 4: frame retained + row durable-or-rolled-back but NOT sent ⇒ boot replay re-drives; never sent-not-durable.

---
## DESIGNER (original)
ic
# R-6d3b — Flip the Durable-Outbox Gate LIVE: Design of Record (IMPLEMENT-READY)

**Scope:** F2 (async block-B wait so a parked durable send never occupies a tokio worker) + F1 (the deterministic durable-before-send ordering pin) + BIN WIRING (open the per-node store, pass the real `SharedOutbox` to `spawn_mesh` in `shard.rs`/`gateway.rs`) + BOOT REPLAY (drain the outbox as N `send_durable(Retained)` calls before `build_app` consumes the transport, then `gc_below`). All line cites against worktree HEAD (4ae5907, R-6d3a landed). The frozen `sim::io` seam + `vd-wire` contract stay UNTOUCHED; `OutboxSink`/`NodeOutbox`/`DurabilityHandle` are io-prod-internal.

**Verdict on the R-6d3a API surface:** sufficient for bin wiring + boot replay AS-IS; **F2 needs ONE minimal additive change** to `DurabilityHandle` + `RedbStore` (an async notify + an async wait method), specified in §1. Everything else composes over existing APIs.

---

## 0. The decisive facts (grounded)

- **Block B today is a SYNC park.** `mesh.rs:1759-1761`: `if let Some((batch_seq, durability)) = gate { durability.wait_durable_through(batch_seq); }` → `DurabilityHandle::wait_durable_through` (store.rs:166-176) → `park_until_durable` (store.rs:281-305), which parks on a `std::sync::Condvar` (`cv.wait_timeout`, store.rs:293). This is a **blocking park of the calling thread** — and `peer_writer` (mesh.rs:1195) is a tokio task, so the park occupies a tokio worker.
- **The runtime has exactly 2 workers.** `shard.rs:17-20` / `gateway.rs:16-19`: `worker_threads(2)`. The accept loop + per-peer `serve_connection` ack-readers + all `peer_writer` tasks share this pool. TWO `peer_writer`s parked in block B occupy BOTH workers → the mesh recv/ack/accept I/O path starves for the fsync duration. (The sim tick loop is on the MAIN thread, `shard.rs:94` `node.step_tick()`, so it is NOT starved — only the mesh I/O is.)
- **The writer already publishes durability under a mutex-guarded condvar.** `run_writer` at store.rs:383-388: `last_durable.store(max_seq, Release); let (lock, cv) = &*durable_cv; let _g = lock.lock()…; cv.notify_all();`. This is the ONE site an async `Notify::notify_waiters()` is added, alongside the existing `notify_all()`.
- **`DurabilityHandle` is `Clone` and Arc-shares every field** with `RedbStore` (store.rs:124-136 / 458-464). Adding one `Arc<tokio::sync::Notify>` field, cloned into `run_writer`, is the whole F2 plumbing. `tokio::sync::Notify` is already imported in mesh.rs (`mesh.rs:921` `Notify::new()`), so no new dep.
- **The orchestrator sync path is a DIFFERENT method on a DIFFERENT caller.** `orchestrator.rs:280` calls `DurabilityHandle::wait_durable_through` (the sync park) on its MAIN thread. `NodeOutbox::commit` (outbox.rs:217-224) also calls the sync `wait_durable_through`. The async variant is **purely additive** — a new `wait_durable_through_async`; the sync method is byte-unchanged.
- **`submit_barrier` already hands block B a cloned `DurabilityHandle`.** `OutboxSink::submit_barrier -> Option<(u64, DurabilityHandle)>` (outbox.rs:182, impl 252-258). So F2 changes only WHICH method block B awaits on that handle — no new plumbing through the gate.
- **Boot replay is N ordinary `send_durable(Retained)` calls (MEDIUM-1).** `MeshTransport::send_durable(to, class, bytes, Durability::Retained)` (mesh.rs:1888-1918) is the ordinary above-seam send; `build_app<T: Transport>(cfg, transport)` MOVES the transport (app.rs:87, 101-106). So the bin drains BEFORE `build_app`, issuing `transport.send_durable(...)` per row, then hands the transport to `build_app`. No `inject_replay`.
- **`scan_all` returns already-envelope-stripped framed bytes.** `NodeOutbox::scan_all` (outbox.rs:226-236) strips the `OUTBOX_FORMAT_VERSION` envelope via `decode_value`; the value is the L2-contract `postcard(ReliableFrame{epoch:u32::MAX,..})` framed bytes (mesh.rs:3293-3309 `expected_encoded`). To re-send via `send_durable`, we must recover the **PAYLOAD** (`.bytes`), NOT re-send the frame — decode `decode_frame::<ReliableFrame>` (framing.rs:107) → `.bytes` (§4).
- **Real peer counts are tiny.** `ORCH=1, GATEWAY=2, SHARD=3` (lib.rs:28-30); a shard's `VD_PEERS` = `{ORCH, GATEWAY}` (lib.rs:362), a gateway's = `{ORCH, SHARD}` + seeded clients (lib.rs:339-344). All `<< OUTBOX_WRITER_CHANNEL_DEPTH = 256` (outbox.rs:49). The R-6d3a F3 boot-check (spawn_mesh rejects `peer count > OUTBOX_WRITER_CHANNEL_DEPTH` when a sink is wired) never trips at real scale — confirm in §3.
- **io-prod is Tier-B region-floor 90, NO `--branch`** (justfile:39-41). `store-test-hooks` is a feature (Cargo.toml io-prod:13, forwarded by bins:59); the F1 pin is `#[cfg(feature = "store-test-hooks")]` so it runs in the R-6d4 process tier, not the default gate.

---

## 1. F2 — the async block-B wait (BINDING; the minimal additive change)

### 1.1 Mechanism decision — (a) async `Notify`, REJECTING (b) and (c)

**CHOSEN: (a) — an `Arc<tokio::sync::Notify>` on `DurabilityHandle`/`RedbStore` that the writer `notify_waiters()` alongside its existing `cv.notify_all()`, + an async `DurabilityHandle::wait_durable_through_async(seq)` that block B `.await`s.**

- **(b) `spawn_blocking` the existing sync park — REJECTED.** It removes the worker-starvation *symptom* (the park moves to the blocking pool) but at a real cost: each concurrent durable send spawns a blocking-pool thread that PARKS on a condvar for the whole fsync, and the default blocking pool is 512 — under a burst of producer-less sends (source fans a `TransientBatch` to several dest peers + `Despawn` to neighbours, the exact R-6d3a MF-3 load) you spawn N blocking threads that each block, which is strictly worse than N async waiters (each async waiter is a suspended future costing a few hundred bytes, no thread). It also re-introduces a thread hop on the hot durable path. `spawn_blocking` is the band-aid the R-6d3a adjudication already flagged as an "R-6d4 load-test lever only" (r6d3a §0 finding #3), not the cure.
- **(c) raise `worker_threads` — REJECTED as a band-aid** (the prompt's own framing). It does not bound: any K concurrent durable parks starve K workers; you would have to size the pool to the peak concurrent durable-send fan-out, which is exactly the unbounded quantity the async path makes free. Also a global knob for a local problem.
- **(a) is the scaling answer:** N concurrent durable sends yield N *suspended futures* on the same 2 workers (the worker is released at the `.await`, free to drive the accept loop / ack readers / other `peer_writer`s), vs (b)/(c) which consume N threads/workers. This is the "N workers vs park N" argument the prompt demands: with (a), block-B waiters cost **zero workers** — the worker returns to the scheduler the instant the future suspends.

### 1.2 The exact signatures

**`RedbStore` + `DurabilityHandle` gain one shared field** (store.rs, additive):

```rust
// store.rs — new field on BOTH DurabilityHandle (~:124-136) and RedbStore (~:204-233), Arc-shared like the
// existing durable_cv. An async wake fired by the writer alongside the sync condvar notify.
durable_notify: Arc<tokio::sync::Notify>,
```

- In `RedbStore::open` (store.rs:412-478): `let durable_notify = Arc::new(tokio::sync::Notify::new());`, `Arc::clone`d into (i) the writer closure (a new param to `run_writer`), (ii) the `DurabilityHandle`, (iii) the `RedbStore`.
- **The writer fires it at the SAME site as the condvar notify** (store.rs:383-388), one line added:

```rust
if apply_batch(&db, &merged) {
    last_durable.store(max_seq, Ordering::Release);   // :384 (unchanged, the Release is the publish)
    durable_notify.notify_waiters();                  // NEW: wake async block-B waiters (runtime-agnostic)
    let (lock, cv) = &*durable_cv;                     // :385 (unchanged)
    let _g = lock.lock().unwrap_or_else(PoisonError::into_inner);
    cv.notify_all();                                   // :387 (unchanged — sync orchestrator/commit path)
    break;
}
```

`Notify::notify_waiters()` requires no tokio runtime handle (it only touches registered waiter wakers), so calling it from the plain `std::thread` writer is sound.

**`DurabilityHandle` gains the async wait** (store.rs, additive — the sync `wait_durable_through` at :166-176 is UNCHANGED):

```rust
/// R-6d3b F2: the ASYNC durable-before-send gate. Awaits until every batch through `seq` is durable WITHOUT
/// parking a thread — the mesh block-B (`write_frame`) awaits this on a tokio worker, which is RELEASED at the
/// `.await` (no worker starvation, unlike the sync `wait_durable_through`). The sync variant stays THE
/// persist-before-effect gate for the orchestrator's MAIN-thread caller (orchestrator.rs:280); this is additive.
/// Fails LOUD (panics, same posture as the sync park's liveness escape) if the writer dies before reaching `seq`.
pub async fn wait_durable_through_async(&self, seq: u64) {
    // Fast path: already durable ⇒ no await, no registration (the ~always case at 50Hz).
    if self.last_durable.load(Ordering::Acquire) >= seq { return; }
    if self.last_durable.load(Ordering::Acquire) < seq {
        self.fsync_backpressure.fetch_add(1, Ordering::Relaxed);   // same back-pressure counter as the sync gate
    }
    loop {
        // LOST-WAKEUP-FREE: register the future BEFORE re-reading last_durable. Notify::notified() enrolls the
        // waiter on creation/first-poll; a notify_waiters() that races in after we hold `notified` but before we
        // await still wakes us (Notify buffers one permit per registered-and-not-yet-consumed waiter). See §1.3.
        let notified = self.durable_notify.notified();
        if self.last_durable.load(Ordering::Acquire) >= seq { return; }   // re-check after enroll
        // Liveness backstop: a dead writer can never bump last_durable ⇒ never notify. Bound the await so a
        // dead writer fails loud instead of hanging the future forever (mirrors park_until_durable's escape).
        match tokio::time::timeout(WRITER_WAIT_POLL, notified).await {
            Ok(()) => {}                       // woken by the writer's notify_waiters(); loop re-checks
            Err(_elapsed) => {                 // timed out: check liveness
                if self.last_durable.load(Ordering::Acquire) < seq
                    && !self.writer_alive.load(Ordering::Acquire)
                {
                    panic!("DurabilityHandle::wait_durable_through_async: the durable writer thread died \
                            without reaching seq {seq} — refusing to hang the send (a refusal is never a \
                            loss; recovery rehydrates + the source re-drives)");
                }
            }
        }
    }
}
```

**Block B** (mesh.rs:1759-1761) becomes the ONLY mesh change:

```rust
// R-6d3b F2: async wait — the tokio worker is RELEASED here (no thread park), so N concurrent durable sends
// suspend as N futures on the 2-worker pool, never starving the accept/ack/recv I/O tasks.
if let Some((batch_seq, durability)) = gate {
    durability.wait_durable_through_async(batch_seq).await;
}
```

**The `already_durable()` test-ctor** (store.rs:190-198) gains the field: `durable_notify: Arc::new(tokio::sync::Notify::new())` (it fast-returns on `last_durable == u64::MAX`, so the notify is never awaited — but the struct must be complete).

### 1.3 Lost-wakeup-free argument

The hazard: the writer bumps `last_durable` + `notify_waiters()` in the window between block B's `last_durable` load and its `.await` — if the notify is missed, the future hangs until the `WRITER_WAIT_POLL` timeout (100 ms), a latency bug (not a correctness bug, since the timeout re-checks and proceeds).

The design is lost-wakeup-free by the standard `Notify::notified()` enroll-then-check pattern, which `tokio::sync::Notify` documents as its intended usage:
1. `let notified = self.durable_notify.notified();` — creating the `Notified` future **enrolls the waiter** in the Notify's intrusive wait list on first poll. Critically, `notify_waiters()` wakes every waiter enrolled at the call instant.
2. We then **re-read `last_durable` AFTER enrolling** (`if … >= seq { return; }`). If the writer's `store(Release)` at store.rs:384 landed before our re-read, we see it and return without awaiting — no lost wakeup.
3. If the writer's `store` lands AFTER our re-read but its `notify_waiters()` fires while we hold the enrolled (un-polled-to-completion) `notified`, the wake is delivered when we `.await` it — `notify_waiters()` wakes all currently-registered waiters, and ours is registered.
4. The ordering that makes this airtight: the writer does `store(Release)` THEN `notify_waiters()` (store.rs:384 then the new line). We do `enroll` THEN `load(Acquire)`. The only interleaving that could miss is "writer notifies, THEN we enroll" — but in that interleaving the writer's `store` happened-before its notify, and our `load(Acquire)` (which follows our enroll) is AFTER the enroll, hence after any notify that preceded the enroll; and a `store(Release)`/`load(Acquire)` pair means if the notify preceded our enroll, the store preceded it too, so our `load` observes `>= seq` and we return. **Every interleaving either observes durability on the re-check or catches the notify.** The `timeout(WRITER_WAIT_POLL)` is a pure backstop (identical role to the sync park's `wait_timeout`, store.rs:293-294), not part of the correctness argument.

One subtlety worth stating: `notify_waiters()` (not `notify_one()`) is correct here because MULTIPLE block-B futures may be waiting for the same or different `seq`s; `notify_waiters` wakes all of them and each re-checks its own `seq` against the monotone `last_durable` — a waiter whose `seq` is not yet reached simply re-enrolls and re-awaits. `notify_one()` would be a lost-wakeup bug (it wakes only one waiter and does NOT buffer for the others).

### 1.4 No-regression proof for the orchestrator sync path

The orchestrator's persist-before-effect gate (`orchestrator.rs:280`) calls `DurabilityHandle::wait_durable_through` — the SYNC method (store.rs:166-176), which is **byte-unchanged**: same `park_until_durable`, same condvar, same liveness escape. `NodeOutbox::commit` (outbox.rs:217-224) likewise still calls the sync `wait_durable_through`. The async method is a NEW method; the writer now fires `notify_waiters()` in ADDITION to `cv.notify_all()` at the same site — the condvar notify is untouched, so the sync waiters wake exactly as before. The only shared-state addition is one `Arc<Notify>` field; it is written only by the writer's `notify_waiters()` and read only by async waiters, never on the sync path. **The orchestrator MAIN-thread gate is behaviourally identical.** (Coverage note: the sync `wait_durable_through` regions stay exercised by the existing orchestrator/store tests; the new async method's regions are covered by the F2 io-prod tests in §6.)

### 1.5 Cancel-safety analysis for the new `.await` in `write_frame`

The R-6d3a comment (mesh.rs:1625-1627) states the invariant: `write_frame` is NEVER wrapped in `timeout`/`select!`, and "the only cancel point in the writer is the `w.rx.recv()` await above" (mesh.rs:1255). R-6d3a could truthfully say block B "is a synchronous condvar park (not an `.await`), so cancel-safety is preserved" (mesh.rs:1758). **F2 changes this: block B is now a genuine `.await`, so a `write_frame` future CAN be dropped mid-block-B.** We must verify this is safe.

The `peer_writer` `select!` (mesh.rs:1213-1255) drives `write_frame(...).await` inside the `maybe = w.rx.recv()` arm (mesh.rs:1255-1257). `write_frame` is `.await`ed as a plain expression, NOT inside a nested `select!`/`timeout` — so the ONLY way its future is dropped is if the WHOLE `peer_writer` task is dropped (the task is aborted, e.g. on `MeshControl::kill` / endpoint teardown / node shutdown). It is never cancelled by a sibling `select!` branch (the outer `select!` completes the chosen arm's body — `write_frame(...).await` — to completion before looping; `biased` does not pre-empt an in-flight arm body).

State at a cancel in block B (future dropped while `wait_durable_through_async` is suspended):
- The frame IS retained in `lane.retry` (block A ran: `assign_and_retain` inserted into RAM, mesh.rs:1675-1688) — but that RAM lane dies with the aborted task's `lanes` map. Since a task-abort means the whole `peer_writer` is gone, the RAM retry is lost regardless — this is the crash-equivalent case, already covered by the outbox.
- The outbox row is **durable-or-becoming-durable but NOT sent** (block C `ensure_connection`/`write_reliable_frame` at mesh.rs:1765+ has not run). This is **exactly crash-window row 4** (r6d3a §5): "durable-but-not-yet-sent." Its recovery IS the R-6d3b boot replay — `scan_all` finds the row, re-sends it at the bumped incarnation. **No half-durable/half-sent state:** the submit (block A `submit_barrier` → `submit_nonblocking`, non-blocking) either handed the batch to the writer (row becomes durable, replayed) or the store was mid-Drop and the F4 guard already returned `Down` (mesh.rs:1746-1752, BEFORE block B). The redb txn is all-or-nothing (`apply_batch`, store.rs:237-271) — the row is on disk or rolled back, never torn.
- **Does this violate the cancel-safety invariant the comment states?** The invariant's REAL content is "never leave a half-written frame on a `Some` stream" (mesh.rs:1625-1626) — i.e. never a torn WIRE write. A cancel in block B is BEFORE any stream write (block C), so no stream is touched → the invariant holds. The new `.await` adds a cancel point that leaves at worst a durable-not-sent row, which is the SAFE direction (replay re-drives; never a sent-not-durable frame). **The comment must be updated** (§7) to say: "block B is now an `.await`; a cancel there is safe — the frame is retained + the row durable-or-rolled-back but NOT sent, so boot replay re-drives it (crash-row-4-equivalent); the load-bearing invariant (never a torn stream write) is preserved because block C has not run."

This is a strict improvement over the pre-F2 sync park, which could not be cancelled mid-park but blocked a worker; F2 trades an uncancellable worker-block for a cancellable-but-safe suspension.

### 1.6 The scale argument (N concurrent durable sends)

With (a): K distinct `peer_writer` tasks each hit block B for their own durable frame. Each calls `wait_durable_through_async(seq).await`, which suspends the future and **releases the worker**. The 2 workers are immediately free to poll the accept loop, the ack readers, and the other `peer_writer`s' non-durable work. When the single `vd-store-writer` thread (store.rs:441-457, OUTSIDE the tokio runtime) fsyncs the coalesced batch and `notify_waiters()`, all K futures are woken and re-check their `seq` against the monotone `last_durable` — those whose seq is covered proceed to block C, the rest re-await. **K waiters cost 0 workers.** With (b) K blocking-pool threads park; with (c) K workers park (up to the pool size, then starve). Only (a) makes the mesh I/O path immune to durable-send fan-out — the property this slice exists to guarantee before flipping the gate live.

---

## 2. F1 — the deterministic durable-before-send ordering pin (BINDING)

R-6d3a's T-DBS-1 uses `pause_on_key_prefix` on the `OUTBOX_TAG` prefix to *try* to prove ordering, but the r6d3a design itself notes the off-tick writer races `scan_all` so a block-B-deletion probe only catches ~1/8. F1 lands a **deterministic** pin.

### 2.1 The exact `store-test-hooks` test

`#[cfg(feature = "store-test-hooks")]` in `crates/io-prod/tests/` (a new `durable_before_send_pin.rs`, or extend `mesh_under_loss.rs`'s hooks section), using the EXISTING `StoreTuning.pause_on_key_prefix`/`pause_marker_path` (store.rs:84-91) — the SAME sentinel-pause the orchestrator crash proof uses (store.rs:335-347 `maybe_pause_before_fsync`). No new store hook is needed.

```
Test: durable_row_absent_from_scan_all_until_block_b_has_waited (store-test-hooks)

Setup:
  - Open a real NodeOutbox on a temp path via a StoreTuning with pause_on_key_prefix = Some(OUTBOX_TAG-prefixed
    sentinel key bytes) + pause_marker_path = Some(<tmp marker>). (NodeOutbox::open forces
    writer_channel_depth = OUTBOX_WRITER_CHANNEL_DEPTH but PASSES THROUGH the pause fields — outbox.rs:201-205.)
  - Wrap it as SharedOutbox; two-node mesh; send ONE Durability::Retained Saga frame from the sender.

Deterministic sequence (NO 1/8 race):
  1. write_frame block A retains + submit_barrier hands off the batch to the writer.
  2. The writer, before fsyncing the sentinel batch, WRITES the marker file then BLOCKS FOREVER
     (maybe_pause_before_fsync, store.rs:336-347). The row is NOT yet durable.
  3. Block B awaits wait_durable_through_async(batch_seq) — it CANNOT return (writer parked pre-fsync).
  4. ASSERT-A (writer parked): wait for the marker file to appear (bounded poll in the TEST, never in
     write_frame). Once it exists, the row is submitted-but-not-fsynced. Call ob2 = a SECOND read-only reopen
     is NOT possible (redb single-writer); instead assert via the SENDER's own outbox handle that scan_all()
     does NOT contain the row (the fsync has not happened ⇒ the committed redb has no such key) AND the
     RECEIVER has NOT received the frame (block C is gated behind block B, which is parked).
  5. UNPAUSE: unpark the writer thread (a test hook — the writer is a named std::thread; the test holds its
     handle via a store-test-hooks accessor, OR the pause loop checks an AtomicBool the test flips). The
     fsync completes ⇒ last_durable bumps ⇒ notify_waiters ⇒ block B returns ⇒ block C sends.
  6. ASSERT-B: scan_all() now CONTAINS the row AND the receiver drains the frame.

Proves: the row is NOT in scan_all until block B has waited (writer parked on the sentinel; block B blocks;
unpause ⇒ durable ⇒ sent) — a DETERMINISTIC durable-before-send ordering pin, not a 1/8 probabilistic catch.
```

**Minimal additive hook for the unpause (step 5):** the existing `maybe_pause_before_fsync` (store.rs:344-346) is `loop { std::thread::park(); }` — a park loop. To make the test deterministic (not "SIGKILL here"), the writer needs an unpark path. **Additive change, store-test-hooks only:** give `StoreTuning` a `#[cfg(feature = "store-test-hooks")] pause_release: Option<Arc<AtomicBool>>` the pause loop checks (`while !release.load(Acquire) { park_timeout(short) }`), so the test can flip it and let the fsync proceed. This is strictly test-scaffolding (feature-gated, absent from release) and keeps the SIGKILL park (`pause_release = None`) unchanged for the R-6d4 process proof. If a hook already exists to unpark the named writer thread, prefer it; otherwise this `AtomicBool` is the smallest addition.

This test is `#[cfg(feature = "store-test-hooks")]` so it runs in the R-6d4 process tier (built via `vd-bins/store-test-hooks` → `vd-io-prod/store-test-hooks`, Cargo.toml:59/13), NOT the default `just gate`.

---

## 3. BIN WIRING (shard.rs + gateway.rs)

`open_node_outbox` (bins/lib.rs:270-292) ALREADY exists: `VD_OUTBOX_PATH`-gated, guarded by `check_durable_path`, returns `Result<Option<NodeOutbox>, …>`. R-6d3b consumes it.

### 3.1 The wrap + the `spawn_mesh` argument

`spawn_mesh` takes `outbox: Option<SharedOutbox>` where `SharedOutbox = Arc<Mutex<Box<dyn OutboxSink + Send>>>` (mesh.rs:63, 788). `NodeOutbox: OutboxSink + Send`, so the wrap is:

```rust
// shard.rs + gateway.rs main, BEFORE spawn_mesh:
let outbox: Option<vd_io_prod::mesh::SharedOutbox> = vd_bins::open_node_outbox(&env)?
    .map(|ob| std::sync::Arc::new(std::sync::Mutex::new(
        Box::new(ob) as Box<dyn vd_io_prod::outbox::OutboxSink + Send>)));
```

**API sufficiency check:** `SharedOutbox` is currently a **private `type` alias** (mesh.rs:63, no `pub`). The bin must name it OR construct the `Arc<Mutex<Box<dyn OutboxSink + Send>>>` inline. **Minimal additive change: make `SharedOutbox` `pub`** (`pub type SharedOutbox = …`) and re-export `OutboxSink` (already `pub` in outbox.rs:150) so the bin can spell the box. This is io-prod-internal surface (not the frozen seam). Then the bin passes `outbox.clone()` (for the boot-replay drain in §4, the bin needs the `NodeOutbox` handle too — see §4.1: the bin drains via the `Arc` OR keeps the `NodeOutbox` before wrapping).

The `None` at `shard.rs:36` / `gateway.rs:35` is replaced by this `outbox` (which is `None` when `VD_OUTBOX_PATH` is unset — dev/test/harness — keeping the byte-identical inert path).

### 3.2 The F3 boot-check interaction

R-6d3a's F3 (spawn_mesh rejects `peer count > OUTBOX_WRITER_CHANNEL_DEPTH` when an outbox is wired) now fires with real peer books. `OUTBOX_WRITER_CHANNEL_DEPTH = 256` (outbox.rs:49). Real peer counts: a shard sees `{ORCH, GATEWAY}` = 2 (lib.rs:362); a gateway sees `{ORCH, SHARD}` + seeded clients (lib.rs:339-344) — client count bounded by the dev roster, far below 256. **Confirmed `<< 256` at every real scale**; the check is an impossible-capacity tripwire, never a live rejection. (If a future large cluster ever approaches 256 peers per node, the tuning const `OUTBOX_WRITER_CHANNEL_DEPTH` is the ONE home to raise — no magic-number spread.)

### 3.3 HR3

The SHARD strictly needs the outbox (it emits producer-less `TransientBatch` + band-exit `Despawn`). The GATEWAY opens one for **uniformity** (empty until it has such a flow) — ONE `open_node_outbox` helper, ONE `SharedOutbox` type, NO per-kind match. Clean HR3.

---

## 4. BOOT REPLAY (before `build_app` consumes the transport)

### 4.1 The exact sequence

In `shard.rs`/`gateway.rs` main, the incarnation is already resolved (`resolve_process_incarnation`, lib.rs:223) and passed to `MeshConfig::new` (shard.rs:32). The replay happens AFTER `spawn_mesh` (so `transport` exists) and BEFORE `build_app` (which MOVES `transport`, app.rs:87):

```rust
let new_incarnation = vd_bins::resolve_process_incarnation(&env)?;   // the R-6a durable monotone bump
// ... MeshConfig::new(..., new_incarnation), spawn_mesh(..., outbox.clone()) → (mut transport, _control) ...

// R-6d3b BOOT REPLAY: drain the outbox as N ordinary send_durable(Retained) calls, ASCENDING, BEFORE build_app.
if let Some(ob) = outbox.as_ref() {
    let rows = { ob.lock().unwrap_or_else(std::sync::PoisonError::into_inner).scan_all() };  // ascending
    for (key, framed) in rows {                          // framed = envelope-stripped ReliableFrame bytes
        // Decode to recover the PAYLOAD (.bytes) — the send path RE-FRAMES, so re-send the payload, NOT the frame.
        let (rf, _) = vd_wire::framing::decode_frame::<vd_io_prod::mesh::ReliableFrame>(&framed)
            .expect("outbox row is a valid framed ReliableFrame (L2 envelope contract)");
        // Ordinary above-seam send; the FRESH lane assigns seq 0,1,2.. at new_incarnation ⇒
        // classify_reliable Reset-then-Accept, gap_drop=0.
        let _ = transport.send_durable(key.peer, key.class, rf.bytes, vd_sim::io::Durability::Retained);
    }
    // STRICTLY AFTER all fresh rows are (re-)submitted-durable: sweep the prior incarnation's window.
    // HIGH-3: gc_below MUST NOT run before the fresh rows are durable, or a crash mid-replay loses both.
    // The fresh sends are durable-gated by block B (each re-mirrors at new_incarnation via the same gate),
    // so we must ensure they are ON DISK before gc. See §4.2 for the ordering enforcement.
    { let mut g = ob.lock()...; g.gc_below(new_incarnation); g.commit(); }
}
let mut node = build_app(NodeConfig { .. }, transport);   // transport MOVED here (app.rs:87)
```

### 4.2 Decode: payload-vs-frame — decisive

**Re-send the PAYLOAD (`rf.bytes`), NOT the framed bytes.** Rationale (grounded): `send_durable` → `OutFrame{bytes, ..}` → `write_frame` block A → `assign_and_retain` RE-FRAMES via `encode_frame(ReliableFrame{from, class, incarnation:<fresh>, epoch, seq:<fresh>, bytes})` (mesh.rs:3304-3309 shows the framing shape). The stored row's frame carries the OLD incarnation (`u32::MAX` epoch, old seq) — re-sending it verbatim would double-frame AND carry a stale incarnation/seq, breaking `classify_reliable`. Decoding to `rf.bytes` and re-sending the payload lets the fresh lane assign `incarnation = new_incarnation, epoch = 0, seq = 0,1,2..` — the deterministically-Accepted replay (r6d design §3.1, verified against `classify_reliable`/`prime_or_contiguous`, mesh.rs A1 ladder: bumped incarnation resets `RecvState` to hw:0 → Reset then contiguous Accept, no Gap/Dedup/Stale).

**`ReliableFrame` visibility:** it must be nameable by the bin for `decode_frame::<ReliableFrame>`. If it is currently `pub(crate)` in mesh.rs, **minimal additive change: make `ReliableFrame` `pub`** (io-prod-internal; NOT the frozen seam). Alternatively, add a `pub fn NodeOutbox::decode_payload(framed: &[u8]) -> Option<Bytes>` on `NodeOutbox` (outbox.rs) that does the `decode_frame::<ReliableFrame>().map(|(rf,_)| rf.bytes)` internally, keeping `ReliableFrame` private and giving the bin a clean, testable seam. **Prefer the `decode_payload` accessor** — it keeps `ReliableFrame` private (tighter HR1 surface) and puts the L2 decode contract in ONE place next to `scan_all`. The bin then calls `ob.decode_payload(&framed)` per row.

### 4.3 The STRICT `gc_below`-after-all-durable ordering (HIGH-3)

The rule: **write + fsync ALL fresh rows, THEN `gc_below(new_incarnation)`**. Never the reverse — a crash after gc but before the fresh rows are durable loses BOTH the old (gc'd) and the new (not-yet-durable) copy.

Enforcement in the bin sequence: the fresh `send_durable` calls each ride the durable-before-send gate (block B), so each fresh row is on disk before ITS frame is sent. But the sends are ASYNC (they enqueue `OutFrame`s onto the per-peer mpsc; the `peer_writer` tasks fsync + send later). The bin's synchronous loop returns before the writer tasks have fsynced. **So the bin must explicitly wait for the fresh rows to be durable before `gc_below`.** The cleanest ordering that respects the gate:

- **Option A (chosen): drain the replay, then `commit()` the outbox (which block-on-priors + waits durable via the SYNC `wait_durable_through`, outbox.rs:217-224) to fence, THEN `gc_below` + `commit` again.** But the replay rows become durable via the `peer_writer` tasks' block-A `submit_barrier`, NOT via a bin-side `retain` — so a bin `commit()` on an empty staged set is a no-op and does NOT fence them. This is the subtle hazard: the fresh rows are staged/submitted by the WRITER TASKS asynchronously.
- **The correct enforcement:** gate `gc_below` on `handle.wait_durable_through(high_water_seq)` where `high_water_seq` is the store seq assigned to the LAST fresh replay row. Since the replay `send_durable`s are enqueued synchronously by the bin but fsynced asynchronously by the peer_writers, the bin cannot know the fresh rows' store seqs directly. **Simplest sound ordering:** DEFER `gc_below` — do NOT gc in the bin's boot path at all; instead gc lazily on the FIRST successful fsync-after-replay. But that adds machinery. 

**Decisive resolution (grounded in the API):** the bin drives the replay sends, then calls the outbox's `commit()` AFTER a bounded settle — but the robust, deterministic answer is to make the **replay itself synchronous-durable at the bin** by NOT routing through the async send path for the durability fence. Concretely: the bin, holding the `NodeOutbox` (before wrapping into `SharedOutbox`), re-stages each fresh row directly (`retain` at `new_incarnation`) and `commit()`s (sync-durable) to establish the fresh durable window, THEN `send_durable`s the payloads (which the peer_writers re-mirror idempotently — same `(peer,class,new_incarnation,seq)` key, LWW), THEN `gc_below(new_incarnation) + commit()`. **But** the fresh seq the bin assigns must match the seq the peer_writer will assign (0,1,2.. per lane) — which the bin cannot predict without the lane. 

**Therefore the clean, decisive design:** keep replay as pure `send_durable` (MEDIUM-1, no bin-side re-mint), and make `gc_below` **incarnation-safe WITHOUT a durability race** by observing that gc only removes `incarnation < new_incarnation` rows. The fresh rows are at `new_incarnation` (never touched by gc). The OLD rows are inert (their incarnation can never be re-sent). So the ONLY loss risk is: gc deletes old rows, then a crash before the fresh rows are durable ⇒ both gone. **Enforce by making gc the LAST durable action, fenced on the fresh window:** after issuing all replay sends, the bin calls `outbox.wait_all_replayed_durable()` — a NEW small `NodeOutbox` method that waits `handle.wait_durable_through_async` (or sync) on the store's CURRENT `last_submitted` (which, after the peer_writers have submitted all fresh rows, covers them). To know all fresh rows are submitted before waiting, the bin waits until `scan_all()` at `new_incarnation` contains exactly `rows.len()` fresh rows (bounded poll), then `wait_durable_through(last_submitted)`, then `gc_below + commit`. This is deterministic and race-free.

**Recommended concrete ordering (implementable, race-free):**
1. `rows = scan_all()` (old-incarnation rows).
2. For each: `transport.send_durable(peer, class, payload, Retained)` — enqueues; peer_writers will re-mirror at `new_incarnation`.
3. Poll (bounded, in the bin) until `scan_all()` shows `rows.len()` rows at `new_incarnation` AND `handle.durable_through() >= store.last_submitted()` — i.e. every fresh row is on disk.
4. `gc_below(new_incarnation); commit()` — sweeps the old window, now provably after the fresh rows are durable.
5. `build_app(transport)`.

This keeps `gc_below` STRICTLY after all fresh rows are durable (HIGH-3). If the bounded poll times out (a stuck writer), fail loud (never gc a un-fenced window). **Flag as OPEN QUESTION #1:** whether to expose a purpose-built `NodeOutbox::replay_fence(expected_fresh: usize, new_incarnation: u64)` that encapsulates the poll+wait (cleaner, one testable home) vs. the bin orchestrating it — the former is preferred for HR3/testability.

---

## 5. Boot-replay crash-window table

(See the `crash_window_table` field for the rendered table.)

---

## 6. Tests per slice + Tier-B ≥ 90

### 6.1 F2 (io-prod, default gate — deterministic, no SIGKILL)
- **`async_block_b_wait_returns_when_durable`** (io-prod integration, `mesh_under_loss.rs` harness): real `NodeOutbox`, two-node mesh, send a `Retained` Saga frame; assert `scan_all` has the row AND the receiver drains it (the async gate fires end-to-end). Covers the new `wait_durable_through_async` happy path + block-B `.await`.
- **`async_wait_fast_returns_when_already_durable`** (store.rs unit, `#[tokio::test]`): a `DurabilityHandle` with `last_durable >= seq` ⇒ `wait_durable_through_async(seq).await` returns without registering (covers the fast-path region).
- **`async_wait_wakes_on_writer_notify`** (store.rs unit, `#[tokio::test]`): open a real `RedbStore`, spawn a task awaiting `wait_durable_through_async(seq)`, `put`+`submit_nonblocking` a batch, assert the task completes after the writer fsyncs (covers the enroll→await→wake loop region).
- **`async_wait_fails_loud_on_dead_writer`** (store.rs unit, `#[tokio::test]`, may need `store-test-hooks` to kill the writer): assert the timeout+liveness panic path (covers the liveness-escape region). If not cleanly triggerable deterministically in the default gate, gate it `store-test-hooks` and count it in the R-6d4 tier.
- **`two_peers_do_not_starve_workers`** (io-prod integration): the F2 analogue of r6d3a T-DBS-3 — with the async wait, assert that while peer A's durable send is parked in block B (writer paused via `pause_on_key_prefix`), peer B's ack path / a datagram to a third peer still makes progress (the worker was released). Observe via a bounded `tokio::time::timeout` in the TEST. This is the property F2 exists to prove.
- **MockOutboxSink** (mesh.rs:3273-3291): its `submit_barrier` returns `already_durable()` (store.rs:190), whose async wait fast-returns — the ephemeral/mock FSM tests keep passing unchanged; add an assertion that the async block-B path is a no-op for `gate == None`.

### 6.2 F1 (io-prod, store-test-hooks tier — R-6d4)
- The `durable_row_absent_from_scan_all_until_block_b_has_waited` pin (§2.1). Runs under `store-test-hooks`, NOT the default gate.

### 6.3 Bins boot-replay test (`crates/bins/tests/`)
- **`boot_replay_redelivers_outbox_rows`**: open a `NodeOutbox` on a temp path, `retain`+`commit` two `Retained` Saga rows at incarnation N; drop it; simulate boot: reopen, `scan_all`, decode payloads, assert the ascending order + that `send_durable` would be called with `(peer, class, payload)` for each (drive against a `MockTransport` or the mem tier that records sends); then `gc_below(N+1)`; assert only `< N+1` rows swept. Covers the bin drain + decode-payload + gc ordering WITHOUT real QUIC.
- **`boot_replay_gc_strictly_after_durable`**: assert `gc_below` is not issued until the fresh-window fence passes (drive the §4.3 poll with a controllable durability handle).
- **`open_node_outbox_wraps_into_shared_outbox`**: with `VD_OUTBOX_PATH` set (temp + `VD_OUTBOX_EPHEMERAL_OK=1`), the shard/gateway boot opens the outbox, wraps it, and `spawn_mesh` accepts it (extends the existing `node_outbox_boot`-style test or `dev_cluster_smoke`). Confirms the F3 boot-check passes at real peer counts.

### 6.4 How Tier-B stays ≥ 90 (justfile:39-41, region-only, no `--branch`)
The new coverable regions are: `wait_durable_through_async` (fast-path, backpressure-count, enroll/re-check loop, timeout/liveness arms); the writer's added `notify_waiters()` line; block B's `.await`; the bin wrap + drain + decode-payload + gc. Every region runs in a DETERMINISTIC in-process test above (F2 io-prod unit/integration; bins boot-replay against mem/mock — NOT real SIGKILL, which is R-6d4). The store-test-hooks F1 pin is EXCLUDED from the default-gate measurement (feature-gated), so it does not depend on flaky real-QUIC timing for coverage. New regions RATCHET the % UP; never lower `tier_b_floor` — ratchet up if the module grows. The `wait_durable_through_async` liveness-panic arm is the one region at risk of being uncovered deterministically; gate that specific test `store-test-hooks` and, if it cannot count in the default measurement, add a `#[cfg_attr(coverage_nightly, coverage(off))]` on the panic arm with a coverage-exemptions.toml note (the sync sibling's panic is exempted the same way).

---

## 7. Files / functions touched per sub-slice

### Sub-slice R-6d3b-1 (io-prod-only: F2 + F1 — make block B production-safe FIRST)
- `crates/io-prod/src/store.rs`:
  - `DurabilityHandle` (:124-136) + `RedbStore` (:204-233): add `durable_notify: Arc<tokio::sync::Notify>` field (Arc-shared).
  - `RedbStore::open` (:412-478): construct + clone the `Notify` into the writer closure, the handle, the store.
  - `run_writer` (:349-405): new `durable_notify` param; add `durable_notify.notify_waiters();` at :384 (alongside the condvar notify).
  - NEW `DurabilityHandle::wait_durable_through_async(&self, seq)` (async, §1.2). Sync `wait_durable_through` (:166-176) UNCHANGED.
  - `already_durable()` (:190-198): add the `durable_notify` field.
  - `StoreTuning` (:73-92) + `maybe_pause_before_fsync` (:335-347): add the store-test-hooks `pause_release: Option<Arc<AtomicBool>>` for the F1 deterministic unpause (§2.1).
- `crates/io-prod/src/mesh.rs`:
  - `write_frame` block B (:1759-1761): `wait_durable_through(batch_seq)` → `wait_durable_through_async(batch_seq).await`.
  - Cancel-safety comment (:1625-1627, :1758): update per §1.5 (block B is now an `.await`; cancel is crash-row-4-safe).
- `crates/io-prod/tests/`: F2 tests (§6.1) + the F1 store-test-hooks pin (§6.2, new `durable_before_send_pin.rs`).

### Sub-slice R-6d3b-2 (bin wiring + boot replay — flip it LIVE)
- `crates/io-prod/src/mesh.rs`: make `type SharedOutbox` `pub` (:63). Re-export/`pub` as needed.
- `crates/io-prod/src/outbox.rs`: NEW `NodeOutbox::decode_payload(&self, framed: &[u8]) -> Option<Bytes>` (the L2 `decode_frame::<ReliableFrame>` → `.bytes`, keeping `ReliableFrame` private); NEW `NodeOutbox::replay_fence(expected_fresh, new_incarnation)` (the §4.3 poll+wait+gc, OPEN Q#1) OR expose the pieces for the bin.
- `crates/bins/src/bin/shard.rs`: `open_node_outbox(&env)?` → wrap `SharedOutbox`; pass to `spawn_mesh` (replace `None` at :36); boot-replay drain (§4.1) before `build_app` (:46).
- `crates/bins/src/bin/gateway.rs`: same wiring (replace `None` at :35; drain before `build_app` at :37) — empty outbox, uniformity.
- `crates/bins/tests/`: the boot-replay tests (§6.3); extend `dev_cluster_smoke` for the wired path.
- `crates/bins/src/lib.rs`: `open_node_outbox` (:270-292) is REUSED as-is (no change unless `replay_fence` needs an env knob for the poll timeout — a named const, ONE home).

**Slice ordering justification:** R-6d3b-1 (io-prod-only) makes block B production-safe (async wait) and lands the deterministic pin — provable entirely at the io-prod seam with a direct-injected `NodeOutbox`, no bin, no boot. It can merge and ratchet Tier-B independently. R-6d3b-2 then flips the gate LIVE by opening the store in the bins + boot-replaying — the risky "changes prod boot" step lands as a unit WITH its bins tests, exactly as R-6d3a §7 argued (opening a live outbox without replay would leave rows unread). This is the "make it safe, then flip it" split the prompt anticipates, and it matches the R-6d3a/R-6d layering.

---

## 8. HR / seam posture
- **Frozen seam UNTOUCHED:** `wait_durable_through_async`, `durable_notify`, `SharedOutbox`, `decode_payload` are io-prod-internal; `sim::io::Transport::send_durable` (the replay's send) is the EXISTING method. `Store`/`Transport`/`Inbound` traits unchanged. `vd-wire` framing unchanged (replay uses the existing `decode_frame`).
- **No magic numbers:** `OUTBOX_WRITER_CHANNEL_DEPTH` (outbox.rs:49) is the ONE home for the F3 depth; `WRITER_WAIT_POLL` (store.rs) is the ONE home for the async timeout (reused, not duplicated); any replay-fence poll bound is a NEW named const with ONE home.
- **HR3:** ONE `spawn_mesh`, ONE `open_node_outbox`, ONE `SharedOutbox`, ONE gate — no shard-kind match. Shard + gateway wire identically.
- **HR5:** async method regions covered deterministically (§6.4); the one liveness-panic arm exempted like its sync sibling.

---
# R-6d3b-2 CORRECTIONS (implementer + opus verifier, wf-free verification — TWO CONFIRMED HOLES in §2)

Before implementing R-6d3b-2 the vetted §2 `replay_all` was adversarially verified and found to have TWO holes:

## HOLE 1 — DEADLOCK (CONFIRMED, certain). §2's `shared.lock().replay_all(&mut self, transport, ...)`
holds the shared `std::sync::Mutex` across its send+fence. But `send_durable(Retained)` enqueues to a
peer_writer whose block A re-`.lock()`s the SAME mutex to re-mirror (mesh.rs:1666-1668, unconditional for a
Retained/Reliable frame) — blocking its tokio worker. The main thread's fence then waits forever for a
durability bump only the blocked peer_writer can produce. Two blocked peer_writers also exhaust
worker_threads(2). CERTAIN deadlock the instant the first replayed row reaches its peer_writer.

## HOLE 2 — FENCE PREMATURE-GC DATA LOSS (CONFIRMED). §2's high-water fence
`wait_durable_through(last_submitted())` reads `last_submitted` ONCE — but `last_submitted` bumps only when
a peer_writer runs block A (store.rs submit_nonblocking), which is ASYNC and lags the synchronous main
thread. If only k of N peer_writers have reached block A when the fence reads, it waits for k, passes, and
`gc_below` sweeps the OLD rows whose N−k fresh re-mirrors haven't landed = D-6 #1 loss. The
poll-until-quiescent variant is ALSO unsound (a slow peer_writer gives a false quiescent reading).

## THE CORRECTED DESIGN (deadlock-free + no premature-gc):
`replay_outbox(shared: &SharedOutbox, transport: &mut dyn Transport, peers: &BTreeMap<NodeId,SocketAddr>,
new_incarnation: u64) -> Result<usize, ReplayError>` — a FREE fn in io-prod (NOT a `&mut self` method under
the lock), lock-scoped:
1. `let (rows, dh, base) = { let g = shared.lock()?; (g.scan_all(), g.durability(), g.durability().last_submitted()) };`
   — ONE brief lock: snapshot rows + a DurabilityHandle CLONE + `base = last_submitted`. RELEASE.
   ⇒ needs `fn durability(&self) -> DurabilityHandle` on the `OutboxSink` trait (NodeOutbox returns its
   clone; MockOutboxSink returns `already_durable()`). (The verifier's "option b use self.durability" is
   impossible here — the outbox is a `Box<dyn OutboxSink>` behind the Arc; the trait accessor is required.)
2. NO LOCK: for each row, `decode_value_payload(framed)` (decode_frame::<ReliableFrame>().bytes — the PAYLOAD,
   io-prod-internal, finding A) → pre-send `peers.contains(&key.peer)` else `Err(Unroutable{peer})` (finding
   B) → `send_durable_with_retry` (QueueFull ⇒ bounded retry `REPLAY_SEND_MAX_RETRIES`/`REPLAY_SEND_RETRY_
   BACKOFF` then `Err(LaneStuck)`; NEVER `let _ =`-swallow). The peer_writers freely `.lock()` to re-mirror
   (no deadlock — the main thread holds no lock here). N = rows.len() (a previously-retained row re-frames
   IDENTICALLY, so NONE shed Unframable ⇒ N is the EXACT submit count).
3. NO LOCK — COUNT-ANCHORED FENCE (fixes HOLE 2): wait `dh.last_submitted() >= base + N` (each replayed send
   = exactly one submit ⇒ +1 to last_submitted; at boot pre-build_app the replay sends are the ONLY submits),
   bounded by a fail-loud total deadline (a stuck-never-scheduled peer_writer ⇒ `Err`, leaving old rows
   un-gc'd = SAFE), THEN `dh.wait_durable_through(base + N)` (the writer fsyncs in seq order ⇒ base+N durable
   ⇒ ALL N fresh rows durable). Only THEN is gc safe.
4. ONE brief lock (atomic): `{ let mut g = shared.lock()?; g.gc_below(new_incarnation); g.commit(); }` — sweep
   the strictly-lower prior incarnation + fsync in ONE guard (HIGH-3 gc-strictly-after-fresh-durable).

Findings A/B/C hold; D's DIRECTION holds (not a scan_all count) but its high-water replacement is HOLE 2 →
use the count fence. `gc_below`+`commit` stay in ONE lock acquisition. `resolve_process_incarnation` is
called ONCE in main (finding C), passed as `new_incarnation`.
