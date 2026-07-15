I have all the evidence needed. The justfile confirms the named-recipe pattern (`spike2a`, `render-smoke`, `orch-crash` are all NOT in `gate` as expected — actually they ARE in `gate` on line 124, but they're documented as slow/GPU/release). Let me note that carefully: the design says "NOT added to gate like render-smoke's GPU requirement" — but render-smoke IS in gate. The real distinction is client-load IS in the gate. So the design's framing of "NOT in default gate" needs precision: `mesh-load` should be a genuinely separate soak recipe (not in `gate`), which is what the design intends, but the analogy to render-smoke is imperfect (render-smoke is in gate). This is a minor prose fix.

I now have everything. Let me produce the final vetted design.

---

# R-4e — FINAL VETTED, IMPLEMENT-READY DESIGN

I read the actual code (`mesh.rs` in full, `mesh_redelivery.rs`, `sim/io/mod.rs`, `saga.rs`, `node/src/saga_runtime.rs`, `DEFERRED.md` 1020-1256, `justfile`). Every reviewer finding is adjudicated against ground truth below. **Verdict: FIX_BEFORE_IMPL was correct for the two independent reviewers; all real findings are now folded.** The design is sound after these corrections.

## Adjudication of the three reviews (against code)

**Reviewer 1 (H2 correctness lens) — all four findings UPHELD:**
- **HIGH #1 (cross-stream race rationale is FALSE) — UPHELD, code-confirmed.** `serve_connection` (mesh.rs:833-844) spawns a NEW `serve_data_stream` per `accept_uni`; `handle_connection_drop`/redial bumps the epoch and the sender opens a fresh stream (on_write_error mesh.rs:593-596). Two `serve_data_stream` tasks for the same `(peer,class)` genuinely overlap across a redial — quinn's close is async, nothing barriers old-reader-exit before new-reader-start. The `StaleEpoch`-before-hw verdict (classify_reliable A3, mesh.rs:363-365) is the entire cure, and it is safe ONLY because both readers serialize on ONE lock over that `RecvState`. The design's stated justification ("serial on the wire anyway") is wrong and dangerous. Per-peer keying preserves safety (both readers take the same peer's inner Mutex), so the fix is a rationale rewrite + a redial-under-load test, NOT a structural change.
- **HIGH #2 (ack_egress single-Arc-fetch is a latent regression) — UPHELD, code-confirmed.** Current `ack_egress` (mesh.rs:996-1006) iterates `acked_keys` and does `led.get(&(peer,class))` PER key — correct for any NodeId set. `acked_keys` is populated purely from `frame.from` (mesh.rs:892), a trust invariant, not a structural one. Collapsing to "fetch that one peer's Arc" silently assumes one distinct NodeId. Keep per-key lookup (grouped-by-NodeId is a fine optimization; assuming-one is not).
- **MEDIUM #3 (or_insert seed must stay frame-valued) — UPHELD, code-confirmed.** mesh.rs:912-917 seeds a fresh entry with `incarnation: frame.incarnation, epoch: frame.epoch`. Only the OUTER `or_insert_with` (peer → empty inner map) is new; the inner seed is byte-identical to today.
- **LOW #4 (SharedInbox is the next node-wide serialization point + lock-order) — UPHELD, code-confirmed.** `push_inbox` locks the one node-wide `Arc<Mutex<BoundedInbox>>` (mesh.rs:199, 269-272) on every reliable delivery, nested under the ledger lock. Must be named as the residual + the lock order (ledger-inner → inbox, never reverse; ack_egress takes ledger-never-inbox) stated.

**Reviewer 2 (anti-flake/feasibility lens) — CRITICAL + both HIGHs UPHELD:**
- **CRITICAL #1 (`inbound_dropped_reliable==0` is a drain-vs-fill RACE, not EXACT) — UPHELD, code-confirmed and decisive.** `BoundedInbox::push` (io/mod.rs:216-242) drops reliable the instant `len >= capacity` with no unreliable to evict; default `inbound_capacity_for(256)=2048` (mesh.rs:166-169) vs `64×200=12,800` fan-in, with the sim thread draining only every 2ms (mesh_redelivery.rs:78). The assertion WILL flake. Fix: size the receiver inbox from the LoadShape so it is structurally un-overflowable.
- **HIGH #2 (`reliable_acked` per-sender wait + deadline must scale with N) — UPHELD.** Sum-wait can hang on one lagging sender; the 50ms idle-flush (mesh.rs:87) tail scales with N. Wait per-sender; derive the deadline from the shape.
- **HIGH #3 (correlated-outage inbox race amplified) — UPHELD.** Same sizing discipline; primary assertion = survivor no-loss on a sized inbox.
- **MEDIUM #4/#5, LOW #6/#7 — UPHELD** (uni-stream cap is per-conn not the bottleneck; LoadShape homes all consts + deadline formula; lock-order invariant executable; per-`from` u64-payload grouping).

**Reviewer 3 (scope/R-5 lens) — SOUND_TO_IMPLEMENT with two HIGH corrections UPHELD:**
- **HIGH #1 (R-5/D-6 flip wording under-states the residual) — UPHELD, code-confirmed and important.** `(BatchHandoff, SourceUnreachable)` at saga.rs:915-918 DOES self-promote the dest; `rehome_event_for`'s `BatchHandoff` arm (saga_runtime.rs:1270-1273) emits `SourceUnreachable` when `is_confirmed_dead(ctx.source)`. So AwaitAdopt source-crash is NOT "no recovery" — it has a saga resolution, but its TRIGGER (`is_confirmed_dead` via `record_unreachable` from `NodeUnreachable` bounces) is gated on M3 (durable incarnation, DEFERRED.md:1201-1203) and L5 (addr-reread). The DEFERRED edit must say this precisely.
- **HIGH #2 (correlated-outage MUST be N=8 all-pairs, not fan-in) — UPHELD, code-confirmed.** `NodeUnreachable` pushes to the LOCAL sender's inbox (mesh.rs:1301-1309 via push_inbox → w.inbox). In pure fan-in the receiver has no outbound dead lanes → emits zero NodeUnreachable → the property is untestable. The confirm-dead-eviction concern is sender-side; the test must hit a node that both sends and receives.
- **MEDIUM/LOW (GhostReliable≡Saga lane FSM; L5 guard narrow-assert; conn_died flake budget) — UPHELD.** io/mod.rs:84 confirms GhostReliable and Saga are both Reliable → identical FSM; R-5 is a semantic capstone, not a new code path.

One correction to a reviewer analogy: `render-smoke`, `spike2a`, `orch-crash` are all IN the `gate` recipe (justfile:124). `mesh-load` is genuinely a NON-gate soak recipe — the correct analogy is that it is documented-and-standalone like those, but it must NOT be added to the `gate` line. Prose adjusted.

---

## (a) Resolved sub-slicing — ordered, each independently gate-able + commit-able

Five sub-slices, de-risking order **measure → fix → prove → cover → capstone** (DEFERRED.md:1191 "measure under the load test; do NOT rework before P3-load"):

- **R-4e1 — the N-peer load HARNESS + baseline + the `just mesh-load` recipe.** New `crates/io-prod/tests/mesh_load.rs`; the `LoadShape` config; the sized inbox; the completion-gated exact assertions + generous progress deadline; the `mesh-load` recipe (a load test with no way to run it is incomplete). Commits with H2 UNCHANGED (baseline passes correctness with the single lock — the H2 fix is a scaling/structure improvement, not a correctness fix, per DEFERRED.md:1189). This slice empirically surfaces the H2 contention and becomes the regression harness.
- **R-4e2 — the H2 per-peer RecvLedger re-key.** The `mesh.rs` data-structure change (:315/:903/:963). Gated by re-running R-4e1 green + `mesh_redelivery.rs` + Tier-B ≥ 90. Commits alone.
- **R-4e3 — correlated-outage (N=8 all-pairs) + pod-reschedule RED guard + conn_died coverage.** Three targeted tests sharing the harness helpers; a cure lands in the same commit as its red test if forced. Splittable if a cure is fat.
- **R-4e4 — R-5 `mesh_under_loss.rs` producer-less capstone + the D-6 #1 partial-flip DEFERRED edit + the seam doc-prose update.** De-risked last. Carries the deploy-gate prose for `mesh-load`.
- **R-4e5 — folded:** the recipe rides R-4e1; the "named soak precondition" prose rides R-4e4. Net **4 commits.** Commit policy per MEMORY: **ask each time.**

---

## (b) The H2 re-key — data structure + the exact adapted invariants (std-only, NO new dep)

**Pivotal constraint (code-confirmed):** a peer's NodeId is NOT known at accept time — `serve_connection` (mesh.rs:806) authenticates the cluster, not the node; the NodeId is `frame.from`, learned per-frame in `classify_and_deliver` (mesh.rs:912). So the lock cannot be keyed at spawn; it is discovered on the first frame.

**Chosen structure (std-only — `RwLock` + `Mutex` + `Arc`, no dashmap/flurry):**
```rust
type RecvLedger = Arc<RwLock<BTreeMap<NodeId, Arc<Mutex<BTreeMap<MsgClass, RecvState>>>>>>;
```
Outer `RwLock`: shared-read on the hot path to fetch a peer's inner Arc; write-locked only on the rare first-frame-from-a-new-peer insert. Inner per-peer `Mutex`: each peer's classify serializes only against another frame from the SAME peer, never peer Q — the RX twin of the per-peer SEND lanes (mesh.rs:6-8).

**`classify_and_deliver` new shape:**
1. read-lock outer, `.get(&frame.from).cloned()` → `Option<Arc<Mutex<..>>>`, drop the read guard. Existing peer (steady state): shared read, zero contention across peers.
2. `None` (rare first frame): write-lock outer, `entry(from).or_insert_with(|| Arc::new(Mutex::new(BTreeMap::new())))`, clone, drop.
3. lock the peer's inner Mutex; `entry(frame.class).or_insert(RecvState { incarnation: frame.incarnation, epoch: frame.epoch, hw: 0, primed: false })` — **the inner seed is byte-identical to mesh.rs:912-917 (finding 1-MED); only the OUTER `or_insert_with` is the new allocation.** Run `classify_reliable` UNCHANGED; on Accept/Reset **hold the inner lock across `push_inbox`** and keep the `*st = before` rollback.

**THE exact invariant that must not break (rewritten rationale — finding 1-HIGH):** classify + push_inbox + rollback are ONE critical section over that `(peer,class)` `RecvState`. This is required NOT because same-(peer,class) frames are wire-serial (they are NOT — two `serve_data_stream` tasks for the same (peer,class) overlap ACROSS A REDIAL, old-epoch straggler vs new-epoch stream). The inner lock is what serializes those two concurrent readers so the `StaleEpoch`-before-hw verdict (mesh.rs:363-365) drops the straggler before it touches hw, AND so the reliable-inbox-drop rollback (`*st = before`, mesh.rs:930-933) is atomic against a concurrent same-peer frame. **Dropping the lock before push_inbox would re-open the reverted-R-1 cross-stream corruption** — the DEFERRED.md:1191 "drop before push_inbox" sketch was written against the node-wide lock and is REJECTED for the per-peer design (holding the per-peer inner lock is both correct and contention-free, since same-(peer,class) already serializes and cross-peer no longer shares a lock).

**`ack_egress` (mesh.rs:963-1024) — finding 2-HIGH:** do NOT collapse to a single-peer Arc fetch. Keep the per-key iteration: read-lock outer once, and per `(peer,class)` in `acked_keys` fetch that peer's inner Arc and read `(incarnation, epoch, hw)` — OR group keys by NodeId and take each distinct peer's inner lock once (a safe optimization). This preserves the current robustness for any NodeId set (`acked_keys` is `frame.from`-populated, a trust not a structural invariant). Same "copy out, DROP before the write" discipline (mesh.rs:1019 write stays outside all locks — cancel-safety).

**Lock-order invariant (finding 4-LOW, must be stated + preserved):** the ONLY order is outer-read → (drop) → inner-Mutex → (optionally) inbox-Mutex. `acked_keys` is acquired and released DISJOINT from any inner-Mutex hold — never nested inside it (serve_data_stream takes acked_keys AFTER classify_and_deliver returns, mesh.rs:889; ack_egress takes acked_keys then outer-read then inner — inner is never held across acked_keys). `ack_egress` takes ledger, NEVER inbox. So the global order is consistent and deadlock-free; the design must not introduce a reverse. **All three lock types use `unwrap_or_else(PoisonError::into_inner)`** (mesh.rs:270) — a poisoned inner peer lock wedges only that peer; a poisoned outer must be into_inner (node-wide).

**Residual named (finding 4-LOW):** after this re-key the `SharedInbox` node-wide Mutex (mesh.rs:199) is the NEXT node-wide RX serialization point (held briefly per reliable delivery, nested under the per-peer inner lock). The re-key relieves the LEDGER contention; it does not touch the inbox mutex. **Ledger a NEW DEFERRED item**: per-peer inbox partition / lock-free MPSC drain is a future scaling slice — do NOT claim the ledger re-key alone delivers full RX-plane isolation.

**Coverage (HR5, Tier-B ≥ 90):** the outer read-then-maybe-write `Option` match — both arms hit by `mesh_redelivery.rs` (first frame = write arm, subsequent = read arm) + the load test. `classify_reliable`/`on_ack` pure unit tests unchanged. The two lock types live only in concrete `classify_and_deliver`/`ack_egress` (no generic fn → no per-monomorphization gotcha).

---

## (c) The robust, non-flaky assertion set + max feasible N (R-4e1)

**File** `crates/io-prod/tests/mesh_load.rs`. Reuses the `cluster()`/`reserve()` UDP-port-reservation pattern (mesh.rs:2341, proven at N=4 in `mesh_volume_all_pairs_burst` mesh.rs:2599).

**Topology — fan-in:** N senders (`NodeId(2..=N+1)`) → ONE receiver (`NodeId(1)`); receiver dials nobody. This is the RX-plane-collapse geometry H2 predicts (every sender's reliable RX serializes on the receiver's one ledger mutex today), N connections + N DATA streams into node 1, not N².

**One runtime, worker_threads = a small fixed named const (justified):** a bounded pool deliberately provokes H2 contention rather than eliminating it (a dev host has ~8-16 cores; the value is a `LoadShape`/const with a one-line justification, not inline).

**`LoadShape` config (ALL consts homed — finding 5-MED):**
```rust
struct LoadShape {
    senders: usize,          // N: env VD_MESH_LOAD_NODES or 16 (floor); soak sets 64
    frames_per_sender: usize,// sustained depth (200, matching mesh_volume BURST)
    payload_len: usize,      // small (lane-count-bound, not bandwidth-bound)
    worker_threads: usize,   // bounded pool to PROVOKE H2 contention
    // deadline is DERIVED, not a bare const:
    deadline_base: Duration, // fixed floor
    per_sender_budget: Duration, // × senders → scales with N and the ack idle-flush tail
}
// receiver inbox capacity = senders * frames_per_sender + headroom  (finding 1-CRIT: structural)
// deadline = deadline_base + per_sender_budget * senders  (finding 2-HIGH: PROGRESS ceiling)
```

**The sized inbox (finding 1-CRITICAL — the flake cure):** the receiver's `MeshConfig.inbound_capacity` is set to `senders * frames_per_sender + headroom` (NOT the default 2048), so even if the sim thread never drained until the end, no reliable frame can EVER be dropped. This converts `inbound_dropped_reliable == 0` from a drain-vs-fill race into a structural invariant.

**Assertions — EXACT (completion-gated) vs GENEROUS (the only wall-clock):**

| Assertion | Kind | Why it can't flake |
|---|---|---|
| Group deliveries into `BTreeMap<NodeId, BTreeSet<u64>>` keyed by `Inbound::Wire.from`; each sender's set == `(0..frames_per_sender)` | **EXACT, per-lane** (finding 7-LOW) | `wait_until` count reaches `N × frames_per_sender`, THEN assert per-sender contiguity. Per-`from` grouping closes the cancelling-loss+dup gap a global multiset would miss. New drain helper reads the FULL u64 payload (not `bytes[0]`). |
| per-sender `wait_until(senders.iter().all(\|s\| s.stats().reliable_acked >= frames_per_sender))`, then each `== frames_per_sender` | **EXACT, per-sender** (finding 2-HIGH) | Per-sender wait (not sum) — a lagging sender trips the deadline, never a false pass. Monotone+clamped `on_ack` (mesh.rs:526-540). |
| receiver `gap_drop == 0` and `stale_epoch_drop == 0` | **EXACT**, post-completion | Blip-free run keeps gap genuinely 0 (mesh.rs:333); asserted after the completion wait. |
| receiver `inbound_dropped_reliable == 0` | **EXACT** (structural, via the sized inbox) | With the inbox sized to the full backlog, a reliable drop is now impossible for the bounded workload → a true invariant, no longer a race. |
| all frames delivered `started.elapsed() < deadline` | **GENEROUS ceiling** | The ONLY wall-clock assertion; a derived PROGRESS ceiling scaling with N. A collapse manifests as never reaching the count → deadline trip (a correct failure, not a flake). |

The send loop tolerates back-pressure with the retry-on-QueueFull idiom (mesh_redelivery.rs:65). Senders live as a `Vec<MeshTransport>` on the test thread, round-robin `send_reliable`; payload = seq as `u64::to_le_bytes` (frames_per_sender may exceed 255, so unique per-frame ids across the window).

**Max feasible N (finding 4-MED, ranked-uncertain per (f)):** `max_concurrent_uni_streams(256)` is PER-CONNECTION (mesh.rs:713) — not the fan-in bottleneck. `max_inbound_connections` default 256 (mesh.rs:188) holds 64. Task count ~400 (64×~3 receiver + 64×~2 sender) on the bounded pool — fine for tokio; the single ledger mutex serializing 64 readers IS the point. FDs: 65 UDP sockets vs macOS `ulimit -n` 256 — fits. **Env default `VD_MESH_LOAD_NODES` unset → 16 (always-feasible audit floor); the recipe pins the soak N. GATING CRITERION: validate the pinned soak N is green 5 consecutive runs before pinning it in the recipe; if 64 is not reproducibly green, pin 32 (32-peer fan-in still proves the H2 property) and record the host ceiling in DEFERRED.** I have NOT measured 64 here — R-4e1 must confirm empirically before committing the recipe constant.

---

## (d) Correlated-outage, pod-reschedule (L5), conn_died (R-4e3)

**Item 3 — correlated outage: N=8 ALL-PAIRS (THE topology, finding 2-HIGH — not fan-in).** 8 nodes, all-pairs (56 lanes), all delivering steadily; kill 4 simultaneously; keep driving the 4 survivors. On a SURVIVING node assert (PRIMARY, exact, completion-gated) its inbound from the 3 other survivors is loss-free, AND (secondary, on a correctly-SIZED inbox per finding 3-HIGH) `inbound_dropped_reliable == 0` despite its 4 dead lanes each bouncing `NodeUnreachable` into its own inbox (mesh.rs:1301-1309). Do NOT conflate no-loss with an inbox-stress test — size the survivor inbox from the outage LoadShape. **Expected: PASSES with the existing backoff-paced bounce** (the R-4a cadence claim, mesh.rs:1287 — the re-bounce cadence is the geometric backoff, can't out-produce the inbox drain). The test converts that argument into evidence. **If it reds** (eviction observed): the cure is the coalesce — a `last_bounced_unreachable: bool` on `ReliableLaneSender` (mesh.rs:439), set on bounce in `confirm_and_maybe_bounce`, cleared in `on_replay_ok`, so at most one `NodeUnreachable` per dead-episode. Lands in the same commit as the red test. The priority-lane alternative is heavier, deferred unless coalesce is insufficient.

**Item 4 — pod-reschedule / address change (L5): a `#[ignore]`d RED GUARD (scoped OUT of R-4e).** `peer_writer` captures `addr` once at spawn (mesh.rs:1031), threaded to `ensure_connection`'s `connect(addr,..)` (mesh.rs:1333-1334) — a NodeId rescheduled to a new addr is dialed stale forever. The fix needs a live address source (CA-1/orchestrator provisioning, DEFERRED.md:1211-1213), which is M3/provisioning scope the audit did not ask R-4e to build. Deliverable: an executable `#[ignore = "L5: peer_writer captures addr once; needs addr re-plumb (CA-1/provisioning)"]` guard mirroring `ca1_reply_on_connection` (mesh.rs:2568-2581). **Assert the DEFECT MECHANISM NARROWLY (finding L5-LOW):** A's sends to B after the move bounce `NodeUnreachable` (the fixed-addr re-dial fails) — NOT a broad "B receives nothing" — with a doc-comment naming the exact line to flip (`ensure_connection`'s addr source, mesh.rs:1333) and a cross-ref to DEFERRED.md:1211-1213, so it doesn't rot when provisioning lands.

**Item 5 — `replay_lanes` conn_died-mid-pass (R-4a Tier-B LOW, mesh.rs:1435-1440).** The dial-ok-then-write-fails arm (`Err(())` → `conn_died = true`) is Tier-B-uncovered. Attempt a deterministic multi-lane-blip test (two owing lanes; blip mid-pass so lane 1 writes OK, lane 2 hits the dead conn) in a bounded loop, asserting eventual exactly-once. **Flake budget (finding conn_died-LOW):** gate on a bounded internal retry-loop that either exercises the arm within a deadline or SKIPS with a logged reason (never fails) — not "3 clean runs." If it can't close cleanly, ledger it in `coverage-exemptions.toml` with the exact mesh.rs:1435-1440 region + the wf_b1d0610c LOW cross-ref (io-prod is Tier-B floor 90, at 93.44%; this is defense-in-depth). Time-box; do not over-invest.

---

## (e) R-5 `mesh_under_loss.rs` — the precise claim (R-4e4)

**File** `crates/io-prod/tests/mesh_under_loss.rs`. Node A sends ONE `MsgClass::GhostReliable` frame carrying a real postcard-encoded `GhostFlow::Despawn`-shaped payload (the producer-less one-shot, DEFERRED.md:1244-1256), A's connection blips (`drop_connections`), A goes idle. Assert B receives the Despawn exactly once — delivered purely by A's R-4a retransmit timer re-driving the owing lane off A's own clock, no application re-drive. Sequencing copies the proven mesh_redelivery.rs:291-304 pattern (drop → sleep 150ms → the one send → idle → wait for delivery).

**Rationale (finding GhostReliable≡Saga-MED, code-confirmed):** `GhostReliable` and `Saga` are both `Reliability::Reliable` (io/mod.rs:84) and ride the IDENTICAL lane FSM. R-5's contribution is the SEMANTIC proof (a genuinely producer-less flow — a ghost Despawn has no saga, no `scan_deadlines`, DEFERRED.md:1251 — reaching at-least-once with no re-driver but the transport), NOT a new transport code path. It is the capstone that retires the D-6 #1 argument. Using a real `GhostFlow::Despawn` envelope also proves the end-goal envelope round-trips.

**Assertions:** (1) B gets exactly one `Inbound::Wire{class: GhostReliable}` with the Despawn payload (completion-gated, then count == 1); (2) `ctl_a.stats().reliable_acked == 1`; (3) `ctl_b.stats().gap_drop == 0`; (4) A's drained inbound has ZERO `NodeUnreachable` (blip ≠ death — the first re-dial recovers before the threshold); (5) `ctl_a.local_addr().is_ok()` (endpoint survived).

### Does flipping D-6 #1 green require the AwaitAdopt egress cure first? — PRECISE ANSWER (finding R-5-HIGH, code-corrected)

**No — R-5 proves the transport-LAYER at-least-once independent of the saga-layer AwaitAdopt gap; and the AwaitAdopt source-crash residual is NOT "no recovery" — it HAS a saga resolution, gated on M3+L5.** The corrected decomposition:

- **What R-5 CAN prove now:** the `MeshTransport` delivers a producer-less reliable one-shot at-least-once across a blip for a lane WHOSE SENDER STAYS UP (idle-after-blip re-drive via the R-4a timer + retained window + receiver contiguity ledger). This is real and complete for the GhostReliable Despawn — "it rides GhostReliable, so the root fix subsumes it" (DEFERRED.md:1251). It also subsumes the EmitCrossing (DEFERRED.md:1230) and the D-37 re-home (cured in 2d) for the sender-stays-up case.
- **What R-5 does NOT prove — and the residual, stated PRECISELY (code-confirmed at saga.rs:915 + saga_runtime.rs:1270-1273):** `BatchHandoff::AwaitAdopt` emits no egress (saga.rs:816-822). If the SOURCE crashes after emitting the `TransientBatch` but before the dest adopts, the transport's at-least-once does NOT survive the source restart (the retry buffer is RAM, dies with the process — io/mod.rs:311-327 "WITHIN A RECEIVER INCARNATION" / "sender stays up"). **HOWEVER** the saga is NOT stranded with no recovery: `(BatchHandoff, SourceUnreachable)` self-promotes the dest (saga.rs:915-918), and `rehome_event_for` emits `SourceUnreachable` once `is_confirmed_dead(ctx.source)` (saga_runtime.rs:1271-1273). The residual is therefore precisely: **that resolution's TRIGGER depends on the transport confirming the crashed source dead via `NodeUnreachable` → `record_unreachable`, which is gated on (i) M3 (durable monotone incarnation — DEFERRED.md:1201-1203, or a sub-second CrashLoop restart at equal/lower incarnation silent-Dedups/StaleDrops), AND (ii) L5 (addr-reread — a source rescheduled to a new addr is never re-dialed, so its death is never confirmed).**

**Therefore the D-6 #1 DEFERRED edit R-5 lands is a PARTIAL flip, worded surgically (this is the deliverable a reviewer must be able to attack precisely):**
- Transport-layer redelivery entries (RX ledger, per-lane replay, idle-after-blip timer, and the producer-less flows whose SOURCE stays up — GhostReliable Despawn, EmitCrossing, D-37 re-home) → **🟩 GREEN**, "proven by R-5 `mesh_under_loss.rs`."
- The `AwaitAdopt` source-crash residual stays **🟥/🟨** with the sharpened note: "the saga HAS a source-crash resolution (saga.rs:915 self-promote) — this is NOT producer-less-with-no-recovery. Its TRIGGER (`E::SourceUnreachable` via `is_confirmed_dead`) is gated on M3 (durable monotone incarnation, R-6) AND L5 (addr-reread for a rescheduled source) — both already-tracked HARD deploy preconditions. R-5 proves the transport at-least-once for a source-that-stays-up; the source-crash case is covered by the saga self-promote once M3+L5 land. Optionally the 2d re-solicit-egress template (an AwaitAdopt Timeout re-prompting the source to re-emit TransientBatch) or the R-6 durable outbox would make it self-sufficient. NONE of these block R-5's transport-layer proof."

**End-goal scope precision (finding composition-MED):** the DEFERRED/commit prose must say: "R-4e proves the MeshTransport RX and TX planes sustain N-peer fan-in without an RX-plane collapse (the H2 cure) — the transport substrate for hundreds-in-one-location. The directory single-writer CAS throughput and the gateway admission/snapshot-fan-out are the NEXT scale gates (above the seam, out of R-4e scope) and must be load-proven separately before a real hundreds-player soak." Do NOT claim R-4e proves hundreds-in-one-location.

---

## (f) The `just mesh-load` recipe + gating

```makefile
# R-4e (L7): the N-peer real-QUIC load/soak gate — sustained reliable fan-in into ONE receiver at
# VD_MESH_LOAD_NODES peers (default 16 floor; soak sets the pinned N) proving no-loss/no-dup, reliable_acked
# keeps pace, gap_drop==0, inbound_dropped_reliable==0 (a SIZED inbox makes it structural), and NO RX-plane
# collapse (the H2 per-peer-ledger property) under a generous PROGRESS deadline. SLOW + resource-heavy (N
# real quinn endpoints). NOT in the default `gate` (unlike render-smoke/spike2a/orch-crash which ARE gate
# steps): load tests are slow ⇒ a standalone BLOCKING soak/deploy precondition, run before any real deploy.
# `mesh_under_loss` is the R-5 producer-less capstone.
mesh-load:
    VD_MESH_LOAD_NODES={{pinned_soak_n}} cargo test -p vd-io-prod --test mesh_load -- --nocapture --test-threads=1
    cargo test -p vd-io-prod --test mesh_under_loss -- --nocapture
```
`--test-threads=1` for `mesh_load` so the multi-endpoint runs don't contend for ports/cores (same discipline as `orch-crash`). **`{{pinned_soak_n}}` = 64 iff green 5/5 in R-4e1, else 32.** NOT added to the `gate` line (justfile:124) — the load test is slow; documented standalone. The DEFERRED prose (landed in R-4e4) names `just mesh-load` a blocking soak/deploy precondition. At the default unset (16), a bare `cargo test -p vd-io-prod --test mesh_load` stays feasible for the inner loop.

---

## (g) Risks / unknowns, ranked (least-sure first)

1. **The pinned soak N feasibility (top unknown).** 64 quinn endpoints + internal FDs vs macOS `ulimit -n` 256 should fit but is untested here. Mitigation: env default 16, the recipe pins only after 5/5 green, fall back to 32. Not yet measured — R-4e1 confirms empirically before the recipe constant is committed.
2. **Whether H2 contention is OBSERVABLE at 16-64 on localhost.** The single node-wide mutex hold is microseconds on fast localhost; the load test may show NO throughput delta before vs after the re-key. That is FINE — the re-key's justification is STRUCTURAL (the mesh.rs:6-8 RX-isolation promise + hundreds-in-one-location), and the load test's job is the correctness invariants (no-loss/gap_drop==0/acks-keep-pace/inbound_dropped_reliable==0 on the sized inbox), NOT a measured speedup. Do NOT gate on a delta. The residual SharedInbox mutex (finding 4-LOW) is the next node-wide point — named as a future slice.
3. **The correlated-outage expected outcome.** Predict PASS with the existing backoff-paced bounce; if eviction appears on the N=8 all-pairs kill, the coalesce cure is small but its sufficiency-vs-priority-lane is unproven. Medium confidence; the test converts an argument to evidence either way.
4. **The conn_died-mid-pass deterministic trigger.** Genuinely narrow over real quinn; bounded-loop-or-skip plan + Tier-B-exemption fallback. Lowest stakes (Tier-B floor already met). Time-box.
5. **The D-6 #1 partial-flip wording.** A judgment call about what "green" means; the corrected framing (transport green; the AwaitAdopt source-crash residual is saga-recovered via self-promote, gated on M3+L5) is now precise and code-cited (saga.rs:915, saga_runtime.rs:1271). The DEFERRED edit must be surgical enough that a reviewer sees exactly which claim is green and which is owed.

---

## Files this design touches
- `crates/io-prod/tests/mesh_load.rs` (new, R-4e1: harness + LoadShape + sized inbox + N=8 all-pairs outage variant + L5 ignore-guard + conn_died test)
- `crates/io-prod/tests/mesh_under_loss.rs` (new, R-4e4: R-5 capstone)
- `crates/io-prod/src/mesh.rs` (R-4e2 H2 re-key at :315 type / :903 classify_and_deliver / :963 ack_egress; + the optional item-3 coalesce field on `ReliableLaneSender` :439 / `confirm_and_maybe_bounce` :1290 ONLY if the outage test reds)
- `justfile` (the `mesh-load` recipe — NOT added to `gate:124`)
- `docs/design/DEFERRED.md` (:1044-1045, :1127-1131, :1186-1213, :1214-1256 — the D-6 #1 partial flip + the M3+L5-gated AwaitAdopt residual wording + the SharedInbox-next-scale-point new item + the end-goal scope precision)
- `crates/sim/src/io/mod.rs` (:311-327 delivery-semantics doc prose only — the frozen seam SHAPE is unchanged; R-4e is below the seam)

---

## VERDICT: SOUND_TO_IMPLEMENT

All CRITICAL/HIGH/MEDIUM findings from the three reviews are code-verified and folded: the inbox is sized from the LoadShape (the CRITICAL flake cure), the H2 rationale is rewritten to the true cross-stream-race invariant with `push_inbox` held under the per-peer lock and `ack_egress` keeping its per-key lookup, the correlated-outage is the N=8 all-pairs topology, the L5 guard asserts the mechanism narrowly, and the D-6 #1 flip is a precise partial flip naming the M3+L5-gated AwaitAdopt source-crash residual (not "no recovery"). No new dependency (std `RwLock`/`Mutex`/`Arc`). The frozen seam shape is untouched.

**Recommended FIRST sub-slice: R-4e1** — the `mesh_load.rs` harness (fan-in, sized inbox, completion-gated exact assertions + derived progress deadline) + the `just mesh-load` recipe, committed with H2 UNCHANGED. It empirically confirms the pinned-N feasibility (green 5/5 before pinning), establishes the regression baseline the H2 re-key (R-4e2) is measured against, and de-risks the single biggest unknown (64-node feasibility) before any code change.