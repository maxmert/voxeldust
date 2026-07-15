# R-6d Durable Outbox — Vetted Design of Record

Design workflow wf_0af5fa8e (design + 3 adversarial opus reviews + synthesis). Verdict: REVISE (core design SOUND; 3 CRITICALs folded into the corrected sub-slice plan below = implement-ready).

=====================================================================
## PART 1 — ADJUDICATED SYNTHESIS (the design of record)
=====================================================================

I have complete grounding. Four `Transport::send` impls exist (`MemTransport`, `MeshTransport`, `ProdTransport`, `FabricTransport`) — a seam-signature change touches all four in lockstep. All findings are adjudicated. Here is the design of record.

---

# R-6d — THE DURABLE OUTBOX: Design of Record (adjudicated synthesis)

## (a) THE LAYERING DECISION + atomicity ruling + scope — STATED AS FACT

**DECISION: Shape (A′) — transport-owned durable outbox, write-through mirror of `ReliableLaneSender.retry`, its own fsync, NOT atomic with any above-seam saga step — ADOPTED. Plus a REQUIRED saga-side change AND a REQUIRED dest-side Arriving-discard. All three are load-bearing; none alone closes D-6 #1.**

**Shape (B) is REJECTED as infeasible (fact, code-cited).** `Transport::send(to,class,bytes) -> MsgId` (`crates/sim/src/io/mod.rs:330`) returns a `MsgId`; the actual send only enqueues an `OutFrame{to,class,bytes,msg_id}` onto a per-peer mpsc and returns (`mesh.rs:1712-1725`). The `(peer,class,incarnation,seq)` key is minted **asynchronously inside the peer_writer task** by `assign_and_retain` (`mesh.rs:566-601`), and the lane itself is lazily created there (`write_frame`, `mesh.rs:1538 entry(...).or_insert_with(ReliableLaneSender::new)`). The seq does not exist above the seam, so the saga runtime cannot write an outbox row keyed by it inside its step `commit()`. (B) also would force the saga to hold framed transport bytes — an HR1 leak. **Rejected on feasibility, not taste.**

**Shape (C) full new seam method REJECTED**, but a **minimal additive seam change is REQUIRED** and must be owned (see HIGH-2): the per-send durability bit cannot be inferred from `MsgClass` and must reach the writer task, and `send()`'s frozen signature carries no field for it.

**Atomicity ruling (fact): atomicity with an above-seam saga step is UNNECESSARY and, for the canonical flow, a category error.** The `TransientBatch` producer is the SOURCE SHARD (`emit_transient_batch`, `stub.rs:2070-2137`), which stages **no durable saga row** for the emit — the go-token commit happens on the ORCHESTRATOR (`saga_runtime.rs:960-1000`), a different node. There is no source-side txn to be atomic with. The correct invariant is a per-node SOURCE property: *a batch frame that can ever be observed as sent must survive the source's crash.* That is satisfied entirely below the source's seam by an **ordering rule** (durable-before-first-send), which is strictly stronger than "atomic with a step that doesn't exist." **BUT** the design-under-review placed that ordering in `flush_outbox` where it cannot live (CRITICAL-1); the corrected home is inside the writer task (§(c) R-6d2).

**Scope (fact): ONLY producer-less reliable one-shots get the outbox — NOT all reliable traffic.** Every saga-DRIVEN reliable flow re-drives via `scan_deadlines` on orchestrator restart, so it needs no transport outbox. Fsync-per-reliable-send on the hot path is rejected. The closed set today:
1. `TransientBatch` (source shard → dest, `MsgClass::Saga`, `stub.rs:2134`) — the D-6 #1 case.
2. `GhostFlow::Despawn` (band-exit, `MsgClass::GhostReliable`, per `mesh_under_loss.rs`).

Scoped via a per-send `Durability` marker at the `push_flow` call site (`runtime.rs:63`) — a property of the individual send, NEVER a match on `ShardProfile`/`NodeKind`, NEVER a lane fork. One `ReliableLaneSender` FSM, one `OutboxSink`. Note: `MsgClass::Saga` and `GhostReliable` each carry BOTH producer-less one-shots AND re-driven traffic, so the bit is genuinely per-send, confirming the marker (not class inference).

---

## (b) DEDUPLICATED, SEVERITY-RANKED FINDINGS THAT MUST CHANGE

### CRITICAL-1 — The §1.2 ordering gate cannot live in `flush_outbox`; relocate into the writer task. [CONFIRMED — Reviewer 1 & 3 both]
**Verified:** `flush_phase` calls `transport.send()` which only enqueues (`app.rs:249` → `mesh.rs:1712-1725`). `assign_and_retain` (the seq mint + proposed write-through) and `write_reliable_frame` (the QUIC send) are back-to-back inside `write_frame` in the peer_writer task (`mesh.rs:1538-1655`). The `DurabilityHandle::wait_durable_through` is owned by the sim thread (`orchestrator.rs:301`, `store.rs:510`); `PeerWriter` (`mesh.rs:1075-1093`) holds none of it. The design's "assign_and_retain → outbox.commit() → wait_durable_through → write_frame in flush_outbox" is impossible as written; the entire "stronger than atomic" proof is unsupported at that location.
**FIX:** Inject an outbox handle into `spawn_mesh` → `PeerWriter`. In `write_frame`'s Reliable arm, after `assign_and_retain` write-throughs the row, **the writer stages + fsyncs the outbox row and blocks on the outbox's own durability barrier BEFORE `write_reliable_frame`** — but ONLY when the frame is `durable=true` (producer-less one-shots), so the fsync is off the common path. Re-derive the §1.2 crash-window table against this location. This is the corrected home of the ordering rule; the safety argument (durable-before-first-send + at-least-once + receiver dedup) then holds.

### CRITICAL-2 — Orphaned `Arriving` is a SURVIVING silent loss; the design rests on a non-existent dest-side GC. [CONFIRMED — Reviewer 2]
**Verified:** On the replayed batch the dest runs `adopt_transient_batch` → `journal_step(TRANSIENT_BATCH_STEP)=FirstApply` → inserts each item as `TransientStatus::Arriving{batch}` (`stub.rs:2155-2167`). An `Arriving` item is removed ONLY by `on_transient_promote` flipping `Arriving→Held` on a `TransientDrop` (`stub.rs:2241-2247`). `on_transient_abandon` runs on the **SOURCE** and removes only `Departing`/`Held{Some}` (`stub.rs:2318-2335`); `EmitTransientAbandon` targets `ctx.source` (`saga_runtime.rs:872`). **There is NO dest-side path that removes an `Arriving` item on abort.** So if the orchestrator abandons+tombstones the saga and the restarted source's outbox later replays the batch, the dest inserts `Arriving` against a tombstoned saga; the `BatchAdopted` ack hits a GC'd saga (idempotent no-op), no `TransientDrop` is ever emitted, and the item is stuck `Arriving` forever — uncounted, un-rendered, authoritative nowhere. **This IS the silent loss D-6 #1 must close, re-created by the interleave.** The design's "harmless orphan the dest GCs" (§4.2, ⟦A7⟧) is false.
**FIX:** Add a dest-side discard as the THIRD element. Introduce `InterShardFlow::TransientDiscard(TransientHandoff{transfer, step_id: TRANSIENT_DISCARD_STEP, fence})` targeting `ctx.dest`, handled by a new `on_transient_discard` that removes any `Arriving{batch==transfer}` item (journaled by `(transfer, TRANSIENT_DISCARD_STEP)`, idempotent, counted into `transients_lost_in_handover` by kind — the SAME budget). Journal-order it so a replayed `adopt_transient_batch` arriving AFTER the discard is `AlreadyApplied` (never re-inserts): the discard must WRITE `(transfer, TRANSIENT_BATCH_STEP)` into `AppliedSteps` (poison the adopt) in addition to removing the item. The AwaitAdopt-source-dead resolution (CRITICAL-3) emits this discard to the dest INSTEAD of a source-addressed abandon.

### CRITICAL-3 — `(BatchHandoff, SourceUnreachable)` self-promotes from `AwaitAdopt` (wrong) AND cannot fire there anyway (park-forever). [CONFIRMED — original design §4 + Reviewer 2 HIGH, merged]
**Verified (two coupled defects):**
- **(3a) The phase-swallow bug is real.** `saga.rs:919 (S::BatchHandoff { new_fence, .. }, E::SourceUnreachable)` uses `..`, and the comment "every BatchHandoff phase is post-adopt" is provably false: `AwaitAdopt` is entered on `CasWon` with `vec![]` (`saga.rs:815-819`), documented producer-less. Self-promoting from `AwaitAdopt` promotes an empty dest. DEFERRED.md:965-966 independently flags this.
- **(3b) But `SourceUnreachable` cannot even fire in `AwaitAdopt`.** `is_confirmed_dead(source)` reads `LivenessTracker.seen` (`saga_runtime.rs:281`), fed only by `Inbound::NodeUnreachable → record_unreachable` (`:1684`, `:256`), synthesized only by `confirm_and_maybe_bounce` (`mesh.rs:1335`) — which requires a reliable lane toward the source with a **non-empty retry buffer**. In `AwaitAdopt` the orchestrator sends NOTHING to the source: `IssueTransientGo` records `batch_gos` + feeds `CasWon` locally, no wire egress (`saga_runtime.rs:815-826`). With no orchestrator→source lane, no `NodeUnreachable{to:source}` accrues → `is_confirmed_dead(source)` stays false → `rehome_event_for` returns `Timeout` (`saga_runtime.rs:1271-1283`) → the saga re-drives forever on `redrive_deadline_ticks` (`:1239`). A permanently-dead source in `AwaitAdopt` PARKS — a leaked live saga + lost batch, an unclosed D-6 #1 in its own right.
**FIX (both):** Split the `saga.rs:919` arm by phase. For post-adopt phases (`AwaitRelease`/`AwaitPromote`/`AwaitComplete`) keep the self-promote (dest provably holds the batch — unchanged). For `AwaitAdopt`, resolve as accounted loss via a NEW distinct event/action (CRITICAL-2's `EmitTransientDiscard` to the DEST + a distinct `A::CountBatchLostSourceCrash`, NOT the source-addressed `EmitTransientAbandon`, and NOT `EmitTransientPromote`). AND make the resolution REACHABLE: give the orchestrator a liveness signal toward the source in `AwaitAdopt`. The DEFERRED.md-blessed template (`:280-281`, the 2d re-solicit-egress FSM-field pattern) is exactly this: add an `AwaitAdopt` re-solicit egress (an idempotent re-prompt toward the source, journaled by `(transfer, step)`) so a dead source accrues `NodeUnreachable` and `is_confirmed_dead(source)` CAN fire — converting park-forever into a bounded terminal resolution. **This re-solicit is now REQUIRED (not the "redundant" the design claimed), because without ANY orchestrator→source egress the confirm-dead path is unreachable.** Gate the destructive discard on `abort_deadline_ticks` budget (mirror the dest-dead ladder, `saga_runtime.rs:1274-1279`) so a source that RESTARTS within budget delivers via its outbox replay before the discard fires (the two recoveries race, budget picks the restart winner).

### HIGH-1 — The 20 Hz datagram claim is code-identical but NOT latency-identical under the shared peer_writer loop. [CONFIRMED — Reviewer 1]
**Verified:** `peer_writer` is ONE `tokio::select!` per peer (`mesh.rs:1118`, `biased`), draining reliable writes, ack-reads, retransmit, AND (via `write_frame`'s `Reliability::Unreliable` arm) datagrams through the same task. With CRITICAL-1's fsync-in-writer, a producer-less reliable send that blocks on `outbox.commit()` head-of-lines a datagram to the SAME peer queued behind it. The CODE path is byte-identical (correct — `assign_and_retain` is reliable-only, `mesh.rs:1528`), but the "zero cost" latency claim is not supported.
**FIX:** State the corrected claim: "datagram CODE unchanged; the added fsync is paid ONLY by producer-less reliable sends (AwaitAdopt batches + band-exit Despawns — low-rate, unlike Snapshot/Input), and bounded off the datagram latency path by <mechanism>." Prefer: stage the retain in the writer, perform the fsync on the store's existing off-tick writer thread (`store.rs:292` — independent of the tokio worker, so no deadlock), and gate ONLY the producer-less QUIC send on `last_durable >= seq`. Add an explicit no-deadlock argument: the outbox `RedbStore` writer thread is independent of the tokio peer_writer, so parking one tokio worker on `wait_durable_through` never blocks the fsync thread.

### HIGH-2 — The per-send `durable` bit requires a REAL additive seam change; "send unchanged" is false. [CONFIRMED — Reviewer 3]
**Verified:** The bit must reach `assign_and_retain` in the writer task; the only channel is `send()→OutFrame→mpsc`. `send()`'s frozen signature (`io/mod.rs:330`) and `OutFrame{to,class,bytes,msg_id}` (`mesh.rs:1720-1725`) have no durability field, so a bit placed only on the `OutboundBox` tuple is dropped at the `flush_phase→send()` boundary (`app.rs:249`). §8's "send unchanged + marker only" is self-contradictory.
**FIX:** Own an explicit additive seam extension, lockstep: grow `Transport::send(to, class, bytes, durability: Durability)` (or a sibling `send_retained`), threaded through `flush_phase` and into a new `OutFrame.durability` field consumed by `write_frame → assign_and_retain`. Update ALL FOUR impls in the same slice — `MemTransport` (`mem.rs:343`), `MeshTransport` (`mesh.rs:1712`), `ProdTransport` (`lib.rs:162`), `FabricTransport` (`fabric.rs:516`) — the mem/fabric tiers no-op the bit (no fsync), behavior-identical. Correct §8 to say the seam DOES change additively.

### HIGH-3 — Boot replay + gc ordering must gate fsync-before-send on the replay path too. [CONFIRMED — Reviewer 1]
**Verified:** §3.2 routes recovered frames "through the normal reliable send path with durable=true" and claims crash-safety, but does not pin WHERE the replay-path fsync sits (same sim/writer split). Also `gc_below` "committed with the first flush" can race: if the old-incarnation rows are deleted+fsynced BEFORE the fresh rows are durable, a crash mid-replay loses both.
**FIX:** Specify replay ordering identical to CRITICAL-1: each recovered frame → re-mint seq at fresh incarnation → write NEW outbox row → fsync → QUIC send. Order GC strictly AFTER: write ALL fresh rows + fsync, THEN `gc_below(new_incarnation)` — never the reverse. Add this interleave to the proptest (below).

### MEDIUM-1 — `inject_replay` is unreachable: `build_app` consumes the transport by value. [CONFIRMED — Reviewer 3]
**Verified:** `build_app<T: Transport>(cfg, transport) -> ShardNode<T>` moves the transport into `ShardNode` (`app.rs:87,101`); the bin holds no `MeshTransport` afterward, and the lanes live in the writer tasks anyway.
**FIX:** Drop the bespoke `inject_replay`. Replay is N ordinary `send(.., Durability::Retained)` calls issued in scan order BEFORE `build_app` consumes the transport (or via a thin pre-consume drain the bin owns). The writer's fresh lane assigns seq 0,1,2… at the bumped incarnation exactly as §3.1 wants — riding the existing send path. The §3.1 epoch-0/seq0-rebase acceptance analysis remains correct against `classify_reliable`/`prime_or_contiguous` (`mesh.rs:358-421`, verified — A1 resets to `hw:0,primed:false` then primes at seq0 → Reset, subsequent contiguous → Accept, no Gap/Dedup/Stale).

### MEDIUM-2 — The named acceptance gate is below the seam and blind to the saga↔dest race. [CONFIRMED — Reviewer 2]
**Verified:** io-prod tests model only the transport FSM + `RecvState` ladder; they cannot exercise the CRITICAL-2 orphan interleave (a saga-runtime↔shard composition property). The design's §7.1 proptest would go GREEN while CRITICAL-2 is live.
**FIX:** Keep the io-prod proptest as the TRANSPORT-delivery gate (rename so it does not masquerade as the D-6 #1 closure gate). Make the D-6 #1 acceptance gate the COMPOSED process proof (§(c) R-6d4), and it MUST include the restart-AFTER-abandon interleave asserting ZERO orphaned `Arriving`.

### MEDIUM-3 — The proptest is Tier-B (io-prod), not Tier-A, and must run against a mock, not real redb. [CONFIRMED — Reviewer 3]
**Verified:** `NodeOutbox` wraps a real `RedbStore` (spawned writer thread + fsync, `store.rs:397-436`); thousands of proptest cases cannot run against real disk deterministically, and io-prod is Tier-B (justfile floor 90), not Tier-A.
**FIX:** The proptest runs in io-prod against an in-memory `MockOutboxSink` (pure over {sender FSM window, mock mirror, `RecvState` ladder}). A SEPARATE small `NodeOutbox`-over-real-`RedbStore` test covers the persistence glue (round-trip, `scan_all` order, `gc_below`). Correct the "Tier-A" label to "pure io-prod unit-tier (mock-backed)."

### MEDIUM-4 — `EmitTransientAbandon` mis-reuse + missing distinct counter for the source-crash loss. [CONFIRMED — Reviewer 1] (subsumed into CRITICAL-3's fix)
Reusing the dest-dead `EmitTransientAbandon` (→ `ctx.source`, a corpse) for a source-dead cause is semantically wrong and would miscount as a zero-loss `source_unreachable_resolutions`. The CRITICAL-3 fix already introduces a distinct action (`CountBatchLostSourceCrash`) + distinct metric — record it here so the ledger is not conflated.

### LOW (ledger — fix in-slice, non-blocking)
- **L1. Stored value contradiction (§2.2 writes `&encoded` = u32::MAX epoch; ⟦A5⟧ says epoch 0).** [CONFIRMED] Resolve to storing `postcard(ReliableFrame{epoch:0,..})` (replay-ready). But because ⟦A5⟧ hand-rolls a SECOND encoding of `ReliableFrame` outside `vd_wire::framing::encode_frame` (the ONE framing home, HR3), wrap the row in a **1-byte `OUTBOX_FORMAT_VERSION` envelope** (mirroring DEFERRED.md #3's WAL versioning) with a fallible/quarantining decode on boot. [Reviewer 1 LOW + Reviewer 2 LOW]
- **L2. Wrong test paths.** `mesh_under_loss.rs` is at `crates/io-prod/tests/` (NOT `crates/bins/tests/`), verified. `orchestrator_crash.rs` IS at `crates/bins/tests/` (correct). Fix throughout. [Reviewer 2 & 3]
- **L3. `OutboxKey` size self-contradiction (§2.3 says both "28 bytes" and "26 bytes" for `1+8+1+8+8`=26).** Settle at 26 bytes, stated once. [Reviewer 3]
- **L4. Magic numbers.** The key-family tag `0x4F`, byte offsets, `VD_OUTBOX_PATH`, and any fsync/replay cadence must be named consts in `outbox.rs` / folded into a `MeshReliabilityTuning` sibling `OutboxTuning`, not inline literals (HR "no magic numbers"). [Reviewer 3]
- **L5. `ReliableLaneSender` gains a `peer: NodeId` field** (⟦A4⟧, set in `new`) to build the key — no behavior change. Cross-peer concurrency: the `NodeOutbox` is shared by all writer tasks behind one handle; retain/release keys are disjoint per `(peer,class,inc,seq)`, so redb LWW-per-key is safe; state this contract explicitly. [Reviewer 1 MEDIUM]

---

## (c) CORRECTED, IMPLEMENT-READY SUB-SLICE PLAN

### R-6d1 — per-node RedbStore for shard (+gateway) + the writer-task durability handle.
**Touched:** `crates/bins/src/lib.rs` (new `open_node_outbox` helper: `VD_OUTBOX_PATH` + `VD_STORE_DURABLE_ROOT`, reuse `boot::check_durable_path` at `boot.rs:120`; absent path → `None`); `crates/bins/src/bin/shard.rs` + `gateway.rs` (open the outbox in `main`; split the tick loop into `run_schedule`/`flush_outbox` mirroring `orchestrator.rs:273-324`, gated on `outbox.is_some()` — `None` keeps `step_tick()` byte-identical); `crates/io-prod/src/outbox.rs` (new `NodeOutbox::open` wrapping `RedbStore`); `crates/io-prod/src/mesh.rs` (`spawn_mesh` + `PeerWriter` gain an outbox handle field). Only the SHARD strictly needs it; gateway opens one for HR3 uniformity (empty until it has a producer-less flow).
**Gate/test:** `crates/bins/tests/node_outbox_boot.rs` — shard boots with `VD_OUTBOX_PATH` (file created); temp path w/o escape → refused loud; absent → `step_tick` fast path unchanged.

### R-6d2 — `OutboxSink` seam + FSM write-through/delete-through + the additive `send` seam (HIGH-2).
**Seam extension (lockstep, all four impls):** `Transport::send(to, class, bytes, durability: Durability)`; new `OutFrame.durability`; `push_flow` gains a `Durability` param (default `Ephemeral`). Update `mem.rs:343`, `mesh.rs:1712`, `lib.rs:162`, `fabric.rs:516`.
**`OutboxSink` (io-prod, `&dyn`, object-safe, branchless per HR5(a)):**
```
trait OutboxSink: Send {
    fn retain(&mut self, key: &OutboxKey, framed: &[u8]);  // staged; after the RAM insert
    fn release(&mut self, key: &OutboxKey);                // staged; on ack retire
    fn commit(&mut self) -> u64;                           // fsync barrier; returns durable seq
    fn scan_all(&self) -> Vec<(OutboxKey, Vec<u8>)>;       // boot rehydrate, ascending
    fn gc_below(&mut self, incarnation: u64);              // sweep stale-incarnation rows
}
```
**Key (26 bytes, big-endian so lexicographic==numeric):** `[0x4F tag][peer u64][class u8][incarnation u64][seq u64]`. Monomorphic `OutboxKey::{new,to_bytes,from_bytes}`, round-trip unit-tested.
**Value:** `OUTBOX_FORMAT_VERSION(1B) || postcard(ReliableFrame{epoch:0,..})` (L1).
**FSM:** `assign_and_retain(from,class,bytes, durable:bool, sink:Option<&mut dyn OutboxSink>)` — after the RAM insert (`mesh.rs:597`), the MONOMORPHIC straight-line `if durable { if let Some(s)=sink { s.retain(&key, &value) } }` (fully covered by `durable×sink` unit cases). `on_ack(.., sink)` deletes-through each retired seq inside the existing retire loop (`mesh.rs:540-551`). `ReliableLaneSender` gains `peer` (L5). All lanes' staged retains/releases flushed by ONE `sink.commit()` per flush-tick (batched — one fsync per burst).
**Gate/test:** synchronous FSM unit tests with `MockOutboxSink` (records exact key bytes + value; `durable=false`/`sink=None` record nothing); `OutboxKey` round-trip; one `NodeOutbox`-over-real-`RedbStore` glue test. io-prod Tier-B floor held/ratcheted.

### R-6d3 — the durability gate (CRITICAL-1) + boot replay (MEDIUM-1, HIGH-3) + the saga+dest closure (CRITICAL-2, CRITICAL-3).
**Transport (CRITICAL-1 + HIGH-1):** in `write_frame`'s Reliable arm, when `frame.durability==Retained`: after `assign_and_retain` retains, stage the row, and gate `write_reliable_frame` behind the outbox row being durable — fsync on the store's OFF-TICK writer thread, the writer polls `last_durable >= seq` (non-blocking select yield, no tokio-worker park on the fsync itself). Non-durable frames: unchanged fast path. No-deadlock: outbox writer thread ⟂ tokio peer_writer.
**Boot (MEDIUM-1 + HIGH-3):** bin `main`, before `build_app` consumes the transport: `incarnation = resolve_process_incarnation` (M3 bump); `outbox = NodeOutbox::open`; `for (key, val) in outbox.scan_all() { send(key.peer, key.class, decode(val).bytes, Durability::Retained) }` in ascending order (fresh lane assigns seq 0,1,2…); write+fsync all fresh rows, THEN `gc_below(incarnation)`. GC-after-durable strictly.
**Saga (CRITICAL-3, pure Tier-A):** split `saga.rs:919`. Post-adopt phases → keep self-promote. `AwaitAdopt` → NEW `E::SourceUnreachablePreAdopt` → `A::EmitTransientDiscard{dest, fence}` + `A::CountBatchLostSourceCrash` + `Tombstone`, budget-gated on `abort_deadline_ticks`. Add an `AwaitAdopt` re-solicit egress (the DEFERRED.md:280 FSM-field template) so `NodeUnreachable{source}` can accrue and the resolution is REACHABLE. New distinct counter `batch_lost_source_crash` (`saga_runtime.rs:1533` region).
**Dest (CRITICAL-2, pure Tier-A):** new `InterShardFlow::TransientDiscard` + `on_transient_discard` — remove `Arriving{batch==transfer}` items, journal `(transfer, TRANSIENT_DISCARD_STEP)` AND poison `(transfer, TRANSIENT_BATCH_STEP)` so a late replayed adopt is `AlreadyApplied`; bucket into `transients_lost_in_handover`.
**Gate/test:** saga FSM unit tests (`assert_eq!` on `(SagaState, Vec<SagaAction>)` per HR5(d)) for every phase×event of the split; `on_transient_discard` idempotence + adopt-after-discard no-op; a synchronous boot-replay test proving fresh lanes carry recovered frames at epoch 0 seq 0.. and `classify_reliable` Accepts (Reset then Accept, `gap_drop=0`) at bumped incarnation.

### R-6d4 — the NAMED acceptance gates (MEDIUM-2, MEDIUM-3).
**`proptest_both_ends_restart_replay_interleave` (io-prod, pure, MockOutboxSink):** interleave (source-restart+full replay at strictly-higher incarnation), (receiver-restart to genesis), (acks retiring an arbitrary prefix), any order/rounds. Assert: every retained-and-committed-not-released frame delivered ≥once; `gap_drop==0` every replay; no over-delivery beyond dedup; mirror ⊆ RAM at every commit; after all acks `scan_all` empty AND `gc_below` swept stale rows; **AND the crash-during-gc interleave (HIGH-3) loses nothing.** Renamed to signal it is the TRANSPORT-delivery gate, not the D-6 #1 gate.
**`sigkill_source_in_await_adopt_redelivers_the_batch` (bins, real QUIC, extends `orchestrator_crash.rs`'s SIGKILL/`store-test-hooks` sentinel + `io-prod/tests/mesh_under_loss.rs`):** three scenarios — **(i)** source SIGKILLed in the durable-but-not-sent window then RESTARTED → dest adopts exactly once, `gap_drop==0`, saga reaches `Done` via normal adopt→promote (NOT `SourceUnreachable`); **(ii)** source SIGKILLed in `AwaitAdopt` and NEVER restarted → saga reaches a BOUNDED terminal via `SourceUnreachablePreAdopt` → discard-to-dest + `batch_lost_source_crash` climbs, dest ends with ZERO orphaned `Arriving` (the CRITICAL-2/3 composed gate); **(iii)** RED control WITHOUT `VD_OUTBOX_PATH` → batch LOST after crash (proves the outbox is load-bearing).
**Gate:** all green + Tier-A sim/saga 100% region+branch + io-prod Tier-B floor held.

---

## (d) EXPLICIT RULINGS

- **Does the transport outbox + the saga-side change TOGETHER close D-6 #1 with zero silent loss and no double-adopt?** As the design-under-review was written: **NO.** Two additional required elements were missing: the AwaitAdopt resolution is UNREACHABLE without an orchestrator→source re-solicit egress (CRITICAL-3b), and there is NO dest-side `Arriving` GC so the abandon-then-replay interleave strands an orphan (CRITICAL-2). **With all THREE elements** (outbox + phase-gated-and-reachable saga resolution + dest-side `TransientDiscard` with adopt-poisoning), **YES**: the restart path delivers-and-adopts (single adopt, journal-idempotent), the permanent-death path resolves bounded + counts the loss loud + discards any late orphan, and no `{source,dest}` double-hold occurs (journal dedup on `TRANSIENT_BATCH_STEP`/`TRANSIENT_DROP_STEP` prevents double-count; the discard poisons the adopt to prevent a resurrected orphan).
- **Is the boot replay deterministically Accepted?** **YES** — verified against `classify_reliable`/`prime_or_contiguous` (`mesh.rs:358-421`): a bumped incarnation resets `RecvState` to `hw:0,primed:false` (A1, `:362`), the re-based seq-0 first frame primes → `Reset`, subsequent contiguous → `Accept`; no Gap/Dedup/StaleEpoch. Sound provided replay re-mints via `assign_and_retain` on a fresh lane in ascending scan order (MEDIUM-1's send-based replay) and GC runs after fresh-row durability (HIGH-3).
- **HR1?** **CLEAN** — the outbox value is opaque already-framed `ReliableFrame` bytes, never a sealed `World`, never decoded by the outbox.
- **HR3?** **CLEAN** — one `ReliableLaneSender` FSM (gains params, no fork), one `OutboxSink`/`NodeOutbox`, one `open_node_outbox` helper, per-send `Durability` PARAMETER (never a shard-kind match).
- **HR5?** **MET** — Tier-A sim/saga stays 100% (`assert_eq!` arms, monomorphic straight-line guards); io-prod Tier-B floor held; proptest mock-backed (MEDIUM-3).
- **Frozen `sim::io` seam?** **TOUCHED — additively, and this MUST be owned explicitly** (HIGH-2): `send` grows a `Durability` param across all four impls in lockstep; the mem/fabric tiers no-op it. This is a justified additive extension, not the "send unchanged" the design claimed. `Store` is NOT modified.
- **No new dependency?** **MET** — the transactional-outbox pattern over the existing `redb` `RedbStore` (`store.rs:186`); no broker/oxide-outbox.
- **20 Hz datagram path?** **CODE byte-identical; NOT unconditionally latency-identical** — the fsync is paid only by low-rate producer-less reliable sends and must be bounded off the datagram path (HIGH-1); the design's unqualified "zero cost" claim must be re-stated.

---

## (e) FINAL VERDICT

**REVISE.** The core layering DECISION (Shape A′, non-atomic, transport-owned), the producer-less-only scope, the boot-replay determinism, and the HR1/HR3/HR5/no-new-dep posture are all SOUND and code-confirmed. But the design is NOT implement-ready: three CRITICALs invalidate load-bearing claims — the §1.2 durability gate cannot live where placed (CRITICAL-1), the AwaitAdopt closure both self-promotes wrongly AND is unreachable so it parks forever (CRITICAL-3), and the abandon-then-replay interleave strands a silent `Arriving` orphan because the assumed dest-side GC does not exist (CRITICAL-2). The transport outbox alone does NOT close D-6 #1; it needs the reachable phase-gated saga resolution AND a new dest-side `TransientDiscard`. Once R-6d3 folds in all three plus the additive `send` seam (HIGH-2), the writer-task fsync home (CRITICAL-1/HIGH-1), and the composed process gate (MEDIUM-2), this is SOUND_TO_IMPLEMENT. The sub-slice plan above is the corrected design of record.

=====================================================================
## PART 2 — ORIGINAL DESIGN (fuller detail the synthesis corrects)
=====================================================================

I now have complete grounding. I've read: the sketch, the mesh FSM (assign_and_retain/on_ack/replay_batch, classify_reliable A1 ladder), the frozen Store/Transport seam, the saga AwaitAdopt residual + SourceUnreachable arm, RedbStore, boot.rs check_durable_path, the bin wiring (orchestrator persist-before-effect gate vs gateway/shard step_tick with no store), the TransientBatch source-shard emit path, and the DEFERRED.md D-6 #1 ruling. 

Key decisive finding: the `(peer, class, incarnation, seq)` key is minted **asynchronously inside the writer task** (`assign_and_retain`), NOT at the above-seam `transport.send()` call — `send()` returns only a `MsgId`. This rules out option B cleanly. Now I'll write the design of record.

# R-6d — THE DURABLE OUTBOX: Design of Record

## 0. Executive summary + the one decisive fact

**Decision: Shape (A), transport-owned, refined into (A′) — a transport-owned durable outbox that is a write-through mirror of `ReliableLaneSender.retry`, persisted with its OWN fsync BELOW the frozen seam, NOT atomic with the saga step, made safe by a boot-ordering rule (replay-before-first-send) that is *stronger* than "atomic with the step." A saga-side companion change is ALSO required: `(BatchHandoff, SourceUnreachable)` must be phase-gated so it never self-promotes from `AwaitAdopt`. Neither piece alone closes D-6 #1; both are required.**

The single fact that decides the layering: **the `(peer, class, incarnation, seq)` key does not exist above the seam.** `Transport::send(to, class, bytes) -> MsgId` (`sim/io/mod.rs:330`) returns a `MsgId` (a monotone send counter), not a seq. The seq/epoch/incarnation are minted *asynchronously* inside the writer task by `assign_and_retain` (`mesh.rs:566-601`), long after `flush_outbox` returned. Shape (B) — "the saga runtime writes the outbox row inside the step `commit()` txn" — is therefore **impossible without inventing a new seam that leaks the transport frame identity (incarnation, epoch, seq) up into the saga runtime**, which is a worse HR1/HR3 violation than the one it tries to avoid, and would require the saga runtime to know the lane's `next_seq` before the lane exists. **(B) is rejected on feasibility, not just taste.** The rejection is detailed in §1.

Everything below is written to be attacked. Assumptions are flagged `⟦A#⟧`.

---

## 1. THE LAYERING DECISION (the crux)

### 1.1 The three candidates, adjudicated

**Shape (B) — saga-runtime-owned outbox row in the step commit() txn — REJECTED (infeasible + a worse inversion).**
- Infeasible: the outbox key is `(peer, class, incarnation, seq)`. At the point the saga runtime stages its step in the group-commit barrier (`drive_sagas` tail, the ONE barrier per DEFERRED.md:981), the frame has not been assigned a seq — the lane may not even exist yet (`lanes.entry(class).or_insert_with(ReliableLaneSender::new)` runs in `write_frame`, `mesh.rs:1539-1541`, inside the writer task on the first send to that class). To write the row above the seam, the saga runtime would have to *pre-compute* the seq, which means moving `next_seq` ownership above the seam — dissolving the entire R-2b FSM.
- Even if we keyed the row by `(transfer, step_id)` instead (which the saga *does* know — it is the `applied_steps` key), the value would have to be the **framed transport bytes** (`ReliableFrame` with incarnation/epoch/seq), which the saga runtime does not possess and must not (HR1: the saga is above the wire framing; framing lives in `vd_wire::framing::encode_frame`, called only in io-prod). So (B) forces either a seam that hands raw frame bytes up, or a re-encode divergence between the saga's stored blob and the transport's actual frame. Both are defects.
- Verdict: (B) couples the saga step encoding to the transport frame encoding (the prompt's own stated cost) AND is not even mechanically realizable without gutting the FSM. **Rejected.**

**Shape (C) — a new seam method (e.g. `Transport::send_durable`) — REJECTED for the general case, PARTIALLY ADOPTED as an additive marker.**
- A full new seam method that threads a durable handle down is unnecessary: the outbox is entirely a transport-internal concern once we accept (A′). But we DO need the above-seam caller to *mark which sends are producer-less one-shots* (see §1.3 scope). That marker is the minimal additive seam touch, justified in §8. It is a bit on the existing enqueue, not a new method with new semantics.

**Shape (A′) — transport-owned durable outbox, write-through mirror, own fsync, NOT atomic with the saga step, boot-ordered — ADOPTED.** Detailed below.

### 1.2 Why the non-atomic window is SAFE (the core proof)

The sketch worries the outbox write is not in the same txn as the saga step. Here is why that is not merely tolerable but *irrelevant*, given the orchestrator's existing persist-before-effect gate and at-least-once semantics.

**Claim: for the source shard emitting the `TransientBatch`, there is NO "saga step commit" to be atomic with in the first place.** The `TransientBatch` is emitted by `emit_transient_batch` (`stub.rs:2070-2137`) — a SOURCE SHARD system that pushes an `InterShardFlow::Transfer` envelope on `MsgClass::Saga` into `OutboundBox`. **The source shard has no Store today and stages no durable saga row for this emit.** The durable go-token commit (`BatchCommitting → CasWon`) happens on the ORCHESTRATOR, not the source. So the "same txn as the producing saga step" framing in the sketch is a category error for the actual producer-less flow: the producer is a shard, the commit is an orchestrator's, and they are already two different nodes. The atomicity the sketch reaches for never existed.

**What actually must hold (the real invariant):** *if the source's `TransientBatch` send is ever going to be counted as sent (the item flipped `Crossing → Held{outbound}`, `stub.rs:2112`, and — post-R-6d — the orchestrator allowed to self-promote on source death), then a durable record of that batch's frame bytes must survive the source's crash so it can be re-sent.* This is a per-node property of the SOURCE, satisfiable entirely below the source's seam.

**The ordering rule that replaces atomicity (stronger than "atomic"):**

The source bin adopts the **same split the orchestrator already has** (`run_schedule` / `flush_outbox`, `app.rs:128-157`) plus a durability wait between them (§5). The write-through outbox `commit()` (its own fsync) happens in `flush_outbox`, and the actual QUIC send is gated behind that fsync being durable:

```
tick T:
  run_schedule()          → emit_transient_batch stages the envelope into OutboundBox
  flush_outbox():
     for each reliable producer-less frame:
        assign_and_retain(...)            // mints (inc,epoch,seq), inserts into RAM retry
        outbox.put(key, framed_bytes)     // write-through mirror, STAGED
     outbox.commit()                      // ← the outbox's OWN fsync barrier
     wait_durable_through(seq)            // block until the mirror is on disk
     write_frame(...)                     // ONLY NOW does the QUIC frame leave
```

Now enumerate the crash windows (SIGKILL of the source at any instruction boundary):

| Crash point | RAM retry | Outbox on disk | QUIC sent? | Recovery |
|---|---|---|---|---|
| before `assign_and_retain` | — | — | no | item still `Crossing`; re-emitted next tick after restart (idempotent by `(transfer, BATCH_STEP)`). No loss. |
| after retain, before `outbox.commit()` fsync | present | **absent** | no | item still `Crossing` (the flip to `Held{outbound}` is ALSO gated — see below); re-emitted. No loss. |
| after outbox fsync, before QUIC send | present | **present** | no | boot replay (§3) re-sends from disk. **This is the window the outbox closes.** |
| after QUIC send, before dest adopts | present | present | yes | boot replay re-sends; dest dedups by `(transfer, BATCH_STEP)` OR receiver seq-dedup. At-least-once double-send, harmless. |
| after dest adopts + acks | retired on ack | deleted-through on ack | yes | nothing to replay. |

**The critical coupling — the `Crossing → Held{outbound}` flip must be co-durable with the outbox write, OR gated behind it.** If the source flips the item to `Held{outbound}` (marking it "emitted, awaiting release") but crashes before the outbox fsync, on restart the item is `Held{outbound}` (in the source's OWN durable state — but transients are NOT durable today, they die with the process; see ⟦A1⟧) and would never be re-emitted, yet the batch never reached disk. **Resolution:** for the source shard, transients are **Transient by policy = not persisted** (HR2: Durable-vs-Transient is the policy fan-out). A Transient source that crashes LOSES its `Held{outbound}` in-RAM status entirely — the item is simply gone from the source. So the "item flipped but batch not durable" divergence cannot strand anything at the source: the source has no memory of it. The ONLY residual is at the DEST/orchestrator: did the orchestrator self-promote a dest that never got the batch? That is closed by the saga-side gate (§4) + the outbox replay (§3) together.

**Conclusion on atomicity:** the non-atomic window is safe because (a) there is no above-seam saga txn for the source's emit to be atomic with; (b) the outbox `commit()` fsync is ordered *before* the QUIC send via the same parked-flush gate the orchestrator already runs; (c) at-least-once + receiver dedup absorbs the "durable but double-sent" window; (d) the "durable-but-source-forgot" window is empty because a crashed Transient source retains nothing. **We get a property strictly stronger than "atomic with the step": the frame is on disk before it can ever be observed as sent, and re-sent deterministically on boot.**

### 1.3 WHICH flows get durability — producer-less-only, scoped WITHOUT an HR3 fork

**Ruling: ONLY producer-less reliable one-shots get the outbox, NOT all reliable traffic.** Rationale: fsync-per-reliable-send on the hot path is unacceptable (the orchestrator's saga control lane would fsync on every `Demote`/`Promote`), and it is *unnecessary* — every saga-DRIVEN reliable flow re-drives on orchestrator restart via `scan_deadlines` (re-execute the step, re-send). The prompt's own framing confirms this: "Saga-DRIVEN flows re-drive on orchestrator restart... those do NOT need a transport outbox; ONLY producer-less one-shots do."

**The two producer-less reliable one-shots that need it (the closed set):**
1. `TransientBatch` (source shard → dest, `MsgClass::Saga`, `stub.rs:2134`) — the D-6 #1 canonical case.
2. `GhostFlow::Despawn` (band-exit, `MsgClass::GhostReliable`, `stub.rs:1525`) — the R-5 `mesh_under_loss` proved this at-least-once *for a source that stays up*; the source-crash residual is the same class.

**How the scope is expressed without an HR3 per-flow / per-shard-kind fork:** a single boolean **`durable: bool`** field on the outbound staging tuple — a property of the *individual send*, decided at the ONE `push_flow` call site by the caller, NOT a match on shard kind or a fork of `ReliableLaneSender`. There is still exactly ONE `ReliableLaneSender` FSM and ONE `OutboxSink` backend. The FSM's `assign_and_retain` gains one parameter `durable: bool`; when true it write-throughs, when false it does not. This is the SAME shape as `MsgClass` being a parameter to `push_flow` (`runtime.rs:59`, "class is a PARAMETER, never hardcoded") — the caller states durability deliberately, per send. No feature ever matches on shard kind; a shard and the orchestrator both call the identical `push_flow(dest, class, flow, Durability::Retained)` for their producer-less one-shots.

⟦A2⟧ **Assumption:** the set of producer-less one-shots is closed at these two today and grows only additively (a future producer-less flow adds a `Durability::Retained` at its push site). If a new reliable flow is producer-less and its author forgets the marker, the loss is silent — mitigated by a conformance test (§7) that asserts every `MsgClass::Saga`/`GhostReliable` flow WITHOUT a `scan_deadlines` re-driver carries the marker. Flagged as the weakest point of the scope model.

---

## 2. THE `OutboxSink` SEAM

### 2.1 Where it lives + the trait

**Lives in io-prod** (`crates/io-prod/src/outbox.rs`, new), NOT in `sim::io`. It is a transport-internal seam, invisible to sim/node — HR1-clean (§8). The frozen `sim::io::Store` trait is NOT extended; instead the io-prod outbox *composes* a `RedbStore` internally (the same type, reused). Rationale: `Store` is the ABOVE-seam durable-state trait; `OutboxSink` is a BELOW-seam transport concern that happens to persist via redb. Keeping them distinct types (both backed by `RedbStore`) is what keeps the seam frozen.

```rust
// crates/io-prod/src/outbox.rs
/// The durable write-through mirror of a lane's RAM retry buffer (R-6d). One row per
/// un-acked producer-less reliable frame, keyed by (peer, class, incarnation, seq).
/// Object-safe (used as &dyn) so ReliableLaneSender holds `Option<&dyn OutboxSink>` with
/// NO per-monomorphization region gotcha (HR5). ALL branching lives in the monomorphic
/// backend impl; the trait body is straight-line.
pub trait OutboxSink: Send {
    /// Write-through one retained frame. Called by assign_and_retain AFTER the RAM insert
    /// succeeded (so the RAM map is always a superset of the disk mirror — a crash between
    /// the two re-sends at most a frame the disk lacks, never strands one). STAGED; not
    /// durable until `commit`.
    fn retain(&mut self, key: &OutboxKey, framed: &[u8]);
    /// Delete-through the acked prefix. Called by on_ack for each retired seq. STAGED.
    fn release(&mut self, key: &OutboxKey);
    /// THE durability barrier — fsync every staged retain/release. Returns the durable seq
    /// the caller waits through before sending (the parked-flush gate, §5).
    fn commit(&mut self) -> u64;
    /// Boot rehydrate: every un-acked row for THIS node, ascending by key. The replay set.
    fn scan_all(&self) -> Vec<(OutboxKey, Vec<u8>)>;
}
```

⟦A3⟧ **Judgment call — `retain` write-throughs AFTER the RAM insert, not before.** Rejected alternative: disk-first. Disk-first would guarantee "disk ⊇ RAM," meaning a crash could leave a disk row with no RAM counterpart, which boot replay handles fine — but it fsyncs on the hot insert path even when the frame is about to be rejected (`BufferFull`/`Unframable`). RAM-first means the mirror is a SUBSET of RAM within a tick, reconciled to equality at `commit()`. Since the send is gated behind `commit()` (§1.2), the frame never leaves before the mirror is durable regardless — so RAM-first is correct AND avoids mirroring un-retained frames. Chosen.

### 2.2 How `ReliableLaneSender` holds it

`ReliableLaneSender` does NOT hold the sink — the sink is per-NODE, not per-lane (one redb file, many lanes). The sink is threaded into the writer task's per-peer state (`replay_lanes`/`write_frame`, alongside `lanes: &mut BTreeMap<MsgClass, ReliableLaneSender>`). The FSM methods take it as a parameter so the FSM stays pure/synchronous (unit-testable without tokio):

```rust
// mesh.rs — assign_and_retain gains two params (durable flag + optional sink):
fn assign_and_retain(
    &mut self, from: NodeId, class: MsgClass, bytes: &[u8],
    durable: bool, sink: Option<&mut dyn OutboxSink>,   // NEW
) -> Result<u64, AssignReject> {
    // ... unchanged RAM logic through the insert at line 597 ...
    self.retry.insert(seq, RetainedFrame { frame, framed_len });
    self.retry_bytes += framed_len as usize;
    self.next_seq += 1;
    // NEW — write-through, branchless-generic shim: the ONE if is monomorphic here, not
    // inside a generic. `encoded` is the SAME bytes the cap check already produced (H3 reuse).
    if durable {
        if let Some(sink) = sink {
            sink.retain(&OutboxKey::new(self.peer, class, self.incarnation, seq), &encoded);
        }
    }
    Ok(seq)
}
```

`on_ack` gains the same optional sink and deletes-through each retired seq inside its existing retire loop (`mesh.rs:545-551`):

```rust
fn on_ack(&mut self, ack_incarnation: u64, ack_epoch: u32, ack_through: u64,
          mut sink: Option<&mut dyn OutboxSink>) -> usize {          // NEW param
    if ack_incarnation != self.incarnation || ack_epoch != self.epoch { return 0; }
    let base_before = self.base;
    while self.base < self.next_seq && self.base <= ack_through {
        if let Some(rf) = self.retry.remove(&self.base) {
            self.retry_bytes -= rf.framed_len as usize;
            if let Some(s) = sink.as_deref_mut() {                    // NEW delete-through
                s.release(&OutboxKey::new(self.peer, rf.frame.class, self.incarnation, self.base));
            }
        }
        self.base += 1;
    }
    (self.base - base_before) as usize
}
```

⟦A4⟧ `ReliableLaneSender` must learn its own `peer: NodeId` to build the key (it does not store it today — the peer is the `BTreeMap<NodeId, …>` key in the writer). Add `peer: NodeId` to the struct (set in `new`). One field, no behavior change to the RAM path.

**The staged retain/release are flushed at `sink.commit()` once per flush-tick, batching all lanes' write-throughs into ONE fsync** — NOT one fsync per frame. This is the hot-path cost containment: a burst of N producer-less frames in one tick = ONE fsync.

### 2.3 Exact key encoding + value

`OutboxKey` = a fixed-layout big-endian byte tuple, so redb's ascending key order == ascending `(peer, class, incarnation, seq)` order (making `scan_all` return replay-ready order, and a prefix scan enumerate one peer/class cleanly):

```
outbox key bytes (28 bytes, all big-endian for lexicographic == numeric order):
  [0]        : 0x4F  ('O') — the outbox key-family tag (one prefix tag per key family,
                              per the Store contract note sim/io/mod.rs:352; separates
                              outbox rows from any co-tenant redb table if the file is shared)
  [1..9)     : peer   NodeId as u64 BE          (8 bytes)
  [9]        : class  MsgClass as u8            (1 byte; the closed enum's discriminant)
  [10..18)   : incarnation u64 BE              (8 bytes)
  [18..26)   : seq    u64 BE                    (8 bytes)
  → 26 bytes total.  encode/decode in one monomorphic OutboxKey::{new,to_bytes,from_bytes}
    (NOT a generic — HR5), unit-tested round-trip.

value = the framed ReliableFrame bytes EXACTLY as encode_frame produced them
        (mesh.rs:588 `encoded`) — the SAME buffer the cap check + framed_len used (H3: one
        encode, three consumers). Opaque to the outbox: it is already-serialized wire bytes,
        never decoded by the outbox (HR1 — the outbox holds opaque InterShardFlow frame
        bytes, never a sealed World; §8).
```

`scan_all()` = `redb scan over prefix [0x4F]` → every un-acked outbox row for this node, ascending → exactly `(peer, class, incarnation, seq)` order. A per-lane enumeration on boot is a prefix scan over `[0x4F][peer][class]` (used by the boot replay to rebuild each lane, §3).

⟦A5⟧ **Judgment call — store the WORST-CASE-epoch framed bytes (epoch = u32::MAX), NOT the live-epoch bytes.** `assign_and_retain` already encodes at `epoch = u32::MAX` for the cap check (`mesh.rs:583`), then overwrites `frame.epoch = self.epoch` for the RAM copy (`mesh.rs:596`). The `encoded` buffer we mirror is the u32::MAX one. But boot replay re-stamps epoch to 0 anyway (§3), so the *stored* epoch is irrelevant — we re-encode at epoch 0 on replay from the decoded frame. **Simpler + correct alternative chosen:** store the frame's *semantic* fields (from, class, incarnation, seq, bytes) as the value, and let boot re-frame at epoch 0. Rejected storing raw framed bytes because the stored epoch would be a stale u32::MAX that boot must strip anyway. So the value is `postcard(ReliableFrame{epoch: 0, ..})` — canonical, replay-ready, no re-stamp needed. This keeps the outbox value = "the frame as it will be replayed," zero transform on read.

---

## 3. THE BOOT REPLAY

Ordering (in the source bin's `main`, before the tick loop, after `spawn_mesh`):

```
1. incarnation = resolve_process_incarnation(&env)   // R-6a BootCounter bump — STRICTLY
                                                      //   higher than any prior boot (M3)
2. (transport, control) = spawn_mesh(cfg{ process_incarnation: incarnation, .. })
3. outbox = RedbStore-backed OutboxSink::open(path)   // §5
4. replay = outbox.scan_all()                         // every un-acked row, ascending
5. for (key, frame_bytes) in replay:
       transport.inject_replay(key.peer, key.class, frame_bytes)   // NEW injection API, §3.1
6. → enter the tick loop; the FIRST flush re-drives the RAM retry buffers over fresh lanes
```

### 3.1 Why the replay is deterministically ACCEPTED (never Gap/Dedup/Stale)

The replay does NOT re-use the crashed process's seqs directly against the receiver's stale watermark. Instead it **re-seeds fresh lanes at epoch 0 with ascending seq re-based from 0**, exploiting the A1 incarnation-reset ladder (`classify_reliable`, `mesh.rs:358-386`):

- **New incarnation dominates (A1, `mesh.rs:362`).** Because `resolve_process_incarnation` bumped the BootCounter (M3, R-6a), every replayed frame carries `incarnation > st.incarnation` at the receiver. Line 362-369: the receiver *resets* `RecvState` to `{incarnation, epoch: <frame epoch>, hw: 0, primed: false}` and falls through to `prime_or_contiguous(st, seq, fresh_incarnation=true)`. The stale watermark from the dead process is wiped.
- **Epoch 0 + seq re-based from 0 primes at the seq0 gate (`mesh.rs:394-402`).** The boot replay rebuilds each lane fresh via `ReliableLaneSender::new(incarnation, cap)` → `epoch = 0, next_seq = 0, base = 0`. It then feeds the recovered frames back through `assign_and_retain` **in ascending original-key order**, which RE-ASSIGNS them contiguous seqs `0, 1, 2, …` (the original crashed seqs are discarded; only order is preserved). The first replayed frame is seq 0 at a fresh incarnation → `prime_or_contiguous` with `seq == 0, fresh_incarnation == true` → hw=0, primed=true, returns `Verdict::Reset` (Accepted). Subsequent frames are seq 1, 2, … contiguous → `seq == hw + 1` → `Verdict::Accept`. **No Gap** (contiguous from 0), **no Dedup** (fresh incarnation reset hw to 0 first), **no Stale** (incarnation is strictly higher; epoch 0 ≥ the reset epoch).

This is exactly the both-ends-restart property R1's LOW flagged and the sketch's "epoch 0, ascending seq re-based from base" prescribes.

### 3.2 The injection API (additive, io-prod-internal)

`inject_replay` is NOT a `Transport` seam method (it must not appear in the frozen `sim::io::Transport`). It is a method on the concrete `MeshTransport` / `MeshControl`, called only by the bin's boot sequence:

```rust
// mesh.rs — concrete MeshTransport, NOT the Transport trait:
impl MeshTransport {
    /// R-6d boot replay: re-seed a lane with a recovered producer-less frame. Rebuilds the
    /// lane at the CURRENT (bumped) incarnation, epoch 0, re-based seq. Idempotent by
    /// construction — feeding the same scan twice re-assigns the same contiguous seqs.
    pub fn inject_replay(&mut self, peer: NodeId, class: MsgClass, frame_bytes: Bytes) { .. }
}
```

Because the value is `postcard(ReliableFrame{epoch:0,..})` (§2.3, ⟦A5⟧), injection decodes the frame, extracts `bytes`, and routes it through the normal reliable send path with `durable = true` — so the re-sent frame is ALSO re-mirrored to the fresh-incarnation outbox rows (the old-incarnation rows are GC'd once acked; see §3.3). This makes boot replay *itself* crash-safe (a second crash mid-replay re-replays).

### 3.3 Old-incarnation row GC

Replayed frames get NEW keys `(peer, class, new_incarnation, re-based_seq)`. The OLD `(peer, class, old_incarnation, *)` rows are now dead weight. **GC rule:** after `scan_all()` on boot, once a replayed frame's fresh-incarnation ack retires it, `on_ack` deletes its NEW row; the OLD rows are swept in ONE pass by `outbox.gc_below(new_incarnation)` — a prefix-range delete of every row whose incarnation `< new_incarnation`, run once, right after step 4, committed with the first flush. ⟦A6⟧ This is safe because a strictly-monotone incarnation (M3) guarantees no live frame ever carries `incarnation < new_incarnation`. Without M3 this GC would be unsound — **hard dependency on R-6a/b/c landed** (it is).

---

## 4. THE AwaitAdopt CLOSURE — transport outbox ALONE is NOT sufficient

**Ruling: the durable outbox is NECESSARY but NOT SUFFICIENT. A precise saga-side change is ALSO required.** Here is the proof and the exact change.

### 4.1 Why the outbox alone leaves a hole

The outbox guarantees: *if the source crashed in the "durable-but-not-acked" window, on restart it re-sends the `TransientBatch`.* Good. But the D-6 #1 loss is a RACE between two independent recoveries:
- **Orchestrator's recovery:** `scan_deadlines` sees `is_confirmed_dead(ctx.source)` (gated on M3+L5) and injects `SourceUnreachable`, which at `saga.rs:915-918` self-promotes the dest **from any BatchHandoff phase including `AwaitAdopt`** — even though in `AwaitAdopt` the dest may never have received the batch.
- **Source's recovery:** the source restarts (a new incarnation) and re-sends the batch from its outbox.

If the orchestrator's confirm-dead fires and self-promotes BEFORE the restarted source's replay reaches the dest, the orchestrator promotes an empty dest, tombstones the saga (`saga.rs:917 A::Tombstone`), and when the source's replayed batch finally lands, the saga is gone — the batch is adopted as `Arriving` but never promoted → **silent loss / a leaked orphan** ⟦A7⟧. The outbox re-sent the bytes; the saga had already given up. So confirm-dead + outbox alone still race.

### 4.2 The required saga-side change (precise)

**Phase-gate the `SourceUnreachable` self-promote so it NEVER fires from `AwaitAdopt`.** The `saga.rs:915` arm currently matches `S::BatchHandoff { new_fence, .. }` — the `..` swallows the phase. Split it:

```rust
// REPLACE the single arm at saga.rs:915-918 with two:

// POST-ADOPT phases (AwaitRelease/AwaitPromote/AwaitComplete): the dest DEMONSTRABLY holds
// the batch (it acked BatchAdopted to leave AwaitAdopt), so the go-token is the sole promote
// authority → self-promote (zero loss). Unchanged behavior for these three phases.
(S::BatchHandoff { new_fence, phase }, E::SourceUnreachable)
    if phase != P::AwaitAdopt => (
        S::Done { new_fence },
        vec![A::EmitTransientPromote { fence: new_fence }, A::Tombstone],
    ),

// AwaitAdopt: the dest has NOT proven it holds the batch (no BatchAdopted yet). A dead source
// here MUST NOT self-promote an empty dest. Instead ABORT-DOWN: the batch is an in-flight
// transient LOSS-within-budget (the source is dead and its outbox replay — if any — will find
// no saga and be a harmless orphan the dest GC's). This is the SAME accounted-loss shape as
// the dead-DEST DestUnreachable arm (saga.rs:919), NOT a self-promote.
(S::BatchHandoff { new_fence, phase: P::AwaitAdopt }, E::SourceUnreachable) => (
    S::Done { new_fence },
    vec![A::EmitTransientAbandon { fence: new_fence }, A::Tombstone],  // count the drop, loud
),
```

**Why abort-down, not "wait for the source's replay"?** Because `AwaitAdopt` has no orchestrator egress to re-drive (the comment at `saga.rs:884-886`), the orchestrator cannot *solicit* the batch — only the source can re-emit it. If the source is CONFIRMED DEAD (M3+L5), it is not coming back with THIS incarnation; a restarted source is a NEW incarnation whose re-emit will (correctly) start a fresh saga or re-drive via a fresh go-token ⟦A8⟧. So the honest resolution when the source is confirmed dead mid-`AwaitAdopt` is: **the batch is lost-within-budget, counted loud via `EmitTransientAbandon`, never a silent self-promote of a batch that was never delivered.** The durable outbox's job is the OTHER branch: a source that RESTARTS (not permanently dead) re-sends from disk and the dest adopts → the confirm-dead never fires because the source came back.

**This is the "self-promote arm should be gated to post-adopt phases" fix that DEFERRED.md:965-966 explicitly names as the alternative to the outbox** — and the correct answer is **BOTH**: the outbox makes the restart path deliver, the phase-gate makes the confirmed-dead path loud-not-silent. Together they eliminate silent loss in every `AwaitAdopt` crash interleaving.

### 4.3 The AwaitAdopt Timeout re-solicit (the 2d-pattern companion, NOT strictly required but recommended)

DEFERRED.md:963 mentions "the 2d re-solicit-egress (an `AwaitAdopt` Timeout re-prompting the source to re-emit)." With the durable outbox, this is **redundant for the source-crash case** (the source's own boot replay re-sends; the orchestrator need not prompt it) but **valuable for the source-ALIVE-but-lost-envelope case** (a blip that the transport's own R-4a idle timer already covers — so also redundant). **Ruling: do NOT add the re-solicit egress.** The outbox (restart path) + R-4a transport timer (blip path) + the phase-gate (confirmed-dead path) form a complete cover. Adding a saga-side re-solicit would be a THIRD redundant recovery for a window already doubly covered — rejected as over-engineering (the memory's "reject over-engineering" discipline). Flagged so a reviewer can challenge: if `AwaitAdopt` ever gains a non-transient (Durable) subject, re-examine.

---

## 5. THE gateway/shard PER-NODE RedbStore

### 5.1 Where the store lives + path/guard

The SOURCE of the producer-less one-shots is the SHARD (`TransientBatch`, `GhostFlow::Despawn`). The GATEWAY currently emits no producer-less reliable one-shot (its saga control is orchestrator-driven), so **strictly only the SHARD needs the outbox today** ⟦A9⟧. But the vetted design (and DRY) says both open a per-node store; we open it in BOTH `shard.rs` and `gateway.rs` via a shared bins helper so the wiring is HR3-uniform (one mechanism, no per-kind fork), and the gateway's outbox is simply empty until it ever has a producer-less flow.

```rust
// crates/bins/src/lib.rs — new shared helper (mirrors resolve_process_incarnation):
pub fn open_node_outbox(env: &EnvConfig, local: NodeId)
    -> Result<Option<vd_io_prod::outbox::NodeOutbox>, Box<dyn Error>>
{
    // VD_OUTBOX_PATH present ⇒ open a durable outbox; ABSENT ⇒ None (dev/test/harness path —
    // the outbox is prod-only, exactly like VD_STORE_PATH is on the orchestrator). No silent
    // in-memory outbox: absence = "no durable outbox," a conscious dev choice.
    let Ok(path) = env.string("VD_OUTBOX_PATH") else { return Ok(None); };
    let durable_root = env.string("VD_STORE_DURABLE_ROOT").ok().map(PathBuf::from);
    let ephemeral_ok = strict_bool(env, "VD_STORE_EPHEMERAL_OK")?;   // reuse R-6a ladder
    // REUSE the R-6a shared guard (boot.rs:120) — one path-durability policy for
    // VD_STORE_PATH, VD_BOOT_STATE_DIR, and now VD_OUTBOX_PATH (DRY).
    vd_io_prod::boot::check_durable_path(Path::new(&path), durable_root.as_deref(), ephemeral_ok)?;
    Ok(Some(vd_io_prod::outbox::NodeOutbox::open(Path::new(&path))?))
}
```

- **Path env:** `VD_OUTBOX_PATH` (its own redb file, e.g. `<pvc>/node-<id>.outbox.redb`), distinct from the orchestrator's `VD_STORE_PATH` so a shard and orchestrator on one host never share a file.
- **Ephemeral guard:** reuses `boot::check_durable_path` (`boot.rs:120`) verbatim — the durable-root ALLOW-list (R-6a) applied to a third path. This is DRY leverage the sketch explicitly called out ("more DRY leverage").
- **Drop/fsync discipline:** the `NodeOutbox` wraps a `RedbStore` (`store.rs:189`) with its off-tick writer thread + `DurabilityHandle`. `Drop` (`store.rs:609`) drops the sender → writer drains-then-exits → graceful fsync of any staged batch. The parked-flush gate (§1.2) ensures nothing is sent before durable, so a hard SIGKILL (no Drop) loses at most the un-fsynced staged batch, which by the gate has NOT been sent → re-emitted next boot. No loss.

### 5.2 This is its own gate-able prerequisite sub-step

Opening the per-node store touches nothing about the outbox FSM — it is purely bin wiring + env plumbing + the guard reuse. It lands and gates as **R-6d1** (§6) with its own test (a shard boots with `VD_OUTBOX_PATH`, the file appears, a temp path is rejected loud, absence yields `None`). It is a hard precondition for R-6d2 (the FSM has nowhere to write without it) but independently verifiable.

---

## 6. SLICING (dependency order, each independently gate-able)

**R-6d1 — per-node RedbStore for gateway/shard.**
- `open_node_outbox` bins helper + `VD_OUTBOX_PATH`/`VD_STORE_DURABLE_ROOT` plumbing into `shard.rs` and `gateway.rs` `main`; reuse `boot::check_durable_path`; `NodeOutbox::open` in io-prod wrapping `RedbStore`.
- Split the shard bin's tick loop from `step_tick()` into `run_schedule`/`wait_durable`/`flush_outbox` (the orchestrator's parked-flush pattern, `orchestrator.rs:297-324`), GATED on `outbox.is_some()` — a `None` outbox keeps the byte-identical `step_tick()` fast path.
- **Test:** `crates/bins/tests/node_outbox_boot.rs` — shard boots with `VD_OUTBOX_PATH` under a temp dir + `VD_STORE_EPHEMERAL_OK=1` (file created); temp path WITHOUT the escape → boot refused loud; absent `VD_OUTBOX_PATH` → boots, `step_tick` fast path unchanged. Gate-able alone.

**R-6d2 — `OutboxSink` seam + FSM write-through/delete-through.**
- `crates/io-prod/src/outbox.rs`: `OutboxSink` trait, `OutboxKey` (26-byte encode/decode, monomorphic), `NodeOutbox` impl over `RedbStore`.
- `ReliableLaneSender`: add `peer` field; `assign_and_retain(.., durable, sink)` write-through; `on_ack(.., sink)` delete-through; thread `durable: bool` from a new `Durability` marker on the outbound tuple / `push_flow`; wire the sink through `write_frame`/`replay_lanes`/`ack_egress` fan-out.
- **Test:** FSM unit tests (synchronous, no tokio) — `assign_and_retain` with `durable=true` + a `MockOutboxSink` records the exact `OutboxKey` bytes + value; `on_ack` releases the retired prefix's keys; `durable=false` records nothing; round-trip `OutboxKey`. The both-lanes-one-fsync batching. Gate-able alone (no boot, no process).

**R-6d3 — boot replay + the saga-side AwaitAdopt closure.**
- `MeshTransport::inject_replay` + `NodeOutbox::scan_all`/`gc_below`; the bin boot sequence (scan → inject → gc) before the tick loop.
- The `saga.rs:915` phase-split (§4.2) — a PURE FSM change (Tier-A sim).
- **Test:** saga FSM unit tests for the two new arms (`AwaitAdopt + SourceUnreachable → Abandon+Tombstone`; `AwaitRelease/Promote/Complete + SourceUnreachable → Promote+Tombstone` unchanged); a synchronous boot-replay test that seeds a `NodeOutbox`, calls scan→inject, asserts the fresh lanes carry the recovered frames at epoch 0 seq 0.. ; a `classify_reliable` test proving the injected replay Accepts (Reset then Accept, gap_drop=0) at a bumped incarnation.

**R-6d4 — the both-ends-restart proptest + the SIGKILL-source process proof (the NAMED acceptance gate).**
- The `proptest` (§7.1) — pure, over the FSM + receiver ladder.
- The process proof (§7.2) — extends `mesh_under_loss.rs` + `orchestrator_crash.rs`'s SIGKILL pattern to a SOURCE crash mid-`AwaitAdopt`.
- **Gate:** both green + Tier-A saga at 100% + io-prod Tier-B floor held.

---

## 7. TEST PLAN

### 7.1 The NAMED both-ends-restart replay-interleave proptest (`io-prod`, pure)
`proptest_both_ends_restart_replay_interleave` — the R-6d acceptance gate the sketch names.
- **Model:** a sender lane's un-acked window `[base..next_seq)` of framed frames; a `NodeOutbox` mirror; a receiver `RecvState`. Generate: an arbitrary interleaving of (source-restart at a strictly-higher incarnation + full outbox replay) and (receiver-restart resetting `RecvState` to genesis) and (acks retiring an arbitrary prefix), in any order, any number of rounds.
- **Asserts:** (1) every frame that was ever `retain`ed-and-committed and NOT yet `release`d is delivered to the receiver's sim AT LEAST ONCE across the whole interleaving (no silent loss); (2) `gap_drop == 0` on every replay (the epoch-0/seq-0-rebase contiguity holds under every reorder); (3) no frame is delivered as a NEW sim message more than the dedup allows (at-least-once, receiver dedups exact dups); (4) the outbox mirror ⊆ the RAM retry at every commit point (the subset invariant, ⟦A3⟧); (5) after all acks land, `scan_all()` is empty AND `gc_below` swept every stale-incarnation row.

### 7.2 The SIGKILL-source-mid-AwaitAdopt process proof (`bins`, real QUIC)
`sigkill_source_in_await_adopt_redelivers_the_batch` — extends `orchestrator_crash.rs`'s RAII-reaped SIGKILL pattern + `mesh_under_loss.rs`'s producer-less-flow assertion, to a SOURCE crash.
- **Setup:** a source shard (with `VD_OUTBOX_PATH`, a fixed dir NOT reaped), a dest shard, an orchestrator. Drive a real `TransientBatch` crossing (the `store-test-hooks` sentinel pauses the source's outbox writer AFTER the fsync but BEFORE the QUIC send — the exact §1.2 "durable-but-not-sent" window, mirroring the orchestrator's mid-fsync sentinel `orchestrator.rs:195-207`).
- **Kill + restart:** SIGKILL the source in that window; reap (releases the redb lock); restart the source on the SAME `VD_OUTBOX_PATH` with a bumped BootCounter incarnation.
- **PASS assertions (via `/metrics` MeshStats + the dest's admin snapshot):** after restart, the dest ADOPTS the batch exactly once (`transients_adopted == item_count`, `TransientBatch` redeliveries counted but not re-adopted); `gap_drop == 0`, `dedup_drop` does not climb on the replayed seq0.. ; the `boot.counter` incremented; the saga reaches `Done` via the normal adopt→promote path (NOT via `SourceUnreachable`). 
- **The phase-gate companion proof:** a SECOND scenario where the source is SIGKILLed mid-`AwaitAdopt` and NEVER restarted (permanent death) → assert the saga resolves via `AwaitAdopt+SourceUnreachable → EmitTransientAbandon+Tombstone` (the loss is COUNTED loud, `transients_abandoned` climbs), NOT a silent self-promote of an empty dest.
- **RED control:** the same test WITHOUT the outbox (`VD_OUTBOX_PATH` unset) must show the batch LOST after the source crash (dest never adopts) — proving the outbox is load-bearing, not theater (mirrors the R-6b RED control).

### 7.3 FSM unit tests + HR5 coverage story
- **Tier-A (sim/saga, 100% region+branch):** the two new `saga.rs` arms exercised with `assert_eq!` on the exact `(SagaState, Vec<SagaAction>)` (no `matches!` false-arm regions, per HR5(d)); every phase×event combo of the split covered. `classify_reliable`/`prime_or_contiguous` gain no branches (unchanged) — the boot-replay determinism rides the EXISTING A1 ladder, already 100%.
- **io-prod (Tier-B, ratcheted floor 90):** the `OutboxSink` write-through/delete-through is the new logic. HR5(a) discipline: `assign_and_retain`/`on_ack` stay branchless-generic — the `if durable { if let Some(sink) }` is a MONOMORPHIC straight-line guard (no generic body), fully covered by `durable=true`/`durable=false` × `sink=Some`/`sink=None` unit cases. `OutboxKey` encode/decode is monomorphic, round-trip covered. `NodeOutbox` over `RedbStore` reuses the already-covered store; only the thin key/scan glue is new surface. Ratchet the io-prod floor UP by the outbox module's coverage (the module is small + fully unit-testable synchronously via a `MockOutboxSink` for the FSM and a real temp `RedbStore` for `NodeOutbox`).

---

## 8. HR + SEAM COMPLIANCE

**HR1 (sealed shards) — CLEAN.** The outbox value is opaque already-serialized `InterShardFlow` frame bytes (the reviewed `vd-wire` framing), NEVER a shard's `World`/rapier/redb, NEVER decoded by the outbox. The `OutboxSink` trait never sees a domain type — only `(OutboxKey, &[u8])`. A shard's sealed sim state is untouched; the outbox is a transport-private durable buffer of wire-frames-in-flight. No inter-shard byte exists that is not already an `InterShardFlow` arm. Confirmed not a leak.

**HR3 (one tooling) — CLEAN.** ONE `ReliableLaneSender` FSM (gains params, not a fork); ONE `OutboxSink` seam with ONE `NodeOutbox` backend; ONE `push_flow` with a `Durability` PARAMETER (never a match on shard kind); the shard and orchestrator call the identical helper for their producer-less one-shots. No `match` on `ShardProfile`/`NodeKind` anywhere in the outbox path. The per-node store opens via ONE `open_node_outbox` helper for both bins.

**HR5 (coverage) — MET.** Tier-A sim/saga stays 100% (the saga arms use `assert_eq!` equality; the split adds only coverable arms). io-prod Tier-B floor held/ratcheted; the new FSM logic is synchronous + unit-testable with a `MockOutboxSink`, and the branching is monomorphic straight-line per HR5(a). The both-ends proptest is the named acceptance gate.

**Frozen `sim::io` seam — TOUCHED, additive + justified + lockstep.** The `Store`/`Transport`/`Inbound` traits are NOT modified. The ONE additive touch is a `Durability` marker on the outbound staging tuple (`OutboundBox` / `push_flow`, in `vd-sim`) — an above-seam producer-intent bit, NOT a new `Transport` method, NOT a change to `send`'s signature. It is additive (a new field defaulting to non-durable, so every existing call site is behavior-identical) and lockstep (added with its ONE consumer, the io-prod write-through, in R-6d2). `inject_replay` is a method on the CONCRETE `MeshTransport`, invisible to the frozen `Transport` trait. Justified: the alternative (a full `send_durable` seam method) is heavier and would force every `Transport` impl (including `MemTransport`) to implement durability semantics it does not have — the marker keeps the seam minimal.

**No new dependency — MET.** The durable tier is the transactional-OUTBOX PATTERN over the EXISTING `redb` `RedbStore` (`store.rs:189`); no oxide-outbox/broker. The `OutboxSink` is a new trait, not a new crate.

**20 Hz unreliable datagram hot path — BYTE-IDENTICAL.** The outbox is reliable-lane-only (`assign_and_retain` is reliable-only, `mesh.rs:1528-1529`; the `Unreliable` arm at `mesh.rs:1657` is UNTOUCHED). No `durable` flag, no fsync, no key encoding touches the datagram path. `write_frame`'s `Reliability::Unreliable` arm is unchanged. Confirmed zero cost on the 20 Hz path.

---

## Flagged assumptions + rejected alternatives (index)

- ⟦A1⟧ Transients are not persisted at the source (Transient policy = die-with-process); this is what makes the "durable-but-source-forgot" window empty. If a future Durable subject ever rides `AwaitAdopt`, §4.3 must be revisited.
- ⟦A2⟧ The producer-less-one-shot set is closed at `{TransientBatch, GhostFlow::Despawn}` and grows only additively with an explicit `Durability::Retained` marker; a conformance test guards forgetfulness (weakest point of the scope model).
- ⟦A3⟧ Chose RAM-first write-through (mirror ⊆ RAM within a tick) over disk-first; rejected disk-first (fsyncs frames about to be rejected).
- ⟦A4⟧ `ReliableLaneSender` gains a `peer` field to build the key.
- ⟦A5⟧ Store the frame as `postcard(ReliableFrame{epoch:0,..})` (replay-ready) not the u32::MAX-framed cap-check bytes; rejected raw framed bytes (stale epoch to strip).
- ⟦A6⟧ Stale-incarnation-row GC is sound only under M3's strict monotonicity — a hard dep on R-6a/b/c (landed).
- ⟦A7⟧ The residual loss without the saga-side gate is a self-promote of an empty dest + a later orphaned adopt.
- ⟦A8⟧ A confirmed-dead source mid-`AwaitAdopt` is resolved as accounted loss (`EmitTransientAbandon`), NOT waited-for; a RESTARTED source is a new incarnation that re-drives cleanly.
- ⟦A9⟧ Only the SHARD needs the outbox today; the gateway opens one for HR3 uniformity (empty until it ever has a producer-less flow).

**Rejected shapes:** (B) saga-runtime-owned outbox (infeasible — the seq/key does not exist above the seam + would leak frame identity up, HR1 violation); (C) full new `send_durable` seam method (heavier than the additive marker, forces durability semantics onto `MemTransport`); the saga-side `AwaitAdopt` Timeout re-solicit egress (redundant — the outbox restart path + R-4a blip timer + phase-gate already cover the window; over-engineering).

---

Files/lines the reviewers will want: sketch `scripts/r6_vetted_design.md:107-152`; the crux code `crates/sim/io/mod.rs:328-369` (frozen seam) + `crates/io-prod/src/mesh.rs:566-625` (FSM) + `crates/sim/src/saga.rs:815-922` (the AwaitAdopt/SourceUnreachable arms to split at :915) + `crates/sim/src/stub.rs:2070-2137` (the source emit) + `crates/bins/src/bin/orchestrator.rs:273-324` (the parked-flush pattern to replicate in shard.rs) + `crates/io-prod/src/boot.rs:120` (the guard to reuse) + `crates/bins/src/bin/shard.rs` (bin to split) + `crates/io-prod/src/store.rs:189,397,609` (`RedbStore` to wrap). New files: `crates/io-prod/src/outbox.rs`, `crates/bins/tests/node_outbox_boot.rs`, the both-ends proptest in io-prod, the SIGKILL-source proof extending `crates/bins/tests/orchestrator_crash.rs`.