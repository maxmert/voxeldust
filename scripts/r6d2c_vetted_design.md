# R-6d2c Vetted Design of Record — FSM durable write-through/delete-through

Workflow wf_5ca0a751 (design + 3 adversarial reviews + synth). Verdict SOUND_TO_IMPLEMENT (1 HIGH [missing io-prod Retained-through-write_frame test T9] + 1 MEDIUM [outbox_key helper] folded).

## SYNTHESIS (design of record)

All claims are now verified against the actual code. Key confirmations:

1. **`ReliableFrame.bytes` is `pub(crate)`** (lib.rs:64 struct is `pub(crate)`, field :81 is private-in-crate) → reachable from an io-prod test. Note: field is `bytes: Vec<u8>` NOT `pub bytes` — but private fields of a `pub(crate)` struct ARE reachable within the same crate, so the T8 fix `rf.bytes` works.
2. **io-prod gate = `--fail-under-regions 90`, NO `--branch`** (justfile:40-41). Tier-A branch gate (justfile:22-24) covers only `{{tier_a}}`, not io-prod. Both LOW findings (rev-1 #1, rev-2 HIGH de-escalation) about "region floor not branch-100%" are CONFIRMED.
3. **`decode_frame` = `split_frame` (strips 4-byte len prefix + validates/strips codec byte) → `postcard::from_bytes`** (framing.rs:107-118, 61-87). `scan_all` already strips the 1-byte OUTBOX_FORMAT_VERSION envelope (outbox.rs:202). So the T8 double-strip finding (rev-3) is CONFIRMED: stripping again corrupts the length prefix.
4. **`encode_frame` (framing.rs:56) and `write_reliable_frame` (lib.rs:424-433) both do `postcard::to_allocvec(&ReliableFrame{..})` → `frame_payload`** → byte-identical. Stored value is wire-identical + `decode_frame`-round-trippable. Rev-3's stored-value ruling CONFIRMED.
5. **`emit_transient_batch` lives in `crates/sim/src/stub.rs`, NOT io-prod** — cannot cover the io-prod `write_frame` `matches!` arm. The HIGH finding's factual core is CONFIRMED.

Here is the adjudicated design of record.

---

# R-6d2c — DESIGN OF RECORD (IMPLEMENT-READY): FSM durable write-through / delete-through

Adjudicated against `crates/io-prod/src/mesh.rs`, `crates/io-prod/src/lib.rs`, `crates/io-prod/src/outbox.rs`, `crates/wire/src/framing.rs`, and `justfile` at HEAD (worktree `worktree-new-system`). Every finding below is CONFIRMED or REFUTED with file:line. **Verdict at the end: SOUND_TO_IMPLEMENT with 1 HIGH + 1 MEDIUM must-fix folded into the design (both are cheap and already incorporated here).**

## Adjudication of the three reviews (deduplicated, severity-ranked)

| # | Finding (source) | Ruling | Evidence |
|---|---|---|---|
| **HIGH-1** | §7's HR5 rationale claims the `matches!(…Retained)` true-arm at mesh.rs:1534 is "covered by … the existing `emit_transient_batch` rig" — FALSE; no io-prod test drives `Durability::Retained` through `write_frame` (rev-2 HIGH; rev-1 LOW-1 same region, softer). | **CONFIRMED** | `emit_transient_batch` is in `crates/sim/src/stub.rs:2073` (sim crate), NOT io-prod — grep of `crates/io-prod` for it returns zero. The ONLY `Durability::Retained` mention in io-prod is the `matches!` line itself (mesh.rs:1534). The only `OutFrame{}` builds are mesh.rs:1731 (`send_durable`) + lib.rs:174 (`Transport::send` = Ephemeral default). So after R-6d2c makes `let durable = matches!(…)` a *consumed* value, its TRUE region is exercised nowhere in io-prod. **De-escalated from CRITICAL to HIGH** because io-prod's gate is `--fail-under-regions 90` with NO `--branch` (justfile:40-41) — one uncovered region need not drop below 90%. But shipping a *false* coverage rationale is not allowed. **MUST-FIX:** delete the false sentence AND add one deterministic in-process test driving `Durability::Retained` through `write_frame` (fix detailed in §d, test T9). |
| **MED-1** | `OutboxKey` is hand-built at two sites with divergent field sources → future-refactor foot-gun (rev-2 MEDIUM). | **CONFIRMED as a real DRY/robustness improvement** (not a live bug — the two keys are equal today). | The retain key (in `assign_and_retain`) and the release key (in `on_ack`) share `peer`/`incarnation` from the lane but source `class`/`seq` differently. A single `fn outbox_key(&self, class, seq) -> OutboxKey` single-sources `peer`+`incarnation` and makes the two varying inputs explicit at each site. HR5-clean (straight-line non-generic). **MUST-FIX (folded into §a).** |
| **LOW-1** | §7 mischaracterizes the io-prod gate as branch-100% when it is a 90%-region floor (rev-1 #1). | **CONFIRMED** | justfile:39-41 `tier_b_floor := "90"`, `--fail-under-regions {{tier_b_floor}}`, no `--branch`. Reframe §d as *region-covered by the mock/glue tests keeps ≥90*; branch coverage is a bonus. Doc-accuracy only. |
| **LOW-2** | T8 (glue test) instructs `decode_frame(strip 1-byte envelope of v)` but `scan_all` ALREADY strips the envelope → a double-strip corrupts the length prefix → false-red test (rev-3). | **CONFIRMED** | `scan_all` calls `decode_value(&v)` (outbox.rs:202) which strips `OUTBOX_FORMAT_VERSION` (outbox.rs:127-131). `decode_frame` (framing.rs:107→`split_frame` 61-87) itself strips the 4-byte length prefix + codec byte. So the value from `scan_all` must be fed to `decode_frame` DIRECTLY — no manual pre-strip. **MUST-FIX in the T8 spec (folded into §d, T8).** |
| **LOW-3** | R-6d3 precondition: `commit()` is never called in R-6d2c — staged retains aren't durable until R-6d3 adds the fsync barrier (rev-1 #2). | **CONFIRMED, correctly-scoped** | `NodeOutbox::commit` (outbox.rs:189-196) is the fsync barrier; the FSM never calls it (correct for this slice — sink is None in prod; tests call `commit()` themselves). Add one sentence to the deferral table. No code change. |
| **LOW-4** | R-6d3 precondition: the SAME sink must reach both `assign_and_retain` and `on_ack` or a durable row leaks (rev-1 #3). | **CONFIRMED, harmless in R-6d2c** | None→None symmetry holds trivially this slice. Note it for R-6d3. No R-6d2c code change. |
| **LOW-5** | §8 should thread `PEER`(NodeId(2)) into ALL ~26 test `new` sites, not just the key-inspecting ones, so peer≠from is a structural test-module property (rev-2 LOW). | **ACCEPTED** | Purely mechanical; strengthens the guard. Folded into §e. |

**No CRITICAL survives.** The two core correctness claims are REFUTED-as-sound (i.e., the design is correct):
- **`encoded` ownership** (rev-1, rev-2 confirm): mesh.rs:588 `encoded` is a fresh owned `Vec` from `encode_frame(&frame)`; the `frame.epoch =` mutation (596) and the `frame` MOVE into `retry` (597) do NOT touch it; it's read for `framed_len` at 590 and remains owned. `&encoded` at the post-insert shim is valid — no clone, no re-encode. **CONFIRMED.**
- **stored-value replay contract** (rev-3 confirms): `encode_frame` (framing.rs:56) and `write_reliable_frame` (lib.rs:424-433) both do `postcard::to_allocvec(&ReliableFrame{..})` → `frame_payload` → byte-identical wire frame. `decode_frame::<ReliableFrame>` round-trips it. `NodeOutbox::retain` wraps with ONE version byte via `encode_value` (outbox.rs:182); `scan_all` unwraps it. No double-wrap. **CONFIRMED.**

---

## (a) The exact FSM changes

### A1. `ReliableLaneSender` — add ONE field `peer: NodeId`

Add after `incarnation` (mesh.rs:459). Do NOT store `class` — it lives in every `rf.frame.class` (lib.rs `ReliableFrame.class`), and the per-class-lane invariant (`lanes: BTreeMap<MsgClass, ReliableLaneSender>`, mesh.rs:1104/1518) makes them equal. Storing it would duplicate data + force a `debug_assert` per assign + break ~40 more callers. `incarnation` already exists (mesh.rs:459); `seq` is the loop/return value.

```rust
    /// R-6d2c: the DEST peer this lane sends to (the ConnRegistry key = `dest`/`w.dest`, NOT
    /// `ReliableFrame.from` which is the LOCAL sender). Constant for the lane's life. Supplies
    /// `OutboxKey.peer` for the durable write-through/delete-through — `on_ack` has no `NodeId` in
    /// scope otherwise. Class is NOT stored (it lives in every `RetainedFrame.frame.class`; the
    /// per-class-lane invariant makes them equal, so on_ack reads `rf.frame.class`).
    peer: NodeId,
```

### A2. `RetainedFrame` — add `retained: bool` (mesh.rs:434-437)

```rust
struct RetainedFrame {
    frame: ReliableFrame,
    framed_len: u32,
    /// R-6d2c: was this frame write-through-mirrored to the durable outbox (`durable && sink.is_some()`
    /// at assign)? `on_ack` releases the outbox key ONLY for `retained` frames — so the outbox is a
    /// STRICT SUBSET of `retry` (the Retained frames). Keyed by seq/incarnation (epoch-independent), so
    /// an epoch bump never orphans a row.
    retained: bool,
}
```

### A3. The `outbox_key` helper (MED-1 fix) — ONE key-construction home

Add to `impl ReliableLaneSender`. Single-sources `peer`+`incarnation` from the lane; the two varying inputs are explicit at each call.

```rust
    /// R-6d2c: the ONE `OutboxKey` constructor — `peer`/`incarnation` single-sourced from the lane, the
    /// two varying inputs (class, seq) passed explicitly. Retain and release build the SAME key iff they
    /// pass the same (class, seq); the per-class-lane invariant guarantees `rf.frame.class == class`.
    fn outbox_key(&self, class: MsgClass, seq: u64) -> OutboxKey {
        OutboxKey { peer: self.peer, class, incarnation: self.incarnation, seq }
    }
```

### A4. `::new` signature (mesh.rs:496) — `peer` leads

```rust
    fn new(peer: NodeId, incarnation: u64, retry_cap: usize) -> ReliableLaneSender {
        ReliableLaneSender {
            stream: None,
            peer,
            incarnation,
            // … all other fields unchanged …
        }
    }
```

### A5. `assign_and_retain` signature (mesh.rs:566-571) + write-through placement

Keep the `class` param (it's the per-send class stamped into the frame AND the OutboxKey — not redundant with a non-stored `class`). Add two trailing params:

```rust
    fn assign_and_retain(
        &mut self,
        from: NodeId,
        class: MsgClass,
        bytes: &[u8],
        durable: bool,
        sink: Option<&mut dyn OutboxSink>,
    ) -> Result<u64, AssignReject> {
```

Placement — after the RAM insert (currently mesh.rs:597-599), before `Ok(seq)` (600). Set `retained` on the frame; the write-through is the straight-line monomorphic shim:

```rust
        frame.epoch = self.epoch; // the real epoch for retention + the first write (unchanged, :596)
        // R-6d2c: mirror ONLY a Retained frame to the durable outbox — and only when a sink is wired
        // (None in prod until R-6d3). `retained` records the SAME predicate so `on_ack` releases exactly
        // this subset. Straight-line monomorphic shim (HR5(a)): the only branch is the `if let … && durable`;
        // `retain` is a monomorphic `dyn` call. `encoded` (the epoch=u32::MAX framed bytes from :588, still
        // OWNED — the `frame` move on the line above does NOT touch it) is the outbox VALUE, wire-identical
        // to write_reliable_frame's output (both `postcard::to_allocvec(&ReliableFrame)` → frame_payload);
        // R-6d3 replay `decode_frame`s it + re-stamps epoch. NO re-encode.
        let retained = durable && sink.is_some();
        self.retry.insert(seq, RetainedFrame { frame, framed_len, retained });
        self.retry_bytes += framed_len as usize;
        self.next_seq += 1;
        if let Some(sink) = sink
            && durable
        {
            sink.retain(&self.outbox_key(class, seq), &encoded);
        }
        Ok(seq)
```

Both `Err` returns (Unframable :589, BufferFull :594) PRECEDE the insert — a shed frame is never mirrored. Outbox ⊆ retained ⊆ never-shed. ✔

### A6. `on_ack` signature (mesh.rs:540) + retained-gated delete-through

```rust
    fn on_ack(
        &mut self,
        ack_incarnation: u64,
        ack_epoch: u32,
        ack_through: u64,
        mut sink: Option<&mut dyn OutboxSink>,
    ) -> usize {
```

Inside the retire loop (currently mesh.rs:545-551), gated on `rf.retained`:

```rust
        while self.base < self.next_seq && self.base <= ack_through {
            if let Some(rf) = self.retry.remove(&self.base) {
                self.retry_bytes -= rf.framed_len as usize; // unchanged (:548)
                // R-6d2c: release ONLY a write-through-mirrored (durable) frame — an Ephemeral-heavy Saga
                // lane must not stage a tombstone per acked ephemeral seq. `class` from the retired frame
                // (per-class-lane invariant: rf.frame.class == this lane's class).
                if rf.retained
                    && let Some(s) = sink.as_deref_mut()
                {
                    s.release(&self.outbox_key(rf.frame.class, self.base));
                }
            }
            self.base += 1;
        }
```

`sink.as_deref_mut()` re-borrows the `Option<&mut dyn>` each iteration (multiple releases per ack). The incarnation/epoch guard (mesh.rs:541-543) still returns 0 BEFORE the loop — a stale ack releases nothing. ✔

### A7. Import (mesh.rs:37-40)

Add `crate::outbox::{OutboxKey, OutboxSink}` (both `pub`/`pub` in outbox.rs:81/138). Extend the existing `use crate::{…}` block.

---

## (b) Replay / epoch-bump correctness ruling — no double / stale / lost outbox row

Traced against the actual code:

- **`replay_batch` (mesh.rs:617-625) does NOT call `assign_and_retain`.** It maps `self.retry.values()` to fresh `ReliableFrame`s with the current epoch for re-SENDING only; touches neither `next_seq`, `retry`, nor any sink. A redial re-SENDS, never re-RETAINS → **no double retain.** The write-through fires exactly once, at the original assign. **CONFIRMED.**
- **`on_write_error` (mesh.rs:607-610)** sets `stream=None` + bumps `epoch`. Retained frames stay in `retry` under their ORIGINAL seq. `OutboxKey` = `(peer, class, incarnation, seq)` — **epoch is NOT a key component** — so an epoch bump leaves the outbox rows untouched. **No stale/duplicate rows. CONFIRMED.**
- **Idempotency corollary:** even a hypothetical same-seq re-retain is `store.put(&key.to_bytes(), …)` (outbox.rs:182) — an overwrite of the same 26-byte key. Idempotent. But per the trace, no re-retain occurs.
- **`owes_redelivery`/`replay_lanes`/`handle_connection_drop`** call `replay_batch`/`on_replay_ok`/`on_replay_failed`, never `assign_and_retain`/`on_ack` — no sink interaction. **CONFIRMED.**

**Subset invariant (strict):** `retry` is mutated in EXACTLY two places — insert@597 (retain staged iff `retained`) and remove@546 (release staged iff `rf.retained`, nested INSIDE the `if let Some(rf) = remove`). So an outbox row exists iff its RAM frame is inserted, and is released iff its RAM frame is removed on the SAME retired seq. Outbox rows ⊆ `retry` rows, stable across redials. **CONFIRMED.** R-6d2c adds NO change to `replay_batch`/`on_write_error`/`replay_lanes`.

---

## (c) Stored-value + R-6d3 replay contract ruling

- **Wire-identical value:** `encoded` = `encode_frame(&frame)` (framing.rs:56 = `frame_payload(postcard::to_allocvec(&ReliableFrame{..}))`) is BYTE-IDENTICAL to `write_reliable_frame`'s on-wire output (lib.rs:424-433: same `postcard::to_allocvec(&ReliableFrame)` → `frame_payload`). So the outbox VALUE is a wire-frame; R-6d3 replay does `decode_frame::<ReliableFrame>(value)` (framing.rs:107→`split_frame` strips the 4-byte len prefix + codec byte → `postcard::from_bytes`) recovering `(from,class,incarnation,epoch,seq,bytes)` exactly. **CONFIRMED.**
- **No double-wrap:** the FSM passes RAW `&encoded`; `NodeOutbox::retain` adds the `OUTBOX_FORMAT_VERSION` byte ONCE via `encode_value` (outbox.rs:182); `scan_all` strips it ONCE via `decode_value` (outbox.rs:202). **CONFIRMED.**
- **`epoch=u32::MAX` is don't-care:** the receiver dedups on `(from,class,incarnation,seq)`; epoch is only the cross-stream-race guard. R-6d3 replay at a FRESH epoch adopt-forwards then dedups on seq — idempotent. **CONFIRMED.**
- **Key matches receiver dedup:** `OutboxKey.peer` = the DEST (`self.peer` = `dest`/`w.dest`), which is the send-side edge; the receiver keys its ledger by `frame.from` = this sender — the SAME edge from the other end. The `(incarnation, seq)` pair drives idempotency. **CONFIRMED.**

---

## (d) `MockOutboxSink` + the exact FSM unit tests + HR5 Tier-B plan

`MockOutboxSink` (test-only, in mesh.rs `#[cfg(test)]` mod, ~1762). `commit`/`scan_all`/`gc_below` are inert (R-6d2c never calls them):

```rust
    #[derive(Default)]
    struct MockOutboxSink {
        retained: Vec<(OutboxKey, Vec<u8>)>,
        released: Vec<OutboxKey>,
    }
    impl OutboxSink for MockOutboxSink {
        fn retain(&mut self, key: &OutboxKey, framed: &[u8]) { self.retained.push((*key, framed.to_vec())); }
        fn release(&mut self, key: &OutboxKey) { self.released.push(*key); }
        fn commit(&mut self) {}
        fn scan_all(&self) -> Vec<(OutboxKey, Vec<u8>)> { Vec::new() }
        fn gc_below(&mut self, _incarnation: u64) {}
    }
```

Add `const PEER: NodeId = NodeId(2);` beside the existing `const FROM: NodeId = NodeId(1)` (mesh.rs:1770) so `peer≠from` is structural.

**The tests** (each names the branch it pins):

1. `durable_true_with_sink_retains_exact_key_and_encoded_value` — `new(PEER, 7, BIG_CAP)`; `assign_and_retain(FROM, Saga, b"x", true, Some(&mut mock))`; assert `mock.retained == [(OutboxKey{peer:PEER, class:Saga, incarnation:7, seq:0}, encoded)]` where `encoded = encode_frame(&frame@u32::MAX)`. Pins peer=dest≠FROM, incarnation, seq, class, AND the value. Covers `durable=true ∧ sink=Some`.
2. `durable_false_records_nothing` — `(…, false, Some(&mut mock))`; assert `mock.retained.is_empty()`. Covers `durable=false ∧ sink=Some` (the `&& durable` false-arm; `retained=false`).
3. `durable_true_no_sink_is_noop` — `(…, true, None)`; assert `Ok(0)`, RAM insert happened. Covers `sink=None` (mirrors prod).
4. `on_ack_releases_only_retained_retired_keys` — assign seq0 `(true, Some)` + seq1 `(false, Some)` on ONE lane; `on_ack(7, 0, 1, Some(&mut mock))`; assert `base==2` (both retired) but `mock.released == [OutboxKey{…seq:0}]` ONLY. Covers `rf.retained ∈ {true,false}` + the subset property.
5. `on_ack_with_no_sink_releases_nothing` — assign `(true, Some)`; `on_ack(7,0,0, None)`; assert retire, no panic. Covers on_ack `sink=None`.
6. `stale_ack_releases_nothing` — assign `(true, Some)`; `on_ack(999, 0, 0, Some(&mut mock))` (wrong incarnation → guard-return :541); assert `mock.released.is_empty()`. Covers the guard-return-before-loop with a live sink.
7. `redial_epoch_bump_does_not_double_retain` — assign `(true, Some)`; `on_write_error()`; `replay_batch()`; assert `mock.retained.len() == 1`. Pins §b.
8. **Real-`NodeOutbox` glue test (LOW-2 fix — NO manual pre-strip):**
```rust
    let path = temp_path("glue"); let _g = TempOutbox { path: path.clone() };
    let mut ob = NodeOutbox::open(&path, StoreTuning::default()).expect("open");
    let mut lane = ReliableLaneSender::new(PEER, 3, BIG_CAP);
    lane.assign_and_retain(FROM, MsgClass::Saga, b"payload", true, Some(&mut ob)).expect("assign");
    ob.commit();
    let scanned = ob.scan_all();
    let (k, v) = &scanned[0];
    assert_eq!(*k, OutboxKey { peer: PEER, class: MsgClass::Saga, incarnation: 3, seq: 0 });
    // scan_all ALREADY stripped the OUTBOX_FORMAT_VERSION envelope (outbox.rs:202) — decode DIRECTLY,
    // no second strip (that would eat the frame's u32 length-prefix high byte).
    let (rf, _) = vd_wire::framing::decode_frame::<ReliableFrame>(v).expect("replay-decodes");
    assert_eq!(rf.bytes, b"payload");
```
   (`ReliableFrame.bytes` is reachable — the struct is `pub(crate)` and the test is in-crate.)
9. **HIGH-1 FIX — the missing `Retained`-through-`write_frame` coverage.** Add ONE deterministic in-process io-prod test that constructs `OutFrame { durability: Durability::Retained, .. }` and drives it through `write_frame` (or `send_durable(.., Retained)` against a live loopback pair, the way `mesh_redelivery.rs` already sets up a real-QUIC pair). This exercises the `matches!` TRUE arm at mesh.rs:1534 so `let durable = …` is a covered region. The MockOutboxSink FSM tests (T1-T8) do NOT touch `write_frame` — they call `assign_and_retain` directly — so T9 is required to cover line 1534's true arm. Delete the false `emit_transient_batch` sentence from §7.

**HR5 Tier-B plan (LOW-1 reframe):** the io-prod gate is `cargo llvm-cov -p vd-io-prod --fail-under-regions 90` (justfile:40-41) — a **region** floor, NO `--branch`. The requirement is that every NEW region be EXERCISED to keep ≥90:
- `assign_and_retain` write-through: `durable∈{T(T1/T4), F(T2)} × sink∈{Some(T1/T2), None(T3+prod)}` — all regions hit; `retained` takes both values (T1/T2/T4).
- `on_ack` delete-through: `rf.retained∈{T(T4),F(T4)} × sink∈{Some(T4/T6), None(T5+prod)}` + guard-return-with-sink (T6) — all regions hit.
- `outbox_key` helper: hit by every retain/release test.
- `write_frame` `matches!` TRUE arm: T9 (the fix). FALSE arm: every existing Ephemeral reliable send in `mesh_redelivery`/`mesh_load`/`mesh_under_loss`.

`assign_and_retain`/`on_ack`/`outbox_key` are non-generic (`&mut dyn OutboxSink` = one monomorphization) — no per-monomorphization gotcha; Mock and real `NodeOutbox` exercise the SAME body.

---

## (e) The ~57 call-site update pattern (ALL in mesh.rs — `mesh_redelivery.rs:446` is a comment, no cross-file edits)

**Production (3 edits, in `write_frame`):**
- mesh.rs:1534: `let _durable =` → `let durable =` (now live-consumed).
- mesh.rs:1546: `ReliableLaneSender::new(incarnation, retry_cap)` → `ReliableLaneSender::new(dest, incarnation, retry_cap)` (`dest` in scope, mesh.rs:1515).
- mesh.rs:1547: `lane.assign_and_retain(local, frame.class, &frame.bytes)` → `…, &frame.bytes, durable, None)`.
- mesh.rs:1128: `lane.on_ack(e.incarnation, e.epoch, e.ack_through)` → `…, e.ack_through, None)`.
- Update the stale R-6d2b comment at mesh.rs:1530-1533 to "write-through now wired; sink=None until R-6d3 injects the real `Arc<Mutex<NodeOutbox>>`".

**Test mod (mesh.rs >1761) — mechanical sweep:**
- **~26 `ReliableLaneSender::new(inc, cap)`** → `ReliableLaneSender::new(PEER, inc, cap)` — thread `PEER`(NodeId(2)) into ALL of them (LOW-5), keeping `FROM`(NodeId(1)) strictly as the `assign_and_retain` sender arg, so `peer≠from` is a whole-module property.
- **~40 `lane.assign_and_retain(FROM, CLASS, b"…")`** → append `, false, None` EXCEPT T1/T2/T3/T4/T7/T8/T9 which pass explicit `durable, Some/None`.
- **~16 `lane.on_ack(inc, epoch, through)`** → append `, None` EXCEPT T4/T5/T6 which pass explicit sink.

No existing test SEMANTICS change: `assign_and_retain`'s return value + RAM accounting are unchanged when `sink=None` / `durable=false`, so all `Ok(0)`/byte-count/retire-count assertions hold. The `Some` path is NEW coverage via T1-T9 only.

---

## (f) Must-fix findings (blocking), severity-ranked

1. **HIGH-1 (blocking):** delete the false `emit_transient_batch` coverage sentence + add test **T9** (drive `Durability::Retained` through `write_frame`) so the `matches!` TRUE arm at mesh.rs:1534 is a covered region. — *folded into §d.*
2. **MED-1 (blocking-for-robustness):** the `outbox_key` helper — single key-construction home called from both retain + release. — *folded into §a A3/A5/A6.*
3. **LOW-2 (blocking-the-test):** T8 must decode the `scan_all` value DIRECTLY (no manual pre-strip). — *folded into §d T8.*
4. LOW-1 (doc): reframe §d as region-floor ≥90, not branch-100%. — *done.*
5. LOW-3 / LOW-4 (R-6d3 preconditions, non-blocking): add the two deferral-table sentences below.

---

## (g) R-6d2c vs R-6d3 boundary

| In R-6d2c | Deferred to R-6d3 |
|---|---|
| `peer: NodeId` + `retained: bool` fields; `outbox_key` helper | The real `Arc<Mutex<NodeOutbox>>` on `PeerWriter`/`spawn_mesh` (the concurrency) |
| `assign_and_retain` `(durable, sink)` + write-through shim | Threading the sink THROUGH async `write_frame`/`peer_writer` (flip both `None`→real handle) — **LOW-4: the SAME sink MUST reach both assign_and_retain AND on_ack for a lane, or a durable row leaks** |
| `on_ack` `sink` param + retained-gated delete-through | The durable-BEFORE-send fsync GATE — **LOW-3: R-6d2c STAGES but never `commit()`s; R-6d3 MUST call `commit()` (before the QUIC send) or every staged retain is inert (not on disk) even after None→real** |
| `write_frame` consumes `durable`, passes `(durable, None)`; peer_writer `on_ack(…, None)` | Boot replay (`scan_all`→re-drive) + `gc_below` post-replay sweep |
| `MockOutboxSink` + tests T1-T7 + glue T8 + `write_frame`-Retained T9 | The saga/dest closure (TransientBatch AwaitAdopt split); commit-cadence placement (must NOT block inline on a tokio worker); the SIGKILL process-crash proof |

**No behavior change in production:** `durable` is computed + passed, but its terminal consumer is `None`, so no outbox row is written until R-6d3 injects the real handle. The 20 Hz datagram path (`Reliability::Unreliable` arm, mesh.rs:1662) never calls `assign_and_retain`/sees `durable` — byte-identical. Frozen `sim::io` seam UNTOUCHED (`OutboxSink` is io-prod-internal). `outbox.rs` UNCHANGED.

**Files touched:** `crates/io-prod/src/mesh.rs` ONLY (struct 434-437 + field ~459; `outbox_key` helper new; `new` :496; `assign_and_retain` :566-601; `on_ack` :540-554; `write_frame` :1534/1546/1547; peer_writer :1128; imports :37-40; test mod >1761 incl. T1-T9 + `const PEER`).

---

## (h) FINAL VERDICT: **SOUND_TO_IMPLEMENT**

The design is architecturally correct — a pure synchronous FSM slice, `None` in prod, no concurrency/fsync/replay/saga (all correctly R-6d3), not dead plumbing (the `durable` lowering is live-in-prod; the `Some` path live-in-test). The two load-bearing correctness claims (`encoded` ownership; wire-identical replay-decodable value with no double-wrap) are CONFIRMED against the code. Ship it with the three folded must-fixes: **(1)** T9 covering the `write_frame`-Retained arm + delete the false `emit_transient_batch` sentence; **(2)** the `outbox_key` single-source helper; **(3)** T8 decodes the `scan_all` value directly. All three are cheap and are already written into §a/§d above.

## ORIGINAL DESIGN

I now have everything. Let me confirm one final detail: whether `bytes.to_vec()` at line 585 means `encoded` (line 588) still owns its bytes at the insert point (line 597). Line 588 `encoded` is a fresh `Vec` from `encode_frame(&frame)` — it borrows `frame` but produces an owned `Vec`. Then line 596 mutates `frame.epoch`, line 597 moves `frame` into `retry`. So `encoded` is fully owned and independent of the `frame` move. It's alive at line 597-599 (used at 590 for `framed_len`, still owned after). Confirmed.

I have complete grounding. Producing the implement-ready diff-plan.

---

# R-6d2c — IMPLEMENT-READY DIFF-PLAN: FSM durable write-through / delete-through

Grounded against `crates/io-prod/src/mesh.rs` and `crates/io-prod/src/outbox.rs` at HEAD (worktree `worktree-new-system`). This slice consumes the dead `_durable` bridge landed by R-6d2b (mesh.rs:1534) and mirrors `Retained` reliable frames to the R-6d1 `OutboxSink`, exercised entirely via `MockOutboxSink` — **`None` in production, zero behavior change** (per Ruling A, doc §b).

## Scope resolution up front (what changed vs the task framing)

Three facts from the actual code tighten the plan:

- **The `sink` param stops at the FSM boundary.** The vetted design's MEDIUM-2 (doc lines 158, 190-192) rules that `write_frame`/`peer_writer` must NOT grow a `sink` param in R-6d2c — threading `Option<&mut dyn OutboxSink>` through async `write_frame` (which opens real QUIC streams, mesh.rs:1615) is inert `None`-only plumbing with no unit test. So `write_frame` passes the **literal `None`** at the `assign_and_retain` call (mesh.rs:1547) and the peer_writer passes the **literal `None`** at the `on_ack` call (mesh.rs:1128). Only `assign_and_retain` and `on_ack` grow the param; the async writer gets no new param. R-6d3 flips both literals to the real `Arc<Mutex<NodeOutbox>>` handle.
- **There is exactly ONE production `ReliableLaneSender::new` site** (mesh.rs:1546, the `or_insert_with` in `write_frame`, which has `dest: NodeId` in scope). Every other `new`/`assign_and_retain`/`on_ack` caller is in the mesh.rs `#[cfg(test)]` mod (starts line 1761). `crates/io-prod/tests/mesh_redelivery.rs` has ZERO real calls (its one `assign_and_retain` hit at :446 is a comment). So the "~57 call sites" are: 1 prod + ~26 test `new` + ~40 test `assign_and_retain` + ~16 test `on_ack`, ALL inside mesh.rs.
- **`peer` = the DEST, not `from`.** `assign_and_retain`'s existing `from: NodeId` param is the LOCAL sender (`local`, mesh.rs:1547) stamped into `ReliableFrame.from`. The `OutboxKey.peer` is the DESTINATION (`dest`/`w.dest`). These are DIFFERENT nodes — the lane must store `dest` as `peer`, NOT reuse `from`.

---

## (1) `ReliableLaneSender` field additions + `::new` signature + the `class` param decision

**Add ONE field: `peer: NodeId`.** Do NOT store `class` on the lane. Reasoning:

- `peer` is **not** otherwise reachable in `on_ack` (mesh.rs:540) — its signature has no `NodeId` at all, and the retire loop needs it to build the delete key. It MUST become a field.
- `class` **is** already reachable in `on_ack` via `rf.frame.class` (each `RetainedFrame` carries the class it was built with, mesh.rs:435 → `ReliableFrame.class`). Adding a `class` field would duplicate data already in every retained frame and force a `debug_assert class == self.class` on every assign. So **keep `class` as the `assign_and_retain` param** and read `rf.frame.class` in `on_ack`. This is the DRY shape (doc LOW-4: `rf.frame.class == the lane's class` by the per-class-lane invariant — `lanes: BTreeMap<MsgClass, ReliableLaneSender>`, mesh.rs:1104/1518).

`incarnation` is already a field (mesh.rs:459) — the third OutboxKey component needs no addition. `seq` is the loop/return value.

**Field addition** (mesh.rs, in the struct at 453-493, e.g. right after `incarnation` at 459):
```rust
    /// R-6d2c: the DEST peer this lane sends to (the ConnRegistry key = `dest`, NOT `ReliableFrame.from`
    /// which is the LOCAL sender). Constant for the lane's life. Supplies `OutboxKey.peer` for the durable
    /// write-through/delete-through — `on_ack` has no `NodeId` in scope otherwise. Class is NOT stored (it
    /// lives in every `RetainedFrame.frame.class`; the per-class-lane invariant makes them equal).
    peer: NodeId,
```

**`::new` signature** (mesh.rs:496) — `peer` leads (it identifies the lane):
```rust
    fn new(peer: NodeId, incarnation: u64, retry_cap: usize) -> ReliableLaneSender {
        ReliableLaneSender {
            stream: None,
            peer,
            incarnation,
            // … rest unchanged …
        }
    }
```

**`assign_and_retain` keeps its `class` param** (not redundant — it's the per-send class stamped into the frame AND the OutboxKey; the lane doesn't store it).

---

## (2) `RetainedFrame` `retained` field

**Add `retained: bool`** to `RetainedFrame` (mesh.rs:434-437). It records, per retained seq, whether that frame was mirrored to the outbox — so `on_ack` (which retires the whole `base..=ack_through` prefix, durable AND ephemeral) releases ONLY the durable subset. Without it, an Ephemeral-heavy Saga lane (TransientBatch=Retained but ~9 saga_runtime flows=Ephemeral share the Saga lane, doc line 30) would stage a pointless `release` tombstone per acked ephemeral seq.

```rust
struct RetainedFrame {
    frame: ReliableFrame,
    framed_len: u32,
    /// R-6d2c: was this frame write-through-mirrored to the durable outbox (i.e. `durable && sink.is_some()`
    /// at assign)? `on_ack` releases the outbox key ONLY for `retained` frames — the outbox is a STRICT
    /// SUBSET of `retry` (the Retained frames), so a release is staged exactly for a durable seq, never for
    /// an Ephemeral one. Keyed by seq/incarnation (epoch-independent), so an epoch bump never orphans a row.
    retained: bool,
}
```

**Subset invariant** (answers Q3): `retain` is staged in `assign_and_retain` iff `durable && sink.is_some()`; `retained` is set to that SAME predicate; `on_ack` releases iff `rf.retained`. Since `on_ack` retires seq `self.base` and reads `rf.retained` from the SAME `RetainedFrame` it removes, the release is staged for exactly the seqs that were retained. Outbox rows ⊆ `retry` rows, and every outbox row is released when its seq retires. ✔

---

## (3) `assign_and_retain` new signature + the monomorphic write-through placement

**Signature** (mesh.rs:566-571) — two trailing params:
```rust
    fn assign_and_retain(
        &mut self,
        from: NodeId,
        class: MsgClass,
        bytes: &[u8],
        durable: bool,
        sink: Option<&mut dyn OutboxSink>,
    ) -> Result<u64, AssignReject> {
```

**Placement** — after the RAM insert (mesh.rs:597-599), before `Ok(seq)` (mesh.rs:600). `retained` is set on the `RetainedFrame` at insert time; the write-through is the straight-line shim:

```rust
        frame.epoch = self.epoch; // the real epoch for retention + the first write (unchanged)
        // R-6d2c: mirror ONLY a Retained frame to the durable outbox — and only when a sink is wired (None
        // in prod until R-6d3). `retained` records the SAME predicate so `on_ack` releases exactly this
        // subset. Straight-line monomorphic shim (HR5(a)): the ONLY branch here is the two-level `if let`;
        // `retain` is a monomorphic `dyn` call. `encoded` (the epoch=u32::MAX framed bytes from the check
        // above, still owned — the `frame` move below does NOT touch it) is the outbox VALUE (L2 seam: a
        // full framed ReliableFrame; R-6d3 replay decodes it + re-stamps epoch). NO re-encode.
        let retained = durable && sink.is_some();
        self.retry.insert(seq, RetainedFrame { frame, framed_len, retained });
        self.retry_bytes += framed_len as usize;
        self.next_seq += 1;
        if let Some(sink) = sink
            && durable
        {
            sink.retain(
                &OutboxKey {
                    peer: self.peer,
                    class,
                    incarnation: self.incarnation,
                    seq,
                },
                &encoded,
            );
        }
        Ok(seq)
```

**Ownership of `encoded` confirmed (answers Q2):** `encoded` (mesh.rs:588) = `encode_frame(&frame)` — an **owned `Vec<u8>`** that borrows `frame` only during the call, then holds its own bytes. It's read for `framed_len` (mesh.rs:590), unaffected by the `frame.epoch =` mutation (596) or the `frame` MOVE into `retry` (597). So `&encoded` at the shim is valid — no clone, no re-encode. The epoch stored in the frame bytes is `u32::MAX` (the worst-case-varint value), which is the vetted L2 seam contract (doc LOW-2): R-6d3 replay strips the envelope, `decode_frame::<ReliableFrame>`, re-sends at epoch 0.

**Reject ordering confirmed:** both `Err` returns (Unframable at 589, BufferFull at 594) PRECEDE the insert (597) — a shed frame never reaches the shim, so a shed frame is never mirrored. Outbox ⊆ retained ⊆ (never-shed). ✔

**HR5(a) monomorphic-shim discipline:** `assign_and_retain` is NOT generic (it takes `&mut dyn OutboxSink`, one monomorphization). The write-through is a single straight-line `if let … && durable` — the only new branch. No `?`, no `match`, no error closure. Clean.

**Import:** add `OutboxKey`, `OutboxSink` to the `use crate::{…}` block (mesh.rs:37-40) — `crate::outbox::{OutboxKey, OutboxSink}` (both are `pub`/`pub(crate)` in outbox.rs:81/138).

---

## (4) `on_ack` new signature + the retained-gated delete-through

**Signature** (mesh.rs:540) — one trailing param (owned `mut` so `as_deref_mut` re-borrows across the loop):
```rust
    fn on_ack(
        &mut self,
        ack_incarnation: u64,
        ack_epoch: u32,
        ack_through: u64,
        mut sink: Option<&mut dyn OutboxSink>,
    ) -> usize {
```

**Delete-through** inside the existing retire loop (mesh.rs:545-551), gated on `rf.retained`:
```rust
        while self.base < self.next_seq && self.base <= ack_through {
            if let Some(rf) = self.retry.remove(&self.base) {
                self.retry_bytes -= rf.framed_len as usize; // unchanged
                // R-6d2c: release ONLY a frame that was write-through-mirrored (durable subset) — an
                // Ephemeral-heavy Saga lane must not stage a tombstone per acked ephemeral seq. `class` is
                // read from the retired frame (per-class-lane invariant: rf.frame.class == this lane's class).
                if rf.retained
                    && let Some(s) = sink.as_deref_mut()
                {
                    s.release(&OutboxKey {
                        peer: self.peer,
                        class: rf.frame.class,
                        incarnation: self.incarnation,
                        seq: self.base,
                    });
                }
            }
            self.base += 1;
        }
```

`sink.as_deref_mut()` re-borrows `Option<&mut dyn OutboxSink>` each iteration (the loop may release multiple keys per ack). The delete key exactly matches the retain key: same `peer` (field), same `class` (from the retired `rf.frame.class`), same `incarnation` (field), same `seq` (`self.base`, the retired seq). ✔

---

## (5) `write_frame` (consume `_durable`) + peer_writer `on_ack` (None)

**`write_frame`** (mesh.rs:1534) — rename `_durable` → `durable` (it becomes live) and pass it through, with `None` for the sink; also thread `dest` into the lane constructor:

- mesh.rs:1534: `let durable = matches!(frame.durability, vd_sim::io::Durability::Retained);` (drop the leading underscore — the value is now consumed).
- mesh.rs:1546 (the `or_insert_with`): `ReliableLaneSender::new(dest, incarnation, retry_cap)` — `dest` is already a `write_frame` param (mesh.rs:1515), so it's in scope.
- mesh.rs:1547: `match lane.assign_and_retain(local, frame.class, &frame.bytes, durable, None)` — `durable` live, sink `None` in R-6d2c (MEDIUM-2).

**peer_writer `on_ack` call** (mesh.rs:1128):
```rust
    let retired = lane.on_ack(e.incarnation, e.epoch, e.ack_through, None);
```
`None` in R-6d2c. R-6d3 flips both `None`s to the real handle (and threads it through `write_frame`/`peer_writer`).

Update the stale R-6d2b comment at mesh.rs:1530-1533 to state the write-through is now wired but sink=None (real handle in R-6d3).

**Datagram path untouched:** the `Reliability::Unreliable` arm (mesh.rs:1662) never calls `assign_and_retain` and never sees `durable`. Byte-identical. ✔

---

## (6) Replay / epoch-bump trace — no double or stale outbox rows (answers Q4)

Traced against the actual code:

- **`replay_batch` (mesh.rs:617-625) does NOT call `assign_and_retain`.** It reads `self.retry.values()` and re-stamps a fresh `ReliableFrame` with the current epoch for re-SENDING only. It touches neither `next_seq`, `retry`, nor any sink. So a redial re-SENDS but never re-RETAINS → **no double retain.** ✔ (The write-through fires exactly once, at the original assign.)
- **`on_write_error` (mesh.rs:607-610)** sets `stream = None` and bumps `epoch`. The retained frames stay in `retry` under their ORIGINAL seq. Since `OutboxKey` uses `(peer, class, incarnation, seq)` — **epoch is NOT a key component** — the outbox rows are untouched by an epoch bump. No stale/duplicate rows. ✔
- **Idempotency corollary:** even in a hypothetical future where a re-retain of the SAME seq happened, `NodeOutbox::retain` does `store.put(&key.to_bytes(), …)` (outbox.rs:181-183) — an overwrite of the same key. Idempotent. But per the trace above, no re-retain occurs at all in R-6d2c.
- **`replay_lanes` (the retransmit-timer path, mesh.rs:1206) and `handle_connection_drop`:** neither calls `assign_and_retain` or `on_ack`; they call `replay_batch`/`on_replay_ok`/`on_replay_failed`. No sink interaction. ✔

Conclusion: the outbox stays a strict, stable subset of `retry`, unperturbed by redials/epoch bumps. R-6d2c adds NO change to `replay_batch`/`on_write_error`/`replay_lanes`.

---

## (7) `MockOutboxSink` + the FSM unit tests + HR5 Tier-B argument

**`MockOutboxSink`** (test-only, in mesh.rs `#[cfg(test)]` mod, ~1762). Records `retain`/`release` calls; `commit`/`scan_all`/`gc_below` are no-ops/empty (R-6d2c never calls them — those are R-6d3):
```rust
    #[derive(Default)]
    struct MockOutboxSink {
        retained: Vec<(OutboxKey, Vec<u8>)>,
        released: Vec<OutboxKey>,
    }
    impl OutboxSink for MockOutboxSink {
        fn retain(&mut self, key: &OutboxKey, framed: &[u8]) {
            self.retained.push((*key, framed.to_vec()));
        }
        fn release(&mut self, key: &OutboxKey) {
            self.released.push(*key);
        }
        fn commit(&mut self) {}
        fn scan_all(&self) -> Vec<(OutboxKey, Vec<u8>)> { Vec::new() }
        fn gc_below(&mut self, _incarnation: u64) {}
    }
```

**The unit tests** (each names the branch it pins):

1. `durable_true_with_sink_retains_exact_key_and_encoded_value` — `new(NodeId(2), 7, BIG_CAP)`; `assign_and_retain(FROM, Saga, b"x", true, Some(&mut mock))`; assert `mock.retained == vec![(OutboxKey{peer:NodeId(2), class:Saga, incarnation:7, seq:0}, encoded)]` where `encoded == encode_frame(&frame_at_epoch_u32MAX)` — pins peer=dest (NOT `FROM`), incarnation, seq, class, AND the u32::MAX-epoch value. Covers `durable=true ∧ sink=Some ∧ retained=true`.
2. `durable_false_records_nothing` — `assign_and_retain(FROM, Saga, b"x", false, Some(&mut mock))`; assert `mock.retained.is_empty()`. Covers `durable=false ∧ sink=Some` (the `&& durable` false-arm; `retained=false`).
3. `durable_true_no_sink_is_noop` — `assign_and_retain(FROM, Saga, b"x", true, None)`; the RAM insert still happens (`Ok(0)`), nothing to assert on a sink (there is none). Covers `sink=None` (the `if let Some` false-arm) — mirrors the production `None` path.
4. `on_ack_releases_only_retained_retired_keys` — assign seq0 as `(true, Some)` and seq1 as `(false, Some)` on ONE lane; `on_ack(7, 0, 1, Some(&mut mock))`; assert both retired (`base==2`) but `mock.released == [OutboxKey{…seq:0}]` ONLY. Covers `rf.retained=true` AND `rf.retained=false` in on_ack, and the subset property.
5. `on_ack_with_no_sink_releases_nothing` — assign `(true, Some)`; then `on_ack(7,0,0, None)`; assert retire happens, no panic. Covers on_ack `sink=None`.
6. `stale_ack_releases_nothing` — assign `(true, Some)`; `on_ack(999, 0, 0, Some(&mut mock))` (wrong incarnation → early return at mesh.rs:541); assert `mock.released.is_empty()`. Covers the guard-return-before-loop path with a live sink.
7. `redial_epoch_bump_does_not_double_retain` — assign `(true, Some)`; `on_write_error()`; `replay_batch()`; assert `mock.retained.len() == 1` (replay re-sends, never re-retains). Pins Q4.
8. **One real-`NodeOutbox` glue test** (LOW-2 replay-consumable proof): `assign_and_retain(FROM, Saga, b"payload", true, Some(&mut real_outbox)); real_outbox.commit(); let (k,v)=real_outbox.scan_all()[0];` assert `k == expected_key` AND `decode_frame::<ReliableFrame>(strip 1-byte envelope of v).bytes == b"payload"` — proving the stored u32::MAX-epoch frame is decode-and-replay-consumable by R-6d3. (Uses a `TempOutbox` guard like outbox.rs:256.)

**HR5 Tier-B coverage argument.** The new branches and their covering test:
- `assign_and_retain` write-through: `durable ∈ {true(T1/T4), false(T2)}` × `sink ∈ {Some(T1/T2), None(T3 + all prod runs)}` — all 4 combos hit; the `retained` bool takes both values (T1/T2/T4).
- `on_ack` delete-through: `rf.retained ∈ {true(T4), false(T4)}` × `sink ∈ {Some(T4/T6), None(T5 + prod)}` × the early-return guard with a sink (T6) — all arms hit.
- The `matches!` lowering in `write_frame` (mesh.rs:1534): `Retained`(covered by any Retained producer path in the sim conformance tests + the existing `emit_transient_batch` rig) / `Ephemeral`(every other reliable send). This is io-prod (Tier-B) but the `matches!` false-arm is real-arm-covered by production reliable sends in the existing mesh tests — no `matches!`-false-region gap.

No per-monomorphization gotcha: `assign_and_retain`/`on_ack` are non-generic (one monomorphization each; `dyn OutboxSink` erases the impl). The `MockOutboxSink` and the real `NodeOutbox` both exercise the SAME monomorphic body, so the object-safe `dyn` call is covered once and counts.

---

## (8) The ~57 call-site update pattern

**Production (1 site, in `write_frame`):**
- mesh.rs:1546: `ReliableLaneSender::new(incarnation, retry_cap)` → `ReliableLaneSender::new(dest, incarnation, retry_cap)`.
- mesh.rs:1547: `lane.assign_and_retain(local, frame.class, &frame.bytes)` → `lane.assign_and_retain(local, frame.class, &frame.bytes, durable, None)`.
- mesh.rs:1128: `lane.on_ack(e.incarnation, e.epoch, e.ack_through)` → `lane.on_ack(e.incarnation, e.epoch, e.ack_through, None)`.

**Test mod (mesh.rs, >1761) — mechanical sweep, ALL get default/None args:**
- ~26 `ReliableLaneSender::new(inc, cap)` → `ReliableLaneSender::new(peer, inc, cap)`. Pick a literal `NodeId` per test; the canonical value is the module's existing `const FROM: NodeId = NodeId(1)` (mesh.rs:1770) reused as the peer, OR a distinct `const PEER: NodeId = NodeId(2)` added to the mod to make peer≠from visible in T1. Recommend adding `const PEER: NodeId = NodeId(2);` beside `FROM` and threading `PEER` — this is what test T1 asserts (peer=dest, not from). Tests that never inspect the key can pass `FROM`; tests that do (T1/T4) pass `PEER`.
- ~40 `lane.assign_and_retain(FROM, CLASS, b"…")` → append `, false, None` (the default: Ephemeral, no sink) EXCEPT the new mock tests T1/T2/T3/T4/T7/T8 which pass the explicit `durable, Some(&mut mock)`/`None` per their branch.
- ~16 `lane.on_ack(inc, epoch, through)` → append `, None` EXCEPT T4/T5/T6 which pass the explicit sink.

The mechanical sweep is: `assign_and_retain(…)` → add `, false, None`; `on_ack(…)` → add `, None`; `new(…)` → prepend a `NodeId` arg. No test SEMANTICS change (all existing tests were Ephemeral/no-sink; the `Some` path is NEW coverage via the new mock tests only). The default-Ephemeral choice keeps every prior assertion (`Ok(0)`, byte counts, retire counts) intact — `assign_and_retain`'s return value and RAM accounting are unchanged when `sink=None`.

---

## What is R-6d2c vs deferred to R-6d3

| In R-6d2c (this slice) | Deferred to R-6d3 |
|---|---|
| `peer: NodeId` field + `retained: bool` field | The real `Arc<Mutex<NodeOutbox>>` handle on `PeerWriter`/`spawn_mesh` |
| `assign_and_retain` `(durable, sink)` params + write-through shim | Threading the sink THROUGH async `write_frame`/`peer_writer` (flip both `None`→real) |
| `on_ack` `sink` param + retained-gated delete-through | The durable-BEFORE-send fsync gate (`commit()` before QUIC write) |
| `write_frame` consumes `durable`, passes `(durable, None)` | Boot replay (`scan_all` → re-drive) + `gc_below` post-replay sweep |
| peer_writer passes `on_ack(…, None)` | The saga/dest closure (TransientBatch AwaitAdopt split) |
| `MockOutboxSink` + 8 FSM unit tests | The SIGKILL process-crash proof (end-to-end) |
| The one real-`NodeOutbox` glue+decode test (LOW-2) | The commit-cadence placement (must NOT block inline on tokio, doc §b ⚠) |

**No behavior change in production** (sink is `None` everywhere in prod): the `durable` bool is computed and passed but its terminal consumer is `None`, so no outbox row is written until R-6d3 injects the real handle. The 20Hz datagram path is provably untouched (write-through is inside the `Reliability::Reliable` arm only).

---

## Assumptions flagged / alternatives rejected

- **ASSUMPTION** (verified): `dest` is in scope at mesh.rs:1546 (it's a `write_frame` param, mesh.rs:1515) and `w.dest` at mesh.rs:1128 (a `PeerWriter` field, mesh.rs:1079). Both confirmed by grep — no extra threading needed.
- **ASSUMPTION** (verified): `mesh_redelivery.rs` has NO real FSM calls (its `assign_and_retain` mention is a comment) — so no cross-file test edits. The full ~57 sweep is contained in mesh.rs.
- **REJECTED — store `class` as a lane field:** duplicates `rf.frame.class`, forces a `debug_assert` per assign, and breaks ~40 more test callers than necessary. Reading `rf.frame.class` in on_ack is DRY (doc LOW-4).
- **REJECTED — reuse `from` as `peer`:** `from` is the LOCAL sender (`local`), `peer` is the DEST (`dest`). Different nodes; conflating them would build the OutboxKey under the wrong node. Test T1 pins `peer==NodeId(2)` ≠ `FROM==NodeId(1)` to catch exactly this.
- **REJECTED — thread the sink through `write_frame`/`peer_writer` now:** inert `None`-only plumbing through an untestable async fn (MEDIUM-2). Deferred to R-6d3 where the SIGKILL proof exercises it.
- **REJECTED — re-encode in the shim / clone `frame`:** `encoded` (the u32::MAX-epoch framed bytes) is already owned and alive at the insert point; `&encoded` is zero-cost and IS the vetted L2 outbox value. Confirmed no move conflict.
- **REJECTED — call `commit()` in R-6d2c:** the mock/glue tests call `commit()` on the real outbox for the LOW-2 proof, but the FSM (`assign_and_retain`/`on_ack`) does NOT — commit cadence + the block-on-durable ordering is R-6d3 (doc §b ⚠: must not block inline on a tokio worker).

**Files touched:** `crates/io-prod/src/mesh.rs` only (struct fields 434-437 + 453-493, `new` 496, `assign_and_retain` 566-601, `on_ack` 540-554, `write_frame` 1534/1546/1547, peer_writer 1128, imports 37-40, test mod >1761). `crates/io-prod/src/outbox.rs` — UNCHANGED (reuse `OutboxKey`/`OutboxSink`/`NodeOutbox`). Frozen `sim::io` seam — UNTOUCHED (`OutboxSink` is io-prod-internal; the `Durability` seam already landed in R-6d2a/b).