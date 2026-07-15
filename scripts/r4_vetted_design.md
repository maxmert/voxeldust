I have verified everything needed. `msg_id` lives on `OutFrame` (assigned at `MeshTransport::send`, mesh.rs:1237), NOT on `ReliableFrame` — confirming the D1 finding is real (the timer arm has no `OutFrame` to source a `msg_id` from). Default backoff is 50ms→5s geometric, confirming the R-4c formula domain-mix finding. All three reviewers' CRITICAL/HIGH findings are grounded in the real code.

Here is my final adjudicated design.

---

# R-4' — FINAL VETTED, IMPLEMENT-READY DESIGN

**Verdict: SOUND TO IMPLEMENT once the four folded-in corrections below are applied.** The original design's *structure* (sub-slicing, MetricsSource trait, load test, the buffer-first/no-shed principle) is sound and verified against the real code. But **three CRITICAL and two HIGH findings are real defects in the design as written** — each would have shipped a lost-redelivery or never-confirm bug. They are corrected below. Two findings are downgraded/rejected with reasons. Line numbers are against `crates/io-prod/src/mesh.rs` and `crates/sim/src/saga.rs` at commit **7c28d50** (verified current HEAD).

## Adjudication summary

| Finding | Reviewer verdict | My ruling | Basis (verified) |
|---|---|---|---|
| **C1** — idle-after-blip gap moved to per-lane, not closed; guard's `connection.is_none()` conjunct is the defect | CRITICAL | **UPHELD — fold in** | mesh.rs:1107-1111 resets ALL lanes `stream=None` on any re-dial; mesh.rs:1142 re-opens a lane's stream ONLY when a new frame for THAT class arrives. The send-arm-success path leaves a sibling lane owed-but-disarmed. Real. |
| **C2** — counter bump in `on_write_error` (per-lane on shared-connection drop) counts "one connection drop" as N lane failures AND never fires on the idle timer path | CRITICAL | **UPHELD — fold in** | mesh.rs:1014 `for lane in lanes.values_mut(){ lane.on_write_error() }` — one blip bumps every lane by 1. Destroys blip-tolerance. Real. |
| **C3 (R-4c)** — `validate_against` assumes constant NodeUnreachable spacing; real cadence is geometric 50ms→5s; off-by-one N+n vs N+n-1 | CRITICAL | **UPHELD — fold in** | mesh.rs:177-178 backoff is `50ms` doubling to `5s`; mesh.rs:1031 sleeps `backoff` then doubles. Pre-threshold phase is geometric, not `retry_delay_ticks_hint`-linear. Real. |
| **H1** — `arm_retransmit` spec self-contradictory | HIGH | **UPHELD** — subsumed by C1's fix (edge-triggered iteration-tail re-eval) | Same root as C1. |
| **H2** — counter-reset scope contradictory (send-arm resets one class, timer-arm resets all) | HIGH | **UPHELD — fold in** | Design §1.3 vs the timer-arm body literally disagree. Real. |
| **H3** — `retry_bytes` stored length must be the WORST-CASE-epoch framed length, not real-epoch | HIGH | **UPHELD — fold in** | mesh.rs:472-488 checks at `epoch=u32::MAX` (5-byte varint) then stamps the real epoch; `replay_batch` (mesh.rs:508-516) re-stamps to a growing epoch. On-wire size grows over life. Real. |
| **M-shed** — no safe shed of a retained frame; producer-backpressure only | MEDIUM/decision | **UPHELD — this IS the decided policy** | mesh.rs:327-390 receiver is strictly contiguous; dropping any retained frame wedges or loses. Confirmed. |
| **M-metrics** — "7 names" is wrong, 9 must be added | MEDIUM | **UPHELD — fold in (9 names)** | `MeshStatsSnapshot` has 10 fields (mesh.rs:234-245); 1 already registered (wire/admin.rs:143). 9 new. |
| **M-open-stalled-lane** (R-4a timer doesn't cover a live-connection dead-ack-path) | MEDIUM | **UPHELD as a documentation/test requirement** — it is R-4b's job, must be stated | Correct division of labour. |
| **M-rebounce-vs-inbox** (`>=` re-bounce floods BoundedInbox) | MEDIUM | **PARTIALLY UPHELD** — coalesce the re-bounce; see D1 resolution | Real risk against mesh.rs:251 push_inbox reliable-drop. |
| D4 gateway/shard admin listener expands CAF-2 loopback/auth contract | MEDIUM note | **UPHELD — bind loopback-only, cite CAF-2, or defer** | gateway.rs:21 / shard.rs:22 discard `_control`; no admin listener today. |

**Rejected / downgraded (with reason):**
- **None of the SOUND-verified notes are contested.** The third reviewer explicitly verified: `stats()` is a pure atomic load off the axum task (mesh.rs:573, off the seam + off the 20 Hz path); the N-peer test rides real QUIC (not the mem hub); `validate_against` in vd-sim has no io-prod dep. All confirmed. The only thing I *reject* is the original design's own **D2 recommendation framing** as under-specified — I make it a firm decision below (writer-side buffer-full = `reliable_shed`, the mpsc bound is the real producer backpressure).

---

## Sub-slicing (unchanged — verified sound; land in this order)

| Slice | Deliverable | Crate | Separable because |
|---|---|---|---|
| **R-4a** | Retransmit/redial timer + per-lane consecutive-failure counter + confirm-dead-after-N bounce | io-prod (mesh.rs) | The concurrency change; deepest review. |
| **R-4b** | `retry_bytes` accounting + shed-loud on buffer-full | io-prod (mesh.rs) | Pure FSM; unit-testable, no tokio. |
| **R-4c** | `LivenessTuning::validate_against(&SagaTuning)` | vd-sim (saga.rs, Tier-A 100%) + bin | Different crate; independent. |
| **R-4d** | MetricsSource wiring (M4) | wire/admin.rs + io-prod/admin.rs + bins | Additive, off seam/hot-path. |
| **R-4e** | N-peer load test (L7) | io-prod/tests | Lands last; exercises finished timer + metrics. |

---

## R-4a — retransmit timer + confirm-dead-after-N (CORRECTED for C1, C2, H1, H2)

### 1.1 Struct/field changes (`ReliableLaneSender`, mesh.rs:397-420)

```rust
struct ReliableLaneSender {
    stream: Option<quinn::SendStream>,
    incarnation: u64,
    epoch: u32,
    next_seq: u64,
    retry: BTreeMap<u64, RetainedFrame>,   // H3/D5: was BTreeMap<u64, ReliableFrame>
    base: u64,
    retry_bytes: usize,                    // R-4b; kept in lockstep with retry (§3)
    // --- R-4a ---
    /// Consecutive FAILED replay CYCLES on THIS lane since its last successful (re)dial-and-write.
    /// Bumped by EXACTLY ONE per failed replay attempt of this lane (NOT per shared-connection drop);
    /// reset to 0 the instant this lane's own replay/write completes Ok. When it reaches
    /// `confirm_unreachable_after_retries` the writer bounces ONE NodeUnreachable for the lane and keeps
    /// re-bouncing (coalesced to the liveness refresh cadence, §1.5). The retained window is NEVER cleared
    /// by a bounce — bounce is a liveness alarm, not a discard. Bounded by the confirm threshold; never wraps.
    consecutive_failures: u32,
    /// The msg_id of the most recently ASSIGNED frame on this lane (set in the write path's assign step).
    /// The idle-timer bounce (which has no triggering OutFrame) carries THIS as `undelivered` — honest
    /// (points at the newest owed frame). `undelivered` is FIFO-diagnostic only, not redelivery-load-bearing
    /// (redelivery is driven wholly by `retry`/`epoch`). `None` ⇒ the lane never sent ⇒ never bounces. (D1)
    last_msg_id: Option<MsgId>,
}

/// A retained unacked frame plus its FROZEN worst-case-epoch framed length (H3). `framed_len` is the
/// `encode_frame` length computed at `epoch = u32::MAX` during `assign_and_retain` — the SAME number the
/// cap check used — so `retry_bytes` upper-bounds every future re-stamp and the increment (assign) and
/// decrement (on_ack) use ONE identical number. Storing the real-epoch length would undercount the on-wire
/// replay size after an epoch re-stamp grows the varint (1→5 bytes) — the exact R-2b oversize-drift class.
struct RetainedFrame {
    frame: ReliableFrame,
    framed_len: u32,
}
```

`ReliableLaneSender::new` initializes `consecutive_failures = 0`, `retry_bytes = 0`, `last_msg_id = None`.

### 1.2 The counter lives on the LANE and is driven by the LANE's own replay outcome (fixes C2)

**Do NOT bump `consecutive_failures` inside `on_write_error` (mesh.rs:498).** That method runs once per lane in the `for lane in lanes.values_mut()` fan-out (mesh.rs:1014), so a single shared-connection blip would bump every lane by 1 — conflating "one connection drop" with "N lanes each failed a replay." Instead:

- `on_write_error` keeps ONLY its existing job: `stream = None` + `epoch = wrapping_add(1)`. **Unchanged.**
- A new **`on_replay_failed(&mut self)`** bumps `consecutive_failures` by 1 and is called **exactly once per lane whose own replay attempt failed during a redial cycle** — from the corrected `replay_lanes` / `handle_write_error*` helpers below, never from the blanket fan-out.
- A new **`on_replay_ok(&mut self)`** sets `consecutive_failures = 0` and is called **only for a lane whose own write/replay just completed Ok** — never a blanket for-all reset (fixes H2).

```rust
impl ReliableLaneSender {
    fn on_replay_failed(&mut self) { self.consecutive_failures = self.consecutive_failures.saturating_add(1); }
    fn on_replay_ok(&mut self)     { self.consecutive_failures = 0; }
    fn owes_redelivery(&self) -> bool { self.stream.is_none() && !self.retry.is_empty() }
}
```

### 1.3 The retransmit-timer arm — per-lane guard, decoupled from connection state (fixes C1, H1)

**The guard is per-lane, NOT `connection.is_none()`:**

```rust
fn any_lane_owes(lanes: &BTreeMap<MsgClass, ReliableLaneSender>) -> bool {
    lanes.values().any(ReliableLaneSender::owes_redelivery) // stream closed AND window non-empty
}
```

This closes the verified C1 hole: after a send-arm success re-opens lane B and disarms on `connection.is_some()`, lane A (still `stream.is_none()`, `retry` non-empty) keeps `owes_redelivery() == true`, so the guard stays live and the timer re-drives lane A. The old `connection.is_none()` conjunct was the defect.

**Re-arm is edge-triggered at the iteration TAIL (fixes H1's self-contradiction).** After every arm's body, one deterministic re-evaluation:

```rust
// Called ONCE at the end of every loop iteration, after the winning arm's body.
fn rearm_retransmit(
    timer: Pin<&mut tokio::time::Sleep>,
    counting: &mut bool,
    lanes: &BTreeMap<MsgClass, ReliableLaneSender>,
    backoff: Duration,
    backoff_max: Duration,
) {
    let owed = any_lane_owes(lanes);
    if owed && !*counting {
        timer.reset(tokio::time::Instant::now() + backoff); // just became owed ⇒ arm
        *counting = true;
    } else if !owed && *counting {
        timer.reset(tokio::time::Instant::now() + backoff_max); // disarm to a harmless far deadline
        *counting = false;
    }
    // owed && counting ⇒ leave the live deadline UNTOUCHED (never push it back — that would stall retransmit).
}
```

`counting` is the `bool` the reviewer required: an already-counting timer is never pushed back; a disarmed-but-owed timer is always re-armed. There is provably no "owed but timer never armed" hole because `rearm_retransmit` runs at every iteration tail and re-derives `owed` from live lane state.

**The error paths set the deadline directly** (they own the backoff growth), and `rearm_retransmit` at the tail sees `counting == true` and leaves it. The exact loop:

```rust
async fn peer_writer(mut w: PeerWriter) {
    let mut connection: Option<quinn::Connection> = None;
    let mut lanes: BTreeMap<MsgClass, ReliableLaneSender> = BTreeMap::new();
    let mut backoff = w.backoff_min;
    let (ack_tx, mut ack_rx) = watch::channel::<Option<AckFrame>>(None);
    let mut ack_reader: Option<JoinHandle<()>> = None;

    // R-4a retransmit timer: reusable, pinned. Disarmed = a far-future deadline (guarded arm never polls it).
    let retransmit = tokio::time::sleep(w.backoff_max);
    tokio::pin!(retransmit);
    let mut counting = false;

    loop {
        tokio::select! {
            biased; // 1) acks retire promptly  2) new sends  3) retransmit LAST
            changed = ack_rx.changed() => {
                // UNCHANGED ack-retire body (mesh.rs:961-978): on_ack per entry, bump reliable_acked.
            }
            maybe = w.rx.recv() => {
                let Some(frame) = maybe else { break };
                let sent = write_frame(/* ...unchanged args...*/, &frame, w.reliability, &w.stats).await;
                match sent {
                    Ok(()) => {
                        backoff = w.backoff_min;
                        if let Some(l) = lanes.get_mut(&frame.class) { l.on_replay_ok(); } // reset THIS lane only (H2)
                    }
                    Err(WriteFail::Down) => {
                        handle_connection_drop(
                            &mut connection, &mut ack_reader, &mut lanes,
                            Some(&frame), &w.inbox, &w.stats, w.reliability,
                        );
                        retransmit.as_mut().reset(tokio::time::Instant::now() + backoff);
                        counting = true;
                        backoff = (backoff * 2).min(w.backoff_max);
                    }
                    Err(WriteFail::Shed) => { /* R-4b: shed already counted+bounced inside write_frame; nothing retained */ }
                }
            }
            () = &mut retransmit, if any_lane_owes(&lanes) => {
                // Idle-after-blip re-drive: dial-if-down, then replay EVERY owing lane's window, no new frame.
                let outcome = replay_lanes(
                    &w.endpoint, w.dest, w.addr, &mut connection, &mut lanes,
                    &mut ack_reader, &ack_tx, &w.connections, w.local, &w.stats, w.reliability,
                    &w.inbox,
                ).await;
                match outcome {
                    ReplayOutcome::AllOk => backoff = w.backoff_min,
                    ReplayOutcome::SomeFailed => {
                        // per-lane: replay_lanes already called on_replay_failed()/on_replay_ok() per lane and
                        // bounced any lane that crossed the threshold (§1.5). Just grow backoff + re-arm.
                        retransmit.as_mut().reset(tokio::time::Instant::now() + backoff);
                        counting = true;
                        backoff = (backoff * 2).min(w.backoff_max);
                    }
                }
            }
        }
        rearm_retransmit(retransmit.as_mut(), &mut counting, &lanes, backoff, w.backoff_max);
    }
    if let Some(h) = ack_reader.take() { h.abort(); }
}
```

**`PeerWriter` gains `reliability: MeshReliabilityTuning`** (threaded at spawn, mesh.rs:676-688: `reliability: cfg.reliability`).

**`replay_lanes`** (new, mirrors `write_frame`'s dial + open + `replay_batch` blocks, mesh.rs:1087-1172, with NO assign):
1. If `connection.is_none()`: dial via the **shared `ensure_connection` helper** (factored out of `write_frame` mesh.rs:1087-1111 — one dial home so ack-reader teardown/respawn cannot drift, fixes Race D). Dial failure ⇒ every owing lane `on_replay_failed()`, then confirm-check-and-bounce each (§1.5), return `SomeFailed`.
2. For **each lane where `stream.is_none() && !retry.is_empty()`** (only owing lanes; already-open lanes untouched): `open_uni` + `STREAM_KIND_DATA` tag + write `replay_batch()`. On Ok ⇒ `lane.on_replay_ok()`. On Err ⇒ `lane.on_replay_failed()` + confirm-check-and-bounce.
3. Return `AllOk` iff every owing lane replayed Ok, else `SomeFailed`.

### 1.4 Concurrency races — final resolution (verified)

- **Race A (double-dial/double-write):** `tokio::select!` runs exactly one arm body to completion per iteration; `connection`/`lane.stream` are owned by the single writer task (no `Arc`/lock). The send arm and timer arm never interleave. After the send arm brings the connection up, the timer's `if any_lane_owes` guard re-evaluates each poll; a lane with an open stream and drained window is not owing. `replay_lanes` uses ONLY the `stream.is_none()` → `replay_batch()` path (never assigns), so a frame is written exactly once. **No double-dial, no double-write.**
- **Race B (timer vs backoff):** the inline `tokio::time::sleep(backoff).await` at mesh.rs:1031 is REMOVED — it currently blocks the whole select (can't drain acks/sends during backoff). Backoff becomes the single timer deadline. One clock. Dead host is dialed at 50ms, 100ms, … capped 5s — the exact `redial_backoff_min..max` schedule, now non-blocking. **Verified: this also fixes a latent liveness bug in the current code.**
- **Race C (timer vs ack):** `biased` puts ack-retire first; a retire always wins in the same iteration, so an already-acked frame is never resent. `replay_batch` starts at `base` (mesh.rs:508 `retry.values()`), so a one-iteration-late ack still yields `Accept`/`Dedup`, never `Gap` (verified mesh.rs:381-390). **No spurious resend, no gap.**
- **Race D (ack-reader lifecycle):** `ensure_connection` is the ONE dial+ack-reader-respawn home called by both `write_frame` and `replay_lanes`. **No leaked/duplicated ack-reader.**

### 1.5 `confirm_unreachable_after_retries` bounce policy (fixes C2 + coalesces the re-bounce)

The R-3' per-frame bounce (mesh.rs:1020-1030) moves into a per-lane, threshold-gated, **coalesced** bounce:

```rust
/// Called on EACH lane whose own replay just FAILED (after on_replay_failed bumped it). Bounces ONE
/// NodeUnreachable iff the lane has crossed the confirm threshold AND has not bounced within the current
/// liveness refresh window (coalesce: at most one bounce per `ack_idle_flush_interval`-equivalent tick span,
/// so a multi-lane × multi-peer outage cannot flood the shared BoundedInbox and evict genuine reliable
/// inbound). A lane that never sent (last_msg_id == None) never bounces.
fn confirm_and_maybe_bounce(lane: &ReliableLaneSender, reliability: MeshReliabilityTuning,
                            inbox: &SharedInbox, stats: &MeshStats, to: NodeId, class: MsgClass) {
    if lane.consecutive_failures >= reliability.confirm_unreachable_after_retries {
        if let Some(msg_id) = lane.last_msg_id {   // D1: idle-timer bounce carries the newest owed frame's id
            push_inbox(inbox, stats, Inbound::NodeUnreachable { to, class, undelivered: msg_id });
        }
    }
}
```

**Reconciliation with buffer-first no-loss (the crux, verified):**
- **Bounce ≠ discard.** `on_write_error`/`on_replay_failed` never clear `retry`. A lane that bounces at N failures then RECOVERS still holds its full unacked window, replays it under the bumped epoch, and the receiver's ledger (`classify_reliable`, mesh.rs:327) delivers it exactly once (`Accept` from `base == hw+1`). **A bounced-then-recovered frame still delivers exactly once.** This is the R-3' → R-4' inversion: separate "transport is retaining and retrying" (always true while `retry` non-empty) from "the peer looks dead, alarm the saga layer" (the bounce).
- **Blip-tolerance restored (fixes C2):** the counter now counts a lane's OWN consecutive failed replay cycles, reset by that lane's own Ok. A single connection blip that recovers before N ⇒ **zero bounce** (one failed cycle, not N). Per-lane isolation: a stuck Saga lane bouncing does not touch a healthy Control lane's counter.
- **Re-bounce coalesced:** the `>=` keeps re-bouncing past threshold to keep the `LivenessTracker` fresh, but coalesced to the liveness refresh cadence so it cannot out-produce the BoundedInbox and evict a real grant/attach/revoke (verified against push_inbox reliable-drop, mesh.rs:251-265). This resolves the M-rebounce finding.

### 1.6 Coverage-seam documentation (M-open-stalled-lane, upheld)

**Division of labour, stated explicitly in the code doc + tested:**
- **R-4a timer** = redeliver a *closed-stream* window (`stream.is_none() && retry non-empty`). It does NOT fire for a lane whose stream is OPEN but whose reverse ack path is silently dead (half-open connection) — the guard `owes_redelivery()` is false there, so the timer correctly does not spin.
- **R-4b** = that live-but-stuck lane grows `retry_bytes` to the cap and **sheds-loud** (`reliable_shed`), with `reliable_acked` stuck-at-0 as the corroborating alarm.
- **Test (R-4b):** a stream-open lane whose acks stop grows `retry_bytes` to the cap and sheds (not silently wedges); assert the retransmit timer does NOT spin in that state (`counting` stays false / guard false).

---

## R-4b — shed policy (DECIDED) + `retry_bytes` (fixes H3, resolves D2)

### 3.1 The decided shed policy: NEVER shed a retained frame; refuse the NEW send

The receiver delivers strictly contiguously (verified mesh.rs:381-390: `seq==hw+1`→Accept, `seq>hw+1`→Gap-forever, `seq<=hw`→Dedup). Therefore dropping the oldest retained (`base == receiver_hw+1`) or any middle seq creates a **permanent contiguity wall**; dropping the newest silently loses a reliable frame the sim believed enqueued. **There is no loss-safe way to shed a retained frame.** The only correct policy is **producer backpressure: refuse the NEW send when the buffer is full, never drop a retained frame.**

**D2 — DECIDED (two-tier), not left open:**
1. **The real producer backpressure is the bounded outbound mpsc** (`cfg.outbound_capacity`, mesh.rs:675). `MeshTransport::send` is already a `try_send` returning `SendError::QueueFull(bytes)` with the payload handed back (verified mesh.rs:1237-1246) — the caller retries; nothing is lost. **Size the 4 MiB `retry_buffer_max_bytes` so the mpsc bound fills FIRST in normal operation** — the retry cap then trips only on a genuinely dead ack path.
2. **The writer-side retry-buffer-full is a `reliable_shed` + bounce**, because at that point the frame has already left `MeshTransport::send` and is in the writer's hands; the buffer is full only because acks stopped (a dead ack path — the M1 root cause). Shed-loud + the `reliable_acked`-stuck alarm IS the M1 cure. This reuses the exact R-2b refuse-count-bounce-don't-retain mechanism (verified mesh.rs:1128-1139).

### 3.2 `retry_bytes` accounting (fixes H3 — store the WORST-CASE-epoch length)

`assign_and_retain` (mesh.rs:465) becomes (three-way reject typed for R-4b):

```rust
enum AssignReject { Unframable, BufferFull }

fn assign_and_retain(&mut self, from: NodeId, class: MsgClass, bytes: &[u8], cap: usize)
    -> Result<u64, AssignReject> {
    let seq = self.next_seq;
    let mut frame = ReliableFrame { from, class, incarnation: self.incarnation,
                                    epoch: u32::MAX, seq, bytes: bytes.to_vec() };
    // ONE encode at worst-case epoch: reused for BOTH the oversize check AND the stored framed_len (H3).
    let encoded = vd_wire::framing::encode_frame(&frame).map_err(|_| AssignReject::Unframable)?;
    let framed_len = encoded.len() as u32;               // worst-case (u32::MAX epoch) — upper-bounds every re-stamp
    if self.retry_bytes.saturating_add(framed_len as usize) > cap {
        return Err(AssignReject::BufferFull);            // R-4b: transient, but writer-side ⇒ shed (D2 tier 2)
    }
    frame.epoch = self.epoch;                            // real epoch for the first write
    self.retry.insert(seq, RetainedFrame { frame, framed_len });
    self.retry_bytes += framed_len as usize;             // increment by the SAME number on_ack subtracts
    self.next_seq += 1;
    Ok(seq)
}
```

`on_ack` (mesh.rs:442) subtracts the stored `framed_len` when it removes a retired frame:

```rust
while self.base < self.next_seq && self.base <= ack_through {
    if let Some(rf) = self.retry.remove(&self.base) { self.retry_bytes -= rf.framed_len as usize; }
    self.base += 1;
}
```

`replay_batch` reads `.frame` and re-stamps epoch (unchanged behaviour; the stored `framed_len` is the frozen worst-case, so accounting never drifts even as the on-wire varint grows).

**Unit test (extend the mesh.rs:~1692 interleave sweep):** after arbitrary `assign_and_retain`/`on_ack`/`on_write_error` including ≥1 epoch re-stamp, assert `retry_bytes == retry.values().map(|rf| rf.framed_len as usize).sum()`.

### 3.3 Writer plug-in

`write_frame`'s reliable arm (mesh.rs:1128) three-ways on `AssignReject`: `Unframable` and `BufferFull` both → `reliable_shed.fetch_add(1)` + a distinct `tracing::warn!` (BufferFull's message cites "ack path likely dead — reliable_acked stuck") + `return Err(WriteFail::Shed)`. The `WriteFail::Shed` arm in `peer_writer` does NOT arm the retransmit timer and does NOT drop the connection (nothing was retained; the connection may be fine — only the ack path is dead).

---

## R-4c — `LivenessTuning::validate_against(&SagaTuning)` (fixes C3 — geometric backoff + off-by-one)

### 4.1 The corrected formula

The pre-threshold phase is **geometric** (verified mesh.rs:177-178 `50ms`→`5s`, mesh.rs:1031-1032 sleep-then-double), NOT linear at `retry_delay_ticks_hint`. And the replays-to-first-bounce is **N + n − 1**, not N + n (the N-th failed replay IS the first bounce). Model both phases:

```
// Pre-threshold: the geometric redial ladder for the first `confirm_after_retries` attempts, in ticks.
// backoff doubles from backoff_min, capped at backoff_max. Converted to ticks via the tick rate.
pre_ticks = ceil( sum_{k=0..confirm_after_retries-1} min(backoff_min * 2^k, backoff_max) / tick_dur )

// Post-threshold: (n_consecutive - 1) further bounces spaced at the down-peer redelivery cadence.
post_ticks = (n_consecutive_unreachable - 1) * retry_delay_ticks_hint

confirm_ticks = pre_ticks + post_ticks
INVARIANT:  confirm_ticks <= abort_deadline_ticks
```

Because vd-sim cannot depend on io-prod (verified dependency rule: sim → wire → core; io-prod is downstream), the geometric `pre_ticks` and the mirrored `confirm_unreachable_after_retries` are **bin-computed from the real `MeshReliabilityTuning` + `redial_backoff_min/max` + tick rate and injected into `LivenessTuning` as pre-derived fields**, then boot-asserted equal to the transport's values. This keeps the invariant *logic* Tier-A-100%-testable in vd-sim (D3 resolution (a), per the prompt's mandate).

### 4.2 Placement + signature (vd-sim, saga.rs, after line 204)

`LivenessTuning` gains two mirror fields:
```rust
pub confirm_unreachable_after_retries: u32,  // mirrors MeshReliabilityTuning; boot-asserted equal
pub redial_confirm_ticks_hint: u64,          // the bin-computed geometric pre_ticks (io-prod backoff→ticks)
```
```rust
impl LivenessTuning {
    /// R-4c cross-struct invariant: the transport's dead-confirmation must resolve INSIDE the saga's
    /// destructive abort window, else the saga force-aborts a possibly-healthy player before the transport
    /// has a dead-vs-slow verdict (the abort storm D-3 evidence-gating exists to prevent). The pre-threshold
    /// phase is the io-prod GEOMETRIC redial ladder (pre-computed by the bin into `redial_confirm_ticks_hint`,
    /// NOT re-modelled linearly here); the post-threshold phase is (n-1) notices at `retry_delay_ticks_hint`.
    /// Replays-to-first-bounce is confirm_after_retries; further bounces are n-1 (off-by-one: NOT n+confirm).
    ///
    /// # Errors
    /// [`LivenessTuningError::ConfirmOutrunsAbort`] when the confirmation window exceeds the abort deadline.
    pub fn validate_against(&self, saga: &SagaTuning) -> Result<(), LivenessTuningError> {
        let post = self.retry_delay_ticks_hint
            .saturating_mul((self.n_consecutive_unreachable.saturating_sub(1)) as u64);
        let confirm_ticks = self.redial_confirm_ticks_hint.saturating_add(post);
        if confirm_ticks > saga.abort_deadline_ticks {
            return Err(LivenessTuningError::ConfirmOutrunsAbort { confirm_ticks, abort: saga.abort_deadline_ticks });
        }
        Ok(())
    }
}
```
Add `ConfirmOutrunsAbort { confirm_ticks: u64, abort: u64 }` (derives PartialEq) to `LivenessTuningError`; extend `validate` to reject `confirm_unreachable_after_retries < 1` (mirrors the mesh reject, verified mesh.rs:142). The bin adds `liveness.validate_against(&saga)?;` right after `liveness.validate()?;` (orchestrator.rs:116) and asserts `liveness.confirm_unreachable_after_retries == cfg.reliability.confirm_unreachable_after_retries`.

**HR5(d):** the arm is equality-asserted with `expect_err`, no `matches!`; the `>` is one monomorphic branch. **Unit test the C3-motivated case:** fast abort (`abort_deadline_ticks` small) + slow geometric backoff (`redial_confirm_ticks_hint` large) is REJECTED at boot — the exact "confirm outruns abort" hole the original linear formula silently passed.

---

## R-4d — MetricsSource wiring (M4; 9 names, not 7)

### 5.1 wire/admin.rs — register 9 names + append to `ALL` (fixes M-metrics)

`MeshStatsSnapshot` has 10 fields (mesh.rs:234-245); `DATAGRAMS_DROPPED_TOO_LARGE` is already registered (wire/admin.rs:143). Add the **9 remaining** to `metric_names` (mesh.rs:126-157) and append all 9 to `ALL`:
```
vd_datagrams_dropped_send_total, vd_inbound_dropped_reliable_total, vd_inbound_dropped_unreliable_total,
vd_stale_incarnation_drop_total, vd_stale_epoch_drop_total, vd_dedup_drop_total,
vd_gap_drop_total (MUST-BE-0 alert), vd_reliable_shed_total, vd_reliable_acked_total (stuck-at-0 = dead ack path)
```
The existing conformance tests then cover them automatically: `metric_names_are_unique_and_prometheus_valid` (wire/admin.rs:280) + `metrics_endpoint_exposes_every_registered_name` (io-prod/admin.rs:120). **Add one test:** `MeshMetrics::values()` has exactly one key per `MeshStatsSnapshot` field AND every such key is in `metric_names::ALL` — because `render_metrics` keys off `ALL` (an unregistered counter is silently never scraped, defeating GW-1).

### 5.2 io-prod/admin.rs — `MetricsSource` trait (replace the hardcoded-0 shell, mesh.rs:48-62)

```rust
pub trait MetricsSource: Send + Sync + 'static { fn values(&self) -> BTreeMap<&'static str, u64>; }
pub struct ZeroMetrics;                       // the P0 shell behaviour: every registered name at 0
impl MetricsSource for ZeroMetrics { fn values(&self) -> BTreeMap<&'static str, u64> { BTreeMap::new() } }

pub fn render_metrics(source: &dyn MetricsSource) -> String {
    let vals = source.values();
    let mut out = String::new();
    for name in metric_names::ALL {                    // every registered name always rendered; unfed ⇒ 0
        let v = vals.get(name).copied().unwrap_or(0);
        let _ = writeln!(out, "# TYPE {name} untyped");
        let _ = writeln!(out, "{name} {v}");
    }
    out
}
```
`admin_router(snapshot, metrics)` bundles both sources into `AdminState { snapshot: Arc<dyn SnapshotSource>, metrics: Arc<dyn MetricsSource> }` (axum single-`State`). `metrics_handler` reads `state.metrics`.

### 5.3 `MeshMetrics` source (io-prod) — verified off-seam/off-hot-path

```rust
pub struct MeshMetrics { control: Arc<MeshControl> }
impl MetricsSource for MeshMetrics {
    fn values(&self) -> BTreeMap<&'static str, u64> {
        let s = self.control.stats(); // pure atomic load (mesh.rs:573), on the axum task — NOT the tick thread
        BTreeMap::from([ /* 10 (name → field) pairs incl. DATAGRAMS_DROPPED_TOO_LARGE */ ])
    }
}
```
Verified: `MeshControl::stats()` (mesh.rs:573) is a lock-free `Ordering::Relaxed` atomic load returning `MeshStatsSnapshot` — no seam change, no 20 Hz touch, scrape decoupled from the writer/sim threads.

### 5.4 Bin wiring (+ D4 decision)

- **Orchestrator:** capture `_control`→`control`, `Arc`-wrap, pass `admin_router(Arc::new(Published(served)), Arc::new(MeshMetrics { control }))` (orchestrator.rs:250).
- **D4 — DECIDED: add a metrics-ONLY admin listener to gateway + shard**, bound **loopback-only** (`127.0.0.1`), with an explicit **CAF-2 note in the code** (these bins previously mounted no admin endpoint — verified gateway.rs:21 / shard.rs:22 discard `_control`; adding a listener expands the CAF-2 loopback/auth contract, so it must be loopback-bound and documented). `VD_ADMIN_ADDR` plumbing already exists via the slot tooling. Their per-node `reliable_acked`-stuck / `gap_drop`>0 alarms are exactly the invisibility M4 flags. A `metrics_only_router(metrics)` (no snapshot) since they have no directory. **If loopback-only binding is unacceptable for the deployment, DEFER D4** and land only the orchestrator (which already has the listener) — R-4d is green either way.
- **Client:** leave `control` as-is; the client is not ops-monitored (its counters serve E2E debugging via dev-control). **Defer.**

---

## R-4e — N-peer load test (L7)

New `crates/io-prod/tests/mesh_load.rs`. N∈16..=64 senders (`N=32` default; `#[ignore]`d `N=64` soak) each sustain RELIABLE Saga traffic into one receiver. Reuse `reserve`/`node`/`wait_until` from mesh_redelivery.rs (verified present, mesh_redelivery.rs:30/38/72 — lift into a shared test util). Each sender `s` sends `(s, round)`-tagged 2-byte payloads. Asserts: (1) **no-loss + no-dup** — receiver gets exactly `N*BURST` unique frames, each once; (2) **`reliable_acked` keeps pace** — each sender's `ctl.stats().reliable_acked` reaches `BURST` (ack round-trip scales, no dead-ack-path under fan-in); (3) **no RX-plane collapse** — completes within a generous WALL-CLOCK bound (empirically surfaces H2: the node-wide `RecvLedger` `Mutex` contention on every reliable frame from every sender — measured, not asserted-lock-free, since H2 is deferred). NO `drop_connections` blip (that's R-5'), single Saga class (clean no-loss assertion). The fan-IN companion to the existing `mesh_volume_all_pairs_burst`.

---

## Test plan (per slice)

- **R-4a:** (1) unit — idle-after-blip single-Saga-lane on a dead peer re-drives on the timer with NO new frame and delivers exactly once on recovery (the C1 scenario: Saga lane owed while a Control frame recovers, assert Saga still redelivers); (2) unit — a single blip that recovers before N ⇒ `consecutive_failures` returns to 0, ZERO bounce (C2); (3) unit — lane A's blip does not reset lane B's counter (H2); (4) unit — at threshold N, exactly one bounce, coalesced re-bounce past threshold; (5) integration (real QUIC) — kill-9 peer, assert the timer confirms dead within the backoff schedule and the retained window survives. Assert the timer does NOT spin when no lane owes.
- **R-4b:** (1) unit `retry_bytes == Σ framed_len` after an interleave incl. an epoch re-stamp (H3); (2) unit — buffer-full ⇒ `AssignReject::BufferFull` ⇒ `reliable_shed++` + no retention; (3) unit — a stream-open lane whose acks stop grows to the cap and sheds-loud, timer does NOT spin (M-open-stalled-lane).
- **R-4c:** Tier-A 100% — `validate_against` Ok on defaults (`n=1`) and prod (`n=3`); **reject** the fast-abort + slow-geometric-backoff case (C3); off-by-one boundary (`confirm_ticks == abort_deadline_ticks` is Ok, `+1` is Err); bin boot-assert that the mirror field equals `cfg.reliability`.
- **R-4d:** the two existing conformance tests + the new `values()`-keys-⊆-`ALL`-and-⊇-snapshot-fields test; a `MeshMetrics` scrape reflects a bumped counter.
- **R-4e:** as above.

**`just gate` after each slice** (fmt + clippy `-D warnings` + tests + coverage). io-prod stays Tier-B floor 90; the pure FSM additions are 100%-unit-coverable per the mesh.rs pure-FSM pattern. vd-sim `validate_against` is Tier-A 100% (equality arms, no `matches!`, no short-circuit).

## Cross-cutting constraint check (verified)

- **Frozen seam untouched** — all changes below `Transport`/`Inbound`/`MsgClass`; `consecutive_failures`/`retry_bytes`/`last_msg_id`/timer never become a seam arm. `NodeUnreachable` already exists (mesh.rs:1024); R-4a only changes WHEN it fires. ✓
- **20 Hz datagram path byte-identical** — only the reliable lane FSM + peer_writer select + saga tuning + admin change; the `Reliability::Unreliable` arm (mesh.rs:1189) and `DatagramFrame` are untouched. ✓
- **No magic numbers** — thresholds in `MeshReliabilityTuning`; timer cadence reuses `redial_backoff_min/max`; the disarmed-timer deadline reuses `backoff_max` (a harmless far deadline the guard ignores — no new sentinel constant, resolving D6). ✓
- **Cancel-safety** — `replay_lanes` runs its `write_all`s to completion inside the arm body (like `write_frame`, mesh.rs:1069-1071 contract); the only cancel points remain `w.rx.recv()`, `ack_rx.changed()`, and `&mut retransmit` (a bare `Sleep`). ✓
- **HR3 one-tooling / no per-kind fork** — one `MeshTransport`, one lane FSM, one tuning struct, one dial helper (`ensure_connection`), one metrics trait fed identically by every bin. ✓

**FINAL: SOUND TO IMPLEMENT** with the four corrections folded in (C1 per-lane guard + iteration-tail edge-triggered re-arm; C2 counter driven by the lane's own replay outcome not the connection-drop fan-out; C3 geometric-backoff bin-computed `redial_confirm_ticks_hint` + off-by-one `n-1`; H3 worst-case-epoch `RetainedFrame.framed_len`), the H2 single-lane reset rule, the decided shed policy (writer-side buffer-full = `reliable_shed`; mpsc bound = the real producer backpressure), 9 metric names, D1 `last_msg_id`, and D4 loopback-only-or-defer. Key files: `crates/io-prod/src/mesh.rs` (ReliableLaneSender 397-420, on_ack 442, assign_and_retain 465, on_write_error 498, replay_batch 508, MeshControl::stats 573, PeerWriter 924, peer_writer 947-1043, write_frame 1073-1188, MeshReliabilityTuning 97-147, MeshStats/Snapshot 191-245); `crates/sim/src/saga.rs` (SagaTuning 68-125, LivenessTuning 132-204 — validate_against after 204); `crates/wire/src/admin.rs` (metric_names 125-157); `crates/io-prod/src/admin.rs` (render_metrics_shell 48-62, admin_router 37); `crates/bins/src/bin/orchestrator.rs` (validate 82/116, admin 244-250); `crates/bins/src/bin/{gateway.rs:21,shard.rs:22}` (D4); `crates/io-prod/tests/mesh_redelivery.rs` (reserve/node/wait_until 30/38/72).