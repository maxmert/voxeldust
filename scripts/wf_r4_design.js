export const meta = {
  name: 'r4-design',
  description: 'Design + adversarially review R-4 (retransmit timer + confirm-dead + shed-loud + tuning invariant + metrics) before implementation',
  phases: [
    { title: 'Design', detail: 'concrete R-4 design from the real code' },
    { title: 'Review', detail: '3 adversarial lenses' },
    { title: 'Synthesize', detail: 'vetted implement-ready design' },
  ],
}

const CONTEXT = `
Voxeldust redelivering (at-least-once) MeshTransport. R-1'..R-3' are LANDED (commit 7c28d50). The mesh is now
at-least-once across a connection blip FOR A (peer,class) LANE THAT KEEPS CARRYING TRAFFIC. R-4' closes the remaining
gaps. READ the real code before designing: crates/io-prod/src/mesh.rs (peer_writer, write_frame, ReliableLaneSender,
MeshConfig, MeshReliabilityTuning, MeshStats, MeshControl::stats), crates/io-prod/src/lib.rs, crates/io-prod/src/admin.rs
(render_metrics_shell), crates/wire/src/admin.rs (metric_names::ALL), crates/sim/src/saga.rs (LivenessTuning + its
validate + n_consecutive_unreachable + abort_deadline_ticks), and DEFERRED.md D-6 (the R-4' owed items + the R-3' holistic
audit H2/M4/M1/L7 findings).

CURRENT R-3' SHAPE (what R-4' builds on):
 - peer_writer (mesh.rs ~947): a biased tokio::select! over ack_rx.changed() (a watch<Option<AckFrame>>) and w.rx.recv().
   On a write Err: connection=None, abort the ack-reader, on_write_error() each lane (epoch bump), bounce ONE
   NodeUnreachable for the failed frame, sleep(backoff), backoff doubles.
 - write_frame: dials on connection.is_none() (registers the conn, spawns the ack-reader, writes STREAM_KIND_DATA);
   the re-dial + replay_batch of the unacked window happens ONLY inside this dial block ⇒ ONLY when the next OutFrame
   arrives (the idle-after-blip gap — a lane that blips then goes quiet strands its unacked tail in lane.retry).
 - ReliableLaneSender{stream, incarnation, epoch, next_seq, retry:BTreeMap<u64,ReliableFrame>, base}; on_ack retires the
   prefix; retry GROWS un-drained if the ack path is dead (no byte cap consumed yet).
 - MeshReliabilityTuning{retry_buffer_max_bytes, ack_idle_flush_interval, confirm_unreachable_after_retries} — all three
   validated at boot but ONLY retry_buffer_max_bytes has ANY consumer (the framed-oversize reject); ack_idle_flush is
   consumed by R-3' ack_egress; confirm_unreachable_after_retries has NO consumer yet.
 - MeshStats: 7 never-silent counters (datagrams_dropped_too_large/send, inbound_dropped_reliable/unreliable,
   stale_incarnation_drop, stale_epoch_drop, dedup_drop, gap_drop, reliable_shed, reliable_acked). NONE are on the
   ops /metrics surface (admin render_metrics_shell hardcodes registered names to 0; metric_names::ALL omits them).

R-4' MUST DELIVER (design each; propose sub-slicing if warranted):
 1. RETRANSMIT/REDIAL TIMER: a peer_writer timer arm that, when a lane has a non-empty retry buffer AND the connection
    is down (post-error), RE-DRIVES the dial + replay_batch WITHOUT waiting for a new OutFrame — closing the idle-after-blip
    gap. Must NOT cause a connect-storm against a dead host (respect/reuse the existing exponential backoff), must NOT race
    the existing write path into a double-dial or double-write, must interact correctly with the biased select + the ack arm.
 2. confirm_unreachable_after_retries: bounce NodeUnreachable only after N CONSECUTIVE failed replays of a lane (a transient
    blip that recovers = ZERO bounce). Where is the per-lane consecutive-failure counter; how does a success reset it; what
    exactly bounces (the whole unacked window? one representative?) and how does that reconcile with the buffer-first
    no-loss guarantee (a bounced-then-recovered frame must still deliver exactly once).
 3. retry_bytes + SHED-LOUD: bound the per-lane retry buffer by retry_buffer_max_bytes. On overflow: reliable_shed++ +
    a bounce + drop the SHED frame (the M1 unbounded-growth cure). Which end sheds (oldest unacked? newest?); how does
    shedding interact with contiguity (shedding a middle seq creates a permanent gap the receiver can never fill — is the
    only safe shed the NEWEST, refusing new sends when full, i.e. producer backpressure, NOT dropping retained frames?).
 4. LivenessTuning::validate_against(&SagaTuning) (or the equivalent cross-struct invariant, vd-sim Tier-A 100%): the
    transport's retry+confirm timing (N retries * backoff) must fit INSIDE the saga's abort_deadline_ticks, or a lane
    declares a peer dead AFTER the saga already force-aborted (or vice versa). Called at orchestrator boot, fail-loud.
 5. METRICS/OPS SURFACE (M4): wire the MeshStats counters to /metrics WITHOUT touching the frozen seam or the 20Hz hot
    path — add the 7 names to the review-gated metric_names::ALL, replace render_metrics_shell's hardcoded 0 with a
    MetricsSource (mirror the existing SnapshotSource), have the shard/gateway/orch bins hold their MeshControl + feed it.
 6. (companion) An N-peer (16-64) sustained-reliable-into-one-node LOAD test (L7) — no-loss/no-dup, reliable_acked keeps
    pace, no receive-plane collapse under a wall-clock bound; the test that empirically surfaces the H2 RecvLedger contention.

CONSTRAINTS: below the frozen Transport/Inbound/MsgClass seam (sim/node unchanged; acks/timers never a MsgClass or Inbound);
no magic numbers (all tuning in MeshReliabilityTuning/the ONE config struct); no cancel-safety violations (never hold a
write_all across a select!); the 20Hz unreliable datagram hot path stays byte-for-byte untouched; HR5 (io-prod Tier-B floor
90; vd-sim Tier-A 100% for any sim-side tuning); no per-kind/per-shard fork.
`

const DESIGN = await (async () => {
  phase('Design')
  return agent(
    `${CONTEXT}\n\nProduce the CONCRETE, implement-ready R-4' design. For EACH of the 6 deliverables: the exact struct/field
    changes, the exact peer_writer select-loop shape (with the timer arm) as Rust, the per-lane failure-counter placement,
    the shed policy decided (with the contiguity reasoning — is producer-backpressure-when-full the only safe shed?), the
    LivenessTuning invariant formula, the MetricsSource wiring, and the load-test shape. Call out every concurrency race you
    resolve (timer vs backoff vs the write path vs the ack arm) and how. Propose the sub-slicing (R-4a/R-4b/...) if landing
    it as one slice is unwise. Flag every open decision that needs a call. Cite mesh.rs line numbers.`,
    { label: 'design', phase: 'Design', model: 'opus' }
  )
})()

const REVIEW_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['lens', 'verdict', 'blocking'],
  properties: {
    lens: { type: 'string' },
    verdict: { type: 'string', enum: ['SOUND_TO_IMPLEMENT', 'FIX_BEFORE_IMPL'] },
    blocking: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'problem', 'fix'],
        properties: {
          severity: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM'] },
          problem: { type: 'string' },
          fix: { type: 'string' },
        },
      },
    },
  },
}

const LENSES = [
  {
    key: 'timer-concurrency',
    prompt: `Adversarially review this R-4' DESIGN for the RETRANSMIT-TIMER concurrency. Read the real mesh.rs peer_writer/write_frame first. Hunt: a connect-storm (timer + backoff interacting to hammer a dead host), a double-dial or double-write (timer re-drives while a normal send also dials), a lost wakeup (the timer arm never fires because the select is parked on a write), a cancel-safety break (a write_all held across the timer select arm), the timer racing the ack arm or the write-error arm into an inconsistent lane state, and whether the timer correctly STOPS once the lane drains (retry empty) so it doesn't spin. Verify the idle-after-blip gap is ACTUALLY closed (a lone frame that blips then idles is re-driven). VERDICT + blocking findings.\n\n---DESIGN---\n${DESIGN}`,
  },
  {
    key: 'shed-backpressure',
    prompt: `Adversarially review this R-4' DESIGN for SHED / BACKPRESSURE / confirm-dead CORRECTNESS. Read mesh.rs ReliableLaneSender + on_ack + on_write_error. Hunt: does shedding a retained frame EVER create a permanent contiguity gap the receiver can't fill (shedding a middle/old seq)? Is producer-backpressure-when-full (refuse the NEW send, keep the retained window) the only loss-safe shed, and does the design pick it? Does confirm_unreachable_after_retries preserve the buffer-first no-loss guarantee (a bounced-then-recovered frame still delivers exactly once)? Does the per-lane consecutive-failure counter reset correctly on success, and can it double-bounce or never-bounce? Does the shed counter (reliable_shed) stay never-silent? Does retry_bytes accounting match the framed size (the R-2b oversize lesson)? VERDICT + blocking findings.\n\n---DESIGN---\n${DESIGN}`,
  },
  {
    key: 'integration-ops-tuning',
    prompt: `Adversarially review this R-4' DESIGN for INTEGRATION, the LivenessTuning invariant, and the METRICS/OPS surface. Read saga.rs LivenessTuning + admin.rs (both io-prod render_metrics_shell and wire metric_names::ALL) + a bin (orchestrator.rs) that mounts the admin router. Hunt: is the LivenessTuning::validate_against formula CORRECT (retry*backoff vs abort_deadline_ticks — units/ticks-vs-Duration mismatch?) and does it fail loud at the right boot point? Does the MetricsSource wiring touch the frozen seam or the hot path (it must NOT)? Is metric_names::ALL the review-gated registry, and are the 7 names added with alert semantics? Does any bin actually hold MeshControl to feed stats (today they discard _control)? Does the design keep sim/node unchanged? Is the N-peer load test real (many peers into ONE node's reliable RX plane, not the mem hub)? Is the sub-slicing sensible? VERDICT + blocking findings.\n\n---DESIGN---\n${DESIGN}`,
  },
]

phase('Review')
const reviews = (await parallel(
  LENSES.map((l) => () =>
    agent(l.prompt, { label: `review:${l.key}`, phase: 'Review', schema: REVIEW_SCHEMA, model: 'opus' })
  )
)).filter(Boolean)

const blocking = reviews.flatMap((r) => (r.blocking || []).map((b) => ({ ...b, lens: r.lens })))
log(`reviews: ${reviews.map((r) => `${r.lens}=${r.verdict}`).join(', ')}; ${blocking.length} blocking`)

phase('Synthesize')
const vetted = await agent(
  `You are the synthesizer for the R-4' pre-implementation design review. Read the real code yourself to adjudicate. The DESIGN:\n\n${DESIGN}\n\nReviewer verdicts + blocking findings (JSON):\n${JSON.stringify(reviews, null, 2)}\n\nProduce the FINAL VETTED, IMPLEMENT-READY R-4' design with every real blocking finding FOLDED IN (drop false alarms with reasons). Give: the resolved struct/field changes, the exact peer_writer select loop with the timer arm as Rust, the shed policy (decided), the confirm-dead counter, the LivenessTuning invariant, the MetricsSource wiring, the load test, the sub-slicing, and the test plan. State clearly whether it is SOUND TO IMPLEMENT as specified. Cite file:line.`,
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return { design_len: DESIGN.length, verdicts: reviews.map((r) => ({ lens: r.lens, verdict: r.verdict })), blocking_count: blocking.length, blocking, vetted }
