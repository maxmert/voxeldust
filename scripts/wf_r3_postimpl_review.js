export const meta = {
  name: 'r3-postimpl-review',
  description: 'Adversarial review of the REAL landed R-3 receiver/ack code (mesh.rs + lib.rs) before commit',
  phases: [
    { title: 'Review', detail: '3 adversarial lenses on the real code' },
    { title: 'Synthesize', detail: 'consolidate + verdict' },
  ],
}

const FILES = [
  'crates/io-prod/src/mesh.rs',
  'crates/io-prod/src/lib.rs',
  'crates/io-prod/tests/mesh_redelivery.rs',
  'crates/bins/src/lib.rs (common_env / launch_incarnation only)',
]

const CONTEXT = `
This reviews the LANDED R-3' slice of the redelivering mesh transport (at-least-once over quinn). Read the
REAL code in ${FILES.join(', ')}. Do NOT trust this description over the code — verify every claim against it.

WHAT R-3' ADDED (the receiver + ack lane; the sender lane FSM R-2b already existed):
- RecvState{incarnation:u64, epoch:u32, hw:u64, primed:bool} + a node-wide RecvLedger (Arc<Mutex<BTreeMap<(NodeId,MsgClass),RecvState>>>) that SURVIVES connection teardown.
- classify_reliable(&mut RecvState, inc, epoch, seq) -> Verdict{Accept|Reset|StaleIncarnation|StaleEpoch|Dedup|Gap}, a PURE ladder: incarnation ▸ epoch ▸ contiguity. StaleEpoch is decided BEFORE any hw compare (the reverted-R-1 cross-stream cure). The higher-incarnation A1 arm resets to {hw:0,primed:false} and FALLS THROUGH to prime_or_contiguous (primes ONLY at seq0; a fresh-incarnation first-frame seq>0 is a Gap, never a silent Dedup).
- classify_and_deliver: takes the ledger lock, classifies, delivers via push_inbox UNDER THE LOCK (push is sync), and on a reliable inbox-drop ROLLS BACK the whole RecvState to its pre-classify snapshot (advancing hw without delivering = permanent loss).
- serve_data_stream: reads a 1-byte STREAM_KIND tag via read_exact(1) BEFORE the framing loop, then DATA→classify loop; ACK/unknown/torn→drop wholesale (never touch the ledger). read_one_reliable_frame is UNCHANGED (the loopback bridge shares it).
- ack_egress (in the data RECEIVER's serve_connection): opens ONE reverse STREAM_KIND_ACK uni stream ON THE SAME accepted connection; a select loop DECIDES to flush (Notify vs interval), the write_ack_frame().await is OUTSIDE the select (cancel-safety); snapshots per-(peer,class) (epoch,hw) under one ledger lock hold then DROPS the guard before the await; dedups identical acks; echoes the sender's incarnation from RecvState.
- peer_writer (the data SENDER): a biased tokio::select! over ack_rx.changed() (a watch channel) and rx.recv(); an ack_reader_task child accept_uni's the reverse ACK stream on ITS OWN dialed connection and feeds the watch; the reader is aborted on the write-error path and on loop exit. on_ack retires the prefix: epoch+incarnation matched, clamped to next_seq, monotone base.
- MeshControl.drop_connections(): closes every dialed connection (a ConnRegistry keyed by dest NodeId, latest-wins) — a transient blip, DISTINCT from kill() (endpoint close).
- common_env exports a per-launch wall-clock-ms VD_PROCESS_INCARNATION.

KEY DECISIONS TO SCRUTINIZE (a prior design review demanded a MANDATORY AckRouter; the synthesizer REJECTED it in favor of the same-connection reverse stream, claiming an AckRouter couples two connections' liveness = deadlock. VERIFY this is actually correct and that the same-connection topology has no gap). Also scrutinize the inbox-drop ROLLBACK refinement (does reverting the whole RecvState — including an epoch adopt or incarnation reset — re-open the reverted-R-1 CRITICAL or lose a frame?).

The reverted-R-1 CRITICAL (a6f8e7d/731b377): per-stream reader tasks shared one high-water with NO cross-stream ordering, so a write-error-then-redial let a HIGHER seq win the race and SILENTLY drop a LOWER never-delivered frame. R-3' must NOT re-open this in ANY path.
`

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['lens', 'verdict', 'findings'],
  properties: {
    lens: { type: 'string' },
    verdict: { type: 'string', enum: ['SHIP', 'FIX_BEFORE_COMMIT'] },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'location', 'problem', 'fix'],
        properties: {
          severity: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'] },
          location: { type: 'string', description: 'file:line or fn name' },
          problem: { type: 'string', description: 'the concrete defect, verified against the real code' },
          fix: { type: 'string' },
        },
      },
    },
  },
}

const LENSES = [
  {
    key: 'receiver-race',
    prompt: `${CONTEXT}\n\nLENS: RECEIVER VERDICT & CROSS-STREAM RACE. Hunt for ANY path where a reliable frame is SILENTLY dropped when it should be delivered, or DELIVERED TWICE, or where the high-water advances past a never-delivered seq. Trace classify_reliable + prime_or_contiguous every arm, the classify_and_deliver ROLLBACK on a reliable inbox-drop (does reverting a whole RecvState that had an epoch-adopt or A1-reset re-open the reverted CRITICAL, or strand a frame?), the interaction of the surviving node-wide ledger with two live serve_connections for the same peer (old-connection straggler + new connection), and the STREAM_KIND demux (can a torn/garbage stream move the hw?). Assume an adversary controls frame arrival order across streams/redials. Confirm the reverted-R-1 CRITICAL is NOT re-opened in any path.`,
  },
  {
    key: 'ack-lifecycle',
    prompt: `${CONTEXT}\n\nLENS: ACK LIFECYCLE, CANCEL-SAFETY & DEADLOCK. Verify: (1) the ack_egress write is truly cancel-safe (write_ack_frame OUTSIDE the select; a torn AckFrame cannot desync the reader). (2) NO lock (RecvLedger / acked_keys / inbox) is ever held across an .await, and the lock ACQUISITION ORDER across ALL tasks (classify_and_deliver: ledger→inbox; ack_egress: acked_keys→ledger; serve_data_stream; peer_writer) has NO cycle → no deadlock. (3) on_ack cannot advance base past next_seq or roll it back under any stale/duplicate/forged/torn ack; the epoch+incarnation guard is complete. (4) the ack_reader_task has NO leak and NO stale-reader cross-talk on redial; the watch latest-wins semantics are safe. (5) the incarnation ECHOED in the AckFrame is always the value the sender's on_ack will match (is the one-connection-one-incarnation assumption actually sound, incl. across a peer restart?).`,
  },
  {
    key: 'topology-concurrency',
    prompt: `${CONTEXT}\n\nLENS: TOPOLOGY & CONCURRENCY INTEGRATION. Verify the same-connection reverse-ACK topology is CORRECT and complete on the real directional mesh (peer_writer owns the dialed conn + hosts the ack-reader; serve_connection owns the accepted conn + hosts ack_egress). Check the MUTUAL case (A dials B AND B dials A → two connections): does each direction's ack reach the right peer_writer with NO cross-talk? Check that the AckRouter rejection is justified (would it really deadlock?). Check drop_connections: does closing a dialed conn tear down BOTH data + reverse-ack coherently; is the ConnRegistry bounded (no unbounded growth); can drop_connections race the writer into a bad state? Check the STREAM_KIND wire change did not break the loopback bridge or the datagram hot path (both must be byte-identical). Check task shutdown (serve_connection aborts streams+ack_task; peer_writer aborts the ack-reader) leaves no orphan tasks. Check the common_env wall-clock incarnation for a genuine collision residual.`,
  },
]

phase('Review')
const reviews = await parallel(
  LENSES.map((l) => () =>
    agent(l.prompt, { label: `review:${l.key}`, phase: 'Review', schema: FINDINGS_SCHEMA, model: 'opus' })
  )
)

const valid = reviews.filter(Boolean)
const allFindings = valid.flatMap((r) => (r.findings || []).map((f) => ({ ...f, lens: r.lens })))
const blocking = allFindings.filter((f) => f.severity === 'CRITICAL' || f.severity === 'HIGH')
log(`reviews: ${valid.map((r) => `${r.lens}=${r.verdict}`).join(', ')}; ${blocking.length} CRITICAL/HIGH`)

phase('Synthesize')
const synthesis = await agent(
  `You are the synthesizer for the R-3' post-implementation review. Read the REAL code in ${FILES.join(', ')} yourself to ADJUDICATE each finding (reviewers can be wrong in BOTH directions — a false alarm or a missed real bug).\n\nReviewer verdicts + findings (JSON):\n${JSON.stringify(valid, null, 2)}\n\nProduce: (1) a de-duplicated, ADJUDICATED list of REAL CRITICAL/HIGH findings (drop false alarms, note WHY), each with the exact code fix; (2) an overall verdict SHIP or FIX_BEFORE_COMMIT; (3) a one-paragraph bottom line. Be adversarial but precise — cite file:line. If a reviewer flagged the inbox-drop rollback or the AckRouter rejection, adjudicate it decisively against the real code.`,
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return { verdicts: valid.map((r) => ({ lens: r.lens, verdict: r.verdict })), blocking_count: blocking.length, blocking, synthesis }
