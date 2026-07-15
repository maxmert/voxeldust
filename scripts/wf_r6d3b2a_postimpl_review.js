export const meta = {
  name: 'r6d3b2a-postimpl-review',
  description: 'Adversarial post-impl review of R-6d3b-2a (io-prod boot-replay machinery: deadlock-free + count-fence)',
  phases: [
    { title: 'Review', detail: '3 opus lenses: deadlock-fence-fidelity / findings-fidelity / hr-seam-scope-test' },
    { title: 'Synthesize', detail: 'dedup + rank + verdict' },
  ],
}

const FINDINGS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['lens', 'verdict', 'summary', 'findings'],
  properties: {
    lens: { type: 'string' },
    verdict: { type: 'string', enum: ['COMMIT_CLEAN', 'FIX_BEFORE_COMMIT', 'RECONSIDER'] },
    summary: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['severity', 'title', 'location', 'claim', 'evidence', 'fix', 'confidence'],
        properties: {
          severity: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NIT'] },
          title: { type: 'string' },
          location: { type: 'string' },
          claim: { type: 'string' },
          evidence: { type: 'string' },
          fix: { type: 'string' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
      },
    },
  },
}

const CTX = [
  'R-6d3b-2a (io-prod ONLY, uncommitted in the working tree atop a123a09) landed the BOOT-REPLAY MACHINERY that a',
  'shard/gateway will call at boot to re-drive durable outbox rows left by a crashed prior process — the RESTART',
  'half of D-6 #1. It is NOT yet wired into any bin (R-6d3b-2b does that), so it is inert in prod. THIS review checks',
  'the LANDED code. The design of record is scripts/r6d3b_vetted_design.md — READ its section 2 AND the "R-6d3b-2',
  'CORRECTIONS" block at the TAIL (the deadlock + fence holes found in the vetted §2 and the corrected design).',
  '',
  'WHAT LANDED (review the ACTUAL CODE, cite file:line):',
  '- crates/io-prod/src/outbox.rs: `pub type SharedOutbox = Arc<Mutex<Box<dyn OutboxSink + Send>>>` (moved here from',
  '  mesh.rs); `fn durability(&self) -> DurabilityHandle` on the OutboxSink trait (NodeOutbox returns its clone,',
  '  MockOutboxSink returns already_durable()); `enum ReplayError {Undecodable, Unroutable{peer}, LaneStuck{peer},',
  '  FenceTimeout{submitted,expected}}`; consts REPLAY_SEND_MAX_RETRIES / REPLAY_POLL_BACKOFF / REPLAY_FENCE_DEADLINE;',
  '  `fn decode_value_payload(framed) -> Option<Bytes>` (decode_frame::<ReliableFrame>().bytes — the PAYLOAD, not the',
  '  frame); `fn send_durable_with_retry(transport, peer, class, payload)` (QueueFull ⇒ bounded retry then LaneStuck,',
  '  reuses the returned payload, NEVER swallows); and the pub `fn replay_outbox(shared, transport, peers,',
  '  new_incarnation) -> Result<usize, ReplayError>`.',
  '- replay_outbox is LOCK-SCOPED: (1) ONE brief `shared.lock()` to snapshot rows + a DurabilityHandle clone + base',
  '  = last_submitted, then RELEASE; (2) NO LOCK: per row, peers.contains_key(&key.peer) else Unroutable, then',
  '  decode_value_payload else Undecodable, then send_durable_with_retry; (3) NO LOCK: the COUNT-anchored fence —',
  '  while durability.last_submitted() < base+N (bounded by REPLAY_FENCE_DEADLINE ⇒ FenceTimeout) sleep; then',
  '  durability.wait_durable_through(base+N) [the SYNC wait, on the bin main thread]; (4) ONE brief `shared.lock()`:',
  '  gc_below(new_incarnation) + commit (atomic). Runs on the bin MAIN thread pre-build_app; the peer_writer tasks',
  '  (already spawned) re-mirror the replayed Retained sends into the SAME shared store at the FRESH incarnation.',
  '- Tests: outbox.rs decode_value_payload (recover + reject-garbage) + send_durable_with_retry (QueueFull x3 then',
  '  succeed, via a FlakyTransport mock); mesh.rs replay_outbox_redrives_retained_rows_delivers_and_gcs_the_prior_',
  '  incarnation (real QUIC: pre-seed 2 rows at incarnation 1 → spawn A@incarnation2 wired to the shared outbox + B →',
  '  replay_outbox → assert B receives both payloads + scan_all has NO incarnation<2 rows).',
  '',
  'THE TWO HOLES THIS SLICE FIXES (verify the fix is real in the code): (1) DEADLOCK — the vetted §2 held the shared',
  'Mutex across the send+fence, but a replayed send_durable(Retained) re-enters a peer_writer block A that re-.lock()s',
  'the SAME mutex to re-mirror ⇒ deadlock. The fix: replay_outbox holds the lock ONLY for the brief scan-snapshot +',
  'the brief gc, NEVER across the send/fence. (2) PREMATURE-GC DATA LOSS — a high-water fence could pass before all N',
  'fresh rows are submitted (last_submitted bumps async as peer_writers reach block A), then gc sweeps the old rows',
  'whose fresh copy never landed. The fix: COUNT-anchored fence (wait last_submitted >= base+N, N exact because a',
  'previously-retained row re-frames identically ⇒ none shed).',
  '',
  'GATE (green): vd-io-prod 108 lib + all integration; clippy -D clean; Tier-B --fail-under-regions 90 PASS (TOTAL',
  '94.29%, outbox.rs 97.71%, mesh.rs 94.71%, store.rs 90.43%); workspace build 0.',
  '',
  'Find what the green gate does NOT catch: any RESIDUAL deadlock/lock-scoping error, a fence off-by-one or',
  'premature-gc path, a swallowed error, an incarnation/gc mismatch, a re-mirror ordering hazard, dead code, a seam',
  'breach, or a coverage/scope dishonesty. Code-grounded (cite file:line). Empty/NIT array if your lens is clean —',
  'do NOT invent problems. Rank by real durable-transport / boot-safety impact.',
].join('\n')

phase('Review')
const LENSES = [
  { key: 'deadlock-fence-fidelity', focus:
    'DEADLOCK-FREEDOM + FENCE CORRECTNESS in the LANDED replay_outbox. Prove against the code: (a) the shared Mutex ' +
    'is held ONLY across the brief scan-snapshot (step 1) and the brief gc (step 4), NEVER across the sends (step 2) ' +
    'or the fence (step 3) — so a peer_writer re-mirroring a replayed Retained send freely acquires the lock (no ' +
    'deadlock); confirm the guard scopes actually drop (the `{ ... }` blocks) before the lock-free sections. (b) the ' +
    'COUNT fence is correct: base captured under the scan lock, target=base+N, wait last_submitted>=target THEN ' +
    'wait_durable_through(target) THEN gc — is N EXACT (each replayed send = exactly one submit; can a replayed row ' +
    'ever shed/coalesce so last_submitted never reaches base+N ⇒ a false FenceTimeout, or reach it with a row NOT ' +
    'durable ⇒ premature gc)? (c) is there ANY interleaving where gc_below runs before all N fresh rows are durable ' +
    '(the data-loss hole)? (d) the FenceTimeout deadline + the wait_durable_through writer-death backstop bound ' +
    'liveness (a stuck/dead writer fails loud, never hangs boot). (e) the base capture: base=last_submitted at scan ' +
    'time — could a concurrent submit (there are none pre-build_app, but verify) skew base? Cite file:line.' },
  { key: 'findings-fidelity', focus:
    'FINDINGS A/B/C/D FIDELITY + re-mirror correctness. Verify: (A) decode_value_payload returns the PAYLOAD ' +
    '(rf.bytes) not the framed frame — so the send path re-frames at the fresh incarnation/seq (re-sending the stored ' +
    'frame would double-frame + carry stale incarnation); and OutboxKey.peer/.class + ReliableFrame are read INSIDE ' +
    'io-prod (the bin never names them). (B) send_durable_with_retry NEVER swallows QueueFull — a route-miss is ' +
    'caught by the pre-send peers.contains_key (Unroutable, loud), a full lane retries bounded then LaneStuck (loud); ' +
    'no `let _ =` drop; the `?` in replay_outbox bails BEFORE the fence/gc so an un-routable rows data is never swept. ' +
    '(C) new_incarnation is a PARAM (replay_outbox never calls resolve_process_incarnation — that double-increment is ' +
    'the bins job in 2b). (D) the fence is count-anchored not a scan_all count. Re-mirror ordering: ascending scan ⇒ ' +
    'fresh lane seq 0,1,2.. ⇒ classify_reliable Reset-then-Accept (no gap_drop) — confirm the ascending-scan claim ' +
    '(scan_all returns ascending by (peer,class,incarnation,seq)). Any hole? Cite file:line.' },
  { key: 'hr-seam-scope-test', focus:
    'HR-COMPLIANCE / SEAM / SCOPE / TEST RIGOR / COVERAGE. Verify: (a) the frozen sim::io seam + wire contract ' +
    'UNTOUCHED — SharedOutbox + the new `durability()` are on io-prod-internal types (OutboxSink), and moving ' +
    'SharedOutbox from mesh.rs to outbox.rs is pure-internal (mesh.rs `use crate::outbox::SharedOutbox`); no dead ' +
    'code. (b) no magic numbers — REPLAY_SEND_MAX_RETRIES / REPLAY_POLL_BACKOFF / REPLAY_FENCE_DEADLINE each have ONE ' +
    'named home with a real justification (are the VALUES sane — e.g. 10_000 retries * 1ms = 10s before LaneStuck; ' +
    '30s fence deadline for boot). (c) TEST RIGOR: does the end-to-end replay test actually PIN no-deadlock (it ' +
    'completes ⇒ no deadlock — real) AND no-premature-gc (it asserts scan_all has no incarnation<2 rows — but would ' +
    'it catch a premature gc? mutation-check: if the fence were removed, would the test still pass by luck?), and the ' +
    're-drive/delivery? Is decode/retry covered deterministically? (d) COVERAGE HONESTY: FenceTimeout + LaneStuck ' +
    'arms are uncovered fail-loud tripwires (like the sync park panic) — riding the Tier-B floor, acceptable? (e) ' +
    'SCOPE: 2a is inert (no bin wired), D-6 #1 RESTART case is not closed until 2b, NEVER-restart is R-6d3c — stated? ' +
    'Cite file:line.' },
]
const reviews = await parallel(
  LENSES.map((l) => () =>
    agent(
      CTX + '\n\nYou are REVIEWER (' + l.key + '). Adversarially review through YOUR lens: ' + l.focus,
      { label: 'review:' + l.key, phase: 'Review', schema: FINDINGS_SCHEMA, model: 'opus' }
    )
  )
)
const valid = reviews.filter(Boolean)
const allFindings = valid.flatMap((r) => r.findings.map((f) => ({ ...f, lens: r.lens })))

phase('Synthesize')
const synth = await agent(
  CTX + '\n\nThree adversarial lenses reviewed the LANDED R-6d3b-2a. Raw findings (JSON):\n' +
  JSON.stringify({ verdicts: valid.map((r) => ({ lens: r.lens, verdict: r.verdict, summary: r.summary })), findings: allFindings }, null, 2) +
  '\n\nSynthesize. Deduplicate (same root cause = ONE, note corroboration). CONFIRM or REFUTE each against the code, ' +
  're-rank by true boot-safety / durable-transport impact (down-grade a reviewer CRITICAL the code already handles ' +
  'with the file:line that handles it; up-grade any real residual deadlock / premature-gc / silent-loss hazard). For ' +
  'each surviving finding: MUST-FIX-BEFORE-COMMIT / FIX-IN-R-6d3b-2b+ / OPTIONAL. Then a one-line VERDICT: ' +
  'COMMIT_CLEAN | FIX_BEFORE_COMMIT (list blockers) | RECONSIDER. Decisive + code-grounded; reward the ' +
  'deadlock-free + count-fence fidelity where sound.',
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return { verdict: synth, findingCount: allFindings.length, verdicts: valid.map((r) => ({ lens: r.lens, verdict: r.verdict })) }
