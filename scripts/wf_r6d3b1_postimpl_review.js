export const meta = {
  name: 'r6d3b1-postimpl-review',
  description: 'Adversarial post-impl review of R-6d3b-1 (F2 async block-B durable wait) — implementation fidelity',
  phases: [
    { title: 'Review', detail: '3 opus lenses: concurrency-fidelity / no-regression-seam / test-coverage' },
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
  'R-6d3b-1 (io-prod ONLY) makes the durable-before-send block-B wait ASYNC so a stalled producer-less durable send',
  'yields its tokio worker instead of parking it — the F2 hard gate from the R-6d3a review (on the worker_threads(2)',
  'mesh runtime, two peer-writers parked in a SYNC durability park occupy BOTH workers and starve the recv/ack/accept',
  'I/O path). Committed atop R-6d3a (4ae5907). The design of record is scripts/r6d3b_vetted_design.md section 1 (F2),',
  'whose async-wait lens already ruled the MECHANISM SOUND (lost-wakeup-free, cancel-safe, orchestrator untouched).',
  'THIS review checks IMPLEMENTATION FIDELITY of the LANDED code — where drift from a sound design hides.',
  '',
  'WHAT LANDED (review the ACTUAL CODE at HEAD, cite file:line):',
  '- crates/io-prod/src/store.rs: DurabilityHandle gains durable_notify: Arc<tokio::sync::Notify> (a field); a NEW',
  '  async fn wait_durable_through_async(&self, seq) — fast-path (last_durable >= seq ⇒ return), else fetch_add',
  '  backpressure then loop { let notified = durable_notify.notified(); if last_durable >= seq return; match',
  '  timeout(WRITER_WAIT_POLL, notified).await { Ok => {} Err(elapsed) => if !durable && !writer_alive => panic } }.',
  '  The SYNC wait_durable_through is UNCHANGED. WriterExitSignal gains durable_notify + fires notify_waiters() in',
  '  Drop (finding-E death-wake) alongside cv.notify_all(). run_writer gains a durable_notify param + calls',
  '  durable_notify.notify_waiters() AFTER last_durable.store(max_seq, Release), alongside cv.notify_all(). open()',
  '  creates one Arc<Notify>, clones into the writer + the handle (NOT stored on RedbStore — RedbStore never reads it,',
  '  so a field there would be dead code). already_durable() gains the field.',
  '- crates/io-prod/src/mesh.rs: block B now `durability.wait_durable_through_async(batch_seq).await;` (was a sync',
  '  park). The guard from block A is DROPPED before block B (unchanged from R-6d3a).',
  '- tests: store.rs async_wait_fast_returns_when_already_durable + async_wait_returns_after_a_real_submit_is_fsynced',
  '  (submit_nonblocking then .await, asserting is_durable_through after). T-DBS-1 (mesh_under_loss real QUIC) now',
  '  drives the async block B end-to-end.',
  '',
  'GATE (green): vd-io-prod 105 lib + all integration; clippy -D warnings incl. #![warn(clippy::await_holding_lock)]',
  '(PASSES with the new block-B .await ⇒ the guard is provably dropped before it); Tier-B --fail-under-regions 90 PASS',
  '(TOTAL 94.09%, store.rs 90.43%); workspace build 0. The orchestrator uses the SYNC wait_durable_through on its MAIN',
  'thread (orchestrator.rs:280) — MUST be unaffected.',
  '',
  'SCOPE (honest): R-6d3b-1 is io-prod only + inert in prod (bins still pass None — R-6d3b-2 wires the live sink +',
  'boot replay). The DETERMINISTIC durable-before-send ordering pin (F1, store-test-hooks) + the two_peers_do_not_',
  'starve_workers proof + the writer-death-wake test are deferred to R-6d4 (store-test-hooks tier). D-6 #1 still open.',
  '',
  'Find what the green gate does NOT catch: implementation drift, a lost-wakeup / missed-notify ordering bug, a',
  'cancel hazard, a regression to the orchestrator sync path, dead code, or a coverage/scope dishonesty. Code-grounded',
  '(cite file:line). If your lens is clean, say so with an empty/NIT array — do NOT invent problems.',
].join('\n')

phase('Review')
const LENSES = [
  { key: 'concurrency-fidelity', focus:
    'CONCURRENCY FIDELITY of the LANDED wait_durable_through_async + writer notify. Prove against the code: (a) the ' +
    'enroll-then-recheck ordering is correct — `let notified = durable_notify.notified()` is created BEFORE the ' +
    'last_durable re-read, so a notify_waiters() racing after enroll but before the .await still completes it (tokio ' +
    'Notify semantics: notified() captures the notify state at creation); (b) the writer notifies AFTER ' +
    'last_durable.store(Release) (store.rs run_writer) so a waiter that observes the notify then re-reads sees >= seq; ' +
    'trace EVERY interleaving of {writer: store then notify} vs {waiter: enroll, read, await} and show none hangs or ' +
    'busy-loops; (c) the timeout(WRITER_WAIT_POLL) backstop + the writer-death panic fire correctly (a dead writer ⇒ ' +
    'bounded fail-loud, never an infinite await); (d) the finding-E death-wake in WriterExitSignal::Drop actually ' +
    'wakes an async waiter (notify_waiters from a std-thread Drop is sound); (e) CANCEL-SAFETY: the new .await in ' +
    'write_frame block B — a task drop there leaves the frame retained + durable-or-becoming-durable but NOT sent ' +
    '(crash row 4), no half-sent state, and does not violate the sole-cancel-point invariant. Any lost-wakeup, ' +
    'busy-spin, or hang? Cite file:line.' },
  { key: 'no-regression-seam', focus:
    'NO-REGRESSION + SEAM + DEAD-CODE. Verify: (a) the SYNC wait_durable_through (store.rs) is BYTE-UNCHANGED and the ' +
    'orchestrator persist-before-effect gate (orchestrator.rs:280, MAIN thread) is unaffected — the async variant is ' +
    'purely additive; (b) the durable_notify is added to DurabilityHandle + WriterExitSignal + the writer + open, but ' +
    'NOT to the RedbStore struct (RedbStore never reads it) — confirm there is no dead field AND no path that needed ' +
    'it on RedbStore; (c) the frozen sim::io seam + wire contract are untouched (DurabilityHandle/Notify are ' +
    'io-prod-internal); (d) the async wait genuinely YIELDS the worker (an .await, not a park) so N concurrent ' +
    'durable sends cost 0 parked workers — the property F2 exists to fix; (e) the Ephemeral / gate==None path is ' +
    'still byte-identical (no async wait when not durable). Any regression or dead code? Cite file:line.' },
  { key: 'test-coverage-scope', focus:
    'TEST RIGOR / COVERAGE HONESTY / SCOPE. Verify: (a) do the two store.rs async tests + T-DBS-1 actually PIN F2 or ' +
    'could a regression pass them? Is the enroll-await path (not just the fast-path) exercised, or is it opportunistic ' +
    '(off-tick writer may race) — and is that honestly acknowledged (the DETERMINISTIC pin is R-6d4)? Would neutering ' +
    'the writer notify_waiters or the enroll leave the tests green (mutation-resistance)? (b) is async_wait_returns_' +
    'after_a_real_submit_is_fsynced free of flake/hang risk (the timeout backstop guarantees termination)? (c) HR5: ' +
    'the new async regions (fast-path, enroll-loop, timeout arm) — covered or honestly Tier-B-floored (the panic arm ' +
    'rides the floor like the sync sibling)? store.rs went 88.86%->90.43% — sanity-check that is real coverage not ' +
    'noise; (d) no magic numbers (WRITER_WAIT_POLL reused, ONE home); (e) SCOPE honesty: F1 + two_peers_do_not_starve ' +
    '+ death-wake test deferred to R-6d4, D-6 #1 still open, bins inert — all stated? Cite file:line.' },
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
  CTX + '\n\nThree adversarial lenses reviewed the LANDED R-6d3b-1. Raw findings (JSON):\n' +
  JSON.stringify({ verdicts: valid.map((r) => ({ lens: r.lens, verdict: r.verdict, summary: r.summary })), findings: allFindings }, null, 2) +
  '\n\nSynthesize. Deduplicate (same root cause = ONE, note corroboration). CONFIRM or REFUTE each against the code, ' +
  're-rank by true durable-transport/concurrency impact (down-grade a reviewer CRITICAL the code already handles with ' +
  'the file:line that handles it; up-grade any real lost-wakeup / hang / cancel / orchestrator-regression hazard). ' +
  'For each surviving finding: MUST-FIX-BEFORE-COMMIT / FIX-IN-R-6d3b-2+ / OPTIONAL. Then a one-line VERDICT: ' +
  'COMMIT_CLEAN | FIX_BEFORE_COMMIT (list blockers) | RECONSIDER. Decisive + code-grounded; reward the F2 fidelity ' +
  'where sound.',
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return { verdict: synth, findingCount: allFindings.length, verdicts: valid.map((r) => ({ lens: r.lens, verdict: r.verdict })) }
