export const meta = {
  name: 'r6d3a-postimpl-review',
  description: 'Adversarial post-impl review of R-6d3a (durable-before-send gate + shared-outbox injection + MF-3)',
  phases: [
    { title: 'Review', detail: '3 opus lenses: concurrency / crash-correctness / test-HR-scope' },
    { title: 'Synthesize', detail: 'dedup + severity-rank + verdict' },
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
  'R-6d3a wired a durable-before-send GATE + a real shared durable outbox into the io-prod mesh transport',
  '(closes the RESTART half of D-6 #1 once R-6d3b boot-replay lands; a source-crash losing a producer-less',
  'reliable one-shot). It is committed atop R-6d2c (ebbe544). The vetted design of record is',
  'scripts/r6d3a_vetted_design.md (READ it, esp. the MF-3 CORRECTION at the TOP — the implementer-found,',
  'adversarially-verified concurrency fix that DEVIATES from the vetted synth).',
  '',
  'WHAT LANDED (review the ACTUAL CODE, cite file:line):',
  '- crates/io-prod/src/store.rs: RedbStore::submit_nonblocking (block-on-prior-FREE; take staged, seq, try_send',
  '  to the writer channel, on Full/Disconnected PANIC + retain fail-safe) + DurabilityHandle::already_durable()',
  '  (test-ctor, cfg any(test, store-test-hooks)). commit() UNCHANGED.',
  '- crates/io-prod/src/outbox.rs: OUTBOX_WRITER_CHANNEL_DEPTH (pub const 256); NodeOutbox::open FORCES',
  '  tuning.writer_channel_depth = that; OutboxSink::submit_barrier(&mut self) -> Option<(u64, DurabilityHandle)>',
  '  (submit_nonblocking + clone the handle); a submit_barrier unit test.',
  '- crates/io-prod/src/mesh.rs: type SharedOutbox = Arc<Mutex<Box<dyn OutboxSink + Send>>>; spawn_mesh gains',
  '  outbox: Option<SharedOutbox> cloned into each PeerWriter; PeerWriter.outbox; write_frame gains',
  '  outbox: Option<&SharedOutbox> and its Reliable arm is now BLOCK A (lock if durable, assign_and_retain retain,',
  '  submit_barrier, capture (batch_seq, handle), DROP guard) -> BLOCK B (outside lock: wait_durable_through(batch_seq))',
  '  -> BLOCK C (verbatim QUIC send). Ephemeral/no-sink: no lock, gate None, block B skipped. peer_writer locks the',
  '  shared sink ONCE per ack fan-out and threads it into on_ack (release-through). MockOutboxSink gains submit_barrier.',
  '- Tests: mesh_under_loss.rs T-DBS-1 (real NodeOutbox + real QUIC: a Retained send is durable on disk before',
  '  delivery, released on ack through the shared sink) + T-DBS-2 (Ephemeral fast path leaves the outbox empty).',
  '- 5 bin/test spawn_mesh callers pass None (bins inert-safe until R-6d3b).',
  '',
  'THE MF-3 DEVIATION (verify it in the code): the vetted synth used a depth-2 channel + try_send-panic-on-Full and',
  'claimed "depth-2 exactly enough". The implementer found (and an opus verifier CONFIRMED) that N distinct',
  'peer_writer tasks sharing ONE Arc<Mutex<NodeOutbox>> can pile >2 batches into a depth-2 channel before one fsync',
  'drains -> panic under realistic multi-peer durable load. CURE (landed): the OUTBOX store gets its own generous',
  'OUTBOX_WRITER_CHANNEL_DEPTH (256 >> peer count) so Full is unreachable-by-capacity; submit_nonblocking is block-on-',
  'prior-free with seq-assign+try_send UNDER the lock (ordering); block B waits a SINGLE bar.seq (durability monotone).',
  'Rationale: the outbox needs NO depth-1 crash-loss bound because an un-fsynced in-channel batch is an un-SENT frame',
  '(block B withholds the wire until durable), re-emitted by the restarted source.',
  '',
  'GATE (already green): vd-io-prod 102 lib + all integration pass; clippy io-prod + vd-bins -D clean; Tier-B',
  '--fail-under-regions 90 PASS (TOTAL 94.05%; store.rs 88.86% is the new submit_nonblocking Full/Disconnected PANIC',
  'tripwires — impossible-invariant unreachable arms; floor is on TOTAL). workspace build green.',
  '',
  'Find what the GREEN GATE does NOT catch. Be adversarial + code-grounded (cite file:line). If your lens is clean,',
  'say so with an empty/NIT findings array — do NOT invent problems. Rank by real durable-transport/scale impact.',
].join('\n')

phase('Review')
const LENSES = [
  { key: 'concurrency', focus:
    'CONCURRENCY / DEADLOCK / SCALE / CANCEL-SAFETY. Verify IN THE ACTUAL CODE (mesh.rs write_frame block A/B/C, ' +
    'peer_writer ack fan-out; store.rs submit_nonblocking; outbox.rs): (a) the shared Mutex is NEVER held across the ' +
    'fsync wait (guard dropped before block B) — no cross-peer serialization; (b) MF-3: is OUTBOX_WRITER_CHANNEL_DEPTH ' +
    '(256) genuinely making try_send Full unreachable under N concurrent peer-writers, and is the seq-assign+try_send ' +
    'kept UNDER the lock so the writer drains in seq order (no out-of-order last_durable.store(max_seq))? Is there any ' +
    'REMAINING path where a peer blocks another under the lock, or where >256 could ever pile (any retry/replay double-' +
    'submit)? (c) no deadlock between a parked tokio worker on wait_durable_through and the off-tick fsync thread; ' +
    '(d) write_frame cancel-safety preserved (no timeout/select wrap; block B is a sync condvar park, not an .await; a ' +
    'cancel leaves no half-durable/half-sent state); (e) the on_ack per-iteration reborrow (as_mut().map(|s| &mut **s)) ' +
    'is correct and the ONE lock per ack fan-out is right; (f) any await-holding-lock (the MutexGuard must not cross an ' +
    '.await). Cite file:line.' },
  { key: 'crash-correctness', focus:
    'CRASH-WINDOW / DURABILITY CORRECTNESS / SEQ SEMANTICS. Verify: (a) durable-before-send genuinely holds — block B ' +
    'wait_durable_through(batch_seq) runs before block C write_reliable_frame for every Retained frame; (b) the CRITICAL ' +
    'seq distinction: block B waits on the STORE batch seq from submit_barrier (NOT the lane/wire seq from ' +
    'assign_and_retain) — confirm the code waits on the right one and that a wrong-seq wait would be a silent gate-bypass; ' +
    '(c) MF-3 crash-safety: is every un-fsynced in-channel batch truly un-SENT (block C unreached) so the "restarted ' +
    'source re-emits, NO LOSS" holds — any window where a frame is sent but its row is not yet durable? (d) redial/replay ' +
    '(replay_batch, replay_lanes) does NOT re-submit/re-commit/re-fsync a Retained frame (gate fires once, at assign); ' +
    '(e) release-through is staged-only in on_ack (no fsync in the ack path) — is that safe (a crash before the tombstone ' +
    'is durable only re-delivers an already-acked frame, receiver dedups)? (f) OutboxKey (peer,class,incarnation,seq) ' +
    'stability across the gate. Cite file:line.' },
  { key: 'test-hr-scope', focus:
    'TEST RIGOR / HR-COMPLIANCE / COVERAGE HONESTY / SCOPE. Verify: (a) do T-DBS-1/T-DBS-2 + the outbox submit_barrier ' +
    'unit test actually PIN the gate, or would a regression pass them? Is T-DBS-1 mutation-resistant (would neutering ' +
    'block B or the submit leave it green)? Does T-DBS-2 prove the Ephemeral path writes NO row (fast-path byte-identity)? ' +
    'Is the durable-BEFORE-send ORDERING actually asserted, or only durable-AND-delivered (note honestly — the strong ' +
    'SIGKILL-window ordering is R-6d4)? (b) HR5: are the new gate regions covered (block A lock, submit_barrier, block B ' +
    'wait, Ephemeral skip, on_ack shared-sink)? Is the store.rs 88.86% drop ONLY the unreachable panic tripwires (verify ' +
    'the uncovered regions are the Full/Disconnected arms, not real logic)? (c) HR no-magic-numbers: is 256 a named const ' +
    'with a real justification (not arbitrary)? Is depth-256 memory bounded (O(peers) x batch, capped by retry buffers)? ' +
    '(d) frozen sim::io seam + wire contract untouched? Ephemeral reliable + 20Hz datagram provably unchanged? (e) scope: ' +
    'is 3a inert-safe in prod (bins None) with D-6 #1 HONESTLY still-open, and 3b/3c/3d cleanly deferred, no over-reach/gap? ' +
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
  CTX + '\n\nThree adversarial lenses reviewed R-6d3a. Raw findings (JSON):\n' +
  JSON.stringify({ verdicts: valid.map((r) => ({ lens: r.lens, verdict: r.verdict, summary: r.summary })), findings: allFindings }, null, 2) +
  '\n\nSynthesize. Deduplicate (same root cause across lenses = ONE, note corroboration). CONFIRM or REFUTE each ' +
  'against the code, re-rank by TRUE durable-transport/scale impact (down-grade a reviewer CRITICAL the code already ' +
  'handles, with the file:line that handles it; up-grade any real deadlock/serialization/silent-loss/panic-under-scale ' +
  'hazard). For each surviving finding give a disposition: MUST-FIX-BEFORE-COMMIT / FIX-IN-R-6d3b+ / OPTIONAL. Then a ' +
  'one-line VERDICT: COMMIT_CLEAN | FIX_BEFORE_COMMIT (list blockers) | RECONSIDER. Be decisive and code-grounded; ' +
  'reward what is genuinely sound (esp. whether the MF-3 fix is correct in the landed code).',
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return { verdict: synth, findingCount: allFindings.length, verdicts: valid.map((r) => ({ lens: r.lens, verdict: r.verdict })) }
