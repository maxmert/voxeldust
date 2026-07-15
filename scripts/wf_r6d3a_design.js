export const meta = {
  name: 'r6d3a-design',
  description: 'Design of record for R-6d3a: transport durable-before-send GATE + real outbox sink injection',
  phases: [
    { title: 'Design', detail: 'implement-ready design for the gate + sink injection' },
    { title: 'Review', detail: '3 opus lenses: concurrency/deadlock / crash-window / HR-seam-scope' },
    { title: 'Synthesize', detail: 'adjudicate then produce design of record' },
  ],
}

const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['design_markdown', 'open_questions', 'crash_window_table'],
  properties: {
    design_markdown: { type: 'string', description: 'the full implement-ready design of record for R-6d3a' },
    crash_window_table: { type: 'string', description: 'the durable-before-send crash-window table' },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
}
const REVIEW_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['lens', 'verdict', 'summary', 'findings'],
  properties: {
    lens: { type: 'string' },
    verdict: { type: 'string', enum: ['SOUND_TO_IMPLEMENT', 'REVISE', 'RECONSIDER'] },
    summary: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['severity', 'title', 'claim', 'evidence', 'fix', 'confidence'],
        properties: {
          severity: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NIT'] },
          title: { type: 'string' },
          claim: { type: 'string' },
          evidence: { type: 'string', description: 'code-grounded, cite file:line' },
          fix: { type: 'string' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
      },
    },
  },
}

const GROUND = [
  'R-6d3a is a sub-slice of the R-6d durable-outbox arc (closes D-6 #1: a source shard crash losing a producer-less',
  'reliable one-shot: TransientBatch AwaitAdopt + band-exit Ghost::Despawn). The arc design of record is',
  'scripts/r6d_vetted_design.md (READ its section (c) R-6d3, "the durability gate (CRITICAL-1) + HIGH-1"). R-6d2c',
  'JUST LANDED (commit ebbe544): the reliable-lane FSM ALREADY has the write-through/delete-through. The method',
  'assign_and_retain(from, class, bytes, durable, sink: Option of &mut dyn OutboxSink) mirrors a Retained frame to',
  'the sink; on_ack(.., sink) releases it. In PRODUCTION today both call sites pass literal None (mesh.rs write_frame',
  'near 1610 for assign, near 1190 in peer_writer for on_ack) so nothing is written. R-6d3a WIRES THE REAL SINK plus',
  'adds the durable-before-send GATE.',
  '',
  'SCOPE OF R-6d3a (design THIS, nothing more):',
  '1. Inject a real, SHARED outbox sink into spawn_mesh then MeshConfig/PeerWriter then write_frame + peer_writer,',
  '   flipping the two literal Nones to the real handle. The sink is ONE NodeOutbox per node shared by ALL per-peer',
  '   writer tasks (OutboxKey carries the peer field, so rows are disjoint per peer). LOW-4 (from the R-6d2c review):',
  '   the SAME sink MUST reach BOTH assign_and_retain AND on_ack for a lane, or a durable row leaks.',
  '2. The durable-before-send GATE (arc CRITICAL-1): in write_frame Reliable arm, when frame.durability==Retained,',
  '   the outbox row must be DURABLE (fsync-ed) BEFORE write_reliable_frame puts the frame on the wire. Ephemeral',
  '   frames keep the EXACT current fast path (no commit, no fsync, byte-identical). The gate proves that a batch',
  '   frame that can ever be observed as sent survives the source crash.',
  'OUT OF SCOPE (name them as the next sub-slices, do NOT design them): boot replay scan_all then re-drive then',
  'gc_below (R-6d3b); the saga AwaitAdopt split + dest TransientDiscard (R-6d3c); the composed SIGKILL e2e + proptest',
  '(R-6d4). Whether shard.rs/gateway.rs main actually OPENS+passes a real NodeOutbox in 3a (vs passing None and',
  'deferring the real open to 3b) is a scope decision you must make and justify; bias toward keeping 3a io-prod-',
  'focused and testable with a real handle injected directly, so 3a is inert-safe in prod like 2c was, if cleaner.',
  '',
  'THE HARD PART: CONCURRENCY (design it correctly, do not hand-wave):',
  '- NodeOutbox (crates/io-prod/src/outbox.rs): impl OutboxSink. retain = store.put (in-memory stage, fast). release',
  '  = store.delete (stage). commit() = store.commit() [submit the staged batch, block-on-prior] THEN',
  '  durability.wait_durable_through(store.last_submitted()) [BLOCKS until the off-tick RedbStore writer thread',
  '  fsyncs]. The fsync happens on RedbStore OWN writer thread (store.rs near 292), INDEPENDENT of the tokio runtime',
  '  (no deadlock possible between a parked tokio worker and the fsync thread).',
  '- DurabilityHandle (store.rs near 124): Clone-able (holds Arc of AtomicU64 + a condvar). is_durable_through(seq)',
  '  returns bool (non-blocking Acquire load), durable_through() returns u64, wait_durable_through(seq) parks,',
  '  last_submitted() returns u64, writer_alive() returns bool. It does NOT touch the store: a writer can wait on it',
  '  WITHOUT holding the NodeOutbox lock.',
  '- THE HAZARD: if a peer_writer holds the shared Mutex of NodeOutbox across the fsync WAIT (commit blocks while',
  '  &mut self is borrowed), it serializes every OTHER peer_writer: they cannot even stage their retains. The gate',
  '  must stage+submit under the lock, capture the target durability seq, RELEASE the lock, THEN wait on the cloned',
  '  DurabilityHandle outside the lock, THEN send. This likely needs the OutboxSink/NodeOutbox API to expose submit',
  '  plus a handle separately from the blocking commit(), OR the writer to hold its own DurabilityHandle clone.',
  '  Design the exact API shape.',
  '- write_frame CANCEL-SAFETY (mesh.rs near 1570 comment): NEVER wrap write_frame in timeout/select. The added',
  '  durable wait must respect this: a cancelled durable-wait must not leave a half-durable/half-sent state. The',
  '  only cancel point is w.rx.recv() in the outer select.',
  '- Redial/retry interaction: replay_batch re-SENDS a retained frame on redial (mesh.rs near 679). A Retained frame',
  '  re-sent on redial is ALREADY durable (retained once at assign): must NOT re-commit/re-fsync per resend. Confirm',
  '  the gate fires only on the FIRST send (the assign path), not on replay_batch re-sends.',
  '',
  'FILES TO READ: crates/io-prod/src/mesh.rs (spawn_mesh near 775, MeshConfig near 58, PeerWriter near 1142,',
  'peer_writer near 1160+, write_frame near 1574 [Reliable arm near 1590-1660, the assign_and_retain call near 1610,',
  'write_reliable_frame], on_ack call near 1190, the FSM assign_and_retain near 607 + on_ack near 566);',
  'crates/io-prod/src/outbox.rs (NodeOutbox + OutboxSink); crates/io-prod/src/store.rs (RedbStore, DurabilityHandle',
  'near 124, the off-tick writer near 292); crates/bins/src/lib.rs (open_node_outbox near 270);',
  'scripts/r6d_vetted_design.md section (c). GATE for io-prod is Tier-B --fail-under-regions 90 (justfile near 40, NO',
  '--branch); the new gate branch must be covered.',
].join('\n')

phase('Design')
const design = await agent(
  GROUND + '\n\n' +
  'You are the DESIGNER. Produce an IMPLEMENT-READY design of record for R-6d3a ONLY. Ground EVERY claim in the ' +
  'actual code (cite file:line). Deliver: (1) the exact API shape for injecting the shared sink + the split ' +
  'submit/wait so the fsync never happens under the shared lock (concrete signatures: MeshConfig field? spawn_mesh ' +
  'param? PeerWriter field? what does write_frame call, in what order?); (2) the durable-before-send gate placement ' +
  'in write_frame Reliable arm with the EXACT ordering (assign_and_retain retains, submit, release lock, wait ' +
  'durable, send) and the Ephemeral fast-path untouched; (3) the no-deadlock + no-datagram-head-of-line + cancel-' +
  'safety arguments; (4) the redial-no-re-commit argument; (5) the crash-window table (crash at each point then ' +
  'outcome, assuming R-6d3b replay exists); (6) the exact tests (unit + io-prod integration with a real NodeOutbox ' +
  'proving durable-before-send + Ephemeral-unchanged + the concurrency non-serialization property) and how they ' +
  'keep Tier-B >=90; (7) the precise files/functions touched. Flag any place the current NodeOutbox/DurabilityHandle ' +
  'API is insufficient and specify the minimal additive change. Be decisive.',
  { label: 'design', phase: 'Design', schema: DESIGN_SCHEMA, model: 'opus' }
)

phase('Review')
const LENSES = [
  { key: 'concurrency-deadlock', focus:
    'CONCURRENCY, DEADLOCK, LATENCY, CANCEL-SAFETY. Is the shared-sink model correct and deadlock-free? Prove: (a) ' +
    'the fsync wait is NEVER held under the shared Mutex of NodeOutbox (else cross-peer serialization); (b) no ' +
    'deadlock between a parked tokio peer_writer worker and the RedbStore off-tick fsync thread; (c) datagrams to ' +
    'the SAME peer are not unboundedly head-of-lined by a durable send fsync (HIGH-1): quantify what IS delayed and ' +
    'argue it is bounded/acceptable; (d) write_frame cancel-safety preserved (no timeout/select wrap; a cancelled ' +
    'durable-wait leaves no half-durable/half-sent state); (e) lock contention across N peers staging retains is ' +
    'bounded (the lock is held only for the fast put/submit, not the fsync). Check the proposed API shape actually ' +
    'achieves this against the real DurabilityHandle/NodeOutbox/store.rs code.' },
  { key: 'crash-window', focus:
    'CRASH-WINDOW + DURABILITY CORRECTNESS. Re-derive the durable-before-send crash-window table against the design ' +
    'exact ordering. Prove: (a) a crash at EVERY point (after retain-stage / after submit / after fsync-durable / ' +
    'after write_reliable_frame / mid-write) leaves the system recoverable once R-6d3b replay exists: no frame both ' +
    'observed-as-sent AND lost; (b) the gate genuinely guarantees durable-BEFORE-first-send for Retained frames; ' +
    '(c) the redial/replay_batch re-send does NOT re-commit and an already-durable Retained frame is not double-' +
    'fsync-ed nor lost; (d) the OutboxKey (peer,class,incarnation,seq) is stable so a crash-then-replay at a bumped ' +
    'incarnation does not collide/orphan (incarnation bump is 3b, but the KEY design must already be crash-correct); ' +
    '(e) the interaction with on_ack delete-through under crash (a frame acked-then-crash vs crash-then-acked). Cite ' +
    'file:line.' },
  { key: 'hr-seam-scope', focus:
    'HR-COMPLIANCE, SEAM INTEGRITY, SCOPE DISCIPLINE, TEST/COVERAGE. Check: (a) the frozen sim::io seam + wire ' +
    'contract untouched (OutboxSink/NodeOutbox io-prod-internal); (b) Ephemeral reliable frames + the 20Hz ' +
    'unreliable datagram path provably byte-identical to pre-3a (the gate fires ONLY for Retained); (c) HR5: the new ' +
    'gate branch + the durable-wait are coverable by Tier-B tests (no uncoverable region; the fsync path exercised ' +
    'with a real NodeOutbox); no magic numbers (any cadence/threshold a named const/tuning); HR3 one-tooling (no ' +
    'per-shard/per-class fork); (d) scope discipline: is 3a genuinely minimal-and-testable, with boot-replay (3b) ' +
    'and saga/dest (3c) cleanly deferred and 3a inert-safe or live-correct as claimed?; (e) does the design over-' +
    'reach (designing 3b/3c) or leave a gap that makes 3a untestable/dead? Cite file:line.' },
]
const reviews = await parallel(
  LENSES.map((l) => () =>
    agent(
      GROUND + '\n\n' +
      'Here is the DESIGNER proposed design of record for R-6d3a:\n--- DESIGN ---\n' + design.design_markdown +
      '\n--- CRASH-WINDOW TABLE ---\n' + design.crash_window_table +
      '\n--- OPEN QUESTIONS ---\n' + design.open_questions.join('\n') + '\n--- END ---\n\n' +
      'You are REVIEWER (' + l.key + '). Adversarially review the design through YOUR lens: ' + l.focus + '\n' +
      'Be code-grounded (cite file:line) and decisive. If sound on your lens, say SOUND_TO_IMPLEMENT with an empty ' +
      'or NIT-only findings array; do NOT invent problems. If it has a real defect, rank it and give a concrete fix.',
      { label: 'review:' + l.key, phase: 'Review', schema: REVIEW_SCHEMA, model: 'opus' }
    )
  )
)
const valid = reviews.filter(Boolean)

phase('Synthesize')
const synth = await agent(
  GROUND + '\n\n' +
  'The DESIGNER proposed a design of record for R-6d3a and three adversarial lenses reviewed it.\n\nDESIGN:\n' +
  design.design_markdown + '\n\nCRASH-WINDOW TABLE:\n' + design.crash_window_table + '\n\nREVIEWS (JSON):\n' +
  JSON.stringify(valid.map((r) => ({ lens: r.lens, verdict: r.verdict, summary: r.summary, findings: r.findings })), null, 2) +
  '\n\nSynthesize the ADJUDICATED, IMPLEMENT-READY design of record. Deduplicate findings (same root cause across ' +
  'lenses = one, note corroboration). For each: CONFIRM or REFUTE against the code, rank by true impact, and FOLD ' +
  'the fix into the final design (do not just list it). Down-grade any reviewer CRITICAL that the design/code ' +
  'already handles (cite where); up-grade any real deadlock/serialization/silent-loss hazard. Output the COMPLETE ' +
  'corrected design of record (API signatures, exact write_frame ordering, the concurrency argument, the crash-' +
  'window table, the tests, the files touched) so it can be implemented directly, plus a one-line VERDICT: ' +
  'SOUND_TO_IMPLEMENT (with the folded must-fixes) | REVISE (blocking issues remain) | RECONSIDER. Be decisive and ' +
  'reward what is genuinely sound.',
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return {
  verdict: synth,
  design_markdown: design.design_markdown,
  crash_window_table: design.crash_window_table,
  reviews: valid.map((r) => ({ lens: r.lens, verdict: r.verdict, summary: r.summary, findings: r.findings })),
}
