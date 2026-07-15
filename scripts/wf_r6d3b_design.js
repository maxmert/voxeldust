export const meta = {
  name: 'r6d3b-design',
  description: 'Design of record for R-6d3b: flip the outbox gate LIVE (async block-B wait + bin wiring + boot replay)',
  phases: [
    { title: 'Design', detail: 'implement-ready design + sub-slicing; deepest on the F2 async wait' },
    { title: 'Review', detail: '3 opus lenses: async-wait/concurrency / boot-replay-crash / HR-seam-scope' },
    { title: 'Synthesize', detail: 'adjudicate then produce design of record' },
  ],
}

const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['design_markdown', 'crash_window_table', 'sub_slices', 'open_questions'],
  properties: {
    design_markdown: { type: 'string' },
    crash_window_table: { type: 'string', description: 'the boot-replay crash-window table' },
    sub_slices: { type: 'array', items: { type: 'string' }, description: 'proposed R-6d3b sub-slice ordering' },
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
          evidence: { type: 'string' },
          fix: { type: 'string' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
      },
    },
  },
}

const GROUND = [
  'R-6d3b flips the R-6d durable-outbox gate from INERT (bins pass None) to LIVE — the slice that closes the RESTART',
  'half of D-6 #1 (a source shard crash losing a producer-less reliable one-shot: TransientBatch AwaitAdopt +',
  'band-exit Ghost::Despawn). R-6d3a JUST LANDED (commit 4ae5907): the durable-before-send GATE + shared-outbox',
  'injection exist but every bin passes None so nothing is written. Read scripts/r6d3a_vetted_design.md (the MF-3',
  'correction at the top) and scripts/r6d_vetted_design.md section (c) R-6d3 (the boot-replay + saga parts).',
  '',
  'R-6d3b SCOPE (design all of it; PROPOSE the sub-slice ordering):',
  '1. F2 (BINDING hard gate from the R-6d3a post-impl review, wf_cfdde7cc): block B (mesh.rs write_frame) currently',
  '   calls durability.wait_durable_through(batch_seq) — a SYNCHRONOUS std Condvar park (store.rs park_until_durable',
  '   ~281). On the 2-worker tokio runtime (shard.rs:18 worker_threads(2)), TWO peer_writer tasks parked in block B',
  '   occupy BOTH workers for the fsync duration, STARVING the mesh recv/ack/accept I/O path (the accept loop +',
  '   per-peer serve_connection readers are tokio tasks on the same pool). The sim tick loop is on the MAIN thread',
  '   (shard.rs:94 node.step_tick()), NOT starved. FIX the block-B wait so a parked durable send does NOT block a',
  '   tokio worker. Candidate mechanisms (weigh them, pick one, justify): (a) an async tokio::sync::Notify on the',
  '   DurabilityHandle that the RedbStore writer notify_waiters() alongside its existing Condvar notify_all()',
  '   (store.rs ~387), + an async DurabilityHandle::wait_durable_through_async(seq) block B awaits — no thread park,',
  '   scales to many concurrent durable sends; (b) tokio::task::spawn_blocking the existing sync park; (c) raise',
  '   worker_threads (band-aid). The ORCHESTRATOR uses the SYNC wait_durable_through on its MAIN thread',
  '   (orchestrator.rs:280 persist-before-effect gate) — that path MUST stay sync + unchanged; the async variant is',
  '   ADDITIVE. write_frame CANCEL-SAFETY: today the ONLY cancel point is w.rx.recv() (mesh.rs ~1576 comment);',
  '   an async block-B wait ADDS an .await inside write_frame — verify a cancel there leaves no half-durable/half-sent',
  '   state (the frame is retained + the row durable-or-becoming-durable but NOT sent ⇒ replay re-drives; is that safe',
  '   and does it violate the cancel-safety invariant the comment states?).',
  '2. F1 (BINDING): the DETERMINISTIC durable-before-send ordering pin. R-6d3a T-DBS-1 only catches a block-B deletion',
  '   ~1/8 (the off-tick writer races scan_all). Land a store-test-hooks pause_on_key_prefix test proving the row is',
  '   NOT in scan_all until block B has waited (the writer is parked on the sentinel; block B blocks; unpause ⇒ durable',
  '   ⇒ sent). This is feature-gated (store-test-hooks) so it runs in the R-6d4 process tier, not the default gate.',
  '3. BIN WIRING: open the per-node store + pass the real SharedOutbox to spawn_mesh, in shard.rs + gateway.rs main',
  '   (open_node_outbox already EXISTS at bins/lib.rs:270 — VD_OUTBOX_PATH gated, returns Option<NodeOutbox>; wrap as',
  '   Arc<Mutex<Box<dyn OutboxSink + Send>>>). The R-6d3a F3 boot check (spawn_mesh rejects peer count >',
  '   OUTBOX_WRITER_CHANNEL_DEPTH when an outbox is wired) now fires with real peer books — confirm real shard peer',
  '   counts are < 256. HR3: shard strictly needs it (has producer-less flows); gateway opens one for uniformity',
  '   (empty until it has such a flow).',
  '4. BOOT REPLAY: before build_app consumes the transport (build_app<T: Transport>(cfg, transport) MOVES it,',
  '   app.rs:87), drain the outbox: incarnation = resolve_process_incarnation (bins/lib.rs:223, the R-6a durable',
  '   monotone bump); for (key, val) in outbox.scan_all() { transport.send_durable(key.peer, key.class,',
  '   decode(val).bytes, Durability::Retained) } in ascending scan order (the fresh lane assigns seq 0,1,2.. at the',
  '   bumped incarnation ⇒ classify_reliable Reset-then-Accept, no gap). Then, STRICTLY AFTER all fresh rows are',
  '   durable, gc_below(new_incarnation) to sweep the prior incarnation window (HIGH-3: never delete old rows before',
  '   the fresh ones are durable, or a crash mid-replay loses both). The arc design MEDIUM-1 says: drop any bespoke',
  '   inject_replay — replay is N ordinary send_durable calls the bin issues before build_app. Decode: scan_all',
  '   returns already-envelope-stripped framed bytes; decode_frame::<ReliableFrame> to recover .bytes (the payload)',
  '   to re-send (or re-send the framed bytes? decide — the send path re-frames, so re-send the PAYLOAD not the frame).',
  '',
  'OUT OF SCOPE (name as later slices): R-6d3c (the saga AwaitAdopt split + dest TransientDiscard — the NEVER-restart',
  'closure, the slice that actually retires D-6 #1); R-6d4 (the composed SIGKILL-source-in-AwaitAdopt e2e + the',
  'both-ends-restart replay proptest, which is where the F1 store-test-hooks pin runs).',
  '',
  'KEY FILES: crates/io-prod/src/mesh.rs (write_frame block A/B/C ~1654-1745, the durable-before-send gate);',
  'crates/io-prod/src/store.rs (DurabilityHandle ~120-185, park_until_durable ~281, the writer notify_all ~387,',
  'run_writer ~353); crates/io-prod/src/outbox.rs (NodeOutbox, scan_all, gc_below, submit_barrier);',
  'crates/bins/src/lib.rs (open_node_outbox ~270, resolve_process_incarnation ~223); crates/bins/src/bin/shard.rs',
  '(~17 runtime worker_threads(2), ~22 spawn_mesh None, ~94 main-thread step_tick); crates/bins/src/bin/gateway.rs;',
  'crates/node/src/app.rs (build_app ~87 consumes the transport); crates/wire/src/framing.rs (decode_frame).',
  'Coverage: shard/gateway bins are Tier-B process tier; io-prod Tier-B region-floor 90 (justfile ~40, NO --branch).',
  'The frozen sim::io seam + wire contract must stay UNTOUCHED (OutboxSink/NodeOutbox/DurabilityHandle are io-prod-',
  'internal). No magic numbers (any const has ONE home). Be code-grounded; cite file:line.',
].join('\n')

phase('Design')
const design = await agent(
  GROUND + '\n\n' +
  'You are the DESIGNER. Produce an IMPLEMENT-READY design of record for R-6d3b, and PROPOSE the sub-slice ordering ' +
  '(e.g. an io-prod-only F2+F1 slice that makes block B production-safe FIRST, then a bin-wiring+boot-replay slice ' +
  'that flips it live — or justify a different split). Deliver, all code-grounded (cite file:line): (1) the F2 async ' +
  'block-B wait — the EXACT mechanism (signatures: what does the DurabilityHandle gain? does the writer add a ' +
  'tokio::sync::Notify it notifies alongside the Condvar? what does block B await?), a LOST-WAKEUP-FREE argument, the ' +
  'no-regression proof for the orchestrator sync path, the cancel-safety analysis for the new .await in write_frame, ' +
  'and the scale argument (N concurrent durable sends yield N workers vs park N); (2) the F1 deterministic pin (the ' +
  'exact store-test-hooks test); (3) the bin wiring (open_node_outbox -> SharedOutbox -> spawn_mesh in shard+gateway ' +
  'main; the F3 boot-check interaction with real peer counts); (4) the boot replay (the exact sequence before ' +
  'build_app, decode payload-vs-frame, the ascending re-send, the STRICT gc_below-after-all-durable ordering); (5) the ' +
  'boot-replay crash-window table; (6) the tests per slice (io-prod unit/integration for F2 + the deterministic F1 ' +
  'pin + a bins boot-replay test) and how they keep Tier-B >= 90; (7) the precise files/functions touched per ' +
  'sub-slice. Flag any place the current API is insufficient and specify the minimal additive change. Be decisive.',
  { label: 'design', phase: 'Design', schema: DESIGN_SCHEMA, model: 'opus' }
)

phase('Review')
const LENSES = [
  { key: 'async-wait-concurrency', focus:
    'THE F2 ASYNC BLOCK-B WAIT — correctness, lost-wakeups, scale, cancel-safety, no-regression. Verify against the ' +
    'code: (a) the chosen mechanism (Notify or spawn_blocking) is LOST-WAKEUP-FREE — for a Notify, the notified() ' +
    'future must be armed BEFORE the is_durable_through re-check, and the writer must notify AFTER last_durable.store ' +
    '(store.rs ~384-387); trace the exact interleaving of writer-bump-then-notify vs waiter-check-then-await; (b) it ' +
    'genuinely YIELDS the tokio worker (an async .await, not a thread park) so N concurrent durable sends do not ' +
    'occupy N workers — the property F2 exists to fix; (c) the orchestrator sync wait_durable_through ' +
    '(orchestrator.rs:280, main thread) is UNTOUCHED + still correct; (d) CANCEL-SAFETY: the new .await in write_frame ' +
    'block B is a new cancel point — a task drop there must leave NO half-durable/half-sent state and must not violate ' +
    'the mesh.rs cancel-safety invariant (only cancel point = w.rx.recv()); is a durable-not-sent row on cancel safe ' +
    '(replay re-drives)? (e) the writer-death liveness escape (store.rs park_until_durable timeout) must have an async ' +
    'equivalent so a dead writer fails LOUD, never hangs the async waiter forever. Cite file:line.' },
  { key: 'boot-replay-crash', focus:
    'BOOT REPLAY + CRASH-WINDOW + gc ORDERING + INCARNATION. Verify: (a) the replay re-sends via send_durable in ' +
    'ascending scan order at the BUMPED incarnation, and classify_reliable Resets-then-Accepts (no gap_drop) — trace ' +
    'against the receiver ledger; (b) the STRICT ordering: ALL fresh rows durable BEFORE gc_below(new_incarnation) — a ' +
    'crash at every point (after scan, mid-resend, after some fresh durable, after gc) loses nothing (the crash-window ' +
    'table); (c) decode: does the bin re-send the PAYLOAD (decode_frame -> .bytes) or the framed bytes? re-sending the ' +
    'framed bytes would double-frame — confirm the design re-sends the payload; (d) the replay runs BEFORE build_app ' +
    'consumes the transport (app.rs:87) and the peer_writer tasks are already live (spawn_mesh spawned them) so the ' +
    'sends are processed; (e) idempotency: a replayed frame the dest already adopted is deduped (receiver seq-ledger / ' +
    '(transfer, step) journal) — at most one duplicate, no double-adopt; (f) the F3 boot check + real peer counts. ' +
    'Cite file:line.' },
  { key: 'hr-seam-scope', focus:
    'HR-COMPLIANCE, SEAM INTEGRITY, SCOPE, TEST/COVERAGE. Verify: (a) the frozen sim::io seam + wire contract stay ' +
    'UNTOUCHED (DurabilityHandle async method + Notify are io-prod-internal; NodeOutbox/OutboxSink unchanged shape); ' +
    '(b) HR3 one-tooling: shard + gateway both wire via the SAME open_node_outbox (no per-kind fork); no match on ' +
    'shard kind; (c) no magic numbers (the F1 sentinel, any replay cadence, are named/derived); (d) the sub-slicing is ' +
    'clean + each sub-slice is independently reviewable + gate-green, and F2 lands BEFORE any live sink (so block B is ' +
    'production-safe before the bin wiring flips it on); (e) HR5 coverage: the F2 async path + boot replay are ' +
    'coverable (or honestly Tier-B process-tier for the bins); the F1 pin is store-test-hooks (R-6d4 tier); (f) SCOPE ' +
    'HONESTY: R-6d3b closes the RESTART case only — the NEVER-restart orphaned-Arriving loss is R-6d3c, so D-6 #1 is ' +
    'NOT fully closed by 3b; is that stated? Cite file:line.' },
]
const reviews = await parallel(
  LENSES.map((l) => () =>
    agent(
      GROUND + '\n\nThe DESIGNER proposed this R-6d3b design of record:\n--- DESIGN ---\n' + design.design_markdown +
      '\n--- SUB-SLICES ---\n' + design.sub_slices.join('\n') +
      '\n--- CRASH-WINDOW TABLE ---\n' + design.crash_window_table +
      '\n--- OPEN QUESTIONS ---\n' + design.open_questions.join('\n') + '\n--- END ---\n\n' +
      'You are REVIEWER (' + l.key + '). Adversarially review through YOUR lens: ' + l.focus + '\n' +
      'Be code-grounded (cite file:line) + decisive. If sound on your lens, say SOUND_TO_IMPLEMENT with an empty/NIT ' +
      'findings array; do NOT invent problems. If a real defect, rank it + give a concrete fix.',
      { label: 'review:' + l.key, phase: 'Review', schema: REVIEW_SCHEMA, model: 'opus' }
    )
  )
)
const valid = reviews.filter(Boolean)

phase('Synthesize')
const synth = await agent(
  GROUND + '\n\nThe DESIGNER proposed an R-6d3b design of record; three adversarial lenses reviewed it.\n\nDESIGN:\n' +
  design.design_markdown + '\n\nSUB-SLICES:\n' + design.sub_slices.join('\n') + '\n\nCRASH-WINDOW TABLE:\n' +
  design.crash_window_table + '\n\nREVIEWS (JSON):\n' +
  JSON.stringify(valid.map((r) => ({ lens: r.lens, verdict: r.verdict, summary: r.summary, findings: r.findings })), null, 2) +
  '\n\nSynthesize the ADJUDICATED, IMPLEMENT-READY design of record. Deduplicate (same root cause = ONE, note ' +
  'corroboration). CONFIRM or REFUTE each finding against the code, rank by true impact, FOLD the fix into the final ' +
  'design (do not just list it). Down-grade any reviewer CRITICAL the design/code already handles (cite where); ' +
  'up-grade any real lost-wakeup / worker-starvation / boot-replay-loss / cancel hazard. Output the COMPLETE corrected ' +
  'design of record (the F2 async-wait mechanism + signatures, the boot-replay sequence + crash table, the bin wiring, ' +
  'the F1 pin, the sub-slice ordering, the tests, the files touched) so it can be implemented directly — plus a ' +
  'one-line VERDICT: SOUND_TO_IMPLEMENT (with folded must-fixes) | REVISE | RECONSIDER. Decisive; reward sound design.',
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return {
  verdict: synth,
  design_markdown: design.design_markdown,
  sub_slices: design.sub_slices,
  crash_window_table: design.crash_window_table,
  reviews: valid.map((r) => ({ lens: r.lens, verdict: r.verdict, summary: r.summary, findings: r.findings })),
}
