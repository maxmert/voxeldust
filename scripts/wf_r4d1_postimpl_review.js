export const meta = {
  name: 'r4d1-postimpl-review',
  description: 'Adversarial post-impl review of R-4d1 (Inbound::SendShed seam + false-confirm cure)',
  phases: [{ title: 'Review' }, { title: 'Synthesize' }],
}

const ROOT = '/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system'

const CONTEXT = [
  'R-4d1 (M3) just landed in the voxeldust greenfield rebuild. It disambiguates a LOCAL transport send-shed',
  'from a peer-unreachability by adding a NEW arm to the frozen sim::io::Inbound seam: SendShed{to,class,',
  'undelivered,reason:ShedReason{Unframable,RetryBufferFull}}. The point (the false-confirm cure): the io-prod',
  'mesh bounded retry buffer (R-4b) refuses a send when full or oversize; that refusal previously surfaced as',
  'Inbound::NodeUnreachable, which the orchestrator saga_runtime routed to record_unreachable -> a live-but-',
  'ack-stalled peer got FALSE-CONFIRMED dead and destructively re-homed. Now the mesh emits SendShed, and',
  'saga_runtime routes it to a counter (runtime.sends_shed) and NEVER to the liveness tracker.',
  '',
  'The change was implemented from a pre-vetted design (3 adversarial reviewers + synthesizer). Two DELIBERATE',
  'deviations from the raw design that you MUST scrutinize:',
  '',
  '(1) COVERAGE STRATEGY (HR5 = Tier-A crates at 100% region+branch, REAL not theater). The design proposed',
  'DISTINCT SendShed arms in every Inbound consumer. But four Tier-A consumers get their inbound only from the',
  'mem MemHub / FaultFabric, which have NO retry buffer and so can NEVER produce a SendShed. A distinct arm',
  'there would be an UNCOVERABLE region (failing HR5). So those four use an exhaustive OR-PATTERN merging',
  'SendShed with the nearest already-covered arm, each with an honest comment:',
  '  - node/src/tracer.rs: NodeUnreachable | SendShed => push(PeerUnreachable) (a bare probe cannot distinguish)',
  '  - harness/src/client.rs: NodeUnreachable | SendShed => unreachable+=1, and shed:0 literal on its TickReport',
  '  - harness/src/chaos.rs: NodeUnreachable | SendShed => None (observed, never forwarded)',
  '  - harness/src/fabric.rs (test): Wire | SendShed => {} (neither is a flap-notice)',
  'The design also wanted a new TraceEvent::SendShed; I DROPPED it (a never-constructed variant makes its',
  'derived Debug/PartialEq arms uncoverable). The LOAD-BEARING distinction is kept DISTINCT + covered by direct',
  'injection: the seam reliability() arm; saga_runtime routing + sends_shed(); node/src/app.rs drain_phase',
  'shed count on TickReport (covered via a PerPeerLanes test transport seeded with a SendShed).',
  '',
  '(2) The design MISSED one exhaustive consumer: io-prod/src/mesh.rs ~line 2360 (a filter_map match in a test,',
  'NOT a matches! filter as the design claimed). I added a SendShed=>None arm there.',
  '',
  'Coverage is GREEN: coverage-fast reports Tier-A 100% region+branch+function (33038 regions, 0 uncovered).',
  'clippy -D warnings clean; fmt clean; all affected suites pass. A NEW io-prod integration test',
  '(mesh_redelivery.rs a_full_retry_buffer_sheds_...) proves the RetryBufferFull emit path end-to-end with a',
  'determinism argument: retry_buffer_max_bytes at the MINIMUM (one maximal framed frame) + near-MAX frames',
  'that fit quinn stream window (~1.25MB, so the write returns BEFORE B acks) => the 2nd frame in a rapid pair',
  'physically cannot be retained => deterministic RetryBufferFull shed, asserting NO NodeUnreachable.',
  '',
  'Files to read (all under ' + ROOT + '):',
  '  crates/sim/src/io/mod.rs (the seam: Inbound::SendShed, ShedReason, reliability(), Transport doc, the test)',
  '  crates/io-prod/src/mesh.rs (WriteFail::Shed(ShedReason), the reject-site AssignReject->ShedReason map, the',
  '    peer_writer Shed arm emit + debug_assert, the 2360 filter_map arm, the flipped oversize test)',
  '  crates/node/src/saga_runtime.rs (the routing arm ~1690, sends_shed field/accessor, the regression test',
  '    a_send_shed_is_counted_and_never_confirms_a_live_peer_dead)',
  '  crates/node/src/app.rs (TickReport.shed, TickPrologue, drain_phase, run_schedule/flush_outbox threading,',
  '    the PerPeerLanes inbound seam + the test a_send_shed_is_counted_separately_from_an_unreachable)',
  '  crates/node/src/tracer.rs, crates/harness/src/{client,chaos,fabric}.rs (the or-pattern consumers)',
  '  crates/sim/src/io/mem.rs (the parity-by-absence doc note)',
  '  crates/bins/tests/process_parity.rs (the /tests/-ignored no-op arm)',
  '  crates/io-prod/tests/mesh_redelivery.rs (the new RetryBufferFull integration test at the file tail)',
  'Use: git -C ' + ROOT + ' diff HEAD to see the exact changeset.',
].join('\n')

const LENSES = [
  {
    key: 'seam-hr1-exhaustiveness',
    prompt: [
      'LENS: the frozen seam + HR1 closed-taxonomy + exhaustive-consumer completeness.',
      'Verify EVERY consumer of sim::io::Inbound across the WHOLE workspace was correctly updated (grep for',
      'every match on Inbound; classify each as exhaustive-match-needing-an-arm vs let-else/matches!/constructor',
      'that falls through). Did I miss ANY exhaustive match that should have a SendShed arm (a silent-swallow or',
      'a compile that only works because of a wildcard that now hides the new variant)? Confirm the mesh.rs 2360',
      'filter_map catch is correct and that there are no OTHER design-missed consumers. Assess whether the',
      'OR-PATTERN merges preserve the compile-time exhaustiveness guarantee (a hypothetical 3rd Inbound variant',
      'still forces every consumer to reconsider) AND are semantically honest (is folding SendShed into',
      'unreachable/PeerUnreachable/None/{} defensible for THAT consumer, or does it hide a real behavior a',
      'consumer owes?). Is ShedReason correctly seam-owned (never leaking AssignReject across the crate boundary)?',
    ].join('\n'),
  },
  {
    key: 'correctness-false-confirm',
    prompt: [
      'LENS: the correctness of the false-confirm cure + no-new-wedge.',
      'Trace the saga_runtime routing arm: is it TRUE that a SendShed never reaches record_unreachable, never',
      'clears liveness evidence, and never wedges or abandons a saga? Consider: a shed toward a peer that is ALSO',
      'genuinely dying (mixed shed + real NodeUnreachable) — does the peer still get correctly confirmed dead via',
      'the real notices? A shed toward a source vs a dest mid-transfer — any path where dropping the shed on the',
      'floor (counting + continue) strands a saga that the OLD NodeUnreachable-routing would have progressed?',
      'Is the mesh peer_writer debug_assert invariant (a SendShed can only arise on a reliable lane) actually',
      'true, and is the reliability gate that follows it correct (not double-counting, not dropping a reliable',
      'shed)? Does the undelivered MsgId correlation stay correct? Is there any at-least-once/redelivery',
      'interaction where a shed-then-later-delivered frame double-counts or mis-orders?',
    ].join('\n'),
  },
  {
    key: 'coverage-honesty-and-test',
    prompt: [
      'LENS: HR5 coverage REALITY (not theater) + the new integration test rigor.',
      'The or-pattern strategy claims llvm-cov treats each match arm as ONE region covered by the matched',
      'alternative (so a never-matched SendShed alternative in an or-pattern is NOT a separate uncovered region).',
      'coverage-fast is reported 100%. Scrutinize: is this REAL coverage, or does an or-pattern merge HIDE a',
      'behavior that a proper test should exercise (i.e. is 100% being bought by making a genuinely-distinct',
      'behavior indistinguishable)? For each of the three DISTINCT+covered arms (seam reliability, saga_runtime',
      'routing, app.rs shed count), confirm the test actually exercises the real code path (not a shim). Attack',
      'the new mesh integration test determinism argument: could it FLAKE (quinn stream_receive_window smaller',
      'than assumed, ack racing the burst, the write blocking instead of returning)? Is asserting reliable_shed>=1',
      '+ SendShed{RetryBufferFull} + no-NodeUnreachable the right set, or is there a false-pass where the test',
      'goes green without proving the emit path? Is the PerPeerLanes inbound-injection seam a legitimate test',
      'double or does it bypass something real?',
    ].join('\n'),
  },
]

phase('Review')
const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['verdict', 'findings', 'summary'],
  properties: {
    verdict: { type: 'string', enum: ['SOUND_TO_COMMIT', 'FIX_BEFORE_COMMIT'] },
    summary: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'location', 'problem', 'fix'],
        properties: {
          severity: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'] },
          location: { type: 'string' },
          problem: { type: 'string' },
          fix: { type: 'string' },
        },
      },
    },
  },
}

const reviews = await parallel(
  LENSES.map((l) => () =>
    agent(
      [CONTEXT, '', 'YOUR REVIEW LENS:', l.prompt, '',
       'Read the actual code before judging. Be adversarial and concrete: cite file:line. A finding must name a',
       'REAL defect with a specific fix, not a style preference. If the slice is sound on your lens, say so',
       'plainly (do not invent findings). Return the structured verdict.'].join('\n'),
      { label: 'review:' + l.key, phase: 'Review', schema: SCHEMA, model: 'opus' }
    )
  )
)

phase('Synthesize')
const synth = await agent(
  [CONTEXT, '',
   'You are the SYNTHESIZER. Three adversarial reviewers returned verdicts on R-4d1:',
   JSON.stringify(reviews, null, 2), '',
   'Adjudicate against the ACTUAL code (read the files yourself to confirm or refute each finding — reviewers',
   'sometimes flag non-issues or miss context). Produce: (a) the deduplicated, severity-ranked list of REAL',
   'findings that must change before commit (CRITICAL/HIGH = blocking; MEDIUM = fix-now-if-cheap; LOW = ledger),',
   'each with a concrete fix; (b) an explicit ruling on the two deviations (the or-pattern coverage strategy and',
   'the dropped TraceEvent::SendShed) — sound or not; (c) a final verdict: SOUND_TO_COMMIT or FIX_BEFORE_COMMIT.',
   'Be decisive and specific.'].join('\n'),
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return { reviews, synthesis: synth }
