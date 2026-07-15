export const meta = {
  name: 'goal-audit-post-r4d',
  description: 'Holistic /goal whole-codebase audit after R-4d (transport ops slice complete)',
  phases: [
    { title: 'Audit' },
    { title: 'Verify' },
    { title: 'Synthesize' },
  ],
}

const ROOT = '/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system'

const GOAL = [
  'This is the recurring holistic /goal audit of the voxeldust greenfield rebuild (a transfer-first',
  '"Star Citizen meets Minecraft" voxel MMO). Answer, for the CURRENT state of the codebase: how do all game',
  'components work together — complementary or will they break? Is it DRY, robust, scalable, cloud-ready,',
  'elegant, error-prone? AAA MMO quality? Is it PREPARED for the planned game without painting into a corner?',
  '',
  'THE END-GOAL COMPLEXITY (keep this in mind — the foundation must not drift from it):',
  '  - spherical voxel PLANETS; player-built SHIPS you walk inside while flying (Newtonian physics); SPACE',
  '    STATIONS and CITIES on planets, all built from BLOCKS.',
  '  - a SIGNAL system for functional-block communication, signal-HEAVY, POSSIBLY CROSS-SHARD (blocks on one',
  '    shard driving blocks on another).',
  '  - MULTISHARDING MESHES: ONE location (a huge station/city/ship) spanning MULTIPLE shards.',
  '  - CLIENT renders MULTIPLE meshes SIMULTANEOUSLY (multi-mesh rendering) — composited seamlessly.',
  '  - players PHYSICALLY COLLIDE across shard boundaries (NO client prediction); PvP; HUNDREDS of users in',
  '    ONE location.',
  '  - transfer-first: transfers must be proven robust under crash/chaos/kill-9 BEFORE features pile on.',
  '',
  'RECENT WORK (the delta since the last audit): the D-6 #1 redelivering (at-least-once) MeshTransport',
  'hardening — R-3 (receiver ledger + reverse cumulative-ack), R-4a (retransmit/redial timer + per-lane',
  'confirm-dead), R-4b (bounded retry buffer + shed-loud producer backpressure), R-4c (LivenessTuning',
  'geometric cross-config boot-assert), and NOW R-4d: (M3) Inbound::SendShed disambiguates a LOCAL transport',
  'shed from peer death so a live-but-ack-stalled peer is never false-confirmed dead and destructively',
  're-homed; (M4) the 10 mesh reliability counters are scraped on the orchestrator /metrics. Commits',
  'fd8cdd9 (R-4d1) + ead52d9 (R-4d2) on branch worktree-new-system.',
  '',
  'RULES / STANDARDS (violations are defects): HR1 sealed shards (closed InterShardFlow + Inbound taxonomies,',
  'EffectFree couplings); HR2 generic transfer (one TransferableKind registry); HR3 one tooling (one FSM/',
  'Fence/registry/shard binary — never match on a shard kind in features); HR4 features-once-run-anywhere',
  '(capability DAG + FrameSpace); HR5 100% Tier-A region+branch coverage REAL not theater (io-prod/bins are',
  'Tier-B floor 90); HR6 agent-operable E2E client harness. No magic numbers (ONE config struct). No client',
  'prediction. Players collide. postcard v1. Fence discipline (directory CAS = the only commit point).',
  '',
  'REPO MAP: crates/core (domain), crates/wire (frozen contract + InterShardFlow + seams + admin metric',
  'registry), crates/sim (ECS + saga/authority FSMs + sim::io traits + mem impls), crates/node (build_app +',
  'saga_runtime + app tick + tracer), crates/connection-plane (gateway), crates/harness (Topology/FaultFabric',
  '/oracle/chaos), crates/io-prod (quinn mesh + redb store + admin), crates/bins (the node binaries + process',
  'tests), crates/tests. Binding docs: docs/design/PLAN.md, docs/design/*.md, docs/design/integration.json,',
  'docs/design/DEFERRED.md (the binding interim/stub registry — every deferral MUST be honestly recorded).',
  'Use git -C ' + ROOT + ' log --oneline -12 and git -C ' + ROOT + ' show for the recent commits.',
].join('\n')

const DIMENSIONS = [
  {
    key: 'hr1-seam-integrity',
    prompt: [
      'DIMENSION: HR1 sealed-shard integrity + the CLOSED type taxonomies, post-R-4d.',
      'R-4d added a NEW arm to the frozen sim::io::Inbound seam (SendShed{reason:ShedReason}). Verify the seam',
      'is STILL closed and honest: is Inbound a closed enum with every consumer exhaustively matching (no',
      'swallowing wildcard)? Is ShedReason correctly seam-owned (the mesh-private AssignReject never leaks',
      'across the crate boundary)? Does the new metrics egress (MeshMetrics reading MeshControl::stats,',
      '/metrics exposition) leak anything SEALED (a shard World/rapier/redb detail, a session id, an entity',
      'blob, cross-shard bytes outside InterShardFlow)? Audit InterShardFlow (crates/wire/src/intershard.rs)',
      'and the connection-plane client families: are they still the ONLY egress, still closed, EffectFree where',
      'required? Any place a new feature could smuggle sealed data out. Cite crates/wire + crates/sim/src/io +',
      'crates/io-prod. Rank findings by severity.',
    ].join('\n'),
  },
  {
    key: 'transfer-robustness-crash',
    prompt: [
      'DIMENSION: transfer-first robustness — the saga + Fence + the redelivering transport under crash/chaos.',
      'This is the PROJECT THESIS: transfers must survive kill-9 without wedge/dup/vanish. Read the saga FSM',
      '(crates/sim/src/saga.rs), the saga runtime (crates/node/src/saga_runtime.rs), the directory/Fence, and',
      'the whole R-3/R-4 mesh (crates/io-prod/src/mesh.rs). Scrutinize the R-4d false-confirm cure END-TO-END:',
      'does routing SendShed to a counter (never record_unreachable) actually prevent a live-but-ack-stalled',
      'peer from being confirmed dead AND still confirm a genuinely-dead peer promptly? Any NEW wedge, orphan,',
      'double-commit, or lost-flow path introduced by R-3..R-4d? Is the at-least-once guarantee real (the',
      'idle-after-blip retransmit timer, the receiver ledger surviving teardown, the producer-backpressure',
      'shed never dropping a retained frame)? What is the RESIDUAL producer-less-flow gap (DEFERRED.md D-6',
      'precondition 1: AwaitAdopt) and is it honestly bounded? Cite file:line. Rank by severity.',
    ].join('\n'),
  },
  {
    key: 'scale-cloud-pvp',
    prompt: [
      'DIMENSION: scale + cloud + PvP + hundreds-in-one-location.',
      'Can this architecture carry hundreds of players in one location with PvP? Audit: the 20Hz hot path',
      '(is it lock-free — ArcSwap routes, no lock across an awaited send)? the mesh per-(peer,class) lane model',
      'under N peers + correlated outages (a mass disconnect — does confirm-dead flood the BoundedInbox and',
      'evict genuine reliable inbound)? the directory single-writer — is it a scaling bottleneck at hundreds of',
      'transfers? the bounded retry buffer under a burst (does producer-backpressure shed the RIGHT thing)? the',
      'new /metrics — is orchestrator-only enough to observe a real soak, or is gateway/shard blindness',
      'cloud-blocking? Is there a LOAD/VOLUME test proving any of this, or is scale unproven (the user',
      'explicitly wants performance + load testing)? Cite the mesh, gateway, directory, harness. Rank by',
      'severity; call out the single biggest scale risk.',
    ].join('\n'),
  },
  {
    key: 'endgoal-drift',
    prompt: [
      'DIMENSION: end-goal readiness / drift — is the foundation converging on the planned game without',
      'painting into a corner? For EACH end-goal pillar, judge whether the current foundation is PREPARED or',
      'has drifted: (1) SIGNALS, signal-heavy, POSSIBLY CROSS-SHARD block communication — is there a seam for',
      'it (the Signal arm of InterShardFlow) or would it require a redesign? (2) MULTISHARDING MESHES (one',
      'location across shards) — does the transfer/coordinate/FrameSpace machinery admit this, or assume',
      'one-shard-per-location? (3) CLIENT multi-mesh simultaneous rendering — does the connection-plane /',
      'snapshot / ghost model admit compositing multiple meshes, or assume one? (4) blocks/ships/stations/',
      'cities on planets — is the block/realm/persistence model shard-agnostic (HR3/HR4) and ready? (5) players',
      'collide cross-boundary, no prediction. Read PLAN.md, the design docs, integration.json, DEFERRED.md, and',
      'the code that exists. Is DEFERRED.md HONEST (every stub/interim recorded with what/where/when)? Flag any',
      'decision that would force a costly rework later. Rank by severity.',
    ].join('\n'),
  },
  {
    key: 'dry-elegance-quality',
    prompt: [
      'DIMENSION: DRY + elegance + error-prone + AAA quality + HR3 one-tooling.',
      'Audit code quality across the codebase, weighted to the recent R-3..R-4d work. Look for: duplicated',
      'logic that should be one machinery (HR3 — one FSM/Fence/registry/shard binary; NEVER a match on a shard',
      'kind in feature code); magic numbers inline instead of the ONE config struct (TransportTuning/',
      'TransferTuning/MeshReliabilityTuning/LivenessTuning); error-prone patterns (unchecked unwrap on a',
      'network path, a silent drop, an ordering invariant enforced only by a comment); non-elegant seams that',
      'will rot. Is the new R-4d code (Inbound::SendShed, MetricsSource, render_metrics, the or-pattern',
      'consumers) DRY + elegant, or does it introduce a smell? Any place the same concept is expressed two',
      'ways. Cite file:line. Rank by severity.',
    ].join('\n'),
  },
  {
    key: 'coverage-e2e-load',
    prompt: [
      'DIMENSION: HR5 coverage REALITY + e2e + load testing (the user explicitly wants 100% + e2e +',
      'performance/load testing).',
      'Is Tier-A coverage 100% REAL, not theater (are branches genuinely exercised, or are there assert-free',
      'tests, tautological tests, or or-patterns hiding an untested behavior)? Spot-check the R-4d tests: does',
      'the SendShed false-confirm regression test actually prove the cure? does the RetryBufferFull integration',
      'test prove the emit path (not a false-pass)? is the render_metrics completeness guard real? For the',
      'BROADER question: where is the E2E coverage (HR6 client harness — does it exist and run?) and where is',
      'the LOAD/PERFORMANCE testing? The redelivering transport has NO real-QUIC N-peer load test yet (R-4e is',
      'planned but not landed). Is that the biggest test gap? Are the process-tier (real-binary QUIC) + chaos',
      'tiers actually gating, or in-process-only? What must be load-tested before a soak/deploy and is it',
      'ledgered? Cite the test files + justfile gates. Rank by severity; name the single biggest test gap.',
    ].join('\n'),
  },
]

const FINDING_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['dimension_summary', 'findings'],
  properties: {
    dimension_summary: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'area', 'problem', 'evidence', 'fix'],
        properties: {
          severity: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'] },
          area: { type: 'string' },
          problem: { type: 'string' },
          evidence: { type: 'string' },
          fix: { type: 'string' },
        },
      },
    },
  },
}

phase('Audit')
const audits = await parallel(
  DIMENSIONS.map((d) => () =>
    agent(
      [GOAL, '', 'YOUR AUDIT DIMENSION:', d.prompt, '',
       'Read the actual code + designs before judging — this is a whole-codebase audit, not a skim. Be',
       'concrete: every finding cites file:line evidence and a specific fix. Do NOT invent findings to seem',
       'thorough; if the dimension is healthy, say so and return few/no findings. Reserve CRITICAL for a defect',
       'that breaks correctness/robustness/the end-goal; HIGH for a real risk owed before the next phase.'].join('\n'),
      { label: 'audit:' + d.key, phase: 'Audit', schema: FINDING_SCHEMA, model: 'opus' }
    )
  )
)

// Collect every CRITICAL/HIGH candidate for adversarial verification (barrier is correct here:
// we need the full finding set before deciding what to verify, and we dedup across dimensions).
const candidates = []
for (let i = 0; i < audits.length; i++) {
  const a = audits[i]
  if (!a) continue
  for (const f of a.findings || []) {
    if (f.severity === 'CRITICAL' || f.severity === 'HIGH') {
      candidates.push({ ...f, dimension: DIMENSIONS[i].key })
    }
  }
}
log('Audit done: ' + candidates.length + ' CRITICAL/HIGH candidates to adversarially verify')

phase('Verify')
const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['real', 'severity_after_review', 'reasoning'],
  properties: {
    real: { type: 'boolean' },
    severity_after_review: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NOT_A_DEFECT'] },
    reasoning: { type: 'string' },
  },
}

const verified = await parallel(
  candidates.map((c) => () =>
    agent(
      [GOAL, '',
       'A whole-codebase audit flagged this candidate finding. ADVERSARIALLY VERIFY it against the ACTUAL',
       'code — default to REFUTING it unless the code confirms it. Auditors over-flag; your job is to find',
       'why it might NOT be real (already handled elsewhere, a misread, chartered/ledgered scope, a',
       'non-triggering path at the real config). Read the cited files.',
       '', 'CANDIDATE (' + c.severity + ', dimension ' + c.dimension + ', area: ' + c.area + '):',
       'PROBLEM: ' + c.problem, 'EVIDENCE CLAIMED: ' + c.evidence, 'PROPOSED FIX: ' + c.fix,
       '', 'Return: real (does the defect genuinely exist + matter?), severity_after_review (downgrade freely;',
       'NOT_A_DEFECT if refuted), and reasoning citing file:line.'].join('\n'),
      { label: 'verify:' + c.dimension + ':' + c.area.slice(0, 20), phase: 'Verify', schema: VERDICT_SCHEMA, model: 'opus' }
    ).then((v) => ({ candidate: c, verdict: v }))
  )
)

const confirmed = verified
  .filter(Boolean)
  .filter((v) => v.verdict && v.verdict.real &&
    (v.verdict.severity_after_review === 'CRITICAL' || v.verdict.severity_after_review === 'HIGH'))

phase('Synthesize')
const synth = await agent(
  [GOAL, '',
   'You are the SYNTHESIZER for the holistic /goal audit. Six dimension auditors ran; every CRITICAL/HIGH',
   'candidate was adversarially verified (default-refute). Here are the full audit summaries and the',
   'confirmed-real CRITICAL/HIGH findings after verification.',
   '', 'DIMENSION SUMMARIES:', JSON.stringify(audits.map((a, i) => ({ dim: DIMENSIONS[i].key, summary: a && a.dimension_summary })), null, 2),
   '', 'CONFIRMED CRITICAL/HIGH (survived adversarial verification):', JSON.stringify(confirmed.map((c) => ({ dimension: c.candidate.dimension, area: c.candidate.area, severity: c.verdict.severity_after_review, problem: c.candidate.problem, fix: c.candidate.fix, why_real: c.verdict.reasoning })), null, 2),
   '', 'ALL candidates + verdicts (for context on what was refuted):', JSON.stringify(verified.filter(Boolean).map((v) => ({ area: v.candidate.area, claimed: v.candidate.severity, after: v.verdict.severity_after_review, real: v.verdict.real })), null, 2),
   '',
   'Read any files you need to adjudicate. Produce the final audit report: (a) the VERDICT — DONE_NO_CRITICAL',
   'if zero confirmed CRITICAL and zero confirmed in-scope-unaddressed HIGH, else FIX_REQUIRED; (b) for each',
   'confirmed finding, a concrete fix + whether it blocks now or is a ledgered follow-up; (c) a holistic',
   'judgement across the end-goal dimensions — is the foundation robust, scalable, cloud-ready, DRY, elegant,',
   'and PREPARED for signals/multisharding-meshes/multi-mesh-rendering/PvP/hundreds-in-one-location without',
   'painting into a corner; (d) the single most important next action. Be decisive.'].join('\n'),
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return { candidates_count: candidates.length, confirmed_count: confirmed.length, audits, confirmed, synthesis: synth }
