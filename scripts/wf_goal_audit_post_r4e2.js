export const meta = {
  name: 'goal-audit-post-r4e2',
  description: 'Holistic /goal whole-codebase audit at HEAD 2123c19 (R-4e1 load harness + R-4e2 H2 re-key)',
  phases: [{ title: 'Audit' }, { title: 'Verify' }, { title: 'Synthesize' }],
}

const ROOT = '/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system'

const GOAL = [
  'The recurring holistic /goal audit of the voxeldust greenfield rebuild (a transfer-first "Star Citizen',
  'meets Minecraft" voxel MMO). Answer for the CURRENT codebase (HEAD 2123c19): how do all components work',
  'together — complementary or will they break? Is it DRY, MODULAR, robust, SCALABLE, cloud-ready, elegant,',
  'error-prone? AAA MMO quality? PREPARED for the planned game (we are building the BASE, but the end-goal',
  'must stay in memory) WITHOUT painting into a corner?',
  '',
  'END-GOAL (the foundation must not drift): spherical voxel PLANETS; player-built SHIPS walked-inside while',
  'flying (Newtonian); SPACE STATIONS + CITIES on planets, all from BLOCKS; a SIGNAL system for functional-',
  'block communication, signal-HEAVY, POSSIBLY CROSS-SHARD; MULTISHARDING MESHES (ONE location spanning',
  'MULTIPLE shards); CLIENT renders MULTIPLE meshes SIMULTANEOUSLY; players PHYSICALLY COLLIDE cross-boundary',
  '(NO client prediction); PvP; HUNDREDS of users in ONE location. Must be robust, scalable, modular,',
  'extremely DRY, elegant. Transfer-first: transfers proven robust under crash/chaos/kill-9 BEFORE features.',
  '',
  'RECENT DELTA since the last /goal audit (which was DONE_NO_CRITICAL at HEAD ead52d9): the D-6 #1',
  'redelivering-transport hardening continued with TWO new commits —',
  '  - R-4e1 (03f3490): the FIRST real-QUIC N-PEER LOAD test. crates/io-prod/tests/mesh_load.rs — N senders',
  '    (default 16, `VD_MESH_LOAD_NODES`/`just mesh-load` pins 64) fan into ONE receiver; a `LoadShape` config;',
  '    a SIZED receiver inbox making `inbound_dropped_reliable==0` a structural invariant (the CRITICAL flake',
  '    cure a review caught); completion-gated EXACT assertions (per-`from` no-loss/no-dup, per-sender',
  '    reliable_acked keeps pace, gap/stale/dedup==0) + a derived generous progress deadline. H2 UNCHANGED.',
  '  - R-4e2 (2123c19): the H2 per-peer RecvLedger RE-KEY. The receiver dedup ledger went from ONE node-wide',
  '    `Arc<Mutex<BTreeMap<(NodeId,MsgClass),RecvState>>>` to per-peer',
  '    `Arc<RwLock<BTreeMap<NodeId, Arc<Mutex<BTreeMap<MsgClass,RecvState>>>>>>` (std-only, NO new dep) so a',
  '    frame from peer P no longer serializes against peer Q — the RX twin of the per-peer SEND lanes.',
  '    Post-impl-reviewed SOUND (zero race/deadlock/behavior defects); N=64 fan-in green.',
  'REMAINING R-4e work (ledgered, NOT yet landed): R-4e3 (correlated multi-peer outage + L5 pod-reschedule',
  'red-guard + conn_died coverage) and R-4e4 (the R-5 mesh_under_loss.rs producer-less capstone + the D-6 #1',
  'PARTIAL flip). The AwaitAdopt source-crash residual (DEFERRED.md D-6 precondition 1a) is gated on M3+L5.',
  '',
  'RULES (violations are defects): HR1 sealed shards (closed InterShardFlow + Inbound taxonomies, EffectFree',
  'couplings); HR2 generic transfer; HR3 one tooling (one FSM/Fence/registry/shard binary — NEVER match on a',
  'shard kind in features); HR4 features-once-run-anywhere (capability DAG + FrameSpace); HR5 100% Tier-A',
  'region+branch coverage REAL not theater (io-prod/bins Tier-B floor 90); HR6 agent-operable E2E. No magic',
  'numbers (ONE config struct). No client prediction. Players collide. Fence CAS = the only commit point.',
  '',
  'REPO MAP: crates/core, crates/wire (frozen contract + InterShardFlow + seams + admin metric registry),',
  'crates/sim (ECS + saga/authority FSMs + sim::io traits + mem impls), crates/node (build_app + saga_runtime',
  '+ app tick + tracer), crates/connection-plane (gateway), crates/harness, crates/io-prod (quinn mesh + redb',
  'store + admin), crates/bins, crates/tests. Binding docs: docs/design/PLAN.md, docs/design/*.md,',
  'integration.json, DEFERRED.md (the binding interim/stub registry). Use git -C ' + ROOT + ' log --oneline',
  '-14 and git -C ' + ROOT + ' show <sha> for the recent commits.',
].join('\n')

const DIMENSIONS = [
  {
    key: 'concurrency-scale-rx',
    prompt: [
      'DIMENSION: concurrency + scale of the RX/TX planes, post-R-4e2 — the hundreds-in-one-location substrate.',
      'The H2 re-key made the receiver ledger per-peer. Audit the WHOLE mesh concurrency model as a system: the',
      'per-peer SEND lanes (peer_writer) + the new per-peer RX ledger + the node-wide SharedInbox Mutex + the',
      'ArcSwap route hot path. Where is the NEXT serialization bottleneck at N=hundreds (the audit already',
      'flagged SharedInbox — is it honestly bounded + ledgered, or a silent wall)? Does the load test actually',
      'PROVE scale, or give false confidence (loopback hides real contention; is that honestly stated)? Any',
      'unbounded growth (the RecvLedger BTreeMap grows per-peer FOREVER — a churn/DoS memory leak? the',
      'acked_keys set? retry buffers?). Does the RwLock writer-starvation tail matter at scale? Cite',
      'crates/io-prod/src/mesh.rs + the load test. Rank findings; name the single biggest scale risk.',
    ].join('\n'),
  },
  {
    key: 'transfer-robustness-crash',
    prompt: [
      'DIMENSION: transfer-first robustness — the saga + Fence + redelivering transport under crash/chaos, with',
      'the R-4e2 re-key in place. This is the PROJECT THESIS. Read saga.rs, saga_runtime.rs, the directory/Fence,',
      'and the mesh. Did the H2 re-key preserve the at-least-once + exactly-once guarantees end-to-end (the',
      'classify ladder, the inbox-drop rollback atomicity, the StaleEpoch straggler cure across a redial)? Any',
      'NEW wedge/orphan/double-commit/lost-flow from R-4e1/R-4e2? Is the residual producer-less gap (AwaitAdopt,',
      'DEFERRED.md D-6 precondition 1a) still honestly bounded + gated on M3+L5, or did anything change its',
      'status? Is the D-6 #1 flip that R-4e4 will do correctly scoped (partial, transport-layer only)? Cite',
      'file:line. Rank by severity.',
    ].join('\n'),
  },
  {
    key: 'cloud-readiness',
    prompt: [
      'DIMENSION: cloud architecture readiness. Will this run nicely in the cloud (k8s/pods, rolling deploys,',
      'pod reschedules, correlated outages)? Audit: the L5 pod-reschedule gap (peer_writer captures addr once —',
      'a rescheduled pod is dialed stale forever; is the address-source re-plumb ledgered + is R-4e3 going to',
      'red-guard it?); the M3 durable-monotone-incarnation boot-counter (a CrashLooping pod at equal/lower',
      'incarnation gets its fast-restart traffic silently dropped — is this correctly a HARD deploy',
      'precondition?); the metrics observability (R-4d2 orchestrator-only — is gateway/shard blindness a',
      'cloud-blocking gap before a real soak?); the correlated-outage handling (a mass pod eviction — does',
      'confirm-dead flood the inbox?). Are all of these honestly recorded as pre-deploy preconditions in',
      'DEFERRED.md, or is something a silent cloud landmine? Rank by severity.',
    ].join('\n'),
  },
  {
    key: 'endgoal-drift',
    prompt: [
      'DIMENSION: end-goal readiness / drift. For EACH pillar judge PREPARED vs DRIFTED: (1) SIGNALS,',
      'signal-heavy, POSSIBLY CROSS-SHARD block communication — is the seam shaped (InterShardFlow::Signal) and',
      'does R-4b/R-4d producer-backpressure/shed compose with a cross-shard signal fan-out, or would signals',
      'inherit a silent NodeUnreachable/shed (the DEFERRED.md P9 note)? (2) MULTISHARDING MESHES (one location',
      'across shards) — does transfer/coordinate/FrameSpace admit it (D-41)? (3) CLIENT multi-mesh simultaneous',
      'rendering — does the connection-plane/snapshot/ghost model admit compositing multiple meshes? (4) blocks/',
      'ships/stations/cities — shard-agnostic (HR3/HR4)? (5) players collide, no prediction. Is DEFERRED.md',
      'HONEST (every stub recorded with what/where/when)? Flag any decision forcing a costly rework. Rank by',
      'severity.',
    ].join('\n'),
  },
  {
    key: 'dry-modular-elegant',
    prompt: [
      'DIMENSION: DRY + modular + elegant + error-prone + HR3 one-tooling (the goal stresses "extremely DRY,',
      'elegant, modular"). Audit code quality, weighted to R-4d..R-4e2. Duplicated logic that should be one',
      'machinery? A match on a shard kind in feature code (HR3 violation)? Magic numbers outside the ONE config',
      'struct (MeshReliabilityTuning/LivenessTuning/LoadShape/TransportTuning)? Error-prone patterns (an',
      'unchecked unwrap on a network path, a silent drop, an ordering invariant enforced only by a comment)?',
      'Is the R-4e2 two-level-lock code elegant or a smell? Is the load harness DRY (does it duplicate',
      'mesh_redelivery.rs helpers that should be shared)? Is the metric registry / MetricsSource DRY? Cite',
      'file:line. Rank by severity.',
    ].join('\n'),
  },
  {
    key: 'coverage-e2e-load',
    prompt: [
      'DIMENSION: HR5 coverage REALITY + e2e + load/perf testing. Is Tier-A 100% REAL (no assert-free/',
      'tautological tests, no or-pattern hiding untested behavior)? Spot-check the R-4e tests: does mesh_load',
      'actually prove no-loss/no-dup under fan-in (or a false-pass)? is the sized-inbox invariant genuinely',
      'structural? For the BROADER question the goal explicitly wants (100% + e2e + performance/load): the load',
      'harness now EXISTS (R-4e1) — but the correlated-outage / pod-reschedule / R-5 producer-less capstone',
      '(R-4e3/R-4e4) are NOT yet landed. Is that the biggest remaining test gap, honestly ledgered + sequenced?',
      'Where is the E2E coverage (HR6 client harness — exists + runs)? Are the process/chaos tiers actually',
      'gating? Cite the test files + justfile gates. Rank; name the single biggest test gap.',
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
       'Read the actual code + designs before judging — a whole-codebase audit, not a skim. Every finding cites',
       'file:line evidence + a specific fix. Do NOT invent findings; if the dimension is healthy, say so and',
       'return few/no findings. CRITICAL = breaks correctness/robustness/the end-goal NOW; HIGH = a real risk',
       'owed before the next phase. A gap that is honestly ledgered in DEFERRED.md + correctly sequenced as a',
       'future/pre-deploy precondition is NOT a CRITICAL — note it but rank it by its true blocking severity.'].join('\n'),
      { label: 'audit:' + d.key, phase: 'Audit', schema: FINDING_SCHEMA, model: 'opus' }
    )
  )
)

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
       'A whole-codebase audit flagged this candidate. ADVERSARIALLY VERIFY against the ACTUAL code — default',
       'to REFUTING unless the code confirms it. Auditors over-flag; find why it might NOT be real (handled',
       'elsewhere, a misread, chartered/ledgered scope, a non-triggering path at the real config). A gap that',
       'is honestly ledgered + correctly sequenced pre-deploy is NOT a blocking CRITICAL/HIGH — downgrade it.',
       '', 'CANDIDATE (' + c.severity + ', dimension ' + c.dimension + ', area: ' + c.area + '):',
       'PROBLEM: ' + c.problem, 'EVIDENCE: ' + c.evidence, 'PROPOSED FIX: ' + c.fix,
       '', 'Return real, severity_after_review (NOT_A_DEFECT if refuted; downgrade freely), reasoning citing file:line.'].join('\n'),
      { label: 'verify:' + c.dimension + ':' + c.area.slice(0, 18), phase: 'Verify', schema: VERDICT_SCHEMA, model: 'opus' }
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
   'You are the SYNTHESIZER for the holistic /goal audit at HEAD 2123c19. Six dimensions ran; every',
   'CRITICAL/HIGH candidate was adversarially verified (default-refute). Below: the dimension summaries and',
   'the confirmed-real CRITICAL/HIGH findings.',
   '', 'DIMENSION SUMMARIES:', JSON.stringify(audits.map((a, i) => ({ dim: DIMENSIONS[i].key, summary: a && a.dimension_summary })), null, 2),
   '', 'CONFIRMED CRITICAL/HIGH (survived verification):', JSON.stringify(confirmed.map((c) => ({ dimension: c.candidate.dimension, area: c.candidate.area, severity: c.verdict.severity_after_review, problem: c.candidate.problem, fix: c.candidate.fix, why_real: c.verdict.reasoning })), null, 2),
   '', 'ALL candidates + verdicts (context on refutations):', JSON.stringify(verified.filter(Boolean).map((v) => ({ area: v.candidate.area, claimed: v.candidate.severity, after: v.verdict.severity_after_review, real: v.verdict.real })), null, 2),
   '',
   'Read any files needed to adjudicate. Produce the final report: (a) VERDICT — DONE_NO_CRITICAL if zero',
   'confirmed CRITICAL and zero confirmed in-scope-unaddressed HIGH (a ledgered/sequenced-pre-deploy item is',
   'NOT blocking), else FIX_REQUIRED with the specific blocking findings + fixes; (b) each confirmed finding',
   'with a concrete fix + blocks-now vs ledgered-follow-up; (c) the holistic judgement across the end-goal',
   'dimensions — robust, scalable, cloud-ready, DRY, modular, elegant, and PREPARED for signals/multisharding-',
   'meshes/multi-mesh-rendering/PvP/hundreds-in-one-location without painting into a corner; (d) the single',
   'most important next action. Be decisive.'].join('\n'),
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return { candidates_count: candidates.length, confirmed_count: confirmed.length, audits, confirmed, synthesis: synth }
