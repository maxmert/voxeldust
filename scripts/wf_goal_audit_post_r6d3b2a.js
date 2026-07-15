export const meta = {
  name: 'goal-audit-post-r6d3b2a',
  description: 'Holistic /goal whole-codebase audit (post-R-6d3b-2a, HEAD 1789e49): compose/robust/scale/AAA/feature-ready',
  phases: [
    { title: 'Audit', detail: '6 opus dimensions over the whole codebase + designs + DEFERRED' },
    { title: 'Synthesize', detail: 'dedup + severity-rank + DONE_NO_CRITICAL-or-findings verdict' },
  ],
}

const FINDINGS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['dimension', 'verdict', 'summary', 'findings'],
  properties: {
    dimension: { type: 'string' },
    verdict: { type: 'string', enum: ['SOUND', 'CONCERNS', 'CRITICAL'] },
    summary: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['severity', 'title', 'area', 'claim', 'evidence', 'recommendation', 'confidence'],
        properties: {
          severity: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NIT'] },
          title: { type: 'string' },
          area: { type: 'string', description: 'subsystem / file' },
          claim: { type: 'string' },
          evidence: { type: 'string', description: 'code/design-grounded; cite file:line or doc' },
          recommendation: { type: 'string' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
      },
    },
  },
}

const CTX = [
  'This is the recurring HOLISTIC /goal audit of the Voxeldust greenfield rebuild (worktree new-system, HEAD 1789e49).',
  'Voxeldust = "Star Citizen meets Minecraft": spherical voxel planets, player-built ships/space-stations/planet-cities',
  'walked-inside-while-flying, Newtonian physics, players PHYSICALLY COLLIDE cross-boundary (NO client prediction), PvP,',
  'HUNDREDS of users in one location, SIGNAL-heavy functional-block gameplay (possibly CROSS-SHARD, enabling stations +',
  'cities), MULTISHARDING MESHES (ONE location spanning shards), client SIMULTANEOUS multi-mesh rendering. Built',
  'transfer-first: prove cross-shard transfers robust BEFORE features.',
  '',
  'THE AUDIT QUESTION (answer for your dimension): how do all game components work together? Complementary or will they',
  'break? DRY? Robust? Scalable? Will it work nicely in the cloud (k3d/K8s)? Elegant? Error-prone? AAA-MMO quality?',
  'Prepared for the planned features? Is PvP + large-scale locations (hundreds in one location) possible? If there are',
  'NO CRITICAL findings the audit is DONE; CRITICAL/HIGH findings must be surfaced with evidence + a recommendation.',
  '',
  'RECENT WORK to scrutinize especially: the R-6d durable-outbox arc just landed — R-6d2c (FSM write-through/delete-',
  'through), R-6d3a (durable-before-send GATE + shared-outbox injection), R-6d3b-1 (async block-B wait, worker-',
  'starvation fix), R-6d3b-2a (boot-replay machinery: deadlock-free + count-anchored fence). This closes D-6 #1 (a',
  'source shard crash losing a producer-less reliable one-shot: TransientBatch handoff / band-exit Ghost::Despawn) —',
  'RESTART case pending R-6d3b-2b (bin wiring), NEVER-restart case pending R-6d3c (saga AwaitAdopt split + dest',
  'TransientDiscard). The redelivering transport arc R-1..R-6 is nearly complete.',
  '',
  'GROUNDING (read what your dimension needs): docs/design/PLAN.md (the 6 hard rules HR1-HR6 + roadmap P0-P11);',
  'docs/design/*.md (connection_plane, transfer_protocol, sealed_shards, generic_transfer, test_harness,',
  'identity_persistence, coverage_e2e); docs/design/integration.json (19 cross-design resolutions + glossary);',
  'docs/design/DEFERRED.md (the BINDING registry of every interim/stub + WHERE/WHEN it lands — read the D-6 / R-6 arc',
  'entries + any 🟥 open items); crates/{core,wire,sim,node,connection-plane,io-prod,harness,bins}; CLAUDE.md.',
  'CURRENT STATE (fact): P0-P3 foundation hardening, pre-voxel; the transport/transfer/durability layers are the built',
  'surface; voxels/physics/blocks/signals/ships/warp are DESIGNED (in the roadmap + designs) but NOT yet built.',
  '',
  'Be code+design-grounded (cite file:line or the design doc). A finding with no evidence is worthless. Distinguish a',
  'REAL will-break/compose-conflict/scale-cliff (CRITICAL/HIGH) from a not-yet-built-but-honestly-tracked item (LOW/NIT',
  '— the DEFERRED registry tracking a future slice is NOT a defect). Reward genuine soundness. Do NOT invent problems.',
].join('\n')

phase('Audit')
const DIMS = [
  { key: 'compose-transfer-durability', focus:
    'COMPOSITION of the transfer + durability + saga + transport layers — do they work together or will they break? ' +
    'Trace: the redelivering transport (R-1..R-5 at-least-once) + the durable incarnation (M3/R-6a-c) + the durable ' +
    'outbox (R-6d: write-through, durable-before-send gate, async block-B wait, boot-replay) + the transfer saga ' +
    '(AwaitAdopt / TransientBatch / Ghost) + the directory CAS commit point + the Fence. Are the SEAMS consistent ' +
    '(no invariant conflict between the outbox re-drive and the saga re-drive; the incarnation used consistently for ' +
    'stamping + gc + dedup; the frozen sim::io seam intact)? Is D-6 #1 closure honestly staged (restart=2b, never-' +
    'restart=3c)? Any DOUBLE-delivery / double-adopt / lost-flow path across the composed layers? Cite file/doc.' },
  { key: 'scale-density-pvp-cloud', focus:
    'SCALE / hundreds-in-one-location / PvP / cloud. Can the built layers carry HUNDREDS of shards + hundreds of ' +
    'players in ONE location (multisharding meshes)? Audit: the mesh transport fan-out (per-peer writers, the outbox ' +
    'shared across N peers — is OUTBOX_WRITER_CHANNEL_DEPTH=256 vs real peer counts OK? the F3 boot-check), the ' +
    'durable-send concurrency (does the async block-B wait actually prevent worker starvation at N concurrent durable ' +
    'sends?), the gateway snapshot/datagram budget, cross-shard signal load (the future signal system riding the ' +
    'transport), the density cures pre-designed vs built (load-tested to what N?), the k3d/CA-1 no-DNS blocker. Where ' +
    'is the next scale CLIFF? Is anything O(bad) in the hot path? Cite file/doc + the load-test evidence.' },
  { key: 'dry-onetooling-hr-elegance', focus:
    'DRY / ONE-TOOLING (HR1-HR6) / ELEGANCE. Audit for: HR1 sealed shards (InterShardFlow the only egress; EffectFree ' +
    'couplings), HR2 generic transfer (ONE TransferableKind machinery, Durable vs Transient = policy fan-out), HR3 one ' +
    'tooling (ONE transfer FSM/Fence/registry/shard binary; NEVER match on shard kind), HR4 features-once-run-anywhere ' +
    '(capability DAG + FrameSpace), no-magic-numbers (operational params in ONE config struct). Any DUPLICATION, any ' +
    'shard-kind fork, any inline literal that should be a named const, any abstraction that leaks or is over-built? Is ' +
    'the R-6d outbox machinery DRY (one replay home, one gate)? Cite file:line.' },
  { key: 'robust-crash-noloss', focus:
    'ROBUSTNESS / CRASH-SAFETY / NO-SILENT-LOSS / ERROR-HANDLING. Audit the fail-loud discipline (a refusal is never a ' +
    'loss; panics-not-silent-degrade at the durable boundary), the crash matrix (saga-phase x victim x CrashWhen), the ' +
    'at-least-once + dedup, the durable-before-effect / durable-before-send gates. Read DEFERRED.md for any 🟥 open ' +
    'item or any silent-loss RESIDUAL (D-6 #1 restart/never-restart; the AwaitAdopt orphan-Arriving; the redelivering ' +
    'transport residuals). Is there any path that DROPS a durable flow silently, DOUBLE-applies, or WEDGES (a stuck ' +
    'saga / a hung boot / a deadlock)? Are the new R-6d error arms (ReplayError, FenceTimeout, LaneStuck) fail-safe? ' +
    'Cite file:line / DEFERRED entry.' },
  { key: 'feature-readiness', focus:
    'FEATURE-READINESS for the planned game — is the foundation prepared, or is it painting into a corner? For EACH: ' +
    '(a) the SIGNAL system (functional-block comms, POSSIBLY CROSS-SHARD — the enabler for space stations + cities); ' +
    '(b) block-built everything (ships/stations/planet-cities as voxel realms; shard-agnostic block runtime); (c) ' +
    'multisharding meshes (ONE location spanning shards) + client SIMULTANEOUS multi-mesh rendering; (d) ships walked-' +
    'inside-while-flying (nested frames, CouplingPort/ShipThrustPort); (e) warp/provisioning; (f) PvP + damage-across-' +
    'transfer. Does the BUILT foundation (InterShardFlow taxonomy, the transfer/coupling/signal arms, FrameSpace, the ' +
    'capability DAG, the durable transport) SUPPORT these per the designs, or is there a structural gap that a future ' +
    'slice cannot bridge without a rework? Cite the design doc + the wire/sim seam that carries each. Not-yet-built is ' +
    'fine IF the seam is planted; a MISSING seam that blocks a feature is the finding.' },
  { key: 'test-coverage-e2e-load', focus:
    'TEST COVERAGE / E2E / LOAD / DETERMINISM (HR5 + HR6). Audit: Tier-A 100% region+branch (is it real, not theater? ' +
    'any mutation-blind tests like the ones prior reviews caught?), the io-prod Tier-B ratcheted floor, the process/ ' +
    'chaos tiers, the determinism gates (byte-identical replay), the load tests (mesh N-peer, gateway) — do perf/load ' +
    'tests exist for the scale-bearing subsystems or are they deferred? the HR6 agent-operable client E2E (dev-control/ ' +
    'vdctl/readback). What is the biggest TEST GAP that could hide a will-break at scale/crash? Is the deferred-to-R-6d4 ' +
    'store-test-hooks pin set (no-premature-gc, F1, two_peers_do_not_starve, writer-death-wake) the right call or does ' +
    'something need a default-tier guard NOW? Cite file/just recipe/DEFERRED.' },
]
const audits = await parallel(
  DIMS.map((d) => () =>
    agent(
      CTX + '\n\nYou are the AUDITOR for dimension: ' + d.key + '.\nFOCUS: ' + d.focus,
      { label: 'audit:' + d.key, phase: 'Audit', schema: FINDINGS_SCHEMA, model: 'opus' }
    )
  )
)
const valid = audits.filter(Boolean)
const allFindings = valid.flatMap((a) => a.findings.map((f) => ({ ...f, dimension: a.dimension })))

phase('Synthesize')
const synth = await agent(
  CTX + '\n\nSix dimension-auditors reviewed the whole codebase. Their verdicts + findings (JSON):\n' +
  JSON.stringify({ verdicts: valid.map((a) => ({ dimension: a.dimension, verdict: a.verdict, summary: a.summary })), findings: allFindings }, null, 2) +
  '\n\nSynthesize the HOLISTIC /goal verdict. Deduplicate (same root cause across dimensions = ONE, note corroboration). ' +
  'CONFIRM or REFUTE each finding against the code/design, and RE-RANK by true impact — CRUCIALLY distinguish a REAL ' +
  'will-break / compose-conflict / scale-cliff / silent-loss (CRITICAL/HIGH, actionable NOW) from a not-yet-built-but-' +
  'honestly-tracked roadmap item (the DEFERRED registry doing its job is NOT a defect — down-rank to LOW/NIT). ' +
  'Down-grade any auditor CRITICAL the code/design already handles (cite where). For each surviving CRITICAL/HIGH give ' +
  'a concrete fix + WHERE it belongs (a specific slice). Then the VERDICT: "DONE_NO_CRITICAL" (no CRITICAL/HIGH that is ' +
  'a real defect in the BUILT surface — roadmap gaps are expected) OR "FINDINGS" with the ranked blocking list. Be ' +
  'decisive + evidence-grounded; reward the genuine soundness of the transfer-first foundation where it holds.',
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return { verdict: synth, findingCount: allFindings.length, verdicts: valid.map((a) => ({ dimension: a.dimension, verdict: a.verdict })) }
