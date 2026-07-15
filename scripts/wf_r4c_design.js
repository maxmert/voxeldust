export const meta = {
  name: 'r4c-design',
  description: 'Design + adversarially review R-4c (LivenessTuning geometric cross-config invariant + transport env-plumb) before impl',
  phases: [
    { title: 'Design', detail: 'concrete R-4c design from the real code' },
    { title: 'Review', detail: '3 adversarial lenses' },
    { title: 'Synthesize', detail: 'vetted implement-ready design' },
  ],
}

const CONTEXT = [
  'Voxeldust redelivering (at-least-once) MeshTransport. R-1..R-4b are LANDED (HEAD e19bae7). R-4a made confirm-dead LIVE:',
  'the io-prod mesh peer_writer bounces Inbound::NodeUnreachable toward a peer after confirm_unreachable_after_retries (default',
  '3) consecutive FAILED redial cycles, spaced by the WALL-CLOCK GEOMETRIC backoff (redial_backoff_min 50ms doubling to',
  'redial_backoff_max 5s). The orchestrator saga LivenessTracker counts those notices: is_confirmed_dead flips after',
  'n_consecutive_unreachable (prod default 3) notices within unreachable_window_ticks; the DESTRUCTIVE re-home/abandon then',
  'waits abort_deadline_ticks from dead_observed_since. Universe tick = 20 Hz = 50 ms/tick (core/src/ids.rs).',
  '',
  'THE PROBLEM R-4c FIXES (the C3 finding, refined by the post-R-4b holistic audit wf_9d72e38c): the transport clock',
  '(MeshReliabilityTuning) and the saga clock (SagaTuning.abort_deadline_ticks + LivenessTuning{n_consecutive_unreachable,',
  'unreachable_window_ticks, retry_delay_ticks_hint}) are two INDEPENDENT config structs, each validated in ISOLATION, with',
  'NOTHING asserting they are ORDERED. The current LivenessTuning::validate cross-checks unreachable_window_ticks >=',
  'n_consecutive_unreachable * retry_delay_ticks_hint — a LINEAR span against a hand-entered tick hint that has ZERO structural',
  'relationship to the transport GEOMETRIC backoff (50ms->5s). So the window check is vacuous w.r.t. the real transport, and a',
  'mis-tuned deployment (a tightened VD_SAGA_ABORT_DEADLINE, or a window below the geometric spread) can silently STRAND sagas',
  '(the transport confirms dead LATER than the operator abort budget models => Timeout re-drive forever) or NEVER RE-CONFIRM a',
  'long-dead peer (once bounces spread past the window at backoff_max=5s, each late notice resets the consecutive run to 1). At',
  'the DEFAULT tuning this is SAFE (the audit proved it: destructive abort is double-gated + strictly downstream of the ~0.35-1.55s',
  'confirm), so R-4c is DEFENSE-IN-DEPTH against MIS-TUNING, not a fix for a live default-path strand.',
  '',
  'R-4c MUST DELIVER (design each precisely against the real code):',
  '1. LivenessTuning::validate_against(&SagaTuning, &MeshReliabilityTuning) (or the equivalent cross-struct check) — assert:',
  '   (a) the transport WORST-CASE confirm latency in TICKS <= abort_deadline_ticks. The confirm latency is the GEOMETRIC sum',
  '   of the backoff series for confirm_unreachable_after_retries fires (backoff_min, 2x, 4x, ... capped at backoff_max),',
  '   converted at the 20Hz tick rate. Get the off-by-one exactly right (how many backoff intervals precede the Nth bounce?',
  '   trace peer_writer: the timer arms at backoff after the first Down, then each failed replay grows backoff; the Nth bounce',
  '   lands after N backoff intervals summed). (b) unreachable_window_ticks >= the GEOMETRIC worst-case spacing of n consecutive',
  '   notices (NOT the linear retry_delay_ticks_hint) so a genuinely-dead peer stays re-confirmable (the L2 hole). Decide',
  '   whether to SUPERSEDE or KEEP the existing linear validate() check.',
  '2. Env-plumb the transport clock: today all bins call MeshConfig::new which hardcodes MeshReliabilityTuning::default(), so',
  '   validate_against has only ONE tunable side. Add VD_MESH_CONFIRM_RETRIES / VD_MESH_BACKOFF_MIN_MS / VD_MESH_BACKOFF_MAX_MS',
  '   (or a MeshConfig::from_env / a reliability override) across the bins, matching how the saga side (VD_LIVENESS_N /',
  '   VD_SAGA_ABORT_DEADLINE / ...) is already env-driven. Keep it DRY (one env-read home if the bins share it).',
  '3. Boot-assert validate_against LOUD in the orchestrator bin AFTER both tunings are read, the same fail-loud discipline',
  '   SagaTuning::validate + LivenessTuning::validate + MeshReliabilityTuning::validate already use. The orchestrator is the',
  '   ONE process that holds BOTH the mesh reliability tuning and the saga tuning, so it is the natural cross-validation point.',
  '',
  'CONSTRAINTS: vd-sim is Tier-A (HR5 100% region+branch — validate_against must be branchless-faithful: monomorphic helpers,',
  'each arm equality-asserted, no matches!, split assert a&&b). No magic numbers (all in the config structs). The geometric',
  'formula must be a PURE function (unit-testable without tokio). vd-sim must NOT depend on io-prod (MeshReliabilityTuning lives',
  'in io-prod) — so validate_against either takes the RAW numbers (confirm_retries:u32, backoff_min/max:Duration) as args, or',
  'the invariant lives where both are visible (the orchestrator bin / a shared tuning type). Resolve this crate-dependency',
  'question decisively (vd-sim -> io-prod is FORBIDDEN by the dependency rule bins->node->sim->wire->core). The M3',
  'Inbound::SendShed disambiguation is OUT OF SCOPE for R-4c (it is R-4d ops-slice work) — do NOT design it here.',
].join('\n')

const REVIEW_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['lens', 'verdict', 'blocking'],
  properties: {
    lens: { type: 'string' },
    verdict: { type: 'string', enum: ['SOUND_TO_IMPLEMENT', 'FIX_BEFORE_IMPL'] },
    blocking: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'problem', 'fix'],
        properties: {
          severity: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM'] },
          problem: { type: 'string' },
          fix: { type: 'string' },
        },
      },
    },
  },
}

const DESIGN = await (async () => {
  phase('Design')
  return agent(
    CONTEXT +
      '\n\nProduce the CONCRETE, implement-ready R-4c design. Give: the EXACT geometric-backoff-sum-in-ticks formula as Rust ' +
      '(with the traced off-by-one — how many backoff intervals precede the Nth confirm bounce, verified against the real ' +
      'peer_writer timer/backoff-growth code), the exact validate_against signature + which crate it lives in (resolving the ' +
      'vd-sim -/-> io-prod dependency constraint — does it take raw numbers or live in the orchestrator?), the window >= ' +
      'geometric-spacing check, whether to supersede or keep the existing linear check, the env-plumb (exact env var names + ' +
      'the DRY read home + the MeshConfig/reliability override path across all bins), the orchestrator boot-assert wiring, ' +
      'and the Tier-A unit test plan (each arm, the geometric-sum boundary, a fast-abort-slow-backoff rejection). Call out ' +
      'every decision that needs a call. Cite file:line against saga.rs / mesh.rs / orchestrator.rs.',
    { label: 'design', phase: 'Design', model: 'opus' }
  )
})()

const LENSES = [
  {
    key: 'formula-correctness',
    prompt:
      'Adversarially review this R-4c DESIGN for the GEOMETRIC-BACKOFF FORMULA correctness. Read the REAL io-prod mesh.rs ' +
      'peer_writer + the backoff growth (backoff = (backoff*2).min(max)) + confirm_and_maybe_bounce, and core/src/ids.rs for ' +
      'the tick rate. Verify the confirm-latency formula EXACTLY: trace how many backoff intervals precede the Nth bounce, the ' +
      'sum of the geometric series (including the cap at backoff_max), the Duration->ticks conversion (rounding direction — a ' +
      'conservative bound must ROUND UP the confirm latency and ROUND DOWN the abort budget, or it can pass a hair-too-tight ' +
      'config), and the off-by-one (N+n-1 vs N+n). Does the formula correctly bound the WORST case (backoff_max reached)? Does ' +
      'the window >= geometric-spacing check use the right spacing (the late-stage backoff_max, not the early min)? Could the ' +
      'invariant PASS a config that still aborts a live-but-slow player or strands a saga? VERDICT + blocking findings.\n\n---DESIGN---\n' +
      DESIGN,
  },
  {
    key: 'crate-deps-tiera',
    prompt:
      'Adversarially review this R-4c DESIGN for the CRATE-DEPENDENCY + TIER-A constraints. The dependency rule is ' +
      'bins->node->sim->wire->core; vd-sim must NOT depend on io-prod (where MeshReliabilityTuning lives). Verify the design ' +
      'does NOT introduce a forbidden vd-sim->io-prod edge (does validate_against take raw numbers, or live in the bin, or is ' +
      'there a shared tuning type in a lower crate?). Verify the geometric formula + validate_against is branchless-faithful ' +
      'for HR5 Tier-A 100% (monomorphic helpers, each arm equality-asserted, no matches!, split a&&b, no uncoverable false ' +
      'arm) — vd-sim is Tier-A 100%, so an untested branch FAILS the gate. Verify the Duration arithmetic has no overflow/panic ' +
      '(a huge backoff_max * 2, or a Duration->ticks cast). Verify the env-plumb keeps sim/node pure (the tuning is read in the ' +
      'bins, never in sim). VERDICT + blocking findings.\n\n---DESIGN---\n' +
      DESIGN,
  },
  {
    key: 'integration-boot',
    prompt:
      'Adversarially review this R-4c DESIGN for INTEGRATION + the boot cross-validation + DRY. Read orchestrator.rs (how it ' +
      'reads env + builds MeshConfig + SagaTuning/LivenessTuning + calls the existing validates), gateway.rs/shard.rs/client.rs ' +
      '(the other MeshConfig::new call sites), and bins/src/lib.rs (the shared env home). Verify: the boot-assert fires at the ' +
      'right point (after BOTH tunings are read, before spawn_mesh / before the saga runs) and fails LOUD like the existing ' +
      'validates; the env-plumb is DRY (a shared read, not duplicated across 4 bins) and every bin that spawns a mesh gets the ' +
      'override (or a documented reason it does not need it); the new env vars follow the existing VD_* naming + default ' +
      'convention; a non-orchestrator bin (gateway/shard) that has a mesh but NOT the saga tuning — does it validate_against ' +
      'anything, or is the cross-check orchestrator-only (is that correct — the saga lives only in the orchestrator)? Verify ' +
      'the design does not regress the existing process_parity / dev_cluster_smoke env contract. VERDICT + blocking findings.\n\n---DESIGN---\n' +
      DESIGN,
  },
]

phase('Review')
const reviews = (await parallel(
  LENSES.map((l) => () =>
    agent(l.prompt, { label: 'review:' + l.key, phase: 'Review', schema: REVIEW_SCHEMA, model: 'opus' })
  )
)).filter(Boolean)

const blocking = reviews.flatMap((r) => (r.blocking || []).map((b) => ({ ...b, lens: r.lens })))
log('reviews: ' + reviews.map((r) => r.lens + '=' + r.verdict).join(', ') + '; ' + blocking.length + ' blocking')

phase('Synthesize')
const vetted = await agent(
  'You are the synthesizer for the R-4c pre-implementation design review. Read the real code yourself to adjudicate. The DESIGN:\n\n' +
    DESIGN +
    '\n\nReviewer verdicts + blocking findings (JSON):\n' +
    JSON.stringify(reviews, null, 2) +
    '\n\nProduce the FINAL VETTED, IMPLEMENT-READY R-4c design with every real blocking finding FOLDED IN (drop false alarms ' +
    'with reasons). Give: the resolved geometric-backoff-in-ticks formula (with the exact off-by-one + rounding direction), ' +
    'the resolved validate_against signature + crate location (respecting vd-sim -/-> io-prod), the window check, the ' +
    'supersede-or-keep decision, the env-plumb (names + DRY home + all bin call sites), the orchestrator boot-assert wiring, ' +
    'and the Tier-A test plan. State clearly whether it is SOUND TO IMPLEMENT as specified. Cite file:line.',
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return {
  design_len: DESIGN.length,
  verdicts: reviews.map((r) => ({ lens: r.lens, verdict: r.verdict })),
  blocking_count: blocking.length,
  blocking,
  vetted,
}
