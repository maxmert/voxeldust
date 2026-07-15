export const meta = {
  name: 'r6d3c-design',
  description: 'Design of record for R-6d3c: the never-restart D-6 #1 closure (saga AwaitAdopt split + dest TransientDiscard)',
  phases: [
    { title: 'Design', detail: 'reachable-now vs CA-1-gated scope + sub-slicing' },
    { title: 'Review', detail: '3 opus lenses: saga-FSM / frozen-wire-closed-set / dest-handler-idempotence-scope' },
    { title: 'Synthesize', detail: 'adjudicate then produce design of record' },
  ],
}

const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['design_markdown', 'sub_slices', 'reachable_now_vs_ca1', 'open_questions'],
  properties: {
    design_markdown: { type: 'string' },
    sub_slices: { type: 'array', items: { type: 'string' } },
    reachable_now_vs_ca1: { type: 'string', description: 'what lands now vs what is CA-1/L5-gated' },
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
          title: { type: 'string' }, claim: { type: 'string' }, evidence: { type: 'string' },
          fix: { type: 'string' }, confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
      },
    },
  },
}

const GROUND = [
  'R-6d3c is the NEVER-restart closure that RETIRES D-6 #1. R-6d3b-2b JUST LANDED (commit 2f140a4): the durable',
  'outbox is LIVE + closes the RESTART case (a crashed source that RESTARTS re-drives its lost TransientBatch /',
  'Ghost::Despawn via boot replay). The residual D-6 #1 case: a source that crashes in the saga BatchHandoff',
  'AwaitAdopt phase AND NEVER restarts strands an orphaned `Arriving` transient item on the DEST (uncounted,',
  'un-rendered, authoritative nowhere) — a silent loss. Read scripts/r6d_vetted_design.md (the ARC design, esp.',
  'CRITICAL-2 = the dest-side TransientDiscard GC, and CRITICAL-3 = the saga AwaitAdopt phase-split + the re-solicit',
  'egress) and docs/design/DEFERRED.md (the D-6 arc + the RC-3 /goal-audit input).',
  '',
  'THE TWO COUPLED PARTS (arc design):',
  '1. DEST (CRITICAL-2): a NEW `InterShardFlow::TransientDiscard(TransientHandoff{transfer, step_id:',
  '   TRANSIENT_DISCARD_STEP, fence})` arm + a `TRANSIENT_DISCARD_STEP` const + a dest `on_transient_discard`',
  '   handler that REMOVES any `TransientStatus::Arriving{batch==transfer}` item, journals `(transfer,',
  '   TRANSIENT_DISCARD_STEP)` AND POISONS `(transfer, TRANSIENT_BATCH_STEP)` in AppliedSteps [so a late replayed',
  '   `adopt_transient_batch` is `AlreadyApplied` and never re-inserts the item], idempotent, and BUCKETS the item',
  '   into the transients-lost-in-handover budget by kind. ⚠️ FROZEN WIRE: `InterShardFlow` is the closed egress',
  '   taxonomy (HR1) with a compiler-closed-set conformance gate + the nested `payload_tripwire`/`ghost_tripwire`',
  '   (intershard_closed.rs) + the durability_class classifier — a NEW arm MUST land through ALL of them (its',
  '   durability_class, the golden byte-freeze test, the tripwires) or it breaks the closed-set guarantee.',
  '2. SAGA (CRITICAL-3, Tier-A pure FSM): split the `BatchHandoff` `(S::BatchHandoff{..}, E::SourceUnreachable)`',
  '   phase-WILDCARD (saga.rs ~922). Post-adopt phases (AwaitRelease/AwaitPromote/AwaitComplete) keep the',
  '   self-promote (the dest provably holds the batch). AwaitAdopt (PRE-adopt) → a NEW resolution: an',
  '   `EmitTransientDiscard{dest, fence}` action + a distinct `CountBatchLostSourceCrash` (NOT the dest-addressed',
  '   `EmitTransientAbandon`, NOT `EmitTransientPromote`) + `Tombstone`, terminal-on-first-fire (no forever-bounce),',
  '   budget-gated on `abort_deadline_ticks` so a source that RESTARTS within budget delivers via its outbox replay',
  '   FIRST (the two recoveries race; budget picks the restart winner).',
  '',
  'THE CA-1/L5 GATING (decisive scope question — resolve it): saga.rs ~914-919 states the AwaitAdopt dead-resolution',
  'is "NOT reachable in the current in-proc/loopback posture — the confirm-dead trigger toward the source needs',
  'CA-1/L5 (unlanded)". `AwaitAdopt` is entered on `CasWon` with NO egress toward the source, so the orchestrator',
  'accrues no `NodeUnreachable{source}` ⇒ `is_confirmed_dead(source)` cannot fire ⇒ `E::SourceUnreachable` cannot',
  'reach the AwaitAdopt arm. So the FULL closure needs an AwaitAdopt RE-SOLICIT egress (an idempotent re-prompt',
  'toward the source so a dead source accrues NodeUnreachable) — which the arc design says is REQUIRED (not',
  'redundant). Determine: what is REACHABLE-NOW as a pure Tier-A FSM + dest-handler slice (the phase-split so a',
  'future confirm-dead does the RIGHT thing = discard-to-dest not self-promote-empty; the TransientDiscard wire arm',
  '+ dest handler, testable via a DIRECT on_transient_discard unit test + a saga FSM unit test of the split arm) vs',
  'what is genuinely CA-1/L5-BLOCKED (the confirm-dead TRIGGER + the re-solicit egress that MAKES the arm reachable',
  'in prod). Propose the R-6d3c scope + sub-slicing accordingly, and state honestly which parts leave D-6 #1',
  'NEVER-restart still-open-until-CA-1 vs which are the pure-logic closure landable now.',
  '',
  'KEY FILES: crates/sim/src/saga.rs (the BatchHandoff/AwaitAdopt/SourceUnreachable/DestUnreachable arms ~830-925,',
  'the SagaEvent/SagaAction enums, BatchHandoffPhase); crates/sim/src/saga_runtime.rs OR wherever rehome_event_for /',
  'is_confirmed_dead / IssueTransientGo live (grep — the dead-detection + the AwaitAdopt egress); crates/sim/src/stub.rs',
  '(the dest handlers: adopt_transient_batch, on_transient_promote, on_transient_abandon, TransientStatus::Arriving,',
  'the AppliedSteps journal, the transients-lost budget); crates/wire/src/intershard.rs (InterShardFlow enum +',
  'TransientBatch + TransientHandoff + TRANSIENT_BATCH_STEP + durability_class + the closed-set); crates/wire/tests/',
  'intershard_closed.rs (the conformance golden + tripwires). Tier-A crates are 100% region+branch (HR5) — the new',
  'saga arm + dest handler + wire arm must be fully covered. Be code+design-grounded; cite file:line. Distinguish a',
  'REAL will-break from a not-yet-built roadmap item.',
].join('\n')

phase('Design')
const design = await agent(
  GROUND + '\n\nYou are the DESIGNER. Produce an IMPLEMENT-READY design of record for R-6d3c + PROPOSE the sub-slice ' +
  'ordering, and DECISIVELY resolve the reachable-now-vs-CA-1-gated scope. Deliver, all code-grounded (cite ' +
  'file:line): (1) the dest side — the exact new InterShardFlow::TransientDiscard arm + TRANSIENT_DISCARD_STEP + how ' +
  'it lands through the frozen-wire closed-set (durability_class arm, the golden byte-freeze, the nested tripwires) ' +
  'WITHOUT breaking HR1; the on_transient_discard handler (remove Arriving{batch}, journal + poison AppliedSteps, ' +
  'idempotent, count into the loss budget); (2) the saga side — the exact BatchHandoff phase-split (the new ' +
  'SagaEvent/SagaAction, the AwaitAdopt-vs-post-adopt arms, terminal-on-first-fire, budget-gated), as assert_eq!-' +
  'testable (SagaState,Vec<SagaAction>) pairs; (3) the reachable-now scope vs the CA-1/L5-gated re-solicit-egress + ' +
  'confirm-dead-trigger — state exactly what closes D-6 #1 NEVER-restart NOW vs what stays open-until-CA-1, and how ' +
  'the reachable-now part is TESTABLE (direct handler + FSM unit tests, no CA-1); (4) the tests + Tier-A 100% ' +
  'coverage plan; (5) the precise files/functions touched per sub-slice. Flag any API insufficiency. Be decisive.',
  { label: 'design', phase: 'Design', schema: DESIGN_SCHEMA, model: 'opus' }
)

phase('Review')
const LENSES = [
  { key: 'saga-fsm', focus:
    'SAGA FSM CORRECTNESS. Verify the BatchHandoff phase-split against saga.rs: (a) post-adopt phases ' +
    '(AwaitRelease/AwaitPromote/AwaitComplete) keep the CORRECT self-promote (the dest holds the batch); AwaitAdopt ' +
    'goes to the NEW discard-to-dest + CountBatchLostSourceCrash + Tombstone (NEVER self-promote-empty, NEVER the ' +
    'source-addressed EmitTransientAbandon). (b) terminal-on-first-fire (straight to Done, no BatchHandoff re-entry ' +
    '⇒ no D-37 forever-bounce). (c) budget-gated on abort_deadline_ticks so a RESTART within budget wins via outbox ' +
    'replay BEFORE the discard fires (the two recoveries race — is the budget ordering correct + is a double ' +
    'resolution [restart delivers AND discard fires] prevented / made idempotent?). (d) the new SagaEvent/SagaAction ' +
    'are exhaustively matched (no new wildcard). (e) is DestUnreachable still correct? Cite file:line.' },
  { key: 'frozen-wire-closed-set', focus:
    'FROZEN WIRE / CLOSED-SET / HR1. Verify the new InterShardFlow::TransientDiscard arm lands WITHOUT breaking the ' +
    'closed egress guarantee: (a) it is added to durability_class() (which class? it is a producer-less orchestrator→' +
    'dest one-shot — ReDriven? ProducerLessReliable? — the wrong class = silent-loss risk, exactly the R-6d2b nested-' +
    'tripwire concern); (b) the golden byte-freeze / every_arm conformance test + the nested payload_tripwire/' +
    'ghost_tripwire are updated so the new arm + any nested variant fails compilation until classified (no vacuous ' +
    'pass); (c) TRANSIENT_DISCARD_STEP is a distinct step id (not colliding with TRANSIENT_BATCH_STEP/DROP/RELEASE); ' +
    '(d) HR1: the arm carries only (transfer, step_id, fence) — no World leak; the EffectClass/effect-carrying ' +
    'conformance (side-effecting arms carry (TransferId, step_id) + acks) holds. Cite file:line.' },
  { key: 'dest-handler-scope', focus:
    'DEST HANDLER IDEMPOTENCE + POISON + SCOPE. Verify on_transient_discard: (a) removes EXACTLY the Arriving{batch==' +
    'transfer} items (not Held/Departing, not other batches); idempotent (a re-delivered discard is a no-op via the ' +
    '(transfer, TRANSIENT_DISCARD_STEP) journal). (b) POISONS (transfer, TRANSIENT_BATCH_STEP) so a LATE replayed ' +
    'adopt_transient_batch (arriving AFTER the discard — the exact restart-races-discard interleave) is AlreadyApplied ' +
    'and never re-inserts the item (else the item resurrects). (c) the loss is counted into the SAME transients-lost ' +
    'budget by kind (no silent loss; the TRANSIENT-CONSERVATION oracle sees it). (d) SCOPE HONESTY: is the reachable-' +
    'now slice genuinely a pure-logic closure (the arm + handler + the phase-split, testable direct) with the ' +
    'confirm-dead TRIGGER honestly CA-1-gated, or does the design over-claim D-6 #1 fully-retired-now? (e) Tier-A ' +
    '100% coverage of the new handler/arm/split. Cite file:line.' },
]
const reviews = await parallel(
  LENSES.map((l) => () =>
    agent(
      GROUND + '\n\nThe DESIGNER proposed this R-6d3c design:\n--- DESIGN ---\n' + design.design_markdown +
      '\n--- SUB-SLICES ---\n' + design.sub_slices.join('\n') +
      '\n--- REACHABLE-NOW vs CA-1 ---\n' + design.reachable_now_vs_ca1 +
      '\n--- OPEN QUESTIONS ---\n' + design.open_questions.join('\n') + '\n--- END ---\n\n' +
      'You are REVIEWER (' + l.key + '). Adversarially review through YOUR lens: ' + l.focus + '\n' +
      'Code-grounded (cite file:line), decisive. SOUND_TO_IMPLEMENT with empty/NIT array if clean; do NOT invent.',
      { label: 'review:' + l.key, phase: 'Review', schema: REVIEW_SCHEMA, model: 'opus' }
    )
  )
)
const valid = reviews.filter(Boolean)

phase('Synthesize')
const synth = await agent(
  GROUND + '\n\nThe DESIGNER proposed an R-6d3c design; three adversarial lenses reviewed it.\n\nDESIGN:\n' +
  design.design_markdown + '\n\nSUB-SLICES:\n' + design.sub_slices.join('\n') + '\n\nREACHABLE-NOW vs CA-1:\n' +
  design.reachable_now_vs_ca1 + '\n\nREVIEWS (JSON):\n' +
  JSON.stringify(valid.map((r) => ({ lens: r.lens, verdict: r.verdict, summary: r.summary, findings: r.findings })), null, 2) +
  '\n\nSynthesize the ADJUDICATED, IMPLEMENT-READY design of record. Deduplicate (same root cause = ONE). CONFIRM or ' +
  'REFUTE each finding against the code, rank by true impact, FOLD the fix into the final design. Down-grade any ' +
  'reviewer CRITICAL the code/design already handles (cite where); up-grade any real silent-loss / closed-set-breach ' +
  '/ resurrect-after-discard / double-resolution hazard. Output the COMPLETE corrected design (the wire arm + ' +
  'durability_class + tripwires, the saga phase-split arms, the dest handler + poison, the reachable-now-vs-CA-1 ' +
  'scope, the sub-slice ordering, the Tier-A tests, the files touched) so it can be implemented directly — plus a ' +
  'one-line VERDICT: SOUND_TO_IMPLEMENT (with folded must-fixes) | REVISE | RECONSIDER. Decisive; reward sound design.',
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return { verdict: synth, design_markdown: design.design_markdown, sub_slices: design.sub_slices,
  reachable_now_vs_ca1: design.reachable_now_vs_ca1,
  reviews: valid.map((r) => ({ lens: r.lens, verdict: r.verdict, summary: r.summary, findings: r.findings })) }
