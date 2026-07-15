export const meta = {
  name: 'ca1-s3s4-review',
  description: 'Adversarial post-impl review of CA-1 S3/S4 (the achievable hard over-discard cure)',
  phases: [
    { title: 'Review', detail: 'opus reviewers per dimension, read-only' },
    { title: 'Adjudicate', detail: 'synthesize verdict + must-fix list' },
  ],
}

// The uncommitted change under review (all in the new-system worktree). Reviewers `git diff` to see it.
const SCOPE = `
CA-1 S3/S4 — the achievable HARD over-discard guarantee. The change (all UNCOMMITTED; run \`git diff\`):
- crates/wire/src/intershard.rs: NEW arm InterShardFlow::ReSolicitBatch(TransientHandoff) APPENDED (discriminant
  16 -> 17-arm set); RE_SOLICIT_STEP=18; joined the TransientHandoff effect_class group (SideEffecting{TransferStep})
  + the ReDriven durability_class group.
- crates/wire/src/lib.rs + crates/wire/tests/intershard_closed.rs: header count 16->17; every_arm + arm_tripwire updated.
- crates/sim/src/saga.rs: NEW SagaAction::EmitReSolicit{fence}; NEW FSM arm (BatchHandoff{AwaitAdopt}, Timeout) =>
  EmitReSolicit (an ADD before the terminal catch-all); the R-6d4-F2 tripwire prose at the SourceUnreachablePreAdopt
  arm retired (guard now wired).
- crates/node/src/saga_runtime.rs: NEW dest_adopted:bool on LiveSaga (RAM-only, re-derived false on rehydrate); NEW
  pre-scan pass latch_adopted_from_inbox called in drive_sagas BEFORE scan_deadlines; EmitReSolicit executor (push
  ReSolicitBatch to ctx.source); rehome_event_for's AwaitAdopt arm gated: SourceUnreachablePreAdopt fires ONLY when
  !dest_adopted (else Timeout).
- crates/sim/src/stub.rs: NEW redrive_pending_adoptions DEST system (re-emit BatchAdopted per DISTINCT Arriving batch,
  gated purely on Arriving-presence) added to register_stub_shard chain after process_inbound; NEW ReSolicitBatch
  counted-no-op handler; StubStats counters batch_adopts_redriven + re_solicits_received.
- docs/design/DEFERRED.md: the over-discard residual marked CLOSED; the sustained-shed + boot-assertion + k3d items ledgered.

THE CURE DECOMPOSES: (a) SAFETY = the dest_adopted latch gating the AwaitAdopt discard, set in the PRE-SCAN pass (the
intra-tick ordering: scan_deadlines runs BEFORE the inbox drain, so latching in deliver would be one tick too late on
the exact budget-maturity tick); (b) LIVENESS = the dest re-drives BatchAdopted every tick it holds Arriving; (c)
DETECTION = the ReSolicitBatch probe makes is_confirmed_dead(source) reachable. ACKNOWLEDGED residual (ledgered): a
sustained dest->orch shed of EVERY BatchAdopted across the whole abort window still over-discards (needs a non-sheddable
ack — a transport change). Spec: scripts/ca1_s3s4_perfect_guarantee_spec.md.

Repo: 6 HARD RULES (HR1 sealed closed InterShardFlow taxonomy; HR2 generic transfer; HR3 one machinery / never match a
shard kind in feature code; HR5 100% region+branch on Tier-A crates sim/wire/node, real-not-theater; postcard v1
APPENDED-discriminant discipline). End-goal: signal-heavy voxel MMO, hundreds of users, many transient batches
(bullets/debris) crossing shards. You are READ-ONLY: do NOT edit any file.`

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['dimension', 'verdict', 'findings'],
  properties: {
    dimension: { type: 'string' },
    verdict: { type: 'string', enum: ['clean', 'concerns', 'blocking'] },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'title', 'where', 'detail', 'fix'],
        properties: {
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          title: { type: 'string' },
          where: { type: 'string', description: 'file:line or symbol' },
          detail: { type: 'string', description: 'the concrete problem + a scenario that triggers it' },
          fix: { type: 'string', description: 'the specific remedy, or "ledger" if a non-blocking follow-up' },
        },
      },
    },
  },
}

const DIMENSIONS = [
  {
    key: 'intra-tick-correctness',
    prompt: `Review the CA-1 S3/S4 over-discard CORRECTNESS PROOF adversarially. ${SCOPE}

Your job: try to BREAK the zero-residual claim (beyond the acknowledged sustained-shed residual) AND try to find a
case where a LEGITIMATE discard is WRONGLY suppressed (a saga WEDGES forever). Read crates/node/src/saga_runtime.rs
(drive_sagas ordering, latch_adopted_from_inbox, scan_deadlines, rehome_event_for, dead_budget_elapsed) and
crates/sim/src/saga.rs (the BatchHandoff FSM arms). Probe specifically:
1. Intra-tick ordering: latch_adopted_from_inbox runs BEFORE scan_deadlines, the drain AFTER. Is the exact
   budget-maturity-tick race truly closed? Any tick where a BatchAdopted is deliverable but the discard still fires?
2. dest_adopted monotonicity: is it EVER set false during live operation? Rehydrate re-derives false + empty liveness
   tracker — is the re-drive guaranteed to re-latch before is_confirmed_dead(source) can fire post-restart?
3. THE WEDGE PROBE: dest adopts (dest_adopted latched true) -> dest CRASHES losing its RAM Arriving item -> source
   ALSO confirmed dead. rehome_event_for checks is_confirmed_dead(source) FIRST -> source-branch -> guard
   dest_adopted=true -> Timeout FOREVER (the dest is also dead but its branch is never reached). Does this WEDGE the
   saga? Is it a REGRESSION vs the pre-CA-1-S3 behavior (where AwaitAdopt source-death was unreachable ⇒ also parked)?
   Is it admin-visible? Should the guard also gate on !is_confirmed_dead(dest), or is ledgering the transient
   double-crash acceptable within the declared loss budget?
4. Does the ReSolicit probe re-drive forever toward a dead source (harmless) and never wedge a LIVE-slow source?
Return the verdict + findings.`,
  },
  {
    key: 'wire-seam-hr1-hr3',
    prompt: `Review the CA-1 S3/S4 WIRE SEAM + HR1/HR3 compliance. ${SCOPE}

Read crates/wire/src/intershard.rs + crates/wire/tests/intershard_closed.rs + crates/wire/src/lib.rs. Verify:
1. ReSolicitBatch is APPENDED (postcard discriminant order preserved — no existing arm's discriminant shifted).
2. effect_class: SideEffecting{TransferStep} is correct (it carries transfer+step; the conformance asserts transfer!=0).
   Is SideEffecting right for an ack-FREE probe whose signal is the SEND outcome, or should it be FireAndForget? Argue.
3. durability_class: ReDriven is correct (orch-emitted, scan re-drives it; NOT producer-less — the golden pin's
   producer-less set must stay exactly {Ghost::Despawn, TransientBatch}).
4. Conformance: every_arm + arm_tripwire + the 16->17 header all consistent; no hard-coded count assertion missed.
5. HR3: does anything match on a shard-kind discriminant? HR1: does the new arm keep the taxonomy closed (no escape)?
Return the verdict + findings.`,
  },
  {
    key: 'coverage-hr5-regression',
    prompt: `Review CA-1 S3/S4 for HR5 100% region+branch coverage completeness + REGRESSION to the transient handoff. ${SCOPE}

Read the new code + its tests in crates/sim/src/stub.rs (redrive_pending_adoptions, ReSolicitBatch handler, the two new
StubStats counters, the 3 new tests), crates/sim/src/saga.rs (the EmitReSolicit FSM arm + its test), and
crates/node/src/saga_runtime.rs (latch_adopted_from_inbox, the dest_adopted gate, the executor arm, the new tests).
Verify:
1. Every NEW branch has a test exercising BOTH arms (the latch's 4 branches; the rehome guard true+false; redrive
   Arriving-true/Held-false + empty/non-empty loop; the FSM arm). Name any branch with an uncoverable or untested arm.
2. REGRESSION: redrive_pending_adoptions now emits an extra BatchAdopted on the adopt tick + every Arriving tick. Does
   this violate any harness monitor invariant (crates/harness/src/oracle.rs — TRANSIENT-CONSERVATION: "a transient
   never renders on two shards in one tick; duplication is always failure")? A DUPLICATE ack is not a duplicate ENTITY
   — confirm the oracle keys on held-set membership, not ack counts. Any test elsewhere that counts total Saga egress?
3. Does the pre-scan double-decode of the inbox (latch pass + drain) risk any correctness issue (e.g. a message
   consumed twice with a side effect)? It should be side-effect-free in the latch pass.
Return the verdict + findings.`,
  },
  {
    key: 'scale-endgoal',
    prompt: `Review CA-1 S3/S4 for SCALE + end-goal alignment. ${SCOPE}

The end-goal has hundreds of users + many transient batches (bullets/debris) crossing shards, signal-heavy. Read
redrive_pending_adoptions (crates/sim/src/stub.rs) + the drive_sagas latch pass (crates/node/src/saga_runtime.rs).
Verify:
1. redrive_pending_adoptions re-emits ONE ack per DISTINCT batch (not per item) — confirm a 1000-bullet batch is one
   re-drive/tick, not 1000. Is the per-tick cost bounded acceptably when many batches are simultaneously Arriving?
2. latch_adopted_from_inbox decodes every Saga-class inbound each tick (a second decode after the drain). Is that a
   scale concern at hundreds of concurrent sagas / high inbox volume? Is a cheaper structure warranted, or fine?
3. Does the per-tick re-drive traffic during the NORMAL (brief) handoff window compose acceptably, or should it be
   interval-gated? Weigh robustness vs traffic.
4. Any drift from the end-goal (does this paint into a corner for multi-shard meshes / ships / signals)?
Return the verdict + findings.`,
  },
]

phase('Review')
const reviews = await parallel(
  DIMENSIONS.map((d) => () =>
    agent(d.prompt, { label: `review:${d.key}`, phase: 'Review', schema: FINDINGS_SCHEMA, agentType: 'Explore', model: 'opus' })
  )
)

phase('Adjudicate')
const packed = reviews.filter(Boolean).map((r) => JSON.stringify(r)).join('\n\n')
const verdict = await agent(
  `You are the adjudicator for the CA-1 S3/S4 post-impl review. ${SCOPE}

Here are the ${reviews.filter(Boolean).length} dimension reviews (JSON):

${packed}

Adjudicate. For each finding, decide if it is a REAL blocker (must fix before commit), a real-but-ledgerable follow-up,
or a false alarm — and say why. Pay special attention to the WEDGE PROBE (dest adopted-then-crashed + source dead): is
it a regression, is it acceptable within the transient declared-loss budget + admin-visibility, or must the guard also
gate on !is_confirmed_dead(dest) now? Give a final verdict: SOUND_TO_COMMIT (with any ledger items) or
NEEDS_FIX (with the exact must-fix list). Be concrete and cite file:line. Do NOT edit any file.`,
  { label: 'adjudicate', phase: 'Adjudicate', agentType: 'Explore', model: 'opus' }
)

return { reviews: reviews.filter(Boolean), verdict }
