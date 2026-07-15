export const meta = {
  name: 'r4e4-postimpl-review',
  description: 'Adversarial post-impl review of R-4e4 (R-5 producer-less capstone + D-6 #1 partial flip)',
  phases: [{ title: 'Review' }, { title: 'Synthesize' }],
}

const ROOT = '/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system'

const CONTEXT = [
  'R-4e4 just landed (uncommitted) in the voxeldust redelivering-transport arc: the R-5 producer-less-flow',
  'CAPSTONE + the D-6 #1 PARTIAL flip. This makes a load-bearing STATUS CLAIM about the transfer-first',
  'thesis (D-6 #1 = the redelivering transport is the root cure for producer-less-flow loss), so its',
  'honesty matters.',
  '',
  'WHAT LANDED:',
  '  - crates/io-prod/tests/mesh_under_loss.rs (NEW): drives a REAL InterShardFlow::Ghost(GhostFlow::Despawn)',
  '    envelope (a band-exit ghost teardown — a PRODUCER-LESS reliable one-shot: it has NO saga and NO',
  '    scan_deadlines re-driver, unlike a transfer) on MsgClass::GhostReliable. Sequence: send a spawn marker',
  '    (establishes the GhostReliable lane, delivered), drop_connections (blip), sleep 150ms, send the LONE',
  '    Despawn, then IDLE. Asserts B receives the Despawn EXACTLY ONCE (delivered purely by the R-4a',
  '    retransmit timer off the sender clock — no application re-drive), the envelope round-trips byte-exact,',
  '    gap_drop==0, ZERO NodeUnreachable (a blip is not a death), the endpoint survives, and (completion-',
  '    gated) reliable_acked>=2 (both frames retired — the full at-least-once cycle). Mirrors the proven',
  '    src `an_idle_after_blip_lone_frame_is_re_driven_by_the_timer` (mesh_redelivery.rs) but with the',
  '    GhostReliable class + the real producer-less envelope. Stable 5/5 in the full concurrent io-prod suite',
  '    (a first cut flaked on a bare reliable_acked sample racing the in-flight ack; now completion-gated).',
  '  - docs/design/DEFERRED.md: the D-6 #1 PARTIAL flip. The transport-layer redelivery (R-3 RX ledger +',
  '    per-lane replay + R-4a idle timer + R-4b bounded retry + R-4d SendShed) is flipped to GREEN "proven by',
  '    R-5 for a producer-less flow whose SOURCE STAYS UP" (subsuming GhostReliable Despawn, the durable',
  '    EmitCrossing, the D-37 re-home). The MeshTransport is declared NO LONGER "at-most-once" for the',
  '    source-up path. The remaining #1 residual is NARROWED to the AwaitAdopt SOURCE-CRASH case: if the',
  '    source crashes before the dest adopts, the transport (RAM retry buffer) does not cover it — BUT the',
  '    saga HAS a resolution ((BatchHandoff, SourceUnreachable) self-promotes the dest, saga.rs:915-922, via',
  '    rehome_event_for once is_confirmed_dead(ctx.source)), gated on M3 (durable incarnation) + L5 (addr',
  '    reread) — both already-tracked HARD deploy preconditions. The D-6 header + the #1 precondition entry',
  '    were both updated.',
  '  - justfile: `just mesh-load` now also runs mesh_under_loss.',
  '',
  'This design was pre-vetted (the R-4e design review wf_e096eefb; its R-5-scope reviewer raised the exact',
  'over-claim risk — that AwaitAdopt source-crash IS recovered by the saga self-promote, so the flip must not',
  'say "no recovery" — and that correction is folded into the partial-flip wording above).',
  '',
  'Read: crates/io-prod/tests/mesh_under_loss.rs (the whole test); crates/io-prod/src/mesh.rs (the R-4a timer',
  'peer_writer/replay_lanes/confirm_and_maybe_bounce — confirm the Despawn can ONLY be delivered by the timer',
  'here, no other path); crates/io-prod/tests/mesh_redelivery.rs (the idle-after-blip test it mirrors);',
  'crates/sim/src/saga.rs (BatchHandoff::AwaitAdopt ~815-823 + the SourceUnreachable self-promote ~915-922);',
  'crates/node/src/saga_runtime.rs (rehome_event_for ~1270); docs/design/DEFERRED.md (the D-6 header + the',
  'partial-flip body + the #1 residual). Use git -C ' + ROOT + ' diff HEAD.',
].join('\n')

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

const LENSES = [
  {
    key: 'test-rigor-not-false-pass',
    prompt: [
      'LENS: does mesh_under_loss.rs GENUINELY prove producer-less at-least-once, or is it a false-pass?',
      'Verify against mesh.rs: after the drop_connections blip + the 150ms sleep + the LONE Despawn send +',
      'IDLE, is the R-4a retransmit TIMER truly the ONLY path that can deliver the Despawn (no follow-up send',
      'dial-block re-drive, no other producer)? Confirm the spawn-marker establish genuinely sets up a live',
      'connection that drop_connections actually blips (not a fresh dial). Is "exactly once" real (could the',
      'Despawn be delivered twice and the test miss it — the count filter looks right, but check the drain',
      'accumulates correctly)? Is the envelope round-trip assertion meaningful (it decodes despawn_bytes, the',
      'SENT bytes — does it actually prove B RECEIVED that envelope, or just that the local bytes decode)?',
      'That last one is the sharpest: assert_eq!(decoded_from_despawn_bytes, despawn) decodes the LOCAL sent',
      'bytes, not what B received — is that a tautology, and if so does the got.contains(&despawn_bytes) gate',
      'already prove B received the exact bytes (making it OK) or should it decode B\'s received copy? Is the',
      'completion-gated reliable_acked>=2 fix correct + non-flaky? Any residual flake (the NodeUnreachable',
      'single-drain check, port reservation races)?',
    ].join('\n'),
  },
  {
    key: 'claim-scope-not-overclaim',
    prompt: [
      'LENS: is the D-6 #1 PARTIAL flip HONEST, or does it over-claim? Read the DEFERRED.md diff + saga.rs +',
      'saga_runtime.rs. (1) Does R-5 actually prove what the flip says it proves — transport-layer at-least-',
      'once for a producer-less flow whose SOURCE STAYS UP? Is "source stays up" the correct + only scope (the',
      'test never crashes A — good, but does the flip claim more)? (2) Is the residual correctly scoped: the',
      'AwaitAdopt SOURCE-CRASH case, recovered by the saga self-promote ((BatchHandoff, SourceUnreachable) at',
      'saga.rs:915-922, triggered by rehome_event_for on is_confirmed_dead) gated on M3+L5 — is that',
      'code-accurate (verify the self-promote arm + its trigger actually exist + are gated as claimed)? (3)',
      'Does flipping the transport-layer to GREEN while the AwaitAdopt egress, WAL version, durable-outbox,',
      'durable-root allow-list all remain OWED create any FALSE impression that D-6 #1 as a whole is done? The',
      'D-6 header still shows 🟧 and lists the residuals — confirm the header + body are CONSISTENT and a',
      'reader cannot conclude "the transport cures everything". (4) Are the subsumed flows (EmitCrossing,',
      'D-37 re-home) genuinely covered by the same source-stays-up timer proof, or is that an over-reach?',
    ].join('\n'),
  },
]

phase('Review')
const reviews = await parallel(
  LENSES.map((l) => () =>
    agent(
      [CONTEXT, '', 'YOUR REVIEW LENS:', l.prompt, '',
       'Read the ACTUAL code + the DEFERRED.md diff before judging. Be adversarial and concrete (file:line). A',
       'finding names a REAL defect (a false-pass test, an over-claim, a residual flake, a code-inaccurate',
       'residual scoping) with a specific fix. If sound on your lens, say so. Reserve CRITICAL for a false',
       'proof or a materially misleading status claim.'].join('\n'),
      { label: 'review:' + l.key, phase: 'Review', schema: SCHEMA, model: 'opus' }
    )
  )
)

phase('Synthesize')
const synth = await agent(
  [CONTEXT, '', 'Two adversarial reviewers returned verdicts on R-4e4:', JSON.stringify(reviews, null, 2), '',
   'Adjudicate against the ACTUAL code + the DEFERRED.md diff (read them to confirm/refute). Produce: (a) the',
   'deduplicated, severity-ranked REAL findings that must change before commit, each with a concrete fix',
   '(CRITICAL/HIGH blocking; MEDIUM fix-now-if-cheap; LOW ledger); (b) an explicit ruling on whether the R-5',
   'test is a genuine (non-false-pass) proof AND whether the D-6 #1 partial flip is honestly scoped (not',
   'over-claiming); (c) a final verdict: SOUND_TO_COMMIT or FIX_BEFORE_COMMIT. Be decisive, cite file:line.'].join('\n'),
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return { reviews, synthesis: synth }
