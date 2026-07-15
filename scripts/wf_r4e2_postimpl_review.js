export const meta = {
  name: 'r4e2-postimpl-review',
  description: 'Adversarial post-impl review of R-4e2 (H2 per-peer RecvLedger re-key)',
  phases: [{ title: 'Review' }, { title: 'Synthesize' }],
}

const ROOT = '/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system'

const CONTEXT = [
  'R-4e2 just landed (uncommitted) in the voxeldust redelivering MeshTransport: the H2 per-peer RecvLedger',
  're-key. Purpose: under N-peer reliable fan-in (hundreds-in-one-location), the receiver RX plane must not',
  'collapse on a single node-wide mutex. The receiver dedup ledger changed from',
  '  Arc<Mutex<BTreeMap<(NodeId,MsgClass), RecvState>>>   (one node-wide lock)',
  'to',
  '  Arc<RwLock<BTreeMap<NodeId, Arc<Mutex<BTreeMap<MsgClass, RecvState>>>>>>   (per-peer inner lock)',
  'so a frame from peer P no longer serializes against a frame from peer Q. std-only, NO new dependency.',
  '',
  'The design was pre-vetted (wf_e096eefb, an H2-correctness reviewer produced the findings folded here).',
  'THE FINDINGS FOLDED (verify each is faithfully implemented):',
  '  - The OUTER RwLock is shared-READ on the steady-state hot path (fetch the peer inner Arc); write-locked',
  '    ONLY on the rare first frame from a never-seen peer (insert the inner map).',
  '  - The INNER per-peer Mutex is HELD ACROSS classify_reliable + push_inbox + the `*st = before` rollback',
  '    (the atomicity invariant): two serve_data_stream tasks for the SAME (peer,class) genuinely overlap',
  '    across a redial (quinn close is async), so the StaleEpoch-before-hw verdict + the reliable-inbox-drop',
  '    rollback must be one critical section over that RecvState. Dropping the lock before push_inbox would',
  '    re-open the reverted-R-1 cross-stream corruption.',
  '  - ack_egress must NOT collapse to a single-peer Arc fetch: keep the per-(peer,class) lookup for any',
  '    NodeId set (acked_keys is frame.from-populated, a trust not a structural invariant).',
  '  - The or_insert INNER seed (incarnation:frame.incarnation, epoch:frame.epoch, hw:0, primed:false) is',
  '    byte-identical to the pre-re-key entry; only the OUTER or_insert_with (peer -> empty inner map) is new.',
  '  - Lock order (the only one, never reversed): outer-read -> (drop) -> inner-Mutex -> (optionally) inbox',
  '    Mutex. ack_egress takes the ledger, NEVER the inbox; acked_keys is acquired-then-released ABOVE (never',
  '    nested inside) an inner-Mutex hold. All locks use unwrap_or_else(PoisonError::into_inner).',
  '',
  'STATE: full io-prod suite green (lib 73, mesh_redelivery 7, mesh_load 1, spike0a_quinn 3); the N=64 fan-in',
  'load test green 3/3 after the re-key; clippy -D warnings + fmt clean; Tier-B mesh.rs 93.98% region (floor',
  '90). The frozen sim::io seam is UNTOUCHED.',
  '',
  'Read the code (all under ' + ROOT + '): crates/io-prod/src/mesh.rs — the RecvLedger type + its doc (grep',
  '"type RecvLedger"), classify_and_deliver (grep "fn classify_and_deliver"), ack_egress (grep "async fn',
  'ack_egress"), the ledger creation in spawn_mesh (grep "let ledger"), serve_data_stream + serve_connection',
  '(the callers + acked_keys). Also classify_reliable + RecvState + push_inbox for the invariants. Use',
  'git -C ' + ROOT + ' diff HEAD -- crates/io-prod/src/mesh.rs for the exact change.',
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
    key: 'concurrency-race-deadlock',
    prompt: [
      'LENS: concurrency correctness — races, deadlock, lock-order, poison. Attack the two-level lock as',
      'IMPLEMENTED: (1) the read-then-maybe-write outer sequence in classify_and_deliver — two threads both',
      'seeing None for a new peer then both write-locking: does or_insert_with make this a benign no-op (one',
      'inserts, the other finds it), or is there a lost inner map / double-init? (2) Is the inner lock',
      'genuinely held across classify + push_inbox + the *st=before rollback (one critical section)? Could the',
      'RwLock read guard still be held while taking the inner Mutex (a nested-lock hazard), or is it dropped',
      'first? (3) Lock order: is there ANY path that takes inner-then-outer, or acked_keys-inside-inner, or',
      'inner-then-inbox-then-something reversed → a deadlock cycle across classify_and_deliver / ack_egress /',
      'serve_data_stream / drop_connections? (4) Poison: outer RwLock read+write and inner Mutex all use',
      'into_inner — a poisoned inner peer lock should wedge only that peer, a poisoned outer is node-wide;',
      'confirm no unwrap() slipped in. (5) In ack_egress, is copying the keys then dropping acked_keys before',
      'the ledger read actually correct (no lost/stale ack that regresses cumulative-ack monotonicity)?',
    ].join('\n'),
  },
  {
    key: 'behavior-preservation',
    prompt: [
      'LENS: behavior preservation vs the single-lock version + the R-3 invariants. Does the re-key produce',
      'BYTE-IDENTICAL observable behavior to the pre-re-key node-wide ledger for every case: (1) the classify',
      'ladder (incarnation>epoch>contiguity) — the RecvState it operates on is seeded + mutated identically?',
      '(2) the reliable-inbox-drop rollback (*st=before) — still atomic against a concurrent same-(peer,class)',
      'frame, so a full-inbox reliable drop still rolls hw back and a later redelivery re-drives cleanly? (3)',
      'the StaleEpoch-before-hw straggler cure across a redial — two overlapping serve_data_stream tasks for',
      'one (peer,class) still serialize on the SAME inner lock (not two different locks)? Confirm the outer',
      'key is NodeId and the inner key is MsgClass so a given (peer,class) always resolves to exactly ONE',
      'RecvState (no accidental duplication of state across the two levels). (4) ack_egress: the incarnation',
      'echo + per-(peer,class) AckEntry set is unchanged for any NodeId set (multi-peer on one connection is',
      'impossible by the trust model, but the per-key loop must still be correct if it ever were). Any',
      'observable divergence is a finding.',
    ].join('\n'),
  },
]

phase('Review')
const reviews = await parallel(
  LENSES.map((l) => () =>
    agent(
      [CONTEXT, '', 'YOUR REVIEW LENS:', l.prompt, '',
       'Read the ACTUAL implemented code (the diff + the surrounding fns) before judging. Be adversarial and',
       'concrete (file:line). A finding names a REAL defect (a race, a deadlock, a behavior divergence, a',
       'coverage/HR gap) with a specific fix — not a style preference. If sound on your lens, say so plainly.',
       'Reserve CRITICAL for a correctness/liveness break (a real race, a deadlock, silent loss).'].join('\n'),
      { label: 'review:' + l.key, phase: 'Review', schema: SCHEMA, model: 'opus' }
    )
  )
)

phase('Synthesize')
const synth = await agent(
  [CONTEXT, '', 'Two adversarial reviewers returned verdicts on the R-4e2 H2 re-key:',
   JSON.stringify(reviews, null, 2), '',
   'Adjudicate against the ACTUAL code (read mesh.rs to confirm/refute each finding — concurrency findings',
   'are easy to misjudge in either direction). Produce: (a) the deduplicated, severity-ranked REAL findings',
   'that must change before commit, each with a concrete fix (CRITICAL/HIGH blocking; MEDIUM fix-now-if-cheap;',
   'LOW ledger); (b) an explicit ruling on whether the re-key is a faithful, race-free, behavior-preserving',
   'implementation of the vetted design; (c) a final verdict: SOUND_TO_COMMIT or FIX_BEFORE_COMMIT. Be',
   'decisive and cite file:line.'].join('\n'),
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return { reviews, synthesis: synth }
