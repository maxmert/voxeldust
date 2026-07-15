export const meta = {
  name: 'r4e-design',
  description: 'Design + adversarial review of R-4e (N-peer real-QUIC load + H2 RecvLedger re-key + R-5 capstone)',
  phases: [{ title: 'Design' }, { title: 'Review' }, { title: 'Synthesize' }],
}

const ROOT = '/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system'

const CONTEXT = [
  'R-4e is the next slice of the D-6 #1 redelivering-transport hardening in the voxeldust greenfield rebuild',
  '(a transfer-first "Star Citizen meets Minecraft" voxel MMO). The post-R-4d holistic /goal audit',
  '(wf_098fdb52) confirmed ONE HIGH: the redelivering MeshTransport is proven correct only at 2 nodes / 1',
  'MsgClass / 1 lane (mesh_redelivery.rs, A=NodeId(1)/B=NodeId(2), MsgClass::Saga only) and client_load caps',
  'at K=4 — nothing exercises fan-out, per-peer lane isolation under contention, reliable_acked keeping pace,',
  'gap_drop==0 under sustained load, or the RX-plane collapse at the hundreds-in-one-location scale the',
  'end-goal demands (hundreds of players in ONE location, PvP, multisharding meshes). A soak/deploy would be',
  'the transport first meeting N-peer sustained load. R-4e must close this with a REAL-QUIC load/performance',
  'test + the concurrency fix it surfaces, and co-locate the R-5 capstone.',
  '',
  'THE KNOWN CONCURRENCY GAP (H2, pre-identified, cure sketched in DEFERRED.md ~1186-1192): the receiver',
  'RecvLedger is ONE node-wide Arc<Mutex<BTreeMap<(NodeId,MsgClass),RecvState>>> (crates/io-prod/src/mesh.rs',
  '~315). Every reliable RX from EVERY peer serializes on this single mutex (classify_reliable + the',
  'inbox-drop rollback hold it across the sync push_inbox). Under N-peer fan-in this is a serialization',
  'bottleneck / the RX-plane collapse. The sketched cure: key the OUTER map per-peer so each peer has its own',
  'lock, and drop the lock BEFORE push_inbox where possible. This must be done WITHOUT breaking the R-3',
  'classify ladder (incarnation > epoch > contiguity), the inbox-drop rollback (RecvState is Copy, *st=before',
  'on a reliable drop), or the ack_egress cumulative-ack snapshot (which reads the ledger under one lock).',
  '',
  'DEPENDENCY DISCIPLINE (BINDING — user-mandated): the transport is HAND-ROLLED, NO new dependency. Do NOT',
  'unilaterally adopt a concurrency crate (dashmap, flurry, etc.). If the H2 re-key genuinely needs one,',
  'FLAG it as a joint-investigation decision for the user — default to std primitives (a sharded map of',
  'per-peer Mutex, or per-connection-task-owned state) that need no new dep.',
  '',
  'THE R-4e WORK ITEMS the audit + DEFERRED.md name (the design must scope + sub-slice these):',
  '  (1) a real-QUIC N-peer (16-64 nodes) load test: sustained reliable into ONE receiver, assert no-loss /',
  '      no-dup + reliable_acked keeps pace + gap_drop==0 + a wall-clock latency/throughput bound. MUST be',
  '      DETERMINISTIC / non-flaky (a load test that flakes in CI is worse than none) and feasible on a dev',
  '      host (fd/port/tokio-task limits at 64 nodes).',
  '  (2) the H2 per-peer RecvLedger re-key (surfaced + measured by (1)), no new dep.',
  '  (3) a CORRELATED multi-peer outage (drop >=50% of peers simultaneously) asserting inbound_dropped_',
  '      reliable ~= 0 for surviving-peer traffic (else confirm-dead floods the BoundedInbox and evicts',
  '      genuine reliable inbound — the audit flagged coalescing confirm-dead to one notice/peer/window OR a',
  '      NodeUnreachable priority lane as the alternative cure).',
  '  (4) a pod-reschedule / addr-change scenario (the L5 concern: peer_writer captures the peer addr ONCE at',
  '      spawn; a rescheduled pod at a new addr in the same NodeId slot may never be re-dialed).',
  '  (5) the replay_lanes conn_died-mid-pass coverage (the R-4a ledgered Tier-B LOW: dial-ok-then-write-fails).',
  '  (6) R-5 (mesh_under_loss.rs): the producer-less-flow CAPSTONE that flips DEFERRED.md D-6 #1 to green —',
  '      a REAL producer-less reliable flow (e.g. a GhostReliable Despawn, or an AwaitAdopt-class one-shot)',
  '      driven across a connection blip, proving at-least-once WITHOUT an application re-drive. NOTE the',
  '      residual: BatchHandoff::AwaitAdopt is still a producer-less saga phase with NO orchestrator re-drive',
  '      egress (DEFERRED.md D-6 precondition #1 1a) — the design must state precisely what R-5 CAN prove',
  '      now vs what remains gated (does flipping D-6 #1 require the AwaitAdopt egress cure first, or does',
  '      R-5 prove the transport-layer at-least-once independent of that saga-layer gap?).',
  '  (7) a `just mesh-load` recipe as a NAMED blocking soak/deploy precondition (NOT in the default fast gate',
  '      — load tests are slow; document it like the existing GPU/release-build gate steps).',
  '',
  'STANDARDS: HR5 (Tier-A 100% region+branch; io-prod is Tier-B floor 90 — a load test lives in',
  'io-prod/tests, coverage-ignored, but any H2 re-key code in mesh.rs must keep Tier-B >= 90). No magic',
  'numbers (node count / duration / bounds in ONE config or clearly-justified test consts). The frozen',
  'sim::io seam must NOT change (R-4e is below the seam). Determinism: the control-plane assertions must be',
  'exact; only the wall-clock throughput/latency bound may be a generous ceiling, and even that must not',
  'flake under CI load (prefer asserting PROGRESS/completion within a deadline over a tight latency number).',
  '',
  'Read before designing: crates/io-prod/src/mesh.rs (the whole R-3..R-4d mesh — RecvLedger, classify_',
  'reliable, classify_and_deliver, ack_egress, peer_writer, replay_lanes, ensure_connection, MeshStats,',
  'MeshControl); crates/io-prod/tests/mesh_redelivery.rs (the current 2-node tests + helpers to extend);',
  'crates/bins/tests/ (client_load, process_parity, dev_cluster_smoke — the existing load/process tier);',
  'justfile (the gate recipes + client-load); docs/design/DEFERRED.md (D-6 #1 entries ~1044/1186-1206/',
  '1214-1256, the H2 cure sketch, R-4e=L7, R-5 capstone); crates/sim/src/io/mod.rs (the frozen seam +',
  'delivery-semantics doc). Use git -C ' + ROOT + ' log --oneline -6.',
].join('\n')

phase('Design')
const design = await agent(
  [CONTEXT, '',
   'Produce the IMPLEMENT-READY R-4e design. Cover, concretely and citing file:line: (a) the SUB-SLICING',
   '(R-4e is large — propose ordered sub-slices, e.g. the load harness, the H2 re-key, the correlated-outage',
   '+ reschedule + conn_died coverage, R-5 capstone — each independently gate-able + commit-able, in an order',
   'where each de-risks the next); (b) the N-peer load-test HARNESS design (how to spawn 16-64 real quinn',
   'nodes deterministically, feasibility on a dev host, and EXACTLY which assertions are exact-deterministic',
   'vs which are generous non-flaky bounds — spell out how each avoids a wall-clock race); (c) the H2 per-peer',
   'RecvLedger re-key: the precise new data structure (std-only, no new dep — or an explicit flag-to-user if',
   'unavoidable), how classify_reliable / the inbox-drop rollback / ack_egress each adapt, and the exact',
   'invariant that must not break; (d) the correlated-outage, pod-reschedule (L5), and replay_lanes-conn_died',
   'items — the mechanism each tests + the cure if it exposes a defect; (e) R-5 mesh_under_loss.rs — what it',
   'proves, whether flipping D-6 #1 green requires the AwaitAdopt egress cure first (be precise), and the',
   'exact assertions; (f) the `just mesh-load` recipe + how it is gated; (g) the risks/unknowns you are least',
   'sure about. Be decisive; give the shape a reviewer can attack.'].join('\n'),
  { label: 'design', phase: 'Design', model: 'opus' }
)

phase('Review')
const REVIEW_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['verdict', 'findings', 'summary'],
  properties: {
    verdict: { type: 'string', enum: ['SOUND_TO_IMPLEMENT', 'FIX_BEFORE_IMPL'] },
    summary: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'area', 'problem', 'fix'],
        properties: {
          severity: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'] },
          area: { type: 'string' },
          problem: { type: 'string' },
          fix: { type: 'string' },
        },
      },
    },
  },
}

const LENSES = [
  {
    key: 'h2-concurrency-correctness',
    prompt: [
      'LENS: the H2 per-peer RecvLedger re-key CORRECTNESS + real concurrency benefit. Attack the proposed',
      'data structure: does per-peer keying preserve the R-3 classify ladder (incarnation>epoch>contiguity),',
      'the inbox-drop rollback (RecvState Copy, *st=before on a reliable drop — is the rollback still atomic',
      'under the new locking?), and the ack_egress cumulative-ack snapshot (does it still see a consistent',
      'per-peer view)? Does dropping the lock before push_inbox open a TOCTOU / lost-update / reordering race',
      '(two frames from the same peer+class racing the classify→push→rollback sequence)? Is the BoundedInbox',
      'still correctly shared / does per-peer RX now race on the inbox push? Does the re-key ACTUALLY relieve',
      'serialization (or just move the contention)? Is it genuinely std-only, no new dep? Flag any correctness',
      'regression the re-key could introduce vs the current single-lock design.',
    ].join('\n'),
  },
  {
    key: 'loadtest-determinism-feasibility',
    prompt: [
      'LENS: load-test DETERMINISM / anti-flake + feasibility. A load test that flakes in CI is a liability.',
      'Attack every assertion: which are exact-deterministic (no-loss/no-dup/gap_drop==0/reliable_acked==N)',
      'and which are wall-clock (throughput/latency) — for each wall-clock one, is it a generous ceiling that',
      'cannot flake under a loaded CI host, or a tight number that will? Is "reliable_acked keeps pace"',
      'expressible without a timing race (e.g. as eventual completion within a deadline)? Feasibility at 16-64',
      'real quinn nodes on ONE host: fd limits, ephemeral-port exhaustion, tokio worker/task counts, memory,',
      'the 30s test deadline — will it actually run green repeatedly, or is 64 too many? Is the correlated',
      'outage (drop >=50%) reproducible deterministically? Is the node-count/duration a magic number or a',
      'justified/config const? Recommend the robust assertion set + the max feasible N.',
    ].join('\n'),
  },
  {
    key: 'scope-seam-r5-capstone',
    prompt: [
      'LENS: scope/seam/HR-compliance + the R-5 capstone soundness + end-goal composition. Confirm R-4e does',
      'NOT touch the frozen sim::io seam (it is below it) and introduces no shard-kind fork (HR3). Scrutinize',
      'R-5 mesh_under_loss.rs: does the proposed producer-less flow GENUINELY prove the D-6 #1 at-least-once',
      'cure, and is flipping D-6 #1 to GREEN justified GIVEN the still-open AwaitAdopt producer-less saga phase',
      '(DEFERRED.md D-6 precondition #1 1a has NO orchestrator re-drive egress)? Precisely: does the transport-',
      'layer capstone prove what D-6 #1 claims, or would flipping it green over-claim while a saga-layer',
      'producer-less gap remains? Is the sub-slicing right (each slice gate-able + commit-able, ordered to',
      'de-risk)? Does the pod-reschedule (L5) item expose a REAL defect (peer_writer captures addr once) that',
      'needs a code fix, not just a test? Does the whole R-4e compose toward hundreds-in-one-location + PvP +',
      'multisharding meshes, or does it prove a scale the architecture cannot actually reach (e.g. the',
      'directory single-writer or the gateway becomes the real bottleneck, not the mesh)?',
    ].join('\n'),
  },
]

const reviews = await parallel(
  LENSES.map((l) => () =>
    agent(
      [CONTEXT, '', 'THE PROPOSED R-4e DESIGN TO REVIEW:', design, '', 'YOUR REVIEW LENS:', l.prompt, '',
       'Read the actual code to ground every judgement. Be adversarial and concrete (file:line). A finding',
       'names a REAL design defect with a specific fix. If the design is sound on your lens, say so. Reserve',
       'CRITICAL for a design flaw that would produce broken/flaky/over-claiming code if implemented as-is.'].join('\n'),
      { label: 'review:' + l.key, phase: 'Review', schema: REVIEW_SCHEMA, model: 'opus' }
    )
  )
)

phase('Synthesize')
const synth = await agent(
  [CONTEXT, '', 'THE PROPOSED DESIGN:', design, '',
   'Three adversarial reviewers returned verdicts:', JSON.stringify(reviews, null, 2), '',
   'Adjudicate against the ACTUAL code (read files to confirm/refute). Produce the FINAL VETTED, implement-',
   'ready R-4e design: fold every real finding, state the resolved sub-slicing (ordered, each gate-able), the',
   'H2 re-key data structure + the exact adapted invariants (std-only or an explicit user-flag), the robust',
   'non-flaky assertion set + the max feasible N, the correlated-outage/reschedule/conn_died items + any code',
   'fixes they force, the precise R-5 claim (what flipping D-6 #1 green does and does NOT require re: the',
   'AwaitAdopt gap), and the `just mesh-load` gating. End with a decisive verdict (SOUND_TO_IMPLEMENT or',
   'FIX_BEFORE_IMPL) and the recommended FIRST sub-slice to implement.'].join('\n'),
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return { design, reviews, synthesis: synth }
