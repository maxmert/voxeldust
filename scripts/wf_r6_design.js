export const meta = {
  name: 'r6-design',
  description: 'Design + adversarial review of R-6 (durable monotone incarnation + durable outbox) for k3d',
  phases: [{ title: 'Design' }, { title: 'Review' }, { title: 'Synthesize' }],
}

const ROOT = '/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system'

const CONTEXT = [
  'R-6 is the FINAL cloud-blocking precondition of the D-6 #1 redelivering-transport arc in the voxeldust',
  'greenfield rebuild (a transfer-first "Star Citizen meets Minecraft" voxel MMO). The R-1..R-5 transport',
  'arc is DONE (RX ledger + per-lane replay + retransmit timer + bounded retry + SendShed + the H2 per-peer',
  're-key + the N-peer load gate + the R-5 producer-less capstone). The user now wants to test the CLOUD',
  'setup LOCALLY in a k3d (k3s-in-docker) K8s cluster — so R-6 is now NEAR-TERM, not P6/P7-deferred.',
  '',
  'THE END GOAL (keep in memory — the foundation must not drift): spherical voxel PLANETS; player-built',
  'SHIPS walked-inside while flying (Newtonian); SPACE STATIONS + CITIES on planets, all from BLOCKS; a',
  'SIGNAL system for functional-block communication, signal-HEAVY, POSSIBLY CROSS-SHARD; MULTISHARDING',
  'MESHES (one location across shards); CLIENT multi-mesh simultaneous rendering; players COLLIDE',
  'cross-boundary (NO client prediction); PvP; HUNDREDS in one location. Robust, scalable, modular,',
  'extremely DRY, elegant. Multithreading where it helps. 100% Tier-A coverage; e2e + load testing.',
  '',
  'THE R-6 PROBLEM (two owed items, both CLOUD-BLOCKING before any real/k3d deploy):',
  '',
  'M3 — the durable MONOTONE incarnation boot-counter. TODAY `VD_PROCESS_INCARNATION` = wall-clock-ms',
  '(`launch_incarnation()` in crates/bins/src/lib.rs:173, set ONCE per cluster launch in `common_env`:190,',
  'threaded to `MeshConfig::process_incarnation`, stamped on every reliable frame at mesh.rs:795). This is',
  'UNSAFE under k8s: (a) a CrashLoopBackOff restart is sub-second ⇒ an EQUAL incarnation ⇒ the receiver',
  'ledger silent-Dedups/Stale-drops the restarted sender\'s traffic (`classify_reliable` incarnation ladder,',
  'mesh.rs ~358) ⇒ clear-on-ack never fires ⇒ a split-brain-ish stall; (b) NTP/reschedule clock-skew ⇒ a',
  'LOWER incarnation ⇒ StaleIncarnation-drops ALL the restarted node\'s traffic. The cure: a DURABLE MONOTONE',
  'boot-counter that INCREMENTS on every boot and survives a wall-clock REWIND (it is NOT clock-derived).',
  'Each NODE (orchestrator, gateway, shard) that runs a mesh + stamps an incarnation needs its own. The',
  'orchestrator already has a redb `Store` (D-6 Slice D, `VD_STORE_PATH` on a persistent volume); gateway/',
  'shard are currently stubs without a durable store. In k3d each pod needs per-pod persistence (a',
  'StatefulSet PVC / hostPath). The counter is ONE u64 — persist it atomically read-increment-write-fsync on',
  'boot. DEP DISCIPLINE (BINDING): NO new dependency — reuse the `sim::io::Store` seam / redb OR a minimal',
  'std durable file; if a new crate seems needed, FLAG it for the user (jointly investigated, never adopted',
  'unilaterally).',
  '',
  'DURABLE OUTBOX — the sender-side retry buffer is RAM (dies with the process). This is the residual the',
  'R-4e4 review sharpened: a source that crashes IN `BatchHandoff::AwaitAdopt` BEFORE its `TransientBatch`',
  'reached the dest loses the batch even with the confirm-dead self-promote (the dest never received it).',
  'A durable outbox (retain-until-acked across a process restart) closes that window. Scope: is this needed',
  'for the FIRST k3d test, or can the boot-counter (M3) land first + the durable outbox follow? (identity_',
  'persistence.md:122 sketches retain-until-acked.)',
  '',
  'L4 — the `AckFrame` carries ONE `incarnation` scalar over its whole entry list (mesh.rs `AckFrame`,',
  'io-prod/src/lib.rs). Provably-correct TODAY (one accepted connection = one sender process = one',
  'incarnation), but if R-6 durable-incarnation work ever lets one accepted connection be REUSED across a',
  'sender restart, the single scalar breaks. Ledgered as an R-6 acceptance item: move `incarnation` into',
  '`AckEntry` per-class OR prove one-conn-one-incarnation holds under durable-incarnation connection-reuse.',
  '',
  'STANDARDS: HR1 sealed shards + closed seam (does R-6 touch the frozen `sim::io` seam? the incarnation is a',
  'bin-level config feeding `MeshConfig`, likely NOT a seam change — confirm); HR3 one tooling (ONE',
  'boot-counter mechanism for every node kind, never a per-kind fork); HR5 100% Tier-A region+branch (io-prod',
  '/bins Tier-B floor 90) — REAL not theater; no magic numbers; the k8s/durable-path work must fail LOUD on a',
  'mis-config (the existing HR1 ephemeral-store guard rejects a /tmp store path — the boot-counter path must',
  'get the SAME treatment). feedback_no_cicd_yet is now LIFTING (the user signalled k3d deploy-readiness) — so',
  'k8s manifests / a StatefulSet + PVC are in-scope IF R-6 needs them, but keep the CODE slice focused.',
  '',
  'Read before designing: crates/bins/src/lib.rs (`launch_incarnation`, `common_env`, the DevCluster env',
  'builders, `ClusterAddrs`, VD_STORE_PATH plumbing), crates/bins/src/bin/{orchestrator,gateway,shard}.rs',
  '(how each reads VD_PROCESS_INCARNATION + VD_STORE_PATH + the HR1 ephemeral-store boot guard),',
  'crates/io-prod/src/mesh.rs (process_incarnation → the stamped frame + classify_reliable incarnation',
  'ladder), crates/io-prod/src/store.rs (RedbStore + StoreTuning + the ephemeral-path guard),',
  'crates/sim/src/io/mod.rs (the Store seam), crates/bins/src/bin/vd-devcluster.rs + scripts/ (the existing',
  'k3d/devcluster tooling), docs/design/DEFERRED.md (D-6 #1, M3/L4/L5, the R-6 entries ~955/1074/1216-1232/',
  '1285-1296), docs/design/identity_persistence.md (the durable-outbox sketch). Use git -C ' + ROOT + ' log',
  '--oneline -6.',
].join('\n')

phase('Design')
const design = await agent(
  [CONTEXT, '',
   'Produce the IMPLEMENT-READY R-6 design. Cover, concretely + citing file:line: (a) the SUB-SLICING',
   '(propose ordered slices — the M3 boot-counter is almost certainly first as the k3d precondition; the',
   'durable outbox + L4 likely follow — each independently gate-able + commit-able); (b) THE M3 BOOT-COUNTER',
   'MECHANISM: where the u64 lives (a dedicated tiny durable file? the redb `Store`? a new `Store` key',
   'family?), the exact read-increment-write-fsync-on-boot flow, how it is ATOMIC + MONOTONE + survives a',
   'wall-clock rewind + a crash DURING the increment (does it double-increment or skip? is either safe?),',
   'how it threads from the bin boot into `MeshConfig::process_incarnation` (replacing `launch_incarnation()`',
   'for prod while dev/tests keep the simple path), and the k3d persistence story (per-pod PVC / StatefulSet /',
   'hostPath — where the file lives, the HR1-style fail-loud-on-ephemeral-path guard); std-only, NO new dep',
   '(or an explicit user-flag); (c) whether R-6 TOUCHES the frozen `sim::io` seam (it should not — confirm the',
   'incarnation stays a MeshConfig/bin concern) and any HR3 one-tooling concern (one mechanism, all node',
   'kinds); (d) the DURABLE OUTBOX scope + design (needed-for-first-k3d-test or a follow-slice? the',
   'retain-until-acked structure, redb-backed vs a WAL, how it composes with the RAM retry buffer + the',
   'incarnation reset); (e) the L4 AckFrame per-class-incarnation resolution (move it into AckEntry, or prove',
   'the invariant holds under durable-incarnation connection-reuse — which is it?); (f) the k3d LOCAL CLOUD',
   'TEST plan: what M3 unlocks — a CrashLoopBackOff test (kill a pod repeatedly / restart at a NEW durable',
   'incarnation, assert NO silent Dedup/Stale loss of the restarted node\'s reliable traffic), plus the L5',
   'reschedule interaction; how to test it in k3d with the existing devcluster tooling, and what the',
   'PASS assertion is; (g) the risks/unknowns you are least sure about. Be decisive; give a shape a reviewer',
   'can attack.'].join('\n'),
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
    key: 'durability-monotonicity-correctness',
    prompt: [
      'LENS: the M3 boot-counter DURABILITY + MONOTONICITY correctness. Attack the mechanism: is it truly',
      'MONOTONE across every failure (a crash DURING the read-increment-write-fsync — does it re-use an',
      'incarnation, skip one, or corrupt the file; is a torn write detected)? Does it survive a wall-clock',
      'REWIND (the whole point — confirm it is not clock-derived anywhere)? Is the fsync ordering correct (the',
      'new value durable BEFORE the process stamps frames with it — else a crash-after-stamp-before-fsync',
      're-uses it)? Interaction with the receiver ledger: is a STRICTLY-HIGHER incarnation on every boot',
      'sufficient to make `classify_reliable` reset the dedup high-water (so restarted traffic is never',
      'silent-dropped), and does a HIGHER incarnation ever wrongly reset a LIVE peer\'s state? The L4 AckFrame',
      'single-incarnation: does the durable boot-counter + any connection-reuse break the one-conn-one-',
      'incarnation invariant the ack path relies on — is the proposed resolution correct? Does the durable',
      'outbox (if in scope) correctly re-drive across a restart WITHOUT double-delivery (the receiver ledger',
      'dedup must still hold across the incarnation bump)?',
    ].join('\n'),
  },
  {
    key: 'k8s-k3d-cloud-fit',
    prompt: [
      'LENS: k8s / k3d cloud-fit + operational reality. Where does the boot-counter file actually live in a',
      'k3d pod, and does the proposed persistence (StatefulSet PVC / hostPath / the redb VD_STORE_PATH volume)',
      'genuinely SURVIVE a pod restart AND a pod RESCHEDULE to a new node (the L5 interaction)? Is a per-pod',
      'PVC the right model for the end-goal (many shards, dynamic provisioning, multisharding meshes), or does',
      'it paint into a corner? Does the CrashLoopBackOff test actually reproduce the hazard in k3d (a real',
      'sub-second restart at the SAME wall-clock second — does the durable counter demonstrably fix what the',
      'wall-clock version breaks)? Is the fail-loud-on-ephemeral-path guard (HR1-style) applied to the',
      'boot-counter path so a mis-mounted emptyDir can never silently make it non-durable? Consider the',
      'existing devcluster/k3d tooling (vd-devcluster.rs, scripts/, dev-cluster.sh) — does R-6 compose with it',
      'or need new manifests, and is that scoped sanely (the user OK\'d k3d, but is any of it over-built)? What',
      'about the orchestrator (single, has a store) vs gateway/shard (many, stubs) — does each get durability',
      'coherently?',
    ].join('\n'),
  },
  {
    key: 'seam-scope-dry-hr',
    prompt: [
      'LENS: frozen-seam impact + scope + DRY + HR compliance + end-goal composition. Confirm R-6 does NOT',
      'mutate the frozen `sim::io` seam (the incarnation is a MeshConfig/bin config, not a seam type) and',
      'introduces no HR3 shard-kind fork (ONE boot-counter mechanism for orchestrator/gateway/shard). Is the',
      'mechanism DRY (reuse the `Store` seam / the existing ephemeral-path guard / the env plumbing, not a',
      'parallel bespoke durability path)? Is the sub-slicing right (M3 first for k3d; durable outbox + L4 as',
      'separately-gate-able follow-ons; does landing M3 alone leave the tree coherent + honest in DEFERRED.md',
      'with the outbox residual still owed)? Is the durable-outbox design elegant or a smell (does it duplicate',
      'the RAM retry-buffer logic — should they be ONE machinery with a durable backend)? Does R-6 compose',
      'toward the end goal (hundreds of shards each with a durable incarnation; cross-shard signals; the',
      'multisharding-mesh + provisioning story) without a costly rework? Any magic numbers (the fsync cadence,',
      'the file path) that must live in ONE config? Does the k3d test give REAL e2e coverage of the cloud',
      'restart path, or is it theater?',
    ].join('\n'),
  },
]

const reviews = await parallel(
  LENSES.map((l) => () =>
    agent(
      [CONTEXT, '', 'THE PROPOSED R-6 DESIGN TO REVIEW:', design, '', 'YOUR REVIEW LENS:', l.prompt, '',
       'Read the actual code to ground every judgement. Be adversarial + concrete (file:line). A finding names',
       'a REAL design defect with a specific fix. If sound on your lens, say so. Reserve CRITICAL for a design',
       'flaw that would produce a non-monotone/lossy/non-durable incarnation, a broken cloud restart, a seam',
       'violation, or a new-dep-without-flag if implemented as-is.'].join('\n'),
      { label: 'review:' + l.key, phase: 'Review', schema: REVIEW_SCHEMA, model: 'opus' }
    )
  )
)

phase('Synthesize')
const synth = await agent(
  [CONTEXT, '', 'THE PROPOSED DESIGN:', design, '',
   'Three adversarial reviewers returned verdicts:', JSON.stringify(reviews, null, 2), '',
   'Adjudicate against the ACTUAL code (read files to confirm/refute). Produce the FINAL VETTED,',
   'implement-ready R-6 design: fold every real finding; state the resolved sub-slicing (ordered, each',
   'gate-able), the EXACT M3 boot-counter mechanism (where the u64 lives, the atomic monotone',
   'read-increment-fsync flow, the crash-during-increment safety, the wall-clock-rewind survival, the',
   'threading into MeshConfig, the k3d per-pod persistence + the fail-loud guard, std-only or an explicit',
   'user-flag), the L4 AckFrame resolution, the durable-outbox scope (in R-6 or a follow-slice, and its',
   'design if in scope), and the k3d CrashLoop e2e test + its PASS assertion. End with a decisive verdict',
   '(SOUND_TO_IMPLEMENT or FIX_BEFORE_IMPL) and the recommended FIRST sub-slice to implement.'].join('\n'),
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return { design, reviews, synthesis: synth }
