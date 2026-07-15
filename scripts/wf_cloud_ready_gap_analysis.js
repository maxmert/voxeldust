export const meta = {
  name: 'cloud-ready-gap-analysis',
  description: 'Gap analysis + phased plan: run the servers on real k3d + continuously test the game in a cloud-ready env',
  phases: [
    { title: 'Survey', detail: 'read-only Explore across 4 fronts' },
    { title: 'Synthesize', detail: 'phased plan + scope options + effort' },
  ],
}

const CTX = `
VOXELDUST greenfield rebuild (worktree new-system, branch worktree-new-system). Transfer-first voxel MMO
(Star Citizen + Minecraft). The transfer/authority/transport FOUNDATION is now robustness-complete for the
reachable-now tier (P0-P3 + the D-6 durable-saga, R-6 at-least-once redelivering MeshTransport, CA-1 no-DNS
peer addressing arcs — all in-process + real-QUIC-loopback tested, HR5 100% Tier-A coverage). HEAD = e692fe3.

USER GOAL (verbatim intent): "make sure we have the base we can build on ... fully prepare and be able to run
our servers on k3d and test game all the time in cloud-ready environment." I.e. BEFORE building P4 voxel
features, stand up a REAL k3d (kubernetes) deployment of the servers + a continuously-testable game (both
agent-driven HR6 vdctl scenarios AND human play) against it — validating the whole hardening arc in-cluster.

KEY CURRENT-STATE FACTS (confirmed):
- The "cluster" today is \`vd-devcluster\` (crates/bins/src/bin/vd-devcluster.rs) — a LOCAL-PROCESS launcher
  that spawns the real bins as child processes; its header says it "replaces k3d for local dev". Real QUIC +
  real redb + SIGKILL reaper. There is NO Docker/k8s/k3d manifest anywhere in the repo.
- Server bins: crates/bins/src/bin/{shard,gateway,orchestrator}.rs (+ auth?). One shard binary =
  ShardProfile config (HR3). io-prod (crates/io-prod) = quinn QUIC + redb + the mesh transport + boot.
- Client: crates/bins/src/bin/client.rs = the HEADLESS dev-control (HR6) shell; features \`render\` (Bevy 0.18,
  Slice-3 render loop IN PROGRESS) + \`dev-control\` (the vdctl JSON-lines listener + wgpu readback capture seam).
- Tooling: scripts/dev-cluster.sh, scripts/client.sh; vd-slot.rs (per-worktree slot). justfile recipes:
  render-smoke (headless cluster + client), client-load, orch-crash, spike2a.
- LEDGERED deploy-bundle items (docs/design/DEFERRED.md): the durable-root ALLOW-list (VD_STORE_DURABLE_ROOT),
  the self_fence_grace>0 cloud-profile boot assertion + lease-timing coherence, the WAL_FORMAT_VERSION +
  universe_epoch byte + Tombstone family, the CA-1 k3d CrashLoop/reschedule e2e (drives update_peer_addr on a
  NAT'd reschedule), the non-sheddable-ack transport refinement, SPIFFE per-shard certs (day-1 = ONE static PSK).
- 6 HARD RULES apply. "No CI/CD yet" was a prior constraint the user is NOW lifting (this deploy-readiness signal).
  NEVER unilaterally adopt a NEW library/tool — surface options, the user decides.

You are READ-ONLY (Explore): git-grep + read the real code/tooling; do NOT edit or guess. Cite file:line.`

const SURVEY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['front', 'built', 'owed', 'risks'],
  properties: {
    front: { type: 'string' },
    built: {
      type: 'array',
      description: 'what ALREADY exists + works (with file:line evidence)',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['item', 'where', 'state'],
        properties: {
          item: { type: 'string' },
          where: { type: 'string' },
          state: { type: 'string', enum: ['done', 'partial', 'stub'] },
        },
      },
    },
    owed: {
      type: 'array',
      description: 'what is MISSING for real-k3d + continuous game testing',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['item', 'why', 'effort', 'blocks'],
        properties: {
          item: { type: 'string' },
          why: { type: 'string' },
          effort: { type: 'string', enum: ['S', 'M', 'L', 'XL'] },
          blocks: { type: 'string', description: 'what this is a prerequisite for' },
        },
      },
    },
    risks: {
      type: 'array',
      description: 'traps / decisions / unknowns the plan must resolve',
      items: { type: 'string' },
    },
  },
}

const FRONTS = [
  {
    key: 'server-deployability',
    prompt: `${CTX}

FRONT 1 — SERVER DEPLOYABILITY (bins + io-prod → runnable k8s pods). Read crates/bins/src/bin/{shard,gateway,
orchestrator}.rs (and any auth bin), crates/io-prod/src/{runtime,boot,mesh,store,trust}.rs, and their Cargo.toml.
Map: (1) how each bin is configured (env vars — VD_* — enumerate them; ports; peer book; durable root; PSK/certs);
(2) what the io-prod runtime needs at boot (redb path/volume, quinn cert/key, the peer topology + how peers are
addressed — the static book vs CA-1 learned/update_peer_addr); (3) what containerization requires (a Dockerfile
per bin or one multi-bin image; build profile; static vs dynamic linking; image size); (4) the persistence story
(redb single-writer per realm on a PVC — what mount + the durable-root allow-list); (5) the trust story (the ONE
static PSK mTLS today — how it is provided, and what a k8s Secret mounting looks like). Return built/owed/risks.`,
  },
  {
    key: 'cluster-substrate',
    prompt: `${CTX}

FRONT 2 — CLUSTER SUBSTRATE (local-process vd-devcluster → real k3d/kubernetes). Read
crates/bins/src/bin/vd-devcluster.rs, vd-slot.rs, scripts/dev-cluster.sh, scripts/client.sh, and the justfile
cluster/render-smoke/client-load recipes. Map: (1) exactly what vd-devcluster does (spawn topology, wait-ready,
runfiles, reap, slot/port assignment) — what it PROVES and what it does NOT (it is local-process, not k8s); (2)
the gap to real k3d — Deployments/StatefulSets (which nodes are stateful? the orchestrator + realm-owning shards
hold redb), Services + how pods address each other (k8s DNS vs the CA-1 no-DNS learned-peer path — CRITICAL: does
CA-1 make k8s Service DNS unnecessary, or is a headless Service still needed for the initial peer book?), the pod
reschedule → new-IP → update_peer_addr path, PVCs, ConfigMaps/Secrets, liveness/readiness probes, resource
limits; (3) how the client reaches the gateway (NodePort/LoadBalancer/port-forward); (4) whether the old-system
k3d slot tooling (per project memory: dev-cluster.sh pins --context, spawn-mode pitfalls) should be revived or
rebuilt. Return built/owed/risks.`,
  },
  {
    key: 'client-and-testing',
    prompt: `${CTX}

FRONT 3 — CLIENT + CONTINUOUS TESTING ("test game all the time"). Read crates/bins/src/bin/client.rs, the
vd-client / vd-client-render / vd-client-harness crates (find them under crates/), and the vdctl CLI + the HR6
dev-control seam + the wgpu readback capture. Map: (1) the client render state — is the Slice-3 Bevy render loop
runnable (can a human SEE + walk a dot), or still a headless step()-only shell? what is owed to "play the game"?;
(2) the dev-control/vdctl harness — what agent-driven commands exist (press/move/walk-to/state/screenshot/record)
and what is owed for continuous agent testing against a cluster; (3) how the client currently connects (localhost
slot port) and what changes to connect to a k3d gateway; (4) the HR6 visual gates (render-smoke, no-magenta,
content-present) — do they run against vd-devcluster today, and what changes for k3d. Distinguish "agent-driven
HR6 testing" from "human manual play" — both are in the user's "test game all the time". Return built/owed/risks.`,
  },
  {
    key: 'cloud-readiness-gaps',
    prompt: `${CTX}

FRONT 4 — CLOUD-READINESS / SECURITY / OBSERVABILITY / DEPLOY-BUNDLE. Read docs/design/DEFERRED.md (the deploy
preconditions), crates/wire/src/admin.rs (the admin endpoint), and grep for metrics/Prometheus, health, graceful
shutdown, config validation. Map every gap between "reachable-now correct" and "safe to run continuously in a real
cluster": (1) the ledgered deploy-bundle items (durable-root allow-list, self_fence_grace>0 boot assertion +
lease-timing coherence, WAL_FORMAT_VERSION + epoch byte + Tombstone quarantining decode, the CA-1 k3d CrashLoop
e2e, non-sheddable-ack, SPIFFE) — for each, state what it concretely requires to LAND and whether it BLOCKS a
first k3d bring-up or can follow; (2) operational readiness — the admin endpoint / Prometheus metrics
(saga_stuck, forced_commit, clock_skew, fsync p99, ghost staleness), structured logs, graceful shutdown/drain,
liveness+readiness probes, resource requests/limits, the atomic dev-wipe of stores; (3) identity/auth in-cluster
(Ed25519 login ticket, HMAC resume, the auth service — is there an auth bin? what does k8s need). Return
built/owed/risks. Flag anything that is a HARD blocker for a first honest k3d bring-up vs a follow-up hardening.`,
  },
]

phase('Survey')
const surveys = await parallel(
  FRONTS.map((f) => () =>
    agent(f.prompt, { label: `survey:${f.key}`, phase: 'Survey', schema: SURVEY_SCHEMA, agentType: 'Explore', model: 'opus' })
  )
)

phase('Synthesize')
const packed = surveys.filter(Boolean).map((s) => JSON.stringify(s)).join('\n\n')
const plan = await agent(
  `${CTX}

You are the synthesizer. Here are the 4 read-only front surveys (JSON):

${packed}

Produce a CONCRETE PHASED PLAN to reach "servers running on real k3d + the game continuously testable (agent-driven
HR6 AND human-playable) in a cloud-ready environment", to bring back to the user for approval. Include:
1. A crisp CURRENT-STATE summary (what is genuinely built vs the local-process illusion of "cluster").
2. The CRITICAL DECISIONS the user must make (e.g. does CA-1 no-DNS obviate k8s Service DNS, or keep a headless
   Service for the seed book? one multi-bin image vs per-bin images? revive old k3d slot tooling vs rebuild?
   complete the Slice-3 client render for human-play now, or agent-HR6-testing-first then human-play later?
   real cloud provider eventually, or k3d-local as the cloud-ready proxy?). For each, give YOUR recommendation.
3. SCOPE OPTIONS as a small menu (e.g. "A: servers-on-k3d + agent-HR6-testing first (human-play deferred)";
   "B: full servers+human-playable-client on k3d"; "C: harden the deploy-bundle items on vd-devcluster first,
   then k3d") with rough effort (S/M/L/XL per slice) and what each de-risks.
4. A RECOMMENDED SEQUENCE of slices (each: goal, deliverable, DoD, which HARD blockers it clears), ordered to
   de-risk earliest — grounded in the transfer-first thesis (prove the deployment robust before features) and
   the user's HR6 / visual-testable-end-state standards.
5. Explicit callouts of any NEW tool/library a slice would need (Docker, a k8s manifest tool, an ingress) so the
   user can decide (never unilaterally adopt). Do NOT edit any file; this is a plan to present.`,
  { label: 'synthesize-plan', phase: 'Synthesize', agentType: 'Explore', model: 'opus' }
)

return { surveys: surveys.filter(Boolean), plan }
