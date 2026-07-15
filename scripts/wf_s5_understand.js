export const meta = {
  name: 's5-understand',
  description: 'Map the client/vdctl/dev-control + in-cluster-agent ground truth to scope S5 correctly',
  phases: [{ title: 'Map', detail: '4 parallel readers over client/harness/e2e/in-cluster' }],
}

const CTX = `
CONTEXT — Voxeldust greenfield rebuild, worktree /Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system.
The cloud-ready k3d base (S0-S4) is DONE + LIVE-validated (HEAD e2f399a): 3 server pods (orchestrator/gateway/shard)
run in a real k3d cluster, all Ready, a shard holds its realm. We are scoping S5: "in-cluster agent-HR6 continuous
testing" — a headless agent that logs into the live cluster's GATEWAY, injects input (walk-to/look-at) via the
dev-control seam, and asserts game state (an avatar crossing a boundary), writing runs/ manifests. HR6 = "client
ships the dev-control harness (vdctl): input injection at the input-resource seam, wgpu readback screenshots/video,
runs/ manifests". KNOWN CONSTRAINTS: GPU-in-pod is DEFERRED (Option-B), so wgpu capture cannot run in a k3d pod
day-1; kubectl port-forward is TCP-only so a LOCAL client cannot reach the in-cluster gateway's QUIC/UDP mesh; the
client render path (Bevy/wgpu, vd-client-render Tier-B) is SEPARATE from a renderer-free harness (vd-client-harness
Tier-A). The client walk-to/look-at nav math reportedly EXISTS but DevRequest returns Unsupported (client.rs:773).
This is READ-ONLY mapping. Report ground truth with exact file:line citations + verbatim type/fn/env names. If
something does NOT exist, say so (a gap is a finding). This map is the sole input to the S5 design — be exhaustive.
`

const READERS = [
  {
    key: 'devcontrol-vdctl',
    prompt: `${CTX}
MAP the dev-control / vdctl input-injection seam + the walk-to/look-at gap. Answer:
- The DevRequest / dev-control protocol: enumerate every DevRequest variant + what each does. WHERE is the
  "input-resource seam" input injection happens at (HR6)? Cite the client bin (crates/bins/src/bin/client.rs) +
  any vdctl/dev-control module.
- The client.rs:773 (or nearby) DevRequest that returns Unsupported: what request(s) are unsupported? Specifically
  walk-to / look-at — what would they need? Quote the Unsupported arm + the surrounding match.
- The NAV MATH that "exists": where is the walk-to / look-at computation (heading, movement toward a target,
  camera aim)? Is it wired to ANYTHING today, or dead? What input does it produce (a movement vector / an input
  frame the gateway forwards to the shard)?
- Is the dev-control seam in the renderer-FREE vd-client-harness (Tier-A) or the render path (vd-client-render,
  Tier-B)? Can input be injected WITHOUT wgpu? Cite the crate boundaries.
- vdctl: what is it (a bin? a lib?), what commands does it expose, and how does it talk to the client (a socket?
  a control channel? shared memory?)? Cite it.
Search: crates/bins/src/bin/client.rs, crates/client*/, any vdctl/dev-control/dev_control module, crates/wire (DevRequest).`,
  },
  {
    key: 'client-connect-headless',
    prompt: `${CTX}
MAP how a CLIENT connects to the gateway + whether a HEADLESS renderer-free client works. Answer:
- The client login/connect path: how does the client DIAL the gateway (QUIC? which addr env — VD_GW_ADDR /
  VD_CLIENT_QUIC?)? The auth: does the client mint a login with VD_AUTH_SIGNING_KEY (the signing seed we generate
  via gen-authkey)? The Hello / login / ResumeTicket flow + the mTLS/client-cert story (is a client cert needed, or
  is the client trust different from the server<->server mesh mTLS?). Cite client.rs + the wire login types.
- HEADLESS: is there a NON-render agent client mode (input inject + state assert, NO wgpu), or does the client bin
  always spin up Bevy/wgpu? What does 'client --capture' vs 'client --window' do, and is there a third headless
  mode? Can vd-client-harness (Tier-A) drive a full login+input+state loop with NO renderer? Cite the client bin's
  arg dispatch + the harness API.
- What env/config does a client need to log into the live cluster's gateway (VD_* vars: the gw addr, the signing
  key, a session seed, trust)? Enumerate them.
- Does the client read authoritative STATE back (the avatar's position / the WorldState / snapshots) in a form an
  agent can assert on (e.g. "the dot crossed x=0")? Where is that state exposed to the harness?
Search: crates/bins/src/bin/client.rs, crates/client*/ (vd-client, vd-client-harness), crates/wire login/session types.`,
  },
  {
    key: 'existing-e2e',
    prompt: `${CTX}
MAP the EXISTING e2e / scenario / agent test harness we can build S5 on. Answer:
- crates/bins/tests/client_load.rs (the SCALE-1 K-client load gate): how does it spawn K dev-control clients, log
  them in, and assert routing? What client mode does it use (headless?)? Can its pattern drive an agent scenario?
- The harness crate (vd-harness): Topology, FaultFabric, ControlOracle, WireMonitor, ChaosRunner — what do they do,
  and can they observe/assert an agent's game state e2e? Cite.
- runs/ manifests (HR6): what is a runs/ manifest, who writes it, what does it contain? Is there a scenario runner
  that loops (agent-HR6 continuous testing)?
- Any existing test that drives an avatar to MOVE + asserts a position/boundary-crossing (the visual-testable
  end-state's "dot crossing a boundary")? Cite the closest existing scenario.
- The process-tier test rig (crates/bins/tests/*, vd_bins Cluster/spawn_node) that spawns real node PROCESSES: can
  it spawn a real CLIENT process against a real gateway + assert? Cite the closest pattern.
Search: crates/bins/tests/ (client_load, probe_endpoints, dev_cluster_smoke), crates/harness/, tests/, any runs/ dir.`,
  },
  {
    key: 'in-cluster-agent',
    prompt: `${CTX}
MAP the feasibility of an IN-CLUSTER agent-client pod (the only way to reach the gateway's QUIC/UDP, since
port-forward is TCP-only + GPU-in-pod is deferred). Answer:
- Can a RENDERER-FREE agent client image be built (the server image is Bevy-free; the client --capture image pulls
  wgpu ~8.8GB)? Is there a client bin build WITHOUT the render feature that still logs in + injects input + asserts
  state? What features/deps would a headless agent image need? Cite crates/bins/Cargo.toml features + the client bin.
- The CLIENT's cluster connection: an in-cluster agent pod on the pod-network dials the gateway Service
  (vd-gateway-0.vd-gateway.voxeldust.svc:9000 QUIC). Does the client's QUIC/mTLS trust work with the SAME
  ca.der/node.der/key.der bundle (vd-mtls Secret), or does a client need a DIFFERENT cert/trust? Cite the trust +
  the client's endpoint setup. Is the client a "peer" in the mesh sense or a separate QUIC client?
- AUTH: the client needs VD_AUTH_SIGNING_KEY (the seed gen-authkey printed) to mint logins. How is that delivered
  to an in-cluster agent (a Secret? an env)? The gateway verifies with VD_AUTH_PUBKEY (already a Secret). Cite.
- The gateway admission: VD_MAX_SESSIONS=8 (cloud manifest). The login → session → attach → Active flow: what does
  a client see once Active (its avatar entity, a frame stream)? Can a headless agent confirm "I am Active + my
  avatar is at position P"? Cite the gateway session + client state.
- What is the SMALLEST viable S5 e2e: an in-cluster agent pod (or Job) that logs in, walks the avatar across a
  boundary, and asserts it — WITHOUT wgpu. Identify every missing piece (image, manifest, code) to reach it.
Search: crates/bins/Cargo.toml, crates/bins/src/bin/client.rs, crates/io-prod/src/trust.rs, deploy/k3d/, crates/connection-plane/ (admission).`,
  },
]

const MAP_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['dimension', 'findings', 'gaps', 'recommendation'],
  properties: {
    dimension: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['fact', 'evidence'],
        properties: {
          fact: { type: 'string' },
          evidence: { type: 'string', description: 'file:line or verbatim name/quote' },
        },
      },
    },
    gaps: { type: 'array', items: { type: 'string' }, description: 'what does NOT exist and S5 must add' },
    recommendation: { type: 'string', description: 'the dimension-specific S5 recommendation' },
  },
}

phase('Map')
const maps = await parallel(
  READERS.map((r) => () =>
    agent(r.prompt, {
      label: `map:${r.key}`,
      phase: 'Map',
      schema: MAP_SCHEMA,
      agentType: 'Explore',
      model: 'opus',
    }),
  ),
)

return { maps: maps.filter(Boolean) }
