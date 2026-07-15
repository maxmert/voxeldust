export const meta = {
  name: 's5a-agent-design',
  description: 'Design the S5a in-cluster agent-HR6 e2e (image + manifest + secret + scenario + justfile)',
  phases: [
    { title: 'Design', detail: '3 independent agent-harness architectures' },
    { title: 'Adjudicate', detail: 'synthesize the vetted plan + flag risks' },
  ],
}

const GROUND = `
GROUND TRUTH (verified by the S5 understand workflow — BINDING):
- The cloud-ready base is LIVE (k3d cluster voxeldust-newsystem): orchestrator/gateway/shard pods 1/1 Ready, a
  shard holds its realm. deploy/k3d/{00-namespace..50-shard}.yaml + docker/entrypoint.sh + docker/server.Dockerfile
  + the just k3d-{up,load,secrets,apply,dod,all} recipes exist and work.
- The dev-control seam is RENDERER-FREE + WORKS: the client bin built '--features dev-control' (NOT render, Bevy-free,
  small image) runs headless (default mode), exposes a LOOPBACK JSON-lines TCP dev-control listener, and vdctl
  (a bin) sends one DevRequest per invocation over 127.0.0.1:<port> and prints one DevResponse (exit 0 Ack/State,
  1 Error, 2 parse, 3 Timeout). Injected input rides the SAME InputState seam as real keys →
  InputDatagram(MsgClass::Input) → gateway → shard SessionInput. vdctl commands that WORK today: move <ax ay az>,
  look <dx dy>, action <bit> <0|1>, close, reset, state, wait <field> <op> <value>. WalkTo/LookAt/Screenshot/Record
  return DevError::Unsupported (walk-to/look-at = the SEPARATE S5b slice; screenshot/record = GPU, Option-B deferred).
- REACHABILITY (hard): kubectl port-forward is TCP-only; the gateway mesh is QUIC/UDP; the dev-control listener is
  loopback-only. So the agent-client AND its vdctl driver MUST run IN-CLUSTER, CO-LOCATED (same pod), reaching the
  gateway over the pod network (QUIC) and vdctl over loopback TCP. A laptop client cannot reach the in-cluster gateway.
- The client is a FULL mesh peer: it presents the SAME ClusterTrust node cert (ca.der/node.der/key.der via
  VD_TRUST_DIR = the vd-mtls Secret, reused unchanged) — NOT a weaker external-client trust. It mints logins with
  VD_AUTH_SIGNING_KEY (the Ed25519 signing seed gen-authkey prints; the gateway verifies with VD_AUTH_PUBKEY, already
  a Secret). TODAY just k3d-secrets stores ONLY the pubkey and DISCARDS the signing key — S5a must ALSO deliver the
  signing key to the agent pod (a Secret + env VD_AUTH_SIGNING_KEY).
- The client dials a raw SocketAddr --gateway; it has NO DNS resolver, so the agent pod must getent-resolve
  vd-gateway-0.vd-gateway.voxeldust.svc → IP before exec (reuse the docker/entrypoint.sh getent pattern). NodeId
  GATEWAY=2 is fixed. VD_MAX_SESSIONS=8 admits it.
- ASSERTION (day-1, GPU-free): DevState carries own_entity (id), per-entity DevEntityRow{entity, pos:[f64;3],
  authoritative_sub}, and a 'location'. NO orientation is published (that's the S5b gap). So the v1 boundary-crossing
  assertion is a POSITION threshold: record start pos via 'vdctl state', inject 'vdctl move' toward +x, poll
  'vdctl state' until entities[own].pos[0] crosses a threshold (or 'location' flips). WaitUntil has NO position
  field yet, so either poll state agent-side OR (optional) add a position WaitField (devproto/predicate.rs).
- The client_load.rs process-tier test already spawns real headless dev-control clients vs a real cluster over
  loopback QUIC + asserts via State/WaitUntil — reuse that login/auth/lifecycle pattern; do NOT rebuild it.
- The image build path: docker/server.Dockerfile builds -p vd-bins --bin vd-orchestrator/gateway/shard (default
  features). An agent image builds -p vd-bins --bin client --bin vdctl --features dev-control (Bevy-free, small).
`

const BRIEF = `
Design the COMPLETE S5a artifact set: an in-cluster agent-HR6 continuous-testing harness that proves the game is
AGENT-OPERABLE END-TO-END against the LIVE cluster, WITHOUT GPU, by logging an avatar in and driving it across a
boundary via injected input, asserting the crossing on DevState. Cover:
1. IMAGE: a docker/agent.Dockerfile (or a server.Dockerfile extension) building client + vdctl --features
   dev-control (Bevy-free); the entrypoint (getent-resolve the gateway → IP, then run the scenario). runAsNonRoot,
   the trust mount, no wgpu. Decide separate-image vs extend-server-image (trade-offs: image count, build reuse,
   the dev-control feature must NOT leak into the server bins).
2. THE SCENARIO: the exact sequence the agent runs — start 'client --gateway <ip>:9000 --client-quic <port>
   --trust-dir /etc/vd/trust --dev-control <port> --allow-dev-control' (verify the real client arg names) in the
   background; 'vdctl wait active eq 1' (login→Active); record start pos ('vdctl state'); inject 'vdctl move 1 0 0'
   (toward +x); poll 'vdctl state' until pos crosses the threshold OR a max-ticks timeout; exit 0 (crossed) / 1
   (timeout). Make it ROBUST (login retry, admission timing, the shard must hold its realm first) and DIAGNOSABLE
   (log each step + the final DevState). Decide: a shell script baked into the image, or a tiny Rust scenario bin.
3. MANIFEST: deploy/k3d/60-agent.yaml — a Job (runs to completion, exits 0/1, restartPolicy Never/OnFailure) OR a
   Pod, labeled app: vd (so the vd-allow-mesh NetworkPolicy admits it), mounting vd-mtls at /etc/vd/trust, envFrom
   vd-cluster-env, env VD_AUTH_SIGNING_KEY (from the new Secret) + the gateway target + the dev-control port +
   VD_NODE_ID (a client NodeId that does NOT collide with 1/2/3) + VD_BIND/VD_PEERS if the client needs them.
   securityContext (non-root, readOnlyRootFilesystem if the scenario writes only /tmp).
4. SECRET DELIVERY: extend just k3d-secrets to ALSO stash VD_AUTH_SIGNING_KEY (from the same gen-authkey run) into a
   Secret, injected as env on the agent pod. Keep the signing key OUT of git.
5. JUSTFILE: a k3d-agent recipe (build the agent image + k3d image import + apply 60-agent.yaml + wait for Job
   completion + report pass/fail from the Job's exit + logs). context-pinned.
6. CODE CHANGES (flag every one): what does the client bin need that it lacks (a DNS-resolve entrypoint; does
   --gateway take an IP:port; does the client need VD_NODE_ID/VD_BIND/VD_PEERS to boot; does vdctl 'state' emit
   parseable JSON for the pos threshold)? Does an assertion need a new position WaitField (optional) or is agent-side
   polling of 'vdctl state' sufficient? Anything in client_load.rs that reveals the exact client boot env.
7. EXTENSIBILITY: S5a must be a REUSABLE scenario foundation (later: build blocks, assert signals, walk into a ship),
   not a one-off — note how the scenario harness generalizes toward the end-goal (Star Citizen + Minecraft:
   signal-heavy blocks, ships, multisharding). Keep the boundary-crossing scenario the FIRST of many.
Be concrete: real Dockerfile lines, real manifest YAML, the real scenario commands, the real justfile recipe. Flag
every code change vs pure packaging, and every risk (login race, NodeId collision, the client needing extra env,
readOnlyRootFilesystem vs the scenario, the Job never terminating).
`

const EMPHASES = [
  { key: 'minimal-viable', lean: 'Lean MINIMAL-VIABLE: the fewest new artifacts that prove the e2e loop against the live cluster TODAY (raw Move + State-poll assertion, no new WaitField, a shell scenario baked in the image). Reuse server.Dockerfile/entrypoint.sh patterns. But never skip a REQUIRED client env or the signing-key delivery.' },
  { key: 'robust-diagnosable', lean: 'Lean ROBUST + DIAGNOSABLE: design for the failure modes — a login race (gateway not Ready / shard has no realm yet), a NodeId collision with 1/2/3, admission rejection (VD_MAX_SESSIONS), a Move that never crosses (timeout), a Job that hangs. Each must fail LOUD with a clear log + a non-zero exit, and the k3d-agent recipe must surface it. Prefer explicit waits + retries + bounded timeouts.' },
  { key: 'extensible-foundation', lean: 'Lean EXTENSIBLE-FOUNDATION: design the scenario harness so the boundary-crossing is the FIRST of many agent scenarios (later: place a block, assert a signal propagates, walk into a ship, cross a real shard band). Prefer a small scenario abstraction (a named scenario the Job selects via env/arg) over a hardcoded one-off, and a renderer-free runs/-style artifact (a DevState/assertion dump) so scenarios are auditable. Keep it GPU-free + reusable, without over-engineering v1.' },
]

const DESIGN_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['emphasis', 'image', 'scenario', 'manifest', 'secret_delivery', 'justfile', 'code_changes_needed', 'extensibility', 'risks'],
  properties: {
    emphasis: { type: 'string' },
    image: { type: 'string', description: 'agent.Dockerfile shape + build cmd + entrypoint (concrete)' },
    scenario: { type: 'string', description: 'the exact command sequence + robustness + where it lives (shell vs bin)' },
    manifest: { type: 'string', description: '60-agent.yaml Job/Pod shape (concrete YAML)' },
    secret_delivery: { type: 'string', description: 'how VD_AUTH_SIGNING_KEY reaches the agent pod' },
    justfile: { type: 'string', description: 'the k3d-agent recipe' },
    code_changes_needed: { type: 'array', items: { type: 'string' }, description: 'every code change vs pure packaging' },
    extensibility: { type: 'string', description: 'how it generalizes to future scenarios (blocks/signals/ships)' },
    risks: { type: 'array', items: { type: 'string' } },
  },
}

phase('Design')
const designs = await parallel(
  EMPHASES.map((e) => () =>
    agent(`${GROUND}\n${BRIEF}\n\nYOUR EMPHASIS: ${e.lean}`, {
      label: `design:${e.key}`,
      phase: 'Design',
      schema: DESIGN_SCHEMA,
      agentType: 'Explore',
      model: 'opus',
    }).then((d) => (d ? { ...d, _key: e.key } : null)),
  ),
)
const valid = designs.filter(Boolean)

phase('Adjudicate')
const plan = await agent(
  `${GROUND}\n${BRIEF}\n\nYou are the ADJUDICATOR + ADVERSARY. Synthesize ONE vetted, buildable S5a plan from the
${valid.length} designs below, and adversarially STRESS it: find any BINDING-CONSTRAINT violation (a missing
required client env so the client won't boot; the dev-control feature leaking into the server bins; a NodeId
collision with 1/2/3; the signing key landing in git; a Job that never terminates; readOnlyRootFilesystem vs a
scenario that writes; asserting on a screenshot instead of DevState). Produce a CONCRETE plan an implementer can
author directly: the exact new/changed files, the agent.Dockerfile lines, the 60-agent.yaml, the scenario commands
(with the REAL client + vdctl arg names — flag any you are unsure of as MUST-VERIFY against client.rs/vdctl.rs/
client_load.rs), the k3d-agent recipe, the signing-key Secret delivery, the explicit list of CODE changes vs pure
packaging, and the residual risks. Confirm the whole thing is GPU-free + runs against the LIVE cluster.

DESIGNS:
${JSON.stringify(valid, null, 1)}`,
  { label: 'adjudicate', phase: 'Adjudicate', model: 'opus' },
)

return { plan, designs: valid }
