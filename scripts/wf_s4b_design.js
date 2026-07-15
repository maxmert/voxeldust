export const meta = {
  name: 's4b-manifest-design',
  description: 'Design the S4b k3d/k8s manifest architecture (judge panel + adversary + adjudicator)',
  phases: [
    { title: 'Design', detail: '3 independent manifest architectures' },
    { title: 'Critique', detail: 'adversarially stress each' },
    { title: 'Adjudicate', detail: 'synthesize the vetted plan' },
  ],
}

// Ground truth from the S4 understand workflow (wf_5cea13fc) — the binding constraints.
const GROUND = `
GROUND TRUTH (verified by the S4 understand workflow — treat as BINDING constraints):
- ONE Bevy-free image docker/server.Dockerfile (debian:bookworm-slim, USER vd uid 10001 nologin, NO ENTRYPOINT/CMD).
  Role = the manifest's command: [vd-orchestrator | vd-gateway | vd-shard] over the single image (HR3). Each bin
  reads ALL config from process env (EnvConfig::from_process_env()); there is NO argv config.
- VD_PEERS is SocketAddr-ONLY (id=IP:port; peer_book does addr.parse::<SocketAddr>()). k8s DNS names FAIL at boot.
  So the pod must DNS-resolve Service names to IPs before exec (entrypoint / initContainer). Empty VD_PEERS = solo.
- Bootstrap is BIDIRECTIONAL: the SHARD dials the ORCH (LeaseGrant registration every tick until granted) AND the
  ORCH dials the SHARD+GATEWAY (ClockSync broadcast every tick to VD_CLOCK_PEERS). shard/gateway readiness gates on
  clock_synced (proof orch->node reachable), so BOTH directions must be live before any non-orch node is Ready.
- COLD-START DEADLOCK CURE (mandatory, named in orchestrator.rs:264): the orch mesh Service MUST be headless +
  publishNotReadyAddresses:true so a shard can dial the orch's endpoint DNS while the orch is itself NotReady.
  orch_ready is own-serving (Ready from tick 1, independent of the cluster) — but its DNS must PUBLISH before that.
- CLOUD PROFILE (VD_PROFILE=cloud) fail-loud REQUIRES per pod: VD_STORE_DURABLE_ROOT + VD_BOOT_DURABLE_ROOT (present,
  non-empty, temp-disjoint PV paths) + VD_BOOT_STATE_DIR (under boot root) + VD_PROBE_ADDR; orchestrator adds
  VD_STORE_PATH (redb, under store root); gateway adds a REAL non-dev VD_AUTH_PUBKEY. FORBIDS VD_*_EPHEMERAL_OK +
  manual VD_PROCESS_INCARNATION. Do NOT set the D-3 knobs (VD_LEASE_*/VD_SELF_FENCE_GRACE/VD_REAPER_*/VD_LIVENESS_*/
  VD_REALM_RECHECK/VD_SESSION_RECHECK) — the cloud profile derives coherent active values from VD_TICK_HZ.
- REQUIRED env every bin: VD_NODE_ID, VD_BIND (0.0.0.0:mesh, QUIC/UDP), VD_PEERS, VD_OUTBOUND_CAP, VD_TRUST_DIR
  (dir with ca.der/node.der/key.der mTLS bundle), VD_TICK_HZ (REQUIRED, no code default). Orch adds VD_ADMIN_ADDR,
  VD_EPOCH, VD_RESERVE_CHUNK, VD_CLOCK_PEERS(=gateway,shard NodeIds), VD_STORE_PATH. Gateway adds VD_ORCH, VD_SHARD,
  VD_AUTH_PUBKEY, VD_SESSION_SEED, VD_MAX_SESSIONS, VD_MAX_BUFFERED_INPUTS. Shard adds VD_REALM_SEED, VD_SPEED,
  VD_TICK_DT(=1/VD_TICK_HZ), VD_ORCH, VD_MINT_SEED, VD_INPUT_LOG_CAP, VD_SNAPSHOT_BUDGET. (Full list in
  crates/bins/src/lib.rs orchestrator_env/gateway_env/shard_env — the canonical per-role var sets.)
- PROBE TIMING: stall_deadline = (1s/VD_TICK_HZ) * 20 ticks. is_live is drain-safe (draining|is_live) so /healthz
  stays LIVE through the SIGTERM drain — ONLY terminationGracePeriodSeconds protects the final fsync. Only the ORCH
  holds un-fsynced durable state (bounded ~few s: 100ms writer-death poll + 3x10ms fsync-retry + one commit); gateway/
  shard drain is ~VD_SHUTDOWN_LINGER_MS + teardown. THREE inequalities the manifest MUST satisfy: (1) liveness
  periodSeconds*failureThreshold >> stall_deadline (never restart a slow-but-progressing node); (2)
  terminationGracePeriodSeconds >= drain worst case (no SIGKILL mid-fsync); (3) grace > readiness de-route lead time.
- STORAGE: orch PVC needs redb store (VD_STORE_PATH) + boot-counter (VD_BOOT_STATE_DIR). shard/gateway PVC needs
  boot-counter (+ optional VD_OUTBOX_PATH, empty day-1). Per-pod monotone boot-counter => StatefulSet
  volumeClaimTemplates (one PVC per replica), fsGroup for uid 10001. Split roots (/var/lib/vd + /var/lib/vd-boot)
  and single-root layouts are both pre-validated in code.
- TRUST: ClusterTrust::from_der_dir(VD_TRUST_DIR) reads exactly ca.der/node.der/key.der (key.der is the private key,
  0600). ONE shared bundle for the whole cluster => a Secret mounted read-only at VD_TRUST_DIR (defaultMode 0400).
- AUTH: cloud vetoes the built-in dev VD_AUTH_PUBKEY (0x42 seed). A real Ed25519 pair must be generated; the
  verifying key goes in the gateway env; the signing seed goes to legitimate clients out-of-band (NOT a server Secret).
- Tooling approved + present: Docker (up), k3d v5.8.3, kubectl v1.35.2, RAW YAML (no Helm day-1), debian-slim/glibc.
- NO manifests / no k3d justfile targets / no entrypoint script exist yet — S4b authors ALL of it.
`

const BRIEF = `
Design the COMPLETE S4b artifact set to deploy the three server bins into a k3d cluster for day-1 agent-HR6 testing:
1. WORKLOADS: StatefulSet vs Deployment per role, replicas, volumeClaimTemplates, securityContext (fsGroup 10001,
   runAsNonRoot), the command: dispatch, resource requests/limits (orch at Guaranteed QoS per the D-3 ratify note).
2. SERVICES: the headless orch mesh Service with publishNotReadyAddresses:true (cold-start cure); Services for
   gateway/shard so the orch can dial them for ClockSync; how stable pod DNS (pod-0.svc...) maps to the mesh.
3. VD_PEERS SEEDING: the entrypoint/initContainer that DNS-resolves the peer Service/pod names to literal IPs and
   builds the id=IP:port VD_PEERS string before exec (SocketAddr-only constraint). Decide: baked entrypoint shell in
   the image (debian-slim has sh + getent), an initContainer, or a tiny resolve step. Handle re-resolution on peer
   reschedule (CA-1 update_peer_addr is a later capability; day-1 a fixed roster + retry-on-Down suffices).
4. CONFIG/SECRET: the ConfigMap (shared cloud env) + per-role env; the mTLS Secret (ca.der/node.der/key.der at
   VD_TRUST_DIR, mode 0400); the auth keypair generation + where the gateway VD_AUTH_PUBKEY comes from.
5. STORAGE: the PVC layout (one PVC per pod with subpaths vs split; the durable-root env values; k3d local-path SC).
6. PROBES: concrete livenessProbe + readinessProbe stanzas (httpGet path+port, initialDelaySeconds, periodSeconds,
   timeoutSeconds, failureThreshold, successThreshold) PER ROLE, with the three-inequality math shown; the chosen
   cloud VD_TICK_HZ and terminationGracePeriodSeconds + VD_SHUTDOWN_LINGER_MS per role. Show the numbers satisfy the
   inequalities with margin.
7. NETWORKPOLICY: admit kubelet->probe port; keep VD_ADMIN_ADDR (unauth topology) + the mesh restricted; the QUIC
   mesh among the three pods.
8. JUSTFILE GLUE: k3d-up (cluster create), k3d-load (docker build + k3d image import), k3d-apply (kubectl apply),
   k3d-down, and a DoD check (pods Ready, /metrics reachable, cluster_bootstrapped true) — context-pinned.
9. FILE LAYOUT: where the YAML lives (deploy/k3d/*.yaml per .dockerignore's pre-listing), one file vs kustomize-free
   split, naming.
Call out every DECISION with a rationale, and flag anything that needs a code change (vs pure manifest) or a user
decision (e.g. the cloud VD_TICK_HZ value, whether to actually bring up a live cluster now).
Be concrete: propose actual YAML shapes (not prose), actual numbers, actual entrypoint commands.
`

const EMPHASES = [
  { key: 'k8s-idiomatic', lean: 'Lean k8s-IDIOMATIC + ROBUST: prefer standard patterns (StatefulSet headless Services, initContainer for DNS-resolve, downward-API for VD_NODE_ID from the pod ordinal, ConfigMap+Secret projection, PodDisruptionBudget, proper resource QoS). Optimize for correctness + operability over minimalism.' },
  { key: 'minimal-day1', lean: 'Lean MINIMAL day-1-SIMPLEST-that-is-correct: the fewest objects/moving-parts that satisfy every BINDING constraint for a 1-orch/1-gateway/1-shard k3d dev cluster (agent-HR6 testing only, no HA). Prefer a baked entrypoint over an initContainer if simpler; a single ConfigMap; avoid speculative generality. But NEVER violate a binding constraint (cloud preflight, cold-start cure, probe inequalities) to save an object.' },
  { key: 'failure-first', lean: 'Lean FAILURE-MODE-FIRST: design backwards from what breaks — cold-start deadlock, a pod rescheduled to a new IP mid-run (stale VD_PEERS), a probe that restarts a healthy-but-slow node, a SIGKILL mid-fsync, an emptyDir masquerading as a durable root, the unauth admin/probe surface, a DNS-resolve race at boot. For each failure, show the manifest feature that prevents it. Prioritize the three probe inequalities + publishNotReadyAddresses + the durable-root guards.' },
]

const DESIGN_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['emphasis', 'workloads', 'services', 'vd_peers_seeding', 'config_secret', 'storage', 'probes', 'networkpolicy', 'justfile_glue', 'file_layout', 'code_changes_needed', 'user_decisions', 'key_risks'],
  properties: {
    emphasis: { type: 'string' },
    workloads: { type: 'string', description: 'StatefulSet/Deployment per role + securityContext + QoS + replicas, with YAML shapes' },
    services: { type: 'string', description: 'headless orch Service + publishNotReadyAddresses + gateway/shard Services' },
    vd_peers_seeding: { type: 'string', description: 'the concrete DNS-resolve mechanism + entrypoint/initContainer commands' },
    config_secret: { type: 'string', description: 'ConfigMap + per-role env + mTLS Secret + auth keypair handling' },
    storage: { type: 'string', description: 'PVC layout + durable-root env values + storageclass' },
    probes: { type: 'string', description: 'per-role probe stanzas with the 3-inequality math + chosen VD_TICK_HZ + grace/linger' },
    networkpolicy: { type: 'string' },
    justfile_glue: { type: 'string', description: 'k3d-up/load/apply/down + DoD check recipes' },
    file_layout: { type: 'string' },
    code_changes_needed: { type: 'array', items: { type: 'string' }, description: 'anything requiring a code change vs pure manifest' },
    user_decisions: { type: 'array', items: { type: 'string' } },
    key_risks: { type: 'array', items: { type: 'string' } },
  },
}

const CRIT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['emphasis', 'violations', 'weaknesses', 'best_ideas', 'verdict'],
  properties: {
    emphasis: { type: 'string' },
    violations: { type: 'array', items: { type: 'string' }, description: 'binding-constraint violations or correctness bugs' },
    weaknesses: { type: 'array', items: { type: 'string' } },
    best_ideas: { type: 'array', items: { type: 'string' }, description: 'the parts worth keeping in synthesis' },
    verdict: { type: 'string' },
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

phase('Critique')
const critiques = await parallel(
  valid.map((d) => () =>
    agent(
      `${GROUND}\n\nADVERSARIALLY CRITIQUE this S4b manifest design (emphasis '${d._key}'). Find BINDING-CONSTRAINT
VIOLATIONS (cloud preflight missing a required env; VD_PEERS given a DNS name; no publishNotReadyAddresses on the
orch Service; a probe stanza that violates one of the three inequalities; terminationGracePeriodSeconds < drain; an
emptyDir where a durable root is required; an unauth surface left open), correctness bugs, and DNS-resolve races.
Then name the BEST IDEAS worth keeping. Be specific + cite the binding constraint each violation breaks.

DESIGN UNDER REVIEW (JSON):
${JSON.stringify(d, null, 1)}`,
      { label: `critique:${d._key}`, phase: 'Critique', schema: CRIT_SCHEMA, agentType: 'Explore', model: 'opus' },
    ),
  ),
)

phase('Adjudicate')
const plan = await agent(
  `${GROUND}\n${BRIEF}\n\nYou are the ADJUDICATOR. Synthesize ONE vetted, buildable S4b plan from the ${valid.length}
designs + their adversarial critiques below. Resolve every flagged violation, graft the best ideas from each, and
produce a CONCRETE plan an implementer can author directly: the exact file list under deploy/k3d/, the YAML shapes
(workloads/services/configmap/secret/pvc/networkpolicy), the entrypoint DNS-resolve command, the per-role probe
stanzas WITH the three-inequality math and the chosen cloud VD_TICK_HZ + grace/linger numbers, the justfile k3d-*
recipes, the list of any CODE changes needed (vs pure manifest), and the explicit USER DECISIONS to surface (call
out the cloud VD_TICK_HZ choice and whether to actually bring up a LIVE k3d cluster in this environment now vs
author + statically validate and defer the live bring-up). Flag any residual risk.

DESIGNS:
${JSON.stringify(valid, null, 1)}

CRITIQUES:
${JSON.stringify(critiques.filter(Boolean), null, 1)}`,
  { label: 'adjudicate', phase: 'Adjudicate', model: 'opus' },
)

return { plan, designs: valid, critiques: critiques.filter(Boolean) }
