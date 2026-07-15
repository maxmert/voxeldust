export const meta = {
  name: 's4-understand',
  description: 'Map the exact ground truth needed to design S4 (k3d manifests) correctly',
  phases: [{ title: 'Map', detail: '5 parallel readers over image/env/probe/peer/storage' }],
}

const CTX = `
CONTEXT — Voxeldust greenfield rebuild, worktree /Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system.
We are about to design Slice 4 (S4) of the cloud-ready-k3d phase: the k3d/k8s MANIFESTS + supporting glue that
deploy the three server bins (orchestrator, gateway, shard) into a real k3d cluster. S0 (server image), S1
(SIGTERM graceful drain), S2 (cloud config profile + split-brain fix), S3 (k8s /healthz + /readyz probes) are
DONE + committed (HEAD e011989). Approved S4 tool stack (already user-ratified): Docker + k3d + kubectl + RAW YAML
manifests + a debian-slim/glibc release image; day-1 = servers-on-k3d + agent-HR6 testing (human play stays a
local/port-forward path). This is a READ-ONLY mapping task — do NOT modify anything. Report ground truth with
exact file:line citations and verbatim env-var / port / knob names. If something does NOT exist, say so plainly
(that is itself a finding). Be exhaustive and precise — this map is the sole input to the S4 design.
`

const READERS = [
  {
    key: 'image-entrypoint',
    prompt: `${CTX}
MAP the server IMAGE + entrypoint + build tooling. Answer concretely:
- The Dockerfile(s): path, base image, build stages, what binaries are copied in, the ENTRYPOINT/CMD, how ONE
  image dispatches orchestrator vs gateway vs shard (an arg? an env? separate images?). Quote the exact dispatch.
- The justfile image/k3d targets (image-build, image-drain-smoke, any k3d-*): what they do, exact recipe bodies.
- Is there an entrypoint SHELL script in the image (for DNS-resolve / env munging), or does the image exec the
  bin directly? If a script exists, quote it.
- How is VD_PEERS currently provided in the docker/image smoke (the S0/S1 smoke)? Literal addrs? Not set?
- The release profile ([profile.release] in Cargo.toml): opt/lto/strip settings, and whether the server bins
  pull in Bevy/wgpu (they must NOT — render is a separate feature/image).
Search: Dockerfile*, justfile, crates/bins/src/bin/*.rs main dispatch, Cargo.toml, scripts/, any *.sh.`,
  },
  {
    key: 'env-contract',
    prompt: `${CTX}
MAP the EXACT boot-time env contract of EACH of the three server bins. For orchestrator, gateway, AND shard,
enumerate EVERY VD_* environment variable the bin reads at startup, marking each REQUIRED vs OPTIONAL(+default),
and what it controls. Include (verbatim names): the bind/listen addrs (which VD_*_ADDR / VD_BIND each binds),
VD_TRUST_DIR (mTLS bundle), VD_AUTH_* (signing/pubkey), VD_PROFILE, VD_STORE_DURABLE_ROOT, VD_BOOT_DURABLE_ROOT,
any OUTBOX root, VD_PROCESS_INCARNATION, VD_PEERS, VD_PROBE_ADDR, VD_PROBE_STALL_TICKS, VD_SHUTDOWN_LINGER_MS,
VD_SELF_FENCE_GRACE / VD_REALM_RECHECK / VD_LEASE_* / VD_REAPER_* and the D-3 knobs, tick-hz, and any others.
Cite the resolve_* fns in crates/bins/src/lib.rs and each bin's main(). Produce a per-role table: var | req? |
default | meaning. This becomes the ConfigMap/Secret/env-block of the manifests, so COMPLETENESS is critical —
a missed required var = a pod that won't boot.
Search: crates/bins/src/lib.rs (resolve_*, *_env helpers), crates/bins/src/bin/{orchestrator,gateway,shard}.rs,
crates/io-prod/src/{boot.rs,runtime.rs,trust.rs}.`,
  },
  {
    key: 'probe-timing',
    prompt: `${CTX}
MAP the probe + drain TIMING so the S4 manifest's livenessProbe/readinessProbe/terminationGracePeriodSeconds
can be sized CORRECTLY. Answer with concrete numbers:
- ProbeTuning::stall_deadline derivation (crates/node/src/health.rs): the formula, the DEFAULT stall_deadline_ticks
  (=20?), and the resulting wall-clock deadline at the shard/gateway/orch tick_hz (what IS each role's tick_hz, and
  where set?). VD_PROBE_STALL_TICKS override + the validate() floor.
- The S1 SIGTERM drain worst case: what is the longest a bin can take to exit after SIGTERM (the final-fsync park;
  RedbStore::Drop join)? VD_SHUTDOWN_LINGER_MS + resolve_shutdown_linger (crates/bins/src/lib.rs).
- The k8s contract inequalities the manifest MUST satisfy: liveness (periodSeconds × failureThreshold) MUST be
  >> stall_deadline (so a slow-but-progressing node is never restarted) AND terminationGracePeriodSeconds MUST be
  >= the drain worst case (so a draining pod is never SIGKILLed mid-fsync) AND > the readiness de-route lead time.
- PRODUCE a concrete recommended stanza (initialDelaySeconds, periodSeconds, timeoutSeconds, failureThreshold,
  successThreshold) for BOTH liveness and readiness, PER ROLE, with the inequality math shown. Flag any knob that
  is not currently env-settable and would need to be.
Search: crates/node/src/health.rs, crates/io-prod/src/probe.rs, crates/bins/src/lib.rs, the bins' tick_hz.`,
  },
  {
    key: 'peer-bootstrap',
    prompt: `${CTX}
MAP peer ADDRESSING + cluster BOOTSTRAP so the S4 Services + entrypoint can wire the mesh with NO DNS gap and NO
cold-start deadlock. Answer:
- How is VD_PEERS parsed (crates/io-prod/src/runtime.rs)? SocketAddr only, or does it accept host:port DNS names?
  If SocketAddr-only, the entrypoint must DNS-resolve k8s Service names to IPs before exec — confirm/deny.
- CA-1 reply-on-connection / update_peer_addr / LearnedPeers (crates/io-prod/src/mesh.rs): WHO dials WHOM at
  bootstrap? Does the orchestrator need VD_PEERS at all, or do shards/gateway dial the orch and the orch learns
  their addrs from the inbound connection? Which side is the DIALER, which is the LISTENER? This decides whether
  only the shard/gateway need VD_PEERS (pointing at the orch Service) or a full mesh peer list is required.
- The bootstrap-deadlock: a shard must dial the orchestrator to register, but the orchestrator's readiness
  (S3) reads NotReady until it is serving — does the orch mesh Service need publishNotReadyAddresses:true so its
  DNS/endpoints resolve while it is itself NotReady? Confirm the exact ordering (who must be reachable before who
  is Ready). Cite the registration path (shard→orch LeaseGrant / register).
- The StatefulSet stable-DNS pattern (pod-N.headless-svc.ns.svc.cluster.local) vs a Deployment: which bins need
  STABLE identity (orch? shards?) and which can be a plain Deployment (gateway)?
Search: crates/io-prod/src/{runtime.rs,mesh.rs}, crates/bins/src/bin/*.rs, crates/wire/ registration/admission.`,
  },
  {
    key: 'storage-gwready',
    prompt: `${CTX}
MAP (a) the durable STORAGE each bin needs (→ PVCs) and (b) the FEASIBILITY of the named S4 code prerequisite:
gateway partition-aware readiness. Answer:
(a) STORAGE: which durable roots does EACH bin write (VD_STORE_DURABLE_ROOT = redb directory+saga WAL+clock
    ceiling; VD_BOOT_DURABLE_ROOT = M3 monotone boot-counter; any OUTBOX durable root)? Which bins need a PVC and
    for WHAT (orchestrator: directory/saga/clock/boot-counter; shard: outbox + boot-counter?; gateway: outbox +
    boot-counter?)? Are the roots separate dirs or can they share one PVC with subpaths? Cite where each is read
    + written (crates/io-prod/src/{boot.rs,store.rs,outbox.rs}).
(b) GATEWAY READINESS: S3's gateway_ready is PARTITION-BLIND — the doc names a fix any_session_confirmed_within(grace).
    Investigate whether the data for it EXISTS: does GatewaySessions (or its per-session state) already track a
    per-session confirmed / last-heard LOCAL tick that a predicate could compare against grace? Where is
    GatewaySessions defined + updated (the session's last-authoritative-contact tick)? Is the follower clock the
    only liveness signal a gateway has, or is there a per-session heartbeat? Assess: is the partition-aware fix
    SMALL (data already present, add a predicate) or LARGE (must add per-session tick tracking + plumbing)? This
    decides whether S4 folds the gateway-readiness fix in or defers it as its own slice.
Search: crates/sim|node/ GatewaySessions, crates/io-prod/src/{store.rs,outbox.rs,boot.rs}, the gateway bin.`,
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
      description: 'concrete facts with file:line citations',
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
    gaps: {
      type: 'array',
      description: 'things that do NOT exist yet and S4 must add (code/shell/manifest)',
      items: { type: 'string' },
    },
    recommendation: {
      type: 'string',
      description: 'the dimension-specific recommendation for S4 (topology/numbers/decision)',
    },
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
