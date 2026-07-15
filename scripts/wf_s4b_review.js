export const meta = {
  name: 's4b-review',
  description: 'Adversarial review of S4b (k3d manifests + entrypoint + 3 code changes)',
  phases: [
    { title: 'Find', detail: '3 read-only dimension reviewers' },
    { title: 'Verify', detail: 'adversarially refute each finding' },
  ],
}

const CTX = `
CONTEXT — Voxeldust greenfield rebuild, worktree /Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system.
S4b (cloud-ready k3d Slice 4b) authors the k3d/k8s deployment for the three server bins. UNCOMMITTED on HEAD
9a73c0a. It was authored from a vetted design (a judge-panel workflow); this review adversarially checks the
IMPLEMENTATION against the binding constraints. Read the working tree:
  - deploy/k3d/00-namespace.yaml, 10-configmap.yaml, 20-networkpolicy.yaml, 30-orch.yaml, 40-gateway.yaml, 50-shard.yaml
  - docker/entrypoint.sh (NEW) + docker/server.Dockerfile (added the entrypoint COPY, still NO ENTRYPOINT/CMD)
  - crates/bins/src/lib.rs (NEW validate_tick_pair + TickPairError + auth_keypair_hex_from_seed + tests)
  - crates/bins/src/bin/shard.rs (hoisted VD_TICK_HZ/VD_TICK_DT reads + validate_tick_pair call)
  - crates/bins/src/bin/vd-devcluster.rs (NEW gen-authkey subcommand reading /dev/urandom)
  - justfile (NEW k3d-validate/up/load/secrets/apply/down/dod/all recipes)

BINDING CONSTRAINTS (from the verified understand + design; a violation is a defect):
- Chosen: VD_TICK_HZ=50 (stall_deadline = (1s/50)*20 = 400ms). Bins read ALL config from process env only.
- VD_PEERS is SocketAddr-ONLY (id=IP:port; a DNS name fails at boot; an empty book silently = solo node). The
  entrypoint DNS-resolves the 3 headless-Service pod names to literal IPv4 and builds the role's book, then execs.
- Cold-start: bootstrap is BIDIRECTIONAL (shard dials orch to register; orch dials shard+gateway for ClockSync).
  shard/gateway readiness gates on clock_synced. The orch (and all 3) Services MUST be headless +
  publishNotReadyAddresses:true so a NotReady peer's A-record still resolves at entrypoint time.
- Cloud profile (VD_PROFILE=cloud) fail-loud REQUIRES per pod: VD_STORE_DURABLE_ROOT + VD_BOOT_DURABLE_ROOT +
  VD_BOOT_STATE_DIR + VD_PROBE_ADDR (all 3); orch also VD_STORE_PATH; gateway a REAL non-dev VD_AUTH_PUBKEY.
  FORBIDS VD_*_EPHEMERAL_OK + manual VD_PROCESS_INCARNATION + any inert D-3. Do NOT set D-3 knobs (derived).
- REQUIRED env each bin: VD_NODE_ID, VD_BIND(0.0.0.0:9000 QUIC/UDP), VD_PEERS(entrypoint), VD_OUTBOUND_CAP,
  VD_TRUST_DIR, VD_TICK_HZ. Orch: VD_ADMIN_ADDR, VD_EPOCH, VD_RESERVE_CHUNK, VD_CLOCK_PEERS, VD_STORE_PATH.
  Gateway: VD_ORCH, VD_SHARD, VD_AUTH_PUBKEY, VD_SESSION_SEED, VD_MAX_SESSIONS, VD_MAX_BUFFERED_INPUTS. Shard:
  VD_REALM_SEED, VD_SPEED, VD_TICK_DT(=1/hz), VD_ORCH, VD_MINT_SEED, VD_INPUT_LOG_CAP, VD_SNAPSHOT_BUDGET(<=
  CONSERVATIVE_DATAGRAM_BUDGET). NodeIds: orch=1, gateway=2, shard=3.
- PROBE inequalities: (1) liveness periodSeconds*failureThreshold >> stall_deadline(400ms); (2)
  terminationGracePeriodSeconds >= drain worst case (orch holds un-fsynced redb, ~few s; gw/shard ~linger+teardown);
  (3) grace > readiness de-route lead. /healthz stays LIVE through drain (draining|is_live) so only grace protects
  the fsync. VD_SHUTDOWN_LINGER_MS=2000.
- STORAGE: per-pod PVCs (volumeClaimTemplates) for the M3 boot-counter (monotone identity) — NEVER an emptyDir at a
  durable mount (would pass check_durable_path but wipe the counter on reschedule = the R-6a dedup-loss landmine).
- exec "$@" in the entrypoint is load-bearing (SIGTERM must reach the bin as PID 1, or SIGKILL-past-grace corrupts
  the orch's un-fsynced redb).
- The dev image-smoke/drain-smoke run the bin DIRECTLY (docker run ... vd-orchestrator, literal VD_PEERS=) — the
  entrypoint must not break them (no ENTRYPOINT/CMD kept; command overrides).

Report ONLY real defects. Severity, file:line, concrete failure (what breaks at apply/boot/run), minimal fix. If a
dimension is clean, say so.
`

const DIMS = [
  {
    key: 'manifests',
    prompt: `${CTX}
DIMENSION: manifest correctness (deploy/k3d/*.yaml). Scrutinize:
- ENV COMPLETENESS per role: is EVERY required VD_* present (ConfigMap envFrom + per-role env) and is nothing
  FORBIDDEN set (no VD_*_EPHEMERAL_OK, no VD_PROCESS_INCARNATION, no D-3 knob)? Cross-check against
  crates/bins/src/lib.rs orchestrator_env/gateway_env/shard_env + the bins' actual env.parse calls. A MISSING
  required var = the pod CrashLoops at boot (ConfigError::Missing) — enumerate any gap. Is VD_LEASE_TTL correctly
  OMITTED (cloud derives it) — verify boot.rs resolve_d3 defaults it, so its absence does not CrashLoop the orch?
- PROBE MATH at VD_TICK_HZ=50: do the liveness (period 5 * failure 6 = 30s vs stall 400ms) and readiness stanzas
  satisfy the three inequalities? Is terminationGracePeriodSeconds (orch 30, gw/shard 15) >= the drain worst case
  AND > the readiness de-route lead? Any stanza that would restart a healthy-but-slow node or SIGKILL mid-fsync?
- COLD-START: is publishNotReadyAddresses:true on ALL THREE headless Services? clusterIP:None? Any Service that
  would withhold a NotReady peer's A-record and deadlock entrypoint DNS-resolve?
- STORAGE: per-pod volumeClaimTemplates (not a shared/emptyDir) for boot + data? Is any durable mount backed by an
  emptyDir? Do the durable-root env values (/var/lib/vd, /var/lib/vd-boot, /var/lib/vd-boot/state) match the mounts
  + satisfy check_durable_path (temp-disjoint)? Does gateway/shard correctly still mount a data PVC (cloud requires
  VD_STORE_DURABLE_ROOT unconditionally)?
- SECURITY: readOnlyRootFilesystem:true — does anything the bin or entrypoint writes land OUTSIDE the PVCs/tmp
  emptyDir (which would EACCES)? runAsUser/fsGroup 10001 match the image? Any securityContext that blocks the bind
  or the probe?
Read all deploy/k3d/*.yaml + crates/bins/src/lib.rs + the three bin mains + crates/io-prod/src/boot.rs (resolve_d3).`,
  },
  {
    key: 'entrypoint-dns',
    prompt: `${CTX}
DIMENSION: the entrypoint + DNS (docker/entrypoint.sh + the Dockerfile COPY + Service DNS). Scrutinize:
- DNS NAMES: are the resolved names correct for a headless StatefulSet Service? The entrypoint resolves
  vd-orch-0.vd-orch.<ns>.svc.cluster.local etc. Confirm: pod-0 name = <statefulset>-0, subdomain = the serviceName
  (vd-orch/vd-gateway/vd-shard), namespace = voxeldust. Any mismatch = never resolves = 90s FATAL.
- getent ahostsv4: does it correctly return the pod IPv4? Is the parse (awk first field) robust to getent's output
  format? Would an IPv6-preferring resolver break the id=IP:port book (VD_BIND is 0.0.0.0 = v4)?
- NodeId MAP: orch=1, gateway=2, shard=3, and the per-role VD_PEERS omits self + lists the other two with :9000.
  Verify the mapping + the port match VD_BIND. Any role that books the wrong peer id?
- exec "$@": is the final exec correct so the bin is PID 1 (SIGTERM path)? Does 'set -eu' + the resolve/guard
  interact badly (e.g. a resolve failure under 'set -e' aborting before the FATAL message)? Is the empty/mis-shaped
  VD_PEERS guard correct (does it actually reject an empty book)?
- readOnlyRootFilesystem vs the shell: does entrypoint.sh write anything to the rootfs (temp files, redirections)?
  getent/awk/sleep/echo-to-stderr — all read-only? Confirm the shell runs under readOnlyRootFilesystem:true.
- Dockerfile: is the COPY + chmod correct (mode 0755, path /usr/local/bin/entrypoint.sh on PATH)? Does keeping NO
  ENTRYPOINT/CMD preserve the direct-bin docker-run smokes (image-smoke passes vd-orchestrator as the command)?
Read docker/entrypoint.sh, docker/server.Dockerfile, deploy/k3d/30-orch.yaml (+gw/shard) Services, justfile image-smoke.`,
  },
  {
    key: 'code-changes',
    prompt: `${CTX}
DIMENSION: the 3 Rust/tooling code changes. Scrutinize:
- validate_tick_pair (crates/bins/src/lib.rs) + its wiring in shard.rs: is the epsilon check correct? Does the
  hoist of VD_TICK_HZ/VD_TICK_DT reads (moved BEFORE register_stub_shard, removed the later duplicate read) change
  ANY behaviour or break a borrow/order? Does the shard still read tick_dt into the StubConfig correctly? Is the
  degenerate tick_hz=0 handled the same as TickPacer? Coverage: do the tests hit both the Ok and Err arms + the
  epsilon boundary?
- gen-authkey (crates/bins/src/bin/vd-devcluster.rs + auth_keypair_hex_from_seed in lib.rs): does the seed→pubkey
  derivation MATCH what the gateway verifies (SigningKey::from_bytes(seed).verifying_key() == the VD_AUTH_PUBKEY the
  gateway's admission checks)? Confirm a gen-authkey key would actually PASS the cloud dev-key veto + be accepted by
  the gateway. Is the /dev/urandom read robust (error handling, exactly 32 bytes)? Does it print the SIGNING seed to
  stdout — is that an acceptable leak for a dev tool (it must go to clients out-of-band, never a server Secret)? Any
  risk it lands in a log/Secret?
- Dockerfile COPY + justfile k3d recipes: does k3d-secrets correctly build the vd-auth Secret from VD_AUTH_PUBKEY_HEX
  and vd-mtls from gen-trust? Is the context-pin correct? Does k3d-validate actually validate offline (no cluster)?
  Any recipe that would silently no-op or fail?
- Regression: does the shard.rs change keep the existing shard tests + process-parity passing (the env contract
  unchanged)? Does adding validate_tick_pair reject the DEV config (VD_TICK_HZ=50, VD_TICK_DT=0.02) — it must PASS?
Read crates/bins/src/lib.rs (the 2 new fns + tests), crates/bins/src/bin/shard.rs, crates/bins/src/bin/vd-devcluster.rs, justfile k3d recipes.`,
  },
]

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['dimension', 'clean', 'findings'],
  properties: {
    dimension: { type: 'string' },
    clean: { type: 'boolean' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'location', 'scenario', 'fix'],
        properties: {
          severity: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'] },
          location: { type: 'string' },
          scenario: { type: 'string' },
          fix: { type: 'string' },
        },
      },
    },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['location', 'refuted', 'reason', 'severity_final'],
  properties: {
    location: { type: 'string' },
    refuted: { type: 'boolean' },
    reason: { type: 'string' },
    severity_final: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INVALID'] },
  },
}

phase('Find')
const reviews = await pipeline(
  DIMS,
  (d) => agent(d.prompt, { label: `find:${d.key}`, phase: 'Find', schema: FINDINGS_SCHEMA, agentType: 'Explore', model: 'opus' }),
  (review, d) => {
    if (!review || review.clean || !review.findings?.length) return { dim: d.key, verified: [] }
    return parallel(
      review.findings.map((f) => () =>
        agent(
          `${CTX}
ADVERSARIALLY VERIFY this claimed S4b defect from the '${d.key}' review. Try HARD to REFUTE it by reading the
actual files. Default refuted=true unless the code/manifest plainly exhibits the failure. INVALID if: the scenario
cannot occur, it is correct for the binding constraint, it is a documented+accepted residual (NetworkPolicy no-op on
flannel; the peer-reschedule manual re-seed; the zero-session gateway blind spot), or it is out-of-S4b scope.
CLAIM: severity=${f.severity} at ${f.location}
SCENARIO: ${f.scenario}
PROPOSED FIX: ${f.fix}`,
          { label: `verify:${d.key}:${f.location}`, phase: 'Verify', schema: VERDICT_SCHEMA, agentType: 'Explore', model: 'opus' },
        ).then((v) => ({ ...f, dim: d.key, verdict: v })),
      ),
    ).then((vs) => ({ dim: d.key, verified: vs.filter(Boolean) }))
  },
)

const confirmed = reviews
  .filter(Boolean)
  .flatMap((r) => r.verified)
  .filter((f) => f.verdict && !f.verdict.refuted)

return {
  confirmed_defects: confirmed.map((f) => ({
    dim: f.dim,
    severity: f.verdict.severity_final,
    location: f.location,
    scenario: f.scenario,
    fix: f.fix,
    why_real: f.verdict.reason,
  })),
  clean_dimensions: DIMS.map((d) => d.key).filter((k) => !confirmed.some((f) => f.dim === k)),
  total_confirmed: confirmed.length,
}
