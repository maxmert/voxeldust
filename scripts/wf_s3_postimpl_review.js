export const meta = {
  name: 's3-postimpl-review',
  description: 'Adversarial post-impl review of the S3 (cloud-ready k3d) probe/health slice',
  phases: [
    { title: 'Find', detail: '5 read-only dimension reviewers over the S3 diff' },
    { title: 'Verify', detail: 'adversarially refute each finding' },
  ],
}

// The S3 slice under review (uncommitted, HEAD=2026903). Read-only Explore reviewers.
const CONTEXT = `
CONTEXT — Voxeldust greenfield rebuild, worktree /Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system.
This is the S3 slice of the cloud-ready-k3d phase: k8s liveness/readiness probes on all THREE node bins
(orchestrator, gateway, shard). It is UNCOMMITTED (working tree; HEAD is 2026903 the Slice-2 cloud profile).

The change surface (read these files in the working tree directly):
  - crates/node/src/health.rs         NEW  — Tier-A PURE decision surface (predicates: is_live, is_confirmed_fresh,
                                             shard_authority_ready, shard_ready, gateway_ready, orch_ready, health_report;
                                             ProbeTuning::stall_deadline/validate). Duration/bool/u64 only; Instant is
                                             clippy-BANNED in vd-node (even in tests). Must be 100% region+branch.
  - crates/io-prod/src/probe.rs        NEW  — Tier-B axum GLUE: HealthSource trait, PublishedHealth (the ONE Instant::now
                                             staleness ref, ops-plane), HealthState/HealthCell ArcSwap, publish_tick,
                                             publish_draining (SIGTERM edge), probe_router (/healthz + /readyz, bare status).
  - crates/bins/src/lib.rs             MOD  — spawn_probe_server, install_shutdown_flag_with_health (drain-edge publish),
                                             resolve_probe (VD_PROBE_ADDR + VD_PROBE_STALL_TICKS), resolve_shutdown_linger
                                             (VD_SHUTDOWN_LINGER_MS), ClusterAddrs probe fields, http_get_status.
  - crates/bins/src/bin/{shard,gateway,orchestrator}.rs  MOD — per-role readiness publish in the tick loop + probe wiring.
  - crates/devproto/src/lib.rs         MOD  — probe port offsets (RESERVED_NODE_PORTS 4->7; PROBE_*_OFFSET 4/5/6).
  - crates/bins/tests/probe_endpoints.rs  NEW — 4 process tests (converge-ready, orch-alone-ready bootstrap,
                                             sigterm-de-route, partitioned-shard-notready-stays-live).

The k8s contract semantics that MUST hold:
  - /healthz 503 => kubelet RESTARTS the pod. Only a detected RUNNING WEDGE (frozen tick loop) may 503 here.
    A booting/syncing/partitioned/draining node is LIVE (200). Draining MUST stay live (never a SIGKILL mid-fsync).
  - /readyz 503 => kubelet DE-ROUTES (removes from Service endpoints), pod keeps running. Booting, clock-unsynced,
    partitioned/self-fenced, session-cap-full, or draining => NotReady.
  - The orchestrator MUST be Ready without any shard (bootstrap-deadlock avoidance).
  - A partitioned shard (lease lapsed, self-fenced) MUST go NotReady (de-route) but STAY live (no restart storm).

Hard rules in force: HR1 (a probe never touches the sim World; probe_router is SEPARATE from the auth-gated admin_router;
the bare status code must leak NO cluster state to the unauth kubelet). HR3 (ONE probe_router for all 3 bins; per-role
polymorphism is the HealthSource trait / injected predicate, NEVER a match on a node kind; config-axis not fork).
HR5 (Tier-A vd-node 100% region+branch; io-prod Tier-B ratcheted floor; exemptions only via cfg_attr/toml).

Report ONLY real defects in THIS slice. For each: severity (CRITICAL/HIGH/MEDIUM/LOW), file:line, the concrete failure
scenario (inputs -> wrong output -> operational consequence), and the minimal fix. Do NOT report style, and do NOT
report pre-existing issues outside the S3 surface. If a dimension is clean, say so explicitly.
`

const DIMS = [
  {
    key: 'k8s-semantics',
    prompt: `${CONTEXT}
DIMENSION: k8s liveness/readiness CONTRACT correctness. Scrutinize:
- Is the /healthz vs /readyz mapping exactly right (503-liveness=restart vs 503-readiness=de-route)? Any inversion?
- Is drain-safe liveness correct — does 'draining' truly force live=true so a graceful shutdown is never restarted,
  AND does the drain window outlast the final fsync (VD_SHUTDOWN_LINGER_MS)? Could a kubelet SIGKILL land mid-fsync?
- Bootstrap: is the orchestrator Ready with zero shards? Is any role's readiness transitively gated on another role
  being up (a deadlock/thundering-herd on cold start)?
- Partition: does a self-fenced shard go NotReady but stay live? Is the stall_deadline (liveness wedge timeout) safely
  LARGER than a normal partition/clock-desync window, so a partitioned-but-healthy node is never restarted?
- The stall_deadline vs real k8s probe cadence (periodSeconds x failureThreshold): is ProbeTuning DEFAULT (20 ticks)
  sane, and is the relationship documented so S4 manifests can't set a probe that restarts a healthy node?
Read health.rs, probe.rs, the three bins, and probe_endpoints.rs.`,
  },
  {
    key: 'concurrency-race',
    prompt: `${CONTEXT}
DIMENSION: concurrency / the ArcSwap publish race. Scrutinize:
- publish_tick reads cell.load().draining then stores a new state. publish_draining (from the SIGNAL-HANDLER task,
  a DIFFERENT thread) also load-then-stores. Is there a lost-update / TOCTOU where a concurrent publish_tick can
  OVERWRITE the draining=true that publish_draining just set, un-draining the pod (which would re-route it into the
  Service mid-shutdown)? Walk the exact interleaving. Is 'draining monotone' actually guaranteed, or only usually?
- The heartbeat staleness detector: if step_tick wedges, does 'at' freeze so PublishedHealth.report() eventually
  reports not-live? Any path where a wedged loop still refreshes 'at'? Is Instant::now().saturating_duration_since
  correct across a clock that only moves forward?
- Is the probe HTTP task truly lock-free vs the tick loop (no shared Mutex, no torn read of HealthState)?
- Ordering: on the SIGTERM edge, is de-route published BEFORE the drain/exit begins, with enough margin for a kubelet
  readiness poll to observe the 503 before endpoints removal? Could the pod exit before any /readyz poll sees NotReady?
Read probe.rs (publish_tick/publish_draining/PublishedHealth), bins/src/lib.rs (install_shutdown_flag_with_health),
and the three bins' tick loops.`,
  },
  {
    key: 'predicate-correctness',
    prompt: `${CONTEXT}
DIMENSION: the PURE predicates in crates/node/src/health.rs. Prove each correct or find the bug:
- is_confirmed_fresh(local_tick, confirmed, grace) = (grace != 0) & (local_tick.saturating_sub(confirmed) <= grace).
  Off-by-one on <= vs <? Behaviour at grace==0 (should mean 'no lease liveness signal')? Overflow/saturation?
- shard_authority_ready(held, local, confirmed, grace) = is_confirmed_fresh(...) | ((grace==0) & held). Is the
  grace==0 fallback (inert DevTest: no self-fence armed => ready iff held) correct AND does the ARMED path
  (grace!=0) correctly IGNORE held so a partitioned holder with a stale confirmed goes NotReady? Any case where an
  armed shard that LOST authority still reads ready?
- gateway_ready(clock_synced, session_count, max_sessions) = clock_synced & (session_count < max_sessions). Off-by-one:
  at exactly max_sessions should it be NotReady (correct: full => de-route new sessions)? What if max_sessions==0?
- shard_ready / orch_ready / is_live / health_report: any branch that yields a wrong bool. Check the boolean & vs &&
  (bitwise is deliberate for branch-coverage; confirm no short-circuit-dependent side effect is lost).
- ProbeTuning::stall_deadline (Duration::from_secs(1)/tick_hz.max(1) * stall_deadline_ticks.max(1)) — integer division
  truncation at high tick_hz making the deadline too SMALL (e.g. hz=50 => 20ms/tick*20=400ms wedge window — is that
  enough vs a GC pause)? validate() rejecting < 2 — is 2 ticks a safe floor? Does the const-assert match validate()?
Read health.rs in full including its unit tests (confirm the tests actually pin these edges, not just happy paths).`,
  },
  {
    key: 'hr-compliance',
    prompt: `${CONTEXT}
DIMENSION: hard-rule + seam compliance. Scrutinize:
- HR3: is there ANY match/if on a node-kind inside the probe/health/readiness path? The per-role readiness MUST be an
  injected predicate (each bin computes its own gateway_ready/shard_ready/orch_ready and publishes a bool) with ONE
  shared probe_router + ONE HealthSource trait. Confirm no forked router, no kind enum switch.
- HR1: does probe.rs or any readyz path touch the sim World, rapier, or redb directly? It must read ONLY the ArcSwap
  cell the tick loop publishes. Confirm probe_router is SEPARATE from admin_router. Confirm the bare status code cannot
  distinguish partitioned-vs-booting-vs-full to an UNAUTH kubelet (no state leak; that distinction must live behind
  auth-gated /metrics only).
- Seam: Instant must NOT appear in vd-node (health.rs) — confirm all node predicates take Duration/u64/bool and the
  single Instant::now lives in io-prod. Any disallowed-method/type creep?
- HR5: is health.rs plausibly 100% region+branch (no untested branch, no unreachable arm)? Any coverage exemption added
  — is it justified? Is the Tier-B probe.rs surface floored?
Read health.rs, probe.rs, io-prod/src/lib.rs, node/src/lib.rs, and grep the bins for any 'match' on node kind near probes.`,
  },
  {
    key: 'ops-plumb',
    prompt: `${CONTEXT}
DIMENSION: ops plumbing — env, ports, dev cluster, S4-readiness. Scrutinize:
- devproto probe port offsets: RESERVED_NODE_PORTS 4->7, PROBE_ORCHESTRATOR/GATEWAY/SHARD_OFFSET 4/5/6. Any collision
  with existing offsets (admin/ops/mesh/quic/dev_control)? Slot overflow at the last slot (does base+6 exceed 65535 or
  collide with the next slot's block)? Confirm the layout doc + tests match the code.
- resolve_probe(env): VD_PROBE_ADDR optional (None => no probe server). Is a MISSING probe addr in a CLOUD profile a
  silent no-op (a pod with no probes k8s treats as always-healthy — dangerous)? Should cloud profile REQUIRE it?
  VD_PROBE_STALL_TICKS parse: bad value handling (loud error vs silent default)?
- resolve_shutdown_linger(env): VD_SHUTDOWN_LINGER_MS default 0. In dev the tests set 1500; in cloud is 0 safe given
  the fsync park? Is the linger applied AFTER de-route but BEFORE the final flush, or could it delay past the k8s
  terminationGracePeriod and get SIGKILLed?
- ClusterAddrs: are the 3 probe fields wired into orchestrator_env/gateway_env/shard_env for the dev devcluster so the
  dev cluster actually binds probe ports? Is there a role that spawns WITHOUT a probe addr?
- Is there a missing write_env_contract VD_PROBE_* emission that S4 manifests will need? Flag it as a gap if so.
Read devproto/src/lib.rs, bins/src/lib.rs (resolve_probe/resolve_shutdown_linger/ClusterAddrs/*_env), vd-devcluster.rs.`,
  },
]

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['dimension', 'clean', 'findings'],
  properties: {
    dimension: { type: 'string' },
    clean: { type: 'boolean', description: 'true if no real defect found' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'location', 'scenario', 'fix'],
        properties: {
          severity: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'] },
          location: { type: 'string', description: 'file:line' },
          scenario: { type: 'string', description: 'inputs -> wrong output -> operational consequence' },
          fix: { type: 'string', description: 'the minimal concrete fix' },
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
    refuted: { type: 'boolean', description: 'true if the finding is NOT a real defect' },
    reason: { type: 'string', description: 'evidence from the code for the verdict' },
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
          `${CONTEXT}
ADVERSARIALLY VERIFY this claimed defect from the '${d.key}' review. Try HARD to REFUTE it — read the actual code at
the cited location and the surrounding logic. Default to refuted=true unless the code plainly exhibits the failure.
A finding is INVALID if: the scenario cannot actually occur (guard exists elsewhere), the 'wrong output' is actually
correct for the k8s contract, it is pre-existing/out-of-S3-scope, or it is style not a defect.

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

const cleanDims = DIMS.map((d) => d.key).filter(
  (k) => !confirmed.some((f) => f.dim === k),
)

return {
  confirmed_defects: confirmed.map((f) => ({
    dim: f.dim,
    severity: f.verdict.severity_final,
    location: f.location,
    scenario: f.scenario,
    fix: f.fix,
    why_real: f.verdict.reason,
  })),
  clean_dimensions: cleanDims,
  total_confirmed: confirmed.length,
}
