export const meta = {
  name: 's4a-review',
  description: 'Adversarial review of S4a (gateway partition-aware readiness)',
  phases: [
    { title: 'Find', detail: '2 read-only dimension reviewers' },
    { title: 'Verify', detail: 'adversarially refute each finding' },
  ],
}

const CTX = `
CONTEXT — Voxeldust greenfield rebuild, worktree /Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system.
S4a (part of the cloud-ready-k3d Slice 4) closes the S3 NAMED blocker: gateway partition-aware readiness. It is
UNCOMMITTED on HEAD e011989. The change (read the working tree):
  - crates/node/src/health.rs (Tier-A, MUST stay 100% region+branch): NEW pure fn
      gateway_sessions_live(freshest_active_confirmed: Option<u64>, local_tick: u64, grace: u64) -> bool
        = match { None => true, Some(c) => is_confirmed_fresh(local_tick, c, grace) | (grace == 0) }
    and gateway_ready GAINED a 4th param sessions_live:
        gateway_ready(clock_synced, session_count, max_sessions, sessions_live) = clock_synced & (count < max) & sessions_live
  - crates/connection-plane/src/gateway.rs (Tier-A, 100%): NEW accessor on GatewaySessions
        freshest_active_confirmed(&self) -> Option<u64>
        = by_session.values().filter(|s| matches!(s.phase, SessionPhase::Active{..})).map(|s| s.confirmed_at.0).max()
  - crates/bins/src/bin/gateway.rs: the readiness site now computes sessions_live via gateway_sessions_live(
        sessions.freshest_active_confirmed(), local_tick, self_fence_grace_ticks) and ANDs it into gateway_ready.

DESIGN INTENT: mirror the shard's shard_authority_ready/is_confirmed_fresh partition gate. Session.confirmed_at
is the local_tick of the last Session-head ROUND-TRIP confirmation (the reply affirming THIS gateway still owns
the session lease); it is set when a session goes Active and re-armed on every affirming recheck reply, and
FREEZES under a partition. A TOTAL gateway↔orchestrator(directory) partition freezes EVERY Active session's
confirmed_at in lock-step, so the FRESHEST (max) going stale ⇒ the whole gateway is partitioned ⇒ /readyz de-route.
A single client's shard vanishing must NOT de-route the gateway. Zero Active sessions ⇒ live (rests on clock_synced;
self-corrects on attach — the acknowledged blind spot). Inert grace==0 (DevTest) ⇒ always live (byte-identical to
before). Report ONLY real defects in THIS change. For each: severity, file:line, concrete failure scenario, minimal
fix. If a dimension is clean, say so.
`

const DIMS = [
  {
    key: 'semantics',
    prompt: `${CTX}
DIMENSION: partition-detection SEMANTICS correctness. Scrutinize hard:
- Is "freshest (max) Active confirmed within grace ⇒ live" the RIGHT reduction? Prove that a TOTAL gateway↔orch
  directory partition freezes EVERY Active session's confirmed_at (so max also freezes) — read where confirmed_at
  is SET (Active transition) and RE-ARMED (affirming recheck reply) in crates/connection-plane/src/gateway.rs.
  Is there any path where a session's confirmed_at advances via a DIFFERENT signal than the gw↔orch round-trip
  (so a partitioned gateway could still look fresh)? Is there any path where a HEALTHY gateway's freshest goes
  stale (false de-route) — e.g. a just-attached Active session whose confirmed_at was NOT set at attach, or a
  recheck interval much longer than grace so confirmed_at legitimately lags?
- Is "ANY active fresh ⇒ ready" (max within grace) correct vs "ALL"? Confirm a single client's shard/session
  freezing does NOT de-route the whole gateway (another fresh session keeps it Ready). Confirm the total-partition
  case (ALL frozen) DOES de-route.
- The zero-session None⇒live blind spot: is it correctly bounded (self-corrects once a session attaches + its
  recheck freezes)? Is there a worse hole — e.g. a gateway that SHEDS all sessions on partition (so it flips to
  None⇒live = falsely Ready forever)? Does a partition cause Active sessions to leave the by_session map (become
  None) rather than freeze — which would defeat the detector? Check the self-fence / lapse path: does a self-fenced
  session get REMOVED from by_session or just change phase to SelfFenced (still counted as non-Active ⇒ None)?
- Inert grace==0: confirm DevTest gateways with Active sessions still read sessions_live=true (byte-identical),
  and confirm the cloud path (grace!=0) is the only one that can de-route.
Read crates/connection-plane/src/gateway.rs (confirmed_at set/re-arm, self-fence/lapse, SessionPhase states),
crates/node/src/health.rs, crates/bins/src/bin/gateway.rs.`,
  },
  {
    key: 'coverage-integration',
    prompt: `${CTX}
DIMENSION: HR5 coverage + integration/regression. Scrutinize:
- Does health.rs stay 100% region+branch? The new gateway_sessions_live has a match (None/Some arms) + the inner
  is_confirmed_fresh | (grace==0). Do the added unit tests exercise BOTH match arms AND is_confirmed_fresh true+false
  AND grace==0 true+false? Any uncovered region/branch? Is the 4-param gateway_ready truth table complete
  (clock t/f, capacity </>=, sessions_live t/f)?
- Does the connection-plane accessor freshest_active_confirmed stay 100%? The filter matches! (Active vs non-Active)
  + .max() (empty→None vs non-empty→Some) — does the new test cover the Active-and-non-Active mix, the empty case,
  and the non-active-only case (so the filter's false arm AND the max's None arm are both hit)?
- Integration: is there any OTHER caller of gateway_ready besides the gateway bin + the health tests that the
  4th-param signature change would break? grep the workspace.
- Regression: DevTest process tests (client_load spawns real gateways with Active sessions at grace=0) — do they
  still reach Ready? Confirm grace==0 ⇒ sessions_live=true so DevTest readiness is byte-identical. Any process test
  (probe_endpoints, client_load, dev_cluster_smoke) that asserts gateway readiness and could regress?
- Borrow/perf: the readiness site holds an immutable world.resource::<GatewaySessions>() borrow across
  freshest_active_confirmed() + len(); is that sound? freshest_active_confirmed iterates ALL sessions every tick —
  at MMO scale (hundreds/thousands of sessions) is an O(n) per-tick scan acceptable, or should it be flagged
  (e.g. a maintained max)? Is that a real concern for day-1 or a noted future optimization?
Read crates/node/src/health.rs (tests), crates/connection-plane/src/gateway.rs (the accessor + its test),
crates/bins/src/bin/gateway.rs, and grep for gateway_ready callers + gateway-readiness process tests.`,
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
ADVERSARIALLY VERIFY this claimed defect from the '${d.key}' review of S4a. Try HARD to REFUTE it by reading the
actual code. Default refuted=true unless the code plainly exhibits the failure. INVALID if: the scenario cannot
occur, the behaviour is actually correct for the design intent, it is a pre-existing/out-of-S4a issue, or it is a
documented+acceptable tradeoff (e.g. the zero-session blind spot, or an O(n) scan that is a noted future opt not a
day-1 defect).
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
