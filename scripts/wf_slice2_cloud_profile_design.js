export const meta = {
  name: 'slice2-cloud-profile-design',
  description: 'Design the cloud config profile (enforce split-brain D-3 + drop footguns, fail loud at boot)',
  phases: [
    { title: 'Design', detail: 'one opus designer produces the spec' },
    { title: 'Attack', detail: '3 opus adversaries per dimension' },
    { title: 'Adjudicate', detail: 'fold → SOUND_TO_IMPLEMENT or fixes' },
  ],
}

const CTX = `
VOXELDUST greenfield voxel MMO, cloud-ready k3d phase, SLICE 2. HEAD has S0 (server image) + S1 (SIGTERM
drain) committed. Goal: a CLOUD CONFIG PROFILE that ENFORCES the split-brain-safe D-3 lease-liveness config
+ drops dev footguns, FAILING LOUD at boot if violated — clearing the DEFERRED.md:984 HARD precondition (a
cluster currently boots GREEN with NO split-brain protection).

CRITICAL GROUNDED FACTS (verify against the code):
- The ENTIRE D-3 lease-liveness system (renewal + self-fence + reaper) is GATED on
  \`lease_renew_interval_ticks != 0\` (crates/sim/src/directory.rs DirectoryTuning::validate: \`let active =
  lease_renew_interval_ticks != 0\` — ALL ordering checks are \`active & ...\`). DEV leaves VD_LEASE_RENEW_INTERVAL
  unset → parse_or 0 → the whole D-3 system is INERT: no lease renewed, no reaper, no self-fence. So a dev/prod
  cluster boots green with ZERO split-brain protection.
- The coherence (when active): \`self_fence_grace_ticks > lease_ttl_ticks\` (SelfFenceWithinTtl) AND
  \`self_fence_grace_ticks <= lease_ttl_ticks + max_self_fence_grace_ticks\` (SelfFenceOutlivesReassign) AND the
  renew-cadence \`renew_interval * min_renews <= lease_ttl\` (RenewTooSparse). DirectoryTuning::default:
  lease_ttl=100, renew=0(!), min_renews=4, grace=150, max=50, reaper=0(!), recovery=200.
- NODE-SIDE self-fence: shard.rs/gateway.rs read VD_SELF_FENCE_GRACE via \`parse_or(..., 0)\` → INERT by
  default; validate_self_fence_cadence(grace, recheck) returns Ok on grace==0 (NO enforcement), else requires
  recheck>0 AND grace >= 2*recheck. The ORCHESTRATOR's DirectoryTuning.self_fence_grace defaults to d3=150.
- FENCE RULE 4 (the split-brain safety, PLAN §Fence): a node failing lease renewal HARD-STOPS authority within
  self_fence_grace; the orchestrator reassigns ONLY after lease_ttl + max_self_fence_grace. Node-self-fence
  MUST precede orch-reassign or two owners co-exist (split-brain). Lapsed lease ⇒ ownership loss ONLY when the
  orchestrator CONFIRMS the owner unreachable (an orchestrator outage FREEZES recovery, never mass-orphans).
- R-4c cross-check: orchestrator.rs calls liveness.validate_against(confirm_unreachable_after_retries,
  redial_backoff_min/max, tick_hz) — the saga LIVENESS window must dominate the transport redial backoff
  spread or a dead peer is never confirmed dead.
- FOOTGUNS (a copy-paste of the dev env into a cloud manifest ships these): VD_STORE_EPHEMERAL_OK,
  VD_BOOT_STATE_EPHEMERAL_OK, VD_OUTBOX_EPHEMERAL_OK (the 3 durable-path escapes), and DEV_AUTH_SEED=[0x42;32]
  (the constant dev auth key the gateway boots with via VD_AUTH_PUBKEY). Also VD_STORE_DURABLE_ROOT must be a
  REAL mounted volume in cloud.
- The DEV config lives in vd_bins::DEV (DevClusterParams) + orchestrator_env/gateway_env/shard_env (crates/
  bins/src/lib.rs). tick_hz=50. NO VD_PROFILE / cloud-mode exists today.

6 HARD RULES: HR3 one tooling (the profile is a config mode, NEVER a per-shard-kind fork); HR5 100% Tier-A
coverage (any new sim/core/wire code); NO MAGIC NUMBERS (operational params in ONE reviewed config struct,
seed/derived where world-facing — the cloud D-3 values must be a DERIVED coherent set, justified, not
scattered literals). End-goal: signal-heavy multi-shard MMO, HUNDREDS of realm-owning shards (planets/stations/
ships), each must self-fence coherently on partition; multisharding meshes (one location spanning shards).
Never unilaterally adopt a new lib.`

phase('Design')
const spec = await agent(
  `${CTX}

You are the DESIGNER. Produce a concrete, implementable spec for Slice 2 — the cloud config profile. Read the
real code (directory.rs DirectoryTuning + validate + validate_self_fence_cadence; orchestrator.rs/gateway.rs/
shard.rs boot + env parsing; crates/bins/src/lib.rs DEV + *_env + the footgun sites; io-prod boot.rs
check_durable_path). Design:

1. THE PROFILE MECHANISM: how a node knows it is in "cloud" mode (e.g. VD_PROFILE=cloud vs dev/test default)
   and what it ENFORCES loud-at-boot in that mode. Must be DRY across the 3 bins (a shared preflight in
   vd-io-prod or vd-bins, ONE place — HR3), and must NOT break the dev/test/in-process path (D-3 inert stays
   valid there). Decide: does cloud mode REQUIRE the operator to supply coherent D-3 values (and just VALIDATE
   them loud), or does it SUPPLY a derived coherent default set (and reject overrides that break coherence)?
   Recommend, with the trade-off.
2. THE COHERENT CLOUD D-3 VALUE SET (no magic numbers): derive lease_ttl, renew_interval, min_renews,
   self_fence_grace, max_self_fence_grace, reaper_interval, recovery_grace (+ the NODE-side self_fence_grace +
   recheck) as ONE justified set from tick_hz + a stated target (e.g. "detect a partitioned owner within X
   seconds; reassign within Y"), satisfying DirectoryTuning::validate AND validate_self_fence_cadence AND the
   R-4c validate_against, AND fence-rule-4 (node self-fence strictly before orch reassign). Show the ordering
   holds. Where does this set LIVE (ONE config struct)?
3. FOOTGUN ENFORCEMENT: in cloud mode, reject VD_STORE_EPHEMERAL_OK / VD_BOOT_STATE_EPHEMERAL_OK /
   VD_OUTBOX_EPHEMERAL_OK, REQUIRE a real VD_STORE_DURABLE_ROOT, and reject the dev auth key (how to detect
   "the dev key" — VD_AUTH_PUBKEY == the DEV_AUTH_SEED pubkey → refuse). All fail LOUD.
4. NODE-vs-ORCHESTRATOR coherence: the shard/gateway node-side self_fence_grace must be < the orchestrator's
   reassign window, and consistent with the orchestrator's DirectoryTuning. How do the two agree in cloud mode
   (both read the same derived set? a cross-validation?).
5. THE TEST PLAN (HR5 + the DoD): cloud-profile boot FAILS LOUD on grace=0 / renew=0 / incoherent ordering /
   any ephemeral footgun / the dev auth key; a coherent cloud profile boots. Where do the unit tests live
   (Tier-A directory.rs validation) + the process/boot test.
6. SCALE/end-goal: hundreds of realm-owning shards each self-fencing on partition — does the design compose
   (per-shard lease, independent self-fence); any O(shards) concern in the reaper.

Output a full spec: the mechanism, the derived value set (with the arithmetic), the exact enforcement points
(file:function), the test plan, and any NEW config surface. Cite file:line. Do NOT edit files.`,
  { label: 'design', phase: 'Design', agentType: 'Explore', model: 'opus' }
)

phase('Attack')
const ATTACK_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['lens', 'verdict', 'findings'],
  properties: {
    lens: { type: 'string' },
    verdict: { type: 'string', enum: ['sound', 'concerns', 'broken'] },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'title', 'detail', 'fix'],
        properties: {
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          title: { type: 'string' },
          detail: { type: 'string' },
          fix: { type: 'string' },
        },
      },
    },
  },
}
const LENSES = [
  {
    key: 'split-brain-safety',
    prompt: `Attack the SPLIT-BRAIN SAFETY of this Slice-2 cloud-profile design. THE SPEC:\n\n${spec}\n\n${CTX}\n\nTry to BREAK it: find a partition/timing scenario where two owners co-exist, or a healthy owner is spuriously mass-self-fenced, under the proposed D-3 value set. Check fence-rule-4 (node self-fence strictly before orch reassign) holds under clock skew + the R-4c backoff spread. Check the orchestrator-outage-freezes-recovery property survives. Check the derived ordering actually satisfies validate + validate_self_fence_cadence for EVERY node role. Return the verdict + findings.`,
  },
  {
    key: 'mechanism-dry-footguns',
    prompt: `Attack the PROFILE MECHANISM + FOOTGUN enforcement + HR3/DRY of this Slice-2 design. THE SPEC:\n\n${spec}\n\n${CTX}\n\nProbe: is the cloud-vs-dev switch DRY (one shared preflight, not duplicated per bin)? Does it break the in-process/dev/test path (which relies on D-3 inert)? Can a footgun slip THROUGH (e.g. an ephemeral escape not covered, the dev auth key detection bypassed, a durable-root that canonicalizes to /tmp)? Is the enforcement HR3 (a config mode, never a shard-kind fork)? Does the fail-loud actually fire BEFORE any durable/authoritative action at boot? Return the verdict + findings.`,
  },
  {
    key: 'no-magic-scale',
    prompt: `Attack the NO-MAGIC-NUMBERS derivation + SCALE/end-goal fit of this Slice-2 design. THE SPEC:\n\n${spec}\n\n${CTX}\n\nProbe: are the cloud D-3 values DERIVED + justified from tick_hz + a stated target, in ONE config struct — or scattered literals (a magic-number violation)? Is the derivation robust across a retuned tick_hz? Does it compose to HUNDREDS of realm-owning shards (each self-fencing independently on partition; any O(shards) reaper cost; per-shard lease independence)? Does it fit the multi-shard-mesh + signal-heavy end-goal (a location spanning shards — do the co-located shards' self-fences interact)? Return the verdict + findings.`,
  },
]
const attacks = await parallel(
  LENSES.map((l) => () =>
    agent(l.prompt, { label: `attack:${l.key}`, phase: 'Attack', schema: ATTACK_SCHEMA, agentType: 'Explore', model: 'opus' })
  )
)

phase('Adjudicate')
const packed = attacks.filter(Boolean).map((a) => JSON.stringify(a)).join('\n\n')
const verdict = await agent(
  `${CTX}\n\nYou are the adjudicator for the Slice-2 cloud-profile design.\n\nTHE SPEC:\n${spec}\n\nTHE 3 ATTACKS (JSON):\n${packed}\n\nAdjudicate: for each finding decide real-blocker / fold-into-spec / false-alarm, with why. Then emit the FINAL SOUND_TO_IMPLEMENT SPEC (fold the must-fixes) — concrete enough to implement directly: the profile mechanism, the ONE config struct + the derived coherent value set (with arithmetic), the exact enforcement points (file:function), the fail-loud errors, the DRY shared preflight location, and the HR5 test plan. If it still has an unresolved structural flaw, say NEEDS_ANOTHER_ROUND with the specific gap. Do NOT edit files.`,
  { label: 'adjudicate', phase: 'Adjudicate', agentType: 'Explore', model: 'opus' }
)

return { spec, attacks: attacks.filter(Boolean), verdict }
