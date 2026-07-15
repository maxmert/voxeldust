export const meta = {
  name: 'r4d2-postimpl-review',
  description: 'Adversarial post-impl review of R-4d2 (MeshStats -> /metrics, orchestrator-only)',
  phases: [{ title: 'Review' }, { title: 'Synthesize' }],
}

const ROOT = '/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system'

const CONTEXT = [
  'R-4d2 (M4) just landed in the voxeldust greenfield rebuild: surface the io-prod mesh reliability',
  'counters (MeshStatsSnapshot: reliable_shed, reliable_acked, gap_drop, dedup_drop, stale_incarnation/',
  'epoch_drop, inbound_dropped_reliable/unreliable, datagrams_dropped_too_large/send) on the orchestrator',
  'admin /metrics endpoint. Previously every bin discarded the MeshControl and /metrics served hardcoded 0,',
  'so a soak/deploy would have INVISIBLE alarms (a dead ack path = reliable_acked stuck at 0; a contiguity',
  'gap = gap_drop MUST be 0; a saturated retry buffer = reliable_shed). This is the M4 owed item from the',
  'D-6 #1 redelivering-transport hardening.',
  '',
  'SCOPE (vetted design wbt45dxj6, resolved by a HIGH finding): ORCHESTRATOR-ONLY this slice. The draft',
  'wanted gateway+shard /metrics listeners too, but that is UNDELIVERABLE in the dev cluster (ClusterAddrs',
  'has ONE admin field, VD_ADMIN_ADDR set only in orchestrator_env, SlotPorts allocates ONE admin port —',
  'gateway/shard would read VD_ADMIN_ADDR unset = no listener, or via common_env collide three binds on one',
  'port). So gateway/shard mesh counters stay UNSCRAPED (ledgered in DEFERRED.md as owed with per-node admin',
  'ports). gateway.rs/shard.rs are UNCHANGED this slice.',
  '',
  'What landed:',
  '  crates/wire/src/admin.rs: +9 metric_names constants (the mesh counters, all _total except the legacy',
  '    un-suffixed vd_datagrams_dropped_too_large which is a FROZEN scraped name kept as-is) appended to',
  '    metric_names::ALL. The existing metric_names_are_unique_and_prometheus_valid test auto-covers them.',
  '  crates/io-prod/src/admin.rs: a MetricsSource trait (mirrors SnapshotSource) + a flat MetricValues',
  '    struct (10 fields) + MeshMetrics(Arc<MeshControl>) (live: reads MeshControl::stats() per scrape, a',
  '    pure atomic load) + FixedMetrics (const, for tests/shell). admin_router is now 2-arg',
  '    (snapshot + metrics) with a combined AdminState. render_metrics(&MetricValues) EXHAUSTIVELY',
  '    DESTRUCTURES MetricValues (no ..) so a new field is a COMPILE ERROR until paired to a metric_names',
  '    entry (the registry-vs-struct completeness guard), pairs each to its name, and iterates ALL emitting',
  '    the live value if wired else 0 (shaped, never a 404). render_metrics_shell was REMOVED.',
  '  crates/bins/src/bin/orchestrator.rs: RETAINS the MeshControl (was _control discarded), wraps it Arc,',
  '    and passes MeshMetrics(control) to admin_router at the existing VD_ADMIN_ADDR axum serve.',
  '  Tests: wire uniqueness/prometheus (auto); io-prod render_metrics pairing + both find branches (hit/miss)',
  '    + a struct-vs-registry completeness test + the /metrics HTTP scrape reflecting seeded FixedMetrics',
  '    values; a NEW mesh_redelivery.rs integration test proving the LIVE MeshMetrics->MeshControl bridge',
  '    (after a real burst retires, MetricsSource::values() mirrors MeshControl::stats() field-for-field).',
  '',
  'Gate GREEN: Tier-A 100% region+branch (wire/admin.rs 100%); Tier-B io-prod >= 90 floor (~93%); clippy',
  '-D warnings clean; fmt clean. Files under ' + ROOT + '. Use git -C ' + ROOT + ' diff HEAD to see the',
  'changeset (R-4d1 is already committed at fd8cdd9; this diff is R-4d2 only).',
].join('\n')

const LENSES = [
  {
    key: 'registry-seam-callsites',
    prompt: [
      'LENS: the frozen metric_names registry + the admin_router signature change + DRY/no-magic-strings.',
      'Confirm all NINE new metric names are in metric_names::ALL, are prometheus-valid (vd_ namespace,',
      'ascii-lowercase, _total counter convention), and are UNIQUE (no collision with the 8 pre-existing).',
      'Is the legacy vd_datagrams_dropped_too_large exception (kept un-suffixed, NOT renamed) correct + is a',
      'frozen-scraped-name rename hazard actually avoided? The admin_router signature went from 1-arg to',
      '2-arg: find EVERY call site across the workspace (grep) and confirm each was updated (orchestrator.rs',
      'AND the in-crate admin.rs test helper AND any other) — a missed one is a compile error, but confirm',
      'none was left on a stale shim or given a misleading empty source. Any magic string literals at a',
      'render/instrumentation site (there must be none — names come only from metric_names)?',
    ].join('\n'),
  },
  {
    key: 'completeness-guard-render',
    prompt: [
      'LENS: the registry-vs-struct completeness guard + render_metrics correctness + HR5.',
      'The claim: render_metrics exhaustively destructures MetricValues (no ..) so a NEW field is a COMPILE',
      'ERROR until paired. VERIFY this is actually true (a hypothetical 11th MetricValues field would fail to',
      'compile until added to the destructure + the live array). Is the `live` array <-> destructured-binding',
      '<-> metric_names pairing correct with NO transposition (each field maps to its RIGHT name)? Is the',
      'find()-based lookup correct (every ALL name emitted once; a wired name shows its value; an unwired name',
      'shows 0; no duplicate/missing lines)? Are BOTH find branches (hit + miss) actually covered by a test?',
      'Does the completeness test genuinely guard the forward direction (every live name in ALL) as well as',
      'the compile-time reverse? Is there a false-pass where render could emit an unregistered or wrong-valued',
      'metric and a test still go green? Assess whether the linear find() is a real concern at ~17 names (it',
      'is not, but confirm no O(n^2) blowup risk as the registry grows).',
    ].join('\n'),
  },
  {
    key: 'orchestrator-wiring-hr1-descope',
    prompt: [
      'LENS: the orchestrator MeshControl retention + HR1 no-leak + the descope soundness + cloud-scale.',
      'The orchestrator now RETAINS MeshControl (was discarded) and shares it via Arc into MeshMetrics.',
      'Verify: MeshControl is Send+Sync+static (the trait bound holds); MeshControl::stats() is genuinely a',
      'cheap non-blocking atomic load (NOT a lock held across the scrape, NOT touching the sim/tick thread) so',
      'a /metrics scrape can never stall the hot path; retaining control does not change shutdown/lifetime',
      'semantics or break the SIGKILL/crash process tests. HR1: do the exposed counters leak anything SEALED',
      '(a shard World/rapier/redb detail, a session id, an entity blob)? They are aggregate transport counters',
      '— confirm they are safe to expose. Is the orchestrator-only DESCOPE sound and honestly ledgered — is',
      'there any DEAD capability shipped (a listener that no-ops, a metrics source wired to nothing), and is',
      'the gateway/shard-unscraped gap a real limitation a reader is warned about (DEFERRED.md)? Does this',
      'compose toward the end goal (hundreds of players, multi-shard, cross-shard signals) — will',
      'orchestrator-only mesh metrics be enough to observe a real soak, or is the gateway/shard blindness a',
      'CLOUD-BLOCKING gap that must be called out now?',
    ].join('\n'),
  },
]

phase('Review')
const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['verdict', 'findings', 'summary'],
  properties: {
    verdict: { type: 'string', enum: ['SOUND_TO_COMMIT', 'FIX_BEFORE_COMMIT'] },
    summary: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'location', 'problem', 'fix'],
        properties: {
          severity: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'] },
          location: { type: 'string' },
          problem: { type: 'string' },
          fix: { type: 'string' },
        },
      },
    },
  },
}

const reviews = await parallel(
  LENSES.map((l) => () =>
    agent(
      [CONTEXT, '', 'YOUR REVIEW LENS:', l.prompt, '',
       'Read the actual code before judging (grep the workspace where the lens needs completeness). Be',
       'adversarial and concrete: cite file:line. A finding must name a REAL defect with a specific fix, not a',
       'style preference. If sound on your lens, say so plainly. Return the structured verdict.'].join('\n'),
      { label: 'review:' + l.key, phase: 'Review', schema: SCHEMA, model: 'opus' }
    )
  )
)

phase('Synthesize')
const synth = await agent(
  [CONTEXT, '',
   'You are the SYNTHESIZER. Three adversarial reviewers returned verdicts on R-4d2:',
   JSON.stringify(reviews, null, 2), '',
   'Adjudicate against the ACTUAL code (read the files yourself to confirm or refute each finding). Produce:',
   '(a) the deduplicated, severity-ranked list of REAL findings that must change before commit (CRITICAL/HIGH',
   '= blocking; MEDIUM = fix-now-if-cheap; LOW = ledger), each with a concrete fix; (b) a ruling on whether the',
   'orchestrator-only descope is the right call or a cloud-blocking gap that must be widened now; (c) a final',
   'verdict: SOUND_TO_COMMIT or FIX_BEFORE_COMMIT. Be decisive and specific.'].join('\n'),
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return { reviews, synthesis: synth }
