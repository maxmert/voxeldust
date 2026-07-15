export const meta = {
  name: 'r4d-design',
  description: 'Design + adversarially review R-4d (MeshStats to /metrics + Inbound::SendShed disambiguation) before impl',
  phases: [
    { title: 'Design', detail: 'concrete R-4d design from the real code' },
    { title: 'Review', detail: '3 adversarial lenses' },
    { title: 'Synthesize', detail: 'vetted implement-ready design' },
  ],
}

const CONTEXT = [
  'Voxeldust redelivering (at-least-once) MeshTransport. R-1..R-4c are LANDED (HEAD 0c98757). R-4d is the OPS SLICE the',
  'post-R-4b holistic audit (wf_9d72e38c) flagged as needed BEFORE any real soak/deploy. TWO deliverables:',
  '',
  'M4 (observability): the io-prod mesh exposes ~10 never-silent counters via MeshControl::stats() -> MeshStatsSnapshot',
  '(reliable_shed, reliable_acked, gap_drop [MUST-BE-0], dedup_drop, stale_epoch_drop, stale_incarnation_drop,',
  'inbound_dropped_reliable [genuine-overload alarm], inbound_dropped_unreliable, datagrams_dropped_too_large,',
  'datagrams_dropped_send). But: ALL server bins DISCARD the control handle (let (transport, _control) = spawn_mesh(...) in',
  'orchestrator.rs / gateway.rs / shard.rs), so stats() is NEVER read; the /metrics handler (io-prod/src/admin.rs',
  'render_metrics_shell) emits every wire/src/admin.rs metric_names::ALL name hardcoded to 0; that registry contains ~8',
  'names, NONE of them the mesh counters. So now that confirm-dead (R-4a) + shed (R-4b) are LIVE, the two alarms that tell',
  'ops WHY a player got bounced (reliable_acked stuck-at-0 = a dead ack path) or WHY a lane wedged (reliable_shed) are',
  'INVISIBLE. Wire them: add the counter names to the review-gated metric_names::ALL; replace render_metrics_shell hardcoded',
  '0 with a MetricsSource trait (mirror the existing SnapshotSource pattern the admin router already uses) reading',
  'MeshControl::stats() (+ optionally the BoundedInbox tallies); have the bins RETAIN their MeshControl and feed /metrics.',
  'CAUTION: the audit noted gateway/shard have NO admin listener / VD_ADMIN_ADDR today (only the orchestrator mounts the',
  'admin_router) — resolve WHERE /metrics lives per bin (orchestrator-only? add a loopback listener to gateway/shard per',
  'the CAF-2 loopback/auth admin contract? or defer the non-orchestrator bins?).',
  '',
  'M3 (a REAL composition gap R-4b introduced, correctness fix): a WriteFail::Shed bounce (io-prod mesh.rs peer_writer, for',
  'BOTH AssignReject::Unframable [oversize] and AssignReject::BufferFull [retry buffer full = a dead ack path, but the PEER',
  'MAY BE ALIVE]) currently pushes Inbound::NodeUnreachable{to,class,undelivered} — BYTE-IDENTICAL to a genuine dead-peer',
  'bounce from confirm_and_maybe_bounce. The orchestrator saga_runtime.rs inbound loop matches Inbound::NodeUnreachable and',
  'calls record_unreachable(to, now) UNCONDITIONALLY. So a BufferFull shed toward an inbound-QUIET peer (a frozen source',
  'shard that went silent) can accrue to n_consecutive_unreachable and FALSE-CONFIRM a live-but-ack-stalled peer, tripping a',
  'destructive re-home/abandon. Two mitigations keep it non-triggering at default tuning (clear-on-ack fires on ANY inbound',
  'Wire; a shed is one-shot per new send, no timer), but it is a real gap under HR5/no-shortcuts. The cure: a DISTINCT',
  'Inbound arm (Inbound::SendShed) the mesh emits for a Shed, that the saga_runtime routes to a METRIC (a shed counter),',
  'NEVER to record_unreachable. This is an ADDITIVE change to the FROZEN Transport/Inbound seam (Inbound is a CLOSED',
  'taxonomy per HR1 — the least-invasive kind, an additive arm) — so it must go through the frozen-wire discipline: EVERY',
  'consumer of Inbound must handle the new arm, and the mem FaultFabric parity + the seam conformance gate must stay green.',
  '',
  'READ the real code before designing: crates/io-prod/src/mesh.rs (peer_writer WriteFail::Down/Shed arms, confirm_and_maybe_bounce,',
  'MeshStats/MeshStatsSnapshot/MeshControl::stats), crates/io-prod/src/admin.rs (render_metrics_shell + the admin router +',
  'SnapshotSource), crates/wire/src/admin.rs (metric_names::ALL + the registry), crates/sim/src/io/mod.rs (the frozen',
  'Transport/Inbound/SendError seam + the Inbound enum), crates/node/src/saga_runtime.rs (the Inbound match loop +',
  'record_unreachable), crates/sim/src/io/mem.rs (the FaultFabric — does it ever shed? parity), the bins',
  '(orchestrator.rs/gateway.rs/shard.rs let (_,_control), VD_ADMIN_ADDR), and any Inbound-conformance/G-SEALED test.',
  '',
  'CONSTRAINTS: HR1 (Inbound is the closed client/shard taxonomy — an additive arm is OK but every match must stay',
  'exhaustive + the conformance gate green). HR5 (Tier-A 100% for sim/node — the new Inbound arm + its handling must be',
  'fully covered; io-prod Tier-B floor 90; the MetricsSource in io-prod is Tier-B). The 20Hz UNRELIABLE datagram hot path',
  'must stay byte-for-byte untouched. No magic numbers. DRY (mirror SnapshotSource; one metrics-name home). MeshControl::stats()',
  'is a pure atomic load (off the seam + off the hot path) — keep it that way. Propose sub-slicing (R-4d1 SendShed /',
  'R-4d2 metrics) if landing as one is unwise.',
].join('\n')

const REVIEW_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['lens', 'verdict', 'blocking'],
  properties: {
    lens: { type: 'string' },
    verdict: { type: 'string', enum: ['SOUND_TO_IMPLEMENT', 'FIX_BEFORE_IMPL'] },
    blocking: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'problem', 'fix'],
        properties: {
          severity: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM'] },
          problem: { type: 'string' },
          fix: { type: 'string' },
        },
      },
    },
  },
}

const DESIGN = await (async () => {
  phase('Design')
  return agent(
    CONTEXT +
      '\n\nProduce the CONCRETE, implement-ready R-4d design. For M3: the exact Inbound::SendShed variant shape (fields), ' +
      'the exact peer_writer change (Shed arm emits SendShed not NodeUnreachable), the exact saga_runtime routing (SendShed ' +
      '-> a shed metric/counter, never record_unreachable), EVERY other Inbound consumer that must add the arm (grep them), ' +
      'the mem FaultFabric parity (does mem ever emit SendShed?), and the seam-conformance/G-SEALED test impact. For M4: the ' +
      'exact MetricsSource trait (mirroring SnapshotSource), the metric_names::ALL additions, the render_metrics_shell ' +
      'replacement, WHERE /metrics lives per bin (resolve the gateway/shard-no-admin-listener question decisively), and the ' +
      'bin wiring to retain MeshControl. Give the Tier-A test plan (the new Inbound arm handling) + the Tier-B metrics test. ' +
      'Propose sub-slicing if warranted. Flag every open decision. Cite file:line.',
    { label: 'design', phase: 'Design', model: 'opus' }
  )
})()

const LENSES = [
  {
    key: 'seam-hr1',
    prompt:
      'Adversarially review this R-4d DESIGN for the FROZEN-SEAM / HR1 correctness of the Inbound::SendShed additive arm. ' +
      'Read crates/sim/src/io/mod.rs (Inbound enum + the seam contract), crates/node/src/saga_runtime.rs (the match), ' +
      'crates/sim/src/io/mem.rs (FaultFabric), and any Inbound-conformance/G-SEALED test. Hunt: a NON-exhaustive Inbound ' +
      'match left after adding the arm (a consumer that silently ignores SendShed, or a `_ =>` that swallows it where it ' +
      'must be handled); a mem-vs-mesh parity break (does the mem transport need to emit SendShed, or is it mesh-only, and ' +
      'is that divergence documented like the cross-class-ordering one?); whether SendShed leaking undelivered=MsgId is ' +
      'right (a shed frame IS un-delivered — should the caller learn its id, like NodeUnreachable?); whether adding the arm ' +
      'regresses the WireMonitor/conservation invariants or the seam conformance gate; whether the 20Hz path is untouched. ' +
      'VERDICT + blocking findings.\n\n---DESIGN---\n' +
      DESIGN,
  },
  {
    key: 'routing-liveness',
    prompt:
      'Adversarially review this R-4d DESIGN for the M3 ROUTING correctness (the false-confirm fix). Read saga_runtime.rs ' +
      '(record_unreachable, is_confirmed_dead, clear-on-ack, rehome_event_for) + mesh.rs (the Shed vs the confirm-dead ' +
      'bounce). Verify the fix ACTUALLY closes the false-confirm: a BufferFull/Unframable shed now routes to SendShed (a ' +
      'metric) and NEVER reaches record_unreachable, so a shed toward a quiet peer can no longer accrue to ' +
      'n_consecutive_unreachable. But verify NO REGRESSION: a genuine dead-peer bounce (confirm_and_maybe_bounce) STILL ' +
      'emits NodeUnreachable and STILL feeds record_unreachable; the oversize-Unframable case still surfaces to the caller ' +
      'that its frame did not deliver (a permanent reject the sim must know about — is losing it in a metric acceptable, or ' +
      'does the caller need a distinct signal?); the retry-buffer-full case does not silently drop a frame the sim believed ' +
      'sent. Does dropping the shed off the NodeUnreachable path lose any liveness signal a real dead peer needs? VERDICT + ' +
      'blocking findings.\n\n---DESIGN---\n' +
      DESIGN,
  },
  {
    key: 'metrics-ops-dry',
    prompt:
      'Adversarially review this R-4d DESIGN for the M4 METRICS wiring + ops + DRY. Read io-prod/src/admin.rs (render_metrics_shell ' +
      '+ the admin router + SnapshotSource), wire/src/admin.rs (metric_names::ALL), and the bins (VD_ADMIN_ADDR, who mounts ' +
      'the admin router). Verify: the MetricsSource trait genuinely mirrors SnapshotSource (no new pattern); metric_names::ALL ' +
      'gains ALL the mesh counters (not a subset — a missing name is silently never scraped, defeating GW-1) with a test that ' +
      'every MeshStatsSnapshot field has a registered name; the render reads MeshControl::stats() off the axum task (a pure ' +
      'atomic load, off the seam + off the 20Hz hot path — confirm no lock/await added); the /metrics-per-bin decision is ' +
      'sound (if gateway/shard get a new loopback admin listener, does that expand the CAF-2 loopback/auth contract safely, ' +
      'or is orchestrator-only correct given the orchestrator sees only ITS OWN mesh counters not the gateway/shard ones?); ' +
      'the bins retaining MeshControl does not change lifecycle/shutdown. Is the sub-slicing sensible? VERDICT + blocking ' +
      'findings.\n\n---DESIGN---\n' +
      DESIGN,
  },
]

phase('Review')
const reviews = (await parallel(
  LENSES.map((l) => () =>
    agent(l.prompt, { label: 'review:' + l.key, phase: 'Review', schema: REVIEW_SCHEMA, model: 'opus' })
  )
)).filter(Boolean)

const blocking = reviews.flatMap((r) => (r.blocking || []).map((b) => ({ ...b, lens: r.lens })))
log('reviews: ' + reviews.map((r) => r.lens + '=' + r.verdict).join(', ') + '; ' + blocking.length + ' blocking')

phase('Synthesize')
const vetted = await agent(
  'You are the synthesizer for the R-4d pre-implementation design review. Read the real code yourself to adjudicate. The DESIGN:\n\n' +
    DESIGN +
    '\n\nReviewer verdicts + blocking findings (JSON):\n' +
    JSON.stringify(reviews, null, 2) +
    '\n\nProduce the FINAL VETTED, IMPLEMENT-READY R-4d design with every real blocking finding FOLDED IN (drop false alarms ' +
    'with reasons). Give: the resolved Inbound::SendShed shape + the exact mesh emit + saga_runtime routing + EVERY Inbound ' +
    'consumer to update + the mem parity decision + the conformance-gate impact; the resolved MetricsSource + metric_names::ALL ' +
    'additions + the /metrics-per-bin decision + bin wiring; the sub-slicing; the Tier-A + Tier-B test plan. State clearly ' +
    'whether it is SOUND TO IMPLEMENT as specified. Cite file:line.',
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return {
  design_len: DESIGN.length,
  verdicts: reviews.map((r) => ({ lens: r.lens, verdict: r.verdict })),
  blocking_count: blocking.length,
  blocking,
  vetted,
}
