export const meta = {
  name: 'r4b-holistic-audit',
  description: 'Holistic /goal audit after R-3/R-4a/R-4b land: how all components compose, scale, and fit the end-goal',
  phases: [
    { title: 'Audit', detail: '3 holistic lenses' },
    { title: 'Synthesize', detail: 'adjudicate + DONE_NO_CRITICAL verdict' },
  ],
}

const CONTEXT = [
  'Voxeldust is a transfer-first "Star Citizen x Minecraft" voxel MMO (greenfield rebuild). Current committed HEAD is',
  '49e0395. The redelivering (at-least-once) MeshTransport is now largely built: R-1..R-3 (receiver contiguity verdict +',
  'reverse cumulative-ack) + R-4a (sender-side retransmit TIMER that closes the idle-after-blip gap + per-lane',
  'confirm-dead-after-N) + R-4b (retry_bytes byte-cap + shed-loud producer backpressure) are LANDED. Still OWED: R-4c',
  '(LivenessTuning::validate_against(&SagaTuning) — the boot invariant that the transport never confirms a peer DEAD',
  'AFTER the saga abort_deadline fires, i.e. confirm_ticks <= abort_deadline_ticks accounting for the GEOMETRIC backoff',
  '50ms->5s), R-4d (MeshStats -> /metrics ops surface), R-4e (N-peer load test), R-5 (mesh_under_loss.rs capstone that',
  'flips D-6 #1 green), R-6 (durable outbox + boot-counter, P6/P7).',
  '',
  'This is a HOLISTIC /goal audit of the CURRENT committed state. Not a re-review of R-4a/R-4b transport correctness',
  '(focused reviews wf_b1d0610c + agent aa95e10c already SHIPped those). The BROADER lens: how do all components COMPOSE',
  'now that confirm-dead + shed are LIVE, does it SCALE to the end-goal, does it advance the planned game. READ the real',
  'code + docs to ground every claim: docs/design/PLAN.md (6 hard rules HR1-HR6, roadmap), docs/design/DEFERRED.md (D-6,',
  'the R-4c/d/e/R-5/R-6 owed items + the ledgered H2 RecvLedger-lock / M4 metrics), crates/io-prod/src/mesh.rs (peer_writer',
  'timer + confirm_and_maybe_bounce + assign_and_retain shed), crates/sim/src/saga.rs (LivenessTuning + n_consecutive_unreachable',
  '+ abort_deadline_ticks + retry_delay_ticks_hint), crates/sim/src/io/mod.rs (the frozen Transport/Inbound/MsgClass seam),',
  'crates/connection-plane (gateway dispatch), crates/wire (InterShardFlow).',
  '',
  'THE SHARPEST NEW QUESTION: R-4a made confirm-dead LIVE (the transport now bounces NodeUnreachable after N failed',
  'redials, off the geometric backoff), but R-4c (the validate_against boot invariant) is NOT landed yet. Is there a LIVE',
  'risk in this window that the transport confirms a peer dead AFTER the saga already force-aborted (or vice versa),',
  'aborting a live-but-slow player? Check the DEFAULT tuning: N=confirm_unreachable_after_retries (default 3) * the',
  'geometric backoff (redial_backoff_min 50ms doubling to redial_backoff_max 5s) vs the orchestrator saga',
  'abort_deadline_ticks + n_consecutive_unreachable * retry_delay_ticks_hint at the default tick rate. Is the default safe',
  '(so R-4c is belt-and-suspenders), or is there a real ordering hazard NOW that must be fixed before R-4c lands?',
  '',
  'END-GOAL to keep in view: spherical voxel planets; player-built ships/stations/cities walked-inside while flying;',
  'SIGNAL-HEAVY cross-shard functional-block gameplay (possibly cross-shard); multi-sharded meshes (ONE location spanning',
  'shards); client simultaneous multi-mesh rendering; Newtonian physics; players physically COLLIDE cross-boundary (NO',
  'client prediction); PvP; HUNDREDS of users in one location. Robust, scalable, DRY, elegant, cloud/k8s-ready, AAA MMO.',
].join('\n')

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['lens', 'verdict', 'findings'],
  properties: {
    lens: { type: 'string' },
    verdict: { type: 'string', enum: ['CLEAN', 'FINDINGS'] },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'area', 'problem', 'recommendation'],
        properties: {
          severity: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'] },
          area: { type: 'string' },
          problem: { type: 'string' },
          recommendation: { type: 'string' },
        },
      },
    },
  },
}

const LENSES = [
  {
    key: 'composition',
    prompt:
      CONTEXT +
      '\n\nLENS: COMPOSITION & INTEGRATION. Now that confirm-dead + shed are LIVE, do the layers still compose cleanly? Focus HARD on the sharpest new question above (the confirm-dead vs saga-abort ordering hazard in the pre-R-4c window — is the DEFAULT tuning actually safe, computed against the real saga.rs LivenessTracker + abort_deadline_ticks + the geometric backoff?). Also: does the R-4a threshold-gated NodeUnreachable still feed the saga LivenessTracker correctly (it counts consecutive NodeUnreachable toward a peer — does the coalesced/threshold-gated bounce give it what it needs, or too few/too many)? Does the R-4b shed (WriteFail::Shed bounce) interact correctly with the LivenessTracker (a shed bounce is NOT a dead peer — could it wrongly trip liveness)? Does the transport-redelivery vs saga-re-drive layering still hold? Any double-cure or gap the new slices introduced? Verify sim/node still never see acks/timers/shed (the frozen seam). VERDICT + findings.',
  },
  {
    key: 'scale-load',
    prompt:
      CONTEXT +
      '\n\nLENS: SCALABILITY, HIGH-LOAD & CLOUD/AAA. Will the CURRENT transport hold at hundreds-in-one-location across shards? Probe: (1) the retransmit timer + confirm-dead re-bounce cadence at scale — during a multi-peer outage, N_peers x N_lanes NodeUnreachable bounces flow into the shared BoundedInbox; is the backoff-limited cadence actually bounded enough to not evict genuine reliable inbound (inbound_dropped_reliable)? (2) the RecvLedger node-wide Mutex (ledgered H2) — still the right call to defer, or does confirm-dead/shed add contention? (3) the retry_bytes cap default (4MiB) vs the mpsc bound — does the mpsc fill first for typical small reliable frames (so the shed is truly a dead-ack-path safety net, not normal-operation backpressure)? (4) MeshStats -> /metrics (M4) still owed — is the dead-ack-path/confirm-dead observability reachable by ops before a real soak? (5) the 20Hz UNRELIABLE hot path stays ZERO-cost (confirm R-4a/R-4b added nothing to it). (6) cloud/k8s: the timer + confirm-dead under pod restart/reschedule; is a load test (R-4e) still a gap? VERDICT + findings.',
  },
  {
    key: 'trajectory-dry',
    prompt:
      CONTEXT +
      '\n\nLENS: END-GOAL TRAJECTORY, DRY & ELEGANCE. Does the completed R-4a/R-4b advance the planned game without painting into a corner? (1) The SIGNAL system (cross-shard functional blocks, HMAC-granted, fence-stamped, idempotent-where-durable, Galaxy-Relay radio) rides reliable cross-shard arms — does the now-complete at-least-once + shed + confirm-dead give Signals what they need, or is there a mismatch? (2) Multi-sharded meshes + client multi-mesh rendering: does the transport generalize, or assume 1-location-1-owner? (3) DRY: ONE codec/framing home, ONE lane FSM, ONE dial home (ensure_connection), ONE config struct (MeshReliabilityTuning), no per-kind/per-shard fork (HR2/HR3)? Any duplication R-4a/R-4b introduced? (4) Is the R-4c/d/e/R-5/R-6 owed-work correctly sequenced + honestly ledgered, or is something load-bearing hidden (e.g. is R-4c actually urgent given confirm-dead is live)? (5) Any DRIFT from the 6 hard rules, or any place the new slices make a FUTURE feature (ship compound transfer, warp provisioning, damage-across-transfer-barrier, PvP damage/death reliability) harder? VERDICT + findings.',
  },
]

phase('Audit')
const audits = (await parallel(
  LENSES.map((l) => () =>
    agent(l.prompt, { label: 'audit:' + l.key, phase: 'Audit', schema: SCHEMA, model: 'opus' })
  )
)).filter(Boolean)

const all = audits.flatMap((a) => (a.findings || []).map((f) => ({ ...f, lens: a.lens })))
const blocking = all.filter((f) => f.severity === 'CRITICAL' || f.severity === 'HIGH')
log('audits: ' + audits.map((a) => a.lens + '=' + a.verdict).join(', ') + '; ' + blocking.length + ' CRITICAL/HIGH')

phase('Synthesize')
const synthesis = await agent(
  'You are the synthesizer for the post-R-4b HOLISTIC /goal audit. Read the real code/docs yourself to ADJUDICATE each finding (drop false alarms with reasons; keep real ones). Reviewer findings (JSON):\n' +
    JSON.stringify(audits, null, 2) +
    '\n\nThe /goal bar: how does it all work together (complementary or will it break?), DRY, robust, scalable, cloud-ready, elegant, AAA MMO quality, prepared for the planned features (signal-heavy cross-shard, multi-mesh, PvP, hundreds-in-one-location). Produce: (1) an adjudicated de-duplicated list of REAL findings by severity, each with a concrete recommendation and whether it is IN-SCOPE-NOW vs correctly-owed-to-a-later-slice (R-4c/d/e/R-5/R-6/Px). DECISIVELY adjudicate the confirm-dead-vs-saga-abort ordering question: is there a LIVE hazard in the pre-R-4c window with the DEFAULT tuning (compute it against the real numbers), or is R-4c belt-and-suspenders? (2) a verdict DONE_NO_CRITICAL (no CRITICAL and no in-scope-unaddressed HIGH) or HAS_CRITICAL. (3) a one-paragraph bottom line. Cite file:line.',
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return {
  verdicts: audits.map((a) => ({ lens: a.lens, verdict: a.verdict })),
  blocking_count: blocking.length,
  blocking,
  synthesis,
}
