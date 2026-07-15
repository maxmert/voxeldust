export const meta = {
  name: 'r4a-postimpl-review',
  description: 'Adversarial review of the REAL landed R-4a retransmit-timer + confirm-dead code before commit',
  phases: [
    { title: 'Review', detail: '3 adversarial lenses on the real code' },
    { title: 'Synthesize', detail: 'adjudicate + verdict' },
  ],
}

const FILES = 'crates/io-prod/src/mesh.rs + crates/io-prod/tests/mesh_redelivery.rs'

const CONTEXT = [
  'This reviews the LANDED R-4a slice of the redelivering mesh transport (at-least-once over quinn). Read the REAL code',
  'in ' + FILES + '. Do NOT trust this description over the code -- verify every claim against it. R-4a pre-implementation',
  'design review (wf_40f7b3eb) caught 3 CRITICALs in the NAIVE design; this checks the ACTUAL implementation folded them in.',
  '',
  'R-4a ADDED (the sender-side retransmit timer that closes the R-3 idle-after-blip gap + per-lane confirm-dead):',
  '- ReliableLaneSender gained consecutive_failures:u32 + last_msg_id:Option<MsgId>; methods on_replay_failed (bump,',
  '  saturating), on_replay_ok (reset to 0), owes_redelivery (stream is None AND retry non-empty).',
  '- Free fns: any_lane_owes (the PER-LANE timer guard -- the C1 cure, NOT connection.is_none()); rearm_retransmit (ONE',
  '  edge-triggered re-eval at the loop TAIL with a counting bool -- the H1 cure); handle_connection_drop (connection=None +',
  '  abort ack-reader + on_write_error each lane, NO bounce/counter -- the bounce is threshold-gated); confirm_and_maybe_bounce',
  '  (bounce iff consecutive_failures >= confirm_unreachable_after_retries AND last_msg_id set); ensure_connection (the ONE',
  '  dial home, shared by write_frame + replay_lanes); replay_one_lane; replay_lanes (re-drive EACH owing lane).',
  '- peer_writer: a pinned tokio time sleep named retransmit + a counting bool + a 3rd biased select arm guarded by',
  '  any_lane_owes(&lanes); the inline sleep(backoff).await that R-3 had is REMOVED (backoff is now the timer deadline);',
  '  the Ok path calls on_replay_ok on THIS lane only; Err(WriteFail::Down) = handle_connection_drop + arm timer;',
  '  Err(WriteFail::Shed) = bounce once (oversize, no timer/no drop); rearm_retransmit runs at the loop tail.',
  '- write_frame: returns Result<(), WriteFail{Down,Shed}>; the RELIABLE arm assigns+retains + captures last_msg_id BEFORE',
  '  ensure_connection (buffer-first-before-dial, so a first-dial failure to a dead peer leaves the frame RETAINED + the timer',
  '  re-drives it -- NOT a silent drop); it re-fetches the lane after the dial (a fresh dial reset lane.stream). The datagram',
  '  (unreliable) hot path is byte-for-byte the R-1 bare DatagramFrame.',
  '',
  'The 3 pre-impl CRITICALs to confirm are ACTUALLY closed in the real code: (C1) the timer guard is per-lane not',
  'per-connection, so a sibling lane re-opening on a fresh send does NOT strand a still-owing lane; (C2) the failure counter',
  'is bumped per-lane on a lane OWN failed replay, NOT in the connection-drop fan-out, and DOES advance on the idle-timer',
  'path; (H2) the reset is per-lane (a lane own Ok), never blanket-all. Also: a blip that recovers = ZERO bounce; a',
  'genuinely-dead peer bounces after N; buffer-first no-loss holds (a bounced-then-recovered frame delivers exactly once);',
  'no connect-storm, no double-dial/double-write, no lost timer wakeup, no spin when drained.',
].join('\n')

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['lens', 'verdict', 'findings'],
  properties: {
    lens: { type: 'string' },
    verdict: { type: 'string', enum: ['SHIP', 'FIX_BEFORE_COMMIT'] },
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

const LENSES = [
  {
    key: 'timer-concurrency',
    prompt:
      CONTEXT +
      '\n\nLENS: RETRANSMIT-TIMER CONCURRENCY (real code). Trace peer_writer select loop + rearm_retransmit + replay_lanes + ensure_connection. Verify the idle-after-blip gap is ACTUALLY closed: a lone frame that blips then idles IS re-driven (the C1 per-lane guard holds against a sibling-lane re-open). Hunt: a connect-storm (timer + backoff hammering a dead host -- is the deadline growth correct now the inline sleep is gone?); a lost timer wakeup (the guarded-off select branch never re-enabling); a spin (timer firing repeatedly when nothing owes, or a zero deadline); a double-dial/double-write (timer re-drives while a send also dials, or replay_one_lane writes a frame twice); the counting bool getting stuck (owed-but-never-armed, or a live deadline pushed back and stalling retransmit); replay_lanes conn_died mid-pass handling (does it correctly leave remaining lanes owed + drop the connection so the next fire re-dials?); the biased-select ordering. VERDICT + findings.',
  },
  {
    key: 'confirm-noloss',
    prompt:
      CONTEXT +
      '\n\nLENS: CONFIRM-DEAD + NO-LOSS (real code). Trace on_replay_failed/on_replay_ok/confirm_and_maybe_bounce + the write_frame buffer-first-before-dial reorder + handle_connection_drop. Verify: (C2) the counter advances on the IDLE-timer path (not only on an in-flight send) and is NOT bumped by the connection-drop fan-out; (H2) reset is per-lane; a blip that recovers before N gives ZERO bounce; a dead peer bounces after N (and the re-bounce cadence is backoff-limited, not an inbox flood); buffer-first no-loss: a frame whose FIRST dial fails is RETAINED + re-driven (the reorder: assign before ensure_connection) and a bounced-then-recovered frame still delivers EXACTLY ONCE (receiver dedup). Hunt any path where a frame is silently dropped (a first-dial failure with no lane, a shed that should retain, a retry cleared on a bounce), or double-delivered, or where last_msg_id is stale/missing so a bounce is lost or carries the wrong id. VERDICT + findings.',
  },
  {
    key: 'integration',
    prompt:
      CONTEXT +
      '\n\nLENS: INTEGRATION & REGRESSION (real code). Verify the WriteFail Down/Shed split routes correctly (oversize gives Shed gives bounce-once-no-timer; connection/write error gives Down gives drop+timer; datagram ConnectionLost gives Down). Verify ensure_connection is the ONE dial home with NO ack-reader leak/duplication (abort-then-respawn) shared by write_frame + replay_lanes. Verify the 20Hz UNRELIABLE datagram hot path is byte-for-byte untouched (still the bare DatagramFrame, no timer/counter/ledger touch). Verify the existing tests semantics still hold (a_dead_peer bounces -- now after N; the blip test zero-bounce; two_reliable_classes; oversize-rejected-lane-survives; mesh_volume no-loss/no-dup). Check the new tests actually DISCRIMINATE (the idle-after-blip test would fail under R-3). Check for any Tier-B coverage gap (a new async arm no test reaches). VERDICT + findings.',
  },
]

phase('Review')
const reviews = (await parallel(
  LENSES.map((l) => () =>
    agent(l.prompt, { label: 'review:' + l.key, phase: 'Review', schema: SCHEMA, model: 'opus' })
  )
)).filter(Boolean)

const blocking = reviews.flatMap((r) =>
  (r.findings || [])
    .filter((f) => f.severity === 'CRITICAL' || f.severity === 'HIGH')
    .map((f) => ({ ...f, lens: r.lens }))
)
log('reviews: ' + reviews.map((r) => r.lens + '=' + r.verdict).join(', ') + '; ' + blocking.length + ' CRITICAL/HIGH')

phase('Synthesize')
const synthesis = await agent(
  'You are the synthesizer for the R-4a post-implementation review. Read the REAL code in ' +
    FILES +
    ' yourself to ADJUDICATE each finding (drop false alarms with reasons; keep real ones). Reviewer findings (JSON):\n' +
    JSON.stringify(reviews, null, 2) +
    '\n\nProduce: (1) an adjudicated de-duplicated list of REAL CRITICAL/HIGH findings, each with the exact code fix; (2) an overall verdict SHIP or FIX_BEFORE_COMMIT; (3) a one-paragraph bottom line on whether R-4a soundly closes the idle-after-blip gap and preserves no-loss. Be adversarial but precise -- cite file:line. Adjudicate the 3 pre-impl CRITICALs (per-lane guard, counter placement, reset scope) as actually-closed-or-not against the real code.',
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return {
  verdicts: reviews.map((r) => ({ lens: r.lens, verdict: r.verdict })),
  blocking_count: blocking.length,
  blocking,
  synthesis,
}
