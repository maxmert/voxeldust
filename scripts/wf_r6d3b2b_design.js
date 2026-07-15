export const meta = {
  name: 'r6d3b2b-design',
  description: 'Design of record for R-6d3b-2b: wire the durable outbox LIVE into shard/gateway boot + close D-6 #1 restart',
  phases: [
    { title: 'Design', detail: 'DRY bin boot helper + boot replay + RC-2a/RC-2b error disposition' },
    { title: 'Review', detail: '3 opus lenses: boot-sequencing / RC-2a-quarantine / RC-2b-seam-scope' },
    { title: 'Synthesize', detail: 'adjudicate then produce design of record' },
  ],
}

const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['design_markdown', 'sub_slices', 'open_questions'],
  properties: {
    design_markdown: { type: 'string' },
    sub_slices: { type: 'array', items: { type: 'string' } },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
}
const REVIEW_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['lens', 'verdict', 'summary', 'findings'],
  properties: {
    lens: { type: 'string' },
    verdict: { type: 'string', enum: ['SOUND_TO_IMPLEMENT', 'REVISE', 'RECONSIDER'] },
    summary: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['severity', 'title', 'claim', 'evidence', 'fix', 'confidence'],
        properties: {
          severity: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NIT'] },
          title: { type: 'string' }, claim: { type: 'string' }, evidence: { type: 'string' },
          fix: { type: 'string' }, confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
      },
    },
  },
}

const GROUND = [
  'R-6d3b-2b flips the durable outbox LIVE in the shard + gateway bins and closes the RESTART half of D-6 #1 (a source',
  'shard crash losing a producer-less reliable one-shot: a TransientBatch handoff / band-exit Ghost::Despawn). R-6d3b-2a',
  'JUST LANDED (commit 1789e49): the io-prod `replay_outbox(shared, transport, peers, new_incarnation)` machinery',
  '(deadlock-free lock-scoping + count-anchored fence) + `open_node_outbox` (bins/lib.rs:270, opens a NodeOutbox iff',
  'VD_OUTBOX_PATH is set). Today both bins pass `None` to spawn_mesh ⇒ the outbox is inert. Read',
  'scripts/r6d3b_vetted_design.md (§2 + the CORRECTIONS tail) and the /goal audit inputs in docs/design/DEFERRED.md',
  '(the "HOLISTIC /goal AUDIT (wf_c158c7d0)" block — the BINDING RC-2a/RC-2b/RC-1 forward-inputs).',
  '',
  'R-6d3b-2b SCOPE (design all; PROPOSE the sub-slice ordering):',
  '1. DRY BIN WIRING (shard.rs + gateway.rs currently DUPLICATE the boot): both do `spawn_mesh(&cfg-with-',
  '   resolve_process_incarnation, None)` then `build_app(transport)` (shard.rs:22-52, gateway.rs:21-37). Introduce a',
  '   SHARED `vd_bins` helper (e.g. boot_mesh_and_replay) that: resolves the incarnation ONCE (finding C — re-calling',
  '   resolve_process_incarnation double-increments the durable BootCounter, lib.rs:223); opens the outbox',
  '   (open_node_outbox → Option<NodeOutbox>) + wraps it as Option<SharedOutbox> (Arc<Mutex<Box<dyn OutboxSink+Send>>>);',
  '   spawn_mesh(&cfg{incarnation}, shared.clone()); if Some(sh) call replay_outbox(&sh, &mut transport, &peers,',
  '   incarnation) BEFORE build_app consumes the transport; return (transport, control). Both bins call it (HR3',
  '   uniformity — no per-kind fork). The peers/book (cfg.peers) must be available for the replay call.',
  '   KEY FACT (state it): replay_outbox fences on DURABILITY (block A submit) NOT DELIVERY (block C QUIC send), so',
  '   boot-replay does NOT hang on peers that are down at boot — their frames retain + the retransmit timer re-drives.',
  '2. RC-2a (BINDING /goal audit): replay_outbox error DISPOSITION must not WEDGE the boot on a bad row. Today',
  '   replay_outbox hard-errors Undecodable / Unroutable on the first bad row (a `?` in the bin ⇒ main returns Err ⇒',
  '   the node fails to boot). Change so: `Undecodable` (a genuinely frame-corrupt row — NOTE: a version-mismatched',
  '   row is ALREADY skipped by scan_all via decode_value, so this is real on-disk corruption, rare) and a roster-diff',
  '   `Unroutable` (a peer legitimately no longer in the book) are QUARANTINED (skipped + LOUD counted/logged) and the',
  '   boot PROCEEDS with the good rows; ONLY a transient `LaneStuck`/`FenceTimeout` (a wedged lane / stuck peer_writer)',
  '   refuses to boot (fail loud). Design the quarantine SEMANTICS: does a quarantined poison row get gc-swept by',
  '   gc_below (it is unrecoverable anyway), or MOVED to a quarantine keyspace/tag preserved for admin forensics (the',
  '   OUTBOX_FORMAT_VERSION comment at outbox.rs:47 promises "quarantine")? Weigh: forensic-preservation vs simplicity',
  '   vs the gc interaction (gc_below sweeps by incarnation and would take an old poison row). Whatever you choose, a',
  '   quarantined row must NEVER be silently lost-without-a-count, and must NOT wedge the boot. + a poison-row',
  '   boot-progress test.',
  '3. RC-2b (BINDING /goal audit): send_durable folds TrySendError::Full | Closed into ONE SendError::QueueFull',
  '   (mesh.rs:1913-1916). Split `Closed` (the peer_writer task DIED ⇒ the lane is permanently dead) from `Full` (a',
  '   transient full queue) so a dead lane surfaces as UNREACHABLE, not recoverable backpressure — else',
  '   send_durable_with_retry spins ~10s (REPLAY_SEND_MAX_RETRIES × REPLAY_POLL_BACKOFF) on a corpse before LaneStuck.',
  '   Decide the mechanism WITHOUT breaking the frozen sim::io seam: SendError is `enum SendError { QueueFull(Bytes) }`',
  '   in sim::io (FROZEN) — do NOT add a variant to the frozen enum lightly; options: (a) send_durable emits an',
  '   Inbound::NodeUnreachable bounce on Closed (the existing dead-peer signal) + returns QueueFull, and',
  '   send_durable_with_retry checks liveness; (b) observe the peer_writer JoinHandles (spawn_mesh currently DROPS them,',
  '   mesh.rs ~874 handle.spawn(...)) so a dead lane is detectable; (c) a mesh-internal (not sim::io) distinction.',
  '   Pick the one that keeps the frozen seam intact + is DRY. Consider: is RC-2b in-scope for 2b or a small follow-up?',
  'OUT OF SCOPE (name as later slices): RC-3 (the saga BatchHandoff AwaitAdopt phase-split — saga.rs:922) belongs to',
  'R-6d3c (the never-restart closure) + is unreachable without CA-1/L5; do NOT design it here. R-6d4 (SIGKILL e2e +',
  'proptest + store-test-hooks pins). CA-1 (no-DNS k3d peer addressing) — note the caveat but do NOT solve it.',
  '',
  'KEY FILES: crates/bins/src/bin/shard.rs (boot 13-97), crates/bins/src/bin/gateway.rs (boot 12-55),',
  'crates/bins/src/lib.rs (open_node_outbox 270, resolve_process_incarnation 223, other pub helpers), crates/io-prod/',
  'src/outbox.rs (replay_outbox 360-421, ReplayError, gc_below, scan_all, decode_value 127), crates/io-prod/src/mesh.rs',
  '(send_durable 1896-1927, spawn_mesh 784 + the peer_writer spawn ~874, SharedOutbox via crate::outbox),',
  'crates/sim/src/io/mod.rs (the FROZEN Transport/SendError/Inbound seam ~125-370). Coverage: bins are Tier-B process',
  'tier; io-prod Tier-B region-floor 90. Frozen sim::io seam + wire contract MUST stay untouched. No magic numbers.',
  'Be code-grounded; cite file:line. Distinguish a REAL will-break from a not-yet-built roadmap item.',
].join('\n')

phase('Design')
const design = await agent(
  GROUND + '\n\nYou are the DESIGNER. Produce an IMPLEMENT-READY design of record for R-6d3b-2b + PROPOSE the sub-slice ' +
  'ordering. Deliver, all code-grounded (cite file:line): (1) the DRY shared bin boot helper — its exact signature, ' +
  'what it returns, how shard+gateway call it, the open→wrap→incarnation-once→spawn_mesh(Some)→replay-before-build_app ' +
  'sequence, and the proof it does not hang on down-at-boot peers; (2) the RC-2a quarantine disposition — the exact ' +
  'semantics (quarantine vs gc-sweep for a poison row), the replay_outbox signature/return change (does it return ' +
  'counts of {replayed, quarantined}? does the bin decide refuse-vs-proceed, or replay_outbox?), the gc interaction, ' +
  'and the no-silent-loss guarantee; (3) the RC-2b Closed disambiguation — the exact mechanism that keeps sim::io ' +
  'FROZEN, and whether it is in 2b or a follow-up; (4) the tests (a bins boot-replay e2e proving a shard re-drives a ' +
  'pre-seeded outbox on restart, the poison-row boot-progress test, the RC-2b dead-lane test) + how they keep the ' +
  'gates green; (5) the precise files/functions touched per sub-slice; (6) the resolve-incarnation-ONCE + the CA-1/k3d ' +
  'caveat. Flag any place the current API is insufficient + the minimal additive change. Be decisive.',
  { label: 'design', phase: 'Design', schema: DESIGN_SCHEMA, model: 'opus' }
)

phase('Review')
const LENSES = [
  { key: 'boot-sequencing-dry', focus:
    'BOOT SEQUENCING + DRY + no-hang. Verify: (a) the shared helper is genuinely DRY (shard+gateway both use it, no ' +
    'per-kind fork — HR3) and the boot order is correct (open outbox → resolve incarnation ONCE → spawn_mesh(Some) → ' +
    'replay_outbox BEFORE build_app consumes the transport → build_app); (b) resolve_process_incarnation is called ' +
    'EXACTLY once (finding C — a second call double-increments the durable BootCounter); the SAME incarnation feeds ' +
    'cfg.process_incarnation AND replay_outbox(new_incarnation). (c) replay does NOT hang on down-at-boot peers ' +
    '(fence on durability not delivery — verify against replay_outbox + write_frame block A/B/C); (d) the peers/book ' +
    'is threaded correctly to replay_outbox; (e) does opening a live outbox change step_tick / the tick loop at all ' +
    '(it must not — the outbox is transport-side, the tick loop is unchanged)? Cite file:line.' },
  { key: 'rc2a-quarantine', focus:
    'RC-2a QUARANTINE DISPOSITION correctness + no-silent-loss. Verify the chosen semantics: (a) a poison Undecodable ' +
    'row and a roster-diff Unroutable row QUARANTINE + boot-PROCEED (never wedge); a transient LaneStuck/FenceTimeout ' +
    'refuses-boot LOUD. (b) is a quarantined row ever SILENTLY lost (no count/log)? it must be loud-counted. (c) the ' +
    'gc interaction: gc_below(new_incarnation) sweeps by incarnation — does it sweep a quarantined poison row (old ' +
    'incarnation)? Is that correct (unrecoverable anyway) or does it destroy forensics the :47 quarantine promise ' +
    'implies? Is there a path where quarantine + gc COMBINE to lose a RECOVERABLE row (a decodable row mis-classified ' +
    'as poison)? (d) does boot-proceed-with-quarantine preserve the invariant that every GOOD row is re-driven before ' +
    'gc (the count fence must count only the GOOD rows, not the quarantined ones — verify base+N uses the replayed ' +
    'count, not rows.len())? THIS is the subtle one. Cite file:line.' },
  { key: 'rc2b-seam-scope-test', focus:
    'RC-2b + SEAM + SCOPE + TEST. Verify: (a) the RC-2b Closed-disambiguation mechanism keeps the FROZEN sim::io ' +
    'SendError/Transport/Inbound seam intact (no new SendError variant on the frozen enum; a dead lane surfaces via ' +
    'the EXISTING NodeUnreachable or a mesh-internal signal); the peer_writer JoinHandle observation (if used) is ' +
    'sound (spawn_mesh keeps them). (b) is RC-2b correctly scoped (in 2b or a clean follow-up) — does the boot-replay ' +
    'actually NEED it (a dead lane at boot ⇒ 10s LaneStuck then refuse-boot is SAFE, so RC-2b is a fail-FAST ' +
    'improvement not a correctness blocker — confirm)? (c) TEST RIGOR: does the bins boot-replay e2e actually prove ' +
    'a restart re-drives (real process or in-process?)? does the poison-row test prove boot-proceeds? is the dead-lane ' +
    'test deterministic? (d) HR: no magic numbers, no per-kind fork, frozen seam untouched; (e) SCOPE honesty: 2b ' +
    'closes the RESTART case only (never-restart = R-6d3c); the CA-1/k3d no-DNS caveat stated. Cite file:line.' },
]
const reviews = await parallel(
  LENSES.map((l) => () =>
    agent(
      GROUND + '\n\nThe DESIGNER proposed this R-6d3b-2b design:\n--- DESIGN ---\n' + design.design_markdown +
      '\n--- SUB-SLICES ---\n' + design.sub_slices.join('\n') +
      '\n--- OPEN QUESTIONS ---\n' + design.open_questions.join('\n') + '\n--- END ---\n\n' +
      'You are REVIEWER (' + l.key + '). Adversarially review through YOUR lens: ' + l.focus + '\n' +
      'Code-grounded (cite file:line), decisive. SOUND_TO_IMPLEMENT with empty/NIT array if clean; do NOT invent.',
      { label: 'review:' + l.key, phase: 'Review', schema: REVIEW_SCHEMA, model: 'opus' }
    )
  )
)
const valid = reviews.filter(Boolean)

phase('Synthesize')
const synth = await agent(
  GROUND + '\n\nThe DESIGNER proposed an R-6d3b-2b design; three adversarial lenses reviewed it.\n\nDESIGN:\n' +
  design.design_markdown + '\n\nSUB-SLICES:\n' + design.sub_slices.join('\n') + '\n\nREVIEWS (JSON):\n' +
  JSON.stringify(valid.map((r) => ({ lens: r.lens, verdict: r.verdict, summary: r.summary, findings: r.findings })), null, 2) +
  '\n\nSynthesize the ADJUDICATED, IMPLEMENT-READY design of record. Deduplicate (same root cause = ONE, note ' +
  'corroboration). CONFIRM or REFUTE each finding against the code, rank by true impact, FOLD the fix into the final ' +
  'design (do not just list it). Down-grade any reviewer CRITICAL the code/design already handles (cite where); ' +
  'up-grade any real boot-wedge / silent-loss / seam-breach / double-increment hazard. Output the COMPLETE corrected ' +
  'design of record (the DRY helper signature + boot sequence, the RC-2a quarantine semantics + the count-fence-uses-' +
  'good-rows fix if needed, the RC-2b mechanism, the sub-slice ordering, the tests, the files touched) so it can be ' +
  'implemented directly — plus a one-line VERDICT: SOUND_TO_IMPLEMENT (with folded must-fixes) | REVISE | RECONSIDER. ' +
  'Decisive; reward sound design.',
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return { verdict: synth, design_markdown: design.design_markdown, sub_slices: design.sub_slices,
  reviews: valid.map((r) => ({ lens: r.lens, verdict: r.verdict, summary: r.summary, findings: r.findings })) }
