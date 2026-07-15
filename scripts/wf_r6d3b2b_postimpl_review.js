export const meta = {
  name: 'r6d3b2b-postimpl-review',
  description: 'Adversarial post-impl review of R-6d3b-2b (flip the outbox LIVE: quarantine + gc_replayed + Closed-disambig + DRY bin wiring)',
  phases: [
    { title: 'Review', detail: '3 opus lenses: RC-2a/F1 fidelity / RC-2b+seam / bins-wiring+scope' },
    { title: 'Synthesize', detail: 'dedup + rank + verdict' },
  ],
}

const FINDINGS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['lens', 'verdict', 'summary', 'findings'],
  properties: {
    lens: { type: 'string' },
    verdict: { type: 'string', enum: ['COMMIT_CLEAN', 'FIX_BEFORE_COMMIT', 'RECONSIDER'] },
    summary: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['severity', 'title', 'location', 'claim', 'evidence', 'fix', 'confidence'],
        properties: {
          severity: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NIT'] },
          title: { type: 'string' }, location: { type: 'string' }, claim: { type: 'string' },
          evidence: { type: 'string' }, fix: { type: 'string' }, confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
      },
    },
  },
}

const CTX = [
  'R-6d3b-2b (uncommitted, atop 1789e49) FLIPS the durable outbox LIVE in the shard + gateway bins and closes the',
  'RESTART half of D-6 #1 (a source shard crash losing a producer-less reliable one-shot: a TransientBatch handoff /',
  'band-exit Ghost::Despawn). The design of record is scripts/r6d3b2b_vetted_design.md (verdict SOUND_TO_IMPLEMENT +',
  '3 folded must-fixes: F1 gc-must-sweep-only-re-driven-keys [my original gc_below(new_incarnation) was a D-6 #1',
  'regression]; F2 send_durable must NOT consume a msg_id on Closed [frozen FIFO contract]; F3 version-mismatch',
  'silent-sweep, fixed free by F1). THIS review checks the LANDED code.',
  '',
  'WHAT LANDED (review the ACTUAL CODE at HEAD, cite file:line):',
  'Sub-slice A (RC-2a quarantine, crates/io-prod/src/outbox.rs): replay_outbox now returns Result<ReplayCounts,',
  'ReplayError> where ReplayCounts{replayed,quarantined} and ReplayError={LaneStuck,LaneDead,FenceTimeout} (Undecodable',
  '/Unroutable REMOVED — they are now quarantine-continue, not errors). A roster-gone peer (peers-miss) or an',
  'undecodable frame is QUARANTINED (warn + count + continue) — the boot PROCEEDS; the fence targets base + replayed',
  '(NOT rows.len(), since a quarantined row never submits); gc uses the NEW gc_replayed(&replayed_keys) [sweeps ONLY',
  'the re-driven keys] instead of gc_below(new_incarnation) [F1 fix — a NOT-re-driven row is RETAINED]; if replayed==0,',
  'early-return with no fence + no gc. The new_incarnation param was DROPPED (gc is key-based now).',
  'Sub-slice B (RC-2b, crates/io-prod/src/mesh.rs + outbox.rs): a NEW pub supertrait `ReplayTransport: Transport`',
  'with `lane_alive(peer)->bool` (MeshTransport impls via self.lanes.get(peer).tx.is_closed()); the frozen',
  'sim::io::Transport seam is UNTOUCHED. send_durable_with_retry takes &mut dyn ReplayTransport; on the FIRST',
  'QueueFull it checks lane_alive → if dead returns ReplayError::LaneDead (fast-fail, no ~10s corpse-spin).',
  'send_durable itself is UNCHANGED (Full|Closed still fold to QueueFull with NO msg_id consumed — F2 verified: the id',
  'is assigned only in the Ok arm).',
  'Sub-slice C (bins wiring, crates/bins/src/lib.rs + shard.rs + gateway.rs): a NEW pub fn boot_mesh_and_replay(env,',
  'runtime, trust) -> (MeshTransport, MeshControl) — resolves the incarnation ONCE (finding C), opens+wraps the outbox',
  'as SharedOutbox, spawn_mesh(Some), then replay_outbox BEFORE build_app; shard.rs + gateway.rs both call it (HR3 DRY),',
  'the GW-1 snapshot-budget assert moved BEFORE the helper. ReplayError impls Display + std::error::Error (for the bin ?).',
  '',
  'KEY FACTS to verify: replay fences on DURABILITY (block A submit) not DELIVERY (block C send) ⇒ no boot hang on',
  'down-at-boot peers. A version-mismatched row is already filtered by scan_all (decode_value None) before the send',
  'loop. The count-fence needs a REAL re-mirror (only a real MeshTransport peer_writer submits) — so mock-transport',
  'unit tests have replayed==0; the MIXED F1 proof (1 re-driven swept + 1 quarantined retained) is the real-QUIC',
  'mesh.rs end-to-end test.',
  '',
  'GATE (green): vd-io-prod 112 lib + all integration; vd-bins tests green (incl. the real-process process_parity boot',
  'through boot_mesh_and_replay); clippy -D (io-prod + bins) clean; Tier-B --fail-under-regions 90 PASS (TOTAL 94.19%,',
  'outbox.rs 97.09%, mesh.rs 94.44%); workspace build 0.',
  '',
  'SCOPE (honest): R-6d3b-2b closes the D-6 #1 RESTART case; the NEVER-restart case is R-6d3c (saga AwaitAdopt split +',
  'dest TransientDiscard). The process-tier SIGKILL-restart e2e + a boot_mesh_and_replay-increments-counter-by-1 test',
  'are DEFERRED to R-6d4 (bundled with the SIGKILL proofs + store-test-hooks pins). CA-1 (no-DNS k3d) unchanged.',
  '',
  'Find what the green gate does NOT catch: any residual F1 sweep-of-a-non-re-driven-row, a quarantine that silently',
  'loses a recoverable row, a fence off-by-one, an F2 msg_id leak, a frozen-seam breach, a boot-hang, a resolve-twice,',
  'or a coverage/scope dishonesty. Code-grounded (cite file:line). Empty/NIT array if clean; do NOT invent. Rank by',
  'real boot-safety / durable-transport impact.',
].join('\n')

phase('Review')
const LENSES = [
  { key: 'rc2a-f1-fidelity', focus:
    'RC-2a QUARANTINE + F1 gc_replayed FIDELITY + no-loss. Verify against the LANDED code: (a) gc uses gc_replayed(' +
    '&replayed_keys) NOT gc_below — a quarantined/roster-gone/undecodable/version-mismatched row is RETAINED (never ' +
    'swept because it was not re-driven — the D-6 #1 no-loss invariant); confirm replayed_keys contains EXACTLY the ' +
    'send-succeeded keys. (b) the count-fence targets base + counts.replayed (NOT rows.len()) — a quarantined row ' +
    'never submits, so fencing on rows.len() would FenceTimeout; confirm. (c) replayed==0 early-returns with NO fence ' +
    'and NO gc (all-quarantined ⇒ everything retained). (d) the mixed F1 (re-driven swept + quarantined retained) is ' +
    'actually PROVEN by the mesh.rs real-QUIC end-to-end test (not just asserted) — would it catch a regression to ' +
    'gc_below? (e) is there ANY path where a RECOVERABLE row (decodable + routable) is quarantined or lost? Cite file:line.' },
  { key: 'rc2b-seam', focus:
    'RC-2b + FROZEN SEAM + msg_id. Verify: (a) the frozen sim::io::Transport / SendError / Inbound seam is UNTOUCHED ' +
    '— ReplayTransport is a NEW io-prod PUBLIC supertrait, not a modification of Transport; MeshTransport::lane_alive ' +
    'is correct (self.lanes.get(peer).is_some_and(|l| !l.tx.is_closed())). (b) F2: send_durable folds Full|Closed to ' +
    'QueueFull with NO msg_id consumed (the id is assigned only in the Ok arm) — a Closed send does not desync the ' +
    'FIFO msg_id contract; confirm against the code. (c) the lane_alive fast-fail: on the first QueueFull, a dead ' +
    'lane ⇒ LaneDead (fast), a live-but-full lane ⇒ bounded retry then LaneStuck; is the ordering correct (a ' +
    'roster-miss is caught by the pre-send membership check BEFORE send, so a QueueFull here is a real lane state)? ' +
    '(d) is LaneDead correctly a REFUSE-TO-BOOT arm (the peer is in-roster but its writer died = infra pathology)? ' +
    'Cite file:line.' },
  { key: 'bins-wiring-scope', focus:
    'BINS WIRING + DRY + HR + no-hang + SCOPE + TEST. Verify: (a) boot_mesh_and_replay is genuinely DRY (shard.rs + ' +
    'gateway.rs both call it, no per-kind fork — HR3) and resolves the incarnation EXACTLY once (finding C — no ' +
    'double BootCounter increment); the SAME incarnation feeds MeshConfig; replay runs BEFORE build_app consumes the ' +
    'transport. (b) no boot-hang on a down-at-boot peer (fence on durability not delivery). (c) the GW-1 assert moved ' +
    'before the helper is still loud/correct. (d) the None-outbox path (VD_OUTBOX_PATH unset) is byte-identical to the ' +
    'old boot (no replay, inert) — the dev-cluster / process_parity tests exercise it. (e) ReplayError: Display+Error ' +
    'is sound. (f) HR: no magic numbers, frozen seam untouched. (g) SCOPE/TEST honesty: the D-6 #1 restart case IS ' +
    'closed by this (the mechanism runs at boot); the process-tier SIGKILL-restart e2e + the boot-counter-once test ' +
    'deferred to R-6d4 — is that acceptable (is there a bin-level gap that a compile + process_parity + the io-prod ' +
    'end-to-end do NOT cover)? Cite file:line.' },
]
const reviews = await parallel(
  LENSES.map((l) => () =>
    agent(
      CTX + '\n\nYou are REVIEWER (' + l.key + '). Adversarially review through YOUR lens: ' + l.focus,
      { label: 'review:' + l.key, phase: 'Review', schema: FINDINGS_SCHEMA, model: 'opus' }
    )
  )
)
const valid = reviews.filter(Boolean)
const allFindings = valid.flatMap((r) => r.findings.map((f) => ({ ...f, lens: r.lens })))

phase('Synthesize')
const synth = await agent(
  CTX + '\n\nThree adversarial lenses reviewed the LANDED R-6d3b-2b. Raw findings (JSON):\n' +
  JSON.stringify({ verdicts: valid.map((r) => ({ lens: r.lens, verdict: r.verdict, summary: r.summary })), findings: allFindings }, null, 2) +
  '\n\nSynthesize. Deduplicate (same root cause = ONE, note corroboration). CONFIRM or REFUTE each against the code, ' +
  're-rank by true boot-safety / durable-transport impact (down-grade a reviewer CRITICAL the code already handles ' +
  'with the file:line that handles it; up-grade any real residual silent-loss / F1-regression / seam-breach / ' +
  'boot-hang / resolve-twice hazard). For each surviving finding: MUST-FIX-BEFORE-COMMIT / FIX-IN-R-6d3c-or-R-6d4 / ' +
  'OPTIONAL. Then a one-line VERDICT: COMMIT_CLEAN | FIX_BEFORE_COMMIT (list blockers) | RECONSIDER. Decisive + ' +
  'code-grounded; reward the deadlock-free + no-premature-gc + quarantine-no-loss + frozen-seam fidelity where sound.',
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return { verdict: synth, findingCount: allFindings.length, verdicts: valid.map((r) => ({ lens: r.lens, verdict: r.verdict })) }
