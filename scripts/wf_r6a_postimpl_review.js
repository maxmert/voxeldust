export const meta = {
  name: 'r6a-postimpl-review',
  description: 'Adversarial post-impl review of R-6a (durable monotone boot-counter + durable-path guard)',
  phases: [{ title: 'Review' }, { title: 'Synthesize' }],
}

const ROOT = '/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system'

const CONTEXT = [
  'R-6a just landed (uncommitted) in the voxeldust transport arc: the M3 DURABLE MONOTONE process-incarnation',
  'boot-counter — the last cloud-blocking precondition (a k8s CrashLoopBackOff sub-second restart mints an',
  'EQUAL wall-clock-ms incarnation ⇒ the receiver ledger silent-Dedups the restarted node\'s reliable traffic;',
  'a clock rewind mints a LOWER one ⇒ StaleIncarnation-drops it all). The cure: a counter derived only from',
  'its own prior durable value + 1.',
  '',
  'WHAT LANDED:',
  '  - crates/io-prod/src/boot.rs (NEW, std-only, no dep): `BootCounter::increment_on_boot(path, genesis_floor)',
  '    -> Result<u64>` reads a 24-byte sidecar file (magic(8)+counter_le(8)+FNV-1a-checksum_le(8)); ABSENT ⇒',
  '    genesis at `genesis_floor.max(1)`; VALID ⇒ `prev.checked_add(1)`; PRESENT-but-CORRUPT (wrong len/magic/',
  '    checksum) ⇒ `Err(Corrupt)` (FAIL LOUD, never reset to a lower value). The write is atomic-durable:',
  '    write-tmp → `sync_all()` (fsync data) → `rename(tmp, path)` (atomic) → open dir + `sync_all()` (fsync',
  '    the rename, best-effort). Plus `check_durable_path(path, durable_root: Option, ephemeral_ok)`: an',
  '    ALLOW-list (path must canonicalize under the declared durable_root) when a root is given, else a temp',
  '    DENY-list; `canon_lenient` resolves symlinks in the longest existing prefix (macOS /var → /private/var,',
  '    even for an uncreated subdir). 8 unit tests (genesis-floor, monotone-across-a-leftover-tmp, rewind-',
  '    survival, corrupt-fail-loud, overflow-guard, guard allow/deny/escape, Display).',
  '  - crates/bins/src/lib.rs: `resolve_process_incarnation(env)` — precedence: (1) VD_PROCESS_INCARNATION',
  '    explicit WINS (dev/test via common_env + future orchestrator-issued); (2) else VD_BOOT_STATE_DIR ⇒ the',
  '    durable self-counter (guarded by check_durable_path with VD_BOOT_DURABLE_ROOT allow-list or the temp',
  '    deny-list; VD_BOOT_STATE_EPHEMERAL_OK=1 escape; genesis_floor = launch_incarnation() so a re-provisioned',
  '    pod with a fresh volume still exceeds a peer\'s surviving wall-clock-era ledger); (3) else FAIL LOUD',
  '    unless the ephemeral escape. Plus `parse_bool_env` (the strict 1/true/yes|0/false/no|error ladder). 4',
  '    unit tests. The 3 bins (orchestrator/gateway/shard) now call resolve_process_incarnation.',
  '',
  'STATE: io-prod lib 81 pass (boot 8), bins lib 4 pass, process_parity green (real bin boot path via',
  'common_env ⇒ precedence #1), clippy -D + fmt clean, Tier-B boot.rs 94.17% region. No frozen sim::io seam',
  'touch (incarnation is a MeshConfig field). No new dep. No HR3 fork (one mechanism, all node kinds).',
  '',
  'DELIBERATE SCOPE-DOWN (adjudicate): the vetted design wanted the store guard (orchestrator.rs VD_STORE_PATH',
  'deny-list) ALSO lifted into the shared check_durable_path for full DRY. R-6a instead added the shared',
  'helper + applied the ALLOW-list to the NEW boot path, and LEDGERED the orchestrator store-guard unification',
  'as an immediate follow-up (to avoid touching the orch-crash-tested store boot path in this slice). Is that',
  'scope-down sound, or must the store guard be unified now?',
  '',
  'Read: crates/io-prod/src/boot.rs (the whole module + tests); crates/bins/src/lib.rs (resolve_process_',
  'incarnation, parse_bool_env, BOOT_COUNTER_NAME); crates/bins/src/bin/{orchestrator,gateway,shard}.rs (the',
  'call sites); crates/io-prod/src/mesh.rs (classify_reliable A1 incarnation ladder ~358, process_incarnation',
  'stamping ~795); crates/bins/src/bin/orchestrator.rs (the existing VD_STORE_PATH deny-list guard, for the',
  'DRY scope-down question). Use git -C ' + ROOT + ' diff HEAD.',
].join('\n')

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

const LENSES = [
  {
    key: 'durability-monotonicity-crashsafety',
    prompt: [
      'LENS: the durability + monotonicity + crash-safety of the BootCounter AS IMPLEMENTED. Verify the fsync',
      'ordering in write_durable: is the DATA fsync\'d before the rename, and is the rename the durability',
      'point (so a crash before it leaves `path` untouched)? Is the crash-during-increment argument correct in',
      'the CODE — a crash before the rename completes re-uses a value that was NEVER on the wire (safe because',
      'the value is durable before the bin stamps frames), and a crash after advances by exactly one? Attack',
      'the "durable before spawn_mesh" invariant: does resolve_process_incarnation actually complete the write',
      'BEFORE the mesh is spawned in each bin (so a re-used value is never wired)? Is the checksum (FNV-1a)',
      'genuinely torn-write-detecting, and is corrupt-fail-loud correct (a present-but-bad file NEVER resets',
      'to genesis)? Is the genesis-floor (launch_incarnation) sound — does it actually keep a re-provisioned',
      'node above a surviving wall-clock-era ledger, and does it ever REDUCE monotonicity (e.g. a floor lower',
      'than the file value)? Does a strictly-higher incarnation on restart actually make classify_reliable',
      'reset the dedup high-water (verify against mesh.rs A1)? Is the best-effort dir-fsync (ignored on',
      'failure) acceptable given the data-fsync+atomic-rename already bound the loss?',
    ].join('\n'),
  },
  {
    key: 'guard-precedence-scope-dry',
    prompt: [
      'LENS: the durable-path guard + the precedence + scope/DRY/HR. Attack check_durable_path: does the',
      'allow-list (canonicalize under durable_root) actually catch a k8s emptyDir (mounted outside the declared',
      'root), and does canon_lenient correctly resolve symlinks for an UNCREATED subdir (the macOS /var case)?',
      'Any bypass (a `..` traversal, a symlink escaping the root after canonicalization)? Is the deny-list',
      'fallback still correct? Attack resolve_process_incarnation precedence: is it BACKWARD-COMPATIBLE (every',
      'existing test/dev path sets VD_PROCESS_INCARNATION via common_env ⇒ #1 wins — confirm nothing relied on',
      'the old default-0), and is the fail-loud-on-misconfig (#3) right (a prod pod that forgets both must not',
      'silently revert to wall-clock)? Is the explicit-but-unparseable VD_PROCESS_INCARNATION handled (loud,',
      'not a silent fall-through)? Rule on the DELIBERATE SCOPE-DOWN (the orchestrator store guard not yet',
      'unified into the shared helper — sound to defer + ledger, or a real DRY defect that must land now)?',
      'Confirm no frozen-seam touch, no new dep, no HR3 fork, no magic-number leak (the file name, the 24-byte',
      'layout, the env keys — are they named consts / one config)?',
    ].join('\n'),
  },
]

phase('Review')
const reviews = await parallel(
  LENSES.map((l) => () =>
    agent(
      [CONTEXT, '', 'YOUR REVIEW LENS:', l.prompt, '',
       'Read the ACTUAL implemented code before judging. Be adversarial + concrete (file:line). A finding names',
       'a REAL defect (a non-monotone/lossy/non-durable path, a guard bypass, a backward-incompat, a seam/dep/',
       'HR violation) with a specific fix. If sound on your lens, say so. Reserve CRITICAL for a durability',
       'correctness break (a re-used-on-the-wire value, a silent reset-to-lower, a guard that lets an ephemeral',
       'path through) or a backward-incompatible boot regression.'].join('\n'),
      { label: 'review:' + l.key, phase: 'Review', schema: SCHEMA, model: 'opus' }
    )
  )
)

phase('Synthesize')
const synth = await agent(
  [CONTEXT, '', 'Two adversarial reviewers returned verdicts on R-6a:', JSON.stringify(reviews, null, 2), '',
   'Adjudicate against the ACTUAL code (read boot.rs + the bins to confirm/refute). Produce: (a) the',
   'deduplicated, severity-ranked REAL findings that must change before commit, each with a concrete fix',
   '(CRITICAL/HIGH blocking; MEDIUM fix-now-if-cheap; LOW ledger); (b) explicit rulings on whether the',
   'BootCounter is genuinely crash-safe + monotone + durable, whether the guard + precedence are correct +',
   'backward-compatible, and whether the store-guard-unification scope-down is sound; (c) a final verdict:',
   'SOUND_TO_COMMIT or FIX_BEFORE_COMMIT. Be decisive, cite file:line.'].join('\n'),
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return { reviews, synthesis: synth }
