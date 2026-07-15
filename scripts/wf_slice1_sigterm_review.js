export const meta = {
  name: 'slice1-sigterm-review',
  description: 'Adversarial review of Slice 1 (SIGTERM graceful-drain in the 3 server bins)',
  phases: [
    { title: 'Review', detail: 'opus reviewers per dimension, read-only' },
    { title: 'Adjudicate', detail: 'verdict + must-fix list' },
  ],
}

const SCOPE = `
Cloud-ready k3d Slice 1 — SIGTERM graceful-drain. The change (UNCOMMITTED; run \`git diff\` + read the files):
- crates/bins/src/lib.rs: NEW \`install_shutdown_flag(runtime: &tokio::runtime::Handle) -> Arc<AtomicBool>\` —
  spawns ONE task on the existing tokio runtime awaiting SIGTERM (tokio::signal::unix terminate()) or SIGINT
  (ctrl_c), flips the flag (Relaxed). Err-installing-SIGTERM degrades to ctrl_c only. tokio \`signal\` rides
  features=["full"] (no new dep).
- crates/bins/src/bin/{shard,gateway,orchestrator}.rs: the bare \`loop { step_tick(); pacer.wait(); }\` became
  \`while !shutdown.load(Relaxed) { ... }\` + a graceful drain after the loop, then \`Ok(())\`.
  * shard/gateway: clean return (their R-6d outbox is fsync-before-send; in-flight mesh sends are
    recovery-covered by the outbox replay) — node/runtime/_control drop, tearing down the endpoint.
  * orchestrator: flush the FINAL parked outbox (durability.wait_durable_through(seq) + node.flush_outbox),
    then return → node drops → the boxed RedbStore's Drop drains the writer channel + JOINS the off-tick
    writer thread (final fsync). runtime (declared earlier) drops AFTER node.
- justfile: image-drain-smoke recipe — docker stop (SIGTERM, 10s grace) the containerized orchestrator, assert
  it exits 0 (clean drain) NOT 137 (SIGKILL escalation). PASSED: exit 0, log "orchestrator drained on shutdown".

CONTEXT: the orchestrator's own tick-loop comment already established today's signal-death is RECOVERY-COVERED
(rehydrate + idempotent saga re-drive; all egress is re-driven saga commands + loss-tolerant ClockSync), and
RedbStore::Drop already does a graceful writer-join — so this slice just makes the loop BREAK so that existing
drain runs. FAILURE MODE IS BOUNDED: an imperfect drain degrades to today's recovery-covered abrupt stop. The
value: routine pod-stop/rolling-deploy becomes a clean flush-and-exit instead of a hard SIGKILL crash-path.
6 HARD RULES apply. End-goal: signal-heavy voxel MMO, hundreds of users, k3d/k8s deploy. You are READ-ONLY
(Explore): git diff + read; do NOT edit. Cite file:line.`

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['dimension', 'verdict', 'findings'],
  properties: {
    dimension: { type: 'string' },
    verdict: { type: 'string', enum: ['clean', 'concerns', 'blocking'] },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'title', 'where', 'detail', 'fix'],
        properties: {
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          title: { type: 'string' },
          where: { type: 'string' },
          detail: { type: 'string' },
          fix: { type: 'string' },
        },
      },
    },
  },
}

const DIMS = [
  {
    key: 'drain-correctness-hang',
    prompt: `${SCOPE}

Review DRAIN CORRECTNESS + HANG-FREEDOM. Read the 3 bins' loops + drains, crates/io-prod/src/store.rs
(RedbStore::Drop, the writer JoinHandle, wait_durable_through, DurabilityHandle), and how the orchestrator's
node owns the boxed store. Probe adversarially:
1. Local DROP ORDER: does \`node\` really drop BEFORE \`runtime\` (so RedbStore::Drop's writer-join runs while
   the tokio runtime is still alive — or does the writer thread not need the runtime)? Is the store writer a
   std thread (runtime-independent) or a tokio task? Confirm no drop-order hazard.
2. HANG: can the orchestrator drain HANG forever (RedbStore::Drop joining a wedged writer; wait_durable_through
   on a dead writer)? If it can hang, is it bounded by the k8s/docker SIGKILL escalation (so bounded, not a
   true wedge)? Is wait_durable_through on the FINAL parked flush safe if the writer already died?
3. MISSED/LATE signal: the flag is checked at loop-top; pacer.wait() blocks mid-iteration. Worst-case latency
   from SIGTERM to loop-break = one tick period — acceptable vs a 10s+ grace? Any path where the flag is set
   but the loop never re-checks (e.g. a step_tick that blocks/panics)?
4. shard/gateway clean-return: is it TRUE they have no un-fsynced durable state to lose (the R-6d outbox is
   fsync-before-send)? Any in-flight durable write that a clean return could drop? Is dropping the runtime
   mid-mesh-send safe (no corruption — only recovery-covered RAM loss)?
Return verdict + findings.`,
  },
  {
    key: 'signal-helper-race',
    prompt: `${SCOPE}

Review the install_shutdown_flag HELPER + its concurrency. Read crates/bins/src/lib.rs install_shutdown_flag.
Probe: (1) the spawned task's lifecycle — does it leak / is it cancelled cleanly when the runtime drops? (2)
the tokio::select! on term.recv() vs ctrl_c() — correct? any lost-wakeup? (3) Relaxed ordering on the flag —
sufficient given the sync loop only polls this one flag with no other ordering dependency? (4) the
Err-fallback-to-ctrl_c arm — correct + does it ever hard-fail boot? (5) is spawning a signal handler on a
multi-worker runtime sound (tokio signal registration is process-global — any issue with multiple bins in one
process? they are separate processes, so N/A, but confirm)? (6) does catching SIGINT (ctrl_c) interfere with
any existing Ctrl-C behavior / the process-tier tests that spawn+kill these bins? Return verdict + findings.`,
  },
  {
    key: 'dod-hr-scale',
    prompt: `${SCOPE}

Review the DoD HONESTY, HR compliance, and SCALE/end-goal fit. Probe: (1) is the image-drain-smoke proof
HONEST — does exit-0-not-137 truly prove the drain ran (vs the process just exiting for another reason)? The
recipe greps the "drained on shutdown" log — is that assertion load-bearing or cosmetic? Should shard/gateway
ALSO have a drain proof, or is the orchestrator (the only bin with a writer-join + parked flush) the
sufficient case? (2) HR: is the shared helper DRY across the 3 bins (HR-style one-tooling)? Any per-bin
divergence that should be unified? (3) k8s fit: the drain must finish within terminationGracePeriodSeconds —
is that documented/bounded? Does anything here need a k8s manifest knob (Slice 4)? (4) is there a coverage/test
gap — vd-bins is Tier-B (not the 100% gate), but is the drain path exercised by a repeatable test (the docker
recipe is not in \`just gate\`/CI)? Should it be wired into a gate? (5) does this compose with the future
RELIABLE non-re-driven egress (the orchestrator comment says a graceful-shutdown flush is 'owed if/when one
lands') — is the current best-effort parked-flush the right seam for that? Return verdict + findings.`,
  },
]

phase('Review')
const reviews = await parallel(
  DIMS.map((d) => () =>
    agent(d.prompt, { label: `review:${d.key}`, phase: 'Review', schema: SCHEMA, agentType: 'Explore', model: 'opus' })
  )
)

phase('Adjudicate')
const packed = reviews.filter(Boolean).map((r) => JSON.stringify(r)).join('\n\n')
const verdict = await agent(
  `You are the adjudicator for the Slice 1 (SIGTERM graceful-drain) post-impl review. ${SCOPE}

The ${reviews.filter(Boolean).length} dimension reviews (JSON):

${packed}

Adjudicate each finding: real blocker (must fix before commit), ledgerable follow-up, or false alarm — with
why + file:line. Keep the bounded-failure-mode context in mind (an imperfect drain degrades to today's
recovery-covered stop, so only a NEW hazard — a hang worse than the SIGKILL fallback, a data-loss the old
path didn't have, or a broken/dishonest DoD — is a true blocker). Final verdict: SOUND_TO_COMMIT (+ ledger
items) or NEEDS_FIX (+ exact must-fix list). Do NOT edit any file.`,
  { label: 'adjudicate', phase: 'Adjudicate', agentType: 'Explore', model: 'opus' }
)

return { reviews: reviews.filter(Boolean), verdict }
