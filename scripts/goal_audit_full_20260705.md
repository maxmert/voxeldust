# Voxeldust Greenfield Rebuild — Final /goal Audit (HEAD 0090c1f, 2026-07-05)

Ten audit dimensions, 43 findings, every MEDIUM+ adversarially verified against code and docs. Verdicts: 2 SOUND (compose-architecture, dry-onetooling), 8 CONCERNS, 0 UNSOUND.

---

## 1. THE ANSWER

**Yes, with conditions — the architecture can reach the end goal; the roadmap as currently written cannot.** Nothing built contradicts the vision, no landed decision forecloses it, and the composed transfer/durability stack is genuinely production-grade. But the audit found that the three pillars that make this game *this game* — hundreds-in-one-location, signal-heavy cross-shard cities, and multishard meshes — are exactly the three things no phase currently owns. The conditions are scheduling and ownership fixes, not rewrites.

The load-bearing reasons:

1. **The foundation is real, not aspirational.** One transfer FSM, one wire file, one outbox, sealed shards enforced by compile-time tripwires (`crates/wire/tests/intershard_closed.rs`), 100% Tier-A coverage with zero exemptions in use, byte-identical determinism gates in the default suite, and SIGKILL-tier process proofs. Both re-drive paths funnel into ONE dest-side idempotency journal — the compose-architecture dimension traced double-adopt, double-delivery, lost-flow, and wedge classes and found the stack coherent (verdict SOUND).
2. **Every future feature seam is planted, additively.** Reserved `InterShardFlow` arms for BlockEdit/Coupling/Signal (`wire/src/intershard.rs:22-23`), the capability lattice, `EffectFree` coupling, `LatticePos` wire shape, D-33/D-35/D-39/D-42 all ledgered with in-code pins (F39). Ships, warp, and PvP compose onto what exists.
3. **BUT the headline scale number has no owner (F1, HIGH).** Whole-realm broadcast (`stub.rs:2809-2884`) walls out at roughly N≈100-200 co-located players — exactly ON the goal — the D-9 AoI cure has inconsistent phase dates across DEFERRED/PLAN/roadmap, roadmap.json P11 terminates at "dozens," and measured evidence stops at 32 sessions. No phase gates a hundreds-run.
4. **The signal model and the city topology contradict (F6, HIGH).** `sealed_shards.md:140` defines Local scope as "zero network (stays in one realm's SignalGraph)"; the user-resolved D-41 direction makes a dense city a realm SPLIT across N region shards. A switch→lamp wire crossing a region seam has no carrier, no semantics, no owner. This is a compose-conflict at the heart of "signal-heavy planet cities" — cheap to fix now, expensive after P4-P6 harden.
5. **The seams protecting the split-realm future are prose-only (F3, HIGH)** — D-41's own entry admits "there is NO exists-to-be-flipped TEST" (DEFERRED.md:2683-2692), and the phase-flip gate already failed once in exactly this way (D-9's P2 label went stale uncaught).
6. **Multithreading is architecturally right and retrofit-safe (see §4)** — the concern is that three future compute-heavy paths (terrain gen, signal eval, meshing) have no owned threading model, not that anything must be undone.

The distance from here to "AAA MMO quality" is: (a) write the missing owners into the ledger and roadmap now at design-time cost, (b) close P2's own client tail, (c) ratchet load evidence on a dated schedule instead of jumping from 32 to "dozens" at P11.

---

## 2. What is genuinely sound

**The transfer-first inversion is being honestly executed.** Transfers got at-least-once + incarnation + cumulative-ack + durable outbox + boot-replay + SIGKILL process proofs *before* a single voxel exists — the exact structural inversion of the old project's failure. The R-1..R-6 core kills old root-cause R9 and is the carrier every future cross-shard feature (signals included) will ride: the signals design already specifies `Signal{SignalId,…}` idempotent on `(SignalId, step_id)` in the same applied_steps discipline (generic_transfer.md HR1 section; F36) — the robustness proof transfers directly rather than needing a parallel mechanism.

**The composed re-drive stack has no invariant conflict (compose-architecture: SOUND).** Outbox boot-replay and saga Timeout re-emit both funnel into one dest `(TransferId, step_id)` journal with discard-poisoning covering both interleave orders; incarnation is single-sourced boot→lane→outbox-key→Reset-ladder; the count-anchored GC fence in replay_outbox provably cannot sweep an undelivered row.

**One-tooling is enforced by machinery, not convention (dry-onetooling: SOUND).** Exhaustive `arm_tripwire` + effect-class conformance gates; ONE saga FSM with Durable/Transient as policy fan-out (`sim/src/saga.rs:342-377`); ONE flow-encode path with the producer-less-reliable guard (`sim/src/runtime.rs:63`); zero shard-kind matches in feature code; decode-to-Default hard-banned (`core/src/tlv.rs:58,266`).

**The test net is anti-theater.** Anti-vacuity asserts (`tests/src/lib.rs:803-858`), RED controls, fresh-store anti-theater controls in p3_orch_kill, honest-RED cells, cross-process byte-identical chaos traces (p0_gates.rs:77), kill-9-mid-fsync in the pre-merge gate. Empty coverage-exemptions.toml.

**The durability spine is load-shaped (F37).** Off-tick fsync writer, depth-1 crash-loss bound, group-commit ~1 fsync/tick, no fsync under the shared outbox lock (`mesh.rs:1244-1262`), starvation regression-tested — redb is not the scale cliff.

**The built client seam is already N-sub general (F40).** Tracks keyed `(SubId, EntityId)` (`view.rs:39`), authority-else-lowest-sub dedup, `BTreeSet` of held subs; the only cap is the gateway's one-sub-per-(session,shard) keying — ledgered (D-39.5) and correctly coupled to D-9. No-prediction is structural: `RenderPose` has no velocity field; `interp.rs:110-139` clamps, never extrapolates.

**The DEFERRED registry itself is an asset.** It has repeatedly caught and closed its own honesty holes, and most of this audit's negative findings were *already characterized* in it — the failures below are the places where it, too, has a gap.

---

## 3. Confirmed gaps, ranked

### HIGH

**F1 — Hundreds-in-one-location has no owning phase gate; all load evidence stops at 32 sessions.**
*Evidence:* `emit_frames` (`sim/src/stub.rs:2809-2884`) sends every emitting dot to every gateway, observer-blind; gateway Arc-fans identical bytes per subscriber (D-9). At ~100B/`EntitySnap` vs the 1200B datagram budget (`wire/src/channels.rs:202`), N=300 co-located ≈ 4.8 Mbps/client, ~1.4 Gbps gateway egress at 20 Hz — the wall sits at N≈100-200, on the goal number. D-9's owner is inconsistent ("P6 InterestSet schedule, or earlier" vs roadmap.json interest work at P11) and its P2 label already went stale uncaught (DEFERRED.md D-9, "the gate could not catch it"). roadmap.json P11: "Scale beyond dozens is explicitly deferred." Ceiling of evidence: 32 in-process sessions (p1_gates.rs:115-165), 16 mesh peers (mesh_load.rs).
*Why it matters:* this is the single seam that makes the game an MMO rather than a lobby shooter.
*Fix and where:* (a) re-own D-9 + D-39.5 as ONE coupled slice with one phase name consistent across DEFERRED/PLAN/roadmap — P6 boundary at latest; (b) add an explicit post-P8 region-split phase owning D-41's N>1 sim + standing AuthorityChanged producer; (c) **insert at the P3→P4 seam now:** a cheap in-process N≥128-one-realm fixture asserting per-client snapshot bytes scale with visible neighbors, not realm population — the measured floor the P6 reshape must beat; (d) dated evidence rungs 32→64→128→goal.

**F3 — D-41 plant items (2)-(5) are prose-only pins with no tripwire tests, guarding against exactly the failure mode that already happened once.**
*Evidence:* DEFERRED.md:2683-2692, the entry's own admission: "pinned ONLY in prose doc-comments today — there is NO exists-to-be-flipped TEST"; the tripwires are owed "with" the very P4/P5 work they must guard; sealed_shards.md:236 + PLAN.md:42 mandate the P6 single-cluster HARD ERROR the seam must first soften; foreclosure cost is a ~14-file wire rewrite (DEFERRED.md D-41).
*Fix and where:* land the tripwires as a small standalone slice **before the first P4 terrain slice**: a >1-anchor-representable test gating the single-cluster hard-error, the action_bits-inert unit test, a red-if-radial-only OverlapBand constructor assertion. Hours of work, fully specified in the entry already.

**F6 — The P9 signal model is realm-scoped but the resolved city topology is realm-SPLIT; the spanning SignalGraph is unowned by any phase or entry.**
*Evidence:* sealed_shards.md:140 ("Local = zero network… The rest cross realms"); roadmap.json P9 repeats the realm==shard assumption; D-41 (DEFERRED.md:2511-2513) resolves dense city = N region shards per realm; D-41's plant and DEFER lists contain no signal-partition item; grep across all design docs returns nothing.
*Why it matters:* "signal-based heavy" + "cities spanning shards" is the user's core differentiator; a Local wire across a region seam currently has no representation at all.
*Fix and where:* add a plant item to D-41 (or a new D-entry) **now, at zero code cost**: graph edges crossing a region seam become fenced cross-shard Signal frames with declared per-hop tick latency, per-channel ordering, a deterministic partitioned-evaluation contract, and a split heuristic that prefers not to cut wired structures. Required section of the P9 charter; tripwire: a G-IDENTICAL fixture with a wire crossing a (degenerate today) region boundary.

### MEDIUM (compressed, grouped)

**Robustness tail owed at the current phase (P3):**
- **F4** — D-36 starved delivery watermark parks a Promoting saga forever (the Timeout producer re-drives Promote but cannot re-arm `DeliveredToObservers`, gateway.rs:851-888 / saga.rs:1115-1145); composes badly: every D-37 re-home lands at a target with no gateway sub, and the reaper skips live-saga subjects. Land the rising-edge delivery re-poke in P3 as ledgered, plus a crash-matrix cell that BLOCKS a dest frame path.
- **F7** — seeded breadth chaos (drop_p/dup_p × multi-seed × crash schedules) has never touched the real transfer cluster (only delay-only LinkPolicy, p2_transfer_gates.rs:551-553); the crash matrix's header claims GATEWAY victims (p3_crash_matrix.rs:7) but zero such cells exist — and the gateway row is the one D-29/D-22 predict is lossy. Land P3 Slice 1b before P4 opens; gateway-victim cells first.
- **F8** — D-29 one-shot OpenInputSlot + take-drained cut buffer = guaranteed input loss on the first real multi-pod deploy (gateway.rs:1516-1546, in-code admission at :204-217). Land the CONFIRMED slot-open step with the D-28 input-conservation gate; gate multi-pod work on it.
- **F17** — drop counters exist everywhere but only the orchestrator publishes an admin snapshot (D-10); a gateway shedding cut buffers is invisible at 2am. Due NOW by the entry's own WHEN; make the real-node matrix assert against exposed counters.
- **F18** — a reliable frame shed from OutboundBox staging (`node/src/app.rs:283-315`) bypasses BOTH the durable outbox and the liveness bounce — a permanent unowned loss/park for producer-less one-shots toward a congested-but-alive peer. Minimum: a DEFERRED entry; better: never shed reliable from staging, or emit a synthetic NodeUnreachable.
- **F19** — the deploy-blocking mesh soak runs with outbox=None (mesh_load.rs:110,181); the production durable-before-send path has zero load evidence. Add an outbox-enabled variant now or before P6 transient storms.

**Scale-up and de-risking (schedule before P4/P5):**
- **F10** — the per-shard colliding-player ceiling is gated only by a *conditional* D-41 spike whose triggering bench no phase owns; roadmap.json numbers only SPIKE-0a/2a/3a/6a/10a. Register a numbered blocking pre-P5 spike with a benched colliding-capsule budget as a P5 ENTRY criterion; note rapier's `parallel` feature is documented incompatible with `enhanced-determinism` — expect single-threaded-per-island + ordered merge.
- **F11** — SPIKE-6a's Category-A half actually gates **P4**, not P5: P4's DoD demands byte-identical terrain cross-binary/cross-platform (PLAN.md:174) while the binding policy explicitly does NOT guarantee cross-platform floats (test_harness.md:341) and existing Cat-A code uses platform-libm transcendentals (`core/src/celestial.rs:49-86`). Split the spike; pin the terrain numeric policy (integer/hash noise or the libm crate) in the P4 design; run the rapier snapshot half in parallel with P4.
- **F20** — the single-orchestrator saga-throughput soak D-41 marks "owed NOW" does not exist anywhere in tests/. A synchronized Durable crowd = O(N) sagas + O(live-sagas)/tick scan_deadlines. Schedule as a concrete P3-era load slice with the D-3 deadline-index sizing.
- **F21** — cross-region collision response (a contact straddling two region shards) is one sentence (DEFERRED.md D-41: "exactly ONE region-shard resolves any straddling contact"); the impulse-feedback path to the ghost's owner is undesigned. Short design charter before P5.

**Signals (the P9 charter's required sections):**
- **F9** — "signal-based HEAVY" has no scale owner on ANY axis: no eval-compute budget/threading model (the only parallelism lever triggers on a *rapier* bench), no batching/coalescing tier (individual SignalFrames on a byte-capped reliable lane would trip the shed ALERT under normal city load, wire/admin.rs:174-176), no signal-volume gate in any DoD — violating the standing load-tests-land-with-the-subsystem rule. Own all three in the P9 charter or a DEFERRED entry now; extend the D-41 spike trigger to "any benched per-shard compute ceiling."
- **F22** — the cross-shard signal ordering/consistency contract (apply-by-which-clock, relay-hop FIFO, skew bound) is unstated; the R-2b lane split explicitly removes cross-class ordering. State it in the P9 charter with a cross-shard skew harness scenario.

**Client track and DRY:**
- **F5** — the band-crossing AUTO-TRIGGER has no production producer and no DEFERRED owner: `start_transfer` (saga_runtime.rs:470) has zero non-test callers; every transfer is harness-fabricated. The geometry half exists and is tested (`core/src/geometry.rs:67-169`). Add D-43; land with D-15/D-30; append to D-30's blocker list so the paired visual can't pass with a scripted trigger.
- **F12** — five P2-dated items (D-4, D-5, D-14, D-15, D-30) aged unmoved while the transport arc recursed to depth R-6d4-F2 (37 R-commits in 8 days); DEFERRED still heads "P2 — THE TRANSFER (in progress)". Sanctioned deferral, but a label-hygiene defect repeating D-9's failure mode. Pivot, re-date, flip P2.
- **F14** — three parallel client session FSMs (ClientState / ScriptedClient / ProcessClient) hand-synchronized; the gates exercise the harness FSM while the product FSM lags it (D-5: net.rs still ignores RequestCut). Fold onto the transport-generic ClientState at Slice 1e, before D-4/D-11/D-35 each triple the edit surface.
- **F15** — the shard-generic transfer-endpoint machinery lives unowned inside the 6,924-line stub.rs with no DEFERRED entry naming its extraction when P4 retires the stub — the one un-ledgered HR3/HR4 seam. Add the entry, owned at P4 entry.
- **F13** — `worker_threads(2)` is a bare literal in all four production bins, violating the project's own one-config rule, and 2 is exactly the value the R-6d3a starvation class exhausted. Add `VD_IO_WORKERS` to EnvConfig with the load-test track.
- **F24** — the own-avatar input-to-photon cost under no-prediction (~RTT + tick + 120ms buffer ≈ 220-300ms at internet RTT) is quantified nowhere and validated only at localhost RTT≈0. Add a DEFERRED entry pinning the budget + a FaultFabric feel-test at 80-120ms simulated RTT; the mitigation levers are all pre-P11 decisions.

**Notable LOWs (fact-state corrections):** F2 — the claim "D-6 #1 closed for BOTH restart and never-restart" over-states: never-restart *detection-in-prod* is CA-1/L5-gated (DEFERRED.md:1731-1733), and the CA-1 decision is HELD with no date while it also blocks the entire cloud/K8s scale-out story (static literal-IP peer book; two k3d pods cannot address each other). Resolve and date it — before P10 at the latest. F16 — the redelivered-trigger-after-completion zombie (no persisted Tombstone) is now the norm-case under at-least-once, not an edge; land with the 1d idempotent-trigger slice. F25 — the durable outbox is configuration-latent (VD_OUTBOX_PATH never set by any launch path). F33 — mechanical/moving blocks (subgrids/pistons/rotors), inventory, crafting/economy, and scripting have no phase and no ledger entry, and P11's demo line over-claims "the complete Voxeldust vision" — add a post-P11 content-horizon section.

---

## 4. The multithreading verdict

**Sound for what is built; under-owned for what is coming. No rework is baked in.**

The single-threaded deterministic sim per shard is the *correct* engineering call, not an omission: `ExecutorKind::SingleThreaded` (`node/src/app.rs:99`), tokio strictly below the `sim::io` seam via crossbeam bridges (test_harness.md:146-160), fsync on a dedicated writer thread with a designed per-Realm parallel pool, a pre-authorized read-only `par_iter` carve-out (test_harness.md:212), and process-level shard parallelism as the primary scale axis. The rayon-not-adopted decision was explicitly recorded and user-flagged (DEFERRED.md:2244-2246). The corner check passed: no non-Send types or hidden globals in sim/node, so deterministic fork-join parallelism retrofits cleanly (F38). io-prod is clean of blocking-in-async (the writer block_on runs on a dedicated OS thread, io-prod/lib.rs:385-413), production queues are bounded with counted sheds, and the one real starvation bug found in this class was cured with a regression test.

What falls short of "multithreading owned where it will be needed":

1. **P4 terrain generation** — the classic worker-pool job, seed-deterministic and thus embarrassingly parallel, yet no doc says where it runs on server or client (F23). Own it in the P4 design: off-tick chunk-gen pool behind a sim::io-style seam; client meshing on Bevy's compute task pool.
2. **P9 signal-graph evaluation** — the user's headline feature has no compute model; the only vertical-parallelism trigger is a *rapier* bench (F9). Decide the deterministic-parallel shape (topological-level fork-join, stable-id-ordered merge) in the charter.
3. **P5 physics ceiling** — the parallel-step spike is conditional prose, not a numbered blocking spike, and rapier's `parallel` feature conflicts with `enhanced-determinism` upstream (F10). Number it, bench it, make the budget a P5 entry criterion.
4. **Hygiene** — the io pool size is a recompile-only literal (F13).

Verdict: the architecture will scale by adding threads exactly where the design says it will; the risk is purely that nobody scheduled proving it before the phases that need it.

---

## 5. The drift verdict

**Direction: not drifting. Schedule: drifting at both ends — the scale end and the client end.**

The transfer-first inversion is being executed with discipline, and nearly every end-goal capability has a named owner with an in-code pin (F39): cross-shard signals (P9 + D-4 + the reserved Signal arm), ships (P8 + D-33/D-35), city persistence (P6 + D-39.1), warp (P10 + D-39.3), PvP lag-comp (D-42). The registry has caught and closed its own honesty holes repeatedly. That is the opposite of drift.

The honest exceptions:

- **The scale end of the vision has a resolved direction but no phase number.** "Hundreds in ONE location" and multisharding meshes terminate, in the current roadmap, at P11's "dozens" (roadmap.json P11 risks; PLAN.md:181). D-41's actual region split is deferred to an unnumbered phase. Until F1's fixes land, the roadmap converges on a smaller game than the one specified.
- **Investment balance on the transport arcs:** the R-1..R-6 core was the right work — end-goal-critical, kills old root-cause R9, and is the carrier cross-shard signals will ride (F36). The *tail* showed diminishing returns: 37 R-commits in 8 days recursing to R-6d4-F2 (~24% of all project commits) while P2's own five 🟥 client-track items sat untouched and the "P2 in progress" header aged exactly the way D-9's label did (F12). The work executed was P3/P6-tier durability under a P2 heading. Sanctioned deferral, sloppy ledger — and worth a standing rule: sub-slice depth >3 on one DEFERRED item triggers a re-prioritization check against the phase's remaining open items.
- **Risk stacking:** the three self-declared riskiest unknowns (rapier snapshot determinism, dual-frame ship interiors, signal volume) all sit at P5/P8/P9 with zero de-risk started, and SPIKE-6a's cross-binary half is actually a P4 gate mislabeled as P5 (F11).

---

## 6. Recommended next moves (ordered)

1. **Close P2 honestly (one named client-track slice, now).** D-5 client cut emit + D-4(a) EventMsg/EntityRemoved eviction + D-15 walk-to + the dual-shard devcluster + D-30 paired capture + the new D-43 band-crossing auto-trigger (F5), folding ScriptedClient onto ClientState while touching the FSM anyway (F14). Re-date all five entries to this slice, then flip the P2 section header (F12).
2. **The ledger/design batch — hours of writing, zero code, closes both HIGH design gaps.** D-41 tripwire tests as a standalone pre-P4 slice (F3); the SignalGraph-partition plant item + the P9 charter's three required sections: eval budget/threading, batching tier, ordering contract (F6, F9, F22); the cross-region collision charter note (F21); DEFERRED entries for the stub extraction (F15), the staging-tier reliable shed (F18), CPO-4 (F29), the own-avatar latency budget (F24), and the post-P11 content horizon incl. mechanical subgrids (F33).
3. **P3 robustness tail before P4 opens.** Slice 1b seeded breadth chaos on the real transfer cluster with GATEWAY-victim cells first (F7); D-36 delivery re-poke jointly with the D-37 refinement (F4); D-29 CONFIRMED OpenInputSlot + D-28 conservation gate (F8); D-10 admin exposure as a matrix prerequisite (F17); the persisted Tombstone/C8 cell (F16).
4. **Re-own the density pillar (F1).** D-9+D-39.5 as one slice with one consistent phase (≤P6); the post-P8 region-split phase for D-41's N>1 sim; the cheap N≥128 in-process fixture at the P3→P4 seam; dated evidence rungs 32→64→128→goal.
5. **De-risk before the XL phases.** Split SPIKE-6a: Cat-A cross-binary terrain-numeric gate before P4 with the numeric policy pinned; rapier snapshot half immediately after P3 close (F11). Register the numbered deterministic-parallel physics spike + colliding-capsule bench as a P5 entry criterion (F10). Land the D-41 saga-throughput soak and the outbox-enabled mesh_load variant in the same load track (F20, F19), plus `VD_IO_WORKERS` (F13).
6. **Date the CA-1/L5 decision (F2).** It gates the entire cloud scale-out story AND the AwaitAdopt never-restart detection in prod; when it lands, the over-discard fix and the durable discard-poison must be hard preconditions of the same slice, per DEFERRED.md:1757-1760.

The one-sentence summary for the owner: **the machine you are building can carry the game you described — the transfer core, the seams, and the honesty discipline are genuinely excellent — but the roadmap currently ends at "dozens" and the ledger does not yet own the three pillars that make it "Star Citizen meets Minecraft"; write those owners in now, while it still costs prose instead of rewrites.**

---

# Addendum — Completeness Supplement (3 flagged holes, covered by supplemental audits)

The completeness critic flagged three unexamined areas. Supplemental auditors covered each; all HIGH findings were adversarially re-checked (none refuted; one downgraded). All three areas land at **CONCERNS** — nothing built today breaks, but each contains obligations invisible to the "phase isn't done until DEFERRED flips 🟩" gate.

## A. Security / anti-cheat / abuse — CONCERNS

The cryptographic tier is genuinely strong for its stage (Ed25519 login validated at `gateway.rs:962`, HMAC resume tickets with key-ring rotation + constant-time compare, real mTLS with no skip-verify in `crates/io-prod/src/trust.rs`; server-authoritative input-only clients kill the movement-cheat class). Surviving findings:

- **HIGH — client binary holds the full cluster mTLS secret** (`crates/bins/src/bin/client.rs:81-99` loads CA + node cert + node *private key* via `ClusterTrust::from_der_dir` and joins the inter-shard mesh). Contradicts the designed separate client-facing TLS tier (`connection_plane.md:275`: server-cert TLS 1.3 + login ticket, *no* cluster secret). D-12/CA-1 owns reachability only, not this trust-tier drift. **Fix:** new DEFERRED entry owning the client TLS tier before any non-loopback client; cross-reference D-12.
- **HIGH — no building/claims permission model** for the headline "cities built by many players." `AccessPolicy{OwnerOnly|AllowList|Public}` is signal-channel-scoped only (`sealed_shards.md:140-151`); D-39.1 names the edit *carrier*, no permission gate; zero design/DEFERRED hits. **Fix:** DEFERRED entry — claim model checked server-side at the realm owner on the D-39.1 forward path, owed P6/P8.
- **MEDIUM (downgraded from HIGH on adversarial check) — D-9 whole-realm broadcast is a structural wallhack/ESP.** Verified: `stub.rs:2803-2845` `emits()` has zero observer term; the gateway fans one Arc body to every subscriber. Ledgered purely as bandwidth. **Fix:** re-scope D-9 to an information-security seam ("client never receives a pose it may not perceive") with a negative test; same P6 home.
- **MEDIUM** — login-ticket replay: single-use nonce + `conn_binding` designed (`identity_persistence.md:273`) but `validate_login` checks signature only; unledgered. **Fix:** DEFERRED entry at P3 alongside the directory nonce store.
- **MEDIUM** — block-edit reach/rate (P6) and hit-legality (P11) validation gates not named in D-39.1/D-42; only carriers and rewind are owned. **Fix:** sub-bullets on the existing entries.
- **LOW** — pre-auth gateway Hello/connection flood has no cap or ledger owner (`gateway.rs:945-983` decodes before any admission gate). **Fix:** DEFERRED entry at deploy-readiness tier (D-12/D-13 family).

## B. Client render feasibility under never-LOD — CONCERNS

The Slice-3 renderer is sound and honestly scoped; never-LOD is viable for the *walking* regime (horizon at eye height ≈ 630 blocks ≈ 10 chunks — curvature is a natural cull). Surviving findings:

- **HIGH (confirmed) — never-LOD + binary render rule collide with the P4 demo itself.** At 500-block altitude the horizon is ~10k blocks (~80k chunk columns); from space a hemisphere is O(10^10) faces. Under the rules as written the planet either ends in a visible wall or renders as *nothing* mid-"seamless" transition. The user's own rules are in tension (warp shrink-to-a-dot vs "no proxies ever"). **Fix:** ONE user decision before P4 T0 — (a) nothing beyond range, (b) a sanctioned far-field marker class, or (c) planetary-far-field-only relaxation — recorded in DEFERRED + the P4 charter. Do not resolve unilaterally.
- **MEDIUM** — *no* client-side performance budget exists anywhere (grep-confirmed zero hits; P4 DoD is determinism-only, P11 load gates are all server metrics) while the server-density twin *is* ledgered — an asymmetric hole in a working discipline. **Fix:** DEFERRED entry (frame time / mesh memory / meshing throughput) + a G-RENDER-PERF gate landing with P4.
- **MEDIUM** — hundreds of co-visible animated avatars have no render-cost owner; AoI cannot cull a plaza crowd and never-LOD forbids imposters. **Fix:** fold an instanced/batched-skinning avatar budget (full fidelity, rule-compliant) into the same entry; P11 dense-crowd run must include a real rendering client.
- **MEDIUM** — chunk re-meshing is steady-state client CPU (terrain is client-generated; 20 Hz signal churn dirties chunks continuously; mechanical sub-grids have no seam per `slice_3_renderer.md:281-284`), with no throughput target or threading design. **Fix:** async mesh worker-pool + re-mesh latency ceiling in the P4 charter, gated by G-RENDER-PERF.
- **LOW** — client spawns one entity per delivered entity with no cull (`client-render/src/lib.rs:409-441`); the client half of interest (in-range-mesh vs radar-track partition) is unnamed. **Fix:** one sentence in D-9 pointing it at the P4 dot-marker replacement.

## C. Data lifecycle — disk/fleet tier DR — CONCERNS

Process-tier no-silent-loss is as strong as the main report found, and the retention ledger is genuinely well kept. Surviving findings:

- **HIGH (confirmed) — no owner for backup/restore/point-in-time recovery of any durable store.** Sole "backup" mention in all of docs/ is `identity_persistence.md:352` (orchestrator-only, appendix, unowned). Worse: the fence/epoch/incarnation machinery actively *repels* naive restore (provably-old rejection, epoch-mismatch discard), so restore is an architecture **protocol**, not a later ops bolt-on. **Fix:** one DEFERRED entry (fleet-tier recovery) naming the consistent-export seams (3-step compaction checkpoint, group-commit barrier) and the restore/epoch-rebase protocol; design seam owed at Slice D + P6/P7.
- **HIGH (confirmed) — lost/recreated orchestrator volume ⇒ SILENT genesis.** `d6_saga_wal.md:66` "empty → genesis" cannot distinguish first-boot from lost-volume; the code guards read *faults* only (`store.rs:629-643`). The identical cell was already closed at the incarnation tier (boot-counter genesis floor, `boot.rs:185-186`) — and the check found the convention is already production truth (Slice D-gamma landed), raising urgency. **Fix:** genesis-guard at the register handshake (peer fence/epoch attestation: genesis orchestrator meeting a non-genesis fleet fails loud), owned at Slice D.
- **MEDIUM** — realm-store single-PV loss trade-off is silently assumed (asymmetric with the orchestrator's exemplary Appendix-B documentation); plus `d6_saga_wal.md:112` defers "replicated WAL" into D-32, whose entry doesn't carry it. **Fix:** two paragraphs — trade-off statement in identity_persistence §5.1/P6 DoD; re-point or extend D-32.
- **MEDIUM** — per-item byte budgets exist but city-scale aggregates (10M-edit store size, boot WAL-replay time, compaction-txn stall vs the single redb writer, join-a-city initial sync) are unmeasured and unowned; no storage spike in PLAN.md:165. **Fix:** synthetic city soak in the P6 DoD + initial-sync assertion at P7, per the project's own load-test rule.
- **LOW** — quarantined outbox rows are retained forever by design with a gauge owner (F3) but no inspect/purge tool owner. **Fix:** one line extending the F3 entry when the gauge lands.

**Net effect on the headline verdict:** none of the three areas overturns the main report's answer, but each contributes required ledger entries — roughly 10 new/extended DEFERRED entries, one pre-P4 user decision (far-field representation), and two P6/P7 gate additions — before "extremely robust at AAA quality" holds across the trust, client-render, and disk tiers.

---

## Audit provenance

- Run: wf_306b3197-1a2 (2026-07-05, HEAD 0090c1f), 63 agents, 10 dimensions -> dedup -> adversarial verify (3 lenses per HIGH, 1 per MEDIUM) -> synthesis -> completeness critic -> 3 supplemental audits.
- Dimension verdicts: compose-architecture=SOUND, scale-density-cloud=CONCERNS, multithreading-concurrency=CONCERNS, robust-crash-noloss=CONCERNS, dry-onetooling-elegance=SOUND, feature-signals=CONCERNS, feature-blocks-worlds=CONCERNS, feature-mesh-client=CONCERNS, test-coverage-determinism=CONCERNS, roadmap-drift=CONCERNS
- Findings: 67 raw -> 43 deduped -> 43 survived, 0 refuted. By calibrated severity: {'HIGH': 3, 'LOW': 17, 'MEDIUM': 19, 'NIT': 4}
- Critic holes covered by supplement: security-anticheat-abuse, client-render-feasibility-noLOD, data-lifecycle-disk-tier-dr
- CAVEAT: the single evidence-lens verifiers for F17, F18, F19, F22, F23, F24 (all MEDIUM) failed on a session limit — those six findings are UNVERIFIED pass-throughs; re-check their citations before acting on them.
