# Voxeldust greenfield — local gates (no CI yet; these ARE the gates, run pre-merge).
# HR5 coverage policy: docs/design/coverage_e2e.md. Coverage runs on a pinned NIGHTLY
# (branch coverage + #[coverage(off)] are nightly-only); product builds stay on stable.

# Pinned coverage nightly (installed 2026-06-07; rustc 1.98.0-nightly 61d7280f3).
# Re-pin deliberately via VD_COVERAGE_TOOLCHAIN or `just coverage-setup`.
coverage_toolchain := env_var_or_default("VD_COVERAGE_TOOLCHAIN", "nightly-2026-06-06")

# Tier-A: the 100%-region+branch domain (HR5). Each crate's OWN tests must cover
# its full surface (llvm counts regions per compiled instance, so leaning on the
# vd-tests binary would double-instance every crate — learned in P1.7).
tier_a := "-p vd-core -p vd-physics -p vd-devproto -p vd-wire -p vd-sim -p vd-node -p vd-connection-plane -p vd-harness -p vd-client -p vd-client-harness"

# Inner loop: full deterministic suite (in-process tiers, fast).
test:
    cargo test --workspace

# Inner-loop coverage check: Tier-A only, 100% of the MERGED report, fails the build under 100%.
# The pass/fail decision lives in scripts/coverage_gate.py (owner ruling 2026-08-15, coverage
# option 3 — docs/design/owner_decisions_2026-08-15.md item 8; DEFERRED.md carries the ledger row):
# cargo-llvm-cov's raw --fail-under-* gates count rows PER COMPILED RECORD, so rows owning no
# source-line miss in the merged report (`?` early-return micro-regions, lazy closure bodies,
# per-crate-hash duplicate instantiations) held the gate red on nothing a source line backs. The
# script drops those by the OBJECTIVE no-source-line rule (never a blessed location list), PRINTS
# the dropped count every run (shed-loud), and still fails on every real miss: a merged-lcov DA
# line with zero hits or a BRDA side never taken. The report steps stay scoped to the SAME
# {{tier_a}} package set — without the -p list the denominator spans whatever objects the target
# dir holds from a prior recipe (audit :741). A toolchain bump was tried first (2026-08-15,
# nightly-2026-07-21 + cargo-llvm-cov 0.8.7): byte-identical artifact rows — rejected.
coverage-fast:
    cargo +{{coverage_toolchain}} llvm-cov --branch {{tier_a}} \
        --ignore-filename-regex '(/bin/|/tests/)' \
        --no-report \
        -- --quiet
    cargo +{{coverage_toolchain}} llvm-cov report --branch {{tier_a}} \
        --ignore-filename-regex '(/bin/|/tests/)' \
        --json > target/coverage-fast.json
    cargo +{{coverage_toolchain}} llvm-cov report --branch {{tier_a}} \
        --ignore-filename-regex '(/bin/|/tests/)' \
        --lcov > target/coverage-fast.lcov
    python3 scripts/coverage_gate.py target/coverage-fast.json target/coverage-fast.lcov

# io-prod Tier-B RATCHETED FLOOR (HR5: io-prod is process-tier, never 100% — a SIGKILL can lose the final
# counter flush). This measures io-prod's OWN in-process unit tests (deterministic — no SIGKILL counter loss)
# and fails under a recorded floor, so the crash-durability (RedbStore) + mesh (quinn) code can never regress
# in coverage silently. The floor is conservative (below the measured baseline, absorbing quinn-loopback
# timing variance) and RATCHETS UP — raise it as coverage stabilizes/improves, never lower it.
# RATCHETED 90 → 94 (2026-08-14, audit :195 — the floor had never moved since it was minted): the ledgered
# series measured ≥ 94.4 across six consecutive recorded runs (94.41 / 94.43 / 94.45+94.74 / 94.61+94.90 /
# 94.81+95.11 / 95.06+95.09, DEFERRED.md) with the largest observed run-to-run drop < 0.4 — so 94 sits below
# every recorded value of that series and still absorbs the loopback timing variance. The DEEPER
# process-tier %c merge (the spawned node BINARIES via `orch-crash-cov`) is owed (DEFERRED.md D-40).
tier_b_floor := "94"
coverage-io-prod:
    cargo +{{coverage_toolchain}} llvm-cov -p vd-io-prod --fail-under-regions {{tier_b_floor}}

# R-6d4: the io-prod Tier-B floor WITH `store-test-hooks` — so the crash-proof pins (R-6d4-M's death
# branches + writer_died_panic, B1 ordering, B2 no-premature-gc, B3 no-starve, B4 writer-death) and the
# fault/pause/wait-poll hook paths are INSTRUMENTED + floored, not merely region-tolerated by the no-feature
# pass (the /goal-audit "REAL not theater" note). A SECOND instrumented build so BOTH the release surface
# (coverage-io-prod) and the hook surface are floored — a store-test-hooks-only regression cannot hide.
coverage-io-prod-hooks:
    cargo +{{coverage_toolchain}} llvm-cov -p vd-io-prod --features store-test-hooks --fail-under-regions {{tier_b_floor}}

# Pre-merge: Tier-A 100% + the io-prod Tier-B ratcheted floor (release + store-test-hooks surfaces). The full
# process-tier %c merge (the spawned node binaries via `show-env` + LLVM_PROFILE_FILE %p-%m%c — continuous
# mode is MANDATORY since the harness SIGKILLs processes) is the DEEPER owed piece (DEFERRED.md D-40);
# `orch-crash-cov` already accumulates it.
coverage: coverage-fast coverage-io-prod coverage-io-prod-hooks

# Open the HTML region report to SEE the uncovered region.
coverage-html:
    cargo +{{coverage_toolchain}} llvm-cov --branch {{tier_a}} --html --open

# Lint with the seam-enforcement rules (sim/node clippy.toml disallowed-methods/types).
lint:
    cargo clippy --workspace --all-targets -- -D warnings

# Lint the OTHER supported vd-bins feature combos (the workspace lint covers only the
# default set): `dev-control` alone is the DAILY agent-loop build (client.sh headless),
# `render` alone exercises the requires-dev-control rejection arms, and the COMBINED
# `dev-control,render` combo carries the capture/cut-cycle code — previously compiled
# ONLY by the GPU-required render-smoke, so a violation there slipped every GPU-less
# gate (audit FG-1; clippy needs no GPU, it only type-checks).
lint-combos:
    cargo clippy -p vd-bins --features dev-control --all-targets -- -D warnings
    cargo clippy -p vd-bins --features render --all-targets -- -D warnings
    cargo clippy -p vd-bins --features dev-control,render --all-targets -- -D warnings
    cargo clippy -p vd-bins --features store-test-hooks --all-targets -- -D warnings

# SCALE-1 (the K-client load/collapse gate): K real dev-control clients log into one
# cluster concurrently — fan-out + per-client routing proven at the per-slot client cap.
# `dev_control_nav` rides here because it is `#![cfg(feature = "dev-control")]` — so the
# workspace run compiles it to NOTHING — and until now no recipe named it either, meaning
# the WalkTo/LookAt closed loops were never executed by any gate at all.
client-load:
    cargo test --release -p vd-bins --features dev-control --test client_load
    cargo test --release -p vd-bins --features dev-control --test dev_control_nav

# THE VANISHED CLIENT (2026-09-05): a logged-in client is killed with SIGKILL (no Bye) and the same
# node id logs in again on the same port. It must be Active within the derived bound, with no
# cluster restart. Before this gate a dead client's session stayed open forever and the re-login
# waited behind it ("awaiting_welcome" until an operator restarted the cluster).
login-after-kill:
    cargo test --release -p vd-bins --features dev-control --test login_after_kill

# D-6 D-delta: the orchestrator durability crash gates. `orchestrator_crash` is the SIGKILL-mid-fsync proof
# (a real kill-9 while a directory grant sits submitted-but-pre-fsync loses <=1 batch + recovers
# consistently); it builds the orchestrator binary WITH the writer-pause hook (`store-test-hooks`).
# `boot_guard` (always compiled) locks the HR1 ephemeral-store boot guard's reject + accept arms.
# `boot_counter_crashloop` is the R-6b M3 CAPSTONE: a real SIGKILL + sub-second shard restart, proving the
# durable boot-counter (higher incarnation) keeps the orchestrator from silently DEDUP-dropping the restart's
# reliable re-grant (RED control: a fixed incarnation DOES get deduped). Serial + single-threaded (each
# spawns a real cluster + SIGKILLs a process — concurrent cluster tests would contend for CPU/ports).
#
# NOTE: the `--test-threads=1` flags below are now BELT-AND-BRACES, not the mechanism. Serialization is
# enforced AT THE SOURCE by `vd_bins::cluster_tier()`, which every cluster-booting test holds for its whole
# body (and `every_cluster_booting_test_holds_the_tier` proves none was missed). That is what makes the
# plain `cargo test --workspace` honest too — a recipe flag could only ever fix the recipe, leaving the
# documented workspace command still producing 1-3 shuffling false failures per run. The flags stay because
# they cost nothing and keep these recipes correct even if invoked against an older tree.
orch-crash:
    cargo test -p vd-bins --features store-test-hooks --test orchestrator_crash -- --test-threads=1
    cargo test -p vd-bins --test boot_guard -- --test-threads=1
    cargo test -p vd-bins --test boot_counter_crashloop -- --test-threads=1
    cargo test -p vd-bins --features store-test-hooks --test outbox_sigkill_restart -- --test-threads=1

# R-4e (L7): the N-peer real-QUIC LOAD/soak gate. Sustained reliable fan-in — VD_MESH_LOAD_NODES sender
# endpoints into ONE receiver — proving no-loss/no-dup, reliable_acked keeps pace, gap_drop==0, and
# inbound_dropped_reliable==0 (a SIZED receiver inbox makes that structural), i.e. NO RX-plane collapse
# under N-peer fan-in. SLOW + resource-heavy (N real quinn endpoints on one host) ⇒ NOT in the default
# `gate`; a STANDALONE BLOCKING soak/deploy precondition, run before any real deploy. The pinned soak N
# (64) was validated green 5/5 on the dev host (12,800 reliable frames in ~0.1s; N=128 also green in
# ~0.8s — the fan-in is correctness-bound, not wall-clock-bound on loopback). Re-validate against the
# target host's fd/port ceiling before a cloud soak. `mesh_under_loss` is the R-5 capstone (lands R-4e4).
mesh-load:
    VD_MESH_LOAD_NODES=64 cargo test -p vd-io-prod --test mesh_load -- --nocapture --test-threads=1
    cargo test -p vd-io-prod --test mesh_under_loss -- --nocapture

# The Tier-B (ratcheted-floor) coverage variant: EVERY child process spawned under these recipes exits ONLY
# by SIGKILL, so its counters survive ONLY in %c CONTINUOUS mode (the mmapped profraw is updated in place; no
# atexit flush after a SIGKILL). This covers all three tests' children: the orchestrator_crash boots (+ shard),
# and the outbox_sigkill_restart node (boot-1 SIGKILLed by `kill_and_reap` while parked at the seed marker;
# boot-2 park-forever, SIGKILLed only by `Cluster::drop` at teardown). %p-%m keep the distinct boots separate;
# each child INHERITS LLVM_PROFILE_FILE from this test process (spawn_node forwards the parent env). D-40 owes
# the enforced `--fail-under` floor over the merged %p-%m%c profraws (this recipe only ACCUMULATES today).
orch-crash-cov:
    #!/usr/bin/env bash
    set -euo pipefail
    source <(cargo +{{coverage_toolchain}} llvm-cov show-env --export-prefix --branch)
    export LLVM_PROFILE_FILE="$(dirname "$LLVM_PROFILE_FILE")/vd-orchcrash-%p-%m%c.profraw"
    cargo +{{coverage_toolchain}} llvm-cov --no-report --branch \
        -p vd-bins --features store-test-hooks --test orchestrator_crash -- --test-threads=1
    cargo +{{coverage_toolchain}} llvm-cov --no-report --branch \
        -p vd-bins --test boot_guard -- --test-threads=1
    cargo +{{coverage_toolchain}} llvm-cov --no-report --branch \
        -p vd-bins --features store-test-hooks --test outbox_sigkill_restart -- --test-threads=1

# SPIKE-2a (the route-swap hot-path gate): the gateway 20Hz route decision stays wait-free
# + torn-read-free under a concurrent route.store publisher, p99 < 50us. RELEASE build (a
# debug/coverage build's instrumentation makes a 50us tail meaningless; the debug run still
# exercises the concurrency + torn-read invariant via `just test`, just not the timing).
# Hand-rolled (no bench crate expresses a concurrent hard-fail p99 gate — investigated).
# Formally BLOCKS the P2 route-swap design.
spike2a:
    cargo test --release -p vd-connection-plane spike_2a -- --nocapture --test-threads=1

# SPIKE-3a (the 2nd hard latency gate): the 20Hz unreliable SNAPSHOT datagram hot path stays
# TIMELY (delivered p99 under one tick) while a bulk RELIABLE burst saturates the SAME quinn
# connection — the "hundreds in one location + a ship-blueprint/terrain-chunk transfer must not
# starve snapshots" property. RELEASE build (a debug/coverage tail is meaningless; `just test`
# still exercises the send/drain/decode + ratio + honesty asserts, just not the timing).
# Shares the ONE `vd_harness::latency::percentile_unstable` with SPIKE-2a (HR3, no drift).
spike3a:
    cargo test --release -p vd-io-prod --test mesh_snapshot_latency -- --nocapture --test-threads=1

# THE CHAIN-LATENCY GATE (the 3rd hard latency gate): what the "only the parent knows where its children
# are" rule COSTS. Every drawn value now crosses one shard per level of the live chain instead of leaving
# the shard that authored it and going straight out, and until this recipe nothing measured that. The one
# latency gate that ever sat over the render path measured the ROUTER re-expressing every body into every
# viewer's space; that work no longer happens, so the tree carried a gate over something gone and none over
# what replaced it.
#
# NOT a release recipe, deliberately, and that is the point: the cost is paid in TICKS (a message sent in
# tick N lands no earlier than N+1), so the figure is the same integer on any machine and a debug build
# measures it exactly. `--nocapture` because the numbers ARE the deliverable — the gate asserts a
# topology-derived budget of one tick per hop, and PRINTS what the chain actually costs, up-leg, down-leg,
# and under a lossy link at three depths.
chain-latency:
    cargo test -p vd-tests --test frame_conversion_e2e the_chain_ -- --nocapture --test-threads=1

# RLM Step 4b (the crash-replay SOAK): a ~30k-op crash/reorder/dup stream through the REAL realm
# reconcile kernel, asserting determinism + crash-no-reap + no-strand + bounded-ledger +
# INV-SNAPSHOT-SAFETY throughout. RELEASE build (the `#[cfg(not(debug_assertions))]` soak is inert in
# debug; `just test` still runs the 1024-case fixed-seed proptest + the 6 named witnesses). A sustained
# ARM-A≠ARM-B divergence here is the concrete trigger to promote the deferred durable snapshot (D-RLM-2).
# THE WINDOW LANE rides the same soak recipe on the composer (window_lane.md §4.5 Topic 5):
# ~30k ticks of session/window churn + lossy/reordered/forged ingest through the REAL composer pass,
# asserting memory/counters MONOTONE-BOUNDED every tick (rings ≤ the derived span, realm-heads ≤ the
# named set) and ZERO composer state after the last session leaves — no leak, asserted, never
# eyeballed.
rlm-soak:
    cargo test --release -p vd-sim rlm_soak -- --nocapture --test-threads=1
    cargo test --release -p vd-connection-plane window_shadow_soak -- --nocapture --test-threads=1

# G-COMPOSE-LOAD (window_lane.md §2.6.7/§4.5 Topic 3 — the owner-adopted AAA refinement: gate the
# p99, never the mean): BOTH sides of the Slice-B engine under a DERIVED load — the ingest-DECODE
# side (postcard `WindowFrame` levels at MTU-full row counts × the live window fan) and the
# compose+fan side (per-tick folds at max_sessions across chain depth 6) — p99 per-tick cost under
# ONE realm-lane tick (50 ms at 20 Hz). RELEASE build (the SPIKE-3a latency-gate pattern: a
# debug/coverage tail is meaningless; `just test` still runs the same bodies functionally).
window-compose-load:
    cargo test --release -p vd-connection-plane g_compose_load -- --nocapture --test-threads=1

# THE WINDOW LANE's COMPOSED SELF-CONSISTENCY GATE (window_lane.md §2.12): the composition engine
# live in real processes — a real demand cluster, TWO real logins, a real crossing leg, a dwell.
# It was the SHADOW-PARITY gate through Slice B; that measurement ran, is recorded in D-WINDOW-1,
# and discharged its purpose at the Slice-C1 client cut, so Slice C2 re-based the scenario onto the
# three §2.12 pins that had no other process-tier home: the exact-cadence proof
# (window_full_chain_folds > 0 — two shard processes stamping IDENTICAL universe ticks), both dedup
# f64 agreements (hop-vs-child-row and shared-vs-per-session, each measured zero), and the shear law
# live (instant_mismatch == 0). It also asserts the four deleted scenery lanes stay SILENT in a real
# cluster. Same port band + serialization posture as rlm-demand-login.
window-parity:
    cargo test --release -p vd-bins --features dev-control --test window_shadow_parity -- --test-threads=1 --nocapture

fmt:
    cargo fmt --all

# The GATE's fmt step: fail-on-drift (a gate must never silently rewrite the tree it
# validates — audit AAA-1). `just fmt` stays the dev fixup.
fmt-check:
    cargo fmt --all --check

# G-RENDER-SMOKE (HR6 permanent visual gate): bring up the local cluster, launch a HEADLESS
# `client --capture`, capture a real wgpu-readback frame, and assert no-magenta +
# content-present over it. Builds the client with `--features dev-control,render` (pulls
# Bevy). REQUIRES A WORKING GPU ADAPTER (the dev Metal GPU) — it is a LOCAL gate (no CI yet,
# no software fallback). Steer the adapter with WGPU_BACKENDS / WGPU_POWER_PREF if needed.
render-smoke:
    cargo test --release -p vd-bins --features dev-control,render --test render_smoke -- --nocapture

# G-RENDER-BOXES-SMOKE (THE world's realm-box pixel proof, re-based Slice C1 §2.11): bring up the
# cluster, launch a HEADLESS `client --capture` drawing ONLY the COMPOSED STREAM (no --realm-boxes
# file exists), wait on the demand loop's own settle signal, capture a real wgpu-readback frame,
# and assert the drawn set EQUALS the gateway-emitted level set (anti-vacuity vs the run manifest),
# the origin marker == the home realm, and the HOME SHELL ITSELF drew where it should — paint at a
# rim probe just inside its silhouette and NONE just outside it (H2, isolated to the shell) + zero
# magenta. Same GPU-required, LOCAL-gate preconditions as render-smoke (no CI, no software
# fallback; steer with WGPU_BACKENDS).
render-boxes-smoke:
    cargo test --release -p vd-bins --features dev-control,render --test render_boxes_smoke -- --nocapture

# G-RENDER-CROSSING-SMOKE (the crossing pixel proof on THE world, re-based Slice C1 §2.11): bring
# up the DUAL cluster, launch a HEADLESS `client --capture` drawing ONLY the COMPOSED STREAM, fly
# the ±Z polar corridor OUT of the home system's own shell and BACK, and capture THREE frames —
# each DOT-SENSITIVE, the dot rect DevState-driven (projected position + body kind, local pixel
# probes — never a full-frame search): INSIDE (dot in the home shell's rect), OUTSIDE (origin =
# the galaxy realm; the home body still draws at the session origin — the galaxy authors its
# placement at ZERO), RETURNED (inside again). J1 is the ORIGIN-MARKER assert: home draws at the
# origin on BOTH sides and the epoch bumps EXACTLY once per crossing; plus the crossing NO-FLICKER
# gate — no capture across the swap where a persisting body's screen delta exceeds the one-tick
# motion bound, and no frame with the scene absent. The camera is RECONSTRUCTED per capture from
# the client's own reported drawn boxes + STREAMED extents. Zero magenta on all three. Same
# GPU-required, LOCAL-gate preconditions as render-smoke.
render-crossing-smoke:
    cargo test --release -p vd-bins --features dev-control,render --test render_crossing_smoke -- --nocapture

# G-WARP-PIXELS + G-HANDOVER (window_lane.md §2.8/§4 Slice D — THE WARP ACCEPTANCE): one DEMAND
# cluster (no shard pre-booked), one headless capture client in the PILOT VIEW (`--capture-pilot`),
# and one flight down THE world's own star ring — out of the home system, across the STAR GAP the
# galaxy's own solver produced, into the ring sibling. Asserts, in real pixels: the destination is
# its PARENT'S point of light at departure (it is asleep because the ring is wider than the
# destination's visibility WAKE RADIUS, by a margin the world's own solver produced — the companion
# test computes it, prints it and asserts it positive); its footprint never shrinks while you
# approach; its own look takes over flicker-free; the system behind hands back to its parent's
# marker and shrinks to a dot; and every drawn row's provenance is in the run manifest (HR6).
# G-HANDOVER measures BOTH
# directions against budgets DERIVED in ticks from the landed cadences and the cluster's own boot
# latency, with the Q2 RELAY HOP asserted APART (the owner's ruling: if the relay breaks the wake
# budget, that is the one measurement D-WINDOW-2 is gated on). The companion test closes the
# derivation before any process runs, so a world-numbers change fails in milliseconds.
# Same GPU-required, LOCAL-gate preconditions as render-smoke.
warp-pixels:
    cargo test --release -p vd-bins --features dev-control,render --test warp_pixels -- --nocapture --test-threads=1

# G-TWO-SHIPS (owner-ordered 2026-08-16, window_lane.md §5 RULINGS): TWO real capture clients on one
# demand cluster, standing in two realms of DIFFERENT DEPTH (the star System and one of its
# Planets), with mutual hull visibility across the realm boundary and one crossing while both watch.
# The five assertions: (a) each observer draws the OTHER hull at its same-tick composed position —
# the two chains must agree on the star-to-planet separation to within the planet's own travel over
# the sampling gap; (b) the inner observer draws the Planet's own body around itself, in pixels (the
# hop row); (c) each hull's look is the REALM'S OWN statement delivered by the Q2 parent relay,
# attested in the manifest; (d) one crossing while both watch — the crossing client's epoch bumps
# exactly once, the WATCHING client's epoch does not move, and its picture is sampled continuously
# through the commit with every body inside its own motion allowance; (e) occupant figures through
# WINDOWS are absent, and nobody standing inside a realm is shown to an observer outside it.
two-ships:
    cargo test --release -p vd-bins --features dev-control,render --test two_ships -- --nocapture --test-threads=1

# THE WORLD IS SEEN FROM INSIDE ANY REALM (owner ruling 2026-09-02 R9 step 1): one DEMAND cluster per
# subject, one real headless GPU client, the SAME body on a player-built hull (written by the shipyard's
# stand-in, entered by a forty-metre walk) and on the home system's innermost planet (entered by the
# governed rendezvous). From inside each: the stars are DRAWN (`stars_drawn`, the new instrument — not
# the held count), the picture names a realm the subject does not parent, and the stars do not vanish
# across the crossing. RED on 2026-09-02 by design: it is the measurement the reach slices turn green.
world-from-inside:
    cargo test --release -p vd-bins --features dev-control,render --test world_from_inside -- --nocapture --test-threads=1

# THE TRUE-SCALE LOOK GATE (celestial_taxonomy_design §9 T2/T3, carrying look_horizon.md §6 slice
# 5's G-NOTHING-OWED + G-IDENTICAL forward): one DEMAND cluster booted onto THE world PLUS the
# planted player-built station/area pair (VD_FIXTURE_PLANT=station-area — the SL5 fixture-forest
# doctrine's process path; ONE derivation in vd_physics::worldgen::station_area_plant, every
# process boots it through
# vd_bins::process_world_config), one headless GPU client. THE STAR wakes by demand at the derived
# login standoff and draws AS A BODY carrying its own photometric datum (class code + luminosity,
# asserted against the generator's own draw, never a transcribed code); flying OUT past its
# tear-down the body hands over to the parent's MARKER exactly once, the vacated Star realm reaps
# behind, and the marker carries THE SAME datum; flying BACK IN it hands back inside a derived wake
# budget that states THE EXTRA RELAY HOP as its own term. Then OUT to a home planet that HOSTS A
# MOON (chosen by derivation, not by which one orbits furthest): crossing INTO its realm, its own
# body draws at the camera-model size far above the three-pixel floor. Then ON to that planet's
# MOON — a Planet under a Planet, demand-spawned at DEPTH 4 and intercepted on its own orbit — and
# back out, where whether the vacated moon reaps is DERIVED from its own wake and then asserted
# either way. THE PARENT LAW rides every sample (every planet row parents on the star system, the
# planted station on the star system, the planted area on its own planet — G-IDENTICAL, HR4, on a
# player-built KIND in the same scene). G-NOTHING-OWED (THE LAW GATE) enumerates the generated
# forest OUT-OF-BAND at the spawn and asserts every subject above the minimum angle draws its OWN
# picture — non-vacuous (depth-2 subjects in the oracle set). A companion test closes every park
# and budget derivation before any process runs.
# Same GPU-required, LOCAL-gate preconditions as render-smoke.
look-pixels:
    cargo test --release -p vd-bins --features dev-control,render --test look_pixels -- --nocapture --test-threads=1

# NODE-PER-REALM WALK GATE (task #149) — the HEADLESS process-tier chain proof on THE world. Brings up
# the CHAIN cluster (orchestrator + gateway + FOUR realm-shards derived through world_roster: the home
# system, the galaxy, the inner planet, the sibling star — NO co-hosting), logs in a REAL headless
# durable player over localhost QUIC, and flies the ±Z polar-corridor legs C,D,A,E,F (rendezvous into
# the inner planet, polar lift home, out to the galaxy, the governed boost across the star gap to
# the sibling, and back). EVERY leg asserts the realm LABEL reached — never a coordinate — and the
# subject Entity's directory fence stays SMALL, with the observed maximum PRINTED per run (the
# thrash guard; the bound is
# unmeasured for this chain until a green history accumulates). In-test deadline 300 s (real flight
# distances; was 120 s for the retired inert walk). No GPU: this is the CI walk gate.
node-per-realm-walk:
    cargo test --release -p vd-bins --features dev-control --test node_per_realm_walk -- --test-threads=1 --nocapture

# RLM 5c-2b: the real-process ProcLaunchBackend gate — forks a real vd-shard (Planet 7 + Galaxy), asserts
# it boots + echoes its incarnation cookie on /whoami + teardown reaps it (pid gone, no zombie). Tier-B
# (vd-bins), so this process proof stands in for coverage on the launch/liveness/teardown syscalls.
rlm-proc-spawn:
    cargo test -p vd-bins --test rlm_proc_spawn_smoke -- --nocapture

# RLM 5e-5: the kill-9 crash-safety CAPSTONE — SIGKILLs the real orchestrator across the launch.redb
# crash windows (mid-fsync pre-fork = no orphan/no double-spawn; adopt = a survivor recovered without
# relaunch; the water_only control = a genuine double-spawn proving rehydrate/adopt is load-bearing). Tier-B,
# feature-GATED: `--features store-test-hooks` is MANDATORY (without it the `#![cfg]` file compiles out +
# silently skips). `--test-threads=1` + disjoint per-arm F2 port bands keep the pre-fork probe witness
# deterministic (like orch-crash, it forks real processes + SIGKILLs — no concurrent cluster tests).
rlm-kill9:
    cargo test -p vd-bins --features store-test-hooks --test rlm_kill9_spawn -- --test-threads=1 --nocapture

# RLM: THE demand-LOOP PROCESS proofs — boots a DEMAND cluster (orchestrator + gateway ONLY, no shard
# pre-booked) in real processes. (1) demand-LOGIN: one real dev-control client logs in, proving from the
# gateway admin snapshot that the login SPAWNED + reached a fresh home shard with no pre-booked address
# (dynamic_shards >= 1, presence_announces >= 1, home_bootstrap_timeouts == 0). (2) gateway-restart re-heal:
# the reactive greeting re-learns a running demand shard after the gateway restarts. (3) demand-WALK: a MOVING
# client walks toward a child realm — its AoI spins the child up AHEAD of it, it CROSSES in (location→"Planet
# 7"), and when it walks back out the vacated realm is torn down + reaped (teardowns_reaped >= 1). Tier-B,
# `--features dev-control`. `--test-threads=1` + a dedicated RLM port band (45000+) keep the demand-spawned
# shards' deterministic ports off rlm_kill9's band.
rlm-demand-login:
    cargo test --release -p vd-bins --features dev-control --test rlm_demand_login -- --test-threads=1 --nocapture

# (Stage-C: the orbit-slowdown knob is DELETED — SL5. The whole demand suite now flies at full
# orbit speed through the ONE shared rendezvous, so the separate flight-speed recipe is retired:
# `rlm-demand-login` above IS the flight-speed gate for every crossing test.)

# ★ G-RENDER-SCALE (the S5 slice's own measurement gate, D-LOOK-3): the render PRECISION BUDGET,
# measured on THE world with no cluster and no GPU. (1) THE DEPTH BUDGET: under Bevy's reverse-Z
# INFINITE perspective one f32 depth-code ulp is a CONSTANT fraction of the range at every distance,
# and that fraction is compared against the thinnest thing the world ever draws (2/visibility factor
# of its own range) plus the two concrete range pairs the parked pixel gates measured. This is what
# says reverse-Z alone spans 1e6 m to 2e15 m and a logarithmic depth buys nothing. (2) THE POSITION
# BUDGET: the f32 look-at direction is still MEASURED lost at every eye magnitude THE world stands
# at (the defect stays real, so the gate is never vacuous), beside the render frame's own residual
# in pixels. Seconds to run; it would have caught the parked gates' failure without a flight.
render-scale:
    cargo test --release -p vd-bins --features dev-control,render --test render_scale -- --nocapture

# ★ G-ACCEPTANCE-FLIGHT (the S7 slice): THE FLIGHT THAT PROVES THE WHOLE ARC. One demand cluster,
# one headless GPU client, five legs on THE world — stand off the home system's innermost world at
# the derived range where its own disc FILLS THE FRAME (asserted against the camera model, in
# pixels); leave that world's space (its picture shrinks monotonically and never blanks); cross the
# system out the licensed polar corridor (the star passes; the system's worlds draw at DISTINCT
# sizes); warp the star gap to a 3-D neighbour (★ REAL PARALLAX: the sibling stars' bearings sweep
# measurably; the destination grows from a point of light, hands over EXACTLY once, and is already
# awake before the crossing); and come home (the reverse handovers, home growing back). Every park,
# budget and expectation is DERIVED from the world the cluster boots — change the seed and the gate
# re-derives. Measured leg times are printed against the flight table's closed form.
# Same GPU-required, LOCAL-gate preconditions as render-smoke.
acceptance-flight:
    cargo test --release -p vd-bins --features dev-control,render --test acceptance_flight -- --nocapture --test-threads=1

# Everything a merge requires (render-smoke/render-boxes-smoke are GPU-required + local; spike2a is
# a release build — all documented in their recipes). fmt-check FAILS on drift (run `just fmt` to
# fix); every gate step is fail-on-violation, none mutates the tree.
gate: fmt-check lint lint-combos test client-load orch-crash spike2a spike3a window-compose-load chain-latency rlm-soak render-smoke render-boxes-smoke render-crossing-smoke warp-pixels look-pixels two-ships world-from-inside node-per-realm-walk rlm-proc-spawn rlm-kill9 rlm-demand-login window-parity render-scale acceptance-flight coverage

# One-time setup helper.
coverage-setup:
    rustup toolchain install nightly --component llvm-tools-preview
    cargo +nightly install cargo-llvm-cov --locked
    @echo "Pin the installed nightly via VD_COVERAGE_TOOLCHAIN (e.g. nightly-2026-06-01) in your shell or .envrc"

# ---- Cloud-ready k3d (before P4) -------------------------------------------------------------
# Slice 0: build the ONE multi-bin SERVER image (orchestrator+gateway+shard; HR3 — role is a
# command+env, not a per-kind image). Bevy-free (default features). The build happens inside a
# Linux glibc toolchain (the host is macOS); .dockerignore keeps target/ (~70 GB) out of context.
server_image := env_var_or_default("VD_SERVER_IMAGE", "voxeldust-server:dev")
image-build:
    docker build -f docker/server.Dockerfile -t {{server_image}} .

# Slice 0 smoke: run the orchestrator from the freshly-built image and assert it boots + serves
# /metrics (the read-only ops endpoint). Mints a throwaway mTLS bundle on the host (portable DER,
# mounted read-only) via the same `gen-trust` that feeds the Slice-4 Secret. Ephemeral store
# in-container (VD_STORE_EPHEMERAL_OK=1 — a DEV escape, never a cloud manifest); a real PVC + durable
# root land in Slice 4. VD_CLOCK_PEERS empty ⇒ a solo orchestrator (genesis clock, no quorum wait).
image-smoke: image-build
    #!/usr/bin/env bash
    set -euo pipefail
    name=vd-slice0-smoke
    trust=$(mktemp -d)
    trap 'docker rm -f "$name" >/dev/null 2>&1 || true; rm -rf "$trust"' EXIT
    cargo run -q --bin vd-devcluster -- gen-trust "$trust"
    docker rm -f "$name" >/dev/null 2>&1 || true
    docker run -d --name "$name" \
        -e VD_NODE_ID=1 -e VD_BIND=0.0.0.0:9000 -e VD_ADMIN_ADDR=0.0.0.0:9100 \
        -e VD_TRUST_DIR=/trust -e VD_STORE_PATH=/tmp/vd.redb -e VD_STORE_EPHEMERAL_OK=1 \
        -e VD_EPOCH=1 -e VD_RESERVE_CHUNK=4096 -e VD_LEASE_TTL=10000 \
        -e VD_OUTBOUND_CAP=256 -e VD_TICK_HZ=50 -e VD_PROCESS_INCARNATION=1 \
        -e VD_PEERS= -e VD_CLOCK_PEERS= \
        -v "$trust":/trust:ro -p 9100:9100 {{server_image}} vd-orchestrator
    echo "waiting for /metrics ..."
    for i in $(seq 1 30); do
        if curl -sf http://127.0.0.1:9100/metrics >/dev/null 2>&1; then
            echo "OK: /metrics served by the containerized orchestrator"
            curl -s http://127.0.0.1:9100/metrics | head -5
            exit 0
        fi
        sleep 1
    done
    echo "FAIL: /metrics never came up"; docker logs "$name" | tail -40; exit 1

# Slice 1 drain proof (SIGTERM graceful-drain): `docker stop` sends SIGTERM (the EXACT signal a k8s
# pod-stop / rolling-deploy sends) then escalates to SIGKILL after the grace. A clean drain must exit 0
# WELL within grace; a 137 means the loop never broke and the SIGKILL escalation fired (drain broken).
# The most cloud-representative SIGTERM test — the real container, the real signal.
image-drain-smoke: image-build
    #!/usr/bin/env bash
    set -euo pipefail
    name=vd-slice1-drain
    trust=$(mktemp -d)
    trap 'docker rm -f "$name" >/dev/null 2>&1 || true; rm -rf "$trust"' EXIT
    cargo run -q --bin vd-devcluster -- gen-trust "$trust"
    docker rm -f "$name" >/dev/null 2>&1 || true
    docker run -d --name "$name" \
        -e VD_NODE_ID=1 -e VD_BIND=0.0.0.0:9000 -e VD_ADMIN_ADDR=0.0.0.0:9100 \
        -e VD_TRUST_DIR=/trust -e VD_STORE_PATH=/tmp/vd.redb -e VD_STORE_EPHEMERAL_OK=1 \
        -e VD_EPOCH=1 -e VD_RESERVE_CHUNK=4096 -e VD_LEASE_TTL=10000 \
        -e VD_OUTBOUND_CAP=256 -e VD_TICK_HZ=50 -e VD_PROCESS_INCARNATION=1 \
        -e VD_PEERS= -e VD_CLOCK_PEERS= \
        -v "$trust":/trust:ro -p 9101:9100 {{server_image}} vd-orchestrator
    echo "waiting for /metrics (readiness before the stop) ..."
    ready=0
    for i in $(seq 1 30); do
        if curl -sf http://127.0.0.1:9101/metrics >/dev/null 2>&1; then ready=1; break; fi
        sleep 1
    done
    [ "$ready" = "1" ] || { echo "FAIL: never became ready"; docker logs "$name" | tail -30; exit 1; }
    echo "sending SIGTERM via 'docker stop -t 10' ..."
    docker stop -t 10 "$name" >/dev/null
    code=$(docker wait "$name" 2>/dev/null || docker inspect -f '{{{{.State.ExitCode}}}}' "$name")
    echo "container exit code: $code"
    if [ "$code" = "0" ]; then
        # exit 0 is necessary but NOT sufficient (a process can exit 0 for other reasons). The drain-log line
        # is the LOAD-BEARING proof the graceful-drain path actually RAN — assert it as a conjunction, never
        # `|| true` (which would make it cosmetic). Both must hold: clean exit AND the drain ran.
        docker logs "$name" 2>&1 | grep -qi "drained on shutdown" \
            || { echo "FAIL: exited 0 but the drain log is ABSENT — the graceful-drain path did not run"; docker logs "$name" | tail -20; exit 1; }
        echo "OK: clean SIGTERM drain — exit 0 AND the drain path ran (loop broke + store Drop joined the writer)"
        exit 0
    fi
    echo "FAIL: expected exit 0, got $code (137 = SIGKILL escalation ⇒ the drain did not exit within grace)"
    docker logs "$name" | tail -30; exit 1

# Slice 4b: k3d deployment. The manifests live in deploy/k3d/*.yaml (numeric-prefix apply order); the two
# Secrets (mTLS bundle + the production auth key) are minted IMPERATIVELY so no key.der / signing-seed ever
# lands in git. All kubectl is context-pinned to the k3d cluster + namespace. `k3d-validate` is the D2
# author+static-validate path (NO live cluster); `k3d-all` is the full LIVE bring-up (deferred behind an
# explicit run per the "confirm it works" rule).
k3d_cluster := env_var_or_default("VD_K3D_CLUSTER", "voxeldust")
# THE DOOR (2026-09-06): the host port a player's client dials to reach the cluster's gateway (mapped onto
# the gateway's NodePort at cluster create), the client's own UDP port and dev-control port on the host,
# and the folder the cluster's trust is exported to for that client. Away from the dev cluster's slots.
k3d_door_port := env_var_or_default("VD_K3D_DOOR_PORT", "19000")
k3d_client_quic := env_var_or_default("VD_K3D_CLIENT_QUIC", "19001")
k3d_devctl_port := env_var_or_default("VD_K3D_DEVCTL_PORT", "17777")
k3d_trust_dir := env_var_or_default("VD_K3D_TRUST_DIR", env_var_or_default("TMPDIR", "/tmp") + "/vd-k3d-trust")
kctx        := "k3d-" + k3d_cluster
kns         := "voxeldust"
k           := "kubectl --context " + kctx + " -n " + kns

# Static-validate the manifests WITHOUT a cluster (D2 path): kubeconform if present, else a client dry-run.
k3d-validate:
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v kubeconform >/dev/null 2>&1; then
        kubeconform -strict -summary -kubernetes-version 1.31.0 deploy/k3d/*.yaml deploy/k3d/agent/*.yaml
    else
        echo "kubeconform not found — falling back to 'kubectl apply --dry-run=client' (offline, schema-lite)"
        kubectl apply --dry-run=client -f deploy/k3d/ -f deploy/k3d/agent/ >/dev/null && echo "client dry-run OK"
    fi
    sh -n docker/entrypoint.sh && echo "entrypoint.sh: sh -n OK"
    # ★ NO ROLLING UPDATES (owner ruling 2026-08-24, slice S1). A durable file now carries a label naming
    # the world, the coordinate units and the epoch that wrote it, and a process whose label disagrees
    # REFUSES TO START. A rolling update therefore does not degrade a release — it partitions the cluster
    # into the pods that restarted and the ones that have not, each refusing the other's files. Every
    # workload that keeps durable state must replace stop-first, so this fails the deploy rather than
    # letting the manifests drift back to the default.
    missing=""
    for f in deploy/k3d/30-orch.yaml deploy/k3d/40-gateway.yaml; do
        grep -Eq '^[[:space:]]*updateStrategy:[[:space:]]*\{[[:space:]]*type:[[:space:]]*OnDelete[[:space:]]*\}' "$f" || missing="$missing $f"
    done
    if [ -n "$missing" ]; then
        echo "REFUSING: these workloads would roll one pod at a time, which partitions a label-checking cluster:$missing" >&2
        exit 1
    fi
    echo "update strategy: stop-and-replace on every durable workload OK"

k3d-up:
    #!/usr/bin/env bash
    set -euo pipefail
    k3d cluster list {{k3d_cluster}} >/dev/null 2>&1 && k3d cluster delete {{k3d_cluster}} || true
    # The door: host {{k3d_door_port}}/udp → the server node's 30900 (the gateway's NodePort, 40-gateway.yaml).
    k3d cluster create {{k3d_cluster}} --wait --timeout 120s -p "{{k3d_door_port}}:30900/udp@server:0"
    kubectl --context {{kctx}} apply -f deploy/k3d/00-namespace.yaml

# Build the server image (reuses image-build) + import it into the k3d node (never a registry pull).
k3d-load: image-build
    k3d image import {{server_image}} -c {{k3d_cluster}}

# Mint the three Secrets from ONE gen-authkey run so the gateway's pubkey and the agent's signing seed are a
# MATCHED pair (a mismatch = every login rejected). Nothing hits git: the trust bundle is mktemp-scoped and the
# signing seed lives only in a shell var + the k8s Secret. vd-auth-signing is the CLIENT's secret (mounted only
# on the agent Job), honoring gen-authkey's "signing key out-of-band, never a SERVER secret" contract.
k3d-secrets:
    #!/usr/bin/env bash
    set -euo pipefail
    # IDEMPOTENT: if the full triad already exists, REUSE it — never rotate a live cluster's auth pair out from
    # under a running gateway (which loaded VD_AUTH_PUBKEY at boot and won't reload the Secret), which would
    # reject every login. A fresh cluster has none, so it mints below.
    if {{k}} get secret vd-mtls vd-auth vd-auth-signing >/dev/null 2>&1; then
        echo "k3d-secrets: vd-mtls + vd-auth + vd-auth-signing already present — reusing (no rotation)."
        exit 0
    fi
    T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
    cargo run -q -p vd-bins --bin vd-devcluster -- gen-trust "$T"
    {{k}} create secret generic vd-mtls \
        --from-file=ca.der="$T/ca.der" --from-file=node.der="$T/node.der" --from-file=key.der="$T/key.der" \
        --dry-run=client -o yaml | {{k}} apply -f -
    authout="$(cargo run -q -p vd-bins --bin vd-devcluster -- gen-authkey)"
    pub="$(printf  '%s\n' "$authout" | sed -n 's/^VD_AUTH_PUBKEY=//p')"
    sign="$(printf '%s\n' "$authout" | sed -n 's/^VD_AUTH_SIGNING_KEY=//p')"
    : "${pub:?gen-authkey produced no VD_AUTH_PUBKEY}"; : "${sign:?gen-authkey produced no VD_AUTH_SIGNING_KEY}"
    {{k}} create secret generic vd-auth --from-literal=AUTH_PUBKEY="$pub" \
        --dry-run=client -o yaml | {{k}} apply -f -
    {{k}} create secret generic vd-auth-signing --from-literal=AUTH_SIGNING_KEY="$sign" \
        --dry-run=client -o yaml | {{k}} apply -f -

# Applies the SERVER manifests only (00-50). `-f deploy/k3d/` is non-recursive, so the agent Job under
# deploy/k3d/agent/ is deliberately EXCLUDED here: its image (voxeldust-agent:dev) is built+imported only by
# `k3d-agent`, so applying it in the base bring-up would leave an ImagePullBackOff pod that fails k3d-dod's
# `wait -l app=vd`. A future server manifest dropped in deploy/k3d/ is still auto-applied.
k3d-apply: k3d-secrets
    {{k}} apply -f deploy/k3d/

k3d-down:
    k3d cluster delete {{k3d_cluster}} || true

# ★ EXPORT THE CLUSTER'S TRUST FOR A CLIENT ON THIS MACHINE (2026-09-06): the mesh certificates the
# agent pod mounts (vd-mtls: ca/node/key) into a local trust folder, and the login signing key (the only
# process-env var the client reads) into `auth.env` beside them, 0600. The same material the agent pod
# gets — never a second trust.
k3d-trust-export:
    #!/usr/bin/env bash
    set -euo pipefail
    T="{{k3d_trust_dir}}"; mkdir -p "$T"; chmod 700 "$T"
    {{k}} get secret vd-mtls -o json | python3 -c '
    import base64, json, sys
    d = json.load(sys.stdin)["data"]
    for f in ("ca.der", "node.der", "key.der"):
        open(sys.argv[1] + "/" + f, "wb").write(base64.b64decode(d[f]))
    ' "$T"
    printf 'VD_AUTH_SIGNING_KEY=%s\n' "$({{k}} get secret vd-auth-signing -o jsonpath='{.data.AUTH_SIGNING_KEY}' | base64 -d)" > "$T/auth.env"
    chmod 600 "$T"/*
    ls -la "$T"
    echo "k3d-trust-export: the cluster's trust is at $T (gateway door 127.0.0.1:{{k3d_door_port}})"

# ★ THE SHIPPED LOGIN FROM THIS MACHINE (2026-09-06): the agent's own boundary scenario — the exact script
# the agent pod runs — driving a RELEASE client on the host through the door. Proves a client outside the
# cluster logs in, walks and crosses the boundary before anyone opens a window. Needs k3d-trust-export.
k3d-agent-host:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build --release -p vd-bins --features dev-control --bin client --bin vdctl
    set -a; . "{{k3d_trust_dir}}/auth.env"; set +a
    export PATH="$(pwd)/target/release:$PATH"
    export VD_GATEWAY_ADDR="127.0.0.1:{{k3d_door_port}}" VD_CLIENT_QUIC="{{k3d_client_quic}}"
    export VD_DEVCTL_PORT="{{k3d_devctl_port}}" VD_TRUST_DIR="{{k3d_trust_dir}}"
    bash docker/scenario-boundary.sh
    echo "k3d-agent-host OK: a client on this machine logged in through the door and crossed the boundary"

# ★ FLY IN A WINDOW AGAINST THE CLUSTER (2026-09-06): the release window client through the door. Same
# launcher as the dev cluster's window (scripts/client.sh), in its k3d mode. Needs k3d-trust-export.
k3d-fly:
    VD_BUILD=release VD_K3D_DOOR_PORT="{{k3d_door_port}}" VD_K3D_CLIENT_QUIC="{{k3d_client_quic}}" \
    VD_K3D_DEVCTL_PORT="{{k3d_devctl_port}}" VD_K3D_TRUST_DIR="{{k3d_trust_dir}}" \
    scripts/client.sh --k3d --window

# DoD (the demand shape, foundation slice 5, 2026-09-06): 2/2 Ready, /metrics served. No shard holds a realm
# until somebody logs in — the orchestrator forks one on demand — so "bootstrapped" is the AGENT's login
# (`k3d-agent`), not a static realm. This gate proves the two long-lived pods boot and serve.
k3d-dod:
    #!/usr/bin/env bash
    set -euo pipefail
    # Cold-start budget: a fresh cluster schedules 2 pods, runs the entrypoint DNS-resolve loop for each and
    # boots the mesh — empirically >120s on Docker-Desktop k3d. 300s absorbs it. `-l app=vd` matches only the
    # 2 server pods (the agent Job lives under deploy/k3d/agent/, not applied here).
    # No `rollout status`: every StatefulSet here updates OnDelete (the no-rolling-updates ruling,
    # 2026-08-24), and kubectl has no rollout status for that strategy. The readiness wait below IS the
    # cold-start gate; this loop only waits for each set to have MINTED its pod so the wait has a subject.
    for i in $(seq 1 60); do
        n=$({{k}} get pods -l app=vd -o name 2>/dev/null | wc -l | tr -d ' ')
        [ "$n" -ge 2 ] && break
        sleep 5
    done
    {{k}} wait --for=condition=Ready pod -l app=vd --timeout=300s
    {{k}} port-forward svc/vd-orch 9100:9100 >/dev/null 2>&1 & pf=$!; trap 'kill $pf 2>/dev/null || true' EXIT
    sleep 2
    curl -sf http://127.0.0.1:9100/metrics >/dev/null || { echo "FAIL: /metrics unreachable"; exit 1; }
    echo "DoD OK: 2/2 Ready, /metrics served (the demand shape: a shard is forked at the first login — see k3d-agent)"

# The full LIVE bring-up (deferred; run explicitly). Secrets are minted self-contained by k3d-secrets.
k3d-all: k3d-up k3d-load k3d-apply k3d-dod

# S6 (the demand shape): prove the peer-addr AUTO-PUSHER survives a real pod RESCHEDULE (new pod IP, same DNS
# name). Kill vd-orch-0 — the pod that holds the directory AND hosts every forked shard, so its death takes
# the whole realm tree with it; the StatefulSet reschedules it with a FRESH pod IP + the same durable PVCs
# (M3 boot-counter +1, redb store + the shards' outboxes survive). ASSERT the IP actually changed (a same-IP
# restart is a vacuous pass), then that the cluster RE-BOOTSTRAPS: the agent logs in again, the rescheduled
# orchestrator forks the home shard at its new address, the agent walks — which requires the gateway's
# INITIATED reliable traffic to resume at the orchestrator's NEW IP (the auto-pusher's job;
# reply-on-connection only covers replies). Reuses k3d-agent-e2e (baseline) + k3d-reschedule-verify.
k3d-reschedule-e2e: k3d-agent-e2e
    #!/usr/bin/env bash
    set -euo pipefail
    echo "S6: baseline: the agent logged in and walked (k3d-agent-e2e passed via the dep chain)."
    old_ip="$({{k}} get pod vd-orch-0 -o jsonpath='{.status.podIP}')"
    old_uid="$({{k}} get pod vd-orch-0 -o jsonpath='{.metadata.uid}')"
    echo "S6: vd-orch-0 OLD podIP=$old_ip — deleting the pod (StatefulSet reschedules: same DNS, NEW IP)."
    # `--wait=false`: a StatefulSet re-creates the SAME pod name within seconds, and `kubectl delete`'s wait
    # then watches the NEW object under the old name and never returns (measured 2026-09-06: six minutes
    # hung). The replacement is recognised by its UID instead.
    {{k}} delete pod vd-orch-0 --wait=false
    for i in $(seq 1 120); do
        uid="$({{k}} get pod vd-orch-0 -o jsonpath='{.metadata.uid}' 2>/dev/null || true)"
        [ -n "$uid" ] && [ "$uid" != "$old_uid" ] && break
        sleep 2
    done
    {{k}} wait --for=condition=Ready pod/vd-orch-0 --timeout=300s
    new_ip="$({{k}} get pod vd-orch-0 -o jsonpath='{.status.podIP}')"
    echo "S6: vd-orch-0 NEW podIP=$new_ip"
    [ "$new_ip" != "$old_ip" ] || { echo "S6 VACUOUS: podIP unchanged ($old_ip) — the reschedule did not move the pod; re-run"; exit 1; }
    just k3d-reschedule-verify
    if {{k}} logs vd-gateway-0 | grep -qi "re-plumbed a peer"; then
        echo "S6: CONFIRMED — the gateway auto-resolver logged a re-plumb to the orchestrator's new address."
    else
        echo "S6: NOTE — no explicit re-plumb log (the round-trip may have ridden reply-on-connection); recovery still proven by the re-login."
    fi
    echo "S6 OK: survived a pod reschedule (podIP $old_ip -> $new_ip) and RE-BOOTSTRAPPED (the agent logged in again)."

# The S6 recovery probe: the agent logs in again after the reschedule and walks. Split out so
# k3d-reschedule-e2e re-runs the identical proof post-kill without duplicating it.
k3d-reschedule-verify:
    #!/usr/bin/env bash
    set -euo pipefail
    just k3d-agent

# ---- S5a: in-cluster agent-HR6 continuous testing --------------------------------------------
# The agent image = client + vdctl, --features dev-control (Bevy-FREE), SEPARATE from the server image so the
# dev-control listener never links into the server bins. It logs an avatar into the LIVE gateway over the pod
# network (QUIC) and is driven by vdctl over loopback TCP, asserting on DevState (GPU-free).
agent_image := env_var_or_default("VD_AGENT_IMAGE", "voxeldust-agent:dev")

agent-image-build:
    docker build -f docker/agent.Dockerfile -t {{agent_image}} .

# Build+import the agent image, run the boundary Job against the LIVE cluster, report pass/fail from the Job's
# terminal condition + dump logs. Run AFTER k3d-dod (the cluster must be bootstrapped so the shard holds a realm).
k3d-agent: agent-image-build k3d-secrets
    #!/usr/bin/env bash
    set -euo pipefail
    k3d image import {{agent_image}} -c {{k3d_cluster}}
    {{k}} delete job vd-agent-boundary --ignore-not-found
    {{k}} apply -f deploy/k3d/agent/60-agent.yaml
    echo "waiting for the agent Job (bounded by activeDeadlineSeconds) ..."
    if {{k}} wait --for=condition=complete job/vd-agent-boundary --timeout=200s 2>/dev/null; then
        echo "k3d-agent OK: avatar logged in + crossed the boundary (pos threshold met on DevState)"
        {{k}} logs job/vd-agent-boundary; exit 0
    fi
    echo "k3d-agent FAIL: the crossing did not pass — logs + status:"
    {{k}} logs job/vd-agent-boundary --tail=200 || true
    {{k}} describe job/vd-agent-boundary | tail -20
    exit 1

# Convenience: full server bring-up THEN the agent proof (deferred, run explicitly).
k3d-agent-e2e: k3d-all k3d-agent
