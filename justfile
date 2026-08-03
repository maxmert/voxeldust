# Voxeldust greenfield — local gates (no CI yet; these ARE the gates, run pre-merge).
# HR5 coverage policy: docs/design/coverage_e2e.md. Coverage runs on a pinned NIGHTLY
# (branch coverage + #[coverage(off)] are nightly-only); product builds stay on stable.

# Pinned coverage nightly (installed 2026-06-07; rustc 1.98.0-nightly 61d7280f3).
# Re-pin deliberately via VD_COVERAGE_TOOLCHAIN or `just coverage-setup`.
coverage_toolchain := env_var_or_default("VD_COVERAGE_TOOLCHAIN", "nightly-2026-06-06")

# Tier-A: the 100%-region+branch domain (HR5). Each crate's OWN tests must cover
# its full surface (llvm counts regions per compiled instance, so leaning on the
# vd-tests binary would double-instance every crate — learned in P1.7).
tier_a := "-p vd-core -p vd-devproto -p vd-wire -p vd-sim -p vd-node -p vd-connection-plane -p vd-harness -p vd-client -p vd-client-harness"

# Inner loop: full deterministic suite (in-process tiers, fast).
test:
    cargo test --workspace

# Inner-loop coverage check: Tier-A only, 100% region + branch, fails the build under 100%.
# (cargo-llvm-cov has no --fail-under-branches; the report step enforces it from the
#  same profdata via the JSON summary.)
coverage-fast:
    cargo +{{coverage_toolchain}} llvm-cov --branch {{tier_a}} \
        --ignore-filename-regex '(/bin/|/tests/)' \
        --fail-under-regions 100 --fail-under-functions 100 \
        -- --quiet
    cargo +{{coverage_toolchain}} llvm-cov report --branch \
        --ignore-filename-regex '(/bin/|/tests/)' \
        --json --summary-only | python3 -c "import json,sys; \
        t=json.load(sys.stdin)['data'][0]['totals']['branches']; \
        missed=t['count']-t['covered']; \
        sys.exit(0 if missed==0 else print(f'BRANCH GATE: {missed} missed branches ({t[\"percent\"]:.2f}%)') or 1)"

# io-prod Tier-B RATCHETED FLOOR (HR5: io-prod is process-tier, never 100% — a SIGKILL can lose the final
# counter flush). This measures io-prod's OWN in-process unit tests (deterministic — no SIGKILL counter loss)
# and fails under a recorded floor, so the crash-durability (RedbStore) + mesh (quinn) code can never regress
# in coverage silently. The floor is conservative (below the ~92% measured baseline, absorbing quinn-loopback
# timing variance) and RATCHETS UP — raise it as coverage stabilizes/improves, never lower it. The DEEPER
# process-tier %c merge (the spawned node BINARIES via `orch-crash-cov`) is owed (DEFERRED.md D-40).
tier_b_floor := "90"
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
    cargo test -p vd-bins --features dev-control --test client_load
    cargo test -p vd-bins --features dev-control --test dev_control_nav

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

# RLM Step 4b (the crash-replay SOAK): a ~30k-op crash/reorder/dup stream through the REAL realm
# reconcile kernel, asserting determinism + crash-no-reap + no-strand + bounded-ledger +
# INV-SNAPSHOT-SAFETY throughout. RELEASE build (the `#[cfg(not(debug_assertions))]` soak is inert in
# debug; `just test` still runs the 1024-case fixed-seed proptest + the 6 named witnesses). A sustained
# ARM-A≠ARM-B divergence here is the concrete trigger to promote the deferred durable snapshot (D-RLM-2).
rlm-soak:
    cargo test --release -p vd-sim rlm_soak -- --nocapture --test-threads=1

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
    cargo test -p vd-bins --features dev-control,render --test render_smoke -- --nocapture

# G-RENDER-BOXES-SMOKE (Visual Crossing Playground V3 pixel proof): bring up the cluster, launch a
# HEADLESS `client --capture --realm-boxes <boxes.json>` (ONE translucent colored realm box),
# capture a real wgpu-readback frame, and assert the box is PIXEL-VISIBLE inside its projected
# screen region (H2 — not a bare content fraction) + zero magenta. Same GPU-required, LOCAL-gate
# preconditions as render-smoke (no CI, no software fallback; steer with WGPU_BACKENDS).
render-boxes-smoke:
    cargo test -p vd-bins --features dev-control,render --test render_boxes_smoke -- --nocapture

# G-RENDER-CROSSING-SMOKE (Visual Crossing Playground V4 pixel proof): bring up the DUAL cluster with
# an injected walk-into crossing trigger (VD_DEVCLUSTER_BOUNDARIES), launch a HEADLESS
# `client --capture --realm-boxes` (TWO translucent boxes A@System(7) / B@System(8)), WalkTo the
# avatar across the boundary, and capture BEFORE (dot in box A) + AFTER (dot in box B) — asserting
# the location/expected_box flip + world-motion (state) AND the dot's pixels move A→B (H2) + zero
# magenta. Same GPU-required, LOCAL-gate preconditions as render-smoke.
render-crossing-smoke:
    cargo test -p vd-bins --features dev-control,render --test render_crossing_smoke -- --nocapture

# NODE-PER-REALM WALK GATE (task #149) — the HEADLESS process-tier walk proof (supersedes the retired
# `triple-crossing-smoke`, whose --triple co-hosting cluster now thrashes under node-per-realm). Brings up the
# `--forest` cluster (orchestrator + gateway + SIX single-realm shards: System 7, Planet 7, Station 7, Area 7,
# Galaxy, System 8, NO co-hosting), logs in a REAL headless durable player over localhost QUIC, and drives it
# with dev-control `WalkTo` along +X through the chain of CROSS-NODE re-homes. Asserts the player ARRIVES at
# every leg (movement never freezes across a crossing) and the subject Entity's directory fence stays SMALL
# (one clean commit per crossing — the thrash guard). No GPU: this is the CI walk gate; the live window is the
# interim visual proof (DEFERRED.md — the GPU pixel-capture walk is owed a re-base onto the --forest cluster).
node-per-realm-walk:
    cargo test -p vd-bins --features dev-control --test node_per_realm_walk -- --nocapture

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
    cargo test -p vd-bins --features dev-control --test rlm_demand_login -- --test-threads=1 --nocapture

# Everything a merge requires (render-smoke/render-boxes-smoke are GPU-required + local; spike2a is
# a release build — all documented in their recipes). fmt-check FAILS on drift (run `just fmt` to
# fix); every gate step is fail-on-violation, none mutates the tree.
gate: fmt-check lint lint-combos test client-load orch-crash spike2a spike3a rlm-soak render-smoke render-boxes-smoke render-crossing-smoke node-per-realm-walk rlm-proc-spawn rlm-kill9 rlm-demand-login coverage

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

k3d-up:
    #!/usr/bin/env bash
    set -euo pipefail
    k3d cluster list {{k3d_cluster}} >/dev/null 2>&1 && k3d cluster delete {{k3d_cluster}} || true
    k3d cluster create {{k3d_cluster}} --wait --timeout 120s
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

# DoD: 3/3 Ready, /metrics served, cluster_bootstrapped=true (a shard holds a realm).
k3d-dod:
    #!/usr/bin/env bash
    set -euo pipefail
    # Cold-start budget: a fresh cluster must schedule 3 pods, run the entrypoint DNS-resolve loop for each,
    # boot the mesh, and grant a realm — empirically >120s on Docker-Desktop k3d. 300s absorbs it. `-l app=vd`
    # now matches only the 3 server pods (the agent Job lives under deploy/k3d/agent/, not applied here).
    for s in vd-orch vd-gateway vd-shard; do {{k}} rollout status statefulset/$s --timeout=300s; done
    {{k}} wait --for=condition=Ready pod -l app=vd --timeout=300s
    {{k}} port-forward svc/vd-orch 9100:9100 >/dev/null 2>&1 & pf=$!; trap 'kill $pf 2>/dev/null || true' EXIT
    sleep 2
    curl -sf http://127.0.0.1:9100/metrics >/dev/null || { echo "FAIL: /metrics unreachable"; exit 1; }
    # "Bootstrapped" = a SHARD holds a realm (admin::AdminSnapshot::cluster_bootstrapped is a Rust method, NOT a
    # serialized field — the snapshot JSON exposes `directory`/`leases`/`sagas`, so read the directory: a
    # `"authority":"shard:..."` entry IS the bootstrap signal).
    for i in $(seq 1 30); do
        snap=$(curl -s http://127.0.0.1:9100/admin/snapshot)
        if command -v jq >/dev/null 2>&1; then
            bs=$(printf '%s' "$snap" | jq -r '[.directory[]? | select(.authority | startswith("shard:"))] | length > 0')
        elif printf '%s' "$snap" | grep -q '"authority":[[:space:]]*"shard:'; then bs=true; else bs=false; fi
        [ "$bs" = "true" ] && { echo "DoD OK: 3/3 Ready, /metrics served, a shard holds a realm (bootstrapped)"; exit 0; }
        echo "no shard-held realm yet, retry $i"; sleep 2
    done
    echo "DoD FAIL: not bootstrapped (no shard holds a realm)"; {{k}} get pods; {{k}} exec vd-shard-0 -- true 2>/dev/null; exit 1

# The full LIVE bring-up (deferred; run explicitly). Secrets are minted self-contained by k3d-secrets.
k3d-all: k3d-up k3d-load k3d-apply k3d-dod

# S6: prove the peer-addr AUTO-PUSHER survives a real pod RESCHEDULE (new pod IP, same DNS name). Kill
# vd-shard-0; the StatefulSet reschedules it with a FRESH pod IP + the same durable PVC (M3 boot-counter +1,
# redb survives). ASSERT the IP actually changed (a same-IP restart is a vacuous pass), then that the cluster
# RE-BOOTSTRAPS (a shard re-holds a realm) — which requires the orch+gateway's INITIATED reliable traffic to
# resume at the shard's NEW IP (the auto-pusher's job; reply-on-connection only covers replies). Asserts ONLY
# the RELIABLE control plane (D-18: unreliable snapshots don't flow on the Docker-Desktop k3d overlay). Reuses
# k3d-all (baseline bootstrap) + k3d-dod (recovery probe).
k3d-reschedule-e2e: k3d-all
    #!/usr/bin/env bash
    set -euo pipefail
    echo "S6: baseline bootstrapped (k3d-all passed via the dep chain)."
    old_ip="$({{k}} get pod vd-shard-0 -o jsonpath='{.status.podIP}')"
    echo "S6: vd-shard-0 OLD podIP=$old_ip — deleting the pod (StatefulSet reschedules: same DNS, NEW IP)."
    {{k}} delete pod vd-shard-0 --wait=true
    {{k}} rollout status statefulset/vd-shard --timeout=180s
    {{k}} wait --for=condition=Ready pod/vd-shard-0 --timeout=180s
    new_ip="$({{k}} get pod vd-shard-0 -o jsonpath='{.status.podIP}')"
    echo "S6: vd-shard-0 NEW podIP=$new_ip"
    [ "$new_ip" != "$old_ip" ] || { echo "S6 VACUOUS: podIP unchanged ($old_ip) — the reschedule did not move the pod; re-run"; exit 1; }
    # RE-BOOTSTRAP after the reschedule — a shard must re-hold a realm, requiring initiated traffic to resume at
    # the new IP (the auto-pusher re-plumb). Reuse the exact k3d-dod probe.
    just k3d-reschedule-verify
    # Direct evidence the auto-resolver PUSHED (not just reply-on-connection recovery): the re-plumb log on a
    # survivor. Non-fatal (recovery is the hard gate; the push log is confirming evidence).
    if {{k}} logs vd-orch-0 | grep -qi "re-plumbed a peer"; then
        echo "S6: CONFIRMED — the orch auto-resolver logged a re-plumb to the shard's new address."
    elif {{k}} logs vd-gateway-0 | grep -qi "re-plumbed a peer"; then
        echo "S6: CONFIRMED — the gateway auto-resolver logged a re-plumb to the shard's new address."
    else
        echo "S6: NOTE — no explicit re-plumb log (the grant round-trip may have ridden reply-on-connection); recovery still proven by re-bootstrap."
    fi
    echo "S6 OK: survived a pod reschedule (podIP $old_ip -> $new_ip) and RE-BOOTSTRAPPED."

# The k3d-dod recovery probe re-used by S6 (a shard re-holds a realm after the reschedule). Split out so
# k3d-reschedule-e2e re-runs the identical bootstrap assertion post-kill without duplicating the probe.
k3d-reschedule-verify:
    #!/usr/bin/env bash
    set -euo pipefail
    {{k}} port-forward svc/vd-orch 9100:9100 >/dev/null 2>&1 & pf=$!; trap 'kill $pf 2>/dev/null || true' EXIT
    sleep 2
    for i in $(seq 1 45); do
        snap=$(curl -s http://127.0.0.1:9100/admin/snapshot)
        if command -v jq >/dev/null 2>&1; then
            bs=$(printf '%s' "$snap" | jq -r '[.directory[]? | select(.authority | startswith("shard:"))] | length > 0')
        elif printf '%s' "$snap" | grep -q '"authority":[[:space:]]*"shard:'; then bs=true; else bs=false; fi
        [ "$bs" = "true" ] && { echo "S6: RE-BOOTSTRAPPED — a shard re-holds a realm after the reschedule."; exit 0; }
        echo "S6: no shard-held realm yet post-reschedule, retry $i"; sleep 2
    done
    echo "S6 FAIL: the cluster did NOT re-bootstrap after the reschedule (initiated traffic never resumed at the new shard IP)"; {{k}} get pods; exit 1

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
