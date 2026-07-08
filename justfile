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
client-load:
    cargo test -p vd-bins --features dev-control --test client_load

# D-6 D-delta: the orchestrator durability crash gates. `orchestrator_crash` is the SIGKILL-mid-fsync proof
# (a real kill-9 while a directory grant sits submitted-but-pre-fsync loses <=1 batch + recovers
# consistently); it builds the orchestrator binary WITH the writer-pause hook (`store-test-hooks`).
# `boot_guard` (always compiled) locks the HR1 ephemeral-store boot guard's reject + accept arms.
# `boot_counter_crashloop` is the R-6b M3 CAPSTONE: a real SIGKILL + sub-second shard restart, proving the
# durable boot-counter (higher incarnation) keeps the orchestrator from silently DEDUP-dropping the restart's
# reliable re-grant (RED control: a fixed incarnation DOES get deduped). Serial + single-threaded (each
# spawns a real cluster + SIGKILLs a process — concurrent cluster tests would contend for CPU/ports).
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

# Everything a merge requires (render-smoke is GPU-required + local; spike2a is a release
# build — both are documented in their recipes). fmt-check FAILS on drift (run `just fmt`
# to fix); every gate step is fail-on-violation, none mutates the tree.
gate: fmt-check lint lint-combos test client-load orch-crash spike2a render-smoke coverage

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
        kubeconform -strict -summary -kubernetes-version 1.31.0 deploy/k3d/*.yaml
    else
        echo "kubeconform not found — falling back to 'kubectl apply --dry-run=client' (offline, schema-lite)"
        kubectl apply --dry-run=client -f deploy/k3d/ >/dev/null && echo "client dry-run OK"
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

# Mint the two Secrets from the dev tooling. VD_AUTH_PUBKEY_HEX must be exported from `gen-authkey` first.
k3d-secrets:
    #!/usr/bin/env bash
    set -euo pipefail
    T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
    cargo run -q -p vd-bins --bin vd-devcluster -- gen-trust "$T"
    {{k}} create secret generic vd-mtls \
        --from-file=ca.der="$T/ca.der" --from-file=node.der="$T/node.der" --from-file=key.der="$T/key.der" \
        --dry-run=client -o yaml | {{k}} apply -f -
    : "${VD_AUTH_PUBKEY_HEX:?export VD_AUTH_PUBKEY_HEX from: cargo run -p vd-bins --bin vd-devcluster -- gen-authkey}"
    {{k}} create secret generic vd-auth --from-literal=AUTH_PUBKEY="$VD_AUTH_PUBKEY_HEX" \
        --dry-run=client -o yaml | {{k}} apply -f -

k3d-apply: k3d-secrets
    {{k}} apply -f deploy/k3d/

k3d-down:
    k3d cluster delete {{k3d_cluster}} || true

# DoD: 3/3 Ready, /metrics served, cluster_bootstrapped=true (a shard holds a realm).
k3d-dod:
    #!/usr/bin/env bash
    set -euo pipefail
    for s in vd-orch vd-gateway vd-shard; do {{k}} rollout status statefulset/$s --timeout=120s; done
    {{k}} wait --for=condition=Ready pod -l app=vd --timeout=120s
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

# The full LIVE bring-up (deferred; run explicitly). Requires VD_AUTH_PUBKEY_HEX exported.
k3d-all: k3d-up k3d-load k3d-apply k3d-dod
