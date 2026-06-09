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

# Pre-merge: full coverage incl. (once io-prod exists) the process-tier merge via
# `show-env` + LLVM_PROFILE_FILE %p-%m%c (continuous mode is MANDATORY: the harness
# SIGKILLs processes; without %c their profile counters are silently lost).
coverage: coverage-fast
    @echo "process-tier coverage merge lands with io-prod (P3); Tier-B ratcheted floor applies then"

# Open the HTML region report to SEE the uncovered region.
coverage-html:
    cargo +{{coverage_toolchain}} llvm-cov --branch {{tier_a}} --html --open

# Lint with the seam-enforcement rules (sim/node clippy.toml disallowed-methods/types).
lint:
    cargo clippy --workspace --all-targets -- -D warnings

# Lint the OTHER supported vd-bins feature combos (the workspace lint covers only the
# default set): `dev-control` alone is the DAILY agent-loop build (client.sh headless) and
# `render` alone exercises the requires-dev-control rejection arms — combo rot in either
# would otherwise surface only when a human next runs client.sh. The full
# `dev-control,render` combo is compiled by `render-smoke`.
lint-combos:
    cargo clippy -p vd-bins --features dev-control --all-targets -- -D warnings
    cargo clippy -p vd-bins --features render --all-targets -- -D warnings

# SCALE-1 (the K-client load/collapse gate): K real dev-control clients log into one
# cluster concurrently — fan-out + per-client routing proven at the per-slot client cap.
client-load:
    cargo test -p vd-bins --features dev-control --test client_load

fmt:
    cargo fmt --all

# G-RENDER-SMOKE (HR6 permanent visual gate): bring up the local cluster, launch a HEADLESS
# `client --capture`, capture a real wgpu-readback frame, and assert no-magenta +
# content-present over it. Builds the client with `--features dev-control,render` (pulls
# Bevy). REQUIRES A WORKING GPU ADAPTER (the dev Metal GPU) — it is a LOCAL gate (no CI yet,
# no software fallback). Steer the adapter with WGPU_BACKENDS / WGPU_POWER_PREF if needed.
render-smoke:
    cargo test -p vd-bins --features dev-control,render --test render_smoke -- --nocapture

# Everything a merge requires (render-smoke is GPU-required + local; see its recipe).
gate: fmt lint lint-combos test client-load render-smoke coverage

# One-time setup helper.
coverage-setup:
    rustup toolchain install nightly --component llvm-tools-preview
    cargo +nightly install cargo-llvm-cov --locked
    @echo "Pin the installed nightly via VD_COVERAGE_TOOLCHAIN (e.g. nightly-2026-06-01) in your shell or .envrc"
