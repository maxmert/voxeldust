# Voxeldust greenfield — local gates (no CI yet; these ARE the gates, run pre-merge).
# HR5 coverage policy: docs/design/coverage_e2e.md. Coverage runs on a pinned NIGHTLY
# (branch coverage + #[coverage(off)] are nightly-only); product builds stay on stable.

# Pinned coverage nightly (installed 2026-06-07; rustc 1.98.0-nightly 61d7280f3).
# Re-pin deliberately via VD_COVERAGE_TOOLCHAIN or `just coverage-setup`.
coverage_toolchain := env_var_or_default("VD_COVERAGE_TOOLCHAIN", "nightly-2026-06-06")

# Tier-A: the 100%-region+branch domain (HR5). Each crate's OWN tests must cover
# its full surface (llvm counts regions per compiled instance, so leaning on the
# vd-tests binary would double-instance every crate — learned in P1.7).
tier_a := "-p vd-core -p vd-wire -p vd-sim -p vd-node -p vd-connection-plane -p vd-harness"

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

fmt:
    cargo fmt --all

# Everything a merge requires.
gate: fmt lint test coverage

# One-time setup helper.
coverage-setup:
    rustup toolchain install nightly --component llvm-tools-preview
    cargo +nightly install cargo-llvm-cov --locked
    @echo "Pin the installed nightly via VD_COVERAGE_TOOLCHAIN (e.g. nightly-2026-06-01) in your shell or .envrc"
