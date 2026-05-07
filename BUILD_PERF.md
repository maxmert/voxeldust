# Build & Runtime Performance Playbook

Tuning history and contributor playbook for keeping `cargo build` fast and
release binaries fast at runtime. Pair with the plan at
`/Users/maxim/.claude/plans/at-the-moment-the-parallel-yeti.md`.

## One-time setup

```bash
# Required (already pinned via rust-toolchain.toml):
rustup show          # confirms 1.94.1 + rustfmt + clippy + rust-src

# Optional dev hygiene:
cargo install cargo-machete   # find unused deps
cargo install cargo-deny      # license + advisory check
cargo install cargo-nextest   # parallel test runner

# Optional linker speedup (Apple Silicon):
brew install llvm
# Then uncomment the link-arg line in .cargo/config.toml.
# macOS 14+ ships ld-prime which is already fast — only install lld if your
# baseline measurements show link time dominating.
```

## Day-to-day workflows

### Fast client iteration (Bevy as a dylib)
```bash
cargo run -p client --features fast -- --gateway 127.0.0.1:7777 --name dev
```
The `fast` feature in `client/Cargo.toml` maps to `bevy/dynamic_linking`. With
it, editing client code does not relink Bevy's ~61 crates. Never ship a release
build with `--features fast` — the dylib is dev-only.

### Schema regen
```bash
./build_protocol.sh
```
Writes to `protocol-fb/src/voxeldust_generated.rs` (post-Phase-2). Schema-only
edits should rebuild only the `voxeldust-protocol-fb` crate plus its direct
importers (`shard_message.rs`, `client_message.rs`).

### Release builds
```bash
cargo build --release             # local release testing (lto = thin)
cargo build --profile dist        # production artifacts (lto = fat)
```

## Profile reference

| Profile | LTO | codegen-units | debug | strip | When |
|---|---|---|---|---|---|
| `dev` | off | 256 | line-tables-only | no | normal iteration |
| `dev.package."*"` (deps) | off | 16 | none | no | deps cached, opt-level=3 for runtime |
| `release` | thin | 1 | line-tables-only | symbols | local release testing |
| `dist` | fat | 1 | line-tables-only | symbols | shipped binaries |

## Heavy-dep overrides

`[profile.dev.package."*"] opt-level = 3` (the workspace-default for deps,
following the Bevy book) is the right call for runtime hot-path crates like
Bevy / wgpu / rapier — but it pays its weight only when the dep is actually
running every frame. A dep that's poorly factored (mandatory pulls of heavy
sub-deps) or rarely called wastes minutes of compile time at `opt-level = 3`.

Current overrides — drop these crates to `opt-level = 0` because they're cold
paths in voxeldust:

| Crate | Why it's on the list |
|---|---|
| `brahe` | Used 2× in `core/` for Keplerian → ECI conversion (planet sim tick); v1.2's `Cargo.toml` declares `pyo3` / `numpy` / `polars` / `rsofa` as **mandatory** deps so even `default-features = false` can't drop them |
| `pyo3` | Pulled by brahe; we never call into Python |
| `numpy` | Pulled by brahe + pyo3; we never call into Python |
| `polars` | Pulled by brahe (DataFrame engine); we never use it |
| `rsofa` | Pulled by brahe (SOFA astronomy time); brahe wraps it for us |

How to spot a new offender:
1. `cargo build --timings` and look for a single dep dominating the wall-clock
   (and not in the rendering hot path).
2. `cargo tree -p <suspect>` to see what it pulls in. If the chain looks
   "Python interop" / "DataFrame" / "ML" — it's a candidate for opt-level=0.
3. `grep -rln '<crate>::' src/` to confirm we hit it on a cold path. If yes,
   add to the override list above.

## Diagnosing a slow build

1. **Capture flame graph**:
   ```bash
   cargo clean
   cargo build -p client --timings
   open target/cargo-timings/cargo-timing.html
   ```
   Look for crates with disproportionate wall-clock time relative to their LOC.

2. **Identify what invalidated the cache**:
   ```bash
   cargo build -p client -v 2>&1 | grep -E 'fingerprint|Compiling' | head -50
   ```
   "Dirty" reasons reveal whether a config, a dep, or a file change triggered
   the rebuild.

3. **Common causes**:
   - **Anything in `core/` was edited** → cascades through 10 dependent crates.
     Check whether the edit could live in a downstream crate instead.
   - **Schema regen** (`protocol-fb/`) → this is unavoidable; the protocol-fb
     extraction in Phase 2 contains the blast radius.
   - **Cargo.toml edited in a workspace dep** → invalidates that crate + all
     downstream.
   - **`RUSTFLAGS` env var set in shell** → wipes target. Never export
     `RUSTFLAGS` globally; use `.cargo/config.toml`.
   - **Toolchain bump** → `rustup update` invalidates everything. Pinned via
     `rust-toolchain.toml` so this only happens deliberately.

## Dependency hygiene

```bash
cargo machete                     # unused deps in each Cargo.toml
cargo tree -p client --duplicates # crates pulled in at multiple versions
cargo deny check                  # licenses, advisories, banned crates
```
Run before merging any change that adds a dep. Removing a dep is a strict
compile-time win; duplicate versions of a dep usually indicate a feature flag
mismatch worth pinning at the workspace level.

## Workspace layout (post-refactor)

```
types/          — foundational primitives: ShardId, BlockId, SignalProperty,
                  FunctionalBlockKind + BlockKindSignalSchema, hud_delta_flags
                  (deps: serde only)
protocol-fb/    — FlatBuffers generated wire types (deps: flatbuffers)
signal/         — signal subsystem: channels, grants, auth, ingress, rate
                  limit, HUD session, wire dict, radio + ship subscriptions,
                  per-block config, seat presets (deps: types + protocol-fb +
                  bevy_ecs + crypto)
core/           — game logic that doesn't fit a deeper crate: shard_message,
                  client_message, block registry, character KCC, autopilot,
                  weather, geophysics, system, planet_rotation, media, ecs
                  (deps: types + protocol-fb + signal + everything else)
shard-common/   — networking + observability + signal pipeline (depends on core)
{*,planet,ship,system,galaxy}-shard/   — per-shard authoritative loops
orchestrator/   — k8s/k3d cluster orchestration
gateway/        — entry-point auth + routing
client/         — Bevy 0.18 client (depends on core)
e2e-tests/      — cross-shard integration tests
```

The layering is the **compile firewall**. Adding a module to `core/` puts it
in the broadcast path of every higher crate. Prefer one of the lower crates
when adding new code:
* primitive type with no logic dependencies → `types/`
* signal-domain logic → `signal/`
* wire types crossing the network → `protocol-fb/` (when generated) or the
  appropriate domain crate (when hand-written)
Only fall back to `core/` if the addition is general-purpose game code that
genuinely belongs at this layer.

## Runtime perf — production checklist

1. **Build with `dist` profile**:
   ```bash
   cargo build --profile dist -p planet-shard
   ```

2. **PGO** (when prepping a real release):
   ```bash
   # 1. Instrumented build
   RUSTFLAGS="-Cprofile-generate=/tmp/pgo" \
     cargo build --profile dist -p planet-shard

   # 2. Drive a representative workload (e.g. a saved demo session)
   ./target/dist/planet-shard --bind 0.0.0.0:7777 < workload.txt

   # 3. Merge raw profiles
   /opt/homebrew/opt/llvm/bin/llvm-profdata merge \
     -o /tmp/pgo/merged.profdata /tmp/pgo

   # 4. Optimised build
   RUSTFLAGS="-Cprofile-use=/tmp/pgo/merged.profdata \
              -Cllvm-args=-pgo-warn-missing-function" \
     cargo build --profile dist -p planet-shard
   ```
   Expected: 5–15 % runtime gain on the simulation hot path.

3. **Allocator** (per-binary opt-in, evaluate against profiling first):
   Add to a shard's `Cargo.toml`:
   ```toml
   [features]
   mimalloc = ["dep:mimalloc"]
   [dependencies]
   mimalloc = { version = "0.1", optional = true }
   ```
   And to its `main.rs`:
   ```rust
   #[cfg(feature = "mimalloc")]
   #[global_allocator]
   static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
   ```

## Measurements log

### Pre-Phase-1 baseline (user-reported)
```
cargo build / cargo run -p client (any cascade)        ~5–6 min
```

### Post-Phase-1 (toolchain pin + .cargo/config.toml + dev profile tuning + tokio feature subset + bevy/dynamic_linking gate)

Cold check (all profile + rustflags changes invalidate target/):
```
cargo check --workspace (cold)                         61m 35s   ⚠️ one-time
```
The 61 m cold cost is dominated by `[profile.dev.package."*"] opt-level = 3`
applied to every dep. Trade-off: a one-shot rebuild buys a faster dev-game
runtime + smaller cached artifacts thereafter. Subsequent invalidations
(toolchain bump only) re-pay this cost.

Steady-state incremental (cargo check, hot cache):
```
no-op cargo check (workspace → -p client warmup)       3m 21s
touch client/src/main.rs                                7.4 s   ✅ goal hit
touch core/src/seed.rs                                 13.4 s   ✅
touch core/src/voxeldust_generated.rs (pre-Phase-2)    13.4 s
```

### Post-Phase-2 (protocol-fb extracted)

One-time delegation cost:
```
cargo check -p voxeldust-protocol-fb (new crate cold)   6.9 s
cargo check -p voxeldust-core (graph rewire)            5m 07 s   one-time
cargo check --workspace (rewire warmup)                36.6 s
```

Steady-state incremental:
```
no-op cargo check                                      16.3 s
touch protocol-fb/src/voxeldust_generated.rs           19.4 s
touch core/src/system.rs                               14.0 s   ✅
touch client/src/main.rs                                6.8 s   ✅ goal hit
```

### Post-Phase-3 (voxeldust-types + voxeldust-signal extracted)

Phase 3a — `voxeldust-types` crate (foundational primitives). Hoists
`ShardId/ShardType/SessionToken/ShardState/ShardEndpoint/ShardInfo/ShardHeartbeat`,
`BlockId`, `SignalProperty`, `FunctionalBlockKind` + `BlockKindSignalSchema`,
and `hud_delta_flags` out of `core/`. Breaks the block↔signal cycle that
existed in `core/` (block::registry imported `SignalProperty` from
signal::types; signal::config took `FunctionalBlockKind` as parameter — both
sides now import from `voxeldust-types`). Backwards-compatible re-export
shims preserve every existing import path (`crate::shard_types::ShardId`,
`crate::block::registry::FunctionalBlockKind`,
`crate::client_message::hud_delta_flags`, etc.).

Phase 3b — `voxeldust-signal` crate (16 files, 8.4 KL):
* Wire-format types (`GrantsSnapshotData`, `GrantPublicView`,
  `SignalSubscribeData`, `SignalUnsubscribeData`) moved to
  `voxeldust-signal/src/wire.rs`. Re-exported from `core::client_message`
  and `core::shard_message` so existing serialize/deserialize logic keeps
  working.
* `core::signal` resolves via `pub use voxeldust_signal as signal;` umbrella
  shim — every `voxeldust_core::signal::*` import in shards/client/shard-common
  keeps resolving without touching 38+ call sites.
* `hmac/sha2/subtle/base64/rand/metrics` moved out of `core/Cargo.toml` into
  `signal/Cargo.toml`. Only `media.rs` reuses signal's HMAC primitives in
  core, so `hmac/sha2/subtle/rand` are dual-listed; `base64`/`metrics` were
  exclusively signal deps.

Steady-state incremental (cargo check, hot cache):
```
no-op cargo check                                       0.3 s   ✅
touch signal/src/lib.rs (signal-only cascade)          18.5 s
touch core/src/system.rs (core-leaf — signal stays cached!)  12.6 s   ✅ firewall
touch protocol-fb/src/voxeldust_generated.rs           25.7 s
touch client/src/main.rs                                6.5 s   ✅ goal hit
```

Tests: 172 signal tests pass, 4 types tests pass, workspace clean.

### Final binary-build verification

`cargo check` skips codegen + link. The user's actual workflow is `cargo run`,
which adds both. End-to-end measurement:
```
cargo test -p voxeldust-protocol-fb (cold, 0 tests)        48 s
cargo test -p voxeldust-core (cold, full suite + doctests) 8m 51 s   one-time
cargo build -p client (first build → binary, with link)   10m 04 s   one-time
touch core/src/lib.rs && cargo build -p client            14.2 s   ✅ goal hit
```
The **14.2 s incremental binary build** is the new steady-state for the
`cargo run -p client` iteration loop after a core-file edit.

### Net result

| Scenario | Before | After Phase 1+2+3 | Speedup |
|---|---:|---:|---:|
| Edit a client file (`cargo check`) | ~5–6 min | **6.5 s** | ~50× |
| Edit a leaf core file (`cargo check`) | ~5–6 min | **12.6 s** | ~25× |
| Edit a signal file (`cargo check`) | ~5–6 min | **18.5 s** | ~17× |
| Schema regen + cascade (`cargo check`) | ~5–6 min | **25.7 s** | ~14× |
| Edit a core file + **link** (`cargo build`) | ~5–6 min | **14.2 s** | ~25× |
| no-op cache hit (`cargo check`) | ~5–6 min | **0.3 s** | ~1000× |
| Cold full-workspace check | (unknown) | 61 m | one-time penalty |
| Cold full client binary build (post-rewire) | (unknown) | 10m | one-time penalty |

## References

- Bevy fast-compile docs: <https://bevyengine.org/learn/quick-start/getting-started/setup/#enable-fast-compiles-optional>
- Cargo profile docs: <https://doc.rust-lang.org/cargo/reference/profiles.html>
- rustc PGO docs: <https://doc.rust-lang.org/rustc/profile-guided-optimization.html>
- `split-debuginfo` on macOS: <https://doc.rust-lang.org/cargo/reference/profiles.html#split-debuginfo>
