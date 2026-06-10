# Voxeldust — Greenfield Rebuild (transfer-first)

Multiplayer voxel planet MMO — "Star Citizen meets Minecraft". Spherical voxel planets,
player-built ships you walk inside while flying, Newtonian physics, distributed shard
architecture. **This worktree is the approved total-greenfield rebuild** (June 2026);
the old code lives on `main`/`ecs-system` as reference/spec ONLY.

## THE binding specs — read before designing anything

- `docs/design/PLAN.md` — the approved plan: context, 6 hard rules, unified architecture, P0–P11 roadmap.
- `docs/design/*.md` — the hardened subsystem designs (connection_plane, transfer_protocol,
  test_harness, identity_persistence, generic_transfer, sealed_shards, coverage_e2e).
- `docs/design/integration.json` — 19 binding cross-design conflict resolutions + glossary.
- `docs/design/DEFERRED.md` — the binding registry of every interim/stub: WHAT proper solution is missing,
  WHERE it lives, WHEN (which slice/phase) it lands. A phase isn't done until its entries flip to 🟩.
- `docs/audit/` — evidence for why the old architecture was unfixable (root causes R1–R10).

## Hard rules (user-mandated; violations are defects)

1. **HR1 sealed shards** — a shard's World/rapier/redb is private; inter-shard bytes exist
   only as `InterShardFlow` arms (one reviewed file in `vd-wire`); sim cross-feeds are
   `EffectFree` `CouplingPort`s; gameplay cross-shard data rides the Signal system.
2. **HR2 generic transfer** — ANY entity kind crosses shards via the `TransferableKind`
   registry; Durable vs Transient is policy fan-out on ONE machinery (batched `TransientGo`).
3. **HR3 one tooling** — ONE transfer FSM/envelope/Fence/registry; ONE `shard` binary;
   shard types are `ShardProfile` capability configs. Never `match` on a shard kind in features.
4. **HR4 features once, run anywhere** — capability DAG + `FrameSpace` seam; every feature
   passes the identical fixture on ≥2 shard kinds (G-IDENTICAL) or it doesn't land.
5. **HR5 100% coverage** — Tier-A crates at 100% region+branch (`just coverage-fast`);
   exemptions only via `#[cfg_attr(coverage_nightly, coverage(off))]` or `coverage-exemptions.toml`.
   Generic-code gotcha (learned in SPIKE-0a/P0.3): llvm counts regions PER
   MONOMORPHIZATION — a branch inside a generic fn must be exercised for every
   instantiated type AND in every test binary that instantiates it. Discipline:
   (a) **generic fns are branchless shims** — serialize/lookup as straight-line
   expressions, ALL branching (`?`, `if`, `match`, error closures) in monomorphic
   helpers (see `core/src/tlv.rs` for the canonical shape; `field_decode_err` for
   hoisting closures out of generic bodies); (b) cover a crate's generics completely
   in its own unit tests; (c) integration tests exercise the full surface or none;
   (d) in tests prefer `assert_eq!`/`expect_err` equality over `assert!(matches!(…))`
   (the false arm is an uncoverable region) and split `assert!(a && b)` (short-circuit
   branches).
6. **HR6 agent-operable E2E** — client ships the `dev-control` harness (`vdctl`): input
   injection at the input-resource seam, wgpu readback screenshots/video, `runs/` manifests.

## Workspace

```
crates/core              vd-core   pure domain (ids, Fence, kind registry, TLV, celestial math, bands)
crates/wire              vd-wire   frozen wire contract + InterShardFlow + seam contracts
crates/sim               vd-sim    pure ECS/FSMs + sim::io ShardIo traits + io::mem test impls
crates/node              vd-node   build_app(NodeKind, cfg, io); step_tick(); universe clock
crates/connection-plane  vd-connection-plane  gateway internals (lib; gateway bin is a shell)
crates/harness           vd-harness  Topology, FaultFabric, ControlOracle, WireMonitor, ChaosRunner
tests                    vd-tests  accumulated scenario suites — never delete a scenario
```
Dependency rule: bins → node → sim → wire → core; harness → node + sim::io::mem; nothing
depends on a bin. Every node is lib + 4-line bin. `io-prod` / `tests-process` appear later.

## Non-negotiable conventions

- **No I/O outside the seam**: sim/node code gets time/rng/persistence/transport ONLY through
  `sim::io` traits (clippy `disallowed-methods`/`disallowed-types` enforce; `HashMap` with the
  default hasher is banned in sim/node — use `BTreeMap` or `DetHashMap`).
- **No magic numbers**: world/sim params seed-derived; entity props per-entity; operational
  params in ONE config struct (`TransportTuning`/`TransferTuning`), never inline literals.
- **NO client-side prediction** (interpolation on a 100–150 ms buffer). Players physically collide.
- **Fence discipline**: every authoritative action carries its `Fence`; receivers reject stale;
  the directory CAS is the only commit point. Source authority retained until dest Committed.
- **postcard v1 everywhere** (codec flag bit reserved); entity blobs are TLV-framed;
  decode-to-Default is BANNED for Durable kinds.
- **Determinism**: closed-form `f(seed, universe_tick)` for celestial math (Category A);
  rapier state is checkpoint-carried, never re-simulated cross-host (Category C); every
  physics→control boundary quantized to integer grids.
- **No `git commit` / `git push` without an explicit user request for that exact action.**

## Build & gates

```bash
cargo test --workspace      # full deterministic suite (fast: virtual clock, no sockets)
just gate                   # fmt + clippy(-D warnings) + tests + coverage — the pre-merge gate
just coverage-fast          # Tier-A 100% region+branch (HR5 inner loop)
just coverage-html          # see the uncovered region
```
Coverage runs on a pinned nightly (`VD_COVERAGE_TOOLCHAIN`); product builds on stable 1.94.1.

## Roadmap position

P0 (foundation: harness + frozen wire + FSMs + clock) → P1 stub shards → P1.5 client+HR6
harness → P2 THE transfer (one class, ghosts, CUT_MARKER, fence CAS) → P3 all classes +
kill-9 crash/chaos matrix → only THEN voxels (P4 terrain, P5 physics, P6 blocks, P7
checkpoints, P8 ships, P9 signals, P10 warp, P11 combat). Transfers are proven robust
before features pile on — the structural inversion of the old project's failure.
