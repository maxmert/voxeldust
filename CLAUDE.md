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

## Standing laws (owner-stated; SAME WEIGHT as HR1–HR6; breaking one is a defect)

Each of these was stated by the owner after a live defect. They are not preferences.

7. **SL1 — ONLY THE PARENT KNOWS POSITIONS.** Every realm is centred on itself. A realm never
   knows, stores, derives or is told its own position, velocity, orientation or spin — not even as
   a zero field, because a field's PRESENCE is the leak. A parent authors its direct children's
   placements in its own frame. Conversions ALWAYS happen in the parent: going down it subtracts
   before shipping; going up the child ships its own-frame pose and the parent adds. Never fold a
   realm's absolute from the root.
8. **SL2 — NO OCCUPANT POSE CROSSES A REALM BOUNDARY.** Not up, not down, not for culling, warming,
   or rendering. The realm you are INSIDE warms what comes next, from occupants it already holds;
   travel is always out into the shared parent and in again, never sibling-to-sibling. Liveness
   needs one bit per level: a live child keeps its parent alive.
9. **SL3 — A REALM DRAWS ITSELF.** The parent authors WHERE a realm is; the realm itself authors
   HOW IT LOOKS (extent, surface, detail, eventually meshes at a detail level it chooses). A
   parent's per-child message carries a PLACEMENT and nothing else, and must shrink toward that,
   never grow. A realm that is not running cannot be drawn — which is WHY visibility is the
   spin-up trigger.
10. **SL4 — PHYSICS AND RE-HOME ARE SEPARATE MACHINERY, one-way.** Physics (orbits, gravity,
    thrust, drag — a per-realm capability) produces A PLACEMENT: where each child is, in my frame,
    at this tick. Re-home/containment/hand-off CONSUMES placements and may never ask HOW a thing
    moves. No orbit/gravity/thrust symbol on the crossing path — a ship, a station, a moon and a
    rock cross by identical code because that code cannot tell them apart. A "does this child have
    orbital elements?" test inside a placement lookup is STILL a specific and is forbidden. A
    static flag may decide WHETHER TO RECOMPUTE, never WHAT ANYONE READS, and is derived from
    whether a child actually has motion — never declared per shard kind. Enforce structurally (a
    module/crate dependency rule), not by care.
11. **SL5 — ONE WORLD.** One universe, generated from the seed, used by the game AND every test
    including e2e. No scale knob, no preset, no reduced or test-only variant, no second generator.
    Numbers change on THE world; a variant is never created. (True astronomical scale + generated
    stations/areas + the galaxy cell lattice are owed changes to this one world.)
12. **SL7 — AN OCCUPIED REALM IS ITS OWN OCCUPANTS' PROXY, AT ITS PARENT'S SCALE.** How area-of-
    interest works at any nesting depth without a pose ever crossing, and without anything central:
    - LIVENESS, decided by the realm itself, looking only at itself and ONE level down: *do I hold
      occupants? do I have a live direct child?* Either ⇒ stay active. Neither ⇒ shut down after
      the cooldown. A realm never reaches upward and has zero control over its parent.
    - AREA OF INTEREST, decided by the PARENT, never by the realm about itself: for each of MY
      direct children, is it within the interest of an occupant I hold, or of an OCCUPIED CHILD
      treated at the placement I authored for it? Yes ⇒ demand it stays alive.
    - So an occupied child stands in for whoever is inside it. The error is bounded by the child's
      own size, which is exactly the resolution at which the parent's decision is meaningful —
      a child is small relative to its parent, which is what nesting means.
    - The parent also authors its children's VELOCITY, so warming-ahead needs nothing told to it.
    - What crosses: ONE BIT of occupancy, upward, which already crosses today (a sealed parent
      cannot see inside a child, so the child reports its own emptiness). Nothing else. No occupant
      pose, no entity set, no central assembly of a global picture, no depth or hop count anywhere.
13. **SL6 — ASK BEFORE NEW DATA CROSSES A REALM BOUNDARY, and before adding a wire arm.** Default
    NO. Find the local formulation first; there usually is one. State what data, from which realm
    to which, why the receiver cannot compute it from what it legitimately holds, and what doing
    without costs.

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
