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
- `docs/design/owner_decisions_*.md` — ★ THE BINDING RULINGS, and **the NEWEST FILE WINS**. Read the
  newest one BEFORE relaying any plan or design content: these are later than every design doc, and a plan
  agreeing with itself proves nothing. Newest first:
  - `owner_decisions_2026-08-27_galaxy_shape.md` — ★ THE GALAXY'S SHAPE, ITS DENSITY, AND HOW GALAXIES
    DIFFER. Written before S12 starts. **EVERY number the placement reads is SEED-DERIVED** — a density
    knob is not a tuning parameter, it is a lever that moves every star a player has ever seen (owner:
    *"otherwise any tiny change might change positions"*). The shape is BELIEVABLE and consumes the
    taxonomy's own galaxy-kind draw; a shell, a ring, a plain filled ball and a lattice-AS-POSITION are
    each refused by name (a lattice stays lawful as a LOOKUP). Placement is stable forever: grow the
    world and nothing already placed moves. ★ GALAXIES DIFFER in what you SEE and how you TRAVEL —
    shape, density, size, stellar population — and NEVER in how much they PAY, because a seed-derived
    rich galaxy is a treasure map and the seed ruling forbids it. **An expensive gate buys ROOM, not
    treasure:** a new galaxy is worth reaching because nobody has taken it yet, which is live state.
    S12 builds ONE galaxy properly; the between-galaxy differences are designed now and built when a
    second galaxy is reachable.
  - `owner_decisions_2026-08-27_movement_answers.md` — ★ THE ANSWERS TO THE MOVEMENT QUESTIONS. Three
    items closed. (1) THE UNGOVERNED ROCK: the containing realm owns the speed of a thing that governs
    nothing itself, **but a transfer NEVER changes that speed** — a destination realm's ceiling limits
    what it may ADD, never what a thing arrives with (a re-clamp is a jump, and a jump is a seam).
    (2) WAKING UP IN TIME: **grow the interest radius with the closing speed; never cap the speed** —
    a cap makes the game unfair, because the player would pay for a server's start-up cost.
    ★ (4) THE CEILING LEAVES THE FLIGHT PATH COMPLETELY — no clamp, no refusal, no stop; a player never
    STATES a speed (a throttle makes a force), so nothing there can be refused. A fence may stand ONLY
    where CODE states a speed (a spawn, a fixture, a tool), where a bad number is a defect and a test
    goes red. CONTAINMENT BECOMES SWEPT: the test reads the LINE from last tick to this tick, so nothing
    must be slowed to be caught. The band never protected the player — it protected a SNAPSHOT, which is
    why deleting the governor broke the bands. Slowing down is GAMEPLAY: a ship's own safety block slows
    it (and a player may switch that off), and air slows a rock. MEASURED: the galaxy's ceiling is
    ~170 million times the speed of light, so it never binds in open space and binds only near SMALL
    bodies — exactly where it felt wrong. ⇒ THE BAND RE-SOLVE DOES NOT RUN YET: fix the test first, and
    the bands shrink on their own.
    (3) THE INTERIM ENGINE RATING: yes, a ship may state its rated cruise speed and acceleration as
    facts about what it IS. ★ Plus THE TEMPORARY CONTROL SEAM: attach the controls to the ship realm,
    make forces directly in the ship's shard (no signals, no functional blocks, no hull), and send them
    up the lane the movement contract already defines — so the whole path is testable NOW. Still open:
    the band re-solve, the HR4 second shard kind, the wake-up constant.
  - `owner_decisions_2026-08-27_seed_and_secrecy.md` — ★ WHAT A SEED MAY DECIDE. Anything a fixed seed
    alone determines must be SAFE TO PUBLISH; anything VALUABLE must depend on world state that CHANGES.
    Binds the block system BEFORE it starts: a block's substance may not be a pure function of
    (position, seed). Secrecy is not the cure — one world + many players makes any static map public by
    wiki, and a shipped generator inverts to its seed. ⚠ Does NOT authorise client-side derivation;
    that collides with the client-only-renders law and is OPEN.
  - `owner_decisions_2026-08-26_movement.md` — ★ THE MOVEMENT CONTRACT. Two lanes upward: **acceleration
    and torque ONLY, per tick, in the child's own frame** (what I am DOING), and **mass + cross-section +
    drag coefficient ON CHANGE ONLY** (what I AM). **Velocity NEVER crosses upward** — it is half a
    placement, and only the parent writes those. Downward: the authored placement, stamped, read-only,
    one hop. The width-over-three-minutes speed rule is RETIRED for band-width ÷ tick. Gravity never
    needed a child's mass (it cancels); drag and collisions do. Still open: the ungoverned rock, the
    wake-up radius. **WARP IS A DECLARED STATE** with a FIVE-TEST GATE and a CLOSED, reviewed set —
    a declared state says what a ship IS, never what it wants; it must survive an empty ship; and it must
    not move you by itself. An autopilot is software pressing the stick and gets NO special path.
  - `owner_decisions_2026-08-26_s9.md` — the S9 walk (hold/derive/send, identity naming, the star cap).
  - `owner_decisions_2026-08-24.md` — the SL1 rewrite, the SL2 clarification, SL9, the movement law
    (a parent never sets a speed), and the nine Universe/Galaxy/sky answers.
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

7. **SL1 — A REALM IS TOLD WHERE IT IS; IT NEVER DECIDES WHERE IT IS.** (Rewritten by the owner
   2026-08-24, deliberately reversing the 2026-08-05 ruling that a realm may never be told its own
   placement — see the REVERSAL note below. The earlier text read "ONLY THE PARENT KNOWS POSITIONS"
   and forbade the datum outright, on the reasoning that a field's PRESENCE is the leak.)
   1. A parent authors its direct children's placements in its own frame. **The parent is the ONLY
      writer, always.** Conversions happen in the parent: going down it subtracts before shipping;
      going up the child ships its own-frame pose and the parent adds.
   2. A parent MAY state to a child the placement it authored for that child. The child holds it as
      a **stamped, read-only reading** — an instrument, never a fact it owns.
   3. A child NEVER derives, adjusts, computes or states its own placement — not to its parent, not
      to its children, not on any lane. A placement a child asserts about itself is refused.
   4. **ONE HOP ONLY.** What you are told about yourself, you NEVER pass on; and you are never told
      your parent's placement. **Never fold a realm's absolute from the root** — the chain does not
      exist anywhere a realm can reach it, which is what makes the iron rule structural rather than
      remembered.
   5. The PLACEMENT, CONTAINMENT and CROSSING machinery may not NAME a realm's own position.
      Enforced by a crate/module dependency rule with an observed-failing control — the same shape
      that fences motion off the crossing path (SL4), never by care.
   6. A STALE reading is REFUSED, never used. Every statement carries its instant; past its derived
      bound a consumer degrades and says so. Absent data stops an autopilot; stale data flies it
      into something.

   **WHY THE REVERSAL (owner, 2026-08-24).** The old law forced TWO mechanisms for one job: a screen
   was sent "the contents plus your own eye", a realm was sent "a view composed for you". Same job,
   two paths, two cost models — a fork decided by what KIND of thing receives, which is what these
   rules exist to prevent. Worse, the composed-per-observer form costs contacts × observers, and the
   sky proved that product fatal (150,000 rows per player per tick). Telling a thing where it is
   turns the product into a SUM for everyone. The clause given up — "a field's presence is the leak"
   — was never about correctness; it was about people. Clauses 4 and 5 replace that taboo with a
   fence the compiler and a test enforce, which is strictly stronger.
8. **SL2 — NO OCCUPANT POSE CROSSES A REALM BOUNDARY.** Not up, not down, not for culling, warming,
   or rendering. The realm you are INSIDE warms what comes next, from occupants it already holds;
   travel is always out into the shared parent and in again, never sibling-to-sibling. Liveness
   needs one bit per level: a live child keeps its parent alive.

   **CLARIFIED 2026-08-24 (owner) — this law governs REALM-TO-REALM, and the connection plane is not
   a realm.** A shard handing its OWN occupants to the gateway was never a crossing: it is how any
   player has ever seen anything. So SEEING another realm's people is lawful and needs no ruling —
   each shard streams its own occupants to its own subscribed clients, and the GATEWAY composes one
   ready-to-draw picture per observer (it alone holds both sides, and it is not a realm). What stays
   forbidden is unchanged: no occupant pose may enter another REALM's simulation. Hence the line:
   **CONTACTS ARE REALMS, PEOPLE ARE SEEN** — a ship's systems track ships, stations and bodies
   (placements the parent authors and may state), never individual people. You SEE someone in a
   spacesuit because the composer draws them; your targeting computer does not hold their position.
   Acting on them is a projectile CROSSING a boundary, which is the transfer machinery, not a leak.
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
14. **SL9 — A PARENT'S CHILD COUNT IS UNBOUNDED.** Owner-stated 2026-08-24. A realm may hold ANY
    number of direct children: six moons, six hundred ships and stations, or a galaxy's hundred and
    fifty thousand star systems. There is no cap, no reserved width and no "bounded child set"
    anywhere. EVERY mechanism that touches children must therefore be written for an unbounded
    count: never a fixed-width bitset over children, never a per-tick walk of all of them, never a
    per-child row on a per-tick lane. Membership is a SHORT LIST of the realms an occupant is
    inside (its own chain, plus the edge of a crossing) — never one bit per watched realm. Finding
    which child holds a point is a LOOKUP, never a scan; sibling realms may not overlap, so at most
    one child can hold a point and the answer is O(1)-ish, not O(children). A cost that grows with
    the number of children is a defect, and it must be measured on a realm with many, not argued.
    (This law does NOT relax SL7: the parent still decides area of interest for its direct children
    — it must simply do so without touching all of them.)

    *Note: this file has no SL8. The seamless law (SL8) was written in the `new-system` worktree and
    is not committed there yet; that number is reserved for it so the two trees do not disagree.*

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
