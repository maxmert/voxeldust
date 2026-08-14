# NPC AI — open-source solutions usable from Rust (research report, 2026-08-13)

**Status: research deliverable for an owner decision (standing law: no unilateral library adoption).
Nothing here has been added to any Cargo.toml.**

Method: a 18-agent research workflow (repo-constraint grounding + six parallel web sweeps → 67
deduplicated candidates → judge shortlist → nine source-level verifications with real measurements →
completeness critique), followed by a four-agent gap-closure workflow (FSM/statecharts, dialogue,
influence-maps/discrete-event scheduling, security advisories + fallback audit) and one decisive local
measurement of the coverage doctrine (§3). Every license was read from the LICENSE file, not an SPDX
guess; every determinism verdict on a finalist is grounded in fetched source, and five finalists were
**measured** (multi-process/multi-arch byte-identity probes that could have failed — twice they did).

---

## 1. The hole a library would fill (what our own designs already say)

The dormant-world design (`scripts/dormant_world_simulation_design.md` §6) already **specifies what an
NPC is** in this game, and it removes most of the naive "NPC AI library" surface:

- **The 99% needs no brain library at all.** Dormant NPC life is *evaluated, not simulated*: population
  is a closed-form integer cohort field; individuals are the field's quantiles
  (`NpcId = child_seed(realm_seed, NPC_SALT, index)`, no storage); an NPC's pose is a seed-derived
  integer waypoint-ring **ephemeris** `g(NpcId, tick)`. An NPC is "a moving realm occupant with an
  ephemeris, exactly like a planet" — the frame-authority law applies unchanged, crossing an SOI is the
  ordinary containment re-home.
- **Materialisation ≠ promotion.** NPCs inside a player's AoI band become real ECS entities (a rendering
  /interaction affordance, float-tolerant); an NPC's *future* leaves the closed form only on **promotion**,
  triggered strictly by a discrete player interaction. Promoted NPCs must **re-converge to the ephemeris**
  to be demoted (the TTL-demotion path was deleted as gate-breaking); promoted state is bounded per realm
  (`max_promoted_subjects`) and — v1 restriction — a promoted subject may not cross a realm boundary
  (ledgered D-80).
- **The economy decision deleted the market-agent use case.** Purely player-driven economy: no computer
  traders, no formula prices. `NeedsOnlyStrategy` is the only NPC strategy; "prices make NPCs smarter,
  never alive."

So the actual library question is **narrow**: (a) decision cores for **promoted/materialised** NPCs while
live on a shard, (b) **navigation/steering** for materialised NPCs, (c) optionally a **scripting/sandbox
runtime** for authored brains, (d) architecture references for the dormant substrate we build ourselves.
No NPC exists in code today; the first durable non-session entity is planned as a shop at P6.

## 2. The constraint frame every candidate was scored against

Grounded from CLAUDE.md, PLAN.md, DEFERRED.md, the dormant/economy designs, and the workspace manifests:

| # | Constraint | Consequence for a candidate |
|---|---|---|
| C1 | No unilateral adoption | this report ends in options, the owner decides |
| C2 | Determinism + `sim::io` seam (clippy-enforced) | wall clock, unseeded RNG, threads, default-hasher HashMap on decision paths, or internal I/O ⇒ disqualified or fork |
| C3 | HR5 100% region+branch on Tier-A | resolved by measurement — see §3 |
| C4 | Server = bare `bevy_ecs` 0.18; full Bevy 0.18 client-only | full-facade or `bevy_app`-mandatory crates fail server-side; version skew = rejection ground |
| C5 | Crate-graph direction; economy one-way overlay | NPC life sits in vd-sim/vd-core and inherits Tier-A constraints |
| C6 | HR1 sealed shards, SL1/SL2/SL7 | AI needing global blackboards, cross-realm perception, world navmeshes, or own messaging is structurally forbidden |
| C7 | Dormant substrate: closed-form, integer, live==dormant | hidden accumulated agent state breaks the woken-equals-never-slept gates; see §7 |
| C8 | HR3/HR4 one tooling, G-IDENTICAL on ≥2 shard kinds | no planet-only or station-only AI stack; see §8 |
| C9 | No magic numbers; per-entity seed-derived params | baked-in tuning constants fail a binding convention |
| C10 | SL5 one world; tests drive the shipped path | no "simple test AI"; whatever ships runs in every e2e scenario, so per-tick cost matters |
| C11 | License posture | MIT/Apache/BSD/Zlib adoptable, MPL with care; GPL/AGPL = ideas only |
| C12 | Maintenance/pinning/supply chain | exact pins on determinism-critical deps; archived upstream = rejection ground; never-assume ⇒ measure |

## 3. NEW MEASUREMENT — the Tier-A coverage doctrine for external code

The critique of the first workflow pass found a direct contradiction: the constraint grounding claimed
dep generics monomorphizing into Tier-A crates enter the 100% gate (and the petgraph rejection partly
rested on that); three verifications asserted the opposite. Nobody had measured it. **Measured now**, in
an isolated scratch workspace using the exact `justfile` invocation
(`cargo +nightly-2026-06-06 llvm-cov --branch -p <crate> --ignore-filename-regex '(/bin/|/tests/)'
--fail-under-regions 100`):

- A probe crate called `pathfinding::astar` (generic, instantiated on our types, with the dep's internal
  "goal unreachable" handling never exercised) **and** derived a `thiserror::Error` enum with one variant
  never constructed (its macro-generated Display arm never executed).
- **Result: the report contains ONLY the probe crate's own `lib.rs`.** The dep's generic internals
  contributed **zero** regions. The proc-macro-generated code contributed **zero** missed regions or
  functions (5/5 functions executed). The filter boundary is *workspace-root containment* — the dep sat
  outside the probe workspace root and was excluded exactly as a registry dep would be.
- The measurement proved its own sensitivity: the gate **did fail (exit 1)** — on one genuinely uncovered
  region in *our* probe file (the `?` operator's None arm we deliberately never exercised).

**Doctrine (measured, not argued):**
1. An external **dependency** costs zero HR5 burden, no matter how generic-heavy — its regions never
   enter the Tier-A report.
2. **Vendored** code is inside the workspace root and enters the gate at 100% region+branch, every
   monomorphization — vendoring a framework crate is expensive under HR5 and usually demands
   restructuring to the branchless-shim discipline.
3. The SPIKE-0a/P0.3 lesson ("branch inside a generic fn must be covered per instantiation") remains
   true **for our own code**; its extension to registry deps in the C3 constraint text was wrong.
4. The petgraph rejection's *outcome* stands on its other grounds (determinism audit surface, supply
   chain, in-house preference), but its stated coverage rationale does not survive this measurement.

This flips the default adoption shape everywhere below: **pin as an external dep where the crate is
usable as-is; vendor only when source changes are unavoidable** (then pay the HR5 restructuring cost).

## 4. The field (67 verified candidates, six lanes) — summary

- **Behavior trees**: bonsai-bt is the clear engine-agnostic leader (MIT, alive, externally dt-ticked).
  All BehaviorTree.CPP-style ports are tokio/async-coupled (C2-fatal). Bevy-native plugins (bevy_behave,
  bevior_tree, beet_flow) need the facade or bevy_app (C4-fatal server-side). No maintained BT editor
  tooling exists in the ecosystem.
- **Utility / GOAP / HTN**: the Bevy AI ecosystem chronically lags Bevy releases; the only planning crate
  verified on bevy_ecs 0.18 sub-crates is bevy_bae (HTN). big-brain (already rejected in the economy
  research) remains archived-at-0.15. Engine-free cores exist (Emergent, whim, dogoap-core,
  reliakit-decide) but are one-person projects; HTN is where the ecosystem's current energy is (two Bevy
  maintainers independently started HTN planners in late 2025).
- **Navigation/crowds**: the ecosystem consolidated on an unbundled stack — rerecast (Recast-port navmesh
  generation, engine-free core, measured deterministic) → landmass (path+steer+ORCA suite, engine-free
  core) or polyanya; dodgy_2d is effectively the only ORCA implementation (and is landmass's engine).
  **Voxel-native gap**: no mature volumetric/3D-voxel pathfinding crate exists; the practical route is
  the `pathfinding` crate over a project-owned voxel graph plus an owned hierarchy layer (bevy_northstar
  is the only living hierarchical 3D-grid implementation, but is facade-bound as shipped).
- **Scripting/sandbox**: two tiers. Only the **WebAssembly route can guarantee** bit-reproducible
  execution across hosts (wasmi, wasmtime, wasmer — fuel metering, NaN canonicalization, no ambient
  imports). Lua/mlua and the pure-Rust languages are "deterministic with discipline" at best (mlua:
  per-state randomized table-iteration order is a verified sharp edge; Koto's only budget is wall-clock —
  disqualified; Steel has no budget primitives).
- **Games to learn from**: every substantial Rust game with real NPC AI is copyleft (Veloren, Egregoria,
  Sulis, citybound, mk48 — GPL/AGPL, ideas only). The two most valuable reads are Veloren's rtsim
  (dual-representation dormant NPCs) and A/B Street (Apache-2.0; discrete-event "agents sleep until their
  next event" scheduling) — see §10.
- **ABM / LLM**: no drop-in coarse-simulation layer for thousands of offline NPCs exists (krABMaga wants
  to own the loop; EpiRust is AGPL+archived but proves millions of coarse agents is easy in Rust). No
  purpose-built NPC-dialogue LLM framework exists in Rust; inference is out-of-band flavor only
  (~0.5–2 s GPU per 100-token reply from a 3–8B model — per-player-conversation only, never
  per-dormant-NPC, never authoritative state).

## 5. Verified finalists — nine source-level verifications

| Candidate | Lane | License | Verdict | One-line reason |
|---|---|---|---|---|
| **bonsai-bt** =0.13.0 | behavior tree | MIT | **strong-fit** | zero default deps (measured via cargo tree); tick is a pure function of tree state + caller dt; alive |
| **pathfinding** =4.15.0 | graph/voxel A* | Apache-2.0/MIT | **strong-fit** | byte-identical paths measured across processes; Ord cost type makes floats unrepresentable; ban-list for its HashMap-based sibling APIs |
| **wasmi** =2.0.0-beta.10 | wasm brain sandbox | MIT/Apache-2.0 | **strong-fit** | determinism measured across processes, arches, dispatch modes; fuel = resumable per-tick budgets; guest world = exactly what we register |
| **landmass + dodgy_2d** | navmesh nav + ORCA | MIT/Apache-2.0 | **conditional** | measured route-level divergence on multi-island meshes (std HashSet iteration); adoption requires a determinism fork |
| **rerecast** =0.3.2 | navmesh generation | MIT/Apache-2.0 | **strong-fit** | byte-identical navmeshes measured across runs and arches (libm feature); engine-free no_std core; wrap its panics |
| **Emergent** =1.9.0 | FSM/utility/GOAP composition | MIT/Apache-2.0 | **conditional** | FSM/BT lane deterministic as shipped; GOAP lane measured nondeterministic (11 distinct plans in 12 runs) and unfixable without fork |
| **bevy_bae** 0.1.0 | HTN on bevy_ecs 0.18 | MIT/Apache-2.0 | **conditional** | planner core deterministic and exactly our ECS version; but bevy_app-only wiring, an unfixed despawn panic, a system leak, no cycle guard — fork/vendor to use |
| **whim** 1.0.0-beta.1 | HTN kernel (vendor seed) | MIT | **conditional** | published crate panics on first run (measured); 199-line core is sound after a one-line fix; vendor-and-own only |
| **bevy_northstar** 0.6.2 | hierarchical 3D-grid HPA* | MIT | **conditional** | determinism audit clean, but hard-requires the bevy facade incl. bevy_render; HPA* core is mechanically extractable |

Key per-pick facts an adoption decision needs (all source-verified; "measured" = a probe that could have
failed):

**bonsai-bt** — the only Instant user is a dead-code Timer helper off the tick path; visualize-gated
telemetry is the only threading; blackboard type is fully caller-supplied (we supply BTreeMap/DetHashMap).
Loop semantics: tick returns None after terminal until `reset_bt()` (blackboard survives). Bus factor 1,
mitigated by a ~1400-line vendorable core. As an external dep: zero HR5 burden (§3).

**pathfinding** — measured byte-identical astar/dijkstra across two processes on a tie-heavy 24³ voxel
grid (SHA-256 equal), and the control fired: `astar_bag_collect` returned the same path set in a
*different order* across processes (RandomState leak). Adoption guardrails: allow only
astar/dijkstra/bfs/fringe/idastar/dijkstra_reach; clippy-ban the HashMap/HashSet-based siblings
(astar_bag, dijkstra_all, yen, components…). Searches are blocking run-to-completion — bound search
extent per tick (no step-budget API; issue #775). Tie-breaking is deterministic but not
documented-stable: re-run the byte-identity fixture on any version bump. Carry the license texts
ourselves (no LICENSE file has ever existed in repo or crate — manifest declaration is the grant).

**wasmi** — the only lane that upgrades determinism from discipline to **guarantee**: measured
byte-identical execution (exact fuel-exhaustion iteration, identical f32 accumulator bits) across
processes, arm64-vs-x86_64, and both dispatch modes; NaN canonicalization verified in source and
measured; a module importing WASI `random_get` fails instantiation. Measured overhead: 56–66 ns per
host→guest call (4096 brains/tick ≈ 0.23 ms), ~75 KiB per instance, ~4 µs instantiation. Conditions:
exact-pin (fuel points differ across versions — measured 1.1.0 vs 2.0-beta: a version bump is a
determinism event; keep the fuel gate); CompilationMode::Eager; disable `wat`; no instance hibernation
yet (transfer brains at call boundaries or snapshot memory+globals ourselves — moot for v1 under D-80).
Bus factor 1, offset by Soroban/Stellar consensus-critical adoption.

**landmass + dodgy_2d** — the negative result that saves us later pain: single-island nav measured
bit-identical across 5 processes, but a 3-island/8-agent scenario produced **12 distinct trajectory sets
in 12 runs** (meters-scale divergence, velocity sign flips), root-caused to std HashSet iteration in
boundary-link creation and A* off-mesh-link expansion — and planet nav will inevitably be multi-island.
dodgy_2d additionally calls `rand::random()` inside ORCA when two agents' velocities exactly coincide.
Both fixes are small and upstreamable (deterministic hasher in ~8 files; deterministic tiebreak
direction + drop the rand dep), but adoption = owning a fork, and output-side integer quantization
cannot repair it (divergence is route-level, not ulp-level). Post-fork cross-arch determinism of the
float-heavy ORCA/funnel math is **unmeasured**.

**rerecast** — full generation pipeline measured byte-identical (SHA-256) across runs and across
arm64/x86_64 binaries with the shipped `libm` feature; core crate is no_std, bevy-free, zero
HashMap/threads/time/rand; feeds voxel-surface TriMeshes directly (no physics-collider extraction);
output is landmass's and polyanya's input format, so the layer above stays swappable. Costs: ~11 ms per
32 m chunk (single-threaded, off-tick under our own scheduler); no incremental/tile regen yet; two open
generation-robustness panics — wrap and fuzz. Pin =0.3.2 (glam 0.30 line, matches workspace).

**Emergent** — FSM (Machinery), Sequencer/Selector/Parallelizer are Vec-ordered and deterministic as
shipped; the Reasoner is salvageable via its selector seam (caller-supplied total order + NaN guard);
the GOAP planner is **not** (HashMap hard-typed into its public API; measured 11 distinct equal-cost
plans in 12 runs; stale-priority open-list defect; NaN-unwrap panic). Zero dependencies, engine-free,
caller-driven. Use only the deterministic lanes; treat the planner as reference.

**bevy_bae** — the one crate exactly on bevy_ecs 0.18 sub-crates: BTreeMap blackboard, Vec-ordered
decomposition, defined MTR tie-break, synchronous exclusive-world execution — the HTN *shape* is right.
But as-published: bevy_app is the only public wiring path; a known unfixed panic when an operator
despawns another Plan entity mid-execution (PR #8 unmerged 9 months — fatal where NPCs despawn NPCs); a
registered-system leak under compound-task churn; no cycle guard or replan budget. Repo idle 6.5 months
while the author is daily-active elsewhere. Usable only as a vendored fork (~1.9k lines, well-tested,
dual-licensed) or as the design template for an in-house planner — its branching already lives in
monomorphic systems, i.e. it is HR5-shaped.

**whim** — published crate panics unconditionally on first `Planner::run()` (measured; index out of
bounds — the shipped doctest never executes). After a one-line fix the 199-line lazy-lookahead HTN core
was measured to work, and it is the only planner in the ecosystem *designed* for cross-tick time-slicing.
Vendor-and-own seed only: swap its Estr interning keys (raw pointers into a process-global cache —
breaks checkpointing) for our own key + BTreeMap, drop the vestigial hashbrown dep, and review the
author's July-2026 `wip` rewrite first. The author (a Bevy maintainer) treats it as a sketch.

**bevy_northstar** — determinism audit clean (all Instant behind the off-by-default `stats` feature,
u32 integer costs, fixed-seed hashing, a genuine serial build path). But it hard-requires the monolithic
bevy facade with bevy_render (an unconditional debug module is why), and its grid is a dense bounded
Array3 — wrong as-is for sparse planet-scale worlds. The HPA* core (chunk/graph/astar/hpa modules) is
algorithmically self-contained and extractable; the de-facade PR (#26) has sat unreviewed since Aug 2025.
Vendor-extract if/when we want hierarchical grid nav; per-agent NavMask layers fit the per-entity-params
law.

## 6. What was dropped and why (highlights)

Full rationales are preserved in the workflow output; the ones worth remembering:

- **bevy_behave** (maintained, MIT, bevy 0.18): needs the bevy facade/bevy_app — same C4 ground that
  rejected big-brain. Revisit only for client-side or if C4 is relaxed for headless bevy_app.
- **big-brain**: already formally rejected (archived upstream, bevy 0.15/0.17); its Codeberg migration
  is easy to misread — the canonical repo showed no commits since migration day. Nothing changed.
- **GOAP crates** (dogoap/dogsoap/goap/goap-ai): the shortlist's "covered by Emergent" rationale was
  invalidated when Emergent's GOAP lane failed verification. Honest current state: **there is no viable
  GOAP adoption path in the ecosystem**; the prior decision already routes GOAP to "reserve for
  station/logistics NPCs", and dogoap-core (engine-free, dormant 15 months, bevy_reflect-0.16-pinned at
  its bevy layer) is the vendor seed if a data-driven GOAP core is ever wanted.
- **polyanya**: subsumed by the landmass pick; carries an unaudited hashbrown per-process-seed
  iteration-order concern + float tie-breaking — flagged as an **unaudited fallback** (gap sweep §12
  covers it).
- **mlua** (historically user-approved for robot programming, not a dep in new-system): for NPC brains
  it is the weaker sandbox — verified per-state randomized `pairs()` order, curated-stdlib burden,
  unverified fixed-seed build. The Luau feature (sandbox + interrupt + memory limits) is the mitigation
  set if Lua authoring ergonomics ever outweigh wasm's guarantee. Wasm and Lua can also coexist:
  first-party brains under wasmi, mlua kept for its original in-game robot-programming role.
- **wasmtime/wasmer/extism**: guarantee-capable but redundant while wasmi holds the lane; wasmtime is
  the perf escape hatch (mirrors wasmi's API), and any engine swap is a determinism event by our own
  cross-version fuel measurement.
- **krABMaga**: wants to own the loop (own scheduler, macros, global statics, timing-dependent parallel
  mode) — conflicts with step_tick sealed shards. Pattern reference only.
- **Koto** (wall-clock-only budget) and **Steel** (no budget primitives): C2-disqualified as shipped.
- **piccolo**: architecturally the dream (fuel-stepped per-agent Lua executors) but stalled 13 months,
  stdlib incomplete — watchlist.

## 7. Substrate fit — the C7 reframe the first pass missed

The dormant design implies the decision core's real requirement is **"resumable from (seed, tick),
re-convergence-checkable"** — not "best planner". Scoring the finalists on the three substrate axes:

| Pick | State reconstructible from (seed, tick)? | Cheap re-convergence predicate? | Breaks woken-equals-never-slept? |
|---|---|---|---|
| bonsai-bt | NO — per-node State<A> accumulates (Wait timers, running indices) | only by convention (design brains to re-derive targets from the ephemeris) | if used for cohort NPCs, yes; **promoted-only** |
| Emergent (FSM/utility lanes) | YES if states are pure functions of memory M, and M is rebuilt per tick from world + seed — stateless-recompute is its natural idiom | yes — utility scores recomputed per call | no, when used stateless |
| bevy_bae / whim (HTN) | NO — plan stacks/MTR are accumulated state | plan-complete boundaries give natural re-convergence points | promoted-only |
| wasmi brains | NO — linear memory is arbitrary state | only by authored contract (brain must expose "am I back on routine?") | promoted-only; memory snapshot IS the checkpoint |
| pathfinding / rerecast / landmass | n/a — stateless queries / build artifacts | n/a | no (recomputable) |

Consequences:
1. **Cohort NPCs (the 99%) use none of these crates.** Their "AI" is the ephemeris + activity-mix
   integer categorical draw — already specified, already ours.
2. **Every stateful brain paradigm is promoted-NPC-only**, and the promoted population is bounded by
   player-hours (`max_promoted_subjects`), so brain cost and brain statefulness are both capped by
   design. The brain's job description literally includes "walk back to your closed form" (demotion
   requires re-convergence while unobserved) — a **utility/FSM stateless-recompute core scores better
   here than deep plan stacks**, because its "state" is re-derivable and its re-convergence predicate
   ("routine behavior selected and pose within one tick of ephemeris") is trivial.
3. If wasm brains land, the demotion predicate must be part of the **brain ABI** (an exported
   "converged?" query), not something the host infers from opaque linear memory.

## 8. Seam mapping — SL4, HR4, and the transfer question

**SL4 (physics vs re-home, one-way).** The lawful shape: an NPC brain is an ordinary *signal producer* —
its output enters the containing realm's physics authority exactly like player input (last-wins signal
integration → pose authored and shipped). Concretely: brain emits intent (desired velocity / next
waypoint) → quantized to the integer grid at the physics→control boundary (C7) → the realm's physics
capability integrates it. Under this shape: landmass's `get_desired_velocity()` output is exactly an
intent signal (right shape); bonsai/HTN operators must terminate in intent emission, never direct pose
writes; ORCA local avoidance is part of intent *formation* (AI side), not the physics integrator — it
consumes only realm-local occupant state, which the shard legitimately holds (C6-clean, nothing crosses).
No orbit/gravity/thrust symbol appears anywhere in this path, and crossing code cannot tell an NPC from
a player or a rock — SL4 holds structurally.

**HR4 (features once, run anywhere).** Nav backends must be a **capability keyed by realm data, not by
shard kind**: a realm that *has* navmesh surfaces (station/ship interiors) gets navmesh+ORCA; a realm
whose walkable space is the voxel field gets voxel-graph A*; both produce the same intent-signal type.
No `match` on shard kind anywhere in the movement path. The G-IDENTICAL fixture an adopted stack must
pass: the same NPC fixture walks A→B on two shard kinds (station interior and planet surface) and the
observer feed asserts identical arrival semantics.

**HR2/D-31 (brain state crossing shards).** Per pick: bonsai has an off-by-default serde feature (whole
tree state); Emergent brains are `Box<dyn>` graphs — not serializable; wasmi has no instance hibernation
(snapshot memory+globals ourselves, or transfer at call boundaries); bevy_bae plans are entity graphs
(assessed: no serialization story); whim's published keys are process-local pointers (fixed by the
vendor swap). **All of this is moot for v1**: the dormant design's D-80 restriction means a promoted
subject may not cross a realm boundary, so no live brain ever transfers before D-80 lands (P8+
timeframe). Note for the owner: **D-80 currently exists only in the dormant design's own ledger, not in
the binding `docs/design/DEFERRED.md` registry** — worth registering so the binding ledger stays
authoritative.

## 9. Ownership-cost totals (the comparison the picks quietly imply)

Four of nine finalists resolved to fork-or-vendor outcomes. Priced honestly, next to the in-house
alternative the project's precedent prefers:

| Pick | Shape | Owned lines (approx) | Work at adoption | Upstream expectation |
|---|---|---|---|---|
| bonsai-bt | external pin | 0 | reset_bt wrapper; ignore Timer | healthy; vendorable escape (~1.4k lines) |
| pathfinding | external pin | 0 | clippy ban-list + byte-identity fixture | healthy; vendor escape ~150 logic lines |
| wasmi | external pin | 0 (+ our tick ABI + fuel gate) | eager compile, wat off, fuel gate in CI | healthy; wasmtime swap path |
| rerecast | external pin | 0 | panic containment + Linux-CI byte gate | healthy; tiny frozen core |
| landmass+dodgy | **mandatory fork** | ~8 files patched, fork owned permanently | det-hasher swap + rand removal + cross-arch measurement | single maintainer; upstream PRs plausible |
| bevy_bae | **vendor/fork** | ~1.9k lines | fix 3 defects, de-facade, macro swap | effectively none — treat as ours |
| whim | **vendor seed** | ~200 lines | 1-line fix, key swap, tests | none — author rewriting anyway |
| Emergent | pin, partial | 0 (GOAP lane unused) | custom Reasoner selector + NaN guard | none needed for the lanes used |
| bevy_northstar | **vendor-extract** (only if wanted) | HPA* core modules | de-facade, paged grids | tracks Bevy fast but ignores headless asks |

Aggregate if everything conditional were adopted: roughly **2.5–4k owned lines** under full HR5
discipline (vendored code enters the 100% gate, §3) plus one permanent nav fork. The in-house
equivalent for the *decision* layer (utility scorer + tiny HTN in the whim shape + FSM enums) is
plausibly the same order of magnitude — which is exactly why the strong-fit externals (zero owned
lines) and the decision-layer build-vs-vendor choice should be decided separately.

## 10. Architecture references (ideas only — GPL/AGPL or unadoptable)

- **Veloren rtsim** (GPL-3): *the* existence proof for our dormant pillar. One slotmap record per NPC
  with `SimulationMode::{Simulated, Loaded}`; **one composable Action-trait brain runs in both modes**
  (simulated mode throttled ~10:1, moved by an abstract straight-line rule; loaded mode a full ECS
  entity); a thin per-tick sync seam copies pos/health back; outbox/inbox messaging. What we must do
  differently: it is a single process over the whole world with weak determinism (rayon, default-hasher
  HashSet, ad-hoc ChaChaRng) — ours shards per-realm and hardens to C2; and our dormant tier is
  closed-form, not throttled-stateful, per LAW-WL-1.
- **A/B Street sim crate** (Apache-2.0 but unpublished/coupled): the discrete-event priority-queue
  pattern — every agent sleeps until its next scheduled state transition — is the cheap way to advance
  a 99%-off galaxy, and its documented event-storm congestion failure mode is a ready-made load test
  (load-tests law).
- **Egregoria** (GPL-3): the best worked example of our exact constraint set — agent AI inside a pure
  deterministic fixed-tick core, all mutation through one WorldCommand seam, stable iteration order as
  a stated project rule. Read the "souls" needs-driven loop and the command-seam discipline.
- **Zemeroth** (Apache-2.0, dormant): ten-minute read; confirms the "AI emits the same commands players
  do" seam.
- **mk48.io** (AGPL): one idea — server-side bots as first-class fake players exercising the exact
  production entity/authority/transfer path (test-exactly-production law; the natural NPC-ship load
  test).
- **Causafera** (AGPL, pre-alpha essay): per-domain named random streams; append-only causal event log;
  bounded-perception believability.
- **Sulis** (GPL-3): engine-core/scripted-content boundary for AI — the pattern to weigh if authored
  brains land (mapped onto wasm rather than Lua).

## 11. Scripting/sandbox lane — conclusion

Only the wasm route **guarantees** reproducible execution across hosts (fuel metering deterministic per
engine version, NaN canonicalization, zero ambient imports); everything else is discipline. wasmi is the
verified pick (interpreter; consensus-proven; measured here); wasmtime is the later perf escape with the
same API shape. The cheap architecture is **one instance per brain archetype + per-NPC state blob per
tick** (sidesteps per-instance memory floors), or per-brain instances at ~75 KiB each if isolation is
wanted. A tick ABI + guest toolchain (Rust-to-wasm brains first; any language later) is the real
adoption cost. mlua remains available for its already-approved role (in-game robot programming) — the
two do not compete.

Whether NPC brains should be *scripted at all* (vs compiled-in Rust behaviors) is a design decision that
is *not* forced now: nothing in P6's first shop NPC needs a sandbox. The lane matters the moment
player-authored or designer-authored brains are wanted.

## 12. Gap-closure sweeps

*(four targeted sweeps run after the completeness critique; results below)*

### 12.1 FSM / statechart crates

**Category verdict: no FSM crate beats hand-rolled Rust enums + exhaustive `match` under our
constraints — proven with specifics, not assumed.**

- **HR5 economics**: the two credible macro/DSL options provably inject structurally-dead wildcard arms
  into *our* files (proc-macro output attributes to the invocation site — and vendored/our-file code
  enters the gate per §3). Source-read: statig's macro emits five dead `_ =>` arms per machine
  (`macro/src/codegen.rs:267-271, 442-444`); rust-fsm's DSL emits `_ => None` in both generated
  `transition()` and `output()`. The `machine` crate auto-adds an Error state per machine. Hand-rolled
  exhaustive matches have zero dead arms by construction.
- **HR2/serialization**: seldom_state's `StateMachine` is boxed closures/systems — explicitly not
  serializable (their own Cargo.toml comment); statig's wrapper embeds storage. Plain data enums as
  components are trivially TLV-framed.
- **Determinism**: every live candidate is a pure transition engine (no time/rng found in statig,
  rust-fsm, seldom_state core) — so no crate buys determinism we don't already own.
- **Engine coupling**: seldom_state (0.16.0, 2026-04-02, MIT/Apache — the only per-entity
  bevy_ecs-0.18-native option, actively tracking Bevy) drags bevy_app + bevy_log into vd-sim and churns
  archetypes on every state change; statig's bevy feature implements the pre-0.15 Component API against
  bevy_ecs 0.12 (dead against 0.18); bevy_state is app-global, categorically the wrong shape for
  thousands of per-entity brains.
- **What a crate would actually save**: statig in `no_macro` mode (MIT, no_std, zero mandatory deps,
  4.5M downloads) supplies just the HSM traversal driver — worth ~200 lines. **Conditional exception**:
  if 3+-level hierarchical semantics with entry/exit choreography ever become a real NPC requirement,
  statig `no_macro` is the one crate to revisit. **Harvest as design reference** (licenses permit):
  seldom_state's ordered flat transition list (deterministic priority) and its states-as-components
  pattern.

### 12.2 Dialogue systems (non-LLM)

**Two credible, permissive, actively maintained v1 routes exist — no build-it-yourself case, no GPL
contamination.** Both engine-free at the core, both need a thin determinism wrapper; the deciding
question is architectural, not ecosystem:

- **Route A — YarnSpinner-Rust** (`yarnspinner` 0.9.0, 2026-08-07, MIT/Apache; official YarnSpinnerTool
  org, maintained by janhohenheim): best authoring ecosystem (.yarn language, pure-Rust compiler,
  VS Code tooling); `Dialogue` is compile-asserted Send+Sync — drops straight into bevy_ecs 0.18 sim.
  Two verified defects: the four RNG built-ins draw OS entropy per call (**fixable via the public
  `library_mut()` API** — re-register them with seed/tick-derived impls; overwrite semantics verified in
  source), and there is **no public mid-dialogue state snapshot** (VM `State` is pub(crate)) — a
  conversation cannot checkpoint or cross a shard mid-node without game-level position modeling
  (persist variables + node id + choice history, restart deterministically). Note: non-optional
  `bevy_platform` 0.19 rides in the tree (std-abstraction crate, not the engine; fixed-seed foldhash
  maps).
- **Route B — blade-ink-rs** (`bladeink` 2.0.0, 2026-07-18, Apache-2.0; author of the Java runtime used
  in shipped commercial games; pure-Rust ink compiler exists): **the serialization champion** — public
  `save_state()/load_state()` round-trips the entire mid-conversation state (callstack, flows,
  variables, seed) as JSON — the natural fit for HR2 transfer and P7 checkpoints as an opaque Durable
  blob. All draws verified deterministic from `storySeed + previousRandom` (source-read). Two verified
  defects: `Story` is `!Send` (Rc/RefCell throughout — thread-confine per shard or reconstruct from
  JSON), and the initial seed comes from OS entropy with no public setter (pin via ink-side
  `SEED_RANDOM` or a state-JSON rewrite at construction; byte-stability of exported JSON unmeasured).
- Dead ends: inkling (dead 2020 + Parity whole-work copyleft — ideas only); bevy_talks (asset-coupled,
  niche format, bus factor 1); the rest of the registry is jam-grade.

**When due**: nothing before the first NPC-interaction feature (post-P4 content era). Until then keep
both viable — the only early commitment worth making is designing NPC interaction so the dialogue
engine sits behind a sim-seam trait with state as opaque bytes, which either engine satisfies. If
dialogue state must survive shard transfer/checkpoint mid-conversation (likely, given HR2/P7), bladeink
fits and yarnspinner does not; if conversations are short-lived and boundary-confined, yarnspinner's
ergonomics win.

### 12.3 Influence maps / Dijkstra maps + discrete-event scheduling

**Influence/Dijkstra maps — verdict: vendor the algorithm, not a crate** (~150–300 LOC in-house port
against our own grid types). Evidence: bracket-pathfinding (MIT) has the canonical RogueBasin multi-goal
DijkstraMap and its `dijkstra.rs` is cleanly extractable (source-read: Vec + VecDeque frontier,
insertion-deterministic, no RNG/HashMap) **but** is f32-throughout with `partial_cmp` sorting and no
tie-break (collides with the integer-grid law), stale since Oct 2022, and drags the bracket-* family;
sark_pathfinding (MIT, 2025) seeds its frontier from an ahash HashSet — unspecified iteration order
feeding the frontier is a real nondeterminism risk — plus a glam 0.32 conflict; our `pathfinding` pick
is integer-native and deterministic but strictly single-source (cannot produce a multi-source descent
field without N runs); bevy_flowfield_tiles_plugin is the only maintained flow-field crate but is a
Bevy-engine plugin (reference only — its sector/integration-field layering is the scaling pattern if
planet-surface crowds outgrow single-grid Dijkstra maps). A vendored port switches to integer costs +
explicit (cost, index) tie-breaking — exactly the shape the dormant-crowd substrate needs
(many-agents-few-goals).

**Deterministic discrete-event scheduling (the dormant wake-on-event kernel) — verdict: build in-house**
— a BinaryHeap of `(wake_tick, insertion_seq)` drained inside `step_tick` behind the sim::io seam
(~200 LOC, 100%-coverable). Every surveyed crate fails a hard constraint, measured: **nexosim** 1.0
(MIT/Apache, the only production-grade DES) is caller-driven (`step_until` would fit an external tick)
but runs a custom multi-threaded work-stealing executor with default worker count = num CPUs
(environment-dependent), its own docs concede same-time-slice ordering ambiguity, and **no
reproducibility claim exists anywhere in its README or docs** (searched: zero hits) — byte-identical
replay would rest on unverified behavior; harvest its ideas (next-event-increment stepping, deadlock
detection, save/restore of the scheduled queue). desim is GPL + nightly-coroutines (double
disqualification); sim/ndebuhr is wasm/npm-first stochastic DEVS; simrs is dead but its architecture is
precisely the ~200-LOC kernel we'd write, confirming the build cost; simulacra (2026, v0.1) independently
validates the design — priority queue + explicit insertion-order tie-break + "same seed ⇒ byte-identical
trace" backed by a run-twice test — but is network-domain and one release old; madsim (6.2M downloads)
is a whole replacement async runtime that owns the loop — architecturally wrong for an externally-ticked
schedule. This closes the one lane where the survey recommended building in-house without having checked
the ecosystem: the ecosystem is now checked, and in-house stands.

### 12.4 Security advisories + polyanya fallback audit

**Advisory pass** (measured via live OSV API queries — mirrors RUSTSEC + GitHub Advisory DB — with
affected ranges cross-checked against manifest requirements, 2026-08-13):

- **Clean, nothing to monitor**: bonsai-bt, pathfinding, landmass, dodgy_2d, rerecast, emergent, whim,
  bevy_northstar, mlua (Rust side), polyanya — zero advisories on themselves; every named transitive dep
  either advisory-free or resolving to a patched version today. One actionable note: dodgy_2d's rand dep
  is patched at fresh resolution (0.9.5) but a stale lockfile at rand 0.9.1/0.9.2 would carry a LOW
  advisory — `cargo update` if the lockfile predates rand 0.9.3.
- **wasmi**: 2 historical HIGH memory-safety CVEs (one from Dec 2025 — the risk class is live even in
  an interpreter); current releases unaffected. Advisory watch + prompt patching is a condition of
  adoption. (The advisory check evaluated the 1.1.0 stable line; the recommended =2.0.0-beta.10 pin is
  outside both advisories' affected ranges but should be re-checked at 2.0.0 final.)
- **wasmtime**: ~40 advisories 2021–2026 including **3 critical sandbox escapes** (one specific to
  aarch64 Cranelift — Apple-Silicon dev machines), clean only at the exactly-current release. Fixes are
  reliably backported across 4 LTS trains, so staying patched is feasible — but adopting wasmtime means
  cargo-audit in the gate + a rapid-patch policy. **The advisory asymmetry (wasmi 2 CVEs ever vs
  wasmtime 40, 3 critical) is a concrete security argument for the interpreter tier**, on top of the
  determinism argument.
- **mlua caveat (stated as a gap, not a finding)**: its vendored Lua C sources fall outside RUSTSEC —
  the C-side CVE stream needs its own watch if untrusted scripts ever run.

**polyanya determinism audit — PASS** (source-read on the published 0.16.1 tarball, cross-checked
against repo HEAD): zero rand/threads/clock on any behavior path (Instant is stats-gated; the async
wrapper advances a fixed 3 steps per poll); BinaryHeap ordering is `f32::total_cmp` on f with a g-value
tie-break — a **total order**, no NaN panic; every hash collection on the search path is keyed-lookup
only, never iterated, so the randomly-seeded default hasher cannot reach results; the one mesh-build
HashMap iteration is key-sorted before use. Bounded residual caveats: spade/bvh2d internals not
line-audited (no rand in their dep trees), and cross-architecture f32 bit-exactness not guaranteed
(same class as landmass; compatible with the per-host Category-C stance). **The fallback pathfinder is
no longer unaudited** — it stands as viable, and notably its total_cmp discipline is exactly the
float-tie-breaking shape our own laws demand.

## 13. Composed cost sanity check (back-of-envelope from measured components)

At the dormant design's promoted-subject bound (say 128 promoted NPCs on a busy realm shard):
128 wasm brain calls ≈ 7–8 µs of call overhead (56–66 ns each, measured) plus authored brain work under
fuel budgets; utility/FSM scoring is sub-µs per NPC; one pathfinding query on a bounded voxel region is
the only per-tick item that needs an extent bound (blocking, no step API); rerecast rebuilds (~11 ms per
32 m chunk, measured) live off-tick under our own scheduler; ORCA cost scales with local neighbor
density and is per-materialised-NPC only. Nothing here threatens the virtual-clock e2e budget at the
promoted cap — the D-9 fan-out wall (byte volume of many materialised entities to observers) remains the
binding scale limit, as already ledgered. **UNMEASURED as a composed whole**; per-component numbers are
measured.

## 14. Options for the owner (C1 — your decision, not an adoption)

The lanes are separable; each can be decided independently, none is due before its phase:

1. **Voxel/graph pathfinding — `pathfinding` =4.15.0 as external pin** (strong-fit, zero HR5 burden
   measured, byte-identity measured, guardrail ban-list specified). Due ~P6 when the first shop NPC
   walks. The voxel graph + hierarchy layer is ours regardless.
2. **Navmesh stack for interiors — rerecast =0.3.2 external pin (strong-fit, determinism measured) +
   a decision on landmass**: adopt-with-mandatory-fork (determinism fork specified, small,
   upstreamable) vs defer interiors nav and revisit. The fork is the only conditional here with
   permanent ownership attached to a moving upstream.
3. **Decision core for promoted NPCs** — three shapes:
   a. **In-house utility/FSM core** (stateless-recompute; best C7 fit; precedent-consistent; ~small,
      HR5-disciplined from birth), using big-brain/IAUS/Emergent as design references.
   b. **bonsai-bt external pin** for tree-shaped brains (strong-fit as a dep; per-node state makes it
      promoted-only and demands re-convergence-by-design).
   c. **HTN**: vendor whim's 200-line kernel or fork bevy_bae (~1.9k lines) — only if plan-shaped
      brains prove necessary; not before.
4. **Brain sandbox — wasmi =2.0.0-beta.10** when (and only when) authored/player brains become a goal;
   the determinism guarantee is measured and the fuel gate is reusable. No decision due now.
5. **Dormant substrate**: build in-house as designed (no adoptable alternative exists — verified);
   steal A/B Street's wake-on-event scheduling and Veloren's two-fidelity seam as patterns; mk48's
   bots-as-fake-players as the load-test idiom.

Recommended if you want a default: **1 + 2(rerecast now, landmass decision deferred until interiors
nav is actually scheduled) + 3a**, everything else parked with its trigger condition written down.

## 15. Open / unmeasured items

- landmass post-fork cross-arch determinism (float ORCA/funnel): UNMEASURED — required before the fork
  is trusted on Linux prod vs Mac dev.
- rerecast byte-identity on a real Linux x86_64 host (current cross-arch evidence is Rosetta on one
  Mac): pending CI measurement.
- pathfinding cross-arch: probably safe by construction (integer Ord costs, no floats on our surface) —
  **by construction, not measured**; one Linux CI run closes it.
- wasmtime-swap equivalence ("mirrors wasmi's API"): unverified claim; any engine swap is a determinism
  event per the cross-version fuel measurement.
- ollama-rs license: unverified (flagged; only relevant if the sidecar LLM shape is ever pursued).
- D-80 lives in the dormant design's ledger but not in the binding DEFERRED.md registry.
- C3 constraint text (and the petgraph precedent's stated rationale) should be corrected per the §3
  measurement: external-dep generics do NOT enter the Tier-A gate; vendored code does.
