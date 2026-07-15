# Visual playground design + adversary (wf_19c81bef)

## SYNTHESIS

All load-bearing claims are confirmed against the code:
- `RealmBoundary`/`Boundary` fully `Copy + Serialize`, `membership_scalar`/`signed_distance`/`swept` all exist as the reusable box math (rendering-first's V1 and endgoal-multimesh's V3 both correctly reuse this — no duplicate branch logic needed).
- `world_pos` is the single one-level chokepoint; `ShipLocal` composes through `hull_pose`, and `hull_pose` today does NOT recurse (the 3-deep gap the plans flag is real).
- `RenderPose` is cell-free `{frame,pos,orient}`; `RenderSnapshot` holds `{view,clock,phase}` — adding a scene is additive.
- `DotEntities`/`DotAssets` is the single asset/registry seam; a sibling box path (not an extension) is correct per the binary-render rule.
- Tier-A set includes `vd-client` + `vd-client-harness`; `vd-client-render` is absent (Tier-B).
- `boundary_depth` exists shard-side as a monomorphic parent-walk (must be reimplemented Tier-A in the harness, as all three plans note).

I have everything needed. Producing the scoring and synthesis.

---

# SCORING

Scale 0-10 per criterion. **RF** = rendering-first (`RealmScene`), **TH** = test-harness-first (`BoundaryProvider`), **EM** = endgoal-multimesh (`RealmMesh`/`MeshPrim`).

| # | Criterion | RF | TH | EM |
|---|---|----|----|----|
| 1 | Renders realms-as-colored-transparent-boxes + multi-mesh correctly | 8 | 7 | 8 |
| 2 | Covers ALL transfer topologies as deterministic e2e captures | 8 | 9 | 9 |
| 3 | HR6 agent-operable (vdctl injection, wgpu readback, runs/ manifests) | 8 | 9 | 8 |
| 4 | 100% Tier-A on logic/harness | 9 | 9 | 8 |
| 5 | DRY / elegant | 9 | 7 | 8 |
| 6 | End-goal non-cornering (box→block-mesh, multi-mesh, signals/PvP/hundreds) | 8 | 8 | 10 |
| 7 | The box-geometry seam choice | 10 | 8 | 8 |
| **Wtd** | **(1,2,6,7 double-weighted — the load-bearing four)** | **8.5** | **8.0** | **8.7** |

**Rationale for the deciding criteria:**

- **Seam choice (crit 7).** All three land on A-now (dev-config `Vec<RealmBoundary>`) / B-later (a reviewed wire arm), correctly — boundaries are default-empty through P3, so there's no production producer to attach an arm to, and A adds zero cross-process bytes (HR1-inert). RF wins outright because it names the *actual* trap the other two skate past: whichever source feeds boxes, **the renderer must bind to neither**. RF interposes `RealmScene` (a `BTreeMap<RealmId, RealmBox>`) as a Tier-A projection that BOTH a dev-config loader and a future wire decoder produce — so A→B is a pure adapter swap with zero renderer churn. TH's `BoundaryProvider` trait achieves the same decoupling but returns `&[RealmBoundary]` (the raw core type), which leaks `to_realm`/`band`/`effect` — shard-authority-internal fields — into the render path; if B ships a render-only *projection* (as TH itself argues it must in §6), the provider's return type changes and the renderer's consumers move. RF's `RealmBox` is already that projection.

- **Non-cornering (crit 6).** EM wins: its `MeshPrim` render-primitive IR (no Bevy types, box arm emits cuboid+band-shells, future block arm emits greedy quads) is the *cleanest* box→block seam — the renderer consumes `Vec<MeshPrim>` and literally never learns the word "box," so P4 terrain is a `to_render_prims` change with zero render-crate churn. RF's `RealmBox { shape: BoxShape }` grows a `mesh` arm (box→chunk-grid) but the renderer still matches on shape today, so P4 touches it. Both correctly make boxes a NEW `RealmBoxEntities`/`RealmMeshEntities` path (not bolted onto `DotEntities`), honor the recursive-`world_pos` fix, and keep `RenderPose`/`EntitySnap` growable for D-35/D-41.

- **DRY (crit 5).** RF's `RealmScene` is the tightest single chokepoint. EM's `MeshPrim` IR is more future-proof but adds an indirection layer that's slightly heavier than the playground needs *now*. TH scatters the geometry across `boundary.rs` + `projection.rs` + a screen-space `project_point` verdict — the projection-into-screen-AABB approach is a real weakness: it makes the capture verdict depend on camera math, where RF/EM assert box-membership on the *world-space state* (the deterministic artifact) and treat pixels as corroborating. RF/EM's state-space verdict is the correct pattern per `capture.rs` ("captured STATE is the deterministic artifact; pixels structurally reproducible only").

- **Topology coverage + HR6 (crits 2,3).** TH and EM are marginally ahead of RF on granularity: both enumerate the through-and-back/hysteresis-no-spurious-commit case and both correctly sequence the recursive-`world_pos` fix as the thing the 3-deep scenario *forces* into existence. TH's mapping to the exact existing backends (`crossing_e2e` born-inside dwell, `crossing_nesting_e2e` physical inward walk `2000→1100→500→0`) is the most concrete.

**Verdict:** No design dominates. The synthesis takes **RF's `RealmScene` projection chokepoint and state-space membership verdict** (best seam + DRY + deterministic oracle), **EM's `MeshPrim` lowering IR and its explicit recursive-`world_pos`-forced-by-3-deep sequencing** (best non-cornering), and **TH's exact backend-reuse mapping and scenario granularity** (best topology enumeration). One reconciliation: RF's `RealmScene` becomes the source-agnostic chokepoint, and EM's `to_render_prims`/`MeshPrim` becomes how a `RealmBox` is *lowered for the renderer* — the projection and the IR compose rather than compete.

---

# SYNTHESIZED PLAN — Visual Crossing Playground

Legend: **[A]** Tier-A (`vd-client-harness`/`vd-client`/`vd-devproto`, 100% region+branch, `just coverage-fast`); **[B]** Tier-B (`vd-client-render`, coverage-exempt by set-absence, gate-proven); **[P]** process bin/test.

**Two architectural chokepoints, fixed for the whole plan:**
1. **Geometry seam:** dev-config `Vec<RealmBoundary>` → a Tier-A `RealmScene` (`BTreeMap<RealmId, RealmBox>`) projection → lowered to `Vec<MeshPrim>` (a Bevy-type-free render IR) → the Tier-B renderer. The renderer reads ONLY `MeshPrim`. Dev-config→scene and (future) wire→scene are both pure Tier-A adapters producing the identical `RealmScene`. **This is the final seam decision** (§ decision block below).
2. **Composition seam:** every box AND every dot composes through the ONE `DeliveredView::world_pos` chokepoint. The box for realm R is placed at `world_pos(RenderPose{frame: R's frame, pos: center.offset(), orient})`. No second composition path is ever added.

### THE BOX-GEOMETRY SEAM DECISION (final)
- **NOW:** dev-config. The harness/bin that plants `Vec<RealmBoundary>` into the shard's `RealmBoundaries` resource writes the *identical* Vec to a `boxes.json` handed to the client (`--realm-boxes <path>` at boot; `DevRequest::LoadRealmBoxes { boxes_json }` for late-bind autonomous scenarios, `as_input_action = None`). Single-sourced with the shard plant. Zero new wire byte → HR1 untouched.
- **PROJECTION LAYER (the non-cornering key):** a Tier-A `RealmScene` sits between source and renderer. `RealmBox { shape: BoxShape, center_offset: DVec3, parent: Option<RealmId>, depth: u8, color_seed: u64 }` where `BoxShape = Sphere{r} | Box{half, orient}` (Shell→Sphere, Aabb/Obb→Box). `RealmBox` carries ONLY render-relevant fields — `to_realm`/`band`/`effect` are dropped at projection (they're shard-authority-internal). The renderer binds to `RealmScene`/`MeshPrim`, never to `RealmBoundary` or the source.
- **LATER (ledgered, design-only now):** a reviewed closed `InterShardFlow`/`EventMsg`-family arm `BoundaryCatalog { boxes: Vec<RealmBoxWire> }`, reliable per-subscription, delivered when boundaries go dynamic (task #133 Station/Area → P8 ships). It produces the SAME `RealmScene`. HR1-sanctioned growth (one reviewed arm WITH its first consumer), not an escape hatch.
- **User confirmation gate:** the A-now/B-later split and the `RealmBox` projection field-set are the one architectural choice to confirm before Slice V2 (where the scene shape is first consumed). Slices V0–V1 are pure Tier-A scaffold and safe to build immediately.

### THE PER-REALM-MESH ABSTRACTION (box now / block-mesh later)
`RealmBox` is the render-agnostic description of one realm's extent. Its lowering `RealmBox::to_render_prims(&self, world_transform) -> Vec<MeshPrim>` (`MeshPrim { shape_or_verts, color_rgba, transform }`, no Bevy types) emits a translucent cuboid/sphere + optional band shells today. At P4+, `RealmBox` gains a `mesh: RealmMesh` arm (`Box(placeholder) | ChunkGrid(greedy-meshed)`) and `to_render_prims` emits greedy quads — **the renderer, consuming only `MeshPrim`, does not change** (this is EM's IR folded onto RF's projection). The box is a genuine placeholder REPLACED, not extended (binary-render rule). Color is `color_seed`-derived (golden-ratio hue of `RealmId`), a pure function of identity — never a `match` on realm KIND (HR3).

### THE MULTI-MESH COMPOSITING
N boxes + N dots render simultaneously in ONE Bevy scene. Each box entity in `RealmBoxEntities: BTreeMap<RealmId, Entity>` (sibling to `DotEntities`, distinct path). Nested child boxes sit inside parents because `center_offset` is parent-relative in P3's zero-cell frame; when hulls move (P8) they compose through `world_pos` identically to dots — no chokepoint reshape. `AlphaMode::Blend` + `cull_mode: None` gives proper translucent volumes with Bevy's back-to-front sort. Capture camera frames the union AABB of all boxes+dots (fit-AABB-to-frustum math is Tier-A `camera.rs`; the render crate only applies it).

---

## Ordered slices

### Slice V0 — `RealmScene` projection + `MeshPrim` lowering (pure, no render, no cluster) [A]
- **Crates/files:** NEW `crates/client-harness/src/realm_scene.rs` (`RealmBox`, `BoxShape`, `RealmScene`, `MeshPrim`, `from_boundaries(&[RealmBoundary]) -> RealmScene`, `to_render_prims`, `color_from_seed`); `boxes.json` serde (`to_json`/`from_json`); module wire-up in `crates/client-harness/src/lib.rs`. Depends only on `vd-core` (reuse `geometry::Boundary`) + `glam`.
- **Depth walk:** reimplement `boundary_depth` (mirrors `stub.rs:2614`) Tier-A as a monomorphic bounded parent-walk over the slice.
- **Tier split:** all **[A]**.
- **Scenario/capture:** none (pure). Unit tests over synthetic `RealmBoundary` vecs: Shell→Sphere, Aabb→Box, Obb→Box+orient carried, 3-deep parent chain → depth 0/1/2, missing/cyclic parent → bounded stop, round-trip serde, malformed json → `expect_err`. `to_render_prims` prim counts + transforms asserted with `assert_eq!`.
- **Coverage:** concrete types (generic-free); all branching in monomorphic helpers; `assert_eq!`/`expect_err` not `matches!`; split any `&&`. 100% region+branch.
- **Proves:** the source-agnostic geometry chokepoint + the box→block lowering IR exist and are fully covered before any GPU code — the seam that makes A→B and box→block-mesh non-cornering.

### Slice V1 — box-membership verdict (the deterministic capture oracle) [A]
- **Crates/files:** extend `crates/client-harness/src/assert.rs`: `innermost_realm_at(scene, world_p) -> Option<RealmId>` (reuse `vd_core::geometry::Boundary::membership_scalar <= 1.0` / `signed_distance <= 0`; deepest containing realm wins = nesting semantics); `entity_in_expected_box(scene, world_p, expected) -> bool`; batch `entities_in_expected_boxes(...)` for the load slice (order-independent reduction into a `BTreeMap`, `rayon`-ready).
- **Tier split:** all **[A]**.
- **Scenario/capture:** none (pure verdict). This is the STATE-space oracle every later capture asserts on; the PNG is corroborating evidence only.
- **Coverage:** point inside innermost of a 3-deep nest → deepest; parent-only region → parent; outside all → None; on-surface (`==`) → inside (matches core `<=`). Reuse proptested `vd-core` box math — zero duplicate branch logic. 100%.
- **Proves:** "entity landed in the CORRECT box after a crossing" is a deterministic pure-math verdict, not GPU-inferred.

### Slice V2 — publish `RealmScene` on the render seam + dev-config load [A]
- **Crates/files:** `crates/client/src/render_snapshot.rs` (`RenderSnapshot` gains `scene: Arc<RealmScene>`, `scene()` accessor, published via the SAME `ArcSwap` as poses — no second seam); `crates/client/src/net.rs` (`ClientCore` holds the loaded scene; `--realm-boxes` at boot; `DevRequest::LoadRealmBoxes` applies it, non-mutating config); `crates/devproto/src/dispatch.rs` (`LoadRealmBoxes { boxes_json: String }`, `as_input_action = None`, `is_mutating = false`).
- **Tier split:** all **[A]** (`vd-client`, `vd-devproto` are Tier-A).
- **Scenario/capture:** none yet (seam wiring). `RenderSnapshot::rendered()`/`location()` unaffected; `LoadRealmBoxes` serde round-trip.
- **Coverage:** scene carried on snapshot; `LoadRealmBoxes` round-trip + `as_input_action` None + `is_mutating` false. 100%.
- **Proves:** the renderer can read boxes lock-free through the existing snapshot seam; boxes are loadable at boot AND via the HR6 dev-control seam (config injection, honest to HR6).

### Slice V3 — translucent box render + capture-camera framing [B] (+ [A] framing math)
- **Crates/files:** `crates/client-render/src/lib.rs` (NEW `boxes` module): `RealmBoxAssets` (per-realm translucent `StandardMaterial { base_color: color_from_seed, alpha_mode: Blend, cull_mode: None }`), `RealmBoxEntities` registry, `sync_boxes` system (Update, in BOTH `run_windowed` and `run_capture`) consuming `scene().to_render_prims(world_transform)` and spawning/updating per-realm meshes via `world_pos`. Capture camera: NEW `CaptureCam` framing the union AABB, selected by `CameraMode` in `CaptureCfg`. Framing math `fit_camera(aabb, fov) -> (eye, look)` lives in **[A]** `crates/client-harness/src/camera.rs`; the render crate only applies it.
- **Tier split:** framing/hue/prim-selection **[A]**; Bevy mesh/material/spawn glue **[B]** (straight-line spawn loop, all branching already in V0's lowering).
- **Scenario/capture:** extend a `render_smoke`-style visual smoke — assert >1 distinct non-clear color region (boxes drew) + zero magenta.
- **Coverage:** framing/hue/selection 100% [A]; mesh glue exempt-by-set-absence, gate-proven.
- **Proves:** N colored transparent boxes + the dot render simultaneously (the multi-mesh end-goal made visible), framed by a capture camera.

### Slice V4 — FIRST VISIBLE SLICE: single-box walk-across-and-see (smallest e2e) [P] + [A] verdict
- **This is the deliberately small, visible first e2e.** In-process (NOT the dual-shard process cluster yet — that's V6). Reuse the born-inside dwell backend: `plant_one_crossing_shell` (`tests/src/lib.rs:493`, one `Shell{r:1000}` System(7)→System(8)), hand the SAME Vec to a `client --capture` via `--realm-boxes`.
- **Crates/files:** NEW `crates/bins/tests/crossing_visual.rs` (feature-gated `dev-control,render`); reuse `tests/src/lib.rs` plant helpers; `--realm-boxes` on `crates/bins/src/bin/client.rs`.
- **Sequence:** warm to delivered frame → **Screenshot BEFORE** (assert dot ∈ box System(7) via V1 `entity_in_expected_box`) → autonomous dwell commit (no `trigger_transfer` — anti-vacuity) → wait `live_sagas==0` + directory head flip → **Screenshot AFTER** (assert dot now ∈ box System(8)). `runs/` manifest records both shots paired with aligned `DevState`.
- **Tier split:** bring-up + wgpu **[B]**; BEFORE/AFTER verdict **[A]** (V1). One PNG decode (`image::open`) as in `render_smoke.rs:186`.
- **Coverage:** verdict Tier-A (V1); gate is a process test. Add `just crossing-visual` → `just gate`.
- **Proves:** the smallest complete loop — a visible marker crosses ONE rendered boundary and the deterministic state-verdict confirms it landed in the right box. Everything after this is added topologies.

### Slice V5 — scenarios (a) NESTED dock/undock + (c) THREE-DEEP + recursive `world_pos` fix [A] + [P]
- **Crates/files [A]:** `crates/client/src/view.rs` — make `hull_pose` resolve its own frame through `world_pos` recursively (the additive change EM/TH both flag; today it's one-level, confirmed at `view.rs:250`), with a **named depth-guard const** to bound cycles. `crates/bins/tests/crossing_visual.rs` scenarios; `tests/src/lib.rs` three-realm nested plant helper.
- **Scenario (a) `crossing_visual_nested`:** reuse `crossing_nesting_e2e` backend (child realm with `parent` set); physical inward walk `2000→1100(crosses 1150)→500→0` via vdctl `WalkTo`. BEFORE: dot ∈ parent box, outside child. AFTER dwell: dot ∈ child box. Reverse walk = the Slice-4b `outward_dest`/self-heal undock (assert re-home to parent). 3 shots (before/nested/undocked).
- **Scenario (c) `crossing_visual_three_deep`:** three boxes depth 0/1/2 (station ⊃ ship ⊃ player) via `parent` chain; assert `innermost_realm_at` returns the deepest and the player marker renders in the ship box which renders in the station box. This scenario is what FORCES the recursion to be correct — guarding the P8 corner today.
- **Tier split:** recursion + nesting/depth logic **[A]** (100%: 1-deep, 2-deep, 3-deep, cycle-guard-trips — cover the guard's false arm with equality, not `matches!`); gate **[B]**.
- **Coverage:** recursive `world_pos` is 100% branch in `vd-client`'s OWN tests (per-depth + guard) — critical, it's a Tier-A crate.
- **Proves:** MESH-INSIDE-MESH (dock/undock + self-heal) and THREE-DEEP nesting render correctly and the one composition chokepoint recurses without a per-realm-kind branch (HR3).

### Slice V6 — dual-shard PROCESS cluster + scenario (b) SIDE-CONNECTED siblings (closes D-30) [P]
- **Crates/files:** `crates/bins/src/bin/vd-devcluster.rs` (roster spawns `vd-shard` ×2: 2nd realm + 2nd bind/probe port set; `await_ready` waits BOTH grants); `crates/bins/src/lib.rs` (`SHARD`→`SHARD_A`/`SHARD_B` = `NodeId(3)`/`NodeId(4)`); `crates/bins/src/bin/gateway.rs` (`known_shards` → 2-element set, `VD_SHARD`→`VD_SHARDS` comma-list — the `gateway.rs:73` comment already anticipates this); shard bin `--dev-boundaries <path>` (the one dev-only, gated non-test producer that writes `RealmBoundaries` at boot). `crates/bins/tests/crossing_visual.rs` scenario `crossing_visual_siblings`.
- **Scenario (b):** two peer realm boxes (the `System(5)`/`Planet(42)` topology, `topology.rs:1251`), lateral crossing `from_realm→to_realm`; client holds BOTH subs (the `held_subs` SET already supports it); both boxes render; dot crosses laterally. Optional paired second capture client (D-30 blocker ii) watches client-1 cross.
- **Tier split:** bin/**[P]** (process gate); verdict **[A]** (V1).
- **Coverage:** bin — process test. Add `just render-cross` (the D-30 owed recipe) → `just gate`.
- **Proves:** the crossings run over the REAL dual-shard process fabric (not in-process only); closes D-30 blockers (i) dual-shard spawn + (ii) paired capture + `render-cross` recipe.
- **User gate:** touches the frozen gateway roster — confirm scope with user before building (same flag TH/EM raised).

### Slice V7 — perf/load gate: hundreds of entities × many boxes [P] + [A]
- **Crates/files:** `crossing_visual_density` scenario reusing `p1_volume_dense_hundreds` (N≥128, `VD_DENSITY_CLIENTS`) with the dual-shard cluster + ~4–8 boxes rendered; reuse `crates/harness/src/latency.rs` (`percentile_unstable`); batch verdict in `crates/client-harness/src/assert.rs` (V1's `entities_in_expected_boxes`, `rayon` over entities — pure, deterministic reduction into a `BTreeMap`).
- **Asserts:** (i) every one of N entities' world-pos ∈ its correct box (Tier-A batch verdict over delivered `DevState`); (ii) content floor / no-magenta holds under load; (iii) **release-only** latency gate — render-snapshot build + box-selection stays sub-ms with N boxes+entities (named-const budget ~10–20× observed, ratio-floor + honesty + competitor-real guards, per the SPIKE-3a pattern).
- **Tier split:** batch verdict + latency math **[A]**; many-mesh render **[B]**.
- **Coverage:** batch verdict + latency percentile 100% [A]; render gate-proven.
- **Proves:** the multi-mesh composite holds at end-goal density (hundreds physically colliding in one location across meshes) with proven sub-ms hot-path latency — lands the load test WITH the perf-bearing render subsystem (per [Load tests when applicable]).

---

## Scenario matrix (each a named e2e capture)
| Topology | Named scenario | Backend reused | Capture assertion (Tier-A verdict) | Slice |
|---|---|---|---|---|
| Single boundary (smallest visible) | `crossing_visual` | `plant_one_crossing_shell` born-inside dwell | dot ∈ box A before, ∈ box B after | V4 |
| (a) MESH-INSIDE-MESH dock/undock | `crossing_visual_nested` | `crossing_nesting_e2e` inward walk + outward self-heal | dot ∈ parent→child (dock), child→parent (undock) | V5 |
| (c) THREE-DEEP (player-in-ship-in-station) | `crossing_visual_three_deep` | three-realm nested plant + recursive `world_pos` | `innermost_realm_at` = deepest; nested boxes render | V5 |
| (b) SIDE-CONNECTED siblings | `crossing_visual_siblings` | `topology.rs:1251` `System(5)`/`Planet(42)` over dual-shard processes | dot leaves box A, enters box B; both render | V6 |
| through-and-back / hysteresis | (assertion inside V5/V6) | `ShellCrossing::ThroughAndBack` | dot enters+exits, NO spurious commit | V5/V6 |
| Perf/load (hundreds × many meshes) | `crossing_visual_density` | `p1_volume_dense_hundreds` + dual-shard + boxes | all N ∈ correct box; sub-ms latency floor | V7 |

## Cross-cutting coverage (HR5)
Tier-A (`vd-client-harness` + `vd-client` + `vd-devproto`): `realm_scene.rs`, `assert.rs` additions, `camera.rs` framing, recursive `world_pos` in `view.rs`, `LoadRealmBoxes` serde — all 100% region+branch on synthetic inputs, named-const thresholds, branchless generic shims, `assert_eq!`/`expect_err` over `matches!`. Tier-B (`vd-client-render`): `boxes` module, box assets, `sync_boxes` — coverage-exempt by set-absence, gate-proven ONLY, kept as straight-line glue (every box/color/prim decision resolved by a Tier-A call). Determinism: `BTreeMap` box keys, run-stable `universe_tick` capture alignment, no wall-clock in verdicts; pixels structurally-reproducible only, `clear` self-calibrated from a corner.

## Deferred ledger (record in `docs/design/DEFERRED.md`)
- **`BoundaryCatalog` wire arm** — production successor to the dev-config box source; a reviewed closed `InterShardFlow`/`EventMsg` arm producing the SAME `RealmScene`; lands with task #133 (Station/Area realms) then P8 (ships). WHERE: `crates/wire`. WHEN: first dynamic-boundary producer. Until then, dev-config is the only source.
- **`RealmBox.mesh` block-grid arm** — box→greedy-meshed `ChunkGrid`; `to_render_prims` emits quads; renderer unchanged (binds to `MeshPrim`). WHERE: `crates/client-harness/src/realm_scene.rs` + `crates/client-render`. WHEN: P4 (planet terrain) / P8 (ship+station hulls).
- **`parent_version` (D-35)** — version-matched hull/interior compositing; the recursive `world_pos` (V5) is TIME-coherent only. WHERE: `EntitySnap`/`SnapshotDatagram` + hold-buffer in `view.rs`. WHEN: P8. V0's `RealmBox`/`RenderPose` add no field blocking it.
- **ly-cell offset (D-41)** — `center_offset` uses `offset()` (cell-zero); V0 introduces no new cell-free assumption beyond `RenderPose`'s existing one. WHERE: `RenderPose.cell` / `sample` rebase. WHEN: P10.
- **`RealmMesh` band-shell render** — `to_render_prims` may emit the create/destroy band shells for hysteresis visualization; deferred (not needed for the box-membership verdict). WHEN: optional polish.

## Files touched (summary, all absolute)
- **NEW [A]:** `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/client-harness/src/realm_scene.rs`
- **Edit [A]:** `.../crates/client-harness/src/{lib.rs,assert.rs,camera.rs}`; `.../crates/client/src/{render_snapshot.rs,net.rs,view.rs}` (recursive `world_pos`); `.../crates/devproto/src/dispatch.rs` (`LoadRealmBoxes`)
- **Edit [B]:** `.../crates/client-render/src/lib.rs` (`boxes` module: `RealmBoxAssets`/`RealmBoxEntities`/`sync_boxes`/`CaptureCam`)
- **Edit [P]:** `.../crates/bins/src/bin/{vd-devcluster.rs,gateway.rs,client.rs,shard.rs}`; `.../crates/bins/src/lib.rs`; NEW `.../crates/bins/tests/crossing_visual.rs`; `.../justfile` (`crossing-visual`, `render-cross` recipes → `gate`)
- **DEFERRED (design-only):** `.../crates/wire` `BoundaryCatalog` arm → `.../docs/design/DEFERRED.md`

## Two user-confirmation gates before building
1. **The A-now/B-later seam + the `RealmBox` projection field-set** (§ decision) — confirm before V2. V0–V1 are safe pure Tier-A scaffold; build immediately.
2. **The dual-shard process cluster** (V6) — touches the frozen gateway roster (`known_shards`, `VD_SHARDS`); confirm scope before building. V4–V5 run in-process and need no roster change.

---

## ADVERSARY

All load-bearing claims confirmed. I now have everything needed. Let me do one final verification on two hostile-hunt targets: (1) whether the `render` composition suppresses cross-sub duplicates correctly (the multi-mesh N-realm double-draw concern), and (2) whether `vd-devproto` is genuinely in the Tier-A gate set — since the plan puts `LoadRealmBoxes` there.

The justfile `tier_a` (line 12) includes `-p vd-devproto` — confirmed. And `chosen_subs` (view.rs:144) collapses each entity to exactly one sub via `BTreeMap` insertion — confirmed the "rendered exactly once" invariant is real, not aspirational. I have full verification.

---

# HOSTILE REVIEW — Visual Crossing Playground synthesized design

I tried to break this before implementation. Every load-bearing claim in the synthesis was checked against the actual code. Findings below, ordered by severity, then a verdict.

## Credit where the design got it right (verified against code)

- **Seam choice is HR1-clean and I could not break it.** `RealmBoundaries` (`stub.rs:415`) is a shard-private Bevy resource, never serialized — grep of `crates/wire/src/` for boundary geometry is genuinely empty. Option A (dev-config `boxes.json` single-sourced with the shard plant) adds *zero* cross-process bytes, so HR1's "cross-shard bytes only via InterShardFlow arms" is untouched by construction. The `RealmBox` projection dropping `to_realm`/`band`/`effect` is the correct fix for TH's leak (those are shard-authority-internal; a render path has no business seeing them). **This is the strongest part of the plan.**
- **Box math reuse is real, not aspirational.** `Boundary::membership_scalar`/`signed_distance`/`swept` (`geometry.rs:262/275/245`) are all `Copy`, all proptested, and the box arms genuinely reuse one AABB shim via `orient.inverse()`. V1's `innermost_realm_at` reusing `membership_scalar <= 1.0` inherits that coverage — no duplicate branch logic. Correct.
- **The 3-deep `hull_pose` gap is REAL and honestly stated.** Confirmed at `view.rs:250-255`: `hull_pose` calls `track.sample(cursor)` directly, NOT `world_pos` recursively. The plan does not paper over this — it makes V5's three-deep scenario the thing that *forces* the recursion. Good.
- **`chosen_subs` genuinely renders each entity once** (`view.rs:144-157`): a `BTreeMap<EntityId, SubId>` collapse. The multi-mesh "N realms without double-draw" concern is structurally handled — a box path keyed in a *separate* `RealmBoxEntities` registry doesn't collide with it. Credit.
- **Anti-vacuity autonomous trigger is proven.** `plant_one_crossing_shell` (`lib.rs:493`) drives an autonomous `should_commit` dwell with NO `trigger_transfer` (`lib.rs:485`). Reusing it means V4's crossing is over the PROVEN trigger, not hand-fed. This directly answers hunt-target (9).

---

## FINDINGS

### H1 — HIGH — The synthesis DELETED the one anti-vacuity probe the codebase already built, and never wires it into the multi-box verdict
**Failure:** The design's capture oracle is V1 `entity_in_expected_box(scene, world_p, expected)` — a pure point-in-box test on the *composited* `world_pos`. But the codebase already ships `DeliveredView::subs_holding` (`view.rs:171`) precisely because "rendered once" / "in the right box" is **vacuously true if no overlap ever occurred** (view.rs:167-169, verbatim: *"otherwise 'rendered once' is vacuously true because no overlap ever occurred"*). The synthesized V4/V6 assert BEFORE-box-A / AFTER-box-B but never assert the *two-holder window was real*. A regression where the dest sub silently never delivers would still pass: dot sits in box A, "crosses" (directory flips), dot is now labeled box B by its `FrameRef`, renders at box-B origin — **green, with the dest shard's stream never having carried the entity.** That is exactly the vacuity class this repo already learned to guard.
**Concrete fix:** V4/V6's AFTER assertion must be a conjunction: `subs_holding(dot)` contained BOTH subs at some captured mid-frame (the overlap was real) AND `entity_in_expected_box == box_B` after. Add `subs_holding` to the `DevState` diagnosis surface (it is pure, Tier-A) and assert on it in the process test. Without this, the whole playground risks being a vacuous green.

### H2 — HIGH — The state-space verdict corroborated by pixels is asserted, but the design never pins the PIXEL of the specific box to the specific screen region — so the "visible" claim is itself partly vacuous
**Failure:** The plan correctly makes `world_pos`-membership the deterministic artifact and pixels "corroborating." But look at what "corroborating" reduces to: V3's smoke assertion is *">1 distinct non-clear color region + zero magenta,"* and the existing gate (`render_smoke.rs:221-237`) asserts `content_fraction >= MIN_SCENE_FRACTION` + `region_nonempty(center_third)`. **None of these asserts that box System(7) is at screen-region X or that the dot's pixels fall inside the box's pixels.** So the pixel side proves "*something* colored drew," not "*the dot* rendered *inside* box B." An entity rendering at the wrong box, or a box drawn but the dot culled, passes the pixel corroboration. The END GOAL is "PIXEL-visible dot-crossing-a-boundary" (memory: visual_testable_endstate) — the plan's pixel assertion does not reach it.
**Concrete fix:** Add a Tier-A `dot_pixels_within_box_region(rgba, dot_screen_aabb, box_screen_aabb)` verdict. The camera framing math is already Tier-A (`camera.rs` `fit_camera`), so projecting the dot's `world_pos` and the box center to screen-space is deterministic and coverable — this is the ONE place a screen-space projection is legitimate (TH's screen-AABB approach was rejected for the *membership* verdict, correctly, but it is exactly right as the *pixel-corroboration* verdict). Assert the dot's non-clear pixels are inside the box's projected AABB before AND after. This closes hunt-target (4): the current corroboration is a count, not a position.

### H3 — MEDIUM — `RealmBox.color_seed → hue` is claimed HR3-clean but the design never states WHERE the seed comes from; the obvious source re-introduces a realm-KIND branch
**Failure:** The plan says color is "`color_seed`-derived (golden-ratio hue of `RealmId`), a pure function of identity — never a `match` on realm KIND (HR3)." But `RealmId` is a flat enum (`System(u64)`, `Planet(u64)`, per task #133 `Station`/`Area`). To get a `u64` seed from `RealmId` you must destructure the enum — and the natural implementation is `match realm { System(s) => s, Planet(p) => p, ... }`, which **IS a per-realm-KIND match in a feature path** (hunt-target 5). Two `System(5)` and `Planet(5)` would then collide to the same hue, or you branch on kind to disambiguate — either a bug or an HR3 violation.
**Concrete fix:** Derive the seed from a KIND-agnostic stable hash of the *whole* `RealmId` (e.g. `RealmId` implements a `fn stable_seed(&self) -> u64` on core that hashes the discriminant+payload uniformly — one place, not a feature match), or add such a method to `RealmId` in `vd-core` and have `color_from_seed` consume only the `u64`. The render/harness code must never see the enum arms. State this explicitly in V0.

### H4 — MEDIUM — The `MeshPrim` IR is claimed to make the renderer "never learn the word box," but V3's `sync_boxes` selects mesh/material by `BoxShape`, re-introducing a shape branch in Tier-B GPU glue
**Failure:** The synthesis reconciles RF+EM as "`RealmBox::to_render_prims -> Vec<MeshPrim>`, renderer consumes only `MeshPrim`." But then V3 describes `RealmBoxAssets` with a per-realm `StandardMaterial` and `sync_boxes` "consuming `to_render_prims`." If `MeshPrim` is `{shape_or_verts, color_rgba, transform}` and `shape_or_verts` still carries a `Sphere{r}` vs `Cuboid{half}` distinction, then `sync_boxes` **branches on that** to build the Bevy `Mesh` — a shape match living in the coverage-EXEMPT Tier-B crate. That is HR5-adjacent (real selection logic hiding in the GPU-exempt renderer, hunt-target 7) AND a mild box→block corner (when P4 emits arbitrary greedy-quad vertex buffers, the sphere/cuboid arm must vanish cleanly).
**Concrete fix:** Make `MeshPrim` carry ONLY a `Vec<Vertex>` + `color_rgba` + `transform` — i.e., `to_render_prims` *tessellates* the sphere/box into vertices in Tier-A (coverable), and the renderer does `Mesh::from(vertices)` with zero shape branch. Then P4's greedy-quads are just more vertices, and V3 is genuinely straight-line glue. The sphere-vs-box tessellation branch belongs in Tier-A V0, not Tier-B V3.

### H5 — MEDIUM — Recursive `world_pos` cycle-guard: the design says "cover the guard's false arm with equality" but `world_pos`/`hull_pose` today has NO depth parameter to guard on
**Failure:** V5 makes `hull_pose` "resolve its own frame through `world_pos` recursively … with a named depth-guard const." But `world_pos(&self, pose, cursor)` and `hull_pose(&self, hull, cursor)` (`view.rs:227,250`) take no depth accumulator. Adding recursion means threading a `depth: u8` through both, and the guard's *false arm* (depth exceeded → stop) must be exercised by a genuinely-cyclic delivered track (hull A's frame is `ShipLocal{B}`, B's is `ShipLocal{A}`). Constructing that cycle from *delivered snapshots* in a Tier-A unit test is non-trivial — the tracks map is keyed `(SubId, EntityId)` and you must plant two mutually-referencing hull tracks. If the test can't build the cycle, the guard's false arm is uncoverable → HR5 fails (hunt-target 6/7). The plan asserts this is coverable but does not show the cycle-construction is reachable from the public track-ingest API.
**Concrete fix:** Confirm (in V5's design) that `on_snapshot` can ingest two entities whose poses reference each other's frames — it can (they're just `EntitySnap`s with `ShipLocal` frames). Add an explicit V5 unit test that plants the A↔B cycle and asserts the recursion returns the depth-capped fallback (`pose.pos`) via `assert_eq!`, not `matches!`. Name it as a required-before-GO coverage item, since a real Tier-A crate (`vd-client`) branch is at stake.

### H6 — LOW — V6 touches the frozen gateway roster (`known_shards`, `VD_SHARD`→`VD_SHARDS`) — correctly gated for user confirmation, but the scope is under-stated
**Failure:** `gateway.rs:76` `known_shards = BTreeSet::from([shard])` and `vd-devcluster.rs:228` spawn exactly one `vd-shard`. V6 changes both plus `bins/src/lib.rs` `SHARD`→`SHARD_A/B`. The design flags a user gate (good), but the routing implication is deeper than "extend the set": the gateway's snapshot-routing and `SubscriptionOpened` currently open exactly `SubId(0)` (per the map). A client seeing BOTH sibling boxes needs the gateway to open TWO subs and the client to hold both in `held_subs` — the `held_subs` SET supports it (`view.rs:56`), but no code opens a second sub today. V6 as written ("both boxes render; dot crosses laterally") silently assumes dual-subscription delivery that isn't built.
**Concrete fix:** V6 must explicitly include the gateway opening a second subscription and the client admitting it — this is real routing work, not just a roster-size bump. Either scope it in (larger V6) or restrict scenario (b) to a single-observer-follows-the-dot crossing (one sub at a time, the dot's authority migrates A→B) and defer simultaneous dual-box viewing to when D-30's dual-shard compositing lands. State which.

### H7 — LOW — Determinism hole: `boxes.json` load ordering into a `Vec<RealmBoundary>` vs `BTreeMap<RealmId, RealmBox>` — the projection must be order-independent, and the depth-walk reads a `Vec`
**Failure:** `from_boundaries(&[RealmBoundary]) -> RealmScene`. The reimplemented Tier-A `boundary_depth` (mirroring `stub.rs:2614`) walks parents by `boundaries.iter().find(|b| b.realm == p)` — a linear scan over the *input Vec order*. If `boxes.json` is loaded in a nondeterministic order (or two realms share a `realm` id by authoring error), `find` returns the first match and depth becomes order-dependent. The `RealmScene` `BTreeMap` key fixes final ordering, but the *depth computation* reads the Vec. Not a live bug (dev-config is authored), but a latent determinism trap the plan claims is fully handled.
**Concrete fix:** In V0, build a `BTreeMap<RealmId, RealmBox>` FIRST, then compute depth over the map (lookup by key, not linear find), and `expect_err`/reject duplicate `realm` ids at load. Makes depth a pure function of the *set*, order-independent — and covers the duplicate-id branch.

### H8 — LOW — Band-shell (hysteresis) visualization is deferred, but the "through-and-back / no spurious commit" scenario (V5/V6) then has NO visible artifact to corroborate
**Failure:** The scenario matrix includes "through-and-back / hysteresis → dot enters+exits, NO spurious commit." But the deferred ledger drops the band-shell render ("not needed for the box-membership verdict"). Without the create/destroy shells drawn, the hysteresis scenario is verifiable ONLY in state-space (no commit fired) — there is nothing *visible* distinguishing "entered the band but didn't commit" from "never entered." For a *visual* playground whose point is making transfer behavior visible, the one scenario that is purely about the band is invisible.
**Concrete fix:** Either render the band shells for this scenario (the `to_render_prims` band-shell arm the ledger defers) so through-and-back is visibly "dot dips into the translucent band shell and backs out," or explicitly scope the hysteresis case as a state-space-only assertion inside V5/V6 (not a *visual* scenario) so the matrix doesn't over-claim. The former is truer to the end goal.

---

## Non-issues I tried and could not turn into findings
- **Box-as-box → per-realm block mesh corner (hunt-target 2):** The `RealmBox.mesh` arm + `to_render_prims` lowering genuinely makes P4 a lowering change, not a renderer change — *provided H4 is fixed* (renderer consumes vertices, not shapes). Ledgered correctly in DEFERRED.
- **Multi-mesh depth/transparency at N realms (hunt-target 3):** `AlphaMode::Blend + cull_mode:None` with Bevy back-to-front sort is the standard answer; 3-deep nesting composes through the one chokepoint. No break found beyond H5's guard-coverage concern.
- **Tier-A/Tier-B line (hunt-target 7):** Verified `tier_a` in `justfile:12` includes `vd-client`, `vd-client-harness`, `vd-devproto`; `vd-client-render` is absent → exempt by set-absence. The line is drawn correctly EXCEPT for the shape-branch leak in H4.
- **Cornering signals/PvP/hundreds (hunt-target 8):** V7 reuses `p1_volume_dense_hundreds` with the SPIKE-3a latency pattern; additive `RenderPose`/`EntitySnap` growth for D-35/D-41 is ledgered. No corner introduced.

---

## VERDICT: **GO-WITH-FIXES**

The architecture is sound: the seam is HR1-clean (verified — zero new cross-process bytes), the box→block lowering is genuinely non-cornering, and the recursion gap is honestly surfaced. But two HIGH findings mean the playground as specified risks being a **vacuous green** — the exact failure mode this codebase has repeatedly hardened against, and the one a hostile reviewer must not let through.

**Required before GO:**
1. **(H1)** Wire `subs_holding` into V4/V6's AFTER verdict — assert the two-holder overlap was REAL before asserting the dot landed in box B. No transfer scenario may assert box membership without proving the crossing actually delivered on the dest sub.
2. **(H2)** Add a Tier-A `dot_pixels_within_box_region` pixel-corroboration verdict (dot's pixels inside the box's projected screen-AABB), so "PIXEL-visible dot-crossing-a-boundary" is actually asserted, not just "something drew."
3. **(H3)** `RealmId → color seed` must be a KIND-agnostic stable hash on `vd-core` (one method), never an enum-arm match in the render/harness path. State it in V0.
4. **(H4)** `MeshPrim` carries vertices (Tier-A tessellation), not shape variants — so `sync_boxes` is branchless glue and P4 greedy-quads slot in with zero renderer change.
5. **(H5)** V5 must include an explicit Tier-A cycle-construction test proving the recursive-`world_pos` depth guard's false arm is coverable (`assert_eq!` on the capped fallback); confirm the A↔B frame cycle is buildable from the public `on_snapshot` ingest.

**Recommended (not blocking):** H6 (state whether V6 opens dual subscriptions or defers simultaneous dual-box viewing), H7 (depth over the `BTreeMap`, reject duplicate realm ids), H8 (render band shells or scope the hysteresis case as state-only).

The user-confirmation gates the plan already names (A-now/B-later seam field-set before V2; dual-shard roster before V6) are correct and should stand.