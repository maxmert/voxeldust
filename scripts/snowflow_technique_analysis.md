# SNOWFLOW brief — what to carry into Voxeldust

**What this is.** An analysis of a third-party implementation brief for a 90-second WebGPU/Babylon.js
snow tech demo ("SNOWFLOW"), read for transferable technique only. Our stack is Bevy 0.18.1 + Rust,
Rapier planned at P5. Every Bevy claim below was read from the vendored sources at
`~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/bevy_*-0.18.1/`; anything I could not verify is
marked **unverified**. Repo paths are relative to the `new-system` worktree.

Nothing here is a decision. It is input to decisions, several of which only the project owner can make
(§7).

---

## 1. Bottom line

The brief is an unusually good *craft* document and a dangerous *architecture* document, and the split
is clean: everything it says about **how to make a surface look real** and **how to work** transfers
almost verbatim; everything it says about **where state lives** is single-player and would corner us.

It is a demo spec, and it shows in five specific places. There is one player, so no mark ever has to
agree between two observers. There is no server, so the client owns the world and a GPU texture can
*be* the truth. There is a loading screen, so every shader can be compiled behind a curtain exactly
once. The world is a flat 60–100 m field with a fixed origin, so float precision, moving reference
frames and a spherical "up" never arise. And it lasts ninety seconds, so "persistent" means "until the
tab closes".

The three ideas most worth taking are not graphics ideas at all: **measure the 1% low, not the
average**; **allocate the frame budget per system before you spend it**; and **build the tuning
overlay first**. We have none of these — the client has no frame-time readout of any kind today — and
at P4 we will be adding terrain meshing, materials, lighting and shadows in overlapping slices with no
way to attribute a cost to any of them.

The three ideas most likely to hurt us are the shared deformation render-target (client-owned world
state), the geometry clipmap (a heightfield structure that forecloses caves), and the working
agreement's "do not build a test suite" (our render crate is coverage-exempt *because* the visual
harness proves it — remove the harness discipline and the crate has no proof at all).

One thing the brief gets right that we should copy without modification: **it refuses to proceed past
an ugly milestone.** That is a governance idea, it costs nothing, and we do not have it.

---

## 2. The ten things worth stealing outright

Ranked by payoff-to-cost.

**1. Frame-time graph showing the 1% low, not average FPS.** *Source: SNOWFLOW §3, §5.*
Our hitches will be event-correlated — a realm spinning into visibility, a chunk remesh, a first-draw
pipeline compile — and an average erases exactly those. `FrameTimeDiagnosticsPlugin` is available with
no feature change (`bevy_diagnostic-0.18.1/src/frame_time_diagnostics_plugin.rs:14-19`); the percentile
helper already exists in `crates/harness/src/latency.rs` and already documents the trap (empty sample
set returns zero and passes a `< budget` assert — pair it with a sample-count floor). Cost: ~1 day.
Land: now.

**2. Build the tuning/perf overlay early — "it will save hours".** *Source: SNOWFLOW §5.*
We already ship egui through `bevy_egui`, and `draw_hud` is already shared by the windowed and headless
capture paths. A slider is the only way to reach a good value for parameters like exposure, fog
coefficient or detail-normal blend; reasoning does not get you there. Cost: half a day plus the policy
in §3. Land: before P4. Note `bevy_dev_tools::FpsOverlayPlugin` is **not** in bevy's default features
(`default = ["2d","3d","ui"]`; `bevy_dev_tools` lives under `dev`) — draw it in our own egui HUD rather
than enabling a feature.

**3. "Spend the budget where the player looks."** *Source: SNOWFLOW §2.6.*
The principle transfers; the brief's conclusion does not, because our camera is first-person today and
our own avatar is never on screen. Our actual dwell-weighted ranking is: your own ship interior (P8,
and player-built so it can never be art-directed), the planet horizon and sky, terrain inside ~50 m,
other players at 20–200 m. Make the allocation a runtime measurement over screen coverage in one named
config struct, not an authoring-time judgement. Cost: the discipline, not the code.

**4. Multi-scale surface detail — "legible at three distinct scales simultaneously".**
*Source: SNOWFLOW §2.2, §8.* This is the single strongest visual criterion in the document and it is
what makes a surface hold up at 0.5 m and 500 m from one mesh — which matters more for us than for the
demo, because the binary-render-rule leaves us no impostor to hide behind at distance. Land: P4/P5 with
the first terrain material. Caveat: mipmapping is mandatory for it and is **not** LOD in the banned
sense — see §5.

**5. Wrapped-diffuse translucency ("subsurface"), which the brief calls the single highest-value term.**
*Source: SNOWFLOW §2.2, §8.* We get it with no custom shader: `StandardMaterial::diffuse_transmission`
is a plain ungated scalar (`bevy_pbr-0.18.1/src/pbr_material.rs:227` — only the *texture* variants sit
behind the non-default `pbr_transmission_textures` feature), completed by `thickness`, `ior`,
`attenuation_distance`/`attenuation_color`, plus `TransmittedShadowReceiver` for the shadowed back
lobe. Generalises far past snow: ice, water, thin hull panels, foliage. Cost: hours to prototype.
Land: P4/P5.

**6. Aerial perspective / distance contrast compression.** *Source: SNOWFLOW §2.4, §8.*
The only cue that tells a player a 200 km planet is 200 km. Take Bevy's physically-based path rather
than the brief's fog: `DistanceFog`/`FogFalloff` has **no height-falloff variant at all**
(`bevy_pbr-0.18.1/src/fog.rs:99` — Linear / Exponential / ExponentialSquared / Atmospheric, all pure
distance), while `bevy_pbr::atmosphere` gives real height-varying extinction and in-scattering via its
aerial-view LUT. It also honestly hides the terrain draw-distance cut. Land: P4. Caveats in §5.

**7. Explicit per-system frame budget, recorded.** *Source: SNOWFLOW §3.*
We have no frame target, no reference machine and no budget anywhere in `docs/design/`. Two
adaptations: the budget belongs in code as a named config struct (the `ClientInterpTuning` pattern),
and the measured numbers belong in the run manifest we already write per capture, against a *named
scene* (n realms visible, n players) — a single global table is meaningless when our cost varies with
the AoI set. Platform caveat: `RenderDiagnosticsPlugin` records GPU timestamps **only on Vulkan and
DX12** — "On other platforms (Metal, WebGPU, WebGL2) only CPU time will be recorded"
(`bevy_render-0.18.1/src/diagnostic/mod.rs:53-55`). On the dev Mac, per-system GPU cost is measured by
A/B toggle-delta, not by span, which is a second reason the toggles in item 2 are an instrument.

**8. Pipeline warm-up before anything is visible.** *Source: SNOWFLOW §4.*
The brief's fear is real and **worse for us**: Bevy hardcodes `cache: None` on both pipeline
descriptors (`pipeline_cache.rs:613,680`) so nothing survives a process restart; wgpu's
`PIPELINE_CACHE` is documented Vulkan-only with DX12 and Metal unimplemented
(`wgpu-types-27.0.1/src/features.rs`); and on macOS the async compile path is compiled *out* —
`create_pipeline_task` takes the `bevy_tasks::block_on` branch under `target_os = "macos"`
(`pipeline_cache.rs:852,871`). So on the dev machine every first-time compile blocks the render thread,
with no loading screen to hide it. The mechanism transfers verbatim (render each pipeline once to a
tiny offscreen target); the *trigger* moves from load-time to the demand loop. Land: design now,
implement P4.

**9. A visual acceptance checklist, verified from a fresh capture and in motion.** *Source: SNOWFLOW §8.*
Our gates prove a thing *drew*; not one of them would fail on an ugly frame. Sort the brief's criteria
into machine-judgeable and human-only and be honest about the split. Already machine-judgeable with
existing primitives: "sparkle does not crawl" is `differing_pixel_count` on two captures of a still
scene. Needs ~10 lines each of new pure Tier-A function: highlight-clip fraction (we only bound the
*mean* luminance today, so a blown sun over a dark foreground passes), shadow-hue statistic,
near-vs-far contrast. Honestly human-only: "no visible faceting", "reads as fabric". Add our own:
no blank or flicker frame across a crossing; no seam when a realm spins into visibility.

**10. "Do not proceed from an ugly milestone" + commit the screenshots.** *Source: SNOWFLOW §7, §9.*
The gate discipline is free and we lack it. `/runs/` is gitignored in full, so no capture we have ever
taken is versioned and there is no way to answer "does this still look as good as it did in June". A
curated `docs/visual/` with one or two downscaled PNGs per landed slice is a few hundred KB a year.
The brief's *order* must be inverted though — it puts performance hardening last, which is the exact
structural inversion this rebuild exists to escape (§5).

---

## 3. Decide before we start rendering

Ordered by cost-of-lateness, worst first.

### 3.1 What is one voxel, in metres?

**Decision:** a named constant for block scale.
**Why now:** it is not stated anywhere in `docs/design/` — `roadmap.json` gives planet radius in
*blocks* (100k–200k) and chunk size in *voxels* (62³), and nothing bridges them. Every downstream
number in this analysis rests on it: at 1 m the ground horizon on a 150 km planet is 714 m and the
surface-chunk disc inside it is ~417 chunks (comfortably one tier); the shadow cascade-0 texel is
8.3 mm, which is ~120 texels per block face (crisp) at 1 m and ~2.4 (marginal) at 2 cm. Two of the
cluster analyses assumed different values 50× apart and reached opposite conclusions about whether
shadows are adequate.
**Cost of deferring:** every chunk-count, shadow, LOD-threshold and parallax-scale number derived
before it is settled is provisional, and they will be quietly copied forward.

### 3.2 Blocky greedy quads, or a smooth iso-surface?

**Decision:** does natural terrain read as Minecraft blocks forever, or as a real planet?
**Why now:** `roadmap.json:60` names `binary-greedy-meshing` on 62³ chunks — axis-aligned blocky
faces, a permanent ceiling on silhouette that no shader removes. Parallax, triplanar and detail
normals make a blocky surface *better-textured*, never *non-blocky*. It also decides the vertex
format (§3.3), whether triplanar costs one sample set or three (axis-aligned faces make the blend
one-hot), the collider path, and the AO scheme. The middle path real games take (Astroneer, Deep Rock,
`godot_voxel`): store density rather than block ids, keep the *block* as the edit unit, extract a
smooth surface.
**Cost of deferring:** discovering at P6 that terrain can never look right means redoing P4 and P5
together.

### 3.3 The Tier-A vertex format — decide once, before the P4 mesher.

**Decision:** what `Vertex` carries, and whether `MeshPrim` gains rotation and indices.
**Why now:** today it is `{ pos: [f32;3], normal: [f32;3] }` and `PrimTransform` is translation +
scale with **no rotation** (`crates/client/src/realm_scene.rs:519-544`) — so a rotating planet, a
banking ship or a spinning station cannot be drawn correctly *at all*. Five separate analyses each
proposed a different addition to the same 100%-covered type: UV+tangent (needed by
`Mesh::generate_tangents` and by `anisotropy_strength`, which is documented meaningless without mesh
tangents), a per-vertex material id, a surface-space coordinate for position-keyed shading, a
per-vertex scalar for along-spine effects, `detail_tier: u8`. The crate's own comment says P4 terrain
emits greedy quads through this same type "with zero renderer change" — so P4 *is* the deadline.
Sizing: 24 B → ~52 B is a 2.2× vertex-buffer cost on all terrain, so this is a budget decision, not a
free add.
**Cost of deferring:** widening it after terrain, blocks and ships are all producers is a
simultaneous migration of every producer plus the tessellation helpers plus their coverage tests.

### 3.4 The render frame: camera-anchored, +Y = local up, integer-snapped re-anchor.

**Decision:** the renderer never inherits "world = the realm frame".
**Why now:** one convention buys three things at once. (a) It is the *only* configuration in which
Bevy's atmosphere is correct — `get_view_position()` is
`view.world_position * scene_units_to_m + vec3(0.0, atmosphere.bottom_radius, 0.0)` and the next
function carries the comment, verbatim: *"We assume the `up` vector at the view position is the y
axis, since the world is locally flat/level. NOTE: this means that if your world is actually
spherical, this will be wrong."* (`bevy_pbr-0.18.1/src/atmosphere/functions.wgsl:303-311`). (b) It is
the f32 precision fix — the client is documented to jitter past ~1e7 m, and the integer lattice cell is
currently *dropped* on the render path (`crates/client/src/interp.rs:113-117`,
`crates/client/src/view.rs:207-212`). (c) Every "local up" shading term degenerates to the cheap
world-+Y form while staying physically correct. And the fix lands in the same `world_pos` chokepoint
the recursive-composition rewrite is already going to touch.
**Cost of deferring:** every shader written first bakes in world-+Y or realm-absolute positions, and
the atmosphere in particular is not incrementally fixable — it is wrong everywhere except one tangent
point.

### 3.5 The pipeline set is a function of the BUILD, never of the WORLD.

**Decision:** all biome/planet/material variety is seed-derived *uniform and texture data* fed to a
small fixed set of materials — never new shader keys, never per-biome `#ifdef`s.
**Why now:** if the permutation count scales with content, hitch-free on-demand streaming is
impossible *in principle*, not just in practice: no loading screen, no persistent cache, and a
blocking compile on macOS. Bevy makes it easy to get wrong — `ShaderDefVal`, `SpecializedMeshPipelines`
keyed on `(MeshVertexBufferLayoutRef, key)`, and `StandardMaterialKey`'s 23 feature bits each fork a
pipeline. The corollary gate: after warm-up, assert `PipelineCache::waiting_pipelines()` is empty, and
after a full VU scenario assert no *new* pipeline was created.
**Cost of deferring:** an unbounded set cannot be warmed. It surfaces as an unreproducible
several-hundred-ms stall at a warp arrival on a cold process — the hardest failure in the game to
reproduce, at the exact moment the whole arc is being judged.

### 3.6 The three-tier mark rule, written down before the first mark ships.

**Decision:** every mark is exactly one of — **V** voxel edit (authoritative, replicated, persisted);
**S** surface state (authoritative, persisted, coarse 2-D attribute, *does not exist in our design
yet*); **C** cosmetic overlay (client-local, GPU, ephemeral, never persisted, never replicated).
Assignment test: it is authoritative if it changes collision, changes containment or line-of-sight, is
a resource/ownership fact, **is evidence a player would act on**, or must survive restart or realm
sleep/wake. Otherwise cosmetic — and cosmetic additionally requires that no player decision may depend
on it.
**Why now:** every later effect copies whatever tier the first one picked. If footprints ship
client-local and a designer later wants tracking, promotion means a new wire arm, a storage schema and
a fairness re-audit, with several effects already copying the wrong pattern.
**Cost of deferring:** also note Tier S *breaks P4's stated premise* ("no chunk streaming, no networked
terrain — only the seed crosses the wire") by construction, since it is non-analytic player-caused
state. That carve-out must be made explicitly in the P4 design, not discovered.

### 3.7 Declare a cosmetic-local simulation tier — and its four rules.

**Decision:** client-side cosmetic simulation (cloth, spray, ripples, micro-detail, camera dynamics)
is permitted and is **not** client-side prediction, provided it is a pure function of already-delivered
authoritative state + the render cursor + a shipped seed, and: never on the wire, never in a checkpoint
or transfer blob, never read by the sim, never authoritative for collision/hit/interaction.
**Why now:** the no-prediction law exists because predicted *positions* get contradicted and players
physically collide. A cloth solver makes no gameplay claim and there is nothing to reconcile against —
the server holds no cloth state. Without this sentence written down, "NO client-side prediction" reads
as banning all secondary motion, which makes good VFX literally impossible; with it, the boundary is
mechanically checkable ("does anything read this back?").
**Cost of deferring:** either effects get built as server state (a particle plume on the wire —
unaffordable), or someone builds local prediction under the banner "it's only cosmetic". Both are
cheap to prevent with one paragraph.

### 3.8 MSAA or the temporal family — one decision, atomically.

**Decision:** we are on `Msaa::Sample4` today purely by Bevy default (required component of `Camera`),
never by choice. TAA hard-requires `Msaa::Off`, `#[require(TemporalJitter, MipBias, DepthPrepass,
MotionVectorPrepass)]`, and its doc says it "does not work well with alpha-blended meshes" — our realm
shells are `AlphaMode::Blend` + `unlit` + double-sided. SSAO also requires `Msaa::Off` and
`#[require(DepthPrepass, NormalPrepass)]`. SSR "currently require[s] deferred rendering" and
auto-inserts `DeferredPrepass`. `AlphaMode::AlphaToCoverage` — the sort-free route for fur and
foliage — requires MSAA and degrades to `Mask(0.5)` without it.
**Why now:** one AA choice determines the shadow-softening choice, whether AO is reachable, whether
PCSS is ever reachable, what our pixel assertions can claim, and the warm-up permutation set.
**Cost of deferring:** effects authored under one model must be re-authored under the other, and the
capture baselines move twice. The switch must also be *atomic* — MSAA off with no TAA yet is the worst
image we could ship.

### 3.9 Terrain residency rides the existing demand loop. No second scheduler.

**Decision:** chunk generation, meshing, pipeline warm-up and GPU residency are declared phases of the
realm demand loop, keyed off the *same* AoI predicate, with the lead derived from measured upload cost.
Resolution of "visible before resident" is always **warm earlier**, never **draw later**.
**Why now:** the lifecycle FSM and its hysteresis are already proven end-to-end in real processes;
adding a phase now is additive. One honest caveat a naive design would get wrong: the warm-ahead margin
today is purely *geometric* (visibility radius ≫ crossing radius) — `boot_ticks_p99` is 0, so there is
**no predictive horizon** in the shipped code.
**Cost of deferring:** a bespoke chunk streamer duplicates the hysteresis and thrash guards that were
expensive to get right once, then diverges from them — and misses the one thing that makes it work,
that the demand loop already knows where the player *will be*.

### 3.10 Eliminate the window/capture asymmetry.

**Decision:** the windowed app and the headless capture must render the same image.
**Why now:** the starfield is created only in Windowed mode; `frame_scene_camera` runs only in Capture
mode. So a screenshot is currently *not evidence about what the player sees* — and our entire render
quality story rests on the visual harness, because the render crate is deliberately coverage-exempt.
Every quality gate built on top inherits the void.
**Cost of deferring:** it is a prerequisite for §2 item 9, not a cleanup.

### 3.11 Fix the content-present predicate before a sky lands.

**Decision:** replace "differs from one self-calibrated corner pixel" with something that survives a
non-uniform background. Today `render_smoke` samples the top-right corner and counts every differing
pixel. With a gradient sky essentially every pixel differs, `content_present_fraction` saturates to
~1.0 unconditionally, and both the 5% scene floor and `region_nonempty` become unfalsifiable.
**Cost of deferring:** the gates do not fail — they go **vacuous**, and keep passing while testing
nothing. Same class as the exposure/tonemapping baseline move; do both in the D-17 re-baseline slice.

### 3.12 One exposure law, physical rather than adaptive.

**Decision:** EV100 derived closed-form from the dominant star's incident illuminance
(`E_v = L / 4πd²`, both already seed/tick-derived), with a bounded rate-limited adaptation filter as a
pure Tier-A function. `AutoExposure` (histogram, `#[require(Hdr)]`, integrates against `delta_time`
with persistent GPU state deliberately not reset) available only as a debug A/B.
**Why now:** auto-exposure is a per-client feedback loop over *that client's* view — two players in
the same spot looking 30° apart get different brightness, and every capture becomes irreproducible,
killing `luminance_in_range` as a gate. Physical exposure checks out numerically against Bevy's own
constants (120 000 lux → EV100 15.55, bracketing `EV100_SUNLIGHT = 15.0`). Corollary: **every emissive
becomes an absolute seed-derived radiance** — our five hand-picked starfield tiers are currently tuned
against the fixed `EV100_BLENDER = 9.7` default and will be invisible in daylight and blinding at
night the moment exposure moves.
**Cost of deferring:** every emissive authored before the decision must be re-derived, over a range
that spans >20 EV from a lit surface to interplanetary night.

---

## 4. Technique-by-technique mapping

### 4.1 Terrain form and the far field (§2.1)

| Technique | Verdict | Bevy 0.18 anchor | When |
|---|---|---|---|
| Geometry clipmap / nested-ring LOD | **adapt** (residency pattern only) | we write it; `VisibilityRange` is the only built-in tier primitive | seam now, tiers ≥P8 |
| Sub-10 cm inner-ring vertex spacing | **adapt** → fragment-shader relief, not vertex displacement | `StandardMaterial::depth_map` + `parallax_depth_scale` + `parallax_mapping_method` | P4/P5 |
| Three explicit noise scale bands (not one fBm stack) | **adopt** | none — `vd-core` on pinned `noise = "=0.9.0"`, CPU, both sides | P4 |
| GPU-composited heightfield | **reject** for authoritative geometry | `gpu_readback` exists; rejected on determinism, not availability | never |
| Prevailing-wind anisotropic / sheared noise | **adopt**, as a *tangent field* on the sphere | we write it in `vd-core` | P4 |
| Sparse rock outcrops for mid-distance silhouette | **adapt** → emergent from a hard-material + erosion band | none (generator math) | P4 |
| Cover material on upward faces (snow-on-rock) | **adopt** | `MaterialExtension` fragment override | P4/P5 |
| Triplanar mapping | **adopt as the default**, not a slope fallback | none — verified zero occurrences of "triplanar" in all `bevy_*-0.18.1/src` | P4/P5 |
| Distant matte ridgeline / impostor ring | **reject** | `VisibilityRange`'s third tier (deliberately unused) | never |
| Aerial perspective with height falloff | **adopt** via atmosphere, not fog | `bevy_pbr::atmosphere` aerial-view LUT | P4 |
| One-pass near+far (no split cameras) | **adopt** | `Mat4::perspective_infinite_reverse_rh` + `Depth32Float` | now |
| Meshlets / GPU virtual geometry | **reject** | `bevy_pbr::meshlet` | never for terrain |
| Texel-snapped player-following structure | **adopt** (the snap discipline) | none — our `LatticePos` integer rebase | P4/P5 |

**The clipmap is a heightfield structure and that is the whole problem.** It presupposes one surface
per (x,z), vertex-displaced by a height texture, with the ring lattice snapped so it does not swim. Our
terrain is a 3-D density field with caves, overhangs and destructibility — there is no single height to
displace, so a clipmap cannot represent it. Adopting it would foreclose caves, and the foreclosure
would be discovered after P4 lays down the chunk contract. What transfers is (a) the *residency*
pattern — concentric bands whose origin is snapped to a band multiple so the set changes in discrete
events rather than churning continuously — and (b) the snap discipline itself, which we need
independently for the render anchor.

**Whether tiers are needed at all is arithmetic, not taste.** On a 150 km-radius planet at 1 m voxels,
eye height 1.7 m, the geometric horizon is `sqrt(h(2R+h)) = 714 m` and the surface-chunk disc inside it
is `π·714²/62² ≈ 417` chunks. That is a completely tractable single-tier draw — the sphere does the LOD
for you at ground level. It stops doing it the moment you leave the ground: at 5 km altitude the
horizon is 39.1 km and the disc is ~1.25 **million** chunks, 3000× more. So single-tier is correct for
a walking player and structurally impossible for a flying one. (All of this scales with §3.1.)

**Meshlets are disqualified by their own documentation**, and it is worth writing down so nobody spends
a week on it: *"The conversion step is very slow, and is meant to be ran once ahead of time, and not
during runtime. This type of mesh is not suitable for dynamically generated geometry."* Plus:
materials get no control over the vertex shader or vertex attributes (kills packed voxel vertices),
opaque only (kills water/ice), incompatible with MSAA, needs `TEXTURE_INT64_ATOMIC`, Vulkan/Metal only
(`bevy_pbr-0.18.1/src/meshlet/asset.rs:22-37`, `meshlet/mod.rs`).

**Parallax is not as cheap as it looks** — this is the one place two of the source analyses were flatly
wrong and it matters. `parallaxed_uv` *always* runs Steep Parallax Mapping first: a raymarch whose
iteration count is `mix(max_layer_count, 1.0, abs(Vt.z))`, with `max_parallax_layer_count` defaulting to
**16.0** (`pbr_material.rs:937`). Only *then* does the method branch — `Occlusion` adds one refinement
lookup, `Relief` adds `max_steps` binary-search lookups. Real worst case per fragment per depth map is
~18 samples (Occlusion) or ~23 (Relief{5}), at grazing angles — which is exactly what a planet horizon
is made of. Bevy's "single lookup" doc phrasing describes the refinement, not the total.

**Triplanar is both more general and cheaper for us.** Texture coordinates become a function of
position, so the mesher emits no UVs, greedy quads of any size tile without stretch, and chunk seams are
invisible. And because greedy quads are axis-aligned by construction, the blend weights are one-hot —
the face normal selects exactly one projection, so it costs **one** sample set, not three. That
property evaporates if we go smooth iso-surface (§3.2).

**A prevailing wind direction is the best generator idea in the brief.** Isotropic fBm has no preferred
direction, which is why procedural terrain reads as noise rather than as a place with a history.
Anisotropic shear along a direction field gives dunes, sastrugi, erosion lineation, ridge alignment.
The brief encodes it as one vector because it has one flat field; on a *sphere* a constant world vector
is not tangent everywhere and produces two antipodal singularities. The correct generalisation is a
tangent vector field derived from the planet's own seed (spin axis + a deflection giving
latitude-banded zonal flow) — and it can be written with only `cross`/`dot`/`normalize`/`sqrt`, all
IEEE-754 correctly-rounded, so it survives the byte-identical cross-CPU DoD. A `sin`/`cos`/`atan2`
formulation routes through libm, which is **not** guaranteed identical across platforms, and would
break the determinism gate silently. (Note `f64::tan` already appears in `visibility_factor` — fine
there, config-time; a defect inside the density function.)

**Noise CPU budget is a real P4 risk.** A 62³ chunk is 238,328 voxels; three bands at ~6 octaves plus a
warp is ~8 evaluations per voxel ≈ 1.9M/chunk. At an *assumed* 30 ns per 3-D simplex eval (unverified —
must be benchmarked) that is ~57 ms/chunk single-threaded, and the 417-chunk horizon is ~24 s of one
core. Mitigations: coarse-lattice evaluation with interpolation for interior samples, interval/Lipschitz
bounds to reject fully-solid and fully-empty chunks before per-voxel work, rayon. Note `Cargo.toml`
already flags that `noise` needs `opt-level=3` even in dev or it is ~40× slower.

### 4.2 Surface shading (§2.2)

| Technique | Verdict | Bevy 0.18 anchor | When |
|---|---|---|---|
| A custom material, not white-albedo stock PBR | **adopt** — but exactly ONE, data-driven | `ExtendedMaterial<StandardMaterial, E>` + `MaterialExtension` | P4 |
| Detail normals at three tiling scales, distance-blended | **adapt** | our extension shader; `StandardMaterial` has one `normal_map_texture` | P4 → P6 |
| Wrapped diffuse + back-scatter (subsurface) | **adopt** | `diffuse_transmission` (ungated) + `thickness` + `TransmittedShadowReceiver` | P4 |
| View-dependent glinting with a stable hash | **adapt** — key on integer voxel coord, not world pos | we write it | ≥P6, after the TAA call |
| Surface *states* read from one shared field | **adapt** — state bits on a base material, server-owned | `#[storage(N, read_only)]` param array + `2d_array` palette | P6 |
| Contact detail / micro-occlusion at edges | **adapt** | `ScreenSpaceAmbientOcclusion`; parallax; `bevy_pbr::decal` | P4 (SSAO) / P6 |
| CC0 PBR scans, vendored | **defer** — user decision | `ktx2`+`zstd_rust` are in bevy's default `3d_api`; `reinterpret_stacked_2d_as_array` | P4 decision |
| Surface states as *pipeline permutations* | **reject** | `StandardMaterialKey` (23 flags) → separate pipeline each | never |
| Blue-shifted ambient | **adapt** — derive it, never dial it | `AtmosphereEnvironmentMapLight` (IBL from the atmosphere) | P4 |

**One material type, not N.** Every additional `MaterialPlugin` is a distinct pipeline family, draw
bucket and warm-up set — and a per-content material type *is* a kind-branch in the renderer, which the
agnostic-client law forbids. All per-surface variation must be data fetched by an index: a storage-buffer
parameter row plus a texture-array palette, addressed by a per-vertex material id. The natural first
step at P4 ("just a `TerrainMaterial` for now") is exactly how this goes wrong.

**Extend rather than replace.** `ExtendedMaterial<StandardMaterial, E>` with an
`#import bevy_pbr::pbr_fragment` in the fragment override inherits shadows, clustered lights, fog and
atmosphere for free. `StandardMaterial` occupies bindings 0..30; extensions go at 100+ (Bevy's own
forward-decal extension uses `#[uniform(200, ..)]`). WGSL can be compiled into the binary with
`embedded_asset!`, so the client ships no runtime `assets/` shader directory — which matters for the
headless capture harness.

**Diffuse transmission is not free at runtime.** `STANDARD_MATERIAL_DIFFUSE_TRANSMISSION` adds a second
`LightingInput` and a second lighting evaluation inside *every* point, spot and directional light loop,
plus a second shadow-map sample per light with `TransmittedShadowReceiver`. Rule of thumb: ~2× per-light
lighting cost on surfaces that use it. With one sun that is fine; with several dynamic lights it is a
budget line. And it is a pipeline permutation, so it must be a *fixed, always-on* variant rather than
toggled per material instance.

**The glint hash is where our architecture beats the demo's.** The brief hashes on world position; ours
is composed, cell-dropped, jitters past ~1e7 m and *changes on re-home* — so glints would crawl exactly
when you move between realms, and two clients would disagree. Our correct key is
`(realm-local integer voxel coordinate, face axis)`: exactly-integer, bit-stable across frames and
re-homes, identical on every client. That is the same key the material id already lives on, so it costs
nothing extra *if* the id reaches the shader.

**Texture memory bounds how many surface *classes* we can afford**, far more tightly than intuition
suggests. My arithmetic (BC7 albedo + BC5 normal at 1 byte/texel, +33% for mips): 256 materials × 2 maps
× 1024² ≈ 680 MiB — over budget; 64 × 512² ≈ 43 MiB — comfortable. That ratio argues hard for *few
shared classes, many seed-parameterised instances*.

**Distance-blended detail scales are not LOD**, and neither is mipmapping — see §5.

### 4.3 Shared terrain state / deformation (§2.3)

| Technique | Verdict | Bevy 0.18 anchor | When |
|---|---|---|---|
| The three-tier mark rule (our replacement for "one buffer") | **adopt** | server-side; Tier C is one `RENDER_WORLD` image | rule now |
| Player-following 4096² R16F target | **reject as specified** / adapt sizing | `Image` + `R16Float`; note `new_target_texture` does *not* set `STORAGE_BINDING` | ≥P6 |
| Toroidal scroll, texel-snapped anchor | **adopt** | `ImageAddressMode::Repeat` gives the wrap free | ≥P6 |
| Mass-conserving depression + displaced-mass berms | **adopt** | server-side integer field; two channels client-side | P4 schema / P6 |
| Persistent surface states (compaction, wetness, ice) | **adapt** — authoritative, seed-derived look | `ExtendedMaterial` + `#[storage]` | P4 schema / P6 |
| Additive splat accumulation, never replay | **adapt** — true for Tier C, our WAL already does the Tier S version | compute pass + storage texture | P6 |
| Diffusion + decay refill | **adapt** — server decay must be **closed-form in tick** | second compute pass, ping-pong | P4 rule / P6 |
| Terrain vertex displacement from the buffer | **defer** — collides with greedy meshing | `MaterialExtension::vertex_shader()` **and** `prepass_vertex_shader()` | P6+ |
| Recompute normals from the same field | **adopt** — cheapest large win here | fragment-shader gradient, 4 taps | P6 |
| ONE shared write path for every effect | **adopt** — but the seam moves to the **wire** | generic kind-blind brush descriptor on the TLV bag | seam at P4/P6 |
| Marks persist after the effect ends | **adapt** — four persistence mechanisms, not zero | `RealmScene::with_delta`'s `added` is the warm hook | P6/P7 |

**The single buffer is simultaneously the render source, the gameplay state and the save file**, and
collapsing those three is the fastest way to corner ourselves. A GPU float texture is non-deterministic
across drivers (accumulation order), unreadable by the sim (an NPC or the dormant-world worldline cannot
query "is this field trampled"), and un-checkpointable. Split it per §3.6. Tier S is the piece that does
not exist in our design yet and is what the brief actually needs — a per-surface-column scalar field
over the analytic terrain, which does *not* change the block grid.

**Mass conservation upgrades from an art trick to a state invariant.** In an integer per-column depth
field, "displaced mass" is not a separate channel — it is the neighbours' depth increasing by exactly
what the centre lost. Conservation becomes a property a proptest can pin, two players necessarily agree
because it is one server-side integer field, and the Tier-S→Tier-V bridge falls out for free: when a
column's accumulated depth crosses one block height, promote to a real voxel edit. Pile snow high
enough and it becomes a block. (Rate-limit it, or a player can grind the block WAL by shovelling.)

**Server decay must be closed-form in time.** The brief's refill is `state(t+1) = f(state(t))` — an
iterative feedback loop on float texture state. Fine on the client. Server-side it breaks cross-binary
determinism *and* requires simulating a week of decay when a dormant realm wakes. Store
`(value, t0)` and evaluate `f(value, t0, universe_tick)` on read — the same Category-A shape as the
celestial math, and exactly the lazy-field kernel the dormant-world design already specifies.

**One GPU foot-gun worth writing down:** do *not* implement Tier C as a main-world `Image` mutated per
frame. `GpuImage::prepare_asset` calls `create_texture_with_data` on every asset change — i.e. it
recreates and re-uploads the **entire** texture. At 4096² R16F that is 33.6 MB per mutation; at 60 Hz,
2 GB/s for nothing. Create once as `RENDER_WORLD`-only and write exclusively from GPU passes.

**Vertex displacement is where the brief structurally collides with greedy meshing** and it must be
settled *before* the P4 mesher, not when the renderer wants it. Greedy meshing produces large flat
quads; a quad spanning 20 blocks has four vertices, and you cannot displace a footprint into it. Three
options: (a) emit a separate dense top-surface mesh inside a small radius — real silhouette, but the
dense↔greedy transition pops; (b) fragment-shader parallax — zero pop, no LOD ladder, but no true
silhouette at grazing angles; (c) displace only where the surface was already meshed densely.
Recommendation: **(b) now, keep (a) possible** — most of the perceived effect comes from state-driven
shading plus recomputed normals, and the mesher's output format should merely *reserve* the dense
variant. Two Bevy traps if displacement ever lands: shadow rendering goes through `PrepassPipeline`, so
displacing in `vertex_shader()` without `prepass_vertex_shader()` makes shadow geometry disagree with
visible geometry; and frustum culling uses the *undisplaced* `Aabb`.

**Recomputing normals is the piece to build first** in this whole cluster: 4 texture taps in the
fragment shader, works perfectly on a greedy quad, no dense geometry, no LOD ladder, no pop, no
interaction with Rapier or the mesher — and it is what makes marks read as *embedded* rather than
painted. Honest caveat: it gives shading response, not shadow-map occlusion; a trail self-*shades* but
does not cast a real shadow into itself. SSAO recovers much of the contact darkening.

**The shared write path is the best idea in §2.3 and the seam moves from the GPU to the wire.** Gameplay
is server-side and sealed; the client is agnostic and must never learn that "footstep" or "fireball"
exists. So the shared path has to be a **generic, kind-blind brush descriptor**: pose + shape + depth
delta + displaced-mass delta + material-state delta + falloff + ttl, one tag on the TLV signal bag. Feet,
thrusters, a landing ship, a mining laser, an explosion, weather — all emit the *same* descriptor; the
client splats it and never learns the cause. An older client skips it by length and just does not see
the mark: degraded, never wrong. This is also the sharpest cornering risk in the cluster — if the first
surface effect ships as `FootstepMsg`, the second adds `ScorchMsg`, and within three phases the client
has a `match effect_kind` in a feature path.

### 4.4 Atmosphere, sky, sun, shadows (§2.4)

| Technique | Verdict | Bevy 0.18 anchor | When |
|---|---|---|---|
| Low warm sun, long shadows | **adapt** — direction/colour/lux all seed-derived | `DirectionalLight{color, illuminance}`, `light_consts::lux`, `SunDisk` | P4 |
| Cascaded shadow maps | **adopt** | `CascadeShadowConfig{bounds, overlap_proportion, minimum_distance}` + builder | P4 |
| PCSS soft filtering | **defer** — API not in our build | `soft_shadow_size` is `#[cfg(feature = "experimental_pbr_pcss")]` | post-P5, user call |
| HDRI *or* a physical sky model | **adopt the model, reject the HDRI** | `bevy_pbr::atmosphere` (Hillaire 2020) | P4 |
| Blue-shifted ambient | **adapt** — generate the IBL from the sky | `AtmosphereEnvironmentMapLight{intensity, size}` on Camera3d **or a LightProbe** | P4 / P8 |
| Fog + aerial perspective with height falloff | **adapt** — no height falloff exists in `DistanceFog` | atmosphere aerial-view LUT | P4 |
| Ground blow / spindrift | **defer** — no particle system in Bevy 0.18.1 | `FogVolume::density_texture_offset` is a cheap stand-in | later |
| Volumetric light shafts | **defer** | `VolumetricFog` + `VolumetricLight`; atmosphere pairing is *untested* per Bevy's own doc | P8 |
| 4–6 dynamic lights budget | **adapt** — a per-observer AoI budget, one generic rule | clustered forward; `MAX_DIRECTIONAL_LIGHTS = 10` | P6/P8 |
| Subsurface term responds to dynamic lights | **adopt** — free | `diffuse_transmission` participates in `apply_pbr_lighting` | P4/P6 |
| One sky for the whole experience | **reject** — implicit demo assumption | `Atmosphere` is a per-camera component; no built-in blend | P4/P10 |

**PCSS does not exist in our build**, and this is worth stating flatly because it circulates in our own
notes: `DirectionalLight::soft_shadow_size` is `#[cfg(feature = "experimental_pbr_pcss")]`
(`bevy_light-0.18.1/src/directional_light.rs:108`), and that feature is not in bevy's default set. Any
plan that says "we will use PCSS" is planning against an API we do not compile, and enabling it is a
Cargo change on a pin we deliberately keep at full defaults. What *is* unconditionally available and
good: `ShadowFilteringMethod::Gaussian` (the default), a 9-tap Castaño/Witness filter. Ship that first.

**Bevy's atmosphere is the right method with three hard caveats**, all verified. (i) `#[require(..., Hdr)]`
— enabling it forces an HDR intermediate target, which moves every existing capture baseline. (ii) It
renders between `MainOpaquePass` and `MainTransparentPass`, so *opaque* geometry gets aerial perspective
automatically and *transparent* geometry does not — our `AlphaMode::Blend` realm shells will float in
front of the sky with zero attenuation, and the behaviour silently flips when P4 terrain lands as opaque
geometry. (iii) The +Y-up assumption quoted in §3.4. Also: the plugin silently `warn!`s and no-ops if the
adapter lacks compute shaders or `Rgba16Float` storage binding — so a sky assertion must assert the sky
is *present*, not merely that pixels exist.

**The per-planet scaling hook is `ScatteringMedium`, and it needs care.** In 0.18 the Rayleigh/Mie
parameters are not inline fields — they live in a `ScatteringMedium` **asset** which builds two GPU LUTs
(default 256×256, ~1 MiB `Rgba32Float`) and rebuilds them whenever the asset is modified. Thousands of
planets cannot each own one. The scaling answer is a small set of archetypal media (8–16 atmosphere
classes, deduplicated by quantised parameters) shared by handle, with per-planet distinctness carried
entirely in the cheap `Atmosphere` fields — `bottom_radius`, `top_radius`, `ground_albedo` — plus sun
colour. Do **not** mutate a medium per frame.

**Cascade arithmetic, for calibration.** `CascadeShadowConfigBuilder` defaults to 4 cascades, min 0.1,
max 150, first bound 10, overlap 0.2; the split schedule is exponential, giving bounds 10.0 / 24.66 /
60.82 / 150.0 m. At Bevy's default 45° vFOV and 16:9, cascade-0's diameter is ~17 m, so a 2048²
`DirectionalLightShadowMap` gives ~8.3 mm texels. Memory: 2048² `Depth32Float` × 4 ≈ 64 MiB **per
shadow-casting directional light** (256 MiB at 4096²) — so a binary star system with two shadow-casting
suns is 8 cascade passes and ~128 MiB. `MAX_CASCADES_PER_LIGHT = 4` is a compile-time array bound; 6
cascades is not reachable without patching Bevy. Derive `maximum_distance` from the same angular-size
machinery as AoI rather than inventing a second constant family.

**Everything about the sun is derivable and nothing is authored.** Direction = (star − observer) in the
composed frame, both already shipped by the realm chain. Colour = blackbody of the star's effective
temperature, which `SpectralDef{teff_lo_k, teff_hi_k}` already carries. Illuminance = `L/4πd²` from
`main_sequence_luminosity`. Apparent size = `2·atan(R/d)`, and `SunDisk{angular_size, intensity}` is
consumed per-light in `sample_sun_radiance`, which loops all directional lights — so binary systems are
representable. The sun *set* must be chosen kind-blind ("luminous bodies in the ancestor chain"), never
`if inside_ship { no sun }`.

**"One sky" is the demo-only assumption that would hurt us most, because it is invisible** — nothing in
the brief states it and a single-planet prototype will never expose it. `Atmosphere` is a per-camera
component with one planet's radii, and there is no built-in blend between two atmospheres (the `medium`
is a handle backed by GPU LUTs, so cross-fading media means re-uploading ~1 MiB). The law-conformant
shape is a **hard swap at a distance where the departing atmosphere's contribution is already below the
noise floor** — the sky you are in is the deepest ancestor whose atmosphere you are inside, every other
body's sky is off, and the swap point is where the physics says the contribution is zero, invisible by
construction rather than by a tuned fade. `AtmosphereMode::Raymarched` exists specifically for the
orbital view and is the natural mode on the space side.

### 4.5 Post-processing and image discipline (§2.5)

| Technique | Verdict | Bevy 0.18 anchor | When |
|---|---|---|---|
| The chain as an explicit ordered named thing | **adapt** — Bevy fixes the order; two of the brief's placements are wrong | `Node3d` graph edges | P4 |
| TAA | **adopt, gated** on continuous composition | `TemporalAntiAliasing{reset}` — `reset` is *the* discontinuity hook | P5 |
| HDR | **adopt** — and re-baseline in the same commit | `Hdr` marker; `#[require(Hdr)]` on Bloom/AutoExposure/Atmosphere | P4 |
| ACES vs AgX vs TonyMcMapface | **adapt** — reject `AcesFitted` | `Tonemapping` enum; `tonemapping_luts` IS in bevy's default `3d_api` | P4 |
| Exposure | **adapt** — physical, not `AutoExposure` (§3.12) | `Exposure{ev100}`, `EV100_SUNLIGHT = 15.0` | P4 |
| SSAO | **defer** to real geometry | `#[require(DepthPrepass, NormalPrepass)]`, needs `Msaa::Off` | P5 |
| SSR on wet/icy only | **reject** | `#[require(DepthPrepass, DeferredPrepass)]` — commits the camera to deferred | never (revisit ≥P6) |
| Restrained bloom | **adapt** — the mechanism that makes a star read as a star | `Bloom::NATURAL`, `#[require(Hdr)]` | P4 |
| Post-TAA sharpening (CAS) | **adopt** — and sharpen *then* grain, not the reverse | `ContrastAdaptiveSharpening{sharpening_strength}` | P5 |
| Depth of field | **reject** on the gameplay camera | `DepthOfField` — `max_depth` exists precisely because backgrounds are infinitely far | never (cinematic only) |
| Film grain | **defer** | none — verified absent from Bevy 0.18.1 | later |
| Motion blur | **defer** — same motion-vector dependency, no variance clamp | `MotionBlur`; transparent objects cannot be blurred | P10 |
| "Do not build a test suite" | **reject** | — | — |

**Bevy fixes the graph order and disagrees with the brief twice.** Verified sequence:
`EndPrepasses → SSAO → Atmosphere LUTs → StartMainPass → MainOpaquePass → RenderSky → MainTransparent →
EndMainPass → MotionBlur → Taa → Bloom → AutoExposure → DepthOfField → PostProcessing → Tonemapping →
Fxaa → Smaa → CAS → Upscaling`. So SSAO is a *lighting input* computed before the main pass, not a
post effect after TAA — and Bevy's own SSAO doc recommends pairing it *with* TAA so TAA denoises the AO
through the lit image. And the brief puts grain before sharpening; AMD's RCAS guidance, quoted verbatim
inside Bevy's CAS source, says apply grain *after* CAS. Believe the source.

**Correction to a claim that circulates in our notes: tonemapping and dither are already running.**
`check_views_need_specialization` sets `TONEMAP_IN_SHADER` and `DEBAND_DITHER` under `if !view.hdr`, and
`main_pass_post_lighting_processing` then calls `tone_mapping(...)` and `screen_space_dither(frag_coord)`
inside the mesh fragment shader — which is precisely *why* `TonemappingNode` early-returns on non-HDR.
This applies to `unlit: true` materials too (the call sits outside the unlit branch), so our realm shells
and all 2800 stars are already TonyMcMapface-tonemapped and dithered in every capture we have ever
taken. Adding `Hdr` **moves** tonemapping and dither from the mesh shader to the post pass; it does not
switch them on. Pixels will still change (different working precision, in-shader dither removed,
bloom/exposure inserted), so the re-baseline stands — but for a different reason, and the magenta
sentinel's exposure to tonemapping already exists today.

**Reject `AcesFitted` for a procedural universe.** Bevy's own doc: *"intentional and dramatic hue
shifting — bright greens and reds turn orange, bright blues turn magenta, significantly increased
contrast"*. We have thousands of seed-coloured bodies and a physically-blue sky; a tonemapper that turns
bright blues magenta will read as an art bug on some planets and not others, and we will chase it as a
shading defect. AgX and TonyMcMapface are both "very neutral, little to no hue shifting" and both are
available today (`tonemapping_luts` is inside the default `3d_api` chain). Keep TonyMcMapface as the
shipped default; A/B AgX. And it must be **one global constant** — a per-planet tonemapper is precisely
the hand-authored per-planet art the standing rule forbids.

**TAA's real problem for us is not aliasing quality, it is our three discontinuity classes.** Motion
vectors are `(unjittered_clip × world_pos) − (previous_clip × previous_world_pos)`, with the previous
term from `PreviousGlobalTransform`. (a) A per-entity frame change collapses our interpolation window, so
a crossing occupant jumps on screen — survivable via TAA's 3×3 YCoCg variance clamp, costing a few
aliased frames on that object. (b) A **camera** frame change makes every pixel's motion vector bogus at
once; in a low-contrast field the clamp passes and you get a full-screen ghost decaying at the 1.5%/frame
floor ≈ 1 s of smear, at exactly the SOI-crossing moment. (c) A **floating-origin rebase** is the worst:
camera and content translate by the same Δ, the image is pixel-identical, and the motion vectors are
wrong by the full Δ. The fix for (c) is cheap *if* the rebase is a single identifiable event with a known
Δ — add Δ to every `PreviousGlobalTransform` and to the camera's `PreviousViewData` before extract; both
are public main-world types with public fields, so it is ~20 lines. If re-centring is instead implemented
as scattered per-entity coordinate rewrites, there is no Δ to apply and no single point to hook.
`TemporalAntiAliasing{reset: true}` is the escape hatch for genuine cuts only — it costs one fully
aliased frame and must never be the routine warp path.

**Depth range is not our problem, and the far plane is not what it looks like.** Bevy uses
`Mat4::perspective_infinite_reverse_rh` with a `Depth32Float` target — near-uniform relative depth
precision to effectively unbounded distance in one pass. `PerspectiveProjection::far` is **not** a
depth-clip bound at all; it is only fed to the frustum-culling helper. So raising it costs nothing in
precision. Our current 6000 was chosen to keep the star sphere at r=2000 inside the frustum. Worth
correcting a live mismatch while we are here: the *capture* camera spawns with no `Projection` override,
so it uses Bevy's default `far: 1000.0`, while Tier-A `CaptureCamera::FAR_PLANE` is `1e12` — a point
beyond 1000 m projects happily in Tier-A and is culled by the GPU, so a verdict can assert about pixels
that were never drawn.

**Depth of field is rejected on three independent grounds**: we already render a 120 ms-old pose with no
prediction, so lens blur compounds the sluggish read; there is no focus target without a crosshair; and
`DepthOfField::max_depth` exists specifically because backgrounds are treated as infinitely far — with
"stars always visible" as a hard rule, DoF would blur the star field by construction, and clamping
`max_depth` just moves the artefact to a visible focus wall in mid-space.

### 4.6 Character, cloth, fur (§2.6)

| Technique | Verdict | Bevy 0.18 anchor | When |
|---|---|---|---|
| Budget by dwell × screen coverage | **adopt** (principle), reject the conclusion | measurement: `FrameTimeDiagnosticsPlugin` | now |
| Shell fur (20–40 shells) | **defer** — needs a total-fragment cap, not a per-character count | shared mesh + `MeshTag(u32)` keeps `AUTOMATIC_BATCHING` | later |
| Verlet cloth, distance + bending constraints | **adapt** — write our own ~200 lines in Tier-A | **none — Bevy 0.18.1 ships no cloth at all** (verified) | later |
| Cosmetic-local simulation as a declared tier | **adopt** | design doc + the dependency rule | now |
| `rebase(delta)` on every stateful visual simulator | **adopt** | none — Bevy has no floating-origin concept | now |
| Cloth shading: sheen + anisotropy + thin SSS | **adapt** — 2 of 3 free | `diffuse_transmission`+`thickness`; `anisotropy_*` **needs tangents** | P4 |
| "Prefer procedural over a bad rig" | **adapt** — small authored core + procedural on top | `SkinnedMesh`, `AnimationGraph` masks (`AnimationMask = u64`, 64 groups) | P5 |
| "Feet plant, not slide" under a 120 ms buffer | **adopt** — easier for us than for a predicting client | none — two-bone IK is ~60 lines | P5 |
| Footfalls displace snow + spray | **adapt** — split cosmetic spray from the authoritative mark | `ForwardDecal` (needs `DepthPrepass`) | P6 |
| Keep the face in shadow; cut what you cannot finish | **adopt** | — | now |
| Animation cost at MMO character counts | **adapt** — no built-in visibility gate | `animate_targets` iterates **every** `AnimationTarget`, no visibility check | P5 |

**There is no sheen/fuzz lobe in Bevy 0.18.1** — verified by grep for "sheen", "fabric", "charlie",
"ashikhmin" across `bevy_pbr` and `bevy_shader`: the only hit is the word "fabrics" inside the
`diffuse_transmission` doc comment. So the grazing-angle fuzz rim is a `MaterialExtension` fragment
shader. Two of the three cloth terms are free (`diffuse_transmission` ≈ 0.15–0.30 with `thickness`
≈ 5–20 mm gives the thin-fabric glow; `anisotropy_strength`/`anisotropy_rotation` gives the weave) —
but anisotropy's doc says *"mesh tangents must be specified in order for this parameter to have any
meaning"*, which is blocked at the vertex seam (§3.3).

**Do not run mikktspace on voxel chunks.** For axis-aligned greedy quads the tangent is analytically
determined by the face normal — exact, free, deterministic, and it survives remeshing after a block edit,
where `Mesh::generate_tangents` would be per-remesh CPU cost (and requires TriangleList + indices +
POSITION + NORMAL + UV_0 anyway).

**Shell fur's cost is a per-character draw multiplier the demo never pays.** Bevy skins per-vertex in the
vertex shader, so 24 shells means the same vertices re-skinned 24 times, in the main pass *and* in every
shadow cascade. Fragment side: a hood at mid-distance covering ~3% of a 1440p frame × 24 shells ≈ 2.6M
fragments for **one** character; twenty visible characters is the entire budget several times over. If it
ever ships, budget a **total** shell-fragment count and allocate greedily by screen coverage, crossfading
the outermost shell's alpha rather than snapping the count. Note this is *parameter* LOD — same mesh, same
material, count → 0 — not geometry LOD; see §5.

**Foot planting is easier for us than for a predicting client, which is genuinely counter-intuitive.**
Drive the walk-cycle phase by the integral of `|Δ drawn position|` rather than by wall-clock: because the
drawn path is exactly a lerp of authoritative samples and never an extrapolation, the integrated distance
is exact, so feet cannot slide *by construction*. Prediction makes this worse (a corrected root produces a
distance discontinuity). And we get free lookahead: the render cursor sits 2.4 ticks behind the freshest
delivered tick, and `EntityTrack` already *holds* that freshest pose — so at draw time we know the
authoritative position up to 120 ms in the future of what we are drawing. Foot-plant selection,
anticipation and turn lean normally have to guess at that; we can read it. That is not prediction — it is
delivered, committed state we are choosing to render later. But `sample` only returns the clamped cursor
pose, so the lookahead is not exposed anywhere; that API decision should be made deliberately and kept
velocity-free in shape.

**Cloth is 100% ours to write** — verified zero matches for "cloth" across `bevy_pbr`, `bevy_render` and
`bevy_mesh`, and no IK or ragdoll anywhere in `bevy_animation`/`bevy_transform`. ~200 lines of pure math
in Tier-A sidesteps the dependency decision entirely and is coverage-visible and deterministic. Two
non-obvious constraints: gravity is radial and per-frame (`FollowCamera::up` is already annotated
"world-Y now; planet-radial later"), so the solver takes a `gravity_dir` parameter from day one; and
under packet loss a remote track *freezes* rather than coasting, so a naive solver settles to rest and
then snaps when the stream resumes — it needs an explicit stale-pose policy.

**The rebase hazard is the biggest cornering risk in this cluster.** Verlet stores `prev_pos` and derives
velocity as `pos − prev_pos`. A frame change collapses the interpolation window; the lattice will
re-centre at P4/P5; a floating origin is owed. Any of those makes `pos − prev_pos` a multi-kilometre
"velocity" and the cloth explodes to infinity in one step — and it will be diagnosed six months later as
a physics bug. The fix is trivial *if designed in*: every stateful client-side simulator implements
`rebase(delta)` applied atomically to both current and previous state, and by preference stores state in
frame-local coordinates so most rebases are no-ops.

### 4.7 Camera, controls, feel of motion (§2.7)

| Technique | Verdict | Bevy 0.18 anchor | When |
|---|---|---|---|
| Over-the-shoulder third-person framing | **defer** — it is an architectural change, not a camera one | none | P5 |
| Camera-relative WASD | **adapt** — already better: server resolves raw axes | `local_axes_from_movement` in `vd-core` | landed |
| Mouse orbit applied locally and instantly | **adopt** — and the exemption argument is worth writing down | `AccumulatedMouseMotion` | landed |
| Eased scroll zoom | **adapt** — forces the local-vs-server input split | `AccumulatedMouseScroll` (beware the `unit` change) | VU / P5 |
| Velocity-aware spring arm | **adapt** — filter state must be **frame-covariant** | `smooth_nudge` exists but is f32/glam; we write the f64 form | P5 |
| "Collision-free" spring arm | **reject** | none — Rapier not a dep yet | P5 |
| FOV widening under speed | **adapt** — speed must be a **shipped signal**, never a differenced pose | `PerspectiveProjection.fov` | P9 |
| "All transitions ease, no snapping" | **adopt** — and enumerate where our snaps hide | — | now |
| Banked camera on turns | **adapt** — the up-frame math has a real singularity | `frame_from_up` is `from_rotation_arc(Y, up)` | now / P5 |
| Screen-space wind streaks | **reject** — our speed cue is real parallax | none | later |
| Subtle camera shake | **adapt** — view-only, deterministic, recorded in the manifest | none | P11 |

**Instant local camera rotation is exempt from the no-prediction ban, and the argument is precise.**
Prediction is the client computing a *server-owned* value ahead of confirmation, creating a reconciliation
obligation. View orientation is not server-owned — nothing in the world depends on it, no other player
observes it, so there is nothing to reconcile. Structurally: `input_system` applies `cam.apply_look(...)`
locally *and* ships the identical `Look` delta, so the local camera and the delivered entity orientation
are two integrations of the same input sequence from the same start. They agree by construction, not by
correction. Position stays interpolated and 120 ms late; orientation is instant. That split is what makes
input feel immediate while the world stays honest. Two hazards: in third person the avatar's rendered
facing lags the camera by ~RTT + 120 ms (≈180 ms, very visible — needs a declared cosmetic aim overlay);
and if the server ever *constrains* the view (a seat with a limited arc, a turret), the local camera has
already integrated deltas the server rejected, and there is no correction signal and no slew mechanism.

**A spring arm's filter state must be frame-covariant.** If the state is a world-space camera position,
any re-parameterisation of world space injects an enormous error: a floating-origin re-centre, a warp, or
a realm crossing whose composed positions differ by the parent's offset would make the spring "catch up"
over its whole time constant — the camera flying across a solar system in 0.3 s. Rule: every piece of
filter state is a vector relative to the anchor in the anchor's current frame, and a frame change applies
the exact `transfer_frame` rotation to all of it. A first-order exponential filter is strongly preferable
because its only state is the current value. Also: a fixed decay rate gives a fixed lag *time*, so lag
*distance* = speed × lag time — at 1000 m/s a 0.15 s lag puts the camera 150 m behind the avatar, outside
the ship it is meant to be inside. Derive the time constant so lag distance is bounded by a fraction of
the avatar's extent.

**There is a real singularity in the up-frame math today, and it is a specific place on every planet.**
`frame_from_up(up) = DQuat::from_rotation_arc(DVec3::Y, up)` is the *minimal* arc, with no roll control
and singular at `up = -Y`; its own doc admits the antiparallel case "resolves to a deterministic
perpendicular axis". On a spherical planet the radial up sweeps the whole sphere, so the camera's yaw
basis will swing wildly and flip as you walk through that region. Worse, there is no inverse of
`forward_in_frame` for arbitrary up — `forward_to_yaw_pitch` is the `up = +Y`-only inverse — so on an up
change we cannot re-solve (yaw, pitch) to preserve world forward, which means **every up change is
currently a snap**, silently, for any non-Y up. And `nav.rs`'s gimbal-pole guard tests `dir.y.abs()`, the
*world*-Y pole, so on a planet-radial up the harness guards the wrong singularity. Adding
`yaw_pitch_in_frame(up, forward)` plus a round-trip proptest is ~40 Tier-A lines now, versus debugging
"the camera spins at one weird spot on every planet" later.

**Animated FOV silently breaks two things.** Angular size is FOV-independent, so a widening FOV does not
change AoI *membership* — but it does change how many *pixels* the smallest in-AoI realm gets, and it
shrinks by a third going 60°→90°, exactly when you are moving fastest, i.e. during warp. So `θ_min` must
be paired with a reference FOV or bounded. Separately, our pixel verdicts reconstruct the camera from a
constant `FIT_FOV_Y = π/4`; animate the FOV and those verdicts do not *fail*, they become silently
**wrong** — a 45°→60° change scales every projected extent by 0.717.

### 4.8 Effects and the spell grammar (§2.8, §2.9)

| Technique | Verdict | Bevy 0.18 anchor | When |
|---|---|---|---|
| The bending grammar: continuous, no instant spawn/despawn | **adopt** — and phase off a **server start tick** | none; evaluate against the render cursor, never wall time | rule now |
| Effects as STATE with a start tick, not one-shot EVENTS | **adopt** | latest-wins pose bag, not the reliable `EventMsg` arm | decision now |
| Swept ribbon/tube meshes from a spline | **adapt** — spine from a shipped pose history | per-frame `Assets<Mesh>` mutation; AABB auto-recomputed | P6/P8 |
| GPU compute particles | **defer** — **user dependency decision** | **none — Bevy 0.18.1 ships no particle system** (verified) | P6+ |
| Refraction | **adapt** — cheap mode first | `specular_transmission`; `screen_space_specular_transmission_steps` default 1 | P6 |
| Chromatic dispersion | **adapt** — full-screen only, wrong scope per-object | `ChromaticAberration` in `effect_stack` | later |
| Depth-based absorption tint | **adopt** — free | `attenuation_distance` + `attenuation_color` + `thickness` | P6 |
| Flow-map normals, leading-edge foam | **adapt** — both blocked on the vertex seam | `uv_transform: Affine2` for a cheap scroll | P6 |
| Shed droplets with motion-blur streaking | **reject** the post route | `MotionBlur` doc: transparent objects cannot be blurred | — |
| Effects emit light that scatters *into* the surface | **adopt** — the highest payoff-per-line item | `PointLight` + `diffuse_transmission`; blocked by `unlit: true` shells | P5/P6 |
| Vortex holding stripped snow aloft | **adapt** | `FogVolume` — a real volumetric primitive, not a particle stand-in | P6+ |
| Pooling every transient | **adapt** — our churn is asset/ECS, not GC | `MeshTag(u32)` + `#[storage]` param row | seam now |

**Encoding effects as state rather than events is the answer to "a one-shot can be missed".** On a lossy
lane, a dropped event means the effect never happens for that player. As state ("this entity has been
emitting class C since tick T for D ticks") on the latest-wins bag, a dropped datagram costs a few tens of
ms and the next datagram self-heals — the same shape as the per-observer scene diff that already
self-heals. It also composes with AoI: an observer arriving mid-effect just sees the current state, no
replay. And it is the only encoding under which the eased envelope is *correct* — phase must be
`(render_cursor − start_tick)/duration`, so two observers who learn about the effect 40 ms apart still
render the same shape.

**Reliable is the wrong default for cosmetics.** D-4 currently folds all signal→client delivery onto one
reliable `EventMsg` arm; effects inheriting that by default means head-of-line blocking risk and an
O(casters × observers) reliable-bytes tax at the exact moment the frame is busiest. Two lanes: reliable
for signals with gameplay consequence, unreliable-state for cosmetic emissions. That is not an HR3
violation — one signal system with a per-signal delivery-class policy.

**One verified pass-ordering bug waiting for us:** *"Specular transmission is rendered before alpha
blending, so any material with `AlphaMode::Blend`, `Premultiplied`, `Add` or `Multiply` won't be visible
through specular transmissive materials."* Our realm shells are exactly `AlphaMode::Blend` + double-sided.
So a refractive effect cast inside a container realm would show the shell as *missing*. Not a tuning
problem — the shells must stop being blended geometry before refraction ships. Start at
`screen_space_specular_transmission_steps: 0` (environment-map refraction only, near-zero cost) rather
than 1 (one full-resolution colour copy per camera per frame).

**There is no particle system in Bevy 0.18.1.** Verified: grepping for "particle" across `bevy_pbr`,
`bevy_render`, `bevy_sprite_render` and `bevy_internal` returns only prose in `medium.rs`, `fog.rs` and
`volumetric_fog.wgsl`. Every primitive to build one exists (`ComputePipelineDescriptor`,
`PipelineCache::queue_compute_pipeline`, the `RenderGraph` `Node` trait — we already ship a custom node —
`ShaderStorageBuffer`, `#[storage]` bindings, indirect draw buffers). The ecosystem option is
`bevy_hanabi`, which is **not** a dependency and is **a proposal to research a new dependency, for the
user to decide** — I have deliberately not verified it against 0.18.1 and am not recommending it.

**Pooling translates, but not for the brief's reason.** Rust has no GC, so the 12 ms pause threat does not
exist — but our churn is real and different: `Assets<Mesh>::add` per transient is a `MeshAllocator` slab
allocation plus a GPU upload; `Assets<StandardMaterial>::add` per transient is a new bind group **and a
batching break** (our current `spawn_realm_box` does exactly this, per realm); and `mesh_from_prim`
collects two fresh `Vec`s per prim per call. The correct "pool" is: one material asset per effect family,
one mesh asset where geometry is fixed, per-instance variation via `MeshTag(u32)` indexing a storage
buffer. That pattern is not only the fast path — it is the mechanically kind-agnostic one, since all
per-effect meaning lives in a parameter row the shader interprets and there is nowhere for an
`if kind ==` to appear even by accident.

**Also cheap and currently wasted:** `sync_realm_boxes` calls `to_render_prims` for every realm box every
display frame — for a sphere that allocates a `Vec<Vec<[f32;3]>>` grid plus a 504-vertex `Vec<Vertex>`
(~12 KB) and then throws all of it away except three floats of translation. At 50 visible realms and
90 Hz that is ~54 MB/s of pure allocate-free churn for 150 useful floats. A `position_only(...)` sibling
for the move path fixes it in an afternoon. `draw_hud` separately re-runs the full interpolated view
sample plus six `format!`s every frame in *both* apps (this is D-14).

### 4.9 Performance engineering and warm-up (§3, §4)

| Technique | Verdict | Bevy 0.18 anchor | When |
|---|---|---|---|
| Zero allocations in the render loop | **adapt** — GC reasoning moot, churn real | `Local<T>` scratch; sample `rendered()` once/frame (D-14) | now |
| "No map/filter/reduce in hot paths" | **reject** — V8-specific; would fight HR5 | — | — |
| Throttled allocation-free overlay | **adopt** | `on_timer` run condition (Bevy's own overlay uses 100 ms) | now |
| Object pools for transients | **adapt** — a bounded GPU-residency cache | `Visibility` (hidden = skipped before frustum work, mesh stays resident) | VU arc |
| Pre-allocated typed arrays for GPU uploads | **adopt** — the most important §3 item for P4 | `RenderAssetBytesPerFrame` (default `None` = unlimited) | P4 |
| Babylon `freeze*` calls | **adapt** — Bevy change detection *is* the freeze, and we defeat it | `Mut::set_if_neq` (`Transform` derives `PartialEq`) | now |
| Thin instances for repeated geometry | **adopt** — already free; **verify**, don't build | `AUTOMATIC_BATCHING` is opt-*out*; `GpuPreprocessingMode::Culling` is the default | VU arc |
| Profile with real tools | **adapt** — Metal gives CPU time only | `RenderDiagnosticsPlugin` + `time_span`/`pass_span` | now |
| Frame-time graph with 1% low | **adopt** | `FrameTimeDiagnosticsPlugin`; our `percentile_unstable` | now |
| Force-compile every pipeline offscreen | **adopt** — trigger moves to the demand loop | no prewarm API; render once + poll `waiting_pipelines()` | VU / P4 |
| Warm every render target, run compute passes | **adapt** — per realm, forever, not once | atmosphere LUTs ≈1.15 MiB/camera; env cubemap ≈12.6 MB at 512² | P4 |
| "4 s load beats an instant load that hitches" | **adapt** — principle yes, loading screen never | — | now |
| Milestones with perf hardening last | **reject** the ordering | — | — |

**There is no partial mesh upload.** Any `Mesh` mutation raises `AssetEvent::Modified`, which
re-extracts and re-prepares the *whole* mesh — Bevy's own comment says "in future we could also consider
partial asset uploads", i.e. there are none today. For destructible voxel terrain that means one block
edit re-uploads an entire chunk mesh: a greedy-quad 62³ chunk at 20k vertices × 24 B ≈ 480 KB per edit,
and a handful of edits across visible chunks per frame is megabytes per frame.
`RenderAssetBytesPerFrame` is the throttle and defaults to **unlimited**, which is what we run today; it
guarantees at least one asset of forward progress per frame and governs meshes *and* textures.

**We defeat Bevy's own change-detection fast paths.** `sync_world` assigns `transform.translation` for
every dot every frame, and `sync_realm_boxes` for every box, even when the value is bit-identical. Any
`&mut Transform` deref stamps the change tick, which re-extracts and feeds the entity specialization
tick — and both `validate_cached_entity` (skip re-binning an unchanged entity) and
`specialize_material_meshes` (skip re-specialization) are built on those ticks. `set_if_neq` is the fix,
and the win lands precisely on the static far scene the visibility model says will dominate.

**Draw-call and triangle counts have no built-in** — verified: grepping
`bevy_render-0.18.1/src/diagnostic/` for `draw_call|DrawCall|batch_count|triangle` returns nothing. The
better answer for us anyway is to compute the triangle count **Tier-A from the snapshot**, since all our
geometry arrives as `MeshPrim` vertex arrays and we therefore know the count before it reaches Bevy: it
is a pure function, 100%-coverable, available to the agent as JSON rather than only as pixels, and the
in-visibility triangle count is a direct readout of whether the server's AoI rule is producing a sane
workload.

**One measurement trap specific to our capture app:** it runs under `ScheduleRunnerPlugin::run_loop(1/60)`,
which sleeps `wait − exe_time`, so wall-clock frame deltas are pinned at 16.7 ms and
`FrameTimeDiagnosticsPlugin` would report a flawless 1% low while hiding everything. A capture-mode gate
must measure app-update duration or run with no wait. Getting this wrong produces a gate that is green
and meaningless — worse than no gate.

**The good news about warm-up: `PipelineCache` never evicts.** Once warm, a pipeline stays warm for the
process lifetime. So warming is a one-time per-process cost **if and only if** the permutation set is
bounded (§3.5). And the atmosphere is the specific per-realm hazard: enabling `Atmosphere` on a camera
creates its LUT set on first use, so adding the component when the player *arrives* at a planet is a
hitch at exactly the worst moment. Warm at connect and never add/remove the component at runtime —
`ExtractedAtmosphere` is also a `MeshPipelineKey::ATMOSPHERE` bit, so toggling it re-specializes every
material in the scene.

### 4.10 Working method and quality gates (§0, §5–§9)

| Item | Verdict | When |
|---|---|---|
| "Visual quality IS the product" | **adapt** — true for Tier-B, where the quality bar *is* the spec | now |
| "Placeholder is a defect, not a stepping stone" | **adapt** — our version is *gated for replacement*, with an expiry | now |
| "Break the requirement if it makes it prettier" | **reject** — scope it to inside `client-render` only | now |
| `DECISIONS.md` (one-line deviation log) | **adopt** — a *different* artefact from `DEFERRED.md` | now |
| Settings + perf overlay built early | **adopt** | now |
| Frame-time graph with the 1% low | **adopt** | now |
| Draw-call / triangle counts | **adapt** — no built-in; compute Tier-A | P4 |
| Per-effect toggles + quality presets | **adopt** — and pin a deterministic **gate preset** | now / P4 |
| Live art-parameter sliders | **adapt** — three legal homes, one forbidden | before P4 |
| Explicit frame budget in `PERF.md` | **adapt** — budget in code, measurements in the run manifest | before P4 |
| Screenshot every milestone, **commit** them | **adapt** — `/runs/` is gitignored; add a curated gallery | before P4 |
| "Do not proceed from an ugly milestone" | **adopt** — a recorded human verdict per slice | P4 |
| Visual acceptance checklist | **adopt** — machine/human columns + a graduation path | now / P4 |
| "Build, don't test-loop" | **reject** — keep "look at your output constantly" | now |
| "Replace, don't patch" | **adopt** — affordable only while techniques live below the seam | ongoing |
| Vendor assets + `ASSETS.md` | **adapt** — we already ship untracked third-party blobs | now |
| One module per domain | **adapt** — split the 1251-line `lib.rs` before P4 | before P4 |
| "No HUD, ever" | **reject** — our HUD is an instrument the gate is calibrated against | — |

**A slider is both an affordance and a storage location, and the storage half is where "AAA look on
thousands of seed-derived planets" dies.** Three legal homes: (a) a named field in ONE render tuning
struct — the `ClientInterpTuning` pattern, with a `DEFAULT` const and a test pinning it; (b) a
*coefficient inside a seed-derivation function*, where what gets committed is the coefficient plus a
byte-identity test, never a per-planet output; (c) an explicitly cosmetic-and-local client preference.
The forbidden fourth is any value the server owns — and the brief's own example gives the trap away: "sun
angle" is an art knob for a snow field and `f(seed, universe_tick)` for us, so a sun-angle slider is a
debug time-scrub, not an art parameter. Same for "deformation depth" and "refill rate". Gate the panel
behind the dev-control feature the way the dev-control listener is (absent from release builds, not
merely disabled), add a "dump current tuning" button so a tuned value is *committed* rather than left in
one developer's prefs, and record the active tuning in the capture manifest so a screenshot is
reproducible.

**`DECISIONS.md` is not `DEFERRED.md` and conflating them is actively harmful.** `DEFERRED.md` entries
are debts with a due date and a status; a decision log records "we chose A over B, deliberately, and we
are not revisiting it" — no debt, no status, no phase. The evidence we need one: the double-sided
container material, the 1284-not-1280 capture width (chosen to force the 256-byte row-padding path), and
the star-sphere radius rationale are all load-bearing, non-obvious decisions currently living *only* as
code comments in a coverage-exempt file. In six months someone will "fix" `cull_mode: None` and containers
will vanish from the inside again — which is exactly the bug that was already hit once.

**A quality preset may change only HOW the in-visibility set is shaded, never WHAT is in it.** AoI is
server-authoritative and one generic rule; a client-side "reduce draw distance" preset re-introduces
client-side visibility, makes two players see different worlds, and is a cheat vector. It will be
proposed as the obvious knob during the first frame-rate crisis, which is why it should be pre-decided.

**We already ship untracked third-party binary assets.** `tonemapping_luts` `include_bytes!`es three
KTX2 lookup tables, and `default_font` embeds `FiraMono-subset.ttf` — both because we deliberately take
bevy's full default features. There is no `assets/` directory, no `ASSETS.md` and no `LICENSE` in the
repo. A `docs/THIRD_PARTY_ASSETS.md` with one line per non-code blob (what, whence, licence, embedded vs
vendored) plus a written "no runtime network fetch of any asset, ever" rule costs an hour. The no-CDN
rule matters more for us than the demo because we deploy to k3d — a realm that spins up on demand cannot
have a startup dependency on the public internet.

---

## 5. Where the brief is wrong for us

Ranked by severity. Each: the trap, then what to do instead.

### 5.1 The shared terrain-state render target as the source of truth — **critical**

The centrepiece of §2.3, and wrong four independent ways: it is a *client-local* mutable world state (two
clients diverge by construction — one player's trail simply does not exist in the other's texture); it is
planar and player-centric (a toroidal wrap in a global XZ plane is invalid on a sphere and singular at the
poles); it assumes *one* buffer (standing on a ship's deck on a planet means two surface-bearing realms
on screen at once); and "never rebuild from a list of past events" is the exact inverse of our recovery
story, which is analytic base + delta overlay reconstructable from seed with zero networking. It also
breaks the renderer seam: "is this surface compressed" would live in a GPU texture the Tier-A harness
cannot read, so no visual gate could ever assert it.
**Instead:** §3.6's three tiers. The *read* path (shader samples surface state) survives verbatim; the
*write* path splits into "server edit" and "local decoration". Sizing note if the cosmetic tier is ever
built: 4096² R16F is 33.6 MB for **one** channel; 2048² at 4 cm covers the same 82 m for 8.4 MB and 4 cm
is still far below what reads at eye height. And the residency radius falls out of the law we already
have: an 82 m patch subtends θ_min at `82 × 14.301 ≈ 1172 m`, so the buffer is live inside ~1.2 km of the
anchor — same formula, no kind branch, no new constant.

### 5.2 "Do not build a test suite" — **critical**

It *sounds* like the standing "no shortcuts, don't stop at it-works". It is the inverse of our governance.
Our render crate is coverage-exempt *precisely because* the visual harness proves it — the documented
division of labour is that WireMonitor proves bytes and HR6 proves rendering, duals not overlap. Remove
the harness discipline and the one crate llvm-cov deliberately does not look at has no proof at all. The
brief's own §8 criteria ("sustains 90 FPS with 1% lows above 60", "no hitch on the first cast") are
literally test assertions it then declines to automate.
**Instead:** keep the half that is excellent — screenshot every milestone, look at your own output
constantly, do not move on from an ugly milestone — and reject the anti-test position. Our rhythm is
already better: the user hand-plays a runnable command, and only after they confirm it works in-game does
the automated gate land, and only then a commit. One genuine warning to take: running the full `just gate`
battery after every shader tweak makes tuning impossible, so name a fast render loop (`just render-smoke`
alone) as blessed during art iteration.

### 5.3 "Break the requirement if it makes it more beautiful" — **critical**

Correct for a sole-engineer demo, a licence to violate binding laws for us — and the laws it hits first
are the ones invisible in a screenshot: sealed shards, the agnostic client, no prediction, server AoI.
Every one can be broken to make a prettier frame *today*, and the damage shows up at planet 200.
**Instead:** a two-tier permission. Inside `crates/client-render` (Tier-B, no wire, no Tier-A types) the
engineer has full latitude. Anything touching the `MeshPrim`/`RenderSnapshot` seam, the wire, or what is
drawn vs not drawn takes the normal design path. That boundary is already typed shut — `vd-client` and
`vd-client-harness` have no Bevy dependency — so the compiler enforces it rather than discipline.

### 5.4 The LOD / impostor / binary-render-rule tension — **critical, and NOT resolved**

This deserves the most honest treatment in the document, because the brief pushes hard on clipmaps and
impostors, our standing preference bans them, and **our own repo contains both sides**:
`docs/design/slice_3_renderer.md:45` says "LOD (never)" and `crates/client/src/realm_scene.rs:546-548`
says "NO LOD"; `crates/wire/src/intershard.rs:303` reserves a WHAT lane for "terrain/constructions at the
LOD the observed realm controls" and `intershard.rs:656-659` reserves a `coarsen_level` precision ladder
(always 0 today). Also worth stating plainly: the exact phrase "binary render rule" appears **nowhere in
the repo** — `grep -rn "impostor" docs/ scripts/ crates/` returns zero hits. It is a user standing
preference, and must be cited as one.

The word "LOD" is being used for at least five different objects. Sorted:

| Object | Is it the banned thing? | My read |
|---|---|---|
| Mesh proxies / impostors / billboards standing in for a real object | **Yes** — this is the thing | Reject, and the arithmetic agrees: on a 150 km planet the ground horizon is 714 m, there is no 30 km ridgeline to fake |
| Texture **mipmapping** | Almost certainly not | Mandatory. Without mips, high-frequency detail normals alias into crawling noise at *any* distance and the glint term is unusable. A literal reading that banned mips would guarantee shimmer |
| Distance-blended **detail normal scales** | Probably not | Nothing is substituted, no proxy exists, the mesh is identical — only which detail octave dominates changes, and it must change or the fine octave aliases |
| **Parameter** LOD (shell count → 0, cloth iterations → 0, animation rate → 5 Hz, same mesh, same material) | Probably not | Same mesh, same material, no proxy — the count simply reaches zero |
| **Server-authoritative** content/precision tiers (the reserved WHAT lane, `coarsen_level`) | Different object entirely | The observed realm chooses what resolution it publishes; the client meshes exactly what it is handed, one level, kind-agnostically |
| Shadow **cascades** | Genuinely ambiguous | Four resolution tiers of the same shadow, cross-blended. There is no non-tiered way to shadow a 2 cm feature at 1 m and a mountain at 5 km from one map. A literal reading means shadows are simply **off** |
| Chunk **detail tiers** for terrain | The real question | 417 chunks at walking eye height; ~1.25 **million** at 5 km altitude. Single-tier is correct for a walking player and structurally impossible for a flying one |

**The choice and what each costs.** Option A — hold the rule literally: no chunk tiers, no cascades, no
distance-blended detail. Cost: shadows are off; terrain draw distance is capped at whatever a single tier
affords, which is fine at ground level and impossible the moment P8 puts you in a ship; and the far-field
cut is a hard pop unless the atmosphere has already faded it to sky (which on a small planet it can, and
that is the seamless-legal way to hide it). Option B — the rule bans *substitute geometry only*: cascades,
mips and detail-scale blending are permitted, and terrain tiers become a *server-authored* decision on the
reserved WHAT lane rather than a client mesh ladder. Cost: one `detail_tier: u8` on the chunk address
(always 0 through P7), plus the honest admission that a downsampled tier must fold in the delta overlay
or a player's dug tunnel disappears at 200 m and reappears at 100 m — which is the genuinely hard part of
voxel LOD and which the brief, having no destructibility, never confronts. Option C — decide only the
cheap insurance now: reserve `detail_tier` and the vertex-format headroom, hold the rule for P4, revisit
at P8.

Bevy's only built-in anti-pop mechanism is `VisibilityRange { start_margin, end_margin, use_aabb }` — a
distance-banded dither crossfade, costing one pipeline bit
(`MeshPipelineKey::VISIBILITY_RANGE_DITHER`). Its own doc table names "Billboard *imposter*" as the far
band, which is why one reading rejects it outright; but the mechanism itself is just a crossfade, and used
as an alpha ramp on the *same* mesh at the AoI edge with no impostor arm it is the only thing that makes
the AoI edge not-a-pop. **This is a user decision, not mine.**

### 5.5 "Persistent" means ninety seconds — **critical**

The brief's "deformation is persistent" means it survives the tab. Ours must survive a shard restart, a
realm spinning down and back up on the demand loop, a *different* player who was never present when the
mark was made, and a client that connected an hour later. Those are four different mechanisms and none is
a GPU texture. Related: the brief's diffusion/decay is a stateful GPU process at whatever framerate this
client happened to get — if a mark's presence ever matters to gameplay, its decay must be a deterministic
function of tick evaluated server-side.
**Instead:** §3.6 plus closed-form decay. On realm spin-down the cosmetic buffer is freed and the
authoritative field is already durable; on spin-up the field loads with the realm; the demand loop is
untouched. Also: a per-realm storage budget with deterministic oldest-first eviction, decided *with* the
schema — a player running circles for a day otherwise dirties every tile in a realm.

### 5.6 The geometry clipmap forecloses caves — **critical**

Covered in §4.1. Restated here because it is the one architectural foreclosure in the brief that is easy
to adopt by accident and expensive to reverse: a clipmap is 2.5-D by construction, and choosing it at P4
means choosing a heightfield world.

### 5.7 Bevy's atmosphere is wrong on a sphere — **critical**

The shader says so itself (quoted in §3.4). This is not a nuisance, it is a scoping decision: adopt the
*method* (Hillaire LUTs, two radii + ground albedo + a medium asset — all seed-derivable) and budget real
work at P4 to fork or extend the shader so it takes a planet **centre** and an **up** vector. Or lock the
camera-anchored planet-tangent frame (§3.4) and use it unmodified, accepting one atmosphere per camera and
no limb-from-orbit until P10. Either is defensible; silently assuming it is a configure-it item is not.

### 5.8 Bespoke materials per surface state and per effect — **critical**

The brief has ~10 known effects and a loading screen. If our *pipeline set* scales with *content variety*,
hitch-free on-demand streaming is impossible in principle. See §3.5.

### 5.9 The post chain is not a shopping list — **high**

Each component drags architecture: SSAO needs `Msaa::Off` + depth + normal prepasses; SSR needs *deferred
rendering* for the whole camera; TAA needs `Msaa::Off` + depth + motion-vector prepasses and handles
alpha-blended meshes badly; forward decals need `DepthPrepass`. And clustered decals — the higher-quality
alternative — require bindless textures and their doc says they "presently can't be used on WebGL 2,
WebGPU, macOS, or iOS", i.e. **not on the dev machine**.
**Instead:** decide the prepass policy and the AA lane as one architectural decision (§3.8), not as a
sequence of individually reasonable component additions.

### 5.10 Magic numbers dressed as expert defaults — **high**

"sub-10 cm vertex spacing", "~2 cm texels", "60–100 m", "4096²", "20–40 shells", "4–6 dynamic lights",
"visible after 60 seconds", "11.1 ms". Each encodes an assumption rather than a measurement: 60–100 m is
the size of the demo's world; 60 seconds is the length of the demo; 11.1 ms is a display refresh nobody
has chosen for us. They read as expert defaults, which is exactly why they get pasted.
**Instead:** every spatial constant is derived — from block size, from the one visibility constant, from
the chunk dimension, or from a measured budget — and lives in one tuning struct with its derivation in the
doc comment. Treat any number from the brief as a hypothesis to re-derive.

### 5.11 A per-effect dynamic-light allowance — **high**

"4–6 lights" is a fixed-content number. A shipyard with forty player-built ships each with thruster glow
blows it instantly, and the failure is multiplayer-visible: if the client drops "some" lights and the
selection rule is not deterministic and shared, two players in the same place see different scenes, and
one may be missing the light showing a hazard.
**Instead:** a per-frame budget plus a deterministic kind-agnostic selection rule — importance by
angular size / intensity / distance, the same shape as the AoI law — computed in Tier-A so the harness can
assert it, with hysteresis so lights do not flicker at the budget boundary.

### 5.12 Per-place, per-asset art direction — **medium**

Hand-placed sparse rock outcrops "to give the horizon something to say"; one hero robe finished to one
standard; one HDRI sky. All are per-place authored art, which is what "thousands of bodies from seed, no
hand-authored per-planet art" forbids. The transferable asset kind is *scale-free and place-free*: tiling
detail normals, grain, blue noise, LUTs, combined by seed-derived parameters.
**Instead:** ask "what must the seed produce for the silhouette to read at mid-distance", not "which rock
do I place". And note the inverse risk the brief does not have: with emergent generation you cannot "keep
them sparse" by hand, so an over-eager erosion parameter makes every planet a spire forest — which needs a
*statistical spread assertion over N seeds*, a concept the brief has no equivalent of.

### 5.13 "No fallbacks" / target one machine — **medium**

Right for a demo, wrong for a shipped MMO — but the obvious correction (feature-detection branching) would
re-create the permutation explosion. Note we are accidentally close to the brief's posture already, for a
different reason: our GPU gates are local-only with no software adapter path.
**Instead:** a small enumerated set of quality tiers chosen *once at startup*, each a complete
pre-warmable pipeline set. Not per-feature runtime branching, not a single hardcoded config.

### 5.14 Performance hardening as the last milestone — **medium**

The brief's milestone order defers perf to a polish phase. That is the exact structural inversion this
rebuild exists to escape. For us, hitch-freedom under demand-driven spin-up is a **correctness** property
of the seamless rule, not a performance nicety — a compile stutter at a realm boundary *is* the visible
seam we promised does not exist.
**Instead:** keep the per-milestone beauty gate; move instrumentation and the warm-up invariant to the
front, because they constrain the architecture rather than tune it.

### 5.15 A stable glint hash keyed on world position — **medium**

The craft advice is right and the instinct (stability) is exactly ours; the trap is that the brief only
needs stability against TAA jitter along a continuous camera path. Ours must also survive a floating-origin
rebase, a frame change, and two players agreeing. A hash keyed on camera-relative or raw float world
position makes the entire world's sparkle jump on every rebase.
**Instead:** a general rule for *all* procedural surface detail (grain, tiling offsets, scatter
placement), not just sparkle: key on a stable, rebase-invariant, integer-derived coordinate.

---

## 6. What the brief never mentions but we need anyway

Ranked by cost-of-lateness.

**1. Camera-relative rendering / floating origin — must be decided before P4's coordinate contract.**
Bevy provides nothing: `Transform` is f32, `GlobalTransform` is an f32 `Affine3A`, and a grep for
"floating origin" / "camera-relative" across `bevy_render`, `bevy_pbr` and `bevy_transform` returns
nothing. So the rebase happens entirely in Tier-A before anything reaches Bevy. f32 has a 24-bit mantissa,
so at 1e7 m one ULP is ~1.2 m — which is exactly the documented jitter ceiling; camera-relative caps it at
0.24 mm at 2000 m. Do it **continuously**, not as discrete threshold rebases, because a discrete rebase
makes every mesh's `Transform` jump by −Δ while its `PreviousGlobalTransform` still holds the old origin —
one frame of globally garbage motion vectors.

**2. Nested moving frames on the render path — the composition engine is not built.**
`world_pos` is one hop: planet/system/galaxy/area/station frames map to identity and only `ShipLocal`
composes, one hull deep. The seamless plan states outright that the recursive engine "is not built". And
`PrimTransform` carries **no rotation**, so a rotating planet, a banking ship or a spinning station cannot
be drawn correctly today. Both are slice-zero preconditions, i.e. **now**.

**3. Voxel chunk meshing and remeshing under a byte-identity DoD — P4.**
Unowned: what re-meshes when one block changes (the chunk plus up to three neighbours); where meshing runs
so it never touches the render thread; how a mesh handle is swapped without a one-frame hole (a hole is a
pop); how many chunk mesh create/destroy cycles per second before `MeshAllocator` slabs fragment; whether
chunk meshes keep a CPU copy (`RenderAssetUsages` defaults to `MAIN_WORLD | RENDER_WORLD` — RAM *and*
VRAM, a memory-doubling default nobody has chosen). And the unusual one: byte-identical terrain across two
target-CPU builds is a *determinism* requirement sitting on a render-adjacent path, easy to violate with an
innocent f32 shortcut, and far more expensive to retrofit than to build in.

**4. How a kind-agnostic client renders content it has never heard of — before P4 materials.**
Our most architecturally distinctive rendering problem and the brief's exact inverse. The signal-bag half
of the streaming contract is **unbuilt** (no `SignalBag` anywhere in `crates/`), and the client already
carries three deliberately kind-keyed Tier-A helpers (`role_hsv`, `frame_of_realm`, `shape_of`), justified
today as cosmetic-only. At P4 the pressure — "planets should look like planets" — will push a kind match
into the *material* path, where it stops being cosmetic. Both fallbacks are closed: magenta is a hard gate
defect, and "unknown renders as nothing" violates seamlessness. Nobody has designed the minimal open-ended
vocabulary (a seed + a small parameter block + a material-class id from an append-only registry) that lets
an *old* client draw something plausible for a *new* thing.

**5. Surface → orbit → interstellar as one continuum, and the dynamic-range problem inside "stars always
visible" — VU arc / P4.** The same star must be a background pixel, then a disc, then a light source, then
the world you stand on, with no mode switch. Our starfield is 2800 unit spheres on a follow-sphere and
exists only in the *windowed* path. `Skybox { image, brightness, rotation }` is the right primitive (and
the atmosphere is explicitly designed to composite over a skybox, so a starry sky shows through by time of
day). The exposure half is §3.12. Honest tension to resolve: a physically correct daytime sky *will* wash
stars out, and our starfield is opaque depth-writing geometry, so the sky pass will attenuate it by
transmittance. Which wins — the law or the physics?

**6. Depth, culling and the far plane at planetary scale — before P4.**
Good news worth knowing so nobody solves a non-problem: infinite reverse-Z + `Depth32Float` is close to
the best available setup and the depth *buffer* is not our bottleneck. Three real items: `far` is a
*culling* bound, not a clip bound (so 6000 is a culling decision masquerading as a clip decision, and it
will silently cull real content as scale grows); the actual precision failure is vertex position in f32
before projection (item 1); and layered near/far cameras are available via `Camera::order` +
`ClearColorConfig::None` but multiply passes and interact badly with one post chain.

**7. Anti-aliasing of voxel edges, and gates hostile to the cure — P4.**
Voxel worlds are the worst case: long straight edges at grazing angles, large coplanar faces, high-contrast
blocky silhouettes. MSAA does nothing for specular aliasing; the standard cure is temporal, and temporal
collides with our deliberate discontinuities *and* with our gates (a still scene must currently differ by
zero pixels; TAA's 8-phase Halton jitter with a 1.5%/frame blend floor means a static scene converges to an
**8-periodic orbit**, never a fixed point). So `differing_pixel_count == 0` should be landed *now*, while
the scene is still post-free and it can genuinely hold; after TAA it is permanently unlandable.

**8. Many players and content that arrives mid-flight — with the wire work, not at P11.**
Every spawn is a potential first-use pipeline compile, a mesh upload, and a material allocation. D-9
already records that snapshot emit is whole-realm broadcast with byte volume unproven; there is no
equivalent render-side number at all — nobody has measured draw calls, mesh churn or frame time with a
hundred avatars plus dozens of player-built ships co-located. Also verified: `animate_targets` iterates
**every** `AnimationTarget` every frame with no visibility gate, so 200 players means ~36k curve
evaluations and 12k transform writes per frame for content nobody can see.

**9. Interiors — P8, but the frame question is now.**
Interior lighting that does not receive the star through a hull; shadows inside a *moving* frame; a
distinct ambient per interior. Bevy's hooks are per-view or capped: `AmbientLight` is a per-**camera**
component (so ambient is a property of the *view*, not the place) and light probes are capped at 8 per
view with an explicit comment that the fragment shader does a linear search. Note
`AtmosphereEnvironmentMapLight` may be attached to a `LightProbe`, which is the hook — but a probe is a
Transform-positioned region, so it belongs to a realm and must ride that realm's frame, making it yet
another consumer of the composition engine.

**10. A pop-in / hitch DETECTOR — now, and the cheapest item on this list.**
"No pop" is law and there is no test for it. We have the primitives (`differing_pixel_count`,
`content_present_fraction`, luminance bounds, the magenta and NaN sentinels) but no assertion of the form
"across this recorded sequence, no frame differs from its predecessor by more than X%" — which is the
actual pop detector and would *also* catch a pipeline stutter, a mesh-swap hole and a scene-swap flash. It
is the arc's own acceptance criterion and currently unassertable.

**11. Planetary curvature, the horizon, and per-position "up" — P4.**
The horizon is the natural far-cull bound, far better grounded than any chosen far-plane number, and we do
not use it. "Up" becomes per-position, so shadow cascade fitting, height-based fog and the atmosphere all
need a local frame. And the ground occludes half the sky — the cheapest occlusion culling available, which
nothing exploits.

**12. Day/night from real orbital math, and the time-multiplier trap — P4.**
The sun direction is closed-form from seed and tick and must be *identical* on server and client, making it
a determinism-bearing quantity that is also the most important render input. The trap: per-realm subjective
time dilates **occupant movement only** and must never touch the parent-authored orbit — so a realm with a
10× multiplier must have **unchanged day length**. That is exactly the kind of thing wired correctly once
and broken by an innocent refactor, so it belongs in the design doc as an explicit prohibition.

**13. Caves and the underground lighting regime — P4/P5.**
Sky and aerial perspective must switch *off* underground, and Bevy's atmosphere has no underground concept
— `clamp_to_surface` pushes the view back up, so it will happily render sky for a camera inside a mountain.
AO and local lights become primary, which changes which post components we need.

**14. The visual language of the signal and economy layers — design with the signal bag.**
Any visualisation must be purely **additive**, **absent-safe** (correct with the whole layer off, per the
one-way dependency law), and **kind-agnostic** (it cannot branch on what a signal *means*). That is harder
than any visualisation problem in the brief, where each effect knows exactly what it is. Retrofitting
"looks right with the layer off" is exactly the coupling the one-way rule exists to prevent.

**15. Colour calibration and a generative colour model — P4.**
The brief's colour criteria are real and checkable, and the discipline transfers. What is missing is the
framework: no tone-mapping choice made, no HDR, and no per-realm colour authority. Crucially, colour must
be *seed-derived* across thousands of bodies — a human eye cannot grade a thousand planets, so the brief's
"tune by eye until it looks right" cannot give us the second half.

**16. Reflections and water — after the forward/deferred decision.**
SSR's own doc: deferred-only, "can only reflect objects on screen", "performs no roughness filtering". The
on-screen-only limitation is fatal for a camera that constantly pans across a planet limb — every
reflection pops as things leave the frustum, which is the pop rule. Bevy's own named alternative
(`LightProbe` + `EnvironmentMapLight`) is cheaper and a far better fit for thousands of seed-derived
bodies, but shares the 8-probe budget with interiors.

**17. Multiplayer agreement about what is LIT — before the block/signal work.**
If shadow or daylight ever gates gameplay (solar power on a block, stealth, crop growth, thermal load), the
authoritative test must be server-side and quantized to an integer grid, with the GPU shadow being only a
picture of it. Deciding this after the block system exists is expensive.

**18. Photosensitivity and accessibility — before the warp capstone.**
A warp with a system growing from a point, plus any flash on crossing, is a real obligation for a shipped
MMO. Every comfort toggle (shake, FOV kick, motion blur) changes the view or projection and therefore what
the gates see — so gates must run under an explicitly *recorded* accessibility profile, not whatever the
developer's local settings happen to be.

---

## 7. Open questions for the user

1. **How big is one voxel, in metres?** It is stated nowhere in `docs/design/`, and every chunk-count,
   shadow-texel, LOD-threshold and parallax number in this document scales with it.

2. **Blocky greedy quads forever, or a smooth iso-surface for natural terrain?** This is a one-way door
   that decides the mesher, the vertex format, the collider path, and whether triplanar costs one sample
   set or three — and nothing in the design docs decides it.

3. **Does "no LOD systems / binary render rule" ban mipmapping, distance-blended detail-normal scales,
   shadow cascades, and parameter LOD (shell count → 0 on the same mesh) — or only mesh proxies and
   impostors?** §5.4 lays out the options; I need a ruling, not an inference, because the answer changes
   whether shadows exist at all.

4. **Exposure: physically derived closed-form from the star (identical for every player, reproducible in a
   capture) — accepting that every emissive becomes an absolute seed-derived radiance — or Bevy's
   `AutoExposure`?**

5. **MSAA or TAA?** They are mutually exclusive. The choice determines shadow filtering, whether SSAO is
   reachable, whether alpha-to-coverage (fur, foliage) is available, and what our pixel assertions can
   claim. We are on MSAA 4× today purely by Bevy's default.

6. **Do you accept a declared cosmetic-local tier in which two players legitimately see different cloth
   motion, different spray, and different individual footprints — while any mark another player could act
   on is server-owned?** Refusing it means secondary motion is either deterministic-from-shared-state or
   absent.

7. **Two dependency/feature decisions, both yours:** (a) enable bevy's `experimental_pbr_pcss` feature for
   soft shadows — not in our build today; (b) particles — build our own GPU compute system (all primitives
   exist) or research an external crate (`bevy_hanabi` is the obvious candidate and is **not** a
   dependency; I have not evaluated it against 0.18.1).

8. **Are we willing to ship authored texture data** — CC0 PBR scans, BC-compressed, vendored with licences,
   using the KTX2/zstd path bevy's default features already include — **or must every texel be synthesized
   from seed on the client?** Fully procedural is stronger for thousands of planets but moves real GPU cost
   to realm spin-up, which is exactly where the seamless law says we must not stutter.
