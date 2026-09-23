# The fine ground — the models, the crates, and what AAA actually ships

Written 2026-09-22, after the owner looked at the belt stand and the river stand beside a frame of
*Crimson Desert* and asked: **"Is it worth to find models that already solved all those issues? Maybe
there are even crates that we can re-use instead of reinvent from the beginning? I would prefer not
assume how it will look, but have proven models that we can apply and continue the development."**

This report answers the FINE scale only — the band between about **30 m and 8 km**, under one macro
node. The macro solve above 8 km, the lakes, the ice and the grid scratches belong to
`lakes_and_landscape_models.md` and to ruling B2; nothing here repeats them.

**Method.** I read the code and the papers. **I ran nothing.** Every number I call MEASURED comes from
a paper's own table or from a ruling file. Where I could not open a source I write UNVERIFIED. The
session's web-search budget was already spent, so this survey is built from papers fetched directly
and from the crates.io, GitHub, HAL and OpenAlex APIs — never from a search snippet dressed as a
reading.

---

## 0. THE SHORT ANSWER

**Yes. The problem is solved, twice, in public, and both solutions take our artifact row as their
input.**

1. ★ **Schott, Galin, Guérin, Peytavie & Paris 2024, *Terrain Amplification using Multi-scale
   Erosion*, ACM TOG 43(4), SIGGRAPH**, [10.1145/3658200](https://doi.org/10.1145/3658200),
   [PDF](https://hal.science/hal-04565030/file/2024-MultiScaleHydro-Author.pdf) (read in full). It
   takes a LOW-RESOLUTION terrain and amplifies it **×8 to ×32** by repeating one loop at each new
   resolution: upsample, stream-power erosion, thermal stabilisation, deposition. Its motivating
   sentence is our defect, verbatim: *"existing erosion simulations generate details at the original
   terrain resolution and often **carve drainage patterns at the size of one cell**."*
2. ★ **Cortial, Peytavie, Galin & Guérin 2020, *Real-Time Hyper-Amplification of Planets*, The Visual
   Computer 36(10), CGI**, [10.1007/s00371-020-01923-4](https://doi.org/10.1007/s00371-020-01923-4),
   [PDF](https://hal.science/hal-02967067/file/PlanetSubdiv-author.pdf) (read in full). It takes
   **control maps at about 50 km** holding *"the elevation, presence of large-scale water bodies and
   landforms types"* and subdivides them to **50 cm** on an **Earth-sized planet, in real time on one
   GPU**. ★ It has a section headed **Determinism**, and the rule it states is the rule SL10 needs.

★ **THE FINDING THAT MATTERS MOST: the INPUT both methods want is the list our artifact row already
carries.** Ten bytes per macro node — the eroded height `Z`, the water level, the D8 receiver with the
facies bits, **the discharge as a quantised logarithm**, three climate bytes and the rock province
(`crates/terrain/src/artifact.rs`, `ARTIFACT_VERSION = 6`). Cortial's control maps are elevation,
water bodies and landform type. Schott's inputs are a coarse height and a hardness field. **We are not
missing the data. We are missing the amplification step that reads it.**

★ **AND THE FIRST ONE SHIPS ITS CODE UNDER MIT.** [`H-Schott/MultiScaleErosion`](https://github.com/H-Schott/MultiScaleErosion)
— *"Code release for Terrain Amplification using Multi-scale Erosion"*, **MIT, last pushed
2026-09-15** (verified through the GitHub API). C++ with four GLSL compute shaders: `erosion.glsl`,
`thermal.glsl`, `deposition.glsl`, `spe_shader.glsl`. **Its uniforms carry the paper's own numbers**,
read out of the shader source: `flow_p = 1.3`, `k = 0.0005`, `p_sa = 0.8` (the drainage exponent `m`),
`p_sl = 2.0` (the slope exponent `n`), `max_spe = 10000`. **The recommendation of §5 has a permissive,
maintained reference implementation to be diffed against.**

★ **AND THE CHEAP CURE IS REFUTED BY ITS OWN AUTHORS.** The published way to keep a noise recipe and
merely align it to the coarse slope is Grenier, Guérin, Galin & Sauvage 2024, *Real-time Terrain
Enhancement with Controlled Procedural Patterns*, CGF 43(1),
[10.1111/cgf.14992](https://doi.org/10.1111/cgf.14992),
[PDF](https://hal.science/hal-04360714/file/01.pdf) (read in full). Its limitations say, verbatim,
that it *"**cannot generate a global coherent river network**, a common limitation of **all
function-based elevation models relying on a fully procedural and parallel algorithm**"*, and that
*"the proposed gradient-aligned noise function **replicates linear patterns over almost flat
regions**."* **That last clause IS the belt stand.** A published paper names the picture the owner is
looking at, and names it as the known failure of the cheap cure. The noise recipe survives only as a
SKIN on ground a real solve has already shaped — never as the shape.

**Nothing can be ADOPTED, and two things can be HARVESTED.** There is **no fixed-point noise or erosion
crate in Rust**, and effectively none in any language; every survivable implementation is float, and
one Rust SIMD noise crate picks a different code path per CPU, which breaks the no-drift gate by
design (§3). But the recommendation's own code and the tiled flow accumulation are both MIT and
maintained. The value here is ALGORITHMS of a few hundred lines, which we must write on `Gi` anyway.

---

## 1. HOW AAA GAMES GET THAT LOOK

Every figure here comes from a slide deck or a doc page that was opened and read. Where a deck could
not be opened I write UNVERIFIED and make no claim.

### 1.1 ★ THE FACT THAT REFRAMES EVERYTHING: their world is 100 km², and it is hand-terraformed at half a metre

| game | world | heightfield resolution | who made the shape |
|---|---|---|---|
| **Far Cry 5** | ★ *"The Far Cry 5 world is **10 km × 10 km** in size"*, 160 × 160 sectors of 64 m | ★ *"We author terrain information at **half meter resolution**"*; per sector an R16 heightmap of **129 × 129** | ★ *"Users will **terraform** the terrain using Dunia editor tools"* — by hand |
| **Ghost of Tsushima** | 200 m tiles, *"~6M budget (~1300)"* → **about 52 km²** | *"~2.5M of **513×513 maps (~0.4 m/texel)**"* per tile | in-engine painting plus *"GPU baked procedural growth/texturing"* |
| **Horizon Zero Dawn** | streamed sections, size not stated | ★ **`Height_Terrain` 0.5 m, 16 bit** | maps *"All Generated. All Paintable"*, placement maps *"Extensively Hand Painted"* |
| **Ghost Recon Wildlands** | Bolivia, 11 biomes, 600 km of roads | virtual texture to *"10 texels per cm"* | a real DEM *"refined … with **World Machine**"*, then hand-sculpted in layers |
| **Unreal Engine** | largest RECOMMENDED Landscape **8 129 × 8 129 vertices**, 16-bit, ±256 m at the default Z scale | ≈ 1 m per quad (the XY scale is not stated in the docs — **UNVERIFIED**) | the artist |
| ★ **Flight Simulator** | ★ *"the world is big — **510.1 million km²** … **2+ petabytes of aerial images**"* | Bing quadtree; streamed LoD 19–20 = **0.30–0.15 m/px**, the tree/collision layer at LoD 15 = **4.8 m** | ★ **nobody. It is a MEASURED DEM.** *"**Limited manual edition** ⇒ **Semi-procedural systems**"* |

Sources, all read: [Far Cry 5 terrain, GDC 2018](https://media.gdcvault.com/gdc2018/presentations/TerrainRenderingFarCry5.pdf)
and [Far Cry 5 procedural world, GDC 2018](https://media.gdcvault.com/gdc2018/presentations/ProceduralWorldGeneration.pdf);
[Ghost of Tsushima streaming, GDC 2021](https://archive.thedatadungeon.com/ghost_of_tsushima_2020/documents/gdc_2021/ghost_streaming_gdc2021.pdf);
[Horizon placement, GDC 2017](https://www.guerrilla-games.com/media/News/Files/GDC2017_VanMuijden_GPUBasedProceduralPlacementInHorizonZeroDawn.pdf);
[Wildlands terrain, GDC 2017](https://web.archive.org/web/2018id_/https://ubm-twvideo01.s3.amazonaws.com/o1/vault/gdc2017/Presentations/WERLE_MARTINEZ_GRWterrainTechnologyTools.pdf);
[UE Landscape Technical Guide](https://dev.epicgames.com/documentation/en-us/unreal-engine/landscape-technical-guide-in-unreal-engine);
[Asobo, Designing the Terrain System of Flight Simulator, GDC 2022](https://www.asobostudio.com/files/inline-images/Designing_Terrain_System_Fuentes_Lionel.pdf).

★★ **READ THE TABLE AS ONE SENTENCE. Every game that GENERATES its ground works at 10 to 100 square
kilometres and authors it at half a metre by hand. The only game at planet scale does not generate
anything — it reads a satellite and streams two petabytes.** Our home planet is **595 million km²**.
**Unreal's biggest recommended landscape is ONE of our macro nodes.**

★ **And the data budget settles it numerically.** Horizon's world data costs **~4 MB/km² exclusively**.
At that rate our planet would need **about 2.4 PETABYTES (DERIVED)**. **The AAA pipeline is not
expensive for us; it is nine orders of magnitude out of reach. Every recommendation in §5 exists
because of that one number.**

### 1.2 ★ What they DO that we can copy: they bake a hydrology field and derive the dressing from it

Horizon Zero Dawn's world data is a stack of 2-D maps under a slide headed **"WorldData: Baked Maps"**:

| map | res | | map | res |
|---|---|---|---|---|
| **Height_Terrain** | **0.5 m, 16 bit** | | Placement_Trees / BlockBush / Undergrowth | 1.0 m |
| ★ **Erosion_Wear / Erosion_Flow / Erosion_Deposition** | ★ **0.5 m** | | Variance_RockColor / Foliage_Color / Lichen_Density | 1.0 m |
| Terrain_Cavity, Water_Flow, Water_Vorticity | 0.5 m | | Topo_Roads / Topo_Water / Topo_Objects | 0.5 m |
| Height_Objects, Height_Water | 0.5 m | | **Ecotopes A–H** | 2.0 m |

★★ **A shipping AAA open world bakes EROSION FLOW and EROSION DEPOSITION as half-metre maps beside the
heightfield, and derives all of its nature from them at runtime. That is exactly what our artifact row
is.** Extending the row (R2) is the industry's own answer, not an invention.

**Far Cry 5 bakes the same family and names it.** Its *"abiotic terrain data"*, generated from the
topology and the basis of every biome rule, is **occlusion, flow, slope, curvature, illumination**,
plus altitude, latitude, longitude and wind. Placement is then a published rule called **viability** —
each species declares favoured terrain attributes and *"Species that accumulate the most viability
will win over species"*, with a viability radius and a priority radius resolving conflicts.

★ **Horizon's "Ecotope" is our biome word, written by somebody who shipped it:** *"Ecotope describes
environment. Ecotopes determine: Asset types, Asset distribution, Colorization, Weather, Effects,
Sound, Wildlife."* **That is 8e's job description.**

★ **AND THEY REQUIRE WHAT OUR LAWS REQUIRE.** Horizon's motivation slide lists *"Data driven,
**Deterministic**, **Locally stable**"*. Far Cry 5's pipeline runs **nightly full-world regeneration on
build farms** with determinism stated outright — the *"same part of the terrain will always give the
same results"* — because the navmesh must be seamless across map junctions. **SL10 and SL8 are not
exotic; they are the standing requirements of AAA procedural terrain.**

### 1.3 The tools that make the shape, and the one knob they share

| tool | erosion | what the docs state |
|---|---|---|
| **Houdini** `heightfield_erode` | hydraulic + thermal | *"uses rainfall and weathering to simulate the removal and transport of material, at a scale specified by the **Erosion Feature Size**"*; writes height, **sediment**, **debris**, **flow**, **flow direction**. ★ *"HeightField Erode will produce very similar features if the resolution of the input terrain is changed."* ([SideFX](https://www.sidefx.com/docs/houdini/nodes/sop/heightfield_erode.html)) |
| **Gaea** `Erosion` | hydraulic, *"a proprietary approach that preserves features across different resolutions"* | Duration, Rock Softness, Strength, Downcutting, Inhibition, and ★ **Feature Scale**, *"the lateral size of the largest erosion features in meters"*. ★ *"a 512 × 512 preview build will maintain essential parity … with a high resolution 4K or 8K build."* ([QuadSpinner](https://docs.quadspinner.com/Reference/Erosion/Erosion.html)) |

★ **Both flagship tools expose the SAME knob under two names — a LENGTH IN METRES that decouples the
erosion's feature size from the grid.** That is §2 ①'s multi-scale idea, sold as a slider: the
industry's own confirmation that the cure for "the drainage is one cell wide" is to make erosion's
scale a physical length, not a pixel count. Houdini even states the floor: *"The smallest possible
feature size is **3× the input terrain's voxel size**."* ★ And its `heightfield_erode_hydro` carries a
**Grid Bias** knob — *"positive = axis-aligned, negative = 45°"* — **a vendor slider for the very grid
anisotropy ruling B2 step 3 is chasing.**

★★ **AND ON TILING THE VENDORS ADMIT WHAT SCHOTT 2024 ADMITS.**

- **World Machine**, the clearest statement any vendor makes, verbatim: *"Some devices, including the
  simulation-based devices such as **Erosion** and **Snow**, can produce different results in tiled
  mode versus normal builds … when tiling, they **do not have access to information from outside of
  the tile region**."* Its cure is a guard band — *"building an extra area around the tile and then
  blending between the tiles during the 'Merging' Phase"* — with **no default blending percentage
  published**.
- **Gaea** hides the same error behind a number: *"increase blending when you see seams in erosion,
  deposits, or any effect that depends on neighborhood sampling"*, and **"around 25 % is often a good
  starting point"**. Its worked example — *"Final tiles: 512 × 512 | Bucket size: 4096 × 4096"* —
  means **erode a 4096² bucket whole, then cut it into 64 tiles.**
- **Houdini** offers `heightfield_tilesplit` with **Voxel Padding** (*"Adds overlapping voxels across
  tile boundaries"*) and **publishes no guidance at all** on eroding tiles independently.
- **World Creator** never simulates per tile: one GPU texture, every filter over the whole map, tiles
  cut at export. It does publish **the only absolute bake time in the industry** — *"4k by 4k pixels
  terrain with snow simulation, flow simulation, two gradient materials + textures"* in **0.183
  seconds** — but not the iteration count behind it.

★ **So no tool ships the guarantee we need. A correct per-tile erosion would be a NEW guarantee, not a
reproduction of what the tools do — and §2.5 shows it has been published, just never productised.**
⚠ One correction to a common belief: **Gaea's docs do not claim GPU erosion.** Erosion2's claim is
*"deterministic results with up to 10× faster performance, **even on the CPU**"*, and the classic
`Erosion` node documents that *"When using Parallel Processing, the erosion algorithm can become
**non-deterministic**"*, with a `Deterministic` toggle that forces a single core. **The industry hits
our exact law and pays for it with a core.**

Unreal's own procedural system shapes no ground: PCG generates **points in 3-D space** that *"conform
to the shape of the Landscape"* and spawn meshes on it
([PCG overview](https://dev.epicgames.com/documentation/en-us/unreal-engine/procedural-content-generation-overview)).
★ **The industry's flagship procedural terrain tool is a SCATTERER.**

### 1.4 Crimson Desert itself — nothing technical is published

★ **The owner's reference frame has no published pipeline.** The GDC 2025 BlackSpace showcase was
closed-door and names only hair, cloth, shallow water, volumetric fog, atmospheric scattering and ray
tracing — **no terrain, no resolution, no world size**
([Pearl Abyss dev archive](https://crimsondesert.pearlabyss.com/en-US/News/Notice/Detail?_boardNo=40)).
Two 2026 talks are announced and summarised by press only: CEDEC 2026 describes a layered "World
First" pipeline using **Houdini for automatic placement across the map, with manual placement reserved
for memorable scenes** ([Inven Global](https://www.invenglobal.com/articles/24083/pearl-abyss-the-only-korean-game-company-to-present-on-crimson-desert-development-at-japans-cedec)),
and a gamescom 2026 talk promises "procedural terrain generation" with no slides released
([MMORPG.com](https://www.mmorpg.com/news/pearl-abyss-announces-crimson-desert-booth-and-behind-the-scenes-developer-presentations-at-gamescom-2000138457)).
**Treat the frame as a TARGET, never as a pipeline we can copy: nobody outside Pearl Abyss knows how
it was made.** Red Dead Redemption 2 is the same — **no terrain talk exists in the GDC Vault at all**,
and every circulating "RDR2 is X m/texel" figure is community reconstruction.

### 1.5 ★ The share of believability — the closest thing to a number anybody publishes

No studio publishes "the terrain is X % of it". Four quantified proxies exist, and all four say the
same thing.

1. ★ **Ghost Recon Wildlands, verbatim: "We estimated that 80 % of the data has been created or
   manipulated using Houdini."** And why: *"Terrain is more than elevation and splatting. We
   extrapolate very useful information that we'll be able to reuse in many other tools"* — roughness
   and crest detection from the ELEVATION drive rock and vegetation placement; river data defines
   wetness, which drives vegetation rules; a sun mask drives vegetation **and ambient sound**.
2. ★ **Ghost of Tsushima's memory table**, per 200 m tile against a ~6M budget: terrain **2.5M**,
   terrain physics **24M naive → 0M derived**, vegetation **12M naive → 1M procedural**. **Storing the
   dressing costs twelve times deriving it. Storing the shape is unavoidable.**
3. ★ **Horizon**: *"We use procedural placement for **all nature**! **500+ asset types. 100 000+ objects
   in scene. ~250 µs avg busy load on GPU**"*, and *"Nature assets created by **3 people**. Ecotopes
   made by **1 person**."* **The dressing is cheap at runtime and expensive in ART.**
4. ★ **Far Cry 5** built a procedural cliff pipeline whose stated purpose is that it *"adds shape and
   variety and **helps reduce a little the 'height map' feel**."* Four more systems (tree roots, rock
   clutter, far grass, displacement decals) all READ the terrain at runtime to hide the seam between
   shape and dressing. Its density figure: **604 824 entities in 1 km²**.

★★ **The conclusion those four support, and the one this report will defend: the dressing carries most
of the PIXELS, and every bit of it is READ OFF THE SHAPE. A percentage is the wrong question; the
right one is an ORDER.** §4.2 carries the argument; ruling B1 already stated it in the owner's words —
*"Paint hides nothing: the faults the owner saw are the model's, not the look's."*

---

## 2. THE ALGORITHMS THAT MAKE VALLEYS AND RIDGES AT THE FINE SCALE

### 2.1 What we are measuring against — our own four constraints

Every candidate below is judged on these four, in this order. The first two are laws; the last two are
costs.

| # | The constraint | Where it comes from |
|---|---|---|
| **C1** | **BYTE-FOR-BYTE ON EVERY HOST.** Integer arithmetic only on the shared path, a FIXED iteration count (never a tolerance loop), and every tie-break STATED. | SL10; ruling F7 |
| **C2** | **NO DRAWN PHYSICAL NUMBER.** Every coefficient is a published law with a named calibration body; a draw only for the seed's identity choices or a NAMED scatter. | ruling T9 |
| **C3** | **BUILDABLE ON DEMAND, PER PATCH, FROM THE MACRO FIELD.** A patch's answer may depend only on a BOUNDED neighbourhood, so a client can build what is under the eye without building a planet. | ruling F9 (the bounded ask); SL9 |
| **C4** | **A COST A SERVER CAN AFFORD**, stated per km² at a stated resolution. | ruling F6 (a share of the cores) |

### 2.2 The size of the hole, measured on our own pictures

The belt stand's own caption reads `ground: 18696.6 m over the recipe … drawn: rungs 5..11 to
830.0 km`. Our rungs double from 1 m, so **rung 5 is a 32 m cell and rung 11 is a 2 048 m cell**, and
the macro lattice's node is **8 192 m** (rung 13; `crates/terrain/src/macro_lattice.rs`: edge 1 216,
**8 871 936 nodes**, node exactly 8 192 m).

★ **So every feature in the belt picture, from 32 m to 8 192 m, is a sum of noise octaves and nothing
else.** And the octave table is not the weak part: `crates/terrain/src/body.rs` records that the home
planet's fine octaves run **25 km down to 49 m**, that the 3.13 km octave carries an RMS slope of 0.40
and 198.8 m of amplitude, and that a band of octaves around the peak is already **RIDGED**. **We
already have ridged multifractal noise, in the right band, at the right amplitude — and it still reads
as dunes.** That is the strongest evidence in this report that the missing ingredient is not
amplitude, not ridging and not more octaves: it is **DRAINAGE CONNECTIVITY**. Noise has no direction,
so nothing joins.

The macro solve's own doc-comment states the same bound from the other side: *"A pilot who looks 50 km
down a valley sees six nodes span that view."* Six samples across the whole view is a silhouette, not
a landscape.
### 2.3 The candidates, ranked

"Does it join?" means: does it turn ripples into ridges and valleys that CONNECT into a network?

| # | model | does it join? | C1 determinism | C3 per patch | C4 cost | verdict |
|---|---|---|---|---|---|---|
| ★ ① | **Multi-scale erosion amplification** — Schott et al. 2024 | **yes, and measured** | **yes** — fixed parallel sweeps, no queue, no tolerance loop | not as published; **yes for us** (§2.5) | 0.99 ns per cell-iteration on a GPU (DERIVED) | **ADOPT the algorithm** |
| ★ ② | **Hierarchical rule-based subdivision** — Cortial et al. 2020 | yes, but DRAWN and too sparse by its own account | **yes, and it states the rule** | **yes, by construction** | 80 ms a run, every ~10 frames, whole planet | **HARVEST the seed law** |
| ③ | **Gradient-aligned structured noise** — Grenier et al. 2024 | ★ **NO — the authors say so** | yes (a pure function) | trivially | real time | harvest as a SKIN only |
| ④ | **Droplet / particle erosion** — Beyer 2015 | partly; one-cell channels | ★ **NO** — order-dependent | an open research question | 40.65 s for 100 k drops on 1024², one core | read, do not build |
| ⑤ | **Grid shallow water / virtual pipes** — Mei 2007, Stava 2008 | yes | risky — needs a stable time step | **yes** — a flux per edge is a stated boundary | GPU-shaped | harvest the boundary idea |
| ⑥ | **Thermal erosion alone** — Musgrave 1989 | **no** — it rounds slopes and makes no network | yes | trivially | negligible | ADOPT, as one of ①'s three operators |
| ⑦ | **More rungs of our own macro solve** | yes | yes | ★ **no** — one global solve | 2.3 h at 1 024 m (DERIVED) | **IGNORE** |
| ⑧ | **Learned amplification** — GAN, diffusion | measured 5× to 1 000× worse than ① | ★ **no** — float weights, and no published law (T9) | yes | — | IGNORE for the shape |

★ **C1 rules out a whole family before any picture is judged.** A **droplet** walks one particle at a
time and each particle reads the field the last one wrote, so the answer depends on the ORDER of the
drops; it is reproducible only if that order is fixed, which forbids the obvious threading. A **grid**
model is a parallel STENCIL: every cell reads the previous iteration and writes its own, so no cell
can see another's half-finished work. **For SL10 the grid model wins on structure, not on taste.**

#### ① Multi-scale erosion amplification — the recommendation

From a coarse terrain `T₀`, repeat per level: bicubic upsample (which adds no detail), then three
operators that each move the ground one way only.

- **Flow routing (§4.1).** Multiple flow direction: `w(p,q) = s(p,q)ᵖ / Σ s(p,p′)ᵖ` over the eight
  neighbours; `p = 1` spreads by slope, `p → ∞` is steepest descent. They use **`p = 1.3`** *"to avoid
  sharp fluvial incision produced by high exponents"* (Holmgren 1994). The drainage area is then ONE
  parallel iteration per erosion step — `a_{i+1}(p) = 1 + Σ w_i(q,p)·a_i(q)` — reused between steps
  *"since the modifications to the terrain are small between two erosion iterations."*
- **Stream power, CLAMPED (§4.2).** `h_{i+1} = h_i − k·ẽ_i` with
  `ẽ_i = min(s_iⁿ, s_maxⁿ)·min(a_iᵐ, a_maxᵐ)`. The text says `m/n` *"typically"* equals ½; ★ **the
  shipped shader uses `n = 2`, `m = 0.8`, `k = 5 × 10⁻⁴`** (read out of `erosion.glsl`'s uniforms), so
  `m/n = 0.4`. The clamps are their own contribution: without them *"plunging erosion features appear
  on steep slopes, whereas the peak regions lack erosion landmarks."*
- ★ **A HARDNESS FIELD.** `k(p) = k·(1 − ρ(p))`, with `ρ` a fractal noise. Their sentence:
  *"Introducing randomness in the hardness function also reduces the **axis-aligned artifacts produced
  by the regular grid discretization**."* **That is a published cure for ruling B2 step 3's scratches —
  and we already hold the field it wants, the ROCK PROVINCE (`ARTIFACT_VERSION 5`, ruling W4).**
- **Thermal stabilisation (§4.3).** `∂h/∂t = −k_γ·max(s − s₀, 0)` — Musgrave, Kolb & Mace 1989's talus
  rule, `s₀ = tan γ₀`.
- **Deposition (§4.4).** A suspended-sediment field: created in proportion to the fluvial erosion,
  transported, deposited; `h_{i+1} = h_i + d_i`. This is what makes a valley floor flat, not a slot.
- ★ **Height retargeting (§5.1).** A diffusion that RESTORES the elevation of chosen points — ridges,
  peaks — after amplification. **This is the mechanism that holds a fine solve to the macro solve's own
  heights**, which is exactly the boundary condition C3 needs.
- **Multi-scale breaching (§5.2).** A Barnes, Lehman & Mulla 2014 breach across the levels, which
  *"guarantees a hydrologically consistent output **without one-cell canyon artifacts**."*

**Does it join? Measured, by a number that can fail.** Table 3 reports the average BREACHING VOLUME
`b` — the material a breach must still remove to make the field drain freely, `b = 0` being perfect:

| terrain | learned (StyleGAN) | procedural noise | sparse modelling | **theirs** |
|---|---|---|---|---|
| A | 23 | 41 | 30 | **0.05** |
| B | 120 | 1.7 | 1.0 | **0.02** |
| C | 62 | 41 | 79 | **4.8** |

★ **Procedural noise — our recipe's own family — loses by one to three orders of magnitude.**

**Cost, MEASURED by the authors** (Table 2, GLSL compute shaders, one up-to-date GPU, milliseconds per
iteration): at 4096², erosion 5.07 + thermal 0.90 + deposition 10.7 = **16.67 ms over 16.8 M cells**;
at 8192², 19.2 + 3.51 + 41.5 = **64.21 ms over 67.1 M cells**. ★ **DERIVED: 0.99 ns and 0.96 ns per
cell per full iteration — the same number twice, which is the check that the derivation is sound.**
Iterations per level (Table 1): **2 600 at 128, falling to 300–800 at 4096**. The CPU-side retargeting
and breaching *"take a few seconds to complete at 4096 × 4096"*, once, at the end.

**Determinism.** Add, multiply, compare, and the power terms — `n = 2` is a square, `m = 0.8` needs a
root the recipe already holds — plus one reciprocal for the routing normalisation, which it also
holds. Two things must be STATED: the tie-break when two neighbours are level, and the exponent `p`
(1.3 is not an integer, but `p = 2` sits inside Hyväluoma 2017's published optimum band of
`W ≈ 1.3–4.1` for grid isotropy). ★ **And MFD routing is itself a listed cure for the D8 scratches:
one mechanism, two defects.**

★ **The one honest obstacle, verbatim from their Limitations:** *"Multi-scale amplification … **requires
the processing of the entire elevation map at every step**, notably to evaluate the drainage (Section
4.2) and the diffusion of sediments (Section 4.4). We limited our experiments to a maximum grid size
of 8192 × 8192, on a single up-to-date GPU."* **As published it is NOT tile-local — §2.5 shows the
published cure and what it costs.**

#### ② Hierarchical subdivision — and the determinism paragraph SL10 has been waiting for

A Poisson-triangulated planet — *"the average length of the edges of a triangle is ≈ 50 km"* — is
subdivided by production rules keyed on what each vertex and edge IS (river, terrace, lake, slope,
ground, unknown), down to **50 cm on an Earth-sized body, in real time**. MEASURED on an i7-6700K and
a GTX 1080 Ti: relief generation runs once every ≈ 10 frames at **80 ms** average (subdivision ≈ 50 ms,
riverbed and valley carving ≈ 3 ms, mesh post-process ≈ 30 ms), and rendering held **≈ 25 Hz** worst
case with 4 million terrain triangles.

> *"**Determinism** The output of the pipeline is deterministic and guarantees the coherent production
> of the terrain at different scales. All compute operations of stochastic nature need to be
> reproducible. Compute nodes produce elevations that need to be deterministic, i.e. **invariant across
> subdivision runs**. This is ensured by using a **unique seed tied to each vertex, which is created as
> the sum of the seeds of the two vertices of each edge** … To ensure that a branching river is always
> created in the same incident triangle to the main river edge, we rely on choosing among the two
> incident faces by drawing a random number from the seed tied to the junction river vertex."*

**That is the whole seam law for a hierarchical amplification, in three sentences: a child's seed is a
function of its parents' seeds, so two neighbours get the same child whoever computes it, and in
whatever order.** It is the discipline `vd_seed::digest` already uses, lifted to a refinement
hierarchy.

⚠ **Two warnings it pays for and we must not.** *"Our method is memory demanding and requires more
than 2 GB on the graphics card."* And because it re-derives every ten frames, a vertex's slope changes
between runs, so *"the snow effect … varies from one subdivision run to another"* — **a flicker, which
is an SL8 seam.** Its own realism limit: *"the final river network is **not dense enough compared to
real data**."*

#### ③ The cheap option, refuted by its own authors

Grenier et al. 2024's Phasor noise aligns erosion patterns to the coarse slope, cascades narrow
ravines inside wide ones, is resolution-independent to ×32 and runs in real time — every constraint
met except the one that matters. Its Limitations, verbatim: it *"**cannot generate a global coherent
river network**, a common limitation of **all function-based elevation models relying on a fully
procedural and parallel algorithm**"*, and *"it is ill-suited to amplify sedimentary valleys or flood
plains … the proposed gradient-aligned noise function **replicates linear patterns over almost flat
regions**."* ★ **That last clause is the belt stand.** Its own honest measurement of where it sits:
on their terrain C the breaching volume is **fBm 128, real erosion 25, theirs 86** — better than
noise, nowhere near a solve. Cost: **4 ms for 100 generations at 2048² on a GTX 980**, which is why it
is the right SKIN.

#### ④ Droplets, and why the measured table is the give-away

Beyer's parameters, verified from [`henrikglass/erodr`](https://github.com/henrikglass/erodr) (MIT,
one C file): inertia 0.3, capacity 8, deposition 0.3, erosion 0.7, evaporation 0.02, minimum slope
0.0001, radius 2, maximum path 32, **70 000 particles** by default. MEASURED in the thesis (Table 6.1,
i5-3470, single thread): **100 000 drops take 32.02 s on 256², 40.65 s on 1024², 43.81 s on 2048².**
★ **Read that the right way round — the cost barely depends on the map and depends almost entirely on
the number of drops, so a bigger area needs proportionally more drops and the per-km² cost is
constant, not free.** The erosion radius costs `radius²`: 0.48 s at radius 1 against 9.69 s at radius
8, for 10 000 drops (Table 6.2).

#### ⑤ to ⑧, in one paragraph each

**⑤ The pipe model** (Mei, Decaudin & Hu 2007, [10.1109/PG.2007.15](https://doi.org/10.1109/pg.2007.15);
Stava et al. 2008) keeps water depth and four pipe fluxes per cell. ★ **A flux crosses an EDGE, so a
patch boundary is a stated boundary condition and not an artefact** — the same shape as our seam
doors. It still loses to ①, because it simulates water in TIME and so needs a stable time step, which
is a tolerance in disguise. **UNVERIFIED: the PDF would not open; the design intent is quoted from the
abstract.**
**⑥ Thermal erosion** is one of ①'s operators and the only one safe to ship alone; it makes scree, not
a network (Musgrave, Kolb & Mace 1989, *Computer Graphics* 23(3), 41–50 — reached through Schott et
al. 2024 §4.3, **UNVERIFIED at first hand**).
**⑦ Refining the macro lattice itself** is refuted twice: our solve is MEASURED at **130.3 s for
8 871 936 nodes over `PASSES = 40`**, which is **367 ns per node-pass (DERIVED)**, so 1 024 m would
take about **2.3 hours** and 128 m is out of reach; and it stays ONE global solve, so no patch can
ever be built on demand.
**⑧ Learned amplification** (Guérin et al. 2017,
[10.1145/3130800.3130804](https://doi.org/10.1145/3130800.3130804); Perche et al. 2023; Lochner et al.
2023 — the last two **UNVERIFIED**, cited from ①'s bibliography) is refused on law before quality: a
network's weights are neither a seed nor a published law (T9), and a float network cannot pass the
byte-for-byte gate.
### 2.4 Why our first attempt made the pictures worse — the published diagnosis

`crates/terrain/src/river.rs` DRAWS the drainage: it reads the macro row's receiver and discharge, then
synthesises tributaries from Horton 1945's bifurcation ratio `Rb ≈ 4` and length ratio `RL ≈ 2.3`,
places junctions at Howard 1971's 60°, and stamps a Leopold & Maddock 1953 channel of
`w = 3.9·√Q` with a Leopold & Wolman 1960 floodplain belt. Every law in that file is sound and
correctly calibrated. **The failure is not in the laws — it is in DRAWING instead of SOLVING**, and
three separate papers name it:

- **Schott et al. 2024**, on why a drawn drainage looks like a slash: existing methods *"carve drainage
  patterns at the size of one cell"*, cured only by multi-scale amplification with multi-scale
  breaching.
- **Schott et al. 2023** on Génevaux et al. 2013, the archetype of the drawn network: *"an essential
  limitation of this approach is that the procedurally generated mountain ranges **may not be
  hydrologically correct, and may not exhibit characteristic dendritic shapes**."*
- **Grenier et al. 2024**, on why no purely local function can do it: *"a common limitation of all
  function-based elevation models relying on a fully procedural and parallel algorithm."*

★ **In the game.** The river stand's own picture (`runs/1790100647__pilot__a0/shots/river_stand.png`,
2 216.8 m over the ground) shows it: a straight blue band on flat brown ground, with a rounded end
where the stamp stops and no valley around it. Under ①, no line is drawn at all: the ground is lowered
where the water runs, so the river is wherever the ground is lowest — and a tributary cannot cross a
hill, because the hill is what the solve left standing between two valleys.

### 2.5 ★ THE TWO-SCALE QUESTION: data, or derivation? — and it is already answered in print

The owner's question, restated: the macro solve gives each patch its boundary heights, its rivers'
entry and exit and their discharge; a local solve builds the fine drainage inside it. **Does that patch
ship as DATA like the artifact, or run on BOTH HOSTS from the seed?**

★ **First, the fact that changes the discussion: EROSION ON TILES WITH A HALO IS PUBLISHED, MEASURED
AND EIGHTEEN YEARS OLD.** Šťava, Beneš, Brisbin & Křivánek 2008, SCA
([PDF](https://www.cs.purdue.edu/homes/bbenes/papers/Stava08SCA.pdf), read in full), verbatim:

> *"Our implementation supports simulation of multiple terrain grids (**tiles**) individually … The
> **synchronization of neighboring tiles at their borders is performed by copying of border values
> before each simulation step**. Tile dimensions and computational domain are extended to handle these
> copied values."* — and Figure 8's caption, *"Terrain is divided into tiles that are calculated
> independently and synchronized at borders."*

Because the stencil is one cell wide and the halo refreshes every step, that is **not an
approximation — it is numerically identical to a whole-map bake.** Their motive was ours in another
dress: *"To address the limited GPU memory, we divide the terrain into tiles that can be processed
independently."*

**Erosion is two problems with opposite tiling behaviour, and they must be separated.**

| half | stencil | tiles? | the published answer |
|---|---|---|---|
| **transport** — thermal talus, sediment diffusion, deposition, the pipe model | 4 to 8 neighbours, strictly local | **yes, exactly** | Šťava 2008: copy the border before each step. **1-cell halo.** |
| **the drainage area** — the stream-power term | a whole catchment | ★ **no, not on an isolated tile** | either exchange the halo every iteration, or Barnes's exact perimeter decomposition |

★ **And the global half has an exact tiled solution with a measured cost.** Richard Barnes, *Parallel
Non-divergent Flow Accumulation For Trillion Cell Digital Elevation Models*,
[arXiv:1608.04431](https://arxiv.org/abs/1608.04431), abstract verified verbatim: *"The largest
dataset tested had **two trillion (2·10¹²) cells**. With **48 cores**, processing required **24 minutes
wall-time** (14.5 compute-hours)."* Each tile solves locally and ships only its **perimeter** — flow
directions, accumulations, and how perimeter cells link THROUGH the tile; a coordinator resolves the
global perimeter graph; offsets come back; a second pass finishes. Communication grows with the tile's
PERIMETER, not its area. The companion, *Parallel Priority-Flood depression filling*
([arXiv:1606.06204](https://arxiv.org/abs/1606.06204)), does the same for the breach at 2 trillion
cells in 4.8 hours. **Code: [RichDEM](https://github.com/r-barnes/richdem) and
[Barnes2016-ParallelFlowAccum](https://github.com/r-barnes/Barnes2016-ParallelFlowAccum).**

★ **BUT OUR CASE IS EASIER THAN BARNES'S, AND THIS IS THE KEY.** Barnes must DISCOVER the global
drainage from raw elevations. **We already know it** — the macro solve computed it and the row stores
it. So we do not need a perimeter graph at all: **seed the drainage area of every fine cell from the
macro row's DISCHARGE class, divided down by the refinement factor.** The patch then STARTS with the
globally correct answer, and the iterations only correct it locally. **That byte is already in the
row, and it is what turns the global half into a local one.**

**Three halo widths, each from a paper.**

1. **Droplets, Beyer 2015:** a drop moves **exactly one cell per step**, `p_maxPath = 64`,
   `p_radius = 4`. A **~68-cell halo makes droplet erosion exactly tile-local with zero exchange.**
   ★ And Beyer measured the defect a missing halo causes — his *"drain valley"* artefact, where
   valleys grow inward from the map edge because drops that leave carry their sediment away. **His own
   proposed cures are the halo: erode a larger map and crop the middle, or simulate a few grid points
   outside the map.**
2. **The pipe model, Mei 2007 / Šťava 2008:** a 4-neighbour stencil — **1-cell halo, exchanged every
   step.**
3. ★ **The drainage area, Schott et al. 2023 §4.2:** a 5 × 5 stencil iterated to a fixed point,
   measured to reach the reference *"after several iterations ranging between `n` and `4n`, with an
   average of **1.5n**"*, and 130 iterations on a 256² grid. **A static halo would have to be 1.5 times
   the tile's own width — useless.** Exchange every iteration, or seed the answer as above.

⚠ **And the counter-warning.** Cordonnier et al. 2017 measured that its runoff event *"exhibits average
complexity of O(n) for every cell"* — an uncapped droplet's path averages the width of the map.
**Beyer's `p_maxPath` is exactly what bounds it. An uncapped particle crosses a tile and the halo
argument collapses.**

★ **THE STATE OF THE ART ADMITS THIS IS OPEN AT THE AMPLIFICATION SCALE.** Schott et al. 2024 §6.6,
verbatim and in full: *"amplifying terrains beyond this size would require **decomposing maps into
patches**, as advocated by Vanek et al. [2011], before assembling them. This decomposition introduces
a series of technical challenges, particularly **the propagation of drainage across the boundaries of
the patches, and their seamless blending after erosion**."* **The 2024 state of the art names our
problem and says it did not attempt it.** (Vanek, Beneš, Herout & Šťava 2011, IEEE CG&A 31(6):35–44,
[10.1109/MCG.2011.66](https://doi.org/10.1109/mcg.2011.66), is paywalled; its boundary handling is
**UNVERIFIED**.)

**So: SHIP what is global, DERIVE what is local, and the line between them is not a taste — it is
where the dependency radius stops fitting in a patch.**

The macro artifact already embodies that rule: a whole-planet iterative solve is shipped precisely
because no client can run it and no two clients could agree on it. The fine amplification is the
opposite: given the level above and the row's discharge, it is a bounded-neighbourhood function of
`(seed, address, the rows above)`, and both hosts hold identical rows because the rows are shipped.
**That makes derivation lawful under SL10** — with four conditions, each of which becomes a gate:

1. **A FIXED patch grid, keyed on the macro node address, never on the viewer's position or tile.** If
   two viewers can ask for differently-aligned patches they get different ground, which is a seam (SL8).
2. **A HALO at least as wide as the iteration count**, so a patch's interior cannot see past its halo.
   A patch of `E` cells with halo `N` does `(E + 2N)²` work instead of `E²`.
3. **A fixed iteration count, stated tie-breaks, integer arithmetic** (C1).
4. ★ **The seed law of Cortial et al. 2020**: a refined cell's seed is a function of its parents'
   seeds, so two neighbours agree whoever computes them, and in whatever order.

**The gate that can fail:** build the same patch twice with two different halo widths, and twice on two
different targets, and compare **byte for byte**.

### 2.6 What it would cost us — every number DERIVED, and a bench owed

Our planet: **8 871 936 nodes**, node **8 192 m**, node area 67.1 km², whole surface **595 M km²**.
Each halving of the cell multiplies the cells by four.

| cell | rung | cells, whole planet | `Z` at 2 B/cell | one artifact tile (64 nodes = 524 km) |
|---|---|---|---|---|
| 8 192 m | 13 | 8.87 M | 17.7 MB | **40 KB (today, 10 B rows)** |
| 4 096 m | 12 | 35.5 M | 71 MB | 32 KB |
| **2 048 m** | **11** | **142 M** | **284 MB** | **128 KB** |
| **1 024 m** | **10** | **568 M** | **1.14 GB** | **512 KB** |
| 512 m | 9 | 2.27 G | 4.5 GB | 2 MB |
| 256 m | 8 | 9.09 G | 18 GB | 8 MB |
| 64 m | 6 | 145 G | 291 GB | 128 MB |

**Reading the table.** Shipping is comfortable to **1 024 m** and stops hard below **512 m**. The belt
stand draws rungs 5 to 11, so a shipped level at 1 024 m would give real drainage to the two coarsest
rungs the picture contains, and the rest must be derived.

**The compute, from the two measured anchors.**

- **A GPU, from ①'s Table 2: 0.99 ns per cell per full iteration (DERIVED).**
- **Our CPU, from our own solve: 367 ns per node-pass (DERIVED)** — but that figure is for the WHOLE
  macro solve (plates, isostasy, flood, flats, receivers, climate, ice, craters), not for an erosion
  iteration, so it is a CEILING and almost certainly a bad one. ★ **The true integer cost of the
  amplification kernel on our CPU is UNMEASURED, and that single number decides the whole plan.**

With those two:

| job | cells × iterations | at 0.99 ns (GPU) | at 367 ns (our ceiling) |
|---|---|---|---|
| whole planet to 2 048 m, 300 iters | 4.26 × 10¹⁰ | **42 s** | 4.3 h |
| whole planet to 1 024 m, 300 iters | 1.70 × 10¹¹ | **169 s** | 17 h |
| one patch, 16 × 16 nodes (131 km) to 64 m, 300 iters | 1.26 × 10⁹ | **1.2 s** | 7.7 min |
| one artifact tile (524 km) to 64 m, 300 iters | 2.01 × 10¹⁰ | 20 s | 2.1 h |

★ **The gap between the two columns is 370×, and it is the only real question in this report.** If the
integer kernel lands within about 20 ns per cell-iteration on one core, everything below is
comfortable; at 367 ns nothing below 1 024 m can be baked and the card becomes mandatory. **Ruling F9
already made the card a budgeted second builder, and ruling F7 already ordered "the bench, the
discussion, the build". This is the same order again, and it is recommendation R1.**

★ **And the field's own survey says we are attempting something nobody has done.** Galin, Guérin,
Peytavie, Cordonnier, Cani, Beneš & Gain, *A Review of Digital Terrain Modeling*, CGF 38(2), EG STAR
2019 ([PDF](https://hal.science/hal-02097510/file/A%20Review%20of%20Digital%20Terrain%20Modeling.pdf))
places grid hydraulic erosion at **1–10 km extent and 1–10 m precision**, and stream-power/tectonic
methods at **10–100 km and 10–100 m**. Its cost statement, verbatim: *"A high resolution terrain, such
as **16 × 16 km² sampled at 32768² (precision ≈ 50 cm), has more than 1 billion grid cells**. Combined
thermal and hydraulic erosion typically takes **several hours** to complete on such large maps."* And
its open-problem list names ours: *"**Generating entire planets while providing control over the
position, extent and shape of landforms at different scales remains to be solved.**"* **Nobody bakes a
planet. That is why §5 ships one level and derives the rest.**

---

## 3. RUST CRATES AND REUSABLE CODE FOR THE FINE SCALE

Our constraint is the same as the lakes survey's: a crate can enter the SHARED recipe path only if it
is integer, and no crate is. A server-only library is admissible where a shared one is not. GPL and
AGPL block code and never block reading a paper.

### 3.1 ★ THE THREE FINDINGS, BEFORE THE TABLES

1. ★ **THE EROSION SHELF IS ALMOST EMPTY.** After removing the planet-scale crates the lakes survey
   already covered, **`symbios-ground` is the only maintained crates.io package that simulates erosion
   at all** — one author, 1 054 lifetime downloads, `f32` throughout.
2. ★ **NO RUST NOISE CRATE HAS A FIXED-POINT MODE. Not one.** All six are float-only. Our integer
   recipe (`Gi`, 40 fraction bits, noise at 28) duplicates nothing that exists.
3. ★ **AND `simdnoise` WOULD BREAK THE NO-DRIFT GATE BY DESIGN** — it does runtime CPU-feature
   detection and picks an SSE2, SSE4.1 or AVX2 path per machine, so two clients take two paths.
   **Name it in the ban list beside the default hasher.**

★ **The corollary the owner asked for, stated plainly: there is nothing to ADOPT as a dependency —
but there is a great deal to HARVEST, and two of the harvests are MIT and maintained**
([`MultiScaleErosion`](https://github.com/H-Schott/MultiScaleErosion), the recommendation's own code;
[`richdem`](https://github.com/r-barnes/richdem), the tiled flow accumulation). **Every valuable item
on these lists is an ALGORITHM of a few hundred lines that we must write on `Gi` anyway.**

### 3.2 Rust crates that touch erosion

| crate | version | last release | licence | downloads | what it is | verdict |
|---|---|---|---|---|---|---|
| ★ [`symbios-ground`](https://crates.io/crates/symbios-ground) ([repo](https://github.com/TheJanusStream/symbios-ground)) | 0.4.1 | 2026-09-06 | MIT | 1 054 | Droplet hydraulic erosion with lake pooling and delta deposition, plus talus thermal erosion, over Diamond-Square / fBm / Voronoi. Holds a `TiledHeightMap` with per-tile seeds. ★ **Claims bit-identical output on every target by routing every transcendental through `libm`** — the same instinct as SL10, reached independently. But `f32`, and **its tiles have no stated boundary condition, no overlap and no seam blend**. | **harvest** (read the structure, not the numbers) |
| [`martini`](https://crates.io/crates/martini) | 0.2.0 | 2026-05-18 | UNVERIFIED | 2 721 | RTIN mesh from a heightfield — MESHING, not relief. | read (meshing only) |

Also surveyed and IGNORED, with the reason: [`bevy_erosion_filter`](https://crates.io/crates/bevy_erosion_filter)
(a per-fragment shader LOOK, not a simulation), [`vinland`](https://crates.io/crates/vinland)
(**LGPL-3.0-or-later**), [`nlmrs`](https://crates.io/crates/nlmrs) (landscape statistics, no erosion),
[`ds-heightmap`](https://crates.io/crates/ds-heightmap) and
[`diamond-square`](https://crates.io/crates/diamond-square) (midpoint displacement only),
[`terrain-forge`](https://crates.io/crates/terrain-forge) (dungeon tiles, not heightfields).

### 3.3 ★ GitHub Rust — the one repository that answers our open question

| repo | ★ | licence | last push | what it is | verdict |
|---|---|---|---|---|---|
| ★★ [`pontusasp/erosion-rs`](https://github.com/pontusasp/erosion-rs) | 2 | **MIT** | 2026-08-08 | ★ **A master's thesis on exactly the question §2.5 leaves open**: *"Boundary Handling for Cohesive Tiling in Particle-Based Hydraulic Erosion Simulations"* (title verbatim from the README). Its `src/partitioning.rs` declares four methods and I read the file: `Default` ("No Tiling"), `Subdivision` ("Naive Tiling"), `SubdivisionBlurBoundary` ("Naive Tiling with Blur"), `GridOverlapBlend` ("Overlapping Grids"). **`GridOverlapBlend` is the halo of §2.5, built and compared.** The thesis document itself could not be opened — **UNVERIFIED**; the code is real. | **READ FIRST** |
| [`MOj0/shaping-terrain-using-advanced-multi-scale-erosion`](https://github.com/MOj0/shaping-terrain-using-advanced-multi-scale-erosion) | 0 | MIT | 2026-06-25 | ★ A Bevy + WGSL implementation of **the Schott et al. 2024 paper of §2.3 ①** — separate erosion, deposition and thermal shaders with 2× upsampling. The nearest existing code to the recommendation. | **read** |
| [`evroon/bevy-hydrology`](https://github.com/evroon/bevy-hydrology) | 13 | MIT | 2026-07-28 | Nick McDonald's hydrology with the erosion in a WGSL compute shader. The cleanest Rust + wgpu erosion kernel found. | harvest |
| [`rj00a/heightmap-erosion`](https://github.com/rj00a/heightmap-erosion) | 7 | MIT | 2021-09-20 | A parallel Rust port of Andrino's simulation, one file. | read |
| [`mate-h/bevy_erosion`](https://github.com/mate-h/bevy_erosion) | 1 | **none** | 2026-07-06 | Compute-shader erosion, **no licence file**. | ignore |
| [`liminalfield/ymir`](https://github.com/liminalfield/ymir), [`muhuk/yer`](https://github.com/muhuk/yer) | 5 / 4 | **GPL-3.0** | 2026 | Node-based terrain generators, active. | read only |

### 3.4 Noise crates — all float, and one that is dangerous

| crate | version | last release | licence | downloads | number type | ridged? | domain warp? | fixed point? |
|---|---|---|---|---|---|---|---|---|
| [`noise`](https://crates.io/crates/noise) | 0.9.0 | **2024-03-23** | MIT/Apache-2.0 | 2 639 286 | **f64 only** | yes (`RidgedMulti`) | yes (`Turbulence`, `Displace`) | **no** |
| [`fastnoise-lite`](https://crates.io/crates/fastnoise-lite) | 1.1.1 | 2024-03-05 | MIT | 173 317 | f32 (f64 behind a flag) | UNVERIFIED for the port | yes | **no** |
| [`bracket-noise`](https://crates.io/crates/bracket-noise) | 0.8.7 | **2022-10-04** | MIT | 374 131 | f32 | yes (`RigidMulti`) | amplitude setters only, no perturb method | **no** |
| [`libnoise`](https://crates.io/crates/libnoise) | 1.2.0 | 2025-04-29 | MIT | 24 290 | float | yes | `Displace` (docs contradictory — UNVERIFIED) | **no** |
| [`noiz`](https://crates.io/crates/noiz) | 0.5.0 | 2026-06-20 | MIT/Apache-2.0 | 9 757 | **"only supports f32 types for now"** | UNVERIFIED | UNVERIFIED | **no** |
| ⚠ [`simdnoise`](https://crates.io/crates/simdnoise) | 3.1.6 | **2020-05-30** | MIT/Apache-2.0 | 80 009 | f32 SIMD, **runtime CPU dispatch** | UNVERIFIED | UNVERIFIED | **no** |

### 3.5 Bevy terrain crates — renderers, not relief

[`bevy_terrain`](https://github.com/kurtkuehnert/bevy_terrain) is the best of them and is **rendering
and LOD only** — a GPU chunked-LOD terrain with an on-disk tile attachment cache; never published to
crates.io, pinned to **Bevy 0.14**, MIT/Apache-2.0. The rest split into renderers
([`bevy_mesh_terrain`](https://crates.io/crates/bevy_mesh_terrain) 58 913 downloads,
[`bevy-clipmap`](https://crates.io/crates/bevy-clipmap),
[`bevy_heightmap`](https://crates.io/crates/bevy_heightmap)) and toy generators
([`bevy_generative`](https://crates.io/crates/bevy_generative),
[`bevy_world_seed`](https://github.com/TheGrimsey/bevy_world_seed)). **None is a deterministic relief
source. Ignore the lot for shape; `bevy_terrain`'s tile cache is worth reading when slice 7's client
edge lands.**

### 3.6 Reference implementations worth reading, in any language

| project | language | licence | why | verdict |
|---|---|---|---|---|
| ★★ [`H-Schott/MultiScaleErosion`](https://github.com/H-Schott/MultiScaleErosion) | C++ + GLSL | ★ **MIT**, pushed **2026-09-15** | ★ **The official code release of §2 ①, the recommendation.** Four compute shaders — `erosion`, `thermal`, `deposition`, `spe_shader` — and the paper's own constants in their uniforms (`flow_p = 1.3`, `k = 0.0005`, `p_sa = 0.8`, `p_sl = 2.0`). **The oracle for R1's bench.** | ★ **HARVEST FIRST** |
| ★★ [`r-barnes/richdem`](https://github.com/r-barnes/richdem) + [`Barnes2016-ParallelFlowAccum`](https://github.com/r-barnes/Barnes2016-ParallelFlowAccum) | C++ | MIT | ★ The exact **tiled** flow accumulation and priority flood of §2.5 — perimeter-graph decomposition, 2 × 10¹² cells on 48 cores. | **harvest** |
| ★ [`bshishov/UnityTerrainErosionGPU`](https://github.com/bshishov/UnityTerrainErosionGPU) | C# + HLSL | **MIT** | Grid pipe-model hydraulic and thermal erosion in compute shaders. ★ **A grid cell has an explicit flux per EDGE, so a patch seam is a boundary condition you SET** — the shape §2.5 needs. | **harvest** |
| ★ [`henrikglass/erodr`](https://github.com/henrikglass/erodr) | C | **MIT** | Beyer's droplet algorithm in one plain C file. **The oracle our integer kernel can be diffed against.** | **harvest** |
| [`SebLague/Hydraulic-Erosion`](https://github.com/SebLague/Hydraulic-Erosion) | C# | **MIT** | The best-known droplet implementation, with a compute-shader version. | harvest |
| [`dandrino/terrain-erosion-3-ways`](https://github.com/dandrino/terrain-erosion-3-ways) | Python | **MIT** | Three methods side by side; ★ its **river-network** branch makes drainage WITHOUT stepping droplets, which is the shape a deterministic kernel wants. | harvest |
| ★ [`aparis69/*`](https://github.com/aparis69) — Desertscapes-Simulation, Meandering-rivers, Karst-Synthesis, Rock-fracturing, Implicit-Volumetric-Terrains | C++ | **MIT, all five** | The Galin group's own paper code, permissive. Meandering-rivers and Implicit-Volumetric-Terrains are the 8f "third dimension" material. | harvest |
| ⚠ [`weigert/SoilMachine`](https://github.com/weigert/SoilMachine), [`SimpleHydrology`](https://github.com/weigert/SimpleHydrology), [`SimpleErosion`](https://github.com/weigert/SimpleErosion) | C++ | ★ **NO LICENCE FILE ON ANY OF THE THREE** | SoilMachine's multi-layer soil column is the closest published analogue to ruling W2's "water = a substance with liquidity". SimpleHydrology keeps only MAPS and throws the particles away, which is the cheapest shape to make deterministic. | **read only — stricter than GPL; the method may be learned, the code may not be copied** |
| [`jobtalle/HydraulicErosion`](https://github.com/jobtalle/HydraulicErosion) + [article](https://jobtalle.com/simulating_hydraulic_erosion.html) | JS | MIT (repo) | A whole island in under half a second — good cost intuition. | read |
| ⚠ `TinyErode` | C++ | MIT (fork only) | ★ **The upstream `tay10r/TinyErode` is deleted**; only a 2-star [fork](https://github.com/Sondro/TinyErode) survives, last pushed 2021. Abandonware. | read |

### 3.7 Integer and fixed-point terrain, anywhere

Searched on purpose. [`Frangitron/FixedPoint3DNoise`](https://github.com/Frangitron/FixedPoint3DNoise)
— one star, no licence file, written for a microcontroller — is **the only fixed-point noise
implementation found in any language**. Beside it, [`alice-physics`](https://crates.io/crates/alice-physics)
(2026-09-17) is a "deterministic 128-bit fixed-point physics engine", which is not terrain but is proof
the idea has a Rust precedent.

★ **State it plainly to the owner: on the integer question there is no prior art to buy, to copy or to
compare against. `vd-recipe` has no competitor. That is a cost we have already paid and cannot
un-pay — and it is also the reason no crate on any of these lists can become a dependency.**

---

## 4. THE CANDID GAP LIST — what shape can never give

### 4.1 What is in the reference frame, item by item

I read the owner's frame and both of our stands as pictures, and listed what each holds. This is a
reading of three images, not a measurement.

| what the reference frame holds | is it SHAPE or DRESSING? | do we have it? | which slice owes it |
|---|---|---|---|
| Ridge lines stacked four and five deep, each with its own valley | **shape** | no — the band 32 m–8 km is noise | **8d, this arc** |
| Valleys that join downhill into one drainage | **shape** | no | **8d, this arc** |
| Cliff faces and rock outcrops breaking the slope | shape (3-D) | no — the heightfield has no overhang | 8f |
| Snow lying only above a line, and thicker on the shaded faces | dressing, read off shape | no | 8e (the paint table), 8L (the light, landed) |
| A dense conifer canopy carpeting the mid-ground hills — **most of the pixels in the frame** | **dressing** | no | 8e + ruling V4 (trees are ART ASSETS, never drawn primitives) |
| Grass and scrub on the near rock | dressing | no | 8e + V4 |
| Aerial perspective: the far ridges going blue and low-contrast with distance | **dressing** | no | **8s** (*"aerial perspective from the charter's scale height"*) |
| Cumulus clouds, with shadows cast on the ground | dressing | no | 8s, then weather (T3's weather list) |
| A low warm sun, long shadows down the valleys | dressing | the light landed (8L); no shadow yet | 8s (*"a sky dome, a shadow"*) |
| Water glinting in the middle distance | dressing | the lakes are being rebuilt (ruling B2 step 1) | 8d step 1, then 8o |
| A settlement, roads, fields | dressing (built) | no | far later; players build it |
| Colour grading — cool shadows, warm highlights | dressing | no | the client's own, no slice yet |

### 4.2 ★ The honest split — and why the share is the wrong question

**Counting pixels, the dressing wins easily** — in the owner's frame the canopy, the haze and the
clouds cover most of the image, and §1.5's four proxies all agree. A naive reading would put the shape
at a fifth of the believability.

★ **That reading is a trap, and our own belt stand is the proof.** Put a conifer forest, a haze and a
cloud layer on the belt stand and it becomes **a forest on dunes**. Every dressing element in the
table above is READ OFF THE SHAPE: the snow follows a height and an aspect; the forest thins on the
ridges and thickens in the valleys; the haze reveals depth only because there are ranked ridges for it
to rank; the shadows are long only because there are valleys to fill. ★ **A dressing cannot invent a
landform it is draped over** — which is why Far Cry 5 had to build a whole cliff pipeline to *"reduce
a little the 'height map' feel"*.

> **The shape is not most of the believability. It is the PRECONDITION for all of it.** Ruling B1
> already said it — *"Paint hides nothing: the faults the owner saw are the model's, not the look's."*

**What this arc can honestly promise, and what it cannot.** After §5's recommendations land, the belt
stand will show ranked ridges with valleys that join into a river that reaches the sea, and the plain
will no longer read as dunes. **It will still be bare brown ground under a plain sky**, because the
forest is 8e, the haze and the clouds are 8s, the cliffs are 8f and the shadow is owed. ★ **Nobody
should look at the first picture after this arc and expect the reference frame. They should expect the
reference frame's SILHOUETTE.**

### 4.3 The gaps no published procedural model closes

These are the items where the literature itself says "author it or buy the asset":

1. **Vegetation as geometry.** Cordonnier et al. 2017 couples an ECOSYSTEM to erosion and is the right
   law for WHERE plants grow ([10.1145/3072959.3073667](https://doi.org/10.1145/3072959.3073667)) —
   ★ but measure its cost before adopting it: its own Table 1 puts one simulation step at **38 s at
   n = 1024**, with worst-case O(n⁴) and the runoff event taking 75 % of it, on **10 m cells over at
   most 10 × 10 km**. It also says nothing about what a tree LOOKS like; ruling V4 settled that — art
   assets, blended dynamically. Far Cry 5's **viability** rule and Horizon's **ecotope** are the
   cheaper published shapes for 8e.
2. **The 3-D landforms — arches, overhangs, slot canyons, hoodoos, stratified cliffs.** A heightfield
   cannot hold them, and Grenier et al. 2024 says so of its own family: *"… do not lend themselves to
   modelling overhangs."* The cure is Paris et al. 2019, *Terrain Amplification with Implicit 3D
   Features*, ACM TOG 38(5) ([10.1145/3342765](https://doi.org/10.1145/3342765)), with **MIT code** at
   [`aparis69/Implicit-Volumetric-Terrains`](https://github.com/aparis69/Implicit-Volumetric-Terrains).
   **8f's material, already bought.**
3. **Atmosphere and cloud.** 8s owes it; nothing here touches it.
4. **Meanders and oxbows.** The lakes report ruled them out of the macro solve; the fine solve reaches
   64 m, still coarser than a meander belt. Paris et al. 2023, *Authoring and simulating meandering
   rivers* ([HAL](https://hal.science/hal-04227965), MIT code at
   [`aparis69/Meandering-rivers`](https://github.com/aparis69/Meandering-rivers)) is the model when a
   slice wants it. **Not this arc.**
5. **Colour grading and post.** The client's, and no slice names it yet.

---

## 5. RECOMMENDATIONS, RANKED

Each carries its cost in days, its law, its calibration body and a gate that can fail. They sit inside
ruling B2 step 4 — *"the river lines … at the coarse nodes first, then into the fine rungs"* — and they
replace the DRAWING in that step with a SOLVE.

★ **The owner's expectation, checked as asked.**
(a) *a per-tile local erosion solve fed by the macro field, shipped as data* — **CONFIRMED, and better
sourced than expected: tiled erosion with a border copied every step is published and measured (Šťava
et al. 2008), and the global half has an exact tiled solution (Barnes 2016). One correction: it can
ship as DATA only down to about 1 024 m; below that the data does not fit and it must be DERIVED on
both hosts (§2.6).**
(b) *the noise recipe kept only as a texture on the slopes the erosion leaves* — **CONFIRMED and
strengthened: Grenier et al. 2024 says a gradient-aligned noise CANNOT do more than that, and on flat
ground makes exactly our dune artefact.**
(c) *the dressing from 8e on top* — **CONFIRMED, with §4.2's warning that a dressing cannot repair a
shape, and §1.5's evidence that four people made Horizon's whole look.**

### R1 — ★ FIRST: THE BENCH. **3 days**

Write the four operators on `Gi` — MFD flow routing at an integer exponent, clamped stream power,
thermal stabilisation, deposition — and measure three numbers on one macro tile of the home planet:
**nanoseconds per cell per iteration on one core** (at 1 024 m and at 64 m); **the halo width at which
a patch's interior stops changing**; and **the iteration count at which the drainage stops moving when
it is seeded from the macro discharge rather than from zero**. Diff the result against
[`MultiScaleErosion`](https://github.com/H-Schott/MultiScaleErosion) (MIT) to a stated tolerance.

- **Law and precedent:** ruling F7's own order — *"the bench, the discussion, the build"* — and ruling
  F9's budgeted card.
- **Why first:** §2.6's two anchors differ by **370×**. At 20 ns per cell-iteration everything below is
  comfortable; at 367 ns nothing under 1 024 m can be baked on a CPU and the card becomes mandatory.
  **No plan should be written across that gap.**
- **Gate:** three numbers, each able to fail.
- **In the game:** nobody sees anything. This decides whether the pilot's valley is baked on the
  server or built under her as she descends.

### R2 — SHIP ONE AMPLIFIED LEVEL IN THE ARTIFACT. **8 days**

Run Schott et al. 2024's loop once on the server over the whole macro field — bicubic upsample, MFD
routing seeded from the row's discharge, clamped stream power with the hardness field, thermal
stabilisation, deposition, then the diffusion retargeting that HOLDS the macro heights and the
multi-scale breach — down to **2 048 m** (rung 11) and, if R1's number allows, **1 024 m** (rung 10).
Store it beside the pyramid; ship it in the tile the client already asks for: **128 KB or 512 KB a
tile against today's 40 KB (§2.6).** One artifact version bump, no new lane, no new payload kind, and
**nothing new crosses a realm boundary** (SL6) — it is more of a row the owner already approved.

- **Laws:** Schott, Galin, Guérin, Peytavie & Paris 2024, [10.1145/3658200](https://doi.org/10.1145/3658200);
  thermal from Musgrave, Kolb & Mace 1989; stream power from Whipple & Tucker 1999 and Braun & Willett
  2013, which our macro sweep already implements.
- **Calibration body: Earth, through Hack's law.** Schott et al. 2023 §7.3 validate exactly this way —
  `L = c·a^n`, literature spread **`c ∈ [1, 6]`, `n ∈ [0.45, 0.7]`** (Sassolas-Serrayet et al. 2018),
  their own runs measuring **`1.3·a^0.558`** and **`1.1·a^0.563`**.
- **Gates, three, each able to fail:**
  1. ★ **G-HACK** — fit `L = c·a^n` on the amplified field; `c` and `n` inside the published bands.
     **We have no such gate today at any scale, and it is the cheapest believability number here.**
  2. ★ **G-BREACH** — Schott et al. 2024's own metric, the average breaching volume. Their field scores
     **0.02–4.8 (×10³)**, procedural noise **1.7–41**, a StyleGAN **23–120**, and Grenier's
     gradient-aligned noise sits **between fBm and erosion** (86 against 128 and 25). Measure ours
     before and after.
  3. **G-RIVER unchanged** — every valley floor still descends along its trunk.
- **In the game:** a pilot at 150 km over the belt sees ranked ridges with valleys between them, not
  ripples. She cannot yet land in one and find it still there at 30 m.

### R3 — DERIVE BELOW THE SHIPPED LEVEL, PER FIXED PATCH, ON BOTH HOSTS. **12 days**

The same kernel, the same crate, compiled into the server and the client (SL10's one generator, never
a port). A patch is keyed on the **macro node address**, never on the viewer. Its inputs are the
shipped rows above it, the seed, and a halo of the width R1 measured, refreshed the way Šťava et al.
2008 refresh theirs. Its cell seeds follow Cortial et al. 2020's rule — a child's seed from its
parents' seeds. Results cache per patch; ruling F9's bounded ask already says the coarser rung stands
whole while the finer one is built.

- **Laws:** SL10; Šťava, Beneš, Brisbin & Křivánek 2008 for the border copy; Cortial, Peytavie, Galin
  & Guérin 2020, [10.1007/s00371-020-01923-4](https://doi.org/10.1007/s00371-020-01923-4), for the
  seed rule and for the proof that a planet CAN be amplified from 50 km control maps to 50 cm in real
  time.
- **Gates — the first two are new axes on gates we already run:**
  1. ★ **THE HALO GATE** — one patch built twice with two halo widths; **byte-identical interiors**, or
     the halo is too narrow. ★ Beyer's *"drain valley"* artefact is what a too-narrow halo looks like.
  2. **THE NO-DRIFT GATE, extended** — the same patch on every shipped target, byte for byte.
  3. **THE SEAM GATE** — two adjacent patches agree on their shared edge to the last bit, and
     `shore_step`'s sibling measures **0 m of step at every rung pair**, as W6 and W10 achieved for
     the sea.
- ⚠ **The warning Cortial et al. pay for and we must not:** they re-derive every ten frames, so a
  vertex's slope changes between runs and *"the snow effect … varies from one subdivision run to
  another"*. **Our patches must be CACHED and stable. A flicker is an SL8 seam.**
- **In the game:** the pilot drops into the valley she saw from 150 km. At 30 m it has side gullies
  that feed it, and the ridge beside it is a ridge she must fly over. The shard's collider reads the
  same ground.

### R4 — RETIRE THE STAMP; DEMOTE THE NOISE TO A SKIN. **3 days**

Delete the drawn tributary network from `crates/terrain/src/river.rs` — the Horton–Strahler synthesis,
the 60° junctions, the stamped channel and belt. **Keep** the Leopold & Maddock hydraulic-geometry
tables, because the solve still needs a width where it cuts a channel, and the Leopold & Wolman belt
for the floodplain the deposition step lays. Keep the octave sum **only below the finest solved
level**, and feed the hardness field `ρ` from the row's ROCK PROVINCE.

- **Law:** Schott et al. 2024 §4.2, verbatim — *"Introducing randomness in the hardness function also
  reduces the **axis-aligned artifacts produced by the regular grid discretization**."* ★ **So this
  item also cures ruling B2 step 3's scratches, for free — and Houdini sells the same fix as a "Grid
  Bias" slider.**
- **Gate:** ruling B2's G-GRID long-axis histogram, run on the amplified field, flat against the
  grid's own directions.
- **In the game:** no blue band is ever drawn. The river is where the ground is lowest, and it is one
  cell wide only where the discharge says one cell is enough.

### R5 — THE DRESSING, AND IT IS NOT THIS ARC. **0 days here**

8e's biomes and soil, ruling V4's tree and grass assets, 8s's aerial perspective and sky dome and
shadow, 8f's implicit 3-D cliffs (Paris et al. 2019, with MIT code). §4 lists what each owes.
★ **Do not start any of them before B1's three stands look like a planet in silhouette: §4.2's whole
argument is that a dressing cannot repair a shape.**

### The order, and the one thing that can stop it

**R1 → R2 → R4 → R3 → R5.** R4 moves ahead of R3 because deleting the stamp makes R2's pictures
judgeable; leaving it in would show the owner a solved valley with a drawn slash across it.

**The one thing that can stop the plan is R1's number.** If the integer kernel cannot be brought under
about 20 ns per cell-iteration on one core, R2's shipped level stops at 2 048 m, R3 needs the card,
and the arc owes a second discussion. **That is why R1 is three days and comes first.**

---

## 6. WHAT IS STILL OWED, AND WHAT I COULD NOT OPEN

### The measurements this report does not have
1. ★ **R1's three numbers.** The integer kernel's cost per cell-iteration, the halo width, and the
   seeded drainage's iteration count. **Everything in §5 is sized on a 370× guess until they exist.**
2. **G-HACK and G-BREACH on our own field, today.** Both are cheap, both read the artifact we already
   have, and neither has ever been run here.
3. **Whether Gaea or World Machine erode per tile or erode whole and cut.** Gaea's worked example
   implies whole-then-cut; World Machine states outright that its tiles cannot see outside themselves.
   Neither vendor publishes a bake time; **World Creator's 0.183 s at 4096² is the only absolute
   figure in the industry.**

### Sources that refused to open
Vanek et al. 2011 (IEEE paywall — the tiled-patch precedent Schott et al. 2024 cite by name, so its
boundary handling is the one paywalled thing I most wanted); Musgrave, Kolb & Mace 1989 at first hand;
Mei, Decaudin & Hu 2007's equations and grid sizes; the GDC Vault (paywalled) for Ghost of Tsushima's
*Samurai Landscapes*, Red Dead Redemption 2 and Far Cry 6; QuadSpinner's CPU-versus-GPU benchmark
values (they exist only inside chart images); Pearl Abyss's two 2026 talks, which are announced and
summarised by press but not published. The session's web-search budget was spent before this report
began, so every source here was reached by a direct fetch or an API.

### Three corrections recorded on purpose
1. ★ **"Terrain Erosion Synthesis with Fluvial Amplification" does not exist.** Checked against HAL,
   Crossref and OpenAlex. The real amplification work is Schott et al. 2024 and Cortial et al. 2020,
   and the second was not in the brief at all.
2. **Musgrave, Kolb & Mace 1989 has no evaporation term** — no `Kr`, no `Ke`. Those are later
   additions (Beneš & Forsbach 2002, Olsen 2004) that the games literature attributes to Musgrave.
3. **Gaea does not document GPU erosion**, and its threaded erosion is documented as
   **non-deterministic** with a toggle that forces one core. The industry meets our own law and pays
   for it the same way.

---

## R1 THE BENCH — MEASURED

Written 2026-09-22, after the owner's go on R1. **This section reports MEASUREMENTS.** Everything
below came out of `cargo run --release -p vd-bins --example erosion_bench`, on a quiet machine, one
job at a time. Where a number could not be measured the section says so and says why.

**The machine.** Apple M4 Pro, ONE core, release build, load under 2 before each run (the runner
waits for two quiet samples and prints the load it started at).

**The tile.** THE home planet, solved once at start-up through `vd_bins::artifact_worker::run_solve`
— not read from the dev-cluster's store, because a store holds whatever `ARTIFACT_VERSION` it was
last written at, and a bench whose input depends on when somebody last flew is not a measurement. The
solve takes **110–128 s** and answers `ARTIFACT_VERSION 6`, sea 3 566 m. The lattice is 1 216 nodes a
face edge, 8 871 936 nodes, node 8 192 m. The tile is the 64 × 64-node tile nearest the belt
direction the owner flew, pulled one tile inward off the face's rim so a 64-cell halo never crosses a
cube seam: **face 5, tile (2, 17), nodes [128..192) × [1088..1152), 524 km a side.** Its relief is
**12 602 m**. The macro heights are upsampled bicubically (Catmull-Rom); the hardness is the nearest
node's rock province through `Province::erodibility_q8`.

**Where the code is.** The four operators are in **`crates/recipe/src/amplify.rs`** — not
`crates/terrain` — because ruling F7 makes the recipe ONE SOURCE for the server's CPU, the client's
CPU and (through rust-gpu) the client's GPU, F8 decision 5 says one crate two compilations, and R3
asks for the fine ground to be derived on both hosts from the same kernel. `vd-terrain` names `Gf`,
holds `Vec` and reads the store; a compute shader holds none of those. Each operator is a pure
function of a cell and its eight neighbours — no allocation, no float, no division, no `std`. The
host loop that walks a tile and ships the halo is **`crates/bins/examples/erosion_bench.rs`**. Twelve
unit tests, each on a stated tiny field with the answer computed by hand.

### The three numbers

#### 1. NANOSECONDS PER CELL PER ITERATION, one core

| resolution | grid | ns a cell an iteration |
|---|---|---|
| **1 024 m** | the whole tile, 512 × 512 = 262 144 cells | **331.8** (median of 3) |
| **64 m** | 32 × 32 nodes, 4 096 × 4 096 = 16 777 216 cells | **332.5** (median of 3) |

Across four runs of the whole bench the 1 024 m figure read **320, 322, 372, 332 ns**; take **≈ 330 ns
with a ±15 % spread between runs**.

★ **THE COST DOES NOT GROW WITH THE GRID.** 16.8 million cells cost the same per cell as 262 thousand.
The loop is COMPUTE-bound, not memory-bound, which is why the 64 m row is not worse than the 1 024 m
row, and why a bigger patch buys nothing and costs nothing extra per cell.

★ **ONE QUARTER OF THE TILE AT 64 m, AND WHY.** The whole tile at 64 m is 8 192 × 8 192 = 67 million
cells; its buffers are about **4.8 GB**, which is unreasonable on this machine. The measurement is
therefore one quarter of the tile (half its side), and since the cost per cell does not move with the
grid, nothing is lost.

★ **WHERE THE TIME GOES** — the five passes timed apart, 1 024 m, median of 3:

| ns | share | pass |
|---|---|---|
| 76.1 | 22.9 % | ① the routing pre-pass — 8 slopes, 8 squares, and ONE `recip_pow2` at 40 bits (41 steps) |
| 56.5 | 17.0 % | ① the drainage gather — 8 weights, 8 products |
| **108.2** | **32.6 %** | ② clamped stream power and the incision — TWO `isqrt` (32 steps each), the clamps, the floor |
| 28.7 | 8.7 % | ③ thermal stabilisation — 16 excesses |
| 62.2 | 18.8 % | ④ the sediment gather and the deposit |
| 331.6 | 100 % | the five added |

★ **THE ACTIONABLE FINDING: over half the cost is the recipe's integer ROOT and RECIPROCAL.** Pass ②
is the biggest single item and it is two 32-step roots (`a^{3/4} = √a · √√a`); the routing pre-pass
carries the only division-shaped loop in the whole kernel. A cheaper `a^{3/4}` (a table, or a
leading-bit Newton seed instead of the restoring loop) is the one lever that moves this number, and
it is UNMEASURED.

**Against §2.6's two anchors.** 0.99 ns on a card, 367 ns for our macro node-pass. Ours is **335× the
card** and **0.90× the macro sweep's ceiling** — so the ceiling §2.6 called *"almost certainly a bad
one"* turns out to be almost exactly right. **The 370× gap is real, and it lands on the wrong side
for a CPU bake below 1 024 m.**

**The host loop is the straightforward one** — five separate walks of the tile, each rebuilding the
eight-neighbour stencil with its bounds tests. An interior fast path (no bounds tests, `present` a
constant) and a fused walk are **UNMEASURED**.

#### 2. THE HALO WIDTH

64 iterations at 1 024 m. The compared region is **the tile itself** (512 × 512, the halo excluded),
byte for byte against a **SEPARATE run at a 128-cell halo** — so the 64-cell row is tested against
something it is not. (A trimmed centre would be a vacuous test: a stencil carries news one cell an
iteration, so any centre 64 cells in from the rim is untouched whatever the halo says. The tile's OWN
RIM is where a missing halo shows, and that rim is exactly the seam a neighbouring patch meets.)

| halo | cells differing of 262 144 | the worst |
|---|---|---|
| 0 | 11 316 | **31.5 m** |
| 4 | 9 183 | 10.6 m |
| 8 | 4 745 | 3.69 m |
| 16 | 846 | 1.62 m |
| 32 | 160 | 0.798 m |
| **64** | **0** | **0.000 m** |

★ **THE HALO IS THE ITERATION COUNT.** At 64 iterations a 64-cell halo is byte-identical to a
128-cell one; 32 cells is not. §2.5's condition 2 — *"a HALO at least as wide as the iteration
count"* — is confirmed by measurement, and it is TIGHT, not conservative. **A patch of `E` cells
running `N` iterations must build `(E + 2N)²` cells.**

★ **AND THE PRACTICAL NUMBER BESIDE THE STRICT ONE.** Half the iteration count (32 cells) already
leaves only **160 of 262 144 cells — 0.06 % — differing, the worst by 0.8 m**. Byte-for-byte is what
SL10 asks for and it needs the full width, so **64 it is**; if a later ruling ever accepts a tolerance
instead of identity, half the width buys most of it.

#### 3. THE ITERATION COUNT the drainage takes to settle

**The stopping rule, STATED** (the paper gives none; Schott et al. 2023 §4.2 only report convergence
*"after several iterations ranging between `n` and `4n`, with an average of `1.5n`"*): a cell has
MOVED when its drainage changed by more than **one part in 256** between two iterations; the loop has
stopped when fewer than **0.1 %** of the cells moved. The grid is 512 cells wide, so `1.5n` would be
768 and the 300-iteration budget cannot reach it.

★ **TWO REGIMES, because the erosion moves the ground under the routing.** The field HELD STILL
(`k = 0`, `k_γ = 0`) isolates the routing operator, which is what the question asks. The field
EVOLVING is what a real run does.

| regime | seeding | settled |
|---|---|---|
| held still | from ZERO (the reference's own start) | **iteration 288** |
| held still | EVERYWHERE from the macro row's discharge (§2.5 literally) | **STILL MOVING at 300** |
| held still | at the INFLOW BOUNDARY from the macro rows (corrected) | **iteration 288** |
| evolving | from ZERO | STILL MOVING at 300 |
| evolving | EVERYWHERE from the discharge | STILL MOVING at 300 |
| evolving | at the INFLOW BOUNDARY | STILL MOVING at 300 |

★ **§2.5's SENTENCE IS WRONG AS WRITTEN, AND THE MEASUREMENT SAYS WHY.** §2.5 reads *"seed the
drainage area of every fine cell from the macro row's DISCHARGE, divided down by the refinement
factor"*. Done literally, the drainage converges **SLOWER, not faster** — it does not settle at all
inside 300 iterations. The reason is the operator's own algebra: the gather is
`a ← seed + Σ a_q·w(q→p)`, so its FIXED POINT is set by what each cell CATCHES (`a_seed`), never by
what `a` starts at. A large start is a transient that has to drain off the tile before the answer
appears. **The correct form of the same idea injects the outside catchment where it actually enters**
— on the rim cells whose macro node just outside drains INWARD, its discharge shared over its edge
cells, added to `a_seed` — and that lands on 288, the same as from zero, while carrying the globally
correct water. **The idea is right; the sentence needs rewriting before R3 reads it.**

★ **288 is 0.56 n, well under Schott's `1.5n` rule of thumb** — a cheaper convergence than the
literature's, on our field.

★ **AND ON AN EVOLVING FIELD NOTHING SETTLES IN 300.** The drainage chases a ground that is still
being cut. **R2 must state its iteration budget as a fixed count, not as a convergence test**, which
is what Schott et al. do (Table 1: 2 600 iterations at 128, falling to 300–800 at 4 096).

### The two believability numbers, before and after

At 1 024 m, the whole tile, 300 iterations. **The loop moved the ground 32.6 m on average and 226.4 m
at the worst** — printed on purpose, because a gate computed on a field the loop never touched is not
a measurement (see "the two defects" below).

| gate | BEFORE (the bicubic upsample) | AFTER 300 iterations |
|---|---|---|
| **G-HACK** | `L = 1.99·a^0.883`, 159 506 network cells — `c` inside [1, 6], **`n` OUTSIDE [0.45, 0.7]** ✗ | **`L = 5.50·a^0.499`**, 99 297 network cells — **`c` INSIDE, `n` INSIDE** ✓ |
| **G-BREACH** | 1.409 × 10⁶ m³ a depression, 24 030 depressions | 1.314 × 10⁶ m³ a depression, 24 029 depressions (**−6.7 %**) |

★ **G-HACK IS THE RESULT OF THIS BENCH.** A gate that could have failed, failed before and passed
after. `n = 0.499` sits in the middle of the published band and close to Schott et al. 2023's own
measured `0.558` and `0.563`; `c = 5.50` is inside [1, 6], near its top. **A bicubic upsample of the
8 km field does NOT obey Hack's law; four passes of integer erosion make it obey Hack's law.** That
is the first believability gate this project has at any scale.

★ **G-BREACH BARELY MOVES, AND THE REASON IS KNOWN.** Ours falls 6.7 %; Schott et al.'s falls by one
to three ORDERS of magnitude against procedural noise. **We did not build the two operators that
drive their number down** — the height retargeting (§5.1) and the multi-scale breach (§5.2, a Barnes,
Lehman & Mulla 2014 breach across levels). R1 was asked for four operators; those two are the
difference, and **R2 must include them or G-BREACH will not move.**

★ **G-BREACH's ABSOLUTE VALUE CANNOT BE SET BESIDE THEIRS.** They publish the metric's VALUES (their
erosion 0.02–4.8 ×10³, procedural noise 1.7–41) but not the code that computes them, and their unit
is not stated. Our definition is written out in the bench: a priority flood from the rim records each
cell's predecessor; for every pit we walk that path outward and add `max(0, h − h_pit)` times the cell
area — the material a one-cell-wide channel at the pit's own level must still remove. **The
before/after ratio on the same field with the same definition is the number that can fail; the
absolute comparison with the paper is UNVERIFIED.**

### The whole planet's cost, extrapolated

One tile at 1 024 m, 300 iterations, **MEASURED wall time: 28.9 s.** The planet is **2 166 tiles** of
this size (6 faces × 19 × 19).

| job | on one core | on eight cores |
|---|---|---|
| **the whole surface to 1 024 m, 300 iterations** | **17.4 core-hours** | **≈ 2.2 h** |
| the same, derived from 1.70 × 10¹¹ cell-iterations at 331.8 ns | 15.7 core-hours | — |
| one artifact tile (524 km) to 64 m, 300 iterations | **111 min** | ≈ 14 min |

The two roads to the planet number agree to 10 %, which is the check that the extrapolation is sound.
Ruling F6 gives the terrain workers a SHARE of the cores, never all, so the eight-core column is a
ceiling, not a plan.

★ **WHAT THIS DECIDES (R2 and R3).** A one-off server bake of the whole planet to 1 024 m costs about
**17 core-hours** — a day's work once, for a world that never re-runs it. That is affordable, and R2
is clear. **Below 1 024 m a CPU bake is not affordable and the data does not fit anyway** (§2.6's
table: 4.5 GB at 512 m), so R3's derivation stands as the only road — and at 330 ns a cell on one
core, **a patch built under a descending pilot needs the card, exactly as ruling F9 anticipated when
it made the card a budgeted second builder.** One artifact tile to 64 m is 111 minutes on a core; the
same work at the card's 0.99 ns is 20 seconds.

### The diff against the reference

★ **WE COULD NOT RUN [`H-Schott/MultiScaleErosion`](https://github.com/H-Schott/MultiScaleErosion)
(MIT).** It is a C++ OpenGL application with GLFW, GLEW and ImGui, and this machine has no such
harness. **The comparison against its RUNNING output is UNVERIFIED, and is stated so.**

What CAN be diffed, and is: a **float transcription of its four shaders' own arithmetic**, written
from the source read at their raw URLs, run beside our integer operators on one stated field. We
copied no code. Their uniforms, read out of the source: `flow_p = 1.3`, `k = 0.0005`, `p_sa = 0.8`,
`p_sl = 2.0`, `max_spe = 10000`, `dt = 1.0` (`erosion.glsl`); `eps = 0.00005`,
`tanThresholdAngle = 0.57` (`thermal.glsl`); `deposition_strength = 1.0` and the two `0.1`
(`deposition.glsl`).

| operator | against the transcription at OUR stated exponent | against the reference's OWN exponent |
|---|---|---|
| ① MFD routing | **7.63 × 10⁻⁷ of one** (the integer rounding) | 0.0888 of one, at their `flow_p = 1.3` |
| ② clamped stream power | **4.55 × 10⁻¹⁶ relative** at our `m = 3/4` | 36.9 % low, against their `p_sa = 0.8` |
| ③ thermal | **1.59 × 10⁻⁵ of a unit slope** | the same operator; their constant carried verbatim |
| ④ deposition | **1.01 × 10⁻⁶ and 6.64 × 10⁻⁸ relative** | the same operator; their constants carried verbatim |

The tolerance the formats set is **one part in 65 536 of a metre (15 µm)**, and every row is inside
it. **The two rows that are NOT tiny are the two stated departures, not defects:**

1. **The routing exponent is `p = 2`, not the published `1.3`** — R1 asked for an INTEGER exponent,
   and 1.3 is not one. Holmgren 1994 chose 1.3 *"to avoid sharp fluvial incision produced by high
   exponents"*; Hyväluoma, Thorne & Turunen 2017 measured the grid-isotropy optimum of the same
   exponent at `W ≈ 1.3–4.1`, so **2 sits inside the published band**.
2. **The drainage exponent is `m = 3/4`, not the published `0.8`** — `4/5` needs a fifth root the
   recipe does not hold; `3/4` is two of the root it does hold (`√a · √√a`), exact on every host. Our
   `m/n` is then **0.375 against the shader's 0.4**, and the paper's own text says *"typically"* 0.5.

A third, smaller departure: the thermal exchange is written as a PAIR (one excess per pair, opposite
signs on the two sides), so mass is conserved by construction; their shader accumulates a
`receiveMul` and a `distributeMul`, which a count is not.

### ★ TWO DEFECTS THE BENCH FOUND IN ITSELF, both MEASURED

These are the reason this section can be trusted, and both are rules for R2 and R3.

1. ★ **AN INCREMENTAL LOOP'S STEP MUST BE REPRESENTABLE.** The height format was first 8 fraction
   bits (1/256 m, 3.9 mm) and **the whole loop did nothing**: over 300 iterations the tile's ground
   moved *"0.000 m on average, 0.0 m at the worst"*. One iteration's cut on an ordinary cell — a
   drainage of a hundred cells at a slope of a tenth — is `80 × 838 >> 24`, which TRUNCATES TO ZERO
   at 8 fraction bits. Only a cell carrying a real river cut anything; the rest of the tile was
   frozen for ever, and the believability gates read identically before and after, which looks
   exactly like a converged answer. The format is now **16 fraction bits (15 µm)**. **A field a loop
   ADDS to needs the resolution of ONE STEP, not of the answer.**
2. ★ **THE REFERENCE'S TWO RATES CANNOT BE CARRIED OVER, AND THE BENCH DERIVES THEM.** Its
   heightfields live in a normalised domain where a slope is of order one; ours are metres over the
   ladder radius, where a macro-upsampled slope is of order a hundredth. The stream power goes as the
   slope SQUARED, so the shipped `k = 5 × 10⁻⁴` reads **four orders of magnitude too small** here.
   The bench therefore DERIVES the two rates from a stated target on the tile itself, the way ruling
   T9 asks (*"every physical fact is COMPUTED by a published law with a stated calibration body"*),
   and prints them:
   - `k` is set so the cell at the **99th percentile** of the initial stream power — a river, not a
     hillside — removes **5 % of the tile's own relief** over the run. On this tile: relief 12 602 m,
     the 99th-percentile stream power 5 raw, so **`k = 1 804 130 912` at 2²⁴**, and the target is
     **2.100 m an iteration**.
   - `max_spe`, the reference's outer clamp, is the **99.9th percentile** of that same initial stream
     power — the same clamp the paper wants, expressed as a percentile of THIS field instead of as a
     number from another unit system. On this tile: **20 raw**.
   - `k_γ` follows from the same target: a slope one talus-width over the talus moves the same height
     per iteration. On this tile: **61 818 384**.

   ★ **THE OWNER'S NUMBER IS STILL OWED.** The believable calibration is Hack's law on Earth, which
   is R2's own gate; this one only makes R1's before/after comparison mean something. The reference's
   other constants — `tanThresholdAngle = 0.57`, `deposition_strength = 1.0`, the two `0.1` shares,
   `s_max = 1.0` — are carried over verbatim.

### What could not be measured, and why

- **Running the reference implementation.** C++/OpenGL/GLFW/ImGui, no harness on this machine.
  UNVERIFIED. The diff above is against a transcription of its shader source.
- **G-BREACH against the paper's own numbers.** Their unit is not published. UNVERIFIED. Only our
  before/after ratio is a measurement.
- **The whole tile at 64 m.** 67 million cells, about 4.8 GB of buffers. One quarter was measured
  instead, and the cost per cell does not move with the grid, so nothing is lost.
- **A tuned host loop.** The interior fast path and the fused walk are UNMEASURED; so is a cheaper
  `a^{3/4}`, which the pass breakdown says is the one lever worth pulling.
- **The GPU.** Nothing in this bench ran on the card. The 0.99 ns anchor is still Schott's Table 2.
- **Two hosts agreeing byte for byte.** The SL10 no-drift gate is R3's, not R1's.
- **COVERAGE IS UNMEASURED.** This bench never runs `llvm-cov`, and `just coverage-fast` was not run.
  `crates/recipe` is Tier-A, so `amplify.rs` owes HR5's 100 % before it can land in a gate; its twelve
  unit tests were written for that, but the number is UNMEASURED.

### The gates that were run

- `cargo fmt --all` — clean.
- `cargo clippy -p vd-recipe --all-targets -- -D warnings` — clean.
- `cargo clippy -p vd-bins --example erosion_bench --features dev-control,render -- -D warnings` — clean.
- `cargo test -p vd-recipe --lib amplify` — **12 passed, 0 failed.**
