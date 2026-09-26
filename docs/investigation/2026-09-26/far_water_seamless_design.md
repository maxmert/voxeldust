# FAR WATER WITH NO SEAM — THE DESIGN FOR RULING B16

**What this file is.** A DESIGN. It changes no source file and no world file. It answers ruling B16
(`docs/design/owner_decisions_2026-09-22_believability.md`, B16): *"We need seamless experience - no flicker at all
please. We need to find the best solution for it. No flicker or changing of the shapes between rungs at all."* The bar
is W16 in full (`docs/design/owner_decisions_2026-09-21_water.md`, W16 items 1–3): one shape from one field; a finer
rung refines a coarser rung and never contradicts it; a rung boundary is invisible.

It stands on branch `game`, HEAD `c399f54`: `GENERATOR_VERSION` 21 (`crates/terrain/src/tag.rs:233`),
`ARTIFACT_VERSION` 9 (`crates/terrain/src/artifact.rs:95`), `PROTO_MINOR` 34 (`crates/wire/src/version.rs:349`).
Three read-only maps of the code fed it (the renderer, the data path, the prior art), three designs (A: the water as a
share; B: the nested curve; C: the rung-free skin) and two reviews of the three (the W16 lens; the feasibility lens).
The designer read the load-bearing anchors again in the code. **MEASURED** marks a printed number and names its
census. **ARITH** marks hand arithmetic. **UNMEASURED** marks a number that nobody printed. **PREDICTED** marks a
reading of a static file, not a run.

**The words of this file.** Each word has one meaning here.

| word | meaning |
|---|---|
| a swap | the handover from rung L to rung L + 1 at the switch distance s_L = 869.1 m × 2^L |
| the band | the crossfade band of a swap, 0.9 s_L to 1.1 s_L (`crates/client/src/ladder_view.rs:81-84`) |
| the front | the sphere of radius 1.1 s_L about the eye, where the fragment stage discards rung L (`crates/client-render/src/ladder_fade.wgsl:156-158`) |
| t | how far rung L has morphed onto rung L + 1: 1 − whole(d); 0 at the band's inner edge, 1 at the front |
| a pop | a change of a pixel in one frame |
| a slide | a change that runs over a band, at the pace of the pilot's motion |
| the share, c | the wet fraction of N2's rung-7 point samples under a footprint |
| the skin | the integer pyramid of the share, drawn as a texture |
| a window | the piece of the skin that one chunk carries (rungs 6–13 on the home planet) |
| the far page | the one piece of the skin that a body carries for rungs 14–18 |
| the hand-over | the blend from the sheet's own colour to the skin's colour, by the pixel's footprint |
| the lid | the surface of a far column from rung 14: the mean of what a pilot sees over it |
| a stand-in | the coarser chunk that stands sunk where a finer chunk is late |

---

## 1. THE SHORT ANSWER

### 1.1 The recommendation

**DESIGN D — THE RUNG-FREE SKIN WITH THE LID.** D is design C, with two parts of design A and every fix of the two
reviews. D has four parts:

1. **THE SKIN.** The colour of the water in a pixel is the SHARE of ONE field under that pixel's own footprint. The
   field is N2's point side at the 128 m samples of rung 7: the samples that rulings B14 and B15 already count. The
   share comes from ONE integer pyramid. Every rung reads the same texels of that pyramid. So no rung decides where
   water is, and nothing compares a count with a threshold. The skin draws all water farther than 111.2 km from the
   eye. Rungs 6–13 carry a window; rungs 14–18 read one far page per body (design A's).
2. **THE NEAR SHEET, AND ITS DECISION MORPHS.** Nearer than 55.6 km, W18's crisp water sheet stays. Inside each band,
   the finer rung's water now moves onto the coarser rung's water (design C's morph). So a changed shore slides once
   across the band. Today it stays to the front and then pops.
3. **THE LID.** From rung 14, the surface of a far column is the mean of what a pilot sees over it: the water's level
   over its wet fine nodes, the ground over its dry fine nodes. No side word, no fold, no hold, no stencil and no
   pocket word decides a far height.
4. **THE RENDERER'S OWN POPS, FIRST.** The water fades on its own ground's band. No band edge jumps 5 % in one frame.
   A late chunk ramps in. The water's culling box covers its morph.

**Example.** A pilot climbs over the high lakes from 7 500 km to 60 000 km. Her pixel over the 5 882 m lake's ragged
piece has the same blue on both sides of every ring. The blue changes only as her pixel grows: about 64 % blue near
14 000 km, about 40 % near 28 000 km, about 59 % near 57 000 km. It changes at the pace of her climb, never at a ring
and never in one frame. The ground under the lake's shore does not rise or drop at a ring either.

### 1.2 What the owner sees

| distance from the eye | what the owner sees | example |
|---|---|---|
| more than 111.2 km | No ring changes any water pixel. A far coast is a soft edge about one pixel wide. A lake narrower than a pixel is a pale tint of its pixel. The tint follows the pixel's footprint, slowly. | From 30 000 km a 5 km tarn is a faint blue cast on its 35 km pixel. From 3 000 km it is a small blue shape. |
| 55.6–111.2 km (rung 7's ring) | The crisp coast softens into the soft coast, by distance alone, over one doubling of distance. | A hull climbs off the gentle coast. Between 56 and 111 km the beach line loses its hard edge. No ring marks the change. |
| less than 55.6 km (rungs 0–6) | W18's crisp coast. At a ring the coast slides once, slowly, across the band. It never pops. | A pilot walks along a belt lakeshore. The far shore, 0.8–1.0 km away, slides by up to 3 m (3.3 px) as the rung-0 ring passes over it, then stays. |
| the far land (rung 14 and up) | No far column rises or drops at a ring because of a count. The relief still morphs across each band, as today. | From 30 000 km the second pocket's floor is a basin at every ring. No ring lifts it into a mesa. |
| every ring | No band edge jumps. The water never fades on another band than its ground. A late chunk rises in over half a second. | On a descent at 1 600 km/s over the belt, the rings slide with the hull. |

### 1.3 Is W16 met literally?

| W16 item | the water's colour, farther than 111.2 km | the near shore, rungs 0–7 | the land's shape |
|---|---|---|---|
| 1. one shape from one field | YES. The skin is the box filter of N2's point side. | YES. The sheet draws the line of the same field. | YES with the lid: one mean surface. |
| 2. a finer rung refines a coarser rung and never contradicts it | YES, exactly. A coarser texel is the mean of its four finer texels. | The line refines inside one coarser cell. It moves by up to one coarser cell at a swap. | The relief: as today (a coarser rung drops octaves). |
| 3. a rung boundary is invisible | YES at every front, to one R8 step (ARITH: at most 1.5 % of the water's own luminance, under W16.5's 2 % gate). | At the front YES: the morph ends on the coarser picture. Inside the band: one slide. | The relief morphs across the band (T7), as today. |

### 1.4 What remains, and why no design can remove it

1. **A pixel cannot hold more than a pixel.** A far lake narrower than a pixel must show as a tint or as a threshold
   (in or out). A threshold on nested footprints flickers. The bench measured it under B15 item 1: the box fold makes
   260 of 693 lake bodies flicker on the steep coast's far window (MEASURED, `n2_s0_fold_kernels_w4_census.txt`). The prior art
   agrees: *"even perfect per-level thresholds do not prevent geometry disappearing with distance"* (Wyman and
   McGuire 2017). So D draws the tint. The tint changes with the pixel's footprint as the pilot climbs: at the pace of
   her climb, never at a ring.
2. **A coarser rung cannot hold the finer rung's line.** At a swap, a coarser cell is two pixels. So a drawn line must
   move by up to one coarser cell at some swap. Design B bounds that move at 1.41 px with exact nested crossings, and
   its line still moves. D makes each move one slow slide per band. Its size is N2's own step at rungs 0–7, MEASURED
   (`n2_s0_w4_cures_census.txt`): the gentle coast's sea at most 1.46 px; the belt's lakes at most 3.29 px; the
   second pocket's lakes at most 4.79 px (the lake lines hold fewer than 200 pairs, so they carry no verdict).
3. **A coarser rung drops octaves.** The land's relief changes across each band, and the morph carries it (W16, W17,
   T7's one-cell tolerance). No drawing of the water changes this.
4. **Smaller residues**, each stated in §3: the softening in the hand-over band; the resolution floor (water narrower
   than 128 m leaves the picture between 56 and 111 km); the rim of a grazing view past 86.4° (§2.3 item 7); the one
   coastal cube corner (Q12); the rounding of an R8 texel and of the GPU's filter weights (one step in 255).

---

## 2. THE MECHANISM

### 2.1 The one quantity

Π(l, τ) is the number of WET rung-F point keys under the global texel τ of level l. A level-l texel is one rung-l
cell: 128 m at l = 7, 8 192 m (one node square) at l = 13, 262 km at l = 18. F = 7 on the home planet (ruling B15 Q6).
A key is N2's point side at the sample: the sign of the held side field S (`point_side`,
`crates/bins/src/shore_rules.rs:1372`). A second count, Λ(l, τ), counts the lake keys by the same rule. Today every
water has one colour, so the renderer draws only Π; Λ waits for the ocean slice (8o).

- **Π nests.** A level-(l + 1) texel covers exactly four level-l texels, because rung cells nest
  (`crates/terrain/src/artifact.rs:1817-1824`). So Π(l + 1) is the sum of its four children, in integers, on both
  hosts. MEASURED on every W4 window: CHECK-PYRAMID-FOLD 0 columns, CHECK-PYRAMID-UNIFORM 0 samples,
  CHECK-PYRAMID-BAND 0 of 4 194 304 quarters (the build design, `n2_build_design.md` §10).
- **Only the shore pays.** The held read's maximum principle gives a uniform node cell ONE key for all its 4 096
  samples (the build design, §3.2). Only the 111 089 mixed node cells of the home planet (1.25 %, MEASURED) need
  their samples.
- **The sources.** Levels 7–12 inside a mixed node cell: its 4 096 sample keys, which the one reader derives (the
  build design's record memo, §3.3) from the tiles at rungs 6–9 and from the shore band at rungs 10–13. MEASURED:
  49–54 µs a mixed cell (f64). Level 13, the node square: the shipped three words of each mixed node square (B14
  item 3, 1.100 MB); for a uniform square, the one key of each of its four quarter cells, from the mask's corner
  words. Levels 14 and up: sums.
- **The share** of a texel is c = Π / n, where n is the number of samples under the texel. The host quantizes each
  texel ONCE, from its own exact count: round(255 · c). No texel is an average of quantized texels.

**Example.** A rung-16 texel over the gentle coast is 65 km a side: 8 × 8 node squares and 2²⁰ samples. The client
sums the words of its mixed squares and the one keys of its uniform squares: 530 000 of 1 048 576 samples are wet.
Its share is 0.505, stored as 129. Its four rung-15 children hold wet counts whose sum is the same 530 000.

### 2.2 Rung by rung (the home planet)

| rungs | cell | drawn at (the unbounded ladder) | the ground (the recipe) | the water's surface | the colour | the skin's source |
|---|---|---|---|---|---|---|
| 0–5 | 1–32 m | 0–30.6 km | N2: the point side (tiles) | W18's sheet, with R2's morph | the sheet's own (φ = 0) | none |
| 6 | 64 m | 25.0–61.2 km | the same | the same | φ ≤ 0.10, in its fade-out band only | a window, 128 m texels |
| 7 = F | 128 m | 50.0–122.3 km | the same | the same | φ rises from 0 to 1 over 55.6–111.2 km; φ = 1 at its front | a window, 128 m texels |
| 8–9 | 256–512 m | 100–489 km | N2: the fold (tiles) | the sheet, with R2's morph | the skin (φ = 1) | windows, 128 m and 256 m texels |
| 10–12 | 1–4 km | 401–3 916 km | N2: the fold (the band, the records) | the same | the skin | windows at half a cell |
| 13 | 8 km | 3 204–7 832 km | N2: the fold (four quadrant records) | the same; deep water as block quads | the skin | a window, 4 km texels (quarter squares) |
| 14–18 (from R_P) | 16–262 km | 6 408 km to the whole globe | THE LID | the lid itself | the skin | THE FAR PAGE |

A window's finest texel is T_L = max(T₀, cell_L / 2), with T₀ = 128 m (the rung-F sample). A window holds five levels,
T_L to 16 T_L. The far page holds the node square (8 km) and its sums.

### 2.3 The fragment: the colour law

For a fragment of a chunk of rung L, at distance d from the eye (the morphed fragment's own distance: the one that the
front's discard reads):

1. **The share.** c = the skin, sampled with the pixel's own gradients: trilinear, anisotropic up to 16×. A vertex's
   texel coordinate (u, v) comes from its own integer lattice position. The radial morph never changes it.
2. **The level floor** (design A's form, read at the fragment): λ_eff = max(λ, t · log2(T_{L+1} / T_L)) in the
   window's own texels, applied by a scale of the gradients (`textureSampleGrad`). It is inert on the unbounded
   ladder (ARITH: over a band λ runs from 0.85 to 1.14, and the floor runs from 0 to 1). Where a bound pulls a band
   in, the floor makes the finer rung read the coarser rung's finest level at the front.
3. **The hand-over weight.** p = d · α, with α = 1.1506e-3 rad (the reference pixel). φ(p) = clamp(2p / T₀ − 1, 0, 1):
   nought where one 128 m texel covers two pixels (55.6 km), one where it covers one pixel (111.2 km). The weight of
   rung L is w_L = max(φ(p), g_L), where g_L = 1 over rung F, g_F = t at rung F, and g_L = 0 under rung F. A late chunk
   draws φ_L = mix(w_{L+1}, w_L, a): its stand-in's weight first, then its own (a is the arrival ramp, §2.7; a = 1
   for a chunk that is not late). The level floor of item 2 takes max(t, 1 − a) in place of t for the same reason.
4. **The colour** = mix(IDENTITY, mix(L_land, L_water, c), φ_L).
   - L_land and L_water are two whole lighting passes (the engine's `apply_pbr_lighting`): the land with the
     ground's base colour srgb(0.55, 0.50, 0.42), roughness 0.95 and the ground's morph-blended normal; the water
     with srgb(0.06, 0.24, 0.42), roughness 0.15 and the radial.
   - IDENTITY is the surface's own lit colour: the ground's on the ground, the water's on the sheet.
   - The two radiances mix BEFORE the fog and the air: `main_pass_post_lighting_processing` runs once, on the mix
     (Bruneton and Neyret 2012: mix the outgoing light, never the parameters).
   - A fragment with c = 0 or c = 1 runs one pass.
5. The sheet carries its ground vertex's packed normal and morph normal (+8 B a vertex). So a sheet pixel and a
   ground pixel at one place shade alike.
6. ONE land-shading function serves the ground material and the mix, so the ground colours of 8e cannot split them.
7. **The margin of a window.** Near a chunk's edge the sampler's probes reach past the chunk. The margin on EACH side,
   after the window is aligned to its coarsest level's global texels, is M = ⌈half the footprint's major axis at
   θ_rim + one texel of the coarsest level that a probe reads⌉. θ_rim = 86.4°, where the 16× sampler saturates
   (1 / cos 86.4° = 16). ARITH: at a front the minor axis is 2.2 T_L, so M = 17.6 + 4 → 22 texels, and a window is at
   most 192 × 192 texels (48 KiB at R8 with its mips). Past θ_rim the probes can reach past the margin: the rim (Q8).

**Example.** At 20 000 km a pilot's pixel covers 23 km of the belt's coast, and the far page gives c = 0.3. The pixel
is 30 % the lit sea and 70 % the lit land, and then the air. A rung-15 chunk and a rung-16 chunk draw that pixel in
the same colour.

### 2.4 The swap band, L → L + 1

The water's colour does not move at a swap (§3, P1). Inside the band these things move:

1. **The ground** slides along each vertex's radial onto the coarser mesh, as today
   (`crates/client-render/src/ladder_fade.wgsl:110-122`).
2. **THE WATER'S DECISION (R2; every rung with a sheet).**
   - `ParentMesh` keeps each vertex's water word: the side and the level. Today it holds positions, normals,
     triangles and buckets only (`crates/client/src/chunks.rs:386-405`).
   - The target of a finer water vertex is the coarser rung's own water on its radial: the parent triangle's three
     water radii, blended by the hit's barycentric weights. Where the parent triangle holds no water, the target is
     the moved ground less q.
   - The finer chunk builds a water triangle wherever EITHER rung has water. A water that only the coarser rung holds
     stands under the finer ground at t = 0 and rises to its level at t = 1.
   - The exact cut and the floor cut read both ends of the new morph (`chunks.rs:1436-1562`, `:1591-1613`).
   - Today the target keeps the finer rung's own side and level (`chunks.rs:1418-1429`). So a changed shore stays to
     the front and pops there.
3. **The skin's level floor** rises from T_L to T_{L+1}. It is inert unless a bound binds.
4. **At rung F the hand-over weight** reaches 1 at the front.
5. **At the front** the fragment stage discards rung L, as today. Past it, rung L + 1 draws the same colour and, at
   the vertices, the same shape.

**Example.** A pilot walks along a belt lakeshore and looks across the lake, 0.8–1.0 km away. At one place the rung-0
shore stands 3 m from the rung-1 shore. Today a point of that 3 m strip stays water until the front (956 m) reaches
it, then turns tan in one frame, and the front shows as a notch in the far shore. With R2 the strip's water sinks
under the rising ground between 782 and 956 m, so each of her steps moves it by a step's worth.

### 2.5 The near/far boundary: the hand-over

- The hand-over reads the pixel's footprint alone. It spans rung 7's own ring, from 55.6 km to 111.2 km: one doubling
  of distance.
- At the 6 → 7 front (61.2 km) φ = 0.10 on both rungs. The two sheets differ by N2's step, and R2 slides them.
- At the 7 → 8 front (122.3 km on the unbounded ladder) φ = 1 on both rungs. N2's shore step there (MEASURED on the
  belt's lakes: 6.92 px) changes no colour, because the skin draws both rungs.
- Where a bound pulls the 7 → 8 front in, g_F makes rung 7 all skin at its front, and rung 8 is all skin by its rule.
- THE CONDITION: rung F's two bands do not meet (s_F ≥ 1.22 s_{F−1}). SKIN-FRONT-BOUND reads it.

**Example.** A hull climbs off the gentle coast. At 60 km the beach line is W18's crisp line with a tenth of the skin
in it. At 90 km it is more than half skin. At 115 km it is all skin. No frame of the climb changes it by more than the
climb itself.

### 2.6 The lid (from rung R_P)

From R_P (14 on the home planet) a column's footprint is exactly one level node (the build design, §2.1;
`crates/terrain/src/artifact.rs:853-876`). The lid of that node is:

  **T = (Σ_wet W + n_dry · s_dry) / n**

- n = the node's fine nodes (4^k); n_wet and n_dry come from the mask's counts (`CoastCounts`, which the client folds
  today, `crates/terrain/src/artifact.rs:379-490`);
- Σ_wet W = the sum of the levels of its wet fine nodes: the sea's level for a sea node, the lake's level for a lake
  node (the lake levels of B14 item 1);
- s_dry = terrace(R + z̄_dry + fine_m) − R: the terraced mean height of its dry fine nodes (a pocket's floor is dry);
  fine_m is nought on the home planet from rung 10 (the build design, §2.1);
- z̄_dry at level 1 = the pyramid's height word where all four fine nodes are dry; for a part-wet level-1 node, the
  DRY WORD (the SL6 ask, §4.1), or without it the height word itself (the interim, T~); over level 1, the
  count-weighted sum of the level-1 values.

One sum and one shift (n = 4^k), in the recipe's integer lengths, on both hosts. The column's water word is ZERO, so
the shore law leaves the column alone (`crates/recipe/src/height.rs:412-415`), and the chunk builds no sheet. At
these rungs the column reads no side word, fold, hold, D7 stencil or pocket word. The lid applies only where the field
holds a mask; a field with no side data keeps today's column (the build design, §1.5 step 8).

What follows from the law:
- **An all-dry node** reads terrace(R + its mean) − R: today's far ground, exactly (the level-1 word, then the
  pyramid's own floored means). So every inland far column does not move.
- **An all-wet node** reads the count-weighted mean of its waters' levels: exactly the sea's level over the open sea.
- **A part-wet node** reads a continuous blend of the two.
- **A lake beside the sea** in one node reads their count-weighted mean level. No level steps at the swap where they
  merge (today `block_water` takes the highest, `artifact.rs:1598-1623`).
- **A pocket's dry floor** enters as dry ground: a basin at every rung, never lifted over the sea.
- **The lid needs no hold.** W6's reason for the land's hold is the fine octaves, which dig a dry pit under the water
  beside the sea. From R₁ = 10 up the fine half is nought, so no far column can dig that pit.
- **The lid's step** from rung 13's surface joins the body's step bound (W17, T7 rule 2), so the sink and the band's
  width stay sized on it. TOP-STEP measures it.

**Example.** The second pocket's floor stands 8.2 km under the sea's level. At rung 13 N2 draws a basin. At rung 14 the
lid of each level node over the floor is the floor's own terraced mean, so the basin stays. At rung 17 the node over
the pocket's rim holds 30 % sea: its lid stands 30 % of the way from the rim's mean to the sea's level. No ring lifts
the floor, and no ring drops the rim.

### 2.7 The renderer's own pops (R1), first

1. **THE BAND BUFFER.**
   - One GPU buffer per realm holds every rung's bands and sinks, indexed by the rung. Every terrain material (the
     ground's, the water's, the caster's, the probe's) binds it once and holds only its rung index. The binding is the
     derive's raw-buffer storage binding, `#[storage(n, read_only, buffer)]`
     (`bevy_render_macros-0.18.1/src/as_bind_group.rs:455-462`).
   - The render queue rewrites the buffer in place every frame (`RenderQueue::write_buffer`). No material asset
     changes, so the engine prepares no bind group again. PREDICTED from the derive's source; R1 proves it with
     FRAME-COST and its control.
   - Today `rebind_bands` rewrites the ground's, the casters' and the probe's materials, and NEVER the water's
     (`crates/client-render/src/terrain.rs:1921-1985`; only `:1582`, `:1727` and `:2085` read the water's map). So
     after a bound moves, the water can fade on another band than its ground.
   - Today the materials follow a horizon only after it moves 5 % (`ASK_BOUND_REBIND`,
     `crates/client/src/ladder_view.rs:360`), because the engine prepares a rewritten material again (MEASURED: 43.2 →
     38.7 frames a second). A `ShaderStorageBuffer` asset does not cure that: a changed asset gets a NEW GPU buffer
     (`bevy_render-0.18.1/src/storage.rs:131-157` ignores the previous asset), so the engine prepares every bind group
     that holds it again.
   - With the buffer, the bands follow the held horizon every frame, at the slew's pace: a quarter of the horizon a
     second (`ASK_BOUND_SLEW_PER_S`, `ladder_view.rs:338`), 0.4 % a frame at 60 Hz.
2. **THE WATER'S CULLING BOX.** The sheet child gets the box that the ground has (`moved_bounds`, `terrain.rs:1557`):
   its points, their morph targets and its sink. Today it gets none (`terrain.rs:3229`).
3. **THE ARRIVAL RAMP.**
   - A finer chunk that arrives while its stand-in stands enters on the stand-in's drawn state: its vertices on the
     stand-in's sunk surface (its morph target less the coarser rung's sink along its radial), its skin at the
     stand-in's finest level, its hand-over weight at the stand-in's.
   - Over the ramp time it relaxes to its own state: drawn = mix(the stand-in's state, its own state, a), with
     a = clamp((now − arrival) / T_ramp, 0, 1).
   - The arrival stamp and the window's layer ride in the mesh's tag (`MeshTag`,
     `bevy_mesh-0.18.1/src/components.rs:152`; the shader reads `get_tag`,
     `bevy_pbr-0.18.1/src/render/mesh_functions.wgsl:165`).
   - T_ramp = 0.5 s (Q7). ARITH: a stand-in stands sunk by at least a cell of each rung plus the dropped octaves,
     three pixels or more at its switch distance; over 30 frames at 60 Hz it rises a tenth of a pixel a frame.

**Example.** A hull dives at 1 600 km/s over the belt, and a rung-9 chunk arrives late. Today the sunk rung-10
stand-in becomes the rung-9 ground in one frame, three pixels higher. With R1 the rung-9 chunk appears exactly where
the stand-in stood and rises into place over half a second.

### 2.8 Another body (SL5)

- F = R₁ − 3, R_P and T₀ = cell(F) come from the node's size alone (the build design, §3.6).
- On a body whose node is a power of two in cells, the far page starts at R_P. On a body whose node is not (the 5 km
  body: 714 cells a node, `crates/terrain/src/macro_lattice.rs:80-82`), node squares do not nest with rung cells.
  There every rung from F − 1 up carries a window, and the client sums its texels from the samples (ARITH: at most
  726 node cells × 50 µs = 36 ms for that body).
- The lid on such a body reads the fine nodes under the footprint rounded outward (`coast_level`,
  `artifact.rs:511-520`), summed directly. A body with no pyramid also sums its nodes directly. There n is not a power
  of two, so the lid takes the recipe's exact reciprocal in place of the shift.
- A body with no water has no skin, and its far column stays today's.
- A body whose head has not landed draws today's rule. The skin starts when the mask and S3's words land.

**Example.** A pilot lands on the 5 km body. Its node is 714 cells, so the client builds no far page. Each chunk from
rung 2 up carries its own window, summed from its samples in a few milliseconds.

---

## 3. WHY IT IS SEAMLESS

### 3.1 The statements

Each statement names its condition and the line that measures it (§5).

**P0 — THE ONE QUANTITY NESTS.** Π(l + 1) is the sum of its four children, in integers, on both hosts (§2.1). No
condition. Lines: SKIN-NEST, SKIN-SQUARE-AGREE.

**P1 — AT EVERY FRONT FROM RUNG F UP, THE TWO RUNGS DRAW THE SAME COLOUR.**
- Both rungs read texels of ONE global grid: a window is aligned to it, and the far page is it.
- Each texel holds round(255 · c) of its own exact count, so a texel has one value in every window and in the far
  page.
- Both rungs compute one pixel's footprint in their own texels. The two texel sizes differ by the exact factor
  T_{L+1} / T_L (1 or 2). So the two rungs name the same absolute level, with the same fraction, the same probe count
  and the same probe places.
- Both hold that level and every texel that the probes reach: five levels and the margin M in a window. From rung 14
  both rungs read the SAME texture, the far page, so even a clamp at its border is the same for both.
- Inside the band only the finer rung shows: the coarser rung stands under it by its sink. The sink bounds the gap
  between the two surfaces (`crates/client/src/chunks.rs:58-77`), and a surface vertex's morph target stays inside
  that reach (`chunks.rs:1248`, `:1286`). Past the front only the coarser rung shows.
- The level floor and the hand-over clamp keep all of this true where a bound pulls a band in.

CONDITIONS: the view stands inside θ_rim; the GPU computes one filter for the two textures. Lines: SKIN-GRID,
SKIN-QUANT, SKIN-FRONT, SKIN-FRONT-BOUND (the bench); PIXEL-RUNG-INVARIANCE (the flight).

**P2 — NOTHING CAN BLINK.** No threshold stands on the colour's path over rung F. c is a linear filter of Π. It
changes only as the pixel's footprint moves or grows: at most 4 dp/p for a growth dp (the rim of a box over its area).
A swap adds nothing (P1). So no piece of water can show, hide and show again. Lines: SKIN-TRACK; LAKE-FLICKER.skin
reads 0 where the box fold reads 260 / 531.

**P3 — THE WATER PER FOOTPRINT IS THE SAME AT EVERY RUNG.** The mean of c over a coarser column is exactly the mean
over its four finer columns (P0). So AREA-NEST reads 0.000 % (the box fold: +1.8 / −6.9 % on the steep coast's lakes,
MEASURED). W16 item 2 holds exactly for the water: a bay that fills 30 % of a far pixel fills 30 % of that pixel in
every ring.

**P4 — THE HAND-OVER IS CONTINUOUS.** φ reads the pixel's footprint; the clamp g_F reads the band's own weight; a mix
of two continuous pictures by a continuous weight is continuous. At the F → F + 1 front φ = 1 on both rungs under
any bound. Lines: PHI-HANDOVER, PHI-FRONT, SKIN-FRONT-BOUND.

**P5 — AT A NEAR FRONT THE PICTURE DOES NOT CHANGE, AND A POINT CHANGES SIDE AT MOST ONCE IN A BAND.** With R2, at
t = 1 every finer water vertex stands on the coarser water, or q under the coarser ground, and every finer ground
vertex stands on the coarser mesh. At a point of a finer triangle the water's height over the ground is
(1 − t) · A + t · B, a line in t, so its sign changes at most once (design B's L6). CONDITION: this is exact at the
vertices. A finer triangle whose three radial hits fall on two parent triangles cuts a chord, so its crossing at
t = 1 can stand off the coarser crossing by part of a finer cell. Lines: WATER-MORPH-END and SWAP-EQUAL (Tier-A, R2),
NEAR-POP (the bench).

**P6 — THE FAR LAND HAS NO THRESHOLD.** From R_P the surface is the lid. n_wet, n_dry, Σ_wet W and the dry sums are
nested counts and sums; the terrace is continuous; the division is one shift. So the lid is a continuous function of
the place and of the level, and no word flips a column. The morph carries the level's change as it carries the
ground's today. Lines: FAR-NO-THRESHOLD, G-RUNG-HOLD.lid, POCKET-BASIN, TOP-STEP.

**P7 — A CUBE SEAM.** A face edge is 1 216 node squares, 77 824 samples = 2¹² × 19, so every level from 7 to 19 has
a texel edge on the face edge. A margin or border texel across the edge takes the partner face's texel through
`vd_seed::seam::across`, by ONE rule for every chunk and every rung. N2's field is seamless there (W-SEAM, MEASURED:
0 of 1 024 … 0 of 64 pairs differ at rungs 3–11). Line: SKIN-SEAM. The one coastal cube corner keeps Q12's phantom, the
same at every rung.

**P8 — THE RENDERER ADDS NO POP.** (a) The ground and its water read one entry of the band buffer. (b) The bands
move every frame at the slew's pace. (c) The water's box covers its morph and its sink. (d) A late chunk enters on its
stand-in's state. Lines: R1's Tier-A tests; G-BLINK and G-FRONT in the flight.

**Example.** At 15 664 km, the 14 → 15 front at the nadir, a pilot's pixel over the 5 882 m lake's piece covers
18 km. The rung-14 chunk and the rung-15 chunk sample the far page at the same place with the same footprint, so both
read one share, about 60 % (the piece and a rim of shore). The pixel does not change when the rung-14 fragment is
discarded. Under the box fold the same pixel turns from lake to land in that frame.

### 3.2 The counterexamples of the two reviews, and the answers

| # | raised by | the counterexample | the answer in D | the line |
|---|---|---|---|---|
| 1 | W16 on A and C; feasibility on A | At a near swap the finer water keeps its own side and level to the front; strips of 3–5 px on the belt's and the second pocket's lakes turn blue ↔ tan in one frame. | R2: the water's decision morphs, so the step becomes one slide over the band. | NEAR-POP; WATER-MORPH-END; SWAP-EQUAL; G-BLINK |
| 2 | W16 on A; feasibility on A | The 5 % rebind jump moves every band edge and front in one frame (whole(d) jumps by up to 0.27). | R1: one band buffer per realm, rewritten every frame; the water's materials read it too. | R1's Tier-A tests; FRAME-COST |
| 3 | W16 on A, B and C; feasibility on C | The far land still flips by count thresholds: G-RUNG-HOLD reads 437 m at 13 → 14, 3 298 m at 15 → 16, 6 312 m at 17 → 18 (MEASURED); that is a 5–10 % Lambert step at a 45° sun at 15 → 16 (ARITH), and the air reads kilometre steps (W18's 5 km chord dip showed through it). | The lid from rung 14: no side, fold, hold, D7 or pocket word there. | HOLD-RETURN and AIR-STEP (before); G-RUNG-HOLD.lid = 0 and FAR-NO-THRESHOLD = 0 (after) |
| 4 | W16 on C | C's own ARITH for the shade (1.7 % at 17 → 18) is low. | Corrected: 3 298 m over a 33–66 km column is 5–10 %; 437 m over an 8–16 km column is 2.7–5 % (ARITH). The lid removes both. | HOLD-RETURN prints the shade in % |
| 5 | W16 on A | Three page levels are too few for a low pilot. | Five levels in a window. Every far front reads the ONE far page on both sides, so a level that the page lacks is lacked alike by both rungs. | SKIN-FRONT at 80–89°; SKIN-RIM |
| 6 | W16 on A; feasibility on A and C | The halo or margin (A: 8 texels; C: 12) is shorter than the 16× sampler's reach. | The margin by rule, on each side after the alignment: 22 texels at θ_rim = 86.4°. | SKIN-FRONT with chunk-edge fragments; control: a 12-texel margin reads red |
| 7 | W16 on A | The hand-over is tied to one band; R-a is not free. | The hand-over reads the footprint (rung-free) with a clamp at rung 7's front; R-a is not taken. | PHI-HANDOVER; PHI-FRONT |
| 8 | W16 on A | A level floor read from a vertex varying does not reach its end at the front. | D reads t from the fragment's own distance, the one that the discard reads. | SKIN-FRONT-BOUND |
| 9 | W16 on C | A late chunk sharpens in one frame. | R1: the arrival ramp starts on the stand-in's shape and finest level. | G-BLINK on a fast descent |
| 10 | W16 on C | The hand-over drops water narrower than 128 m. | Stated as the resolution floor; a finer T₀ costs four times the samples per halving. | SUB-SAMPLE-WATER; Q5 |
| 11 | W16 on C | Per-chunk windows at the far rungs can differ at their edges. | A's one far page from rung 14: no chunk edge, built once per head. | SKIN-GRID; SKIN-FRONT r13-r14 |
| 12 | W16 on C | C's morph is exact at the vertices only. | SWAP-EQUAL (the drawn wet area at t = 1) stands beside WATER-MORPH-END. | SWAP-EQUAL |
| 13 | feasibility on C | Stage 1 keeps the box fold in the far geometry: the 5 882 m lake's piece is held under its water at rung 14, over it at 15, under it at 16. | Under the lid no far column folds. At rungs 8–13 the fold moves heights only (ARITH ≤ 160 m at rung 13) under one colour. | GEOM-FLIP; HOLD-RETURN |
| 14 | feasibility on C | A band buffer written as a changed storage asset gets a new GPU buffer every frame and re-prepares every material. | A raw buffer bound once through the derive's `buffer` storage attribute, written in place. | FRAME-COST; control: the asset path |
| 15 | feasibility on C | The hand-over changes the far look where the owner flies most (W18.7: *"all works amazing, so please don't break it"*). | The owner sees K0's panels at W18's seven stands before S1 (Q1). | the owner's look |
| 16 | feasibility on C | More sheet and more shading. | Measured, not argued; the lid removes the far sheet. | FRAME-COST; SKIN-MB |
| 17 | feasibility on C; A's risk 2 | The bench cannot judge the GPU's own filter. | The bench emulates each chunk's OWN window (margins, clamp, alignment, quantization), so it can go red; the GPU's own λ and weights meet the harness gate at 2 % luminance. | SKIN-FRONT (bench); PIXEL-RUNG-INVARIANCE, G-BLINK (flight) |
| 18 | W16 on A; feasibility on A | PIXEL-RUNG-INVARIANCE at 1/255 reads red on correct code (8-bit filter weights). | The gate is W16.5's 2 % of the local luminance; the gate prints \|Δc\| as a reading. | PIXEL-RUNG-INVARIANCE |
| 19 | W16 on B | B slides up to 1.41 px per band and can slide back; its crisp sub-pixel coasts crawl under 4× MSAA. | D draws sub-pixel water as a share: no crawl. The near slides stay (§1.4 item 2). | — |
| 20 | feasibility on B | B's conform is not defined against the surface-nets extractor; its far lines are the wet tie's lines; it forces a new SL6 ask. | D does not take B. DEFERRED registers B's exact near line for rungs 0–6, for the owner's word only (Q4). | — |
| 21 | feasibility on A | A's lift leaks into `vd-terrain`'s fallback and the parent cache. | A's lift is not taken. The lid lives in the one reader on both hosts, and `ParentMesh` builds on it through the same reader. | TOP-PIN; G-SEAM-MORPH |

---

## 4. DATA, RENDERER, RECIPE, COST

### 4.1 The data

| host | today | after S3 (ruled in B14 and B15) | the lid adds |
|---|---|---|---|
| the shard | the whole artifact: the rows, the pyramid, the mask (96 850 524 B) | derives the band, the lake levels and the mixed node squares' words at boot, and ships them | on a yes to Q3: the dry word at boot (one pass over the rows); the lid pyramid for its pins |
| the client holds | the head, six pyramid levels, the 2-bit mask (2.2 MB), `CoastCounts`, the tiles within rung 9's ring | + the band (0.631 MB), the lake levels (0.178 MB), the node squares' words (1.100 MB): 1.91 MB with the head | + the dry word (≈ 56 KB), on a yes to Q3 |
| the client derives | — | the records (4 096 keys a mixed cell, near the eye), the windows, the far page | the lid pyramid (T~, or exact with the dry word) |

**THE SKIN NEEDS NO NEW DATA** (no SL6 ask). It reads only what B14 items 1 and 3 and B15 item 3 approved, and what
the client derives from them.

**THE SL6 ASK: THE DRY WORD (only for the exact lid; Q3).**
- **What data:** for every level-1 node (16 km) whose four fine nodes are part wet and part dry, ONE word: the mean
  height of its dry fine nodes, in whole metres, 2 B; a delta list in node order, folded into the band's digest.
- **From which realm to which:** the planet's shard to every client, once, with the head, in the band's arm
  (`PROTO_MINOR` 35; no new arm).
- **Why the receiver cannot compute it:** the dry nodes' heights are the solve's output. The pyramid holds only the
  mean of ALL four fine nodes. The band holds a height only within 30 m of a water, so a deep wet node's height is not
  in it, and the dry nodes' mean cannot come out of the pyramid's mean.
- **The size:** ARITH ≈ 27 800 nodes × 2 B ≈ 56 KB. A level-1 node is part wet only where the node cell between its
  four fine nodes is mixed, and that cell is one node cell in four (111 089 mixed cells, MEASURED).
  MIXED-LEVEL-NODES measures it.
- **What doing without costs:** the interim T~ stands LOW where deep water drags a part-wet node's mean down. Example
  (design C's): a steep-coast node with 30 % sea at −3 km and 70 % land at +200 m reads −532 m against the true
  +140 m, 0.04–0.07 px at rung 14 (ARITH). It is continuous: no flicker and no step.
- `ARTIFACT_VERSION` stays 9: the shard derives the dry word from its stored rows at boot, as it derives the band.

**WHAT BECOMES UNNEEDED.** Slice 0b's crossing byte (B13 item 3's SL6 question for narrow water): the skin draws a
one-column inlet as its share, and a share cannot flicker. Under the lid, the highest-level word of each mixed node
square has no reader (−0.38 MB if the owner drops it later; B14 stands until then).

### 4.2 The renderer changes

**R1 — the pops, first** (`vd-client-render`; `vd-client`, Tier-A): §2.7 items 1–3.

**R2 — the water's decision morph** (`vd-client`, Tier-A): §2.4 item 2.

**K1 — the skin** (`vd-client`, Tier-A; `vd-client-render`, Tier-B):
1. **The window** of a chunk, built on the chunk's worker from the book's shared records and far counts
   (`vd-terrain`'s integer sums), and handed over in `ChunkGeometry` beside the mesh. A chunk whose footprint is all
   dry carries no window (c = 0); an all-wet chunk carries a constant (c = 1) in its tag. READY MEANS FINAL: a chunk
   arrives with its window, never before it.
2. **The skin atlas:** one R8 2D-array texture per realm and window size, one layer per mixed chunk, and a free list.
   The Image asset holds no data; the render world writes each layer in place (`RenderQueue::write_texture`). The
   engine uploads a changed Image with data whole (`bevy_render-0.18.1/src/texture/gpu_image.rs`, `prepare_asset`),
   and that stalls the frame. The layers count against the adapter's limit (2 048 on Metal), and the harvest's byte
   budget (V15) counts their bytes.
3. **The far page:** one 6-layer R8 array per body with water, built on the terrain workers' share (F6) when the mask
   and S3's words land. Level 0 is the node square. Its border holds partner-face texels by §2.3's margin rule at the
   top rung (ARITH: 22 rung-18 texels, 352 node squares). 352 is a multiple of 32, so the page's levels 0–5 (8 km to
   262 km) lie on the face's own global texels; the 13 → 14 front reads levels 0–3 only. Every level is an exact sum
   over the page. Its coarser levels are ordinary prefilters for a planet a few pixels wide, and every far front reads
   this one page on both sides, so no front can see a difference between two textures there.
4. **The vertex:** (u, v) as two 16-bit fractions of the window's (or the face page's) extent, from the vertex's
   integer lattice position (+4 B). The sheet's vertices carry their ground vertex's packed normal and morph normal
   (+8 B).
5. **The shader** (`ladder_fade.wgsl`): a varying carries (u, v) and the radial; the fragment computes c
   (`textureSampleGrad` with the level floor), φ_L and the colour of §2.3; the far-edge discard stays. The prepass does
   not change: the skin discards nothing.
6. **The probe material** takes the same skin, so the harness reads the drawn share.
7. **THE COVERAGE VIEW:** a dev switch that writes c in place of the lit colour, and a dev switch that scales every
   switch distance (a drawing knob for the harness, never a world knob).
8. **The lid's chunks** (rungs 14–18) build no sheet and no deep-water blocks.

### 4.3 The recipe changes (SL10, F7/F8, T9, HR5)

1. **The skin changes no recipe word.** Its counts come from N2's integer point side (S1's kernel), through ONE
   `vd-terrain` function per mixed node cell (the record) and one per uniform cell (the four-corner key), on both
   hosts. The shard ships the node squares' words; the client derives the keys under them. SKIN-SQUARE-AGREE is the
   measured no-drift gate between the two hosts for the skin's inputs. The skin itself is drawing, as the sheet is
   (W18.3): no pin reads it.
2. **The lid is a recipe change,** in the one reader (`shore_column`, the build design, §1.7), for every caller at
   once: `ColumnRead::column` (the chunk, its halo, the span) and `field_column` (the morph target of the seam and
   fallback vertices). The eye and the ruler read rung 0 and do not change. A NEW recipe entry, `column_lid_at`: the
   terrace of z̄_dry and one weighted sum, written out in the card's measured-safe shapes (the build design, §5).
3. **Versions.** The lid joins S4's one switch: `GENERATOR_VERSION` 22, once. `ARTIFACT_VERSION` stays 9.
   `PROTO_MINOR` 35 (the band's arm; the dry word rides in it on a yes).
4. **Pins.** G-N2-PIN's far keys at rungs 10 and 13 stand under R_P and keep their meaning. Its derived-pyramid leg
   (rungs 14–18) becomes TOP-PIN: the lid's digests over a stated level-3 block of coast, built from the rows and from
   the shipped words, equal. terrain-pin and mesh-pin (bodies with no sea: with no wet node the lid equals today's
   column, and a field with no mask reads no lid), the identity's top keys (no mask) and gpu-drift (the card builds
   no artifact chunk) do not move — MEASURED at S4, never argued.
5. **The card (F7/F8).** Its production path does not change. The lid's entry gets a probe leg in gpu-drift, as S1's
   shore probe does: 2¹⁶ inputs, 0 differing words, and a failing control with one input bit flipped.
6. **T9.** D takes no physical number from a random draw. T₀ is the rung-F sample (B15 Q6). The texel of half a
   cell and the five levels come from the band's geometry and the 16× sampler. φ's two ends are a rendering
   criterion — a texel covers two pixels, then one — from the ladder's own tier rule (one cell to one reference
   pixel), Proland's 1.1 px a texel and Ulrich's 2 px. θ_rim is where the sampler's own 16× maximum saturates. R8's
   step (1/255) stands under W16.5's 2 % gate. T_ramp keeps a stand-in's rise at a tenth of a pixel a frame. The lid
   is a mean; the dry word is a solve product.
7. **B15's two far cures.** The WET TIE and the LONE HOLD fire only at rungs ≥ 14 (the build design, §10). Under the
   lid they are inert. Without the lid they stay as ruled.
8. **HR5.** The new code in `vd-terrain` (the keys, the uniform key, the texel sums, the window cut and its margins,
   the quantization, the far page, the lid pyramid), `vd-client` (the window build, R2's morph, the arrival state,
   the band buffer's content) and, on a yes to Q3, `vd-wire` and `vd-connection-plane`: 100 % region and branch, each
   rule with a failing control. Generic helpers stay branchless shims.

**Example.** The shard boots the home planet, derives the band and the node squares' words from its rows, and ships
them with the head. A pilot's client builds a rung-15 chunk over the steep coast through the same `shore_column` that
the shard's TOP-PIN runs, and the two digests are equal.

### 4.4 The cost

| item | the skin | the lid | R1 and R2 |
|---|---|---|---|
| wire | 0 | 0 (T~); ≈ 56 KB (the dry word) | 0 |
| the shard | 0 past S3 | the dry word at boot: one pass over 8 871 936 rows (milliseconds, ARITH) | 0 |
| the client, once per head | the far page: 8.9 million node squares, the border's copies and the sums over them, ≈ 0.1–0.3 s on one core (ARITH; `CoastCounts::of` does 11.8 million adds) | the lid pyramid: 2.2 million level-1 nodes and their sums, ≈ 20–50 ms (ARITH) | — |
| the client, per chunk | a window: ≤ 49 K texel sums, ≈ 0.1–0.3 ms (ARITH); the records of its mixed cells once, 49–54 µs a cell (MEASURED, f64; the integer cost is the build design's S0 item 7) | one pyramid read per column | R2: one parent hit per water vertex |
| CPU memory | the records, ≈ 1 KiB of keys a mixed cell near the eye | ≈ 6–24 MB for the level sums (ARITH) | one word a parent vertex |
| GPU memory | the far page ≈ 28 MiB (R8 with its mips, the 352-square border) or ≈ 13.8 MiB (a 64-square border, with a seam rim at grazing views); windows ≤ 48 KiB each, ≈ 25–40 MB at a low coastal stand (ARITH; SKIN-MB measures) | none | the band buffer: bytes |
| vertex bytes | +4 B (u, v); +8 B a sheet vertex | the far sheet goes | — |
| GPU per frame | one anisotropic sample per terrain fragment from rung 6 (mixed chunks only); a second lighting pass on coast fragments (0 < c < 1) | fewer triangles (no far sheet) | one buffer write a frame |

UNMEASURED: every frame cost. FRAME-COST (`VD_CLIENT_UNPACED=1`, the coast leg) measures it against today.

---

## 5. THE BENCH PLAN — SLICE K0 (before S1; `vd-bins` only; no world change)

Ruling B16 item 2: find the best solution, measure it on the bench against the bar, and bring it to the owner before
S1. Every judged line below has the value it must read and a FAILING CONTROL. A check that cannot go red proves
nothing. Readings carry no verdict; each one names the question it feeds.

### 5.1 The modules (`crates/bins/examples/shoreline_bench/`)

| module | new or changed | what it holds |
|---|---|---|
| `pyramid.rs` | changed | a mixed cell keeps its 4 096 sample keys (2 bits a sample: dry, sea, lake) beside its quarter words, so levels 7–12 are exact |
| `skin.rs` | NEW | the skin pyramid (Π, Λ at levels 7–18 over a window); the window as the client will cut it (the rung, the chunk, T_L, five levels, the margin rule, the alignment to the coarsest level's global texels, the clamp to its edge, R8 quantization once); the far page (per face: the node squares and the partner faces' border through `vd_seed::seam::across`) |
| `sampler.rs` | NEW | THE EMULATED FILTER: bilinear, trilinear, anisotropic (N = min(⌈P_max / P_min⌉, 16) probes spaced along the major axis, λ = log2(P_max / N)), the clamp to the edge, the weights in f64 AND at 8 sub-texel bits, the level floor by a scale of the gradients. THE EMULATED VIEW: a stand's eye, a grid of reference pixels (1.1506e-3 rad), each ray's hit on the sphere at the sea's radius, the footprint's two axes from the incidence, the fragment's distance, and the rung and chunk that the ladder draws that point with (`vd_client::ladder_view`'s own bands) |
| `skin_lines.rs` | NEW | the lines of §5.2, groups 1–5 |
| `lid.rs` | NEW | the lid pyramid from the rows (exact), from the shipped words with the dry word, and as T~; the lines of group 6 |
| `flight.rs` | NEW | the emulated climbs: the high lakes 7 500 → 60 000 km; the gentle coast 60 → 1 400 km; a low pilot 10 km over the gentle coast who faces the horizon. One frame per 0.5 % of distance |
| `far_window.rs` | changed | W4 runs the skin's and the lid's lines at 13 → 14 … 17 → 18 |
| `seam.rs` | changed | W-SEAM runs SKIN-SEAM |
| `panels.rs` | changed | the coverage panels of §5.4 |
| `crates/bins/examples/n2_far_data.rs` | changed | MIXED-LEVEL-NODES and TOP-BYTES |

The hand-over lines run on W1 (rungs 3–8), which already spans the hand-over band. Each line carries the census's
usual name, `CENSUS <LINE>.<stand>.<window>.<rule>.rNN`.

### 5.2 The lines

**GROUP 1 — EXACTNESS (integers).**

| line | what it reads | must read | failing control |
|---|---|---|---|
| SKIN-NEST | every texel's (Π, Λ) minus the sums of its four children, levels 8–19 | 0 words | +1 on one child's Π → ≥ 1 |
| SKIN-UNIFORM | a uniform cell's four-corner key against the kernel's key at every one of its samples (every 64th uniform cell with a water, as CHECK-PYRAMID-UNIFORM) | 0 samples | one corner word flipped → > 0 |
| SKIN-SQUARE-AGREE (W4) | each node square's three words summed from records built over the band-held nodes, against the rows' | 0 squares | the records over S0's off-by-one band → > 0 |
| SKIN-QUANT | every R8 texel of every window and of the far page against round(255 · Π / n) of its own count | 0 texels | mips averaged from quantized children → > 0 |
| SKIN-GRID | every texel of every window (all five levels, the margins) and of the far page's levels 0–5 against the global pyramid's texel at the same absolute level and place | 0 texels | a window's origin shifted one texel → > 0 |
| SKIN-SEAM (W-SEAM) | the margin and border texels across the cube edge against the partner face's, through `vd_seed::seam::across` | 0 texels | the border read without the seam's rotation → > 0 |
| CORNER-TEXELS | the texels at the one coastal cube corner (Q12's phantom) | a reading | — |

**GROUP 2 — THE FRONT (the emulated filter).**

| line | what it reads | must read | failing control |
|---|---|---|---|
| SKIN-FRONT.r<L>-r<L+1>.{nadir, o30, o60, o80, o84, o86} | at every front 6 → 7 … 17 → 18: the pixels within half a pixel of the front, chunk-edge fragments included; max \|c_finer − c_coarser\| | 0 R8 steps with f64 weights; ≤ 1 step with 8-bit weights | (i) the coarser window's finest level at one cell, not half → > 0; (ii) a 12-texel margin at o84 and o86 → > 0; (iii) the box fold's water per column through the same filter → 255 steps on the 5 882 m lake's piece |
| SKIN-FRONT-BOUND | the same, with every band pulled in to 0.5 s and to 0.8 s | the same values | the level floor and the hand-over clamp off → > 0 |
| SKIN-RIM.{o88, o89, low10km} | the same past θ_rim, and the low pilot's horizon | a reading, with the margin that would make it 0 (Q8) | — |
| SKIN-UV-TRI | at t = 1, the finer mesh's interpolated (u, v) at a pixel against the coarser mesh's, on real chunk meshes (`vd_client::chunks::geometry_with`, today's rule) | a reading, in texels (ARITH ≤ 0.002 cells at rung 18) | — |

**GROUP 3 — B16's OWN LINES.**

| line | what it reads | must read | failing control |
|---|---|---|---|
| AREA-NEST.skin, and per lake body | the coarser footprint's wet samples minus its four children's, at every swap 7 → 8 … 17 → 18 | exactly 0 | the box fold on the same windows: +1.8 / −6.9 % (MEASURED) |
| LAKE-FLICKER.skin, FOLD-RETURN.skin, LAKE-APPEAR.skin | a body counts as drawn at a rung where its coverage mass is over nought | 0, 0, 0 — printed beside the box (260 / 531), the tent (28 / 49), no lone column (15 / 34) and today's rule (53 / 87) | the box fold's own numbers on the same windows |
| SKIN-TRACK.<climb> | every frame of each emulated climb: the largest \|c_drawn − c_ref\|, where c_ref is the same filter on ONE global pyramid with no rung and no window | 0 (f64); ≤ 1 R8 step (8-bit) | the box fold's water per column through the same filter → 255 steps |
| SKIN-LAKE-PIECE | the 5 882 m lake's measured piece: its c at every frame of the high-lakes climb, beside the box fold's 1 / 0 / 1 | a reading for the owner | — |
| SKIN-IDEAL | the drawn share against the exact box share of the pixel's footprint, per rung | a reading (max, p99), the same on both sides of every front | — |

**GROUP 4 — THE HAND-OVER.**

| line | what it reads | must read | failing control |
|---|---|---|---|
| PHI-HANDOVER.W1.{nadir, o80} | per pixel \|c_sheet − c_skin\| between 55.6 and 111.2 km; c_sheet = four samples a pixel of the drawn rung-6 and rung-7 columns' wet flags (the 4× MSAA pattern) | a reading (mean, p99; target p99 ≤ 0.5) for Q5 | — |
| PHI-FRONT | φ at the 7 → 8 front, on both rungs, on the unbounded ladder and with the band pulled to 0.5 s | exactly 1 | the clamp off, the band at 0.5 s → < 1 |
| SUB-SAMPLE-WATER | the bodies and the area narrower than 128 m per window (the resolution floor) | a reading for Q5 | — |

**GROUP 5 — THE NEAR SLIDE.**

| line | what it reads | must read | failing control |
|---|---|---|---|
| NEAR-SLIDE.<stand>.{sea, lake}.r<r>-r<r+1> | every stand's seas and lakes at 0 → 1 … 7 → 8: N2's SHORE-STEP, read as the slide that R2 carries | a reading for Q4 (MEASURED today: the gentle coast's sea ≤ 1.46 px; the belt's lakes ≤ 3.29 px; the second pocket's lakes ≤ 4.79 px) | — |
| NEAR-POP | the points of the near transects whose drawn side changes AT a front, under the emulated two-ended morph | 0 | the morph that keeps the finer water's own level to the front (today's) → > 0 |

**GROUP 6 — THE LAND.**

| line | what it reads | must read | failing control |
|---|---|---|---|
| HOLD-RETURN (W4) | with no lid: far columns whose hold flips at a swap and flips back at the next (the land shows, hides, shows), the largest jump in m and in px, and the Lambert step in % at a 45° sun | a reading for Q2 (ARITH expects 5–10 % at 15 → 16) | — |
| GEOM-FLIP (W2, W3) | rungs 8–13 (the fold stays): columns whose drawn height moves by more than q between two rungs and back at the next, with the Lambert step times the land's share | GATE: under 2 % (W16.5) | — |
| AIR-STEP (W4) | for each HOLD-RETURN jump at the grazing stands, the change of the air's density at the surface point, exp(−Δh / H), with H the body's own scale height | a reading for Q2 | — |
| TOP-NEST (W4) | each lid level's counts and sums against the sums of its four children | 0 | one child left out → > 0 |
| TOP-EXACT (W4) | the lid from the shipped words (the mask, the lake levels, the height pyramid, the dry word) against the lid from the rows | 0 level nodes, band 1 m (the pyramid's floor) | the dry word left out (T~) → > 0 |
| TOP-APPROX (W4) | \|T~ − T\|, max and p99, in m and in px at the swap distance, rungs 14–18 | a reading for Q3 | — |
| TOP-STEP (W4) | at 13 → 14 … 17 → 18, the finer surface against the coarser lid on the same radials, max in px at the swap distance | at most one coarser cell (T7) | — |
| G-RUNG-HOLD.lid (W4) | today's G-RUNG-HOLD, on the lid | 0 at every far swap (N2 today: 437 m at 13 → 14, 3 298 m at 15 → 16, 6 312 m at 17 → 18, RED) | N2 with no lid → RED |
| FAR-NO-THRESHOLD (W4) | far columns at rungs ≥ R_P that read a side word, a fold, D7, a hold or the pocket word | 0 | no lid → > 0 |
| POCKET-BASIN (W4) | level nodes whose fine nodes are all a pocket's floor, whose lid stands over their own terraced mean | 0 | a lid with the sea's side hold → > 0 |
| MIXED-LEVEL-NODES, TOP-BYTES | the ask's size per level (in `n2_far_data`) | a reading for Q3 (ARITH ≈ 27 800 nodes, ≈ 56 KB) | — |

**GROUP 7 — THE COST (readings).** SKIN-MB (the windows that the ladder wants at each stand, per rung, and their bytes at
θ_rim = 86.4° and at 89°; the far page's bytes at a 64-square and a 352-square border); SKIN-MS (the window build per
chunk, p50 and p99; the far page's build); RECORDS-NEEDED (the mixed node cells whose records rings 6–13 need, and
their time).

### 5.3 The harness gates after the build (HR6; release binaries; the window)

| gate | what it reads | must read | failing control |
|---|---|---|---|
| PIXEL-RUNG-INVARIANCE | the coverage view at W18's seven stands (6 374, 11 497, 16 952, 52 000, 68 000, 118 000 and 408 000 km) and on the high-lakes climb; each stand drawn again with every switch distance scaled by 0.95 and by 1.05, so another rung or band phase draws each pixel | pixels whose luminance differs by more than 2 % of the local mean (W16.5), outside a one-pixel dilation of the ground's silhouette: 0; \|Δc\| printed | today's sheet: thousands of pixels |
| G-BLINK | pixels whose luminance rises and falls by at least 2 % inside one band while the eye crosses it: at the nadir, oblique and at the limb; every swap 0 → 1 … 17 → 18; 4× MSAA; and a descent at 1 600 km/s with the bound binding | 0 | today's build → > 0 |
| G-FRONT | `ring_step`'s rows mode across each far front, on oblique stands | no sector step over 2 % | — |
| FRAME-COST | `VD_CLIENT_UNPACED=1`, the coast leg: the skin path against today's | a reading | — |
| G-EMPTY-GLOBE | a login over the moon: the time to the first far picture with coasts | within the owner's number (the build design's Q3) | — |

### 5.4 What the owner sees

The census, with every judged line green or red beside its failing control; then the panels, each captioned "no far
shadow" (B6.5):
1. W4's rungs 13–18 as coverage images at the stand's pixel scale; for each swap, the two rungs' images and their
   difference, which must be black; beside them, the box fold's three images of the 5 882 m lake's piece.
2. W18's seven stands as coverage globes.
3. The high-lakes climb as a strip of frames, with each swap distance marked, and SKIN-LAKE-PIECE's curve.
4. The hand-over band (rungs 6–8): the crisp line, the skin and their blend, side by side.
5. The lid against N2's far holds on the worst G-RUNG-HOLD profile.

**Example.** On the high lake's window the owner sees three panels of the box fold — the lake's edge blue, then tan,
then blue — beside three panels of the skin, where the same edge is one tint that softens from rung to rung, and a
black difference image under every swap.

---

## 6. THE SLICES, IN BUILD ORDER

| slice | crates | builds | gates | the world |
|---|---|---|---|---|
| **K0 — the skin bench** | `vd-bins` | §5 | every judged line of §5.2 with its control | unchanged |
| the owner | — | answers Q1–Q8 on K0's numbers and panels | — | — |
| **R1 — the renderer's pops** | `vd-client`, `vd-client-render` | the band buffer (the water reads it too), the water's culling box, the arrival ramp | Tier-A with failing controls; FRAME-COST not worse than today (control: the storage-asset path); a flight of today's world with G-BLINK and G-FRONT against a control run | unchanged |
| **R2 — the water's decision morph** | `vd-client` | §2.4 item 2 | WATER-MORPH-END 0 vertices and SWAP-EQUAL 0 area (Tier-A; control: today's target reads the fronts); a flight with G-BLINK at the near swaps | unchanged |
| **S1 — the integer kernel** | `vd-recipe` | as the build design; + `column_lid_at` and its probe | as designed; + the lid's probe leg | unchanged |
| **S2 — the host** | `vd-terrain` | as designed; + the sample keys in the record, the four-corner key, the texel sums, the window cut, the quantization, the far page, the lid pyramid | as designed; + SKIN-NEST, SKIN-UNIFORM, SKIN-QUANT, SKIN-GRID, SKIN-SEAM, TOP-NEST and TOP-EXACT as Tier-A tests | unchanged |
| **S3 — the band on the wire** | + `vd-wire`, `vd-connection-plane` | as designed; + the dry word on a yes to Q3; the far page and the lid pyramid built when the head's parts land | as designed; + SKIN-SQUARE-AGREE on the wire; G-EMPTY-GLOBE with the far page | unchanged |
| **K1 — the skin, OFF** | `vd-client`, `vd-client-render` | §4.2 K1, with the skin path off | a control picture equal to today's; Tier-A for the window build | unchanged |
| **S4 — THE SWITCH** | all | N2 at every rung, the skin ON, the lid ON from R_P, no far sheet | as designed; + the harness gates of §5.3; the flight; the owner's look | `GENERATOR_VERSION` 22 |
| **S5 — after the look** | — | as designed, minus slice 0b; + B's exact near line only on the owner's word | — | — |

**How this changes the build design's S1–S5** (`n2_build_design.md` §6):
- **S0** stands measured (items 1–6; item 7 after S1). **K0 comes before S1**, as B15 item 1 and B16 item 2 order.
- **S1:** its content stands. S1 builds the one naming rule for rungs 8–13 only: under the lid no far column folds
  (at rungs 8–13 the rule changes nothing, MEASURED: 0 of 16 748 lines). The lid's entry joins.
- **S2:** the sample count pyramid becomes the skin's input (drawing), not the recipe's input, from R_P. The record
  memo gains the 4 096 sample keys of a mixed cell (1 KiB). The skin's functions and the lid pyramid are new.
- **S3:** the arm stands; the dry word rides in it on a yes to Q3. The client builds the far page and the lid pyramid when the
  head's parts land, on the workers' share.
- **S4:** grows. The skin and the lid switch in the same change, with one `GENERATOR_VERSION` bump. READY MEANS FINAL
  now covers a chunk's window and the far page: a far chunk waits for them. The flight adds §5.3's gates.
- **S5:** slice 0b (the per-water fold with a crossing byte) retires, because narrow water is a share. B's exact near
  line waits here, on the owner's word.
- **New slices:** K0, R1, R2, K1. ARITH count: about ten slices in all, S1–S5 included.
- **Owed in the change that commits this file:** the status file's §8f gains B16's row
  (`docs/investigation/2026-09-24/terrain_believability_status.md`); `docs/design/DEFERRED.md` registers the rim
  (θ_rim), the resolution floor, the lake channel for 8o, and B's near line as an option.

**Example.** A pilot flies the build after R1 and R2: today's world, with no ring jump in fast flight and no strip of
water that pops at a near ring. After S4 she climbs over the high lakes, and no lake's edge changes at a ring.

---

## 7. THE QUESTIONS FOR THE OWNER

| # | the question | the recommended answer | its price |
|---|---|---|---|
| Q1 | **THE FAR LOOK.** Draw the far water as the share: past 111 km a coast is a soft edge about one pixel wide, and a lake narrower than a pixel is a pale tint that follows the pixel's footprint? | YES. A pixel cannot hold more; a threshold flickers (MEASURED under B15 item 1; B16 item 1 refuses all three folds); Cesium, Proland, Space Engine and Google Earth's imagery all draw sub-pixel water this way. | The far look moves away from W18's crisp sheet, which the owner called *"amazing ... please don't break it"* (W18.7). The owner sees K0's panels at W18's seven stands before S1, and the flight at S4. D refuses a contrast curve on the share: it is a threshold again. |
| Q2 | **THE LID.** From rung 14, the surface of a far column is the mean of what a pilot sees over it, and a far column has no side. This re-reads W15.1 items 1 and 4 and B15 Q4 at rungs ≥ R_P: "a far cell's side is its samples' wet share" becomes "a far cell's DRAWN water is its samples' wet share, never thresholded, and its surface is the lid". | YES, after K0 reads HOLD-RETURN and AIR-STEP (expected red: 3 298 m at 15 → 16 is a 5–10 % Lambert step, ARITH). | It joins S4's `GENERATOR_VERSION` 22. TOP-PIN replaces the derived-pyramid leg. B15 item 2's far cures become inert. The far coast's land shades on the lid's slope: a small look change, which TOP-STEP and the panels show. |
| Q3 | **SL6: THE LID'S DRY WORD.** Ship, in the band's arm, each part-wet level-1 node's dry-node mean height: ≈ 27 800 nodes × 2 B ≈ 56 KB (ARITH; MIXED-LEVEL-NODES measures)? | NO for now: take T~, which needs no new data. Ask again only if TOP-APPROX reads over a tenth of a pixel or AIR-STEP sees the bias. | NO: T~ stands low where deep water drags a part-wet node's mean down (−532 m against +140 m on a steep-coast node, 0.04–0.07 px, ARITH). YES: 56 KB with the head, a digest word, the gateway's cache. |
| Q4 | **THE NEAR SLIDES.** Accept one slow slide per band at rungs 0–7 (R2), at most N2's measured step, and bench B's exact nested line only if the owner sees the slides in R2's flight? | YES. | The slides stay: the gentle coast's sea ≤ 1.46 px, the belt's lakes ≤ 3.29 px, the second pocket's lakes ≤ 4.79 px, each over a fifth of the distance. B would bound them at 1.41 px for about ten slices, and its conform is not yet defined against the ground's surface-nets mesh. |
| Q5 | **THE HAND-OVER AND THE RESOLUTION FLOOR.** Hand the crisp coast over to the share by the pixel's footprint alone, across rung 7's ring (55.6–111.2 km), with T₀ = 128 m (the rung-F sample)? | YES. | Water narrower than 128 m (a pond, a strait) leaves the picture in that band; a pond that one sample hits grows to a 128 m texel; a finer T₀ costs four times the samples per halving. A's alternative (inside rung 7's fade-out band, a fifth of the distance) is sharper and tied to one band. |
| Q6 | **WHAT LEAVES.** Retire slice 0b (the crossing byte for narrow water, B13 item 3's SL6 question), and build the one naming rule for rungs 8–13 only? | YES. | None: it removes planned work. B14's three words stay as ruled; the highest-level word has no reader under the lid (−0.38 MB if the owner drops it later). |
| Q7 | **THE ARRIVAL RAMP'S TIME.** A late chunk enters on its stand-in's state and relaxes over 0.5 s? | YES: the stand-in's sink (three pixels or more) then rises a tenth of a pixel a frame at 60 Hz. | A late chunk shows its stand-in's shape half a second longer. |
| Q8 | **THE RIM.** Size the window margins for θ_rim = 86.4° (where the 16× sampler saturates: windows of at most 192 × 192 texels)? | YES now; SKIN-RIM and SKIN-MB print 88° and 89°; wider margins only if the flight shows the rim. | Past 86.4° — the skyline of a pilot lower than about 50 km — a faint line can show along a chunk edge at a front. Margins for 89° take windows of about 288 × 288 texels: 2.25 times the memory. |

**Example.** If the owner answers Q1 NO, no drawing of sub-pixel water can meet B16: the far water must then be a
threshold, and the fold's flicker (B16 item 1) returns in some form. If the owner answers Q2 NO, the water's colour
still meets B16, and the far land keeps G-RUNG-HOLD's jumps under one colour.

---

## 8. THE THREE DESIGNS AND THE TWO REVIEWS

| design | the W16 review | the feasibility review | the main counterexample, W16 lens | the main counterexample, feasibility lens | what D takes from it |
|---|---|---|---|---|---|
| **A — the water as a share** (coverage from rung 8, a lifted ground, pages, recipe change R-a) | 6 | 5.5 | The near swaps still pop: the finer water keeps its own side and level to the front, so strips of 3–5 px on the belt's and the second pocket's lakes turn blue ↔ tan in one frame. | The halo (8 texels) is shorter than the 16× sampler's reach from about 80°: the tint steps along a chunk edge at a front. | the one far page per body; the level floor; the swap-pixel and pixel-invariance gates |
| **B — the nested curve** (exact nested crossings, a conformed mesh per shore cell) | 5 | 3 | The owner's own example fails: the 5 882 m lake's piece slides by up to one coarser cell per band and can slide back; crisp sub-pixel coasts crawl under 4× MSAA. | The conform is not defined against the surface-nets extractor: a chord crosses a quad that is not a shore cell (a crack or a depth fight). | nothing now; DEFERRED registers the exact near line for rungs 0–6 (Q4) |
| **C — the rung-free skin** (coverage by the footprint from rung 6, the water's decision morph, the pops first, a top surface as Stage 2) | 7.5 | 6.5 | The near slides are N2's own steps (up to 3.29 / 4.79 px on lakes), not 1.40 px; Stage 1 keeps the land's count thresholds (3 298 m at 15 → 16). | Stage 1 keeps the box fold in the far geometry; a band buffer written as a changed storage asset re-prepares every material every frame. | the core: the skin, the hand-over by the footprint, the water's decision morph, the pops first, and its Stage 2 top surface as the lid from rung 14 |
| **D — this design** | — | — | The near slides stay (≤ 4.79 px, one per band); sub-pixel water is a tint. | The largest risks: the owner's look at the soft far coast (Q1), and the GPU's own filter at the fronts (PIXEL-RUNG-INVARIANCE). | — |

**The two verdicts, in one line each.** The W16 review: C best meets B16, as a combination with A's far page and
gates, C's Stage 2 or another threshold-free far hold, five page levels, margins sized to the 16× sampler, and an
arrival ramp; B's exact near line only if the near slides show. The feasibility review: C's Stage 1 is the best core
and B the worst; take A's far page and level floor, fix the margins and the band buffer's Bevy path first, and choose
the far land's cure on the bench's numbers.
