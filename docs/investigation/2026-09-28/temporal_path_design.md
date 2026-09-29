# The temporal path — the design for the speed step

**Status: DESIGN. Nothing is built. The design waits for the owner's ruling.** No code starts before the ruling (the
owner, 2026-09-28: *"Design the temporal path"*).

*Written 2026-09-29 on branch `game` at `ba2496b`, with the uncommitted speed-step work in the tree (the GPU's own clock,
the sealed hollows). The inputs: the code, the rulings, the GPU clock's measurements of 2026-09-28, three research
reports (the engine, the renderer, the industry) and two competing drafts (draft A: the picture and the seams first;
draft B: the headroom first). This document builds nothing, runs nothing and changes no other file. §15 names every point
where the two drafts disagree, which side this design takes, and why.*

*Revised on 2026-09-29 after a review of 18 findings. The writer checked each finding against the code or the numbers.
All 18 held. §19 gives one line for each: the finding, the verdict, and what changed. The largest changes: the budget
model now prices every full-screen pass by its memory traffic, so the frame at half size reads about 1.2 ms dearer; the
first picture slice (T2) now builds the smaller drawn picture, so the owner judges the edge at the size that ships; the
hole census and the motion gate get instruments that can fail.*

**Tags.**
- **MEASURED**: the GPU's own clock of 2026-09-28 at the owner's stand (with the owner's word), or an earlier run that is
  named with its source.
- **DERIVED**: arithmetic on MEASURED numbers only. The document shows the arithmetic.
- **VERIFIED**: read in the code at the cited lines. "(re-read)" marks the lines that decide this design; the writer read
  them again for this document. "(research)" marks lines that a research report read and this document did not read again.
- **REPORTED (H / M / L)**: an outside source as the industry research read it. H is a primary source (a vendor document,
  a paper, source code). M is a secondary source. L is weak evidence.
- **ESTIMATE**: arithmetic on a stated assumption. It stays UNMEASURED until a slice measures it.
- **UNMEASURED / UNVERIFIED**: nobody measured it / nobody read it.

**Binding.**
- Ruling B22: item 1.2 (speed, no world change, no stall) and item 3 (60 frames a second at FULL SCREEN).
- The owner, 2026-09-28: *"Clock, then the looks"*; *"Design the temporal path"*.
- V16 and V18: no quality drop without the owner's word; a lossy packing may move a content pixel by at most one level of
  255 in any channel.
- SL8 (a seam is a defect), SL1, SL6, SL9, SL10, no client-side prediction, HR5, HR6.
- F6: the game targets an average machine; the Mac is the measuring instrument.
- The Bevy upgrade gate: no Bevy 0.19 before S11 and S12 are done.
- The owner's standing rules: a new library only on the owner's word; test exactly the production path; open a game window only
  with the owner's word at that moment; measure one job at a time; never assume, measure.

---

## 0. For the owner

You asked: *"one-pass MSAA - how good is it for the AAA game? … everything will be way more complex, with constructions,
clouds, weather, hundreds of players. Is one-pass MSAA the better solution for a star-citizen like mmo?"* Then you chose:
*"Design the temporal path."* This section answers in plain words. The sections after it give the proof.

**What the design does.**
1. Each frame draws the picture smaller: half the width and half the height, so a quarter of the pixels (the proposal;
   D8 decides the size after the first measurements). Each frame shifts the picture by a different small part of a pixel.
2. One new pass builds the full-screen picture from this frame and the frames before it. It follows every surface to the
   place where that surface stood in the last frame.
3. The stars and the sun draw after that pass, at full size and sharp, with today's sprites.
4. The four samples a pixel (multisampling, "MSAA") go away, but only after your look at the new edge.

**Why.**
- On the GPU's own clock, your stand on foot costs 32.25 ms a frame. 60 frames a second needs 16.7 ms.
- Without any multisampling the same frame still costs 21.75 ms (MEASURED). So no form of multisampling, one pass or
  many, reaches 60 frames a second at full screen while the GPU draws every one of the 7.5 million pixels.
- Multisampling costs 10.5 ms and the marched air 9.7 ms (they share 3.1 ms). The GPU pays both for each of the 7.5 million
  pixels. A smaller drawn picture divides every such cost at once. Only a pass that uses the earlier frames can then give
  back a sharp full-size picture.
- Some effects that the game still needs do not work with multisampling in this engine: contact shadows, reflections and
  layered glass. The deferred light, a cheaper way to light many lamps, also refuses it; the lamps themselves work with
  multisampling. The shipped games that we studied also build clouds and weather on the earlier frames.
- Every planet and space game that we studied ships a temporal method, except Elite Dangerous. Star Citizen forces one.

**The numbers** (ESTIMATES until the first slice measures them; the budget section below shows the arithmetic). Each
number is a MEDIAN at a still stand. The step's gate is the 99th percentile on the moving legs, and nobody measured that
yet.

| What each frame draws | GPU time a frame | 60 frames a second? |
|---|---|---|
| Today: full size, four samples a pixel | 32.25 ms (MEASURED) | no |
| The new pass, full size | 24–28 ms | no |
| The new pass, two thirds of the width and height | 17–21 ms; about 17.8 ms with the air's cheaper march | no |
| **The new pass, half of the width and height** | **15–19 ms, about 16.8 ms; about 16.0 ms with the air's cheaper march** | **likely at the middle with the cheaper march, NOT assured: the top of the range misses 16.7 ms, two costs are unmeasured, and the 99th percentile on the moving legs is unmeasured** |
| The new pass, a third of the width and height (the floor) | 13–17 ms, about 15.1 ms; about 14.7 ms with the air's cheaper march | yes at the middle, with about 2 ms to spare; the softest picture that the floor allows |

About 7 ms of fixed work (the ground's triangles and the fixed passes) and 4–8 ms of full-size work (the new pass, the pass
for the stars and the sun, the sharpen, bloom, the colour grade) stay at every drawn size. So the goal of 11 ms, which keeps a third of the
frame for the game still to come, needs a geometry step and fewer full-size passes after this one.

**What you will see change.** Each change is your decision, shown in pictures beside today's picture: the horizon's edge;
the softness when you turn or fly fast; the sharpness at half size; the air's cheaper march; the edge of a shadow. The
stars and the sun keep today's look.

**The decisions asked now** (the decisions section below has all sixteen, each with its price and its picture test):
1. **D1.** Build the temporal path. Multisampling stays until your look at the new edge (D5).
2. **D2.** The order: measure first (the first slice changes no pixel), then the motion, then the first picture. The first
   picture already comes at the smaller drawn sizes, so your edge look sees the size that ships.
3. **D3.** Our own new pass, which runs on every GPU. Apple's own scaler comes only later as a challenger, and only if you
   allow its new library.
4. **D4.** The GPU draws the ground in ONE pass that writes its surface, its depth and its motion, after your edge look.
   It needs three small changes in the engine copy that we already keep (the air's own changes come apart: §4.8 lists
   every change by slice).

**Example.** A pilot on foot on the home planet turns their head. Today the GPU marches the air for every pixel of the view
and colours each of four samples. With the design it marches one pixel in four, the new pass sharpens the horizon from
the last frames, and the stars stay exactly where they are.

---

## 1. The words used in this document

| Word | Meaning | Example in the game |
|---|---|---|
| the display picture | The window's picture: 3 456 × 2 168 = 7 492 608 pixels. | The owner's full-screen window. |
| the render picture | The smaller picture that each frame draws, shifted by the jitter. | 1 728 × 1 084 at half the width and height. |
| the render scale `s` | The render picture's rows ÷ the display picture's rows. The render picture has `s²` of the pixels. | `s` = 0.5 draws a quarter of the pixels. |
| the jitter | A shift of the whole render picture by a part of a pixel. Each frame takes the next shift of a fixed sequence. | Eight frames see eight points inside one horizon pixel. |
| the motion | For each render pixel: where its surface stood on the screen in the last frame. | A hill 1 km beside a hull at 240 m/s moves about 10 display pixels a frame. |
| the history | The reconstruction's own display picture of the last frame. | The belt as it stood 16.7 ms ago. |
| the reconstruction | The new pass. It builds the display picture from the render picture and the history. The industry calls it a temporal upscaler (TAAU). | 1.9 million new pixels become 7.5 million. |
| a reset | The reconstruction throws the history away. | The first frame after a window resize. |
| a disocclusion | A surface that was hidden in the last frame, so it has no history. | The ground that a departing hull uncovers. |
| the reactive mask | A word per pixel: "trust this frame more than the history here". | The translucent hull box over the moving ground. |
| the sky share | For each pixel, the share through which the open sky (the stars, the sun) shows: 1 − the render picture's alpha. | Half a pixel of ridge and half of sky: 0.5. |
| the realm's step | The rigid map from this frame's render space to the last frame's render space, for one realm. | All chunks of the home planet move by one step when the eye walks. |
| the points pass | The pass that draws the stars and the sun disc after the reconstruction, at the display size. | A neighbour star 0.6 display pixels wide. |
| the one geometry pass | The picture's triangles go through the GPU once a frame and write the depth, the surface words and the motion together (a "deferred" pass). | The 81 M ground triangles are read once. |
| the surface words | The per-pixel record of a surface's colour, roughness and normal that one light pass lights later (the "g-buffer"). | The belt's rock, lit once per render pixel. |
| the floor law | One render pixel never covers more than one pixel of the reference view (45° over 720 rows). | On the owner's display, `s` ≥ 0.33. |
| the picture set | The owner's stands and flights for every look (§14): P1–P10. | P5: the hull at 528 m/s. |

---

## 2. The baseline: the measurements and the facts

### 2.1 The GPU's own clock at the stand

The method: two empty stamp passes of the clock's own bracket the whole render graph; the second one loads and stores
the camera's output picture, so the GPU cannot run it before the last pass that draws (`gpu_clock.rs:11-21`, re-read).
The window's frame time moves in whole 8.33 ms steps of the 120 Hz display, so it cannot resolve these costs
(`DEFERRED.md:10702-10707`, re-read). Paced, the stand drew 26.7 frames a second at full screen
(`DEFERRED.md:10656-10658`).

On foot at the owner's stand, full screen (3 456 × 2 168), 14 261 chunks drawn, 81 M triangles resident after the cave
sealing, 12 samples a mode, the median GPU time of the frame (`DEFERRED.md:10727-10741`, re-read):

| Id | Mode | GPU ms | Tag |
|---|---|---|---|
| A1 | All looks on: the raymarched air, MSAA 4×, bloom, 4-cascade shadows | 32.25 | MEASURED |
| A2 | A1 without MSAA | 21.75 | MEASURED |
| A3 | A2 with the lookup air in place of the raymarch | 15.12 | MEASURED |
| A4 | All four looks off: the lookup air, no MSAA, no bloom, no shadows (the definition: `DEFERRED.md:10666-10667`) | 13.27 | MEASURED |
| A5 | A4 at 2 560 × 1 440 (49 % of the pixels) | 10.16 | MEASURED |
| A6 | A4 with half / a quarter of the triangles | 11.24 / 9.45 | MEASURED |
| A7 | A1 with the lookup air / without bloom / without shadows / without the shadow read | 22.54 / 27.62 / 28.10 / 28.34 | MEASURED |
| A8 | A1 with four times the draws (each part holds only its own vertices) | 37.12 | MEASURED |

### 2.2 The pieces

| Piece | ms | Arithmetic | Tag |
|---|---|---|---|
| MSAA 4× (with the raymarched air) | 10.5 | A1 − A2 | DERIVED |
| — MSAA on the ground (with the lookup air) | 7.4 | A7 (lookup) − A3 | DERIVED |
| — the air's extra work for each sample | 3.1 | 10.5 − 7.4 | DERIVED |
| The raymarched air over the lookup air, once per pixel | 6.6 | A2 − A3 | DERIVED |
| The raymarched air over the lookup air, under MSAA | 9.7 | A1 − A7 (lookup) | DERIVED |
| Bloom and shadows together, without MSAA | 1.85 | A3 − A4 | DERIVED |
| Bloom; shadows (their marginal costs under MSAA) | 4.6; 4.2 | A1 − A7 | DERIVED |
| The shadow read; the shadow casting (under MSAA) | 3.9; ~0.3 | A1 − A7 (no read); 4.2 − 3.9 | DERIVED |
| The draws, with the looks on | ~1.6 | (A8 − A1) ÷ 3 | DERIVED. With the looks off, three times the draws cost nothing measurable (`DEFERRED.md:10710-10714`). |
| The looks-off pixel work (the ground's light and depth, the lookup air, the post passes) | ~6.1 | (A4 − A5) ÷ 0.51 | DERIVED, if the work is linear in the pixels |
| The ground's triangles | ~4.1–5.1 | (A4 − A6 half) ÷ 0.5; (A4 − A6 quarter) ÷ 0.75 | DERIVED, MIXED: the thinning switch opens holes in the picture (F34), so the two runs also change the pixel work (the sky or a far rung shows through each hole). The two cuts disagree by 1 ms. M-3 prices a second geometry pass directly. |
| The fixed passes (the air's tables, the light probe, the culling, the clock's own end pass) | ~2.1–3.1 | A4 − 6.1 − the triangles | DERIVED (it inherits the mixed split of the row above) |

### 2.3 What the numbers say

1. **The full-screen frame is the pixels'.** About 25 ms of the 32.25 ms grows with the pixels or the samples. About 7 ms
   does not: the triangles and the fixed passes (the split between these two is mixed, see the row above).
2. **The looks overlap.** Their marginal costs sum to 29 ms against 19 ms together (A1 − A4). So a budget uses differences
   from measurements in the TARGET configuration (no MSAA), never a subtraction from today's configuration. Example:
   bloom reads 4.6 ms with MSAA on, but bloom and the shadows together read 1.85 ms without MSAA.
3. **MSAA's cost is mostly memory traffic.** Bevy stores each multisampled texture and loads it again in the next pass,
   and the vendored air pass loads every sample again (facts F15, F21). At this size the 4× colour is 239.8 MB and the 4×
   depth 119.9 MB (ESTIMATE: 7.49 M pixels × 8 B × 4; × 4 B × 4). wgpu 27 itself can keep the samples in the tile (F26):
   the ENGINE's pass structure stores them, not the wgpu version.
4. **60 frames a second needs 15.6 ms less.** MSAA (10.5 ms) and the air (9.7 ms) are the two largest pieces. Fewer
   triangles remove neither of them.

### 2.4 What the code and the engine say (VERIFIED)

| Id | Fact | Where | Tag |
|---|---|---|---|
| F1 | Bevy 0.18's TAA does not upscale. It sets only the camera's viewport, and its history has the target's full size. | `bevy_anti_alias-0.18.1/src/taa/mod.rs:224-226, 410-450` | VERIFIED (re-read; 410-450 research) |
| F2 | Bevy's TAA restarts its confidence when a pixel moves more than 0.01 px, and blends 10 % of the new frame, down to 1.5 % when still. The shader fixes these constants. | `taa/taa.wgsl:9-10, 164-178` | VERIFIED (re-read) |
| F3 | Bevy's jitter is 8 Halton phases chosen by the frame count; the first phase is (0, 0). | `taa/mod.rs:384-401` | VERIFIED (re-read) |
| F4 | The engine's render-scale seam is a render-world component that shrinks only the drawn rectangle. Its own text: "Does not affect post processing". | `bevy_camera-0.18.1/src/camera.rs:83-109` | VERIFIED (re-read) |
| F5 | The ground's prepass writes no motion. Its previous place is last frame's placement plus THIS frame's displacement. Under the motion switch its vertex stage outputs that previous place as four floats for each vertex. | `crates/client-render/src/ladder_fade_prepass.wgsl:166-172, 182-204` | VERIFIED (re-read) |
| F6 | The arrival ramp reads its age as a WRAPPING unsigned difference of the clock and the stamp. A stamp newer than the clock reads as an age of about 2³² ticks, so the weight reads 1. | `crates/client/src/arrival.rs:63-68`; `ladder_fade.wgsl:83-87`; `ladder_fade_prepass.wgsl:54-58`; `probe.wgsl:62-66` | VERIFIED (re-read) |
| F7 | On the GPU path the engine gives a mesh that is new this frame its current transform as its previous one. | `vendor/bevy_pbr/src/render/mesh.rs:1215-1250`; `render/mesh_preprocess.wgsl:215-233`; `prepass/mod.rs:215-236` | VERIFIED (re-read) |
| F8 | The main pass skips a deferred material in its opaque branch but NOT in its alpha-mask branch. The ground is alpha-masked. | `vendor/bevy_pbr/src/material.rs:1238-1245, 1275-1300`; `crates/client-render/src/terrain.rs:2281-2284` | VERIFIED (re-read) |
| F9 | The deferred light pass sets no viewport. The air pass sets the render rectangle. | `vendor/bevy_pbr/src/deferred/mod.rs:192-223`; `atmosphere/node.rs:266-270` | VERIFIED (re-read) |
| F10 | The deferred output writes the motion in the same pass as the surface words. The words pack the colour and the roughness in 8 bits each and the normal in 24 bits. | `vendor/bevy_pbr/src/deferred/pbr_deferred_functions.wgsl:25-80, 131-155` | VERIFIED (re-read) |
| F11 | The ground's material is the standard lighting on one colour (sRGB 0.55, 0.50, 0.42), roughness 0.95, no texture. | `terrain.rs:2275-2286`; `ladder_fade.wgsl:191-202` | VERIFIED (re-read) |
| F12 | An extended material can state its own deferred stages; the ground's prepass vertex stage already handles the deferred normals. | `vendor/bevy_pbr/src/extended_material.rs:78-89, 338, 372-385`; `ladder_fade_prepass.wgsl:156-165` | VERIFIED (re-read) |
| F13 | The deferred light runs between the start of the main pass and the opaque pass; the air pass runs after the opaque pass. The deferred light writes alpha 1. | `deferred/mod.rs:119-127`; `atmosphere/mod.rs:185-205`; `deferred/deferred_lighting.wgsl:55-99` | VERIFIED (re-read) |
| F14 | The air pass adds the sun disc on sky pixels and keeps the picture's alpha (its alpha blend is Zero, One). A masked or opaque surface writes alpha 1. | `atmosphere/render_sky.wgsl` (the depth-0 branch); `atmosphere/resources.rs:404-417`; `render/pbr_functions.wgsl:97-122` | VERIFIED (re-read) |
| F15 | The per-sample air (the second patch) and its two Rgba32Float view-size textures (the fourth patch) exist only under MSAA. | `vendor/bevy_pbr/PATCH.md`; `atmosphere/resources.rs:537-560` | VERIFIED (re-read) |
| F16 | The clear colour is opaque black: alpha 1. | `crates/client-render/src/lib.rs:132, 1275-1279` | VERIFIED (re-read) |
| F17 | The stars draw in the opaque phase before the air pass, One + One, with no depth write. A star is never smaller than 1 px: "A quad under a pixel misses the pixel centre and BLINKS". | `lib.rs:819-870`; `star_sky.wgsl:146-157`; `crates/client/src/realm_scene.rs:686-690` | VERIFIED (re-read) |
| F18 | Every drawn chunk stands at `centre + facing · origin` from its realm's row each frame: one row per realm. | `terrain.rs:4520-4571` | VERIFIED (re-read) |
| F19 | The renderer writes every band table every frame. The table's header is `[frame_ticks, ramp_ticks, 0, 0]`: two words are free. | `terrain.rs:2193-2196, 3142-3169, 4430-4441` | VERIFIED (re-read) |
| F20 | The reference view is 45° over 720 rows. The ladder switches a rung where one cell is one such pixel. | `crates/core/src/geometry.rs:1156-1173`; `crates/client/src/ladder_view.rs:199-209` | VERIFIED (re-read) |
| F21 | wgpu 27 has no transient (memoryless) texture, and Bevy stores every colour attachment (`StoreOp::Store`) and loads it again in the next pass. A memoryless texture only removes an allocation; F26 is the in-tile resolve itself. | `wgpu-types-27.0.1/src/lib.rs:5598-5624`; `bevy_render-0.18.1/src/texture/texture_attachment.rs:40-80` | VERIFIED (re-read for the store op) |
| F22 | The deferred light, SSAO, SSR, OIT and meshlets refuse MSAA in Bevy 0.18. | `bevy_core_pipeline-0.18.1/src/core_3d/mod.rs:977-988`; `vendor/bevy_pbr/src/ssao/mod.rs:505-508`; `ssr/mod.rs:444-446`; `meshlet/mod.rs:288-290`; `bevy_core_pipeline-0.18.1/src/oit/mod.rs:164-167` | VERIFIED (re-read) |
| F23 | A native command that the wgpu hal records runs before every wgpu command that the same encoder recorded earlier. | `wgpu-core-27.0.3/src/as_hal.rs:336-360`; `command/mod.rs:1292-1345` | VERIFIED (research) |
| F24 | The dot floor is a RADIUS of 3 pixels of the camera's viewport. The pixel laws read the main-world camera's viewport. | `crates/client-harness/src/camera.rs:270-291`; `lib.rs:2066-2077, 2196-2216` | VERIFIED (re-read) |
| F25 | The capture draws 1 284 × 720; its picture camera takes the default MSAA (4×); its probe camera is MSAA off and copies the picture camera's transform and projection every frame. | `lib.rs:162-168, 1244-1252, 3244-3288` | VERIFIED (re-read) |
| F26 | wgpu 27 resolves multisamples in the tile: a colour attachment with a resolve target and `StoreOp::Discard` becomes Metal's "multisample resolve" store action, so the GPU never writes the samples; a discarded depth becomes "don't care". | `wgpu-core-27.0.3/src/command/render.rs:70-75`; `wgpu-hal-27.0.4/src/metal/conv.rs:298-305`; `metal/command.rs:546-561, 576-580` | VERIFIED (re-read) |
| F27 | Under the render-scale override the view uniform's `viewport` keeps the DISPLAY size; only `main_pass_viewport` takes the render size ("In shaders, use `View::main_pass_viewport`"). No shader of the vendored crate reads `main_pass_viewport`. The deferred light rebuilds each pixel's world place through `frag_coord_to_ndc`, which divides by `view.viewport`; the cluster lookup (forward and deferred) and SSAO read `view.viewport` too. The air pass is correct: it draws into the render rectangle and takes its ray from the full-screen `uv` and its depth from the pixel's own place. | `bevy_render-0.18.1/src/view/mod.rs:945-949, 989`; `bevy_camera-0.18.1/src/camera.rs:99-106`; `vendor/bevy_pbr/src/deferred/pbr_deferred_functions.wgsl:116`; `render/view_transformations.wgsl:225-232`; `render/clustered_forward.wgsl:52`; `render/light.rs:1233-1234`; `ssao/ssao.wgsl:50, 144`; `atmosphere/render_sky.wgsl` (the ray from `in.uv`, the depth at `in.position`); `atmosphere/node.rs:266-270` | VERIFIED (re-read) |
| F28 | Many lamps, decals and alpha-to-coverage work WITH MSAA in this engine: the forward light walks each pixel's cluster of point and spot lights, and the forward pipeline runs with MSAA; the clustered decals "can be used with forward or deferred rendering and don't require a prepass"; the forward decals refuse MSAA only on the web; alpha-to-coverage runs only under MSAA, and without MSAA the same material falls back to a hard cut. Only a camera with the deferred prepass forces MSAA off. | `vendor/bevy_pbr/src/render/pbr_functions.wgsl:400-455`; `decal/clustered.rs:1-10`; `decal/forward.rs:61`; `material.rs:626-629`; `bevy_core_pipeline-0.18.1/src/core_3d/mod.rs:977-988` | VERIFIED (re-read) |
| F29 | Bevy's sharpen is a full-screen pass from the source picture to the destination, in the HDR format on an HDR view (8 bytes a pixel), after the tone map. Every main-pass target has the DISPLAY size, and each pass stores it; the override sets only the viewport; the Vulkan render area is the whole target, and the Metal pass sets no smaller target size. The deferred targets are Rgba32Uint, R8Uint and Depth16Unorm, the motion Rg16Float: 23 bytes a pixel. | `bevy_anti_alias-0.18.1/src/contrast_adaptive_sharpening/node.rs:72-115`, `mod.rs:126-133, 263-267`; `bevy_core_pipeline-0.18.1/src/core_3d/main_opaque_pass_3d_node.rs:71-72, 95-99`; `core_3d/mod.rs:1042-1046`; `texture_attachment.rs:40-80`; `wgpu-hal-27.0.4/src/vulkan/command.rs:861-867`; `bevy_core_pipeline-0.18.1/src/deferred/mod.rs:17-19`; `prepass/mod.rs:52-53` | VERIFIED (re-read) |
| F30 | Nothing after the air pass writes the picture's alpha again: the star sprites add alpha 0 (One + One); bloom's blend keeps the alpha (Zero, One); the tone map keeps it. The capture saves RGBA, and the exact-picture compare reads all four bytes of a content pixel. | `lib.rs:824, 862`; `bevy_post_process-0.18.1/src/bloom/upsampling_pipeline.rs:121-128`; `bevy_core_pipeline-0.18.1/src/tonemapping/tonemapping.wgsl:33`; `lib.rs:3475`; `crates/bins/tests/terrain_pictures.rs:1398-1405` | VERIFIED (re-read) |
| F31 | The hole census reads ONLY the probe image. The probe is a second camera on its own layer, with MSAA off and no prepass; it draws twin meshes with its own material (`probe.wgsl`). So it cannot see a hole that the picture camera's own depth test makes. | `crates/bins/tests/terrain_pictures.rs:857, 1651`; `crates/client-harness/src/probe.rs:181-195`; `lib.rs:3244-3262`; `terrain.rs:3629-3656` | VERIFIED (re-read) |
| F32 | The pop detector reprojects each pixel as ONE point that stands still in the body's frame: the probe's distance (a whole number of cells) through both stamped cameras. Its near limit keeps the miss of that distance under 0.25 px. | `crates/client-harness/src/pop.rs:10-19, 58, 134-136, 305-332` | VERIFIED (re-read) |
| F33 | The fourth patch's partial step serves the samples of ONE pixel: they share one ray direction from the pixel's centre and differ only in their distance. | `vendor/bevy_pbr/src/atmosphere/sky_air.wgsl:3-13, 40-41`; `vendor/bevy_pbr/PATCH.md` (the fourth change) | VERIFIED (re-read) |
| F34 | The thinning switch keeps every n-th triangle of every drawn chunk, so it opens holes in the picture. | `terrain.rs:174-180`; `DEFERRED.md:10670-10671` | VERIFIED (re-read) |

### 2.5 What outside sources say (REPORTED)

| Id | Claim | Confidence |
|---|---|---|
| R1 | Star Citizen forces TSAA. Since Alpha 3.23 (May 2024) it offers DLSS, FSR and "an in-house TSR solution". | H for DLSS, M for the rest |
| R2 | MSFS 2020 and 2024, No Man's Sky and Starfield ship TAA with upscalers. Elite Dangerous lists no temporal method. | M |
| R3 | Apple: "temporal AA and upscaling is here to stay". 4× MSAA is efficient on Apple GPUs only when the samples stay in tile memory with a resolve inside the pass (the pass discards the samples; a memoryless texture also removes their allocation). Apple's TBDR article: "Apple GPUs have an efficient MSAA implementation. The hardware tracks whether each pixel contains a primitive's edge so it runs the per-sample blending only when necessary" (the review quoted it; this revision did not read it again). The same article: the vertex stage's outputs flow through a tiled vertex buffer in external memory back to the fragment stage (as the review quoted it). A depth prepass for speed is unnecessary on Apple GPUs. | H |
| R4 | The MetalFX temporal scaler exists on macOS 13 and later; its jitter lies in [−0.5, 0.5] px; a reactive mask exists from macOS 14.4. Apple publishes no cost. One community campaign measured about 1.3 ms at 3 024 × 1 800 on an M5 Max, and 72 ms hitches from accidental scaler rebuilds. | H for the API, L for the costs |
| R5 | AMD FSR 2.2.1 is MIT. Its cost at 4K (Quality): 0.7 ms on an RX 7900 XTX, 1.4 ms on an RX 6800, 5.4 ms on an RX 590. No maintained WGSL port was found. Godot ships FSR 2.2.1. | H (M for "no port") |
| R6 | Unreal TSR: the time to one new sample per display pixel is 1000 ÷ (s² × the frame rate) ms: 66.7 ms at s = 0.5 and 60 Hz. | H |
| R7 | Hillaire 2020: the lookup sky costs 0.31 ms at 1280 × 720 on a GTX 1080; views from space switch to a per-pixel march. | H |
| R8 | Guerrilla (Nubis³) dropped the 16-frame temporal amortization for near clouds because fast cameras outran it; it uses two resolutions over depth instead. | H |
| R9 | wgpu 28 adds transient textures, and panics when one encoder mixes wgpu commands with the hal. A transient texture removes the samples' allocation; the in-tile resolve itself already works on wgpu 27 (F26). | H |
| R10 | The M4 Pro has up to 273 GB/s of memory bandwidth. | H |
| R11 | The survey (Yang et al. 2020): "MSAA is often cheaper than TAA on lower-end platforms (especially tile-based mobile GPUs)". | H |
| R12 | Star Citizen 3.23 listed "Misaligned HUD Markers with Ultrawide or TSR graphics upscaling enabled" as a known issue. | M |
| R13 | Frostbite (2016) traces its clouds at half size with temporal integration: 1.6 ms in the worst case at 1080p on Xbox One. Horizon Zero Dawn updates one pixel of each 4 × 4 block a frame: 20 ms became about 2 ms on PS4. | H |

---

## 3. The answer to the AAA question

**The question** (the owner): *"one-pass MSAA - how good is it for the AAA game? Today we have just the beginning and
only terrain - everything will be way more complex, with constructions, clouds, weather, hundreds of players. Is
one-pass MSAA the better solution for a star-citizen like mmo?"*

**The answer: no.** Temporal reconstruction is the foundation. One-pass MSAA is a good edge filter for a forward renderer
on a tile GPU. It is not the foundation for this game. **The decisive number comes first:** without any MSAA the stand's
frame is 21.75 ms (A2, MEASURED), over the 16.7 ms line. So no form of MSAA, one pass or many, reaches 60 frames a second
at full screen while the GPU draws every display pixel. Six reasons follow.

1. **It needs an engine change, not the Bevy upgrade.** "One pass" means that the four samples stay in the GPU's tile
   memory and never go to memory. wgpu 27 already does this when a pass resolves its samples and discards them (F26).
   Memoryless textures (wgpu 28, R9) only remove the samples' allocation, about 360 MB here. The block is the engine's
   pass structure: Bevy stores every colour attachment (F21), and the main pass, the air pass and the transparent pass are
   separate passes that each store and load the samples again (F15). One-pass MSAA needs the main passes, the sky and the
   air in ONE render pass with a discarded, resolved sample target. That is a change of the core pipeline (a second
   vendored crate, `bevy_core_pipeline`) and of the vendored air. D1 prices it as option (d).
2. **It smooths triangle edges only.** It does nothing for the shading: the glint of the sun on a hull's metal, the edge of
   a shadow. The game will hold both everywhere. (Thin rails and far ridges ARE triangle edges: MSAA smooths them to four
   samples. The temporal path needs FSR's locks for them later, risk 18.)
3. **It gives no render scale.** The full-screen frame is the pixels' (§2.3). Only a smaller drawn picture divides every
   per-pixel cost at once, and only a temporal reconstruction gives the full picture back. MSAA also grows with the
   triangles: smaller triangles shade more fragments along their edges.
4. **Some effects that the game still needs refuse it in this engine** (F22): contact shadows (SSAO), reflections (SSR),
   layered glass and water (OIT), meshlets, and the deferred light. The deferred light is a cheaper way to light many
   lamps under overdraw (UNMEASURED here); the lamps themselves and the decals work under MSAA through the clustered
   forward path (F28). The shipped games build clouds and weather on the earlier frames or on split resolutions (R8, R13).
5. **The industry.** Every planet and space game in the survey ships a temporal method as its main anti-alias, except
   Elite Dangerous (R1, R2). Apple itself: "temporal AA and upscaling is here to stay" (R3).
6. **The measured cost.** Without any MSAA the frame is 21.75 ms (A2): that number decides. Today's MULTI-pass MSAA with the
   per-sample air costs 10.5 ms (§2.2), mostly sample traffic (§2.3 item 3). One-pass MSAA removes exactly that traffic,
   and Apple states that its in-pass MSAA is efficient (R3), so its cost on this stack is UNMEASURED and can be much
   smaller. Even at zero cost it leaves the frame 5 ms over the line. A reconstruction costs 1.5–3.5 ms (ESTIMATE, §10.1),
   and it comes with the render scale that removes those 5 ms.

**What MSAA does better, stated plainly.**
- It is exact in each frame. It has no history, so it has no ghost and no lag.
- Every headless gate expects it.
- Thin geometry (rails, antennas, far ridges) gets four samples in every frame, with no history. The temporal path needs
  FSR's locks for it later (risk 18).
- Alpha-to-coverage (foliage, fences) works only under MSAA in this engine; without MSAA the same material cuts hard (F28).
- The survey states that MSAA "is often cheaper than TAA on lower-end platforms (especially tile-based mobile GPUs)" (R11).
  That holds for the in-tile case. wgpu 27 can resolve in the tile (F26), but this engine's pass structure stores and loads
  the samples today (reason 1).
- Elite Dangerous ships without a temporal method (R2).

So the temporal path must win on the owner's own pictures, not on this argument. D5 is that test.

**What can change the answer.**
- The owner refuses the temporal edge at every drawn size that meets the frame (D5, judged at 1.0, 0.67 and 0.5). Then MSAA
  stays, and 60 frames a second at full screen needs another source of about 15 ms. No such source is in sight: without a
  render scale, the looks' own cuts give a few milliseconds (ESTIMATE).
- One-pass MSAA through the engine change of reason 1 (D1 option (d)). The pass structure decides it, not the Bevy
  upgrade: wgpu 27 supports it. It removes the sample traffic. It still gives no render scale, so the frame stays at
  21.75 ms or more at full screen, and it still blocks SSAO, SSR, OIT and the deferred light.
- The reconstruction's own cost reads over its line at T2 (§11).

**Example.** A hull at 240 m/s crosses the belt at dusk. Later the sun glints on its metal, a station's rails pass under
it, and far ridges stand thin against the sky. Four samples smooth the triangle edges of the rails and the ridges, but not
the glint on the metal. The reconstruction smooths the glint too, from a quarter of the pixels, and every future effect
uses the same history. The thinnest rails need its locks later (risk 18).

---

## 4. The architecture

### 4.1 The frame, stage by stage

| # | Stage | Size | Jittered | Enters the history | Change against today |
|---|---|---|---|---|---|
| 1 | The sun's shadow maps (4 cascades, 2 048 px) | the light's own | no | no | none |
| 2 | **THE ONE GEOMETRY PASS**: the ground writes its depth, its surface words and its motion | render | yes | — | new: it replaces the ground's forward draw (T3) |
| 3 | The light pass: the sun, the shadow read and the air's light, once per render pixel | render | yes | yes | new (T3) |
| 4 | The forward things: the dots, the markers, the lit bodies (their depth and motion come from the engine's prepass) | render | yes | yes | MSAA off |
| 5 | The air pass, per render pixel (its cheaper form at T5) | render | yes | yes | no per-sample march, no second air pass, no sun disc |
| 6 | A copy of the colour, for the reactive mask | render | — | — | new, small |
| 7 | The transparent pass: the realm boxes today; glass, water and particles later | render | yes | yes, with the reactive mask | none |
| 8 | **THE RECONSTRUCTION** | render → display | its output is unjittered | writes it | new |
| 9 | **THE POINTS PASS**: the stars and the sun disc × the air's transmittance × the sky share | display | no | no | moved from stages 4–5 |
| 10 | Bloom | display | no | no | it reads the rebuilt picture |
| 11 | The tone map, the dither, the sharpen (D7) | display | no | no | the sharpen is new |
| 12 | The HUD (egui), the blit to the window | display | no | no | none |

In slices T1–T2 (route G3, §4.5) stages 2–3 are the engine's depth-and-motion prepass and today's forward pass.

**Example.** On foot at the stand at `s` = 0.5, stages 2–7 draw 1.9 million pixels. Stages 8–12 run on 7.5 million
pixels, as today.

### 4.2 How the reconstruction builds a display pixel

The reconstruction follows the published structure of AMD FSR 2.2 (MIT, credited; §8.2), in our own code. Every stage
runs at the render size except the last one.

1. **PREPARE** (render size).
   1. The sky's motion: a sky pixel (depth 0) gets its motion from the two unjittered views and the air's rotation step
      (§6.3).
   2. The dilated motion: each pixel takes the motion of the NEAREST surface in its 3 × 3 neighbourhood, so a silhouette
      moves with its foreground (Karis 2014; Bevy's TAA does the same with five taps, research).
   3. The rebuilt previous depth: each pixel writes its depth at its last-frame place into a storage buffer with an atomic
      maximum (reverse depth: the nearest wins). Each frame's depth reads with that frame's own near plane.
2. **THE DEPTH TEST** (render size). A pixel whose last-frame place holds a nearer rebuilt depth was hidden in the last
   frame. It gets no history (a disocclusion).
3. **THE REACTIVE MASK** (render size): the difference between the colour before and after the transparent pass (FSR 2's
   "generate reactive mask" rule). A translucent box, and later glass, water and particles, mark their own pixels with no
   second target.
4. **ACCUMULATE** (display size), for each display pixel:
   1. It gathers the render samples near the display pixel's centre with a Lanczos-2 weight, which counts each sample by
      its distance from the centre under this frame's jitter.
   2. It reads the history at the pixel minus the motion with a sharp filter (Catmull-Rom).
   3. It clips the history into the colour box of the new samples' neighbourhood (the mean ± a multiple of the deviation,
      in YCoCg). The reactive mask shrinks the box.
   4. It blends. The new frame's weight is 1/n, where n counts the pixel's samples. n starts again at 1 after a
      disocclusion or off the screen. n rises to a floor that changes smoothly with the motion: PROPOSED 1/64 below a
      hundredth of a pixel a frame, rising to 1/16 at one pixel a frame or more (D9). **Motion alone never restarts n.**
   5. It weights each sample by 1 / (1 + its luminance) in a reversible tone space, so one bright point cannot dominate
      (Karis 2014).
   6. It carries the sky share as a fourth channel with the same weights (§7.7).
   7. It writes the display colour (linear HDR, with alpha 1: §7.7, change two), the new history (the sky share in its
      fourth channel) and the count.
5. **LATER, ONLY IF A METER ASKS:** FSR's "locks" for thin features (rails, antennas) and FSR 3.1's shading-change stages.

**Why "motion alone never restarts n".** Bevy's TAA restarts its confidence when a pixel moves more than 0.01 px (F2).
Example: a pilot in orbit sees the planet turn 0.02 px a frame. Under Bevy's rule the limb stays at a 10 % new-frame weight
for ever, and it shimmers. Under this rule it converges like a still picture.

**The still eye.** At the 1/64 floor the jitter's remaining crawl is about 1/64 of an edge's contrast: under one level on
most edges (ESTIMATE; G-FLICKER measures it). Example: the owner stands still at the stand and looks at the horizon.
After about half a second the edge holds within one level, frame after frame.

**One settings struct.** Every constant of the reconstruction lives in ONE named settings struct, each constant with its
measured basis (the project's rule: no magic numbers). The look constants (the floor's ends, the box's width, the sharpen
strength) are the owner's (D7, D9).

### 4.3 The jitter

- A Halton (2, 3) sequence of ⌈8 / s²⌉ phases (FSR 2's rule): 8 at `s` = 1, 15 at 0.75, 18 at 0.67, 32 at 0.5. The
  sequence skips the zero shift: FSR 2 says that a jitter "should never generate a null vector", and Bevy's first phase is
  (0, 0) (F3).
- The phase index comes from the plug-in's own clock, which starts at 0 at each reset. So two captures of one stand see
  the same phases (Bevy takes the frame count: F3).
- The jitter moves only the projection of the render picture. The motion uses the unjittered matrices (research: Bevy's
  prepass uses the unjittered clip matrix).
- The texture bias is log2(s) − 1 (FSR, MetalFX): −1 at 1.0, −1.58 at 0.67, −2 at 0.5. The ground has no texture today
  (F11), so the bias waits for the ground step's textures.
- A HUD marker never uses the jittered projection. Example: Star Citizen 3.23 shipped with misaligned HUD markers under its
  upscaler (R12).

### 4.4 The render scale and its floor

- **The seam.** The engine's render-world override (F4), inserted as Bevy's DLSS plug-in inserts it (research). The
  override shrinks only the drawn rectangle; the textures keep the display size, and every pass stores its whole target
  (F29). The main-world camera keeps the display size, so every pixel law that reads the camera's viewport stays in display
  pixels: the dot floor and the star uniforms (F24). A design that shrank the main-world viewport instead would break those
  laws.
- **The view's two viewports.** Under the override the view uniform's `viewport` keeps the display size, and only
  `main_pass_viewport` takes the render size (F27). A shader that runs at the render size and maps a pixel to a place
  must read `main_pass_viewport`. The forward ground takes its world place from its vertex stage, so it needs no change;
  the deferred light and the cluster lookup need one (patch seven, §4.5). Example: at `s` = 0.5 the deferred light reads
  render pixel (1 700, 1 000) as display pixel (1 700, 1 000), so a ridge of the belt reads the shadow map far up and to
  the left of its true place.
- **THE FLOOR LAW.** The ladder switches a rung where one cell is one pixel of the reference view (F20). A shard ships an
  occupant while the occupant is at least one such pixel. So one render pixel may never cover more than one reference
  pixel: `s` ≥ 720 ÷ the display's rows (for a 45° view; for another view, the same angle). That is 0.33 on the owner's
  2 168 rows, 0.67 on a 1 080-row display and 1.0 on a 720-row display.
- **What the law guarantees.** At the floor, no ladder cell at its switch distance and no shipped occupant is smaller than
  one render pixel. The reconstruction never invents ground or a person from nothing. At `s` = 0.5 a cell at its switch
  distance covers 1.5 render pixels (3.0 display pixels today).
- **The dots.** The dot floor is a radius of 3 display pixels (F24). So a dot is at least 3 render pixels wide at 0.5 and
  2 at the floor.
- **A render-scale change keeps the history,** because the history lives at the display size. Dynamic resolution (D16)
  needs this. Only a change of the display size is a reset (§6.1).
- **Example.** A one-metre figure 870 m away is one reference pixel. At `s` = 0.5 on the owner's display it is 1.5 render
  pixels; at the floor it is exactly one.

### 4.5 One geometry pass: the route

| | G2: the deferred pass | G3: the engine's depth-and-motion prepass | G1: the forward pass with a motion target |
|---|---|---|---|
| What it is | The ground writes its depth, its surface words and its motion in one pass; one light pass lights each render pixel once | A second geometry pass writes the depth and the motion; today's forward pass then draws the colour | Today's forward pass writes the motion as a second target |
| Geometry passes for the picture | one | two | one |
| Engine change | three small patches in the crate that we already vendor: (5) the light pass takes the render rectangle (F9); (6) the main pass skips a masked deferred material (F8); (7) every pass that runs at the render size reads the render size's viewport (`main_pass_viewport`): the deferred light's pixel-to-place rule and the cluster lookup (F27). The air's own changes come apart (§4.8). | none: engine components only | a second vendored crate (`bevy_core_pipeline`: the main pass's attachments) and more `bevy_pbr` changes; every forward material's pipeline gains a target |
| Cost at the stand | UNMEASURED; the light runs once per pixel with no overdraw (a possible gain); plus G2's full-size targets outside the render rectangle (row W of §10.2, M-2b prices it) and the motion's vertex output (row V) | UNMEASURED until M-3 (the same chunks drawn a second time into depth and motion); the split of §2.2 suggests up to ~4–5 ms, but it is mixed with pixel work (F34); Apple states that a prepass gives no speed on its GPUs (R3) | about today's forward cost plus ~0.1 ms for the motion picture (ESTIMATE), plus the motion's vertex output (row V) |
| What it opens | SSAO (contact shadows) and SSR (reflections), which refuse MSAA (F22); each render pixel lit once, which makes many lamps cheaper under overdraw (UNMEASURED). Lamps and decals also work under MSAA through the clustered forward path (F28). | Bevy's GPU occlusion culling (experimental; it needs a depth prepass) | nothing new |
| Main risk | the surface words are a lossy packing (F10); a masked deferred ground draws twice without patch six (F8); at `s` < 1 every lit ground pixel takes a wrong world place without patch seven (F27) | HOLES: the main pass loads the prepass depth and tests "greater or equal" (research) against a depth from another shader module with no `@invariant`; ruling V18's GPU spike measured Metal compilations that differ by fused multiply-add. The probe cannot see these holes (F31). | the largest vendored diff; it must be re-based at Bevy 0.19 |

**The decision (D4).**
1. **The rule: the product draws the picture's triangles ONCE a frame.** Both drafts agree. A second pass for the motion
   would cost up to the triangles' ~4–5 ms (ESTIMATE from a mixed split, F34; M-3 prices it directly).
2. **The product route is G2 (deferred).**
   - It needs three small patches in the crate that we already vendor. The ground must stay masked, because the mask makes
     the shadow pass run the far-edge discard (`terrain.rs:2281-2284`).
   - The ground's own stages need no engine change. Its lighting is the standard one (F11), an extended material can state
     its own deferred stages (F12), and the deferred output writes the motion in the same pass (F10).
   - It opens SSAO (contact shadows) and SSR (reflections), which refuse MSAA (F22). It lights each render pixel once,
     which makes many lamps (ship interiors, stations) cheaper under overdraw; that gain is UNMEASURED here. The lamps and
     the decals for block damage work under MSAA too, through the clustered forward path (F28).
   - Its price: the surface words are a lossy packing, so ruling V18's tolerance applies exactly. G-DEFERRED-PARITY holds
     every content pixel within one level of the forward picture, at EVERY drawn size (patch seven decides the lit place
     at `s` < 1, F27). The ground is one colour with no texture today (F11), so the error is small, but it is UNMEASURED.
3. **The early slices (T1–T2) use G3, IF T0's M-3 counts no holes ON THE PICTURE CAMERA.** The probe cannot see a G3
   hole (F31), so M-3 counts them with a census of its own on the picture camera, and that census must first read its
   control (§11, T0). Until it does, G3's holes are UNMEASURED, and M-3 does not choose T1's route. Why G3 early: the first
   look at the new edge (T2, D5), at the sizes that ship, must come BEFORE the largest engine change. If the owner refuses
   the edge, the design does not need the deferred work, because MSAA and the deferred light cannot run together (F22).
   If M-3 counts holes, T1 builds G2 at once.
4. **G1 is the fallback** if G2 fails its parity gate or its cost gate. The owner decides (D4 again), because it adds a
   second vendored crate.
5. **G3 stays the geometry step's candidate** for the engine's GPU occlusion culling.

**Example.** On foot at the stand the 81 M ground triangles cost the GPU about 4–5 ms (a mixed split, F34). G2 reads them
once and lights 1.9 M render pixels once at `s` = 0.5. G3 reads them twice.

### 4.6 The contract for the game still to come

Every future drawn thing does ONE of three things: it states its motion, it marks itself reactive, or it draws after the
reconstruction. A thing that does none is a defect, and G-MOTION-COVERAGE catches it (§11, T1).

| Feature | Its cost class | How it enters | Motion, reactive, or after | Proposed budget line (D15) |
|---|---|---|---|---|
| Constructions, hulls, stations | geometry | the one geometry pass | the realm's step: exact also for block meshes that an edit spawns again | in the geometry line |
| Hundreds of players | geometry | the one geometry pass | the engine's per-mesh motion from the interpolated poses; a new player marks its first frame new; skinned bodies carry last frame's joints (research) | in the geometry line |
| The water fill (B22 step 4) and the ocean (8o) | render pixel | forward, with depth and motion | EXACT: the waves are a closed form of the tick (ruling T3), so last frame's surface is the same law one frame earlier; refraction and reflection are reactive | 1.0 ms |
| Clouds | render pixel, at a fraction | their own history; two resolutions over depth, never a 16-frame amortization for near clouds (R8) | the motion of the cloud's own depth where it dominates; else reactive | 2.0 ms |
| Weather: fog, dust, rain, snow | render pixel | the fog's density into the air volume (§7.5); the streaks reactive or in the points pass | reactive | 0.8 ms |
| Particles: exhaust, sparks, debris | render pixel | the transparent pass | reactive; large particles write motion | in the weather line |
| Screen-space light: SSAO, contact shadows, SSR | render pixel | the surface words hold the normals; at `s` < 1 its passes read the render size's viewport (patch seven's rule: SSAO reads `view.viewport` today, F27) | a noise that changes each frame and accumulates (Bevy's pattern for SSAO) | 0.7 ms |
| Many lamps | render pixel | the light pass, once per pixel; at `s` < 1 the cluster lookup reads the render size's viewport (patch seven, F27). The forward lamps also work under MSAA (F28). | — | in the light line |
| Decals (block damage) | render pixel | the one pass gives the depth; the engine's clustered decals need no prepass (F28) | — | in the light line |
| HUD markers | display pixel | after the tone map, with the display camera only | — | in the HUD line |
| An automatic exposure (a cave into daylight) | — | the exposure enters the reconstruction each frame | — | — |

### 4.7 Memory (ESTIMATE, at 3 456 × 2 168)

- **Goes:** the 4× colour (Rgba16Float) 239.8 MB; the 4× depth 119.9 MB; the per-sample air's two Rgba32Float textures
  239.8 MB. About 600 MB.
- **Comes:**
  - the history: two display-size Rgba16Float textures, 119.8 MB; the count, about 15 MB;
  - the surface words (Rgba32Uint) 119.9 MB, the light pass's id about 15 MB, the prepass depth copy 30 MB and the motion
    (Rg16Float) 30 MB. The engine allocates these at the full target size; the override shrinks only the drawn rectangle
    (F29). Their stores outside the drawn rectangle are memory TRAFFIC too, not only memory: row W of §10.2;
  - our own render-size textures and buffers (the dilated depth and motion, the rebuilt previous depth, the colour copy,
    the reactive mask): about 40 MB at `s` = 0.5 and 160 MB at `s` = 1.

  About 370–490 MB.
- **Net:** about 110–230 MB less.

### 4.8 The laws, checked

| Law | Verdict |
|---|---|
| SL6 | No new data crosses a realm boundary. The gateway ships the same rows. The client keeps its own copy of the rows that it drew, for one frame. |
| SL1 | The client states no placement. The realm's step is the drawing's own arithmetic on told rows. |
| No client-side prediction | The history is the past. The motion is this frame minus the last. Nothing extrapolates. The design does not propose frame generation. |
| SL8 | No reset on the player's path (§6). The pop detector keeps reading the frame before the history (§12). |
| SL9 | One step per realm, never per chunk: no cost grows with a realm's children. |
| SL10 | No world change. The generator, the wire and the realms stay untouched. |
| HR5 | The pure rules go into Tier-A crates at 100 % (§12.5). |
| HR6 | The captures read the frame before the history and the rebuilt picture; the stamp carries each frame's jitter. |
| V16, V18 | Every picture change waits for the owner's word. The surface words are a lossy packing held at one level (G-DEFERRED-PARITY). |
| F6 | The Mac is the instrument. The render scale is the dial for weaker machines. The reconstruction runs on every backend. |
| The Bevy upgrade gate | Nothing needs Bevy 0.19. The reconstruction is project code. The vendored crate grows by slice: **T2** the air pass stops adding the sun on sky pixels (`render_sky.wgsl`, the depth-0 branch); **T3** patches five, six and seven (the light pass's rectangle, the masked deferred skip, the render size's viewport); **T5** up to four air changes (the tables built once per body in `atmosphere/node.rs`; the light probe at its own pace in `atmosphere/environment.rs`; the open sky at half size and the 2 × 2 block march in `render_sky.wgsl` and `sky_air.wgsl`; the air volume). The MSAA-only changes (two and four) go at T8. Every vendored change dies at the upgrade gate (`PATCH.md`: "the patch dies at the Bevy upgrade gate (0.19)"), so each one must move to the new engine or go upstream then. |
| A new library | None without the owner (MetalFX's binding: D12). |

---

## 5. Motion for every drawn thing

### 5.1 The one rule: the realm's step

**The previous place of a drawn thing is its realm's step applied to its place now, then the thing's own change inside
the realm.**

- **The realm's step** `S_r` maps this frame's render space to the last frame's render space for realm `r`:
  `S_r = P_r(last frame) · P_r(this frame)⁻¹`. Here `P_r(x) = centre + facing · x` is the placement that the window DREW
  for the realm: its row's centre and facing, as the placement builds them (F18). The client keeps last frame's drawn
  placements for one frame.
- Where a row was absent and the chunks stood still, the drawn placement did not change, so `S_r` is the identity.
- **The motion describes the drawn picture, not the truth.** A drawing error stays the geometry's defect. The
  reconstruction neither causes it nor hides it.
- **The own change inside the realm:** for the ground, the morph, the sink and the ramp at last frame's band row and clock
  (§5.2); for a dot, its previous interpolated pose; for a static block, none.

**Why one rule.**
1. A ship, a station, a moon and a rock get their motion from identical code (the spirit of SL4 and HR4).
2. A NEW chunk gets its true motion in its first frame, which the engine cannot give (F7).
3. A scene swap needs nothing special: for each realm, both placements are the ones drawn in their own frames.
4. The laws hold (§4.8): no new data crosses a boundary; the client states no placement; it looks one frame back and
   predicts nothing.
5. The cost does not grow with the chunks: every chunk of a realm rides one row (F18), so one step per realm is exact for
   all of them (SL9). The placement already walks the scene's rows each frame.

The client computes the step in f64 from the two drawn placements and narrows it once, as the placement does
(`terrain.rs:4541-4543`).

**Example.** A chunk that the harvest spawns this frame stands 1 km beside a hull at 528 m/s. The engine's rule gives it no
motion: an error of about 23 display pixels for one frame (8.8 m ÷ 1 km ÷ 0.38 mrad a display pixel; ESTIMATE). The realm's
step gives it the planet's own step, so it enters with no mark.

### 5.2 The ground's exact previous place

**Today's rule is not enough** (F5): it moves last frame's placement by THIS frame's displacement. That is exact only when
the eye distance, the band row, the clock and the axes stay the same. Its fragment stage writes no motion at all, so every
ground pixel reads "no motion".

**The new rule.** This frame a vertex stands at PLACED(own, d, row, a): the morph toward the coarser surface by the row at
the eye distance d, the sink, then the ramp weight a (`ladder_fade.wgsl:136-158`; `arrival.rs:96-127`, re-read). Last
frame it stood at PLACED(own′, d′, row′, a′):
- own′ = `S_r`(own);
- d′ = |own′|, because the eye stands at the origin in both frames;
- the radial′ = the rotation of `S_r` applied to the radial (a chunk carries no scale);
- row′ = last frame's band row;
- a′ = the ramp weight at last frame's clock, **with the age read as SIGNED**.

**★ THE WRAPPING TRAP (F6, VERIFIED).** The ramp's weight reads `now − stamp` as a wrapping unsigned difference. A chunk
that lands this frame carries a stamp that is one frame NEWER than last frame's clock. The wrapping difference reads that
as an age of about 2³² ticks, so the weight reads 1 ("whole") instead of 0 ("on its stand-in"). The previous-place rule
must read the difference as signed: a negative age is weight 0.

**The landing identity.** A landing chunk's stamp is this frame's clock (`arrival.rs:48-51`), so a′ = 0, and its previous
place is its stand-in's drawn surface at last frame's state: exactly where the stand-in stood. Its first motion equals the
stand-in's motion, so the history carries across the landing.

**Example.** A hull dives over the belt, and a rung-9 chunk lands late. With the wrapping read, its first motion carries
its whole rise in one frame (at least one cell of each rung: at least 3 reference pixels; ESTIMATE), and the history
smears it. With the signed read, its previous place is the stand-in's surface, and the landing leaves no mark.

**The words it needs.**
- The band table keeps last frame's rows beside this frame's rows.
- The header's two free words (F19) carry last frame's clock. Three more header rows carry `S_r` (a 3 × 4 matrix).
- The renderer already writes every table every frame (F19). The picture's stages, the prepass stages and the probe read
  the same layout, so all three transliterate ONE Tier-A layout.

**The cost.** Two parts.
- The vertex stage places each vertex twice. That part is arithmetic.
- The vertex stage also OUTPUTS last frame's place for each vertex, because the fragment stage needs it. On this GPU the
  vertex stage's outputs go through a tiled vertex buffer in memory and flow back to the fragment stage (R3, Apple's
  article as the review quoted it). So the motion adds memory traffic for each drawn VERTEX. That traffic grows with the
  ~81 M ground triangles and does not shrink with the render scale (row V of §10.2). The ground's prepass already outputs
  four floats for it under the motion switch (F5). Its size is UNMEASURED.
- T1 prices both parts on the clock at the stand, with the motion output on and off.
- The output stays the smallest EXACT one: three components (last frame's clip x, y and w; z is not needed). Two
  components would lose the perspective division, so the motion would bend across a large triangle. Half floats come only
  behind the adapter's f16 feature, and only if G-MOTION-TRUTH still holds with them.

**Example.** A hull dives over the belt. Every drawn vertex of every rung sends last frame's place through memory, at
`s` = 0.5 exactly as at 1.0, because the vertices do not shrink with the picture.

**The Tier-A mirror (HR5).** In `vd_client`, beside `arrival.rs` and in its pattern ("the shaders transliterate it, step for
step"), in f64, with unit tests for: the realm's step; the previous placed vertex; the landing identity; a band slide; a
swap; the signed age at the clock's wrap.

### 5.3 Every drawn thing

| Drawn thing | Where the code makes it | It rests in | Its motion |
|---|---|---|---|
| Ground chunk | `terrain.rs:3583-3596`; placed `4539-4571` | its realm | the rule of §5.2 in its own vertex stage; the ground does not use the engine's per-mesh memory |
| Shadow caster | `terrain.rs:3478-3512` | — | none: shadow views keep no history |
| Probe twin (capture only) | `terrain.rs:3629-3656` | — | none (an instrument); it takes the picture's jitter (§12.1) |
| Dot (a player) | `lib.rs:2082-2172` | its realm | the engine's per-mesh previous transform; a new dot carries a spawn stamp (the ground's arrival-stamp pattern) and marks its first frame new (reactive) |
| Marker, lit body, the ruler | `lib.rs:2457-2614`; `terrain.rs:3920-3980` | its realm | the engine's; the marker-to-body handover keeps ONE entity (today it despawns and spawns in one frame: `lib.rs:2481-2497`) |
| Translucent realm box | `lib.rs:2700-2737` | its realm | none: the reactive mask (§4.2 step 3) |
| The air on sky pixels | the air pass | the air's body | the reconstruction's prepare stage: the two unjittered views and the air's rotation step (§6.3) |
| The sun disc, the stars | the points pass | the star system, the galaxy | none: they draw after the reconstruction (§7.7) |
| The HUD | egui | — | none: it draws after the reconstruction |

### 5.4 What today's formula gets wrong

ESTIMATES from the renderer research:

| Term | Size |
|---|---|
| The morph and the sink by distance | ≤ ~0.1 px a frame at 528 m/s through rung 2's 700 m band |
| The band table's slide | the same order or less (≤ 0.25 of the horizon a second: `ladder_view.rs:377`) |
| The arrival ramp | ≥ ~0.3 px a frame for 30 frames (a stand-in's sink is at least one cell of each rung) |
| The axes at a swap | the displacement turned by the swap's rotation: several pixels at a far rung, for one frame |
| A new chunk | about 23 display pixels for one frame (1 km beside a hull at 528 m/s) |
| The fragment stage | writes no motion at all: every ground pixel reads zero (F5) |

---

## 6. The history: resets and seams

### 6.1 The policy

**The history is NEVER reset on the player's path.** A reset happens only when no history exists, or when the history holds
another picture:
1. the first frame;
2. a change of the display size (a window resize, the full-screen switch);
3. a device loss;
4. a tool's jump: a gate's teleport, the capture's scene-fitted camera (`lib.rs:2623-2681`), a stand's start.

The dev stamp counts each reset with its cause (G-RESET). Every flight leg must read 0 resets on the player's path.

**Why.** The first frames after a reset look "overly sharp and aliased" (the survey, REPORTED H). Unreal's sky restarts its
TAA at very fast camera moves, "so it is visible in space views" (REPORTED H). Under SL8 a visible reset is a seam.

### 6.2 Event by event

| Event | Where | What changes on the screen | The history's rule |
|---|---|---|---|
| Every frame | `lib.rs:1866-2055` | the eye; every placement | the realm's step, per realm |
| A realm crossing: a boarding, a departure | `lib.rs:1900-1956` | the render frame's axes turn; the view takes the delivered facing | each realm's step spans its two drawn placements; the air's rotation step for the sky; NO reset |
| The view switch (V: the eye goes 60 m back) | `lib.rs:1688-1697` | a translation; the hull box appears | the realm's steps carry it; the uncovered strip is a disocclusion; NO reset (the drafts differ: §15 row 4) |
| The lock (L: a snap turn to the nose) | `lib.rs:1704-1735` | a rotation | the two views carry it; NO reset |
| The eye held by a refusal, then its catch-up step | `lib.rs:2022-2027` | a jump of the eye | the realm's steps carry it; NO reset |
| The render cursor re-anchors on a newer tick | the render clock | the world steps in time | the true positions; NO reset |
| A rung swap, a harvest, a landing chunk | `ladder_fade.wgsl:191-195`; `arrival.rs` | a surface ends at its far edge; new chunks enter | exact motion; the landing identity; NO reset |
| A render-scale change | — | the render picture's size | the history lives at the display size; NO reset |
| The air's handover (the pick of the air's body) | `sky.rs:175-190` | the air changes | the clip; the handover measured no step on any pixel (`DEFERRED.md:8607-8615`); NO reset. If T5 builds the air volume (§7.5), ITS own accumulation restarts at the handover. |
| A warp (a declared state; fast motion) | — | fast motion | the motion and the disocclusion test; NO reset |
| A window resize, the full-screen switch | — | the display size | RESET: the history takes the new size |
| A gate's teleport, the capture's scene-fitted camera, a stand's start | `lib.rs:2623-2681` | a cut by a tool | RESET, counted apart from the player's path |

### 6.3 The sky at a swap

- A sky pixel (depth 0) reprojects by rotation only: from this frame's unjittered view back to last frame's view, turned
  by the air's rotation step `R_air` = the rotation part of the air body's realm step.
- On the home planet, which is the origin, `R_air` is the identity. In a rolling hull, `R_air` is the roll. At a boarding,
  `R_air` turns the planet's axes into the hull's axes.
- `R_air` needs the air body's row in both frames' scenes. The boarding instrument already reports whether the realm that
  the pilot LEFT keeps a row in the new scene (`lib.rs:1928-1935`, re-read).
  - Fallback 1: the rotation between the star cloud's two transforms (its rotation is the origin's orientation in the
    galaxy, inverted; research). The air body's own spin in one frame is about a millionth of a radian (ESTIMATE), far
    under a pixel.
  - Fallback 2: a restart of the sky pixels' history only, counted by G-CROSSING.
- The air's glow on sky pixels belongs to its body, so a rotation-only reprojection ignores its parallax. On the fast leg
  that parallax is about 0.23 display pixels a frame (ESTIMATE: 528 m/s ÷ 60 = 8.8 m a frame; 8.8 m ÷ 100 km = 0.088 mrad;
  ÷ 0.382 mrad a display pixel, from the window's 45° view over 2 168 rows, `lib.rs:1422`). At 120 frames a second it is
  0.12. So the sky's history can trail the air's glow on the 528 m/s leg.
- G-CROSSING's sky reading and the sky judge's lag reading measure that trail on the 528 m/s leg (T2). If it shows, the sky
  pixels reproject with the air body's full step, as a warp near a body needs anyway.

**Example.** A hull berthed 90° off the planet's axes departs (a realm crossing). In the swap frame `R_air` turns the
history of every sky pixel by those 90°, and the air stays still on the screen. Without `R_air`, the whole sky would
reproject 90° wrong for one frame.

### 6.4 Seams that exist today

The reconstruction neither causes these seams nor hides them. The pop detector reads the render picture before the history
(§12.3), so they stay visible to the gates.
- **A row absent inside its 120 ms grace:** the ground rides the eye, then jumps (`terrain.rs:3232-3258`). UNMEASURED.
- **A rebuilt key that was drawn enters WHOLE** (stamp 0): its shape changes in one frame (`arrival.rs:42-51`).
- **Forget and wholesale release:** the band builds again (`terrain.rs:3259-3294, 3389-3440`).
- **Two instants in one frame:** the camera and the ground read one moment; the dots, the realm boxes and the star cloud
  read their own clocks (research). At 528 m/s one millisecond is half a metre. UNMEASURED.

---

## 7. The sky and the air at reduced resolution

### 7.1 The reference

- The owner ruled ONE sky mode, the raymarched one (`DEFERRED.md:8621-8627`, re-read). The lookup mode differed by up to
  142 channel steps at the ground stand and left 7 % of the day side black from orbit: "a wrong picture, not a cheaper
  one".
- So G-AIR judges every cheaper air in this design against the per-pixel march: at most one level at every stand.

### 7.2 The air per render pixel (T2, T4)

- With MSAA off, the air pass takes its per-pixel path (`render_sky.wgsl`, the non-multisampled branch, re-read). The
  per-sample patch and the second air pass have no work (F15).
- The air pass already takes the render rectangle (F9). Its ray comes from the full-screen `uv` inside that rectangle and
  its depth from the pixel's own place, so it needs no change at `s` < 1 (F27).
- Its cost is about 6.6 × s² ms: 1.7 ms at 0.5 and 3.0 ms at 0.67 (DERIVED, if linear in the pixels).
- Each render pixel casts its ray from its jittered place, so the ground's edge and the air's edge share one jitter. The
  reconstruction then resolves the horizon as a supersampled edge, with no edge detection (edge detection was SMAA's
  weakness).

### 7.3 The exact cuts (T5 step 1)

- The transmittance table and the multiscattering table read only the body's medium and radii. The design builds them once
  per body, and again at a handover. 0 pixels change.
- The sky-view table stays per frame (the eye and the sun move), because the light probe reads it.

### 7.4 The cheaper march (T5 steps 2–3; draft A)

1. **The light probe at its own pace:** one cube face a frame, so a light lag of six frames (0.1 s). Gate: at most one
   level.
2. **The open sky marched at half the render size on each axis** (a quarter of its marches), then up-sampled with the full
   depth. The sun is no longer in this field (it moved to the points pass), so the field is smooth.
3. **The air over the ground marched in 2 × 2 blocks.** A block marches once to its mean distance when its four depths lie
   within one march step. Each pixel then extends or shortens that march by ONE PARTIAL STEP (the medium at the partial
   step's middle, the march's own step formula). An edge block marches per pixel.
   - **This block rule is NEW and UNMEASURED.** The fourth patch measured its partial step at 0 pixels over one level
     against the per-sample march (`DEFERRED.md:8940-8944`, re-read), but only for the samples of ONE pixel: they share
     one ray direction and differ only in distance (F33). A 2 × 2 block has four ray directions, and one march along one
     direction does not follow each pixel's own direction.
   - T5 therefore measures the block's DIRECTION error first, at the sun's glow and at the horizon (the dusk, aloft and
     orbit stands), before T5 ranks §7.4 over §7.5.
   - Example: near the sun's glow at the dusk stand, two neighbour pixels of one block look through different air. The
     measured rule says nothing about that difference; the direction measurement does.

- **Why first:** no noise and no history. Its partial step reuses a rule measured in this tree for one pixel's samples; the
  block's direction error is the open part. The lookup table stays refused.
- A temporal march (a moving march offset that the history accumulates) is NOT proposed first: it trades a sky with no
  noise for noise that motion shows.
- The saving: 40–60 % of the air's per-pixel cost (ESTIMATE).

### 7.5 The air volume (T5 step 4; draft B), measured beside §7.4

- A table of the air's light and transmittance along each view direction, by distance: 1/8 of the render grid on each
  axis, and 64 slices that grow with the distance (exponential), out to the drawn radius.
- Each column is one ray with a moving offset, marched with the raymarch's own medium functions, and accumulated by
  reprojection with the air body's step.
- Each render pixel reads the table at ITS OWN depth, so a depth edge has no halo.
- Rays that graze the shell from space keep the per-pixel march (Hillaire's method also switches to a per-pixel march for
  views from space: R7).
- **Why it is not first:** the owner refused one lookup table already. That table had 32 LINEAR slices (28 km a slice at a
  900 km reach) and a 32 × 32 screen grid. The volume has 64 slices that grow with distance and 8-pixel columns, but it is
  still a lookup, and its moving offset brings noise under motion.
- **Why the design keeps it:** weather writes its fog, dust and rain density into the same table later, and one march
  function serves both.
- G-AIR decides between §7.4 and §7.5. The owner rules any cut that fails the one-level gate (D10).

### 7.6 What stays per pixel

- The composite of each pixel with its OWN depth. Example: a ridge 2 km away stands in front of a valley 40 km away; each
  takes its own air.
- The sun disc: analytic and sharp.
- The edge between the ground and the sky: under MSAA it is per sample; under this design it comes from the jitter and the
  history (D5).

### 7.7 The points pass: the sun and the stars after the reconstruction (D6)

**Why.**
- They are the brightest and the sharpest things in the picture.
- They rest in other frames than the air: the stars in the galaxy, the sun in the star system.
- A star's core is σ 0.6 display pixels with a 1-pixel minimum crop (F17). At `s` = 0.5 that core is 0.3 render pixels.
  A jittered render picture holds it in some frames and not in others, so the history's clip deletes it. That is the known
  failure for points under a pixel (Karis 2014; Bevy's own warning, research).
- The star law states its sizes in display pixels (F24). At the display size after the reconstruction, the sprites are exactly
  today's sprites.

**How.**
- A pass between the reconstruction and bloom draws today's star sprites and the analytic sun disc additively at the
  display size.
- The pass multiplies each sprite pixel by two factors:
  1. the air's transmittance along its own direction, from the transmittance table (the engine builds it every frame in
     both air modes);
  2. the SKY SHARE at its display pixel.
- **The sky share = 1 − the RENDER picture's alpha.** Three VERIFIED facts make it exact:
  1. an opaque or masked surface writes alpha 1, and the deferred light writes alpha 1 (F13, F14);
  2. the air pass keeps the alpha, because its alpha blend is Zero, One (F14);
  3. a translucent box blends "over": the engine's blend for translucent things treats the alpha channel as "over"
     (`vendor/bevy_pbr/src/render/mesh.rs:2470-2472`; `wgpu-types-27.0.1/src/lib.rs:1785-1792`, re-read), so 1 − alpha
     is exactly the share through which the stars show.
- **★ CHANGE ONE:** the clear colour's alpha must be 0. Today it is 1 (F16).
- The reconstruction carries the sky share as a channel of its own with the colour (the history's fourth channel). So a
  star beside a ridge takes the ridge's true accumulated coverage. The points pass reads the sky share from that channel.
- **★ CHANGE TWO, THE PAIR OF CHANGE ONE:** the DISPLAY picture's alpha must be 1 again. The reconstruction writes the
  display colour with alpha 1, and the sky share stays in its own channel. Nothing else would do it: the star sprites add
  alpha 0, bloom and the tone map keep the alpha, and the capture saves RGBA (F30). Without change two every sky pixel of
  the owner's pictures reaches the PNG with alpha 0, and every edge pixel with a part alpha. The window would not show the
  defect (the tree sets no alpha mode, so its surface stays the engine's opaque default), but the pictures would.
- **The check (G-ALPHA, T2):** every pixel of the capture's display picture has alpha 255.
- **The cost:** the points pass is a pass of its own at the display size, so it loads and stores the 8-byte display
  picture once: 120 MB, about 0.44 ms at 273 GB/s (row S of §10.2).

**Examples.**
- At the dusk stand a star sets behind the belt's ridge. It dims smoothly over the last display pixel and goes out. It
  never flickers in and out with the jitter.
- Without change two, the P7 dusk picture opens in the owner's viewer with a white or checked sky over the belt, while the
  same frame looks right in the window.

**The same light as today, with two named differences.**
- Today the stars draw before the air pass (F17), and the air pass composes the in-scatter + the transmittance × the stars.
  The new order gives the same sum.
- **Difference 1:** the transmittance comes from the table (40 samples a ray) and not from the pixel's march (16 steps). The
  star census measures it (target: the summed light within 1 %).
- **Difference 2:** the vendored air pass stops adding the sun on sky pixels (a vendored change: `render_sky.wgsl`, the
  depth-0 branch).
- A translucent star-system shell dims the stars behind it exactly as today, through the sky share.

---

## 8. The reconstruction: the choice of upscaler

### 8.1 The options

| | U1 Bevy's TAA | U2 Bevy's TAA + a spatial up-scale | **U3 OUR reconstruction (recommended)** | U4 A faithful FSR 2.2 / 3.1 port | U5 Apple MetalFX temporal scaler | U6 NVIDIA DLSS | U7 Intel XeSS, AMD FSR 4 |
|---|---|---|---|---|---|---|---|
| Up-scales | no (F1) | yes, spatially | yes | yes | yes | yes | yes |
| Runs on the owner's Mac through this stack | yes | yes | yes | yes, if the adapter reports read-write storage (T0 reads it) | yes, through the hal (prior art for Bevy 0.18 / wgpu 27: `bevy_metalfx` 0.1–0.3, REPORTED M) | no: Vulkan and RTX only | no |
| Runs on Windows / Linux GPUs | yes | yes | yes (the portable rules, §8.4) | yes (modern GPUs) | **no** | RTX only | Windows only |
| Licence | MIT | MIT | our code; FSR 2.2 (MIT) credited as the reference | MIT | Apple's closed framework + a NEW binding library (`objc2-metal-fx` 0.3.2) | NVIDIA's SDK; the game ships its DLL | binary |
| Effort | small | medium | large: 4–5 passes, the motion, the points pass, the gates | larger: HLSL to WGSL, typed storage, 16-bit types; no maintained WGSL port found (R5) | medium: its own encoder (F23), the input contract | small–medium (Bevy has the node) | — |
| Quality | fixed constants (F2), no depth test, the confidence restarts on motion | below a temporal up-scaler: the jitter's detail stays at the render size | FSR-2 class, tuned on our stands | FSR 2.2's published level | Apple's scaler; likely the best on this Mac at `s` ≤ 0.67 (UNMEASURED) | the best on RTX | — |
| Cost at 3 456 × 2 168 | ~1–2 ms, but the pixel side stays native: the frame 24–28 ms (§10.2, `s` = 1) | ~1.5 ms (ESTIMATE) | 1.5–3.5 ms (ESTIMATE, §10.1) | 0.7–5.4 ms at 4K by GPU class (R5); the M4 Pro is UNMEASURED | ~1.3 ms at 3 024 × 1 800 on an M5 Max (R4, L); the M4 Pro is UNMEASURED | 0.9–3.2 ms on an RTX 3070 (REPORTED H) | — |
| Pinned for the gates | no (the frame count, F3) | yes | yes | yes | no: an OS update can move its picture | no | — |
| Takes our special inputs (the sky share, the air's rotation, the reactive mask) | no | no | yes | with changes | only through its fixed inputs (the reactive mask from macOS 14.4) | through its fixed inputs | — |
| Use here | a first price and look only (T2, optional); never shipped | no | **the path** | the reference, and the fallback if U3 fails its pictures | a challenger at T9, on D12 | later, per platform, on D12 | no |

### 8.2 The recommendation: U3, our own reconstruction

1. **Our special cases need a pass that we own:** the air's own rotation (§6.3), the sky share (§7.7), the motion floor
   that never restarts on motion (§4.2), the reactive mask from the boxes, the reset counter. A closed scaler takes only its
   fixed inputs.
2. **One picture on every GPU.** Ruling B17 chose the shader's own level over the vendor's filter, so that "the same picture
   on every GPU" holds. A vendor's machine-learning scaler gives a different picture per vendor.
3. **The average machine (F6).** A Windows or Linux player needs a portable path anyway.
4. **The gates.** Our pass is deterministic on one machine, and no OS update moves the frozen pictures.
5. **The honest price.** U3 is the largest effort, and MetalFX may beat it on this Mac at 0.5. That is why MetalFX runs as a
   measured challenger (T9), on exactly the same inputs, if the owner allows the library (D12).

**The reference algorithm: FSR 2.2's structure** (fewer passes than FSR 3.1, one published text, a shipped port in
Godot). FSR 3.1's added stages (the shading-change and luma-instability stages) come only when a meter asks for them.

### 8.3 The seam (one per camera)

- **Place:** after the main passes, before bloom, where Bevy places DLSS (research).
- **Inputs:** the render colour (linear HDR, jittered, the air composited, no stars and no sun disc); the depth (reverse
  and infinite, with the frame's own near plane); the motion (Rg16Float, this frame minus the last, in UV units); the
  jitter in render pixels; the reset flag; the exposure (fixed today); the reactive mask; the sky share (the colour's
  alpha).
- **Outputs:** the display colour, with alpha 1 (§7.7, change two), and the sky share in a channel of its own.
- **An accelerator** maps the same inputs: MetalFX takes the motion from the current place to the previous one in render
  pixels (a sign and a scale), the jitter in [−0.5, 0.5], and a flag for the reverse depth (R4). Bevy's DLSS node passes
  the jitter negated and the motion scaled by minus the render size (research).

### 8.4 The portable rules (for U3)

1. Read by sampling. Write by render targets or by write-only storage. Never use read-write Rgba16Float storage, which on
   Metal exists only at the second read-write tier (research).
2. Put the atomics in storage BUFFERS, never in textures.
3. Use f32. Use f16 only as an optional fast path behind the adapter's f16 feature.
4. Require no subgroup operations.

**Example.** A player on a Windows laptop with an RTX 3060 and a 1080p screen gets the same reconstruction. The floor for
1 080 rows is 720 ÷ 1 080 = 0.67, so the render scale there is 0.67 or more.

### 8.5 If the owner allows MetalFX (D12)

- It encodes on its OWN encoder, as the first action, and the frame submits that encoder in order (F23; `bevy_metalfx`
  does this; wgpu 28 panics on a mixed encoder: R9).
- The client makes the scaler at the start and at a display-size change, never per frame (R4: 72 ms hitches from
  rebuilds).
- Its exact contract (the legal formats, the motion's sign and scale, the minimum macOS) comes from Apple's pages, which
  the research did not read in full: UNVERIFIED.
- It is never the gates' reference: an OS update can move its picture.

---

## 9. Shadows and bloom

**Shadows.**
- The maps do not change: 4 cascades, 2 048 px, out to rung 2's switch distance, about 3.5 km (`terrain.rs:109-121`,
  re-read).
- The shadow read runs once per render pixel, so it scales with s². Under G2 it has no overdraw. Under G2 at `s` < 1 it
  reads the right place only with patch seven, because the deferred light rebuilds each pixel's place from the viewport
  (F27).
- The jitter moves each pixel's sample over the map, so the history smooths the shadow's edge.
- **D11:** the engine's Temporal filter (8 taps, turned each frame) against today's Gaussian filter (9 taps; research). It
  is a picture change. Its gain is a smoother penumbra after accumulation, not speed.
- **The crawl risk:** the engine snaps the cascades to texels in the light's space (research). With the floating origin
  that snap grid follows the eye, not the ground, so a shadow's edge can crawl while the owner walks. UNMEASURED. T6
  measures it with the flicker meter on shadowed ground; the cure (a snap grid fixed to the realm) is an item of its own.
- **Example.** The owner walks at the stand past a boulder. Today its shadow's edge is a fixed 9-tap blur. With the
  Temporal filter and the history, the edge is a smoother penumbra; if the grid crawls, the flicker meter reads it.

**Bloom.**
- It runs after the reconstruction at the display size: the engine's own order (TAA or DLSS, then bloom, then the tone
  map; research). Its input is the stable rebuilt picture, so bloom stops flickering on bright points under a pixel.
- T6 times its 16 passes one by one. Its mip chain has a fixed size (816 × 512 at the top; research).
- The cause of its 4.6 ms under MSAA, against 1.85 ms (with the shadows) without MSAA, is still unknown (§17).
- If bloom costs more than its line (0.8 ms), it reads a half-size copy that the reconstruction writes on its way (Unreal's
  TSR does this: REPORTED H). This is a small glow change: D11.

**The tone map, the sharpen, the HUD.**
- The tone map does not change. The exposure is constant (EV100 9.7), so the reconstruction needs no exposure input. A future
  automatic exposure feeds it each frame.
- The sharpen: Bevy's contrast-adaptive sharpen after the tone map, at the owner's strength (D7). It is a full-screen pass
  of its own on the 8-byte HDR picture (F29): one read and one write, 120 MB, about 0.44 ms at 273 GB/s (row H of §10.2).
- The HUD draws after the tone map at the display size, never jittered and never scaled.

---

## 10. The GPU budget against 16.7 ms

### 10.1 The model

The frame at render scale `s`:

`frame(s) = F + V + O + s²·P_r + s²·A + (1 − r)·B + r·B·s² + M + Rc + S + H + W(s)`

| Symbol | Piece | Value | Tag |
|---|---|---|---|
| F | the fixed part: the ground's triangles (~4–5) and the fixed passes (~2–3); the split between them is mixed (F34) | 7.2 ms (13.27 − 6.1) | DERIVED |
| V | the motion's vertex output: last frame's place for each drawn vertex, through the tiled vertex buffer (§5.2). It is per vertex, so it does not shrink with `s`. | not in the sums | UNMEASURED; T1 prices it |
| O | the post passes at the display size: the tone map, the blit, the HUD, the clock's own end pass | 1.0–1.5 ms | ESTIMATE. The floor is their traffic: the tone map reads and writes the 8-byte picture (120 MB), the blit reads it and writes the 4-byte window picture (90 MB), the end pass loads and stores that (60 MB): 270 MB, about 1.0 ms at 273 GB/s. |
| P_r | the looks-off pixel work at the render size: the ground's light and depth, the lookup air's share | 6.1 − O | DERIVED − ESTIMATE |
| A | the raymarched air over the lookup air, per render pixel | 6.6 ms | DERIVED (A2 − A3) |
| B | bloom and shadows without MSAA | 1.85 ms | DERIVED (A3 − A4) |
| r | the share of B that is the shadow read (per render pixel); the rest is bloom, the casting and their draws | 0.3–0.5 | ESTIMATE, bounded: bloom's own full-size traffic (about 180 MB, §17 item 4) takes at least 0.66 ms at 273 GB/s, and the casting about 0.3 ms, so r ≤ (1.85 − 0.3 − 0.66) ÷ 1.85 ≈ 0.48 |
| M | the motion's pixel side, the surface words, the reactive copy | 0.2–1.0 ms | ESTIMATE |
| Rc | the reconstruction | 1.5–3.5 ms | ESTIMATE |
| S | the points pass: a pass of its own that loads and stores the 8-byte display picture (120 MB, 0.44 ms at 273 GB/s), plus the sprites' own work | 0.45–0.8 ms | ESTIMATE; the traffic is the floor |
| H | the sharpen (D7): one read and one write of the 8-byte display picture (F29) | 0.45–0.6 ms | ESTIMATE; the traffic is the floor; 0 if the owner chooses no sharpen |
| W(s) | the WHOLE targets: G2's full-size targets outside the render rectangle (W, because T names the slices): each pass stores its whole target (F29), so the untouched tiles of 23 bytes a pixel move (1 − s²) × 172 MB at each store | not in the sums; at most about 0.47 ms at 0.5 if every untouched tile stores once; 0 if the GPU skips or compresses them | UNMEASURED; M-2b prices it |

**One rule for every full-screen pass.** The design prices the reconstruction by its traffic at the M4 Pro's 273 GB/s
(below). The same rule now prices S, H, O and bloom's share of B. If the GPU's lossless compression applies to these
textures, every such floor reads lower; M-1 measures them.

**The reconstruction's own cost (Rc).**
- The traffic floor: about 40 bytes a display pixel (the history, the count, the output, the new history, the render
  samples), so about 300 MB a frame, so about 1.1 ms at the M4 Pro's 273 GB/s (R10; the rest ESTIMATE).
- The published costs of FSR 2 at 4K run from 0.7 ms on a large desktop GPU to 5.4 ms on an older mid-range GPU (R5). The
  M4 Pro's place on that line is UNMEASURED.
- So the range is 1.5–3.5 ms, 2.5 ms in the middle. T2's first gate measures it.

**The assumptions.** Low: O = 1.0, r = 0.5, M = 0.2, Rc = 1.5, S = 0.45, H = 0.45. High: O = 1.5, r = 0.3, M = 1.0,
Rc = 3.5, S = 0.8, H = 0.6. Middle: O = 1.25, r = 0.4, M = 0.6, Rc = 2.5, S = 0.6, H = 0.5. V and W stay out of every sum
until a slice measures them.

**The model restates the measurements; it does not check them.** With MSAA added it gives A1 and A2 back (7.2 + 6.1 +
6.6 + 10.5 + 1.85 = 32.25; without the 10.5: 21.75). That match holds by construction: F, the 6.1, A, B and MSAA are all
differences of A1–A5. No measured frame outside those inputs tests the model yet. Nobody drew the stand at fewer than 49 %
of the pixels, so the s² rule below that is untested. **M-2 is the model's first test:** its readings at 0.75, 0.67, 0.5
and 0.33 check the s² rule (§11, T0). Example: no run has drawn the stand at a quarter of the pixels yet.

**The limits of the model: its error goes both ways.**
- It may read HIGH. It assumes that each render-size piece is linear in the pixels and that the pieces add, and the GPU
  overlaps passes (§2.3 item 2). The traffic floors of S, H, O and bloom drop if the GPU's lossless compression applies.
- It may read LOW. V and W stay out of the sums. Every cell is a MEDIAN at a STILL stand, and the gate is the p99 on the
  MOVING legs (B-FRAME (2)); M-5 measures that spread.
- Only the clock decides: T0's M-2 measures rows 1–6 together at each scale, M-2b prices W, T1 prices V, and T2 measures
  Rc, S and H.

### 10.2 The table (GPU ms a frame, on foot at the stand, full screen)

| # | Piece | Grows with | Today (A1) | No MSAA (A2) | `s` = 1 | 0.75 | 0.67 | **0.5** | 0.33 (the floor) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | F: the triangles and the fixed passes | nothing | 7.2 | 7.2 | 7.2 | 7.2 | 7.2 | 7.2 | 7.2 |
| 1V | V: the motion's vertex output | the drawn vertices | 0 | 0 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| 2 | P_r: the ground's pixel work at the render size | s² | 4.6–5.1 | 4.6–5.1 | 4.6–5.1 | 2.6–2.9 | 2.1–2.3 | 1.2–1.3 | 0.5–0.6 |
| 3 | O: the post passes at the display size | display | 1.0–1.5 | 1.0–1.5 | 1.0–1.5 | 1.0–1.5 | 1.0–1.5 | 1.0–1.5 | 1.0–1.5 |
| 4 | A: the raymarched air | s² | 6.6 | 6.6 | 6.6 | 3.7 | 3.0 | 1.7 | 0.7 |
| 5 | MSAA: the ground's samples 7.4, the air per sample 3.1 | — | 10.5 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | B: bloom and shadows | mixed | 1.85 (8.8 as marginals under MSAA) | 1.85 | 1.85 | 1.4–1.6 | 1.3–1.5 | 1.2–1.4 | 1.0–1.4 |
| 7 | M: the motion's pixel side, the surface words, the reactive copy | mixed | 0 | 0 | 0.2–1.0 | 0.2–1.0 | 0.2–1.0 | 0.2–1.0 | 0.2–1.0 |
| 8 | Rc: the reconstruction | display | 0 | 0 | 1.5–3.5 | 1.5–3.5 | 1.5–3.5 | 1.5–3.5 | 1.5–3.5 |
| 9 | S: the points pass | display | — | — | 0.45–0.8 | 0.45–0.8 | 0.45–0.8 | 0.45–0.8 | 0.45–0.8 |
| 9H | H: the sharpen (D7) | display | — | — | 0.45–0.6 | 0.45–0.6 | 0.45–0.6 | 0.45–0.6 | 0.45–0.6 |
| 9W | W: G2's full-size targets outside the render rectangle | display × (1 − s²) | — | — | 0 | ≤ 0.28 | ≤ 0.35 | ≤ 0.47 | ≤ 0.56 |
| | **TOTAL of rows 1–9H (V and W not in it): the range (the middle)** | | **32.25 (MEASURED)** | **21.75 (MEASURED)** | 24.4–27.7 (26.0) | 18.8–22.5 (20.6) | 17.4–21.2 (19.2) | **14.9–18.8 (16.8)** | 13.1–17.2 (15.1) |
| 10 | T5's cheaper air: 40–60 % of row 4 | s² | | | −2.6 to −4.0 | −1.5 to −2.2 | −1.2 to −1.8 | −0.7 to −1.0 | −0.3 to −0.4 |
| | **TOTAL with row 10: the range (the middle)** | | | | 20.4–25.0 (22.7) | 16.6–21.0 (18.8) | 15.6–20.0 (17.8) | **13.9–18.2 (16.0)** | 12.7–16.9 (14.7) |

Rows 1–6 in the A1 and A2 columns are MEASURED and DERIVED (O and P_r split a DERIVED 6.1 by an ESTIMATE); every other
cell is an ESTIMATE, and rows 1V and 9W are UNMEASURED. Row 9W's cells are ceilings for route G2 (the product route):
(1 − s²) × 172 MB at 273 GB/s, if every untouched tile stores once.

### 10.3 What the table says

1. **The reconstruction at full size never reaches 16.7 ms** (24–28 ms; 20–25 ms with the cheaper air). `s` = 1 is a
   look slice (T2), never the budget.
2. **At 0.67 the frame is 17–21 ms.** With the cheaper air the middle is 17.8 ms: it misses 16.7 ms.
3. **At 0.5 the frame is 15–19 ms, 16.8 ms in the middle. With the cheaper air the middle is 16.0 ms: about 0.7 ms of
   margin, and NOT assured.**
   - The top of the range (18.2 ms with the cheaper air) misses 16.7 ms.
   - The middle is the midpoint of ASSUMED inputs (O, r, M, Rc, S, H). It is neither a measurement nor an expectation.
   - V and W are not in it.
   - The margin sits on a MEDIAN of 12 samples at a STILL stand. The step's gate is the p99 on the MOVING legs (B-FRAME
     (2): the p99 ≤ 16.7 ms, at most one frame in 1 000 over it), and M-5 measures that spread. A frame of 16.9 ms shows
     for 25 ms on the 120 Hz display.
   - So the design proposes 0.5 (D8) on three conditions: M-2's reading at 0.5 (§11, T0), T2's own-cost gate for Rc, and
     M-5's p99. The cheaper air (T5) is part of the plan at 0.5, not an option: without it the middle is 16.8 ms.
4. **The floor (0.33) gives 13–17 ms** (15.1 ms in the middle; 14.7 ms with the cheaper air: about 2 ms of margin). It is
   the fallback if 0.5 misses, and the valve of dynamic resolution (D16): from 0.5 to 0.33 it frees about 1.3–1.7 ms in the
   middle (ESTIMATE).
5. **The display's fixed price.** F (7.2 ms) plus the display-size work (O + Rc + S + H + bloom's fixed share: 4.3–7.7 ms)
   stay at every scale: 11.5–14.9 ms. So the 11 ms goal (§10.4) needs the geometry step AND a cheaper display side. Each
   full-screen pass over the 8-byte display picture moves about 120 MB, about 0.44 ms, so the display side gets cheaper with
   FEWER passes (for example, the sharpen inside the tone map's pass: ESTIMATE, unbuilt). On a 1080p average machine the
   display-size work is about a quarter of this (ESTIMATE, by pixels).
6. **The biggest unknowns are Rc, V and W.** T2 measures Rc first; over 3.5 ms, the budget gets a new plan with the numbers
   before T3. T1 prices V. M-2b prices W before T3 builds G2.

**Example.** At `s` = 0.5 the pilot on foot at the stand pays about 7 ms for the ground's geometry and the fixed passes,
about 6 ms for the full-size passes (the reconstruction, the points pass, the sharpen, bloom, the tone map), about 3 ms for
the ground's light and the air on 1.9 million render pixels, and about 0.6 ms for the motion (the middle, before the
cheaper air, and before V and W).

### 10.4 The frame with the game still to come (a proposal for D15)

| Line | ms | Basis |
|---|---|---|
| Today's content at `s` = 0.5 with the cheaper air (the middle) | 16.0 | §10.2; V and W not yet in it |
| The geometry step and a cheaper display side (their own design: occlusion culling, the vertex bytes, the fixed passes, fewer full-screen passes) | −5.8 | the NEED to reach the line below, not a priced gain. F is 7.2 ms, so the geometry step alone would have to remove about four fifths of it; a cheaper display side (4.3–7.7 ms today) must share the cut |
| **Today's content after them** | **10.2** | draft B's proposal; the one-volume design's B-FRAME (3) target is 11.0 ms, reported and not yet a gate (`one_volume_design.md:1812`) |
| Constructions, hulls, stations, players | 1.5 | proposal |
| Clouds | 2.0 | Nubis³ 2.1 ms (PS5, 60 Hz); Frostbite 1.6 ms at half size (R8, R13) |
| The water fill and the ocean | 1.0 | proposal |
| Weather and particles | 0.8 | proposal |
| Screen-space light | 0.7 | proposal |
| The spike margin | 0.5 | a frame over 16.7 ms shows for 25 ms |
| **The total** | **16.7** | |

---

## 11. The slices, in build order, with their gates

**The order's reasons.**
- The clock first: the owner's own order ("Clock, then the looks").
- The instruments before any change: the "before" row must read today's picture.
- The motion before any history: a history with a wrong motion ghosts.
- The owner's look at the new edge, AT THE SIZES THAT SHIP (1.0, 0.67 and 0.5), before the largest engine change. So T2
  builds the up-scale on route G3, which needs no engine change. If the owner refuses the edge at every size that can meet
  the frame, the design does not need the deferred pass, and MSAA can stay (a deferred camera forces MSAA off, F22). A
  look at `s` = 1 alone would not protect this: only 0.5 or lower meets the frame (§10.3), and the owner would first see
  the up-scaled edge after T3 built the deferred pass.
- The one geometry pass before the budget: the budget must read the final route.
- The looks' own costs last.

The window opens only with the owner's word at that moment. Every measurement runs one job at a time on an idle machine
(load under 1.5), 12 samples a mode, the median, on the GPU clock.

**The switches** below are measurement switches, never product settings (the precedent: DEFERRED "THE SPEED STEP" item 4).
T8 ends with ONE path, by the owner's law: one world, and no second implementation chosen by a flag. The switches: MSAA
against temporal; the render scale; the order of the stars; the geometry route; the reconstruction (ours, Bevy's TAA,
MetalFX); the sharpen strength; the shadow filter; each T5 step.

### T0 — MEASURE AND INSTRUMENT. No picture change. No product change.

**Builds.**
1. **The per-pass clock:** the clock's own stamp passes between the camera's graph nodes (the proved two-pass method,
   `gpu_clock.rs:11-21`). A stamp pass after a node loads and stores the texture that the node wrote, so the GPU cannot
   take the stamp early. Each such pass costs one load and one store of its texture (about 0.3 ms at full screen, as the
   clock's own end pass does), so a per-pass run is a measurement of its own and never the frame's total. On a GPU that
   overlaps passes, the per-pass numbers are approximate: the price by absence stays the judge.
2. **The instruments**, each with a control that it must read correctly (a synthetic two-level flicker, a 3-pixel trail, a
   missing star, a one-frame step):
   - the edge meter (G-EDGE): the blended share and the number of blend levels of the horizon and limb pixels (step 19's
     measure made an instrument: 0.41 against a 0.43 reference, commit `4c155d0`);
   - the flicker meter (G-FLICKER): each pixel's change from frame to frame over 64 frames at a still stand, by region
     (the sky, the ground, the edges, the stars, the shadow edges);
   - the star census (G-STARS): the stars visible in each of 64 frames and their summed light, still and turning;
   - the ghost rig (G-GHOST): the trail behind a moving hull box and a dot, in display pixels;
   - the sharpness meter (G-SHARP): the fine-detail energy of the ground at each stand;
   - **the picture census (G-PICTURE-HOLES), new:** a run with a sentinel clear colour, the air pass off and the star
     cloud off. It counts the pixels that the probe calls ground (with all four neighbours ground in the probe, so a
     silhouette's edge does not count) and that show the sentinel in the PICTURE. It sees what the probe cannot: a hole of
     the picture camera's own depth test (F31). Its control: one chunk drawn in the prepass one depth step nearer than in
     the main pass must count holes.
3. **The measurement switches:** the render scale (the render-world override with the engine's plain blit: the picture sits
   in a corner, and T0 wants only the number); the engine's depth-and-motion prepass with MSAA off, its outputs unread; the
   star cloud off; a small light probe; the sentinel clear colour; a measurement pass that clears and stores G2's target
   set (M-2b).
4. **The adapter report:** the read-write storage tier, the texture atomics, the f16 feature and the shader language
   version, as wgpu reports them on this macOS.

**Measurements.** Each one can fail.
- **M-1:** every pass with all the looks on and MSAA off. It explains the overlap (bloom 4.6 ms under MSAA against ≤ 1.85
  ms with the shadows without it) and splits O.
- **M-2:** the frame at `s` = 1, 0.75, 0.67, 0.5 and 0.33, with MSAA off and the raymarched air, with the per-pass clock
  under the override (so each pass shows the part of its load and store that stays at the display size). It is the
  MODEL'S FIRST TEST (§10.1): the model predicts rows 1–6 at 12.3–12.9 ms at 0.5 (the middle 12.6), and the readings at
  0.75, 0.67, 0.5 and 0.33 check its s² rule. **It can refute this design at 0.5.** The add-on M + Rc + S + H is 2.6 ms low
  and 4.2 ms in the middle, and the cheaper air takes 0.7–1.0 ms (the middle 0.8):
  - over about 12.5 ms, the middle add-on misses 16.7 ms, so T5's cheaper air is a must (the model's middle already
    predicts this);
  - over about 13.3 ms, the middle misses 16.7 ms even with the cheaper air, and D8 moves toward the floor (0.33);
  - over about 15.1 ms, even the lowest add-on with the best cheaper air misses, and §10 gets a new plan before T1.
  Each measured millisecond of V (T1) and W (M-2b) lowers every threshold by one millisecond.
- **M-2b:** G2's full-size targets, priced BEFORE T3 builds them: the measurement pass clears and stores four display-size
  textures of G2's formats (Rgba32Uint, R8Uint, Depth16Unorm, Rg16Float: 23 bytes a pixel), with the viewport at the render
  rectangle, at 1.0 and 0.5. It prices row W of §10.2 (its ceiling: about 0.47 ms at 0.5). If W reads high, the cure is an
  engine change: the main passes draw into render-size textures.
- **M-3:** the price of route G3 (the same chunks drawn a second time into depth and motion), and its holes ON THE PICTURE
  CAMERA: the picture census (item 2) with the prepass on, after the census reads its control. The probe's own census
  cannot see these holes (F31). M-3 decides T1's route (§4.5) only after its census reads its control; until then G3's
  holes are UNMEASURED.
- **M-4:** the price of the stars, the light probe and the air's tables.
- **M-5:** the p99 of the moving legs with MSAA off, at `s` = 1 and 0.5: the spread between the still stand's median and the
  legs' p99 that D8's margin must cover.
- **M-6: THE "BEFORE" ROWS.** Every instrument of item 2 on today's MSAA picture, on the picture set (§14). Every later
  slice compares with these rows.

**Gate.** Every instrument reads its control, the picture census too. The per-pass sum stands within the frame's span. The
numbers go into the registry entry (DEFERRED "THE SPEED STEP"), each with its mode.

### T1 — THE MOTION. No picture change. Behind a switch; MSAA stays the default.

**Builds.**
1. The Tier-A rule (§5): the realm's step, the previous placed vertex, the signed age.
2. The band table's previous rows, last frame's clock and the realm's step in its header; the three shaders on one layout.
3. The ground's prepass fragment writes the motion (route G3 if M-3's picture census counts no holes after it reads its
   control; else route G2 at once). The vertex stage outputs the smallest exact previous place (three components, §5.2).
4. The marker-to-body handover keeps ONE entity. A new dot carries a spawn stamp and marks its first frame new.
5. A motion sentinel: the motion target clears to a value that no surface writes.
6. A readback of the motion in the capture.
7. The probe's MOVING MARK: the probe marks each pixel whose surface moves against its body in this frame (inside a band's
   morph or sink, or a ramp weight that changed since the last frame), for example in its free fourth byte.

**Why the gate changes shape.** The harness's reprojection (the pop detector's method) keeps each ground point STILL in the
body's frame (F32). A rising chunk (the ramp) and a morphing vertex move against the body. In the bands and the ramps the
exact rule therefore differs from that reference by the surface's own motion (§5.4: the ramp moves ≥ ~0.3 px a frame),
and today's formula AGREES with the reference there, because it makes the same "still ground" assumption (F5). A pixel gate
there would fail the correct rule and pass the defect. The reference's distance is also a whole number of cells, so its own
miss reaches 0.25 px at the pop detector's near limit (F32), above any 0.05 px tolerance. Example: a rung-9 chunk lands
late over the belt and rises for half a second; its correct motion reads about 0.3 px off the reference in each frame of
the rise, and today's formula reads 0.

**Gates.**
- **G-MOTION-COVERAGE:** 0 pixels that hold a depth and keep the sentinel.
- **G-MOTION-TRUTH, on the still ground:** the residual against the harness's own reprojection (each pixel's probe distance
  through both stamped cameras, as the pop detector reprojects), ONLY on the ground that the probe does not mark as moving,
  and only where the reference's own miss stays under 0.02 px (the pop detector's near limit taken at 0.02 px in place of
  0.25 px). PROPOSED: ≤ 0.05 render pixel on 99.9 % of those pixels, on the walk, a turn, the 240 and 528 m/s legs and a
  boarding.
- **G-MOTION-TRUTH, in the bands and the ramps:** the Tier-A unit tests judge the rule exactly there (the previous placed
  vertex at last frame's band row and clock; the landing identity; a band slide; a swap; the signed age). The shaders
  transliterate that rule step for step.
- **THE CONTROLS** (each must fail):
  1. today's formula (its previous place, with a fragment that writes its motion) FAILS the pixel gate on the new-chunk
     rows: in a chunk's first frame the engine gives it no previous transform (F7), and that ground stands still, so the
     reference is exact there;
  2. today's formula, as a Tier-A fixture, FAILS the Tier-A ramp and band tests;
  3. the correct rule with its motion shifted by 0.2 render pixel FAILS the pixel gate.
- The picture stays within one level of the exact pictures, and both hole censuses (the probe's and the picture's) stay
  unchanged.
- The clock: the price of the motion, with the vertex output on and off (row V of §10.2).

### T2 — THE RECONSTRUCTION AND ITS UP-SCALE (`s` = 1, 0.67 and 0.5), on route G3. THE FIRST PICTURE CHANGE. Behind a switch.

A LOOK slice. Its frame on route G3 is not the budget (ESTIMATE: 24–33 ms at `s` = 1 and 15–24 ms at 0.5; the G2 column of
§10.2 plus G3's second pass at M-3's price, which the mixed split puts at up to ~5 ms).

**Builds.**
1. The reconstruction (§4.2) with its up-scale, with MSAA off and the air per pixel.
2. The render scale: the override in the render world (§4.4); ⌈8 / s²⌉ jitter phases; the floor law in Tier-A; the texture
   bias (no texture today, F11). Route G3 draws the ground forward, so each ground pixel takes its world place from the
   vertex stage and needs no seventh patch. At `s` < 1 only the cluster lookup reads a wrong cluster (F27), and the scene
   holds no lamp today (only the sun), so nothing shows it.
3. The jitter's clock; the reset list and its counter (G-RESET).
4. The points pass: the stars and the sun disc after the reconstruction; the sky share; the clear alpha 0; the display
   alpha 1 again (§7.7, change two); the vendored air pass stops adding the sun on sky pixels (§4.8).
5. The sky's motion (the air's rotation step and its fallbacks).
6. The reactive mask from a colour copy before the transparent pass.
7. The capture on the same path: the display picture = the render grid ÷ `s` (§12.1); the alpha check (G-ALPHA).
8. OPTIONAL T2a: Bevy's TAA as a first price only, never shipped.

**Gates.**
- **★ THE RECONSTRUCTION'S OWN COST, FIRST:** ≤ 2.5 ms at 7.5 M display pixels. Over 3.5 ms, §10 gets a new plan, with
  the numbers, before T3. The clock also prices S and H (rows 9 and 9H of §10.2).
- Every T0 instrument against its "before" row, with the targets of §12.4, at each scale.
- The pop detector on the render picture and on the display picture, each with a known-pop control.
- The sky judge with a known-step control, and its lag reading on the 528 m/s leg (the air's parallax, §6.3).
- G-RESET = 0 on the player's path; G-CROSSING (a boarding, a departure, the V switch); G-STARS; G-DETERMINISM;
  G-NO-STALL; G-ALPHA.

**The owner's look:** D5 (the edge) at 1.0, 0.67 and 0.5, still and turning; D6 (the sky's points); a first look at D7
(the sharpen). The final scale (D8) and the response in motion (D9) wait for T4 at the frame rate that ships, because the
frame rate changes the motion per frame.

### T3 — THE ONE GEOMETRY PASS (route G2). After D5 = yes at a size that can meet the frame (0.5 or lower). No picture change beyond V18's tolerance.

**Builds.** The ground's deferred stages (the vertex rule of T1; the fragment: the far-edge discard, the surface words,
the motion); the vendored patches five (the light pass's rectangle), six (the masked deferred skip) and seven (the render
size's viewport in the deferred light's pixel-to-place rule and in the cluster lookup, F27), each recorded in
`vendor/bevy_pbr/PATCH.md`; the new pipelines warmed at the boot.

**Gates.**
- **G-DEFERRED-PARITY:** deferred against forward, both without MSAA, at EVERY scale of T2 (1.0, 0.67, 0.5): every content
  pixel within one level at every stand (V18). Without patch seven the parity fails at `s` < 1, so this gate also holds
  patch seven's control.
- The per-pass clock: the one pass plus the light against the forward pass without MSAA, against M-3, and row W against
  M-2b. The main pass draws no ground (patch six works).
- G-MOTION-COVERAGE and G-MOTION-TRUTH again; 0 holes in both censuses; G-NO-STALL.
- If parity or cost fails: D4 again (route G1, a second vendored crate).

### T4 — THE BUDGET AND THE SCALE (after T3).

**Builds.** The sharpen at the owner's strength (D7); the ruled scale as the product's one path. The up-scale came at T2,
and the one pass at T3.

**Gates.**
- **G-BUDGET** at 0.75, 0.67 and 0.5 on the stands and the legs: the p50 at or under the ruled line (D15; the design sets
  the line only after M-2 and T2 read, because the model's middle moved by about 1.2 ms in one review), the p99 ≤ 16.7 ms,
  at most one frame in 1 000 over 16.7 ms.
- G-DEFERRED-PARITY again at the ruled scale (the deferred picture against the forward picture at the same `s`).
- G-SHARP; every T2 gate at each scale; the moving-eye gate green.

**The owner's look:** D8 (the scale), D9 (the response, at 60 frames a second), D7 (the sharpen).

### T5 — THE AIR AT ITS OWN COST, cheapest risk first.

1. **EXACT:** the transmittance and multiscattering tables built once per body, again at a handover. Gate: 0 pixels change.
2. **The light probe at its own pace** (one cube face a frame). Gate: ≤ 1 level.
3. **The cheaper march** (§7.4): the open sky at half the render size; the air over the ground in 2 × 2 blocks with one
   partial step per pixel. FIRST the block's direction error (new, UNMEASURED, F33) at the sun's glow and at the horizon
   (the dusk, aloft and orbit stands). Gate: G-AIR ≤ 1 level against the per-pixel march at every stand (the ground, the
   hill, aloft, dusk, orbit, far); the halo count at the ridges; the sky judge on the descent leg.
4. **The air volume** (§7.5), built beside step 3 and measured on the same gates.

**Overall gate:** the air ≤ 1.0 ms at the stand at the ruled `s`. **The owner's look:** D10.

### T6 — SHADOWS, BLOOM AND THE POST CHAIN.

**Builds.** Bloom timed per pass, and a cure where a pass costs more than its bytes (a hypothesis to test: 16 small
dependent passes stall the GPU between them); the half-size bloom input if needed; the Temporal shadow filter as a switch;
the crawl measured on the walk.

**Gates.** Bloom ≤ 0.8 ms, or D11; G-FLICKER on the shadowed ground of the walk. **The owner's look:** D11.

### T7 — THE GATES FROZEN AGAIN (after D5, D8 and D13).

The exact pictures frozen again from converged stills at the display size; the new floors measured; every new gate in the
suite.

### T8 — ONE PATH (after D14).

- The MSAA path goes: the per-sample air, the second air pass and its textures (PATCH.md changes two and four).
- PATCH.md states the change; the upstream pull request shrinks.
- The registry flips its entries, among them D-TERRAIN-6's "TEMPORAL ANTI-ALIAS FOR THE FLIGHT". Its text still names SMAA
  (`DEFERRED.md:8640-8643`), but the code runs MSAA with the per-sample air (`lib.rs:1461-1466`).

### T9 — ACCELERATORS BEHIND THE SEAM (each only on D12).

MetalFX on the same inputs, on its own encoder, made at the start and at a display-size change; the same gates and the
owner's look side by side. DLSS (Bevy's node) when an RTX machine joins the test set. An accelerator is never the gates'
reference.

### After this design

- **Dynamic resolution (D16):** a scale between the floor and a ruled maximum, driven by the GPU clock, changed slowly.
- **The geometry step** (its own design): occlusion culling on the one pass's depth, the vertex bytes, the far-edge discard
  against Apple's hidden-surface removal, the ladder's detail for each machine.

---

## 12. The effect on the headless gates

### 12.1 The capture keeps its grid

- Today the capture draws 1 284 × 720, the reference view, with MSAA 4× on its picture camera and MSAA off on its probe
  (F25).
- Under the path the capture's RENDER picture stays 1 284 × 720. Its DISPLAY picture grows to 1 284 ÷ `s` × 720 ÷ `s`:
  2 568 × 1 440 at 0.5. At `s` = 1 nothing changes.
- So every gate that reads the render picture or the probe keeps its reference pixel, and the display picture tests exactly
  the ratio that ships ("test exactly production").
- The capture's render pixel is then the coarsest lawful one (it stands AT the floor law), while the window's render pixel
  at 0.5 is two thirds of it. So the capture tests the harder case of the two.
- The probe takes the SAME jitter as the picture. The probe already copies the picture camera's transform and projection
  every frame (F25); the jitter is one more thing to copy. So the probe and the render picture sample the same points.
- The capture's readback grows by 1/s² (UNMEASURED cost; today the capture maps both images every frame with a full wait:
  research).
- The capture checks the display picture's alpha: every pixel reads 255 (G-ALPHA, §7.7 change two). The saved PNG keeps
  all four channels (F30), so a sky pixel with alpha 0 would show the viewer's background in the owner's pictures.

### 12.2 A converged still

1. The band settles.
2. The capture resets the history once (a tool's reset, counted apart from the player's path).
3. It waits whole jitter cycles until every pixel's count reaches the floor. At 32 phases and a 1/64 floor that is about two
   cycles, about 64 frames (ESTIMATE; T2 measures it). Today's pre-roll is 8 frames (`lib.rs:168`).
4. It takes the frame at the first phase of a cycle.

**The control:** two runs of one stand compare within one level on the content pixels, beside a control run (two runs of
one binary already differ by up to 220 pixels and 22 levels where a far body's artifact lands late:
`DEFERRED.md:8936-8939`). The jitter's clock restarts at the reset, so both runs see the same phases.

### 12.3 Gate by gate

| Gate | What it reads today | What it reads under the path | How it keeps its meaning | The control that must fail it |
|---|---|---|---|---|
| The pop detector (the seam law) | picture and probe pairs; the floor at 99.9 % of the still pixels; a quarter-pixel near limit (`pop.rs:52-61`, re-read) | (1) the RENDER picture before the history, tone-mapped for the gate, with the probe at the same jitter; the stamp carries each frame's jitter into the reprojection; (2) a new reading on the DISPLAY picture | (1) is the seam law, unchanged: the history can never hide a pop from it; (2) is what the player sees, with its own floor. The jitter moves the edges between frames, so the gate measures both floors again. | a known pop (one rung's chunks forced to enter whole) must fail both |
| The sky judge | the mean sky colour per pair against the air's law | the display picture, plus a lag reading against a converged still | accumulation can only smooth a step; the lag reading catches a sky that trails | a forced one-frame sky step must fail it |
| The hole instrument | the probe, unjittered | the probe at the picture's jitter, counted over whole jitter cycles | a crack that shows in some phases counts (under accumulation it becomes a faint line). It cannot see a hole of the picture camera's own depth test (F31); the picture census reads those. | a chunk shifted by one cell must count in the phases that see it |
| The picture census (new at T0; G-PICTURE-HOLES) | — | the pixels that the probe calls ground and that show the sentinel clear colour in the PICTURE, with the air pass and the stars off | it reads the picture camera itself, so it sees route G3's holes (the main pass's depth test against the prepass depth) | one chunk drawn in the prepass one depth step nearer than in the main pass must count holes |
| The exact pictures (≤ 1 level, V18; `terrain_pictures.rs:222-226`, re-read) | frozen under MSAA | converged stills at the display size, frozen again once (D13) | G-DETERMINISM | a one-level change at one pixel must show |
| look_pixels, the star, warp and box gates | projected rectangles; 3-pixel markers | converged stills at the display size; the pixel laws in display pixels (F24) | the stars are today's sprites through the points pass | — |
| The ruler gate | the picture and the stamped camera | the render picture and the stamped camera with its jitter | the model does not change | — |
| The moving-eye gate | the legs, the pop detector, the sky judge | the same flight; its readings on both pictures | as above | as above |

### 12.4 The new gates and their proposed targets

The owner rules the targets. T0 first measures today's values.

| Gate | Slice | Proposed target |
|---|---|---|
| G-EDGE | T0 before, T2 on | the blended share of the horizon and limb pixels ≥ today's (0.41 at step 19), still and turning |
| G-FLICKER | T0 before, T2 on | at a still stand after convergence: 99.9 % of the content pixels change by ≤ 1 level a frame; the edges by ≤ 2 |
| G-STARS | T0 before, T2 on | every star of today's picture present in 64 of 64 frames, still and turning; the summed light within 1 % |
| G-GHOST | T0 before, T2 on | a trail ≤ 1 display pixel behind a hull box and a dot at 240 and 528 m/s (D9 sets the bound) |
| G-SHARP | T0 before, T2 on (the up-scale), T4 | the owner's reading at D5 and D8 |
| G-PICTURE-HOLES | T0 (M-3), T1, T3 | 0 pixels on the picture camera; its control (a chunk one depth step nearer in the prepass) counts holes |
| G-MOTION-COVERAGE | T1 | 0 pixels with a depth and the sentinel |
| G-MOTION-TRUTH | T1 | on the ground that the probe does not mark as moving, where the reference's own miss is under 0.02 px: ≤ 0.05 render pixel on 99.9 % of the pixels; in the bands and the ramps: the Tier-A tests, exactly; today's formula fails both (§11, T1) |
| G-ALPHA | T2 | every pixel of the capture's display picture has alpha 255 |
| G-RESET | T2 | 0 resets on the player's path on every leg |
| G-CROSSING | T2 | at a boarding, a departure and a V switch: the display picture's change in the swap frame no larger than in the frames beside it; the rejected share inside the leg's spread |
| G-DETERMINISM | T2 | two runs of one stand within one level, beside a control run |
| G-NO-STALL | T2 | the worst frame over a start, a full-screen switch, a scale change and a boarding ≤ 16.7 ms after the pre-roll; 0 pipelines compiled after the pre-roll |
| G-DEFERRED-PARITY | T3 | every content pixel within one level of the forward picture (V18), at every scale of T2 |
| G-BUDGET | T4 | the p50 at or under the ruled line (set after M-2 and T2 read); the p99 ≤ 16.7 ms; at most one frame in 1 000 over 16.7 ms |
| G-AIR | T5 | ≤ 1 level against the per-pixel march at every stand |

### 12.5 HR5

The Tier-A parts stay at 100 % region and branch: the realm's step; the previous placed vertex with the signed age; the
jitter sequence and its clock; the floor law; the reset decision; the sky-share arithmetic; the gate math (the edge meter,
the flicker meter, the star census, the ghost rig, the picture census, the probe's moving mark).

---

## 13. Risks

| # | Risk | What the player sees | How it is caught | The cure | Slice |
|---|---|---|---|---|---|
| 1 | A drawn thing with no motion or a wrong motion (the morph, the ramp, a new chunk, a swap) | a smear or a ghost | G-MOTION-COVERAGE; G-MOTION-TRUTH on the still ground (the harness's reference is exact only there, F32); the Tier-A tests in the bands and the ramps | the realm's step; last frame's rows and clock | T1 |
| 2 | The ramp's wrapping clock read at last frame's time (F6) | a landing chunk smears its whole rise in one frame | the Tier-A test "a landing chunk's previous place equals its stand-in's" | the signed age | T1 |
| 3 | A second geometry pass eats the gain (G3) | a frame slower by M-3's price (the mixed split suggests up to ~5 ms, F34) | M-3; D4 | G2 as the product route | T0, T3 |
| 4 | Holes from the prepass depth against the main pass's depth (G3) | a speck of sky through the ground | the picture census (M-3, G-PICTURE-HOLES) after it reads its control; the probe's census cannot see these holes (F31) | G2 | T0, T1 |
| 5 | The surface words' packing moves pixels (G2) | steps in the shading | G-DEFERRED-PARITY (≤ 1 level) | route G1 (D4) | T3 |
| 6 | A masked deferred ground draws twice (F8) | a dearer frame, no picture change | the per-pass clock; the draw count of the main pass | patch six | T3 |
| 7 | Stars, a far sun or far dots blink or vanish | a thinner, blinking sky | G-STARS | the points pass (§7.7) | T2 |
| 8 | Ghosts behind a fast hull or a translucent box | a trail | G-GHOST at 240 and 528 m/s | the depth test; the reactive mask; the dilated motion | T2, T4 |
| 9 | The horizon loses the edge that the owner chose | a soft or crawling horizon | G-EDGE; the owner's look (D5) | the floor of the weight; the sharpen (D7) | T2 |
| 10 | The history hides pops from the seam gate | a pop reads as a smear, and the gate goes blind | the pop detector reads the render picture; a known-pop control | §12.3 | T2 |
| 11 | A reset at a crossing | a burst of stairs over the whole picture (SL8) | G-RESET = 0; G-CROSSING | the policy of §6.1 | T2 |
| 12 | The swap frame holds no row for the air's body; or, on the fast leg, the rotation-only sky trails the air's glow (about 0.23 display pixels a frame at 528 m/s, §6.3) | the sky smears for one frame; a faint trail of the glow | G-CROSSING's sky reading; the sky judge's lag reading on the 528 m/s leg | the star cloud's rotation as the fallback; the air body's full step for the sky pixels | T2 |
| 13 | Captures differ from run to run | red picture gates | G-DETERMINISM, beside a control run | the jitter's own clock | T2, T7 |
| 14 | A pipeline compiles during a flight | a stalled frame, or a chunk missing for some frames | G-NO-STALL | the boot warms every pipeline | T2, T3 |
| 15 | The reconstruction costs more than its line | the budget fails | T2's first gate (≤ 2.5 ms; over 3.5 ms a new plan) | fewer passes, a lower scale, or MetalFX (D12) | T2 |
| 16 | The cheaper air differs from the per-pixel march; the 2 × 2 block's one direction differs from each pixel's own (UNMEASURED, F33) | a halo at the ridges; a changed sky; a blocky glow near the sun | the block's direction error first; G-AIR; the halo count; the sky judge | the next candidate (§7.4 against §7.5); D10 | T5 |
| 17 | Shadow edges crawl under the floating origin | a shimmer on the walk | G-FLICKER on shadowed ground | a snap grid fixed to the realm (an item of its own) | T6 |
| 18 | Thin far features shimmer (the far ridges; later antennas and rails) | a crawl at the far stands | G-FLICKER at 6 374 and 16 952 km | the continuous weight floor; FSR's locks later | T2, T4 |
| 19 | Another GPU lacks a feature | a broken picture on a player's machine | the portable rules (§8.4); the adapter report; a second machine's run (owed) | the portable subset | T0, later |
| 20 | The budget creeps as features arrive | frames over 16.7 ms | the per-pass clock; the budget lines (§10.4); the contract (§4.6) | each feature states its line before its first frame | always |
| 21 | The vendored code grows | a dearer upgrade to Bevy 0.19 | the upgrade gate's checklist; §4.8 lists every vendored change by slice | the reconstruction is project code; each vendored change recorded in `PATCH.md` and offered upstream: T2 the sun on sky pixels; T3 patches five, six and seven; T5 up to four air changes | T2, T3, T5 |
| 22 | An accelerator's picture drifts with an OS update | a look change with no code change | never the gates' reference | — | T9 |
| 23 | A HUD marker uses the jittered projection | markers shake by half a pixel | review; the rule of §4.3 | the display camera only | always |
| 24 | A new dot has no previous transform in the engine (F7) | a dot fades in over a few frames | G-GHOST | the spawn stamp marks its first frame new | T1, T2 |
| 25 | The stale row inside the grace (a seam that exists today) | the ground rides the eye, then jumps | measured apart (§6.4) | the terrain's own item | open |
| 26 | At `s` < 1 the deferred light rebuilds each pixel's world place from the display-size viewport (F27) | wrong shadows, a wrong view vector and a wrong air term on the ground; later wrong lamps and SSAO | G-DEFERRED-PARITY at every scale | patch seven | T3 |
| 27 | The display picture keeps the sky's alpha 0 (F30) | the owner's pictures show the viewer's background in place of the sky | G-ALPHA | the reconstruction writes the display alpha 1 (§7.7, change two) | T2 |
| 28 | The full-size targets store their untouched tiles at every scale (F29) | the frame at 0.5 up to about 0.5 ms dearer than the model's sums | M-2 under the override; M-2b before T3 | render-size main-pass textures (an engine change) | T0, T3 |
| 29 | The motion's vertex output costs memory traffic per vertex (§5.2) | a dearer frame at every scale | T1's clock, the output on and off | the smallest exact output (three components) | T1 |

---

## 14. The owner's decisions

**The picture set** (every look uses it):
- P1 the stand on foot, still;
- P2 P1 turning at 90° a second;
- P3 the walk;
- P4 the hull at 240 m/s, in the first and the third person;
- P5 the hull at 528 m/s;
- P6 6 374 km: the limb and the stars;
- P7 the dusk stand;
- P8 a boarding and a departure;
- P9 11 497 and 16 952 km;
- P10 the stand framed like the reference picture (ruling B6).

Each picture compares today's MSAA picture with the candidate, on the same world and tick, beside a control run (§12.2).
The instruments' numbers stand beside each picture. The window opens only with the owner's word at that moment.

### Asked now

**D1 — THE DIRECTION.**
- Options: (a) the temporal path (recommended); (b) keep MSAA and cut the looks' own costs; (c) MSAA at a render scale with
  a spatial up-scale; (d) one-pass MSAA: the main passes, the sky and the air in ONE render pass that resolves its samples
  in the tile and discards them (§3 reason 1).
- Price: (a) the largest build of the speed step, a look change (D5), new gates, and vendored changes by slice (§4.8: T2,
  T3, T5). (b) cannot reach 16.7 ms: even with no MSAA at all the frame is 21.75 ms (A2), and the looks' own cuts give a few
  milliseconds. (c) a softer picture with no temporal detail; the deferred light, SSAO, SSR and OIT stay blocked. (d) a
  second vendored crate (`bevy_core_pipeline`: the main pass's store operations and its pass structure) and the vendored
  air moved inside the main pass; wgpu 27 supports it (F26). It removes the samples' traffic, at most today's 10.5 ms, and
  its own cost is UNMEASURED. It cannot reach 16.7 ms at full screen alone (the frame without any MSAA is 21.75 ms), and it
  still blocks SSAO, SSR, OIT and the deferred light. With a render scale it becomes (c) with a cheaper edge.
- Why the design does not recommend (d): the temporal path removes the samples' whole cost, not only their traffic. It also
  pairs the render scale (the 5 ms that no full-size MSAA gives) with detail from the earlier frames, while MSAA at a
  render scale, (c) or (d), can only up-scale spatially.
- Picture test: none now; D5 decides the look.

**D2 — THE ORDER.**
- T0 → T9 as §11. The first picture change comes at T2, at 1.0, 0.67 and 0.5 (the up-scale on route G3). The deferred pass
  (T3) comes after the edge look (D5) at the sizes that ship.
- Price: T0 and T1 show nothing new on the screen (about the first third of the work). T2 grows by the up-scale, which
  moved there from T4.
- Picture test: none (T0 changes no pixel; T1 stays within one level).

**D3 — THE RECONSTRUCTION.**
- Options: U1–U7 (§8.1). Recommended: U3, our own, on FSR 2.2's published structure.
- Price: the largest effort and tuning time; MetalFX may beat it on this Mac at 0.5 (UNMEASURED).
- Picture test: T2 and T4 on the picture set; T9 side by side if D12 allows MetalFX.

**D4 — THE GEOMETRY ROUTE.**
- Options: G2, G3, G1 (§4.5). Recommended: G3 for T1–T2 if M-3's picture census shows no holes after it reads its
  control; G2 as the product after D5; G1 only as the fallback.
- Price: G2 needs three small patches in the vendored crate (five, six and seven; they die at the Bevy upgrade or go
  upstream), a lossy packing held at one level at every scale, and its full-size targets' traffic (row W, M-2b). The air's
  own vendored changes (T2, T5) come with the path, whatever the route (§4.8). G1 needs a second vendored crate.
- Picture test: G-DEFERRED-PARITY's difference images at every stand and at every scale.

### Asked at the slices, with pictures

**D5 — THE EDGE (T2).**
- The temporal edge replaces the per-sample MSAA edge that the owner chose on 2026-09-19 (`lib.rs:1461-1466`; PATCH.md, the
  second change).
- Price, the look: still, more blend levels than four samples give; in a fast turn, one or two fresh frames on a strip that
  just came into view; at 0.5, a softer edge that the earlier frames must sharpen.
- Price, the time (the whole of it, ESTIMATE, §10): MSAA's own 10.5 ms goes, but the motion, the reconstruction, the points
  pass and the sharpen come. At `s` = 1 the frame reads about 24–28 ms on route G2 and 24–33 ms on route G3 (T2's route),
  against 32.25 ms today: a saving of about 5–8 ms on G2 and 0–8 ms on G3. At 0.5 on route G3 the frame reads about
  15–24 ms. 60 frames a second comes only with the render scale (D8) and the one pass (T3).
- Picture test: P1, P2, P6 horizon and limb crops at 1:1, MSAA against temporal at `s` = 1, 0.67 and 0.5; the edge meter
  against 0.41.

**D6 — THE SKY'S POINTS (T2).**
- The sun and the stars after the reconstruction (recommended), or through it. A translucent star-system shell keeps
  dimming the stars behind it, as today.
- Price: the transmittance comes from the table (40 samples a ray), not from the pixel's march (16 steps), a change that
  the star census measures (target: the summed light within 1 %); a vendored change in the air pass (no sun on sky pixels);
  a pass of its own at the display size (row S of §10.2: 0.45–0.8 ms); the display alpha written 1 again (§7.7).
- Picture test: P6, P7, P2 turning; the star census over 64 frames.

**D7 — THE SHARPEN (T2, T4).**
- Options: none; the contrast-adaptive sharpen at 0.3; at 0.6.
- Price: halos at a high strength; about 0.45–0.6 ms (ESTIMATE: a full-screen pass of its own that reads and writes the
  8-byte picture, 120 MB, F29; the first draft's 0.2–0.3 ms needed the GPU's lossless compression).
- Picture test: P1 and P3, ridge crops at the belt.

**D8 — THE RENDER SCALE (T4).**
- Options: 0.5 (recommended ON THREE CONDITIONS: M-2's reading at 0.5, T2's own-cost gate for the reconstruction, and
  M-5's p99 on the moving legs), 0.67, 0.75; the floor is 0.33. If 0.5 misses, the fallback is a lower scale toward the
  floor, or dynamic resolution between them (D16).
- Price: §10 (0.5: 15–19 ms, the middle 16.8, and 16.0 with the cheaper air; 0.67: 17–21 ms, the middle 19.2, and 17.8
  with the cheaper air; 0.33: 13–17 ms, the middle 15.1, and 14.7 with the cheaper air; V and W not in them). A softer
  picture at 0.5. The time to one new sample per display pixel: 67 ms at 0.5, 37 ms at 0.67 (R6). The owner already saw
  the edge at 0.5 and 0.67 at T2 (D5); D8 rules the scale at the frame rate that ships.
- A note on the panel (ESTIMATE; the panel is UNVERIFIED): the display's numbers fit the built-in 16-inch panel (254 pixels
  an inch). At 50 cm one display pixel then subtends about 0.7 arc-minute, under the eye's limit of about 1 arc-minute. A
  render pixel subtends about 1.0 arc-minute at 0.67 and about 1.4 at 0.5, so at 0.5 the reconstruction must recover the
  rest from the earlier frames.
- Picture test: P1, P3, P6, P10 at 1.0, 0.67 and 0.5, still and turning, with the GPU milliseconds printed on each; the
  sharpness meter.

**D9 — THE RESPONSE (T4).**
- The floor of the new frame's weight (proposed: 1/64 still, 1/16 moving).
- Price: sharper stills against shorter trails.
- Picture test: P3, P4 and P5 over a ridge; the still stand; the ghost rig. Judged at 60 frames a second.

**D10 — THE AIR'S CUTS (T5).**
- Only for a cut that saves time but fails the one-level gate; and the choice between the cheaper march (§7.4) and the air
  volume (§7.5). The ranking waits for the 2 × 2 block's direction error (§7.4: new and UNMEASURED); the partial step's
  measurement covers one pixel's samples only.
- Price: the measured difference; the volume serves the fog later.
- Picture test: the ground, hill, aloft, dusk, orbit and far stands; a ridge in front of a far valley; the difference image.

**D11 — SHADOWS AND BLOOM (T6).**
- The shadow filter (Gaussian today, or Temporal); bloom's input from a half-size copy (only if T6 needs it).
- Price: a new shape of the penumbra; a small change of the glow.
- Picture test: P1 and P3 for the shadows, still and on the walk; P1 and P7 for bloom.

**D12 — THE ACCELERATORS (T9).**
- A new library (`objc2-metal-fx`) for MetalFX as a challenger; DLSS later for RTX machines (the game ships NVIDIA's DLL).
- Price: a dependency; a Mac-only picture that an OS update can move; a picture per vendor.
- Picture test: T9 side by side on the picture set.

**D13 — THE EXACT PICTURES (T7).**
- Freeze them again from the temporal path, from converged stills at the display size; V18's one level holds against the
  new pictures.
- Price: every frozen picture changes once.
- Picture test: the old picture against the new one at every stand.

**D14 — ONE PATH (T8).**
- MSAA, the per-sample air and the second air pass go, after D5.
- Price: no way back without a rebuild.
- Picture test: none.

**D15 — THE HEADROOM.**
- The budget lines for the game still to come (§10.4), and the geometry step next.
- Price: a step of its own, before or inside the ground step (B22 item 1.3) — the owner decides where.
- Picture test: none.

**D16 — DYNAMIC RESOLUTION (later).**
- A scale between the floor and a ruled maximum, with a rate limit, driven by the GPU clock.
- Price: the sharpness changes with the load.
- Picture test: P4 and P5 under load.

---

## 15. How this design joins the two drafts

| # | Question | Draft A (the picture first) | Draft B (the headroom first) | This design | Why |
|---|---|---|---|---|---|
| 1 | The geometry route | a forward pass with a motion target (a patched core pipeline) preferred; the prepass as the cheap control; deferred only if both fail | the deferred pass recommended; the prepass priced; the forward pass with motion is "the largest vendored diff" | **B**: deferred as the product; the prepass for the early slices if it shows no holes; A's route as the fallback | Both drafts want one geometry pass. Deferred needs three small patches in the crate that we already vendor (F8, F9, F27 re-read), while A's route needs a second vendored engine crate that the upgrade gate makes dear. A's concern (the packing) is exactly V18's tolerance, which a gate can hold (G-DEFERRED-PARITY). |
| 2 | The build order | instruments → motion → route → full-size look → scale → air → post → shadows → gates → MetalFX | measure → one pass → full-size look → scale → air → shadows and bloom → gates → one path → accelerators | **neither alone**: T0 (B's measurements + A's instruments) → T1 motion → T2 the look with its up-scale (1.0, 0.67, 0.5) on route G3 → T3 one pass → T4 the budget and the scale → … | A's instruments first, because the "before" row must read today's picture. NEW: the edge look, at the sizes that ship, before the largest engine change, because MSAA and the deferred light cannot run together (F22). If the owner refuses the edge, the design does not need the deferred work. A look at `s` = 1 alone would not give that protection (the review, §19). |
| 3 | The render scale | 0.67 favoured (quality) | 0.5 now, dynamic later | **B**: 0.5 proposed, on three conditions (M-2, T2's own cost, M-5's p99); 0.67 beside it in every picture; the floor (0.33) as the fallback | In §10's revised model, 0.5 reaches 16.7 ms only in the middle and only with the cheaper air (16.0 ms: about 0.7 ms of margin at a still stand's median, before V and W); 0.67 misses (17.8 ms). The floor law keeps every cell ≥ 1.5 render pixels at 0.5 (A's quality concern). |
| 4 | The view switch (V) | no reset | a reset (a cut) | **A** | With an exact motion the depth test rejects only the pixels that the switch uncovers. A reset throws away the valid history of the sky and the far ground and flashes stairs over the whole picture (SL8). |
| 5 | A render-scale change | a reset | not stated | **neither**: no reset; only a display-size change resets | The history lives at the display size. Dynamic resolution changes the scale every few frames. |
| 6 | The disocclusion test | the surface's own previous depth (A's forward route carries it in a free motion format) against last frame's stored depth | FSR 2's rebuilt previous depth (atomics in a buffer) | **B** | It works with every route and with the engine's motion format, and it needs no stored depth. |
| 7 | The reference algorithm | FSR 3.1's structure | FSR 2.2's structure | **B**, with A's rule: FSR 3.1's added stages only when a meter asks | Fewer passes, one published text, a shipped port (Godot). |
| 8 | The jitter's index | the frame count, with a capture procedure | the plug-in's own clock, which restarts at a reset | **B's clock + A's procedure** (§12.2) | Repeatable captures. |
| 9 | The blend weights | 1/n with a floor from 1/64 (still) to 1/16 (moving); motion never restarts the count | a confidence weighting; D7 | **A** | Bevy's rule (a restart on any motion, F2) makes a slowly turning planet shimmer at its limb. |
| 10 | The sky share | "the star window" = 1 − the picture's alpha, from the blend states | "the sky coverage", mechanism not stated | **A's mechanism**, named "the sky share" | Exact under today's blend states (F13, F14 re-read). NEW: the clear alpha must change to 0 (F16: it is 1 today), and the display picture's alpha must be 1 again after the reconstruction, with the sky share in a channel of its own (F30; §7.7 change two). |
| 11 | The capture | the display at 1 284 × 720 (the floor law forces `s` = 1 there) + a second size 1 926 × 1 080 | the render grid stays 1 284 × 720; the display grows to 1 284 ÷ `s` × 720 ÷ `s` | **B** | The gates then test the shipped ratio ("test exactly production"), and the probe keeps the reference pixel. |
| 12 | The probe's jitter | unjittered, plus a jittered run | the same jitter as the picture | **B**, counted over whole cycles | The pop detector needs the probe and the render picture sampled at the same points. Counting every phase keeps A's aim (a crack in any phase counts). |
| 13 | The cheaper air | deterministic cuts: the tables once, the light probe one face a frame, half-size open sky, 2 × 2 blocks with the partial step; no temporal march first | the air volume (1/8 grid, 64 exponential slices, a moving offset, accumulated); half size with offsets as the fallback | **A first, B beside it** in the same slice; B's volume kept as the weather step's structure | A's cuts carry no noise and no history. Their partial step reuses a rule measured at 0 pixels over one level, but only for one pixel's samples; the 2 × 2 block's direction error is new and UNMEASURED, and T5 measures it before the ranking (F33). The owner refused one lookup table already. The volume serves fog later. |
| 14 | The air's handover | the colour clip | the air layer's history restarts | **A now**; B's rule only if T5 builds the volume | It follows from row 13: the first cuts keep no air history. |
| 15 | A new dot's first frame | the realm's step alone; the handover carries the old transform | one frame with no motion, measured | **A's aim with a mechanism that needs no engine patch**: the handover keeps one entity; a new dot marks its first frame new | On the GPU path the engine gives a new mesh no previous transform (F7 re-read), so A's rule needs a patch or a mark. |
| 16 | The reconstruction's cost | 1.0–2.0 ms | 1.5–2.5 ms | **1.5–3.5 ms**, measured first in T2 | The published FSR 2 costs at 4K span 0.7–5.4 ms by GPU class (R5); the bandwidth floor alone is about 1.1 ms. |
| 17 | The instruments | first (T0), with "before" rows on today's picture | at the slices that need them | **A** | The "before" row must come from today's picture. |
| 18 | The dot floor | "a 3-pixel floor … ≥ 2 render pixels at `s` ≥ 0.67" | "must read the output height" | **neither: corrected** | The floor is a RADIUS of 3 display pixels (F24), and it reads the camera's viewport, which stays the display size under the render-world override. |
| 19 | The budget model | the pieces subtracted from today's frame | the pieces in three cost classes, lines per class | **B's classes**, with one formula that restates A1 and A2 (by construction, so not a check: §10.1) | One model: it states its assumptions, and it prices every full-screen pass by the same traffic rule. M-2's readings at 0.75, 0.67, 0.5 and 0.33 are its first check. |
| 20 | The headroom | the owner's decision | budget lines and the geometry step next | **B's lines as the proposal** (§10.4); the owner decides (D15) | — |
| 21 | MetalFX | a challenger at T9 | an accelerator at T8, never the gates' reference | **both**: a challenger on the same inputs, never the reference | — |

---

## 16. What this design does not cover

- **The rest of the speed step:** the main thread's walk (the release and the request: 1.3 + 1.4 ms a frame in the window;
  B-MAIN ≤ 1 ms) and the builders (B-RATE at 240 m/s): DEFERRED "THE SPEED STEP", OWED.
- **The geometry step:** occlusion culling, the vertex bytes, the far-edge discard against the hidden-surface removal. Its
  own design (D15). The same design needs a cheaper display side too (fewer full-screen passes, §10.3 item 5). ★ THE
  OWNER'S QUESTION (2026-09-29): *"should we also use cpu cores together with gpu? Will that help?"* — the pixel work of
  this design stays on the GPU: the CPU cannot start before the GPU has drawn the frame and the GPU cannot show it before
  the CPU ends, and the two share one memory bus on this machine. The CPU helps in its OWN lane, beside the GPU: it can
  decide what the GPU must NOT draw. A software occlusion test on the CPU (a coarse depth of the near hills, then each
  chunk's box against it) is one candidate for the geometry step, beside the engine's GPU occlusion culling (§17 items
  15 and 26).
- **One-pass MSAA** (D1 option (d)): priced as an option, not designed here.
- **The looks of the water fill, the clouds and the weather:** they enter through the contract (§4.6).
- **Frame generation** (MetalFX's frame interpolation): NOT proposed. It adds latency, needs macOS 26 and costs about
  5–7 ms on an M5 Max (REPORTED L).
- **120 frames a second** on the 120 Hz display (§17 item 13).

---

## 17. Open questions

1. The M4 Pro's feature set as wgpu reports it on this macOS: the read-write storage tier, the texture atomics, f16, the
   shader language version. The adapter report (T0) answers it.
2. The net cost of route G3 (the second geometry pass against the overdraw that it saves under the ground's discard), and
   its holes on the picture camera (M-3, with the picture census).
3. The reconstruction's own cost at 7.5 M display pixels on the M4 Pro (T2's first gate).
4. Why bloom reads 4.6 ms under MSAA when its full-size traffic is about 180 MB, and what it costs without MSAA (M-1). The
   same 180 MB bounds r (§10.1).
5. The display-size part O of the looks-off pixel work (M-1), and whether the GPU's lossless compression lowers the traffic
   floors of O, S, H and bloom.
6. Whether the swap frame holds the air body's row in both scenes. The pose and the picture ride two lanes with two clocks
   (`lib.rs:1900-1910`).
7. The effective near plane: DERIVED as 0.1 m by the renderer research, not measured. The depth test's precision at 1.6 m
   on foot depends on it.
8. Whether the shadow cascades crawl under the floating origin (T6).
9. How many milliseconds apart the "two instants in one frame" stand (§6.4).
10. The stale row inside the grace: a seam that exists today, UNMEASURED (§6.4).
11. How many frames a converged still needs, and so the picture gates' wall time (§12.2).
12. Whether the panel is the built-in 16-inch one (254 pixels an inch). It changes what 0.5 looks like (D8).
13. Whether 120 frames a second will ever be a target on this display. The budget would halve.
14. The average machine in numbers (F6: owed by the owner). It sets the scale's range and the geometry budget.
15. Whether Bevy's GPU occlusion culling works with the deferred prepass (the geometry step).
16. MetalFX's exact contract: the legal formats, the motion's sign and scale, the minimum macOS (only if D12).
17. How exact per-pass stamps are on a GPU that overlaps passes. The price by absence stays the judge.
18. How many new pipeline variants the deferred ground adds, and whether the boot warms all of them (G-NO-STALL).
19. Whether the translucent look-arm boxes stay drawn at all: they are stand-ins.
20. The deferred route with the probe's layer and the casters' layer: read in the code only, never run (T3).
21. Row V: the motion's vertex output through the tiled vertex buffer (T1's clock).
22. Row W: whether each pass stores the untouched tiles of its full-size targets at `s` < 1 (M-2, M-2b).
23. One-pass MSAA's own cost on this stack (D1 option (d)); nobody measured it.
24. The 2 × 2 block's direction error at the sun's glow and the horizon (T5, before D10's ranking).
25. Whether the rotation-only sky trails the air's glow on the 528 m/s leg (about 0.23 display pixels a frame, §6.3).
26. How much of the drawn ground no ray reaches on the three stands (hidden behind nearer hills or under the horizon),
    and so what an occlusion test could save — on the CPU (MEASURED 2026-09-28: the window's client used about half of ONE
    core while the GPU worked 32.25 ms a frame, so the CPU has spare time at a still stand) or on the GPU (item 15).

---

## 18. References

**This tree** (branch `game` at `ba2496b` plus the uncommitted speed-step work):
- `crates/client-render/src/gpu_clock.rs:1-35` — the clock's two-pass method.
- `crates/client-render/src/lib.rs:132` (the clear colour), `162-168` (the capture's size and pre-roll), `786-870` (the star
  material: the opaque phase, One + One, no depth write), `1244-1252` (the probe follows the picture camera), `1420-1475`
  (the window camera; the edge note at `1461-1466`), `1688-1735` (V and L), `1900-1956` (the scene swap), `2066-2115`
  (the camera's viewport; the dot floor), `2196-2216` (the star uniforms), `2481-2497` (the marker-to-body handover),
  `3244-3288` (the probe camera and the capture's picture camera).
- `crates/client-render/src/terrain.rs:109-121` (the shadow's reach), `2193-2196` (the clock words), `2263-2299` (the
  ground's material), `3142-3169` (the band tables every frame), `4430-4441` (the table's bytes), `4474-4571` (the
  placement).
- `crates/client-render/src/ladder_fade.wgsl:83-87, 136-185, 191-202`; `ladder_fade_prepass.wgsl:54-58, 156-172, 182-204`;
  `probe.wgsl:62-66`; `star_sky.wgsl:146-157`.
- `crates/client/src/arrival.rs:37-68, 96-127`; `crates/client/src/realm_scene.rs:686-690`;
  `crates/client/src/ladder_view.rs:76-84, 199-209, 377`.
- `crates/core/src/geometry.rs:1156-1173`; `crates/client-harness/src/camera.rs:270-291`;
  `crates/client-harness/src/pop.rs:10-19, 52-61, 134-136, 305-332`; `crates/client-harness/src/probe.rs:16-20, 181-195`;
  `crates/bins/tests/terrain_pictures.rs:222-226, 857, 1398-1405, 1651`.
- The review's re-reads (2026-09-29): `crates/client-render/src/lib.rs:824, 862` (the stars' alpha), `1422` (the window's
  45° view), `3244-3262` (the probe camera), `3475` (the capture saves RGBA); `terrain.rs:174-180` (the thinning switch),
  `3629-3656` (the probe twin); `ladder_fade_prepass.wgsl:166-172` (the four-float previous place).
- `vendor/bevy_pbr/PATCH.md` (the four changes); `vendor/bevy_pbr/src/atmosphere/render_sky.wgsl`;
  `atmosphere/resources.rs:404-417, 537-560`; `atmosphere/node.rs:227-276`; `atmosphere/mod.rs:185-205`;
  `deferred/mod.rs:115-127, 192-223`; `deferred/pbr_deferred_functions.wgsl:25-80, 131-155`;
  `deferred/deferred_lighting.wgsl:55-99`; `material.rs:1238-1245, 1275-1300`; `extended_material.rs:78-89, 338, 372-385`;
  `prepass/mod.rs:188-236, 1082-1170`; `render/mesh.rs:1215-1250, 2470-2472`; `render/mesh_preprocess.wgsl:215-233`;
  `render/pbr_functions.wgsl:97-122, 400-455`; `deferred/pbr_deferred_functions.wgsl:116`;
  `render/view_transformations.wgsl:225-232`; `render/clustered_forward.wgsl:51-52`; `render/light.rs:1233-1234`;
  `ssao/ssao.wgsl:50, 144`; `ssao/mod.rs:505-508`; `ssr/mod.rs:444-446`; `meshlet/mod.rs:288-290`;
  `decal/clustered.rs:1-10`; `decal/forward.rs:61`; `material.rs:626-629`; `atmosphere/sky_air.wgsl:3-13, 40-41`;
  `atmosphere/node.rs:60-125` (the table passes); `atmosphere/environment.rs`; `PATCH.md:59` (the patch dies at the upgrade
  gate).

**The engine as linked** (the cargo registry: Bevy 0.18.1, wgpu 27):
- `bevy_anti_alias-0.18.1/src/taa/taa.wgsl:9-10, 164-178`; `taa/mod.rs:121, 163-166, 224-226, 384-401, 410-450`.
- `bevy_camera-0.18.1/src/camera.rs:83-109`.
- `bevy_core_pipeline-0.18.1/src/core_3d/mod.rs:977-988, 1042-1046` (re-read); `core_3d/main_opaque_pass_3d_node.rs:71-72,
  95-99` (re-read); `deferred/mod.rs:17-19` (re-read); `prepass/mod.rs:52-72`; `oit/mod.rs:164-167` (re-read);
  `tonemapping/tonemapping.wgsl:33` (re-read).
- `bevy_render-0.18.1/src/texture/texture_attachment.rs:40-80` (re-read); `view/mod.rs:945-949, 989` (re-read).
- `bevy_anti_alias-0.18.1/src/contrast_adaptive_sharpening/node.rs:72-115`, `mod.rs:126-133, 263-267` (re-read).
- `bevy_light-0.18.1/src/cascade.rs:286-296` (research); `bevy_post_process-0.18.1/src/bloom/mod.rs:161-378` (research),
  `bloom/upsampling_pipeline.rs:121-128` (re-read).
- `wgpu-types-27.0.1/src/lib.rs:1785-1792` (re-read), `5598-5624`; `wgpu-core-27.0.3/src/as_hal.rs:336-360` (research),
  `command/render.rs:70-75` (re-read); `wgpu-hal-27.0.4/src/metal/conv.rs:298-305`, `metal/command.rs:546-561, 576-580`,
  `vulkan/command.rs:861-867` (re-read).

**The rulings and the registry:**
- `docs/design/owner_decisions_2026-09-22_believability.md:605-667` (B22, B23).
- `docs/design/owner_decisions_2026-09-07_voxels.md:703-729, 758-789` (V16, V18).
- `docs/design/owner_decisions_2026-09-12_frozen_patches.md:84-93` (F6).
- `docs/design/DEFERRED.md:10619-10747` (THE SPEED STEP; the clock at `10727-10741`; the looks off at `10666-10667`; the
  window's steps at `10702-10707`), `8595-8645` (D-TERRAIN-6: the one sky mode at `8621-8627`; the stale SMAA note at
  `8640-8643`), `8925-8972` (the window's own frame; the kept partial step at `8940-8944`; the control runs at
  `8936-8939`).
- `docs/investigation/2026-09-27/one_volume_design.md:1812` (B-FRAME).
- Commits: `4c155d0` (step 19: the edge kept, 0.41 against 0.43), `610acc4` (step 20: one sky mode), `b5f0510` (step 21:
  the handover without a pop).

**Outside sources** (as the industry research read them):
- Bevy: https://docs.rs/bevy_anti_alias/0.18.1/bevy_anti_alias/taa/struct.TemporalAntiAliasing.html ;
  https://docs.rs/bevy_camera/0.18.1/bevy_camera/struct.MainPassResolutionOverride.html ; https://bevy.org/news/bevy-0-17/
- wgpu: https://github.com/gfx-rs/wgpu/blob/trunk/CHANGELOG.md
- Apple: https://developer.apple.com/documentation/metalfx/mtlfxtemporalscalerdescriptor ;
  https://developer.apple.com/documentation/metalfx/mtlfxtemporalscalerbase/reactivemasktexture ;
  https://developer.apple.com/videos/play/wwdc2022/10103/ ; https://developer.apple.com/videos/play/wwdc2025/211/ ;
  https://developer.apple.com/videos/play/wwdc2020/10632/ ;
  https://developer.apple.com/documentation/metal/tailor-your-apps-for-apple-gpus-and-tile-based-deferred-rendering ;
  https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/
- MetalFX from Rust: https://crates.io/crates/objc2-metal-fx ; https://github.com/mistakeknot/ushas (`bevy_metalfx`);
  https://github.com/mistakeknot/ushas/blob/main/docs/m5-max-performance-research.md (the community cost, L)
- AMD: https://github.com/GPUOpen-Effects/FidelityFX-FSR2 ;
  https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/tree/v1.1.4 ; https://github.com/godotengine/godot/pull/81197
- NVIDIA: https://github.com/NVIDIA/DLSS/blob/main/doc/DLSS_Programming_Guide_Release.pdf
- Epic: https://dev.epicgames.com/documentation/en-us/unreal-engine/temporal-super-resolution-in-unreal-engine ;
  https://dev.epicgames.com/documentation/en-us/unreal-engine/sky-atmosphere-component-in-unreal-engine
- The sky and clouds: https://sebh.github.io/publications/egsr2020.pdf ;
  https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/s2016-pbs-frostbite-sky-clouds-new.pdf ;
  https://www.guerrilla-games.com/read/nubis-cubed ;
  https://d3d3g8mu99pzk9.cloudfront.net/AndrewSchneider/The-Real-time-Volumetric-Cloudscapes-of-Horizon-Zero-Dawn.pdf
- Temporal methods: http://behindthepixels.io/assets/files/TemporalAA.pdf (the survey);
  https://de45xmedrsdbp.cloudfront.net/Resources/files/TemporalAA_small-59732822.pdf (Karis 2014)
- The games: https://starcitizen.tools/Update:Star_Citizen_Alpha_3.23.0 ;
  https://www.nvidia.com/en-us/geforce/news/dlss-hellblade-2-serum-ships-at-sea/ ;
  https://www.pcgamingwiki.com/wiki/Star_Citizen ; https://www.pcgamingwiki.com/wiki/Elite_Dangerous ;
  https://www.pcgamingwiki.com/wiki/No_Man%27s_Sky ; https://www.pcgamingwiki.com/wiki/Starfield ;
  https://www.pcgamingwiki.com/wiki/Microsoft_Flight_Simulator_2024 ; https://www.nomanssky.com/2023/06/no-mans-sky-on-mac/
- Tile GPUs: https://github.com/KhronosGroup/Vulkan-Samples/blob/main/samples/performance/msaa/README.adoc

---

## 19. Review answers

A review of 2026-09-29 sent 18 findings. The writer checked each one against the code at the cited lines or against the
numbers. **All 18 held; none was refuted.** One fix was refined (row 13), and one rule went further than the finding asked
(row 10). Each row: the finding, the verdict with its check, and what changed.

| # | The finding | Verdict | What changed |
|---|---|---|---|
| 1 | M-3's hole census reads only the probe, and the probe cannot see a route-G3 hole, so M-3 could not fail (§4.5, §11 T0, risk 4) | HELD: `ground_holes(&probe, …)` (`terrain_pictures.rs:857, 1651`); the probe is a camera of its own with MSAA off, no prepass and twin meshes (F31) | T0 builds the picture census (a sentinel clear colour, the air and the stars off) with a control that must fail (one chunk one depth step nearer in the prepass). M-3 chooses T1's route only after that control reads; until then G3's holes are UNMEASURED. §4.5 item 3, §11 T0/T1/T3, §12.3, §12.4 (G-PICTURE-HOLES), risk 4, F31. |
| 2 | Under the override the view's `viewport` keeps the display size; the deferred light, the cluster lookup and SSAO read it, so G2 lights every ground pixel at a wrong place at `s` < 1 (§4.5, §4.6, §4.8, D4, risk 21) | HELD: `view/mod.rs:945-949, 989`; `pbr_deferred_functions.wgsl:116` through `view_transformations.wgsl:225-232`; `clustered_forward.wgsl:52`; `ssao.wgsl:50, 144`; no vendored shader reads `main_pass_viewport`; the air pass is correct (its ray from `in.uv`) | Patch seven (the render size's viewport for the deferred light and the cluster lookup) joins T3, §4.5, §4.6, §4.8, D4 and risk 21. §4.4 explains the two viewports with an example. G-DEFERRED-PARITY runs at every scale. The T2 sky change joins §4.8's list. F27, risk 26. |
| 3 | G-MOTION-TRUTH's reference keeps each ground point still in the body's frame, so in the ramps and the bands it fails the correct rule and passes today's formula; its own miss reaches 0.25 px (§11 T1, §12.4) | HELD: `pop.rs:10-19` (one point in the body's frame), `:323` (whole cells), `:58` (0.25 px); today's formula makes the same assumption (`ladder_fade_prepass.wgsl:166-171`) | The pixel gate reads only the ground that the probe does not mark as moving, where the reference's own miss is under 0.02 px. The Tier-A tests alone judge the bands and the ramps. Three controls must fail. T1 builds the probe's moving mark. F32, §12.4, risk 1. |
| 4 | In-tile MSAA needs one pass and a discard store, which wgpu 27 has; memoryless textures only save memory (§3 reason 1, §2.5 R3, F21, D1) | HELD: `render.rs:70-75`; `metal/conv.rs:298-305` (no store + a resolve = "multisample resolve"); `metal/command.rs:546-561, 576-580`; Bevy always stores (`texture_attachment.rs:40-80`) | Reason 1 names the engine's pass structure (a second vendored crate and the vendored air), not the Bevy upgrade. D1 gains option (d) with its price and the reason the design does not recommend it. F26; R3, R9, F21 and §2.3 item 3 corrected. |
| 5 | D5 judges the edge only at `s` = 1, but only 0.5 or lower meets the frame, and the owner would first see the up-scaled edge after T3 built the deferred pass (§11, D2, D5, D8) | HELD: the order's reason, T2 at `s` = 1, T4's up-scale; a deferred camera forces MSAA off (`core_3d/mod.rs:977-988`) | T2 builds the up-scale on route G3 (no engine change: the forward ground takes its place from the vertex stage, and no lamp exists today). D5 judges at 1.0, 0.67 and 0.5. T3 waits for D5 = yes at a size that can meet the frame. T4 is the budget and the scale; G-BUDGET stays after T3. §15 row 2. |
| 6 | Many lamps work under MSAA; only the deferred light refuses it (§0, §3 reason 4) | HELD: the forward light walks the cluster's point and spot lights (`pbr_functions.wgsl:400-455`); only a deferred camera forces MSAA off | §0 and §3 call the deferred light "a cheaper way to light many lamps" and take the lamps out of the refusals. F28. |
| 7 | Nothing writes alpha 1 after the points pass, so the owner's PNGs carry alpha 0 on the sky (§7.7, §12) | HELD: the stars add alpha 0 (`lib.rs:824, 862`); bloom and the tone map keep the alpha; the capture saves RGBA (`lib.rs:3475`); the compare reads four bytes (`terrain_pictures.rs:1398-1405`) | §7.7 change two: the reconstruction writes the display colour with alpha 1, and the sky share rides in a channel of its own. G-ALPHA (every display pixel 255) in T2, §12.1 and §12.4; §8.3's outputs. F30, risk 27. |
| 8 | The 2 × 2 block rule is new: the fourth patch measured its partial step only for one pixel's samples (§7.4, T5, D10) | HELD: `sky_air.wgsl:3-5` and `:40-41` (one ray per pixel, from its centre) | §7.4 names the block rule NEW and UNMEASURED. T5 measures its direction error at the sun's glow and the horizon (dusk, aloft, orbit) before D10's ranking. F33, risk 16, §15 row 13. |
| 9 | "Yes, with about 2 ms to spare" at 0.5 does not follow: the top of the range misses, the middle is assumed, a still median is not a moving p99, and M-2's threshold uses the lowest add-on (§0, §10.3, §15 row 3, M-2) | HELD: 13.3–17.9 ms (12.3–17.3 with the cheaper air); 14.9 + 3.35 − 0.825 = 17.4 ms | §0 reads "likely at the middle with the cheaper march, NOT assured" and shows the floor's row. §10.3 item 3 lists the caveats. M-2's thresholds come from the middle add-on (12.5, 13.3 and 15.1 ms at 0.5). D8 is conditional on M-2, T2's own cost and M-5's p99; M-5 runs at 1.0 and 0.5. |
| 10 | The model reads low: no sharpen row, S from 0, r up to 0.7 against bloom's own traffic, and full-size targets under the override (§10.1, §10.2, D7, G-BUDGET) | HELD: the sharpen is a full-screen HDR pass (`contrast_adaptive_sharpening/node.rs:72-115`, `mod.rs:263-267`); 180 MB of bloom takes ≥ 0.66 ms, so r ≤ 0.48; every main-pass target has the display size and is stored (`main_opaque_pass_3d_node.rs:71-72, 95-99`; `vulkan/command.rs:861-867`) | One traffic rule for every full-screen pass: new rows H (0.45–0.6 ms) and W (UNMEASURED, at most 0.47 ms at 0.5, out of the sums); S 0.45–0.8 ms; r 0.3–0.5. The same rule also lifts O's low end to 1.0 ms (beyond the finding). The middle at 0.5 moves from 15.6 to 16.8 ms (14.8 to 16.0 with the cheaper air). The error is stated both ways. M-2 runs the per-pass clock under the override; M-2b prices G2's targets before T3. G-BUDGET's line waits for M-2 and T2. D7's price is 0.45–0.6 ms. F29, risk 28. |
| 11 | The one-pass MSAA answer rests on a wrong premise (the upgrade) and the wrong cost (today's multi-pass 10.5 ms) (§3 reasons 1 and 6, §0) | HELD: the code of row 4; §2.3 item 3 itself calls the 10.5 ms mostly sample traffic | The decisive number comes first in §0 and §3: without any MSAA the frame is 21.75 ms. Reason 6 calls one-pass MSAA's cost UNMEASURED. "What can change the answer" names the pass structure. Apple's quote joins R3 as REPORTED (quoted by the review, not read again). |
| 12 | The answer overstates what MSAA cannot do: lamps, decals, thin rails, far ridges, foliage (§0, §3 reasons 2 and 4, §4.5, D4) | HELD: the clustered forward lights; the clustered decals (`decal/clustered.rs:9-10`); the forward decals refuse MSAA only on the web (`decal/forward.rs:61`); alpha-to-coverage only under MSAA (`material.rs:626-629`); SSAO, SSR, OIT, meshlets and the deferred light re-read as the true refusals | Reason 2 keeps the shading examples; rails and ridges move to "MSAA smooths them; the temporal path needs locks". Reason 4 lists the refusals exactly. "What MSAA does better" gains thin geometry and alpha-to-coverage. G2 opens SSAO, SSR and lighting once per pixel (§4.5). §4.6 notes the forward lamps and the clustered decals. F22 re-read; F28. |
| 13 | The motion's vertex output is memory traffic for each vertex on this GPU (§5.2, §4.5, §10.1) | HELD: four floats of previous place for each vertex (`ladder_fade_prepass.wgsl:166-172`); the tiled vertex buffer is REPORTED (Apple, as the review quoted it) | §5.2 names the traffic. New row V (per vertex, not scaled by `s`, UNMEASURED, out of the sums). T1 prices it with the output on and off; G1's cost row adds it; risk 29. REFINED: the smallest EXACT output is three components (last frame's clip x, y and w), because two would lose the perspective division. |
| 14 | D5's price compares MSAA with the reconstruction alone (D5) | HELD: the T2 frame on route G3 was 24–32 ms against 32.25 ms today | D5 states the whole price: at `s` = 1 about 5–8 ms saved on G2 and 0–8 ms on G3 (the revised model); 15–24 ms at 0.5 on G3; 60 frames a second only with D8 and T3. |
| 15 | The triangle split is mixed, because the thinning opens holes (§2.2, §4.5, risk 3) | HELD: `terrain.rs:177-179` ("holes in the picture"); `DEFERRED.md:10670-10671` | §2.2 tags the split MIXED. M-3 (the same chunks drawn a second time) prices G3; the split is only a hint (§0, §4.5, risk 3). F34. |
| 16 | The vendored crate changes more than twice: T2 and T5 change the vendored air (§4.8, risk 21, D1) | HELD: `render_sky.wgsl`, `node.rs`, `environment.rs` and `sky_air.wgsl` live in `vendor/bevy_pbr`; `PATCH.md:59` ("the patch dies at the Bevy upgrade gate") | §4.8 lists every vendored change by slice (T2; T3 five, six, seven; T5 up to four; T8 removes two and four). Risk 21, D1 and D4 follow; §0 D4 says three changes. |
| 17 | "The model gives A1 and A2 exactly" holds by construction and checks nothing (§10.1, §15 row 19) | HELD: F, the 6.1, A, B and MSAA are differences of A1–A5 | §10.1 says the model restates the measurements, and M-2 at 0.75, 0.67, 0.5 and 0.33 is its first test. §15 row 19 follows. |
| 18 | The sky's parallax at 528 m/s and 100 km is about 0.23 display pixels a frame, not under 0.1 (§6.3) | HELD: 8.8 m a frame ÷ 100 km = 0.088 mrad; ÷ 0.382 mrad a display pixel (45°, `lib.rs:1422`) = 0.23; 0.12 at 120 frames a second | §6.3 corrected. G-CROSSING's sky reading and the sky judge's lag reading on the 528 m/s leg measure the trail; the cure is the air body's full step (T2 gates, risk 12, §17 item 25). |
