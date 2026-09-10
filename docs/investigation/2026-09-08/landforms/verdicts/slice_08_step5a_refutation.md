# Slice 8 step 5, first half — THE REFUTATION (ruling V16: pack the bytes, keep the picture)

The refuter read the working tree against HEAD (73a9758) on 2026-09-10. The refuter ran no build and
no test: a measured flight holds the machine. Every finding below stands on a file and a line.

The step-4 findings and the throughput findings are not repeated here
(`slice_08_step4_refutation.md`, `slice_08_throughput_refutation.md`).

## The findings

| # | Severity | Where | Claim | Evidence | What to do |
|---|---|---|---|---|---|
| P-1 | DEFECT | `crates/client-render/src/terrain.rs:735`–`758`, `:881`, `:866`; `crates/client-render/src/lib.rs:700` | A ground material born in the harvest draws ONE FRAME with the body's centre at the EYE. The morph and the sink then run along the line of sight, not along the vertex's radial. SL8 calls a one-frame jump a seam. | `LadderFade::new` writes `centre_sink: Vec4::new(0.0, 0.0, 0.0, sink_m)` (`lib.rs:700`), so a new material states the centre as the render frame's own origin — the eye. The centre loop runs at `terrain.rs:735`; `ground_material` (`:881`) first creates the material inside the harvest loop at `:866`, LATER in the same system. The chunk entity spawns this frame (`place_chunks` is chained after `sync_terrain`, `terrain.rs:1287`), so the engine draws it before the next frame's centre loop. For a rung nearer than its own fade-in edge, `risen(d)` is 0 and the WHOLE sink applies: tens of metres sideways at a coarse rung. It happens once per realm and rung, during the fill the picture gate waits on. | State the centre when the material is BORN: hand the realm's eye-relative centre to `ground_material` and `probe_material`, or move the centre loop after the harvest. |
| P-2 | DEFECT | `crates/client-render/src/terrain.rs:735`–`758`; `crates/client-render/src/lib.rs:704`–`714`; `slice_08_ladder_discussion.md:763`–`766` | On a MOVING eye every ground material and every probe material is written EVERY frame, so the engine re-prepares them all. The step-5 cure ("write only when the centre moved") is inert exactly where V16 wants the frames. The cost there is UNMEASURED. | The centre is the realm's centre in the render frame, and the render frame's origin is the eye. A hull that flies moves the centre every frame, so `centre_is` (`lib.rs:707`) is false every frame and `set_centre` runs on about ten ground materials and eleven probe materials. The same write on the STILL stand was MEASURED at 26 → 18 frames a second (`lib.rs:704`–`706`, §17.3). Nobody measured the moving leg (`terrain_moving_eye`). | MEASURE the frame rate on M8-1's 240 m/s leg with the centre write and without it. If it costs, drop the uniform: pack the vertex's radial as an octahedral pair (4 bytes — still 32 bytes a vertex against 48), and the shader then needs no centre and no per-frame write. |
| P-3 | DEFECT | `crates/bins/tests/terrain_pictures.rs:920`, `:937`, `:940` | The compare masks the overlay with THIS run's HUD rectangle only. The picture ON DISK was drawn by an earlier run whose HUD was a different size, and its extra text over ground counts as a content difference. | `let hud = stamp.hud_rect_px;` (`:920`) reads one rectangle, from this run's stamp. Nothing stores the reference picture's rectangle. The HUD's width follows its longest line, and that line prints `chunks_per_rung` as a vector (`lib.rs:2333`–`2343`), whose width changes with the rungs and the counts — that is, with every code change V16 asks the gate to judge. | Write the reference picture's HUD rectangle beside it (a small sidecar file) and mask the UNION of the two; or give the probe camera its own HUD, so the probe marks the text; or draw no HUD for the identity capture. Grow the rectangle by a pixel or two: egui antialiases its glyph edges. |
| P-4 | DEFECT | `crates/bins/tests/terrain_pictures.rs:154`, `:490` | The fixed capture tick is NOT the same moment for two flights. It is the same moment only while both settles fall inside one grid step. | `capture_tick = (settled_tick / 1_200 + 1) * 1_200` (`:490`). A settle at tick 1 199 captures at 1 200; a settle at tick 1 201 captures at 2 400 — a minute of world time apart, and the star turns 0.24° in it. The settle is a wall-clock race (the fill was MEASURED at 2.7 s to 11.9 s, §17.1), so two flights straddle the boundary now and then. The document calls it "the SAME moment" (§17.1, line 712). | Capture at a CONSTANT tick. The test already asserts the clock is near its genesis (`:437`), so a fixed tick past the longest settle is exact and repeatable. Wait for it; do not round up from the settle. |
| P-5 | LAW (V16) | `crates/bins/tests/terrain_pictures.rs:911` and the copy after the compare block; `slice_08_ladder_discussion.md:768`–`774`; `git status` on `docs/investigation/2026-09-07/pictures/*.png` | V16 says "the picture gate measures the packed picture against the UNPACKED one and reports the difference; the owner sees the number". That measurement is lost, and the gate cannot make it again, because the reference is the same file the test overwrites. | The test compares against `docs/investigation/2026-09-07/pictures/<name>.png` (`:911`) and then copies the new picture over it. All four pictures and their probes are modified in the working tree. §17.3 states the loss plainly. What stands in its place is a packed-against-PACKED floor and a library unit test under 1e-4 m. | Tell the owner, in the report, that V16 item 1's own measurement is NOT made, and that the geometry's exactness rests on the unit test and on the packed floor. Then make the reference immutable: keep the reference pictures in a directory the test never writes, and write this run's output beside the run. |
| P-6 | COVERAGE (HR5 d) | `crates/client/src/chunks.rs:2282`, `:2283` | Two asserts join two conditions with `&&`. HR5 rule (d) forbids it: the short circuit leaves an uncoverable branch, and `vd-client` is Tier-A (`justfile:12`). | `assert!(lo[k] <= v[k] && v[k] <= hi[k]);` and the same line for the target. | Split each into two asserts. |
| P-7 | COVERAGE | `crates/client/src/chunks.rs:2272`, `:2277`–`2286`; `:162` | The bounds test cannot catch a wrong SINK in the box. It reads a rung-0 chunk, whose sink is zero, so the box's third corner (`v − s`, `chunks.rs:162`) is the vertex itself. | The test builds `surface_key(&body, 0, 300, 700)` — rung 0 — and `sink_m(body, 0)` returns 0.0 (`chunks.rs:62`–`64`). The loop checks only the vertex and the target against the box. | Repeat the box check on a rung-1 chunk, and check the sunk position `v − sink_of(i)` as well. |
| P-8 | DEFECT (cost) | `crates/client-render/src/terrain.rs:401`–`404`, `:887`; `crates/client/src/chunks.rs:145`–`169` | The culling box became five to ten times dearer, in `f64`, on the MAIN THREAD, inside the harvest loop — the loop V15 named as the frame's cost. | The old box was three `f32` min/max steps a vertex (HEAD `terrain.rs:262`–`280`). The new `ChunkGeometry::bounds` calls `morph_target(i)` and `sink_of(i)`, and each calls `radial(i)` — one `f64` normalise each, so two square roots a vertex, plus a scalar `while` loop over three components. The harvest takes 24 chunks a frame at about 7 250 vertices a chunk. | Build the box in the WORKER: compute it inside `geometry_with` and carry it on `ChunkGeometry`. The worker already holds the exact radial in `f64` and pays nothing extra. |
| P-9 | DOC | `slice_08_ladder_discussion.md:705`–`707` against `:745`–`751` | "The frame rate follows the bytes… THE PACKING is the frame's lever before anything else" is not what the packing measured. | The bytes fell 44 % and the still frame rate rose 13 % on the ground stand (26.0 → 29.4). If the bytes were the frame's cost, the frames would have risen about 79 %. The measurement says the bytes are ONE cost among larger ones — the vertex count and the seven thousand draws, which the packing does not touch. | Rewrite the sentence to what the numbers support: the packing bought an eighth of the frame; the vertex count and the draw count hold the rest, and they are the next levers. |
| P-10 | DOC | `slice_08_ladder_discussion.md:709`–`727` against `:776`–`784` | Three noise-floor tables sit in one section under three different rules, and a reader compares them. He must not. | §17.1's floor is UNMASKED and comes from two UNPACKED flights (ground 0, hill 76, aloft 162, orbit 71). §17.3's first floor is MASKED by the probe and comes from two PACKED flights (ground 5, hill 158, aloft 0, orbit 6). §17.3's second is masked by the probe AND the overlay, from a third pair (ground 7, hill 0, aloft 2, orbit 5). A mask can only REDUCE a count on ONE pair; ground 5 → 7 proves these are different pairs. §17.1 states the rule "a packing must not exceed it", and by that rule the ground stand already exceeds its own floor. | Name the flight pair and the rule beside every table. Delete the "must not exceed" rule, or restate it against a floor measured under the SAME mask. |
| P-11 | DOC | `crates/client-render/src/lib.rs:420`–`422`, `:677`; `crates/client-render/src/ladder_fade.wgsl:5`–`16` | Three places still describe the deleted vector form. | `lib.rs:420`–`422` calls `ATTRIBUTE_MORPH` "where it stands on the next coarser rung's surface, relative to the chunk's origin" — it is now ONE metre along the radial. `lib.rs:677` names `ATTRIBUTE_SINK`, which the diff deletes; the symbol exists nowhere else in the tree. `ladder_fade.wgsl:5`–`16` calls `morph` a second position and `sink` a per-vertex drop. | Rewrite the three. |
| P-12 | DOC | `slice_08_ladder_discussion.md:735` (§17.2, THE BODY'S CENTRE row) | Two numbers in the exactness row are wrong, though the conclusion survives. | "its `f32` rounding is half a metre at most": at the ORBIT stand the eye is 2 000 km up, so the centre is 8.4e6 m away, past 2^23, and the rounding is ONE metre; the reach allows two body radii (`ladder_view.rs:139`), which is 1.3e7 m. "nanometres on a target": a direction error of 1e-7 on a morph metre of a thousand metres is 1e-4 m — a TENTH OF A MILLIMETRE, not a nanometre. Both stay far under a pixel. | Correct the two numbers and keep the conclusion. |
| P-13 | NIT | `crates/client-render/src/lib.rs:707`–`709`, `:650`–`652` | `centre_is` compares two `f32` vectors for exact equality and uses the answer as a cache key. One last-bit wobble in the delivered interpolation turns the saving off with no sign. | `self.centre_sink.truncate() == centre`. The centre comes from `draw_center_of` (`lib.rs:1773`), which interpolates the row and the eye at the wall clock. The still stand measured well, so it holds today; nothing states it must. | Compare with a tolerance — a millimetre is far under the metre of `f32` rounding the design already accepts — and say so in the doc comment. |
| P-14 | NIT | `crates/client-render/src/terrain.rs:747`–`757`, `:492`–`501` | The RULER's probe material is rewritten every frame the eye moves, for no effect at all. | The loop covers every entry of `probe_materials`, the ruler's included. A ruler probe holds `sink = 0.0` (`:499`) and the ball's morph attribute is zero (`:1229`–`1230`), so the centre never changes a ruler pixel. The write still makes the engine re-prepare the material. | Skip the write for a kind that is not `PROBE_KIND_TERRAIN`. |
| P-15 | NIT | `crates/client-render/src/ladder_fade.wgsl:87`; `probe.wgsl:68`; `ladder_fade_prepass.wgsl:57` | A vertex exactly at the body's centre makes `normalize` return NaN, and the NaN survives a zero morph metre (`NaN × 0 = NaN`), so the vertex and its triangles vanish. Nothing guards it and nothing states it cannot happen. | `let radial = normalize(own.xyz - fade.centre_sink.xyz);` with no length test. The ladder puts no chunk at a body's centre, so today it cannot happen; that is an argument, not a fence. | State the reason in the shader comment, or guard with a length test. |
| P-16 | NIT | `crates/client-render/src/terrain.rs:586` | The census calls the number "on the GPU". The client pays it TWICE: the mesh also stays in the main world. | `Mesh::new(..., RenderAssetUsages::default())`; the engine's own doc says the default is `MAIN_WORLD | RENDER_WORLD`, and that an asset which does not change should use `RENDER_WORLD` alone (`bevy_asset-0.18.1/src/render_asset.rs:40`–`48`). Nothing reads a chunk mesh back: the culling box is supplied (`:887`), and the flat path duplicates before `meshes.add`. So 1 880 MB on the GPU is 1 880 MB in RAM as well. | Try `RENDER_WORLD` alone and MEASURE the client's resident memory. It is a second 44 % with no pixel at risk. |
| P-17 | NIT | `crates/client-render/src/terrain.rs:592`–`595`, `:920`–`924` | Under `VD_TERRAIN_FLAT` the census's two numbers describe two different meshes. | `bytes_drawn` reads the BUILT mesh, after `duplicate_vertices` (three vertices a triangle, no indices); `vertices` reads `geometry.vertices.len()`, before it. | State in the stamp's doc that `vertices` is the library's count, or count the built mesh's vertices. |

## What the refuter tried to break and could NOT

1. **The morph metre is exact.** Every source of `target_m` in `geometry_with` is a RADIUS along the
   same `dir` — the vertex's own radial: the top rung's `len` (`chunks.rs:817`), a parent mesh's
   `radial_hit_m`, which the code defines as "the radius at which the radial along `dir` meets this
   mesh" (`chunks.rs:369`–`392`), and the coarser field `height_m(body, dir, coarser)`
   (`chunks.rs:844`–`853`). The seam vertex and the fallback vertex both read the field along the
   SAME `dir`. No path makes a target off the radial, so one signed metre carries it whole. The unit
   test measures the two forms apart under 1e-4 m over more than 5 000 vertices
   (`chunks.rs:2233`–`2267`).
2. **The shader's radial is good enough at every rung.** The centre is at most 1.3e7 m away, so its
   `f32` rounding is at most one metre and the direction error is under 2e-7. On a morph metre of a
   thousand metres that is 1e-4 m across. At the switch distance a pixel is about 1.1e-3 radians
   (45° over 720 rows), so the error is a ten-thousandth of a pixel. The old form was exact per
   vertex, but it was narrowed to `f32` against the chunk's origin, and the world-space sum rounds
   at the same place in both forms. No regression.
3. **The skirts agree.** A skirt vertex is its top dropped along the radial (`chunks.rs:928`–`929`),
   so it shares the top's radial, and carrying the top's METRE puts its target exactly where the old
   code put it (the top's target less the same drop, HEAD `chunks.rs:642`). The prepass and the
   probe run the identical expression, so the shadow and the probe morph the skirt with the picture.
4. **The 16-bit indices are safe.** `packed_indices` allows at most 65 535 vertices, so the largest
   index is 65 534. The flat path is no hazard: `Mesh::duplicate_vertices` TAKES the indices
   (`bevy_mesh-0.18.1/src/mesh.rs:993`) and leaves the mesh unindexed, whatever their width. No gate
   sets `VD_TERRAIN_FLAT`.
5. **The mask does not hide a hole.** A pixel counts when EITHER probe marks it
   (`terrain_pictures.rs:938`–`939`), so ground that VANISHED still counts through the reference's
   probe. The hole gate is a second, independent net on the same flight: `ground_holes` on this
   run's own probe, `0` blocks and `0` pixels (`terrain_pictures.rs:573`–`589`).
6. **Two `#[uniform(100)]` fields are lawful.** The engine's own derive doc says that fields sharing
   a binding index combine into one struct in declaration order
   (`bevy_render-0.18.1/src/render_resource/bind_group.rs:244`–`264`). `bands` then `centre_sink`
   matches the WGSL struct in all three shaders.
7. **The culling box holds the whole vertex stage.** The vertex's offset along its radial is
   `morph_m × (1 − whole(d)) − sink × (1 − risen(d))`. The sink ramp ENDS before the morph ramp
   BEGINS (`sink_end` sits between the fade-in edge and `out_lo`, `ladder_view.rs:117`–`126`), so
   the two never add. The offset therefore runs over `[−sink, 0] ∪ [0, morph_m]`, and the box takes
   exactly those three points (`chunks.rs:160`–`162`).
8. **The prepass's motion vector is no worse.** `previous_own + (morphed − own)` and the old
   `previous_from_local × local` differ only by the row's rotation over one frame applied to the
   morph vector — millimetres. No motion-vector prepass is even requested: there is no temporal
   antialiasing anywhere in `client-render`, and `Msaa::Off` stands at `lib.rs:2595`.
9. **The ruler is not hurt by the centre.** Its probe holds sink 0 and its ball's morph metre is 0,
   so `coarser` is the vertex and the `mix` returns the vertex, whatever the centre says.
10. **The census's arithmetic holds.** 6 659 chunks and 48.3 M vertices is 7 254 vertices a chunk;
    at 48 bytes plus 12 bytes a triangle that is 495 KB, and at 28 bytes plus 6 it is 277 KB — the
    table's 495 → 276 KB. The `vertices` count is the library's, skirts INCLUDED (`add_skirts` runs
    before `geometry_with` returns, `chunks.rs:877`).

## The answers (2026-09-10, after the refutation)

| # | Verdict | The answer |
|---|---|---|
| P-1 | SUPERSEDED | No material carries a centre any more: the radial rides the vertex (`ATTRIBUTE_RADIAL`, four signed 16-bit quanta), so nothing is born wrong and nothing is rewritten. |
| P-2 | MEASURED, then removed | With the centre in the uniform: 26 → 19 frames a second on the walk and 41 → 18 at 240 m/s (a millimetre's tolerance), 29.7 and 18.6 with a millionth of the distance. With the radial on the vertex: 29.9 and 45.1. The finding was right, and the fallback it named is the design now. |
| P-3 | FIXED | The reference's overlay rectangle rides a sidecar beside the picture; the compare masks the union of both, grown by 2 px for the glyphs. |
| P-4 | FIXED | The capture tick is a constant per stand (1 200 × the stand's index + 1); a settle past it is a red gate, not a silent shift. |
| P-5 | STANDS, stated | V16's own measurement, packed against unpacked, is lost: the gate overwrote its reference before the mask existed. The owner reads that plainly in §17.3 and in the report. From here every compared picture is kept under `target/terrain_pictures`; what stands is the geometry's reconstruction bound (a unit test) and the masked packed noise floor at one channel step. |
| P-6 | FIXED | Split. |
| P-7 | FIXED | The bounds test runs on rung 0 and on rung 1 (which sinks) and checks the sunk corner; the worker's box equals the measured box. |
| P-8 | FIXED | `geometry_with` builds `bounds` after the skirts; the harvest reads the field. |
| P-9 | FIXED | §17.1 restated: the packing bought an eighth of the frame for 44 % of the bytes; the vertex count and the draw count hold the rest. |
| P-10 | FIXED | Every noise-floor table names its flight pair and its rule; the "must not exceed" rule is dropped in favour of the per-stand judgement. |
| P-11 | FIXED | The attribute's doc, the material's doc and the shader's header describe the metre and the uniform. |
| P-12 | FIXED | A metre at most, a tenth of a millimetre on a kilometre's target. |
| P-13 | SUPERSEDED | No centre, no tolerance. |
| P-14 | SUPERSEDED | No centre, no rewrite loop. |
| P-15 | FIXED, stated | The comment says why: the ladder's shell begins at the crust, so no vertex stands at the centre. |
| P-16 | STANDS | `RENDER_WORLD` alone is a memory measurement for step 5's second half (a resident-memory instrument first). |
| P-17 | FIXED | The stamp's doc says the bytes count the built mesh, the vertices the library's. |
