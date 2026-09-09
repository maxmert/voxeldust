# Slice 8 step 3 (the crossfade) — THE REFUTATION

Read-only review of the working tree, 2026-09-09. I read the code, not the docs. Every number below
comes from the code or from arithmetic on it. Where I could not measure, I write UNMEASURED.

Severity: **DEFECT** = the shipped path is wrong or unsound. **UNMEASURED** = the claim may hold,
but nothing here could have failed. **DOC** = the prose and the code disagree. **NIT** = small.

---

## 1. DEFECT — the far column's disc is measured with the EYE's radius, so a far column is culled at altitude

`crates/client/src/ladder_view.rs:376-394`, `crates/client/src/skyline.rs:224-242`.

`column_geometry` states `r_out` as a CHORD in metres on the sphere of radius `surface`:
`r_out = max |corner − centre| × 1.01`. `clears` then reads it as a distance on the eye's CHART:

```
phi_n = (arc_m − r_m) / self.eye_m;   phi_f = (arc_m + r_m) / self.eye_m;
ray_range(arc_m, az, r_m)             // r_m used as chart metres
```

`arc_m` comes from `EyeFrame::ground`, which multiplies the central angle by the EYE's radius, so
`arc_m / eye_m` is the exact central angle. `r_out` is a length at the SURFACE's radius. The
column's true central half-angle is `r_out / surface`, not `r_out / eye_m`. The disc is therefore
too narrow by `1 − surface/eye_m`:

| stand | eye radius / surface | the disc is narrow by |
|---|---|---|
| ground 3.4 m | 1.0000005 | 0.00005 % |
| hill 300 m | 1.000047 | 0.005 % |
| aloft 60 km | 1.0094 | **0.93 %** |
| orbit 2 000 km | 1.3138 | **23.9 %** |

Both errors cull. A narrow span understates `theta` (the highest elevation the column can reach),
and a narrow `ray_range` reads the lowest wall over fewer rays, which raises `lowest`. The
`DISC_MARGIN` of 1 % is the only guard, and the aloft stand alone eats it whole. At orbit a rung-12
column is 254 km wide and its half-angle is understated by 24 %.

**Fix.** State the disc as a central angle once: `phi_half = asin(r_out / surface)`, and give
`clears` an angle, not metres. Where the chart wants metres, multiply the angle by `eye_m`. Add a
test at the orbit stand that a column just past the limb still clears.

## 2. DEFECT — `raise` never narrows its ray range, so every wall costs 3 600 rays

`crates/client/src/skyline.rs:194-204`.

```
while i < 4 { arc_m = arc_m.max(quad[i].length()); i += 1; }   // distance from the ORIGIN
let centre = (quad[0]+quad[1]+quad[2]+quad[3]) * 0.25;
ray_range(centre.length(), az, arc_m).unwrap_or((0, SKYLINE_RAYS as i64 - 1))
```

`arc_m` is the largest distance of a corner FROM THE CHART'S ORIGIN, and it is passed as the disc's
RADIUS around the centre. For any convex quad `|centroid| ≤ max |corner|`, so `arc_m ≤
centre.length()` never holds. `ray_range` returns `None` on every call, and every `raise` walks all
3 600 rays. The comment says "a superset of those crossing it"; it is the whole fan, always. The
intended radius is `max |corner − centre|`.

Cost, from the code's own numbers: about 2 000 near columns from the ground × 3 600 rays × one
`quad_exit_m` (four edge crossings each) = **29 million edge tests per recompute**, on the main
thread inside `sync_terrain`. The eye recomputes every `EYE_STEP_M` = 0.5 m
(`crates/client-render/src/terrain.rs:72`). A walker at 1.4 m/s recomputes about three times a
second; a hull at 240 m/s recomputes 480 times a second. The `Some` arm of that `ray_range` call is
DEAD CODE today.

**Fix.** `arc_m = max |quad[i] − centre|`. Then a 62 m column at 1 km spans about 36 rays, not
3 600 — a hundredfold cut. Measure the recompute on a moving eye before step 6 (D-TERRAIN-5 item 4
owes this already; the bug makes the owed measurement far worse than the doc's estimate).

## 3. DEFECT — the skyline bounds the FIELD, and the picture draws the MESH, one cell lower

`crates/client/src/ladder_view.rs:542-545`, `crates/terrain/src/digest.rs:166-216`.

The wall's floor is `sampled_low − column_bound − dropped_bound`. Those three terms bound the
recipe's FIELD over the column. The eye does not see the field. It sees the extractor's MESH, and
the mesh stands within about ONE CELL of the field — the same cell the sink pays for
(`chunks.rs:57-70`). So a drawn crest can lie a cell BELOW the wall, and a line of sight that grazes
the drawn crest passes under a wall that culled the valley behind it.

The size of the error is the tier rule's own scale: one cell at the switch distance is one pixel.
That is the size of the defect step 3 has just cured — 47 pixels of sky between two crests.

Near ground is also SUNK while a finer rung is drawn over it, and the floor does not subtract the
sink either. There the finer rung's own wall covers the case, but nothing states that invariant.

**Fix.** Lower each wall by one cell of its own rung, and by `sink_m` of that rung where the sink
can apply. It costs a few culls and nothing else.

## 4. DEFECT — at the band's far edge the coarser mesh can still show through, and no sink ramp can cure it

`crates/client-render/src/ladder_fade.wgsl:82`, `crates/client/src/chunks.rs:472-530`.

The morph is exact AT THE VERTICES: each finer vertex lands on the parent triangle its own radial
meets. Between vertices the two surfaces are different triangulations. Where a finer triangle spans
a coarser CREASE, the morphed finer surface is the chord under (or over) the coarser edge. Under a
convex coarser crease the morphed finer mesh lies BELOW the coarser one by the crease's sagitta.

At the band's far edge the coarser rung's sink is exactly ZERO by construction (`risen(in_hi) = 1`,
and `in_hi(L+1) = out_hi(L)` is the same line). So at the one distance where the two meshes must
coincide, nothing holds the coarser down. This is the measured residue — "2 dark pixels on the
hill ... the residual crease where a finer triangle cuts across a coarser edge" — and
§15.4 / D-TERRAIN-5 item 5 names the wrong lever: **a faster sink ramp cannot fix it, because the
sink is zero there whatever the ramp.**

**Fix, one of.** Keep a small residual sink at the far edge (end the ramp past `out_hi`). Or split
the finer triangles that straddle a coarser edge. Or bias the coarser rung's depth inside the band.
State which, and delete the "faster ramp" sentence from the doc.

## 5. DEFECT (unmeasured) — the morph target differs across a CUBE-FACE seam

`crates/client/src/chunks.rs:380-418` (`parent_keys` names only `key.face`) and `chunks.rs:420-441`
(`parent_cell` is a same-face computation).

`parent_keys` never crosses a face. A finer chunk at the edge of a face reads its halo vertices'
parent cell as 62 (or −1) of a parent that does not exist across the seam, finds no bucket, and
falls back to the coarser FIELD. The chunk on the other face reads the same shared vertex from ITS
parent MESH. The field and the mesh stand up to a cell apart — the exact difference the parent-mesh
target was built to remove (it cost 34 dark specks). So inside a fade-out band every cube-face seam
can open by a cell.

The doc concedes the cause in one clause — "a halo across a face seam reads the field" — and never
states the consequence. The shared-vertex test
(`chunks.rs:1348+`, `the_morph_targets_stand_on_the_parent_mesh_and_neighbours_share_them`) checks
only chunks 300 and 301, which share one parent in the MIDDLE of a parent's box: the easiest case.
No test covers a parent-boundary seam (301|302), a y or z seam, or a face seam. All four picture
stands look at mid-face ground.

**Fix.** Let `parent_keys` name the neighbouring FACE's parent (the extractor already maps
cross-face sites, `extract.rs:383-392`), or make both sides read the field at a seam. Add a test
that a vertex shared across a face seam gets one target.

## 6. UNMEASURED — the sink's "one cell" is a claim about VERTICES, not about the surface a radial meets

`crates/client/src/chunks.rs:57-70`, and the reach filter at `chunks.rs:471,497`.

`sink_m = gap bound + cell(L) + cell(L−1)` rests on "the extractor's placement lies within a
vertex's group". That is true of a VERTEX. The morph and the sink both read the surface along a
RADIAL, and a radial meets a TRIANGLE, not a vertex. On a steep slope a triangle's radial hit
differs from the field's radial crossing by the ground's own relief across the triangle, which is
bounded by the SLOPE, not by a cell. Nothing in the tree measures a slope bound.

Two consequences. The coarser rung may not be fully under the finer one where the ground is steep,
which is a show-through. And a legitimate parent hit farther than `reach_m` is discarded as "a cave"
and falls back to the field (`chunks.rs:497-499`), silently.

**Fix.** Measure the largest radial gap between the mesh and the field over a dense sample of the
home planet, per rung, and state the sink from THAT — as `column_bound_m` already does for the
column. Or add the slope term.

## 7. UNMEASURED — `morph_fallbacks` is counted and then thrown away

`crates/client/src/chunks.rs:85,528`. The only reader is a unit test (`< 1 %` on one chunk). The
count never reaches `DevTerrainStamp` (`crates/devproto/src/state.rs:425-460`), so the picture gate
cannot see it and no flight measures it. A fallback is a vertex that morphs to the FIELD instead of
the MESH — the failure mode that cost the 34 specks. Put it on the stamp with a ceiling the gate
asserts.

## 8. UNMEASURED — the parent cache is a FIFO of 48, and the census assumes no eviction

`crates/client/src/chunks.rs:313-345`, `PARENT_CACHE_ENTRIES = 48`.

`get` does not refresh the order and `insert` pushes only on a fresh key: the cache is a strict
FIFO, never an LRU. A hot parent leaves on schedule while its children are still building. A finer
chunk asks for up to EIGHT parents, and the workers run one job per thread over a ring of several
hundred columns, so the live parent set passes 48 easily. Every miss rebuilds a whole `sample_box`
plus `extract_all_edges`, which emits more triangles than a drawn chunk.

D-TERRAIN-5 item 2 states the optimistic cost — "a parent once per its eight children and their halo
users". Nothing measures the real one. Two threads may also build the same parent at once (`get`
builds outside the lock): correct, wasteful.

Worse, the cost instrument does not measure the shipped path.
`crates/bins/examples/terrain_cost.rs` calls `geometry_of`, which makes **a fresh private
`ParentCache` per chunk** (`chunks.rs:436-440`). Its per-chunk number is neither the lane's cost nor
the old slice-7 cost.

**Fix.** Make the cache an LRU, give the example the lane's shared cache, and put "parent builds per
finer chunk" into the M8-2 census.

## 9. DEFECT (small) — `forget(realm)` leaves that realm's parent meshes in the cache

`crates/client/src/chunks.rs:678-692`. `forget` drops the body, the chunks and the pending jobs, but
not the entries of `parents`, which are keyed `(realm, key)`. If the same `RealmId` is later stated
with a different seed, the geomorph reads a stale surface and no counter says so.

**Fix.** Drop that realm's entries in `forget`, or key the cache by the body's seed.

## 10. UNMEASURED — the straight-edged quad over-claims ground on the chart's near side

`crates/client/src/skyline.rs:126-148`, `ladder_view.rs:381`.

The corners are exact on the azimuthal chart. The EDGES are drawn straight. On this chart the true
geodesic between two corners at chart radius `d` and half-azimuth `α` passes at
`arctan(tan d · cos α)`, which is FARTHER from the origin than the chord `d cos α`. So the straight
quad claims a lens of ground the column does not own, on the side facing the eye.

For a chunk-sized column the lens is negligible. For a TOP-RUNG ROOT column — pass A raises a wall
for one whenever its nearest point lies inside the horizon — `d ≈ 1 rad` and `α ≈ 0.4 rad` give a
lens about 0.04 rad deep, which is 260 km at the eye's radius. A ray that crosses only the lens
raises a wall on an azimuth the column does not span.

The error is mostly safe, because the wall reads the elevation at a SMALLER central angle, and the
elevation rises with the angle up to the floor's tangent point. It is unsafe only past that tangent
point. Nothing measures it.

**Fix.** Shrink the quad toward its centre by the chord-to-arc sagitta before raising, or raise a
wall only for columns whose own angular size is small — the near ground, which is what hides things.

## 11. UNMEASURED — `ray_range` uses a flat-chart half-angle, which is too narrow far out

`crates/client/src/skyline.rs:115-124`. `beta = asin(r/arc)`. The true azimuth half-span of a cap of
angular radius `ρ` at central distance `φ` is `asin(sin ρ / sin φ)`, which is larger. At
`φ = 0.7 rad` — the orbit stand's limb — the true span is about 1.5 times the computed one. In
`clears` a narrow span reads the lowest wall over too few rays, which culls. The 1 % margin does not
cover 50 %. With finding 1, the orbit stand's rung-12 columns are judged over a disc that is far too
small.

## 12. NIT — the far edge discards on the MORPHED distance, the morph reads the UNMORPHED one

`crates/client-render/src/ladder_fade.wgsl:81,95`. The vertex stage reads `d = length(own)` (the
un-morphed vertex); the fragment stage discards on `length(in.world_position)` (the morphed,
interpolated one). The two differ by the morph's own displacement, so the fringe is discarded a few
centimetres before the two surfaces coincide. It is harmless only because the coarser rung is never
discarded on the near side. Say so in the shader's comment, or read one distance in both stages.

## 13. NIT — the shadow pass has no far edge, so both rungs cast the same shadow past `out_hi`

`crates/client-render/src/ladder_fade_prepass.wgsl` has no `discard`. Past `out_hi` the finer rung is
fully morphed onto the coarser mesh, so two coincident casters enter the shadow map. That is depth
fighting by construction, and a candidate source of the residual dark pixels. Add the same far-edge
test to the prepass fragment, or mark the chunk `NotShadowCaster` past its edge.

## 14. NIT (risk) — the mesh's bounding box is built from the UNMOVED positions

`crates/client-render/src/terrain.rs:385-400`. Bevy computes the `Aabb` from `ATTRIBUTE_POSITION`.
The vertex stage then moves every vertex by the morph and by up to `sink_m`, which at a coarse rung
is metres to hundreds of metres. A chunk at the edge of the frustum can be culled while its moved
geometry is still on screen. Nothing measures it. The four stands are still, and a hole at the frame
edge with sky above it is not a "hole" for `ground_holes`.

**Fix.** Set the `Aabb` explicitly, grown by `sink_m` plus the largest morph.

## 15. NIT — the prepass vertex stage fills only two fields

`crates/client-render/src/ladder_fade_prepass.wgsl:42-57` writes `world_position`, `position`,
`unclipped_depth` and `instance_index`. If the app ever turns on the normal prepass, motion vectors
(TAA) or deferred shading, `world_normal` and `previous_world_position` are never written and the
pass degrades silently. Guard them with the same `#ifdef`s Bevy's own `prepass.wgsl` uses, or state
in the comment that this material forbids those passes. (Decals, OIT and the visibility-range dither
are absent from the forward fragment too; none of the three is used today.)

## 16. DOC — `fade_weights` is dead, and its doc describes the DITHER that was refused

`crates/client/src/ladder_view.rs:108-116`. Nothing outside its own unit test calls it. Its
doc-comment still says "the material keeps the fragment while the pixel's threshold `t` has
`t < w_out` and `1 − t ≤ w_in`" — the complementary dither §15.2 records as MEASURED and REFUSED
(116 holes). The shaders compute `whole` and `risen` themselves. Delete the function, or state that
it is the shader's reference expression and pin it against the WGSL.

## 17. DOC — the module doc of `ladder_view` still describes step 2's skyline

`crates/client/src/ladder_view.rs:20-24`: "every column inside the horizon raises a wall — its
guaranteed floor over **the disc inscribed in its footprint**". The code raises the wall over the
QUAD (`skyline.rs:190`), and §15.2 records the inscribed disc as measured and refused ("nothing
culled, 8 779"). One paragraph, two designs.

## 18. DOC — "every crossing of the parent's box is in it, owned or not" is not what the code does

`crates/terrain/src/extract.rs:427-436` with `ring_fits` at `extract.rs:396-400`.
`extract_all_edges` takes an edge only when the chunk owns it OR its four groups fit in the box.
Edges at the outermost halo lateral position are not emitted. The mesh still reaches cells −1 and 62
through the rings of the edges at 0 and 61, so the geomorph works. The sentence claims more than the
code gives.

## 19. NIT — the bucket lookup rests on an unstated HALF-CELL alignment

`crates/client/src/chunks.rs:191-194` (`cell_of` on the parent's quanta) against `chunks.rs:420-441`
(`parent_cell` on the finer vertex's quanta). A vertex's world position is the trilinear blend of
CELL CENTRES (`crates/terrain/src/position.rs:34+`), so quanta `v` in cell `C` sits at the face
parameter `C + 0.5 + frac`, not at `C + frac`. The query cell from `parent_cell` therefore maps to
the true angular interval `[C + 0.25, C + 1.25]`, while bucket `C` holds triangles covering about
`[C − 0.5, C + 1.5]`. It fits, with a quarter of a cell to spare, ONLY because every surface-nets
triangle spans two adjacent cells and is bucketed in all of them. The doc says "exact in integers",
which is true of the integers and silent about the alignment that makes the lookup sound. State the
invariant, and pin it with a test at the extreme query offsets.

## 20. NIT — a tie in the morph fold is resolved by the parent list's ORDER

`crates/client/src/chunks.rs:249-268` and `486-500`. Both folds keep the first hit when `|h − len|`
ties. Two neighbouring finer chunks build their parent list in a different order (`parent_keys`
iterates `[0, side(key.x)]`, and `side` flips across the boundary). Two DIFFERENT hits at exactly
equal distance would then give two targets for one shared vertex. It is vanishingly rare in floating
point and free to remove: break the tie on the hit's own value, not on the order.

## 21. UNMEASURED — the gate is weaker than step 2's, in four named ways

`crates/bins/tests/terrain_pictures.rs`.

- **The rung rule is now a SHARE, not an equality.** Step 2 asserted every terrain pixel's rung
  equalled the flag's rung. Step 3 asserts `|rule − rung| ≤ 1` per pixel and `exact_share ≥ 0.9`
  outside the bands. One pixel in ten may sit on the wrong rung outside every band for ever, and the
  gate stays green. The measured shares (0.999 / 0.966 / 0.938) leave little headroom aloft: the
  floor is 0.9 and the measurement is 0.938.
- **"Zero hole pixels" is not "no missing chunk".** `ground_holes`
  (`crates/client-harness/src/probe.rs`) counts a `NONE` pixel only when TERRAIN stands ABOVE it in
  the same column. A missing chunk at the TOP of the drawn ground in its column reads as sky. The
  horizon-row test is the only guard there, and it allows the whole gap down to the horizon of the
  LOWEST ground — 28 rows on the ground stand, by the gate's own comment. It also accepts a RULER
  pixel as "drawn".
- **A missing FINE chunk over a drawn coarse one is invisible to every assertion.** That is the
  design (the sunk coarser shows instead of a hole), but the gate then cannot tell "the crossfade
  works" from "the finer rung never arrived". Both give zero holes, and the coarse pixel reads a
  self-consistent rung. Only the rung share sees it, with 10 % of slack.
- **All four stands are STILL, at mid-face ground, on one planet.** No stand looks along a cube-face
  seam (finding 5), none moves (D-TERRAIN-5 item 4), and none re-takes the picture after the eye
  steps 0.5 m — which is where the skyline recompute, the release hold and the arrival of a finer
  rung over a sunk coarser one all live.

## 22. UNMEASURED — the arrival of a finer rung is a designed POP, with a size

`crates/client-render/src/terrain.rs:522-560`, with the sink. While a finer chunk builds, the sunk
coarser rung shows, `sink_m` metres BELOW the surface the finer rung will draw. When the finer lands,
the ground rises by that much in one frame. `sink_m` at a coarse rung is the dropped octaves'
amplitude plus two cells — metres to tens of metres. SL8 calls a jump a seam. The doc lists this as
"a measurement on a moving eye"; it deserves naming as a POP with a stated size.

## 23. NIT — HR5 risk: a guard arm real terrain may never take

`crates/client/src/chunks.rs:258-263`. The arm `(Some(b), Some(h)) if (h−len).abs() < (b−len).abs()`
needs TWO hits in one bucket of one parent, the second one nearer — a fold or a cave under the
queried radial. If the golden chunk has none, the guard's false path is an uncovered branch in
`radial_hit_m`. (The fold in `geometry_with` is safe: two overlapping parents give EQUAL hits, which
takes the `(b, _)` arm.) Add a hand-built two-triangle mesh test on one radial.

---

## What I tried to break and could NOT

- **The band arithmetic.** `fade_bands` gives rung `L` the fade-in band `band(L−1)` and the fade-out
  band `band(L)`, and `switch(L) = 2·switch(L−1)`. So `in_hi(L) = out_hi(L−1)` exactly, and the two
  bands of one rung never overlap (`1.1 s_{L−1} < 1.8 s_{L−1}`). Rung 0's always-in and the top
  rung's always-out survive the f32 narrowing (1e30 and 2e30 are finite) and clamp correctly.
- **The sink never lets the coarser rung through INSIDE the band.** In the shared band the finer
  moves toward the coarser mesh by fraction `t`, and the coarser rises from `−sink` by the SAME `t`
  (one band, one ramp). The gap is `(1 − t)·(fine − coarse + sink) ≥ 0` wherever `sink` bounds
  `|fine − coarse|`. This is correct at the vertices; finding 4 is about the surface BETWEEN them.
- **The parent SET is symmetric across every same-face seam.** For two finer chunks that differ by
  one in a single coordinate, the `side()` parity gives them the SAME parent pair on that axis
  (`2m+1` faces `+`, `2m+2` faces `−`, both name `{m, m+1}`), and the parents they do not share lie
  outside the shared vertex's cell range and hold no bucket for it. Corner and edge neighbours check
  out the same way. The seam claim holds inside a face.
- **The triangle bucketing is complete.** A triangle goes into the whole rectangle its three
  vertices' cells span. Every point of a triangle has an `(a, b)` between the vertices' minima and
  maxima, so no cell of a long thin triangle is missed.
- **`elevation_rad` and the tangent-point logic in `clears`.** The elevation rises with `φ` up to
  `acos(r/e)` and falls after it, and it rises with `r`. Taking the maximum of the near edge, the far
  edge and the clamped tangent point IS the maximum over the span, and only when `peak < eye`. For
  `peak ≥ eye` the function falls monotonically and the near edge wins, which is what the code takes.
  The mixing of an eye-radius arc with a peak radius is CORRECT: `arc_m / eye_m` is a pure central
  angle. Only the disc's own radius is wrong — finding 1.
- **The exit's 1 % shortening is safe where it matters.** Shortening moves the reading to a smaller
  central angle, where the elevation is lower, as long as the exit lies before the floor's tangent
  point. Every column that raises a wall lies inside the eye's horizon, and the tangent point of a
  floor BELOW the eye's surface is farther than that horizon. Where the eye stands over the quad,
  every angle up to the exit is on the plateau, so the reading is valid outright.
- **The floating origin.** `camera_transform` puts the camera at `Vec3::ZERO` in both the windowed
  and the capture paths (`crates/client-render/src/lib.rs:1439-1442`, used by `place_camera` and by
  `frame_scene_camera`), and `place_chunks` places every chunk at `draw_center ⊕ facing·origin`
  against the same eye. So `length(world_position)` IS the distance from the eye in all three
  stages, the shadow pass included.
- **`quad_holds_origin` handles either winding** (`positive | negative`), so a face whose chart image
  is mirrored still reads correctly. The `±π` seam is handled by `rem_euclid` in both `raise` and
  `clears`, and a test covers a ridge due south.
- **The wanted set is a superset of what the shader draws,** in the direction that matters. The
  descent's `near` and `far` come from the surface sphere and the circumscribed disc, and the error
  from reading the eye's own surface height for another column's radius pushes a column toward being
  wanted, not away from it — except past a rung's fade-out end, where the coarser rung already covers
  the ground.
- **`extract_all_edges` cannot walk out of the box.** Every owned edge also passes `ring_fits` (an
  owned edge's `u` and `w` are core coordinates), so the `every_edge` arm only adds edges and never
  reaches a group index outside the map.
- **The ruler ball** carries its own positions as its morph target and zeros as its sink, and its
  probe material carries the always-in and always-out bands. It never morphs, never sinks, and is
  never discarded.

## The answers (2026-09-09, after the refutation)

Every finding is answered below. FIXED = the code changed and a test or a flight measures it.
KEPT = the code stands, with the reason. OWED = registered in `DEFERRED.md` with its step.

1. **FIXED.** A column is now a CAP: `ColumnGeometry` states `phi` (the centre's central angle,
   `arc / eye radius`) and `rho` (`asin(chord / surface)`), and `Skyline::clears` takes angles. The
   azimuth half-width is the sphere's own, `cap_half_width(phi, rho) = asin(sin ρ / sin φ)` (a half
   turn when the cap holds the foot or the ratio passes one). Tests: `cap_half_width` at the foot,
   past the antipode's meridian, and against the flat reading (wider, as finding 11 states).
2. **FIXED.** `raise` takes its ray span from the CORNERS' azimuths about the centroid's (a convex
   quad's azimuth extremes are corners); the dead `ray_range` is deleted. From the ground a 62 m
   column at 1 km now walks about 36 rays, not 3 600. The recompute on a moving eye is owed to
   M8-1 (D-TERRAIN-5 item 4) with the measurement.
3. **FIXED.** The wall's floor is the DRAWN ground's: `sampled_low − column_bound − dropped_bound −
   cell(rung) − sink_m(rung)`. A cell for the extractor's placement, the sink for what the mesh may
   stand under while a finer rung is drawn over it.
4. **FIXED.** The sink ramp ends PAST the fade-in edge (`ladder_view::sink_end_m`): at the edge one
   finer cell of sink remains, so the coarser mesh never stands over the finer chord across a
   crease; past the edge only the coarser is drawn and it rises to its own surface with distance,
   continuously. The uniform's second slot carries the ramp's end; the shaders' `risen` reads it.
   The "faster ramp" sentence is deleted from §15.4 and D-TERRAIN-5.
5. **FIXED, one way.** A vertex on or past the face's edge reads the coarser FIELD from both faces
   (`on_seam` in `geometry_with`, counted as `morph_seam`); the parents stay one face's. Both
   sides read one target, at the cost of the field-vs-mesh crease along the twelve cube edges —
   the residue finding 4 names, over one row of vertices. Test: the chunk at `x = 0` has seam
   vertices whose targets stand on the rung-1 field; a middle chunk has none. Reading the
   neighbouring face's parent mesh (the exact cure) is owed: D-TERRAIN-5 item 8.
6. **MEASURED.** `the_parent_mesh_stands_within_the_sink_of_its_field_on_every_radial`: over the
   radials of four chunks (rungs 0, 1 and 5) the parent mesh's hit and the coarser field stand
   apart by less than a cell of each rung — the two cells the sink adds to the gap bound. The
   slope term the finding asks for is not needed by the measurement; if a stand ever shows the
   coarser through the finer, this test is where the bound is re-read.
7. **FIXED.** `DevTerrainStamp` carries `morph_fallbacks`, `morph_seam` and `vertices`, summed over
   the drawn chunks; the gate asserts fallbacks ≤ 1 % of vertices and prints the three. A fallback
   is now a SURFACE vertex (within the sink bound of the coarser field) without a parent hit; a
   cave's own vertex reads the field by nature and is not counted (MEASURED: the edge chunk at
   `x = 0` has 1 475 cave vertices and 5 surface fallbacks of 9 998).
8. **FIXED / OWED.** The cache is an LRU: a hit moves its key to the back (tested: a refreshed key
   survives the cap, an untouched one leaves). The census of parent builds per finer chunk, and
   the concurrent double build, are owed to M8-2 (D-TERRAIN-5 item 2). `terrain_cost` still
   builds with a private cache per chunk: it measures the old slice-7 cost, and is retired by M8-2.
9. **FIXED.** `ChunkLane::forget` drops the realm's parent meshes (`ParentCache::forget`, tested
   with two realms).
10. **FIXED.** A wall is raised only for a column whose angular radius is under
    `WALL_MAX_HALF_ANGLE` (0.02 rad, about 127 km): the near ground. A top-rung root raises none;
    its floor was loose anyway.
11. **FIXED** with finding 1.
12. **KEPT, documented.** The far edge is read on the morphed fragment, the morph on the unmorphed
    vertex; the fringe ends a little early, and the coarser rung stands there. The shader says so.
13. **FIXED.** The base material is `AlphaMode::Mask(0.5)` (the paint is opaque, nothing is ever
    cut by alpha), which makes the engine run the extension's prepass fragment stage; that stage
    discards past the far edge, so past its edge a rung casts no shadow either.
14. **FIXED.** Each chunk and its twin carry an explicit `Aabb` grown over the positions, the
    morph targets and the positions less the whole sink (`moved_bounds`).
15. **FIXED.** The prepass vertex writes `world_normal` under `NORMAL_PREPASS_OR_DEFERRED_PREPASS`
    and `previous_world_position` under `MOTION_VECTOR_PREPASS`; the fragment writes the normal and
    the emulated depth under their defines.
16. **FIXED.** `fade_weights` is deleted with its test; the module doc no longer names it.
17. **FIXED.** The module doc names the quad and the lowered floor.
18. **FIXED.** The doc of `extract_all_edges` now states the rule: every crossed edge the chunk owns
    or whose four groups fit in the box.
19. **KEPT, documented.** The half-cell alignment is stated at `parent_cell`; the bucketing over
    the vertices' whole cell span is what makes the lookup sound, and the parent-mesh test reads
    every own vertex's radial at arbitrary quanta.
20. **FIXED.** Both folds break a tie on the hit's own value (the lower one). Test: two triangles
    on one radial, in either order, answer the same.
21. **KEPT, registered.** The share floor stays 0.9 (a column straddling a band's edge is drawn
    whole; the partition by fragment was measured and refused), the horizon-row slack stays (no
    tighter horizon model exists yet), a missing finer chunk cannot occur in a still (the gate
    waits for zero pending, so every wanted chunk is resident), and the still stands are
    D-TERRAIN-5 item 4. The cube-face seam stand is owed with item 8.
22. **REGISTERED.** D-TERRAIN-5 item 4 names the arrival of a finer rung over a sunk coarser one
    as a POP of `sink_m` metres, for the pop detector (step 6).
23. **FIXED.** `two_triangles_on_one_radial_answer_the_nearer_one_and_a_tie_the_lower` covers the
    guard's both paths.

**After the answers, one more flight.** The nineteenth flight (with the fixes above) left ONE pixel
of nothing on the hill at 970 m, the same pixel on a second flight: where four rung-1 chunks meet.
Not a crossfade defect — a hairline crack from the engine's single-precision rounding of each
chunk's own origin plus its own offsets, which two neighbours do not round alike. The cure is the
standard one: `chunks::add_skirts`, a strip two cells deep under every boundary edge, facing
outward, morphing and sinking with its edge. Measured on the twenty-second flight.
