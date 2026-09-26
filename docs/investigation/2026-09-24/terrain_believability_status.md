# Terrain believability — THE LIVING STATUS of point 8

> **A LIVING DOCUMENT (ruling B6, 2026-09-24).**
> - **Update rule.** Every change that touches an item in this file updates this file IN THE SAME CHANGE: flip
>   the state, correct each number that moved, and add one line to the change log (§12). A reader must never be
>   misled by this file. When the code and this file disagree, the code wins and this file is the defect.
> - **Delete rule.** Delete this file when every step in §8 is done and the owner gave G-LOOK. The rulings
>   (`docs/design/owner_decisions_*.md`) and `docs/design/DEFERRED.md` (D-TERRAIN-8, D-TERRAIN-9) keep the history.
> - **Written** 2026-09-24 on branch `game` at `e69ed91`: GENERATOR_VERSION 19 (`crates/terrain/src/tag.rs:174`),
>   ARTIFACT_VERSION 7 (`crates/terrain/src/artifact.rs:77`). **Updated for step 2** (2026-09-25, on `a985e5c` plus
>   the uncommitted step-2 change): GENERATOR_VERSION 20, ARTIFACT_VERSION 8, `PROTO_MINOR` unchanged. **Updated for step 2b**
>   (2026-09-25, on `4442fa7` plus the uncommitted step-2b change): GENERATOR_VERSION 21, ARTIFACT_VERSION 9, `PROTO_MINOR`
>   unchanged.
> - **How it was made.** Nine read-only agents (workflow `wf_1780e367-7e9`): five readers, one synthesis, and one
>   skeptic per complaint who tried to refute each cause from the code. The key lines were then read again by hand.
>   Nothing was built and nothing was measured for this file.
> - **Marks.** `path:line` = read in the code. DOC = a number from a ruling file, not re-measured. ARITH = hand
>   arithmetic. UNMEASURED = nobody printed it.

---

## 1. The owner's words and the target

The owner flew the home planet (2026-09-24) and said the terrain is not believable:

1. **The dunes** — *"the land look like dunes everywhere"*.
2. **The teeth** — *"shores are squared like teeth"*.
3. **The water positions** — *"water positions for lake and rivers is not believable"*.

**THE TARGET PICTURE** — *"we need to have this quality of the terrain and world"*, and *"please make sure that
visually we are moving towards this target picture"* (ruling B6). The picture is the owner's photograph of a
Crimson Desert vista, kept OUTSIDE the repo at `/Users/maxim/Downloads/crimson desert.jpeg` (the repo is public, and
the picture is a photograph of a commercial game). It is read as numbers in
`docs/investigation/2026-09-08/landforms/01_reference_target.md` §2, and its skyline is traced by our own rule in
`docs/investigation/2026-09-08/landforms/11_reference_trace.md`. §9 of this file is the checklist that keeps every
step pointed at it.

**Example.** From a stone pillar, the reference shows snow on a ridged range 30–60 km away, valleys 1–5 km apart
that join into drainage, forest over whole hillsides, and one lake in a valley. On the home planet today, a pilot at
120 km sees sand-coloured ripples, a coast of straight 8 km steps, and square ponds.

---

## 2. The short answer

- The arc stands inside 8d. Ruling B1 holds 8e until the owner says the belt, the coast and a plain look like a
  planet.
- **One root under all three complaints:** the solve knows the ground only on nodes 8 192 m apart. Every hill,
  valley, shore and lake edge under that node is either noise (the ground) or a whole node square (the water).
- The cure for the dunes (B2 step 4, the drawn rivers) was built and then RETIRED by W16. Its successor, the
  survey's R2 (one eroded level shipped in the artifact), is NOT built. Only its bench (R1) exists.
- **Step 1 MEASURED all three (2026-09-24, generator 19; the lines in `step1_census_gen19.txt`):** the shipped belt
  holds 2.84 closed basins ≥ 10 m deep per 100 km² against the solve's 0.003, with Hack's n = 0.22 (RED); the drawn
  shore stands a median 2.9 km off the ground's own waterline with a median 309 m wall on it; 345 of 506 lakes vanish
  whole at the rung 9 → 10 swap, and 21 130 of 37 966 rivers end in a dry closed basin (§5–§7 "MEASURED").
- **Step 2 BUILT (2026-09-25, generator 20; ruling B7; the lines in `step2_census_gen20.txt`; NOT committed):** the sea
  is the CONNECTED ocean and a pocket is a closed basin; every hollow takes its whole watershed and passes its overflow
  on; the discharge is the runoff over the final receivers; a small lake keeps its level at pyramid level 1. MEASURED:
  lakes 3.51 % of the land (was 0.068 %), 0 lakes change at the rung 9 → 10 swap (was 427 of 506), 0 lip columns (was
  3 753), the discharge row agrees with its tree at every node (was 1 304 002 off), 0 pocket nodes drawn as sea (was
  59 470). NEW REDS it exposed (§4, §7): G-RIVER-DESCENT (3 430 trunk steps rise inside filled hollows), G-GRID worse
  (1.484 against 1.284) on the filled pockets' flats, G-RUNG-HOLD (19 columns at 9 → 10; 5 after the review fixes),
  and far-rung lake losses at 14 → 15 and 15 → 16.
- **Step 2's review fixes (2026-09-25, NOT committed; §7 "MEASURED (step 2's review fixes)"):** a pocket's dry floor is
  its own side word (the coast mask's word 2), so no pocket floor is lifted into a mesa at any rung (G-POCKET-HELD 0 of
  60 383 columns); a far chunk waits for the coast mask; a floor under the sea is no beach; the connected sea states
  an answer that does not balance; the cascade skips full rings; the lakes' sea fixed point has its own tests. The
  discharge row still carries 1 607 km³/yr to the sea that the budget holds (ROW-OUT-OF-CLOSED-HOLLOWS): the owner's
  word on question 3 B is owed. (Given in B8 item 2; built as step 2b below.)
- **Step 2b BUILT (2026-09-25, generator 21, artifact 9; ruling B8 item 2; the lines in `step2b_census.txt`; NOT
  committed):** the shipped receivers are THE WATER'S OWN WAY and the discharge row is THE BUDGET'S OWN WATER (§7 W10,
  W11). MEASURED: G-RIVER-DESCENT 0 rising steps (was 3 430); the row into the sea is the cascade's to the unit,
  3 851.69 km³/yr (was 5 447.10 against 3 839.73); 0 of 2 571 closed hollows that meet the sea leak (was 4 544 of
  5 322); every closed body's ends carry what it holds (ROW-HELD 0 of 6 408); W10-DISCHARGE-TREE 0. The solved ground
  did not move (0 of 8 871 936 heights, `artifact_diff`). THE ROW IS EXACT because the cascade now pours by the
  saddle's steepest open side and follows the ground from a saddle toward the sea (plan questions 1 and 2, choice A):
  MEASURED, the variant with only those two (V3, `step2b_census.txt`) is already exact. Beside them a flat is swept from its drains
  (question 3, choice A) — for item 1's flat directions and to close the 8 795 leaves of zero depth (pit at the
  spill's height) that held 462.7 km³/yr on flats that drain, NOT for the row — and a node at its hollow's spill height costs that hollow nothing (a law
  the plan did not name; §7 W11). ★ **THE PLAN'S STOP CONDITION TRIPPED:** G-LAKE-SHARE **4.828 %** against the 4 %
  ceiling (3.510 % at generator 20) — the owner's word before any commit (§11 item 10). ★ **AND ITEM 1's FLAT LAW IS
  NOT MET** (found by the review, 2026-09-25): the directions over a filled flat and across a spilling lake lean on the
  lattice — G-GRID-VALLEYS-BY-CLASS flats 1.470 and spilling lakes 1.553 against ~1.006, G-LAKE-WALK 14.07° against
  test T9's 12°, both RED; like for like against V3 the flat sweep made G-GRID-VALLEYS WORSE (1.093 → 1.205; the
  1.484 of generator 20 is the routing's tree, not a control) (§11 item 11).

---

## 3. Where point 8 stands (code-verified)

| stage | state | evidence |
|---|---|---|
| 8a the spectrum | BUILT. The ridged band is ISOTROPIC. The flow stretch that the code promises to 8c is NOT built. | `terrain/src/body.rs:52-137`, `:128-130`; `recipe/src/height.rs:439-446` |
| 8b the charter | BUILT. The T8 asks are ASSUMED. | `physics/src/worldgen/charter.rs` |
| 8s the sky | BUILT (the air, the aerial perspective). The sun's shadow map is OFF when the eye stands more than 3.5 km over the highest ground the body can have. | `client-render/src/terrain.rs:109`, `:3339-3353` |
| 8c the solve | BUILT (C1–C5): 40 passes on the 8 192 m lattice, 1 216 nodes a face edge, 8 871 936 nodes. C6 (the owner's look) owed. | `terrain/src/solve.rs:1505-1720`; `terrain/src/macro_lattice.rs:48` |
| 8d W4-1 each bed's hardness | BUILT. The cap share (a half) is ASSUMED (W5). | `recipe/src/terrace.rs:152` |
| 8d W4-2 the province word | BUILT. The solve reads the erodibility; the column kernel does not. | `terrain/src/land.rs:483`; `terrain/src/strata.rs:214`; `terrain/src/solve.rs:980` |
| B2-1 the lakes from a budget | BUILT; ★ the WATERSHED and the CASCADE by plan step 2 (Fill–Spill–Merge; §7 W3, W4 cured, generator 20). ★ Step 2b (generator 21): the shipped receivers and the row follow the budget's own water, the cascade pours by the saddle's steepest open side, a flat is swept from its drains (§7 W11). | `terrain/src/lakes.rs` `budget`, `descent`, `cascade`, `open_side`, `flats`, `water_tree`; `terrain/src/solve.rs` `follow_the_water` |
| B2-2 the summer ice line, the crater age | BUILT. No drawn reader reads the ice bit; the snow in the pictures comes from the placeholder biome. | `terrain/src/climate.rs:560`; `terrain/src/craters.rs:360`; `recipe/src/height.rs:501-529` |
| B2-3 the flats, the scratch instruments | PARTIAL. Flats by summed chord; ★ since step 2b the ground's own flats are swept from their drains and a spilling lake is crossed by Rho8's split. D8 stays. G-GRID RED (1.205 in all; by class the flats 1.470 and the spilling lakes 1.553, RED — step 2b's flat law is not met, §11 item 11). | `terrain/src/solve.rs:784-934`; `bins/src/grid_bias.rs` |
| B2-4 the river lines, the carve, the floodplain | BUILT, THEN RETIRED (W16.1). The width law has no caller outside its tests. R1 is a bench. R2, R3 NOT built. | `terrain/src/river.rs:1-26`; `recipe/src/amplify.rs:1-6` |
| B2-5 the lake side word | BUILT (two bits a node); ★ the level-1 word keeps a lone lake (`block_water`, step 2 part 5). The lake's EDGE law (W6) and the sea's shore off the node squares are OWED. | `terrain/src/artifact.rs` `coast_side`, `sample_side`, `block_water` |
| W18 the water sheet | BUILT (client geometry; it draws what the columns say). | `terrain/src/position.rs:271-499` |
| 8d the authored override | NOT BUILT. | no code |
| 8e biome and soil | NOT BUILT. Every ground pixel is ONE colour, `srgb(0.55, 0.50, 0.42)`. | `client-render/src/terrain.rs:1974` |
| 8f the third dimension (cliffs, overhangs) | NOT BUILT. | no code |
| 8o the ocean | NOT BUILT. The water is one colour, `srgb(0.06, 0.24, 0.42)`. | `client-render/src/terrain.rs:2034` |

**Example.** The shard solves the home planet once, on 8 871 936 nodes. The client adds the fine octaves over that
field. What the solve knows stops at the node. Under the node, only the noise stands.

### 3a. THE CLIENT READ THE EYE'S GROUND FROM THE RECIPE, NOT FROM THE ARTIFACT (found and cured 2026-09-24)

Found on the reference stand's first flight: the eye stood 400 m over the solved ground (8 306 m over the ladder
radius), yet the stamp said 7 054 m, the horizon 299 km (the truth ~71 km) and the finest rung drawn was 3. The ladder's
descent (`client/src/ladder_view.rs` `wanted`), its band probe, the eye's ground truth (`client/src/chunks.rs`
`eye_surface`, the HUD and the stamp) and the ruler (`ruler_on_surface`) all read `vd_terrain::height::height_m(body,
dir, 0)` — the recipe's OWN relief, which the artifact REPLACED — even while the client held the artifact. On the
reference stand the two stood 6.6 km apart; on the belt the eye read 155 km over the recipe while it stood 150 km over
the ground; wherever the solve lowered the ground the error runs the other way.

**CURED, then REVIEWED and reworked, committed 2026-09-24** (workflow `wf_dca58f2b-13d`, three lenses, a skeptic per finding):
- `vd_terrain::height::drawn_height_field_m` is THE SURFACE THE CHUNKS DRAW: the ground over its water, the water sheet's
  own radius under it (`position::water_radius`, the sheet's one rule) — because over the sea the chunks drop the floor
  and draw the sheet, and the first cure read the FLOOR there (up to ~7 km under a hull over the deep sea: the review's
  confirmed medium finding). A terrain test walks the moon's stated coast: dry columns read the ground, wet ones the sheet.
- `vd_client::chunks::drawn_field_m(body, artifact, dir, rung)`: the recipe for a realm with no artifact (or a body with
  no macro lattice); else the field the rung reads, else the finest whole pyramid level held; `None` where nothing is drawn
  yet. `drawn_surface_m` is it at rung 0 with the recipe as a STAND-IN for the head-only window (so the ladder never
  releases the realm's chunks); the ruler plants NO ball where nothing is drawn.
- `LadderView::wanted` and `probe`, `eye_surface` (the HUD and the stamp) and `ruler_on_surface` read it; the renderer
  passes each realm's artifact; the ruler's cache is keyed on the realm, the reach and the artifact's head and epoch (it
  kept a ball from before the tiles landed on a still stand: confirmed).
- Tests (vd-client 334 pass, vd-terrain 211 pass, clippy `-D warnings` clean on terrain, client, render and bins): every
  cache state with discriminating readings; the ruler on the tiles, the recipe, the level and nowhere for a head alone;
  AND TWO GUARDS OF THE DEFECT ITSELF — the ladder's altitude and the probe's horizon read over the drawn ground, and an
  eye on the moon's most-raised solved stand asks rung 0 where the recipe's reading asks a coarser ring.
- OWED: the ruler's march now reads the whole field per sample (the review's arithmetic: ~3-6 µs a sample, up to ~12-25 ms
  on a frame whose camera moves, capture mode only — the probe build), UNMEASURED; the cure is one field and charter per
  march and a node memo, measured on the stamp's work row. And the visible depth: the sheet stands where the extractor's
  step cuts the floor today; 8o's clarity moves it, and this seam must follow.
- ★ PROVEN IN FLIGHT (2026-09-24 19:17, the reference stand, the rebuilt client, `pictures/reference_after_eye_fix.png`):

  | the stamp | before | after |
  |---|---|---|
  | the ground under the eye, over the ladder radius | 1 654 m (the recipe) | **8 306 m** (the solved ground) |
  | the eye's altitude | 7 053.6 m | **401.5 m** |
  | the horizon | 299.2 km | **71.4 km** |
  | the rungs drawn | 3 – 11 | **0 – 9** |
  | the ruler | 31.7 km off at rung 6, buried under the solved ground | **2.06 km off at rung 2, on the drawn ground, its shadow cast** |

  No hole (`hole_columns` 0, `margin_missing` 0). The ladder now draws 16 962 chunks there against 10 410, the finer
  rungs it owed; the HEADLESS capture's frame read 45.6 ms against 22.2 ms (the shadow map is on at a true 400 m) — a
  capture frame, not the window's: the window's cost of the finer near rungs is UNMEASURED.

### 3b. THE DARK TEARS AT A RING'S FAR EDGE — CLOSED CAVES SHADED AS CAVES (found and cured 2026-09-24)

The owner, in the window after 3a, low over the belt's highland: *"the underneath of the first rung is visible when
flying"* — dark torn pieces along a line near the hull. 3a made rungs 0-2 appear there (the ladder had read the eye
6.6 km too high), and with them a defect the ladder always had at every ring's far edge.

MEASURED, one refutation at a time (the 494 m stand, `pictures`-style capture and probe):
- not the shadow: the dark rung-0 pixels at the edge stand with the shadows off (17 against 18) and with every chunk
  casting its own (18);
- not a skirt or a fold seen from behind: the ground material culls back faces (a hole, never a dark piece);
- not the ground's own geomorph: over 1.3 million ground vertices per band, no morph over 0.41 m and no shade turn
  over 3° (`bins/examples/morph_normal_probe.rs`);
- IT: `bins/examples/ray_speck.rs` casts the capture camera's own ray through the chunks the client builds, placed as
  the shader places them. At every speck the first kept front face is rung-0 GROUND in the right place (≤ 3 cm) but
  SHADED 47-75° off the vertical (lit −0.70 to +0.02 against the ground's +0.40), coincident with a correct copy;
  its corners stand 26-62 m UNDER the ground in the same column: a CAVE's own roof and walls. Rung 1 holds no cave
  there, so the cave's vertices morph onto the coarser FIELD — the ground — by design, and at the ring's far edge
  (956 m) the closed cave lay on the ground with its OWN normals and won the depth test in dark tears (and rung 1's
  at 1.9 km, rung 2's at 3.8 km — the pattern an older note in the code had recorded).

A first cure (close a cave TWO CELLS UNDER the field) was REFUTED by its own test: 100 steep-riser SURFACE vertices of
the test box also stand past the reach from the field, and it would have dented the ground. THE CURE (committed with
this file's next change): a vertex that morphs onto the coarser field takes the coarser surface's SHADE there — the
parent triangle its radial meets nearest the target within the reach, else the coarser field's own normal
(`vd_client::chunks::field_normal`) — so a closed cave shades like the ground it closes onto; no position moves.
MEASURED after, same stand, same sun: the dark rung-0 pixels at the far edge 18 → 0 (the 10 dark ground pixels left
are the ruler ball's own shadow edge, before and after); `ray_speck` on the former specks: lit +0.38 to +0.43.
Tests: the cave vertex of the morph-target test shades as the ground; the field normal faces out and answers nothing
for a head alone (vd-client 335 pass, clippy clean). ★ The owner's window test (2026-09-24): *"I don't see it in
the ring part"* — CONFIRMED.

### 3c. THE CRACKS IN FAST LOW FLIGHT — SKIRTS SHORTER THAN THE COARSER SINK (found and cured 2026-09-24)

The owner, after 3b: *"I see cracks from time to time, only when I fly veeeery low"* — a long straight dark line with
light bits, at 29 km/s and ~670 m. The window's samples: 400 to 2 450 wanted chunks URGENT at a time in that flight.
Where a finer chunk is still building, the coarser rung shows in its place SUNK under the finer ground by its sink, so
at the edge of a drawn finer chunk with a missing neighbour the ground steps down, and only the finer chunk's skirt
closes the step. MEASURED (`bins/examples/skirt_gap.rs`, the 494 m eye): the skirts hung two cells and the coarser
rung sank more, so the skirt bottoms stood OVER the sunk coarser ground by up to 1.50 m (rung 0), 3.37 m (1), 7.98 m
(2), 25.36 m (3) — the sink less the skirt, a crack a grazing eye sees into.

CURED: `vd_client::chunks::skirt_drop_m` — a skirt under a sky-facing edge hangs as deep as the coarser rung's sink
plus one cell (4.45 m, 9.25 m, 19.63 m, 47.98 m at rungs 0-3), never under two cells; the top rung two cells.
MEASURED after: 0 skirt bottoms over the sunk coarser ground at rungs 0-2 (worst 0.95 m, 1.88 m, 0.45 m UNDER it).

THE OWNER'S QUESTION — *"not break the future terrain editing?"* — answered and acted on. A skirt is client drawing
only: the block store, the server's collision and every edit read the shape, and an edit rebuilds its chunk and
skirts. But a skirt always hung radially DOWN, so under a cave's roof or a dug tunnel's ceiling at a chunk's border it
hung into open space as a curtain. Now a skirt hangs INTO THE ROCK its edge stands on: the deep drop down under a
sky-facing surface, only two cells UP over a ceiling. A first form (the deep drop both ways) was REFUTED by measurement:
ceiling skirts of rungs 2-3 rose up to 62 m toward the sky through shallow caves' roofs. MEASURED after: no skirt vertex
stands over its own drawn ground at any rung (the highest 4.44 m, 9.23 m, 16.13 m, 9.70 m UNDER it). Tests: the drop's
law at every rung and the top rung's two cells; a ceiling triangle's skirt rises and a ground's falls; every skirt of a
built box hangs the drop down or the ceiling's rise up (vd-client 337 pass, clippy clean). OWED for slice 9 (editing):
a ceiling skirt of two cells can still pierce a roof THINNER than two cells (a tunnel dug just under the ground) —
the cure is to clamp each skirt inside the solid cells of the chunk's own samples; and a pick (which block a player
points at) must read the shape, never the drawn mesh, because a skirt is not ground. ★ The owner's window flight
(2026-09-24): *"Yes, that is fixed"* — CONFIRMED.

**STILL OWED — THE PICTURE GATE IS STALE SINCE 8c.** `bins/tests/terrain_pictures.rs` (last changed at Step 22, before
the solve) seats its stands over the recipe's relief (`height_m(&body, dir, 0)`) and judges against references frozen
before the solve; on the solved world a "1.8 m" stand can stand kilometres under the ground. Its M8-4 recompute now
passes no artifact and keeps that meaning. Owed: seat the stands on the solved ground, recompute with the artifact,
and re-freeze the references under the owner's look. Whether it passes today is UNMEASURED.

---

## 4. The gates of ruling B3

| gate | state | last number |
|---|---|---|
| G-LAKE-SHARE | **RED since step 2b** (the owner's word owed, §11 item 10) | **4.828 %** of the land (every node that is not the ocean; MEASURED gen 21); 3.510 % at gen 20, 0.068 % at gen 19. The attribution, MEASURED by census runs with one change at a time (§7 W11): the water's tree over gen 20's own budget 3.510 %; + the cascade's steepest open side 4.162 %; + the flats swept from their drains 4.741 %; + the brim 4.828 % |
| G-LAKE-COUNT | read, verdict not restated | **3 868 patches**, median 1 node, p90 11 nodes, the largest 12 562 nodes (843 022 km²) (MEASURED gen 21); 3 883 and 6 398 nodes at gen 20, 506 at gen 19; the power-law fit UNMEASURED |
| G-BASIN | GREEN | the water bodies DRY 5 127 (58.7 %), PARTIAL 1 281, SPILLING 2 319 over 10 694 leaves (the 8 795 leaves of zero depth are gone); the books EXACT: runoff 5 449.00 km³/yr = held 1 597.31 + to the ocean 3 851.69 + stranded 0 (MEASURED gen 21); gen 20: DRY 9 599, PARTIAL 1 549, SPILLING 5 839, 5 447.10 = 1 607.37 + 3 839.73 + 0 |
| G-ICE | GREEN as a law test | 11.29 % on the Earth-like charter (DOC W14.5); the home planet reads 3.74 % (MEASURED gen 21; 3.75 % at gen 20, 4.02 % at gen 19) |
| G-BUZZSAW | GREEN | 1 296 m (MEASURED gen 21; 1 295 m at gen 20, 1 316 m at gen 19) |
| G-CRATER | GREEN on the count | 1 668 (gen 20; 1 610 at gen 19) against ~190; the size floor 16.4 km (`terrain/src/craters.rs:104`) |
| G-GRID | **RED**; ★ step 2b's flat law NOT met (G-GRID-VALLEYS-BY-CLASS and G-LAKE-WALK RED since the review fixes; §11 item 11) | valleys **1.205** by the drainage-area rule, 1.139 by the step-1 rule, 1.337 outside the routing's filled hollows (MEASURED gen 21); by the water's class (G-GRID-VALLEYS-BY-CLASS, RED on the flats and the lakes): dry ground 1.414, on a flat of the ground 1.470, across a spilling lake 1.553; lakes 1.128 (947 patches). The aggregate reads under every class because the dry ground leans to the diagonals (45°, 135°) and the flats to the rows (0°, 90°), and the two cancel in one histogram. LIKE FOR LIKE (the attribution, one change at a time): without the flat sweep (V3) 1.093 / 1.124 / 1.132, with it (V2) 1.207 / 1.138 / 1.335 — the flat sweep made the lean WORSE. Gen 20: 1.484 / 1.522 / 1.306, lakes 1.112 (the routing's tree, not a control of step 2b's rule); gen 19: 1.284 against 1.006. THE WALK ACROSS A SPILLING LAKE (G-LAKE-WALK, six steps toward the spill, the planet's own T9): mean error 14.07° against the 12° bound, RED; exact runs 21.3 %, against the rule before the split 20.06° and 57.4 %; the walks lean 1.605, the lakes' own straight lines to their spills 1.425, so the step adds lean beyond the lakes' node-square shapes |
| G-HYPSO | GREEN on two humps and on the age; NO BAND on the share over 1 000 m | humps at −9 241 m and +9 m over the sea; basin integral median **0.590** (Strahler 0.35–0.60: at its top edge; since step 2b read from the solve's own `FullReport::integrals`, the erosion's routing tree, plan question 7); land 21.24 % of the surface, its mean +1 546 m; the sea floor's mean −6 911 m; 63.11 % of the land over 1 000 m (MEASURED gen 21; the same at gen 20 but the upper hump +8 m; 0.484, 21.41 %, +1 399 m, 55.12 % at gen 19) |
| G-RIVER | **GREEN** since step 2b (every river ends, no floor rises) | 6 101 heads (drainage-area rule): the ocean 3 745, a closed lake 1 442, a dry closed basin 914, a pocket drawn as sea 0, an end the budget does not name 0; **0 trunk steps rise** (MEASURED gen 21). The step-1 rule: 5 703 heads, 0 rising steps. Gen 20: 6 752 heads (3 513 / 1 588 / 1 651), 3 430 rising steps inside hollows the routing filled, the worst 4 987 m; gen 19: 37 966 heads, 0 rising steps |
| G-LOOK | **RED**, by the owner's word | the flight of 2026-09-24 |

Re-run at generator 19 on 2026-09-24 (step 1); every line is in `step1_census_gen19.txt`. NEW LINES, all RED where
they can be: W10-DISCHARGE-TREE 1 304 002 nodes; G-RUNG-LAKE 529 nodes / 427 patches; G-RUNG-LIP 3 753 columns on
116 nodes; the teeth and dunes lines in §5 and §6.

**Re-run at generator 20 on 2026-09-25 (step 2); every line is in `step2_census_gen20.txt`.** GREEN by construction and
MEASURED so: W10-DISCHARGE-TREE 0; G-RUNG-LAKE 0 nodes / 0 patches / 0 vanished at 9 → 10; G-RUNG-LIP 0; CHECK-OCEAN 0;
CHECK-POCKET-NOT-SEA 0; CHECK-BUDGET-CONSERVE exact. NEW LINES of step 2: G-RUNG-LAKE-LOST/MIXED/SHIFTED and G-RUNG-HOLD
at the swaps 9 → 10, 14 → 15 and 15 → 16 (§7 W6); G-RIVER by both river rules; G-GRID-VALLEYS by three river sets.
NEW LINES of step 2's review fixes (2026-09-25): G-POCKET-HELD and POCKET-MINORITY-HELD at rungs 9, 10, 13, 14, 15
and 16; ROW-OUT-OF-CLOSED-HOLLOWS; the fixed point's unsettled word; the cascade's walk length.

**Re-run at generator 21 on 2026-09-25 (step 2b); every line is in `step2b_census.txt`.** GREEN by construction and
MEASURED so: G-RIVER-DESCENT 0 (both river rules); W10-DISCHARGE-TREE 0 (the census's own sum over the shipped
receivers, less the sinks at the spills); ROW-SEA equal to the unit; ROW-OUT-OF-CLOSED-HOLLOWS 0 of 2 571; ROW-HELD 0
of 6 408 closed bodies; CHECK-DRY-FOLLOWS-DESCENT 0; CHECK-EXIT-AGREES 0; CHECK-TREE-CYCLIC 0 rings, 0 falls, 0 short
sinks. NEW LINES of step 2b: the three W0 readings (M1-OCEAN-SIDE-POURS, M2-SADDLE-SIDES, M3-GROUND-PLATEAUS, first
read at generator 20 before anything moved), G-GRID-VALLEYS-BY-CLASS, G-LAKE-WALK, and the checks above. RED:
G-LAKE-SHARE 4.828 % (§11 item 10); ★ G-GRID-VALLEYS-BY-CLASS (flats 1.470, spilling lakes 1.553) and G-LAKE-WALK
(14.07° against 12°) — both printed NO-BAND until the review fixes, now gates (§11 item 11); the far-swap lake losses (4 501 columns at 14 → 15, 745 at 15 → 16; 3 698 and 558
at generator 20 — they grow with the lakes) and
G-RUNG-HOLD (5 / 2 / 3 columns) as before — step 4's. ★ THE THREE OTHER INSTRUMENTS on generator 21 (the same file,
each against its own generator-20 run after step 2's review fixes): `reference_stand` — its 13 CENSUS lines UNCHANGED;
`shipped_drainage` — the belt's and the plain's lines UNCHANGED, the coast's all moved, because `pick_stands` picks the
coast by the greatest discharge class and step 2b changes that byte's meaning (node 4 718 858 → 5 642 175); `shore_teeth`
— every line moved, because both its stands moved (the coast by the same rule, the largest lake 6 398 → 12 562 nodes),
so its numbers read other ground and compare with nothing at generator 20. The step-4 teeth are still RED on every
stand. ★ CURED BY THE REVIEW FIXES (findings 4 and 9, 2026-09-25): the coast and the plain are PINNED to generator 20's
picks (`drainage_gates::COAST`, `PLAIN`: nodes 4 718 858 and 1 219, each at its node's exact centre) and `shore_teeth`'s
lake stand to generator 20's largest lake's edge (`LAKE_STAND`: face NegX, wet (256, 674), still a lake at generator
21); the rule's own picks print beside them as a reading (the coast rule picks node 5 642 175 today). RE-RUN at
generator 21 on the pinned stands (`step2b_census.txt`, "AFTER THE REVIEW FIXES"), against generator 20 on the same
stands: `shipped_drainage` — EVERY CENSUS line UNCHANGED on all three stands (step 2b moved no dunes number); `shore_teeth`
— the coast's sea lines UNCHANGED at rungs 0–10, at rung 13 the wall 50.62 → 50.87×; the lake near the coast at rung 13
a little; on the lake stand the sea lines UNCHANGED (one disagreeing column more at rung 13) and every LAKE line moved,
because the lake at that edge grew with step 2b (at rung 7 the offset 11.8 → 18.3 km, the reach 11.0 → 16.3 km, the
drawn shore's lean 1.974 → 1.919 against its ground's 1.646 → 1.478; all RED before and after — step 4's).

**THE BLIND SPOTS — CLOSED BY STEP 1 (2026-09-24).** Before step 1 no gate saw the three complaints; `drain_census` at
generator 19 read 0 closed hollows on all three stands (the deepest 0.02 m), on the very ground the owner calls dunes.
What each blind spot was:

- **The dunes.** `drain_census` counts closed hollows at rung 3 (8 m columns) with a 0.1 m floor
  (`bins/examples/drain_census.rs:54`, `:219-225`). At the floor of a 3.125 km ripple 199 m tall, two columns 8 m
  apart differ by 0.026 m (ARITH). So it reads a dune field as drained ground. Its stand is 5.95 km a side (less
  than two ripples), and its "plain" stand is where the roughness is smallest. Its header still says it counts
  minima "on a drainage line", which is stale since W16.
- **The teeth.** No instrument measures the distance from the drawn shore to the ground's own waterline, or the
  wall on it, or the shoreline's direction.
- **The water positions.** No instrument counts the lakes lost at the rung 9 → 10 swap, the inland nodes under the
  sea's level, or a hollow's watershed supply against the budget's supply.

---

## 5. Why the land looks like dunes

- **D1. The noise owns every wave under 32 km, not under 8 km.** `FINE_ABOVE_M` = 4 × 8 192 = 32 768 m
  (`terrain/src/body.rs:94-96`). So the fine octaves run from 25 km down to 49 m. The solved field is a Catmull-Rom
  read of 8 192 m nodes (`terrain/src/artifact.rs:1641-1660`), so it holds no feature narrower than about two nodes
  (16 km). The 25 km octave rides on the solve; the 12.5 km octave and every finer one has NO solve under it. The
  rulings (B2 step 4, W16.1, W16.9 item 5) say "the fine relief under 8 km is noise"; the code makes it 32 km, and
  the comment at `body.rs:86-92` still says 8 224 m and 32 896 m.
- **D2. Nothing drains the noise.** The column kernel reads five words: z, first, slope_share, water, side
  (`recipe/src/plan.rs:138-156`). The discharge row reaches only instruments (`terrain/src/artifact.rs:1433`); the
  receiver byte's shipped readers take only the facies bits (`artifact.rs:515`, `:625`, `:634`). The drawn drainage
  is retired (`terrain/src/river.rs:1-10`). The erosion operators run only in the bench (`recipe/src/amplify.rs:1-6`).
- **D3. The spectrum is a bump with one peak, the shape of a dune field.** The octave slope peaks at the 3.125 km
  octave (`O_PEAK` 7, `SIGMA` 1.4, `body.rs:56-77`). The heights at full roughness: 496, 402, 320, 199, 80, 25, 7.8,
  2.5, 0.9, 0.3 m (sum 1 532 m, `body.rs:66-69`). The ratio between octaves is about 0.8 above 6 km and about 0.3
  under 1.5 km. Natural ground keeps one ratio. So the ground has ONE main hill spacing, 3–6 km, and faces under
  1.5 km are almost smooth.
- **D4. The slope is anchored at the angle of repose.** The fine octaves' RMS slope is `TALUS_RMS × TAN_REPOSE` =
  0.70, about 35° (`body.rs:78-85`) — the slope of a dune's slip face. Where the roughness is one (a macro slope of
  12.5 % or more), the four octaves drawn at rung 7 carry an RMS slope of 0.56 (ARITH).
- **D5. The ridged fold makes closed cells, not networks.** A ridged octave is 1 − 2|n| (`recipe/src/height.rs:443`)
  on isotropic Perlin noise (`terrain/src/noise.rs:101-131`). Its crest is sharp on the noise's zero set and its
  trough is round, so crests close into rings around round pits. River ground is the reverse: soft divides and sharp
  channels that join downhill. The flow stretch promised at `body.rs:128-130` ("8c adds the stretch") was never built.
- **D6. The roughness sets height, never direction.** m = the greater of a 20–120 km placeholder noise (floor 0.06)
  and the solve's slope share (`recipe/src/height.rs:318`). The slope share keeps only the gradient's size
  (`artifact.rs:1765-1783`). On a plain, and on the level flats the deposit leaves at each spill
  (`solve.rs:1018-1049`), m falls back to the placeholder.
- **D7. The ridged LIFT swells the plains.** The ridged term has a mean above zero (`body.rs:132-136`, "THE LIFT"),
  and m multiplies the whole fine sum (`height.rs:325-331`). So on a plain the mean ground rises and falls with the
  placeholder's 20–120 km noise, which the solve never routed. These swells can hold basins the lakes never see.
  Upper bound ~960 m; the real size UNMEASURED.
- **D8. At 100–150 km every feature the eye sees is noise.** Rung 7 draws to 111 km, rung 8 to 222 km, rung 9 to
  445 km (`client/src/ladder_view.rs:13-14`). Rung 7 keeps the 25, 12.5, 6.25 and 3.125 km octaves; rung 8 the
  first three; rung 9 the 25 km smooth octave and the 12.5 km crest (`body.rs:203-232`). From rung 10 no fine octave
  survives and the ground is the solve alone at pyramid level 1. At 100 km a pixel is ~115 m, so the ripples are
  27–217 pixels long (ARITH).
- **D9. The octave lattices nest (medium).** Each wave is exactly half the one before (`body.rs:976-985`); the
  lattice point is direction × radius/λ with no rotation and no offset (`height.rs:163-165`). The crests line up only
  near the coarse corners the sphere passes closely. Its share of the look is UNMEASURED, and it shares the face
  grid's axes, so it may also feed G-GRID (UNMEASURED).
- **D10. The look amplifies it; it is not the cause.** One sand colour for all ground (`client-render/src/terrain.rs:1974`),
  and no cast shadow at 100–150 km (§3, 8s). So only soft light shows the relief. Ruling B1: paint hides nothing.

**MEASURED (step 1, `shipped_drainage`, rung 7, ~206 km squares, generator 19).** The noise makes the dunes; the solve
alone drains:

| reading | belt: shipped | belt: solve only | coast: shipped | coast: solve | plain: shipped | plain: solve |
|---|---|---|---|---|---|---|
| G-HACK n (band 0.45–0.7) | **0.224 RED** | 0.735 RED | **0.205 RED** | 0.725 RED | **0.251 RED** | 0.488 |
| G-HACK c (band 1–6) | 2.63 | 9.89 RED | 2.46 | 8.88 RED | 2.95 | 5.54 |
| closed basins ≥ 10 m deep, per 100 km² | **2.84** | 0.003 | **2.55** | 0.017 | **2.00** | 0 |
| closed basins ≥ 100 m deep, per 100 km² | 0.52 | 0 | 0.51 | 0 | 0.35 | 0 |
| dry ground inside a basin ≥ 1 m deep | **35.6 %** | 3.7 % | **41.3 %** | 9.1 % | 9.8 % | 1.4 % |
| G-BREACH, shipped over solve | 28× | — | 59× | — | 4.3× | — |
| hill spacing | **3.2 km** | 28.4 km | 3.2 km | 22.5 km | 4.2 km | 13.4 km |
| the fine relief's mean (the LIFT, D7) / RMS | +287 m / 330 m | — | +267 m / 315 m | — | +324 m / 423 m | — |

The hill spacing of the shipped ground, 3.2 km, is the spectrum's peak octave (3.125 km, D3); a third of the belt's
dry ground lies in closed hollows; n ≈ 0.2 says water never travels far before a hollow stops it. The solve alone reads
n ≈ 0.73 and c ≈ 9 on the belt and the coast, just over the band: its 8 km network is long and straight (W9).

**Re-read at generator 20 (step 2; `step2_census_gen20.txt`): the dunes did not move, as step 2 never meant them to.**
Shipped G-HACK n 0.226 (belt), 0.207 (coast), 0.252 (plain), all RED; closed basins ≥ 10 m per 100 km² 2.94, 2.37,
2.00; the hill spacing 3.2, 3.3, 4.2 km. The cure is step 3 (R2).

**Example.** Under the belt stand, a solved trunk valley is a round trough at least 16 km wide. On its floor stand
ripples of 25, 12.5, 6.25 and 3.125 km, up to ~500 m tall, all at 35°, with no side valley leading into the trunk.
From 120 km, in one sand colour and with no shadow, that is a dune sea.

---

## 6. Why the shores look like teeth

- **T1. The side is the nearest node's word.** At rungs 0–13 the coast mask's footprint level is zero, so
  `sample_side` reads the ONE nearest fine node (`terrain/src/artifact.rs:1134-1159`, the nearest-node branch at
  `:1148`). One side covers each 8 192-cell square on the face grid (nominal 8 192 m; the face bend changes the size).
  At 120 km one square is ~60 pixels wide (ARITH).
- **T2. The shore law obeys the side, not the ground.** When a host states the side, `shore` ignores the ground's
  own sign and holds the column at least a quarter of |base − water| on the stated side (`recipe/src/height.rs:395-413`,
  `SHORE_SHIFT` 2 at `:336`). So the shoreline is the square's edge, not the place where the ground crosses the
  water, and a wall of about (|g_land| + |g_sea|)/4 stands on it. At rungs 10–13 the base comes from level-1 means
  (16 km nodes), so the band where side and ground disagree can fill a whole square.
- **T3. The shore law allows no coast feature under one node.** No bay, headland, inlet or islet smaller than 8 km
  can exist. So a cure must do BOTH: remove the square side AND give the coast detail under the node. A smooth side
  on today's 8 km data gives a smooth 8 km contour, not a natural coast.
- **T4. The lake's edge has the same fault and one more.** The water level is also the nearest node's word
  (`artifact.rs:1027-1041`). At rungs 0–9 a dry column beside a lake takes the SEA's level (`artifact.rs:1084-1087`),
  so nothing holds it over the lake. A lake column is held under its level by a quarter of how far its ground stands
  over it, so the lake is a flat square pit with a wall on each node line. The doc comment at `artifact.rs:1053-1055`
  says a land column takes "the neighbouring lake's level"; at rungs 0–9 the code never does.
- **T5. At rungs 10–13 the water word comes from pyramid level 1 (16 km).** `fold_water` gives DRY under two wet
  children of four and averages every wet child, sea and lake alike (`artifact.rs:1395-1401`, `:1455-1467`). So a lake
  node alone in its block turns to LAND; a land column in a lake-majority block takes the lake's level and the shore
  law lifts it into a LIP; a block half sea and half lake takes the MEAN of two levels no water has. A second, 16 km
  staircase with lips stands at ~450–700 km. ★ CURED BY STEP 2 (generator 20): the word is the HIGHEST lake level of
  the block's lake nodes, else the sea, else DRY (`artifact::block_water`), and a land column never reads the word —
  it holds the sea, or no water at all where no node of its read stencil is the sea (`column_water`,
  `stencil_holds_sea`). MEASURED: 0 lake columns lost, mixed or shifted at 9 → 10, 0 lip columns.
- **T6. From rung 14 the side is the majority of a 2^(rung−13) block** (`artifact.rs:384-396`). Near one pixel where
  the ladder draws it, but many pixels when the bounded ask pulls a coarse rung near on a fast climb (the W18.5 case).
- **T7. Inland ground under the sea's level is drawn as sea**, with no test of a path to the ocean
  (`terrain/src/solve.rs:1334-1336`; `terrain/src/lakes.rs:186-197`). It shows as square sea-level pools far from any
  coast. ★ CURED BY STEP 2 (generator 20; ruling B7 item 1): the ocean is the largest connected water under the level
  (`land::ocean_mask`), the sea is solved over it alone (`land::sea_level_connected`), and every facies, water word,
  flood seed, load and climate read of the sea reads that ocean (`MacroSolve::ocean`). An inland dry column under the
  sea's level reads `SIDE_INLAND` and the shore law leaves it alone (no mesa); the water sheet and the eye read the
  column's own water, never raised to the sea. MEASURED: CHECK-POCKET-NOT-SEA 0 of 61 393 pocket nodes. ★ AND (review
  finding 1, 2026-09-25) a pocket's dry floor is its own coast-mask word, so it is INLAND at every rung even with the
  sea in its read stencil: G-POCKET-HELD 0 at every rung the census reads (§7).
- **T8. The water and the ground are one colour each** (`client-render/src/terrain.rs:1974`, `:2034`); the beach bit
  reaches no pixel. So the square edge shows at full contrast. LATENT: the beach is also a node square
  (`terrain/src/chunk.rs:404-411`), and it will outline every tooth in sand the day 8e draws colour.
- **T9. D8 may also square the coast above the node scale.** No coastline-direction histogram exists. UNMEASURED.

**MEASURED (step 1, `shore_teeth`, the coast stand and the largest lake, generator 19).**

| rung (square) | the drawn shore's offset from the ground's waterline, median / p90 | the wall on the drawn shore, median / p90 / max | slope's own step | columns on the wrong side | drawn shore's grid lean vs the ground's own |
|---|---|---|---|---|---|
| sea, rung 5 (31 km) | 2.2 km / 8.2 km | 295 m / 495 m / 621 m | 1.8 m | 6.0 % | 2.32 RED vs 1.25 |
| sea, rung 7 (125 km) | **2.9 km / 8.1 km** | **309 m / 674 m / 1 168 m** | 5.0 m | 2.4 % | **1.93 RED** vs 1.42 |
| sea, rung 9 (125 km) | 2.9 km / 7.9 km | 179 m / 1 051 m / 1 532 m | 8.8 m | 2.4 % | 1.78 RED vs 1.60 |
| sea, rung 10 (125 km) | 6.2 km / 21.3 km | 213 m / 1 578 m / 2 064 m | 4.1 m | 5.0 % | 1.46 vs 2.00 |
| lake, rung 7 (120 km) | 3.2 km / 4.1 km | 210 m / 282 m / 425 m | 2.9 m | 2.0 % | 2.03 RED vs 2.03 RED |

**Re-read at generator 20 (step 2):** the coast stand MOVED with the field (`pick_stands` now takes node 4 718 858 on
face NegY, a steep coast), so its numbers read other ground: at rung 7 the drawn shore stands a median 3.6 km (p90
10.9 km) off the ground's waterline with a median 1 396 m wall; the drawn shore's grid lean 2.95 against the ground's
own 2.61. The largest LAKE is now 6 398 nodes: its edge at rung 7 stands a median 3.3 km off its ground's waterline
with a median 261 m wall, lean 1.97. The teeth are step 4's; step 2 did not touch them.

At rung 0 (1 m columns) the coast's node edge carries a **614 m** wall across one column. The drawn shore leans on the
grid while the ground's own waterline in the same square does not: the teeth are the node squares, not the land. The
lake's own waterline leans too, because its level is one word per node (T4). The noise's LIFT (+~290 m, §5) is why the
land column at a square's edge stands so high over the water.

**Example.** A land node stands 50 m over the sea, and its sea neighbour 200 m under it. At the square's edge the
solved ground reads ~75 m under the sea. The land column there is held ~19 m over the water, the sea column ~19 m
under it: a bank runs straight for 8 km, turns 90°, and runs straight again (ARITH from `shore`).

---

## 7. Why the water positions are not believable

- **W1. No river is drawn.** The stamp is retired (`terrain/src/river.rs:1-10`). The width law has no caller outside
  its tests; `river.rs:14-15` and W16.1 say the solve still uses it, and the code says it does not (the sweep cuts by
  stream power with no width, `solve.rs:960-1000`). A river exists only as a dry trough at least ~16 km wide.
- **W2. A lake is whole 8 km squares.** The budget writes one level per basin to each node under it
  (`lakes.rs:300-319`, `:464-487`). The smallest lake is one node, ~67 km² at a face's middle.
- **W3. The budget ignores the watershed — a defect against its own law.** After a hollow closes at its spill, every
  higher node, including the range that drains into it, books its water to the PARENT or to the ocean
  (`lakes.rs:209-259`, `:343-349`, `:354-380`). So a hollow is full only where the rain beats the evaporation over its
  own brim (`lakes.rs:384-390`). Fill-Spill-Merge, which ruling W11 cites, feeds a hollow from its whole watershed.
  The module's own example ("takes a river off the range", `lakes.rs:40-41`) contradicts its code. ★ CURED BY STEP 2
  (generator 20): every node's runoff runs down the ground (`lakes::descent`, Fill–Spill–Merge's flow directions) to
  the leaf whose pit it ends at, or to the ocean. MEASURED: a leaf's watershed supplies a median 11.54 times (p90 74
  times) the runoff of its own nodes.
- **W4. A full hollow's overflow is lost** when its parent holds no water (`lakes.rs:264-281`, `:446-452`): a lake
  spills over its sill and the next basin gets none of it. ★ CURED BY STEP 2: `lakes::cascade` passes the overflow over
  the spill into the sibling (at the leaf its gate descends to), into the parent's own ring, or into the sea.
  MEASURED: 6 839 pours; the books exact (runoff 5 447.10 km³/yr = held 1 607.37 + the ocean 3 839.73 + stranded 0).
- **W5. Too few lakes, and the gate cannot see it.** Every pass's deposit fills the valley and ice hollows up to their
  spill (`solve.rs:997`, `:1003-1040`). Lakes cover 0.07 % of the land against the 0.2–0.8 % ruling W11 names for dry
  interiors. G-LAKE-SHARE is a ceiling (≤ 4 %), so it stays green. The reference's valley lake is the glacial kind this
  fill removes.
- **W6. The small lakes vanish at the rung 9 → 10 swap.** Every lake node that is the only wet node of its level-1
  block turns to LAND at rungs 10–13 (T5), so every lake's outline changes at the swap. That breaks W16's law that a
  rung boundary is invisible. The ring is ~450–700 km since W17 (UNMEASURED); the count UNMEASURED. ★ CURED AT 9 → 10
  BY STEP 2 (`block_water`): 0 of 67 233 lake columns change there. STILL READ at the level swaps (the side there is the
  WET MAJORITY of the footprint, ruling W15): at 14 → 15, 3 698 lake columns turn to land, 211 take another lake's
  level (the largest step 270 m = 0.016 px), 1 319 columns become lake; at 15 → 16, 558, 251 and 448 (MEASURED gen 20).
  Each such column is one cell of the finer rung, which is one pixel at the swap distance.
- **W7. The climate.** P/PET = 0.99 × belt × lift (`climate.rs:776`, `:118`, `:161`). The lift reaches 4 on a windward
  slope, but a hollow's floor usually sits in the lee. At generator 14 only 2 hollows spilled (DOC W11); the count at
  16–19 is not stated.
- **W8. D8 leans on the grid.** One receiver of eight (`solve.rs:784-809`); G-GRID RED at 1.284. For the LAKES the
  strong grid lean is W2 (whole squares), not D8.
- **W9. The 8 km network itself is not dendritic.** On the bench, the bicubic upsample of one belt tile fails Hack's
  law (L = 1.99·a^0.883, n outside 0.45–0.7; DOC survey R1). Four erosion operators at 1 024 m make it pass
  (L = 5.50·a^0.499). This is the measured case for R2.
- **W10. Low weight.** The flexural rebound lifts 16 × 16-node blocks (~131 km, ARITH) by one number each after the
  last sweep (`solve.rs:1098-1124`): a straight scarp, more likely a small dam across a valley. The solve cuts with the
  rain while the budget reads the runoff, and the discharge row is pass 39's rain while the receiver byte comes from
  later routes (`solve.rs:940`, `:1587-1588`, `:1654`, `:1705`). R2 seeds its rivers from that row. UNMEASURED sizes.
  ★ THE DISCHARGE HALF CURED BY STEP 2 (ruling B7 item 2): the shipped row is the Turc–Pike RUNOFF accumulated over the
  receivers of the LAST routing (`MacroSolve::accumulate_runoff`). MEASURED: W10-DISCHARGE-TREE 0 nodes. The passes
  still cut with the rain's drainage (plan question 4). ★ AND SINCE STEP 2b (ruling B8 item 2, amending B7 item 2) THE
  ROW IS THE BUDGET'S OWN WATER over THE WATER'S OWN RECEIVERS (`MacroSolve::follow_the_water`): the runoff accumulated
  down the water's tree, less at each spill what its spilling lake hands the air (the tree's SINKS,
  `MacroSolve::spill_sinks`). MEASURED gen 21: the census's own sum by the same rule differs at 0 nodes; the row into
  the sea is the cascade's `to_ocean` to the unit (3 851 690 606 676 394 mm·m²/yr); every closed body's ends carry what
  it holds. The passes still cut with the rain (question 4, R2's track): the solved heights did not move.
- **W11. THE SHIPPED RECEIVERS LEAVE A SPILLING LAKE UPHILL** (found by step 2's census). The receivers are the routing
  fill's (ruling W11: a scratch surface). The fill raises a whole nested hollow to its OUTER spill, and its flat's
  receivers carry a full inner hollow's river across the parent's DRY ring toward the parent's own outlet — while the
  budget pours that water over the inner saddle into the sibling. Before step 2 no hollow spilled, so the walk never
  went there. MEASURED gen 20: G-RIVER-DESCENT 3 430 rising trunk steps, all inside filled hollows, the worst 4 987 m.
  The discharge row inherits it (plan question 3: A is built). The cure proposed: the shipped receivers follow the
  budget's water (Fill–Spill–Merge's own flow directions: steepest descent on dry ground, across a lake's flat to its
  outlet, over a full lake's spill to the sibling's gate). A change of the receiver byte's meaning: the owner's word
  before step 3, because R2 seeds its rivers from the receivers. ★ CURED BY STEP 2b (generator 21, artifact 9; ruling
  B8 item 2): the shipped receivers are THE WATER'S OWN WAY (`lakes::water_tree`) — dry ground takes the budget's own
  descent; every node of a SPILLING lake's hollow takes the flat step toward the lake's spill (the summed chord to the
  spill says which way, Rho8's split between the two neighbours that bracket the straight line says which neighbour:
  Fairfield & Leymarie 1991, a named scatter under the `FLOW` salt, T9); the spill points at the gate of the side the
  budget pours into; a node under a CLOSED lake's water and a dry hollow's pit ENDS the river. MEASURED: G-RIVER-DESCENT 0
  (3 430 at gen 20). ★ THREE THINGS HAD TO MOVE WITH IT, each MEASURED first at generator 20 by the census's W0 lines:
  1. **The cascade pours by the saddle's STEEPEST OPEN SIDE** (`lakes::open_side`; plan question 2, choice A): the
     binary merge paired the sides by root index, and at a saddle of three sides the overflow went to a side the
     ground did not lead into (M2: 185 saddles of three or four sides; 51 pours, 9.5 km³/yr). ★ AND TOWARD THE SEA IT
     RUNS DOWN THE GROUND (question 1, choice A; Fill–Spill–Merge's own link from an ocean-bound depression to the
     depression its overflow first reaches): the ocean-side gate's descent ended in a lower hollow that met the sea
     earlier for 3 181 of 3 989 pours to the sea (969.3 km³/yr), 613 of them (36.4 km³/yr) in one that was NOT full, so
     no receiver tree could carry the cascade's water (M1). The saddle's sides are sorted with the side the node's
     own water descends into FIRST (off a flat that is the steepest; on a flat every side ties at no drop): MEASURED
     before that order, two closed bodies on a flat divide traded 1.1 km³/yr between the budget and the row.
  2. **A flat is swept from its drains** (`lakes::flats`; question 3, choice A; Barnes, Lehman & Mulla 2014's "toward
     lower ground" in the lattice's own chord; a closed flat is ONE hollow with its pit at its middle): by `(height,
     index)` the sweep opened a leaf of ZERO depth wherever a node of a draining flat was reached before its neighbours
     (M3: 79 721 flats, 716 574 nodes; 8 795 zero-depth leaves holding 462.7 km³/yr; 37 625 of 87 405 descent trunks on
     a flat, their lean 1.486). MEASURED after: 0 zero-depth leaves; the descent trunks on a flat lean 1.213.
  3. **A node at its hollow's spill height costs that hollow nothing (THE BRIM)** — a law the plan did not name,
     found by the stated trough's test: a flat swept from its drains leaves every node of it that drains into a hollow
     at that hollow's spill height, and the budget charged the hollow for covering nodes the full lake never covers (the
     level covers `z < level` only). The budget now charges such a node to the ring above.
  ★ THE COST: the lakes grew. MEASURED by census runs with one change at a time: the water's tree over gen 20's own
  budget keeps G-LAKE-SHARE at 3.510 % but its row into the sea falls 188.2 km³/yr short of the cascade's and 632
  closed bodies hold more than the budget books (ROW-SEA, ROW-HELD RED: B8's "the row's sum into the sea equals the
  cascade's" cannot hold); + the steepest open side and the ground's way to the sea 4.162 %; + the flats 4.741 %; + the
  brim 4.828 %. The largest lake 12 562 nodes (843 022 km²); its edge stands at aim −0.397 −0.825 −0.403 on face NegY (the
  census's own stand line). ★ CORRECTED 2026-09-25: an earlier line put it "in the second-largest inland pocket"; that
  pocket's deepest node stands 43° (about 4 800 km) away and holds a PARTIAL lake only, so the claim was wrong. §11 item 10.

**MEASURED (step 1, `lake_census`, generator 19).**

- **W6 confirmed:** at the rung 9 → 10 swap 529 of 1 287 lake nodes turn to land (30 628 km²); **427 of 506 lakes change
  their drawn water and 345 vanish whole**; 3 753 dry columns on 116 nodes become a LIP over a lake level their ground
  stands under. G-RUNG-LAKE and G-RUNG-LIP RED.
- **W3 confirmed, and sized:** a hollow's watershed supplies a median 12 times (p90 72 times) what the budget gives it.
  Fed from its watershed by the budget's own rule, 2 274 dry hollows turn PARTIAL and 7 687 turn SPILLING; the lakes
  would cover **1.914 %** of the land (today 0.068 %; ceiling 4 %). A floor under the repair: W4's lost overflow stays.
- **Where the rivers end:** of 37 966 river heads (drainage ≥ ~6 700 km²), 21 130 end in a dry closed basin, 13 217 in
  the ocean, 2 611 in an inland pocket, 1 008 in a closed lake. Every river ends and no trunk rises (G-RIVER GREEN).
- **T7 / W7 far larger than read:** 148 inland pockets, 59 470 nodes, **3.2 million km²**, stand at or under the sea's
  level with no path to the ocean; the largest is 1.35 million km² and 11.8 km deep, the second 0.70 million km² and
  9.2 km deep. The client draws every one at the sea's own level. The owner's question §11 item 3.
- **W10 confirmed:** the shipped discharge row disagrees with the discharge re-accumulated over the shipped receivers
  at 1 304 002 nodes (RED).

**MEASURED (step 2, `lake_census`, generator 20; `step2_census_gen20.txt`).**

- **The lakes:** 67 233 lake nodes, 3.883 × 10⁶ km², 2.19 × 10⁵ km³; 3.51 % of the land; 3 883 patches, the largest
  6 398 nodes (429 363 km²). The bodies DRY 9 599, PARTIAL 1 549, SPILLING 5 839; the leaves DRY 9 720, PARTIAL 2 532,
  SPILLING 7 036 of 19 288. The plan's floor from the step-1 what-if was 1.914 %; the cascade added the rest.
- **The pockets:** 236 pockets, 61 393 nodes, 3.30 million km², 1.52 × 10⁷ km³ under the sea's level; by the budget
  DRY 181, PARTIAL 45, SPILLING 10. The largest (28 913 nodes, 1.57 million km², 11 867 m deep) is DRY. The FIRST land
  held 3 pockets of 17 799 nodes (plan question 9). The flood raises them and the deposit fills them only a little:
  the routing's pit volume at the last route is 8.73 × 10⁹ steps (1.27 × 10⁹ at gen 19).
- **The sea:** 3 591 m (3 566 m at gen 19, +25 m; the plan's ARITH ≤ ~24 m); the ocean share 78.11 % (78.59 %). The
  lakes' fixed point: 1 re-solve, then still; no connected solve re-seeded.
- **The rung swaps:** as W6 above; G-RUNG-HOLD (plan risk R7) RED: 19 dry columns at 9 → 10 whose INLAND/LAND word
  changes while their ground stands under the sea — the largest surface step 6 090 m = 11.9 px at the swap distance;
  5 at 14 → 15 (0.32 px), 1 at 15 → 16. ★ After the review fixes: 5 at 9 → 10 (731 m = 1.43 px), 2 at 14 → 15, 3 at
  15 → 16 (below).
- **The rivers:** as §4 G-RIVER; the pocket ends are 0 by construction (2 611 at gen 19).
- **The cost:** the solve 140.6–143.0 s in the census (140.6–151.6 s over six runs; 114.2 s in the gen-19 census: about
  +25 %, the plan's ARITH +15–20 %); the hierarchy, the descent, the budget and the cascade 0.3 s.

**MEASURED (step 2's review fixes, `lake_census`, generator 20, 2026-09-25; the lines in `step2_census_gen20.txt`).**

- **A pocket's floor is never lifted (review finding 1):** the floor under a pocket read LAND, and where the sea stood in
  its read stencil (whose reach doubles with the pyramid level) the shore law lifted it a quarter of its depth over the
  sea. Now a dry node whose row stands under the sea's level is the coast mask's word 2 (`artifact::Side::Pocket`,
  counted in `CoastCounts`), and it reads INLAND at every rung. G-POCKET-HELD 0 of 60 383 pocket columns at rungs 9,
  10 and 13; 0 of 15 657, 3 853 and 944 at 14, 15 and 16. A far cell whose footprint holds a pocket's floor as a
  minority reads LAND by W15, and the law holds 1 of 953 such cells at rung 14, 1 of 801 at 15 and 2 of 457 at 16
  (the deepest 1 413 m under the sea). G-RUNG-HOLD: 5 columns at 9 → 10 (731 m = 1.43 px; was 19 and 6 090 m), 2 at
  14 → 15 (was 5), 3 at 15 → 16 (0.106 px; was 1). What stands is a LAND rim column beside both a pocket and the sea.
- **The row against the budget (review finding 3, plan question 3):** the row delivers 5 447.1 km³/yr from the land
  to the sea, the cascade 3 839.7 km³/yr: the row carries to the sea the 1 607.4 km³/yr the budget holds in its
  hollows. Of the 5 322 hollows that spill into the sea and are not full, 4 544 have a row leaving them; the largest
  row out of one is 1 044.8 km³/yr (ROW-OUT-OF-CLOSED-HOLLOWS). The cure is B (the budget's own water) or the
  receivers of W11: the owner's word.
- **The cascade's walk (review finding 6):** 28 098 rings visited over 19 288 leaves, with the skip over full rings.
- **The sea:** the lakes' fixed point 1 re-solve, converged; no connected solve re-seeded, none unsettled (review
  finding 4). The artifact's digest moved (the mask's word 2); the sea, the ocean share, the golden fields and the
  identity did not.
- **The land (review finding 10):** every line of the census now divides by every node that is not the ocean: the
  lakes' share by area 3.510 % (the line that divided by `z > sea` read 3.62 %); by node count 3.40 %, labelled.

**MEASURED (step 2b, `lake_census`, generator 21, artifact 9, 2026-09-25; `step2b_census.txt`).**

- **The tree:** 1 880 042 dry nodes take the budget's descent (0 do not); 47 162 nodes of 2 319 spilling lakes' hollows
  cross to 2 315 spills (steps: 38 465 split, 7 428 straight, 1 269 nearest); 49 336 closed-lake ends; 0 rings, 0
  disagreeing exits, 0 short sinks.
- **The row:** W10-DISCHARGE-TREE 0; ROW-SEA 3 851.69 = 3 851.69 km³/yr; 0 of 2 571 closed hollows that meet the sea
  leak (4 544 of 5 322 at gen 20); ROW-HELD 0 of 6 408.
- **The rivers:** G-RIVER-DESCENT 0 (3 430); 6 101 heads — the ocean 3 745, a closed lake 1 442, a dry basin 914; the
  ground rises under a lake's flat surface on 3 335 trunk steps, the floor on none.
- **The budget:** the hierarchy 17 623 depressions (29 265), 10 694 leaves (19 288); the bodies DRY 5 127, PARTIAL
  1 281, SPILLING 2 319; 5 401 pours into another hollow's side, 1 909 into the ocean's side that the ground leads into a
  lower hollow, 144 off saddles of three sides or more; the books 5 449.00 = 1 597.31 + 3 851.69 + 0 km³/yr.
- **The lakes:** 93 225 lake nodes (67 233), 5.342 × 10⁶ km², 3.14 × 10⁵ km³; G-LAKE-SHARE **4.828 %** RED (3.510 %).
- **The sea:** 3 591 m and the ocean share 7 811 / 10 000, both unchanged in the pinned words; the lakes' fixed point
  moved the level FOUR SIXTEENTHS DOWN (MEASURED by the review, 2026-09-25: from the same 57 462 sixteenths before the
  lakes, 57 453 at gen 20 and 57 449 at gen 21 — a quarter of a metre; the lakes hold 3.14 × 10⁵ km³, 2.19 × 10⁵ at gen
  20; 2 pocket nodes of 61 393 changed side), and the climate over it moved its classes on some thousands of rows (`artifact_diff`: temperature 3 068, rain
  5 113, aridity 28 131).
- **What moved in the artifact** (`artifact_diff` against the gen-20 artifact): the solve's heights 0 of 8 871 936, the
  rows' heights 0, the province 0, the pyramid's heights 0, the sea's metre 0; the lake levels 77 697 nodes, the rows'
  water 76 008, the facies 30 574, the pyramid's water 35 116 words, the coast mask 11 192 bytes; and the two meanings,
  the receiver slot 427 513 rows and the discharge class 492 787 rows.
- **The flat walk across a spilling lake** (`G-LAKE-WALK`, the planet's own T9, 37 303 six-step walks): mean error
  14.07°, p90 23.37°, exact lattice runs 21.3 %; the rule before the split from the same starts 20.06°, 34.99°, 57.4 %.
  The walks' axes lean 1.605 against the control's 1.808 and the straight lines' own 1.425 (the lakes are node squares).
  ★ RED against test T9's 12° bound since the review fixes (it was printed NO-BAND after it failed on the planet: a gate
  is never turned into a reading after it fails); the owner's word on a bound for node-square lakes is owed (§11 item 11).
- **The flat law by the water's class** (`G-GRID-VALLEYS-BY-CLASS`, the area rule's 97 157 trunks by the class of their
  start): dry ground off the flats 51 570 trunks, 1.414 (they lean to the diagonals: 45° 14 317, 135° 18 224); on a flat
  of the ground 34 699 trunks, 1.470 (they lean to the ROWS: 0° 9 462, 90° 12 753 — read in the code, each flat node
  heads for its ONE nearest drain (`lakes::flats`), and where a flat's drain edge is a lattice row that line is a
  lattice axis; how much of the lean that explains is UNMEASURED); across a spilling lake
  10 888 trunks, 1.553. ★ RED since the review fixes on the flats and the lakes (it stated the band ~1.006 and printed
  NO-BAND before).
- **The cost:** the solve 152–171 s over five runs (141–152 s at gen 20); the hierarchy, the flats, the budget and the
  tree inside it.

**Example.** A hollow lies at the foot of the belt's wet range, and the range drains into it by the solve's own
receivers. The budget counts only the rain under the hollow's brim, so the hollow is a dry pan. A pond stands
instead where a hollow's own floor is wet; from 150 km the pilot sees it as a square, and past ~450 km not at all.

---

## 8. THE PLAN (agreed in general by the owner, ruling B6, 2026-09-24)

Each step ends with the census numbers, the three stands' pictures, and the REFERENCE STAND's picture beside the
reference (§9). State legend: ⬜ not started · 🟨 in progress · ✅ done (with the commit).

★ **THE ORDER SINCE RULINGS B10 AND B11 (2026-09-26): 1 → 2 → 2b → 4 slice 0 (the bench) → 5+2c (THE WASHBOARD: measured, §8d; its cause is the DEPOSIT; NEXT a SLOPED basin fill with a published slope law, measured in the bench, then built WITH D∞ in one world change) → 4 slice 0 round 2 (N2) → 4 built → the flight of ALL stands against BOTH references → 3 (R2) → 6 → 7.** (Rulings B8 and B9 set 2b → 4 → 3 with 2c before 3.) The step numbers stay the names every
other file uses ("step 3" is R2, "step 4" the shoreline); only the order changed. Step 2's pictures showed every small
lake as an 8 km square, so the edge law (step 4) now comes BEFORE R2 on today's ground, and the rivers follow the
water (step 2b) before either, because R2 seeds its rivers from them. R2 starts with slice 0 (the bench and a
hillshade for the owner's look) before any bake service.

| # | step | cures | state |
|---|---|---|---|
| 1 | **Measure first, no world change.** The "before" pictures (the three stands + the reference stand). New census lines that can go red: G-HACK and G-BREACH on the SHIPPED ground at rung 7 over ~200 km (the dunes number); a teeth number (the drawn shore's distance to the ground's waterline, the wall height, the shoreline's direction); G-HYPSO and G-RIVER on the final field; the lakes lost at 9 → 10; the inland nodes under the sea's level; the watershed supply against the budget's. The skyline numbers on the reference stand (11_reference_trace's rule). One job at a time: only while the `perf` tree is idle (2026-09-24: it is testing; the instruments are written meanwhile, no build in this tree). | makes each complaint a number | ✅ NUMBERS (generator 19, `step1_census_gen19.txt`) and the "before" PICTURES (`pictures/*_before.png`, §8b), 2026-09-24, committed |
| 2 | **Repair the lake budget.** Feed each hollow from its whole watershed; pass a full hollow's overflow on; re-accumulate the discharge after the final route; keep a small lake's level at pyramid level 1 (no lake lost at 9 → 10). New GENERATOR_VERSION and pins. Goes BEFORE R2, because R2 seeds its rivers from the discharge row. ★ AND (ruling B7) the inland pocket is a closed basin and the sea stands over the connected ocean. | water positions (part), W6's rung seam | ✅ BUILT and MEASURED 2026-09-25 (generator 20, artifact 8, `step2_census_gen20.txt`), pins re-recorded; the PICTURES taken (`pictures/*_step2.png`, §8b: the lakes are there, each one an 8 km square); committed on branch `game` 2026-09-25 (ruling B8 item 4) |
| 2b | **The rivers follow the water** (ruling B8 item 2; plan question 3 choice B; amends B7 item 2). The shipped receivers and the discharge row follow the budget's own water: a full hollow's river runs over its spill into the sibling (Fill–Spill–Merge's flow directions), and a closed body that never fills passes nothing on. New GENERATOR_VERSION and pins. | G-RIVER-DESCENT (3 430 rising steps at generator 20); the row's 1 607 km³/yr out of held hollows | ✅ BUILT, MEASURED and COMMITTED 2026-09-25 (generator 21, artifact 9, `step2b_census.txt`, `step2b_lake_depths.txt`; pins re-recorded; §7 W10, W11; pictures `pictures/*_step2b.png`, §8b): G-RIVER-DESCENT 0, the row the cascade's to the unit, 0 closed hollows leak. Committed with two REDs by ruling B9: G-LAKE-SHARE 4.828 % (judged after step 4) and the flat law (G-GRID-VALLEYS-BY-CLASS flats 1.470 / spilling lakes 1.553, G-LAKE-WALK 14.07°; cured as step 2c) |
| 2c | **The flat directions stop leaning on the lattice** (ruling B9 item 2): across a filled flat and a spilling lake the water's direction reads Barnes, Lehman & Mulla's two combined fields (or a target that does not collapse onto one nearest drain), measured one change at a time against V3 and generator 21; G-LAKE-SHARE read again with it. | G-GRID-VALLEYS-BY-CLASS, G-LAKE-WALK | ⬜ BEFORE step 3 (R2 must not carve along a grid diagonal) |
| 3 | **R2: one eroded level at 1 024 m in the artifact**, with the two operators the bench lacked (the height retargeting that holds the macro heights, the multi-scale breach); then R4's second half: the noise only BELOW the solved level. Cost (survey): ~17 core-hours once for the home planet, 512 KB a tile against 40 KB, an artifact version, a lattice over `MACRO_EDGE_CEILING` 2 048. Calibration: Earth, through Hack's law (T9). | the dunes at 100–150 km; valleys that join | 🟨 DESIGNED (`r2_design.md`, 2026-09-24) with 23 skeptic findings OPEN (§16 of it: 6 HIGH); slice 0, the bench with the owner's look, comes before any service (ruling B8 item 3); AFTER step 4 since B8 |
| 4 | **One shoreline from one field, on today's ground** (ruling B8 item 1 moved it BEFORE R2; it was "on the R2 ground"). The sea's shore and the lake's edge from where the ground crosses the water's level, not from a node square; the lake's level held by its dry neighbours (W6's owed law); plan question 5 (the dry-land hold) and the far-swap lake losses decided here. The law does not depend on which ground it reads: R2 later gives it the 1 km bays and headlands (T3), and the pictures are judged again then; until R2 a shore may look dune-shaped (stated and accepted, B8). | the teeth (sea and lake); step 2's square lakes | 🟨 SLICE 0 BUILT AND RUN (the shoreline BENCH, §8c, 2026-09-25; no world change): the smooth rule N1 is steady at rungs 0–9 (the shore moves p99 ≤ 1.4 px, walls on 0–2 % of the shore against today's 84–97 %); the rough rule O1 jumps 3–9 px at almost every swap (a W16 breach); EVERY rule, today's too, jumps 8–9 px at the far swaps 9→10 and 10→11 (the bench's second slice, 0b); the lakes and valleys themselves follow the lattice (the washboard, §8c) — the owner's word asked on N1, on the far rungs and on the washboard's cure |
| 5 | **G-GRID again on the R2 field.** The D8 decision (MFD with a weight, or a hexagonal mesh = a new world, SL5) only if still red. | water positions (grid lean) | ⬜ |
| 6 | **Lakes with outlets and river mouths** (the spill carve, a delta); **R3**: the 1–5 km valleys under the shipped level, derived per fixed patch on both hosts, the card as the builder (a CPU core takes ~330 ns a cell an iteration, DOC R1). | the reference's valley scale | ⬜ |
| 7 | **G-LOOK.** The owner's word on the three stands and the reference stand. Only then 8e starts (B1). | closes B1 | ⬜ |

**After the arc — THE DRESSING, registered so it is not forgotten** (`docs/design/DEFERRED.md` D-TERRAIN-8): the snow
on the tops (8e reads the ice bit), the ground's colour and the beach (8e), the forest as a mass (8e + the V4 art
assets + the canopy fold), the cliffs, mesas and bedded rock faces (8f), the long far shadows (8s, the `perf` tree — wanted, but not needed to judge the pictures, B6.5 amended),
the water's look (8o), the clouds (the weather plan).

### 8a. Step 1's instruments (written 2026-09-24 by workflow `wf_68fd94d6-341`; built, linted and RUN the same day)

| instrument | what it reads | the lines that can go RED |
|---|---|---|
| `lake_census` (extended) | G-HYPSO on the final field; G-RIVER; the inland sea; the discharge tree against the final receivers (W10); the lakes and the lips at the rung 9 → 10 swap; the watershed what-if (W3) | G-HYPSO-HUMPS, G-HYPSO-AGE, G-RIVER-END, G-RIVER-DESCENT, W10-DISCHARGE-TREE, G-RUNG-LAKE, G-RUNG-LIP, CHECK-* |
| `shipped_drainage` (new) | THE DUNES NUMBER: the shipped ground at rung 7 over ~206 km on the belt, the coast, the plain (or a stated direction), shipped / solve only / their difference: G-HACK, G-BREACH, the spill-depth census, the fine relief's mean, RMS, slope and hill spacing | G-HACK-c, G-HACK-n, SHIPPED-PATH-DIFFER, SPILL-SPLIT |
| `shore_teeth` (new) | THE TEETH NUMBER at rungs 0, 5, 7, 9, 10, 13, for the sea and a lake: the offset of the drawn shore from the ground's waterline, the wall, the grid lean of the drawn shore against the ground's own waterline, the columns on the wrong side | G-TEETH-OFFSET, G-TEETH-GRID, G-TEETH-GRID-GROUND, G-TEETH-SELFCHECK, G-TEETH-BASECHECK |
| `reference_stand` (new) | THE REFERENCE STAND: the pose by a stated rule, the commands for the picture tools, the skyline beside the reference's numbers, the relief 30–60 km | G-REF-DETAIL-TO-HORIZON, G-REF-SHIPPED-PATH, G-REF-SUN-* (the skyline lines are a REPORT by ruling T6 ask 5, never a gate) |

Shared code moved into `crates/bins/src`: `drainage_gates.rs` (G-HACK, G-BREACH and the routing out of `erosion_bench`;
`pick_stands` and `cell_of` out of `drain_census`), `shore_geometry.rs` (`RungReader`, a mirror of the chunk builder's
crate-private column read through public functions; every instrument checks it against `vd_terrain::chunk::column_field`),
`skyline.rs` (the march and the break rule out of `skyline_march`).

**DONE 2026-09-24 (on `af21326`, the `perf` tree merged into main):** `cargo fmt -p vd-bins`; the release build of
`vd-bins` and its examples with no error and no warning; `cargo clippy -p vd-bins --examples --lib -- -D warnings` clean;
the three new modules' unit tests pass (3 of 3). ★ THE THREE MOVES CHANGED NOTHING, MEASURED: the old sources from
`af21326` were built beside the new ones and run with the same arguments — `skyline_march` byte for byte, `drain_census`
and `erosion_bench` (`--iters 20 --fine-span 512 --fine-iters 1 --halo-iters 8`) equal on every line but the timings.
The four instruments then ran one at a time (15:25–15:43); their lines are in `step1_census_gen19.txt`. Every
instrument's self-check read GREEN (SHIPPED-PATH-DIFFER 0, CHECK-RUNG-SWAP 0, CHECK-WHATIF 0, G-TEETH-SELFCHECK all,
G-REF-SHIPPED-PATH 0, SPILL-SPLIT 0), so the numbers are the shipped path's.

**CHANGED BY STEP 2 (2026-09-25):** `lake_census`'s watershed WHAT-IF is gone — the budget itself is the watershed
now, so the replica (`budget_rule`, `fill_level`, `deficit_rate`, CHECK-WHATIF) is deleted and the budget's own lines
print instead (CHECK-BUDGET-STALE, CHECK-BUDGET-CONSERVE, G-LAKE-SHARE, BUDGET-LEAVES). `terrain_descent` is deleted
(the census reads `lakes::Descent`). The inland-sea block is an independent CHECK of the solve's ocean (CHECK-OCEAN,
CHECK-POCKET-NOT-SEA). The rung-swap block reads three swaps and splits G-RUNG-LAKE into LOST, MIXED and SHIFTED, with
G-RUNG-HOLD beside them. The river set is read by the drainage area and, for comparison, by the step-1 rule.

**Read with care.** `G-HACK-n` can read GREEN on a dune field (radial flow into round troughs gives n ≈ 0.5); only
`G-HACK-c` under 1 separates it, so read both. The spill-depth, breach and hill-spacing lines have no published band in
our units: they are READINGS that step 3 (R2) must move, not gates. `G-TEETH-GRID` treats a shoreline's windows as
independent, so read it against `G-TEETH-GRID-GROUND`, the ground's own waterline in the same square.

### 8b. The "before" pictures (2026-09-24, generator 19, headless pilot capture, captioned "no far shadow")

One cluster boot on slot 37, the pilot camera at each stand, a picture only after the home planet's artifact was HELD
and the home ladder had nothing pending and nothing urgent (a far planet's six rung-18 chunks stay urgent while its
artifact is on the wire — the known far-body pop-in; the settle rule excludes rungs above the home ladder's own).
The sun cannot be stated: it follows the cluster's clock and reads the same at every fresh boot (the star at
`-0.0297 -0.9993 -0.0241` in the planet's frame).

| picture | the stand | the sun | what it shows |
|---|---|---|---|
| `pictures/reference_before.png` | the reference stand (§9), eye 400 m over the solved ground | 24.8° up, 141.7° off the nose | a flat tan plain to a flat horizon, faint ripples, no ridge against the sky (the skyline's 1.2° range) |
| `pictures/belt_before.png` | 150 km over the belt aim, nose inland, 20° down | 25.7° up | THE DUNES: ripples a few kilometres apart over the whole frame, no valley; the far-left coast a staircase |
| `pictures/coast300_before.png` | 150 km over a point 300 km inland of the coast stand, nose toward the sea | 65.7° up | THE TEETH: the shoreline a staircase of straight runs and 90° steps along the grid |
| `pictures/plain_before.png` | 150 km over the plain aim, nose inland | 35.0° up | THE DUNES again, the same spacing; a strip of sea on the upper-left horizon |

Refused and not kept: a first reference picture taken before the artifact landed (the recipe's own relief drawn),
and a coast picture from the coast node itself (the frame held only the sea: the shore stood under the camera).
The red ball is the ruler (slice 8p). The HUD's "over the recipe" and the finest rung 3 at the reference stand are
the CLIENT DEFECT of §3a, cured the same day.

**AFTER STEP 2 (2026-09-25, generator 20, artifact 8, NOT committed; the same poses, the same settle rule, one cluster
boot for the four stands and a second boot for the pocket stand, because one slot holds four clients; "no far
shadow").** The poses are the "before" poses, so the pictures compare the same place. The sun is the same.

| picture | the stand | what it shows, against the "before" picture |
|---|---|---|
| `pictures/reference_step2.png` | the reference stand | the same flat tan plain to a flat horizon; the eye now 779 m over the drawn ground (402 m before: the field under the pose moved, the pose did not) |
| `pictures/belt_step2.png` | the belt | the same dunes; ★ NEW: about fifteen small lakes, EACH ONE A FLAT BLUE SQUARE, one node (8 km) wide, on the dune ground — the node-square edge of T4, which step 2 made visible by making the lakes exist |
| `pictures/coast300_step2.png` | 300 km inland of the coast stand | the same staircase coast (step 4's work); a few small square lakes near the shore |
| `pictures/plain_step2.png` | the plain | the same dunes; ★ NEW: about forty square lakes, a chain of them along one grid line, and larger rectangles on the right horizon — the squares are the strongest new feature in the frame |
| `pictures/pocket_step2.png` | NEW STAND: 150 km over the deepest node of the largest pocket (1.57 million km², 11.9 km under the sea's level, DRY by its budget), nose across the sun (the pose from the census's aim, `scratchpad/pocket_pose.py`, the stand tool's own framing rule) | a dry tan basin floor with a few crater rings, NO sea; at generator 19 the client drew this ground as ocean at the sea's level (59 470 pocket nodes carried the sea's bit) |

**READ.** Step 2 did what it was for: the water budget is right in amount (3.51 % of the land) and no pocket is sea.
The dunes and the sea's teeth did not move, as planned. But the new lakes are drawn as 8 km squares, so on the belt
and the plain the pictures now show a NEW visible fault: blue tiles on the dunes. The cure is step 4's law (the lake's
edge where the ground crosses the lake's level, T4); the plan's order puts step 4 after R2. The owner's word is asked
on that order.


**AFTER STEP 2b (2026-09-25, generator 21, artifact 9, NOT committed; the same poses and settle rule; boot 1 the four
stands, boot 2 the pocket and a stand over the second pocket's deepest node, boot 3 a stand looking straight down from
1 000 km over the largest lake; "no far shadow").**

| picture | the stand | what it shows |
|---|---|---|
| `pictures/reference_step2b.png`, `belt_step2b.png`, `coast300_step2b.png`, `plain_step2b.png` | the four "before" poses | the same as step 2 (the heights did not move, MEASURED by `artifact_diff`): the dunes, the staircase coast, the square lakes; on the plain one lake is now a longer rectangle |
| `pictures/pocket_step2b.png` | the largest pocket | unchanged: a dry basin floor with crater rings |
| `pictures/basin2_step2b.png` | NEW: 150 km over the deepest node of the second pocket (642 554 km², 8.2 km under the sea's level, PARTIAL) | ★ A FLOOR NOBODY SAW BEFORE STEP 2 (it was drawn as sea at generator 19): ROWS OF REGULAR ROUND BUMPS in lines, like an egg carton, long straight water strips along grid lines, square lakes, and a tall straight wall at the lower right. Cause UNMEASURED |
| `pictures/highlake_nadir_step2b.png` | NEW: 1 000 km straight over the largest lake's edge node (12 562 nodes, level 5 983 m, 2.4 km over the sea) | ★ NOT ONE WATER SURFACE: blue and tan stripes with staircase and dotted diagonal edges. The probe: 84 % of the frame is rung 11 ground (2 048 m cells); the blue is the sheet over that ground where the ground stands under the level, and over the deep floors the sheet hides (not drawn) |

**MEASURED BESIDE THE PICTURES (`artifact_diff depth`, `step2b_lake_depths.txt`).** The lakes are shallow on the
whole: by area 35.7 % of the lake stands under 10 m of water at generator 21 (39.7 % at generator 20), the area median
depth 19 m (17 m). ★ The largest lake is NOT a thin sheet: its median depth is 75 m and 2.2 % of its nodes stand under
10 m. So the first reading of the nadir picture ("a level that stands on a flat") is REFUTED for that lake; the stripes
come from the far rungs' drawing (rung 11 reads the pyramid's block means under the block's highest lake word), and
their cause is UNMEASURED — owed to the shoreline bench's far-rung lines (CHECK-DRAWN-IS-SIDE and G-SEA-OVER-DRY at
rungs 10–13), with this lake and the second pocket added to the bench's stands.
---

### 8c. THE SHORELINE BENCH — plan step 4, slice 0 (2026-09-25, generator 21; built, run, NOT committed)

`cargo run --release -p vd-bins --example shoreline_bench -- --census docs/investigation/2026-09-25/shoreline_bench_census.txt
--panels docs/investigation/2026-09-25/shoreline_bench` (383 s with a fresh solve; 3.6 GB peak). An INSTRUMENT: no file under
`crates/terrain`, `crates/recipe`, `crates/seed`, `crates/wire` or `crates/client` changed (`git diff --stat` empty). For each
candidate rule it computes the DRAWN set — the drawn ground `s` and the drawn water `l` per column, wet by the sheet's own rule
`s < l` — at rungs 0–13 on eight stands (the belt, the coast, the plain, the reference stand, the largest pocket, a GENTLE coast
of slope 1/96, the second pocket's floor, the high lake), and paints 50 panels (a top-down hillshade with the drawn water, and an
oblique view; four tiles each: R0 today | C0 the pure contour / N1 | O1). Built by workflow `wf_b30fd2c2-c03` (its build agent was
stopped by the stall watch while it waited on a run; the code, 7 800 lines, was finished and checked in the main session).

**THE INSTRUMENT IS TRUE (self-checks, all GREEN):** CHECK-R0 176 of 176 (today's rule equals the shipped column read,
`vd_terrain::chunk::column_field`, bit for bit), CHECK-R0-F64 176, CHECK-SHEET-COLUMNS 72 of 72 (the column-drawn set equals
the sheet's own vertices), CHECK-CACHE-FIELD 176; the controls read a line 1.000, a Koch curve 1.27 against 1.262, fractional
surfaces at H 0.3/0.5/0.7 inside their bands. One stated weakness: CHECK-D-STAIR.KINK is RED — the kink gate cannot see an 8 km
staircase in a 31-node window, so the grid lean (G-SHORE-LATTICE) carries the staircase instead.

**THE DECIDING NUMBER (survey §6.4), the gentle coast, the shore's sideways step at a swap (band: p99 ≤ 1 px, max ≤ 2 px):**

| rule | swaps 0→1 … 8→9 | 9→10 | 10→11 | reading |
|---|---|---|---|---|
| R0 today | p99 1.35–2.27 px, max 2.5–4.1 px | max 2.5 px | **8.4 px** | the node square is steady, its wall is not |
| N1 the node-scale side, one field | **p50 0.00 px, p99 0.58–1.40 px, max 1.0–2.5 px** | **8.9 px** | **8.5 px** | steady until the pyramid; the far rungs fail |
| O1 the ground's crossing, a coarser cell from the line | p99 1.9–6.6 px, **max 3.3–8.9 px at nearly every swap** | 8.9 px | 8.6 px | a W16 breach at every rung |
| C0 the pure contour (control) | max 3.5–8.9 px | 8.8 px | 2.5 px | W6's measured shore creep, as predicted |

By the survey's rule O1 FAILS (its step is not bounded where its clamp binds: 95.5 % of the pairs at rung 7, walls on 416 of
7 358), so **N1 stays until R2**; and **no rule is steady at 9 → 10 and 10 → 11** — the far rungs read the pyramid, and their
cure (the far fold, V's crossing byte, the SL6 sizes) is the bench's second slice (0b).

**THE WALLS AT THE SHORE, rung 7 (a step across the drawn shore over 1 ‰ of the ground's own neighbour step):** R0 today walls
84–97 % of the shore pairs (the teeth; up to 36 × the ground's step on the coast); **N1: 0 of 688 (belt), 0 of 2 943 (coast), 32
of 5 572 (plain, 0.6 %), 92 of 4 518 (gentle, 2.0 %), the worst 3.3 ×**; O1: 5–16 %, the worst 14.5 ×. N1's remaining walls stand
where a mirrored node enters the read (v2's finding 1): G-SIDE-VS-GROUND reads p99 0 m on the plain but up to 72–104 m at a few pairs.

**THE LAKES' SHAPE, pooled over the stands at rung 7 (Earth's lakes 1.12–1.40):** R0 1.25 on 38 lakes (squares joined into
staircases read rough), C0 1.19 on 173, **N1 1.04 on 36 (too smooth: round ponds)**, O1 1.12 on 143. On the plain N1 passes every
drawn-set line (no water over dry ground, no lake that appears or flickers); on the gentle coast it fails a few at rungs 7–12.

**THE HIGH LAKE'S STRIPES, MEASURED (DEFERRED "A HIGH LAKE SEEN FROM 1 000 km …"):** (1) at rungs 7–9, 130 979 / 32 476 / 7 722
columns of the lake draw as LAND under today's rule: the drawn ground stands EXACTLY on the lake's level (the shore law holds a
column with no depth at its level, and the sheet draws only `s < l`) — N1 and O1 read 0 there; (2) at rungs 10–12 no column inside
the lake draws dry, but the sheet stands over 7 111 / 2 346 / 283 side-dry columns beside it; (3) ★ THE PANELS SHOW THE MAIN CAUSE
IN THE GROUND ITSELF: parallel ridges at exactly 45° cross the lake and the second pocket's floor, one node apart, with regular bumps
along each crest — the WASHBOARD of D8 routing on a smooth slope (Fairfield & Leymarie 1991: *"the derived water channels tend to
flow in parallel lines along preferred directions engendered by the sampling grid orientation"*; `lakes_and_landscape_models.md`
§3.4; DEFERRED "THE GRID STILL SHOWS IN THE VALLEYS"). No shore rule cures it; R2 would carve its 1 km valleys along it.

**WHAT THE PANELS SHOW THAT THE LINES DO NOT (read in the main session, all eight stands at rung 7, the gentle coast and the
plain at rung 3, two oblique views):**
- ★ **N1 FLATTENS THE GROUND AROUND EVERY LAKE AT CLOSE RANGE** (`gentle_r03_top_…`, `plain_r03_top_…`): at rung 3 (32 m
  cells, a pilot landing on a shore) each N1 lake stands in a wide disc of smooth, textureless ground that ends at STRAIGHT
  edges on the node lines — v2's findings 19 and 23 (the fade takes the whole fine relief; its reach is node-aligned). No line
  of slice 0 reads it (FINE-RMS-NEAR-SHORE and the halo's edge are owed readings).
- ★ **N1 KEEPS A SCALLOPED STAIRCASE ON A STEEP COAST** (`coast_r07_top_…`): where the rows jump kilometres between a shelf
  node and a deep-sea node, the smoothed field still crosses the water on the node staircase, with a regular notch at every
  node along the diagonal coast. A candidate cure for the bench's next round: CLAMP the node words to a band around the water
  before the read, so the side, not the depth, decides the line (a diagonal staircase of pure sides reads a straight line).
- **O1** keeps the texture but draws thin dark curves on the ground near its shores (`gentle_r03_top_…`), and it jumps at every
  swap.
- **The belt and the reference stand hold no structure at all at rung 7** (`belt_r07_top_…`, `reference_r07_top_…`): the
  dunes, even and valley-less — R2's work; no shore rule touches it.

**THE PANELS TO OPEN FIRST:** `shoreline_bench/plain_r07_top_R0-C0-N1-O1.png` (forty lakes: squares | noise ponds / round ponds |
ragged ponds; a chain of lakes along one diagonal in every rule), `gentle_r07_top_…` (the staircase coast | bays),
`pocket2_r07_top_…` (the washboard floor), `highlake_r07_top_…` and `highlake_r11far_top_R0-R0w-R0b.png` (the 45° stripes).

**Example.** From 150 km over the plain, N1 draws round ponds where today's rule draws blue tiles, and the ponds stay put as the
hull climbs to 700 km; at 704 km every rule's shore jumps about nine pixels. From 1 000 km over the high plateau the stripes are
the ground's own 45° washboard with lakes between its ridges — a routing fault of the 8 km solve, not of the shore.

---

### 8d. THE WASHBOARD BENCH — ruling B10 item 2 (2026-09-26, generator 21; built, run, NOT committed)

`cargo run --release -p vd-bins --example router_bench` with `vd_bins::router_variants` — an INSTRUMENT (no world code
changed). Stage 1 reads Tarboton's cone on the home moon; stage 2 solves the home planet with a LINE-FOR-LINE COPY of
`solve_full` whose drains, accumulation and (for the two-link and the fill variants) cut and deposit follow the variant; the
final routing, the lakes, the water's tree and the artifact are the solve's own. Evidence:
`docs/investigation/2026-09-26/washboard_bench/` (the census, 16 pictures).

**THE INSTRUMENT IS TRUE:** with today's D8 the copy gives the pinned artifact bit for bit (CHECK-COPY-IS-THE-SOLVE equal)
and 0 of 8 871 936 heights and 0 lake levels differ from `solve_full` (CHECK-STOCK); the copied cut, given the solve's own
one-receiver tree, equals the solve's sweep height for height (a unit test); the stage-1 D8 row repeats `grid_bias`'s
numbers of 2026-09-22 to the digit.

**THE TWO ROUTING CURES (B10 item 2), MEASURED:**

| reading | D8 today | Rho8 (seeded spread) | D∞ (two-neighbour split) |
|---|---|---|---|
| Tarboton's cone error, inward / outward, face middle (nodes²) | 57 768 / 63 | 53 934 / 275 | **9 886 / 16** |
| the same near a face corner | 244 218 / 57 | 75 944 / 226 | **33 793 / 100** |
| G-GRID's valleys (four bins; ~1.006 is no lean) | 1.205 | 1.238 | **1.116** |
| G-GRID's lakes (four bins) | 1.128 | 1.277 | 1.242 |
| the lake share | 4.828 % | 4.772 % | 4.966 % |
| the second pocket's floor (panel) | 45° ridge rows with beads | the same | the same |

★ **NEITHER CURE REMOVES THE WASHBOARD.** Rho8 helps nothing (its one draw a node is frozen over forty passes and adds
noise); D∞ improves the valleys and the cone error four to six times, but the 45° ridges on the pocket floor stand in both.

★★ **THE WASHBOARD'S CAUSE, FOUND BY A STAGE TRACE** (the second pocket's floor, 81 × 81 nodes, shaded after each stage):
the fine texture (a node less the mean of its eight neighbours) reads 3 m after the first land, 25 m after the craters,
**405 m after the FIRST erosion pass**, then 695 / 732 / 872 / 1 004 m after passes 5 / 10 / 20 / 40, its direction turning
from the regional slope to EXACTLY 45°; the pictures show one-node lines at 45° born in pass 1 and multiplying. A filled
hollow is never cut (the sweep skips it), so the lines are the DEPOSIT's: the solve carries each pass's cut down the tree and
fills the FIRST lake node the flux reaches up to its spill — and inside a filled hollow the flux runs along the flat's own
straight receiver paths, so the sediment lies in one-node lines along those paths, with a bead where each path enters. The
routing cures change the drains of the SLOPES and keep the flat's paths and the node-by-node deposit, which is why neither
touched the ridges.

**A THIRD CURE, THE BASIN FILL (measured):** the cut that reaches a filled hollow is held by the whole hollow, which fills from
its lowest nodes to ONE level, never over its spill, and only a surplus runs on (`basin_fill_deposit`; ruling W7's "the cut
is laid in the first hollow downstream" kept). The pocket floor's fine texture after 40 passes: **21 m against 1 004 m**, no
45° line (`trace_pocket2_fill_02_pass040.png`). ★ BUT THE LEVEL FILL MAKES DEAD-FLAT PLAINS, and a dead-flat plain floods
whole: the second pocket becomes ONE lake across the whole panel, the high lake becomes water blocks with straight node-line
edges (`highlake_r07_fill.png`), the largest lake grows 12 562 → 43 776 nodes, the lakes 3 868 → 1 764 patches, the lake share
5.04 %, and the planet-wide grid lean WORSENS (valleys 1.281, lakes 1.711; the finest texture's row lean 14 % → 27 %: a level
plain holds its one-step terraces along the rows). D∞ with the fill reads the same (1.335 / 1.566 / 5.15 %).

**READ.** The washboard is a deposit fault, not a routing fault. A cure must lay the sediment the way a real basin fills: from
where the rivers enter, the floor sloping gently down to the lowest part, which alone keeps the water — never on the inflow's
straight path (today) and never as one dead-flat level (the level fill). D∞ is a separate, measured gain for the valleys.

**Example.** From 150 km over the second pocket a pilot sees today rows of beaded ridges at 45°; with the level fill she sees one
flat lake to the horizon; the owed cure should show a basin whose floor falls gently from its rim streams to a smaller lake
in its lowest part.

---

## 9. THE REFERENCE TARGET — the checklist every step reports against

**THE REFERENCE STAND.** One more stand beside the belt, the coast and the plain, framed like the reference: an eye
~400 m over its local ground on a high point near the belt's range (the home planet needs ~373 m to see 50 km of
ground, DOC 01 §2), looking across valleys toward the range 30–60 km away, a ~55° field of view (DOC 11 §1), the sun
low and across the view. **POSE (chosen by `reference_stand`'s rule, 2026-09-24, generator 19):** node 8 857 505 (face
NegZ, node 161, 1 204), direction 0.631321 −0.422643 −0.650236, 131 km off the belt aim; the ground 8 306 m over the
ladder radius (~4 740 m over the sea), the eye 400 m over it; the look at bearing 243° of the eye's tangent basis
(direction 0.690995 0.687205 0.224222), the view axis −12.4°; the sun wanted 12° up and 120° off the nose
(`VD_STAND_SUN="-0.504935 0.076557 -0.859755"` from the right). The spawn line is in `step1_census_gen19.txt`.
⚠ THE RULE PICKS A SUMMIT: the eye stands over its ring by ~700 m and looks DOWN on its range, while the reference looks
from a lower point toward taller snowy ranges. The owner's look at the picture decides whether the rule changes.

★ **A SECOND REFERENCE FRAME (the owner, 2026-09-25: *"As a reference that we need to get to is the"* picture, with
*"After those fixes please make sure that you've also checked all visuals"*).** Kept outside the public repo like the
first (ruling B6); described here so the checklist can read it. A viewpoint on a tall pale rock pillar high over a wide
valley country, looking across it to a far range under a cloudy sky:
- **the valleys**: a dense web of branching valleys and ridges at 1–5 km, the slopes dark with forest in patches, the
  valley floors and the tracks lighter; one winding river corridor and small bright lakes in the far valleys;
- **the relief**: hills a few hundred metres over their valleys in the middle ground; snowy peaks and a long snowy ridge
  on the left; a far range with steep ROCK TOWERS and mesa-like blocks on the horizon (8f's third dimension);
- **the dressing**: conifer forest as a mass with clearings (8e + the art assets), snow on the high ground and on the
  rock pillar's top (8e), settlements in the valleys (not the terrain arc), cumulus clouds and a hazy far range
  (the weather plan, the sky's haze);
- **what it has and we lack today** (the pictures of §8b and the bench's panels of §8c): no branching valleys at 1–5 km
  (we draw dunes — R2/R3), no river corridor (the drawn rivers are retired, W16 A), the lakes on the grid (squares,
  and chains along a diagonal — step 4's edge, and the grid lean), no forest, no snow, no rock towers, no clouds.

**Example.** From the first reference's summit the pilot sees ranges; from this one she sees the valley country under
it — the believability arc owes the valleys (R2, R3) and the water in them (steps 2–4); 8e and 8f owe the rest.

**THE SKYLINE NUMBERS** (11_reference_trace's rule, per 60° of bearing):

| reading | the reference | ours |
|---|---|---|
| breaks at 0.25° | 2.5 – 5.5 | **0** (the reference stand, 55° sector and 360°, generator 19); 0 at M1b; **0 at generator 20** |
| breaks at 0.50° | 0.0 – 3.3 | 0 (generator 19); 0 (generator 20) |
| largest rise over the ±5° trend | 0.43° – 1.07° | **0.200°** (generator 19); 0.054° – 0.064° at M1b; **0.096°** (generator 20) |
| the skyline's range over the view | 4.87° – 7.49° | **1.21°** (generator 19); **0.90°** (generator 20) |
| the relief 30–60 km away | ridges 1 500 – 3 000 m tall (01 §2) | 1 988 m (shipped ground, generator 19); 1 612 m (generator 20) |

★ **STEP 2 MOVED THE REFERENCE STAND'S NUMBERS AWAY FROM THE REFERENCE** (ruling B6 item 2 asks each step to say so):
the pose rule re-picked the stand with the field — node 8 857 509 (4 nodes from generator 19's), the eye 8 327 m over
the ladder radius, the look at bearing 27° — and its skyline is flatter: the largest rise 0.200° → 0.096°, the range
1.21° → 0.90°, the relief 30–60 km 1 988 → 1 612 m. Step 2 changed the water, not the relief; the new pose stands on a
slightly different summit. The spawn line and the sun for the picture are in `step2_census_gen20.txt`. The picture
itself is OWED (§8b).

**THE FEATURES** (scales from 01 §2):

| in the reference | its scale | who supplies it | state today |
|---|---|---|---|
| snow-capped ridges on the skyline | 1.5–3 km tall, 30–60 km away | the solve's relief (8c) + R2's ridges + the snow (8e reads the ice bit) | relief at 16+ km only; snow from a placeholder |
| valleys that join into drainage | 200–800 m deep, 1–5 km apart | R2 (to ~4 km), R3 (1–5 km) | ❌ noise, no drainage (§5) |
| hills that are ridges, not ripples | 1–5 km | R2 + R4 (noise only below the solved level) | ❌ the dune bump (§5 D3–D5) |
| a lake in a valley | ~1–3 km | step 2 (budget) + step 4 (shoreline) + R3 (scale) | ❌ 8 km squares, too few (§7) |
| a river with a valley floor | 20–100 m wide | R2/R3 + step 6 (outlets) + 8o (the surface) | ❌ none drawn |
| cliffs, rock pillars, mesas | 20–200 m vertical | 8f | ❌ not built |
| bedded rock in the faces | bands 1–20 m | 8d beds (built: hardness, cap-rock bench) + 8f faces | partial |
| forest as a mass over hillsides | canopy 15–30 m | 8e biome + V4 art assets + the canopy fold | ❌ not built |
| long shadows across the slopes | kilometres | the far shadow (8s, the `perf` tree) — wanted, NOT a prerequisite (B6.5 amended) | ❌ no shadow above 3.5 km |
| blue haze with distance | 20–50 km | 8s | ✅ built |
| clouds | 1–8 km up | the weather plan | ❌ not built |
| detail to the horizon | ~50 km | the ladder | ✅ built |

**Example.** After step 3 the reference stand should show ranked ridges with valleys between them, and the skyline
count at 0.25° should rise from 0 toward 2.5–5.5. If a step turns a gate green but moves the reference-stand picture
AWAY from the reference, the step's report says so (ruling B6).

---

## 10. Where the docs and the code disagree (fix the doc when the step touches it)

- The rulings say the noise band is "under 8 km"; the code makes it under 32 768 m (`body.rs:94-96`); the comment at
  `body.rs:86-92` says 8 224 m / 32 896 m.
- W14.3 describes `FieldRead::fine_ceiling` and `cut`; W16.1 deleted them, and the code agrees with W16.1.
- `river.rs:14-15` and W16.1 say the solve uses the width law; nothing calls it outside its tests.
- ~~`artifact.rs:1053-1055` says a land column beside a lake takes the lake's level~~ — FIXED by step 2: the doc of
  `column_water` now states the land rule (the sea, or no water inland; never a lake's word).
- ~~`lakes.rs:40-41` says the northern hollow "takes a river off the range"~~ — TRUE since step 2 (the watershed).
- ~~`drain_census.rs:3-8` says it counts minima "on a drainage line"~~ — FIXED 2026-09-24 (the header now states its blind spot).
- W11's "a brim-full lake has no dry catchment left to feed it" describes the gen-19 rule, not Fill-Spill-Merge; since
  step 2 the code feeds each hollow from its watershed.
- The step-2 plan said the drainage-area river rule reads "the same river set as at gen 19"; MEASURED it does not (the
  step-1 rule weighted a node by its own rain): the census prints both.
- W14.4/W14.6 say the octave spectrum ends at 781 m; the fine octaves reach 48.83 m (781 m is the ridged band's floor).

---

## 11. Questions still open for the owner

1. ~~The measuring bench~~ — ANSWERED (B6.5): the `perf` tree is testing; step 1 runs when it is idle.
2. ~~The pictures~~ — ANSWERED (B6.5, amended): *"We can go without long shadows for now"* — the pictures are judged as drawn, each captioned "no far shadow".
3. ~~An inland hollow under the sea's level~~ — ANSWERED (B7): a closed basin by its budget; the sea over the connected ocean. MEASURED: 148 pockets, 3.2 million km², the
   largest 1.35 million km² and 11.8 km deep — enclosed seas, not puddles.
4. ~~The discharge row~~ — ANSWERED (B7): the runoff, over the final receivers.
5. R2's skin: which octaves stay below the solved level and how tall; does the placeholder roughness retire (T9);
   may each octave get its own rotation (every address moves)?
6. From earlier rulings, still open: the W6 quarter, the W5 cap share, the W17.10 statistic (max or p99).
7. **Step 2's own questions** (plan §9; each BUILT as the plan recommended unless stated, the owner's word owed):
   Q1 the ocean = the largest connected water, ties to the smaller index (built); Q2 a pocket's air on its own ground
   (built); Q3 the discharge row — A, the runoff over the receivers literally, is BUILT (B7's words); B, the budget's
   own water, is recommended and ties to W11 in §7 (MEASURED after the review fixes: the row carries 1 607 km³/yr to
   the sea that the budget holds) — ★ ANSWERED B (ruling B8 item 2), ★ BUILT as plan step 2b (generator 21: the row the
   cascade's to the unit, §7 W10); Q4 the passes cut with the rain (unchanged); Q5 the dry-land hold (built: near the
   sea W6's lift, inland only over the sea's level, and a pocket's floor its own word — MEASURED G-RUNG-HOLD 19
   columns, 5 after the review fixes, §7); Q6 SHIFTED lakes (a one-pixel bound read by the census: 0.016 px); Q7 the
   coast bit as the storm's reach — NOT built, the coast bit still reads `|z − sea| ≤ band` on every node, a pocket
   floor near the level included (the beach now reads the HOLD, so a floor under the sea's level is no beach); Q8 the lakes' sea fixed
   point (built: 1/16 m, at most 4); Q9 the first land's pockets (MEASURED: 3 pockets, 17 799 nodes; the province rule
   names a dry pocket's continental floor a SHELF, unchanged).
8. ~~**New by step 2:** should the shipped receivers follow the budget's water (§7 W11)?~~ — ANSWERED (B8 item 2, with
   question 7's Q3 choice B): yes, before R2, as plan step 2b.
9. ~~The order of the edges and R2~~ — ANSWERED (B8 item 1): the edges (step 4) before R2. ~~R2's start~~ — ANSWERED
   (B8 item 3): slice 0, the bench and a hillshade for the owner's look, before any bake service.
10. **New by step 2b (2026-09-25): THE LAKES OVER THEIR CEILING.** To make the shipped row the budget's own water
   (B8 item 2) the cascade had to follow the ground (plan questions 1 and 2, choice A) — MEASURED, that alone makes the
   row exact (V3 below). The flats were swept from their drains (question 3, choice A) for item 1's flat directions and
   to close the 8 795 leaves of zero depth that held 462.7 km³/yr on flats that drain — NOT for the row (corrected by
   the review, 2026-09-25: this item said before that the row needed the flats). With the flats and the brim law the
   lakes cover **4.828 %** of the land against G-LAKE-SHARE's ceiling of 4 % (ruling B3; 3.510 % at generator 20). MEASURED attribution (§7 W11): 3.510 %
   → 4.162 % (the steepest open side and the ground's way to the sea) → 4.741 % (the flats) → 4.828 % (the brim). The
   plan's own risk line says: over 4 % is red, stop and report. The choices:
   A. **Keep all of step 2b and rule on the ceiling** (the budget is Fill–Spill–Merge's own and every B8 row gate is
      green; G-LAKE-SHARE stays RED until the owner restates the ceiling or a later law — 8e's climate — moves the water).
   B. **Keep only the water's tree over generator 20's budget** (questions 1–3 choice B): G-LAKE-SHARE 3.510 %, but the
      row into the sea is 188.2 km³/yr short of the cascade's and 632 closed bodies carry more than they hold (MEASURED),
      so B8 item 2's "the row's sum into the sea equals the cascade's" does not hold.
   C. **Keep A's laws and look for a law the budget still lacks** before the ceiling is read again, each candidate
      measured before any number moves (T9: no drawn number); none is named here, because none is measured.
   D. **Keep questions 1 and 2 (choice A) and drop the flat sweep and the brim** (`step2b_census.txt` V3, MEASURED;
      added by the review, 2026-09-25): the row EXACT (ROW-SEA equal, ROW-HELD 0 of 10 342, 0 of 4 982 closed hollows
      leak), G-LAKE-SHARE **4.162 %** (still over 4 %, by 0.162 points instead of 0.828), G-GRID-VALLEYS 1.093 / 1.132
      outside the filled hollows (1.205 / 1.337 with the sweep). ITS COST: a flat is swept in index order again, so the
      8 795 leaves of zero depth come back (at generator 20 they held 462.7 km³/yr on flats that drain), a filled flat's
      water ENDS at such a leaf, and where it runs on its direction follows the index tie-break, not the drain; V3 ran
      before the saddle's own-descent order, and what that order moves without the sweep is UNMEASURED.
   ★ A MEASURED FACT FOR ALL FOUR: no measured variant that keeps B8's exact row stays under the ceiling; the smallest
   is D (4.162 %), and the flats and the brim add 0.666 percentage points on top of it.
   ★ ANSWERED 2026-09-25 (ruling B9 item 1): A — keep all of step 2b; the ceiling is read again on the pictures after
   the shoreline step. RECOMMENDATION WAS: A — the laws are the published ones, the rows agree with the budget exactly, and the ceiling is a
   believability bound whose owner is the owner; the pictures after step 4 show whether 4.8 % reads as too wet. But
   read item 11 with it: the flat sweep that A keeps does not yet meet item 1's flat law.
11. **New by step 2b's review (2026-09-25): ITEM 1's FLAT LAW IS NOT MET.** The plan says the directions over a flat
   lake surface or a filled flat must not lean on the lattice, and G-GRID reads them. MEASURED (gen 21): G-GRID-VALLEYS-
   BY-CLASS reads the flats 1.470 and the spilling lakes 1.553 against ~1.006, and G-LAKE-WALK reads a mean error of
   14.07° against test T9's 12° — both printed NO-BAND before the review, both RED now. Like for like, the flat sweep
   made the aggregate WORSE (V3 1.093 → 1.205), and the walks lean 1.605 where the lakes' own straight lines to their
   spills lean 1.425. Read in the code (`lakes::flats`, `flat_step`): each flat node aims at ONE target, its nearest
   drain by the summed chord — Barnes, Lehman & Mulla's first field only (DEFERRED "THE FLAT STEP READS ONE OF
   BARNES' TWO FIELDS"). NOT CURED by the review: a cure changes the flat step, which moves which hollow a flat's
   water reaches, so it moves the budget and the lake share on top of item 10's open question; whether Barnes' two
   combined fields read under 1.05 on node-square flats is UNMEASURED. The choices:
   A. **Cure the flat direction before step 2b is committed** (Barnes' two combined fields, or a target that does not
      collapse onto one nearest drain), measured one change at a time against V3 and today, then item 10 read again.
   B. **Commit step 2b with the flat law RED** and register it for step 5 (G-GRID on the R2 field, with the D8
      decision), where the flats are no longer node-square pits of the 8 km solve.
   C. **Restate the bound** for a lattice whose flats and lakes are node squares (for example the lakes' own straight
      lines, 1.425, as the lake class's control) — the owner's word, because a bound set after a reading is not a gate.
   ★ ANSWERED 2026-09-25 (ruling B9 item 2): commit step 2b with this RED, and cure the flat directions BEFORE R2
   carves any valley (G-LAKE-SHARE read again with the cure).

---

## 12. Change log (newest last; one line per change that touched this file's items)

- 2026-09-24 `e69ed91` — written from the read-only workflow `wf_1780e367-7e9`; plan agreed in general (ruling B6).
- 2026-09-24 `e69ed91` — the owner's answers (B6.5): the bench waits for the `perf` tree; the far shadow becomes plan step 0.
- 2026-09-24 `e69ed91` — B6.5 amended: *"We can go without long shadows for now"*; plan step 0 removed, the far shadow stays in D-TERRAIN-8.
- 2026-09-24 `e69ed91` — step 1's four instruments WRITTEN (not built, not run; §8a); G-HYPSO and G-RIVER have instruments; the `drain_census` header fixed.
- 2026-09-24 `af21326` — rebased onto main (the `perf` tree's merge; no generator change); step 1's instruments built, linted, proven equal on the three moved tools, and RUN: numbers into §2, §4–§9, `step1_census_gen19.txt`; the reference stand's pose recorded; the pictures owed.
- 2026-09-24 `af21326` — the four "before" pictures (§8b, `pictures/`); the client's eye-ground defect found and cured (§3a); the picture gate registered stale since 8c.
- 2026-09-24 `af21326` — the eye-ground cure reworked after its review (the water sheet, one lookup order, the ruler's key, two guards of the defect) and PROVEN in flight on the reference stand (§3a).
- 2026-09-24 — committed on branch `game` in two commits: the eye-ground cure, then step 1 (the instruments, the numbers, the pictures and these documents).
- 2026-09-24 — §3b: the dark tears at a ring's far edge measured (closed caves shaded as caves) and cured; two instruments (`morph_normal_probe`, `ray_speck`).
- 2026-09-24 — §3c: the fast-low-flight cracks measured (skirts shorter than the coarser sink) and cured; skirts hang into the rock their edge stands on (the owner's editing question); `skirt_gap`.
- 2026-09-24 — the owner's window flights confirm §3b and §3c; committed.
- 2026-09-24 — ruling B7: the inland pockets are closed basins, the discharge row is the runoff; step 2 built and R2 designed in two tracks.
- 2026-09-25 — step 2 BUILT (not committed): generator 20, artifact 8; the connected ocean, the watershed and the cascade, the runoff row, the level-1 lake word, the inland column; pins re-recorded; §2, §3, §4, §5, §6 T5 T7, §7 W3 W4 W6 W10 (+ W11 new), §8, §8a, §9, §10, §11 updated from `step2_census_gen20.txt`. Gates MEASURED green: `terrain-pin` (debug, release, native), `artifact-pin`, `charter-pin`, `gpu-drift` (5 979 of 5 979 boxes); coverage of vd-recipe and vd-terrain: 0 misses on the lines step 2 touched, 103 misses elsewhere (the HR5 pass ruling B4 owes); the pictures OWED.
- 2026-09-24 — R2 designed (`r2_design.md`) and attacked: 23 findings open in its §16; slice 0 (a measured bench and a hillshade for the owner) before any service.
- 2026-09-25 — step 2's review findings applied (NOT committed): the pocket's own mask word (G-POCKET-HELD 0; G-RUNG-HOLD 19 → 5 at 9 → 10), far chunks wait for the coast mask, the beach reads the hold, the connected sea states an unsettled answer, the lakes' sea fixed point extracted and tested, the cascade's skip over full rings, the instruments read each column's own water, the census's land is every node that is not the ocean, the row-against-the-budget line (question 3 B still owed); DEFERRED D-TERRAIN-9 entries for the far-swap lake losses, the row, the unsettled sea and the far view. The artifact digest re-recorded (the mask's word 2); the sea, the share, the golden fields and the identity unchanged (MEASURED by `golden_z_record`); `artifact-pin` and `terrain-pin` (debug, release, native) green. ★ THE HR5 BASELINE (review finding 12), `coverage_gate.py` over `cargo llvm-cov --branch -p vd-recipe -p vd-terrain` (the `coverage-fast` filters, nightly-2026-06-06): at HEAD `a985e5c` 137 real misses (81 lines, 56 branches; 339 tool rows), every one in vd-terrain — position.rs 34, solve.rs 31, artifact.rs 28, height.rs 11, climate.rs 11, lakes.rs 6, craters.rs 6, land.rs 4, chunk.rs 4, strata.rs 2; with step 2 and its review fixes 113 real misses (65 lines, 48 branches; 284 tool rows) — position.rs 34, artifact.rs 22, solve.rs 13, height.rs 11, climate.rs 11, craters.rs 6, land.rs 5, chunk.rs 5, lakes.rs 4, strata.rs 2. Compared by file and line text: 26 of HEAD's misses gone, 2 new ones, both a short-circuit in test code, removed and MEASURED covered. vd-client's coverage (Tier-A as well) is UNMEASURED.
- 2026-09-25 — step 2 checked after its review fixes: `cargo clippy --workspace --all-targets -D warnings` clean; `gpu-drift` (5 979 of 5 979 boxes) and `charter-pin` (4 of 4) green again on the fixed tree. The step-2 PICTURES taken on the four "before" poses plus a new pocket stand (§8b): the lakes exist and the pocket is a dry basin, but every small lake draws as an 8 km square (T4); the owner's word asked on moving step 4's edge law before R2.
- 2026-09-25 — ruling B8 (the owner chose each recommended option): the order becomes 1 → 2 → 2b → 4 → 3 → 5 → 6 → 7 (the edges before R2; the rivers follow the water as step 2b); R2's slice 0 first; step 2 committed as generator 20's checkpoint.
- 2026-09-25 — step 4 (the shoreline) designed, v1 (`shoreline_design.md`), and attacked: 23 findings (6 HIGH) open in its §18; a v2 round started (read-only) that answers them and measures two variants in slice 0.
- 2026-09-25 — step 2b BUILT (NOT committed): generator 21, artifact 9; the water's own tree and the budget's own row (`lakes::water_tree`, `MacroSolve::follow_the_water`), the cascade's steepest open side and the ground's way to the sea (`lakes::open_side`), the flats swept from their drains (`lakes::flats`), the brim; §2, §3, §4, §7 W10 W11 (+ "MEASURED (step 2b)"), §8 row 2b, §11 items 7 and 10 updated from `step2b_census.txt`. MEASURED green: G-RIVER-DESCENT 0, W10 0, ROW-SEA equal, ROW-HELD 0, the checks 0; RED: G-LAKE-SHARE 4.828 % — the plan's stop condition, the owner's word owed. Pins re-recorded by their own tools (`golden_z_record`, `VD_ARTIFACT_RECORD=1 … home_artifact_pin`; `DECLARED_PIN` from its failing assert, it has no recorder); the solved heights, the sea's metre, the ocean share and the identity's measured half unchanged (MEASURED); `artifact-pin`, `terrain-pin` (debug, release, native), `charter-pin`, `gpu-drift` (5 979 of 5 979) green.
- 2026-09-25 — step 2b VERIFIED on the finished tree (NOT committed; one job at a time, release builds): `step2b_census.txt` ASSEMBLED from final runs of `lake_census`, `shipped_drainage`, `shore_teeth`, `reference_stand`, `artifact_diff` and `golden_z_record` (11:41–12:00). `lake_census`'s CENSUS lines equal the earlier step-2b run line for line (G-LAKE-WALK is now a reading, NO-BAND: test T9's 12° bounds a ROUND lake, the planet's lakes are node squares). `golden_z_record` wrote the golden file byte-identical, the identity, the sea (3 591 m), the share (7 811) and the digest unchanged; `artifact_diff` 0 in the solve's heights, the rows' heights, the province, the pyramid's heights and the sea's metre. `reference_stand` unchanged; `shipped_drainage`'s coast stand and both `shore_teeth` stands MOVED (`pick_stands` reads the discharge class; the largest lake grew), §4. Green: `cargo test --release` on vd-terrain (257 + the pins; 258 under coverage after the two tests below), vd-recipe, vd-bins `--lib`, vd-client; `artifact-pin`, `terrain-pin` (debug, release, native), `charter-pin`, `gpu-drift` (5 979 of 5 979); `cargo clippy --workspace --all-targets -D warnings`; `cargo fmt --check`. ★ HR5 (`coverage_gate.py` over `cargo llvm-cov --branch -p vd-recipe -p vd-terrain`, nightly-2026-06-06, after `llvm-cov clean`): the first run found 2 misses on lines step 2b touched (the closed flat's rim test never met a corner's missing eighth neighbour; the empty ring never had an open first hollow) — two tests added (`a_closed_flat_round_a_cube_corner_is_one_hollow_with_its_pit_at_the_corner`, the third case of `a_saddle_of_three_sides_pours_into_its_steepest_open_side`); the second run: 111 real misses (65 lines, 46 branches) against HEAD's 113, 0 on a touched line, 0 new by file and line text. The far-swap lake losses corrected to 4 501 (was written 4 497).
- 2026-09-25 — step 2b's review findings applied (NOT committed; no generator change, the census numbers unchanged but the verdicts and one new line): G-GRID-VALLEYS-BY-CLASS and G-LAKE-WALK carry real verdicts and read RED (the flat law of item 1 is NOT met, §11 item 11 new; §2, §3, §4, §7, §8 row 2b corrected, and the report compares G-GRID with V3, not with generator 20's routing tree); §11 item 10 corrected (the row never needed the flat sweep; option D, V3's 4.162 %, added) and the B8 note in the ruling file with it; `Depression::outflow` deleted (it counted a shared saddle's water twice; T1 reads `passed`, T7's second case reads the row at the shared saddle); the split's chance has its own test (`the_split_takes_the_farther_neighbour_with_the_chance_p1_over_p1_plus_p2`, observed failing with p1 and p2 swapped: 12 737 of 20 000); T9's comment states its f64 cosine; the sea's shift MEASURED in sixteenths (57 453 at gen 20, 57 449 at gen 21, from 57 462; tag.rs and §7); the coast, the plain and the lake stand PINNED and both instruments re-run on them (§4). DEFERRED: the flat-step entry restated with the two RED gates; the lakes' ceiling entry with option D; the stands entry 🟩. Checks (release, one job at a time): `cargo fmt --all --check`; `cargo test --release -p vd-terrain` (259 + the pins), `-p vd-bins --lib` (85); `home_artifact_pin` green (the digest unchanged); `cargo clippy --workspace --all-targets -D warnings` clean. HR5: the full `cargo llvm-cov --branch -p vd-recipe -p vd-terrain` run (after `llvm-cov clean`) read 113 real misses, 2 of them on touched lines — the new test's two assert-message lines — cured by hoisting the two values; a targeted `llvm-cov` run of the new and the two edited tests then reads 0 line and 0 branch misses in them. The full gate was NOT re-run after that cure (expected 111, UNMEASURED).
- 2026-09-25 — two surveys (read-only): `docs/investigation/2026-09-25/terrain_models_and_crates_survey.md` (the owner: "ok not to apply any open sourced solutions" — no library adopted; the cross-host solve identity found UNMEASURED) and `shoreline_papers_survey.md` (the coast and lake laws as bench gates; the LOD-continuity methods against the design's findings; §6 the bench's rules, lines and deciding number).
- 2026-09-25 — step 2b's pictures (§8b): the four stands as at step 2; two NEW stands show faults hidden until step 2 (the second pocket's floor: regular bump rows and a straight wall; the high lake from 1 000 km: stripes, not one water surface); the lake depths measured (`artifact_diff depth`): 35.7 % of the lake area under 10 m, the largest lake a 75 m median — the thin-sheet reading refuted; the far-rung cause owed to the shoreline bench.
- 2026-09-25 — ruling B9: the lake ceiling kept RED and judged after step 4; the flat directions cured before R2; step 2b committed (generator 21, artifact 9).
- 2026-09-25 — step 4 slice 0: the shoreline BENCH built and run (§8c; the census and 50 panels in `docs/investigation/2026-09-25/`); N1 steady at rungs 0–9 and nearly wall-free, O1 a W16 breach, every rule red at the far swaps, the washboard (D8) found as the main cause of the stripes and the bump rows; NOT committed.
- 2026-09-26 — ruling B10: the bench committed; N2 (one more bench round) for the shore; the washboard's two cures (seeded Rho8, D∞) measured before R2 and FIRST; then N2, then both built, then the flight of all stands against both references.
- 2026-09-26 — the WASHBOARD BENCH (§8d): the copy of the solve proven exact; Rho8 no help, D∞ better valleys (1.205 → 1.116) and cone error, neither touches the washboard; a stage trace finds its cause in the DEPOSIT (one-node lines along the flat's paths from pass 1); the level basin fill removes the ridges (1 004 m → 21 m) but floods whole flat plains; NOT committed.
- 2026-09-26 — ruling B11: the washboard is the deposit's; a sloped basin fill with a published slope law, measured first; D∞ adopted with it in one world change; the washboard bench committed.
