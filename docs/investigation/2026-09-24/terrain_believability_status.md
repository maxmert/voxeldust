# Terrain believability — THE LIVING STATUS of point 8

> **A LIVING DOCUMENT (ruling B6, 2026-09-24).**
> - **Update rule.** Every change that touches an item in this file updates this file IN THE SAME CHANGE: flip
>   the state, correct each number that moved, and add one line to the change log (§12). A reader must never be
>   misled by this file. When the code and this file disagree, the code wins and this file is the defect.
> - **Delete rule.** Delete this file when every step in §8 is done and the owner gave G-LOOK. The rulings
>   (`docs/design/owner_decisions_*.md`) and `docs/design/DEFERRED.md` (D-TERRAIN-8, D-TERRAIN-9) keep the history.
> - **Written** 2026-09-24 on branch `game` at `e69ed91`: GENERATOR_VERSION 19 (`crates/terrain/src/tag.rs:174`),
>   ARTIFACT_VERSION 7 (`crates/terrain/src/artifact.rs:77`). **Updated for step 2** (2026-09-25, on `a985e5c` plus
>   the uncommitted step-2 change): GENERATOR_VERSION 20, ARTIFACT_VERSION 8, `PROTO_MINOR` unchanged.
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
  word on question 3 B is owed.

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
| B2-1 the lakes from a budget | BUILT; ★ the WATERSHED and the CASCADE by plan step 2 (Fill–Spill–Merge; §7 W3, W4 cured, generator 20). The shipped receivers still follow the routing fill, not the cascade (§7 W11). | `terrain/src/lakes.rs` `budget`, `descent`, `cascade` |
| B2-2 the summer ice line, the crater age | BUILT. No drawn reader reads the ice bit; the snow in the pictures comes from the placeholder biome. | `terrain/src/climate.rs:560`; `terrain/src/craters.rs:360`; `recipe/src/height.rs:501-529` |
| B2-3 the flats, the scratch instruments | PARTIAL. Flats by summed chord. D8 stays. G-GRID RED. | `terrain/src/solve.rs:784-934`; `bins/src/grid_bias.rs` |
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
| G-LAKE-SHARE | GREEN, near its CEILING | **3.510 %** of the land (every node that is not the ocean; MEASURED gen 20); 0.068 % at gen 19. The plan predicted 2–3.5 % |
| G-LAKE-COUNT | read, verdict not restated | **3 883 patches**, median 1 node, p90 10 nodes, the largest 6 398 nodes (429 363 km²) (MEASURED gen 20; 506 at gen 19); the power-law fit UNMEASURED |
| G-BASIN | GREEN | the water bodies DRY 9 599 (56.5 %), PARTIAL 1 549, SPILLING 5 839; the budget's books EXACT: runoff 5 447.10 km³/yr = held 1 607.37 + to the ocean 3 839.73 + stranded 0 (MEASURED gen 20; DRY 97.5 % and SPILLING 0 at gen 19) |
| G-ICE | GREEN as a law test | 11.29 % on the Earth-like charter (DOC W14.5); the home planet reads 3.75 % (MEASURED gen 20; 4.02 % at gen 19) |
| G-BUZZSAW | GREEN | 1 295 m (MEASURED gen 20; 1 316 m at gen 19) |
| G-CRATER | GREEN on the count | 1 668 (gen 20; 1 610 at gen 19) against ~190; the size floor 16.4 km (`terrain/src/craters.rs:104`) |
| G-GRID | **RED, WORSE** | valleys **1.484** by the drainage-area rule, 1.522 by the step-1 rule, and **1.306 OUTSIDE every hollow the routing filled** (MEASURED gen 20); 1.284 at gen 19 against 1.006. 193 k of the 269 k trunk nodes now run across FILLED hollows' flats — the pockets the flood now raises — and the flat's distance field leans on the grid. Lakes 1.112 (842 patches) |
| G-HYPSO | GREEN on two humps and on the age; NO BAND on the share over 1 000 m | humps at −9 242 m and +8 m over the sea; basin integral median **0.590** (Strahler 0.35–0.60: at its top edge); land 21.24 % of the surface, its mean +1 546 m; the sea floor's mean −6 911 m; 63.11 % of the land over 1 000 m (MEASURED gen 20; 0.484, 21.41 %, +1 399 m, 55.12 % at gen 19) |
| G-RIVER | **RED on DESCENT** (every river ends) | 6 752 heads (drainage-area rule): the ocean 3 513, a closed lake 1 588, a dry closed basin 1 651, a pocket drawn as sea **0**; **3 430 trunk steps RISE, every one inside a hollow the routing filled, the worst 4 987 m** (§7 W11). The step-1 rule: 7 587 heads, 4 467 rising steps (MEASURED gen 20). Gen 19: 37 966 heads, 0 rising steps, 21 130 dry-basin ends |
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
  still cut with the rain's drainage (plan question 4).
- **W11. THE SHIPPED RECEIVERS LEAVE A SPILLING LAKE UPHILL** (found by step 2's census). The receivers are the routing
  fill's (ruling W11: a scratch surface). The fill raises a whole nested hollow to its OUTER spill, and its flat's
  receivers carry a full inner hollow's river across the parent's DRY ring toward the parent's own outlet — while the
  budget pours that water over the inner saddle into the sibling. Before step 2 no hollow spilled, so the walk never
  went there. MEASURED gen 20: G-RIVER-DESCENT 3 430 rising trunk steps, all inside filled hollows, the worst 4 987 m.
  The discharge row inherits it (plan question 3: A is built). The cure proposed: the shipped receivers follow the
  budget's water (Fill–Spill–Merge's own flow directions: steepest descent on dry ground, across a lake's flat to its
  outlet, over a full lake's spill to the sibling's gate). A change of the receiver byte's meaning: the owner's word
  before step 3, because R2 seeds its rivers from the receivers.

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

**Example.** A hollow lies at the foot of the belt's wet range, and the range drains into it by the solve's own
receivers. The budget counts only the rain under the hollow's brim, so the hollow is a dry pan. A pond stands
instead where a hollow's own floor is wet; from 150 km the pilot sees it as a square, and past ~450 km not at all.

---

## 8. THE PLAN (agreed in general by the owner, ruling B6, 2026-09-24)

Each step ends with the census numbers, the three stands' pictures, and the REFERENCE STAND's picture beside the
reference (§9). State legend: ⬜ not started · 🟨 in progress · ✅ done (with the commit).

★ **THE ORDER SINCE RULING B8 (2026-09-25): 1 → 2 → 2b → 4 → 3 → 5 → 6 → 7.** The step numbers stay the names every
other file uses ("step 3" is R2, "step 4" the shoreline); only the order changed. Step 2's pictures showed every small
lake as an 8 km square, so the edge law (step 4) now comes BEFORE R2 on today's ground, and the rivers follow the
water (step 2b) before either, because R2 seeds its rivers from them. R2 starts with slice 0 (the bench and a
hillshade for the owner's look) before any bake service.

| # | step | cures | state |
|---|---|---|---|
| 1 | **Measure first, no world change.** The "before" pictures (the three stands + the reference stand). New census lines that can go red: G-HACK and G-BREACH on the SHIPPED ground at rung 7 over ~200 km (the dunes number); a teeth number (the drawn shore's distance to the ground's waterline, the wall height, the shoreline's direction); G-HYPSO and G-RIVER on the final field; the lakes lost at 9 → 10; the inland nodes under the sea's level; the watershed supply against the budget's. The skyline numbers on the reference stand (11_reference_trace's rule). One job at a time: only while the `perf` tree is idle (2026-09-24: it is testing; the instruments are written meanwhile, no build in this tree). | makes each complaint a number | ✅ NUMBERS (generator 19, `step1_census_gen19.txt`) and the "before" PICTURES (`pictures/*_before.png`, §8b), 2026-09-24, committed |
| 2 | **Repair the lake budget.** Feed each hollow from its whole watershed; pass a full hollow's overflow on; re-accumulate the discharge after the final route; keep a small lake's level at pyramid level 1 (no lake lost at 9 → 10). New GENERATOR_VERSION and pins. Goes BEFORE R2, because R2 seeds its rivers from the discharge row. ★ AND (ruling B7) the inland pocket is a closed basin and the sea stands over the connected ocean. | water positions (part), W6's rung seam | ✅ BUILT and MEASURED 2026-09-25 (generator 20, artifact 8, `step2_census_gen20.txt`), pins re-recorded; the PICTURES taken (`pictures/*_step2.png`, §8b: the lakes are there, each one an 8 km square); committed on branch `game` 2026-09-25 (ruling B8 item 4) |
| 2b | **The rivers follow the water** (ruling B8 item 2; plan question 3 choice B; amends B7 item 2). The shipped receivers and the discharge row follow the budget's own water: a full hollow's river runs over its spill into the sibling (Fill–Spill–Merge's flow directions), and a closed body that never fills passes nothing on. New GENERATOR_VERSION and pins. | G-RIVER-DESCENT (3 430 rising steps at generator 20); the row's 1 607 km³/yr out of held hollows | ⬜ NEXT |
| 3 | **R2: one eroded level at 1 024 m in the artifact**, with the two operators the bench lacked (the height retargeting that holds the macro heights, the multi-scale breach); then R4's second half: the noise only BELOW the solved level. Cost (survey): ~17 core-hours once for the home planet, 512 KB a tile against 40 KB, an artifact version, a lattice over `MACRO_EDGE_CEILING` 2 048. Calibration: Earth, through Hack's law (T9). | the dunes at 100–150 km; valleys that join | 🟨 DESIGNED (`r2_design.md`, 2026-09-24) with 23 skeptic findings OPEN (§16 of it: 6 HIGH); slice 0, the bench with the owner's look, comes before any service (ruling B8 item 3); AFTER step 4 since B8 |
| 4 | **One shoreline from one field, on today's ground** (ruling B8 item 1 moved it BEFORE R2; it was "on the R2 ground"). The sea's shore and the lake's edge from where the ground crosses the water's level, not from a node square; the lake's level held by its dry neighbours (W6's owed law); plan question 5 (the dry-land hold) and the far-swap lake losses decided here. The law does not depend on which ground it reads: R2 later gives it the 1 km bays and headlands (T3), and the pictures are judged again then; until R2 a shore may look dune-shaped (stated and accepted, B8). | the teeth (sea and lake); step 2's square lakes | ⬜ after 2b |
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
   the sea that the budget holds) — ★ ANSWERED B (ruling B8 item 2), built as plan step 2b; Q4 the passes cut with the rain (unchanged); Q5 the dry-land hold (built: near the
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
