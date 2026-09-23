# Owner decisions, 2026-09-21 — the roughness onto the solve, the 8d order, and water that finds its level

Written during the 8d discussion, after the far-view ship and the coast fix landed and the owner flew the
day-side belt. Newest ruling file; it wins over every earlier design document. It amends ruling T2 of
`owner_decisions_2026-09-16_terrain.md` (the currents) and answers the open ruling L20 (a player's dam).

## W1. THE ROUGHNESS FACTOR READS THE SOLVE, THEN 8d (owner: *"Agreed"*)

The owner looked at the belt from 16 km and said the land is "almost flat surface with hills". The order
agreed: (1) the fine relief's per-column roughness factor reads the SOLVED field's own macro slope, one day,
judged in pictures over the belt stand; (2) then the 8d design; (3) the discussion of 8d's asks before its
code. Item (1) is BUILT the same day (the share `min(1, |∇Z| / slope_ref)`, the reference the body's own
first fine octave's slope, the factor the larger of the noise factor and the share; `GENERATOR_VERSION` 8;
the belt's factor 0.53 → 0.88). The owner's second look: ridges and basins now, and the cap-rock bench's
stripes on every slope — which is 8d's first piece (the per-bed hardness), pulled forward.

## W2. A PLAYER'S DAM AND A PLAYER'S CRATER — THE ROUTE NEVER CHANGES, THE WATER FINDS ITS LEVEL LOCALLY (owner: *"Ok, agree with that one"*)

Answers the open ruling L20. Two rules:

1. **The route never changes.** Where the seed's water runs comes from the solve and never moves in play
   (ruling T3: no erosion in time, no re-solve on a player's action). A wall across a river does not move
   the river; the river is an endless source, as a flow that never runs out. A crater far from any water
   stays dry.
2. **Water finds its level as LIVE STATE, locally.** When a player's edit opens a cell that stands below
   the level of a CONNECTED river or lake, the owning shard runs a BOUNDED flood fill from that water, cell
   by cell, at that one level, inside the edit's own box. The wet cells are rows of the block store, like
   placed blocks: live state, shipped as a diff, folded into the far view's pyramid. This is communicating
   vessels, not fluid dynamics: no wave, no current, no pressure, no draining. It is deterministic,
   server-authoritative and bounded by the edit. It belongs to SLICE 9 (the block store), where water is a
   substance with the LIQUIDITY parameter the owner asked for on 2026-09-07 (V4 15.4); its cost is one
   flood per edit, measured there. 8d builds the static half (the rivers, the lakes, the shores) now.

**Example.** A player digs a crater beside the great river. The crater's floor stands under the river's
level, so the shard floods it to that level and every client draws water in the crater; the river itself
is not lowered. A player who digs a canal from the river to a dry basin fills the basin to the river's level.

## W3. WAVES AND CURRENT ARE REQUIRED; PRESSURE AND DRAINING MAYBE, LATER (owner: *"we will need to work on the waves and current. Maybe also on the pressure and draining, but later"*)

Amends ruling T2 ("currents: NOT planned"): the CURRENT is now OWED, priced before it is built, beside the
waves. The order stays after 8d: **8o the ocean** holds the waves (as look: a shader on the delivered tick;
as physics: a closed-form height field from the seed, the address and the tick, computed by both hosts,
never part of the static shape) and now the CURRENT (a flow field a hull feels as a force through the
movement contract, and a walker in a river feels the same way — its source is the river's stored direction
and the width law's speed for rivers; the sea's current needs the ocean heat transport the climate model
lacks, so it is priced as its own model before it is promised). PRESSURE and DRAINING (a filled crater that
empties through a dug hole, water under a pressure head) are LATER and MAYBE: they are a cell simulation of
the Dwarf Fortress kind, and the owner will decide with a price.

**Example.** A hull drifts down the great river without its engine: the shard reads the row's direction and
the width law's speed and pushes the hull along; every client draws the same ripples moving the same way at
the same tick. On the sea the same hull rides the swell the almanac's wind raised.

## W4. THE 8d ASKS, ANSWERED (owner: *"Agree with all: 1, 2, 3, 4, 5, 7"*)

Every recommendation of the register in `docs/investigation/2026-09-08/landforms/slice_8d_design.md` §5 is
now a RULING:

| # | Ruling |
|---|---|
| 1 | The river's width and depth come from the hydraulic geometry of Leopold and Maddock, calibrated on EARTH, with gravity from bed shear (T9: a published law, a named body). |
| 2 | A valley narrower than a solve node whose floor converges carries a SMALL STREAM drawn from a table at the fine rungs — no discharge, nothing stored — so visible valleys do not run dry. |
| 3 | A lake's surface is the sea's own mechanism: the stored water word, the one sheet, the one coast band and beach. No second water surface. |
| 4 | ★ SL6 APPROVED: the PROVINCE WORD rides the artifact's node row (nine bytes → ten, about 9 MB on the home planet, one artifact version bump). A client cannot recompute it (the solve's crust and belt fields are transient), and it adds no lane and no payload kind. |
| 5 | The authored override's row: a coarse cell address at a stated rung, a height change in whole metres, a falloff radius, an optional substance; written by the OWNER'S authoring tool through the realm that owns the ground, never by a player (a player's change is a cell edit with its own path). 8d owns the read path and the pyramid fold; slice 9 the store; slice 10 the lane. |
| 6 | W2 above. |
| 7 | The mesh extractor's crease (rounded surface nets, or sharp dual contouring) is DECIDED ON 8d's OWN CLIFF AND MESA PICTURES, both drawn on the same stands, while the change is free (before the freeze at 14). |

The order of work inside 8d stands as the design's §4, with the per-bed hardness (step 1) already pulled
forward by W1: 1 the beds' hardness → the rock map word and the substance at a fixed radius → the river lines
and the width law → the carve and the floodplain → lakes, shores, outlets → the override's read path. One
picture and one measured gate per step; a red gate goes to no picture.

## W5. 8d STEP 1 LANDED — THE BEDS' HARDNESS — AND ONE NUMBER IS OWED BY THE OWNER

BUILT (2026-09-21 evening): each bed top has a hardness from a seed identity draw (the recipe's own corner
hash of the bench seed and the bed index, T9-lawful as an identity choice); the bench pulls a column toward
the nearer bed top by that top's hardness, scaled from nothing at the threshold to the whole body strength
at the hardest bed; the body strength is now a ceiling, so no bound and no address moved. `GENERATOR_VERSION`
9; the identity re-pinned; the golden tables re-recorded; the card equal to the CPU on 5 979 of 5 979 boxes.
MEASURED on the home planet: over 5 000 m of raw height the bench drew 37 treads before and 11 after, the
tread length down 74 %; of 600 columns every one was pulled before, 263 are pulled now. The 8 km picture over
the belt shows smooth hillsides with no contour stripe.

★ **THE CAP SHARE IS ASSUMED, NOT DERIVED — an ask.** The reference's bands are 1–20 m thick and its cliffs
20–200 m tall, but this body's bed spacing (twice its own sediment band) is 138 m, thirty times the
reference's band, so the reference states no share of caps for a stack this coarse. The build takes the
smallest-information law: half the bed tops are caps (a hardness over one half). OWED: the owner's word on
the share — half; one cap in three; one in five; or the bed spacing itself brought down to the reference's
bands (which multiplies the treads a column crosses and is the bigger change). Recommended: judge the 8 km
picture first, then rule.

## W6. THE SEA DECIDES THE SHORE — built on the owner's second report, one number owed

The owner, flying the coast after 8d steps 1 and 2: *"Again, after your changes when I'm flying, the shores are
changing all the time."* MEASURED (`vd-bins/examples/shore_step`): the shoreline is where the ground crosses the
sea, the FINE octaves decided that crossing, and a coarser rung keeps fewer of them, so at every ring swap the
crossing moved sideways by the dropped octaves' height over the coast's slope — a median of 234 m at the 128 m
rung and 848 m at the 256 m rung, on lines where the ground itself moved by under a cell. W1 (the roughness onto
the solve) made it worse by design. And a land node's row carries no water word, so the land node's own columns —
which the shore runs through — held no water at all: the sheet stopped at the midline between two nodes.

BUILT (2026-09-21, late evening; `GENERATOR_VERSION` 11): **the fine octaves may not carry a column across its
water.** After the bench a column's surface is held on its GROUND's side of its water — the ground being the
solved field plus the coarse octaves, which every rung shares — by at least a QUARTER of the ground's own height
over or under the water (`vd_recipe::height::shore`, one source, the card compiles it). A dry row's column reads
the body's sea, so the ground decides the side on both sides of a coast. MEASURED after: the crossing's step is
0 m at every rung pair on all 33 crossing lines; `gpu-drift` 5 979 of 5 979; the golden tables and the artifact
digest unchanged. What the picture gains: a coastal plain where the octaves would have cut under the sea, and a
bar under the water where they would have raised an island the next rung loses.

★ **THE QUARTER IS A STATED CHOICE — an ask.** The physical mechanism (waves plane the coast) states a BAND (the
storm wave, 11.8 m on the home planet), and a band cannot hold the crossing where a fine octave is taller than it.
The share of its own height a column keeps against the octaves has no published law; a quarter is the smallest
power of two that leaves a valley near the coast a floor of its own. OWED: the owner's word — a quarter, a half,
an eighth — after a look at the coast from the hull. A lake's shore is owed the same law at 8d step 5.

**Example.** Along the belt's coast the solved ground rises one metre in fifty. Two kilometres inland it stands
forty metres over the sea; the fine octaves may dig thirty of those, so the valley's floor stands ten metres over
the sea at every rung a descending hull draws, and the shoreline the pilot sees from 5 km is the one at 500 m.

## W7. THE LAKES — TRIED, AND THE DROWNING FOUND UNDER THEM (2026-09-22; owner: *"Yes, let's try it out"*)

MEASURED: 16 % of the land was lake; the pits were the craters (137 173 pits under 144 967 stamped craters);
the flood ran every tenth pass and the cut was placed nowhere. BUILT AND KEPT: the flood every pass and the
deposit (the cut laid in the first hollow downstream, up to its spill; the rebound reads the net). The lakes
now drain and fill: 137 173 → 7 055 pits over the passes.

FOUND UNDER IT: the old solve eroded for a tenth of the age by a defect (the stale flood skipped every cut
node), and the 10 % land accepted on ask 8 was that accident. With the rivers running the whole age every
continent grades to the sea and the re-solved sea drowns it: 1.4 % land at any age, 1 % with hard rock, 0.1 %
with a cratonic uplift (the shelf rose under the sea and the basins lost their volume). The sea at +4 455 m —
Earth's dry step, the coincidence C5 could not explain — IS the continental freeboard: the inventory fills the
basins to the continents' brim.

★ **THE ASK (T9): the law that keeps the continents up.** Recommended order: (1) the ocean basins' depth from
the oceanic crust's own isostasy (the L2 densities; Earth 3.8 km mean depth), (2) the inventory checked
against that volume so the sea stands at the shelf edge and not over the shield (Earth: 29 % land on 40 %
continental crust), (3) then the plains' height as a slow cratonic uplift over a rock-dependent erodibility.
Each a published law with Earth as its body; each measured by `lake_census` (land share, lakes' share, pits).
Until the ruling lands the tree's solve drowns the home planet; the owner's window runs the previous build.

**Example.** A pilot over the belt today sees a continent of hills with a lake in every hollow. After the
breach and the deposit the hollows are plains with a river through each; after the basins' law the same
continent stands 400 m over a sea that stops at its shelf, with two long rift lakes and none in the hills.

## W8. THE THREE STEPS, BUILT (2026-09-22; owner: *"Agree with all three steps, please proceed in that order"*)

**Step 1, the floor's depth** — already the law: the initial land computes both crusts by Airy isostasy
(the continent 5 303 m, the floor 848 m over the compensation depth at Earth's numbers). No change.

**Step 2, the sea at the shelf edge** — MEASURED first: continental crust 40.0 % of the area, the platform
7 454 m over the floor, the basins under the platform 2.36 × 10⁹ km³ against an inventory of 2.74 × 10⁹ km³
(1.16 ×; this planet has twice Earth's water by its formation ratio), the sea 861 m over the platform before
one river ran. BUILT: **the freeboard law** (Wise 1974) — the continental thickness is SOLVED, not Earth's
constant: the factor on the Earth-scaled thickness at which the sea the inventory fills stands at the
continental crust's shelf quantile, `1 − 0.29 / 0.40` (Earth: 29 % land on 40 % crust; the median put half
the crust under a few metres of water and the coast came out a checkerboard). Factor range 0.5–3, clamped
and stated at its ends (a water world, a desert world). And **the running sea**: re-solved over the field at
every climate step, so the base level the rivers cut to follows the inventory over the eroding continents
(before, the sea moved once at the end and the graded shelf came up as a field of pits).

**Step 3, the plains** — BUILT: **the erodibility by rock** — the stream-power constant times the inverse
square of the province's four rocks' mean tensile strength over shale's (Sklar & Dietrich 2001; a table of
published typical strengths per stratum): the shield and the belt cut at 23/256 of the shelf's 92/256, the
rift's basalt at 20/256. TRIED AND REMOVED: a secular cratonic uplift (5 and 10 m/Myr, Braun 2010) — it
raised the shelf under the sea without bound and drowned the globe; the plains' height in this model is the
TRANSIENT decay of the freeboard, which the rock's low erodibility and the rebound stretch over the age.
FOUND UNDER STEP 3: **the envelope scaled the whole globe** — one belt at 11.9 km under the 8.3 km cap
scaled every height by 0.69 and the basins lost a third of their depth; NOW a per-node clamp (a mountain
the rock cannot hold collapses where it stands). That scale was the drowning's other half in every run.

**MEASURED, the home planet, `GENERATOR_VERSION` 12:** land **22.0 %** (19.8 dry + 2.1 lake) against 8.6 %
before W7 and 1.4 % after the lakes' cure; sea 78.0 % (Earth 71 %); the pits 170 k → 14 k over the passes;
the lakes' share of the land **9.7 %** (Earth 3.7 %, Canada's shield 9 %). The remaining lakes are GLACIAL:
the ice line stands over 2.4 million nodes, more than all the land, so the climate glaciates every continent
and the ice cuts a hollow into every valley — the climate's word, which is 8e's discussion, not the
lakes'. The solve: 114 s (85 before W7). The sea stands +4 114 m (was 4 455); the ocean share 78.2 %; the
identity and the artifact digest re-pinned; the chunk golden tables unchanged (the recipe did not move).

OWED: the ice line's extent (8e); a delta where the sediment reaches the sea (today it sinks); 400 passes
for the cratons' transient at the rock's slower cut (MEASURED 167 s, no land gained while the envelope
scaled — re-measure now); the owner's look at the belt and the coast under the new sea.


## W10. THE COAST MASK — EVERY HOST READS THE WATER'S SIDE FROM THE ROW, AT EVERY RUNG (2026-09-22; owner: *"Perfect"*)

The owner, flying down from 1 400 km after W6: *"during flight the shores changes again all the time"*.
MEASURED (`vd-bins/examples/shore_step`, 400 lines of 600 km across the belt's coast): the FINE rungs step
0 m at every swap — W6 holds — and the coast steps a MEDIAN of 11 536 m (p90 112 km) at the swap from the
rows to level 1 (rung 9 → 10), 20 823 m at level 1 → 2 (14 → 15) and 19 982 m at level 2 → 3 (15 → 16). THE
CAUSE: a pyramid level's height is the MEAN of its children, and W6 decides the side by that height, so the
ground's crossing of the sea moves by about a level node and the morph carries the shoreline over that
distance as the ring passes.

**THE LAW, as the owner agreed it.**

1. **Every host reads the water's SIDE from the row's own word, at every rung.** The word is one bit per
   fine macro node — set where the node stands at or under the sea — stored with the artifact, folded into
   its digest, and shipped once with its head.
2. **No host derives a side from a mean.** A level's height may be a mean; a side may not. A side is a bit,
   and a bit does not fold.
3. **A coarse rung's shape is a LOOK of the same field, never a second field.** The far view draws fewer
   octaves and coarser heights of ONE ground; it may not draw another shoreline.
4. **Collision reads rung 0, where the mask bit and the row are one word.** A pilot's boots and the tile
   under them read the same byte, so the ground she walks on is the ground she saw.

**Example.** A hull descends on the belt's coast from 1 400 km. At rung 15 one of its cells covers half a
bay: the level's mean stands under the sea, and the headland inside the bay stands dry because the
headland's own row says LAND. At rung 5 the pilot draws the same headland from the tiles under her, and at
rung 0 her boots stand on it. One shoreline, three rungs, one word.

**BUILT the same day** (`GENERATOR_VERSION` 13, `ARTIFACT_VERSION` 6, `PROTO_MINOR` 34 with the floor at 34):
`Artifact::coast` (1.1 MB on the home planet, one part in eighty of the rows), `ZField::sea_side`,
`artifact::sample_side` on the FINE lattice at every rung, `vd_recipe::height::shore` taking the side as a
fourth word (`SIDE_UNKNOWN` / `SIDE_LAND` / `SIDE_SEA`), the store's coast part rows, the wire's
`BulkMsg::ArtifactCoast`, the gateway's cache and the client's book. The card holds no artifact and reads
`SIDE_UNKNOWN` — its bytes did not move.

**MEASURED AFTER** (`shore_step`, the same 400 lines): the swap from the rows to level 1 falls from a
median of 11 536 m to **7 m**, level 1 → 2 from 20 823 m to **0 m**, level 2 → 3 from 19 982 m to **0 m**
— every median under one fine node (8 192 m). `gpu-drift` 5 979 of 5 979; the seed-only chunk tables
stand; the identity's measured half is unchanged.

**OWED:** the owner's look at the coast from the hull under the new mask; the QUARTER of W6 is still a
stated choice; a lake's shore still keeps today's rule until 8d step 5.

## W11. THE FILL IS A SCRATCH SURFACE — A LAKE IS WATER THAT STANDS (2026-09-22; recommendation 1 of the lakes report, ordered by the owner)

Answers `docs/investigation/2026-09-22/lakes_and_landscape_models.md` §6.1. It does not touch W4 item 3
(a lake's surface is still the sea's own mechanism); it states what a LAKE IS.

**THE LAW, in three lines.**

1. **The routing flood is a SCRATCH SURFACE.** It raises every closed hollow to its spill so that water
   can be ROUTED across it, and it decides the receivers, the flats, the topological order and the
   sweep's base level. It decides NOTHING ELSE, and no host may read it as water. Every published model
   already does this — Barnes calls the fill "an important preconditioning step"; Landlab writes it "to a
   scratch surface, never to `topographic__elevation`"; Cordonnier calls filling and carving "metaphors …
   without altering elevation values". We kept it and drew it, and that was the whole defect.
2. **A lake is water that STANDS, from a budget with a published balance.** The depression hierarchy of
   Fill–Spill–Merge (Barnes, Callaghan & Wickert 2021, Earth Surf. Dynam. 9, 105–121) over the final
   field; the closed-lake balance of Langbein (1961, USGS Professional Paper 412) over the water the
   climate already computes. Every hollow ends **DRY, PARTIAL or SPILLING — decided by the water supply,
   never by the shape of the hole.**
3. **A hollow the rain cannot fill is a DRY BASIN.** About a fifth of Earth's land drains internally and
   most of those basins are playas, salt pans and desert sinks. A closed hollow is not evidence of a lake.

**THE WATER, node by node.** A node states its rain `P` and its potential evaporation `PET` (Earth's land
mean scaled by the Tetens water capacity — the aridity byte's own law, now named). Its RUNOFF is the
Turc–Pike partition of the Budyko framework, `E = P/√(1 + (P/PET)²)`, `Q = P − E` (Turc 1954; Pike 1964;
Budyko 1974; calibration body: Earth's catchments). A hollow's SUPPLY is `Σ Q·area` over its own nodes; a
node UNDER the water costs `Q + PET − P`, the runoff it stops yielding plus what the open surface hands
the air. The water rises until the two balance, capped at the spill; what is over the cap leaves through
the outlet, which is the merge.

★ **A DEFECT FOUND BY MEASUREMENT, and recorded because it nearly shipped.** The first build took the
runoff as `max(0, P − PET)`. Earth's land mean rain is about 750 mm/yr against a potential evaporation
over 1 000 mm/yr, so that difference is negative over most land and **Earth's own rivers would carry
nothing**. MEASURED on the home planet: 74 657 of 74 682 hollows ran dry and **31 nodes of 8.87 million
held water**. The Budyko partition is the published cure: a desert node still yields a trickle, a wet one
yields most of its rain.

**MEASURED, the home planet, `GENERATOR_VERSION` 14** (`lake_census 0.617270 -0.437286 -0.654033 60`,
the same stand as W8; the FIELD did not move — the sea stands at 4 114 m and the ocean share at 7 822 of
10 000, both unchanged, and the seed-only chunk tables stand):

| Reading | Before | After | Earth |
|---|---|---|---|
| Land (nodes over the sea) | 1 948 290 (21.96 %) | **the same field, 21.96 %** | 29 % |
| Nodes drawn as fresh water | **188 848** (the flood's own raised count) | **3 169** | — |
| ★ The lakes' share of the LAND | **9.69 %** | **0.16 %** | 3.7 % of the non-glaciated land, of which most is glacial; **0.2–0.8 % in non-glaciated interiors** |
| ★ Lake patches over one node (67 km²) | **51 209** | **2 263** | ≈ 2 000 |
| The largest patch | — | 20 nodes, 1 342 km² | no upper limit |
| The patches by size (nodes) | median 1 | 1: 1 737 · 2–3: 442 · 4–7: 76 · 8–15: 7 · 16–31: 1 | a ten-fold fall per decade of area |
| The hierarchy | — | 96 493 depressions, 74 080 leaves, 74 080 water bodies | — |
| ★ The bodies' three ends | — | **DRY 71 817 (96.9 %) · PARTIAL 2 261 (3.1 %) · SPILLING 2** | — |
| The standing water | — | 1.921 × 10⁵ km², 3 288 km³ | 181 900 km³ |
| The solve | 114 s (W8) | **130.3 s**, of which the hierarchy and the budget are **0.2 s**, measured on their own | — |

**THE BELIEVABILITY GATES (ruling B3 of `owner_decisions_2026-09-22_believability.md`, the same day):**
**G-LAKE-SHARE GREEN** — 0.16 % of the land against a ceiling of 4 %. **G-LAKE-COUNT GREEN on the count**
— 2 263 patches over one node, the order of 10³ the gate asks for, against Earth's ≈ 2 000; the power-law
fit of the sizes is UNMEASURED. **G-BASIN GREEN** — every depression ends dry, partial or spilling by its
own water supply, and the inventory balances. The other seven gates belong to the later steps of B2.

★ **THE COST IS 0.2 s ON 8.87 MILLION NODES**, timed by itself in `lake_census`. Fill–Spill–Merge's authors
ran 933 million cells in 46.47 s; ours is one hundredth of that grid and the reading agrees.

★ **THE LAKE DOUBLE COUNT IS CLOSED** (`DEFERRED.md` has carried it since C5). The lakes hold 3 288 km³
against an inventory of 2.736 × 10⁹ km³ — one part in 830 000. Re-solving the sea over the inventory less
the lakes leaves the level exactly where it stood, **4 114 m**: the lakes' whole volume is far under the
sea solve's own sixteenth of a metre over the ocean's area. The solve takes the subtraction anyway and
says which of the two happened, so a wetter body gets the right sea.

**WHY ONLY TWO HOLLOWS SPILL, and it is the climate's word, not the lakes'.** A hollow spills only where
its own rain beats its own potential evaporation over the WHOLE basin — a brim-full lake has no dry
catchment left to feed it. This climate's rain and its evaporation carry the same Clausius–Clapeyron
factor with calibrations of 990 and 1 000 mm/yr, and the belt factor only ever REDUCES the rain, so
`P/PET ≤ 0.99` nearly everywhere and even the rising branch is neutral. On Earth the tropics and the
mid-latitude west coasts stand well over one. That is 8e's discussion (the climate), not this one.

**WHAT IS NOT IN IT.** The GLACIAL lakes — the ice line, the summer temperature and the ice inside the
pass loop — are recommendation 2 of the same report and are not built. Earth's lake area is mostly
glacial, so 0.16 % against Earth's 3.7 % is the number a planet with no glacial-lake mechanism should
show; the non-glaciated interior figure (0.2–0.8 %) is the one it stands beside. The carve of a spill
(Cordonnier 2019) and the square patches at the coarse rungs (the report's §6.5, a lake's own side word)
are also not built.

**Example.** A pilot follows a river off the belt's range into a hollow. In the wet north the hollow
stands full to its sill and the river runs out of the far side; in the dry interior the same shape of
hollow holds a salt lake half-way up its walls; one valley further on the hollow is a white pan she can
land on. One law decided all three, and it read the rain, never the ground.

---

## W12 — THE SUMMER ICE LINE AND THE CRATER RECORD OF A WET SURFACE (2026-09-22, ruling B2 step 2)

Built after W11, in the same slice. Two laws were asked for; two more had to land under them, because
each of the two read a temperature that was wrong before it was read.

### W12.1 The two laws under the two laws

**THE TEMPERATURE'S DATUM IS THE BODY'S OWN MEAN SURFACE.** The charter states a mean SURFACE
temperature, and a surface temperature stands at the surface: at the sea on a body with one, at the
area-weighted mean of the ground on a body with none. Before this the lapse cooled from the LADDER
RADIUS, a geometric datum the home planet's sea stands 3 566 m above, so the sea surface read −11 °C,
every land node stood over the freezing line, and the land's rain was a sixth of Earth's. Calibration
body: EARTH, 288 K at sea level. `vd_terrain::climate::datum_z`.

**THE AIR OVER WATER STANDS AT THE WATER'S OWN SURFACE.** A node four kilometres down on the abyssal
plain carries the sea's air, not air four kilometres' worth of lapse warmer. MEASURED before the line:
the sea floor read 344 K, asked for 16 160 mm of evaporation a year and was given 6 708 mm of rain;
after it, 288 K, 1 000 mm and 575 mm. The orographic lift reads the same surface, so a coastal node no
longer lifts air out of the abyss.

### W12.2 The summer ice line, and the ice inside the passes

**THE SEASON.** The ablation season's warmth over the annual mean is the first Legendre term of the
daily-mean insolation over the seasonal energy balance:

- `s₁(ε) = 2·sin ε` — North 1975's expansion, the same one `insolation_s2` reads its second term from;
  North & Coakley 1979 state `s₁ = −0.796` at Earth's 23.44° tilt, and `2·sin 23.44° = 0.796`
  (★ UNVERIFIED as a source opened in this session; the Earth value is the check).
- `ΔT(φ) = S̄(1 − α)·s₁·|sin φ| / |B + 2D + iCω|` — the one-mode balance at the `n = 1` mode's own
  eigenvalue `n(n+1) = 2`, with the surface's seasonal heat store `C` turning once a body-year:
  the wet share times a 50 m ocean mixed layer, plus the dry share times the ground's ANNUAL SKIN
  DEPTH `√(2κ/ω)`. Calibration bodies: Earth's ocean mixed layer and rock's own `κ` and `ρc`.
- ★ ITS STATED LIMIT: ONE well-mixed column stands for the ocean AND the land, so a continental
  interior really runs hotter in its own summer than the law says. The line it gives is therefore
  LOW, never high, and the census's G-ICE line measures the consequence.

**THE LINE.** `ela_z` no longer reads the mean-annual freezing line plus a drawn 1 200 m dryness
offset. It stands where the ABLATION-SEASON temperature falls to the one Ohmura, Kasser & Funk 1992
(J. Glaciol. 38(130), 397–411) name for the node's OWN rain — their curve `P = 645 + 296·T + 9·T²`
(mm w.e./yr against °C, 70 glaciers, standard error 200 mm), inverted for `T`. The paper's own anchor
holds: `P(1 °C) = 950 mm`; `350 mm` gives −1.0 °C, `2 000 mm` gives +4.1 °C. A dry glacier needs a
colder summer than a wet one, which is what the deleted offset was groping for.

**THE ICE.** `MacroSolve::ice` is Egholm et al. 2009's mass balance (Nature 460, 884–887) and NOTHING
IS DRAWN. Accumulate `0.1·|Ts|` a year a kelvin over the line, ablate `0.15·Ts` under it, lose
`0.05·T` at a bed the year's own mean keeps over freezing; carry that balance down the receiver tree
the water already uses; where the sum runs out the glacier ENDS. The thickness is the LESSER of two
perfect-plastic readings of one yield stress — `τ/(ρgS)` over the slope (Nye 1952) and `√(2τd/ρg)`
over the distance to the margin (the plastic ice-sheet profile). The speed is the flux over the
cross-section, `u = q/H`, which is greatest where the catchment above the line is greatest — AT THE
LINE — so the overdeepening falls at the equilibrium line and below every confluence with nobody
placing it (MacGregor et al. 2000). The bed erodes at `kₑ·u`, `kₑ = 10⁻⁴`, `l = 1`, at most by its own
thickness, and only where `T_bed = T_year + G·H/k_ice` stands over the pressure-melting point: COLD
ICE DOES NOT ERODE (Earth's continental geothermal flux 0.065 W/m²; ice's conductivity 2.1 W/(m·K);
the melting gradient 8.7 × 10⁻⁴ K a metre, Paterson 1994).

**THE ORDER.** The ice and the talus ran AFTER the last sweep and the last deposit, so nothing could
ever drain or fill what they cut and the final flood found every hollow they left. They now stand
INSIDE the pass loop, between the routing and the sweep, so this pass's rivers answer them at once.
The passes' count is unchanged (40); the talus keeps its own total of 8, spread over the loop; the
ice cuts the glacial epoch's fortieth each pass and states its mask once more over the final field.
THE COST: the solve went from 130 s to **103 s** — the ice is three O(n) walks of a tree the routing
already built, and the field it leaves is cheaper to route than the one the old order left.

### W12.3 The crater record on a wet body

The production function stays Neukum, Ivanov & Hartmann 2001, capped at the lunar saturation. The AGE
IT INTEGRATES OVER becomes the surface's own CRATER RETENTION AGE — the lesser of the body's age, a
PLATE CLOCK and a DENUDATION CLOCK:

- **The plate clock**, on OCEANIC crust only: the distance to the nearest boundary over the plate's own
  speed, which is the drift share the seed drew times EARTH'S MEAN PLATE SPEED, 50 mm/yr. A half-ocean
  of 10 000 km at 50 mm/yr is Earth's own 200 Myr ceiling. Continental crust is not consumed and a
  one-plate body has no boundary to reach, so both answer "no clock".
- **The denudation clock**: a crater is gone once the ground has fallen by the crater's OWN depth.
  The rate is Portenga & Bierman 2011's measured outcrop lowering, 12 m/Myr (GSA Today 21(8), 4–10;
  ★ UNVERIFIED as a source opened in this session), scaled by the province's erodibility against the
  CRYSTALLINE BASEMENT's — the rock the cosmogenic method is measured on — and by the node's own rain
  against Earth's land mean of 750 mm/yr. ★ THE RAIN AND NOT THE RUNOFF: the runoff is the Turc–Pike
  remainder after the air has taken what it can, so it collapses toward zero over any semi-arid
  ground — MEASURED on the home planet, 373 mm of rain left 74 mm of runoff and the p10 node kept its
  craters for four billion years on ground that really wears. Denudation is chemical as well as
  mechanical, and the water that does the chemistry is the water that FALLS.

Every candidate the production function draws keeps its face, its cell and its size on the body's own
crater stream; the RETENTION reads a SECOND stream (index 1 of the same salt), so a surface that
renews nothing draws exactly what it always drew and its record is byte for byte the one it had. Each
kept crater also carries the age of its own impact, found by inverting the chronology over the
retention age.

### W12.4 The numbers, before and after

| reading | before (W11) | after (W12) | the body it stands beside |
|---|---|---|---|
| **G-ICE** — nodes under ice | over 2 400 000, MORE than all the land | **78 554 = 4.06 % of the land** | Earth about 10 % today, 30 % at a glacial maximum |
| **G-BUZZSAW** — the highest peak over its own snowline | not measured | **1 322 m; 0 land nodes over 1 500 m** | Egholm 2009: about 1 500 m, everywhere on Earth |
| **G-CRATER** — craters kept | **144 967** | **1 610** of 144 967 drawn | Earth 190 structures — within an order of magnitude |
| craters wider than 20 km | 137 173-odd | 1 419 | Earth 43 — high, and the cause is the 16.4 km floor |
| the median crater | — | 33.2 km | Earth 8 km — the macro node is 8 192 m |
| the age histogram | every crater the body's own age | 8 % younger than 200 Ma, median 3 976 Ma | Earth 45 % younger than 200 Ma (4.4 % of its history) |
| the land's share | 21.96 % | 21.78 % | Earth 29 % |
| lakes, share of the land | 0.16 % | 0.05 % | Earth 3.7 % with its shields, 0.2–0.8 % without |
| lake patches | 2 263 | 362 | HydroLAKES: order 10³ over one of our nodes |
| basins that end dry | 96.9 % | 97.9 % | Earth: 18–25 % of the land drains internally |
| the sea | 4 114 m | 3 566 m | the ocean share 78.22 % → 78.60 % |
| the solve | 130 s | **103 s** | — |

**WHAT IS STILL WRONG, AND WHY.** G-ICE reads 4.06 % against Earth's 10 %, and the cause is the
charter, not the law: the home planet's obliquity is 55°, which makes its annual latitude contrast
almost nothing (`s₂ = 0.006`) and its polar SUMMER the hottest place on the planet, so it has no polar
ice caps at all. That is what a high-tilt world looks like, and the OWNER'S WORD IS OWED on whether
this world wants that tilt. The lakes fell rather than rose, because the ice's own hollows are now
drained or filled by the passes that follow them; the glacial lakes the report asks for need the
overdeepening to survive the deposit, which is the river step (B2 step 4) and the lake side word
(step 5).

**Example.** A pilot flies the belt at the equator. The season there is nothing, so the snowline sits
where the year's own mean puts it, 2 500 m over the sea, and she sees no ice under it. She flies north
and the line RISES with the summer instead of falling, because this planet's tilt gives its north a
hot summer. On the one range that beats the line she finds a trough cut deepest just where the snow
line crosses it, a rock bar below, and no peak standing more than fourteen hundred metres over the
snow. On the plain between she finds one crater in a continent, and it is worth flying to.

---

## W13. THE GRID'S SCRATCHES: MEASURED, THE FLAT CURED, D8 PUT TO THE OWNER (ruling B2 step 3, gate G-GRID)

Written 2026-09-22 after ruling B2 step 3. The owner saw ponds in a line along one diagonal from 100-150 km.
The research report §3.4 names the defect the literature already measured. This section states the three
measurements, the one cure that landed, and the one decision that is the owner's.

### W13.1 THE MEASUREMENTS FIRST, AND THEY CAME FIRST

Three instruments, each a number a change can move, each written and read BEFORE anything was changed.
They live in `crates/bins/src/grid_bias.rs` (with their own unit tests), in the example
`crates/bins/examples/grid_bias.rs`, and in the `G-GRID` line of `crates/bins/examples/lake_census.rs`.

1. **TARBOTON'S CONE** (1997, Water Resour. Res. 33(2), 309-319, read in full). The ground rises with the
   great-circle distance from one node of the home moon, so every direction is equal and the upslope area
   of a node is an exact number — `w·r·(1 + cos θ)/sin θ`, the sphere's own reading of his `r/2`. The error
   is his own Table 2 quantity, `Mean((A − Â)²)` on upslope area counted in NODES. ★ A GRID-ALIGNED PLANE
   PASSES WHILE THE DEFECT IS WHOLE — he measures D8 at 0.065 on the plane and 118.88 on the inward cone —
   so no plane is measured here. His absolute numbers are NOT comparable to ours: his domain is 16 × 16 and
   never reaches a hundred pixels upslope, ours reaches ten thousand nodes, which is why the relative
   reading is printed beside the MSE.
2. **HYVALUOMA'S ROTATION** (2017, IJGIS 31(11), 2272-2285; the open PDF was read). His own test rotates the
   DEM by an angle, routes it, rotates the accumulation back by bilinear interpolation and takes the Pearson
   correlation against the unrotated one (his Eq. 8). ★ OURS IS A DIFFERENT REALISATION OF THE SAME IDEA,
   stated here so nobody reads it as his: the cone is the SAME FIELD at every azimuth, so the true
   accumulation is unchanged by a turn about the cone's own pole; we lay the accumulation on a ring-by-
   azimuth map about that pole (24 rings by 72 bins of 5°) and correlate the map with itself turned by whole
   bins. No interpolation is needed, and a perfect router scores exactly one at every angle. Beside it the
   FOURFOLD AMPLITUDE: each ring divided by its own mean, the fourth Fourier mode of what remains — his
   *"fourfold rotational symmetry which reflects the underlying grid structure"*, as a percentage. A perfect
   router scores zero.
3. **THE LONG-AXIS HISTOGRAM** (gate G-GRID). A long axis is fitted to every lake patch of four nodes or more
   and to every valley trunk — a land node whose discharge is at least a hundred times the rain on its own
   node, over eight receiver steps, because one step is one of eight directions by construction and says
   nothing. ★ THE AXIS IS STATED IN THE FACE'S OWN CELLS, never in metres: the two directions of a cube face
   are not at a right angle in metres away from the face's middle, and a first form of this instrument read
   174° for a row that is 0°. The FLATNESS is the biggest bin over the mean bin; a router with no direction
   of its own reads one.

### W13.2 THE CURE THAT LANDED: THE FLAT'S DISTANCE IS IN METRES, NOT IN HOPS

★ **THE ARITHMETIC SLIP THE REPORT CLEARED FOR THE SLOPES WAS STILL WHOLE INSIDE EVERY FLAT.** §3.4 measured
that `steeper()` divides the drop by the real chord, so our diagonals are not artificially cheap on a slope.
The FLAT had no such care: `assign_flat_receivers` ranked on a breadth-first HOP COUNT, which charges the
stencil's diagonal — the root of two of a row — the same one step as a row. A filled hollow is a flat, so
the cheapest way across every lake on the planet ran on one fixed diagonal.

The distance across a flat is now the lattice's OWN CHORD IN WHOLE METRES, summed along the path — a
shortest path, so it depends on no visit order — and a flat node takes the neighbour with the smallest
`distance + chord`, ties to the smaller index. The winner's distance is exactly this node's own less the
chord between them, so the distance FALLS along every step and the flat's tree can hold no ring. Only the
flat's draining SHORE seeds the walk, never every routed node on the planet (SL9).

★ **WHY THE SUMMED CHORD AND NOT CORDONNIER'S STRAIGHT LINE.** Cordonnier, Bovy & Braun 2019 §2.3.2 rank on
`cost(n) = ||n − n_out||`, a straight line to ONE outlet, and say plainly that it *"does not yield the
perfect path patterns that one would obtain by including obstacles in the computation of the Euclidean
distance on a regular grid"*. The summed chord IS that obstacle-aware distance; and our flats do not have
one outlet — the priority flood leaves a whole SHORE of already-draining nodes — so there is no single node
a straight line could be measured to. Their own warning against the naive alternative is the one we were
guilty of: a breadth-first assignment *"leads to more pronounced straight lines after erosion, due to the
four- or eight-connectivity."*

### W13.3 THE NUMBERS, BEFORE THEN AFTER

| reading | before | after | a perfect router |
|---|---|---|---|
| G-GRID valleys, flatness at 4 bins (96 804 then 96 893 trunks) | **2.101** | **1.284** | 1.006 (95th 1.012, measured by simulation) |
| the 135° bin of those trunks | 50 840 (52.5 %) | 23 953 (24.7 %) | a quarter |
| G-GRID valleys, flatness at 12 bins | 4.937 | 2.903 | 1.017 (95th 1.028) |
| G-GRID lakes, flatness at 4 bins (50 then 80 patches) | 1.440 | 1.600 | 1.200 median, 95th 1.450 — NOT YET A GATE |
| the filled disc: the detour out of the flat | 1.2406 (worst 1.404) | **1.0461** (worst 1.094) | 1.0000 |
| the filled disc: the bearing against the straight line to its exit | 2.82° (worst 12.0°) | **1.04°** (worst 8.1°) | 0.00° |
| Tarboton's inward cone, pole at a face's middle | 57 768 nodes², 140.3 % | **57 768 nodes², 140.3 %** | 0 |
| Tarboton's inward cone, pole near a face corner | 244 239 nodes², 275.6 % | **244 239 nodes², 275.6 %** | 0 |
| Tarboton's outward cone (both poles) | 63.2 and 56.3 nodes² | **63.2 and 56.3 nodes²** | 0 |
| the rotation score at 15°, 45°, 90°, face's middle | 0.032, 0.582, 0.964 | **unchanged** | 1.000 at every angle |
| the fourfold amplitude, middle and corner | 5.64 % and 7.03 % | **unchanged** | 0 % |

★ **THE CONE DID NOT MOVE BY ONE DIGIT, AND THAT IS THE PROOF THE CURE TOUCHES ONLY FLATS.** A cone holds no
flat at all (`flats 0 of 27 744 nodes`), and a test that could have failed says the same of the routing:
a field with no flat keeps the steepest descent it always had, receiver for receiver and chord for chord.

The census's other lines moved because the receivers moved, before then after: the solve **117.6 s to
111.5 s** (the flats shrink, so the walk is shorter than the hop count it replaced); land 21.78 % to
21.78 %; lakes 0.05 % to 0.07 % of the land, 362 to 506 patches, largest 72 to 87 nodes; the depressions
25 292 to 29 123 and the basins that end dry 97.9 % to 97.5 %; the ice 4.06 % to 4.02 % of the land,
thickest 8 325 m to 7 599 m; G-BUZZSAW 1 322 m to 1 316 m; the craters 1 610 to 1 610, unchanged, because
they are stamped before the water works. The sea stands at 3 566 m as it did; the ocean share moved by one
part in ten thousand (7 860 to 7 859). `GENERATOR_VERSION` is 16, and the seed-only chunk tables stand.

### W13.4 ★ G-GRID IS STILL RED, AND THE NEXT CURE IS THE OWNER'S WORD

1.284 is not 1.006. The rest is D8's own fourfold bias, and **no rule about flats can touch it** — the cone
and the rotation readings say so by not moving. Ruling B2 step 3 names the branch, and this is it.

- **THE MFD WITH A CARDINAL WEIGHT (Hyvaluoma).** It costs the SINGLE-RECEIVER TREE. Every node would carry
  up to eight receivers with weights, and the sweep, the deposit, the topological order, the basins, the
  depression hierarchy, the lakes' budget and the artifact's own receiver byte every one of them read ONE
  receiver. And the weight is a FITTED number — Hyvaluoma abandons the geometric interpretation and the
  optimum runs from about 2.6 to about 8 by flow exponent and by terrain — which ruling T9 calls a defect
  unless a published law computes it. **NOT BUILT. THE OWNER'S WORD IS OWED.**
- **RHO8 IS REFUSED BY NAME.** A random tie-break breaks SL10's byte-for-byte gate, and Tarboton rejects it
  on principle: *"Upslope and specific catchment areas are deterministic quantities that we should be able
  to compute in a repeatable way."*
- **A HEXAGONAL MESH** is the only cure that REMOVES the anisotropy instead of tuning it — six neighbours,
  all equidistant, no cardinal-and-diagonal split at all — and it is **A NEW WORLD under SL5**. Named here;
  not built; the owner's decision.
- **AND STEP 4 MAY DO IT FOR FREE.** The river lines, the width law and the carve come next. A valley cut by
  a river has a reason to point where it points, and the histogram may flatten without any change to the
  router at all. The cheapest order is to measure G-GRID again after step 4, before spending the owner's
  decision.

Still owed: the LAKES' histogram is not a gate yet (80 patches; a perfectly flat router reads a median
1.200 at four bins with a 95th percentile of 1.450, so 1.600 says almost nothing) — it becomes one when
step 4's rivers raise the lake count. HR5 coverage of the new lines is UNMEASURED: `coverage-fast` was not
run in this session.

**Example.** A pilot flies the belt at 120 km. Before this, a line of ponds ran north-east across her whole
view, one node wide, straight as a ruler, because eight diagonal hops counted eight and eight row hops
counted eight. Now the water in each hollow leaves by its own nearest shore — the detour is four percent
instead of twenty-four, and the bearing is one degree off the straight line instead of three. The line of
ponds is gone. What she can still see, if she looks for it, is that more valleys run along the grid's rows
than should: one bin of four holds a third of the trunks where it should hold a quarter. That is D8 itself,
and curing it is a decision about what the world's grid IS.

---

## W14. THE DRAWN DRAINAGE: THE RIVER LINES, THE WIDTH LAW, THE CARVE AND THE FLOODPLAIN (ruling B2 step 4)

Written 2026-09-22 after ruling B2 step 4 — *"the river lines with the width law, the carve and the
floodplain … this is the cure for the dunes"*. `GENERATOR_VERSION` is **17**.

### W14.1 THE DEFECT, IN ONE PARAGRAPH

The fine relief under one macro node of 8 192 m was a sum of noise octaves times a roughness factor,
and **nothing drained it**. Noise has no direction, so from 100–150 km it reads as ripples — the
owner's dunes. Real ground between a hundred metres and eight kilometres is shaped by water: valleys
that join into rivers, and RIDGES between the valleys. The solve has held the river network at the
node scale since 2026-09-19 — every row carries which neighbour its water leaves to and how much
water passes, and `slice_8d_design.md` §2 said plainly of the discharge *"built, and nothing reads it
back"*. This step is that reader.

### W14.2 THE LAWS, EACH WITH ITS BODY (ruling T9: no physical number is drawn)

1. **THE WIDTH AND THE DEPTH — Leopold & Maddock 1953** (USGS Professional Paper 252), calibrated on
   EARTH: `w = 3.9·√Q` metres and `d = 0.4·Q^0.4` metres at Earth's gravity, their own downstream
   exponents `b = 0.5` and `f = 0.4`. The two coefficients are the project's own committed
   calibration (`03_erosion_rivers.md` §6.1) and are NAMED as calibrations. **Gravity is not one**: a
   bed's shear is `ρ·g·d·S`, so at a fixed critical shear the depth falls as `1/g` and continuity
   `Q = w·d·U` puts the width up as `g` — a low-gravity moon gets deep, narrow rivers with no new
   constant, and a unit test measures it. The float fence forbids a power, so both laws are
   **256-entry committed tables** indexed by the row's own discharge class, each entry at its class's
   own midpoint (`crates/terrain/src/river/tables.rs`).
2. **THE FLOODPLAIN — Leopold & Wolman 1960** (USGS PP 282-B): a river's meander BELT is
   `A = 2.7·w^1.1`, and the exponent is one within the fit's own scatter, so the strip the river
   reworks — which IS the floodplain — is **2.7 times the channel's width**. It stands AT the
   bankfull water surface by the definition of bankfull, so it needs no freeboard of its own.
3. **THE VALLEY AND THE SMALL STREAMS — Horton 1945 and Strahler 1957** (ruling W4 item 2). The
   bifurcation ratio `Rb ≈ 4` and the length ratio `RL ≈ 2.3`. Each stream carries `Rb` tributaries
   of the next order down, each `1/RL` of its length and `1/Rb` of its discharge — and **that quarter
   of the discharge IS the table ruling W4 asks for**: the class is a base-two logarithm with two
   fraction bits, so a quarter is EIGHT classes down and the same committed width law reads the small
   stream's own width. **No discharge is stored anywhere.**
   * **The junction angle is computed, not drawn.** Howard 1971's minimum-power law gives
     `cos θ = S_trunk/S_trib`; the slope ratio between one order and the next is `√Rb = 2`, from the
     solve's OWN stream-power exponent (`c = Δt·K·√Q/L`, so `S ∝ A^−1/2`) and Horton's area ratio.
     `cos θ = 1/2` is **60°**, with a NAMED ±15° scatter about it (T9 allows a named scatter).
   * **The valley's half-width** is half the spacing between two streams of one order: `Rb` of them
     branch along a parent of length `RL·ℓ`, so they stand `RL·ℓ/Rb` apart — the half-spacing is
     `RL/(2·Rb) = 23/80` of the stream's own length. On the home planet the trunk's valley is
     4 710 m wide and the fourth order's is 168 m.
   * **The stream's own descent** is computed too: a tributary's mouth stands on its parent's ramp,
     and its drop is its length times its own slope, twice its parent's by the same `√Rb = 2`.

### W14.3 WHAT THE COLUMN KERNEL DOES WITH THEM

The carve is TWO WORDS on the read the host already hands the one column kernel
(`vd_recipe::plan::FieldRead`), and the kernel is the one both hosts compile (SL10):

* **`fine_ceiling`** — the valley's ceiling on the roughness factor. **Not a second factor**: the
  fine half of the octave sum has always been multiplied ONCE, by the greater of the placeholder
  noise's reading and the solved field's slope share, and the valley now CAPS that one product. On a
  stream's own floor the ceiling is zero and the fine octaves vanish; on a divide it is one and they
  stand whole. The relief is ARRANGED, never added.
* **`cut`** — how far under the valley floor the column's bed stands: the column's own height above
  **the line's straight descent ramp**, times the blend, plus the channel's shallow parabola.

Then the shore law closes it: inside a channel the column's water becomes **the river's own surface**
— the ramp, not the ground — and its side word says SEA, so `vd_recipe::height::shore` holds the bed
under the water exactly as it holds a sea floor. **The sea sheet then draws the river with no new
machinery at all.**

**THE GEOMETRY LIVES IN THE BODY'S OWN FRAME.** A stream is a pair of unit DIRECTIONS, never a pair
of face cells, so a cube seam is invisible to it and two chunks on either side of one measure the
same distance to the same line. The receiver slot is read through the lattice's own ring
(`neighbours`), which states it in the NODE's own face — reading it in the asking box's face sent a
river the wrong way at every seam, and that defect was found and cured here.

**THE CARD CARVES NOTHING.** The GPU column pass holds no artifact and states `FieldRead::none`,
whose ceiling is one and whose cut is zero. `lesser(m, one)` is `m` and `h − 0` is `h`, so its
arithmetic is byte for byte the one it always ran, and `just gpu-drift` compares like with like.

### W14.4 THE COST, AND THE CAPS IT BOUGHT

The lines are gathered ONCE A BOX, never once a column: the macro nodes the box's cells fall in plus
a ring of two, then the tributary tree under each trunk, pruned at every step against the box's own
ball. A column then walks the surviving lines, rejecting each in ten integer operations against its
line's own ball before any division or root is spent.

### W14.5 THE NUMBERS, MEASURED

**THE SOLVE DID NOT MOVE, AND THAT IS THE POINT.** Not one pass changed, so `lake_census` reads the
field it read before: land 21.78 % of the body, sea 78.21 %, the sea at 3 566 m, the lakes 0.07 % of
the land over 506 patches, G-ICE 4.02 %, G-BUZZSAW 1 316 m, G-CRATER 1 610, G-GRID's valleys 1.284 at
four bins and 2.903 at twelve — every digit the one it was. `HOME_ARTIFACT_DIGEST`,
`HOME_PLANET_SEA_M` (3 566), `HOME_PLANET_OCEAN_SHARE_Q4` (7 859) and `HOME_IDENTITY_MEASURED`
(`0xaef4c19c77ab7dbe`) are all UNCHANGED, and so are the seed-only chunk tables — `terrain_pin` and
`mesh_pin` pass on the committed bytes, because a body with no artifact reads a ceiling of one, a cut
of zero and a wall of zero. `just gpu-drift` is green over **5 979 of 5 979** boxes on an Apple M4
Pro. Only `GENERATOR_VERSION` moves, 16 → 17, and `DECLARED_PIN` with it.

**G-GRID, ASKED OF THE LINES A PLAYER ACTUALLY SEES.** The solve's own receiver trunks read 1.284 at
four bins and 2.903 at twelve (ruling W13, planet-wide over 96 893 trunks) and they are unchanged.
The DRAWN network — trunk and tributary together, 4 137 lines over the three stands — reads **1.222**
at four bins and **1.740** at twelve. The twelve-bin reading is the one that moved, and the cause is
Howard's junction angle: a tributary leaves its trunk at 60° ± 15°, which is not a direction the grid
has. The two samples are not the same population, and that is stated rather than hidden: the solve's
is planet-wide and the drawn one is three stands.

**G-DRAIN** (`crates/bins/examples/drain_census.rs`, rung 3, 8 m cells, 553 536 columns a stand; a
hollow counts only where it stands a tenth of a metre under its lowest neighbour, because a step of
one unit of the height word is 3 × 10⁻¹¹ m and no landform):

| the stand | hollows BEFORE | AFTER | over 1 m | over 10 m | the deepest |
|---|---|---|---|---|---|
| the belt | 0 | 492 (0.089 %) | 99 | 1 | 13.65 m |
| the coast | 0 | 19 (0.003 %) | 0 | 0 | 0.21 m |
| a plain | 0 | 359 (0.065 %) | 64 | 0 | 7.75 m |

**THE COST.** The gather runs ONCE A BOX and costs 18.7 to 28.7 µs there — five to seven nanoseconds
a column of the box; the per-column read costs 102 to 162 ns. The shipped column pass with the
artifact AND the drainage stands at **3 227 µs a chunk, 839 ns a column**, of which the drainage is
the sum above: about a fifth, so the pass was about 2.7 ms a chunk before this step. A box holds at
most **50 lines** of a ceiling of 128; no box overflowed and none was refused.

**THE ICE LAW'S OWN GATE (ruling B5), built beside this step.** `climate::tests::g_ice_the_snowline_
law_on_an_earth_like_charter` hands the snowline law Earth's own charter — obliquity 23.4°, Earth's
insolation, albedo, pressure, year and day — over land at Earth's mean elevation of 840 m on 29 % of
the nodes, and reads **11.29 % of the land under ice** against Earth's about 10 %, with the
equatorial snowline at 4 014 m against Earth's 4 500 to 5 000. The home planet's own 4.02 % stays a
CENSUS READING, as ruling B5 says it must.

**THE CAPS, AND WHY EACH IS WHERE IT IS.** The tributary tree stops at FOUR orders under the trunk —
a valley 168 m wide on the home planet — for two measured reasons. The body's own octave spectrum
ENDS at a 781 m wavelength, so a valley narrower than about a hundred metres carves relief the ground
does not otherwise have; and each order multiplies the lines a box holds by four, which at rung 3
overflowed a box's own ceiling (a 496 m box against a 73 m valley holds two hundred of them). A
deeper network is a later slice's work and it needs a spatial index under the box, not a bigger cap.

### W14.6 SIX DEFECTS FOUND BY MEASUREMENT, AND THE ONE THAT STANDS

★ **G-DRAIN IS THE GATE THAT FOUND THEM ALL.** A drained land has few closed hollows. The pre-carve
field holds NONE deeper than a tenth of a metre at the fine rungs — it is smooth, because the body's
octave spectrum stops at 781 m — so every hollow the carve leaves is the carve's own, and the number
could not hide. Six causes, each cured at its cause and each with its own before-and-after:

1. **A drawn stream had a HEAD and simply stopped.** A groove and a trench that stop dead leave a
   closed bowl. THE CURE: a tributary THINS to nothing at its own source, its width, depth and
   valley all going with the share of the stream already run — Hack's law `L ∝ A^0.6` with the width
   law gives `t^0.83`, `t^0.67` and `t`, all taken linear, a stated rounding. 0.378 % → 0.243 %.
2. **The floor followed the COLUMN's own ground**, so a groove rose and fell with the ground it
   crossed and left a hollow wherever a line crossed a dip. THE CURE: the floor is the straight
   descent RAMP between the line's two ends, and a tributary's own drop is computed — twice its
   parent's slope, by Horton's area ratio and the solve's own stream-power exponent.
3. **Damping the octaves RAISED the ground where the noise dipped**, so one groove was a chain of
   hollows and bumps. THE CURE: the carve may only ever LOWER
   (`vd_recipe::height::ReliefParts::carved`).
4. **The shore law cut a vertical trench at a channel's edge** wherever a line ran through high
   ground, because it holds a column a quarter of its ground's height from its water. THE CURE: the
   river's surface is stated only where the column's ground stands within the channel's OWN depth of
   it, and the shore then reads the CARVED ground and not the uncarved one. The coast stand: 1 355
   hollows → 82.
5. ★ **THE TWO-RUNG FADE READ THE LINE'S WIDEST VALLEY AND NOT THE COLUMN'S OWN.** A tributary's
   valley thins along it, so its last few metres carried a valley two cells wide AT FULL STRENGTH:
   the fine octaves fell from whole to nothing inside ONE cell. MEASURED: a slot **313.86 m deep and
   8 m across**. THE CURE: the fade is taken per column on the TAPERED width. The belt's deepest
   hollow fell from 313.86 m to 8.12 m and the count from 823 to 505.
6. ★ **A VALLEY MAY NOT BE DEEPER THAN ITS OWN WIDTH ALLOWS.** A 168 m valley cutting a mountain
   belt's 1 500 m of fine relief leaves walls of one in a tenth, which no ground holds. THE CURE is
   the solve's own published constant: a valley's floor may stand at most its HALF-WIDTH times the
   angle of repose (`TAN_REPOSE_DRY` = 0.70, the gradient dry rock stands at before it sheds) under
   the ground beside it, and both the damping and the channel's trench are held to that allowance.

★ **WHAT STILL STANDS, NAMED AND NOT CURED: a drawn tributary does not follow the ground's own
descent.** A trunk does, because the solve chose its receiver by the steepest drop; a tributary takes
its direction from Howard's junction angle and the seed's own hash, so it CROSSES the ground rather
than draining it, and its ramp then cuts a straight slot across a hill. The cure is a stream that
READS the field along itself — a per-column artifact read along every line, an order of magnitude
past this step's budget — and its home is the slice that gives a box a spatial index. The hollows
that remain are that, and they are metres rather than hundreds of metres.

★ **THE WORLD IDENTITY'S MEASURED HALF DOES NOT MOVE, AND THAT IS A GAP.** The self-check's eight
chunks read `GoldenFields`, whose `SparseRows` carry the height, the water, the facies and the
province — and NO receiver and NO discharge. So `ZField::drain` answers nothing there and those
chunks carve no valley. The two hosts still agree, because they fold the same literal through the
same kernel; but the identity no longer PROVES the carve agrees. OWED: two bytes on the golden row
and a re-record, so the number the client and the shard compare at the world hello covers the
drainage too.

★ **ALSO OWED.** The floodplain is the meander BELT and not the whole alluvial valley, so a great
river's flat strip is kilometres and not tens of kilometres. The meander itself is not drawn: a trunk
runs straight across its own node, and `03_erosion_rivers.md` §6.3's midpoint subdivision is not
built. A wet node carries no carve at all, so a river stops at a lake's shore and starts again at the
far one by OMISSION rather than by design (8d step 5's lake side word owns it). HR5 coverage of the
new lines is **UNMEASURED**: `coverage-fast` was not run in this session.

**Example.** A pilot flies the belt at 18 km. Under her a trunk river runs to the sea in a floodplain
a kilometre across; four valleys a kilometre wide join it, each with its own stream, and under those
four more of four hundred metres. Between them stand ridges the fine octaves still own whole. Where
she flew before, the same ground was a sheet of ripples with no direction in it at all.


### W14.7 THE TWO PICTURES (ruling B2: *"each step ends with the census's numbers and ONE picture"*)

* **THE BELT at 18.7 km**, the owner's own stand, the whole flight through the shipped path (the
  cluster up, the hull berthed, the pilot flying, rungs 5 to 11 drawn over 830 km):
  `runs/1790100310__pilot__a0/shots/belt_stand.png`.
* **A RIVER at 1.86 km**, the stand the census picked — the land node next to the home planet's
  highest-discharge coast node, discharge class 192, which the width law gives a channel 391 m wide
  and 16 m deep — the nose straight down:
  `runs/1790100521__pilot__a0/shots/river_stand.png` (the first, under-loaded) and
  `runs/1790100647__pilot__a0/shots/river_stand.png` (the one to look at).

★ **WHAT THE RIVER PICTURE SHOWS, AND THE ONE DEFECT IN IT.** The river is DRAWN, by the sea's own
sheet and with no new machinery: a band of water running across a flat floodplain the carve laid at
the line's own level. The defect is at its upstream end — the water stops in a ROUNDED CAP in the
middle of the plain. That is the water word's own rule showing: the river's surface is stated only
where the column's ground stands within the channel's own depth of the line's ramp (the fourth cure
in W14.6), and where the ground climbs past that the channel is still CARVED but no water is drawn.
On a hillside that is right — the water in a gorge is a thread. On a plain it is a river that ends,
and it is the first thing to fix when the ground-following stream lands.


## W15. A CELL TAKES THE SIDE MOST OF ITS GROUND STANDS ON — THE COAST MASK GETS A FOOTPRINT (2026-09-22; the owner, from 41 000 km: the globe *"shows squares of water on the land"*, and the same ground *"flips between water and land"* as the rings sweep under a moving hull; *"We need to fix it"*)

W10 made the water's SIDE a BIT of the fine row, read at every rung. It read that bit at the ONE
fine node nearest the cell's centre. A fine node is 8 192 m; a cell at rung 18 is 262 144 m and
covers about a thousand fine nodes, so one node in a thousand decided the whole cell — and each rung
picks ANOTHER centre node, so the same ground changed side at every ring swap. That is aliasing, and
the owner photographed it.

### W15.1 THE LAW, AS THE OWNER ORDERED IT

**A cell's side is the WET FRACTION of the fine nodes under its whole footprint: wet where the
fraction is at least a half.**

1. A side is a BIT and a bit does not fold — but a COUNT folds. Each host folds a COUNT PYRAMID from
   the mask ONCE: level `k` holds, per coarse node of `2^k × 2^k` fine nodes on each face, how many
   of them are wet.
2. **A parent's count is the SUM of its four children's.** So a coarse cell shows the side most of
   its ground stands on, and the finer rung REFINES that edge instead of contradicting it. That is
   the whole cure, and it is an invariant a test checks over every cell of the globe.
3. The footprint's level comes from the NODE'S OWN SIZE, never from a typed rung: ZERO while a cell
   is no wider than a fine node — where the nearest node's own bit is EXACT and nothing changes —
   then one level per doubling. On the home planet that is rung 13 and under unchanged, rung 14 and
   up by the fraction.
4. The compare is on whole counts, `count · 2 ≥ 4^k`, so no host divides and no host rounds. **A TIE
   IS WET**, stated: half a cell of sea reads as sea, which keeps a strait open from orbit instead
   of closing it into a land bridge the next rung cuts open again.
5. **The counts are DERIVED, never shipped.** The server folds them from its artifact; the client
   folds them from the mask when the mask lands. No byte crosses the wire, the artifact's digest does
   not move, and the two hosts fold one mask (SL10).
6. The MORPH reads the cell OF ITS OWN RUNG. A morph that read a rung-0 cell would pull a vertex
   toward a shoreline the rung it morphs to does not draw.
7. **The seam.** Where a cell's footprint reaches past a face edge the cell takes its OWN face's
   coarse node. A footprint reaches past an edge only by the half node a column's halo overhangs,
   which is a rendering margin and never a drawn cell.
8. The card is untouched: it holds no artifact, reads `SIDE_UNKNOWN`, and its bytes did not move.

**Example.** A hull at 41 000 km draws the belt at rung 18. One of its cells covers a headland and
the bay beside it: 300 of the cell's 1 024 fine nodes are sea, so the cell is LAND — as the ground
under it mostly is. At rung 17 the quarter that holds the bay counts 200 of its 256 nodes wet and
stands SEA, so the bay opens as the ring sweeps in and the headland never flips.

### W15.2 THE DEFECT'S OWN NUMBER, BEFORE AND AFTER

`vd-bins/examples/far_side_probe`, EVERY cell of the globe: how many cells stand on a side the
ground under them does not mostly stand on.

| rung | cell | nodes a side | cells | disagree BEFORE (the centre node) | disagree AFTER (the footprint) | parent ≠ sum of children |
|---|---|---|---|---|---|---|
| 14 | 16 384 m | 2 | 2 217 984 | **2 999** | **0** | 0 |
| 15 | 32 768 m | 4 | 554 496 | **780** | **0** | 0 |
| 16 | 65 536 m | 8 | 138 624 | **208** | **0** | 0 |
| 17 | 131 072 m | 16 | 34 656 | **83** | **0** | 0 |
| 18 | 262 144 m | 32 | 8 664 | **60** | **0** | 0 |

Sixty wrong cells at rung 18 is 0.7 % of the globe's cells — and every one of them stands at a coast,
which is where the eye is. They are the squares the owner saw. The AFTER column is zero BY
CONSTRUCTION, and the instrument is the test that could have failed.

`vd-bins/examples/shore_step` on the belt's coast (400 lines of 600 km, the owner's own stand
`0.617270 -0.437286 -0.654033`), how far the shore's crossing moves at each ring swap. BEFORE is the
same binary with `VD_SHORE_RULE=old`, which builds the levels with no counts — ruling W10's centre
node — so the two columns are the SAME lines under the two rules.

| rung pair | the coarser cell | median BEFORE | median AFTER | p90 BEFORE | p90 AFTER | max BEFORE | max AFTER |
|---|---|---|---|---|---|---|---|
| 9 → 10 | 1 024 m | 9 | 9 | 49 | 49 | 20 698 | 20 698 |
| 10 → 11 | 2 048 m | 0 | 0 | 0 | 0 | 0 | 0 |
| 11 → 12 | 4 096 m | 0 | 0 | 0 | 0 | 0 | 0 |
| 12 → 13 | 8 192 m | 0 | 0 | 0 | 0 | 0 | 0 |
| 13 → 14 | 16 384 m | 0 | **1 152** | 9 241 | 10 457 | 35 866 | 60 058 |
| 14 → 15 | 32 768 m | 10 457 | **8 473** | 20 633 | 21 977 | 34 112 | 33 817 |
| 15 → 16 | 65 536 m | 21 888 | **39** | 44 441 | 33 753 | 75 072 | 61 376 |
| 16 → 17 | 131 072 m | 40 | **39** | 103 961 | 62 298 | 190 656 | 192 998 |
| 17 → 18 | 262 144 m | 38 | **38** | 62 720 | **39** | 192 960 | **128 704** |

The rungs under 14 read ONE fine node under both rules, so their rows are identical by
construction — and they are, to the metre, which is the instrument's own self-check. The fine swaps
from the same run: 3 → 4 median 0 (max 28 m), 4 → 5 0 (16), 5 → 6 0 (6), 6 → 7 0 (14), 7 → 8 1 (26),
8 → 9 2 (max 20 726, the drawn drainage's own tail, ruling B2 step 4 — not the coast's).

★ **READ THESE AGAINST THE COARSER RUNG'S OWN CELL, not against a fine node.** A cell's side is a
BIT, so the finest shoreline a rung can draw is its own cell edge: at rung 15 that is 32 768 m. The
after medians are 0.07 of a cell (13 → 14), 0.26 (14 → 15) and under 0.001 for every swap above. A
rung is drawn only from the distance where its cell stands about ONE PIXEL high, so those medians
are a quarter of a pixel and less. The p90 at the top swap falls from 62 720 m to 39 m and the max
from 192 960 m to 128 704 m. The two rises are at 13 → 14 (median 0 → 1 152 m): rung 13's cell IS
one node, and the old rule's rung-14 cell read that same node, so the two agreed by accident; the
new rung-14 cell states what its FOUR nodes mostly are, which is the answer the owner asked for and
which costs a twelfth of a cell.

THE TAIL IS NOT ZERO, for the reason ruling W10 already named: the instrument takes the FIRST
crossing along a line, and where a coarse cell drowns or raises a whole shallow lagoon the first
crossing becomes ANOTHER crossing. The p90 says how few.

### W15.3 THE NEAR FLICKER IS ANOTHER DEFECT, MEASURED AND NAMED, NOT CURED

The owner, after W10: *"It's not only far-sight. When I fly very close over water, it also changes
all the time."* The near rungs read ONE fine node under both rules, so W15 is neither the cause nor
the cure. A new instrument names what is.

`vd-bins/examples/water_edge_step` walks lines across a water body and reads, per rung, the column's
GROUND and the water it really holds — the row's water through `artifact::sample_water`, then the
drawn drainage's own water word on top of it, which is exactly the pair `height::height_field_m`
reads. It reports how far the crossing of the two moves at each ring swap, how many sample points
HOLD a stream's surface at one rung and not at the other, and how many stand under their own water.

**1. THE SEA'S COAST DOES NOT FLICKER.** The first attempt read nothing and says why: `shore_step`
at rungs 0 to 9 with a 16 m step, over 400 lines of 20 km around the belt stand, found **0 lines
that cross a shore** — the belt stand is over 500 km from the nearest coast node, so a near probe
must be told where a coast IS. The instrument therefore searches for one. 200 lines of 8 km across
the coast node nearest the belt stand, 16 m steps, 100 000 sample columns a rung, rungs 0 to 9:

| rung pair | lines | edge step median | p90 | max |
|---|---|---|---|---|
| 0 → 1 | 103 | 0.0 m | 0.0 m | 0.0 m |
| 1 → 2 | 103 | 0.0 m | 0.0 m | 0.0 m |
| 2 → 3 | 103 | 0.0 m | 0.0 m | 0.1 m |
| 3 → 4 | 103 | 0.0 m | 0.1 m | 5.4 m |
| 4 → 5 | 103 | 0.0 m | 0.1 m | 7.8 m |
| 5 → 6 | 103 | 0.0 m | 0.9 m | 251.7 m |
| 6 → 7 | 102 | 0.1 m | 1.8 m | 9.7 m |
| 7 → 8 | 101 | 0.6 m | 6.7 m | 5 229.5 m |
| 8 → 9 | 101 | 1.1 m | 3.6 m | 8.2 m |

7 528 of the 100 000 columns stand under the sea at rung 0 and 7 520 at rung 9. W6 and W10 hold at
the near rungs, as they were measured to. **The sea is not what flickers close up.**

**2. ★ A LAKE IS NOT DRAWN AT ALL — A DEFECT FOUND BY LOOKING, NOT A FLICKER.** 200 lines of 24 km
across the lake node with the most lake neighbours on the home planet (node 1 150 231, all eight of
its neighbours lake), 16 m steps, 300 000 sample columns a rung:

| rung | columns under their own water | of those, holding a STREAM's surface |
|---|---|---|
| 0 | **65** of 300 000 | 65 |
| 5 | 64 | 64 |
| 9 | 28 | 28 |

**Not one column stands under the LAKE's own surface at any rung**, and the sixty-five that hold
water hold a stream's. THE CAUSE, read in the code and not searched for: the coast mask carries the
row's `FACIES_SEA` bit only, so a LAKE node reads `SIDE_LAND`; `vd_recipe::height::shore` then holds
that column at least a QUARTER of its ground's own height ABOVE its water — and its water is the
lake's own level. The shore law drains every lake on the planet, at every rung, and it has done so
since W10 stated a side for every node. Before W10 a lake column read `SIDE_UNKNOWN` and its own
ground's sign kept it under the water.

★ **NOT CURED HERE, and the reason is the ruling's own order.** A lake does not FLICKER — it is
absent — so the near-flicker question it was measured under does not answer it. The cure is one
word: the mask's bit must mean *"this node stands at or under ITS OWN water"*, sea or lake, which
moves every mask bit of every lake, the artifact's DIGEST and `ARTIFACT_VERSION` (a stored artifact
would have to re-solve), and it is exactly the law W6 and W10 both left owed to **8d step 5**. It is
put to the owner with its number, not taken.

**3. ★ THE RIVERS AND THE STREAMS ARE WHAT FLICKERS CLOSE UP.** 200 lines of 6 km across the
highest-discharge trunk near the belt stand, 8 m steps, 150 000 sample columns a rung:

| rung | columns under their own water | columns holding a stream's surface |
|---|---|---|
| 0 | 3 322 | 3 322 |
| 3 | 3 319 | 3 319 |
| 5 | 3 279 | 3 279 |
| 7 | 3 197 | 3 197 |
| 9 | **2 864** | **2 864** |

| rung pair | edge step median | p90 | max | WATER FLIPS |
|---|---|---|---|---|
| 0 → 1 | 0.0 m | 0.0 m | 2 968 m | 1 |
| 2 → 3 | 0.0 m | 0.0 m | 736 m | 2 |
| 3 → 4 | 0.0 m | 0.0 m | 1 152 m | 11 |
| 4 → 5 | 0.0 m | 0.0 m | 3 504 m | 31 |
| 5 → 6 | 0.0 m | 0.0 m | 3 689 m | 55 |
| 6 → 7 | 0.0 m | 0.0 m | 4 688 m | 53 |
| 7 → 8 | 0.0 m | 376 m | 4 584 m | 143 |
| 8 → 9 | 0.0 m | 0.0 m | 5 416 m | **190** |

Every wet column at this stand holds a STREAM's surface, not the sea's. **A water flip is a sample
column that holds the river's surface at one rung and not at the other**: 190 of them at the 8 → 9
swap against about 3 000 wet columns — one wet column in sixteen appears or vanishes as that ring
sweeps — and the water's edge jumps up to 5.4 km on the lines where it does. The cause is stated in
`river.rs` and needs no search: **the river's surface word is written only where the rung draws the
valley at FULL strength** (`rung == NOISE_ONE`), and the two-rung fade turns that off two rungs
before the valley folds away, while the channel's trench is scaled by the same fade, so the ground
rises out of its own channel at the same swap. The count of wet columns falls 3 322 → 2 864 from
rung 0 to rung 9: one river column in seven dries as the rings sweep out.

★ **NOT CURED, BY THE OWNER'S OWN ORDER.** Step 4 (the drawn drainage) is under the owner's review
for removal, so this is measured, named and left standing. If step 4 stays, its cure is a rung-free
water word: a channel either holds water or it does not, and the FADE may move the ground but may
not delete the surface.

### W15.4 THE PINS AND THE GATES

| pin | before | after |
|---|---|---|
| `GENERATOR_VERSION` (`vd-terrain/src/tag.rs`) | 17 | **18** — every column at a rung whose cell is wider than a macro node moved |
| `DECLARED_PIN` (`tag.rs`) | 11 998 781 851 654 533 174 | **2 752 796 325 948 008 293** |
| `ARTIFACT_VERSION` | 6 | **6, unchanged** — the counts are DERIVED, so no byte of the artifact, the store or the wire moved |
| `HOME_IDENTITY_MEASURED` (`vd-terrain/src/home.rs`) | `0xaef4_c19c_77ab_7dbe` | **unchanged** — the golden fields carry no mask and no counts, so their top level answers `SIDE_UNKNOWN` as before and the fine keys read one node |
| `HOME_ARTIFACT_DIGEST` (`vd-bins/tests/home_artifact_pin.rs`) | `[0xa2e7_b429_0add_d121, 0xb802_c886_c18d_7e72]` | **unchanged** |
| `HOME_PLANET_SEA_M` / `HOME_PLANET_OCEAN_SHARE_Q4` | 3 566 m / 7 859 | **unchanged** |

GATES, each with its number:

* `just gpu-drift` — **5 979 of 5 979** boxes byte for byte through the card's own gears, 4 of 4
  tests, adapter Apple M4 Pro. The card holds no artifact, reads `SIDE_UNKNOWN`, and its bytes did
  not move.
* `cargo test -p vd-terrain --test terrain_pin --test mesh_pin` — **3 of 3** and **2 of 2**: the
  SEED-ONLY chunk tables did not move, because a chunk with no artifact never reads a count.
* `cargo test --release -p vd-bins --test home_artifact_pin` — **1 of 1**, the home planet's artifact digests to its committed words (135.9 s).
* `cargo test --lib`: vd-terrain **208 of 208**, vd-client **328 of 328**, vd-bins **82 of 82**.
* `cargo clippy --all-targets -- -D warnings` on vd-terrain, vd-client and vd-bins
  (`--features dev-control,render`) — clean; `cargo fmt --all` — clean.
* HR5 coverage of the new lines: **UNMEASURED** (`coverage-fast` and `llvm-cov` were not run in
  this session).
* THE PICTURE: `runs/1790107614__pilot__a0/shots/belt_stand.png` — the owner's own belt stand, the avatar's eye
  18 696 m over the ground, after a clean store wipe and a fresh solve under `GENERATOR_VERSION` 18.
  It draws 6 460 chunks over rungs 5 to 11 out to 830 km with **0 pending** at 24 ms a frame, so the
  new read costs the picture nothing and leaves no hole. ★ IT SHOWS NO WATER: the belt stand looks
  over a dry tundra plain and no sea is in frame, so this picture neither confirms nor denies the
  shore. A picture that would is a flight at the owner's own 41 000 km, and this tree holds no
  headless flight to that altitude — none was built, by the ruling's order.

**OWED:** the owner's look at the globe from 41 000 km under the footprint — this tree holds no
headless flight to that altitude, so only the belt stand was pictured; a cell's side is still a BIT,
so a coarse rung's shoreline is a staircase on its own cell grid, and the FRACTION is now a number
every host holds, so a later rule could hold the column BY the fraction and straighten that
staircase (an ask, not a defect); the QUARTER of W6 is still a stated choice; HR5 coverage of the new
lines is UNMEASURED.

## W16. ONE SHAPE AT EVERY DISTANCE (2026-09-23; the owner, after flying: *"It's a bit better, but definitely not fixed. When I fly over the water very close, it changes from water to surface and back. And when I'm flying away too far, it also changes, so I can't trust the far view and be sure that the shape will not change when I will fly closer. At some point water is not visible at all, just land. Also even when shape do not change from far view the moving rings are visible, because for some reason they have different color. Please solve it once and for all distances!"*)

**THE LAW, in the owner's own terms.**

1. **At EVERY distance the water and the land are ONE SHAPE from ONE FIELD.** A rung is a LOOK of
   that field, never a second field.
2. **A finer rung REFINES a coarser one and never CONTRADICTS it.** What the far view shows must
   still be there when the pilot arrives: a bay that opened at rung 16 does not close at rung 14,
   and a headland that stood at rung 14 does not drown at rung 16.
3. **A rung boundary is INVISIBLE.** No shape change, no water appearing or vanishing, no colour or
   brightness step. A ring you can see is a seam, and a seam is a defect (SL8).
4. **The drawn drainage of 8d step 4 is RETIRED.** It was the near flicker: a stamped stream's
   surface was written only where a rung drew its valley at full strength, so the water appeared and
   vanished as the rings swept. A channel may not be a thing a rung decides to draw.

**Example.** A pilot lifts off the belt's coast and climbs to 3 000 km. The bay under her shrinks
but never changes shape; the sea stays the sea at every ring she crosses; the tarn in the pass
behind her holds its own water until it is smaller than a cell, and then it simply folds away with
the ground it sits in. Nothing on the globe brightens or darkens as a ring sweeps over it.

**THE FOUR FAULTS, EACH MEASURED BEFORE IT WAS CURED.**

### W16.1 FAULT A — THE NEAR FLICKER WAS THE DRAWN RIVERS, AND THEY ARE RETIRED

Ruling W15 §3 measured it and left it standing under the owner's own order. The number:
`water_edge_step river`, 200 lines across the highest-discharge trunk near the belt stand, 150 000
sample columns a rung — **190 wet columns of about 3 000 appeared or vanished at the 8 → 9 swap
alone**, the water's edge jumped up to **5.4 km**, and one river column in seven dried between rung
0 and rung 9. THE CAUSE was stated in `river.rs` and needed no search: a stream's surface was
written only where a rung drew its valley at FULL strength, and the two-rung fade turned that off
two rungs before the valley folded away, while the channel's trench faded with it — so the ground
rose out of its own channel at the same swap.

**RETIRED, not repaired** (the survey `docs/investigation/2026-09-22/fine_terrain_models_and_crates.md`
§5 R4): the Horton–Strahler tributary synthesis, the 60° junctions, the stamped channel and belt,
the valley profile and the per-column river water word are DELETED, with `FieldRead::{fine_ceiling,
cut, wall}`, `ReliefParts::{fine_top, carved}` and `ZField::drain`. KEPT: Leopold & Maddock 1953's
hydraulic geometry and Leopold & Wolman 1960's meander belt, with their three law tests, because
the SOLVE still cuts a channel and lays a floodplain. `drain_census` is re-pointed at the shipped
kernel's own closed hollows.

MEASURED AFTER: **0 columns of 100 000 hold a stream's surface at any rung 0 to 9, and 0 flips at
every swap.** The sea was not disturbed — `water_edge_step coast` reads **7 518 wet columns at every
rung**, the same number at every one, with a median edge step of 0.0 m up to the 6 → 7 swap and
1.5 m at 8 → 9 (p90 3.7 m, max 8.2 m).

★ **WHAT THIS COSTS, STATED.** The fine relief under eight kilometres is the recipe's octaves under
the roughness factor again, so a plain reads as noise from the air until the survey's R2 lands a
SOLVED level under it. R4 was moved ahead of R3 for exactly this reason: a solved valley with a
drawn slash across it cannot be judged.

### W16.2 FAULT B — THE LAKES ARE BACK, AND THE MASK CARRIES A THREE-WAY SIDE

W15 found it by looking: not ONE column of 300 000 stood under a lake's own surface at any rung,
because the mask carried the sea bit alone, a lake node read `SIDE_LAND`, and the shore law held
every lake column a quarter of its ground's height ABOVE its own water. The shore law drained every
lake on the planet.

**THE LAW.** The mask's word is a SIDE, and there are three of them: land, sea, LAKE. Two bits per
fine node; `ARTIFACT_VERSION` 6 → 7 and the mask's bytes double (2.2 MB on the home planet). The
count pyramid folds BOTH counts, and `CoastCounts::side` is one rule at every rung: WET where
`wet · 2 ≥ 4^k` — ruling W15's own law, untouched — then a LAKE where `lake · 2 ≥ wet`. **A tie is
wet, and a tie between the two waters is a lake**, both stated: a lake's level is the higher of the
two, so reading a mixed cell as a lake leaves its ground under water at the cell's edge instead of
over it. ONE reader names a column's water and its side together (`artifact::column_water`), which
the chunk's column pass and the morph's height both call — a SEA cell takes the body's own sea
exactly, a LAKE cell the row's own level, a LAND cell the nearest row's (ruling W6: a land column
beside the sea must hold the sea's level, or the fine octaves dig a dry pit under the water beside
it). A LAKE cell whose field names no level falls back to LAND, stated, because a guessed level is
a drop and a drop is a seam. `sample_water` is deleted: two readers of one thing is how a chunk and
a morph come to disagree.

MEASURED AFTER (the same lines W15 read): **279 989 of 300 000 columns stand under their own water
at EVERY rung 0 to 9**, against zero before; the lake's edge steps a median of 0.0 m at every swap
(max 0.7 m). Over 120 km lines at rungs 9 to 16 the lake is there at every one (147 252 wet columns
of 375 000 at rung 9, 97 005 at rung 16 as it folds into a cell wider than itself) and every median
edge step is under the coarser rung's own cell — 5 364 m at 14 → 15, which is 0.16 of that rung's
32 768 m cell. Ruling W15's invariant still holds with the lake bit beside the wet one
(`far_side_probe`, every cell of the globe at rungs 14 to 18): **0 disagreements and
`parent ≠ sum of children` 0 at every level.**

**THE WIRE DID NOT CHANGE SHAPE.** `BulkMsg::ArtifactCoast` still carries a blob of mask bytes in
parts; only the bytes' meaning moved, and the artifact's own `version` word inside `ArtifactHead`
guards it, exactly as the store's open already refuses a version it does not know. `PROTO_MINOR`
stays at **34**. THE COUNT, STATED: the home planet's 8 871 936 fine nodes take 2 217 984 bytes of
mask against 1 108 992 before — 2.2 MB, **68 parts of 32 KiB against 34** — and the whole artifact
is 96 850 524 bytes.
★ **AND THE CLIENT REFUSED EVERY MASK UNTIL ONE LINE MOVED.** `ArtifactReceiver::accept_coast`
checked the assembled mask against `nodes.div_ceil(8)`, a count it spelled for itself. MEASURED the
first time a flight ran under the new mask: `artifacts.refused` 4 of 4 realms, `coast_held` empty,
`awaiting_artifact` 112 494 columns, and every far chunk drawing the RECIPE's own relief instead of
the solve's. The line now reads `vd_terrain::artifact::coast_bytes(nodes)` — the generator's own
count, never a second opinion.

**STILL OWED:** the lake's EDGE is a staircase on the macro lattice — a wet row holds the lake's
level and its dry neighbour the sea's, so the surface jumps at the line between two nodes. Ruling
W6 owes a lake's shore the same law it gave the sea's.

### W16.3 FAULT C — THE WATER WAS THERE; THE SHEET SANK THROUGH THE GROUND

The measurement came first and it decided which half was wrong. `far_water_share`, EVERY chunk of
the globe at rungs 12 to 18, through the shipped box: **the columns standing under their own water
are 78.5, 78.5, 78.6, 78.5, 79.7, 79.4 and 79.7 % of the globe**, against the sea's own 78.6 % of
the surface. The MODEL is whole at every rung. So the owner's "no water at all, just land" is a
DRAWING fault, and the number says so.

THE CAUSE: the water sheet drew ONE FLAT QUAD per block of eight cells at every rung. A flat quad
is a CHORD of the water's sphere, so its middle stands `W² / 4R` under that sphere, and the ground's
own mesh is a quad per CELL, which dips sixty-four times less. The EXTRA dip is 41.7 m at rung 12
and **170 669.8 m at rung 18**, while the shore law holds the drawn sea floor only about a kilometre
under the water over a four-kilometre abyss. MEASURED: the wet columns whose sheet dived through
the ground were 175 033 at rung 12, 158 587 of 165 391 at rung 16 (95.9 %), **ALL 73 284 at rung 17
and ALL 18 379 at rung 18.** From rung 17 up the whole globe's sheet was buried.

**THE LAW: the sheet is built on the GROUND'S OWN GRID, and a block stands in for that grid only
while its extra dip is smaller than the water's own depth there.** The quad measures its own chord
against its own depth. No rung is named and no number is drawn. AFTER: zero buried columns at every
rung, by construction, with the three statements that could fail held in
`position::water_sheet_tests`.

### W16.4 FAULT D — THE RINGS HAVE DIFFERENT COLOURS BECAUSE A RUNG IS DRAWN UNDER ITS OWN RADIUS

MEASURED FIRST, in the ladder's own arithmetic (`vd-bins/examples/ring_sink`, every rung of the home
planet). A rung's chunk is drawn SUNK along each vertex's radial while the FINER rung still covers
it — `ladder_fade.wgsl` subtracts `sink · (1 − risen(d))` — and `risen` ramped LINEARLY from the
fade-in band's inner edge to a `sink_end` that stood far PAST the fade-in edge, so that AT the edge
the rung still stood ONE FINER CELL under its own surface. That residual is the chord a finer
triangle cuts under a coarser crease, and it had been measured as dark specks when the ramp ended at
the edge. The consequence is arithmetic, and it is the ring:

| rung | the residual AT the edge | the annulus it decays over | as a share of the distance the rung is drawn from |
|---|---|---|---|
| 12 | 2 048 m | 141 373 m | **7.2 %** |
| 13 | 4 096 m | 284 245 m | **7.3 %** |
| 14 | 8 192 m | 711 981 m | **9.1 %** |
| 18 | 131 072 m | 11 391 689 m | **9.1 %** |

Over that annulus the drawn ground is tilted by about two thirds of a degree and lifted through
kilometres of air, so its Lambert shade and its aerial perspective both differ from the ring outside
it. The owner's own screenshot draws rungs 12, 13 and 14 — three such annuli, one inside the next.

**THE LAW: A RUNG IS DRAWN AT ITS OWN RADIUS WHEREVER NOTHING COVERS IT.** The sink exists to hold a
coarser mesh UNDER a finer one while both are drawn, and the finer one is drawn only inside the
band, so the ramp ends AT the fade-in edge and the sink past it is ZERO.
`ladder_view::AskBound::sink_end_m` is now `fade_bands(rung, rungs).0[1]` — one line.

MEASURED AFTER: **the residual at the edge is 0 m and the annulus is 0 m at every rung from 1 to
18.** ★ WHAT THIS GIVES BACK, STATED: the specks the old end cured stand at the edge again, a coarse
triangle showing through a finer chord in a thin ring, at most one finer cell tall. A ring of specks
one cell wide is a smaller seam than a ring of wrong altitude 7 % of the screen wide.

### W16.5 THE PIXEL GATE, AND WHAT IT CAN AND CANNOT JUDGE

`vd-bins/examples/ring_step` reads a NADIR screenshot and prints the mean luminance of concentric
annuli about the point under the eye, in SECTORS — because the ring's own mechanism TILTS the ground
radially, so it brightens the sun-facing side and darkens the far side by the same amount and a
whole-circle mean CANCELS it (measured: 0.155 % over whole circles against 1.19 % over sectors, on
the same picture). The gate is **2 % of the local mean**, the common working figure for a
just-noticeable luminance step (Blackwell 1946; Barten 1999; DICOM PS 3.14 builds its ladder on
about one per cent).

★ **THREE THINGS THE PICTURE COULD NOT DO, AND THE NUMBERS THAT SAY SO.**

1. **A NADIR PICTURE AT 3 000 km SHOWS ALMOST NOTHING.** The whole frame reads within 1.2 % of one
   luminance: the aerial perspective at that range washes the ground to a flat brown. The owner's
   own screenshot at 3 300 km is CRISP, and it was taken from a hull looking at the globe across
   space, not straight down. A nadir picture at that altitude cannot judge a ring, and that is a
   reading, not an opinion.
2. **AT 30 km AND 3 km THE GROUND'S OWN RELIEF DOMINATES THE NUMBER.** A dune ridge crossing an
   annulus makes a sector step of ten per cent, so the worst-step figure there measures the
   landscape and not the ladder.
3. ★ **AND A NADIR FRAME CANNOT HOLD A RUNG BOUNDARY AT ALL.** A 45° frame at a 300 km stand spans
   eye distances of 300 to about 360 km, while the nearest residual annulus stands at 489 to
   513 km. MEASURED, and it is the cleanest reading of the four: the BEFORE and AFTER pictures at
   3 000 km, 300 km, 30 km and 3 km give the SAME worst sector step to three decimals
   (1.191 % / 2.317 % / 10.323 % / 15.014 %), and the two files differ only in the HUD's own text.
   The cure moved nothing in those frames because no ring was ever in them.

So the ring's own defect is measured where it lives — in the ladder's arithmetic (§W16.4) — and the
pixel gate's four stands are recorded as what they are: a control that could have shown a change
and did not, for a reason that is itself a number.

★ **OWED, AND NAMED:** a picture that can judge a ring at 3 000 km — a hull's own third-person view
across the globe, the framing the owner flew — and the aerial perspective that washes a nadir view
at that range, which is a rendering defect of its own and is not one of this ruling's four faults.

### W16.5a FAULT E — THE COASTLINE ACROSS FOUR ALTITUDES: WHAT WAS AND WAS NOT MEASURED

`vd-bins/examples/coast_line` reads a nadir picture and walks outward along a stated bearing,
finding the first crossing between WATER and LAND by the two materials the renderer really uses —
the sea `srgb(0.06, 0.24, 0.42)`, the ground `srgb(0.55, 0.50, 0.42)`, so a pixel is water where its
blue channel stands over its red one — and it converts the screen radius to GROUND METRES through
the ray-sphere sine rule, so four altitudes can be compared in one unit.

★ **THE FOUR PICTURES COULD NOT FEED IT, AND THE REASON IS A NUMBER.** A stand over a coast NODE
puts the shoreline within about 8 km of the point under the eye, and a 45° nadir frame at 300 km
covers 1 200 px of ground about 340 km across — so the shore stands inside the avatar's own ball at
the centre of the frame. At 3 km the shore is off the frame the other way. The crossings the
instrument then reports are the classifier firing on a grey terrain whose blue channel beats its red
by a count, which is noise and is reported as such.

**OWED, AND NAMED:** a stand chosen so that a NAMED BAY stands a third of the frame from the nadir
at 3 000 km — which means picking the stand by the bay's own width, not by a coast node — and the
same four pictures again. The instrument stands; only the stand is owed.

### W16.6 THE PINS

| pin | before | after |
|---|---|---|
| `GENERATOR_VERSION` (`vd-terrain/src/tag.rs`) | 18 | **19** — step 4's carve leaves every column it touched, and every lake column moves under its own water |
| `DECLARED_PIN` (`tag.rs`) | 2 752 796 325 948 008 293 | **11 554 407 130 392 614 244** |
| `ARTIFACT_VERSION` (`artifact.rs`) | 6 | **7** — the coast mask's word is two bits, so its bytes double and a version-6 store re-solves |
| `HOME_ARTIFACT_DIGEST` (`vd-bins/tests/home_artifact_pin.rs`) | `[0xa2e7_b429_0add_d121, 0xb802_c886_c18d_7e72]` | **`[0xab78_092a_0bc2_e1f0, 0x0727_7e1b_4ba7_b64f]`** — the mask is folded into the digest |
| `HOME_IDENTITY_MEASURED` (`vd-terrain/src/home.rs`) | `0xaef4_c19c_77ab_7dbe` | **unchanged, and re-measured to say so** |
| `HOME_PLANET_SEA_M` / `HOME_PLANET_OCEAN_SHARE_Q4` | 3 566 m / 7 859 | **unchanged** — the SOLVE did not move, not one pass and not one row |
| `PROTO_MINOR` (`vd-wire/src/version.rs`) | 34 | **34, unchanged** — the wire's arm keeps its shape; only the bytes' meaning moved, and the artifact's own version word guards that |

★ **WHY THE IDENTITY DID NOT MOVE, AND IT COULD HAVE.** The golden self-check fields are
`SparseRows`, which never stated the drainage words, so step 4's carve never reached the eight
golden chunks; and no golden row stands on a lake, so the three-way side word reads the same there.
Both halves were re-measured rather than argued.

### W16.7 THE GATES, EACH WITH ITS NUMBER

* `water_edge_step river` — **0 of 100 000 sample columns hold a stream's surface at any rung 0 to
  9, and 0 water flips at every swap** (before: about 3 000 wet columns and 190 flips at the 8 → 9
  swap alone). The stamp is gone.
* `water_edge_step coast` — **7 518 wet columns at every rung 0 to 9**, the same number at every
  one; the edge's median step 0.0 m up to 6 → 7, 1.5 m at 8 → 9 (p90 3.7 m, max 8.2 m), 0 flips.
* `water_edge_step lake` — **279 989 of 300 000 columns stand under their own water at every rung 0
  to 9** (before: ZERO at every rung); the edge's median step 0.0 m at every swap, max 0.7 m. Over
  120 km lines at rungs 9 to 16 the lake holds at every rung and every median swap is under the
  coarser rung's own cell.
* `far_side_probe` — **0 cells disagree with their own footprint at rungs 14 to 18, and
  `parent ≠ sum of children` is 0 at every level**, with the lake count folded beside the wet one.
* `far_water_share` — **78.5 to 79.7 % of the globe's columns stand under their own water at every
  rung 12 to 18**, against the sea's own 78.6 % of the surface; the sheet's buried columns fall
  from 175 033 / 341 748 / 171 023 / 86 234 / 158 587 / 73 284 / 18 379 (rungs 12 to 18) to **0 at
  every rung**.
* `ring_sink` — the residual at a rung's fade-in edge falls from one finer cell (2 048 m at rung
  12, 4 096 m at 13, 131 072 m at 18) to **0 m**, and the annulus it decayed over from 7–9 % of the
  distance the rung is drawn from to **0 m**, at every rung 1 to 18.
* `cargo test -p vd-terrain --test terrain_pin --test mesh_pin` — **2 of 2** and **3 of 3**: the
  SEED-ONLY chunk and mesh tables did not move, because a chunk with no artifact reads no drainage
  and an unknown side, which is the arithmetic the card has always run.
* `cargo test --release -p vd-bins --test home_artifact_pin` — **1 of 1** in 124.8 s: the home
  planet's artifact digests to its newly committed words, and the sea and the ocean share stand.
* `just gpu-drift` — **5 979 of 5 979** boxes byte for byte through the card's own gears, 4 of 4
  tests, adapter Apple M4 Pro (Metal). The card holds no artifact, reads no drainage and takes the
  unknown side, so its bytes could not move.
* `cargo test --lib`: vd-client **328 of 328**, vd-connection-plane **263**, vd-devproto **30**,
  vd-recipe **73**, vd-sim **664** (3 ignored), vd-terrain **205**, vd-wire **104**, vd-bins **82**.
* `cargo clippy --all-targets -- -D warnings` on vd-recipe, vd-terrain, vd-wire, vd-sim,
  vd-connection-plane, vd-client, vd-client-render, and on vd-devproto and vd-bins with
  `--features dev-control,render` — clean; `cargo fmt --all --check` — clean.
* HR5 coverage of the new lines: **UNMEASURED** (`coverage-fast` and `llvm-cov` were not run in this
  session, by the slice's own instruction).
* ★ ONE RED STANDS AND IT IS NOT THIS SLICE'S: `vd-bins --test dual_cluster_crossing_smoke` fails at
  a DERIVATION assert — the ring sibling's wake radius reads 3.962e14 m WIDER than its own centre
  distance — before any cluster starts. It reads only `vd-core`'s world and roster, and this slice
  touched no line of `vd-core`, `vd-node` or `vd-wire`'s geometry. Named here so it is not mistaken
  for a consequence.

### W16.8 THE PICTURES

* `runs/1790153933__pilot__a0/shots/belt_stand.png` — the owner's own belt stand, the eye 18 696 m
  over the ground, after a clean store wipe and a fresh solve under `GENERATOR_VERSION` 19 and
  `ARTIFACT_VERSION` 7. It draws **6 460 chunks over rungs 5 to 11 out to 830 km with 0 pending at
  23.9 ms a frame** (ruling W15's own reading on the same stand: 6 460 chunks, 24 ms), so the
  three-way mask, the wider mask bytes and the sheet's new law cost the picture nothing. ★ IT SHOWS
  NO WATER — the belt stand looks over a dry tundra plain — so it neither confirms nor denies the
  shore; what it does show is the fine relief WITHOUT step 4's drawn valleys, which is the dunes
  the survey's R2 owes a solved level.
* The four NADIR stands over a coast under a star 22° up, before the sink's cure:
  `runs/1790151306__pilot__a0/shots/rings_3000km_sunbefore.png`,
  `runs/1790151368__pilot__a0/shots/rings_300km_sunbefore.png`,
  `runs/1790151431__pilot__a0/shots/rings_30km_sunbefore.png`,
  `runs/1790151493__pilot__a0/shots/rings_3km_sunbefore.png`; and after it,
  `runs/1790152123__pilot__a0/shots/rings_3000km_sunafter.png`,
  `runs/1790152186__pilot__a0/shots/rings_300km_sunafter.png`,
  `runs/1790152248__pilot__a0/shots/rings_30km_sunafter.png`,
  `runs/1790152311__pilot__a0/shots/rings_3km_sunafter.png`. §W16.5 states what they can and cannot
  judge, with the numbers.
* An earlier set of four at the same stands under a star **81.7°** up
  (`runs/1790150887…`, `1790150951…`, `1790151013…`, `1790151076…`) is kept as the control that
  says why a subsolar nadir picture judges nothing: its whole frame reads within **0.12 %** of one
  luminance.

### W16.9 WHAT IS OWED

1. **The owner's look**, on the belt stand and on a stand that shows water.
2. **A picture that can judge a ring**: an OBLIQUE stand — the owner's own framing, the ground
   running from under the nose out to the limb — for which `ring_step` already holds a `rows` mode.
   The four nadir stands provably cannot hold a rung boundary (§W16.5).
3. **A stand for fault E's coastline gate**, chosen by a BAY's own width rather than by a coast
   node (§W16.5a).
4. **The lake's EDGE**, which is still a staircase on the macro lattice: ruling W6 owes a lake's
   shore the same law it gave the sea's.
5. **The fine relief under eight kilometres**, which is noise again until the survey's R2 lands a
   solved level under it. That is the survey's own order, and R4 was moved ahead of R3 for it.
6. **HR5 coverage** of this slice's new lines, UNMEASURED here.
7. **The QUARTER of ruling W6** is still a stated choice, and the owner's word is still owed.

## W17. THE HANDOVER'S STEP KNEW THE OCTAVES AND NOT THE FIELD (2026-09-23; the owner, after flying with W16: *"Now it's way better, but for some of the far-view rungs the change is still visible — not for close or very far view"*)

**WHERE THE OWNER'S WORDS POINT, AND THE NUMBERS AGREE.** The near rungs are clean, the very far
rungs are clean, and the change shows in the middle. The middle is where the artifact's ROWS hand
the ground to the PYRAMID's first level: on the home planet rungs 0 to 9 read the rows and rungs 10
to 14 read level 1, whose node is the MEAN OF FOUR of the rows under it. That swap is the only one
a pilot in a hull can see, and it is the only one that breaks the ladder's own tolerance.

### W17.1 THE MEASUREMENTS CAME FIRST, AND FOUR OF THE FIVE CANDIDATES WERE REFUTED

Every reading is through the SHIPPED reader — `vd_terrain::height::height_field_m`, the very
function the client's geomorph calls — over 9 600 directions a rung pair, on the solved home planet
(`vd-bins/examples/rung_swap`, new).

**1. THE SHAPE — THE ONE THAT FAILED.** The surface at rung `L` against the surface at rung `L + 1`:

| L → L+1 | the field | median m | p90 m | p99 m | max m | max in CELLS of L+1 | max in PIXELS at the swap |
|---|---|---|---|---|---|---|---|
| 8 → 9 | rows → rows | 82.2 | 190.4 | 260.8 | 318.5 | 0.62 | 1.24 |
| **9 → 10** | **rows → level 1** | **107.7** | **273.8** | **442.3** | **1 442.5** | **1.41** | **2.82** |
| 10 → 11 … 12 → 13 | level 1 → level 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 |
| 13 → 14 | level 1 → level 1 | 0.0 | 0.0 | 0.0 | 1 467.8 | 0.09 | 0.18 |
| 14 → 15 | level 1 → level 2 | 0.5 | 36.4 | 290.8 | 3 116.3 | 0.10 | 0.19 |
| 15 → 16 | level 2 → level 3 | 1.2 | 72.5 | 438.6 | 3 084.2 | 0.05 | 0.09 |
| 16 → 17 | level 3 → level 4 | 8.1 | 214.2 | 1 266.3 | 9 382.8 | 0.07 | 0.14 |
| 17 → 18 | level 4 → level 5 | 41.5 | 555.6 | 2 020.6 | 9 199.5 | 0.04 | 0.07 |

The ladder's own pass line is the judge's (`rung_disagreement`): **p99 ≤ 1 pixel, max ≤ 2 pixels**,
where the tolerance is ONE CELL of the rung that takes over. **9 → 10 is the only pair over it** —
1.41 cells, 2.82 pixels — and it is over it at 445 km, the middle of a hull pilot's view. Rungs 10
to 13 agree to the METRE, because they read one level and the recipe keeps no fine octave there;
the far level swaps move kilometres but they happen at 14 000 km and more, where a kilometre is a
fifth of a pixel.

**2. THE SHADE — REFUTED.** The mean Lambert term `max(0, n · l)` over the extracted mesh's own
smooth normals, the SAME ground at the two rungs (one coarse chunk against its four children, a
sun 22° over the belt): **0.13 % at 9 → 10**, and the worst pair of the whole ladder is **1.95 %
at 17 → 18** — every one under the eye's 2 % working threshold (Blackwell 1946; Barten 1999;
DICOM PS 3.14). ★ So TOKSVIG IS NOT OWED: a coarse rung's smoother normals do not change the
ground's mean brightness on this body. The measurement could have failed and did not.

**3. THE WATER — REFUTED.** The side and the level the same directions read at the two rungs,
through `artifact::column_water` and `sample_side`: **0 columns of 9 600 flip side at 9 → 10** (and
0 at every swap up to 12 → 13; the far swaps flip 20 to 212 columns as a cell grows past a whole
bay). The sea's own drawn edge across the belt's coast (`water_edge_step sea`, 60 lines of 400 km
at rungs 8 to 13, 93 720 samples a rung): **43 250 samples under their own water at EVERY rung**,
and the edge moves a median 52.6 m (p90 197.6 m, max 233.4 m — 0.23 of a rung-10 cell) at 9 → 10
and **0.0 m at every swap above it**.

**4. THE CROSSFADE — REFUTED BY READING AND BY W16'S OWN NUMBER.** There is no alpha and no double
draw: a finer chunk's morph target IS the parent chunk's own mesh along each vertex's radial
(`chunks::geometry_from`), its morph NORMAL is the parent's smooth normal at that hit, and both
blend on ONE weight, so the two rungs COINCIDE at the fade-out edge by construction. The coarser
rung is held under the finer one by the sink, and since ruling W16 the sink's ramp ends AT that
edge — `ring_sink` reads a residual of 0 m at every rung 1 to 18.

**5. THE SINK'S START — REFUTED.** The ramp starts at the fade band's inner edge, which is where
the finer rung is still whole; the coarser rung is fully sunk there and the finer one covers it.
The sink at rung 10 is 2 433.7 m, and **0 of 9 600 directions** stand farther apart than that at
any pair, so no coarse mesh can show through a finer one at any swap.

**THE PICTURES SAY THE SAME.** Six OBLIQUE stands over the belt — the owner's own framing, the
ground running from under the nose out to the limb, the horizon a third of the frame down — at
20, 60, 150, 400, 1 000 and 2 500 km. Over LAND the row profile is a smooth monotone decay with no
kink at any rung boundary: at the 20 km stand the median row luminance falls from 123.5 to 94.7
over 460 rows and every second difference stands under 0.1 % of the local mean, which is the air's
own gradient and nothing else. What the frames DO show is the coastline and the aerial perspective.
★ **So the colour ring of W16 fault D is gone, and what is left is the SHAPE.**

### W17.2 THE CAUSE, NAMED WITH ITS NUMBER

`ladder_view::handover_step_m` read `BodyDefinition::step_bound_m` — **the OCTAVE TABLE alone**.
That is exact where both rungs read ONE field: at 8 → 9 the table states **319.4 m** against a
measured max of **318.5 m**. It is silent where the FIELD ITSELF changes under the ladder: at
9 → 10 the table states **897.7 m** against a measured **1 442.5 m**.

The consequence is arithmetic. Ruling T7's rule 2 widens the crossfade band by the step in cells,
and rule 3 floors the switch distance at `step / (2 · pixel)`. With 897.7 m both answer ONE:
`897.7 / 1 024 = 0.88` of a cell, and the floor stands at 390 km inside the tier rule's own 445 km.
**So both rules slept through the only handover on the home planet that needed them.**

### W17.3 THE LAW, AND NOTHING NEW CROSSES A REALM BOUNDARY

★ **A HANDOVER'S STEP IS THE STEP THE HANDOVER MAKES — the octaves the coarser rung drops PLUS the
field's own fold where the field changes under it.**

A fold takes FOUR nodes to ONE, so the coarse level states a height for the CENTRE of a square one
node wide, while each child's own height stands at `node / (2·√2)` from that centre — a quarter of
a node in each axis, the distance from a parent node's centre to a child node's. The two therefore
differ by the FIELD'S OWN SLOPE over that distance. The body already states that slope
(`BodyDefinition::slope_ref`, the first fine octave's own RMS slope, **0.124693** on the home
planet), and the lattice states the node, so **every host computes the step from what it already
holds**: no new word on any wire, no ask under SL6, and no drawn number under T9.

MEASURED, and it is a BOUND: the law states **722.3 m** at the rows → level-1 swap against a
measured field contribution of **544.8 m** (the whole 1 442.5 m less the octave table's 897.7 m) —
the law stands a third over the measurement, which is what a bound must do. The level's own fold
spread, folded over all 8 871 936 fine nodes, reads median 0.0 m, p90 63.0 m, p99 292.0 m and max
3 927.0 m: the MAX is 5.4 times the law and would have pushed the swap out fourfold for one ridge,
so the law is the geometry of the fold and not the worst node of the globe.

★ **AND THE SHARD SHIPS THE TILES TO THE SAME LINE.** `switch_floor_m` now lives in
`vd_terrain::artifact` beside `switch_m`, and `tile_reach_m` floors its ring with the very same
call, so a rung whose step pushes its ring out never waits on a tile nobody sent.

### W17.4 THE NUMBERS, BEFORE AND AFTER, PER RUNG BOUNDARY

| L → L+1 | the swap's distance BEFORE | AFTER | the stated step BEFORE | AFTER | max px BEFORE | AFTER | p99 px BEFORE | AFTER |
|---|---|---|---|---|---|---|---|---|
| 8 → 9 | 222 494 m | 222 494 m | 319.4 m | 319.4 m | 1.24 | 1.24 | 1.02 | 1.02 |
| **9 → 10** | **444 988 m** | **703 993 m** | **897.7 m** | **1 620.0 m** | **2.82** | **1.78** | **0.86** | **0.55** |
| 10 → 11 | 889 976 m | 889 976 m | 544.5 m | 544.5 m | 0.00 | 0.00 | 0.00 | 0.00 |
| 11 → 12 | 1 779 951 m | 1 779 951 m | 1 061.1 m | 1 061.1 m | 0.00 | 0.00 | 0.00 | 0.00 |
| 12 → 13 | 3 559 903 m | 3 559 903 m | 2 067.7 m | 2 067.7 m | 0.00 | 0.00 | 0.00 | 0.00 |
| 13 → 14 | 7 119 806 m | 7 119 806 m | 0.0 m | 0.0 m | 0.18 | 0.18 | 0.00 | 0.00 |
| 14 → 15 | 14 239 611 m | 14 239 611 m | 0.0 m | 1 444.6 m | 0.19 | 0.19 | 0.02 | 0.02 |
| 15 → 16 | 28 479 222 m | 28 479 222 m | 0.0 m | 2 889.2 m | 0.09 | 0.09 | 0.01 | 0.01 |
| 16 → 17 | 56 958 444 m | 56 958 444 m | 0.0 m | 5 778.4 m | 0.14 | 0.14 | 0.02 | 0.02 |
| 17 → 18 | 113 916 888 m | 113 916 888 m | 0.0 m | 11 556.7 m | 0.07 | 0.07 | 0.02 | 0.02 |

**ONE rung moves, and it is the swap the owner flies through.** The far level swaps now state their
field step too, and their floors stand far inside their own switch distances, so not one of them
moves a metre — which the client's own test asserts as a list of exactly one rung.

★ **AND ONE THING THE PICTURES SHOW THAT IS NOT A RUNG BOUNDARY, NAMED SO IT IS NOT MISTAKEN FOR
ONE.** At the 150 km stand the sea's shore runs as a STAIRCASE of axis-aligned steps in the middle
distance. It is the SAME staircase before and after, pixel for pixel, at every rung: it is the
coast mask's own footprint on the macro lattice, which ruling W16 already left standing (*"the
lake's EDGE is a staircase on the macro lattice — ruling W6 owes a lake's shore the same law it
gave the sea's"*), and the sea's shore wants that law too. It is not a seam between two rungs: the
drawn edge moves a median 52.6 m at the 9 → 10 swap and 0.0 m at every swap above it.

★ **WHAT IT COSTS, MEASURED ON THE OWNER'S OWN STAND.** Rung 9's ring grows from 445 km to 704 km,
so the crossfade band and the TILES follow it: the asked edge goes from 573 km to 906 km, 2.5 times
the area wherever the eye stands high enough for the ask to bind rather than the horizon. On the
20 km oblique stand, the same pose before and after: the tiles the pilot holds **51 → 91**, the
chunks drawn **7 304 → 8 926** (rung 9's own ring **946 → 2 738**, rung 10's **827 → 657**), the
vertices **24.7 M → 32.1 M**, and the frame **23.16 ms → 25.12 ms** (the peak 26.18 → 25.98).
Eight per cent of a frame for a rung boundary the pilot flies through.

### W17.5 THE PINS — NOT ONE MOVED, AND IT IS A MEASUREMENT

The cure moved the LADDER and the shard's TILE REACH. It moved no line of the recipe and no byte of
the solve: `field_step_m` and `switch_floor_m` only READ the body's charter and the lattice, and
`tile_reach_m` only asks farther. So no pin is owed — and the pins were run rather than argued:

| pin | before | after |
|---|---|---|
| `GENERATOR_VERSION` (`vd-terrain/src/tag.rs`) | 19 | **19, unchanged** |
| `DECLARED_PIN` (`tag.rs`) | 11 554 407 130 392 614 244 | **unchanged** |
| `ARTIFACT_VERSION` (`artifact.rs`) | 7 | **7, unchanged** |
| `HOME_ARTIFACT_DIGEST` (`vd-bins/tests/home_artifact_pin.rs`) | committed words | **unchanged** |
| `PROTO_MINOR` (`vd-wire/src/version.rs`) | 34 | **34, unchanged** — no new word crosses any wire |

### W17.6 THE PICTURES: THE SAME SIX OBLIQUE STANDS, BEFORE AND AFTER

The stands are the owner's own framing — the ground running from under the nose out to the limb,
the horizon **0.341 of the frame down from its top** — over the belt, the nose aimed along the
bearing to the nearest sea (322 km away), under a star **25.7°** up. The poses and the look-at
targets are `vd-bins/examples/oblique_stand`'s own; the two sets differ in the ladder and in
nothing else.

| stand | before | after |
|---|---|---|
| 20 km | `runs/1790164854__pilot__a0/shots/oblique_20km_before.png` | `runs/1790174782__pilot__a0/shots/oblique_20km_after.png` |
| 60 km | `runs/1790164930__pilot__a0/shots/oblique_60km_before.png` | `runs/1790174858__pilot__a0/shots/oblique_60km_after.png` |
| 150 km | `runs/1790165006__pilot__a0/shots/oblique_150km_before.png` | `runs/1790175008__pilot__a0/shots/oblique_150km_after.png` |
| 400 km | `runs/1790166125__pilot__a0/shots/oblique_400km_before.png` | `runs/1790175075__pilot__a0/shots/oblique_400km_after.png` |
| 1 000 km | `runs/1790172059__pilot__a0/shots/oblique_1000km_before.png` | `runs/1790175150__pilot__a0/shots/oblique_1000km_after.png` |
| 2 500 km | `runs/1790172138__pilot__a0/shots/oblique_2500km_before.png` | `runs/1790175226__pilot__a0/shots/oblique_2500km_after.png` |

★ **WHAT THE TWO SETS DIFFER BY, ROW BY ROW** (`scratchpad/frame_diff.py`: the same stand's two
frames, the share of pixels that differ at all and the rows whose own median luminance moved by
over one per cent, with those rows' eye distances):

| stand | pixels that differ | rows that moved over 1 % | their eye distances | the worst row move |
|---|---|---|---|---|
| 20 km | 0.02 % | 0 | — | 0.00 % |
| 60 km | 0.09 % | 0 | — | 0.11 % |
| 150 km | 0.40 % | 0 | — | 0.10 % |
| **400 km** | **3.53 %** | **18** | **556 677 … 572 645 m** | **38.44 %** |
| 1 000 km | 0.00 % | 0 | — | 0.06 % |
| 2 500 km | 0.00 % | 0 | — | 0.05 % |

The one frame that really moved is the 400 km stand's NEAR FIELD, and it moved the way the law
asks: the ground 557 to 573 km from the eye used to be drawn at rung 10 off pyramid level 1 — a
smooth blob with no relief and a soft coast — and is now drawn at rung 9 off the ROWS, with the
recipe's fine octaves and the solve's own shoreline. **A finer rung reaches further, which is what
"a finer rung refines a coarser one" means.** Everywhere else the two frames are the same picture
to within a few hundredths of a per cent: the cure did not repaint the world, it moved ONE ring.

### W17.7 WHAT THE PIXEL GATE COULD AND COULD NOT JUDGE, WITH ITS NUMBERS

`ring_step`'s `rows` mode now takes the stand's own ALTITUDE and PITCH, turns every row band into
the EYE DISTANCE of the ground drawn there and the RUNG the ladder draws at that distance, and
judges a boundary against ITS OWN NEIGHBOURHOOD — a line fitted to the rows outside the crossfade
band on each side, both extrapolated to the band's middle row, the GAP between them the ring's own
step with the air's gradient removed. **A ring is a gap; haze is a slope, and a slope leaves no
gap.** It had to be built this way: at the 20 km stand the worst BAND step read **6.3 %** two bands
under the horizon, at no rung boundary at all, because a band there spans a hundred kilometres of
ground and the aerial perspective alone changes that much.

* **THE LAND'S OWN PROFILE HOLDS NO KINK.** At the 20 km stand the median row luminance falls
  smoothly from **123.5 to 94.7** over 460 rows and every second difference stands under **0.1 %**
  of the local mean — the air's own gradient and nothing else, at the 6 → 7, 7 → 8 and 8 → 9
  boundaries alike. So the colour ring ruling W16 fault D cured has not come back.
* **THE BOUNDARY READINGS DID NOT MOVE, AND THAT IS THE CONTROL.** The boundaries these six frames
  hold are 6 → 7 and 7 → 8 (20 km), 8 → 9 (60 km), 11 → 12 (1 000 km) and 12 → 13 (2 500 km) —
  **not one of them is the boundary the cure moved**, and every gap reads the SAME before and
  after to three decimals (3.840 / 1.650 / 6.483 / 68.015 / 88.862 %). A measurement that could
  have changed and did not.
* ★ **AND THE BIG GAPS MEASURE A COASTLINE, NOT A LADDER.** The sea is a second material, and the
  instrument prints the WATER'S SHARE on each side of every band beside the gap: 91.7 % against
  79.0 % at the 1 000 km reading, 67.0 % against 53.8 % at 2 500 km. Where those two disagree the
  gap is a shoreline. At the 150 km and 400 km stands the bay fills the band outright and the
  instrument reports **0 judgeable boundaries** rather than a number, which is the honest answer.
  WHAT IS OWED: a stand aimed INLAND so the whole frame is one material (`VD_STAND_AWAY=1` prints
  its poses), and a water classifier that reads the haze out — the haze tints far ground blue, so
  the present one calls 97 % of a plainly sandy 60 km frame "water".

### W17.8 THE MOVING EYE: THE SAME GROUND FROM TWO STANDS 40 km APART

The framing is identical at both stands, so a RUNG BOUNDARY — which stands at a fixed EYE DISTANCE
— keeps its SCREEN ROW while the ground slides under it, and the LANDSCAPE does the opposite. Two
stands of the 150 km sequence, 40 km apart along the bearing, read row by row
(`scratchpad/read_seq.py`; the frames `runs/1790175302__pilot__a0/shots/seq0_after.png` and
`runs/1790175382__pilot__a1/shots/seq1_after.png`):

* **Every row of the frame changes by under 1.5 % when the eye moves 40 km, except rows 436 to 484**,
  which change by **36.9 to 37.3 %** — the eye distances 353 to 389 km, where the bay's near
  SHORELINE crosses the frame in one stand and not in the other. That is the ground sliding with the
  eye, which is what ground must do.
* **Not one step stands still at a screen row.** Outside the shoreline the two profiles agree to
  about a tenth of a per cent from row 256 (1 008 km of ground) to row 712 (235 km), which is the
  air's own smooth gradient and no ring at all.
* The 9 → 10 boundary now stands at 704 km, which is row ~290 in this framing; the profile there
  reads 90.79 against 90.79 — **0.00 %**.

### W17.9 THE GATES, EACH WITH ITS NUMBER

* `rung_swap` (new, `vd-bins/examples`) — THE SHAPE: every rung pair of the home planet now stands
  inside the ladder's own tolerance at the distance it really hands over. The 9 → 10 swap's worst
  step falls from **2.82 px to 1.78 px** and its p99 from **0.86 px to 0.55 px**; **0 of 9 600
  directions** stand farther apart than the coarser rung's own sink at any pair. THE SHADE: the
  mean Lambert over the same ground differs by **0.13 % at 9 → 10** and by at most **1.95 %** at
  any pair — under the eye's 2 % threshold, unchanged by the cure (it moves no normal). THE WATER:
  **0 columns of 9 600 flip side at 9 → 10**, unchanged.
* `water_edge_step sea` over the belt's coast, 60 lines of 400 km at rungs 8 to 13 — **43 250 of
  93 720 samples under their own water at EVERY rung**, the edge's median step 52.6 m at 9 → 10
  (p90 197.6 m, max 233.4 m) and **0.0 m at every swap above it**, 0 water flips.
* `ring_sink` — the residual at a rung's fade-in edge is **0 m and the annulus 0 m at every rung 1
  to 18**, read now through the ladder's own bound with the body and the pyramid in it (ruling
  W16's cure stands under W17's).
* `ring_step` on six OBLIQUE stands, before and after — the boundaries those frames hold read the
  SAME gap to three decimals (3.840 / 1.650 / 6.483 / 68.015 / 88.862 %), because not one of them
  is the boundary the cure moved. §W17.7 states what those numbers can and cannot judge.
* `frame_diff` on the same six stands — **0.02 / 0.09 / 0.40 / 3.53 / 0.00 / 0.00 %** of pixels
  differ; only the 400 km stand's near field moved a row by over one per cent, and it moved to the
  ROWS' own ground.
* `cargo test --release -p vd-terrain --test terrain_pin --test mesh_pin` — **2 of 2** and **3 of
  3**: the SEED-ONLY chunk and mesh tables did not move.
* `cargo test --release -p vd-bins --test home_artifact_pin` — **1 of 1 in 132.6 s**: the home
  planet's artifact digests to its ALREADY committed words. **No pin moved**, which is the proof
  that `switch_floor_m` moving into `vd_terrain::artifact` changed no byte of the solve.
* `cargo test --lib`: vd-terrain **207**, vd-client **329**, vd-connection-plane **264**, vd-bins
  **82**, vd-wire **104**, vd-recipe **73** — all green. vd-sim is untouched and green in debug;
  under `--release` two of its `#[should_panic]` tests read green-less because `debug_assert` is
  off there, which is a property of the profile and not of this slice.
* `cargo clippy --all-targets -- -D warnings` on vd-terrain, vd-client, vd-connection-plane,
  vd-client-render, and on vd-bins with `--features dev-control,render` — clean;
  `cargo fmt --all --check` — clean.
* HR5 coverage of the new lines: **UNMEASURED** (`coverage-fast` and `llvm-cov` were not run, by
  the slice's own instruction). The new lines are `artifact::{switch_floor_m, tile_rung,
  field_step_m}`, `tile_reach_m`'s floor, `ladder_view::{handover_step_m, switch_floor_m,
  AskBound::for_body}` and the two new tests.
* `just gpu-drift` — **5 979 of the seam stand's 5 979 boxes byte for byte through the card's own
  gears in 17.1 s, 4 of 4 tests**, adapter Apple M4 Pro (Metal). The card holds no artifact and
  reads no ladder, so its bytes could not move — and they did not.
* THE MOVING EYE (§W17.8) — two stands 40 km apart read the same profile to about a tenth of a per
  cent at every row but the shoreline's, which moves 37 % because the ground moved.

### W17.10 WHAT IS OWED

1. **The owner's look**, on the 150 km and 400 km oblique stands — the framing he flies — before
   and after.
2. **A stand aimed INLAND** so the whole frame is ONE material and the pixel gate can judge the
   9 → 10 boundary itself. `oblique_stand` prints those poses under `VD_STAND_AWAY=1`; they were
   computed and not yet flown.
3. **A water classifier that reads the haze out**: the present one calls a far, hazy, plainly sandy
   row "water" because the air tints its blue channel over its red one.
4. **THE SEA'S SHORE IS A STAIRCASE ON THE MACRO LATTICE**, before and after alike, pixel for
   pixel — the same defect ruling W16 left standing for a LAKE'S edge. It is not a rung boundary
   (the drawn edge moves a median 52.6 m at the 9 → 10 swap and 0.0 m above it), and ruling W6 owes
   the sea's shore the same law it owes the lake's. The cure makes it show FURTHER OUT, because the
   rows now reach further.
5. **HR5 coverage** of this slice's new lines, UNMEASURED here.
6. **The step's own statistic is a stated choice.** Rule 3 is sized on the WORST direction of the
   globe (2.82 px), because the ladder's own pass line is stated as a max. The p99 stood inside the
   line before the cure (0.86 px), so a reader who thinks the p99 is the right judge would say this
   handover never needed moving. The owner's word on which statistic the ladder owes its tolerance
   to is owed.

## W18. THE WATER IS DRAWN ON THE GROUND'S OWN LATTICE (2026-09-23; the owner, after W17, with six pictures: *"Not fixed"*, *"It changes still when I'm leaving farer"*)

### W18.1 WHAT THE OWNER SAW, READ OFF HIS OWN PICTURES

Six screenshots from a climb over the berth, each with its HUD line. The fault sits in the COARSER
rung of each pair, on the far side of the disc, and changes shape with the rung:

| altitude | rungs drawn | the far coast |
|---|---|---|
| 6 374 km | 13..15 | speckled at the right limb |
| 11 497 km | 14..15 | speckled at the lower left |
| 16 952 km | 15..16 | a staircase of ~400 km teeth at the lower left |
| 52 000 km and up | 16..18 | smooth to the eye |

The same three stands flown HEADLESS, still, facing the planet's centre (`owner_stands.sh`,
`runs/1790179570__pilot__a1`, `runs/1790179650__pilot__a2`, `runs/1790180358__pilot__a3`), draw the
speckle and the teeth exactly — so the fault is a STANDING one, not a flicker of the ring, and it
does not need the eye to move. The bounded ask was NOT binding on the owner's climb: the builders
read 375 chunks a second and every stand settled with zero pending, so the rings were whole.

Two ablations on the 16 952 km stand: the water sheet off (`VD_TERRAIN_WATER=0`) drew a whole
planet of smooth land with NO teeth; hiding rung 15 (`VD_TERRAIN_HIDE_RUNG=15`) drew a uniform
blue ball — the hide flag hides a chunk's ground and the sheet rides as its child, so that picture
says nothing about the cause and is recorded only so nobody flies it again.

### W18.2 THE CAUSE, MEASURED

`vd-bins/examples/sheet_poke` rebuilds every chunk the ladder wants at a stated eye, extracts its
ground mesh, and for every ground vertex whose four group columns are all WET compares the
vertex's radius with the radius of the sheet's quad (the bilinear of the four corner columns at the
water's radius, which is the per-cell quad the client drew) at the vertex's own fraction. A vertex
ABOVE that quad is sea floor drawn through the water.

| rung | sea-floor vertices above the sheet | worst excess |
|---|---|---|
| 13 | 0.4 % | 18 m |
| 14 | 8.7 % | 18 m |
| 15 | 8.1 – 13.7 % | 137 m |
| 16 | 3.3 – 5.4 % | 18 m |
| 17 | 5.1 – 5.5 % | 1 958 m |
| 18 | 7.4 % | 3 765 m |

(the five eyes: 6 374, 11 497, 16 952, 52 158 and 84 469 km over the berth; the per-cell quad is
a LOWER bound on the fault, because W16's eight-cell block dips further still.)

The two surfaces stood on TWO LATTICES. The sheet was a quad per cell (and per block, W16) on the
CORNER COLUMNS' directions; the ground is a surface-nets mesh whose vertices lie INSIDE the cells,
off the column directions. A flat quad is a chord of the water's sphere and dips `w² / 4R` under it
at its middle — 10 m at rung 14, 167 m at rung 16, 2.7 km at rung 18 — and a sea floor a few metres
under the water stood OVER the dipped quad wherever the vertex fell near the quad's middle. Each
rung's quads dip differently, so the same shelf drew as sea at one rung and as speckle or teeth at
the next: the coast changed with the rung, which is what the owner has reported three times.

### W18.3 THE LAW AND THE BUILD

**The water is drawn on the ground's own lattice.** `vd_terrain::position::water_sheet` now takes
the chunk's ground mesh: every ground triangle that reaches down to the water (some vertex no
higher than the water plus the hide bound) gets ONE water triangle whose three points are the
ground vertices' OWN positions scaled to the water's radius, at the highest level among the three
vertices (a vertex's level: the highest water word among its group's four columns, or the body's
sea). The two surfaces share every vertex direction, so wherever the ground stands under the water
the water point is farther from the centre — at the vertices exactly, and along a triangle by the
same convex sum of the same three directions — and the ground can cross the water only where it
stands over it, which is the shore: a line inside a triangle, the ground's own crossing of one
surface, carried by the morph. No chord is measured because there are not two of them; no rung is
named. W16's eight-cell block is retired with the quad; the hide bound (2026-09-21) stays. The
cost is stated: a wet chunk's sheet holds as many triangles as its ground.

The statement that could fail is `position::water_sheet_tests::the_ground_never_stands_over_the_water_it_is_under`:
on a real box at rung 13, every water triangle's middle stands over the ground's middle wherever
the three ground vertices are wet, and the same box read by the RETIRED rule puts wet vertices
over their quad. Tier-A tests green in `vd-terrain` and `vd-client`; clippy clean; the sheet is
client-only geometry (the shard never builds one), so no pin and no drift gate moves.

### W18.4 MEASURED AFTER, ON THE SAME THREE STANDS

The three stands flown again on the new build (`runs/1790181790__pilot__a1`,
`runs/1790181870__pilot__a2`, `runs/1790182638__pilot__a3`): at 16 952 km the staircase of teeth
and the speckle at the lower left are GONE and the rung-16 coast matches the rung-15 coast; at
11 497 km and 6 376 km the wide speckle is gone and ONE small speckle patch remains on one coast at
the lower left. That patch is a TIE: the shore law (W6) holds a wet column whose fine octaves would
rise above the water EXACTLY at the water, so over such ground the sheet's triangle and the
ground's triangle are one surface, and the depth test flips per pixel with the single-precision
rounding of each vertex about the chunk's origin.

### W18.5 THE NEAR-TIE: A DEPTH TIE-BREAK TRIED AND REFUTED, THEN THE BUFFER'S OWN STEP

**Tried first:** the water's clip depth scaled by one part in 2²⁰ in the fade shader, so that at a
tie the water is drawn. MEASURED on the same stands (`runs/1790182735__pilot__a1`,
`runs/1790182815__pilot__a3`): at 11 497 km the patch did not change; at 6 376 km the far coast
dissolved MORE — a millionth of the distance is eleven metres along the ray at 11 500 km, and at a
grazing view that lifts the water over every coastal plain lower than that. The owner saw it on the
pictures: *"some land dissolves (feels like it becomes transparent)"*. Removed; the record stays
on `LadderFade::splat`.

**Tried second, refuted before a flight:** a nudge of the water by the vertex buffer's own
single-precision step (six centimetres a million metres from the chunk's origin). The probe below
showed the gap a thousand times wider than that step, so it was removed unflown.

**What the patch is, MEASURED** (`vd-bins/examples/mask_at_pixel`, the pixel of the patch on the
11 497 km stand unprojected to the sea): in a 200 km window around it the coast mask says SEA at
every level but one ridge two nodes wide, EVERY column stands under its water by the shore law
(0 of 2 123 columns over their water at rung 13, 0 of 588 at rung 14), and yet the DRAWN ground
stands more than a metre OVER the water at 1 823 of 2 178 vertices at rung 13 and 515 of 587 at
rung 14 — none within a metre either way. The extractor places a vertex at its gap bytes' zero
crossing, and a gap byte is 1/128 of a cell: 64 m at rung 13, 128 m at rung 14, 2 km at rung 18.
The shore law holds a shelf's floor under the water by a quarter of its depth — metres — so near
a shore the column's depth is SMALLER than the step the drawn ground can show, and the drawn floor
lands on either side of the water. The quad per cell hid this before W18 because it dipped under
the water by metres; the sheet on the ground's own lattice stands exactly at the water and exposes
it. Nothing about the world is wrong here: the drawn mesh cannot show a depth under its own step.

**The cure (built):** the sheet stands at the drawn ground's own resolution, on the columns' side.
A vertex's side is its COLUMNS' word — the shore law's surfaces against their own water, never the
drawn radius: UNDER where all four columns of its group stand under, OVER where all four stand
over, MIXED at the shore — and the water point is held one extractor's step (`cell / 128`) off the
drawn ground on that side wherever the water's own radius would put it nearer: at least `r + q`
over a wet vertex, at most `r − q` under a dry one, at the water at a mixed vertex, whose triangle
is the shore's own (`position::water_sheet`). The water bulges over a shallow floor by at most one
gap step, under what the rung can draw. The step is the extractor's, the side the recipe's, and
nothing is drawn. MEASURED after (`runs/1790184682__pilot__a1`, `runs/1790184763__pilot__a3`): at
11 497 km the patch is gone and ONE dot stands where the mask states its ridge of land two nodes
wide, which is geography; at 6 376 km the coast that dissolved is solid land and the far shore is
one line. Tests green in `vd-terrain` (the four sheet tests, one of which holds the retired quad's
own failure beside the law) and `vd-client`; clippy clean on the four crates.

### W18.7 THE WATER MORPHS WITH ITS GROUND (the owner's three far pictures, 20:10–20:13: *"the
land disappears and appears in the same shape again after the rung is changing"* at 118 000 km and
68 000 km; *"the whole planet start to fade in and out, almost like very light blinking"* at
408 000 km; *"for the rest of the rungs and distances — all works amazing, so please don't break
it"*)

118 000 km is rung 17's own switch distance (a 131 km cell over one pixel), the middle of its
crossfade band. In the band every ground vertex morphs along its radial toward the COARSER rung's
mesh, and that mesh stands at ITS extractor's step — 2 km at rung 18, 8 km at rung 20 — so a
continent's vertices slide down by kilometres as the eye moves out, while the water sheet carried a
morph of zero and stayed at the sea. The ground sank under a water that stood still: a regular dot
screen over the whole continent (the coarser mesh's own lattice), gone at the ring swap when the
coarser rung's own sheet took over with its own side rule — *"the same shape again"*. At 408 000 km
the same band between rungs 18 and 20 moved the whole planet's ground by rung 20's 8 km step under
a still water: the shimmer.

THE LAW: **the water's morph target is the same rule read from the ground's morph target.** The
sheet builder returns each water vertex's source (`WaterSource`: its ground vertex, its side, its
level), and the client sets the water vertex's morph metre to `water_radius(side, level, ground's
moved radius, q) − own radius` — the ONE function that placed the water at build time
(`position::water_radius`), so in every crossfade the water rides with the land, one step off it on
the columns' side, and stays at the sea over deep floor. The top rung morphs to itself, so its
water's morph is zero, as before. The second picture's staircase on the far coast at 68 000 km is
rung 18 drawn at its own 262 km cells under the bounded ask at 1 600 km/s (ruling F9: the coarser
rung stands whole where the builders cannot deliver the finer), not a sheet fault; it is left as
is. MEASURED after: owed below.

### W18.6 WHAT IS OWED

1. **The owner's look**: the three stands after, and a climb.
2. **The frame cost** of the sheet on the ground's triangles, measured on the coast leg against the
   block's number (the block was W16's saving; this ruling gives it back).
3. **A water skirt**: the ground hides its seam cracks with skirts and the sheet has none; a
   one-pixel crack at a chunk seam over deep water is possible and UNMEASURED.
4. The W17.10 items stand.
