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
