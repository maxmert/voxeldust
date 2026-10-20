# A lake's shore and floor by their laws — a short design (ruling B119; the owner, 2026-10-19: "Shores by law first")

Binding context read first: B119 (the shores by law first), B118 (the drowned basin, the game's rule until this lands),
B115 (a lake stands at its 128 m spill; the lakes together), B105 Q1 (b) (a standing water's points under its level are
outlets), B84 (each 128 m node's mean held to its 8 km row), B23 / B38 (no caps, no knobs, published laws, each with its
calibration body), T9 (no physical number drawn), B41 (a law reads the body's own facts; an Earth place only fixes a
constant it cannot compute yet), B93 item 2 (judged from the side), W16 (one shape at every distance), SL6, SL10.

## 0. For the owner

### 0.1 What the research found (two agents, primary sources checked where marked)

1. **Every landscape model that keeps lakes uses the lake's surface as a base level** — Powell (1875, pp. 203–204), Davis
   (1899, 1902): rivers grade to the lake's surface, nothing erodes under it, and the sediment that reaches the lake stays in
   it until the lake is full. Our B118 rule erodes the basin to its deepest node and then floods it; that is the rule of
   Cordonnier et al. 2016, whose paper says it neglects deposition. Its result is a YOUNG RESERVOIR.
2. **A young lake and a mature lake differ in plan** (Johnson 1919; Palermo et al. 2024, *Sci. Adv.* 10:eadn4192): a flooded
   river landscape gives drowned valleys (rias) — Lake Murray, Lake Lanier, Fort Peck classify as "incised then flooded";
   waves then cut the headlands back and close the bays, so a mature lake's exposed shore is "simply curved or nearly
   straight" (Sebago Lake, Lake Rotoehu classify as wave-eroded), while sheltered bays keep their arms.
3. **Sediment fills every lake in time** (Einsele & Hinderer 1997; Brune 1953): a lake survives only while it is younger than
   its fill time, or while evaporation keeps its basin closed. Under the 8 km world's own history this is ALREADY counted:
   its solve keeps a closed basin's rock, filling from its floor (B51), so an 8 km lake's rows stand at the floor after its
   infill.
4. **The floor's shape**: sediment focuses into the deep part (Lehman 1975; Hilton 1985) — a flat floor below the wave base
   (half the wave length, `g·T²/(4π)`), a ring of erosion and transport above it, deltas at the river mouths (Gilbert 1885:
   topset at the lake's level, foreset at the underwater angle of repose, 25–35° sand and gravel, about 10° clay; Adams et al.
   2001).
5. **The shoreface**: the equilibrium profile `h = A·y^(2/3)` (Bruun 1954; Dean 1977, 1991) down to the depth of closure
   `h_c = 2.28·H_e − 68.5·H_e²/(g·T_e²)` (Hallermeier 1981), with `A = 2.25·(w²/g)^(1/3)` (Kriebel, Kraus & Larson 1991; `w`
   the grain's settling velocity by Ferguson & Church 2004, which reads the body's gravity and the fluid), the waves from the
   wind and the fetch (CEM 2003: `gH/u*² = 0.0413·(gX/u*²)^(1/2)`, `gT_p/u* = 0.651·(gX/u*²)^(1/3)`). These are EQUILIBRIUM
   forms: they need no time.
6. **What needs time**: a delta's growth, a headland's retreat (Huppert, Perron & Ashton 2020: cliff retreat scales with
   the mean wave power) and the plan's smoothing (Pelnard-Considère 1956; Ashton & Murray 2006). Our rows hold no lake age.
7. **Licenses**: fastscape, goSPL, Veloren are GPL — the rules are written again from the papers; Fill–Spill–Merge and
   Landlab are MIT.

### 0.2 The proposal — three parts, the first two need no age

**PART 1 — THE RIVERS END AT THE LAKE (Powell's base level).** Today (B118) the land is eroded to the lake's deepest node
and its valleys run on under the water. The new rule: the tile is solved as today (the drowned basin gives the shape the
basin had when the water rose), and then the land OVER the lake's level is solved again with the shoreline as its outlet at
the lake's level — so each river ends at the water's surface, and nothing is cut under it.
- Example: the river that reaches the basin lake from the belt ends at −2 002 m, at the shore; its valley under the water is
  no longer a canyon that runs on to the lake's deepest node.

**PART 2 — THE FLOOR BY SEDIMENT FOCUSING, THE SHORE BY THE WAVES (equilibrium forms, no age).**
- *The deep floor*: below the wave base the drowned land's own volume is spread flat (the focusing moves sediment from the
  slopes to the deep part; no rock is added, the node means held as B84 holds them) — a flat floor at one level per lake.
- *The shoreface*: from the shoreline to the depth of closure the floor follows `h = A·y^(2/3)`, cut into a headland and
  filled in a bay, with the wave height from the climate's yearly wind at the lake and the lake's fetch along that wind.
  A small pond (a fetch of 100 m: waves of 5–8 cm) gets no shoreface at the grid's size — its shore stays the graded land's.
- Example (the CEM's fetch-limited law, Earth's gravity for the estimate): the basin lake under a 7 m/s wind with a 60 km
  fetch: waves about 0.83 m with a period of 3.5 s, a closure depth of about 1.5 m, a shoreface about 58 m wide on its
  downwind shore (sand of 0.2 mm) — four 16 m cells; on the 128 m level less than a cell.

**PART 3 — DELTAS AND THE PLAN'S SMOOTHING (need a lake's age).** A delta at each river mouth (topset at the lake's level,
foreset at the angle of repose of the river's grain, Kenyon & Turcotte 1985 / Swenson et al. 2000 for its reach from the
supply and the time), and headlands cut back by the wave power over the lake's age (Huppert et al. 2020). The age is not in
the rows today (§0.3 Q1).

### 0.3 The questions

- **Q1. A LAKE'S AGE** (for Part 3):
  - (a) **recommended: Parts 1–2 now, Part 3 at the next generator change** — the 8 km solve records, per lake node, the
    pass at which its water last rose (one new byte a node in the artifact, a world move; it joins the changes already
    queued for the next generator, B113 / B114);
  - (b) Parts 1–2 only, Part 3 registered without a date;
  - (c) all three now with an age from the landscape's time (B45, per plate) — refused by me: it is the time since the
    ground was last RAISED, not since the lake formed, and it would fill nearly every lake (fill times of thousands of years
    against hundreds of millions).
- **Q2. THE GRAIN SIZE** (for the shoreface's `A` and the foreset's angle): a grain size is a fact the rows do not hold.
  - (a) **recommended: by the river's competence at the mouth** — the largest grain the river can move there by the Shields
    threshold (Shields 1936: `D = τ/((ρ_s − ρ)·g·θ_c)`, `τ = ρ·g·h·S` from the river's depth and slope at the mouth, which
    the 128 m solve holds), so a steep mountain river brings gravel and a lowland river sand; a lake with no river reads the
    sand of its nearest inflow; the body's gravity and fluid enter by law;
  - (b) one sand of 0.2 mm for every lake — refused by me: a drawn number (T9).
- **Q3. THE LEVEL THAT CARRIES THE SHOREFACE**: at 128 m the shoreface is less than a cell for fetches under about 50 km, at
  16 m it shows for fetches over about 10 km. Recommended: Parts 1–2 at the 128 m level (the base level, the focused floor,
  the shoreface of the large lakes) and the shoreface again at the 16 m level by the same law.

## 1. Prices (estimates)

| Part | Where | Price |
|---|---|---|
| 1 the base level | the shard's 128 m solve, lake tiles only | a second land solve per lake tile (1 959 of the home planet's 138 624 tiles; about +4 s each); a world move of the fine levels (FINE_VERSION 12) |
| 2 the floor and the shoreface | the same solve, after Part 1 | a flood from the shoreline (the distance `y`), the fetch along the wind (one ray walk per shore cell), the focused level per lake: small next to a solve |
| 2 at 16 m | the 16 m solve | the same law on its finer cells |
| 3 deltas and smoothing | the 8 km solve + the 128 m solve | a generator change (the lake's age byte, every first solve again), then a delta per river mouth |
| the underwater drawing (B119 item 2) | the client | after this, no world change |

## 2. The checks

| Check | Reads | Failing control |
|---|---|---|
| G-GRADE | no cell under a lake's level cut by a river (the land's lowest point at each shore at the lake's level) | B118 (valleys run on under the water) |
| G-FLOOR | below the wave base, the floor's spread around its focused level | B118's drowned valleys |
| G-SHOREFACE | the profile's misfit to `A·y^(2/3)` from the shoreline to the closure depth on the exposed shores | B118 |
| G-MASS | each lake node's mean held (B84) | — |
| Side views | the basin lake and a mountain lake from 1.5 km and 14 km, beside B118, the owner's look (B93 item 2) | — |

## 3. Sources (checked by the research agents; see their reports in the session's record)

Powell 1875 (*Exploration of the Colorado River of the West*, pp. 203–204); Davis 1899 (*Geogr. J.* 14:481–504) and 1902
(*J. Geol.* 10:77–111); Johnson 1919 (*Shore Processes and Shoreline Development*, pp. 339–341); Palermo et al. 2024 (*Sci.
Adv.* 10:eadn4192) and NEWTS (*GMD* 17:3433–3445); Einsele & Hinderer 1997 (*Geol. Rundsch.* 86:288–310); Brune 1953 (*Trans.
AGU* 34:407–418); Lehman 1975 (*Quat. Res.* 5:541–550); Hilton 1985 (*L&O* 30:1131–1143); Adams, Schlager & Anselmetti 2001
(*Sedimentology* 48:661–679); Gilbert 1885 (USGS 5th Annual Report); Kenyon & Turcotte 1985 (*GSA Bull.* 96:1457–1465);
Swenson et al. 2000 (*EJAM* 11:433–452); Bruun 1954; Dean 1977, 1991 (*J. Coastal Res.* 7:53–84); Kriebel, Kraus & Larson 1991;
Ferguson & Church 2004 (*JSR* 74:933–937); Hallermeier 1981 (*Coastal Eng.* 4:253–277); Birkemeier 1985; USACE Coastal
Engineering Manual (2002–2003), Parts II-2 and III-3; Huppert, Perron & Ashton 2020 (*Geology* 48:499–503); Pelnard-Considère
1956; Ashton & Murray 2006 (*JGR-ES* 111); Shields 1936; Cordonnier et al. 2016 (*CGF*); Barnes, Callaghan & Wickert 2021
(*ESurf* 9:105–125); Hayes et al. 2013 (*Icarus* 225:403–412) for waves on other bodies.

## 4. The bench of Parts 1–2 (2026-10-19, after ruling B120)

MEASURED on the basin lake's 3 × 3 ring (`planet_stand water-game … 3,96,22 graded joint`; bench rule `OutletRule::Graded`):
1. **PART 1 ALONE CHANGES ALMOST NOTHING OVER THE WATER.** The drowned basin's shoreline points already stand at the lake's
   level, so its rivers already end there; solved again with the water's surface as the base, the land over the water is
   nearly the same (side views from 14 km). Part 1's real content is UNDER the water: no canyon there.
2. **THE FILL WITHOUT AN AGE BRINGS BACK THE ROUND LAKE.** Filling the drowned basin from the bottom up to the 8 km world's
   own floor (its smooth surface, the rows after the 8 km infill, B51), never over the lake's level: the canyons fill, but
   the water then stands only where that smooth surface is under the level — its Catmull-Rom contour again, rounded blobs —
   and the drowned arms outside the 8 km lake fill to the level as wide flat plains (the basin lake's water in tile (96, 22):
   152 026 → 100 016 cells). From the side: a lake among flat mud plains.
3. **CONCLUSION**: the forms that need no age do not give a mature lake's plan; how far the deltas fill the arms and how far
   the waves cut the headlands is set by the supply over the lake's AGE (Part 3). Without it, the choice is between the
   young reservoir (B118) and the 8 km floor's round contour (B105 Q1 (b), and this fill). Picture:
   `~/Downloads/voxeldust_lakes_2026-10-19/floors/drowned_vs_graded_fill.png`.

## 5. Part 3 on the bench (2026-10-19/20, after ruling B121)

### 5.1 The lake record, cured

The first generator-28 solve recorded a node as its lake's only while water stood over it. The deposit (B51's fill) fills
a lake node up to the lake's level and passes the rest on, so every node it filled counted as dry and forgot its age and
its rock. MEASURED: 8 094 of the 16 445 records stood under less than 1 m of water, and 91 % of them read 250 kyr and no
rock. The cure: a node is its lake's while it stands AT OR UNDER the level of the water body that covers its depression
(`lakes::Lakes::body_level`). Solved again: the rows the same byte for byte; 23 353 records (6 908 filled up to the level);
laid 0 under 1 m of water 91 % → 28 %.

### 5.2 The model (`l1::OutletRule::Deltas`, bench only)

1. THE BEDROCK (`l1::bedrock_view`): each record's laid rock is taken off its row; a record node that the rock filled up to
   the level gets the word of the lake beside it. The drowned basin (B118) is solved on these rows.
2. THE SHARES: each lake's rock in the tile's domain (laid metres × node area) is shared among the rivers that enter that
   lake, by their water. This is a stated closure (one load of rock per cubic metre of water); the BQART law (Syvitski &
   Milliman 2007) can replace it when the catchments' relief and temperature are read.
3. THE DELTA (Gilbert 1885): from each mouth, the rock's top stands at the lake's level and its front falls at the angle its
   grain rests at. The grain is the river's competence at the mouth (Shields 1936; the critical Shields number of Soulsby &
   Whitehouse 1997; the depth from the channel law, the slope from the river's last link): sand and gravel at 35° (the talus
   law's angle), mud under 63 µm at 10° (Adams et al. 2001). The top's reach along the water is found so the delta holds its
   share. A share too small for the whole front fills the floor at the front's foot first.
4. The walk goes only through the lake's water and over earlier deltas' tops, so a delta never crosses a headland. Rock a
   water has no room for runs on down the drainage to the next water (the 8 km deposit's own rule).

Example: a river that carries a quarter of the deep lake's water in a tile's domain (face 5, tile 2,103) gets a quarter of
the lake's rock there; it builds a plain at the lake's level (−295 m) up its drowned arm, and where it brings sand or
gravel the plain ends in a front at 35°.

### 5.3 Measured (`planet_stand water-game … drowned|deltas joint`, world `home28c.art`; pictures in
`~/Downloads/voxeldust_lakes_2026-10-19/part3/`)

| Place | Lake | Today (B118) | The deltas |
|---|---|---|---|
| face 3, tile 126,12 | old (10.6 Myr), 473 m laid | spiky drowned valleys, the main lake at −2 429 m | smooth shores, delta plains fill the arms; the main lake at −2 214 m (its 8 km word) |
| face 3, tile 95,23 | the basin lake, young (563 kyr), 22 m laid | an archipelago at −1 516 m (616 m under its word) | −955 m (55 m under its word); bays closed by plains |
| face 5, tile 2,103 | deep (954 kyr), 71 m laid | branching drowned valleys | the arms are flat valley floors at the lake's level |
| face 1, tile 40,88 | under 1 m of water on the rows | no water (the rows read the water level equal to the ground) | no water |

- The rock laid equals the records' rock: at tile 126,12 771.690 of 771.691 km³; at the basin lake and the deep lake every
  tile to 0.001 %. Three tiles at the old lake keep 11–125 km³ with no room in their domain.
- THE PRICE: the rock moves between 8 km nodes. At the old lake the record nodes' 128 m means stand a median 253 m under
  their rows and the plains up to 547 m over them (today's rule: ±220 m, 5th–95th percentile). Under the water the eye
  does not see the floor; the far lake list (B116) carries each node's wet share and visible surface from the 128 m tiles,
  so the far rungs show the same shore.
- THE WAVES: their forms need far less time than the youngest lake's age (Pyramid Lake's barriers formed in under seven
  months; a wave-cut bench limits its own width, Anderson, Densmore & Ellis 1999). The age decides the deltas only; the
  waves' shore is an equilibrium form (the Dean shoreface, under one 128 m cell for most lakes).

### 5.4 The owner's look and the second form (2026-10-20)

The owner at the old lake: *"Shores are not believable, but the rest is ok"*. Two causes and their cures:
1. THE 16 m LEVEL (walls, rectangles, specks): a delta's top stood exactly at the lake's level, so the 16 m law's detail put
   half its cells under the water; the 16 m water spread only inside the 128 m band and stopped beside dry ground under the
   same level. Cure (16 m version 5; W6, W16): a 16 m cell whose 128 m sample holds no water is held at the level of the
   water beside it.
2. THE CONES (rows of scallops, circles): a cone about each of thousands of gully mouths. Cure (FINE_VERSION 13): only a
   river at least a cell wide (the width law, the carve's own rule for a drawn channel) builds a delta; the narrower
   streams' rock is the lake's SHOREFACE — the waves spread it along the whole shore (longshore drift, far faster than any
   lake's age) as a band of one reach from every shore point, top at the level, front at the grain's angle.
3. REFUTED on the way: each node's rock over its own shallowest water first. In a drowned valley landscape the shallowest
   water lies over the drowned ridges; the fill made plains in the lakes' middles and kept the branching arms.

Example: the old lake's arms close into flat valley floors and its shore runs as a smooth band of plain between the water
and the mountains; no stream there reaches the width of a 128 m cell, so the lake holds no single delta.
