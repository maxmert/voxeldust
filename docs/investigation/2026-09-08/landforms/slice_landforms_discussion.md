# THE LANDFORM ARC — how a planet gets a face: water, rock, climate and light

Written 2026-09-08 for the owner's discussion, before any code. ASD-STE100, in-game examples, drawings
where they help. **Every recommendation in §9 waits for a YES or a NO.**

**Your words this arc answers.** *"Make sure that we reach that quality on the picture for earth-like
planets (biomes can be different of course, should be dependent on the planet position, spin, trajectory,
size and gravity, etc.). It should be very believable, as we also should simulate the weather."* And, of
our own three pictures: *"no orienters and no details at all … The only thing I'm worried about is that
the surface will not be interesting enough."*

**What this document is.** The single decision document for the whole landform arc. It stands alone. Its
evidence is `00_proposed_landforms.md` (revision 4) in this directory and the eight domain reports
`01`–`08` behind it, each refuted twice.

**The one-line answer to your worry.** Your worry is correct, and it is now MEASURED. **The surface is
flat by arithmetic, not by taste**, and three of the four causes are in the terrain and one is in the
light. This arc cures all four.

**Every number carries a mark.** MEASURED (a program here produced it), REPLAYED (a domain re-ran the
crate's arithmetic outside the crate and reproduced a value the crate already measured), COMPUTED
(arithmetic on a measured number, shown), ESTIMATED (a model, not a result), UNMEASURED (nobody knows;
the measurement that would settle it is named).

---

## 1. What this arc decides, and what it leaves to later slices

**Decides.**
- **The body charter** — a planet's physical facts as quantised integers, authored once and stored: its
  gravity, its spin, its tilt, its water, its air, its age. Your five words become data.
- **The macro layout and the macro solve** — plates, continents, ocean basins and ranges, then one
  erosion solve on one coarse global lattice, ONCE EVER per body, kept as a small artifact.
- **Rivers, lakes and a real coast** — a drainage network that runs downhill everywhere.
- **Climate inside the erosion loop** — insolation, a lapse rate, wind belts from the spin, orographic
  lift, a rain shadow. Rain where rain belongs, so the erosion is not uniform. **And the aridity reaches
  the ROCK's shape, not only its colour.**
- **The slope spectrum** — the relief redistributed into the 1 km–6 km band the eye actually reads.
- **The fine rungs (the fold)** — ridged crests aligned to the flow, benched cliffs, mesas, the channel
  carve, a fine floor. They read the artifact and solve nothing.
- **The ladder's coarsening law** — restated per term, so the far view is the near hill.
- **The biome and the PAINT** — nineteen biomes, and the client's material table that finally makes a
  snow cap white and a desert tan.
- **The light** — a raking star, a cast shadow, and a blue haze from the body's own atmosphere.
- **The live weather** — a derived almanac (seasons, a moving snow line, a river that rises in the melt)
  and a short list of live weather systems the realm ships.

**Leaves to later slices.**
- **The forest itself** — slice 14 places the tree assets. This arc builds the skeleton that says where
  they go.
- **The standing character for scale** — slice 16, and ruling V11 (up is the server's).
- **Fields, roads and walls** — they are built, not seeded. They are live state, like a player's house.
- **Volcanoes, dunes, karst and landslide scars** — named, priced and put to you as L21 and L22.
- **A landscape that changes over game time** — SL10 forbids time inside the recipe.

**Example, in the game's own words.** A pilot approaches the home planet. Its realm holds a charter of
about twenty integers and states it in its own self-look. Its shard solved the planet once, long ago,
and keeps 17.6 MB. The client derives the same bytes from the same seed and the same integers and checks
the digest. The pilot lands in a valley the water cut. He walks up the valley side; the ground under his
boots is the same shape the shard's collider used. The low star throws the ridge's shadow across the
valley floor, and the far range stands behind a blue haze.

---

## 2. The words, explained once

Every term below is the real industry word. The example is in the game's own words.

| Term | What it means, and an example here |
|---|---|
| **Hypsometry** | How much of a surface stands at each height. Earth's is TWO-HUMPED: one hump of continents, one of ocean floor. **Our home planet has one hump, so it has no shelf and no stable coastline.** |
| **Isostasy** | Crust floats on the mantle like a raft: thick, light crust rides high; thin, dense crust rides low. **A continent on our planet must be a raft, not a bump on a noise field.** |
| **Crustal dichotomy** | Two kinds of crust at two densities — light continental, dense oceanic — which is WHY hypsometry is two-humped. |
| **Tectonic uplift** | Plates push together and the crust rises. **The orogen is the belt where our reference picture's range stands.** |
| **Orogeny / orogen** | Mountain building, and the belt it makes. |
| **Craton** | An old, cold, stable continental interior. **A craton in our world must read as a plain, not as a mountain range with the volume turned down.** |
| **Passive margin** | A coast with no plate boundary: a shelf, then a slope, then abyssal plain. **The place a beach can exist.** |
| **Angle of repose** | The steepest angle loose rock holds — about 33°–37° for dry scree. **Our fine roughness is anchored on it, so a player can still walk.** |
| **Talus** | The loose rock that slides to the angle of repose. **The scree cone under every cliff in your reference picture.** |
| **Flow accumulation** | For each point, how much land drains through it. **It is what tells a channel from a hillside.** |
| **D8 routing** | The cheapest drainage rule: each cell sends its water to the steepest of its eight neighbours. **Our receiver byte per macro node IS this.** |
| **Priority flood** | Filling every pit so water can always reach the sea, and RECORDING each filled pit as a lake with a spill level. **This is how our lakes are born; we do not place them.** |
| **Base level** | The height a river erodes down toward — the sea, or a lake's spill level. |
| **Drainage basin** | All the land that drains to one outlet. **The unit our seasonal river stage is computed per.** |
| **Stream power law** | Erosion rate ≈ K · (discharge)^m · (slope)^n. It is the standard river-incision law. We use m = ½, n = 1, because ½ is a square root, which the float fence permits, and n = 1 needs no inner iteration. |
| **Drainage density** | Channel kilometres per square kilometre. Earth: 0.5–5. **Our gate band.** |
| **Hydraulic erosion** | Erosion by water. **The pass that cuts our valleys.** |
| **Thermal erosion / mass wasting** | Rock breaking and sliding without water. **The pass that rounds our crests and builds our scree.** |
| **Isostatic rebound** | Land rises as erosion removes weight, like a raft rising as cargo is unloaded. **Without it, eroded ranges sink and disappear.** |
| **Glacial equilibrium line** | The height above which more snow falls than melts. **Above it there is NO liquid water, so river erosion does not act at all — which is why an ice pass exists.** |
| **Cirque / arête / U-shaped trough** | The three shapes ice makes: a bowl, a knife ridge between two bowls, a valley with a flat floor and steep walls. **The top third of your reference picture.** |
| **Insolation** | The star's power per square metre at the body. **The first number the climate reads, and it comes from the parent system, one hop down.** |
| **Albedo** | The share of light a surface throws back. Snow is high, rock is low. |
| **Equilibrium temperature** | What a body's temperature would be with no air. Air adds the greenhouse on top. |
| **Optical depth** | How much the air blocks. It sets the greenhouse and, on the client, **the blue haze**. |
| **Scale height** | The height over which air thins by a factor of e. **It is the one number the haze needs.** |
| **Lapse rate** | How fast temperature falls with height — about 6.5 °C per kilometre on Earth. **It is why a snow cap sits on a ridge and not in the valley.** |
| **Hadley cell** | The tropical overturning of the air: rising wet air at the equator, sinking dry air at about 30° — which is where Earth's deserts are. **Its WIDTH depends on the SPIN, which is one of your five words.** |
| **Coriolis** | A spinning body bends moving air. A faster spin makes narrower belts and more of them. |
| **Obliquity** | The tilt between the spin axis and the orbit's normal. **It makes seasons.** We carry its COSINE, because `sin` and `cos` are compile-fail inside our float fence. |
| **Tidal locking** | A planet close enough to its star turns once per orbit — one face always lit. **A close-in planet with a nine-hour day is a body a reader refuses.** |
| **Orographic lift** | Air forced up over a range cools and rains. **The wet side.** |
| **Rain shadow** | The dry side behind the range. **A green windward valley beside a tan leeward cliff at the same height is the most striking thing in your reference picture.** |
| **Whittaker chart** | The standard biome chart: temperature on one axis, precipitation on the other, biomes as regions. **Our classifier is this chart plus a short ordered chain of overrides.** |
| **Aerial perspective** | Distant things go pale and blue because air scatters. **It is what separates four depth planes in your reference picture, and no slice owned it until now.** |
| **Cascaded shadow map** | The standard way a game casts one sun's shadow over a big view: several shadow textures at several distances. **Without it, a 300 m spur and a 3 m bump look identical.** |
| **The ladder / rung** | Our detail levels. Rung 0 is 1 m cells; rung 11 on the home planet is 2 048 m cells. |
| **The fold** | Our per-column recipe, coarse to fine: the macro height, then octaves, then the terrace, then the carve. It solves nothing; it reads. |
| **The artifact** | The small file the solve leaves behind: one row per macro node. **17.6 MB on the home planet.** |
| **The charter** | The body's physical facts as integers, authored once and stored. |
| **The almanac** | What the calendar says today: the season's phase, the sub-solar point, the snow line, the river's stage. **Derived from the clock; never shipped.** |

---

## 3. The path from a body's facts to a drawn ridge

```text
  THE PARENT SYSTEM            THE BODY'S REALM (its shard)          BOTH HOSTS, PER COLUMN
  -----------------            ----------------------------          ----------------------
  owns the ORBIT relation
  computes + quantises:
    insolation                                                        [E] THE FOLD
    equilibrium temperature                                            s0 = MACRO + Z
    eccentricity                                                       s  = s + A_k*r*ridged noise
        |                                                              s+ = TERRACE(s)   (beds, dip)
        | ONE HOP DOWN (SL1 cl.4)                                       h  = CARVE(s+)    (channel)
        v                                                              + the 3-D removals (8f)
  [A] THE CHARTER  ~20 integers, ~42-76 bytes                                |
      the forest draws the body-local half                                   |
      the realm HOLDS all of it and STATES it in its own self-look           |
        |                                                                    |
        v                                                                    |
  [B] THE MACRO LAYOUT   closed form, no iteration, ~8 224 m per node         |
      plates -> crust dichotomy -> isostasy -> orogeny (under a strength      |
      ceiling) -> the sea, solved from the WATER INVENTORY                    |
        |                                                                    |
        v                                                                    |
  [C] THE MACRO SOLVE    iterative, single-threaded, a stated pass count      |
                                                                             |
     +--> CLIMATE (nested sub-lattice, inside the schedule)                   |
     |      insolation, lapse rate, Hadley edge from the SPIN,                |
     |      orographic lift, rain shadow  ---> ARIDITY --------------+        |
     |        |                                                      |        |
     |        v                                                      |        |
     |    D8 ROUTING -> PRIORITY FLOOD -> DISCHARGE -> STREAM POWER   |        |
     |        |                                                      |        |
     |        v                                                      |        |
     |    ISOSTATIC REBOUND -> TALUS (angle reads the ARIDITY) <------+        |
     |        -> ICE -> CRATERS (airless) -> COAST                            |
     |        |                                                               |
     +--------+   the LOOP: relief -> wind -> rain -> discharge -> relief      |
              |                                                               |
              v                                                               |
  [D] THE ARTIFACT KEPT, 17.61 MB on the home planet -----------------------> read here
      6 B / macro node: Z | water level | D8 receiver + facies | discharge     (one tile
      2 B / pyramid node, five levels                                           gather per
      8 B / climate node: temperature, precipitation, season, ARIDITY           chunk)
      the rivers and the lakes are DERIVED from these bytes                     |
                                                                               v
                                                                        [H] THE EXTRACTOR
                                                                          mesh + collider
                                                                               |
                                                                               v
                                                                        [I] THE CLIENT
                                                                          draws it, with
                                                                          a PAINT TABLE,
                                                                          a CAST SHADOW
                                                                          and a BLUE HAZE
```

**What crosses a realm boundary, and nothing else.** The charter (one statement, on the realm's own
self-look). The orbit-derived facts (one hop, parent to body). The live weather list (one statement per
realm, ~1 KB on change — never one per observer). **No pose, no velocity, no per-observer field.**

---

## 4. The macro map — where continents, ranges and coasts come from

### 4.1 Why not more noise

Three measurements say the surface is empty, and a fourth says why more noise cannot cure it.

1. **The skyline holds no break the eye can name.** From four ridge stations, marched over 360 bearings
   to 540 km through the crate's own height field, the largest rise over the local trend is
   **0.054°–0.067°** (REPLAYED). **A full moon is 0.52° across.** The tallest feature on this planet's
   horizon is an eighth of a moon's width.
2. **The ground under the boots is a plane.** Inside 50 m, after the tilt is removed, the surface departs
   from a flat tilted plane by **0.26 m rms**; the median slope is **1.36°**, and it is the SAME at a 1 m,
   a 10 m, a 100 m and a 1 000 m baseline (REPLAYED). **A field with one slope at every baseline is
   smooth by definition.**
3. **Nothing exists between 1 m and 48.8 m.** `SHORT_WAVE_M = 30` stops the octave table at the first
   wavelength under 30 m, which on the home planet is 48.83 m (COMPUTED). The cell is 1 m.
4. **More roughness alone is a gravel heap.** Setting the roughness to Earth's own spectral exponent,
   with the relief unchanged, makes **30 % of the planet steeper than 45°** and still buys only 0.8
   skyline breaks per 60° (REPLAYED). **Roughness is not landform.**

**The relief is not the problem.** The home planet carries 14 304.9 m of amplitude and its surface spans
17.0 km (REPLAYED). Earth's dry land spans about 9.3 km (PUBLISHED). **We carry nearly twice Earth's
land relief and show none of it**, because the relief has no structure below the width of a province,
and because a sum of isotropic noise has **no lines** — no ridge crest, no coastline, no scarp, no basin
rim, no valley axis, at any scale.

### 4.2 What we build instead

```text
   L1  PLATES      a spherical Voronoi cell field, warped once by noise
                   -> each node knows its plate and its boundary class
   L2  CRUST       TWO crust types at TWO densities (continental, oceanic)
                   + isostasy, with the WATER LOAD standing on it
                   -> a TWO-HUMPED hypsometry: a shelf, a slope, an abyssal plain
   L3  OROGENY     convergent / divergent / transform boundaries, and the range
                   height capped by the crust's own strength:  h_max = sigma_y/(rho*g)
                   -> and the orogen publishes a BED DIP direction
   L7  THE SEA     the sea level SOLVED from the body's WATER INVENTORY against
                   this field's own hypsometry -- never a typed ocean fraction
```

**Why the crust dichotomy is not optional.** Earth's coastline is stable because light continental crust
and dense oceanic crust float at different levels. Without two crust types there is no shelf, no abyssal
plain, and the land–sea split swings wildly with the water draw. **The hand-tuned ocean fraction is the
symptom of the missing mechanism, not its cure.**

**The number that matters.** The macro node is **8 224 m** on the home planet. Your 50 km reference frame
is about six nodes across, and one node is about **210 pixels wide** in a 1 280-pixel frame (COMPUTED).
**So nothing you look at in that frame is resolved by the solve.** The solve carries the SKELETON — the
drainage, the hardness, the dip, the ice. **What you SEE is the slope spectrum and the closed-form terms
keyed to that skeleton.** This is the most important sentence in the arc, and it is why the first content
slice buys the spectrum alone, with no solve at all.

### 4.3 The relief law, and the sea

Today: `0.4 % of the radius, clamped 200–12 000 m, times a draw in [0.5, 1.5)` (MEASURED,
`crates/terrain/src/body.rs:145-148`). It is a typed constant wearing a physical hat.

Proposed: **`draw × min(strength bound, shape bound)`**.
- The STRENGTH bound is `sigma_y / (rho · g)` — how high rock can stand before it fails under its own
  weight. Calibrated on Everest it gives Mars **94 %** of Olympus Mons and Venus **112 %** of Maxwell
  Montes: a ±15 % spread across three real bodies.
- The SHAPE bound at `0.077 R` comes from Vesta, an OBSERVED small body. Without it, the strength bound
  alone deletes every body under about 313 km of radius.

**Gravity is now in the relief, which is one of your five words.** A heavier planet has lower mountains,
and it is arithmetic, not a knob.

### 4.4 One rock, or a rock map?

MEASURED (`crates/terrain/src/body.rs:194-201`): a body draws **one** sediment of three and **one**
bedrock of five — for the whole planet. And today's strata table selects a substance by **depth under the
surface**, so every layer drapes over the hill and the order is soft over hard. **A mesa and a hoodoo are
a HARD cap over a SOFT layer, so today's table makes both impossible at any erosion rule.**

Two changes:
- **The stratigraphic column sits at a fixed RADIUS**, not at a fixed depth. Beds then cut the hill, and
  a benched cliff, a mesa and a hoodoo become possible.
- **A ROCK MAP, keyed to the layout, with a BED DIP.** A craton shows crystalline basement; an orogen
  shows steeply dipping rock; a passive margin shows flat carbonate; a rift shows basalt. **A range of
  flat beds reads as Monument Valley, never as the Alps — and your reference picture is an alpine
  range.** The dip is one dot product per column.

**The seed ruling is honoured.** It forbids a seed-derived map of **VALUE**. A map of rock TYPE and
COLOUR is a map of LOOK. **An ORE map stays refused** (L18): a published ore map is a 20× prospecting
advantage for anyone who reads a wiki.

---

## 5. Rivers and erosion — how water writes a landscape

### 5.1 The solve, once ever, per body

```text
   PASS ORDER, repeated for a stated number of passes:

   1  CLIMATE      every CLIMATE_EVERY passes, on the nested sub-lattice
   2  D8 ROUTING   each node picks its steepest of eight neighbours (integer
                   cross-multiplied comparison; no division, no epsilon)
   3  PRIORITY     every pit is filled so water always reaches the sea; each
      FLOOD        filled pit becomes a LAKE with a SPILL LEVEL
   4  DISCHARGE    flow accumulation in WHOLE SQUARE METRES, weighted by each
                   node's own area on the bent cube grid
   5  STREAM       erosion = K * Q^(1/2) * slope, solved implicitly (one divide)
      POWER        against the base level: the sea, or the lake's spill level
   6  REBOUND      isostatic rebound on a pyramid: eroded land rises again
   7  TALUS        anything over the angle of repose slides; THE ANGLE READS
                   THE COLUMN'S ARIDITY
   8  ICE          above the equilibrium line: no liquid water at all; instead
                   trunk over-deepening and cirque marking
   9  CRATERS      only where the charter says the body is airless
  10  COAST        the shoreline is cut where the sea meets the field
```

**Why the climate must be INSIDE the loop, not before it.** It is a circle: relief makes wind, wind makes
rain, rain makes discharge, discharge cuts relief. Run the climate once at the start and the rain shadow
sits behind a range the erosion has not yet built.

### 5.2 What the artifact keeps

```text
   RECOMPUTED, the home planet, R = 3 350 759 m
   macro nodes    6 x 640^2 = 2 457 600  x 6 B  = 14.75 MB
       Z (i16 m) | water level (i16, or dry) | D8 receiver + facies (1 B)
       | log discharge (1 B)
   pyramid        five levels, 2 B                =  1.64 MB   (top level 4.8 KB)
   climate        6 x 160^2 =   153 600  x 8 B  =  1.23 MB
       mean temperature | precipitation | seasonal amplitude | ARIDITY
   rivers, lakes  DERIVED from the bytes above    =  0 B
   ------------------------------------------------------------
   TOTAL KEPT                                     = 17.61 MB
```

**The rivers keep nothing.** Every channel is a chain of receiver bytes already stored; every lake is a
run of the water-level field already stored. The polylines are a runtime index, and their size is
**UNMEASURED** — `M-L6` measures it.

**The cost of the solve: 12–20 s of one core, plausibly 40 s, and 125–130 MB while it runs** (ESTIMATED
by two independent models that agree). **Once ever, per body.**

### 5.3 What water will NOT do here, stated plainly

- **No meander on a great river.** COMPUTED: a 931 m-wide river's meander wavelength is 10.2 km, longer
  than one macro node, so a D8 tree cannot express it. **A great river in this world runs straighter
  than the Mississippi.**
- **No re-solve after a player changes the ground.** A player raises a ridge across the valley below his
  berth. The channel still runs where the seed's water ran, straight through his new rock; the water
  sheet still stands at the old level, **so his dam holds nothing back and his new lake never fills.**
  A re-solve is 12–40 s on a player action and is refused. **The local diff wins inside its own width,
  and nothing further.** This is L19, and you should rule on it rather than let a player find it.
- **No live landscape change.** SL10 forbids time inside the recipe.

### 5.4 The measurement that decides whether the solve exists at all

**`M-L13`: does erosion move a DRY planet's statistics at all?** Our home planet drew a puddle — about
**one column in a hundred** is under water (MEASURED). If forty passes of stream power do not move the
slope p95, the flat fraction, the drainage density and the hypsometric integral, **then the whole solve
closes**, and the cheapest answer is more shaped octaves and a better biome rule. **The arc then falls
from about 26 weeks to about 12, with a plainer world.** This measurement runs before one more line of
solve design.

---

## 6. Climate, biomes and the weather coupling

### 6.1 Your five words, and where each one lands

| Your word | Where it enters | What it changes on screen |
|---|---|---|
| **position** | insolation and equilibrium temperature, from the parent system, one hop | a hot world or a frozen one; the snow line's height |
| **spin** | the Hadley cell's width, the Coriolis strength, the length of the day | how many wind belts and where the deserts sit; a tidally locked world has a burning day side, a frozen night side and a habitable terminator ring |
| **trajectory** | eccentricity, the year's length, the season's shape | a world with a hard winter; the almanac's own calendar |
| **size** | the radius, the escape velocity (which decides the air), the lattice | how far you see; whether there is air at all |
| **gravity** | the relief ceiling `sigma_y/(rho·g)`, the angle of repose, the lapse rate | a heavy world has low mountains and shallow slopes |

### 6.2 The chain, and the three scales of rain

```text
   charter integers                the macro relief             the column's own halo
        |                               |                               |
        v                               v                               v
   insolation, lapse rate ---> OROGRAPHIC LIFT on the ---> a LOCAL term inside the
   Hadley edge from spin       grid, and a MID term         chunk's own halo
   -> the GRID precipitation   5-15 km UPWIND               (~0.35 ms per chunk)
        |                               |                               |
        +-------------------------------+-------------------------------+
                                        v
                            temperature + precipitation
                                        |
                                        v
                            THE WHITTAKER CHART + a short ordered
                            chain of overrides -> 19 biomes in 5 bits
                                        |
                    +-------------------+--------------------+
                    v                   v                    v
              THE SOIL           THE PAINT TABLE      THE VEGETATION
              (a topsoil byte)   (the client's        SKELETON (where a
                                  material)            tree goes; slice 14
                                                       places the asset)
```

**Why three scales and not one.** An 8–33 km climate grid alone puts your whole 50 km vista inside ONE
climate cell. The biome would then vary with height alone, **and the picture would be contour bands**.
The three terms together give a green windward valley beside a dry leeward cliff at the same height.

### 6.3 The two things that make the climate honest, and are not yet true

1. **The precipitation has no unit.** Our supply is computed in pascals; the Whittaker chart's axis is
   millimetres per year; **nothing converted one into the other.** `U15` anchors the constant on two
   points — 2 000 mm/yr at the wet tropics, 100 mm/yr under the dry subtropics — and publishes the
   residuals. **Without it the whole map's realism rests on a number nobody wrote down.**
2. **The wind-belt fit misses Mars by a whole cell.** The Hadley constant is a REVERSE FIT to Earth
   alone. `U8` re-fits it over more bodies and publishes the miss instead of a tick.

**And three families the model refuses by construction, so you learn them here and not later:** no cold
coastal upwelling (so no Atacama and no Namib), no ice–albedo feedback (so every polar cap is a circle of
latitude), and no ocean heat transport and no monsoon (so poles run too cold and the equator too hot).

### 6.4 The climate must reach the SHAPE, not only the paint

`05` calls this *"the deepest believability gap"*: **in a paint-only model, a desert is a grassland with a
sand-coloured topsoil byte.**

A geologist tells an arid landscape from a humid one in ONE photograph at any distance:

```text
   ARID                                  HUMID
   sharp scarps, mesas, angular crests   convex rounded hilltops
   bare rock, no soil mantle             soil-mantled slopes
   alluvial fans at every range front    no bare rock below the tree line
```

**The cure is one lookup and one multiply**: the column's roughness ceiling AND the talus angle read the
column's aridity. The climate grid is already read per column for the biome. **This is the cheapest
believability in the whole arc** (L14).

### 6.5 The weather: three layers, and only one of them ships

```text
   STATIC (the recipe)         THE ALMANAC (derived)        LIVE (the realm's state)
   f(seed, charter, address)   f(charter, universe tick,    a SHORT LIST OF WEATHER
   the shape, the permanent    the placements the window    SYSTEMS: centre, radius,
   ice cap, the climate mean   already carries)             strength, phase, drift
                               the season, the sub-solar    ~16 B each, <= 64 per body
   NO MESSAGE. Both hosts      point, the day, the snow     ~1 KB per body ON CHANGE
   compute it.                 line, AND THE PER-BASIN      ONE statement per REALM,
                               RIVER STAGE                  never one per observer
                               NO NEW LANE (ruling S7-7)
```

**Behind the list, on the shard only**, an anomaly field is stepped by advection with a relaxation and a
seeded forcing: **0.98 MB per home-size body, 3.93 MB per Earth-size body, about 4 ms a step**
(ESTIMATED, `05` §8.5). That is shard memory per AWAKE body, and it needs a stated ceiling.

**The law: a season NEVER invalidates a chunk.** The seed decides the shape and the permanent cover. The
almanac and the storm list decide the PAINT and a thin DERIVED cover both hosts compute from the same
rows. **Only a cell a player's HAND changes becomes a record.**

**Why lying snow is derived.** COMPUTED: one snowfall over one square kilometre, stored as per-cell
diffs, costs **12 MB**. Derived from the shipped list, it costs nothing and both hosts agree by
construction.

**And the river gets a season too.** Discharge is the most seasonal thing in a landscape: a snow-fed
river doubles or triples in the melt, and a dry-season channel shows its bars. **One per-basin stage
offset, derived from the same almanac, applied to the water surface and the drawn channel width.** Same
shape of rule, one lookup, no stored byte (L15).

**A sleeping realm ships nothing.** On waking, the almanac is simply a function of the tick, so a planet
that slept through half its year wakes in the right season.

---

## 7. The fine rungs and the ladder

### 7.1 The fold — what a column actually computes

```text
   s0  = MACRO(dir) + Z(dir)          the layout plus the solved height, read
                                      through a smooth interpolation at a matched
                                      pyramid level; ONE tile gather per chunk
   dc, flow = CHANNEL(dir)            distance to the nearest channel, and the
                                      stored D8 flow direction
   r   = ROUGHNESS(dir) in [r_min,1]  ONE per-column factor, DOWNWARD only, from
                                      the macro slope and capped by the ARIDITY
   s_k = s_(k-1) + A_k * r * n_k(dir) one octave per rung. Between the hillslope
                                      length and the macro node the octave is
                                      RIDGED (1 - |noise|) and sampled in a frame
                                      STRETCHED ALONG THE FLOW -- so the crests
                                      agree with the drainage instead of fighting it
   s+  = TERRACE(s_k)                 the stratigraphic column at a FIXED RADIUS,
                                      TILTED by the bed dip -> benched cliffs, mesas
   h   = CARVE(s+)                    the channel and its floodplain
   gap = (r_cell - h)/(cell*sqrt(1+|grad h|^2))   + the 3-D removals (slice 8f)
```

**The fine rungs SOLVE NOTHING.** They read the artifact and evaluate closed forms. That is what keeps
them inside the 8 ms budget and what lets the client compute the identical bytes.

### 7.2 The free-coarsening law, per term

Today the crate drops one octave per rung **by count** and bounds the rung-to-rung disagreement by the
dropped amplitudes. **That bound is bigger than half a cell at six of the home planet's twelve rungs** —
32.8 m against 32 m at rung 6, and 1 469 m against 1 024 m at rung 11 (COMPUTED). **The ladder's own
promise is broken today, and nobody was looking for it.** The far view already moves the ground by more
than the rung can show, which is a seam in shipped code.

The new law is per TERM, not one claim for the whole fold:

| Term | Its rule | Its dropped magnitude | Status |
|---|---|---|---|
| `Z`, the macro field | rung-independent in DEFINITION; its READING picks a pyramid level by distance, **so it HAS a level step** | the pyramid's own level-to-level difference, printable per node | derivable |
| the octaves | an octave survives while its **wavelength ≥ 4 cells of the rung** | ≈ **3.3 cells of the rung** times the finest surviving octave's per-octave slope | **DERIVED** |
| the terrace | fades to zero over one rung once the bed is under 2 cells | at most one bed thickness (a charter integer) | asserted |
| **the channel carve** | stops when the channel is under 2 cells, and **fades over TWO rungs** | at most the channel depth the artifact states — hundreds of metres on a trunk valley | **the one risky term** |
| the 3-D removals | present only where the cell is under a stated share of the removal's diameter, same two-rung fade | the removal's own depth | asserted |
| the roughness `r` | it does not coarsen at all | zero | **EXACT** |

**COMPUTED under the new spectrum:** the dropped bound at rung 6 falls to **3.7 m**, against 32.8 m today
and 116 m under the count rule. At the vista's own rung the rung-to-rung disagreement falls from **15 m
to 1.2 m, against a 57.5 m drawn pixel.**

**In the game's words.** A pilot flies toward the home planet's great valley. Under today's rule the
valley floor could rise two hundred metres under his nose at one rung boundary. Under the two-rung fade
the same two hundred metres arrives as two steps of about 65 m each, at a distance where one drawn pixel
is 57.5 m. **`M-L14` says at what speed that is still visible. If it is visible, the fade widens to three
rungs — that is the dial.**

### 7.3 The octave table — ONE law, and the trap inside it

`LONG_WAVE_CAP_M` and `SHORT_WAVE_M` are both **deleted**: the ladder decides the ends, and no metre is
typed. The amplitudes come from a **slope spectrum** peaked in the 1–6 km band the eye reads, anchored on
the **angle of repose** so a player can still walk. Every amplitude is capped by the ladder's own
half-cell budget at the rung where it disappears.

**RECOMPUTED: `Z` replaces the coarse octaves, so the count falls from 14 to 11 on the home planet.** The
saving is real and must be credited when the column pass is re-priced.

**★ The trap, and it is measured.** `crates/terrain/src/body.rs:180-184` re-normalises every amplitude so
their SUM equals the relief. **Shorten the coarsest wavelength without changing that rule and the whole
world becomes eight times steeper — 56° at a 2 m baseline.** The slope spectrum and the half-cell budget
together are what make the change safe. **They are ONE decision, not five** — which is why L5 must be
answered before anything else in this arc is built.

### 7.4 What the picture gains, in numbers

| Quantity | Today | After | Mark |
|---|---|---|---|
| Relief inside a 50 m window | 1.14 m | 6.13 m | COMPUTED |
| RMS slope at a 1 m baseline | 1.7° | 18.9° | COMPUTED |
| RMS slope at an 8 km baseline | 1.7° | 1.3° | COMPUTED |
| Amplitude at a 6.25 km wavelength | 80 m | 319 m | COMPUTED |
| Rung-to-rung disagreement at the vista's rung | 15 m against a 57.5 m pixel | 1.2 m | COMPUTED |
| Largest skyline rise over the local trend | 0.054°–0.067° | **ESTIMATED 0.22°–0.27°** | see below |

**★ The honest caveat, and the measurement that removes it.** 0.22°–0.27° is `0.054 × 3.99` and
`0.067 × 3.99`, where 3.99 is the amplitude gain in the band. **Linearity is an assumption.** The march
takes a maximum over 360 bearings of a sum of bands, and a maximum of a sum does not scale with one term.
**So `M1b` re-runs the real four-station march on the proposed amplitudes and publishes the number BEFORE
one week is spent.** It is one afternoon. It either says the arc works, or it saves nineteen weeks.

---

## 8. The measurements the arc owes, FIRST

**Nothing below the line is built before the three starred rows are in hand.** All three are days, not
weeks.

| # | What | How | Gate |
|---|---|---|---|
| **★ U1** | **Is the home planet the right body?** | print the home body's own taxonomy row: mass, density, gravity, insolation, equilibrium temperature, the atmosphere | it is earth-like by the existing predicate, or it is not. **The census's earth-like body is 6 515.5 km; the voxel home planet's look radius is 3 350 759 m — half that, and probably a different, airless, hot body.** If so, every picture you have judged was taken on the wrong world |
| **★ M1b** | **The skyline PREDICTION** | re-run the four-station, 360-bearing, 540 km march on the PROPOSED amplitudes, and on a synthetic macro field | **publish the predicted break count per 60° and the largest rise.** If it stays under 0.10°, the spectrum is not the cure and the arc stops here |
| **★ M8-L** | **THE LIGHT** | re-shoot the three pictures you judged with the star at 12°–18° elevation and 100°–140° off the camera's nose, shadows ON | you judge *"is it the terrain, or was it the light?"*. **Hours of work** |
| U16 | The reference picture's own break count under OUR rule | the same march over the reference | so the headline gate can be a floor and not only a ratchet |
| M0 | The cost baseline re-taken on an idle machine | `just terrain-cost` | reproduces within 10 % |
| M-L10 | The charter's cross-host spread | draw the home system on both target legs; print every float's bits | **every quantum far larger than the spread.** Nobody has ever measured this |
| M1 | The diagnosis re-measured INSIDE the crate | slope at 1 m / 32 m / 512 m; relief in 50 m / 1 km / 20 km windows | the replays reproduce |
| **M-L25** | **Can the new ground be played on?** | the fraction steeper than the character controller's limit; flat area per km² a hull can land on; the fraction buildable without terracing | **stated bands, published BEFORE the spectrum's constants are fixed.** Playability decided the amplitude and then nobody measured it |
| M9 | The spectrum, the slope histogram and the drainage density, ours and the reference's | over a 50 km window against a real elevation model | **median slope 3°–12°; p95 slope 25°–40°; drainage density 0.5–5 km/km²** |
| M-L1 | The solve's pass schedule, nanoseconds per node per pass kind | at 6×64², 6×256², 6×640² | the home lattice solves inside the stated time and memory |
| **M-L13** | **Does erosion change a DRY planet at all?** | four statistics before and after | **the flat fraction rises by at least half; the p95 slope rises; the hypsometric integral moves by ≥ 0.05.** This decides whether the solve exists |
| M-L12 | No drift on the iterative field | the macro digest on four legs, **one a REAL x86-64 machine**, plus a tampered control | every digest equal; the tampered control RED |
| M-L17, M-L4 | The cube net, the areas, and the seam | neighbour counts; areas sum to 4πR²; the height from each side of a shared column | exact counts; area within 1e-5; **the seam's max difference is ZERO quanta, never a p99** |
| **M-L24** | **What does the VEGETATION SKELETON cost?** | one chunk, with the anchor digest | recorded, **BEFORE the 8 ms budget question goes to you.** Your reference picture is mostly forest and no document prices it |
| M-L5 / M2 | The per-chunk budget with the landform work on | the same named chunk set, **including the densest measured chunk, a water sheet AND the skeleton** | under 8 ms; the top rung still cheaper than rung 0 |
| M-L7 | The client's first visit, THREE legs | approach from outside; cold login on the home planet; **arrival at a body whose artifact is NOT pinned** | no black frame on any leg; the ground refused, never wrong |
| **M-L19** | **The vista census** | standing at a 50 km horizon: chunks, vertices, bytes, fill time — work off and on, rock-only then with the skeleton | **under the ceiling L23 states; the fill under 2 s at the client's real thread count.** This decides whether your reference picture is affordable at all |
| M-L22, M-L23 | The biome edge width; the fenced polynomials' error as a BIOME DISPLACEMENT | a transect; per-quantity error | edge under 5 km; displacement under one macro node |
| U8, U15 | The wind-belt fit; the precipitation's UNIT | a wider anchor set; a two-point anchor | residuals published; 2 000 and 100 mm/yr reproduced |
| M-L14 | The pop, on FIVE terms | octaves, a river's arrival, a `Z` level change, a terrace fade, a removal fade | no visible step at a walk and at 240 m/s |
| M-L6 | The river field | ns per column; channel-node fraction; drainage density; the runtime index's bytes | ≤ 100 ns per column; density 0.5–5 km/km² |
| U-L8 | The weather statement's bytes and rate, and the anomaly field's memory per awake body | against the lane's budget and a shard ceiling | **the SL6 ask for weather is not made before this** |

---

## 9. What you decide

**Twenty-eight rows. The register row in the evidence document is named for each.**

| # | Question | Options | Recommendation | Why |
|---|---|---|---|---|
| **L1** | **Does the client DERIVE the macro artifact, or does the realm SHIP it?** (D1) | derive / ship / derive with a pinned home artifact and a fallback | **DERIVE**, with the home planet's artifact a committed build artifact, a disk cache keyed on the FULL input tuple, the 4.8 KB pyramid top first, and shipping kept as a MEASURED fallback | The client generates eight home-planet chunks AT LOGIN, before any realm has spoken, so no lane exists at that moment (ruling S7-2). The pinned home artifact removes the case shipping was protecting |
| **L2** | **Simulation, bake, or closed form?** (D2) | plate simulation / fine-grid iteration / pure closed form / **a closed-form layout plus ONE coarse solve** | **The layout plus one coarse solve** | A fine global bake is 3.6 hours and 131 GB. A pure closed form cannot make a coast or a slope–area law. The coarse solve is the only mechanism with a MEMORY of the surrounding landscape |
| **L3** | **The macro lattice's size** (D3, D4) | 16 448 m (~3 MB) / **8 224 m (17.61 MB)** / 5 140 m (~45 MB) | **8 224 m, with a chooser that walks to the nearest coarser exact divisor until a memory ceiling holds** | RECOMPUTED: an Earth-sized body cannot take 8 224 m at all (it does not divide its grid) and its nearest legal 8 704 m lattice is **49.8 MB and ~58 s**. Under an 18 MB ceiling that body walks to **14 336 m**. **A big planet gets a coarser landscape, and you should know before it ships** |
| **L4** | **The relief law** (D6) | keep `0.4 % of radius × [0.5,1.5)` / **`draw × min(strength bound, shape bound)`**, and it lands WITH the spectrum | **The physical law, at the same slice as the spectrum** | Gravity then decides the mountains. And the spectrum's amplitudes are anchored on the relief, so a vista you approve under the old relief would be re-drawn by the new one |
| **L5** | **★ THE OCTAVE TABLE — one law, and NOTHING ELSE IN THE ARC IS SAFE UNTIL IT IS SETTLED** (D5) | keep three competing proposals / **one merged law** | **The merged law**: the ladder owns the ends (both metre constants deleted); a slope spectrum anchored on the angle of repose owns the amplitudes; the half-cell budget owns the cap; ONE downward-only per-column factor owns the modulation; the middle band is ridged and flow-aligned; the count falls from 14 to 11 and the saving is credited; coarsening is by wavelength | Today the amplitudes are re-normalised so their SUM is the relief. **Change the coarsest wavelength without changing that rule and the world becomes eight times steeper — 56° at a 2 m baseline.** The spectrum and the budget are what make the change safe |
| **L6** | **Does the roughness stay one number per body?** (D7) | one number / **one per-column factor, downward only** | **Per column, downward only** | One number gives 19° everywhere or 2° everywhere. Downward-only keeps the ladder's bound exact, because it never scales up |
| **L7** | **Does the arc land an ICE pass?** (D8) | yes / no, and alpine skylines are fluvial shapes with white paint | **YES** | **Above the snow line there is no liquid water, so river erosion does not act there at all.** Fluvial incision alone gives a V-notch to the summit. The top third of your reference picture is a glacial skyline |
| **L8** | **Two crust types at two densities?** (D9) | yes / no, and the ocean fraction stays hand-set | **YES** | Without it there is no shelf, no abyssal plain and no stable coastline, and the land–sea split swings with the water draw |
| **L9** | **The strata at a fixed RADIUS instead of a fixed depth?** (D10) | yes / keep today's depth-following cake | **YES** | MEASURED: today's table picks a substance by depth under the surface, so every layer drapes over the hill, soft over hard. **A mesa and a hoodoo are a HARD cap over a SOFT layer, so today's table makes both impossible at any erosion rule** |
| **L10** | **Craters on an airless body instead of rivers?** (D11) | yes / no | **YES** | A 100 km moon with smooth noise and no crater is not a believable airless body. It is also how our test pair covers the dry code paths |
| **L11** | **Arches, overhangs and pillars?** (D12, D13) | nobody owns them / a 3-D removal term in its own slice, AFTER the collider slice | **YES, in its own slice after slice 11**, plus PLACED ROCK OBJECTS with their own colliders | A single-valued height cannot make an arch, and your reference picture holds four kinds. **And every cliff in it stands on a scree cone.** Talus and boulders need colliders, so ruling V4's *"trees, grass, decoration"* does not cover them |
| **L12** | **How do the body's facts reach the recipe?** (D14, D15, D16) | floats the recipe snaps / **quantised integers, authored ONCE and STORED** / every host re-derives | **Quantised integers, authored once, stored.** The forest draws the body-local half; the parent system computes and quantises the ORBIT half and states it one hop down; the body's realm holds all of it and states it in its own self-look | VERIFIED: **there is no float fence on `crates/physics`**, so two shards on two architectures can quantise either side of a grid line. And **a realm rescheduled from one machine to another would re-address every stored edit** |
| **L13** | **Is the SPIN draw conditioned by physics?** (D17) | a free draw / **conditioned: tidal locking from the orbit and the star's mass; the tilt damped for close-in bodies** | **Conditioned** | **In the game's words: the pilot approaches the close-in planet of the home system, its charter says the day is nine hours, and its star should stand still in its sky forever.** A free draw makes a lottery ticket, not a fact |
| **L14** | **★ Does the climate reach the SHAPE, or only the paint?** (D45) | only the paint / **the roughness ceiling AND the talus angle read the ARIDITY** | **The shape** | *"A desert is a grassland with a sand-coloured topsoil byte"* is the deepest believability gap the refutations found. A geologist tells arid from humid in one photograph at any distance. **Cost: one lookup and one multiply — the cheapest believability in the arc** |
| **L15** | **Does the almanac reach the WATER?** (D46) | a static water surface all year / **one per-basin STAGE offset derived from the same almanac** | **The river gets a season** | Discharge is the most seasonal thing in a landscape. Today's plan moves the snow line up the range in spring and leaves the river below it unchanged all year. Same rule shape as the derived snow; no stored byte |
| **L16** | **What does the LIVE weather ship?** (D25) | a field / a per-observer window / **a short list of weather systems, one statement per realm** | **The list**, ~16 bytes each, ≤ 64 per body, ~1 KB on change | A field is 393 KB per statement for a 50 km front. A per-observer window is the composed-per-observer shape the reach ruling killed by name. **One statement per realm costs the realm nothing per observer** |
| **L17** | **The biome set and the CLIMATE RUNG** (D21, D22) | keep four biomes and read the height at the drawn rung / **nineteen biomes in five bits, classified at ONE derived climate rung** | **Nineteen, at one derived rung** | A height tolerance in metres cannot bound a CLASS: a column 20 m under the snow line at one rung and 20 m over it at another flips from grassland to ice, **and the white patch moves as the pilot flies in.** That pop exists TODAY. **No saved byte changes: a biome is a column property both hosts derive** |
| **L18** | **Ore from the seed?** (D33, D34a) | yes / **no** | **NO ore map, no placer bar.** Sand, gravel and clay stay lawful as bulk stock | A seed-derived map of value is a treasure map, refused by name by the 2026-08-27 ruling. A published map is a 20× prospecting advantage for anyone who reads a wiki |
| **L19** | **★ A ROCK map and a bed DIP?** (D34b) | one sediment and one bedrock per planet forever / **a per-province palette and a dip from the orogen field** | **YES** | MEASURED: a body draws ONE sediment and ONE bedrock for the whole planet. **One palette for every world, forever.** And flat beds everywhere read as Monument Valley, never as the Alps. **The seed ruling forbids a map of VALUE, not a map of LOOK** |
| **L20** | **What does the artifact owe a surface a PLAYER changed?** (D50) | re-solve / **nothing, and the local diff wins inside its own width** | **Nothing, and it is stated in the acceptance document** | A player raises a ridge across the valley. The channel still runs where the seed's water ran; **his dam holds nothing back and his new lake never fills.** A re-solve is 12–40 s on a player action. **You should rule on this rather than let a player find it** |
| **L21** | **★ Does a VOLCANIC family land?** (D43) | never / a later arc / **this arc, as a closed-form radial term on the plate field** | **This arc if the budget allows; otherwise a later arc, and SAY SO** | **The body already draws Basalt, Gabbro and Andesite as bedrocks, so the world asserts a volcanic history it never shows, and the rock is a lie.** Volcanic is the second landform family after fluvial on an earth-like planet, and a cone and a caldera are cheaper than the ice pass |
| **L22** | **Dunes, karst and landslide scars?** (D44) | never / **named as NOT DELIVERED and registered for a later arc** | **Named, not delivered** | The world has a Desert biome with a Sand topsoil and no dune; it draws Limestone and has caves and no sinkhole. **Telling you before you pay is the point of the row** |
| **L23** | **★ WHO OWNS THE SKY AND THE SHADOW?** (D41) | nobody, and every arc picture is judged unlit and hazeless / **hours of harness fix now, plus a 2-week slice for the atmosphere and a cascaded shadow map** | **Fund both** | MEASURED: both client lights are created with shadows OFF, a fill light runs on the shadow side, the star sits 25° up and **the camera's nose is pointed at the star's own azimuth.** Every picture you judged was shot into the light, with no shadow, on 1.4° ground. **On a ridge lit from behind the camera, a 300 m spur and a 3 m bump look the same** |
| **L24** | **★ WHO OWNS THE GROUND'S COLOUR?** (D42) | nobody — one material for the whole world / **a PAINT TABLE in the biome slice, with its picture verdict as a landing condition** | **The paint table** | MEASURED: the client draws all terrain with ONE material, `srgb(0.55, 0.50, 0.42)`, and the word *"biome"* appears nowhere in the client. **The biome slice's four pictures — a snow cap, forest, a desert behind a range, wet and dry at the same latitude — are ALL COLOURS.** With one brown material the snow cap is brown and the river is brown rock in the shape of a river |
| **L25** | **The 8 ms per-chunk budget, now that the ROCK ALONE is over it** (D29, D30) | keep it and shrink / raise it | **Decide when the measurements are in hand; and shrink FIRST, with a dial that touches the ROCK** | The densest measured chunk is estimated at **7.09–8.07 ms with NO water and NO trees** (the term that grows is the extraction, and the extraction grows with the roughness, which is the whole point of the arc). Dials, in order: a per-chunk vertex budget or a roughness cap keyed to the cell; then the water sheet only where water exists; then the water at a coarser rung. **Your own words stand behind raising it:** *"I think 8 should be ok. If that over time will become a problem, we can rethink and reimplement"* |
| **L26** | **The client's ceilings and the ocean's share** (D28, D31, D49) | none stated | **State four numbers: at most 1 concurrent derive; at most 4 resident artifacts (~70 MB); a mesh ceiling of about 1.5 GB (against a MEASURED 205 MB a client holds today for a 13×13×3 patch); and an ocean target of 55 %–70 % for the median body** | Your 50 km vista is **1.04–1.14 GB of mesh, rock and water only**, and no document holds a ceiling to judge it against. And **the home planet drew a puddle — about one column in a hundred is under water**, which returns desert everywhere, because two of the four rain terms read the sea |
| **L27** | **★ IS THE HOME PLANET THE RIGHT BODY?** (D27) | keep it / **re-pick by the existing earth-like predicate, after one bench measures it** | **Measure first, then almost certainly re-pick** | The census's earth-like body is **6 515.5 km**; the voxel home planet's look radius is **3 350 759 m** — half that, and picked because it is the first body the LADDER accepts, not because it is earth-like. **ESTIMATED airless and hot. If so, every picture you have judged was taken on the wrong world.** The re-pick costs a re-recorded golden table and a new world tag, and it is free until the identity is pinned |
| **L28** | **Does a REAL x86-64 machine block the solve slice, and which gate tier does a believability gate join?** (D26, D51) | emulation is enough / **a real machine first**; and: every merge / hand-run / **a RATCHET tier with a stored baseline and a named owner** | **A real machine first; and a ratchet tier** | Ruling S5-5 already calls the emulated leg a smoke test. **A forty-pass accumulation over 2.46 M nodes is exactly what an emulator is least likely to reproduce.** And VERIFIED: `just gate` already runs three terrain gates and not the cost bench, so the project has three tiers and no written rule for which one a new gate joins. **A believability gate that blocks every merge will be switched off, and a gate nobody runs is not a gate** |

---

## 10. Laws — how each is honoured, and the asks

### 10.1 The laws

- **SL10 clause 1 (the seed decides the static shape).** The shape becomes
  `f(seed, address, STORED CHARTER)`. **This is a widening, and only you may grant it** (L12). Two
  consequences: a body's shape cannot be derived before its realm RUNS and states its self-look (a
  charter rides a running realm's statement), so a dormant realm cannot be warmed ahead; and re-tuning a
  charter draw re-shapes every stored body.
- **SL10 clause 2 (ONE generator, a port forbidden).** Everything here lives in `vd-terrain` and
  `vd-seed`, compiled into the server and into every client. The solve is a new function in the same
  crate, under the same float fence.
- **SL10 clause 3 (no drift is MEASURED).** An iterative field on a bent grid adds failure modes a closed
  form does not have, so it gets **twelve stated rules**: integer accumulators; a deterministic flood
  order; integer slope comparison; an integer flat-resolution; discharge in whole square metres;
  `m = ½` (a square root the fence grants) and `n = 1` (no inner iteration); stated pass counts, never a
  convergence test; single-threaded with stated reduction orders; committed literal polynomials, never a
  build-time transcendental; a cell-centred lattice that divides the grid exactly; the cube seam crossed
  by construction; **and the lattice chooser reads only the charter and committed constants, never the
  host.** Four new gates check the cube: the seam's height and slope equal to **ZERO quanta**, the corner
  net, the area sum, and no channel ending on a face edge.
- **SL10 clause 7 (the client derives no pose, no velocity, no entity state).** It derives SHAPE, and it
  draws. The light and the paint are style: they move no vertex the collider shares.
- **SL5 (ONE world).** No variant, no scale knob, no test-only planet. Where a coverage arm needs a wet
  body, a Tier-A **fixture takes a stated macro field as an INPUT**. A fixture invents a test input,
  never a world.
- **SL8 (a seam is a defect).** §7.2 gives every term of the fold a stopping rung and a fade, and five
  pop measurements chase them. **And the ladder's own promise is broken TODAY at six of twelve rungs;
  this arc is what fixes it.**
- **SL1 (a realm is told where it is).** The parent computes the orbit-derived facts because it owns the
  orbit relation, and states them ONE HOP down. Nothing here is a placement.
- **SL3 (a realm draws itself).** The charter rides the realm's own self-look. The parent's per-child bag
  is untouched.
- **SL9 (a parent's child count is unbounded).** Nothing here is per-child or per-observer. The weather
  is one statement per realm.
- **HR3 (one tooling).** No feature branches on a shard kind. A hull holds terrain and holds NO artifact;
  the fold then reads a NULL artifact, and the code cannot tell the two apart.
- **HR4 (features once, run anywhere).** Every slice names a PAIR — a spherical planet and a Cartesian
  hull, station or asteroid — and one identical fixture. **The macro lattice sits BELOW the geometry
  seam**, so it owes the seam's own two gates.
- **HR5 (100 % coverage in Tier-A).** The airless-body pair covers the dry paths; a fixture fed a stated
  macro field covers the wet ones.
- **The seed-and-secrecy ruling.** A map of shape and colour is lawful; a map of value is not. **No ore
  map, no placer bar** (L18). The rock map is a map of LOOK (L19).
- **No magic numbers.** Every parameter derives from the seed or from a physical fact of the body. The
  erosional AGE becomes a charter draw, not a typed world constant, so a young world's sharp ridges and
  an old world's rounded stumps are free variety.

### 10.2 The asks (SL6 — the default is NO)

| Ask | Data, direction | Why it cannot be computed locally | Recommendation |
|---|---|---|---|
| **A1 — the body charter** | ~20 quantised integers, ~42–76 bytes, inside the widened surface tag on the realm's own self-look; realm → its observers' clients | The client links no motion crate by an isolation row, and must not re-derive an unfenced float in its binary | **ASK: YES.** A protocol-minor bump. **The byte count is MEASURED against the 1 200-byte self-look budget before it lands.** Integers, never floats |
| **A2 — the orbit-derived facts** | insolation, equilibrium temperature, eccentricity, quantised by the system; system realm → the body's realm, ONE hop | The body's realm does not own the orbit relation; the parent does | **ASK: YES.** None of these is a placement |
| **A3 — the macro artifact on a bulk lane** | the 4.8 KB pyramid top, then the 17.61 MB field; realm → client | Only if the client cannot derive it in time | **REGISTERED, NOT ASKED.** It is made only if the client's first-visit measurement fails. **The lane is designed now and opened on a measurement** |
| **A4 — the live weather list** | ≤ 64 rows of 16 bytes, ~1 KB per body on change, on a NEW lane with its own rate | Weather is live state, and the client may never derive live state | **ASK LATER**, after the bytes and the rate are measured. It may NOT ride the surface tag, which is *"once per realm on change, carried retained"* |
| ~~A5 — a per-observer weather window~~ | — | — | **NO.** That is the composed-per-observer shape the reach ruling killed by name |

---

## 11. How it is built

**The ordering rule, corrected.** The store's identity is pinned at **slice 14**, not slice 9 (the
foundation's own rule 1: *"the stores of slices 9 to 13 are THROW-AWAY until slice 14 lands"*). **So this
arc does not have to sit in front of the store. It has to close before slice 14.** That lets the erosion
slices land AFTER the collider slice, where they belong.

| Order | Slice | What lands | The picture it earns |
|---|---|---|---|
| 1 | **8 — the ladder** | the tier rule, the crossfade, the residency band; the one-rung dev flag DELETED | — |
| 2 | **8p — the instrument** | the picture probe (each pixel states what drew it and how far), the on-frame stamp, the ruler, FOUND stands, tolerances taken from the body's own measured field | the calibration picture |
| 3 | **8L — the light (HOURS)** | the star moved off the camera's nose, lowered to a raking angle, and the shadow switched ON; the three judged pictures re-shot | **the answer to *"terrain, or light?"*** |
| 4 | **8a — the picture, cheap** | the slope spectrum, the roughness field, the wavelength rule, ridged crests, the cap-rock bench, **and the new relief law and the sea** | **the pilot's 50 km vista at 1.8 m — the first honest test of *"not interesting enough"*** |
| 5 | **8b — the charter** | ~20 integers, the conditioned draws, the widened tag, the home pin | the ocean-fraction sweep, judged by you |
| 6 | **8s — the sky and the shadow** | the atmosphere from the scale height, aerial perspective, a sky dome, a cascaded shadow map | **depth planes, and a ridge that reads as a ridge** |
| — | *(slices 9, 10, 11 — the store, the diff lane, the collider)* | | |
| 7 | **8c — the solve** | the lattice, the climate in the schedule, D8, the flood, discharge, stream power, isostasy, talus, **ice**, craters, the coast, the bed dip, the twelve determinism rules, the four seam gates, the artifact and its cache | the body from orbit; a cube corner; a face seam; a glacial skyline beside a fluvial one |
| 8 | **8d — the water and the rock** | rivers, lakes, the clipped water sheet, the carve with its two-rung fade, the stratigraphic column, the rock map, differential erosion, alluvium | **a river to the horizon, a lake, a coast, a benched cliff, a mesa — THE VISTA YOU JUDGE** |
| 9 | **8e — the biome and the paint** | nineteen biomes at a derived climate rung, the soil law, the snow and tree lines, sea ice, **the PAINT TABLE**, the vegetation skeleton's inputs | **the same frame in its own colours** |
| 10 | **8f — the third dimension** | the 3-D removal term with its own fade; placed rock objects with slice 11's colliders | the arch; the scree cone |
| 11 | **18 — the almanac and the weather** | the derived almanac including the river's stage, the live weather lane, the client's cloud and rain, wind in the realm's medium | a cloud deck; a storm on one side of the frame; the same ridge in two seasons |

**ESTIMATED 22–28 weeks for one engineer**, measurement runs included.

### 11.1 What the flagship frame holds at the end of each slice

**This is the most useful table here: it says which slice first makes a picture worth your judgement.**

| After | The pilot's 1.8 m frame holds |
|---|---|
| today | a lit plain, a hard 400 m horizon, a black sky, no shadow, one brown material |
| **8L** (hours) | **the same ground under a raking light with a cast shadow** |
| **8a** (+2 wk) | **ridges, crests and a 50 km skyline** — still brown, no haze, no river |
| **8s** (+2 wk) | the same, **with a blue haze separating four depth planes** |
| **8c** (+6 wk) | a drainage skeleton, a coast, a glacial skyline — mostly visible from orbit |
| **8d** (+4 wk) | **a carved valley, a river to the horizon, a lake, a benched cliff, a mesa.** ★ **the first frame that answers your sentence**, still in one colour |
| **8e** (+2.5 wk) | the same frame **in colour: a white cap, a green windward side, a tan leeward desert, blue water** |
| **8f** (+2 wk) | an arch; a scree cone under the cliff |
| **18** (+3 wk) | a cloud deck, rain on one side, a river in flood, one ridge in two seasons |
| **slice 14** | **the forest** |
| **slice 16** | **the character, standing upright, for scale** |

**Read the last two rows.** Your reference picture is mostly forest with a figure in it, and **neither is
in this arc.** They are slices 14 and 16.

### 11.2 The four things that could move the estimate most

1. **`M1b`.** If the predicted skyline stays flat, the spectrum is not the cure and the diagnosis re-opens
   before a week is spent.
2. **`M-L13`.** If erosion does not move a dry planet's statistics, the solve closes and the arc falls to
   about **12 weeks** — with a plainer world.
3. **`U1` and L27.** A re-picked home planet costs **1–2 weeks** of re-pinning, free until slice 14.
4. **The 8 ms fight**, which now starts from a measured FAIL on the rock alone.

---

## 12. The pictures protocol

**Why a protocol at all.** MEASURED: three of the last picture set's verdicts used `relief_bound_m` as
their tolerance. On the home planet that is **14 304.9 m**, which at 300 m of eye height is a **277 pixel
tolerance against a 69.4 pixel signal**. **So a verdict PASSED the very picture this work exists to
refuse.** An instrument that cannot fail measures nothing, which is why the instrument lands before the
first content slice.

**Every judged picture carries four things, and each one is measured, never typed.**

1. **THE STAMP** — an on-frame readout and an aligned state file: the realm, the stand's address, the
   eye's altitude, the geometric horizon, the DRAWN horizon, the radius of terrain drawn, every rung
   drawn with its chunk count, **the star's elevation AND its angle from the camera's nose**, the biome
   under the eye, the height over the sea, the ruler's measured pixels, the world identity and the
   universe tick. **One struct, three consumers, one source.**
2. **THE PROBE** — a second aligned buffer in which each pixel states WHAT drew it and HOW FAR it is,
   holding an AUTHOR and a MESH INSTANCE. It replaces today's fitted colour classifier, which calls a
   pixel ground when `r >= g >= b && r >= b + 8 && r > 24`. **An author alone cannot refuse an impostor;
   the instance can.** And once the paint table lands, a colour classifier stops working at all.
3. **THE RULER** — a subject of stated size in frame, measured in pixels against a prediction. The avatar
   marker is exact inside **145 m** (MEASURED). It is an INSTRUMENT, not the reference picture's scale
   figure.
4. **THE LIGHT, STATED** — the star at **12°–18°** of elevation and **100°–140°** from the camera's nose,
   with the terrain light's shadow map ON. The stamp asserts all three. One light, one shadow; the fill
   light retires.

**The stands are FOUND, never typed.** A satisfying predicate at a stated scale, over a co-prime stride of
ladder addresses, inside the terrain crate under the float fence, first hit wins. An argmax is refused:
on a homogeneous field it is not a landform, and it costs 33 minutes.

**The flagship is the PILOT'S EYE at 1.8 m, not a drone.** Your reference picture's 50 km comes from
RELIEF, not from altitude: a point at 50 km needs **323 m** of prominence to clear the horizon plane, and
at 100 km it needs **1 390 m** (MEASURED arithmetic on R = 3 350 759 m). **A flat world then FAILS the
flagship instead of being flown over.**

**Every tolerance comes from the body's own MEASURED field** — three times the body's own rms wobble at
the view length, which is four to six pixels at every scale, because the field is nearly self-similar.
**No verdict may use `relief_bound_m`.**

**The headline gate is a RATCHET until the reference is counted under our own rule.** The reference's
*"8–12 breaks per 60°"* was counted by eye off a screenshot; our 0 was computed by a stated rule. **They
are different quantities.** `U16` runs our rule over the reference, and only then does the ratchet become
a floor. The gate joins a ratchet tier with a stored baseline and a named owner (L28) — **because a
believability gate that blocks every merge will be switched off.**

---

## 13. The one page to answer this week

**Answer six rows and let two weeks of work run.**

| Answer | Row |
|---|---|
| Derive or ship the artifact | **L1** |
| The merged octave law — nothing else is safe until it is settled | **L5** |
| The charter as stored integers, which widens SL10 clause 1 | **L12** |
| Who owns the sky and the shadow | **L23** |
| Who owns the ground's colour | **L24** |
| Is the home planet the right body | **L27** |

**And let three measurements and two slices run**: `U1` (one bench), `M1b` (one afternoon), `M8-L` (a few
hours), then slice 8p and slice 8a (two weeks). **They open no lane, they change no saved byte, and they
produce the one picture that says whether the rest is worth building.**
