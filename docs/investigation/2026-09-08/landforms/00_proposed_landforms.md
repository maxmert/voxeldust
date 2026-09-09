# 00 — THE PROPOSED LANDFORMS: one world, believable land, from the body's own facts

**Date:** 2026-09-08. **Revision 4 — THE FINAL REVISION.** It answers every finding of
`verdicts/00_critic_a.md` (15 findings) and `verdicts/00_critic_b.md` (18 findings). §12 is the
answer table.
**The synthesis** of the eight domain reports of this date (`01`–`08` in this directory), each at
revision 3 after two rounds of refutation.
**Status:** an investigation report for the owner. **It decides nothing by itself.** §7 is the decision
register. Nothing in it is built before the owner rules on it.
**The owner reads the DISCUSSION document** `slice_landforms_discussion.md`, which stands alone. This
document is its evidence.

**The owner's words this report answers.** *"Make sure that we reach that quality on the picture for
earth-like planets (biomes can be different of course, should be dependent on the planet position, spin,
trajectory, size and gravity, etc.). It should be very believable, as we also should simulate the
weather."* And, of our own three pictures: *"no orienters and no details at all … The only thing I'm
worried about is that the surface will not be interesting enough."*

**Binding law read first.** CLAUDE.md (HR1–HR6, SL1–SL10);
`docs/design/owner_decisions_2026-09-07_voxels.md` (V1 SL10, V2.1–V2.9, V4, V6, V8–V12, S5-x, S7-x);
`docs/design/owner_decisions_2026-08-27_seed_and_secrecy.md`;
`docs/design/owner_decisions_2026-09-02_reach.md`; `docs/design/owner_decisions_2026-09-05_suit.md`;
SL8 (a seam is a defect). Where a design document and a ruling disagree, the ruling wins.

**Every number carries a mark.**
- **MEASURED** — a program in this repository produced it, and the program is named.
- **REPLAYED** — a domain re-implemented the crate's own arithmetic outside the crate and ran it. Each
  replay reproduces a value the crate's own bench already MEASURED, so the replay could have failed.
- **COMPUTED** — arithmetic on a measured number, with the arithmetic shown.
- **RECOMPUTED (rev 4)** — this revision re-did the arithmetic and states the result, which sometimes
  corrects revision 3 or a critic.
- **ESTIMATED** — an operation model, not a result.
- **UNMEASURED** — not known. The measurement that would settle it is named.

---

## 1. THE SUMMARY, IN ONE PAGE

**The surface is empty by arithmetic, and three measurements say why.**

1. **The skyline holds no break the eye can name.** From four named ridge stations, ray-marched through
   the crate's own height field over 360 bearings and out to 540 km, the largest rise over the local
   trend is **0.054°–0.067°** (REPLAYED, `01` §1.4). A full moon is 0.52° across. The tallest feature on
   this planet's horizon is an eighth of a moon's width.
2. **The ground under the boots is a plane.** Inside 50 m of a column, after the tilt is removed, the
   surface departs from a flat tilted plane by **0.26 m rms** (REPLAYED, `01` §1.3). The median slope is
   **1.36°**, and it is the same at a 1 m, a 10 m, a 100 m and a 1 000 m baseline. A field with the same
   slope at every baseline is smooth by definition.
3. **The height field carries nothing between 1 m and 48.8 m.** `SHORT_WAVE_M = 30`
   (`crates/terrain/src/body.rs:25`) stops the octave table at the first wavelength under 30 m, which on
   the home planet is 48.83 m (COMPUTED, `02` §1.6). The cell is 1 m.

**The relief is not the problem.** The home planet carries 14 304.9 m of amplitude and its realised
surface spans 17.0 km (REPLAYED, `01` §1.2–§1.3). Earth's dry land spans about 9.3 km (PUBLISHED). We
carry nearly twice Earth's land relief and show none of it, because **the relief has no structure below
the wavelength of a province**, and because **an isotropic sum of band-limited noise has no lines** — no
ridge crest, no coastline, no scarp, no basin rim, no valley axis, at any scale (`04` §0).

**A rougher noise alone does not cure it, and that is measured.** Setting the roughness to Earth's own
spectral exponent, with the relief unchanged, makes 30 % of the planet steeper than 45° — an unwalkable
gravel heap — and still buys only 0.8 skyline breaks per 60° (REPLAYED, `01` §1.6). **Roughness is not
landform.**

**A fourth cause is not the terrain at all, and revision 4 gives it a slice.** The picture harness puts
the star 25° over the horizon and then points the camera's nose toward the star's own azimuth (MEASURED,
`crates/bins/tests/terrain_pictures.rs:44,458-462`), and every light in the client is created with
`shadows_enabled: false` (MEASURED, `crates/client-render/src/terrain.rs:459,473`), with a fill light on
the shadow side at `FILL_SHARE = 0.08` (`:56`). Every picture the owner judged was shot into the light,
with no shadow, on 1.4° ground. **Revision 3 named this cause and then left it homeless. Revision 4 gives
it slice 8L inside the instrument, and gives the sky and the shadow pass slice 8s with a cost**
(critic A F1, critic B B3).

### 1.1 The proposal, in one sentence

**Solve erosion ONCE per body on one coarse global lattice; keep a small artifact; and make everything
the player's eye resolves a closed-form function of the address, the artifact and the body's own
physical facts — so the picture comes from water, rock, climate and LIGHT, and never from more noise.**

### 1.2 The eight parts

| # | The part | What the eye gets | Owner |
|---|---|---|---|
| 1 | **The body charter** — the body's physical facts as quantised INTEGERS, authored once and stored | biomes and relief that follow the owner's five words: position, spin, trajectory, size, gravity | `02` §5.4, `05` §3, `06` §4.2 |
| 2 | **The macro layout** — plates, isostasy on a crust field, orogeny under a strength ceiling | continents, ocean basins, ranges, trenches, rifts, passive margins, a two-humped hypsometry | `02` §4 |
| 3 | **The macro solve** — D8 routing, a priority flood, discharge, stream power, isostatic rebound, talus, ice, craters, the coast | drainage that runs downhill everywhere, carved trunk valleys, lakes, U-shaped glacial troughs, a real coast | `03` §4 |
| 4 | **The climate, inside the solve's schedule** — insolation, the lapse rate, a Hadley edge from the spin, orographic lift, a rain shadow | rain where rain belongs, so the erosion is not uniform, and **the ARIDITY reaches the SHAPE** (rev 4, F6) | `05` §5, `06` §6.4 |
| 5 | **The slope spectrum and the roughness field** — the relief redistributed by a spectrum anchored on the angle of repose, modulated DOWNWARD per column | the 1–6 km band the eye reads; a craton is a plain, an orogen is the reference picture's range | `03` §4.2, `02` §4.6 |
| 6 | **The fold** — the fine rungs read the artifact and solve nothing: a ridged flow-aligned band, a stratigraphic column and its terraces, the channel carve, a fine floor | spurs, gullies, ridge lines, benched cliffs, mesas, a river bed a pilot can follow | `04` §4 |
| 7 | **The ladder, coarsening BY WAVELENGTH** — an octave survives while its wavelength covers four cells of the rung | the far view is the near hill, with a bound that is finally true at every rung | `03` §7.1, `04` §5 |
| 8 | **The live layer** — a derived almanac and a shipped list of weather systems | seasons, a moving snow line, clouds, rain, wind that a ship feels, **and a river that rises in the melt** (rev 4, F13) | `05` §8 |

**And a ninth part revision 3 did not have: THE LIGHT AND THE PAINT.** A shape with no shadow is not
read; a biome with no colour is not seen. Slices 8L, 8s and the paint table in 8e are new in revision 4
and they are the cheapest believability in the whole arc (critic A F1, F2).

### 1.3 What the picture gains, in numbers

| Quantity | Today | After | Mark |
|---|---|---|---|
| Relief inside a 50 m window | 1.14 m | 6.13 m | COMPUTED, `02` §4.6 |
| RMS slope at a 1 m baseline | 1.7° | 18.9° | COMPUTED, `02` §4.6 |
| RMS slope at an 8 km baseline | 1.7° | 1.3° | COMPUTED, `02` §4.6 |
| Amplitude at a 6.25 km wavelength | 80 m | 319 m | COMPUTED, `03` §0 item 3 |
| The abyssal plain below the continental surface | no continent exists | 6 476 m | COMPUTED, `02` §4.2 |
| The rung-to-rung disagreement at the vista's rung | 15 m, against a 57.5 m pixel | 1.2 m | COMPUTED, `03` §15 |
| **The largest skyline rise over the local trend** | **0.054°–0.067°** | **ESTIMATED 0.22°–0.27°** — see below | REPLAYED today; ESTIMATED after |
| Skyline breaks per 60° of view | 0 | **PREDICTED BY `M1b`, WHICH RUNS BEFORE SLICE 8a** | `01` §6.1, U16 |

**The prediction, and why it is only ESTIMATED (critic A F3).** The skyline march is a function of the
octave amplitudes alone. The proposed spectrum raises the 6.25 km amplitude from 80 m to 319 m, a factor
of **3.99** (COMPUTED, `03` §0). If the largest rise scaled linearly with that band's amplitude, it would
move from 0.054°–0.067° to **0.22°–0.27°** (RECOMPUTED rev 4: 0.054 × 3.99 = 0.215; 0.067 × 3.99 = 0.267),
against a 0.10° break threshold and a moon's 0.52°.
**Linearity is an assumption, not a result** — the march takes a maximum over 360 bearings of a sum of
bands, and a maximum of a sum does not scale with one term. **So revision 4 does not print a break count.
It adds `M1b`: re-run `01` §1.4a's four-station march on the proposed amplitudes, and publish the
predicted count and the largest rise BEFORE slice 8a starts.** It is one afternoon of the same replay
that produced the 0.054°, and it either says the arc works or it saves nineteen weeks.

### 1.4 What it costs

**Every rock figure below is ROCK AND WATER ONLY. No figure includes the vegetation skeleton, and the
reference picture is mostly forest** (critic A F7). `M-L24` prices the skeleton on one chunk, and it runs
BEFORE the owner answers D29.

| Item | Cost | Mark |
|---|---|---|
| The macro solve, ONCE EVER per body | **12–20 s of one core, plausibly 40 s**; 125–130 MB transient | ESTIMATED by two independent models that agree (`03` §10, `07` §5.3) |
| **The artifact KEPT, home planet, every field named** (rev 4, B1) | **17.61 MB**: 14.75 MB of 6-byte nodes + 1.64 MB pyramid + 1.23 MB of kept climate. The river graph and the lake table are DERIVED from the same bytes and keep none of their own; their runtime index is UNMEASURED | RECOMPUTED, §3.3 |
| A rung-0 surface chunk | 3.30 ms → **4.10–4.34 ms** of 8 ms | MEASURED base + ESTIMATED delta, `07` §7.1 |
| The cave-dense SEAM chunk | 6.11 ms → **7.07–7.42 ms** of 8 ms | MEASURED base + ESTIMATED delta, `07` §7.1 |
| **★ The DENSEST measured chunk (23 084 vertices)** | **5.22 ms → 7.09–8.07 ms — OVER the 8 ms budget, with NO water sheet** | MEASURED base + ESTIMATED delta, `07` §7.1. **Revision 3 omitted this row** (critic B B2) |
| A cave-dense chunk that also meshes a water sheet | **8.46–10.6 ms — further OVER** | ESTIMATED, `03` §7.4, `06` §5 |
| The owner's own 50 km vista, ROCK AND WATER ONLY | **`08`: 2 805 chunks at 405 KB = 1 137 MB. `07`: 4 200 chunks at 248 KB = 1 042 MB.** Two estimates under two residency rules, not one range | RECOMPUTED rev 4 (critic B B17.3) |
| The live weather the shard steps | **0.98 MB per awake body today, 3.93 MB on an Earth-like body, about 4 ms per step** | `05` §8.5. **Revision 3 priced the OUTPUT list and never the model** (critic A F14) |
| The work | **10 slices, ESTIMATED 22–28 weeks** | §10 |

**The three costs the owner should see first.**
1. **The rock itself already breaks the 8 ms budget** — 8.07 ms on the densest measured chunk, before
   any water and before any tree. The term that grows is the extraction, and the extraction grows with
   the roughness, which is the whole point of the arc. D29 gains a fourth dial that acts on the rock.
2. **The vista is a hundred times the chunk budget**, and the 17.6 MB artifact is under two per cent of
   the mesh the same picture holds.
3. **The forest is in none of these numbers.**

### 1.5 The TEN things this proposal does NOT deliver, and the three climate families it refuses

Revision 3 stated six. Revision 4 states ten, plus three refused climate families (critic A F5, F10).

1. **No fields, no roads, no walls.** They are cultivated and built. They are live state, the same class
   as a player's house. They are out of this arc (`06` §6.9). The reference picture holds all three.
2. **No moving glacier.** The ice pass MARKS where ice stood and shapes the trough. A glacier with its
   own surface and its own motion is not a static shape (`04` §13).
3. **No arch, no overhang and no free-standing pillar from the height field.** A single-valued `h(dir)`
   cannot make one. They need the 3-D removal term of slice 8f (`02` §4.8, `04` §6.2).
4. **No meander on a great river.** COMPUTED: a 931 m river's meander wavelength is 10.2 km, longer than
   one macro segment, so the D8 tree cannot express it (`03` §13). A great river in this world runs
   straighter than the Mississippi.
5. **No live landscape change.** The land does not erode while a player watches. SL10 forbids time inside
   the recipe (`06` L10).
6. **★ NO VOLCANIC FAMILY** — no cone, no caldera, no lava plain, no neck, no volcanic plateau. **The body
   already draws Basalt, Gabbro and Andesite as bedrocks** (`crates/terrain/src/strata.rs:136-142`), so
   the world asserts a volcanic history it never shows. D43 asks whether it lands.
7. **★ NO AEOLIAN FAMILY** — no dune, no yardang, no ventifact, on a world that has `Biome::Desert` and a
   `Sand` topsoil and an owner who asked for weather by name.
8. **★ NO KARST** — no sinkhole, no doline, no dry limestone valley, on a world that draws `Limestone` and
   already has caves.
9. **★ NO MASS WASTING BEYOND TALUS** — no landslide scar, no rock-avalanche deposit.
10. **No blue haze and no cast shadow until slice 8s lands.** Revision 3 left both homeless. Revision 4
    gives them a slice and a cost, and D41 asks whether the owner funds it.

**The three climate families the model refuses by construction** (`05` §5.4, §5.6, and a search of all
eight documents):
- **No cold coastal upwelling**, so no Atacama and no Namib.
- **No ice-albedo feedback**, so every polar cap is a circle of latitude.
- **No ocean heat transport and no monsoon.** A world with no ocean heat transport runs its poles too
  cold and its equator too hot, which moves every biome boundary the picture shows.

---

## 2. THE ARCHITECTURE

### 2.1 The static shape: the layers, top to bottom

```text
 ================================ ONCE PER BODY, EVER ================================

  [A] THE PHYSICAL FACTS OF THE BODY  -- the charter, INTEGERS, authored once, STORED
      radius | mass | bulk density | surface gravity | escape velocity
      mean insolation | equilibrium temperature | optical depth | star class
      atmosphere scale height + mean molecular weight | liquid-water flag
      SPIN PERIOD (tidal locking applied) | OBLIQUITY'S COSINE | ECCENTRICITY
      water inventory | effective elastic thickness | glacial equilibrium-line offsets
      EROSIONAL AGE  <- new in rev 4 (critic B B14): a body fact, not a world constant
                |                                     ^
                |                              drawn by the FOREST; the orbit-derived
                v                              facts come ONE HOP from the parent system
  [B] THE MACRO LAYOUT  (closed form, no iteration)          ~8 224 m per node
      L1 plates ......... spherical Voronoi, warped by one noise call
      L2 isostasy ....... a CRUST FIELD with the WATER LOAD standing on it
      L3 orogeny ........ convergent / divergent / transform, under h_max = sigma/(rho_c*g)
                          and it publishes a BED DIP direction  <- new in rev 4 (F11)
      L7 the sea ........ solved from the WATER INVENTORY against the field's hypsometry
                |
                v
  [C] THE MACRO SOLVE  (iterative, single-threaded, a stated pass count)
      +---> CLIMATE ....... insolation, the lapse rate, the Hadley edge from the SPIN,
      |                     orographic lift, the rain shadow      every CLIMATE_EVERY
      |                     ON ITS OWN NESTED SUB-LATTICE (1/4 of the macro), in the
      |                     SAME SCHEDULE  <- rev 4 (critic B B1)
      |          |
      |          +--- ARIDITY --> the TALUS ANGLE and the ROUGHNESS CEILING   <- rev 4 (F6)
      |          |
      |          v
      |     D8 ROUTING -> PRIORITY FLOOD -> DISCHARGE -> STREAM POWER (m=1/2, n=1)
      |          |         (two surfaces:      (whole square      (a lake's SPILL LEVEL
      |          |          terrain + flood)    metres)            is the base level)
      |          v
      |     ISOSTATIC REBOUND (a pyramid) -> TALUS -> ICE -> CRATERS (airless) -> COAST
      |          |
      +----------+  the LOOP: relief -> wind -> rain -> discharge -> erosion -> relief
                |
                v
  [D] THE ARTIFACT KEPT  ---- 17.61 MB on the home planet, every field named in 3.3
      6 B per macro node: Z (i16 m) | water level (i16, or a dry sentinel)
                          1 B: D8 receiver + surface facies | 1 B: log discharge
      2 B per pyramid node over five levels
      8 B per CLIMATE node on the nested sub-lattice: mean temperature, precipitation,
             seasonal amplitude, ARIDITY
      the river graph, the lake table and the bed dip are DERIVED from these bytes

 ============================== PER COLUMN, EVERY CHUNK ==============================

  [E] THE FOLD -- coarse to fine, stopped early at a coarse rung, SOLVING NOTHING
      s0 = MACRO(dir) + Z(dir)              read through a Catmull-Rom at a matched
                                            pyramid level; one tile gather per chunk
      dc, flow = CHANNEL(dir)               distance to the nearest channel, and the
                                            stored D8 direction
      r  = ROUGHNESS(dir) in [r_min, 1]     ONE per-column factor, DOWNWARD only,
                                            capped by the column's own ARIDITY
      s_k = s_(k-1) + A_k * r * n_k(dir)    one octave per rung; RIDGED and FLOW-ALIGNED
                                            between the hillslope length and the macro node
      s+ = TERRACE(s_k)                     the stratigraphic column at a FIXED RADIUS,
                                            TILTED by the bed dip the layout published
      h  = CARVE(s+)                        the channel and its floodplain
      gap = (r_cell - h) / (cell*sqrt(1 + |grad h|^2))   + the 3-D removals
                |
                v
  [F] THE LADDER      EVERY TERM STATES ITS OWN COARSENING RULE AND ITS OWN DROPPED
                      MAGNITUDE (rev 4, critic B B8). The octaves survive while their
                      wavelength >= 4 cells of the rung. Z is rung-INDEPENDENT in its
                      DEFINITION and level-dependent in its READING.
                |
                v
  [G] THE BIOME, at a DERIVED CLIMATE RUNG (so it never changes with the detail level)
      the Whittaker chart + a short ordered chain of overrides -> the soil, the snow,
      the sea ice, the PAINT TABLE the client draws with (rev 4, F2), and what the
      VEGETATION SKELETON reads (V4: art assets, no primitives)
                |
                v
  [H] THE EXTRACTOR -> the mesh, the collider, the digests (cell, mesh, ANCHOR)
                |
                v
  [I] THE CLIENT -- draws it, WITH A CAST SHADOW AND A SKY (slice 8s).
                   It derives the STATIC SHAPE (SL10) and nothing else.
```

**In the game's own words.** The home planet's realm holds its charter in its own store and states it in
its own self-look. Its shard solves the planet once, keeps 17.61 MB, and answers chunk requests from it.
A pilot in a hull approaches; the client derives the same artifact in the background from the same seed,
the same address and the same stated integers, and checks its digest against the realm's. The pilot lands
in a valley the water cut. He walks up the valley side; the ground under his boots is the same fold the
shard's collider used, so he never falls through it. The low star throws the ridge's shadow across the
valley floor, and the far range stands behind a blue haze.

### 2.2 The live layer

```text
   STATIC (the recipe)          THE ALMANAC (derived)         LIVE (the realm's own state)
   -------------------          ---------------------         ----------------------------
   f(seed, charter, address)    f(charter, universe tick,     a SHORT LIST OF WEATHER
   the shape, the permanent     the placements the window     SYSTEMS: centre, radius,
   ice cap, the climate mean    already carries)              strength, phase, drift
                                the season's phase, the       ~16 bytes each, <= 64 per
   NO MESSAGE. Both hosts.      sub-solar point, the day,     body, ~1 KB on change
                                AND THE PER-BASIN STAGE
                                OFFSET (rev 4, F13)           behind it, on the SHARD only:
                                NO NEW LANE (ruling S7-7)     an anomaly field, 0.98 MB on
                                                              the home body, 3.93 MB on an
                                                              Earth-like one, ~4 ms a step
        |                              |                              |
        +------------------------------+------------------------------+
                                       v
                    THE CLIENT PAINTS: cloud, rain, haze, lying snow,
                    a snow line that moves with the season, AND A RIVER
                    THAT RISES IN THE MELT.
                    THE REALM'S MEDIUM CARRIES THE WIND, so the parent's
                    existing drag reads it and a ship feels the storm.

   THE LAW: a season NEVER invalidates a chunk. The seed decides the shape and the
   permanent cover. The almanac and the storm list decide the PAINT and a thin DERIVED
   cover both hosts compute from the same rows. Only a cell a player's HAND changes
   becomes a record.
```

**Why lying snow is derived and not a diff.** COMPUTED (`05` §8.3): one snowfall over one square
kilometre, stored as per-cell diffs, costs 12 MB. Derived from the shipped system list, it costs nothing
and both hosts agree by construction.

**Why the river's stage is derived too (rev 4, critic A F13).** Discharge is the most seasonal thing in a
landscape: a snow-fed river doubles or triples in the melt, and a dry-season channel shows its bars.
Revision 3 shipped a season and froze the water inside it. The cure is the same shape of rule as the
lying snow: **ONE per-basin STAGE OFFSET, derived from the almanac and the basin's own snow fraction,
applied to the water surface `z_w` and to the drawn channel width.** It is one lookup and one add, both
hosts compute it from the same rows, and it changes no stored byte. D46 asks it.

**The dormant world.** A realm that sleeps ships no charter and no storm list, because nobody draws it.
On waking, the almanac needs no rule at all — it is a function of the tick — and the live offset's decay
is advanced in one step (`05` §8.4). A planet that slept through half its year wakes in the right season.

---

## 3. THE DATA: WHAT IS COMPUTED WHERE

### 3.1 The table

| Datum | Computed where | Size | Cost | Cache | In the world identity? |
|---|---|---|---|---|---|
| **The charter** (the body's physical facts, as integers) | the FOREST draws the body-local facts; the parent SYSTEM computes and quantises the orbit-derived ones and states them ONE HOP down; the body's realm holds all of them | ESTIMATED **42–76 bytes** packed (rev 4 adds the erosional age), against `SELF_LOOK_BUDGET_BYTES = 1 200` shared with the outline and the luma (`crates/core/src/look.rs:78`) | one draw per body, ever | the body realm's own store | **No.** A per-body charter may not ride the ONE world tag. It rides the PER-BODY digest and the per-realm surface refusal S7-3 already built (`05` §3.7) |
| **The macro layout** (L1–L3, L7) | inside the solve; it is the solve's initial surface | none kept separately | ESTIMATED 335 ns per column where a fine rung reads it closed-form (`02` §9.1) | — | its constants fold into `GENERATOR_VERSION` |
| **The macro artifact** (Z, water level, receiver + facies, discharge, the pyramid, the kept climate) | **the owning shard solves it; the client DERIVES the same bytes** (§7 D1) | **17.61 MB** on the home planet, itemised in §3.3; **4.8 KB** at the pyramid's top | 12–20 s of one core, 125–130 MB transient, ONCE EVER | the shard's own store (REQUIRED, not optional); the client's disk cache, **keyed by the FULL INPUT TUPLE — the universe seed, the body's address, the charter bytes and `GENERATOR_VERSION`** (rev 4, critic B B10) | its own digest, checked on arrival and after a cache read. **NOT folded into the boot self-check** — §5.4 |
| **The home planet's artifact** | **a committed, digest-pinned BUILD artifact**, in the `crates/terrain/src/home.rs` shape | ESTIMATED ~18 MB compressible, in the repository and in every client binary | zero at run time | linked in | its digest is pinned and cross-pinned, as `HOME_PLANET_RADIUS_BITS` is today |
| **The river graph and the lake table** | **DERIVED from the artifact's receiver chain and its water-level field. They keep no bytes of their own** (rev 4, critic B B1) | zero kept; a runtime index whose size is **UNMEASURED** and bounded by the channel-node count `M-L6` measures | a prefiltered per-column lookup, ESTIMATED ≤ 100 ns per column | with the artifact | with the artifact |
| **The kept climate field** | on a NESTED SUB-LATTICE of the macro lattice, inside the solve's schedule | **8 bytes per climate node — mean temperature, precipitation, seasonal amplitude, aridity — 1.23 MB on the home planet at a factor-4 sub-lattice.** The solve's own 16-byte working state is TRANSIENT | ESTIMATED ≤ 200 ms for the whole global grid on one core | with the artifact | with the artifact |
| **The fold** (the fine rungs) | both hosts, per column, per chunk | nothing kept | ESTIMATED +0.45 ms per chunk of column-pass arithmetic, plus an extraction that grows with the roughness | none — it is a pure function | the cell and mesh digests already cover it |
| **The vegetation skeleton** | both hosts, per chunk, from the biome | nothing kept; a hashed scatter | **UNMEASURED. `M-L24` prices it on one chunk BEFORE the owner answers D29** (rev 4, critic A F7) | none | a NEW **anchor digest**, beside the cell and mesh digests, because a tree a player stands on is part of the shape (`05` C-9) |
| **The paint table** (biome → material, water → material) | the client, from the biome the fold already derives | ESTIMATED under 2 KB of constants | one material lookup per chunk | none | no — it is style (ruling S6-5), and it moves no vertex |
| **The almanac** | both hosts, derived from the charter, the universe tick and the window's own placements | zero bytes | two operations per column, plus one per-basin stage lookup | none | no — it is time, and time never enters the recipe |
| **The live weather** | the body's own shard steps it (an anomaly field: **0.98 MB per home-size body, 3.93 MB per Earth-size body, ~4 ms a step**); it ships a LIST | ~16 bytes per system, ≤ 64 systems, ~1 KB per body on change | the STEP is now priced (rev 4, critic A F14); the LANE's rate is UNMEASURED (`06` U-L8) | the realm's checkpoint | no — it is live state |

### 3.2 The one number that decides the artifact's fate

The macro node is **8 224 m** on the home planet. The owner's 50 km reference frame is about six nodes
across, and one node is about **210 pixels wide** in a 1 280-pixel frame (COMPUTED, `06` §4.3).

**So nothing the player looks at in that frame is resolved by the solve.** The solve carries the
DRAINAGE SKELETON, the hardness classes, the bed dip and the ice classes. **What the player SEES is the
slope spectrum plus the closed-form terms keyed to that skeleton, under a light that casts a shadow.**
This is the most important sentence in the proposal, and it is why §6's first content slice builds the
spectrum with no solve at all — and why slice 8L fixes the light before any of it is judged.

### 3.3 The artifact's kept bytes, itemised (rev 4, critic B B1)

Revision 3 printed **14.75 MB** and kept "the climate field with the artifact" without counting it.
Critic B is right: D19's "ONE lattice" put a 16-byte climate cell on 2 457 600 macro nodes, which is
**39.3 MB** — 6.25 times what `05` priced on its own grid. Revision 4 fixes it three ways.

1. **D19 is restated. "One lattice" is a SCHEDULE claim, not a MEMORY claim.** The climate must be
   INSIDE the solve's loop, because relief makes wind, wind makes rain, rain makes discharge and
   discharge makes relief. It need not have the hydrology's resolution: climate varies over hundreds of
   kilometres, and the sub-grid detail is D20's MID and LOCAL terms, not the grid.
2. **The climate runs on a NESTED sub-lattice**: the macro lattice's per-face count divided by a stated
   integer, so a climate node contains exactly `f × f` macro nodes and the lookup is one integer shift.
   RECOMMENDED **f = 4**: 160 climate nodes per face on the home planet, **32 896 m per climate node**.
3. **What is KEPT is 8 bytes, not 16.** The solve's 16-byte working cell holds scratch; the fine pass
   reads four values — mean temperature, mean precipitation, seasonal amplitude and aridity.

```text
   RECOMPUTED (rev 4), the home planet, R = 3 350 759 m, N = 5 263 360
   ------------------------------------------------------------------
   macro nodes    6 x 640^2 = 2 457 600  x 6 B  = 14 745 600 B = 14.75 MB
   pyramid        five levels, 2 B                =  1 636 800 B =  1.64 MB
   climate        6 x 160^2 =   153 600  x 8 B  =  1 228 800 B =  1.23 MB
   river + lakes  DERIVED from the bytes above    =          0 B
   ------------------------------------------------------------------
   TOTAL KEPT                                     = 17 611 200 B = 17.61 MB
```

**The river graph keeps nothing.** Every channel is a chain of the receiver bytes already stored, and
every lake is a run of the water-level field already stored. The polylines and the lake table are a
RUNTIME INDEX built from those bytes. **Their size is UNMEASURED**, bounded by the channel-node count
that `M-L6` measures, and it is transient, not kept.

**What this moves.** D1's *"the artifact is one per cent of the mesh"* becomes **under two per cent**
(17.61 MB against 1.04–1.14 GB). D3's 5 140 m option rises from 37.7 MB to about **45 MB**. D31's ceiling
and A3's lane payload both read 17.61 MB, not 14.75 MB.

### 3.4 A correction the owner should see

`06` and `07` both price the artifact at **9.83 MB** (four bytes per node). `03` revision 3 re-counted
the bytes the solve must keep and reported **six bytes per node, 14.75 MB**. Revision 4 adds the pyramid
and the kept climate and reports **17.61 MB**. **Every "ship it" byte figure in `06` is therefore 1.8
times larger than that document states**, which makes the ship road weaker, not stronger, and the derive
road is what §7 D1 recommends anyway.

---

## 4. THE COARSENING LAW, TERM BY TERM, AND WHAT EACH BOUND IS WORTH

Revision 3 proved the prefix property for the octaves and claimed it for the whole fold. Critic A F4 and
critic B B8 are both right. Revision 4 states a rule and a bound for **every** term.

### 4.1 What is broken today

The crate drops one octave per rung BY COUNT (`crates/terrain/src/body.rs:251-259`) and bounds the
disagreement by the dropped amplitudes (`body.rs:266-277`). **That bound is bigger than half a cell at six
of the home planet's twelve rungs**: 32.8 m against 32 m at rung 6, and 1 469 m against 1 024 m at rung 11
(COMPUTED, `02` §6.2). **The ladder's own promise is broken today, and nobody was looking for it.**

### 4.2 The rule, per term

| The fold's term | Its coarsening rule | Its dropped magnitude | Status |
|---|---|---|---|
| **`Z`, the macro field** | rung-INDEPENDENT in its DEFINITION. Its READING is a pyramid level chosen by drawing distance (`03` §9 rule 2), so **it has a level step** | the pyramid's own level-to-level difference, which the pyramid can print per node when it is built | **DERIVED once the pyramid is built.** Revision 3's *"no crossfade to design for it"* is WITHDRAWN (critic A F4) |
| **The octaves** | an octave survives while `lambda(o) >= LADDER_SAMPLES_PER_WAVE * cell_m(rung)`, recommended 4 | `Lip(T) · r(dir) · sum of the dropped amplitudes`; RECOMPUTED as **3.3 cells of the rung times the per-octave slope of the finest surviving octave** | **DERIVED** |
| **The terrace** (the stratigraphic column) | the terrace step fades to zero over one rung once the bed thickness falls under `2 × cell_m(rung)` | the terrace's own fade step, which is **at most one bed thickness**, and the bed thicknesses are charter integers | **STATABLE. `03`/`04` assert it; revision 4 owes the derivation at 8d** |
| **The channel carve** | the carve stops at the rung where the channel's own width falls under `2 × cell_m(rung)`, and **it fades over the two rungs above that**, never in one step | at most **the channel depth the artifact states at that node** — hundreds of metres on a trunk valley, against a 32 m half-cell budget at rung 6 | **★ THE ONE UNSAFE TERM.** A two-rung fade is the RULE revision 4 adds; `M-L14` measures whether it is enough |
| **The 3-D removals** (8f) | the removal band is present only at rungs whose cell is under a stated fraction of the removal's own diameter, with the same two-rung fade | at most the removal's own depth | **RULE STATED in rev 4; it had none** (critic B B8) |
| **The roughness `r`** | **it does not coarsen.** It is one per-column factor with no rung argument | zero | **EXACT** |

### 4.3 What the honest statement now is

**A coarse rung's answer is a PREFIX of the fine rung's answer FOR THE OCTAVES, and every other term
states its own stopping rung and its own fade.** The total rung change is the sum of six named terms.
One is derived, one is exact, one becomes derived when the pyramid is built, and three are asserted with
a stated fade and a measurement that could refuse them.

```text
  |h(dir,0) - h(dir,L)|  <=   |Z_level(0) - Z_level(L)|        the pyramid step   (derivable)
                          +   Lip(T) * r * sum(dropped A_i)     the octaves        DERIVED
                          +   one bed thickness                 the terrace        asserted
                          +   the channel depth * fade(L)       the carve          asserted  <- the risk
                          +   the removal depth * fade(L)       the 3-D term       asserted
                          +   0                                 the roughness      EXACT
```

**In the game's words (critic B B8's own example, answered).** A pilot flies toward the home planet's
great valley. Under revision 3 the valley floor could rise two hundred metres under his nose at one rung
boundary. Under revision 4 the carve fades over TWO rungs, so the same two hundred metres arrives as two
steps of about 65 m each at a distance where one drawn pixel is 57.5 m — and `M-L14` says at what speed
that is still visible. **If it is visible, the carve's fade widens to three rungs, and that is the dial.**

### 4.4 The bound in cells, and why it is the right unit

RECOMPUTED (`04` §5.2): the rung change from the octaves, measured in cells of that rung, is about
**3.3 times the per-octave slope of the rung's finest surviving octave**.

| The per-octave slope | The bound, in cells | In pixels, conservative | In pixels, at the code's own drawable factor |
|---|---|---|---|
| 0.3 | 0.98 | 1.0 | 0.6 |
| 0.6 | 1.96 | 2.0 | 1.1 |

**The ratio does not depend on the screen's resolution**, which is why the gate is stated in CELLS.
**And the honest half.** A proposal that puts real roughness on the mountains **loosens** the bound
there and tightens it on the plains (`04` §0). That is the price of the picture.

### 4.5 The gate

The property test at `crates/terrain/src/height.rs:80-105` grows **one leg per term of §4.2**, not one
leg for the octaves (critic A F4):

1. the wavelength rule at every rung with every layer on, and its derived cut-off rung (`02` M7);
2. **a channel-following fixture**, because a channel covers about a thousandth of the surface and the
   existing 400-sample test would pass while measuring nothing (`03` M10);
3. a terrace-crossing fixture at a bed boundary;
4. a `Z` pyramid-level fixture at a level boundary;
5. a removal-band fixture at 8f.

---

## 5. THE NO-DRIFT GATE FOR THE MACRO MAP

**SL10 clause 3 is a MEASUREMENT, never an argument.** An iterative field on a bent grid adds failure
modes a closed form does not have. Each one gets a rule and a control that could fail.

### 5.1 The TWELVE determinism rules

| # | The rule | Why an iterative field needs it |
|---|---|---|
| 1 | **Every accumulator is `Gf` or an integer.** The macro height lives as an `i32` in 1/16 m | The fence bans `f32` by name, and **no control catches a plain `f32` add today** (`07` §12) |
| 2 | **The priority flood orders by `(integer height, node index)`** | Two nodes at the same height must pop in one order on every host |
| 3 | **The D8 receiver compares cross-multiplied integers** | A slope comparison by division is a float comparison |
| 4 | **Flats are resolved by an integer distance field**, never by an epsilon | An epsilon runs a continent's rivers in index stripes (`03` §4.4) |
| 5 | **Discharge accumulates in WHOLE SQUARE METRES**, weighted by the bend's own area density | A float sum's answer depends on its order |
| 6 | **`m = 1/2`, so `Q^m` is the `sqrt` ruling V1.4 grants by name. `n = 1`**, so the implicit update is one division and needs no inner iteration | A `powf` is banned; an inner iteration is a branch on a float |
| 7 | **The pass counts are STATED constants**, folded into `GENERATOR_VERSION` — never a convergence test | One drifted ulp makes one host run one more pass, and the two worlds differ (`06` L13) |
| 8 | **The solve is SINGLE-THREADED, and every reduction has a stated order** | A thread pool's reduction order is not a law |
| 9 | **Every transcendental is a COMMITTED LITERAL polynomial or table**, never computed at build time | A build-time `powf` puts the build machine inside the world identity (`03` D4) |
| 10 | **The macro lattice is CELL-CENTRED and divides `N` exactly**, so a rung-L cell finds its node by one integer division | `face_param` returns a cell centre in the OPEN interval; no node lies on a cube edge (`03` §1) |
| 11 | **The cube seam is crossed by construction**, on `crates/seed/src/seam.rs`'s 24 directed edges | A per-face C0 field puts **63 000 km of scarp** on the home planet (COMPUTED, `03` §1) |
| **12** | **★ THE LATTICE CHOOSER READS ONLY THE BODY'S OWN CHARTER INTEGERS AND COMMITTED CONSTANTS. It never reads the host's free memory, its core count or its measured speed** | If the chooser reads the host, two hosts pick two lattices for one body and the shape drifts. **This is a twelfth drift class and revision 3 had no rule for it** (critic B B16) |

### 5.2 The four gates the landform layer owes, beyond the seam's own

| Gate | What it asserts |
|---|---|
| **G-MACRO-EDGE** | The height and the slope agree across all twelve cube edges. The bound is **exact equality, ZERO quanta**, never a p99 — one column in the tail is one hole in the collider (`07` D-C6) |
| **G-MACRO-CORNER** | Every node has eight neighbours except the eight cube corners, which have seven. The interpolation inside the blend radius is exact |
| **G-MACRO-AREA** | The six faces' node areas sum to `4 pi R^2` within 1e-5 relative. **RECOMPUTED (rev 4, critic B B17.2): the bend's area weight runs from 0.702 at a face-edge midpoint to 0.758 at a cube corner — those two differ by 8 %, and both lie 24 % to 30 % below a face centre's 1.0.** Revision 3's "a 30 % spread" named the right magnitude for the wrong pair; the discharge must weight by the node's own area either way |
| **G-MACRO-SEAM** | No channel terminates on a face edge, and a channel polyline crossing a seam enters within one node |

These are the landform layer's OWN gates. `G-MAPPING-TABLE` and `G-MAPPING-ROUNDTRIP` are the geometry
seam's gates. **Revision 4 places the macro lattice BELOW the `GridMapping` seam** (critic B B4): it is
an addressing of the same cube-sphere the ladder already addresses, it is a physics anchor, and HR4's own
words put any physics anchor below the seam. So it owes `G-MAPPING-TABLE` and `G-MAPPING-ROUNDTRIP`
alongside the ladder, and the features ABOVE it owe G-IDENTICAL — see §6.6.

### 5.3 The legs, and the one that is not satisfied

`just terrain-pin` runs three host legs and `just terrain-legs` two. **The fifth leg is EMULATED
x86-64**, which ruling S5-5 and `justfile:753-755` both call a smoke test, with a real machine owed as
`D-TERRAIN-1` (🟧, `docs/design/DEFERRED.md:7721`).

**A forty-pass accumulation over 2.46 M nodes is exactly what an emulator is least likely to reproduce.**
This synthesis therefore carries `06`'s **L22**: a REAL x86-64 machine lands before the solve slice, or
the arc ships with *"no drift"* UNSATISFIED by the owner's own definition. That is an owner decision
(§7 D26), not ours.

### 5.4 What the BOOT gate folds, and what it must not

**The measured half of the world identity evaluates EIGHT home-planet chunks, and it costs 13.5 ms**
(MEASURED, `slice_05_generator.md:386`). Its two production callers are
`crates/bins/src/bin/gateway.rs:193` and `crates/bins/src/bin/client.rs:118`. **So the gateway — which
owns no planet and draws nothing — self-checks the home planet at boot, and so does every client.**

Once the height field reads a macro artifact, neither may do that by solving. The cure has four parts:

1. **The home planet's artifact is a committed build artifact**, so the eight golden chunks are evaluated
   on the SHIPPED PATH with the REAL artifact's REAL bytes. That is not a stand-in.
2. **A kernel canary** folds the solve's own arithmetic in microseconds, on a small named fixture
   lattice, about 1 ms (`07` D-C13). Two binaries that would draw different ground still state different
   numbers.
3. **The artifact's own digest is checked whenever it is read** — from the build, from a cache or from a
   lane — **and the cache's KEY is the full input tuple, not a content digest** (§3.1, critic B B10).
4. **A structural control proves the gateway holds no artifact.**

**What is refused:** folding the home planet's whole solve into the identity. It is 12–40 s in the
gateway, in every client, three times in `terrain-pin`, and again under coverage. That repeats the
measured 5.3 GB-at-boot defect in a new place.

### 5.5 The client's own ceilings (rev 4, critic B B11)

Revision 3 fenced the gateway and left the client unfenced. Three numbers become owner rows in D49:

- **A ceiling on CONCURRENT derives** on one client. Recommended **1**, because the derive is
  single-threaded by D38 and a second one only steals a core from the renderer.
- **A ceiling on RESIDENT artifacts.** Recommended **4 bodies**, which is 70 MB at 17.61 MB each.
- **The residency band's stated COMPLETION DISTANCE**, and the rule that sets it. The owner's ruling of
  2026-08-27 forbids capping the closing speed, so the required lead is `closing speed × derive time`.
  COMPUTED at a 40 s derive: a hull at 240 m/s needs **9.6 km** of lead; at 30 km/s it needs
  **1.2 million km**; at a warp leg's speed it needs a lead no window can carry.

**So the derive alone cannot serve a fast arrival, and revision 4 says so.** The answer is the two-stage
one already in the design: the **4.8 KB pyramid top** arrives first and is enough to draw the body at a
coarse rung, and the full artifact completes behind it. **If `M-L7`'s unpinned-body leg shows a black
frame or wrong ground at any closing speed, SL6 ask A3 is made and the realm ships the artifact.**
`M-L7` gains a third leg: **an arrival at a body whose artifact is NOT pinned**, because the home planet
is the one body where the problem cannot appear.

---

## 6. THE SLICE PLAN

### 6.1 The rule that fixes the order — CORRECTED (critic B B5)

Revision 3 said the free window closes at slice 9. **The foundation says it closes at slice 14**, in its
own words (`00_proposed_voxel_foundation.md:492-499`, rule 1): *"the stores of slices 9 to 13 are
THROW-AWAY until slice 14 lands and the world identity is pinned with the feature anchors in it, OR the
seed-placed feature anchors move into slice 5."*

**So the landform arc does not have to sit in front of slice 9. It has to close before slice 14.** That
changes four things revision 3 got wrong:

1. Nineteen to twenty-four weeks are no longer inserted in front of the store, the diff lane, the
   collider and the character.
2. **8f can land where it belongs**, after slice 11, because it ships placed rock objects with their own
   colliders and slice 11 owns colliders.
3. **Slice 18 can land where it belongs**, after slice 9, because it checkpoints the live weather in the
   realm's store.
4. **8e's ANCHOR DIGEST belongs with slice 14's identity pin**, which is exactly what rule 1 schedules.

**Slice 8 still goes first, with its scope unchanged.** It deletes `D-TERRAIN-3` (🟥, the one-rung dev
flag the owner refused: *"I don't want it to survive"*), and its machinery does not depend on WHAT the
field is, only on the field having rungs.

**Slice 8 gains two preconditions.** (a) The client must not build a chunk of a realm before that realm's
whole-body data is complete for that neighbourhood — the pyramid top at a coarse rung, the full artifact
at a fine one. (b) The residency band must STATE the completion distance of §5.5.

### 6.2 The arc

| Slice | Name | Lands before / after | What lands | Measured FIRST | The picture it earns |
|---|---|---|---|---|---|
| **8** | the ladder | before 9 | the tier rule, the crossfade, the residency band, `D-TERRAIN-3` deleted | the pop detector's baseline | — |
| **8p** | **THE INSTRUMENT** | before 9 | the picture PROBE (an aligned id + distance buffer, an AUTHOR plus a MESH INSTANCE), the STAMP, the RULER, the found stands, the nineteen verdicts, the tolerance from the body's own measured field | M8-2 (the probe moves no pixel of the colour frame), M8-1, M8-9 | the calibration picture |
| **8L** | **★ THE LIGHT (hours, not a slice)** | inside 8p | move the star off the camera's own azimuth (100°–140°), lower it to 12°–18°, and switch `shadows_enabled` ON for the terrain's directional light; re-shoot the three pictures the owner has already judged | the same three stands, before and after | **the answer to "is it the terrain or the light?" — for hours of work** |
| **8a** | **THE PICTURE, CHEAP** | before 9 | the SLOPE SPECTRUM, the ROUGHNESS FIELD on a stated placeholder, the LADDER'S WAVELENGTH RULE, ridged crests, the cap-rock bench, **and D6's relief law and the sea, moved forward from 8b** (critic B B7) | **`M1b` the skyline PREDICTION**, M1, M9 with its three stated bands, **`M-L25` the walkable / landable / buildable fractions** | **THE PILOT'S VISTA at 1.8 m, under a raking light. The first honest test of "not interesting enough", and it is cheap** |
| **8b** | **THE CHARTER** | before 9 | the charter as quantised integers; the missing draws (spin under TIDAL LOCKING, obliquity damped for close-in bodies, water inventory, optical depth, elastic thickness, the glacial offsets, **the EROSIONAL AGE**); the home planet's charter PINNED and cross-pinned | **M-L10** the charter's cross-host spread; **M-L8**; **U1 — is the home planet the right body?** | the ocean-fraction sweep, judged by the owner |
| **8s** | **★ THE SKY AND THE SHADOW** | before 9 | the atmosphere from the charter's scale height: aerial perspective (the blue haze), a sky dome, a cascaded shadow map on the terrain light, the fill light retired to a sky ambient | the frame cost of the shadow map and of the haze, against the client's own frame budget | **depth planes, a ridge that reads as a ridge, and the reference picture's own legibility** |
| **8c** | **THE SOLVE** | after 11 | the macro lattice; the climate in the schedule on its nested sub-lattice; D8; the priority flood; discharge; stream power; isostasy; talus **whose angle reads the aridity**; **ice**; **craters for an airless body**; the coast; the BED DIP; the twelve determinism rules; the four seam gates; the envelope; the pyramid; the artifact, its digest and its cache; the kernel canary | **M-L1** the pass schedule; **★ M-L13 does erosion move a DRY planet's statistics at all?**; **M-L12** the digest on four legs, one a REAL x86-64 machine | the body from orbit; a cube corner; a face seam; a glacial skyline beside a fluvial one |
| **8d** | **THE WATER AND THE ROCK** | after 8c | rivers, lakes, the clipped water sheet, the channel and floodplain carve **with its two-rung fade**, the sub-macro stream, the stratigraphic column at a FIXED RADIUS **tilted by the bed dip**, differential erosion, alluvium | **M2 / M-L5** the densest chunk with water AND the vegetation skeleton against 8 ms; **M-L6** the river field; **U-L12** the segment census | a river to the horizon; a lake; a coast; **a benched cliff, a mesa, a rock pillar**; **THE 50 km VISTA THE OWNER JUDGES** |
| **8e** | **THE BIOME AND THE PAINT** | after 8d | the Whittaker classifier at a derived CLIMATE RUNG; 19 biomes in five bits; the soil law; the snow line and the tree line as temperatures; sea ice; **★ THE PAINT TABLE — a biome-to-material map and a water material on the client** (critic A F2); what the vegetation skeleton reads; the anchor digest (pinned at 14) | **M-L22** the biome edge width; **M-L23** the fenced polynomials' error AS A BIOME DISPLACEMENT | the snow-capped ridge **in white**; forest at 5 km and at 50 km; **a desert behind a range, in desert colours**; the same latitude wet and dry |
| **8f** | **THE THIRD DIMENSION** | after 11 | the 3-D REMOVAL term in a thin band at the surface (arches, undercuts, a cave mouth in a cliff), **with a stated stopping rung and a two-rung fade**; PLACED ROCK OBJECTS as OBJECT-form catalogue kinds, seed-placed, inside the world identity, with slice 11's colliders | **M4-11** the undercut's cost, which is UNKNOWN today | the arch; the scree cone under a cliff |
| **18** | **THE ALMANAC AND THE WEATHER** | after 9 | the almanac DERIVED with no new lane, **including the per-basin river STAGE**; the live weather as a list of systems on its own lane; the anomaly field on the shard with its own memory ceiling; the client's cloud and rain on 8s's sky; wind in the realm's medium | **U-L8** the bytes and the rate; **the anomaly field's memory per awake body**. **The SL6 ask is not made before this measures** | a cloud deck; a storm on one side of the frame; the same ridge in summer and in winter; **a river in flood** |

### 6.3 Why this order

- **8L before everything**, because every picture the arc produces is otherwise taken with a known defect
  in its instrument. It is hours.
- **8p before 8a**, because `08` MEASURED that three of the last set's verdicts used a tolerance that
  **cannot fail**: `relief_bound_m` is 14 304.9 m at every rung that matters, which at 300 m of eye height
  is a 277 px tolerance against a 69.4 px signal, **so it PASSED the very picture the work exists to
  refuse** (`08` §0.2). An instrument that cannot fail measures nothing.
- **`M1b` before 8a**, because otherwise the owner funds nineteen weeks on a promise with no predicted
  number (critic A F3).
- **8a before the charter**, because it needs no ask, no lane, no artifact and no other domain, and it is
  the cheapest possible test of the owner's own sentence. **But D6's relief law and the sea move INTO 8a**
  (critic B B7), because the spectrum's amplitudes are anchored on the relief, and a YES the owner gives
  at 8a must not be re-drawn at 8b.
- **8s before the vista is judged**, because the reference picture's own legibility is aerial perspective
  and raking shadow.
- **8c and 8d after slice 11**, because the collider must agree with the shape and slice 11 owns the
  collider (critic B B5).

### 6.4 The picture protocol, carried from `08`

Every judged picture carries three things, and every one of them is measured, never typed.

1. **THE STAMP** — an on-frame readout and an aligned state file: the realm, the stand's address, the
   eye's altitude, the geometric horizon, the DRAWN horizon, the radius of terrain drawn, every rung
   drawn with its chunk count, **the star's elevation AND its azimuth relative to the camera's nose**,
   the biome under the eye, the height over the sea, the ruler's measured pixels, the world identity and
   the universe tick. One struct in `vd-devproto`, filled by `vd-client`, drawn by the renderer, asserted
   by the gate. **Three consumers, one source.**
2. **THE PROBE** — a second aligned buffer in which each pixel states WHAT drew it and HOW FAR it is,
   holding an AUTHOR and a MESH INSTANCE. It replaces the fitted colour classifier at
   `crates/bins/tests/terrain_pictures.rs:279`, which calls a pixel ground when
   `r >= g >= b && r >= b + 8 && r > 24`. **An author alone cannot refuse an impostor; the instance can.**
   **And with the paint table of 8e, a colour classifier stops working at all**, which is a second reason
   the probe must land first.
3. **THE RULER** — a subject of stated size in frame, measured in pixels against a prediction. The avatar
   marker is exact inside **145 m** (MEASURED, `08` §5.2). It is an INSTRUMENT, not the reference
   picture's scale figure.

**The stands are FOUND, never typed**: a satisfying predicate at a stated scale over a CO-PRIME stride of
ladder addresses, inside `vd-terrain` under the float fence, first hit wins. An argmax is refused.

**The flagship picture is the PILOT'S EYE at 1.8 m**, not a drone. A point at 50 km needs **323 m** of
prominence to clear the horizon plane, and at 100 km it needs **1 390 m** (MEASURED arithmetic on
`R = 3 350 759 m`, `08` §0.2). **A flat world then FAILS the flagship instead of being flown over.**

**Every tolerance comes from the body's own MEASURED field** — three times the body's own rms wobble at
the view length, which is four to six pixels at every scale. **No verdict may use `relief_bound_m`.**

**THE LIGHT IS NOW A STATED PART OF THE PROTOCOL** (rev 4). Every judged picture is shot with the star at
**12°–18°** of elevation and **100°–140°** from the camera's nose, with the terrain light's shadow map
ON. The stamp asserts all three. The two-light doubling stays dropped; one light, one shadow.

### 6.5 The reference picture, mapped to a slice — with the column revision 3 lacked

| What the reference picture shows | What produces it | Slice | Delivered by this arc? |
|---|---|---|---|
| Ridges a pilot can navigate by | the SLOPE SPECTRUM first, then the drainage skeleton and the carve keyed to it | 8a, 8c, 8d | **YES** |
| Carved valleys with floors and shoulders | stream power on the skeleton, then the closed-form valley | 8c, 8d | **YES** |
| A river running to the horizon | the river graph and its clipped water sheet | 8d | **YES** |
| A glacially sculpted skyline — cirques, arêtes, U-shaped troughs | the ICE pass. Above the snow line there is no liquid water, so stream power does not act there at all | 8c | **YES** |
| Cliffs and rock pillars | the stratigraphic column at a fixed radius, the bed dip, differential erosion; a free-standing pillar needs the 3-D term | 8d, 8f | **YES** |
| Snow caps on the ridges only | the lapse rate and the permanent snow line; the season moves it; **8e paints it white** | 8e, 18 | **YES** |
| Forest as a mass | the biome and the density at 8e; **the ASSETS at slice 14** | 8e + **slice 14** | **PARTLY — the skeleton is in the arc, the trees are not** |
| Blue haze with distance | the atmosphere's scale height from the charter, on the client | **8s (NEW)** | **YES, if D41 funds 8s** |
| Raking shadow that makes a ridge read | a cascaded shadow map | **8L + 8s (NEW)** | **YES, if D41 funds 8s** |
| Ground colour that differs by biome | **the PAINT TABLE (NEW)** | **8e** | **YES** |
| Clouds, weather that differs across the frame | the live weather list on 8s's sky | 18 | **YES** |
| A character for scale, standing upright | the character slice, and ruling **V11** (up is the server's) closes `D-TERRAIN-4` 🟥 | **slice 16 — OUTSIDE this arc** | **NO. Until 16, the stand's up is stated by the operator** |
| Detail to the horizon, no visible jump | the ladder, the crossfade, §4.2's per-term rules, the C1 macro read, the pyramid | 8, re-run after 8c, 8d, 8f | **YES** |
| Volcanoes | **nothing** | **D43 asks** | **NO, unless D43 says yes** |
| Dunes, karst, landslide scars | **nothing** | **D44 asks** | **NO** |
| Fields, roads, walls | built, not seeded. Live state | **out of this arc** | **NO** |

**Read the last column.** At the end of this arc the flagship frame holds shaped rock, water, a coast,
snow, a painted biome, a blue haze and a cast shadow — and **no tree, no figure, no volcano, no dune, no
field and no road.** §10.1 says which slice first makes a frame worth the owner's judgement.

### 6.6 HR4: the pair and the identical fixture, per slice (rev 4, critic B B4)

**The law:** *"every feature ABOVE the seam passes the identical fixture on ≥ 2 shard kinds
(G-IDENTICAL) or it doesn't land."* Revision 3 named no pair for any slice. The macro lattice is placed
**BELOW** the seam (§5.2), so it owes the seam's own two gates and not G-IDENTICAL. Everything ABOVE it
owes a pair.

**Q11 is answered, and the answer is the fixture** (critic B B4). **A hull or a station holds terrain and
holds NO artifact**, because it has no seed and no charter. So the fold reads a **NULL artifact**: `Z = 0`,
no channel, no water level, a stated default roughness and a stated default biome. **Every landform
feature must produce the same answer on a spherical planet chunk and on a Cartesian hull chunk given the
same fold inputs.** That is the identical fixture, and it is what makes the miner's trench the same trench
in both places.

| Slice | The PAIR | The identical fixture body |
|---|---|---|
| 8a | planet (`Spherical`) / station soil bay (`Cartesian`) | the same column inputs give the same octave sum, the same roughness and the same terrace, to the last quantum |
| 8b | planet / station | the same charter integers give the same relief bound and the same sea offset |
| 8c | planet / asteroid (`Cartesian`, airless) | **the airless leg exercises the CRATER path and the NULL-water path**, which is also the answer to Q14 (HR5's fluvial arms) |
| 8d | planet / hull soil bay | the carve and the terrace on a NULL artifact produce no channel and the same terrace |
| 8e | planet / station | the same climate inputs give the same biome, the same soil and the same paint row |
| 8f | planet / asteroid | the same removal band and the same placed-rock scatter |
| 18 | planet / station | a station with no atmosphere ships an EMPTY weather list, and the client draws no cloud |

**HR5's fluvial arms (Q14) are reached by the asteroid leg for the dry arms and by a Tier-A fixture fed a
stated macro field taken FROM a real body for the wet arms.** A fixture invents a test input, never a
world, so SL5 holds.

---

## 7. THE DECISION REGISTER

Each row is a decision only the owner may take. **Recommended** is this synthesis's answer, and the
"Why" says which domain's answer won where they disagreed. **D41–D51 are new in revision 4.**

### 7.1 The shape of the answer

| # | Decision | Options | Recommended | Why, and who won |
|---|---|---|---|---|
| **D1** | **Does the client DERIVE the macro artifact, or does the owning realm SHIP it?** | (a) both hosts derive; (b) the realm ships it on a bulk lane; (c) **derive, with the HOME planet's artifact a committed build artifact, a disk cache keyed on the FULL INPUT TUPLE, the 4.8 KB pyramid top first, and (b) kept as a measured fallback** | **(c)** | **`03` and `07` won; `06` lost.** `06` recommended shipping and did not weigh ruling **S7-2**: the client generates eight home-planet chunks AT LOGIN, before any realm has spoken, so no lane exists at that moment. `03` D3 pins the home artifact and the problem disappears. **The fallback is real**: if `M-L7`'s cold-login or UNPINNED-BODY leg fails, (b) returns and ask A3 is made |
| **D2** | Does the macro map come from a simulation, a bake, or a closed-form stack? | (a) plate simulation; (b) uplift + fluvial iteration on the fine grid; (c) pure closed form; (d) **a closed-form LAYOUT plus ONE coarse iterative SOLVE** | **(d)** | `02` refuses (a) and (b) on four structural grounds — not addressable, a second shape under the ladder, four ordering drift classes, a cost of hours. (c) cannot make a coast or a slope–area law. §5 states the twelve rules that make (d) lawful |
| **D3** | The macro lattice's resolution | 16 448 m (~3 MB, 3–5 s); **8 224 m (17.61 MB, 12–20 s)**; 5 140 m (~45 MB, 31–51 s) | **8 224 m**, confirmed ON PICTURES at 8d | `06`'s `2^E` rule is withdrawn by its own author: it gave a 2 454 m skeleton to an airless moon with no rivers and 19 544 m to an Earth-sized planet with rivers — the resolution ran backwards |
| **D4** | Is the lattice size chosen per body, or fixed? | (a) one node size everywhere; (b) **a METRIC TARGET of 8 224 m, and a chooser that walks to the nearest coarser exact divisor of `N` until a stated memory and time ceiling hold** | **(b), with determinism rule 12** | **RECOMPUTED (rev 4, critic B B16): 8 224 does not divide the Earth-like candidate's `N = 10 235 904` at all.** Its nearest legal node sizes are **7 616 m and 8 704 m — within 7 %, not the ±20 % the critic estimated**. But its node COUNT at 8 704 m is 8 297 856, which is **49.8 MB and about 58 s**. To hold an 18 MB ceiling that body's node walks to **14 336 m**. **A big planet gets a coarser landscape, and the owner should know that before it ships** |
| **D5** | **The octave table: ONE law, owned by ONE document** | (i) keep three proposals; (ii) **the merged law of §7.2** | **(ii)** | `04` D4-18 states the problem plainly: `body.rs` cannot hold two tables and three documents each propose one. **Nothing else in this arc is safe until it is settled** |
| **D6** | The relief law, **and it lands at 8a, not 8b** | (a) keep `0.4 % of radius, capped 200–12 000 m, x [0.5, 1.5)`; (b) **`draw x min(strength bound, shape bound)`, `draw` in `[0.5, 1.0]`** | **(b), landed WITH the spectrum at 8a** (critic B B7) | The strength bound `sigma_y/(rho_c*g)` calibrated on Everest gives Mars 94 % of Olympus Mons and Venus 112 % of Maxwell Montes. The shape bound at `0.077 R` comes from Vesta, an OBSERVED small body; without it every small round body is deleted. **The spectrum's amplitudes are anchored on the relief, so a vista judged at 8a under the OLD relief is a vista 8b would re-draw** |
| **D7** | Does the roughness stay one number per body? | (i) keep one `k_rough`; (ii) **modulate it per column, DOWNWARD only, from the macro state AND from the column's ARIDITY** | **(ii)** | One number gives 19° everywhere or 2° everywhere. A downward-only factor keeps the ladder bound exact because it never scales up. The aridity term is new in rev 4 and is D45 |
| **D8** | Does the arc land an **ICE pass**? | (a) yes: an ice mask from the equilibrium line, trunk over-deepening, cirque marking; (b) no | **(a)** | The reference picture's top third is a glacial skyline. **Above the snow line there is no liquid water, so stream power does not act there at all.** `06` calls this the single largest believability gap the refutations found |
| **D9** | Does the arc land a **CRUST DICHOTOMY** — two crust types at two densities? | (a) yes; (b) no | **(a)** | Earth's two-humped hypsometry comes from light continental crust and dense oceanic crust floating at different levels. Without it there is no shelf, no abyssal plain and no stable coastline. **The hand-tuned ocean fraction is the symptom of the missing mechanism, not its cure** |
| **D10** | Do the strata become a **stratigraphic column at a fixed RADIUS**? | (a) yes for the sediments and the bedrock; the soil stays depth-following; (b) keep today's depth-following cake | **(a)** | MEASURED (`crates/terrain/src/strata.rs:187-210`): the table selects a substance by DEPTH UNDER THE SURFACE, so every layer drapes over the hill and the order is soft over hard. **A mesa and a hoodoo are a HARD cap over a SOFT layer, so today's table makes both impossible at any erosion rule** |
| **D11** | Does an airless body get **craters** instead of rivers? | (a) yes; (b) no | **(a)** | **A 100 km moon with smooth noise and no crater is not a believable airless body.** It is also the asteroid leg of HR4's pair and the answer to HR5's dry arms |
| **D12** | Who owns the **overhang, the arch and the pillar**? | (a) nobody; (b) a 3-D removal term in a thin band at the surface, in its own slice | **(b), at slice 8f, AFTER slice 11** | A single-valued `h(dir)` cannot make one, and the reference picture holds four kinds |
| **D13** | Where does the **1 m to 8 m** relief come from? | (a) more octaves; (b) **a fine floor plus PLACED ROCK OBJECTS with their own colliders** | **(b)** | Talus, boulders and outcrops are landform, and ruling V4's *"trees, grass and decoration"* does not cover them — they need a collider. **Every cliff in the reference picture stands on a scree cone** |

### 7.2 D5 in full — the merged octave law

**One table, four owners, no overlap, and ONE count rule** (critic B B6 fixed).

| The part | The rule | Who proposed it |
|---|---|---|
| **The ends** | The table runs from `C * cell_m(the rung below the macro node)` down to `C * cell_m(0)`, with `C = 8`. **`LONG_WAVE_CAP_M` and `SHORT_WAVE_M` are both deleted** — the ladder decides, and no metre is typed | `04` D4-3 |
| **★ The count — ONE RULE, and the saving IS credited** | `Z` REPLACES the coarse octaves. **RECOMPUTED (rev 4): on the home planet the table runs from `8 × 2048 = 16 384 m` to `8 × 1 = 8 m`, which is `log2(16384/8) = 11` octaves against the 14 the crate has today** (VERIFIED, `crates/terrain/src/home.rs` test `octave_count == 14`). **So the count FALLS from 14 to 11, and the column pass must be re-priced with that saving credited.** Revision 3 carried `06`'s "the count does not fall" beside `04`'s ends, and the two contradict | `04` owns the ends; rev 4 settles the contradiction |
| **The floor** | The finest wavelength is between **2 m and 8 m**, and a PICTURE decides which. Below that band the fine floor, the surface form and the placed rock objects own the detail, never more octaves | `02` D3 wants 2 m; `04` wants 8 m. **UNMEASURED — `M12` and `M16` decide** |
| **The amplitudes** | A **SLOPE SPECTRUM** peaked in the band the eye reads, anchored on the **ANGLE OF REPOSE** — never a normalisation that makes the sum equal the relief | `03` §4.2 |
| **The budget** | Every amplitude is capped by the ladder's own **half-cell** budget at the rung where it disappears | `02` §4.6 |
| **The modulation** | **ONE per-column factor `r(dir)` in `[r_min, 1]`, downward only**, read from the macro state and **capped by the column's ARIDITY** | `04` D4-4, `02` D2, `03` §4.2.4, and rev 4's D45 |
| **The middle band** | Between the hillslope length and the macro node the octave is **RIDGED** (`1 - |noise|`) and sampled in a frame **stretched along the stored D8 flow direction** | `04` D4-5 |
| **The clamp** | **The "never fewer than one octave" clamp is removed** and the `M-16` test is rewritten in the same slice, because at the top rung `Z` carries the shape and one 8 192 m octave against a 2 048 m cell aliases | `04` D4-15 |
| **The coarsening** | **By WAVELENGTH, four samples per wave** (§4.2) | `03` §7.1 |

**The trap this merge closes.** MEASURED at `crates/terrain/src/body.rs:180-184`, the amplitudes are
re-normalised so their SUM is the relief. **Shortening the coarsest wavelength without changing that rule
makes the whole world eight times steeper** — 56° at a 2 m baseline. Both refuters of `02` found it. The
slope spectrum and the budget together are what make the change safe, and **they are ONE decision, not
five.**

### 7.3 The facts, the climate and the weather

| # | Decision | Options | Recommended | Why, and who won |
|---|---|---|---|---|
| **D14** | **How do the body's physical facts reach the recipe?** | (a) floats the recipe snaps; (b) **quantised INTEGERS, authored ONCE and STORED, never re-derived**; (c) every host derives them independently | **(b)** | **`02`, `03` and `06` won; `05` lost.** `crates/physics` **has no float fence** — there is no `crates/physics/clippy.toml` (VERIFIED) — and its own doc says *"the cross-host bit-equality gate is SPIKE-6a"*. Two shards on two architectures can quantise either side of a grid line. And **a realm rescheduled from an M4 host to an x86-64 pod would re-address every stored edit** |
| **D15** | **Who authors which fact?** | (a) the parent authors all; (b) the body's realm authors all; (c) **the FOREST draws the body-local facts; the parent SYSTEM computes and quantises the ORBIT-derived facts and states them ONE HOP down; the body's realm holds the whole charter and states it in its own self-look** | **(c)** | This is `06` L16(A) and it satisfies `02`, `03` and `05` at once. It obeys SL1 clause 1, SL1 clause 4 (one hop) and SL3 |
| **D16** | **Which facts, and how many bytes?** | 13 (`02`, `06`), 24 (`05`), or the union | **the union — about 19 to 21 integers, ESTIMATED 42–76 bytes — MEASURED against the 1 200-byte self-look budget before it lands** | The union is: gravity, bulk density, escape velocity, mean insolation, equilibrium temperature, optical depth, atmosphere scale height, mean molecular weight, the liquid-water flag, the solid-surface flag, spin period, the obliquity's COSINE, eccentricity, star class, water inventory, effective elastic thickness, two glacial equilibrium-line offsets, the year, **and the EROSIONAL AGE** (rev 4, D-age). **The obliquity crosses as a COSINE and never as an angle**, because `sin` and `cos` are compile-fail inside the fence |
| **D17** | **Who draws the facts that do not exist, and IS THE DRAW CONDITIONED?** | (a) the forest, unconditioned; (b) **the forest, CONDITIONED BY PHYSICS**; (c) the recipe | **(b), and the draw laws are WRITTEN AND TUNED before the store's identity is pinned** | **Rev 4 adds the conditioning** (critic A F9): the **TIDAL LOCKING** test from the semi-major axis and the star's mass sets the spin where it binds, and the obliquity's draw is DAMPED for close-in bodies. `05` §9 already designs the tidally locked world. **In the game's words: the pilot approaches the close-in planet of the home system; its charter says the day is nine hours; its star should stand still in its sky forever.** The home planet's drawn values are pinned in `home_body_pin.rs` the way `POLE_AXIS` is pinned today |
| **D18** | **`POLE_AXIS` does not move** | (a) rotate the pole by the obliquity; (b) **keep `POLE_AXIS = +Z` and read the obliquity as a SCALAR** | **(b)** | Obliquity is the angle between the spin axis and the ORBIT's normal — a relation between two frames, and **the parent already owns that relation**. The `POLE_AXIS` pin at `crates/bins/tests/home_body_pin.rs:44-60` survives |
| **D19** | **ONE grid or two?** — RESTATED (critic B B1) | (a) two independent grids; (b) one lattice for both memory and schedule; (c) **ONE SCHEDULE, and the climate on a NESTED integer sub-lattice of the macro lattice, factor 4** | **(c)** | The physics is a LOOP, so the climate must be IN the schedule. But climate varies over hundreds of kilometres and the sub-grid detail is D20's MID and LOCAL terms. **(b) would cost 39.3 MB of climate on 14.75 MB of hydrology.** (c) costs 1.23 MB kept |
| **D20** | **Precipitation at how many scales?** | (a) the grid alone; (b) **three: a GRID term, a MID term one coarse evaluation 5–15 km upwind, and a LOCAL term from the column pass's own halo** | **(b)** | A grid alone puts the whole 50 km vista in one climate cell, so **the picture would be contour bands**. The three terms give a green windward valley beside a dry leeward cliff at the same height. ESTIMATED 0.35 ms per chunk |
| **D21** | **The CLIMATE RUNG** — the classifier reads the height at ONE derived rung | (a) yes; (b) a tolerance in metres | **(a)** | A height bound in metres cannot bound a CLASS: one column 20 m under the snow line at rung 0 and 20 m over it at rung 8 flips from Grassland to Ice, **and the white patch moves as the pilot flies in**. That is an SL8 arrival pop that exists TODAY (`crates/terrain/src/height.rs:39-65`) |
| **D22** | The biome set | (a) keep four; (b) **nineteen in five bits: twelve from the Whittaker chart and seven overrides** | **(b), and the owner names the list** | The vista needs forest, taiga, savanna, wetland, beach and ice. **The cell record does not change** — a biome is a COLUMN property both hosts derive |
| **D23** | **The three-layer weather split** | (a) static + live; (b) **static CLIMATE + a DERIVED ALMANAC + LIVE weather** | **(b)** | Without the almanac the owner's word *"trajectory"* moves an annual mean by 1.5 % and nothing else moves at all. A season moves a snow line by kilometres |
| **D24** | **Does the ALMANAC need a new lane?** | (a) ship it as a small row; (b) **derive it, with no new lane** | **(b)** | **`05` won; `06` lost.** Ruling **S7-7** — *"One directional light from the brightest luminous row in the window"* — has the client derive the light already. The almanac is that rule read a little further |
| **D25** | **What does the LIVE weather ship?** | (a) a coefficient field; (b) a per-observer window; (c) **a SHORT LIST OF WEATHER SYSTEMS, one statement per realm** | **(c)** | (a) is COMPUTED at **393 KB per statement** for a 50 km front, against a 1 200-byte bag. (b) is the composed-per-observer shape the reach ruling killed by name. (c) is ~16 bytes per system, ≤ 64 systems, ~1 KB per body on change |
| **D26** | **Does a REAL x86-64 machine block the solve slice?** | (a) yes; (b) no | **(a)** | Ruling S5-5. **A forty-pass accumulation over 2.46 M nodes is what an emulator is least likely to reproduce** |

### 7.4 The world, the budget and the pictures

| # | Decision | Options | Recommended | Why |
|---|---|---|---|---|
| **D27** | **★ IS THE HOME PLANET THE RIGHT BODY?** | (a) keep it; (b) **re-pick by the existing `earth_like` predicate, AFTER U1 measures it** | **(b), and U1 runs FIRST** | Seed 2298's earth-like body is 1.023 R⊕ = **6 515.5 km** (MEASURED, `crates/physics/src/worldgen/census.rs:57-63`). **The voxel home planet's look radius is 3 350 759 m — half that, and a different planet** (`crates/terrain/src/home.rs:22`). `home_body` returns the first planet the LADDER accepts, not the earth-like one (`crates/bins/src/lib.rs:987-1003`). ESTIMATED that body is airless and hot. **UNMEASURED, and it is one bench** |
| **D28** | **The ocean fraction** — **rev 4 proposes a number** (critic B B13) | today about **one column in a hundred** (MEASURED, `slice_05_generator.md:388-392`); Earth is 71 % | **a target of 55 %–70 % of the surface under water, and the water inventory's draw range set so the MEDIAN body lands in it** | This is a QUALITY choice and it is dressed as physics if it is not said plainly. **The number the owner can judge**: an Earth-like 71 % needs a sea offset of +1 270 m, which about 30 % of the world's bodies already draw — **the home planet drew a puddle**. And a 1 %-ocean world returns **desert everywhere**, because two of the four precipitation terms read the sea |
| **D29** | **The 8 ms budget, now that the ROCK ALONE is estimated OVER it** | (a) keep it and shrink the work; (b) raise it | **decide when `M-L5`, `M2` and `M-L24` are in hand; and shrink FIRST, by FOUR dials** | **RESTATED (critic B B2): the densest measured chunk is 7.09–8.07 ms with NO water and NO trees.** The dials, in order: **(1) a per-chunk VERTEX BUDGET, or a roughness cap keyed to the cell, or a coarser rung for the cave band — the only dial that touches the ROCK, and it is new in rev 4**; (2) mesh the water sheet only where a water level exists in the column field's own range, and cache it across the radial chunks of one column stack; (3) mesh the water at a COARSER rung than the rock; (4) `SUBDIV = 3`, worth only about 0.2 ms. The owner's own words stand behind (b): *"I think 8 should be ok. If that over time will become a problem, we can rethink and reimplement"* |
| **D30** | **Whose budget is the 8 ms?** | (a) the server's; (b) the client's; (c) **both, with the client's own measured baseline** | **(c)** | The constant lives in a server example, and the client's measured chunk is **4.18 ms** against the server's 3.30 ms. A missed budget is an arrival RATE on the server and a STALL on the client |
| **D31** | **The client's mesh residency ceiling** — **rev 4 proposes a basis** (critic B B13) | this project has never stated one | **state it as a MULTIPLE of what a client holds today: MEASURED 205 MB for a 13×13×3 patch at rung 0 (`slice_07_client_link.md` M7-3). Recommended ceiling 1.5 GB of mesh plus 100 MB of artifacts, and `M-L19` reports against it** | The owner's own 50 km vista holds **1.04–1.14 GB of mesh, rock and water only**. Without a stated ceiling `M-L19` cannot go red, and its kill row fires *"fewer rings"*, which the same table admits is a detail-by-distance seam |
| **D32** | **The crease** — does a cliff keep its edge? | (a) surface nets, rounded by half a cell of the rung drawn; (b) dual contouring | **decide at slice 8d's pictures, while it is still free** | Ruling V10 reserved it. Slice 7's three pictures held no cliff to judge it on |
| **D33** | **Do rivers, beaches and deltas carry ore?** | (a) no; (b) yes | **(a) no** | A seed-derived map of value is a treasure map, refused by name by the 2026-08-27 ruling. Sand, gravel and clay are bulk stock and stay lawful. A placer bar is refused |
| **D34a** | **An ORE map keyed to the layout?** | (a) no; (b) yes | **(a) no** | A published ore map is a 20× prospecting advantage for anyone who reads a wiki. **The seed ruling refuses it by name** |
| **D34b** | **★ A ROCK map keyed to the layout — the rock TYPE, its colour and its bed DIP?** (SPLIT in rev 4, critic A F11) | (a) no rock map, one sediment and one bedrock per planet; (b) **a rock map: the layout's own plate class chooses the palette per province, and the orogen field publishes a BED DIP** | **(b)** | MEASURED (`crates/terrain/src/body.rs:194-201`): a body draws **one** sediment of three and **one** bedrock of five for the WHOLE planet. **One palette for the whole world, forever.** And D10 makes the beds horizontal everywhere, so a range built of flat beds reads as Monument Valley and never as the Alps — while the reference picture is an alpine range. **The seed ruling forbids a map of VALUE, not a map of LOOK.** A craton shows crystalline basement, an orogen shows steeply dipping rock, a passive margin shows flat carbonate, a rift shows basalt. **The bed dip is one dot product per column** |
| **D35** | **Does the landscape change over game time?** | (a) no, static for this arc; (b) yes, as live state | **(a)** | SL10 forbids time inside the recipe. (b) stays possible later as a diff family |
| **D36** | **When is a body's first solve paid?** | (a) at first demand, on the owning shard, cached; (b) offline at world creation | **(a), with (b) for the home system** | A cold first boot of the home planet's realm answers no chunk for 12–40 s, and the home planet is where every gate and every new player starts. **Caching the solved artifact in the realm's own store is REQUIRED, not optional** |
| **D37** | **The solve's pass counts** | (a) stated constants with `dt = age / PASSES`; (b) a convergence test at run time | **(a)**, and **the AGE is a CHARTER DRAW, not a world constant** (rev 4, critic B B14) | One drifted ulp makes one host run one more pass. Naming a physical age makes the pass count a RESOLUTION rather than a dial. **And an AGE is exactly the kind of number the no-magic-numbers law makes a body fact: a young world's sharp unworn ridges and an old world's rounded stumps are free variety.** `M12` proves the resolution claim: solve at `PASSES` and at `2 × PASSES` with the SAME age and assert the two fields agree |
| **D38** | **May `vd-terrain` use a thread pool?** | (a) no: single-threaded, the HOST parallelises whole bodies; (b) yes | **(a)** | `vd-terrain` has ONE dependency today and ships as a `staticlib` another engine links. A new library is the owner's to approve. **What (a) costs, named: the 12 s solve stays 12 s** |
| **D39** | **Where does the artifact's cache live?** | (a) inside `vd-terrain`; (b) **`vd-terrain` stays pure and exposes `solve(seed, body) -> Artifact`; the HOST owns the cache, its residency and its eviction** | **(b)** | A cache inside a Tier-A `staticlib` would put 100 % coverage out of reach and would stop the crate linking into another engine |
| **D40** | **The picture protocol** | (a) keep today's colour classifier and typed numbers; (b) **the stamp, the probe, the ruler, found stands, tolerances from the body's own measured field, AND THE STATED LIGHT** | **(b), and it lands BEFORE the first content slice** | Three of the last set's verdicts used a tolerance that **cannot fail**, and one **passed the very picture the work exists to refuse**. **And the paint table of 8e breaks a colour classifier outright** |

### 7.5 THE ELEVEN NEW ROWS OF REVISION 4

| # | Decision | Options | Recommended | Why |
|---|---|---|---|---|
| **D41** | **★ WHO OWNS THE SKY AND THE SHADOW PASS?** | (a) nobody, and every arc picture is judged unlit and hazeless; (b) **slice 8L (hours, in 8p) fixes the harness's own light, and NEW SLICE 8s lands the atmosphere and a cascaded shadow map** | **(b)** | **The fourth measured cause of "no details at all" is the light.** VERIFIED: `shadows_enabled: false` on both lights, `FILL_SHARE = 0.08`, the star 25° up, and the camera's nose pointed at the star's own azimuth. **A 300 m spur and a 3 m bump look the same on a ridge lit from behind the camera.** The reference picture's whole legibility is a low raking sun, a cast shadow and aerial perspective separating four depth planes. **8L costs hours. 8s costs 2 weeks, and without it the arc's own flagship comparison cannot be run** |
| **D42** | **★ WHO OWNS THE GROUND'S COLOUR?** | (a) nobody — one material for the whole world; (b) **a PAINT TABLE in slice 8e: a biome-to-material map and a water material, with verdict V7 as 8e's landing condition** | **(b)** | MEASURED: the client draws all terrain with ONE material, `Color::srgb(0.55, 0.50, 0.42)` (`crates/client-render/src/terrain.rs:362`), and the word "biome" occurs nowhere in `crates/client-render/src` or `crates/client/src`. **8e's four named pictures — a snow-capped ridge, forest, a desert behind a range, the same latitude wet and dry — are ALL COLOURS.** With one brown material the snow cap is brown and the river is brown rock in the shape of a river |
| **D43** | **★ DOES A VOLCANIC FAMILY LAND?** | (a) never; (b) a later arc; (c) **this arc, as a closed-form radial term on the plate field: a cone, a caldera, a lava plain, a neck** | **(c) if the budget allows; (b) otherwise, and SAY SO** | **The body already draws Basalt, Gabbro and Andesite as bedrocks** (`crates/terrain/src/strata.rs:136-142`), so **the world asserts a volcanic history it never shows, and the rock is a lie.** `03` calls volcanic *"the second landform family after fluvial on an earth-like planet"*. **A cone and a caldera are cheaper than the ICE pass the arc already buys** |
| **D44** | **AEOLIAN, KARST AND MASS WASTING** | (a) never; (b) **named as NOT DELIVERED, and registered for a later arc** | **(b)** | The world has `Biome::Desert` with a `Sand` topsoil and no dune; it draws `Limestone` and has caves and no sinkhole. **Telling the owner before he pays is the point of the row** |
| **D45** | **★ DOES THE CLIMATE REACH THE SHAPE, OR ONLY THE PAINT?** | (a) only the paint — the roughness reads the macro slope alone; (b) **the roughness ceiling AND the talus angle read the column's ARIDITY** | **(b)** | `05` §7.3 calls this *"the deepest believability gap in this document"*: *"a desert is a grassland with a sand-coloured topsoil byte."* **A geologist tells an arid landscape from a humid one in one photograph at any distance**: arid gives sharp scarps, mesas, angular crests and no soil mantle; humid gives convex rounded hilltops and soil-mantled slopes. **The climate grid is already read per column for the biome, so the cost is one lookup and one multiply — the cheapest believability in the whole arc** |
| **D46** | **★ DOES THE ALMANAC REACH THE WATER?** | (a) no — a static water surface all year; (b) **one per-basin STAGE OFFSET, derived from the almanac and the basin's snow fraction, applied to `z_w` and the drawn channel width** | **(b)** | **Discharge is the most seasonal thing in a landscape.** Revision 3 moves the snow line up the range in spring and leaves the river below it the same width, depth and colour all year. (b) is the SAME shape of rule as the lying snow, derives on both hosts and changes no stored byte |
| **D47** | **★ DOES THE ARC KEEP THE AUTHORING DOOR OPEN?** | (a) no; (b) **yes, as `04` §8's two kinds: kind A an authored delta on a coarse pyramid rung, applied ON TOP of the fold, LIVE STATE, never in the identity (ruling S5-3); kind B an authored landform held in the generator as a pure function of the address, evaluated INSIDE the fold so the octaves roughen it, a WORLD-EPOCH change that enters the identity** | **(b), the law stated now, the mechanism built when the owner names a landform** | **The reference picture is a hand-authored world; every ridge in it was placed by a person.** Our answer is that the seed and the water produce the ridges instead, which is right for a galaxy — **but the owner will want to place ONE named canyon**, and the mechanism that lets him is designed in `04` §8 and was dropped between two documents |
| **D48** | **★ SL10 CLAUSE 1 IS WIDENED, AND ONLY THE OWNER MAY WIDEN IT** | (a) refuse the charter and keep `f(seed, address)`; (b) **accept `f(seed, address, STORED CHARTER)`** | **(b), and the owner rules it explicitly** | D14 and D15 make the shape a function of stored state. **Two consequences revision 3 did not draw.** (i) `TAG_SURFACE` rides `BodyStmt::SelfLook`, which **only a RUNNING realm may send** (VERIFIED, `crates/core/src/look.rs:52-56`), so **no body's shape can be derived before its realm runs and speaks**, and nothing can be warmed ahead for a dormant realm. (ii) **D17's tuning re-authors every stored charter and therefore re-shapes every stored body** — a one-way door of its own |
| **D49** | **★ THE CLIENT'S THREE CEILINGS** | the client derives with no ceiling at all | **state three: at most 1 concurrent derive; at most 4 resident artifacts (~70 MB); and a residency-band COMPLETION DISTANCE of `closing speed × derive time`, served by the 4.8 KB pyramid top first** | §5.4 fences the GATEWAY and revision 3 fenced nothing on the client. COMPUTED at a 40 s derive: a hull at 240 m/s needs 9.6 km of lead, at 30 km/s 1.2 million km. **That is a new time constant of exactly the class the suit ruling deleted for the walk**, and the two-stage arrival is what keeps it out of the player's way |
| **D50** | **★ WHAT DOES THE ARTIFACT OWE A SURFACE A PLAYER CHANGED?** | (a) re-solve; (b) **nothing, and the local diff wins inside its own width** | **(b), and it is stated in the acceptance document so a player does not find it first** | A player raises a ridge across the valley below his berth. The artifact is not re-solved. **The channel still runs where the seed's water ran, straight through his new rock; the water sheet still stands at the old level, so his dam holds nothing back and his new lake never fills.** (a) is a 12–40 s solve on a player action and is refused. **Q10's rain shadow was the small half of this question** |
| **D51** | **★ WHICH GATE TIER DOES A BELIEVABILITY GATE JOIN?** | (a) `just gate`; (b) a hand-run gate; (c) **a RATCHET tier with a stored baseline, a named owner for the baseline, and a stated rule for who may move it and on what evidence** | **(c)** | VERIFIED (`justfile:408`): `gate` runs `terrain-pin`, `terrain-link-scan` and `terrain-fence-control` and **not** `terrain-cost`; `terrain-legs` needs Docker and sits outside on purpose. **So the project already has three tiers and no written rule for which tier a new gate joins.** The arc turns nineteen picture verdicts, a slope histogram, a pop detector, a vista census and a drainage-density band into gates. **A believability gate that blocks every merge will be switched off, and a gate nobody runs is not a gate** |

### 7.6 The conflicts, and how each was settled

| The conflict | Who said what | Settled |
|---|---|---|
| **Derive or ship the artifact** | `03` D3 derive; `07` D-C20 derive; `06` L1 ship | **Derive** (D1), with the pyramid top first and a full-input-tuple cache key. `06` did not weigh ruling S7-2 |
| **Does the server need a charter message?** | `05` no; `02`/`03`/`06` yes | **Author once, store, state** (D14). `crates/physics` has no float fence |
| **Who authors the charter** | `02`/`05` the body's realm; `06` the parent | **Both, split by fact** (D15) |
| **How many charter facts** | 13 / 24 / 28 / 52 / 64 bytes | **The union, ESTIMATED 42–76 bytes, MEASURED before it lands** (D16) |
| **The octave table** | four proposals | **The merge in §7.2**, with **ONE count rule: `Z` replaces the coarse octaves, the count falls from 14 to 11, and the saving is credited** (rev 4 settles `04` against `06`) |
| **Two valleys cut twice** | `02` L4a's multifractal weight against `03`'s channel carve | **One valley, cut once.** The carve owns the channel and its floodplain; the multifractal weight is damped to zero inside the carve's own width, through the same factor `r`. The ridged band reads `03`'s stored D8 direction byte |
| **The artifact's size** | `06`/`07` 9.83 MB; `03` 14.75 MB | **17.61 MB, itemised** (§3.3). `06`'s ship road is 1.8× weaker than it states |
| **One climate grid or two** | `05` a climate grid; `03`/`07` a hydrology lattice; rev 3 "one lattice" | **One SCHEDULE, a nested factor-4 sub-lattice, 8 kept bytes** (D19) |
| **The almanac's lane** | `06` L-6 ask for a row; `05` derive it | **Derive** (D24). Ruling S7-7 has the client derive the light already |
| **The weather's shape** | `05` a system list; `07` D-C15 a per-observer window | **The system list** (D25) |
| **The vista's chunk count** | `07` ~4 200 chunks; `08` 2 805 chunks | **RECOMPUTED (rev 4): they are TWO estimates under two residency rules, not a range.** `08`: 2 805 × 405 KB = 1 137 MB. `07`: 4 200 × 248 KB = 1 042 MB. More chunks giving fewer bytes is not a range |
| **The per-chunk cost** | `07` reports a FAIL at 8.07 ms; rev 3 printed a pass | **`07`'s fail is carried** (§1.4, D29) |
| **Five octaves or six inside the solved field** | `07` §5.3 five by Nyquist; `03` six | **UNMEASURED.** The difference is the 12.5 km octave and its 171 m of amplitude. One document must give way before either is built |
| **The free window** | rev 3 slice 9; the foundation slice 14 | **Slice 14** (§6.1). The arc no longer sits in front of the store |
| **The lattice divisor** | `07` §5.1's 1 396 per face | **REFUSED. RECOMPUTED: `10 235 904 mod 1 396 = 432`, so 1 396 does not divide that body's `N` and the figure breaks determinism rule 10.** D4's case survives on the corrected table |

---

## 8. THE OPEN QUESTIONS AND THE SL6 ASKS

### 8.1 The SL6 asks

**The default is NO.**

| # | Data | From → to | Why it cannot be computed locally | Cost of refusing | Recommendation |
|---|---|---|---|---|---|
| **A1** | **THE BODY CHARTER**, about 19–21 quantised INTEGERS, ESTIMATED 42–76 bytes, inside the widened `TAG_SURFACE` on the realm's own self-look | the body's realm → its observers' clients | The client links no motion crate by an isolation row (ruling S7-1) and must not re-derive an unfenced float in its binary | The climate cannot depend on the planet's size, gravity, spin, tilt or orbit — **which is exactly what the owner asked for** | **ASK: yes.** A protocol-minor bump. **The byte count must be MEASURED against the 1 200-byte budget before it lands.** Integers, never floats. **And it widens SL10 clause 1 — see D48** |
| **A2** | **THE ORBIT-DERIVED FACTS** — insolation, equilibrium temperature, eccentricity — quantised by the SYSTEM and stated ONE HOP down | the system realm → the body's realm | The body's realm does not own the orbit relation; the parent does (SL1 clause 1) | The greenhouse, the season and the eccentric-orbit world all go | **ASK: yes**, and the owner should rule on SL1 clause 4 explicitly. **None of these facts is a placement** |
| **A3** | **THE MACRO ARTIFACT** on the bulk lane — the 4.8 KB pyramid top for the approach, then the 17.61 MB field for the body the player is at, each with its own digest | the body's realm → the client | Only if the client cannot derive it in time | No macro field on the client, so no rivers, no coasts and no eroded shape | **REGISTERED, NOT ASKED.** It is made if `M-L7`'s cold-login OR **UNPINNED-BODY** leg fails, or if `M15` shows the reach lead is shorter than the derive. **The lane is designed now and opened only on a measurement** |
| **A4** | **THE LIVE WEATHER**, as a short list of weather systems on a NEW lane with its own rate | the body's realm → its own observers | Weather is live state, and the client may never derive live state | No weather, which the owner asked for by name | **ASK LATER, after `U-L8` measures the bytes and the rate.** It may NOT ride `TAG_SURFACE`, which is *"ONCE PER REALM ON CHANGE, carried retained, never on a keep-alive"* |
| **A5** | ~~A weather field composed PER OBSERVER~~ | — | — | — | **NO.** That is the composed-per-observer shape the reach ruling killed |
| **A6** | ~~The macro-artifact digest inside the look~~ | — | — | — | **WITHDRAWN.** `crates/core/src/look.rs:70-73` forbids the measured half in a look by name. **And the cache no longer needs it: the key is the FULL INPUT TUPLE** (D1, critic B B10) |

### 8.2 The open questions that remain open

Revision 3 had fifteen. Revision 4 promotes Q4, Q10 and Q11 into the register (D51, D50, §6.6) and
answers Q5 with `M-L24`. Nine remain.

| # | Question | Why it is open |
|---|---|---|
| **Q1** | **Does erosion actually change the picture on THE world?** | `M-L13` is the honest test and it must run before one more line of landform design. **If forty passes of stream power do not move the slope distribution, the flat fraction and the hypsometric integral on a nearly-dry home planet, the cheapest answer is to spend the milliseconds on octaves and on the biome rule, and the solve closes** |
| **Q2** | **Is the closed form believable, or only cheaper?** | An offline erosion ORACLE — a test-only crate that produces no world — is the only honest answer. It converts *"the closed form looks right"* into *"its slope–area exponent and its crest–divide agreement match a real simulation's"* |
| **Q3** | **What is the reference picture's own skyline break count, under OUR rule?** | The reference's *"8–12 breaks"* was counted by eye off a screenshot; our 0 was computed by a stated rule. **They are different quantities.** `U16` runs the same rule over the reference so the ratchet can become a floor |
| **Q6b** | **Is 8s's shadow map affordable at the vista's chunk count?** | 2 805–4 200 resident chunks, each one draw call, each one shadow-caster. **UNMEASURED, and it is 8s's own first measurement** |
| **Q7** | **Is a year long enough to live through?** | The almanac needs the orbital period in universe ticks. A 1 000-hour year makes a season a thing a player reads about; a 20-hour year makes it a thing he plays. **Nobody has made that game-design decision** |
| **Q8** | **Does the star's colour reach the ground?** | An M-dwarf world is lit red. Is the plant colour a function of the star class, or always green? That is art, not physics |
| **Q9** | **How many biomes should ONE planet show?** | Nineteen exist; a single Earth-like world shows perhaps twelve. Should a continent read as one place rather than a patchwork? |
| **Q12** | **Do the new soils break the seed ruling?** | Laterite is a bauxite proxy; peat is fuel. The proposed answer: a soil names the CLIMATE, not the deposit. **If the owner disagrees, the soils are named by LOOK** — `RedSoil`, `DarkSoil`, `SourSoil` |
| **Q13** | **Who owns roads, terraced fields and walls?** | They are in the reference picture, they are culture rather than geology, and no domain here is named for them |
| **Q15** | **Five octaves or six inside the solved field?** | The difference is the 12.5 km octave and its 171 m of amplitude |

---

## 9. THE MEASUREMENT PLAN, AND THE KILL CRITERIA

**The method: measurements first, on THE world, never a proxy.** Every measurement runs on the home
planet, in release, on an unloaded machine. **One job at a time.**

### 9.1 The order

| # | Measurement | Pass bound | What it decides |
|---|---|---|---|
| **1. M0** | The baseline, re-taken: `just terrain-cost` on an idle machine | the cost tables reproduce within 10 % | that every later delta is real |
| **2. U1** | **Is the home planet the right body?** Print the home body's own taxonomy row: mass, bulk density, surface gravity, insolation, equilibrium temperature, the atmosphere, XUV and bolometric irradiation | the body is earth-like by the existing predicate, or it is not | **D27, and every picture the owner has judged so far** |
| **3. ★ M1b** | **THE SKYLINE PREDICTION.** Re-run `01` §1.4a's four-station, 360-bearing, 540 km march on the PROPOSED amplitudes, and on a synthetic macro field with the proposed macro relief | **publish the predicted break count per 60° and the largest rise**; the arc proceeds only if it clears 0.10° | **whether nineteen weeks are worth funding.** ESTIMATED 0.22°–0.27° under a linearity assumption that could be wrong (§1.3) |
| **4. ★ U16** | **The reference picture's own break count under OUR rule** | published | whether the headline gate is a floor or only a ratchet (Q3) |
| **5. ★ M8-L** | **THE LIGHT.** Re-shoot the three pictures the owner judged, with the star at 12°–18° and 100°–140° off the nose, shadows on | the owner judges *"terrain or light?"* | **D41, and whether every later picture is trustworthy** |
| **6. M-L10** | **The charter's cross-host spread.** Draw the home system on both target legs and print every taxonomy float's bits | **every quantum far larger than the measured spread** | D14 and D16's soundness. **Nobody has ever measured this** |
| **7. M1** | **The diagnosis, re-measured inside the crate.** Slope at 1 m / 32 m / 512 m; local relief in a 50 m / 1 km / 20 km window; the height histogram | the replays of `01` and `02` reproduce | that the diagnosis is a crate measurement |
| **8. ★ M-L25** | **CAN THE NEW GROUND BE PLAYED ON?** The fraction steeper than the character controller's limit; the flat area per km² a hull can land on; the fraction buildable without terracing — at three roughness settings | **stated bands, published BEFORE the spectrum's constants are fixed** | whether the picture is a game. **Playability decided the amplitude and then nobody measured it** (critic A F12) |
| **9. M9** | Over a 50 km window against a real elevation model: the amplitude spectrum, **the slope histogram** and the drainage density — of the design AND of the reference | **the three bands STATED, not "inside stated bands"**: median slope 3°–12°, p95 slope 25°–40°, drainage density 0.5–5 km per km² | the spectrum's constants. **The slope histogram is what would have caught a 53° planet before the owner did** |
| **10. M-L1** | **The pass schedule: nanoseconds per node, per PASS KIND**, at 6 × 64², 6 × 256² and 6 × 640² | the home planet's lattice solves inside the stated time and memory ceilings | every solve figure; D1, D3, D36 |
| **11. ★ M-L13** | **Does erosion change a DRY planet at all?** Four statistics before and after: the slope p95, the flat fraction under 2°, the drainage density, the hypsometric integral | **the flat fraction rises by at least half; the p95 slope rises; the hypsometric integral moves by at least 0.05** | **whether this whole domain exists** |
| **12. M-L12** | **No drift on the iterative field.** The macro digest equal on all four legs, one a REAL x86-64 machine, plus a tampered-table control | every digest equal; the tampered control RED | whether an iterative field may enter the generator at all |
| **13. M-L17** | **The cube net and the metric.** Every node has 8 neighbours except the 8 corners; the six faces' areas sum to `4 pi R^2`; no channel terminates on a face edge | exact neighbour counts; area within 1e-5; zero channels on a face edge | G-MACRO-CORNER, G-MACRO-AREA |
| **14. M-L4** | **The cube seam.** The height at a shared column from each side; the extracted channel polyline crossing it | **MAX difference = ZERO quanta**, not a p99 | whether the field is lawful under SL8 |
| **15. ★ M-L24** | **THE VEGETATION SKELETON'S COST**, on one chunk, with the anchor digest | recorded, **BEFORE D29 goes to the owner** | D29's real input. **The reference picture is mostly forest and no document prices it** |
| **16. M-L5 / M2** | **The per-chunk budget with the landform work on**, over the SAME named chunk set, **including the DENSEST measured chunk (23 084 vertices), a cave-dense chunk with a river AND a meshed water sheet AND the skeleton**, and its vertex count | **under 8 ms**; the top rung still cheaper than rung 0 | D29, D30; which components ship |
| **17. M12** | **`PASSES` is a RESOLUTION, not a dial.** Solve at `PASSES` and at `2 × PASSES` with the same erosional age; assert the two fields agree within a stated bound | they agree | D37 |
| **18. M-L7** | **The client's first visit, THREE legs.** (i) an approach from outside; (ii) a COLD LOGIN standing on the home planet; **(iii) an arrival at a body whose artifact is NOT pinned** | **no black frame on any leg; the ground refused, never wrong** | **D1's derive-or-ship fork, and D49's ceilings** |
| **19. ★ M-L19** | **THE VISTA CENSUS.** Standing at a 50 km horizon: chunks resident, vertices, bytes, and the wall time from arrival to a full vista, landform work off and on, **rock and water only, then again with the skeleton** | **under D31's stated ceiling (1.5 GB of mesh, 100 MB of artifacts)**; the fill under 2 s at the client's real thread count | **whether the reference picture is affordable at all** |
| **20. M-L22** | **The biome edge width.** A transect across a forest-to-desert boundary | **under 5 km**, because the reference picture's forest ENDS at a ridge | whether the climate grid can draw the picture's own edge |
| **21. M-L23** | **The fenced polynomials.** Per climate quantity: the degree, the maximum error, and the BIOME-BOUNDARY DISPLACEMENT that error causes | **the displacement under one macro node** | whether the fence can carry the climate |
| **22. ★ U8** | **The circulation fit.** Re-fit the Hadley edge constant over a wider anchor set and publish the residual per body | **a single constant holds Earth AND Mars within one cell, or the miss is published** | `05` §5.3 says `dH = 0.388` is a **reverse fit to Earth alone** and **Mars misses by a whole cell** (critic A F10) |
| **23. ★ U15** | **The precipitation's UNIT.** A two-point anchor for `K_mm_per_year` on an Earth-like charter | **2 000 mm/yr at the ITCZ and 100 mm/yr under the subtropical high, residuals published** | `05` §5.4: supply is in pascals and the classifier's axis is millimetres, and **nothing converted one into the other.** Without it the whole map's realism rests on a number nobody wrote down |
| **24. M-L14 / M8** | **The pop**, for FIVE terms now: the octaves, a river's arrival, a `Z` pyramid-level change, a terrace fade, a 3-D removal fade | no visible step at the drawn-pixel floor, at a walk and at 240 m/s | the fade widths of §4.2 |
| **25. M-L6** | **The river field.** Nanoseconds per column; the channel-node fraction; the DRAINAGE DENSITY; the rung at which the carve stops; **the runtime index's bytes** | ≤ 100 ns per column; density 0.5–5 km per km² | D18's dial and §3.3's UNMEASURED row |
| **26. U-L8** | **The weather statement's bytes and rate**, and **the anomaly field's memory per awake body against a stated shard ceiling** | stated against the lane's budget and a shard memory ceiling | **A4 is not asked before it** (critic A F14) |
| **27. M-L11** | **The emulation factor**, `linux/amd64` against `linux/arm64` | the macro leg at most 5 minutes | whether the leg joins `just gate` or is hand-run (D51) |

**The order is not free.** `M-L1` gates `M-L5` and `M-L7`. `M-L17` gates `M-L4`. `M-L24` gates `D29`.
`M-L19` gates the whole picture question. **`U1`, `M1b` and `M8-L` gate everything, and all three are
days of work, not weeks.**

### 9.2 The kill criteria

| Candidate | KILL at | Then what |
|---|---|---|
| A global solve at the vista's own resolution | **already dead: 3.6 hours and 131 GB at 257 m per node** | the 8 224 m solve |
| The global solve at 8 224 m | `M-L1` over **20 s** or over **160 MB** after the chooser has walked | the next coarser exact divisor: 16 448 m, about 3 s, ~3 MB — **and the trunk valleys are then 32 km apart, which reads as a lumpy planet** |
| **THE WHOLE ARC** | **`M1b`'s predicted largest rise stays under 0.10°** | the spectrum is not the cure; re-open the diagnosis before spending a week |
| **The macro solve AT ALL** | `M-L13` shows no change in the flat fraction, the p95 slope or the hypsometric integral | the alternative, PRICED: **four more octaves at 0.19 ms per chunk, a re-normalised roughness (free), and the biome rule.** The arc falls to about 12 weeks with a plainer world |
| The deposition and ice passes | `M-L1` shows them over **+40 %** of the schedule, or `M-L13` shows the flat fraction unmoved without them | drop them, **and state in the acceptance document that the flat alluvial valley floors and the U-shaped snowy valleys of the reference picture are NOT delivered** |
| The channel DISTANCE FIELD | `M-L6` over **150 ns per column** | the closed-form `segment_distance_m` term seeded by the macro receiver chain |
| Bicubic macro reading | `M-L5` over budget | bilinear plus one extra octave, then re-run `M-L14` |
| **An iterative field in the generator at all** | `M-L12` shows one digest differing on any leg | **the landforms must be a closed-form function of `(seed, address, charter)` — no erosion, and the vista is bought with shaped octaves alone** |
| **The client DERIVING the artifact** | `M-L7`'s cold-login OR UNPINNED-BODY leg fails, or the transient peak fails D49's ceiling | **ship it**: SL6 ask A3 is made |
| **The whole domain's per-chunk cost** | `M-L5`'s densest chunk over 8 ms after the FOUR dials of D29 | ask the owner to raise the budget, with the measured number in hand |
| **The vista** | `M-L19`'s mesh bytes over D31's 1.5 GB, or its fill over 2 s at the real thread count | fewer rings — **which is a detail-by-distance seam and must then be measured by `M-L14`** |
| **★ THE WEATHER MODEL** | `U-L8` shows the anomaly field over a stated shard memory ceiling with the shard's real awake-body count | drop the anomaly field to a coarser grid, or drive the system list from the climate mean alone with a seeded phase |
| **★ 8s, the sky and the shadow** | the shadow map costs more than a stated share of the client's frame at the vista's chunk count | cascades reduced, or the shadow drawn only at the two nearest rungs — **and the far ridge then reads flat, which is the defect the slice exists to cure** |
| The undercut and arch term | `M4-11` prices it over the remaining headroom | **no arches. The picture loses a landform family, and the owner is told** |

---

## 10. WHAT IT COSTS, IN SLICES AND IN WEEKS

**ESTIMATED, from the shape of each slice's work.** One engineer, the measurement runs included, the
gates green. It excludes the home-planet re-pick (D27) and the real x86-64 machine (D26), both
prerequisites rather than slices.

| Slice | The work | Weeks, ESTIMATED |
|---|---|---|
| **8p + 8L** the instrument and the light | the probe's second render target, the stamp through three consumers, the fenced stand search, nineteen verdicts, two recordings; **and the harness's own light fix, which is hours** | **2** |
| **8a** the picture, cheap | the slope spectrum, the roughness placeholder, the wavelength rule, ridged crests, the cap-rock bench, **D6's relief law and the sea**; `M1b`, `M1`, `M9`, `M-L25`, the vista | **2** |
| **8b** the charter | the draws with their conditioning, the quantisation, the widened tag, the client's use, the home pin and its cross-pin; `M-L10`, `U1` | **2** |
| **8s** the sky and the shadow | the atmosphere from the scale height, aerial perspective, a sky dome, a cascaded shadow map, the fill light retired | **2** |
| **8c** the solve | the lattice, D8, the flood, discharge, stream power, isostasy, talus with aridity, ice, craters, the coast, the bed dip, the climate on its nested sub-lattice, the four seam gates, the envelope, the pyramid, the artifact and its digest, the cache, the kernel canary, the **twelve** determinism rules, **HR5 at 100 % over a heap and a pointer chase**, and the HR4 pair | **5–6** |
| **8d** the water and the rock | rivers, lakes, the clipped water sheet, the carve with its two-rung fade, the sub-macro stream, the stratigraphic column, the rock map and the dip, differential erosion, alluvium, **the 8 ms fight with a fourth dial** | **4** |
| **8e** the biome and the paint | the classifier at a climate rung, nineteen biomes, the soil law, the snow and tree lines, sea ice, **the paint table and V7**, the anchor digest, the fenced polynomials | **2.5** |
| **8f** the third dimension | the 3-D removal term with its stopping rung and fade; placed rock objects as OBJECT-form kinds with slice 11's colliders | **2** |
| **18** the almanac and the weather | the derived almanac including the river stage, the live weather lane, the anomaly field with its ceiling, the client's cloud and rain on 8s's sky, the wind in the medium | **3** |
| **The measurement runs**, interleaved | `M-L1`, `M-L13`, `M-L12`, `M-L19`, `M-L7`, `M-L24`, `U8`, `U15`, the oracle | **1.5** |
| | **TOTAL** | **26–27, ESTIMATED 22–28** |

### 10.1 What the flagship frame holds at the end of each slice (rev 4, critic A F8)

**The most useful table in the document: it says which slice first produces a picture worth judging.**

| After | The pilot's 1.8 m frame holds |
|---|---|
| **today** | a lit plain, a hard 400 m horizon, a black sky, no shadow, one brown material |
| **8L** (hours) | **the same ground under a raking light with a cast shadow.** The first honest separation of *"flat terrain"* from *"flat light"* |
| **8a** (+2 wk) | **ridges, crests and a 50 km skyline** under that light — still brown, still no haze, still no river |
| **8s** (+2 wk) | the same, **with a blue haze separating four depth planes and a sky** |
| **8c** (+6 wk) | a drainage skeleton, a coast, a glacial skyline — **visible mostly from orbit at this rung** |
| **8d** (+4 wk) | **a carved valley, a river to the horizon, a lake, a benched cliff and a mesa.** ★ **THE FIRST FRAME THAT ANSWERS THE OWNER'S SENTENCE**, still in one colour |
| **8e** (+2.5 wk) | the same frame **in its own colours: a white cap, a green windward side, a tan leeward desert, blue water** |
| **8f** (+2 wk) | an arch, a scree cone under the cliff |
| **18** (+3 wk) | a cloud deck, rain on one side of the frame, a river in flood, and the same ridge in two seasons |
| **slice 14** | **the forest** |
| **slice 16** | **the character, standing upright, for scale** |

**Read the last two rows.** The reference picture is mostly forest with a figure in it, and neither is in
this arc. **8d's frame is the one the owner should judge the arc on**, and 8e's is the one he should
judge the world on.

**The four things that could move the total most.**

1. **`M1b`.** If the predicted skyline stays flat, the spectrum is not the cure and the diagnosis
   re-opens before a week is spent.
2. **`M-L13`.** If erosion does not move a dry planet's statistics, 8c and 8d shrink to the priced
   alternative and the arc falls to about **12 weeks** — with a plainer world.
3. **`U1` and D27.** If the home planet is re-picked, every golden table, the world tag, the cost table
   and the macro lattice are re-recorded: **1–2 weeks**, free today and a world epoch after slice 14.
4. **The 8 ms fight in 8d**, which now starts from a measured FAIL on the rock alone.

**And one line the whole arc rests on.** The store's identity is pinned at slice 14. **Every decision in
§7 is free until then.** The cheapest thing the owner can do this week is answer D1, D5, D14, D27, D29
and D41, and let `U1`, `M1b`, `M8-L`, 8p and 8a run — because they are two days of measurement and two
weeks of work, they open no lane, and they produce the one picture that says whether the rest is worth
building.

---

## 11. THE SIX-SLICE CHECK AGAINST THE FULL END GOAL

| The end goal | This arc's contribution | The gap it leaves |
|---|---|---|
| A seamless world (SL8) | §4.2's per-term coarsening rules and fades; `M-L14` on five terms | the carve's fade is asserted, not derived |
| One world (SL5) | no variant anywhere; every fixture takes a stated input, never a second world | HR5's wet arms need a Tier-A fixture fed from a real body |
| The client only renders (SL10) | the client derives SHAPE and nothing else; the light and the paint are style, and move no vertex | **D48: the shape is now `f(seed, address, stored charter)`** |
| A ship is a realm like any other (HR3) | §6.6 gives every slice a planet/Cartesian pair; a hull's fold reads a NULL artifact | Q11's terrain-in-a-hull seam still needs its own picture |
| The economy is an overlay | no ore map, no placer, no value from the seed (D33, D34a) | none |
| The dormant world advances believably | the almanac is a function of the tick; a sleeping realm ships nothing | **D48 (i): a dormant realm cannot be warmed ahead, because its charter rides a running realm's self-look** |

---

## 12. THE CRITIC ANSWERS

**Critic A — completeness and believability (15 findings).**

| # | Severity | The finding | Answer |
|---|---|---|---|
| **F1** | blocker | The light has no slice, no owner, no cost; §6.2 and §1.5 contradict each other about the sky | **FIXED.** Slice **8L** (hours, inside 8p) moves the star off the camera's azimuth and switches the shadow on; **NEW SLICE 8s** owns the atmosphere and a cascaded shadow map, at 2 weeks; **D41** puts it to the owner; the light becomes part of the picture protocol (§6.4); the contradiction is resolved — 8s owns the sky and slice 18 paints cloud and rain ON it |
| **F2** | blocker | No slice paints the ground, so 8e's four pictures cannot be judged | **FIXED.** The **PAINT TABLE** enters 8e's scope with its own cost (2.5 weeks, up from 2), **verdict V7 becomes 8e's landing condition**, a water material is named, and **D42** puts it to the owner |
| **F3** | defect | The arc never predicts the number it exists to move | **FIXED.** **`M1b`** is measurement 3 of 27 and runs BEFORE slice 8a; §1.3 publishes an ESTIMATED 0.22°–0.27° with the linearity assumption named as an assumption; **`U16`** is measurement 4 so the ratchet can become a floor; the kill table gains a row that ends the whole arc if `M1b` stays under 0.10° |
| **F4** | defect | The bound is stated for one term of three; `Z` contradicts itself across three sections | **FIXED.** §4.2 is a six-row table: one term DERIVED, one EXACT, one derivable when the pyramid is built, three asserted with a stated fade. §4.3 prints the whole sum. **Revision 3's *"no crossfade for `Z`"* is WITHDRAWN by name**: `Z` is rung-independent in its DEFINITION and level-dependent in its READING. §4.5 grows one test leg per term |
| **F5** | defect | Four landform families the domains named are missing from the omission list | **FIXED.** §1.5 grows from six to ten and names volcanic, aeolian, karst and mass wasting with the code that already asserts each. **D43** asks whether volcanic lands (recommended: yes, it is cheaper than the ice pass); **D44** registers the other three as NOT DELIVERED |
| **F6** | defect | The climate never reaches the shape: an arid range and a humid range are the same geometry | **FIXED.** **D45**: the roughness ceiling AND the talus angle read the column's ARIDITY. It enters the architecture diagram, §7.2's modulation row and 8c's scope. Cost: one lookup and one multiply |
| **F7** | defect | The vegetation skeleton is in no cost line | **FIXED.** **`M-L24`** prices the skeleton on one chunk and **runs before D29 goes to the owner**; §1.4 is headed *"rock and water only"*; §3.1's row says so; `M-L5` and `M-L19` both gain a with-skeleton leg |
| **F8** | defect | The arc's acceptance needs slices the arc does not contain, and the estimate does not say so | **FIXED.** §6.5 gains a *"Delivered by this arc?"* column, and **§10.1 is the new per-slice flagship-frame table** the finding asked for |
| **F9** | weakness | The spin and obliquity draws are unconditioned | **FIXED.** **D17** now requires the TIDAL LOCKING test from the semi-major axis and the star's mass, and damps the obliquity for close-in bodies; the home planet's drawn values are pinned in `home_body_pin.rs` |
| **F10** | weakness | A one-point Hadley fit, a precipitation field with no unit, three refused climate families | **FIXED.** **`U8`** and **`U15`** enter §9.1 as measurements 22 and 23 with stated pass bounds; the three refused families are stated in §1.5 so the owner learns them here |
| **F11** | weakness | One rock palette per planet, and every bed horizontal | **FIXED.** **D34 is SPLIT**: D34a refuses an ORE map (unchanged); **D34b proposes a ROCK map — a per-province palette and a BED DIP from the orogen field**, because the seed ruling forbids a map of VALUE and not a map of LOOK, and a range of flat beds reads as Monument Valley and not as the Alps |
| **F12** | weakness | No measurement asks whether the ground can be walked, landed on or built on | **FIXED.** **`M-L25`** is measurement 8 and runs before the spectrum's constants are fixed; **`M9`'s three bands are now STATED** (median slope 3°–12°, p95 25°–40°, drainage density 0.5–5 km/km²) instead of *"inside stated bands"* |
| **F13** | weakness | The almanac moves the snow and never the water | **FIXED.** **D46**: one per-basin STAGE OFFSET, derived from the almanac and the basin's snow fraction, applied to `z_w` and the drawn channel width. It is in §2.2 and in slice 18 |
| **F14** | note | The weather model the shard runs is in no cost table or kill criterion | **FIXED.** §1.4 and §3.1 carry **0.98 MB per home-size body, 3.93 MB per Earth-size body, ~4 ms a step**; §9.2 gains a weather-model kill row; `U-L8` gains a shard memory ceiling |
| **F15** | note | No authoring path reaches the synthesis | **FIXED.** **D47** carries `04` §8's two kinds and their separating law verbatim in substance: kind A live state on a coarse pyramid rung, kind B in the generator and in the identity |

**Critic B — the laws, the cost, the integration and the slice plan (18 findings).**

| # | Severity | The finding | Answer |
|---|---|---|---|
| **B1** | BLOCKER | The 14.75 MB artifact omits the climate, and D19 multiplies it by 6.25 | **FIXED.** **§3.3 itemises every kept byte and the total is 17.61 MB.** D19 is restated: *"one lattice"* is a SCHEDULE claim; the climate runs on a NESTED factor-4 sub-lattice and keeps **8** bytes, not 16 — 1.23 MB on the home planet. The river graph and the lake table are DERIVED and keep nothing; their runtime index is UNMEASURED and named. Every dependent figure is re-stated: D1's *"one per cent"* becomes *"under two per cent"*, D3's 5 140 m option becomes ~45 MB, D31 and A3 read 17.61 MB |
| **B2** | BLOCKER | §1.4 reports a pass where `07` reports a FAIL, and omits the densest chunk | **FIXED.** §1.4 carries the **densest measured chunk at 7.09–8.07 ms with NO water**, marked OVER; the surface and seam rows are corrected to `07`'s own 4.10–4.34 and 7.07–7.42. **D29 gains a FOURTH dial that acts on the ROCK extraction** — a per-chunk vertex budget, a roughness cap keyed to the cell, or a coarser rung for the cave band — and it is dial number one |
| **B3** | BLOCKER | Three things the flagship needs have no slice, one of them a 🟥 nine slices away | **FIXED.** The sky and the shadow become **8s** with a cost and **D41**; the paint becomes 8e's and **D42**; **V11 and `D-TERRAIN-4` are now named** in §6.5 with the honest answer: **the standing character is slice 16 and is OUTSIDE this arc**. §6.5's new column says which reference rows the arc cannot deliver at any budget |
| **B4** | BLOCKER | HR4 is not answered for one slice of the eight | **FIXED.** **§6.6 gives every slice a PAIR and an identical fixture body.** The macro lattice is placed **BELOW** the `GridMapping` seam by name, so it owes `G-MAPPING-TABLE` and `G-MAPPING-ROUNDTRIP`. **Q11 is answered and becomes the fixture: a hull holds terrain and holds NO artifact; the fold reads a NULL artifact and both hosts and both geometries must agree** |
| **B5** | DEFECT | The free window closes at slice 14, not slice 9 | **FIXED.** §6.1 is re-derived against the foundation's rule 1 and quotes it. **8, 8p, 8L, 8a, 8b and 8s land before slice 9; 8c, 8d and 8f land AFTER slice 11 where the collider exists; 18 lands after slice 9; the anchor digest is pinned at slice 14.** §6.2 gains a *"before / after"* column |
| **B6** | DEFECT | D5 contradicts itself: the ends cut the count and the count says it does not fall | **FIXED.** §7.2 states **ONE rule**: `Z` replaces the coarse octaves, **the count falls from 14 to 11 on the home planet (RECOMPUTED), and the saving IS credited** — the column pass must be re-priced. `06`'s *"no saving is credited"* is withdrawn where it collides with `04`'s ends |
| **B7** | DEFECT | 8a's picture is taken at a relief 8b replaces | **FIXED.** **D6's relief law and the sea move INTO 8a**, with the spectrum, because the amplitudes are anchored on the relief. 8a grows from 1.5 to 2 weeks; 8b keeps the rest of the charter |
| **B8** | DEFECT | The prefix claim covers the octaves and is asserted for the whole fold | **FIXED, with F4.** §4.2 states a stopping rung, a fade and a dropped magnitude for TERRACE, CARVE, the 3-D removals and the `Z` reading. **The carve gains a TWO-RUNG fade rule** so a trunk valley never arrives in one step, and 8f's removal gains the same. §4.3 prints the whole six-term sum |
| **B9** | DEFECT | The shape becomes `f(seed, address, stored charter)` and the text never says so | **FIXED.** **D48** is its own register row with the two consequences drawn: a dormant realm's shape cannot be derived before its realm runs and speaks (`look.rs:52-56`), and D17's tuning re-authors every stored charter and re-shapes every stored body. §11 carries it as a gap |
| **B10** | DEFECT | D1(c)'s content digest has no lawful source after A6 | **FIXED.** The cache key becomes **the FULL INPUT TUPLE — the universe seed, the body's address, the charter bytes and `GENERATOR_VERSION`** — stated in §3.1 and §5.4. A6 stays withdrawn and is no longer needed |
| **B11** | DEFECT | The client's derive has no ceiling, and the warm-up lead is unstated | **FIXED.** **§5.5 and D49**: at most one concurrent derive, at most four resident artifacts (~70 MB), and a residency-band completion distance of `closing speed × derive time` with the COMPUTED leads printed. **The two-stage arrival (the 4.8 KB pyramid top first) is the answer to a fast approach**, and **`M-L7` gains a third leg at an UNPINNED body** |
| **B12** | DEFECT | A player who reshapes the ground gets a river that ignores him | **FIXED.** **D50** widens Q10 into *"what does the artifact owe a changed surface?"* and gives the stated answer: **nothing, and the local diff wins inside its own width** — with the consequence written plainly so the owner rules on it instead of a player finding it |
| **B13** | WEAKNESS | D28 and D31 carry no recommendation, and D31 gates the decisive measurement | **FIXED.** **D28** proposes 55 %–70 % of the surface under water for the median body. **D31** proposes 1.5 GB of mesh and 100 MB of artifacts, anchored on the MEASURED 205 MB a client holds today for a 13×13×3 patch at rung 0 |
| **B14** | WEAKNESS | The erosional age is a typed world constant where the law makes it a body fact | **FIXED.** **D37**: the erosional age is a CHARTER DRAW, one of the 19–21 integers, and `PASSES` stays the resolution. `M12` is unaffected — it asserts agreement at the SAME age |
| **B15** | WEAKNESS | 8f's placed rock objects have no format, no host and no collider | **FIXED.** 8f now states all three: **seed-placed, so they are SHAPE and belong inside the world identity with slice 14's anchors**; **an OBJECT-form catalogue kind**, because `object_param` is legal only on such a kind and is REJECTED at decode elsewhere; and **slice 11's collider**, which is why 8f moved after slice 11 |
| **B16** | WEAKNESS | D4's chooser may read the host — a twelfth drift class; and the divisor claim is wrong | **FIXED.** **Determinism rule 12** forbids the chooser reading the host. And **RECOMPUTED**: 8 224 does not divide the Earth-like candidate's `N` at all; its nearest legal node sizes are **7 616 m and 8 704 m — within 7 %, not ±20 %** — but that body's 8 704 m lattice is **49.8 MB and ~58 s**, so the chooser walks to **14 336 m** under an 18 MB ceiling. D4 now states that |
| **B17** | NOTE | Four numbers do not reproduce | **ALL FOUR FIXED.** (1) `07`'s 1 396-per-face figure is REFUSED by name: `10 235 904 mod 1 396 = 432`, and the corrected divisor table is in D4. (2) G-MACRO-AREA now says **8 % between the two quoted weights and 24–30 % below a face centre's 1.0**. (3) The vista is printed as **two estimates under two residency rules, not a range**: `08` 2 805 × 405 KB = 1 137 MB; `07` 4 200 × 248 KB = 1 042 MB. (4) The chunk figures are `07`'s own: 4.10–4.34, 7.07–7.42, and the densest at 7.09–8.07 |
| **B18** | NOTE | Q4 is the arc's whole enforcement mechanism and has no owner | **FIXED.** Q4 becomes **register row D51**: a RATCHET tier with a stored baseline, a named owner for the baseline, and a stated rule for who may move it and on what evidence — with the VERIFIED fact that `just gate` already runs three of the terrain gates and not `terrain-cost` |

**Nothing is KEPT-with-a-reason and nothing is OWED-only.** Every finding of both critics is answered in
the document above. Four answers are *"fixed by naming the gap and giving it a decision row"* rather than
by designing the missing thing here — D41 (the sky), D43 (volcanoes), D44 (aeolian, karst, mass wasting)
and §6.5's *"slice 16"* row for the character — **because each is a scope purchase only the owner may
make, and the honest fix is to put the purchase in front of him rather than to assume it.**
