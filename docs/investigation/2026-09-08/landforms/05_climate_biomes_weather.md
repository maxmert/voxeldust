# 05 — Climate, biomes and the weather coupling

**Date:** 2026-09-08. **Revision 2** (answers refutation A and refutation B in full; see §16).
**Domain:** the climate field, the biome classifier, the vegetation skeleton and the weather split.
**Status:** an investigation report for the owner. It designs the domain. It decides nothing by itself.
**Binding law read first:** CLAUDE.md (HR1–HR6, SL1–SL10), `docs/design/owner_decisions_2026-09-07_voxels.md`
(V1 SL10, V2.1–V2.9, V4 the review, V6 the format sitting, V9 the generator, V10 the extractor and §12 the
vista, V11 up is the server's, V12 slice 7), `docs/design/owner_decisions_2026-08-27_seed_and_secrecy.md`
(a seed-derived map of value is a treasure map), and `docs/design/owner_decisions_2026-08-26_movement.md`.

Every number below is marked **MEASURED** (with how), **DERIVED FROM A MEASUREMENT**, **ESTIMATED**, or
**UNMEASURED**. Every claim about the code cites `file:line`. The investigation base is not binding; where
it disagrees with a ruling, the ruling wins.

**The owner's words this document answers.** *"Make sure that we reach that quality on the picture for
earth-like planets (biomes can be different of course, should be dependent on the planet position, spin,
trajectory, size and gravity, etc.). It should be very believable, as we also should simulate the weather."*

**What revision 2 changed, in one paragraph.** The per-octave cost is now the MEASURED 13.19 ns, not the
27.9 ns of revision 1, and every cost in the document was recomputed from it (§5.7). The charter no longer
rides the parent's per-child row, which the one-radius law closes by name; a body's own shard derives its
charter from the seed and states it about ITSELF, and only the client is told (§3). The biome is now read at
a derived CLIMATE RUNG, so it is identical at every rung by construction instead of bounded by a height
bound that does not bound a class (§5.10). Precipitation gained two finer terms, because a 41 km grid alone
gives contour bands and not the owner's picture (§5.4). The live weather is now a model with state, a rate,
a size, a determinism rule and its own decision rows, because the owner asked to simulate the weather and
revision 1 answered with a list of nouns (§8.5–§8.9). Lying snow is derived from the shipped storm list, not
a per-cell diff that would cost 12 MB per snowfall (§8.3).

---

## 0. The recommendation, in one page

1. **A body's climate needs facts the generator cannot compute, and the body's OWN shard already holds
   them.** The generator crate is fenced: it holds no logarithm, no sine and no power
   (`crates/terrain/src/gf.rs:1-40`). The star's luminosity, the body's mass, its equilibrium temperature
   and its atmosphere all live in `crates/physics/src/taxonomy.rs`, and every one of them comes out of
   `powf`. But a planet's own shard already derives them: *"every process derives the whole forest from the
   seed at boot (the complete SL6 answer — a planet shard computes its own mass, gravity, temperature and
   atmosphere locally, from the seed, with no message)"* (`crates/physics/src/worldgen/body.rs:47-51`).
   **So the server needs no message at all.** The answer is a BODY CHARTER: a short row of INTEGERS the
   realm derives about ITSELF and states in its OWN self-look, beside the surface tag that already rides
   there (`crates/core/src/look.rs:50-56`, ruling V6 row R-8). The ONE host that cannot derive it is the
   client, which links no motion crate. §3.
2. **Three facts do not exist anywhere yet: the body's SPIN, its OBLIQUITY and its surface PRESSURE.** The
   generator states `POLE_AXIS = +Z` with no obliquity and no day length (`crates/terrain/src/height.rs:30-35`);
   `crates/bins/src/lib.rs:1005-1008` says the planet does not spin; and the taxonomy says honestly that
   pressure has no derivation (`taxonomy.rs:875-879`, D-TAX-1). Each is one seed draw appended to the body's
   own stream. Obliquity also costs a shipped pin and a parent-authored orientation, and this document says
   so (§3.5). A fourth fact, the CONDENSABLE volatile, is DERIVED and not drawn (§3.4).
3. **Climate physics runs under the float fence by three techniques and one rule.** Every transcendental
   whose argument is a per-body constant is evaluated once outside the fence and enters as a charter
   integer — the circulation's cell count and cell edges included, which revision 1 wrongly left inside.
   Where an argument varies per column, a fixed-degree polynomial or a 256-entry table ships with a measured
   error bound. Nothing on the per-column path is worse than add, subtract, multiply, divide and square
   root. §4.
4. **The static climate is a field of DIRECTION and HEIGHT, never of the clock.** Temperature,
   precipitation, seasonal range and wind are `f(seed, charter, direction, height)`. The clock never
   enters. That is what makes the climate SL10-legal and what lets both hosts compute it. §5.
5. **Precipitation needs THREE scales, not one.** A coarse grid alone puts the whole of the owner's 50 km
   vista inside one climate cell, so the biome would vary with height alone and the picture would be
   contour bands. The model carries a GRID term (the rain shadow behind a whole range, accumulated at
   40–50 km), a MID term (one coarse-height evaluation 5–15 km upwind, which resolves a single range) and a
   LOCAL term (the slope along the wind, read from the column pass's own neighbours, which resolves ONE
   ridge). Together they give a green windward valley beside a dry leeward cliff at the same height, which
   is the picture's most striking property. §5.4.
6. **The biome is read at a derived CLIMATE RUNG, so it never changes with the detail level.** A height
   bound in metres cannot bound a CLASS: one column 20 m under the snow line at rung 0 and 20 m over it at
   rung 8 flips from Grassland to Ice, and the white patch moves as the pilot flies in. The cure is not a
   tolerance; it is to read the classifier's height at ONE rung derived from the body's own lapse rate, at
   every drawn rung. Then the biome, the snow line and the sea-ice line are IDENTICAL at every rung by
   construction, and the gate is `assert_eq`, not a share. §5.10.
7. **The biome set is 19 values in five bits: twelve from the chart and seven overrides.** The record does
   not change: a biome is a COLUMN property that both hosts derive; only an EDITED cell gets a record
   (`topic_00_format_sitting.md:34-37`). §6.
8. **The biome decides the soil, the ice and the vegetation skeleton.** Five strata are appended after
   `Empty`. Sea ice replaces sea water where the annual mean is below the condensable's freezing point, so a
   planet grows its own ice caps. The vegetation skeleton (V4: art assets, never drawn primitives) is a
   hashed scatter whose density, kind mix, size and aspect come from the biome, and it is folded into a new
   ANCHOR DIGEST beside the cell and mesh digests, because a tree a player can stand on is part of the
   shape. §7.
9. **THE WEATHER SPLITS THREE WAYS, and the split is the whole design.**
   - **STATIC — the climate.** `f(seed, charter, address)`. No message. Both hosts.
   - **THE ALMANAC — closed-form time, DERIVED, not shipped.** The season's phase, the sub-solar point and
     the day's phase are functions of the charter, the universe tick and the placements the window ALREADY
     carries. Ruling S7-7 already has the client derive the light from the brightest luminous row in its
     window (`owner_decisions_2026-09-07_voxels.md:502`). The almanac is that rule read a little further.
     **No new lane.**
   - **LIVE — the weather proper.** A field of pressure, moisture and temperature anomalies on the climate
     grid, owned and stepped by the realm's own shard, carried in the realm's checkpoint, and shipped to
     that realm's own occupants as a SHORT LIST OF WEATHER SYSTEMS — never as a field. About 16 bytes per
     system, at most 64 systems, so about 1 KB per body on change. §8.5–§8.9.
   **A season must never invalidate a chunk.** The seed decides the shape and the permanent cover. The
   almanac and the storm list decide the PAINT and a thin DERIVED cover both hosts compute from the same
   shipped rows. Only a cell a player's hand changes becomes a record. §8.2, §8.3.
10. **★ THE HOME PLANET IS PROBABLY THE WRONG BODY, and the sea it draws is a second problem.** Seed 2298
    was chosen by a seed search because it holds an Earth-like world of 1.087 M⊕ and 1.023 R⊕ — 6 515.5 km,
    g 10.20 m/s², Rocky, temperate, around a G star, and it RETAINS ITS ATMOSPHERE
    (`crates/physics/src/worldgen/census.rs:57-63`, MEASURED and pinned). The voxel foundation's home planet
    has a look radius of **3 350 759 m** (`crates/terrain/src/home.rs:22`, MEASURED) — half that, and a
    different planet. ESTIMATED from the shipped laws, that body is airless and hot at either of the two
    orbital rungs it can occupy. **UNMEASURED and owed as U1**; the estimate is stated as an estimate, and
    §10 says exactly what the measurement must print. Independently MEASURED and already reported to the
    owner: the current home planet's sea covers **about one column in a hundred**
    (`slice_05_generator.md:388-392`). A world with no sea has no water supply, so the precipitation model
    returns desert everywhere on it. §5.8 and §10.

---

## 1. What the code holds today (the only current truth)

| Fact | Where | What it means for the climate |
|---|---|---|
| The biome is decided by three inputs: the latitude as `abs(dir[POLE_AXIS])`, one temperature noise, one humidity noise, plus a height override. | `crates/terrain/src/height.rs:38-64` | There is no insolation, no star, no spin, no wind, no ocean and no lapse rate. The "temperature" is a shape, not a temperature. |
| Four biomes exist: Desert, Grassland, Tundra, Highland. | `crates/terrain/src/strata.rs:114-131` | The vista needs forest, taiga, savanna, wetland, beach and ice. Fifteen more. |
| The biome field's parameters are two noise seeds, two wavelengths (30–120 km and 20–80 km) and one height threshold at 55 % of the relief. | `crates/terrain/src/body.rs:204-215` (the struct is `body.rs:73`) | Every one of them is a pure seed draw with no physical meaning. They are the placeholder the climate replaces. |
| `biome_at` reads `surface_m` at whatever rung the caller built, and the column pass passes the rung's own height. | `crates/terrain/src/height.rs:40-46`; `crates/terrain/src/chunk.rs:180-183` | **The biome already changes with the detail level today.** §5.10 states the cure and the gate. |
| The pole axis is `+Z` and the code states that an obliquity is a later slice. | `crates/terrain/src/height.rs:30-35` | Obliquity does not exist. Neither does spin: `crates/bins/src/lib.rs:1007` says "while the planet does not spin". |
| `POLE_AXIS = +Z` is cross-pinned against the orbits' own axis, and the pin asserts an uninclined orbit stays at `z = 0`. | `crates/bins/tests/home_body_pin.rs:44-60` | The spin axis and the orbit's axis are the SAME axis today, which is exactly zero obliquity. **An obliquity makes this pin false as written.** §3.5. |
| The generator's float type offers add, subtract, multiply, divide, negate, square root, floor, truncate, absolute value and comparison, and NOTHING else. `sin`, `asin`, `mul_add` and every transcendental are compile-fail. | `crates/terrain/src/gf.rs:1-40,96-140` | Every climate law must reach the per-column path as a polynomial, a table or a per-body constant. §4. |
| The body's whole recipe is drawn from the seed: the relief (0.4 % of the radius, clamped 200–12 000 m, times a factor in [0.5, 1.5)), the coarsest wave (a quarter to a half of the radius, clamped 20–400 km), the roughness in [0.45, 0.55), the sea offset, the strata depths, the caves. | `crates/terrain/src/body.rs:139-232` | `LONG_WAVE_CAP_M = 400 000` (`body.rs:24`) CLAMPS for every body over 1 600 km, so on every planet worth a vista the coarsest wave is that constant. §5.1 states this instead of claiming a derivation. |
| The strata are ONE thickness for the whole body: one topsoil, one subsoil, one sediment depth, one sediment kind, one bedrock. | `crates/terrain/src/body.rs:193-201` | **There is no lithological contrast anywhere on any body**, so differential erosion — a hoodoo, a layered mesa — cannot happen. §7.3 corrects revision 1's claim; §12 hands the finding to domain 03. |
| The column pass computes, per column, the direction, the surface radius and the biome; the chunk keeps the lowest and the highest surface. | `crates/terrain/src/chunk.rs:145-200` | The climate slots into exactly this pass. One climate value per column, not per cell. |
| The taxonomy already derives, per body: the class, the mass, the radius, the insolation relative to Earth, the equilibrium temperature, the Bond albedo, and the atmosphere (mean molecular weight, scale height, reference density). | `crates/physics/src/taxonomy.rs:890-975` | **Almost every fact a climate needs already exists**, and a planet's own shard already computes it locally. It is in the wrong number format for the fence, and it is missing on the client. §3. |
| The taxon row is never lowered onto a region and never on the wire, because *every process derives the whole forest from the seed at boot — "the complete SL6 answer"*. | `crates/physics/src/worldgen/body.rs:47-51` | **The server side of the charter needs no message.** Only the client leg is an ask. §3.6. |
| The retention law takes **XUV irradiation**, not bolometric insolation, and says so by name; for a G star the two are equal (`xuv_rel_of(1.0, G) == 1.0`). | `crates/physics/src/taxonomy.rs:791-802,1511-1518` | §10's arithmetic survives only because the home star is G class (`census.rs:57-63`). U1 must print both numbers. |
| Eccentricity is hard-capped: `ECC_SIGMA = 0.03`, `ECC_CAP_SIGMAS = 4.0`, so `ecc_cap = 0.12`. | `crates/physics/src/worldgen/config.rs:61-71` | The largest periapsis-to-apoapsis light ratio in this world is `((1.12)/(0.88))² = 1.62`, not the 2.25 revision 1 used. §5.5. |
| The parent's point-of-light bag may carry ONE number, by a verbatim owner ruling: *"no surface, no detail, no mesh, no second number"*. | `crates/core/src/look.rs:41-47` | **The charter may never ride the parent's per-child row.** Revision 1 put it there. §3.6 moves it. |
| The realm's OWN self-look carries the surface tag, on change, retained, inside a 1 200-byte budget. | `crates/core/src/look.rs:50-56,78-80` | This is where the charter rides: a realm's statement about what it IS. 64 bytes fit. |
| The world identity is ONE pair for the whole world: a declared tag folded from the version and the seed, and a measured half folded from the HOME body's eight golden chunks. | `crates/terrain/src/tag.rs:19-50`; `crates/terrain/src/digest.rs:1-8` | **A per-body charter cannot ride the world tag.** §3.7 puts it in the per-body digest and the per-realm surface refusal S7-3 already built. |
| The cube-sphere has a direction-to-face inverse and a face-to-direction map: `face_of`, `face_coords`, `direction`, and the ladder's `index_of`. | `crates/seed/src/bend.rs:224-262`; `crates/seed/src/ladder.rs:145-160` | **A march across faces is a shipped primitive after all**, and it needs no tangent-frame rotation, because the step is taken in the body's 3-D frame. §5.1 corrects revision 1's wrong citation of `site_of`. |
| A record exists only for a cell a player changed. Base terrain has no record. Provenance has three values: terrain, biome feature, placed. | `docs/investigation/2026-09-07/topic_00_format_sitting.md:34-54` | A bigger biome set costs the format nothing. A felled tree has a provenance value waiting for it. §7.2. |
| **MEASURED** cost: the column pass at rung 0 costs **710 µs** for 14 octaves, falling to 266 µs at the top rung with 3 octaves. | `docs/investigation/2026-09-07/slice_05_generator.md:369-379`; ruling V9 (`owner_decisions_2026-09-07_voxels.md:382-383`) | `710 µs / (3 844 columns × 14 octaves) = ` **13.19 ns per octave evaluation**. Every ESTIMATED cost in this document is derived from this one number. |
| **MEASURED** cost: a surface chunk 3.3 ms, a cave-dense chunk 6.1 ms, the budget 8 ms; the client's geometry 4.18 ms and 405 KB per chunk. | ruling V10 (`owner_decisions_2026-09-07_voxels.md:440-444`); `crates/bins/examples/terrain_cost.rs` | Headroom: **4.7 ms** on a surface chunk, **1.9 ms** on a cave-dense one. The climate must fit inside the SMALLER one. §5.7. |
| The home planet: seed 7 701 581 858 760 374 086, look radius 3 350 759 m, 12 rungs, 14 octaves, relief 14 304 m. | `crates/terrain/src/home.rs:19-42`; `slice_05_generator.md:367-369` (MEASURED) | The face edge at rung 0 is `n = 5 263 360` cells (recomputed here from `R·π/2` snapped to `2570 × 2048`; the crate's own test asserts the radius and the rung count, not the edge). |
| **MEASURED:** the home planet's sea stands 5 297 m under the ladder radius against a relief of 14 305 m, so **the sea covers about one column in a hundred**. | `slice_05_generator.md:388-392`; ruling V9 | **This alone returns desert everywhere.** §5.8. |
| The world's Earth-like planet at seed 2298 is 1.023 R⊕ = 6 515.5 km with g 10.20 m/s², around a G star, and it retains its atmosphere. | `crates/physics/src/worldgen/census.rs:57-63,94-107` | It is not the home planet of the voxel foundation. §10. |
| `home_body` returns the first planet of the home system **whose look radius the ladder accepts**, not simply the innermost. | `crates/bins/src/lib.rs:987-1003` | §10's orbital rung is therefore UNMEASURED. The conclusion survives at either candidate rung; the numbers do not. §10. |

---

## 2. The words, explained once

Each word gets its plain meaning and an example in the game's own terms.

| Word | Plain meaning | Example in the world |
|---|---|---|
| **Insolation** | How much starlight falls on one square metre of the body, measured against what Earth gets. It falls with the square of the distance to the star. | The home system's third planet gets 0.748 of Earth's light (MEASURED, `census.rs:96-98`). |
| **XUV irradiation** | The hard ultraviolet and X-ray part of the star's light. It is what strips an atmosphere, and it is a different number from the total light. | The retention law reads it by name (`taxonomy.rs:791-794`). For a yellow (G) star it happens to equal the total light, which is why §10's sum works at all. |
| **Bond albedo** | The share of all the starlight the body throws straight back. A white cloud deck throws back most of it; bare rock throws back little. | A cloudy planet stays colder than a bare one at the same orbit. |
| **Equilibrium temperature (`T_eq`)** | The temperature a body would sit at with no atmosphere: the balance of light in against heat out. Earth's is 255 K; Earth's real surface is 288 K. | The shard already computes it (`taxonomy.rs:949-953`) and the charter carries it. |
| **Greenhouse warming** | How much warmer the surface is than `T_eq`, because the air lets light in and holds heat back. Earth +33 K, Venus +505 K, Mars +5 K. | It decides whether the pilot's home valley holds liquid water or ice. |
| **Lapse rate (Γ)** | How fast the air cools as you climb. On Earth about 6.5 K per kilometre. The dry value is gravity divided by the air's specific heat. | It is why the ridge behind the valley is white and the valley is green. On a heavy planet the ridge goes white sooner. |
| **Scale height (H)** | The height over which the air thins to a bit more than a third. Earth 8.5 km. Already derived (`taxonomy.rs:762-767`). | It sets how far the haze reaches. |
| **Optical depth** | How much a beam of light is scattered away along its whole path. It is the air's density times a per-gas cross-section times the distance. | It is the real cause of the blue fade on the far mesas: a long path through thick air. The scale height alone does not give it. §7.3. |
| **Rayleigh scattering** | Scattering by molecules much smaller than the wavelength. It goes as one over the wavelength to the fourth power, so blue scatters far more than red. | It is WHY the haze is blue on an air world, and why a thick carbon-dioxide world's haze is not the same colour. |
| **Obliquity (ε)** | The tilt between the body's spin axis and the plane of its orbit. Earth 23.44°. | It is the reason a world HAS seasons. At zero tilt every day of the year is the same day. |
| **Coriolis effect** | A planet turns. Air that moves north keeps its eastward speed and so runs ahead of the ground under it, and bends. The faster the spin, the stronger the bend. | It is why the pilot's wind blows from the east near the equator and from the west at mid latitude. |
| **Hadley cell** | The loop of air that rises at the hot equator, moves poleward high up, sinks dry at about 30°, and returns along the ground. | Its sinking branch is the subtropical desert belt. |
| **Thermal Rossby number** | One number that says whether a planet's spin is fast or slow COMPARED WITH the heating that drives its winds. A big number means slow spin and one big loop; a small number means fast spin and many thin bands. | It is the number that turns the pilot's "the day is 8 hours" into "you will fly over six wind bands". |
| **Ferrel cell and polar cell** | The next two loops poleward, driven by the ones beside them. | They set the wet storm belt at about 60° and the dry cold pole. |
| **ITCZ** | The inter-tropical convergence zone: the band where the two Hadley cells meet and the air rises. It rains there. | The rainforest belt of a planet. It moves with the season, and that movement is the monsoon. |
| **Orographic lift** | Air that must climb a ridge cools and drops its water on the way up. | The windward side of the home planet's ridge is forest. |
| **Rain shadow** | The dry ground behind the ridge, because the air already dropped its water. | The leeward side of the same ridge is scrub, then desert. This is the contrast the owner's picture shows. |
| **Continentality** | How far a place is from the sea. Far inland means hotter summers, colder winters and less rain. | The middle of the largest land mass is a cold steppe; the coast at the same latitude is a forest. |
| **Hypsometry** | The distribution of height over a body: how much of it is deep sea, shelf, plain and mountain. | The sea radius (`body.rs:186-190`) and the relief together decide the world's ocean share — and today they draw a world with 1 % sea. §5.8. |
| **Isostasy** | Rock floats on rock. A mountain's weight pushes a root down into the softer layer under it, and the crust fails when the load is too great. The tallest a mountain can be falls roughly as one over gravity. | Mars carries a 22 km volcano at 3.7 m/s²; Earth carries 9 km at 9.8. §5.9 checks the recipe's relief against this and finds a real problem for the Earth-like body. |
| **Whittaker classification** | A chart with mean temperature on one axis and mean rainfall on the other, cut into biomes. | Warm and wet gives rainforest; warm and dry gives desert; cold and dry gives tundra. |
| **Holdridge life zones** | A finer chart that adds potential evapotranspiration: how much water the sun COULD take away. | It separates a hot dry savanna from a cool dry steppe at the same rainfall. We take its aridity term only. |
| **Biotemperature** | Holdridge's own temperature axis: the mean temperature with every value below freezing counted as zero, because a plant does not grow in a frozen month. | Two valleys with the same annual mean grow different forests if one of them freezes for half the year. |
| **Potential evapotranspiration (PET)** | The water the sun and the wind would lift from wet ground, if the ground were wet. | Rain of 400 mm makes grassland where PET is 500 mm and desert where PET is 2 000 mm. |
| **Aridity index** | Rainfall divided by PET. Under 0.05 is hyper-arid; over 0.65 is humid. | One number that decides sand against grass. |
| **Legendre coefficient, `P2`** | A way to write a smooth curve over a sphere as a sum of a few standard shapes. `P2(x) = (3x² − 1)/2` is the second one: high at both poles, low at the equator. Its coefficient `s2` says how much of that shape the light has. | Earth's `s2` is about −0.48: the poles get less. Above a 54° tilt the sign flips and the poles get more. It is ONE number per body, so it rides in the charter. |
| **Energy-balance climate model** | The simplest real climate model: light in, heat out, heat carried poleward, solved for a temperature that varies with latitude only. It is a textbook model from the 1970s, not a guess. | It is what turns "this planet gets 0.748 Suns and has 1 bar of air" into "the pole is 45 K colder than the equator". |
| **Tetens form** | A short fitted formula for how much water vapour air can hold at a given temperature. It is the practical form of the exact law (Clausius–Clapeyron). | It is the one place the model needs a curve rather than a line, so §4 fits a polynomial to it and measures the error. |
| **Chamfer pass** | A cheap way to fill in "how far is the nearest sea" over a whole grid: sweep the grid once one way and once the other, and each cell takes its neighbour's distance plus one step. | It is how the home planet learns that its inland steppe is 2 000 km from water without asking every cell to search. On a closed cube it needs more than two passes; §5.1 says how many and in what order. |
| **Degree-days** | The warmth of a season added up: how many degrees above freezing, times how many days. | It is how deep the summer thaws the tundra before the pilot's shovel hits permafrost. |
| **Aspect** | Which way a slope faces. | On the home planet's northern hemisphere a south-facing slope gets more sun, so its trees climb higher and its snow goes first. |
| **Aerial perspective** | The way distant things go pale and blue because you look through more air. | It is what makes the far mesas in the owner's picture read as far. |
| **Ulp** | "A unit in the last place": the smallest step a computer can take between two neighbouring numbers of that size. Two machines that disagree by one ulp disagree by the least amount they can. | The ladder's snap test moves the home planet's radius by 1 000 ulps and still builds the identical body (MEASURED, `home.rs:44-77`). |
| **The almanac** | Our name for the part of the weather that is a closed-form function of the seed, the charter and the clock: the season, the day, the sub-solar point. | A dormant planet nobody visits still has a correct sunrise, because the almanac needs no shard. |
| **A weather system** | Our name for one live storm, front or high: a centre, a size, a strength and a drift. | The shard says "a storm 300 km wide, centred here, drifting east at 12 m/s". The pilot's client draws its cloud and its rain from those four facts. |

---

## 3. The body charter: the facts a climate needs, and where each one comes from

### 3.1 The path today, and the wall

```text
   THE FOREST (crates/physics)                    THE GENERATOR (crates/terrain)
   ---------------------------                    ------------------------------
   seed -> mass -> class -> luminosity            seed -> relief, octaves, sea, strata, caves
        -> orbit  -> insolation -> T_eq                 -> height field -> biome (two noises)
        -> albedo -> atmosphere (mu, H)
                                                  Fence: + - * / sqrt floor abs compare
   Uses: ln, powf, powi, cos, sin, tan            Uses: nothing else. By design.
        (generate.rs:571,598,633,807;
         taxonomy.rs:616,672,684,734)

   A PLANET'S SHARD links BOTH.  (worldgen/body.rs:47-51: every process derives the whole
                                  forest from the seed at boot.)
   THE CLIENT links only the RIGHT-HAND side.
                                  (crates/client/Cargo.toml:24-26: vd-physics is DEV-ONLY.)
```

**So the wall has one hole and one wall.** A planet's shard stands on both sides: it derives its own taxon
with `libm` and it builds its own chunks under the fence. The client stands on one side only. Revision 1
asked the parent to state the charter to the child; that was wrong twice over — the child needs no message,
and the parent's bag may not carry one.

### 3.2 The charter: integers, and why integers

**The rule.** Every fact that crosses the fence, or the wire, crosses as a whole number in a stated unit.
The producer evaluates the fact with full `libm` precision and floors it to that unit.

**Why quantising is safe.** The body's look radius is already an input from outside the fence. A drift of one
thousand ulps gives the SAME body, byte for byte, because the ladder snaps the radius to a whole number of
cells before the recipe reads it (`crates/terrain/src/home.rs:44-77`, MEASURED). The charter uses the same
trick with a coarser quantum.

**What quantising does NOT prove, and revision 1 claimed.** A perturbation test cannot measure the risk. The
risk is a DIFFERENT `libm` on a different target — glibc against Apple's, x86-64 against aarch64 — where a
chained `powf`, `ln` and `cos` expression can differ by far more than one ulp of the final value. The honest
measurement is the one `D-TERRAIN-1` already does for chunks: compute the charter on both targets and
compare the integers. §12, U5.

**The proposed charter.** Twenty-four fields, about 64 bytes. Every one names its source.

| # | Field | Unit (integer) | Source today | State |
|---|---|---|---|---|
| 1 | `gravity_mm_s2` | mm/s² | `surface_gravity_mps2(mass, radius)` (`taxonomy.rs:750`) | quantise |
| 2 | `insolation_q12` | 1/4096 S⊕ | `BodyTaxon.insolation_rel` | quantise |
| 3 | `t_eq_q4_k` | 1/16 K | `BodyTaxon.t_eq_k` | quantise |
| 4 | `t_surface_q4_k` | 1/16 K | `t_eq_k` + the greenhouse law (§5.2) | the producer's law, not the fence's |
| 5 | `bond_albedo_q10` | 1/1024 | `BodyTaxon.bond_albedo` | quantise |
| 6 | `has_air` | one bit | `BodyTaxon.atmosphere.is_some()` | ships as is |
| 7 | `mu_q8` | 1/256 atomic mass units | `Atmosphere.mean_molecular_weight` | quantise |
| 8 | `scale_height_m` | metres | `Atmosphere.scale_height_m` | quantise |
| 9 | `p_surf_pa` | pascals | **missing** (D-TAX-1) | a new draw, §3.3 |
| 10 | `condensable` | one tag byte | **missing** | DERIVED, §3.4 |
| 11 | `freeze_q4_k` | 1/16 K | the condensable's freezing point at `p_surf` | a table lookup |
| 12 | `boil_q4_k` | 1/16 K | the condensable's boiling point at `p_surf` | a table lookup |
| 13 | `lapse_mk_per_km` | milli-K per km | `Γ = g/c_p × (1 − w·humidity)` (§5.2) | the producer's law |
| 14 | `dt_pole_equator_q4_k` | 1/16 K | the transport law (§5.2) | the producer's law |
| 15 | `s2_q12` | 1/4096, signed | the annual-mean insolation profile from `ε` | **the producer's integral** |
| 16 | `obliquity_q16` | 1/65536 turn | **missing** | a new draw, §3.3 |
| 17 | `day_s` | seconds | **missing** | a new draw, §3.3 |
| 18 | `spin_phase_q16` | 1/65536 turn | **missing** | a new draw, §3.3 |
| 19 | `year_s` | seconds | derived from `sma` and the central mass | quantise |
| 20 | `ecc_q16` | 1/65536 | `OrbitalElements.ecc` (capped at 0.12) | quantise |
| 21 | `arg_periapsis_q16` | 1/65536 turn | `OrbitalElements.arg_periapsis` | quantise |
| 22 | `cells_per_hemisphere` | a count, 1–6 | the Held–Hou law (§5.3) | **the producer's arcsine** |
| 23 | `cell_edge_sin_q15[6]` | 1/32768, sine of latitude | the same law's cell edges | **the producer's sines** |
| 24 | `star_class` | one tag byte | `StarPhotometrics.class` | ships as is |

Rows 15, 22 and 23 are the ones revision 1 got wrong. Revision 1 put the cell COUNT inside the fence and
claimed *"the square root is inside the fence"*. The square root is; the ARCSINE that turns `sin φ_H` into a
latitude is not, and neither are the sines of the cell edges. All three are functions of the body alone, so
all three are charter integers. §4 states the rule they follow.

The radius and the sea radius are not charter fields: the ladder radius already rides as the look shell, and
the sea radius is already in the recipe (`body.rs:186-190`).

### 3.3 The three missing draws

**A. Obliquity (ε).** The physical prior: the solar system runs from Mercury's 0.03° to Uranus's 98°, and a
giant impact can produce any value. A strictly isotropic prior (uniform in the cosine of the tilt) gives
about a third of all worlds a tilt over 60°, which is more exotic than the solar system.
RECOMMENDED: draw `cos ε` uniformly over [0.2, 1.0] for nine bodies in ten, and over [−1.0, 0.2] for the
tenth. Most worlds then have familiar seasons, and one in ten is a Uranus-like curiosity. **The split is an
owner decision (C-14), not a physics result, and the code must say so.**

**B. Spin period (the day).** The physical prior:
- The lower bound is real physics: the break-up period, at which a point on the equator would fly off. It is
  `2π√(R/g)`. For the world's Earth-like planet (R 6 515.5 km, g 10.20 m/s²) that is **1.39 hours**
  (computed here from the census's published values). For the current home planet it is about 1.76 hours.
- The upper bound is tidal locking. A planet close enough to its star turns once per orbit. The standard law
  is Kasting, Whitmire & Reynolds 1993: the locking radius grows as the sixth root of the age and as the
  cube root of the star's mass. For a Sun-like star at 4.5 Gyr it is about 0.5 AU.
- Between the two, accretion leaves rocky bodies at hours to tens of hours, and tides from moons slow them.

RECOMMENDED: draw `day_s` log-uniformly over [6 h, 100 h], then apply the locking test; a body inside the
locking radius gets `day_s = year_s` and is a locked world (§9). Both bounds are physical; the 6 h and 100 h
ends are anchored on the solar system's rocky bodies and the code must say so (C-2).

**C. Surface pressure (p₀), which closes D-TAX-1.** The code states honestly that pressure has no derivation
(`taxonomy.rs:875-879`). It is a history, not a law. RECOMMENDED: one seed draw, log-uniform, over a band
the retention physics BOUNDS:
- the shoreline verdict fails ⇒ `p_surf_pa = 0`;
- a secondary atmosphere is retained ⇒ draw log-uniformly over [1 000 Pa, 10 000 000 Pa], that is 10 mbar to
  100 bar. That band holds Mars (610 Pa, just below), Titan (146 700 Pa), Earth (101 325 Pa) and Venus
  (9 200 000 Pa);
- a hydrogen envelope is retained ⇒ the base pressure follows from `envelope_reference_density_kgm3`
  (`taxonomy.rs:769`), and there is no rock surface to stand on anyway.

This is a DRAW, not a derivation, and the document must not dress it up as one. It is the same honesty the
albedo draw already uses (`GEOMETRIC_ALBEDO_BOUNDS`, `worldgen/body.rs:84-90`).

**★ The append rule.** Every one of these draws is APPENDED to the body's own stream, after the frozen
prefix (`crates/physics/src/worldgen/generate.rs:100-107`). Nothing already placed moves.

### 3.4 The condensable volatile — DERIVED, not drawn

Revision 1's charter named no condensable, yet §5.4 asked for a saturation vapour pressure and §5.6 asked
for a freezing point. That is a real hole: Earth's air is mostly nitrogen (mu 28) and it rains water
(mu 18); Titan's air is also mostly nitrogen and it rains methane (mu 16). The bulk mean molecular weight
does not name the rain.

RECOMMENDED, and it needs no new draw. The condensable is the one substance from a short physical table
whose liquid range STRADDLES the body's surface temperature at the body's surface pressure:

| Code | Substance | Triple point | Boiling point at 1 bar | The world it makes |
|---|---|---|---|---|
| 0 | none | | | an airless or a too-hot body: no rain, no rivers, no snow |
| 1 | water | 273.16 K | 373 K | the Earth-like world, and the only one with a forest |
| 2 | carbon dioxide | 216.6 K at 5.2 bar | sublimes at 195 K | a cold thick world; the "snow" is dry ice |
| 3 | methane | 90.7 K | 112 K | a Titan-like world; the rivers run methane |
| 4 | ammonia | 195.4 K | 240 K | a cold mid-range world |

The lookup takes the surface temperature (charter row 4) and the surface pressure (row 9) and returns the
first row whose liquid range holds the temperature; the boiling and freezing points at `p_surf` come from
the same table through the Clausius–Clapeyron slope, evaluated by the producer, and ride as rows 11 and 12.
**This is a derivation with an anchor set, so it belongs to the taxonomy's house style, not to a draw.**

*In the game's words.* The pilot lands on a moon at 95 K with half a bar of nitrogen. The charter says
condensable 3. The classifier then reads a methane sea, methane rain, and a snow line where the methane
freezes. Nothing in the classifier changed: it read rows 10, 11 and 12 like any other number.

### 3.5 The cost of an obliquity, stated

The box in §5.2 is right that a tilt does not rotate the biome bands in the RECIPE, because the recipe lives
in the planet's own frame (V12) and the pole is `+Z` in that frame whatever the tilt is. It is not free of
the FRAME, the ROW and the PIN, and revision 1 did not say so.

- `crates/bins/tests/home_body_pin.rs:44-60` pins `POLE_AXIS = 2` **because the orbits' perifocal plane is
  `z = 0`**. With an obliquity the spin axis is no longer the orbit's axis, so the pin's REASON changes and
  the test must be rewritten to pin the spin axis against the parent-authored orientation.
- The tilt is carried by the parent-authored ORIENTATION, which already exists: `StampedPose` carries an
  `orient` quaternion (`crates/core/src/pose.rs:919`, the rotation law at 865-905). The parent authors a
  tilted, spinning orientation; the body holds it as a stamped reading (SL1 clause 2). No new field.
- `crates/bins/src/lib.rs:1004-1008` states the picture gate's method as *"the star's direction from the
  planet's centre is minus the planet's position in the system, in the planet's own frame while the planet
  does not spin"*. Every existing picture gate reads that. A spin makes it wrong, and each gate must read
  the orientation instead.

**This is a real cost, and C-2 must carry it.**

### 3.6 Where the charter rides, and the ONE SL6 ask

```text
   THE BODY'S OWN SHARD                                  THE CLIENT
   derives its own taxon from the seed at boot           links no motion crate
      (worldgen/body.rs:47-51 — no message)                 (Cargo.toml:24-26 — dev only)
   derives the charter with libm, quantises it           receives the charter in the realm's
   builds the body under the fence                       OWN self-look, beside the surface tag
   computes the climate                                  builds the SAME body, computes the
   extracts the surface (collision)                      SAME climate, extracts the SAME surface
              |                                                     ^
              +-------- BodyStmt::SelfLook, beside TAG_SURFACE -----+
                        on change, retained, inside SELF_LOOK_BUDGET_BYTES = 1200
```

**The SL6 statement, in the form the law requires.**
- **What data:** twenty-four whole numbers, about 64 bytes, per drawable round body.
- **From which realm to which:** it is NOT realm to realm. It is a realm's statement ABOUT ITSELF, on the
  bag that already carries its surface tag, to the clients the gateway composes for. The connection plane is
  not a realm (SL2's 2026-08-24 clarification).
- **How often:** once, on change only, retained. It never rides a per-tick lane. A dormant realm sends no
  self-look, so a dormant body ships no charter — and a dormant realm is never drawn, so nobody needs one.
- **Why the receiver cannot compute it:** the shipped client links no motion crate, by an isolation row
  (ruling S7-1). The facts come out of `powf`, `ln` and `cos`, which the fence forbids by compile error. A
  port of the taxonomy into the fence is a second implementation of one law, which V1.2 forbids by name.
- **What doing without costs:** the climate keeps today's two meaningless noises. Biomes then cannot depend
  on position, spin, trajectory, size or gravity — which is exactly what the owner asked for.
- **The precedent is exact:** `TAG_SURFACE` is already the realm's own statement about what shapes it, and
  ruling V6 row R-8 approved it. The 2026-09-05 suit ruling blesses the pattern by name — a thing states its
  rating as a fact about what it IS.
- **What it is NOT:** it is not on the parent's per-child bag. The one-radius law forbids that by name
  (`crates/core/src/look.rs:41-47`), and SL3 says the same: *"A parent's per-child message carries a
  PLACEMENT and nothing else."* Revision 1 broke both. Revision 2 does not.

**Does the charter tell a body where it is (SL1)?** Four rows — the insolation, the eccentricity, the
argument of periapsis and the year — are ORBIT facts. Two answers, and the owner should see both:
- **On the server there is no statement at all.** The shard derives them from the seed, as it already
  derives its mass and its atmosphere. Nothing is told to it, so SL1 clause 3 is not touched.
- **On the client the rows tell it nothing new.** The client already holds the star's row and the body's row
  in the same window, and ruling S7-7 already has it derive a light direction from them. A client that can
  place a star and a planet can already compute the distance between them.

**But the SHAPE is an owner decision, and revision 1 denied that one existed.** It is C-15.

### 3.7 The identity: where a charter mismatch is caught

Revision 1 folded the charter into the world identity. That is the wrong container, and the code says why:
`WorldIdentity` is ONE pair for the WHOLE world — a declared tag from the version and the seed, and a
measured half from the HOME body's eight golden chunks (`crates/terrain/src/tag.rs:19-50`). A charter is per
body and arrives long after login, so a pair the client stated at login can never catch a moon's charter.
Folding a per-body datum into a per-world tag would also reopen a world epoch every time any body's charter
moved.

RECOMMENDED: the charter is folded into the **per-body chunk digest's key material** — the digest already
folds the key and every cell (`crates/terrain/src/digest.rs:29-42`), and a charter that changes the climate
changes the cells anyway — and a mismatch is refused by the **per-realm surface refusal S7-3 already built**
(*"a foreign recipe tag is refused and counted, never drawn"*). One realm's surface is refused; the world
stands. §12, U5 measures it across targets.

**Whose charter wins on a disagreement.** The OWNING realm's. The shard derives it, states it, and computes
the collider from it; the client's copy is a reading it must match or be refused for that realm. There is
never a vote, and nothing merges.

---

## 4. How climate physics runs under a float fence

The fence bans every transcendental. Climate physics is full of them. Three techniques cover every case, and
each one has a test shape.

**Technique 1 — THE PER-BODY CONSTANT.** If the transcendental's argument is a per-body constant, the
producer (the shard, outside the fence) evaluates it once with `libm` and the charter carries the quantised
answer.
- *Example, and the one revision 1 missed.* The circulation's cell count needs an arcsine, and its cell
  edges need a sine each. Both depend only on the body's gravity, radius, scale height and day. So the home
  planet's shard computes "three cells, edges at sin 30° and sin 60°" once, and the pilot's client reads
  three whole numbers off the realm box. Neither one owns a sine.

**Technique 2 — THE FITTED POLYNOMIAL.** If the argument varies per column, fit a polynomial offline against
the exact function over the range that actually occurs, ship the coefficients as constants, and write a test
that walks the range and asserts the error bound.
- *Example, in the game's words.* The air over the pilot's valley holds more water when it is warm. The
  exact law is Clausius–Clapeyron in the Tetens form, `e_s = 610.78 · exp(17.27 T / (T + 237.3))` pascals. A
  degree-6 polynomial in `T/100` reproduces it over −80 °C to +60 °C, which is every temperature a landable
  body has. **UNMEASURED:** the error bound is owed (U9). The test walks 1 401 steps of 0.1 K against a
  table of exact values recorded offline.

**Technique 3 — THE TABLE AND THE BLEND.** Where a polynomial is awkward, ship a 256-entry table indexed by
a quantised argument and blend linearly between two entries. One integer index, two array reads, one
multiply and two adds. Fully determined.
- *Example, in the game's words.* The haze over the far mesas thins as the pilot climbs. Air density against
  height is `ρ = ρ₀ · exp(−z/H)`. Index by `z/H` in steps of 1/32 out to eight scale heights. Above that the
  answer is zero for every purpose the game has.

**The rule that binds all three.** *No transcendental function name appears anywhere in `vd-terrain`.* The
existing fence already enforces it by compile error (`crates/terrain/src/gf.rs:20-36`, with `compile_fail`
controls). The climate adds no exception, and §5.3 no longer asks for one.

**A fourth technique is REFUSED by name.** Do not approximate a transcendental with a Newton iteration or
with a series whose term count depends on the value. The term count becomes a branch on a float, the branch
is a coverage hole (HR5), and a value near the branch is a drift risk. Fixed-degree polynomials only.

**★ The same refusal binds this document's OWN iterations.** Revision 1 broke its own rule: it asked for a
distance transform over a closed surface and a moisture relaxation of "4 sweeps" without pinning the order.
Revision 2 states the rule that follows: **every grid pass has a FIXED number of rounds, in a FIXED face
order, with no early exit and no convergence test.** §5.1 states the round count and the order, and §12
measures the residual against an exact answer.

**And the fence does NOT bind the live weather.** The live layer runs only on the owning shard and is
shipped, never derived twice. It is Category C state in CLAUDE.md's own terms — checkpoint-carried, never
re-simulated — exactly like rapier's. It may use `libm` freely. §8.6.

---

## 5. The static climate field

### 5.1 The macro grid: one per body, and its resolution stated honestly

**Why a grid at all.** Two of the terms are not local:
- a rain shadow ACCUMULATES along the wind over hundreds of kilometres;
- the distance to the sea is a global property of the land.

**What revision 1 claimed, and what is true.** Revision 1 said the resolution is DERIVED from the recipe's
own coarsest wavelength and so is "not a magic number". The code refutes it: `long_wave_m` is clamped at
`LONG_WAVE_CAP_M = 400 000` (`crates/terrain/src/body.rs:24,150-153`), and the clamp binds for **every body
with a radius over 1 600 km** — that is, every planet a player will ever stand on. Both candidate home
bodies clamp (3 350 759 × 0.25 = 837 690; 6 515 500 × 0.25 = 1 628 875).

**So the honest statement is this.** The grid's target cell is

```text
   target_cell_m = long_wave_m / SAMPLES_PER_WAVELENGTH        SAMPLES_PER_WAVELENGTH = 8
   C = the smallest power of two with (face_edge_m / C) <= target_cell_m
```

On a small moon `long_wave_m` is a genuine seed draw. On a planet it is the constant 400 km, so
`target_cell_m` is the constant 50 km, and the grid resolution is a function of the RADIUS alone.
`SAMPLES_PER_WAVELENGTH = 8` is a **NAMED CONSTANT with a stated reason**, in the house style of
`CHUNK_EDGE = 62` and `TOP_RUNG_CHUNKS = 64`: a Nyquist argument needs two samples per wavelength to see a
wave at all, and eight are needed to keep its SHAPE, because a rain shadow reads the slope along the wind
and not just the peak. It is not a derivation, and §11's no-magic-numbers row now says so.

**The two candidate bodies, worked.**

| | current home | Earth-like candidate |
|---|---|---|
| ladder radius | 3 350 759 m (MEASURED, `home.rs:22`) | 6 516 379 m (computed here: `n = R·π/2` snapped) |
| rungs | 12 (MEASURED) | **13** (computed here: 2 580 chunks per face edge ⇒ top rung 12) |
| face edge at rung 0 | 5 263 360 cells | 10 235 904 cells |
| `face_edge / 128` | 41 120 m ≤ 50 000 ⇒ **C = 128** | 79 968 m > 50 000 ⇒ refused |
| `face_edge / 256` | | 39 984 m ≤ 50 000 ⇒ **C = 256** |
| samples over six faces | 98 304 | **393 216** |
| memory at 16 B per cell | **1.57 MB** | **6.29 MB** |
| build, ESTIMATED (§5.7) | about 25 ms | **about 100 ms** |

Revision 1 quoted 1.18 MB and 30 ms for the body its own §10 says is wrong. The right headline for the body
this document recommends is **6.29 MB and about 100 ms**. For scale: regional climate models on Earth run at
10–50 km, so a 40 km cell sits inside that band.

**The grid cell's 16 bytes, listed** (revision 1 borrowed the cell record's 12 bytes, which is a different
thing entirely):

| Field | Width | Why |
|---|---|---|
| coarse height above the sea | `f64`, 8 B | the same `Gf` the recipe uses; a narrower type would drift |
| distance to the sea, in grid cells | `u16`, 2 B | 65 535 cells of 40 km is more than any body's circumference |
| upwind sea fetch, 0–255 | `u8`, 1 B | the share of the upwind march that crossed sea |
| accumulated moisture | `u16`, 2 B | 1/16 of a millimetre-per-year unit |
| accumulated shadow depth | `u16`, 2 B | the drop across the upwind march |
| the sea mask and spare bits | `u8`, 1 B | one bit used |
| **total** | **16 B** | |

**The march across a cube face is a SHIPPED primitive, and revision 1 named the wrong one.** `site_of`
resolves a halo cell one step past a chunk's edge (`crates/terrain/src/lattice.rs:124-160`); it is not a
general walk. The right primitive already exists in `vd-seed`:

```text
   ONE MARCH STEP, in the BODY's own 3-D frame — no tangent frame, so no rotation at a seam
   ---------------------------------------------------------------------------------------
   d      = direction(face, a, b)            bend.rs:251   grid cell -> a unit direction
   d'     = normalize(d + step * wind_3d)    bend.rs:263   step along the wind, in 3-D
   face'  = face_of(d')                      bend.rs:224   which face the new direction lands on
   (a',b')= face_coords(face', d')            bend.rs:243   where on that face
   (i',j')= index_of(a'), index_of(b')        ladder.rs:157 which grid cell
```

Because the step is taken in the body's own frame, a face seam needs no rotation rule and a cube corner
needs no special case: `face_of` answers for every direction on the sphere. Revision 1's "the wind vector
must rotate at every seam" problem does not arise, and revision 1's hand-wave is replaced by five named
functions. **The gate stays owed:** a numeric test that a shared direction gives the same climate from both
faces, plus a picture across a face seam (U11).

**The distance to the sea, with a FIXED round count and a FIXED order.** A two-pass chamfer is correct on a
bounded rectangle, where every shortest path is monotone in the scan order. A cube sphere is CLOSED: a
column on the `+X` face may have its nearest sea across the `−X` face, and two raster passes cannot carry a
distance that wraps. RECOMMENDED: **four double sweeps (forward then backward) over the six faces in the
fixed face order PosX, NegX, PosY, NegY, PosZ, NegZ, with the neighbour graph read through the march
primitive above, no early exit and no convergence test.** Four is chosen because each double sweep carries a
distance across at least one whole face, and the cube's face graph has a diameter of three. **UNMEASURED:**
the residual against an exact bucketed Dijkstra on the home planet's grid, which is U13.

**★ The grid is SHARED with the hydrology domain.** Flow accumulation, base level and drainage basins want
exactly this grid: a coarse height, a sea mask and a neighbour walk. To build it twice is two mechanisms for
one job. The two domain documents must agree on one grid. §14, Q3.

**The client pays for this grid too, and it is on the critical path.** The grid is built on a worker before
the first chunk of that body is drawn, on the server and on the client alike: about 100 ms and 6.29 MB per
body on the Earth-like candidate. The home system holds 9 planets and 19 moons (`census.rs:57-63`), so a
client that visits them all holds up to about 180 MB if it never forgets one. RECOMMENDED, and it is a
decision row (C-16): the grid is built PER FACE, so the first chunk waits for one face (about 17 ms) and not
for six; and a body's grid is dropped when its realm leaves the window, exactly as its chunks are. The
arrival hitch is an SL8 measurement (U14), not an assumption.

### 5.2 Temperature

The mean annual surface temperature of a column:

```text
   T(d, h) = t_surface_k                            (charter row 4: T_eq + the greenhouse)
           + dT_lat(x)                              (the latitude profile, charter rows 14-15)
           - lapse * (h_climate - sea_radius)       (the lapse rate, charter row 13)
           + dT_cont(distance_to_sea)               (the continental term, from the grid)
           + T_noise                                (a small slow noise, for texture)
```

`h_climate` is the height at the CLIMATE RUNG, not at the rung being drawn. §5.10 states why.

**The greenhouse (charter row 4).** `t_surface = T_eq + dT_greenhouse`. The greenhouse warming is not
derivable from mass and orbit; it depends on what the air is made of. RECOMMENDED law, in the house style (a
named form, anchored, tested):

```text
   dT_greenhouse = T_eq * g_max * ( 1 - 1 / (1 + p_surf / p_ref)^a )
```

a saturating law with fitted constants. The anchor set is four measured bodies:

| Body | p₀ | T_eq | measured surface T | greenhouse |
|---|---|---|---|---|
| Mars | 610 Pa | 210 K | 215 K | +5 K |
| Earth | 101 325 Pa | 255 K | 288 K | +33 K |
| Titan | 146 700 Pa | 82 K | 94 K | +12 K |
| Venus | 9 200 000 Pa | 232 K | 737 K | +505 K |

**UNMEASURED:** the fit of `g_max`, `p_ref` and `a` to these four points (U7). The test reproduces each
anchor within a stated tolerance, exactly as the shoreline golden recomputes `SHORELINE_COEFF` from its
eight-body table (`taxonomy.rs:783-789,1466-1471`). Venus is the hard anchor and may need a second term for
a runaway greenhouse. If one fit cannot hold all four, the honest answer is to publish the residual, not to
hide it.

**The lapse rate, DERIVED, and the owner's "gravity" answered.** The dry adiabatic lapse rate is gravity
divided by the air's specific heat at constant pressure:

```text
   Gamma_dry = g / c_p ,   c_p = (7/2) * R_gas / mu   for a diatomic gas
```

Earth check: `9.81 / 1004 = 9.77 K per km` against the textbook 9.8 (computed here). The real environmental
lapse rate is smaller, because condensing water gives heat back: Earth's is 6.5 K per km, a factor of 0.665.
So `Γ = Γ_dry × (1 − w · humidity)`, with `w` fitted so that Earth's mean humidity gives 0.665.

**The game consequence, in the game's words.** The world's Earth-like planet has g 10.20 m/s², about 4 %
above Earth. Its dry lapse rate is 4 % steeper, so its permanent snow line sits about 4 % lower on every
mountain. A pilot who lands on a heavier world sees white ridges sooner. That is the owner's "size and
gravity" requirement, and it needs no new knob: it is `g / c_p`.

**The latitude profile, and the owner's "position, spin, trajectory" answered.** Let `x` be the sine of the
latitude. **Careful:** `crates/terrain/src/height.rs:45` computes `dir[POLE_AXIS].abs()`, which is
`|sin lat|`. That is enough for `P2(x)`, which is even, and NOT enough for a seasonal term, which needs the
hemisphere's sign. The climate reads the signed `dir[POLE_AXIS]`.

The annual-mean insolation follows the standard two-term Legendre form used in energy-balance climate models
(North 1975):

```text
   S(x) = S_mean * ( 1 + s2 * P2(x) ) ,   P2(x) = (3 x^2 - 1) / 2
```

`s2` is negative for a small tilt: the poles get less. It changes sign at a tilt near 54°, where the poles
begin to receive MORE annual light than the equator. `s2` is a function of the obliquity alone, so it is
charter row 15. Earth's value is about −0.48 (published).

The equator-to-pole contrast then depends on how well the atmosphere carries heat poleward. Thick air
carries much; no air carries none:

```text
   dT_pole_equator = dT_max / ( 1 + (p_surf / p_ref2)^m )        -> charter row 14
```

anchored on three measured bodies: the Moon (no air; the pole is about 100 K against a 390 K sub-solar
point), Earth (1 bar; about 45 K from equator to pole in the annual mean) and Venus (92 bar; a few kelvin).
**UNMEASURED:** the fit of `dT_max`, `p_ref2` and `m` (U7).

```text
   OBLIQUITY DOES NOT BEND THE BIOME BANDS. IT SETS HOW FAR THEY BREATHE.
   ----------------------------------------------------------------------
   The recipe lives in the planet's own frame (owner, V12). The spin axis is
   +Z in that frame whatever the tilt is, because the tilt is measured against
   the ORBIT, and the parent authors the orbit. So the tilt enters the STATIC
   field only through s2 (the annual-mean profile) and through the seasonal
   swing (5.5). It never rotates the bands.
   (What it is NOT free of: the pin, the parent-authored orientation and every
    picture gate that reads "the planet does not spin". See 3.5.)

   eps = 0        eps = 23        eps = 60        eps = 90
   pole  ####     pole  ####      pole  ....      pole  ....   <- warmest at the pole
         ####           ###             ....            ....
   30    ....           ...             ####            ####
   eq    ....           ...             ####            ####   <- coldest at the equator
   #### = cold ice,  .... = warm ground
```

**The continental term.** Far from the sea the summer is hotter and the winter colder. The annual MEAN moves
a little; the RANGE moves a lot (§5.5). The mean gets a small positive term with the distance from the sea,
read from the grid.

### 5.3 The circulation: what the spin decides

The owner asked for spin, Coriolis and the cells by name. Here is the model, with its fit re-run and its
misses published.

```text
   A HEMISPHERE OF A THREE-CELL PLANET (Earth-like: the day is about 24 h)

    pole 90  ..-- sinking, cold, DRY  ......................  the polar cell
             |     ground wind from the EAST
        60  -'-- rising, WET  ........................  the storm belt
             |     ground wind from the WEST
        30  -.-- sinking, hot, DRY  .................  THE SUBTROPICAL DESERT BELT
             |     ground wind from the EAST (the trades)
    eq   0  -'-- rising, hot, VERY WET  .............  the ITCZ, the rainforest belt

   A ONE-CELL PLANET (slow spin, Venus-like)   A SIX-CELL PLANET (fast spin, Jupiter-like)
    pole  ..-- sinking, DRY                      pole ..--..--..--..--..--..--  many thin bands
          |                                            the wind reverses at every band edge
    eq   -'-- rising, WET
```

**How many cells, DERIVED from the spin — OUTSIDE the fence.** The Hadley cell reaches a latitude set by the
thermal Rossby number (Held & Hou 1980):

```text
   sin(phi_H) = sqrt( 5 * dH * g * H / (3 * Omega^2 * a^2) )
   Omega = 2 pi / day_s ,  a = radius_m ,  H = scale_height_m
   cells_per_hemisphere = clamp( round(90 deg / phi_H), 1, 6 )
```

The square root is inside the fence; **the arcsine that turns `sin φ_H` into a latitude is not, and neither
are the sines of the cell edges.** So the producer evaluates all of it once and the charter carries
`cells_per_hemisphere` (row 22) and `cell_edge_sin_q15[6]` (row 23). Revision 1 claimed this ran inside the
fence; it cannot.

**The anchor table, RE-RUN and honest.** Computed here at `dH = 0.388`, from published body values
(g, scale height, day, radius):

| Body | φ_H computed here | cells this gives | reality | verdict |
|---|---|---|---|---|
| Earth | **30.08°** | 3 | 3 | ✓ |
| Mars | **42.78°** | **2** | usually described as ONE strong cross-equatorial cell | **MISS** |
| Venus | **90°** (`sin ≥ 1`) | 1 | 1 | ✓ |
| Jupiter | **3.06°** | 6 (clamped) | many bands | ✓ |

Revision 1 published this table with a ✓ on Mars and a φ_H of 90°, and with 1.9° for Jupiter. Neither
reproduces. The correct statement is: **`dH = 0.388` is a REVERSE FIT to Earth alone, it reproduces Venus
and Jupiter, and it MISSES Mars by one cell.** Mars is the hard anchor, because a thin-air, slow-ish, small
body is exactly the case a moon of the home system will hit. **UNMEASURED and owed (U8):** a fit of `dH`
over a wider anchor set, published with its residual per body, in the shape of the shoreline golden. If no
single `dH` holds Earth and Mars together, the honest answer is a two-term law or a published miss — not
a ✓.

**The wind direction at a column.** The cell index comes from comparing the signed `x` against the charter's
cell edges. The zonal direction alternates east, west, east from the equator outward. The Coriolis sign
flips with the hemisphere. The meridional part points toward the cell's rising branch. Everything is a
comparison and a multiply: about 20 operations per column.

**The wind is a static field, and that is on purpose.** The prevailing wind carves the rain shadow, and a
rain shadow is a shape of the world. The GUST is weather, and it is live (§8.5).

### 5.4 Precipitation, at three scales

Revision 1 had ONE spatial term at 41 km. A 41 km cell means the owner's whole 50 km vista sits inside one
or two samples, so the only thing that varies inside the picture is the height — and the visible result is
biomes as elevation contour bands, a green band then a brown band then a white band following an isoline
around every hill. That is not the picture. The picture's most striking property is a green windward valley
beside a dry leeward cliff **at the same height, kilometres apart.**

**So the model carries three scales, and each one has its own cost.**

```text
   P(d, h) = supply(T_source, sea_fetch_upwind)      how much water the air carries   [GRID]
           * cell_factor(x)                          ITCZ wet, 30 deg dry, 60 deg wet [closed form]
           * shadow_grid(accumulated_drop)           the range-scale rain shadow      [GRID, 40 km]
           * shadow_mid(dh at 5-15 km upwind)        ONE range's windward/leeward     [MID]
           * shadow_local(slope along the wind)      ONE RIDGE's two faces            [LOCAL]
           * continental(distance_to_sea)            dry far inland                   [GRID]
           * K_mm_per_year                           THE UNIT CONSTANT                [anchored]
```

| Term | Scale it resolves | How it is computed | Cost per rung-0 chunk |
|---|---|---|---|
| GRID | 40–400 km: the shadow behind a whole range, the fetch, the distance to the sea | built once per body; one blended lookup per column | 0.02 ms |
| MID | 5–15 km: the windward and leeward sides of ONE range | one extra coarse-height evaluation at the upwind point, 6 octaves | **0.30 ms** |
| LOCAL | 0.1–2 km: the two faces of ONE ridge | a central difference over the column pass's OWN neighbours, dotted with the wind | 0.05 ms (the halo only) |

**The LOCAL term is nearly free, and it is the one that makes the picture.** The column pass already holds
62 × 62 heights (`chunk.rs:174-192`). A central difference needs one column of halo on each side, which the
extractor's halo rule already establishes (S6-2: *"the neighbour cells are computed by the same per-cell
rule from the seed, never fetched"*). A 64 × 64 column pass costs `(64² − 62²)/62² = 6.6 %` more, that is
**+47 µs at rung 0** (DERIVED FROM A MEASUREMENT: 6.6 % of 710 µs). The term is then `slope · wind`, four
array reads and three multiplies.

*In the game's words.* The pilot stands in a valley. The ridge in front of him faces the wind, so its slope
term is positive and its face is forest. He flies over the crest; the far face's slope term is negative, the
rain has already fallen, and the ground under him is scrub over bare rock. The two faces are 800 m apart and
at the same height. THAT is the reference picture, and it comes from four array reads.

**A per-column upwind MARCH is still refused, but for a different number than revision 1 gave.** Recomputed
at the MEASURED 13.19 ns per octave: `16 steps × 3 octaves × 13.19 ns = 633 ns per column × 3 844 columns =`
**2.43 ms per chunk**, not the 5.1 ms revision 1 stated.

| Chunk | MEASURED cost today | headroom to 8 ms | the march plus the climate | verdict |
|---|---|---|---|---|
| surface | 3.3 ms | 4.7 ms | 2.43 + 0.44 = 2.87 ms | it fits |
| **cave-dense** | **6.1 ms** | **1.9 ms** | **2.87 ms** | **it does NOT fit** |

A budget is sized on the worst case, and ruling V10 set 8 ms after the 4 ms budget failed on exactly the
cave-dense chunk. So the march is refused — on the worst chunk, by a MEASURED headroom, and not on the
number revision 1 published. The three-scale model above costs **0.44 ms** and fits both.

**The unit constant, which revision 1 left out entirely.** `supply` is a saturation vapour pressure, in
pascals. The classifier's axis is millimetres of rain per year. Nothing converted one into the other.
RECOMMENDED: one constant `K_mm_per_year`, with units of millimetres per year per pascal of supply,
**anchored so that an Earth-like charter over an Earth-like ocean at the ITCZ returns about 2 000 mm/yr and
under the subtropical high returns about 100 mm/yr**. That is a two-point anchor with published residuals,
in the same shape as the greenhouse fit. **UNMEASURED and owed (U15).** Without it the biome chart is read
in unknown units, and the whole map's realism rests on a number nobody wrote down.

**The cell factor, stated correctly.** Revision 1 said *"This one term places every large desert on Earth."*
That is false and a reviewer will catch it in a minute. The subtropical-high term places the Sahara, the
Arabian, the Kalahari and the Australian deserts. It does NOT place the Gobi and the Taklamakan (continental
interior plus rain shadow), Patagonia (rain shadow), the Atacama and the Namib (cold upwelling coastal
currents plus subsidence) or the polar deserts (cold). The honest sentence: **the cell term places the
subtropical desert belts; the shadow and continental terms place most of the rest; cold coastal currents are
out of scope, and this model will not produce an Atacama.**

### 5.5 The seasonal range (a static number, not a live one)

```text
   dT_season(x) = A_obliquity(eps) * seasonal_shape(x)      the tilt term
                + A_eccentricity(e)                          the orbit-shape term
                * continental_gain(distance_to_sea)          land swings, sea does not
                / thermal_inertia(p_surf, sea_fraction)      thick air and deep water damp it
```

- The **tilt term** is what makes a season at all. It grows with the latitude and with the tilt.
- The **orbit-shape term** comes from the eccentricity: the light at periapsis over the light at apoapsis is
  `((1+e)/(1−e))²`. At Earth's e = 0.0167 that is 1.069, about 7 %. **This world's eccentricity is
  hard-capped at 0.12** (`crates/physics/src/worldgen/config.rs:61-71`), so the largest possible ratio is
  `((1.12)/(0.88))² =` **1.62** (computed here). Revision 1 used e = 0.2 and 2.25, which this world cannot
  draw; §9's "high-eccentricity world" row is re-based on 0.12.
- `arg_periapsis` decides WHICH hemisphere gets its summer at periapsis. Earth's northern winter falls at
  periapsis, which softens it. That asymmetry is free: the element already exists.
- The **damping** is why a coast is mild and a continent is not.

**Why this is static.** It is the SIZE of the swing, not the swing. The swing itself is the almanac (§8).

### 5.6 The sea, the ice line and the permanent snow line

Today every cell below `sea_radius_m` is water, and the seed draws that radius
(`crates/terrain/src/body.rs:186-190`). The climate makes three changes, and each is one comparison against
a CHARTER number rather than a hard-coded 273 K:

1. **Sea ice.** Where the annual mean surface temperature is below `freeze_q4_k` (charter row 11), the top
   of the sea is `Ice`, not `Water`. A planet grows its own polar caps, at the latitude where the profile
   crosses freezing. Nothing is placed by hand. On a methane world the sea and its ice are methane, because
   row 11 came from row 10.
2. **The permanent snow line.** Where `T(d, h_climate)` is below `freeze_q4_k`, the topsoil is `Snow`
   whatever the biome says. The line is a height, and it falls toward the pole because the latitude term
   falls. **This is what puts white on the far ridges in the owner's picture**, and it is on the horizon at
   every rung because §5.10 makes it rung-independent.
3. **A frozen world and a boiled world.** If the mean is below `freeze_q4_k` everywhere, the body holds no
   liquid and no rain. If the mean is above `boil_q4_k`, the same happens at the other end.

**A stated limit: there is no ice-albedo feedback.** On Earth the ice line is sharp and hysteretic because
ice is bright, bright ground stays cold, and the line reinforces itself; that feedback is why a Snowball
Earth is possible at all. Here `bond_albedo_q10` is a per-body constant and the temperature field is a
smooth analytic profile, so the sea-ice edge is a circle of constant latitude plus a small noise, on every
ocean of every world. It will read as a drawn line, and the runaway states §9 wants arrive as a threshold on
a mean instead of as a bistable outcome with a history.

RECOMMENDED cheap cure, and it costs one grid pass: after the temperature field is built, run **two fixed
rounds** in which a grid cell below freezing raises the local albedo by a fixed step and the temperature is
re-evaluated. Two rounds are not a feedback solved to convergence; they are enough to make the line
irregular and to widen it where the ground is high. Fixed rounds, fixed order, no convergence test — the §4
rule. **UNMEASURED**; the picture decides whether two rounds are enough (U12).

### 5.7 The cost, against the 8 ms budget — every number recomputed

**The base rate, MEASURED:** the column pass costs 710 µs for 3 844 columns at 14 octaves
(`slice_05_generator.md:369-379`), so **one octave evaluation costs 13.19 ns**. Revision 1 used 27.9 ns from
a number that was never a measurement of this pass; every figure below is re-derived from 13.19 ns.

| Item | Per what | Cost | Source |
|---|---|---|---|
| a surface chunk today | rung-0 chunk | 3.3 ms | MEASURED, ruling V10 |
| a cave-dense chunk today | rung-0 chunk | 6.1 ms | MEASURED, ruling V10 |
| the budget | rung-0 chunk | 8.0 ms | ruling V10 |
| **the headroom on a surface chunk** | | **4.7 ms** | |
| **the headroom on a cave-dense chunk** | | **1.9 ms** | **the one that binds** |
| the climate's closed-form part, about one octave-equivalent per column | 3 844 columns | 0.05 ms | ESTIMATED at 13.19 ns |
| one grid lookup and blend per column | 3 844 columns | 0.02 ms | ESTIMATED |
| the MID orographic term, 6 octaves at the upwind point | 3 844 columns | **0.30 ms** | ESTIMATED at 13.19 ns |
| the LOCAL slope term (the halo's extra columns) | 3 844 columns | 0.05 ms | DERIVED: 6.6 % of 710 µs |
| the vegetation anchor scatter (one hash per candidate cell) | 3 844 cells | 0.02 ms | ESTIMATED |
| **the climate's total, rung 0** | | **0.44 ms** | **ESTIMATED** |
| the CLIMATE RUNG's extra octaves at a COARSE rung (at most 5) | 3 844 columns | +0.25 ms | ESTIMATED, §5.10 |
| **the climate's total, the top rung** | | **0.69 ms** | ESTIMATED |
| a per-column upwind march (REFUSED) | 3 844 columns | 2.43 ms | ESTIMATED at 13.19 ns |
| the grid build, current home body (C = 128) | once per body | about 25 ms | ESTIMATED |
| **the grid build, Earth-like candidate (C = 256)** | once per body | **about 100 ms** | **ESTIMATED** |
| **the grid memory, Earth-like candidate** | once per body | **6.29 MB** | computed from 16 B × 393 216 |

**The conclusion, quoting BOTH chunks as ruling V10 does:** the climate costs **9 % of the headroom on a
surface chunk and 23 % on a cave-dense one**. Revision 1 quoted the easy chunk only. Everything expensive
happens once per body. **UNMEASURED:** every row marked ESTIMATED. §12 names the bench.

### 5.8 ★ The sea problem: a 1 %-ocean world returns desert everywhere

**MEASURED, and already reported to the owner by the slice-5 record:** the home planet's sea stands 5 297 m
under the ladder radius against a relief of 14 305 m, so **the sea covers about one column in a hundred**
(0 to 5 of 300 sampled columns per rung; `slice_05_generator.md:388-392`, repeated in ruling V9).

Two of the four supply terms in §5.4 read the sea: the upwind fetch and the distance to the sea. On a
1 %-ocean world every column's upwind fetch is land and every column's distance to the sea is thousands of
kilometres. The supply term is near zero everywhere, the continental term multiplies it down again, and the
chart returns Desert or ColdDesert for the whole planet. No forest, no wetland, no river, no beach — none of
the reference picture.

**This is INDEPENDENT of §10.** Even after the home body is re-picked, the sea radius stays a seed draw
between 40 % of the relief below the ladder radius and 30 % above it (`body.rs:186-190`), so a 1 %-ocean
world is an ordinary outcome of the draw, not an accident of this one body.

**The three ways out, and the owner chooses (C-17).**

1. **Leave it.** A 1 %-ocean world is a legitimate world; it is just not the reference picture, and the
   vista must be judged on a different body. Cost: the home planet is not the picture's planet.
2. **Re-draw the sea offset for the home body only.** Refused: SL5 forbids a variant, and a per-body
   exception is a variant with a friendly name.
3. **Make the sea a HYPSOMETRIC law instead of an offset.** The draw states the world's OCEAN SHARE — the
   fraction of the surface under water — and the sea radius is then SOLVED so that the share comes out
   right, by reading the height histogram off the climate grid that §5.1 builds anyway. The share's own draw
   is bounded by a physical prior (Earth 71 %; the solar system's rocky bodies run 0 % to 100 %; a band of
   [0.2, 0.85] for a body that keeps a condensable, 0 otherwise). Cost: one extra pass over the grid, about
   2 ms, and a recipe version bump. **This is the recommendation**, because it makes the ocean share a
   stated fact about the world instead of a side effect of an offset, and because every seed then draws a
   world with a coastline.

*In the game's words.* Today the seed says "put the sea 5 297 m below the ladder radius" and the world
happens to be a desert continent with puddles. Under the hypsometric law the seed says "this world is 63 %
ocean", and the generator finds the height that makes it so. The pilot lands on a coast because the world
HAS one.

### 5.9 ★ Isostasy: the recipe's relief is not checked against gravity

The brief names isostasy and revision 1 never used the word. Rock floats on rock: a mountain's root sinks
into the softer layer under it, and the crust fails when the load is too great. The tallest a mountain can
be falls roughly as one over the surface gravity:

```text
   h_max ~ sigma / (rho * g)
   Earth:  9 km at 9.81 m/s^2   ->  sigma/rho = 88 300 m^2/s^2   (computed here)
   Mars:   88 300 / 3.71        =  23 800 m   against Olympus Mons's 22 km   (checks)
```

The shipped relief law is `relief = 0.4 % of the radius, clamped 200–12 000 m, times [0.5, 1.5)`
(`crates/terrain/src/body.rs:145-148`). It grows with the RADIUS. Gravity also grows with the radius, so the
law runs the wrong way: a bigger, heavier body gets MORE relief where physics gives it less.

| Body | g | isostatic ceiling `88 300/g` | what the recipe draws | verdict |
|---|---|---|---|---|
| current home | about 3.3 m/s² (ESTIMATED, §10) | 26 800 m | 14 304 m (MEASURED) | under the ceiling ✓ |
| Earth-like candidate | 10.20 m/s² (MEASURED, `census.rs`) | **8 660 m** | **6 000–18 000 m** | **the upper half of the draw is unphysical** |

**What it costs the picture.** At the 6.5 K/km lapse rate, 14 304 m of relief is 93 K from the sea to the
peak, so the permanent snow line sits below most of the land and the planet reads white. On the Earth-like
candidate an 18 km relief at 10.2 m/s² would give a mountain twice as tall as anything the crust can hold,
and a snow line so low that the forest belt vanishes.

**RECOMMENDED (C-18):** the relief's cap becomes `min(the present cap, ISOSTATIC_COEFF / g)` with
`ISOSTATIC_COEFF` anchored on Earth and Mars and tested against both, in the taxonomy's house style. The
relief law is domain 02's, not this document's, so **this is a finding handed over**, with the number, not a
change proposed here. §14, Q8.

### 5.10 ★ The climate rung: why the biome never changes with the detail level

**The defect, which exists today.** `biome_at` reads `surface_m` at whatever rung the caller built
(`crates/terrain/src/height.rs:40-46`; `crates/terrain/src/chunk.rs:180-183`). The coarse answer of a rung
drops the fine octaves, and the height it returns differs from rung 0's by `dropped_bound_m`
(`crates/terrain/src/body.rs:274-276`). Computed here for a relief of 12 000 m at roughness 0.5:

| rung | dropped share of the relief | metres |
|---|---|---|
| 3 | 0.00043 | 5.1 |
| 6 | 0.00385 | 46.1 |
| 8 | 0.01556 | **186.8** |
| 11 | 0.12495 | **1 499.4** |

A vista draws its far ground at rung 8 and beyond (ruling V10 §12). At rung 8 a column's height is up to
187 m away from its rung-0 height. §5.6 makes the snow line a comparison on exactly that height. **So the
far ridge is white and the near ridge is not, and the white cap MOVES as the pilot flies in.** That is the
"detail-by-box" and "arrival-pop" seam of the eleven-seam taxonomy, on the most visible feature of the
owner's reference picture.

**Revision 1 claimed the ladder gate held, with `dropped_bound_m` as the bound. That is wrong in kind:** a
bound in METRES cannot bound a CLASS. One metre of height flips a column across a threshold, and a threshold
has no tolerance.

**The cure is not a tolerance. It is to read ONE height.**

```text
   THE CLIMATE RUNG
   ----------------
   Every rung's classifier reads h_climate = height_m(body, dir, CLIMATE_RUNG),
   never height_m(body, dir, the rung being drawn).

   rung 0 drawn:   height sums 14 octaves; h_climate is a PARTIAL SUM of them   -> free
   rung 6 drawn:   height sums  8 octaves; h_climate is the same 8 octaves      -> free
   rung 11 drawn:  height sums  3 octaves; h_climate needs 5 more               -> +0.25 ms

   Consequence: the biome, the snow line and the sea-ice line of a column are
   THE SAME VALUE at every rung, by construction. The gate is assert_eq over the
   golden columns, not a share-of-columns bound.
```

**The climate rung is DERIVED, not chosen.** It is the coarsest rung whose dropped bound is worth less than
a stated temperature tolerance through the body's OWN lapse rate:

```text
   CLIMATE_RUNG = the coarsest L with dropped_bound_m(L) <= T_TOL_K / lapse_k_per_m
   T_TOL_K = 0.5 K      a named perception constant: half a kelvin is under the width of
                        the blend the painter dithers across a biome edge, so no eye can
                        see a class decided half a kelvin either way.
```

On the home planet at 6.5 K/km, `0.5 / 0.0065 = 77 m`, and the table above gives rung 6 (46 m ✓, rung 8 is
187 m ✗). So `CLIMATE_RUNG = 6` on that body — 64 m cells — and it is a per-body number that falls out of
the body's own relief, roughness and lapse rate. On a heavier body with a steeper lapse rate the tolerance
buys fewer metres and the climate rung is finer, which is correct: a steeper lapse rate makes the snow line
more sensitive to height.

**Two prices, stated.**
- At a rung COARSER than the climate rung, the column pass pays the extra octaves: at most five on the home
  planet, **+0.25 ms per chunk** (ESTIMATED at 13.19 ns). The ladder law still holds — every rung's column
  pass costs at most rung 0's — but the MEASURED fall from 710 µs to 266 µs becomes a fall to about 430 µs,
  and the bench must be re-measured (U16).
- The drawn surface at a coarse rung is the coarse hill, so a white cap's EDGE follows the climate rung's
  contour while the ground under it is up to `dropped_bound_m` away. That is a paint offset, and it is
  CONTINUOUS: it shrinks as the pilot flies in and the rung refines. It is not a class flip, so it does not
  pop.

**The blend, on top.** The classifier's thresholds carry a blend width, and the painter dithers across it,
so a biome boundary is a gradual change of ground and never a contour line. The blend is a paint rule; the
climate rung is what makes the CLASS stable. Both are needed.

---

## 6. The biome classifier

### 6.1 The two charts, explained

**Whittaker (1975)** plots mean annual temperature against mean annual precipitation and cuts the plane into
biomes. It is simple, it is famous, and it captures most of what an eye sees.

**Holdridge (1947)** uses three axes: biotemperature, precipitation, and the ratio of potential
evapotranspiration to precipitation. The third axis separates a hot dry savanna from a cool dry steppe at
the same rainfall.

RECOMMENDED: **Whittaker with Holdridge's aridity term.** Two fields we compute plus one ratio we derive.
Three inputs, one table lookup, no branch on a body kind.

```text
   MEAN ANNUAL PRECIPITATION (mm/yr, through K_mm_per_year of 5.4)
   4000 |                              TROPICAL RAINFOREST
        |
   3000 |  TEMPERATE RAINFOREST
        |
   2000 |     TAIGA        TEMPERATE FOREST      TROPICAL SEASONAL FOREST
        |
   1000 |            WOODLAND                    SAVANNA
        |
    500 |   TUNDRA        GRASSLAND              SHRUBLAND
        |
    200 |            COLD DESERT                 DESERT
        |
      0 +----------------------------------------------------------------
        -15    -5     5     15     25     35   MEAN ANNUAL TEMPERATURE (deg C)

   The overrides run in a FIXED order, before the chart is read:
     1. no condensable (charter row 10 == 0)  -> REGOLITH  (ICE where it is cold and shadowed)
     2. below the sea radius                  -> OCEAN, or SEA_ICE below freeze_q4_k
     3. within one cell of the sea            -> BEACH
     4. above the permanent snow line         -> ICE
     5. slope steeper than a limit            -> ALPINE (bare rock; today's Highland)
     6. a closed basin with inflow            -> WETLAND
     otherwise                                -> the chart
```

### 6.2 The biome set: nineteen values in five bits, counted once

Revision 1 gave four different counts and drew a biome (`WOODLAND`) that appeared in no list. Here is one
list, counted.

**The twelve chart biomes:** `TropicalRainforest`, `TropicalSeasonalForest`, `TemperateRainforest`,
`TemperateForest`, `Taiga`, `Woodland`, `Savanna`, `Shrubland`, `Grassland`, `ColdDesert`, `Desert`,
`Tundra`.

**The seven override biomes:** `Ocean`, `SeaIce`, `Beach`, `Ice`, `Wetland`, `Alpine`, `Regolith`.

**Twelve plus seven is nineteen.** Four bits hold sixteen, so **five bits**, with thirteen values spare. The
five bits live in the generator's own `Biome` enum (today four values,
`crates/terrain/src/strata.rs:114-131`) and in the client's painter. **They are not a wire field and not a
store field**, because a biome is a COLUMN property both hosts derive; only an edited cell gets a record
(`topic_00_format_sitting.md:34-37`). C-7 therefore approves a SET SIZE and the version bump it costs, not a
wire width — revision 1 asked the owner to approve a field that does not exist.

### 6.3 The classifier is one function, on every body

HR4 holds without effort: the classifier reads three numbers, a charter row and a table. It never asks what
kind of realm it is on. A planet's shard, a moon's shard and a hull that holds terrain (ruling V4: a hull MAY
hold terrain) all call the identical function. A hull's charter is its own — see §14, Q5.

**How HR5 gets to 100 % on one world.** `crates/terrain/src/home.rs:5` requires every unit test of the crate
to run on the home body, and one body cannot show nineteen biomes: `Regolith` needs an airless body,
`SeaIce` needs a frozen ocean, the terminator ring needs a locked world. The crate already solves exactly
this: `crates/terrain/src/height.rs:112-140` drives *"forced cases, so every arm is driven whatever the seed
draws"*. The climate's equivalent is to **construct charters directly in tests** — a charter is DATA about a
body of the one world, so a test that builds "the charter of an airless rock" is not an invented world and
not an SL5 variant, any more than a test that forces a latitude is. §11 states this in both the SL5 row and
the HR5 row, which revision 1 left contradicting each other.

---

## 7. What the biome decides

### 7.1 The soil, extended to all nineteen rows

Today `StrataTable::at(biome, depth)` picks a topsoil, a subsoil, a sediment and the bedrock
(`crates/terrain/src/strata.rs:180-210`). The climate makes the topsoil real. **Every one of the nineteen
biomes gets a row** — revision 1 covered eleven and silently left six with no soil.

| Biome | Topsoil | Subsoil | Why (the real soil name) |
|---|---|---|---|
| TropicalRainforest | thin red soil (`Laterite`) | deep weathered clay | laterite: the rain takes the nutrients down |
| TropicalSeasonalForest | red-brown loam | clay | a drier laterite |
| TemperateRainforest | deep dark loam | clay | heavy leaf fall, heavy rain |
| TemperateForest | brown loam | clay | the reference picture's forest floor |
| Taiga | thin acid soil (`Podzol`) | leached sand | podzol: cold wet ground leaches it |
| Woodland | thin loam | gravel | open canopy, thin soil |
| Savanna | loam over a hard pan | clay | seasonal rain, seasonal drought |
| Shrubland | stony loam | gravel | |
| Grassland | deep dark soil (`Loam`), 1–2 m | clay | chernozem: grass roots build it |
| ColdDesert | stone pavement (`Gravel`) | sandstone | freeze and thaw, no rain |
| Desert | dune sand or stone pavement, by the aridity | sandstone | a hyper-arid index gives dunes; a merely arid one gives pavement |
| Tundra | a thin active layer (`Dirt`) | `Permafrost` | the active layer's DEPTH is derived, not drawn |
| Ocean | `Water` | sediment | |
| SeaIce | `Ice` | `Water` | new |
| Beach | `Sand` | `Gravel` | |
| Ice | `Ice` | `Ice` | |
| Wetland | `Peat` | clay | waterlogged ground stops decay |
| Alpine | `Gravel` | bedrock | today's Highland row, kept |
| Regolith | shattered rock (`Regolith`) | bedrock | an airless body has no soil at all |

**Five new strata are APPENDED after `Empty` (code 18):** `Loam` 19, `Podzol` 20, `Laterite` 21, `Peat` 22,
`Regolith` 23. To append is the shipped pattern — `Empty` itself was appended in slice 6 — and it bumps the
recipe version without moving any existing code (`crates/terrain/src/strata.rs:41-61`).

**Do the new soils break the seed law (S5-6: no indicator material)?** The question must be asked, and
revision 1 asserted the answer. Laterite is the world's bauxite and nickel proxy; peat is fuel. A publicly
computable map that names where laterite lies IS an indicator material by the ruling's own words. **The
answer that keeps the law:** a soil is a SURFACE FACT anyone can see by standing on it — a rainforest floor
is red whether or not the map says so — while the ORE inside it stays live state and is not a function of
the soil. So the soil names the CLIMATE, not the deposit. A rainforest is not a bauxite mine; it is a place
where bauxite MIGHT be, and where it actually is stays live state. **This is a decision row (C-19), not an
assertion**, and if the owner disagrees the cure is to name the soils by look (`RedSoil`, `DarkSoil`,
`SourSoil`) rather than by ore-bearing type.

**The active layer, worked as an example.** On the home planet's tundra the summer thaws the top of the
permafrost. The depth follows from the summer degree-days and the soil's thermal properties — a square root
of the warm season's length times the degree-days. A square root is inside the fence. A pilot who digs on
the tundra hits frozen ground at a depth the climate decided, not at a depth somebody typed.

### 7.2 The vegetation skeleton (ruling V4: art assets, never drawn primitives)

The generator emits ANCHORS. It never draws a tree.

```text
   for each candidate site in the chunk (a jittered grid, one hash per site):
       density   <- biome, moisture, slope, height above sea, ASPECT
       accept    <- hash(seed, chunk, site) < density
       kind      <- the biome's kind mix, drawn from the same hash
       param     <- size, age, lean, drawn from the same hash
   emit (cell, kind, param)          <- the client blends the ART ASSET
```

- **Aspect.** On the home planet's northern hemisphere the south-facing slope gets more sun, so the trees
  climb higher there and the north-facing slope holds snow later. It is one dot product between the slope
  normal and the sun's mean direction — a per-column constant times the normal the extractor already
  computes.
- **The anchor rate.** A rung-0 chunk covers 3 844 m². A temperate forest at about 0.06 stems per square
  metre gives **231 anchors per chunk** (computed; 600 stems per hectare is a real temperate density). At 8
  bytes each that is 1.8 KB, against the 405 KB of geometry the chunk already carries (MEASURED).
- **The scatter's cost.** 3 844 hashes at about 5 ns is **0.02 ms per chunk** (ESTIMATED).
- **★ Where an anchor lives in the format, which revision 1 never answered.** A tree a player can stand on
  is part of the SHAPE (ruling V4 step 9(d) derives the collider from the server-side skeleton), so the two
  hosts must agree about it exactly as they agree about a cell. RECOMMENDED: the anchors of a chunk are
  folded into an **ANCHOR DIGEST**, a third digest beside `digest_of` and `mesh_digest`
  (`crates/terrain/src/digest.rs:29-42,47-60`) — exactly as slice 6 added the mesh digest beside the cell
  one. An anchor is DERIVED, never stored. It becomes a RECORD only when a player changes it: a felled tree
  writes one record with provenance *"biome feature"*, which the format already reserves
  (`topic_00_format_sitting.md:54`).
- **The far canopy fold** (ruling V10 §12): at a coarse rung the same rule yields a canopy HEIGHT and a
  canopy DENSITY per coarse cell, which the client draws as a textured surface. It falls out of the same
  biome function evaluated at the coarse rung. No second mechanism, and no impostor.

### 7.3 What the eye in the owner's picture is actually seeing — and what it will NOT see

| In the picture | The field that makes it |
|---|---|
| snow-capped ridges | the permanent snow line: `T(d, h_climate)` below `freeze_q4_k` (§5.6), stable at every rung (§5.10) |
| green forest as a mass | the biome from temperature and precipitation, plus the canopy fold at range |
| open fields | grassland, where the aridity sits between desert and forest |
| a river in the valley | the hydrology domain, fed by THIS document's precipitation field |
| blue haze with distance | **NOT the scale height alone.** See below. |
| clouds | the almanac's nominal cover, perturbed by the live weather systems (§8.5) |
| the contrast between one slope and the next | the LOCAL and MID orographic terms (§5.4) — this is the term revision 1 lacked |
| **dry cliffs and rock pillars** | **NOT the rain shadow.** See below. |

**★ Two rows revision 1 got wrong, corrected.**

**The haze is an optical depth, not a scale height.** Aerial perspective is set by the optical depth along
the line of sight: the surface number density (from `p_surf`, `mu` and the temperature — charter rows 9, 7
and 4), times a per-gas scattering cross-section, times the path length, and it is wavelength-dependent
(Rayleigh scattering goes as one over the wavelength to the fourth, which is WHY it is blue). The scale
height gives only the vertical falloff. The CROSS-SECTION depends on which gas, and charter row 10 (the
condensable) plus row 7 (the bulk mean molecular weight) are what name it. On a carbon-dioxide world the
haze is a different colour and a different depth from a nitrogen world at the same pressure.

**The rock pillars do not follow from the climate, and cannot on today's recipe.** A hoodoo is DIFFERENTIAL
erosion of layered rock of unequal hardness. The shipped strata are horizontal layers of a single drawn
thickness for the whole body — one topsoil, one subsoil, one sediment depth, one sediment kind, one bedrock
(`crates/terrain/src/body.rs:193-201`). **There is no lithological contrast anywhere on any planet**, so
nothing can erode differentially. The aridity term changes a topsoil byte; it changes no cell's hardness and
no cell's position.

**And that is the deepest believability gap in this document, stated plainly.** A geologist tells an arid
range from a humid one in one photograph at any distance: an arid landscape has dunes, yardangs, mesas,
alluvial fans and sharp scarps, because there is no soil creep and no vegetation to hold the regolith; a
humid one has rounded convex hilltops and soil-mantled slopes. **In this model a desert is a grassland with
a sand-coloured topsoil byte.** The cure is not in this domain: it is the climate feeding the EROSION
domain's rate laws. §12 states the handover as a binding cross-domain requirement, and §14, Q9 asks the
owner for varied bedding.

---

## 8. The weather

### 8.1 The three layers, corrected

```text
   +----------------------------------------------------------------------+
   |  STATIC — THE CLIMATE            f(seed, charter, address)            |
   |  temperature, precipitation, wind, seasonal range, biome, soil,       |
   |  permanent snow, permanent sea ice, the vegetation skeleton           |
   |  WHO: both hosts compute it. Nothing crosses. SL10 clause 1.          |
   +----------------------------------------------------------------------+
   |  THE ALMANAC — CLOSED-FORM TIME  f(seed, charter, tick, the window)   |
   |  the season's phase, the sub-solar point, the day's phase             |
   |  WHO: DERIVED, on whichever host needs it. NO NEW LANE.               |
   |       The client already derives its light from the brightest         |
   |       luminous row in its window (S7-7); this is that rule read a     |
   |       little further, with the charter's day and tilt.                |
   +----------------------------------------------------------------------+
   |  LIVE — THE WEATHER PROPER       owned state, shipped as a DIFF       |
   |  a field of pressure, moisture and temperature anomalies on the       |
   |  climate grid, stepped by the realm's own shard, carried in its       |
   |  checkpoint, and shipped as A SHORT LIST OF WEATHER SYSTEMS           |
   |  WHO: the realm's own shard. One hop, to that realm's occupants only. |
   |       SL10 clause 7. SL2: a moon's weather is the moon's.             |
   +----------------------------------------------------------------------+
```

**Revision 1 mis-cited V12 to justify shipping the almanac in a row.** V12's recorded answer is *"the parent
authors the spin on the row, the recipe lives in the planet's own frame, only the light changes on the
surface"*, and S7-7 rules *"One directional light from the brightest luminous row in the window; normals
derived on the client"* (`owner_decisions_2026-09-07_voxels.md:484-503`). **The light is DERIVED on the
client, from a placement the window already carries.** V12 adds no row and blesses no almanac lane; it is a
precedent for NOT adding one. Revision 2 therefore ships no almanac.

**Who authors the sub-solar point?** Nobody authors it; it is a derivation from two things that already
cross: the star's row and the body's parent-authored ORIENTATION (§3.5). Revision 1 said the owning realm
computed it in §8.1 and the parent authored it in §11; both were guesses and they contradicted each other.

### 8.2 ★ A season must never invalidate a chunk

This is the load-bearing rule of the whole section, and both refutations agree it is right.

- The SEED decides the shape and the PERMANENT cover. A chunk's cells and its extracted surface are
  functions of the seed alone. They never depend on the clock.
- The ALMANAC and the STORM LIST decide the PAINT and a thin DERIVED cover. The seasonal snow line, the leaf
  colour, the dry grass, the frozen pond surface — the client paints them from the season's phase, the local
  temperature and the shipped systems, with a threshold on height and slope. **No cell record moves.**

If a season changed cell records, every chunk on every planet would rebuild four times a year, and every
saved diff would need a season stamp. The cost is the whole world's terrain, four times over.

### 8.3 ★ Lying snow is DERIVED, not a per-cell diff

Revision 1 said lying snow is a live diff on the surface cells, on the same lane a player's edit uses. **The
arithmetic kills it.** A rung-0 cell is 1 m. A snowfall over 1 km × 1 km is 1 000 000 surface cells; at the
record's 12 bytes that is **12 MB for one snowfall on one planet**, and a 10 km × 10 km valley is 1.2 GB. It
must also be UN-written at the thaw, which is a mechanism that writes and deletes a million authoring
records per season — not a player's edit lane at all — and it collides with a trench a player dug.

**The cure. Snow depth is a DERIVED FIELD, computed identically on both hosts from data that already
crosses.**

```text
   snow_depth(column) = f( the static climate at this column,
                           the almanac's season phase (derived),
                           THE SHIPPED WEATHER SYSTEM LIST (about 1 KB, 8.5),
                           the column's composed height and slope )
   quantised to 1/8 of a cell   (the seating quantum the extractor already ships, S6-7)
   capped at ONE cell           so it can never change a cell record
```

- **No records are written.** The store holds nothing for snow.
- **Both hosts agree**, because both read the same shipped system list and the same static climate. It is
  exactly SL10's "everything the seed does not decide is a one-hop DIFF from the owning realm" — the DIFF is
  the system list, and the snow is a function of it.
- **It sits ON TOP of the composed surface**, so a player's trench survives the snowfall and reappears at
  the thaw.
- **A player who DIGS the snow writes ONE record**, on the ordinary edit lane, for the cells his hand
  touched. That is the same mechanism as any edit, and it is bounded by the same store.
- **The collider.** The server is authoritative for collision, and it computes the same quantised depth. A
  disagreement smaller than the 1/8-cell quantum is impossible; a bigger one is a defect a test catches
  (U17).

*In the game's words.* A storm crosses the pilot's valley. The shard states one row: a system 300 km wide,
centred here, drifting east, intensity 0.7. Both his client and the shard compute 14 cm of snow on the
north-facing slope and 3 cm on the south-facing one, quantised to eighths. He walks through it. He digs a
path to his hull, and those two hundred cells become two hundred records. The thaw comes; the derived snow
goes; his path is still there because it was never snow, it was a record.

### 8.4 The dormant world: 99 % of the galaxy is off

**The problem.** A dormant planet has no shard. Nobody advances its weather. Yet a pilot who arrives must
find a believable sky, and a planet a player left in autumn must be in winter when he returns.

**The answer, in two halves that revision 1 ran together.**
- **The almanac needs no shard, and it needs no carrier.** It is closed form over the universe tick and the
  charter. Any process that holds both can state the season and the sub-solar point for any body at any
  instant, past or future. That is a SERVER fact.
- **A dormant realm ships no charter, and needs to ship none.** A dormant realm sends no self-look
  (`crates/core/src/look.rs:52-54`: *"it rides `BodyStmt::SelfLook`, which only a RUNNING realm may send —
  so a dormant realm is still never drawn"*). A realm that is never drawn needs no sky. The client draws it
  as a correctly-sized point of light from the parent's one radius, which is exactly what the one-radius law
  is for.

**The wake-up, and why it makes no seam (SL8).**

```text
   tick:      ... T-2      T-1       T        T+1      T+2 ...
   almanac:      A(T-2)   A(T-1)    A(T)     A(T+1)   A(T+2)      always defined
   live:            -        -        0       small    bigger     starts at ZERO
   what is drawn:  A       A         A       A+small  A+bigger

   The realm wakes at T. At T the live layer contributes exactly zero, so the
   sky at T is what the almanac alone says, which is what was drawn at T-1.
   The two only begin to diverge afterwards, which is what a live storm IS.
```

**The measurement this owes, corrected.** Revision 1 asked for a byte-identical sky over the waking tick.
That gate must go red for reasons unrelated to waking: between two ticks the star has moved, the camera has
moved and the exposure has adapted. The measurable version: **run the SAME tick twice from an identical
camera, once with the realm dormant and once with it just woken, and require the two images to match within
a stated per-pixel tolerance.** That comparison can fail for the right reason. §12, U10.

**The going-to-sleep side.** When the realm sleeps, the live layer is discarded and the almanac carries on
alone. That IS a change: a storm the player was standing in vanishes. The cure is the standard one — a realm
with an occupant does not sleep (the 2026-09-01 visibility ruling). A realm nobody watches may forget its
storm freely.

### 8.5 ★ THE LIVE WEATHER: the model the owner asked for

The owner said *"we also should simulate the weather"*. Revision 1 answered with a list of nouns. Here is a
model, with state, a rate, a size, a determinism rule, a cost and its own decision rows.

**What is simulated.** A field of ANOMALIES on the climate grid — how far today is from the average — never
an absolute weather. The average is the static climate, which both hosts already hold.

```text
   PER GRID CELL (the same C-per-face grid as 5.1), the LIVE state:
      pressure anomaly       i16      Pa above or below the climate's mean
      moisture anomaly       i16      1/16 mm/yr-equivalent above or below
      temperature anomaly    i16      1/16 K above or below
      wind anomaly (u, v)    2 x i16  m/s in the cell's own tangent frame
                             10 bytes per cell

   Earth-like candidate, C = 256:  393 216 cells x 10 B = 3.93 MB of SHARD state per body.
   Current home body,    C = 128:   98 304 cells x 10 B = 0.98 MB.
```

**How it steps.** One weather step is a semi-Lagrangian advection of the anomalies along the static
prevailing wind plus the wind anomaly, then a fixed relaxation back toward zero (the climate is the
attractor), then a source term: the orographic lift and the sea fetch inject moisture, the latitude gradient
injects pressure, and a fixed pseudo-random forcing seeded by `(body seed, weather step index)` injects the
instability that makes weather happen at all. **Fixed step count, fixed face order, no convergence test** —
the §4 rule, applied to itself this time.

**What is shipped.** *Never the field.* The shard finds the field's extrema — a low is a storm, a high is
clear weather — and states each as ONE ROW:

```text
   A WEATHER SYSTEM ROW  (16 bytes)
   +------------------+------------+--------------+----------+-------------+---------+
   | centre dir 3xi16 | radius u16 | intensity i8 | phase u8 | drift 2 x i8 | kind u8 |
   +------------------+------------+--------------+----------+-------------+---------+
   centre dir : a unit direction on the body, 1/32768 -> about 100 m on an Earth-sized body
   radius     : metres / 256, so up to 16 000 km
   intensity  : -128 (a deep low, a storm) .. +127 (a strong high, clear)
   phase      : where it is in its life: building, mature, decaying
   drift      : the system's own motion in the cell's tangent frame, m/s
   kind       : cyclone, front, high, squall line, dust storm, ... (the art's list)
```

At most **64 systems per body**, which is about 1 KB. Earth carries perhaps 30–60 synoptic systems at once,
so 64 is generous rather than tight.

**What the client does with them.** It draws the cloud field, the rain, the wind in the trees and the snow
depth from the systems plus the static climate. That is a RENDERING of shipped state, not a derivation of
state, so the client-only-renders law holds exactly as it does for a pose.

### 8.6 The live weather's determinism, and why the fence does not bind it

The live weather is **Category C state** in CLAUDE.md's own terms — like rapier's, it is
**checkpoint-carried, never re-simulated cross-host**. It runs on one host only: the realm's own shard. It
is shipped, never derived twice. Therefore:

- **The float fence does not apply to it.** It may use `libm` freely. It is not in `vd-terrain`.
- **No cross-target byte-identity is required or measured**, because no second host computes it.
- **A crash must not restart the storm.** The anomaly field rides the realm's checkpoint, exactly as the
  physics state does. A shard that restarts resumes the same storm. **UNMEASURED and owed (U18):** a kill-9
  test that a woken shard's system list matches the one it stated before the kill.
- **It never enters the recipe.** A chunk's cells are a function of the seed alone (§8.2). The weather
  paints and it derives a capped cover; it never re-shapes.

### 8.7 The live weather's cost

| Item | Per what | Cost | Source |
|---|---|---|---|
| the anomaly field, Earth-like candidate | per body, shard memory | 3.93 MB | computed, §8.5 |
| one weather step | per step | about 4 ms | ESTIMATED: 393 216 cells × about 20 operations |
| the step rate | | one step per 10 s of world time | RECOMMENDED, C-20 |
| **the shard's weather load** | per body | **about 0.04 % of one core** | DERIVED from the two rows above |
| the system list | per body, on change | about 1 KB | computed, §8.5 |
| the re-state rate | | on change, at most once per second | RECOMMENDED, C-20 |
| **the lane's load** | per body per subscribed client | **at most 1 KB/s** | DERIVED |

**UNMEASURED:** every row marked ESTIMATED (U19). For scale, the interest lane already ships per-cube
snapshot bodies at 20 Hz; 1 KB/s per body is small beside it.

### 8.8 The SL6 ask for the weather lane

SL6 covers a new wire arm as well as new data, and revision 1 asked for neither. Here is the statement.

- **What data:** at most 64 weather-system rows of 16 bytes, about 1 KB, per body.
- **From which realm to which:** from a realm to its OWN occupants and their clients. One hop. Not realm to
  realm. A moon's weather is the moon's (SL2).
- **How often:** on change, at most once per second. Never per tick.
- **Which arm (HR1):** a new `InterShardFlow` arm is NOT needed for the occupant leg, because a realm
  already states things about itself to its own subscribed clients through the window lane's bag; the system
  list is a new TAG on that bag, in the skip-unknown shape the VU streaming contract promises. **A
  realm-to-realm weather arm is REFUSED** and no arm is asked for.
- **Why the receiver cannot compute it:** it is live state. SL10 clause 7 forbids deriving it, and the
  client-only-renders law forbids simulating it.
- **What doing without costs:** the sky is the almanac alone — a correct sun, a correct season, and no
  weather. The owner asked for weather.

### 8.9 What crosses a boundary, and what never does

- The charter: a realm's statement about ITSELF, to its own clients (§3.6). Not realm to realm. It is
  DERIVED from the star's luminosity and the body's orbit, which are the parent's facts — but the realm
  derives them from the seed, so nothing crosses to make it.
- The weather system list: from a realm to its own occupants. One hop.
- **Nothing weather-shaped crosses realm to realm.** A ship in a planet's atmosphere is inside the planet's
  realm at that moment, so it receives the planet's weather as an occupant, not as a crossing. §14, Q4
  raises the one case that is not obvious.

---

## 9. Different planets, different biome sets

Every row below comes out of the SAME model, with no branch on a body kind (HR4).

| World | The charter says | The biomes that exist | What the pilot sees |
|---|---|---|---|
| **Earth-like** (seed 2298's own: 1.023 R⊕, g 10.20, temperate, air, G star) | insolation near 1, air about 1 bar, condensable water, tilt about 20°, day about a day | the full chart: rainforest, forest, taiga, grassland, savanna, desert, tundra, ice, wetland, beach | the owner's picture |
| **A cold world** | insolation low, air thin, condensable water or ammonia | ice, sea ice, tundra, taiga only | a white world with dark conifer bands near the equator |
| **A hot dry world** | insolation high, air present, no sea | desert, shrubland, savanna, regolith | red rock, dune fields, a dry wind |
| **An ocean world** | the ocean share high (§5.8) | ocean, sea ice, beach, a few island biomes | archipelagos; the vista is water and cloud |
| **A Titan-like world** | condensable 3 (methane), 1.5 bar of nitrogen at 94 K | methane sea, methane sea ice, a methane-carved shore | orange haze, methane rain, rivers that are not water |
| **An airless rock** | `has_air = 0`, condensable 0 | regolith, ice in permanent shadow | no wind, no rain, no soil, no plant, no haze, and a hard black sky |
| **A tidally locked world** | `day_s = year_s` | a burning day side, a frozen night side, and a habitable TERMINATOR RING between them | the star stands still on the horizon forever; the ring is the only green |
| **A high-tilt world** (`ε > 54°`) | `s2` positive | ice at the EQUATOR, forest at the poles | the bands are inside out, and the seasons are violent |
| **A high-eccentricity world** | `e = 0.12`, the world's CAP, so **1.62** times the light at periapsis | seasonal deserts that green over | the biome map breathes across the year — a real effect, and about half the swing revision 1 claimed |
| **A fast-spinning world** | `day_s` about 8 h | six banded desert and rain belts | thin alternating stripes from orbit |

**The tidally locked world deserves a note.** It falls straight out of the locking test in §3.3 with no
special code. It is also the most striking place the game can offer: a permanent sunset, a wind that always
blows the same way, and a ring of habitable ground. The model produces it for free.

---

## 10. ★ The home planet is probably the wrong body — what is measured, and what is not

**The measured facts.**

| Fact | Value | Where |
|---|---|---|
| Seed 2298's Earth-like planet | 1.087 M⊕, 1.023 R⊕ = **6 515.5 km**, ρ 5 601 kg/m³, g 10.20 m/s², Rocky, temperate, **G star**, RETAINS ITS ATMOSPHERE | `crates/physics/src/worldgen/census.rs:57-63`, MEASURED and pinned |
| The voxel foundation's home planet | look radius **3 350 759 m**, 12 rungs, 14 octaves, relief 14 304 m | `crates/terrain/src/home.rs:19-42`; `slice_05_generator.md:367-369`, MEASURED |
| Its sea | 5 297 m under the ladder radius; **about one column in a hundred is water** | `slice_05_generator.md:388-392`, MEASURED |
| How the home planet is chosen | the first planet row of the home system **whose look radius the ladder ACCEPTS** | `crates/bins/src/lib.rs:987-1003` |
| The insolation ladder | rung 2 is 0.748 S⊕ for every star at every seed; the ladder ratio is 1.7, so insolation falls as `1/2.89ⁿ` | `census.rs:96-98` (MEASURED); `worldgen/config.rs:58-59` |
| The retention law reads XUV, and for a G star XUV equals insolation | `xuv_rel_of(1.0, G) == 1.0` | `taxonomy.rs:791-802,1511-1518` |

**What is NOT measured, and revision 1 stated as fact.** `home_body` takes the first ACCEPTED planet, not
simply the innermost. **The picked body's orbital rung is UNMEASURED.** Every number below depends on it.

**The estimate, with the rung left open.** ESTIMATED from the shipped laws (`ROCK_MR_SEGMENTS`,
`escape_velocity_mps`, `xuv_rel_of`, `cosmic_shoreline_retains`):

- a rocky body of 3 350 759 m is about **0.093 M⊕** (Chen–Kipping inverse; 0.10 M⊕ by an independent
  recomputation, close to Mars);
- its surface gravity is about **3.30 m/s²** and its escape velocity about **4 700 m/s**;
- the shoreline law keeps a secondary atmosphere only where the XUV irradiation is below
  `6.0229 × (4 700 / 11 180)⁴ =` **0.187 S⊕**;
- the home star is class **G**, so its XUV irradiation EQUALS its insolation, and the comparison is legal;
- **if the body is at rung 0**, its insolation is **6.25 S⊕**, which is 33 times the bound;
- **if the body is at rung 1**, its insolation is **2.16 S⊕**, which is 11.6 times the bound.

```text
   THE CONCLUSION IS ROBUST TO THE RUNG; THE NUMBERS ARE NOT.
   ---------------------------------------------------------
   retention bound          0.187 S(+)
   rung 0 insolation        6.25   S(+)   ->  33.4 x over
   rung 1 insolation        2.16   S(+)   ->  11.6 x over
   rung 2 insolation        0.748  S(+)   ->   4.0 x over   (and rung 2 IS the earth-like body,
                                                             which is 6 515 km, not 3 351 km)
   At every rung a 3 351 km body in this system is over the bound. The body is airless.
   BUT this is an ESTIMATE built on a Chen-Kipping inverse, and U1 must MEASURE it.
```

**What that costs the vista, if the estimate holds.** No atmosphere means: no wind, so no rain shadow; no
rain, so no rivers and no soil; no plants, so no forest and no canopy fold; no haze, so no aerial
perspective; no cloud. The owner's picture is a picture of an ATMOSPHERE acting on a landscape for a million
years. On this body there is nothing to act.

**It also explains the owner's own complaint.** *"No orienters and no details at all… the surface will not
be interesting enough."* A bare rock with fourteen octaves of noise and no climate has, correctly, no
orienters: no treeline, no snow line, no river, no coast, no forest edge. Those are the things an eye reads
distance from. **And the 1 % sea, which IS measured, would produce the same complaint on a body that DID
have air** (§5.8).

**A general rule for a rocky world, ESTIMATED from the same laws.** At Earth-like insolation a rocky body
needs a radius of about **4 600 km** (0.30 M⊕) before the shoreline law lets it keep air. Any smaller planet
in the world is airless wherever it is warm.

**The fix, and what it actually costs.** Revision 1 called it "the two-line fix". It is not.

1. `home_body` picks the body that satisfies the existing `earth_like` predicate (`census.rs:94-107`), not
   the first accepted row. The predicate already exists and is already tested.
2. `crates/terrain/src/home.rs` states that body's seed and look-radius bits, and
   `crates/bins/tests/home_body_pin.rs` proves the two agree.
3. **What re-measures:** the golden table (2 592 digests: 72 columns × 3 chunks × 12 rungs — and the new
   body has **13 rungs**, computed here, so the table's SHAPE changes); the boot self-check (13.5 ms
   MEASURED); the world tag's measured half (`tag.rs:41-50`); the whole cost table (a 13th rung and 15
   octaves); the climate grid at **C = 256** instead of 128, which is four times the samples (§5.1).
4. **What it does not cost:** it is not a variant and not a knob. It is ONE world, and this is the choice of
   which body in it we start on. The seed itself was chosen by a search for exactly this property.

**Its timing.** The golden table and the world tag change, so it must land BEFORE the first saved world,
which the extractor document already names as the freeze point (`slice_06_extractor.md:298`). **This is
time-critical, and U1 must run first.**

---

## 11. Law gates

| Law | How this proposal passes |
|---|---|
| **SL10** — a function of seed and address only, byte-identical on both hosts | The climate reads the seed, the charter (whole numbers), the direction and the height. The clock never enters. Every operation is add, subtract, multiply, divide, square root, floor and compare — the circulation's arcsine and sines now sit OUTSIDE the fence as charter rows 22 and 23. The charter is quantised, and the drift is measured ACROSS TARGETS (U5), not by a perturbation that cannot fail. |
| **SL10, live state is a one-hop diff** | The weather system list is the diff. It never enters the recipe. Lying snow is a capped function OF the diff, on both hosts, quantised to the shipped 1/8-cell seating quantum (§8.3). |
| **SL1 — a realm is told where it is; it never decides where it is** | On the server nothing is told at all: the shard derives its charter from the seed, as it already derives its mass (`worldgen/body.rs:47-51`). On the client, four orbit rows tell it nothing it does not already hold, because S7-7 already has it place the star and the body from rows in its own window. **The SHAPE is still an owner decision: C-15.** No realm asserts a placement about itself. |
| **SL3 — a parent's per-child message carries a PLACEMENT and nothing else** | **The charter does NOT ride the parent's bag.** The one-radius law (`crates/core/src/look.rs:41-47`) forbids a second number there, and revision 2 obeys it: the charter rides the realm's OWN self-look, beside `TAG_SURFACE`, which ruling V6 row R-8 already approved. |
| **HR1 — sealed shards; no new wire arm without an ask** | No new `InterShardFlow` arm is asked for. The charter is a new TAG on the self-look bag; the weather system list is a new TAG on the same bag. Both are skip-unknown, in the VU streaming contract's shape. §8.8 states the ask. |
| **The seed ruling (a treasure map)** | The climate decides SHAPE and COVER — where it rains, where it snows, where trees grow. **The argument, not just the conclusion:** a published climate map names the timber, the fresh water, the arable ground and, on a locked world, the habitable terminator ring. Those are ROOM, and the 2026-08-27 ruling settles room by name — *"an expensive gate buys ROOM, not treasure … a new galaxy is worth reaching because nobody has taken it yet, which is live state"*. Who has taken the good ground is live state. Every ORE stays live state (`strata.rs:1-5`). The one open risk is a soil that NAMES an ore, and C-19 asks the owner about it rather than asserting an answer. |
| **SL5 — one world, no variant** | One climate model on one world. No preset, no scale knob, no test-only body. §10 changes WHICH body is home; it does not make a second one. **A test that constructs a charter is not a variant**: a charter is per-body DATA of the one world, and the crate already forces inputs the same way (`height.rs:112-140`). |
| **No magic numbers** | Every parameter is a charter fact (mass, radius, orbit, luminosity, spin, tilt, atmosphere, condensable), or a seed draw, or a literature-anchored constant with a named source and an anchor test. **Stated honestly and NOT claimed as derivations:** `SAMPLES_PER_WAVELENGTH = 8` (a named constant with a reason, §5.1), `T_TOL_K = 0.5` (a named perception constant, §5.10), the cell clamp at 6 (a stated art decision), the four fixed sweep rounds and the two albedo rounds (§5.1, §5.6), the 64-system cap (§8.5) and the chart's cut lines. `long_wave_m` is NOT a derivation on a planet, because the 400 km cap always binds; §5.1 says so instead of claiming otherwise. |
| **The 8 ms per-chunk budget** | The climate costs an ESTIMATED 0.44 ms at rung 0 and 0.69 ms at the top rung, against **1.9 ms of MEASURED headroom on a cave-dense chunk** (the worst case, which is what a budget is sized on) and 4.7 ms on a surface chunk. The per-column upwind march is REFUSED by measurement (2.43 ms, which fits the surface chunk and NOT the cave-dense one). Every number is derived from the MEASURED 13.19 ns per octave. |
| **The ladder — rung L cheaper than rung 0, and the coarse answer of it** | The classifier reads the CLIMATE RUNG's height at every rung, so the biome, the snow line and the sea-ice line are IDENTICAL at every rung by construction — the gate is `assert_eq` over the golden columns, not a bound (§5.10). The cost of that at a coarse rung is at most five extra octaves, so every rung's column pass still costs at most rung 0's; the MEASURED fall from 710 to 266 µs becomes a fall to about 430 µs and must be re-measured (U16). |
| **The record (12 bytes)** | Unchanged. A biome is a column property both hosts derive. A vegetation anchor is derived and folded into a new ANCHOR DIGEST (§7.2); it becomes a record only when a player fells it, with the provenance value the format already reserves. Snow writes no record (§8.3). |
| **The edit pyramid as the authoring override** | A player who terraforms changes the height, so the temperature under the new height follows automatically through the lapse rate, and the MID and LOCAL shadow terms follow too, because both read the composed height. The GRID shadow does not, because the grid is built from the seed. §14, Q6 states that limit honestly. |
| **The collider on the same shape** | The climate changes materials and cover. Two things change the CELL — the snow line and sea ice — and both are computed on both hosts from the same function at the same climate rung. Derived lying snow is capped at one cell and quantised to the extractor's own 1/8-cell seating quantum, so it can never move a cell record (§8.3). |
| **SL8 seamless** | Five seams are designed against, each with an owed measurement: the DETAIL-LEVEL seam (the climate rung makes the class identical at every rung, U16); the cube-face seam (the march is taken in the body's 3-D frame through `face_of`, so no rotation rule is needed, U11); the biome edge (continuous inputs plus a dithered blend, U12); the wake-up (the live layer starts at zero, measured as the same tick twice within a per-pixel tolerance, U10); and the ARRIVAL HITCH while a client builds a body's grid (U14). |
| **HR4 — features once, run anywhere** | One classifier, one climate function, no branch on a body kind. A planet, a moon and a hull that holds terrain call the same code with different charters. |
| **HR5 — 100 % coverage in the generator** | Every law is a straight-line expression over `Gf`. The branching sits in monomorphic helpers: the chart is a table walk, the overrides are an ordered chain of comparisons. **Coverage is reached by constructing charters in tests, not by finding a body that shows all nineteen biomes** — the pattern `height.rs:112-140` already uses. The fitted polynomials are branchless. Every grid pass has a fixed round count, so no loop's trip count is a branch on a float. |
| **SL9 — an unbounded child count** | Nothing here walks children. The charter rides the realm's OWN self-look, which only a running realm sends. The climate grid is per BODY, and it is built only for a body somebody is close enough to draw ground on. |
| **V4 — vegetation are art assets** | The generator emits anchors: a cell, a kind and a parameter word. It draws no trunk and no leaf. The client blends the art asset. The far fold is a canopy surface, not a card, so impostors stay refused. |

---

## 12. What is UNMEASURED, and the bench that measures each

| # | The claim | How to measure it |
|---|---|---|
| **U1** | **The home planet is airless and hot (§10), and its orbital rung.** | Extend `crates/bins/examples/terrain_cost.rs`, or add an example, to print for EVERY planet of the home system: the orbital rung, the mass, the radius, the surface gravity, the escape velocity, the **bolometric insolation AND `xuv_rel_of`**, the star class, `T_eq`, the shoreline verdict, the atmosphere, the drawn sea offset and the **sea column fraction**, the isostatic ceiling `88 300/g` against the drawn relief, and the biome histogram the classifier returns. One run. **Do this first: it decides §10, §5.8 and §5.9.** |
| U2 | The climate costs 0.44 ms per rung-0 chunk and 0.69 ms at the top rung. | Add a climate pass to the same bench; print the delta on a surface chunk and on a cave-dense chunk, at rung 0 and at the top rung. |
| U3 | The grid costs about 100 ms and 6.29 MB per body on the Earth-like candidate. | Build the grid in the bench; print the wall time per face and the allocation. |
| U4 | A per-column upwind march costs 2.43 ms per chunk. | The same bench with the march switched on. It should be green on a surface chunk and RED on a cave-dense one; that is the point. |
| **U5** | **The charter is byte-identical across targets.** | Compute the charter for a census of bodies on x86-64 and on aarch64 and compare the INTEGERS, in the shape `D-TERRAIN-1` already uses for chunks. **A perturbation test is refused: it cannot fail.** Publish how many bodies land within one quantum of a boundary. |
| U6 | The climate is byte-identical across x86-64 and aarch64. | The existing no-drift gate widens to fold the climate grid's digest and a set of column climates. |
| U7 | The greenhouse fit reproduces Mars, Earth, Titan and Venus; the transport fit reproduces the Moon, Earth and Venus. | Unit tests with the anchor tables, in the shape of the shoreline golden. **Publish the residual for each body.** |
| **U8** | **The circulation fit.** | Re-fit `dH` over a wider anchor set and publish the residual per body. §5.3 states that at `dH = 0.388` **Mars MISSES by one cell**. If no single `dH` holds Earth and Mars, publish that instead of a ✓. |
| U9 | The saturation-vapour polynomial's error bound. | Walk 1 401 steps of 0.1 K against a recorded exact table; assert the relative error. |
| U10 | To wake a realm changes no pixel of the sky. | **Run the SAME tick twice from an identical camera**, once dormant and once just woken; require a per-pixel match within a stated tolerance. |
| U11 | No biome seam is visible at a cube-face boundary. | A picture across a face seam, plus a numeric test that a shared direction gives the same climate from both faces of the march. |
| U12 | The biome edges do not look like contour lines, and the sea-ice edge does not look drawn. | Pictures from 2 km up, over a coast and over a mountain front, with and without the two albedo rounds of §5.6. This is a judgement, and it is the owner's. |
| **U13** | **The four fixed sweeps give the right distance to the sea on a closed cube.** | Compare the fixed-round chamfer against an exact bucketed Dijkstra over the same neighbour graph, on the home planet's grid; publish the worst-cell residual. |
| U14 | The arrival hitch when a client builds a body's grid. | Time the first-chunk-drawable latency on approach, per face and for all six; state it against the SL8 seam list. |
| **U15** | **`K_mm_per_year` gives an Earth-like world 2 000 mm at the ITCZ and 100 mm under the subtropical high.** | A two-point anchor test on an Earth-like charter, with published residuals. **Without this the biome chart is read in unknown units.** |
| U16 | The climate rung makes the biome identical at every rung, and what it costs. | `assert_eq` on the biome, the snow line and the sea-ice line over the golden columns at every rung; and re-measure the column pass at every rung with the climate rung's extra octaves. |
| U17 | Derived lying snow agrees between the server's collider and the client's picture. | Compute the quantised depth on both hosts over a golden column set under a fixed system list; assert equality. |
| U18 | A shard that is killed resumes the same storm. | A kill-9 test: the woken shard's system list matches the one it stated before the kill. |
| U19 | The weather step costs about 4 ms and the lane about 1 KB/s. | A bench that steps the anomaly field on the Earth-like candidate's grid and counts the stated bytes. |
| U20 | The vegetation anchors agree between the two hosts. | The ANCHOR DIGEST joins the golden table and the no-drift legs, exactly as the mesh digest did. |

**★ A binding cross-domain requirement, handed to domain 03 (erosion and rivers).** §7.3 shows that a desert
and a grassland have identical relief in this model, which a geologist rejects on sight. This document's
precipitation field, temperature field and freeze-thaw count are the INPUTS the erosion domain's
stream-power and creep laws need; the climate must feed them, and the coupling must be stated in that
document. Nothing in this domain can produce an arid landform on its own.

---

## 13. What you decide

| # | The question | The recommendation | The cost if you say no |
|---|---|---|---|
| **C-1** | **The body charter.** A realm derives twenty-four whole numbers about ITSELF and states them in its own self-look, beside the surface tag that already rides there. (An SL6 ask, for the CLIENT leg only.) | **YES.** The server needs no message at all — its shard already derives the forest. The client cannot, by an isolation row. About 64 bytes, on change. | The climate keeps today's two meaningless noises. Biomes cannot depend on position, spin, trajectory, size or gravity. |
| **C-2** | **The two missing draws: obliquity and spin** — appended to the body's own draw stream. **And their cost:** the `POLE_AXIS` pin is rewritten, the parent authors a tilted spinning orientation, and every picture gate that reads "the planet does not spin" is fixed. | **YES**, with the cost accepted openly. | No seasons, no Coriolis, no circulation cells, no rain-shadow direction, no tidally locked world. |
| **C-3** | **Surface pressure.** One seed draw, log-uniform over a band the retention verdict bounds (10 mbar to 100 bar), closing D-TAX-1. | **YES**, and state it plainly as a DRAW, not a derivation. | No greenhouse, no heat transport, no haze depth. The air is a bit with no size. |
| **C-4** | **The condensable volatile** — DERIVED from the surface temperature and pressure over a five-row physical table, not drawn. | **YES.** Without it the classifier cannot decide sea ice, a snow line or a frozen world on any body but an Earth-like one. | The model works on water worlds only, and a methane moon gets water snow. |
| **C-5** | **Precipitation at three scales** — a grid term, a mid term (one coarse evaluation 5–15 km upwind) and a local term (the slope along the wind, from the column pass's own halo). | **YES.** Without the two finer terms the whole 50 km vista sits in one climate cell and the biome varies with height alone, so the picture is contour bands. It costs 0.35 ms per chunk. | The reference picture's green-valley-beside-dry-cliff contrast is out of reach. |
| **C-6** | **The CLIMATE RUNG** — the classifier reads the height at one derived rung, so the biome is identical at every detail level. | **YES.** It costs at most 0.25 ms at a coarse rung and it turns an SL8 defect that exists TODAY into an `assert_eq`. | The far ridge's snow cap moves as the pilot flies in — the arrival-pop seam, on the picture's most visible feature. |
| **C-7** | **Nineteen biomes in five bits**, up from four in two, with five strata appended. | **YES.** The record does not change; a biome is derived per column. | The vista has no taiga, no savanna, no wetland, no beach and no ice. |
| **C-8** | **The soil law and sea ice.** The topsoil follows the climate; the sea freezes below the CONDENSABLE's freezing point; the permafrost active layer's depth is derived; two fixed albedo rounds keep the ice edge from reading as a drawn circle. | **YES.** One table and three comparisons, plus one grid pass. | Polar caps must be placed by hand, which is a magic number the size of a hemisphere. |
| **C-9** | **The vegetation skeleton and the ANCHOR DIGEST.** A hashed scatter, derived not stored, folded into a third digest beside the cell and mesh digests, becoming a record only when a player fells it. | **YES**, per ruling V4. The digest is what stops two hosts disagreeing about where a player can stand. | No forest mass, no canopy fold, and a tree the client draws where the server has none. |
| **C-10** | **★ A season never invalidates a chunk**, and **lying snow is DERIVED from the shipped system list, not a per-cell diff.** | **YES, as a stated law.** The diff form costs 12 MB per snowfall over one square kilometre. | Every chunk rebuilds four times a year, or the diff store carries a season's snow. |
| **C-11** | **The three-layer split** — a static climate, an almanac DERIVED with no new lane, and live weather as a one-hop diff. | **YES.** S7-7 is the precedent for deriving the light, not for shipping it. | Either the client derives live state, which is forbidden, or a dormant planet has no sky. |
| **C-12** | **The dormant world runs on the almanac**, the live layer starts at zero on waking, and a dormant realm ships no charter because it is never drawn. | **YES.** It costs nothing and it is the only no-pop answer. | A planet pops its whole sky when its shard wakes. |
| **C-13** | **★ Re-pick the home planet** by the existing `earth_like` predicate, before the first saved world — **after U1 measures it.** | **YES, and U1 FIRST.** The estimate is strong but it IS an estimate; the re-pick costs a 13-rung golden table, a new world tag, a new cost table and a C = 256 grid. | Every picture and every judgement of "is the surface interesting" is taken on the wrong world. |
| **C-14** | **The obliquity prior**: nine bodies in ten at a tilt under about 78°, the tenth a Uranus-like curiosity. | **YES**, as an art decision stated as one. | An isotropic prior makes a third of all worlds exotic, which makes exotic ordinary. |
| **C-15** | **★ May a body's charter name ORBIT facts** — its insolation, its eccentricity, its argument of periapsis and its year? On the server nothing is told; on the client the rows tell it nothing it does not hold. **But it is a decision, and revision 1 denied one existed.** | **YES**, with the reasoning written into the code beside the fields. | The greenhouse, the season and the eccentric-orbit world all go, because the climate cannot know how much light the body gets. |
| **C-16** | **The client's grid budget**: built PER FACE so the first chunk waits about 17 ms and not 100 ms, and dropped when the realm leaves the window. | **YES.** Otherwise a pilot pays 100 ms and 6.29 MB per body before the ground appears. | An arrival hitch at the most visible moment in the game. |
| **C-17** | **★ The sea becomes a HYPSOMETRIC law** — the seed draws the world's OCEAN SHARE and the sea radius is solved to match it — instead of an offset that MEASURED gives the home planet 1 % ocean. | **YES.** Two of the four precipitation terms read the sea; a 1 %-ocean world returns desert everywhere. It costs one grid pass and a version bump. | Every seed is a coin flip between an ocean world and a desert world, and the reference picture needs the coin to land right. |
| **C-18** | **The relief's cap becomes isostatic** — `min(the present cap, ISOSTATIC_COEFF/g)`, anchored on Earth and Mars. **This is domain 02's law**, handed over with the number. | **YES**, and it belongs in that document. At g 10.2 the ceiling is 8 660 m against a draw of 6 000–18 000 m. | The Earth-like candidate can draw an 18 km mountain that no crust can hold, and a snow line that erases the forest belt. |
| **C-19** | **Do the new soils break S5-6 (no indicator material)?** Laterite is a bauxite proxy; peat is fuel. | The proposed answer: a soil names the CLIMATE, not the deposit, and the ore inside it stays live state. **If you disagree, name the soils by LOOK** (`RedSoil`, `DarkSoil`, `SourSoil`). | Either a publicly computable ore map, or soils that read as art rather than geology. |
| **C-20** | **The live weather's rates**: one weather step per 10 s of world time; the system list re-stated on change, at most once per second; at most 64 systems per body. | **YES** as a starting point, to be re-measured. | Either a weather nobody can see moving, or a lane that ships a field. |
| **C-21** | **The fence technique**: per-body constants in the charter — **including the circulation's cell count and cell edges** — fitted polynomials and 256-entry tables on the per-column path; a Newton iteration refused; **every grid pass a fixed round count in a fixed order.** | **YES.** | Either a transcendental enters the fence, which breaks SL10, or an iteration's trip count becomes a branch on a float, which breaks HR5 and drift. |

---

## 14. Open questions

**Q1 — What is a year, in universe ticks?** The almanac needs the orbital period in the universe's own
clock. `year_s` comes from the elements, but the mapping from a real second to a universe tick, and whether
a player is expected to LIVE through a season, is a game-design decision nobody has made. A 1 000-hour year
makes a season a thing a player reads about. A 20-hour year makes it a thing he plays.

**Q2 — Does the star's colour reach the ground?** An M-dwarf world is lit red. Real photosynthesis under a
red star would look different. Is the plant colour a function of the star class, or is it always green? That
is art, not physics, and the owner should say.

**Q3 — One grid or two?** This document wants a coarse grid for the rain shadow and the distance to the sea.
The hydrology domain wants one for flow accumulation and base level. **They must be the same grid.** The two
documents must agree before either slice starts.

**Q4 — Does a ship in the air feel the planet's wind?** A ship inside a planet's atmosphere is an occupant
of that realm, so the wind reaches it without a crossing. But a ship in LOW ORBIT sits at the edge, and the
atmosphere's drag is already a physics fact (`Atmosphere.reference_density_kgm3`). Exactly where the wind
stops acting is a physics question this document does not answer.

**Q5 — What is a hull's charter?** Ruling V4 says a hull or a station MAY hold terrain. A hull has no seed,
no mass drawn from a star and no orbit. It has an interior. Does a hull's terrain get a charter its BUILDER
states — a garden deck at 20 °C and 60 % humidity — and is that then a player-authored climate? It is a good
idea, and it is out of scope here.

**Q6 — Does a terraformed ridge cast a new rain shadow?** The grid is built from the seed's height, not from
the composed height. A player who raises a mountain gets no new GRID shadow. **He does get a new LOCAL and
MID shadow**, because those two terms read the composed height at the column, which is a real improvement
over revision 1. The honest answer is "the ridge shades its own two faces, and it does not dry a whole
region", and the owner should know it before a player notices.

**Q7 — How many biomes should one planet actually show?** Nineteen exist. A single Earth-like world shows
perhaps twelve. Should the classifier be tuned so that a world shows FEWER, larger biomes — so that a
continent reads as one place — rather than a patchwork? That is a judgement on pictures.

**Q8 — Should the relief law be isostatic?** §5.9 shows the shipped law runs the wrong way against gravity,
and that the Earth-like candidate can draw an unphysical mountain. The law belongs to domain 02. This
document states the number and hands it over.

**Q9 — Should a body draw VARIED BEDDING?** Today one sediment kind and one thickness cover a whole planet,
so no hoodoo, no layered mesa and no differential erosion can exist anywhere. Varied bedding — a few
alternating hard and soft layers with drawn thicknesses — is what turns the owner's "rock pillars" from a
wish into a consequence. It is a recipe change (domain 02) with an erosion consequence (domain 03), and it
is out of scope here, but nothing in the reference picture's cliffs happens without it.

---

## 15. Revision log

**Revision 1 (2026-09-08).** First issue, against the code at commit `41b0ba0`.

**Revision 2 (2026-09-08).** Answers refutation A (3 blockers, 8 defects, 14 weaknesses, 3 notes) and
refutation B (3 blockers, 14 defects, 10 weaknesses) in full; the table is §16. The eight changes that moved
the design, not just the prose:

1. **The cost base is now MEASURED.** 13.19 ns per octave from `slice_05_generator.md:369-379`, not 27.9 ns
   from a number that measured a different pass. Every ESTIMATED cost was recomputed. The per-column march
   is still refused, but on the cave-dense chunk's 1.9 ms headroom, not on a figure twice too big.
2. **The charter moved off the parent's row.** The one-radius law and SL3 close that lane by name. The
   server needs no message at all; only the client leg is an ask, and it rides the realm's own self-look.
3. **The CLIMATE RUNG replaces a bound that could not bind.** The biome is now identical at every rung by
   construction, and the gate is `assert_eq`.
4. **Precipitation gained a MID and a LOCAL term.** A 41 km grid alone gives contour bands, not the owner's
   picture.
5. **The live weather became a model.** State, rate, size, determinism, checkpoint, cost, an SL6 statement,
   four decision rows and four measurements — where revision 1 had a list of nouns.
6. **Lying snow became derived.** The diff form costs 12 MB per snowfall over one square kilometre.
7. **Three holes were closed:** the condensable volatile (charter row 10, derived), the precipitation UNIT
   constant, and the circulation's arcsine, which revision 1 wrongly placed inside the fence.
8. **Three findings were added that revision 1 did not see:** the MEASURED 1 % sea on the home planet
   (§5.8), isostasy against the relief law (§5.9), and the absence of any lithological contrast, which means
   this model cannot make an arid landform without domain 03 (§7.3).

Every arithmetic claim in this revision was recomputed from the cited file. Where a refuter's number and
mine disagreed, the code decided; where the code was silent, the claim is marked UNMEASURED and §12 names
the bench.

---

## 16. Refutation answers

Every finding of refutation A and refutation B, with its answer. **FIXED** means the section was rewritten.
**KEPT** means the refuter is wrong and the evidence is named. **OWED** means the answer belongs to another
owner or another domain.

### Refutation A — the laws and the code

| # | Finding | Severity | Answer |
|---|---|---|---|
| A-B1 | The per-octave cost is 13.19 ns (`slice_05_generator.md:369-379`), not 27.9 ns; the march refusal rests on a wrong number. | blocker | **FIXED.** The refuter is right and the source is exact: 710 µs / (3 844 × 14) = 13.19 ns. §1 now carries the measured row, and §5.4, §5.7 and §5.10 are recomputed from it. The march is 2.43 ms, not 5.1 ms — and it is still REFUSED, because the budget is sized on the cave-dense chunk's **1.9 ms** headroom, not the surface chunk's 4.7 ms. The grid's justification is re-argued on the PICTURE (§5.4's three scales), not on a budget that does not bind. |
| A-B2 | The charter rides the parent's row, which the one-radius law and SL3 close by name; the server needs no lane at all. | blocker | **FIXED.** Verified: `crates/core/src/look.rs:41-47` forbids a second number on the parent's bag, `TAG_SURFACE` is the realm's OWN statement (`look.rs:50-56`), and `worldgen/body.rs:47-51` says the shard derives the forest locally. §3.6 deletes the parent→child leg entirely; the charter rides the realm's own self-look, and only the client leg is an ask. §11 gains an SL3 row and an HR1 row. |
| A-B3 | The home planet's sea is 1 % of its surface, and two of four precipitation terms read the sea. | blocker | **FIXED.** Verified in the slice-5 record and ruling V9. New §5.8 states the measurement, shows the model returns desert everywhere on such a body, refuses a per-body re-draw as an SL5 variant, and recommends a HYPSOMETRIC ocean-share law (C-17). U1 must print the sea fraction and the biome histogram. |
| A-D1 | The circulation needs an arcsine and per-edge sines, which §4 refuses. | defect | **FIXED.** The refuter is right. §3.2 rows 22 and 23 move the cell count and the cell edges (in sine of latitude) into the charter, computed by the producer outside the fence. §5.3 says so explicitly and withdraws revision 1's "the square root is inside the fence" claim. |
| A-D2 | `WorldIdentity` is a per-WORLD pair and cannot catch a per-body charter. | defect | **FIXED.** Verified at `tag.rs:19-50` and `digest.rs:1-8`. §3.7 folds the charter into the per-body chunk digest's key material and refuses a mismatch through the per-realm surface refusal S7-3 already built. |
| A-D3 | The ladder gate is claimed for a discrete class with a continuous bound. | defect | **FIXED**, and more strongly than the refuter asked. §5.10 introduces the derived CLIMATE RUNG, so the class is IDENTICAL at every rung by construction and the gate is `assert_eq`, not a share-of-columns ceiling. The blend width is kept as a paint rule on top. |
| A-D4 | `e = 0.2` cannot occur: `ecc_cap = 0.12`. | defect | **FIXED.** Verified at `config.rs:61-71`. §5.5 and §9 are re-based on 0.12, giving a periapsis ratio of 1.62 rather than 2.25. |
| A-D5 | The cube-sphere march is hand-waved; `site_of` is the wrong primitive; the chamfer does not converge; the sweep order is unpinned. | defect | **FIXED, and one half KEPT with evidence.** The refuter is right that `site_of` is the wrong primitive and that a two-pass chamfer fails on a closed surface. The refuter is wrong that no walk primitive exists: `bend.rs:224-262` ships `face_of`, `face_coords` and `direction`, and `ladder.rs:157` ships `index_of`. §5.1 names all five and shows that a step taken in the body's 3-D frame needs NO seam rotation and NO corner case, which also answers the rotation half. The chamfer becomes four fixed double sweeps in a fixed face order with no early exit, and U13 measures the residual against an exact Dijkstra. |
| A-D6 | Four different biome counts; `WOODLAND` in no list; a Rust enum has no bit width. | defect | **FIXED.** §6.2 states ONE list: twelve chart biomes plus seven overrides = nineteen, five bits, thirteen spare, and `Woodland` is in the list. C-7 now approves a SET SIZE and its version bump, not a wire field. |
| A-D7 | §8.1 and §11 disagree about who authors the sub-solar point. | defect | **FIXED.** §8.1 states that NOBODY authors it: it is derived from the star's row and the body's parent-authored orientation, which is exactly what S7-7 already does for the light. Both contradicting sentences are gone. |
| A-D8 | The charter's determinism is probabilistic; no authority is named; the test population is short. | defect | **FIXED.** §3.2 replaces the perturbation argument with a cross-target measurement (U5), §3.7 names the OWNING realm's derivation as authoritative and the client's as a reading, and U5 asks how many bodies land within a quantum of a boundary rather than asserting none do. |
| A-W1 | "The resolution is not a magic number" is hollow: `long_wave_m` clamps at 400 km on every planet. | weakness | **FIXED.** Verified at `body.rs:24,150-153`. §5.1 now states plainly that on any body over 1 600 km the wavelength IS the constant, that the grid resolution is a function of the radius alone, and that `SAMPLES_PER_WAVELENGTH = 8` is a NAMED CONSTANT with a reason — not a derivation. §11's no-magic-numbers row says the same. |
| A-W2 | A 41 km cell cannot make the picture's contrast; C-4 drops the local term; two questions share one number. | weakness | **FIXED.** §5.4 carries three terms (grid, mid, local) and costs each. The decision rows are renumbered so no two questions share a label. |
| A-W3 | Isostasy is missing, and the measured relief makes the snow line absurd. | weakness | **FIXED.** New §5.9 states the law (`h_max ≈ 88 300/g`, checked against Earth and Mars), computes the ceiling for both candidate bodies, shows the shipped relief law runs the wrong way against gravity, and hands the change to domain 02 as C-18 and Q8. |
| A-W4 | C-11's cost is understated; the new body has 13 rungs. | weakness | **FIXED.** Recomputed here: 2 580 chunks per face edge gives top rung 12, so **13 rungs**. §10 lists everything that re-measures — the golden table's shape, the boot self-check, the world tag, the cost table and the C = 256 grid. |
| A-W5 | HR5 at 100 % collides with "every test runs on the home planet". | weakness | **FIXED.** §6.3 and §11's SL5 and HR5 rows now agree: a charter is DATA of the one world, so a test may construct one, exactly as `height.rs:112-140` forces its cases today. No second world is needed and none is proposed. |
| A-W6 | The vegetation anchor has no home in the format, and a colliding tree must be in a digest. | weakness | **FIXED.** §7.2 adds the ANCHOR DIGEST beside the cell and mesh digests, states that an anchor is derived and never stored, and states that a felled tree writes one record with the provenance value the format already reserves. U20 adds it to the no-drift legs. |
| A-W7 | The dormant almanac has no carrier: a dormant realm sends no self-look. | weakness | **FIXED**, and the refuter's own evidence resolves it. §8.4 splits the two halves: the almanac is a SERVER fact that needs no carrier, and a dormant realm needs none because it is never drawn — it is a point of light from the parent's one radius. |
| A-W8 | The new soils may be the indicator material S5-6 forbids. | weakness | **FIXED.** §7.1 asks the question instead of asserting the answer, gives the argument that keeps the law (a soil names the climate, the ore stays live state), and raises C-19 with a fallback (name the soils by look). |
| A-W9 | §8.3 contradicts §3.4 about what crosses realm to realm. | weakness | **FIXED.** After A-B2 the charter is not a realm-to-realm datum at all. §8.9 states the qualification the refuter asked for: the charter derives FROM the parent's facts, but the realm derives them from the seed, so nothing crosses to make it. |
| A-W10 | The client's grid build is unbudgeted and ungated. | weakness | **FIXED.** §5.1 states the client's cost (about 100 ms and 6.29 MB per body), recommends a per-face build so the first chunk waits about 17 ms, and names the arrival hitch as an SL8 seam with its own measurement (U14) and decision row (C-16). |
| A-W11 | §10 compares a bolometric insolation against an XUV bound. | weakness | **FIXED**, and the arithmetic survives. Verified: `xuv_rel_of(1.0, G) == 1.0` (`taxonomy.rs:1511-1518`) and the home star is G class (`census.rs:57-63`), so the two numbers are equal here. §10 now names `xuv_rel_of`, states the G ratio, and U1 must print BOTH numbers. |
| A-W12 | §10's insolation chain assumes an unverified orbital rung. | weakness | **FIXED.** Verified: `home_body` is a `find_map` over the ladder's acceptance (`lib.rs:987-1003`). §10 now gives both candidate rungs (6.25 and 2.16 S⊕), shows the conclusion holds at either, and marks the numbers UNMEASURED with U1 to decide them. |
| A-W13 | C-2 invalidates a landed pin, and the tilt's carrier is left implicit. | weakness | **FIXED.** New §3.5 states all three costs: the pin's reason changes, the parent-authored `orient` quaternion (`pose.rs:919`) is the carrier, and every picture gate that reads "while the planet does not spin" must change. C-2 now carries the cost. |
| A-W14 | Citation drift: six wrong `file:line` ranges. | weakness | **FIXED.** All six corrected: `dropped_bound_m` is `body.rs:274-276`; `relief_bound_m` is 263; `BiomeField` is `body.rs:73` with the draw at 204-215; `Empty` is `strata.rs:61` inside the 41-61 block; the biome enum is `strata.rs:114-131`; and §1 no longer claims the face-edge count is asserted by a test — it is recomputed here and marked as such. |
| A-N1 | Nine terms are used and never explained. | note | **FIXED.** §2 gains: thermal Rossby number, chamfer pass, Legendre coefficient and `P2`, energy-balance climate model, Tetens form, biotemperature, degree-days, aerial perspective, optical depth, Rayleigh scattering, isostasy, ulp, and a weather system. `Aspect` keeps its gloss and is now in §2 as well. §4's techniques 2 and 3 now carry examples in the game's words. |
| A-N2 | §5.7 quotes the easy chunk only. | note | **FIXED.** §5.7 quotes both: 9 % of the surface chunk's headroom and 23 % of the cave-dense one's, and the budget is argued on the cave-dense case throughout. |
| A-N3 | "12 bytes" per grid cell is borrowed from the cell record. | note | **FIXED.** §5.1 lists the grid cell's fields and their widths: 16 bytes, and every memory figure is recomputed from it. |

### Refutation B — believability, cost and the owner

| # | Finding | Severity | Answer |
|---|---|---|---|
| B-B1 | The ladder gate uses a metre bound for a class; the biome flips by up to 1 499 m; the defect exists today. | blocker | **FIXED.** Same cure as A-D3, and the refuter's table is kept: §5.10 reproduces the dropped-bound table (5.1 m at rung 3 to 1 499.4 m at rung 11), names the shipped defect at `height.rs:40-46` and `chunk.rs:180-183`, and replaces the bound with the CLIMATE RUNG so the class is identical at every rung. |
| B-B2 | Lying snow as a per-cell diff costs 12 MB per square kilometre, and the melt is a second mechanism. | blocker | **FIXED.** The arithmetic is right and it kills the design. §8.3 makes snow depth a DERIVED field computed on both hosts from the shipped system list, quantised to the extractor's own 1/8-cell quantum, capped at one cell, writing no records; only a cell a player DIGS becomes a record. The melt, the trench collision and the budget line are each answered. |
| B-B3 | The owner asked to simulate the weather; the live layer is one paragraph with no model. | blocker | **FIXED.** New §8.5–§8.8: the anomaly field and its per-cell state, the step (semi-Lagrangian advection, fixed relaxation, seeded forcing), the shipped form (at most 64 weather-system rows of 16 bytes), the determinism rule (Category C, checkpoint-carried, the fence does not bind it), the cost table, the SL6 statement with its arm, four decision rows (C-20 and the split rows) and four measurements (U17–U19 plus the kill-9 test). |
| B-D1 | Every cost is computed on the body §10 says is wrong; the real numbers are four times larger. | defect | **FIXED.** Recomputed here: the Earth-like candidate's face edge is 10 235 904, so C = 256, 393 216 samples, **6.29 MB** at 16 bytes and about **100 ms**. §5.1's table gives both bodies side by side, and §5.7, §11 and U3 carry the new numbers. |
| B-D2 | The resolution "derivation" is degenerate; the clamp binds on every planet. | defect | **FIXED.** Same as A-W1. |
| B-D3 | The circulation anchor table does not reproduce: Mars gives 2 cells, not 1; Jupiter 3.06°, not 1.9°. | defect | **FIXED.** Recomputed here and the refuter is right at every row. §5.3 publishes the true table with **Mars marked MISS**, states that `dH = 0.388` is a reverse fit to Earth alone, and U8 owes a wider fit with published residuals. The ✓ is withdrawn. |
| B-D4 | V12 is mis-cited: it puts the light on the client DERIVING, not on a new row. | defect | **FIXED.** Verified at `owner_decisions_2026-09-07_voxels.md:484-503`. §8.1 states that S7-7 derives the light on the client and that V12 is a precedent for NOT adding a lane. The almanac is now derived, not shipped, and C-11 says so. |
| B-D5 | SL6 is asked for the charter and not for the almanac lane or the weather lane; three "almanac" items are fields, not scalars. | defect | **FIXED.** The almanac lane is deleted (B-D4), which removes the field-shaped items from any row. §8.8 gives the weather lane a full SL6 statement with its data, its direction, its rate, its arm and its cost. |
| B-D6 | §11's SL1 row is false: insolation IS a distance and the elements are a placement generator. | defect | **FIXED.** §3.6 and §11's SL1 row now state both halves: on the server nothing is told at all, and on the client the rows add nothing to what the window already carries. **The decision the refuter says exists is now C-15**, stated as a decision rather than denied. |
| B-D7 | Obliquity re-defines the body's frame and breaks a shipped pin; the cost is unstated. | defect | **FIXED.** Same as A-W13: new §3.5 states the pin, the orientation carrier and the picture gates, and C-2 carries the cost. |
| B-D8 | Precipitation is never computed in millimetres, yet the classifier's axis is millimetres. | defect | **FIXED.** §5.4 adds `K_mm_per_year` with its units, a two-point anchor (2 000 mm/yr at the ITCZ, 100 mm/yr under the subtropical high on an Earth-like charter) and a measurement row (U15). |
| B-D9 | A 2-pass chamfer is wrong on a closed cube, and the iterations have no determinism statement. | defect | **FIXED.** Same as A-D5: four fixed double sweeps in a fixed face order, no early exit, no convergence test, with U13 measuring the residual against an exact Dijkstra. §4 now applies its own no-data-dependent-iteration rule to this document's own passes. |
| B-D10 | The biome set does not add up, and the chart names a biome the set does not hold. | defect | **FIXED.** Same as A-D6. |
| B-D11 | The charter names no condensable volatile, yet the model needs one. | defect | **FIXED.** New §3.4 adds charter rows 10, 11 and 12, DERIVED from a five-row physical table (none, water, carbon dioxide, methane, ammonia) rather than drawn, with C-4 as its decision row. §5.6 and §5.4 now read them instead of an implied 273 K. |
| B-D12 | Nothing couples climate to SHAPE; a desert and a grassland have identical relief; the pillars claim does not follow. | defect | **FIXED, and stated as the deepest gap.** §7.3 withdraws the pillars claim, proves from `body.rs:193-201` that the body has ONE sediment kind and one thickness so no differential erosion can exist, states the arid-versus-humid landform difference plainly, and §12 makes the coupling to domain 03 a binding cross-domain requirement. Q9 and C-18 raise varied bedding and the isostatic cap. |
| B-D13 | At 41 km per cell the whole vista is one cell, so the output is contour banding. | defect | **FIXED.** §5.4's MID and LOCAL terms exist because of this finding, and §0 item 5 states the reason. The LOCAL term costs 0.05 ms and produces the picture's slope-to-slope contrast at the same height. |
| B-D14 | §10's headline rests on an inference the cited code does not support. | defect | **FIXED.** Same as A-W12: §10 is hedged, both rungs are given, the conclusion is shown to be robust to the rung, and C-13 now reads "YES, and U1 FIRST" rather than "YES, and first". |
| B-W1 | There is no ice-albedo feedback, so every polar cap is a circle of latitude. | weakness | **FIXED.** §5.6 states the limit by name, states what is lost (a bistable Snowball state), and recommends two FIXED albedo rounds as a cheap cure, with U12 to judge it on pictures. |
| B-W2 | "This one term places every large desert on Earth" is false. | weakness | **FIXED.** §5.4 names the four mechanisms, says which two the model has, which one it lacks (cold coastal currents), and states plainly that it will not produce an Atacama. |
| B-W3 | §5.7's per-column climate cost implies 0.217 ns per operation with no stated rate. | weakness | **FIXED.** §5.7 now states the unit: the closed-form part costs about ONE OCTAVE-EQUIVALENT per column at the MEASURED 13.19 ns, which is traceable arithmetic rather than an unstated operation rate. The total rose from 0.15 ms to 0.44 ms, and both chunk cases are quoted. |
| B-W4 | Six terms are used and never explained, and two of §4's examples are not in the game's words. | weakness | **FIXED.** Same as A-N1, plus §4's polynomial and table examples now speak about the pilot's valley and the haze over the far mesas. |
| B-W5 | U5 is a measurement that cannot fail, and it does not measure the real risk. | weakness | **FIXED.** Same as A-D8: U5 becomes a cross-target comparison of the charter integers in the shape `D-TERRAIN-1` already uses, and it must publish how many bodies sit within a quantum of a boundary. The refuter's related question — which charter wins on a disagreement — is answered in §3.7. |
| B-W6 | "byte-identical" and "the required result is zero" are false precision for a rendered sky. | weakness | **FIXED.** §8.4 and U10 now specify the same tick rendered twice from an identical camera, dormant against just-woken, compared within a stated per-pixel tolerance. |
| B-W7 | The haze needs more than a scale height. | weakness | **FIXED.** §2 gains optical depth and Rayleigh scattering, and §7.3 states the correct row: the haze comes from `p_surf`, `mu`, the temperature and the GAS IDENTITY over the scale height, and it differs in colour between a carbon-dioxide and a nitrogen world. |
| B-W8 | The seed ruling is checked for ore only, not for what the climate makes valuable. | weakness | **FIXED.** §11's seed-ruling row now gives the argument: timber, fresh water, arable ground and a terminator ring are ROOM, which the 2026-08-27 ruling settles by name, and who has taken them is live state. C-19 asks about the one open risk instead of asserting it away. |
| B-W9 | HR5 and "every test runs on the home planet" cannot both hold. | weakness | **FIXED.** Same as A-W5. |
| B-W10 | Two sourcing lapses: "12 bytes" per grid sample, and a ruling cited in a "what the code holds" table. | weakness | **FIXED.** The grid cell's 16 bytes are listed field by field (§5.1). The realm-box row is removed from §1's code table; §1 now cites `crates/core/src/look.rs` for what the self-look bag carries, which is code. |

### What both refuters agreed was sound, and is unchanged

- The fence analysis of §4 and its three techniques, and the refusal of a data-dependent iteration. Both
  refuters checked `gf.rs` line by line and confirmed it.
- The three-layer split as a SHAPE (static, closed-form time, live diff), and "a season must never
  invalidate a chunk" as a load-bearing law.
- The wake-at-zero construction as the right no-seam shape (the MEASUREMENT of it is corrected).
- The obliquity box: a tilt enters the static field through `s2` and the seasonal swing only, and `s2`
  changes sign near 54°.
- The look-radius precedent, the client's missing motion crate, the transcendental citations, `POLE_AXIS`'s
  cross-pin, the ladder arithmetic, the Earth lapse-rate check, the eccentricity arithmetic, the vegetation
  anchor counts, and the answer to all five of the owner's words (position, spin, trajectory, size,
  gravity).
