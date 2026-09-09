# 03 — Erosion, rivers, coasts and ice: how a planet gets its landforms

**Date:** 2026-09-08. **Revision 3** (answers refutation A round 2 and refutation B round 2; §16 holds
the answer tables for both rounds).
**Domain:** the shape water and ice cut into a body — valleys, drainage basins, river networks, lakes,
deltas, canyons, coasts, fjords and glaciers — and how that shape reaches every rung of the ladder.
**Status:** an investigation report for the owner. It designs the work. It decides nothing by itself.
**Binding law read first:** CLAUDE.md (HR1–HR6, SL1–SL10), the 2026-09-07 voxel ruling
(`docs/design/owner_decisions_2026-09-07_voxels.md`: V1 SL10, V2.1–V2.9, V4, V6, V8, V9, V10, V11, V12,
and **S7-1, S7-2 and S7-3**), the 2026-08-27 seed-and-secrecy ruling, the 2026-09-02 reach ruling, and
SL8.

**The owner's task, in his words:** *"Make sure that we reach that quality on the picture for earth-like
planets (biomes can be different of course, should be dependent on the planet position, spin, trajectory,
size and gravity, etc.). It should be very believable, as we also should simulate the weather."* And, of
our own pictures: *"no orienters and no details at all … the surface will not be interesting enough."*

**Neighbouring documents.** DOMAIN 02 owns the macro layout and the climate. This document consumes two
fields from it — precipitation and the snow line — and states the interface in §4.5 and §4.9. It states
four asks to DOMAIN 02 and the weather domain by name (§12 D12, D13, D14 and D17).

**What revision 3 changed.** Two refuters read revision 2 against the code and against the physics.
Refutation A found five blockers, ten defects, nine weaknesses and nine notes. Refutation B found five
blockers, eleven defects, eight weaknesses and two notes. **They were right about both headline changes
of revision 2, and both are rebuilt here.** Twelve sections are rewritten:

* **The slope spectrum no longer normalises to the relief (§4.2).** Revision 2 wrote *"scale every `a(o)`
  so that `Σ a(o) = relief_m`"*. Refutation B computed what that forces: `s_peak = 0.759` and a planet
  standing at 53° everywhere. **Revision 3 anchors `s_peak` on a PHYSICAL fact — the root-mean-square
  slope of the fine octaves at the roughest place equals the loose-rock talus tangent, 0.70 — and makes
  the envelope a CONSTRAINT (`Σ a ≤ relief_m`), not a normalisation.** COMPUTED on the home planet's real
  literals: the fine octaves then sum to 1 532 m of a 14 305 m relief, so the constraint is slack, and the
  band does not move.
* **A ROUGHNESS FIELD ends the uniform planet (§4.2).** Refutation B was right that one global spectrum
  gives no plains, no valley floors and no fields — and the owner's world is a building game. The
  roughness is now DERIVED from the macro field the solve already produces: rough where `Z` is high and
  steep, smooth where `Z` is flat. No new noise, no new draw, and it is erosion's own consequence.
* **The ladder's coarsening rule changes from BY COUNT to BY WAVELENGTH (§7.1, §9).** Both refuters
  showed the new spectrum multiplies `dropped_bound_m` by five to ten under `octaves_at`'s count rule.
  COMPUTED under the wavelength rule (keep an octave while `λ ≥ 4 · cell_m`): the dropped bound at rung 6
  falls from 32.8 m TODAY to **3.7 m**, and at rung 5 — the rung the 50 km vista is drawn at — to 1.2 m
  against a 57.5 m pixel. The rule fixes the new spectrum AND improves the ladder by an order of
  magnitude.
* **The macro node is a CELL CENTRE, and the seam argument is rebuilt on that (§5.4).** Refutation A
  proved `face_param` returns a cell centre in the OPEN interval (`crates/seed/src/ladder.rs:142-145`), so
  NO node sits on a cube edge and revision 2's shared-edge-node rule described a lattice that does not
  exist. The replacement is simpler and stronger: two partner faces' cell centres straddle the edge
  symmetrically at one node spacing, so the grid continues uniformly across the seam with no ownership
  rule at all.
* **The cube corner gets a stated rule (§5.4).** Refutation B was right that a 4 × 4 Catmull-Rom stencil
  reaches into a quadrant that does not exist at a corner.
* **The artifact carries a WATER LEVEL per node (§4.14, §6.1).** Refutation A proved the artifact could
  not answer its own water rule: three facies bits say *a* lake, never *which* lake. The artifact grows
  from 4 to **6 bytes per node, 14.75 MB** on the home planet. That number replaces 9.83 MB everywhere.
* **The realm no longer SHIPS the artifact (§5.2, §10, §12 D3).** Refutation A found ruling S7-2, which
  revision 2 never cited and which the ship breaks: the client generates eight home-planet chunks AT
  LOGIN (`crates/terrain/src/tag.rs:43-50`, `crates/terrain/src/digest.rs:22-26`). Refutation B found the
  ship has no lane and reverses SL10. **Revision 3 keeps SL10: the realm states the 28 bytes of FACTS on
  the statement that already exists, the client DERIVES the artifact, and the HOME PLANET's artifact is a
  committed, digest-pinned build artifact linked into both hosts — the `home.rs` shape.**
* **The facts are STORED, never recomputed (§5.2).** Refutation B showed revision 2 moved the drift from
  the client to the server, where a realm rescheduled from an M4 host to an x86-64 pod would re-address
  every stored edit. The facts are drawn once at the realm's creation and kept in the realm's own store.
* **A lake's base level is its SPILL LEVEL (§4.4, §4.6).** Refutation B found a gorge the depth of the
  lake at every lake inlet. The sweep's receiver term now reads the water level, not the lake floor.
* **Flat ground no longer drains in index stripes (§4.4).** Refutation A computed 625 m of artificial
  gradient over 10 000 flat nodes. `ε` is now zero and a flat is resolved by an integer distance field.
* **A SEDIMENT BUDGET lands, at one number per basin (§4.11).** Refutation A was right that a
  detachment-limited solve draws a delta and a floodplain out of nothing.
* **The cliff claim is withdrawn, and a real cliff mechanism replaces it (§7.3).** Both refuters showed
  the design's two rules DESTROY cliffs. Refutation B also proved the world already draws a SOFT rock —
  Sandstone, Limestone or Shale, 20–80 m thick (`crates/terrain/src/body.rs:194,198`) — which revision 2
  wrongly said did not exist. A cap-rock bench at that boundary is buildable today.

Every number below is marked MEASURED (with how), COMPUTED (with the arithmetic), ESTIMATED (with the
model) or UNMEASURED (with the bench that would settle it). **Revision 3 recomputed every spectrum number
on the HOME PLANET's own literals** (`HOME_PLANET_SEED = 7 701 581 858 760 374 086`, radius bits
`0x41499139_1e692dfa`) instead of on the round numbers revision 2 assumed; refutation A was right that
those numbers described a body that does not exist. Nothing here was run against `cargo`: this
investigation is read-only on the code by instruction.

---

## 0. The recommendation, in one page

**Erosion runs ONCE per body on a coarse lattice. Everything the player stands close to is a function of
the address and a 14.75 MB table. The relief the player SEES is redistributed across the octave ladder by
a slope spectrum anchored on the angle of repose and modulated by a roughness field.**

1. **A macro solve, per body, cached.** Take the body's COARSE octaves — those whose wavelength is at
   least four macro nodes — as the land before water touched it. Route the water with D8. Fill the pits
   with a priority flood, keeping the flooded surface and the terrain surface as two named fields.
   Accumulate the rain. Cut with the stream power law `dz/dt = U − K·Q^m·S^n` at `m = 1/2` and `n = 1`,
   solved implicitly in one downstream-to-upstream sweep, with a lake's SPILL LEVEL as the base level of
   its inlet. Carry one sediment number per basin. Rebound isostatically on a pyramid. Relax the
   over-steep slopes to the talus angle. Mark the ice and the coast. The macro lattice is its OWN uniform
   cube-sphere lattice of **640 nodes per face edge, 8 224 m per node, 2 457 600 nodes** on the home
   planet (COMPUTED from `N = 5 263 360`, `crates/seed/src/ladder.rs:70-96`; 640 divides `N` exactly).
2. **The solve keeps four small fields and throws the rest away.** An eroded macro height (`i16`,
   metres, relative to the ladder radius). A WATER LEVEL (`i16`, the same encoding, or a sentinel for
   dry). One byte holding the D8 receiver direction and the surface facies. One byte of
   quantised-logarithm discharge. **COMPUTED total for the home planet: 6 bytes per node,
   14.75 MB**, and a coarse pyramid of `Z` whose total is 1.64 MB and whose top level is **4.8 KB**
   (§6.1, §10). Revision 2 said 9.83 MB and 19 KB; both were wrong and both are corrected everywhere.
3. **The empty middle distance is a SPECTRUM defect, not a resolution defect — and the cure must not
   make a 53° planet.** COMPUTED from `crates/terrain/src/body.rs:143-183` on the home planet's own
   literals: `relief = 14 304.89 m`, `k_rough = 0.468 338`, and the recipe spends **7 606 m at 400 km,
   80 m at 6.25 km, 37.6 m at 3.1 km and 17.6 m at 1.6 km**. The reference picture needs 300–600 m
   there. The cure keeps the same relief budget and redistributes it with a SLOPE SPECTRUM whose peak
   amplitude is set by the ANGLE OF REPOSE, not by a normalisation (§4.2). COMPUTED at
   `o_peak = 7, σ = 1.4` and a fine-octave RMS slope of 0.70: **319 m at 6.25 km, 199 m at 3.1 km,
   80 m at 1.6 km**, summing to 1 532 m — well inside the relief.
4. **A ROUGHNESS FIELD makes a plain a plain.** The spectrum above is what the ROUGHEST place gets. The
   amplitude of every fine octave is multiplied by a field derived from the macro solve — high and steep
   `Z` gives 1, flat `Z` gives `m_min` (recommend 0.06). A craton is then a plain, an orogen is the
   reference picture's range, and the flat fields beside the river exist (§4.2).
5. **The fine rungs never solve anything.** `height_m` gains three terms: the eroded macro field `Z`,
   which REPLACES the coarse octaves; a channel and floodplain carve built on `segment_distance_m`
   (`crates/terrain/src/carve.rs:160`); and a detail term — scree, ridged crests, cap-rock benches,
   flow-warped gullies — whose slope comes from an ADDRESS-DEFINED stencil.
   **`height_m` gains the ADDRESS in its signature** (§7.3 rule 1). Every production caller already holds
   it (`crates/terrain/src/lattice.rs:59-63`, `Site { face, i, j }`); no caller inverts the bend.
6. **The ladder coarsens BY WAVELENGTH, not by count.** `octaves_at` keeps an octave while
   `λ(o) ≥ LADDER_SAMPLES_PER_WAVE · cell_m(rung)` (recommend 4). COMPUTED on the home planet under the
   new spectrum: the dropped bound at rung 6 is **3.7 m** against 32.8 m today and 116 m under the count
   rule. This is the change that lets the spectrum land without an arrival pop (§7.1, §9).
7. **`m = 1/2` is chosen so the float fence holds.** `Q^(1/2)` is `sqrt`, which SL10 V1.4 grants by name.
   Any DYADIC exponent is reachable by repeated square root. With `n = 1` the implicit update is one
   division per node, so the sweep needs no inner iteration and no stability test.
8. **The queue is integer.** The macro height lives as an `i32` in 1/16 m. The priority flood orders by
   `(height, node index)`. The D8 receiver compares cross-multiplied integers. §5.1 states EVERY float.
9. **The body's physical facts are DRAWN ONCE, STORED, and STATED — never recomputed.** The client links
   no motion crate (ruling S7-1), so it cannot compute the gravity, the surface temperature, the
   obliquity, the eccentricity or the spin the solve needs. The owning realm draws them at the realm's
   creation, keeps them in its own store, and states them as 28 bytes of quantised integers beside the
   surface statement it already sends (`crates/core/src/look.rs:65-74`). Storing them is what stops a
   realm rescheduled from an M4 host to an x86-64 pod from re-addressing every stored edit (§5.2).
10. **The client DERIVES the artifact; the HOME PLANET's is pinned in the build.** Ruling S7-2 makes the
    client generate eight home-planet chunks at login. So the home planet's artifact must exist BEFORE
    the first packet: it is a committed, digest-pinned build artifact in the `crates/terrain/src/home.rs`
    shape, linked into every host. Every other body's artifact is derived on the host that needs it, from
    the seed, the address and the 28 stated bytes. **SL10 is not reversed** (§5.2, §12 D3).
11. **Rivers are shape, never riches.** The graph may decide where sand, gravel, clay and a beach lie.
    It may never decide where a valuable deposit lies. A seed-derived placer bar is a treasure map.
12. **Water gets a level per node and per column, and the water sheet is CLIPPED.** The water surface is
    meshed by a second run of the extractor over `min(water_level − r, r − rock_surface)`, so the sheet
    ends at the shoreline. No cell record changes.
13. **The band does not move.** `Z` replaces the coarse octaves inside the same envelope, and the
    envelope is now `|Z| ≤ relief_m − A_fine`, an integer assertion. The DOWNWARD budget under the
    surface is stated in full and totals 64 m: channel 32 m, glacial floor 16 m, detail 16 m (§4.13).
14. **The gates are pictures and three curves that CAN fail.** The vista test (ruling V10 §12). The LAND
    hypsometric curve. The per-basin hypsometric integral, which sets the world's EROSIONAL AGE in years
    — a physical time, not a pass count (§4.12, §12 D9). And a slope histogram, which is the measurement
    that would have caught a 53° planet before the owner did.

**The one-sentence version.** One coarse erosion solve per planet decides where the water goes, how deep
it cut and how rough the ground is; a slope spectrum anchored on the angle of repose and modulated by
that roughness puts the missing relief into the band the eye is looking at without standing the planet on
end; the ladder coarsens by wavelength so the far view no longer throws away the octaves that carry it;
everything the player's eye resolves is a function of the seed, the address and a 14.75 MB table the
client derives for itself; the home planet's table is pinned in the build so the world hello still works;
and the body's physical facts are drawn once, stored by the realm and stated in 28 bytes, so no host
recomputes a transcendental and no reschedule moves the ground under a player's house.

---

## 1. What the code holds today (the only current truth)

Every row was re-opened for revision 3. Nine citations were corrected against the two refutations; the
corrections are named in §16.

| Fact | Where | What it means for this design |
|---|---|---|
| The surface is a sum of octaves and nothing else: `h = radius + Σ amplitude·noise(dir·frequency)`. | `crates/terrain/src/height.rs:17-27` | There is no erosion of any kind today. Every valley in a screenshot is a noise trough, which is why the hills read as blobs and not as land. |
| Octaves run from a 20–400 km coarsest wave down to 30 m, amplitudes falling by a factor `k_rough` in `[0.45, 0.55)`, scaled so their SUM is the relief. | `crates/terrain/src/body.rs:143-183` | **COMPUTED on the home planet's own literals** (`SplitMix64`, `crates/seed/src/rng.rs:22-28`; `child_seed`, `:70-74`; `draw_unit`, `crates/terrain/src/body.rs:113-115`): `relief = 14 304.89 m`, `k_rough = 0.468 338`, 14 octaves. The per-octave slope FALLS from 0.1195 at 400 km to 0.0510 at 49 m — a factor of 2.34. Revision 2 assumed `k = 0.5` and claimed a constant 0.094; refutation A is right that this described a body that does not exist. §4.2. |
| The relief is 0.4 % of the radius, clamped to `[200, 12 000]` m, times a factor in `[0.5, 1.5)`. | `crates/terrain/src/body.rs:147-151` | COMPUTED: `0.004 × 3 350 759 = 13 403`, so the clamp binds at 12 000 and the draw gives 14 304.89 m. There is plenty of relief; it sits at the wrong wavelengths. |
| The coarsest wavelength is a quarter to a half of the radius, clamped to `[20 km, 400 km]`. | `crates/terrain/src/body.rs:152-155` | COMPUTED: on the home planet the cap binds, so the coarsest wave is exactly 400 km. |
| The band (crust below, room above) is `crust_m = relief + strata + caves + 64` and `above_m = relief + 64`. | `crates/terrain/src/body.rs:233-236` | The 64 m on each side is SLACK. §4.13 budgets every new downward term inside it and states what the margin was probably for, which revision 2 spent to zero without asking. |
| A rung drops the finest octaves BY COUNT; the disagreement between rung `L` and rung 0 is the sum of the dropped amplitudes, measured over a 20 × 20 parameter grid on each face. | `crates/terrain/src/body.rs:251-277`; `crates/terrain/src/height.rs:80-105` | **This rule is the reason the slope spectrum could not land in revision 2**, and it is a defect today: at rung 6 the cell is 64 m and the rule throws away a 1 562 m octave. §7.1 replaces it with a WAVELENGTH rule. And the 400-sample test cannot measure a channel term — §14 M10 adds a channel-following fixture. |
| Caves are carved by a value-noise CAVERN FIELD on a per-face node lattice, and by TUBE segments with a radius; both switch off where a cell is too wide. | `crates/terrain/src/carve.rs:36-45,127-176` | The river carve is the same shape of machinery: a polyline, a radius, a distance and a rung gate. |
| `segment_distance_m` exists, is exact on the ends and the middle, and uses one `sqrt`. | `crates/terrain/src/carve.rs:160-176` | The river channel needs no new geometry primitive. |
| **`face_param(i, n)` returns a cell CENTRE, `(2i + 1)/n − 1`, in the OPEN interval `(−1, 1)`. `corner_param` returns `2i/n − 1` in the CLOSED interval.** | `crates/seed/src/ladder.rs:142-151` | **Revision 2's seam design was built on a lattice that does not exist.** No cell-centre node lies on a cube edge; the outermost sits half a node inside it — COMPUTED 4 112 m of arc on the home planet at `n_macro = 640`. §5.4 is rebuilt on the true geometry, and the true geometry turns out to be BETTER. |
| The cavern node lattice is built PER FACE, and `value_at` interpolates with `trilinear8`, which is C0. | `crates/terrain/src/chunk.rs:359-365,502-520,523-583` | The macro field cannot borrow it. A C0 field with per-face nodes gives a height scarp along the twelve cube edges — COMPUTED 63 000 km of it on the home planet. §5.4. |
| The fluid rule is one radius test: water under `sea_radius_m`, air over it, for EVERY body. | `crates/terrain/src/chunk.rs:205-213` | The sea is a perfect sphere today, a lake cannot exist, and an airless moon has an ocean. §8 changes this one function. |
| The sea radius is `radius_m + sea_offset.floor()`, and `radius_m = 2N/π` is not a whole metre. | `crates/terrain/src/body.rs:186-190`; COMPUTED 3 350 759.045 m | The OFFSET is floored, not the radius. The base level of every river already exists and is already seed-derived. |
| The extractor meshes where the gap byte changes sign, and water counts as NOT rock. | `crates/terrain/src/extract.rs:71-75` | The seabed is meshed. The top of the water is NOT meshed by anything today. §8.3. |
| The gap byte is 1/128 of a cell, signed; a cell exactly on the surface reads 0 and is air. | `crates/terrain/src/chunk.rs:44-48,118-127` | A channel carve changes the height, not the byte convention. Nothing in the record changes. |
| **The body draws ONE SEDIMENT of three — Sandstone, Limestone or Shale — 20–80 m thick, over ONE bedrock of five, all five igneous or metamorphic.** | `crates/terrain/src/body.rs:194,196-201`; `crates/terrain/src/strata.rs:136-151` | **Revision 2 said there is no soft rock in the world. That is false, and refutation B is right.** A wave-cut platform is metres deep and a cap-rock bench is tens of metres, so BOTH live inside the 20–80 m sediment. The lithology ask (§12 D14) shrinks to the DEEP case only. Revision 2 also cited `:196` for the sediment list; it is at `:194`, and `:196` is `topsoil_m`. |
| The biome reads latitude, height over the sea and two slow noises. | `crates/terrain/src/height.rs:40-64` | There is no rain, no wind and no rain shadow. DOMAIN 02 owns that. |
| `Stratum::Snow` is given by BIOME (`Tundra`), never by height above a snow line. | `crates/terrain/src/strata.rs:190-196` | The reference picture's most obvious feature is a snow-capped ridge, and nothing in the code puts snow on a high equatorial ridge. §4.9 and §12 D18. |
| The pole axis is `+Z`, the orbit's axis. There is no obliquity, no spin and no day length anywhere. | `crates/terrain/src/height.rs:29-35`; MEASURED: a grep for obliquity and spin over `crates/physics` and `crates/terrain` finds no body spin | The snow line and the climate need obliquity and a day length. Both are missing and must be drawn (§4.9, §12 D6). |
| The forest draws `ecc`, `sma`, `inclination`, `raan` and `arg_periapsis` per planet. | `crates/physics/src/worldgen/generate.rs` | **The ECCENTRICITY exists and revision 2 never used it.** It is the owner's word "trajectory", and §4.9 now consumes it. |
| The forest computes `mass_kg`, `radius_m`, `insolation_rel`, `t_eq_k`, `bond_albedo` and an optional atmosphere. Surface gravity is NOT stored: it is one line from the mass and the radius. | `crates/physics/src/taxonomy.rs:891-909` | The gravity costs nothing new. The temperature does not: `t_eq_k` is the EQUILIBRIUM temperature, 255 K for Earth. §4.6. |
| `Atmosphere` holds exactly three fields and no optical depth and no surface pressure; the struct's own doc records `D-TAX-1`. | `crates/physics/src/taxonomy.rs:874-888` | A greenhouse term has NO INPUT today. §4.6 states the one missing number. |
| Those facts are computed with `powf`, `ln` and `sqrt` OUTSIDE the float fence, and the module records its own determinism note: the cross-host bit-equality gate is DEFERRED to SPIKE-6a. | `crates/physics/src/taxonomy.rs:21-24,616,672,684,692-693,734-737` | This is why §5.2 draws the facts ONCE and STORES them — on the server as well as off the client. |
| `vd-terrain` and `vd-seed` are dependencies of `vd-client`; `vd-physics` stays dev-only. | ruling S7-1, `docs/design/owner_decisions_2026-09-07_voxels.md:497`; `crates/terrain/src/home.rs:1-9` | The client CANNOT compute any body fact. §5.2. |
| **Ruling S7-2, the world hello: the client states both halves of the world identity AT LOGIN, computed on the home planet's literals, and a client that cannot compute them does not log in.** `WorldIdentity::of` calls `golden_self_check(home)`, which calls `generate(body, key)` on EIGHT home-planet chunks. | ruling S7-2, `docs/design/owner_decisions_2026-09-07_voxels.md:498`; `crates/terrain/src/tag.rs:43-50`; `crates/terrain/src/digest.rs:22-26` | **Revision 2 never cited S7-2 and its recommendation broke it.** A chunk needs `Z`; `Z` came from an artifact that arrived after login. §5.2 and §12 D3 are rebuilt around it. |
| A realm already states its surface to the client: `SurfaceStmt { frame, generator }` under `TAG_SURFACE`, once per realm on change, retained, inside a **1 200-byte** self-look budget. | `crates/core/src/look.rs:52-79`; `crates/sim/src/stub/tests/window_lane.rs:2563` | 28 bytes of facts fit. **14.75 MB does not, by a factor of 12 000**, which is why revision 3 does not ship the artifact. |
| A body's look radius already enters the recipe from outside the fence, and the ladder's SNAP makes a drift in it harmless — MEASURED at a thousand ulps, with a stated tolerance of one top-rung cell edge, COMPUTED 1 304 m. | `crates/terrain/src/home.rs:44-77` | It is a SNAP with a margin of `10^12`, not a quantum. §5.2 does not rely on it. |
| The chunk sample box carries a one-cell halo, each halo column's height is an extra `height_m` call, and **every column of the box — core and halo — already carries its `Site { face, i, j }`**. | `crates/terrain/src/lattice.rs:48-50,59-63,124-140,246-280`; `crates/terrain/src/chunk.rs:92-107` | **No caller has to invert the bend.** Refutation B's estimate of 0.2–0.4 ms of `unbend` per chunk does not apply: the address is already in hand at every column, and §7.3 rule 1 states the signature change that uses it. |
| MEASURED costs (ruling V10, `terrain_cost`): a surface chunk 3.3 ms, a **cave-dense chunk 6.1 ms**, the budget 8 ms per chunk of worker time for the sample box AND the extraction together; a **synthetic 3-D checkerboard 26 ms**, called *"a BOUND no chunk of THE world reaches"*; one noise evaluation 12.19 ns. | `crates/bins/examples/terrain_cost.rs:38-45`; the noise figure from `scripts/noisebench` | The binding baseline is 6.1 ms. **And that baseline was MEASURED on today's recipe, which §4.2 replaces**, so it is not a baseline for the new design — §14 M2 must re-measure it. Revision 2's *"the column pass at rung 0 about 1.5 ms"* has NO source in the file it cited; it is deleted. |
| The home planet: `N = 5 263 360` cells per face edge, ladder radius 3 350 759.045 m, 12 rungs (top rung 11, a 2 048 m cell), 14 octaves. `N = 2^12 · 5 · 257`. | `crates/terrain/src/home.rs:16-40`; COMPUTED from `Ladder::for_radius` | `N` is a multiple of 2 048 and NOT of 8 192. Surface area COMPUTED 141.09 million km². §4.1. |

---

## 2. The words, explained once

The owner asked for the industry terms. Each is explained in plain words, then in the game's own words.
Revision 3 adds the six terms the round-2 refuters found used and never explained, and it fixes the two
words that carried two meanings.

**Hypsometry** — the share of a planet's surface that lies at each height. Earth's curve has two flat
steps — the deep ocean floor and the low continents — because Earth has TWO KINDS OF CRUST that float at
two heights. *In the game:* if the home planet's curve has one smooth hump, the planet reads as a lumpy
potato. This design cannot produce the two-step curve, because it has one crust; §12 D12 asks DOMAIN 02
for the crust field that would.

**The hypsometric integral (Strahler 1952)** — for ONE drainage basin, the average height of the basin
above its outlet, divided by the basin's total height range. A young, uncut basin scores near 0.8; a
mature one 0.4–0.6; an old worn one under 0.35. *In the game:* the number that says whether the home
planet's valleys are freshly cut or worn out, and the number that sets the world's EROSIONAL AGE in years
(§12 D9).

**Uplift (`U`)** — the rate at which tectonics pushes rock up, in metres per year. *In the game:* this
design sets `U = 0`, so the coarse octaves are not an uplift RATE; they are the INITIAL CONDITION of a
decay. The home planet is a world whose mountains were built and then left to the water. §4.6 states what
that costs and §12 D9 states what it fixes.

**Isostasy** — the crust floats on the mantle like a raft. A mountain has a deep root that holds it up.
When erosion takes a metre of rock off the top, the raft rises again by about four fifths of a metre.
*In the game:* without it, forty sweeps flatten a range into a plain and the home planet loses its
highlands.

**The flexural wavelength** — the crust is a stiff sheet, so it rises over a wide region around the place
the rock was removed. The width of that region is the flexural wavelength, 100–200 km on Earth's
continents. *In the game:* the reason a valley cut into the home planet's range lifts the ridge NEXT to
it too, which is what keeps the ridge tall.

**Effective elastic thickness (`T_e`)** — how thick the stiff part of the crust is, in kilometres. It
sets the flexural wavelength. *In the game:* a fact about what the home planet IS, like a hull's rating.
It is missing from the code and §12 D6 draws it.

**Base level** — the lowest height a river can cut down to. For a river that reaches the sea, the base
level is the sea; **for a river that reaches a lake, the base level is the lake's SPILL LEVEL, not the
lake's floor.** *In the game:* the body's `sea_radius_m` (`crates/terrain/src/body.rs:190`), and the fix
in §4.6 that stops a gorge appearing at every lake inlet.

**Drainage basin (catchment)** — all the land whose rain leaves through one point. *In the game:* every
macro node owns a basin; the pilot who lands in a valley is standing inside one.

**Flow accumulation (`A`)** — for each point, the area of land upstream of it. **Discharge (`Q`)** — the
volume of water per second passing a point: the accumulated RAIN, not the accumulated area. *In the
game:* a desert basin the size of a continent carries less water than a wet mountain valley one tenth its
size. **This document uses `Q` everywhere; `A` appears only in the explanation above.**

**D8 routing** — each lattice node sends ALL its water to the ONE of its eight neighbours that lies
steepest downhill. The result is a tree. *In the game:* D8 gives the river GRAPH directly — each macro
node has one downstream node, so a river is a chain of nodes, which is exactly the polyline the chunk
pass carves against.

**A pit (a local minimum)** — a node with no lower neighbour. **Priority flood** is the standard cure
(Barnes, Lehman and Mulla 2014): start from the coast, walk inward always taking the lowest unprocessed
node, and raise each node to at least the level it was reached at. *In the game:* without it the home
planet holds a million puddles and no rivers.

**A flat, and flat resolution** — after the flood, a region of equal height has no downhill neighbour at
all, so D8 cannot choose. The standard cure (Garbrecht and Martz 1997) routes a flat by the DISTANCE to
the nearest outlet, not by height. *In the game:* on the home planet's continental interior the rivers
must run toward the sea and not in stripes down the node index, which is what revision 2's `ε` trick
would have given (§4.4).

**Stream power law** — `dz/dt = U − K·Q^m·S^n`. The land rises at the uplift rate and is cut down at a
rate set by how much water passes, how steep it is, and how soft the rock is (`K`, the **erodibility**).
**Detachment-limited** means the removed rock vanishes; **transport-limited** means the removed rock is
carried and dropped. *In the game:* this design is detachment-limited with ONE deposit number per basin
(§4.11), so a delta is built from rock that was really removed upstream.

**Thermal erosion and the talus angle (the angle of repose)** — loose rock cannot stand steeper than
about 35°, whose tangent is 0.70 (COMPUTED: `atan(0.70) = 35.0°`). Anything steeper sheds until it
reaches that angle, and the shed material piles at the foot as **scree**. *In the game:* the scree fan
under a cliff in the reference picture — and the number that anchors the whole slope spectrum (§4.2).

**Frost weathering (freeze-thaw)** — water in a crack freezes, expands and splits the rock. Its rate
depends on how many times a year the surface crosses freezing, which depends on the DAY LENGTH's
temperature swing. *In the game:* this is where the owner's word "spin" reaches the shape — a slowly
spinning planet has a huge day-night swing and sheds rock faster (§4.6).

**Glacial carving** — above the snow line, snow becomes ice, the ice flows downhill, and it grinds the
valley into a **U** shape. The bowl it starts in is a **cirque**. **Over-deepening** is what a glacier
does that a river cannot: it cuts a hollow in the valley floor that water would have to flow UPHILL to
leave, which is why a glaciated valley holds a chain of lakes.

**The snow line, and the equilibrium line altitude (ELA)** — the snow line is where snow lasts the year.
The ELA is the height on a glacier where the snow that falls exactly balances the ice that melts. It
depends on how much snow FALLS as much as on how cold it is: the dry Andes carry an ELA near 6 000 m
where the 0 °C isotherm is near 4 800 m. *In the game:* the white caps on the ridges, and the reason a
desert range gets no glacier even when it is high.

**A fjord** — a glacial valley whose floor the glacier cut BELOW today's sea level, and which the sea
then flooded. *In the game:* free, once the ice pass and the sea level run in the right order.

**Orographic lift and the rain shadow** — air forced up over a range cools, and cool air cannot hold its
water, so it rains on the windward side. The far side is a desert. *In the game:* why the wet side of a
range grows big rivers and forest, and the dry side grows a desert biome.

**The lapse rate (`Γ`)** — how fast the air cools as you climb. The **dry adiabatic** rate is `g / c_p`,
COMPUTED 9.76 K/km on Earth. The **environmental** rate is what the real atmosphere shows, about
6.5 K/km, lower because rising air condenses water and gives back its latent heat. *In the game:* the
snow line moves by 500 m between the two, so §4.9 says which it uses.

**Optical depth (`τ`)** — how much of the ground's heat the atmosphere holds in. A grey slab of optical
depth `τ` warms the surface by the factor `(1 + 0.75·τ)^(1/4)`. COMPUTED for Earth: 288 / 255 = 1.129,
so `τ = 0.835`. *In the game:* the one number that separates an Earth from a rock, and the one number the
code does not hold (`D-TAX-1`).

**Eccentricity (`e`)** — how far an orbit departs from a circle. It makes the insolation swing through
the year, which widens the range the snow line moves over. *In the game:* the owner's word "trajectory",
already drawn by the forest and consumed in §4.9.

**Nyquist** — a lattice of spacing `s` can carry a wave only if the wave is at least `2s` long; a shorter
wave FOLDS and appears as a false long wave locked to the lattice. *In the game:* the reason the macro
solve reads only octaves whose wavelength is at least FOUR macro nodes (§4.2), and the reason revision 2
would have painted a regular 8 km pattern over every continent.

**A knickpoint** — a step in a river's long profile, where the bed drops suddenly. *In the game:* the
waterfall the channel term draws where the bed's slope jumps (§7.2).

**A delta** — the fan of sediment a river drops where it meets still water. **An alluvial fan** — the
same thing where a mountain stream leaves a range onto a plain, and the landform that most often carries
a settlement in a range like the reference picture's. **A floodplain** — the flat strip beside a river
that the river built out of its own sediment. *In the game:* the flat fields in the reference picture,
and the only ground flat enough to build a settlement on without terracing.

**FACIES — one word, one meaning.** In this document a FACIES is one of eight DEPOSITIONAL ENVIRONMENTS:
upland, floodplain, delta, beach, scree, glacier, lake bed, sea bed. It is not a substance. The
SUBSTANCE the cell record holds is a `Stratum` (`crates/terrain/src/strata.rs`), and a stated table maps
a facies plus a depth to a stratum — a beach facies gives `Sand`, a scree facies gives `Gravel`, a
floodplain facies gives `Clay` under `Dirt`. The macro artifact holds the NODE FACIES, at 8 224 m; the
fine pass derives the COLUMN FACIES, at the cell. Both take values from the SAME eight. Revision 2 used
"facies" for both the environment and the substance, and both refuters caught it.

**Hydraulic geometry** — the measured rule that a river's width and depth grow as powers of its
discharge (Leopold and Maddock 1953). *In the game:* the way the chunk pass turns one discharge byte into
a channel a pilot can see from the air.

**Freeboard** — how high the floodplain stands above the water surface. *In the game:* why the fields
beside the river are dry and the river is not.

**A cap rock, and a bench** — a hard layer over a soft one. The soft layer wears back, the hard layer
stands out, and a flat step with a vertical face forms at the boundary. *In the game:* the world already
draws a 20–80 m soft sediment over a hard bedrock (`crates/terrain/src/body.rs:194,198`), so a bench of
up to 80 m with a real vertical face is buildable TODAY. That is the reference picture's near cliff band,
and it is the mechanism revision 2 did not have (§7.3 rule 2).

**An arête** — the sharp knife-edge ridge left between two glacial valleys. **Dendritic** — branching
like a tree, the shape a natural river network has. **Smoothstep** — a blending curve that starts and
ends flat, so two surfaces meet without a crease; the crate already holds the quintic form
(`crates/terrain/src/noise.rs:66-75`). **A placer** — a deposit of heavy valuable mineral concentrated by
a river in its gravel; exactly the thing the seed ruling forbids the graph to decide.

**Aeolian** — shaped by wind: dunes, wind-cut rock. **Karst** — the landscape limestone makes when water
dissolves it. **Volcanic landforms** — cones, calderas, lava plains, volcanic necks. *In the game:* three
landform families this document does NOT build, named in §8.4 so none is a surprise later.

---

## 3. The shape of the answer: one solve, one spectrum, one roughness field, then closed form

```
   ONCE PER BODY (cached; derived on the host that needs it)   EVERY CHUNK (closed form)
   ─────────────────────────────────────────────────────────   ─────────────────────────
   the COARSE octaves of BodyDefinition                        height_m(body, site, rung)
   (λ ≥ 4 macro nodes; the land before water)                          │
            │                                                          │
            ▼                                                          │
   ┌──────────────────────┐  precipitation, ELA ◄── DOMAIN 02          │
   │   THE MACRO SOLVE    │                                            │
   │   640 nodes/edge     │  D8 → priority flood → flat resolve        │
   │   8 224 m per node   │  → discharge → stream power → sediment     │
   │   2 457 600 nodes    │  → flexural rebound → talus                │
   │   fixed pass counts  │  → ice mark → coast mark                   │
   └──────────┬───────────┘                                            │
              │                                                        │
              ▼                                                        ▼
   ┌────────────────────────────────────────┐   read ───►  h = radius
   │  THE ARTIFACT (14.75 MB, home planet)  │                + Z(site)     REPLACES the coarse octaves
   │  • eroded height Z       i16 per node  │ ──────────►    + Σ FINE octaves kept at this rung,
   │  • water level           i16 per node  │                    on the SLOPE SPECTRUM (§4.2)
   │  • D8 receiver + facies  1 byte/node   │                    × the ROUGHNESS FIELD (§4.2)
   │  • discharge             1 byte/node   │ ──────────►    − C(site, rung)  channel + floodplain
   │  • a coarse PYRAMID of Z (4.8 KB top)  │                + T(site, rung)  scree, benches, gullies
   └────────────────────────────────────────┘              and, per column, a WATER LEVEL
```

**Why the split is the only affordable one.** A river's width at your boots depends on every square metre
of land upstream of you. That is a global fact, and no per-chunk rule can compute it. A lattice fine
enough to draw a 5 m stream would hold 6.3 million nodes at 5 140 m and 158 million at about 1 km on the
home planet (COMPUTED), which is gigabytes in the solve. So the fine part must be closed-form. §12 D1
puts the line at 8 224 m.

**Why a coarse solve is not enough by itself, and why more amplitude alone is not either.** Across a
50 km vista the solve at 8 224 m decides the trunk valley, the ridge line and where the snow starts. It
CANNOT put a ridge every 1–3 km, because it has no nodes there. And the octave recipe puts only 37.6 m of
amplitude at 3 km, where the reference picture has 300–600 m. **But amplitude alone gives a 53° planet**
— refutation B computed exactly that from revision 2's own law. **So the answer has FOUR parts:**

1. the SOLVE gives the STRUCTURE — where the valley runs, where the water is, where the ice was;
2. the SLOPE SPECTRUM gives the AMPLITUDE in the 1–6 km band, anchored on the angle of repose so the
   ground stays walkable;
3. the ROUGHNESS FIELD gives the CONTRAST — a rough orogen against a smooth plain, so the picture has
   both a range and the fields at its foot;
4. the closed-form DETAIL gives the ORGANISATION — sharp crests, cap-rock benches and gullies that all
   run downhill toward the same river, which turns amplitude into landforms instead of blobs.

*In the game:* the pilot stands on a ridge on the home planet and looks 50 km down the valley. The solve
decided which way the valley runs and where the river is. The spectrum decided that the spurs on the far
side stand 300 m tall instead of 38 m. The roughness field decided that the valley floor under him is
flat enough to land on and to build on. The detail decided that the spurs are sharp-crested, that a
cap-rock bench runs along them at one height, and that the gullies on their flanks all run toward the
same river.

**At which rung is that vista drawn?** COMPUTED: at 50 km, with 720 rows and a 45° vertical field of
view, one pixel subtends 57.5 m, so the far ridge is drawn at rung 5 (a 32 m cell) or rung 6 (64 m).
Refutation B was right that revision 2 never named the rung, and it is the rung at which the ladder rule
of §7.1 has to hold. COMPUTED there: the rung-to-rung disagreement is 1.2 m at rung 5 and 3.7 m at
rung 6, both far under one pixel.

---

## 4. The macro solve, step by step

### 4.1 The macro lattice — its own lattice, a divisor of `N`, and a CELL-CENTRE convention

**THE MACRO LATTICE IS ITS OWN UNIFORM CUBE-SPHERE LATTICE.** It has `n_macro` nodes along each face
edge. A node is the CELL `(face, i, j)` of that lattice, and its direction is
`direction(face, face_param(i, n_macro), face_param(j, n_macro))` — the same call the column pass makes
(`crates/terrain/src/chunk.rs:137-141`), with a different `n`. It is not a rung of the ladder, and it
never appears in an address: it is a DERIVED FIELD, so it may sit on a lattice the address never names.

**The node is a CELL CENTRE, and no node lies on a cube edge.** `face_param` is
`(2i + 1)/n − 1`, in the OPEN interval `(−1, 1)` (`crates/seed/src/ladder.rs:142-145`). The outermost
node therefore sits at `a = 1 − 1/n_macro`, half a node inside the cube edge — COMPUTED 4 112 m of arc on
the home planet. **Revision 2 said node planes sit ON the cube edges and built its whole seam rule on
that. Refutation A is right that this is false**, and §5.4 is rebuilt. The convention is stated once,
here, and used everywhere: cell centres, never corners.

**`n_macro` must DIVIDE `N`.** Not because the seam needs it — the seam table is size-independent and
takes `n_l` (`crates/seed/src/seam.rs:10-11,183`) — but because it makes the per-chunk lookup INTEGER: a
rung-`L` cell index `i` maps to its macro node by `(i · 2^L) / (N / n_macro)`, an exact integer division
with no float comparison anywhere. It also removes any ragged strip at a face edge.

**The rule, stated so it cannot contradict itself.** Refutation A found revision 2's wording ambiguous on
a small body, and it was. The rule is:

```
   candidates  =  every divisor d of N with d ≥ MIN_MACRO_EDGE          (recommend MIN_MACRO_EDGE = 8)
   n_macro     =  the candidate minimising | N/d  −  MACRO_CELL_TARGET_M |,
                  ties to the SMALLER candidate
```

There is no clamp after the choice, so `n_macro` always divides `N`. `N ≤ 2^26`
(`crates/seed/src/ladder.rs:25`), so the divisors are found by trial division to `sqrt(N) ≤ 8 192` —
exact integer work, and identical on every target. **Worked on refutation A's own small body:** a 5 km
body of THE world snaps to `N = 7 854 = 2 · 3 · 7 · 11 · 17`; every divisor gives a node smaller than the
8 192 m target, so the rule takes the smallest candidate at or above 8, which is 11 — a 714 m node and
726 nodes. That is a stated answer, not a contradiction, and the same code ran on it (HR4).

COMPUTED for the home planet (`N = 2^12 · 5 · 257`), at 6 bytes per node:

| candidate `n_macro` | node size | nodes | artifact |
|---|---|---|---|
| 320 | 16 448 m | 614 400 | 3.7 MB |
| 512 | 10 280 m | 1 572 864 | 9.4 MB |
| 514 | 10 240 m | 1 585 176 | 9.5 MB |
| **640** | **8 224 m** | **2 457 600** | **14.75 MB** |
| 1 024 | 5 140 m | 6 291 456 | 37.7 MB |
| 1 285 | 4 096 m | 9 907 350 | 59.4 MB |

**640 wins:** it divides `N`, its node is 0.4 % over the 8 192 m target, and it is the largest that keeps
the solve's transient state inside about 130 MB (§10). COMPUTED for other bodies of THE world: a 200 km
moon gets `n_macro = 32` and 6 144 nodes; a 50 km body gets `n_macro = 8` and 384 nodes.

**A node's AREA is not constant, and the sign is the opposite of what revision 1 said.** COMPUTED from
the bend's own constants (`crates/seed/src/bend.rs:23-30`) with the area density
`W'(a)·W'(b) / |n + W(a)u + W(b)v|³`:

| position on the face | area density | ratio to the face centre |
|---|---|---|
| centre `(0, 0)` | 0.616 850 | 1.000 |
| edge midpoint `(1, 0)` | 0.432 739 | **0.702** |
| cube corner `(1, 1)` | 0.467 391 | **0.758** |

A node near a cube corner is about 24 % SMALLER than one at a face centre. Flow accumulation must add
REAL square metres, or a basin near a face corner carries the wrong river.

**The area is a MIDPOINT QUADRATURE, not a closed form, and the SUMMATION is what is exact.** Refutation
A is right to separate the two, and revision 2 called the wrong half exact. The density is evaluated at
the node's centre from the polynomial `W'(a) = k₁ + 3k₂a² + 5k₃a⁴` — the same expression `unbend`
differentiates (`crates/seed/src/bend.rs:52-65`) — and multiplied by the node's parameter area. COMPUTED
error: the density varies by 24 % across a whole face, so across one node of 1/640 of a face the midpoint
error is parts per ten thousand. The result is then FLOORED to a whole square metre and summed as a
`u64`, so the ACCUMULATION is exact integer arithmetic with no rounding order at all.

**Three area numbers, and which is which.** Refutation B is right that revision 2 quoted the largest
everywhere. COMPUTED on the home planet at `n_macro = 640`: a face-centre node is 67 634 176 m²; the MEAN
node is `1.411 × 10^14 / 2 457 600 = 5.74 × 10^7 m²`; a corner node is about `4.75 × 10^7 m²`. This
document uses the MEAN for a typical figure, the MINIMUM for the discharge floor (§6.1), and never the
maximum for either.

**Neighbours across a seam.** A node on a face edge has neighbours on the partner face, found through
`across(face, edge, along, n_l)` (`crates/seed/src/seam.rs:183`), which takes `n_l` and so works at
`n_macro` unchanged. Some seam records REVERSE the along-edge direction (`crates/seed/src/seam.rs:191`),
so the stencil must be built through the table and never by adding one to `i`. At a cube CORNER only
three faces meet, so a corner node has SEVEN neighbours, not eight — COMPUTED from the cube-sphere's own
geometry; the extractor states the same fact about its mesh ring
(`crates/terrain/src/extract.rs:13-18`), which is a different object and is cited here only as the
crate's own record that three faces meet.

```
   face +X                       face +Y
   ┌───────────────┐ seam ┌───────────────┐
   │           n7 n0 n1   │               │   a node on the +X face's +u edge finds n1, n2 and n3
   │           n6  X  n2  │               │   on face +Y, with the axes swapped and, on some
   │           n5 n4 n3   │               │   edges, reversed. Never i+1.
   └───────────────┘      └───────────────┘
```

### 4.2 The initial surface, THE SLOPE SPECTRUM, and THE ROUGHNESS FIELD

This is the section the owner's complaint lives in, and it is the section both round-2 refuters broke.
Revision 3 rebuilds it in three parts: which octaves the solve may read, how much amplitude the rest
carry, and where that amplitude is spent.

#### 4.2.1 The octave selector: FOUR samples per wavelength, not one

**The solve may read only an octave the macro lattice can carry.** Refutation A is right that revision 2
folded two octaves: at an 8 224 m node the Nyquist wavelength is 16 448 m, so a 12 500 m octave folds and
paints a false long wave locked to the lattice — the eleven-seam taxonomy's *"detail-by-box"* row at
continental scale. Three samples per wavelength is not enough either, because a value-noise octave is not
band-limited.

**The rule:** an octave is COARSE (the solve reads it, and `Z` replaces it) when
`λ(o) ≥ MACRO_SAMPLES_PER_WAVE · macro_node_m`, with `MACRO_SAMPLES_PER_WAVE = 4`. Every other octave is
FINE, and the fine octaves are what `height_m` still sums.

COMPUTED on the home planet at an 8 224 m node (threshold 32 896 m): octaves 0 to 3 (400 km, 200 km,
100 km, 50 km) are COARSE; octaves 4 to 13 (25 km down to 49 m) are FINE. `A_coarse` — the sum of the
coarse amplitudes today — is 13 617 m of the 14 305 m relief.

#### 4.2.2 There is no uplift field, and there must not be one

SL5 and the no-magic-numbers rule both say the shape comes from the seed. The coarse octaves ARE the
seed's statement of where the land is high. `U = 0`: the solve redistributes what the seed drew and adds
no new material.

**What that costs, stated.** Refutation A is right that `U` in the stream power law is a RATE and a
coarse octave is a HEIGHT. With `U = 0` there is no steady state; the landscape decays toward the base
level. So this design models a POST-OROGENIC world — a range that was built and then left to the water —
and it does NOT model an active orogen with a steady-state knickpoint belt. That is a real and named
limitation. §12 D9 fixes the free-constant half of it, and §13 open question 11 names the missing
landform family.

#### 4.2.3 THE SLOPE SPECTRUM, anchored on the angle of repose

COMPUTED from `crates/terrain/src/body.rs:157-183` on the HOME PLANET's own literals (relief 14 304.89 m,
`k_rough = 0.468 338`, 14 octaves from 400 km):

| wavelength | amplitude TODAY | per-octave slope TODAY |
|---|---|---|
| 400 km | 7 605.6 m | 0.1195 |
| 25 km | 365.9 m | 0.0920 |
| 12.5 km | 171.4 m | 0.0861 |
| 6.25 km | **80.3 m** | 0.0807 |
| 3.13 km | **37.6 m** | 0.0756 |
| 1.56 km | **17.6 m** | 0.0708 |
| 49 m | 0.4 m | 0.0510 |

The reference picture's ridges repeat every 1–3 km and stand 300–1 000 m above the valley floor. The
recipe gives 37.6 m and 17.6 m there.

**The diagnosis, corrected.** Revision 2 said the spectrum is *"a constant 0.094 at every wavelength …
uniformly gentle"*. That is false for the home planet and refutation A proved it. The truth is two
things, and both matter:

* the per-octave slope FALLS from 0.1195 to 0.0510, so the surface gets progressively SMOOTHER toward
  fine scale — a big smooth swell that loses what little texture it has as you look closer;
* the RMS surface slope is COMPUTED `sqrt(Σ s(o)²) = 0.3126`, which is 17.4° — **not gentle at all**, as
  refutation B's note N1 says. Earth's mean land slope is a few degrees.

**So the defect is NOT "too little slope everywhere". It is two defects at once: NO CHARACTERISTIC SCALE
(nothing for the eye to fix on) and NO CONTRAST (the same roughness on a plain and on a range).** Adding
amplitude everywhere fixes neither and breaks walkability. §4.2.3 fixes the first; §4.2.4 fixes the
second.

Real eroded topography has a characteristic scale: its roughness is largest at the VALLEY SPACING, where
hillslopes stand near their threshold angle, and it falls away at continental scale (gentle) and below
the hillslope length (smoothed by soil creep). **Refutation B is right that the published picture is
usually described as a spectral BREAK rather than a PEAK, and that revision 2 gave this sentence no
source.** Revision 3 states it honestly: the design uses a BUMP, the literature's shape is arguable
between a bump and a break, and **§14 M9 decides which by measurement against a real elevation model
before the constant is frozen.** The bump is the recommendation, not a finding.

**The law.** State the per-octave SLOPE, not the per-octave amplitude ratio:

```
   s(o)  =  s_peak / ( 1 + ((o − o_peak)·ln2)² / σ² )      a rational bump in the OCTAVE INDEX
   a(o)  =  s(o) · λ(o) / 2π                               slope to amplitude
```

**`s_peak` is anchored, not normalised. THIS IS THE FIX FOR THE 53° PLANET.** Revision 2's third line was
*"then scale every `a(o)` so that `Σ a(o) = relief_m`"*. Refutation B computed what that forces on the
home planet: `s_peak = 0.759` and an RMS surface slope of 1.345, which is 53.4°. A player could not walk,
and the collider is the same shape, so he would slide. That line is DELETED. In its place:

```
   ANCHOR:      sqrt( Σ over the FINE octaves of s(o)² )  =  TALUS_RMS · tan(θ_loose)
   CONSTRAINT:  Σ over all octaves of a(o)                ≤  relief_m
```

* the ANCHOR is a physical fact: at the ROUGHEST place on the body the fine octaves together stand at the
  loose-rock angle of repose. `tan(θ_loose) = 0.70`, from the bedrock table (§4.8); `TALUS_RMS = 1.0` is
  the recommendation, and §14 M9 measures it;
* the CONSTRAINT replaces the normalisation. It is an INEQUALITY, so spending LESS than the relief is
  lawful — the band is derived from `relief_m`, not from the spectrum's sum, so a smaller sum cannot
  break an address. If the constraint ever binds, scale every `a(o)` down uniformly, exactly as the
  recipe already does (`crates/terrain/src/body.rs:180-183`).

COMPUTED on the home planet, `o_peak = 7` (3.13 km), `σ = 1.4`, fine octaves 4–13, anchored at 0.70:

| wavelength | amplitude TODAY | amplitude NEW (roughest place) | slope NEW |
|---|---|---|---|
| 400 km (coarse) | 7 605.6 m | 7 605.6 m — unchanged, and `Z` replaces it | 0.1195 |
| 50 km (coarse) | 781.3 m | 781.3 m — unchanged | 0.0982 |
| 25 km | 365.9 m | **496.1 m** | 0.1247 |
| 12.5 km | 171.4 m | **401.6 m** | 0.2019 |
| **6.25 km** | 80.3 m | **319.4 m** | 0.3211 |
| **3.13 km** | 37.6 m | **198.8 m** | 0.3998 |
| **1.56 km** | 17.6 m | **79.8 m** | 0.3211 |
| 781 m | 8.2 m | 25.1 m | 0.2019 |
| 195 m | 1.8 m | 2.5 m | 0.0812 |
| 49 m | 0.4 m | 0.3 m | 0.0407 |

* `s_peak = 0.3998`. The fine octaves sum to **1 532 m**, against `relief_m = 14 305 m`. **The constraint
  is slack by a factor of nine, so the envelope never binds and the band never moves.**
* The RMS slope of the fine octaves is 0.70 by construction — 35°, a real steep mountain hillslope, not
  53°. Adding the coarse `Z`'s own slope (a 1 000 m relief over 8 224 m is 0.12) leaves the total under
  0.72.
* **It is fence-legal.** `o` is an integer and `ln2` is a constant, so `(o − o_peak)·ln2` is one multiply
  of a constant by an integer. No logarithm, no exponential — only add, multiply and one divide, all in
  `from_seed`, which already runs under the fence.
* **`o_peak` is derived, not chosen:** it is the octave whose wavelength is the VALLEY SPACING, which
  erosion sets from the hillslope length. §14 M9 measures it against a real elevation model.

#### 4.2.4 THE ROUGHNESS FIELD: a plain is a plain

**Refutation B's second blocker is correct and it was the more important of the two.** One spectrum
applied to the whole body gives the same roughness on a craton and in an orogen. The reference picture is
half made of the exception: flat fields beside the river, a broad valley floor, a plain running to the
forest edge, and a range behind it. And the owner's world is a building game: with a globally rough
planet there is nowhere to build without terracing.

**The field is DERIVED from the solve, not drawn.** No new noise, no new seed draw, no new artifact
field. Erosion itself decides where the ground is rough: a place with high, steep macro relief is an
orogen; a place where `Z` is flat is a plain, a valley floor or a craton.

```
   r_raw(site)  =  |∇Z| / SLOPE_REF        the macro slope, from the same 4 × 4 gather §5.4 already does
   m(site)      =  m_min  +  (1 − m_min) · smoothstep(0, 1, min(r_raw, 1))
   a(o, site)   =  a(o) · m(site)                      every FINE octave, one multiply
```

* `SLOPE_REF` is a world constant: the macro slope at which the ground is fully rough. Recommend the
  loose-rock talus tangent again, 0.70, so *"as steep as rock can stand at 8 km"* means *"as rough as it
  gets at 1 km"*. One number, one meaning, taken from a physical fact.
* `m_min` is the roughness of the flattest ground. Recommend 0.06: COMPUTED, a craton's fine RMS slope is
  then `0.70 × 0.06 = 0.042`, which is 2.4° — Earth's plains.
* The field is SLOW: `Z` varies over 8 224 m, so `∇m` contributes a slope of at most
  `a(o)/8 224 ≈ 0.04` at the peak octave. It cannot itself make a step.
* It is a function of the ADDRESS, because `Z` is, and it is C1, because §5.4's interpolation is.
* It costs ONE multiply per octave per column, on a gather the column pass already performs.

**Two consequences the owner will see.** In the range, the spurs stand 300 m tall every 3 km. Two
kilometres away on the valley floor, the same 3 km octave stands 12 m and the ground is buildable. The
transition is the macro slope's own gradient, so it is smooth and it follows the landform, not a line.

*In the game:* the pilot lands his hull on the flat ground beside the river, walks a kilometre, and
starts climbing. Nothing told the generator "here is a plain" — the plain is where the water already flattened
the macro field, and the roughness follows it.

#### 4.2.5 Whose edit is it?

The law belongs to this domain (erosion makes both the peak and the contrast). The code lives in
`crates/terrain/src/body.rs` and `height.rs`, which DOMAIN 02 and DOMAIN 04 also change. §12 D13 states it
as a joint decision, so the three documents do not each write a different spectrum.

### 4.3 Flow routing: D8, with an exact tie-break

For each node, choose the neighbour that maximises the downhill gradient `(z_i − z_j) / L_ij`, where
`L_ij` is the chord distance between the two nodes' surface points.

Never compare two floating-point ratios. Instead:

* hold `z` as an `i32` in 1/16 m;
* hold `L` as a `u32` in whole metres (macro distances are kilometres, so a metre is 0.01 % of one);
* compare candidate `j` against the best `k` by the cross-multiplied integers
  `(z_i − z_j) · L_ik  >  (z_i − z_k) · L_ij`, in `i64`;
* on an exact tie, keep the neighbour with the SMALLER global node index
  (`face · n_macro² + j · n_macro + i`).

That is a total order over integers. Two hosts cannot disagree, and the answer does not depend on the
order the neighbours are visited. A node with no lower neighbour has no receiver, and it is a pit or a
flat.

*In the game:* two nodes on the home planet's plateau sit at exactly the same height, to the sixteenth of
a metre. The tie-break sends the water the same way on the pilot's client and on the planet's shard, so
the river he sees is the river the collider holds.

### 4.4 Pits, lakes and FLATS: two surfaces, one water level, and no index stripes

* **`z_terrain`** — the ground. This is what the artifact keeps and what `height_m` reads.
* **`z_flood`** — the routing surface: `z_terrain` with every pit raised until water can leave. It exists
  only inside the solve and is never kept.

Run the priority flood on a copy of `z_terrain`:

1. Push every node at or below the sea level into a binary heap, keyed by `(z, node index)`. These are
   the outlets.
2. Pop the lowest. For each unvisited neighbour, set `z_flood = max(z_flood, z_popped)` and push it with
   its new key.
3. Repeat until the heap is empty.

**`ε` IS ZERO, and a flat is resolved by a DISTANCE FIELD.** Revision 2 raised each flooded node by
`ε = 1/16 m` so a receiver always existed. Refutation A computed the consequence and it is fatal on a
plain: 10 000 nodes of flat continental interior accumulate `10 000/16 = 625 m` of artificial gradient,
which dominates the real one, so the rivers of a continental plain would run in index-order stripes. The
cure is the standard one (Garbrecht and Martz 1997), and it is exact-integer:

* after the flood, a FLAT is a maximal set of connected nodes with equal `z_flood` and no lower
  neighbour outside it;
* run a breadth-first search inward from the flat's OUTLET nodes, giving each flat node an integer
  DISTANCE-TO-OUTLET in steps;
* inside a flat, the D8 receiver is the neighbour with the smaller distance, ties broken by the smaller
  global node index.

The BFS is integer, its frontier is processed in index order, and its result does not depend on the visit
order. No float, no epsilon, and the river on a plain runs toward the sea.

**The WATER LEVEL replaces the lake table.** Refutation A proved the artifact could not answer its own
water rule: three facies bits say a node is in *a* lake, and nothing says *which*, so nothing gives it a
spill level. Revision 3 keeps a WATER LEVEL PER NODE, an `i16` in metres from the ladder radius, in the
same encoding as `Z`:

```
   water_level(node) =  the lake's SPILL LEVEL   if the flood raised this node
                        sea_radius_m             if z_terrain ≤ sea_radius_m and the body holds water
                        DRY (a reserved value)   otherwise
```

The spill level of a lake is the `z_flood` of the node at which the lake overflows, and it is ONE level
for every node of that lake — flat water, as it must be. With `ε = 0` the flood already gives every node
of one lake the same `z_flood`, so the level falls out of the flood and needs no separate pass. The lake
TABLE is deleted; the per-node level is simpler, is exactly what §8.1 reads, and it is what makes the
sweep's base level correct (§4.6).

Cost: +2 bytes per node, `2 457 600 × 2 = 4.92 MB`, taking the artifact from 9.83 MB to **14.75 MB**.
That is refutation A's own arithmetic and it is accepted in full.

**Which surface each pass reads:**

| pass | reads | writes |
|---|---|---|
| D8 receivers (§4.3) | `z_flood`, then the flat distance field | the receiver byte |
| discharge (§4.5) | — | `Q` |
| stream power (§4.6) | `z_terrain`; the RECEIVER term reads `max(z_terrain(receiver), water_level(receiver))`; a node whose own `water_level > z_terrain` is INSIDE A LAKE and is SKIPPED | `z_terrain` |
| sediment (§4.11) | the removal per node | the per-basin deposit |
| flexural rebound (§4.7) | the removal `z_terrain(before) − z_terrain(after)` | `z_terrain` |
| talus (§4.8) | `z_terrain` | `z_terrain` |

**Skipping the lake nodes leaves the hole.** A lake node's receiver is higher than it is in `z_terrain`;
substituting that into the implicit update would pull the node UP every sweep and fill the lake with
rock. Skipping it leaves the hole exactly as the seed drew it, which is what a lake IS.

**A planet with no sea.** If no node lies at or below the sea level, there is no outlet. Seed the heap
instead with the `K` lowest nodes by `(z, index)`, `K` a stated world constant (recommend 6, one per
face). Rare, deterministic, and it keeps the solve total.

### 4.5 Discharge: accumulate the rain, not the nodes

Walk the D8 tree from the leaves to the outlets, in the order the flood's own stack gives, and add:

```
   Q_i  =  P_i · dA_i  +  Σ over the nodes whose receiver is i of Q
```

`P_i` is the precipitation at the node, in millimetres per year, from DOMAIN 02, **clamped below at
`P_MIN = 1 mm/yr`**, a stated world constant. Refutation B is right that nothing stopped `P = 0` over a
hyper-arid basin, and a logarithm cannot hold zero. `dA_i` is the node's true area in whole square metres
(§4.1). Both are integers, so `Q` is a `u64` in millimetre·square-metres per year, and the sum is exact
whatever order the tree is walked in. COMPUTED: the largest possible `Q` on the home planet is
`1.41 × 10^14 m² × 10 000 mm = 1.41 × 10^18`, against a `u64`'s `1.8 × 10^19`. It cannot overflow.

**ONE ceiling, stated once.** Refutation A and refutation B both found revision 2 using 10 000 mm/yr in
§4.5 and 1 000 mm/yr in §6.1 — a factor of ten apart, in the same document, with the byte's fit resting
on the smaller one. **The ceiling is `P_MAX = 10 000 mm/yr`** — Earth's wettest station, Mawsynram, is
about 11 900 mm/yr, so 10 000 is the honest number and 1 000 is not. §6.1 is re-sized against it.

**The unit conversion is stated once.** `Q_m3s = Q / (1 000 · 31 556 952)` — millimetres to metres, and
the seconds in a Julian year. Both divisors are world constants in the declared tag.

**The interface to DOMAIN 02.** This document needs one function:
`precipitation_mm_per_year(body_facts, site, height_m) -> u32`. It must be a pure function of the seed,
the address, the body's STATED facts (§5.2) and the CURRENT macro height, because orographic lift depends
on the topography the solve is changing. §4.12 states the schedule that closes that circle. **It must
also be `O(1)` per node**, or an upwind streamline integration turns a 2.46 M-node recompute into the
solve's dominant cost — §10 now carries a row for it and §14 M1 benches it.

**The risk, named.** DOMAIN 02 writes that a live wind field would be *"LIVE STATE and therefore a diff,
not a seed derivation"*. If the weather domain chooses a live wind, this document's input is not
derivable and the solve stops being a function of the seed. **The resolution this document asks for
(§12 D17):** the SHAPE reads a seed-derived CLIMATOLOGY — the long-run average rainfall — and live
weather modulates only the LOOK. A rainstorm must not move the ground.

**The fallback, if DOMAIN 02 is not ready.** Use `P = P_MIN` everywhere. Then `Q` is the drainage area
and the law degrades to the classic area form. The rivers are then right in shape and wrong in size: a
desert continent grows an Amazon. Recommend the fallback for the FIRST slice only, and say it out loud at
the picture review (§12 D5).

### 4.6 The stream power law, solved implicitly, with a lake's SPILL LEVEL as the base level

```
   dz/dt  =  U  −  K · Q^m · S^n            with  U = 0,  m = 1/2,  n = 1
```

With `n = 1` the equation is linear in `z`, and Braun and Willett's implicit form (2013) gives, in ONE
sweep from the outlets upward:

```
                z_i  +  Δt·K·sqrt(Q_i)/L_i · b_i
   z_i(new)  =  ─────────────────────────────────      where  b_i  is the BASE LEVEL at the receiver
                     1  +  Δt·K·sqrt(Q_i)/L_i
```

**`b_i` is the receiver's WATER LEVEL where the receiver stands under water, and its terrain height
otherwise.** This one line is refutation B's defect D3 and it is a real one. Revision 2 read
`z_terrain(receiver)`. At a lake shore the receiver is the first lake node, whose `z_terrain` is the LAKE
FLOOR — the hole the flood deliberately left. The sweep would then measure the slope down to the floor,
hundreds of metres over 8 224 m, and pull the inlet down toward it on every pass. *In the game:* the pilot
follows the river down to the mountain lake, and the last 8 km before the shore is a gorge the depth of
the lake, deeper after every pass. The rule is one line:

```
   b_i  =  max( z_terrain(receiver), water_level(receiver) )
```

so a river's base level at a lake is the lake's SPILL LEVEL, at the sea it is the sea level, and on dry
land it is the ground — the same expression in all three cases, with no branch on a landform kind (HR4).
The water level is in the artifact (§4.4), so the rule costs one load.

Three properties matter:

* **`m = 1/2` means `sqrt`.** `Gf::sqrt` exists (`crates/terrain/src/gf.rs:102`) and SL10 V1.4 grants
  square root by name. **Any DYADIC exponent is reachable under the fence** by repeated square root and
  multiply. `m = 1/2` is chosen because it is the cheapest point inside the literature's 0.4–0.6 band,
  not because it is the only lawful one.
* **It is unconditionally stable.** No time-step test, no inner loop, no data-dependent iteration count.
* **It is one division per node.** The whole sweep is `O(nodes)`.

**`Δt·K0` IS NOT A FREE CONSTANT ANY MORE.** Refutation A is right that revision 2 left one degree of
freedom and then promised to "measure" the other half of the same product, which measures nothing. The
fix is to give `Δt` a physical meaning:

```
   EROSIONAL_AGE_YR   a world constant: how long the water has worked since the mountains stopped rising
   Δt                 =  EROSIONAL_AGE_YR / PASSES
   K0                 an erodibility with published units, calibrated once against the hypsometric integral
```

Then `PASSES` is a NUMERICAL RESOLUTION, not a dial: raise it and the answer converges to the same
landscape, because `Δt` falls to match. The PHYSICAL dial is `EROSIONAL_AGE_YR`, it is measured against
Strahler's published mature range for the per-basin hypsometric integral (§12 D9), and it is recorded in
the world tag as one number with a unit. **§14 M12 is the convergence measurement that proves `PASSES` is
a resolution and not a dial:** solve the home planet at `PASSES` and at `2 · PASSES` with the same age,
and assert the fields agree within a stated bound.

**`K`, the erodibility, and where the owner's five words reach the shape.**

```
   K  =  K0 · f_rock(bedrock, sediment) · f_gravity(g/g_earth) · f_water(T_surface) · f_frost(day_length)
```

* **`f_rock` is a table over the SIX substances the code actually draws**: five bedrocks — Granite,
  Basalt, Gabbro, Andesite, Quartzite (`crates/terrain/src/strata.rs:145-151`) — and the one SEDIMENT the
  body drew of Sandstone, Limestone or Shale (`crates/terrain/src/body.rs:194`), which sits in the top
  20–80 m (`:198`). **Revision 2 said the world has no soft rock. Refutation B is right that it does.**
  The bedrocks span about a factor of two; the sediment is five to twenty times softer than any of them,
  which is exactly the contrast a wave-cut platform and a cap-rock bench need. The DEEP contrast — a
  1 km canyon wall with ledges all the way down — still needs a lithology field, and §12 D14 now asks only
  for that.
* **`f_gravity` is `g / g_earth`**, with `g = G·M/r²` from the mass and radius the forest draws
  (`crates/physics/src/taxonomy.rs:891-893`). A heavy world's rivers cut harder. *The owner's word:
  gravity, and size.*
* **`f_water` reads the SURFACE temperature, never `t_eq_k`.** `t_eq_k` is the EQUILIBRIUM temperature —
  what a body would have with NO atmosphere — and it is 255 K for Earth
  (`crates/physics/src/taxonomy.rs:684`). Reading it would give the home planet no rivers at all. The
  correct number is the grey-slab surface temperature, which depends on the body's INSOLATION and so on
  its distance from its star. *The owner's word: position.*

```
   T_surface_mean  =  t_eq_k · (1 + 0.75·τ)^(1/4)          and  x^(1/4) = sqrt(sqrt(x)), inside the fence
```

  COMPUTED for Earth: `288 / 255 = 1.1294`, so `(1 + 0.75τ) = 1.6266` and `τ = 0.835`.
  **`τ` IS THE ONE MISSING INPUT.** `Atmosphere` holds only the mean molecular weight, the scale height
  and an optional reference density (`crates/physics/src/taxonomy.rs:874-888`), and its own doc records
  `D-TAX-1`: *"surface pressure has no derivation"*. **Owed (§12 D6):** a per-body optical depth, drawn
  on the body's own seed stream inside a physical range and stated with the other facts. Until it lands,
  `f_water` is `0` where `atmosphere` is `None` and `1` otherwise — a stated PLACEHOLDER, named as one.
* **`f_frost` reads the DAY LENGTH.** Refutation B is right that revision 2 asked the owner to approve a
  spin fact and then never consumed it. Frost weathering is driven by how often the surface crosses
  freezing, and that is driven by the day-night temperature swing, which grows as the day lengthens. The
  term is one bounded function of the spin period, at most a factor of two, and it makes a slowly
  spinning world shed rock faster. *The owner's word: spin.* The fifth word, TRAJECTORY, reaches the
  shape through the eccentricity in §4.9's ELA.

### 4.7 Isostasy: a pyramid, not a box blur

After every `ISOSTASY_EVERY` sweeps, compute the thickness removed at each node, smooth it over the
flexural wavelength, and add back `ρ_crust / ρ_mantle` of it — about 0.83.

**The smoother is a pyramid on the macro lattice.** `n_macro = 640 = 2^7 · 5`, so it halves cleanly:
640 → 320 → 160 → 80 → 40 → 20. COMPUTED node sizes: 8 224 → 16 448 → 32 896 → 65 792 → 131 584 →
263 168 m.

1. **Restrict** the removal field up the pyramid by averaging each 2 × 2 block — an integer mean, exact
   and order-free.
2. **Smooth** at the level whose node size is nearest the flexural wavelength (COMPUTED: 131 584 m for a
   150 km wavelength, level 4, 9 600 nodes) with a few 8-neighbour diffusion passes built through the
   SEAM TABLE, so the smoother crosses a cube edge by construction.
3. **Prolong** back down with the same C1 interpolation §5.4 states for `Z`.
4. Add `0.83 ×` the result to `z_terrain`.

A separable box blur cannot do this job: COMPUTED, two 5-tap passes give `σ = 2` nodes = 16.4 km against
a 100–200 km flexural wavelength, six to twelve times too narrow; and a separable pass runs along a
face's `i` and `j` axes, which swap and sometimes reverse across a seam
(`crates/seed/src/seam.rs:191-203`), so it walks off the face.

Cost: the smoothing runs on 9 600 nodes, not 2.46 million. COMPUTED total: under 0.05 s per rebound.

**The flexural wavelength needs `T_e`, and `T_e` is missing.** `α = (4D / (Δρ·g))^(1/4)` with
`D = E·T_e³ / (12(1 − ν²))`. The fourth root is two square roots — inside the fence. `E`, `ν`, `ρ_crust`
and `ρ_mantle` are physical constants of the world, in the same class as the albedo bounds at
`crates/physics/src/worldgen/body.rs:84`, which the code itself calls *"not a tuning knob"*. `T_e` is
not held anywhere and must be drawn per body inside a physical range (5–100 km on Earth) and stated with
the other facts. §12 D6.

*In the game:* the home planet's shard cuts a 400 m valley into a range. The rebound lifts the whole
150 km around it by about 40 m, so the ridge beside the valley ends up TALLER than it started, and the
range keeps its snow line instead of wearing down into hills.

### 4.8 Talus: nothing stands steeper than its angle, and the mass is conserved

For `TALUS_PASSES` passes, and for each node in index order:

* compute the drop to each of the eight neighbours, and the EXCESS `e_j = drop_j − L_ij · tan(θ)` for
  each neighbour whose drop exceeds the talus limit;
* compute the node's TOTAL excess `E = Σ e_j` over those neighbours;
* move `TALUS_FRACTION · E` out of the node, **distributed among the lower neighbours in proportion to
  `e_j`**, with `TALUS_FRACTION ≤ 0.5`;
* accumulate the moves in a SECOND buffer and apply them all at the end of the pass, so the result never
  depends on the visiting order.

**The proportional split is refutation A's defect D10 and it is correct.** Revision 2 said *"move half
the excess to the lower neighbour"* inside a loop over eight neighbours, so a node with eight lower
neighbours shipped half of EACH excess and up to four times its own excess left the node in one pass.
Mass was not conserved and the pass could overshoot and oscillate. With the proportional split and
`TALUS_FRACTION ≤ 0.5` the scheme is an explicit non-linear diffusion whose per-pass change is bounded by
half the local excess, so it cannot overshoot; **§14 M13 measures the maximum slope after the pass and
asserts it falls monotonically**, which is the convergence statement revision 2 did not have. Convergence
is a different property from determinism, and revision 2's §5.1 row answered only the second.

`tan(θ)` is a CONSTANT, not a call: state the angle as its tangent directly — 0.70 for loose rock
(COMPUTED: `atan(0.70) = 35.0°`) and 1.40 for solid rock (`atan(1.40) = 54.5°`) — chosen by the bedrock
table. No trigonometric function enters the shape.

**What this pass can and cannot see.** Refutation B is right that an 8 224 m lattice cannot cap a 1.5 km
octave's slope. Under revision 3 it does not have to: the slope spectrum is ANCHORED at the angle of
repose (§4.2.3), so the fine octaves never exceed it in the first place. The macro talus pass caps only
what the macro solve itself creates — an over-steep 8 km wall left by the stream-power sweep or by the
glacial cut. That is a much smaller job, and it is one the lattice can see.

### 4.9 Ice: what the SOLVE decides, and what the FINE PASS draws

A U-valley is 1–3 km wide (0.1–0.4 macro nodes), a cirque 0.5–2 km, a fjord 1–5 km. **A cirque cannot be
carved by a lattice whose nodes are 8 224 m apart.** So the work splits.

**What the solve decides — four numbers per node, and nothing shaped:**

| number | how | why the macro lattice can decide it |
|---|---|---|
| the ICE MASK: was this node ever under ice? | the node's height is above the body's glacial ELA | an ice sheet's EXTENT is a 100 km fact |
| the ICE DEPTH proxy | grows with the height above the ELA and with the precipitation; a fixed number of passes moves it down the D8 receivers, so it pools in valleys and thins on ridges | a proxy, not a flow solve; the pooling is a 10 km fact |
| the TRUNK OVER-DEEPENING | where the ice is thick and the slope steep, `dz = −Kg · H_ice · S` on `z_terrain`, bounded by `MAX_GLACIAL_CUT_M` (§4.13) | it lowers the trunk valley's floor, which IS an 8 km fact |
| the GLACIAL DIRECTION | the D8 receiver at the moment the ice pass ran, kept in the receiver byte | it tells the fine pass which way the valley runs |

**What the fine pass draws — closed-form, at the rungs where a cell can hold it:**

* **The U cross-section.** Inside the ice mask, replace the channel term's V profile (§7.2) with a
  parabolic U whose width comes from the ice depth proxy and whose floor is flat.
* **The cirque.** At the head of a glaciated channel, subtract a hemispherical bowl whose radius comes
  from the ice depth. One subtraction, evaluated per column.
* **The arête.** Where two cirques or two U-valleys are close, what remains between them is a knife-edge
  by construction. Nothing draws it; it is what the two subtractions leave.
* **The hanging valley.** A tributary whose own ice was thin keeps its floor while the trunk's floor was
  cut. The step between them is a waterfall, and it falls out of the two depths.
* **The fjord.** No step at all. A glacial valley whose floor the over-deepening cut below
  `sea_radius_m` is flooded by the water rule of §8.

**THE SNOW ON THE GROUND, which revision 2 computed and never delivered.** Refutation A's defect D9 is
right and it is the single most visible thing in the reference picture. Revision 2 derived the ELA
carefully and then used it only to carve valleys, so a 6 000 m equatorial ridge on the home planet still
carried `Stratum::Gravel` and the white cap had to come from client style alone. Revision 3 states the
answer in two halves:

* **The SHAPE half.** The COLUMN FACIES (§2) takes the value `glacier` where the column stands above
  today's ELA, and the facies-to-stratum table gives `Stratum::Snow` there. That is ONE comparison per
  column against a number the solve already holds, it changes no record (§12 D8 keeps the facies
  derived), and it makes `strata.rs`'s biome-only snow rule
  (`crates/terrain/src/strata.rs:190-196`) a special case of a height rule instead of the only rule.
* **The LOOK half.** Ruling V4 puts the white on the client as style, keyed by that same facies. The
  client already reads the surface stratum.

§12 D18 puts the change to the owner, because it touches the shared strata table that DOMAIN 05 also
edits.

**The ELA, and the snow line, are two different heights.** The ELA — where accumulation balances melting
— sits ABOVE the 0 °C isotherm in a dry climate: COMPUTED from the published Andean case, 0 °C at
4 800 m and an ELA near 6 000 m, a dry-climate offset of about 1 200 m. So:

```
   T_surface(lat, h)  =  T_surface_mean · f_lat(lat, obliquity)  −  Γ_env · h
   Γ_env              =  Γ_dry · f_moist        with  Γ_dry = g / c_p,  c_p = (7/2)R / μ
   ELA(lat)           =  h(T_surface = 273 K)   +  Δ_dry · (1 − P / P_ref)
   ELA_range          =  ELA(lat)  ±  Δ_ecc · e         the seasonal swing, from the ORBIT
```

* `Γ_dry` IS derivable: `μ` is `mean_molecular_weight`, which `Atmosphere` holds. COMPUTED for Earth:
  `9.81 / 1005 = 9.76 K/km`.
* `f_moist` converts the dry adiabat to the environmental lapse rate. COMPUTED for Earth:
  `6.5 / 9.76 = 0.666`. It is a stated world constant.
* `Δ_dry` and `P_ref` are world constants pinned to the published Andean case.
* `f_lat(lat, obliquity)` needs obliquity, which is MISSING (§12 D6).
* **`Δ_ecc · e` is the owner's word TRAJECTORY.** The forest already draws `ecc` for every planet
  (`crates/physics/src/worldgen/generate.rs`). A high-eccentricity world has a bigger seasonal swing, so
  its glaciers reach lower in the cold season and its glacial ELA sits below its mean ELA. One multiply,
  one existing number, and refutation B's weakness W1 closes.

**The glacial epoch is a FACT, never a clock.** SL10 forbids a function of time. So the body does not
have a "last ice age"; it has a seed-drawn GLACIAL ELA — the lowest ELA the body ever had — stated as a
fact about the body. Everything above that line was carved by ice; everything below it was not. §12 D7
asks whether the body carries one such line or two.

**A body with no water and no ice.** `f_water = 0` and the ELA sits above the highest ground, so both
passes do nothing and the body keeps its raw octaves on the slope spectrum. A dead moon looks like a
dead moon, from the same code, with no branch on a body kind (HR4).

### 4.10 The coast: what the solve decides, and what the fine pass draws

The same split: a wave-cut platform is 10–500 m wide, three orders of magnitude below the macro node.

**The solve decides one thing per node:** the COASTAL BAND flag — is this node within `h_wave` of the sea
level? `h_wave` derives from the body's gravity and its atmosphere's reference density (bigger waves on a
low-gravity world under a thick atmosphere), never from a literal.

**What survives at macro scale is the COASTLINE'S PLAN SHAPE**, because it is the intersection of `Z`
with the sea level, and it is what a pilot sees from orbit. Nothing else coastal does.

**The fine pass draws three things, closed-form, per column:**

* **The platform.** Inside the coastal band, pull the height toward the sea level by a factor that falls
  with the substance's hardness. **The soft sediment gives the wide flat shelf**, and where the sediment
  has already been stripped the hard bedrock gives a narrow one with a cliff above it. That contrast
  exists TODAY (§4.6 `f_rock`), which revision 2 wrongly denied.
* **The beach.** Where the platform is wide and the slope gentle, the COLUMN FACIES is `beach` and the
  stratum is `Sand`; where it is narrow and steep, bare bedrock. One comparison per column.
* **The delta.** At a channel node whose receiver lies below the sea level, the channel ENDS. The fine
  pass fans it over a radius that grows with the basin's DEPOSITED VOLUME (§4.11, not with `sqrt(Q)` out
  of nothing), raises the ground to just above the sea level inside the fan, and sets the facies to
  `delta`. COMPUTED from §6.1's table: a delta radius of 1–50 km, so the LARGEST deltas are the only
  coastal landform the macro lattice resolves on its own.

### 4.11 The sediment budget: one number per basin (NEW in revision 3)

**Refutation A's weakness W3 is right and it is the largest physical omission of revision 2.** The stream
power law as written is DETACHMENT-LIMITED: rock is removed and vanishes. Yet the design draws a delta, a
floodplain and a freeboard, each of which is made of sediment. Volume was not conserved anywhere, and the
depositional half of the hypsometric curve did not exist.

A full transport-limited solve is a second field, a second sweep and a second stability question. This
design does not buy that. It buys the cheapest thing that makes the deposits real:

```
   per node:   removed_i  =  z_terrain(before) − z_terrain(after)   (already computed for isostasy, §4.7)
   per basin:  V_basin    =  Σ over the basin's nodes of  removed_i · dA_i        a u64, in m³
   at the outlet:  the delta's VOLUME is a stated share of V_basin
   at a RANGE FRONT: an ALLUVIAL FAN takes a stated share, where a channel's slope falls by more than
                     FAN_SLOPE_DROP between two consecutive nodes
```

* The sum is over integers and is exact, and it reuses the removal field the isostatic pass already
  builds — so it costs one `u64` add per node and one number per basin.
* **The alluvial fan is now named and built.** Refutation A is right that it is the landform that most
  often carries a settlement at the foot of a range like the reference picture's, and that revision 2 did
  not mention it once. It is one comparison per channel node and one fan radius, in the same shape as the
  delta's.
* **The floodplain's freeboard stops being a bare constant.** It scales with the basin's deposited volume
  per unit of channel length, so a big muddy river builds a wide high floodplain and a small clear
  mountain stream does not.
* **What this still does NOT do:** it does not fill the sea, it does not move sediment between basins,
  and it does not make the hypsometric curve's depositional half. §14 M6 measures the LAND curve only,
  and §13 open question 12 names the gap.

### 4.12 The schedule, the circle between rain and mountains, and the world's EROSIONAL AGE

```
   Δt = EROSIONAL_AGE_YR / PASSES
   for pass in 0 .. PASSES:                       (every count is a world constant)
       if pass % CLIMATE_EVERY  == 0:  recompute the precipitation field (DOMAIN 02, O(1) per node)
       if pass % FLOOD_EVERY    == 0:  priority flood -> z_flood, water levels; flat resolve; D8 receivers
       accumulate the discharge
       one implicit stream-power sweep on z_terrain, skipping lake nodes, base level from §4.6
       accumulate the per-basin sediment (§4.11)
       if pass % ISOSTASY_EVERY == 0:  the flexural rebound on the pyramid
   TALUS_PASSES relaxation passes
   the ice pass  (mask, depth proxy, trunk over-deepening)
   the coastal band mark
   THE ENVELOPE ASSERTION  (§4.13)
```

Every count is a constant. Nothing depends on a convergence test, so nothing depends on a float
comparison. Recommended starting values, all OPEN to the owner and all part of the world tag:
`PASSES = 40`, `CLIMATE_EVERY = 10`, `FLOOD_EVERY = 10`, `ISOSTASY_EVERY = 5`, `TALUS_PASSES = 8`, and
`EROSIONAL_AGE_YR` measured by §12 D9.

### 4.13 The envelope: why the band does not move, and the FULL downward budget

**The rule.** `Z` REPLACES the coarse octaves; it does not add to them.

```
   h(site, rung)  =  radius  +  Z(site)  +  Σ over the FINE octaves kept at this rung  −  C  +  T
```

Write `A_fine` for the sum of the FINE octaves' amplitudes under the new spectrum (COMPUTED 1 532 m on
the home planet, §4.2.3). The envelope is then:

```
   THE ENVELOPE ASSERTION:   |Z_i|  ≤  relief_m − A_fine     at every node
   ⇒  |h − radius|  ≤  (relief_m − A_fine) + A_fine  =  relief_m   — exactly today's envelope
```

COMPUTED on the home planet: the cap on `|Z|` is `14 305 − 1 532 = 12 773 m`, against a coarse-octave sum
of 13 617 m. So the cap can bind, and when it does, scale every `Z` uniformly by
`(relief_m − A_fine) / max|Z|` — a single `Gf` division applied uniformly with one stated rounding.
Uniform scaling preserves the SHAPE and only lowers the relief slightly. It exists so the envelope is a
proof rather than a hope, and so the isostatic rebound — which genuinely raises un-eroded nodes above
their initial height — cannot break the address.

**THE DOWNWARD BUDGET, stated in full.** Refutation A's note N9 and refutation B's weakness W3 both say
the accounting was incomplete, and it was: revision 2 spent the crust's 64 m entirely on the channel and
never budgeted the glacial cut or the detail's downward half. The full budget:

| term | budget | why |
|---|---|---|
| `MAX_CHANNEL_DEPTH_M` | **32 m** | COMPUTED from §6.1: an Amazon-sized channel is 32.0 m deep, so the clamp binds only on an extremely wet world |
| `MAX_GLACIAL_CUT_M` | **16 m** | the trunk over-deepening (§4.9) |
| `MAX_DETAIL_CUT_M` | **16 m** | the downward half of the scree and bench rules (§7.3) |
| **total** | **64 m** | exactly the spare 64 m at `crates/terrain/src/body.rs:234` |

**And the 64 m is not spent to zero without asking.** Refutation B's weakness W3 is right that a design
that consumes an undocumented margin owes the owner a sentence about what the margin was for. The code
names the purpose of the 64 m ABOVE (*"room to stand on the highest peak"*) and not of the 64 m BELOW.
The likeliest purpose is the extractor's: `below_surface_cell` (`crates/terrain/src/chunk.rs:225-231`)
fills a chunk under the surface, and the mesh needs a rock cell below the deepest cave for the surface to
close. **§12 D16 puts the question to the owner rather than assuming the answer**, and it names the
cheap alternative: `crust_m` adds the strata depth AND the cave depth while only the deeper of the two is
needed below the surface (COMPUTED, a further 23–92 m of unused crust on every body), so a reader who
wants more room can take it there without moving `floor_m`.

**What stays deleted.** Growing `crust_m` lowers `floor_m` (`crates/seed/src/ladder.rs:95`), and every
cell's radial index `k` is measured from `floor_m` (`:132-134`). Growing the crust by 64 shifts EVERY `k`
on EVERY body by 64 cells at rung 0. That proposal is not revived.

### 4.14 What comes out

| Output | Type | Size on the home planet (COMPUTED at 2 457 600 nodes) | Read by |
|---|---|---|---|
| `Z`, the eroded macro height | `i16`, metres from the ladder radius | 4.92 MB | every rung's `height_m`; the roughness field |
| the WATER LEVEL | `i16`, the same encoding, or a DRY sentinel | 4.92 MB | the water rule, the sweep's base level |
| the D8 receiver (3 bits), the NODE FACIES (3 bits), the ice mask and the coastal band (1 bit each) | 1 byte | 2.46 MB | the river polyline, the gully warp, the art-asset skeleton |
| the discharge | 1 byte, quantised logarithm | 2.46 MB | the channel width and depth |
| **total** | | **14.75 MB** | |
| a coarse PYRAMID of `Z` (levels 320, 160, 80, 40, 20) | `i16` | +1.64 MB; **top level 4.8 KB**, level 40 is 19.2 KB | the far view |

COMPUTED for the pyramid, `6 · n² · 2` bytes: 1 228 800 + 307 200 + 76 800 + 19 200 + 4 800 = 1 636 800 B
= 1.64 MB. **Revision 2 called 19 KB the top level; that is level 40, and the stated top of 20 is
4.8 KB.** Both refuters caught it and the corrected number is used everywhere.

---

## 5. Determinism, concretely

SL10 clause 4 says the server and every client compute the same bytes on x86-64 and on aarch64. The macro
solve is the hardest thing in the generator to make deterministic, because it holds a queue, a graph and
an iteration. Seven rules make it safe.

### 5.1 Integers between passes — and EVERY float named

The macro height lives as an `i32` in 1/16 m. A planet's relief is at most 18 000 m
(`crates/terrain/src/body.rs:147-151`), which is 288 000 steps — four decimal orders of headroom.

| pass | float work | how it stays deterministic |
|---|---|---|
| node area (§4.1) | `W'`, one divide, one `sqrt` | floored to a whole m², summed as `u64` |
| chord length `L` (§4.3) | one `sqrt` | floored to whole metres, then integer cross-multiplication |
| flat resolve (§4.4) | none | an integer breadth-first search |
| stream power (§4.6) | one `sqrt`, one divide | written back through `Gf::to_i64_floor` (`crates/terrain/src/gf.rs:96`) |
| sediment (§4.11) | none | `u64` adds over floored integers |
| flexural rebound (§4.7) | `× 0.83`, the pyramid's mean, the prolongation | integer mean on restriction; one floor on write-back |
| talus (§4.8) | `× tan(θ)`, the proportional split | a second buffer, one floor on apply |
| ice (§4.9) | the depth proxy, `Kg·H·S`, the trunk cut | one floor on write-back |
| coast (§4.10) | `h_wave` comparison only | the flag is a comparison |
| envelope (§4.13) | one divide, applied uniformly | one stated rounding, then an integer assertion |

Every one is add, subtract, multiply, divide, `sqrt` and the rounding family — the operations
`crates/seed/src/bend.rs:9-14` grants by name — under `Gf`, with ONE evaluation order. And after every
pass the state is exactly integer, so no float ever survives from one pass into the next.

### 5.2 The body's physical facts are DRAWN ONCE, STORED and STATED — and the artifact is DERIVED

This section is rebuilt twice over: refutation A found the ruling that revision 2's answer breaks, and
refutation B found that revision 2 moved the drift onto the server instead of removing it.

**Ruling S7-2 forbids the ship.** *"The client states both halves at login, computed on the home planet's
literals; a client that cannot compute them does not log in"*
(`docs/design/owner_decisions_2026-09-07_voxels.md:498`). The code implements exactly that:
`WorldIdentity::of` (`crates/terrain/src/tag.rs:43-50`) folds `golden_self_check(home)`, which calls
`generate(body, key)` on eight home-planet chunks (`crates/terrain/src/digest.rs:22-26`). Under revision
2's recommendation the chain did not close:

```
   login  ─►  the client must generate 8 home-planet chunks
          ─►  those chunks hold Z
          ─►  Z comes from the artifact
          ─►  the artifact arrives from the realm AFTER login          ✗
```

**And 14.75 MB has no lane.** `SELF_LOOK_BUDGET_BYTES = 1200` (`crates/core/src/look.rs:76-79`), *"one
conservative datagram"*. The artifact is twelve thousand times the whole self-look bag. Refutation B is
right that D3 asked the owner to approve a new bulk transport — chunked, retained, resumable, invalidated
by the world tag, on the reach path, for every body a player approaches — in the same breath as 28 bytes
on an existing field. And it is right that shipping static shape reverses the reason SL10 exists.

**The answer, in three parts.**

1. **The FACTS are stated, on the lane that already exists.** The realm states 28 bytes beside
   `SurfaceStmt` under `TAG_SURFACE` — once per realm, on change, retained, one hop
   (`crates/core/src/look.rs:52-79`; the test at `crates/sim/src/stub/tests/window_lane.rs:2563`). That is
   one SL6 ask, and §12 D3 puts it.
2. **The ARTIFACT is DERIVED, on whichever host needs it.** The solve is a function of the seed, the
   address and those 28 bytes, so both hosts can run it and SL10 holds unchanged. Nothing static crosses
   the wire.
3. **The HOME PLANET's artifact is a COMMITTED BUILD ARTIFACT.** It is pinned by a digest and linked into
   every server and every client, in exactly the shape `crates/terrain/src/home.rs:16-22` already uses for
   the home planet's seed and radius bits, and pinned by a test the way `home_body_pin.rs` pins those. Then
   the world hello of S7-2 works unchanged, a player who logs in standing on the home planet has the
   ground under his boots before the first packet, and the 12–40 s derive never blocks a login.

| fact | unit | bytes | source |
|---|---|---|---|
| surface gravity | 1/1024 m/s² | 4 | `G·M/r²` from `crates/physics/src/taxonomy.rs:891-909` |
| mean surface temperature | 1/16 K | 4 | `t_eq_k · (1 + 0.75τ)^(1/4)` (§4.6) |
| atmospheric optical depth `τ` | 1/4096 | 4 | MISSING — §12 D6 |
| mean molecular weight | 1/256 | 2 | `Atmosphere::mean_molecular_weight` |
| obliquity | 1/4096 turn | 2 | MISSING — §12 D6 |
| spin period | 1 s | 4 | MISSING — §12 D6 |
| effective elastic thickness `T_e` | 1 m | 4 | MISSING — §12 D6 |
| the glacial ELA offset, and the eccentricity | 1 m, 1/4096 | 4 | drawn per body (§4.9); `ecc` already exists |
| **total** | | **28 bytes** | inside the 1 200-byte budget with room to spare |

**THE FACTS ARE STORED, NOT RECOMPUTED. This is refutation B's blocker B5 and it is correct.** The
argument revision 2 made against a client recomputing a quantised transcendental does not depend on the
host being a client. `crates/physics/src/taxonomy.rs:21-24` records the deferred cross-host bit-equality
gate for exactly this arithmetic, and the project runs shards on an M4 host AND on x86-64 pods and
reschedules a realm between them as a shipped feature. So:

* the realm DRAWS its facts ONCE, at the moment the realm is first created, and WRITES them to its own
  durable store — the same store that already remembers what people built;
* every later boot READS them; no boot recomputes them;
* the home planet's facts are additionally PINNED as literals in the `home.rs` shape, with a test that the
  forest still draws exactly those numbers — the pattern `crates/bins/tests/home_body_pin.rs` already
  uses;
* the world identity digest FOLDS the stated facts, so a mismatch is DETECTED as well as prevented.

*In the game:* the home planet's realm is spun down by the demand loop at midnight and boots on a
different pod at dawn. It reads its gravity, its optical depth and its elastic thickness back out of its
own store, solves the same erosion, and the house a player built at midnight is standing on the same
ground.

**A quantum is not a snap.** The `home.rs` test proves a look radius moved by a thousand ulps gives the
same body because the LADDER SNAPS the radius to an integer edge count, with a stated tolerance of one
top-rung cell edge — COMPUTED 1 304 m against a drift of about `10^-9` m, a margin of `10^12`. A surface
temperature quantised to 1/16 K has NO such margin. Revision 2 called the snap the proven pattern for
quantising an astrophysical fact; it is not, and revision 3 does not rely on it. Storing the drawn value
is what removes the question.

### 5.3 Every order is a stated order; the solve is SINGLE-THREADED and OFF THE TICK

* the priority queue orders by `(height, global node index)` — a total order on integers;
* the D8 receiver breaks ties by the smaller global node index, and a flat by the smaller distance then
  the smaller index;
* the accumulation walks the flood's own stack, which was built in index order;
* the talus pass writes to a second buffer and applies it at the end of a pass;
* the pyramid's restriction is an integer mean of a 2 × 2 block, which is order-free;
* the seam table is compile-time generated and digest-pinned (`crates/seed/src/seam.rs:206-215`);
* **the solve runs on ONE thread.** A 12–40 s figure is exactly the number that makes somebody reach for
  `rayon`, and a parallel priority flood destroys byte identity. A parallel decomposition may land ONLY
  if M4 measures byte identity with it enabled, on both targets, over a full body.
* **the solve does NOT run on the shard's tick thread.** This is refutation B's defect D11 and it is
  correct and important. Every shard is a `bevy_ecs` tick loop, and a 40 s blocking call is not a slow
  boot — it is a realm that stops answering the lane, which the peer book reads as a lost connection and
  which the gateway's readiness machinery has already tripped on once in this project. So:
  * the solve runs on the injected worker pool ruling S7-6 already provides (*"an injected trait: inline
    in Tier-A tests, `std::thread::spawn` + `crossbeam-channel` in the binary"*), never inline on the
    tick;
  * until the artifact lands, **the shard REFUSES a chunk request with a stated "not ready" reason and a
    counter** — it never answers one from a partial field. A refusal is a fact the client can act on; a
    half-solved chunk is a shape that would be replaced under the player's feet;
  * the tick at which the artifact becomes available is therefore a function of wall-clock scheduling.
    **That does not touch determinism:** the artifact's BYTES do not depend on when the work finished.

### 5.4 The macro field's interpolation and its seam, rebuilt on the true lattice

**What revision 2 got wrong.** It claimed *"node planes sit ON the cube edges"* and built an ownership
rule, a shared-value rule and a symmetry argument on top. Refutation A proved the premise false:
`face_param` is a cell centre in the OPEN interval (`crates/seed/src/ladder.rs:142-145`), the outermost
node sits half a node inside the edge, and `across()` returns an ADJACENT CELL on the partner face
(`crates/seed/src/seam.rs:183-203`, the partner index is `n_l − 1` or `0`), never a shared one. There is
no shared node to own.

**The true geometry, and why it is better.** Unfold the two faces about their common edge and write the
face parameter as one continuous coordinate `a`. The last cell centres of this face are at
`… , 1 − 5/n, 1 − 3/n, 1 − 1/n`; the partner's first cell centres continue at
`1 + 1/n, 1 + 3/n, 1 + 5/n, …`. **The samples straddle the edge symmetrically, at a uniform spacing of
`2/n`, with the cube edge exactly at the midpoint of the central interval.** So:

```
   face +X                    │ cube edge                  face +Y
   ──●────────●────────●──────┼──────●────────●────────●──────
   1−5/n    1−3/n    1−1/n    1     1+1/n    1+3/n    1+5/n
                     └──────── one uniform Catmull-Rom stencil ────────┘
```

**The rule for `Z`, stated:**

1. **The lattice is CELL-CENTRED everywhere.** No node lies on a cube edge, and no node is shared. There
   is no ownership rule, which deletes two of revision 2's four rules.
2. **The interpolation is Catmull-Rom in the face parameter**, with the parameter step converted to
   METRES by the local bend scale `W'(a)/|p|` that §4.1 already computes. The 4-wide stencil is gathered
   THROUGH the seam table (`crates/seed/src/seam.rs:183`), so it reaches across a cube edge, and some
   seams reverse the along-edge direction, so the gather never adds one to `i`.
3. **C1 across a cube edge follows from three facts.** The stencil is UNIFORMLY spaced across the edge
   (shown above). `W'` is EVEN in `a`, so `W'(1)` is the same approached from either face — COMPUTED
   `W'(1) = K1 + 3K2 + 5K3 = 1.5584` from `crates/seed/src/bend.rs:23-30`. And the along-edge parameter
   of the two faces' cell centres coincides. So a Catmull-Rom expressed in metres has the same value and
   the same derivative from both sides. **That is an argument, so it gets a MEASUREMENT.**
4. **AT A CUBE CORNER the stencil is not defined, and the rule says so.** This is refutation B's defect
   D8 and it is correct: only three faces meet at a corner, so a 4 × 4 stencil for a node one or two
   nodes from the corner reaches into a quadrant that does not exist. **The rule:** within
   `CORNER_BLEND = 2` nodes of a cube corner, use the QUINTIC-fade interpolation the crate already holds
   (`crates/terrain/src/noise.rs:66-75`) over the single 2 × 2 node cell the point lies in — which needs
   no stencil beyond the cell — and blend from it into Catmull-Rom over the next node with the same
   quintic weight, so the join is C1 from both sides. COMPUTED cost of the compromise: eight corners,
   each covering about four macro nodes, is about 2 200 km² of 141.09 million km² — 0.0016 % of the
   surface, on ground a player can walk but is very unlikely to notice.

**Why a scarp would be fatal.** COMPUTED, each cube edge on the home planet is
`2π · 3 350 759 / 4 = 5 264 km`, twelve of them, **63 000 km of cliff**, with `Z`'s own range in the
thousands of metres. SL8 calls one seam a defect.

**Three gates, of the `G-MAPPING-TABLE` kind that ruling V6 A1 already requires:**

* **G-MACRO-EDGE.** Walk every one of the twelve cube edges at a stated sample spacing. Assert that `Z`
  agrees across the edge to a stated tolerance, and that the surface SLOPE agrees to within a stated
  fraction of a rung-0 cell per cell.
* **G-MACRO-CORNER.** At each of the eight cube corners assert BOTH facts: the node has exactly seven D8
  neighbours and its receiver is one of them, AND the INTERPOLATED `Z` is continuous and C1 across each
  of the three faces' boundaries inside the blend radius. Revision 2's gate tested only the first.
* **G-MACRO-SEAM.** Solve a small real body of THE world, then solve it again with the faces visited in
  reverse order, and compare every byte.

**The named fallback.** If G-MACRO-EDGE measures a slope jump above the bound, drop the whole field to
the quintic fade, whose derivative vanishes at every node plane, so the across-edge slope is zero from
both sides and C1 holds by construction — at the cost of a mild 8 km ripple aligned to the lattice, which
§14 M11 would then measure against the eleven-seam taxonomy's "detail-by-box" row.

### 5.5 The solve's constants are part of the world identity

Change one and every stored edit on every planet sits on ground that moved. The full list, for the
DECLARED half of the world tag (`docs/investigation/2026-09-07/03_generator_sl10.md` §3.3), beside the
bend constants:

| group | constants |
|---|---|
| the lattice | `MACRO_CELL_TARGET_M`, `MIN_MACRO_EDGE`, the divisor rule |
| the spectrum (§4.2) | `o_peak`'s derivation, `σ`, `TALUS_RMS`, the anchor's talus tangent, the `Σ a ≤ relief` rule |
| the roughness field (§4.2.4) | `SLOPE_REF`, `m_min`, the smoothstep's form |
| the octave selectors | `MACRO_SAMPLES_PER_WAVE` (the solve's, 4), `LADDER_SAMPLES_PER_WAVE` (the ladder's, 4) |
| the solve | `EROSIONAL_AGE_YR`, `K0`, `m`, `n`, `P_MIN`, `P_MAX`, `PASSES`, `CLIMATE_EVERY`, `FLOOD_EVERY`, `ISOSTASY_EVERY`, `TALUS_PASSES`, `TALUS_FRACTION` |
| the no-sea case | the outlet count `K` |
| rock and slope | the six-substance erodibility table, the two talus tangents, `f_frost`'s bounds |
| isostasy | `ρ_crust/ρ_mantle`, `E`, `ν`, the pyramid level rule, the diffusion pass count |
| climate | `f_moist`, `Δ_dry`, `P_ref`, `Δ_ecc`, the 0.75 in the grey slab, the seconds-per-year and mm-per-m divisors |
| ice | `Kg`, the depth-proxy pass count, the cirque radius law, the U-width law, `MAX_GLACIAL_CUT_M` |
| coast | `h_wave`'s derivation, the platform factor, the beach thresholds |
| sediment (§4.11) | the delta's share, the fan's share, `FAN_SLOPE_DROP`, the freeboard law |
| rivers | `a_w`, `a_d`, the gravity exponents, the **committed literal** `Q^0.4` table, the log-discharge floor, ceiling and span |
| meanders | `SUBDIV`, the meander wavelength's multiple of the width, the per-level salts |
| the fine pass | the bank width `B`, the floodplain coefficient, the facies thresholds, the scree, bench and ridged-noise blends |
| the envelope | `MAX_CHANNEL_DEPTH_M`, `MAX_GLACIAL_CUT_M`, `MAX_DETAIL_CUT_M`, the renormalisation's rounding |
| the seam | `CORNER_BLEND` |
| **the body facts** | **the drawn ranges for `τ`, obliquity, spin, `T_e` and the glacial ELA offset — refutation B is right that revision 2 left these off the list while making the shape depend on them** |

**The `Q^0.4` table must be a COMMITTED LITERAL.** If it is computed at build time with `powf`, it is a
transcendental outside the fence and two toolchains can differ. It is 256 numbers, committed as source
and pinned by the world tag, exactly as the seam table is pinned by a digest.

### 5.6 The band, and the address, do not move

Stated in full in §4.13. In one line: the eroded field REPLACES the coarse octaves inside the same
envelope, the assertion `|Z| ≤ relief_m − A_fine` makes that a proof, every downward term fits the 64 m
budget the crust term already carries, and `floor_m`, `band_m` and `n` are all byte-identical to today.

---

## 6. The river graph

### 6.1 What it holds, and what it costs

A river is not stored as a list of rivers. It IS the D8 tree: node → receiver. Per macro node:

* **the eroded height `Z`**, an `i16` in metres from the ladder radius;
* **the WATER LEVEL**, an `i16` in the same encoding, or a reserved DRY value (§4.4);
* **the receiver direction**, 3 bits; **the NODE FACIES**, 3 bits (upland, floodplain, delta, beach,
  scree, glacier, lake bed, sea bed); **the ice mask** and **the coastal band**, 1 bit each — those eight
  bits pack into ONE byte;
* **the discharge**, a separate byte, as a quantised logarithm.

So the artifact is `2 + 2 + 1 + 1 = 6` bytes per node. COMPUTED: `2 457 600 × 6 = 14 745 600 B =
14.75 MB`. Revision 2 said 4 bytes and 9.83 MB and could not answer its own water rule; refutation A's
arithmetic is accepted in full.

**The log-discharge byte, re-sized against ONE ceiling.** Both refuters found revision 2 quoting two
ceilings a factor of ten apart and resting the byte's fit on the smaller. COMPUTED with §4.5's stated
`P_MAX = 10 000 mm/yr` and the MINIMUM node area (a corner node, §4.1):

```
   floor    =  P_MIN · min_node_area  =  1 mm × 4.75 × 10^7 m²  =  4.75 × 10^7
   ceiling  =  P_MAX · total_area     =  10 000 mm × 1.411 × 10^14 m²  =  1.41 × 10^18
   span     =  ceiling / floor        =  2.97 × 10^10  =  2^34.8      ⇒ DECLARE 36 octaves
```

So 256 steps at `2^(36/256)` per step. COMPUTED half-step error: `2^(18/256) − 1 = 4.9 %` on `Q`, which
is **2.5 % on a width** (a width goes as `sqrt Q`) — under the 4.43 % revision 2 quoted for `Q` and well
inside a metre on a 40 m river. The floor, the ceiling and the 36-octave span all ride the world tag so
the fit is a stated fact and not a coincidence.

**Width and depth are not stored.** They are one closed-form step from the discharge, by the hydraulic
geometry of Leopold and Maddock (1953):

```
   width_m  =  a_w · sqrt(Q_m3s) · (g / g_earth)
   depth_m  =  a_d · Q_m3s^0.4   · (g_earth / g)
```

`a_w = 3.9` and `a_d = 0.4` are EARTH CALIBRATIONS — that is what they are, and they ride the world tag
by name (§5.5). The gravity factors are NOT calibrations: a river's bed shear is `ρ·g·D·S`, so at a fixed
critical shear the depth goes as `1/g`; the mean velocity goes as `sqrt(g·D·S)`, which is then
independent of `g`; and continuity `Q = w·D·U` gives the width as `Q·g`. **A low-gravity world has deep,
narrow rivers**, from physics, with no new constant.

`Q^0.4` needs a `powf`, which the fence forbids. **The cure is already in the format:** the discharge is
a LOG BYTE, so a 256-entry COMMITTED LITERAL table indexed by that byte gives `Q^0.4` to within a
quantisation step, at the cost of one array read (§12 D4, §5.5).

COMPUTED against Earth's hydraulic geometry at a runoff of 0.3 m per year and `g = g_earth`:

| drainage area | discharge | width | depth |
|---|---|---|---|
| 100 km² | 1.0 m³/s | 3.8 m | 0.4 m |
| 1 000 km² | 9.5 m³/s | 12.0 m | 1.0 m |
| 10 000 km² | 95 m³/s | 38 m | 2.5 m |
| 100 000 km² | 951 m³/s | 120 m | 6.2 m |
| 1 000 000 km² | 9 506 m³/s | 380 m | 15.6 m |
| 6 000 000 km² (an Amazon) | 57 034 m³/s | 931 m | 32.0 m |

**The drawn-watercourse threshold stays DELETED.** COMPUTED: one channel link per node on a lattice of
spacing `s` gives a drainage density of `1/s`; at `s = 8.224 km` that is **0.122 km of channel per km²**,
against a literature range for real terrain of 0.5–5 km/km². Drawing EVERY node is already four to forty
times BELOW Earth. A channel head on Earth sits at a support area of about 0.1–1 km², and one macro node
is 57.4 km² at the MEAN (§4.1), so every node is far past it.

**AND MOST OF THE VISIBLE VALLEYS WOULD BE DRY. This is refutation B's defect D6 and revision 2 gave the
number without the consequence.** COMPUTED: a 50 km vista spans about six macro nodes, so it holds about
six MACRO channel links; the valley density the eye reads from the slope spectrum is 0.33–0.67 km/km².
**Between a half and four fifths of the valleys a player can see would carry no water at all.** On Earth
a mountainside with twenty valleys has twenty streams; a dry valley is a rare and specific landform. The
owner's reference picture has water as the thing the eye follows.

**The cure: a SUB-MACRO STREAM, closed-form, with no discharge.** In a sub-macro valley — a place where
the gully warp's flow field converges and the local relief is a trough — draw a stream of a stated small
width, whose water level follows the trough's own floor and whose base level is the macro channel it
drains into. It is not a hydrological object: it carries no `Q`, joins no graph and is stored nowhere. It
is the visual and physical completion of a valley that the eye already reads as a valley. Two closed-form
terms per column, gated to rungs where a 2–10 m channel is representable, and it takes the DRAWN drainage
density from 0.122 to the same 0.33–0.67 km/km² the valleys already have. §12 D19 puts it to the owner
because it is new work; §14 M9 measures the density it produces.

*In the game:* the pilot on the ridge sees ONE macro river and, between him and it, twenty side valleys
the closed-form pass drew, all running the same way — and now a thread of water in each of them.

### 6.2 The polyline lives in the body's own frame

Each channel node has a 3-D point: `dir(face, i, j, n_macro) · (radius + Z + Σ fine octaves)`. A channel
segment is the pair (node, receiver). So a river is a chain of 3-D points in the body's own metres, and a
face seam is invisible to it — a segment that crosses a seam is just a segment between two points. This
is exactly why the graph must NOT be stored in face coordinates.

The carve is then `segment_distance_m(start, end, p)` — the function at
`crates/terrain/src/carve.rs:160`, unchanged, already tested exactly on the ends, the middle and a
degenerate segment (`crates/terrain/src/carve.rs:187-217`).

### 6.3 A river must not be a chain of 8 km straight lines

At the fine rungs an 8 224 m straight segment with a kink at each end reads as a canal. Subdivide each
segment deterministically, by midpoint displacement:

* for `SUBDIV` levels (recommend 4, giving 514 m segments), replace each segment by two;
* offset the new midpoint sideways, in the local tangent plane, by
  `amplitude · unit_value(child_seed(body.seed, SALT_MEANDER, key))` where
  **`key` folds the two node ids AND the level AND the child index** — `min(id_a, id_b)`,
  `max(id_a, id_b)`, `level`, `child`;
* the `min` and `max` make the offset the same whichever end asks, so two chunks on either side of a
  segment carve the same channel;
* `amplitude` halves at each level and starts at a multiple of the channel width — a real river's meander
  wavelength is about eleven times its width, so the constant comes from a physical fact.

**The band it serves, stated honestly and CORRECTED.** Revision 2 said 47–745 m of width. Refutation A is
right that the lower bound is wrong: at `w = 47 m` the meander wavelength is `11 × 47 = 517 m`, which is
ONE 514 m sub-segment, and one sample cannot draw a wavelength. Two samples per wavelength needs
`w ≥ 94 m`; **four samples, which is the least that reads as a curve rather than a zigzag, needs
`w ≥ 187 m`.** The upper bound is right: `11 × 745 = 8 195 m`, one macro segment. **The served band is
about 190–745 m of width — a factor of four, not sixteen.** So:

* SMALL streams get their sinuosity from the sub-macro valley structure (§6.1, §7.3 rule 5), not from
  this subdivision;
* the TRUNK's large meanders are a macro-scale fact the design does not hold — §13 open question 9.

That costs 16 extra points per macro segment, computed on demand, and it is closed-form.

### 6.4 The per-chunk lookup, with a prefilter

**A rung-0 chunk is 62 m across and a macro node is 8 224 m, so a chunk lies inside at most TWO macro
nodes per axis.** COMPUTED: `8 224 / 62 = 132.6`, so a chunk edge is not commensurate with a node and a
chunk near a node boundary straddles it. Revision 2 said a chunk *"sits inside ONE macro node"*, which is
false; refutation A is right, and the 5 × 5 gather already covers the consequence.

* the SAMPLE BOX gathers the channel segments of the 5 × 5 macro nodes around the chunk ONCE — at most 25
  macro segments before the width filter, typically none to six after it;
* each surviving macro segment is subdivided into 16 sub-segments, and the sub-segments are then
  PREFILTERED against the box's bounding sphere before any per-column distance runs;
* the per-column work is then a distance to a handful of sub-segments, exactly like the tube prefilter
  `crates/terrain/src/lattice.rs:290-310` already runs;
* segments that cross into a partner face are found, because the 5 × 5 neighbourhood is taken through the
  seam table and the polyline is 3-D anyway.

---

## 7. The fine rungs: the pass that reads a table and solves nothing

**One word, one meaning: "TABLE-DRIVEN", not "closed-form".** Refutation A's note N7 is right that
revision 2 used "closed-form" for a rule that reads a 14.75 MB table. This section's pass performs a
bounded number of arithmetic operations and a bounded number of table reads per column, with no
iteration, no queue and no dependence on a neighbouring chunk's array. That is the property that matters,
and this document calls it TABLE-DRIVEN from here on.

### 7.1 The new height equation, and THE LADDER'S NEW COARSENING RULE

```
   h(site, rung)  =  radius
                  +  Z(site)                 the eroded macro field, REPLACING the coarse octaves
                  +  Σ over the FINE octaves KEPT AT THIS RUNG of  A(o)·m(site)·noise(dir·f)
                  −  C(site, rung)           the channel and floodplain   (drops by width)
                  +  T(site, rung)           the detail                   (drops by cell size)
```

**`Z` exists at EVERY rung and is identical at every rung.** The macro node is 8 224 m, which is wider
than the coarsest ladder rung (2 048 m on the home planet). So `Z` can never cause a rung disagreement,
and it contributes exactly ZERO to `dropped_bound_m`. That deserves a test, and it is true — **and it is
not the question.**

**THE QUESTION IS WHAT THE OCTAVES DO, AND BOTH REFUTERS ANSWERED IT.** `octaves_at(rung)` keeps the
coarsest `octave_count − rung` octaves — a rule by COUNT
(`crates/terrain/src/body.rs:251-258`) — and `dropped_bound_m` is the sum of the dropped amplitudes
(`:274-276`). The slope spectrum moves amplitude INTO the octaves the count rule throws away first.
COMPUTED on the home planet under §4.2.3's spectrum:

| rung | cell | dropped bound TODAY | under the spectrum, COUNT rule | **under the spectrum, WAVELENGTH rule** |
|---|---|---|---|---|
| 4 | 16 m | 6.9 m | 11.5 m | **0.32 m** |
| 5 | 32 m | 15.2 m | 36.6 m | **1.19 m** |
| 6 | 64 m | 32.8 m | **116.4 m** | **3.71 m** |
| 7 | 128 m | 70.3 m | **315.2 m** | **11.5 m** |
| 8 | 256 m | 150.6 m | **634.6 m** | **36.6 m** |
| 10 | 1 024 m | 687.9 m | 1 532.4 m | 315.3 m |

**And the gate cannot catch it, which is the worse half.** `crates/terrain/src/height.rs:98-101` asserts
`(hl − h0).abs() <= dropped_bound_m(rung)`, and `dropped_bound_m` is ITSELF the sum of the dropped
amplitudes. Grow the amplitudes and the bound grows with them: **the test stays green while the picture
pops.** Refutation A is right that revision 2 named no gate that could fail on this, and right that
`docs/investigation/2026-09-07/02_smooth_terrain.md:477` already estimates the arrival pop at 4 px and
calls it *"visible"*.

**THE RULE, and it is a change to the ladder, not to this domain alone.** Coarsen BY WAVELENGTH:

```
   octaves_at(rung)  keeps every octave with   λ(o)  ≥  LADDER_SAMPLES_PER_WAVE · cell_m(rung)
                     never fewer than one           LADDER_SAMPLES_PER_WAVE = 4 (recommended)
```

* **It fixes today's defect too.** At rung 6 the cell is 64 m and the count rule throws away a 1 562 m
  octave — an octave a 64 m cell represents with twenty-four samples. The wavelength rule keeps it.
* **The dropped bound falls by an order of magnitude at every mid rung**, as the table shows. **At the
  vista's own rung** — COMPUTED, one pixel subtends 57.5 m at 50 km with 720 rows at 45°, so rung 5 or 6
  — the disagreement is 1.2 m or 3.7 m against a 57.5 m pixel. It is SUB-PIXEL, which is what SL8 needs.
* **Rung `L` is still cheaper than rung 0.** COMPUTED, the kept count on the home planet runs
  14, 14, 14, 14, 13, 12, 11, 10, 9, 8, 7, 6 from rung 0 to rung 11 — monotone non-increasing and never
  above rung 0's 14. Ruling V8/V9's "a coarser rung is cheaper" holds. The cost at rung 6 rises from 8
  octaves to 11, which is 11/14 of a rung-0 column.
* **It bumps `GENERATOR_VERSION`** (`crates/terrain/src/tag.rs:19`), because it moves bytes at every rung
  above 3. That is expected: so does the whole slice.

**`C` and `T` drop by their own rule**, the same discipline `crates/terrain/src/carve.rs:36-45` uses for
caves: carve a channel only where `cell_m ≤ width/2`, and add a detail term only where the cell is no
wider than a quarter of the term's own wavelength. Their contribution to `dropped_bound_m` is the deepest
channel dropped at that rung plus the detail's amplitude — bounded by §4.13's 32 m and 16 m.

**The bound's test cannot be the existing one.** `crates/terrain/src/height.rs:80-105` walks 400
directions on a 20 × 20 parameter grid per face. A drawn channel covers of the order of a thousandth of
the surface, so 400 spaced samples land in one with near-zero probability. §14 M10 adds a fixture that
samples ALONG a known channel of the home planet, and §14 M8 measures the OCTAVE pop as well as the
river pop — which revision 2's M8 did not.

### 7.2 The channel and the floodplain

For a point at distance `d` from the nearest channel sub-segment, with that segment's water surface
`z_w`, its width `w` and its depth `D`:

```
   ^ height
   │      raw surface
   │  ‾‾‾\                                         /‾‾‾‾
   │      \___                                 ___/         the floodplain: blended flat
   │          \______   z_w + freeboard   ____/
   │                 \_______________ ___/                  the banks
   │                        \______/                        the bed, at z_w − D
   └──────────────────────────────────────────────► distance from the centreline
        |<-- w -->|<--- bank --->|<---- floodplain ---->|
```

* inside the channel (`d ≤ w`): `h = z_w − D · (1 − (d/w)²)`, a parabolic bed — or a flat-floored U where
  the ice mask is set (§4.9);
* over the bank (`w < d ≤ B`): blend from `z_w` to the raw surface with the quintic smoothstep
  (`crates/terrain/src/noise.rs:66-75`), so the bank carries no crease;
* over the floodplain (`B < d ≤ F`): blend the raw surface toward `z_w + freeboard`, with `F` and the
  freeboard growing with the basin's DEPOSITED VOLUME (§4.11) — the flat fields of the reference picture,
  and the only ground a settlement can stand on without terracing.

Every step is add, multiply, divide and the quintic fade. No new primitive.

**The descent gate, in four statements.** A lake's surface is flat, so a naive *"every receiver is
strictly lower"* assertion goes red on the first lake.

1. On `z_flood`, every node's receiver is lower or equal. That is what the priority flood guarantees.
2. Inside a FLAT, the receiver's distance-to-outlet is strictly smaller (§4.4). That is what makes the
   D8 graph a tree even where the heights tie.
3. A channel node INSIDE a lake takes the lake's WATER LEVEL as its `z_w`, so `z_w` is CONSTANT across a
   lake and the lake's outlet is the node where the descent resumes.
4. Outside a lake, `z_w` is strictly falling along every polyline. Assert it over every channel node of
   the home planet.

*In the game:* the pilot follows the river upstream into a mountain lake. The water is level all the way
across the lake, the far shore is where the river resumes, and the shard's assertion says so on every
build.

### 7.3 Detail with no solve at all

The macro solve gives structure, the spectrum gives amplitude and the roughness field gives contrast.
Five table-driven rules give the ORGANISATION, and each costs a few operations per column.

1. **The slope is address-defined, and `height_m` gains the ADDRESS.** `height_m(body, dir, rung)` is a
   pure function of ONE direction (`crates/terrain/src/height.rs:17-27`). A slope taken from a finite
   difference over a CHUNK's own array is a function of the STENCIL, so two chunks sharing a boundary
   would compute different heights for the same column — a crack, and a shape that is not a function of
   (seed, address). **The rule:** the slope at a column is the central difference over the FOUR
   NEIGHBOURING CELLS OF THAT COLUMN'S OWN CELL AT THAT RUNG, an address-defined stencil, of the RAW
   surface (`radius + Z + fine octaves`, before `T`), so the term is not circular.

   **The signature change, stated.** Refutation B's defect D7 is right that the design owed this sentence
   and revision 2 never wrote it. A direction cannot name a cell without inverting the bend, so the
   entry point becomes `height_m(body, site, rung)` with `Site { face, i, j }`
   (`crates/terrain/src/lattice.rs:59-63`) and the direction computed inside from `face_param`.
   **Refutation B's COST estimate does not apply, because no caller has to invert anything.** Every
   production caller already holds the site: the column pass builds it (`crates/terrain/src/chunk.rs:181-183`
   inside `column_field`), the halo columns carry it in `SampleBox::sites` through `site_of`
   (`crates/terrain/src/lattice.rs:124-140,246-280`), and the digest's golden chunks build their
   directions from a face and parameters (`crates/terrain/src/digest.rs:125,147`). `vertex_position_m`
   does not call `height_m` at all (`crates/terrain/src/position.rs:34-40`, it interpolates the box's own
   column surfaces). So the change is a SIGNATURE and four call sites, not 0.2–0.4 ms of `unbend` per
   chunk. It also states the law more strongly than before: SL10 says a function of the seed and the
   ADDRESS, and the address is what the signature now carries.

   **The cost, stated honestly.** `ColumnField` holds 62 × 62 columns and no halo
   (`crates/terrain/src/chunk.rs:92-107`); the halo lives in `SampleBox` and its columns' heights are
   already extra `height_m` calls. So a CORE column's slope is free from the one-cell halo the box
   already pays for, and a HALO column's is not. The column ring must grow from one cell to two —
   COMPUTED: 64² = 4 096 columns to 66² = 4 356, **+6.3 % of the box's columns**. **That is a
   RESTRUCTURE, not a line item**, and refutation A's weakness W1 is right to say so: `HALO = 1`
   (`crates/terrain/src/lattice.rs:48-49`) sizes `BOX_EDGE = CHUNK_EDGE + 2` and therefore `BOX_CELLS`,
   and `SampleBox::column_index` indexes the column arrays by that same `BOX_EDGE`. Two ring depths means
   two index functions and a changed `sites`/`dirs`/`surfaces` layout that `extract` also reads. **§12
   D16 makes it a decision**, with the cheaper alternative named: keep one ring and take the halo
   column's slope from a one-sided difference, which is address-defined too and costs nothing, at the
   price of a slightly different scree amplitude in the outermost ring.

2. **Scree by slope, AND A REAL CLIFF — the claim about rock pillars is WITHDRAWN.** Both refuters are
   right, and this is the most honest correction in revision 3. Revision 2 wrote *"where the slope
   exceeds the talus tangent, subtract a term that pulls the surface toward the local mean … That is the
   reference picture's rock pillars and their debris."* **A rock pillar, a tor or a hoodoo is by
   definition a column of rock that stands STEEPER than the talus angle and does not collapse.** A rule
   that relaxes everything above the talus tangent toward the local mean removes every pillar in the
   world by construction. Revision 2 claimed a landform from the rule that erases it.

   **What the design actually has, and what it now builds:**

   * **The scree rule stands, as a scree rule.** Where the address-defined slope exceeds the talus
     tangent, pull the surface toward the local mean and set the COLUMN FACIES to `scree` (the stratum
     table then gives `Gravel`). That makes a soft debris foot under a steep face. It makes no cliff.
   * **THE CAP-ROCK BENCH is the cliff mechanism, and it is buildable TODAY.** Refutation B's defect D9
     is right that the world already draws a SOFT sediment — Sandstone, Limestone or Shale
     (`crates/terrain/src/body.rs:194`) — 20–80 m thick (`:198`) over a hard bedrock. Where the surface
     cuts through that boundary, the soft layer wears back and the hard layer stands out. The rule is one
     comparison per column against a depth the strata table already holds: inside the sediment, pull the
     surface back by a stated amount that grows with the local slope; at and below the boundary, do not.
     The result is a flat bench with a face of up to 80 m along a whole hillside at ONE height — which is
     what a real cap-rock escarpment looks like, and what the reference picture's near cliff band is. It
     costs no new field, no new draw and no ask of another domain.
   * **A free-standing PILLAR still needs a JOINT field** — a seed-derived pattern of vertical fractures
     that erosion follows — and the world does not draw one. It is NOT claimed here. §12 D14 asks for it
     with the deep lithology, and §13 open question 11 names it as a missing landform.
   * **A DEEP cliff, hundreds of metres tall, still needs the deep lithology.** Below 92 m the world is
     one bedrock, so a 1 km wall is smooth below its cap-rock bench.

3. **Ridged noise on ridges.** Where the slope is high, add `A · (1 − |noise|)` instead of `A · noise`.
   Ridged noise makes sharp crests and smooth troughs — the shape a weathered arête has. Blend the two by
   the slope, so no line appears where the rule changes. Cost: one extra noise evaluation per column
   (12.19 ns MEASURED, `scripts/noisebench`).

4. **Strath terraces — from the CHANNEL, not from the strata.** COMPUTED from
   `crates/terrain/src/body.rs:196-201`: the topsoil is 1–4 m, the subsoil 2–8 m and the sediment 20–80 m,
   so the deepest stratum boundary sits 23–92 m below the surface and everything under it is ONE bedrock.
   A 1 km canyon wall would carry ledges in its top 92 m and nothing below. **The replacement:** quantise
   the height a little toward a ladder of levels derived from the channel's depth and the local relief,
   so a deep valley carries a step at each of a few stated fractions of its depth. That is what a strath
   terrace IS. Its downward cut is inside `MAX_DETAIL_CUT_M` (§4.13).

5. **Flow-warped gullies — the cheapest large win.** The artifact holds the D8 direction. Reconstruct the
   flow VECTOR at a column by interpolating the neighbouring nodes' unit flow vectors and normalising,
   then add a ridged noise whose coordinates are stretched ALONG that direction and compressed across it.
   The result is a dendritic gully field that runs the right way down the hill. It is also the field
   §6.1's sub-macro stream follows, so the visible valleys carry water.

Rule 5, the slope spectrum (§4.2.3) and the roughness field (§4.2.4) are the three changes that answer
the owner's complaint. All three should land in the first slice.

### 7.4 What each term costs

**The baseline is UNMEASURED under the new recipe, and that is the honest statement.** Refutation B's
defect D1 is right: the MEASURED 6.1 ms cave-dense chunk (`crates/bins/examples/terrain_cost.rs:38-41`)
was measured on TODAY'S recipe, and §4.2 replaces that recipe. Two mechanisms move it, both from §4.2's
own numbers:

* **more chunks hold the surface.** The fine-octave RMS slope rises to 0.70 in an orogen, so the surface
  crosses more chunk columns per column of ground and the `AboveSurface`/`BelowSurface` skips
  (`crates/terrain/src/chunk.rs:216-247`) fire on fewer chunks. **The roughness field bounds this**: only
  the rough fraction of the body pays it, and a plain at `m_min = 0.06` pays less than today. What
  fraction of the surface is rough is UNMEASURED until M9.
* **the extractor works harder inside a rough chunk.** The same file states a MEASURED synthetic worst
  case of 26 ms for a 3-D checkerboard (`:43-45`), called *"a BOUND no chunk of THE world reaches"*.
  Roughening the world moves THE world toward that bound by an unknown amount.

**So §14 M2 must bench the four cases AFTER §4.2 lands, not before, and this table is an ESTIMATE of the
ADDED cost only:**

| Term | Where it runs | ESTIMATED added cost |
|---|---|---|
| `Z` and the roughness field | column pass; one 4 × 4 node gather, then one Catmull-Rom per column | **0.07–0.15 ms** |
| the two-cell column ring for the slope (§7.3 rule 1) | column pass; +6.3 % of the box's columns | 0.10 ms |
| `C`, the channel | column pass; up to six macro segments × 16 sub-segments, prefiltered against the box sphere, then `segment_distance_m` per column | 0.25–0.75 ms |
| the water level per column | column pass; one comparison against the node water level and the channel level | < 0.05 ms |
| `T`: scree, bench, ridged crests, terraces, gullies | column pass; three extra noise evaluations and a few multiplies | 0.20–0.35 ms |
| the sub-macro stream (§6.1) | column pass; two terms where the flow field converges | 0.05–0.15 ms |
| **the WATER SURFACE mesh (§8.3)** | **a SECOND full extractor run over a derived field, only for chunks the water range crosses** | **2–3 ms** |
| **total, a chunk with a river and no water surface** | | **+0.67 to +1.50 ms** |
| **total, a chunk with a river AND a water surface** | | **+2.67 to +4.50 ms** |

**The water pass is re-costed, and it is the number that decides D15.** Refutation A's defect D4 is
right: revision 2 charged a SECOND full run of the same extractor over the same box at 0.3–1.0 ms,
against a MEASURED 6.1 ms for a chunk in which ONE such extraction is one of two halves. A second run
cannot cost a fifth of a run that already includes one. ESTIMATED honestly: if the extraction half of
6.1 ms is 2–3 ms, the second run is 2–3 ms.

COMPUTED headroom against the OLD baseline, for scale only: `6.1 + 1.50 = 7.6 ms` of 8 ms for a
cave-dense chunk with a river; `6.1 + 4.50 = 10.6 ms` for one that also meshes a water surface —
**over budget by 33 %, not by 3 %.** `SUBDIV` is worth perhaps 0.2 ms and cannot buy 2 ms. **So §12 D15
changes its options:** the dial that can pay is the water mesh itself, not `SUBDIV`.

---

## 8. Water

### 8.1 Three surfaces, one rule — and a body that has none

```
   water_level(column) = the GREATEST of
        sea_radius_m                          if the body can hold liquid water at all
        the node WATER LEVEL under this column  (a lake's spill level, or the sea)   §4.4
        z_w along the nearest channel         if the column is inside its channel or bank
        otherwise: none — the column is dry
```

**The first line is new.** Today `fluid_at` fills every radius under `sea_radius_m` with `Stratum::Water`
for ANY body (`crates/terrain/src/chunk.rs:205-213`), and `sea_radius_m` is drawn for every body
(`crates/terrain/src/body.rs:186-190`), so an airless moon in THE world has an ocean. The sea exists only
where `f_water > 0` (§4.6).

**The second line is what the per-node water level bought.** The column reads ONE `i16` from the
artifact, interpolated the same way `Z` is. Nothing has to find which lake it is in.

### 8.2 The column pass gains one number

`ColumnField` (`crates/terrain/src/chunk.rs:92-107`) gains a fourth per-column entry: the water radius,
or a sentinel for dry. Then `fluid_at` changes from `fluid_at(body, r)` to `fluid_at(body, column, r)`.

**The skip rules still work, with one correction.** The `AboveSurface` skip fills a chunk by a per-LAYER
rule (`crates/terrain/src/chunk.rs:216-247`), which assumes every column of the chunk holds the same
fluid at a given radius. With rivers that is false. The fix is one extra test: take the skip only when
the chunk lies entirely above the column field's HIGHEST water level, or entirely below its lowest.
Rivers are thin, so almost every above-chunk still skips.

### 8.3 The water surface must be meshed, CLIPPED, and BUDGETED

Today nothing meshes the top of the water: `is_rock(gap)` is `gap < 0`
(`crates/terrain/src/extract.rs:71-75`) and water sits on the air side, so the extractor walks past it.
The seabed IS meshed; the sea's top is not.

**The rule:** run the SAME extractor a second time over the signed field

```
   d(cell)  =  min( water_level(column) − r ,  r − rock_surface(column) )
```

quantised by the same `quantise_gap` (`crates/terrain/src/chunk.rs:118-127`). That is the INTERSECTION of
"below the water" and "above the rock", so the water body ends exactly where the rock rises through it.
Emit only the faces whose ACTIVE term is the water level, so the lake bed — already meshed by the rock
pass — is not meshed twice. Nothing in the cell record changes; the water field is derived, never stored.

A field that is only `r − water_level` would change sign over EVERY column that has a water level,
whether or not rock stands above it, and would put a vertical wall of water at the bank. The shoreline is
exactly where a viewer's eye goes.

**What it costs beyond worker time.** Refutation B's weakness W6 is right that revision 2 charged this
only in milliseconds. A second mesh is **a second vertex buffer, a second index buffer, a second material
and a second draw call per water-crossing chunk**, and ruling V10's MEASURED client figure is 405 KB of
geometry per chunk. Coastal and lake chunks are exactly the chunks a player looks at. §14 M14 measures
the water mesh's vertex count, its byte size and its draw-call count on a coastal chunk of the home
planet, and §12 D15 now names the water mesh — not `SUBDIV` — as the dial that can pay for the budget.

The alternatives are worse. A single analytic sphere cannot hold a lake or a river. A bespoke water
mesher is a second implementation of a thing we already have, which HR3 refuses.

### 8.4 What is postponed, by name

* **Water does not flow after an edit.** A player who digs a trench from a river gets a dry trench. A
  player who digs under a lake does not drain it. This is live state on a fine grid, it is a whole
  subsystem, and the owner already postponed cross-realm edits (ruling V4).
* **The macro solve never runs again.** Erosion is a fact of the world, not a process inside it.
* **There are no tides, no seasons and no floods in the shape.** Each is a function of TIME, which SL10
  forbids in the static shape.
* **SEDIMENT TRANSPORT proper.** §4.11 gives one number per basin, which makes a delta and a fan real.
  It does not move sediment between basins, does not fill the sea and does not make the depositional half
  of the hypsometric curve.
* **AEOLIAN landforms.** No dunes, no yardangs, no ventifacts. The code already has `Biome::Desert` and a
  `Sand` topsoil (`crates/terrain/src/strata.rs:192`), and the owner's task says *"we also should simulate
  the weather"*. A dune field is a table-driven field oriented by the prevailing wind, so it is a natural
  later slice on top of DOMAIN 02's wind.
* **KARST.** No sinkholes, no dolines, no dry limestone valleys. The world draws `Limestone` as one of
  three sediments (`crates/terrain/src/body.rs:194`) and already has caves, so karst is nearly free
  later: a sinkhole is a cavern that reached the surface.
* **VOLCANIC LANDFORMS.** No cones, no calderas, no lava plains, no volcanic necks, no volcanic plateaus.
  Refutation A's weakness W4 is right that this omission was not even named:
  `crates/terrain/src/strata.rs:136-142` draws Basalt, Gabbro and Andesite as bedrocks, **so the world
  already asserts a volcanic history it never shows.** It is the second landform family after fluvial on
  an earth-like planet, and the reference picture's rock pillars are as likely volcanic necks as fluvial
  remnants. The owner should know before the picture review.
* **Mass wasting other than talus.** No landslide scars, no rock-avalanche deposits.
* **A JOINT FIELD, and therefore free-standing pillars and tors.** §7.3 rule 2.

### 8.5 What a live weather layer would need from this artifact

Nobody owns weather today. This document holds the water, so it must state the interface even though it
does not build it.

A live weather layer on a planet's shard would read, from the artifact: **the node WATER LEVEL, the
channel DISCHARGE and the NODE FACIES — all three of which the artifact keeps.** It would also want the
ELA per node and the precipitation climatology, and **revision 3 states honestly that the artifact does
NOT keep those two**: §10's memory table throws the precipitation away with the rest of the solve state,
and the ELA is a function the solve evaluates rather than a field it stores. Refutation A's blocker B5
caught revision 2 offering four fields of which two did not exist. The weather layer's options are to
recompute the ELA from the 28 stated facts (it is a closed-form function of latitude, height and the
climatology, so this is cheap) or to ask for two more bytes per node — which is an SL6 ask nobody has
made and this document does not make on their behalf.

What weather ADDS — a storm, a cloud, a river running high — is a LOOK, and it must never move a cell.
**This document's ask (§12 D17):** state that the shape reads the CLIMATOLOGY and never the weather, so a
rainstorm can never re-address a chunk.

---

## 9. The ladder: how a river reaches the far view

The reach ruling makes a planet drawable from very far away, and SL8 makes any pop a defect. Five rules
keep a river continuous from orbit to the boot.

1. **`Z` is rung-independent.** One field at 8 224 m, read at every rung. So the valleys the solve cut are
   in the shape at every distance. That is what makes a planet read as a planet from orbit: continents
   with drainage, not a noise ball.
2. **The pyramid answers the far view.** Levels 320 down to 20 (§4.14). A level is chosen so its node is
   no wider than a stated fraction of a pixel at the drawing distance. COMPUTED, the top level is 4.8 KB
   — not the 19 KB revision 2 stated, which is level 40.
3. **The octaves coarsen BY WAVELENGTH (§7.1).** This is the rule that makes the whole design safe on the
   ladder, and revision 2 did not have it. COMPUTED: 1.2 m of rung-to-rung disagreement at the vista's
   own rung, against a 57.5 m pixel.
4. **`C` and `T` fade at a stated width and cell size**, exactly as the cave tubes already do
   (`crates/terrain/src/carve.rs:36-45`), and their contribution to the bound is budgeted in §4.13.
5. **THERE IS NO CROSSFADE TO LEAN ON, and revision 2 leaned on one.** Refutation A is right:
   `grep -rn "crossfade" crates/` finds it in no source file, and ruling V10 and
   `docs/investigation/2026-09-07/00_proposed_voxel_foundation.md:719-721` put the crossfade in **slice
   8**, which is not built. So every rule above must hold on its own, with no blend to hide behind, and
   §14 M8 measures it. `docs/investigation/2026-09-07/03_generator_sl10.md` §11 must grow a river case
   and an OCTAVE case.

---

## 10. Cost and memory

**Per body, once (the macro solve).** ESTIMATED from a per-operation model, NOT measured.

| Step | Work at `n_macro = 640` | ESTIMATED time, one thread |
|---|---|---|
| D8 receivers | 2.46 M nodes × 8 neighbours through the seam table, integer | 0.1 s |
| one priority flood | 2.46 M pushes and pops, 21 heap levels, a 20 MB working set larger than L2 | **1.0–3.2 s** |
| the flat resolve | one breadth-first search over the flat nodes, integer | 0.05–0.2 s |
| the stack and the accumulation | 2 × 2.46 M, integer | 0.1 s |
| one stream-power sweep | 2.46 M, one `sqrt`, one divide, two random loads | **0.15–0.25 s** |
| the sediment accumulation | 2.46 M `u64` adds, sequential | < 0.05 s |
| **one climate recompute (DOMAIN 02)** | **2.46 M evaluations, IF the field is `O(1)` per node** | **0.1–0.5 s, UNMEASURED** |
| one flexural rebound (pyramid) | restrict, 9 600-node smooth, prolong | < 0.05 s |
| 40 sweeps, 4 floods, 4 climate recomputes, 8 rebounds | | **11–26 s** |
| talus (8 passes), ice (about 20 passes), coast, the envelope | fixed pass counts, integer | 1–3 s |
| **the whole solve** | | **ESTIMATED 12–30 s, and plausibly up to 45 s** |

**Two rows are new because two refuters asked for them.** Refutation A's weakness W5 is right that
revision 2's table had no row for the climate recompute at all, and that an upwind streamline integration
would not be `O(nodes)` and could dominate the headline. **§4.5 now REQUIRES `O(1)` per node from
DOMAIN 02 by name, and §14 M1 benches it.** Refutation B's weakness W5 is right that the flood's lower
bound was optimistic: at five to eight cache misses per heap operation and about 80 ns each, the honest
model is nearer 1.3 µs per node and 3.2 s per flood, so the range widens upward rather than staying at
1.0–1.5 s.

**M1 MUST RUN BEFORE §12 D3 AND D11 ARE PUT TO THE OWNER.** Every number above is a model.

**Memory during the solve.** COMPUTED from what the algorithm holds at once: `z_terrain` (4) +
`z_flood` (4) + water level (2) + receiver (1) + flat distance (4) + area (8) + `Q` (8) +
precipitation (2) + ice (4) + the talus second buffer (4) = 41 B per node = **100.8 MB**, plus the heap
(up to 2.46 M × 8 B = 19.7 MB) and the flood's stack (2.46 M × 4 B = 9.8 MB): **130 MB**.

**Memory kept.** 14.75 MB for the artifact, plus 1.64 MB for the pyramid (§4.14).

**Where the solve runs, and what it costs the player.**

* **On a shard**, the solve runs ONCE EVER per body, on the injected worker pool and never on the tick
  thread (§5.3), cached in the shard's store keyed by `(world tag, body seed)`. It must not re-run when
  the demand loop spins the realm down and up. Until it lands, the shard REFUSES chunk requests with a
  stated reason and a counter.
* **On the client**, the solve runs on the same injected worker seam ruling S7-6 gives the chunk workers.
  A client that flies toward a new planet derives that planet's artifact while it approaches.
* **The HOME PLANET is never solved at run time on either host.** Its artifact is a committed,
  digest-pinned build artifact (§5.2), so the world hello of S7-2 works, and a player who logs in
  standing on it has the ground under his boots at once.

**What the ship would have cost, for the record.** Refutation B's blocker B4 asked for a number revision 2
never computed. COMPUTED: a system with eight planets and twenty moons is 28 bodies; at 14.75 MB each
that is **413 MB for a player who tours one system**, on a lane that does not exist, with retention,
resumption and world-tag invalidation to design. **That is why revision 3 derives instead of ships.** The
derive costs the client 12–45 s of one background thread per body it approaches, and nothing on the wire.

**The reach-to-arrival time is UNMEASURED, and it is the number that says whether the derive is fast
enough.** Turning a distance into minutes needs a closing speed; the right source is the suit's and the
hull's rated cruise (the 2026-09-05 ruling). §14 M15 computes it, and §13 open question 13 names what
happens if it is shorter than the derive.

**Per chunk (the fine pass).** ESTIMATED +0.67 to +4.50 ms (§7.4) on a baseline that is UNMEASURED under
the new recipe. §14 M2 settles both halves.

---

## 11. Law gates, one by one

A gate row states what it is gating and whether the gate has been run. Where it has not, the row says
UNMEASURED and names the measurement. Refutation A's defect D5 and refutation B's defect D10 both caught
revision 2 asserting a pass on an unmeasured premise; the HR5 row is rewritten because of them.

| Law | How this design passes |
|---|---|
| **SL10 — one generator, two hosts, no drift** | The solve is a function of `(seed, address, the realm's STATED facts)` and of nothing else. No host recomputes an astrophysical transcendental: the facts are DRAWN once, STORED, and stated (§5.2). Its state is integer between passes; every float is named in §5.1; its ordering is total over integers; it runs on one thread and off the tick. **UNMEASURED until M4**, the byte-for-byte cross-target gate extended to fold the artifact AND the stated facts. |
| **SL10 — the shape is DERIVED, not shipped** | **Revision 3 restores this, which revision 2 reversed.** The artifact crosses no wire. The client derives it from the seed, the address and 28 bytes; the home planet's is a committed build artifact so ruling S7-2's login still works. |
| **SL10 — the client only renders** | The solve derives SHAPE. It derives no pose, no velocity and no entity state. A player's dug trench stays a diff from the owning realm. |
| **Ruling S7-2 — the world hello** | The client generates eight home-planet chunks at login (`crates/terrain/src/tag.rs:43-50`, `crates/terrain/src/digest.rs:22-26`). Those chunks need `Z`, so the home planet's artifact is linked into the client build and pinned by a digest and a test, in the `crates/terrain/src/home.rs:16-22` shape (§5.2, §12 D3). **Revision 2 broke this ruling without citing it.** |
| **SL6 — ask before new data crosses** | ONE ask: 28 bytes of body facts, on the statement that already exists, retained, on change, one hop (`crates/core/src/look.rs:52-79`). §12 D3. Revision 2's second ask — a 14.75 MB bulk lane — is withdrawn. |
| **The seed ruling — a public map may not be a treasure map** | The graph decides shape and BULK facies only: sand, gravel, clay, a beach, a delta. It decides no ore, no placer and no deposit. **And the FLOODPLAIN MAP is examined by name, not skipped:** a seed-derived map of the only flat buildable ground IS public, and the 2026-08-27 ruling permits it — *"an expensive gate buys ROOM, not treasure … which is live state"*. Who has already taken a valley is live state; where the valley is, is not. Refutation A's note N8 is right that a reader asks about this one first, so it is answered here rather than only for ore. |
| **SL5 — one world** | One solve, one set of constants, the home planet as the fixture, real small bodies of THE world for the seam and coverage fixtures. No test-only planet, no reduced lattice, no scale knob. |
| **No magic numbers** | `K` derives from the substance table, the gravity, the surface temperature and the day length. `Γ` derives from gravity and the molecular weight. `h_wave` derives from gravity and the atmosphere. The talus tangent comes from the substance. **`s_peak` derives from the talus tangent (§4.2.3) — it is no longer set by a normalisation, which is what forced revision 2's 53° planet.** `SLOPE_REF` is the same talus tangent. The meander amplitude comes from the channel width. The width's and depth's GRAVITY factors come from the threshold-channel balance. `Δt` derives from `EROSIONAL_AGE_YR / PASSES`, so the pass count is a resolution and the age is the physical number (§4.6). What remains — `EROSIONAL_AGE_YR`, `K0`, `a_w`, `a_d`, the pass counts, `MACRO_CELL_TARGET_M`, `σ`, the `Q^0.4` table and the rest of §5.5's list — are WORLD constants in the declared tag, listed in full and never per-body tuning. |
| **The 8 ms chunk budget (V10)** | **NOT PASSED, and stated as a risk.** ESTIMATED +0.67 to +1.50 ms for a chunk with a river; **+2.67 to +4.50 ms when a water surface is meshed.** And the 6.1 ms baseline was MEASURED on the recipe §4.2 replaces, so it is not a baseline. **UNMEASURED until M2**, which must run AFTER §4.2 lands. §12 D15 names the dial, and it is the water mesh, not `SUBDIV`. |
| **The ladder (V8/V9)** | `Z` is identical at every rung and contributes zero to `dropped_bound_m`. **The OCTAVES are the term that moves, and §7.1 changes the coarsening rule to fix it**: COMPUTED, the dropped bound at rung 6 falls from 32.8 m today and 116 m under the count rule to 3.7 m under the wavelength rule, and the kept-octave count stays monotone so a coarse rung stays cheaper than rung 0. `C` and `T` drop by the same width and cell-size rule the caves already use. **UNMEASURED until M8**, extended to the octave pop. |
| **The record (V6)** | Nothing changes. The water field is derived, never stored. The COLUMN FACIES is derived per column and never stored (§12 D8). |
| **The edit pyramid** | Unchanged: an edit overrides the composed shape at the composition order of `docs/investigation/2026-09-07/03_generator_sl10.md` §4.4. Erosion sits in the seed layer, below every diff. |
| **The collider on the same shape (V1.5)** | The channel, the floodplain, the lake bed and the sea bed all live in the height field, so the extractor meshes them and the collider IS that mesh. **And the walkability question is now part of it:** the anchor of §4.2.3 exists because a 53° planet would have been a 53° COLLIDER, and a player would have slid off the world. |
| **SL8 — seamless** | §9 rules 1–5, plus §5.4's G-MACRO-EDGE and G-MACRO-CORNER, which exist because revision 2 would have put 63 000 km of scarp along the cube edges and left eight corners undefined. **There is no crossfade to lean on — it is slice 8 and not built.** UNMEASURED until M8 and M11. |
| **HR5 — 100 % coverage** | **UNMEASURED, and the enabling fact is unknown.** The solve holds data-dependent branches: a pit, a lake, a FLAT, a seam, a corner, a no-sea planet, a dry planet, a node with no receiver, a glaciated node, a coastal node, a delta, an alluvial fan. COMPUTED, a 200 km moon of THE world has 6 144 macro nodes and a 50 km body has 384, so a full solve is AFFORDABLE inside a unit test even instrumented — but affordability is not reachability. **§14 M7 states the open fact: nobody has found a body of THE world that is both small enough to instrument and wet enough to reach the fluvial arms.** The two lawful branches if none exists are (a) run the home planet under instrumentation, COMPUTED at minutes per test binary, or (b) exercise the fluvial arms through a Tier-A unit fixture that feeds the solve a stated macro field of a REAL body rather than inventing a planet — which SL5 permits, because it invents no world, only a test input. §12 D20 puts the fork to the owner. |
| **HR4 — features once, run anywhere** | The solve names no realm kind. A moon, a planet and a hull that holds terrain (ruling V4) run identical code; a body with no liquid water simply gets `f_water = 0`. The base-level rule of §4.6 is one `max` with no branch on a landform kind. `n_macro`'s rule (§4.1) is stated so it gives an answer on a 5 km body as well as on the home planet. |
| **V4 — vegetation is an art asset placed by a server skeleton** | The COLUMN FACIES is exactly the input that skeleton needs: forest on the floodplain, grass on the plain, gravel on the talus, nothing above the ELA. This document produces the field and places no plant. |

---

## 12. What you decide

### 12.0 The build order, and what the first slice can be

Refutation B's weakness W2 is right that revision 2 listed fifteen decisions and no build order, and that
the format the owner is used to (`docs/investigation/2026-09-07/slice_07_client_link.md` §11) states how
the work is built. Five decisions are prerequisites in other crates or domains — D3 (the 28-byte
statement), D6 (five facts in `vd-physics`), D12, D13 and D17 (asks of DOMAIN 02 and the weather domain).
So here is what CAN be built first, and with what stubbed:

| slice | what lands | what is stubbed | why it is first |
|---|---|---|---|
| **E1 — THE PICTURE** | The SLOPE SPECTRUM (§4.2.3), the ROUGHNESS FIELD (§4.2.4) driven by a stated placeholder field, the LADDER'S WAVELENGTH RULE (§7.1), ridged crests and the cap-rock bench (§7.3 rules 2 and 3). | No solve at all. The roughness field reads the coarse octaves' own slope instead of `Z`. No rivers. | **It answers the owner's complaint with no artifact, no ask and no new lane**, and M3's before/after picture and M9's slope histogram both run on it. If E1's vista does not read as land, nothing later will. |
| **E2 — THE SOLVE** | The macro lattice, D8, the priority flood, the flat resolve, the discharge, the stream-power sweep, the envelope assertion, `Z` in `height_m`, the pyramid, the seam gates. `P = P_MIN` (§4.5 fallback). | The climate (DOMAIN 02), isostasy, ice, coast, sediment. `Z` replaces the coarse octaves, and the roughness field switches from the placeholder to `Z`. | It is the largest single piece and everything else hangs on it. M1 and M4 run here. |
| **E3 — THE WATER** | The per-node water level, lakes, the channel and floodplain carve, the meander subdivision, the water rule, the clipped water mesh, the sub-macro stream. | — | This is where the 8 ms budget is decided, so M2 and M14 gate it. |
| **E4 — THE FACTS** | The 28-byte statement (D3), the five drawn facts (D6), the erodibility's `f_water` and `f_frost`, isostasy with a real `T_e`. | — | It needs `vd-physics` work and an owner ruling, so it must not block E1–E3. |
| **E5 — ICE, COAST AND SEDIMENT** | The ice pass, the ELA, snow on the ground (D18), the coastal band, the platform, the beach, the delta, the alluvial fan, the per-basin sediment. | — | Each is a landform family on top of a working solve. |

**E1 is buildable today**, needs no ask of any other domain, and is the slice that answers the words the
owner actually wrote.

### 12.1 The decisions

| # | Question | Options | Recommended | Why |
|---|---|---|---|---|
| **D1** | The macro lattice. | (a) `n_macro = 320`, 16.4 km, 614 k nodes. (b) **`n_macro = 640`, 8 224 m, 2.46 M nodes, 14.75 MB.** (c) `n_macro = 1024`, 5 140 m, 6.3 M nodes, 37.7 MB and about 330 MB in the solve. | **(b)** | (c) triples the solve time and the artifact. (a) makes trunk valleys 32 km apart, which reads as a lumpy planet. (b) divides `N` exactly, holds the solve in 130 MB, and is 0.4 % off the 8 192 m target. |
| **D2** | Do we ever refine the solve near a player? | (a) never; the sub-macro scale stays table-driven. (b) a fixed tiling refined to about 2 km with a halo, in a later slice. | **(a) now, (b) as a named later slice** | The picture is now dominated by the spectrum, the roughness field and the table-driven terms, all of which work below 8 km. Refinement adds a seam between tiles that must then be measured and crossfaded — and there is no crossfade until slice 8. |
| **D3** | **How do the client's inputs arrive, and who computes the artifact?** REVISION 3 CHANGES THIS ANSWER AGAIN. | (a) revision 1: the client derives everything, including the facts. (b) revision 2: the realm STATES the facts and SHIPS the artifact. (c) **the realm STATES the 28 bytes of facts on the existing statement; every host DERIVES the artifact; the HOME PLANET's artifact is a committed, digest-pinned build artifact linked into both hosts.** | **(c)** | (a) is impossible: ruling S7-1 refuses the client a motion crate. (b) breaks ruling S7-2 — the client generates eight home-planet chunks AT LOGIN, before any realm has spoken — has no lane for 14.75 MB against a 1 200-byte budget, would cost 413 MB for a 28-body system, and reverses the reason SL10 exists. (c) keeps SL10, makes ONE SL6 ask of 28 bytes on a statement that already exists, and puts the home planet's ground under the player's boots before the first packet. Cost: about 15 MB of committed, compressible data in the repository and in the client build, and a background derive per body a player approaches. |
| **D4** | The channel depth law, given that `Q^0.4` needs a forbidden `powf`. | (a) `depth = a_d · sqrt(Q)` with a smaller constant. (b) a 256-entry COMMITTED LITERAL table indexed by the log-discharge byte. | **(b)** | The discharge is ALREADY a log byte. The table gives the true `Q^0.4` to within a quantisation step and costs one array read. It must be committed source, never computed at build time with `powf`. |
| **D5** | The precipitation field, if DOMAIN 02 is not ready. | (a) `P = P_MIN` everywhere (pure drainage area). (b) block this work on DOMAIN 02. | **(a) for slices E1 and E2; never (b)** | Shape first, sizes second. But say it out loud at the picture review: with (a) a desert grows an Amazon, and the rain shadow does not exist until DOMAIN 02 lands. |
| **D6** | **Five body facts are MISSING from the world and every one of them is in the shape.** | (a) draw all five per body, quantised, and STORE them in the realm's own store: obliquity, spin period, atmospheric optical depth `τ`, effective elastic thickness `T_e`, and the glacial ELA offset. Surface gravity and eccentricity are free — the forest already holds them. (b) defer, and keep the orbit axis as the spin axis. | **(a)** | The owner's task names spin, trajectory, size, gravity and position as inputs; §4.6 and §4.9 now consume all five. Without `τ` the design cannot tell an Earth (288 K) from a rock, because `t_eq_k` is 255 K for both. Without `T_e` the flexural rebound has no width. Without obliquity the ELA has no latitude law. `τ` is `D-TAX-1`'s own owed item. **STORED, not recomputed** — that is what stops a rescheduled realm re-addressing every stored edit. |
| **D7** | The glacial ELA — one fact or two? | (a) one seed-drawn glacial ELA per body. (b) a seed-drawn glacial ELA AND today's, giving both fresh and relict glacial forms. | **(b)** | Two numbers, one draw each, and it is the difference between "mountains with U-valleys" and "mountains with U-valleys, hanging valleys and fjords". The cost is one extra flag per node. |
| **D8** | Where does the COLUMN FACIES ride? | (a) the cell record's biome/object param. (b) more `Biome` arms. (c) derived per column at the fine pass and never stored. | **(c)** | It is a table-driven function of the artifact byte, the slope and the ELA. Storing it costs record space the V6 sitting has not settled. |
| **D9** | **The world's EROSIONAL AGE, which replaces "the pass count".** | (a) `PASSES = 40` frozen by taste, with `Δt·K0` free. (b) **state `EROSIONAL_AGE_YR` as a physical time, set `Δt = age / PASSES`, MEASURE the age against the per-basin hypsometric integral, freeze it, and prove by M12 that `PASSES` is a resolution and not a dial.** | **(b)** | With `U = 0` the landscape's maturity IS the elapsed time. Revision 2 left `Δt·K0` free and promised to measure `PASSES`; any hypsometric integral can be hit at any `PASSES` by moving `Δt·K0`, so that measured nothing. Naming a physical age makes the number a fact about the world, records it in the tag, and makes the pass count converge instead of choose. |
| **D10** | Water after an edit. | (a) postponed and stated. (b) designed now. | **(a)** | Flowing water on a live grid is its own subsystem with its own cross-realm question, and the owner already postponed cross-realm edits. |
| **D11** | **When is a body solved, and where?** | (a) lazily on each host, on the injected worker pool, cached. (b) OFFLINE at world creation for the home system's bodies, committed with the build. | **(b) for the home planet — it is forced by D3 — and (a) for everything else** | Ruling S7-2 makes the home planet's artifact a login prerequisite, so it cannot be lazy. Everything else is derived in the background while the player approaches, and until it lands the shard REFUSES chunk requests with a stated reason rather than answering from a partial field (§5.3). |
| **D12** | **The two-step hypsometric curve needs two kinds of crust, and this design has one.** | (a) DOMAIN 02 adds a seed-drawn continental/oceanic crust field with two floating heights. (b) accept a unimodal curve and retire the Earth-curve comparison. | **(a), asked of DOMAIN 02** | Earth's two steps are a TECTONIC fact (35 km of light continental crust against 7 km of dense oceanic), not an erosional one. This design's initial surface is a sum of fourteen bounded noises and is therefore unimodal. §14 M6 measures the LAND curve, which erosion does shape. |
| **D13** | **The slope spectrum and the roughness field edit the shared octave recipe.** | (a) this domain states the law and DOMAIN 02/04 implement it in `body.rs` and `height.rs`. (b) each domain writes its own. | **(a)** | Erosion makes both the roughness peak and the plain-against-range contrast, so this domain owns the law; the code is shared, so the edit must be agreed once. |
| **D14** | **Two fields the world does not draw, that this design would use — the ask is SMALLER than revision 2's.** | a DEEP LITHOLOGY (bedded hard and soft rock below 92 m, for a canyon's full-height ledges); a JOINT FIELD (vertical fractures, for tors and free-standing pillars). | **ask DOMAIN 02 now; build without them** | **Revision 2 also asked for a shallow soft rock and was wrong to:** the world already draws Sandstone, Limestone or Shale at 20–80 m (`crates/terrain/src/body.rs:194,198`), which is enough for the wave-cut shelf, the badland and the cap-rock bench. Only the DEEP contrast and the pillar are missing. Asking for a field we do not need is a slice the owner did not have to buy. |
| **D15** | **The worst chunk exceeds the 8 ms budget, and the dial has changed.** | (a) accept the risk until M2, then decide. (b) mesh the water surface at a COARSER rung than the rock, so a water chunk pays a fraction of a second extraction. (c) mesh the water only where a water level exists in the column field's own range, and cache the water mesh across the radial chunks of one column stack. (d) `SUBDIV = 3`. | **(c), with M2 and M14 before the slice lands; (b) as the fallback** | ESTIMATED at +2.67 to +4.50 ms on a 6.1 ms baseline, a cave-dense chunk with a river and a water surface is 8.8–10.6 ms. **`SUBDIV` is worth about 0.2 ms and cannot pay for 2–3 ms**, which is what revision 2's recommendation rested on. The water mesh is the term that costs, so it is the term that must be dialled. |
| **D16** | **Two structural changes this design needs, which revision 2 charged as line items.** | (a) the SAMPLE BOX's column ring grows from one cell to two — two ring depths, two index functions, a changed `sites`/`dirs`/`surfaces` layout that `extract` also reads. (b) keep ONE ring and take a halo column's slope from a one-sided difference. Separately: is the crust's spare 64 m free to spend, or does `below_surface_cell` need it? | **(b) first, (a) only if M3's picture shows a seam in the outermost ring; and ASK about the 64 m** | (a) is a refactor of the sample box, not a `+6.3 %` line. (b) is address-defined too and costs nothing. On the 64 m: the code names the purpose of the 64 ABOVE and not of the 64 BELOW, and a design that spends an undocumented margin to zero should ask first. If it is needed, `crust_m` already carries 23–92 m of unused depth (it adds the strata AND the cave depth where only the deeper is needed), which can be recovered without moving `floor_m`. |
| **D17** | **The shape must read a CLIMATOLOGY, never live weather.** | (a) the weather domain states that live wind and rain modulate the LOOK only. (b) leave it open. | **(a), asked of the weather domain** | If the wind becomes live state, this document's precipitation input is not derivable and the whole solve stops being a function of the seed. A rainstorm must never re-address a chunk. |
| **D18** | **Snow on the ground above the ELA.** | (a) the COLUMN FACIES takes `glacier` above today's ELA and the stratum table gives `Stratum::Snow` there, so a high equatorial ridge is white. (b) leave snow to the biome, as `crates/terrain/src/strata.rs:190-196` does today, and let the client's style paint the cap. | **(a)** | The reference picture's single most obvious feature is a snow-capped ridge. Today a 6 000 m equatorial ridge on the home planet carries `Gravel`. (a) is one comparison per column against a number the solve already holds and changes no record. It touches the shared strata table, which DOMAIN 05 also edits, so it is an owner decision and not a quiet edit. |
| **D19** | **Half to four fifths of the visible valleys would be DRY.** | (a) draw a SUB-MACRO STREAM in a converging sub-macro valley — table-driven, no discharge, no graph, no storage (§6.1). (b) accept dry valleys and say so at the picture review. | **(a)** | COMPUTED: a 50 km vista holds about six macro channel links against a valley density of 0.33–0.67 km/km². On Earth a mountainside with twenty valleys has twenty streams, and in the reference picture the water is the thing the eye follows. (a) is two table-driven terms per column on a flow field the design already computes. |
| **D20** | **HR5's fluvial arms may be unreachable on any small body of THE world.** | (a) run the home planet's full solve under coverage instrumentation, COMPUTED at minutes per test binary. (b) exercise the fluvial arms from a Tier-A unit fixture that feeds the solve a stated macro field taken FROM a real body of THE world. (c) find and name a small wet body, if one exists. | **(c) if M7 finds one; otherwise (b)** | 100 % region and branch in a Tier-A crate is not optional, and a dry 200 km moon reaches none of the fluvial arms. (b) invents no world — only a test input — so SL5 holds; (a) makes the gate too slow to run. Revision 2 claimed this gate passed on an unmeasured premise, which is exactly what the "never assume, measure" rule forbids. |

---

## 13. Open questions

1. **How wide is the picture's believability window?** The vista test (V10 §12) is qualitative. §14 M3,
   M6 and M9 are quantitative. There is still no measurement for "does this vista read as land", and the
   proposals — drainage density, a slope histogram and an amplitude spectrum, each over a 50 km window
   against a real elevation model — are the honest numeric form of the owner's actual request.
2. **Is the roughness law a BUMP or a BREAK?** §4.2.3 states a bump and says so. The published picture of
   topographic spectra is often described as a spectral break at the hillslope-to-valley transition.
   **§14 M9 must decide which before the constant is frozen**, and revision 2 presented the bump as
   settled geomorphology with no source.
3. **Does the roughness field ever draw a visible boundary?** It is slow and it follows the macro slope,
   so it should not. UNMEASURED, and M3's picture is where it would show.
4. **Does the closed-form gully field ever contradict the macro graph?** A gully that runs across a macro
   flow line reads wrong. The warp makes it unlikely and does not forbid it. UNMEASURED.
5. **What does the artifact cost on the largest body the ladder allows?** COMPUTED for the home planet
   and for two moons. A body ten times the radius holds a hundred times the nodes, and `N ≤ 2^26` bounds
   it, but nobody has computed the worst case.
6. **Is the flood really 1.0–3.2 s?** It is the solve's dominant cost and it is a cache-behaviour
   question, not an arithmetic one. Determinism is not at risk; the 12–45 s headline is.
7. **Where does a derived artifact live on disk, and who invalidates it?** The world tag changes when any
   solve constant changes (§5.5), which is the invalidation key. But the shard's store and the client's
   cache are different machinery, and neither holds a place for a per-body derived blob today. This is an
   architectural dependency of §12 D11, not a curiosity.
8. **Does DOMAIN 02's precipitation field close the circle stably, and is it `O(1)` per node?** Alternating
   climate and erosion can oscillate. Fixed pass counts make an oscillation deterministic; they do not
   make it less ugly. And if the field is an upwind streamline integration it is not `O(nodes)` and it
   could dominate §10's headline. UNMEASURED; §14 M1 benches it.
9. **The trunk river's own meanders do not exist.** COMPUTED: a 931 m river's meander wavelength is
   10.2 km, longer than one macro segment, so the D8 tree cannot express it and the subdivision cannot
   either. And the subdivision's served band is 190–745 m of width, not 47–745 m. A great river in this
   world runs straighter than the Mississippi. Nobody has decided whether that is acceptable.
10. **A hull or station that holds terrain (ruling V4) has no basin and no rain.** No macro solve, no
    rivers — but a station with a built valley and a built river is a thing a player will want, and
    nobody has ruled on it.
11. **Three landform families are named and not built: VOLCANIC, AEOLIAN and JOINTED (tors and pillars).**
    The world's own bedrock table asserts a volcanic history it never shows. The owner should see the list
    before the picture review, because the reference picture's rock pillars may be volcanic necks.
12. **The sediment budget is one number per basin, so the sea never fills.** §4.11 makes a delta and a fan
    from rock that was really removed. It does not conserve volume across basins and does not make the
    depositional half of the hypsometric curve. §14 M6 measures the LAND curve only.
13. **Is the reach lead time longer than the derive?** If a player closes on a planet faster than 12–45 s
    of one background thread, the artifact is late and the shard refuses chunks (§5.3). §14 M15 computes
    the lead time from the suit's and the hull's rated cruise. If it is short, the answer is the pyramid
    first — the top level is 4.8 KB and covers the approach — and the full field behind it.

---

## 14. Measurements owed, in order

| # | Measurement | How | Why it sits there |
|---|---|---|---|
| **M1** | The macro solve's real time and memory at `n_macro = 640` on the home planet, **including a separate row for DOMAIN 02's climate recompute**. | A new mode in `crates/bins/examples/terrain_cost.rs`. | Every cost claim in §10 is a per-operation model. **§12 D3 and D11 must not be put to the owner before this runs.** The climate row is new because revision 2's table had none. |
| **M2** | The fine pass's added cost per rung-0 chunk, for four cases: no river; a river; a river through a CAVE-DENSE chunk; a river through a cave-dense chunk that also meshes a water surface. **The baseline must be RE-MEASURED under §4.2's recipe, and the extraction half of the 6.1 ms must be split out.** | The same bench, run AFTER slice E1 lands. | §7.4 ESTIMATES the worst case at 8.8–10.6 ms of an 8 ms budget. This decides §12 D15. Revision 2 benched against a baseline this design deletes. |
| **M3** | The vista: a 50 km view down a valley with a river, framed like the reference picture, **before and after slice E1** (the spectrum, the roughness field and the wavelength rule). | The client, through the existing screenshot harness (HR6). | This is the owner's actual acceptance test. E1 is buildable with no artifact and no ask, so this picture can be taken first. |
| **M4** | Byte-for-byte identity of the artifact on x86-64 and on aarch64, with the world identity folding the STATED facts. | The existing digest gate, extended. | SL10 is a measurement, never an argument. It is also the gate any future parallel decomposition must pass (§5.3). |
| **M5** | G-MACRO-EDGE (value and slope across all twelve cube edges), **G-MACRO-CORNER including the INTERPOLATION inside the blend radius**, G-MACRO-SEAM. | Unit tests on a small real body of THE world. | A seam mistake is silent; revision 2's design would have put 63 000 km of scarp on the home planet and left eight corners undefined. |
| **M6** | The home planet's LAND hypsometric curve, and the per-basin hypsometric integral, against a real elevation model. | A histogram over sampled macro nodes, offline, against ETOPO or SRTM. | The land curve is the part erosion shapes and this design can fail. The basin integral sets `EROSIONAL_AGE_YR` (§12 D9). The two-step ocean-and-land curve is NOT measured here: §12 D12 says why. |
| **M7** | **Does THE world hold a body that is both small enough to instrument and wet enough to reach the fluvial arms?** | Walk the forest's bodies for the home galaxy and test each against the solve's arm list. **The body must be NAMED, with its seed and radius bits, the way `crates/terrain/src/home.rs:16-22` names the home planet.** | It is the enabling fact for the HR5 gate, and §11's row says UNMEASURED because of it. §12 D20 states the fork if the answer is no. |
| **M8** | **The pop at a rung transition, for THREE terms: the OCTAVES under the new wavelength rule, a river's arrival, and a pyramid-level change of `Z`.** | Extend the V10 pop detector. | SL8, with no crossfade to hide behind — the crossfade is slice 8 and is not built. Revision 2's M8 measured only the river, and the octaves are the term that moved. |
| **M9** | Over a 50 km window against a real elevation model: the AMPLITUDE SPECTRUM, **the SLOPE HISTOGRAM** and the drainage density — of the design AND of the reference. | Offline, from the artifact plus the fine pass. | This calibrates `o_peak`, `σ` and `TALUS_RMS`; it decides whether the law is a bump or a break (open question 2); and **the slope histogram is the measurement that would have caught revision 2's 53° planet before the owner did.** |
| **M10** | `dropped_bound_m` for the channel term, sampled ALONG a known channel of the home planet. | A new fixture beside `crates/terrain/src/height.rs:80-105`. | The existing 400-sample test would pass while measuring nothing: a channel covers about a thousandth of the surface. |
| **M11** | If G-MACRO-EDGE forces the quintic fallback (§5.4): the size of the 8 km ripple against the eleven-seam taxonomy's "detail-by-box" row. | The pop detector, on a flat-lit slope. | The fallback trades a crease for a ripple, and only a picture says which is worse. |
| **M12** | **`PASSES` is a RESOLUTION, not a dial:** solve the home planet at `PASSES` and at `2 · PASSES` with the same `EROSIONAL_AGE_YR`, and assert the two `Z` fields agree within a stated bound. | A unit test on a small real body, and once on the home planet. | It is the proof behind §12 D9. Without it the age and the pass count are still one free constant wearing two names. |
| **M13** | **The talus pass converges:** the maximum slope over the body falls monotonically across `TALUS_PASSES`, and the total mass moved never exceeds the excess. | A unit test on a small real body. | Revision 2 answered only determinism for this pass, and convergence is a different property. |
| **M14** | **The water mesh's cost beyond worker time:** vertices, bytes and draw calls for a coastal chunk and a lake chunk of the home planet, against ruling V10's MEASURED 405 KB per chunk. | The client, through the existing geometry counters. | §12 D15's dial is the water mesh, and the 8 ms budget covers worker time only. |
| **M15** | **The reach-to-arrival lead time**, from the suit's and the hull's rated cruise (the 2026-09-05 ruling), against the 12–45 s derive. | Arithmetic on the shipped ratings plus the reach ruling's radius. | It says whether a client can derive a planet's artifact before it arrives, and it is the number §10 has called UNMEASURED twice. |

---

## 15. The recommended design in one paragraph

Solve erosion ONCE per body on its own uniform cube-sphere lattice of 640 CELL-CENTRE nodes per face edge
— 8 224 m, 2 457 600 nodes and about 130 MB on the home planet, with 640 dividing `N` so every lookup is
an integer division — starting from only those octaves the lattice can carry (four samples per
wavelength, which is octaves 0 to 3 on the home planet), with D8 routing whose tie-breaks are integer, a
priority flood whose key is `(integer height, node index)` and whose flats are resolved by an integer
distance field rather than by an epsilon that would run a continent's rivers in index stripes, a WATER
LEVEL kept per node so a lake is flat and a river's base level at a lake is its spill level and not its
floor, discharge accumulated as exact integers from DOMAIN 02's rain between a stated floor and one
stated ceiling, the stream power law at `m = 1/2` so `Q^m` is the `sqrt` the fence grants, solved
implicitly with `Δt = EROSIONAL_AGE_YR / PASSES` so the pass count is a resolution and the age is the
measured physical number, one sediment total per basin so the delta and the alluvial fan are made of rock
that was really removed, an isostatic rebound on a pyramid that reaches the real 150 km flexural
wavelength and crosses a cube seam by construction, a talus pass that splits its excess proportionally so
mass is conserved, an ice pass that MARKS where ice was and leaves every cirque and arête to the fine
pass, a coastal band mark, and an envelope assertion `|Z| ≤ relief_m − A_fine` that makes the address a
proof; keep six bytes per node, 14.75 MB, plus a pyramid whose top level is 4.8 KB, and throw the solve
away. Redistribute the relief the recipe already draws with a SLOPE SPECTRUM whose peak is anchored on
the ANGLE OF REPOSE — 319 m at 6.25 km and 199 m at 3.1 km against today's 80 m and 38 m, summing to
1 532 m of a 14 305 m relief, so the envelope never binds and the planet stands at 35° at its roughest
instead of the 53° a normalisation to the relief would have forced — and modulate it by a ROUGHNESS FIELD
derived from the macro slope, so a craton is a plain a player can build on and an orogen is the reference
picture's range. Change the ladder to coarsen BY WAVELENGTH instead of by count, which takes the
rung-to-rung disagreement at the vista's own rung from 15 m to 1.2 m against a 57.5 m pixel and fixes a
defect that exists today. At every rung, read `Z` in place of the coarse octaves through a Catmull-Rom
whose stencil straddles a cube edge uniformly because the lattice is cell-centred, with a stated quintic
rule inside two nodes of each cube corner, carve the channel and its floodplain with the
`segment_distance_m` the cave tubes already use, and add table-driven detail — scree by an
address-defined slope, a cap-rock bench at the soft sediment's own base that is the picture's near cliff,
ridged crests, strath terraces, gullies warped along the stored flow direction, and a sub-macro stream so
the visible valleys are not dry — for an estimated 0.67 to 4.50 ms on top of a baseline that must be
re-measured. Water gets one level per node and per column and one second run of the same extractor over
`min(water_level − r, r − rock)`, so the sheet ends at the shoreline; nothing in the cell record changes;
the body's physical facts are DRAWN ONCE, STORED by the realm and stated to the client as 28 bytes beside
the surface statement that already exists, so no host recomputes a transcendental and a realm rescheduled
from a laptop to a pod does not move the ground under a player's house; the artifact is DERIVED on both
hosts and never shipped, and the home planet's is committed and digest-pinned so the world hello still
works at login; a river may decide where sand and gravel lie and may never decide where riches lie; and
no part of it is a function of time, so a fixed seed, an address and a stated fact give the same valley
on an M4, on a pod, and on every client, forever.

---

## 16. Refutation answers

Severity is the refuter's own. **"FIXED"** means the section was rewritten, not annotated. **"KEPT"**
means the refuter is wrong and the reason is given with evidence. **"OWED"** means the finding is right
and the answer belongs to somebody else, named.

### 16.1 Round 2, refutation A — the laws and the code

| # | Finding | Severity | Answer |
|---|---|---|---|
| A2-B1 | Ruling S7-2 refuses D3, and the document never cites S7-2: the client generates eight home-planet chunks AT LOGIN, and those chunks need `Z`. | Blocker | **FIXED.** The refuter is right and the chain is broken exactly as drawn. §1 gains a row for S7-2 with `crates/terrain/src/tag.rs:43-50` and `crates/terrain/src/digest.rs:22-26`; §5.2 is rebuilt; §11 gains an S7-2 row; §12 D3 changes its recommendation for the second time. **The answer is not to change S7-2.** The artifact is DERIVED on both hosts (SL10 restored) and the HOME PLANET's artifact is a committed, digest-pinned build artifact linked into every host, in the `crates/terrain/src/home.rs:16-22` shape. Then the world hello works unchanged and the login needs no packet. |
| A2-B2 | §5.4's C1 seam rests on a false premise: `face_param` is a CELL CENTRE in the open interval, so no node lies on a cube edge and there is no shared node to own. | Blocker | **FIXED, and the true geometry is better than the false one.** §1 gains a row citing `crates/seed/src/ladder.rs:142-151`; §4.1 states the cell-centre convention once; §5.4 is rewritten. Unfolded about the edge, the two faces' cell centres sit at `…, 1−3/n, 1−1/n, 1+1/n, 1+3/n, …` — a UNIFORM continuation at one node spacing with the edge at the midpoint. So Catmull-Rom is C1 across the seam with NO ownership rule and NO shared value, which deletes two of revision 2's four rules. The refuter's `corner_param` alternative is not taken, because it would break §4.1's integer lookup. |
| A2-B3 | The slope spectrum multiplies `dropped_bound_m` by up to ten, and the only gate named is a tautology that grows with it. | Blocker | **FIXED, and it changed the ladder.** §7.1 is rewritten with the computed table on the home planet's real literals, and it states the gate's tautology outright (`height.rs:98-101` asserts against a bound that is itself the sum of the dropped amplitudes). The cure is a new COARSENING RULE: keep an octave while `λ ≥ 4 · cell_m`. COMPUTED, the dropped bound at rung 6 goes from 32.8 m today and 116 m under the count rule to **3.7 m**, and the kept count stays monotone so a coarse rung stays cheaper. §14 M8 now measures the OCTAVE pop, not only the river's. And §9 rule 5 states plainly that **there is no crossfade to lean on** — the refuter is right that `grep -rn "crossfade" crates/` finds none and that ruling V10 puts it in slice 8. |
| A2-B4 | The macro lattice under-samples the octaves it calls coarse, by a factor of two, and cannot cap the slopes it is asked to cap. | Blocker | **FIXED in both halves.** §4.2.1 states the rule the refuter asks for and makes it a world constant: an octave is COARSE only when `λ ≥ MACRO_SAMPLES_PER_WAVE · node`, with `MACRO_SAMPLES_PER_WAVE = 4`. COMPUTED on the home planet, that moves octaves 4 and 5 out of the solve, exactly as the refuter computed, and `A_coarse` becomes 13 617 m. The second half — that the macro talus pass cannot see a 1.5 km slope — is fixed at the source instead: §4.2.3 ANCHORS the spectrum at the angle of repose, so the fine octaves never exceed it and there is no overshoot to cap. §4.8 now says what the macro talus pass is actually for. |
| A2-B5 | The artifact cannot answer §8.1's water rule: three facies bits say *a* lake, never *which* lake. | Blocker | **FIXED, at the refuter's own price.** §4.4 replaces the lake TABLE with a WATER LEVEL PER NODE, an `i16` in the same encoding as `Z`. The artifact goes from 4 to **6 bytes per node, 14.75 MB**, and that number replaces 9.83 MB in §0, §3, §4.1, §4.14, §6.1, §10, §11, §12 D1, D3 and §15. It also fixes refutation B's D3 (a lake's base level) at no extra cost. The second half — §8.5 offering the weather domain two fields the solve throws away — is FIXED by §8.5 stating honestly that the ELA and the precipitation are NOT kept, and naming the two lawful ways to get them. |
| A2-D1 | The headline spectrum numbers are not the home planet's, and the "constant slope" diagnosis is false for it. | Defect | **FIXED, and the diagnosis is rewritten, not only the digits.** Recomputed here from `SplitMix64`, `child_seed` and `draw_unit`: `relief = 14 304.89 m`, `k_rough = 0.468 338`, amplitudes 7 605.6 / 80.3 / 37.6 / 17.6 m, per-octave slope falling 0.1195 → 0.0510. Every one of the refuter's numbers reproduces. §1, §0.3 and §4.2.3 now carry the real body. The DIAGNOSIS is restated as two defects — no characteristic scale, and no contrast — which is what led to §4.2.4's roughness field. |
| A2-D2 | The discharge byte does not fit the range the document's own §4.5 states; the KEPT half of round 1's A-D2 rests on the wrong ceiling. | Defect | **FIXED.** §4.5 states ONE ceiling, `P_MAX = 10 000 mm/yr`, with Mawsynram as the reason 1 000 is not honest. §6.1 re-sizes the byte against it and against the MINIMUM node area: span `2^34.8`, declared at 36 octaves, half-step error 4.9 % on `Q` and 2.5 % on a width. The round-1 KEPT answer is withdrawn: it was arithmetic on the wrong ceiling. |
| A2-D3 | `PASSES` and `Δt·K0` are degenerate, so §12 D9's "measure it" is not a measurement; and `U = 0` gives no steady state. | Defect | **FIXED.** §4.6 gives `Δt` a physical meaning: `Δt = EROSIONAL_AGE_YR / PASSES`, so `PASSES` becomes a numerical RESOLUTION and the AGE becomes the measured physical number. §14 M12 proves the resolution claim by solving at `PASSES` and `2·PASSES`. §12 D9 is rewritten. The second half is accepted and stated: §4.2.2 says outright that this is a POST-OROGENIC world with no steady state, and §13 open question 11 names the missing active-orogen landforms. |
| A2-D4 | The second extractor run is charged at a fifth of the first, and it is the number that decides D15. | Defect | **FIXED.** §7.4 charges the water pass at **2–3 ms**, on the refuter's own reasoning: a second run of the same extractor cannot cost a fifth of a run that already contains one. The worst case becomes 8.8–10.6 ms of an 8 ms budget — over by 33 %, not 3 %. §12 D15's options change: the dial is the WATER MESH, not `SUBDIV`, which is worth about 0.2 ms and cannot pay for 2 ms. §14 M2 must also split the extraction half out of the 6.1 ms. |
| A2-D5 | §11's HR5 row answers a cost question and calls it a coverage answer. | Defect | **FIXED.** The row now reads UNMEASURED, says affordability is not reachability, names §14 M7 as the open fact, and names the fork (§12 D20). §11's preamble states the rule: a gate row says what it gates and whether the gate has run. |
| A2-D6 | §4.1's `n_macro` rule contradicts itself on a small body; and a rung-0 chunk does not sit inside ONE macro node. | Defect | **FIXED in both halves.** §4.1 restates the rule as a search over the divisors of `N` at or above `MIN_MACRO_EDGE`, with no clamp after the choice, so `n_macro` always divides `N`; the refuter's own 5 km body (`N = 7 854`) is worked through to `n_macro = 11`, a 714 m node. §6.4 says a chunk spans at most TWO macro nodes per axis, with `8 224 / 62 = 132.6` computed. |
| A2-D7 | §7.3 rule 2 claims the reference picture's rock pillars and describes the rule that erases them. | Defect | **FIXED, and the claim is WITHDRAWN.** §7.3 rule 2 says so outright: a pillar stands STEEPER than the talus angle by definition, and a rule that relaxes everything above it removes every pillar by construction. In its place the section builds the cliff mechanism the world can actually support today — the CAP-ROCK BENCH at the soft sediment's base (see A2's own D9 in refutation B). The free-standing pillar needs a JOINT field, is not claimed, and is named in §12 D14 and §13 open question 11. |
| A2-D8 | The pyramid's top level is 4.8 KB, not 19 KB. | Defect | **FIXED.** §4.14 carries the level-by-level table: 1 228 800 + 307 200 + 76 800 + 19 200 + 4 800 = 1 636 800 B. 19.2 KB is level 40; the stated top of 20 is 4.8 KB. Corrected in §0.2, §3, §4.14, §9 rule 2, §10, §12 D3 and §15. |
| A2-D9 | The ELA is computed and never reaches the ground; a 6 000 m equatorial ridge still carries `Gravel`. | Defect | **FIXED, and it became a decision.** §1 gains a row for `crates/terrain/src/strata.rs:190-196` (snow is biome-keyed today). §4.9 adds THE SNOW ON THE GROUND in two halves: the COLUMN FACIES takes `glacier` above today's ELA and the stratum table gives `Stratum::Snow`; ruling V4 puts the white on the client as style keyed by the same facies. §12 D18 puts it to the owner because it touches the shared strata table. The refuter is right that the owner's acceptance picture turns on it. |
| A2-D10 | The talus pass moves the same excess to up to eight neighbours; mass is not conserved and it can oscillate. | Defect | **FIXED.** §4.8 distributes `TALUS_FRACTION · E` in PROPORTION to each neighbour's excess, with `TALUS_FRACTION ≤ 0.5`, so the total leaving a node is bounded by half its own excess. §14 M13 is a new measurement: the maximum slope falls monotonically across the passes. The refuter is right that revision 2's §5.1 row answered determinism, which is a different property from convergence. |
| A2-W1 | "The cell box does not grow" implies a restructure of `SampleBox` that is never named. | Weakness | **FIXED.** §7.3 rule 1 says it is a RESTRUCTURE — two ring depths, two index functions, a changed `sites`/`dirs`/`surfaces` layout that `extract` also reads — and §12 D16 makes it a decision, with the cheaper alternative recommended: keep one ring and take a halo column's slope from a one-sided difference, which is address-defined too and costs nothing. |
| A2-W2 | The priority flood's `ε` writes a false drainage direction across flat ground. | Weakness | **FIXED.** `ε` IS ZERO. §4.4 resolves a flat by an integer breadth-first DISTANCE-TO-OUTLET field (Garbrecht and Martz 1997), with ties on the smaller node index. §2 explains flat resolution as a term. §7.2's descent gate gains a fourth statement for it. The refuter's 625 m over 10 000 nodes is right and it is why. |
| A2-W3 | The solve is transport-free, so every depositional landform is cosmetic; and the alluvial fan is not named. | Weakness | **FIXED.** §4.11 is a new section: one `u64` deposited volume per basin, summed from the removal field the isostatic pass already builds, spent on the delta and — newly named and built — the ALLUVIAL FAN at a range front, with the freeboard scaled by the same number. §8.4 names SEDIMENT TRANSPORT proper in the postponed list, and §13 open question 12 states what the one-number version still cannot do. |
| A2-W4 | No volcanic landform is built or postponed. | Weakness | **FIXED.** §8.4 names volcanic landforms with the refuter's own point: `crates/terrain/src/strata.rs:136-142` draws Basalt, Gabbro and Andesite, **so the world already asserts a volcanic history it never shows**, and the reference picture's pillars may be volcanic necks. §13 open question 11 puts the three missing families in front of the owner before the picture review. |
| A2-W5 | The climate recomputation has no cost row, and it may not be `O(nodes)`. | Weakness | **FIXED.** §10 gains a row (0.1–0.5 s per recompute, UNMEASURED), §4.5 REQUIRES `O(1)` per node from DOMAIN 02 by name, §14 M1 benches it, and §13 open question 8 states the risk. |
| A2-W6 | One word, two meanings: "facies". | Weakness | **FIXED.** §2 states it once: a FACIES is one of eight depositional environments, never a substance; the SUBSTANCE is a `Stratum` and a stated table maps facies plus depth to stratum. The artifact holds the NODE FACIES, the fine pass derives the COLUMN FACIES, and both take values from the same eight. Every *"set the facies to `Gravel`"* is rewritten. |
| A2-W7 | The meander band is narrower than stated, by a factor of four. | Weakness | **FIXED.** §6.3 states the band as **190–745 m of width**, with the refuter's reasoning: four samples per wavelength is the least that reads as a curve, so `w ≥ 187 m`. §13 open question 9 carries the corrected figure. |
| A2-W8 | The 9.83 MB ship has no lane. | Weakness | **FIXED by deleting the ship.** §5.2 computes the mismatch (12 000 × the 1 200-byte self-look budget), §10 computes what a 28-body system would have cost (413 MB), and §12 D3 withdraws the ask. The remaining SL6 ask is 28 bytes on a statement that already exists. |
| A2-W9 | The macro node's area is a midpoint approximation described as exact. | Weakness | **FIXED.** §4.1 separates the two: the AREA is a midpoint quadrature with a COMPUTED parts-per-ten-thousand error over one node; the SUMMATION is what is exact and order-free. |
| A2-N1 | `body.rs:196` is cited for the sediment list; it is at `:194`. | Note | **FIXED.** §1, §4.6 and §8.4 cite `crates/terrain/src/body.rs:194` for the list and `:196-201` for the depths. Verified here against the source. |
| A2-N2 | `bend.rs:56-58` is cited for `W'`; it is computed at `:60`, and `K1` is at `:23`. | Note | **FIXED.** §4.1 cites the function range `crates/seed/src/bend.rs:52-65` rather than one line, and the constants at `:23-30`. |
| A2-N3 | The 1.5 ms column-pass figure has no source in the file cited. | Note | **FIXED by deletion.** §1 and §7.4 no longer state it. Every added cost in §7.4 is now labelled ESTIMATED with its model, and §14 M2 measures them. |
| A2-N4 | `chunk.rs:47` is cited for the gap byte; `chunk.rs:95-108` for `ColumnField`. | Note | **FIXED.** §1 cites `crates/terrain/src/chunk.rs:44-48` for the gap constant and `:92-107` for `ColumnField`, as ranges, so a one-line drift cannot make them wrong again. |
| A2-N5 | `extract.rs:13-18` states a fact about the mesh RING, not about a node's neighbour count. | Note | **FIXED.** §4.1 says the seven-neighbour count is COMPUTED from the cube-sphere's geometry, and cites `extract.rs:13-18` only as the crate's own record that three faces meet at a corner. |
| A2-N6 | The `Z` gather's `< 0.05 ms` is optimistic against a floor of about 0.065 ms plus cache misses. | Note | **FIXED.** §7.4 charges `Z` and the roughness field together at 0.07–0.15 ms. |
| A2-N7 | "Closed-form" is used for a rule that reads a 9.83 MB table. | Note | **FIXED.** §7's heading and preamble replace it with TABLE-DRIVEN, defined once: a bounded number of operations and table reads per column, no iteration, no dependence on a neighbouring chunk's array. |
| A2-N8 | The seed ruling's answer should cover the buildable-land map, not only ore. | Note | **FIXED.** §11's seed row answers the floodplain map by name and quotes the 2026-08-27 ruling's own words: where the valley is, is not treasure; who has already taken it, is live state. |
| A2-N9 | `crust_m` adds both the strata and the cave depth where only the deeper is needed, so there is more room than claimed — but §7.3 rule 2's downward cut is still unbudgeted. | Note | **FIXED.** §4.13 states the FULL downward budget — channel 32 m, glacial floor 16 m, detail 16 m, total 64 m — and §12 D16 names the 23–92 m of unused crust as the place to recover more room without moving `floor_m`. |

### 16.2 Round 2, refutation B — believability, cost and the owner

| # | Finding | Severity | Answer |
|---|---|---|---|
| B2-B1 | The slope spectrum's own normalisation forces `s_peak = 0.759` and a 53° planet, and the two named cures cannot act at the scale that makes it. | Blocker | **FIXED, and it is the largest change in revision 3.** The refuter's arithmetic reproduces exactly. §4.2.3 DELETES the line *"scale every `a(o)` so that `Σ a(o) = relief_m`"*. In its place: `s_peak` is ANCHORED so the fine octaves' RMS slope equals the loose-rock talus tangent (0.70, 35°), and the envelope becomes an INEQUALITY `Σ a ≤ relief_m` — which is lawful, because the band is derived from `relief_m` and spending less cannot move an address. COMPUTED on the home planet: `s_peak = 0.3998`, 319 m at 6.25 km, 199 m at 3.1 km, fine octaves summing to 1 532 m of 14 305 m. The refuter's own fork is taken, and taken explicitly. |
| B2-B2 | The spectrum is GLOBAL, so the reference picture's fields, plains and valley floors cannot exist, and there is nowhere to build. | Blocker | **FIXED, and it is the second-largest change.** §4.2.4 adds a ROUGHNESS FIELD, and the refuter is right that it was the missing piece. It is DERIVED from the macro slope of `Z`, not drawn: `m = m_min + (1 − m_min)·smoothstep(|∇Z| / SLOPE_REF)`, with `m_min = 0.06` giving a craton a 2.4° fine RMS slope. No new noise, no new draw, no new artifact field, one multiply per octave. §3 restates the answer as FOUR parts, and §12 D13 covers the shared edit. |
| B2-B3 | The spectrum multiplies the ladder's rung-to-rung error by five, and §11 claims the gate on `Z` alone. | Blocker | **FIXED.** Same rebuild as A2-B3: §7.1's WAVELENGTH coarsening rule, with the computed table, the monotone kept-count check, and the vista's own rung named (rung 5 or 6, one pixel 57.5 m at 50 km). §11's ladder row now says the octaves are the term that moves. The refuter's N2 — that revision 2 never named the rung its vista claim was about — is fixed in §3 and §7.1. |
| B2-B4 | "The realm SHIPS the artifact" has no lane, and it reverses the reason SL10 exists. | Blocker | **FIXED by withdrawing the ship.** §5.2 computes the mismatch against `SELF_LOOK_BUDGET_BYTES = 1200`; §10 computes the 413 MB a 28-body tour would have cost, which the refuter is right that nobody had computed; §11's SL10 row states the reversal in words and says revision 3 does not make it. The artifact is DERIVED on both hosts. The home planet's is a committed build artifact, which also answers A2-B1. |
| B2-B5 | The shipped facts move the drift from the client to the server, where a rescheduled realm re-addresses every stored edit. | Blocker | **FIXED.** §5.2 accepts the argument in full — it never depended on the host being a client — and answers it: the realm DRAWS its facts once at creation and WRITES them to its own durable store; every later boot READS them; the home planet's are additionally PINNED as literals with a test, in the `home_body_pin.rs` shape; and the world identity FOLDS them so a mismatch is detected as well as prevented. §5.5's constant list gains the drawn ranges, which the refuter is right were missing. §12 D6 now says STORED, not recomputed. |
| B2-D1 | The 8 ms budget is checked against a baseline this design deletes. | Defect | **FIXED.** §7.4 states outright that the 6.1 ms was MEASURED on the recipe §4.2 replaces, names both mechanisms that move it (fewer skips, and the 26 ms checkerboard bound the world moves toward), notes that the roughness field BOUNDS the first because a plain pays less than today, and requires §14 M2 to bench AFTER slice E1 lands. §11's budget row says NOT PASSED. |
| B2-D2 | The discharge byte's range fails on the document's own two ceilings; the floor is soft in two ways. | Defect | **FIXED.** §4.5 states ONE ceiling (10 000 mm/yr) and adds `P_MIN = 1 mm/yr` as a stated world constant, because the refuter is right that `P = 0` gives `Q = 0` and a logarithm cannot hold it. §6.1 re-sizes against the MINIMUM node area, not the face-centre one, and declares a 36-octave span with a 2.5 % width error. |
| B2-D3 | A gorge at every lake inlet: the sweep's receiver term is undefined at the lake edge. | Defect | **FIXED, in one line, exactly as the refuter prescribes.** §4.6 sets `b_i = max(z_terrain(receiver), water_level(receiver))`, so a river's base level at a lake is the SPILL LEVEL, at the sea the sea level, and on dry land the ground — one expression, no branch on a landform kind. §4.4's table gains the row. The per-node water level (A2-B5) is what makes it a single load. |
| B2-D4 | The pyramid's top level is 4.8 KB, not 19 KB. | Defect | **FIXED.** Same as A2-D8, with the level table in §4.14. |
| B2-D5 | The design REMOVES cliffs and no rule in it MAKES one; the "rock pillars" claim is false. | Defect | **FIXED, and the claim is withdrawn.** Same as A2-D7. The refuter's three real mechanisms are named, and the one the world can support TODAY is built: the CAP-ROCK BENCH at the soft sediment's base, up to 80 m of face at one height along a hillside, which is the reference picture's near cliff band. The pillar waits for a joint field (§12 D14). |
| B2-D6 | Twenty side valleys, and every one of them is dry. | Defect | **FIXED, and it became a decision.** §6.1 states the visible consequence the refuter is right revision 2 omitted: between a half and four fifths of the visible valleys would carry no water, and a dry valley is a rare landform on Earth. The cure is a SUB-MACRO STREAM — table-driven, no discharge, no graph, no storage — drawn where the gully field's flow converges, which lifts the DRAWN density to the same 0.33–0.67 km/km² the valleys have. §12 D19 puts it to the owner; §14 M9 measures it. |
| B2-D7 | The divisor rule's "integer lookup" does not apply at the generator's only entry point, and avoiding it needs a signature change that is never named. | Defect | **FIXED in the second half; the FIRST HALF IS KEPT.** The signature change is right and is now stated: §0.5 and §7.3 rule 1 make the entry point `height_m(body, site, rung)`, and §11's SL10 row is stronger for it, because SL10 asks for a function of the ADDRESS and the address is what the signature now carries. **But the unbend cost does not exist.** Every production caller already holds the site: `Site { face, i, j }` is built for EVERY column of the box, core and halo, by `site_of` (`crates/terrain/src/lattice.rs:59-63,124-140`) and carried in `SampleBox::sites`; the column pass builds it inside `column_field`; the digest's golden chunks build their directions from a face and parameters; and `vertex_position_m` does not call `height_m` at all (`crates/terrain/src/position.rs:34-40` interpolates the box's own column surfaces). So the change is a signature and four call sites, not 0.2–0.4 ms of `unbend` per chunk. §1 gains a row stating this. |
| B2-D8 | Catmull-Rom is undefined at a cube corner, and G-MACRO-CORNER does not test it. | Defect | **FIXED.** §5.4 rule 4 states the corner rule: within `CORNER_BLEND = 2` nodes of a cube corner, use the quintic fade over the single 2 × 2 node cell — which needs no stencil beyond the cell — and blend into Catmull-Rom over the next node with the same quintic weight, so the join is C1 from both sides. COMPUTED, the eight corners cover about 2 200 km² of 141.09 Mkm², 0.0016 % of the surface. G-MACRO-CORNER now asserts the interpolation as well as the D8 neighbour count. |
| B2-D9 | "There is no soft rock in the world" is overstated, and it drives an unnecessary ask. | Defect | **FIXED, and the ask shrinks.** The refuter is right and revision 2 was wrong: the body draws Sandstone, Limestone or Shale (`crates/terrain/src/body.rs:194`) at 20–80 m (`:198`), verified here. §1's row is rewritten, §4.6's `f_rock` becomes a table over SIX substances, §4.10's platform keys on the sediment, and §7.3 rule 2's cap-rock bench is built ON that boundary. §12 D14 now asks only for the DEEP lithology and a joint field. |
| B2-D10 | §11 claims the HR5 gate passes while §14 M7 marks the enabling fact UNMEASURED. | Defect | **FIXED.** Same as A2-D5: §11's HR5 row reads UNMEASURED and names the fork, §14 M7 is rewritten to be a search for a named body, and §12 D20 puts the three lawful branches to the owner — including the one that invents no world, only a test input, which SL5 permits. |
| B2-D11 | A 12–40 s single-threaded solve inside a shard, with nothing said about the tick loop. | Defect | **FIXED, and both owed sentences are written.** §5.3 gains a rule: the solve runs on the injected worker pool ruling S7-6 already provides, never inline on the tick; and until it lands the shard REFUSES chunk requests with a stated reason and a counter, never answering from a partial field. §5.3 also states that wall-clock scheduling does not touch determinism, because the artifact's BYTES do not depend on when the work finished. The refuter is right that this project has already wedged a realm by doubling a tick. |
| B2-W1 | Two of the owner's five named inputs never reach the shape: spin is asked for and unused, and trajectory is never mentioned. | Weakness | **FIXED.** §4.6 adds `f_frost(day_length)` to the erodibility — frost weathering is driven by the day-night temperature swing, which is the owner's word SPIN. §4.9 adds `± Δ_ecc · e` to the ELA's seasonal range, from the `ecc` the forest already draws — the owner's word TRAJECTORY. §1 gains a row for the orbital elements. All five words now reach the shape, and §12 D6's fact list drops from six to five because eccentricity is free. |
| B2-W2 | The first slice is not buildable, and no minimal slice is named. | Weakness | **FIXED.** §12.0 is a new build order in five slices, E1 to E5, each with what lands and what is stubbed. **E1 — the spectrum, the roughness field, the wavelength rule, the cap-rock bench — is buildable TODAY**, needs no artifact, no ask and no new lane, and is what M3's before-and-after picture measures. |
| B2-W3 | The 64 m under the crust is consumed to zero and nobody says what it was for. | Weakness | **FIXED.** §4.13 states the full downward budget (32 + 16 + 16 = 64) and §12 D16 asks the owner rather than assuming, naming the refuter's own candidate — `below_surface_cell` (`crates/terrain/src/chunk.rs:225-231`) needs a rock cell below the deepest cave — and naming where more room can be recovered without moving `floor_m`. |
| B2-W4 | "Real eroded topography is NOT self-similar; its roughness PEAKS at the valley spacing" carries no source. | Weakness | **FIXED.** §4.2.3 says outright that the design uses a BUMP, that the published shape is arguable between a bump and a break, and that **§14 M9 decides which by measurement before the constant is frozen** — the bump is the recommendation, not a finding. §13 open question 2 carries it. The refuter is right that this was the one claim in the document presenting geomorphology as settled to justify the largest change. |
| B2-W5 | The solve's headline is probably still low, and D11 turns on it. | Weakness | **FIXED.** §10's flood row widens to 1.0–3.2 s on the refuter's per-miss model, and the headline becomes 12–30 s and plausibly 45 s. §14 M1 stays a blocker on §12 D3 and D11, and §13 open question 6 names the flood as the term that decides it. |
| B2-W6 | The water surface's second mesh is charged in time and not in memory or draw calls. | Weakness | **FIXED.** §8.3 states the second vertex buffer, index buffer, material and draw call per water-crossing chunk against ruling V10's MEASURED 405 KB per chunk, and §14 M14 is a new measurement of vertices, bytes and draw calls on a coastal chunk. §12 D15's dial changes to the water mesh because of it. |
| B2-W7 | A macro node's area is quoted at its largest value in three arguments. | Weakness | **FIXED.** §4.1 states three numbers and which is which: face centre 67 634 176 m², MEAN 5.74 × 10^7 m², corner about 4.75 × 10^7 m². §6.1 uses the MEAN for the channel-head argument and the MINIMUM for the discharge floor, and never the maximum. |
| B2-W8 | "Facies" is used for two different things. | Weakness | **FIXED.** Same as A2-W6: §2 fixes one meaning, and the NODE / COLUMN split names the two resolutions of the SAME eight values. |
| B2-N1 | "A gentle, featureless spectrum" mis-describes the code; the RMS slope is 19.4°, and the real defect is self-similarity. | Note | **FIXED, and it changed the cure.** §4.2.3 computes the home planet's own RMS slope, 0.3126 = 17.4°, says the surface is NOT gentle, and restates the defect as two things: no characteristic scale, and no contrast. The refuter is right that "gentle" is the word that made revision 2 reach for more amplitude instead of for a roughness field, and §4.2.4 is the field. |
| B2-N2 | §3's vista claim cannot be checked because no rung is named. | Note | **FIXED.** §3 and §7.1 name it: COMPUTED, one pixel subtends 57.5 m at 50 km with 720 rows at 45°, so the far ridge is drawn at rung 5 or 6, and the rung-to-rung disagreement there is 1.2 m or 3.7 m under the new coarsening rule. |

### 16.3 Round 1 answers, retained for the record

Revision 2 answered refutation A's 27 findings and refutation B's 24 findings on revision 1. Those
answers stand except where round 2 overturned them, and the overturned ones are listed here so no reader
follows a stale answer:

| round-1 finding | what revision 2 answered | what revision 3 changed |
|---|---|---|
| A-D2, second half (the discharge byte's range) | KEPT, on a floor of `6.76 × 10^7` and a ceiling of `1.41 × 10^17` | **WITHDRAWN.** The ceiling was wrong by a factor of ten and the floor used the face-centre area. §6.1 re-sizes against `P_MAX = 10 000 mm/yr` and the minimum node area: 36 octaves, 2.5 % on a width. |
| A-B4 / B-D8 (`f_climate` and the greenhouse) | FIXED with the grey slab | Still right; §4.6 adds `f_frost` and the six-substance `f_rock` on top. |
| A-W5 (`U = 0` makes the relief a function of an integer) | FIXED by promising to measure `PASSES` | **REPLACED.** Measuring `PASSES` beside a free `Δt·K0` measures nothing. §4.6 makes `Δt = EROSIONAL_AGE_YR / PASSES` and §12 D9 measures the AGE. |
| A-D4 / the erodibility table (no soft rock in the world) | FIXED by naming five hard bedrocks and asking DOMAIN 02 for a lithology field | **CORRECTED.** The world draws a soft SEDIMENT at 20–80 m. The ask shrinks to the deep case (§12 D14) and the shallow contrast is built today. |
| B-B1 / B-B2 (the client cannot hold the facts) | FIXED by SHIPPING the facts and the artifact | **HALF KEPT, HALF REVERSED.** Stating the 28 bytes is right and stays. Shipping the artifact breaks ruling S7-2, has no lane and reverses SL10; §12 D3 now derives instead. |
| B-B3 (the macro field cannot borrow the cavern seam machinery) | FIXED with shared edge nodes owned by the lower face | **REBUILT.** The premise was false — `face_param` is a cell centre — and §5.4 is rewritten on the true straddling geometry, which needs no ownership rule at all. |
| B-B4 (the 500 m – 5 km band gets no amplitude) | FIXED with the slope spectrum normalised to the relief | **REBUILT.** The normalisation forced a 53° planet. §4.2.3 anchors on the angle of repose and §4.2.4 adds the roughness field. |
| B-D6, final sentence (only the trunk valley survives at macro scale) | KEPT with the coastline's plan shape added | Still right, and §4.10 still says so. |
| B-D4, first half (`ColumnField` holds no halo) | KEPT with the correction that `SampleBox` does | Still right; §7.3 rule 1 now also states that the ring growth is a RESTRUCTURE and recommends the one-sided difference instead (§12 D16). |
