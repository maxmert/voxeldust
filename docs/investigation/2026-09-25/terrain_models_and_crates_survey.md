# Terrain models and crates — what we can simply use, factor by factor

Written 2026-09-25 on branch `game` at `4442fa7` plus the uncommitted step-2b change (GENERATOR_VERSION 21,
ARTIFACT_VERSION 9). Step 2b waits for the owner's word on G-LAKE-SHARE (status §11 item 10). Meanwhile the owner
asked:

> *"While we are waiting, are there crates with some math models that we are simply able to use? For the believable
> terrain generation, using all the factors we are taking into the account."*

**What this file is.**
- It EXTENDS and CORRECTS two surveys of 2026-09-22. It does not repeat them:
  - [`fine_terrain_models_and_crates.md`](../2026-09-22/fine_terrain_models_and_crates.md) §2 (the algorithms) and
    §3 (the crates for the fine scale);
  - [`lakes_and_landscape_models.md`](../2026-09-22/lakes_and_landscape_models.md) §2–§4 (the lake, glacial and
    flow laws) and §5 (the crates for the macro scale).
- It reads against the living status file
  [`terrain_believability_status.md`](../2026-09-24/terrain_believability_status.md) and the R2 design
  [`r2_design.md`](../2026-09-24/r2_design.md).

**How it was made.**
- Five read-only research groups read our code, the licence files and the registries:
  1. the whole-planet landscape models;
  2. hydrology, lakes, rivers and shores;
  3. the crust, the climate, the ice and the craters;
  4. the Rust numerics and geometry;
  5. the games, the engines and the commercial tools.
- One verifier checked 50 of their claims again, in our code and at each source. It REFUTED six claims and
  corrected them. This file carries the corrected forms only.
- Nobody built anything, ran cargo, or edited another file. Nothing here is MEASURED on our world unless it says so.

**Marks.**
- `path:line` = read in the code (the working tree of 2026-09-25; the lines can move).
- VERIFIED = a reader opened the licence file, the registry record, GitHub's licence record or the project's site.
  The source is in brackets: [file], [crates.io], [PyPI], [Zenodo], [GH], [site]. UNVERIFIED = nobody opened it.
- MEASURED = a number that a run printed (ours, or a paper's own table). ARITH = hand arithmetic. DOC = a number from a
  ruling or a design file. UNMEASURED = nobody printed it.

**Verdicts.**
- **ADOPT** — use it as a dependency; the place is named.
- **HARVEST** — port the law or the algorithm onto our integers and our lattice; copy no code.
- **READ** — a reference only.
- **IGNORE** — no use; the reason is named.

---

## 0. THE SHORT ANSWER

**"Simply use" is true for the tools that CHECK our world. It is false for the tools that would MAKE our world.**

1. **No library can make the shape of our world for us.** The server, every player's machine and the graphics card
   must compute the same ground, to the last bit. Every library we found computes with floating-point numbers, and
   those numbers round differently on different machines. Most of the libraries work on a flat map, not on a planet.
   The best whole-planet models carry the GPL licence: a program that holds their code must publish its own code when
   it is given to anyone. None of them holds the laws we already built: the sea level from the crust, the strength of
   the rock, the craters, and the planet's charter from the seed.
2. **The planet's big solve runs on the server only, once for each planet.** A library can live there in principle.
   Three things must come first: the owner's ruling to move that solve out of the shared generator; a test that the
   solve gives the same bytes on every server machine (nobody measured this yet, even for our own code); and a
   published law with a calibration for every number that the library uses. The gain is small. Our solve already uses
   the same fast method as the best library.
3. **About fifteen tools can run today as INSTRUMENTS.** They run beside the game and never ship. Each one gives a
   second opinion on one of our numbers: the lakes' tree, the river network, the bend of the crust, the sunlight at the
   home planet's steep tilt, the age that our craters show. Two small geometry libraries can measure the shorelines for
   the step that we build now.
4. **The real prize is the laws, not the code.** Six published laws fix gaps that our own code states:
   - rain that dries out far from the sea;
   - a crust that bends with a raised rim beside a mountain range;
   - glaciers that cut along the rivers' own paths;
   - the true size curve of the craters;
   - an evaporation that reads the sunlight;
   - a flat that drains away from high ground.

   We write each one on our own integers, measure it, and the owner rules on it.
5. **Two earlier beliefs were wrong.** One library that we called free to copy (RichDEM) is GPL. And we are not the
   first to put rivers on a cube-shaped planet: a small new project did it, but it blurs its face seams, and our law
   refuses a blur.

**Example.** A pilot stands on a lake shore on the home planet. The shore under her boots comes from our own solve on
the server and from our own code on her machine. A checker such as Whitebox can read a copy of that ground and say
"this lake spills at the same height in my model". It can never put the shore there itself.

---

## 1. EACH OF OUR FACTORS, AND THE BEST THING OUTSIDE FOR IT

The factors are the ones the generator models today (read in `crates/terrain/src/*.rs` and `crates/recipe/src/*.rs`),
plus the later arcs 8e, 8f and 8o. Appendix A holds the full record of every item (the link, the language, the
licence, the dates, the signal, the number type, sphere or flat).

**Where an item can live, and the laws each place touches.**

| mark | the place | the laws it touches |
|---|---|---|
| **I** | an INSTRUMENT: a census line, a bench or a script beside the game; it never ships a byte into the world | none of the shape laws; only the owner's word on each new tool ("never adopt a new library unilaterally") |
| **H-S** | a HARVEST into our once-per-body SOLVE (our own code, integer or under the float fence `Gf`) | SL5 (a new GENERATOR_VERSION; every pin moves on THE world), T9 (a calibration body for every number), HR5 (100 % coverage) |
| **H-R** | a HARVEST into the SHARED RECIPE (the server, every client, the card) | SL10, F7/F8 (integer only, one source for the card), HR5, T9 |
| **D-S** | a library as a DEPENDENCY inside the server-only solve | the isolation rule (an owner ruling), the cross-host pin (NOT MEASURED today, §2.4), T3, T9, F6, HR5, the licence |
| **C** | the client DRAWING (the look, never the shape) | SL10 ("the client only renders") |

**The general finding.** The shared recipe (H-R) can take NO external crate:
`tests/tests/crate_isolation.rs:317-326` asserts that `vd-terrain` and `vd-recipe` link no external crate, and
`:342-349` bans `glam`, `libm`, `noise`, `nalgebra` and `rapier3d` there by name. The solve lives in `vd-terrain`
(`crates/bins/src/artifact_worker.rs` calls `vd_terrain::solve::solve_full`), and the client links `vd-terrain`. So a
dependency in the solve (D-S) needs a ruling first. Every useful row below is therefore I or a HARVEST.

### 1.1 The body, the crust and the plates

| our factor (our law today) | the best thing outside | verdict | where | the law it touches |
|---|---|---|---|---|
| **The charter** (gravity, spin, tilt, water, air, age, insolation; `physics/src/worldgen/charter.rs`) | Nothing makes a charter. climlab (MIT) checks the sunlight that a charter implies (§1.4). | IGNORE for the charter | — | — |
| **Plates and provinces** (one Voronoi snapshot, a boundary kind, one bump profile for each kind; `land.rs:452-483`) | Cortial et al. 2019, *Procedural Tectonic Planets* (paper; plates that move over time). `minos` (Rust, MIT): plates on a cube map. | READ both | a plate history would be H-S | SL5 (a new world), T9 |
| **Crust thickness and the sea** (the freeboard law, Wise 1974, `land.rs:136`) | No library holds it. England & McKenzie 1982 (a thin viscous sheet) is a published law for crustal thickness, named in the paper list of `ymir` (read only). | READ | H-S, if ever | SL5, T9 |
| **The ocean floor's depth** (a ridge is a bump; no cooling law: a reader searched `land.rs`, `craters.rs`, `solve.rs`) | The ocean-floor age–depth cooling law behind GPlates' age grids. It is a published law, not a library. | READ; a candidate law | H-S | SL5, T9 |
| **The flexural rebound** (the removed rock averaged up the pyramid to the level nearest α, then two nine-node means, `solve.rs:1281-1330`, `SMOOTH_PASSES` = 2 at `:97`; α in the LINE-LOAD form, `land.rs:266-281`) | ★ Landlab `Flexure` (MIT): a Kelvin-function (`kei`) kernel, with the 2-D α. | I first (the oracle), then HARVEST the kernel as a table | I, then H-S | SL5, T9 |
| **Uplift** (computed in the solve) | Every landscape model takes uplift as an INPUT map. | IGNORE | — | — |

★ **Two findings on the rebound (ARITH, confirmed by the verifier).**
- Our α is the line-load form `(4D/Δρg)^¼`. A 2-D load needs `(D/Δρg)^¼`, which is smaller by √2. Landlab's own
  doctest shows the pair (119 966 m and 84 829 m). For the home planet (Te 36 789 m, g 9.818): α = 126 km as a line
  load and 89 km as a 2-D load. The nearest pyramid level moves from 131 km nodes to 65.5 km nodes.
- Our rebound is a positive mean, so it has no forebulge: no low plain beside a range. The `kei` kernel gives both the
  forebulge and the moat.
- A verifier's note, UNVERIFIED as a defect: our α uses Δρ = mantle − crust (500 kg/m³), the sediment-filled case.
  With the mantle's density alone, α is 78.6 km (line) or 55.6 km (2-D). Both pick the 65.5 km level. The design's
  intent (`03_erosion_rivers.md` §4.7) needs a check. The Landlab oracle answers both questions in one run.

**Example.** A hull flies over the belt. With the kernel, a long low plain lies beside the range, and a river runs
along it, as the Ganges plain lies beside the Himalaya. Today the ground only rises smoothly toward the range.

### 1.2 Erosion, and the ground's shape at the node

| our factor (our law today) | the best thing outside | verdict | where | the law it touches |
|---|---|---|---|---|
| **Stream power** (implicit, Braun & Willett 2013, `solve.rs:1161`; 40 passes, `:69`) | goSPL, fastscapelib (both GPL-3.0; the same method). `H-Schott/StreamPowerErosion` (MIT): the GPU kernel of Schott et al. 2023. Barnes 2019 (paper): a level-ordered parallel sweep, exact for any thread count. | READ the models. HARVEST the GPU kernel form (for R3 and the F9 card builder). HARVEST Barnes 2019 only if the solve needs threads. | H-R (R3); H-S | SL10, F6, SL5 |
| **Erodibility from rock strength** (Sklar & Dietrich 2001, `strata.rs:181`) | Landlab `Lithology` / `LithoLayers` (MIT): a K for each layer, with no strength law. | READ | — | — |
| **Hillslope and thermal** | Landlab `LinearDiffuser`, `TaylorNonLinearDiffuser` (MIT). ★ Perron et al. 2008/2009 (code in `Tadpole`, MIT): the valley spacing comes from the ratio of creep to channel cutting. | READ the diffusers. HARVEST Perron's law as R2's T9 calibration of the valley spacing. | H-S (R2) | T9 |
| **Deposition** (the cut laid in the first hollow downstream, W7) | goSPL (river and marine), Landlab `SPACE` (MIT). | READ | — | — |
| **Glacial erosion** | §1.4. | | | |

On Perron: our docs cite Perron as a MEASUREMENT (the spectral break at the valley spacing,
`docs/investigation/2026-09-08/landforms/01_reference_target.md:553`, `:1302`). The code does not use the law as a
calibration. The reference picture's "valleys 1–5 km apart" can come from this law instead of from a chosen number.

**Example.** R2 must put valleys on the belt's flank. With Perron's law, the valley spacing comes from the rock's creep
and the river's cut on that flank. Without it, a person chooses a spacing, and T9 refuses a chosen number.

### 1.3 Water: the sea, the lakes, the flats, the receivers, the rivers, the shore

| our factor (our law today) | the best thing outside | verdict | where | the law it touches |
|---|---|---|---|---|
| **The running and connected sea** (W8, B7) | Nothing. No model solves its sea level. | — | — | — |
| **The lakes** (Fill–Spill–Merge with a Langbein budget, `lakes.rs`) | The Fill–Spill–Merge reference (MIT, C++) and its Python wrapper `jkingslake/fillspillmerge` (MIT). WTM (GPL-3.0; Callaghan & Wickert 2025: lakes with evaporation). | ★ I: the oracle of the tree of hollows. READ WTM. | I | none |
| **The flats** (a summed chord toward lower ground; Barnes' second field left out on purpose, `lakes.rs:344`) | Barnes 2014 / Garbrecht & Martz 1997: the second gradient, "away from higher ground". goSPL's 2026 docs name the banding defect. | HARVEST the second field; measure it in step 5 | H-S | SL5, T9 |
| **The receivers and the grid lean** (D8; G-GRID RED at 1.205) | Whitebox Next Gen (Rust, MIT/Apache): FD8, Quinn, Qin, MDInf, D∞, Rho8. D8-LTD (Orlandini 2003) and its path-independent form (Li 2022, UNVERIFIED). `subsphere` (a hex control). | HARVEST candidates for step 5. `subsphere` as I. | H-S; I | SL5 (a hex world is a new world) |
| **The river width** (Leopold & Maddock 1953, `river.rs:12`; the solve only since W16) | ★ Frasson et al. 2019 database and GRWL (both CC BY 4.0 DATA). | ADOPT as DATA in an instrument: the T9 calibration body of the width law | I | T9 |
| **Outlets and deltas** (step 6) | Lindsay 2016's least-cost breach (Whitebox's Rust code, MIT). Nienhuis et al. 2020 (the delta's class from the river, wave and tide fluxes). `pyDeltaRCM` (MIT). | HARVEST | H-S | SL5, T9 |
| **Meanders** (retired from drawing by W16) | `meanderpy` (Apache-2.0), `aparis69/Meandering-rivers` (MIT), CHILD (GPL). | READ | — | — |
| **The shoreline** (step 4) | No library gives the law. `geo` (MIT/Apache) and `imageproc` (MIT) give the bench's measures. | ADOPT both as `vd-bins` dev-dependencies (defaults off) | I | none |
| **One water depth over the ground** (later: R3, 8o) | GraphFlood (the licences disagree; treat it as GPL). | READ | — | — |

On the flats (the verifier's limits): goSPL's cure lives in a pre-processing script (`scripts/fill_mesh_pits.py`), not
in its run. It diagnoses a Priority-Flood + ε fill, where the gradient follows the flood order. Our flats already use the
distance toward lower ground, so only the second field is new. Its effect on G-GRID is UNMEASURED.

**Example.** A crater bowl on the plain fills to its sill. Today its water leaves toward the nearest sill point. With
the second field its water also turns away from the rim, and the bands on the flat bend toward the sill.

### 1.4 The climate, the ice and the craters

| our factor (our law today) | the best thing outside | verdict | where | the law it touches |
|---|---|---|---|---|
| **The rain** (the Tetens supply × the Hadley belt × a lift from ONE upwind neighbour 8 km away, `climate.rs:398`; *"no fetch, no continental term"*, `:737`; the gain names Smith 2003 but runs one plus eight times the upwind slope, clamped to [0.1, 4], `:124-130`) | ★ LFPM, Hergarten & Robl 2022, *GMD* 15:2063 (the paper is open; the code, OpenLEM, is GPL-2.0-or-later [site]). Smith & Barstad 2004 (`orographic-precipitation`, MIT). | HARVEST the LFPM law. READ Smith & Barstad. | H-S | T9 (the calibration body UNVERIFIED), SL5 |
| **The Hadley belts at a steep tilt** (Held & Hou, fitted to Earth's 30° edge, `climate.rs:316`) | ExoPlaSim (GPL-2.0): a real general-circulation model for any tilt. Run as a separate program. | I | I | none |
| **The annual temperature by latitude** (the second Legendre term only, `climate.rs:284-289`) | climlab (MIT): `daily_insolation` for any orbit, `EBM_seasonal`. | I; if the oracle confirms, HARVEST the exact annual integral (a closed form, T3) | I, then H-S | SL5 |
| **Potential evaporation** (Earth's value × the air's water capacity, `climate.rs:604`) | Priestley–Taylor (`pyet`, MIT; Landlab `pet`, MIT). It reads the insolation, the albedo, the pressure and the temperature. | HARVEST, and I | H-S | T9 |
| **Runoff** (Turc–Pike, `climate.rs:610`) | Nothing needed: the law is one line. | — | — | — |
| **The ice line and glacial erosion** (Egholm 2009 along the water's receiver tree, `solve.rs:1401-1470`; Ohmura's ELA) | ★ SIIM (MIT, 2026): glacial and river erosion on ONE implicit receiver tree, sub-grid carving, closed-form profiles. Hergarten 2021 (*ESurf* 9:937), the law. OGGM (BSD-3) for calibration numbers. iSOSIA (Egholm's model) READ. | HARVEST SIIM's n = 1 law or Hergarten's; SIIM's closed forms as I | H-S (at R2's level); I | SL5, T9 |
| **The craters** (Neukum 2001 chronology, one slope D⁻², Pike's shapes, `craters.rs:80-84`) | ★ craterstats (BSD-3): the 12-coefficient production polynomial, and a model-age fit. Cratermaker (GPL-3.0) READ. | HARVEST the 12 numbers (the owner's choice); craterstats as I | H-S; I | SL5, T9 |

Three facts under this table:
- **The rain has no memory of the sea.** A continent's interior gets the coast's rain, and a rain shadow is one 8 km node
  deep. LFPM sweeps two linear equations (vapour, cloud water) along the wind in linear time, at about 2.4 × the cost of
  the simplest stream-power step (the paper's own figure). Its paper states that the Smith–Barstad model cannot make a
  high plateau drier than the plain before it; LFPM can. Its example transport lengths are 119–1 191 km.
- **The home planet's tilt hides a real profile.** The tilt is 55.1° (`home.rs:114`, cos 586/1024). There our second
  term is 0.0055, so our annual profile is flat to ±0.5 %. Two independent integrations of the exact daily-mean
  insolation (ARITH) give the pole 1.044 × the mean, latitude 40° 0.975 ×, the equator 1.020 ×: a W shape of about
  7 %. The temperature effect is UNMEASURED (a reader estimates 3–4 K). The tilt stays (ruling B5); the question is our
  profile, not the tilt.
- **The crater curve.** Tied at 1 km, the published Neukum polynomial gives 18 % fewer craters over 16 km than our D⁻²
  (ARITH, two readers: the ratio 0.816). Its mean slope is −2.23 from 16 to 300 km, −1.26 from 5 to 16 km and −2.75
  from 64 to 128 km. Our chronology constants match craterstats exactly.

**Example.** A pilot crosses the belt from the windward side. Today the lee valley is dry for one node and then wet
again. With LFPM the high plateau behind the range stays a steppe for hundreds of kilometres, and its hollows are dry
pans, not lakes.

### 1.5 The fine ground under the node (R2, R3)

| our factor (our law today) | the best thing outside | verdict | where | the law it touches |
|---|---|---|---|---|
| **The noise spectrum** (8a; the integer noise in `vd-recipe`) | No integer noise crate exists in Rust. ⚠ `fastnoise2` picks a SIMD level at run time, like `simdnoise`. | IGNORE every noise crate; add `fastnoise2` to the named ban list | — | SL10 |
| **R2's amplification** (Schott et al. 2024; the integer operators in `recipe/src/amplify.rs`, "THE BENCH ONLY") | `MultiScaleErosion` (MIT; already read for the operators). `StreamPowerErosion` (MIT). | Already HARVESTED; HARVEST the second | H-R | SL10, F7/F8 |
| **The shape of the levels** | ★ Proland (BSD-3 [site]): each finer level = the parent level upsampled + a STORED residual + noise. | HARVEST the DESIGN (R2/R3, and the 8d authored override as stored residuals) | H-R | SL10 |
| **The mesh** (`terrain/src/extract.rs`: exact integer vertices, edges owned across face seams) | `fast-surface-nets` (f32, seams by padding). `transvoxel` (transition cells). | IGNORE the first. READ the second against our skirts. | — | — |

**Example.** The belt at rung 7 today is the 8 192 m node plus noise. Proland's shape says: store what erosion made at
1 024 m (R2), and add the noise only below that level. The ripples that the owner calls dunes then sit on a real valley
floor, not on a node square.

### 1.6 The later arcs (8e, 8f, 8o)

| our factor | the best thing outside | verdict | where | the law it touches |
|---|---|---|---|---|
| **Biomes (8e)** | Whittaker and Holdridge (tables; WorldEngine holds Holdridge under MIT). Köppen needs the rain's seasonality, which we do not compute. | READ. The row already ships the temperature, the rain, the aridity and the province (`artifact.rs:115-136`), so Whittaker and Holdridge need no SL6 ask. The Whittaker polygons' copyright is UNVERIFIED. | H-S or C | T9 |
| **Soil depth (8e)** | Landlab `weathering`, `depth_dependent_diffusion` (MIT): soil made under its own depth, and soil creep. | HARVEST the laws | H-S | T9 |
| **Sun on the slope (8e)** | Landlab `radiation` (MIT). | HARVEST later | H-S or C | T9 |
| **Cliffs, overhangs (8f)** | `aparis69` (MIT), in fine survey §3.6. | unchanged | — | — |
| **The current (8o)** | `charney` (Rust, MIT/Apache; shallow water on an icosahedral sphere; one day of commits). | READ | — | — |

### 1.7 The numbers under every factor

| our need | the best thing outside | verdict | where | the law it touches |
|---|---|---|---|---|
| Fixed point in the recipe | `fixed` 1.31.0: its 64-bit product widens to i128 (`arith.rs:673`), its root is a `while` loop (`sqrt.rs:172`), its division is an i128 `/`. The card has no 128-bit word (`recipe/src/wide.rs:1-8`), and rust-gpu once miscompiled a loop root (`isqrt(1)` read 0; `recipe/src/root.rs:24-38`). | IGNORE for the recipe | — | SL10, F7/F8 |
| An oracle for our integer roots | std `u64::isqrt` (no crate, Rust ≥ 1.84); `num-integer` (the exact fifth root that `amplify.rs` lacks). | ADOPT as test oracles | I | none |
| An oracle for `Gf`'s exp and ln (`gf.rs:164-229`; about ten values tested today) | `astro-float` (MIT, correctly rounded), `libm` (MIT; already in `Cargo.lock` through `glam`), `core-math` (MIT, C). | ADOPT `astro-float` or `libm` as a dev-dependency oracle | I | none |
| Threads for the solve | `std::thread::scope` (allowed today: `terrain/clippy.toml:44-50` bans only the clocks, `sleep` and the default hashers). `rayon` (not in the lock). | The no-crate option first, only after a profile | H-S; D-S | F6 (a share), SL10 (the bytes may not depend on the thread count) |
| An FFT (flexure, rain) | `rustfft`: the default planner switches to AVX, SSE4.1 or NEON at run time. | READ; never in the solve without the scalar planner and a pin on every host | — | SL10 |
| Sphere grids | `subsphere` (triangles and hexagons only, zero dependencies); `hexasphere`, `h3o`, `a5`, DGGAL. | `subsphere` as I; IGNORE the rest | I | none |

---

## 2. THE BIG OPTION: A WHOLE LANDSCAPE MODEL AS THE MACRO SOLVE

The owner's words "using all the factors" point here: is there ONE model that does everything, which we can run in
place of our solve? This section answers that question.

### 2.1 What exists, and what it covers

Three whole-planet models exist (already in lakes survey §5.1), and two small projects joined them in 2026. In the
matrix, "no" means a reader stated the lack; "—" means no reader reported the factor either way.

| our factor | goSPL | fastscapelib | Landlab | minos | Arda | **ours** |
|---|---|---|---|---|---|---|
| The charter from the seed | no | — | — | — | — | **yes** |
| Plates and provinces | plate advection (an input) | — | faults only | crack-propagation plates, fossil belts | plates at 4 km | Voronoi + boundary kinds |
| Crust thickness, a solved sea | no | — | — | — | — | **the freeboard law, the connected sea** |
| Flexure | ★ global, spherical harmonics (2026) | — | `Flexure`, gFlex (flat) | — | — | a block rebound |
| Uplift | an input | an input | an input | — | — | computed |
| Stream power | yes (three forms) | yes | yes | Davy–Lague | implicit | Braun & Willett |
| Erodibility | two rock types | — | a K per layer | tuned constants | — | **rock tensile strength** |
| Hillslope | yes | ADI, raster only | yes | — | creep, talus | yes |
| Deposition | river and marine | — | `SPACE` | Davy–Lague | — | the first hollow downstream |
| Lakes | pit filling with lake volumes | an MST sink router (no lake water) | `LakeMapperBarnes` | an ε-fill (no lake water) | an annual water balance | Fill–Spill–Merge + Langbein |
| Climate | ★ orographic rain on the mesh (rain is an input map) | — | PET, radiation | a wind bake, moisture | Köppen at 1 km | insolation, lapse, Turc–Pike, Hadley, season |
| Ice | ★ diagnostic glacial erosion (2026) | — | no | — | — | Egholm + the summer ice line |
| Craters | no | — | no | — | — | **yes** |
| Grid | any triangulated sphere | HEALPix (conda, Linux and macOS) | icosphere (small levels tested) | cube map, **seam blur** | flat | **cube-sphere, exact seams** |
| Numbers | f64, PETSc, MPI | f64 | f64 NumPy | f64/f32 | **integer** | integer routing + fenced f64 laws |
| Licence | GPL-3.0 | GPL-3.0 | MIT, with a REQUIRED GPL dependency (`py-richdem`) | MIT | Apache-2.0 | ours |

**No model covers our factors.** The widest one (goSPL) covers about half of them and lacks the charter, the solved sea,
the crust and the craters. Arda is the only integer code, and it is flat. Its own README (line 330) lists as OPEN
"repetitive parallel drainage, angular shorelines, and large rectangular regional basins" — our G-GRID, our teeth and our
square lakes. So a harvest from Arda cannot cure them.

### 2.2 The ways to use one, and the laws each way touches

| way | what it means | the laws it touches | what the owner must rule |
|---|---|---|---|
| **A. Link it** (FFI) into the server | the model runs inside the shard | GPL code enters a binary; SL10 (the solve lives in the one generator); the cross-host pin (Clang fuses a·b + c on arm64 but not on baseline x86-64; UNMEASURED for these libraries); HR5 (our shim to 100 %; the library's branches stay outside our gate) | the licence, and everything in B |
| **B. A subprocess** (Python, PETSc, MPI on every shard pod) | the shard calls the model and reads its output | SL10 (the solve becomes a stored bake, not a function a host can recompute); SL5 (every body goes through the same stack); F6 (goSPL wants MPI ranks; the 10 M-node paper used 168 CPUs); T9 (every K, m, D, ELA and orographic time needs a law); T3 (a lost store must re-solve to the same world) | all of these; goSPL ships no binary wheels ("sdist only") |
| **C. A bake by a FOREIGN tool** (goSPL, Gaea, World Machine) for chosen bodies | we load the tool's output | SL5 refuses it: a foreign tool is a second generator. SL9: the bodies are unbounded, so we cannot bake them all by hand. T9: the tools' knobs are chosen numbers. | refused by the law; not proposed |
| **D. An INSTRUMENT** (an oracle) | we export our start surface and compare the tool's answer with our census | none; nothing enters the game and no code is copied | only whether a GPL tool may run on our own machine |
| **E. OUR OWN prebake** (r2_design §6.4) | the same `bake_tile`, run outside play on the whole machine | lawful and already designed: G-AMP-PIN proves the bytes equal; the home planet is prebaked, every other body bakes on demand | nothing new (R2's own questions) |

The verifier corrected an earlier reading here: a bake is not "impossible". A bake by a FOREIGN tool is unlawful (SL5)
and unbounded (SL9). A prebake by OUR OWN code is lawful and already in the R2 design.

### 2.3 The gain, stated honestly

- **Speed: none.** Our sweep is already Braun & Willett's one-pass method. The whole solve takes 141–171 s (DOC, status
  file, five runs), and that time also holds the flood, the hierarchy, the lakes and the climate. fastscapelib runs the
  same O(n) method (ARITH). goSPL's whole-planet run used 168 CPUs (VERIFIED, Salles et al. 2023, more than 10 M nodes
  at about 5 km); the paper states no wall time.
- **Features: some, and each one is a law.** goSPL has what we lack: marine deposition, groundwater, a global flexure
  with a forebulge, orographic rain on the mesh, glacial erosion with routed ice. Each one is a published law that we
  can harvest (§1). None of them needs the library.
- **A second opinion: yes.** That is way D's job, not the solve's.
- **One supporting fact.** goSPL's authors REMOVED their full shallow-ice solve in 2026, because it diverged and
  over-thickened continental ice at coarse resolution (`docs/ICE_SHEET_SUMMARY.md`, VERIFIED). That supports our
  closed-form ice law at 8 km nodes.

### 2.4 The cost, law by law

- **The isolation rule and SL10.** The solve lives in `vd-terrain`, and the client links it. A library in the solve
  therefore needs a ruling: split the solve into a server-only crate (the worker in `vd-bins` is the precedent), or
  relax the rule for `vd-terrain`.
- ★ **The cross-host pin is NOT MEASURED today — even for our own solve.** The verifier refuted the belief that
  `home_artifact_pin` proves the bytes on every target:
  - `just artifact-pin` (`justfile:828-829`) runs the pin on the host only.
  - `scripts/terrain_legs.sh` runs only the `vd-seed` and `vd-terrain` tests in Docker, and those tests read the
    solve's rows from a RECORDED text (`home.rs:22-26`, `golden_home_z.txt`); they do not solve again.
  - The x86-64 leg is emulated (DEFERRED D-TERRAIN-1). DEFERRED records `home_artifact_pin` on the Apple M4 Pro only.

  A shard with an empty store solves again on whatever host it runs on (`artifact_boot.rs:7-12`). So the whole solve's
  bytes across hosts are UNMEASURED. A float library adds four more ways to drift: FMA contraction, run-time SIMD,
  parallel sums and a platform maths library.
- **The solve's own floats.** The solve is integer where it routes the water (the flood, the flats, the receivers) and
  fenced f64 (`Gf`: IEEE add, multiply, divide, square root; no FMA, no libm) where it computes laws (the climate, α,
  the crater laws; `solve.rs:26` "A SOLVE MAY DIVIDE"). A library must meet the same fence, or the pin fails. The
  reason to refuse a float library is the isolation rule and the drift, not "the solve holds no float".
- **T3.** A library upgrade is a new world, so its version pins forever. A lost store re-solves, and the lake that a
  player camps beside must come back at the same height.
- **SL5 and the size.** Every body goes through the same stack. One planet's artifact is 88.7 MB of rows (10 bytes ×
  8 871 936 nodes, ARITH; `artifact.rs:101`) plus the pyramid (5.9 MB, MEASURED, `artifact_store.rs:12-15`) plus the
  coast mask. (`artifact_store.rs:58` still says "nine a row": a stale comment.)
- **T9.** Every constant a library holds (a K, an m, a D, an ELA, an FD8 exponent, a flat increment, a breach's cost)
  needs a law or a named calibration body.
- **F6.** rayon's default pool and MPI take every core; the solve may take only a share.
- **HR5.** Our shim must reach 100 %. The library's own branches stay outside our gate.
- **The licence.** goSPL and fastscapelib are GPL-3.0. Landlab is MIT but REQUIRES `py-richdem` (GPL-3.0). Our repo has
  no LICENSE file (VERIFIED: none at the root, no field in `Cargo.toml`), and the repo is public. The GPL binds when a
  binary is GIVEN to someone. On the common reading, a server binary that we only run ourselves is not given to anyone;
  a binary that we ever give out is. This is a legal question for the owner, not a technical one (a common reading, not
  advice, UNVERIFIED). The client links the one generator, so no GPL code may ever enter it.

### 2.5 What the owner would rule, and the recommendation

To put ANY model into the solve, the owner must rule six things: (1) move the solve (or one stage) into a server-only
crate; (2) a leg that solves the whole home planet again on linux/arm64 and on a real x86-64 machine and compares the
digest; (3) pin the library's version forever (T3); (4) a thread cap inside F6's share; (5) a calibration body for every
constant (T9); (6) the coverage of our shim (HR5).

**The recommendation: do not.** The gain is a second copy of work that our integers already do, plus a float, plus a
gate. Use goSPL as an INSTRUMENT (way D) if, after R2, the census still cannot explain the dunes. The leg in item (2) is
worth building anyway, because a pod that loses its store takes that path today (§5, R3).

**Example of way D.** We export the home planet's start surface and uplift, cut each lattice quad into two triangles,
and run goSPL over the same age. Then we compare Hack's n, the hypsometry and the lake share with our census. If goSPL
makes the belt branch like a tree where ours makes dunes, the defect is in our solve and not under the 8 192 m node.

---

## 3. THE INSTRUMENTS: WHAT WE CAN SIMPLY USE TODAY

`vd-bins` is not in the Tier-A list (`justfile:12`). Its dev-dependencies never ship (`crate_isolation.rs:30` skips
them). An external program needs no Cargo change at all. So the rows below touch no shape law. Each one still needs the
owner's word (the memory rule).

### 3.1 The candidates, ranked by value to the arc now

| rank | tool | licence | what it checks | our gate or step | how it runs |
|---|---|---|---|---|---|
| 1 | `geo` 0.33.1 | MIT/Apache [crates.io] | the directed Hausdorff distance between two rungs' drawn shores; `MinimumRotatedRect` (G-LAKE-AXIS); `Area` and the ring length (the shape laws of lakes survey §1.5); the XOR area of two shores | step 4, the shoreline bench (`shoreline_design.md` §19) | a `vd-bins` dev-dependency, `default-features = false` (the defaults pull `spade`, `earcut` and threads); it adds `rstar` 0.12 beside our 0.13 |
| 2 | `imageproc` 0.27.0 | MIT [crates.io] | connected components (G-PERCHED, G-LAKE-CLUSTER, G-LAKE-ISLAND on the DRAWN set); the exact Euclidean distance transform (a D4-DIST-ERR line against the design's first-order distance); a one-cell dilation; contours | step 4 | a `vd-bins` dev-dependency, `default-features = false` (the defaults turn rayon ON); `nalgebra` 0.35, `rand` 0.10 and `getrandom` 0.4 enter the lock anyway |
| 3 | Whitebox Next Gen (tools crate unpublished) or the `whitebox-workflows` 2.0.6 wheel | MIT OR Apache-2.0 [file; PyPI] | the fill height at every node (FillDepressions, FloodOrder, DepthInSink); D8 accumulation and the longest upslope path (G-HACK); the least-cost breach (G-BREACH); stream orders | steps 2b, 5; G-HACK, G-BREACH, G-HYPSO, G-LAKE-COUNT | an external program on one exported face |
| 4 | The Fill–Spill–Merge reference, or `jkingslake/fillspillmerge` | MIT [file; GH] (it builds against RichDEM's GPL headers: run it, never vendor it) | for each hollow: the pit height, the spill height, the area and the volume at the spill — values that do not depend on the tie order | step 2b (the tree under G-LAKE-SHARE) | an external program |
| 5 | Landlab `HackCalculator`, `ChiFinder`, `Flexure` | MIT [GH; PyPI] (it installs `py-richdem`, GPL: fine to run) | Hack's n; χ; the rebound, both α forms and the density (§1.1) | G-HACK; the rebound | Python, outside the build |
| 6 | climlab 0.9.2 | MIT [PyPI; GH] | the annual insolation profile at the 55.1° tilt; the seasonal energy balance | `seasonal_amplitude_k`, the annual profile | Python |
| 7 | craterstats 3.6.10 | BSD-3 [file; PyPI] | the model age of our crater list and area; it must equal the retention age, and about the system's age on the moon | G-CRATER | Python |
| 8 | SIIM `siim.analytical` 0.9.10 | MIT [GH; PyPI] | the closed-form long profile of a glacial valley | G-BUZZSAW, the ice | Python |
| 9 | pyet 1.5.0 | MIT [PyPI] | Priestley–Taylor and other PET forms | the PET | Python |
| 10 | Frasson 2019 + GRWL data | CC BY 4.0 [Zenodo] (attribution owed) | a band of width against discharge, and the meander wavelength | a G-RIVER-WIDTH line (T9 for `river.rs`) | a data file |
| 11 | `subsphere` 0.7.1 | MIT/Apache [crates.io] | the same stated cone routed on a hex lattice, beside `grid_bias.rs`'s rotation score | step 5 (G-GRID's hex question) | a `vd-bins` dev-dependency |
| 12 | `astro-float` 0.9.6 or `libm` 0.2.16 | MIT [crates.io; file] | `Gf::exp`, `ln`, `powf` against correctly rounded values over millions of inputs | the climate's laws | a dev-dependency oracle |
| 13 | ExoPlaSim 3.4.2 | GPL-2.0 [GH; PyPI] | the Hadley belts and the belt rain at 55.1° | the belt law | a separate program; its run time on our charter is UNVERIFIED |
| 14 | goSPL v2026.7.14 | GPL-3.0 [GH; PyPI] | a whole-landscape second opinion (§2.5) | G-HACK, G-HYPSO, G-LAKE-SHARE | PETSc + MPI outside the build; a heavy install |
| 15 | World Machine (build 4065, macOS and Linux since July 2026) | closed, paid ($299 Pro) | a picture of industry erosion on one of our tiles | G-LOOK (a picture, not a number) | a desktop tool; "repeatable output" across machines UNVERIFIED |

### 3.2 The rules for every instrument

- **The export.** An instrument writes one face of the rows (1 216 × 1 216 nodes) or one rung-7 square as an ESRI ASCII
  grid. That is plain text, so no crate is needed. Whitebox, Landlab and pysheds read it.
- **The limits.** Their cells are square with √2 diagonals; our chords are the lattice's own, so the receivers can
  differ at near-ties. A face edge is not a seam for them: compare only the hollows whose spill lies inside the face,
  with a margin.
- **Compare what does not depend on the tie order.** The fill height, the spill height, the area and volume at the
  spill. Our heap keys on `(height, index)`; theirs uses another order. The lake LEVELS differ by design: FSM spreads a
  finite volume, and our budget balances inflow against evaporation. The oracle checks the tree, not the water.
- **A GPL tool may RUN on our numbers.** Its code may never enter the repo.
- **One job at a time.** Each new dev-dependency costs build time on this machine, and only one build runs at a time.

**Example.** The census exports face PosX of the rows. Whitebox fills it, and the Fill–Spill–Merge reference builds its
tree of hollows. A census line then reads either "every hollow inside the face spills at the same height in both", or
it names the one hollow on the belt where our flood and theirs disagree. (No such run exists yet; the line is the form,
not a result.)

---

## 4. WHAT CHANGED SINCE THE 2026-09-22 SURVEYS

Each row names the earlier text and the fact now. A reader of the earlier surveys should read these rows first.

| # | the earlier claim, and where | the fact now | mark |
|---|---|---|---|
| 1 | `richdem` is "MIT and maintained"; HARVEST (fine §0, §3.1, §3.6 row) | **GPL-3.0** (`LICENSE.txt`); last push 2024-06-24; last PyPI release 0.3.4 of 2018-07-13. READ ONLY. | VERIFIED [file; GH; PyPI] |
| 2 | `Barnes2016-ParallelFlowAccum` is MIT; harvest (fine §3.6) | **No licence file**, archived. Also no licence: Barnes2013-Depressions, Barnes2013-FlatSurfaces, Barnes2016-ParallelPriorityFlood, Barnes2019-Landscape (the upstream of `oxscape`). READ ONLY; the papers are the prize. | VERIFIED [GH] |
| 3 | Fill–Spill–Merge is "licence-clean" (lakes §5.3) | FSM and DepressionHierarchy are MIT, but DepressionHierarchy pulls RichDEM (GPL-3.0) as a git submodule. A port is clean (`lakes.rs` copies none of it); a vendored build is not. | VERIFIED [file] |
| 4 | Landlab is MIT (lakes §5.1, §5.5) | MIT, but `pyproject.toml` REQUIRES `py-richdem` (GPL-3.0), and `PriorityFloodFlowRouter` is raster-only and wraps RichDEM. The claim "only DepressionFinderAndRouter works on the icosphere" is not in the tutorial: UNVERIFIED. | VERIFIED [file] |
| 5 | WhiteboxTools is "maintained" (lakes §5.2) | The repo is marked **legacy** (last release v2.4.0, 2024-05-22). The successor `whitebox_next_gen` (2026-03-31) is **MIT OR Apache-2.0**; its tools crate has `publish = false`. | VERIFIED [file] |
| 6 | "No standalone Rust priority-flood crate" (lakes §5.2) | Partly false: `hydro-analysis` (f64 ndarray), `sedlink-core`, `surtgis-algorithms`, `oxigeo-terrain` (2025–2026; all float, all flat). The verdict does not change. A Rust port of Fill–Spill–Merge still does not exist. | VERIFIED [crates.io] |
| 7 | "On a cube sphere we would be first" (lakes §5.2) | **False.** `minos` (Rust, MIT, 2026-08-12) runs a priority flood, MFD and Davy–Lague stream power on a 256² × 6 cube map, and joins the faces with a seam BLUR. We are the first with EXACT seams, as far as we found (UNVERIFIED beyond minos). | VERIFIED [GH; source read] |
| 8 | "No fixed-point terrain code exists" (fine §3.1 finding 2, §3.7; lakes §5.2) | Partly false: **Arda** (Rust, Apache-2.0, 2026-08-24) has integer erosion and lakes, and a golden world on Linux, macOS and Windows. It is flat, one month old, has one author, and is not a crate. No integer NOISE crate exists. `vd-recipe` still has no competitor on a sphere. | VERIFIED [file; GH] |
| 9 | "No prior art on the integer question" (fine §3.7) | True for noise and terrain. False for fixed-point transcendentals: `fixed_analytics` (MIT, 2026). It is built on `fixed` (i128, loops), so it is an oracle only. | VERIFIED [crates.io] |
| 10 | `alice-physics` cited as precedent (fine §3.7) | **AGPL-3.0-or-later.** A blocker for code. | VERIFIED [crates.io] |
| 11 | `oxscape_erode`: "harvest, and read first" (lakes §5.2, §5.5) | Down to **READ**: 248 lines, f64, raster only, `powf`, and a Newton loop that stops on a tolerance (`tol` 1e-3) even at n = 1. Our sweep is already its integer form. | VERIFIED [source read] |
| 12 | goSPL's feature list (lakes §5.1) | Out of date. 2026 added diagnostic glacial erosion, orographic rain on the mesh, global flexure by spherical harmonics, pit filling that is partition-invariant "to within a negligible residual (order 10⁻⁶ of nodes)", and groundwater. The shallow-ice solve was removed. | VERIFIED [file] |
| 13 | Salles et al. 2023: "UNVERIFIED, figures from search snippets" (lakes §5.1) | Now VERIFIED from the full text: more than 10 M nodes, 21 M cells, about 5 km, 168 CPUs; no wall time. | VERIFIED [PMC] |
| 14 | eSCAPE (lakes §5.1 family) | The repo returns 404; goSPL replaced it. | VERIFIED [GH] |
| 15 | fastscapelib (lakes §5.1) | No commit on the main branch in 2026; its triangle mesh is 2-D only (`trimesh.hpp:102`). | VERIFIED [GH; source] |
| 16 | `rstar`: "the one crate worth ADOPTING, if the origin work needs one" (lakes §5.2, §5.5) | Already a workspace dependency (`Cargo.toml:56-57`, `vd-core` `child_index`; owner-approved 2026-09-04). | VERIFIED [code] |
| 17 | Memory note: "petgraph and rayon are not deps" | `rayon` is absent from the lock. `petgraph` 0.8.3 IS in the lock through Bevy (not a direct dependency). `hexasphere` 16.0.0 is in the lock through `bevy_mesh`. | VERIFIED [Cargo.lock] |
| 18 | The ban list names `simdnoise` (fine §3.1) | Add `fastnoise2` (run-time SIMD choice). `rustfft`'s default planner does the same. | VERIFIED [README; docs.rs] |
| 19 | `cargo-gpu` as its own repo | Archived; merged into rust-gpu, which released v0.10.0-alpha.1 (2026-04-17). `justfile:841` already installs from the rust-gpu repo. | VERIFIED [GH] |
| 20 | "The climate model has no season at all" (lakes §3.2) | Stale: the code now has the season (`climate.rs:428-475`, ruling B2 step 2). | VERIFIED [code] |
| 21 | symbios-ground's claim: libm gives bit-identical output (fine §3.2) | Supported by reading libm 0.2.16's hardware paths: each one is correctly rounded on our targets; x87 appears only on i586. The measurement on our hosts is still owed. | VERIFIED [source]; UNMEASURED on our hosts |
| 22 | (new) the H-Schott code | `H-Schott/StreamPowerErosion` (MIT) is the code of Schott et al. 2023, beside `MultiScaleErosion` (2024). | VERIFIED [GH] |
| 23 | (new) glacial erosion code | SIIM (MIT, 2026-07-15) is the first permissive glacial eroder on the river tree. | VERIFIED [GH; PyPI] |
| 24 | (gap) river width calibration | No earlier survey named Earth data for the width law. Frasson 2019 and GRWL do (CC BY 4.0). | VERIFIED [Zenodo] |

**Example.** A copy of RichDEM's tiled flow accumulation, taken under the belief in row 1, would have put GPL code into
the one generator — and so into every player's client.

---

## 5. RECOMMENDATIONS, RANKED

Ranked by value to the believability arc NOW (step 2b waits, step 4 is in design v2, R2 is designed), against the cost
and the laws. Each one ends with the question the owner must answer. Nothing here is a request to adopt a library
without that answer.

**R1 — The two bench libraries for step 4.** Add `geo` and `imageproc` as `vd-bins` dev-dependencies with the defaults
off. They give step 4's slice 0 its shore distances, its components and its exact distance transform, with no law
touched. *Owner: may the shoreline bench use `geo` and `imageproc` (dev-only, never shipped)?*

**R2 — The oracles on one exported face.** Whitebox Next Gen, the Fill–Spill–Merge reference and Landlab's
`HackCalculator`, run outside the build. They check the fill heights, the tree of hollows under G-LAKE-SHARE, and Hack's
n — independently, before the owner rules on 4.828 %. *Owner: may a dev script run these three tools (Landlab installs
a GPL library; we run it, we copy nothing)?*

**R3 — A leg that solves the whole home planet on another host.** Not a library, but this survey found the gap: the
artifact pin runs on the Mac only, and a pod that loses its store re-solves today. Run `home_artifact_pin` on
linux/arm64 now (Docker) and on a real x86-64 machine when one exists (D-TERRAIN-1). Every server-only library question
waits on this number. *Owner: may we add this leg to the gates?*

**R4 — LFPM as a candidate for status §11 item 10, choice C.** Our rain has no memory of the sea; a wet interior feeds
inland lakes. Harvest the law onto the solve, run the census with it, and read G-LAKE-SHARE again. The link to the lake
share is UNMEASURED, and the calibration body must be named (T9). *Owner: is LFPM a candidate law under choice C,
measured before any number moves?*

**R5 — The rebound: the oracle first, then the kernel.** Run Landlab's `Flexure` against our rebound (the 2-D α and the
density in one run). If it confirms, harvest the `kei` kernel as a table: the forebulge and the plain beside a range.
*Owner: may the oracle run; and if it confirms, may the rebound change (a new generator version)?*

**R6 — The sunlight at 55°.** Run climlab on the home planet's charter. If the pole reads warmer than latitude 40°,
replace the second-term profile with the exact annual integral (a closed form, T3). ExoPlaSim for the belts comes
later. *Owner: may climlab run; and if it confirms, may the profile change?*

**R7 — The T9 calibrations that published data can give.** Perron's valley spacing for R2; Frasson 2019 and GRWL for
the river width law; craterstats' 12 numbers for the crater curve (18 % fewer craters over 16 km). *Owner: which of the
three become calibration bodies, and D⁻² or the Neukum polynomial?*

**R8 — The glacial eroder on the river tree for R2 or 8e.** SIIM's n = 1 law or Hergarten 2021, with sub-grid carving,
where a 1–3 km valley glacier becomes visible (R2's 1 024 m level). *Owner: does this belong to 8d (with R2) or to 8e?*

**R9 — The flats' second field in step 5.** Barnes 2014 / Garbrecht & Martz 1997, measured on G-GRID next to the MFD
candidates from Whitebox's code. The effect is UNMEASURED. *Owner: may step 5 measure it beside the MFD choice?*

**R10 — Priestley–Taylor PET for 8e.** It reads the sunlight, the albedo and the pressure, which the charter holds.
*Owner: 8e, or earlier as a choice-C candidate beside R4?*

**R11 — Two ban-list entries.** Add `fastnoise2` and `rustfft`'s default planner to the named ban list beside
`simdnoise` and the default hasher. *Owner: agree?*

**R12 — What NOT to do.** No whole landscape model in the solve (§2.5); goSPL only as an instrument, and only if the
census cannot explain the dunes after R2. A hexagonal world lattice only if G-GRID stays red after step 5 (SL5). No code
from Arda, Veloren, World Orogen or any GPL project. *Owner: no answer needed unless the owner wants the big option.*

**Example.** R1 and R2 touch no law, and their cost is tool work only (UNMEASURED). After them, the owner reads beside
each step-4 picture a line of the form "the drawn shore moved at most N coarser cells at the swap", and beside
G-LAKE-SHARE a line of the form "Whitebox and Fill–Spill–Merge agree with our tree on K of M hollows".

---

## 6. WHAT COULD NOT BE VERIFIED

**Sources that did not open.**
- Li et al. 2022, *WRR* (HTTP 403): the path-independent D8-LTD.
- The No Man's Sky GDC slides; the Elite Odyssey forum recaps (403 or truncated).
- The full text of the two high-tilt Hadley papers (Ferreira et al. 2014; arXiv 1903.11656): abstracts only.

**Claims that stay UNVERIFIED.**
- Which Whitebox Next Gen tools are Pro-only. Even `pro_stream_tools.rs` reads `LicenseTier::Open`, so the tier label is
  not the gate.
- The LFPM paper's calibration body. OpenLEM's licence comes from its site (GPL-2.0-or-later), not from a file.
- ExoPlaSim's run time on our charter; goSPL's and Landlab's times on our lattice; Landlab's eroders on the icosphere.
- The copyright of the Whittaker polygons (digitised from a textbook figure).
- Nienhuis 2020's code and data availability.
- World Machine's "repeatable output" across machines (the notes say "between builds").
- Arda's "Freeman MFD, p = 1.1, in a Q32 table" and "45–50 min, 1.76 GiB"; SurtGIS's claim of identical output across
  platforms.
- The licences of `H-Schott/StreamPowerErosion`'s submodules (they matter only for a copy, not for a law).
- Hergarten 2021's upper cost figure (2.7 ×); the paper gives about 1.7 ×.
- The density in our α (§1.1): a plausible inconsistency, not a measured defect.
- The GPL's reach over a server binary that is never given out (§2.4): a common reading, not advice.

**UNMEASURED, and owed before anyone claims them.**
- The whole solve's bytes across hosts (§2.4, R3).
- libm's bytes on x86-64 against aarch64; `FftPlannerScalar`'s bytes across hosts.
- The parallel share of the solve's 141–171 s.
- `Gf::exp` and `Gf::ln` error over the climate's input range (about ten points tested today).
- rust-gpu on a 128-iteration R2 loop.
- fastscapelib's FMA drift between arm64 and x86-64.
- Every cost and every believability effect named in this file.

**Example.** Our census gives a lake on the belt its level on the Mac. Until R3's leg runs, nobody can say that a pod
on x86-64 that re-solves the home planet puts that lake at the same level.

---

## APPENDIX A. THE REGISTER OF EVERY ITEM

Columns: the item and its link; the language; the licence and its mark; the last release or push, and the signal
(★ = GitHub stars, dl = crates.io or PyPI downloads); our factors; the number type, the determinism and the grid; the
verdict and the place (§1's marks). Items already in the 2026-09-22 surveys and unchanged are not repeated
(`symbios-ground`, `MultiScaleErosion`, `erosion-rs`, `bevy-hydrology`, `sphere_terrain`, `cdshealpix`, `realpix`, the
six noise crates, the Bevy terrain crates, `aparis69/*`, the weigert repos).

### A.1 Whole-landscape models

| item | lang | licence | release / push · signal | our factors | numbers · grid | verdict |
|---|---|---|---|---|---|---|
| [goSPL](https://github.com/Geodels/gospl) | Python + Fortran, PETSc, MPI | GPL-3.0-or-later [GH; README] | v2026.7.14 (2026-07-10); push 2026-09-23 · 89★, 100+ commits in 2026 | uplift, stream power, hillslope, deposition, lakes, glacial erosion, orographic rain, global flexure, groundwater | f64; iterative solves with tolerances; partition by core count; any triangulated sphere | READ; I (way D) |
| [fastscapelib](https://github.com/fastscape-lem/fastscapelib) | C++ (xtensor), Python | GPL-3.0 [GH] | v0.3.0 (2025-06-25); no main commit in 2026 · 46★ | stream power, routing, the MST sink resolver | f64; `std::pow` and a Newton loop for n ≠ 1; raster, 2-D trimesh, HEALPix | READ |
| [fastscape](https://github.com/fastscape-lem/fastscape) | Python | BSD-3 [GH], but imports GPL `fastscapelib_fortran` | v0.1.0 (2023-09-25) · 76★ | the same | f64; flat | IGNORE |
| [Landlab](https://github.com/landlab/landlab) | Python | MIT [GH; PyPI]; requires GPL `py-richdem` | 2.11.0 (PyPI 2026-04-06; GH release 2026-08-31); push 2026-09-21 · 444★ | stream power, SPACE, diffusion, landslides, lithology, faults, flexure, PET, radiation, lakes, Hack, χ, soil | f64 NumPy (run-time SIMD); icosphere and flat | HARVEST the laws; I |
| [terrainbento](https://github.com/TerrainBento/terrainbento) | Python | MIT [GH] | 2024-09-11 · 21★ | 28 model combinations | flat | READ |
| eSCAPE | Python | GPLv3 (JOSS) | repo 404 | — | — | IGNORE |
| [Badlands](https://github.com/badlands-model/badlands) | Python | LGPL-3.0-or-later [README] | v2.3.1 (2026-05) · 192★ | stream power, marine, waves, carbonates, flexure | f64; flat | READ |
| [CHILD](https://csdms.colorado.edu/wiki/Model:CHILD) | C++ | GPL v2 [site] | 2010 tarball; page 2020-09-16 | stream power, meandering, storm rain | planar TIN | READ (8f meanders) |
| LSDTopoTools2, MuddPILE | C++ | GPL-3.0 [GH] | 2023-06, 2022-02 | χ analysis | raster | READ the χ method |
| [TTLEM](https://github.com/TopoToolbox/TTLEM) | MATLAB | none (a 1-byte LICENSE.md) | — | a landscape model | raster | READ ONLY |
| CAESAR-Lisflood, HAIL-CAESAR | C++ | GPL-2.0, GPL-3.0 [GH] | 2025-08 · 45★ | floods with sediment | catchment, years | IGNORE (the wrong time scale) |
| [Veloren `erosion.rs`](https://gitlab.com/veloren/veloren/-/blob/master/world/src/sim/erosion.rs) | Rust | GPL-3.0 [file] | push 2026-09-24 · GitHub mirror 7 585★ | Cordonnier 2016 stream power with lakes, diffusion, soil, rivers | f64/f32 + rayon; flat, 32 m | READ the paper only |
| [H-Schott/StreamPowerErosion](https://github.com/H-Schott/StreamPowerErosion) | C++ + GLSL | MIT [GH] | push 2026-09-17 · 52★ | uplift + stream power on the GPU (Schott et al. 2023) | f32 GPU; flat | HARVEST (H-R) |
| [oxscape_erode / oxscape](https://codeberg.org/feefladder/oxscape) | Rust | MIT OR Apache-2.0 [crates.io] | 0.1.0 / 0.1.1 (2025-12-31); commit 2026-07-13 · 26 / 175 dl | implicit stream power; a parallel flow-graph walk | f64, `powf`, tolerance loop; raster | READ (provenance caution: the upstream has no licence) |
| Barnes 2019, level-ordered traversal (paper) | — | a paper | — | an exact parallel sweep for any thread count | — | HARVEST the paper if threads are needed |
| [fastlem](https://github.com/TadaTeruki/fastlem) | Rust | MPL-2.0 [GH] | archived | a Delaunay LEM | f64; flat | READ |
| [Tadpole](https://github.com/MITGeomorph/Tadpole) | MATLAB, C | MIT [GH] | push 2020 | Perron 2008/2009 valley spacing | flat | HARVEST the law (T9, R2) |
| [iSOSIA (annvrowan)](https://github.com/annvrowan/isosia); [iSOSIA (davidlundbek)](https://github.com/davidlundbek/iSOSIA) | C, MATLAB | none; GPL-3.0 [GH] | push 2026-07-31; 2021 | Egholm's glacial model | f64; flat | READ ONLY |
| [DAC](https://github.com/yanyanwangesd/DAC_versions) | Fortran | NOASSERTION [GH] | — | Goren 2014 | — | READ |
| Cortial et al. 2019 ([PDF](https://perso.liris.cnrs.fr/eric.galin/Articles/2019-planets.pdf)) | paper | — | — | plates on a sphere over time | sphere | READ |
| [minos](https://github.com/owenyuwono/minos) | Rust | MIT [GH] | created 2026-08-12, push 2026-08-20 · 6★ | plates, priority flood, MFD (p = 6), Davy–Lague, wind, moisture | f64/f32, tuned constants; cube map, seam blur | READ |
| [Arda](https://github.com/DiceMasterIO/arda) | Rust | Apache-2.0 [file; GH] | created 2026-08-24, push 2026-09-25 · 1★, 98 commits | plates, implicit stream power, creep, talus, MFD, basins, lakes by water balance, Köppen | **integer** (`HeightMm` i32, u128 litres); golden world on three OSs; flat | HARVEST methods (a power law as a baked table; exact volumes); never ADOPT |

### A.2 Hydrology, lakes and flats

| item | lang | licence | release / push · signal | our factors | numbers · grid | verdict |
|---|---|---|---|---|---|---|
| [WhiteboxTools (legacy)](https://github.com/jblindsay/whitebox-tools) | Rust | MIT [file] | v2.4.0 (2024-05-22); push 2026-05-26 · 1 205★ | fill, breach, D8/D∞/FD8, orders | f64; flat | READ; its binary as I |
| [Whitebox Next Gen](https://github.com/jblindsay/whitebox_next_gen) | Rust | MIT OR Apache-2.0 [file] | created 2026-03-31, push 2026-08-31 · 101★; wheel `whitebox-workflows` 2.0.6 (2026-06-14) | 47 hydrology, 12 flow, 26 stream tools | f64, rayon, std `HashSet`, `ureq`; flat | I; HARVEST the MFD family and the least-cost breach |
| [hydro-analysis](https://crates.io/crates/hydro-analysis) | Rust | MIT [crates.io] | 0.2.3 (2026-09-25) · 2 611 dl | fill, breach, D8 | f64 ndarray; flat | IGNORE |
| [sedlink-core](https://crates.io/crates/sedlink-core) | Rust | MIT OR Apache-2.0 [crates.io] | 0.8.0 (2026-08-29) · 179 dl | flood, two-gradient flats, D8/D∞/MFD | float; flat | IGNORE (READ its flats) |
| [SurtGIS](https://github.com/franciscoparrao/surtgis) | Rust | MIT OR Apache-2.0 [crates.io; file] | push 2026-09-25 · 16★, 1 828 dl | fill, D8, accumulation, geomorphons | float, rayon; flat | IGNORE (an instrument candidate at most) |
| [oxigeo-terrain](https://crates.io/crates/oxigeo-terrain) | Rust | Apache-2.0 [crates.io] | 0.2.4 (2026-08-18) · 145 dl | flood, D8, D∞ | float; flat | IGNORE |
| [Fill–Spill–Merge](https://github.com/r-barnes/Barnes2020-FillSpillMerge) + [DepressionHierarchy](https://github.com/r-barnes/Barnes2019-DepressionHierarchy) | C++ | MIT [file] (pulls GPL RichDEM) | push 2024-05-24 · 21★ | the tree of hollows, spill, merge | double; flat | already HARVESTED; I |
| [jkingslake/fillspillmerge](https://github.com/jkingslake/fillspillmerge) | Python + pybind11 | MIT [GH] | push 2026-08-25 · 0★ | Callaghan's FSM fork | flat | I (the easiest oracle) |
| [RichDEM](https://github.com/r-barnes/richdem) | C++, Python | **GPL-3.0** [file] | push 2024-06-24; PyPI 2018-07-13 | flood, flats, breach, accumulation | flat | READ ONLY |
| [WTM](https://github.com/KCallaghan/WTM) ([paper](https://gmd.copernicus.org/articles/18/1463/2025/)) | C++ + PETSc | GPL-3.0 [GH] | push 2026-08-14 | FSM lakes with evaporation and groundwater | flat | READ |
| [pysheds](https://github.com/mdbartos/pysheds) | Python | GPL-3.0+ [GH; PyPI] | 0.5 (2025-08-14) · 902★ | fill, flats, D8/D∞/MFD, HAND | flat | READ (run on our numbers at most) |
| [TauDEM](https://github.com/dtarb/TauDEM) | C++ + MPI | GPL-3.0 [README] | v5.5.0 (2026-05-23) · 276★ | the constant-drop test for a channel head | flat | READ the law |
| [libtopotoolbox](https://github.com/TopoToolbox/libtopotoolbox) | C | GPL-3.0 [GH] | push 2026-09-21 | a flat routed along its centre line (`gwdt`), knickpoints | flat | READ the idea |
| GraphFlood ([paper](https://esurf.copernicus.org/articles/12/1295/2024/)) | C++, Python | GPL-3.0 / MIT / CC BY 4.0 (they disagree: treat as GPL) | 2025-07 | one steady water depth | flat | READ (R3, 8o) |
| Bangerth 2026 ([arXiv 2606.12800](https://arxiv.org/abs/2606.12800)) | paper | — | 2026-06-11 | parallel routing | — | READ (not needed) |
| D8-LTD (Orlandini 2003), LTD-PI-LD (Li 2022) | papers | — | — | the grid lean of D8 | — | HARVEST candidates (Li 2022 UNVERIFIED) |

### A.3 Rivers, deltas, data and shore geometry

| item | lang | licence | release / push · signal | our factors | numbers · grid | verdict |
|---|---|---|---|---|---|---|
| Frasson et al. 2019 ([data](https://zenodo.org/records/2579161)) | data | CC BY 4.0 [Zenodo] | 2019-02-27 | width, slope, area, meander wavelength, discharge | — | ADOPT as data (I) |
| GRWL ([data](https://zenodo.org/records/18624542)) | data | CC BY 4.0 [Zenodo] | 2026-02-12 | global river widths | — | ADOPT as data (I) |
| [meanderpy](https://github.com/zsylvester/meanderpy) | Python | Apache-2.0 [GH; PyPI] | 0.2.0 (2026-02-13) · 174★ | meander migration | float | READ |
| [Meandering-rivers](https://github.com/aparis69/Meandering-rivers) | C++ | MIT [GH] | push 2024-02-05 · 19★ | Paris et al. 2023 | float | READ |
| [pyDeltaRCM](https://github.com/DeltaRCM/pyDeltaRCM) | Python | MIT [GH; PyPI] | 2.3.0 (2026-07-30) · 23★ | a reduced-complexity delta | float; random parcels | READ; HARVEST as a seeded integer port only |
| Nienhuis et al. 2020, *Nature* 577:514 | paper | — | 2020 | the delta's class | — | HARVEST the law (step 6) |
| Mogensen & Isenbecker ([arXiv 2504.08383](https://arxiv.org/abs/2504.08383)) | paper + code | CC0 | 2025-04-11 | rivers and fjords by subdivision | float; flat | READ (R3 precedent) |
| [geo](https://crates.io/crates/geo) | Rust | MIT OR Apache-2.0 [crates.io; Cargo.toml] | 0.33.1 (2026-04-20) · 1 940★, 22.9 M dl | shore measures | f64 | ADOPT as a dev-dependency (I) |
| [imageproc](https://crates.io/crates/imageproc) | Rust | MIT [crates.io] | 0.27.0 (2026-06-02) · 980★, 13.3 M dl | components, EDT, contours | image rasters | ADOPT as a dev-dependency (I) |
| [contour](https://crates.io/crates/contour) | Rust | MIT OR Apache-2.0 [file] | 0.13.1 (2024-04-30) · 123 k dl | marching squares | f64 | optional |
| contour-isobands | Rust | AGPL-3.0-or-later [crates.io] | 2024 | isobands | — | IGNORE (licence) |
| marching-squares, edt, dt, distance-transform, fdsm | Rust | MIT/Apache | 2018–2026 | isolines, distance fields | float | IGNORE |

### A.4 The crust, the plates and the flexure

| item | lang | licence | release / push · signal | our factors | numbers · grid | verdict |
|---|---|---|---|---|---|---|
| [pyGPlates](https://github.com/GPlates/GPlates) / [GPlately](https://github.com/GPlates/gplately) | C++, Python | GPL-2 [PyPI; GH] | 1.0.0 (2025-03-26); 2.0.0 (2025-07-14) · 95★ | the real Earth's plates, seafloor age | f64; sphere | IGNORE the code |
| [plate-tectonics (platec)](https://github.com/Mindwerks/plate-tectonics) | C++ | LGPL-3.0 [GH] | push 2025-11-23 · 95★ | moving plates | float; flat wrapped | READ |
| [WorldEngine](https://github.com/Mindwerks/worldengine) | Python | MIT [GH] | 0.20.0 (2025-11-18) · 1 071★ | plates, rain (noise), Holdridge | float; flat | READ (8e table) |
| [tectonics.js](https://github.com/davidson16807/tectonics.js) | JS | CC-BY-4.0 [file] | push 2020-08-22 · 233★ | a plate cycle | float; sphere | READ |
| [tectonics.cpp](https://github.com/davidson16807/tectonics.cpp) | C++ | **CC BY-NC-SA 4.0** [file] | push 2026-07-29 · 19★ | the same | float; sphere | READ ONLY (non-commercial) |
| [World Orogen](https://github.com/raguilar011095/planet_heightmap_generation) | JS | GPL-3.0 [GH] | push 2026-09-20 · 103★ | tectonics, glacial carving, rain, gyres | float; sphere | READ (8e step order) |
| [Frey](https://github.com/new-umechan/Frey) | Rust + TS | MIT [GH] | push 2026-08-23 · 5★ | plates, climate, glaciology, lakes; benches against WorldClim | f32, `HashMap`; sphere | READ |
| [Red Blob planet generation](https://github.com/redblobgames/1843-planet-generation) | JS | Apache-2.0 [GH] | push 2026-06-06 · 168★ | Voronoi plates | float; sphere | IGNORE (what `land.rs` does) |
| [gFlex](https://github.com/awickert/gFlex) | Python | GPL-3.0 [GH] | 1.3.0 (2026-05-27) · 26★ | flexure | f64; flat | READ |
| [Landlab Flexure](https://github.com/landlab/landlab/tree/master/src/landlab/components/flexure) | Python | MIT [GH] | with Landlab | the `kei` kernel, 1-D and 2-D α | f64, scipy; flat | HARVEST (H-S) + I |
| [ymir (FifionRibana)](https://github.com/FifionRibana/ymir) | Rust | proprietary ("All rights reserved") [file] | push 2026-09-24 · 0★ | thin viscous sheet, Airy isostasy, droplets, Whittaker | float; flat | READ its paper list only |
| [Undiscovered Worlds](https://github.com/JonathanCRH/Undiscovered_Worlds) | C++ | GPL-3.0 [GH] | push 2024-09-09 · 105★ | a globe; its README: "the same seed does not always produce the same planet" | float | IGNORE |
| Rust plate toys: vectorTectonic (MIT), platec-rs (LGPL-2.1), plate-tectonics-rust (LGPL-3.0), planetgen-rs (MIT); `use-tectonic-plate`, `use-geology`, `planetsfactory` (repo 404), `khanij` (GPL-3.0-only) | Rust | as listed | 2022–2026, tiny | — | — | IGNORE |

### A.5 The climate

| item | lang | licence | release / push · signal | our factors | numbers · grid | verdict |
|---|---|---|---|---|---|---|
| LFPM, Hergarten & Robl 2022 ([paper](https://gmd.copernicus.org/articles/15/2063/2022/)); [OpenLEM](http://hergarten.at/openlem/) | paper; C++ | paper open; OpenLEM GPL-2.0-or-later [site] | 2022 | rain with long-range transport | linear-time sweep; flat | HARVEST the law (H-S) |
| [orographic-precipitation](https://github.com/fastscape-lem/orographic-precipitation) | Python | MIT [GH; PyPI] | 1.0 (2020-06-24); push 2024-08-12 · 9★ | Smith & Barstad 2004 | numpy FFT, one wind; flat | READ |
| [PISM](https://github.com/pism/pism) + LT plugin | C++, Python | GPL-3.0 [GH] | push 2026-09-21 | linear-theory rain, ice | flat | READ |
| [climlab](https://github.com/climlab/climlab) | Python | MIT [PyPI; GH] | 0.9.2 (2026-03-18) · 248★ | insolation for any orbit, seasonal EBM | f64; 1-D in latitude | I |
| [ExoPlaSim](https://github.com/alphaparrot/ExoPlaSim) | Python + Fortran | GPL-2.0 [GH; PyPI] | 3.4.2 (2025-01-02); push 2026-09-19 · 122★ | a GCM for any tilt | float; sphere | I (a separate program) |
| [speculative-koppen](https://github.com/PHarvey7/speculative-koppen) | Python | MIT [GH] | push 2021-01-06 · 4★ | Köppen from ExoPlaSim | float | READ |
| [pyet](https://github.com/pyet-org/pyet) | Python | MIT [PyPI] | 1.5.0 (2026-05-26) · 165★ | Priestley–Taylor PET | f64; point | HARVEST + I |
| [mapgen4](https://github.com/redblobgames/mapgen4) | TS | Apache-2.0 [GH] | push 2026-03-29 · 882★ | humidity swept in wind order | float; flat | READ |
| [Azgaar FMG](https://github.com/Azgaar/Fantasy-Map-Generator) | JS | MIT [file] | push 2026-09-24 · 6 027★ | winds, moisture, biomes (tuned) | float; flat | READ |
| KoppenClimate (MIT), hylken Köppen maps (GPL-3.0) | Python | as listed [GH] | 2025 | the Köppen rules | — | IGNORE as code |
| [plotbiomes](https://github.com/valentinitnelav/plotbiomes); Holdridge 1967 | R; paper | MIT code [GH]; polygon copyright UNVERIFIED | push 2025-02-12 · 42★ | Whittaker, Holdridge | — | READ |
| [tpt-sci-climate](https://crates.io/crates/tpt-sci-climate) | Rust | MIT OR Apache-2.0 [crates.io] | 0.1.0 (2026-08-22) · 21 dl | a 0-D EBM | f64 | IGNORE |
| [charney](https://crates.io/crates/charney) | Rust | MIT OR Apache-2.0 [crates.io] | 0.1.0 (2026-08-25) · 17 dl | shallow water on an icosahedral sphere | f64, RK4; sphere | READ (8o) |
| rscm-two-layer (Apache-2.0), badal (GPL-3.0-only), hxgm30-eco (Apache-2.0) | Rust | as listed | — | — | — | IGNORE |

### A.6 The ice

| item | lang | licence | release / push · signal | our factors | numbers · grid | verdict |
|---|---|---|---|---|---|---|
| [SIIM](https://github.com/ericdealgeomorph/siim) | Python, Numba | MIT [GH; PyPI `siim-lem`] | 0.9.10 (2026-09-24); created 2026-07-15 · 0★, one author, paper in preparation | glacial + river erosion on one tree; ELA; sub-grid carving; closed forms; FFT flexure | f64; n ≠ 1 is a bracketed Newton with a tolerance; flat | HARVEST the n = 1 law; I |
| Hergarten 2021 ([paper](https://esurf.copernicus.org/articles/9/937/2021/)) | paper | open | 2021 | a glacial stream-power law, U-shaped and hanging valleys | D8; flat | HARVEST the law |
| [OGGM](https://github.com/OGGM/oggm) | Python | BSD-3 [GH; PyPI] | 1.6.3 (2026-04-13) · 265★ | real glaciers' mass balance | f64; per glacier | READ (calibration numbers) |
| [IGM](https://github.com/instructed-glacier-model/igm) | Python, TensorFlow | GPL-3.0 [GH] | push 2026-09-25 · 93★ | a learned ice emulator | GPU float | IGNORE |

### A.7 The craters and the soil

| item | lang | licence | release / push · signal | our factors | numbers · grid | verdict |
|---|---|---|---|---|---|---|
| [craterstats](https://github.com/ggmichael/craterstats) | Python | BSD-3 [file; PyPI] | 3.6.10 (2026-07-15) · 53★ | Neukum 2001 production and chronology; model ages | f64 | HARVEST the 12 numbers; I |
| [Cratermaker](https://github.com/MintonGroup/cratermaker) | Python + Rust | GPL-3.0 [GH] | 2026.9.0a0 (2026-09-22) · 8★ | cratered terrain on an icosphere | f64, `rustfft`, `rayon`, `libm`, `noise` | READ |
| [CTEM](https://github.com/MintonGroup/CTEM) | Fortran | GPL-3 [file] | push 2026-03-14 | cratered terrain | f64 | READ |
| [Landlab `weathering`, `depth_dependent_diffusion`](https://github.com/landlab/landlab/tree/master/src/landlab/components) | Python | MIT [GH] | with Landlab | soil production and creep | f64; flat | HARVEST the laws (8e) |

### A.8 Numerics, sphere grids, meshes, threads, the card

| item | lang | licence | release / push · signal | our need | numbers · determinism | verdict |
|---|---|---|---|---|---|---|
| [fixed](https://crates.io/crates/fixed) | Rust | MIT/Apache-2.0 [file] | 1.31.0 (2026-03-20) · 14.65 M dl | the recipe's words | i128 product, loop root | IGNORE for the recipe |
| [fixed_analytics](https://crates.io/crates/fixed_analytics) | Rust | MIT [crates.io; GH] | 3.1.0 (2026-09-03) · 1 096 dl | fixed-point exp/ln/pow | on `fixed` | READ; an oracle |
| [cordic](https://crates.io/crates/cordic) | Rust | BSD-3 [crates.io] | 0.1.5 (2021-05-16) · 105 k dl | shift-and-add trig | on `fixed`; loops | READ the method |
| `az`, `integer-sqrt`, `fixed-sqrt`, `core_maths`, `rustc_apfloat`, `softfloat` | Rust | MIT/Apache family | — | — | — | IGNORE |
| `corvid_fixed`, `fixed-math-taylor`, `simple-soft-float`; `fixed_trigonometry`; `alice-physics` | Rust | GPL-2.0+; GPL-2.0+; LGPL-2.1+; non-standard; AGPL-3.0+ | — | — | — | IGNORE (licence) |
| [num-integer](https://crates.io/crates/num-integer) | Rust | MIT OR Apache-2.0 [crates.io] | 0.1.47 (2026-08-12) · 703 M dl | an exact fifth root | exact; a float first guess under `std` | an oracle only |
| std `u64::isqrt` | Rust std | Rust's own | stable since 1.84 | the stream-power root | exact by contract | ADOPT as a test oracle |
| [libm](https://crates.io/crates/libm) | Rust | MIT [file] | 0.2.16 (2026-01-24; in our lock) · 480 M dl | `Gf`'s exp/ln | correctly rounded hardware paths; cross-host UNMEASURED | an oracle; replacing `Gf`'s own exp/ln is an owner option (isolation rule + a pin on every host) |
| [astro-float](https://crates.io/crates/astro-float) | Rust | MIT [crates.io] | 0.9.6 (2026-08-07) · 192 k dl, 121★ | reference values | arbitrary precision, correctly rounded | ADOPT as an oracle (I) |
| [dashu-float](https://crates.io/crates/dashu-float); [core-math](https://crates.io/crates/core-math) | Rust; Rust + C | MIT OR Apache-2.0; MIT [crates.io] | 0.6.1 (2026-09-25); 1.3.0 (2026-08-30) | reference values | arbitrary precision; correctly rounded | alternative oracles |
| [rstar](https://crates.io/crates/rstar) | Rust | MIT OR Apache-2.0 [crates.io] | 0.13.0 (2026-05-24) | spatial lookups | integer coordinates allowed | ALREADY ADOPTED (`vd-core`) |
| `kiddo` | Rust | MIT OR Apache-2.0 | 6.3.0 | nearest points | no i64 axes | IGNORE |
| [spade](https://crates.io/crates/spade), [delaunator](https://crates.io/crates/delaunator) | Rust | MIT OR Apache-2.0; ISC | 2.15.1; 1.1.0 (2026) | an irregular mesh | f64, exact predicates; 2-D | READ |
| [delaunay](https://crates.io/crates/delaunay) | Rust | BSD-3 [crates.io] | 0.8.2 (2026-09-15) · 23★ | a spherical Delaunay mesh | exact determinants, stated tie-breaks | READ (only for a ruled irregular mesh, SL5) |
| `stripack`, `d3_geo_voronoi_rs` | Rust | wrapper MIT/Apache (the Fortran core UNVERIFIED); MIT | 2026; 2025 | spherical meshes | — | IGNORE; READ |
| [subsphere](https://crates.io/crates/subsphere) | Rust | MIT OR Apache-2.0 [crates.io; file] | 0.7.1 (2025-05-29); push 2026-09-08 · 5.7 k dl | the hex control | f64, zero dependencies; triangles and hexagons only | ADOPT as I |
| `hexasphere`, `h3o`, `a5`, DGGAL | Rust; C | MIT/Apache; BSD-3; Apache-2.0; BSD-3 | 2026 | sphere grids | float; H3 cells do not nest exactly | IGNORE (DGGAL READ) |
| [fast-surface-nets](https://crates.io/crates/fast-surface-nets); [transvoxel](https://crates.io/crates/transvoxel) | Rust | MIT OR Apache-2.0 | 0.2.1 (2025-01-03); 2.0.0 (2025-11-11) | meshes | f32; seams by padding | IGNORE; READ |
| [fastnoise2](https://crates.io/crates/fastnoise2) | Rust (C++ bindings) | MIT [crates.io] | 0.4.0 (2026-01-21) | noise | run-time SIMD choice | BAN LIST |
| `noise-functions`, `opensimplex2`, `squirrel_noise5` | Rust | MIT/Apache; CC0; CC-BY-3.0-US | 2024–2026 | noise | f32; float; an integer hash | IGNORE; READ; IGNORE |
| [rayon](https://crates.io/crates/rayon) / `std::thread::scope` | Rust | MIT OR Apache-2.0 / std | 1.12.0 / stable | the solve's per-node passes | integer reductions exact in any order; f64 sums change with the split | a server-only candidate after a profile; the no-crate option first |
| petgraph, chili, forte | Rust | MIT/Apache | — | a hierarchy | — | IGNORE |
| [rust-gpu](https://github.com/Rust-GPU/rust-gpu) (+ cargo-gpu, merged) | Rust | MIT/Apache | v0.10.0-alpha.1 (2026-04-17); commits to 2026-09-14 · 3 373★ | the card's recipe | our pin rev 7fa56ad6 | KEEP (already adopted) |
| CubeCL, krnl, wgpu bump | Rust | MIT/Apache | — | GPU kernels | a second kernel source | IGNORE (F8 decision 5) |
| [rustfft](https://crates.io/crates/rustfft), faer/pulp, nalgebra | Rust | MIT/Apache; MIT; Apache-2.0 | 6.4.1; 0.24.4; 0.35.0 | FFT, linear algebra | run-time SIMD; nalgebra banned in `vd-terrain` | READ; IGNORE; banned |

### A.9 Games, engines and commercial tools

| item | what it publishes | verdict and the lesson |
|---|---|---|
| [Outerra](https://outerra.blogspot.com/2009/02/procedural-terrain-algorithm.html) | 90–150 m real elevation data; all detail below it procedural, modulated by slope | READ: its detail starts from eroded ground |
| [Proland](https://proland.inrialpes.fr/) | BSD-3 [site]; stored, procedural, or a mix "to amplify real data with fractal details" | HARVEST the design (§1.5) |
| [SpaceEngine](https://spaceengine.org/manual/making-addons/creating-a-planet) | `erosion` is a 0–1 knob; rivers are noise | IGNORE (the knob that T9 refuses) |
| Elite Dangerous ([80.lv](https://80.lv/articles/generating-the-universe-in-elite-dangerous)) | noise graphs seeded by the planet ID; no landable planet has surface liquid | READ: it skipped water, our open problem |
| Star Citizen ([wiki](https://starcitizen.tools/Planet_Tech_v5)) | data layers: temperature, humidity, geology, soil, sunlight, slope aspect; no documented river tech | READ (8e's layers) |
| Space Engineers ([wiki](https://spaceengineers.wiki.gg/wiki/Modding/Tutorials/Creating_a_Planet)) | six hand-made cube-face heightmaps, 20 m a pixel, seams fixed by hand | READ |
| No Man's Sky, Minecraft | noise; "erosion" is a noise parameter | IGNORE |
| Gaea 2 | Windows only; unitless river knobs | READ |
| World Machine (build 4065, [CG Channel](https://www.cgchannel.com/2026/07/world-machine-software-ships-world-machine-dragontail-peak/)) | macOS and Linux since July 2026; "erosion output now 100 % matches between builds" | I only (§3.1 row 15) |
| Instant Terra, Terragen, Houdini, World Creator | closed tools; Terragen erodes flat patches only | READ |
| [Terrain Diffusion](https://github.com/xandergos/terrain-diffusion) | MIT [GH]; a learned relief | IGNORE (T9: no law; SL10: GPU floats) |

**The industry's lesson (from A.9).** Every engine that looks believable starts its procedural detail from eroded
ground at 20–150 m. Our detail starts from 8 192 m: 55 (Outerra's 150 m) to 256 (Veloren's 32 m) times coarser
(ARITH). The engines that derive a planet from a seed alone ship no real rivers or lakes. This supports ruling B6's
order: ship one eroded level at 1 024 m (R2), then derive below it (R3).

**Example.** A hull over the belt at 120 km sees rungs 5–11. With R2, the two coarsest of them show real drainage in
place of noise, as Outerra's ground shows a measured valley under its fractal detail.
