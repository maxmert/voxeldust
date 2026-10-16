# The rain and the seasonal winds by law — generator 27 (ruling B99 step 1) — design, revision 2

Status: a design; L1 (the sun) and the Earth readers are built on the bench, nothing else. Revision 1 had one adversarial
review (2 blockers, 13 major, 10 minor; kept outside the repo, `/tmp/vd-g27/review_A.md`); revision 2 answers every
blocker and major finding (§10) and takes the review's simpler route, which is also the proven skeleton of the
open-source planet model spacesim (MIT; its `planet-sim/src/climate.rs:303–621`): a seasonal energy balance on a coarse
grid, the vapour carried by the wind and by eddy diffusion on the same grid, the rain out of the vapour, and the
mountains' own rain on the fine grid. Sources: `/tmp/vd-g27/laws.md` (the laws with DOIs), `/tmp/vd-g27/proven.md`
(proven codes and Earth data), the code map (§6), `docs/investigation/2026-10-11/macro_rain_rise_study.md`.

Marks: MEASURED = a run or a paper's measurement; STATED = a source states it; ESTIMATE = hand arithmetic, to be
measured on the bench; READ = read in the code.

## 0. For the owner

Today the home planet is dry for three wrong reasons, and its rain cannot move with the seasons:

1. The rain belt takes away 42 % of the rain on average and never gives it back (a defect).
2. The rain belt is one fixed annual belt. At the planet's 55° tilt the sun's heating moves far into each summer
   hemisphere, so the rain belt must move with it (a missing law, allowed by B36 Q3).
3. The rain falls with height, because each node's rain comes from its own air's water capacity. On a real planet a
   windward slope gets MORE rain up to 2–3 km (a wrong law).

The new chain follows the water from the sea to the ground, season by season:

- the sun heats each latitude by its season;
- an energy balance gives the temperature over land and sea, and the heat the air carries toward the poles;
- the rain belt stands where that heat flow turns (the energy-flux equator), and the surface winds blow toward it, turned
  by the planet's spin, with the monsoon's pull toward a hot summer continent;
- the sea evaporates by its energy; the wind and the storms carry the vapour; it rains out of the air;
- on the 8 km ground, a slope facing the wind takes more of the vapour, and the land behind the crest lies in its shadow.

No number in the chain is a clamp or a knob. Earth's 990 mm a year is no longer an input: on Earth's charter and Earth's
real relief, the law is READ against Earth's measured rain and winds.

**Example.** Today the belt's windward ramp at −41° gets 96 mm of rain a year. With the new laws, in the southern summer
the rain belt stands near the belt, the summer wind carries the sea's vapour up the ramp, and the ramp takes the rain; the
high plateau behind the crest stays drier in the shadow, and the valley under the pass gets a river.

What you see first, with no world change: the home planet's winds in four seasons and its year's rain beside today's, on
the stored relief (step S2, §5). Then the whole macro solve with the new climate on the bench, and its pictures on three
bodies (S3). The world moves once, on your word (S4).

## 1. Binding

- **B99** (the base first; generator 27 is step 1). **B78** (the rain's registered defects). **B36 Q3** (the seasonal
  circulation allowed). **B36 item 1** (no biome class or stand rests on today's climate).
- **T9, B23, B38, B41**: every number from a published law with a stated calibration body; no clamp, no knob; Earth's
  numbers are readings, except a constant the law cannot compute yet, calibrated on Earth and named.
- **B39, B47, B56, B69**: the world converges in time (N and 2N passes, twice the routings, the noise rule); the climate
  every pass.
- **SL10**: one generator crate; the macro solve runs on the planet's shard; the client reads the artifact.
- **SL6**: no new data crosses a realm boundary (§6: the rain byte stays one annual number).
- **B58, B76, B93**: no grid lean (no straight or 45° streaks); the pictures decide; judge from the side.
- **B64, B72, B91, F6**: the same bytes on any number of cores; the solve's threads below the window.
- **B74**: no test suite, gate or coverage run without the owner's word.

## 2. What retires, what stays

Retires from the solve's climate: the belt shape `0.09 + 0.91·(1 − u²)²`; the one fixed annual cell; the supply
`990 mm × e_s(T)/e_s(288 K)` at the node's own air; the lift `1 + 8·slope` and its clamp [0.1, 4]; the rain's ceiling
10 000 mm; the PET `1 000 mm × e_s(T)/e_s(288 K)`; Earth's 990 mm as an input.

Stays: the lapse rate law; Turc–Pike's runoff (Earth's catchments); Ohmura's ELA curve; the aridity byte's axis; the
frost line's form. The client's `ClimateLaws` and its two-mode `summer_k` stay FOR THE CLIENT'S SNOW LINE ONLY until the
look step (the registered defect THE CLIENT'S SUMMER READS A DRY PLANET; the two lines disagree in place until then). The
rain's floor `P_MIN_MM_YR` (1 mm) retires only when the fine levels' zero-water path is measured on the bench (§3.6);
until then it stays a registered cap. The instrument path `solve()` with no climate keeps its 1 mm rain.

**Scope.** The seasonal chain runs on a body with air that orbits its star (`year_s` is its year around the star). An
AIRLESS body keeps today's temperature law (no air carries no heat and no vapour; the seasonal part changes no row the
game reads on it), so the moon's rows and digest do not move. A body with air whose charter year is not its year around
its star (a moon with air) or that is locked to its star has no lawful seasonal sun in the charter today: the chain
REFUSES it with a named error, never a silent wrong (registered; none exists in the world today — the bench lists the
world's bodies with air to prove it).

## 3. The laws

Two grids. **The climate grid** is the macro lattice `coarser(k)` (k = 4: 131 km cells, 34 656 on the home planet; k = 3:
65 km, 138 624): the energy balance, the winds and the vapour. **The node grid** is the 8 km macro lattice (8 871 936
nodes): the mountains' rain and the rows. `k` is a resolution decided by its reading (§5), never a knob. A climate field
reaches a node by bilinear reading between the four nearest climate cells in the face's own coordinates, across a face
edge by the lattice's seam table (a new function of `MacroLattice`; no 131 km blocks reach a node).

### 3.1 L1 — the sun by latitude and season (BUILT, `seasons/sun.rs`)

- The daily-mean sunlight `Q = (S/π)·(h₀·sin φ·sin δ + cos φ·cos δ·sin h₀)`, `cos h₀ = −tan φ·tan δ` (the polar day and
  night are the formula's own domain), `sin δ = sin ε·sin λ`, the season's angle even in time.
- The orbit: a pass (~12 Myr) is far longer than a precession cycle, so the landscape sees the precession-averaged year;
  the year's mean flux is the orbit's time mean `S/√(1 − e²)`. A stated choice: it also averages away the cold summers
  of one precession phase (Milankovitch), which the ice's ELA would otherwise read.
- Fenced: the year's angles from square roots, the hour angle by `Gf::acos`. Its tests: Earth's solstice pole 541 W/m²
  and equator 397.5 W/m², the annual equator 415.5 and pole 172.3 W/m², the global mean `S/4`, 256 against 512 samples.

### 3.2 L2 — the seasonal moist energy balance (North, Short & Mengel 1983's structure; Siler, Roe & Armour 2018's transport)

- `C·∂T/∂t = Q·(1 − α) − (A + B·T) + ∇·((p_s/g)·D·∇h)`, `h = c_p·T + L·RH·q*(T)`.
- **Solved by implicit steps through the year** (the proven route: spacesim, climlab): `N_s` steps a year, each one
  linear solve with the moist diffusivity `(p_s/g)·D·∂h/∂T` taken at the step's start (symmetric, positive: a
  Jacobi-preconditioned conjugate gradient in a fixed order, its stop a stated residual), the cube-sphere's own cell
  areas and chords in the operator (no face-edge print). The periodic year is reached by WARM START: the solve keeps the
  last pass's year (its temperature at each step) and steps from it until two years agree within the reading's
  tolerance (§5); the first pass starts from the annual-mean balance. `N_s` is a resolution read by `N_s` against `2N_s`.
- **The humidity**: over the sea `RH` = 0.8 (Earth's oceans, STATED by Siler et al.); over land `RH` = 0.8 × the land's
  wetness, the previous pass's `E/PET` of Turc–Pike (lagged one pass, no inner loop); `RH` = 0 on a body with no water.
- **The heat capacity** of a cell: its sea share × a 50 m mixed layer (today's constant, Earth's mid-latitude ocean) + its
  land share × (the air column `(p_s/g)·c_p` + the ground's annual skin, today's rock law). Land and sea differ only by
  their capacity (NSM83): a summer continent heats fast — the monsoon's cause.
- `A = Q̄·(1 − α) − B·T_charter` (exact while `α` is fixed: the diffusion conserves energy); `B` = 1.9 W/(m²·K) and the
  Bond albedo `α` are today's. An ice–albedo law and sea ice are REGISTERED (WK97 and NSM83 carry them; not in this step).
- **`D` and the spin**: `D` = 1.16 × 10⁶ m²/s (Siler et al. 2018, ERA-Interim, calibration Earth). The sources DISAGREE
  on the spin (Kaspi & Showman 2015 measured the transport flat below Earth's spin with no tilt; Guendelman & Kaspi 2022
  put the spin into `D`; Ferreira et al. 2014 measured an effective `D` far lower at a 54° tilt). The bench runs `D` and
  `2D` and reports what moves; ExoPlaSim (Q1) judges it. Q3 asks the owner after the bench.
- Outputs per climate cell and step: `T`, `h`, the storage `G = C·∂T/∂t`, the northward heat flow `F`.

### 3.3 L3 — the rain belt and the surface winds of each season

1. **The rain belt** (the rising branch): the ENERGY-FLUX EQUATOR of the step's zonal-mean transport — the latitude where
   `F` turns from north to south (Bischoff & Schneider 2014; Donohoe et al. 2013 measured on Earth's seasons). It
   follows the relief smoothly and needs no constant. Singh 2019's diagnostic is NOT the belt; it is the cell's
   poleward limit only, read as a cross-check (its `ΔT` = 85 K is Singh's measured GCM constant, named).
2. **The meridional wind**, one rule: the Hadley cells carry `F_H = w·F`, `w = exp(−((x − x_rise)/σ_x)²)`, `x = sin φ`,
   centred on the rising branch (Siler et al.'s share, `σ_x` = 0.3, their Earth fit, named); their mass flux is
   `ψ = F_H/g_ms` with the gross moist stability `g_ms = h_T − h(φ)`, `h_T = 1.06·h(φ_rise)` — positive by construction,
   since the rising branch stands at the zonal `h` maximum. The eddy belt carries `(1 − w)·F` by its thermally
   indirect cell. The lower branch moves in the lowest tenth of the column's mass (Siler et al.: below 900 hPa, named),
   so `v = ψ·g/(2πa·cos φ·0.1·p_s)`.
3. **The zonal wind** from the lower branch's balance of the Coriolis turn and the surface drag, `f·v·δ = C_D·|U|·u`
   (`C_D` = 1.3 × 10⁻³, Earth's sea, named; `δ` the lower branch's depth, the same tenth of the column), solved in
   closed form: `u² = (√(v⁴ + 4a²) − v²)/2`, `a = f·v·δ/C_D`. One rule gives the trades, a cross-equator monsoon's
   westerlies, the Ferrel westerlies and the polar easterlies.
4. **The land–sea departure** (the only zonal asymmetry): the boundary layer's pressure follows its temperature
   departure from the zonal mean, and the wind follows the pressure, the Coriolis turn and the same drag (Lindzen & Nigam
   1987, their linear damped balance; the same `δ` and drag, no constant of their own).
5. No longitude sectors and no lag: the energy balance's own heat store gives the lag.

### 3.4 L4 — the evaporation and the potential evaporation (Priestley & Taylor 1972)

- `E = α_PT·Δ/(Δ + γ)·(R_n − G)/L`, `α_PT` = 1.26 (saturated surfaces, Earth), `Δ` the slope of Clausius–Clapeyron at
  the cell's temperature, `γ = c_p·p_s/(0.622·L)`. The SEA'S HEAT STORE `G` comes from L2 (Ferreira et al. 2014 measured
  ~90 % of the seasonal air–sea flux stored locally), so the summer polar sea stores its sunlight and gives it back in
  winter.
- `R_n` (Q2, recommended): Earth's share of the absorbed sunlight that reaches the surface as net radiation, times the
  step's own absorbed flux (a constant the law cannot compute yet; Earth's 98/240, named). With `R_n` calibrated on
  Earth, Earth's global ~1 000 mm/yr cannot fail: the real readings are the zonal profile, the land and the named places.
- **The phase rule**: a frozen surface (the sea below its freezing point) or an air under water's triple-point pressure
  (611 Pa) evaporates nothing (physics, not a clamp).
- Over land L4 gives the PET, which the runoff, the aridity and the next pass's wetness read.

### 3.5 L5 — the vapour and the rain on the climate grid

- The column vapour `W` (kg/m²) per climate cell and season, steady within the season (its life is days, a season is
  weeks): `∇·(W·v) − ∇·((p_s/g)·D_q·∇(W/(p_s/g))) + W/τ = E + E_land`, solved as one sparse linear system a season
  (upwind advection by the low-level wind, eddy diffusion with Siler's `D` — the same constant as L2 — and the rain-out),
  in a fixed order to a stated residual. The eddy diffusion carries the storm tracks' vapour, spreads the convergence
  line over its physical width, and removes any grid lean at the climate grid.
- **The rain-out time** `τ`: the vapour's measured residence time at sea level, 8.9 ± 0.4 days (van der Ent &
  Tuinenburg 2017, MEASURED on Earth; the constant the law cannot compute yet, named), times LFPM's height response
  (Hergarten & Robl 2022: the slow mode's time `(β + 2)·τ_c` with `β = β₀·e^{−H/H₀}`, `H₀ = R_v·T₀²/(L·Γ)` from
  Clausius–Clapeyron and the lapse rate) — the height form only, never the temperature form (the review's M-7: the
  Arrhenius form over a 50 K climate span reverses Earth's measured residence pattern).
- **The saturation**: a column over its saturation `W_sat(T)` condenses the excess at the condensation time `τ_c` = 1 000 s
  (Smith & Barstad 2004, field values). This is the physics of condensation (only above saturation), not a clamp: cold
  air that receives warm vapour rains it out.
- **The land's own evaporation** `E_land`: the previous pass's Turc–Pike evaporation of the node's rain (lagged one pass,
  no inner loop), summed to the climate cell.
- Outputs per climate cell and season: `W`, the rain `P_c = W/τ + condensation`, and the vapour flux.
- READINGS (never targets): Earth's coast-to-interior decay (~600 km over non-forested land, Makarieva et al. 2009, the
  number LFPM was anchored on) and the residence pattern (van der Ent & Savenije 2011: under 7 days in the Congo, ~5 in
  northern Siberia).

### 3.6 L6 — the mountains' rain on the node grid, and the rows

- **The orographic share** (LFPM's short mode at 8 km): per season, each node's rain is its climate cell's rain
  redistributed by the node's own rain-out time `τ(H_node)` along the season's wind: an upwind march on the node graph
  carries the cell's vapour across the cell's nodes, the windward slope (low `β`) rains more, the lee receives the
  depleted column, and the cell's total equals `P_c` (the water is conserved; the excess the slope takes is the lee's
  loss). The wind's vector is split EXACTLY on the two neighbours that bracket it (`v = a₁ê₁ + a₂ê₂`, each neighbour's own
  chord; no arc tangent; the review's B-2 fix 1), and the march relaxes back to the climate cell's own field over the
  eddy diffusion's length `√(D_q·τ)` (the same constants). Lean-free by construction is a CLAIM to measure: the cone bench
  (§5) reads the plume's length and width at 0°, 22.5° and 45°.
- `rain_mm_yr`: the sum over the seasons, whole millimetres (a resolution). The discharge's u64 bound is restated from the
  law: the body's total rain is its total evaporation, bounded by `α_PT` × the absorbed sunlight over `L`; the bound is
  written with the largest body the world can draw, or the discharge widens to `u128` (the build decides by the arithmetic).
- **The zero-water path**: a node whose rain rounds to 0 mm is lawful; the fine levels' handling (`fine/refine.rs`
  `river_slope`, `fine/l1.rs` `network`'s logarithm) is measured on the bench before `P_MIN_MM_YR` retires.
- `pet_mm_yr`: L4's land PET summed over the seasons. `runoff_mm_yr`: Turc–Pike on the year's rain and PET (Milly 1994's
  seasonal term registered as owed). `aridity_q8`: today's axis (its classes were drawn on Penman's PET; PT reads lower
  in hot deserts — a reading).
- `temperature_dk`: L2's annual mean read to the node, less the lapse to the node's air height. `frost_z`: today's form.
- `ela_z`: Ohmura's curve with the warmest quarter's temperature of L2 (the ablation season it was fitted on) and the
  node's annual rain.
- `upwind`: the slot of the year's vapour-flux-weighted wind (no reader today).
- Readers whose numbers move with the rain (the code map, READ): the joint law's discharge, the lakes' budget, the ice,
  the talus angle (aridity), the craters' retention (rain over Earth's 750 mm: G-CRATER moves), the fine levels (the rain
  class), and the benches `macro_bench`, `lake_census`, `crater_probe`, `cover_census`, `erosion_bench`,
  `fine_settle_bench`, `artifact_diff`.

## 4. The rules this design must keep, and how

- **No caps** (B23): every clamp listed in §2 goes; the geometry's polar day and night, the phase rule, the saturation and
  the energy bound are laws. `P_MIN_MM_YR` stays registered until §3.6's zero-water path is measured.
- **Earth only for a constant the law cannot compute yet** (B41), each named with its body in the code: `B`, `D`, `RH`,
  the sea's mixed layer, `α_PT`, `R_n`'s surface share, `σ_x`, `h_T`'s 1.06, the lower branch's tenth of the column, `C_D`,
  the residence time 8.9 d, `τ_c`, LFPM's `β₀` (its own calibration), Singh's 85 K (a cross-check only). Earth's rain,
  winds, deserts and coast decay are READINGS (§5).
- **One generator** (SL10): the laws live in `vd_terrain::seasons` under the float fence; the game's solve calls them only
  in the build (§7); the bench copy calls them first.
- **The same bytes on any cores** (B64): the climate grid's solves on one core in a fixed order; the node grid's seasons in
  parallel, each on one core, summed in their order; a test proves it.
- **No grid lean** (B58): the eddy diffusion at the climate grid; the exact vector split and the cone bench at the node
  grid; the bilinear reading from the climate grid to the nodes.

## 5. The bench and its readings (no world change; ruling B38's one bench)

- **S0 — Earth's data** (BUILT): ETOPO 2022 binned onto the home lattice (the connected ocean 70.65 % against Earth's
  70.8 %), GPCP 1991–2020 (the global mean 983 mm/yr, the land 791 mm/yr on this lattice), NCEP's 10 m winds;
  `climate_bench earth-data`, `vd_bins::earth`; the data in `~/voxeldust_earth/climate/`.
- **S1 — the energy balance and the winds**: Earth's charter on Earth's relief: the winds by month against NCEP's (the
  share of the land and of the sea whose direction agrees within 45°, January and July), the rising branch's track
  against GPCP's zonal rain maximum by month; an Earth-charter aquaplanet's zonal profile. The home planet on its stored
  relief: the rising branch's track and the winds by season. Pictures: the season wind maps.
- **S1b — the controls**: the cone under a steady wind at 0°, 22.5° and 45° (the plume's length and width — B-2); the
  zero-water path of the fine levels (§3.6).
- **S2 — the rain**: Earth's annual and seasonal rain against GPCP (the zonal profile, the land's mean and median, the
  dry share, coasts against interiors, the monsoon's season, the Amazon, the Sahara, India's summer, the Andes' and the
  Himalaya's windward sides). The home planet and one more wet planet of the galaxy (another tilt and spin) on their
  stored relief, beside today's rain. ★ THE OWNER'S FIRST LOOK HERE (the season winds and the year's rain), before S3.
- **THE RESOLUTION READINGS** (each against its double; one tolerance, the world gate's own, B47): the land's rain median
  within 5 %, the land's dry share (under 250 mm) within 1 point, the land's mean within 5 %, the zonal profile's peaks
  within one climate cell; read on the climate alone, on the home planet's stored relief and on Earth's: `k` against
  `k − 1`, `N_s` against `2N_s`, the seasons `M` against `2M`, the periodic year's tolerance against a tenth of it.
- **S3 — the macro solve with the new climate** (the bench copy, a switch in the copy only): 414 passes and the twins,
  828 passes (B39, the noise rule); the land, the mountains, the old plains, the lakes, the rivers, the 8 km valleys'
  straightness (B82); the census and pictures on three bodies (the home planet, the second wet planet, the moon —
  airless, its rows unchanged by §2's scope); the owner's look.
- **S4 — the build** (generator 27): §7, on the owner's word after S3's look.

Every Earth number is a reading. A miss is reported with its cause; no constant is turned toward it.

## 6. What the code map says (READ, 2026-10-15)

- The rain byte stays ONE annual number: the fine levels read only it; the client reads no climate row at runtime; the
  wind and frost rows have no reader outside the climate. The artifact's form stays (ARTIFACT_VERSION 9); no new data
  crosses a realm boundary.
- The world moves: `GENERATOR_VERSION` 27, `ARTIFACT_KEY_GENERATION` 28, the home artifact's digest (four places), the
  sea and its share in `home.rs`, the golden z text, the measured identity word, the declared pin. Every fine tile is keyed
  by the artifact's digest, so every 128 m and 16 m tile solves again by itself; no fine version moves.
- A latent defect, cured in the same change (READ): the shard's artifact worker gives the climate a sea on a body with no
  water (`artifact_worker.rs:55-62`), so the desert world's datum is `i32::MIN` and its stored temperature byte is 0.
- The tests whose numbers follow the rain law change their expected values in the build; the bench copy's
  `RAIN_RENORMALISED` probe and the `ClimateLaws` circulation go with the belt (the client keeps its snow line's words).

## 7. The build (S4, after the owner's look)

1. The game's solve keeps the seasonal state across passes (the warm start) and calls `vd_terrain::seasons`; the old belt,
   lift and supply go; the bench copy's probe goes.
2. The desert world's datum defect cured.
3. The version words, the pins and the golden text recorded (one core), the release pin gate on 14 cores.
4. The tests on the owner's word (B74).
5. The living status file and DEFERRED (THE RAIN LAW'S DEFECTS → 🟩; the new interims registered: sea ice, the ice–albedo
   law, the moon with air and the star-locked body, Milly's seasonal runoff, the client's snow line).
6. The window on the owner's word.

## 8. Questions for the owner

- **Q1 — ExoPlaSim as an outside measuring stick.** A published planet climate model (GPL-2.0, Fortran and Python) run
  OUTSIDE the game on the home planet's charter and land map, never linked or shipped; it reads the rain belts, the winds
  and the rain by month where no paper has run a 55° tilt with a 3-day spin, and it judges the spin question (Q3).
  Price: a Fortran compiler, MPI and its Python package installed outside the repo; 1–3 h a run (ESTIMATE; Apple
  silicon untested by its authors). Recommended: yes.
- **Q2 — the net radiation at the surface** for the evaporation: Earth's share of the absorbed sunlight (one Earth
  constant, named; recommended), or a gray column with a convection model (a new law with its own risks).
- **Q3 — the heat transport's spin law**: asked with the bench's `D`/`2D` readings and (if Q1) ExoPlaSim's.
- **Q4 — the price**: the climate becomes a few seconds a pass (ESTIMATE 2–5 s; measured at S2), so a first solve of the
  home planet grows by ~15–35 min (ESTIMATE), and the whole world solves again once (every planet with air and every fine
  tile).

## 9. Risks

- The circulation at 55° and a 3-day spin is an extrapolation of aquaplanet theory (Q1 checks it).
- The subtropical deserts on the west coasts come from cold sea currents and oceanic highs, which a zonal circulation
  leaves out (registered if the Earth reading shows it).
- The rain-out by one residence time may give belts too broad (Earth's rain rises steeply with the column's saturation,
  Bretherton et al. 2004); the S2 reading decides whether that law joins.
- The cost: measured at S2 before S3.

## 10. The review ledger (revision 1 → 2)

- B-1 (the singular mass flux): `h_T` from the season's rising branch, positive by construction; `ψ` to a speed by the
  lower branch's tenth of the column (§3.3 item 2).
- B-2 (the 45° lean): the vapour on the climate grid with eddy diffusion (§3.5); the node grid's march with the exact
  vector split, a relaxation by the physical length, and the cone bench (§3.6, §5).
- M-1 (the moist linearisation): implicit steps through the year, warm-started (§3.2).
- M-2 (`D` and the spin): the bench reads `D` and `2D`; Q1, Q3.
- M-3 (RH everywhere): the land's RH by its wetness, lagged; RH 0 on a dry body (§3.2).
- M-4 (Singh misread): the belt is the energy-flux equator; Singh only as a cross-check with his 85 K; no sectors (§3.3).
- M-5, M-6 (the singular line; the storm track): the eddy diffusion of the vapour in the first build (§3.5).
- M-7 (β's calibration): the residence time names the sea-level rain-out; LFPM's height form only; Makarieva's 600 km a
  reading (§3.5).
- M-8 (PT): `R_n − G`; the phase rule; sea ice registered (§3.4).
- M-9 (moons, locked bodies): the scope and the refusal; airless bodies keep today's law (§2).
- M-10 (missed readers): listed (§3.6); the zero-water path measured before the floor retires.
- M-11 (unnamed constants): named (§3.3, §4); the bilinear reading is a new lattice function (§3).
- M-12 (resolution readings): stated with the world gate's tolerance (§5).
- M-13 (the cost): no inner fixed point (the recycling lagged); the vapour on the climate grid; measured at S2.
- Minor: m-1 (the eccentricity's mean) fixed in §3.1; m-2 (the u64 bound) in §3.6; m-3 (Earth's global mean cannot fail)
  in §3.4; m-4 (the gray column) in Q2; m-5 (the aridity's PET) in §3.6; m-6 (`τ_c`'s pick barely matters at 8 km;
  the windward maximum a reading, not a promise) in §0 and §3.5; m-7 (the two snow lines) in §2; m-8 (ExoPlaSim on Apple
  silicon) in Q1; m-9 (`Gf::acos`) built; m-10 (`A` while `α` is fixed) in §3.2.

## 11. The bench's first readings (2026-10-15; bench only, no world change)

Built in `vd_terrain::seasons` (bench only; the game's solve still reads `crate::climate`): `sun` (L1), `grid`, `balance`
(L2), `wind` (L3), `vapour` (L4–L5); `vd_bins::earth` and `examples/climate_bench.rs` (modes `earth-data`, `earth-s1`,
`home-s1`, `earth-s2`, `home-s2`). Earth's data in `~/voxeldust_earth/climate/` (ETOPO 2022 60″, GPCP 1991–2020, NCEP R1
10 m winds; decoded from their netCDF files with Python's own zlib, no new tool).

**DECISIONS TAKEN BY MEASUREMENT** (each amends the text above):

1. **THE CLIMATE GRID IS LATITUDE–LONGITUDE** (amends §3's cube grid). MEASURED on the sphere's eigenfunction
   `∇² sin φ = −2·sin φ/a²`: the cube-sphere's two-point flux was wrong by 12–16 % and by a factor of ten at a cube corner,
   an error that does not fall with the cell's size (its cells are skewed). The latitude–longitude grid is orthogonal:
   the error under 10⁻³ at 2° and falling as the square of the cell, energy conserved exactly, the pole's flux nought by
   geometry. A node reads the grid's fields by a bilinear reading in its latitude and longitude (`Gf::acos`).
2. **THE CELLS' EDGES BY THE EDDY LIMIT, CALM AT THE EDGE** (amends §3.3 item 2). The mean cells span from the rising
   branch to Held 2000's eddy limit, `φ_d = 30° × √(Ω_E·a_E/(Ω·a))` (Earth's 30° the named calibration; 52° on the home
   planet); their share is Siler et al.'s Gaussian scaled to each cell's own half-width and falling to nought AT the edge,
   and the storms' westerly stress rises from nought beyond it (the steady angular momentum returns the cells' torque).
   MEASURED: without the calm the wind jumped at the edge and drew a fixed rain stripe there; with Siler's global
   Gaussian the storms' westerlies reached into the trades (Earth's July at 15° S: +0.6 against NCEP's −5.1 m/s). Now
   Earth's zonal winds: 15° S −5.2/−4.5 m/s (NCEP −2.5/−5.1), 15° N −5.3 (−4.8), 50° S 5.2 (7.2); the direction within 45°
   of NCEP's on 70–73 % of the sea and 39–51 % of the land.
3. **THE RAIN-OUT IS THE RESIDENCE TIME FOR NOW** (amends §3.5). THE 8-CASE MATRIX on Earth (the law's winds or NCEP's ×
   the residence or the saturation law × the eddy diffusion in the eddy belts or everywhere): the residence law with the
   eddy diffusion everywhere is best — the pattern's correlation with GPCP all/land 0.70/0.70 with NCEP's winds, 0.54/0.68
   with the law's own. THE SATURATION LAW (Bretherton et al. 2004) MAKES THE LAND RAINLESS (72–81 % of it under 250 mm):
   MEASURED by a probe, the balance's tropics stand ~4 K cold (the Atlantic at 5° N 23 °C; the missing ice–albedo law),
   and land the first round leaves dry gets a low humidity, cannot export its heat, and overheats (the Congo 40–43 °C, its
   column 17 % saturated) — the balance has no evaporative cooling of land. The saturation law WAITS for the ice–albedo
   law and the land's evaporative cooling (registered).
4. **THE EDDY DIFFUSION OF THE VAPOUR RUNS EVERYWHERE** (amends §3.5's `1 − w`): Siler's partition on the vapour drew the
   rain belt as two node-wide lines (correlation 0.27/0.32 against 0.54/0.68).
5. **THE LAND–SEA DEPARTURES ARE OFF** (amends §3.3 item 4): Lindzen & Nigam 1987's balance was made for gentle tropical
   sea temperatures; over the home planet's 55° summers (land–sea contrasts of 10–20 K) it gives winds of tens of m/s and
   pale dry halos on the sea around every continent. MEASURED on Earth's land: off 0.72, on 0.68. A calibrated law of the
   monsoon's pull is registered.
6. **THE SEASONS ARE 16** for the readings (8, 16 and 32 give the home planet's land 4.5 / 4.1 / 4.1 % under 250 mm with
   the departures on; the stripes were the cells' edges, not the seasons' count).

**THE READINGS.**
- S1, Earth's charter on Earth's relief (90 rows, 64 steps a year): the periodic year in 13 years from a flat 288 K, 20 s
  on one core; the global mean 14.84 °C (the charter's 14.85); the energy-flux equator by month −6 / −8 / −8 / −6 / 0 / 4 /
  10 / 14 / 12 / 8 / 2 / −2° against GPCP's zonal rain maximum −7.5 / −7.5 / −5 / 5 / 5 / 7.5 / 7.5 / 10 / 10 / 7.5 / 7.5 /
  5° (a month's lag in spring; Earth's ITCZ stays north of the equator by the ocean's heat transport, which a slab balance
  has not). The poles too warm and the equator too cool (equator–pole 31 K against Earth's ~45 K): no ice–albedo law.
- S1, the home planet: the annual mean flat (~15 °C at every latitude), the seasons large (the south polar continent 27 °C
  in January, 8 °C in July), the moist energy's maximum running from pole to pole through the year.
- S2, Earth (the law's winds, residence, eddy everywhere, no departures, 16 seasons): the global mean 801 mm/yr (GPCP 983),
  the land's 542 (785), the land under 250 mm 25.6 % (21.4 %), the correlation all/land 0.49/0.72; too wet: the Sahara
  339 (23), the Atacama 568 (64); too dry: western Europe 479 (813), the Amazon 1 342 (2 395).
- S2, the home planet (the same switches): the global mean 565 mm/yr (less sunlight than Earth), the land's mean 440, the
  land under 250 mm 14.3 % against today's 37.7 % (generator 26's stored rows, by nodes).
- THE COST: the balance 17–20 s from a flat start (one core; ~2 s a pass when warm-started), the winds 0.03 s, the
  water ~1.5 s for 16 seasons.

**NEXT** (in order): the 8 km mountains' rain (L6, the windward slope and the lee); the deserts (the saturation law after
the ice–albedo law and the land's evaporative cooling); ExoPlaSim (Q1) for the home planet's circulation; then S3.

### 11.1 The second round of the bench (2026-10-15, after the owner's "Both ok": the order kept, ExoPlaSim approved)

7. **THE ICE–ALBEDO LAW JOINS THE BALANCE** (Williams & Kasting 1997; North, Short & Mengel 1983: fully frozen under
   263.15 K, free over 273.15 K, Earth's ice 0.62 against its ice-free planet's 0.30 — named), the ice-free albedo SOLVED
   each year so the body's sunlight-weighted mean is the charter's Bond albedo (the same closure as the longwave's `A`).
   MEASURED on Earth's charter: the equator–pole contrast 31 → 42 K (Earth ~45 K); 80° N −17.3 °C (Earth ~ −18 °C);
   central Siberia −20.9 / 14.1 °C in January / July; the global mean still 14.85 °C exactly; the ice-free albedo 0.289;
   31 years to the periodic year from a flat start (49 s). The global rain rose 801 → 843 mm/yr (warmer tropics).
8. **THE SATURATION LAW STILL MAKES THE LAND RAINLESS** on the better balance (81 % of the land under 250 mm, the land's
   correlation 0.21): the second round's land, left dry, takes a low humidity and overheats — the balance has no
   surface split of a land cell's energy into evaporation and heat. REGISTERED: the land's surface energy split, then
   the saturation law again (it is the law that makes the deserts dry; the residence law leaves the Sahara at 340 mm
   against GPCP's 23).
9. **THE MOUNTAINS' RAIN (L6) IS BUILT** (`seasons::orography`): the depletion factor `φ` along the season's wind
   (`v·∇φ = 1/τ₀ − φ/τ(H)`, LFPM's height form, the same 8.9-day sea-level time as the climate grid's), the wind split
   exactly on the two bracketing neighbours, each node's rain `W_c·φ/τ(H)`, each climate cell's mean `⟨φ/τ(H)⟩` handed
   back to the climate grid's next round (the water conserved). MEASURED on Earth (16 seasons): the Andes' east face
   2 354 mm/yr (GPCP's cell 1 978), the Tibetan Plateau 245 (290), the Cascades' west 737 against the Columbia basin
   behind it 599 (1 087 / 525) — but the Himalaya's south face 102 (1 153) and India's Western Ghats 768 (1 869): with the
   land–sea departures off there is no monsoon, so the south face is a lee. Norway's west coast 519 (2 014): the
   westerlies at 61° N too weak. The home planet: the land's median 450 mm/yr (10th/90th percentiles 204/707), the land
   under 250 mm 16.0 % (today 37.7 %); the ranges' windward sides wet, the big southern continent's interior ~200 mm.
10. **THE MONSOON IS OWED A LAW**: the land–sea departures barely move the land's winds on Earth's charter (July's land
    direction 40 % on against 37 % off: with one humidity everywhere the balance keeps the land almost as cool as the
    sea), and on the home planet's 55° summers they overshoot. The monsoon needs the land's own heating (the surface
    split of item 8) and a calibrated pull; ExoPlaSim's home run (running now, outside the repo) shows what the home
    planet's own monsoon is.
11. **THE COST**: the mountains' rain per season 3.9 s on one core at first (Earth), 24–35 s on the home planet (its
    slow-relaxing loops where two winds meet); the per-node setup moved onto all cores (the same bytes) and the meeting
    pairs solved exactly inside the sweeps.

### 11.2 The outside measuring stick: ExoPlaSim on the home planet (2026-10-15; the owner's word "Both ok")

ExoPlaSim 3.4.2 (Paradise et al. 2022; GPL-2.0) installed OUTSIDE the repo (`~/voxeldust_earth/exoplasim/`, never linked or
shipped) and run at T21 on the home planet's charter and its stored land map (generator 26): 37 model years, the
climatology of years 27–36 (`home_run/`, monthly rain, lowest-level winds, temperatures, evaporation). ITS LIMITS: it cannot
reach the charter's 288 K at 0.75 of Earth's sunlight with ≤ 1 bar of CO₂ (281.6 K at 1 bar, the air 98.6 % CO₂ in its
radiation); from its own cold start the planet freezes over (two stable climates; the warm one is read); a slab ocean;
5.6° cells. MEASURED READINGS beside the chain's (generator 26's relief, 16 seasons, no departures):

- THE RAIN BELT: in each summer at 55–80° (the equatorial belt leads only in months 4–5) — the chain's belt also runs far
  poleward (its moist energy's maximum from pole to pole).
- THE ZONAL PROFILE (the year): ExoPlaSim wetter at 50–85° (2.4–3.0 mm/day), drier at 20–40° (1.65–2.2) — the chain wetter
  at high latitudes too (600–830 mm/yr) but WITHOUT the dry subtropical bands.
- THE TOTALS: ExoPlaSim 789 mm per Earth year at 281.6 K (~15 % low for the cold, by its 2–3 %/K) — the chain 566 at 288 K:
  THE CHAIN'S EVAPORATION IS LOW (its surface split subtracts Earth's fixed 63 W/m² of net longwave on a body with 0.75 of
  Earth's sunlight; both terms of Earth's split should scale with the body's own sunlight).
- LAND AGAINST SEA: ExoPlaSim's land WETTER (2.39 against 2.09 mm/day; 3.1–3.6 % of the land under 250 mm/yr) — the chain's
  land drier (16 % under 250 mm): the summer continents' pull is off in the chain.
- THE WINDS (July, zonal mean): the summer westerlies at 50° N agree (+7.3 against +8.9 m/s); elsewhere the chain's winds are
  weak (15–30° N +1.8–3.2 against +5–9.4; 15° S −1.4 against −7.1; January ±1.8 against ±8) and at 70° N in July WRONG
  (−23.8 against +2.0). THE WIND LAW IS NOW THE WEAKEST LINK.
- THE SHARED PICTURE: the big southern continent's dry interior and west, the wet high latitudes, the far-travelling belts.

### 11.3 The owner's choice (a): fix the three causes now (2026-10-15)

1. **THE SURFACE SPLIT SCALES WITH THE BODY'S SUNLIGHT** (built): `R_n = (161/240)·kept − (63/240)·kept̄` (Earth's two
   shares, Trenberth, Fasullo & Kiehl 2009, named). MEASURED: Earth's charter unchanged in kind (the global mean 851 mm/yr,
   the land's correlation 0.72); the home planet's global rain 566 → 692 mm/yr (ExoPlaSim 789 at 6 K colder), its land under
   250 mm 14.3 → 8.3 % (ExoPlaSim 3–4 %).
2. **THE WINDS** and 3. **THE SUMMER LAND'S PULL**: a research round on Gill 1980's damped shallow-water response to the
   season's heating (the monsoon, the Walker cells, Rodwell & Hoskins's deserts west of the monsoons) and on the zonal
   surface wind's closure at any spin, read against NCEP (Earth) and ExoPlaSim (the home planet).

### 11.4 The winds from the heating (Gill's balance) — built, first readings (2026-10-15)

The research round (`/tmp/vd-g27/winds_laws.md`, outside the repo) MEASURED on ExoPlaSim's raw fields that the drag is
right (`f·v = r·u`, `r` 1.1–1.5 × 10⁻⁵ s⁻¹) and the chain's north wind too weak: the energy closure `ψ = F/g_ms` reads a
gross moist stability of 40–55 kJ/kg where ExoPlaSim's cross-equator cell has 13–16 (and 0/0 at the equinoxes). Its
cure: the winds from the column's HEATING by Gill's damped balance on the sphere in its weak-temperature-gradient limit
(tested on ExoPlaSim's heating: the direction within 45° on 71–84 % of the cells). BUILT (`seasons::gill`): the heating
`Q = N − ∇·F_e + L·(P − E − C_e)`, the stability `Δs = 1.06·h_max − c_p·T`, the near-surface mass budget with `β_s` 0.2,
the momentum with the bulk drag (the log law's land roughness, a 1 m/s convective gust in place of a floor), one
geopotential solve per drag round. MEASURED:
- THE SOLVE: GMRES with ILU(0) left 14 % of the residual after 4 000 iterations (the Coriolis terms make the operator
  strongly non-normal); BiCGStab stalled likewise; an EXACT BANDED SOLVE (partial pivoting, half-band `2·cols`: the date
  line's diagonal neighbour) reaches 3 × 10⁻¹¹ — the first band of `cols + 1` dropped that entry silently (now refused).
- COUPLED ON EARTH (16 seasons, three rounds): the direction within 45° of NCEP 58–66 % of the sea and 44–49 % of the land,
  the rain's correlation all/land 0.42/0.60 — WORSE than the earlier wind law (70–73 / 39–51 %, 0.49/0.72); the Sahara's
  July 957–1 025 mm/yr.
- WITH GPCP'S RAIN AS THE ONLY HEATING (the research's control): the rain's correlation 0.65/0.77, but the direction 45–52 %
  of the sea — THE STORMS' WESTERLIES NEVER RAN (0 of 48): the cells' own torque came out westerly because the winter
  trades are weak, so the zero-torque gate kept the storm belts calm (the research's risk 3).
- WHAT IS STILL MISSING, each a law of its own: the trades' downward momentum mixing (Stevens et al. 2002; Back &
  Bretherton 2009), the storm belts' closure once the trades are right, and the desert's radiative sink (Charney 1975: a
  bright dry desert loses more than it keeps even in summer — the balance has one albedo and one longwave law everywhere,
  so a hot summer desert reads as a heat source, pulls the air in and rains).

### 11.5 The macro solve with the chain (S3) and its speed (2026-10-15; the owner: "Let's look on it first and then decide")

THE FIRST WORLD (`belts_bench phase2+seasons`, 414 passes, the chain as it stood before §11.3's evaporation cure and §11.4's
heated winds; the artifact `/tmp/vd-g27/s3/phase2+seasons.art`, outside the repo), read beside today's generator-26 world by
`climate_bench art-maps` (the same reader for both):

| reading (the land, by nodes)            | today (generator 26) | the first world (S3) |
|-----------------------------------------|----------------------|----------------------|
| the land's share of the nodes           | 24.00 %              | 23.44 %              |
| the rain's median (10th / 90th), mm/yr  | 391 (105 / 847)      | 471 (251 / 687)      |
| the land under 250 mm                   | 37.7 %               | 9.7 %                |
| the height's median / 90th / top, m     | 207 / 1 272 / 5 483  | 194 / 1 075 / 4 757  |
| over 1 / 2 / 3 km                       | 13.2 / 5.5 / 1.4 %   | 11.0 / 2.9 / 0.7 %   |

The pictures (`~/Downloads/voxeldust_rain_worlds_2026-10-15/`, the globe and a window on the southern continent): today's
rain is WET IN THE HOLLOWS AND DRY ON THE CRESTS (B78's defect 3: the crater's floor and the eastern basin wet, the belt's
crest dry); the first world's rain is wet on the belt's windward (sea) face and drier inland, with no wet blot on a hollow.
The relief follows: the belts lower, the crater keeps its high rim country (no wet blot carves it), the eastern basin stays.
The first world's bench readings by cause (old plains 230 / 155 m mean / median against Earth's 485 / 349 m; the collision
belts 1 507 / 1 392 m; 77.6 % of the land drains to the sea) have no generator-26 control at hand (a control solve, about an
hour, not run).

THE SPEED (no law changed): the second world (today's chain) ran at ~115 s a pass (~13 h). MEASURED by a sample: half the
time the heated winds' exact solve, half the mountains' rain, both on one core. Three cures:
1. THE SEASONS IN WAVES — a call's sixteen seasons read only the last pass's state; they run one season a core and join in
   their order: the same rows word for word on 1, 5 and 16 cores (test `the_chain_gives_the_same_bytes_on_any_number_of_cores`).
2. THE BAND ORDER — the heated winds' unknowns by latitude row with the longitudes folded at the date line: the half-band
   `cols + 2` against `2·cols`, a quarter of the elimination's work, still exact.
3. THE LOOPS SOLVED EXACTLY — the mountains' rain after each round takes the nodes still moving, finds their loops (Tarjan's
   strongly connected parts) and solves each up to 512 nodes as one system, upwind first. MEASURED on the home planet's 16
   seasons with the heated winds (three rounds, cold starts): 2 300 rounds in all without it, 962 with it; the worst seasons
   202–250 rounds against 6–17. In the macro solve a pass now takes 13–33 s (the first warm call 30 s against 121 s), but
   about one call in ten still reaches the round bound (400): its slow part lies inside a connected part of millions of
   nodes the exact solve does not take; the warm start carries the answer on, so the error does not grow (a reading owed:
   the bound's calls counted at the end). A cure if it matters: an accelerated fixed-point iteration (Anderson's) over the
   sweeps.
4. REFUTED — THE WHOLE PLANET IN THE SEARCH'S OWN ORDER (every part of the upwind links in Tarjan's order, the large parts
   swept in the search's finishing order, one pass a round): on the moon's vortex 21 one-pass rounds against 28 four-pass
   rounds, but on the home planet's seasons 300–400 one-pass rounds a season, several unmet (against 6–99 four-pass rounds):
   the home planet's upwind links form ONE connected part of 6–9 million nodes, and inside it the search's order follows
   the wind worse than the grid's four orders. Reverted; the grid sweeps with the loops' exact solve stand.
5. THE SPEED ROUND OF 2026-10-16 (the owner: "Ideally we should have such planets solved in minutes"; no law changed; the
   first twelve calls of the home planet's macro solve on 8 cores, the same cases each time):

   | change | twelve calls | worst rounds a call |
   |---|---|---|
   | §11.5 items 1–3 (the second world's run) | 332 s | 126 59 34 37 **400** 30 14 140 63 80 36 14 |
   | + ANDERSON'S ACCELERATION (Walker & Ni 2011, depth 4) over the rounds | 325 s | 23 34 15 13 37 12 11 127 22 16 49 15 |
   | + the seasons TAKEN IN TURN by the cores (`par::map_in_turn`) + the accelerator's kept products | 250 s | the same |
   | + ONE DRAG ROUND a pass from the last pass's wind (the cold start keeps three) | 223 s | 23 55 15 19 22 17 14 17 12 47 83 33 |

   On the 48 cold season-runs (`climate_bench home-s2`) the accelerator gives 703 rounds against the loops' 962 (worst 55
   against 99); its round costs more, so it pays where the sweeps stall — the warm starts inside the solve, where it ends
   the calls at the round bound (62 of 421 in the second world's run). A PROFILE (sampled, symbols kept) of the warm passes
   before these cures: the mountains' rain ~40 % and the heated winds' exact solve ~33 % of the CPU, threads waiting ~30 %;
   the year's balance runs on one core (~8 % of the wall); outside the climate a pass costs ~3.4 s (the lakes' budget
   2.4 s of it, on one core).
6. THE CLIMATE'S OWN SPACING (the owner: "Let's try every 8th pass please, we can increase if that will make sense"): the
   bench's `+seasons8` re-solves the climate every 8th pass and the running sea every pass (ruling B68 found the sea's stale
   spacing, not the climate's, made the jump near the present); measured against the second world (every pass).
7. THE FLICKERING RAIN (2026-10-16, MEASURED by the land's rain median and desert share at every call): with the climate
   every pass the rain swung from pass to pass (desert 0 → 8 %, the median 463–566 mm) on a relief that hardly moved, and
   the every-8th-pass run repeated the every-pass values CALL BY CALL — the climate followed its own call count, not the
   ground. CAUSE: the heated winds' drag fed a round's wind speed straight into the next round's drag ("speed = c ÷
   speed"), which jumps between two speeds forever. CURE (no law changed): each round gives every cell its own drag balance
   under the round's field exactly (a quadratic in `|U|²`, the spin included) and the rounds repeat to a stated stop
   (`DRAG_TOL_M_S`, Anderson's acceleration over them) — passes 14–24 then hold 510–521 mm and 1.1–2.6 % desert.
8. THE SPIN-UP (`SPIN_TOL`): the climate's slow words start cold (the median 468 → ~530 mm over ten calls); from a cold
   state the chain repeats on the starting relief until every cell's year of rain moves less than 1 % (or a millimetre).
   MEASURED: it did NOT meet that stop in 30 calls (22 min) on the home planet's starting relief.
9. THE CLIMATE HAS MORE THAN ONE STATE ON ONE GROUND (MEASURED, both runs from the same spun-up climate, 513 mm and 2 %):
   every pass, the rain dips now and then (473 mm at pass 8, 469 at passes 44–46) and returns to ~510 within four calls;
   every 8th pass, it fell to ~471 mm at pass 56 and STAYED for 240 passes. The final every-8th-pass world (98 min on 6
   cores, `/tmp/vd-fast/s_k8/`): land 23.2 %, mountains 1.66 %, over 1 / 2 / 3 km 10.9 / 3.4 / 1.0 %, the land's rain
   559 mm by the stored classes (295–719, 0.5 % desert) against the second world's 423; six tracked plates' old plains
   within 10 % of the second world's, four 12–39 % lower. B39's gate cannot judge a cadence while the climate follows its
   own history; the suspect is the latent heating's feedback (rain heats the column, the heated winds gather more vapour).
10. THE CAUSE FOUND AND CURED (2026-10-16, the owner: "Option 1" — one day to find it): the chain on a FIXED ground
    (`climate_bench chain-fixed`, the second world's final relief), 30 calls from cold, one slow word held still at a
    time:

    | variant | settles? | the land's mean rain |
    |---|---|---|
    | as it was | no — 3 300–13 000 cells still over 1 % a call after 30 calls | wanders 491–546 mm |
    | the land's wetness held still | no — worse | runs away to 905 mm |
    | THE LATENT HEATING held still | YES, in 11 calls (the largest change ×0.66 a call) | settles at 622 mm |

    THE CURE is the published form of the same physics: the vapour the lower branch gathers IS the latent heating, so the
    heating carries no latent term and the dry stability becomes the GROSS MOIST STABILITY `M = h_T − h` (Neelin & Held
    1987), `h_T` 1.06 × the season's most energetic cell (convective quasi-equilibrium, Emanuel 1994; Siler, Roe & Armour's
    ratio) — positive at every cell by construction, and no lag left. MEASURED on the same fixed ground: settled in 12
    calls (186 s on 14 cores; the largest change ×0.68 a call), the drag in 3 rounds (from up to 30); the land's mean rain
    668 mm, the median 573 mm, 0.2 % desert — WETTER than the second world's (the owner's look owed).
11. THE CADENCE UNDER THE GATE (2026-10-16, the moist form, both runs on 7 cores sharing the machine): the climate every
    pass and every 8th pass agree at every pass both computed (the land's rain median 550–551 mm, the desert 1.7–1.8 %
    over passes 0–100; before the cure 510 against 470 mm). The two final worlds (`/tmp/vd-fast/m_k1/`, `m_k8/`):

    | reading | every pass | every 8th pass | gate (B47: 2 points, 10 %) |
    |---|---|---|---|
    | land / plains / tablelands / mountains | 22.95 / 45.97 / 51.64 / 2.39 % | 22.76 / 46.85 / 50.91 / 2.23 % | pass |
    | land over 1 km; relief p50 / p90 / p99 | 10.32 %; 98 / 526 / 1 440 m | 10.63 %; 96 / 522 / 1 380 m | pass |
    | heights by cause (means) | crater 564, collision 1 419, Andean 771, plateau 949, old plains 216 m | 577, 1 485, 777, 991, 208 m | pass (≤ 4.6 %) |
    | drainage to the sea; past a lake; mean rain | 79.3 %; 25.8 %; 583 mm | 79.0 %; 27.5 %; 575 mm | pass |
    | the tracked plates' old plains (B65: 5, 13) | 196, 51 m | 186, 48 m | pass (−5, −6 %) |
    | the other plates' old plains | 10 of 13 within 10 % | plates 7, 1, 8: −15, −12, −11 % | three over (untracked) |

    THE TIMES (7 cores, shared): every pass 2 h 6 min (the climate 87 min, 12.6 s a call); every 8th pass 63 min (the
    climate 18 min; the lakes' budget 21 min on one core is now the largest part). The moist world's look: the land's rain
    median 559 mm (343–783), no desert; heights over 1 / 2 / 3 km 10.4 / 2.5 / 0.5 % (every pass).

THE SECOND WORLD (today's chain: §11.3's evaporation, §11.4's heated winds as departures; `/tmp/vd-g27/s3b/`, 414 passes
in 5 h 35 min on 8 cores, a climate call 43 s on the mean; 62 of 421 calls at the mountains' round bound):

| reading (the land, by nodes)            | today (generator 26) | the first world | the second world |
|-----------------------------------------|----------------------|-----------------|------------------|
| the rain's median (10th / 90th), mm/yr  | 391 (105 / 847)      | 471 (251 / 687) | 423 (279 / 591)  |
| the land under 250 mm                   | 37.7 %               | 9.7 %           | 2.2 %            |
| the height's median / 90th / top, m     | 207 / 1 272 / 5 483  | 194 / 1 075 / 4 757 | 199 / 1 080 / 4 664 |
| over 1 / 2 / 3 km                       | 13.2 / 5.5 / 1.4 %   | 11.0 / 2.9 / 0.7 % | 11.0 / 3.3 / 0.7 % |

Its bench readings: mountains 2.19 % of the land (a reading; Earth 2.39 %), the old plains 244 / 168 m (Earth 485 / 349 m),
the collision belts 1 527 / 1 435 m, 79.3 % of the land drains to the sea. THE RAIN IS NEARLY EVEN: the land's spread 279–591
mm between its 10th and 90th share and 2.2 % under 250 mm (ExoPlaSim's home run 3–4 %, Earth 21 %); wet spots on the
southern coasts and the polar belt's windward face; the belt's windward face wet, its lee and the interiors moderate; no
desert. The laws still missing are the ones that make deserts: the desert's radiative sink (Charney 1975), the trades'
momentum mixing, the storm belts' closure, the land's surface energy split and the saturation law.
