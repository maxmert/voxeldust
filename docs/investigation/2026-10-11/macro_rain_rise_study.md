# Why the home planet's land is dry and almost still: a read-only study (2026-10-11)

Ruling B77 item 3 ("the macro world MEASURED BESIDE"). A read-only pass: no edit, no build, no program. Labels: MEASURED
(the code or the stored world gives the number), STATED (a source says it), INFERENCE (hand arithmetic from the code's laws
or reasoning — to be measured before it is believed).

The readings it explains (MEASURED on generator 26's stored world, the bench's `--macro-rates`): the land's rain median
383 mm/yr, 1.5 % of the land at 1 000 mm or more; the joint law's present erosion rate median 0.00005 mm/yr, over 0.1 mm/yr
on 1.2 % of the land.

## 1. The rain

**How the solve makes a node's rain** (code):
- `supply(T) × belt(u) × lift(slope)`, rounded and clamped to [1, 10 000] mm/yr (`climate.rs:831-835`, `solve.rs:57-59`).
- The supply: 990 mm × e_s(T)/e_s(288 K), e_s by Tetens (`climate.rs:390-393`); T the node's own air temperature,
  `t_mean + T2·P2(sin φ) − Γ·h` (`:411-413`); over the ocean the air stands at the sea's surface (`:887-892`).
- The belt: `0.09 + 0.91·(1−u²)²`, u the place between a cell's rising and sinking branch (`:122-123`, `:374-377`); the
  spin alone sets the cells' count (Held–Hou, `:321-349`).
- The lift: `1 + 8·|rise/chord|` from ONE upwind neighbour, divided on a fall, clamped to [0.1, 4] (`:129-130`, `:398-407`).
- The season changes only the ice line (`:465-476`); the rain is an annual mean with one fixed wind.
- The sea enters only as the temperature's datum and the season's heat store (`:779-780`).
- The code states its own limit: "no fetch, no continental term" (`:745-747`); no moisture budget and no seasonal cell exist
  in the terrain crate (MEASURED by a search).

**The home charter** (`home.rs:112-121`): 288 K, the tilt's cosine word 586/1024 (55°), a 3.0-day day, 0.75 of Earth's
sunlight.

**What makes the land dry**, largest first:
1. **THE BELT FACTOR IS A SHARE, NEVER RENORMALISED — A DEFECT.** Its area mean over one cell is 0.09 + 0.91·8/15 = 0.575.
   The law takes Earth's mean rain (990 mm) as its calibration and then only removes from it, so the water budget never
   closes. The home planet's sea-level mean ≈ 570 mm; on Earth's charter the same law gives ≈ 710 mm (INFERENCE, 3 cells,
   T2 = −25 K); Earth's measured global mean ≈ 950–990 mm (GPCP, STATED). The law does not give back its own calibration
   body.
2. **THE 55° TILT FLATTENS THE ANNUAL TEMPERATURE PROFILE, AND THE SEASONAL CIRCULATION IS MISSING — A MISSING LAW.**
   s₂ ≈ +0.005, T2 ≈ +0.2 K (INFERENCE): the equator gets no Clausius–Clapeyron boost (Earth's ×2.2, home ×1.0). At 54° the
   annual-mean Hadley circulation vanishes (Ferreira et al. 2014, STATED); the real circulation is a seasonal cell moving
   into each summer hemisphere. The code's one fixed cell rains as cos⁴φ, so all land poleward of ~40° is dry by
   construction. Ruling B36 Q3 already allowed the seasonal circulation; the design lists it (A1, A3); it was never built.
3. **HEIGHT DRIES THE LAND — THE WRONG LAW FOR HEIGHT.** The supply follows e_s at the node's own air: at Γ = 6.29 K/km the
   rain falls ~35 % in the first kilometre (INFERENCE). The belt stand at −41° would get ~380 mm at sea level and reads 96 mm
   (a high plateau ~3 km up, INFERENCE). On Earth windward slopes often get MORE rain up to 2–3 km (Roe 2005, STATED); rain
   scales far more weakly with temperature than vapour (≈2 %/K against 7 %/K, Held & Soden 2006, STATED).
4. **Not the causes**: the clamps almost never bind (the lift's needs a slope ≥ 0.375 over 8 km; 10 000 mm is out of reach)
   — but they break B23; the missing fetch makes interiors too WET relative to coasts, not dry; the 76 % ocean share does
   not enter the rain at all.

**Against Earth** (STATED): Earth's land rain ≈ 113 × 10³ km³/yr over ≈ 149 M km² ≈ 760 mm (Trenberth et al. 2007 — the
table not seen in full; the figure from memory and a secondary source); drylands (P/PET < 0.65) 41 % of the land, arid +
hyper-arid ≈ 17 % (MEA 2005). The home planet: median 383 mm, 38.6 % of the land under 250 mm (MEASURED); about two thirds
dryland (INFERENCE).

**Verdict: the dryness is NOT the law's honest answer for this planet** — a calibration defect (the unnormalised belt) and a
missing seasonal-circulation law plus the wrong height law.

**The right law**: a moisture budget along the wind with condensation (the design's A2) — Smith 1979 / Smith & Barstad 2004
(vapour from the ocean, lifted, condensed and falling downwind), the global mean bounded by the energy budget (O'Gorman et
al. 2012), on the seasonal cell of Lindzen & Hou 1988. What it changes (INFERENCE): coasts wetter than interiors; windward
ranges wet to their crests; rain shadows that build over a range (today one node only); the summer hemisphere's high
latitudes wet in their season; the land's median rain perhaps ~600–750 mm; the rivers' discharge, and the joint law's cut,
about doubled.

## 2. The present rise

**The schedule** (code):
- Every belt's window ENDS AT THE PRESENT: `start_yr = age − share·t_m` (`belts.rs:412`); the wedge grows until the present
  (`:156-167`, `solve.rs:948-962`, called at `:3218`) — every belt is active today by construction.
- The plate speed: 100 mm/yr for each whole drift share (`belts.rs:52`).
- A collision's influx falls as e^(−t/τ), τ = 1/(7.03 × 10⁻¹⁶ s⁻¹) ≈ 45 Myr (INFERENCE from `:68`); an Andean margin's
  influx stays steady (`:400-404`).
- When a crest reaches its force ceiling it stops rising; only the plateau widens and the flanks move outward (`:172-210`).
- Every other node's "even uplift" is spread over the WHOLE 5 Gyr (`solve.rs:957`): its present rate is near zero.
- The landscape's time only starts each plate's erosion later (`:879-889`, `:922-944`); it raises nothing.

**The present uplift** (INFERENCE by hand; UNMEASURED on the artifact): an Andean wedge at 50 mm/yr takes in ~312 m²/yr of
rock, ~47 m²/yr of it at the surface; below the ceiling the crest rises ~0.65 mm/yr at 2 km; above it the plateau widens
~15 km/Myr, the inland ramp rises ~0.6 mm/yr on a band ~80 km wide and the plateau top 0; an old collision (a 100 Myr
window) keeps ~11 % of its first influx.

**Why the present erosion reads so low**, ranked:
1. **No present uplift on ~98 % of the land**: a relic graded by the joint law to slopes of ~10⁻⁴ (INFERENCE). Earth's land
   also rises by dynamic topography, glacial rebound and base-level falls, and hillslope weathering wears even flat cratons
   at ~5 m/Myr (outcrop median, Portenga & Bierman 2011, STATED); the solve has no hillslope law below 8 km.
2. **THE SCALE EFFECT**: Adams et al. calibrated `E = K_lp·Q·S^2.2` on CHANNEL slopes (STATED); the solve reads the fall
   between node means over 8 km, which follows the ridge-and-valley envelope, not the steep low-order channels — with
   n = 2.2 a slope three times too low reads eleven times too slow (INFERENCE). Example: the wedge's pro taper (Taiwan's
   critical 2.2°) with Taiwan's 2.5 m of rain on one 64 km² node reads ~0.3 mm/yr; Taiwan erodes 3–6 mm/yr (Dadson et al.
   2003, STATED).
3. **The dry climate**: Q about half of what Earth's rain would give, the cut ×0.5 (INFERENCE).
4. **Plateaus at their ceiling do not rise.**

No single defect in the uplift code; to check against B40's text: the wedge's widening reads the influx only
(`belts.rs:156-167`), and erosion acts separately on the height.

**Earth** (STATED): Portenga & Bierman 2011 — basin median 54 m/Myr (0.054 mm/yr), mean 218 m/Myr; outcrop median 5.4 m/Myr.
Larsen et al. 2014 — the steepest ~10 % of the land gives over half the denudation. Montgomery & Brandon 2002 — in active
ranges erosion climbs toward mm/yr as the mean local relief nears ~1 km. The home median 0.05 m/Myr is ~1 000 times slower
than Earth's basin median and ~100 times slower than Earth's bare outcrops; 1.2 % of the home land over 0.1 mm/yr against
Larsen's ~10 % (MEASURED vs STATED).

## 3. Recommendations, ranked by what most changes the look

1. **THE RAIN — a cure in the macro laws is needed** (wet forested valleys, full rivers and lakes, snow on windward crests):
   the seasonal cell from the body's own s₁ and s₂ (Lindzen & Hou 1988; Ferreira et al. 2014 for the high-tilt check) plus a
   moisture budget along the wind (Smith & Barstad 2004); Earth's GPCP global mean and Trenberth's land rain as READINGS only
   — the law must give them back on Earth's charter as a test, never a knob; it replaces the belt's share, the lift clamps
   and the rain clamps (B23). Price: a world change (generator 27, new pins), a full first solve on many cores, an upwind
   march where today each node is O(1) (`climate.rs:7`), the design's estimate of 3.5–5.5 engineer-weeks, new data only if
   the season's rain must be shipped (an SL6 ask; the annual row can stay).
2. **THE ACTIVE RANGES — measure first, then the forms below 8 km**: (a) measure the present uplift map (the wedge's rise
   over the last pass at each node; no world change; both hosts can compute the belts from the seed's first land, so no new
   data crosses); (b) let B77's per-point landforms read that uplift U and the rain — the steady stream-power relief grows as
   (U/K)^(1/n) (Whipple & Tucker 1999): jagged crests where U > 0; (c) do NOT re-tune K_lp for 8 km (a knob, refused by B38)
   — the lawful cure for the scale effect is to run the law at its own scale (B77); (d) the parked Andean law joins later as
   one change (81 % of the belts' rock).
3. **Uplift outside the belts** (e.g. dynamic topography): a new law and a large world change; the lowest priority for the
   look.

## Sources

Adams et al. 2020, Sci. Adv. 6, eaaz3166, doi:10.1126/sciadv.aaz3166 · Adler et al. 2018 (GPCP V2.3), Atmosphere 9, 138,
doi:10.3390/atmos9040138 · Dadson et al. 2003, Nature 426, 648–651, doi:10.1038/nature02150 · Ferreira, Marshall, O'Gorman &
Seager 2014, "Climate at high-obliquity", Icarus 243, 236–248, doi:10.1016/j.icarus.2014.09.015 · Held & Soden 2006, J. Climate
19, 5686–5699, doi:10.1175/JCLI3990.1 · Larsen, Montgomery & Greenberg 2014, Geology 42, 527–530, doi:10.1130/G35136.1 · Lindzen
& Hou 1988, J. Atmos. Sci. 45, 2416–2427 · Millennium Ecosystem Assessment 2005, Dryland Systems (ch. 22) · Montgomery & Brandon
2002, EPSL 201, 481–489, doi:10.1016/S0012-821X(02)00725-2 · O'Gorman et al. 2012, Surv. Geophys. 33, 585–608,
doi:10.1007/s10712-011-9159-6 · Portenga & Bierman 2011, GSA Today 21(8), 4–10 · Roe 2005, Annu. Rev. Earth Planet. Sci. 33,
645–671 · Smith & Barstad 2004, J. Atmos. Sci. 61, 1377–1391 · Trenberth et al. 2007, J. Hydrometeorol. 8, 758–769,
doi:10.1175/JHM600.1 · Whipple & Tucker 1999, JGR 104, 17661–17674, doi:10.1029/1999JB900120.
Code read: `crates/terrain/src/climate.rs`, `belts.rs`, `solve.rs`, `home.rs`, `land.rs`;
`docs/investigation/2026-10-02/ground_step_design.md` (§A1–A3, Q3).
