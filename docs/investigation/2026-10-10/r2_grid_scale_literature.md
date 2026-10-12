# R2's grid-scale stripes and straight channels: the literature (2026-10-10)

Ruling B76's first step ("the literature on straight channels"). A read-only research pass; no file, build or test.
Labels: STATED or MEASURED = the source says or measures it; INFERENCE = the research pass's own reasoning, to be
benched before it is believed.

## Short answer

R2's 1 024 m grid cannot hold the valleys its own laws make. The laws space the first-order valleys about 130 m to
1.9 km apart; at the cell scale creep is 3 × 10³ to 8 × 10⁵ times weaker than incision, so every cell becomes its own
channel and the routing only chooses the stripes' direction. The grid-scale ratio alone does not prove a defect (item 5a).

## 1. Valley spacing

- STATED (Perron, Kirchner & Dietrich 2009): the Péclet number `Pe = K·L^(2m+1)/D`; the characteristic length
  `L_c = (D/K)^(1/(2m+1))`, where creep and incision take the same time; in the model `6.4·L_c ≤ λ ≤ 12.7·L_c`; `L_c²` is
  about the area where a hillslope becomes a valley.
- MEASURED (Perron et al. 2009, five US sites, laser altimetry): λ from 30 ± 3 m (Dragon's Back) to 321 ± 33 m (Eaton
  Hollow); `L_c` from 4.0 m to 46 m; λ follows `L_c`.
- STATED (Theodoratos, Seybold & Kirchner 2018, eq. A8): for `E = K·A^m·S^n`, `l_c = (D^n·U^(1−n)/K)^(1/(n+2m))` — with
  `n ≠ 1` the length depends on the uplift. Their runs used a mean mesh edge of 0.8·`l_c`; the mesh must be much finer
  than `l_c`.
- MEASURED (Clubb et al. 2016): drainage density grows with the erosion rate at four of five US sites (agrees with n > 1).
- MEASURED (Kargère et al. 2025): channel heads at Gabilan Mesa at area r², r ≈ 62 m.
- INFERENCE, the game's law (m = 1 on `Q = R·A`, n = 2.2, `K′ = K_lp·R`, D = 0.003 m²/yr, R = 1 m/yr; `l_c ∝ R^−0.24`;
  the 6.4–12.7 factor is measured for n = 1 only — for n = 2.2 UNMEASURED):

| Uplift | `l_c` | Valley spacing λ |
|---|---|---|
| 0.01 mm/yr | 147 m | 0.9–1.9 km |
| 0.1 mm/yr | 76 m | 0.5–1.0 km |
| 1 mm/yr | 40 m | 250–500 m |
| 10 mm/yr | 21 m | 130–260 m |

  The grid Péclet number at 1 024 m, `(Δx/l_c)^4.2`, is 3.5 × 10³ to 8 × 10⁵ — why Earth's creep coefficient lowers the
  stripes by only 13 % (MEASURED on the bench the same day). To resolve the valleys: `Δx ≤ l_c` (8 or more cells per
  wavelength), about 32 m, and 16 m on fast belts.

## 2. When the grid is coarser than the valleys

- Lawful, with a calibration body: a channel-head threshold (Montgomery & Dietrich 1988/1992 measured `A·S²`
  thresholds); for the game, `A_c = l_c²` from its own laws (Perron 2009; Kargère 2025) — no new constant. Hergarten 2020
  rescales the river term by the hillslope area per river cell, `α(A_c/δ²)^β`, β ≈ 0.47–0.5 fitted to real networks
  (STATED: less grid dependence, some stays; INFERENCE: built for `δ² ≤ A_c`, R2 is the opposite regime). Ganti et al.
  2012: an upscaled closure of nonlinear hillslope flux on Oregon lidar (hillslopes only).
- Knobs or numerical closures: a hand-chosen area threshold (Howard 1994; Tucker & Bras 1998); an enlarged "effective"
  D; Passalacqua et al. 2006 (no physical calibration body). FastScape-family models (Braun & Willett 2013; Yuan et al.
  2019) make a channel at every node (INFERENCE: R2's behaviour).

## 3. Routing and the grid's directions

- MEASURED (Prescott et al. 2025): D∞ reads about 25 % high along the cardinal and diagonal directions, about ten times
  MFD's error; MFD (Freeman 1991, p = 1.1) hardly changes when the grid turns; with p > 1.1 MFD leans to the grid too; D8
  is unrealistic on any slope off a multiple of 45°.
- Other cures: Quinn 1991; Holmgren 1994; Seibert & McGlynn 2007 (MD∞); Pelletier 2010; Bonetti et al. 2020 (a
  specific-catchment-area PDE, grid-independent).
- INFERENCE: no routing removes grid-scale channels — routing sets their direction, not their scale (the bench agrees).
  MFD's p is a numerical constant, not a physical calibration.

## 4. Computer graphics

- STATED: Cordonnier et al. 2016 solve the stream power on a RANDOM PLANAR GRAPH, not a grid, then blend landform
  kernels that read the graph; Génevaux et al. 2013 build the river network first and the terrain as a construction tree
  of primitives with levels of detail; Schott et al. 2023 edit the uplift and solve with a fast drainage approximation;
  Schott et al. 2024 (the code's README) alternate erosion, thermal and deposition with ×2 upsampling; Tzathas et al.
  2024 an analytical stream-power solution with time as a parameter.
- The full texts could not be opened (HAL blocks the fetch): the grid sizes are UNVERIFIED.
- INFERENCE: none erodes at 1 km and expects valleys; detail comes from primitives or from erosion in steps of ×2.

## 5. Recommendation (INFERENCE — each item benched before it is believed)

- a) **The gate before the cure.** For a self-affine profile with Hurst exponent H, `ms(Δ²)/ms(Δ) = 4 − 2^(2H)`. Earth's
  spectrum β ≈ 2.09 (Gagnon et al. 2006; the game's own B60 constant), scaling to about 40 m: H ≈ 0.545, the ratio about
  1.87 — Earth sampled at 1 km probably reads about 1.9, like R2. The defect is the DIRECTION and the ONE-CELL FORM: gate
  on the directional spectrum (power at 0°, 45°, 90° against other angles) and the channel spacing in cells.
- b) **R2 carries each cell's MEAN height plus only the channels it resolves** (at least about 4 cells apart — a sampling
  rule stated once, not physics). An unresolved cell cuts no channel of its own: it lowers with its outlet and keeps the
  sub-grid relief the same laws compute (the analytic steady profile from `A_c = l_c²` down to the cell's outlet area,
  plus Roering's hillslope relief over `l_c`) — no new constant. Across a tilted plain neighbouring cells stay equal, so no
  stripe forms.
- c) **R3 at 32 m, baked near the viewer**, its mean held to R2, its outlets from R2's network; 32 m resolves
  `l_c ≥ 40 m` (uplift ≤ about 1 mm/yr) with 8 or more cells a valley; where `l_c` < Δx (fast belts, ≈ 21 m) the closure
  of (b) or a 16 m rung.
- d) **Below R3**: Roering's steady hillslope profile, plus noise in the scaling range below `l_c` only.
- e) **Routing**: MFD with p = 1.1 on the unresolved parts removes the direction bias (p is a numerical constant).

## Sources

Perron, Kirchner & Dietrich 2009, Nature 460:502, doi:10.1038/nature08174 · Perron, Dietrich & Kirchner 2008, JGR
113:F04016, doi:10.1029/2007JF000977 · Perron et al. 2012, Nature 492:100, doi:10.1038/nature11672 · Theodoratos,
Seybold & Kirchner 2018, ESurf 6:779, doi:10.5194/esurf-6-779-2018 · Clubb et al. 2016, JGR-ES 121:1724,
doi:10.1002/2015JF003747 · Kargère et al. 2025, ESurf 13:403, doi:10.5194/esurf-13-403-2025 · Hergarten 2020, ESurf
8:367, doi:10.5194/esurf-8-367-2020 · Ganti, Passalacqua & Foufoula-Georgiou 2012, JGR 117:F02012,
doi:10.1029/2011JF002181 · Passalacqua et al. 2006, WRR 42:W06D11, doi:10.1029/2006WR004879 · Montgomery & Dietrich
1988, Nature 336:232, doi:10.1038/336232a0 · Montgomery & Dietrich 1992, Science 255:826, doi:10.1126/science.255.5046.826
· Howard 1994, WRR 30:2261, doi:10.1029/94WR00757 · Tucker & Bras 1998, WRR 34:2751, doi:10.1029/98WR01474 · Braun &
Willett 2013, Geomorphology 180:170, doi:10.1016/j.geomorph.2012.10.008 · Yuan et al. 2019, JGR-ES,
doi:10.1029/2018JF004867 · Pelletier 2010, Geomorphology 122:91, doi:10.1016/j.geomorph.2010.06.001 · Prescott et al.
2025, ESurf 13:239, doi:10.5194/esurf-13-239-2025 · Freeman 1991, Computers & Geosciences 17:413 · Quinn et al. 1991,
Hydrological Processes 5:59 · Holmgren 1994, Hydrological Processes 8:327 · Tarboton 1997, WRR 33:309 · Fairfield &
Leymarie 1991, WRR 27:709 · Seibert & McGlynn 2007, WRR 43:W04501 · Bonetti et al. 2020, PNAS 117:1375,
doi:10.1073/pnas.1911817117 · Gallant & Hutchinson 2011, WRR 47, doi:10.1029/2009WR008540 · Roering, Kirchner &
Dietrich 1999, WRR 35:853, doi:10.1029/1998WR900090 · Gagnon, Lovejoy & Schertzer 2006, NPG 13:541 · Cordonnier et al.
2016, CGF 35(2):165, doi:10.1111/cgf.12820 · Génevaux et al. 2013, ACM TOG 32(4):143, doi:10.1145/2461912.2461996 ·
Schott et al. 2023, ACM TOG, doi:10.1145/3592787 · Schott et al. 2024, ACM TOG 43(4), doi:10.1145/3658200 (code:
https://github.com/H-Schott/MultiScaleErosion) · Tzathas et al. 2024, CGF 43(2), doi:10.1111/cgf.15033
