# Shoreline papers — a natural shore in numbers, and one shape across detail levels

Written 2026-09-25 on branch `game` at `4442fa7` plus the uncommitted step-2b change (GENERATOR_VERSION 21,
ARTIFACT_VERSION 9). Two design rounds for the shoreline (believability step 4, ruling B8 item 1) found 47 faults
(`../2026-09-24/shoreline_design.md` §18, §19). The next step is a SHORELINE BENCH that paints the DRAWN water at several
rungs. Before it, the owner asked for a focused paper search on two questions only:

> *"Yes, let's explore those 2 only"*

- **Question 1.** What does a natural shoreline look like below our 8 192 m grid, as numbers that a bench can compute
  on a drawn shore and turn red?
- **Question 2.** How do water and land stay one shape when the view changes its detail level?

**What this file is.**
- It answers the two questions and nothing else. It does not repeat three earlier surveys:
  - [`fine_terrain_models_and_crates.md`](../2026-09-22/fine_terrain_models_and_crates.md) (the fine-scale algorithms);
  - [`lakes_and_landscape_models.md`](../2026-09-22/lakes_and_landscape_models.md) §1 (the HydroLAKES counts, the
    size law, the Cael & Seekell corrigendum, the shoreline development by size class);
  - [`terrain_models_and_crates_survey.md`](terrain_models_and_crates_survey.md) (`geo` and `imageproc` for the bench,
    Proland's residual design for R2).
- It reads against `shoreline_design.md` §0–§3, §7, §18, §19, and the rulings W6, W10, W15, W16, W17, W18
  (`../../design/owner_decisions_2026-09-21_water.md`) and B6–B8 (`../../design/owner_decisions_2026-09-22_believability.md`).

**How it was made.**
- Four read-only research groups: (1) the sea-coast laws; (2) the lake-shore laws and how procedural work makes
  shores; (3) the level-of-detail techniques; (4) the planet engines and the vector features.
- One verifier opened the sources again for 51 claims. It CONFIRMED 29, REFUTED 17 and left 5 UNVERIFIED. This file
  carries only the corrected forms. §7.2 lists every refuted claim beside its correction.
- Nobody built anything, ran cargo, or edited another file. No number here is MEASURED on our world.
- The web requests used the neutral User-Agent "voxeldust-research" and carried no identity.

**Marks.**
- Read status of a source: **OPENED** = the full text was read; **ABSTRACT** = the abstract or the landing page only;
  **SECONDARY** = the number comes from another source that cites it; **UNVERIFIED** = nobody read it.
- The verifier's result: **CHECKED** = the verifier opened it again and the claim holds; **CORRECTED** = the verifier
  refuted the first claim, and this file gives the corrected form.
- **ARITH** = hand arithmetic. **DOC** = a number from a ruling or an investigation file. **PREDICTED** = what a
  bench line is expected to print; a prediction is never a result. **UNMEASURED** = nobody printed it.
- Rung 7 = 128 m cells, rung 9 = 512 m, rung 10 = 1 024 m (R₁, the first rung that reads a pyramid level on the home
  planet), rung 13 = 8 192 m (one node). At a swap, one pixel is one cell of the finer rung (half a coarser cell).
- Source ids (C1, L1, P1, G1, E1) point to the register in Appendix A.

---

## 0. THE SHORT ANSWER

**Question 1 — what a natural shore looks like, in numbers.**

1. **A real coast is rough in a regular way at every scale.** One number says how rough: the "fractal dimension".
   A straight line reads 1.0. A sandy coast reads about 1.0–1.15. A typical coast reads about 1.2. A rocky coast
   reads about 1.3. A fjord coast reads about 1.4. The exact number depends on the measuring method, so the bench
   compares our coast with Earth only by the same method (§1.1).
2. **A shore is the line where the ground meets the water, so its roughness comes from the ground.** A published law
   links the roughness of a surface to the roughness of its level lines. Six of ten measured coasts sit on that law.
   The other four are coasts that ice cut or a rising sea drowned. So a natural coast needs the right ground and the
   right history. It needs no noise of its own.
3. **A lake shore follows the same law.** One typical lake reads about 1.1. A whole population of lakes reads
   1.16–1.4 together. Square lakes are rare on Earth. Our node squares are not a natural shape.
4. **Real coasts have natural breaks** at some scales (for example about 1 km on the coast of Australia). **No real
   coast has a break exactly at a grid spacing and lined up with the grid.** Our 8 km staircase has exactly that. The
   bench can find it and turn it red.
5. **Our own ground predicts rocky-like coasts from 4 to 25 km and smooth arcs below about 2 km**, even with a perfect
   shore law. The fine noise is smooth below 3 km by the way it is built. R2's 1 km level can change that.

**Question 2 — how water and land stay one shape across detail levels.**

6. **The engines that keep a shore consistent make one curve the master and bend the ground to meet the water near
   it.** Proland and Flight Simulator do this. The others paint the water on the ground as it is drawn, and they accept
   that the shore slides when the detail changes.
7. **Every published error measure limits a HEIGHT error. A shore moves SIDEWAYS by that height error divided by the
   slope.** On a gentle coast that sideways move has no limit. No paper measured it. Our bench must measure it, from
   above, on a gentle coast.
8. **The published pattern is "a detail level drops only what it cannot show".** For a shore this means: a ring swap
   may move the shore by at most one cell of the coarser ring. Variant O, restated as finding 6 of the design asks, is
   that rule.
9. **A small lake may vanish when it is smaller than a cell, but only once, and it must never come back.** Our fixed
   half-threshold at the far rings already gives "never come back". An exact lake area at every ring and "never come
   back" do not come free together.
10. **Only two works measured their shores against a law** (one graphics paper for coasts, one geology model for lake
    shores). **Nobody measured a shore across detail levels.**

**What the bench should paint** (§6): today's rule as the control; the pure contour of the ground as a second control;
variant N with the ground bent to its own line (one field); variant O restated as a bounded sideways move; and, for
the far rings only, a coast curve carried as one byte per grid edge beside today's rule and the height band. **THE
DECIDING NUMBER** is variant O's largest sideways step on the gentle coast, beside the walls where its bound acts. If
both stay inside the one-pixel line, the papers and ruling B8 both point to O. If not, N stays until R2.

**Example.** A hull 150 km over the belt's coast sees a stretch of about 200 km. On Earth that stretch reads about
1.2, and its length measured with a 0.9 km ruler is about 1.8 times its length measured with a 16 km ruler (ARITH:
17.8^0.2). Today the same stretch reads 1.0, with a sharp bend in its length plot at exactly 8 192 m. As the hull climbs
to 900 km, each ring swap may move the drawn coast by at most one pixel; today nothing below the node moves, because
nothing below the node is drawn.

---

## 1. THE COAST LAWS AND THE GATES FOR THE BENCH

### 1.1 The length law and the estimators — one band per method

- **The length law** (C1): `L(ε) = k·ε^(1−D)`, with `ε` the ruler length. Halving the ruler multiplies the length by
  `2^(D−1)`: 1.07 at D = 1.1, 1.15 at D = 1.2, 1.26 at D = 1.33 (ARITH).
- **What that means for a rung swap.** A rung-(r+1) cell is two rung-r cells. So each swap halves the finest ruler
  the picture can show. At D = 1.2 the finer rung adds 15 % of shore length below its own scale, and it changes
  nothing above 7 of its cells (ARITH). That is W16 item 2 ("refine, never contradict") as a statistic.
- **From 8 km to 128 m (64 times)** the length grows 1.52 times at D = 1.1, 2.30 at D = 1.2, 3.94 at D = 1.33
  (ARITH, CHECKED). Today's staircase grows by 1.0, plus at most a factor 4/π from its steps.
- **The estimators differ, so the bands differ (CORRECTED).** The first draft pooled five methods into one band. The
  verifier refused that. The bench compares our coast with Earth only by the same estimator at a similar cell size:

| estimator | Earth cell | ruler range | Earth values | source | our rung |
|---|---|---|---|---|---|
| correlation dimension (Grassberger–Procaccia) on 11 503 segments of ≈ 200 km | ≈ 200 m | one coast fitted over 300 m – 20 km | sediment shores "close to 1"; rocky "about 1.3 or larger"; most probable 1.2; 1 146 segments at 4/3 ± 5 %; West Scotland 1.30 ± 0.04 | C2 (CHECKED) | 7 |
| yardstick over about three decades | 15″ ≈ 460 m | three decades | ten coasts 1.19–1.43 (± 0.02–0.07): Norway 1.425, British Columbia 1.359, NE USA 1.329, China 1.298, NW Australia 1.273, N Brazil 1.239, Britain 1.233, Iceland 1.229, Croatia 1.216, Colombia 1.191 | C3 (CHECKED) | 9 |
| divider on atlas maps (Richardson's data) | atlas | 10 – 1 000 km (the figure's ranges UNVERIFIED) | west Britain 1.25, Australia 1.13, South Africa 1.02 ("one of the smoothest in the atlas") | C1 (CHECKED) | 13 |
| box counting, 17 sizes | — | 0.5 – 1 000 km | Australia 1.143 (1.15 % from Richardson's 1.13); the scaling stops below about 1 km | C9 (CHECKED) | 10–13 |
| GIS on 1:24 000 maps | ≈ 12 m (ARITH, 0.5 mm on the map) | — | Maine: SW 1.11, S-central 1.35, N-central 1.23, Cobscook 1.37, rest of NE 1.18 | C10 (CHECKED) | 7 |
| perimeter–area slope by pixel edges on 131 063 islands | 30 m | — | slope 0.52 ± 0.01; a single-loop D of 1.04 by the contour law (the comparison with lake shores is UNVERIFIED, §1.5) | C28 | 7 |
| box counting of fjord coasts | — | — | Norway ≈ 1.52 | C11 (UNVERIFIED) | — |

- **The fjord upper edge stays UNVERIFIED** until Feder (C11) is read. Abril's Norway by yardstick is 1.425 (C3).
- **A D near 1.0 is lawful on a sediment coast.** So the D reading alone can never turn a staircase red. The lattice
  gates (§1.7) do that.

**Example.** The bench reads the belt's coast at rung 7 over 200 km segments by the correlation dimension, and it
prints Boffetta's band beside it. It reads the same coast at rung 9 by the yardstick and prints Abril's ten coasts
beside it. It never prints one pooled band.

### 1.2 The contour law — the coast follows the ground

- **The law** (C4, CHECKED). For a rough surface with Hurst exponent H (how the relief grows with scale), ONE contour
  loop has the dimension `Df = (3 − H)/2`. The WHOLE level set (every loop together) has `D = 2 − H`. These are two
  different objects, not two rival theories (CORRECTED; the first lake report said they "differ").
  - A divider or a perimeter–area measure on one shore compares with `(3 − H)/2`.
  - A box count of all shore lines in one square, every island and lake included, compares with `2 − H`.
  - The law is for Gaussian self-affine surfaces. It rests on a conjecture that is exact at H = 0 and H = 1 and checked
    by simulation between them (measured 1.28 ± 0.015 at H = 0.5 against 1.25 predicted, cited by C4).
- **Real coasts against the law** (C3 values, ARITH, CHECKED):

| coast | observed D | law D = (3 − H)/2 | difference |
|---|---|---|---|
| China | 1.298 | 1.295 | +0.00 |
| NW Australia | 1.273 | 1.264 | +0.01 |
| Britain | 1.233 | 1.227 | +0.01 |
| Iceland | 1.229 | 1.205 | +0.02 |
| Croatia | 1.216 | 1.215 | +0.00 |
| Colombia | 1.191 | 1.207 | −0.02 |
| British Columbia (fjords) | 1.359 | 1.307 | +0.05 |
| Norway (fjords) | 1.425 | 1.303 | +0.12 |
| N Brazil | 1.239 | 1.122 | +0.12 |
| NE USA (Maine to the Carolinas) | 1.329 | 1.180 | +0.15 |

- **This is a reading, not a gate (CORRECTED).** The H error bars in C3 are ± 0.08 to ± 0.37. For Britain the law's D
  is 1.23 ± 0.18. A tolerance of 0.05 has no support. The bench calibrates this line on a synthetic Gaussian field
  (CHECK-D-FBM, §1.8), where H is known, and never on Earth's coasts.
- **The same number from two other roads.**
  - Erosion damped by the coast's own roughness gives D = 1.332(3), up to the width of the eroded band: *"whatever its
    initial shape, a rocky shore will end fractal"* (C5, CHECKED).
  - A coast eroded in a field of rock hardness has D = 1.33 for random hardness, and from 1.34 (H = 0) down to 1.04
    (H = 1) for correlated hardness (C6, CHECKED).
- **What it means for T9.** A coast's D must never be drawn. Three lawful roads give it: the ground's H (the contour
  law), the hardness field's H (the erosion front, our W5 beds), and history (ice, a rise of the sea).
- **Our spectrum's prediction** (ARITH on `crates/terrain/src/body.rs:50-67`, CHECKED). The code sets each octave's
  slope as a bump that peaks at 3.125 km (`O_PEAK` 7, `SIGMA` 1.4). The local Hurst exponent is then
  `H(λ) = 1 + 2x/(σ² + x²)`, `x = (o − O_PEAK)·ln 2`:

| wavelength | 25 km | 12.5 km | 6.25 km | 3.125 km | 1.56 km | 780 m | 390 m |
|---|---|---|---|---|---|---|---|
| H | 0.34 | 0.29 | 0.43 | 1.00 | 1.57 | 1.71 | 1.66 |
| one-loop D = (3 − H)/2 | 1.33 | 1.36 | 1.28 | 1.00 | 1.0 | 1.0 | 1.0 |

  - PREDICTED for the ground's own waterline: D ≈ 1.28–1.36 between about 4 and 25 km (a rocky Earth coast), and
    D → 1.0 below about 2 km (smooth arcs). Earth's rocky coasts stay near 1.2–1.3 down to 300 m (C2).
  - Caveats (CHECKED): the ridged octaves are not Gaussian; the roughness factor cuts the fine relief on plains; W6
    holds the drawn surface on its ground's side; and between 8 and 32 km the drawn coast mixes the solve and the noise.
    So the bench measures it.
  - A note for the owner under T9: `body.rs:993` says *"THE WHOLE SPECTRUM IS A DRAW"*. Under the contour law that draw
    sets the coast's D.

**Example.** A pilot lands on the coast stand's shore under the ground's own waterline. She sees rocky bays and capes
5–20 km wide, and between them smooth arcs a kilometre long, like a sandy coast (PREDICTED).

### 1.3 The sea smooths below its level; rivers and ice roughen above it

- On Earth the line 100 m under the sea is smoother than the line 100 m over it (C7). D stays almost constant from 0 to
  3 000 m (rivers), and it is highest (up to about 1.8) where the mean year is near 0 °C (ice).
- **CORRECTED.** The main run of C7 used a 2′ grid (about 3.7 km), and its sea floor is predicted from satellite
  altimetry, which is smooth by nature. So part of the smooth line under the sea can come from the data. The line
  G-ISOLINE-JUMP stays a reading only.
- Today's coast stands at that jump because the sea rose over a river-cut land. The sea FELL slowly to −134 m from
  29 000 to 21 000 years ago, reached its lowest level at about 21 000 years, rose about 12 m per 1 000 years from
  16 500 to 8 200 years ago, and stayed almost constant for the last 2 500 years (C23, CORRECTED).

### 1.4 The coast types and what makes each one

| type | share of Earth's coast | cause (the land behind it) | D |
|---|---|---|---|
| cliffed | ≈ 52 % (C20) | high relief; a narrow shelf on a collision coast (C18); hard rock stands out as headlands, weak rock retreats into bays, and the pair can be stable with no change of rock along the coast (C17) | rocky ≈ 1.2–1.35 by C2/C3 |
| sandy | 31 % of the ice-free coast (C21) | sediment; a wide low shelf on a trailing-edge coast (C18) | ≈ 1.0–1.15 (C1 South Africa, C2, C10 SW Maine) |
| barrier islands | ≈ 10 % of continental coasts; 2 149 islands, 20 783 km (C22) | *"a flat piece of land…, sand…, waves…, and a recent sea-level rise that caused a crooked shoreline"* (C22) | — |
| capes, spits, sand waves | — | waves that reach the shore at more than about 45° (35°–50° across the formulas), from kilometres to hundreds of kilometres (C16); needs 8o's wave climate | — |
| ria (drowned river valley) | inside the 22 % tidal class (C19) | a relative rise of the sea after the valleys were cut, and too little sediment to fill them | — |
| fjord and fjärd | 26 %; 75 % of the coast north of 70° N (C19) | hard rock AND the maximum glacier cover AND the coast shape AND the bathymetry; deeper than 1 000 m at Sognefjord; a moraine sill at the mouth (C19) | 1.36–1.43 by C3; 1.52 by C11 (UNVERIFIED) |

- **The coast follows the drainage.** On the front of a straight, rising mountain belt the river mouths stand about
  half the distance to the main divide apart (spacing ratio 1.91–2.23 in 11 belts; the Nepalese Himalaya is an
  exception; C24). **CORRECTED:** the band holds for linear mountain fronts only, not for any coast, and the claim that
  it holds "whatever the climate and the tectonics" is UNVERIFIED.
- **Bays.** No paper we found gives a global size law for bays or headlands. D and the island law carry the size
  statistics instead.
- **Equilibrium bays.** A log-spiral bay `r = e^(θ·cot α)` lies in the lee of a headland under one main wave direction,
  α from 41.26° to 85.64° on four beaches (C14). The parabolic bay (C15) is UNVERIFIED. These are shapes of a few
  kilometres and need 8o's wave direction.
- **What we compute already, and what we lack.** We compute the hardness (W5), the divides and the rivers. 8e computes
  the ice line. The fjord class can then be derived as a hard bed under the glacial-maximum ice. **The one input we lack
  is a history of the sea's level** (§1.9).

**Example.** A pilot flies the home planet's polar coast. Where the beds are hard and 8e says the ice stood at its
maximum, the law says fjords with sills and D near 1.4. On the warm trailing coast it says sand and D near 1.1. Without
a sea-level history, no coast shows drowned-valley bays.

### 1.5 Islands

- **One prediction (CORRECTED).** The count of islands larger than `a` follows `N(A > a) ∝ a^(−B)` with `B = 1 − H/2`.
  Mandelbrot's rule `B = D/2` gives the same number when D is the dimension of the whole LEVEL SET, `2 − H` (C4, C28).
  The first draft put the single-loop D into Mandelbrot's rule and found a false disagreement.
- Mandelbrot 1975 gives K ≈ 0.65 as the mean island exponent (C25, through C28). The range 0.5 (Africa) to 0.75
  (Indonesia) is UNVERIFIED.
- A 2025 manuscript measures 131 063 islands on a 30 m model: perimeter–area slope 0.52 ± 0.01 by pixel edges, and
  *"subject to intense erosion"* (C28). By the contour law that is a single-loop D of 1.04. **The comparison "sea shores
  are much smoother than lake shores" is UNVERIFIED:** it sets a pixel-edge estimator beside other estimators, and
  C2 says rocky coasts are most probably 1.2 by the correlation dimension. The bench states its estimator before it
  picks a sea-coast band.

### 1.6 The measuring rules (the traps)

1. **The small end bends toward 1.** Near the grid size the slope of the length plot falls toward D = 1 (C7). The bench
   starts at 7 cells. **CORRECTED:** 7 cells is our own choice, not Earth's measured law (C7's smallest box was about
   1.8 cells of its 2′ grid). It gets its own control: CHECK-D-FBM read from 2, 4, 7 and 14 cells.
2. **The large end fills.** Near the square's size every box is full and D → 2. Stop at a tenth of the square. One
   decade then needs a square of 700 cells: 90 km at rung 7, 717 km at rung 10, 5 734 km at rung 13 (ARITH).
3. **A good r² proves nothing** (C8). Print the slope at each ruler length, per octave of the ruler.
4. **Two perimeters.** A divider step can cross the coast more than once. Print both the first crossing (the complete
   perimeter) and the last crossing (the accessible perimeter). A fjord coast separates the two. (The first draft cited
   C9 for this trap; the verifier did not find it there. It is a known property of the divider; its source is
   UNVERIFIED here.)
5. **The method moves D.** Algorithm choices move the mean D by about 0.1 (C7). So §1.1's rule: one estimator, one band.
6. **The staircase.** A line drawn along cell edges is `|cos φ| + |sin φ|` times longer than the true line; the mean
   over φ is 4/π = 1.273 (ARITH, CHECKED). It shows as a bump in the slope at the step's scale, not as a D.
7. **Measure on a sub-cell contour.** Trace the shore by marching squares on (surface − level). Never count pixel
   edges, except for the one line that copies C28's pixel-edge method.
8. **A grain is not a defect.** A real coast may lean on a rock grain (Maine's NE strike, C10). A lattice test judges a
   lean against a rotated and shifted null, never against the face grid's axes alone.
9. **Breaks are natural at feature scales (CORRECTED).** West Britain is not self-similar; its complexity peaks at
   characteristic scales (C8). Australia's scaling stops near 1 km (C9). The red line is a break AT the lattice scale
   and lined up with the lattice, never "a break anywhere".

### 1.7 THE COAST GATES

Every line reads the DRAWN shore: the boundary of the columns whose drawn surface stands under their standing water
(the sheet's own `s < l` rule, finding 19 of `shoreline_design.md` §19). ★ = a line that can turn red on its own.
The other lines are readings that stand beside the picture.

| gate | how to compute it on the drawn shore | band, and its kind | rungs | source | TODAY PREDICTED |
|---|---|---|---|---|---|
| ★ G-SHORE-LATTICE.<stand>.rNN | the four-fold tangent moment `|Σℓ·e^(4iθ)| / Σℓ` of the traced shore's segments; the divider D along the two lattice axes against D along the diagonals; the same two numbers on the same coast rotated by 45° and shifted by half a node (the null) | inside the null's spread (a bench tolerance from the null; W13's null method) | 7, 9, 10 | the directional D of L8 (SECONDARY); W13.3 | RED (a staircase's moment is near 1) |
| ★ G-COAST-KINK@node | the local slope of `log L` against `log ε`, per octave of ε; its change across ε = one node (8 192 m) and across 7 cells of the rung | no larger than the spread of the same change at the same ε over many CHECK-D-FBM fields of the same size and on the shifted null (a bench tolerance; the first draft's "≤ 0.08" has no source) | 7, 9, 10 | C8, C9 (breaks at feature scales are natural) | RED at 8 192 m |
| ★ G-COAST-REFINE@r-(r+1) | W16 item 2 as a statistic: `L_r(ε) / L_(r+1)(ε)` for every ε of 7 or more cells of rung r+1, and `|D_r − D_(r+1)|` on the shared band | a bench tolerance calibrated on CHECK-D-FBM drawn at two rungs (the first draft's ± 0.02 and 0.04 have no source) | 0–13 | C1; W16 | GREEN (nothing below 8 km refines) |
| G-COAST-D.corr.<stand>.r07 | the correlation dimension on 200 km segments of the drawn coast, 0.9–20 km | Earth by the same method: sediment ≈ 1; rocky ≈ 1.3 or more; most probable 1.2 (a reading) | 7 | C2 | ≈ 1.0 |
| G-COAST-D.yard.<stand>.r09 | the yardstick D over two decades or more, first and last crossing | Earth by the same method at ≈ 460 m: 1.19–1.43 (a reading) | 9 | C3 | ≈ 1.0 below 8 km |
| G-COAST-D.div.<continent>.r13 | the divider over 10–1 000 km; the box count over the same range | Earth: 1.02–1.25 (divider), 1.143 (box, Australia) (a reading) | 13 | C1, C9 | the solve's own reading |
| G-COAST-CONTOUR | the measured D against `(3 − H)/2`, H from the drawn ground's variogram over the same band | calibrated on CHECK-D-FBM; Earth's ten coasts only as a reading (§1.2) | 7, 9 | C3, C4 | the staircase breaks the law |
| G-COAST-WIND | the slope of the winding-angle variance against `ln L`; the angle's spread at L = 5, 10 and 20 km | ONLY on a stand whose measured D is about 4/3: slope `2(D − 1)/D`, a Gaussian spread (CORRECTED: C2 tested it on the D ≈ 4/3 segments only; C3 finds coasts break the SLE picture) | 7 | C2, C3 | the staircase is its failing control |
| G-ISLAND | `N(A > a)` over drawn islands of 16 cells or more; the pixel-edge perimeter–area slope beside C28 | `B = 1 − H/2` (a reading); C28's 0.52 ± 0.01 at 30 m (a reading, a different cell) | 7, 10, 13 | C4, C25, C28 | a reading |
| G-ISOLINE-JUMP | the box-count D of the −100 m and +100 m lines over 6.7–67 km | Earth: D(−100) < D(+100), partly from smooth sea-floor data (a reading only) | 10 | C7 | UNMEASURED |
| G-OUTLET-SPACING | on the SOLVE: the divide distance over the spacing of the trunk-river mouths | 1.91–2.23, ONLY where a straight, rising range front meets the sea | nodes | C24 | UNMEASURED |
| G-BAY-VALLEY | the share of bays deeper than 2 cells that head at a trunk mouth | a reading until the sea-level history is ruled | 7, 10 | C19, C22 | — |
| G-HEADLAND-HARD | the sign of the link between a seaward bulge and the bed's hardness | positive on Earth (a reading until R2) | 7, 10 | C10, C17 | — |

### 1.8 The failing controls

Each control must turn its line red when the thing it checks is broken:
- **CHECK-D-LINE:** a straight line reads 1.00 ± 0.01.
- **CHECK-D-KOCH:** a Koch curve reads 1.262 ± 0.02. (A Koch curve also shows that a D alone is not realism, C27.)
- **CHECK-D-FBM:** a Gaussian field with known H (0.3, 0.5, 0.7). One contour loop reads `(3 − H)/2` (about 1.25–1.28
  at H = 0.5, C4), the whole level set reads `2 − H`. It calibrates the H estimator, the divider, the 7-cell start, and
  the spread that G-COAST-KINK and G-COAST-REFINE use.
- **CHECK-D-STAIR:** an 8 km staircase read at rung 7 must turn G-SHORE-LATTICE and G-COAST-KINK red, and it must give
  G-COAST-WIND its ± 90° turns.

### 1.9 The owner question these laws raise (not a bench item)

Rias and barrier islands need a recent rise of the sea (C22, C23). The solve holds the sea at its freeboard level for
the whole age (W8). **Should the home planet carry a glacial sea-level history?** Two roads exist:
- a law: the ice volume of 8e's cold phase gives the drop of the sea;
- a T9 "unrecoverable history": a seed identity choice, as the tilt is.

Without such a history a lawful coast is a steady coast: deltas, cliffs and graded shores, and no drowned-valley bays.

---

## 2. THE LAKE LAWS AND THEIR GATES

### 2.1 The laws

1. **The perimeter–area law** `P = C·A^(d/2)` (L1, L2, L3). Earth's POPULATION band is d from 1.16 to 1.40:
   - 106 Scandinavian lakes on 1:50 000 maps: d/2 = 0.63 (0.59–0.66), ln C = 2.07 (1.98–2.15) (L1, CHECKED);
   - 111 global lakes: d/2 = 0.58 (0.56–0.60), ln C = 1.75 (1.63–1.88) (L1, CHECKED);
   - lakes by origin: D/2 = 0.65 (0.59–0.70) (L2, CHECKED);
   - the corrected census: d = 1.4 above 0.46 km² (L3, CHECKED; the corrected Methods say the global d is sensitive to
     the weighting function);
   - percolation theory: 4/3; Kent & Wong's "typical" 1.28 (L6, SECONDARY).
2. **One lake is smoother than the population (CORRECTED).** The individually measured d of the 111 global lakes has
   a median of **1.10** and a range of **1.02–1.37** (L1 Table 1). So a single lake with d = 1.08 is Earth-like.
3. **The scale law** (L1, L4): `D_L ∝ δ^(1−d)` with δ the ruler, and `D_L ∝ A^((d/2) − 0.5)` at a fixed ruler.
   - The paper writes "+14 % per doubling of area", but the exponent is 0.14, so the factor is `2^0.14` = **1.10 per
     doubling** (ARITH, CHECKED).
   - HydroLAKES quotes +15 % of shoreline for each doubling of the map resolution (L5 through L4). That is d ≈ 1.20
     (ARITH).
   - HydroLAKES' rise of `D_L` from 1.6 to 7.8 across its size classes is mostly this bias. The bias-corrected index
     `D_BC = P / (2√π·A^(d/2))` has no trend with area: Kendall τ = −0.02 (−0.16 to 0.12) (L1, CHECKED).
   - Lake Vänern reads `D_L` = 3.72 at 1:1 000 000 and 7.38 at 1:10 000, which is d ≈ 1.15 (ARITH). An average d in
     place of the lake's own d gives a 21 % error over a tenfold map scale (L1).
4. **The contour law for lakes** (L9, L11, P2). A lake is the ground's level set inside its catchment, up to its pass.
   Cael & Seekell 2022 SET H = 0.4 from the literature for Earth's topography; a free fit gives 0.38 (L9, CHECKED).
   `(3 − 0.4)/2 = 1.30` matches Earth's lakes, but across the literature's H range 0.4–0.7 the loop law gives 1.30–1.15,
   so the match leans on the low end. The model *"implies that lakes are formed by flooding preexisting landscape
   depressions"* (L9). This is B8 item 1 in one sentence.
5. **The shape constant by origin** (L2, CORRECTED). A circle has C = 2√π = 3.545 (collapse and crater lakes). A
   triangle has C = 6/3^(1/4) = 4.559 (interdune lakes). Glacial lakes average C = 8.33. **Square lakes (C = 4) "are
   sometimes found in permafrost regions where ice wedges are present"** — the paper has no "only" and no scale. Dolines
   on a 1 m model are 39.8 % circular (elongation ≤ 1.21), 41.1 % elliptical and 13.5 % elongated (L14).
6. **The inlet law** (L2): `N = (2/π)·D_d·C·A^(D/2)`, with `D_d` the drainage density. The inlet count rises 67 % per
   doubling of area; the mean error is 0.17 inlets with the junction-angle factor and 2.35 without it. It checks B8
   item 2 (the rivers follow the water).
7. **The relief outside predicts the relief inside** (L15; L4's "topographic lake depth" extends the slope near the
   shore inward, with a 3 × 3 slope kernel; Heathcote's buffer is 25 % of the equal-area diameter). A qualitative line:
   no wall and no drowned strip at the drawn shore.
8. **A knee and a directional D** (L8, SECONDARY). On Titan 21 lakes show a knee (D changes with the scale, which marks a
   change of process), and D differs east–west from north–south. On our lattice that is a direct grid test.
9. **The roughness–fetch classifier** (P11, P12). River-cut shores (Fort Peck, Murray, Lanier) classify at more than
   90 %. Before 8o our shores should read as drowned river valleys.

### 2.2 What Earth's lakes would read at our rungs (CORRECTED)

The scale law holds only where the lake is much larger than the ruler: `√A ≥ 10δ`. Every cell that breaks that rule
reads "the law does not apply". ARITH from L1's fits, with an ASSUMED map ruler of 25 m (1:50 000) and 90 m (the
global set):

| lake | √A | rung 7 (10δ = 1.28 km) | rung 10 (10δ = 10.24 km) | rung 13 (10δ = 81.92 km) |
|---|---|---|---|---|
| 67 km² (one node) | 8.2 km | `D_L` 2.15–2.53 | the law does not apply | the law does not apply |
| 670 km² | 25.9 km | 2.6–3.4 | 1.9–2.0 | the law does not apply |
| 6 700 km² | 81.9 km | 3.1–4.6 | 2.2–2.7 | at the rule's edge: not a gate |

Reference shapes on the same measure: a square reads `D_L` = 2/√π = 1.128; a disc traced on pixel edges reads 4/π =
1.273 (ARITH, CHECKED). So a pixel-edge count ranks our node squares as rounder than a disc. That is why every lake
line traces a sub-cell contour.

### 2.3 THE LAKE GATES

The same rules as §1.7: the DRAWN set, a marching-squares contour, the rung's ruler δ stated, only lakes with
`√A ≥ 10δ`. ★ = a line that can turn red on its own.

| gate | how to compute it on the drawn lakes | band, and its kind | rungs | source | TODAY PREDICTED |
|---|---|---|---|---|---|
| ★ G-LAKE-CONTOUR.<stand>.rNN | the drawn outline against the contour of the DRAWN ground at the lake's level inside its catchment: the Hausdorff distance, and the XOR area over A | ≤ one extractor step (the law is B8 item 1; the step is the extractor's own) | 0–10 | L9, L11, P2; B8 | RED (the node square) |
| ★ G-LAKE-FRACTAL.rNN | the POPULATION slope of `ln P` against `ln A` over the drawn lakes | Earth: 1.12–1.40 (the published fits and their 95 % ranges) | 7, 9, 10 | L1, L2, L3 | UNMEASURED |
| ★ G-LAKE-DBC.rNN | Kendall τ between `D_BC` and A | Earth: −0.16 to 0.12 (τ = −0.02) | 7, 10 | L1 | UNMEASURED |
| G-LAKE-RICHARDSON (per lake) | `P(δ)` for δ = 1, 2, 4 … cells up to √A/4; the slope `1 − d`; the knee's position | the distribution of per-lake d against Earth's 111 lakes (median 1.10, range 1.02–1.37); a knee AT the node scale and lined up with the lattice is red through G-SHORE-LATTICE | 7 | L1, L8 | a knee at 8 km |
| ★ G-SHORE-LATTICE (lakes) | §1.7's moment and directional D, on each lake outline and on all lakes together | inside the null's spread | 7, 9, 10 | §1.7 | RED |
| G-LAKE-SHAPE-C | ln C per lake; the share of lakes with d ≈ 1 AND C ≈ 4 AND axes on the lattice | a reading; the authority is G-SHORE-LATTICE | 7, 10 | L2 | the node-square share |
| G-LAKE-AXIS | the bearing of each lake's principal axis against the flat null (finding 15 of §19) | inside the null | 7 | §19 | leans on the lattice |
| ★ G-LAKE-CLIMB-LENGTH@r-(r+1) | per lake `P(r+1)/P(r)` and `A(r+1)/A(r)`, only where `√A ≥ 10` coarser cells | length: `2^(1−d)` with the lake's own d and its fit error (Earth's range gives 0.77–0.93); area: steady within W16's step | 0–13 | L1, L4; W16 | ≈ 1.0 (no sub-node shape) |
| G-LAKE-INLETS | inlets per drawn lake against `(2/π)·D_d·C·A^(D/2)` | +67 % per doubling of area (a reading) | 7 | L2 | — |
| G-SHORE-THROUGH-SLOPE | the wet slope against the dry slope in a buffer of 25 % of the equal-area diameter on each side of the drawn shore | continuity (a qualitative reading) | 3, 7 | L4, L15 | — |

**Two corrections to the 2026-09-22 lake survey.**
- It writes "d = 1.00 below [the 0.46 km² break]". In the text the verifier read, d = 1.00 belongs to the SWEDISH lakes
  under 4.7 km² (1.34 above). A global "1.00 under 0.46 km²" was not found. Neither touches our scale: one node is
  67 km².
- Use +10 % of `D_L` per doubling of area, not +14 %.

**Example.** A tarn on the belt, seen from 450 km, measured at rung 7. G-LAKE-CONTOUR says whether its water fills its
own hollow up to the notch in the rim. G-LAKE-RICHARDSON says whether its bays stop at 3 km or at 8 km. G-LAKE-CLIMB-
LENGTH says whether its shore shortens smoothly or pops as the hull climbs to 900 km. A reading of d = 1.08 on this one
tarn is Earth's median, not a defect.

---

## 3. HOW PROCEDURAL WORK MAKES SHORES, AND WHAT IT MEASURED

### 3.1 Three ways to make a shore

| way | who | at every level of detail | our laws |
|---|---|---|---|
| **(a) an input curve** | Génevaux 2013 (P1), Huftier 2026 (P7), Fischer 2022 (P6), Doran & Parberry 2010 (P5, coastline agents) | the same curve at every level | an authored or drawn curve: T9 and SL5 refuse it |
| **(b) the zero crossing of an amplified height** | Cortial 2019 (P3), Voss / Mandelbrot level sets (P14) | the shore moves when a level drops detail (our W6 measurement: 234 m and 848 m) | Cortial does not treat level of detail |
| **(c) the flood of a basin inside its catchment up to its pass** | Cordonnier 2016 (P2, the interactive preview only, CORRECTED), Fill–Spill–Merge, Palermo 2024 (P11) | a law, but its detail change is our Question 2 | meets B8 and K2 by construction |

### 3.2 The works, and what each measured

| work | how it makes the shore | scale | did it measure the shore against a law? |
|---|---|---|---|
| P1 Génevaux et al. 2013 | the user's input contour Γ; river mouths on Γ, large mouths in concave parts; *"the river network can subdivide only upstream; thus … cannot represent deltas or oxbow lakes"* | 970–3 386 km²; about 1.5 kB per km²; "native multiresolution" | no (Horton–Strahler, Rosgen, a look beside a real island) |
| P2 Cordonnier et al. 2016 | a super-graph of lakes with pass heights; the drawn lake is the points flowing to the lake's bottom under the pass height (the interactive preview, not the final render) | continental | no (a look beside the San Gabriel mountains) |
| P3 Cortial et al. 2019 | the zero crossing of the amplified height; crust at 50–500 km, sea at 0, SRTM exemplars at 90 m | planet | no (*"validated by a geophysics expert"*) |
| P4 Peytavie et al. 2019 | the bed carved into the ground; the water surface a separate blend tree; the bank where the two meet | river reach | no (visual checks and timings) |
| P5 Doran & Parberry 2010 | coastline agents grow the land outline; beach agents flatten beside it | map | no (*"no formal attempt … to evaluate"*) |
| P6 Fischer, Boeckers & Zachmann 2022 | a spline plus Voronoi or noise growth; a lake is a noise-perturbed border around a river cell | map | no (a look beside a Magadan elevation model) |
| P7 Huftier et al. 2026 | nested iso-contours grown by Open Eden Growth from a sketched coastline, meshed by marching triangles | region | histograms of elevation, slope and drainage against the Western Alps; the Strahler order; no shore law ("contour fractalization" is future work) |
| **C27 Thorimbert & Chopard 2016/2018** | noise heights; the coast is the zero crossing | 512–1 024 px maps | **YES: box counting from 2 to 64 px, compared with Richardson's values (the set includes the German land frontier, which is not a coast); D depends on the persistence and the octave count; a warning that D alone is not realism** |
| **P11 Palermo et al. 2024** (geoscience) | red noise (spectral slopes −1.6, −1.8, −2.0; variance 10 000 m²) + an inverted-cosine basin + stream-power incision of 6 % of the relief, then a flood; then wave erosion (by fetch, saturated at 20 km) or uniform erosion | lakes | **YES: the Morlet wavelet roughness against the fetch, classified by a Kullback–Leibler divergence, 96 % correct; seven Earth lakes (Fort Peck, Murray, Lanier > 90 %; Sebago, Rotoehu > 90 %; Prošćansko, Kozjak about 70 %)** |
| P12 NEWTS1.0 (Palermo et al. 2024, GMD) | the model's code: a 40 m flood; uniform erosion gives *"rounded features punctuated by pointy promontories"*, wave erosion *"blunted headlands"* | — | a four-connected grid within 2 % of the analytic solution |
| P8 Derzapf 2011, P9 Teoh 2008, P13 Tewelde 2013 | rivers first then terrain on a planet; coastal erosion; shore sinuosity of a flooded landscape | — | UNVERIFIED |

### 3.3 What it means for us

- **Our rule is way (c)**, and B8 item 1 already states it. The graphics papers offer no shore law to copy.
- **Nobody in graphics measured a shore across levels of detail.** Only C27 measured a coast's D at all, and it tested
  no winding, contour or lake law.
- **P11 is the model for 8o.** It classifies a shore as river-cut, wave-cut or uniformly dissolved. Before 8o our shores
  should classify as river-cut (drowned valleys). The article is CC BY 4.0; the code licence at the USGS record is
  UNVERIFIED. Running NEWTS as an instrument needs the owner's word; we copy no code.

**Example.** A pilot lands on the shore of a plain lake. Under way (a) her shore would be a line a designer drew. Under
way (b) it would move when the ring swaps. Under way (c) it is the edge of the water that stands in the ground's own
hollow — and Question 2 decides whether it stays there as she climbs.

---

## 4. THE LEVEL-OF-DETAIL TECHNIQUES, AGAINST OUR FINDINGS AND OUR LAWS

### 4.1 Two families, and which one removes walls (CORRECTED)

The first draft said "the papers force the ground to the water, never the water to the ground". The verifier refuted
that. Two families exist:

| family | who | walls and drowned strips | the shore across levels |
|---|---|---|---|
| **A: the ground is bent to the water** | Bruneton & Neyret 2008 (G9: `z_m = b·z_e + (1 − b)·z_o`, `b` = 1 inside the feature and 0 across a border strip, the `b` values combined by max *"to ensure continuity"*); Proland's clamp `max(zf, 0.0)` per level before the morph (G10; in the Terrain1 tutorial and the demo shader; the docs say `z_m` is *"by default equal to zf"*); Flight Simulator 2020's flattening under water polygons by a distance field (E3) | removed inside one level | only BN08's CURVE part keeps the position: every level rasterises the same ancestor curve. Proland's clamp does NOT: each level's shore is its own zero crossing, and the blend slides the shore between them. The clamp also floods every pocket below 0. |
| **B: the water is paint or an overlay on the ground as drawn** | Cesium (G13: a mask read in the fragment shader only); Schneider & Klein 2007 (G16: *"independent of the underlying terrain geometry"*); Flight Simulator 2020 (*"Drawn using terrain geometry"*); Outerra (*"the terrain patches are also being used for the water"*); our W18 sheet | none from the sheet itself | the shore is where the drawn ground crosses the level, so it slides with the ground |

- **Our position.** W18 puts the sheet in family B. W6's hold is a weak family-A rule: it bends the ground to its own
  side by a quarter of its height. The root of the §19 findings is that the side comes from one field and the ground
  from another. **The published cure is family A with ONE field**: the ground near the line is built from the same
  field that decides the side. Finding 1's fix `B′ = B + b·(W_b + S_b − B)` has exactly BN08's form.

### 4.2 The error measures limit a height; the shore moves sideways

- **The published pixel limits** support T7 and W17.1 (p99 ≤ 1 px, max ≤ 2 px):
  - about 1 px avoids popping (Hoppe 1998, G2);
  - popping disturbs above 2 px with no morph, and a morph lets the tolerance double or triple (Lindstrom & Pascucci
    2002, G7);
  - the root-mean-square and the 99.9th percentile screen error both stay under 1 px (Losasso & Hoppe 2004, G4);
  - 4 m of data tolerance at about 2 px on a 1024 × 768 screen (Ulrich 2002, G3);
  - 1.1 px per texel (BN08, G9).
- **Every one of them limits a VERTICAL error.** A shore moves sideways by `δ_h = δ_v / |∇z|`. No paper states or
  measures that sideways error.
- **The nested bound gives a band, and it has no limit on flat ground** (ARITH on ROAM's wedgies, G8, CHECKED). A
  coarse triangle's wedgie of thickness `e_T` holds every finer surface. So a swap can flip a side only inside a band of
  half-width `e_T / |∇z|`. As the slope goes to zero (a plain lake), the band grows without limit.
- **The anisotropic measure hides the worst case.** It makes *"errors viewed from above appear relatively smaller"*
  (G7). A shore's sideways move is largest exactly when the eye looks down. So the bench judges shores on a top-down
  paint as well as an oblique one.
- **Losasso & Hoppe's coastline sentence** (*"visually indistinguishable … except at the sharp color transition
  associated with the coastline"*) is about the COMPRESSION error, not a level-of-detail transition (CORRECTED). It
  still shows that a small vertical error is visible wherever colour depends on the sea level.

**Example (ARITH).** The belt's gentle coast rises 1 m in 50 m. A 10 m change of the drawn height moves the shore 500 m:
one pixel at the 704 km swap (rung 9 → 10), where one pixel is 512 m. W6's measured moves (234 m at the 128 m rung,
848 m at the 256 m rung, DOC) are about 1.8 px and 3.3 px at their own swaps.

### 4.3 Geomorphing

- **Distance-based morphs** (Losasso & Hoppe's `z′ = (1 − α)·z + α·z_c` over a transition of n/10; CDLOD's morph over
  *"the last 15%-30%"* of each range, G6; Proland's blend `clamp((d/l − k − 1)/(k − 1), 0, 1)`; Ulrich's
  `t = clamp(2ρ/τ − 1, 0, 1)`, whose formula is UNVERIFIED because the notes' text did not extract). Hoppe 1998's
  geomorphs are TIME-based: one second over the frame rate (CORRECTED).
- **A morph turns a pop into a slide.** Only Proland applies a water rule inside the morph: the clamp at each level,
  then the morph of the clamped surfaces, with the target read from the parent's clamped channel (CHECKED). That is our
  W18.7 law: the water's morph target is the same rule read from the ground's morph target.
- **What it means for us.** A morph cannot cure a side that changes at a swap (finding F5). It only makes the change a
  slide, and the slide's length is the sideways move of §4.2.

### 4.4 Nested bounds and the maximum pyramid (CORRECTED)

- The saturation rule `ε_i = max(ε̂_i, max over the children of ε_j)` (G7) and ROAM's nested wedgies (G8) keep each
  child's band inside its parent's. From them a cell is SURELY DRY where `z_T − e_T > W`, SURELY WET where
  `z_T + e_T < W`, and ambiguous between. That gives K1 and W16 item 2 a number the bench can compute (SHORE-BAND).
- **A maximum pyramid does NOT bound an interpolated coarser mesh.** The first draft claimed "water can only shrink on
  the climb" by construction. The verifier gave a counter-example: node A = 100 m and node B = 0 m; a fine sample of
  100 m a quarter of the way from A to B lies under the interpolated coarse surface of 75 m. Only a DILATED maximum (the
  maximum over each coarse vertex's whole interpolation support) bounds every finer surface. Tevs 2008's maximum
  mipmaps (G21, UNVERIFIED) bound ray-cast boxes, not interpolated meshes.
- **Use:** the dilated maximum pyramid is a CONTROL for the bench (no new water on the climb, and the coast move that it
  costs, which is W10's defect). It is not a cure.

### 4.5 Masks per level: the area against the nesting (CORRECTED)

- **Cesium** ships a water mask for each tile at each level: 1 byte for a tile of one kind, else 256 × 256 = 65 536
  bytes; 0 is land, 255 is water, and the values between are there *"to support anti-aliasing of the coastline"*. A tile
  with no mask reuses its ancestor's with a translation and a scale. Only the fragment shader reads it. Nothing makes a
  level refine its parent (G13, CHECKED).
- **BN08** carries water as a mask channel `m` in one RGBm texture, 192 × 192 over a 25 × 25 elevation grid (so its
  shore is 7.7 times finer than its mesh), with *"correct filtering via the mipmapping of masks"* (G9).
- **Castaño 2010** keeps the covered area equal at each mip level by solving a NEW threshold per level with a bisection
  (G15).
- **The trade-off.** With ONE fixed threshold at every level, "a wet parent has a wet child" and "a dry parent has a dry
  child" both hold (mean ≥ t ⇒ max ≥ t). With a threshold PER level, a parent whose mean lies between the two
  thresholds is wet while all four children are dry. So exact area and nesting are not free together.
- **What it means for us.** W15's count pyramid, with one fixed half and "a tie is wet", already nests, and a pond that
  folds never returns under it. The bench keeps that rule, prints the area each fold loses, and uses a per-level
  area-keeping threshold only as a FAILING CONTROL for AREA-NEST.

**Example (from the verifier).** A belt tarn covers 0.45 of each of four child cells. Under a per-level threshold of
0.40 at the coarser rung and 0.5 at the finer, it shows at the coarser rung and at none of the finer ones: water that
appears on the climb.

### 4.6 Vector curves and distance fields

- **BN08** (G9, CHECKED): curves are 2-D Bézier splines with a width `w` and a border width `w_f`; an area is a loop of
  curves. Every level takes its position and its height profile from the uncut ANCESTOR curve *"to avoid
  z-discontinuities"*. A curve is refined until it is under one pixel of the appearance texture. A river's profile is
  `z(u_i) = min over j ≤ i of z_s,j`, so a river never flows uphill. A 3 × 3 Gaussian runs on `z_m` only where `b ≠ 0`.
- **Proland** (G10): *"The graph for any other quad is produced by clipping the graph of the parent quad, and by
  flattening the result with half the threshold used for the parent graph."* So a finer level refines a coarser one.
- **Yu et al. 2009** (G11): a river channel is two bank curves and a flow rate; the data *"could come from a GIS
  database, be simulated from an erosion model, or be generated … procedurally"*; the water meshes are *"created from
  the channel boundary curves"*; each tile holds a quadtree of bank segments for distance queries.
- **Schneider & Klein 2007** (G16): a polygon becomes a volume and a stencil marks the pixels inside: *"per-pixel
  exact"* at every terrain level.
- **Green 2007** (G14): a 4096² binary image becomes a 64² signed distance field with its edge at 0.5; two fields keep a
  sharp corner.
- **What it means for us.** All of these keep the shore one shape across levels because the curve is AUTHORED data that
  every level reads. For us the curve must come from the solved field (T9). §5.3 treats that.
- ARITH, not in any paper: cross-fade two signed fields and cut the blend at zero. The shore then moves continuously
  from the fine line to the coarse line. That answers F5 (a side change at a swap is a pop).

### 4.7 Features that vanish (CORRECTED)

- **Progressive Meshes** (G1): a material edge keeps its topology and geometry through the energy term `E_disc`; a
  change of its topology is *"permitted but penalised"*; features *"too small to be visible when viewed from a
  distance"* can vanish, in one fixed order.
- **BN08's "w < 2δ" rule drops only the FOOTPRINT** (the ground's flattening) of CURVES narrower than two elevation
  texels. The curve itself still draws in the appearance texture (the water mask), and the sentence names curves (roads,
  rivers), not areas (lakes). The first draft read it as "Proland folds a lake away". That is wrong.
- **So a monotone fold rung per lake is our own proposal**, not Proland's rule. W16's own example already allows it (*"it
  simply folds away with the ground it sits in"*), and W15's fixed threshold gives "never back" at the far rungs. Below
  R₁ the side is a point sample, and finding 7 of §19 (a pond that appears on the climb and flickers at 9 → 10) stays
  open until the bench measures a monotone rule there.

### 4.8 The kind of pyramid

- **Proland interpolates:** a finer tile is a quarter of its parent upsampled plus a stored integer residual (1 m at
  scale 1); *"a red sample is simply equal to the parent sample"*; the edge samples use (−1/16, 9/16, 9/16, −1/16) and
  the centres the tensor product; a 2-pixel border is enough for any number of steps (G10, CHECKED). These are our own
  Catmull-Rom weights at a half (`shoreline_design.md` C7). The limit surface is C¹, not the C∞ that Proland's doc says
  (G4 calls the same 4-point scheme C¹; CORRECTED).
- **Losasso & Hoppe** downsample with an optimal 11 × 11 linear filter and name an "impulse" downsample (keep the fine
  value) as future work (G4).
- **Ours takes the mean of four** (W17.3; `crates/terrain/src/artifact.rs:3533`). A mean can stand under the sea on the
  land side, which was W10's defect; W10 and W15 cure the SIDE with the mask counts, but the far HEIGHTS still disagree
  with the side near a coast.
- **What it means for us.** An interpolating decimation plus residuals makes "coarse = fine at the coarse nodes" true by
  construction, at the price of aliasing in the far heights. That is a question for how R2 stores its 1 024 m level,
  not for step 4. UNMEASURED.

### 4.9 Skirts for the water sheet (W18.6 item 3)

- Ulrich's rule: a skirt must reach *"below the full LOD mesh at the edge, as well as below any possible
  simplifications"* (the quote is UNVERIFIED; the notes' text did not extract). Cesium keeps edge index lists *"to add
  skirts to hide cracks"* (CHECKED).
- For our sheet the skirt must reach below its neighbour's water edge by the coarser rung's chord dip (10 m at rung 14,
  167 m at rung 16, 2.7 km at rung 18, DOC W18.2) plus the morph's own change. The bench prints SEAM-GAP.

### 4.10 Our findings against the techniques

`shoreline_design.md` §19 lists 24 findings against design v2; §18 lists v1's 23 (F1–F23), most of which §19 carries
forward. The table maps each open finding to what the papers offer.

| finding (§19 #, §18 F) | the published technique | what it asks of us | law touched | verdict |
|---|---|---|---|---|
| #1, #5 — a wall and a drowned strip; the surface on B, the side on the mirrored read | BN08's flattening; Proland's clamp per level | build the drawn node-scale ground from the side's own field inside the presence, so the ground meets W_b on the side's line | W6 stays; T9 needs a law for the border's width (BN08 sets `w_f` by hand; Flight Simulator tunes a strength) — ours: the read's own reach | ADOPT the pattern; line G-SIDE-VS-GROUND |
| #2, #4 — the sea drawn over side-dry columns; a pocket's rim takes the sea | Proland's clamp shows the TRAP (it floods every pocket below 0); BN08 flattens only inside the curve; Cesium takes the look from the mask | the side decides where water is; no height rule may create water | W10 item 2, unchanged | lines CHECK-DRAWN-IS-SIDE and G-SEA-OVER-DRY, band 0 |
| #3, #6, #12 — O's hold tests the coarser field, not the drawn side; O's band edge; O's wet hold | "a level drops only what it cannot show": BNH10 cancels waves shorter than `N_min` × the projected cell and moves their slope variance into the shading (G12); Proland halves its threshold per level | restate O as a continuous, bounded displacement from the node-scale line (at most one coarser cell), with the hold and the fade on that same field | W6's letter (owner's word) | ADOPT the pattern; lines SHORE-STEP, CLAMP-BINDS |
| #7, #21, F4 — a pond appears on the climb or flickers at 9 → 10 | PM96's ordered collapses; one fixed threshold (§4.5) | a monotone fold at every rung, the point-sampled rungs below R₁ included | W15.1 item 4 (the tie) if a new rule changes it | lines LAKE-APPEAR and LAKE-FLICKER, band 0 |
| #8, F1, F17 — the fold does not bound the sideways step | no paper bounds a sideways step; the wedgie band is unbounded on flat ground (§4.2) | split the line: a sideways step (max ≤ 1 coarser cell) and a separate fold-away list | T7/W17.1 restated for a shore's sideways move | lines SHORE-STEP, LAKE-FOLD |
| #9, F10 — SL5: an exact fold on a body whose node is not a power of two | BN08 draws each level from the curve, never from the children | fold by rung cells (they always nest) and test on the 714-cell body | SL5 | line AREA-NEST on both bodies |
| #10, #16, #20 — square halos on node lines; the smooth dish; the fade removes walking-scale detail | BN08 changes the ground only where `b > 0` and blurs only there; Outerra and Flight Simulator build a distance field from the shore | a border measured from the shore's own distance, not from the node square of the presence | T9 needs the width's law | lines HOLD-RAMP, halo extent, fine RMS near the shore |
| #11 — the held read depends on its pass order | the published 4-point filter is linear, so its order does not matter; the per-pass clamp is ours | take the mean of both orders | none | test 2b on axis-swapping seams |
| #14 — the rule depends on which rows a host holds | CDLOD's *"consistent filter"* on both sides of an edge (G6) | the side at every rung is a function of data both hosts hold | SL10; SL6 if the far rungs need rows | a drift line with and without the pilot's tiles |
| #17 — the cost | BN08: 7.7 ms a frame on 2008 hardware (4 ms to draw, 3.7 ms to update) | measure the whole rung-7 and rung-9 rings with `VD_CLIENT_UNPACED=1` | none | line COST |
| #19 — the lines read the law's side, not the drawn water | Losasso & Hoppe measure the drawn mesh against the finest level; Stuetzle et al.: RMS and maximum error *"fail to capture whether a reconstructed terrain preserves the drainage network"* (G17) | read every line on the drawn set | none | every line of §6 |
| #24, F19 — what crosses the wire | Cesium ships a mask per tile per level; Veloren sends its whole map once (E13) | the published pattern is data shipped for the far view | SL6 | §5.4 |
| #13, #15, #18, #22, #23 — the amendments list, the spilling tongue, the recommendation, HR5, the kernel cases | no published technique | ours to fix | — | — |

**Example.** A pilot descends on the belt's coast. Family B alone (today) gives her no wall, but the shore slides by its
height change over the slope as each ring passes. Family A with one field gives her ground that meets the sea exactly on
the line the side draws. The bound of §4.2 then says how far that line may move at each swap, and the bench measures it
from above.

---

## 5. THE PLANET ENGINES' COASTS, AND A COAST DERIVED FROM THE SEED

### 5.1 What the engines do with the coast

| engine or product | the coast is… | across levels of detail | water on which geometry | what we take |
|---|---|---|---|---|
| Proland (E1 = G10) | vector curves and areas; a level clips its parent's graph and refines it at half the threshold | a finer level refines a coarser one; the ground is flattened to the feature in a border | the ocean is a screen-space grid on the sea sphere and reads no terrain; *"only works for deep water … not for coasts and shores"* (G12) | the curve-as-master pattern and the refinement rule |
| Microsoft Flight Simulator 2020 (E3, Asobo GDC 2022) | vector water bodies; per-tile WaterMask, WaterBitmap, WaterBlurMask; a distance field from the polygons | *"DEM: flat water VS hill ⇒ some popping when LoD changes ⇒ adjust flattening strength according to height difference"*; a hand-edited blur mask | *"Drawn using terrain geometry"*; level 15 read back for collision | the flattening band; its tuned strength is refused (T9) |
| Flight Simulator X (E4, Szofran GDC 2006) | vector shorelines drawn into the land-class map as wide textured lines that *"help soften the edge"* | *"vector river data and raster elevation data may not match … a river climbs a mountain ridge"*; fixed by hand in GIS | terrain | the defect class; the manual fix is refused |
| Flight Simulator community (E5, E6) | — | *"change in water level along a straight line"* at a tile edge, fixed by a hand-made ramp; add-ons still sell fixes for *"rivers floating above their valleys, lakes at incorrect elevations … incorrect coastlines"* | — | a water's level belongs to the water, not to a tile (our D1) |
| Cesium (E7 = G13) | a raster mask per tile per level, shading only | no refinement rule | the mesh never reads the mask | the coverage value (an area fraction), if it comes from ONE curve |
| Outerra (E2) | the terrain's crossing of the sea; a distance map only for tiles whose highest point is over the sea and lowest under it: *"finds the nearest point of opposite type"*, then a filter and a Sobel gradient | "LOD level switching" is on its to-do list among other errors (the post does not say the shore distance changes with the level; CORRECTED) | *"the terrain patches are also being used for the water"* | support for W18; no cure |
| Unreal Engine 5 (E8) | authored splines; a quadtree water mesh where *"four quads morph into a single quad"* | the far mesh is *"located just below the Ocean Water Body base height position, so it limits clipping and avoids any seams"* | its own mesh | the sunk far mesh is the cure W18.5 refuted for grazing views |
| SpaceEngine (E9) | sea level plus the crossing; rivers are a landform function | *"Fixed many artifacts with ocean rendering"*; no method published | terrain | nothing |
| No Man's Sky (E10) | mesh-based water, dual marching cubes | no method published (the GDC 2017 talk is a video, UNVERIFIED) | — | nothing |
| Star Citizen (E11) | Alpha 4.8.0 *"New water … rivers, lakes, oceans"* | no method published in text | — | nothing |
| Elite Dangerous (E12) | no liquid on any landable body | — | — | avoids the problem |
| Veloren (E13) | the whole map sent once: 13 bits of altitude per 32 m chunk, *"plans to use 7 bits for water depth"*; *"TODO: Add information for rivers (currently, we just prerender them on the server, but this is not a great solution for LoD"* | far-view rivers unsolved | — | the same shape as our mask sent once |
| Procedural World (E14) | lakes from a path search to the sea level over ~50 km tiles | no rule | — | nothing |

### 5.2 The published water-level artefacts and their cures

| artefact | where | published cure | for us |
|---|---|---|---|
| a river flows uphill; a lake at the wrong height; a wrong coastline | Google Earth (G9 Fig. 1), Flight Simulator X, Flight Simulator add-ons | bend the ground to the feature (BN08); edits by hand; paid fixes | bend the ground: YES. By hand: NO (T9) |
| flat water on a hill pops at a detail change | Flight Simulator 2020 | a tuned flattening strength | refused (T9); the width must come from the rung's cell and the body's bound |
| a lake's level steps along a tile line | Flight Simulator 2020 | a hand-made ramp | our D1 gives each water ONE level; BN08 computes a feature's height on the ancestor curve for the same reason |
| the far water mesh clips the terrain | Unreal Engine 5 | sink it a little | W18.5 refuted this for grazing views |
| coasts break the ocean model | BNH10 (G12) | none; coasts are future work *"for planet-size rendering"* | — |

### 5.3 A coast derived from the seed: three grades

The planet-engines group asked whether a vector or distance coast can come from our solved field instead of an
authored line. All three grades below are PROPOSALS; nothing here is in a paper.

- **(a) From the coast mask alone.** Both hosts can do it today with no new data. But the mask does not say WHERE
  between two nodes the shore runs. Near the pilot the rows say it, and they disagree with a mask-only curve. So the
  shape would depend on which tiles a host holds (finding 14), or the band would drag the ground by up to half a node's
  relief. **Refuse.**
- **(b) From the rows, with ONE BYTE per crossing edge, shipped once.** For each lattice edge whose two nodes carry
  different side words, one byte says where along the edge the node-scale ground crosses the water's level. The byte
  is a pure function of two nodes' rows, and the rows are a pure function of the seed and the charter, so it is static
  shape in SL10's sense. Both hosts find the crossing edges in the mask they already hold, in node order, so no index
  crosses. Where a pilot holds tiles, the client derives the same bytes, and a drift line compares them.
  - Size: ARITH 0.2–0.5 MB at one byte (32 m steps), on an assumed 0.5 million mixed nodes and about 1.27 crossings per
    node length of shore. UNMEASURED. The mask is 2.2 MB (DOC W16.2); D8's height band is 0.9–3.5 MB (DOC ARITH).
  - The integer form (proposal): the crossing `t = S0/(S0 − S1)` rounded down to k/256 with k held in 1…255 (so a node
    centre keeps its own word, K5); the curve traced through the mask's boundary by marching squares, a saddle cell
    joining its WET corners (W15's "a tie is wet"); smoothing by the recipe's own 4-point weights, exact in integers,
    each new point held inside its cell's crossing edges; the far fold by the exact area inside the curve (the shoelace
    formula on integer coordinates), wet where twice the area is at least the cell's area.
- **(c) From D8's band of heights.** The client derives the curve itself everywhere near a coast. It is larger, and it
  also gives true heights where the ground bends to the curve.

**The condition of one field (this file's own analysis, ARITH).** A curve carried to the far rungs must be the SAME line
the near rungs draw, or the swap at R₁ contradicts W16. Variant N's line (the held Catmull-Rom of the mirrored node
words) and a 4-point curve through edge crossings are two different curves inside a cell. So either:
- the near rungs also read the carried curve (the coast is the master at every rung, as in the engines), or
- the carried curve is measured against N's line at R₁ and accepted only if the two stay within a pixel at every far
  swap. The bench prints that distance (LINE-N-VS-V).

**The laws, for grade (b).**
- **SL10:** holds — one integer function on both hosts, with a drift line.
- **SL6:** an ask (§5.4).
- **T9:** the crossing is a law. The border's width must come from the rung's cell and the body's bound, never a typed
  metre.
- **W6:** its letter holds (no fine HEIGHT octave carries a column across the water).
- **W10, W15:** amended. Between nodes the side is the curve's, not the nearest node's word, and areas replace node
  counts. Both still nest.
- **W16:** held by construction for the curve.
- **F7, F8:** integer only. The card holds no artifact, so this runs on the CPU, as the mask does today.

**Sideways octaves on the curve — NOT lawful yet (UNVERIFIED as a technique).** The planet-engines group proposed to
put the shore's sub-node detail in a SIDEWAYS displacement of the curve, driven by the recipe's seeded noise, with
amplitudes from Question 1's D. No source supports it: BNH10 moves filtered waves into the SHADING; it displaces no
coast. It conflicts with three things:
- the coast report: a coast's D must come from the ground's H, the hardness or the history, never from a drawn scatter
  that the ground does not share;
- B8 item 1: *"the water inside the ground's own hollow … a ragged outline that follows the ground"* — sideways detail
  puts bays where the ground has no valley, and then the band bends the ground to the coast instead;
- geometry: a curve displaced along its normal by more than its radius of curvature crosses itself (a belt bay 2 km
  wide under a 1.5 km octave, ARITH).

So the bench does not paint it until the owner rules (T9, W6, B8). If it is painted later, it needs a self-crossing
line.

### 5.4 The SL6 question, as it would be asked

Only if the bench shows that the far rungs need data the client does not hold:
- **What data:** either (i) one byte per lattice edge where the coast mask changes side (grade b), or (ii) the rows'
  heights within two nodes of every mixed node (D8's band, grade c).
- **From which realm to which:** the planet's shard to every client, once, with the artifact's head, like the mask.
- **Why the receiver cannot compute it:** the rows are the solve's output, and the solve runs only on the shard. Near
  the pilot the client holds the tiles and derives the same bytes; far away it holds only the 2-bit mask.
- **What doing without costs:** past the 704 km swap the side stays the node word's majority fold. A smooth pond becomes
  a node-majority shape there (up to half a node, about 5 px, ARITH in `shoreline_design.md` D8) — a W16 breach — or
  the tiles must reach out to where half a node is one pixel, about 3.7 Mm (ARITH, the same place).
- **The sizes** are ARITH today. The bench prints them (BAND-BYTES, V-CROSSING-BYTES) before the question is asked.

**Example.** A tarn about 3.3 km across in a pass on the belt. Under grade (b) the rung-10 ring draws it inside its
carried curve at 704 km. It folds by W15's fixed rule at the first rung where it holds less than half of every cell it
touches (about rung 12, near 1 800 km, for the design's own example pond, DOC `shoreline_design.md` §0), and it stays
folded as the hull climbs. Without new data the 704 km swap turns the same tarn into its node's majority word, and the
owner sees a step.

---

## 6. THE RECOMMENDATION FOR THE SHORELINE BENCH

### 6.1 The rules to paint

The bench follows `shoreline_design.md` §7.1: a what-if beside the shipped column read, the recipe's integers where a
public function exists and `f64` otherwise, and CHECK-WHATIF-TODAY (the what-if under today's rule equals the shipped
column bit for bit). The rule the bench selects is re-measured by the same lines in its integer form before any
kernel work.

| rule | what it is | family (§4.1) | laws it amends | PREDICTED |
|---|---|---|---|---|
| **R0 TODAY** (control) | W6's hold + W10/W15's side + W18's sheet | B, with a weak A | none | the node squares, the teeth, the 8 km kink |
| **C0 THE PURE CONTOUR** (control) | the drawn ground's own waterline at the near rungs, W6's hold switched off | none | — (a control only) | D ≈ 1.28–1.36 at 4–25 km (§1.2); SHORE-STEP RED (W6's 234 m and 848 m come back) |
| **N1** | variant N + ONE FIELD (finding 1's fix a: the ground blended to `W_b + S_b` inside the presence) + the dry hold per water (finding 2, 4) + the symmetric held read (finding 11) + a monotone fold below R₁ (finding 7) | A, one field | W10 and W15 (finding 13) | continuity lines GREEN; D ≈ 1.0 below 8 km; a lattice lean on the steep coast; a smooth dish near every shore |
| **O1** | variant O restated as finding 6 asks: the drawn ground's own crossing, moved from the node-scale line by at most ONE COARSER CELL, continuously, with no band edge; the hold and the fade read that same field (finding 12) | A, one field, with a bounded sideways move | W6's letter (the fine octaves may carry a column across within one coarser cell), and W10 and W15 as N1 | ragged shores that follow the ground; D rises toward C0's above ~2 km; SHORE-STEP bounded by construction; the risk is a wall where the bound acts |
| **V** (far rungs only) | N1's line (or O1's, identical at R₁) carried past R₁ as one crossing byte per mixed edge, derived by the bench from the rows (no wire), with the exact-area fold | A (curve as master) | W10, W15; SL6 if chosen | area nests on both bodies; its bytes and LINE-N-VS-V decide it |
| **R5-DILATED** (control) | a dilated maximum pyramid for the far heights | — | — | no new water on the climb; the coast moves about one pyramid node (W10's defect) |

**Not painted:** sideways octaves (§5.3, the owner's word first); a per-level area-keeping threshold (only as the
failing control of AREA-NEST); the anisotropic error measure.

### 6.2 The lines — every one on the DRAWN set, top-down AND oblique

**Continuity (Question 2).** ★ = band 0 or a pixel line; each has a failing control.

| line | what it counts | band |
|---|---|---|
| ★ CHECK-DRAWN-IS-SIDE.rNN (every rung 0–18) | dry columns whose surface stands under their standing water, plus wet columns over it (the sheet's `s < l`) | 0 |
| ★ G-SEA-OVER-DRY.rNN | side-dry columns under a drawn sheet | 0 |
| ★ G-SIDE-VS-GROUND.rNN | `|drawn ground − W|` on the side's line | half the rung's extractor step |
| ★ WALL.rNN at rungs 0, 3, 7 | the step across the drawn shore against the column's own slope step | a quantile with a count (finding 19) |
| ★ SHORE-STEP@r-(r+1), every swap 0 → 18 | the drawn shore's sideways step in pixels, nearest crossing, on the gentle coast first | p99 ≤ 1 px, max ≤ 2 px (T7, W17.1) |
| SHORE-BAND@r-(r+1) | the bound `e_T / |∇z|` in pixels, per slope class | a reading (unbounded as the slope goes to 0) |
| SHORE-CREEP | the slide inside each crossfade band | a reading |
| ★ CLAMP-BINDS (O1) | the share of shore columns where O1's bound acts, and WALL on those columns | the WALL band |
| ★ LAKE-APPEAR@r-(r+1), LAKE-FLICKER@r | lakes that appear on the climb; lakes dry at r and wet at r ± 1 | 0 at every swap 0 → 18 |
| LAKE-FOLD, ★ FOLD-RETURN | per lake: the fold rung, its width, the swap distance; returns at a coarser rung | a list; returns 0 |
| LAKE-AREA@r | each lake's drawn area at each rung against its finest area | a reading (the area each fold loses) |
| ★ AREA-NEST | parent wet ⇒ some child wet, parent dry ⇒ some child dry, every cell, on the home planet AND the 714-cell body | 0 |
| ★ SEAM-GAP, ★ FACE-SEAM | the sheet's crack at a chunk seam; the held read on both sides of an axis-swapping face seam | 0 |
| LINE-N-VS-V@R₁ | the distance between N1's line and V's carried curve | in rung-R₁ cells (a reading) |
| ★ V-CROSSING-DRIFT | bytes derived from the pilot's tiles against the shard's bytes | 0 |
| BAND-BYTES, V-CROSSING-EDGES, V-CROSSING-BYTES, V-SADDLE-CELLS | the sizes for the SL6 question | readings |
| HOLD-RAMP, HALO-EXTENT, FADE-SLOPE-MAX, FINE-RMS-NEAR-SHORE | findings 10, 16, 20, 21 | readings |
| COST | chunk build over the whole rung-7 and rung-9 rings on the plain and the belt, `VD_CLIENT_UNPACED=1` | a reading |

**Shape (Question 1).** The ★ lines of §1.7 and §2.3: G-SHORE-LATTICE, G-COAST-KINK@node, G-COAST-REFINE,
G-LAKE-CONTOUR, G-LAKE-FRACTAL, G-LAKE-DBC, G-LAKE-CLIMB-LENGTH; the D readings by estimator; and the controls of §1.8.

### 6.3 The stands, the rungs and the pictures

- **The six stands** of `shoreline_design.md` §7.2: the belt, the coast 300 km inland (steep), the plain (about forty
  lakes), the reference stand, the pocket stand, and **the gentle coast** (slope ≤ 1/50, the decisive stand for the
  sideways step, §4.2). Plus the 714-cell body for AREA-NEST.
- **The rungs:** 0, 3 and 7 for the walls; 7, 9 and 10 for the shape gates; every swap 0 → 18 for the continuity lines.
- **The panels:** top-down wet maps and oblique hillshades for R0, C0, N1 and O1 at rungs 3, 7 and 11, each captioned
  with the rule, the rung and "no far shadow"; the far panel at rung 11 also for V against today's fold. Ruling B6
  item 2 still applies to the step's report: the three stands, the fourth stand framed like the reference, and its
  skyline numbers.
- **Libraries:** the shape gates need a contour tracer and distance measures. `geo` and `imageproc` as dev-only
  dependencies wait for the owner's word (`terrain_models_and_crates_survey.md` §5); the bench can also trace the
  contours itself.

### 6.4 What decides the owner's choice

**Choice A — the near shore's look (rungs 0 to R₁): O1 or N1.**
- **Choose O1** when, on the gentle coast and on the plain:
  1. SHORE-STEP reads p99 ≤ 1 px and max ≤ 2 px at every swap, top-down;
  2. CHECK-DRAWN-IS-SIDE, G-SEA-OVER-DRY, LAKE-APPEAR and LAKE-FLICKER read 0;
  3. WALL stays inside its band on the columns where CLAMP-BINDS says the bound acts;
  4. G-SHORE-LATTICE and G-COAST-KINK stand nearer the null than N1's, and G-LAKE-FRACTAL and G-LAKE-DBC stand nearer
     Earth's band than N1's.
  Then O1 is B8 item 1's letter (the ground's own waterline), the papers' physics (the contour law, §1.2), and the
  papers' detail rule (drop only what the ring cannot show, §0 item 8, §4.10). Its price is W6's amended letter.
- **Choose N1** when O1 fails a continuity line that N1 passes. The shape gates then print N1's price: D ≈ 1.0 below
  8 km and a lean on the lattice, until R2's 1 024 m level brings the sub-node shore through the same law.
- **If both fail** a continuity line, neither goes to the kernel, and the bench iterates (the rule of `shoreline_design.md`
  §19 "WHAT FOLLOWS").
- **THE DECIDING NUMBER:** O1's largest sideways step on the gentle coast, beside the walls where its bound acts.

**Choice B — the far shore (rungs above R₁) and the wire: no new data, D8's band, or V's crossing bytes.**
- Compare on the same stands: LAKE-AREA loss at each fold, SHORE-STEP at 9 → 10 … 13 → 14, AREA-NEST on both bodies,
  LINE-N-VS-V, and the bytes.
- The SL6 question (§5.4) goes to the owner with those numbers.

### 6.5 What the bench does not decide

- The sideways octaves (§5.3): an owner ruling first (T9, W6, B8).
- The sea-level history (§1.9): a question for 8e.
- W6's quarter and D5's `K` (`shoreline_design.md` §15 Q3, Q4): the owner's words, asked beside the panels.
- The dev-only libraries (§6.3).

**Example.** The owner opens the plain's panels at rung 7. R0: forty blue tiles. N1: forty smooth ponds in shallow
dishes. O1: forty ragged ponds whose bays follow the dune valleys. C0: the same ragged ponds, but the census beside it
says their shores jump by hundreds of metres at a swap (PREDICTED from W6's 848 m). Beside O1 the census prints its
largest sideways step on the gentle coast in pixels, the walls where its bound acts, the ponds that appear on the climb,
and the lake population's d beside Earth's 1.12–1.40. On those numbers the owner chooses.

---

## 7. WHAT COULD NOT BE VERIFIED, AND WHAT THE VERIFIER CORRECTED

### 7.1 Not verified

- **Not read:** the parabolic bay coefficients (C15); Jiang & Plotnick's numbers (C12); Feder's Norway value (C11);
  Ashton, Murray & Arnoult 2001 (C16); Klinkenberg's review (C13); Goodchild 1988 (L11); Tewelde 2013 (P13); Derzapf
  2011 (P8); Teoh 2008 (P9); Carr et al. 2004 (G20); Tevs et al. 2008 (G21); Koppe 1902 (G23); Dyn, Levin & Gregory
  1987 (G24); Frasson et al. 2018 (G25, HTTP 403); the Flight Simulator thread *Water Polygon Accuracy* (a TLS error).
- **Read only in part:** Sharma & Byrne 2010 and 2011 (L7, L8: the per-origin D, `D_L` and elongation values);
  the GLOBathy method (L16); the wet-to-dry slope ratio of Heathcote 2015 (L15); the deltaic-barrier numbers of Stutz &
  Pilkey 2002 (C22); Mandelbrot 1967's figure ranges (C1); the island-exponent range 0.5–0.75 (C25); the Castelltort &
  Simpson statement (C24); Ulrich's morph formula and skirt quote (G3, the notes' text did not extract); the licence of
  the NEWTS code (P12); the Earth shore resolutions of 10 m and 30 m in P11.
- **Videos not watched:** No Man's Sky at GDC 2017; Star Citizen's CitizenCon talks and *The H2O Show*.
- **Assumptions:** the map-scale-to-ruler conversion of §2.2 (0.5 mm on the map); the 0.5 million mixed nodes behind
  V's size.
- **Every "ours" number here is PREDICTED or ARITH. None is measured.** The bench measures them.

### 7.2 The refuted claims and their corrected forms

| # | the first claim | the correction | where |
|---|---|---|---|
| 1 | C7 ran on 30″ land data, so "start at 7 cells" is Earth's law; C7 is a 1 km row | C7's main run used a 2′ grid (≈ 3.7 km); its smallest box was ≈ 1.8 cells; the 7-cell start is our choice with its own control; G-ISOLINE-JUMP is a reading | §1.1, §1.3, §1.6 |
| 2 | G-COAST-WIND is a red line on every coast | only on stands with D ≈ 4/3; C2 tested only those segments, and C3 finds coasts break SLE | §1.7 |
| 3 | one Earth band 1.00–1.15 / 1.15–1.35 / 1.33–1.52 | one band per estimator and cell; the 1.52 edge is UNVERIFIED | §1.1 |
| 4 | G-COAST-CONTOUR: `|ΔD| ≤ 0.05` | a reading; C3's H errors reach ± 0.37; calibrate on CHECK-D-FBM | §1.2 |
| 5 | the two island rules disagree (0.75 against 0.625) | they agree when D is the level-set dimension `2 − H` | §1.5 |
| 6 | `2 − H` and `(3 − H)/2` are two rival theories of the lake shore | they describe two objects: the whole level set and one loop | §1.2 |
| 7 | a per-lake d under 1.1 means ovals (red) | the per-lake Earth median is 1.10 (1.02–1.37); 1.16–1.40 is the POPULATION slope | §2.1, §2.3 |
| 8 | the climb shortens a lake's shore by 0.80–0.87 per swap | 0.77–0.93 over Earth's range, or a band from the lake's own d | §2.3 |
| 9 | the expected `D_L` table at 1 km and 8 km | every cell with `√A < 10δ` reads "the law does not apply" | §2.2 |
| 10 | "only" ice-wedge permafrost makes square lakes, "only" at metres | "sometimes found" there; no scale; the lattice test is the authority | §2.1 |
| 11 | no graphics paper measured its shores against a law | Thorimbert & Chopard (C27) measured coast D against Richardson's values | §3.2 |
| 12 | the papers force the ground to the water, never the water to the ground | two families exist (§4.1); only BN08's curve part keeps the position across levels | §4.1 |
| 13 | BN08 folds a lake narrower than 2δ away for good (Proland's rule) | it drops only the FOOTPRINT of CURVES; the look stays; a per-lake fold rung is our proposal | §4.7 |
| 14 | a per-level area-keeping threshold keeps W15's nesting | only one fixed threshold nests; exact area and nesting are not free together | §4.5 |
| 15 | a maximum pyramid means water only shrinks on the climb | only a DILATED maximum bounds an interpolated coarser mesh | §4.4 |
| 16 | Earth's coasts show no break, so any break is red | Earth has breaks at feature and process scales; the red line is a break AT the lattice scale and lined up with it | §1.6, §1.7 |
| 17 | the sea stood at −134 m from 29 to 21 ka | it FELL slowly to −134 m over that time and was lowest at ≈ 21 ka | §1.3 |

Five claims stay UNVERIFIED and are marked where they stand: the KINK and REFINE tolerances (now bench tolerances,
§1.7); the sea-against-lake comparison of C28 (§1.5); the outlet band beyond linear fronts (§1.4); Ulrich's formula
(§4.3); the sideways octaves (§5.3).

---

## APPENDIX A. THE REGISTER OF SOURCES

Each row: the citation, the link, the read status, the verifier's result, the numbers this file takes, and what it
means for us.

### A.1 The sea-coast laws

| id | citation and link | read | check | the numbers taken | for us |
|---|---|---|---|---|---|
| C1 | Mandelbrot B. 1967, *How long is the coast of Britain?*, Science 156:636–638, [doi](https://doi.org/10.1126/science.156.3775.636), [copy](http://gsp.humboldt.edu/OLM/courses/GSP_510/Articles/Mandelbrot1967.pdf) | OPENED | CHECKED (Richardson's data, reproduced; the figure's ruler ranges UNVERIFIED) | `L(G) ∝ G^(1−D)`; west coast of Britain 1.25; German land frontier 1.15; Spain–Portugal frontier 1.14; Australian coast 1.13; South African coast 1.02 | the length law; the divider band at rung 13 |
| C2 | Boffetta G., Celani A., Dezzani D., Seminara A. 2008, *How winding is the coast of Britain?*, GRL 35, [10.1029/2007GL033093](https://doi.org/10.1029/2007GL033093), [arXiv 0712.3076](https://arxiv.org/abs/0712.3076) | OPENED | CHECKED (D is the correlation dimension) | 11 503 segments of ≈ 200 km at ≈ 200 m; sediment close to 1; rocky ≈ 1.3 or more; most probable 1.2; 1 146 at 4/3 ± 5 %; West Scotland 1.30 ± 0.04 over 300 m – 20 km; `⟨θ²⟩ = a + (2(D−1)/D)·ln L`, a = 0.98; Gaussian spread at 5, 10, 20 km; D = 1.2 and 1.5 fail the winding law | the rung-7 band by the same estimator; the winding law only at D ≈ 4/3 |
| C3 | Abril A., Oliveira E., Moreira A., Andrade J., Herrmann H. 2024, *Coastlines violate the Schramm–Loewner Evolution*, [arXiv 2409.20315](https://arxiv.org/abs/2409.20315) (preprint) | OPENED | CHECKED | GEBCO 15″; `D = 1 + κ_Df/8`; ten coasts and their H (§1.2); H errors ± 0.08–0.37; D errors ± 0.02–0.07; contour lines are true fractals; κ_Df and κ_θ differ by up to 18 % | the yardstick band at rung 9; the contour-law reading |
| C4 | Kondev J., Henley C., Salinas D., *Nonlinear measures for characterizing rough surface morphologies*, [arXiv cond-mat/9907229](https://arxiv.org/abs/cond-mat/9907229) (the journal reference PRE 61 (2000) UNVERIFIED); Kondev J., Henley C. 1995, PRL 74:4580, [doi](https://doi.org/10.1103/PhysRevLett.74.4580) | arXiv OPENED; 1995 ABSTRACT | CHECKED | one loop `Df = (3 − α)/2` (Eq. 4.13); the level set `2 − α` (Eq. 4.11); `τ_ZO = 2 − α/2`; 1.28 ± 0.015 measured at α = 0.5 (cited) against 1.25; Isichenko's `(10 − 3α)/7` fails at α = 0; the conjecture `x_l = 1/2` | the contour law; which object the bench measures |
| C5 | Sapoval B., Baldassarri A., Gabrielli A. 2004, PRL 93:098501, [doi](https://doi.org/10.1103/PhysRevLett.93.098501), [arXiv cond-mat/0311509](https://arxiv.org/abs/cond-mat/0311509) | OPENED | CHECKED | `Df = 1.332(3)` up to the eroded width `σ ∼ g^(−4/7)`; *"whatever its initial shape, a rocky shore will end fractal"* | the erosion-front road to 4/3 (8o) |
| C6 | Morais P., Oliveira E., Araújo N., Herrmann H., Andrade J. 2011, PRE 84:016102, [arXiv 1102.5615](https://arxiv.org/abs/1102.5615) | ABSTRACT | CHECKED | random hardness D = 1.33; correlated D from 1.34 (H = 0) to 1.04 (H = 1); not fractal above a critical sea force | the hardness road (W5) |
| C7 | Baldassarri A., Montuori M., Prieto-Ballesteros O., Manrubia S. 2008, *Fractal properties of isolines at varying altitude revealing different dominant geological processes on Earth*, JGR 113, [10.1029/2007JE003066](https://doi.org/10.1029/2007JE003066), [arXiv 0811.1183](https://arxiv.org/abs/0811.1183) | OPENED | CORRECTED | SRTM30-plus, the main run on 2′; boxes 0.06°–0.6°; −100 m line near 1.0 (on altimetry-predicted sea floor), +100 m higher; D ≈ constant 0–3 000 m; up to ≈ 1.8 near 0 °C; the method moves D by ≈ 0.1; the slope bends low near the grid | the traps; G-ISOLINE-JUMP a reading |
| C8 | Andrle R. 1996, *The west coast of Britain: statistical self-similarity vs. characteristic scales*, ESPL 21:955–962, [doi](https://doi.org/10.1002/(SICI)1096-9837(199610)21:10%3C955::AID-ESP639%3E3.0.CO;2-Y) | ABSTRACT | CHECKED | not self-similar; complexity peaks at characteristic scales; a high r² proves nothing | print the slope per octave; natural breaks |
| C9 | Husain A., Reddy J., Bisht D., Sajid M. 2021, *Fractal dimension of coastline of Australia*, Sci. Rep. 11:6304, [PMC7973742](https://pmc.ncbi.nlm.nih.gov/articles/PMC7973742/) | OPENED (PMC) | CHECKED | 17 box sizes 0.5–1 000 km, D = 1.143 against 1.13; the scaling's lower limit ≈ 1 km | a natural break near 1 km |
| C10 | Tanner B., Perfect E., Kelley J. 2006, *Fractal analysis of Maine's glaciated shoreline tests established coastal classification scheme*, J. Coastal Res. 22:1300–1304, [doi](https://doi.org/10.2112/05-0474R.1); Island Institute, [Maine's fractal coast](https://www.islandinstitute.org/working-waterfront/maines-fractal-coast/) | ABSTRACT; page SECONDARY | CHECKED (only the geologically different compartments separate at 95 %) | SW 1.11, S-central 1.35, N-central 1.23, Cobscook 1.37, rest of NE 1.18; 1:24 000; granite headlands, weaker bays, a NE grain | D follows the rock; a grain is not a defect |
| C11 | Feder J. 1988, *Fractals*, Plenum | UNVERIFIED | — | Norway ≈ 1.52 | the fjord edge stays open |
| C12 | Jiang J., Plotnick R. 1998, Math. Geol. 30:535–546, [doi](https://doi.org/10.1023/A:1021790111404) | UNVERIFIED | — | (search summary only) | — |
| C13 | Klinkenberg B. 1994, *A review of methods used to determine the fractal dimension of linear features*, Math. Geol. 26:23–46 | UNVERIFIED | — | — | — |
| C14 | Yasso W. 1965, *Plan geometry of headland-bay beaches*, J. Geol. 73:702–714, [doi](https://doi.org/10.1086/627111) | ABSTRACT | CHECKED | `r = e^(θ·cot α)`, α 41.26°–85.64° on four beaches | 8o, sub-node |
| C15 | Hsu J., Evans C. 1989; Hsu J., Yu M., Lee F., Benedet L. 2010, *Static bay beach concept*, Coastal Eng. 57:76–91 | UNVERIFIED | — | — | — |
| C16 | Ashton A., Murray A. 2006, *High-angle wave instability and emergent shoreline shapes*, parts 1 and 2, JGR-ES 111, [10.1029/2005JF000422](https://doi.org/10.1029/2005JF000422), [10.1029/2005JF000423](https://doi.org/10.1029/2005JF000423); Ashton, Murray & Arnoult 2001, Nature 414:296–300 | 2006 ABSTRACT; 2001 UNVERIFIED | CHECKED | above ≈ 45° (35°–50°); km to hundreds of km; capes, spits, sand waves; Carolina Capes, Long Point | 8o's wave climate |
| C17 | Limber P., Murray A. 2011, Geology 39:1147, [doi](https://doi.org/10.1130/G32315.1); Limber P., Murray A., Adams P., Goldstein E. 2014, JGR-ES 119:854–873, [doi](https://doi.org/10.1002/2013JF002950) | ABSTRACT | CHECKED | hard rock → headlands; stable with no change of rock along the coast | G-HEADLAND-HARD |
| C18 | Inman D., Nordstrom C. 1971, *On the tectonic and morphologic classification of coasts*, J. Geol. 79:1–21, [doi](https://doi.org/10.1086/627583) | ABSTRACT | CHECKED | collision, trailing-edge, marginal-sea; shelf width and relief | the coast type from the plates |
| C19 | Dürr H. et al. 2011, *Worldwide typology of nearshore coastal systems*, Estuaries & Coasts 34:441–458, [PDF](https://link.springer.com/content/pdf/10.1007/s12237-011-9381-y.pdf) | OPENED | CHECKED | small deltas 32 %, tidal 22 %, lagoons 8 %, fjords and fjärds 26 %, little filter 12 %; the fjord rule; Sognefjord > 1 000 m; sill; 75 % north of 70° N; lagoons < 5 m | the fjord class from W5 and 8e |
| C20 | Young A., Carilli J. 2019 (online 2018), *Global distribution of coastal cliffs*, ESPL 44:1309–1316, [doi](https://doi.org/10.1002/esp.4574) | ABSTRACT | CHECKED | cliffs ≈ 52 %; sampled every 1 km; mangroves 17 % | a reading after 8e |
| C21 | Luijendijk A. et al. 2018, *The State of the World's Beaches*, Sci. Rep. 8:6641, [doi](https://doi.org/10.1038/s41598-018-24630-6) | ABSTRACT | CHECKED | 31 % of the ice-free coast is sandy | a reading after 8e |
| C22 | Stutz M., Pilkey O. 2011, J. Coastal Res. 27:207–222 (paper UNVERIFIED), through [Duke Today](https://today.duke.edu/2011/04/barrier); Stutz & Pilkey 2002, JCR SI 36:694–707 | press page SECONDARY; 2002 ABSTRACT | CHECKED (the 2002 deltaic numbers UNVERIFIED) | 2 149 barrier islands, 20 783 km, ≈ 10 %; Pilkey's four conditions | barriers need a sea-level history (§1.9) |
| C23 | Lambeck K., Rouby H., Purcell A., Sun Y., Sambridge M. 2014, *Sea level and global ice volumes from the Last Glacial Maximum to the Holocene*, PNAS 111:15296–15303, [doi](https://doi.org/10.1073/pnas.1411762111) | ABSTRACT | CORRECTED | a slow FALL to −134 m from 29 to 21 ka; ≈ 52 × 10⁶ km³ more grounded ice; a rise of ≈ 12 m/ka from 16.5 to 8.2 ka; almost constant for 2.5 ka | §1.9 |
| C24 | Hovius N. 1996, *Regular spacing of drainage outlets from linear mountain belts*, Basin Res. 8:29–44, [doi](https://doi.org/10.1111/j.1365-2117.1996.tb00113.x); Castelltort S., Simpson G. 2006, Basin Res. 18:267–276; Talling P. et al. 1997, Basin Res. 9:275–302 | ABSTRACT | UNVERIFIED in part | spacing ratio 1.91–2.23 in 11 belts at the front; Nepal an exception | G-OUTLET-SPACING on linear fronts only |
| C25 | Mandelbrot B. 1975, *Stochastic models for the Earth's relief, the shape and the fractal dimension of the coastlines, and the number-area rule for islands*, PNAS 72:3825–3828, [doi](https://doi.org/10.1073/pnas.72.10.3825); Imre A., Novotný J. 2016, EPJ H 41:69–91 | ABSTRACT; Imre & Novotný UNVERIFIED | CHECKED (K ≈ 0.65 through C28) | a Brownian relief gives D = 1.5, "excessive"; K ≈ 0.65 | G-ISLAND reading |
| C26 | Sayre R. et al. 2019 (published 2018), *A new 30 meter resolution global shoreline vector and associated global islands database*, J. Oper. Oceanogr. 12:S47–S56, [doi](https://doi.org/10.1080/1755876X.2018.1529714) | ABSTRACT | CHECKED | 21 818 islands > 1 km², 318 868 < 1 km², 5 mainlands | island counts, a reading |
| C27 | Thorimbert Y., Chopard B. 2016/2018, *Polynomial methods for fast procedural terrain generation*, [arXiv 1610.03525](https://arxiv.org/abs/1610.03525) | OPENED | CHECKED | box counting 2–64 px on 512–1 024 px maps; D depends on persistence and octaves; compared with Richardson; the Koch warning | the only graphics paper that measured a coast's D |
| C28 | Oline, Hoskins, Seekell, Silber, Cael 2025, *Self-Affine Scaling of Earth's Islands*, [arXiv 2512.16659](https://arxiv.org/abs/2512.16659) (submitted to GRL, v2 May 2026) | OPENED | numbers CHECKED; the comparison UNVERIFIED | 131 063 islands on ASTER 30 m; pixel-edge perimeter–area slope 0.52 ± 0.01; H ≈ 0.95 by their convention (≈ 0.92 by C4's single loop); area law H 0.6–0.8; volume law H 0.57; *"subject to intense erosion"* | a fourth estimator; state the estimator |

### A.2 The lake-shore laws

| id | citation and link | read | check | the numbers taken | for us |
|---|---|---|---|---|---|
| L1 | Seekell D., Cael B., Byström P. 2022, *Problems with the shoreline development index*, GRL 49 e2022GL098499, [doi](https://doi.org/10.1029/2022GL098499), [PDF](https://nora.nerc.ac.uk/id/eprint/533173/) | OPENED | CHECKED | §2.1 items 1–3: `D_L = L/(2√(πA))`; `D_BC`; Scandinavian d/2 0.63, ln C 2.07; global d/2 0.58, ln C 1.75; per-lake d median 1.10 (1.02–1.37); τ = −0.02; Vänern 3.72 / 7.38; 21 % error; +10 % per doubling | G-LAKE-FRACTAL, -DBC, -RICHARDSON, -CLIMB-LENGTH |
| L2 | Seekell D., Cael B., Lindmark E., Byström P. 2021, *The fractal scaling relationship for river inlets to lakes*, GRL 48 e2021GL093366, [doi](https://doi.org/10.1029/2021GL093366), [PDF](https://nora.nerc.ac.uk/id/eprint/530379/) | OPENED | CORRECTED (squares "sometimes found") | `P = C·A^(D/2)`; C by origin (§2.1 item 5); D/2 = 0.65 (0.59–0.70); inlets `(2/π)·D_d·C·A^(D/2)`, +67 % per doubling, error 0.17 against 2.35 | G-LAKE-SHAPE-C, G-LAKE-INLETS |
| L3 | Cael B., Seekell D. 2016, Sci. Rep. 6:29633, [PMC4937396](https://pmc.ncbi.nlm.nih.gov/articles/PMC4937396/), and the corrigendum Sci. Rep. 7:42155, [doi](https://www.nature.com/articles/srep42155) | OPENED | CHECKED | corrected: ≥ 0.46 km², τ = 2.14, d = 1.4; Swedish d = 1.34 above 4.7 km², 1.00 below; the global d sensitive to the weighting | the population band's top |
| L4 | Messager M., Lehner B., Grill G., Nedeva I., Schmitt O. 2016, HydroLAKES, Nat. Commun. 7:13603, [PMC5171767](https://pmc.ncbi.nlm.nih.gov/articles/PMC5171767/) | OPENED | CHECKED | `D_L` by class 1.6, 2.2, 3.7, 6.1, 7.8, 5.8; most lakes 1:100 000–1:250 000, 1:1 M for N Russia and Greenland; sources 1:24 000–1:1 M and 30–250 m; +15 % per resolution doubling; 3 × 3 slope kernel; buffers 1 px – 1 000 m | the scale bias; the slope through the shore |
| L5 | Winslow L. et al. 2014, Freshw. Biol. 59:213 | SECONDARY (L4) | — | +15 % shore per resolution doubling | d ≈ 1.20 (ARITH) |
| L6 | Kent C., Wong J. 1982, CJFAS 39:847 | SECONDARY (L1) | — | d = 1.28 typical | inside the population band |
| L7 | Sharma P., Byrne S. 2011, GRL 38 L24203, [doi](https://doi.org/10.1029/2011GL049577) | UNVERIFIED (HTTP 403) | — | through L1: the 111 lakes; the origins overlap (search text); per-origin values UNVERIFIED | — |
| L8 | Sharma P., Byrne S. 2010, Icarus 209:723 | SECONDARY ([Jameson 2010 summary](https://pages.astro.umd.edu/~dphamil/teaching/ASTR630fall10/abstracts/Jameson.pdf)) | — | Titan D = 1.32 ± 0.1 (box), 1.27 ± 0.1 (ruler), 190 of 290 lakes; a knee in 21; D differs east–west from north–south | the knee and the directional D |
| L9 | Cael B., Seekell D. 2022, L&O Letters, [doi](https://doi.org/10.1002/lol2.10269), [preprint](https://eartharxiv.org/repository/object/3150/download/6361/) | OPENED (preprint) | CHECKED (H = 0.4 set from the literature) | H = 0.4 set, 0.38 free fit; literature 0.4–0.7; 8 164 LAGOS lakes; KS 0.033 at 0.4 and 0.027 at 0.38; ℓ = 0.17; *"lakes are formed by flooding preexisting landscape depressions"* | the contour law for lakes |
| L10 | de Castro C. et al. 2018, Sci. Rep. 8, [arXiv 1712.03115](https://arxiv.org/abs/1712.03115) | OPENED | — | perimeter dimension 7/4 (uncorrelated), 3/2 (H = 0); `S(q) ∝ q^(−2(H+1))` | the spectrum–H link for CHECK-D-FBM |
| L11 | Goodchild M. 1988, *Lakes on fractal surfaces: a null hypothesis for lake-rich landscapes*, Math. Geol. 20:615 | UNVERIFIED | — | lakes as the flooded pits of a fractional-Brownian surface | the null model |
| L12 | Cael B., Lambert B., Bisson K. 2015, PRE 92:052128, [NORA](http://nora.nerc.ac.uk/id/eprint/527322/) | ABSTRACT | — | tidal-flat ponds, a power law over three orders; percolation's unscreened perimeters | — |
| L13 | *Validity, applicability, and universality of fractal scaling laws for lakes in China*, Aquatic Sciences 2023, [doi](https://doi.org/10.1007/s00027-023-00980-w) (authors not read) | ABSTRACT | — | 23 259 lakes, 11 zones; the Korčák–Mandelbrot equality is *"a critical, uncommon condition"* | not a gate |
| L14 | Kobal M. et al. 2015, PLoS ONE 10(3):e0122070 | OPENED | CHECKED | 2 660 dolines on 1 m lidar: 39.8 % circular (≤ 1.21), 41.1 % elliptical, 13.5 % elongated | shape reading |
| L15 | Heathcote A., del Giorgio P., Prairie Y. 2015, CJFAS 72:643 | ABSTRACT | — | the relief outside predicts the depth | G-SHORE-THROUGH-SLOPE |
| L16 | Khazaei B. et al. 2022, GLOBathy, Sci. Data 9:36, [doi](https://doi.org/10.1038/s41597-022-01132-9) | ABSTRACT | — | 1.4 M waterbodies; 1 503 validation sites; the method UNVERIFIED | — |

### A.3 Procedural and graphics shore-making

| id | citation and link | read | check | the numbers taken | for us |
|---|---|---|---|---|---|
| P1 | Génevaux J.-D., Galin É., Guérin É., Peytavie A., Beneš B. 2013, *Terrain generation using procedural models based on hydrology*, ACM TOG 32(4):143, [PDF](https://www.cs.purdue.edu/cgvlab/www/resources/papers/Genevaux-ACM_Trans_Graph-2013-Terrain_Generation_Using_Procedural_Models_Based_on_Hydrology.pdf) | OPENED | CHECKED | §3.2 | way (a); no shore statistic |
| P2 | Cordonnier G., Braun J., Cani M.-P., Beneš B., Galin É., Peytavie A., Guérin É. 2016, *Large scale terrain generation from tectonic uplift and fluvial erosion*, CGF 35(2), [PDF](https://www.cs.purdue.edu/cgvlab/www/resources/papers/Cordonnier-Computer_Graphics_Forum-2016-Large_Scale_Terrain_Generation_from_Tectonic_Uplift_and_Fluvial_.pdf) | OPENED | CHECKED (the preview only) | the lake super-graph; the drawn lake below the pass height | way (c) |
| P3 | Cortial Y., Peytavie A., Galin É., Guérin É. 2019, *Procedural tectonic planets*, CGF 38(2), [PDF](https://perso.liris.cnrs.fr/eric.galin/Articles/2019-planets.pdf) | OPENED | CHECKED | crust 50–500 km; sea 0; SRTM exemplars at 90 m; expert validation | way (b) |
| P4 | Peytavie A., Dupont T., Guérin É., Cortial Y., Beneš B., Gain J., Galin É. 2019, *Procedural riverscapes*, CGF 38(7):35, [PDF](https://perso.liris.cnrs.fr/eric.galin/Articles/2019-riverscapes.pdf) | OPENED | CHECKED | a separate water blend tree | — |
| P5 | Doran J., Parberry I. 2010, *Controlled procedural terrain generation using software agents*, IEEE TCIAIG 2(2), [PDF](https://ianparberry.com/pubs/terrain.pdf) | OPENED | CHECKED | coastline and beach agents; no evaluation | way (a) |
| P6 | Fischer R., Boeckers J., Zachmann G. 2022, CGI (LNCS), [PDF](https://cgvr.cs.uni-bremen.de/papers/cgi22/CGI22.pdf) | OPENED | CHECKED | spline + Voronoi or noise growth; noise-perturbed lake borders | way (a) |
| P7 | Huftier, Schott, Galin, Argudo, Peytavie, Guérin 2026, CGF 45(2) e70389, [HAL hal-05608822](https://hal.science/hal-05608822) | OPENED | CHECKED | Open Eden Growth from a sketched coast; histograms against the Western Alps | way (a) |
| P8 | Derzapf E. et al. 2011, CGF 30(7):2031 | UNVERIFIED | — | rivers first on a planet, with level of detail (second-hand) | — |
| P9 | Teoh S. 2008 | UNVERIFIED | — | coastal erosion, beaches, cliffs, deltas | — |
| P11 | Palermo R., Ashton A., Soderblom J., Birch S., Hayes A., Perron J. 2024, *Signatures of wave erosion in Titan's coasts*, Sci. Adv., [PMC11186481](https://pmc.ncbi.nlm.nih.gov/articles/PMC11186481/) | OPENED | CHECKED | §3.2 | the 8o classifier |
| P12 | Palermo R. et al. 2024, NEWTS1.0, GMD 17:3433, [doi](https://doi.org/10.5194/gmd-17-3433-2024); code [USGS 10.5066/P9Q6GDGP](https://doi.org/10.5066/P9Q6GDGP) | OPENED | CHECKED (the code licence UNVERIFIED) | a 40 m flood; four-connected within 2 %; the shape phrases | an instrument after the owner's word |
| P13 | Tewelde Y. et al. 2013, JGR Planets 118:2198 | UNVERIFIED (captcha) | — | shore sinuosity as a measure of fluvial erosion | — |
| P14 | Voss / Mandelbrot: the level set of a fractional-Brownian ground has D = 2 − H | SECONDARY (C28) | CORRECTED (the level set, not one loop) | — | §1.2 |

### A.4 Level-of-detail continuity

| id | citation and link | read | check | the numbers taken | for us |
|---|---|---|---|---|---|
| G1 | Hoppe H. 1996, *Progressive meshes*, SIGGRAPH 96, 99–108, [PDF](https://hhoppe.com/pm.pdf) | OPENED | CHECKED | `E_disc`; topology changes of discontinuity curves penalised; small features vanish in order | the ordered fold |
| G2 | Hoppe H. 1998, *Smooth view-dependent level-of-detail control and its application to terrain rendering*, IEEE Vis 98, [PDF](https://hhoppe.com/svdlod.pdf) | OPENED | CHECKED (time-based morphs) | ≈ 1 px avoids popping; geomorphs over the frame rate (30–72 frames, 1 s); the largest vertical gap | T7 support |
| G3 | Ulrich T. 2002, *Rendering massive terrains using chunked level of detail control*, SIGGRAPH course notes, [PDF](http://tulrich.com/geekstuff/sig-notes.pdf), [page](http://tulrich.com/geekstuff/chunklod.html) | page OPENED; notes' text garbled | UNVERIFIED in part | 4 m tolerance at ≈ 2 px on 1024 × 768 (CHECKED); the morph formula and the skirt quote UNVERIFIED | skirts for the sheet |
| G4 | Losasso F., Hoppe H. 2004, *Geometry clipmaps*, SIGGRAPH 2004, [PDF](https://hhoppe.com/geomclipmap.pdf) | OPENED | CHECKED (the coastline sentence is about compression) | n = 255, 640 px, 90°, ≈ 3 px triangles; transition w = n/10; `z′ = (1−α)z + αz_c`; U = (−1/16, 9/16, 9/16, −1/16), C¹; an optimal 11 × 11 D; an impulse downsample as future work; rms 1.8 m; rms and 99.9th percentile < 1 px | the pyramid kind; the pixel line |
| G5 | Asirvatham A., Hoppe H. 2005, *Terrain rendering using GPU-based geometry clipmaps*, GPU Gems 2 ch. 2 | OPENED (through a fetch summary) | — | `z_f`, `z_c` packed; the same α; normals blended | — |
| G6 | Strugar F. 2009/2010, *Continuous distance-dependent level of detail for rendering heightmaps (CDLOD)*, JGGGT 14(4):57–74, [GitHub](https://github.com/fstrugar/CDLOD) | OPENED | CHECKED | the range doubles per level; the morph over *"the last 15%-30%"*; one *"consistent filter"*; stitching strips over-draw transparent terrain water | finding 14 |
| G7 | Lindstrom P., Pascucci V. 2002, *Terrain simplification simplified*, IEEE TVCG 8(3):239–254 | OPENED | CHECKED | saturation `ε_i = max(ε̂_i, max ε_j)`; `ρ = λε/d`; anisotropic `ρ·cos θ` saves ≈ 2.5 %; > 2 px disturbs without a morph; ×2–3 with a morph; 20–25 % of vertices morph at `τ_max = 32 τ_min` | the nested band; the anisotropic trap |
| G8 | Duchaineau M. et al. 1997, *ROAMing terrain*, IEEE Vis 97 | OPENED | CHECKED | `e_T = max(e_T0, e_T1) + |z(v_c) − (z(v0) + z(v1))/2|`; nested wedgies | SHORE-BAND (ARITH, unbounded on flat ground) |
| G9 | Bruneton E., Neyret F. 2008, *Real-time rendering and editing of vector-based terrains*, CGF 27(2):311–320, [page](https://maverick.inria.fr/Publications/2008/BN08/), [PDF](http://maverick.inria.fr/Publications/2008/BN08/article.pdf) | OPENED | CHECKED (the 2δ rule CORRECTED) | §4.1, §4.6; `w + 2√2δ`; K = 2.8 → 1.1 px; 192² appearance, 25² elevation; 7 precomputed levels; 7.7 ms (4 + 3.7); 1.49 footprints per frame; 0.6–25 ms for 99.8 %; 100 × 100 km², a 5 000² DEM, 940 k nodes, 600 k curves, 190 k areas; depth 17 | family A with a curve; the one-field form |
| G10 | Proland 4.0 (INRIA), [docs](https://proland.inrialpes.fr/doc/proland-4.0/graph/html/index.html) (graph, core, terrain, ocean), BSD 3; shaders `upsampleShader.glsl`, `terrainShader.glsl` from the LarsFlaeten/Proland_dev mirror | OPENED | CHECKED (the clamp is the tutorial and demo form; C¹) | §4.1, §4.6, §4.8; blend `clamp((d/l − k − 1)/(k − 1), 0, 1)` over `(k − 1)·l`; integer residuals | the refinement rule; the pyramid |
| G11 | Yu Q., Neyret F., Bruneton E., Holzschuch N. 2009, *Scalable real-time animation of rivers*, CGF 28(2), [page](https://maverick.inria.fr/Publications/2009/YNBH09/) | OPENED | CHECKED | two bank curves + a flow rate; 25 × 25 km², 800 × 600, r = d = 20 px; meshes from the bank curves | banks from a solve are possible |
| G12 | Bruneton E., Neyret F., Holzschuch N. 2010, *Real-time realistic ocean lighting using seamless transitions from geometry to BRDF*, CGF 29(2), [page](https://maverick.inria.fr/Publications/2010/BNH10/) | OPENED | CHECKED | 60 trochoids 2 cm – 30 m; `N_min` = 1.0, `N_max` = 2.5; 8 × 8 px grid cells; 52 / 87 / 130 fps at 4 / 1 000 / 8 000 m; *"does not work for coasts and shores"* | "drop only what you cannot show" |
| G13 | CesiumGS, quantized-mesh-1.0 and its WaterMask extension, [GitHub](https://github.com/CesiumGS/quantized-mesh); CesiumJS `GlobeSurfaceTile.js`, `GlobeFS.glsl`, `QuantizedMeshTerrainData.js` | OPENED | CHECKED | §4.5 | a mask per level; coverage as an area fraction |
| G14 | Green C. 2007, *Improved alpha-tested magnification for vector textures and special effects*, SIGGRAPH 2007 course (Valve) | OPENED | CHECKED | 4096² → 64² distance field; edge at 0.5; two fields for corners | the cross-faded signed field |
| G15 | Castaño I. 2010, *Computing alpha mipmaps*, the-witness.net | OPENED | CHECKED (the nesting claim CORRECTED) | coverage kept by a per-level threshold found by bisection; in NVTT | a failing control only |
| G16 | Schneider M., Klein R. 2007, *Efficient and accurate rendering of vector data on virtual landscapes*, J. WSCG 15:59–64, [PDF](https://dspace5.zcu.cz/bitstream/11025/1431/1/Schneider.pdf) | OPENED | CHECKED | stencil shadow volumes; *"per-pixel exact"*; *"independent of the underlying terrain geometry"* | family B |
| G17 | Stuetzle C., Franklin W. R., Cutler B. et al. 2008, *Evaluating hydrology preservation of simplified terrain representations*, ACM GIS 2008 | OPENED | CHECKED (a student paper) | RMS and maximum error *"fail to capture whether a reconstructed terrain preserves the drainage network"* | measure the drawn set |
| G18 | Fugacci U., Kerber M., Manet H. 2019/2020, *Topology-preserving terrain simplification*, [arXiv 1912.03032](https://arxiv.org/abs/1912.03032) | ABSTRACT | CHECKED | flips and vertex removals that keep persistent homology | a pond is a minimum–saddle pair |
| G19 | Crest ocean, *Shorelines and Shallows* documentation | OPENED | — | a depth cache from above; waves weaken where the depth is under half their wavelength | 8o |
| G20 | Carr H., Snoeyink J., van de Panne M. 2004, IEEE Vis | UNVERIFIED | — | contour-tree simplification by area, volume, persistence | — |
| G21 | Tevs A., Ihrke I., Seidel H.-P. 2008, *Maximum mipmaps*, I3D | UNVERIFIED | — | — | — |
| G22 | Lindstrom P. et al. 1996; Pajarola R. 1998 | SECONDARY (G7) | — | — | — |
| G23 | Koppe C. 1902 | UNVERIFIED | — | a contour's accuracy against the slope | — |
| G24 | Dyn N., Levin D., Gregory J. 1987 (the 4-point scheme) | UNVERIFIED (C¹ confirmed through G4) | — | — | — |
| G25 | Frasson A., Engel T., Pozzer C. 2018, *Efficient screen-space rendering of vector features on virtual terrains*, I3D 2018, [ACM](https://dl.acm.org/doi/10.1145/3190834.3190851) | UNVERIFIED (HTTP 403) | — | screen-space lines and polygons over any level | family B |

### A.5 Planet engines and shipped products

| id | source and link | read | check | the numbers taken | for us |
|---|---|---|---|---|---|
| E1 | Proland (= G10) | OPENED | CHECKED | *"clipping the graph of the parent quad … half the threshold"* | §5.1 |
| E2 | Outerra blog: [*Ocean Rendering*](https://outerra.blogspot.com/2011/02/ocean-rendering.html) (2011-02-18), *Procedural terrain algorithm visualization* (2009-02), *Evaluation of 30m elevation data* (2015-05), *Preliminary comparison of NasaDEM* (2018-01) | OPENED | CHECKED (the to-do note CORRECTED; the 2018 mismatch is about data dates, not the level of detail) | a distance map only for mixed tiles; the terrain patches carry the water; 150 m base data (2009), wavelet noise scaled by slope and curvature; bathymetry 500 m – 1 km, *"cannot refine them using fractal"* | family B; no cure |
| E3 | Fuentes L. (Asobo) 2022, *Designing the Terrain System of Flight Simulator: Representing the Earth*, GDC 2022, [PDF](https://www.asobostudio.com/files/inline-images/Designing_Terrain_System_Fuentes_Lionel.pdf) | OPENED | CHECKED | §5.1; a 4 × 4 tile mask plus a discard; level 15 read back for collision; 510.1 M km²; more than 2 PB of images | family A by a distance field; the tuned strength refused |
| E4 | Szofran A. 2006, *Global Terrain Technology for Flight Simulation*, GDC 2006, [fsdeveloper](https://www.fsdeveloper.com/forum/resources/global-terrain-technology-for-flight-simulation-paper.299/download) | OPENED | CHECKED | §5.1; textures built at run time below about 300 m/px | the defect class |
| E5 | FSDeveloper forum, [*Can change in water level be fixed?*](https://www.fsdeveloper.com/forum/threads/can-change-in-water-level-be-fixed.454181/) (2021-11-26) | OPENED | CHECKED | a level step along a tile line; a hand-made ramp | one level per water |
| E6 | flightsim.to, [Terrain Enhancements](https://flightsim.to/scenery/terrain-enhancements/) | OPENED (community) | CHECKED (the quote leaves out one item) | §5.1 | the defect still ships |
| E7 | Cesium (= G13) | OPENED | CHECKED | §4.5 | — |
| E8 | Epic Games, UE5 docs: *Water System*; [*Water Meshing System and Surface Rendering*](https://dev.epicgames.com/documentation/en-us/unreal-engine/water-meshing-system-and-surface-rendering-in-unreal-engine) | OPENED | CHECKED (exact wording in §5.1) | tile 2 400 units; extent 64 tiles | the sunk far mesh refused (W18.5) |
| E9 | SpaceEngine blog: [2017-11-20](https://spaceengine.org/news/blog171120/), [2018-12-10](https://spaceengine.org/news/blog181210/), 2019-03-28 | OPENED | CHECKED | 256² texture and 33² vertices per node; level 12 for an Earth-sized planet (9.5 m/px) | nothing |
| E10 | Hello Games, [*Worlds Part I*](https://www.nomanssky.com/worlds-part-i-update/) (2024); McKendrick, GDC 2017 (video) | page OPENED; talk UNVERIFIED | — | mesh-based water; dual marching cubes | nothing |
| E11 | Star Citizen: the fan wiki [*Star Engine*](https://starcitizen.tools/Star_Engine) and *Planet Tech v5*; CitizenCon; *The H2O Show* (2024-04-18) | wiki SECONDARY; talks UNVERIFIED | — | Alpha 4.8.0 *"New water … rivers, lakes, oceans"* | nothing |
| E12 | Elite Dangerous (Frontier) | UNVERIFIED this round (the 2026-09-25 survey read 80.lv) | — | no liquid on landable bodies | avoids the problem |
| E13 | Veloren, `common/net/src/msg/world_msg.rs` (master), GPL-3.0, [GitLab](https://gitlab.com/veloren/veloren) | OPENED (comments only; no code taken) | CHECKED | 13 altitude bits per 32 m chunk; a plan for 7 bits of water depth; the river LoD TODO | a map sent once |
| E14 | Cepero M. 2013, [*Water bodies*](http://procworld.blogspot.com/2013/10/water-bodies.html), Procedural World | OPENED | CHECKED | ≈ 50 km tiles; ≈ 200 source points per tile; a path search to the sea level | nothing for the detail levels |
