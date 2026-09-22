# Lakes and the landscape models — what the world does, what the papers do, what our solve does

Written 2026-09-22, after the owner flew the home planet at 100–150 km and said the water and the land
are "far from believable", the lakes are everywhere, many are scratches on one diagonal, and many are
square patches at the coarse rungs.

This report answers four questions. How are lakes distributed on Earth? Why do the published landscape
models make so few of them? Why does ours make so many? What can we take from other people's work?

I read the code. **I did not run anything.** Every number I call MEASURED comes from the ruling file
`docs/design/owner_decisions_2026-09-21_water.md` (W7, W8, W10) or from the `lake_census` run those
rulings record — never from an argument of mine. Where I could not open a source I write UNVERIFIED.

---

## 0. THE SHORT ANSWER

**Our solve has no lakes. It has DEPRESSIONS, and it draws every one of them as a lake.**

The priority flood raises every closed hollow to its spill so that water can be ROUTED across it. Every
published model does the same — **and every published model then throws that surface away.** Barnes
calls it *"an important **preconditioning** step"*; Landlab writes it *"to a scratch surface, **never to
`topographic__elevation`**"*; Cordonnier calls filling and carving *"**metaphors** … our algorithm only
changes the flow graph connectivity **without altering elevation values**."*

We keep it. `facies` reads `z_flood > z` and calls it water. So every closed hollow on the planet
becomes a lake — at any depth, with no water budget, no origin and no minimum size. The census counted
**51 000 patches, median one node**. Earth has **about 2 000 lakes** that big, on the whole planet.

★ **And Earth really does have closed basins — over about a fifth of its land drains internally.** Our
51 000 pits are not wrong in themselves. **What is wrong is that every one of them is full of water.**
On Earth an endorheic basin is usually DRY: a playa, a salt pan, a desert sink. The solve never asks
whether any water reached the hollow it found.

Three things make the count worse and the shapes wrong.

1. **The ice and the talus run AFTER the last sweep and the last deposit.** Nothing can ever drain or
   fill what they cut, so the final flood finds every one of their hollows and calls it a lake.
2. **The ice line is the MEAN-ANNUAL freezing line.** Both published laws put it at a SUMMER
   temperature. On Earth the mean-annual 0 °C isotherm reaches sea level near 60°. So our climate
   glaciates every continent — MEASURED: 2.4 million nodes, more than all the land.
3. **The flow routing is D8 on a square grid, and every tie goes to the smaller node index** — one fixed
   diagonal, planet-wide. A filled hollow is a FLAT, so every tie inside a lake runs that way.
   Cordonnier et al. 2019 measured this exact pair of steps and reported *"more pronounced straight
   lines after erosion, due to the four- or eight-connectivity."*

**The square patches at the coarse rungs are a different defect.** Ruling W10 gave the SEA a one-bit side
mask read at every rung. A LAKE still folds as a 2 × 2 majority mean, so its shore quantises to the
coarse node and draws as a square. That is the far view, not the model — and W10 records the gap itself.

**Nothing here needs a new dependency.** Fill-Spill-Merge is MIT and ran 933 million cells in 47 seconds;
our whole planet is 8.87 million nodes.

## 1. THE REAL WORLD'S LAKES — THE GATE

### 1.1 The counts and the size classes

**HydroLAKES** — Messager, Lehner, Grill, Nedeva & Schmitt 2016, *Estimating the volume and age of water
stored in global lakes using a geo-statistical approach*, Nature Communications 7:13603,
[10.1038/ncomms13603](https://www.nature.com/articles/ncomms13603). **1 427 688 lakes ≥ 0.1 km²**,
**2.67 × 10⁶ km² = 1.8 % of global land**, **181 900 km³**, shoreline 7.2 × 10⁶ km (four times the ocean
coastline), global mean depth 3.8 m.

★ **Table 1 of that paper is the gate.** `D_L` is the shoreline development: the shoreline over the
circumference of a circle of equal area, so 1.0 is a perfect disc.

| Class (km²) | Lakes | Area (10³ km²) | Area share | `D_L` | Mean depth |
|---|---|---|---|---|---|
| 0.1–1 | 1 241 200 | 348.4 | 13.0 % | 1.6 | 3.5 m |
| 1–10 | 165 100 | 411.0 | 15.4 % | 2.2 | 5.4 m |
| **10–100** | **13 400** | 331.6 | 12.4 % | 3.7 | 10.4 m |
| **100–1 000** | **1 220** | 313.5 | 11.7 % | 6.1 | 19.6 m |
| 1 000–10 000 | 115 | 313.3 | 11.7 % | 7.8 | 32.3 m |
| > 10 000 | 18 | 959.1 | 35.8 % | 5.8 | 139.6 m |

Two readings a generator must not miss. **The count falls about ten-fold per decade of area.** And
**the area is FLAT across five decades at 12–15 % each** — small lakes do not dominate lake area; only
the top class breaks the pattern.

Beside it, **Verpoorter, Kutser, Seekell & Tranvik 2014**, GRL 41:6396–6402,
[10.1002/2014GL060641](https://doi.org/10.1002/2014GL060641): **117 million lakes ≥ 0.002 km²**,
**5 × 10⁶ km² = 3.7 % of the NON-GLACIATED land** — a different threshold and a different denominator
from HydroLAKES, and the two must never be compared directly. 90 million of those are in the smallest
class and contribute 0.27 % of the surface.

### 1.2 The size-frequency law

- **The fitted cumulative law** — Lehner & Döll 2004, J. Hydrology 296:1–22
  ([PDF](https://www.geo.uni-frankfurt.de/45217760/PDF_file.pdf)):
  ★ **`N = 155 791 × A^(−0.9926)`**, `N` the number of lakes larger than `A` km², valid over
  **1 ≤ A ≤ 100 000 km²**. It extrapolates to 1.5 M lakes over 0.1 km²; HydroLAKES later measured
  1.42 M — within 6 %. **A law that predicted, then was measured.**
- **The tail exponent** — Cael & Seekell 2016, Sci. Rep. 6:29633,
  [10.1038/srep29633](https://www.nature.com/articles/srep29633), ★ **as corrected in the 2017
  corrigendum**, [10.1038/srep42155](https://www.nature.com/articles/srep42155), which must be used
  because a subset of the census was fitted by mistake: power law above **0.46 km²**, density exponent
  **τ = 2.14** (percolation theory says 187/91 = 2.054), shoreline fractal dimension **d = 1.4** above
  the break and **d = 1.00 below it** — ★ small lakes really are near-circular.
- ★ **The law BREAKS at the small end, and the break is real:** *"there are an order of magnitude fewer
  lakes 0.01–1 km² … than would be expected if lakes conformed to a power-law size distribution"*, which
  *"creates a pattern whereby **medium sized lakes dominate the total lake surface area**."* Earth's
  topography stops being scale-invariant below about 0.9 km.
- **There is no upper limit.** The Caspian sits at the 78th–87th percentile of the simulated
  largest-lake distribution; it would have to be **4.3× larger** to be an outlier.

### 1.3 ★★ THE GATE THIS MODEL MUST PASS

Our macro node is 8 192 m — **67 km²**. The solve cannot represent anything smaller, so the only honest
comparison is at that size and up.

| Reading | Earth | Ours today | Off by |
|---|---|---|---|
| ★ Lakes bigger than one node (67 km²) | **≈ 2 000** (DERIVED: 14 753 over 10 km², 1 353 over 100 km², interpolated on the ten-fold-per-decade law) | **51 000 patches**, median one node | **≈ 25×** |
| ★ Lake area at that size, share of land | **≈ 1.1–1.3 %** (DERIVED: lakes ≥ 10 km² cover 1.92 × 10⁶ km² of 1.49 × 10⁸ km²) | **9.7 %** | **≈ 8×** |
| Lake cover, glaciated shield | **7–10 %** (Canada 8.6 %, Finland 9.4 %, Sweden 7.7 %) | 9.7 % everywhere | right value, wrong place |
| Lake cover, non-glaciated interior | **0.2–0.8 %** (Africa 0.8 %, South America 0.6 %, Australia+Oceania 0.16 %) | 9.7 % everywhere | **12–60×** |
| ★ Glaciated-to-unglaciated lake-COUNT ratio | **≈ 80×** at ≥ 0.1 km², **≈ 20×** at ≥ 10 km² | **1×** | the whole pattern |
| Lakes ≥ 10 km² per million km² | North America **391**, Europe **195**, Africa **19** | one global density | — |
| Lake area north of 50° N | **45 %** (52 % without the Caspian) on **17 %** of the land | UNMEASURED | — |
| Size-frequency tail above 0.46 km² | **τ = 2.14** | UNMEASURED | — |
| Shoreline development `D_L` | **1.6 at 0.1–1 km² rising to 7.8 at 10³–10⁴ km²** | UNMEASURED | — |
| ★ Land that drains internally | **18–25 %** (Australia 21 %, North America 5 %) | 100 % of pits are wet | — |

★ **The last row reframes the whole problem.** Earth really does have closed basins over about a fifth
of its land. Our 51 000 pits are not, by themselves, wrong. **What is wrong is that every one of them is
full of water.** On Earth an endorheic basin is usually DRY — a playa, a salt pan, a desert sink. The
defect is not that the solve finds depressions. It is that it never asks whether any water reached them.

Sources for the rows above: HydroLAKES Table 1 and its per-country limnicity; Lehner & Döll 2004 Table 4
(Meybeck's census beside GLWD); [BAWLD, Olefeldt et al. 2021, ESSD 13:5127](https://essd.copernicus.org/articles/13/5127/2021/)
for the northern band; [Wang et al. 2018, Nature Geoscience 11:926](https://doi.org/10.1038/s41561-018-0265-7)
for the endorheic area (33.7 × 10⁶ km²). Finland's official figure is **34 330 km² of 338 145 = 10.15 %**.

### 1.4 Where the lakes are, and what made them

The pattern is one sentence: **the lakes are where the ice was.**

- Messager et al. 2016: *"areas with high lake densities are nearly all encompassed within the extent of
  the **last glacial maximum** in Northern Canada and Scandinavia as well as in some areas of Alaska and
  Russia."*
- Lehner & Döll 2004: *"the highest concentration of lakes [is] clearly marked throughout all size
  classes in the **de-glaciated areas between 50 and 70° North**"*, and lakes **under 10 km² peak there
  even more strongly**.
- ★ **The large classes have a different origin.** Lakes ≥ 10 km² show secondary peaks at 30–35° N
  (Tibet, the lower Yangtze floodplain) and near the equator (the Amazon and Congo floodplains, the
  African Rift). **Glacial origin dominates the SMALL lakes; tectonic and fluvial origin dominate the
  LARGE ones.**
- **Volume splits the other way round.** Europe's 781 × 10³ km² of lake holds 103.8 × 10³ km³ (the
  Caspian and Baikal); North America's 1 229 × 10³ km² holds only 36.6 × 10³ km³, because a shield lake
  is shallow and a rift lake is deep.

★ **THE ORIGIN SPLIT IS A CONFIRMED GAP, not a gap in my reading.** Hutchinson 1957's *A Treatise on
Limnology* vol. 1 gives 11 major origin types in 76 subtypes, and Meybeck 1995
([10.1007/978-3-642-85132-2_1](https://doi.org/10.1007/978-3-642-85132-2_1)) is the likeliest holder of
a modern apportionment — **but no published percentage split of lake NUMBER or lake AREA by origin could
be opened in any source.** Report this as an open question; do not invent a split. The usable proxies
are the two bullets above, plus one hard bound: **impact origin is negligible by count** — at most about
45 crater lakes against 1.42 million, or **3 × 10⁻⁵ of lake number** (§6.4).

### 1.5 ★ LAKE SHAPE — three laws a noise-driven lake placer cannot pass

**1. Raggedness is a function of SIZE, not of region.** `D_L` rises **1.6, 2.2, 3.7, 6.1, 7.8, 5.8**
across the six classes; by continent it barely moves at all (North America 1.8, Africa 1.7, Europe 1.5).
The matching perimeter-area law is `l ~ a^(d/2)` with **`d = 1.4` above 0.46 km² and `d = 1.00` below**.

**2. Elongation is BIMODAL by origin.** The Finger Lakes, all cut in Laurentide troughs, all running
north–south with the ice (DERIVED from published dimensions):

| Lake | Length | Width | **L / W** |
|---|---|---|---|
| Hemlock | 11.0 km | 0.80 km | **13.8** |
| Seneca | 61.0 km | 5.00 km | **12.2** |
| Cayuga | 61.0 km | 5.60 km | **10.9** |
| Canandaigua | 24.9 km | 2.40 km | 10.4 |
| Canadice | 4.8 km | 0.48 km | 10.0 |
| Honeoye | 7.2 km | 1.30 km | 5.5 |
| **median of eleven** | | | ★ **10.2** (range 5.5–13.8) |

★ **A glacial trough lake is about 10 : 1, and the ratio hardly moves across a thirteen-fold range of
length.** The depositional streamlined forms carry a different ratio: drumlins run **1.7 to 4.1**,
250–1 000 m long, in fields of thousands. **Two populations, two ratios, one azimuth.**

**3. ★ ORIENTATION IS COHERENT, AND IT NEVER POINTS AT A LATTICE.** Within one glaciated cell the trough
lakes' long axes and the drumlins' long axes must be **PARALLEL TO EACH OTHER**, because both are the
ice-flow direction. In permafrost lowlands a different law applies: the oriented thaw lakes of arctic
Alaska run **N10–15° W, PERPENDICULAR to the prevailing wind**, cut by wind-driven littoral circulation
(Carson & Hussey 1962, J. Geology 70(4):417–439, [10.1086/626834](https://doi.org/10.1086/626834); the
mechanism confirmed spatially by Zhan et al. 2014, Remote Sensing 6(10):9170). ★ **A generator with one
orientation law cannot produce both — and a generator whose orientation comes from the grid produces
neither.**

**In the game.** A pilot flies the belt at 150 km. She should cross a thousand kilometres of hills with
rivers and no standing water at all, then reach the cold high-latitude shield and find it pitted with
water, twenty times more thickly than anywhere she has been. Today she finds the same water everywhere.

## 2. WHY THE PUBLISHED MODELS MAKE FEW LAKES

### 2.1 ★ THE DOCTRINE, STATED OUTRIGHT — and the one line that names our defect

Barnes, *Accelerating a fluvial incision and landscape evolution model with parallelism*,
[arXiv:1803.02977](https://arxiv.org/pdf/1803.02977) §6.2, verbatim:

> *"Depressions may arise spuriously from random initial conditions, or may also represent endorheic
> basins. **Regardless, they are usually a transient feature.**"*

The same section lists the only three sanctioned responses: **ignore** (the erosion will fill them or cut
an outlet), **fill to the lowest outlet**, **breach**. ★ **"Become a lake" is not on the list.**

★ **AND THE FILL IS NEVER WRITTEN TO THE GROUND.** This is the single sentence that separates every
published model from ours.

- Barnes 2016 ([arXiv:1606.06204](https://arxiv.org/abs/1606.06204)): *"Depression filling is an
  important **preconditioning** step."*
- Landlab's `LakeMapperBarnes`: the fill *"is written to a scratch surface, **never to
  `topographic__elevation`**"*
  ([docs](https://landlab.readthedocs.io/en/latest/generated/api/landlab.components.sink_fill.sink_fill_barnes.html)).
- Cordonnier et al. 2019, more explicit still: *"we use carving and filling as **metaphors** as our
  algorithm only changes the **flow graph connectivity without altering elevation values**."*

**They all compute the fill and throw it away.** We compute it, store it in `z_flood`, and then
`facies` reads it back as WATER. That is the whole defect, in one line.

### 2.2 What each work does with a pit a river cannot drain

| Work | What it does |
|---|---|
| Braun & Willett 2013, Geomorphology 180–181, 170–179, [10.1016/j.geomorph.2012.10.008](https://doi.org/10.1016/j.geomorph.2012.10.008) | The implicit sweep needs a cycle-free receiver tree; a local minimum has none, so it is removed before the sweep. **The paper our `sweep` implements.** |
| Barnes, Lehman & Mulla 2014, Computers & Geosciences 62, 117–127, [10.1016/j.cageo.2013.04.024](https://doi.org/10.1016/j.cageo.2013.04.024) | FILLS every depression to its spill, optimally, in one pass — as a routing device that labels watersheds. **The flood our `route` implements.** |
| Cordonnier et al. 2019, Earth Surf. Dynam. 7, 549–562, [10.5194/esurf-7-549-2019](https://doi.org/10.5194/esurf-7-549-2019) | A graph of adjacent basins, a minimum spanning tree, then THREE strategies: **simple correction**, **carving** ("mimics river erosion" — cut a path from the pit's floor to the spill), **filling**. O(n) with Mareš's planar MST. The paper declines to call one physically right. |
| ★ Barnes, Callaghan & Wickert 2021, **Fill-Spill-Merge**, Earth Surf. Dynam. 9, 105–121, [10.5194/esurf-9-105-2021](https://doi.org/10.5194/esurf-9-105-2021) (with Part 2, [10.5194/esurf-8-431-2020](https://doi.org/10.5194/esurf-8-431-2020)) | **Keeps the lake, if the water is there.** See below. |

### 2.3 ★ Fill-Spill-Merge — the one work that treats a lake as water

Its stated motivation is our exact argument. Depressions *"often **host lakes and wetlands** by retaining
water locally"* and arise from *"glacial erosion and/or deposition, compressional and/or extensional
tectonics, and cratering"*; the old assumption that a pit is a data error *"is no longer justified"*,
and earlier methods *"eliminated depressions entirely through filling or breaching, **producing
unrealistic results**."*

**How it works.** A FINITE water volume runs downhill to a pit. Depressions form a binary-tree
**hierarchy**: a leaf fills first, then SPILLS into its sibling through a geolink, then the two MERGE and
flood their parent metadepression. The lake surface is flat and solved in closed form:
**`z_w = (V_w + Σ zᵢ·aᵢ) / Σ aᵢ`**. ★ **Every depression ends DRY, PARTIALLY FILLED, or FULL-AND-SPILLING
— decided by the water supply, never by the topology.**

**And it is cheap.** O(N log N); the hierarchy builds in O(N) with a radix heap. MEASURED by the
authors: **the global GEBCO grid, 933 million cells, in 46.47 s**; 0.23 s on 67 000 cells; 86 to 2 645
times faster than FlowFill. Our home planet is 8.87 million nodes — **about one hundredth of GEBCO.**

### 2.4 The graphics papers say it themselves

- ★ **Schott et al. 2023**, *Large-scale terrain authoring through interactive erosion simulation*, ACM
  TOG 42(5), [10.1145/3592787](https://doi.org/10.1145/3592787)
  ([open PDF](https://hal.science/hal-04049125/document)), Limitations, verbatim: *"Our implementation
  of the stream power simulation ensures that the water flows outside the terrain. Although this gives
  realistic large-scale results, **it does not allow for the placement of lakes.**"*
- ★ **Génevaux et al. 2013**, ACM TOG 32(4), [10.1145/2461912.2461996](https://doi.org/10.1145/2461912.2461996)
  ([PDF](https://www.cs.purdue.edu/cgvlab/www/resources/papers/Genevaux-ACM_Trans_Graph-2013-Terrain_Generation_Using_Procedural_Models_Based_on_Hydrology.pdf)):
  the network *"can subdivide only upstream; thus our algorithm **cannot represent deltas or oxbow
  lakes**"*, and pits are *"small local minima, but they remain **negligible** in the context of
  large-scale hydrology."*
- ★ **Cortial et al. 2019**, *Procedural tectonic planets*, CGF 38(2),
  [10.1111/cgf.13614](https://doi.org/10.1111/cgf.13614)
  ([PDF](https://perso.liris.cnrs.fr/egalin/Articles/2019-planets.pdf)) — the closest published work to
  our initial land. **The word "lake" does not occur once.** Its only water is *"The sea level was set
  to 0 as the reference elevation."*
- ★ **CORRECTION.** **Cordonnier et al. 2016** (*Large scale terrain generation from tectonic uplift and
  fluvial erosion*, CGF 35(2), [10.1111/cgf.12820](https://doi.org/10.1111/cgf.12820),
  [PDF](https://www.cs.purdue.edu/cgvlab/www/resources/papers/Cordonnier-Computer_Graphics_Forum-2016-Large_Scale_Terrain_Generation_from_Tectonic_Uplift_and_Fluvial_.pdf))
  **DOES keep lakes, and I said earlier that it did not.** It builds a **lake super-graph `G_L`**, parses
  candidate arcs *"in increasing pass height order"* at **O(N + M log M)** with M the number of lakes,
  far under N, and draws a lake by *"comparing the **pass height** with the height of all the points
  flowing into the bottom of the lake."* Skipping that graph costs **4–5× more iterations** to converge
  and leaves *"discontinuities in the hydrology network."* It runs on a **random planar graph**, not a
  lattice. **This is a graphics precedent for exactly what we need, and it is a decade old.**

### 2.5 How our ONE lake rule differs

| | Braun & Willett | Barnes flood | Cordonnier carve | Cordonnier 2016 | Fill-Spill-Merge | **Ours** |
|---|---|---|---|---|---|---|
| Pit filled for routing | yes | yes | no (cut) | yes | no | **yes** |
| ★ The fill is written to the GROUND | no | no | no | no | no | **YES — the defect** |
| The spill is breached | no | no | **yes** | no | no | **no** |
| A water VOLUME limits the lake | no | no | no | no | **yes** | **no** |
| A lake is kept on purpose | no | no | no | **yes** | **yes** | by accident |
| Sediment fills the hollow | no | no | no | no | no | **yes (W7)** |

We added the deposit (W7): the sweep's cut is carried down the tree and laid in the first hollow, up to
its spill. That was right, and the pits fell 137 173 → 7 055 over the passes. But the deposit runs INSIDE
the pass loop, and **the ice and the talus run AFTER it**, so the deposit can never reach what they cut.
What it cannot reach, the final flood turns into a lake.

And we do NOT carve the spill. Cordonnier's carving is the published statement of what a river actually
does to a hollow it drains into: it cuts the sill down. Without it, a hollow drains only where the tree
happens to carry sediment.

## 3. GLACIAL LAKES, ICE, AND THE DIAGONAL SCRATCHES

### 3.1 What the glacial models do

The common thread: **a glacial lake is where the ice discharge converges, along the trunk's own line.**
It is long because the trough is long, and it points the way the ice came.

★ **Egholm, Nielsen, Pedersen & Lesemann 2009**, *Glacial effects limiting mountain height*, Nature 460,
884–887, [10.1038/nature08263](https://doi.org/10.1038/nature08263)
([PDF](https://www.d.umn.edu/~kgran/Geol4550/Egholm%202009.pdf), read in full). This is the model we
should copy, and its whole ice law is five numbers:

| Term | Law | Coefficient |
|---|---|---|
| Accumulation | `Ma = −0.1 · min(0, Ts)` | 0.1 m yr⁻¹ °C⁻¹ |
| Surface ablation | `Ms = −0.15 · max(0, Ts)` | 0.15 m yr⁻¹ °C⁻¹ |
| Basal melt | `Mb = −0.05 · max(Tm, Tb)` | 0.05 m yr⁻¹ °C⁻¹ |
| Surface temperature | `Ts = T0 − 5h` (h in km) | lapse **5 °C km⁻¹** |
| Erosion | `ė = ke · \|us\|`, **l = 1** | `ke = 1 × 10⁻⁴`, dimensionless |

★ **The ELA is not an input. It is the `Ts = 0` isotherm**, `h_ELA = T0 / 5` km. And ★ **the ice extent
is never drawn either** — it falls out of the mass-balance integral: *"the limited catchment area does
not allow for ice influx to exceed ablation below the snowline and, in effect, **the snowline acts like
a climatic base level limiting the down-valley extent of glacial erosion**."* Cold-based ice does not
erode. `ke` is calibrated by requiring mean erosion under 1 mm yr⁻¹. The observational half: **peaks
stand less than about 1 500 m above the local snowline, everywhere on Earth, under every tectonic
style.** Note the mesh: **31 480 Voronoi cells at 400 m — irregular, not a lattice.**

**Where an overdeepening goes — two rules, both published.**

1. ★ **At or just below the long-term ELA.** MacGregor, Anderson, Anderson & Waddington 2000, Geology
   28(11), [10.1130/0091-7613(2000)28<1031:NSOGVL>2.0.CO;2](https://doi.org/10.1130/0091-7613(2000)28%3C1031:NSOGVL%3E2.0.CO;2)
   (abstract): *"both sliding speed (which dictates abrasion rate) and water-pressure fluctuations
   (which strongly modulate quarrying rate) **should peak at the equilibrium-line altitude**"*, giving
   *"rapid flattening of the longitudinal profile."* It is maximal there by construction, because ice
   discharge is the integral of net balance upstream.
2. ★ **Immediately downstream of every confluence**, where the trunk is still too narrow for its new
   load: *"Valley steps and overdeepenings develop at tributary junctions in response to sudden
   increases in ice discharge"*
   ([Geografiska Annaler 2023](https://www.tandfonline.com/doi/full/10.1080/04353676.2023.2217047),
   abstract). Anderson, Molnar & Kessler 2006, JGR 111:F01004,
   [10.1029/2005JF000344](https://doi.org/10.1029/2005JF000344), report the same step. **UNVERIFIED —
   both abstract-only.**

**The U-shape** comes from the cross-valley distribution of SLIDING — a **central minimum** in basal
velocity, so the glacier cuts its walls back faster than its floor (Harbor 1992, GSA Bulletin 104(10),
`E = C · u_b²`, `C = 10⁻⁴ a m⁻¹`; quoted verbatim inside [Yamada & Blatter, arXiv:0901.1177](https://arxiv.org/pdf/0901.1177),
since Harbor itself is an image-only scan). V to U takes order 10⁴ years.

**The erosion exponent, and why the published values disagree.**

| Source | `l` | `K` | Calibration |
|---|---|---|---|
| Humphrey & Raymond 1994 (Variegated) | 1 | 1 × 10⁻⁴ dimensionless | sediment yield through a surge |
| Harbor 1992; MacGregor 2000 (models) | 2 | 1 × 10⁻⁴ a m⁻¹ | chosen |
| **Egholm 2009** (the buzzsaw) | **1** | **1 × 10⁻⁴** | tuned to mean erosion < 1 mm yr⁻¹ |
| Herman et al. 2015 (Franz Josef), [10.1126/science.aab2386](https://doi.org/10.1126/science.aab2386) | **2.02** | 2.7 × 10⁻⁷ | 5 months, 1 m stereo, sliding to ~1 100 m yr⁻¹ |
| Cook et al. 2020, [Nat. Commun. 11:759](https://pmc.ncbi.nlm.nih.gov/articles/PMC7005307/) | **0.65** | 1 × 10⁻⁴ mean | 38 glaciers, R² = 0.54 |

**The disagreement is a scale effect, not a contradiction:** `l ≈ 2` at one glacier, where abrasion
physics dominates one bed (Hallet 1979 — the contact force is viscous drag from basal melting, so it
grows with sliding too, hence the square); `l ≈ 0.65` ACROSS glaciers, where lithology and hydrology
flatten it. **For a planet, take Egholm's `l = 1, K = 10⁻⁴`** — dimensionless, and the pair that
produced correct overdeepenings. Herman's own caveat is that spatially integrated data *"cannot
rigorously constrain K_g and l independently because of the trade-off."*

### 3.2 The equilibrium line altitude — the law we do not use

Ohmura, Kasser & Funk 1992, *Climate at the equilibrium line of glaciers*, Journal of Glaciology 38(130),
397–411, [10.1017/S0022143000002276](https://doi.org/10.1017/S0022143000002276)
([PDF](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/8DCF1C24E32A6781CC365853535CD16F/S0022143000002276a.pdf/climate-at-the-equilibrium-line-of-glaciers.pdf),
read in full).

★ **THE CURVE, verbatim:** *"The best-fit polynomial regression curve for the **70 glaciers** under
consideration is **P = a + bT + cT²**, whereby **a = 645, b = 296 and c = 9**, and **P and T are in mm
w.e. and °C**, respectively. **The standard error of estimate is 200 mm w.e.**"*

- `P` is the annual precipitation at the ELA in mm w.e. yr⁻¹; `T` is the mean **summer (June–August)
  free-atmosphere** temperature at the ELA in °C. Tropical glaciers are explicitly excluded.
- ★ **The three-way equivalence at the ELA:** *"approximately **1 °C, 350 mm w.e. and 7 W m⁻²**."*
- **Sensitivity:** *"a change in precipitation of **300–400 mm w.e. corresponds to only 1 °C**"*, and a
  rule the biome step can use directly — *"**glaciers in arid environments must behave insensitively
  towards climatic changes, and vice versa**."*
- Degree-day factors, for a melt model: Braithwaite 1995, J. Glaciol. 41(137) — **DDF_ice 7.5–8.2 mm
  d⁻¹ °C⁻¹** measured over 512 and 415 summer days in Greenland, and **DDF_snow 36–44 % of that**.

★ **OUR `ela_z` READS THE MEAN-ANNUAL FREEZING LINE.** `crates/terrain/src/climate.rs`: `frost_line_z`
solves for the height at which the ANNUAL MEAN air temperature reaches freezing, and `ela_z` adds a
dryness offset of at most 1 200 m. Both published models put the line at the **SUMMER** isotherm —
Ohmura at about +1 °C JJA, Egholm at `Ts = 0` with `Ts` the ablation-season temperature. The mean-annual
0 °C isotherm reaches sea level near 60° on Earth, where the real ELA stands at 1 100–1 500 m. **That
one word — annual instead of summer — is why W8 measured 2.4 million nodes above the line, more than
all the land.** Earth carries ice on **10 % of its land today and about 32 % at the Last Glacial
Maximum** ([NSIDC](https://nsidc.org/learn/parts-cryosphere/glaciers/glacier-quick-facts); the spread
in the literature is 25–32 %, depending on what counts as land).

**The climate model has no season at all.** `ClimateLaws` holds a mean temperature, a latitude contrast
(Legendre P₂) and a lapse rate, and nothing else. The obliquity is already read
(`climate.rs::insolation_s2`), so a seasonal amplitude can be built on words the charter already states.
**That is the missing input, and it is a small one.**

### 3.3 What our ice does, read in the code

`MacroSolve::ice` (`crates/terrain/src/solve.rs`) does this, per node, independently:

1. A node above its ELA, not under water, is ice.
2. Its thickness is the perfect-plastic law `H = τ / (ρ_ice · g · S)` over the slope to its D8 receiver
   (Nye 1952; Paterson 1994), capped by the height above the line.
3. The cut is `rate · H · S`, capped at `H`, with `rate = 10⁻³ m/yr × 2.6 Myr / 15 m ≈ 173`.

Substitute the plastic thickness into the cut and the slope CANCELS: `cut = rate · τ / (ρ_ice · g)`, a
CONSTANT wherever the plastic law binds. So over a glaciated plateau the ice lowers every node by the
same amount — which erodes nothing differentially — and the hollows come only from where the two caps
(`above` and `H`) switch over, which is noise at the node scale. That produces **one-node pits**, which
is exactly the median lake patch the census found.

It also has no along-flow continuity. Real ice carries its discharge down the trunk, deepens where
tributaries join, and leaves a riegel below. Ours reads one node and its receiver, and nothing else.

### 3.4 ★ THE DIAGONAL SCRATCHES — the literature names our exact defect

**The published description of what the owner saw.** Fairfield & Leymarie 1991, *Drainage networks from
grid digital elevation models*, Water Resour. Res. 27(5),
[10.1029/90WR02658](https://doi.org/10.1029/90WR02658), abstract, verbatim:

> *"Current algorithms … share a fault: **Unless the terrain is rugged, the derived water channels tend
> to flow in parallel lines along preferred directions engendered by the sampling grid orientation.**"*

Note the condition: **the artefact is strongest on SMOOTH terrain.** A filled hollow is the smoothest
terrain there is.

★ **And Cordonnier et al. 2019 measured it on the very two steps we run.** In that paper, depression
FILLING on an eight-connected grid produces a star pattern whose branches *"are due to the grid
eight-connectivity used here"*, and a naive breadth-first receiver assignment *"leads to more pronounced
straight lines after erosion, due to the four- or eight-connectivity."* **Our `assign_flat_receivers` IS
a breadth-first distance field over a filled — that is, flat — surface.** The literature has already
measured that this exact pair of steps draws straight lines on an eight-connected grid.

**What I can tell from our code.**

- `MacroLattice::neighbours` returns **8 neighbours** on a square face grid. `route` takes the steepest
  lower neighbour on the flooded surface. So every flow direction on the planet is one of eight,
  separated by **45°** (Tarboton 1997, Water Resour. Res. 33(2),
  [10.1029/96WR03137](https://doi.org/10.1029/96WR03137), [PDF](https://hydrology.usu.edu/dtarb/96wr03137.pdf)).
- ★ **We are NOT guilty of the cheap version of this bug.** The common implementation defect is charging
  a diagonal step as one cell instead of √2, which makes diagonals artificially cheap and the router
  prefers them. **MEASURED, by a test that could have failed** (`macro_lattice.rs`,
  `chords_between_neighbours_are_about_a_node`): our diagonal chord is √2 of a side to within 0.02, and
  `steeper()` divides the drop by that real chord. The bias we have is the classical one, not an
  arithmetic slip.
- **The tie-break is deterministic AND directional.** `steeper()` breaks a tie by `m < best` — the
  smaller node index. The index is `(face · edge + j) · edge + i`, so a tie prefers the smaller `j`,
  then the smaller `i`: the stencil's `(−1, −1)` corner. **One fixed diagonal, planet-wide.**
  `assign_flat_receivers` breaks its tie the same way.
- A FILLED HOLLOW IS A FLAT, so the whole lake surface is resolved by that BFS field, and every tie
  inside it runs to the same diagonal.

★ **REFINING THE GRID WILL NOT FIX IT.** Hyväluoma 2017, *Reducing the grid orientation dependence of
flow routing on square-grid digital elevation models*, IJGIS 31(11), 2272–2285,
[10.1080/13658816.2017.1358365](https://doi.org/10.1080/13658816.2017.1358365)
([open PDF](https://www.mobt3ath.com/uplode/book/book-120273.pdf), read in full). On a cone with exact
circular symmetry, the flow accumulation *"has a **fourfold rotational symmetry which reflects the
underlying grid structure. Therefore this unphysical pattern cannot be removed by using finer
grid.**"* Confirmed across a 10× range of cell size. **Adding octaves or rungs will not help. It is a
symmetry defect, not a resolution defect.**

Hyväluoma's rotation test is the measurement we should run: compute flow accumulation, rotate the field,
recompute, cross-correlate. A perfect router scores 1.0 at every angle. Measured: *"cross correlations
are close to 1 when M = 1 … Especially when **θ ≈ 45°, cross correlations can be close to 0 for large
flow exponent. Even negative cross correlations were observed.**"* D8 is the `M → ∞` limit of that dial.

★ **AND THE TEST WE WOULD NATURALLY WRITE WOULD PASS WHILE THE BUG IS PRESENT.** Tarboton's Table 2
(read in full):

| Method | cone-out MSE | cone-in MSE | **plane MSE** |
|---|---|---|---|
| **D8** | **2.13** | **118.88** | **0.065** |
| D∞ | 0.20 | 30.58 | 0.065 |

D8 is **10.7× worse than D∞ on the outward cone and 3.9× worse on the inward cone — and exactly equal
on the plane.** Tarboton's own explanation: *"D8 does well for the plane **because the area is the same
whether one counts along the grid or perpendicular to contours. This would not have been the case had
the ridge not been aligned with the grid.**"* **Test on a cone, never on a grid-aligned plane.**

**What I CANNOT tell without running anything.** I cannot say what share of the 51 000 patches lie on
that diagonal, nor whether the ice or the flats dominate. §6.3 states the measurement. **No claim about
the diagonal should be made without it.**

**In the game.** A pilot over the belt sees a line of ponds running north-east, one node wide, straight
as a ruler, for two hundred kilometres. No river made that. The grid did.

### 3.5 Why a real glacial lake is long — and what it proves about water

Five causes point the same way, and none of them is a grid.

1. ★ **The erosion law reads a field that varies ALONG the flow, not across it.** `ė = K · u_s^l` reads
   sliding, sliding tracks discharge, and **discharge is an integral along the flowline**. It changes
   over kilometres down the valley as catchment accumulates, and over hundreds of metres across it from
   wall drag. **The erosion anomaly is kilometres long and hundreds of metres wide by construction.**
2. The trough is a flow-parallel conduit, usually re-cut along a PRE-EXISTING river valley; the lake is
   a flooded segment of it.
3. Confluences are kilometres apart down a trunk, so the basins come in a CHAIN on one axis — the
   paternoster lakes.
4. A soft band along the valley gives the ice weak rock; a hard **riegel** survives as the dam.
5. Every subglacial landform elongates the same way, and §1.5 gives the two ratios: **10 : 1 for an
   erosional trough lake, 2–4 : 1 for a depositional drumlin, one shared azimuth.**

★ **THE FACT THAT DECIDES WHY LAKES ARE GLACIAL AT ALL.** Cayuga's floor stands 16–53 m **BELOW SEA
LEVEL** and Seneca's about 53 m below, with another 300 m of post-glacial fill under the water, so the
rock basin is over twice as deep as the lake. **A river cannot cut below its base level. Only ice can**,
because ice is a pressurised fluid and flows uphill out of a basin while the ice SURFACE slopes the
right way. **An overdeepening is the one landform running water cannot make — which is precisely why it
is the one landform that reliably leaves a lake behind.** That single sentence explains §1.4's whole
pattern: the lakes are where the ice was, because everywhere else the rivers already drained them.

---

## 4. A SIZE-FREQUENCY AND ORIGIN-BUDGET LAW FOR LAKES

Ruling T9 says every physical number comes from a published law with a stated calibration body. Today a
lake has no law at all: it is whatever the routing flood left. Here is a T9-lawful replacement.

★ **THE RULE: a node is WET only where a water budget puts water in it, and a node stays wet over the
age only where an ORIGIN keeps the basin open. Everything else is a DRY closed basin — which is what a
fifth of Earth's land is.**

| Origin | The law that admits it | Calibration body | Can our solve make it? |
|---|---|---|---|
| ★ **Glacial trough / overdeepening** | ice discharge converges at a confluence and peaks at the ELA (MacGregor 2000); thickness by the plastic law (Nye 1952); `ė = 10⁻⁴·\|u_s\|` (Egholm 2009) | Alpine and Scandinavian troughs; the Finger Lakes | **Partly.** The stage exists but reads one node and its receiver. No along-flow discharge, no riegel. It makes noise pits, not troughs. |
| ★ **Rift / tectonic** | a divergent continental boundary makes a graben; `land.rs` already draws the profile `−A/4 · bump(b/(W/4))` | the East African Rift; Baikal | **YES.** The basin exists. It is drowned among 51 000 others, and it is the origin that holds the VOLUME. |
| **Floodplain / oxbow** | a meander cut off; length **11 channel widths** for a chute, **20** for a neck; **the floodplain SLOPE picks the type** (Li et al. 2026, [10.1126/sciadv.aef6658](https://doi.org/10.1126/sciadv.aef6658)) | the Amazon: **0.42 cutoffs per 1 000 river-km per year**; oxbows about **5 %** of the Housatonic floodplain | **NO, and it should stay no in the solve.** A node is 8 192 m; a meander is not resolved, and an oxbow fills in **decades to a few centuries**. This belongs to the FINE rungs as a drawn feature. |
| **Impact crater** | a bowl younger than the surface's retention age (§6.4) | the Earth Impact Database's 190; roughly 30–45 hold a lake | **Partly.** Stamped, but at the wrong age, and the deposit now fills them. |
| **Karst / solution** | dissolution of a soluble bed under enough rain | Florida; the Dinaric karst | **NO.** Needs a soluble stratum on the province word and a dissolution law. |
| **Aeolian / deflation** | wind scours a basin toward the water table in an arid belt; ★ the long axis runs **normal to the prevailing wind** (Carson & Hussey 1962) | the Alaskan oriented thaw lakes; the Qattara depression | **NO.** No aeolian mechanism exists. |
| **Volcanic (caldera, maar)** | a collapsed magma chamber | Crater Lake; the Eifel maars | **NO.** |
| **Landslide-dammed** | a slope failure blocks a valley | Usoi dam, Sarez | **NO**, and it should stay no: they do not last. |

★ **Note what the origin table predicts, and check it against Earth.** Glacial origin dominates the
SMALL lakes; rift and floodplain origin dominate the LARGE ones; and the volume sits in the rift lakes,
not the shield lakes. That is exactly the pattern §1.4 measured. **The two agree, which is the best
evidence that this is the right set of origins.**

**THE BUDGET GATE.** `lake_census` already counts patches. Extend it to count them BY ORIGIN, and add
the readings of §1.3 — each one a number that can fail:

1. Lakes bigger than one node **of order 2 000**, not 51 000.
2. Lake area **1.1–1.3 %** of the land at our resolution.
3. ★ **The glaciated-to-unglaciated ratio, about 20× at this size.** This is the strongest gate in the
   whole report, because a model with one global lake density fails it by twenty and a model that has
   merely reduced its lake count still fails it.
4. **18–25 % of the land drains internally, and nearly all of it is DRY.**
5. The count falls about ten-fold per decade of area; `D_L` rises from 1.6 to 7.8 across the classes,
   and does NOT vary by region.
6. ★ **Shape:** a glacial trough lake is about **10 : 1** (median 10.2, range 5.5–13.8), and within one
   glaciated cell every trough lake's long axis is **parallel to every other's**, because both read the
   ice flow. ★ **This gate is the direct refutation of the diagonal scratches**, because a grid artefact
   aligns with the LATTICE and a real trough aligns with the ICE — and the two can be told apart by one
   histogram.
7. Every lake node names an origin. A lake with no origin is a DEFECT and the gate goes red.

**THE WATER BUDGET.** `sea_level_loaded` spends the WHOLE inventory on the sea (`land.rs`), so every
lake the flood fills holds water the inventory has already spent. `DEFERRED.md` records this as THE LAKE
DOUBLE COUNT, owed at C5. Fill-Spill-Merge is the published shape of the cure, and it closes both
defects at once.

**In the game.** The pilot flies the belt. Every body of water she passes can name what made it: a rift
she follows for six hundred kilometres, a crater with a rim, a trough with a rock bar at its mouth. The
dry basins outnumber them twenty to one, and they are salt flats she can land on.

## 5. RUST CRATES AND REUSABLE CODE

Our constraint is hard. The shared recipe path is integer-only and must give ONE answer on the server's
CPU, the client's CPU and the card. The once-per-body solve may use `f64` under the fence (`Gf`), so a
server-only library is admissible where a shared one is not. The licence must be permissive; GPL is a
blocker for code and NOT for reading a paper.

### 5.1 ★ A LANDSCAPE MODEL ON A WHOLE SPHERE ALREADY EXISTS — THREE OF THEM

This corrects what I believed before the survey. We are not first.

| Work | Language | Licence | Grid | What it does |
|---|---|---|---|---|
| ★ **fastscapelib `healpix_grid`** ([planetary example](https://fastscapelib.readthedocs.io/en/latest/examples/planetary_py.html)) | C++/Python | **GPL-3.0 — a blocker for code** | HEALPix, `n_neighbors_max = 8`, on `healpix_cxx` | A working PLANETARY run: nside 100, ~120 000 nodes, ~65 km cells, Earth's radius; single-direction routing + an MST sink resolver + the stream-power eroder; uplift 9 × 10⁻⁴ m/yr, 40 steps of 10 kyr. **The closest existing thing to our solve.** |
| ★ **Landlab `IcosphereGlobalGrid`** ([docs](https://landlab.csdms.io/generated/api/landlab.grid.icosphere.html)) | Python/NumPy | ★ **MIT** | icosahedron subdivision; the dual cells are hexagons plus twelve pentagons | `FlowAccumulator` with `DepressionFinderAndRouter` runs on the GLOBAL sphere; the tutorial recovers the Amazon and the Paraná from real global topography. The eroders exist but **UNVERIFIED on the icosphere**. Landlab itself warns that the depression router "does not scale well". |
| **goSPL** ([JOSS 5(56):2804](https://joss.theoj.org/papers/10.21105/joss.02804)) | Python + PETSc + MPI | **GPL-3.0** | unstructured icosahedral sphere, variable resolution | Stream power (Howard 1994) on multiple flow direction, an implicit parallel drainage-area solver (Richardson, Hill & Perron 2014), **priority flood (Barnes 2014)**, marine diffusion, hillslope diffusion, compaction, and horizontal advection for plate motion. |
| Salles et al. 2023, *Nature* [10.1038/s41586-023-06777-z](https://www.nature.com/articles/s41586-023-06777-z) | a goSPL run | paywalled | icosahedral, **> 10 M nodes, ~5 km global** | The largest published whole-planet landscape run. ⚠ **UNVERIFIED — figures from search snippets, not the paper.** |

Our macro lattice is 8.87 M nodes at 8 192 m. Salles et al. ran 10 M nodes at 5 km. **We are at the
published state of the art for scale, and behind it for the lake treatment.**

### 5.2 Rust — what exists, and what does not

| Name | Version / state | Licence | What it is | Verdict |
|---|---|---|---|---|
| **WhiteboxTools** ([jblindsay/whitebox-tools](https://github.com/jblindsay/whitebox-tools)) | maintained | **MIT** | **Written in Rust.** 52 hydrology tools: `FillDepressions`, `BreachDepressions`, `BreachDepressionsLeastCost`, `D8Pointer`, `DInfFlowAccumulation`, `StochasticDepressionAnalysis` | **Harvest.** The largest body of Rust hydrology there is. `f64` and raster-square, so it cannot go on the shared path; read it for the breaching algorithms. |
| **`oxscape_erode`** | published | MIT / Apache-2.0 | A permissive Rust reading of FastScape's implicit stream-power solver | **Harvest, and read first.** The one Rust rendering of the very paper our `sweep` implements. |
| **`fastlem`** ([TadaTeruki/fastlem](https://github.com/TadaTeruki/fastlem)) | **archived 2025-06-05** | MPL-2.0 | Delaunay on a 2-D PLANE; the Salève analytical model plus a Cordonnier drainage network | Harvest the structure. Planar, archived, `f64`. |
| **`sphere_terrain`** ([OptimisticPeach](https://github.com/OptimisticPeach/sphere_terrain)) | 2025-01-08 | ★ **Apache-2.0** | Droplet erosion, wetness and rivers on an ICOSPHERE, in Rust | **Harvest.** The only permissive Rust erosion-on-a-sphere. A droplet model, not stream power. |
| **`rstar`** | 0.13.0 | MIT / Apache-2.0 | R-tree; `RTreeNum` accepts `i32` and `i64` **by design** | **The one crate worth ADOPTING.** An integer-safe spatial index, if the origin work needs one. |
| **`cdshealpix`** ([cds-astro](https://github.com/cds-astro/cds-healpix-rust)) | 0.9.1, 2026-03-09 | Apache-2.0 / MIT | HEALPix. The NEST index is a **Morton bit-interleave** on a `u64`; `nested::neighbours`, `siblings`, edge walks | **Server-side only, or harvest ~200 lines of NEST bit maths.** Heavy deps (png, flate2, serde, chrono, rayon). |
| **`realpix`** | 0.2.0, 2026-08-30 | Apache-2.0 / MIT | The lightest Rust HEALPix (~2 180 lines), `nested::Layer::neighbours` with a direction enum | Harvest or watch. 306 downloads, one release. |
| **`h3o`** | 0.11.0, 2026-08-29 | BSD-3-Clause | The Rust port of Uber's H3 | Server-side only. `float_eq` and `ordered-float` trip the fence, and **H3 hexagons do not nest exactly**. |

Also surveyed and **ignored**: `s2` (pulls `bigdecimal`), `hexasphere` (pulls `glam` f32), `a5`,
`icosphere` (no adjacency at all), `geodesy` (a datum toolkit, not a grid), DGGRID (**AGPL-3.0, a
blocker**), and every Rust noise crate — `noise-rs`, `libnoise`, `fastnoise-lite`, `bracket-noise`,
`simdnoise` — because we have our own integer noise and a float noise crate cannot enter the recipe.

**WHAT DOES NOT EXIST, ANYWHERE:** a fixed-point noise or terrain crate in Rust; a Rust binding of
TopoToolbox, RichDEM or fastscapelib; a Rust port of Fill-Spill-Merge; a standalone Rust priority-flood
crate; a Rust ice-flow crate; and any permissively-licensed Rust project that puts flow routing on a
CUBE SPHERE. On the last one we would be first.

### 5.3 Lakes specifically — the one item worth taking

★ **Fill-Spill-Merge and its `DepressionHierarchy`**
([r-barnes/Barnes2020-FillSpillMerge](https://github.com/r-barnes/Barnes2020-FillSpillMerge)) are
**MIT**. C++, but the algorithm is a few hundred lines of hierarchy walk. **This is the single most
valuable item in the whole survey**: licence-clean, small, and exactly the mechanism our solve lacks.
Beside it, **RichDEM** (Barnes' own priority flood and breaching) and **Landlab's `LakeMapperBarnes`**
(MIT) are worth reading.

### 5.4 Is the algorithm portable to our integers?

Yes, part by part, with two cautions and one obstacle.

- The **receiver choice** is a cross-multiply comparison in fixed point — exact, no division. Our
  `steeper()` already does it this way.
- **Flow accumulation** is a topological sort plus an integer add. Exactly reproducible.
- The **priority flood** is a heap keyed on elevation — portable, **but the tie-break MUST be stated**,
  because equal integer heights are common and two hosts that break a tie differently drift. We state
  ours (`(height, index)`), and §3.4 says what that costs us in shape.
- **Stream power** at `m = ½` is the square root the recipe already holds; `n = 1` needs nothing. The
  implicit solver needs a reciprocal, and a Newton iteration for `n ≠ 1`; the recipe has both, but
  **the iteration count must be a fixed number, never a tolerance loop**, or two hosts stop on
  different steps. Our sweep is `n = 1` and takes one division, so this does not bind today.
- **The one real obstacle is mesh geometry** — cell areas and node distances. HEALPix removes it by
  having equal-area cells; our `area_m2` instead pays for it once per node with a midpoint quadrature
  under the float fence, floored to whole square metres. **We have already paid that price.** Changing
  the grid would move every node on every body, which is a new world under SL5. I raise HEALPix as a
  fact about the alternatives, **NOT as a proposal**.

### 5.5 The verdict, as options — the owner decides

- **ADOPT (link):** `rstar` only, and only if the origin work needs a spatial index.
- **HARVEST (read, then write in integers):** Fill-Spill-Merge and `DepressionHierarchy` (MIT) for the
  lakes; Landlab (MIT) for the depression router and the lake mapper; WhiteboxTools (MIT, Rust) for the
  breaching; `oxscape_erode` (MIT/Apache) for a Rust reading of Braun & Willett; `sphere_terrain`
  (Apache-2.0) for sphere structure.
- **READ, NEVER COPY:** fastscapelib's `healpix_grid` — the best design in existence for our problem,
  and GPL-3.0. Also goSPL and RichDEM. **Their CITED PAPERS carry no licence and are the real prize.**

Nothing here should become a dependency. Every valuable item is an ALGORITHM of a few hundred lines that
we must write in integers anyway, because no existing library can meet the no-drift gate.

## 6. RECOMMENDATIONS, RANKED

Each carries its cost, its law, its calibration body and its gate.

### 6.1 ★ FIRST — STOP WRITING THE ROUTING FILL TO THE GROUND

**The smallest true statement of the fix: every published model computes the fill and throws it away.
We keep it and draw it.** Three changes, one idea.

1. **The fill is a scratch surface.** `z_flood` may decide receivers and nothing else. `facies` must stop
   reading `z_flood > z` as water. That alone deletes 51 000 lakes in one line — and leaves none, which
   is why the next two are not optional.
2. **Carve the spill.** Take Cordonnier et al. 2019's CARVING strategy where a river reaches a
   depression: the river cuts the sill down, the hollow drains, the hollow is gone. A hollow no river
   reaches keeps its pit for step 3.
3. ★ **Budget the water.** A node is lake where a FINITE volume actually stands, by Fill-Spill-Merge
   over the depression hierarchy, from the inventory the sea did not take. A depression ends **dry,
   partially filled, or full-and-spilling — decided by the water supply, never by the topology.** A
   hollow the rain cannot fill stays a dry basin, which is what a playa IS.

- **The laws:** Cordonnier et al. 2019, [10.5194/esurf-7-549-2019](https://doi.org/10.5194/esurf-7-549-2019);
  Barnes, Callaghan & Wickert 2021, [10.5194/esurf-9-105-2021](https://doi.org/10.5194/esurf-9-105-2021),
  reference code **MIT**. The graphics precedent is Cordonnier et al. 2016's lake super-graph.
- **Calibration body:** Earth — lake area 3.7 % of the non-glaciated land (Verpoorter et al. 2014).
- ★ **Cost, MEASURED BY THE AUTHORS, not argued:** Fill-Spill-Merge ran the global GEBCO grid, **933
  million cells, in 46.47 s**. Our home planet is 8.87 million nodes — one hundredth of that. Against a
  114 s solve this is **noise**. The carve is O(n) on the basin graph the flood already builds.
- **The gate:** `lake_census` — lake area ≤ 4 % of the land; lakes bigger than one node of order 10³,
  not 10⁴; **and the inventory balances** (sea + lakes = the charter's water), which finally closes THE
  LAKE DOUBLE COUNT that `DEFERRED.md` has carried since C5.
- **In the game:** a pilot follows a river into a hollow. The river runs THROUGH it and out the far side
  through a gorge it cut. There is no lake, because there was never enough water to make one.

### 6.2 ★ SECOND — THE ICE LINE IS A SUMMER LINE, AND THE ICE MUST RUN INSIDE THE PASSES

Three halves, all cheap, and the first is one word.

1. ★ **`ela_z` must read a SUMMER temperature, not the annual mean.** Egholm et al. 2009 take the ELA as
   the `Ts = 0` isotherm of the ablation-season temperature at a **5 °C km⁻¹** lapse; Ohmura et al. 1992
   put it at about **+1 °C JJA, 350 mm w.e., 7 W m⁻²**, with **`P = 645 + 296·T + 9·T²`** (mm w.e. yr⁻¹
   against °C, 70 glaciers, standard error 200 mm w.e.) as the cross-check against the rain we already
   compute. The climate needs a seasonal amplitude, and the charter's obliquity already supplies it.
2. ★ **Never draw the ice extent — let the mass balance decide it.** Egholm's whole ice law is five
   numbers: accumulate at `−0.1·min(0,Ts)`, ablate at `−0.15·max(0,Ts)` and `−0.05` basally, all
   m yr⁻¹ °C⁻¹, and erode at `ė = 10⁻⁴·|u_s|` (`l = 1`, dimensionless, calibrated so mean erosion stays
   under 1 mm yr⁻¹); cold-based ice does not erode. *"The snowline acts like a climatic base level
   limiting the down-valley extent of glacial erosion."* A small catchment stalls at the snowline on its
   own. **No extent is ever stated.**
3. **The order.** Move the ice and the talus INSIDE the pass loop, before the last sweep and the last
   deposit. Today they run after, so nothing can drain or fill what they cut. **That reorder is free.**

- **The laws:** Ohmura, Kasser & Funk 1992, J. Glaciol. 38(130), 397–411,
  [10.1017/S0022143000002276](https://doi.org/10.1017/S0022143000002276); Egholm et al. 2009, Nature 460,
  884–887, [10.1038/nature08263](https://doi.org/10.1038/nature08263).
- **Calibration bodies:** Ohmura's 70 mid- and high-latitude glaciers; Earth's present ice cover.
- **The gates, two of them, both able to fail:**
  1. **the ice covers about 10 % of the land** on an Earth-like charter (today the line stands over 2.4
     million nodes, **more than ALL the land** — red on its own), with about 30 % as the glacial-maximum
     figure ([NSIDC](https://nsidc.org/learn/parts-cryosphere/glaciers/glacier-quick-facts) says 32 %;
     the literature spreads 25–32 %);
  2. ★ **the buzzsaw gate** — no peak stands more than about **1 500 m above its local snowline**. That
     is Egholm's own observational result, it holds on Earth under every tectonic style, and it is a
     free reading over a field we already have.
- **In the game:** the pilot flies the belt at the equator and sees no ice at any height under 5 000 m.
  She flies north and the ice starts where the summer stops melting it, not where the year averages
  freezing.

### 6.3 THIRD — THE SCRATCHES: MEASURE FIRST, AND KNOW WHAT WILL NOT WORK

★ **What will NOT work: a finer grid.** Hyväluoma 2017 measured that on a perfectly circular cone the
fourfold pattern *"cannot be removed by using finer grid"*, and holds across a 10× range of cell size.
**More rungs, more octaves and more nodes will not touch this.**

★ **One cheap suspect is already cleared.** The common implementation slip is charging a diagonal step
as one cell instead of √2. **MEASURED, by a test that could have failed** (`macro_lattice.rs`): our
diagonal chord is √2 of a side to within 0.02. We have the classical bias, not an arithmetic slip.

**Step 1 — the measurements, before any cure.**
  1. **Test on a CONE, never on a grid-aligned plane.** Tarboton's D8 scores 0.065 MSE on the plane and
     118.88 on the inward cone. **A plane test passes while the bug is fully present.**
  2. **Run Hyväluoma's rotation test:** accumulate, rotate, re-accumulate, cross-correlate. A perfect
     router scores 1.0 at every angle; D8 falls toward zero at 45°. It is a number, and it can fail.
  3. **Fit a long axis to every lake patch** in `lake_census` and histogram it against the face grid's
     own directions. A spike at 45° names the tie-break.

**Step 2 — the cures, in rising cost. ★ Start at the flat receiver, because that is where the
literature already found this.** Cordonnier et al. 2019 measured that a naive breadth-first receiver
assignment *"leads to more pronounced straight lines after erosion, due to the four- or
eight-connectivity"*, and cured it by ranking on a **minimal Euclidean distance** instead of the BFS
visit order. **Our `assign_flat_receivers` is exactly that naive BFS.** Their fix is a few lines, stays
integer, stays deterministic, and touches only flats — which is where our lakes are.

| Cure | What it buys | The catch |
|---|---|---|
| **Euclidean cost on the flat, not BFS order** (Cordonnier 2019) | removes the straight lines the paper measured | a few lines; the tie must still be stated |
| **D8-LTD** (Orlandini et al. 2003, [10.1029/2002WR001639](https://doi.org/10.1029/2002WR001639)) | carries the accumulated deviation, so the straight scratch breaks; keeps 8 directions and one outflow | ⚠ **path-dependent** — the answer depends on traversal order, a real hazard against a byte-for-byte two-host gate |
| **MFD with a tuned cardinal weight `W`** (Hyväluoma) | isotropy; optimal `W ≈ 1.3–4.1`, and Quinn's geometric 1.4124 is **not** the optimum | one constant; changes the discharge on every node |
| **D∞** (Tarboton 1997) | 4–11× better MSE on cones | dispersion; changes every node |
| **A hexagonal or unstructured mesh** | ★ the only cure that REMOVES the anisotropy instead of tuning it: six neighbours, all equidistant, no cardinal/diagonal split at all | **a new world under SL5.** Note both flagship codes abandoned the lattice — FastScape on a natural-neighbour mesh, Egholm 2009 on 31 480 Voronoi cells |
| **Rho8** (random tie-break) | — | ★ **REFUSE.** It breaks SL10's byte-for-byte gate, and Tarboton rejects it on principle: *"Upslope and specific catchment areas are deterministic quantities that we should be able to compute in a repeatable way."* |

**Step 3 — and make the lakes long for the RIGHT reason.** Accumulate the ice down the same receiver
tree the water uses, deepen at the confluences and just below the long-term ELA, and leave a riegel at
the mouth. Then the lake runs **10 : 1 to 20 : 1 along its own trough**, as Cayuga and Seneca do, and the
grid has nothing to do with it.

- **The gate:** the long-axis histogram is flat against the grid's directions, and every glacial lake
  aligns with its own trunk within a stated angle.
- **In the game:** the pilot flies a glacial valley. The lake fills the valley floor, ends at a rock bar,
  and points the way the ice came. It does not point north-east because the world's grid does.

### 6.4 FOURTH — THE CRATER RECORD ON A WET BODY

`crater_population` integrates the Neukum production function over `words.age_yr` — **the system's own
age, about 5 Gyr** — capped only by the LUNAR geometric saturation (Gault 1970; Hartmann 1984's 5 %
share). That is a Moon's record, and it stamps **144 967 craters** on the home planet. W7 measured that
137 173 of the pits were those craters.

**Earth carries 190 confirmed impact structures** ([Earth Impact Database](http://www.passc.net/EarthImpactDatabase/);
Osinski et al. 2022, Earth-Sci. Rev. 232:104112,
[PDF](https://impact.uwo.ca/wp-content/uploads/uwo_earth_science_reviews_232_2022_104112.pdf), gives 188
hypervelocity). **43 are wider than 20 km; the median is 8 km.** ★ **We stamp 760 times Earth's count.**

★ **The erasure is measurable, and the measurement names the right law.** Counting Osinski's Table 4 by
age: **45 % of the whole record is younger than 200 Ma — which is 4.4 % of Earth's history**, a ten-fold
over-representation of the recent. 200 Ma is also the age ceiling of ocean floor, and ocean is 70 % of
the target, so **about 70 % of the planet resets on a 200 Myr clock.** Of 185 structures, **only two
still hold their ejecta blanket and rim**; the paper states that *"all complex craters on Earth are
eroded"*, and two thirds of the record sits *"in the ancient cratonic areas … where (relative) tectonic
stability and low rates of erosion favour preservation."*

- **The law:** the production function stays Neukum, Ivanov & Hartmann 2001. The AGE it integrates over
  becomes the **surface's crater retention age**, derived from that surface's own resurfacing — the
  plate's speed and the province's erodibility, both of which the solve already holds. **The ocean floor
  gets 200 Myr; a craton gets far more.** That also makes the record vary across the planet, which is
  what Earth shows.
- **Calibration body:** Earth — 190 structures, 43 over 20 km, median 8 km, 45 % younger than 200 Ma.
- **Cost:** one number in `crater_population`, plus the law that derives it per province.
- **The gate:** the crater count on an Earth-like charter within an order of magnitude of 190, not
  145 000; the age histogram skewed toward the recent; **and the airless moon's count unchanged**, which
  is the reading that proves the law reads resurfacing and not a body kind (HR4).
- **The bonus:** a crater lake becomes a landmark rather than noise. On Earth roughly **30 to 45 of the
  190 hold a lake** (Manicouagan, Siljan, Bosumtwi, El'gygytgyn, Lappajärvi), concentrated in the
  glaciated cratons — **UNVERIFIED as a published figure**, tallied from the named structures.
- **In the game:** a pilot on the belt finds a crater lake once in a continent, and it is worth flying to.
  On the moon overhead she finds craters everywhere, as she should.

### 6.5 FIFTH — THE SQUARE PATCHES ARE THE FAR VIEW, NOT THE MODEL. SAY SO.

The blocky lakes at the coarse rungs are **a separate defect from everything above**, and the cure is
already written for the sea.

Ruling W10 gave the SEA a one-bit side mask per fine node, read at EVERY rung ("a side is a bit, and a
bit does not fold"). A LAKE still folds as `pyramid_water`: the mean level of the wet children of a 2 × 2
block WHEN AT LEAST HALF ARE WET, else dry (`artifact.rs`). So at a coarse rung a lake either fills the
whole coarse node or vanishes — a square edge, at the node's own size. W10 itself records the gap: *"a
lake's shore still keeps today's rule until 8d step 5."* The 8d design says the same: *"the lake has no
SHORE: the coast band is measured against the SEA alone."*

- **The cure:** widen the coast mask from one bit to a side WORD (sea / lake / land) per fine node, and
  let `height::shore` take a lake's own level the way it takes the sea's. Same machinery, one more state.
  Ruling W4 item 3 already ruled that a lake's surface IS the sea's own mechanism.
- **Cost:** the mask goes from 1.1 MB to about 2.2 MB on the home planet (one part in forty of the rows).
  One artifact version bump. No new lane and no new payload kind.
- **The gate:** `shore_step` run on lake shores as well as sea coasts — 0 m step at every rung pair,
  which is what W6 and W10 achieved for the sea.
- **In the game:** the pilot descends on a highland lake from 1 400 km. Its shore stands in one place the
  whole way down, and it has the shape of the valley, not the shape of a pixel.

**One warning.** Fixing 6.5 alone would make the lakes look better and be just as wrong. 6.1 and 6.2 are
the model; 6.5 is the picture. Do the model first, or we will draw 51 000 beautiful lakes.

---

## 7. WHAT IS STILL OWED

### The measurements we must take
★ **The three of §6.3 step 1** — the cone test, Hyvaluoma's rotation test, and the lake long-axis
histogram. **No claim about the diagonal should be made without them.**

### The three CONFIRMED gaps in the literature
These were searched hard and are genuinely not available. They are gaps in the record, not in the search.

1. ★ **No published apportionment of lakes BY ORIGIN exists.** Hutchinson 1957's eleven types in
   seventy-six subtypes is confirmed, but no percentage split of lake number or lake area could be
   opened anywhere. Meybeck 1995 is the likely holder and is paywalled. **Do not let an invented split
   reach a design document.** The usable proxies are the two measured patterns of section 1.4.
2. **Canadian Shield lake azimuth against bedrock jointing** — no rose-diagram study obtained. This is
   the one claim in the original brief with zero numeric support anywhere.
3. **The spread of thaw-lake orientation**, and mega-scale glacial lineation ratios. The drumlin and
   Finger Lakes numbers are sourced; the often-quoted 100 : 1 lineation figures are **deliberately not
   printed**, because nothing could be opened to support them.

### Sources that refused to open
Harbor 1992 (an image-only scan, so its erosion rule is quoted at second hand); MacGregor 2000 and
Anderson 2006 (abstract only — **no closed-form law for an overdeepening's DEPTH exists anywhere**;
depth is a model output, never a rule); Egholm 2011 (iSOSIA); Shelef & Hilley 2013, whose title names
our failure exactly and is worth institutional access; Salles et al. 2023's node count. Wiley,
ScienceDirect, GeoScienceWorld and MDPI blocked every route, and the web-search budget ran out.

### Four corrections recorded on purpose, because each was a belief this report started with

1. **"No landscape evolution model has been run on a whole sphere."** WRONG. fastscapelib has a HEALPix
   planetary example, Landlab a global icosphere, goSPL an icosahedral planet past ten million nodes.
   **We match the state of the art for SCALE.** We are behind on one thing: everyone else treats a
   depression as a routing problem, and we treat it as water.
2. **"Cordonnier et al. 2016 removes depressions."** WRONG. It builds a **lake super-graph** and draws
   lakes by pass height at O(N + M log M). A graphics precedent for what we need is a decade old.
3. **The Cael & Seekell figures that are widely quoted (tail exponent 1.97, break at 8.5 km2) were
   CORRECTED in 2017.** Use the corrigendum: **exponent 2.14, break at 0.46 km2.** A subset of the
   census had been fitted by mistake.
4. **Tarboton 1997 writes that D8's directions are separated by 45 degrees**, not 22.5. The 22.5
   per-cell maximum is our own derivation, and it is stated as such.

### One thing we are NOT guilty of
The cheapest explanation for diagonal scratches is charging a diagonal step as one cell instead of the
square root of two, which makes diagonals artificially cheap and the router prefers them. **MEASURED by
a test that could have failed** (`macro_lattice.rs`): our diagonal chord is the square root of two times
a side, to within 0.02. We carry the classical D8 bias, not an arithmetic slip.
