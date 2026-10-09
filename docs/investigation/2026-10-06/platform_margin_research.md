# Research: the relief inside a platform, the first surface, the margin width, the boundary shape

Date: 2026-10-06. Kind: read-only research. I changed no file in the repository.

## 0. How I did the work

- The session's web-search budget was empty before this task started. Thus I ran no open web search.
- I used these services with the agent string `voxeldust-research/1.0` and with no name or e-mail:
  1. Crossref, OpenAlex and Semantic Scholar, for the records and the abstracts.
  2. The HAL open archive, the Copernicus journals, PubMed Central and university repositories, for the full texts that are open.
  3. The FastScape code on GitHub, for the initial surface that FastScape uses.
  4. The CRUST1.0 data page, for the published crust grid.
- Each source line below says what I read: "full text", "abstract" or "record only".
- I made one measurement myself: statistics of the published CRUST1.0 grid (Appendix A). It is my reading of published data. It is not a published number.
- "UNVERIFIED" means that I did not read the text that holds the claim.
- The source list comes from known literature, checked against the records. "No law found" means "not found in these sources". It does not prove that no law exists.

## 1. The answers in one table

| Question | The single best-founded law | Earth calibrates | Still unknown | A measurement on the model's own output that can falsify it |
|---|---|---|---|---|
| Q1 relief inside a platform | The isostatic (flexural) response of the ground to a crust-thickness field inside the plate. The field is a NAMED scatter around the plate's computed thickness. | The scatter's relative spread (about 0.09 inside one continent) and its correlation length (about 365 km). Both from my CRUST1.0 reading. | No law scales the spread or the length to another body. On Earth's old platforms the ground does not follow the crust (r = 0.08). No source gives the relief at the moment the erosion starts. | (a) The share of initial land nodes on an exact flat (today 87 %; the law gives 0 %). (b) At the end of the solve: the height difference against the distance inside old plates, read against Earth's 131 m at about 110 km to 375 m at about 1 700 km. |
| Q2 the first surface of a landscape model | No published law fixes the first surface. The literature uses a small random field to break the symmetry and a routing without lattice lean. For this model: the Q1 scatter IS the first surface, so no flat stays for the tie-break. | Nothing new. The amplitude comes from Q1. | No published test of the first field's SPECTRUM on the final network (not found). | (a) The azimuth histogram of river links against the lattice rows and diagonals, after pass 1 and at the end. (b) Twin seeds: the same statistics, other river positions. (c) An amplitude series (×0.5, ×1, ×2). |
| Q3 the margin width | Rifted margin: the necking instability of the strong layer. The necking zone is half the extended dominant wavelength; the wavelength is 3 to 4 times the strong-layer thickness. Andean margin: the coast stands over the place where the plate interface meets the upper plate's Moho. | Rifted: the ratio 3 to 4 and the extension 12 to 30 %. Andean: the interface dip (23 ± 8°). | The widths of the hyperextended domain (about 50 km) and of the exhumation domain (about 60 km): measured, no law. Transform margins: measured only (necking about 50 km), no law. | The width over which the crust share falls from 1 to 0 at each boundary type, and the steepest coast slope. Earth: a continental slope of 3° to 4.2° on average, 14° at most. A 9 km step over one 8.2 km node is 48°. |
| Q4 the boundary shape | Transform boundaries lie on circles of latitude about the pole of relative rotation (Morgan 1968). Ridge segments stand at right angles to them. | The spacing of the transform offsets (no law found). | No law gives the whole planform of the plates. The arc rule of Frank (1968) fails on Earth's data. | The angle between each boundary segment and the local relative motion; the share of boundary length on exact great circles (today 100 %). |

## 2. Q1 — the relief inside a continental platform

### 2.1 What the model does today (from the task)

- Deep in a continent the crust share saturates at exactly 1. Thus the isostatic height is one number for the whole plate.
- Example: on the home planet one plate is one flat at 5 952.8 m over 23.5 % of all land. 87 % of the initial land nodes stand on an exact flat.

### 2.2 What the sources say

#### (a) The crust's thickness inside stable continents

- Christensen & Mooney (1995) (abstract). Extended crust has an average thickness of 30.5 km. Orogens have 46.3 km. "Shields and platforms have an average crustal thickness nearly equal to the global average." The weighted global average is 41.1 km. The mean crustal density is 2 830 kg m⁻³.
- Durrheim & Mooney (1991) (abstract). Archean crust is about 35 km thick. Proterozoic crust is about 45 km thick and has a thicker fast layer at its base. My CRUST1.0 reading does not show this difference (Appendix A: Archean 39.6–40.2 km, Proterozoic 38.5–40.4 km). The two sources disagree. I did not resolve it.
- Mooney, Laske & Masters (1998) (abstract): "A simple calculation of crustal isostasy indicates significant lateral variations in upper mantle density."
- Szwillus, Afonso, Ebbing & Mooney (2019) (abstract). A geostatistical (kriging) Moho model. The Moho uncertainty is less than 4 km in North America and Europe. It reaches 10 km or more in most of Africa. The paper fits covariances. I could not read them (UNVERIFIED).

#### (b) My reading of CRUST1.0 (Appendix A has the method)

- Cells: 1° × 1°, dry land without ice, crust types "platform", "Archean" and "Proterozoic". Count: 9 008 cells.
- The crust: 39.5 ± 4.2 km (1σ). The ground: 378 ± 323 m.
- The correlation of crust thickness with ground height: r = +0.08. The slope: 6.3 m per km of crust.
- The Airy law with 2 830 and 3 300 kg m⁻³ gives 142 m per km of crust. Applied to each region's crust spread, Airy gives a height spread 1.2 to 6.3 times larger than the observed height spread.
- In the well-studied regions the slope stays small: North America r = 0.40 and 21 m per km; Europe r = 0.19 and 4 m per km.
- For comparison, orogens follow the crust: r = 0.83 and 109 m per km.
- Inside one continent (six regions pooled), the root-mean-square difference between two points is:

| Distance (km) | Crust difference (km) | Height difference (m) |
|---|---|---|
| about 110 | 2.5 | 131 |
| 150–250 | 3.4 | 187 |
| 250–350 | 4.0 | 216 |
| 350–500 | 4.3 | 242 |
| 500–700 | 4.8 | 269 |
| 700–1 000 | 5.0 | 299 |
| 1 000–1 400 | 5.0 | 328 |
| 1 400–2 000 | 5.2 | 375 |

- Fit for the crust: an exponential field with σ = 3.7 km and a correlation length a = 365 km.
- Fit for the ground: a power law, Δh ≈ 138 m × (L / 100 km)^0.36, from about 110 km to 1 700 km.
- Between continents: the mean crust of the seven regions has σ = 1.4 km on 40.4 km (3.4 %). The mean height has σ = 143 m.
- A reading for the task: the per-plate "crust scatter" in [0.7, 1.4] has a relative spread of 19 % (uniform draw). Earth's stable continents differ by 3.4 %. The draw spans a factor of 2. Earth's province averages span a factor of 1.52 (extended crust 30.5 km to orogens 46.3 km). Earth's stable continents span a factor of 1.09 (39.1 to 42.5 km). This scatter drives the steps of up to about 9 km at plate boundaries.

#### (c) The laws that hold the ground up

1. Airy isostasy. The ground over a crust of thickness t:
   `h = t (ρm − ρc) / ρm − h0`
   where ρm is the mantle density, ρc the crust density and h0 a reference depth. It reads the body's own crust. It does not read gravity.
2. Flexural isostasy for a load inside the plate (the textbook thin-plate result; Forsyth 1985 states that continents carry surface loads and loads inside the plate, about equal in Kenya; Watts 2001). A thickness change t′(k) at wavenumber k gives the ground
   `h(k) = (Δρ / ρm) · t′(k) · Φ(k)`, with `Φ(k) = 1 / (1 + D k⁴ / (ρm g))` and `D = E Te³ / (12 (1 − ν²))`.
   Here Δρ = ρm − ρc, g is the body's gravity, Te its elastic thickness, E Young's modulus and ν Poisson's ratio. Φ = 1/2 at the wavelength `λ½ = 2π (D / (ρm g))^(1/4)`.
   Example (Earth-like inputs E = 70 GPa, ν = 0.25, g = 9.81 m s⁻²): Te = 20 km gives λ½ = 221 km; Te = 40 km gives 372 km; Te = 80 km gives 626 km. The ground shows less than half of a crust change that is shorter than λ½. The equation numbers in the sources: UNVERIFIED (I derived the filter from the thin-plate equation).
3. Whole-lithosphere isostasy — Lamb, Moore, Pérez-Gussinyé & Stern (2020) (abstract). The thickness of the conductive lithosphere controls the height of continents together with the crust. The mantle lithosphere's negative buoyancy counters the crust's positive buoyancy. Density contrasts: mantle lithosphere against crust 300 to 550 kg m⁻³; against asthenosphere 20 to 40 kg m⁻³. Regional density changes explain the global heights "without invoking dynamic topography greater than a few hundred meters". Te is 25 to 50 % of the conductive lithosphere's thickness.
4. Thermal isostasy — Hasterok & Chapman (2007, parts 1 and 2) (abstracts). After a correction for crust thickness and density (reference column 39 km, 2 850 kg m⁻³), the ground follows the heat flow. Thermal isostasy "can produce nearly 3 km of relief between cold shield platforms and hot rift zones". The best fit: a reference heat flow of 46.6 mW m⁻² and a 60:40 split between the reduced and the upper-crust heat flow. Raw heights show little correlation with heat flow.
5. Mantle roots — Kaban, Schwintzer, Artemieva & Mooney (2003) (abstract): mantle gravity anomalies of −100 to +100 mGal over cratons; chemical depletion lowers the root density by 1.1 to 1.5 % (1.7 to 2.5 % in Archean parts). Artemieva & Mooney (2001) (abstract): lithosphere 200–220 km and 300–350 km under Archean cratons, 200 ± 50 km early Proterozoic, 140 ± 50 km middle and late Proterozoic; the mantle lithosphere is 1.5 % (Archean and early Proterozoic) or 0.6–0.7 % (middle and late Proterozoic) less dense than the asthenosphere.

#### (d) Dynamic topography

- Hoggard, White & Al-Attar (2016) (abstract). Observed dynamic topography has peak amplitudes of about ±1 km at wavelengths of about 10³ km and about ±500 m at 10⁴ km. "These dynamic topographic signals evolve quickly with time."
- It needs the mantle's convection, which the model does not have. It changes in time; the land's shape in the game does not. Thus it does not fit as an initial field now.

#### (e) The spectrum of continental topography

- Vening Meinesz (1951), as quoted by Gagnon et al. (2006): the power spectrum E(k) ∝ k^−β with β ≈ 2.
- Turcotte (1987) (abstract): Earth's topography is "a well-defined fractal with D = 1.5", Brown noise, amplitude proportional to wavelength. "The data for Mars and the moon correlate well with the data for the earth." Venus has a smaller amplitude.
- Huang & Turcotte (1989) (abstract): Arizona, D = 2.09 for two-dimensional spectra; D = 1.52 for one-dimensional spectra, near Brown noise.
- Gagnon, Lovejoy & Schertzer (2006) (full text): continents β = 2.09, oceans β = 1.63; H = 0.66 for continents; the scaling holds from about 20 000 km down to about 40 m; the field is multifractal (α = 1.79, C1 = 0.12).
- Perron, Kirchner & Dietrich (2008, spectral signatures) (abstract): the spectral power "declines sharply above a frequency that corresponds roughly to hillslope length". This scale is far below an 8.2 km node.
- Rexer & Hirt (2015): a data-based degree-variance model of Earth's topographic potential to degree 90 000 (record only; content UNVERIFIED).
- All these spectra describe today's ground. Today's ground already carries the erosion of billions of years. Thus a spectrum used as the FIRST surface counts the erosion twice.

#### (f) Old surfaces and how long relief lives

- Beauvais & Chardon (2013) (full text, West Africa): denudation of 2 to 20 m per Myr "over most of its surface whatever the considered time interval"; a weathering-limited "steady state cratonic denudation regime". The local relief between successive paleosurfaces: mean 96 ± 72 m (45–24 Ma), 61 ± 47 m (24–11 Ma), 83 ± 55 m (11–0 Ma).
- Baldwin, Whipple & Tucker (2003) (abstract): the plain stream-power law decays a mountain range in 1 to 10 Myr; Airy rebound makes this at most 6 times longer; a transport-limited river 2 to 3 times; a critical shear stress about 20 times. Together they "could account for the presence of residual topography for hundreds of million years".

### 2.3 Which candidate can give a law that reads the body's facts

| Candidate | Does it read the body's facts? | What Earth fixes | Verdict |
|---|---|---|---|
| Crust-thickness spread (CRUST1.0, Szwillus 2019) | Partly: the spread can be a fraction of the body's own crust thickness. | The fraction (0.09) and the length (365 km). | A named scatter, not a law. Usable. |
| Airy or flexural isostasy | Yes: crust, densities, g, Te. | Nothing new (the model has them). | A law. It turns a field into heights. It makes no field. |
| Whole-lithosphere isostasy | Yes: crust, lithosphere thickness from thermal age, depletion. | The density contrasts (300–550 and 20–40 kg m⁻³). | A law. It explains why old ground does not follow the crust. It needs a second field (the lithosphere's thickness). |
| Thermal isostasy | Yes: heat flow from age and heat production. | The reference heat flow and the split. | A law for the height DIFFERENCE between plates of different thermal age. Not for the relief inside one cold plate. |
| Dynamic topography | Only with a mantle convection model. | Amplitude ±0.5–1 km. | Not now. It also changes in time. |
| Topography spectrum | No body fact, except perhaps gravity (not established). | β ≈ 2.09 and an amplitude. | A statistic of today's eroded ground. Not a first surface. |
| Residual relief data | No. | Readings only. | A reading for the end of the solve. |

The literature gives NO law for the relief of a platform at the moment its erosion starts. It gives laws that turn fields into heights, and measurements of today's fields.

### 2.4 RECOMMENDATION (Q1)

- The law: the model's own isostasy, made flexural with the body's Te and g, applied to a crust-thickness field inside each plate (Section 2.2 (c), item 2).
- The field: a NAMED scatter, "the platform crust scatter": a Gaussian random field around the plate's computed thickness t̄. It keeps the per-plate value as its mean.
- Earth calibrates two constants, from my CRUST1.0 reading (Appendix A):
  1. The relative spread σt / t̄ ≈ 0.09 (σ = 3.7 km on about 40 km).
  2. The correlation length a ≈ 365 km (exponential).
- Example: if a plate of the home planet has a 40 km crust, the plate gets σt ≈ 3.6 km. Airy alone gives about ±510 m of first relief. The flexure filter then lowers the parts shorter than λ½. The flats disappear. The first rivers follow this relief, not the lattice.
- Still unknown:
  1. No source scales the spread or the length to another body. The desert world and a moon get the same fraction and the same length unless the owner rules otherwise.
  2. On Earth's old platforms the ground does not follow the crust (r = 0.08). The mantle lithosphere or the crust's density compensates it. Thus the crust field gives more first relief than Earth's old platforms show today. The solve's erosion then removes it. Whether the solve removes the right amount is a measurement, not an assumption.
  3. Whole-lithosphere isostasy can explain the decoupling. It needs a lithosphere-thickness field, which is a second scatter.
  4. No source gives the relief at the raise time itself (the relief of the reworking event).
- The second option (not recommended): a named scatter put straight into the height, with a Brown spectrum. It injects shape, not cause. It also counts erosion twice (Section 2.2 (e)).
- A side reading for the owner: the per-plate scatter [0.7, 1.4] is about 5.6 times wider than the spread of Earth's stable continents (19 % against 3.4 %). The task does not state its source.
- Measurements on the model's own output that can falsify this choice:
  1. The share of initial land nodes on an exact flat. Today: 87 %. With the law: 0 %.
  2. At the end of the solve, inside plates older than about 1 Gyr: the root-mean-square height difference against the distance, from 100 km to 2 000 km. Earth's readings: 131 m at about 110 km, 216 m at 300 km, 269 m at 600 km, 328 m at 1 200 km, 375 m at 1 700 km. A curve far above or far below these readings falsifies the scatter's size or the decay. (The readings are a check, never a target.)
  3. A reading, not a gate: the correlation of crust thickness with height on old plates at the end. Earth reads r = 0.02 to 0.46 and 2 to 35 m per km. A model that reads r near 1 and 142 m per km keeps a crust signal that Earth's old platforms do not show.

## 3. Q2 — the first surface in landscape evolution models

### 3.1 What the models use

- FastScape (code, full text): the process `FlatSurface` is "a flat surface at sea-level with random perturbations (white noise)". The code draws a uniform random number in [0, 1) per node, in metres. Braun & Willett (2013) is the solver paper (record only; its own statement on the initial surface UNVERIFIED).
- Cordonnier, Bovy & Braun (2019) (full text), FastScape runs: "a 500 m high flat surface with small random perturbations". "With no perturbation, all local minima are removed after the first time step." "A greater magnitude of perturbation of uplift rates reduces the rate at which the local minima disappear."
- Perron, Dietrich & Kirchner (2008, valley spacing) (full text): "The initial condition consists of a low-relief (1 m), pseudofractal surface generated by taking the inverse Fourier transform of a two-dimensional, red noise power spectrum." They use the D∞ routing of Tarboton (1997).
- Theodoratos, Seybold & Kirchner (2018) (full text): "Initial elevations were a uniform white noise, ranging between 0 and 0.1 hc", on random irregular meshes (TINs). For the law K√A|∇z| the height scale is hc = U/K.
- Kwang & Parker (2017) (full text): "The initial condition is a gently sloped plane oriented towards the outlet with small random elevation perturbations." D8 routing.
- Bonetti, Hooshyar, Camporeale & Porporato (2020) (full text): "Convex profiles were used as an initial condition. Over wide rectangular domains … a white noise with SD equal to 10⁻⁶ m was also added". D∞ routing.
- Barnhart et al. (2019, terrainbento) (full text): "Each landscape was initialized with the same random noise field".
- CHILD (Tucker et al. 2001) uses triangulated irregular networks (record). Braun & Sambridge (1997) (abstract) show the advantages of an irregular mesh over "'classical' models based on regular (rectangular) discretization". Willgoose, Bras & Rodriguez-Iturbe (1991) and Landlab (Hobley et al. 2017): I found no statement on the first surface (UNVERIFIED).

### 3.2 Why a perturbation: the symmetry and the lattice

- Fairfield & Leymarie (1991) (abstract): "Unless the terrain is rugged, the derived water channels tend to flow in parallel lines along preferred directions engendered by the sampling grid orientation." Their cure is the random Rho8 rule.
- Tarboton (1997) (abstract): eight-direction routing introduces grid bias; slope-proportional routing introduces unrealistic dispersion.
- Perron et al. (2008) (full text), after Willgoose (2005): steepest-descent routing "inhibits convergent flow, such as the incision of incipient valleys in a nearly planar surface, because subtle changes in the topography are not sufficient to cause an increment of 45° in the drainage direction".
- Barnes, Lehman & Mulla (2014) (full text) on flats: the gradient toward lower terrain alone "results in parallel flows"; "parallel flow patterns are uncommon in nature". Their method combines both gradients to get convergent flow. Their flats come from elevation data; they state that many flats are technical artefacts of the data.
- My inference for this model (not a source claim): both gradients count lattice steps (a king-move distance). On a flat that is thousands of kilometres wide, the lines of equal step count are lattice squares. Thus the flow lines follow rows and diagonals. This matches the task's measurement. Rho8 does not help on an exact flat: every slope is zero, so the random factor has nothing to act on.
- Pelletier (2004) (abstract): steepest-descent routing reaches an ideal steady state; bifurcation routing gives "continually-evolving drainage basins". The routing rule changes the result.

### 3.3 What the sources say about the AMPLITUDE

- Howard (1994) (abstract): "The arrangement of divides and valleys is sensitive to initial conditions, although average morphology is insensitive. Dissected, initially flat surfaces in which downstream concavity is slight exhibit nearly parallel drainage, compared to very wandering main valleys when concavity is great."
- Simpson & Schlunegger (2003) (abstract): "Surface and channel network morphologies are also strongly influenced by the overall surface slope relative to the magnitude of initial topographic roughness." "Roughness exponents are found to be independent of De but dependent on the magnitude of initial roughness."
- Ijjász-Vásquez, Bras & Moglen (1992) (abstract): "Evolutions were found to separate in time following a power law, explaining the apparent randomness present in catchments simulated with only slight perturbations in their initial conditions. This apparent randomness is the result of the weakly chaotic features of the model."
- Theodoratos et al. (2018) (full text): the dimensionless equation "has only one solution for any given set of (dimensionless) boundary and initial conditions"; landscapes "whose initial conditions are rescaled vertically by hc will follow temporally and geometrically similar evolutions". The initial conditions "influence the final relief; e.g., Howard, 1994". Thus the amplitude matters in units of hc.
- Perron et al. (2008) (full text): basins "that initially capture more area due to slight differences in initial conditions are able to stunt the growth of smaller adjacent basins". Valley spacing does not change with resolution; statistically identical initial surfaces give the same mean spacing.
- Kwang & Parker (2017) (full text): "The scatter in the synthetic profile is due to the randomness in the pathway, as dictated by the initial conditions."
- Rinaldo et al. (2014) (full text): networks settle in "feasible optimality, i.e., the search for optima that are accessible to the dynamics given the initial conditions"; "channel networks cannot change widely regardless of their initial condition, because these conditions leave long-lived geomorphic signatures". The ground-state networks show "somewhat unrealistic regularity".
- Bonetti et al. (2020) (full text): a sensitivity analysis shows the ridge-and-valley organization "robust to the choice of initial conditions" (with D∞ routing and a 10⁻⁶ m noise).
- On the SPECTRUM of the perturbation: the papers use white noise (FastScape, Theodoratos, Bonetti) or red noise (Perron). I found no paper that tests the spectrum's effect on the final network.
- On parallel valleys: on a uniformly tilted plane, evenly spaced parallel valleys are physical at the hillslope scale (Perron, Kirchner & Dietrich 2009, abstract). That scale (hundreds of metres) is far below an 8.2 km node. Lattice-aligned rivers at 8.2 km are an artefact.

### 3.4 RECOMMENDATION (Q2)

- The literature gives NO law for the first surface. It uses a small random field as a numerical device, and a routing without lattice lean (D∞, Rho8, irregular meshes).
- For this model, the best-founded choice: make the Q1 scatter the first surface. Do not add a second, numerical noise. The first relief then has a physical size (hundreds of metres over hundreds of kilometres, slopes near 10⁻³). The flat tie-break then acts only on true flats (for example a filled lake).
- Earth calibrates nothing new.
- Still unknown: the effect of the field's spectrum on the final network (not found in the sources).
- Measurements that can falsify the choice:
  1. The azimuth histogram of the river links against the lattice rows and diagonals, after pass 1 and at the end of the solve. A lattice-free network shows no excess at 0°, 45° and 90°. Example: on the home planet today, the first rivers on a flat plate run along rows and diagonals (the task's measurement); the histogram then shows spikes at 0°, 45° and 90°.
  2. Twin seeds of the scatter: the statistics (Hack's exponent, the share of land over 1 km, the relief) agree within the gate's tolerance; the river positions differ (Howard 1994; Ijjász-Vásquez et al. 1992).
  3. An amplitude series (the scatter × 0.5, × 1, × 2). If the statistics move by more than the gate's noise rule, the result depends on the amplitude. Then the amplitude needs its own law.

## 4. Q3 — the width of a continental margin across a plate boundary

### 4.1 Rifted (passive) margins

- The domains (Sutra et al. 2013, used by Chenin et al. 2017): the proximal domain (crust about 30–35 km), the necking domain (the crust thins from about 30–35 km to about 10 km), the hyperextended domain (10 km to 0 km), the exhumation domain (mantle at the sea floor), the oceanic domain.
- Chenin et al. (2017) (full text), magma-poor margins:
  - the distal domain: 130 to 240 km wide, mean 170 km, median 165 km;
  - the necking domain: 10 to 100 km, mean about 55 km, standard deviation 25 km;
  - the hyperextended domain: 20 to 70 km, mean 50 km, standard deviation 15 km;
  - the exhumation domain: 20 to 110 km, mean 60 km, standard deviation about 30 km;
  - "no correlation exists between the width of any of the distal sub-domains and the total width of the distal domain".
- Chenin, Schmalholz, Manatschal & Karner (2018) (full text). Models without strain softening give necking zones of 45 to 105 km, "independent on the type of applied boundary conditions and the initial Moho wavelength". "The necking width is likely controlled by the thickness of the brittle layer." Appendix A2 applies Fletcher & Hallet (1983):
  - the amplification `A / A0 = exp(−[1 + α] ε)`, with the amplification rate α = 40 to 100 for common flow laws, which gives a bulk extension ε of about 12 % to 30 %;
  - the dominant wavelengths are 25 to 123 km; the ratio of the dominant wavelength to the strong-layer thickness is about 3 to 4;
  - the necking zone is half of the extended wavelength: `W_neck ≈ ½ (1 + ε) λd`, with `λd ≈ (3 to 4) · Hs`;
  - this predicts 14 to 83 km; natural margins show 20 to 100 km.
- Fletcher & Hallet (1983) (abstract): a plastic surface layer over a power-law substrate (n = 3) necks; "If the depth to the brittle/ductile transition is roughly 10 km … spacings … of about 25–60 km".
- Buck (1991) (abstract): three modes, set by crust thickness, heat flow and strain rate. The wide-rift mode thins crust and mantle "over a width greater than the lithospheric thickness". The narrow-rift mode concentrates the extension.
- Huismans & Beaumont (2011) (abstract): type I margins (Iberia–Newfoundland, mantle at the sea floor, crustal necking first) and type II margins (ultrawide central South Atlantic, thin crust over wide regions, mantle breakup first).
- Svartman Dias, Lavier & Hayman (2015) (abstract): four end members; asymmetric conjugates with one side less than 100 km and the other 100 to 300 km; very wide conjugates 200 km to more than 350 km.
- Brune, Heine, Pérez-Gussinyé & Sobolev (2014) (full text): hyperextended crust 70 km wide off Iberia, 200 km off Angola; "faster extension results in wider margins"; the wider margin's width "increases almost linearly with rift velocity" (models at 4, 8 and 10 mm per year full rate, and the South Atlantic). Thin hot lithosphere (90 km) rifts symmetrically without migration. No closed-form constant is published in the text I read.
- Kooi, Cloetingh & Burrus (1992) (abstract): the depth of necking (15–20 km at margins; 25–35 km in the Gulf of Lions) sets the flexural state. Braun & Beaumont (1989) (abstract) and Weissel & Karner (1989) (abstract): flank uplift and the breakup unconformity follow from flexural (regional) compensation during necking; the lithosphere keeps a finite flexural rigidity. These give no single width law in the abstracts.
- Péron-Pinvidic, Manatschal & Osmundsen (2013): the review of the domains (record only; UNVERIFIED content).

### 4.2 Subduction (Andean-type) margins

- The coast. Ruff & Tichelaar (1996) observed "a correlation between the coastline location and downdip limit of the major circum-Pacific subduction interface seismogenic zones" and inferred that "both … are determined by the intersection of the overlying plate's Moho with the slab top" (as quoted by Heuret et al. 2011; the chapter itself: record only). Melnick (2016) (abstract): the rasa of the central Andes rises 0.13 ± 0.04 mm per year over more than 2 000 km; the uplift follows slip in earthquakes "straddling the crust–mantle transition, the Moho"; "the coastline marks the surface expression of the transition between the shallow, locked seismogenic domain and the deeper, conditionally stable domain".
- The interface. Heuret et al. (2011) (full text): the updip limit at 11 ± 4 km depth; the downdip limit at about 51 ± 9 km; the dip of the seismogenic zone 23 ± 8°; the downdip width 112 ± 40 km. The downdip limit reaches 65 ± 20 % of the trench–arc distance. Andean distances from the trench to the downdip limit: Colombia 108 km, northern Peru 115 km, southern Peru 123 km, northern Chile 135 km, southern Chile 210 km. A caution: "The seismogenic zone was usually found to end in a fore-arc mantle, rather than at a Moho depth" (mean 23 ± 20 km below the forearc Moho, 17 segments). Tichelaar & Ruff (1993) (abstract): the coupling ends at 40 ± 5 km for most zones.
- The law that reads the model's own facts:
  `x_coast ≈ (zM − dT) / tan δ`
  where zM is the upper plate's Moho depth (the model's own continental crust thickness), dT the trench depth (the model's own ocean floor) and δ the dip of the shallow interface.
  Example: zM = 45 km, dT = 6 km, δ = 23° gives x_coast ≈ 92 km.
- The dip δ has no law from the body's facts in these sources. Lallemand, Heuret & Boutelier (2005) (abstract): the dip follows the upper plate's absolute motion; slabs dip about 20° more steeply under oceanic upper plates; the dip does not follow the slab age, the convergence rate or the slab pull. Slab2 (Hayes et al. 2018, abstract) gives the measured geometry of all active slabs.
- The arc. Syracuse & Abers (2006) (abstract): the depth to the slab under the arc, H, averages 105 km over 500 km segments (72 to 173 km); it correlates with the slab dip (r = 0.54). England, Engdahl & Thatcher (2004) (abstract): H is 65 to 130 km, constant inside a segment, and falls as the descent speed rises; it does not follow the slab age or the thermal parameter. England & Katz (2010) (abstract): arcs stand "above the place where the boundary defined by the anhydrous solidus makes its closest approach to the trench". Grove et al. (2009) (abstract): H ≈ 60 to 173 km; the arc position follows the slab dip and the convergence rate through the wedge's thermal structure. The geometric rule: `x_arc ≈ H / tan δarc`, with δarc the mean slab dip down to depth H.
- The gap grows in time. Dickinson (1973) (abstract): "Widths of arc-trench gaps in modern arc-trench systems are proportional to the past duration of magmatic activity in the associated arcs."

### 4.3 Transform margins

- Mercier de Lépinay et al. (2016) (full text): 78 transform margins; 16 % of all continental margins by length, 31 % of the non-convergent margins. "Transform margins show a narrow necking zone for the continental crust, typically 50 km-wide, and a steep Moho slope". Off French Guiana the Moho dips 11° to 18° across a 50 km necking zone; the next divergent margin (Amazon) has a 200 km necking zone and about 3°. The continental slope: 4.2° on average (divergent margins about 3°); 14 % of transform margins exceed 9°; the maximum is 14° (Falklands). The slope is 16 to 165 km wide.
- Basile (2015) (full text): transform faulting starts where the deformed zone is oblique to the relative motion; the obliquity and the rift width set the transform length. On the vertical motions: "Enhanced models are still needed."
- No law for the width was found. Only measurements.

### 4.4 RECOMMENDATION (Q3)

- The single best-founded law with a closed form: the necking instability (Fletcher & Hallet 1983, applied by Chenin et al. 2018):
  `W_neck ≈ ½ (1 + ε) λd`, `λd ≈ (3 to 4) · Hs`.
  Hs is the thickness of the strong upper layer. The model's nearest body fact is its elastic thickness Te; the identity Hs ≈ Te is an assumption, not a source result.
- Earth calibrates: the ratio λd / Hs (3 to 4) and the extension ε (12 % to 30 %). Check against Earth's 10 to 100 km necking zones.
- For an Andean margin: `x_coast ≈ (zM − dT) / tan δ`. Earth calibrates δ = 23 ± 8° (Heuret et al. 2011). It reads the model's own crust thickness and ocean depth.
- Still unknown (no law found):
  1. The hyperextended domain (about 50 km) and the exhumation domain (about 60 km): measured, no law. Their widths do not follow the total width.
  2. The rift-velocity effect (Brune et al. 2014): a linear trend in models and in the South Atlantic, with no published constant in the text I read.
  3. The transform margin: measured only (about 50 km necking, Moho 11–18°).
  4. The interface dip δ: no law from the body's facts.
- Measurements that can falsify the choice:
  1. At each boundary type, the distance over which the crust share falls from 1 to 0 in the initial land. Earth's readings: rifted necking 10 to 100 km (distal domain 130 to 240 km); transform necking about 50 km; Andean trench to downdip limit 108 to 210 km.
  2. The steepest coast slope in the initial land. Earth: 3° (divergent) and 4.2° (transform) on average, 14° at most. On an 8.192 km node these are 0.43 km, 0.60 km and 2.04 km per node. Example: the home planet's 9 km step over one node is 48°. Every reading above rejects it.

## 5. Q4 — the shape of plate boundaries

### 5.1 The laws and the evidence

- Transform boundaries. Morgan (1968) (abstract): the motion of one block against another is a rotation about a pole; "If two adjacent blocks have as common boundaries a number of great faults, all of these faults must lie on 'circles of latitude' about the pole of relative rotation." The relative speed is greatest at the pole's equator and zero at the pole. Circles of latitude are curved on the sphere, except the pole's equator. McKenzie & Parker (1967): the same rotation geometry for the North Pacific (record only).
- Ridges. Oldenburg & Brune (1972) (abstract): in freezing wax, "the characteristic orthogonal ridge transform fault system is a preferred mode of separation". Gerya (2010) (abstract): transform faults grow from an instability of the spreading boundary; "the resultant curved ridges become transform faults within a few million years". Macdonald, Scheirer & Carbotte (1991) (abstract): a hierarchy of segments; long segments tend to lengthen. Segment lengths: UNVERIFIED (not in the abstract).
- Subduction arcs. Frank (1968) (abstract): a flexible, inextensible thin spherical shell bends inwards through an angle θ only on a circle whose angular radius is ½θ; arcs of about 20° (22° at the trench) match dips of 45°. Tovish & Schubert (1978) (abstract), 28 segments: slow convergence (below 5 cm per year) gives arc radii below 10°; faster convergence gives 10° to more than 90°; "Slab dip and arc curvature are not correlated." Yamaoka, Fukao & Kumazawa (1986) (abstract): the slab shape follows the bending of an inextensible spherical shell, with tears at gaps in seismicity. Mahadevan, Bendick & Liang (2010) (abstract): free subduction gives "arc-shaped dimpled segments" from edge-buckling modes of thin spherical shells; forced subduction gives straight segments. Schellart et al. (2007) (abstract): slab width controls the curvature; narrow slabs (≤ 1 500 km) retreat fast and curve; slabs of about 2 000–3 000 km are nearly straight; wide slabs (≥ 4 000 km) stay nearly fixed at the centre with a convex shape and concave edges. Stegman et al. (2006) (abstract): the fastest retreat for trenches about 600 km wide.
- Collision fronts. Bendick & Bilham (2001) (abstract): the Himalayan boundary is a small circle of radius 1 696 ± 55 km (about 15°), centred at 91.6° E, 42.4° N.
- Rifted margins. Basile (2015) (full text): obliquity between the rift and the relative motion splits the margin into alternating divergent and transform segments.
- The plate layout. Mallard et al. (2016) (abstract): convection models with plate-like behaviour reproduce Earth's plate sizes; "The spacing between the slabs controls the layout of large plates, and the stresses caused by the bending of trenches break plates into smaller fragments." Bird (2003) (abstract): 52 plates; the cumulative number–area distribution follows a power law from 0.002 to 1 steradian. Sornette & Pisarenko (2003) (abstract): a power law with exponent 0.25 ± 0.05.
- Graphics practice, not a law. Cortial et al. (2019) (full text) build plates as spherical Voronoi cells and warp the distances "using a simple noise function" for irregular shapes. Under this project's rules that warp is a hand-made adjustment unless its spread is a named scatter from a source.

### 5.2 RECOMMENDATION (Q4)

- The literature gives NO law for the whole planform of the plates from a body's facts. Only convection models produce it (Mallard et al. 2016).
- The single best-founded local law: Morgan's (1968) rule. A transform boundary lies on a circle of latitude about the two plates' pole of relative rotation. A spreading boundary runs in segments at right angles to it (Oldenburg & Brune 1972). The rule reads the model's own plate motions. Example: two plates of the home planet that slide past each other get a shared fault on a circle about their relative pole, not on the great-circle bisector of their two sites.
- Earth calibrates: the spacing of transform offsets along a ridge (no law found; Macdonald et al. 1991 give a hierarchy).
- Still unknown:
  1. Convergent boundaries: Frank's rule (radius = ½ dip) fails on Earth's data (Tovish & Schubert 1978). Slab width sets the shape in models (Schellart et al. 2007) but no closed form was found.
  2. The planform as a whole: no law.
- Measurements that can falsify the choice:
  1. For each boundary segment, the angle between the segment and the local relative-motion vector. Transform parts: near 0°. Ridge parts: near 90°. Today's bisectors have no relation to the motion, so the histogram is flat.
  2. The share of boundary length on exact great circles. Today: 100 %.
  3. The azimuth histogram of coast segments against the lattice rows and diagonals (the same test as Q2.1).

## 6. References

Q1
- Artemieva, I. M., & Mooney, W. D. (2001). Thermal thickness and evolution of Precambrian lithosphere: A global study. J. Geophys. Res. 106(B8), 16387–16414. doi:10.1029/2000JB900439 (abstract)
- Baldwin, J. A., Whipple, K. X., & Tucker, G. E. (2003). Implications of the shear stress river incision model for the timescale of postorogenic decay of topography. J. Geophys. Res. 108(B3). doi:10.1029/2001JB000550 (abstract)
- Beauvais, A., & Chardon, D. (2013). Modes, tempo, and spatial variability of Cenozoic cratonic denudation: The West African example. Geochem. Geophys. Geosyst. 14, 1590–1608. doi:10.1002/ggge.20093 (full text, HAL)
- Christensen, N. I., & Mooney, W. D. (1995). Seismic velocity structure and composition of the continental crust: A global view. J. Geophys. Res. 100(B6), 9761–9788. doi:10.1029/95JB00259 (abstract)
- Durrheim, R. J., & Mooney, W. D. (1991). Archean and Proterozoic crustal evolution: Evidence from crustal seismology. Geology 19(6), 606–609. doi:10.1130/0091-7613(1991)019<0606:AAPCEE>2.3.CO;2 (abstract)
- Forsyth, D. W. (1985). Subsurface loading and estimates of the flexural rigidity of continental lithosphere. J. Geophys. Res. 90(B14), 12623–12632. doi:10.1029/JB090iB14p12623 (abstract)
- Gagnon, J.-S., Lovejoy, S., & Schertzer, D. (2006). Multifractal earth topography. Nonlin. Processes Geophys. 13, 541–570. doi:10.5194/npg-13-541-2006 (full text)
- Hasterok, D., & Chapman, D. S. (2007). Continental thermal isostasy: 1. Methods and sensitivity; 2. Application to North America. J. Geophys. Res. 112, B06414 and B06415. doi:10.1029/2006JB004663, doi:10.1029/2006JB004664 (abstracts)
- Hoggard, M. J., White, N., & Al-Attar, D. (2016). Global dynamic topography observations reveal limited influence of large-scale mantle flow. Nature Geosci. 9, 456–463. doi:10.1038/ngeo2709 (abstract)
- Huang, J., & Turcotte, D. L. (1989). Fractal mapping of digitized images: Application to the topography of Arizona and comparisons with synthetic images. J. Geophys. Res. 94(B6), 7491–7495. doi:10.1029/JB094iB06p07491 (abstract)
- Kaban, M. K., Schwintzer, P., Artemieva, I. M., & Mooney, W. D. (2003). Density of the continental roots: compositional and thermal contributions. Earth Planet. Sci. Lett. 209, 53–69. doi:10.1016/S0012-821X(03)00072-4 (abstract)
- Lamb, S., Moore, J. D. P., Pérez-Gussinyé, M., & Stern, T. (2020). Global whole lithosphere isostasy: Implications for surface elevations, structure, strength, and densities of the continental lithosphere. Geochem. Geophys. Geosyst. 21(10), e2020GC009150. doi:10.1029/2020GC009150 (abstract)
- Laske, G., Masters, G., Ma, Z., & Pasyanos, M. (2013). Update on CRUST1.0 — A 1-degree global model of Earth's crust. Geophys. Res. Abstracts 15, EGU2013-2658. Data: igppweb.ucsd.edu/~gabi/crust1.html (data read)
- Mooney, W. D., Laske, G., & Masters, T. G. (1998). CRUST 5.1: A global crustal model at 5° × 5°. J. Geophys. Res. 103(B1), 727–747. doi:10.1029/97JB02122 (abstract)
- Mooney, W. D., & Vidale, J. E. (2003). Thermal and chemical variations in subcrustal cratonic lithosphere: evidence from crustal isostasy. Lithos 71, 185–193. doi:10.1016/j.lithos.2003.07.004 (record only)
- Perron, J. T., Kirchner, J. W., & Dietrich, W. E. (2008). Spectral signatures of characteristic spatial scales and nonfractal structure in landscapes. J. Geophys. Res. 113, F04003. doi:10.1029/2007JF000866 (abstract)
- Rexer, M., & Hirt, C. (2015). Spectral analysis of the Earth's topographic potential via 2D-DFT: a new data-based degree variance model to degree 90,000. J. Geod. 89, 887–909. doi:10.1007/s00190-015-0822-4 (record only)
- Szwillus, W., Afonso, J. C., Ebbing, J., & Mooney, W. D. (2019). Global crustal thickness and velocity structure from geostatistical analysis of seismic data. J. Geophys. Res. Solid Earth 124, 1626–1652. doi:10.1029/2018JB016593 (abstract)
- Turcotte, D. L. (1987). A fractal interpretation of topography and geoid spectra on the Earth, Moon, Venus, and Mars. J. Geophys. Res. 92(B4), E597–E601. doi:10.1029/JB092iB04p0E597 (abstract)
- Watts, A. B. (2001). Isostasy and Flexure of the Lithosphere. Cambridge University Press (not read here; textbook source of the flexure filter).

Q2
- Barnes, R., Lehman, C., & Mulla, D. (2014). An efficient assignment of drainage direction over flat surfaces in raster digital elevation models. Comput. Geosci. 62, 128–135. doi:10.1016/j.cageo.2013.01.009 (full text, arXiv:1511.04433)
- Barnhart, K. R., et al. (2019). Terrainbento 1.0. Geosci. Model Dev. 12, 1267–1297. doi:10.5194/gmd-12-1267-2019 (full text)
- Bonetti, S., Hooshyar, M., Camporeale, C., & Porporato, A. (2020). Channelization cascade in landscape evolution. PNAS 117(3), 1375–1382. doi:10.1073/pnas.1911817117 (full text)
- Braun, J., & Sambridge, M. (1997). Modelling landscape evolution on geological time scales: a new method based on irregular spatial discretization. Basin Res. 9, 27–52. doi:10.1046/j.1365-2117.1997.00030.x (abstract)
- Braun, J., & Willett, S. D. (2013). A very efficient O(n), implicit and parallel method to solve the stream power equation governing fluvial incision and landscape evolution. Geomorphology 180–181, 170–179. doi:10.1016/j.geomorph.2012.10.008 (record only); FastScape code, module `processes/initial.py` (read)
- Cordonnier, G., Bovy, B., & Braun, J. (2019). A versatile, linear complexity algorithm for flow routing in topographies with depressions. Earth Surf. Dynam. 7, 549–562. doi:10.5194/esurf-7-549-2019 (full text)
- Fairfield, J., & Leymarie, P. (1991). Drainage networks from grid digital elevation models. Water Resour. Res. 27(5), 709–717. doi:10.1029/90WR02658 (abstract)
- Garbrecht, J., & Martz, L. W. (1997). The assignment of drainage direction over flat surfaces in raster digital elevation models. J. Hydrol. 193, 204–213. doi:10.1016/S0022-1694(96)03138-1 (record only)
- Hobley, D. E. J., et al. (2017). Creative computing with Landlab. Earth Surf. Dynam. 5, 21–46. doi:10.5194/esurf-5-21-2017 (full text; no initial-surface statement found)
- Howard, A. D. (1994). A detachment-limited model of drainage basin evolution. Water Resour. Res. 30(7), 2261–2285. doi:10.1029/94WR00757 (abstract)
- Ijjász-Vásquez, E. J., Bras, R. L., & Moglen, G. E. (1992). Sensitivity of a basin evolution model to the nature of runoff production and to initial conditions. Water Resour. Res. 28(10), 2733–2741. doi:10.1029/92WR01561 (abstract)
- Kwang, J. S., & Parker, G. (2017). Landscape evolution models using the stream power incision model show unrealistic behavior when m/n equals 0.5. Earth Surf. Dynam. 5, 807–820. doi:10.5194/esurf-5-807-2017 (full text)
- Pelletier, J. D. (2004). Persistent drainage migration in a numerical landscape evolution model. Geophys. Res. Lett. 31, L20501. doi:10.1029/2004GL020802 (abstract)
- Perron, J. T., Dietrich, W. E., & Kirchner, J. W. (2008). Controls on the spacing of first-order valleys. J. Geophys. Res. 113, F04016. doi:10.1029/2007JF000977 (full text)
- Perron, J. T., Kirchner, J. W., & Dietrich, W. E. (2009). Formation of evenly spaced ridges and valleys. Nature 460, 502–505. doi:10.1038/nature08174 (abstract)
- Rinaldo, A., Rigon, R., Banavar, J. R., Maritan, A., & Rodriguez-Iturbe, I. (2014). Evolution and selection of river networks: Statics, dynamics, and complexity. PNAS 111(7), 2417–2424. doi:10.1073/pnas.1322700111 (full text)
- Simpson, G., & Schlunegger, F. (2003). Topographic evolution and morphology of surfaces evolving in response to coupled fluvial and hillslope sediment transport. J. Geophys. Res. 108(B6), 2300. doi:10.1029/2002JB002162 (abstract)
- Tarboton, D. G. (1997). A new method for the determination of flow directions and upslope areas in grid digital elevation models. Water Resour. Res. 33(2), 309–319. doi:10.1029/96WR03137 (abstract)
- Theodoratos, N., Seybold, H., & Kirchner, J. W. (2018). Scaling and similarity of a stream-power incision and linear diffusion landscape evolution model. Earth Surf. Dynam. 6, 779–808. doi:10.5194/esurf-6-779-2018 (full text)
- Tucker, G. E., Lancaster, S. T., Gasparini, N. M., & Bras, R. L. (2001). The Channel-Hillslope Integrated Landscape Development model (CHILD). In Landscape Erosion and Evolution Modeling, 349–388. doi:10.1007/978-1-4615-0575-4_12 (record only)
- Willgoose, G. (2005). Mathematical modeling of whole landscape evolution. Annu. Rev. Earth Planet. Sci. 33, 443–459. doi:10.1146/annurev.earth.33.092203.122610 (abstract; quoted through Perron et al. 2008)
- Willgoose, G., Bras, R. L., & Rodriguez-Iturbe, I. (1991). A coupled channel network growth and hillslope evolution model: 1. Theory. Water Resour. Res. 27(7), 1671–1684. doi:10.1029/91WR00935 (abstract)

Q3
- Basile, C. (2015). Transform continental margins — part 1: Concepts and models. Tectonophysics 661, 1–10. doi:10.1016/j.tecto.2015.08.034 (full text, HAL)
- Braun, J., & Beaumont, C. (1989). A physical explanation of the relation between flank uplifts and the breakup unconformity at rifted continental margins. Geology 17, 760–764. doi:10.1130/0091-7613(1989)017<0760:APEOTR>2.3.CO;2 (abstract)
- Brune, S., Heine, C., Pérez-Gussinyé, M., & Sobolev, S. V. (2014). Rift migration explains continental margin asymmetry and crustal hyper-extension. Nat. Commun. 5, 4014. doi:10.1038/ncomms5014 (full text)
- Buck, W. R. (1991). Modes of continental lithospheric extension. J. Geophys. Res. 96(B12), 20161–20178. doi:10.1029/91JB01485 (abstract)
- Chenin, P., Manatschal, G., Picazo, S., Müntener, O., Karner, G., Johnson, C., & Ulrich, M. (2017). Influence of the architecture of magma-poor hyperextended rifted margins on orogens produced by the closure of narrow versus wide oceans. Geosphere 13(2). doi:10.1130/GES01363.1 (full text, HAL)
- Chenin, P., Schmalholz, S. M., Manatschal, G., & Karner, G. D. (2018). Necking of the lithosphere: A reappraisal of basic concepts with thermo-mechanical numerical modeling. J. Geophys. Res. Solid Earth 123, 5279–5299. doi:10.1029/2017JB014155 (full text, HAL)
- Dickinson, W. R. (1973). Widths of modern arc-trench gaps proportional to past duration of igneous activity in associated magmatic arcs. J. Geophys. Res. 78(17), 3376–3389. doi:10.1029/JB078i017p03376 (abstract)
- England, P., Engdahl, R., & Thatcher, W. (2004). Systematic variation in the depths of slabs beneath arc volcanoes. Geophys. J. Int. 156, 377–408. doi:10.1111/j.1365-246X.2003.02132.x (abstract)
- England, P. C., & Katz, R. F. (2010). Melting above the anhydrous solidus controls the location of volcanic arcs. Nature 467, 700–703. doi:10.1038/nature09417 (abstract)
- Fletcher, R. C., & Hallet, B. (1983). Unstable extension of the lithosphere: A mechanical model for basin-and-range structure. J. Geophys. Res. 88(B9), 7457–7466. doi:10.1029/JB088iB09p07457 (abstract; law as applied by Chenin et al. 2018)
- Grove, T. L., Till, C. B., Lev, E., Chatterjee, N., & Médard, E. (2009). Kinematic variables and water transport control the formation and location of arc volcanoes. Nature 459, 694–697. doi:10.1038/nature08044 (abstract)
- Hayes, G. P., et al. (2018). Slab2, a comprehensive subduction zone geometry model. Science 362, 58–61. doi:10.1126/science.aat4723 (abstract)
- Heuret, A., Lallemand, S., Funiciello, F., Piromallo, C., & Faccenna, C. (2011). Physical characteristics of subduction interface type seismogenic zones revisited. Geochem. Geophys. Geosyst. 12, Q01004. doi:10.1029/2010GC003230 (full text, HAL)
- Huismans, R., & Beaumont, C. (2011). Depth-dependent extension, two-stage breakup and cratonic underplating at rifted margins. Nature 473, 74–78. doi:10.1038/nature09988 (abstract)
- Kooi, H., Cloetingh, S., & Burrus, J. (1992). Lithospheric necking and regional isostasy at extensional basins 1. J. Geophys. Res. 97(B12), 17553–17571. doi:10.1029/92JB01377 (abstract)
- Lallemand, S., Heuret, A., & Boutelier, D. (2005). On the relationships between slab dip, back-arc stress, upper plate absolute motion, and crustal nature in subduction zones. Geochem. Geophys. Geosyst. 6, Q09006. doi:10.1029/2005GC000917 (abstract)
- Melnick, D. (2016). Rise of the central Andean coast by earthquakes straddling the Moho. Nature Geosci. 9, 401–407. doi:10.1038/ngeo2683 (abstract)
- Mercier de Lépinay, M., Loncke, L., Basile, C., Roest, W. R., Patriat, M., Maillard, A., & De Clarens, P. (2016). Transform continental margins — Part 2: A worldwide review. Tectonophysics 693, 96–115. doi:10.1016/j.tecto.2016.05.038 (full text, HAL)
- Péron-Pinvidic, G., Manatschal, G., & Osmundsen, P. T. (2013). Structural comparison of archetypal Atlantic rifted margins: A review of observations and concepts. Mar. Pet. Geol. 43, 21–47. doi:10.1016/j.marpetgeo.2013.02.002 (record only)
- Ruff, L. J., & Tichelaar, B. W. (1996). What controls the seismogenic plate interface in subduction zones? In Subduction: Top to Bottom, AGU Geophys. Monogr. 96, 105–111. doi:10.1029/GM096p0105 (record only; quoted through Heuret et al. 2011)
- Sutra, E., Manatschal, G., Mohn, G., & Unternehr, P. (2013). Quantification and restoration of extensional deformation along the Western Iberia and Newfoundland rifted margins. Geochem. Geophys. Geosyst. 14, 2575–2597. doi:10.1002/ggge.20135 (record only)
- Svartman Dias, A. E., Lavier, L. L., & Hayman, N. W. (2015). Conjugate rifted margins width and asymmetry: The interplay between lithospheric strength and thermomechanical processes. J. Geophys. Res. Solid Earth 120, 8672–8700. doi:10.1002/2015JB012074 (abstract)
- Syracuse, E. M., & Abers, G. A. (2006). Global compilation of variations in slab depth beneath arc volcanoes and implications. Geochem. Geophys. Geosyst. 7, Q05017. doi:10.1029/2005GC001045 (abstract)
- Tichelaar, B. W., & Ruff, L. J. (1993). Depth of seismic coupling along subduction zones. J. Geophys. Res. 98(B2), 2017–2037. doi:10.1029/92JB02045 (abstract)
- Weissel, J. K., & Karner, G. D. (1989). Flexural uplift of rift flanks due to mechanical unloading of the lithosphere during extension. J. Geophys. Res. 94(B10), 13919–13950. doi:10.1029/JB094iB10p13919 (abstract)

Q4
- Bendick, R., & Bilham, R. (2001). How perfect is the Himalayan arc? Geology 29(9), 791–794. doi:10.1130/0091-7613(2001)029<0791:HPITHA>2.0.CO;2 (abstract)
- Bird, P. (2003). An updated digital model of plate boundaries. Geochem. Geophys. Geosyst. 4(3), 1027. doi:10.1029/2001GC000252 (abstract)
- Cortial, Y., Peytavie, A., Galin, E., & Guérin, E. (2019). Procedural tectonic planets. Comput. Graph. Forum 38(2), 1–11. doi:10.1111/cgf.13614 (full text, HAL)
- Frank, F. C. (1968). Curvature of island arcs. Nature 220, 363. doi:10.1038/220363a0 (abstract)
- Gerya, T. (2010). Dynamical instability produces transform faults at mid-ocean ridges. Science 329, 1047–1050. doi:10.1126/science.1191349 (abstract)
- Macdonald, K. C., Scheirer, D. S., & Carbotte, S. M. (1991). Mid-ocean ridges: Discontinuities, segments and giant cracks. Science 253, 986–994. doi:10.1126/science.253.5023.986 (abstract)
- Mahadevan, L., Bendick, R., & Liang, H. (2010). Why subduction zones are curved. Tectonics 29, TC6002. doi:10.1029/2010TC002720 (abstract)
- Mallard, C., Coltice, N., Seton, M., Müller, R. D., & Tackley, P. J. (2016). Subduction controls the distribution and fragmentation of Earth's tectonic plates. Nature 535, 140–143. doi:10.1038/nature17992 (abstract)
- McKenzie, D. P., & Parker, R. L. (1967). The North Pacific: an example of tectonics on a sphere. Nature 216, 1276–1280. doi:10.1038/2161276a0 (record only)
- Morgan, W. J. (1968). Rises, trenches, great faults, and crustal blocks. J. Geophys. Res. 73(6), 1959–1982. doi:10.1029/JB073i006p01959 (abstract)
- Oldenburg, D. W., & Brune, J. N. (1972). Ridge transform fault spreading pattern in freezing wax. Science 178, 301–304. doi:10.1126/science.178.4058.301 (abstract)
- Schellart, W. P., Freeman, J., Stegman, D. R., Moresi, L., & May, D. (2007). Evolution and diversity of subduction zones controlled by slab width. Nature 446, 308–311. doi:10.1038/nature05615 (abstract)
- Sornette, D., & Pisarenko, V. (2003). Fractal plate tectonics. Geophys. Res. Lett. 30(3), 1105. doi:10.1029/2002GL015043 (abstract)
- Stegman, D. R., Freeman, J., Schellart, W. P., Moresi, L., & May, D. (2006). Influence of trench width on subduction hinge retreat rates in 3-D models of slab rollback. Geochem. Geophys. Geosyst. 7, Q03012. doi:10.1029/2005GC001056 (abstract)
- Tovish, A., & Schubert, G. (1978). Island arc curvature, velocity of convergence and angle of subduction. Geophys. Res. Lett. 5(5), 329–332. doi:10.1029/GL005i005p00329 (abstract)
- Yamaoka, K., Fukao, Y., & Kumazawa, M. (1986). Spherical shell tectonics: Effects of sphericity and inextensibility on the geometry of the descending lithosphere. Rev. Geophys. 24(1), 27–53. doi:10.1029/RG024i001p00027 (abstract)

## Appendix A. My CRUST1.0 reading — method and numbers

- Data: CRUST1.0 (`crust1.bnds`, nine boundaries per 1° cell) and its type add-on (`CNtype1-1.txt`, `CNtype1_key.txt`), from the CRUST1.0 page. The page states that in CRUST1.0 the crust types assign only the elastic parameters, not the thickness. The topography is ETOPO1 averaged over each 1° cell.
- The ground height: the bottom of the ice boundary (the top of sediment or rock), in metres.
- The crust thickness: from the top of the solid ground to the Moho, sediments included.
- Cells: water depth 0, ice thickness 0, height above 0, and crust type D- (platform), E- (slow thin platform), G1 (early Archean), G2 (late Archean), H1 (early/middle Proterozoic), I1 (late Proterozoic), I2 (slow late Proterozoic). The types for Antarctica are out.
- Per type:

| Type | Cells | Crust (km) | Height (m) | r | Slope (m per km) |
|---|---|---|---|---|---|
| D- platform | 747 | 39.9 ± 4.0 | 204 ± 240 | +0.23 | 13.6 |
| E- slow thin platform | 323 | 37.1 ± 3.7 | 198 ± 200 | −0.02 | −1.3 |
| G1 early Archean | 1 260 | 40.2 ± 4.1 | 597 ± 368 | +0.17 | 15.1 |
| G2 late Archean | 1 042 | 39.6 ± 4.1 | 335 ± 281 | +0.12 | 8.2 |
| H1 early/middle Proterozoic | 3 375 | 40.4 ± 3.8 | 305 ± 262 | +0.09 | 6.4 |
| I1 late Proterozoic | 2 052 | 38.5 ± 4.4 | 478 ± 356 | +0.08 | 6.4 |
| I2 slow late Proterozoic | 209 | 34.5 ± 4.0 | 376 ± 193 | −0.20 | −9.9 |
| All stable | 9 008 | 39.5 ± 4.2 | 378 ± 323 | +0.08 | 6.3 |
| Phanerozoic (Z1, Z2), for comparison | 1 398 | 37.1 ± 3.5 | 362 ± 327 | +0.14 | 12.9 |
| Orogens (P-, Q-, R1, R2), for comparison | 2 440 | 45.3 ± 9.5 | 1 629 ± 1 251 | +0.83 | 109.4 |

- Per region (latitude–longitude boxes):

| Region | Cells | Crust (km) | Height (m) | r | Slope (m per km) | Airy σ of height (m) |
|---|---|---|---|---|---|---|
| North America 40–70N, 115–60W | 1 145 | 40.0 ± 4.2 | 340 ± 224 | +0.40 | 21.3 | 596 |
| Eastern Europe and Baltic 45–70N, 20–60E | 799 | 42.5 ± 3.6 | 150 ± 81 | +0.19 | 4.2 | 512 |
| Siberia 50–75N, 90–130E | 742 | 42.5 ± 3.5 | 430 ± 298 | +0.42 | 35.3 | 500 |
| Africa 35S–20N, 20W–40E | 1 562 | 39.4 ± 3.3 | 648 ± 392 | +0.13 | 15.2 | 466 |
| South America 35S–5N, 70–35W | 891 | 39.1 ± 2.9 | 297 ± 230 | +0.02 | 1.7 | 409 |
| Australia 35–15S, 113–145E | 485 | 39.3 ± 4.2 | 283 ± 162 | +0.46 | 17.7 | 597 |
| India 8–25N, 72–88E | 109 | 39.8 ± 5.2 | 403 ± 188 | +0.19 | 7.1 | 734 |

- "Airy σ of height" = the crust's σ × (3 300 − 2 830) / 3 300 = σ × 0.142.
- The structure function (Section 2.2 (b)) pools all point pairs inside each of six regions (all regions except India). The distance is the great-circle distance on a 6 371 km sphere.
- Limits of this reading:
  1. CRUST1.0 fills data gaps by interpolation. The Moho error reaches 10 km or more in parts of Africa (Szwillus et al. 2019). Errors add variance and lower the correlation. The well-studied regions still show slopes far below Airy (4 to 21 m per km).
  2. The 1° cells smooth all relief shorter than about 100 km.
  3. Today's ground carries billions of years of erosion, sediment and dynamic topography. These numbers are readings of today's Earth, never targets for the first surface.
