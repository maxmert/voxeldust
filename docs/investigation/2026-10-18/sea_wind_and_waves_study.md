# The sea's wind and the waves' sources — a study (ruling B111)

> Written in ASD-STE100. Two read-only studies and one bench (2026-10-18), after the owner's look at slice S3: *"waves
> definitely not like on the sea (waves are not big enough), more like on windless lake"*. Marks: MEASURED (a script or
> the bench), STATED (a source says it), ESTIMATE (arithmetic). The scripts are outside the repo (`/tmp/vd-wind-laws/`,
> `/tmp/vd-waves-check/`); the bench is `climate_bench chain-winds`.

## 1. The wind: what the bench measured

The bench runs the game's seasons chain from cold on a stored world and writes each season's wind in its two parts: the
cells' law (`seasons::wind`) and Gill's departures (`seasons::gill`). The readings are compared with ExoPlaSim's home run
(the same body's charter, T21) and NCEP R1's 10 m winds (Earth), as READINGS, never targets (B38, B41).

| Reading (the sea, the year's mean of the seasons' speeds) | The chain | The outside model |
|---|---|---|
| Earth's charter, median | 4.1 m/s | NCEP 4.1 m/s |
| The home planet, median | 2.0 m/s | ExoPlaSim 6.0 m/s (~140 m) |
| The home planet, 20° S–20° N | 1.4 m/s | 6.4–6.5 m/s |
| The home planet's mean cell, its mass flux at the solstices | 0.1–0.4 × 10¹¹ kg/s | 5.7–6.2 × 10¹¹ kg/s |

So the chain holds on Earth and fails on the home planet (55° tilt, a third of Earth's spin, 0.75 of Earth's sunlight).
The cells' mass flux `ψ = w·F/g_ms` is 15–35 times too weak, by three factors that multiply (MEASURED at the equator in
the solstice seasons):

1. **The energy the cell carries across the equator, F: ≤ 2.0 PW against 7.8–8.6 PW.** The balance (`seasons::balance`)
   carries energy only by the diffusion of moist energy, with Siler et al.'s Earth-fitted `D`. On ExoPlaSim's own
   temperatures that law carries at most 1.3 PW, with the wrong sign or nought in 4 of 12 months: diffusion cannot carry a
   cell's energy across the flat moist energy of the tropics.
2. **The gross moist stability g_ms: 40–61 kJ/kg against 13–16.** `h_T = 1.06 ×` the zonal maximum of the moist energy, and
   at the 55° tilt that maximum stands at the summer POLE in most seasons. Quasi-equilibrium reads the CONVECTING TROPICS.
3. **The cells' share w: 0.4 against ~0.9 on Earth** (Siler et al.'s Gaussian about a rising branch that jumps with the
   seasons).

Two more defects: the storm belts' rule spread EASTERLIES when the cells' torque came out westerly (−12 m/s at 60–90°);
and Gill's departures are about half of ExoPlaSim's stationary eddies (rms 0.5–1.6 against 2.3–3.5 m/s).

On Earth the three factors sit where the law was fitted (h's maximum in the tropics, g_ms ~21 kJ/kg, w ~0.9), and two
Earth-fitted numbers cancel (the 1.06 makes g_ms about 2× too large; the lower tenth of the mass makes v about 2× too large
for its ψ). With ExoPlaSim's own ψ the chain's wind law gives ExoPlaSim's zonal sea winds (median 5.70 against 5.68 m/s,
correlation u 0.84, v 0.93; MEASURED): **the wind law is right when the cell is right.**

## 2. The wind: the laws, in three steps (the research's recommendation)

1. **STEP 1 — the cells' law only** (`seasons::wind`). (a) `h_T` over the convecting tropics: the band of the equatorial
   deformation radius `φ_eq = √(c/(2Ω·a))` (Matsuno 1966; Gill 1980), `c = N·D/π` (Gill's eq. 2.1) from the body's own
   lapse rate and the tropopause at 0.1 bar (Robinson & Catling 2014) — Earth ~14°, the home planet ~24°; on Earth the
   maximum already stands there, so Earth does not move. (b) The storm belts' westerlies only when the cells' torque is
   easterly (the rule `seasons::gill` already holds). ESTIMATE: ψ ×2–4, the sea's median ~3–3.5 m/s.
2. **STEP 2 — the balance carries the cell's energy** (`seasons::balance`): inside the cells the free troposphere's
   temperature is even (the weak temperature gradient; Sobel, Nilsson & Polvani 2001), so the column's longwave reads one
   temperature per cell, solved each step from the cell's energy; the eddies diffuse outside the cells. Siler's Gaussian
   share retires (`w = F_cell/F`). ESTIMATE: F ×2–2.5, the sea's median ~4.5–5.5 m/s. A world change.
3. **STEP 3 — later, measured:** the Earth-fitted pair (1.06, the lower tenth) replaced by a lawful pair, and the trades'
   momentum from the free troposphere (Stevens, Duan, McWilliams & Neelin 2002; Back & Bretherton 2009).

Refused as a closure here: one angular-momentum-conserving cell (Held & Hou 1980; Lindzen & Hou 1988) — with the rising
branch at 60–70° its upper wind over the equator reads about −115 m/s, far outside the theory's range (ESTIMATE). Refused
as a cap: holding the rising branch inside a band (Privé & Plumb's rule keeps its own place).

**THE WAVES' SECOND MISSING PART: the wind of the moment.** ExoPlaSim's 4.2-day records give the home planet's sea an rms
speed of 7.3 m/s against 6.2 m/s for the months' vectors (a lower bound: its records are means); Earth's satellite
climatologies give a scalar mean near 7 m/s against NCEP's 4.1 m/s of monthly vectors (STATED, memory). The sea's height
goes as U², so the weather's wind matters as much as the climate's. It belongs to the weather step (B35, B67).

## 3. The waves: the open sources (read only; the clones outside the repo)

| Source | Licence | What it gives our plane-wave form |
|---|---|---|
| Bruneton, Neyret & Holzschuch 2010 (OceanLighting) | INRIA BSD-3 on the GLSL; `Main.cpp` has no header (port from the GLSL and the paper) | A sum of Gerstner plane waves with ONE continuous rule from geometry to normals to roughness; the anisotropic glint; a planet mode on a projected grid |
| Water Surface Wavelets (lecopivo) | MIT | The Gerstner profile channels in our 1D-profile form (height, horizontal, both slopes) |
| Dupuy & Bruneton 2012 (whitecaps) | INRIA BSD-3 on the GLSL | Foam as a closed-form statistic of the Jacobian, no persistent texture |
| GodotOceanWaves (2Retr0) | MIT | JONSWAP + TMA with the fetch laws, the Hasselmann spread with the sea's own age |
| FFT-Ocean (gasgiant) | MIT | A second spectrum for the swell, the Donelan–Banner spread |
| bevy-aqua | MIT / Apache-2.0 | A continuous resolved fraction per band (`optics.wgsl`) |
| Crest 4, Atlas (GDC 2019) | MIT / slides | Methods only (Crest draws on a clipmap chart; its numbers are knobs) |
| ecWAM | Apache-2.0 | The wave-action balance for a planet's swell (WAVEWATCH III is LGPL-3: read only) |

Every source shows its sea at 10–42 m/s with hand-set knobs (MEASURED in their demo files); at 1.9 m/s each one is a lake
by law (Hs 0.08 m, 3 m waves; Beaufort 2, "small wavelets, crests glassy"). An open-sea look is Beaufort 4–6, U₁₀ 8–12 m/s
over 100 km or more, or swell.

**Found in S3 and CURED the same day** (`sea_waves_design.md` §6 item 5): the energy was 1.69 times too large (β 0.74 at
the peak frequency; the law's β at the peak is 5/4), and a whole band left the drawn normal at one footprint (now the
pixel's own filter, one continuous rule).

**The re-use plan (inside B106, B107, W16, T3, B23):**

1. **The spectrum:** JONSWAP + TMA with the fetch (GodotOceanWaves, MIT) to the Pierson–Moskowitz limit, or the
   Elfouhaily et al. 1997 unified spectrum (one law from the peak to the capillaries, its slopes fitted to Cox & Munk, so
   the glint and the drawn waves come from one law). The fetch at each place from the coast mask and the wind (a possible
   SL6 ask).
2. **The swell:** a second spectrum at each place, its numbers from the wave-action balance solved once on the planet's own
   seasonal winds (the storm belts the sources; great circles at the group speed; Ardhuin et al. 2009's decay; land from
   the coast mask) — ported from ecWAM, run by the planet's shard at load like the wind (an SL6 ask).
3. **The near waves (S5):** the profile carries the Gerstner channels (Water Surface Wavelets); linear theory gives the
   horizontal motion `a·coth(kh)` with no free factor (every source's "chop" knob refused); the normal from the displaced
   tangents; the mesh either Bruneton's projected grid near the eye or today's water triangles tessellated near the eye
   (W18 at the shore) — chosen by a picture.
4. **The foam (S4):** Dupuy & Bruneton's coverage `P(J < ε)` from the spectrum's statistics, its share set by Monahan &
   O'Muircheartaigh 1980's whitecap law `W = 3.84 × 10⁻⁶ U^3.41`; no persistent texture.
5. **The far field:** the anisotropic Cox & Munk glint (σ_u² = 3.16 × 10⁻³ U, σ_c² = 0.003 + 1.92 × 10⁻³ U).

**Example.** Off the coast stand today the climate blows 1.9 m/s and the sea shows glassy wavelets. With the cells' law
cured the trades over the home planet's tropical sea blow several metres a second; a swell from the southern storm belt,
about 225 m from crest to crest, rolls through the local wavelets; near the walker its crests lift the water, and at 3 km
its slopes are a sheen with no line where the near waves end.

## 4. Step 1 on the bench (MEASURED, 2026-10-18)

Built in `seasons::wind` (not yet in the game's solve): `h_T` over the convecting tropics' band (`tropics_band_sin`), the
local quasi-equilibrium (`g_ms ≥ 0.06·h`: without it a summer pole more energetic than the tropics' free troposphere drove
g_ms to nought and ψ to 30 × 10¹¹ kg/s at 59° S), and the storm belts' sign rule.

| Reading | Before | Step 1 | Outside model |
|---|---|---|---|
| Earth, the sea's median | 4.1 m/s | 4.1 m/s | NCEP 4.1 m/s |
| Home, the sea's median | 2.0 m/s | 2.2 m/s | ExoPlaSim 6.0 m/s |
| Home, g_ms at the equator | 20–61 kJ/kg | 20–30 kJ/kg | 13–16 kJ/kg |
| Home, 60–90° | 5.4 m/s | 2.7 m/s | 3.1 m/s |
| Home, ψ at the equator | 0.0–0.18 × 10¹¹ | 0.0–0.28 × 10¹¹ | 3.7–6.0 × 10¹¹ |

Step 1 is right and small: the energy across the equator (F ≤ 2 PW) and the share (0.4) still hold the cell. Step 2 (the
balance carries the cell's energy) is the main lever.

## 5. Step 2's closure: the research (2026-10-18, ruling B112; a NumPy replica of `seasons::balance`, outside the repo)

1. **No published conceptual model is proven for a seasonal cross-equatorial cell at a high tilt and a slow spin**
   (Rose, Cronin & Bitz 2017; Ferreira et al. 2014; Mbengue & Schneider 2018 a dry annual model with a hand-set Hadley
   diffusivity; Bischoff & Schneider 2014/2016 diagnostic; Vladilo et al. 2015 state diffusion "is inappropriate inside the
   Hadley cells"). The nearest proven family is THE WEAK TEMPERATURE GRADIENT with two layers (Sobel & Bretherton 2000;
   Bretherton & Sobel 2002; Pierrehumbert 1995): one free-troposphere temperature over the equatorial band, a surface
   temperature per column, the surface flux `λ_i = c_q·∂h/∂T` (`c_q` 0.009 kg m⁻² s⁻¹, Bretherton & Sobel's Table 1,
   Earth's tropical ocean, named).
2. **The diagnosis on ExoPlaSim's own fields (MEASURED):** ExoPlaSim's summer half radiates LESS than its winter half
   (convecting, humid) — the balance's `A + B·T` makes it radiate more (a 6.4 PW swing); diffusion follows the sea's lagged
   temperature while ExoPlaSim's transport follows the sun (April −0.70 against the balance's +1.47 PW). Inside the cell
   the OLR does not follow the local surface temperature (R² 0.00–0.23).
3. **Refused on the replica:** the diffusivity's spin law `D ∝ Ω⁻²` (Williams & Kasting 1997; 12 PW and the wrong phase;
   no GCM supports it); one layer with an even OLR (the storage takes it; F stays ~1 PW); two layers over the whole cell
   (10–21 PW, overshoot).
4. **The first candidate — two layers over the deformation band, the band's mean sunlight to the air:** the home planet
   8.3 / −8.4 PW (ExoPlaSim 8.55 / −7.83) but Earth's charter 3.9 / −3.5 PW against the atmosphere's 2.1 / −2.3 (Donohoe et
   al. 2013) and a 50 m slab aquaplanet 3.2 PW against GFDL AM2.1's 0.5 — a match by compensating errors (its storage share
   69 % against ExoPlaSim's 86 %: the sunlight given to an air layer with no heat store bypasses the sea's storage).
   NOT built. NEXT: the corrected form (the surface keeps its sunlight; the air gets the surface flux; `A` closed on the
   surface's global mean) on the same three references, in the replica first.
5. **The corrected form (the surface keeps its sunlight, or the air takes Wild et al.'s 0.25 / 0.33; `A` closed on the
   surface's global mean, 288.0 K held) — MEASURED in the replica, NO variant passes all three references:**

   | Variant | Home Jan / Jul (ExoPlaSim 8.55 / −7.83) | Home equinoxes (≤ 0.9) | Storage (86 %) | Earth Feb / Aug (atm. 2.1 / −2.3) | 50 m aquaplanet (GFDL 0.5) |
   |---|---|---|---|---|---|
   | Today's law | 1.07 / −1.20 | 1.47 / −1.63 | 90 % | 1.70 / −1.39 | 1.18 |
   | Surface keeps its sun (f 0–0.33) | 8.9–9.2 / −9.0–−9.3 | 1.6–3.2 | 67–68 % | 4.0 / −3.9 | 3.3 |
   | + the exchange by a READ wind (diagnostic) | 7.3 / −7.2 | 1.4–1.9 | 73 % | 2.7 / −2.9 | 1.7 |

   The air's share of the sunlight was NOT the cause. THE CAUSE (MEASURED in ExoPlaSim inside ±24° over the sea): the
   evaporation FALLS with the surface temperature (−6.2 to −6.7 W m⁻² K⁻¹) and RISES with the wind (r +0.90) — the calm
   summer sea under the rising branch keeps its sunlight (+13 to +30 W/m² into the sea), the windy winter trades give
   theirs to the air (−89 to −121 W/m²); a fixed-wind exchange has the opposite sign. Two more defects of the two-layer
   form: the band's surface runs 3–6 K hot (the free troposphere needs its own outgoing-heat intercept), and with a read
   wind the tropical land overheats by 15–36 K (a lawful land exchange is missing).
6. **WHAT A PASSING FORM NEEDS (ESTIMATE, not built):** the surface exchange by the bulk law `E = ρ·C_k·|U|·(q*(T_s) − q_a)`
   with the cells' OWN wind (a loop between the wind and the balance, lagged one pass as the land's wetness is; its
   convergence unmeasured), the land's own exchange (gustiness, wetness, sensible and longwave terms), and the free
   troposphere's outgoing heat — a two-layer intermediate climate; each piece measured in the replica on the three
   references before any Rust.
7. **THE TWO-LAYER PROGRAM'S BLOCKER (B113, the replica, MEASURED).** Step 1 (the sea's exchange by the bulk law
   `E + H = ρ·C_k·|U_eff|·(…)`, `C_k` 1.2 × 10⁻³ Large & Pond 1982 / COARE, the gustiness of Fairall et al. 1996 from
   Deardorff's convective velocity) cannot stand on today's cells' wind law: THE LAW GIVES THE BAND'S WIND BACKWARDS —
   strong on the SUMMER half (7.2–9.7 m/s at 10–20° S in the home planet's January) and calm on the WINTER half (3.1–3.8),
   where ExoPlaSim reads 3.6–5.9 and 9.7–10.0 (and NCEP on Earth 1.5 against the law's 5.3 at 10° S in January); the
   cause is `ψ = w·F/g_ms` with the share centred on the summer rising branch, where g_ms is smallest. The balance ↔ wind
   loop does NOT converge on it (Earth dF 3.8–4.3 PW between passes, the 50 m aquaplanet swings sign). With the wind given
   as a READING the loop settles in 2–3 passes and the sea, the land's exchange and the free troposphere's intercept give
   the home planet 8.0 / −8.0 PW (ExoPlaSim 8.55 / −7.83) but still fail the equinoxes (1.6 / −2.0 against ≤ 0.9), the
   storage (70 % against 86 %), the peak month (one late), Earth (2.85 against 2.1–2.3) and the 50 m aquaplanet (1.75
   against 0.5); the band's surface stays 4.6 K hot on Earth — its cause at the band's border (the band presents `h(T_R)`,
   7–8 K under its surface; quasi-equilibrium at the border, untested). NEXT PIECE: a wind law with strong winter trades
   and a calm summer convergence — the heating-driven Gill/WTG law (L3′) with a vapour model in the replica — its first
   gate the band halves' speeds against ExoPlaSim and NCEP. The research stopped on the account's session limit; it
   resumes on that piece.
