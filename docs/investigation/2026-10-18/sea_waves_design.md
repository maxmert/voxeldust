# Slice S3 — the sea's waves as surface detail: a build note (rulings B107, B110)

> Written in ASD-STE100. The parent designs: `docs/investigation/2026-10-17/sea_look_design.md` (§3 S3, §5 the answers of
> B107) and `docs/investigation/2026-10-18/sea_light_design.md` (S2, built). Ruling B107 fixed the method; this note fixes
> the numbers and the order. The waves are LOOK ONLY (B106): no server reads them.

## 1. The laws the waves read

| Part | The law | Its inputs (each one already on the client) |
|---|---|---|
| The energy of each wavelength | Pierson & Moskowitz (1964), the fully developed wind sea: `S(ω) = α g² ω⁻⁵ exp(−β (ω_p/ω)⁴)`, α = 0.0081, β = 5/4 at the peak ω_p = 0.855 g / U (the law's 0.74 goes with ω₀ = g/U₁₉.₅; corrected in §6 item 5) | the wind's speed U (S2's wind), the body's gravity g |
| The shallow water | the TMA factor (Bouws et al. 1985, Kitaigorodskii's form) on the spectrum | the depth at the point (S2) |
| How a wave moves | the dispersion `ω = √(g k tanh(k h))` | g, the depth |
| The spread of directions | Hasselmann et al. (1980), `cos^{2s}(θ/2)` with s by ω/ω_p | the wind's direction (S2's wind, its east and north) |
| The slopes no pixel shows | LEADR: their variance joins the glint's roughness | the pixel's footprint; Cox–Munk's total (S2) |
| The time | the universe's own clock (ruling T3: closed form from the tick) | the picture's cursor |

The total slope variance stays Cox & Munk's (S2): the resolved waves take their own part of it, and the roughness keeps
the rest, `σ²_rough = σ²_CM(U) − σ²_resolved(footprint)`, never below the spectrum's part finer than the finest drawn band.
So the far sea looks as it does today, and the near sea gains its waves with no change of brightness (B16, B27).

## 2. The form: plane waves in the planet's own space (B107)

1. **Bands.** Wavelengths from 0.5 m to 512 m in ten octaves. Each band holds ONE 1D PROFILE (Jeschke et al. 2018, *Water
   Surface Wavelets*): `p_b(s, t) = Σ_j a_j cos(k_j s − ω(k_j) t + φ_j)`, the wavenumbers of the band on a period of 32
   of its longest waves, 1 024 samples, its height and its slope. The phases φ_j come from the world's seed (an identity
   choice, the same on every client); the shape within a band is the spectrum's saturation slope (k⁻³), so the profile's
   variance is one and each place scales it.
2. **Directions.** Sixteen fixed 3D unit directions d_i spread evenly over the sphere (a spherical Fibonacci set), turned
   once for each band by a seed-derived turn, so no two bands cross the same lines. A component is `x · d_i` along d_i,
   so it exists over the whole sphere with no chart and no seam.
3. **At each place.** The local wave vector of d_i is its part in the tangent plane, `k · d_t`. A component counts only
   where `|d_t| ≥ 1/√2` (its local wavelength within half an octave of the band's), with a smooth taper, and its weight is
   the spread at its angle to the wind: `w_i = D(θ_i − θ_wind) · taper(|d_t|) / Σ`. The band's amplitude at the place is
   `√(E_b(U, h))`, the spectrum's variance in the band.
4. **The normal.** `n' = normalize(n − Σ_b √E_b Σ_i √w_i p'_b(x·d_i + o_bi, t) d_t,i)` for each band whose shortest wave
   spans at least two pixels; the finer bands go to the roughness (LEADR). The offsets `o_bi = (x_eye · d_i) mod period_b`
   are taken in f64 on the CPU each frame, so the shader works near the eye in f32 (B107).
5. **No displacement** in S3: the waves are normals and roughness. Near 3D waves are S5 (B107 Q3).

## 3. The data

- **Per frame, per realm with a sea** (the band table's own buffer, after the sun's row): the profiles (10 bands × 1 024
  samples × height and slope, 80 KB), the directions in render space (16 × 3 floats per band), the offsets (160 floats).
  The CPU builds the profiles from the bands' few wavenumbers (an inverse transform of 16–32 terms per sample); measured,
  and moved to the card if it costs the frame.
- **Per water point:** the wind as a vector in the planet's frame (m/s), beside S2's depth and slope variance.
- **The time:** the picture's cursor in universe seconds, so two players see one sea.

## 4. The order and the checks

1. The spectrum and the profiles on the CPU, with unit tests (the variance, the dispersion, the band split).
2. The data path (the band table's wave block, the wind per water point).
3. The shader: the normal and the roughness.
4. MEASURED: the frame's cost with the waves on and off (`VD_TERRAIN_WATER_WAVES=0`), the 60 fps rule (B22).
5. Pictures headless at 100 m, 3 km and 30 km, then the owner's look in the window.

## 5. Interims named now (registered at the build)

- **No fetch:** the sea is fully developed everywhere; near a coast a real sea is younger and smaller (JONSWAP needs the
  fetch, the distance up the wind to the shore; the shard could compute it from the coast mask and the wind).
- **No swell:** waves from far storms come with the weather step.
- **The chain's wind is weak** (S2's reading: 2.9 m/s at the median), so most of today's seas are small.

**Example.** Off the coast stand under the 1.9 m/s breeze the sea shows ripples a few metres long running with the wind;
far out, where the trade blows 8 m/s, swells of about 50 m cross the glint; from 30 km the same sea is a soft sheen whose
brightness does not change as the eye climbs.

## 6. Built (2026-10-18) — what changed from §2–§3 on measurement

1. **48 directions a band, not 16.** With 16, two or three directions carried most of a band's spread, and their long
   straight crests crossed as a net in the sun's path (the picture at 30 m). With 48 the crests are short, as the
   continuous spread makes them.
2. **The spread is read under the eye, once a frame, on the CPU** (`WaveBands::spread`): each band keeps a list of the
   directions over a thousandth of its spread, the strongest first, with their shares. MEASURED: the spread at every
   pixel cost +57 ms a frame at 100 m (16 directions, before the next cure); the waves a pixel can show lie within a few
   kilometres of the eye, where the wind and the sea's plane do not change. Each place keeps its own energy (its own
   wind's speed and depth).
3. **The profiles hold the slope alone**, four samples a row (the normals read nothing else; the height returns with S5).
4. **The cost now:** the CPU block 0.47 ms a frame for each sea realm; the shader within the noise of a loaded machine
   (32.6 ms a frame with the waves at 100 m, 30–38 ms without them in the same hour).

5. **Two defects found by the open-source study (2026-10-18, ruling B111) and cured:**
   - **The energy was 1.69 times too large.** §1 wrote Pierson & Moskowitz's β = 0.74 with the PEAK `ω_p = 0.855 g/U`; their
     β = 0.74 goes with `ω₀ = g/U₁₉.₅`. At the peak the law reads `exp(−(5/4)(ω_p/ω)⁴)` (`U₁₉.₅ ≈ 1.026 U₁₀`), so
     `WAVE_BETA` is 5/4 now. At the coast stand's 1.9 m/s, Hs 0.105 m → 0.081 m (the law's `0.21 U²/g` = 0.077 m).
   - **A whole band left the drawn normal at one distance** (`λ_min < 2 × footprint`): a step at one footprint, against
     W16 and B16. Now every band is weighed by THE PIXEL'S OWN FILTER, the footprint's box by its second moment:
     `H² = exp(−(k·foot)²/12)` of its slope variance in the drawn normal, the rest in the glint's roughness — one
     continuous rule. A band whose share is under 10⁻⁴ (its slopes a hundredth of their size) costs no reads.
