# The sea's look — a short design (ruling B106; the study of 2026-10-17)

> Two read-only studies stand under it (outside the repo): the open-source code — bevy-aqua (MIT OR Apache-2.0) and
> GodotOceanWaves (MIT), with Crest's methods inside bevy-aqua (MIT) — and our client's water path. Nothing from them is
> built yet. Written in ASD-STE100.

## 1. What we take, and what we leave

| Method | Source | Take? | Why |
|---|---|---|---|
| The Atlas BSDF (GGX, exact Smith), Schlick F0 0.0204, sun glint width `sqrt(α² + σ²)` | bevy-aqua `optics.wgsl` | YES | Physical; reads no triangle |
| Beer–Lambert absorption + single scatter by the water's depth | bevy-aqua `medium.wgsl` | YES | The water's clarity (owed since W18) |
| The sky's reflection | our own engine light probe (the atmosphere's environment map) | YES | Already on the camera |
| TMA spectrum with Hasselmann spread, finite-depth dispersion | GodotOceanWaves `spectrum_compute.glsl` | YES | The waves from wind, fetch and the solved depth |
| FFT cascades (exclusive bands, packed derivatives, E[g²] in a channel, moment mips) | bevy-aqua `fft_*.wgsl`, `displace.wgsl` | YES | Normals at every distance from one field |
| LEADR roughness: the slopes a pixel cannot show become roughness | bevy-aqua `optics.wgsl` L27–97 | YES | The cure for distance shimmer; no rung in it |
| Foam: Jacobian whitecaps, a persistent field, shore foam by depth | bevy-aqua `foam.wgsl` | YES (later slice) | Foam coverage goes into albedo and roughness by the footprint |
| Motion: the same anchor at the previous time | bevy-aqua `motion_prepass.wgsl` | YES (with displacement) | The reconstruction needs it |
| 0.8 m RMS renormalisation, the fixed 150 km/h table, turbulence 0.145, roughness cap 0.28 | bevy-aqua | NO | Tuning knobs and a cap (B23, B38) |
| Fades by distance | GodotOceanWaves, bevy-aqua's foam | NO | They change the look with distance (B16, W16) |
| Flat bodies with one level, a bed map in world XZ, planar mirrors, the Crest ring mesh | bevy-aqua | NO | Flat plane only; the server's water map says where water stands |
| GPU wave queries for buoyancy | bevy-aqua | NO | The server decides; the GPU's floats differ (F4) |
| Crest's textures (WaveNormals.png, Foam2.png) | bevy-aqua | NO | Not ours to ship |
| Godot's subsurface colour terms | GodotOceanWaves | LATER | Art terms: the owner's art discussion (B36 Q14) |

## 2. The laws every slice keeps

1. **Where water stands is the server's** (B105 Q2, Q10; SL10): today the sea's level and the coast mask (land, sea, pocket,
   lake for each 8 km node); later the water map in each tile. The shader never decides it.
2. **One lattice** (W18): the water triangle is the ground triangle's own directions at the water's radius, so the ground
   crosses the water only at the shore, at every rung.
3. **One shape at every distance, no flicker** (W16, B16, B22, B27's eye rule): no shading term reads a triangle or a rung;
   wave energy too fine for the pixel becomes roughness (LEADR), never a fade.
4. **The waves are look only** until a law both hosts compute alike (W3, the ocean's 8o; F4).
5. **No knob, no cap** (B23, B38): the waves' height comes from the spectrum's wind, fetch and depth; the colour comes from
   the medium's absorption and scattering.
6. **60 frames a second at full screen** (B22 item 2); the cost of each slice is measured.

## 3. The slices

- **S1 — THE SEA ON THE GROUND'S LATTICE, PLAIN.** Ship the coast mask and the pyramid's water words again (the data class
  W9/W16 approved; B105 Q10's far map; it also cures the registered dry-planet snow line). Port the sheet from git
  (`ba2496b~1`: `position::water_sheet_sided`, `chunks::water_sheet`, `terrain::water_mesh_of`) to today's vertex format:
  the sea's level only, a vertex's side from its columns' heights and the mask's word (a pocket stays dry), the one-step
  hold, the `hide` bound, the deep floor cut; no flat block quads. A plain opaque material (deep colour by depth, the
  sky's reflection, the haze). Measure with the restored benches (`sheet_poke`, `mask_at_pixel`, the teeth and edge
  steps) and the frame cost. Then the owner's look.
  *Built 2026-10-18* (the status file's entry): the sheet on the ground's own triangles, the deep floor cut only under a
  drawn sheet, one opaque colour; measured headless from 100 m to 3 000 km; the owner's look and W18's benches owed.
- **S2 — THE LIGHT.** The Atlas BSDF, Fresnel, the sky, Beer–Lambert by depth, the sun glint from the total slope variance
  (the orbit look). The owner's look.
- **S3 — THE WAVES AS NORMALS.** The spectrum and the FFT cascades on the GPU (4–5 × 256²), the domain on the sphere (Q2),
  the filtered derivative mips and LEADR; the wind (Q1). The owner's look.
- **S4 — THE FOAM.** Whitecaps and shore foam.
- **S5 — NEAR DISPLACEMENT** (Q3), with its exact motion.
- **The lakes and the rivers** join when slices W1–W2 deliver the water map; the same material reads their depth.

**Size (the study's estimate for the whole set):** about 2 400 lines of WGSL and 2 500 of Rust. S1 is mostly the port of
the old sheet (about 550 lines from git) and the data path.

## 4. The questions for the owner (recommendation first)

- **Q1 — THE WIND.** The spectrum needs a wind (speed, direction) for each place; the weather list does not exist yet.
  *Recommended:* ship the seasons' chain's surface wind for each climate cell with the artifact (34 656 cells on the
  home planet, a few bytes each, about 0.1 MB; an SL6 ask: the client cannot run the chain); the weather list modulates it
  later. *Other:* wait for the weather step (no waves until then).
- **Q2 — THE WAVE DOMAIN ON A SPHERE.** *Recommended:* triplanar sampling in planet-fixed coordinates (the period taken in
  f64 on the CPU): no seam, no re-anchor pop; the price is three samples a cascade. *Others:* a tangent plane re-anchored
  near the eye (a pop at each re-anchor, so a crossfade in time); the cube faces with blend bands (a seam risk, B81).
- **Q3 — NEAR DISPLACEMENT.** True 3D waves within about 1–2 km need a dense water grid near the eye — a second lattice
  there (amends W18 near the eye; the shore read from the 16 m ground). *Recommended:* LATER, after the look of S2–S3;
  the normals carry the waves until then.
- **Q4 — THE NOTICES.** *Recommended:* a `THIRD_PARTY_NOTICES.md` in the repo naming bevy-aqua (MIT / Apache-2.0), Crest
  (MIT) and GodotOceanWaves (MIT) for the methods and the code we port; their textures are not shipped.

**Example.** From the coast stand of B103 the walker sees the sea at the server's level: near the beach the sand shows
through clear water (Beer–Lambert by the depth), farther out the swell's normals catch the sky and the sun; from orbit the
same sea is one calm sheen whose glint has the width of the waves' total slope — and the coastline does not move as the
eye climbs.

## 5. The owner's answers (ruling B107, 2026-10-17)

- **Q1 — THE CLIMATE'S WIND** (SL6 YES): the seasons' chain's surface wind for each climate cell ships with the artifact;
  the weather list modulates it later.
- **Q2 — "the best quality, no seams, only physics, only laws produced by the planet".** Each of the three offered domains
  fails it (triplanar: a crossed sea in its blend zones; a tangent plane: a jump at each re-anchor; cube faces: seams). THE
  DOMAIN IS NO CHART: the waves are a sum of PLANE WAVES IN THE PLANET'S OWN 3D SPACE — each component a 1D profile of
  `x · d` along a fixed 3D direction `d` (the profile buffers of Jeschke, Skřivan, Müller-Fischer, Chentanez, Macklin &
  Wojtan 2018, *Water Surface Wavelets*, ACM ToG 37(4)), so every component exists on the whole sphere with no chart, no
  blend and no seam. At each place the local wind's directional spectrum (TMA with Hasselmann's spread, the depth's
  dispersion) weights each direction by how it lies in the local tangent plane; the bands a pixel cannot resolve become
  roughness by the same spectrum (LEADR). The phase `x · d` is taken modulo each band's period in f64 at the camera and in
  f32 beside it. Its price, measured in S3: about 16–32 directions times a few bands a pixel.
- **Q3 — NEAR DISPLACEMENT LATER**, after the owner's look at S2–S3.
- **Q4 — `THIRD_PARTY_NOTICES.md`** names bevy-aqua (MIT / Apache-2.0), Crest (MIT) and GodotOceanWaves (MIT); their
  textures are not shipped.

