# Slice S2 — the sea's light: a short design (rulings B106, B107, B108)

> Written in ASD-STE100. The parent design: `docs/investigation/2026-10-17/sea_look_design.md` (§3, S2). Nothing here is
> built yet.

## 1. What S2 draws, each by a published law

| Part | The law | Its inputs |
|---|---|---|
| The reflection's strength by the angle | Fresnel (Schlick's form), water's index 1.333 → F0 = 0.0204 | none |
| The sky in the water | the atmosphere's own environment light (already on the camera, B25) | none |
| The sun's glint and its width | GGX with exact Smith (the Atlas BSDF, bevy-aqua `optics.wgsl`); the width from the surface's slope variance, Cox & Munk (1954): σ² = 0.003 + 0.00512 · U (U the wind speed, m/s) | THE WIND'S SPEED |
| The colour by depth | Beer–Lambert absorption of pure water (Pope & Fry 1997) and single scatter (Morel 1974), along the light's path down to the floor and up to the eye | THE DEPTH at each point |

No knob and no cap (B23, B38): the colour comes from the medium's coefficients, the glint from the wind.

## 2. Facts found (2026-10-18)

1. **The climate keeps the wind's direction only.** `vd_terrain::climate::Climate::upwind` is a stencil slot per node; the
   seasons chain computes each climate cell's surface wind (speed and direction) inside the solve and drops it.
2. **Bevy 0.18's transmission** (screen-space refraction, `specular_transmission` with `thickness` and Beer–Lambert
   `attenuation_*`) copies the frame and draws the water in its own later pass. Its price here: the floor must be drawn
   under all water (no cut), and the water leaves the depth prepass that our reconstruction (B24, B90) and the haze's
   march (B25) read — both would need a change.
3. **The water point and the ground vertex share one lattice** (W18): each water point can carry its own depth (its
   radius less its ground vertex's) and its floor's paint (the ground vertex's surface word), at no extra search.
4. **The cost of one seasons-chain run** on a home-planet ground: about 10 s on 12 cores (B101's numbers: 63 min more for
   362 more runs).

## 3. The questions (recommendation first)

- **Q1 — HOW THE FLOOR SHOWS THROUGH THE WATER.**
  *(a) Recommended:* an opaque sheet whose colour is the medium's law over the depth each point carries — the floor's
  paint lit by the sun through the water, attenuated down and up by Beer–Lambert, plus the water's single scatter. The
  floor's relief shows as the depth's change across the lattice (1–2 m points near the eye). Cost: a few shader lines,
  one more vertex word. The deep floor stays cut where the water lets less than one 8-bit level of the floor through in
  its clearest colour (B27's eye rule), never by a chosen depth.
  *(b)* Bevy's screen-space transmission: the floor's true shape refracted near the eye. Price: §2 item 2.
  *(c)* (a) now, (b) later near the eye only, if the look asks for it.
- **Q2 — WHERE THE WIND COMES FROM** (SL6 is already YES by B107 Q1).
  *(a) Recommended:* the planet's shard runs the seasons chain once on its stored ground when it loads (about 10 s, the
  same function on the same words, so the same bytes), keeps the cells' surface wind in its own store under the
  artifact's digest, and ships it as one bulk part (34 656 cells, about 0.1 MB). No world change, no re-solve.
  *(b)* The solve keeps the wind in the artifact: the world moves once (a new artifact key; the home planet solved again,
  about 1 h on 12 cores, every other body at its next first visit).
  *(c)* S2 without wind: the calm sea (Cox–Munk's own floor, σ² = 0.003); the wind joins with S3.

**Example.** At the coast stand at noon the walker sees, through 2 m of water, the sand of the bay a little green-blue;
50 m out the floor fades into a deep blue that holds the sky's reflection at low angles; from orbit the sun's glint on
the home planet's sea is a soft oval whose width follows the trade wind's speed from the climate.

## 4. Built (2026-10-18, ruling B109)

Both answers as recommended: Q1 (a) and Q2 (a). One change from §3 Q1's words: the deep floor's cut stays the
extractor's step, since under an opaque sheet the floor is never seen and its colour comes in the sheet's own law. The
status file's entry of 2026-10-18 holds the numbers and the pictures.

