# 01 — THE TARGET AND THE REFERENCE: what a believable vista is, in numbers

**Date:** 2026-09-08. **Revision 3** (after two rounds of refutation; every finding of both rounds is
answered in §14).
**Domain:** the TARGET of the landform work. This document says what the picture must contain and how a
gate measures it. It does not design the recipe. The mechanism documents of this investigation do that.
**Status:** an investigation report for the owner. It decides nothing by itself.
**The owner's words that started it:** *"Make sure that we reach that quality on the picture for
earth-like planets (biomes can be different of course, should be dependent on the planet position, spin,
trajectory, size and gravity, etc.). It should be very believable, as we also should simulate the
weather."* And of our own pictures: *"no orienters and no details at all … The only thing I'm worried
about is that the surface will not be interesting enough."*

**Binding law read first:** CLAUDE.md (HR1–HR6, SL1–SL10), `owner_decisions_2026-09-07_voxels.md`
(V1 SL10, V2.1–V2.9, V4, V5, V6 Parts A–D, V8–V12), `owner_decisions_2026-08-27_seed_and_secrecy.md`,
`owner_decisions_2026-09-02_reach.md`, SL8 (a seam is a defect).

**What changed in revision 2.** Revision 1 read the owner's complaint off three screenshots. Two refuters
showed that those screenshots carry three registered defects at once, so they cannot carry a diagnosis
(§1.1). Revision 2 therefore measures the SHIPPED FIELD instead of the picture: a ray-marched skyline, a
slope distribution, a departure-from-a-plane and a hypsometry, all computed from the crate's own height
arithmetic (§1.3–§1.6). The diagnosis survives, and it is now a measurement that could have failed.
Revision 2 also withdraws revision 1's relief-law recommendation (§4.2), corrects the cost case (§8), and
stops claiming that the ladder memory obviously fits (§5.6).

**What changed in revision 3.** Two refuters read revision 2. They found that revision 2's own instrument
was not reproducible: the skyline camera had no address, the march stopped at 90 km while the dominant
wave is 400 km long, and the headline number disagreed with itself between two sections. Revision 3
answers that with a MEASUREMENT, not a promise:

- **The camera is now a RULE, and the rule names four addresses** (§1.4, §7.1). The ridge station is *the
  highest column inside a named 100 km patch, on a named 101 × 101 grid*. Four patches give four stations,
  each printed as a face and a cell pair.
- **The march is converged, and the truncation was real.** At every station the march runs to 90 km,
  180 km, 360 km and 540 km. Refutation B was right that 90 km truncates: at station S2-c the largest rise
  grew **from 0.0489° to 0.0636°** — 30 % — when the march grew to 180 km, and at S2-a the skyline's low
  end moved 0.79°. Every station is stable from 360 km on. So the instrument changed; **the conclusion did
  not**.
- **The headline is restated honestly.** Across the four stations the largest rise over the local trend
  runs **0.054°–0.067°**, the break count is **zero at every threshold at or above 0.10°**, and at 0.05°
  the whole 360° horizon holds one to three breaks. Revision 2's "0.040°, zero at every threshold down to
  0.05°" was one unnamed camera at a truncated range, and it is withdrawn.
- **The maximum slope is a lower bound, not a property.** A two-minute hill climb from the twenty steepest
  of 30 000 columns reaches **11.15°**, against the 8.78° a sample of 30 000 gives (REPLAYED; refutation A
  measured 11.17° independently). The "no cliff" conclusion now rests on the analytic estimate of 28.1°
  with its own unknown attached (U2), never on a sample extremum.
- **Two recommendations are withdrawn.** The strata are NOT an unused asset: they drape parallel to the
  surface and put the hard rock at the BOTTOM, so they cannot make a mesa, a hoodoo or a bedded cliff
  (§3.5, D12). And β leaves the gate list, because the β = 2 variant of our own field is the unwalkable
  gravel heap §1.6 measured (§6.5, D2).
- **Three things nobody owned are now named:** the LIGHT the pictures are shot in (§1.1, §7.2, D20), WATER
  above the sea for a river to run on (§3.5, D19), and how a red gate LANDS without turning `just gate`
  red for every unrelated change (§6.11, D2).

Every number below carries a mark:
- **MEASURED** — a program in this repository produced it, and the program is named.
- **REPLAYED** — I re-implemented the crate's own arithmetic in Python from the source and ran it. The
  replay reproduces five values that the crate's own bench MEASURED and that ruling V9 records: the relief
  14 305 m, the sea 5 297 m under the ladder radius, 12 rungs, 14 octaves, and "about one column in a
  hundred under the sea" (`slice_05_generator.md:367-394`). Each of those checks could have failed. None
  did. A REPLAYED statistic is a measurement OF THE REPLAY; the crate-side bench that makes it MEASURED is
  owed as U1 (§10).
- **PUBLISHED** — a standard reference value for Earth or for a solar-system body, quoted for
  calibration. **No PUBLISHED number may enter a recipe before a named source is attached to it.**
- **ESTIMATED** — arithmetic on a measured number, with the arithmetic shown.
- **UNMEASURED** — not known. The measurement that would settle it is stated.

---

## 0. The recommendation, in short

**The problem is not the amount of relief. It is that the relief has no structure below the wavelength of
a province.**

The home planet carries 14 304.9 m of amplitude (REPLAYED, §1.2), and its realised surface spans
17.0 km from its lowest sampled point to its highest (REPLAYED, §1.3). Earth's dry LAND spans about
9.3 km, from Everest to the Dead Sea shore (PUBLISHED). Only 0.99 % of our planet is under its sea, so
17.0 km is almost all land: **on the same ruler our planet carries nearly twice Earth's land relief.**
(Revision 2 printed Earth's 19.9 km Everest-to-Challenger-Deep span beside it. Over half of that span is
sea floor, so the two numbers were measured with different rulers. Refutation B was right, and the
correction makes the point stronger, not weaker.) The relief is there. The picture is still empty, and
four measurements say why:

- **The skyline holds no break the eye can name.** From four named ridge stations 373 m over the local
  surface, ray-marched through the crate's own height field over 360 bearings and out to 540 km, the
  skyline's largest rise over its own local trend runs **0.054°–0.067°** (REPLAYED, §1.4 and §6.1). The
  break count is **zero at every threshold at or above 0.10°**; at 0.05° the whole 360° horizon holds one
  to three. For scale, a full moon is 0.52° across: the tallest feature on this planet's horizon is an
  eighth of a moon's width.
- **The ground under the player is a plane.** Inside 50 m of a column, after the local tilt is removed,
  the surface departs from a flat tilted plane by **0.26 m rms and 0.47 m at most** (REPLAYED, the median
  over 20 000 columns). Inside 1 km it departs by 7.0 m. ("rms" is the root-mean-square: the typical size
  of the leftovers, counting a bump and a dip alike.)
- **No slope is steep.** Over 200 000 columns the slope's median is **1.36°** and its p99 is 5.10°, and
  the numbers are the same at a 1 m, a 10 m, a 100 m and a 1 000 m baseline (REPLAYED, §1.4). A field with
  the same slope at every baseline is smooth by definition. **The steepest ground I could FIND anywhere,
  by hill-climbing out of the steepest samples, is 11.15°** — a lower bound on the field's true maximum,
  not the maximum. The analytic estimate bounds one octave sum at about 28°, which is still under the 45°
  a cliff needs (§1.4, U2).
- **The height distribution is one hump.** Skew −0.03, kurtosis 2.80, where a Gaussian gives 0 and 3
  (REPLAYED, §1.3). There is no shelf, no plain and no mountain tail. The cause is NOT that fourteen
  noises average out: §1.2 measures 78.1 % of the amplitude in the two coarsest octaves, so there are
  about two effective terms, not fourteen, and the measured kurtosis of 2.80 is BELOW 3, which is what a
  few bounded terms give and not what a many-term average gives. Refutation B was right and the true
  reason is more useful: **the height distribution is the distribution of one long wave.** Adding fine
  octaves will never give it a second hump.

So the owner saw what the arithmetic says he must see. **"No orienters" is 0.06° of skyline relief.
"No details" is 26 cm of shape inside 50 m.**

**And a fifth cause is not the terrain at all: the LIGHT.** MEASURED, `crates/bins/tests/terrain_pictures.rs:43-45,458-462`: the picture harness puts the star
`SUN_ELEVATION_DEG = 25.0` degrees over the horizon and then points the camera's nose toward the star's
own azimuth. So every picture the owner judged was shot into the light, with the visible slopes facing
away from it. Relief reads through shading, and a 1.36° slope under a light that casts no cross-shadow
reads as nothing at all. The light is a typed constant with no owner and no line in the overlay (§7.2,
D20).

**A rougher noise alone does not fix it, and that is now measured too** (§1.6). Setting the roughness to
Earth's spectral exponent, with the relief unchanged, makes 30 % of the planet steeper than 45° — an
unwalkable gravel heap — and still buys only **0.8 skyline breaks per 60°**. Buying five breaks out of
pure noise needs a field that puts 92 % of the surface over 45°. Roughness is not landform.

**The target this document proposes**, in one sentence: an Earth-like planet must show, from a viewpoint
373 m above the plain, at least fifty kilometres of ground carrying several silhouette breaks, with
a slope distribution, a hypsometry, a spectrum and a slope–area law inside stated bands; and every band
must be a FUNCTION of the body's own physical facts — its radius, its gravity, its insolation, its spin
and its atmosphere — wherever a law exists to derive it (§6, and the honest gap in §6.9).

**The six things I recommend the owner adopt** (the full table is §11):

1. **A named picture set, with a rule for every station and a fixed light.** Seven camera stations on the
   home planet. Each station's address is a CONSEQUENCE of a printed rule, never a choice — the ridge
   station is the highest column inside a named 100 km patch (§7.1). Each picture carries a scale
   reference and a readout that states the RUNG, the draw radius and **the star's elevation and azimuth**.
   A vista becomes a regression test, not taste. Four of the seven cannot be shot until `D-TERRAIN-3`
   closes (§7.4).
2. **Eight acceptance proxies**, three as numeric gates and five as printed reports (§6), plus a stated
   way for a RED gate to land without blocking every unrelated merge (§6.11). The gates are cheap, they
   read the crate's own `height_m` with no renderer in the path, and they are RED today.
3. **The skyline-break count** as the headline number, computed by ray-marching the shipped height field
   — not by counting pixels in a rendered frame. A picture gate would also measure the renderer, the
   exposure and the star field, which is exactly the confusion that made revision 1 wrong (§1.1, §6.1).
   **Its band is not yet honest**: the reference's "8–12 breaks" was counted by eye off a screenshot and
   the 0 was computed by a stated rule, so the two are different quantities. Until somebody runs the same
   rule over the reference (§10 U16), P1 lands as a RATCHET — it may not get worse — never as a fixed
   floor (§6.1, D2).
4. **The physical-facts rule**: every band that HAS a law is a function of the body's facts. This is the
   owner's "dependent on the planet position, spin, trajectory, size and gravity" written as arithmetic.
   It needs facts the generator crate cannot reach today (§4.6), and two facts no body draws at all —
   **spin and obliquity** (§4.5). Both are open questions, not decisions I may take. For β, for the slope
   distribution and for the peak density, **no published law derives the band from a body's facts**, and
   this document says so rather than inventing one (§6.9).
5. **The relief law is unanchored, and the fix is a CEILING, not a new formula.** The code gives
   relief ∝ radius, clamped, times a seed factor. Measured against a crustal-strength ceiling
   `h ≤ σ/(ρ_c g)` calibrated on Everest, the code uses **0.02 % of the ceiling on a 100 km body and up to
   600 % of it on a super-Earth** (REPLAYED, §4.2). Revision 1 recommended moving to relief ∝ 1/gravity
   and called it a one-way door. **I withdraw that.** Both refuters showed the six-body table carries no
   such relation (the log–log slope of relief against radius is −0.03, correlation −0.09), and a 1/g law
   calibrated on Earth gives Vesta three times its own radius of relief. §4.2 now proposes the ceiling as
   a BOUND and leaves the value's law open.
6. **Bedding is NEW WORK, and the cube corners are already decided.** Revision 2 called the 19 strata
   "the cheapest believability, already built and unused". **I withdraw that.** Both refuters read
   `strata.rs:187-210` and they are right: the table selects a substance **by depth under the surface**,
   so every layer DRAPES over the hill instead of lying flat, and the order is soft over hard — topsoil,
   subsoil, sediment, then bedrock forever. A mesa and a hoodoo are a HARD cap over a SOFT layer, so this
   table makes both impossible at any erosion rule, and a cliff can expose at most four substances because
   a body draws one sediment and one bedrock (`body.rs:194-202`). Bedding at a constant RADIUS, with a
   hardness per bed, is a new mechanism with its own cost (§3.5, D12). Ruling V6 A5 does bind and is
   carried: **the generator always places a landform at all eight cube corners** (§6.8, D11).

**What the target costs.** The owner's budget is 8 ms of worker time per chunk, and the gate holds the
COSTLIEST named chunk to it (`terrain_cost.rs:345,357`). That chunk is the cave-dense seam chunk at
**6.11 ms** (MEASURED, slice 6), and the owner set 8 ms because of it (ruling V10). The headroom is
therefore **1.89 ms**, not revision 1's 4.7 ms. At the measured price of 13.2 ns per octave-column that
buys about **35 more field evaluations per column** — 14 today, about 49 after — which is 3.5× the field
work, not 4.7× (§8.2). **Most of those 35 are already spoken for by prices nobody has measured:** a
three-dimensional domain warp costs three noise evaluations per warped octave, not one (refutation A was
right), and the analytic derivative's price is this document's own U7. The ladder memory of §5.6 spends
between 0.24 ms and 2.1 ms on top, and the spread is the halo, which nobody has measured (§8.3). **And
one chunk is neither a frame nor a boot:** a 50 km vista holds between 215 and 3 448 chunks in a 60°
field, which is 0.09 GB to 1.4 GB of mesh, and the boot self-check already costs 13.5 ms per body before
a single chunk is trusted (§8.4).

**What the target forces.** No sum of noise octaves can produce a slope–area law or a valley spacing.
Both are consequences of water flowing downhill over a whole landscape, and a per-point noise knows
nothing about downhill. §1.6 measures that: raising the roughness buys steepness and buys almost no
skyline. So the target implies a mechanism with a memory of the surrounding landscape. §5.6 shows that the
LADDER is the natural shape for that memory — and states the two hard parts revision 1 skipped: the coarse
rung is 10 584 chunks and needs its own bounded stencil, and a channel must cross a chunk edge and a
cube-face seam without stopping. Those are the design, and they are not solved here.

---

## 1. What our own world measures today

### 1.1 The three pictures, and why they cannot carry a diagnosis

The three pictures at `docs/investigation/2026-09-07/pictures/` show the home planet from the dev client.
Revision 1 read them as "a smooth brown arc against a black sky, therefore the recipe is too smooth". That
reading is not safe. The tree records three separate causes inside those frames (MEASURED,
`slice_07_client_link.md:237,279-287`):

| Picture | Rung | Eye | What was drawn | The registered defect in it |
|---|---|---|---|---|
| `ground.png` | 0 | 1.8 m | 13 × 13 columns = 806 m of ground | the far edge is a PATCH EDGE 403 m away, not a horizon — `D-TERRAIN-3` 🟥 |
| `hill.png` | 3 | 300 m | the three finest octaves already dropped | the same one-rung limit |
| `aloft.png` | 9 | 60 km | 25 × 25 columns of 32 km = 800 km | nine octaves dropped, 322 m of amplitude removed |
| all three | — | — | a black sky | the star field held **0 stars** at the shutter, cause UNMEASURED — a separate defect, not terrain |
| all three | — | — | the light | the star stands **25°** up and the camera looks TOWARD it — the worst light for reading relief (below) |

`crates/client-render/src/terrain.rs:54` also sets `DEFAULT_RADIUS = 2`, so a session that does not set
`VD_TERRAIN_RADIUS` draws 5 × 5 chunks = 310 m of ground.

**The light is the fourth cause, and revision 2 missed it.** MEASURED, `crates/bins/tests/terrain_pictures.rs`:
line 45 sets `SUN_ELEVATION_DEG = 25.0`; lines 451-455 place the standing spot on the radial where the
star stands exactly that high; lines 458-462 point the camera's nose *toward the star's own azimuth*,
tilted below level. So the sun sits in front of the camera and above the frame. Every slope facing the
camera faces AWAY from the light, so it carries no cross-shadow and no shading gradient. A photographer
would call this shooting into the sun. Relief reads through shading: at the measured median slope of
1.36° a ridge under a high or a head-on light is invisible, and the same ridge under a low cross light
throws a shadow a kilometre long and reads at once. **Refutation B is right that the picture set has no
light rule, and wrong that nothing sets the light** — a typed constant sets it, and it happens to set the
worst one. Two shots of identical ground under different light are not comparable, which is exactly what a
regression set must be (§7.2, D20).

**The consequence.** Inside 403 m the recipe can only put a few metres of shape, so "no details" in
`ground.png` is partly the recipe, partly a 403 m patch, and partly a head-on light. A reader who acted on
revision 1 might have paid for a world epoch to fix a frame that slice 8 and a light angle fix for free.

**What this revision does instead.** Every diagnostic number in §1.3–§1.6 comes from the height field
itself, with no renderer, no rung flag and no star field in the path. The pictures return in §7 as an
ILLUSTRATION and as a regression set, never as the instrument. The control is still owed: re-shoot station
1 and station 2 at the widest draw radius the lane serves, and at the rungs slice 8 chooses, before any
recipe conclusion is drawn from a picture again (§10 U4).

```text
   what the pictures show                    what the reference shows
   ─────────────────────────                 ────────────────────────
   ┌───────────────────────────┐             ┌──────────────────────────┐
   │        black sky          │             │ haze, cloud, sun         │
   │   (0 stars: a defect)     │             │   ▁▃▅█▇▅▃▁ snow ridges   │
   │ ─────────────────────     │             │  ▂▄█▆▄ mesas, cliffs     │
   │      flat brown           │             │ ▁▂▃ forest mass, river   │
   │ (ends at 403 m: a patch   │             │ ▃▅ rock, road, figure    │
   │  edge, D-TERRAIN-3)       │             │                          │
   └───────────────────────────┘             └──────────────────────────┘
   0 silhouette breaks                        ~8–12 silhouette breaks
   1 depth plane                              4–5 depth planes
   0 scale references                         a character, trees, a tower
```

### 1.2 The home planet's own numbers

REPLAYED. I re-implemented `SplitMix64`, `child_seed`, `Ladder::for_radius` and
`BodyDefinition::from_seed` (`crates/seed/src/rng.rs:22-28,70-74`; `crates/seed/src/ladder.rs:50-98`;
`crates/terrain/src/body.rs:140-200`) in Python and ran them on the two literals in
`crates/terrain/src/home.rs:16-23`.

**The checks that could have failed, counted honestly.** Ruling V9 records the bench's own MEASURED
values (`slice_05_generator.md:367-394`): relief 14 305 m, 12 rungs, 14 octaves, ladder radius
3 350 759 m, and "the home planet's sea stands 5 297 m under the ladder radius and covers about one column
in a hundred". My replay produces 14 304.887 m, 12 rungs, 14 octaves, 3 350 759.045 m, a sea offset of
−5 297 m, and 0.99 % of 200 000 sampled columns under the sea.

Revision 2 called that five independent checks. Refutation A is right that it is **three**, and the
correction matters:

| Pin | Could it have failed? |
|---|---|
| The ladder radius 3 350 759.045 m | **no** — `Ladder::for_radius` reads the radius alone (`ladder.rs:73-99`) |
| 12 rungs | **no** — the same function |
| 14 octaves | **no** — `long_wave_m` saturates at `LONG_WAVE_CAP_M` for every seed on this body, so the count follows from the radius |
| **The relief, 14 304.887 m** | **yes** — it is three `SplitMix64` draws deep in the seed tree |
| **The sea offset, −5 297 m** | **yes** — a fourth draw under its own salt |
| **The 1-in-100 share under the sea** | **yes**, and it is the only one that exercises `noise3` — but it is a coarse statistic |

**The bigger half of the finding, which I accept in full.** Every headline of this document — the skyline,
the plane departure, the slope, β, the hypsometry — is a function of `noise3`, and **no listed pin tests
`noise3` against the crate.** The crate offers an exact one: `golden_self_check` prints the home body's
digest, `0x9331e1fdfd902272` (`slice_05_generator.md:386`). Reproducing that digest in the replay, or
printing one named column's surface radius from `terrain_cost` and matching it, would make the noise a
measurement that could have failed. It is owed as **U13**, and until it is paid every REPLAYED number here
rests on two independent re-implementations agreeing (mine and refutation A's), which is evidence and not
a pin. (Revision 1 cited `home.rs`'s own unit test instead; both refuters showed that test asserts only
the three radius-only quantities above.)

| Fact | Value | Where it comes from |
|---|---|---|
| Universe seed | 2298 | `crates/terrain/src/home.rs:19` |
| Home planet realm seed | 7 701 581 858 760 374 086 | `home.rs:21` |
| Look radius | 3 351 154.238 m | `home.rs:23` (bits `0x41499139_1e692dfa`) |
| Ladder radius | 3 350 759.045 m | `Ladder::for_radius`, `crates/seed/src/ladder.rs:73,103` |
| Cells along a face edge (rung 0) | 5 263 360 | same |
| Rungs | 12 (rung 0 = 1 m cells … rung 11 = 2 048 m cells) | same |
| Surface area | 141.09 million km² (Earth: 510.06, PUBLISHED) | 4πR² |
| **Relief (the amplitude sum)** | **14 304.9 m** | `body.rs:147-149,182` |
| Sea radius | 3 345 462 m (5 297 m below the ladder radius) | `body.rs:186-190` |
| Octaves | 14 | `body.rs:158-184` |
| Roughness `k_rough` | 0.4683 | `body.rs:155` |
| Coarsest wavelength | 400 000 m — **the constant `LONG_WAVE_CAP_M`, not a draw** | `body.rs:24,151-153` |

**The relief is not the clamp.** `body.rs:147-149` reads:

```rust
let relief_share = radius_m * Gf::from_f64(0.004);
let relief_cap   = relief_share.clamp(Gf::from_i64(200), Gf::from_i64(12_000));
let relief_m     = relief_cap * (Gf::HALF + draw_unit(&mut octave_rng));   // × [0.5, 1.5)
```

The seed factor multiplies AFTER the clamp. So the 12 km cap and the 200 m floor bound the PRE-FACTOR,
never the relief: the smallest relief any body can draw is 100 m, the largest is 18 000 m, and the home
planet's 14 304.9 m is 0.427 % of its radius. Revision 1's §4.2 printed "12.0 km (capped), 0.358 %" for
the home planet and so contradicted its own §1.2. Both refuters caught it. It is corrected here and in
§4.2.

**The coarsest wave is a typed constant on this body.** `radius × long_share` is at least 837 690 m, which
is above `LONG_WAVE_CAP_M = 400 000`, so the clamp saturates for every seed. The headline fact — that most
of the amplitude sits in the two longest waves — is therefore caused by a `u64` in the source, and §9's
magic-number row now names it.

The octave table, coarsest first (REPLAYED). "Wave" is the wavelength on the surface. "Amp" is the most
that octave may raise or lower the ground.

```text
  oct  wavelength        amplitude    what a player at that scale would call it
  ───  ──────────        ─────────    ─────────────────────────────────────────
   0   400 000 m         7 605.55 m   a continent-wide swell
   1   200 000 m         3 561.97 m   another swell
   2   100 000 m         1 668.21 m   a province
   3    50 000 m           781.28 m   a day's flight
   4    25 000 m           365.91 m   a long walk
   5    12 500 m           171.37 m   an hour's walk        ┐
   6     6 250 m            80.26 m   half an hour          │  the band the
   7     3 125 m            37.59 m   ten minutes           │  eye actually
   8     1 562 m            17.60 m   five minutes          │  reads
   9       781 m             8.24 m   a minute              │
  10       391 m             3.86 m   thirty seconds        │
  11       195 m             1.81 m   a few steps           │
  12        98 m             0.85 m   two steps             │
  13        49 m             0.40 m   one step              ┘
        ────────────────────────────
        sum               14 304.9 m
```

The two coarsest octaves hold **78.1 %** of the sum. Everything from 5 km down holds 70.35 m, which is
0.49 %. Everything from 200 m down holds 3.06 m, which is 0.021 %.

**A caution about the sum.** 14 304.9 m is a BOUND: it assumes every octave's noise reaches ±1 at one
point at once. The gradient noise's own sampled maximum is 0.98 (REPLAYED, 400 000 points), and fourteen
independent octaves rarely align. §1.3 gives the realised number instead.

### 1.3 What the surface really does, measured

All REPLAYED, over 200 000 directions drawn uniformly on the sphere, through the replay of `height_m`.

**The realised shape.**

| Quantity | Value |
|---|---|
| Lowest sampled point, against the ladder radius | −8 579 m |
| Highest sampled point | +8 424 m |
| **Realised span, over the sample** | **17 003 m** — a LOWER bound on the true span, because it is the extreme of a sample (§1.4) |
| Earth's dry-LAND span, the like-for-like ruler | 9 300 m, Everest to the Dead Sea shore (PUBLISHED) |
| Earth's Everest-to-Challenger-Deep span, for reference only | 19 900 m (PUBLISHED); over half of it is sea floor |
| Share of columns under the sea | 0.99 % (ruling V9's bench: "about one column in a hundred") |
| Altitude above the sea: p1 / p50 / p99 | 6 m / 5 312 m / 10 475 m |
| Skew, kurtosis of the altitude | −0.031, 2.795 (a Gaussian gives 0 and 3) |

"p50" is the middle value: half the columns are lower. "p99" is the value only one column in a hundred
beats. "Skew" says whether the hump leans left or right; "kurtosis" says whether the tails are heavy (over
3) or light (under 3).

Three owner-facing facts fall out.

1. **The relief is not the problem, and the comparison must use one ruler.** Our 17 003 m is almost all
   dry land, because only 0.99 % of the planet is under its sea. Against Earth's dry-land span of 9 300 m
   our planet carries **nearly twice** Earth's land relief. Revision 2 compared it with Earth's
   land-plus-ocean span and so understated its own case; refutation B caught it.
2. **The sea is a puddle in a low place, not an ocean basin.** Half the land stands 5.3 km above it. A
   coast is one of the reference's strongest landforms, and this planet has almost none (§12 Q5).
3. **One hump, and the reason is two waves, not fourteen.** Revision 2 blamed the central limit theorem —
   the rule that a sum of many independent bounded terms tends toward a bell curve. Refutation B is right
   that our field is not in that regime: §1.2 measures 78.1 % of the amplitude in the two coarsest
   octaves, so there are about **two** effective terms, and the measured kurtosis of 2.795 is BELOW a
   Gaussian's 3, which is the signature of a few bounded terms and not of many. **The height distribution
   of this planet is the distribution of one long wave.** That is the more useful statement, because it
   says what will not help: no number of fine octaves gives it a shelf. Only a mechanism that separates a
   base level from an uplifted block can.

**What a player crosses.** For each column I put a ring of 24 points around it, fitted the best flat
tilted plane through the ring, and measured what is LEFT.

| Ring radius | The local tilt (median) | Departure from that plane: rms | Departure: max | p99 of the max |
|---|---|---|---|---|
| **50 m** | 2.28° | **0.258 m** | **0.471 m** | 0.99 m |
| 200 m | 2.20° | 1.204 m | 2.264 m | 4.72 m |
| 1 000 m | 2.05° | 7.049 m | 13.232 m | 27.18 m |

**This is "no details", stated exactly.** Inside the first fifty metres of the player's boots the ground
is a tilted plane to within a quarter of a metre. It is not a staircase of 48 m plateaus — the gradient
noise is smooth between its lattice corners, so the ground is a gently curved ramp. (Refutation B is right
that revision 1's "forty-seven metres out of every forty-eight carry no shape" teaches a false picture; it
is deleted.) A ramp with 26 cm of texture on it is what the owner is looking at.

**The horizon.** At eye height 1.7 m the horizon on this planet is 3.38 km (ESTIMATED, `d = √(2Rh)`;
Earth's is 4.65 km). The whole visible world from a standing player is one tilted plane with metres of
texture on it.

### 1.4 No cliff exists, and now it is measured rather than argued

Revision 1 argued from a bound: each octave's slope is at most `4A/L`, the sum is 0.72, so 36°. Both
refuters showed that `4A/L` is a MEAN slope over a quarter wave, not a bound on the derivative, and that
revision 1 then printed "0 cliffs, provably". They were right to object. Two measurements replace the
argument.

**The noise's own steepness.** REPLAYED over 400 000 random points of `noise3` (the exact algorithm of
`crates/terrain/src/noise.rs:88-120`: 16 gradients from {−1, 0, 1}, quintic fade), by central differences:

| Quantity of `\|∇ noise3\|` in lattice units | Value |
|---|---|
| median | 1.14 |
| p99 | 2.16 |
| p99.9 | 2.49 |
| **sampled maximum** | **2.97** |
| sampled maximum of the value itself | 0.98 |

So one octave's steepest slope is `G·A/L` with `G ≈ 2.97`, and `4A/L` is a LOOSER bound with margin. The
sum over the fourteen octaves is `2.97/4 × 0.7204 = 0.535`, which is **28.1°** (ESTIMATED; the true
supremum of `|∇ noise3|` is UNMEASURED, and a sampled maximum is only a lower bound on it — U2).

**The slope the surface actually has.** REPLAYED, 200 000 columns, a random bearing at each, finite
differences at four baselines:

| Baseline | median | p90 | p99 | steepest IN THE SAMPLE | share over 45° | share over 60° |
|---|---|---|---|---|---|---|
| 1 m | 1.36° | 3.29° | 5.10° | 8.74° | **0** | **0** |
| 10 m | 1.36° | 3.28° | 5.09° | 8.79° | 0 | 0 |
| 100 m | 1.32° | 3.19° | 4.97° | 8.24° | 0 | 0 |
| 1 000 m | 1.21° | 2.95° | 4.58° | 8.30° | 0 | 0 |

(The ring fit of §1.3 gives a median tilt of 2.28°, which is the full gradient; a random bearing samples
its cosine, and the median of that is 0.6 of it — 1.37°. The two agree.)

**A sample's steepest column is NOT the field's steepest column, and revision 2 wrote it as if it were.**
Both refuters caught it, and both are right. 200 000 columns is 1.2 × 10⁻⁹ of the body's
6N² = 1.66 × 10¹⁴ rung-0 columns, and the extreme of a sample grows with the sample. Two measurements
make the correction concrete:

| How the steepest column was sought | Result |
|---|---|
| The steepest of 30 000 random columns (REPLAYED) | 8.78° |
| The steepest of 200 000 random columns (REPLAYED, the table above) | 8.74° |
| **A hill climb on the full gradient from the 20 steepest of 30 000**, steps shrinking from 200 m to 5 cm (REPLAYED, two minutes) | **11.15°** |
| Refutation A's own independent hill climb | 11.17° |

So the steepest ground anybody has FOUND is 11.15°, and that is a **lower bound** on the field's maximum,
not the maximum. The same caution applies to §1.3's 17 003 m span, which is also a sample extreme.

**The conclusion survives on the analytic estimate, not on the extreme.** The octave sum gives about
28.1° with `G ≈ 2.97` (above), and the true supremum of `|∇ noise3|` is UNMEASURED (U2). **The home
planet cannot hold a cliff, a pillar, a mesa rim, a fjord wall or a canyon**: every one of those needs a
slope over 45°, and the best available bound on this field is about 28°, with the steepest ground actually
found at 11°. The reference picture is built almost entirely out of shapes this recipe cannot make.

**The equal-at-every-baseline row is the deeper finding.** On Earth the slope grows as the baseline
shrinks, because there is structure at every scale. Here the 1 m slope equals the 1 000 m slope. That is
the signature of a band-limited field whose finest wave is 48.8 m.

### 1.4a The skyline, measured at NAMED stations and marched until it converges

Revision 2 printed one skyline number from an unnamed camera at a 90 km march limit. Both refuters
refused it, for two different and both correct reasons: refutation A showed the number moves by a factor
of two between cameras, and refutation B showed that 90 km is 0.225 of the dominant wavelength, so the
march can stop before the ground turns over. Revision 3 replaces it.

**The station is now a RULE.** A ridge station is *the column of greatest surface radius inside a named
100 km patch of a named face, on a 101 × 101 sampling grid*. The patch is chosen; the station is a
consequence. Four patches give four stations, and each one prints as a face and a rung-0 cell pair, the
way slice 5 and slice 6 name every chunk they time:

| Station | Face | Face params (a, b) | Rung-0 cell (i, j) | Its height over the ladder radius |
|---|---|---|---|---|
| S2-a | PosX | 0.296580, 0.142400 | (3 412 184, 3 006 432) | +2 294.8 m |
| S2-b | PosY | −0.190120, 0.418999 | (2 131 344, 3 734 351) | +4 624.3 m |
| S2-c | PosZ | 0.052660, −0.342400 | (2 770 264, 1 730 591) | +1 247.2 m |
| S2-d | NegX | 0.449620, 0.235561 | (3 814 936, 3 251 600) | +1 267.9 m |

**The march is now converged.** REPLAYED: from each station, eye 373 m over its own surface, 360 bearings,
steps of 50 m out to 5 km and 300 m after that, to four limits. The break rule is §6.1's: a local maximum
that stands over the median of the skyline within ±5° of bearing.

| Station | Largest rise over the local trend at 90 / 180 / 360 / 540 km | Skyline elevation range at 360 km | Breaks per 60° at 0.05° | at 0.10° and above |
|---|---|---|---|---|
| S2-a | 0.0542° / 0.0544° / **0.0544°** / 0.0544° | −2.35° … −0.89° | 0.17 | **0** |
| S2-b | 0.0602° / 0.0602° / **0.0602°** / 0.0602° | −2.43° … +0.40° | 0.17 | **0** |
| S2-c | 0.0489° / 0.0636° / **0.0636°** / 0.0636° | −3.11° … −0.87° | 0.33 | **0** |
| S2-d | 0.0668° / 0.0668° / **0.0668°** / 0.0668° | −2.16° … −0.79° | 0.50 | **0** |

**What the convergence check found.** At station S2-c the 90 km march understated the largest rise by
23 % (0.0489° against 0.0636°), and at S2-a it cut 0.79° off the skyline's low end. **Refutation B's
blocker is confirmed: 90 km truncates.** Every station is stable from 360 km on, so 360 km is the march
limit this document now states. And the conclusion is unchanged: the largest rise anywhere is 0.067°, and
**no station holds a single break at 0.10° or above**. At 0.05° the entire 360° horizon holds one to three
breaks — features an eighth the width of a full moon (0.52°, PUBLISHED), which no player would call an
orienter.

**What this replaces.** Revision 2's "0.040°, and zero at every threshold down to 0.05°" is withdrawn: it
was one unnamed camera, at a truncated range, and it disagreed with revision 2's own §1.6 table, which
printed 0.045° for the same quantity. Refutation B was right to call that out. There is now ONE number
with its parameters stated, and §1.6 is restated on the same instrument.

There is no ridge in the frame because there is no ridge on the planet.

### 1.5 The spectrum

A **power spectrum** answers "how much of the shape sits at each wavelength". Write it as
`P(k) ∝ k^(−β)`, where `k` is one over the wavelength and β is the **spectral exponent**. A big β means
the long waves hold everything and the small scales are smooth. A small β means the small scales are as
busy as the big ones.

Our recipe halves the wavelength and multiplies the amplitude by `k_rough` at each step. That makes the
**Hurst exponent** `H = log₂(1/k_rough)` and the one-dimensional spectral exponent `β = 2H + 1`.

| `k_rough` | H | β | Which body |
|---|---|---|---|
| 0.45 (the low end of the draw) | 1.152 | 3.30 | the smoothest body the law allows |
| **0.4683** | **1.094** | **3.19** | **the home planet** |
| 0.55 (the high end) | 0.862 | 2.72 | the roughest |

REPLAYED. Two honest caveats, both raised by the refuters and both kept:

- `k_rough` is a per-body draw in `[0.45, 0.55)` (`body.rs:155`), so β is a per-body number, not "the
  recipe's". The range is 2.72–3.30.
- `β = 2H + 1` describes a fractional Brownian surface with `0 < H < 1`. Our H is above 1, which is
  outside the model's own range, and the field is a comb of fourteen discrete octaves rather than a power
  law. The formal reading of `H > 1` is simply: **the field is smooth, not fractal.** The measured slope
  table of §1.4 says the same thing without any model at all.

Earth's topography, measured along profiles, gives β ≈ 2 (PUBLISHED; the literature's range is about
1.6–2.4 by region, and §13 flags that this citation is unverified). Earth is much rougher at small scales
than our planet.

```text
   log P(k)
      │
      │╲                     β = 3.19  (ours: everything is in the long waves)
      │ ╲╲
      │  ╲ ╲╲
      │   ╲   ╲╲             β = 2     (Earth: energy at every scale)
      │    ╲     ╲╲
      │     ╲       ╲╲___
      │      ╲          ╲╲   ← and Earth has a BREAK here: the valley spacing
      │       ╲            ╲╲   (Perron et al.). A pure fractal has no break.
      └───────────────────────────► log k  (1/wavelength)
        100 km      1 km      10 m
```

To reach β = 2 the recipe would need `k_rough = 2^(−0.5) = 0.707` instead of 0.4683. §1.6 measures what
that actually does.

### 1.6 What a rougher noise alone buys — measured, and it is not the answer

REPLAYED. I re-ran the diagnostics of §1.3 and §1.4 on four variants of the SAME octave machinery, each
with the same total relief unless the row says otherwise. This is the experiment that decides §5.4's first
statement. Every variant is a BENCH computation; none of them is a proposal for a second world (SL5).

| Variant | slope median / share over 45°, at 1 m | shape inside 50 m after the tilt (rms) | **skyline breaks per 60°, at 0.5°** | biggest rise over the trend |
|---|---|---|---|---|
| **Today**: k 0.468, 400 km coarsest, 30 m floor | 1.36° / **0 %** | 0.26 m | **0.0** | **0.054°–0.067°** (§1.4a, four stations, 360 km) |
| k 0.707 (β ≈ 2), same relief | 33.2° / **30.1 %** | 20.5 m | **0.8** | 1.46° |
| k 0.707, relief cut to 35 % | 12.9° / 0.26 % | 7.2 m | 0.0 | 0.22° |
| 50 km coarsest, k 0.707, 2 m floor, same relief | 81.9° / **92.4 %** | 70.3 m | **5.2** | 8.91° |
| 50 km coarsest, k 0.60, 2 m floor, half relief | 31.5° / 27.4 % | 9.5 m | 0.3 | 0.60° |

**One honest caveat on the four variant rows.** Their skyline column was computed on revision 2's 90 km
march, which §1.4a has now measured to UNDERSTATE the largest rise by up to 23 %. Only the "Today" row is
re-measured on the converged four-station instrument. Understating strengthens the two rows that already
buy breaks (rows 2 and 4 would buy at least that many at 360 km) and may understate rows 3 and 5. A
re-run of all four at 360 km is owed as part of **U1**, and the reading below does not change with it,
because the argument turns on the 30 % and 92 % slope shares, which are not marched at all.

Read the second row and the fourth row together. **To buy skyline breaks out of pure noise, you must make
the planet unwalkable.** At Earth's spectral exponent with our relief, one column in three is steeper than
45°, and the skyline still holds less than one break per 60°. To reach five breaks the field must put
92 % of the surface over 45°: a world of scree with no floor to stand on.

**Game example.** The pilot lands his hull on the second row's planet. The ground where the ramp touches
tilts at 33°, the hull slides, and the player cannot walk to the ridge he can see. The ridge is not a
ridge either: it is the top of the same gravel, one metre higher than its neighbour.

**The conclusion.** Roughness and landform are different things. A landform is a shape that stands OVER
its neighbourhood — a ridge with valleys either side, a mesa with a rim, a scarp with a plain below. Noise
makes texture everywhere and structure nowhere. A mechanism with a memory of the neighbourhood is what
makes structure, and §5 says which ones exist.

---

## 2. The reference picture, read as numbers

The owner gave one reference: a Crimson Desert vista. It is a hand-authored AAA open world, so it is a
statement of TASTE and of DENSITY, not of technique. What it holds:

| What is in the picture | Its scale | What our world must hold |
|---|---|---|
| Snow-capped ridges on the skyline | 1 500–3 000 m tall, 30–60 km away | relief at the 10–50 km wavelength AND a snow line |
| Carved valleys between them | 200–800 m deep, 1–5 km apart | a drainage network with a characteristic spacing |
| Cliffs and rock pillars in the middle ground | 20–200 m of vertical over under 50 m of horizontal | slopes over 70°, against the steepest ground anybody has found on our planet: 11.15° |
| Bedded rock in the cliff faces | many bands, 1–20 m thick | **NEW WORK**: beds at a constant RADIUS with a hardness each. Today's table drapes four substances at a constant DEPTH (§3.5) |
| Forest as a mass, not as trees | canopy 15–30 m, over whole hillsides | a biome field plus the canopy fold (ruling V10 §12) |
| Fields and roads | 100 m–2 km | live state and settlement, not the seed |
| A river with a valley floor | 20–100 m wide, a concave profile | a channel network with a base level — **and a WATER SURFACE above the sea, which nothing in the world can hold today** (§3.5) |
| A LOW SUN, across the view | shadows kilometres long, cast along the slope | our picture harness puts the star 25° up and looks straight at it (§1.1) |
| Blue haze with distance | the far ridges lose contrast over 20–50 km | an atmosphere the client does not have (§2.1) |
| Clouds | 1–8 km up, and they cast shadow | the weather plan (§4.7) |
| A character for scale | 1.8 m | our own occupant, in the frame |
| Detail all the way to the horizon | about 50 km | a rung ladder that does not go blank |

**The view distance the reference implies, on OUR planet.** To see 50 km of ground the camera must stand
`h = d²/(2R) = 50 000²/(2 × 3 350 759) = 373 m` above the plain (ESTIMATED). On Earth the same view needs
196 m. **The home planet's horizon is 27 % closer than Earth's at every eye height**, so the same picture
needs either more relief or a higher camera.

At the measured median slope of 1.36° a climb of 373 m takes **15.7 km of walking** (ESTIMATED). A player
would climb the height of a mountain and never once feel that he had climbed. That is what "no orienters"
means, stated as a number. (Revision 1 said "the ground rises 688 m over a 20 km walk". Both refuters
noted that 688 m was an amplitude sum, not a rise anyone walks. The measured slope replaces it.)

### 2.1 The haze has no owner

MEASURED: `grep -niE "atmosphere|haze|fog|scatter"` over `crates/client-render/src` and
`crates/client/src` returns nothing. There is no depth cue in the client at all. Most of what makes the
reference read as fifty kilometres deep is aerial perspective — the far ridges going pale and blue. The
terrain can deliver every landform in this document and the frame will still read flat without it. This
document gates none of it, and it names no slice for it. **It belongs to somebody, and today it belongs to
nobody** (§11 D13).

**The light has the same problem, one level worse.** The haze is missing; the light is present, typed and
wrong (§1.1). `SUN_ELEVATION_DEG = 25.0` sits in a test file, no ruling names it, no overlay prints it,
and the camera looks into it. A regression set whose light nobody owns compares nothing (§11 D20).

### 2.2 The reference is not yet a number, and the gate must say so

The reference gives DENSITY and RANGE, and both are measurable — but nobody has measured THEM. Refutation
B is right about this and it is the sharpest finding of the round: **"8–12 silhouette breaks in 60°" was
counted by a person's eye off a screenshot, and "0" was computed by a stated rule over a height field.**
Those are two different quantities. A person counting orienters counts a snow cap, a colour change, a haze
band and a tree line; the rule counts none of them. The ratio between 8–12 and 0 means nothing, and this
document's own marking scheme exposes it: the 8–12 carries no mark at all — it is not MEASURED, not
REPLAYED, not PUBLISHED and not ESTIMATED.

**What follows for the gate.** A gate needs three fixed things: a subject, an instrument and a band. P1
now has a subject (four named stations) and an instrument (§1.4a). Its band is not yet honest. Two ways
out, and the owner chooses (D2):

1. **Measure the reference with our own rule** (U16): trace the reference picture's skyline by hand as a
   curve of elevation against bearing, using its own camera height and field of view, then run the same
   break rule on it. Then 8–12 becomes a number in our units and the band is calibrated.
2. **Land P1 as a RATCHET until then**: the count may never fall, and the largest rise may never fall.
   A ratchet has a subject and an instrument and needs no band, and it still catches a regression.

I recommend (2) now and (1) before the number is ever written as a floor.

**What the reference is NOT.** It is not a claim about our art budget, and it is not a claim that the
world must be hand-painted. It is a claim about the DENSITY of readable landforms per square kilometre,
and about the RANGE of slopes in one frame. Both are measurable. §6 measures ours; U16 measures the
reference's.

---

## 3. The landform vocabulary, with its scales

This section names the shapes the target asks for, and the words the mechanism documents will use. Each
term gets a plain explanation the first time, and a game example. The scales are PUBLISHED
order-of-magnitude values for Earth.

### 3.1 The shapes rock and water make

| Term | Plain words | Typical scale (Earth, PUBLISHED) | In the game's words |
|---|---|---|---|
| **Tectonic uplift** | The crust rising, because two plates push together or because hot rock pushes from below. It is the process that MAKES relief; everything else removes it. | 0.1–10 mm per year | The range the pilot flies along exists because the crust rose there for ten million years. |
| **Isostasy** | The crust floats on a heavier mantle, so a mountain carries a root under it. | a root about five times the height | Why a 5 km peak is not simply 5 km of extra rock. |
| **Range** | A long line of mountains that uplift raised. | 100–3 000 km long, 50–200 km wide, 2–8 km tall | The pilot sees the range from orbit and knows which hemisphere he is over. |
| **Ridge** | The high line where two slopes meet. | 1–50 km long, 100–1 500 m above the valley | The ridge the player climbs to get his 50 km view. |
| **Valley** | The low line that water follows. | spaced 0.5–5 km apart, 50–800 m deep | The route between two ridges; a road follows it. |
| **Drainage basin** | All the ground whose water leaves by one river mouth. | 10 km² to 7 000 000 km² (the Amazon) | The realm's whole continent drains to one delta. |
| **Stream order (Horton–Strahler)** | The rule that numbers channels: a brook is order 1, and two channels of one order join into the next order. | order 1 to 12 on Earth | A first-order brook beside the player's camp; a twelfth-order river at the coast. |
| **Bifurcation ratio** | How many channels of one order feed one channel of the next. | 3 to 5 | A cheap check on any network we build: if ours is 12, the network is wrong. |
| **Hack's law** | The longest channel's length grows as the basin's area to the power 0.57. | `L ∝ A^0.57` | Another cheap check on the same network. |
| **Base level** | The lowest point a river can cut to: the sea, or a lake. | sea level, or a closed basin floor | The sea radius the body definition already draws (`body.rs:186-190`). **Only the sea**: a body holds ONE `sea_radius_m`, so there is no local base level anywhere and no lake can stand above it (§3.5, §6.6). |
| **Fluvial incision** | Water cutting rock as it flows. Fast water on steep ground cuts fastest. | 0.01–10 mm per year | The canyon the river cut through the plateau. |
| **Hydraulic erosion** | Moving water picking rock up, carrying it, and dropping it. In a generator it is a simulation over a grid of cells. | — | What would turn our smooth swell into ridges and gullies, if we could afford to run it. |
| **Hillslope diffusion (soil creep)** | Soil sliding downhill grain by grain, which rounds a hill. | it smooths features under about 100 m | Why hilltops are round and canyon walls are not. |
| **Angle of repose** | The steepest angle loose grains hold before they slide. | 32–37° for dry sand and scree | The slope a pile of mined gravel settles at beside the player's camp. |
| **Thermal erosion / talus** | Rock breaks off a cliff and piles at its foot at the angle of repose. | angle of repose 32–37° | The scree the player slides down under the cliff. |
| **Cliff** | A slope steeper than soil can hold; bare rock. | 10–1 000 m of vertical | El Capitan is 900 m (PUBLISHED). The steepest ground found on our planet is 11.15° (§1.4). |
| **Plateau / mesa / butte** | A flat top left standing while the ground around it wore away, because a HARD layer caps a SOFT one. | mesa 1–20 km across, 100–600 m tall | The red-rock top the reference's camera stands on. It needs strata (§3.5). |
| **Hoodoo / pillar** | A column left where a hard cap protected the soft rock under it. | 5–50 m tall, 2–15 m across | An orienter a player names and returns to. Our height field cannot make one (§3.5). |
| **Canyon** | A deep narrow valley in rock. | Grand Canyon: 1 600 m deep, 16 km wide, 446 km long | A flight path a pilot threads. |
| **Delta / floodplain** | Flat ground of dropped sediment where a river slows. | 10–100 km across | Where the first settlement goes. |
| **Coast** | Where the land meets the sea. | the sea radius, plus the tide | The shore the ship lands on. 0.99 % of our planet is under water. |
| **Fjord** | A valley that ice cut and the sea then drowned. | 1–5 km wide, 500–1 500 m walls, 50–200 km long | A hull flying below the rim, with sky in a strip. |
| **Glacier / snow line (the equilibrium line altitude)** | The height at which a year's snowfall equals a year's melt. Above it snow lasts. | tropics about 4 900 m, poles about 0 m | The white caps in the reference picture. Our `biome_at` turns cold with height and gives `Tundra`, whose topsoil is `Stratum::Snow` — so a white cap CAN draw today; what is missing is a law that puts it at the right height (§6.7 P7b). |
| **Tree line (the timber line)** | The height above which trees cannot grow, because the growing season is too cold or too short. | 3 500–4 000 m in the tropics, near sea level at 70° north | **The cheapest scale reference on a hillside**, and the reference picture's strongest orienter after the ridge: a player reads a mountain's height from where the forest gives up. Refutation A is right that revision 2 omitted it. `biome_at` already cools with height (`height.rs:52-53`), so the line is one biome boundary away — once a forest biome exists at all (§11 D18). |
| **Dune field** | Sand that the wind moved and piled. | 10–300 m tall, 0.5–3 km spacing | The desert biome's own landform. |
| **Volcano** | A cone that erupted rock built. | 1–9 km tall, 10–600 km across | Olympus Mons is 21.9 km on Mars (PUBLISHED). |
| **Impact crater** | The hole a rock made. | 10 m to 2 000 km | The first landform on any body without air. Vesta's Rheasilvia basin is most of Vesta's 22 km of relief. |
| **Karst** | Rock that water dissolved: caves, sinkholes, towers. | caves 1 m–100 km long | The cave system ruling V5 asks for, seen from its mouth. |

### 3.2 The words the recipes use

The mechanism documents will use these constantly. They are explained here once, because both refuters
noted that revision 1 used them without a gloss.

| Term | Plain words | In the game's words |
|---|---|---|
| **fBm** — fractional Brownian motion | Adding several noises together, each with half the wavelength and a fixed fraction of the height of the one before. It is exactly what `height_m` does today. | The home planet's fourteen octaves ARE an fBm. |
| **Octave** | One of those noises. | Octave 13 is the 48.8 m ripple, and rung 13 would drop it. |
| **Ridged multifractal** | Fold the noise about zero and turn it upside down, so every zero crossing becomes a sharp crest instead of a smooth wave. Then let the height already built decide how much the next octave adds. | It makes a ridge line a player can follow with his eye, out of the same per-point arithmetic. |
| **Billowed noise** | The same fold without the flip, so the field becomes rounded lumps. | Dunes and rolling hills. |
| **Domain warping** | Before you ask the noise for a height, move the point you ask about, using another noise. The shape stops looking like a grid of blobs and starts looking like it flowed. | The difference between "noise" and "a landscape that water once ran over". |
| **Analytic derivative** | The noise reports its own slope at the same time as its height, for a small extra cost. | You can then damp the next octave where the ground is already steep, which flattens valley floors and sharpens crests. |
| **Flow accumulation** | How much ground drains through a point. | The brook at the player's camp has a small one; the river at the delta has a huge one. |
| **Stream power law** | Erosion goes as the drainage area to a power times the slope to a power: `E = K·A^m·S^n`. | Why a big river cuts a deep valley and a brook only scratches. |
| **Slope–area law** | The consequence of the one above: `S ∝ A^(−θ)`, with the **concavity** θ about 0.4–0.6. | The brook at the camp runs steeply; the river at the delta runs almost flat; and the ratio between them is the same everywhere on a believable planet. |
| **Drainage density** | Channel length per square kilometre. | How often the player crosses a stream on a walk. |
| **Jeans escape** | Gas molecules at the top of an atmosphere move fast enough to leave. A small hot body loses its air. | Whether the planet the pilot approaches has a sky or is a bare rock. |
| **Cosmic shoreline** | The empirical line, in escape velocity against sunlight, that separates worlds with air from worlds without. | `taxonomy.rs:794` already computes it. |
| **Hypsometry** | The distribution of heights over a surface. | How much of the planet is plain, how much is slope, how much is peak. |
| **rms** — root mean square | The typical size of a set of leftovers, counting a bump and a dip alike: square them, average, take the root. | "The ground departs from a plane by 0.26 m rms inside 50 m" means a typical bump under the player's boots is a quarter of a metre. |
| **p90, p99** | The value only one column in ten, or one in a hundred, beats. | The p99 slope is what the player meets on the steepest walk in a hundred. |
| **Skew** | Whether a distribution leans left or right of its middle. | A planet of plains with a few peaks skews right. Ours is −0.03: it leans neither way. |
| **Kurtosis** | Whether a distribution has heavy tails (over 3) or light ones (under 3). | Ours is 2.80, which is lighter than a bell curve: no tail of high ground, so no mountains. |
| **The central limit theorem** | A sum of MANY independent bounded terms tends toward a bell curve. | It does NOT apply to our field: 78 % of the amplitude sits in two octaves, so there are about two terms, not fourteen (§1.3). |
| **Hadley cell edge** | The latitude where the air that rose at the equator sinks again. The world's great deserts sit there. | On our planet it decides whether the pilot finds a desert belt at ±30° or a single band of dry ground at the pole (§4.5). |
| **Whittaker classification** | Biomes laid out on two axes: the mean temperature against the mean rainfall. | The natural home for a forest, a wetland, a savanna and an ice cap — and for V4's asset kits. Our world has four biomes on one axis and a height class (§11 D18). |

### 3.3 The scales, on one line

```text
  1 m      10 m     100 m     1 km      10 km    100 km   1000 km
  │────────│────────│─────────│─────────│────────│────────│
  boulder  hoodoo   cliff     mesa      range    province continent
  step     tree     ridge     valley    basin    plate    hemisphere
           talus    dune      canyon    fjord
  ▲                 ▲                   ▲                 ▲
  the cell          what the eye        the horizon       what the pilot
  (1 m)             reads on a walk     from a ridge      sees from orbit
  ▲─────────────────────────────▲
  our finest octave is 48.8 m, and inside 50 m the ground is a plane to 26 cm (§1.3)
```

### 3.4 What "detail to the horizon" means, in geometry

```text
        eye at height h
             ▲
             │              horizon distance  d = sqrt(2·R·h)
             │        ╭──────────────────────────╮
             ●────────┴──────────────────────────┴─── the ground curves away
            ╱ ╲
           ╱   ╲   R = 3 350 759 m (the home planet)
```

| Eye height | Home planet horizon | Earth horizon (PUBLISHED R) |
|---|---|---|
| 1.7 m (standing) | **3.38 km** | 4.65 km |
| 10 m (a hull's bridge) | 8.19 km | 11.3 km |
| 100 m (a low pass) | 25.9 km | 35.7 km |
| 373 m (the reference's viewpoint) | **50.0 km** | 68.9 km |
| 1 000 m (a ridge) | 81.9 km | 112.9 km |
| 3 000 m (a peak) | 141.8 km | 195.5 km |

A distant object taller than the ground is visible past the horizon: a 1 000 m ridge is visible from
85.2 km at eye height, and a 3 000 m peak from 145.2 km (ESTIMATED, `√(2Rh₁) + √(2Rh₂)`).

**What this costs in rungs — corrected.** One metre at 50 km subtends 0.0275 of a pixel at a 45° field of
view over 1 080 rows (ESTIMATED), so a cell must be about **36 m** to cover one pixel there. That is
**rung 5 (32 m cells) or rung 6 (64 m cells)**, not rung 9. Revision 1 said rung 9 because it compared a
CHUNK's width with the distance instead of a CELL's; refutation B caught it, and the error was 16× in cell
size:

| Rung | Cell | Pixels at 50 km | Chunk width |
|---|---|---|---|
| 4 | 16 m | 0.44 | 1.0 km |
| **5** | **32 m** | **0.88** | 2.0 km |
| **6** | **64 m** | **1.76** | 4.0 km |
| 9 | 512 m | 14.08 | 31.7 km |

Rung 9 at 50 km puts fourteen pixels on one cell, which is a visible staircase and one of SL8's own named
seams. The correction matters because the rung decides how many chunks a frame holds, and §8.4 prices
that.

**But a cell size is not detail, and refutation B is right that the pixel table answers the wrong
question.** `octaves_at` (`body.rs:253`) keeps *the coarsest `octave_count − rung`* octaves. So the rung
that fixes the cell size also throws away the shape:

| Rung | Cell | Finest LIVE octave (§1.2's table) | What a 50 km ridge can show |
|---|---|---|---|
| 0 | 1 m | 48.8 m | everything the recipe has |
| **5** | **32 m** | **1 562 m** | nothing smaller than a kilometre and a half |
| **6** | **64 m** | **3 125 m** | nothing smaller than three kilometres |
| 9 | 512 m | 25 000 m | one swell |

The reference's far ridges do the opposite: at 30–60 km they show sub-ridges and gullies of 200–500 m,
which subtend 0.2°–0.6° and therefore five to fourteen pixels. **Those are exactly the features the
ladder removes.**

**The deeper observation, which revision 2 never made.** The finest live octave is always about 48 CELLS
wide, at every rung: 48.8 m over a 1 m cell at rung 0, and 1 562 m over a 32 m cell at rung 5. So the
ladder guarantees the same relative smoothness at every distance. That is a second, independent statement
of §1.4's "the slope is the same at every baseline", and it means **"detail to the horizon" is not a rung
question at all.** It is the question of whether a coarse rung may carry structure that the free coarsening
does not remove — a change to the ladder's own contract (rulings V8/V9), and therefore an owner decision
this document must raise rather than price. It is **D21**, and §12 Q4 holds the bound it needs.

**Game example.** The pilot stands at station S2-b and looks at a ridge 50 km away. At rung 5 that ridge
is drawn on 32 m cells that carry no shape under 1 562 m, so it is one smooth swell. The gullies the
reference shows on the same ridge are three rungs finer than the cell size the pixel rule allows.

### 3.5 What the shape and the record cannot hold at all

Five structural limits, none of them about slope. Revision 1 argued only about slope and missed all five;
revision 2 found two and then claimed a third was already solved. Both refuters corrected that, and they
are right.

1. **No overhang, ever.** `height_m` returns ONE radius per direction
   (`crates/terrain/src/height.rs:16-27`). That is a single-valued height field. A hoodoo with a cap wider
   than its stem, a natural arch, an undercut sea cliff and a karst tower are all impossible in it, at any
   slope and any roughness. Only the 3-D carvers (`carve.rs`) make shape a height field cannot, and they
   are fenced (below). §3.1 asks for three of those landforms; a mechanism must say which layer makes
   them. This is also a collider fact: the boots cannot stand on an arch the shape cannot represent.
2. **No cave mouth, today.** `crates/terrain/src/chunk.rs:377-383` gates every carve between
   `body.caves.min_depth_m` and `max_depth_m`, and the minimum is drawn in `8..30` m (`body.rs:219`). So
   at least eight metres of roof always stand between a cavern and the surface. Ruling V5 asks for *"when
   we are flying over the cave and there is enough light, we should see what's inside"*, and slice 7 owes
   a cave-mouth picture. **The code guarantees there is no opening to photograph.** That is a defect
   against a ruling, and it is recorded here for the mechanism documents.

3. **The strata DRAPE; they cannot cap.** Revision 2 called them "the cheapest believability, already
   built and unused". **I withdraw that entirely.** MEASURED, `crates/terrain/src/strata.rs:187-210` and
   `crates/terrain/src/chunk.rs:631`:

   ```rust
   let depth = h - r;                    // h = the COLUMN's surface radius, r = the cell's radius
   body.strata.at(site.biome, depth.floor()…)   // the substance BY DEPTH UNDER THE SURFACE
   ```

   Three consequences, all of which kill the recommendation:
   - **A layer at a constant DEPTH follows the hill down its own side.** Real bedding sits at a constant
     RADIUS. A draped layer caps nothing, because the same layer is on the valley floor too.
   - **The hard rock is at the BOTTOM, by construction.** The order is topsoil (1–4 m), subsoil (2–8 m),
     sediment (20–80 m), then bedrock forever (`body.rs:194-202`). A mesa and a hoodoo are a HARD cap over
     a SOFT layer. No erosion rule of any kind can leave a cap standing when the cap material is never on
     top.
   - **A cliff can expose at most FOUR substances.** A body draws ONE sediment of three and ONE bedrock of
     five (`body.rs:198-201`), so the face shows topsoil, subsoil, one sediment, one bedrock. The
     reference's cliffs show many bands 1–20 m thick. And "19 strata" is a substance PALETTE, not a
     sequence: `Stratum::ALL` (`strata.rs:42-60`) holds 19 entries of which `Air`, `Water` and `Empty` are
     not rock at all and five are the bedrock choices.

   **What survives.** The IDEA is still right — bedding is what makes a cliff read as rock instead of as a
   brown wall, and a hard cap over soft rock is what a mesa and a hoodoo ARE. It is simply **new work**:
   beds at a constant radius, each with a hardness, with a thickness that varies over the body. That
   costs field evaluations, it touches the substance the record already carries, and nobody has priced it.
   D12 is rewritten to say so.

   **Game example.** The player stands under the red-rock rim the reference shows. On our planet the rim's
   substance is whatever sits one metre under the local surface, which is the same dirt as the plain
   behind him, and the granite is ninety metres under his boots wherever he walks.

4. **No water above the sea, so no river SURFACE.** MEASURED: a body holds ONE `sea_radius_m`
   (`body.rs:186-190`), and `fluid_at` (`chunk.rs:207-213`) returns `Water` below it and `Air` above it,
   with nothing else anywhere. `Stratum::Water` exists; nothing places it above the sea radius. So a river
   surface, a lake, a tarn and a valley-floor pond are not expensive — they are **inexpressible**. Half
   the land on this planet stands 5.3 km above its sea (§1.3), so every channel the target asks for would
   run dry. Refutation B is right that revision 2 asked for the channel and never asked for the water.
   This is three questions at once, and none has been asked: a RECORD question (a per-column water level,
   or a `Water` stratum the generator writes), a COLLIDER question (does the pilot's hull float on it),
   and an SL10 question (a lake level derived from the seed, or a one-hop diff from the realm). It is
   **D19**.

5. **The generated cell holds no biome, no moisture and no temperature.** MEASURED,
   `crates/terrain/src/chunk.rs:64-69`: a cell is `Cell { stratum: Stratum, gap: i8 }`. `biome_at` runs
   once per COLUMN (`chunk.rs:637`), its only effect is to choose which topsoil `strata.at` returns, and
   the biome value itself is thrown away inside the generator. Ruling V6 Part B's approved twelve bytes
   (`owner_decisions_2026-09-07_voxels.md:186-193`) are a 16-bit kind, the density byte, six rotation
   bits, the tree's shape byte and the attachment key — **no climate parameter is listed.** Both refuters
   caught that revision 2's §9 closed this question by asserting it was already answered. It is not; it is
   the one place where the target does touch the format, and §9 now states it as an ask.

---

## 4. "Believable" means the body's own facts decide its landforms

The owner asked for biomes that depend on *"the planet position, spin, trajectory, size and gravity"*.
That request is also the answer to "no magic numbers": every band in §6 that HAS a law must be a FUNCTION
of the body's facts, never a constant a person typed. This section writes each link as arithmetic, names
the fact it needs, and says whether the code holds that fact today.

### 4.1 The chain, in one drawing

Revision 1's drawing ran from the star to the rain to the rivers. Every arrow in it REMOVED ground. Both
refuters pointed out that nothing in it RAISED any. The missing box is where the interest lives.

```text
   THE PROCESS THAT BUILDS                    THE PROCESS THAT REMOVES
   ───────────────────────                    ────────────────────────
   the interior: mass, heat, age              the star ──► insolation S
        │                                          │
        ├──► plate convergence ──► RANGES          ├──► temperature ──► ice or water?
        ├──► a plume ──► VOLCANOES                 │
        ├──► impacts ──► CRATERS, BASINS           ├──► spin Ω ──► Coriolis f = 2Ω sin φ
        └──► isostasy ──► the root under it        │        └──► how many Hadley cells
                          │                        │             └──► WHERE DESERTS ARE
                          ▼                        │
                    ┌──────────────────────────────┴───────────┐
                    │   uplift SET AGAINST erosion decides     │
                    │   the ridge spacing and the valley depth │
                    └──────────────────────────────────────────┘
                          ▲                        ▲
   gravity g ─────────────┘                        │
     ├──► the strength CEILING h ≤ σ/(ρ_c g)       ├──► obliquity ──► seasons, tropics
     ├──► the lapse rate Γ = g/c_p ──► SNOW LINE   ├──► wind ──► orographic lift
     └──► the scale height H ──► THE HAZE          │             └──► RAIN SHADOW
                                                   └──► rain ──► rivers ──► VALLEYS, DELTAS
```

Cordonnier's 2016 work, cited in §5.2, is exactly this balance solved as one system: uplift against
fluvial erosion. Revision 1 kept only the erosion half, and then asked for a gate on peaks — which are
what uplift leaves behind.

Most of the right-hand arithmetic already exists in `crates/physics/src/taxonomy.rs` (§4.6). The generator
crate reads none of it. Nothing anywhere holds the left-hand column: the world draws a mass and a radius,
and no tectonic history at all.

### 4.2 SIZE and GRAVITY set a CEILING on relief — and the code is not anchored to it

**What revision 1 got wrong.** It recommended moving the relief law from `∝ radius` to `∝ 1/gravity`, and
called that a one-way door the owner should take before the first world is saved. **I withdraw that
recommendation.** Both refuters refuted it with the document's own table, and they are right:

- Fitted on the six bodies below, the log–log slope of relief against radius is **−0.028** with
  correlation **−0.09**; against gravity it is **−0.038** with correlation **−0.17** (REPLAYED). A `1/R`
  or `1/g` law needs a slope of −1. There is no such relation in the data.
- Calibrated so Earth is right, `relief = 19.9 km × 6371/R` gives Vesta 484 km of relief — nearly twice
  its own diameter. `Ladder::for_radius` would refuse the body outright (`ladder.rs:89`, a crust deeper
  than the surface), so the realm would have no grid at all.
- Four of the six reliefs are not isostatic anyway. Vesta's 22 km is the Rheasilvia IMPACT basin, the
  Moon's 19.9 km is the South Pole–Aitken basin, Mars's 29.4 km is a volcano over a plume, and Iapetus's
  20 km is an equatorial ridge nobody has explained. **Relief is set by the PROCESS, and gravity is one
  term.**

| Body | Radius (km) | g (m/s²) | Relief (km) | Relief ÷ radius | What made it |
|---|---|---|---|---|---|
| Earth | 6 371 | 9.81 | 19.9 | 0.31 % | plates, ice, water |
| Mars | 3 390 | 3.72 | 29.4 | 0.87 % | a plume under a still lid |
| Mercury | 2 440 | 3.70 | 9.9 | 0.40 % | impacts, cooling |
| The Moon | 1 737 | 1.62 | 19.9 | 1.15 % | one enormous impact |
| Iapetus | 735 | 0.22 | 20 | 2.72 % | unexplained |
| Vesta | 262 | 0.25 | 22 | 8.40 % | one enormous impact |

All PUBLISHED. The honest reading: **absolute relief is roughly constant at 10–29 km over a 24-fold range
of radius**, and the ratio to the radius grows only because the radius shrinks.

**What gravity DOES decide: a ceiling.** A mountain cannot stand taller than its own rock can bear its own
weight. The limit is `h ≤ σ/(ρ_c g)`: the rock's strength over its weight. Calibrate σ on Everest at a
crustal density of 2 700 kg/m³, and σ comes out at **234 MPa**, which sits inside granite's published
compressive strength of 100–250 MPa. The ceiling then holds for every body in the solar system
(REPLAYED):

| Body | g | Ceiling `σ/(ρ_c g)` | Its tallest real relief | Share of the ceiling used |
|---|---|---|---|---|
| Earth | 9.81 | 8 850 m | Everest, 8 850 m | 100 % (the calibration) |
| Venus | 8.87 | 9 788 m | Maxwell Montes, 11 000 m | 112 % |
| Mars | 3.72 | 23 338 m | Olympus Mons, 21 900 m | 94 % |
| Mercury | 3.70 | 23 464 m | 4 500 m | 19 % |
| The Moon | 1.62 | 53 592 m | 10 800 m | 20 % |
| Vesta | 0.25 | 347 274 m | 22 000 m | 6 % |
| Iapetus | 0.22 | 394 630 m | 20 000 m | 5 % |

**What this ceiling is, stated honestly.** Revision 2 wrote "a real bound: no body passes it by more than
the uncertainty in σ". Refutation B refuses that on its own table, and is right on three counts:

- **Its own calibration set breaks it.** Venus's Maxwell Montes stands at **112 %** of the ceiling. A
  bound its own data exceeds is a fit with an outlier, not a bound. If σ may flex 12 % to absorb Venus,
  then the headline consequence ("600 % on a super-Earth") carries the same 12 % of slack.
- **It is the wrong physics for a mountain.** `h ≤ σ/(ρ_c g)` is the crushing limit of a rock COLUMN under
  its own weight. That is not what limits a mountain. Olympus Mons stands 21.9 km because a thick, cold,
  stiff lithosphere carries the load in FLEXURE, not because basalt is strong. Everest stands where it
  does because uplift, isostatic rebound and glacial erosion balance.
- **Three different rulers sit in one column.** Everest's 8 850 m is above sea level, Maxwell Montes's
  11 000 m is above Venus's mean radius, and Olympus Mons's 21.9 km is above the Martian datum.

**So label it what it is: an order-of-magnitude sanity bound, good to about a factor of two.** It is NOT a
predictor either — the Moon uses a fifth of it and Earth uses all of it, because the process differs. What
it still does, and what nothing else in this document does, is catch a law that is wrong by three orders
of magnitude. That is exactly the defect it catches below.

**Now measure the code against the ceiling** (REPLAYED; the code's relief is
`clamp(0.004 R, 200, 12 000) × [0.5, 1.5)`, and the ceiling assumes a crustal density of 2 700 kg/m³ over
a body of the density shown):

| Radius | Body density | g | Ceiling | The code draws | Share of the ceiling |
|---|---|---|---|---|---|
| 100 km | 2 700 | 0.08 | 1 150 km | 200–600 m | **0.02–0.05 %** |
| 200 km | 2 700 | 0.15 | 575 km | 400–1 200 m | 0.07–0.21 % |
| 262 km (Vesta) | 3 456 | 0.25 | 343 km | 524–1 572 m | 0.15–0.46 % |
| 1 737 km (the Moon) | 3 344 | 1.62 | 53 km | 3 474–10 422 m | 6.5–19.5 % |
| 3 351 km (home) | 5 000 (a guess, U6) | 4.68 | 18.5 km | 6 000–18 000 m | 32–97 % |
| 6 371 km (Earth) | 5 514 | 9.82 | 8.8 km | 6 000–18 000 m | **68–204 %** |
| 12 000 km (a super-Earth) | 8 608 | 28.9 | 3.0 km | 6 000–18 000 m | **200–599 %** |

**That is the real defect, and a factor-of-two slack in the ceiling does not touch it.** The law is
anchored to no physical bound at all. A small moon gets a thousandth of what it could carry, and a big planet gets six times what its crust
can hold. The 12 km cap and the 200 m floor are magic numbers, and they are the only thing between the law
and a mountain that would fall over.

**Game example.** A pilot lands on a 200 km asteroid realm. The code gives it an 800 m swell: nothing to
fly through, nothing to hide behind, nothing to name. Its rock could hold 575 km of relief. Vesta, at
262 km, carries a 22 km impact scar with a central peak taller than Everest and a rim wall a hull can fly
along. The second one is a place. The first one is a sphere.

**What I now recommend the owner decide** (§11 D5, rewritten): the relief law is OPEN. The evidence says
`∝ radius` is wrong; it does not say what is right. A defensible shape is (a) a ceiling from gravity as a
BOUND, and (b) the value drawn from the body's own process history — impacts for an airless small body,
volcanism for a young one, plates for an Earth-like one — which the world does not model yet. **No one-way
door is put to the owner in this revision.** A change to the relief law still moves every body and every
golden digest, so it stays a world-epoch decision whenever it is taken (U8).

### 4.3 GRAVITY also decides the snow line and the haze

**Lapse rate** is how fast the air cools as you go up. Dry air cools at `Γ = g/c_p`, where `c_p` is the
air's heat capacity. On Earth that is 9.76 K/km, and the real average is about 6.5 K/km, because water
vapour gives heat back when it condenses.

Snow stays where the temperature is under freezing, so a first estimate of the **snow line** is
`z_snow = (T_surface − 273 K)/Γ`.

| Body | g | Dry lapse `g/c_p` | Snow line at T = 300 K (env. lapse = ⅔ dry) |
|---|---|---|---|
| Earth | 9.81 | 9.76 K/km | 4.15 km (PUBLISHED tropical value: about 4.9 km) |
| A half-Earth planet | 4.90 | 4.88 K/km | **8.30 km** |
| Mars-like | 3.70 | 3.68 K/km | 6.92 km at T = 290 K |

ESTIMATED, with `c_p = 1005 J/(kg·K)` for a nitrogen–oxygen air (PUBLISHED).

**Two caveats a climatologist would insist on** (refutation B, N4 — accepted):

- The ⅔ factor is Earth's own number. The environmental lapse rate is set by moist convection and by
  radiation, and the moist part depends on how much water the air holds — which depends on the body's
  ocean, not on its gravity. So the "half-Earth planet" row silently assumes Earth's water.
- The real snow line is the **equilibrium line altitude**: the height where a year's snowfall equals a
  year's melt. A dry cold mountain carries no snow cap; a wet warm one does. Aridity belongs in the
  arithmetic, and §4.7 lists aridity while this table omits it.

So the row above is a first estimate, not a law to gate on. What survives is the DIRECTION: **a
low-gravity planet cools slowly with height, so its snow line sits high, so its mountains must be taller
before they wear a white cap.** §4.2 says a low-gravity planet also has a higher ceiling. The two agree,
which is worth noting — and it is an argument, not a result, because the ceiling is a bound and not a
predictor of the actual relief.

**Scale height** `H = kT/(μ g)` is the height over which the air thins by a factor of e. It sets how deep
the blue haze is and how far a ridge stays visible. `crates/physics/src/taxonomy.rs:762` already computes
it. The client has nothing that would use it (§2.1).

### 4.4 POSITION and TRAJECTORY decide the temperature and the seasons

- **Insolation** is the light the body receives: `S = L/(4πd²)`, where `L` is the star's luminosity and
  `d` the orbit radius. `taxonomy.rs:324` draws `L` from the star's mass;
  `crates/physics/src/worldgen/generate.rs` draws the orbit.
- **Equilibrium temperature** `T_eq = ((1−A)S/(4σ))^¼`, where `A` is the **Bond albedo** — the share of
  light the body reflects. `taxonomy.rs:681` computes it, and `taxonomy.rs:387,389` holds the albedo with
  and without an atmosphere.
- **Eccentricity** is how far from a circle the orbit is. It makes the whole planet warmer at periapsis
  and colder at apoapsis. `worldgen/generate.rs:147,159` draws it. This is the owner's "trajectory".
- The **habitable zone** (`taxonomy.rs:332`) and the **frost line** (`taxonomy.rs:519`) already say
  whether liquid water is possible at all. A body outside the zone gets no rivers, and therefore no
  fluvial valleys. Its whole landform vocabulary changes.

**Game example.** Two planets of one size in one system. The inner one sits inside the habitable zone, so
it wears rivers, forests and a snow line near its poles. The outer one sits past the frost line, so its
whole surface is ice, its valleys come from ice flow and from impacts, and it has no rivers at all. The
pilot reads which is which from orbit, before he lands. §6.7 makes that difference a printed proxy, so a
gate can fail when two such planets look the same.

### 4.5 SPIN decides where the deserts are — and no body draws a spin

Air rises at the hot equator, travels toward the pole, cools, and sinks. That loop is a **Hadley cell**.
The **Coriolis** effect — the sideways push a spinning planet gives to moving air, `f = 2Ω sin φ` — breaks
the loop into several cells. Earth spins fast enough for three cells in each hemisphere, so air sinks at
about ±30° and makes the world's great deserts there.

- A **fast spinner** gets many narrow cells and many climate belts.
- A **slow spinner** gets ONE cell per hemisphere: air rises at the equator and sinks at the pole, and
  there is no ±30° desert belt at all.

**Obliquity** is the tilt of the spin axis against the orbit. It puts the tropics and the polar circles
where they are, and it makes the seasons. Earth's is 23.4°; Uranus's is 98°, so its poles get more
sunlight in a year than its equator (PUBLISHED).

**MEASURED, stated precisely.** `grep -rniE "obliquity|axial_tilt|rotation_period|spin_rate|day_length"`
over `crates/` returns exactly **two** hits, and both are comments: `crates/terrain/src/height.rs:32`
(which calls an obliquity "a later slice") and `crates/physics/tests/celestial_ephemeris_pin.rs:29`
(`// ~23.4° obliquity (rad)`, a test fixture's inclination). **No body draws a spin rate or an
obliquity.** `height.rs:35` hardwires `POLE_AXIS = 2` (+Z), so every body's spin axis IS its orbit's axis.

**But the world already holds the SHAPE of a spin, and revision 1 said it did not.** Refutation B is
right: `crates/core/src/frame.rs:66` gives every frame an `angular_velocity`, `frame.rs:243-273` carries
the Coriolis term through every crossing, and `crates/sim/src/stub/placement.rs:149` fills a placement's
`angular_velocity` from an occupant's own `spin_radps`. So a spin is a **placement** datum, and SL1
already answers "who authors it": **the parent authors its children's placements, and the spin is part of
a placement.** A body's obliquity is the orientation of that placement against the orbital plane.

**But a placement may NOT be what shapes the ground, and revision 2 missed that.** Refutation A is right,
and the argument is short:

- SL1 clause 6: a stamped reading carries its instant, and **a stale reading is REFUSED, never used**.
- SL10: the client may derive the static shape from `(seed, address)` alone, *never a function of time or
  live state*, and `just terrain-legs` compares server-built and client-built chunks byte for byte.
- A datum that arrives on a lane, carries an instant, and may be refused as stale is the exact opposite of
  an input to a byte-identical seed-derived shape. If the desert belt of §6.7 P7a were a function of a
  placement reading, two hosts holding different readings would build different ground — and the no-drift
  gate could not even express the comparison, because one leg has no lane at all.

**The lawful form, which revision 2 did not name.** The body **DRAWS its spin rate and its obliquity from
its own seed**, under the float fence, exactly as it draws its relief, its roughness and its sea. The
placement lane then REPORTS the same number to whoever moves. One value, one owner, one law: the seed
decides the shape, the placement carries the motion. D10 is rewritten around that, and Q2 now asks who
draws the value rather than which lane carries it.

**A named consequence for the world identity.** A spin draw inside `BodyDefinition::from_seed` adds draws
to the octave stream's neighbours. Under ruling V6 Part D the identity is APPEND-ONLY with a zero
tolerance (ruling V9 S5-2), so the draw must be taken under its OWN salt (`salt::SPIN`, beside
`salt::SEA`), or every existing body moves. That is cheap if it is done once and correctly, and it is a
world epoch if it is done twice.

Today's biome field (`height.rs:38-66`) reads latitude, height and two slow noises, and nothing else. It
cannot put a desert at ±30°, because it does not know how fast the planet turns.

### 4.6 The facts exist. The generator cannot reach them.

MEASURED, from the crate manifests and CLAUDE.md's dependency rule: `vd-terrain` depends on `vd-seed` and
on nothing else, exactly so a client can link it without linking the motion crate. The physical facts live
in `crates/physics/src/taxonomy.rs`:

| Fact the target needs | Where it lives today | Reachable from `vd-terrain`? |
|---|---|---|
| Star luminosity | `taxonomy.rs:324 main_sequence_luminosity` | no |
| Insolation, equilibrium temperature | `taxonomy.rs:681 equilibrium_temperature_k` | no |
| Bond albedo, with and without air | `taxonomy.rs:387,389` | no |
| Surface gravity | `taxonomy.rs:750 surface_gravity_mps2` | no |
| Escape velocity, scale height | `taxonomy.rs:756,762` | no |
| Does the body keep an atmosphere? | `taxonomy.rs:794 cosmic_shoreline_retains`, `:814 jeans_retains` | no |
| Orbit: semi-major axis, eccentricity, inclination | `worldgen/generate.rs:147-162` | no |
| Spin, obliquity | the placement lane holds the SHAPE (`frame.rs:66`); **no body draws a value** | — |

**This is the target's hardest architectural consequence, and it is not mine to decide.** Three shapes
exist, and the mechanism documents must choose one and put it to the owner:

1. **The facts travel.** A small `BodyFacts` record (gravity, insolation, `T_eq`, scale height, spin,
   obliquity, "has air") becomes an INPUT to `BodyDefinition::from_seed`, beside the radius, which is
   already an input from outside the fence (`body.rs:6-8`). This is an SL6 ask. **It carries a no-drift
   problem revision 1 missed, and refutation A named it:** MEASURED,
   `grep -c "powf\|ln()\|cos()\|sin()\|sqrt()" crates/physics/src/taxonomy.rs` returns 35. The forest's
   arithmetic is NOT under the float fence, so a gravity computed there is a platform-dependent `f64`
   today. If the shape reads it, the same planet can differ between an x86-64 pod and an aarch64 pod, and
   that is exactly the drift `just terrain-legs` and `D-TERRAIN-1` exist to catch. **Shape 1 is lawful
   only if every crossing fact is an integer or a fenced value**, and the world identity
   (`home.rs:16-23`, two literals under a zero-tolerance version, V6 Part D) must grow to include them.
2. **The facts move down.** The physical laws move into `vd-seed` or into the generator, and the forest
   reads them there — the same direction ruling **V6 Part D** already set for the radius (*"the generator
   crate owns a body's radius and the forest reads it"*, `owner_decisions_2026-09-07_voxels.md:203`),
   which is DECIDED and NOT YET BUILT (`body.rs:6-8` still calls the radius an input). Every law that
   moves must be rewritten under the fence, without `powf`, `ln` or `cos`.
3. **The generator re-derives them.** REFUSED here: two implementations of one law is exactly the drift
   SL10 exists to stop.

Shape 1 is the smallest change and it keeps one owner per law. Shape 2 is the cleanest and the most
expensive. I recommend that the mechanism documents cost both, and that either way the crossing datum is
fenced.

**And the boundary that actually binds is the CLIENT's, not the crate's.** Revision 2 framed this whole
question as a crate dependency rule, one boundary short of where it bites. Refutation A is right.
MEASURED, `crates/client/src/chunks.rs:266-289`:

```rust
let body = match (surface.frame, look) {
    (FrameRef::PlanetCentered { planet_seed }, Boundary::Shell { r }) =>
        BodyDefinition::from_seed(planet_seed, *r)
    …
```

The client builds its body from exactly two things that arrive **over the wire**: the planet's seed,
inside the surface statement's frame, and the look shell's radius. The statement is
`SurfaceStmt { frame, generator }` under `TAG_SURFACE` (`crates/core/src/look.rs:56-75`), stated by the
realm about itself, retained, once per realm on change.

So a `BodyFacts` record that shapes the ground must reach the client too, and that means four things at
once:

| What it needs | Where the law is |
|---|---|
| A new field on `SurfaceStmt`, or a new tag beside it | HR1: inter-shard bytes exist only as reviewed arms; SL6: ask before new data crosses |
| Room in the self-look bag | `SELF_LOOK_BUDGET_BYTES = 1200` for the WHOLE bag — outline, luma and surface together (`look.rs:79`). Seven facts as fenced 64-bit integers is 56 bytes, about 5 % of the bag (ESTIMATED) |
| A bit-stable postcard encoding | the two hosts must decode the identical bytes to the identical values |
| A place in the world identity the handshake refuses on | ruling V6 Part D, and the tolerance is ZERO (ruling V9 S5-2) |

Without all four, the client derives one planet and the server derives another — which is
`D-TERRAIN-1`'s whole subject. **This is the hardest architectural consequence in the document, and §9's
SL6 row now names the client lane by name.**

### 4.7 The weather, as far as it touches the ground

The owner said *"we also should simulate the weather"*. The weather lands on the terrain in five places,
and each one is measurable:

| Weather term | Plain words | What it puts in the picture |
|---|---|---|
| **Orographic lift** | Air must rise to cross a range. Rising air cools and drops its rain. | The windward side is green and cut by streams. The lee side is a **rain shadow** — dry, bare, sharp-edged. The reference picture is a rain shadow. |
| **Hadley cell / trade winds** | The big circulation that the spin breaks into belts. | Where the deserts and the rain belts sit, by latitude. |
| **Snow line** | Where snow lasts the year (§4.3). | The white caps on the skyline. |
| **Aridity** | Rain minus evaporation. | Whether a slope wears soil and forest, or bare rock and talus. It decides the SLOPE distribution as much as the rock does. |
| **Wind direction** | Where the sand goes. | Dune crests stand across the wind, and they all point one way. A dune field with random crest directions reads as wrong at once. |

**The split I recommend.** The moving weather — cloud, rain, storm, the hour of the day — is LIVE state,
and it belongs to the realm and to the client's atmosphere. The STANDING climate — the long-run
temperature, the long-run rain, the wind rose, the snow line — is a function of the seed and of the body's
facts, and it belongs in the generator, because the terrain's SHAPE depends on it (a rain shadow has a
different shape, not only a different colour). SL10 PERMITS that split in principle: the standing climate
is a function of `(seed, address)`; the storm over the player's head is not.

**"SL10 permits it" is not "it is free", and revision 2 priced it at zero.** Refutation A is right, and
the objection is exact. §5.6 is careful and honest about the river: *how much water passes here depends on
the whole basin above, and a basin can be continental*. **Orographic precipitation has the same shape.**
To know the rain at a column you must carry the air's moisture along the wind path from the ocean, over
every ridge the air already crossed — tens to hundreds of kilometres upwind. At rung 0 a 200 km fetch is
3 200 chunks in ONE direction. That is not a bounded stencil. It is the river's problem wearing a
different word, and it needs its own coarse-rung memory, its own stencil and its own halo, exactly as
§5.6 demands for the channel. §8.3 now prices it beside the channel rather than at zero.

**Game example.** The pilot flies over the lee side of a range. If the rain there is a local noise, the
green stops at a random line and he learns nothing. If it is a real shadow, it stops at the crest — and to
know where the crest was, the generator had to read two hundred kilometres of ground that this chunk does
not hold.

**The four questions the word "weather" opens, which revision 1 never asked** (refutation B, D11 —
accepted; all four go to the owner as §12 Q6):

1. **What crosses a realm boundary?** SL6 is default-NO and needs the ask written out: what data, from
   which realm to which, and why the receiver cannot compute it. A cloud over the home planet's shard,
   seen by a pilot in the star system realm — is that a placement, a look tag, or a new wire arm?
2. **Who authors it?** Under SL3 a realm draws itself, so the planet realm authors its own weather. Under
   SL1 the parent authors placements. A cloud is both a look and a position. The tension is unnamed.
3. **Does the weather change the SHAPE?** If the standing climate feeds the rivers of §5.6, then a storm
   that changed the rain would change the shape — and SL10 forbids the client deriving anything that
   depends on live state. The line between "climate shapes the ground" and "weather colours the ground"
   must be drawn explicitly, and drawn on the STANDING side.
4. **What does it cost, at what rate, in how many bytes?** Nothing is known.
5. **Can weather exist at all in a world where 99 % of the galaxy is off?** The dormant-world design keeps
   almost every realm shut down. A planet realm that is not running simulates nothing. So the weather a
   pilot sees on approach must EITHER be a function of `(seed, time)` that needs no running realm, OR the
   realm must spin up before any weather is visible. That is a hard fork, it is the same fork SL10 forces
   on the shape, and revision 2 never asked it. Refutation B found it.

**And the weather now gets its own decision row, not only a question.** Revision 2 gave the owner's second
sentence — *"we also should simulate the weather"* — a vocabulary table, a recommended split and four
questions, and it put no row in §11. Refutation B is right that a document whose stated job is the TARGET
left the owner's own second requirement with no target at all. **D17** asks the owner for the smallest
thing that counts as weather in the picture, so that a target can exist before a mechanism does.

---

## 5. How the industry and the literature do it

Each entry below is a REFERENCE, never a crate to adopt. CLAUDE.md's standing rule holds: we never adopt a
new library on our own; we research the options and the owner decides.

### 5.1 Games and engines

| Source | What it does | What we can take |
|---|---|---|
| **No Man's Sky** (Hello Games; GDC 2017, "Continuous World Generation in No Man's Sky") | One analytic "uber noise" per point: fBm plus ridged and billowed terms, plus domain warping, on the GPU. No global pass. | The proof that a purely per-point function can fill a galaxy at speed. Also the warning: players read the repeated motifs, and the planets have no rivers. |
| **Star Citizen planet tech v4** (Cloud Imperium; CitizenCon 2017/2018) | Hand-painted macro maps for the continent shape, procedural mid-scale detail, and material- and slope-driven asset scatter ("ecosystems"). | The ecosystem idea maps exactly onto V4's art-asset blend: pick the asset kit from slope, height and moisture. The hand-painted half is impossible for us (SL5, one world, no artist per planet). |
| **Outerra** | Real Earth elevation data, with fractal detail invented below the data's resolution and matched to the local statistics. | The technique our ladder needs in reverse: add detail under a coarse level without moving the coarse level. Their result also shows what a real drainage network is worth — their terrain looks right because Earth's does. |
| **Elite Dangerous** (Frontier) | Per-point procedural over a whole galaxy, with crater and erosion-like operators. | The scale proof, and the "same but different" complaint that follows a scale-free recipe. |
| **Space Engine** | Per-planet-class noise recipes: craters, dunes, ridged multifractals, canyons, by body type. | Its per-class recipe table is close to what §4 asks for: the body's facts pick the recipe. |
| **Parallax** (a Kerbal Space Program mod) | Asset scatter and terrain shading over a procedural base. | The reference for V4's "assets blended dynamically", and for how much of "believable" is the scatter and not the shape. |

**Game example for this section.** Every one of these worlds solves the problem our home planet has: the
pilot must be able to say "I will land beside THAT ridge, not the other one". No Man's Sky solves it with
motifs, Star Citizen with a painted map, Outerra with real Earth data. We may use none of the three, which
is why §5.6 matters.

### 5.2 The literature

| Work | What it gives | Why it matters here |
|---|---|---|
| Musgrave, Kolb, Mace, *The Synthesis and Rendering of Eroded Fractal Terrains*, SIGGRAPH 1989 | Thermal and hydraulic erosion on a height grid; the ridged multifractal. | The origin of "fractal terrain looks wrong until you erode it". Thermal erosion is what makes a talus slope at the angle of repose. |
| Ebert, Musgrave, Peachey, Perlin, Worley, *Texturing & Modeling: A Procedural Approach* | Ridged multifractal, multifractal with a local dimension, domain warping. | The cheapest way to buy ridges and a varying roughness inside a per-point function. Our recipe uses none of it. |
| Quílez, *fBm with analytic derivatives* (the "Elevated" technique) | Damp each octave by the slope built so far, with the noise's own derivative. | The single highest-value per-point trick: flat valley floors and sharp ridge crests from a closed form, at the cost of one derivative per octave. It attacks §1.4 directly. |
| Whipple & Tucker, *Dynamics of the stream-power river incision model*, JGR 1999 | The **stream power law**: `E = K·A^m·S^n`. | It predicts the concave river profile and the **slope–area law** `S ∝ A^(−θ)` with θ about 0.4–0.6. That law is our best single acceptance proxy (§6.4). |
| Perron, Kirchner, Dietrich, *Spectral signatures of characteristic spatial scales and nonfractal structure in landscapes*, JGR 2008; *Formation of evenly spaced ridges and valleys*, Nature 2009 | Real landscapes have a spectral BREAK at the valley spacing, and the spacing comes from soil creep balanced against channel cutting. | It says the target cannot be reached by any fractal — which §1.6 has now measured on our own field. It also gives us a number to measure: the break wavelength. |
| Génevaux, Galin, Guérin, Peytavie, Benes, *Terrain generation using procedural models based on hydrology*, SIGGRAPH 2013 | Build the RIVER NETWORK first, as a tree, then build the ground around it. | The most promising shape for us: a tree can be grown from the coarse rung downward (§5.6), IF the growth rule is a pure function and the stencil is bounded. |
| Cordonnier, Braun, Cani, Benes, Galin, Peytavie, Guérin, *Large scale terrain generation from tectonic uplift and fluvial erosion*, Eurographics 2016 | Solve uplift against stream-power erosion on a graph until it settles. | It gives the right ridge and valley spacing from physics, and it is the only entry that models the BUILDING half of §4.1. It is a GLOBAL iterative solve, which is where SL10 bites. |
| Schott, Paris, Fournier, Guérin, Galin, *Large-scale terrain authoring through interactive erosion simulation*, ACM TOG 2023 | Erosion fast enough to be interactive at large scale. | The state of the art in cost, and a source for how few iterations are enough. |
| Guérin, Digne, Galin, Peytavie, *Sparse representation of terrains*, Eurographics 2016 | A terrain as a small set of atoms plus a residual. | The shape of our edit pyramid: a coarse summary plus a sparse difference. |
| Guérin et al., *Interactive example-based terrain authoring with conditional GANs*, SIGGRAPH Asia 2017 | Learn the landform statistics from real terrain. | **REFUSED for us by SL10**: a neural network is not byte-identical across hosts, and it cannot pass the float fence. It is useful only as an offline authoring tool, whose output would then have to ship. |
| Hack, *Studies of longitudinal stream profiles*, USGS 1957; Horton 1945; Strahler 1957 | Hack's law `L ∝ A^0.57`; bifurcation ratios of 3–5. | Cheap statistical checks on any channel network we build. |
| Kelley, Malin, Nielson, *Terrain simulation using a model of stream erosion*, SIGGRAPH 1988 | The first stream-network terrain in graphics. | The historical proof that the network-first order works. |

### 5.3 The comparison

Scored against what this project needs. "Per-point" means a chunk can be computed from `(seed, address)`
with a bounded stencil. "Global" means the whole body must be solved before any chunk is right.

| Approach | Believable | Exploration interest | Per-point or global | Deterministic on two hosts | Cost per chunk | Authoring and override |
|---|---|---|---|---|---|---|
| fBm sum (**ours today**) | low — no landforms (§1.4, §1.6 measured) | low — nothing to name | per-point | yes, our golden gate proves it | 3.3 ms surface, 6.11 ms cave-dense (MEASURED) | the edit pyramid only |
| Ridged multifractal + domain warp | medium — ridges and cliffs appear | medium — the motifs repeat | per-point | yes | about 1–2× fBm | the same |
| Slope-damped fBm with derivatives (Quílez) | medium-high — valley floors flatten, crests sharpen | medium | per-point | yes (one extra derivative per octave, all `+ − × ÷`) | about 2× fBm (ESTIMATED, U7) | the same |
| Uber-noise (No Man's Sky) | medium-high | medium | per-point | yes | about 2–4× fBm (ESTIMATED) | the same |
| Hand-authored macro + procedural detail (Star Citizen, Outerra) | **high** | high | it needs a shipped map | yes, but the map must ship | cheap per chunk, big download | the best of all |
| Grid hydraulic and thermal erosion (Musgrave; Mei) | high | high | **global** per body | only if the iteration order is fixed and both hosts run it | very high; not per chunk | poor |
| Stream-power uplift + erosion (Cordonnier) | **highest** | highest | **global** per body | the same problem | the highest | medium |
| Hydrology-first network (Génevaux) | high | **highest** — real rivers to follow | global as published, **and the top-down growth is UNPROVEN for us** (§5.6) | only if the growth rule is a pure function with a bounded stencil | UNMEASURED | good: a river is an object you can move |
| Sparse atoms (Guérin 2016) | medium | medium | per-point with a bounded atom list | yes | low | **the best** — an atom is an authored feature |
| Learned / GAN (Guérin 2017) | high | high | global, and a network | **no** — SL10 refuses it | not applicable | not applicable |

### 5.4 The verdict this document draws

Three statements follow from the table and from §1.6. None of them is a mechanism decision.

1. **A better noise is worth taking, and §1.6 measures that it is not enough.** Ridged multifractals,
   domain warping and slope-damped fBm are cheap, per-point, and they pass every law we hold. They would
   give cliffs, ridges and flat valley floors, which is most of §1.4's gap. They cannot give a drainage
   network, so they cannot give the slope–area law or the valley spacing — and raising the roughness of a
   plain fBm buys 30 % of the planet over 45° for 0.8 skyline breaks per 60°.
2. **A global solve gives the rest, and it collides with SL10.** A stream-power solve over the home
   planet's 141 million km² is not a per-chunk computation. Both hosts would have to reproduce it byte for
   byte, or the server would have to ship its result (§5.5).
3. **The ladder is the most promising way out, and §5.6 states honestly what is unproven about it.**

### 5.5 What shipping a solved map would cost

If a global solve ran once per body on the server and shipped its result, the raw size is fixed by the
resolution (ESTIMATED, two bytes per sample over 141.09 million km²):

| Resolution | Samples | Raw at 2 bytes | With a 3–6× compression (UNMEASURED) |
|---|---|---|---|
| 8 km | 2.2 million | 4.4 MB | 0.7–1.5 MB |
| 4 km | 8.8 million | 17.6 MB | 3–6 MB |
| 1 km | 141 million | 282 MB | **47–94 MB** |
| 100 m | 14.1 billion | 28 GB | 5–9 GB |

Revision 1 wrote "282 MB, that is not shippable" and stopped. Refutation B is right that this is a
conclusion, not a measurement. Three things must be said instead:

- **Compression is not measured.** A 1 km integer height map delta-codes well; a 3–6× factor is routine
  on real elevation data. Nobody has measured ours. Until somebody does, 282 MB is an upper bound.
- **The whole sphere is the wrong unit.** 141 million km² of the home planet will never be walked (§12
  Q8). A shipped 8 km map for the whole body plus a 1 km map for the basins players actually reach is a
  third option, and it is priced nowhere.
- **A shipped map is still a shipped map.** It moves the target from "derive" to "download", changes the
  world identity, and needs a store. That is a bigger architectural change than its size suggests.

A valley spacing of 0.5–5 km needs a 1 km map at least. The honest statement is therefore: **a whole-body
1 km map is 47–94 MB compressed (UNMEASURED), which is large but not obviously impossible, and a derived
network ships zero bytes.** §11 D14 asks whether shipping is on the table at all.

### 5.6 The ladder as the memory: what is promising, and what is unproven

The trouble with a drainage network is that "how much water passes here" depends on the whole basin above,
and a basin can be continental. So it is not a bounded local question — **unless the network is BUILT from
the top down.**

Our ladder has that shape. A rung-11 chunk is 127 km across; a rung-0 chunk is 62 m across. Let the coarse
rung decide the big channels, and let each finer rung refine only the channels its parent handed it,
inside its own area:

```text
   rung 11   ▁▁▁╲▁▁▁▁     the big channels, decided on 2 048 m cells
                  ╲
   rung 10   ▁▁▁▁╲▁▁▁▁    the child refines them inside its own area
                   ╲╲
   ...              ╲╲
   rung 0    ▁▁▁▁▁▁▁▁╲╲   the brook the player steps over
```

**What is sound.** The ANCESTOR COUNT is bounded: a rung-0 chunk has eleven ancestors, and each is shared
— one rung-1 chunk serves 4 rung-0 chunks, one rung-2 chunk serves 16, one rung-11 chunk serves
4¹¹ = 4.2 million. The amortised ancestor work with a perfect cache is `Σ_{L=1..11} 4^(−L) = 0.333`
ancestor evaluations per rung-0 chunk (ESTIMATED; revision 1 said 1.33, which wrongly counted the chunk
itself — both refuters caught it). The hierarchy really does nest 4:1: `N = 2570 × 2¹¹`, so every rung
divides the face evenly.

**What is unproven, and revision 1 asserted it.** Three things, and each one can sink the design:

1. **The coarse rung is not one chunk.** MEASURED from `ladder.rs:109` (`cells_per_edge(rung) = n >> rung`)
   and `CHUNK_EDGE = 62`: rung 11 is 2 570 cells per face edge, so 42 chunks per edge, so **10 584 chunks
   over the body** — 39.6 million coarse cells. The top of the ladder faces the same question the bottom
   does: how much water passes through THIS coarse cell depends on every coarse cell that drains into it,
   and a coarse basin can span a hemisphere. **A mechanism must state the coarse chunk's stencil.** If the
   answer is "the basin", the rule is not a function of `(seed, address)` and SL10 refuses it as written
   (§12 Q3).
2. **Siblings, not only ancestors.** A channel crosses a chunk edge. A rung-L chunk that refines its
   parent's channel must agree with its neighbours at the shared edge, or the brook stops at the chunk
   border — an SL8 seam this document forbids one page earlier. A 3 × 3 halo at every rung multiplies the
   0.333 by up to 9. The extractor already pays a halo and measured it: **35 % over the bare chunk at
   rung 0** (MEASURED, `slice_06_extractor.md:355`).
3. **The cube-face seam and the eight corners.** The present height field has NO seam problem, because
   `height.rs:16-27` evaluates 3-D noise at `dir × frequency`, which is continuous across every face. A
   chunk-grid mechanism is not. Slice 6 measured the corner cells at 60°–120°, and the corner was a HOLE
   until it became a barycentric prism (ruling V10). Flow routing on that grid needs the metric, and the
   three faces at a corner must agree byte for byte. Ruling V6 A5 also puts a LANDFORM at every cube
   corner, which is the one place where three faces must agree on one shape.

**Game example.** The player walks a brook at rung 0 on face `+Y`. The brook must join a river that face
`+Z` also draws, or the pilot flying the seam sees a river end in mid-air. That is the worst seam this
whole investigation could ship.

**A second customer for the same memory.** The rain shadow of §4.7 is the same problem: the moisture at a
column depends on hundreds of kilometres of ground upwind. If the ladder memory works, ONE mechanism
serves both, and the wind field is simply a second quantity carried down the same rungs. If it does not
work, the target loses the channel network AND the rain shadow together. That doubles what U3 settles and
it is the strongest argument for spiking the ladder memory before anything else.

**The honest statement.** The ladder is the most promising shape for a network under SL10, and it is not
yet a feasibility result. §8.3 prices both ends of it. §10 U3 names the spike that settles it. Revision 1
called this "the single most important feasibility statement in this document"; it was an assertion, and
it is demoted here to a candidate with three named unknowns.

---

## 6. The acceptance proxies: how a gate measures "believable"

An **acceptance proxy** is a number a program computes, which stands for a judgement a person would make.
The owner should never have to say "it looks wrong" twice about the same defect. Eight proxies follow.
Each one states what it measures, how a bench computes it, what Earth gives, what our world gives today,
and whether I recommend it as a GATE (the build goes red) or a REPORT (the number is printed and a person
reads it).

**Two rules for the whole section.**
- **Every proxy reads the crate's own `height_m`, `biome_at` or extractor.** None of them reads a rendered
  frame. A picture gate would also measure the renderer, the exposure, the star field and `D-TERRAIN-3`,
  and it could not run in `cargo test --workspace`, which CLAUDE.md defines as the fast deterministic
  suite with no sockets. §1.1 is the evidence for that rule.
- **Every threshold below is a PROPOSAL.** Under the no-magic-numbers rule a threshold lives in the GATE,
  never in the shipped recipe. §6.9 says honestly which bands have a law and which do not.
- **A proxy states its sample size, and reads a SHARE, never an extreme.** A sampled maximum is a lower
  bound on the field's maximum and grows with the sample (§1.4). "Zero columns over 60° in 10⁶" is
  reproducible; "the maximum is 8.7°" is not.
- **A proxy's VERDICT must be host-independent**, and how a red proxy LANDS is a decision of its own.
  §6.11 answers both, and D2 puts the landing shape to the owner.

### 6.1 P1 — The skyline break count (the headline number)

**What it measures.** The owner's exact complaint: "no orienters". A **silhouette break** is a place where
the skyline steps up or down enough for the eye to name it.

**How to compute it, without a GPU.** From a camera 373 m above the surface at a station the RULE of §7.1
names, for each of 360 bearings, step outward through `height_m` from 300 m to **360 km** and take the
largest elevation angle:

```text
   for a point at ground distance s, from a camera at radius r_c:
       θ    = s / R                                (the angle around the planet's centre)
       elev = atan2( r_t·cos θ − r_c , r_t·sin θ ) (r_t = height_m at that direction)
   skyline(bearing) = max over s of elev
   a BREAK = a local maximum of skyline that stands 0.5° over the median of
             skyline within ±5° of bearing
```

**Three parameters this gate may not leave unstated, and revision 2 left all three unstated.**

1. **The station**, as a face and a rung-0 cell pair, chosen by the §7.1 rule and printed. Refutation A
   showed the number moves by a factor of two between cameras; §1.4a shows 0.054°–0.067° over four.
2. **The march limit, 360 km**, with the convergence measured. Refutation B showed 90 km truncates, and
   §1.4a measured a 23 % error at one of four stations.
3. **The bearing count and the step**, because the break rule reads a local maximum and a coarse bearing
   grid loses one.

This is a pure function of `(seed, address)`, it runs in the fast suite, and it measures the shipped
function — which is what §9's SL10 row promises and what revision 1's picture-reading gate did not
deliver.

**The result today** (REPLAYED, §1.4a: four named stations, 360 bearings, marched to 360 km):

```text
   skyline of the reference               skyline of our world, measured
      ▁▃▅█▇▅▃▁   ▁▂▄▆▅▃▁                    ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
      ▲  ▲   ▲   ▲  ▲                       (one smooth arc; the whole 360°
      "8–12 orienters in 60°",               range spans about 2.5° of elevation)
      counted BY EYE — a different           0 breaks at 0.10° and above;
      quantity, see §2.2                     1–3 over the WHOLE horizon at 0.05°
```

| | Value |
|---|---|
| Reference picture, counted by eye | 8–12 per 60° — **UNMARKED**, and not the same quantity as the row below (§2.2) |
| **Our world today, computed** | **0 per 60° at every threshold at or above 0.10°**; at 0.05° the whole horizon holds 1–3 |
| The largest rise over the local trend | **0.054°–0.067°** across the four stations |
| A full moon, for scale | 0.52° across (PUBLISHED) — the biggest feature on this horizon is an eighth of it |
| Proposed gate | **a RATCHET**: neither the break count nor the largest rise may fall below the last recorded value at any of the four stations. **No numeric floor until U16 measures the reference with the same rule.** |
| Recommendation | **GATE, as a ratchet** |

**Why a ratchet and not a floor.** A gate needs a subject, an instrument and a band. §1.4a fixes the
subject and the instrument. The band cannot be honest yet, because its only calibration is a number
counted by eye off another game's screenshot — the exact instrument §1.1 forbids one page earlier. A
ratchet needs no band, it catches every regression, and it converts to a floor the day U16 lands.
Refutation A and refutation B both found this from different directions, and both are right.

**Why this proxy first.** It is cheap, it reads the shipped field, and it is red today by any reading: the
largest feature on the whole horizon is an eighth the width of the moon. A gate that cannot fail proves
nothing; this one fails now and would pass if the mechanism worked.

### 6.2 P2 — The slope distribution

**What it measures.** Whether cliffs, ridges and flat floors exist at all. Slope depends on the baseline
you measure it over, so the baseline must be named every time.

**How to compute it.** Sample the crate's own `height_m` over 10⁶ columns of the home planet. Take the
finite difference at 1 m, 10 m and 100 m baselines. Print the median, the p90, the p99, the maximum and
the share over 45° and 60°.

| Baseline | Earth land (PUBLISHED, typical) | **Our planet today (REPLAYED, 200 000 columns)** | Proposed band |
|---|---|---|---|
| 100 m | median 2–5°, p99 about 35° | median **1.32°**, p99 **4.97°**, max **8.24°** | median inside 2–6°, p99 ≥ 25° |
| 10 m | median 3–8°, and a heavy tail | median **1.36°**, p99 **5.09°** | p99 ≥ 40° |
| 1 m | cliffs give 70–90° | median **1.36°**, share over 60° = **0 of 200 000**; the steepest ground FOUND by hill climbing is 11.15° | the share over 60° must be > 0 |

Recommendation: **GATE** on "the share over 60° at a 1 m baseline is greater than zero", which is the
"can a cliff exist" test, and **REPORT** for the rest until a mechanism lands.

**One rule this proxy must carry.** A sampled maximum is a LOWER BOUND on the field's maximum and grows
with the sample (§1.4). So the gate must read a SHARE over a stated sample size, never a maximum: "zero
columns over 60° in 10⁶" is a reproducible statement, and "the maximum is 8.74°" is not. Both refuters
found revision 2 breaking that rule, and the same discipline the document demands of a bound applies to
an extreme.

**Game example.** The player walks to the foot of what the map calls a cliff. On today's planet the
steepest ground anybody has found tilts at 11°, which is a ramp, not a wall.

### 6.3 P3 — The hypsometry

**Hypsometry** is the distribution of heights: how much of the surface sits at each altitude. Earth's is
famously two-humped — the continents at about +840 m mean, and the ocean floor at about −3 800 m — with a
steep continental slope between them.

```text
   share of the surface                        ours, measured:
        │        ▄▄                                 │      ▄██▄
        │       ████            continents          │    ▄█████▄
        │  ▄▄   ████                                │  ▄█████████▄     one hump,
        │ ████▄▄████                                │▄█████████████▄   skew −0.03,
        │ ██████████ ▄▄▄▄▄▄     ocean floor         │                  kurtosis 2.80
        └──────────────────────► altitude           └──────────────────► altitude
         +8 km    0     −11 km                       the sea is at the far left
```

**How to compute it.** Sample 10⁶ directions uniformly on the sphere, take `height_m` at rung 0, subtract
the sea radius, and histogram.

**The result today** (REPLAYED, 200 000 directions): skew **−0.031**, kurtosis **2.795**, where a Gaussian
gives 0 and 3. **One hump, no shelf, no plain, no mountain tail**, exactly as fourteen independent noises
and the central limit theorem predict. And 0.99 % of the surface is under the sea, so there is almost no
coast.

A believable Earth-like planet needs at least: a continental shelf, a plain that is flat over tens of
kilometres, and a tail of high ground that is a small share of the area.

Recommendation: **REPORT** now, **GATE** once a mechanism claims to produce a shelf.

### 6.4 P4 — The slope–area law (the strongest single test)

**Flow accumulation** at a point is how much ground drains through it. **Drainage density** is the total
channel length per square kilometre.

The stream power law predicts `S ∝ A^(−θ)` — the more water passes, the gentler the slope — with the
**concavity** θ between 0.4 and 0.6 on Earth (PUBLISHED). Plotted on log axes it is a straight line over
three or four decades of drainage area.

```text
   log slope
      │ ●●●
      │    ●●●●            Earth: a straight line, θ ≈ 0.5
      │        ●●●●
      │            ●●●●
      │
      │  ○ ○  ○ ○ ○  ○ ○   fBm: a shapeless cloud, no relation at all
      └──────────────────────► log drainage area
```

**How to compute it.** Take a patch of the home planet — say 200 km × 200 km at rung 3 — route flow
downhill by the steepest descent, accumulate the area, then fit `log S` against `log A`.

| | Value |
|---|---|
| Earth | θ = 0.4–0.6, a straight fit over 3–4 decades |
| Our planet today | UNMEASURED; the prediction is "no relation", because the field knows nothing about downhill |
| Drainage density, Earth | 2–5 km per km² in humid temperate land (PUBLISHED) |
| Proposed band | θ inside 0.35–0.65, with a fit over at least two decades |

**Game example.** The brook beside the player's camp runs steeply. The river at the delta runs almost
flat. On a believable planet the ratio between them is the same everywhere, and that ratio IS θ.

Recommendation: **REPORT** now — it would be red today and no mechanism yet claims it. It becomes a
**GATE** the moment a mechanism claims a channel network. One honest caveat: a flow router is itself a new
model, so the bench must be checked against a synthetic field with a known answer before it judges the
world (§9).

### 6.5 P5 — The spectrum and its break

**How to compute it.** Sample `height_m` along a great circle at 1 m spacing over 100 km, take the power
spectrum, and fit β over each decade. Then look for a break: a wavelength where the slope of the fit
changes.

| | Value |
|---|---|
| Earth | β ≈ 2 (range 1.6–2.4), with a break at the valley spacing (0.05–5 km) |
| Our planet today | **β = 3.19 for the home planet, 2.72–3.30 across bodies**, no break, by construction (REPLAYED) |
| Proposed band | β inside 1.6–2.6 over the 10 m–10 km decades, and a detectable break |

**Recommendation: REPORT, not GATE. Revision 2 recommended a gate and it was wrong.** Both refuters
showed why, and their argument is the document's own §1.6:

- **A gate on β would PASS the worst answer.** §1.6 measures that `k_rough = 0.707` reaches β ≈ 2 and puts
  **30.1 %** of the surface over 45° with **0.8** breaks per 60°. That is the unwalkable gravel heap. A
  gate must be able to fail the bad answer; this one rewards it.
- **A gate on β would also FAIL a good answer.** A channel network buys landform without moving the
  spectrum much, so a mechanism that works could still read outside the band.
- **The band is unreachable by every body at once.** `k_rough` is drawn in `[0.45, 0.55)`
  (`body.rs:155`), so β runs 2.72–3.30 across every body the world can draw (§1.5). A band of 1.6–2.6 is
  red for every planet in the universe on the day it lands, which makes it a version-bump alarm and not a
  landform test.

β stays a printed number beside the break wavelength. It is necessary and nowhere near sufficient.

### 6.6 P6 — The landform census

**What it measures.** The density of things worth walking to.

**How to compute it.** Over a 100 km × 100 km patch, count:
- peaks that stand at least 300 m above their surroundings within 5 km (their **prominence**);
- cliff segments with over 30 m of vertical over under 50 m of horizontal;
- closed basins (a hole with no outlet);
- channel junctions.

**A closed basin is not a lake site on this world, and revision 2 called it one.** Refutation A is right:
a body draws ONE `sea_radius_m` and nothing else holds water (§3.5 item 4), so a hole above that radius
holds no water at all. The proxy counts HOLES. It becomes a lake count only if the world gains a per-basin
base level, which is D19's question and exists nowhere today.

| Feature | Earth, humid mountain land (PUBLISHED order of magnitude) | Ours today | Proposed floor |
|---|---|---|---|
| Peaks over 300 m of prominence | 5–40 per 100 km × 100 km | UNMEASURED; the steepest ground found anywhere is 11.15° (§1.4), which makes a 300 m prominence within 5 km implausible | ≥ 5 |
| Cliff segments | thousands | **0 in 200 000 columns** (REPLAYED, §1.4) | ≥ 1 |
| Closed basins | 1–50 | UNMEASURED, likely 0 | ≥ 1 |
| Channel junctions | thousands | 0 | ≥ 100 |

Recommendation: **REPORT** for the whole census. Its "a cliff exists" line is already P2's gate (§6.2),
and it is stated there rather than twice, because a threshold in two places is two thresholds. (Revision 1 printed "0, provably" over an unproved bound. The
number is now a measured zero over 200 000 columns, which is a different and honest claim.)

### 6.7 P7 — The climate proxies: does the body's own physics show on the ground?

Revision 1 proposed six proxies and every one measured SHAPE. The owner asked for two things, and the
second — biomes that depend on the planet's position, spin, trajectory, size and gravity, plus the weather
— had no proxy at all. Both refuters caught it. Four cheap proxies close the gap, and every one reads
`biome_at` and `height_m` directly:

| # | What it measures | How | Earth's answer | Ours today |
|---|---|---|---|---|
| P7a | **The desert belt sits where the spin puts it.** | Sample 10⁶ directions, histogram the biome against latitude, and compare the desert share's peak with the **Hadley cell edge** the body's own facts predict (below). | a peak at ±30° | **cannot pass**: no body draws a spin (§4.5), and `biome_at` reads two slow noises |
| P7b | **The snow line sits where the lapse rate puts it.** | The height at which the biome turns to `Tundra` (whose topsoil is `Stratum::Snow`), against `(T_surface − 273)/Γ` with `Γ` from the body's own gravity. | about 4.9 km in the tropics | **the shape is computable; the law is not**: `biome_at` already cools with height (`height.rs:52-53`), so a white cap draws today — but no `T_surface` and no `Γ` reach the generator (§4.6) |
| P7c | **A range has a rain shadow.** | Along a wind bearing, compare the moisture and the biome on the windward and lee sides of the twenty tallest ridges. | a sharp step at the crest | **cannot pass**: no wind, no moisture, no ridges |
| P7d | **Two planets of one size differ by their orbit.** | Run the census on the planets nearest and farthest from the star in one system. Their biome histograms must differ. | — | UNMEASURED, and near certain to be "they look the same" |

**P7a's law, corrected — and revision 2's was not a law at all.** Revision 2 wrote *"the latitude
`f = 2Ω sin φ` predicts"*. Refutation B is right that this is nonsense as written: `f = 2Ω sin φ` is the
**Coriolis parameter**, the sideways acceleration per unit speed that a spinning planet gives to moving
air at latitude φ. It is a function OF the latitude; it does not have a latitude as its output, and no
rearrangement gives one. A programmer handed that row could compute nothing.

The quantity the target wants is the **Hadley cell edge**: the latitude at which air that rose at the
equator sinks again. The published scaling (Held and Hou, 1980) writes it through the **thermal Rossby
number**

```text
   Ro_T = g·H·Δθ / (Ω²·a²)          φ_H ≈ sqrt(5·Ro_T/3)

   g  the body's surface gravity        Ω  the body's spin rate
   H  the depth of the weather layer    a  the body's radius
   Δθ the equator-to-pole warmth contrast, as a fraction
```

Every input is already in §4's list. ESTIMATED with Earth's own numbers (g = 9.81, H = 15 km, Δθ = ⅓,
Ω = 7.292 × 10⁻⁵, a = 6 371 km): Ro_T = 0.227 and φ_H = **35.3°**, against the observed ±30°. Doubling the
spin gives 17.6°; halving it drives φ_H past 90°, which IS the one-cell planet §4.5 describes. So the
scaling has the right inputs and the right direction, and it is **10–20 % off Earth's own answer**, which
is why §13's rule binds: **nobody may turn it into a numeric gate before the paper is opened** (U12).

Recommendation: **REPORT** on all four, and make P7d a **GATE** as soon as any body's facts reach the
generator, because "two planets look the same" is the exact failure the owner's sentence is about.

### 6.8 P8 — The corner landform and the bedded cliff

Two things the target owes to a ruling and to the code, and revision 1 named neither.

- **Ruling V6 A5, approved and binding:** *"the generator always places a landform at all eight cube
  corners."* The reason is structural: three squares meet at a cube corner at 120°/60°, a prefab stamp
  refuses there, and the owner chose a landform so that no flat build site exists at latitude ±35.2644°.
  **The proxy:** at each of the eight corner addresses of the home planet, the census of §6.6 must return
  at least one landform, and the three faces must agree on the shape byte for byte. Recommendation:
  **GATE**, because it is exact, it is cheap, and it is already decided.
- **Bedding.** **The proxy:** on any cliff face the census finds, print how many distinct substances the
  face exposes and at how many distinct RADII a substance changes. Two numbers, because the second is the
  one that separates bedding from a drape. **Today the first is bounded at FOUR by construction and the
  second is bounded at ZERO** (§3.5 item 3): a body draws one sediment and one bedrock, and every boundary
  sits at a constant depth under the surface rather than at a radius. So the proxy already has its answer
  and it is the wrong one. Recommendation: **REPORT**, and understand it as a measurement of new work, not
  of an unused asset.

### 6.9 Which bands have a law, and which do not

D3 in §11 recommends per-body bands derived from the body's facts. §4 gives a law for only two of them,
and honesty requires saying so plainly (refutation B's W5 — accepted):

| Band | Is there a law from the body's facts? |
|---|---|
| The relief ceiling | **yes** — `h ≤ σ/(ρ_c g)` (§4.2), as a bound, not a value |
| The snow line | **partly** — `Γ = g/c_p` gives the gradient; the surface temperature and the aridity are missing (§4.3) |
| The concavity θ | **yes, and it is not Earth's** — θ ≈ 0.5 is a property of the stream power law itself, so it transfers to any body with liquid flow |
| The spectral exponent β | **no** — it depends on the erosional agent (water, ice, wind, or nothing) and on gravity, and no published law gives it as a function of a body's facts |
| The slope distribution | **no**, for the same reason |
| The peak density and the skyline break count | **no** — they are properties of the uplift history, which the world does not model |
| The desert latitude | **partly** — the Held–Hou scaling `φ_H ≈ sqrt(5·Ro_T/3)` takes the body's own gravity, radius, spin and weather-layer depth, and it lands 10–20 % off Earth's own answer (§6.7). It is a law with the right inputs and an unverified constant, **not** the `f = 2Ω sin φ` revision 2 printed, which is not a law for a latitude at all |
| The skyline break count's BAND | **no**, and worse: its only calibration is a count made by eye off another game's screenshot (§2.2). This is why P1 lands as a ratchet |

So four of the eight proxies must gate on Earth-calibrated constants for now, one has no honest band at
all, and this document marks them as such rather than pretending a law exists. Closing that gap is a
literature question (§10 U12) and a tracing question (§10 U16).

### 6.10 The proxies against the laws we already gate

These four already exist and must not weaken:

| Existing gate | Where | What the target must not break |
|---|---|---|
| Byte-for-byte no drift on every target | `just terrain-pin`, `just terrain-legs`, `D-TERRAIN-1` | Every new field must be `Gf` arithmetic under the fence. |
| The rung agreement bound | `crates/terrain/src/height.rs` test; `body.rs:274 dropped_bound_m` | A rung-L surface must stay inside a STATED bound of rung 0. A new mechanism must state its own bound (§12 Q4). |
| The 8 ms per-chunk budget | `crates/bins/examples/terrain_cost.rs:41,357` — the gate holds the COSTLIEST named chunk, at every rung | §8. |
| 100 % region and branch coverage in Tier-A | `just coverage-fast` | Every new branch in the generator needs a test. |
| **A gate's VERDICT must be host-independent** | new; `just terrain-legs` is the precedent | §6.11 |

### 6.11 Where a proxy lives, how a RED gate lands, and why its verdict must be host-stable

Revision 2 recommended four numeric gates that all fail today and never said where the code lives or how
it lands. Both refuters called that out and both are right. Three answers are owed before D2 can be acted
on.

**1. A gate that cannot pass blocks every merge.** `justfile:408` lists the pre-merge gate:
`fmt-check lint lint-combos terrain-pin … coverage`. A gate that is red on the day it lands makes
`just gate` red for every unrelated change until the whole landform mechanism ships — which, by §5.6, is
unsolved. The document says *"a gate that cannot fail proves nothing"* and never asked the opposite
question. Three landing shapes exist, and the tree already uses two of them:

| Shape | What it does | Where the tree already does it |
|---|---|---|
| **A RATCHET** | The number may not get WORSE than the last recorded value. Green today, red on a regression. | This is what P1 becomes (§6.1). |
| **An expected-red with a named closing slice** | A `DEFERRED.md` row states the red, names the slice that closes it, and the recipe skips it until then. | `D-TERRAIN-1..4` are exactly this shape. |
| **A hard floor** | Red until a mechanism lands. | Only lawful for P8's corner landform, which is decided and small. |

**Recommendation:** P1 and P2 land as ratchets, P8 lands as a hard floor, and every band that waits on a
measurement gets a `DEFERRED.md` row. Note also that `terrain-cost` and `terrain-legs` are NOT in
`just gate` today, so adding any of this is itself a change to the pre-merge recipe, which D2 must name.

**2. Where the code lives decides two laws.** If a proxy lives in `crates/bins/examples` beside
`terrain_cost.rs`, it runs only under `just`, NOT under `cargo test --workspace` — which contradicts §6's
own reason for refusing a picture gate. If it lives in `vd-terrain`, it is Tier-A and every branch of a
histogram, a percentile and a log–log fit needs 100 % region and branch coverage (HR5), which is real
work. **Recommendation:** the proxies live in `crates/bins/examples` as a bench, and only the RATCHET
comparison — a handful of branchless comparisons against a recorded file — lives anywhere a coverage gate
reads. The bench's own run cost is small: ESTIMATED at release speed, P2 over 10⁶ columns is about 1.1 s,
P3 about 0.2 s, P1 about 0.5 s at the 360 km limit. **But `cargo test --workspace` builds DEBUG**, and
this project's own record says debug binaries miss deadlines by a wide margin (refutation B, N3). So the
bench states `--release` or it states a sample count a debug build affords. Neither is free, and the
document does not choose for the owner.

**3. A gate's verdict must be host-independent, and nothing said so.** `just terrain-legs` compares two
architectures byte for byte because `D-TERRAIN-1` exists. The proposed proxies are threshold verdicts
computed from UNFENCED `f64`: P1 uses `atan2`, P4 a log–log fit, P5 a spectrum. Two legs may compute 5.99
and 6.01 breaks at a band edge and disagree about the build's colour. **The rule this document adds:** a
proxy's VERDICT must be a function of quantised inputs, or its margin must be stated and must exceed the
platform's own variance. This is cheap now and expensive to discover later. Refutation A found it.

---

## 7. What a judgeable picture must contain

The owner's complaint was partly about the pictures themselves: a flat brown frame with no scale reference
tells nobody anything. In revision 2 the picture is an ILLUSTRATION and a regression set. §6 holds the
instruments.

### 7.1 The named camera set

Seven fixed stations on the home planet. The set is a DEV tool: it reads THE world (SL5), it makes no
variant, and it ships in no product binary.

**Every station's address is a CONSEQUENCE of a rule, never a choice.** Refutation B is right that a
hand-picked ridge lets a future mechanism be fitted to its own answer. The rule for the ridge station,
which §1.4a already ran:

```text
   THE RIDGE STATION RULE
   1. Name a face and a face-parameter centre (a0, b0)  ← this is the only choice, and it is printed
   2. Take the 100 km patch around it: |a − a0| and |b − b0| under 50 000 / (R·k1)
   3. Sample a 101 × 101 grid of columns through height_m at rung 0
   4. The station is the column of GREATEST surface radius
   5. Print the face and the rung-0 cell (i, j) it lands on
```

Four patches give the four stations §1.4a measured, and they are the same four the gate reads:

| Station | Face | Rung-0 cell (i, j) | Height over the ladder radius |
|---|---|---|---|
| S2-a | PosX | (3 412 184, 3 006 432) | +2 294.8 m |
| S2-b | PosY | (2 131 344, 3 734 351) | +4 624.3 m |
| S2-c | PosZ | (2 770 264, 1 730 591) | +1 247.2 m |
| S2-d | NegX | (3 814 936, 3 251 600) | +1 267.9 m |

**And every station states its LIGHT.** MEASURED: the harness today puts the star 25° up and looks toward
it (§1.1), which is the worst light for reading relief and is a typed constant no ruling names. A station
without a stated light angle is not a regression station, because two shots of identical ground under two
lights are not comparable. Every station below therefore carries a star elevation and a star azimuth
relative to the camera's own bearing, and D20 asks the owner to choose them.

| # | Station | Where | What it must show |
|---|---|---|---|
| 1 | **The boots** | eye 1.7 m, on a named rung-0 chunk, looking at the horizon | near ground detail, the 3.4 km horizon, an occupant in frame |
| 2 | **The ridge** | eye 373 m above the local surface, at S2-a…S2-d | 50 km of ground, beside P1's computed break count, at all four stations |
| 3 | **The valley** | eye 1.7 m, on the floor of a channel | walls on both sides, a base level, a river |
| 4 | **The cliff** | 50 m from a face over 30 m tall | that a cliff exists, and how many strata it exposes |
| 5 | **The approach** | 10 km up, looking down at 45° | the mid rungs, the crossfade, no visible LOD jump (SL8) |
| 6 | **From orbit** | 2 000 km up | the coarse rungs, the continents, the ice caps, the terminator |
| 7 | **The cube corner** | on a corner address, at latitude 35.2644° | ruling V6 A5's landform, and that three faces agree there |

### 7.2 The overlay every picture carries

The current overlay states the realm, the entity and the position. It must also state what a judge needs:

```text
  VOXELDUST — dev client
  realm:     Planet 7701581858760374086     seed:  ...  world tag: ...
  camera:    station 2 "the ridge"          chunk: face 2 / 1181 / 77 / rung 0
  altitude:  +412 m above the sea           latitude: 34.2°   biome: Grassland
  horizon:   50.1 km                        body radius: 3 350 759 m
  RUNGS IN FRAME: 0,1,2,4,6                 DRAW RADIUS: 40 chunks
  STAR: 12° up, 68° left of the nose        finest live octave: 48.8 m
  skyline breaks (computed by P1): 7        scale: the occupant is 1.8 m tall
                                            ─────  100 m at this distance
```

Seven of these lines are new: the camera station, the altitude with the horizon distance, **the rungs in
the frame and the draw radius**, **the star's elevation and its azimuth against the nose**, **the finest
live octave**, the P1 count, and the scale bar. Three of them each answer a defect this investigation
found:

- The RUNG line would have prevented revision 1's mistake: `ground.png` never said it was thirteen columns
  of rung 0.
- The STAR line would have shown that the owner judged three pictures shot into a 25° sun (§1.1).
- The FINEST LIVE OCTAVE line states what the rung actually threw away, which §3.4 shows is the real
  content question: at rung 5 nothing under 1 562 m survives, whatever the cell size says.

### 7.3 The rule I recommend

**No picture goes to the owner without a scale reference, a stated light, and a readout that states the
rung and the draw radius.** A picture without them cannot be judged, cannot be compared with last week's,
and wastes the owner's time. The three pictures of 2026-09-07 are the evidence twice over: a patch edge
was read as a horizon, and the light that flattened them was a constant in a test file.

### 7.4 What can be shot today, and what cannot

MEASURED, from `D-TERRAIN-3` (`DEFERRED.md:7775-7792`): slice 7 draws ONE rung per realm, named by a dev
flag, and the client library asserts that no two rungs of one realm are resident at once. So:

| Station | Shootable today? |
|---|---|
| 1 the boots | yes, at one rung and a wide `VD_TERRAIN_RADIUS` |
| 2 the ridge | **partly** — the 50 km vista wants several rungs in one frame, which the assertion forbids |
| 3 the valley | no channel exists to stand in |
| 4 the cliff | **no** — no cliff exists (§1.4) |
| 5 the approach | **no** — a crossfade needs two resident rungs |
| 6 from orbit | yes, at a coarse rung |
| 7 the cube corner | yes as a picture; the A5 landform does not exist yet |

Four of the seven stations wait on slice 8. That is the point of naming them.

**And §6's gates inherit that, which revision 2 stated here and forgot in §6.** Refutation A is right:
a reader of §6 alone would take four gates as runnable today. They are not, and the split is not the one
a reader would guess:

| Proxy | Runnable today? |
|---|---|
| P1 skyline, P2 slope, P3 hypsometry, P5 spectrum | **yes** — every one reads `height_m` directly, with no renderer and no rung flag. They need no station to be SHOOTABLE, only ADDRESSABLE. |
| P8's corner landform | **yes** as an arithmetic check; the landform it checks for does not exist |
| P4 slope–area, P6 census, P7a–P7d | **no** — each waits on a mechanism or on a body's facts |
| Every PICTURE of stations 3, 4, 5 and 7 | **no** — `D-TERRAIN-3` and the missing landforms |

So the instruments and the pictures wait on different things, and only the pictures wait on slice 8. That
is why §6 measures the field and §7 illustrates it.

---

## 8. What the target costs

### 8.1 The measured base

| Quantity | Value | Source |
|---|---|---|
| Budget per chunk (worker time), gated on the COSTLIEST named chunk at any rung | **8 ms** | ruling V10; `terrain_cost.rs:41,345,357` |
| A surface chunk (sample box + extraction) | 3.3 ms | MEASURED, `slice_06_extractor.md:363` |
| **A cave-dense seam chunk (the one the budget was set for)** | **6.11 ms** (4.38 box + 1.73 extract) | MEASURED, `slice_06_extractor.md:352` |
| Sample box, rung 0 | 1.98 ms | MEASURED, slice 6 |
| Cell pass, rung 0 | 716 µs | MEASURED, slice 6 |
| The halo's share of the sample box at rung 0 | +35 % (1.46 → 1.98 ms) | MEASURED, `slice_06_extractor.md:355` |
| **Column pass, rung 0 (the height field), 62 × 62 columns, 14 octaves** | **710 µs** | MEASURED, `slice_05_generator.md:374` |
| Sample box, rung 11 | 1.05 ms | MEASURED, slice 6 |
| Columns in a sample box (62 + halo) | 64 × 64 = 4 096 | `lattice.rs:50` `BOX_EDGE` |
| **Cost of one octave, one column** | **13.2 ns** | MEASURED-derived: 710 µs ÷ (3 844 × 14). Counting `biome_at`'s two extra noises the divisor is 16, giving 11.5 ns, which agrees with the noise bench's 12.19 ns per raw evaluation. |
| Client geometry per chunk, one thread | 4.18 ms, 405 KB | MEASURED, `slice_07_client_link.md:236` (M7-2, M7-3) |
| Client geometry, 14 threads | 2 004 chunks/s | MEASURED, M7-2 |

Revision 1 ESTIMATED the column pass by subtracting the cell pass from the sample box and got 1.26 ms and
22.0 ns per octave-column. Both refuters showed the bench PRINTS the column pass directly, and that the
subtraction mixed a 64³ box with a 62³ chunk. The measured price is 1.7× cheaper — and the headroom still
shrinks, because §8.2 corrects a bigger error in the other direction.

### 8.2 The headroom, against the chunk the gate actually holds

The budget holds the WORST named chunk, not the cheapest. `terrain_cost.rs:345,357` takes the costliest
chunk of the named set at any rung and asserts it under 8 ms, and ruling V10 set 8 ms **because** the
cave-dense chunk cost 6.1 ms. So:

```
   8.00 ms   the budget
 − 6.11 ms   the costliest named chunk today (MEASURED)
 ─────────
   1.89 ms   free per chunk

   one extra field evaluation over the whole sample box
        = 4 096 columns × 13.2 ns = 54.1 µs
   1.89 ms ÷ 54.1 µs = 35 extra evaluations per column
```

So the field work may grow from 14 to about **49 evaluations per column** — about **3.5×** — before the
budget binds. Revision 1 said 4.7× against the surface chunk; that was the wrong chunk, and both refuters
were right to call it a blocker.

**What those 35 buy, priced honestly.** Revision 2 spent 32 of the 35 on a shopping list whose prices
nobody has measured. Refutation A re-priced it and is right on every line:

| Item | Revision 2 said | The true price |
|---|---|---|
| Ridging (fold the noise about zero, flip it) | part of "+14" | **free** — a fold and a subtraction on a value already computed |
| An analytic derivative on every octave | +14 | **UNMEASURED (U7)**. It re-uses the eight corner dot products but adds the fade's derivative and eight more multiply-adds per axis. "+1×" is optimistic and nobody has run a `noise3_d`. |
| A three-dimensional domain warp, per warped octave | +14 for fourteen octaves | **+3 per warped octave** — a 3-D warp needs a 3-D offset, so three noise evaluations. Fourteen warped octaves cost **42**, which alone exceeds the whole headroom. Warping ONCE at a coarse level costs 3. |
| A climate field of four slow noises | +4 | +4, and this one is sound |

**So the honest reading is the opposite of revision 2's.** Warping every octave does not fit. A defensible
shopping list inside 35 is: warp once at a coarse level (+3), a derivative on every octave (+14 if U7 says
+1×, and unknown otherwise), ridging (free), and the climate (+4) — 21 of 35, with the derivative's price
the whole risk. **And it still leaves nothing for a channel field**, which §8.3 prices separately.

**Two warnings.**

- **The cave-dense share is UNMEASURED, and the target may raise it.** A rougher surface (D6, D7) and a
  channel network both raise the number of chunks that carry a cave band. Nobody has measured what share
  of the home planet's chunks are cave-dense today, nor what it becomes. `terrain_cost.rs` already walks a
  named radial column and counts skips; the classification is a small addition (§10 U9).
- **The client's budget is a different budget.** The client DOES hold a worker pool — MEASURED,
  `crates/client-render/src/terrain.rs:210-214` starts `ThreadedWorkers` with
  `std::thread::available_parallelism()`, and ruling V12 S7-6 settled the injected-worker question.
  Revision 1 said the client had no worker pool and cited a document written before slice 7 landed;
  refutation A is right, and this is the exact trap "read the code, not the docs" names. What remains open
  is the ARRIVAL RATE, which §8.4 prices and slice 8 owes.

### 8.3 The ladder-memory price

From §5.6, a top-down channel network costs `Σ_{L=1..11} 4^(−L) = 0.333` ancestor chunk evaluations per
rung-0 chunk with a perfect cache. The price depends entirely on two unknowns:

| Assumption | Ancestor evaluations per rung-0 chunk | Cost, at the stated ancestor price | Share of the 1.89 ms headroom |
|---|---|---|---|
| Perfect cache, ancestors only, an ancestor costs a column pass (0.71 ms) | 0.333 | **0.24 ms** | 13 % |
| Perfect cache, an ancestor costs a full rung-0 sample box (1.98 ms) | 0.333 | 0.66 ms | 35 % |
| A 3 × 3 halo at every rung, no sharing between neighbours | up to 3.0 | **up to 2.1 ms** | **111 % — it does not fit** |

**So the answer is "it depends on the cache, and nobody has measured the cache".** The optimistic end
leaves room for the per-point improvements of §5.4 item 1; the pessimistic end spends the whole budget on
the memory and leaves nothing. Revision 1 wrote "both fit". That was not supported then, and it is not
supported now. §10 U3 names the spike.

### 8.4 What a FRAME costs, and what a BOOT costs

§8.1–§8.3 price one chunk. The target is a VISTA, which is a chunk COUNT. Revision 1 never computed it;
refutation B is right that the affordability case answered the wrong question.

ESTIMATED from geometry alone. With a screen-space rule of `c` pixels per cell, a chunk at distance `d`
spans `62 · c · θ_px · d` metres, so the chunk count is fixed per octave of distance. Integrating from
62 m to 50 km, with `θ_px = 45° ÷ 1 080 rows = 7.272 × 10⁻⁴ rad`:

| Pixels per cell | Chunks over 360° | In a 60° field | **360°** on one thread at 4.18 ms | **360°** on 14 threads at 2 004/s | **360°** mesh bytes at 405 KB |
|---|---|---|---|---|---|
| 1 | 20 685 | 3 448 | 86.5 s | 10.3 s | **8.4 GB** |
| 2 | 5 171 | 862 | 21.6 s | 2.6 s | **2.1 GB** |
| 4 | 1 293 | 215 | 5.4 s | 0.6 s | 0.5 GB |

**The last three columns are computed on the 360° count, not on the 60° column beside them** (refutation
B, N1: a reader taking a row straight across gets a number six times too large for one frame). Divide by
six for a 60° field: at two pixels per cell that is 0.35 GB of mesh and 0.4 s on fourteen threads.

**And the 405 KB is a rung-0 measurement.** `slice_07_client_link.md:235` measured it over 243 chunks at
rung 0. Refutation B is right to flag that it is applied to every rung here — but M7-3 in that same
measurement already gives the structural reason it holds: *"a chunk is 62³ cells at every rung, so the
bytes per chunk are the same order at every rung and the count of chunks is what the band controls"*. The
surface crosses about 62² cells of a chunk whatever the rung is. That is an argument, not a measurement,
so the coarse-rung byte count stays **UNMEASURED (U14)**, and one extraction per rung settles it.

**The wall is memory, not time.** At two pixels per cell a full 360° vista around station 2 holds 2.1 GB
of mesh. M7-3 already measured the same wall from the other side: a 13 × 13 × 3 patch is 507 chunks and
about 205 MB. Time is survivable — 2.6 s of cold build on fourteen threads — but the residency band must
hold far fewer chunks than the geometry asks for, or the client dies.

**What this means for the target.** The screen-space rule is not a rendering detail. It sets the chunk
count, the memory and the arrival rate, and it decides whether the owner's 50 km vista is affordable at
all. Slice 8 owes it, and the owner will meet it as a decision (§11 D15, §10 U11).

**And the BOOT, which revision 2's own section title promised and never priced.** Refutation A is right.
The tree holds one measured boot number: the golden self-check over 8 chunks costs **13.5 ms**
(`slice_05_generator.md:386`), and it runs before a body is trusted (`terrain_cost.rs:332-338`). A richer
field raises it roughly in proportion to the field work, so 3.5× the field work is about 47 ms per body —
still small. The boot number the PLAYER feels is a different one: how long after login until the first
chunk is on screen. **That is neither measured nor listed anywhere**, and it belongs to slice 8 (U5).

**Game example.** The pilot lands his hull at station S2-b and steps out. The shard must trust the body
before it hands him one cell of ground, which costs the 13.5 ms self-check once. Then the client must
build the vista he is looking at: at two pixels per cell over his 60° view that is 862 chunks, 0.4 s on
his fourteen threads and 0.35 GB of mesh — for ONE facing. If he turns around, the band must already hold
the other side, or he watches the ground arrive.

### 8.5 What it costs in bytes and in the record

Nothing, if the mechanism stays inside the seed. A channel network derived from `(seed, address)` ships
zero bytes and stores zero bytes, exactly like the height field today. It is the strongest argument for
the derived form over the shipped map of §5.5.

---

## 9. The law gates: how this target passes each one

The task requires every proposal to state how it passes the standing laws. This document proposes a TARGET
and a set of GATES, not a recipe, so each row says how the target and its gates behave.

| Law | How this document's proposal passes it |
|---|---|
| **SL10 — one generator, two hosts, no drift** | Every proxy in §6 reads the crate's own `height_m`, `biome_at` or extractor, and **none of them reads a rendered frame** (§6.1 replaces revision 1's GPU gate with a ray-march). So a gate measures the shipped function and never a model of it. Two honest exceptions: P4 needs a flow router and P5 a spectrum estimator, and each is a new model whose own defects can fail the gate — so each bench must first be checked on a synthetic field with a known answer. The target FORBIDS any mechanism that is not a function of `(seed, address)` or a one-hop diff; §5.2 refuses the learned approach by name, and §5.5 prices the shipped-map alternative. **The one place the target strains SL10 is §4.6 shape 1**: a fact drawn by unfenced forest arithmetic and read by the shape would break the server-to-server leg, so every crossing fact must be an integer or a fenced value. The picture set and the benches are dev tools; they compile into no product binary. |
| **SL6 — ask before new data crosses a realm boundary** | The target itself asks for none. Four of its consequences do, and each is written as an ask rather than assumed. **The one that actually binds is the CLIENT lane**, which revision 2 missed: the client builds its body from a seed and a radius that arrive inside `SurfaceStmt` under `TAG_SURFACE`, in a self-look bag capped at 1 200 bytes (`crates/client/src/chunks.rs:266-289`; `crates/core/src/look.rs:56-79`). Any fact that shapes the ground must ride that statement, be bit-stable, and enter the world identity the handshake refuses on. The other three: the facts reaching the generator at all (§4.6, Q1), the spin's VALUE (§4.5 — now recommended as a SEED DRAW, not a lane datum, Q2), and the weather (§4.7's five questions, Q6). Default is NO until the owner rules. |
| **The seed ruling (2026-08-27)** | Everything the target asks for is SHAPE and CLIMATE, which are safe to publish: a wiki that lists every mountain and every river of the home planet takes nothing from any player. The target asks for NO seed-derived map of value. §4.4's habitable zone and §4.7's climate decide where the good land is, which is a fact about the world, not a treasure. Where a landform would betray a deposit — a gossan on a ridge, an ore-stained cliff — `03_generator_sl10.md`'s **open** decision D11 is where it is argued (it is a proposal, not a ruling), and this document adds nothing to it. |
| **SL5 — one world** | Every number in §1 and every camera station in §7 reads the home planet of seed 2298, THE world. No proxy uses a reduced planet, a test seed or a scale knob. §1.6's four variants are a MEASUREMENT of what a parameter change would do; they run in a bench, they change no shipped byte, and none is proposed as a second world. A bench may sample fewer columns; it may never sample a different world. |
| **No magic numbers** | §6 states thresholds for the GATE, not for the recipe, and §6.9 says plainly which of them have no law yet. §4 gives the law for the bands that have one. The document names four magic numbers already in the code: the 12 000 m relief cap and the 200 m floor (`body.rs:148`), the 30 m short-wave floor (`body.rs:25`), and — the one revision 1 missed, which causes its own headline — **`LONG_WAVE_CAP_M = 400 000` (`body.rs:24`), which saturates for every seed on this body**, plus the 20 000 m floor beside it. **The floor's reach, corrected:** `body.rs:151-153` draws `long_share` in `[0.25, 0.5)` and then clamps, so the floor ALWAYS binds under 40 km of radius and binds under 80 km only when the draw sits near the low end. Revision 2's "any body under 80 km" was true only for the smallest draw (refutation A, W5). Two more join the list in revision 3: `SUN_ELEVATION_DEG = 25.0` in the picture harness (§1.1) and the `0.35`/`0.75`/`0.3` biome thresholds in `height.rs:38-66`, which decide where every desert and every white cap on every planet sits. |
| **The 8 ms per-chunk budget** | §8 prices the target against the chunk the gate actually holds: 1.89 ms of headroom, about 35 extra field evaluations per column, and a ladder memory costing between 0.24 ms and 2.1 ms depending on a cache nobody has measured. §8.4 adds the frame and the boot, which the per-chunk budget does not cover at all. |
| **The ladder (V8/V9)** | The target does not change the ladder's contract: rung L must stay cheaper than rung 0 and must stay inside a stated bound of it (`body.rs:274`). §5.6's top-down network is built ON the ladder, so the coarse rung's answer is the parent of the fine rung's. Any new field must state its own dropped bound (§12 Q4), and §5.6 warns that the coarse rung is 10 584 chunks and needs its own stencil. |
| **The record (12 bytes, the density byte, the object parameter)** | **This row was FALSE in revision 2 and both refuters caught it.** A generated cell is `Cell { stratum, gap }` (`chunk.rs:64-69`) — no biome, no moisture, no temperature. `biome_at` runs once per column, chooses a topsoil, and its value is then thrown away (`chunk.rs:637`). Ruling V6 Part B's twelve bytes name no climate parameter (`owner_decisions_2026-09-07_voxels.md:186-193`). **What is true:** shape still rides the density byte and a channel is a shape, so the SHAPE half asks for nothing new. **What is an ASK:** V4's asset skeleton needs a biome, a slope and a MOISTURE readable at a site; §4.7's rain shadow needs moisture; §6.7 P7c reads it on two sides of a ridge; and §3.5 item 4's river needs a WATER LEVEL. None of those has a home in the record, in the generated cell or on the wire today. This is the one place the target touches the format, and it is stated as an ask (D19, D22), not as a solved question. |
| **The edit pyramid as the authoring override** | Untouched. A richer recipe makes the seed answer better; every player edit still overrides it by the same composition order. §5.2's sparse-atom reference is noted as the closest literature to our pyramid, not as a change to it. |
| **The collider on the same shape** | Untouched, and reinforced: every proxy in §6 reads the same surface the collider reads, so a gate that passes on the picture passes on the boots. §6.2's cliff test is exactly the case where a picture and a collider could disagree, which is why it is a gate. §3.5's overhang limit is also a collider fact: the boots cannot stand on an arch the shape cannot represent. |
| **SL8 — seamless** | §7.1 station 5 exists only to shoot the crossfade, and §6.5's spectrum is measured per rung, so a mechanism that adds detail at rung 0 and not at rung 3 shows up as a spectral step. The target adds one requirement, stated in SCREEN SPACE: **a landform must not appear or vanish while it is larger than a pixel.** Revision 1 wrote "a landform must not appear or vanish at a rung change", which forbids the free coarsening the ladder is built on (`octaves_at` drops the L finest octaves by design, and `dropped_bound_m` bounds it). Refutation B is right; the rule is rewritten and it now has a decision row (§11 D16). |
| **HR5 — 100 % coverage in the generator crate** | Every branch a richer recipe adds needs a test. §8's headroom is worker time, not review time: a mechanism with many branches costs coverage work, and CLAUDE.md's generic-code rule (branchless generic shims, monomorphic helpers) applies to every new field. This is a real cost of the target, and §11 D4 puts it to the owner. |
| **V4 — vegetation are art assets placed by a server skeleton** | Untouched, and the target depends on it: §2's "forest as a mass" is the canopy fold of ruling V10 §12, and §6.6's census counts landforms, never trees. The terrain's job is to give the skeleton a believable biome, a slope and a moisture value to place assets by. |
| **V6 A5 — a landform at all eight cube corners** | Binding and approved, and now carried: §6.8 makes it a gate, §7.1 gives it camera station 7, and §5.6 names it as the one place three faces must agree on one shape. Revision 1 omitted it entirely. |
| **V5 — a cave you can see into** | The target records that the code today FORBIDS it: every carve sits at least 8 m under the surface (§3.5). A mechanism must open a mouth, or ruling V5's picture cannot be taken. |

---

## 10. What is UNMEASURED, and how each is measured

In the order I recommend running them.

| # | Unknown | The measurement | Cost |
|---|---|---|---|
| **U1** | The §1.3–§1.6 statistics, computed by the CRATE rather than by my Python replay. | A bench beside `terrain_cost.rs`: the slope histogram, the plane departure, the hypsometry and the ray-marched skyline, all through `height_m`. **It inherits the four station addresses of §1.4a, the 360 km march limit and the sampling seed**, so its numbers are comparable with this document's. It is also the P1/P2/P3 gate implementation. | ESTIMATED at release speed: about 1.1 s for P2 over 10⁶ columns, 0.2 s for P3, 0.5 s for P1 at the 360 km limit (§6.11). A DEBUG build is a different question, and §6.11 states it. A day to write. |
| **U2** | The true supremum of `\|∇ noise3\|` (my 2.97 is a sampled maximum over 400 000 points), **and of `\|noise3\|` itself** — the ladder's `dropped_bound_m` and the chunk band both call themselves exact on the unproved claim that it is at most 1 (Q4). | A dense search inside one lattice cell over many corner-gradient sets, or an analytic bound on the quintic-faded trilinear form. One paragraph of algebra settles both. | hours |
| **U3** | Whether the ladder memory of §5.6 fits: the ancestor cache hit rate, the halo multiplier, and the coarse rung's stencil. | Build the cheapest possible top-down channel field; measure the added milliseconds on the named chunk set and the cache hit rate over a walked path. | a day of work; it is a spike, not a slice |
| **U4** | Whether the pictures agree with §1's field measurements once `D-TERRAIN-3` closes. | Re-shoot stations 1, 2 and 6 at the widest draw radius the lane serves, at the rungs slice 8 chooses. | slice 8 |
| **U5** | The client's ARRIVAL RATE with the richer field: chunks per second against the residency band of §8.4. | Slice 8's own measurement, with the new field in place. | it blocks the client leg |
| **U6** | The home planet's own physical facts: insolation, `T_eq`, gravity, its density, whether it keeps an atmosphere. | `cargo run -p vd-bins --example system_census`, plus the taxonomy calls of §4.6 on the home planet's row. §4.2's ceiling for the home planet guesses a density of 5 000 kg/m³ until this runs. | minutes |
| **U7** | The cost of one analytic derivative per octave. | Extend the noise bench: `noise3` against a `noise3_d` that returns the value and the gradient. | hours |
| **U8** | Whether any relief-law change keeps every body on the ladder. | Re-run `Ladder::for_radius` over every body the forest draws, under the candidate law, and count the refusals. | minutes |
| **U9** | What share of the home planet's chunks are cave-dense today, and what a rougher field makes it. | Extend `terrain_cost.rs`'s radial-column walk to classify each chunk. | hours |
| **U10** | The halo's cost for a channel field, as opposed to the extractor's 35 %. | Part of U3. | — |
| **U11** | The screen-space rule: how many pixels a cell must cover, and the residency band that follows. | Slice 8; it decides §8.4's whole table. | slice 8 |
| **U12** | Whether any published law derives β, the slope distribution or the peak density from a body's facts — and whether Held–Hou's constant is the one §6.7 prints. | A literature pass, with the papers actually opened (§13). | days |
| **U13** | **Whether the Python replay's `noise3` matches the crate's.** Every headline in §1 is a function of it, and no pin tests it (§1.2). | Reproduce `golden_self_check`'s home-body digest `0x9331e1fdfd902272` in the replay, or match one named column's surface radius against `terrain_cost`'s printout. | hours; it is the cheapest unpaid debt in this document |
| **U14** | The mesh bytes of a chunk at a COARSE rung (§8.4 applies a rung-0 measurement to every rung). | Extract one chunk at rungs 0, 3, 6 and 9 and print the byte count of each. | minutes |
| **U15** | Which light shows this terrain honestly. | Re-shoot station 2 at star elevations of 5°, 15°, 25° and 45°, and at azimuths of 0°, 45°, 90° and 135° from the camera's nose. Sixteen frames settle a constant that nobody chose. | slice 8; hours |
| **U16** | **The reference picture's break count, in OUR units** (§2.2: the 8–12 was counted by eye, the 0 was computed by a rule). | Trace the reference's skyline by hand as elevation against bearing, using its stated camera height and field of view, then run §6.1's break rule on it. | hours; it converts P1 from a ratchet into a floor |
| **U17** | Whether the four variants of §1.6 keep their skyline numbers at the 360 km march limit. | Re-run §1.6 on the converged instrument. Part of U1. | — |

**Nothing in §6's "proposed band" column is a result.** The "our world today" column is REPLAYED, and U1
makes it MEASURED.

---

## 11. What you decide

| # | Question | Options | Recommended | Why |
|---|---|---|---|---|
| **D1** | Does a named camera set with a readout become the way every terrain picture is produced? | (a) yes, seven stations whose addresses follow a printed RULE, re-shot every slice, the overlay stating the rung, the draw radius, the finest live octave and the star's angle; (b) ad-hoc screenshots as today. | **(a)** | A picture without a rung line cannot be judged, and one without a light line cannot be compared with last week's. The 2026-09-07 pictures are the evidence twice: a patch edge was read as a horizon, and a 25° head-on sun flattened what was there. |
| **D2** | Which proxies become gates, and HOW do they land without turning `just gate` red for every unrelated change? | (a) **P1 skyline and P2 "a cliff exists" as RATCHETS** (they may not get worse), **P8 the corner landform as a hard floor**, every band that waits on a measurement as a `DEFERRED.md` expected-red with a named closing slice; (b) revision 2's four hard floors — P1, P5, P6, P8 — red from the day they land; (c) none, print and judge by eye. | **(a)** | Revision 2 chose (b) and both refuters refused it. β must leave the list entirely: §1.6 measures that a β-gate PASSES the unwalkable gravel heap and could FAIL a working mechanism (§6.5). P1 has no honest band until U16 measures the reference with our own rule (§2.2). And `terrain-cost` and `terrain-legs` are not in `just gate` today, so this is a change to the pre-merge recipe (§6.11). |
| **D3** | What is an "Earth-like" planet measured against? | (a) Earth's own statistics, one fixed set of bands; (b) a per-body target derived from that body's facts, with Earth as the calibration point. | **(b) where a law exists, (a) where none does, and say which is which** | It is the owner's own requirement and the no-magic-numbers rule. §6.9 lists the four bands with no law today, and inventing one would be worse than admitting it. |
| **D4** | How much may the field work grow? | (a) stay inside the 8 ms budget, about 49 field evaluations per column; (b) raise the budget; (c) stay near today's 14. | **(a)** | The owner set 8 ms and said "if that becomes a problem over time, we rethink". §8.2 shows 3.5× headroom against the chunk the gate holds — enough for the per-point improvements, and NOT enough for them plus a channel memory with a halo. Raising the budget should follow a measurement. |
| **D5** | The relief law. | (i) keep `relief ∝ radius` with the clamp; (ii) **withdrawn** — revision 1's `∝ 1/gravity`; (iii) treat the law as OPEN, adopt the strength ceiling as a BOUND only, and let a mechanism document propose the value's law with the process behind it. | **(iii)** | The six-body table shows no relation between relief and radius or gravity (correlation −0.09 and −0.17). The present law uses 0.02 % of the physical ceiling on a 100 km body and 600 % of it on a super-Earth. That is the defect. **No one-way door is put to you in this revision.** |
| **D6** | The finest scale the recipe carries. | (a) keep the 30 m short-wave floor, so the finest feature is 48.8 m; (b) carry shape down to a few metres; (c) down to the 1 m cell. | **(b)** | (a) leaves the ground a plane to 26 cm inside 50 m (§1.3). (c) makes every cell noisy, which fights V2.1 and costs octaves for detail an art asset should carry. **This is a one-way door too:** `body.rs:182` normalises every amplitude by the weight sum, so one extra octave moves EVERY existing amplitude — on the home planet a fifteenth octave moves the 7 605.55 m octave by 0.099 m, and the identity's tolerance is zero (ruling V9 S5-2). A 10 cm move is a version bump exactly like a 10 km one. |
| **D7** | The spectral target. | (a) β ≈ 2 by changing `k_rough` alone; (b) β ≈ 2 AND a break at the valley spacing, which needs a mechanism. | **(b)** | §1.6 measures (a): it makes 30 % of the planet steeper than 45° and still gives 0.8 skyline breaks per 60°. Roughness is not landform. |
| **D8** | Where the standing climate lives. | (a) in the generator, as a function of seed and the body's facts; (b) on the server only, shipped to the client; (c) not modelled — keep the two noises. | **(a)** | The terrain's SHAPE depends on the climate (a rain shadow is a shape), so the client must derive the same climate to derive the same shape. SL10 permits it, because the standing climate is not a function of time. |
| **D9** | "No impostors" (ruling V10) written as a measurable rule. | (a) a far tree is a mesh at some detail level, or it is folded into the canopy surface, never a camera-facing card; (b) leave it as a word. | **(a)** | A word cannot be gated. Station 2 is where a card would be caught. |
| **D10** | Spin and obliquity: does the world gain them, and who owns the VALUE? | (a) **the body DRAWS them from its own seed under the float fence**, beside its relief and its sea, under their own salt; the placement lane then REPORTS the same number to whoever moves; (b) revision 2's answer — they ride the placement lane and the terrain reads them there; (c) no, keep `POLE_AXIS = +Z`. | **(a)** | Revision 2 chose (b) and refutation A refuted it on the law. A placement reading carries an instant and SL1 clause 6 refuses it when stale; SL10 forbids the shape depending on time or live state; and `just terrain-legs` could not even express the comparison, because the client leg has no lane. A seed draw is a function of `(seed, address)` by construction. **Its salt must be its own, or every existing body moves** (§4.5). This still changes the physics crate, not only the terrain. |
| **D11** | Ruling V6 A5's corner landform: is it part of this target? | (a) yes — a gate and a camera station; (b) leave it to the grid work. | **(a)** | It is approved and binding, it is about the picture, and it is the only place three faces must agree on one shape. |
| **D12** | Bedding: does the target ask for beds at a constant RADIUS? | (a) yes, and it is understood as NEW WORK with its own cost — beds at a constant radius, a hardness per bed, a thickness that varies over the body; (b) no, leave the draped soil profile as a colour. | **(a), with its price named** | **Revision 2's reason was wrong and is withdrawn.** The 19 strata are a substance palette, and the table drapes four substances at a constant DEPTH with the hard rock at the BOTTOM (`strata.rs:187-210`), so it cannot cap a mesa or bed a cliff at any erosion rule. Both refuters proved it from the code. The IDEA still stands — a cliff without bedding reads as a brown wall, and three of §3.1's landforms are cap-and-soft-layer effects by definition — but it is a mechanism to build, not an asset to switch on. |
| **D13** | Who owns the aerial perspective (the blue haze)? | (a) the client's atmosphere work, named as a slice now; (b) later. | **(a)** | MEASURED: there is no fog, haze or atmosphere anywhere in the client. Every landform in this document will still read flat without it. |
| **D14** | Is shipping a solved map on the table at all? | (a) no — the target must derive everything; (b) yes for a coarse map (a few MB), no for a fine one; (c) yes. | **(b), as a fallback to price, not a plan** | §5.5: a whole-body 1 km map is 47–94 MB compressed (UNMEASURED); an 8 km map is about 1 MB. A shipped map changes the world identity and needs a store, which is a bigger change than its size. |
| **D15** | The screen-space rule (pixels per cell) that decides the rung. | (a) 1 px — sharpest, 8.4 GB of mesh for a 360° vista; (b) 2 px — 2.1 GB; (c) 4 px — 0.5 GB and a visibly softer far view. | **your call; slice 8 owes the measurement** | §8.4. It decides whether the owner's 50 km vista is affordable, and it has never been stated. |
| **D16** | The SL8 rule for a landform across a rung change. | (a) "a landform must not appear or vanish while it is larger than a pixel" (screen space); (b) "a landform must not appear or vanish at a rung change" (revision 1's wording). | **(a)** | (b) forbids the free coarsening the ladder is built on: `octaves_at` drops octaves by design and `dropped_bound_m` bounds it. |
| **D17** | **THE WEATHER: what is the smallest thing that counts?** The owner's second sentence had no target, no proxy, no cost and no decision row in revision 2. | (a) the STANDING climate only — a wind rose, a rain field, a snow line, all seed-derived and shaping the ground; (b) (a) plus visible moving cloud with shadow, which is live realm state; (c) (b) plus rain, storms and a day–night cycle the player feels. | **your call; (a) is the only part this document can target today** | Every proxy in §6.7 measures standing climate, because standing climate is a function of `(seed, address)` and a storm is not. **And the dormant world forces a fork nobody has asked:** a planet realm that is not running simulates nothing, so weather on approach is either `f(seed, time)` with no running realm, or it waits for spin-up (§4.7 item 5). |
| **D18** | **The biome vocabulary.** | (a) keep four arms — `Desert`, `Grassland`, `Tundra`, `Highland`; (b) a **Whittaker** grid on two axes, mean temperature against mean rainfall, with forest, wetland, savanna, alpine and ice cap as cells of it. | **(b)** | MEASURED (`strata.rs:114-123`): there are four arms and `Highland` is a HEIGHT class, not a climate. There is no forest — and §2 asks for "forest as a mass", V4 places art assets FROM the biome, §3.1's tree line is the cheapest scale reference on a hillside, and every one of P7a–P7d needs a biome axis to histogram. Two things refutation B got wrong here, both worth stating: `Tundra`'s topsoil IS `Stratum::Snow` and `biome_at` DOES cool with height, so a white cap and an ice cap draw today. What is missing is the vocabulary, not the snow. |
| **D19** | **Water above the sea.** Does a river have a surface? | (a) yes — a per-column water level or a `Water` stratum the generator writes, with a per-basin base level; (b) no — channels are dry valleys and only the sea holds water. | **(a), and it must be decided before any channel work starts** | MEASURED: a body holds ONE `sea_radius_m` and `fluid_at` returns `Water` only below it (§3.5 item 4). Half this planet's land stands 5.3 km above its sea, so every channel the target asks for runs dry. It is three questions at once — the record, the collider, and whether the level is seed-derived or a one-hop diff — and §6.6's "closed basins" proxy counts holes rather than lakes until it is answered. |
| **D20** | **The light every picture is shot in.** | (a) a stated star elevation and azimuth per station, chosen after U15 measures which light shows the terrain honestly; (b) leave `SUN_ELEVATION_DEG = 25.0` and the camera pointing at the star. | **(a)** | MEASURED (`terrain_pictures.rs:45,458-462`): the harness puts the star 25° up and looks toward it, so every visible slope faces away from the light. Relief reads through shading. The owner judged three pictures under the worst light there is, and no ruling names the constant that chose it. |
| **D21** | **May a coarse rung carry structure the free coarsening does not remove?** | (a) no — keep the ladder's contract exactly as ruling V8/V9 states it, and accept that a 50 km ridge shows nothing under 1 562 m; (b) yes — a coarse rung may carry its own landform term, and the rung-agreement bound is restated for it. | **your call; it is a change to the ladder's contract, not a tuning** | §3.4: the finest live octave is always about 48 CELLS wide at EVERY rung, so the ladder guarantees the same relative smoothness at every distance. "Detail to the horizon" is therefore not a rung question. The reference's far ridges show 200–500 m sub-ridges at 30–60 km, which is five to fourteen pixels of structure that `octaves_at` removes by design. Q4 holds the bound this needs. |
| **D22** | **Does a site carry a biome, a moisture and a slope the asset skeleton can read?** | (a) yes — the generator states them per site, wherever they live; (b) no — the skeleton re-derives them from the shape. | **(a), as an ASK, not a claim** | MEASURED: a generated cell is `Cell { stratum, gap }`, the biome is thrown away after choosing a topsoil, and V6 Part B's twelve bytes name no climate parameter. V4 says vegetation is an art asset placed FROM the biome, and §4.7's rain shadow and §6.7 P7c both need moisture. Revision 2's §9 asserted this was already answered; it is not. |

---

## 12. Open questions

| # | Question | Why it is open |
|---|---|---|
| **Q1** | Which of the three shapes in §4.6 carries the body's physical facts into the generator, and how is the crossing datum fenced? | It crosses a crate boundary the float fence and the client's dependency rule both guard, and the forest's arithmetic is not fenced (35 transcendental calls in `taxonomy.rs`). It is an SL6 ask. |
| **Q2** | Who DRAWS the spin's value, and under which salt? | §4.5 now answers the lane half: a fact that shapes the ground may not arrive as a stamped, staleable placement reading (SL1 clause 6, SL10), so the body draws it from its own seed and the placement lane reports the same number. What stays open is who owns the draw — the generator, beside the relief, or the forest, which then must state it — and under which salt, because a draw taken in an existing stream moves every body. The same two numbers serve the day length, the seasons, the tides, the Coriolis force and the terrain. One number needs one owner. |
| **Q3** | Is a per-body precomputation lawful under SL10 at all, if both hosts run the identical fenced code and the golden gate covers its output? | SL10 says a function of `(seed, address)`, never of time or state. A per-body solve IS such a function, but it is not per-address, and its cost lands at spin-up rather than per chunk. §5.6's coarse rung may need exactly this. |
| **Q4** | What is the coarse-answer bound for a mechanism that is not a sum of octaves? | Today `dropped_bound_m` sums the dropped amplitudes. **It is exact only if `\|noise3\| ≤ 1`, and that is not proved.** The crate's own comment says the value is *"in about `[−1, 1]`"* (`noise.rs:85`) while `relief_bound_m` claims exactness on the stronger reading (`body.rs:261-262`); my replay's sampled maximum is 0.98 over 400 000 points and refutation A's is 0.905 over 200 000. Both are under 1 and neither is a proof. The supremum is one paragraph of algebra on a quintic-faded trilinear form, it belongs beside U2, and the ladder's contract and the chunk band both lean on it. A channel network or an erosion term has no such simple bound at all, and D21 cannot be answered without one. |
| **Q5** | Does the ocean exist as water the player swims in, and should the sea level move? | MEASURED: 0.99 % of the surface is under the sea, and the median land stands 5.3 km above it. A coast is one of the reference's strongest landforms, and this planet has almost none. The sea level is a seed draw, and changing it is a version bump. |
| **Q6** | The weather: what crosses a realm boundary, who authors it, does it change the SHAPE, and what does it cost? | §4.7's four questions. The owner named the weather explicitly and nothing in the tree answers any of them. |
| **Q7** | How much of the reference's "interesting" is the terrain, and how much is the asset scatter and the light? | Star Citizen's ecosystems suggest the scatter carries a large share. If it does, the shape budget may be smaller than §8 assumes. **UNMEASURED, and it can only be measured with both halves built.** |
| **Q8** | Does the target apply to every body, or only to the ones a player walks on? | 141 million km² of the home planet alone will never be walked. A gate that measures the whole sphere is honest; a gate that measures where players go is cheaper. It also decides §5.5's shipped-map unit. |
| **Q9** | Which body is the reference for "Earth-like"? | The home planet has half Earth's radius, a 27 % closer horizon at every eye height, an unknown density and an unknown insolation (U6). If the owner wants an Earth-like planet, the world may need to DRAW one. |

---

## 13. Sources

**Code, read on 2026-09-08 in this worktree.** `crates/terrain/src/body.rs` (the relief at 147-149,
`k_rough` at 155, the octave loop at 158-184, the sea at 186-190, the strata at 194-202, the caves at
205-225, `octaves_at` at 253, `dropped_bound_m` at 274), `height.rs` (16-27, 32-35, 38-66), `home.rs`
(16-23), `noise.rs` (20-120), `strata.rs`, `chunk.rs` (146-200, 377-383), `lattice.rs` (50);
`crates/seed/src/ladder.rs` (50-112), `rng.rs` (22-28, 70-74); `crates/core/src/frame.rs` (65-66,
243-273), `placement.rs`; `crates/client-render/src/terrain.rs` (46-54, 210-214);
`crates/physics/src/taxonomy.rs`, `crates/physics/src/worldgen/generate.rs`;
`crates/bins/examples/terrain_cost.rs` (41, 70-110, 345-360). Every line number was re-checked in this
revision; revision 1's `body.rs` citations were off by 4 to 14 lines and both refuters caught every one.

**Rulings and plans.** `docs/design/owner_decisions_2026-09-07_voxels.md` (V1, V2, V4, V5, V6 Parts A–D
including **A5** at line 184, V8–V12); `docs/design/owner_decisions_2026-08-27_seed_and_secrecy.md`;
`docs/design/DEFERRED.md` (D-TERRAIN-1..4; D-TERRAIN-3 at 7775-7792);
`docs/investigation/2026-09-07/slice_05_generator.md` (§11 the measurements, 367-394);
`slice_06_extractor.md` (§12 the vista, §13 the measurements, 344-368);
`slice_07_client_link.md` (§12.1 M7-2..M7-4, §12.3, §12.4); `03_generator_sl10.md` (its D11 and D13 are
OPEN decisions, not rulings — revision 1 called them rulings and refutation A was right); CLAUDE.md.

**Pictures.** `docs/investigation/2026-09-07/pictures/ground.png`, `hill.png`, `aloft.png`, viewed, with
their parameters read from `slice_07_client_link.md:237` rather than guessed from the image.

**The replay.** A Python re-implementation of `SplitMix64`, `child_seed`, `Ladder::for_radius`,
`BodyDefinition::from_seed`, `noise3`, `height_m` and `bend`/`direction`. **Three of its pins could have
failed and did not** — the relief, the sea offset and the share of columns under the sea; three more (the
ladder radius, the rung count, the octave count) are functions of the radius alone and prove nothing
(§1.2). **Nothing yet pins `noise3` against the crate**, and every headline depends on it: U13 names the
one-hour measurement that closes it. Refutation A ran an independent replay and reproduced the relief, the
octave amplitudes, the slope percentiles and the hypsometry to three digits, which is corroboration and
not a pin. Every REPLAYED number in §1 comes from my replay. It is not the crate, and U1 replaces it.

**The revision-3 measurements**, all REPLAYED and all reproducible from the addresses printed here: the
four-station skyline at four march limits (§1.4a), the hill climb on the slope (§1.4), and the Held–Hou
estimate of the Hadley cell edge (§6.7).

**Literature and industry**, as listed in §5.1 and §5.2. Each is cited as a reference to read, never as a
dependency to add. **Every year and author in §5.2 is written from knowledge; I did not open the papers.**
Three of this document's claims lean on them — the spectral break, the slope–area law's θ, and Hack's
exponent — and §6 proposes gate bands built on two of those. **No band from §5.2 may become a numeric gate
before somebody opens the paper and checks the number** (§10 U12).

**Published astronomy and rock values** (radii, gravities, reliefs, lapse rates, snow lines, albedos,
compressive strengths) are standard reference figures quoted for calibration. **No such number may enter a
recipe before a named source is attached to it** — "never assume, measure" applies to a borrowed number as
much as to our own.

---

## 14. Refutation answers

Two rounds. Round one read revision 1 (fifty-six findings, §14.3 and §14.4). Round two read revision 2
(thirty-nine findings, §14.1 and §14.2). In both rounds refutation A read for law, code and evidence, and
refutation B for believability, cost and the owner. **"FIXED" means the section was rewritten, not
annotated. "KEPT" means I believe the refuter is wrong and the reason is stated. "OWED" names who must
close it.**

### 14.1 Round two, refutation A (the laws and the code)

| # | Finding | Severity | Answer |
|---|---|---|---|
| A2-B1 | P1 is not reproducible: the camera has no address, the claim moves by a factor of 2.1 between cameras, and the band has no subject | blocker | **FIXED.** §7.1 makes the station a printed RULE and §1.4a prints four resulting addresses as a face and a rung-0 cell pair. §6.1 states the three parameters a skyline gate may never leave unstated. My own four stations give 0.054°–0.067°, which confirms the spread the refuter measured and withdraws revision 2's 0.040°. The band problem is answered separately: P1 becomes a RATCHET until U16 measures the reference with our own rule (§2.2, D2). |
| A2-D1 | The strata drape, the hard rock is at the bottom, one sediment and one bedrock — so no mesa, no hoodoo, no bedding | defect | **FIXED, and the recommendation is WITHDRAWN.** §3.5 item 3 is rewritten from the code, §0 item 6 now says bedding is new work, §2's row says the same, §6.8's proxy states that its answer is bounded at four substances and zero radii by construction, and D12 is rewritten to ask for beds at a constant radius with a named price. |
| A2-D2 | The maximum slope is a sample extremum used as a property; a hill climb beats it by 28 % | defect | **FIXED, and re-measured independently.** My own hill climb reached **11.15°** against the refuter's 11.17°. §1.4 now prints the sample size beside every extreme, states 11.15° as a LOWER bound, and rests "no cliff" on the analytic estimate of 28.1° with U2 attached. §1.3's span and §6.2's census carry the same correction, and §6.2 adds the rule: a gate reads a SHARE over a stated sample, never a maximum. |
| A2-D3 | §1.2 counts three radius-only quantities as independent checks, and nothing pins the replay's `noise3` | defect | **FIXED.** §1.2 now has a per-pin table saying which could have failed: three, not five. The bigger half is accepted in full and made a first-class debt — **U13**, reproducing `golden_self_check`'s `0x9331e1fdfd902272` — and §13 says plainly that two agreeing replays are corroboration, not a pin. |
| A2-D4 | The rain shadow is exactly as non-local as flow accumulation, and the document prices it at zero | defect | **FIXED.** §4.7 now states the 200 km fetch and the 3 200-chunk stencil, and says SL10 permits the split without making it free. §5.6 names the rain shadow as a second customer for the same ladder memory, which doubles what U3 settles. §8.3 prices it beside the channel. |
| A2-D5 | D10 routes the spin through the placement lane, which SL1 makes a stamped, staleable reading | defect | **FIXED.** §4.5 carries the three-line argument (SL1 clause 6, SL10, the client leg with no lane) and names the lawful form the refuter proposed: **the body draws its spin from its own seed and the placement lane reports the same number.** D10 and Q2 are rewritten around it, and §4.5 adds the salt consequence the refuter did not raise: a draw in an existing stream moves every body. |
| A2-D6 | §4.6 and §9 miss the CLIENT lane, which is where the facts must actually cross | defect | **FIXED.** §4.6 gains the measured client path (`chunks.rs:266-289`), the statement it rides (`SurfaceStmt` under `TAG_SURFACE`), the budget it must fit (`SELF_LOOK_BUDGET_BYTES = 1200`, so seven fenced facts are about 5 % of the bag), and the four things it needs at once. §9's SL6 row names the client lane as the binding one. |
| A2-D7 | Four RED gates have no home; `just gate` would go permanently red; where the code lives decides two laws | defect | **FIXED.** New **§6.11** answers all three parts: the three landing shapes (ratchet, expected-red with a named closing slice, hard floor) with a recommendation each; where the bench lives and what that costs under HR5; the release-versus-debug question; and the fact that `terrain-cost` and `terrain-legs` are not in `just gate` today. D2 is rewritten to put the landing shape to the owner, not only the list. |
| A2-D8 | D2 gates on β while the document proves β is the wrong target, and the band is unreachable | defect | **FIXED.** β is **removed from the gate list**. §6.5 states all three reasons: a β gate passes the unwalkable variant, could fail a working mechanism, and its band is red for every body the world can draw because `k_rough` is drawn in `[0.45, 0.55)`. D2's option (a) no longer contains it. |
| A2-D9 | §9's record row states something the code does not do | defect | **FIXED.** §9's record row is rewritten from `chunk.rs:64-69` and ruling V6 Part B, §3.5 item 5 states it in the structural-limits list, and **D22** puts the ask to the owner. |
| A2-W1 | §8.2 spends 32 of 35 evaluations on unmeasured prices; the warp is under-priced 3× | weakness | **FIXED.** §8.2 has a per-item price table: ridging free, the derivative UNMEASURED (U7), a 3-D warp **3 per warped octave** so fourteen warped octaves cost 42 and exceed the whole headroom. The honest list is 21 of 35 with the derivative as the whole risk. §0 carries the same correction. |
| A2-W2 | §8.4 promises a BOOT cost and prices none | weakness | **FIXED.** §8.4 now prices the boot: the golden self-check is a MEASURED 13.5 ms per body, about 47 ms at 3.5× the field work, and the number the player feels — login to first chunk — is UNMEASURED and belongs to slice 8 (U5). |
| A2-W3 | A gate's own arithmetic must be host-stable, and nothing says so | weakness | **FIXED.** §6.11 item 3 states the rule (a verdict must be a function of quantised inputs, or its margin must exceed the platform's variance) and §6.10 gains it as a row. |
| A2-W4 | P6 counts "closed basins (a lake site)" and the world has no local base level | weakness | **FIXED.** §6.6 now says the proxy counts HOLES, §3.1's base-level row states that only the sea holds water, and **D19** asks the owner for a water level above the sea. |
| A2-W5 | The 20 000 m floor's reach is stated wrongly | weakness | **FIXED.** §9's magic-number row: the floor ALWAYS binds under 40 km of radius and binds under 80 km only for a low draw. |
| A2-N1 | The ladder's "exact" bound rests on an unproven property of the noise | note | **FIXED.** Q4 is rewritten: it carries the crate's own "about `[−1, 1]`" comment against `relief_bound_m`'s exactness claim, both sampled maxima (0.98 and the refuter's 0.905), and the statement that neither is a proof. It sits beside U2. |
| A2-N2 | The tree line is missing from the vocabulary | note | **FIXED.** §3.1 gains a tree-line row naming it the cheapest scale reference on a hillside, and D18 asks for the biome vocabulary that would let it exist. |
| A2-N3 | Four of §7.1's stations are unshootable and §6's gates depend on them | note | **FIXED, and the dependency is different from what the refuter assumed.** §7.4 gains a table separating what is ADDRESSABLE from what is SHOOTABLE: P1, P2, P3 and P5 read `height_m` and run today with no picture at all; only the PICTURES of stations 3, 4, 5 and 7 wait on `D-TERRAIN-3` and on missing landforms. |

### 14.2 Round two, refutation B (believability, cost and the owner)

| # | Finding | Severity | Answer |
|---|---|---|---|
| B2-B1 | The strata cannot make a mesa, a hoodoo or a bedded cliff, and the document says they can | blocker | **FIXED**, as A2-D1. B's third point is carried too: "19 strata" is a substance PALETTE — `Air`, `Water` and `Empty` are not rock and five entries are the bedrock choices — so the rock vocabulary a column can hold is four deep (§3.5 item 3). |
| B2-B2 | The headline ray-march is truncated at 90 km and no convergence check is reported | blocker | **FIXED, and the finding is CONFIRMED by measurement.** §1.4a marches every station to 90, 180, 360 and 540 km. At station S2-c the 90 km limit understated the largest rise by 23 % (0.0489° against 0.0636°), and at S2-a it cut 0.79° off the skyline's low end. The march limit is now **360 km**, where all four stations are stable. **The instrument changed; the conclusion did not** — the largest rise anywhere is 0.067°, and no station holds a break at 0.10° or above. |
| B2-B3 | The gate's target and its measurement come from two different instruments | blocker | **FIXED.** New **§2.2** states it plainly: the 8–12 was counted by eye off a screenshot and carries no mark, the 0 was computed by a stated rule, and the ratio between them means nothing. P1 therefore lands as a RATCHET, and **U16** names the work — trace the reference's skyline and run our own break rule on it — that converts it into a floor. §6.9 gains a row saying the break count's band has no honest source. |
| B2-D1 | P7a's `f = 2Ω sin φ` cannot yield a latitude, and §6.9 lists it as a law that exists | defect | **FIXED.** §6.7 replaces it with the **Hadley cell edge** and the Held–Hou thermal-Rossby scaling, with every input named and an ESTIMATED check: Earth's own numbers give 35.3° against the observed 30°, doubling the spin gives 17.6°, and halving it drives the edge past 90°, which is the one-cell planet. §6.9's row now reads "partly", with the 10–20 % error stated and U12 owed. |
| B2-D2 | §9's record row is false, and the target does need new per-site data | defect | **FIXED**, as A2-D9, with B's consequence carried: a slope-and-moisture asset kit is what §5.1 recommends from Star Citizen's ecosystems, and it needs a biome, a moisture and a slope readable at a site. **D22.** |
| B2-D3 | "Detail to the horizon" is answered with a pixel table, not a content table | defect | **FIXED, and it produced a new decision.** §3.4 gains the live-octave table (rung 5 carries nothing under 1 562 m) and B's deeper observation: the finest live octave is always about 48 CELLS wide at every rung, so the ladder guarantees the same relative smoothness at every distance. That makes "detail to the horizon" a change to the ladder's CONTRACT, and it is now **D21** with Q4 holding the bound it needs. |
| B2-D4 | The four recommended gates are red on the shipped world and no landing plan is given | defect | **FIXED**, as A2-D7 — §6.11 and a rewritten D2. |
| B2-D5 | "Twelve times under even a 0.05° test" is arithmetically wrong | defect | **FIXED.** The sentence is deleted. 0.040° against 0.5° is 12.5× under and against 0.05° is 1.25× under; the strong statement is the break COUNT, and §0 and §6.1 now make that one instead, with the full moon's 0.52° as the scale a reader can feel. |
| B2-D6 | The headline statistic disagrees with itself between sections (0.040° and 0.045°) | defect | **FIXED.** There is now ONE instrument with its parameters stated (§1.4a), §1.6's "Today" row reads from it, and the four variant rows carry an explicit caveat that their skyline column came from the 90 km march and is owed a re-run (U17). |
| B2-D7 | The weather gets no target, no proxy, no cost and no decision | defect | **FIXED.** §4.7 gains a fifth question — **the dormant world**, which B found and which nobody had asked — and **D17** puts the weather to the owner as a decision with three levels. The honest limit is stated rather than hidden: every proxy in §6.7 measures STANDING climate, because a storm is not a function of `(seed, address)`. |
| B2-D8 | The camera set has no light, so its pictures are not reproducible | defect | **FIXED, and the finding is stronger than B stated — with one correction.** B says the light is missing. It is not: `terrain_pictures.rs:45` sets `SUN_ELEVATION_DEG = 25.0` and lines 458-462 point the camera TOWARD the star, which is the worst light for reading relief. So the owner judged three pictures shot into the sun. §1.1 states it as the fourth cause, §7.1 and §7.2 put the star's elevation and azimuth in the rule and the overlay, **U15** measures which light is honest, and **D20** puts it to the owner. |
| B2-W1 | The strength ceiling is broken by its own second row and is the wrong physics | weakness | **FIXED.** §4.2 now carries all three objections — Venus at 112 %, flexure rather than crushing, three different data in one column — and relabels the ceiling *an order-of-magnitude sanity bound, good to about a factor of two*. The defect it catches (0.02 % on a small moon, 600 % on a super-Earth) survives a factor of two easily, and the text says so. |
| B2-W2 | Earth's 19.9 km is a land-plus-ocean span; ours is 17.0 km of almost pure land | weakness | **FIXED, and it strengthens the case.** §0 and §1.3 now compare like with like: Earth's dry-land span is about 9 300 m, ours is 17 003 m of almost pure land, so our planet carries nearly twice Earth's land relief. Both rulers are printed so a reader can see which is which. |
| B2-W3 | The single hump is blamed on the central limit theorem, which the document's own table rules out | weakness | **FIXED.** §0 and §1.3 now give the correct reason: 78.1 % of the amplitude sits in two octaves, so there are about two effective terms, and the measured kurtosis of 2.795 is BELOW 3, which is what a few bounded terms give. **The height distribution is the distribution of one long wave** — which is more useful, because it says no number of fine octaves will ever give it a shelf. |
| B2-W4 | A river needs water above the sea, and nothing in the world can hold it | weakness | **FIXED.** New §3.5 item 4 states it from `body.rs:186-190` and `chunk.rs:207-213`: water is inexpressible above the sea radius, and half this planet's land stands 5.3 km above it. **D19** puts the three questions — record, collider, SL10 — to the owner, and §6.6's basin proxy is corrected to say it counts holes. |
| B2-W5 | Four biomes, no snow, no forest, and no decision row for the vocabulary | weakness | **FIXED in the half that is right; KEPT in the half that is not.** FIXED: `Highland` is a height class, there is no forest, no wetland and no Whittaker axis, and **D18** now asks the owner for the vocabulary — it is a prerequisite for every one of P7a–P7d and for V4's asset kits. KEPT: the claim that *"there is no snow biome to turn to"* is wrong. `Biome::Tundra`'s topsoil is `Stratum::Snow` (`strata.rs:191-196`) and `biome_at` cools with height (`height.rs:52-53`), so a white cap and a polar ice cap DO draw today. §6.7 P7b is corrected accordingly: the shape is computable, the law is not. |
| B2-W6 | Six terms are never explained; §8 still carries no game example; a "FIXED" over-claimed | weakness | **FIXED.** §3.2 gains rms, p90/p99, skew, kurtosis, the central limit theorem, the Hadley cell edge and the Whittaker classification; §3.1 gains the angle of repose as its own row rather than a phrase inside another definition; §8.4 gains a game example (the pilot at S2-b, the 13.5 ms self-check, 862 chunks and 0.35 GB for one facing). The over-claimed "FIXED" is a fair hit and is the reason this table names sections rather than saying "fixed". |
| B2-N1 | §8.4's table mixes a 60° count with 360° columns | note | **FIXED.** The three cost columns are labelled **360°** and the text gives the 60° division: 0.35 GB and 0.4 s at two pixels per cell. |
| B2-N2 | The 405 KB is a rung-0 measurement applied to every rung | note | **FIXED as an unknown; the reasoning is KEPT.** §8.4 now states that M7-3 already gives the structural reason — a chunk is 62³ cells at every rung and the surface crosses about 62² of them whatever the rung — so the bytes are the same ORDER at every rung. That is an argument, not a measurement, so the coarse-rung byte count becomes **U14** and one extraction per rung settles it. |
| B2-N3 | U1's "under 30 s" carries no arithmetic, and the fast suite builds debug | note | **FIXED.** §6.11 carries the release arithmetic per proxy and states the debug problem plainly: the bench states `--release`, or it states a sample count a debug build affords. The document does not choose for the owner. |
| B2-N4 | The ridge station is hand-picked, and D2 adopts a gate for a class Q9 says is undefined | note | **FIXED.** §7.1 gives the station a printed RULE and §1.4a prints the four addresses it produces. The "Earth-like" half stands as Q9: the gate reads THIS planet's four stations, and whether this planet is the reference for Earth-like is still the owner's to answer. |

### 14.3 Round one, refutation A

| # | Finding | Severity | Answer |
|---|---|---|---|
| A-B1 | The picture diagnosis omits `D-TERRAIN-3`, the 403 m patch edge, the rungs and the 0-star sky | blocker | **FIXED.** §1.1 rewritten: the three picture parameters and all three registered causes are stated, and the diagnosis MOVES OFF the pictures onto the field measurements of §1.3–§1.6. §7.2 adds the rung and the draw radius to the overlay; §7.4 lists what is unshootable; U4 is the control. |
| A-B2 | The recommended relief law is refuted by the document's own table, and it has no constant | blocker | **FIXED.** §4.2 rewritten. The `1/gravity` recommendation is WITHDRAWN. The fitted correlations (−0.09 against radius, −0.17 against gravity) are printed. The replacement is a strength CEILING as a bound, calibrated at σ = 234 MPa, with the code measured against it (0.02 % to 600 %). D5 becomes "the law is OPEN" and no one-way door is put to the owner. |
| A-B3 | "The facts travel" breaks the no-drift law and §9 does not notice | blocker | **FIXED.** §4.6 shape 1 now carries the measurement (35 transcendental calls in `taxonomy.rs`, the crate outside the fence) and the condition: every crossing datum must be an integer or a fenced value, and the world identity must grow. §9's SL10 row names it as the one place the target strains SL10. |
| A-D1 | §4.2's relief numbers contradict §1.2; the cap does not bound the relief | defect | **FIXED.** §1.2 prints `body.rs:147-149` and states that the clamp holds the PRE-factor: the range is 100 m to 18 000 m, and the home planet is 0.427 % of its radius. §4.2's table is rebuilt on the true range. |
| A-D2 | §8.2's headroom is overstated; the gate binds on the cave-dense chunk | defect | **FIXED.** §8.2 rewritten against 6.11 ms: 1.89 ms of headroom, about 35 extra evaluations per column, 3.5× not 4.7×. §0 and §9 carry the same number. U9 adds the cave-dense share. |
| A-D3 | §8.1 estimates a column pass the bench prints; the ruling records 710 µs | defect | **FIXED.** §8.1 uses the MEASURED 710 µs and derives 13.2 ns per octave-column (11.5 ns counting `biome_at`), which agrees with the noise bench's 12.19 ns. The subtraction is deleted. |
| A-D4 | "The client holds no worker pool" is false; D13 is closed | defect | **FIXED.** §8.2 states the measured `ThreadedWorkers::start(available_parallelism())` at `terrain.rs:210-214` and ruling V12 S7-6. The open question is restated as the ARRIVAL RATE, and §8.4 prices it. |
| A-D5 | The REPLAYED mark's cited check could not have failed | defect | **FIXED.** The mark's definition names five checks that COULD have failed and did not: the relief, the sea offset, the rung count, the octave count and the 1-in-100 under-water share, all against ruling V9's MEASURED bench values. The `home.rs` unit test is no longer cited as the pin. |
| A-D6 | Almost every `body.rs` citation points at the wrong line | defect | **FIXED.** Every terrain citation re-checked with `grep -n` in this worktree and corrected throughout, including in §13. |
| A-D7 | Ruling V6 A5 (a landform at all eight cube corners) is missing | defect | **FIXED.** Added as §6.8's gate, §7.1 station 7, a §5.6 seam consequence, a §9 law row and decision D11. |
| A-D8 | §5.6 hand-waves sibling halos and the face seam | defect | **FIXED.** §5.6 rewritten with three named unknowns: the coarse rung's stencil (10 584 chunks at rung 11), the sibling halo (the extractor's measured 35 %), and the cube-face seam and corner. "Eleven ancestors and nothing else" is withdrawn, and the feasibility claim is demoted. |
| A-D9 | A height field cannot hold a hoodoo, an arch or an undercut | defect | **FIXED.** New §3.5 states the structural limit, §3.1's hoodoo row points at it, and §9's collider row repeats it. |
| A-D10 | A cave mouth is impossible today, so ruling V5's picture cannot be taken | defect | **FIXED.** §3.5 states the 8 m minimum roof (`chunk.rs:377-383`, `body.rs:219`) and §9 adds a V5 row. |
| A-D11 | §1.4's `4A/L` is a mean slope, and §6.6 escalated it to "provably" | defect | **FIXED.** §1.4 rewritten around two measurements: `\|∇ noise3\|` with a sampled maximum of 2.97 (so `4A/L` is a looser bound, and the octave sum gives 28.1°), and the slope distribution over 200 000 columns (max 8.74°, zero over 45°). §6.6 now says "0 in 200 000 columns". U2 owns the true supremum. |
| A-D12 | A finer octave floor is a one-way door too | defect | **FIXED.** D6 now carries the arithmetic: `body.rs:182` normalises by the weight sum, a fifteenth octave moves the coarsest amplitude by 0.099 m, and the identity tolerance is zero. |
| A-W1 | β is per-body, and the fBm model is outside its range at H > 1 | weakness | **FIXED.** §1.5 prints β across the `k_rough` draw (2.72–3.30) and states that H > 1 means "smooth, not fractal". |
| A-W2 | 14 304.9 m is a bound, not a realised relief | weakness | **FIXED.** §1.2 marks it as a bound; §1.3 gives the measured realised span (17 003 m) against Earth's 19 900 m. §0's over-claim is deleted. |
| A-W3 | "The ground rises 688 m over a 20 km walk" is a bound, not a walk | weakness | **FIXED.** §2 replaces it with the measured median slope: a 373 m climb takes 15.7 km of walking. |
| A-W4 | The haze has no owner | weakness | **FIXED.** New §2.1 states the measured absence of any fog, haze or atmosphere in the client, and D13 asks who owns it. |
| A-W5 | The headline gate contradicts the headline rule (a flat six breaks) | weakness | **FIXED.** §6.9 lists which bands have a law and which do not; the skyline break count is named as one that has none, and D3 is qualified to match. |
| A-W6 | §9 over-claims what the proxies read | weakness | **FIXED.** P1 no longer reads a frame (§6.1), and §9's SL10 row states the two honest exceptions (P4's flow router and P5's spectrum estimator are new models and must be checked on a known field). |
| A-W7 | The 8 ms is per CHUNK at any rung, not per rung-0 chunk | weakness | **FIXED.** §8.1, §8.2 and §9 all say "per chunk, gated on the costliest named chunk at any rung", and §5.6 notes that the coarse rung pays its network work inside its own budget. |
| A-W8 | "The grep returns NOTHING" is MEASURED and returns two hits | weakness | **FIXED.** §4.5 states both hits and what they are. |
| A-W9 | §8.3 mixes a count with the cheapest unit cost | weakness | **FIXED.** §8.3 is a three-row table spanning 0.24 ms to 2.1 ms, with the assumption behind each row named, and the ancestor count corrected to 0.333. |
| A-W10 | "At 50 km the ladder draws rung 9" assumes a rule slice 8 owes; the stations are unqualified | weakness | **FIXED.** §3.4 corrects the rung (5 or 6, with the pixel table), §7.4 lists every station against `D-TERRAIN-3`, and D15 and U11 own the missing screen-space rule. |
| A-W11 | Open decisions are called rulings | weakness | **FIXED.** §9 and §13 now say `03_generator_sl10.md`'s D11 and D13 are OPEN decisions, and that the radius move is ruling V6 Part D and is not yet built. |
| A-W12 | §1.3's "about 90 m" is not reproducible from its own rule | weakness | **FIXED.** The construct is deleted. §1.3 reports measured spans and plane departures instead. |
| A-N1 | The literature citations are unverified, and gates are built on them | note | **FIXED.** §13 states it in bold and adds the rule that no band from §5.2 becomes a numeric gate before the paper is opened (U12). |
| A-N2 | Q5 (does the ocean exist) is larger than a question | note | **FIXED.** §1.3 measures it (0.99 % under water, median land 5.3 km above the sea), §3.1's coast row carries it, and Q5 is rewritten. |
| A-N3 | The overlay does not state the rung | note | **FIXED.** §7.2 adds the rung line and the draw radius, and names it as the line that would have prevented A-B1. |

### 14.4 Round one, refutation B

| # | Finding | Severity | Answer |
|---|---|---|---|
| B-B1 | The cost case prices the cheapest chunk, not the worst | blocker | **FIXED**, as A-D2. §8.2 uses 6.11 ms. B's own arithmetic reached 21 extra evaluations using revision 1's 22 ns; with the MEASURED 13.2 ns the figure is 35, and §8.3 shows the channel memory can still consume all of it. |
| B-B2 | D5's relief law is refuted by the document's own table, and the replacement explodes | blocker | **FIXED**, as A-B2. The fitted slopes, the Vesta explosion and the "relief is set by the process" objection are all carried into §4.2, and §4.1's chain gains the building half. |
| B-B3 | §5.6 hand-waves flow accumulation; the top rung is 10 584 chunks | blocker | **FIXED**, as A-D8. The 10 584 figure and the face-seam example are in §5.6, and the feasibility claim is demoted to a candidate with three unknowns. |
| B-D1 | "At 50 km the ladder draws rung 9" contradicts the 36 m floor by 16× | defect | **FIXED.** §3.4 carries the pixel-per-cell table and names rung 5–6. |
| B-D2 | The document never prices a FRAME or a BOOT | defect | **FIXED.** New §8.4: the chunk count, the wall time on one and on fourteen threads, and the mesh bytes, at 1, 2 and 4 pixels per cell. The finding that MEMORY is the wall (2.1 GB at 2 px/cell) is new, and D15 puts the screen-space rule to the owner. |
| B-D3 | "No cliff … provably" rests on a constant nobody derived | defect | **FIXED**, as A-D11, with the measured `\|∇ noise3\|` and the measured slope distribution. |
| B-D4 | The headline gate reads a GPU frame, contradicting §9's SL10 row | defect | **FIXED.** §6.1 is rewritten as a ray-march through `height_m` — exactly the alternative B proposed — and it produced revision 2's headline number (0.040°, zero breaks), which revision 3 then replaced with a converged four-station measurement (§1.4a). B's three consequences (renderer coupling, backend variance, no place in the fast suite) are the stated reason. |
| B-D5 | §4.2's relief table drops the seed factor | defect | **FIXED**, as A-D1. |
| B-D6 | The magic number that CAUSES the headline is never named | defect | **FIXED.** §1.2 and §9 name `LONG_WAVE_CAP_M = 400 000` as a typed constant that saturates for every seed on this body, plus the 20 000 m floor. |
| B-D7 | Every `body.rs` citation is wrong | defect | **FIXED**, as A-D6. |
| B-D8 | "The world holds no spin" is not what the code says; SL1 names the owner | defect | **FIXED.** §4.5 states both halves: no body draws a value, and `frame.rs:66` already carries `angular_velocity` on every frame with the Coriolis term at `frame.rs:243-273`, filled from `spin_radps` at `placement.rs:149`. D10 and Q2 are rewritten around the placement lane. |
| B-D9 | Six proxies, and none measures a biome, a climate or the weather | defect | **FIXED.** New §6.7 adds P7a–P7d (the desert belt, the snow line, the rain shadow, two planets of one size), each reading `biome_at` and `height_m`. §11 D8 and D13 and §12 Q6 carry the rest. |
| B-D10 | §4.1's chain has no arrow for the process that BUILDS relief | defect | **FIXED.** §4.1 is redrawn with a building column (uplift, plumes, impacts, isostasy) set against the removing column, and §3.1 gains a tectonic-uplift row and an isostasy row. |
| B-D11 | The weather gets one paragraph, and the boundary question is never asked | defect | **FIXED.** §4.7 gains the four questions (what crosses, who authors, does it change the shape, what does it cost), §9 gains an SL6 row, and Q6 carries them to the owner. |
| B-D12 | The 19 strata never enter the target | defect | **FIXED.** §3.5 states what is built, §2's table asks for bedding, §6.8 makes it a proxy, §7.1 station 4 photographs it, and D12 puts it to the owner. |
| B-W1 | Ten terms are used and never explained | weakness | **FIXED.** New §3.2 glosses fBm, octave, ridged multifractal, billowed noise, domain warping, the analytic derivative, flow accumulation, stream power, slope–area, drainage density, Jeans escape, the cosmic shoreline and hypsometry, each with a game example. Horton–Strahler, Hack's law, the bifurcation ratio, hydraulic erosion and tectonic uplift are named in §3.1. |
| B-W2 | §5, §6 and §8 carry no example in the game's words | weakness | **FIXED.** Examples added to §1.6, §5.1, §5.6, §6.2 and §6.4 — including B's own slope–area sentence, which was better than mine. |
| B-W3 | "Not shippable" is asserted, never measured | weakness | **FIXED.** §5.5 gains a compressed column (UNMEASURED, 3–6×), the "whole sphere is the wrong unit" option, and a rewritten conclusion; D14 puts it to the owner. |
| B-W4 | The new SL8 rule contradicts the ladder | weakness | **FIXED.** §9's SL8 row is restated in screen space, and D16 puts the wording to the owner. |
| B-W5 | D3(b) recommends per-body bands, and §6 gives no law | weakness | **FIXED.** New §6.9 lists band by band which has a law; θ is noted as a property of the stream power law rather than of Earth; β and the slope distribution are marked as having no published law, with U12 to close it. |
| B-W6 | "Forty-seven metres out of forty-eight carry no shape" teaches a false picture | weakness | **FIXED.** Deleted. §1.3's measured plane departure (0.26 m rms inside 50 m) replaces it, and the text says the field is a smooth ramp, not a staircase. |
| B-N1 | Σ 1/4^L is 0.33 ancestors, not 1.33 | note | **FIXED.** §5.6 and §8.3 use 0.333. |
| B-N2 | The 22 ns mixes two cell counts and ignores two noise calls | note | **FIXED**, superseded by the MEASURED 710 µs; §8.1 states both divisors (14 octaves and 16 field evaluations). |
| B-N3 | The 12.19 ns bench figure carries a caveat | note | **FIXED.** §8.1 uses the crate's own 710 µs as the primary number and quotes 12.19 ns only as a cross-check. |
| B-N4 | A climatologist would not accept the snow line as written | note | **FIXED.** §4.3 states both objections (the ⅔ is Earth's water; the real quantity is the equilibrium line altitude) and demotes the table to a first estimate. |
| B-N5 | The home planet's horizon is 27 % closer than Earth's — a good owner-facing fact | note | **FIXED.** Stated in §2 and in Q9. |

### 14.5 What the round-one refuters checked and could not break

Both replayed the octave table, the relief, `k_rough`, the 78.1 % share, the sea radius, β, the ladder
arithmetic, the horizon table, the pixel arithmetic, the shipped-map byte sizes, the `taxonomy.rs`
citations, the 4:1 chunk nesting, `octaves_at` and `dropped_bound_m`, and the refusal of the learned
approach under SL10. Those parts of revision 1 stand unchanged in revision 2, and every number above that
carries no correction was reproduced twice independently.

### 14.6 What the round-two refuters checked and could not break

Refutation A ran its own independent Python replay of `SplitMix64`, `child_seed`, `top_rung_for`, `snap`,
`Ladder::for_radius`, `draw_unit`, `BodyDefinition::from_seed`, `noise3` and `height_m`, and reproduced
the ladder radius, N, the rung and octave counts, the relief to the last digit, `k_rough`, the saturated
coarsest wave, the sea offset, the four coarsest amplitudes, the slope percentiles and the hypsometry.
Refutation B re-derived the relief clamp, the octave table and its weight sum, the `LONG_WAVE_CAP_M`
saturation, the horizon table, the pixel arithmetic, the ladder's 10 584 coarse chunks and `Σ 4⁻ᴸ`, the
frame integral, the 8 ms budget and its 1.89 ms of headroom, the client's worker pool, the overhang and
cave-mouth limits, `POLE_AXIS`, `DEFAULT_RADIUS`, σ = 234 MPa, and the cube corner's 35.264°.

**Neither could break these, and revision 3 changes none of them:** the body's own numbers (§1.2), the
octave table and its 78.1 % share, the plane-departure and hypsometry statistics (§1.3), β and the
band-limited reading (§1.5), §1.6's direction — that roughness buys steepness and not skyline — the
horizon and pixel geometry (§3.4), the six-body relief table and the withdrawal of the `1/g` law (§4.2),
the shipped-map byte sizes (§5.5), the ladder's own arithmetic (§5.6), the cost base (§8.1), the frame
integral (§8.4), the refusal of the learned approach under SL10, and the move of the headline gate off a
rendered frame onto `height_m`, which refutation B called the single best change in revision 2.
