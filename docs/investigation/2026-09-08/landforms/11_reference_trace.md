# 11 — THE REFERENCE, TRACED BY OUR OWN RULE (U16)

**Date:** 2026-09-16. **Owed by:** ruling T1 in `docs/design/owner_decisions_2026-09-16_terrain.md`
(*"The reference as a number"*), and `01_reference_target.md` §10 U16.

**What it answers.** `01` §2.2 says the reference's *"8–12 silhouette breaks per 60°"* was counted by a
person's eye, and our *"zero"* was computed by a stated rule. Two different instruments, so the ratio
between them means nothing, and P1 could land only as a ratchet. This document traces the reference
picture's skyline and runs **the same break rule** over it. 8a now has a calibrated band.

**The picture.** `/Users/maxim/Downloads/crimson desert.jpeg` — the owner's own reference. It stays
outside this repository; it is a screenshot of a commercial game. Every figure names it by path.

**Every number carries a mark.**
- **MEASURED** — a program in the scratchpad produced it, and the program is named.
- **ESTIMATED** — arithmetic on a measured number, with the arithmetic shown.
- **ASSUMED** — a value nobody can read off the picture. Every assumption is swept, and the sweep is
  printed, so the reader sees how far the answer moves instead of trusting a choice.

---

## 0. The answer, in short

**MEASURED.** Traced by our own rule, over 70 runs of the instrument's own settings (§5):

| Per 60° of bearing | Reference (this trace) | Our world TODAY (M1b) | The SPECTRUM alone (M1b) | The SPECTRUM + the RIDGED band (M1b-2) |
|---|---|---|---|---|
| Breaks at 0.05° | **4.0 – 8.9** | 0.17 – 0.33 | 2.67 – 3.83 | 3.50 – 5.33 |
| Breaks at 0.10° | **2.5 – 7.9** | 0 | 0.83 – 1.83 | 0.83 – 3.50 |
| Breaks at **0.25°** | **2.5 – 5.5** | 0 | 0 | 0 – 1.17 |
| Breaks at **0.50°** | **0.0 – 3.3** | 0 | 0 | 0 – 0.33 |
| Largest rise over the ±5° trend | **0.43° – 1.07°** | 0.054° – 0.064° | 0.138° – 0.168° | 0.233° – 0.689° |

Three readings follow. Each is a measurement that could have failed.

1. **The eye and the rule agree, and that is the finding U16 was asked for.** A person counted 8–12
   orienters per 60°. Our rule, over the same picture at its loosest threshold, counts 4.0–8.9 per 60°,
   and the count is a **lower bound** (§3, the edge loss). `01` §2.2 was right that 8–12 : 0 proved
   nothing. The honest comparison is 4.0–8.9 : 0.17–0.33, and it is worse, not better.
2. **The ridged middle band already reaches the reference's TALLEST feature.** The reference's largest
   rise is 0.43°–1.07°. M1b-2 measured 0.233°–0.689° from the proposed spectrum plus a ridged band at
   octaves 5..=9. At station S2-d (0.689°) our predicted field stands inside the reference's own band.
   **Height is nearly bought by the spectrum.**
3. **The reference's DENSITY is three to six times ours, and that gap is the whole of 8c.** At 0.25° the
   reference holds 2.5–5.5 breaks per 60°; the ridged prediction holds 0–1.17. At 0.50° the reference
   holds up to 3.3; the prediction holds up to 0.33. A per-column noise makes ONE tall crest where a
   range and its drainage make a LINE of them. **The missing term is the macro layout, not more noise.**

**Example, in the game's words.** The pilot stands on station S2-b's hill, 373 m over the plain. Under
the spectrum with the ridged band she sees one crest as tall as the reference's tallest, and about one
more that the eye can name in the whole 60° in front of her. In the reference she would see four.

---

## 1. What the picture is, before anything is measured

The file is **not a screenshot**. It is a photograph of a monitor, and three facts prove it.

- **MEASURED** (`prep.py`): the screen's top bezel crosses the frame as a straight dark edge, from row
  −0.3 at column 0 to row 19.5 at column 1079. Fitted over 986 columns the tilt is **1.0195°** with a
  residual of **0.84 px rms**. A screenshot has neither a bezel nor a tilt.
- **MEASURED**: the left, right and bottom edges of the screen are not in the frame. Only the top is. So
  the photograph is a crop of the screen, not the whole of it.
- The colours are washed and the JPEG carries photographic noise. §5 prices that.

**The consequence for the field of view, and it is a hard bound.** The photograph is 1080 × 810 — four
by three. A monitor is sixteen by nine (ASSUMED; an ultrawide would tighten the bound). Let `f` be the
share of the screen's width the photograph holds. The screen's height, in the photograph's own pixels,
is `1080/f × 9/16 = 607.5/f`. The photograph holds 810 rows and does not reach the screen's bottom edge,
so `810 < 607.5/f`, hence **`f < 0.75`** (ESTIMATED; the arithmetic is shown).

The photograph's own angular width is `2·atan(f · tan(game FOV / 2))`:

| `f` | game FOV 60° | 65° | 70° | 75° | 90° |
|---|---|---|---|---|---|
| 0.75 | 46.8° | 51.1° | **55.4°** | 59.8° | 73.7° |
| 0.70 | 44.0° | 48.1° | 52.2° | 56.5° | 70.0° |
| 0.65 | 41.1° | 45.0° | 48.9° | 53.0° | 66.0° |

**ASSUMED: the photograph spans 45° to 65° of bearing, and 55° is the base.** Fifty-five degrees is what
`f = 0.75` with a 70° third-person camera gives. Every result in §4 is printed at 45°, 50°, 55°, 60° and
65°, so the reader sees the sensitivity instead of trusting the base.

**This is why the count is stated per 60°.** The picture never held 60° of bearing. After the rule's own
edge loss (§3) it carries 38° to 60°. Every count is scaled to 60° from the usable span, and the usable
span is printed beside it.

---

## 2. How the skyline was traced, and how cloud was told from ridge

**Trace A is the primary instrument, and it is a HAND trace.** The de-rotated picture was cut into nine
overlapping panels and enlarged five to twenty-two times, each with a labelled pixel grid
(`strips.py`, `z_a`, `zoom_summit`, `z_b`, `z_c`, `z_d`, `z_e`, `z_f`, `z_spires`, `z_peak310`). The
crest was read by eye at **134 control points**, spaced three to twenty columns apart, closer where the
silhouette turns. §7 prints all 134. Between them the crest is linear.

**A bounded colour refinement then sharpens it.** For each column the rule looks in the band from three
rows above the hand reading to six rows below, and takes the first row that is ground by the colour
test. Outside that band it cannot move. **MEASURED: the refinement moved 901 of 1080 columns, by a mean
of −0.25 px and an rms of 2.71 px.** So the hand reading sets the shape and the colour test sets the
edge — it cannot invent a feature the eye did not see.

**Trace B is an independent check**, built the other way round: a coarse hand ceiling, then one colour
threshold searching downward with no upper bound. §4 prints both families. They agree.

**How cloud was separated from ridge, and the honest answer.** Cloud is not the hard part. The hard part
is that **the blue sky and a hazy distant ridge are the same colour**, because both are scattered air.
MEASURED over named boxes of the de-rotated picture, with the score `s = (B − R) − 0.6 × luminance`:

| Region | R, G, B (median) | B − R | luminance | score `s` |
|---|---|---|---|---|
| blue sky, upper left | 158, 183, 197 | +37 | 178 | **−70** (its bluest pixels reach −43) |
| cloud | 194, 185, 184 | −17 | 186 | −128 |
| haze bank, right | 178, 178, 174 | −5 | 176 | −111 |
| the far snowy range | 115, 176, 208 | +90 | 166 | −10 |
| the mid blue ridge | 60, 146, 195 | +141 | 135 | +60 |
| **the far-right ridge under the haze** | 142, 172, 183 | **+41** | **166** | **−58** |

The score separates cloud and haze from every ridge by a wide margin. It does **not** separate the blue
sky (−70, and −43 at its bluest) from the far-right ridge under the haze (−58). **No single colour
threshold can, anywhere in this frame.** That is exactly why the primary trace is a hand trace: the eye
settles what the colour cannot, and the colour then places the edge to a pixel or two.

**It matters, and it is measured.** Trace B, run at the threshold that is right for the left half,
misses the far-right rock spires at columns 1000–1030 by about 30 px, because there the ridge scores
−58 to −72 while the blue sky elsewhere scores −70. The hand trace holds them. A single-threshold
instrument would have understated the reference.

**The overlay for the owner's eye:**

```text
<scratchpad>/reference_skyline_overlay.png
```

It shows the traced skyline in red over the de-rotated picture, one tick per 1° bearing sample, and a
coloured tick with its threshold at every bearing the rule calls a break. **The overlay carries the
reference picture, so it stays in the scratchpad and never enters this repository.** §8 gives the full
path.

---

## 3. The rule, applied exactly as the march applies it

The break rule is `crates/bins/examples/skyline_march.rs` (`judge`), which is `01` §6.1:

```text
   sample the skyline on a 1-degree bearing grid
   rise(b)  = skyline(b) − median( skyline(b−5) .. skyline(b+5) )
   a BREAK  = a bearing whose skyline stands over BOTH neighbours AND whose rise is over the threshold
   thresholds: 0.05, 0.10, 0.25, 0.50 degrees
```

Five points where a picture forces a choice the march never faces. Each is stated, not hidden.

1. **From pixel to angle.** The picture is a rectilinear projection. For a camera with no roll, pitched
   `p` degrees down, principal point `(x_c, y_c)`, focal length `F` in pixels:

   ```text
   F         = (1080/2) / tan(photo span / 2)
   elevation = asin( ( (y_c − y)·cos p − F·sin p ) / sqrt( (x − x_c)² + (y_c − y)² + F² ) )
   bearing   = atan2( x − x_c , −(y_c − y)·sin p + F·cos p )
   ```

   The square root is what makes this a projection and not a ruler: at the frame's edge the same pixel
   step is a smaller angle than at its centre. A flat pixels-to-degrees conversion overstates every rise
   near the frame's edge by about 16 % at the 65° assumption.

2. **The principal point is ASSUMED to be the photograph's own centre, (540, 405).** The photograph is a
   crop, so the frame's true centre is unknown. It shifts the pitch and nothing else, and §5 measures
   what the pitch costs.

3. **The camera's pitch is ASSUMED at 15° down.** ESTIMATED from the scene: the far range sits at rows
   150–200, and it must lie near 0° of elevation, because a range 30–60 km away cannot stand 15° high.
   Solving for the pitch that puts it there gives 14.7°. §5 sweeps 0° to 30°.

4. **The rule's edge loss, and why the counts are a LOWER BOUND.** The march wraps around a full 360°
   horizon; a picture does not. The first five and the last five bearings carry no complete ±5° window,
   so they are dropped, and the usable span is the traced span less 10°. MEASURED: at the 55° base the
   far-right rock spires at column 1010 sit at bearing **+26.1°**, outside the usable ±24.5°. **The
   picture's most obvious cluster of orienters never enters a count.** Every count below therefore
   understates the reference.

5. **The sample count.** The march reads 360 bearings; the picture gives 38 to 60. A break rule that
   reads a strict local maximum on a coarse grid loses one whenever a crest falls between two samples.
   This is why the count moves by one or two between field-of-view assumptions in §4: the grid lands on
   different columns. It is a property of the picture, not of the rule.

**The camera's height above the valley floor does not enter this arithmetic at all.** The break rule
reads angles. The height matters only when angles become metres; there it is ASSUMED at 300–800 m, in
line with `01` §2's own 373 m estimate for a 50 km view, and no number in this document rests on it.

---

## 4. The result

**MEASURED**, trace A (the hand trace with the −85 refinement), pitch 15°:

| Photo span ASSUMED | Traced span | Usable span | Largest rise over the ±5° trend | Skyline elevation range | Breaks per 60° at 0.05° | 0.10° | 0.25° | 0.50° | raw counts |
|---|---|---|---|---|---|---|---|---|---|
| 45° | 47.0° | 38.0° | 0.642° | 5.06° | 6.32 | 6.32 | 4.74 | 1.58 | 4 / 4 / 3 / 1 |
| 50° | 53.0° | 44.0° | 0.877° | 5.62° | 4.09 | 4.09 | 4.09 | 1.36 | 3 / 3 / 3 / 1 |
| **55° (base)** | **59.0°** | **50.0°** | **0.716°** | **6.09°** | **4.80** | **3.60** | **3.60** | **1.20** | 4 / 3 / 3 / 1 |
| 60° | 63.0° | 54.0° | 0.826° | 6.60° | 6.67 | 6.67 | 4.44 | 2.22 | 6 / 6 / 4 / 2 |
| 65° | 69.0° | 60.0° | 0.952° | 7.22° | 6.00 | 6.00 | 5.00 | 2.00 | 6 / 6 / 5 / 2 |

**The envelope**, over both trace families and every setting of each (MEASURED, `judge2.py`):

| Quantity | Trace A, 40 runs (lowest / median / highest) | Trace B, 30 runs | **BOTH, 70 runs** |
|---|---|---|---|
| Largest rise | 0.642 / 0.907 / 1.067 | 0.431 / 0.668 / 0.966 | **0.431 / 0.820 / 1.067** |
| Skyline range | 4.87 / 6.08 / 7.22 | 5.03 / 6.27 / 7.49 | **4.87 / 6.11 / 7.49** |
| Breaks/60° at 0.05° | 4.00 / 5.51 / 7.89 | 4.09 / 6.91 / 8.89 | **4.00 / 6.00 / 8.89** |
| Breaks/60° at 0.10° | 3.60 / 4.82 / 7.89 | 2.45 / 5.51 / 7.89 | **2.45 / 5.23 / 7.89** |
| Breaks/60° at 0.25° | 2.45 / 4.09 / 5.00 | 2.45 / 3.50 / 5.45 | **2.45 / 4.00 / 5.45** |
| Breaks/60° at 0.50° | 1.20 / 1.58 / 3.33 | 0.00 / 2.00 / 3.00 | **0.00 / 2.00 / 3.33** |

Trace A's runs are the four refinement variants × two median widths × five field-of-view assumptions;
trace B's are three colour thresholds × two median widths × five assumptions.

**Every break the rule finds is a feature a player would name.** At the base setting the four breaks
are, in the picture's own columns (MEASURED; each one is ringed in the overlay):

| Bearing | Column | Rise | What it is in the picture |
|---|---|---|---|
| −21° | 181 | 0.332° | the broad summit of the far snowy range, left of frame |
| −13° | 324 | 0.474° | the pointed summit where the snow ridge turns |
| −2° | 507 | 0.084° | a crest on the middle range |
| **+6°** | **639** | **0.716°** | **the pyramid peak** — the tallest silhouette in the frame |

Not one of them is a cloud, a snow cap or a haze band. **The rule counted exactly what a person would
call an orienter**, which is the whole point of U16. What the rule did NOT count, because the edge loss
dropped it, is the rock-spire cluster at columns 1000–1030 (§3, point 4).

---

## 5. How far the answer can move, and the instrument's own noise

**The refinement threshold** (trace A at −70, −85, −100): the per-column trace moves by a median of
**2.0 px**, a p90 of 5 px and a maximum of 9 px (MEASURED). On the quantity that decides a break — the
rise over the ±5° window median — it moves by a median of **0.051°**, a p90 of 0.161° and a maximum of
0.265° (MEASURED). Dropping the refinement entirely, and judging the raw hand crest, changes the rise by
0.239° rms; that variant is the coarsest of the four and it is inside the envelope above.

**The colour threshold** (trace B at −45, −55, −70): per-column median 3 px, p90 13 px, rms 5 px; the
rise moves by a median of 0.052° and an rms of 0.055°–0.135° (MEASURED). The window median removes the
slow part of the error, which is why the per-column spread is far worse than the per-break spread.

**The median filter width**: five columns against nine changes no count by more than 1.5 per 60°.

**The camera pitch** (trace B base, 55° span; MEASURED — trace A behaves the same):

| Pitch ASSUMED | Largest rise | Range | Breaks/60° at 0.05 / 0.10 / 0.25 / 0.50 |
|---|---|---|---|
| 0° | 1.027° | 6.36° | 8.00 / 5.33 / 5.33 / 4.00 |
| 10° | 0.799° | 6.29° | 10.21 / 7.66 / 5.11 / 3.83 |
| **15°** | 0.588° | 6.27° | 8.57 / 7.35 / 4.90 / 2.45 |
| 20° | 0.898° | 6.21° | 9.23 / 6.92 / 5.77 / 2.31 |
| 30° | 0.651° | 6.34° | 9.15 / 7.12 / 3.05 / 1.02 |

Over a 30° sweep of an ASSUMED angle the 0.25° count stays between 2.2 and 5.8. **The pitch does not
decide the answer**, and neither does the principal point, which only shifts the pitch.

**What this means for each threshold:**

| Threshold | The instrument's own rise noise | Verdict |
|---|---|---|
| 0.05° | 0.051° median spread — the SAME size | **unusable as a band.** Report it; never gate on it. |
| 0.10° | inside the p90 spread of 0.161° | **marginal.** Report it. |
| 0.25° | 1.5–5 × the median spread, above the p90 | **usable.** |
| 0.50° | 3–10 × the spread | **usable, and the firmest number here.** |

---

## 6. What this trace counts that our rule on a height field CANNOT

The trace and the march measure the same quantity — a silhouette against the sky — so they compare. They
are not identical, and several of the reference's breaks are made of things our height field cannot hold
today. Naming them is the difference between a calibrated band and a wish.

**In the reference, and NOT in a height field:**

- **Rock spires and pinnacles.** Columns 1000–1030 carry a line of thin rock needles, and columns
  697–730 carry a serrated crest of them. A height field gives one height per column, so a needle
  thinner than a cell cannot exist. These belong to **8f, the third dimension** (arches, overhangs,
  pillars, placed rock objects). MEASURED: the serrated crest alone puts five notches of 6–8 px into
  40 columns — about 2.2° of bearing at the base assumption.
- **The cliff at columns 741–746.** The silhouette drops from row 200 to row 226 — **26 px, about 1.4°
  at the base assumption — in five columns.** `01` §1.4 measured the steepest ground anybody has found
  on our planet at 11.15°. A near-vertical silhouette step is **8a's cap-rock bench and 8d's
  stratigraphy**, not a noise octave.
- **The tree canopy.** Every ridge from column 740 rightward is forested. A canopy of 15–30 m stands on
  the silhouette. That is ruling V10's canopy fold, and it belongs to **8e**.
- **A settlement.** Roofs and towers sit on the middle ground. They are live state, never the seed.

**In the reference, counted by a PERSON, and counted by NEITHER instrument:**

Snow caps, the tree line, the colour change from green to blue with distance, the haze bands, cloud
shadow on a slope, and the low sun's long shadows. `01` §2.2 warned about exactly this, and the warning
holds: a person's 8–12 includes them; the rule's 4.0–8.9 does not. **That the two numbers still land in
the same place says the silhouette alone carries most of the reference's structure.** It does not make
the colour cues free — `01` §2.1 records that the haze has no owner, and D13 is still open.

**What the trace EXCLUDES, deliberately:**

- **The foreground pillar and the character on it.** They are placed objects, not the skyline. The
  exclusion needs no special case, and it is MEASURED rather than argued: over the pillar's columns
  (450–700) the traced skyline never goes below row 210, the character's banner begins at about row 400,
  and the pillar's top is at about row 468. **The pillar stands 190 px below the skyline and never
  enters a count.**
- **Clouds and the haze deck**, by the colour test of §2.
- **The dark screen bezel** at the frame's top, by the hand crest.

---

## 7. The trace itself, so the measurement outlives the picture

Two tables carry the whole measurement. Anyone can re-run the break rule from them without the owner's
picture.

**The hand-read crest, 134 control points as `(column, row)`** in the de-rotated 1080 × 810 frame:

```text
(0,150) (20,150) (40,149) (60,150) (80,151) (100,151) (120,152) (140,151) (150,149) (160,148)
(170,146) (180,146) (190,146) (200,148) (210,150) (220,152) (230,154) (240,156) (250,157)
(260,162) (270,167) (280,168) (290,167) (300,166) (305,165) (310,163) (315,161) (320,159)
(325,157) (330,157) (335,160) (340,163) (345,166) (350,169) (360,172) (370,173) (380,173)
(390,174) (400,174) (410,176) (420,181) (430,184) (440,189) (450,196) (460,204) (470,212)
(480,211) (490,207) (500,202) (510,199) (520,198) (530,196) (540,197) (550,204) (560,207)
(570,208) (580,206) (590,197) (600,193) (610,190) (615,187) (620,183) (625,174) (630,172)
(635,172) (640,173) (645,175) (650,178) (655,181) (660,185) (665,187) (670,188) (675,189)
(680,187) (685,187) (690,190) (697,199) (701,193) (706,200) (710,192) (714,199) (718,193)
(722,200) (726,197) (730,202) (735,199) (740,200) (743,210) (746,226) (750,224) (755,231)
(760,233) (765,236) (770,237) (775,238) (780,238) (790,242) (800,245) (810,246) (820,246)
(830,251) (840,254) (850,256) (860,257) (870,258) (880,260) (890,261) (900,262) (910,264)
(920,265) (928,259) (935,259) (940,260) (950,258) (958,256) (968,256) (978,254) (988,253)
(996,251) (1002,249) (1006,245) (1010,242) (1014,244) (1018,250) (1024,251) (1030,251)
(1036,253) (1042,255) (1048,258) (1054,259) (1060,256) (1066,251) (1072,247) (1079,248)
```

**The finished trace A**, one row per ten columns (columns 0, 10, 20 … 1070):

```text
147 147 147 147 146 147 147 149 150 150 148 152 155 152 151 154 145 143 143 143 145 150
149 151 153 158 160 164 165 166 163 162 159 154 160 166 169 170 173 171 171 174 178 181
186 193 201 209 208 204 202 196 197 193 198 201 208 209 206 197 193 187 181 173 177 180
186 188 186 188 198 192 196 200 200 224 238 242 240 242 246 248 247 254 256 258 258 259
262 263 264 264 266 262 262 258 256 256 259 252 250 242 252 256 257 263 256 254
```

**The skyline on the 1° bearing grid**, at the 55° base and a 15° pitch, in degrees of elevation
(bearing −30 … +29; the outer five at each end carry no window and are dropped by the rule):

```text
-0.92 -0.93 -0.89 -0.94 -1.11 -1.00 -1.35 -1.13 -0.86 -0.78 -0.89 -1.08 -1.24 -1.60 -1.86
-1.97 -1.77 -1.39 -1.71 -2.25 -2.21 -2.27 -2.43 -2.69 -3.10 -3.76 -4.29 -4.02 -3.66 -3.74
-3.71 -4.25 -4.24 -3.69 -3.31 -2.67 -2.59 -3.01 -3.16 -3.16 -3.50 -3.61 -3.95 -5.52 -5.81
-5.98 -6.33 -6.49 -6.65 -6.66 -6.85 -6.88 -6.69 -6.57 -6.32 -6.31 -5.62 -6.14 -6.50 -5.79
```

---

## 8. Where the instrument lives

The scripts sit in this session's scratchpad. They are small and they hold no game code.

```text
<scratchpad>/prep.py           de-rotate, and the colour statistics of the named boxes
<scratchpad>/strips.py         the gridded enlargements the crest was read from
<scratchpad>/crest.py          TRACE A: the 134 hand control points + the bounded refinement
<scratchpad>/trace4.py         TRACE B: a hand ceiling + one colour threshold, searching downward
<scratchpad>/judge.py          pixel -> (bearing, elevation), and 01 §6.1's break rule
<scratchpad>/judge2.py         the 70-run sweep
<scratchpad>/final_overlay.py  the overlay for the owner's eye
<scratchpad>/reference_skyline_overlay.png     THE OVERLAY
```

`<scratchpad>` is
`/private/tmp/claude-501/-Users-maxim-Projects-my-voxeldust--claude-worktrees-warp/028caf03-05b1-4acd-a128-3f67d796b1d3/scratchpad`.
It does not survive the session. §7's tables are the part that must survive, which is why they are
printed here. Re-running the trace needs the owner's own copy of the picture, and that copy never enters
this repository.

---

## 9. Beside M1b's table

`09_step0_results.md` M1b and M1b-2, with the reference's own row added. Both rows now come from the
same rule, over the same thresholds, and each says how far it can move.

| | Largest rise | Breaks/60° at 0.05° | at 0.10° | at 0.25° | at 0.50° | Skyline range |
|---|---|---|---|---|---|---|
| Our world TODAY, 4 stations | 0.054° – 0.064° | 0.17 – 0.33 | 0 | 0 | 0 | ~2.5° over 360° |
| The SPECTRUM alone | 0.138° – 0.168° | 2.67 – 3.83 | 0.83 – 1.83 | 0 | 0 | ~2.5° over 360° |
| The SPECTRUM + the RIDGED band | 0.233° – 0.689° | 3.50 – 5.33 | 0.83 – 3.50 | 0 – 1.17 | 0 – 0.33 | −3.1° … +7.4° |
| **THE REFERENCE, this trace** | **0.431° – 1.067°** | **4.00 – 8.89** | **2.45 – 7.89** | **2.45 – 5.45** | **0.00 – 3.33** | **4.9° – 7.5° over 45–65°** |

**Three readings, each now a comparison of like with like.**

1. **Today is 7 to 20 times short on height, and it holds none of the count at all.** Zero at every
   threshold the reference fills is not a small gap; it is the absence of the quantity.
2. **The spectrum alone closes a quarter of the height gap and none of the 0.25° count gap.** M1b's own
   sentence — *"necessary and not sufficient"* — is now a number: 0.15° against 0.82°, and 0 against 4.0.
3. **The ridged band closes the height gap and about a fifth of the count gap.** At its best station the
   ridged field's 0.689° is inside the reference's 0.43°–1.07°. Its 0–1.17 breaks per 60° at 0.25° is
   between a fifth and a half of the reference's 2.45–5.45. **What is missing is not amplitude. The
   reference's crests come in LINES, several per 60°, because a range and its drainage put them there.**
   That is 8c, and §6 says a further share belongs to 8e and 8f.

**One honest caveat on the skyline RANGE, and it argues against gating it.** The reference's 4.9°–7.5°
is measured over 45°–65° of bearing from a camera that looks down a valley from a pillar; most of the
range is the near ground falling away on the right, not relief on the far skyline. Our four stations
stand 373 m over the local surface and read a whole 360° horizon. The two ranges are not the same
quantity. **Report the range; do not gate on it.**

---

## 10. The recommended calibrated band for 8a's skyline gate

**P1 changes from a ratchet into a band with a floor, a target and a refusal.** A band, not a point,
because the reference is ONE hand-authored vista of ONE mountainous zone, and ruling T1 is explicit that
*"not the whole planet should look like on the picture"*. A floor the whole planet must clear would make
every desert and every plain a defect.

**Gate only at 0.25° and 0.50°.** §5 measures that the instrument's own noise on the reference is the
same size as the 0.05° threshold and inside the 0.10° one. A band calibrated inside its own noise is not
a band. The 0.05° and 0.10° counts stay printed reports.

**The proposal** (the owner rules; every number's source is given):

| Row | Floor — RED below this | Target — the band 8a aims at | Refusal — RED above this | Where the number comes from |
|---|---|---|---|---|
| Breaks/60° at **0.25°**, BEST of the four stations | **1.0** | **2.5 – 5.5** | **8.0** | the target is the reference's own measured band. The floor is 1.0 because M1b-2 already predicts 0–1.17 from the spectrum and the ridged band, so 1.0 is reachable by work 8a is already committed to and is not free. The refusal stands above the reference's own top: a skyline with more breaks than a hand-authored alpine vista is a saw, not a landform. |
| Breaks/60° at **0.25°**, MEDIAN of the four stations | **0.5** | 1.5 – 4.0 | 8.0 | half the best-station floor. A station rule that names the highest column in a 100 km patch draws an alpine patch and a rolling one alike. |
| Breaks/60° at **0.50°**, BEST station | **0.3** | **1.2 – 3.3** | 5.0 | the reference's measured band is 0.0–3.3 with a median of 2.0. The floor of 0.3 is M1b-2's own best (0.33 at S2-d): 8a may not lose what the arc already predicts. |
| **Largest rise** over the ±5° trend, BEST station | **0.25°** | **0.43° – 1.07°** | **2.0°** | the target is the reference's measured band. The floor is the ridged prediction's worst station (0.233°) rounded up. The refusal is about twice the reference's top: a rise over 2° from a 373 m eye is a wall, and `01` §1.4's slope work says our field cannot honestly make one. |
| Skyline elevation range | *report only* | — | — | §9: the reference's range and ours are not the same quantity. |
| Breaks/60° at 0.05° and 0.10° | *report only* | — | — | §5: inside the instrument's own noise on the reference. |

**Two reasons the band is conservative, and both push the floor DOWN, never up.**

- The picture's count is a **lower bound**: the rule's edge loss drops the rock-spire cluster (§3).
- Part of the reference's count is spires, a cliff and canopy — 8f, 8d and 8e, not 8a (§6). **8a alone
  should not be asked for the whole of it**, which is why the floor sits at 1.0 and not at 2.5.

**Why a floor is now honest where a ratchet was needed before.** `01` §6.1 refused a floor because the
band's only calibration was a number counted by eye off another game's screenshot — the very instrument
`01` §1.1 forbids. The band above is computed by the SAME rule, at the SAME thresholds, with its
sensitivity to every assumption printed. It can be wrong, and §5 says by how much. That is what a gate
needs.

**Keep the ratchet as well.** The band gives a floor and a refusal; the ratchet says the numbers may not
fall from the last recorded value at any of the four stations. Between the floor at 1.0 and the target
at 2.5 there is room for a regression no floor would catch. Both cost the same 1.5 s run.

**How the gate lands** is unchanged and still D2(a): P1 as an expected-red entry in `DEFERRED.md` with
8a named as its closing slice, so an unrelated change never turns `just gate` red.

---

## 11. What is still UNMEASURED

| | What is not known | How it would be measured |
|---|---|---|
| U16-a | **The reference's true field of view.** §1 bounds the photograph at `f < 0.75` of the screen, and §4 shows the answer moves little over 45°–65°. The game's own camera angle is still not known. | The owner names the game's field-of-view setting, or supplies a screenshot instead of a photograph. Cost: minutes, and it removes the widest assumption here. |
| U16-b | **How much of the reference's count comes from spires, a cliff and canopy** rather than from a height field (§6). | Re-trace the reference with the spires, the cliff step and the canopy masked by hand, and re-run the rule. It would give 8a a floor 8a alone can reach, and give 8d, 8e and 8f their own shares. Cost: one afternoon. |
| U16-c | **Whether one reference frame is enough.** This is one vista of one zone. | The owner supplies a second picture — a plain, a desert, a coast. Until then the median-station floor of 0.5 is the only guard against gating a whole planet to an alpine band. |
| U16-d | **What the picture holds beyond its own edges.** The rule drops 5° at each end, and the dropped band holds the rock spires (§3). | Nothing here can fix it; a wider reference frame would. It makes every count in this document a lower bound, which is the safe direction for a floor. |
| U16-e | **The reference's camera height.** ASSUMED at 300–800 m. It does not enter the break rule, but it enters any conversion of these angles into metres of relief. | It cannot be read off the picture. No metre figure in the arc may rest on it. |

---

## 12. What changes in the other documents

- `01_reference_target.md` §2.2 — the two ways out are no longer a choice. Way (1) is done, in the order
  the section itself recommended, and P1 may now carry a band.
- `01_reference_target.md` §6.1 — the "Proposed gate" row becomes §10's band plus the ratchet, and the
  "Reference picture, counted by eye — UNMARKED" row gains a MEASURED row beside it.
- `01_reference_target.md` §10 — U16 closes; U16-a to U16-e in §11 replace it.
- `09_step0_results.md` — M1b's table gains the reference's row (§9), and its verdict sentence "the
  spectrum alone gives about one break per 60° at 0.10°" gains its calibration: the reference gives
  2.45–7.89 at the same threshold.
- `08_pictures_acceptance.md` — the picture protocol should print the same four numbers under every
  frozen stand, so a picture and a gate say the same thing.
