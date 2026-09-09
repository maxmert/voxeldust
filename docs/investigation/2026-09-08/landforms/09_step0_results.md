# Step 0 — the three measurements before the arc (2026-09-09)

Ruling V13 (foundation first): nothing in the landform arc is built before U1, M1b and M8-L are in hand.
Each result below is MEASURED by a program in this repository, named beside it.

## M1b — the skyline prediction (`crates/bins/examples/skyline_march.rs`)

**What ran.** From the four stations `01` §1.4a names, an eye 373 m over its own surface marched 360
bearings out to 540 km (50 m steps to 5 km, 300 m after) through the crate's own height field, twice:
TODAY (the shipped octave table) and PROPOSED (`03` §4.2.3: the fine octaves 4–13 from the rational
bump `s(o) = 0.3998 / (1 + ((o − 7)·ln2)² / 1.4²)`, `a(o) = s(o)·λ(o)/2π`; the coarse octaves 0–3
unchanged; the roughness field at its maximum everywhere). The break rule is `01` §6.1: a local maximum
of the skyline over the median of its ±5° window. 1.0 s for both fields.

**The prediction's own check.** The computed proposed amplitudes reproduce `03`'s table to the decimal
(496.2 / 401.6 / 319.4 / 198.8 / 79.8 / 25.1 m; fine sum 1 532 m of 14 305 m; fine RMS slope 0.700 =
tan 35°). TODAY's largest rises (0.0572°, 0.0539°, 0.0636°, 0.0551°) reproduce `01`'s replayed
0.054°–0.067° at the same four stations, so the march is the same instrument.

| Station | Largest rise TODAY | Largest rise PROPOSED | Breaks per 60° at 0.05° / 0.10° / 0.25° / 0.50°, TODAY | PROPOSED |
|---|---|---|---|---|
| S2-a PosX | 0.057° | **0.168°** | 0.33 / 0 / 0 / 0 | **2.67 / 0.83** / 0 / 0 |
| S2-b PosY | 0.054° | **0.145°** | 0.17 / 0 / 0 / 0 | **3.67 / 1.83** / 0 / 0 |
| S2-c PosZ | 0.064° | **0.138°** | 0.33 / 0 / 0 / 0 | **3.83 / 1.50** / 0 / 0 |
| S2-d NegX | 0.055° | **0.138°** | 0.33 / 0 / 0 / 0 | **3.33 / 1.33** / 0 / 0 |

**What it says.**

1. The stop line is passed: every station's largest rise stands over 0.10° (the discussion document §7.4
   said "if it stays under 0.10°, the spectrum is not the cure and the arc stops here"). The arc goes on.
2. The linear estimate (0.22°–0.27°, `0.054 × 3.99`) was too high by a third: the maximum of a sum of
   bands does not scale with one band, as the document warned. The measured gain is 2.2–2.7×.
3. The spectrum ALONE gives about one break per 60° at 0.10° and none at 0.25°. It is necessary and not
   sufficient. The document's own sentence stands: what the eye sees is the spectrum AND the closed-form
   terms keyed to the solve's skeleton (ridged, flow-aligned crests; the terrace; the macro field `Z`).
   This march used the shipped SMOOTH noise with new amplitudes; the proposed middle band is RIDGED
   (`1 − |noise|`), which makes sharper crests and is not in this prediction. A second march with a ridged
   middle band is the next cheap measurement (one afternoon), owed before slice 8a's constants are fixed.
4. The skyline's whole range stays about 2.5° of elevation, because the coarse octaves (7.6 km at 400 km)
   are unchanged and the macro field `Z` that replaces them is not built yet.

**Example, in the game's words.** The pilot stands on station S2-b's hill, 4.6 km over the ladder radius.
Today she sees one smooth arc on the horizon. Under the spectrum she sees about ten small rises around the
whole horizon, each an eighth to a sixth of a moon's width, and none as tall as half a moon. The ridge that
fills a quarter of the sky in the reference picture needs the solve's range and the ridged crests.

**M1b-2, the ridged middle band (the same afternoon).** The march ran a third field: the proposed
amplitudes with octaves 5..=9 (12.5 km to 781 m) as RIDGED noise (`(1 − |n|)·2 − 1`, re-centred so the
amplitude means the same), the rest unchanged.

| Station | Largest rise, +RIDGED | Breaks per 60° at 0.05° / 0.10° / 0.25° / 0.50° | Skyline range |
|---|---|---|---|
| S2-a | **0.490°** | 4.50 / 3.17 / 0.83 / 0 | −0.86° .. +7.37° |
| S2-b | **0.353°** | 4.33 / 3.50 / 0.50 / 0 | −1.39° .. +4.12° |
| S2-c | **0.233°** | 3.50 / 0.83 / 0 / 0 | −3.11° .. −0.79° |
| S2-d | **0.689°** | 5.33 / 2.50 / 1.17 / 0.33 | −0.89° .. +5.10° |

The ridged band is the term that makes lines: the largest rise grows another 1.7–5× over the smooth
spectrum, to a quarter to a full moon's width, and the first breaks at 0.25° and 0.50° appear. Note the
skyline's TOP rises to +4°..+7° at three stations: ridged noise is one-sided (its mean is above zero),
so it lifts the ground by about the band's summed amplitude (~800 m) where the crests stand; the fold's
per-column roughness factor and the macro field decide WHERE that happens, which this march cannot
model. Both marches are the same 1.5 s program, so slice 8a's constants can be tried against them in
minutes.

## U1 — is the home planet the right body? (`crates/bins/examples/home_body_facts.rs`)

**No.** The voxel home planet is the FIRST planet of System 7 in the forest's order whose radius the
ladder accepts (`vd_bins::home_body`); the census's own row on it, read through the home system's
subtree (`body_facts_in_subtree`, seconds — the galaxy-wide read cost twenty minutes of one core
before it was stopped):

| Fact | Value |
|---|---|
| star | class G, 1.025 M☉, 1.104 L☉ |
| mass | 0.093 M⊕ (5.53 × 10²³ kg) |
| radius | 0.526 R⊕ (3 351 km) |
| density | 3 508 kg/m³ |
| surface gravity | 3.29 m/s² |
| insolation | 6.25 S⊕ |
| equilibrium temperature | 429 K |
| class | Rocky |
| atmosphere | NONE |

It fails five of the six earth-like clauses: the radius (needs 0.8–1.25 R⊕), the flux (0.53–1.10 S⊕),
the temperature (217–261 K), and the air. Only "yellow sun" and "rocky" hold. It is a hot, airless
Mercury-sized rock at 0.4 of Earth's distance from its star. **Every picture judged so far was taken on
a body that can never have rain, rivers or a blue sky.**

**And the home system holds NO earth-like body.** Its nine planets, in the forest's order, with the
ladder's verdict:

| Planet | look radius | class | mass | g | flux | T_eq | air | ladder |
|---|---|---|---|---|---|---|---|---|
| 7701581858760374086 (the voxel home) | 3 351 km | Rocky | 0.093 | 3.29 | 6.25 | 429 K | no | accepts, 12 rungs |
| 1164718096683563219 | 33 172 km | SubNeptune | 36.6 | 13.26 | 2.16 | 309 K | yes | accepts, 15 |
| 15792791038712096226 | 28 095 km | SubNeptune | 8.3 | 4.19 | 0.75 | 237 K | yes | accepts, 15 |
| 9946628211997337570 | 15 063 km | SubNeptune | 17.2 | 30.16 | 0.26 | 182 K | yes | accepts, 14 |
| 15864076763714344450 | 12 078 km | IceGiant | 3.3 | 9.07 | 0.09 | 140 K | yes | accepts, 14 |
| 11637559958268695646 | 89 156 km | GasGiant | 217.9 | 10.92 | 0.03 | 105 K | yes | REFUSES |
| 15698659692757365133 | 7 972 km | IceGiant | 0.26 | 1.65 | 0.011 | 82 K | yes | accepts, 13 |
| 6052346702578586179 | 3 758 km | Ocean | 0.071 | 2.00 | 0.004 | 63 K | yes | accepts, 12 |
| 6692336264940193456 | 10 522 km | IceGiant | 1.6 | 5.89 | 0.001 | 48 K | yes | accepts, 14 |

(mass in M⊕, g in m/s², flux in S⊕.) The census's earth-like body of "1.023 R⊕, 6 515 km, in a system
of nine planets" that chose seed 2298 (`census.rs` header) is NOT in System 7 today, or the taxonomy
moved after the choice. **The galaxy-wide sweep runs detached (`earth_like_galaxy`) to find where the
earth-like body of this world is, and whether the ladder accepts it**; its result is appended below.

**A cost found on the way.** The census's earth-like read looks up every body's parent and star with
a linear search over the whole forest (`earth_like_in_forest`, `census.rs`), so a galaxy-wide sweep is
quadratic in the forest: MEASURED, the sweep at the home seed passed 38 minutes of one core without
finishing (the subtree read of one system takes seconds). The seed-search tool pays this per seed. An
index over the forest (one map from realm to row) makes it linear; it is a tooling cost, not a game
one, and it is registered for the tool, not built here.

**The galaxy half, with the census indexed** (`forest_index`: every parent and star lookup a map
read; the sibling-gap scan too): the sweep at the home seed takes **92 s** and finds **402 earth-like
bodies**, every one at 0.748 S⊕ and 236.8 K (the orbital ladder quantises the flux), all accepted by
the ladder. Ranked by the smallest distance from Earth in radius, gravity, flux and temperature, and
among the ties the star closest to the Sun, the home is now `Planet(4030111653607004909)` in
`System(1469594322681260607)`: 6 370.7 km, 1.000 M⊕, 9.82 m/s², an N₂-like atmosphere with a 7 161 m
scale height, a G star of 0.953 M☉, nine planets and fifteen moons. The switch landed the same day
(ruling V13, L27 landing note).

**What this decided.** L27 was no longer "re-pick inside the home system": the home system has nothing
to pick. The choice is between (a) moving the dev cluster's home system to the system that holds the
earth-like body, (b) keeping System 7 and a non-earth-like voxel home, or (c) a seed re-search. The
owner's word is needed; the re-pin is free until slice 14 either way.

## M8-L — the light (`crates/bins/tests/terrain_pictures.rs`, `crates/client-render/src/terrain.rs`)

**What changed.** The star stands 15° over the horizon and 120° off the camera's nose (over the
shoulder); the sun casts a shadow through a four-cascade shadow map sized to the drawn patch; the fill
light is retired; one light, one shadow. Four flights were needed, and each found a cause:

1. **Flight 1: every horizon tilted 27° and the whole near patch stood in shadow.** A roll instrument
   went into the gate (the delivered facing against the radial: nose below level, up off the radial,
   roll). It read ZERO roll on the ground stand, so the roll was made after the reading.
2. **Flight 2 confirmed it and named the caster.** The roll matched, to the degree, the roll a camera
   with WORLD `+Y` as its up would show (26.2° predicted, 26.9° in the picture). The cause is in the
   SHARD: the look integration rebuilt the body's orientation from yaw and pitch about the frame's
   `+Y` on every input datagram (`dot.rs`, `orient_from_yaw_pitch`), so a stand born on a planet lost
   its up the moment the first zero-look datagram arrived. **This is ruling V11's own defect: up was
   the server's, and the server's up was world `+Y` everywhere.** The aloft instrument, read after
   the rebuild, showed it directly: up 29.7° off the radial, roll 25°.
3. **Flight 3 refuted shadow acne.** A one-metre depth bias and a ten-texel normal bias changed
   nothing: the near ground stayed at the ambient level, (28, 25, 21), and only the ground BEYOND the
   shadow's reach was lit. So a real caster shadowed the whole patch. The caster is the star system's
   own look shell: a translucent sphere drawn AROUND the star, which the sun's light crosses on its
   way to every planet in the system.
4. **Flight 4: both fixed.** The dot carries its UP (`Dot::up`, the up the stand was born with: the
   radial on a planet, the frame's `+Y` in open space), the look turns about it through the kinematics
   module's one up-relative frame (`orient_in_frame`, `yaw_pitch_in_frame`; for world up the result is
   bit-identical to before, tested), and every realm outline, marker sprite, dot and star cloud is
   marked a non-caster. All three stands now read roll 0.00° and the nose at its stated tilt; the
   ground is lit at (94, 86, 73) with a soft gradient toward the horizon; paint share 1.000 / 1.000 /
   0.998.

**What the re-lit pictures show.** Under a raking star with a cast shadow the ground reads as a plain
with a gentle undulation, and the hill picture shows one soft ridge line on the horizon. The answer to
"terrain, or light?" is: BOTH were wrong, and with the light right the terrain is still flat — which
is what M1b measured by number. The shadow itself is not visible as a shape, because on 1.4° ground
nothing stands high enough to cast one; that is the spectrum's job (slice 8a).

**What landed in the foundation from this measurement.** The server-side up (V11) is now real for a
stand: `Dot::up`, set at birth and carried across a crossing, replaces world `+Y`. The realm's gravity
function that STATES the up per tick is still `D-TERRAIN-4`'s; today the up is the stand's own. The
`vd-sim` and `vd-core` changes are Tier-A and carry their tests.

## M8-0 — the rung disagreement, measured (`crates/bins/examples/rung_disagreement.rs`)

Slice 8's own precondition ("runs BEFORE this slice builds the ladder"). Over 2 304 directions per
face on every face of the home planet, the vertical disagreement between rung `L` and rung `L + 1`,
and its size in pixels at the SWITCH DISTANCE — where one cell of the coarser rung is one pixel high
at the reference view (45° over 720 rows):

| L → L+1 | cell | switch distance | max | p99 | max in cells | max in pixels | the bound |
|---|---|---|---|---|---|---|---|
| 0 → 1 | 2 m | 1.8 km | 0.33 m | 0.25 m | 0.17 | 0.17 px | 0.40 m |
| 3 → 4 | 16 m | 14.7 km | 3.4 m | 2.4 m | 0.21 | 0.21 px | 3.9 m |
| 6 → 7 | 128 m | 117 km | 35 m | 24 m | 0.27 | 0.27 px | 38 m |
| 9 → 10 | 1 024 m | 939 km | 338 m | 230 m | 0.33 | 0.33 px | 366 m |
| 10 → 11 | 2 048 m | 1 877 km | 757 m | 492 m | 0.37 | 0.37 px | 781 m |

**Worst over the whole ladder: max 0.37 px, p99 0.24 px.** Slice 8's pass line (max ≤ 2 px, p99 ≤ 1 px)
holds today with a margin of five. The MEASURED disagreement sits at 0.17–0.37 cells; the BOUND (the
sum of the dropped amplitudes) sits at 0.2–0.4 cells and is loose by about 2.2×, because the dropped
octaves rarely all peak together.

**A correction to the discussion document.** Its §7.2 said "the ladder's own promise is broken today at
six of the home planet's twelve rungs — 32.8 m against 32 m at rung 6". That compared the BOUND with
HALF a cell, which is a stricter promise than the ladder makes, and it was not a measurement. Measured,
the disagreement is under 0.4 cells and under 0.4 pixels everywhere. The bound stays the ladder's
statable guarantee (the coarse rung can state it), and the crossfade band in slice 8 is sized from it;
the picture is not at risk from today's octaves. Under the new spectrum (slice 8a) the same bench
re-runs; `04` computed 1.96 cells at an alpine slope, which is where the band earns its width.

## Step 0's verdict

- M1b: the arc goes on. The spectrum alone gives one break per 60° at 0.10°; the ridged band gives
  the first breaks at 0.25° and 0.50°. Both marches run in 1.5 s and stand ready for slice 8a's constants.
- U1: the home planet is the wrong body, and the home system holds no right one. L27 goes back to the
  owner in a new form (§U1). The galaxy-wide answer is appended when the sweep ends.
- M8-L: the light was wrong (into the star, no shadow) AND the shape is flat. Fixing the light found a
  foundation defect (the server's up was world +Y) and a renderer defect (an outline cast a shadow);
  both are fixed and measured. The pictures are level and lit; the terrain is what M1b says it is.
- M8-0: slice 8's rung gate passes today at 0.37 px; the ladder may be built on the shipped octaves.

## The first pictures on the earth-like home (2026-09-09, after the switch)

The same three stands, flown on `Planet(4030111653607004909)` through the same gate: ground 510
chunks, hill 507, aloft 1 250; paint share 1.000 on all three; every stand at roll 0.00° with its
stated nose. The pictures (`docs/investigation/2026-09-07/pictures/`) show the same one-colour ground
under the raking star, with more relief than the old rock (the relief cap scales with the radius):
a rolling skyline with a few soft crests from the ground and from the hill, and a wide undulating
plain from 60 km up. Nothing else changed in the recipe, so this is the shipped octave table on a
body twice the size, and the spectrum (8a) is still the cure for lines.

