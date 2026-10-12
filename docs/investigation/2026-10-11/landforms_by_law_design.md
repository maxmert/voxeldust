# Landforms by law below 8 km — the design (ruling B77)

Written 2026-10-11 for ruling B77 (`docs/design/owner_decisions_2026-09-22_believability.md`, B77 items 1–2).
**REVISION 2 (2026-10-11)** answers three adversarial reviews — A (the rules and rulings, A1–A22), B (the code, the
cost and the locality, B1–B19), C (the look against the references, C1–C16); §7 answers every finding — and ruling B78
(the rain's cure waits for the landforms' first pictures, so this design works on today's dry world). A design only: no
code, no build, no test run.

**Labels.** MEASURED = a number a run of this project or a cited source measured. STATED = a source or the code states
it. ESTIMATE = a number this design guesses, with its reason. INFERENCE = this design's own reasoning, to be benched.
UNMEASURED = nobody has measured it; §6 names the bench that measures it. ARITH = arithmetic on stated numbers.

**What the code is today (read 2026-10-11, the code is the truth).**
- The macro solve stores ten bytes per 8 192 m node (`crates/terrain/src/artifact.rs:115-135`): the height `Z`, the
  water level, the receiver slot with the facies bits, the discharge (a quarter-octave log of the budget's runoff,
  `solve.rs:718-724`), the temperature, the rain, the aridity and the rock province. The home planet holds 8 871 936
  nodes (`macro_lattice.rs:332`).
- A column reads `Z` by a 16-node Catmull-Rom read (`artifact.rs:1748-1770`), then adds the FINE OCTAVES of noise —
  ten octaves from 25 km down to 49 m, a ridged band from 12.5 km to 781 m — times a roughness factor, then the cap-rock
  bench (`recipe/src/plan.rs:191-215`, `recipe/src/height.rs:305-368`, `recipe/src/terrace.rs:246-262`).
- So under 8 km the ground is noise. It has no valleys that join, no cliffs and no floors. The owner saw this (B75).
- The shipped discharge byte is the budget's RUNOFF over the final tree, not the joint law's `Q = rain × area`
  (`solve.rs:718-724`, `artifact.rs:1484`). `Z` is whole metres (`artifact.rs:117`).
- The rows serve rungs 0–9. Rungs 10 and up read the pyramid's floored means, which hold `Z` and a water word only —
  no receiver, no discharge, no climate (`artifact.rs:732-817`).

---


## 0. For the owner

**What changed in revision 2.**
1. **The ladder carves into the ground that stands today.** Where the laws cut no valley, today's ground stays. So no
   picture gets smoother than today, and nothing retires before you look (C1). The moon, the desert world and the dry
   sea floor keep today's ground until their own families exist (A4, C10).
2. **A side stream runs DOWN the slope it falls on**, from its source to the larger stream. It bends where the slope
   turns and meets the larger stream at the angle the slopes give. No comb of straight, parallel side valleys (C2).
3. **No water word enters the ground.** The ladder reads the solve's river TREE and the climate. It never reads the
   sea's or a lake's side or level (A1).
4. **The far view is exact and stated.** The ladder draws rungs 0–9. A valley that a rung cannot draw enters as its
   exact cell mean. From rung 10 the far rungs draw what they draw today (A2, A6, B7).
5. **Tall narrow forms always keep their outline** (towers, fins, narrow buttes), and so do deep narrow slots (A3).
6. **Rock walls need a new hand-over rule between rungs** before they enter the game (B1, Q6). The bench is not blocked.
7. **The river byte that ships is the runoff, not the law's water.** The steepness needs the law's own water (B3, Q3).
8. **You look after the FIRST bench step**, and that step proves the method on Earth's real ground (A12, C12).
9. **The cost is higher than revision 1 said**: 2–15 µs a column on the CPU against 0.84 µs today (B5). The card and a
   per-node cache are part of the build, not options.

**What the pilot will see.**
- On today's dry, almost still world (B78), the laws favour the forms of dry country: cliffs and mesas where hard beds
  stand, talus aprons under the walls, fans where streams leave the hills, gorges where a river cuts hard rock. The
  valleys stand few and far apart on the dry belt (the law's honest answer, 2.6–12 km), and today's ground stays
  between them.
- Where the world is wet or active (1.5 % and 1.2 % of the land, MEASURED), the side valleys are dense and branch.
- When the rain's cure comes (B78), the same design draws the wetter world with no change to its code.

**How it works.** The planet's server solved the large rivers at 8 km. Under each node, both machines build THE
DRAINAGE LADDER from the seed and the node's row: smaller and smaller streams, each one running down the ground to a
larger one. The stream law gives each stream its height. The hillslope law and the cliff law give the slopes between
the streams. These cut into the ground that stands today; deposits (talus, fans) fill under walls and at slope breaks.

**Example.** A pilot flies low over a sandstone shelf of the home planet. A river from the solve runs in a canyon. A
side stream starts on the plateau, runs across the contours, turns down the canyon's side and joins the river. Where
the sandstone bed is hard, the canyon wall is a cliff; under it the soft shale lies at the angle of its fallen blocks,
and a fan spreads where the side stream reaches the canyon floor. Between two side canyons the hard bed stands as a
mesa; where three meet, a tower stands. She climbs to 30 km: the canyons and mesas stay, the tower keeps its outline.

**The order of work** (§5).
1. This revision and your answers.
2. **Bench step 1, 5–8 days: the method on Earth.** The same generator code on Earth's own 8 km heights and rain —
   Taiwan, Oregon, and a Colorado Plateau tile with its published rock layers — beside Earth's real ground, with the
   numbers. **Your look.**
3. **Bench step 2, 5–8 days: the home planet's dry country.** A sandstone shelf with beds, talus and fans; the belt; a
   plain; the wettest and the most active stands; the moon and the desert world unchanged as controls. **Your look.**
4. **Bench step 3, 1–2 weeks:** the glacial past, the tall forms' outlines, the beds' tilt, sunny and shady slopes.
5. **The build, 6–10 engineer-weeks** (ESTIMATE): integers, the card, the caches, the client; walls only after Q6.

**The questions** (each with its options, its price and the recommendation).
- **Q1. Earth's ground as a measuring stick.** (a) Yes: a bench program runs the same generator code on Earth's 8 km
  heights, Earth's rain and, for the Colorado Plateau, its published rock layers; the data stays outside the
  repository; nothing in the game reads it; no constant is set from it. (b) No. Price of (a): 1–2 days. **Recommended:
  (a).**
- **Q2. The far view past rung 9.** (a) The ladder draws rungs 0–9, where the rows reach today; rung 10 and farther draw
  today's far ground; the 9 → 10 swap drops the 4 km valleys within T7's bound (under one rung-10 cell) and the flight
  gate measures it; no new data. (b) First ship the rows out to rung 13's ring (the one-volume design's ask A2, accepted
  by B22, not built: about 14.9 MB at the worst eye height, about 7 s), so the ladder reaches rung 12. **Recommended:
  (a) now; (b) with the one-volume far fold.**
- **Q3. The river's water for the steepness** (an SL6 ask). (a) Ship the law's own water per node (rain × area over the
  solve's tree), one byte: 8.9 MB stored for the home planet, a tile +10 % (40 960 → 45 056 B), a re-solve. (b) Ship
  the steepness itself, one byte, the same size. (c) No new data: the ladder reads the runoff byte; on today's world
  its steepness then reads about 0.29 of the law's and its areas are 10–100 times wrong where a river leaves a wet
  range (ARITH, B3). **Recommended: (a)**; the bench measures (a), (b) and (c) against the solve it holds.
- **Q4. What retires.** (a) Nothing until a stand passes your look; then, only on ground the ladder reaches, the octaves
  above the hillslope length, the ridged band and the roughness factor; on the moon, the desert world and the dry sea
  floor nothing until their own families (craters, wind, sea floor) exist. (b) Revision 1's plan. **Recommended: (a).**
- **Q5. How tall narrow forms keep their outline** (not whether). Both machines list them per area from the same
  ladder — towers, fins, narrow buttes, and slots deeper than wide; each has its own coarser mesh once the cells cannot
  draw it, then a tint under one pixel. Price ESTIMATE 2–3 engineer-weeks and a silhouette gate (≤ 1 pixel).
  **Recommended: build it with the rock family.**
- **Q6. Walls between rungs.** Today a hand-over is measured straight up, so a 138 m wall moved sideways by one cell
  counts as a 138 m step and would push the finest rung's switch from 870 m to 60 km (ARITH, B1). (a) Measure the
  hand-over along the ground's normal and morph wall vertices along it: price 1–2 engineer-weeks in the client's ladder
  and morph, and the flight gate. (b) No slopes over 45° in the game (no cliffs). **Recommended: (a), decided before
  the rock family enters the build.**
- **Q7. The ground the ladder carves into, on land.** (a) The solved field alone (pure law; smoother than today between
  sparse valleys on the dry world). (b) Today's ground, centred so its fine relief has a zero mean. (c) The older
  landscape's relief of B60 carried under 8 km by its own spectrum (tens of metres). Price: all three on the bench, one
  day. **Recommended: show all three in bench step 2 and choose on the pictures; (b) needs no seam at the coast, (a)
  and (c) need a fade at the ladder's edge.**
- **Q8. A diagnostic picture**: the home planet's own heights under Earth's rain, labelled a diagnostic and never a
  candidate (B38 item 4), to separate the method from the world for B78's rain discussion. Price: half a day.
  **Recommended: yes.**
- **Q9. The rock beds.** Today they are 138 m shells at fixed radii, half of them hard by an ASSUMED share. (a) Each
  bed's thickness a named scatter by the published power law of bed thickness, the hard share by the sedimentary record
  (Blatt 1970), the beds still flat. (b) As (a), with the beds tilted on the belts by the solve's own convergence (an SL6
  ask of about 2 bytes per node: 17.7 MB for the home planet, a tile +20 %). (c) Keep today's beds. **Recommended: (a)
  in bench step 2; (b) after your look at (a).** Both change the chunks once at the build.
- **Q10. The river tree as an input.** The ladder reads each row's receiver slot, its "no receiver" included (the water
  ends there: the ocean, a closed lake, a hollow), and no sea or lake bit and no water level. **Recommended: confirm
  that this is not a water word in the solid pass.**

---

## 1. The references, split into landform families

All distances and sizes here are ESTIMATES read from the pictures (no picture states a scale). The pictures stay
outside the repository (`~/Downloads/voxeldust_references_2026-10-11/`, and the third reference of B23).

### 1.1 What each picture shows

- **Ref 1 — a forested valley with a lake under a jagged massif.** The eye stands 50–100 m over the valley. Near: a
  layered outcrop, beds 2–5 m thick, a cliff 20–40 m tall with a tread on top; a lake shore. Middle (0.5–5 km): a wide
  floor with low hills and forest in patches. Far (5–20 km): a massif 600–1 200 m high with rock towers 50–150 m tall
  and 20–50 m wide on its crest, rock ribs between forested gullies. Farthest: a serrated range in haze.
- **Ref 2 — a canyon at night.** Near: jointed walls 30–80 m tall; a stream 5–15 m wide with bars and pools; a flat
  green floor 50–150 m wide. Middle: the far side in tiers of 10–40 m with treads; the canyon bends. Far: forested hills,
  a peak 800–1 500 m over the floor, a spire on a ridge.
- **Ref 3 — a dry grass plain, red mesas, a snowy range.** Near: relief under 2 m, stones 0.1–0.5 m. Middle (1–5 km): a
  red sandstone massif 200–500 m tall (domes, buttes, fins, near-vertical faces, rounded tops) and a butte about 100 m
  tall; the plain meets the massif in a zone of debris. Far (15–40 km): a range 1.5–2.5 km high, sharp crests, snow in
  streaks in the gullies and on ledges.
- **Ref 4 — an aerial view.** The eye 1–2 km over a valley. Near: a forested flank with grey rock ribs; snow patches.
  Middle (5–20 km): valleys that branch and join; small lakes; fog; a red massif 500–1 000 m tall with a pinnacled
  crest. Far: a snowy massif with a horn.
- **The third reference** (status §9): forested hills a few hundred metres high; red-bed badlands with mesas, buttes and
  pillars; a dark jagged ridge; a snowy range far away; a pond; pale boulders on grass.

**What they teach together.** One view holds three to five kinds of country, each tens of kilometres across. Each form
shows the process that made it. Every valley runs down into a larger one. No line stays straight for long, no pattern
repeats, and rock shows in four of the five pictures (C12).

### 1.2 The families

| Family | Seen in | Scales (ESTIMATE) | What makes it believable to the eye |
|---|---|---|---|
| **F1 Alpine ridges and peaks** | ref 3, 4, 2 far; third ref far | relief 1–2.5 km; ridges 1–5 km apart | steep rock slopes rough at every scale (ribs, chutes); crests that branch; peaks where crests meet; sharp crests and horns from a glacial past |
| **F2 Dissected hills, forested ranges** | ref 4, ref 1, ref 2 right, third ref | relief 100–800 m; valleys 0.3–3 km apart | valleys that start across the contours and curve into larger ones; valleys of many sizes; round crests on soil, sharp on rock |
| **F3 Gorges and canyons** | ref 2, ref 4 | walls 30–500 m; floors 10–200 m | a floor sized by the river; walls stepped where beds alternate; the course bends |
| **F4 Rock control** (cliffs, mesas, buttes, fins, towers) | ref 1, 3, 4, third ref | cliffs 10–300 m in tiers; mesas 0.2–5 km; towers 20–50 m wide | tops at bed heights; rims with bays and fins along joints; forms shrink toward the valleys: mesa → butte → tower |
| **F5 Valley floors, corridors, lakes** | ref 2, 1, 4, third ref | floors 50 m–2 km; lakes 0.3–3 km | flat floors that widen downstream; the river winds inside a straighter valley; lakes in valleys |
| **F6 Plains** | ref 3, third ref near | relief of metres over kilometres | gentle, smooth at the metre scale; stones and boulders (objects of the cover phase) |
| **F7 Snow and ice forms** | ref 3, 4 | snow lines 1.5–3 km over the valleys | snow on ledges and in gullies, rock on steep faces; cirques, troughs and horns where ice once stood |
| **F8 Dry-country deposits** (NEW, C9) | ref 3 middle | aprons 0.1–1 km; fans 0.5–10 km | talus aprons under every wall; fans and bajadas where streams leave the hills; pediments at range fronts |

**Example.** A pilot on the home planet's belt looks across a valley: F2 under her, F3 where the trunk enters hard rock,
F4 where a sandstone bed caps the hills, F8 where the side streams spread their fans on the floor, F1 on the skyline,
F7 on that range's ledges over the summer line.

### 1.3 Which layer makes which part

- **The macro solve (8 km) says WHERE**: the belts, the plains, the large rivers (the tree), the climate, the rock
  province. This design does not change the solve, except for the optional bytes of Q3 and Q9.
- **The one generator says HOW under 8 km, at each point**: the ladder (§3), the slopes, the walls, the floors, the
  deposits.
- **Later layers**: the materials and the snow paint (K12; today's `surface.rs`), the cover (K13), the water fill
  (B22 step 4), the VOLUME forms (undercuts, alcoves, arches).

---

## 2. The families: laws, inputs, the generator function, the far view, the cost, the checks

### 2.0 What every family shares: ONE height field

The families are not separate fields. They are choices the laws make inside ONE construction, at each stream point,
from the facts there:

```text
h(p) = max( D(p), min( S(p) + L(p), P(p) ) ) + m(p)

S(p)  THE START: the ground the ladder cuts into (Q7: Z alone; today's ground, centred; or Z + B60's relief)
L(p)  THE LIFT: the expected volume the ladder's carves remove there (§3.6), so every rung cell keeps the start's mean
P(p)  THE PROFILE ENVELOPE: the lowest, over the stream points q_i that the ladder names (§3.5), of b_i + H_i(d_i):
      b_i the bed of q_i (the stream law, §3.4), d_i = |p − q_i|, H_i the side profile the facts at q_i choose;
      a divide is rounded by the law's own curvature (S2), a confluence never is
D(p)  THE DEPOSITS: talus aprons under walls, fans at slope breaks (F8); a deposit only adds, so it is a maximum
m(p)  THE ROUGHNESS: zero on soil and on floors (Perron 2008); on bare rock a named scatter over the joint spacing (S6)
```

**The composition rule, stated once.** A carve is a minimum, a deposit is a maximum, the roughness comes last. A family
never adds a term of its own outside these three; it only chooses a side profile or a deposit. Where no stream reaches
(the moon, the dry sea floor, a body with no rain), `P` holds no candidate and `L` is zero, so `h` is the start
exactly: no body gets a worse picture (A4). No water word appears anywhere (A1).

**Why a lower envelope.** Each stream point carries its own side profile. Where two profiles meet, the lower one wins,
so a crest forms where the slopes meet, and a crest moves toward the higher stream by itself: no step at a divide. A
hard bed between two canyons stays as a mesa only where both canyons' profiles stand over its top. A minimum never adds,
so nothing is counted twice.

**The shared laws.**
- **S1 THE SOIL SLOPE** (Roering, Kirchner & Dietrich 1999, WRR 35:853, doi:10.1029/1998WR900090; Roering, Perron &
  Kirchner 2007, EPSL 264:245, doi:10.1016/j.epsl.2007.09.035). `q = K_h·S / (1 − (S/S_c)²)`; in a steady hillslope
  `q = (ρ_r/ρ_s)·E·x` at the distance `x` from the divide. At the stream (`x = L_H`, half the local spacing) the slope
  in its rationalised closed form — no `0/0` as `q → 0` (A8):
  `S_h = 2q / (K_h + √(K_h² + 4q²/S_c²))`, `q = (ρ_r/ρ_s)·E·L_H`.
- **S2 THE ROUNDED DIVIDE** (Roering 2007). The divide's curvature is `C = −(ρ_r/ρ_s)·E/K_h`; a quadratic smooth
  minimum of width `k` joining slopes `±S_h` makes a parabola of curvature `−2S_h²/k` (ARITH), so `k = 2S_h²/|C|`. As
  `E → 0`, `k → 2(ρ_r/ρ_s)·E·L_H²/K_h → 0` in metres (ARITH; A8's "k goes to infinity" does not hold). The rounding acts
  only at a DIVIDE: its weight is `(1 − u_a·u_b)/2`, `u` the unit directions from the two stream points to `p` — one at a
  divide (opposite directions), near zero at a confluence (the same direction). So no pit forms at a join (B6).
- **S3 THE ROCK SLOPE — ONE LAW WITH S4** (A21). The soil's depth follows the soil production law (Heimsath et al. 1997,
  Nature 388:358, doi:10.1038/41056): `h_s = h_0·ln(P_0/E)` while `E < P_0`, none past it. So the soil's cover share is
  `1 − E/P_0`, continuous, zero at `P_0` (ARITH). The side slope blends from S1's to the rock's by that share — no switch
  line across a valley (B8). The rock's slope is S4's angle for the slope's own height, with the ROCK MASS's strength
  (Schmidt & Montgomery 1995; Selby 1980, Z. Geomorph. 24:31) — ARITH at `c` 50 kPa, `φ` 35°: a 300 m slope stands at
  about 45°, so threshold slopes come out of the cliff law and not from a second number.
- **S4 THE CLIFF LAW** (Culmann 1866; Schmidt & Montgomery 1995, Science 270:617, doi:10.1126/science.270.5236.617):
  `H_c(β) = 4c·sin β·cos φ / (ρg[1 − cos(β − φ)])`. A massive hard bed (its own `c`, ARITH 1 MPa → 301 m vertical on the
  home planet) stands as a wall; a jointed rock mass (its mass `c`) lays back. A limit that IS a law (B23), one table of
  `H_c(β)` per rock and per body (A5), its inverse a table walk.
- **S5 THE SLOPE–AREA REGIMES** (C4, A8). Near a stream's head the bed follows the colluvial and debris-flow regime, whose
  slope stays near the threshold (Montgomery & Foufoula-Georgiou 1993, WRR 29:3925, doi:10.1029/93WR02463; Stock &
  Dietrich 2003, WRR 39:1089, doi:10.1029/2001WR001057); the fluvial law `S = k_s·Q^(−1/n)` takes over where it is
  gentler. So `S` never grows without limit at a small area. The finest level on a threshold slope is the debris-flow
  chute (C4).
- **S6 THE ROUGHNESS.** Soil-mantled ground is smooth under the hillslope length (Perron, Kirchner & Dietrich 2008, JGR
  113:F04003, doi:10.1029/2007JF000866), so `m` is zero on soil and on floors (B6). Bare rock is rough at every scale
  over its joint spacing (C4): a named scatter (T9) weighted by the rock's share, its amplitude read on Taiwan's
  threshold slopes by bench step 1 — a registered interim until a rock-roughness law with its own body is found.

**The interims** (A14). `K_h` 0.003 m²/yr and `S_c` 1.25 (Oregon), `ρ_r/ρ_s` (Roering 2007), `P_0` and `h_0` (Heimsath
1997), each rock's `c` and `φ` and the rock mass's (Selby 1993; Schmidt & Montgomery 1995), the width prefactors
(Leopold & Maddock 1953; Finnegan et al. 2005), Bull's fan constants (1964): each is an Earth calibration. Each is read
at first hand BEFORE bench step 1 and registered as an interim in the form of B41 item 2 when the build lands.

**Example.** A side stream runs over shale under a sandstone bed at the foot of the belt. Its facts there: `E` 0.05 of
`P_0`, spacing 600 m — soil nearly everywhere, a round divide. Two kilometres up, the stream cuts the hard bed: the
profile there is the bed's wall over a talus apron. The envelope joins the two continuously; the soil share fades as
the rate climbs, and no line crosses the valley.

### 2.1 F1 — Alpine ridges and peaks

- **Laws.** The ladder (§3) at a high steepness; S3/S4 rock-mass slopes; S5's debris-flow chutes as the finest level;
  S6's rock roughness (no "folded paper", C4); the glacial past for horns and cirques (F7).
- **Calibration bodies.** The threshold slopes: the northwestern Himalaya (Burbank et al. 1996, Nature 379:505,
  doi:10.1038/379505a0, modal slopes near 32°, STATED). The steepness and `n`: Bhutan and Nepal (the joint law,
  `solve.rs:105-113`).
- **Inputs.** `Z`, the tree, the law's water (Q3), the rain and temperature bytes, the province; the charter's `g`.
- **Generator.** The envelope with the rock profile; crests where profiles meet; peaks where several crests meet. No
  peak law of its own.
- **Far view.** Crest lines are site functions (§3.7); a crest the rung cannot draw enters by its cell mean (§3.6). The
  far skyline's teeth (C11) are read under that rule; if the reference stand loses them, rule S extends to crest
  outlines (Q5).
- **Checks.** The slope histogram's mode against ~32°; the skyline numbers of status §9 (breaks at 0.25°: the reference
  2.5–5.5, ours 0; the largest rise 0.43–1.07°, ours 0.096°); the prominence distribution against Earth's tiles
  (G-PROMINENCE; Kirmse & de Ferranti 2017, Progress in Physical Geography 41:788, doi:10.1177/0309133317738163).

### 2.2 F2 — Dissected hills and forested ranges

- **Laws.** The ladder down to the law's valley spacing (§3.3); each side stream routed DOWN the ground (§3.2), so it
  starts across the contours and curves into its parent (C2); S1 soil slopes, S2 round divides; each stream's area
  summed from its real tributaries (C3).
- **Calibration bodies.** Oregon (S1, S2; the bench's tile N43 W124); Perron et al. 2009's five sites for the spacing.
- **Generator.** The envelope with S1/S2 where the soil share is high, S3 where it falls.
- **Far view.** §3.6, rule M for every level a rung cannot draw.
- **Checks** (Earth MEASURED on the bench's tiles, status log 2026-10-11): straight valleys — Taiwan 2.8 / 6.1 / 2.8 /
  3.2 %, Oregon 3.2 / 1.1 / 2.0 / 4.4 % at 1 km / 512 / 256 / 128 m; the lean 22–23 %; the roughness (x / y / diagonal)
  at 256 m — Taiwan 0.78 / 0.89 / 1.09, Oregon 1.47 / 1.44 / 1.79; Hack's `n` (Hack 1957); the area distribution
  `P(A ≥ a) ∝ a^−0.43` (Rodríguez-Iturbe & Rinaldo 1997, *Fractal River Basins*, Cambridge UP); the junction angles'
  mean AND spread against the aridity (Seybold, Rothman & Kirchner 2017, GRL 44:2272, doi:10.1002/2016GL072089); the
  spread of tributary bearings along a trunk against Earth's (C2); G-CREASE (no step along any bin or cell edge, C5);
  G-PIT (no closed hollow the solve does not hold).

**Example.** On a wet hill the side valleys start every few hundred metres, run across the slope, turn and join the
larger valley. On the dry belt they lie kilometres apart, and today's ground stands between them.

### 2.3 F3 — Gorges and canyons

- **Laws.** The bedrock channel width `w ∝ Q^(3/8)·S^(−3/16)` (Finnegan et al. 2005, Geology 33:229,
  doi:10.1130/G21171.1); the cover of the bed by sediment as a continuous share (Sklar & Dietrich 2004, WRR 40:W06301,
  doi:10.1029/2003WR002496), so the floor widens from bedrock to alluvial with no line (B8); the walls by S4 and F4.
- **Generator.** `H(d) = H_wall(max(0, d − w/2))` with `w` blended by the cover share between Finnegan's width and F5's
  floor.
- **Far view: the SLOT is rule S's inverse** (A3). ARITH: a slot 50 m wide and 300 m deep, at rung 5's switch distance
  of 27.8 km, shows its far wall 10.8 mrad ≈ 9.4 pixels tall while it is 1.6 pixels wide; rule M would draw it at rung 7
  as a trough 300 × 50 / 128 = 117 m deep — a shape change. So a slot deeper than wide goes on the tall-form list (Q5):
  the cells draw it while it is two cells wide; past that, the cells hold the ground without it and its own cut mesh
  draws it; under one pixel it is a darker tint.
- **Checks.** Floor width against `Q`; no wall over `H_c`; the slot list's silhouette (≤ 1 pixel).

### 2.4 F4 — Rock control: cliffs, mesas, buttes, fins, towers

- **Laws.**
  1. THE BED STACK (Q9 (a), A13, C6): each bed's thickness a named scatter around the body's sediment by the published
     power law of bed thickness (Rothman, Grotzinger & Flemings 1994, J. Sed. Res. A64:59, doi:10.2110/jsr.64.59); the
     hard share by the sedimentary record (Blatt 1970), replacing the ASSUMED half (`terrace.rs:105-131`); each bed's
     hardness a continuous share (`terrace.rs:152`, B13), not a flag; the beds' tops as a per-body table of radii
     (prefix sums), found by a binary search (B10). Flat until Q9 (b).
  2. A bed's face: S4 with the bed's own strength, scaled by its hardness share.
  3. The soft bed under a cap: the slope of the cap's fallen blocks (Glade, Anderson & Tucker 2017, Geology 45:311,
     doi:10.1130/G38665.1; Ward, Berlin & Anderson 2011, ESPL 36:1023, doi:10.1002/esp.2129).
  4. The rim's bays: the finest level on a cap's rim is the sapping and rockfall alcove, spaced by the cap's thickness
     (Laity & Malin 1985, GSA Bull. 96:203; Howard, Kochel & Holt 1988, NASA SP-491) — not the soil's `λ` (C7).
  5. Fins along joints: two joint sets per region spaced about the bed's thickness (Narr & Suppe 1991, J. Struct. Geol.
     13:1037, doi:10.1016/0191-8141(91)90055-N); the joints' directions a named scatter per region (T9, an unrecoverable
     history); the rim's alcove streams march along the weaker joint direction (C7).
- **Generator: the run of the stack.** From a stream point with bed `b`, walk up the beds: each adds a horizontal run of
  `thickness / slope(bed)`; the side profile is the inverse of that run. With the envelope: tops at a bed height between
  canyons whose runs are short → a MESA; closer canyons → a BUTTE; three canyon heads → a TOWER; a cap cut along a
  joint set → FINS; runs past the half spacing → cones and badlands. Nobody places them.
- **Far view.** Mesas and buttes wider than tall: rule M. Towers, fins and narrow buttes taller than wide: rule S, the
  tall-form list (Q5): the remnants around the points farthest from every stream, at a cap's height, listed per area by
  both hosts from the same ladder. WALLS: the radial hand-over metric fails on them (B1) — they enter the build only
  after Q6.
- **Cost.** The bed walk is a binary search in the body's table plus the beds between the stream and the point: ESTIMATE
  0.1–0.4 µs per kept candidate.
- **Checks.** No face over its `H_c`; tops at bed heights; the rim's bays per km against the cap's thickness; the
  mesa / butte / tower / fin counts and ratios on the Colorado Plateau tile WITH its published stratigraphy as Earth's
  facts (Q1; A16 — the seed's own stack would test the geometry only).

### 2.5 F5 — Valley floors, river corridors, lakes

- **Laws.** The alluvial channel width `w = a·Q^0.5` (Leopold & Maddock 1953, USGS PP 252); the meander belt and its
  wavelength (Leopold & Wolman 1960, GSA Bull. 71:769: wavelength about 11 widths); the floor's width against the
  discharge as a reading (Clubb, Weir & Mudd 2022, ESurf 10:437, doi:10.5194/esurf-10-437-2022).
- **Generator.** The FLOOR follows the stream's smooth valley axis (the coarse path, §3.2); the CHANNEL meanders inside
  the floor at the meander law's wavelength; the floor's half-width is half the meander belt, blended with F3 by the
  cover share. So the walls do not copy the meanders (C13).
- **Lakes.** No lake rule: the ladder ends where the tree ends (a closed hollow or the ocean), and the hollow is what
  the solve and the carves leave. No lake extent and no level is read (A1). The water fill (B22 step 4) finds the basin.
- **Far view.** A floor two cells wide or more draws itself; a narrower one enters by its cell mean (§3.6).
- **Checks.** Floor width against `Q`; the floors' cross-slope; the solve's lake share read again on the drawn ground
  (A10).

### 2.6 F6 — Plains

- **Laws.** None of their own. On old, gentle ground the steepness is small, so the carves are shallow, and the start
  (Q7) shows. S6 keeps soil smooth. Stones and boulders are objects of the cover phase (K13); tors and inselbergs on
  crystalline basement (Linton 1955) a later item.
- **Checks.** The relief at 1 km on old plains against Earth's stable continents (UNMEASURED; the macro's own reading is
  131 m at 110 km, B60).

### 2.7 F7 — Snow and ice forms, and the glacial past

- **Snow is paint, not shape** (K12): today's first surface look paints snow over the summer's freezing line
  (`surface.rs`; MEASURED: 2.3 km over the sea at the equator, 13–18 km toward the poles). The ladder's slopes give the
  paint its places (gullies and ledges hold snow, steep faces shed it: Bernhardt & Schulz 2010; Winstral, Elder & Davis
  2002 — K12, owed).
- **THE GLACIAL PAST** (C8). The sharp crests, horns and cirques of refs 3 and 4 come from ice that stood in a colder
  epoch. Laws: cirques and U troughs (Harbor 1992, GSA Bull. 104:1364; MacGregor, Anderson & Waddington 2009,
  Geomorphology 103:189) above the coldest epoch's equilibrium line — about 1 km under today's on Earth (Porter 2001,
  QSR 20:1067, doi:10.1016/S0277-3791(00)00178-5); the peaks cut near that line, the glacial buzzsaw (Brozović, Burbank
  & Meigs 1997, Science 276:571, doi:10.1126/science.276.5312.571; Mitchell & Montgomery 2006, Quat. Res. 65:96,
  doi:10.1016/j.yqres.2005.08.018). The epoch's cooling: the solve paces its strath terraces by a glacial epoch
  (`solve.rs:142`); whether the client can read its cooling from the charter is UNMEASURED; if not, it is an SL6 ask
  at bench step 3.
- **Checks.** G-BUZZSAW (B3) as a reading; the troughs' cross-profile exponent as a reading; the snow's share on slopes
  over its holding angle.

### 2.8 F8 — Dry-country deposits: talus aprons, fans, bajadas, pediments (NEW, C9)

These suit today's dry world better than soil hills, and ref 3's plain meets its massif in such a zone.
- **Laws.** Talus under a retreating wall, its slope the debris' repose, its volume the wall's retreat (Ward, Berlin &
  Anderson 2011). Alluvial fans: the fan's area and slope against its basin's area (Bull 1964, USGS PP 352-E); a bajada
  is where fans meet (the maximum does it). Pediments: the range front's gentle rock surface (Strudley, Murray & Haff
  2006, Geology 34:805, doi:10.1130/G22482.1) — later.
- **Generator.** Each deposit is a surface `D` that only adds (the maximum of §2.0): an apron from each wall's foot; a
  fan cone from each stream's slope break, its apex at the break, its slope and its radius from Bull's laws on the
  stream's own area.
- **Far view.** Fans and aprons are wide and gentle: rule M by their cell means; their volume is part of `L` (§3.6).
- **Checks.** Fan area against basin area (Bull's exponent near 0.9 as a reading); apron slopes at repose; no fan on a
  stream with no slope break.

**Example.** At the foot of the red massif of ref 3 every gully that leaves the cliffs spreads a fan on the plain, the
fans meet in a bajada, and under each wall a grey apron of fallen blocks stands at 35°.

---

## 3. THE CRUX — the drainage ladder: a network built first, evaluated at each point

### 3.1 The idea, and where it comes from

- Under 8 km a real landscape is a TREE of streams; the hills are what stands between them. So the generator builds
  the tree first and the ground from it. Both hosts build the same tree at each point, with no global solve (SL10).
- **The model: Dendry** (Gaillard, Benes, Guérin, Galin, Rohmer & Cani 2019, I3D, doi:10.1145/3306131.3317020; code
  https://github.com/mgaillard/Noise, `NoiseLib/include/noise.h`, read 2026-10-11): levels that halve the cell; one
  jittered point per cell; level 1 joins its lowest neighbour by a control function (here the macro ground); finer
  points join the nearest coarser segment; Catmull-Rom refinement; bounded windows (9 × 9 points at level 1, 5 × 5
  after). Its terrain mode joins at `acos(S_main/S_trib)` (Howard 1971, WRR 7:863, doi:10.1029/WR007i004p00863) and
  uses hand-set slopes per level and hand-set displacements.
- **Taken:** levels, points, bounded windows, a tree any host evaluates locally. **Replaced** (B23, B38): the hand-set
  slopes by the stream law's analytic profile (§3.4); the straight joins and the displacement by a ROUTE down the
  ground (§3.2); the primitives' blend by the hillslope and cliff laws (§2.0).
- **Older sources:** Kelley, Malin & Nielson 1988 (SIGGRAPH, doi:10.1145/54852.378519); Prusinkiewicz & Hammel 1993
  (Graphics Interface); Génevaux et al. 2013 (ACM TOG 32(4):143, doi:10.1145/2461912.2461996). Cordonnier et al. 2016
  (doi:10.1111/cgf.12820) and Tzathas et al. 2024 (CGF 43(2), doi:10.1111/cgf.15033; a multigrid iteration over a
  drainage tree, STATED) solve over a region — a bake; this design uses the 1-D analytic solution ALONG each stream.

### 3.2 The levels, the points and the routes

- **Level grids per body** (B15). Level `j` lays cells of `2^(m_j)` rung-0 cells, `2^(m_j)` the power of two nearest
  `node/2^j`. On the home planet and the moon a node is 2¹³ cells (`macro_lattice.rs:331`), so the levels nest in the
  nodes; on a body whose node is not a power of two they do not, and nothing needs them to.
- **Points.** One source per cell, uniform over the cell. The hash folds the seed, the face and the level into its seed
  word and takes `(i, j, axis)` as `corner_hash`'s three coordinates (`recipe/src/rng.rs:37`, B12).
- **Level 0, the root.** One anchor per node, uniform over the node; the link runs to the receiver's anchor. "No
  receiver" ends the tree (the ocean, a closed lake, a hollow — Q10). A link whose receiver stands HIGHER is a spill out
  of a hollow (`lakes.rs:1569-1572`): it carries no bed, and the hollow's anchor is its low point (B9).
- **THE ROUTE** (C2). A level-`j` source marches DOWN the ground that the start and the coarser levels make (`h_<j`), by
  the steepest descent of that ground's analytic gradient, until it reaches a coarser stream's floor: that is its join.
  So a side stream starts across the contours, follows the slope, and turns into its parent; its bends are the ground's,
  not a displacement constant (A7). The step is a sampling rule proved by convergence (halved on the bench until the
  route moves under a tenth of the floor width), not a knob.
- **The junction angle emerges** from the descent on a valley side (Howard 1971's geometric model). Seybold, Rothman &
  Kirchner 2017 (about 45° in the driest places, about 72° in humid ones, with their measured spread; the aridity index
  is `P/PET`, the byte is a log of `PET/P` that saturates at 0.5 and 20 — the mapping is stated in the build, A22) is the
  READING. If the emerging angles miss it, a named scatter around the aridity's mean (T9) is benched next (C2).
- **The bound** (B2). A route ends within `M_j` steps, `M_j` the steps to travel four coarser cells: every point lies
  within `√2·s_(j−1)` of a coarser source (ARITH). A route that has not joined by then ends in a closed hollow of `h_<j`;
  G-ORPHAN counts those the solve does not hold.
- **The path's form** (B19). The route's points are joined by a Catmull-Rom spline and stored as chords whose sag
  stays under a tenth of the floor's width — a fixed site function at every rung, so no straight chord shows near the
  eye. The trunk is routed the same way between its two anchors. The join sits on the parent's own path.

### 3.3 How far down the ladder goes

- **The spacing.** `λ = c_λ·l_c`. `c_λ` is a property of the law: Perron et al. 2009 (Nature 460:502,
  doi:10.1038/nature08174) give `6.4–12.7` for `n` = 1 only. For `n` = 2.2 the bench solves the law's own
  non-dimensional form once, as Perron 2009 and Theodoratos et al. 2018 (ESurf 6:779, doi:10.5194/esurf-6-779-2018) did:
  a constant of the law, Earth only a reading (A7).
- **The length.** `l_c = (K_h^n·U^(1−n)/K′)^(1/(n+2m))` (Theodoratos 2018, eq. A8), `m` = 1 on `Q`, `n` = 2.2,
  `K′ = K_lp·R`, computed in base-2 logs (A8). `U = K_lp·k_s^n`, the steady rate of §3.4. MEASURED on today's belt:
  `l_c` 369–979 m, so `λ` 2.6–12 km (B77); on Earth's numbers (`U` 1 mm/yr, `R` 1 m/yr) `l_c` is about 40 m.
- **The heads** (A7). A level exists where `s_j ≥ λ`. Along each route a stream starts where its own area passes the
  head area of the slope–area law (S5); above that point the route carries no channel. So heads move smoothly with the
  facts, and no fade band is chosen.
- **The areas** (C3). Each stream's area is summed from its real tributaries inside the node's ladder: each route adds
  its source cell's area to every stream point downstream on its path. A trunk's area comes from Q3's byte.

### 3.4 The channel heights: the stream law's analytic profile

- **The law.** In a steady state the joint law `E = K_lp·Q·S^n` (`solve.rs:105-113`) gives `S = k_s·Q^(−1/n)` along a
  stream (Flint 1974, WRR 10:969; Whipple & Tucker 1999, JGR 104:17661; Wobus et al. 2006, GSA SP 398,
  doi:10.1130/2006.2398(04)). The height is its integral, the χ form (Perron & Royden 2013, ESPL 38:570,
  doi:10.1002/esp.3302): `b(x) = b(x_0) + k_s·∫ Q^(−1/n) dx`, in closed form for an area linear along a chord.
- **The steepness from the rows.** `k_s = S·Q^(1/n)` at each node. `S` is the 3 × 3 least-squares slope of `Z` along
  the link, not a difference of whole metres (B9). `Q` is the law's own water: Q3 (a). Without it, Q3 (c)'s error (B3).
  `log k_s` is read across nodes with the 16-node weights over the nodes that vote (`S > 0`), renormalised in the box's
  gather (A8).
- **The integers** (A5, B10). An integer `log₂`/`exp₂` with interpolated mantissa tables carries every power (`1/2.2`,
  `2.2`, `0.5455`, `1/(n+2m)`); the leading-zero count split into two 32-bit halves for the card.
- **THE DATUM: `Z` IS THE NODE'S MEAN** (A10, C16). The trunk's bed at node `n` stands `M_n` under `Z_n`, `M_n` the
  expected height of the drawn ground over its trunk (§3.6). Along the link `n → r` the bed is the χ shape between
  `Z_n − m` and `Z_r − m`, `m = min(M_n, M_r)`. It descends strictly because `Z_n > Z_r` on a draining link; at `r` the
  incoming end stands at or over `r`'s own bed — a down-step, a cascade, never a pit (A9).
- **The DRAWN bed descends** (A9). Along a cut stream `min(S + L, b) = b`; `m` is zero on floors (S6); the divide
  weight is near zero at a join (S2); a fan (F8) descends from its apex. So the drawn bed is `b` or a fan surface, both
  falling downstream. G-PIT measures it. The drawn rivers run `M` under `Z`, so the solve's lake share is read again on
  the drawn ground (A10).

### 3.5 The ground at a point: candidates by value, never by count

1. **The gather** (P6; B2). For each level the box gathers every source within `R_j` of any of its columns:
   `R_j = M_j·σ_j + ρ_j`, the route's bound plus the CULL RADIUS `ρ_j` — the farthest a level-`j` stream can still win
   at a column: `(the box's highest start + lift − the level's lowest bed) / the level's gentlest side slope`, all from
   the box's own bounds. Each source, route, bed and wall is a function of its own site, never of the box.
2. **The candidates** (B2, C5). A column keeps EVERY stream point whose lower bound `b_i + H_min(d_i)` stands under the
   current best. The answer then equals an unbounded search over all streams — a stated exactness bound, proved by a
   test against the unbounded search (A7, A19). The set depends on the column alone, so two boxes give an edge column
   one answer: no crack, no crease along a bin edge.
3. **One candidate per stream** (B4, B6): a tributary and its parent at their join, and duplicates from a corner walk,
   count once.
4. **The ground**: `h = max(D, min(S + L, P)) + m` (§2.0).

### 3.6 The mean at every rung: the telescoping carves (A6, B7)

- Let `h_≤j` be the ground with levels 0..`j` and `C_j = h_≤(j−1) − h_≤j ≥ 0` the CARVE of level `j`. Then, exactly,
  `h = h_≤0 − Σ_j C_j` (the sum telescopes). This is what makes "the mean of a minimum" well defined: each level's
  effect is its own carve, measured against the ground the coarser levels left.
- **Rung `L` draws** `h_≤k*(L) − Σ_(j>k*) ⟨C_j⟩_L`: the levels whose floors are at least two cells wide (P3's survival
  constant, MEASURED for the ladder's own profiles), and each finer level as its CELL MEAN `⟨C_j⟩_L`:
  - (i) few streams in the cell: the sum over the level's streams of the cross-section area times the length inside the
    cell, over the cell's area (closed form per chord);
  - (ii) many streams (the cell over four spacings): the level's expected carve density from the law (cross-section
    area × stream length per area). The two agree in expectation; the bench measures their join.
- **The lift** `L(p) = Σ_j E[C_j](p) − E[D](p)`: a smooth site function, so every rung's mean over a cell is the start's
  mean, which is `Z` (the start's fine relief has a zero mean, Q7).
- **The kinds** (A15): each profile kind (soil, rock, wall, talus, floor, fan) is carried as a SHARE by the same rule —
  the paint never reads a winner that flips at a swap.
- **The readings**: G-CLOSURE per rung cell (the drawn rung against the mean of the finer rung over it, A6); G-MEAN per
  node; both on every bench stand.
- **THE LAST LADDER RUNG IS 9** (A2, B7; Q2 (a)). Rungs 0–9 read the rows. At rung 9 (512 m cells) only levels 0–1
  are drawn on steep ground. From rung 10 the far rungs draw today's far ground (the pyramid's `Z`; later the one-volume
  far fold). At the 9 → 10 swap the remaining carves drop to their means: the step is a level-1 carve's departure from
  its mean, a few hundred metres on steep belts (ARITH), under one rung-10 cell (1 024 m), so T7 holds and the flight
  gate measures it. W17's MEASURED 1 442.5 m pyramid step is a separate registered defect with the one-volume cure.
  `L_ref` stays 10 (from rung 10 the ground is `Z` in expectation); B22 Q1 is unchanged. Q2 (b) moves the last rung
  to 12.

### 3.7 Why no grid directions, no straight gullies, no seams and no flicker

- **Grid directions.** Points are uniform over their cells; routes follow the ground. A jittered grid keeps a weak trace
  of its axes; G-LEAN reads it against Earth's 22–23 %.
- **Straight gullies.** Streams are a tree that converges; each route bends with the ground. G-STRAIGHT against Earth's
  1–6 % (the R2 bench's best was 17.9 %).
- **Seams between nodes.** `Z`, `log k_s` and `M` come by the one C1 read; routes cross node edges freely.
- **Cube faces and corners** (B4). The windows walk the lattice graph cell to cell through the seam table, never by
  `i ± k` offsets, so at a cube corner the three faces' cells are all there, with no missing quadrant and no duplicate.
  All geometry is in 3-D body-frame directions, as the noise is.
- **Flicker.** Every source, route, bed and wall is a site function. The rung only chooses which carves are drawn and
  which enter by their cell mean (§3.6) — P1's own exception for the prefilter's cell. Walls: Q6. Tall forms: Q5.

### 3.8 The cost and the reads (B4, B5)

- **Reach.** The route bounds stack over the levels: ESTIMATE ±6–13 nodes of rows (B4's ARITH: ±13), against ±2 today.
  At a tile's edge the chunks that wait for a second tile go from about 18 % to up to 65 % (B4 ARITH), so the client asks
  the tile ring one reach wider: ESTIMATE about twice today's tile bytes at low height.
- **Per column.** ESTIMATE 2–5 µs with about 8 candidates in all, 7–15 µs with about 8 per level, on the CPU (B5),
  against 0.84 µs MEASURED today: +8–60 ms a chunk with its halo, against today's 11–22 ms builds and B-BUILD's 6.5 ms.
  So THE CARD's column step and THE PER-NODE CACHE are required parts of the build.
- **Per node, cold.** About 1 364 sources over five levels (ARITH), each a route of tens of steps that reads the coarser
  ground: ESTIMATE 30–100 ms a node. A far rung builds only the levels it draws (rung 9: levels 0–1, about 5 sources a
  node). The sampled-mean form of revision 1 is dropped (B5: 0.5–4 ms a node).
- **All of it is timed on bench step 1** before the build is priced again.

---

## 4. Integration into the game

### 4.1 The one-volume model's solid pass

- The column's surface becomes `h(p)` of §2.0, in the column pass (`recipe/src/plan.rs:191-215`); the cell kernel
  (`recipe/src/cell.rs`) still turns it into the gap byte and the substance. On ladder ground F4 does the cap-rock
  bench's work, so the bench runs only where the ladder does not reach, and only until Q4 (B14).
- **No water word** (A1, one-volume §2.1): no facies bit, no lake extent, no water level and no coast word is read.
  The ladder reads the tree (the receiver slot, "no receiver" included — Q10), `Z`, and the climate bytes.
- **The VOLUME layer** comes after the height, removals only, inside the walls F4 names: undercuts, alcoves, arches
  (Bruthans et al. 2014, Nature Geoscience 7:597, doi:10.1038/ngeo2209). Not in the first build.
- **The trench** (A20, B21). A player digs a trench across a mesa: the cells become `Empty` by a one-hop diff (SL10).
  The ladder never reads the diff — it is the seed's static shape; the fill pass reads the cells; the far rungs read the
  edited cells by the one-volume rules (§3 of that design). A pilot 50 km up sees the trench because the far copy comes
  from the same cells, not because the ladder knows of it.
- **The frozen patch** (A17, F1). A built area keeps the generator it was built on, the ladder included: the patch
  stores the generator's word in place of an octave count; the blend band at its edge stands. F3's details stay for
  slice 9.

### 4.2 The extractor and the ladder of rungs

- The extractor is unchanged (surface nets on the gap byte).
- **Walls** (B1, B18). The gap is vertical, so a wall sits on its column's midline and its outline snaps to 0°/45° facets
  on each rung; at rung 0 a 138 m wall spans three radial chunks. With the radial hand-over metric, a wall moved one
  cell sideways counts as its whole height (B1). So walls wait for Q6: the hand-over measured along the ground's normal
  and wall vertices morphed along it. The facets are the ground step design's dual-contouring bench (M4-9).
- **The sink and the reach of the morph** (`chunks.rs:69-79`, `:1351`). The coarser rung's sink reads the body's step
  bound; on ladder ground that bound becomes the ladder's own — the largest difference between a rung's drawing and
  the next rung's (§3.6), measured along the normal under Q6 (a), P3's constant MEASURED on the bench.
- **The slope reference** (B17). W17's field step and the slope charter read the first fine octave's slope. While the
  octaves stand (Q4 (a)) nothing changes; where they retire, the field step reads the ladder's step bound and the
  slope share retires with the roughness factor.

### 4.3 Collision on the server

The shard builds its collision chunks with the same generator, the same gather and the same column pass (SL10). The
no-drift gate (the golden chunk pins and `just gpu-drift`) gains chunks on a belt, a canyon, a mesa and a fan.

### 4.4 The client's builders: the CPU and the card

- **The trait** (B16). `ZField` today answers `Z`, the water words, the province and the side. The ladder needs the
  receiver and the climate bytes, which `TileCache` already holds: a trait change, no new data.
- **The tables, once per body**: `H_c(β)` per rock and for the rock mass; the beds' tops as prefix sums of the
  thicknesses; the `log₂`/`exp₂` mantissas. Each is drawn through `Gf`'s lawful once-per-body door (`gf.rs:1-12`) and
  rounded into integers, so the card reads plain words (A5, B10).
- **The CPU** runs the gather and the per-node cache; **the card** runs the column step (a fixed loop over a bounded
  candidate list, masks, two-word products) — required, not optional (§3.8).
- **The fixtures** (HR5, A19): a window at a cube corner, a face edge, a node with no receiver, a spill link, zero rain,
  a level whose heads move, a full bed stack, a tie in the steepest descent (broken by the site order); every trait read
  in a generic function a branchless shim.

### 4.5 What retires, and the world change

| Today | After the ladder (Q4 (a)) | Why |
|---|---|---|
| The fine octaves 25 km – 49 m | Stay as the start (Q7 (b)) until your look; on ladder ground, retire above the hillslope length after it; on the moon, the desert world and the dry sea floor, stay until their own families exist | A4, C1, C10 |
| The ridged band, the roughness factor, the slope share | As the octaves | the ladder carries their meaning by law |
| The cap-rock bench | Stays off the ladder; on ladder ground F4 replaces it | B14 |
| `Z`, the tree, the rows, the bed stack, the provinces, the caves | Stay | the root and the inputs |

- **The chunks change once**: the generator's version, the golden heights (`crates/terrain/tests/golden_home_z.txt`)
  and the chunk pins are recorded again.
- **The artifact does not change for the ladder alone**: its key is `ARTIFACT_KEY_GENERATION` (`tag.rs:380`), which the
  generator's version does not fold (B11). A new byte (Q3, Q9 (b)) bumps `ARTIFACT_VERSION` and costs one first solve
  per body (MEASURED 2 245 s for the home planet on 14 cores, status log 2026-10-10).

### 4.6 The surface look

The first surface look (`surface.rs`) keeps working and finally gets steep ground to paint. Better, with no new data:
the column carries the profile KIND SHARES of §3.6 (soil, rock, wall, talus, floor, fan), so the paint shows bare rock
on walls and threshold slopes, grey talus under walls, soil on S1 slopes, gravel and alluvium on floors and fans — and a
share never flips at a swap (A15).

### 4.7 Data across a realm boundary (SL6)

| Item | Size | Why the client cannot compute it | Status |
|---|---|---|---|
| The law's water per node (Q3 (a)), or the steepness (Q3 (b)) | 1 B a node: 8.9 MB stored, tile +10 % | the upstream basin reaches far past the client's tiles; the shipped byte is the runoff (B3) | ASK, recommended |
| The beds' tilt from the belts (Q9 (b)) | about 2 B a node: 17.7 MB, tile +20 % | the solve's convergence field is thrown away | ASK, after bench step 2 |
| The rows to rung 13's ring (Q2 (b), one-volume A2) | about 14.9 MB at the worst eye, about 7 s | already accepted (B22 Q6), not built | later, with the far fold |
| The drawn band (A11) | none, if the per-body bound holds | the band is `[start's lowest − deepest carve, start's highest + largest lift]`, from the head's two ends (B55) and the laws' own bounds; both hosts compute it | if the bench finds it too loose: an ask for the drawn band's two ends in the head, 4 B |
| The long-term lowering per node (C1) | 1 B a node | the solve's history is thrown away | NAMED, not asked: B78 item 3 measures the uplift map first |
| The coldest epoch's cooling (F7) | a charter word, if the charter lacks it | UNMEASURED | NAMED, at bench step 3 |

### 4.8 The binding rules, checked

SL3 and B35 item 6: the planet's ladder is the planet's own look; the moon's is the moon's. SL5 and B23: one generator,
every body by its own facts; Earth's facts only as a measurement (Q1, A16). SL10: one crate, two hosts, the same bytes,
integers only. HR5: the kernels in Tier-A crates at 100 %, run only on the owner's word (B74). B16: §3.6–3.7, Q5, Q6.
B21: §4.1. B23 and B38: the caps list (§6.2). B36 Q16: every bench step uses the pinned stands, three bodies, two suns
and four views (A18).

---

## 5. The order of work

**Why this order** (C12, A12, B78). Rock is the main subject of four of the five references. Today's dry world suits
the rock family better than soil hills (B78 keeps that world for now). So the method is proved on Earth's real ground
first, where the owner can judge the METHOD apart from the WORLD; then the home planet shows its dry country; the owner
looks after each step.

### 5.1 Bench step 1 — the method on Earth (5–8 working days, ESTIMATE)

- **The ground.** Taiwan (`~/voxeldust_earth/srtm1/N23E120.hgt`) and Oregon (`N43W124.hgt`), both outside the
  repository already; a Colorado Plateau tile (N36 W113, the Grand Canyon region) fetched the same way — a public
  SRTM-derived tile, its licence checked, a request with no identity in it (B36 Q2, A16).
- **The facts.** `Z` = each 8 km node's MEAN height (the meaning the game's `Z` now has, A10, C16); the tree = the
  game's own routing on that `Z`; the law's water computed on that tree, so Q3 does not arise on Earth; Earth's rain
  from a public climatology kept outside the repository; for the Colorado tile, the published stratigraphy of the
  canyon (formations, thicknesses, rocks) as Earth's bed stack (A16).
- **The rules** (Q1, A16). The same generator code with no Earth branch; an example program only; nothing in the game
  reads it; no constant is set from Earth — Earth is a reading.
- **The families**: F1, F2, F3, F4 (Colorado), F5, F8; the starts Q7 (a) and (c) (Earth has no octaves of its own;
  the game's octaves on Earth's `Z` as (b) for comparison).
- **The pictures**: shaded relief at 30 m and 128 m over 30–150 km beside Earth's real shaded relief of the same square;
  PERSPECTIVE views at the references' eye heights (50–100 m as ref 1, 1–2 km as ref 4) painted by the profile kinds
  (rock, talus, soil, floor, fan) — grey relief hides the rock-against-soil contrast the owner judges (C12). Each is
  captioned "no far shadow, no cover, no water".
- **The readings**: G-STRAIGHT, G-LEAN, the roughness by axis, Hack's `n`, the area distribution, the junction angles'
  mean and spread against the aridity, the tributary bearings' spread, the valley spacing against `c_λ·l_c` (with
  `c_λ` solved for `n` = 2.2), G-PROMINENCE, G-PIT, G-ORPHAN, G-CREASE, G-CLOSURE, G-MEAN; on Colorado the cliff
  heights and the mesa / butte / tower / fin counts; the cost per column, per box and per node.
- **THE OWNER'S LOOK** (A12).

### 5.2 Bench step 2 — the home planet's dry country (5–8 working days, ESTIMATE)

- **The stands** (A18, C12): B36 Q16's pinned stands (three bodies, two suns, four views); a sandstone shelf chosen by
  rule (the shelf province with the most relief); the belt's tile; a plain; the most active 1.2 % and the wettest 1.5 %
  of the land, chosen by rule. The moon and the desert world are CONTROLS: their pictures must not change (A4).
- **What it shows**: the three starts (Q7); the beds by Q9 (a); F3, F4, F5, F8 on the dry world; walls drawn at one rung
  (the bench has no hand-over).
- **The diagnostic** (Q8): the home planet's own `Z` under Earth's rain, labelled a diagnostic.
- **The seams and swaps**: G-SEAM on a node edge, a cube face edge and a cube corner; the 9 → 10 swap on a far view;
  the tile-edge waiting share; the band's bound (A11).
- **THE OWNER'S LOOK** — and with it B78's rain discussion comes back.

### 5.3 Bench step 3 (1–2 weeks, ESTIMATE)

F7's glacial past and G-BUZZSAW; the tall-form list (Q5) and its silhouettes at the swaps; the beds' tilt (Q9 (b));
the sunny and shady slopes from the planet's own sun (C14; Poulos et al. 2012, GRL 39:L10401,
doi:10.1029/2012GL051283; B36 Q17 (a)); more rim bays and fins (C7) if step 2's pictures call for them.

### 5.4 The build (on the owner's word; 6–10 engineer-weeks, ESTIMATE, B5)

1. The integer kernels in `vd-recipe`: the chord distance, the χ integral, the `log₂`/`exp₂` tables, S1's root, the
   divide's smooth minimum, the bed walk (1–2 weeks).
2. The gather, the routes and the per-node cache in `vd-terrain`; the trait change (2–3 weeks).
3. The column pass with the telescoping carves, the kind shares and the last ladder rung (1 week).
4. The card's column step (1–2 weeks; rust-gpu is owed — until then the CPU path with the cache, its frame cost
   MEASURED before the owner's look).
5. The client: the wider tile ring, the paint by kind shares (1 week).
6. Q6's normal hand-over, before walls enter (1–2 weeks under (a)).
7. The world change: the chunks once; the artifact only with a Q3 or Q9 byte.
8. The visual gates often (B74): the flight over the swaps, the seams, a walk in a canyon.
9. The owner's look in the window; the heavy gates each on the owner's word.

### 5.5 Later

- The TRANSIENT profile: relict uplands over young gorges, a knickpoint climbing since the ground was last raised
  (Royden & Perron 2013, JGR 118:497, doi:10.1002/jgrf.20031; Tzathas et al. 2024), with the long-term lowering (§4.7).
- The VOLUME forms; pediments; tors and inselbergs.
- The sea floor (K16), the small craters (K9), the dunes (K5): when each exists, the octaves may retire on its ground
  (Q4).

---

## 6. Risks, the caps list, and the unmeasured items

The owner questions are on the owner's page (§0, Q1–Q10).

### 6.1 The risks, largest first

1. **THE DRY, STILL WORLD** (MEASURED, B77; kept by B78). The ladder draws the law's answer: few, wide valleys on the dry
   belt. The start (Q7) keeps today's ground between them, the dry-country family suits the world, and the diagnostic
   (Q8) shows the method apart from the world for B78's rain discussion.
2. **THE COST** (B5). 2–15 µs a column on the CPU against 0.84 µs; 30–100 ms a node cold. The card and the per-node
   cache are required; if the card's path is late, the CPU path may miss B-BUILD. Bench step 1 times it.
3. **THE STEADY STATE UNDER THE NODE** (INFERENCE, C1). The hillslopes read the steady rate from the steepness; a relict
   landscape needs the long-term lowering (named in §4.7) and the transient profile (§5.5).
4. **WALLS** (B1). No wall enters the game before Q6.
5. **THE MEAN** (A6, B7). The two cell-mean forms of §3.6 and their join; G-CLOSURE reads every rung.
6. **THE WIDER REACH** (B4). ±6–13 nodes of rows; more chunks wait at tile edges; about twice the tile bytes at low height.
7. **PITS** (A9, B6). The proof of §3.4 and S2's divide weight; G-PIT and G-ORPHAN read what remains.
8. **THE JITTERED GRID'S TRACE.** G-LEAN.
9. **THE BEDS** (A13, C6). Q9 (a) replaces the assumed share; the joints' directions are a named scatter (T9).
10. **THE CARD'S COMPILER.** New shapes may meet new faults (the ledger in `recipe/src/height.rs:195-203`); the
    self-check gate catches a drift.

### 6.2 The caps list (B23, A8): every bound on this design's path, and its law or its proof

| Bound | What stands there |
|---|---|
| The candidates a column keeps | the value cull of §3.5: exact, proved by a test against an unbounded search |
| The gather radius `R_j` | the route bound plus the cull radius, both from the box's own bounds (§3.5) |
| The route's length `M_j` | four coarser cells, from the ARITH that every point lies within `√2·s_(j−1)` of a coarser source; orphans counted |
| The route's step, the chords' sag | sampling rules proved by halving until the route moves under a tenth of the floor |
| The bed walk's loop | per body from the charter (the band over the thinnest bed), never a fixed 58 |
| The level count | per body from its node and the smallest `λ` its laws allow |
| `k_s` near a lake or a spill | logs; a spill link carries no bed; a node with `S ≤ 0` does not vote — no floor |
| S1's root as `q → 0` | the rationalised form (§2.0) |
| S2's width | `k → 0` as `E → 0` (ARITH); no ceiling |
| `l_c` | base-2 logs; no overflow |
| `S` at a small area | S5's slope–area regime: a limit that IS a law |
| A face's height, a talus slope | Culmann (S4) and the debris' repose: laws (B23) |
| The 0/1 field `W` and its overshoot | removed (A1, B7) |
| INHERITED, REGISTERED | the rain clamp `[1, 10 000]` mm/yr and the lift clamp (B78 item 2); the aridity byte's ends at 0.5 and 20 (A22); the discharge byte's top class. The ladder reads the macro as it is; these go with B78's cure. |
| `CORNER_BLEND` | the ladder walks the lattice graph and needs no fallback; the `Z` read keeps it (registered) |

### 6.3 Every unmeasured item, and the bench that measures it

| Item | Measured by |
|---|---|
| G-STRAIGHT, G-LEAN, roughness, Hack, area distribution, bearings, junction angles' mean and spread | step 1 (Earth), step 2 (home) |
| `c_λ` for `n` = 2.2 (the law's own non-dimensional form) | step 1 |
| The route's step, the chords' sag (convergence) | step 1 |
| The cell-mean forms (i) and (ii) and their join; G-CLOSURE; G-MEAN | steps 1 and 2 |
| G-PIT, G-ORPHAN, G-CREASE | steps 1 and 2 |
| The cost per column, per box, per node; the reach in nodes; the tile-edge waiting share | step 1, step 2 |
| The error of Q3 (c) against (a) and (b) on the solve the bench holds | step 2 |
| The 9 → 10 swap's step in pixels; the band's bound (A11) | step 2, then the flight gate |
| The cliff heights and the mesa / butte / tower / fin counts against the Colorado Plateau | step 1 |
| The rock roughness' named scatter on Taiwan's threshold slopes (S6) | step 1 |
| The towers' and slots' count per km² and their silhouettes at the swaps | steps 2 and 3 |
| The desert world's rain rows; the moon and the desert world unchanged (A4) | step 2 |
| The coldest epoch's cooling in the charter (F7) | step 3 |
| The solve's lake share on the drawn ground (A10) | step 2 |
| `K_h`, `S_c`, `ρ_r/ρ_s`, `P_0`, `h_0`, `c` and `φ` per rock and per rock mass, the width prefactors, Bull's constants | read at first hand before step 1 (A14) |

---

## 7. The review findings and their answers

| # | Finding (short) | Answer |
|---|---|---|
| A1 | `W` reads the water: a water word in the solid pass | FIXED: `W` removed; no facies, lake extent or level read; the domain is where the tree's streams carve (§2.0, §4.1); Q10 confirms the tree |
| A2 | The streams change at the 9 → 10 swap | FIXED: the last ladder rung is 9; the far rungs draw today's ground; the swap bounded by T7 and measured; `L_ref` 10 and B22 Q1 unchanged (§3.6); the rows to rung 13 as Q2 (b) |
| A3 | Rule S made conditional; fins, buttes, slots missing | FIXED: Q5 asks HOW; the list holds towers, fins, narrow buttes and slots; the slot's pixel ARITH (§2.3, §2.4) |
| A4 | Retirements make the moon, the desert world, the sea floor worse | FIXED: nothing retires before the owner's look, and never where no stream carves until those bodies' families exist (Q4, §4.5); the two bodies are controls in step 2 |
| A5 | Floats in the recipe path | FIXED: per-body integer tables through `Gf`'s lawful door; integer `log₂`/`exp₂` (§3.4, §4.4) |
| A6 | Rule M fails: the mean of a minimum | FIXED: the telescoping carves and their exact cell means; G-CLOSURE per rung (§3.6); option (b) of revision 1 removed |
| A7 | Constants that are knobs | FIXED: `c_λ` solved from the law; heads by the slope–area law, no fade band; bends from the route; `m` zero on soil; the junction from physics; every bound proved (§3.2, §3.3, §3.5, §6.2) |
| A8 | Hidden clamps | FIXED: the caps list (§6.2); PARTLY REFUTED: S2's width goes to zero, not infinity, as `E → 0` (ARITH, §2.0) |
| A9 | "No pit by construction" false | FIXED: `W` gone; the cascade rule at nodes; the divide weight; `m` zero on floors; the drawn bed proved to fall (§3.4) |
| A10 | `Z_n` has two meanings | FIXED: `Z` is the node's mean; the trunk runs `M` under it; the lake share read again (§3.4) |
| A11 | "Needed: nothing" false for the band | FIXED: a per-body bound from the head's two ends and the laws' bounds; a 4 B ask only if the bench finds it loose (§4.7) |
| A12 | The owner's look too late | FIXED: the owner looks after step 1 (§5) |
| A13 | The beds rest on an assumed share | OWNER QUESTION Q9: thickness scatter and Blatt's share in step 2 (§2.4) |
| A14 | Earth's constants on every body | FIXED: each read before step 1, each a registered interim (§2.0) |
| A15 | The profile kind flips at a swap | FIXED: kind shares carried by rule M (§3.6, §4.6) |
| A16 | Q1 lawful only under conditions | FIXED: the conditions stated in Q1 and §5.1; the Colorado stack is Earth's published one |
| A17 | The frozen patch loses its meaning | FIXED: a patch keeps the generator's word it was built on (§4.1) |
| A18 | B36 Q16's stands | FIXED: every bench step (§4.8, §5.2) |
| A19 | HR5 fixtures | FIXED: the fixture list (§4.4) |
| A20 | The trench walk-through | FIXED (§4.1) |
| A21 | S3's three sources; hard switches | FIXED: S3 is S4 at the slope's own height with the rock mass; the soil share blends (§2.0) |
| A22 | Seybold's index against the byte | FIXED: the mapping is stated; the byte's ends registered (§3.2, §6.2) |
| B1 | A wall breaks the step bound | OWNER QUESTION Q6: the normal metric before walls enter the build; the bench not blocked (§4.2) |
| B2 | The candidate set is incomplete | FIXED: the value cull and the gather radius; the set depends on the column alone (§3.5) |
| B3 | The river byte is runoff | OWNER QUESTION Q3 rewritten as SL6 options with sizes (§4.7) |
| B4 | The reach is ±13 nodes; the corners | FIXED: the reach stated, the graph walk at corners, 3-D geometry (§3.7, §3.8) |
| B5 | The cost is 2–10× the estimate | FIXED: the honest costs; the card and the cache required; timed in step 1 (§3.8) |
| B6 | The smooth minimum digs pits at joins | FIXED: the divide weight; one candidate per stream; `m` zero on floors (§2.0, §3.5) |
| B7 | The mean of a minimum; no ladder past rung 9; `W` overshoots | FIXED: as A2, A6; `W` removed |
| B8 | Profile switches make step lines | FIXED: the soil share by the soil production law; the cover share for floors (§2.0, §2.3) |
| B9 | `k_s` noisy or undefined | FIXED: the 3 × 3 slope, logs, non-voting nodes, spill links (§3.4) |
| B10 | The integer tools | FIXED: named (§3.4, §4.4) |
| B11 | The artifact key | FIXED: `ARTIFACT_KEY_GENERATION`; no re-solve for the ladder alone (§4.5) |
| B12 | `corner_hash` takes three coordinates | FIXED (§3.2) |
| B13 | Hardness is a share | FIXED (§2.4) |
| B14 | "Then the bench" kept and retired | FIXED (§4.1, §4.5) |
| B15 | Sub-lattices only for powers of two | FIXED: levels need no nesting (§3.2) |
| B16 | `ZField` lacks the bytes | FIXED: a trait change, no new data (§4.4) |
| B17 | `slope_ref` disappears | FIXED (§4.2) |
| B18 | The vertical gap snaps walls to facets | Q6 and the dual-contouring bench (§4.2) |
| B19 | The bends' depth | FIXED: chords by the floor's sag, fixed at every rung (§3.2) |
| C1 | Smoother than today on the dry world | FIXED: the carve into today's ground; nothing retires before the look (Q4, Q7); the long-term lowering named (§4.7) |
| C2 | The comb | FIXED: routes down the ground; the junction emerges; Seybold's mean and spread the reading (§3.2) |
| C3 | Too regular | FIXED: areas from real tributaries; G-PROMINENCE and the area distribution (§2.1, §2.2, §3.3) |
| C4 | Folded paper | FIXED: rock roughness, debris-flow chutes, the slope–area regimes (§2.0 S5, S6) |
| C5 | The candidate cut makes creases | FIXED: as B2; G-CREASE (§3.5) |
| C6 | The beds cannot make the references' rock | Q9: thickness scatter now, tilt as an SL6 ask after the look; strike-following streams come with the tilt (§2.4) |
| C7 | Mesa outlines look artificial | FIXED: rim bays by the cap's thickness, fins along joints, talus retreat (§2.4) |
| C8 | No glacial past | FIXED: F7's coldest-epoch cirques and the buzzsaw, step 3 (§2.7) |
| C9 | The dry world's forms missing | FIXED: F8, in step 2 (§2.8) |
| C10 | The sea floor and the moon go flat | FIXED: nothing changes where no stream carves (§2.0, Q4) |
| C11 | The far skyline loses its teeth | Measured at the reference stand; rule S for crests if they fail (§2.1, Q5) |
| C12 | The weakest picture first | FIXED: Earth first, then the dry country, kind paint, perspective views (§5) |
| C13 | The valley walls copy the meanders | FIXED: a smooth valley axis, the channel meanders inside (§2.5) |
| C14 | Both valley sides alike | Step 3: aspect from the planet's own sun (§5.3) |
| C15 | The coast loses its valleys | FIXED: `W` removed; the tree's links reach into the outlet node (§2.0, §3.2) |
| C16 | A datum error in the Earth comparison | FIXED: `Z` is the node's mean on Earth and on the home planet (§3.4, §5.1) |
