# Slice 8c, stage C1 — THE BENCH: the solve's core, measured on THE world's bodies

**Status:** MEASURED 2026-09-19 on this Mac (Apple M4, release build, one thread, the machine quiet at a
load under 2). The design's §9 names this stage as measurement M-L1: the solve's core — the lattice,
the priority flood, the D8 receivers, the flats, the discharge, the stream-power sweep — timed per
phase and as a whole schedule, with its memory. Every number below is a measurement; the estimate it
replaces is quoted beside it.

**What was built.** `crates/terrain/src/macro_lattice.rs` (the divisor rule, the node addressing, the
eight neighbours through the seam table, the node area by midpoint quadrature, the chord) and
`crates/terrain/src/solve.rs` (the flood seeded by the sea or by the six lowest nodes, the receivers by
cross-multiplied integers with the smaller index on a tie, the flats by a level-by-level distance
field, the topological order by a stack, the discharge as exact `u64` sums, the implicit sweep with the
receiver's water level as its base, the schedule). The bench is `crates/bins/examples/macro_bench.rs`
(`cargo run --release -p vd-bins --example macro_bench`; `VD_BENCH_BODY=<seed>` for one body).

**Every explanation carries a game-word example, as the owner asked.**

---

## 1. The lattice on THE world's bodies (the design's §2, corrected)

The design's rows were computed on the OLD home planet's `N = 5 263 360`. THE world's home planet has
`N = 9 961 472 = 2¹⁹ · 19` since the extended ladder (generator version 3), and the divisor rule finds
1 216 — a node of EXACTLY 8 192 m.

| body | ladder radius | `N` | edge | node | nodes |
|---|---|---|---|---|---|
| the home moon (`Planet(2918819812335288845)`) | 354.6 km | 557 056 | 68 | 8 192 m | 27 744 |
| the rocky planet `Planet(6548333207712198820)` | 4 589 km | 7 208 960 | 880 | 8 192 m | 4 646 400 |
| the ocean planet `Planet(14964011474468519458)` | 4 840 km | 7 602 176 | 928 | 8 192 m | 5 167 104 |
| **THE HOME PLANET** | 6 342 km | 9 961 472 | **1 216** | **8 192 m** | **8 871 936** |
| the rocky planet `Planet(17782101822166660494)` | 7 393 km | 11 534 336 | 1 408 | 8 192 m | 11 894 784 (not run) |
| the four giants | 17 000–38 000 km | 26.7–59.8 M | 2 048 (the ceiling) | 13–29 km | 25 165 824 each (not run) |

The home system holds no 200 km moon and no 50 km body: the smallest round body the ladder accepts is
the home planet's own moon, now pinned in the generator (`vd_terrain::home::home_moon`,
cross-pinned by `crates/bins/tests/home_body_pin.rs`) as the solve's driver-test body. Its whole
schedule runs in 18 ms, so the crate's unit tests solve a REAL body of THE world (06 §3.3).

*In the game:* a pilot who looks 50 km down a valley on the home planet sees six nodes span the
view. The solve decides where the valley runs; the spectrum draws the spurs inside each node.

## 2. The time (M-L1)

One thread, release, each phase timed alone, then the whole standard schedule (forty sweeps, a
routing and an accumulation every ten) timed as one.

| body | nodes | new (directions, areas, surface) | one routing | one accumulation | one sweep | THE SCHEDULE |
|---|---|---|---|---|---|---|
| the home moon | 27 744 | 6 ms | 2 ms | 0 ms | 1 ms | **18 ms** |
| the rocky planet | 4 646 400 | 0.99 s | 0.48 s | 0.010 s | 0.076 s | **3.30 s** |
| the ocean planet | 5 167 104 | 1.10 s | 0.51 s | 0.015 s | 0.152 s | **4.04 s** |
| **THE HOME PLANET** | **8 871 936** | **1.89 s** | **0.93 s** | **0.030 s** | **0.247 s** | **7.18 s** |

Against the design's model (03 §10, on 2.46 M nodes): one flood 1.0–3.2 s ESTIMATED → **0.93 s
MEASURED on 3.6 times the nodes** (about 105 ns a node, where the model feared 1.3 µs); one sweep
0.15–0.25 s ESTIMATED → 0.25 s MEASURED on 3.6 times the nodes; the whole core 12–30 s ESTIMATED →
**7.2 s MEASURED**, on the larger lattice. The routing is the cost that matters: four of them are 3.7 s
of the 7.2; the forty sweeps are about 3 s; the construction 1.9 s.

*In the game:* a player warps toward a planet nobody has visited. The interest radius grows with the
closing speed (the movement ruling), so the planet's shard wakes minutes out; the core of its solve
would be done in seven seconds of one worker thread. What C3 adds on top (the climate every ten
passes, the rebound every five, the talus, the ice) is UNMEASURED and is the number ask 2 waits for.

## 3. The memory

| reading | value |
|---|---|
| the state, by its rows (`MacroSolve::bytes_per_node`) | **52 B a node**: the terrain, the flooded surface, the receiver, the chord (4 B each), the area and the discharge (8 B each), the rain and the order (4 B each), the direction cache (12 B) |
| the home planet's state | 461 MB |
| the routing's transients (the visited bits, the distance field, the donor counts, the heap, the frontiers) | about 16 B a node at the peak, freed inside the routing |
| the process's peak resident set over the rocky planet, the first large body solved (the cleanest reading; the peak never falls) | 299 MB over the process's start on 4 646 400 nodes: **67.5 B a node** |
| the process's peak resident set with the home planet solved ALONE | **506 MB over the process's start: 59.8 B a node** (§3.1) |

The design's model (03 §10) held 41 B a node during the solve on 2.46 M nodes, 130 MB. The build
holds 52 in the state plus about 16 transient, on 8.87 M nodes: about 600 MB for the home planet.
The direction cache (12 B) is the row the model did not have; it is what makes a routing cost 0.93 s
instead of the bend's cost times eight neighbours times four routings.

### 3.1 The home planet alone

`VD_BENCH_BODY=4030111653607004909`, the process fresh (load 1.89): after the construction, one
routing, one accumulation and one sweep the peak resident set stood **506 MB over the process's
start — 59.8 B a node** (629 MB in all, the forest and the binary being the rest); the whole schedule
that followed in the same process raised the peak to 723 MB (a second state built before the first's
pages were returned). The schedule: **7.30 s** (the sequence run read 7.18 s; the two runs agree to
two percent). So: the home planet's solve core needs about half a gigabyte of one worker thread for
seven seconds, and the state it keeps between passes is 461 MB. The artifact C4 keeps is the
design's 9 B a node — 80 MB on this lattice, not the 22 MB the design's stale count gave.

## 4. What the routing found — the SHAPE is not judged here

| body | outlets (under the sea) | lake nodes (raised by the flood) | flat nodes | undrained | cyclic |
|---|---|---|---|---|---|
| the home moon | 1 143 | 2 744 | 2 746 | 0 | 0 |
| the rocky planet | 2 616 712 | 3 964 | 3 966 | 0 | 0 |
| the ocean planet | 34 894 | 1 387 335 | 1 387 350 | 0 | 0 |
| THE HOME PLANET | 16 839 | **3 148 015** | 3 148 246 | **0** | **0** |

Gate G-DRAINAGE holds on every body and every routing: every node reaches an outlet, no cycle.

Two facts about the shape, MEASURED and NOT judged (the design's §12 and ruling T8: the pictures are
judged dry until C2 gives the ground its second hump):

1. **A third of the home planet is a lake.** The starting surface is the recipe's one-humped noise,
   so the priority flood finds closed basins under 3.15 M of 8.87 M nodes and the drawn sea seeds only
   16 839 outlets. This is the drowning of ruling T8 seen from the solve's side; C2's isostasy is the
   cure, and this row is the number it is measured against.
2. **The land is cut flat inside the schedule.** With no uplift term and the literature's middle
   erodibility (`K₀ = 2 × 10⁻⁶ /yr`) over the system's five-billion-year age, `c = Δt·K·√Q/L` is
   about 7 at a headwater node and hundreds on a trunk, so the implicit sweep reaches its base levels
   in a few passes: the LAST sweep lowers ZERO nodes on every body (the first sweep on the home planet
   cut 5.7 M nodes by up to 300 m). A landscape with no mountains rising erodes to its lakes and its
   sea in five billion years — which is the physics, and which is why C3 carries the uplift from L3
   and gate G-AGE calibrates `K₀` on the per-basin hypsometric integral. No number here is tuned.

*In the game:* today the solve would give the home planet a flat coast plain sixteen thousand
kilometres wide with three million lakes; nobody flies over that. C2 and C3 are what stand between
this bench and the first picture.

## 5. What C1 proves and what it does not

- PROVES: the core's cost on THE world's own bodies, on one thread, with its memory — the number the
  ship-or-derive decision (ask 2) and the wake-ahead lead read; the divisor lattice on the real `N`;
  the seam-crossing stencil (24 corner nodes with seven neighbours, every relation symmetric); the
  areas summing to the sphere within the quadrature's error; every node drained on every body.
- DOES NOT PROVE: any shape (§4); byte identity across chips (C4's G-DRIFT, when the artifact exists
  to digest); the cost of the climate, the rebound, the talus and the ice (C3); the chunk read's cost
  with the artifact (C4, G-COST).

## 6. The ledger

`docs/design/DEFERRED.md` D-TERRAIN-7 names every fallback of the core by name: the starting surface,
the uniform rain, the missing uplift and the uncalibrated erodibility, the direction cache's rounding,
the memory ceiling and the giants, the solve not yet off the tick nor stored, the ocean planet's
39 km relief.

## 7. Stage C2 — THE INITIAL LAND, measured (2026-09-19)

`crates/terrain/src/land.rs`: the plates by the plate law, the crust field with Airy isostasy and the
water load, the belts under the relief, the sea by an integer bisection over the nodes with the load
inside it, the hypsometry. The same bench, the same quiet window (load 1.99 at the start).

| body | plates | land, one thread | continental crust | sea over the mean | ocean share | G-LAND-HYPSOMETRY | the schedule from the land |
|---|---|---|---|---|---|---|---|
| the home moon | 1 (a stagnant lid) | 4 ms | 40.0 % | none | 0 % | TWO HUMPS at −5 423 m and +8 327 m, valley 0.03 — GREEN | 34 ms; no lake, 15 523 flat nodes, every node drains |
| the rocky planet (24 Earth oceans) | 6 | 0.69 s | 40.0 % | +84 km | 100 % | TWO HUMPS at −42.6 km and −33.1 km — GREEN | 2.6 s; every node an outlet |
| the ocean planet (356 Earth oceans) | 3 | 0.76 s | 40.0 % | +1 125 km | 100 % | TWO HUMPS — GREEN | 2.9 s; every node an outlet |
| **THE HOME PLANET** (2.03 Earth oceans) | **14** | **1.33 s** | **40.0 %** | **+3 676 m** | **89.5 %** | **TWO HUMPS at −4 688 m (the floors) and +1 812 m (the platforms), valley 0.08 — GREEN** | **6.5 s; 7.93 M outlets, 3 285 lake nodes, 755 k flat nodes, every node drains** |

**The second hump exists.** On the home planet the ocean floors stand 6 500 m under the continental
platforms once the water loads them — Earth's own 6 476 m, from the same two laws. The plates: 14
(Earth's 15 by the same law), with 7.3 % of the surface within 250 km of a convergent boundary, 6.1 %
of a divergent one, 1.6 % of a transform. The land costs 150 ns a node: 1.33 s on the home planet,
on top of the core's 7 s. Within 250 km of a boundary the belts rise to +5 183 m over the mean, the
trenches fall to −6 188 m.

**★ THE FINDING THE OWNER MUST SEE: the home planet is 10.5 % land.** The census's water law (8b
stage 5) gives the home planet 2 735 928 089 km³ of water — 2.03 Earth oceans (the census's own
column `earth_oceans`). Over an Earth-like hypsometry that much water stands 3 676 m over the mean,
1 864 m over the continental platforms, and only the belts break the surface: 89.5 % ocean, where
Earth is 71 % and the design's §7.1 promised "a water-rich earth-like planet gets 60–75 % ocean".
This is not the drowning of ruling T8 (that was one hump; this is two humps with too much water on
them), and it is not a defect of the isostasy: an Earth with twice its water would look like this.
The two other planets of the home system that hold water hold 24 and 356 Earth oceans and are
water worlds outright, which their classes say. So the number that decides whether the home planet
has continents is the WATER INVENTORY LAW's calibration (8b's formation-zone model), or the
continental share (Earth's 40 %, the one calibration with no law behind it), or both. Nothing here
is tuned to hide it; ask 8 in §8 puts it to the owner.

**The sweeps still cut the land flat** (the last sweep lowers zero nodes on every body): the C3
finding stands — no uplift, and the erodibility uncalibrated against the age.

## 8. The asks C2 adds

8. **The home planet's land share.** 10.5 % land at the census's 2.03 Earth oceans and Earth's 40 %
   continental crust. Options: (a) re-read the water law's calibration in 8b (the formation-zone
   retention gave the home planet twice Earth's water); (b) state the continental share as a law of
   the water inventory (a wetter body differentiates more crust — no published law); (c) accept a
   world of island continents. Recommend (a): the water inventory is a derived word with a model
   behind it, and the model is the place a factor of two hides.
9. **The crust's scaling by the relief law's cap** (ledger D-TERRAIN-7, C2): the investigation's `1/g`
   alone gives a 350 km moon a thousand kilometres of crust; the build bounds it by the shape arm.
   Recommend accepting it as the law's statement.

## 9. Stage C3 — THE SOLVE, measured (2026-09-19)

`crates/terrain/src/climate.rs` (the climate inside the schedule), `crates/terrain/src/craters.rs`
(the impact record), and in `solve.rs` the uplift in the sweep, the sediment budget, the flexural
rebound on the coarse lattice, the talus, the ice, the coast band, the envelope, the facies byte and
gate G-AGE's reading; `Gf::exp`, `Gf::ln` and `Gf::powf` as fenced polynomials (ask 6). The same
bench, the home planet alone, the machine quiet.

### 9.1 The cost — and ask 2's answer

| phase, the home planet (8 871 936 nodes) | one thread |
|---|---|
| the whole full solve (the land, the craters, 40 sweeps, 4 climates, 4 routings, 8 rebounds, 8 talus passes, the ice, the coast, the envelope, the gates' readings) | **57 s** |
| one climate recompute (four in the schedule) | 8.2 s |
| the craters applied (144 967 craters) | 11.1 s |
| the core (C1's routings and sweeps) and the land (C2) | about 9 s |
| one talus pass (eight) | 0.18 s |
| one rebound (eight) | 0.018 s |
| the ice | 0.011 s |
| the crater population's draw | 0.04 s |
| peak memory over the process's start | about 60 B a node, as C1 |

**Ask 2 is answered by the number: SHIP the artifact.** Fifty-seven seconds of one core is not "a
few seconds", and a client approaching a planet must not spend a minute of a core before it can draw
the ground. So the realm solves once, keeps the artifact, and ships it on the bulk lane (one SL6
row, the design's §8), the home planet's pinned in the build. The climate is the cost to cut first
if the solve must be faster: it is 930 ns a node, and its latitude-only rows (the temperature
contrast, the belt) are the same for every node on a latitude and can be tabulated once; the craters
are once per body and their 11 s are the breadth-first stamps of 145 000 craters.

### 9.2 The gates and the shape

- **Every routing drains, no cycle**, on every body and every pass (G-DRAINAGE).
- **THE TALUS DIVERGED, and was cured by a trace.** The first full run sent the talus's worst
  excess from 41 km to 143 000 km over its eight passes and the envelope then scaled the planet to
  nothing. The pass-by-pass trace (`VD_BENCH_TRACE=1`, §9.3) showed the forty sweeps, the uplift
  and the rebound holding the field inside ±9.3 km, and the talus alone running away: it moved half
  the excess SUMMED over eight neighbours, so a peak with many low neighbours dropped far under them
  and the pair grew. Bounded by half the LARGEST single excess (03 §4.8's proportional split kept),
  the excess halves every pass: 8.2 km, 3.1 km, 833 m, 417 m, 383 m, 192 m, 192 m, 96 m — the
  monotone fall 03 §14 M13 asks for.
- **THE ICE**: 678 694 nodes (7.6 % of the area) stand over the equilibrium line; the thickest ice
  5 435 m; the deepest trough cut 1 925 m.
- **THE COAST BAND**: 11.8 m on the home planet (Earth's 10 m storm wave under its denser air).
- **THE ENVELOPE**: the trenches' subsidence first carried the field to −9 333 m under an 8 276 m
  relief; the uplift is now clamped against the CRATERED, LOADED field (twice: in the land against
  the loaded height, in the driver against the crater-clipped one), and the floor holds at −8 276 m
  through all forty passes. The top reaches 8 993 m: the rebound lifting un-eroded ridges over their
  start, exactly the case 03 §4.13 built the uniform scale for; the scale of 0.92 is applied at the
  end and the field leaves at ±8 276 m.
- **THE SHAPE**: sea 83.4 %, lakes 2.3 %, coast 0.4 %, ice 7.6 % by area — 16.6 % land; 721 414
  basins carry a deposit; the land share rose from C2's 10.5 % because the belts now stand as uplift
  over the age.

### 9.3 ★ GATE G-AGE DOES NOT MEASURE THE ERODIBILITY — the finding

The median hypsometric integral over the 1 434 basins of a hundred land nodes or more reads **0.71**,
and the erodibility scan reads **0.73, 0.71, 0.73 and 0.71 at a tenth, a hundredth, a thousandth and
ten times** the stated `K0`. Four decades of erodibility move the integral by two hundredths. So the
integral in this form does not read the age of the rivers: on an 8 km lattice a basin of a hundred
nodes is a continental platform with a few incisions, whose mean stands near its maximum whatever
the trunk was cut to, and Strahler's 0.35–0.60 band was measured on kilometre-scale basins with
continuous topography. The gate stays as a reading; it is NOT the calibration of `K0`, which stays
the literature's middle (Stock & Montgomery 1999) and is ledgered as owed to a measurement that can
move: the design's convergence test M12 (the same age at twice the passes) and a relief-against-age
statistic over the belts, both C4's or later.

### 9.4 The trace (the instrument)

| step | z after it, the home planet |
|---|---|
| the land (isostasy, loaded) | −4 647 .. 4 395 m |
| the craters, clipped at the relief | −8 276 .. 8 276 m |
| pass 0: the first sweep after the first climate and routing | max cut 13 062 m (a coast next to a trench cut to its base in one pass: `c` is hundreds on a trunk) |
| passes 1–4 | max cut 32, 64, 96, 128 m (the uplift's step, cut back each pass) |
| the rebounds (level 4, 131 km nodes) | max lift 572, 313, 439, 251, 385, 214, 346, 198 m |
| after 40 passes | −8 276 .. 8 998 m (the clamp holding the floor; the rebound's lift on the top) |
| the talus, eight passes | worst excess 8 213 → 96 m |
| the ice | −8 276 .. 8 993 m; the envelope's scale of 0.92 follows |

## 10. Stage C4 — THE ARTIFACT, THE STORE, THE SHIP (built 2026-09-19)

What is MEASURED and what is not, stage by stage. The design's §14 says what was built; this section says what
the numbers are.

| item | measured | value |
|---|---|---|
| the home planet's artifact | yes (`just artifact-pin`) | 8 871 936 rows × 9 B + 6 pyramid levels = 85.8 MB; digest `[0x7d45_41f7_7236_b266, 0xe47e_822c_6e65_0d8d]`; 71 s of one core in release |
| the corner blend's step | yes (moon, G-MACRO-CORNER) | ~160 m over a metre inside the two-node clamp band; < 2 m outside it |
| the golden chunk tables under the recipe (`None`) | yes (`just terrain-pin`) | unchanged by C4a–c (the recipe did not move) |
| the world identity through the golden fields | yes (`golden_z_record`, the tag test, the artifact pin) | `HOME_IDENTITY_MEASURED` pinned; the recipe-only word differs |
| the ship's pace | stated, not measured on a link | 8 pyramid parts + 4 tiles a tick a session; a tile is up to 64 × 64 × 9 B = 36 KB, so 144 KB a tick, 2.9 MB/s at 20 Hz; the head and the six levels of the home planet are 2.96 M words = 5.9 MB, shipped in 184 parts (16 384 words each) over 23 ticks (1.2 s) |
| the client's per-chunk field pick | unit-tested (moon); re-ruled 2026-09-20 | rung ≥ 10 reads a pyramid level on the home planet (`level_for`): the FINEST level whose node is at least the cell — level 1 (16 km) for rungs 10–14, one level up per rung above; the first rule took the coarsest level with four nodes across the chunk (262 km nodes at rung 14), measured as blobs and a coast that moved at every rung swap; a rung-0 chunk waits for 1–2 tiles, 4 at a face edge |
| the recipe→artifact rebuild at login | UNMEASURED | the count is `ChunkCounters::artifact_rebuilds`; the visible step is judged on the first window flight |
| the far view's pop at the crossing into the planet's realm | UNMEASURED | the window-holder ship is owed; the step is bounded by the artifact's own relief against the recipe's coarse octaves |
| the tiles' arrival against the finest ring's horizon | UNMEASURED | `ChunkCounters::awaiting_artifact` on the first window flight |

The other targets' legs (the x86-64 leg, the k3d pod) of the artifact pin are UNMEASURED until the legs script runs
`just artifact-pin` there.

## 11. Stage C5 — THE SEA, measured (2026-09-20)

| item | measured | value |
|---|---|---|
| the home planet's sea, re-solved over the eroded field | yes (`golden_z_record`, `home_artifact_pin`) | **+4 455 m** over the ladder radius; the initial land's sea stood +3 676 m — erosion lowers the land and the basins fill, so the level that holds 2.03 Earth oceans rises |
| the ocean share (G-SEA, area-weighted) | yes | **89.95 %** (8 995 / 10 000); 10.05 % land — the low end of ask 8's 10.5–16.6 % |
| the artifact's digest with the sea | yes | `[0x071e_adfc_3a98_db8c, 0x35c0_949b_1c9e_f2d7]` (version 2) |
| the solve's wall time | yes | 68.9 s of one core in release (the re-solve adds one bisection, under a second) |
| the level's coincidence with Earth's 4 455 m dry step | NOTED, unexplained | two laws, one number; a halved inventory must move the level (owed at C6) |
| the water sheet's cost and look | UNMEASURED | judged at C6's stands; the ocean's own look is 8o's |

## 12. The far-view ship's first flight (2026-09-20) — the store's field cap

The first `dev-cluster up --demand` + window flight of C4c and C5 together. The flight's measurements:

| item | measured | value |
|---|---|---|
| the login, the sky, the hull at the berth | yes | as before: the whole sky in hand 20 s after the window opened; the berthed hull 40 m from the spawn |
| the far planets before any artifact | yes | 30 recipe chunks at rung 20 drawn from the spawn (the nearest planet 5.5e10 m away); every planet box stated `artifact: None` because no solve had landed |
| the moon's solve | yes | landed and armed its ship 98 ms after the boot (a 353 km body); its store rows fit |
| **the home planet's solve** | **yes — a DEFECT** | the shard panicked on the tick the solve landed: `a pyramid's bytes encode infallibly: FieldTooLarge { tag: 1 }` at `built_store.rs:240`. The six pyramid levels are 2.96 M heights = 5.9 MB in ONE store row, and a TLV field caps at 1 MiB (`MAX_FIELD_BYTES`). Every big planet's shard died the same way (17 solves started, 1 landed); the demand loop re-spawned each into another 70 s solve and the same panic — four shards at 100 % CPU, no far view. |
| the cure | built, unit-tested | the pyramid stored as an index row + one row per part of `PYRAMID_PART_WORDS` (16 384 heights, 32 KB — the wire's cut; the home planet's pyramid is 181 rows); `ARTIFACT_VERSION` 3; a 1.2 MB level round-trips through 37 rows; stale rows dropped on rewrite |
| the home artifact's digest, version 3 | yes (`just artifact-pin`, re-recorded) | `[0xc194_6467_da41_3505, 0x3232_0b11_b552_d9b6]` (85 760 604 bytes, 6 levels, 68.2 s; the rows did not move — only the version word in the digest's head did) |
| the far view's arrival, the sea sheet, the tiles under the boots | UNMEASURED | the second flight, after the cure |

### 12.1 The second flight (2026-09-20, after the cure)

| item | measured | value |
|---|---|---|
| the solves, five planets at once on the 14-core machine | yes | the home planet 78.3 s (sea +4 455 m, the pin's word); the moon 97 ms; the rocky water world 36.3 s; two more 43.9 s and 122.9 s; **0 panics**, no re-spawn |
| the far-view path from the spawn | yes | the client held **3 whole artifacts (12 pyramid levels)** within the beat after each landing, 0 tiles (no occupant on a planet), 0 refused; the drawn set names only the bodies in range, so the two farther landings stayed on their shards |
| **the water world's sea word** | **yes — a DEFECT** | the rocky water world armed its ship with `sea_m=Some(32767)`: `i16` metres saturate, the solve found +84 km (§7). Ledgered in DEFERRED's C5 block; the cure rides C6's version bump |
| the far view on the artifact, the sea sheet's look, the tiles under the boots | UNMEASURED | needs a flight to a planet (the owner's stick or an agent leg); the lane's `artifact_rebuilds` / `awaiting_artifact` are not in the dev state yet — owed with the gateway's hidden artifact counters (`admin.rs` drops them as `_`) |

### 12.2 The owner's pictures and the pilot-eye captures (2026-09-20, 11:25–13:38)

The instruments first (the stamp's lane refusals and the two artifact words, `DevArtifacts::held`, the gateway's
artifact counters and its three log lines, `client.sh --capture --pilot`, the `land_stand` finder), then the
measurements, then the cures, each measured again:

| item | measured | value |
|---|---|---|
| the shard's tick after its artifact landed | yes | 90–96 ms (home), 47 ms (the water world), 0.3 ms (the moon): the source hashed 85 MB per tick for its digest and re-encoded 181 parts per tick. CURED (cached once, parts on a want): **0.05 ms** |
| the realm the occupant stands in | yes | the gateway wanted every planet in the sky except the one under the pilot (the origin has no composed row); the globe underfoot took its shape only when a boarded hull made the planet an ancestor. CURED (`drawn_realms`): the head wanted, cached and served **within one second** of the landing; `artifact_expected == artifact_held` on the client; the solved globe from the spawn with nobody boarding |
| the level a far-view rung reads | yes | rung 14 read level 5 (262 km nodes), every rung another level. CURED: the finest level whose node ≥ the cell (rungs 10–14: level 1, 16 km); `GENERATOR_VERSION` 7, the golden fields hold the level the top rung (18) reads: level 5 |
| the lines on every chunk edge | yes | a one-pixel line of sea colour on every edge at 410 km and at 20 km (the halo read the recipe, the core the artifact). CURED (`ColumnRead`): the 20 km capture over the same land is a continuous surface |
| the capture client's view | yes | without `--capture-pilot` the capture frames the whole realm-box scene and draws no ground under the pilot; with it, the window's own view |
| **a moving eye in a HULL: the tiles for a viewer** (`flight_hull_leg.sh`, boarded by a slow strafe, half stick on a 50 m/s² hull, 40 s) | yes | BEFORE the one tile path: the pilot in the hull held only the 9 tiles received on foot at the berth. AFTER: 37 tiles, 0 missing and 0 revealed on the descent, 989 missing for one sample at 930 m/s into rung 1 (the builders' budget) |
| **a moving eye: the tiles under the occupant** (a 40 s descent from 20 km at 500–1 000 m/s, the stamp every 3 s) | yes | BEFORE: 1 tile received, 730 chunks missing under drawn ground standing still, refusals 2.2 M climbing 19 k/s, 598 revealed on the descent. AFTER (`tile_reach_m`): 9 tiles, 0 missing standing still, refusals frozen, 0 revealed; 201 missing for one sample in the last second at 1 km/s into rung 1 (the builders' budget) |

Lesson for the ledger: the store's field cap is a number the artifact pin never met, because the pin solves and
digests without a store; the moon's round-trip test used a body whose pyramid fits. A store test with a level
over the cap now exists. Every new row family must be tested at the LARGEST body's size, not the smallest's.

### 12.3 The coast flight: the span against the artifact (2026-09-20, evening)

The owner, on the level stand with the shipyard hull: *"the water shores' shapes are constantly changing
while I'm flying."* Flown by the code (`flight_coast_leg.sh`: the 50 m/s² hull, level at 12 km, a quarter
stick, 40 s to 420 m/s, two frames a second): the frames show the sea in CHUNK-SHAPED squares with straight
edges where the land is, and the squares differ from frame to frame; the probe frames show those squares as
BLACK — no terrain mesh under the sheet at all.

**The cause, measured** (`crates/bins/examples/span_miss.rs`, ±150 km around the coast stand, the 5 × 5
sample grid of every column): the ladder asked a column's chunk slices from the recipe's own relief
(`digest::surface_column` → `height::height`), while the chunk it built stood on the artifact's `Z` plus
the fine octaves. The two surfaces are 2.1 km apart on average, 5.9 km at worst.

| rung | columns | slices missing the ground somewhere | land columns with NO ground in any asked slice | column bound |
|---|---|---|---|---|
| 4 | 92 112 | 86.96 % | 5 840 | 346 m |
| 5 | 23 104 | 73.43 % | 1 031 | 566 m |
| 6 | 5 929 | 51.24 % | 218 | 811 m |
| 7 | 1 521 | 3.75 % | 0 | 970 m |
| 8–11 | 555 | 0 | 0 | 0.9–1.8 km (a slice is 16 km and up) |

An empty chunk counts as built and drawn, so no stamp counter ever saw it; the water sheet is drawn per
chunk at the sea radius whatever the slice, so the hole read as sea. That is the changing shore, the squares,
and part of "the sea appeared when I boarded" (a scene swap re-descends and asks other slices).

**The cure:** `digest::surface_column_field` — the span through the builder's own column read
(`ColumnRead`), the bound widened by the field's step between neighbouring samples; the ladder's
`column_span` reads it on the field the rung reads, provisional while the tiles are not here (the finest
whole level, else the recipe; read again each descent; the renderer re-descends when the artifact's epoch
changes while any span is provisional). **Measured after:** 0 field-span misses at every rung 4–11 over the
same 300 km (the table's recipe columns unchanged, the field column all zero). The moon's own artifact pins
it in a unit test at a rows rung and a level rung. The pilot-eye coast leg on the rebuilt client is the
judge for the picture.

**The second half (the same evening).** The rebuilt client still drew the same squares. Ablations that
changed nothing: the engine's frustum culling off (`VD_TERRAIN_NO_CULL=1`), the bounded ask off; the wanted
set replayed offline for the recorded eye (`examples/wanted_probe.rs`) asked the right slices for every
column (0 hole columns of 4 574). The lane's own stamp (`hole_columns`, `empty_keys @L<level>`,
`awaiting_keys` with the missing tiles, `margin_missing`, `stale_builds`) named it: the first descents ran
BEFORE the artifact's head arrived, the recipe span was returned as FINAL with no artifact and never read
again. Cured: a recipe span on a body that has a macro lattice is PROVISIONAL until the head lands; a chunk
that lands built on another artifact than the lane holds is refused (`built_on`). **Measured, two flights of
one binary on the same leg: 0 hole pixels inside the drawn ground from frame 15 on, against 42 000–76 000 a
frame before.**

| frame | hole pixels BEFORE (run 1789933423) | AFTER (run 1789935981) | AFTER (run 1789937031) |
|---|---|---|---|
| 0 | 9 782 | 42 | 52 |
| 20 | 53 784 | 0 | 0 |
| 40 | 76 714 | 0 | 0 |
| 60 | 42 204 | 0 | 0 |
| 75 | 31 045 | 0 | 0 |

Owed from the instrument: the far ring's tiles (about 23 rung-9 chunks past the horizon wait forever for a
tile whose centre stands past the reach; ledgered).

