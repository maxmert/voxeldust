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
