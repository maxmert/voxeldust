# Slice 2 — The geometry seam

What is built, where, how it is tested, and what is measured. Topic 1 explained the design. This file
explains the implementation. Simplified Technical English, examples from the game, drawings where they
help.

---

## 1. What this slice lands, in one picture

```text
   a position in the realm's own frame            a cell address
   (the position lattice, 1/1024 m steps)          (body, face, rung, i, j, k)

        LatticePos  ---- addr_of ---->  CellAddr
                    <--- cell_center --

                           |
                    GridMapping (one value per realm)
                     /                    \
             Identity                    CubeSphere
      hull, station,                  planet, moon,
      small asteroid                  large round body
      shifts only                     face bend + one root
```

The slice lands one module in the core crate, one wiring line where a realm's app is built, and the
gates. Nothing else changes. No record, no store, no wire arm, no mesher.

---

## 2. The files

```text
  crates/core/src/grid/
    mod.rs        GridMapping, GridDomain, Dir6, Neighbor, the six operations
    addr.rs       CellAddr, Rung; the packing helpers (chunk key + cell index, for slice 3's record)
    identity.rs   the flat arm: half_cells, shifts, no bend
    shell.rs      the round arm: N, floor, ceiling, rungs; the ladder rule; the radial band
    bend.rs       the face bend W(a), its inverse (four Newton steps from a0 = t), the face basis table
    seam.rs       the twelve edge pairs, the eight corners, the apron slots; generated + pinned literal
    tests/        the gates of §6
  crates/sim/src/…/build_app   one line: the realm's geometry value picks the arm
```

The module depends on nothing but the core crate's own position lattice. Its arithmetic uses only
operations IEEE-754 fixes on every target: add, subtract, multiply, divide, square root, the
round-to-integral family, comparison, clamp, saturating casts and integer shifts. No transcendental
function, no fused multiply-add, no min or max on floats. The boundary to the position lattice goes
through the vector library's component-wise arithmetic, so the pinned version of that library joins
the world identity's dependency list at the generator topic.

---

## 3. The six operations

```text
  addr_of(pos, rung)       -> Some(CellAddr) | None         which cell holds this position, at this rung
  cell_center(addr)        -> LatticePos                    the centre of a cell
  cell_corners(addr)       -> [LatticePos; 8]               the eight corners (the mesher's input later)
  neighbor(addr, dir)      -> Same(addr) | AcrossSeam { addr, axis_swap } | Outside
  radial(addr)             -> DVec3                         the grid's outward direction (NOT gravity)
  domain()                 -> GridDomain                    the box or the shell band the grid covers
```

On the flat arm every operation is an integer shift. On the round arm `addr_of` runs the inverse bend
and one floor per axis; `cell_center` runs the forward bend and one normalisation.

**Example.** A pilot on Moon 7 aims at a boulder. The moon's shard calls `addr_of` on the pilot's own
position at rung 0 and gets one address. It ships that address back with the highlight. The client
draws the cell the server named, so the mined cell and the highlighted cell are the same by
construction.

---

## 4. The round arm: the face bend, forward and inverse

A face position `a` runs from -1 to +1 across a face. The bend maps it to a tangent value, and the
normalisation puts it on the sphere:

```text
   W(a) = k1*a + k2*a^3 + k3*a^5        k1 = pi/4, k2 = 0.15, k3 = 1 - k1 - k2
   dir  = normalize(1, W(a), W(b))     for face +X; the face basis table permutes the axes

   forward:  (face, i, j) --> a, b --> W --> normalize --> a direction on the sphere
   inverse:  a direction --> the face (largest axis) --> t = y/x, u = z/x --> a = W^-1(t) --> i = floor
```

The inverse has no closed form. Four Newton steps from the first guess `a0 = t`:

```text
   residual in cells at the largest legal planet (N = 2^26), measured in Python on 200,001 samples:

   steps:    1        2        3         4
   a0 = t    large    ~1e-3    7e-4     2e-16   <- the f64 floor: frozen
   a0 = t/k1 large    ~1e-1    0.348    2e-16   <- three steps would leave a third of a cell
```

Both the first guess and the step count are constants in the module and part of the world identity.
The slice re-measures the residual inside the crate in Rust; the Python figures are the requirement,
the crate's own number is the gate.

**Example.** A client with a different first guess could name the neighbouring cell for the same
boulder, and the version tag would not refuse it, because only the arithmetic inside one function
differs. That is why the guess and the count live inside the frozen identity.

---

## 5. The seam table and the apron

```text
   the twelve cube edges as 24 directed records, generated from the face basis table:

   face +X, its +u side (u = +Y)  <-->  face +Y, its +v side (v = +X)   the along axes agree
   face +X, its -v side (v = +Z)  <-->  face -Z, its +v side (v = +X)   the along axes agree
   ...                                                                  (24 records in all)

   With this basis NO record is reversed: every face's u and v are positive axes, so the
   along-edge axes of two partners always agree. A compile-time assertion keeps it so; a
   basis change that reversed a seam would re-pair every saved planet.

   an apron gather at a face-edge chunk:

        +Y face
        |  strip from +Y, axis swapped
   -----+---------------------------
        | c c c c c c c | <- the chunk on +X
   +X   | c c c c c c c |
        | c c c c c c c |
   -----+---------------------------
   corner slot: no fourth face exists, so the nearest in-face cell fills it
```

The table depends on the cube net only, not on the planet's size, not on the rung. So an exhaustive
test at 62 cells per edge proves the table for every body. The round trip is size-dependent and gets its
own gate at the largest legal size.

---

## 6. The ladder and the radial band, per body

```text
   R_seed  --> N_ideal = pi * R / 2  -->  snap N to a multiple of 2^(T-1)  -->  R = 2N/pi
   T = the rung at which a face is at most 64 x 64 chunks, from N

   the radial band:                          k counts whole metres from the floor
        ceiling = R + H_scale    ---------   k = D_crust + H_scale
                                   air
        surface = R              ---------   k = D_crust
                                   rock
        floor   = R - D_crust    ---------   k = 0
```

`N`, `T`, the floor and the ceiling are seed-derived per body and stored nowhere: the module derives
them at each use from the body's taxonomy. The crust depth and the scale height do not exist in the
code today. This slice plants one small derivation for them beside the ladder, marked provisional; slice
5 moves it into the generator crate's body definition, where it becomes part of the world identity. The
planet's drawn radius snaps by at most 0.01 percent when the ladder lands, which is a world-tag change
and is free before the first chunk is saved.

**Example.** The home planet's seed draws 6,371,000 m. The module snaps the cell count to a multiple of
4,096 and the planet states its look at 6,370,354 m. Thirteen rungs. A chunk at the top rung is 254 km
on a side.

---

## 7. The flat arm and a hull's box

```text
   a hull's slot, a box of half-extents (hx, hy, hz) in whole cells:

            +hz
             |     . . . . . .
             |     . . . . . .
     -hx ----+---- +hx     origin = the hull's own frame origin
             |     . . T . . .      T at k = -1: the address is signed
             |     . . . . . .
            -hz

   addr_of(pos)     = pos.cell >> 10                 (signed shift: the stern keeps its sign)
   cell_center(a)   = (a << 10) + 512 fine steps     (the middle of the metre)
   sub-site         = (pos.cell >> (10 - s)) & (2^s - 1)   (slice 3 reads it; the maths is planted here)
```

A built realm's bound is a box by your ruling. Today's fixtures still state some bounds as shells. The
derivation accepts both: a box gives its half-extents directly; a shell gives the inscribed cube. The
switch of the berth's stored shape to a box is a record change and belongs to slice 3, not here.

---

## 8. The gates

| Gate | What it proves | Size |
|---|---|---|
| G-MAPPING-TABLE | `neighbor` is its own inverse across every seam; a four-step lateral loop closes everywhere except the eight corners, where it closes in three; every cell has four lateral neighbours; the apron is watertight at 24 strips and 24 slots; the table is byte-identical at every rung | exhaustive at N = 62 |
| G-MAPPING-ROUNDTRIP | `addr_of(cell_center(x)) == x` for every sampled cell; `addr_of` is `None` outside the domain; the residual stays under a stated fraction of a cell | N = 2^26, sampled at the worst bend position, every rung, both arms |
| The constants' sum | `(k1 + k2) + k3 == 1.0` exactly, in the order the bend evaluates it | a compile-time assertion, run on both targets |
| The negative half | every cell of a hull's box on both sides of the origin on all three axes round-trips | exhaustive over a fixture box |
| The sub-lattice identity | the shift form of a sub-site equals the index-space form, negative indices included | a property test |
| The no-drift digest | a golden digest of `cell_center` and `addr_of` over a fixed sample of addresses on every body kind, equal byte for byte between debug and release, and between this Mac and an x86-64 build | this Mac now; emulated x86-64 as a smoke test; a real x86-64 machine before the law is called satisfied |

**Example.** The four-step loop test walks east, north, west, south from a cell on the +X face next to
the +Y edge. Two of the four steps cross the seam with an axis swap. The walk must return to the cell it
started from. At a corner, three faces meet, so the loop closes in three steps.

---

## 9. The measurements

| # | Number | Pass |
|---|---|---|
| U-1 | the inverse bend's residual in cells at N = 2^26, in Rust | under 1e-6 cells |
| U-2 | the constants' sum on x86-64 and aarch64 | exactly 1.0 |
| U-3 | the cell-edge spread across a face under the chosen bend, measured by the crate | reported; expected 0.707 to 1.005 m |
| U-4 | the cost of one `addr_of` on a planet, and calls per tick with a hundred lookers | microseconds; constant, off the containment path |
| U-5 | the largest look-shell move over every body of the generated world when the ladder snaps | under one top-rung cell for every body |
| U-6 | the round trip over a hull's box on both sides of the origin | zero misses |

---

## 10. What is NOT in this slice

- No block record, no store, no wire arm, no mesher, no extractor.
- No corner landform: that is the generator's (slice 5). The corner refusal itself cannot happen yet,
  because nothing places blocks.
- No gravity. The grid's `radial` is the outward direction of the grid for terrain. Gravity is a
  function the realm states, and it lands with physics.
- No change to any realm's bound or berth on disk.

---

## 11. Laws, performance, seamlessness

- **SL1.** The address holds no parent frame. The module never reads a placement.
- **SL5.** One grid, one world. The 62-cell run proves the table, which does not depend on size; the
  production size runs the round trip.
- **SL9.** Every operation is constant time. Nothing scans a face.
- **SL10.** The module uses add, subtract, multiply, divide and square root only, so the same code runs
  under the generator's fence on the client.
- **HR3.** The one match on geometry matches a value the realm carries, never a shard kind. The stale
  doc comments that still say "FrameSpace" and "re-anchoring" are corrected in this slice.
- **HR4.** Two arms behind one seam, with the split gate you approved.
- **Performance.** An address on a planet costs about forty float operations. A hull's costs three
  shifts. Neither sits on the containment path.
- **Seamless.** The ladder snap happens before the first chunk. The terrain later is a function of the
  direction on the sphere, so the world has no line at a face edge; only the index space does, and the
  seam table hides it.

---

## 12a. Results (landed 2026-09-07)

What landed, measured, after an Opus 5 refutation of twenty findings, every one answered:

- **The grid family comes from the BODY, not the profile.** The sim's derivation takes a typed
  extent: a round body's look radius, a lump's look radius, or a built realm's sold slot. The capability
  profile's per-kind geometry is not read. A look and a bound cannot be confused, because the type
  names them. No caller exists yet; the block store (slice 9) is the first. The round-versus-lump
  threshold is the body definition's (slice 5).
- **Refusals, never garbage.** The body's centre, a NaN or infinite position, a rung the body lacks, an
  index off the face or outside the band, a crust as deep as the radius, an absurd slot and a sub-metre
  scale above a byte all answer None. The partial top slice of a coarse rung is not a cell for any
  operation.
- **The ladder reads the rung count from the snapped count.** At an octave boundary the old code gave
  one rung too many. Pinned: a 40,427.67 m body has 63,488 cells and five rungs.
- **No seam is reversed with this basis**, asserted at compile time; the crossing rule keeps the
  general arm and is exercised with a synthetic record.

| # | Number | Measured |
|---|---|---|
| U-1 | the inverse bend's residual at N = 2^26 | under 1e-6 cells (the test's gate; the refuter's model: 7.45e-9) |
| U-2 | the constants' sum | exactly 1.0 in both orders, at compile time, on aarch64 and x86-64 |
| U-3 | the cell-edge spread | 0.707 to 1.005 m, centre exactly 1.000 m |
| U-4 | one address lookup, release | 19 ns on a 500 m asteroid, on Earth and on the largest body (spread 1.02); 4 ns on a hull; 100 lookers × 27 lookups = 51 µs of a 20 ms tick |
| U-5 | the ladder snap on the home system | 10 planets and 11 moons: worst 4,024 m (0.025 %), none over half a unit, none over a top-rung cell; one gas giant of 89,156 km above the address |
| U-6 | the flat arm's negative half | exhaustive over a slot box at rungs 0 to 4 |
| golden | the no-drift digest | equal in debug, release, and on x86-64 under emulation |
| tests | the grid module | 38 in the core crate, 4 in the sim; the sub-site sweep 3,600 checks |

## 12. How it is built

I write the module myself, step by step: the address and the flat arm first, then the bend with its
tests, then the seam table and its generator, then the ladder, then the wiring line and the digest.
After it is green, one Opus 5 agent attacks the bend arithmetic, the seam table and the tests as a law
and feasibility refuter, and I answer every finding before the slice is called done. No new design run
is needed: the design was refuted twice already, and what remains is measurement.

Estimated size: about 1,500 lines of code and 1,500 lines of tests. The whole slice is covered at
100 percent, because the core crate is Tier-A.
