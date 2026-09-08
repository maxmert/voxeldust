# Slice 6 — THE EXTRACTOR: cells become a surface, inside the one generator

Written 2026-09-08 for the owner's discussion, before any code. ASD-STE100, in-game examples,
drawings where they help. Every recommendation in §9 waits for a YES or a NO.

Status of what this slice builds on: slice 5 (`fb47926`, `9f47af9`) gives every terrain cell a
substance and a GAP BYTE, byte-identical on five legs. Slice 6 turns those bytes into a surface: a list
of points and triangles a client can draw and a shard can collide. It lives INSIDE `vd-terrain`, so both
hosts run the same code (SL10 clause 2: a port is forbidden).

---

## 1. What this slice decides, and what it leaves to later slices

**Decides:** how a chunk's cells become triangles (the extractor), which diagonal splits a four-sided
face (the triangulation rule), how a chunk gets the cells it needs from beyond its own edge (the halo),
how the output is written so a digest can pin it (integer vertices), the order in which edits, blocks
and attachments are composed onto the generated shape before the extractor runs (the composition
order), and where a small block sits on a slope (the seating rule). All six are part of the WORLD
IDENTITY (Format D, ruling V6 Part D): both hosts must compute them identically, forever.

**Leaves:** how the client picks a rung per chunk and hides the joint between two rungs (slice 8, the
detail ladder); the collider itself (slice 11: it consumes this slice's triangles); the sub-metre block
record (slice 12: it consumes the seating rule); trees (slice 14, art assets per ruling V4); normals
and colours (the client's, see §6).

**Example.** A pilot lands on the home planet. Her client asks the generator for the chunk under the
hull at rung 0. Slice 5 answers with 238 328 cells. Slice 6 answers with about a few thousand points
and twice as many triangles: the hillside as a surface. The client draws that surface. The planet's
shard computes the same triangles and hands them to the physics engine (slice 11), so her boots stand
on exactly what she sees.

---

## 2. The words, explained once

| Industry term | What it means here |
|---|---|
| **Density field / signed distance field (SDF)** | A number at every point that says "how far inside or outside the ground am I". Ours is the GAP BYTE: negative inside rock, positive in air, zero exactly on the surface (slice 5, Format B). It is a RADIAL gap, not a true distance: it is measured along the line from the planet's centre. |
| **Isosurface** | The surface where the field equals one value. Ours is where the gap equals zero: the ground. |
| **Surface extraction / meshing** | Turning the field into triangles. The code that does it is THE EXTRACTOR (or the mesher). |
| **Mesh** | A list of VERTICES (points in space) and a list of TRIANGLES (three vertex indices each). That is what a graphics card draws and what a physics engine collides against. |
| **Marching cubes** | The classic extractor (1987). It looks at each cube of eight samples, reads the eight signs as a 256-case table, and emits up to five triangles per cube from the table. Correct, but many thin triangles and a big table to test. |
| **Dual contouring** | An extractor that puts ONE vertex inside each cube that the surface crosses and joins vertices across cube faces. "Dual" because the vertices sit in the cubes, not on the edges. The full method places the vertex by solving a small least-squares problem from surface NORMALS ("Hermite data", a "QEF" solve). It keeps sharp creases. The solve is where a platform maths call creeps in. |
| **Surface nets (naive)** | Dual contouring's simple sibling: one vertex per crossed cube, placed at the AVERAGE of the points where the surface crosses the cube's edges. No table, no solve. Creases get rounded by up to half a cell. This is the recommendation (register row V9 of the proposal). |
| **Edge crossing** | On an edge between two samples of opposite sign, the point where the field reaches zero, found by one division: `t = d0 / (d0 − d1)`. |
| **Quad** | A four-sided face. Surface nets emit quads (one per crossed edge); a graphics card wants triangles, so every quad is cut along one DIAGONAL into two triangles. |
| **Triangulation rule** | The rule that says WHICH diagonal. On a saddle-shaped quad the two choices differ by up to half a cell in the middle. |
| **Halo / apron / ghost cells** | The one-cell-thick layer of neighbour cells a chunk needs beyond its own edge, because the extractor looks at groups of 2×2×2 cells that straddle the edge. |
| **Crack / T-junction** | A hole between two neighbouring meshes whose vertices do not match along the shared edge. Between two chunks at the SAME rung, surface nets give no crack when both use the same halo. Between two RUNGS there is a crack; slice 8 hides it with a skirt. |
| **Skirt** | A short curtain of triangles hanging down from a chunk's edge so a crack against a coarser neighbour is not seen through. Slice 8. |
| **Watertight / manifold** | A mesh with no holes and no edge shared by more than two triangles. A collider wants this; surface nets over a closed field give it inside a chunk. |
| **Quantisation** | Rounding a number to a fixed grid of steps. We quantise every vertex to a whole number of small steps so the output is exact bytes with no float in it. |
| **Normal** | The direction a surface faces at a point. Lighting needs it; collision does not. |
| **Greedy meshing** | Merging many coplanar square faces into a few big ones. It is the CUBE lane's trick (placed blocks), not this slice's. |

---

## 3. How surface nets work, in a drawing

A slice through the ground, two cells wide and two high. Each cell's centre holds a gap value in cells
(negative = inside rock).

```text
   air   +0.6 ●─────────● +0.8        ● = a cell centre with its gap
             │         │
             │    x    │              x = the ONE vertex this 2×2 group gets:
             │         │                  the average of the two crossings ○
   rock  −0.4 ●────○────● −0.2        ○ = where the gap reaches zero on an edge
                                          (between −0.4 and +0.6: t = 0.4 / (0.4 + 0.6) = 0.4)
```

In three dimensions the group is 2×2×2 cells (eight centres, twelve edges). The rule, in order:

1. Read the eight signs. All the same: no surface here, no vertex, go on. (Zero counts as air.)
2. For each of the twelve edges whose two ends differ in sign, find the crossing: `t = d0 / (d0 − d1)`.
3. The vertex is the average of those crossings (a sum divided by a count, in a fixed edge order).
4. For each crossed edge, emit one quad joining the four vertices of the four groups that share that
   edge, facing from rock to air.
5. Cut each quad into two triangles by the triangulation rule (§5).

Every operation is `+ − × ÷`, allowed under the fence. Nothing else is needed. That is why the
proposal chose it over dual contouring (a least-squares solve) and marching cubes (a 256-case table
that HR5 must cover).

**What it looks like in the game.** A hill is smooth. A cliff between two columns whose gaps jump from
`+0.9` to `−0.9` is a vertical face at the column boundary. A cave mouth is round. A crease at the foot
of a cliff is rounded by up to half a cell (50 cm at rung 0). A player who mines one cell leaves a
rounded bowl one metre across, not a cube socket; a player who heaps three cells of dirt on a slope
gets a knoll that merges into the slope.

---

## 4. The halo: what a chunk needs from beyond its edge

A chunk holds 62 × 62 × 62 cells. A 2×2×2 group on the chunk's edge needs one cell from the neighbour.
So the extractor works on 64 × 64 × 64 samples: the chunk plus a one-cell halo on every side. THAT is
why the chunk edge is 62 and not 64 (ruling A4's 64-chunk rule fits it): 62 + 2 = 64.

```text
        ┌──────────────────────────┐
        │ h  h  h  h  h  h  h  h  h│      h = a halo cell (the neighbour's)
        │ h ┌────────────────┐  h  │      c = the chunk's own cell
        │ h │ c  c  c  c  c  │  h  │
        │ h │ c  c  c  c  c  │  h  │      The groups that make the chunk's vertices
        │ h │ c  c  c  c  c  │  h  │      cover the c's and reach one h in.
        │ h └────────────────┘  h  │
        │ h  h  h  h  h  h  h  h  h│
        └──────────────────────────┘
```

**The question is where the halo comes from.** Two ways:

- **(A) Fetch the neighbour chunks.** The extractor needs six neighbours (twelve at edges, eight at
  corners) before it can run. On the client that is a wait; on the shard it is a dependency; and a
  chunk's triangles then depend on WHICH neighbours were held.
- **(B) Generate the halo with the chunk.** Slice 5's column pass computes 62 × 62 columns per chunk
  column. Make it 64 × 64 (the halo columns included: 6.5 % more columns) and the radial layer above and
  below (two more cell layers). Then the extraction of ONE key is a function of `(seed, key)` and
  nothing else: no neighbour, no wait, no order.

**Recommend (B).** The reason is SL10 itself: no drift means two hosts compute the same triangles from
the same key, and (B) makes the key the ONLY input. The cost is 6.5 % more column work and two more
cell layers; the seams between same-rung chunks close by themselves because both chunks compute the
same halo values from the same seed. **Edits complicate this:** a mined cell in the halo belongs to the
neighbour chunk's diff. §7 answers it: the composition step applies the neighbour's edits to the halo
cells, so a chunk's diff INPUT is its own edits plus its neighbours' edge edits. That is data the host
already holds (it holds the neighbour chunk's diff if it holds the neighbour), and the rule is still one
hop.

**Example.** A player mines the last cell of chunk (19, 1, 4), right on its edge with chunk (20, 1, 4).
Both chunks re-extract: (19, 1, 4) because its own cell changed, (20, 1, 4) because a cell in its halo
changed. The hole is smooth across the chunk edge because both extractions saw the same cell.

---

## 5. The triangulation rule: which diagonal

```text
   A ●───────● B        A quad on a saddle. Cutting A–C makes a ridge;
     │ ╲     │          cutting B–D makes a valley. The middle differs
     │   ╲   │          by up to half a cell.
     │     ╲ │
   D ●───────● C        The rule must be ONE rule on both hosts.
```

Three candidate rules:

- **(A) A fixed diagonal by position.** Always A–C, or alternate by the parity of the group's index.
  Simplest; never a data-dependent compare. On a saddle it picks wrong half the time and the ground
  shows a small "pinch" the player can see and stand on.
- **(B) The SHORTER diagonal.** Compare the two diagonals' lengths, take the shorter one. This is the
  usual choice in terrain meshers: it follows the surface better on saddles. It is a compare of two
  numbers, which is a branch on a float — the thing §3 of the slice 5 document warned about ("a branch
  cannot flip on one ulp"). BUT with §6's integer vertices, the two lengths are integers: `dx² + dy² +
  dz²` in whole steps. The compare is exact on every target. A tie takes A–C.
- **(C) The diagonal with the smaller gap sum at its ends** (data from the field). Same exactness
  question, and it is not obviously better than (B).

**Recommend (B) on integer vertices with the A–C tie-break.** It makes a better surface than (A) and,
because the vertices are integers, the compare cannot drift.

---

## 6. The output: integer vertices, and what is NOT in the identity

**The vertex is written as three integers in CELL space at a named quantum.** The extractor computes
the average of the crossings in the fenced float, then rounds it to a whole number of steps of
`1/256 cell` (the `vertex_quantum`; slice 7's `ChunkGeometry` already names it). Why:

- The golden gate pins BYTES. A float vertex is 24 bytes that a reader cannot check by eye; an integer
  vertex is exact and small (three `i16` cover a chunk with its halo at 1/256: `64 × 256 = 16 384`).
- The triangulation compare (§5) becomes an integer compare.
- The collider (slice 11) and the client's mesh (slice 7) read the same integers; nothing is rounded
  twice.

One honest cost: the rounding is a snap of at most half a step, `1/512` cell = 2 mm at rung 0, 4 m at
rung 11. At rung 11 the cell is 2 048 m, so 4 m is the same fraction. It is far under the extractor's
own placement error (§8, the snap error, up to half a cell on a crease).

**Normals and colours are NOT in the identity.** A normal changes how light falls on a triangle; it
never changes where the triangle is. So the extractor does not emit normals: the CLIENT derives them
from the triangles it holds (the average of the adjoining triangles' facings), and a client on another
engine may do it its own way. Same for the substance colour. This keeps the identity as small as it
can be: what the player STANDS ON is frozen, what it LOOKS like is style (ruling V2.6, V72).

**The digest of an extracted chunk** folds the vertex list and the triangle list in order, with the
key, exactly as slice 5's chunk digest folds cells.

---

## 7. The composition order and the seating rule

**Composition.** Before the extractor runs, the chunk's cells are the generated shape PLUS everything
the seed did not decide, applied in one fixed order:

```text
   1 the generated shape (slice 5)      the recipe's cells, the halo included
   2 terrain cell edits                 a mined cell → gap +1 (air), Empty; a placed dirt cell → gap −1
   3 catalogue blocks                   the cell keeps its gap byte (ruling B: the gap survives under a
                                        block); the form changes the LANE (cube, shaped), not the field
   4 sub-metre blocks                   the same, plus the seating rule
   5 attachments                        no volume; they never touch the field
```

The order matters when two rows touch one cell: an edit and a block on the same cell resolve as "the
last row wins", and the order says which is last. This slice lands the composition as a PURE FUNCTION:
`compose(generated chunk, edit rows) → composed chunk`, where the edit rows are an in-memory list in
the record format slice 3 froze. The store that holds those rows is slice 9's; the lane that carries
them is slice 10's. Neither is needed to test the function.

**The seating rule.** A sub-metre block placed on a slope must sit ON the ground, and both hosts must
agree where the ground is inside that cell. The rule: the seat is the height where the gap crosses zero
along the cell's own radial, found from the cell's gap and its neighbour's above or below by the same
`t = d0 / (d0 − d1)` division, snapped to the sub-metre grid's step (1/8 cell). If the whole cell is
air, the seat is the cell floor; if the whole cell is rock, the placement is refused ("solid ground,
mine first").

**Example.** A player sets a 25 cm lamp post on a 30° slope. The cell's gap says the surface crosses
at 0.37 of the cell; the seat snaps to 3/8 = 0.375. Her client draws the post standing on the slope;
the planet's shard collides it at the same 0.375. Nobody stored a height, and nobody invented one.

---

## 8. The measurements the slice owes, FIRST (the owner's order)

| # | What | How | Gate |
|---|---|---|---|
| M6-1 | Extraction cost per chunk, every rung, on the home planet: the surface chunk of the same columns slice 5 measured | `terrain_cost` gains the pass | under 4 ms of worker time per rung-0 chunk (the proposal's share of the budget) |
| M6-2 | The WORST case: a 3-D checkerboard gap field (every group crossed) | a synthetic chunk in the bench, named as synthetic (it is a bound, not a world) | under 4 ms |
| M6-3 | Output size: vertices, triangles, bytes per chunk, typical and worst | the same bench | recorded; it gates the client's residency lead (slice 7) |
| M6-4 | The snap error: the distance between the extractor's vertex and the TRUE seed surface along the radial, over 10 000 crossings | a test on the home planet | recorded as a distribution; a crease shows up to half a cell, a plain far less |
| M6-5 | The halo's cost: 64 × 64 columns against 62 × 62 | the bench | about 6.5 % more, recorded |
| M6-6 | Five legs of the TRIANGLE table (§9's gate) | `just terrain-pin`, `just terrain-legs` | equal bytes |

Nothing below is built before M6-1 and M6-2 show the extractor fits its budget.

---

## 9. What you decide

| # | Question | Recommendation |
|---|---|---|
| S6-1 | The extractor | **Naive surface nets** over cell-centred gap samples (register row V9 (A)); dual contouring stays a reserved upgrade, and switching is a content door before the first saved world |
| S6-2 | The halo | **Generated with the chunk (§4 B):** 64 × 64 columns and two extra radial layers, so one key is the only input |
| S6-3 | The triangulation rule | **The shorter diagonal on integer vertices, tie → A–C (§5 B)** |
| S6-4 | The output form | **Integer vertices in cell space at 1/256 cell (three `i16`), triangle indices; the digest pins both** |
| S6-5 | Normals and colours | **Not in the identity; the client derives them from the triangles** |
| S6-6 | The composition order | **The five rows of §7, as a pure function landed now with an in-memory edit list** |
| S6-7 | The seating rule | **Landed now, as §7 states it: the zero crossing along the cell's radial, snapped to 1/8 cell; a rock cell refuses** |
| S6-8 | The tree expansion function | **Not this slice.** Ruling V4 made trees art assets on a server skeleton; the skeleton is slice 14's |
| S6-9 | The gate | **A TRIANGLE table beside the cell table: the surface chunk of the 72 columns at every rung (864 rows), a COMPOSED row (the generated shape + one mined cell + one placed block + one sub-metre block, digested after composition and extraction), a SEATING row (a 1/8 m post on a 30° slope); all five legs** |
| S6-10 | The version | **Stays 1 if the cave lattice step (V35, "decide on pictures") is not changed by slice 7's first pictures.** If the pictures change it, the version bumps BEFORE the first saved world, at no cost |

---

## 10. Laws

- **SL10 clause 2 (one generator, a port forbidden):** the extractor, the triangulation rule, the
  composition and the seating are in `vd-terrain`, under the same fence, the same lint, the same link
  scan, the same golden legs.
- **SL10 clause 5 (the server collides on the same shape):** slice 11 reads this slice's triangles;
  nothing else makes a collider.
- **SL5 (one world):** every gate row is the home planet's; the checkerboard is named a synthetic
  BOUND, never a world, and no test draws a surface from it.
- **HR3:** the extractor does not know what realm it serves. A hull that holds terrain (ruling V4) runs
  the same code on a flat grid (the bend is the identity there).
- **HR5:** surface nets have no case table, so 100 % is a matter of one crossed group of each sign
  pattern that matters, not 256.
- **SL8 (seamless):** the crack between rungs and the arrival pop are slice 8's; this slice must leave
  them solvable: a chunk's edge vertices at rung L are a pure function of the seed, so the skirt has
  fixed edges to hang from.

---

## 11. How it is built

1. Measurements M6-1..M6-5 with a first straight-line extractor on the bench (nothing pinned).
2. `crates/terrain/src/extract.rs` (the nets, the rule, the integer output), `compose.rs` (the five
   rows), `seat.rs` (the seating rule), the halo in `chunk.rs`'s column pass.
3. The triangle table recorded, the composed and seating rows, the five legs.
4. The Opus 5 refuter, every finding answered, the gates, the report, and your word to commit.

---

## 12. The vista (owner, 2026-09-08): what the horizon must look like, and what carries it

The owner showed a vista from a red-rock plateau: a forest to the horizon, mesas with sharp cliff tops
ten kilometres away, snow mountains behind them, a built tower on a far ridge, clouds, and haze that
fades the far ground. **The owner refuses impostors** (a far tree as a flat card) as not believable.
What makes the picture believable, and which slice carries each part:

| What the eye needs | What carries it | Slice |
|---|---|---|
| Ground to the horizon, tens of kilometres | the ladder: rung L draws a chunk of `62 · 2^L` m; rung 11 chunks are 127 km wide; the far rungs come from the same recipe with fewer octaves | 5 (done), 8 |
| No jump between rungs | the crossfade band per chunk (the library owns the weight, the engine paints it); geomorphing if the measured pop stays visible | 7, 8 |
| Sharp cliff tops at distance | the extractor's creases: surface nets round a crease by half a cell of the rung drawn; dual contouring keeps it sharp. SAME topology (one vertex per crossed group, quads), so the switch changes vertex PLACEMENT only: a version bump, no structural change. Decide on PICTURES before the first saved world | 6 (S6-1), 7 |
| Believable landforms (mesas, terraces, valleys) at all | the RECIPE's own features (terracing, erosion-like shaping), not the extractor's. The recipe is frozen at the first saved world, so the pictures of slice 7 come before that freeze | 5's version, 7 |
| Fine detail on far ground | the client evaluates the rung-0 field on a texture for a coarse chunk and bakes a normal map from the seed (SL10 lets the client derive static shape) | 7 |
| A forest to the horizon, no cards | NEAR: the tree's full art asset; MID: the asset's own mesh detail levels, instanced (still meshes); FAR: the CANOPY FOLD (ruling C-4, R-15 YES): the coarse rung carries canopy height and density, drawn as a textured surface — the "textured noise" look the owner liked | 14, 8 |
| A built tower visible from ten kilometres | the pyramid's coarse entry holds occupancy and substance for built cells (S5-3), so a tower stands as a coarse column at every rung | 9 |
| Haze, clouds, sky | the client's atmosphere (aerial perspective hides the last of any rung transition); the weather plan | client, later |

**What this changes in slice 6:** nothing in the decisions, one addition to S6-1's note — the output
topology is shared by surface nets and dual contouring, so the extractor choice is reversible at the
price of a version bump until the first saved world, and the pictures of slice 7 are the gate that
decides it.

---

## 13. Results (2026-09-08, after the refuter)

**What landed** (uncommitted until the owner's word):

```text
  crates/seed      seam.rs MOVED from the core (re-exported there) with THE crossing rule `across`,
                   which the grid's neighbour step and the generator's halo now share
  crates/terrain   lattice  the SAMPLE BOX: a chunk plus a one-cell halo, GENERATED from the seed by
                            the same per-cell rule the neighbour runs; `site_of` (own face / partner
                            face across a seam / corner phantom) and its inverse `local_of_site`;
                            partial chunks at a face's far edge hold the partner's cells beyond the face
                   extract  naive surface nets in EXACT INTEGER ARITHMETIC (per-axis rationals in i64,
                            rounded half to even to 1/256 cell); a PRISM at every cube corner; quads per
                            owned crossed edge; the shorter diagonal; zero-area triangles dropped
                   position the ONE mapping of a vertex to metres: trilinear (cube) or barycentric
                            (corner prism) over the cell centres, summed in one global order
                   compose  the five-row composition order as a pure function; halo cells are targets
                   seat     the seating rule in eighths, exact, with refusals
                   chunk    the cell pass is SITE-aware (a partial chunk's cells beyond the face ARE the
                            partner's, byte for byte); NodeLattice + foreign lattices; the shared
                            per-cell tail and fill rules
                   digest   mesh_digest; the boot self-check folds cell AND mesh digests
                   strata   Stratum::Empty (the removal, code 18, registry substance `void`)
  gates            tests/mesh_pin.rs + golden_home_mesh.txt (1 082 rows: 90 columns × 12 rungs — the
                   strides, the corner, the last full edge chunk, and the three LAST chunks (partial)
                   per face — plus a COMPOSED row and a SEATING row); golden_home.txt re-recorded with
                   the same columns (3 240 rows); `just terrain-pin` runs both; `terrain-legs` runs
                   both on aarch64 Linux and emulated x86-64
  measurements     terrain_cost: the sample box and the extraction per rung on a named set (every
                   rung's column, the seam chunk, the corner chunk, a cave-dense seam chunk), the
                   halo's cost, the checkerboard bound, the snap error
```

**Measured** (release, this Mac; every chunk named so the numbers can be re-derived):

| Chunk | Sample box | Extraction | Vertices | Triangles |
|---|---|---|---|---|
| +X rung 0, chunk (3, 5) | 1.98 ms | 1.32 ms | 5 650 | 10 852 |
| +X rung 2, chunk (3, 5), cave-dense | 2.83 ms | 2.39 ms | 23 084 | 44 342 |
| +X rung 0, the last (partial) chunk (84 892, 7): a seam | 1.89 ms | 1.34 ms | 6 057 | 11 692 |
| +X rung 0, the corner chunk (84 892, 84 892) | 1.66 ms | 1.22 ms | 3 359 | 6 488 |
| +X rung 2, the last chunk (21 223, 7): a seam, cave-dense | 4.38 ms | 1.73 ms | 8 234 | 15 384 |
| +X rung 11, chunk (3, 5) | 1.05 ms | 1.28 ms | 4 055 | 7 856 |

- The halo costs 35 % over the bare chunk at rung 0 (1.46 → 1.98 ms): ten per cent more cells,
  six per cent more columns, the partner lattices, and the loss of the skips on the halo layers.
- The snap error on surface vertices (57 137 of them; 32 202 cave-wall vertices set aside by the
  sign rule): p50 0.0039 cells, p90 0.0061, p99 0.0074, max 0.0090 — under a centimetre at rung 0.
- The synthetic checkerboard bound (every cell edge crossed, 250 046 vertices, 1.43 million
  triangles): 26 ms.
- The cell pass at rung 0 went from 578 µs (slice 5) to 716 µs with the site plumbing.

**THE BUDGET, decided.** §8 set 4 ms of worker time per rung-0 chunk. A surface chunk costs 3.3 ms
(box + extraction); a cave-dense chunk 6.1 ms. The owner set the budget at 8 ms of worker time
(2026-09-08: "if that becomes a problem over time, we rethink and reimplement"); the bench gates the
costliest named chunk against it. The extractor runs on a worker; what a descent needs is the ARRIVAL
RATE, which slice 8 measures. Recorded in `D-TERRAIN-2`.

**The refuter.** 20 findings, 8 stands; every finding answered (`verdicts/slice_06_refutation.md`,
the answers table at its end). Four moved the shape before any world was saved: the corner hole (the
phantom is diagonal to every real corner cell, so nobody owned its edges — now a PRISM group), the
three faces' disagreement on the corner vertex (now barycentric weights quantised to a common 256 with
a tie to the smaller site), a seam vertex that could land one quantum apart (round half to even), and
the golden tables that never covered a partial chunk (they do). Two properties are recorded as
measured facts rather than fixed: naive surface nets are non-manifold at an ambiguous face (four quads
on one edge), and a cell exactly on the surface gives coincident vertices whose zero-area triangle is
dropped, leaving a T-junction. Both are the collider slice's to judge, and dual contouring's manifold
variant is the reserved upgrade.

**Tests (release):** terrain 51 unit + 4 doc + the two pins (3 240 and 1 082 rows), seed 26, core
372, bins 58 + the home-body cross-pin; clippy clean on the workspace.

**Full `just coverage` (2026-09-08, after the refuter, third run):** Tier-A 100 % of merged source
lines and branch sides (the first run found 5 real misses in the new tests — a weak assertion, a
short-circuit, an `if let` arm, a dead clamp, a dead water arm — and the second run 5 more; all
closed with tests, then 0); io-prod 94.82 % / 95.32 % regions over the 94 floor.

**The other targets (`just terrain-legs`, Docker up):** the leaf's and the generator's tests with both
tables (3 240 cell rows, 1 082 triangle rows) pass on aarch64 Linux (`aarch64-unknown-linux-gnu`)
and under x86-64 emulation (`x86_64-unknown-linux-gnu`): every digest equal. The real x86-64 machine
stays owed (`D-TERRAIN-1`).
