# Topic 1 — The address and the grid

The first topic of the voxel foundation, explained for the owner's review before any code starts.
Source: `01_grid_family.md` (refuted twice, revised) and `00_proposed_voxel_foundation.md` §2.1.
Written in Simplified Technical English with examples from the game.

---

## 1. What this topic decides

This topic decides how a cell is named. Every later topic reads the name: the record, the pyramid, the
diffs, the mesher, the collider, the trees. If the name changes after the first world is saved, every
saved edit on every planet, hull and station must be rewritten. So the name is frozen first, and it is
frozen with care.

The name has three parts:

1. The grid family: how cells cover a round planet and how they cover a flat hull.
2. The address: the six numbers that name one cell.
3. The ladder: how cells get bigger for a far view, and how a planet's radius fits the ladder.

---

## 2. The grid family

A planet is round. A block is square. The cube-sphere grid joins the two.

Take a cube. Divide each face into square cells. Push every cell out to the sphere. Now each cell sits on
the round surface, and its four corners still line up with its neighbours. Six faces cover the whole
planet with no gap.

```text
            +---------+
            |   +Y    |
            |  face 2 |
  +---------+---------+---------+---------+
  |   -X    |   +Z    |   +X    |   -Z    |
  |  face 1 |  face 4 |  face 0 |  face 5 |
  +---------+---------+---------+---------+
            |   -Y    |
            |  face 3 |
            +---------+

  The cube net. Twelve edges join the faces. A cell on the right edge of face 4
  has a neighbour on the left edge of face 0. A table of twelve edge pairs says
  which face and which edge, and whether the direction flips.
```

A hull, a station and a small or irregular asteroid use a plain flat grid. Their cells are exact
one-metre cubes. The origin of the grid is the realm's own origin. Nothing is bent.

**The grid family is per-body data, never per realm kind (owner, V5).** A body the seed makes large
and round enough to be a world gets the cube-sphere, whether the taxonomy calls it a planet, a moon or
an asteroid. A small or irregular body gets the flat grid, and its shape comes from the density field
on that grid. Every asteroid is its own realm. It moves as its parent authors, and a player lands,
builds and mines on it under either grid.

**Why not the other families.** Each was scored against the laws and the requirements:

- A literal cube planet: a cube from orbit, and "down" tilts by 55 degrees at a corner. Refused for the look.
- A hex grid: blocks are not square, and a second mesher and a second catalogue are needed. Refused, because one grid must serve a hull and a planet (HR4).
- A flat grid laid over a round planet, as Space Engineers does: a wall at latitude 40 degrees becomes a staircase. Refused for building.
- A planet as a set of flat areas: a visible line at every area border. Refused, because a seam is a defect (SL8).

**The price of the cube-sphere, stated once.** The cells are not all the same size. Pushing a flat face
onto a sphere stretches the middle and squeezes the edges. A bend function corrects most of it. Under the
chosen face bend a cell is exactly 1.000 m at the centre of a face, 0.707 m by 0.992 m at the middle of a cube
edge, and 1.005 m near a corner.

```text
  cell edge (m)
  1.00 |                 .-----.
       |              .-'       '-.
  0.90 |           .-'             '-.
       |        .-'                   '-.
  0.80 |     .-'                         '-.
  0.71 |  .-'                               '-.
       +-------------------------------------------
        edge          face centre            edge
        middle                               middle

  The size of a cell across one face, from the middle of one cube edge,
  through the face centre, to the middle of the opposite edge.
```

The grid angle is 90 degrees at a face centre. At a cube corner three squares meet, so the angle there
is 60 or 120 degrees. No bend removes this. It is a property of a cube.

```text
         \   face +Y   /
          \    ____   /
           \  |    | /
   face -X  \ |____|/   face +Z
             \  |  /
              \ | /       Three squares meet at one corner.
               \|/        A box blueprint cannot be stamped across it.
                *         There are eight corners per body, at latitude 35.26 degrees.
```

**Example.** A player builds a stone wall along the equator of the home planet. Each block is a 1.000 m
square at the face centre. Forty kilometres later the wall crosses the middle of a cube edge, and each
block there is 0.707 m by 0.992 m. The player cannot see the change. A survey tool could measure it. A
hull built next to the wall is a flat grid, and every block on it is exactly 1 m.

**The corner is the one seam this topic leaves open.** A blueprint stamp that reaches a corner returns a
refusal. The player did not cause it and cannot see why. That is a "tier refusal" seam under SL8. The
recommendation is that the generator always places a landform at all eight corners: a mountain, a crater
or a sea, so no flat build site exists there. The corner never becomes a place where anybody builds.

---

## 3. The address

```text
  CellAddr = ( body , face , rung , i , j , k )

   body : the realm whose grid this is (Planet 7, Ship 44, Station 12 ...)
   face : 0 to 5 on a planet; always 0 on a hull
   rung : the detail level; a cell is 2^rung metres; rung 0 is the only writable rung
   i, j : the two positions along the face, whole numbers, signed
   k    : the radial position, whole metres from the body's floor radius, signed
```

Three rules make this address lawful and stable:

- **The address is realm-local.** A planet is centred on itself. The face points along the planet's own
  axes. The star system's placement of the planet is nowhere in the address. A cell never learns where
  its planet is (SL1).
- **The address is an integer.** The server and the client compute the same integer from the same
  position with the same crate. No frame conversion sits between the address and the realm's own
  position lattice, because the grid's origin is the realm's origin (SL10).
- **The indices are signed.** A hull's origin is its own frame origin, so a thruster one metre aft has
  `k = -1`. An unsigned number would file it at the far bow. A bias instead of a sign would force the
  client to know the hull's slot size, which does not reach the client today. The sign costs one bit
  and removes a request across a boundary.

```text
  A hull's flat grid, seen from the side (k = fore/aft):

            k
            ^
     bow  +3 | . . . . . .
          +2 | . . . . . .
          +1 | . . . . . .
   origin  0 | . . O . . .      O = the hull's own origin
          -1 | . . T . . .      T = a thruster at k = -1
          -2 | . . . . . .
   stern  -3 | . . . . . .
```

**Example.** The moon's shard and the pilot's client both name the cell under the landing pad as
(Moon 7, face +Y, rung 0, i 126971, j 127004, k 8400). The star system moves the moon along its orbit
every tick. The address does not change by one bit.

**The chunk is a packing, not a name.** Cells are stored and streamed in chunks of 62 by 62 by 62. The
chunk key comes from the cell address by an integer divide. The address does not depend on the chunk.

**The address does not fit in one 64-bit number.** With signed fields, a planet of 42,723 km radius, a
65 km radial band, four rung bits, and the sub-site of Topic 3, the address needs 95 bits. So the stored
form splits into a chunk key and a cell-in-chunk index. Topic 2, the record, owns that split.

**Spin.** A planet's cells rotate with the body. The star system authors the planet's placement and its
orientation every tick, as it does today. The grid adds no statement.

---

## 4. The ladder: rungs, and a planet's radius

Each rung doubles the cell. Rung 0 is one metre. Rung 3 is eight metres. A far view reads a coarse rung.
Only rung 0 is ever written.

```text
  rung 0 :  1 m   |#|#|#|#|#|#|#|#|
  rung 1 :  2 m   |# #|# #|# #|# #|
  rung 2 :  4 m   |#   #  |#   #  |
  rung 3 :  8 m   |#       #      |
```

For every rung to tile a face exactly, the number of cells along a face edge must be a multiple of the
coarsest cell. The rule: with T rungs, the cell count N along an edge is any multiple of 2 to the power
(T minus 1). The radius is then 2N over pi metres.

The old design forced whole chunks along a face edge and got a 39.47 m step between allowed radii. That
step was an artefact. The radial axis already tolerates a partial chunk at the top of the band, so the
face edges can too. With the new rule a body lands within 0.01 percent of its seed radius:

| body | seed radius | rungs | snapped radius | error |
|---|---|---|---|---|
| asteroid | 500 m | 1 | 500 m | 0.3 m |
| moon | 200 km | 8 | 199,970 m | 30 m |
| Luna | 1,737 km | 11 | 1,737,310 m | 310 m |
| Earth | 6,371 km | 13 | 6,370,354 m | 646 m |

The number of rungs comes from the radius: the top rung is the one at which a face is at most 64 by 64
chunks. Beyond the top rung the realm draws its own look, the same smooth sphere it draws as a dot at
its reach. Nothing is authored. It is per-body data from the seed.

**Example.** The home planet's seed draws a radius of 6,371,000 m. The generator snaps the cell count to
a multiple of 4,096 and the planet states its look at 6,370,354 m. A pilot watches the planet grow from
a dot. When the first chunks arrive at rung 12, they sit on that same sphere, so nothing pops.

**Today's code draws a continuous radius** from the mass-radius law. The snap changes the world tag. It
is free before the first chunk is saved, and a migration after.

---

## 5. The face bend, and why its inverse is frozen

The face bend maps a face position to a direction on the sphere. Under SL10 it may use only add, subtract,
multiply, divide and square root. The chosen face bend is a polynomial of degree five with three constants
that sum to exactly 1.0, followed by one normalisation. This is the same lineage as the quadrilateralized
spherical cube used for sky maps, and the area-corrected cube maps that whole-planet engines use.

The inverse, from a direction back to a cell, has no closed form. It runs a fixed number of Newton
steps from a first guess. The first guess and the step count decide which cell a position falls in near
a cell boundary. Measured on 200,001 samples: three steps from the wrong first guess leave a third of a
cell of error at the largest legal planet, which is a margin of only 1.4 against half a cell. Four steps
from the recommended guess reach the floating-point floor. So the first guess and the step count are
both frozen with the face bend, and both are part of the generator's identity.

**Example.** A pilot mines a cell on the home planet. The shard runs the inverse face bend on the pilot's
position and names one cell. A client with a different first guess could name the neighbouring cell.
The version tag would not refuse that client, because only the arithmetic inside one function differs.
That is why the guess and the count live inside the frozen identity.

**One more rule, for aiming.** The no-drift gate proves the function, not the inputs. If the client
holds a position that differs from the server's by a hair, the floor could land one cell apart. So the
server names the cell, and the client draws the highlight from the server's answer. The mined cell and
the highlighted cell are the same by construction.

---

## 6. The seam: two grid arms behind one value

Everything above the seam is written once: the chunk container, the palette, the mesher, the smooth
extractor, the edit path, the collider derivation, the diff codec, the pyramid. It never sees which grid
it runs on.

```text
      features: mesher, extractor, edits, pyramid, collider   (written ONCE)
                              |
                         GridMapping
                        /           \
                Identity            CubeSphere
             hull, station,        planet, moon
               asteroid
```

Six operations: the address of a position, the centre of a cell, the eight corners of a cell, the
neighbour of a cell in one of six directions, the grid's radial direction, and the domain. The neighbour
operation is the only one that knows about face edges: on a planet it can answer "across the seam, with
this axis swap"; on a hull it never does.

This is the only match on geometry in the whole workspace, and it matches on a value the realm carries,
never on a shard kind (HR3). The hard rule HR4 today names this seam "FrameSpace" with a tangent anchor
that could be re-centred. That anchor existed for a physics library that wanted 32-bit floats. Today no
32-bit float sits on the authoritative path, so the grid seam is stateless. Renaming the hard rule's
wording is your decision (D8).

**Gravity is not a grid operation.** After your ruling on stations and hulls, the seam's "up" is only
the grid's radial direction for terrain. Gravity is a separate function the realm states: toward a
centre for a moon, away from an axis for a cylinder, nothing for a station, and later a field from a
graviblock. The character controller reads the gravity function, never the grid.

**Example.** A miner digs the same trench with the same tool in two places: on a moon, and in a
station's soil bay. One fixture, two grid arms, two stores compared byte for byte. That is HR4's gate.

---

## 7. Small blocks inside the cell

The position lattice already counts in steps of 1/1024 m. A small block at 1/8 m is addressed by three
of those bits below the whole metre. No division, no float. On a planet the same shift runs in index
space after the inverse face bend, so a small block inherits its cell's shape exactly.

```text
  One 1 m cell seen from above, on the 1/8 m lattice:

  +--+--+--+--+--+--+--+--+
  |  |  |  |  |  |  |  |  |
  +--+--+--+--+--+--+--+--+
  |  |  |##|##|##|##|  |  |     ## = a rail, 1/8 m by 1/8 m in section,
  +--+--+--+--+--+--+--+--+          half a metre long
  |  |  |  |  |  |  |  |  |
  +--+--+--+--+--+--+--+--+
  ...
```

The address reserves four bits of level and twelve bits of index, so 1/16 m fits without a change. The
shipped step is 1/8 m and stays provisional until the frame cost of a full cell of 512 small blocks is
measured.

On a planet a 1/8 m part measures 0.088 m at the middle of a cube edge and 0.125 m at the face centre.
A player can see 3.7 cm on a rail at arm's length. Refusing small blocks on planets would give a ground
base less detail than a hull, which is a seam. The recommendation allows them everywhere (D11).

---

## 8. Areas on a planet

A spaceport district is its own realm with its own frame. Its cells are the planet's cells. It reads
its told berth as an instrument to evaluate the planet's shape under itself. That is lawful (SL1 clause
2). But it stores and ships every edit at an area-local address, never at a planet-global one, because
a planet-global address would carry the told berth onward, and clause 4 forbids that. The conversion
happens in the parent and, for the picture, in the gateway.

One cell has one writer. The area writes every cell inside its box. The planet writes every cell outside
every area box, and refuses a cell inside a live area.

**Example.** A player digs a trench that runs out of the spaceport district onto open ground. The
district writes the cells inside its box. The planet writes the cells outside. No cell has two authors,
and the trench is continuous in the picture the gateway composes.

---

## 9. Your questions, answered

**Is a separate design run needed?** No, for this topic. The report was written, refuted by a law lens
and a feasibility lens, and revised against every finding. The structure is settled. What remains is
not design but measurement: the face bend residual, the constants' sum, the cell-size spread and the
round-trip test, all measured in Python by an agent and not yet by our crate. The slice's own tests are
those measurements. If you want one more check, a single adversarial pass on the face bend arithmetic and
the corner rule takes an hour and adds no new agents beyond that.

**Is it performant?** The address of a position costs about forty float operations on a planet and a
few integer shifts on a hull. It is a lookup, never a scan, so the cost never grows with the planet's
size or the number of children (SL9). It is not on the containment path, which reads a boundary and
never asks which cell anything is in. The neighbour operation is an integer compare and a table read.
The operations that cost real time, meshing and colliding, belong to later topics.

**Is this the approach of large games?** The cube-sphere with an area-corrected face bend is the standard
for whole-planet engines: Kerbal Space Program's planets, Outerra, and the planet technology of the
large space games use a cube-mapped sphere with a quadtree of detail levels. Voxel games that ship round
planets mostly lay a flat lattice over the sphere, which gives the staircase we refuse. Our choice joins
the planet engines' grid with the voxel game's cell lattice. The polynomial face bend descends from the
quadrilateralized spherical cube of sky-survey astronomy. The sub-metre lattice as the high bits of the
position lattice is our own, and it costs nothing.

**Does it keep the laws?** SL1: the address holds no parent frame, and an area stores area-local
addresses. SL5: one grid, one world; the exhaustive seam test at 62 cells proves the face table, which
does not depend on the planet's size, and the round-trip test runs at the largest legal size. SL9:
every lookup is constant time. SL10: integer address, polynomial-plus-root face bend, a frozen inverse. HR3:
one match on a geometry value. HR4: two arms behind one seam, with a gate for each half. SL6: no new
data crosses a boundary, because the address is signed.

**Is it seamless?** The terrain is a function of the direction on the sphere, so the world has no line
at a face edge. Only the index space does, and the seam table hides it. The ladder snap happens before
the first chunk, so no player sees a radius change. The one open seam is the corner refusal, and the
recommended cure removes it from the player's experience.

---

## 10. What exactly is built for this topic

One slice, in the core crate, before any generator or mesher code:

1. A grid module with the address type, the seam value with its two arms, the six operations, the
   face bend with constant coefficients, the constant first guess and step count, and the generated face
   table with its pinned literal.
2. The grid parameters derived at each use from the realm's seed and its bound. No record changes. No
   store changes. No wire arm.
3. One wiring line where a realm's app is built: the realm's geometry value selects the arm.
4. The gates: the exhaustive face-table test at 62 cells; the round-trip test at the largest legal size
   sampled at the worst face bend position; the constants' sum as a compile-time assertion; the negative-half
   test for a hull's stern; the sub-lattice identity test; and the no-drift golden digest of cell
   centres on both targets.
5. The measurements: the inverse residual in cells, the cell-size spread, the cost of one address
   lookup, and the largest look-shell move on the world when the ladder snaps.

Before this slice, the sequence puts two slices that are not design topics: splitting the 15,550-line
sim file, and widening the persistence seam. Both can start without a decision from you.

---

## 11. What you decide for this topic

| # | Question | Recommendation |
|---|---|---|
| D1 | The grid family | Cube-sphere for a body, flat for a built realm |
| D2 | The finest small-block step | Width frozen at 1/16 m; ship 1/8 m, provisional until the cost is measured |
| D3 | How HR4's gate applies to the seam itself | Split: features above the seam pass the identical fixture; the seam passes the exhaustive table test and the largest-size round trip |
| D4a | An area's addresses | Area-local; the berth is an instrument only |
| D4b | One writer per cell | The area inside its box; the planet outside |
| D5 | A built realm's slot | A box, not a shell |
| D6 | The rung count | Top rung = a face is at most 64 chunks |
| D7 | The physics library and its precision | Investigate together, later; it decides only whether an anchor exists below the grid |
| D8 | May HR4's wording change from a stateful FrameSpace to the stateless grid seam? | Yes |
| D9 | The corner refusal | The generator always places a landform at all eight corners |
| D10 | A rotation joint's grid | A mount: a second flat grid inside the same realm |
| D11 | Small blocks on planet cells | Allowed everywhere |
