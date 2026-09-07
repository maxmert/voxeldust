# The proposed voxel foundation, explained step by step

A plain walk through `00_proposed_voxel_foundation.md`, in Simplified Technical English, with an
example from the game at each step. This file explains. The proposal decides nothing until the owner
rules.

---

## Step 1. Where a cell is: the address

Every realm has its own grid. A planet and a moon have a cube-sphere grid: six faces of a cube, bent
onto a ball. A hull, a station and an asteroid have a flat grid. Both grids use one address: the realm,
the face, the rung (the detail level), and three whole numbers. The numbers count cells from the
realm's own centre. They can be negative.

The address holds no parent frame. A cell never says where its realm is in the star system. This keeps
the law "a realm is told where it is; it never decides".

The finest rung is one metre. Each coarser rung doubles the cell size. A planet's radius decides how many
rungs it has.

**Example.** A pilot lands on Moon 7. The moon's shard names the cell under the landing pad: face up,
rung 0, and three numbers. The pilot's client names the same cell with the same six values. Neither
asked the star system where the moon is. On the pilot's hull, the cell one metre aft of the hull's own
origin has the third number set to minus one.

---

## Step 2. Two forms in one grid: smooth ground and square blocks

A cell holds one form. A terrain cell holds a substance (rock, dirt, sand) and one small signed number:
the distance from the cell centre to the ground surface. Negative means "inside the rock". A smooth
extractor reads these numbers across many cells and builds a smooth surface. That is how the ground
looks realistic while it is still made of cells.

A built cell holds one of the ~20 square shapes (cube, slope, corner, and so on) with a rotation. The
cube lane and the shaped lane build walls from these cells.

The form of the cell chooses the lane. The kind of realm never does. A hull can hold terrain cells, for
example soil in a ballast hold, if the owner says yes.

Mining removes what a cell holds and the surface reflows. Placing a terrain cell adds a full cell of
rock and the surface reflows again. Placing a square block adds a hard shape.

**Example.** A player sets a steel foundation into a hillside on a moon. The dirt around the foundation
stays smooth. The foundation stays a cube. The moon's shard sends both through one collider builder.
When the player removes the foundation, the slope returns exactly, because the gap number under the
block was kept.

---

## Step 3. Small blocks inside the metre cell

A block smaller than one metre is the same record with a sub-site: a scale (half, quarter, eighth) and
a position on a lattice of eight by eight by eight inside the cell. The metre cell stays the unit of
space-taking, of mining and of the far view. A small block never crosses a cell edge.

Collision inside a cell is the union of the small boxes. Nothing outside the cell learns that small
blocks exist.

The width of the sub-site address is frozen now and holds 1/16 m. The shipped step is 1/8 m, and it
stays provisional until one measurement runs: the frame cost of one cell filled with 512 eighth-scale
blocks.

**Example.** An engineer builds a railing on a hull's bridge from 1/8 m posts. The four cells become
sub-grid cells. A crewmate cannot drop a crate into them. The hull's shard collides with the posts as
small boxes.

---

## Step 4. Attachments: a HUD, a joint, a rail

An attachment is not a cell. It is a row in a side table of the realm's own store. Its key is the block's
full address, a face, and a slot number. It has no volume, no collider and no place in the mesh. The
block above it stays placeable.

A HUD is an attachment on a face. In the future it listens to a signal and shows it with a widget. The
foundation only reserves the row and the signal binding tags. It draws nothing until the signal system
lands, because a picture that never changes is a placeholder, and placeholders are refused.

A rotation joint is the same kind of row, and it creates a second rigid body inside the same realm.
The turret is never a child realm. A constraint needs both bodies in one physics world, and a shard's
physics world is private. The turret's plates keep integer addresses on the turret's own grid, under
their own body id.

A rail and a piston are the same mechanism with a different degree of freedom.

**Example.** A gunner turns a turret 37 degrees on a hull's spine. The hull's shard integrates the joint
and authors the turret's pose every tick. The turret's angle rides in the hull's own row in the picture,
under the same time stamp as the hull, so the platform never slides under the gunner's boots for one
frame.

---

## Step 5. The one generator: what the client derives, what the server sends

One Rust crate holds everything that is a function of the seed and the address, and of nothing else:
the height and density fields, the strata, the biomes, the cave carvers, the tree skeletons, the
surface extractor, the rule that splits a square into two triangles, the rule that seats a small block on
a slope, and the order in which the seed's shape and the realm's edits combine.

The server compiles this crate. Every client compiles the same crate. A client on another engine links it as a
static library through a C interface. A port to another language is forbidden.

The arithmetic is fenced three ways. A float type exposes only add, subtract, multiply, divide, square
root, floor and abs. A lint refuses every other float call. A scan of the built library refuses any
sine, cosine, exponent or fused multiply-add symbol, even one that arrives through a dependency. Each
fence has a control test that must be seen to fail.

No drift is a measurement. A gate generates about 832 chunk digests on the server build and on the
client build, on this Mac and on an x86-64 machine, in debug and in release. One differing byte is red.
No leg of this gate has run yet.

Derived on both hosts from the seed: the terrain, the seed forests, the common substances, the coarse
rungs. Sent as a diff from the owning realm: every edit, every placed block, every small block, every
attachment, every growth stage, every damage state, every valuable deposit.

**Example.** A client draws a hill from the moon's seed while the hull descends. The moon's shard
evaluates the same crate for the chunk under the boots. A tunnel that another player dug last week
arrives as a diff. Both hosts compose the tunnel over the derived hill before either one draws or
collides.

---

## Step 6. The record: what is saved for one cell

Base terrain has no record. Only what a player changed has a record. One record is twelve bytes, not
eight. Smooth terrain needs the gap number on every planet cell, and a tree needs a shape parameter that
the server reads to build its collider. Neither fits in the old width.

The record holds: the cell index inside its chunk, the kind, the rotation, the provenance (terrain,
biome feature, or placed), the sub-site, the style variant, the gap number, the object parameter, and
21 reserved bits. A reader refuses any record with a set reserved bit, an unknown kind, or an illegal
rotation. It never fills in a default, because a skipped record is a lost build.

A removal is a positive record: an Empty kind with provenance Placed. It must beat the seed's shape for
all time, because the client re-derives the seed's shape every time.

Mass and durability come from the kind's row in the registry. They are never saved per cell. Style is a
six-bit variant per cell. The client derives the decorative trim from the variant and the neighbours,
and never stores it. The trim may add geometry, but it may never move a vertex that the collider shares.

**Example.** A player cuts down an oak on a moon. A week later her client re-derives that chunk from the
seed. The Empty record on the oak's cell keeps the oak down. A shipwright plates a hull with fluted
plates. The client bevels every welded seam. The moon's shard collides with the plain plates, and the
bevel never lifts a crewmate's boots off the spine she can see.

---

## Step 7. The pyramid: how a far view sees edits

Only the finest rung is written. Each coarser rung is folded from the rung below by one exact integer
rule: which of the eight children hold something, the mean fill, the dominant substance. An entry exists
only where the fold differs from the generator's own coarse answer at the same address. An entry that
equals the generator is deleted. This keeps the store sparse.

The pyramid is derived. A crash can rebuild it from the finest rung. So it is written at the checkpoint,
not inside the write-ahead log.

This puts one hard requirement on the generator: its coarse answer must equal the fold of its own fine
cells. Dropping noise octaves is not the fold. The naive exact fold of one top-rung cell would cost
about 1.9 hours. This is the largest unresolved requirement in the foundation, and a bench must answer
it before the generator crate lands.

A small edit rounds away at a coarse rung. A 20 m quarry survives to the 128 m rung and vanishes at the
256 m rung. Whether the player can see the step is unmeasured.

**Example.** A quarry 200 m wide is a torrent of fine rows for the players inside it. For a pilot five
kilometres away it is eight coarse cells that say "rock, half full, bottom half".

---

## Step 8. The diff lane: how edits reach a window

The client derives the seed's shape and needs only the diffs for the chunks in its interest. A diff is a
window statement per chunk: an address plus a tagged bag of rows. It rides its own reliable, paced lane,
so a large city never queues in front of the transfer's cut marker. A joint's live angle rides the
realm's per-tick lane instead, because it is live state, not a change of shape.

When a chunk enters a client's interest, the client receives the chunk's whole diff set first, then the
incremental rows. Diffs ship durable-first, one tick after the write, so a client never draws a wall
that a restarted shard lacks.

Two players who edit one cell in one tick: a two-stage receipt refuses the loser.

An edit made from inside a landed hull on the moon below is forwarded from the hull's shard to the
moon's shard, one hop, in the hull's own frame. The moon adds the placement it authored and writes once
under its own fence. This forward is a new wire arm and needs the owner's YES. The design can live
without it in the first version by refusing cross-realm edits.

**Example.** A pilot inside a hull berthed on a moon breaks a rock beside the ramp. The hull's shard
sends the edit up to the moon. The moon writes it once. If the outbox delivers it twice, the second copy
is a no-op.

---

## Step 9. Trees as one object

A tree is one record on one cell, plus one small side row while it grows. One function in the generator
crate expands the kind, the instance seed, the growth stage and the size into an integer skeleton. The
client draws the full tree from the skeleton. The shard turns the skeleton into a compound of capsules
and collides with the trunk. The canopy lets a player pass through, per kind, by default.

A seed forest costs zero bytes on both hosts. Felling a seed tree is one twelve-byte removal record.
Planting is a record plus a stored growth stage. The instance seed comes from the cell address, so a
replanted cell grows the same tree, and the record can prune.

Felling must reach every coarse rung the canopy spanned, or a far client keeps drawing a forest that is
gone. That canopy fold needs a home, and the proposal puts it in a second store family under the same
key.

The object horizon, the distance at which a tree becomes part of the terrain fold, is derived per kind
and per growth stage from the tree's own size and the drawable floor. It is never a constant.

**Example.** A pilot walks to within ten centimetres of the bark, walks through the leaves, and chops the
oak. A stump and a leaning crown appear in the same frame. A logger fells a hundred oaks around a
landing pad. A pilot at four kilometres reads the coarse rungs and sees the clearing.

---

## Step 10. The render seam: what an engine gets

The engine-free client library hands an engine one thing per chunk: chunk geometry. It holds integer
vertices in cell space at a named quantum, packed attributes, indices, a part table for joints, and a
header with the rung and the skirt. The engine never learns the word "sphere".

The library owns the tier rule (which rung at which distance), the chunk order (coarse before fine),
and the crossfade band (how two rungs blend). The engine owns only the thread that runs the work and
the shader that paints the blend. This split is what lets the same seam gate pass on Bevy and on
any later engine.

The mesher lives in the shared crate. The drawn surface and the collided surface are one function.

**Example.** A hull descends onto a moon. On the Bevy client the coarse ground arrives first and the
fine cells dissolve in over four frames. If another engine's plugin owned the queue, fine cells could arrive
first and the ground would appear in patches. The same seam gate would pass on one client and fail on
the other, and nobody could tell which is the world's fault. So the library owns the order.

---

## Step 11. Physics: the same surface under the boots

The shard builds one collider per chunk per realm as a triangle mesh from the same extractor output the
client drew with. One rule in the crate says which diagonal splits a square, so the picture and the
collider never choose opposite diagonals on a saddle.

The build runs on a worker pool, never on the tick thread. A chunk enters the physics world only at a
tick boundary, from a completed build. A body is never stepped past the frontier of built chunks. If the
frontier falls behind, that is a gate failure, never a silent tunnel.

The ceiling left the flight path, so a tick's swept segment can be longer than any shard can build. The
answer is not a clamp. The shard asks the crate for the surface along the segment analytically, cell by
cell, builds no chunk, and builds the one chunk a contact lands in.

A parked child's shell body sleeps. The per-tick cost grows with the moving children, never with the
parked ones.

Gravity is a function of position from the realm's authored mass. Up is the radial. Slope and step
thresholds are character data, never constants.

There is no crossing at the ground. A planet's bound is its sphere of influence, so a hull became the
planet's child hundreds of thousands of kilometres up. The descent, the touch-down and the take-off all
happen in one realm.

**Example.** A pilot switches off a hull's safety block and dives at a moon at ten kilometres a second.
In one tick the hull travels 500 metres. The shard walks that line through the crate, finds the ridge,
and builds one chunk to resolve the contact. Six hundred hulls parked at a spaceport stay asleep while
one pilot fires her thrusters.

---

## Step 12. The landing from orbit to a walk: no seam

The client's resident band of fine chunks always contains the shard's collider set for every body the
observer can touch, at every closing speed. The band is sized from data the server states: the realm's
own reach, the interpolation buffer, and a warm lead the owning shard states beside them. The client
never subtracts two delivered poses to work out a speed, because that is a derived velocity, and the law
forbids it. Whether the reach plus the buffer is enough without the lead is unmeasured; if not, the
lead is a real request to the owner.

Rungs cross over inside a dither band. The realm's own look is the rung above the coarsest generated
rung, drawn from the same crate at the same radius, so the far view and the near view are one ladder.

The acceptance gate flies one scripted path from orbit to a walk, with a physical tolerance for each of
the eleven seam kinds. A twelfth seam kind is proposed: the diff lag. The diff for every cell inside the
character's reach is applied before the derived shape for that cell is first drawn. The frame count
between the two must be zero.

**Example.** A hull dives at a moon at its rated cruise. The moon states how much ground to warm. The
client reads that number and builds ahead. The pilot lands, walks off the ramp, and the ridge under her
boots is the ridge her client drew, because both hosts ran the same crate over the same diff.

---

## Step 13. The four formats the owner freezes in one sitting

- **A, the address.** Cube-sphere for a body, flat for a built realm. Signed indices. Four-bit rung.
  The sub-site width holds 1/16 m. A body id in the store key, from a counter that is never reused.
- **B, the cell record.** Twelve bytes, fixed. The fields of Step 6. A removal is an Empty record.
- **C, the pyramid entry.** Eight bytes. Fourteen reserved bits, and three wanted spends need seventeen.
  The recommendation keeps eight bytes and puts the canopy fold in a second family.
- **D, the generator's world identity.** Not a byte layout but the exact function from seed and address
  to ground: the crate version, the arithmetic profile, the noise, the octaves, the warp constants, the
  extractor, the triangle rule, the seating rule, the fold, and the one owner of a body's radius. After
  the first saved world, the only lawful change is append-only detail below a stated tolerance in
  metres. Cost if wrong: every edit ever saved.

**Example.** A player builds a landing pad on a hill. Six months later the generator gains one octave.
Under the append-only rule the hill moves less than the stated tolerance and the pad still rests on it.
Without the rule the hill is 40 cm higher and the pad floats.

---

## Step 14. The sequence of slices

Each slice has a gate that can go red and a number with a pass threshold. Every feature slice passes the
identical fixture on two shard kinds.

0. **The format sitting.** No code. The owner answers the four formats and says YES or NO to every SL6
   request by name. Nothing is planted without a YES.
0b. **Split the 15,550-line sim shard file.** Blocked on nothing, and it gets more expensive every day.
1. **Widen the persistence seam.** A point read, a bounded range, per-realm handles, and the memory twin.
2. **The geometry seam.** The address and the two grid arms behind one value. Exhaustive seam test at a
   small planet, round-trip test at the largest.
3. **The registry and the shape catalogue.** Five tables, the theme column, the identity digest, the ~20
   shapes and the 24 rotations.
4. **The wire plant.** Only the arms the owner approved, in one reviewed file, with the closed-set gate.
5. **The generator crate with its fence.** The three fences, the no-drift gate on two targets, the
   per-chunk cost re-derived from a hull at 250 m/s.
6. **The extractor inside the crate.** The triangle rule, the seating rule, the composition order. A
   pinned triangle list and a composed golden row.
7. **The client links the crate; the render seam.** One rung drawn, no shape branch in the renderer, a
   control that fails on a placement symbol.
8. **The detail ladder and the crossfade**, all in the client library. The pop detector on a hull at
   walking pace, at 240 m/s and at 528 m/s.
9. **The block store, the write-ahead log and the pyramid.** Two shard kinds compared row for row. A
   kill during an edit storm replays byte-identical. The store is throw-away until slice 14.
10. **Mining and placing on terrain; the diff lane.** A placed terrain cell reads as solid rock. A player
    may raise ground. The re-extract of one edited chunk under 16 ms.
10b. **The realm re-states what a build changed.** Mass, cross-section, drag and look radius, once per
    settle window, never per placement.
11. **The terrain collider and the residency rule.** The landing fixture flown by a hull. The swept
    analytic query. Parked hulls asleep.
12. **The square lanes on a planet, and small blocks.** The shell swap with a capped settle, so a welded
    plate never lifts a hull by a metre in one tick.
13. **Attachments and the body tree.** Joints on an integer angle grid with a carried sine table. A
    byte-identical body pose on both targets.
14. **Trees as one object.** Fell a thousand seed trees at twelve bytes each. No stump without a crown
    for one frame.
15. **The cross-shard edit forward.** The store's bulk moves before the transfer's cut. The cut carries
    only the tail.
16. **The character on the surface.** Gravity from position, the walk off the ramp, and the walked legs
    of slices 8, 11 and 12.
17. **The seamless landing gate.** One scripted flight from orbit to a walk, one number per seam kind.

Thirteen slices are renderer-agnostic and can run beside a later client-engine change. The seam contract of slice 7
must shut before either another engine's plugin or the Bevy encoder is written.

---

## Step 15. What the owner decides

Band A has 32 starred rows. The ones that block the first code:

- The grid family, and the twelve-byte record.
- May a hull or a station hold terrain cells? If NO, the smooth lane cannot pass on two shard kinds, and
  slice 10 cannot land.
- Is common ore such as iron seed-decided and public? This relaxes the seed ruling's absolute sentence by
  name. The ruling's own test: if this were printed on a public wiki tomorrow, would the game still work?
- How to read "one block type with different params". The proposal reads mass and durability per kind
  and style per cell. The literal reading is per instance, and the record width depends on it.
- The realm store file name. Two star systems in two galaxies with the same seed overwrite each other's
  buildings today. Silent data loss.
- Where the canopy fold lives, how a cell returned to the seed is encoded, how a body id is allocated,
  and where a small block sits on a slope.

The SL6 table has 21 requests. The default is NO. The requests the design cannot live without: the diff
lane to the client, the reliable world action from the client, the realm's surface tag in its own look
bag, the client's generator handshake, the body pose in the realm's own row, the diff tags for
attachments and trees, the canopy fold, the exact cheap coarse fold inside the generator, and felt
acceleration down to a child so a crewmate can stand in a thrusting hull. The requests the design can
live without in the first version: the cross-realm edit forward and its ack, a landed hull's collider
surface up to its parent, the falling crown, the planet's diff down to a surface area, a sibling's
placement down to a child, the warm lead, and buoyancy.

---

## Step 16. Open risks

- The exact cheap coarse fold inside the generator. If it does not exist, the store is never sparse.
- The no-drift gate has never run, and the x86-64 leg has no host in this project. An emulated green is
  not an x86-64 green.
- The client handshake carries no generator tag today. The refusal must be built.
- The speed governor is still in the dot's drive code. Its deletion waits for the suit. No slice may
  size a number on it.
- Seventy-six numbers are unmeasured. Each has a bench, ordered by the slice that needs it.
