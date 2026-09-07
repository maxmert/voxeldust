# Slice 0 — The format sitting

The four formats the owner freezes together, and the SL6 table. Written in Simplified Technical English
with examples from the game. Source: `00_proposed_voxel_foundation.md` §2 and §5, and the domain reports
04 (the record), 06 (storage) and 03 (the generator).

A format is a thing that can never change after the first world any player builds on is saved. A wrong
answer costs a migration over every planet, hull, station and blueprint. That is why the four are decided
in one sitting: each reads a field the others define.

---

## Part A — The address (five answers still open)

The address itself is decided (Topic 1). Five questions around it are open.

| # | Question | Recommendation | Why |
|---|---|---|---|
| A1 | HR4's gate on the seam itself | Split it: features above the seam pass the identical fixture on two shard kinds; the seam passes the exhaustive face-table test at 62 cells and the round trip at the largest legal planet | No fixture can give equal results on a bent grid and a flat one. Without the split the one piece everything stands on has no gate |
| A2 | HR4's wording | Change "FrameSpace" to the stateless grid seam; an anchor, if a physics library ever needs one, sits below it | No 32-bit float sits on the authoritative path today, so the anchor has no reason yet |
| A3 | A built realm's slot | A box | A flat grid's domain is a box. A round slot wastes 42 percent of the sold volume |
| A4 | The rung count | The top rung is the one at which a face is at most 64 by 64 chunks | Any other rule forces radii onto a coarse ladder, and you asked for realistic radii |
| A5 | The eight corners | The generator always places a landform there | No flat build site exists at a corner, so no player meets the refusal |

**Example for A5.** A colonist lays a straight wall out from a spaceport on the home planet. Forty
kilometres later the wall reaches a cube corner. Under A5 there is a crater there. The wall stops at the
crater's rim, which the colonist can see. Without A5 the stamp refuses at an invisible line.

---

## Part B — The cell record

### B1. What a record is

Base terrain has no record. The generator makes it from the seed on both hosts. A record exists only
for a cell a player changed: a placed block, a mined cell, a felled tree, a small block. The record is
twelve bytes, fixed. It is written once and never rewritten by the clock.

```text
  WORD A  (8 bytes)  where it is, and what it is
  +----------+---------+-------+-------+-------+--------+-------------+
  | cell 18  | kind 16 | rot 6 | prov 2| scl 2 | sub 9  | reserved 11 |
  +----------+---------+-------+-------+-------+--------+-------------+

  WORD B  (4 bytes)  what it looks like, and what shape it takes
  +---------+------------+-----------+-------------+
  | style 6 | density 8  | object 8  | reserved 10 |
  +---------+------------+-----------+-------------+
```

- **cell**: which of the 62 x 62 x 62 cells of its chunk. The chunk key and the body id are the store key, so they cost the record nothing.
- **kind**: one row of the registry. 65,536 rows.
- **rot**: one of 24 rotations for a shaped block. Zero on a terrain cell. One bit spare.
- **prov**: provenance. Terrain, biome feature, or placed. The fourth value is refused.
- **scl, sub**: the small-block site. Scale 0 means a whole cell. Scale 3 means 1/8 m, and sub is the position on the 8 x 8 x 8 lattice.
- **style**: the variant. 64 styles per kind. The client derives decoration from it.
- **density**: the gap between the cell's centre and the ground, in 128ths of a cell. Negative means the centre is inside rock. Present on every terrain cell.
- **object**: a shape parameter for a landscape object such as a tree. Zero otherwise.
- **reserved**: 21 bits. A reader refuses a record with a set reserved bit. Nobody can spend them by accident.

### B2. Four records, written out

```text
  A fluted steel plate on Ship 44, whole cell, rotated code 5:
    cell 1234 | kind STEEL_PLATE | rot 5 | prov Placed | scl 0 | sub 0 | style 3 (fluted) | density 0 | object 0

  A mined cell on Moon 7 (the scoop left the ground 40/128 below the centre):
    cell 8801 | kind BASALT      | rot 0 | prov Placed | scl 0 | sub 0 | style 0 | density +40 | object 0

  A felled oak on Moon 7 (a removal):
    cell 512  | kind EMPTY       | rot 0 | prov Placed | scl 0 | sub 0 | style 0 | density 0 | object 0

  A 1/8 m steel post on a hull's bridge, at lattice position (3, 0, 5):
    cell 77   | kind STEEL_POST  | rot 0 | prov Placed | scl 3 | sub (3,0,5) | style 0 | density 0 | object 0
```

**Why twelve bytes and not eight.** Smooth terrain needs the density byte. A tree needs the object byte.
Neither fits in the old width. The cost: one hundred million edits take 1.2 GB instead of 0.8 GB, about
8 percent more than the pyramid beside them. ESTIMATED.

### B3. The rules that ride with the record

- **A removal is a positive record.** The kind is Empty, the provenance is Placed, everything else is
  zero. The client re-derives the seed's shape every time, so a removal must beat the seed for all time.
  Air is a substance, the atmosphere. It is never the removal marker. (Your ruling.)
- **The gap survives under a placed block.** When the block is removed, the slope returns exactly.
- **A small block's seat is derived, never stored.** The record holds a lattice offset. One shared rule
  in the generator crate drops the block onto the surface inside its cell on both hosts. A stored seat
  would go stale if the generator ever moved the surface.
- **Mass, durability, liquidity, melting point and flammability are per kind**, in the registry's
  substance row. They are never in the record. A new behaviour is a new registry column. (Your 15.4.)
- **Style is per cell.** The client derives trim from the style and the neighbours. Trim may add
  geometry. It may never move a vertex the collider shares.
- **The decoder refuses, never defaults.** An unknown kind, a bad rotation, a set reserved bit, a style
  above the kind's count: the store refuses to open. A skipped record is a lost build.

### B4. The attachment row

An attachment is not a cell. It is a row in a side table.

```text
  key   = (block address, face 0..5, slot byte)
  value = tagged fields: kind, params, placed_by, fence,
          tags 16..31 reserved for a signal binding (a name and a scope),
          tags 32..47 reserved for a joint's live motor state
```

The key has a slot byte from the first write, because a key byte cannot be added later. The value is
tagged, so new fields are added forever without a migration.

**Example.** A pilot bolts a fuel gauge onto the aft face of a tank block, slot 0. Later she adds a
warning lamp on the same face, slot 1. Both rows sit beside the tank's record. Neither takes a cell.

### B5. What you decide for Part B

| # | Question | Recommendation |
|---|---|---|
| B-1 | The width | Twelve bytes, fixed |
| B-2 | The kind id | 16 bits: 65,536 kinds, and the registry caps at the full width, not the old 512 |
| B-3 | The density byte | The radial gap at the cell centre, 1/128 of a cell, signed; present on every terrain cell; kept under a placed block |
| B-4 | The rotation width | 6 bits, the sixth zero and reserved |
| B-5 | The tree's shape byte | In the record; the growth stage in a side row, because the stage changes with time |
| B-6 | The removal | Kind Empty, provenance Placed (answered by you) |
| B-7 | The attachment key | Block address, face, and a slot byte from the first write |
| B-8 | Attachment kinds | A second typed registry table, one digest over both |
| B-9 | The body id | Body 0 is the realm's own grid; every other id from a counter that is never reused, persisted in the realm's store |
| B-10 | The small-block seat | Derived by one shared rule; nothing stored |
| B-11 | Slots per face | Several, because the slot byte is in the key (gameplay may still limit the count) |
| B-12 | The registry digest at the client's handshake | A prefix digest, plus a per-chunk refusal when a chunk names a kind the client lacks. Equality would refuse every unpatched pilot at the gateway on every theme update |
| B-13 | What the digest covers | Only what changes what a saved record means: the kind triple, the shape, the object flag, the theme, the density convention. Never the style count, so a new style never refuses a saved world |

**Example for B-12.** An art update ships a crystal spire for a themed moon. Under equality a pilot who
has not patched cannot enter the galaxy. Under the prefix rule she flies everywhere except that moon's
chunks, and the refusal names the kind she lacks.

---

## Part C — The pyramid entry

### C1. What it is

Only rung 0 is written. A coarse rung is folded from the rung below. One entry summarises eight children.

```text
  eight rung-0 children            one rung-1 parent
  +---+---+                        +-------+
  | # | # |  top layer             |       |
  +---+---+                        |  ###  |   occupancy: which of the 8 hold something
  | # |   |                        |  ##   |   fill: the mean, 0..255
  +---+---+                        |       |   substance: the dominant one
  +---+---+  bottom layer          +-------+
  | # | # |
  +---+---+
  |   |   |
  +---+---+
```

```text
  PyramidEntry (8 bytes)
  +----------+--------------+---------+---------+-------------+
  | cell 18  | substance 16 | occ 8   | fill 8  | reserved 14 |
  +----------+--------------+---------+---------+-------------+
```

An entry exists only where the fold differs from the generator's own coarse answer at the same address.
Where they are equal, the entry is deleted. That is what keeps the store sparse. The client folds the
diffs it holds by the same rule, so the far view of an edit needs no second message. (Your step 7.)

The entry is derived. A crash rebuilds it from rung 0. So it is written at the checkpoint, not inside
the write-ahead log.

**Example.** A quarry 200 m wide on a moon. At rung 0 it is tens of thousands of records. At rung 7,
128 m cells, it is a handful of entries that say "rock, half full, bottom half". At rung 8 the quarry is
under half a unit of fill and the entry is deleted: from 256 m cells the quarry is gone.

### C2. The bit budget problem

Fourteen bits are reserved. Three spends want seventeen:

| Spend | Bits | What it buys | Without it |
|---|---|---|---|
| A signed surface-height delta | 8 | a coarse rung whose silhouette matches the fine ground | the coarse rung draws a stepped fill |
| A sticky "differs from the seed" bit | 1 | a 20 m quarry never vanishes at the 256 m rung | a fly-away seam, if the step is visible (unmeasured) |
| The canopy fold: occupancy, height, colour of trees | 8 to 12 | the far view draws the forest as it is after felling | a felled forest comes back at a distance |

### C3. What you decide for Part C

| # | Question | Recommendation |
|---|---|---|
| C-1 | The prune rule | Equality with the generator's coarse answer |
| C-2 | The surface-height delta | Decide with the renderer before the first world; default no |
| C-3 | The sticky bit | Default no, provisional; the fly-away measurement decides, and the stores before slice 14 are throw-away so the cost of turning it on later is a rebuild, not a migration |
| C-4 | Where the canopy fold lives | A second store family under the same key, so the 8-byte entry never widens |
| C-5 | Where the pyramid persists | At the checkpoint, under a watermark |

---

## Part D — The generator's world identity

### D1. What it is

Not a byte layout. The exact function from the seed and an address to the ground. Every saved byte of
Parts B and C is a delta against it. If the function moves, every edit in the world sits on the wrong
ground.

```text
  +---------------------------------------------------------------+
  |  WORLD IDENTITY  (sealed when the first world is saved)       |
  |                                                               |
  |  the crate version          the arithmetic profile            |
  |  the noise and its hash     the octave table                  |
  |  the cave lattice step      the face bend constants,          |
  |                               first guess and step count      |
  |  the strata tables          the biome tables                  |
  |  the feature anchors        the tree skeleton format          |
  |  THE EXTRACTOR              THE TRIANGLE RULE                 |
  |  THE COMPOSITION ORDER      THE SEATING RULE                  |
  |  the exact coarse fold      the ONE owner of a body's radius  |
  +---------------------------------------------------------------+
```

After the seal, the only lawful change is append-only: a new octave may add detail below a stated
tolerance in metres. Nothing may move a cell's surface by more than that tolerance. There is no re-roll.

**Example.** A player builds a landing pad on a hill. Six months later the generator gains one octave.
Under the append-only rule the hill moves less than the stated tolerance and the pad still rests on it.
Without the rule the hill is 40 cm higher and the pad floats.

**Why the triangle rule is inside.** A square on a saddle splits into two triangles two ways. The two
ways differ by half a cell in the middle. If the client splits one way and the shard the other, the boots
sink into a crest the player can see.

**Why the coarse fold is inside.** The pyramid deletes an entry where the fold equals the generator's
coarse answer. If the generator's coarse answer is not the exact fold of its own fine cells, nothing ever
prunes and the store is never sparse. The naive exact fold of one top-rung cell costs about 1.9 hours.
This is the largest unresolved requirement in the foundation. A bench decides it before the generator
crate lands. If no cheap exact fold exists, Part C changes: the coarse rungs become stored, not derived.

### D2. What you decide for Part D

| # | Question | Recommendation |
|---|---|---|
| D-1 | Freeze the identity as a document beside A, B and C, with the append-only tolerance in metres | Yes; you state the tolerance |
| D-2 | Who owns a body's radius | The generator crate; the forest reads it. Today the radius comes out of a power function on two targets, which is the drift SL10 exists to stop. The cost is a little astrophysical fidelity |
| D-3 | The cave lattice step | Decide on pictures before the identity is sealed, because under smooth terrain the step is the cave wall a player walks into |
| D-4 | The noise source | Vendor about 400 lines of gradient noise on our integer hash, inside the fenced float type; delete the unused noise crate pin. No new dependency |
| D-5 | The x86-64 leg of the no-drift gate | Emulation as a smoke test now; a real x86-64 machine before the law is called satisfied |
| D-6 | The world tag | Two halves: a declared half on the store stamp and the handshake, a measured boot digest on live handshakes only, so a store is never refused because a pod landed on a different chip |

---

## Part E — The SL6 table

SL6 says: ask before new data crosses a realm boundary, and before a wire arm is added. The default is
NO. Slice 4 plants only the rows you mark YES.

```text
  the three directions a datum can travel:

   parent realm  <-------------->  child realm        a REALM-TO-REALM crossing (SL6 proper)

   realm  ------>  gateway  ------>  client           not a realm boundary (SL2, 2026-08-24),
                                                      but a new wire arm, so it is asked too

   client ------>  gateway  ------>  realm            the same, upward
```

### E1. The rows the design cannot live without (recommend YES)

| # | What | Direction | Why the receiver cannot compute it | Without it |
|---|---|---|---|---|
| R-3 | The chunk diff rows and manifest on a paced reliable class | realm to gateway to named sessions | the client derives the seed's shape and cannot derive an edit | no edit is ever visible; the client's shape disagrees with the shard's collider on every edited cell |
| R-5 | A reliable world action from the client | client to gateway to the session's authority | the input lane is latest-wins and unreliable; a dropped datagram is a vanished placement | no reliable edits |
| R-8 | The realm's own surface tag in its look bag: its frame and its declared generator tag | the realm about itself, to a subscribed client | the client cannot know whether the moon's shard runs a matching generator, nor that a hull has no surface | the client draws every realm from its seed on faith, or none |
| R-9 | The client's generator handshake | client to gateway | the gateway cannot know what generator a client binary holds | a client draws a hill the shard does not have and finds out when the boots fall through. MEASURED: this refusal does not exist today |
| R-11 | A body's pose, such as a turret's angle, as a tagged entry in its realm's own row | realm to gateway to observers | the client only renders; the angle is live state from a motor | every mechanism is drawn welded shut |
| R-13 | New tags inside the chunk diff: cell records, box records, attachments, body id, object rows | realm to client, inside R-3 | an edit, an attachment and a chop are player state | a placed HUD, hinge, piston, rail and every felled tree are invisible |
| R-15 | The canopy fold in the coarse rung | realm to client | the fold must include what players felled, which is state | a felled forest comes back at a distance |
| R-7 | An exact cheap coarse fold inside the generator crate | a code ask, not a wire ask | the prune rule compares against it | the store is never sparse |
| R-20 | Felt acceleration down to a child, in the child's frame, on change | a hull's parent realm to the hull's shard | the hull states an acceleration up and never learns what the parent did with it | a crewmate floats or falls to the wrong wall whenever the hull burns. This carries "walk inside a ship while it flies" |

**Example for R-20.** A pilot burns at 3 g out of a moon's well. Her crewmate in the hold must be pressed
to the deck. The deck the hull calls "down" is the parent's answer, not the hull's.

### E2. The rows the design can postpone (recommend NO for the first version)

| # | What | Direction | Why it can wait |
|---|---|---|---|
| R-1, R-2 | The cross-realm edit forward and its ack | child realm to parent realm | you postponed it; cross-realm edits are refused in v1 |
| R-6 | A landed hull's collider surface, centre of mass and inertia | hull to its parent | without it a hull lands on its bounding box. The better answer (V60 option c) is that the hull resolves its own contact against the parent's terrain, which it can generate from the seed. It needs the parent's nearby edits, which is R-16's shape. Decide at the collider topic |
| R-10 | A mesh peer's measured arithmetic profile | node to node | the build gate catches drift instead |
| R-12 | A HUD's live value | realm to observers | the overlay draws nothing until signals land |
| R-14 | A falling crown as a transient entity | realm to client | needs a continuity model that is not built; a chopped crown can vanish at the cut for now, which is a known black-frame seam to fix at the tree topic |
| R-16 | The planet's diff for the cells under an area | planet to area | disappears entirely if an area's floor is built |
| R-17 | A sibling's placement down to a child, for a contact list | star system to hull | the voxel foundation does not need it; targeting and docking do, later |
| R-18 | A warm lead per realm | realm to client | a real ask only if the realm's reach plus the interpolation buffer is not enough; a measurement decides |
| R-19 | One interest radius per window | gateway to realm | held in reserve |
| R-21 | Buoyancy volume up on change | child to parent | matters when oceans and thick atmospheres land |

### E3. Withdrawn

| # | What | Why |
|---|---|---|
| R-4 | A rung floor pushed down from the gateway | the realm knows its own bound and the drawable floor, so it works out its own floor. A pushed floor is a clamp on the realm's own drawing, which SL3 forbids |

### E4. Refused without an ask

A grandparent's gravity field into a child. Any realm-to-realm chunk fetch. Any occupant pose on the
edit forward. The originating session's server pose.

---

## What happens after you answer

- Part A's five answers open slice 2, the address.
- Part B's answers open slice 3, the registry.
- Part E's YES rows open slice 4, the wire plant, which plants only those.
- Parts C and D open slice 5, the generator crate.
- Slices 0b and 1 wait on nothing and can start today.
