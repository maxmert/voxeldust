# 00 — The proposed voxel foundation

> **PROPOSED — NOT BINDING until the owner rules; promoted to `docs/design/` on the owner's word.**

**Date:** 2026-09-07. **Status:** the synthesis of ten domain reports, each refuted twice and revised.
**Serves:** `docs/design/owner_decisions_2026-09-07_voxels.md` V3 — the design, the formats, the
implementation sequence, and the register of decisions still open.

**Revision 2, 2026-09-07.** Two critics refuted revision 1: a law critic (twenty findings, three of them
law breaks in the first slices) and a completeness critic (twenty-two dropped or contradicted items,
four of them format-level). This revision answers every finding. §9, the revision log, lists each one and
what changed. **V3.2 asked for THREE formats; this revision hands the owner FOUR**, because the
generator's own identity is the thing every saved byte is a delta against, and revision 1 left it in the
register instead of the freeze.

**How to read a number.** MEASURED means a program ran and this document says which. ESTIMATED means
arithmetic on a cited number. UNMEASURED means nobody has run it. A number with no mark is a defect in
this document.

**How to read a code claim.** Every one cites `file:line` in this worktree. The code is the only truth
about what exists today. The investigation base of 2026-08-03/04 is an input, not a decision; §8 lists
what it says that a ruling now refuses.

**What this document does NOT do.** It adopts no library. It states no ruling. It closes no one-way
door. Every choice below is a recommendation with a named alternative, and §4 is the list the owner
answers.

---

## 1. The design in one page

**The grid.** A planet and a moon carry a cube-sphere grid; a hull, a station and an asteroid carry a
plain flat grid. One address space serves both: `(body realm, face, tier, i, j, k)`, all integers, all
signed, all inside the realm's own frame. No parent frame is in the address, so a realm never learns
where it is from a cell (SL1). The two arms sit behind ONE seam value, `GridMapping`, and every feature
above the seam is written once (HR4). *Example: the moon's shard and the pilot's client both name the
cell under the landing pad `(Moon 7, face +Y, tier 0, i 126 971, j 127 004, k 8 400)`; on the pilot's
hull, one metre aft of the hull's own origin is `(Ship 44, face 0, tier 0, i 0, j 0, k −1)`.*

**Two forms in one grid.** A cell holds ONE form. A TERRAIN cell holds a substance and one signed byte
— the radial gap between the cell centre and the surface, negative inside rock, in steps of 1/128 of a
cell. A BUILT cell holds one of the ~20 square shapes with a rotation. A smooth extractor turns the gap
lattice into the ground; the cube and shaped lanes turn built cells into walls. The lane is chosen by
the cell's FORM, never by the realm's kind (HR3). *Example: a player sets a steel foundation into a
hillside on a moon. The dirt around it stays smooth, the foundation stays a cube, and both go through
one collider builder.*

**Sub-metre blocks inside the 1 m cell.** A block smaller than a metre is the SAME record with a
sub-site: a scale and a position on a 1/8 m lattice inside the cell. The metre cell stays the unit of
space-taking, of mining and of the pyramid, so the seam table, the aprons and the fills never learn
that sub-blocks exist. *Example: an engineer builds a railing on a hull's bridge from 1/8 m posts; the
four cells become sub-grid cells, a crewmate cannot drop a crate into them, and the shard's collider is
the rail's own boxes.*

**Attachments that take no space.** A HUD, a lamp, a hinge or a rail clamp is NOT a cell. It is a row
in a side table of the realm's own store, keyed by the block's full address plus a face plus a slot. It
has no volume, no collider and no place in the mesh, and the block above it stays placeable. A ROTATION
JOINT is the same row and it makes a second rigid BODY inside the SAME realm — never a child realm,
because a constraint needs both bodies in one physics world and a shard's physics world is private
(HR1). *Example: a gunner turns a turret 37° on a hull's spine; the turret's plates keep their integer
addresses on the turret's own grid, and the hull's shard authors the turret's pose every tick.*

**The one generator under SL10.** ONE Rust crate holds everything that is a function of `(seed, address)`
and of nothing else: the body definition, the height field, the density field, the strata, the biome,
the carvers, the seed-placed feature anchors and their geometry, THE SURFACE EXTRACTOR, and THE
COMPOSITION ORDER in which the generated shape and the realm's diff combine. The server compiles it and
every client compiles the same crate; a client on another engine links its static library through a C interface.
Its arithmetic is fenced by a float newtype that exposes `+ − × ÷ √ floor abs` and nothing else, by a
crate-scoped lint, and by a link scan of the built archive that no inlining can hide from. DERIVED: the
terrain, the trees, the common substances, the coarse rungs. DIFFED: every edit, every placed block,
every sub-metre block, every attachment, every growth stage, every damage state, every valuable deposit.
*Example: a client draws a hill from the moon's seed while the hull descends; the moon's shard evaluates
the same crate for the chunk under the boots; a tunnel dug last week arrives as a diff and BOTH hosts
compose it over the derived hill before either draws or collides.*

**The edit pyramid and the diff lane.** Only tier 0 is written. Every coarser rung is folded from the
rung below by one exact integer rule, and an entry exists only where the fold differs from the
generator's own coarse answer at the same address. That prune rule puts a hard requirement on the
generator: its coarse answer must EQUAL the fold of its own fine cells, or nothing ever prunes. The diff
reaches a client as a window statement per chunk — an address plus a tagged bag — on its own reliable,
paced lane, so a 23 MB city never queues in front of the transfer cut marker. *Example: a quarry 200 m
wide is a torrent of tier-0 rows to the players in it and eight coarse cells that say "rock, half full,
bottom octants" to a player 5 km away.*

**Trees as one object.** A tree is ONE record on ONE cell plus, while it grows, one small side row. One
function in the generator crate expands `(kind, instance seed, stage, size)` into an integer skeleton;
the client draws it, the shard turns it into a capsule compound and collides with it. A seed forest
costs zero bytes on both hosts. Felling is a TWELVE-byte removal record — one `Empty` record at Format B's
frozen width — and the removal must reach every coarse rung the canopy spanned, or a far client keeps
drawing a forest that is gone. *Example: a pilot walks to within ten centimetres of the bark, walks
through the leaves, and chops the oak; a stump and a leaning crown appear in the same frame.*

**The engine-agnostic render seam.** The client library hands an engine `ChunkGeometry`: cell-space
integer vertices at a named vertex quantum, packed attributes, indices, a part-transform table for
joints, and a header with the tier and the skirt. The engine never learns the word "sphere", exactly as
today (`crates/client/src/realm_scene.rs:792-794`). The library owns the TIER rule, the chunk ORDER and
the crossfade BAND; the engine owns only the thread that runs the work and the shader that paints the
blend. That split is what makes the detail-by-box tolerance gateable on a client on another engine as it is on
Bevy, and it is what stops SL8's continuity forking with the engine. The mesher lives in the shared
crate, so the drawn surface and the collided surface are ONE function.

**Physics on the same surface.** The shard builds each terrain chunk's collider as a triangle mesh from
the SAME extractor output the client drew with, one collider per chunk per realm, demanded by a SWEPT,
altitude-aware ring around each body — never a fixed ball, because the owner took the ceiling off the
flight path and no per-tick motion bound exists any more. ONE rule inside the generator crate splits a
quad into two triangles, so the client's picture and the shard's collider never pick opposite diagonals.
When a tick's swept segment is longer than the built frontier, the shard asks the crate for the surface
along the segment ANALYTICALLY, cell by cell, composed with the realm's OWN diff, which it holds; it
builds no chunk and allocates no collider, and it builds the ONE chunk a contact lands in. A re-clamp of
the speed is a jump and SL8 refuses it. The build runs
on a worker pool, never on the tick thread, and a chunk enters the physics world only at a tick boundary
from a COMPLETED build. A parked child realm's shell body SLEEPS, so the per-tick cost grows with the
MOVING children and never with the parked ones (SL9). Gravity is a function of position from the realm's
authored mass; up is the radial; slope and step thresholds are character data, not constants.

**There is no crossing at the ground.** A planet realm's bound is its sphere of influence
(`crates/physics/src/worldgen/generate.rs:1796-1798`), ESTIMATED at about 1.5 × 10⁹ m for an
Earth-class planet, so a hull became the planet's child hundreds of thousands of kilometres up. The
whole descent, the touch-down and the take-off happen inside ONE realm, with one authority, one
integrator and one collider world. This deletes a seam nobody has to build, and the owner should know
that it is deleted. *Example: a pilot dives at a moon from orbit; the star system handed the hull down
at the sphere of influence long before the atmosphere, so nothing hands over at the landing pad.*

**How the landing stays seamless.** The client's resident tier-0 band STRICTLY CONTAINS the shard's
collider set for every body its observer can touch, at every closing speed. The band is sized from data
the SERVER states and the client only reads — the realm's own reach, the interpolation buffer
(`crates/wire/src/channels.rs:328`, `INTERP_BUFFER_MS`) and a lead the owning shard states beside them —
because SL10 V1.7 says the client never derives a velocity, and the spread of a delivered track over
time IS a velocity. The shard already sums a lead of the same shape: *"the interest lead is the closing
speed times this buffer plus one tick"* (`crates/wire/src/channels.rs:322-328`).
Rungs cross over inside a dither band. The realm's own look is the rung ABOVE the coarsest generated
rung, drawn at the same radius from the same crate, so the proxy hand-over is one more rung and not a
different kind of thing. And a TWELFTH seam kind is proposed: THE DIFF LAG — the diff for every cell
inside the character's own reach is applied BEFORE the derived shape for that cell is first drawn, and
the gate counts the frames between the two.

---

## 2. The formats the owner freezes together

**FOUR formats, ONE sitting.** They share one deadline — before the first world any player builds on is
saved — and each reads a field the others define. V3.2 named three; revision 2 adds Format D, the
generator's world identity (§2.4), because the address, the record and the pyramid entry are all deltas
against it and freezing them without it freezes nothing.

**What must be settled before this sitting can shut anything.**

1. The owner answers §5 — the SL6 requests — as well as §4 band A. Slice 4 plants seven items the law
   refuses by default, and nobody has asked.
2. ★V38 is answered, because HR4 depends on it: if a Cartesian realm may not hold terrain-form cells,
   slice 10 has no second shard kind and cannot land.
3. The body-id ALLOCATION rule joins the address freeze (★V104).
4. The pyramid's three candidate bit spends are costed together (★V102, the table in §2.3).
5. The TRIANGULATION rule joins the extractor's freeze (§2.4), and the SEATING rule joins the record's
   (★V106).

### 2.1 Format A — the grid family (the address)

```text
CellAddr {
  body:  RealmId   // the realm whose grid this is (Planet(seed) | Ship(id) | Station(seed) | Area(..))
  face:  u8        // 0..=5 on a cube-sphere; always 0 on a flat grid
  tier:  u8        // the detail rung; a cell is 2^tier metres; tier 0 is the only writable rung
  i, j:  i32       // SIGNED tangential index at this tier
  k:     i32       // SIGNED radial index at this tier (whole metres from the body's floor radius)
}

BlockAddr { body_id: BodyId(u16), chunk: ChunkKey, cell: CellIndex(18 bits),
            sub_scale: 2 bits, sub_addr: 9 bits }     // the STORE and EDIT address
```

**What it reserves.**

- `i, j, k` are SIGNED. A flat grid's origin IS the hull's own frame origin
  (`crates/core/src/pose.rs:91`), so a thruster one metre aft of it has `k = −1`. An unsigned field
  would file it at the far bow or refuse the placement. A bias by `half_cells` is worse: the client
  could not decode an address without the realm's `bound` (`crates/core/src/built.rs:76`), which does
  not reach the client today, so a bias creates an SL6 request the signed field dissolves.
- `tier` is 4 bits, `face` 3 bits, `i, j` 27 bits each to reach `N = 2²⁶` (a 42 723 km radius),
  `k` 18 bits, the sub-site 16 bits reserved (level 4 + index 3×4). **Total 95 bits.** A cell address
  does not fit one `u64`; the packing splits into a chunk key and a cell-plus-sub-site index.
- `BodyId(u16)` costs the cell record ZERO bits — it lives in the store key and in the diff frame's
  header. Six reserved bits of a chunk key would cap a construction at 64 actuators, which is a cap
  discovered in play.
- **How a body id is ALLOCATED is part of the address, not a later detail.** Body 0 is reserved for the
  realm's own grid. Every other id is drawn from a MONOTONE counter held in the realm's own store, and
  an id is NEVER reused, even after the body it named is cut off. The counter is persisted beside the
  body family and survives a store reopen. *Example: a gunner bolts a turret onto a hull's spine and its
  plates are filed under body 1. The gunner cuts the turret away and welds a crane in its place. The
  crane takes body 2, so the turret's old chunks can never answer the crane's key, and a saved hull
  never loads with plates that belong to a machine that is gone.*
- `tier` is FOUR bits — sixteen rungs against the thirteen the ladder uses. The render report's door row
  that names five bits (`08_render_seam_seamless.md:991`) is SUPERSEDED: the tier width is frozen with
  the address here, and one width is stated once.
- The chunk edge (62 cells) is a PACKING, not part of the address. An 18-bit `CellIndex` holds any edge
  up to 64.

**Its one-way doors, with deadlines.**

| Door | Deadline | Cost if wrong |
|---|---|---|
| The grid family: cube-sphere for a body, flat for a built realm | before any generator code | every planet's shape; whether ONE block system exists |
| The warp `W(a) = k₁a + k₂a³ + k₃a⁵`, its evaluation order, **the inverse's first guess `a₀ = t`**, and the `const` step count FOUR | at the first golden digest | a one-cell aiming disagreement that no version tag can catch. MEASURED 2026-09-07 (200 001 samples, python3 IEEE f64): three steps from `a₀ = t/k₁` leave **0.348 cells** at the largest legal body, a margin of 1.4 against half a cell; four steps from either guess reach the f64 floor (2.22 × 10⁻¹⁶) |
| The face basis table and the seam table's orientation conventions | same digest | every seam re-pairs; every apron gather is wrong |
| The indices are SIGNED | with the address | a hull cannot address its own stern; a migration over every built realm |
| The ladder `N = q·2^(T−1)`, `R = 2N/π`, the radial floor in whole metres | before the first body is saved | a body cannot be re-radiused under player builds. MEASURED against the rule: Earth-sized lands within 646 m (0.01 %) of its seed radius |
| The rung-count rule (top rung = a face is ≤ 64 chunks) | same | the persisted tier ceiling per body |
| The sub-cell address width (level 4 bits + index 3×4 bits) | with the record | a migration over every planet, hull, station and blueprint |
| `BodyId` in `BlockAddr` and in the chunk STORE prefix | with the record | every saved multi-body construction is ambiguous |
| **The body-id ALLOCATION rule** — body 0 = the realm's own grid; a monotone per-realm counter, never reused, persisted in the realm's own store | with the record | a reused id makes a cut-off turret's chunks answer a new crane's key; every saved multi-body construction is ambiguous and the ambiguity is silent |
| The `CellIndex` linearisation `(c·62 + b)·62 + a`, `a` fastest, and the sub-lattice `(z·8 + y)·8 + x`, `x` fastest | with the record | a renumber reinterprets every saved cell |
| The fine step 2⁻¹⁰ m | **ALREADY SHUT** (`crates/core/src/store_stamp.rs:118`; `crates/core/src/pose.rs:255`) | a 1 024× misplacement, refused at the door |

**Recommended answers.** Cube-sphere plus flat, signed indices, `a₀ = t` with four Newton steps, the
`N = q·2^(T−1)` ladder, the ≤ 64-chunks rung rule, 1/8 m shipped with 1/16 m reserved in the width,
`BodyId(u16)` in the store key, and the two linearisations confirmed deliberately (MEASURED: no grid,
chunk or block module exists in any crate today, so nothing is "already spent").

### 2.2 Format B — the saved cell record (bit by bit)

```text
BlockRecord — 12 bytes: word A (u64) then word B (u32), both MSB first.
PERSISTED. PERMANENT. ONE-WAY DOOR.

WORD A — where it is and what it is
  bits 63..46  cell         (18)  CellIndex inside the body's chunk
  bits 45..30  kind         (16)  BlockKindId — one row of the (substance, form, function) table
  bits 29..24  orient        (6)  bits 24..28 = one of 24 proper rotations; bit 29 MUST be zero.
                                  MUST be zero entirely on a Terrain-form cell
  bits 23..22  provenance    (2)  0 Terrain, 1 Feature, 2 Placed; 3 REJECTED
  bits 21..20  sub_scale     (2)  0 = whole cell (1 m); 1 = half; 2 = quarter; 3 = eighth
  bits 19..11  sub_addr      (9)  the sub-block's origin on the 8×8×8 lattice; MUST be zero at scale 0
  bits 10.. 0  reserved     (11)  MUST be zero; decode REJECTS non-zero

WORD B — what it looks like and what shape it takes
  bits 31..26  variant       (6)  the style index; a value >= the kind's variant count is REJECTED
  bits 25..18  density       (8)  i8, the RADIAL GAP r − h at the cell centre, in 1/128 cell,
                                  NEGATIVE inside solid; present on EVERY planet cell
  bits 17..10  object_param  (8)  a large landscape part's own shape parameter (a tree's structure
                                  class); MUST be zero unless the kind's form row carries OBJECT
  bits  9.. 0  reserved     (10)  MUST be zero; decode REJECTS non-zero

Store key:  (body, chunk).      Dedup key: (body, chunk, cell, sub_scale, sub_addr).
```

**What it reserves.** Eleven bits in word A and ten in word B — 21 in total. They may later buy a
`locked` bit for a claim system, a second style axis, or a finer sub-lattice (3 more address bits and
1 more scale bit, affordable in word A). A fourth provenance class is NOT there: provenance 3 is the
rejected value and that is the escape.

**Why twelve bytes and not eight.** Smooth terrain needs a density byte on every planet cell, and a
planted tree needs a shape parameter the server reads to build a collider. Neither fits in five reserved
bits. ESTIMATED: 100 M edits cost 1.2 GB of records instead of 800 MB; against a pyramid this document
sizes at 3.2–4.9 GB for the same edits, the extra 400 MB is about 8 %.

**The decode rule.** A decoder REJECTS, never defaults, on: a non-zero reserved bit; provenance 3; an
unknown kind; a variant ≥ the kind's count; a non-zero `orient` on a Terrain cell; an orientation outside
the shape's legal set; a non-zero `object_param` on a kind without the OBJECT flag; a sub-site off the
lattice or non-zero at scale 0; an `Empty` record with any other non-zero field. On the wire a bad record
tears the connection; in a store it refuses the store and never skips the record, because a skipped
record is a lost build.

**How a REMOVAL is written.** A cell the player returned to nothing is an `Empty` KIND record with
provenance `Placed`, every other field zero. It is a positive record, not an absence, because the client
derives the seed's shape and every removal must beat that derivation for all time. A dug tunnel, a
felled oak and a mined ore cell are the same encoding. On the wire the removal is one `ChunkRow` like
any other; there is no revert tag and no flag bit. **This is a format field and it belongs in the same
sitting** — the storage report's D-8 asks the same question as a decision
(`06_storage_diff_lane.md:890`) and it is booked as ★V101. *Example: a player cuts down an oak on a
moon. A week later her client re-derives that chunk from the seed. The `Empty` record on the oak's cell
keeps the oak down.*

**Where a SUB-METRE block sits on a smooth slope.** The `sub_addr` is a LATTICE OFFSET inside the cell
and never a surface-relative seat. A shared SEATING rule, inside the generator crate beside the
extractor, drops the block onto the derived-plus-diff surface inside its own cell, so both hosts compute
the same seat and NOTHING about the seat is stored. A stored seat would re-bind if the generator's
version ever moved the surface, which §2.4 forbids. *Example: a player sets a 25 cm lamp post on a
moon's hillside. The lamp's cell is a lattice address; the moon's shard and the client drop the lamp
onto the same slope by the same rule, so the boots and the picture agree.*

**What the record does NOT hold.** Mass and durability are per KIND, derived from the substance row.
Damage, heat, growth stage and a joint's motor angle are sparse side tables, so the permanent log is
never rewritten when a block warms up. An attachment is not a cell and is not this record.

**The attachment row** (a second table, ONE mechanism):

```text
AttachmentKey = (BlockAddr, face: u8 in 0..=5, slot: u8)     ← 6..=255 on face REFUSED
AttachmentRow (TLV, append-only tags)
  tag 1 kind (AttachmentKindId u16)   tag 2 params bytes   tag 3 placed_by AccountId   tag 4 fence
  tags 16..=31 RESERVED for signal bindings (a NAME and a SCOPE, never a resolved id)
  tags 32..=47 RESERVED for a joint's live motor state
```

**Its one-way doors, with deadlines.**

| Door | Deadline | Cost if wrong |
|---|---|---|
| The two words' field orders and widths; the twelve-byte fixed width | before the first WAL frame | a read-modify-write of every record in every saved world, an epoch bump, a fleet-wide refusal |
| The density's resolution, sign and sample point (`r − h`, 1/128 cell, cell centre, combined with the carver by a comparison, never `max`) | same | a change reinterprets every trench ever dug |
| The `object_param`'s meaning per OBJECT form | same | every planted tree grows the wrong shape |
| The numberings: provenance 0/1/2, the 24 rotation codes, the two linearisations | same | a renumber inverts the provenance privilege comparison |
| The attachment key's SHAPE — the block's FULL address plus a face plus a **slot byte from the first write** | with the address freeze | widening a KEY byte later costs a version floor on every persisted attachment in the world; the skip-unknown rule covers TAGS, never key bytes (`crates/core/src/tlv.rs:19-23`) |
| **The FACE numbering is the chunk's own basis — the same basis a block's orientation code uses** (`05_attachments_bodies.md:896`) | with the orientation encoding | a second numbering drifts from the block's on the first basis change, and every stored gauge is on the wrong face |
| **The REMOVAL encoding** — an `Empty` kind record with provenance `Placed`, every other field zero; no revert tag, no flag bit (★V101) | with the record | every tunnel and every stump in the world re-encodes; the client re-grows a felled forest |
| **The sub-site is a LATTICE OFFSET and the SEAT is DERIVED** by one shared rule inside the generator crate | with the record | a stored seat re-binds when the generator's version moves the surface; two invented seats put the boots off the picture |
| **The BOX record's 20-byte layout** — the compacted form of a run of cells (`04_block_record_registry.md:851`) | before the first compaction (slice 9) | a migration over every compacted chunk |
| **The realm store's FAMILY PREFIX BYTES** (`06_storage_diff_lane.md:872`) | before the first block is written | a wrong prefix byte silently changes which rows one `scan` returns |
| **The TREE SKELETON format** — a segment lattice in 1/1024 m, radii in 1/1024 m, canopy volumes — inside the generator tag; it is what `object_param` MEANS (`07_trees_composites.md:900`) | before the first client links the crate (slice 7) | a fleet-wide client refusal, and a changed collider under an existing treehouse |
| **The scoped channel key `H(scope ‖ name)`** for a signal binding; the TYPE is planted in the foundation (`05_attachments_bodies.md:900`) | before the first binding is persisted | every saved binding re-resolves to a different channel, blueprints included |
| **The integer ANGLE GRID a joint's angle counts on, and the carried sine and cosine table keyed by it** (`05_attachments_bodies.md:901`) | before the first joint angle is checkpointed | a re-valued grid changes every stored angle in the world; a table swap changes every body pose |
| Dense append-only numbering in all five registry tables, and the identity digest's COLUMN SET | before the first record | a column wrongly INSIDE the digest takes every store offline on a style change |
| Prune-on-equality over the FULL record | before the compactor | an unbounded store, or a laundering exploit |
| The registry digest equality at the handshake | before the first store is stamped | a client with a different registry meshes a plate as the wrong kind |
| Style as a per-cell 6-bit variant | before the first theme is authored | a 65th style is a new kind |

**Recommended answers.** Twelve bytes fixed; orientation 6 bits with the sixth zero-reserved; the
density byte on every planet cell; `object_param` in the record and the growth STAGE in the side table;
two typed registry tables under one mechanism, one digest over both; the variant count, the slot count
and the sub-scale mask deliberately OUTSIDE the digest, so shipping a new style never refuses a saved
world; and — **as the recommendation, changed by the law critic** — a PREFIX digest at the client handshake plus
a per-CHUNK refusal when a chunk names a kind the client does not hold. Equality at the handshake is the
alternative and its cost must be read against V2.7: every theme that appends one substance refuses every
unpatched pilot AT THE GATEWAY, which is the hardest seam the game has. *Example: an art update ships a
crystal spire for a themed moon. Under equality a pilot who has not patched cannot enter the galaxy at
all. Under the prefix rule she flies everywhere except that moon's chunks, and the refusal names the
kind she lacks.*

### 2.3 Format C — the edit-pyramid entry

```text
PyramidEntry — 8 bytes, one u64, MSB first
  bits 63..46  cell       (18)  CellIndex inside the tier-L chunk
  bits 45..30  substance  (16)  the dominant substance id; the EMPTY id when occ_mask == 0
  bits 29..22  occ_mask    (8)  octant o is set when child o holds anything
  bits 21..14  fill_256    (8)  mean fill of the eight children, 0..=255
  bits 13.. 0  reserved   (14)  MUST be zero; refused, never decoded to Default

Store key: (body, tier L, chunk).      Fold: mask = OR of non-empty children;
fill = rounded mean of the eight children; substance = the child substance with the largest summed
fill, ties to the lowest id.
```

**What it reserves.** Fourteen bits, and THREE candidate spends want them. They do not all fit, so they
are costed in one table and the owner picks which two survive. The entry is rebuildable on disk, so the
DISK half of this door is soft; the WIRE half is a protocol event and is hard.

| The spend | Bits | Who needs it | What it buys | What it costs to leave out |
|---|---|---|---|---|
| A signed SURFACE-HEIGHT delta for smooth terrain (★V19) | 8 | the renderer, at every coarse rung | a coarse rung whose silhouette matches the fine ground, so the arrival-pop tolerance can be met | the coarse rung draws a stepped fill instead of a surface |
| A sticky "something below me differs from the seed" bit (★V20) | 1 | the prune rule, against the quantisation loss | a 20 m quarry never vanishes at the 256 m rung | a fly-away seam, if the drawn step is above the drawable floor (UNMEASURED, U-33) |
| The CANOPY fold — per-coarse-cell canopy occupancy, mean height, mean colour (R-15) | ESTIMATED 8–12 | trees, which cannot ship without it | a far rung that draws the forest as it IS, after felling | the far view draws a resurrected forest, which is the arrival-pop seam |

Eight plus one plus eight is seventeen, against fourteen. **The owner must therefore either drop one
spend, or widen the entry to 12 bytes before the first world is saved, or carry the canopy fold in a
SECOND family keyed by the same `(body, tier, chunk)` key.** The third option keeps the 8-byte entry and
pays one more store family; it is the one this document would defend, and it is booked as ★V102.
*Example: a logger fells a hundred oaks around a landing pad. A pilot at four kilometres reads the
coarse rungs. Without a home for the canopy fold, the far view keeps drawing the forest that is gone.*

**Three rules that come with it.**

1. **Prune on equality with the GENERATOR's coarse answer** — not on "an authored cell exists below".
   The alternative draws a straight cliff along a coarse-cell boundary wherever an entry meets a
   generated neighbour, which is the detail-by-box seam.
2. **Exit on the EFFECTIVE summary.** The walk-up stops at the first rung whose new summary equals what
   a reader would get today — the stored entry, or the generator when no entry exists. The base's loop
   compared against the stored entry only, so an unedited parent never matched and every edit walked
   every rung.
3. **The entry is a DELTA for terrain, never an absolute.** At rung L an unedited cell's sample comes
   from the generator with L octaves dropped; an absolute summary of eight rung-0 samples comes from the
   FINE world. Laying one over the other makes a step at the edited cell's rim, up to four metres at
   rung 2. A delta adds onto whatever base the rung supplies, so the rim is flat by construction.

**The hard requirement this puts on the generator.** The coarse answer must EQUAL the fold of the
generator's own fine cells under the same address. The cheap way to coarsen fractal terrain — drop
octaves — is NOT the fold, and if the generator returns the octave-dropped answer nothing ever prunes.
The naive exact fold is not affordable either: one rung-13 address covers 8¹³ = 549 755 813 888 tier-0
cells, and at the MEASURED 12.19 ns per noise evaluation (`scripts/noisebench`, Apple M4 Pro) that is
ESTIMATED 1.9 hours for ONE coarse cell. **This is the single largest unresolved requirement in the
foundation and it is booked as SL6 request R-7 and measurement U-16.**

**The quantisation bound.** `fill_256` states volume in units of 1/255 of a coarse cell, so an edit
under half a unit rounds away and rule 1 DELETES the entry. A 20 m quarry (8 000 m³) survives to the
128 m rung (fill 254) and vanishes at the 256 m rung (fill 255) — **ESTIMATED**: it is arithmetic on the
fold rule, and this document's own rule says arithmetic on a cited number is ESTIMATED, never MEASURED.
No program has run it. Whether the resulting step is under the drawable floor (1 px at 45° over 720 rows,
`crates/core/src/geometry.rs:1156-1173`) is UNMEASURED and is what the fly-away gate owes.

**Its one-way doors, with deadlines.**

| Door | Hard or soft | Deadline | Cost if wrong |
|---|---|---|---|
| The `CellIndex` width and the substance width (shared with the record) | **hard** | before the first world is saved | a migration over every saved planet, station and hull |
| The fold rule AND the exact coarse `generate_summary` in the generator crate | **hard on the wire, soft on disk** | before the first world is saved | a protocol event and a world-generation tag change; and, if the coarse call is not the fold, no store is ever sparse |
| A surface-height datum for smooth terrain | decide before the first world | before the first world is saved | the RENDER contract on the wire changes |
| A sticky "differs from the seed" bit | decide before the first world | same | it changes the layout's meaning and the prune rule together |
| **Where the CANOPY fold lives** — in the entry's reserved bits, in a widened entry, or in a SECOND family under the same key (★V102) | **hard on the wire** | before the first coarse rung ships | three spends want seventeen of fourteen bits. If the fold has no home, the far rung draws a resurrected forest, which is the arrival-pop seam, and curing it later is a protocol event |
| The 8-byte layout itself | **soft** | — | a rebuild pass per realm plus a protocol-minor bump. The entry is DERIVED and rebuildable from tier 0; the base's "PERSISTED, PERMANENT" is too strong |
| `pyramid_storage_block_cells` | hard | before the first pyramid block | a persisted-key migration |

**Recommended answers.** Keep 8 bytes and the fourteen reserved bits, and put the canopy fold in a
SECOND family under the same key (★V102) so the entry never has to widen; store the pyramid at the
CHECKPOINT under its own watermark with a bounded tail replay, not inside the WAL's transaction. **The
real ground for that answer is that the pyramid is DERIVED and rebuildable from tier 0, so it does not
need the write-ahead log's crash guarantee.** The 5 508× write amplification the base quotes for the WAL
shape is **ESTIMATED and inherited** (`storage_and_streaming.md:1057-1086`); its own source report says
in its own words that *"every number in the list is UNMEASURED on this codebase"*
(`06_storage_diff_lane.md:396`). U-26 owes the run; write only from a `Tier0Key`; and decide the surface-height datum and the sticky bit
together with the renderer, before the first world is saved.

### 2.4 Format D — the generator's WORLD IDENTITY (the fourth freeze)

The owner reads §2 as the list of things that must never move. The item with the largest blast radius in
the whole investigation belongs in that list, and revision 1 left it in the register as V21. It is a
FORMAT: not a byte layout, but the exact function that turns `(seed, address)` into ground, which every
saved byte of Format B and Format C is a delta against.

```text
WorldIdentity — the frozen contents of the generator tag
  the crate's version                     the arithmetic profile (target, opt level, the Gf operation set)
  the noise source and its position hash  the octave table (count, lacunarity, gain, per-octave seeds)
  the cave lattice step                   the warp constants k1,k2,k3, the first guess, the step count
  the strata and biome tables             the seed-placed feature anchors and the tree SKELETON format
  THE EXTRACTOR's rule                    THE TRIANGULATION rule (which diagonal splits a quad)
  THE COMPOSITION ORDER                   THE SEATING rule for a sub-metre block
  the exact coarse summary (the fold)     the ONE owner of a body's radius
```

**Its door.** It shuts when the golden literals are pinned, which is when the first world any player
builds on is saved. After that the only lawful change is APPEND-ONLY: a new octave may add detail only
below a stated tolerance, and nothing may move a cell's surface by more than that tolerance. There is no
re-roll. The retrofit cost is THE WHOLE EDIT CORPUS.

**Why the triangulation rule is inside it.** A quad on a saddle splits into two triangles two ways, and
the two ways differ by up to half a cell in the middle. SL10 V1.5 says what the player sees is what the
player stands on, so the diagonal is not a renderer's choice. *Example: a prospector stands on a smooth
ridge. The client split the ridge quad one way and drew a crest; the moon's shard split it the other way
and collided a hollow. The boots sink into the crest the player can see.*

**Why the seating rule is inside it.** Both hosts must drop a 25 cm lamp post onto the same slope, and
neither may invent its own rule. SL10 V1.2 forbids a second implementation by name.

**What is DERIVED and therefore OUTSIDE it.** The pyramid entry's bytes (rebuildable from tier 0), the
client's decorative trim (slice 7), and the drawn material colour.

**Its one-way doors, with deadlines.**

| Door | Deadline | Cost if wrong |
|---|---|---|
| The whole identity above, pinned by the golden literals | before the first world any player builds on is saved | the whole edit corpus: one added octave moves the ground under every placed block and every saved diff |
| The APPEND-ONLY octave rule and its stated tolerance in metres | with the identity | with no stated tolerance no later change is lawful at all, so the world can never be improved |
| The TRIANGULATION rule | with the extractor, before the first collider | the drawn surface and the collided surface disagree by up to half a cell on every saddle |
| The SEATING rule for a sub-metre block | with the record | a stored seat, which re-binds; or two invented seats, which put the boots off the picture |
| The ONE owner of a body's radius | with the body definition | a ladder step depends on an ulp of a `powf` on two targets (`crates/physics/src/taxonomy.rs:616`) |

**Recommended answer.** Freeze it as one document beside Formats A, B and C, in the same sitting, and
state the append-only tolerance in metres. *Example: a player builds a landing pad on a hill on a moon.
Six months later the generator gains one octave. Under the append-only rule the hill moves less than the
stated tolerance and the pad still rests on it; without the rule the hill is 40 cm higher and the pad
floats.*

---

## 3. The implementation sequence

Each slice names what it lands, which laws it exercises, its GATE (a test that can go red) and its
MEASUREMENT (a number with a pass threshold). **R** marks a renderer-agnostic slice; **S** marks a slice
that touches the render seam, so a later client-engine investigation must see it.

**Where the sequence starts, in code.** Nothing in this domain exists. MEASURED 2026-09-07:
`grep -rn "GridMapping\|ChunkKey\|CellIndex\|Tier0Key\|BlockKindId\|PyramidEntry" crates --include='*.rs'`
returns one doc comment (`crates/io-prod/src/store.rs:6`). `FrameSpace`, `reanchor` and `AnchorGen`
appear in six doc comments in three files and in one unrelated test name
(`crates/core/src/fence.rs:18`; `crates/sim/src/capability.rs:11,18,35,40`;
`crates/sim/src/stub/transient.rs:239`; `crates/node/src/saga_runtime/tests.rs:3513`). `rapier3d` and
`noise` are in NO `Cargo.lock` entry (MEASURED: `grep -c` returns 0 for each), while `noise = "=0.9.0"`
is declared and unused (`Cargo.toml:78`). `BulkKind::{ChunkSnapshot, ChunkDelta, Catalog}` is declared
and unroutable (`crates/wire/src/channels.rs:288-296`), and `MsgClass` has no bulk arm
(`crates/sim/src/io/mod.rs:51-85`). `InterShardFlow` has **44 arms** (MEASURED by `awk` over the enum,
`crates/wire/src/intershard.rs:124`), so the board's "27 + 3 = 30, no headroom" is stale by seventeen.

**The board's ordering survives, with one addition.** The decision board puts the persistence seam, the
geometry seam, the registry and the wire plant before any generator code. No revised report refutes it,
and two strengthen it: the geometry seam is terrain's FIRST slice by the owner's word of 2026-09-05
(`docs/design/DEFERRED.md:309-311`), and the persistence seam's `Store` trait has five methods with no
point read, no bounded range and no per-realm handle (`crates/sim/src/io/mod.rs:477-495`), through which
two of this design's own gates are unexpressible. The addition: **slice 0 is a decision sitting, not
code**, because Formats A, B, C and D read one another's fields.

**Two rules bind every slice below.**

1. **No store that must survive is written before the last slice that changes the seed's shape.** A tree
   is a record on a cell, so a chunk digest taken at slice 5 differs from the same chunk's digest at
   slice 14. The fail-safe on a stamp mismatch DISCARDS the file
   (`crates/core/src/store_stamp.rs:182-192`). Therefore the stores of slices 9 to 13 are THROW-AWAY
   until slice 14 lands and the world identity is pinned with the feature anchors in it, OR the
   seed-placed feature anchors move into slice 5. *Example: a tester digs a tunnel on a moon at slice 10.
   Slice 14 plants the moon's forests, the chunk hashes differently, and the tester's store is refused
   and discarded. That is correct while the world is being built and intolerable after it ships.*
2. **HR4 binds every feature slice, not only two.** *"Every feature passes the identical fixture on
   ≥ 2 shard kinds (G-IDENTICAL) or it doesn't land"* (`CLAUDE.md:110-111`). The profiles exist as data
   today: `crates/sim/src/capability.rs:246` gives the planet `VoxelGeometry::Spherical`, and `:262`,
   `:278` and `:290` give the ship, the station and the asteroid `VoxelGeometry::Cartesian`. So every
   feature slice below names its PAIR and its identical fixture body. *Example: a miner digs the same
   trench with the same tool in two places — on a moon, and in a station's soil bay. One fixture, two
   profiles, two stores compared byte for byte.*

---

**Slice 0 — THE FORMAT SITTING. (R, no code.)**
*Lands:* TWO sets of answers, both written into `docs/design/`. **(a)** every row of §4 band A. **(b)**
every row of §5 — the SL6 requests — answered YES or NO by name.
*Laws:* SL5 (one world), SL6 (**the default is NO, so an unanswered request may not be built**), SL10
(the address is the generator's input), V3.2.
*Gate:* the FOUR format documents exist (A the address, B the record, C the pyramid entry, D the world
identity) and every field in §2 has an owner-stated value; every §5 row carries a written YES or NO; and
the pin tests of slices 3–5 can be WRITTEN against them.
*Measurement:* none. This slice produces answers, not numbers.
*Why (b) is here:* slice 4 plants seven items that §5 marks refused-by-default — the block-edit forward
and its ack, the Bulk class with its chunk rows, the rung floor on a window, the world action, the
client's world handshake and `TAG_SURFACE`. Without the owner's word, slice 4 builds a wire plant whose
every item the law refuses. *Example: a pilot inside a hull berthed on a moon breaks a rock beside the
ramp. That edit rides `InterShardFlow::BlockEdit` from the hull's shard to the moon's shard
(`crates/wire/src/intershard.rs:124`). That arm is R-1, and nobody has asked whether it may exist.*

**Slice 0b — Split the sim shard file. (R.) — CLOSED: ALREADY LANDED 2026-08-21 in commit 40b5ce2; the 15 550-line figure was the board's stale number, measured false on 2026-09-07 (26 modules, largest 2 182 lines).**
*Lands:* a mechanical decomposition along capability lines.
*Laws:* HR3, HR5.
*Gate:* coverage unchanged; the demand-loop end-to-end runs green; the engine arity workarounds are gone.
*Measurement:* Tier-A region and branch coverage stays at 100 %.
*Why here:* it is the only slice blocked on nothing, and it is the only one that gets strictly more
expensive with every other slice, because most of the sim work below lands in that file.

**Slice 1 — Widen the persistence seam. (R.) — LANDED 2026-09-07: `Store::get` (the point read) and `Store::range` (the bounded half-open range read, ascending, limited, inverted = empty) on the seam, implemented on `MemStore` and `RedbStore`, the redb-vs-mem parity test extended to both reads across every window and a reopen; the per-realm handle already existed (`RealmStore` resource). MEASURED: 643 sim unit tests pass; Tier-A gate PASS 100 %; io-prod regions 94.99 % and 95.17 % (floor 94); the sim suite's median wall time 115.75 s after vs 115.76 s before (three runs each, idle machine).**
*Lands:* a point read, a bounded range scan or cursor, per-realm store handles, and the memory twin for
all of them, on `crates/sim/src/io/mod.rs`'s `Store`.
*Laws:* the no-I/O-outside-the-seam convention; HR5.
*Gate:* the twin and the redb backend agree on every new method under the existing property tests, and
every block-store fixture below runs ENTIRELY on the twin — no real file.
*Measurement:* the "fast deterministic suite, virtual clock, no sockets" property holds — three runs
before and three runs after, on an idle machine, and the median wall time grows by **less than 5 %**.
A bare "does not grow" cannot go red, because a suite's wall time always moves.

**Slice 2 — The geometry seam. (R.) — LANDED 2026-09-07: `crates/core/src/grid/` (the address, the face
bend, the compile-time seam table, the flat arm, the round arm, the seam value) and the sim's
`grid_for` over a typed body extent; refuted by an Opus 5 pass (twenty findings, all answered — see
`verdicts/slice_02_refutation.md` and `slice_02_geometry_seam.md` §12a for the measured results).
Deviations from the plan below, stated: `warp.rs` is `bend.rs`; the grid parameters are derived by the
sim's `grid_for` from a typed body extent with NO caller yet (the first is slice 9); the crust and the
above-surface band are a PROVISIONAL derivation until slice 5; the round-versus-lump threshold of
ruling V5 is the body definition's (slice 5).**
*Lands:* `crates/core/src/grid/{mod,addr,warp,seam,identity,shell}.rs` — `CellAddr`, `GridMapping` with
two arms and six operations, the warp with `const` constants, the `const` first guess and step count, the
generated seam table with its pinned literal; `GridParams` DERIVED at each use from the realm's seed and
`bound`, so **no record and no store changes in this slice**; one line in `build_app`.
*Laws:* HR3 (a match on a geometry VALUE, never on a shard kind), HR4, SL1 (the address names no
position outside the realm), SL10 V1.4.
*Gate:* **G-MAPPING-TABLE** at `N = 62`, exhaustive: `neighbor` involutive on every seam cell; the
four-step lateral loop closes everywhere except the eight corners where it closes in three; every cell
has four lateral neighbours; the apron is watertight at 24 strips and 24 slots; the seam table is
byte-identical across tiers. **G-MAPPING-ROUNDTRIP** at `N = 2²⁶`, sampled at the worst `a ≈ ±0.85`:
`addr_of(cell_center(x)) == x`, and `addr_of` is `None` outside the domain. The split is by
`N`-DEPENDENCE: a run at `N = 62` passes a coarse inverse by a factor of 1.5 million and cannot catch it.
*Measurement:* the inverse warp's residual in CELLS at `N = 2²⁶`, from the frozen guess and count, run
inside the crate in Rust — pass under 10⁻⁶ cells; and `addr_of` microseconds per call, to show it is O(1)
and off the containment path (`crates/core/src/geometry.rs:1442` reads a `Boundary`, never a cell).

**Slice 3 — The registry and the shape catalogue. (R.) — LANDED 2026-09-07, see
`slice_03_registry_catalogue.md` §12.**
*Lands:* the substance, form, function, block-kind and attachment-kind tables as `const` arrays whose
index is the id, each row with an identity KEY beside a display NAME, a `def()` that refuses an unknown
number; no theme column (ruling V7 dropped themes); the identity prefix digest over the keys and the
form geometry only; the twenty-shape catalogue declared with everything but the cube unplaceable; the
24-rotation integer table; mass and integrity derived per kind from cited facts.
*Laws:* HR3, HR5, SL5, the no-magic-numbers rule.
*Gate:* the drift tripwire for all five tables, including the round-trip assertion that a hand-edited
row fails the build; and **the digest-column test**: append a kind and raise a variant count from 3 to 4,
assert the stored-length prefix digest is UNCHANGED in both cases; then change one row's triple and
assert the store refuses, naming both digests.
*Measurement:* the digest's cost at open and at handshake, in microseconds over the identity columns
(ESTIMATED ~23 KiB) — **pass under 200 µs at open and under 200 µs at the handshake**, because a store
open and a client handshake each already carry a budget in milliseconds. A number, never the word
"negligible".

**Slice 4 — The wire plant. (R.)**
*Lands, while it is still free:* `BLOCK_STORE_FLUSH_STEP = 19` and `BLOCK_EDIT_FORWARD_STEP = 20` in ONE
file with the disjointness test (ids 7..18 are taken, `crates/wire/src/intershard.rs:65-115`); the
`BlockEdit(BlockEditForward)` and `BlockEditAck` arms; `MsgClass::Bulk` (reliable, paced);
`BulkMsg::ChunkRows` and `ChunkManifest` as appended variants with the `ChunkRow` bag schema and its
tags; `ShardToGateway::BulkFor`; `rung_floor: u8` on `WindowOpen` AND on the relay's downward request;
`ClientControlMsg::WorldAction` with `BlockEdit` and a reserved `Fire`;
`ClientControlMsg::HelloWorld { declared, measured }`; `TAG_SURFACE` beside `TAG_LOOK`
(`crates/core/src/look.rs:31`); `ChannelKey` in `vd-core`; the block store's schema ids and the
`last_owner_fence` row; `BlockStoreTuning` with every operational number named.
*Laws:* SL6 (each item is a request in §5, default NO), HR1 (one reviewed file), the incremental-freeze
rule.
*What it may plant:* **ONLY the §5 rows slice 0 recorded as YES.** A planted arm with no recorded
approval is a defect the closed-set test names by its R-number. Rows the owner defers arrive later
behind the skip-unknown rule where that rule applies — it covers TLV TAGS and never key bytes or enum
arms (`crates/core/src/tlv.rs:19-23`), so a deferred ARM costs a second wire slice and a deferred TAG
costs nothing.
*Gate:* the closed-set gate — `every_arm`, `arm_tripwire` and the durability golden pin
(`crates/wire/tests/intershard_closed.rs`) — green with the new arms represented in the same commit; and
`no_realm_inbound_payload_carries_a_placement_or_a_centre`
(`crates/wire/tests/intershard_closed.rs:485`) still green.
*Measurement:* the `TAG_SURFACE` bytes against the 1 200-byte datagram budget, stated on change and
never on a keep-alive — pass under the budget with the reach datum's own margin.

**Slice 5 — The generator crate `vd-terrain`, with the fence. (R.)**
*Lands:* one crate depending on a 150-line `vd-seed` leaf (the integer hash of `crates/core/src/rng.rs`
plus the digest of `crates/core/src/digest.rs`) and nothing else; the `Gf(f64)` newtype with a PRIVATE
field, no `From`/`Deref`, and exactly `+ − × ÷ Neg sqrt floor trunc abs from_i64 to_i64`; a crate-scoped
`clippy.toml` banning every transcendental, `mul_add`, `f64::min` and `f64::max`; the body definition
(and it becomes the ONE owner of a body's radius, which the forest then reads); the height stack with
octave dropping; the density output; the strata; the carvers; the exact coarse summary; the chunk digest;
the golden self-check.
*Laws:* SL10 V1.2/V1.3/V1.4, SL1 clause 5 (a structural fence with an OBSERVED-FAILING control, never
care), SL5, HR5.
*Its crates' TIER:* `vd-terrain` and `vd-seed` are **Tier-A at 100 % region and branch**. The `Gf`
newtype wraps every arithmetic operation the generator uses, which is exactly the shape HR5's
monomorphization discipline is about (`CLAUDE.md:112-124`). `just coverage-fast` is part of this gate,
or the exemption is written into `coverage-exemptions.toml` with its reason.
*Gate:* three layers, each with a control that must be seen failing. **Type:** a compile-fail case that
reaches for `x.0` and `x.sin()`. **Lint:** inject `x.sin()`, run `just lint`, expect red. **Link:**
`nm -u` over the built staticlib finds no `sin|cos|tan|exp|pow|log|cbrt|fma` symbol — this is the only
layer no inlining can hide from, and it catches a transcendental that arrived through a dependency
(`glam 0.30.10` depends on `libm`). Plus the no-valuable-substance test.
*Measurement:* **the no-drift gate.** ~832 chunk digests (ESTIMATED: 64 keys × 13 rungs) equal byte for
byte between a debug and a release build, with and without `-C target-cpu=native`, on
`aarch64-apple-darwin` and on `x86_64-unknown-linux-gnu`, server build against client build. Red on one
differing byte. **No leg of this gate has ever run**; the x86-64 leg has no host in this project, and an
emulated green is not an x86-64 green. Plus the per-chunk cost re-run under the mandated hash.
**The 1.20 ms threshold is INHERITED and must be re-derived here.** The base's 0.885 ms
(`block_system_design.md:17461-17475`) measured a BLOCKY field with no extractor, and §8 item 27 refuses
that bench's hash composition. The gap field plus slice 6's extractor is a heavier job. So slice 5
states the threshold from the shard's own tick budget: the chunk build runs on a worker pool, and the
pass number is **the wall time at which one worker cannot keep the frontier ahead of a hull at 250 m/s**
— ESTIMATED 70–120 surface chunks per second over strong relief (`09_physics_controller.md:661-665`),
which puts the per-chunk budget at 8–14 ms of worker time and the tick thread's own share at zero.
*Example: a pilot dives at a mesa. The moon's shard builds the mesa's chunks a hundred metres ahead of
the nose on a worker thread, and the tick thread only inserts finished ones.*
*Its place in the shape freeze:* the golden literals this slice pins are PROVISIONAL until slice 14
plants the seed-placed feature anchors, unless those anchors land here. Nothing that must survive is
saved against a provisional digest (§3's rule 1).

**Slice 6 — The surface extractor and the composition order, INSIDE the crate. (R.)**
*Lands:* naive surface nets over cell-centred density samples — one vertex per 2×2×2 group with a sign
change, at the mean of the edge crossings, in a fixed corner order; normals from the gradient; **the ONE
TRIANGULATION rule that says which diagonal splits a quad**; **the SEATING rule that drops a sub-metre
block onto the derived-plus-diff surface inside its own cell**; the FIXED five-step composition order
(generated shape → terrain cell edits → catalogue blocks → sub-metre blocks → attachments); the tree
expansion function. The triangulation rule and the seating rule live HERE, beside the extractor, because
both hosts must compute them identically and Format D freezes them (SL10 V1.2).
*Laws:* SL10 V1.2 (a port is forbidden, so the rule that makes the ground must be ONE rule), V1.5 (the
server collides on the same shape), V1.6 (the diff composes before either host draws or collides).
*Gate:* **a pinned TRIANGLE list for a fixture chunk** — a quad list cannot go red on the defect that
matters, because a saddle quad splits two ways and the two ways differ by up to half a cell in the
middle. Plus the golden set gains a COMPOSED row — the generated shape plus one mined cell plus one
placed block plus one sub-metre block, digested AFTER composition and AFTER extraction. Without that row
the byte gate is green while two hosts stand on two slopes. Plus a SEATING row: one 1/8 m post on a 30°
slope, its seat digested on both builds.
*Measurement:* extraction milliseconds, vertices, quads and bytes per chunk, typical and worst (a 3-D
checkerboard gap field) — ESTIMATED 0.3–1.5 ms, UNMEASURED. **Pass under 4 ms of worker time per tier-0
chunk in the worst case**, which is the share slice 5's re-derived budget leaves the extractor. It also
gates the client's residency lead.

**Slice 7 — The client links the crate; the render seam. (S.)**
*Lands:* **`vd-terrain` added to `[dependencies]` in `crates/client/Cargo.toml`. `vd-physics` STAYS a
dev-dependency**, and the crate-isolation test gains a row that refuses a client dependency on it.
Revision 1 said "`vd-physics`'s successor", which is wrong twice: the file says in its own words
*"DEV-ONLY: scene tests build THE world; the shipped client never names a motion (SL4)"*
(`crates/client/Cargo.toml:20-22`), and `vd-physics` IS a motion crate
(`crates/physics/src/motion.rs`, MEASURED by `ls`). `vd-terrain` is a NEW crate over a `vd-seed` leaf
(slice 5), and `vd-physics` keeps the forest and READS the body radius from it. *Example: a pilot's
client draws a moon's hills from the seed. The same binary must not hold the code that moves the moon.*
Plus `ChunkGeometry` with cell-space integer vertices at a named
`vertex_quantum`, packed attributes with a provenance bit, indices, a part-transform table, and a header
with the tier and the skirt; `chunk_tier_for`, `chunk_request`, `chunk_poll`, `chunk_release`,
`chunk_changed`, `chunk_cells`, `realm_shape_params`; the worker-pool seam as an injected trait with an
inline implementation for Tier-A tests and a threaded one in the binary; the C mirror.
*The seam's SPLIT, stated before either encoder is written:* the LIBRARY owns which chunk is built next
(the coarse-before-fine ORDER), the TIER per chunk, and the crossfade BAND with its blend weight per
chunk. The ENGINE owns the thread that runs the work and the shader that paints the blend. Without this
split a client on another engine writes its own order and SL8's continuity forks with the engine, which is the
opposite of V2.8. *Example: a hull descends onto a moon. On the Bevy client the coarse ground arrives
first and the fine cells dissolve in over four frames. On a client on another engine whose plugin owns the queue,
fine cells arrive first and the ground appears in patches. The same seam gate passes on one client and
fails on the other, and nobody can tell which is the world's fault.*
*The TRIM rule, stated here because slice 7 defines what the seam hands out:* client-derived decoration
(V2.6, V72) NEVER changes the extractor's output, never changes a collider, and never moves a vertex the
collision mesh shares. It may only ADD geometry that no collider reads. *Example: a shipwright plates a
hull and the client bevels every welded seam. The moon's shard collides the plain plates. The bevel must
never lift a crewmate's boots off the spine she can see.*
*Laws:* V2.8 (nothing assumes Bevy), HR5 (`vd-client` is Tier-A at 100 %, and it spawns no thread today
— MEASURED: `grep -rn 'std::thread\|rayon' crates/client/src/` returns nothing), SL10 V1.7, SL1 clause 5.
*Gate:* the Bevy renderer builds a mesh from `ChunkGeometry` with NO shape branch, as it does from
`MeshPrim` today (`crates/client-render/src/lib.rs:1828-1846`); a structural control that FAILS when the
client-linked crate's surface grows a placement or an orbit symbol — SL1 clause 4 forbids folding an
absolute from the root, and the crate the client links must expose SHAPE AT AN ADDRESS only; a
crate-isolation row that FAILS when `vd-physics` becomes a client dependency; **an assertion that this
slice draws ONE rung inside one chunk band and never a second one**, because the tier rule and the
crossfade land in slice 8 and a hard edge between two rungs must not reach a picture before its detector
does; and a geometry-against-collider comparison that FAILS when derived trim moves a shared vertex.
*Its crates' TIER:* `vd-client` stays Tier-A at 100 % region and branch, so `just coverage-fast` is part
of this gate.
*Measurement:* the binary size and cold-build cost of the link on both shipped targets — **pass under
8 MB added to the shipped client and under 60 s added to a cold build**, the two numbers a shipped
installer and a working day can absorb; the client's generation budget — chunks per second on ONE core
and on eight, at 1.4 m/s, at 240 m/s and at 528 m/s, **pass when one core keeps the resident band whole
at 1.4 m/s and eight cores keep it whole at 528 m/s** (MEASURED precedent: a 6 800-chunk 100 km view is
~6 s on one core and 0.75 s on eight); and the C-ABI copy cost per chunk against the ESTIMATED ~715 KB
packed payload (`03_generator_sl10.md:659`), which is what another engine's plugin pays per chunk.

**Slice 8 — The detail ladder and the crossfade on the client. (S.)**
*Lands, ALL of it in the client LIBRARY (slice 7's split):* the tier per chunk column from the one
angular rule with ONE reference view; coarse-before-fine ordering; the dither crossfade band and its
blend weight; the rule that the client derives ONLY for a realm that holds a row in the composed scene
(`crates/wire/src/channels.rs:158,179`); the realm's own look as the rung ABOVE the coarsest generated
rung, drawn from the same crate at the body definition's radius; **and the residency band, sized from
SERVER-STATED data only** — the realm's own reach, the interpolation buffer
(`crates/wire/src/channels.rs:328`) and a lead the OWNING shard states beside them. The client must not
subtract two delivered rows and call the difference a closing speed: SL10 V1.7 says *"the client never
derives a pose, a velocity, or any state of any entity or realm"*, and the spread of a delivered track
over time IS a velocity. If no delivered datum is enough, R-18 is a real ask and goes to the owner as
one. *Example: a hull dives at a moon at its rated cruise. The moon states how much ground to warm; the
client reads that number and builds. It never works the speed out for itself.*
*Laws:* SL8 (detail-by-box, arrival pop, brightness pop, tick hitch), the 2026-09-01 visibility ruling
(a dormant realm is never drawn by anybody), SL3.
*Gate:* the pop detector on a HULL flown at 1.4 m/s, at 240 m/s and at 528 m/s — no pixel on a tier
boundary changes by more than the dither's own noise between two consecutive frames. **The legs are
flown by a hull, never by a walker.** The suit ruling binds what a gate may do — *"The gates fly the
shipped path: berth a hull, board it, push"* and *"never a walking dot"*
(`docs/design/owner_decisions_2026-09-05_suit.md` S4) — and no character controller exists until slice
16 (U-48). The 1.4 m/s leg is the walking-speed REGIME flown by the shipped path available now; slice 16
re-runs it under the real character. **The rung-disagreement measurement runs BEFORE this slice builds
the ladder**, because at `k_rough = 0.5` the bound subtends ESTIMATED 4 pixels at the switch distance,
independent of the rung, and 4 pixels is visible.
*Measurement:* the maximum and 99th-percentile vertical disagreement between rung L and rung L+1, in
pixels at the switch distance — **pass at p99 ≤ 1 pixel and max ≤ 2 pixels**, from the drawable floor
(one pixel at 45° over 720 rows, `crates/core/src/geometry.rs:1156-1173`); and the chunk ARRIVAL RATE
during a scripted descent against a NAMED thread budget — **pass when the resident band is never
incomplete for one frame**, with the deepest queue reported.

**Slice 9 — The block store, the WAL and the pyramid. (R.)**
*Lands:* the `block_wal`, `chunk_delta`, `chunk_pyramid` and `chunk_meta` families in the realm's own
store (`StoreRole::RealmStore`, `crates/core/src/store_stamp.rs:64-75`, which holds berths and the body
today); the walk-up in memory at commit with the effective-summary exit; the checkpoint with the
watermark and the three-step discipline; the `last_owner_fence` refusal; the type-level owned handle.
*Laws:* HR1 (a realm's store is private and travels with the realm), the fence discipline, SL9.
*Gate:* **G-IDENTICAL, on a PLANET-profile shard and a STATION-profile shard**
(`crates/sim/src/capability.rs:246,278`): the identical fixture writes the same 10 000 edits into each
realm's own store and the two stores compare row for row under the same key order. Plus a `kill -9`
during an edit storm replays the pyramid tail and the result is byte-identical to a `verify-pyramid`
rebuild; and the store REFUSES to open when `last_owner_fence` is not strictly less than the opener's.
*The store's status:* throw-away until slice 14 pins the world identity (§3's rule 1).
*Measurement:* store size for 100 M scattered edits (the case that happens — ESTIMATED 6.1 entries per
edit and 4.9 GB, not the base's 1.1 and 0.9 GB); write amplification per entry; and the walk-up cost per
edit at 13 rungs, which exists only if the coarse summary costs a handful of evaluations.

**Slice 10 — Mining and placing on terrain; the diff lane. (R + S at the apply.)**
*Lands:* the edit path with reach, occupancy, rate and nonce validation; the two-stage receipt that
refuses the loser when two players edit one cell in one tick; the diff rows on the new Bulk class with
the HIERARCHICAL manifest; the per-chunk baseline on subscribe; the gap under a placed square block; the
mining yield; the cell-change hook that deletes or re-validates a cell's attachment rows in the SAME
transaction.
**And the PLACING half of V2.1, which revision 1 left with no deliverable.** V2.1 asks that *"when we
place voxels, the terrain should change accordingly"*. So this slice states three things: a placed
TERRAIN-form cell carries a FULL density (the most negative legal gap), so it reads as solid rock; a
player MAY raise ground above the seed's surface, cell by cell, up to the per-cell record cap; and V67
(no natural substance placed as a SQUARE block in the first slice) refuses only the square lane, never
the terrain lane. *Example: a builder fills a ditch beside her landing pad with dirt. Each filled cell is
a terrain-form record at full density, the smooth extractor rounds the fill into the slope, and the
moon's shard collides the new ground on the same tick it draws.*
*Laws:* SL10 V1.6, SL2 (the connection plane is not a realm), SL6 (**this slice consumes R-3, R-5 and
R-13; it may not ship before slice 0 records the owner's YES on each**), SL7 (a realm ships to the
observers inside its own reach), SL8 (lane flood, re-state rate, the proposed diff lag), HR4.
*Gate:* **G-IDENTICAL on a PLANET-profile shard and a SHIP-profile shard** — the identical fixture digs
the same trench and fills it, in a moon's hillside and in a hull's ballast hold, and the two stores
compare byte for byte. This gate is why ★V38 (may a Cartesian realm hold terrain-form cells?) moves into
band A: if the answer is NO, the smooth lane can never run on two kinds and this slice cannot satisfy
HR4 at all. Plus the fly-away-and-back test over a dug tunnel, a filled quarry and a built tower — each
edit stays visible up to its stated quantisation rung, and **no RIM STEP appears around a coarse-rung
dimple**; plus the crossing case (a player digs while the subscription flips) — a client that
re-subscribes receives the chunk's whole diff set BEFORE any incremental row.
*Measurement:* **the CLIENT's re-extract of one edited chunk, pass ≤ 16 ms of worker time**, which is
one frame at 60 Hz and is the number this slice controls. **No whole-path click-to-pixel threshold is
stated here.** Revision 1 quoted 33 ms; that figure is a BUDGET for remeshing CUBE faces
(`09_physics_controller.md:589`), the network half of the path alone is ESTIMATED ~101 ms, and this
document's own U-54 forbids quoting a click-to-pixel figure before it runs. Plus diff bytes per second
under live digging; the STANDING bytes a client holds and receives at first sight of a long-dug moon,
swept across four orders of magnitude of edit count; and the drawn step at every rung boundary against
the drawable floor.

**Slice 10b — The realm re-states what a build changed. (R.)**
*Lands:* the derivation of a realm's MASS, its CROSS-SECTION, its DRAG COEFFICIENT and its LOOK RADIUS
from its own block field, plus the COALESCING rule that states them upward on a SETTLE and never per
placement.
*Laws:* the movement contract — those facts cross up *"ON CHANGE ONLY — when the hull is rebuilt, never
per tick"* (`docs/design/owner_decisions_2026-08-26_movement.md:42`); the reach ruling — *"a realm's
reach comes from its own LOOK, not only its size … the realm states its own reach"*
(`docs/design/owner_decisions_2026-09-02_reach.md:79-81`, the lane at `:135`); SL3 (the realm states its
own look); SL8 (sprite cull); HR4.
*Why it exists:* every block a player places changes all four numbers, and revision 1 derived none of
them, re-stated none of them, and said nothing about a builder who places ten blocks a second — which
turns an "on change" lane into a per-tick lane the contract refuses by name. *Example: an engineer welds
a 500-metre mast onto a station. The station's mass grows, its cross-section grows, its look grows, and
so its reach grows. Until the station re-states its reach, its parent planet keeps testing the old
number, and a pilot who should already see the mast sees nothing. That is the sprite-cull seam, caused
by a build.*
*The coalescing rule:* a placement marks the realm dirty; the realm re-derives and states at most ONCE
per settle window, and the window is a field of the realm's own tuning record, never a constant.
*Gate:* **G-IDENTICAL on a SHIP-profile shard and a STATION-profile shard** — the identical fixture
welds 1 000 blocks onto each realm and asserts the same derived mass, cross-section, drag coefficient
and look radius, to the last bit. Plus: a watcher at the old reach edge SEES the mast within one settle
window of the last weld.
*Measurement:* the count of upward statements while 1 000 blocks are placed at ten per second — **pass at
one statement per settle window, never one per placement**; and the derivation's wall time on a
50 000-block hull.

**Slice 11 — The terrain collider and the residency invariant. (R; blocked on the library decision.)**
*Lands:* tier-0 extraction over each body's SWEPT SEGMENT for the tick, fattened by the body's bounding
radius plus reach; one collider per chunk per realm, reference-counted; the `Tier0Key` type so a coarse
collider does not compile; the chunk-local origin rule.
**And four rules revision 1 left out.**
1. **The unbounded swept segment has an answer that is not a clamp.** The owner deleted the ceiling from
   the flight path and the galaxy's own ceiling is about 170 million times the speed of light
   (`docs/design/owner_decisions_2026-08-27_movement_answers.md`), so a tick's segment can ask for more
   chunks than any shard can build, and a re-clamp is a jump, which SL8 refuses. The rule: **the swept
   test asks the generator crate for the surface along the segment ANALYTICALLY, cell by cell, with no
   chunk built and no collider allocated; a contact found that way then builds the ONE chunk it needs.**
   SL10's own grant makes this legal, because the surface at an address is a pure function.
   *Example: a pilot switches off a hull's safety block and dives at a moon at ten kilometres a second.
   In one 50 ms tick the hull travels 500 metres. The shard walks that line through the crate, finds the
   ridge, and builds one chunk to resolve the contact.*
2. **The chunk build runs on a WORKER POOL, off the tick thread**, and a chunk enters the physics world
   only at a tick boundary from a COMPLETED build. A descent demands tens of builds in one tick at an
   ESTIMATED 0.5 ms each (`09_physics_controller.md:668-672`); on the tick thread that is a TICK HITCH,
   one of the eleven seam kinds. The pool already in the dependency tree is `bevy_tasks`
   (`Cargo.lock:544`, MEASURED); `rayon` is a NEW library and needs the owner's word (V103).
3. **A body is never stepped past the frontier of built chunks.** If the frontier falls behind, that is
   a GATE FAILURE, never a stalled body and never a silent tunnel.
4. **A parked child realm's SHELL BODY sleeps** the moment its authored velocity and its contact set
   stop changing, and it wakes on a contact, on a stated drive or on a stated shell change. SL9 says the
   per-tick cost must grow with the MOVING children and never with the parked ones. *Example: six
   hundred hulls sit parked at a spaceport on a moon. A pilot fires one hull's thrusters; that hull
   wakes, and the hulls its shell touches wake with it. The other five hundred and ninety stay asleep.*
*Laws:* SL10 V1.5, SL4 (physics produces a placement; the crossing path names no motion symbol), SL9
(the cost grows with occupants and edits, never with the planet's radius), the 2026-08-27 movement answer
(no speed cap; containment is swept).
*Gate:* **the landing fixture, flown by a HULL only** — a hull descends at its rated cruise onto a moon
and touches down, and the drawn surface height equals the collider surface height at the contact point,
at every approach speed a hull states. **The walk off the ramp moves to slice 16**, because no collider,
no shape cast and no character controller exists until then (U-48) and the suit ruling forbids a walking
dot standing in for one. Note that NOTHING crosses a realm boundary at touch-down: the planet's bound is
its sphere of influence, so the hull became the planet's child hundreds of thousands of kilometres up
(§1). Plus: drop a rigid capsule on 10 000 random surface points and never sink below the surface by more
than the solver's slop plus the extractor's snap error. Plus **G-IDENTICAL on a PLANET-profile shard and
an ASTEROID-profile shard** (`crates/sim/src/capability.rs:246,290`) — the identical fixture drops the
same capsule on the same relief in each, and the contact heights agree to the last bit. Plus **S12, the
gate that can fail**: 1, 100 and 600 parked hulls in one planet realm, and the per-tick core time must
not grow with the parked count.
*Measurement:* collider build time and bytes per chunk; the swept chunk count per tick for a landing
hull; the see-versus-stand vertical gap at contact — **pass under one extractor snap error**; the tick
thread's own share of the chunk work — **pass at zero**; and the collider cache's bytes against 1/10/100
occupants on a 500 m asteroid AND an Earth-sized body — equal to within one chunk, because the cost must
not grow with the radius.

**Slice 12 — The square lanes on a planet, and sub-metre blocks. (R + S.)**
*Lands:* the cube and shaped lanes over the same grid through ONE collider builder per lane; the sub-grid
form; the 512-bit occupancy mask that the placement path and the store open both use to refuse an
overlap; the per-cell record cap.
**And the SHELL SWAP, which revision 1 dropped.** When a build changes a realm's exterior shell, the
parent replaces the collider and lets the SOLVER resolve the new resting contact, with the
penetration-correction velocity CAPPED so the change is a SETTLE and not a jump. A block is one metre,
and a one-metre lift in one tick is a jump SL8 refuses. **The cap is a FIELD of the realm's own record,
never a constant** (the no-magic-numbers rule), and §2.2's record field list reserves it. At a settle
speed of 1 m/s that is 20 mm per tick on a 50 Hz shard (ESTIMATED, `09_physics_controller.md:562-572`).
**And the V2.6 style derivation.** The client derives the connected-block TRIM from the records it holds
(V72), under slice 7's rule that trim never moves a collided vertex. The derivation function is
byte-identical on both hosts, because a trim that differs between two clients is a detail-by-box seam
between two players standing side by side.
*Laws:* HR3 (lane selection by the cell's FORM), HR4, the FINAL-backend law (no second implementation
selected by `VoxelGeometry`), V2.3, V2.4.
*Gate:* **G-IDENTICAL** — the identical fixture places a wedge, breaks it, re-places it as a ramp and
chisels a quarter-metre block into the corner, on a PLANET-profile shard and on a SHIP-profile shard, and
the two stores compare byte for byte. Plus **S14, the settle gate**: weld a plate onto the belly of a
parked hull with a crew member on the ramp, 100 repeats, and demand that no tick moves the hull
vertically by more than the realm's stated cap. Plus a rigid-capsule sweep across a foundation cut into a
30° slope at 20 offsets — no vertical impulse, no climbable step. **The walked leg of this sweep moves to
slice 16**, for the same reason as slice 11's.
*Measurement:* the FRAME cost of one cell at 512 eighth-scale sub-blocks — collider build, greedy remesh
and upload, on a named thread — **pass under 16 ms of worker time, one frame at 60 Hz**. The 33 ms
figure revision 1 used here is a budget for remeshing CUBE faces and does not carry to a smooth extractor
(`09_physics_controller.md:589`). **★V7's STEP stays PROVISIONAL until this runs; the WIDTH is already
frozen and holds 1/16 m either way.**

**Slice 13 — Attachments and the body tree. (R + S for the overlay.)**
*Lands:* the attachment family and the body family in the realm store; the closed kind registry with
`Display` and `Joint`; the `Dof` enum `{Revolute, Prismatic, RailPath}`; the motor scalar on an integer
angle grid; **the sine and cosine TABLE carried in the crate**, keyed by that grid; the swept-box index
over bodies; the incremental inertia sum; the chunk-to-carriage index; the engine-free face-overlay
primitive, which DRAWS NOTHING until the value lane lands at P9.
*Laws:* HR1 (a constraint needs both bodies in one private physics world, so a joint child is a BODY and
never a realm), SL4, SL6 (**this slice consumes R-11, the body pose on the wire; it may not ship before
slice 0 records the owner's YES**), SL9 (only a body whose angle changed is re-inserted and queried),
SL10 V1.4 (no libm `sin`/`cos` — the earlier claim that a kinematic joint needs no transcendental was
FALSE), the no-placeholder-rendering rule.
*Gate:* the same joint at the same angle produces a byte-identical body pose on x86-64 and on aarch64;
and the piston-lifts-a-crate fixture passes on a ship profile and a planet profile with NO `match` on
`voxel().geometry` in joint code.
*Measurement:* a hull with 1/8/64/512 bodies, 0/4/N of them moving — the shard's tick time for the pose
derivation, the inertia update and the index queries, **pass under 10 % of the shard's stated physics
budget at 512 bodies with 4 moving**; and the bytes per tick to one observer, **pass under 200 bytes per
tick with 4 bodies moving**, so a turret's angle never crowds the datagram budget of 1 200 bytes. Plus
the CLIENT's frame time and draw-batch count for the same hull, **pass under 2 ms and under 64
batches**. Plus click-to-pixels through a body: an edit on a ROTATING turret, round trip
(`05_attachments_bodies.md:921`).

**Slice 14 — Trees as one object. (R + S for the art route.)**
*Lands:* the object record on the air cell above the ground; the side row holding the growth stage and
the plant tick; `expand` on the integer core alone; the capsule compound on the server; the seed-forest
placement draw appended AFTER every existing draw (the discovery-permanence law,
`crates/physics/src/worldgen.rs:17-36`); the felling diff; the support rule.
*Laws:* V2.2, SL6 (**this slice consumes R-13, R-14 and R-15 — the felling diff, the falling crown and
the canopy fold; it may not ship before slice 0 records the owner's YES on each**), SL10 V1.1 and V1.5,
SL9 (a moon with a million growing trees and no observers does zero per-tree work per tick), SL8
(arrival pop, lane flood), HR4.
*Gate:* fell 1 000 seed trees and assert **12 B ± 0 of permanent record each** — one `Empty` record at
Format B's frozen width — and 0 B of side row. Revision 1 said 8 B, which was written against the older
eight-byte record and would have made this gate fail on the shipped format
(`07_trees_composites.md:331`; the record widened at `04_block_record_registry.md:107-140`). Re-derived:
report 07's clear-cut disc of 19 406 trees is **233 KB, not 155 KB** (ESTIMATED, arithmetic on 12 B).
Plus: observe a felled forest at 4 km and approach — zero trees appear or vanish; observe a DORMANT
moon's forest with no diff held and assert no felled tree returns; chop an oak 1 000 times on the shipped
lanes and assert no frame shows a stump with no crown and no frame shows two crowns. Plus **the SUPPORT
CASE, which revision 1 left as three words**: mine the cell under a standing tree and assert the stated
outcome — the tree falls, or it stands and its record is untouched — with the same outcome on both
profiles. Plus **G-IDENTICAL on a PLANET-profile shard and a STATION-profile shard**: the identical
fixture plants, grows and fells one oak in a moon's soil and in a station's growing bay.
*Measurement:* the skeleton bytes identical on both builds and both targets; the derived PLACEMENT SET
identical likewise; every tube vertex within a stated phantom-hit bound of its capsule surface; and
`expand` per tree at the largest stage, plus the object-horizon disc's wall time and main-thread time.

**Slice 15 — The cross-shard edit forward. (R.)**
*Lands:* the `BlockEdit` forward from a child realm to its PARENT ONLY, routed by `ParentRealmNode`
(`crates/sim/src/stub/realm_head.rs:75-82`), `Retained` on the durable outbox that is on for every shard;
the origin realm's own monotone edit sequence and the owner's ONE low-water mark per origin realm;
`BLOCK_STORE_FLUSH_STEP` in the re-shard drain, before the pose flush.
*Laws:* SL1 clauses 1, 3 and 4 (**the child states the target point in its OWN frame; the parent adds the
placement it authored**), SL2 (no occupant pose crosses — the reach test reads the ORIGIN REALM's
placement and extent, never a person's pose), HR1, the fence discipline.
*Lands, second half — THE STORE TRAVELS WITH THE REALM.* HR1 makes a realm's store private and it moves
when the realm re-homes. So the rule: **the block store's BULK moves BEFORE the transfer's cut, and the
cut carries only the TAIL** — the rows written after the bulk copy began. Revision 1 added
`BLOCK_STORE_FLUSH_STEP` to the drain and sized nothing. *Example: a fifty-thousand-block hull flies from
one star system to another and its realm re-homes to another node. The berth row moves in milliseconds.
The hull's block store is measured in gigabytes, and without this rule the cut marker waits behind it.*
*Gate:* a pilot inside a hull berthed on a moon breaks a rock beside the ramp; the moon writes it once,
under its own fence, and a redelivery from the outbox is a no-op. Plus: the author crosses out of the
hull on the same tick, and the echo is dropped counted while the diff still reaches her through the
moon's own rows. Plus **the dig-during-a-hand-over fixture** (`10_law_audit_stale_register.md:445-450`):
a player carves a doorway in a hull's wall on the exact tick the hull leaves System 7. The doorway must
exist ONCE, in the galaxy's copy, with no lost tick and no duplicate, and the diff's fence must order
against the transfer saga's steps by a stated rule.
*Measurement:* 200 edits per second for one hour from one hull into one moon — the owner's idempotency
state must be ONE row per origin realm, never one per edit. Plus **the CUT duration for a 50 000-block
hull against the transfer's own budget, and the tick hitch a watching client sees — pass when the cut is
no longer than a berth-row-only transfer's cut, to within one tick.**

**Slice 16 — The character on the surface. (R.)**
*Lands:* gravity as a function of position from the realm's authored mass; up as the radial; slope and
step thresholds as CHARACTER RECORD FIELDS; the swept containment test on terrain; the realm's dynamics
frame for a spinning body.
*Laws:* SL4, the movement contract (a parent never sets a speed), the per-realm time multiplier, HR4.
*Gate:* a capsule stands still on the equator of a spinning body for 10 000 ticks and does not drift by
more than one fine cell per 1 000 ticks; a hull at 250 m/s does not tunnel through a 1 m ridge in 10 000
dives. **Plus the three walked legs deferred from slices 8, 11 and 12, which land HERE because the
character lands here** (the owner's order: the foundation, then blocks and terrain, then the character,
then the suit): the character walks off a landed hull's ramp and the drawn surface height equals the
collided height under her boots; the pop detector runs at true walking speed under the real controller;
and the character sweeps a foundation cut into a 30° slope at 20 offsets with no vertical impulse and no
climbable step. Plus **G-IDENTICAL on a PLANET-profile shard and a SHIP-profile shard**: the identical
fixture walks the same character across the same relief on a moon's hillside and along a hull's deck.
*Measurement:* the tick-time distribution over a 250 m/s descent across strong relief — no tick exceeds
the shard's stated physics budget. **That budget is named ONCE, as a field of the shard's own profile
record, and every slice cites it rather than restating a number** (the no-magic-numbers rule). Slice 11
and slice 13 read the same field.

**Slice 17 — The seamless landing gate. (S.)**
*Lands:* no new machinery. It is the acceptance sweep.
*Laws:* SL8 and V2.9, over all eleven named seam kinds plus the proposed twelfth.
*Gate:* one scripted flight from orbit to a walk on the ground, driven by `vdctl` with no human hands
(HR6), with the pop detector on every kind: jump, black frame, flicker, detail-by-box, brightness pop,
arrival pop, tick hitch, re-state rate, tier refusal, sprite cull, lane flood, **and the diff lag — the
diff for every cell inside the character's own reach is applied BEFORE the derived shape for that cell is
first drawn, and the frame count between the two must be zero.**
*Measurement:* one number per seam kind, each against the PHYSICAL tolerance below. **Revision 1 stated
no tolerance, so its gate could not go red: a first run against a baseline always passes.** Every
tolerance comes from report 08 §3.3 and carries its own mark.

| # | Seam kind | The physical tolerance | Mark |
|---|---|---|---|
| 1 | Jump | a drawn chunk's screen position across the swap moves by less than ONE PIXEL; `f64` re-expression at 1 × 10⁷ m carries an error below 2 mm | ESTIMATED (f64 epsilon) |
| 2 | Black frame | ZERO frames in which a resident region has no rung drawn | the rule; the pop detector must be built |
| 3 | Flicker | frame-to-frame luminance variance in a fixed region below a stated threshold while the camera moves slowly | ESTIMATED; the bench owes the threshold |
| 4 | Detail-by-box | for every realm kind, the resident rung at distance `d` is `L = ceil(log2(d × cell_angle_max))` from ONE field; a fixture asserts it for a moon, a hull and a station at the same distance | the rule; gateable only because the tier rule lives in the library |
| 5 | Brightness pop | the frame's mean luminance changes by less than ONE EIGHTH OF A STOP between successive frames outside a physical cause | ESTIMATED |
| 6 | Arrival pop | at the hand-over distance the coarse rung's silhouette differs from the sphere by less than ONE PIXEL | ESTIMATED (the relief bound `4·k_rough·2^L` over the distance, against the drawable angle) |
| 7 | Tick hitch | the scene's angular motion in one frame exceeds a constant-rate frame's by more than the drawable angle `1.1506 × 10⁻³` rad; at a 90°/s look turn the permitted extra frame time is 0.73 ms | ESTIMATED, from the drawable angle (`crates/core/src/geometry.rs:1171-1174`) |
| 8 | Re-state rate | ZERO new per-tick data for terrain; a chunk moves as its row moves, at the render rate; a joint's angle is send-on-change | the rule |
| 9 | Tier refusal | the refused-tier counter in `DevState` is ZERO in every flight, beside the engine-override counter | the rule |
| 10 | Sprite cull | every star the law draws paints at its predicted pixel (**MEASURED**: 0 misses of 41 345, `DEFERRED.md:1334`); and a one-pixel star's painted luminance is within a stated fraction of the law's amplitude under the chosen anti-aliasing | half MEASURED, half owed by U-24 |
| 11 | Lane flood | no diff lane queue exceeds its paced budget, and no keep-alive restates a roster | the rule |
| 12 | Diff lag (proposed) | the diff for every cell inside the character's reach is applied BEFORE the derived shape for that cell is first drawn; the frame count between the two is ZERO | the rule |

Each ESTIMATED row names the run that turns it into a threshold: U-19 for the rung disagreement, U-24
for the star's luminance, U-32 for the regime hand-over, U-33 for the drawn step. **A tolerance that is
still ESTIMATED when this gate runs is reported as a NUMBER WITH NO PASS, and the slice is not
complete.**

**What a later client-engine investigation can run in parallel.** Slices 0b, 1, 2, 3, 4, 5, 6, 9, 10, 10b, 11, 15
and 16 are renderer-agnostic. Slices 7, 8, 12, 13, 14 and 17 touch the render seam. The seam's contract —
`ChunkGeometry` with its vertex quantum, its part-transform table and its collision line — must shut
BEFORE another engine's plugin starts AND before the Bevy encoder is written, because both consume it.

---

## 4. The owner's decision register

Deduplicated across all ten reports, ordered by cost of lateness. **★ must be answered at the format
sitting** — because the first code cannot start without it, or because it asks the owner to change a
standing ruling. Revision 2 adds ★V101 to ★V107 and V103, V108 to V112, moves ★V38 into band A, and
stars ★V36.

### Band A — the format sitting (a wrong answer costs a migration over every saved world)

| # | Question | Options | Recommended | Why |
|---|---|---|---|---|
| ★V1 | The grid family | (A) cube-sphere + flat; (B) literal cube; (C) hex/Goldberg; (D) flat octree; (E) tangent-plane areas | **(A)**, signed with V6 in view | the only family that keeps blocks square, planets round, one grid with a hull, and no visible seam in the world |
| ★V2 | The warp's inverse: first guess and step count | 3 or 4 Newton steps; `a₀ = t` or `a₀ = t/k₁` | **`a₀ = t`, FOUR steps** | three steps from `t/k₁` leave 0.348 cells at the largest legal body — a margin of 1.4 against half a cell (MEASURED). Four steps reach the f64 floor from either guess, so the guess stops being load-bearing |
| ★V3 | The radius ladder and the rung count | (A) `N = q·2^(T−1)`, top rung = a face is ≤ 64 chunks; (B) the face-covering rung with `m` a power of two | **(A)** | (B) forbids realistic radii (Earth would land at 5 341 or 10 683 km), which the owner asked for |
| ★V4 | Signed or biased index fields | (A) signed `i32`; (B) biased by `half_cells` | **(A)** | a bias makes the client need the realm's `bound`, which does not reach it — an SL6 request the signed field dissolves |
| ★V5 | The record's width | (A) twelve bytes fixed; (B) eight or twelve, selected by the kind's row; (C) eight, and smooth terrain is refused | **(A)** | (B) costs a fixed stride, so a chunk delta stops binary-searching by index; (C) refuses V2.1. **This door and the density door are ONE door** |
| ★V6 | The corner build refusal | (A) accept it; (B) the generator ALWAYS places a landform at all eight corners; (C) a per-body landform; (D) bend the blueprint walk | **(B)** | eight corners per body at latitude ±35.2644° refuse a prefab stamp for a reason the player cannot see — SL8's "tier refusal". (B) removes it from experience without changing the grid |
| ★V7 | The finest sub-block step, and the address width | 1/2, 1/4, **1/8**, 1/16 m | **the WIDTH freezes now at 16 bits (level 4 + index 3×4). The STEP ships PROVISIONAL at 1/8 m** | the door is the WIDTH, not the step, and the width can be frozen today because it holds 1/16 m either way. The binding cost is runtime — 512 colliders in one cell, or 4 096 — and U-35 measures it at slice 12, nine slices after this row is answered. **So the sequence cannot meet revision 1's condition and the answer is split.** The re-open rule: after U-35 the owner may lower the shipped step (1/8 → 1/4) at the cost of refusing existing chiselled cells, or raise it (1/8 → 1/16) for free, because the width already holds it. Widening the WIDTH after slice 9 is a migration over every planet, hull, station and blueprint |
| ★V8 | The terrain sample: what it is and how fine | (A) a radial gap `r − h`, i8, 1/128 cell, cell-centred; (B) a corner-centred sample; (C) 1/127 | **(A)** | 1/128 is exactly 8 fine cells at tier 0 and `2^(L+3)` at tier L; 1/127 is 8.06 fine cells at no rung. Cell-centred means an edit changes exactly the cells the player clicked |
| ★V9 | The smooth extractor | (A) naive surface nets; (B) dual contouring; (C) marching cubes | **(A)** | fewest operations, no case table, no least-squares solve where a libm call creeps in, quad output for the collider. (B) is the reserved upgrade, and switching it is a CONTENT door |
| ★V10 | Does the terrain gap survive under a placed square block? | (A) yes, the byte is on every planet cell; (B) no, removal leaves air | **(A)** | the only rule that makes a foundation cut into a slope look right and restores the slope exactly on removal |
| ★V11 | The store key carries a BODY id | (A) `(body, chunk)`, body 0 = the realm's own grid; (B) one grid per realm, no turrets | **(A)** | without it a plate on a turret and a plate on the hull one metre below share a key. It costs the record zero bits |
| ★V12 | The attachment key | (A) `(BlockAddr, face, slot)` with the slot byte from the FIRST write; (B) `(BlockAddr, face)` and widen later | **(A)** | widening a KEY byte later costs a version floor on every persisted attachment in the world; skip-unknown covers tags, never key bytes |
| ★V13 | Attachments: one registry table or two? | (A) one table with an ATTACHMENT form class; (B) two typed tables, one mechanism, one digest | **(B)** | two typed ids make "an attachment kind in a cell record" a COMPILE error rather than a runtime refusal |
| ★V14 | The identity digest's column set | (A) everything; (B) only what changes what a SAVED record MEANS — the triple, the form's geometry, the OBJECT flag, the theme, the density convention | **(B)** | (A) puts the variant count inside, so shipping a fluted hull plate takes every store on every shard OFFLINE. That is the single most common content change |
| ★V15 | The registry digest at the CLIENT handshake | (A) equality; (B) prefix, plus a per-CHUNK refusal when a chunk names a kind the client does not hold | **(B) — CHANGED from revision 1's (A)** | a client that does not know a kind cannot generate a chunk holding it, so a refusal is needed either way; the question is WHERE. Under (A), V2.7 makes every theme a fleet-wide flag day: appending one substance refuses every unpatched pilot AT THE GATEWAY, which is the hardest seam the game has and SL8 has no tolerance for it. Under (B) a themed planet refuses only the pilots who fly to it, and the refusal names the kind. The STORE keeps its own prefix rule (V14) and this row only moves the CLIENT half |
| ★V16 | Orientation width | (A) 6 bits with the sixth zero-reserved; (B) 5 bits, the catalogue closed under mirroring | **(A)** | one bit now against auditing every shape for a twin after players have built. The two domains agree on the width and the zero rule; they differ only on what a future build may do with the bit |
| ★V17 | Where a tree's shape parameter lives | (A) `object_param` in the record; the growth STAGE in the side table; (B) both in the side table; (C) both in the record | **(A)** | the server reads the shape to build a collider, so it must be in the record the collider reads; the stage CHANGES with time and the permanent record must never be rewritten by the clock |
| ★V18 | The pyramid's prune rule | (A) prune on equality with the generator's coarse answer; (B) prune on "no authored cell below" | **(A)** | (B) draws a straight cliff along every coarse-cell boundary where an entry meets a generated neighbour |
| ★V19 | Does a coarse entry carry a surface-height datum? | (A) no: 8 bytes, 14 reserved bits; (B) yes: 8 reserved bits become a signed height delta | **(A) now, decided with the renderer before the first world is saved** | the entry is rebuildable, so the disk cost of a later change is a rebuild; the WIRE cost is a protocol event |
| ★V20 | A sticky "differs from the seed" bit against quantisation | (A) no; (B) yes, at the cost of an entry per edited column at every rung | **(A), PROVISIONAL, with a stated re-open rule** | a 20 m quarry vanishes at the 256 m rung (ESTIMATED, arithmetic on the fold rule). Whether the step is visible is UNMEASURED and U-33 owes it at slice 10, one slice after the first store is written at slice 9. **The re-open rule:** the stores of slices 9 to 13 are throw-away (§3 rule 1), so turning the bit on after U-33 costs a pyramid rebuild per realm plus a protocol-minor bump, and nothing a player built. After slice 14 the same change costs a wire event to every live client. The owner should say now which of those two he is willing to pay |
| ★V21 | The generator crate's arithmetic profile and version as WORLD IDENTITY | (A) freeze it and adopt an append-only octave rule (a new octave may only add detail below the stated tolerance); (B) leave it open | **(A)** | once a player builds on ground the client derived, one added octave or one rounding fix moves the ground under every placed block and every saved diff. There is no re-roll; the cost is THE WHOLE EDIT CORPUS |
| ★V22 | Who owns a body's radius? | (A) the generator crate, and the forest reads it; (B) the forest keeps it and the generator snaps to it | **(A)** | today the drawn radius comes out of a `powf` (`crates/physics/src/taxonomy.rs:616` → `crates/physics/src/worldgen/generate.rs:1232`). (B) makes a ladder step depend on an ulp of a `powf` on two targets, which is the exact drift SL10 exists to stop. (A) costs astrophysical fidelity and the owner should say how much is acceptable |
| ★V23 | `RealmId::System(seed)` aliases across galaxies and names the store FILE | (A) disambiguate the file name before the first world is saved; (B) leave it | **(A)** | `crates/core/src/realm_coord.rs:1-8` says the collapse in its own words, and `crates/bins/src/lib.rs:2310-2333` names the file from the id. Two star systems in two galaxies overwrite each other's buildings. **DATA LOSS, and silent** |
| ★V24 | May an area realm name its parent's cells through its told berth? | (A) yes, planet-global addresses stored and shipped; (B) an area is a shard partition, not a grid; (C) the berth is an INSTRUMENT for evaluating the shape, and every stored or shipped address is AREA-LOCAL | **(C)** | (A) folds the told placement into a stored address and passes it on, which SL1 clause 4 refuses |
| ★V25 | Which realm is the single writer of a cell inside an area's box? | (A) the area, and the planet refuses cells inside a live area's box; (B) the planet; (C) both, with precedence | **(A)** | `RealmId` is the unit of single-writer durable state (`crates/core/src/pose.rs:34-35`); (C) gives one cell two authors and the gateway nothing to compose |
| ★V38 | May a Cartesian realm hold terrain-form cells? **MOVED into band A by the law critic** | (A) yes (a station's growing bay, a hull's ballast hold); (B) no | **(A)** | HR4 depends on the answer: if it is NO, the smooth lane can never run on two shard kinds, and slice 10 cannot land at all. The code already permits the pair (`crates/sim/src/capability.rs:134,260-272`). It adds soil in hulls to the game, which is why it is the owner's |
| ★V101 | **How is "a cell returned to the seed" encoded?** (the storage report's D-8, `06_storage_diff_lane.md:890`; dropped by revision 1) | (A) an `Empty` KIND record with provenance `Placed`, every other field zero; (B) a provenance value; (C) a separate revert TAG on the wire | **(A)** | the client derives the seed's shape, so every removal must beat that derivation for all time, and the encoding is a format field. If it is decided after the first store is written, every tunnel and every stump in the world re-encodes. (A) needs no new field and no new tag. *Example: a player cuts down an oak on a moon; a week later her client re-derives that chunk from the seed and the `Empty` record keeps the oak down* |
| ★V102 | **Where does the CANOPY fold live?** Format C reserves fourteen bits and three spends want seventeen (§2.3) | (A) drop one spend; (B) widen the entry to 12 bytes; (C) a SECOND store family keyed by the same `(body, tier, chunk)` | **(C)** | it keeps the 8-byte entry and pays one more family. (A) costs either the arrival-pop tolerance or a resurrected forest at distance. (B) is a wire event as well as a disk change. The entry is rebuildable on disk and a protocol event on the wire, so the WIRE half is the hard half |
| ★V104 | **How is a `BodyId` allocated?** (dropped by revision 1; it sits in a persisted key) | (A) body 0 = the realm's own grid, then a monotone per-realm counter, never reused, persisted in the realm's own store; (B) dense re-use after a body is cut off | **(A)** | (B) makes a cut-off turret's chunks answer a new crane's key, and the wrongness is silent. *Example: a gunner bolts a turret onto a hull's spine, files its plates under body 1, cuts the turret away and welds a crane in its place. Under (B) the crane takes body 1 and a saved hull loads with plates that belong to a machine that is gone* |
| ★V105 | **How is V2.6 read?** The owner said *"it can be one block type with different params — durability, mass, and style"* | (A) mass and durability are per KIND from the substance row, and style is a 6-bit per-cell variant (this document's reading); (B) per-INSTANCE parameters in the record | **(A)** | the record width depends on the answer, so it belongs in band A. (A) is a defensible reading and it is the OPPOSITE of the owner's sentence read literally. Report 10 lists it as an open contradiction (`10_law_audit_stale_register.md:361`) and revision 1 answered it without showing the question. (B) costs bits the record does not have |
| ★V106 | **Where does a sub-metre block SIT on a smooth slope?** (`10_law_audit_stale_register.md:441-452`; the word "seat" did not appear in revision 1) | (A) a LATTICE OFFSET address plus a SHARED, DERIVED seating rule inside the generator crate; (B) a stored SURFACE-RELATIVE seat | **(A)** | (B) re-binds whenever the generator's version moves the surface, which §2.4 forbids. Under (A) nothing about the seat is stored and both hosts compute it by one rule. Without a rule at all, a quarter-metre lamp post on a hillside floats or sinks, and the seat is invented twice — which SL10 V1.2 forbids by name |
| ★V107 | **Is a built realm's slot a BOX?** Sell slots as an `Aabb`, or keep `Shell` slots and inscribe a cube (`01_grid_family.md:895`, D5; dropped by revision 1) | (A) `Aabb` slots; (B) `Shell` slots with an inscribed cube | **the owner decides; (A) is the simpler grid domain** | it decides a hull's grid DOMAIN, which slice 2 needs before it writes `GridParams` |

### Band B — before the first generator or collider code

| # | Question | Options | Recommended | Why |
|---|---|---|---|---|
| V26 | HR4's own wording names a stateful `FrameSpace` seam (`CLAUDE.md:110`, `docs/design/PLAN.md:42`). May it become the stateless `GridMapping`? | (A) rename and move any anchor below it; (B) keep `FrameSpace` as written | **(A)** | no `f32` sits on the authoritative path (`crates/core/src/pose.rs:547-551`), so the anchor's only remaining reason is a physics library nobody has chosen. This answer also decides the fate of the owed reanchor fixture (`docs/design/DEFERRED.md:331`) |
| V27 | The physics library and its precision | (A) `rapier3d-f64` in the realm frame; (B) `rapier3d` f32 with a per-island anchor; (C) another; (D) hand-written for the capsule case | **investigate together; nothing adopted here** | f64 deletes the ANCHOR half of the seam; f32 means it must be BUILT. Neither is in `Cargo.lock` (MEASURED). The standing rule: research options, the owner decides |
| V28 | The noise source | (A) vendor ~400 lines of gradient noise on the integer hash, inside `Gf`; (B) `noise 0.9.0` with a const table; (C) another crate | **(A)**, and delete the unused pin | no dependency, no `rand` drift class. **The bench's 0.885 ms does NOT transfer as-is**: its hash COMPOSITION differs from `child_seed` |
| V29 | `#![no_std]` for the generator crate | (A) `std` + the lint + the link scan; (B) `no_std` + a `libm`-class dependency decision | **(A)** | `no_std` does not stop an `extern "C"` libm call, and on stable it deletes `f64::sqrt`, which SL10 V1.4 grants by name. The link scan is the fence either way |
| V30 | The x86-64 leg of the no-drift gate | (A) emulation on this Mac; (B) a cloud x86 runner; (C) a contributor's x86 box | **(A) as a smoke test now, and (B) or (C) before V1.3 may be called satisfied** | an emulator can FIND a difference and can never prove its absence. Subnormal handling and flush-to-zero are the classic divergence, and a height field that divides two nearly-equal metres can produce a subnormal |
| V31 | The world tag: one value or two? | (A) one folded value on every carrier; (B) a DECLARED half on the store stamp, the ALPN and the look tag, and a MEASURED boot digest on the live handshakes only | **(B)** | a store must never be refused because a pod landed on a different chip; the only remedy on a stamp mismatch DISCARDS the file (`crates/core/src/store_stamp.rs:182-192`), and a saved diff is a player's tunnels |
| V32 | The client-facing tag carrier | (A) append `world_generation` to `ProtoVersion`; (B) a new trailing `HelloWorld { declared, measured }` | **(B)** | (A) is unavailable: `ProtoVersion::CURRENT` is a `const` folded at compile time and the code states in its own words that no caller may state it (`crates/wire/src/version.rs:355-373`), while `:10-14` prescribes a trailing variant |
| V33 | `mul_add` on the generation path | (A) keep it off `Gf` until the gate measures it on both targets; (B) allow it | **(A)** | SL10 V1.4 forbids FMA CONTRACTION; the base says the explicit `f64::mul_add` call is bit-exact. These are different things and the two texts disagree. `mul_add` lowers to a hardware instruction on one target and a library call on the other. UNMEASURED |
| V34 | `f64::min` / `f64::max` on the generation path | (A) removed from `Gf`; write `if a < b { a } else { b }`; (B) keep them | **(A)** | Rust documents them as returning either input non-deterministically when the inputs compare equal — the `+0.0` against `−0.0` case, which a height field can produce at sea level. It is the one operation whose result IEEE-754 does not fix |
| V35 | The cave lattice's step, which is now VISIBLE geometry | (A) every 4th cell (0.885 ms MEASURED, a cave mouth resolved to 4 m); (B) every 2nd; (C) every cell (over the 1.20 ms gate) | **decide on pictures, before the digest is pinned** | under a blocky field the step only chose which cubes were air; under a smooth extractor it IS the cave wall a player walks into. The step sits INSIDE the digest door |
| ★V36 | Which materials are seed-decided (public)? **(B) ASKS THE OWNER TO RELAX S5.1 BY NAME** | (A) bulk structural stock only; (B) bulk plus common ore (iron, copper, coal) | **(B)**, with the line stated: what the client computes cannot PAY, and what pays the client never computes | the 2026-08-27 seed ruling's sentence is absolute — *"a block's SUBSTANCE may not be a pure function of (position, seed)"* (S5.1) — and SL10 V1.1's *"seed-decided common materials"* relaxes it for common stock. Turning an absolute sentence into a graded one is the owner's word, not this document's. Carry the ruling's own test into the sitting: *"if this were printed on a public wiki tomorrow, would the game still work?"* *Example: a prospector lands on a moon to survey a copper seam. If copper is seed-decided, a second player computes every copper seam on every moon in the galaxy from the client he already runs and prints the map. The survey instrument still exists and nobody needs it* |
| V37 | How strong may a seed-derived INDICATOR material be? | (A) none for the strategic tier; (B) a hint with a small DECLARED factor; (C) a hint whose factor is live state | **(A) for the strategic tier, (B) only for materials V36 already made public** | a declared multiplier on a static position is a prospecting map in every client binary, and it devalues the survey instrument the concealment ruling paid for |
| V39 | May a sub-metre block go in a PLANET cell? | (A) yes, inheriting the cell's anisotropy (0.0884–0.1256 m per 1/8 m part, MEASURED); (B) flat grids only | **(A)** | (B) gives a surface base less detail than a hull, which is a detail-by-box seam and a rule the player meets as "my workshop looks worse on the ground" |
| V40 | A rotation joint's geometry | (A) a MOUNT: a second flat grid inside the same realm, orientation authored by the hull; (B) the turret is a child REALM; (C) off-grid rigid geometry | **(A)** | it forks no code — the same arm instantiated twice. (B) makes forty joints forty realms, and a constraint solved across two realms works only while the scheduler co-hosts them. (C) contradicts V2.3 |
| V41 | Which capability gates a body tree? | (A) none, `block_edit` is enough; (B) a derived `mechanisms`, set from `block_edit` in the one constructor | **(B)** | `integrates_children` is the CHILD-REALM drive lane's switch (`crates/sim/src/stub/drive.rs:321,378`), and an asteroid — a realm players mine and build in — does not have it (`crates/sim/src/capability.rs:276-282`) |
| V42 | Kinematic joints first, or adopt a rigid-body library now? | (A) analytic first, WITH the carried sine/cosine table and its cross-target gate; (B) a solver now | **(A)** | without the table the "deterministic" half of (A) is false: a revolute joint at angle q builds `(cos(q/2), axis·sin(q/2))`, and SL10 V1.4 forbids the libm call |
| V43 | The realm's dynamics frame when the body spins | (A) one NON-ROTATING frame with the terrain as a kinematic set carrying the spin; (B) body-fixed with centrifugal, Coriolis and Euler terms; (C) split the frames at an altitude | **(A)**, falling back to (B) if a sweep against a moving trimesh fails | (C) is a new seam. This is the one question the physics domain could not settle |
| V44 | Who evaluates chunks on the client, and on what budget? | (A) the render bin owns a pool and feeds finished chunks into Tier-A `vd-client` through a seam; (B) `vd-client` gains a pool; (C) single-threaded with a per-frame slice | **(A)** | it keeps the Tier-A crate pure and matches the existing clock seam (`crates/client/clippy.toml:6-10`). `vd-client` spawns no thread today (MEASURED). The one-core case must be measured before it settles |
| V45 | The CLIENT's worker-pool mechanism | (A) `std::thread` + `crossbeam-channel` (already a workspace dependency, `Cargo.toml:66`); (B) rayon | **(A)** | it adopts nothing new, and the clippy fence bans `thread::sleep`, not `thread::spawn` |
| V103 | The SHARD's worker-pool mechanism for chunk builds — a row revision 1 never wrote | (A) `bevy_tasks`, already in the lock file (`Cargo.lock:544`, MEASURED) and used by no crate yet; (B) `std::thread` + `crossbeam-channel`; (C) rayon | **(A) or (B); rayon is a NEW library and needs the owner's word** | the shard's chunk build must leave the tick thread or every player in the realm stutters, and the tick hitch is one of the eleven seam kinds. Revision 1 answered only the client side. *Example: a pilot dives a hull at a mesa at 250 m/s. The moon's shard builds the mesa's chunks a hundred metres ahead of the nose; on the tick thread, every player in that realm feels it* |
| V46 | Who owns the tier rule, the chunk ORDER and the crossfade BAND? | (A) the client LIBRARY owns all three, with a counted debug override; (B) each engine | **(A) — WIDENED from revision 1, which named only the tier** | without the tier rule the detail-by-box tolerance is ungateable on a client on another engine. Without the ORDER and the BAND, SL8's continuity forks with the engine: the Bevy client dissolves fine cells in over four frames and another engine's plugin's own queue delivers them first, so the ground appears in patches. The engine keeps the THREAD that runs the work and the SHADER that paints the blend |
| V47 | The vertex quantum | 1/16 m or 1/32 m | **1/16 m**, unless the sub-metre answer wants finer | both fit one `u16` per axis at every legal tier. If the geometry's integers counted whole metres, the seam contract would shut ON V2.4 |
| V48 | The C header | (A) hand-written plus a signature test; (B) `cbindgen` as a build dependency; (C) `cbindgen` as a TOOL, header committed, diff gate | **(A) now, (C) if the surface grows** | about a dozen functions; (B) adds a new crate to the product graph, which is the owner's call |
| V49 | The generator crate's dependency shape | (A) depend on `vd-core`; (B) a leaf `vd-seed` (rng + digest) under both; (C) `vd-core` depends on the generator | **(B)** | one hash, one digest, the smallest static library, and no `libm` edge through glam |
| V50 | The crate's name | `vd-terrain`, `vd-shape`, `vd-surface` | **`vd-terrain`** | `worldgen` already names the FOREST generator in `vd-physics`; one word, one meaning |
| V51 | Does the golden set carry literals per TIER? | per tier / tier 0 only | **per tier** (ESTIMATED ~832 digests) | `h(dir, L)` is a different function per rung; a tier-0 pin proves nothing about tier 3 |

### Band C — before the subsystem is built

| # | Question | Recommended | Why |
|---|---|---|---|
| V52 | Where does the pyramid persist — inside the WAL's transaction, or at the checkpoint under a watermark? | **the checkpoint** | the pyramid is DERIVED and rebuildable from tier 0, so it does not need the write-ahead log's crash guarantee. That is the whole reason. The 5 508× write amplification is **ESTIMATED and inherited** (`storage_and_streaming.md:1057-1086`); its own source report states that every number in that list is UNMEASURED on this codebase (`06_storage_diff_lane.md:396`), and U-26 owes the run |
| V53 | The owner fence on the realm store — a file lock, or `last_owner_fence` compared on open? | **the fence** | the cloud volume is network-backed; advisory locks are not reliable there |
| V54 | The per-realm byte budget's sizing | **the scattered row (~6 entries/edit), hierarchical per claim** | a realm-wide refusal lets one logging operation stop everybody's editing on a moon |
| V55 | The diff lane's carrier | **a paced `Bulk` class on its own stream** | a 23 MB city must not queue in front of the transfer cut marker |
| V56 | Diffs ship durable-first or fast-first? | **durable-first, at tick T+1** | a client never draws a wall the restarted shard lacks; the cost is one tick |
| V57 | How does a client that arrives after a week of digging learn the week? | **the whole diff set per chunk on subscribe for the first slice; a per-chunk digest when the standing bytes hurt** | the optimisation's need must be a measurement, not a guess |
| V58 | Does a valuable-deposit diff ship the moment a chunk enters interest? | **the owner decides; nothing recommended** | the letter of the seed law holds (a deposit is live state, so it is a diff), but the lane hands the survey mechanic away: a changed client reads the ore 60 m down and drills straight to it |
| V59 | Which lane carries a LIVE attachment value (a HUD reading, a joint angle)? | **the realm's own per-tick statement lane, not the paced chunk-diff lane** | a joint angle is live state at the tick rate, not a diff of the world's shape; on the paced lane it queues behind a city |
| V60 | Does a landed hull's collider surface cross to its parent? | **the owner decides; option (c) deserves the look** | (a) resolves the contact against one radius, so a hull sinks into a hill or floats over it. (b) is chunk data crossing a realm boundary. (c) — the HULL resolves its own contact against the parent's terrain, which it can generate from the seed but needs the parent's nearby EDITS for — keeps the one-hop rule |
| V61 | The HUD content lane | **a value level with the full key for the shape; measure it against a per-observer handle at P9** | the owner's own words are "a widget from a predefined set"; the server states the range, the unit and the label ONCE on placement, so the client invents no scale |
| V62 | A body's row identity on the wire | **the body's pose rides its REALM'S OWN row, in the look bag, under that row's stamp** | two rows on two stamps, each interpolated on its own schedule, IS the seam: the platform slides under the gunner's boots for one frame |
| V63 | What becomes of a body whose anchor block is destroyed? | **refuse the removal while a joint holds the block, for the foundation; a small BUILT REALM later** | `DEBRIS_DEF` is Transient, 128 bytes, loss budget 4 (`crates/core/src/entity_kind.rs:225-232`), so a player's built turret could be DROPPED. Combat cannot refuse, so P11 needs the real answer |
| V64 | Does the edit request keep an `anchor_gen` field? | **leave it OUT and decide with the physics library** | no anchor state exists on the server for the grid; the lattice does not move, so a stored address cannot silently mean a different place |
| V65 | Mining yield on a partial surface cell | **the cell's fill, quantised to 1/8 unit** | mass is conserved; a player who scrapes a dune gets sand in proportion |
| V66 | Walkability on natural slopes | **everything but vertical (climb ≈ 67°), as CHARACTER data** | under smooth terrain the threshold judges every hillside, not just player-built ramps |
| V67 | May a natural substance be placed as a SQUARE block? | **no for the first slice; a later manifest row** | it keeps the rim-warp machinery out of the first slice entirely, and (a) is additive |
| V68 | Which mechanism owns Nub, Quarter and Post? | **the sub-metre sub-grid, if a half-metre grid ships** | two mechanisms for one 0.5 m cube is the fork HR3 exists to prevent |
| V69 | The tier switch's pop remedy | **geomorph, INSIDE the ladder slice, if the measurement says so** | narrowing the pixel target raises the chunk count everywhere; accepting it breaks SL8. The measurement runs first |
| V70 | Dig-and-fill: the abuse gate | **a per-account per-realm edit RATE limit** | a refilled hole NEVER prunes, on purpose (a `Placed` cell can never equal a `Terrain` cell), so a griefer can dig and fill until the shard refuses honest edits |
| V71 | A per-CELL record cap on the placement path | **yes, in the tuning struct, refusing the N+1st sub-block with a typed error** | 40 000 cells at eighth scale is 246 MB — well inside a gigabyte budget, and 20.5 million colliders no shard has been shown to build inside a tick |
| V72 | Who authors the connected-block TRIM, and what keeps it harmless? | **the CLIENT derives it from the records it holds, under a stated rule: derived decoration never changes the extractor's output, never changes a collider, and never moves a vertex the collision mesh shares** | SL10 V1.7 does NOT grant it — the trim's inputs are placed records and a neighbour mask, not `(seed, address)`. So this is still the base's open user decision, and the record survives either answer. The harmlessness rule is new in revision 2: without it a client bevel lifts a crewmate's boots off the spine she can see. Slice 7's gate asserts it by comparing the geometry the seam hands out with the geometry the collider reads |
| V73 | The translucent placement preview | **the owner rules** | it is a DRAWING and never state — nothing is applied, so a refusal deletes the ghost and rolls nothing back. Without it the player waits ~101 ms plus the buffer with no feedback on every block |
| V74 | Structural settling after a support is mined | **none in v1: support is a placement-time check, and a block whose ground is dug out hangs** | a settling pass is a per-tick cost over an unbounded structure; adding it later moves every hanging structure on every saved world at once |
| V75 | Terrain under a surface realm (a spaceport area, a station on the ground) | **the LOCAL formulation: an area's floor is BUILT, so it holds only its own cells** | if the owner wants bare rock, the area evaluates the generator from its stamped placement as an instrument, and TWO further things must be true — the stale rule and the edit-diff ask |
| V76 | The tree's instance seed | **derived from the cell address, zero bytes** | a replanted cell growing the same tree is CORRECT, not a defect — it is what lets the record prune |
| V77 | The tree art route | **both, with the procedural reference mesh shipped first** | an art-pack table is NOT free: it puts a shape branch back in the renderer, which the seam exists to prevent |
| V78 | Does a falling tree collide while it falls? | **PENDING U-52. Neither option is clean and the report says so in its own words** (`07_trees_composites.md:919`); the leaning recommendation is a ONE-WAY pusher, with the cost stated | it pushes a player an ESTIMATED 2.4 m before the player sees contact. Revision 1 recommended on the estimate; revision 2 marks it pending, because the measurement is what decides |
| V79 | Growth: memoryless, or a stored plant tick plus a fixed schedule? | **the schedule** | it stores one tick, walks nothing, costs zero on a moon with no observers, and needs no integer binomial sampler nobody has named. It changes the base's stated intent, so the owner should say |
| V80 | Where is the object horizon, and what is beyond it? | **DERIVED per object kind and per growth stage from the object's own size and the drawable floor, with a canopy fold on the terrain surface beyond. PENDING U-53 for the crossfade** | revision 1 recommended a constant, "~800 m", which the no-magic-numbers rule refuses and which its own row then contradicted: a 20 m oak is 22 pixels at 800 m. A drawing distance is a consequence of what a thing LOOKS like — the same rule that already decides a realm's reach (`crates/core/src/geometry.rs:1156-1173`). *Example: a themed world (V2.7) grows a crystal spire eighty metres tall. Under a constant it drops into a flat fold while still ninety pixels wide, and beside it a one-metre shrub is meshed in full detail where it covers one pixel.* The expand-cost bound then falls out of the derivation and must be re-measured, not inherited from the 20 100-tree figure |
| V81 | Do hand-stacked wood blocks ever become a tree? | **no** | V2.2 already made a tree one object; a recogniser reopens every cost it removed |
| V82 | Is the canopy pass-through for every kind? | **a per-kind flag, default pass-through** | one data bit; a dense conifer or a themed crystal spire may want a solid crown |
| V83 | Small rocks: cells or objects? | **cells up to one cell, objects above** | a `Nub` is already a cell with a collider and a drop |
| V84 | The anti-aliasing family | **temporal — and NOT before the star's painted luminance is measured** | multisampling at 4× is SUSPECTED of dimming a one-pixel star to a quarter; that is coverage arithmetic on paper, not a measured luminance. The door must not open on an unmeasured number |
| V85 | The exposure law | **physical closed-form** | adaptive exposure makes stars vanish on warp arrival, which is the brightness-pop seam |
| V86 | Who computes the camera's local up? | **the client, from delivered rows** | it is display arithmetic and derives no pose |
| V87 | The atmosphere hand-off | **a physics-invisible swap at the noise floor, plus limb shells on every other body's own look** | a density ramp with a re-anchor is a second mechanism |
| V88 | How does the client warm ahead without a predicted pose? | **the band is sized from SERVER-STATED data only: the realm's own reach, the interpolation buffer, and a LEAD the owning shard states beside them. If no delivered datum is enough, R-18 goes to the owner as a real ask** | **CHANGED: revision 1's answer — "the residency rule reads the delivered track's spread" — breaks SL10 V1.7.** The spread of a delivered track over time IS a velocity, and V1.7 says the client never derives a velocity. The arithmetic belongs where the poses are authored: the reach ruling already sums *"closing speed × boot time"* on the server (`docs/design/owner_decisions_2026-09-02_reach.md:184`) and the shard's interest lead is server-side (`crates/wire/src/channels.rs:322-328`). *Example: a hull dives at a moon at its rated cruise. Under revision 1 the client subtracted two delivered rows, called the difference a closing speed and decided how much ground to build ahead of the boots. That is the exact class of drift the client-only-renders law exists to stop* |
| V89 | The diff reach rule for an outside looker, **and the rung floor** | **the LOCAL formulation, applied to BOTH: the realm derives fine interest from the poses it authors and publishes down to the rung drawable at its own bound. It works out its own floor and needs no number from the gateway** | SL6 says find the local formulation first. **Revision 1 applied this rule to the interest radius (R-19) and not to the rung floor (R-4), which is the same question with a coarser answer.** R-4 also collides with SL3: a realm draws itself *"at a detail level it chooses"* (`CLAUDE.md:177-181`), and the reach ruling repeats that a realm's smaller statement is *"the realm's choice, never a parent's clamp"* (`docs/design/owner_decisions_2026-09-02_reach.md:181-183`). A rung floor pushed down from the gateway IS a clamp on the realm's own drawing. *Example: a pilot in orbit looks at a mining moon. Under R-4 the gateway tells the moon "send nothing finer than eight-metre cells". Under this rule the moon works it out alone: it knows its own bound and it knows the drawable floor.* If a floor is still wanted for a byte budget, it must be re-argued against SL3 by name, stating what the realm gives up |
| V90 | `BlueprintId` and the stamp's content hash | **keep the `u128` the code already has (`crates/core/src/built.rs:26-30`) AND state that the id is never a trust boundary** | an ACCIDENTAL collision at 128 bits is impossible in practice; a DELIBERATE one costs 2⁶⁴ work, which is reachable, and blueprints are player-authored |
| V91 | Does a sub-block carry its own damage? | **yes, keyed `(body, cell, sub-site)`** | a shared pool would let one nub's damage break a console. The crack-stage wire entry is unbuilt, so widening it is free today |
| V92 | Blueprint stamps and the snap lattice | **lazy stamps plus an 8 m snap** | large savings on hulls and stations; loosening is free later, tightening strands stamps. The feel is the owner's |
| V93 | A theme's placement rule | **a per-realm allow-list of themes, realm data, built later** | the registry decodes ALL themes; reserve nothing in the record |
| V94 | Cross-realm edits (a player on the ground editing a parked hull) | **refuse in v1; route as a crossing command at P8** | the gateway routes input by authority and cannot express a second target today |
| V95 | What wakes a realm, once the client can draw its shape? | **the trigger STAYS and its stated reason changes: a realm must run so that its DIFF exists** | SL3's stated reason ("a realm that is not running cannot be drawn") dies for the static half of the picture. If the planet never wakes, the tunnels players dug are not there |
| V96 | The proposed TWELFTH seam kind — the diff lag | **adopt it, with the physical tolerance: the diff for every cell inside the character's reach is applied BEFORE the derived shape for that cell is first drawn, and the frame count between the two is ZERO** | the eleven kinds do not name it; the nearest, "arrival pop", is about a realm arriving, not a surface correcting itself under a standing player |
| V97 | A mint pays an O(children) fold on its parent | **make a minted realm a mover for the placement layer, or replace the vector copy with a per-row edit** | `rebuild_static_rows_for` walks every child (`crates/sim/src/stub/regions.rs:876`, called at `:752`). MEASURED 2026-09-05: a hull's adoption into the galaxy (279 380 children) made the slowest tick 60.0 ms against a 20 ms budget. Ships are exempt; a minted STATION is not |
| V98 | A curated starter world | **no override table; the home is a FOUND Earth-like planet, and the city there is built by hand as live state** | the owner said "seed plus configuration" on 2026-08-03 and "no hand-authored table on the generation path" on 2026-08-15 and 2026-08-27. Newest wins, but the base carries the older words as a ruling |
| V99 | Light-lag between two realms | **in-system light-lag only between direct children of one realm; deeper endpoints ride the relay plane with a constant per hop** | the approved visibility scalar says how far the nearest OUTSIDE LOOKER is, with no direction (`crates/wire/src/intershard.rs:1319-1342`). It cannot carry a distance between two named realms, and nobody can compute one without new data |
| V100 | Per-realm budgets for bodies and attachments | **measure, then set them as fields in the one tuning struct** | the base's 512 and 1 024 are not derived from anything that still holds |
| V108 | The substance and form registry CAPS — the full `u16`, or the base's 512 and 256? (`04_block_record_registry.md:825`, D-6; dropped by revision 1) | **the full `u16`, with the tables sized on demand** | slice 3 builds the tables, and V2.7's themes need the room. A cap discovered when a theme ships is a cap discovered in play |
| V109 | ONE attachment per face, or several slots? (`04_block_record_registry.md:833`, D-14; dropped by revision 1) | **the owner decides; report 04 calls it a GAMEPLAY call** | §2.2 puts a SLOT byte in the key, so this document already answers "several" without asking. The key byte is frozen with the address, so a "one per face" answer would remove a byte the design cannot add back later |
| V110 | Who may attach to a block on a realm they do not own? (`05_attachments_bodies.md:1015`; dropped by revision 1) | **owner-only until an access list lands, refused LOUDLY with a typed reason** | a silent refusal is a bug report; a loud one is a rule the player learns once. *Example: a visitor bolts a gauge onto another pilot's hull while it is berthed. The hull's shard refuses and names the reason* |
| V111 | The MARK rule and the cosmetic-local tier (`08_render_seam_seamless.md:833`; two of the eight rendering answers the owner still owes) | **the owner decides; nothing recommended** | they were owed before this investigation and they are still owed |
| V112 | The RESIDENCY split — the SERVER holds realms at the dot angle, the client LIBRARY holds chunks at the drawable angle (`08_render_seam_seamless.md:834`) | **(as stated): two ladders with two owners, and one reference view under both** | V46 answers who owns the TIER rule and never who owns RESIDENCY. Without this row a chunk can be resident for a realm the server has not woken, or a woken realm can have no resident chunk |

---

## 5. SL6 requests

Default NO. Nothing below is added by this document. Each states the data, the direction, why the
receiver cannot compute it, and what doing without costs. **The realm-to-realm crossings are R-1, R-2,
R-6, R-16, R-17, R-20 and R-21** (revision 1 named R-9, which is a client build talking to the gateway,
and omitted R-16 and R-17, which do cross).

**★ SLICE 0 ANSWERS THIS TABLE, ROW BY ROW.** SL6's default is NO, so an unanswered row may not be
built. Slice 4 plants only the rows the owner recorded as YES, and slices 10, 13 and 14 each name the
R-numbers they consume.

| # | What | From → to | Why the receiver cannot compute it | Cost of doing without | Can the design live without it? |
|---|---|---|---|---|---|
| **R-1** | `InterShardFlow::BlockEdit(BlockEditForward)` — target points **in the ORIGIN's own frame**, the verbs, the origin realm coord and fence, the session id, the origin's edit sequence | a child realm's shard → its PARENT's shard, one hop, routed by `ParentRealmNode` | the parent holds the store and the child's placement; HR1 forbids the child writing the parent's file; SL1 clause 4 forbids the child stating a parent-frame address | no edits from inside a landed hull; a pilot must step outside to dig | **YES, at a cost.** Refuse cross-realm edits in v1 (V94) |
| **R-2** | `InterShardFlow::BlockEditAck { origin, seq, outcome }` | the parent → the child's shard | the child must tell the client the outcome and stop retrying | silent loss or endless retry | only with R-1 |
| **R-3** | `ShardToGateway::BulkFor` + `BulkMsg::ChunkRows` / `ChunkManifest` on a new `MsgClass::Bulk` | the owning realm → the gateway → named sessions. **Not a realm boundary** (SL2's 2026-08-24 clarification) | the client derives the seed's shape and cannot derive an edit | **no edit is ever visible to anybody**, and the client's shape disagrees with the shard's collider on every edited cell, which breaks SL10 V1.5 outright | **NO** |
| **R-4** | ~~`rung_floor: u8` on `GatewayToShard::WindowOpen` and on the relay's downward request~~ **WITHDRAWN in revision 2, not asked** | ~~the gateway → the watched realm's parent → the watched realm~~ | — | — | **YES — and the ask is withdrawn.** V89's local rule answers it: the realm derives fine interest from the poses IT authors and publishes down to the rung drawable at its own bound. It knows its own bound and it knows the drawable floor, so it needs no number from the gateway. The ask also collides with SL3 — a realm draws itself *"at a detail level it chooses"* (`CLAUDE.md:177-181`) and a smaller statement is *"the realm's choice, never a parent's clamp"* (`docs/design/owner_decisions_2026-09-02_reach.md:181-183`) — and a rung floor pushed down IS a clamp. If a floor is still wanted for a byte budget, it must be re-argued against SL3 by name, stating what the realm gives up |
| **R-5** | `ClientControlMsg::WorldAction` + `GatewayToShard::SessionAction` | the client → the gateway → the session's authority shard | a block edit must never be lost; `action_bits` is latest-wins and unreliable (`crates/wire/src/channels.rs:262-267`) | a dropped datagram is a vanished placement | **NO** |
| **R-6** | A landed child's COLLIDER SURFACE (its exterior shell, covering any moving part's swept envelope), plus its centre of mass and inertia, on change | a hull's shard → its parent's shard | HR1 seals the hull's block field; the one-radius law gives the parent a radius only (`crates/core/src/look.rs:37-42`) | a hull lands on its bounding box: it floats over hills, sinks into hollows, rests level when it should tip, and landing legs never touch | **the owner must choose (V60).** Option (c) — the hull resolves its own contact against the parent's terrain — keeps the one-hop rule and moves the question to "who computes the contact" |
| **R-7** | An exact, CHEAP coarse summary at every legal rung, inside the generator crate (**a code ask, not a wire ask**) | the generator → the pyramid | the prune rule compares against it; an octave-dropped answer is not the fold of the fine cells | the pyramid is never sparse: an entry per coarse cell of every touched chunk, at every rung. The whole storage table collapses | **NO.** This is the largest unresolved requirement in the foundation |
| **R-8** | `TAG_SURFACE` in a realm's own look bag — the realm's own `FrameRef` plus the DECLARED generator tag | the realm ITSELF, about ITSELF, to a client already subscribed. It rides `BodyStmt::SelfLook`, which only a RUNNING realm may send | the client cannot know whether the moon's shard runs a generator matching its own build, nor that the moon HAS a surface at all (a hull has none — `ShipLocal` carries no seed, `crates/core/src/pose.rs:87-113`) | the client draws every realm from its seed on faith and draws a hill for a hull that has none; or draws none, which discards SL10 V1.1 | **NO.** Conditions: stated ONCE PER REALM ON CHANGE, carried `Retained`, never on a keep-alive, and gated against the 1 200-byte budget before it ships |
| **R-9** | `ClientControlMsg::HelloWorld { declared, measured }` | a client build → the gateway | the gateway cannot know what generator a client binary holds, nor what its CPU and compiler pair produce | a client draws a hill the shard does not have and finds out when the boots fall through. **SL10 V1.3 asks for this refusal and states, wrongly, that it already exists** — the client handshake carries `coordinate_generation` only (`crates/wire/src/version.rs:337-351`) | **NO** |
| **R-10** | A mesh peer's MEASURED arithmetic profile, on the admin lane after the handshake | node → node | a peer's profile is a property of that peer's binary and chip | two shards of one world compute two different hillsides and neither knows | **YES**, if the declared half rides the ALPN alone and drift is caught by the build gate instead |
| **R-11** | A body's POSE (a turret's angle), as a tagged entry in the realm's own `SceneRow` look bag, under that row's stamp | the owning realm → the gateway → the observers that already receive that row | the client only renders, and a body's angle is live state driven by a motor. The gateway holds no joint | every V2.5 mechanism — a turret, a door, a piston, a rail carriage — is drawn welded shut | **NO** for V2.5. The shape matters: a separate row on the occupant lane puts the one-stamp rule back into care |
| **R-12** | A HUD's value level `{ key, value, state }` on the unreliable latest-wins lane, to the observers holding the attachment's chunk | the owning realm → the gateway → those observers. The realm reads its OWN channel | the value is live state from a signal graph the client does not hold and must never hold | a HUD is a picture that never changes, which is a placeholder, which the no-placeholder rule refuses | **YES for the foundation** — which is why the overlay primitive DRAWS NOTHING until P9 |
| **R-13** | New TLV tags inside `BulkKind::ChunkDelta`: cell records, box records, the attachment list, the body id, object rows and their `apply_tick` | the owning realm → the client, inside the existing arm | an edit, an attachment and a chop are all player state | a placed HUD, hinge, piston, rail and every chopped tree are invisible to every client | **NO.** SL10 V1.6 already approves the DIRECTION; the ask is that these bytes ride at all |
| **R-14** | A new `FallingGroup` entity kind tag, plus its ~16 B parameter blob | the realm's shard → the gateway → the client | a falling crown is live state | a chopped crown vanishes at the cut, which is a black-frame seam | **YES.** It also needs the `RealmAnchored` continuity model, which is NOT built (`crates/core/src/entity_kind.rs:23`) |
| **R-15** | The canopy fold in the pyramid entry — per-coarse-cell canopy occupancy, mean height, mean colour | the realm's shard → the client | the fold must include what players FELLED, which is state | the far rung draws a resurrected forest, or no forest. Both are the arrival-pop seam | **NO**, if trees ship |
| **R-16** | The planet's edit diff for the cells under a surface realm's footprint | a planet → its child area realm | the seed does not know about a tunnel dug last week, and the planet owns the cell | the area builds its pad from the seed alone, so the pad is solid where the client draws a hole | **YES.** It disappears entirely if V75 takes the built-floor branch |
| **R-17** | A SIBLING's placement DOWN into a child, so a ship can hold a contact list | a star system → a hull | a child holds no placement but the stamped reading of its own (SL1 clause 2), and folding an absolute is refused (clause 4) | a ship's contact list is empty: targeting, docking approach and collision warning have no source | **YES for the voxel foundation.** Booked here because SL2's own words ("placements the parent authors and may state") read as if it were built law. It is NOT: `no_realm_inbound_payload_carries_a_placement_or_a_centre` (`crates/wire/tests/intershard_closed.rs:485`) pins the absence, and the `ChildSceneSet` tombstone protects it (`crates/wire/src/intershard.rs:416-440`) |
| **R-18** | A warm LEAD per realm, on the existing send-on-change lane | the owning realm → the gateway → the client | the client may not derive a velocity, and SL10 V1.7 says so by name: *"the client never derives a pose, a velocity, or any state"* | the client's resident band cannot be sized ahead of the boots from anything it lawfully holds, so it either warms too little (the ground is late) or too much (the memory cap) | **REAL ASK, raised in revision 2.** Revision 1 recommended NOT asking, on the ground that the residency rule could read the delivered track's spread. That spread IS a velocity, so the alternative was unlawful, not local. The server already sums a lead of this shape in two places — the reach ruling's *"closing speed × boot time"* (`owner_decisions_2026-09-02_reach.md:184`) and the interest lead (`crates/wire/src/channels.rs:322-328`) — so the datum exists and only the lane is new. **The design can live without it only if the realm's own reach plus the interpolation buffer is enough, which is UNMEASURED** |
| **R-20** | FELT ACCELERATION down to a child — the proper acceleration the parent integrated, in the child's own frame, on change | a hull's parent realm → the hull's shard | a walker inside a thrusting hull has no "down" that the hull can compute: the hull states an acceleration UP and never learns what the parent did with it (the movement contract) | *"walk inside a ship while it flies"* is the core promise of the game, and without it a crewmate floats or falls to the wrong wall whenever the hull burns. Revision 1 carried no row at all (`10_law_audit_stale_register.md:337`, ask 1) | **NO for the walking half of the game.** It is a REALM-TO-REALM crossing and it is the owner's word. *Example: a pilot burns at 3 g out of a moon's well. Her crewmate in the hold must be pressed to the deck, and the deck the hull calls "down" is the parent's answer, not the hull's* |
| **R-21** | BUOYANCY VOLUME and the centre of buoyancy, up on change, beside the mass and the shell | a child realm's shard → its parent's shard | the parent integrates the medium and owns the ambient density; it cannot know how much of the child displaces it | a hull cannot float, sink or trim in an ocean or a dense atmosphere; R-6 carries the shell, the centre of mass and the inertia, and revision 1 dropped this one (`10_law_audit_stale_register.md:339`, ask 3) | **YES for the voxel foundation.** It matters when water and thick atmospheres land, not before |
| **R-19** | One interest radius per open window, gateway → realm | the gateway → the realm | a realm's shard knows where a looker is only if the looker is its own occupant, and SL2 forbids the pose entering it | a watcher hovering just outside a large realm's bound gets a coarser picture than her eyes could resolve | **YES — held in reserve, not asked today** (V89) |

**Refused by name, without an ask:** the grandparent's uniform-gravity field (SL6 default NO and SL2 —
naming the right law matters, because the owner can lawfully approve an SL6 ask and cannot lawfully
approve an SL1 breach); any realm-to-realm chunk FETCH verb; any occupant pose on the forward; the
originating session's server pose (the base's own admission rule, which SL2 refuses).

---

## 6. Disputed items

Where a revised report kept a claim a refuter contested, or where two revised reports disagree.

1. **Two new wire surfaces or three?** The terrain report says two (client-facing plus shard→gateway)
   and disputes the refuter's third, an `InterShardFlow` arm; the storage report asks for two
   `InterShardFlow` arms (the forward and its ack). **Judgement: both are right about different things.**
   No sibling shard reads another realm's cells, so terrain needs no arm; a CHILD forwarding an edit to
   its PARENT does. Count them as R-1, R-2, R-3, R-5 — four surfaces, three of which the design can
   live without if cross-realm edits are refused in v1.

2. **`ShardToGateway::RealmSceneDelta` as the existing path.** A refuter cited it; the terrain report
   answers that it is a TOMBSTONE — "nothing produces it … Do not revive"
   (`crates/wire/src/session_flow.rs:236-248`). **Judgement: the report is right.** Verified in the tree.
   The live shard→gateway scene lanes are `WindowBody` and `WindowMembership`.

3. **The radial gap divided by the local gradient.** A refuter asked for a true signed distance; the
   terrain report refuses, because dividing needs a square root over neighbouring samples, so the stored
   byte would depend on an apron's evaluation order and MORE would sit inside the pinned digest.
   **Judgement: the report is right**, and its own two honest limits must be written into the format:
   the cosine error on a slope, and the loss of accuracy where a tunnel mouth meets a hillside.

4. **The largest legal `N` and the three-step margin.** A refuter said `N ≈ 4.3 × 10⁷` and a 0.22-cell
   residual; the grid report measures `N = 2²⁶ = 6.711 × 10⁷` and 0.348 cells. **Judgement: the report is
   right, and the correction makes the finding WORSE** — the margin against half a cell is 1.4, not 2.4.
   The conclusion (four steps) is unchanged.

5. **When the 20 m quarry vanishes.** A refuter said the 128 m rung; the storage report's fold gives 254
   at 128 m and 255 at 256 m. **Judgement: the report is right about the rung and the refuter is right
   about the defect.** The vanish happens one rung later than claimed, and it still happens.

6. **The rung-7 manifest on a scattered moon.** A refuter estimated 7 MB assuming nearly every chunk
   carries a diff; the storage report shows one 1 m mark against 5.0 × 10¹¹ m³ is far under the
   half-unit threshold, so the entry PRUNES and the manifest is nearly empty. **Judgement: the report is
   right about the number and the refuter is right that it was never sized.** The worst case is the
   MIDDLE rungs, where an edit is just large enough to survive the prune, and the bench must sweep them.

7. **`min`/`max` "met constantly".** A refuter said a generator meets the non-determinism constantly;
   the generator report narrows it: two f64 values that compare equal are the same BITS except `+0.0`
   against `−0.0`, so only a signed zero can move an output byte. **Judgement: the report is right, and
   the narrowing is worse for a gate, not better** — the case is rare and unpredictable, so a golden set
   may never reach it. The fix (remove them from the fence) is adopted in full anyway.

8. **The noise bench's hash.** A refuter said "a different hash function"; the generator report narrows
   it to "a different COMPOSITION of the same avalanche". **Judgement: the narrowing is correct and the
   consequence is unchanged** — the 0.885 ms figure does not transfer, and every number built on it is
   ESTIMATED at one further remove.

9. **The sixth orientation bit's purpose.** The record report reserves it for a future mirror flag; the
   terrain report says a chiral shape must get its twin as a ROW so the collider hull and the winding
   stay baked. **Judgement: they agree on the WIDTH (6) and on the RULE (zero today), so the frozen bytes
   are identical either way.** The owner picks one sentence; nothing in the format moves.

10. **A packed attachment word against a TLV row.** The record report's first revision defined a packed
    8-byte word; the attachments report defines a TLV row with `placed_by` and a fence. The record report
    CONCEDED. **Judgement: the concession is right** — a packed word has no room for who stuck the gauge
    there, and the packed design already carried two artefacts per attachment.

11. **The rotation joint: a MOUNT or a body tree?** The grid report recommends a second flat grid inside
    the realm ("a mount"); the attachments report recommends a BODY TREE with joints. **Judgement: these
    are the same answer at two altitudes.** A mount IS a body: a second grid, carried rigidly by a pose
    the realm authors, in one physics world, in one store. V40 and V41 should be answered together, and
    the store key's `BodyId` serves both.

12. **The mesher's placement.** The generator report's first revision put the extractor outside the
    crate; both refuters showed it breaks SL10 V1.5, and the report moved it in. **Judgement: correct,
    and it is the biggest single change in this synthesis.** If the extractor sits outside, a
    client linking the static library owns a SECOND ground, which V1.2 forbids by name.

13. **An emulated x86-64 green.** The generator report's first revision claimed Rosetta executes x86-64
    arithmetic exactly; revision 2 withdrew it. **Judgement: the withdrawal is right.** Emulation can
    FIND a difference and can never prove its absence, and V1.3 says "every target the game ships on".

14. **"No chunk data crosses a realm boundary, ever."** The storage report's first revision stated it as
    settled; revision 2 reopened it because a landed hull's contact needs the hull's shape.
    **Judgement: reopening is right**, and it is V60 and R-6.

15. **The apron depth.** A refuter caught the depth mixing metres with cells; the terrain report
    re-derived it as `8 · k_rough · skirt_safety` CELLS of the chunk's own rung — 4 cells at
    `k_rough = 0.5`, constant at every rung. **Judgement: the correction stands**, and the old form would
    have sized the residency budget sixteen times too high at rung 4.

16. **The tier field's width: four bits or five?** §2.1 says four, with report 01
    (`01_grid_family.md:690`); report 08's door table says five
    (`08_render_seam_seamless.md:991`). **Judgement: FOUR, and report 08's row is SUPERSEDED.** Four bits
    hold sixteen rungs and the ladder uses thirteen. Report 08's own row says it *"only reads"* the
    number and names the grid report as its owner. Format A's door table now states the width once, and
    no other document may state it again.

17. **"Combined with the carver by `max`."** Report 02's door says the density is combined with the
    carver *"by `max`"* (`02_smooth_terrain.md:625`); §2.2's door says *"by a comparison, never `max`"*.
    **Judgement: they agree on the OPERATION and differ on how a programmer writes it.** V34 removes
    `f64::max` from the fenced float type, because Rust documents it as returning either input when the
    two compare equal — the `+0.0` against `−0.0` case a height field can produce at sea level. The
    frozen bytes are identical either way. Report 02's door text is corrected by this sentence.

18. **DISPUTED — the low-speed leg of slice 8's pop detector.** The law critic's F5 is right that a gate
    may not use a walking dot (`docs/design/owner_decisions_2026-09-05_suit.md` S4) and right that the
    character lands at slice 16. Its fix — move the walking legs to slice 16 — is adopted for the leg
    that needs a CONTROLLER: slice 11's walk off the ramp, and slice 12's foundation sweep. **This
    document keeps a LOW-SPEED leg at slice 8, flown by a HULL at 1.4 m/s.** The evidence: the detail
    ladder's slowest regime is where a crossfade band is widest in time and where a tier pop is easiest
    to see, and nothing about that leg needs a character — a hull at 1.4 m/s is the shipped path S4 names
    (*"berth a hull, board it, push"*). Deleting the leg outright would leave the ladder's slowest regime
    untested from slice 8 to slice 16. Slice 16 re-runs the same detector under the real character.
    *Example: a pilot noses a hull along a ridge at a walking pace. Every tier boundary she crosses is on
    screen for seconds, so a pop that a 240 m/s dive hides is unmissable.*

---

## 7. UNMEASURED

Every number the design depends on that has no measurement, ordered by the slice that needs it.

**Before slice 2 (the geometry seam).**

| # | Unmeasured | The bench |
|---|---|---|
| U-1 | The inverse warp's residual in CELLS at `N = 2²⁶`, inside the crate in Rust, for the frozen guess and count | the crate's own round-trip test; the python figures are the requirement, the crate's number is the gate |
| U-2 | `k₁ + k₂ + k₃ == 1.0` exactly, in the order the warp evaluates it, on x86-64 AND aarch64 | a `const` assertion inside the crate. MEASURED so far only in python3 on one target |
| U-3 | The tangential cell-edge spread under the chosen warp, measured by OUR crate | a sampled arc-length sweep. The 0.7071–1.0050 m figure is python, not this tree |
| U-4 | `addr_of` cost with four Newton steps, and calls per tick on a planet with a hundred lookers | a microbench; it must show O(1) and off the containment path |
| U-5 | The look-shell move on THE world when the ladder snaps | the largest radius change over the generated forest; it must stay under one top-rung cell for every body |
| U-6 | An exhaustive round trip over a hull's slot box on BOTH sides of the origin on all three axes | the negative-half test |

**Before slice 5 (the generator).**

| # | Unmeasured | The bench |
|---|---|---|
| U-7 | **x86-64 against aarch64 byte equality of the generator AND the extractor.** Never measured anywhere in this project | the no-drift gate: emulation first (necessary, not sufficient), then real hardware |
| U-8 | opt-level 0 against 3, debug against release, stable against the coverage nightly, `target-cpu=native` | the same gate, more legs |
| U-9 | Whether clippy `disallowed-methods` resolves a primitive inherent path such as `f64::sin` | inject `x.sin()`, run `just lint`, expect red. If it cannot, the fallback is a 30-line source scan |
| U-10 | Whether `f64::sqrt`/`floor` are callable under `no_std` on stable 1.94.1, and what a `libm`-class dependency would cost | one `cargo check` on a stub |
| U-11 | Per-chunk generation cost under the MANDATED hash, at three cave-lattice steps, with a picture of a lava-tube mouth at each | the noise bench re-run inside the crate. The 0.885 ms figure used a different hash composition |
| U-12 | `mul_add`'s bit-identity on both targets | the gate, with one `mul_add` in the warp |
| U-13 | The boot self-check cost on the slowest shipped client | time it in `vdctl gen-digest` |
| U-14 | **The exact coarse summary equals the fold, and its cost** | 10 000 random coarse addresses at every legal rung: compare the generator's coarse answer with the fold of its children, and time both. Zero differing entries; the coarse call must not cost 8^L evaluations. **This gates the whole storage design** |
| U-15 | `band_L ⊇ band_0` and the rung-disagreement bound on the SHIPPED octave table | two property tests |

**Before slice 6–8 (the extractor, the link, the ladder).**

| # | Unmeasured | The bench |
|---|---|---|
| U-16 | **The extractor's cost per chunk and its output size.** ESTIMATED 0.3–1.5 ms; never measured. It gates the client's residency LEAD constant | extend the bench with the surface-nets pass |
| U-17 | Whether the COMPOSED chunk (generated + edit + block + sub-block) is byte-identical on two hosts | the composed row in the golden set |
| U-18 | The cost of linking the crate into the shipped client — binary size and cold-build time, on both targets | move the dependency and measure |
| U-19 | The vertical disagreement between rung L and rung L+1, in PIXELS at the switch distance | 10⁵ directions; report max and p99. ESTIMATED 4 pixels, which is visible |
| U-20 | The chunk ARRIVAL RATE during a descent, against a NAMED thread budget | a scripted descent through four rungs; requests per second and the deepest queue |
| U-21 | The client's generation budget on ONE core and on eight, at 240 m/s and 528 m/s | a ring benchmark. MEASURED precedent: 6 s on one core, 0.75 s on eight, for a 6 800-chunk view |
| U-22 | The column FLUX at a stated speed (not the residency — 379 columns per rung is a RESIDENCY) | columns entering a rung's annulus per second |
| U-23 | The reference view. THREE exist in the tree and the base: 45°/720 rows, 70°/1920 px, and an angle knob | one must survive, and every rung range must be re-derived from it before any residency number is quoted |
| U-24 | A one-pixel star's PAINTED luminance under the chosen anti-aliasing | the star probe. **A precondition of the anti-aliasing door**, not a follow-up |

**Before slice 9–12 (the store, the edits, the square lanes).**

| # | Unmeasured | The bench |
|---|---|---|
| U-25 | Store size for 100 M SCATTERED edits, and the pyramid's entries per edit | write 100 M marks 57 m apart; run the checkpoint |
| U-26 | Write amplification of one pyramid entry in a built region | redb with real record sizes |
| U-27 | Pyramid tail replay after `kill -9`, byte-identical to a rebuild | the process-tier crash proof |
| U-28 | The walk-up cost per edit at 13 rungs with the effective-summary exit | it exists only if U-14 passes |
| U-29 | Chunk interest cost at scale, INCLUDING the new `(chunk, rung)` → sessions index that does not exist | 128 observers, 10 000 dirty chunks in one tick |
| U-30 | Manifest size, the SCATTERED case, swept at every rung | the middle rungs are the worst case |
| U-31 | One window, many sessions: the moon's egress at the floor and the gateway's per-session drop rate | one close pilot and 127 distant ones behind ONE gateway |
| U-32 | The regime hand-over: the drawn rung per chunk per frame across a walk-out from a hull | no rung goes backward; no chunk is undrawn for one frame; the cache survives |
| U-33 | The fly-away-and-back drawn step at every rung boundary, against the drawable floor | over a tunnel, a filled quarry and a built tower |
| U-34 | Whether two overlapping colliders under a foundation are walkable | a sweep across a foundation cut into a 30° slope at 20 offsets |
| U-35 | The FRAME cost of one cell at 512 eighth-scale sub-blocks | collider build, remesh and upload, on a named thread, against **16 ms of worker time** (not the base's 33 ms, which was a CUBE-face remesh budget). **★V7's STEP stays provisional until this runs; the WIDTH is frozen already** |
| U-36 | The overlap validation cost at store open | 40 000 chiselled cells |
| U-37 | The observer's record lane under a berth ring of twenty 50 000-block hulls | bytes per second on one observer's lane, and the tier at which records STOP |
| U-38 | The forward's idempotency state after 200 edits/s for an hour | it must be ONE row per origin realm |

**Before slice 11 and 13–16 (colliders, joints, trees, the character).**

| # | Unmeasured | The bench |
|---|---|---|
| U-39 | Trimesh chunk collider build time and bytes, in BOTH float flavours | 1 000 seed chunks, release |
| U-40 | The drawn triangle against the collided triangle at a planet's radius, with and without the chunk-local origin rule | ESTIMATED failure without it: ~0.76 m in f32 |
| U-41 | The extractor's SNAP error distribution | 10 000 surface crossings |
| U-42 | Collider build rate under a hull at 30, 250 and 7 000 m/s, and at 100 km altitude; **extended: the speed at which a hull crosses a WHOLE CHUNK in one tick, and the analytic segment query that must answer beyond it** | the frontier must never be late; the 100 km run must demand ZERO chunks; and past the whole-chunk speed the swept test must resolve with no chunk built, because a re-clamp is a jump |
| U-43 | Continuous-collision cost, and CCD against a MOVING kinematic trimesh | if the second fails, the spinning-frame decision falls back |
| U-44 | The collider cache's bytes against 1/100/600 PARKED hulls in one planet realm | per-tick core time must NOT grow with the parked count |
| U-45 | A standing boot on a spinning planet: drift per 1 000 ticks | it must stay under one fine cell |
| U-46 | A hull with 1/8/64/512 bodies, 0/4/N moving: shard tick time and bytes per tick; and the CLIENT's frame time and batch count | never measured on either side |
| U-47 | Cross-target joint determinism through the carried sine/cosine table | the same joint at the same angle, byte-identical pose on both targets |
| U-48 | A gunner keeps contact with a turning platform | no collider, no shape cast and no character controller exists today, so this is machinery P5 owes |
| U-49 | `expand` per tree per kind at the largest stage; the object-horizon disc's wall time | the tree bench |
| U-50 | Stored collision bytes and broad-phase query time per 64 m disc of forest | ESTIMATED 4.49 B/m² against the base's 199–650 B/m² |
| U-51 | Entries per felled tree, and per mined cell under a standing tree | fell 19 406 trees; then mine under 1 000 standing ones |
| U-52 | The falling collider's lag behind the drawn crown | ESTIMATED 2.4 m; the measurement decides whether the pusher ships |
| U-53 | The object-horizon crossfade: silhouette area and mean colour across the crossover | a 20 m oak driven across it |
| U-54 | The client's smooth re-extract of one edited chunk | **no click-to-pixel figure may be quoted to the owner before this runs** |
| U-55 | A moon holding one million growing trees and NO observers: per-tick work | it must be zero |

**Rows revision 1 dropped, restored here.** Each names the report that owes it.

| # | Unmeasured | The bench | Who needs it |
|---|---|---|---|
| U-56 | The C-ABI copy cost per chunk. The payload is ESTIMATED ~715 KB packed, not 476 KB (`03_generator_sl10.md:659`) | time one copy across the C interface at the packed size | V2.8's other-engine interface; slice 7's seam |
| U-57 | The k3d image architecture on this host (`03_generator_sl10.md:660`) | read the image manifest of the running pods | whether the dev cluster ALREADY runs two architectures, which is the no-drift gate's first real leg |
| U-58 | `verify-pyramid` on a 100 M-edit realm; the base's 54 s is an estimate and the suite runs it per scenario (`06_storage_diff_lane.md:783`, M-7) | build the realm, time the rebuild | slice 9's gate is written against `verify-pyramid` |
| U-59 | The realm store under an edit storm, and under a STALLED disk (`06_storage_diff_lane.md:786`, M-10) | fault-inject the disk under 200 edits/s | the typed refusal and the parked-tick counter: what every builder on a moon meets when the disk stalls |
| U-60 | Palette width over WHOLE-CELL records on a 50 000-block hull (`04_block_record_registry.md:801`, M6) | count distinct kinds per chunk | the chunk delta's compaction crossover |
| U-61 | The style-detail function is byte-identical on both hosts (`04_block_record_registry.md:802`, M7) | the no-drift gate, extended to the trim function | V2.6's client-derived detail; without it two players side by side see two hulls |
| U-62 | The content bake assertion that `volume_units(kind)` is a multiple of `1 << (3 × max_scale)` (`04_block_record_registry.md:812`, M15) | a bake-time check over the catalogue | a thin decorative fin whose volume shifts to ZERO at eighth scale |
| U-63 | Rapier snapshot and restore across targets (`09_physics_controller.md:812`, S8) | snapshot on one target, restore on the other | how a checkpoint travels; Category C forbids cross-host re-simulation |
| U-64 | The interpolation-buffer falsifier on today's client (`09_physics_controller.md:814`, S10) | the report runs it *"before any latency figure is quoted"* | this document quotes latency figures |
| U-65 | What the pinned `parry` actually holds (`09_physics_controller.md:819`, S15) | read the pinned version's feature set | V27's comparison keeps an UNMEASURED column |
| U-66 | ONE integrator, no bend at a crossing — a hull under constant drive crossing from a stub realm into a rapier realm (`09_physics_controller.md:822`, S18) | fly the crossing under constant thrust and measure the path's curvature | a JUMP seam at the shell of every realm that gains physics |
| U-67 | The window-versus-capture pixel diff at one tick (`08_render_seam_seamless.md:872`) | capture both paths on one tick and diff | HR6 says the harness captures what the player sees; nobody has measured whether the two paths draw the same image |
| U-68 | The atmosphere's cost, ESTIMATED 0.70 ms (`08_render_seam_seamless.md:876`) | GPU timing on a landing | the client's frame budget on a landing |
| U-69 | The residual sky error under the `+Y` flat-ray approximation (`08_render_seam_seamless.md:879`) | compare against a curved-ray reference | V87's atmosphere hand-off rests on it |
| U-70 | Resident chunks of a 100 km view WHILE MOVING, against the ≤ 7 500 chunk cap (`08_render_seam_seamless.md:862`) | a scripted flight, counting residents per frame | the client's memory cap and the crossfade band fraction |

**Rows this revision adds.**

| # | Unmeasured | The bench | Who needs it |
|---|---|---|---|
| U-71 | The SHELL-SWAP settle: the vertical movement per tick when a plate is welded onto a parked hull's belly with a crew member aboard | S14 — 100 repeats, the maximum per-tick vertical movement against the realm's stated cap | slice 12; a one-metre lift in one tick is a JUMP and SL8 refuses it |
| U-72 | The ANALYTIC swept query's cost: the surface along a 500 m segment, cell by cell, with no chunk built | a microbench at 30, 250, 7 000 and 10 000 m/s | slice 11; it is the answer that replaces a clamp, so it must be affordable at every legal speed |
| U-73 | The block store's CUT duration for a 50 000-block hull re-homing to another node, and the tick hitch a watching client sees | the transfer gate with a built hull | slice 15; HR1 makes the store travel with the realm |
| U-74 | The upward statement COUNT while 1 000 blocks are placed at ten per second | slice 10b's coalescing gate | the movement contract refuses a per-tick facts lane by name |
| U-75 | Whether the realm's own reach plus the interpolation buffer alone sizes a resident band that contains the collider set at every legal closing speed | a scripted approach at 30, 250 and 7 000 m/s, counting frames where the band is short | **it decides R-18.** If the answer is no, the warm lead is a real SL6 ask |
| U-76 | Where a themed client refusal lands under V15 option (B): the count of chunks refused per flight when a client lacks one kind | fly a themed moon with a stale registry | V15; it is the seam a pilot meets when a theme ships |

---

## 8. The stale-text register summary

These claims of the investigation base are refused BY NAME by a later ruling or by the code. The
synthesis must not promote any of them, and none appears above except as a refusal.

**Refused by a ruling.**

1. *"No smooth or melted terrain. No marching cubes, no surface nets, no dual contouring."* — refused by
   V2.1, for TERRAIN. It stands for built blocks.
2. *"The base cell is never subdividable"* / *"below the catalogue's smallest extent there is no cell and
   therefore no collider, ever"* — refused by V2.4. The base's own escape (a strictly additive finer
   layer) is the shape that fits, and a sub-site of `(0,0)` is exactly today's whole cell.
3. The secret-keyed deposit: `gen_concealed(secret, key)`, `concealed_key_id`, a master key in a
   deployment store — refused by the 2026-08-27 seed ruling and by SL10 V1.6. A deposit placed by
   `f(position, secret)` is a static map, and secrecy is not the cure.
4. *"Give the Cartesian profile a TEST-ONLY generator emitting three boulders"* — refused by SL5 (one
   world, no test-only variant) and by "test exactly production".
5. The governor, the target speed band, the derived envelope AS A CAP, the hard limiter, and the "minimum
   drawn terrain radius ≥ v_max × 4 s" — refused by the 2026-08-27 movement answers. **⚠ THE CAP IS
   STILL IN THE CODE**: `crates/sim/src/stub/dot.rs:457-474` scales the stick through
   `governed_ceiling_for_frame` and `ramp_cap_mps` every tick (`crates/core/src/flight.rs:93,105,118`).
   Its deletion is deferred with the suit. **No slice above may size any number on those symbols.**
6. *"Absolute positions since the A5 flip"* as a distance source — refused by SL1 clause 4.
   `pin_abs` and `anchor_epoch` were REMOVED at wire minor 8 (`crates/wire/src/version.rs:67-79`).
7. The grandparent's uniform-gravity field — refused by SL6 (default NO) and SL2, not by SL1 clause 4.
   Naming the right law matters: the owner can lawfully approve an SL6 ask.
8. Thrust in newtons per tick; a downward ambient-density feed; a 1 Hz heartbeat of what-I-am facts; a
   per-tick aero struct — refused by the movement contract. What crosses up is an ACCELERATION and a
   torque, six numbers, in the child's frame; facts cross ON CHANGE.
9. *"The realm proxy is the parent's marker"* / *"tier 0 is the parent-authored marker"* — refused by SL3
   and by the reach ruling. The marker is deleted; the coarsest rung is the realm's OWN look.
10. The composite pattern grammar, the 998-cell tree, the bought-greybox tree, and the emit-exclusion
    hook — refused by V2.2. A tree is ONE object.
11. *"The client may not derive; that door does not close again"* — superseded by SL10, for the STATIC
    SHAPE only. The star field stays shipped once.
12. *"The server collides on the generator's shape"* — refused: the collision surface is the derived
    shape COMPOSED WITH THE DIFF, on both hosts, before either draws or collides.
13. *"A parent may state a sibling's placement to a child"* as built law — REFUSED-UNTIL-ASKED. The lane
    does not exist and a passing test pins its absence.

**Refused by the code.**

14. *"27 arms + 3 reserved = 30 is the review ceiling; there is no headroom"* — MEASURED: 44 arms
    (`crates/wire/src/intershard.rs:124`). The successor is the approval-citation build test
    (`crates/wire/src/version.rs:800`), not a count. The header prose (16) and `lib.rs` (24) are also
    stale and owed a re-sync.
15. *"The wire already carries a `coarsen_level` precision ladder"* — the field survives only on a
    TOMBSTONE payload nothing produces, beside a field documented as "the SL2 breach that condemned it"
    (`crates/wire/src/intershard.rs:1173-1186`). Reading it as room to build in would resurrect the
    breach.
16. *"`BlockEdit` is a reserved wire arm"* — it is a doc-comment NAME, not a variant
    (`crates/wire/src/intershard.rs:34`), and that contract does not reach a client in any case.
17. *"The `CellIndex` linearisation door is ALREADY SPENT"* — MEASURED: no grid, chunk or block module
    exists. Nothing is spent. Confirm it deliberately.
18. *"The durable placement record is five bytes"* — superseded twice, to eight and now to twelve. The
    roadmap line that says five is a one-line edit the owner must approve.
19. *"The pyramid entry is PERSISTED, PERMANENT, a ONE-WAY DOOR"* — it is DERIVED and rebuildable from
    tier 0. A soft door on disk, a protocol door on the wire.
20. *"The walk-up exits at rung 1 or 2 for an edit in solid rock"* — false as written: the exit compared
    against the STORED entry, which is absent for an unedited parent, so every edit walked every rung.
21. *"Collider residency is 64 m, derived as greater than any body's per-tick swept motion"* — the
    quantity no longer has a value: the ceiling left the flight path and containment became swept.
22. *"Everything inside one chunk-width of the camera is tier 0"* as the client's floor — 62 m is
    narrower than the base's own 64 m collider ball, which is how the residency invariant went missing.
23. *"`FrameSpace` / `feature::voxel::register` exist"* — MEASURED: no such type and no such module. Six
    doc-comment sites in three files name them.
24. *"The planet radius is per-body data on a 39.47 m ladder"* — the code draws a CONTINUOUS radius from
    the mass–radius law (`crates/physics/src/taxonomy.rs:653`,
    `crates/physics/src/worldgen/generate.rs:1232`); the 39.47 m step is an artefact of forcing whole
    tangential chunks.
25. *"A full-depth body's `m` is a power of two"* — refused: it forbids realistic radii, which the owner
    asked for.
26. *"The world-generation tag already refuses a client whose generator disagrees"* (stated inside SL10
    V1.3 itself) — MEASURED: the tag refuses a PEER on the ALPN and a FILE on the stamp; the player
    handshake carries `coordinate_generation` only (`crates/wire/src/version.rs:337-351`), and
    `crates/client` holds zero hits for `world_generation`. **The client refusal must be BUILT** (R-9).
27. *"The noise bench inlines exactly the repo's `SplitMix64`"* — the bench's mixer is the FINALISER
    alone and its composition adds its own per-axis multiplies; the repo's stream generator adds the
    golden increment first. Two different functions, so the bench's 0.885 ms is a PROXY.
28. *"A regrown forest and a refilled hole leave nothing"* — a `Placed` cell can NEVER equal a `Terrain`
    cell, deliberately, so a refilled hole is PERMANENT. The forest prunes; the hole does not, and that
    is why dig-and-fill needs a rate gate rather than a prune rule.
29. *"240 m/s is the mesher-bound flight ceiling"* — it is the terrain UPLOAD bandwidth ceiling at the
    old vertex format. The generation-and-mesh ceiling is 528 m/s pessimistic, and the pop detector must
    run at BOTH.
30. *"~379 new columns per rung per ring step"* — 379 is a RESIDENCY, not a flux. The flux is unmeasured.
31. *"cross-platform float determinism explicitly not pursued"* (`docs/design/PLAN.md:135`) — superseded
    by SL10 V1.3: no drift is a byte-for-byte MEASURED gate on every target.
32. *"`mul_add` is bit-exact and allowed"* — SL10 V1.4 forbids the contraction; the explicit call is a
    different thing, the two texts disagree, and the call stays OFF the fence until it is measured.
33. Every physics figure quoted "at 20 Hz" — the tick rate is a per-shard knob
    (`crates/core/src/kinematics.rs:148-153`) and the dev cluster runs 50 Hz
    (`crates/bins/src/lib.rs:412-414`). Each figure must be re-derived at the shard's own rate, never
    scaled by one constant.
34. *"`PrimTransform` has no rotation; a rotating planet cannot be drawn"* — the field landed and the
    renderer applies it (`crates/client/src/realm_scene.rs:783-790`;
    `crates/client-render/src/lib.rs:1569`). ⚠ the doc comment two lines above the field still says "with
    no rotation" and is itself stale.

**Base mechanisms that SURVIVE, and that this synthesis promotes.** A stale register must also say what
a ruling KEEPS: the float newtype that makes a transcendental unreachable BY CONSTRUCTION (SL1 clause 5
demands a structural fence with an observed-failing control, never care — and a byte gate DETECTS drift,
it does not EXCLUDE it); the drag term the parent computes from the child's stated mass and drag
coefficient; the terrain-following autopilot block as software pressing the stick; the pop detector as
the instrument every seam tolerance is measured with; and the ~20 square building shapes with their
catalogue.

---

## Revision log

**Revision 2, 2026-09-07.** Two critics read revision 1: the LAW critic
(`verdicts/synthesis_law.md`, twenty findings) and the COMPLETENESS critic
(`verdicts/synthesis_completeness.md`, twenty-two). Every finding is listed below with what this
revision did. Where this document keeps its text, the row says DISPUTED and points to the evidence.

### The law critic's findings

| # | The finding | What changed |
|---|---|---|
| F1 | BREAKS SL6 — the sequence never asks the owner the §5 questions it wrote | **FIXED.** Slice 0 gains a second half: the owner answers every §5 row YES or NO by name. Slice 4 may plant ONLY the rows recorded YES, and a planted arm with no approval is a defect the closed-set test names. §5 gains a bold ★ line saying so, and §2's intro lists it first |
| F2 | BREAKS SL10 V1.7 — the client derives a velocity to size its own band | **FIXED.** V88 is rewritten: the band is sized from SERVER-STATED data only — the realm's own reach, the interpolation buffer, and a lead the owning shard states. §1 and slice 8 are rewritten with the same rule and cite V1.7's own words. R-18 becomes a REAL ask, and U-75 measures whether the design can live without it |
| F3 | BREAKS HR4 — five slices land a feature with no two-shard-kind gate | **FIXED.** §3's preamble states HR4 as a rule binding every feature slice and cites the profiles (`crates/sim/src/capability.rs:246,262,278,290`). Slices 9, 10, 12, 14 and 16 each name a PAIR and an identical fixture; slice 11 gains one too; the new slice 10b names one. ★V38 moves into band A |
| F4 | BREAKS SL6 AND SL3 — R-4 asks for a rung floor after the document found the local rule | **FIXED.** R-4 is WITHDRAWN and marked so. V89 is widened to apply its local rule to the rung floor as well as the interest radius, and it now names SL3 and the reach ruling's "the realm's choice, never a parent's clamp" |
| F5 | BREAKS THE SUIT RULING — two gates need a character that lands nine slices later | **FIXED, with one DISPUTED part.** Slice 11's walk off the ramp and slice 12's foundation sweep move to slice 16, where the character lands. **DISPUTED:** slice 8 keeps a LOW-SPEED leg, flown by a HULL at 1.4 m/s, which is the shipped path S4 names and needs no controller; deleting it outright would leave the ladder's slowest regime untested for eight slices. §6 item 18 records the dispute and its evidence |
| F6 | MISSING — an edit changes what a realm IS, and nothing carries the change up | **FIXED.** New **slice 10b** derives mass, cross-section, drag coefficient and look radius from the realm's own block field, states them on a SETTLE and never per placement, and gates the count (U-74) |
| F7 | MISSING — `BodyId` sits in a persisted key and nothing says how one is allocated | **FIXED.** §2.1 states the rule (body 0 = the realm's own grid; a monotone per-realm counter, never reused, persisted); Format A's door table gains the row; ★V104 puts it in band A |
| F8 | MISSING — Format C's reserved bits are not budgeted against the canopy fold | **FIXED.** §2.3 carries a three-spend table (8 + 1 + ~8 bits against 14) and three options; ★V102 asks the owner which |
| F9 | MISSING — the quad-to-triangle rule is not frozen | **FIXED.** The triangulation rule lives in the generator crate (§2.4, slice 6's *lands*), and slice 6 now pins a **TRIANGLE** list, not a quad list |
| F10 | MISSING — nothing keeps client-derived trim off the collided surface | **FIXED.** Slice 7 states the rule (trim never changes the extractor's output, never changes a collider, never moves a shared vertex) and gates it by comparing the seam's geometry with the collider's. V72 carries the rule too |
| F11 | BREAKS THE NO-MAGIC-NUMBERS RULE — the object horizon is a constant | **FIXED.** V80 no longer recommends ~800 m. The horizon is DERIVED per object kind and per growth stage from the object's own size and the drawable floor (`crates/core/src/geometry.rs:1156-1173`), and the row is marked PENDING U-53 |
| F12 | MISSING — the chunk order and the crossfade band are not placed in the engine-free library | **FIXED.** Slice 7 states the split before either encoder is written (library: order, tier, band; engine: thread, shader) and slice 8's *lands* is re-headed "ALL of it in the client LIBRARY". V46 is widened from the tier alone to all three |
| F13 | MISSING — a swept collider demand has no bound, and a clamp is refused | **FIXED.** Slice 11 gains the analytic segment rule: the shard walks the segment through the crate cell by cell, builds no chunk, and builds the ONE chunk a contact lands in. §1 carries the same sentence. U-42 is extended and U-72 measures the query |
| F14 | MISSING — the registry digest makes every new theme a fleet-wide flag day | **FIXED.** ★V15's recommendation CHANGES from equality to a PREFIX digest plus a per-chunk refusal, and the row states the V2.7 cost of equality. §2.2's recommended answers follow. U-76 measures the refusal's reach |
| F15 | MISSING — a realm's block store must travel with the realm, and the cut is not sized | **FIXED.** Slice 15 gains the rule (the bulk moves BEFORE the cut; the cut carries only the tail) and the measurement (U-73: the cut duration for a 50 000-block hull, and the tick hitch) |
| F16 | MISSING — HR5 never gets a tier for the new crates | **FIXED.** Slice 5 states `vd-terrain` and `vd-seed` as Tier-A at 100 % and adds `just coverage-fast`; slice 7 does the same for `vd-client` |
| F17 | UNMEASURED STATED AS MEASURED — the write amplification and the quarry rung | **FIXED.** Both are re-labelled ESTIMATED, with the source cited and the honest ground for V52 stated instead (the pyramid is rebuildable). U-26 owes the run |
| F18 | MISSING — seed-decided common ore needs the owner to relax S5.1 by name | **FIXED.** V36 is starred, says in its title that option (B) asks the owner to relax S5.1 by name, and carries the seed ruling's own public-wiki test |
| F19 | MISSING — three slices adopt an SL6 request without naming SL6 | **FIXED.** Slices 10, 13 and 14 name SL6 and cite the R-numbers they consume (R-3/R-5/R-13; R-11; R-13/R-14/R-15). Slice 4 states what it may plant and what the skip-unknown rule does and does not cover |
| F20 | MISSING — slice 7 draws terrain before slice 8 gives it a detail rule | **FIXED.** Slice 7 states that it draws ONE rung inside one chunk band, and its gate asserts that no second rung is drawn |

### The completeness critic's findings

| # | The finding | What changed |
|---|---|---|
| F1 | The write amplification is called MEASURED and nobody ran it | **FIXED** — see the law critic's F17 |
| F2 | A felled tree costs eight bytes and the frozen record is twelve | **FIXED.** §1 and slice 14's gate say TWELVE bytes; the clear-cut disc is re-derived at 233 KB (ESTIMATED, arithmetic on 12 B) |
| F3 | The formats hold no NEGATIVE entry | **FIXED.** §2.2 states the removal encoding (an `Empty` KIND record with provenance `Placed`, every other field zero), Format B's door table gains the row, and ★V101 carries the storage report's D-8 into band A |
| F4 | A sub-metre block on a smooth slope has no seat | **FIXED.** §2.2 states the seating rule; §2.4 freezes it; slice 6 lands it and gates it with a seating row in the golden set; ★V106 asks the owner |
| F5 | Two starred format doors shut before the measurements that decide them | **FIXED by splitting each answer.** ★V7: the WIDTH freezes now (16 bits, which hold 1/16 m either way), the STEP ships PROVISIONAL at 1/8 m, with a stated re-open rule. ★V20: (A) PROVISIONAL, with the re-open cost stated on both sides of slice 14 |
| F6 | Slice 17's acceptance gate cannot go red | **FIXED.** Slice 17 carries a twelve-row tolerance table copied from report 08 §3.3, each row marked, and states that an ESTIMATED row still unmeasured at gate time means the slice is not complete |
| F7 | "Edit echo to pixels ≤ 33 ms" is a budget from a different mesher | **FIXED.** Slice 10 gates the CLIENT re-extract at ≤ 16 ms of worker time and states NO whole-path threshold, citing report 09's refusal and this document's own U-54. Slice 12's reference to the 33 ms budget goes with it |
| F8 | Slice 7 links a motion crate into the client | **FIXED.** Slice 7 links `vd-terrain`; `vd-physics` STAYS a dev-dependency; the crate-isolation test gains a row that refuses a client dependency on it. The file's own words are cited (`crates/client/Cargo.toml:20-22`) |
| F9 | Nothing puts the SHARD's chunk build off the tick thread | **FIXED.** Slice 11 states the rule, the tick-boundary insertion and the frontier rule; V103 asks the owner for the SHARD's pool mechanism, naming `bevy_tasks` (`Cargo.lock:544`) and saying rayon is a new library |
| F10 | The SL9 sleep rule for parked hulls disappears | **FIXED.** Slice 11 lands the sleep rule and gates it with S12 (1 / 100 / 600 parked hulls, per-tick core time must not grow) |
| F11 | The shell-swap pop rule and its settle cap are dropped | **FIXED.** Slice 12 lands the shell swap with a CAPPED penetration-correction velocity, states that the cap is a FIELD of the realm's own record, and gates it with S14 (U-71) |
| F12 | Five format-level one-way doors are absent from §2 | **FIXED.** Format B's door table gains the tree SKELETON format, the BOX record's 20-byte layout, the realm store's FAMILY PREFIX BYTES, the scoped channel key and the joint ANGLE GRID, plus the face-numbering tie to the orientation basis |
| F13 | The generator's arithmetic profile is a FOURTH freeze | **FIXED.** New **§2.4, Format D — the generator's world identity**, with its contents, its append-only octave rule, its door table and its cost |
| F14 | Slice 5 pins the golden set before trees exist | **FIXED.** §3's preamble states rule 1 (no store that must survive is written before the last slice that changes the seed's shape); slice 5 marks its literals PROVISIONAL; slice 9 marks its store throw-away |
| §2 | Nine dropped register rows, plus V78/V80 recommended on estimates, plus the V2.6 reading | **FIXED.** ★V101, ★V102, ★V104, ★V105, ★V106, ★V107 and V103, V108–V112 are added. V78 and V80 are marked PENDING their measurements |
| §3 | Slices whose gate cannot go red, or whose measurement is not a number | **FIXED.** Slices 1, 3, 5, 6, 7, 8, 10, 10b, 11, 13, 15, 16 and 17 carry pass numbers. Slice 5's inherited 1.20 ms is re-derived from the frontier requirement. Slice 16 names the shard's physics budget as a FIELD of the shard's own profile record, cited by slices 11 and 13 |
| §4 | Three cases with no home | **FIXED.** The dig-during-a-hand-over fixture joins slice 15's gate; the tree-support case joins slice 14's gate; report 09's sphere-of-influence finding — *there is NO crossing at the ground* — is stated in §1 and repeated in slice 11 |
| §5 | Fifteen dropped UNMEASURED rows | **FIXED.** U-56 to U-70 restore them, each citing the report that owes it; U-71 to U-76 add the rows this revision creates |
| §6 | V2.1's PLACING half and V2.6's style slice are thin | **FIXED.** Slice 10 lands the placing half (a placed terrain-form cell carries a full density; a player may raise ground; V67 refuses only the square lane). Slice 12 lands the client's style derivation, and U-61 measures that it is byte-identical on both hosts |
| F15 | The tier field is four bits here and five in report 08 | **FIXED.** §2.1 states FOUR and names report 08's row superseded; §6 item 16 records the reconciliation |
| F16 | Report 02's door still says the carver combines by `max` | **FIXED.** §6 item 17 states that the two agree on the operation and differ on how it is written, and corrects report 02's door text |

### Corrections this revision made on its own

1. §5's preamble named the realm-to-realm crossings as "R-1, R-2, R-6 and R-9". R-9 is a client build
   talking to the gateway and is not a realm crossing; R-16 and R-17 are crossings and were omitted. The
   list now reads R-1, R-2, R-6, R-16, R-17, R-20 and R-21.
2. §5 gains **R-20** (felt acceleration DOWN to a child) and **R-21** (buoyancy volume UP on change),
   which report 10 owes as asks 1 and 3 and which revision 1 dropped. R-20 carries the core promise of
   the game — *walk inside a ship while it flies* — so it is marked **NO** for the walking half.
3. The completeness critic's fix for F14 named slices 9 and 10 as throw-away. Slices 11, 12 and 13 also
   write stores, so §3's rule 1 covers slices 9 to 13.
