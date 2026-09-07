# 04 — The saved block record, the registry, and the room for parameters, style and themes

**Date:** 2026-09-07. **Revision 2** (after the law refutation `verdicts/record_law.md` and the
feasibility refutation `verdicts/record_feasibility.md`). **Domain:** the SECOND format the owner freezes
(V3.2): the saved block record.
**Serves:** V2.1 (a smooth surface that a placed cell reshapes), V2.2 (a tree as one object that carries
its own shape), V2.4 (sub-metre blocks), V2.5 (attachments), V2.6 (params + client style), V2.7 (themes),
V2.8 (an engine-free record), V2.9 (SL8), under SL10 and every earlier law. **Status:** an investigation
report. Nothing here is binding until the owner rules and the result moves to `docs/design/`.

**How to read the numbers.** Every number carries a mark. MEASURED means a command or a file read
produced it today, and the method is named. ESTIMATED means this report did the arithmetic by hand
against a cited source, and no code ran. UNMEASURED means nobody has the number. A number with no mark
is a defect in this report.

**How to read the code citations.** `file:line` names the worktree at
`/Users/maxim/Projects/my/voxeldust/.claude/worktrees/voxels`. The code is the only current truth. The
investigation base (`docs/investigation/*.md`, 2026-08-03/04) is an input, not a decision.

**What revision 2 changed.** Section 13 is the revision log. The two largest changes: the cell record
grows from 8 bytes to 12, because the smooth-terrain domain needs a density byte on every planet cell
and the tree requirement needs an object parameter; and the packed attachment word is WITHDRAWN in
favour of the attachment domain's TLV row.

---

## 0. The recommended design in one page

A saved cell record is TWELVE bytes: a 64-bit address-and-identity word and a 32-bit attribute word. It
records what a player changed and nothing else. It never records what the seed decides (SL10 V1.6): a
mountain has no record, a trench cut into the mountain has one record per cut cell. It never records
what changes every tick (damage, heat, fluid, a joint's motor angle): those live in sparse side tables,
so the permanent log is never rewritten when a block warms up.

The first word holds the cell address inside the chunk, the kind, the orientation, the provenance and
the sub-site. The second word holds the style variant, the terrain density that makes the surface
smooth, and one object parameter that a large landscape part (a tree) states about its own shape. Both
words keep reserved runs that a decoder refuses when they are not zero.

The store key is `(body, chunk)`. A body is a rigid grid inside one realm — the hull's own grid is body
0, a turret joined to it is body 1. The record's cell field addresses inside that body's chunk, so a
plate on the turret and a plate on the hull one metre below never share a key.

A sub-metre block (V2.4) is an ordinary block whose sub-site is not zero. It shares the layout, the
table, the registry and the decoder with a whole-metre block. An attachment (V2.5) is NOT a cell and is
NOT this record: it is the attachment domain's TLV row, keyed by the block's address plus a face, and
this report contributes only the kind registry and the theme column to it.

Parameters (V2.6) are registry data, never per-cell data. A hull kind's mass and durability come from
its substance row (density and work of fracture), the same way today's hull states its mass as a fact
about what it is. Style is one per-cell parameter: a six-bit variant. The client maps (kind, variant,
the neighbour mask, the cell hash) to small details when connected blocks mesh, and stores none of them.
A theme (V2.7) is a column on a kind row inside the ONE registry mechanism, never a second registry
mechanism.

The registry is dense and append-only. A kind is never renumbered and never reused. Compatibility is a
prefix digest over what a SAVED record MEANS — the kind's triple, its form's geometry, and its form
class. A count that bounds what a player may place NEXT (the variant count, the slot count, the
sub-scale mask) is deliberately OUTSIDE the digest, so shipping a new style never refuses a saved world.
An unknown kind refuses the store; it never decodes to a default. A registry change migrates zero saved
records.

**Example in the game's words.** A player berths a hull at a station. She picks the "riveted" style of
the titanium hull plate and places forty plates. Each plate is one record: body 0, the titanium-plate
kind, the rotation she chose, provenance `Placed`, variant `riveted`, sub-site zero, density "solid"
(a hull is not a planet, so the byte is inert), object parameter zero. Her client draws rivet lines
where two riveted plates meet, from the plates it holds, and never asks the server for a rivet. She
places a quarter-metre titanium wedge in the corner of one cell: one more record, same layout, sub-site
"quarter scale at corner seven". She sticks a shield gauge on a plate's face: one attachment row, no
cell taken, and the plate above it is still placeable. When the station's shard restarts, it reads the
registry digest in its store header, finds this build knows every kind, and opens the store. Her hull
states its mass — the sum of forty plates and one wedge — to the station on change, and the station
applies drag to it.

---

## 1. What exists today — MEASURED

Everything below comes from reading the tree today. It bounds what the record can reuse.

| Fact | Where | What it means for the record |
|---|---|---|
| No block, grid, voxel or chunk module exists. | `ls crates/core/src` shows `built.rs child_index.rs collections.rs controls.rs digest.rs entity_kind.rs fence.rs flight.rs frame.rs geometry.rs home.rs ids.rs incarnation.rs kinematics.rs lib.rs look.rs placement.rs pose.rs realm_coord.rs realm_path.rs rng.rs store_stamp.rs taxonomy.rs tlv.rs units.rs worldgen.rs` (MEASURED, this session). `grep -rn "BlockTypeId\|block_wal\|PaletteEntry\|ChunkKey\|Tier0Key" crates/` returns ONE hit, a doc comment at `crates/io-prod/src/store.rs:6` (MEASURED, re-run this revision). | No world format has ever been written. Every "already spent" door in the investigation base is NOT spent in code. The record starts from zero. |
| The reserved arm names `BlockEdit` (P6) and `Signal` (P9) exist as doc-comment names only, not as variants. | `crates/wire/src/intershard.rs:34`, `crates/wire/src/lib.rs:24` — both `//!` lines (MEASURED) | Giving that name a shape IS adding an arm. It goes to the owner under SL6 (§8.1). |
| The bulk lane declares `BulkKind::{ChunkSnapshot, ChunkDelta, Catalog}` and is declared unroutable. | `crates/wire/src/channels.rs:288-296`; `docs/design/DEFERRED.md:869` | The SL10 diff lane has a name and no route. |
| The TLV envelope pins `CODEC_FLAGS_V1 = 0` and refuses any other value. | `crates/core/src/tlv.rs:46`, `:193-195` | The WIRE envelope's codec tag is planted. The block STORE's codec field does not exist, because no block module exists. §6 item 2 states which is which. |
| The registry idiom is proven: dense explicit discriminants, an `ALL` array, `from_tag` that errors on an unknown tag, and `def()` as an exhaustive match. | `crates/core/src/entity_kind.rs:46-52` (`ALL`, seven kinds), `:58-69` (`from_tag`), `:73-84` (`def()`) (MEASURED, line ranges corrected this revision) | The block registry copies this idiom. |
| The decode-to-Default ban is applied in code, with one deliberate conservative default on a hot path. | `crates/core/src/entity_kind.rs:56-57`; `:92-105` (`continuity_of`/`durability_of` resolve a corrupt id to a safe class, never a panic) | A block kind has no safe class. "Some block" cannot be meshed or collided. The record decoder refuses. |
| A durable file carries a stamp with `layout: u16`, `role`, `universe_seed`, `epoch`, `coordinate_generation`, `world_generation`, and every field is compared for equality. | `crates/core/src/store_stamp.rs:42`, `:104-125`, `:284-311` | The registry prefix check is NOT an equality, so it does not belong in the stamp. It belongs in the block store's own header row (§4.2). |
| The handshake refuses a peer whose `coordinate_generation` differs, by equality, and names both values. | `crates/wire/src/version.rs:350`, `:397-400`, `:418-428` | The registry identity digest joins the handshake the same way (§4.2), and that is a new handshake field (SL6, §8.1). |
| Between shards, the world-law generation rides the TLS ALPN string. | `crates/io-prod/src/trust.rs:41-43` | The registry digest can ride the same string. |
| `ProtoVersion` carries `major`, `minor` and `coordinate_generation` and nothing else. | `crates/wire/src/version.rs:338-351` (MEASURED, re-run this revision) | The CLIENT handshake has no world-generation tag today. |
| `world_generation` DOES appear in `crates/wire` — six hits in `crates/wire/src/admin.rs` (lines 373, 375, 419, 431, 595, 597) — and in `crates/node`, `crates/connection-plane`, `crates/physics` and eight files under `crates/bins`. It does NOT appear in `crates/client` or `crates/client-render`. | `grep -rn world_generation crates --include='*.rs'` (MEASURED, re-run this revision; revision 1's "nothing in `wire/`" was WRONG) | The world-generation tag exists on the ADMIN path and the shard path, never on the client handshake struct. The client half is owed by the generator domain. |
| One redb table exists, a plain `kv` table. | `crates/io-prod/src/store.rs:75` | The block store is a new table set. Nothing constrains its key shapes. |
| `sha2 = "0.10"`, `redb = "2"`, `postcard = "1"`, `noise = "=0.9.0"` are workspace dependencies. | `Cargo.toml:83`, `:70`, `:41`, `:78` | The identity digest uses `sha2`. No new dependency is needed for this domain. |
| A hull states what it IS as whole-number facts. | `crates/core/src/built.rs:37-54` (`BuiltFacts`; `mass_g` at `:40`) (MEASURED, corrected this revision); the rating derives from the facts at `crates/sim/src/stub/drive.rs:637-643` | Per-entity facts already have a home. A block system fills them from placed cells instead of an operator typing them. |
| The doc promise "Today an operator writes these rows; later a shipyard writes them" is at `crates/core/src/built.rs:9-10`. | MEASURED, corrected this revision (revision 1 cited `:10-13`) | The block system changes WHO fills the row, not the row. |
| `BlueprintId(pub u128)` exists, written and never read (its own doc says so at `:27`). | `crates/core/src/built.rs:26-30` | The blueprint-stamp freeze (O6.5) must agree with this width (§6 item 5, D-7). |
| `ShardProfile` already carries `voxel`, `block_edit`, `surfaces`, `seats`, `functional_blocks`. | `crates/sim/src/capability.rs:106-116` | An attachment reuses `surfaces`. No new capability. HR4's two-kind run is possible today (M11). |
| The client already holds a `RealmId` per scene level. | `crates/client/src/net.rs:136` (`origin: Option<RealmId>`), and every scene level carries its realm | The style hash needs NO new data: its salt is folded from the `RealmId` the client already holds (§3.3, and this retires SL6 request 3 of the law verdict). |
| The coordinate ruler's finest rung is a `2⁻¹⁰ m` cell inside a star system. | `crates/core/src/pose.rs:476-478` | A sub-metre block at 1/8 m sits on the ruler exactly. |

---

## 2. The record, bit by bit (answer 1)

### 2.1 Why the record is twelve bytes and not eight

Revision 1 froze one 64-bit word with five reserved bits. Two sibling domains of this same investigation
need fields that do not fit in five bits.

1. **The smooth-terrain domain (V2.1) needs a density byte.** `02_smooth_terrain.md:185` states
   *"terrain density | 8 bits, `i8` | present on EVERY planet cell"*, and `02_smooth_terrain.md:445`
   names it a one-way door that shuts "before the first world is saved". Eight bits do not fit in five.
   The investigation base marks the same door and revision 1 walked past it:
   *"Explicit ids rather than an iso-surface representation — YES in content terms"*
   (`block_system_design.md:5502`, MEASURED by `grep -n` this revision).
2. **The tree requirement (V2.2) needs an object parameter.** The owner's words: a tree is *"one
   object/block on the surface … with different height and structure depends on the seed and height
   params inside this block … so server somehow should know the shape."* A PLANTED tree is not seed
   decided, so its height and structure are per-cell data that the SERVER reads to build the collider.
   The variant is spent on style and is a client-side look; it cannot carry a collider input.

**The consequence if the record stays eight bytes.** A player digs a trench into a hillside on a moon.
The record can say "this cell is dirt" and "this cell is air". It cannot say "this cell is dirt cut two
thirds of the way down". The hillside becomes a staircase, and V2.1 is refused. Or the grid domain adds
a SECOND per-cell record for the surface value after the first world is written, which is the second
machinery HR3 forbids.

**The cost of twelve bytes, ESTIMATED.** 100 million edits cost 1.2 GB of records instead of 800 MB.
Against the pyramid, which §5 now sizes at 3.2 GB to 4.9 GB for the same edits, the extra 400 MB is
about 8 % of the total. The base's own R1 reasoning applies unchanged: *"The extra three bytes cost
≈300 MB across a hundred million edits — noise against a format migration"* (`block_system_design.md:52`,
MEASURED; revision 1 cited `:53`).

**The alternative, named for the owner (D-12).** A variable-width record: eight bytes for a kind whose
registry row declares no attribute word, twelve for one that does. It saves the 400 MB and it costs a
fixed stride, so a chunk's delta stops being a sorted flat array that binary-searches by index. This
report recommends the fixed twelve.

### 2.2 The disagreements inside the investigation base, and how this report resolves them

1. **Orientation is 5 bits in §2 and 6 bits in §4 and R1.** `block_system_design.md:715-724` says
   mirroring is a SHAPE and orientation carries the 24 proper rotations in 5 bits. `:5372-5396` and R1
   say 6 bits with a mirror bit reserved from the first commit. This report keeps 6 bits and
   zero-reserves the sixth (D-1).
   **DISPUTED: the smooth-terrain domain says the sixth bit's reserve is for a future chiral shape's
   ROW, not for a mirror flag (`02_smooth_terrain.md:292-300`)** — and on the evidence the two domains
   agree on the WIDTH (6) and on the RULE (the bit must be zero today), so the disagreement is about
   what a future build may do with the bit, not about the format the owner freezes. Either reading
   leaves the same bits on disk. Recorded so the synthesis can pick one sentence.
2. **The 8-bit "state" field is never specified.** `block_provenance_collapse.md:137-139` names it;
   `:1984-2029` puts every named state bit (grass, leaf, snow, wetness) in the SPARSE side table, not in
   the record. So the record's eight bits had a name and no owner. This report gives the attribute word
   three owners: `variant` (6), `density` (8) and `object_param` (8).
3. **`orient` on a `Terrain` cell.** The smooth-terrain domain requires that `orient` MUST be zero on a
   `Terrain`-form cell and be rejected otherwise (`02_smooth_terrain.md:184`). Revision 1's decode rule
   had no such refusal. It is added to §2.6.

### 2.3 The recommended cell record

```
BlockRecord — 12 bytes: word A (u64) then word B (u32), both MSB first.
PERSISTED. PERMANENT. ONE-WAY DOOR.

WORD A — where it is and what it is
  bits 63..46  cell        (18)  CellIndex inside the body's chunk, 0..=238_327 (the grid domain owns
                                 the chunk edge; 18 bits hold any edge <= 64)
  bits 45..30  kind        (16)  BlockKindId — one row of the generated (substance, form, function) table
  bits 29..24  orient       (6)  bits 24..28 = one of 24 proper rotations; bit 29 reserved, MUST be zero.
                                 MUST be zero entirely on a Terrain-form cell (02_smooth_terrain.md:184)
  bits 23..22  provenance   (2)  0 Terrain, 1 Feature, 2 Placed; 3 REJECTED (R6)
  bits 21..20  sub_scale    (2)  0 = whole cell (1 m); 1 = half; 2 = quarter; 3 = eighth
  bits 19..11  sub_addr     (9)  the sub-block's origin on the 8x8x8 lattice inside the cell,
                                 (z*8 + y)*8 + x, x fastest; MUST be zero when sub_scale = 0;
                                 MUST be a multiple of the sub-block's own size
  bits 10.. 0  reserved    (11)  MUST be zero; decode REJECTS non-zero
  18 + 16 + 6 + 2 + 2 + 9 + 11 = 64.

WORD B — what it looks like and what shape it takes
  bits 31..26  variant      (6)  the style index, 0..=63; a value >= the kind's declared variant count
                                 is REJECTED (V2.6)
  bits 25..18  density      (8)  i8, the smooth surface's signed value at the cell centre, in cell units,
                                 negative inside (02_smooth_terrain.md:185, :445). On a Cartesian realm
                                 (a hull, a station) the canonical value is "fully solid" for an occupied
                                 cell and "fully air" for an Air record; the smooth lane emits nothing
  bits 17..10  object_param (8)  a large landscape part's own shape parameter (V2.2: a tree's height and
                                 structure class). MUST be zero unless the kind's form row carries the
                                 OBJECT flag; REJECTED otherwise
  bits  9.. 0  reserved    (10)  MUST be zero; decode REJECTS non-zero
  6 + 8 + 8 + 10 = 32.
```

**The store key is `(body, chunk)`, and the record addresses inside it.** The attachment-and-bodies
domain states that a joint makes a second rigid BODY inside the SAME realm and that an edit names
`(realm, body, chunk, cell, face)` (`05_attachments_bodies.md:30-40`, `:153`). Revision 1's key
`(chunk, cell, sub_scale, sub_addr)` had no body and would collide.
*Example: a pilot builds a turret on her hull. The turret is a second grid the hull's own shard turns.
She welds a plate onto the turret. Under revision 1's key that plate has the same address as the hull's
own plate one metre below. Under `(body, chunk)` the turret's grid is its own chunk space, body 1, and
the two plates never meet.* The body id costs the record ZERO bits, because it is part of the store key
and of the diff frame's header, not of the cell word. That is a ONE-WAY DOOR and §11 now lists it.

**The dedup-to-final key is `(body, chunk, cell, sub_scale, sub_addr)`.** A whole-metre block has
sub-site `(0, 0)`.

**What each field buys, with the game's example.**

- `kind` — a titanium hull plate, a granite cube, an oak log, a glass pane. One number carries substance,
  shape and function because the registry allocates a row only for a triple the content declares
  placeable (`block_system_design.md:5323-5333`). A wedge of titanium and a cube of titanium are two rows.
- `orient` — the plate's rotation. Integer rotation matrices (`:5388-5392`), so a rotated collider is
  exact.
- `provenance` — R6 (`block_system_design.md:52`; `block_provenance_collapse.md:176-212`). A cave wall
  never falls; a felled tree's logs fall; a placed beam falls.
- `sub_scale`, `sub_addr` — V2.4. A quarter-metre wedge in corner seven of the cell. §2.4 owns the rest.
- `variant` — V2.6. The player picks "riveted" or "smooth" for a hull plate. §3.3 owns the rest.
- `density` — V2.1. *A player digs a trench into a hillside on a moon. Each cut cell writes a density
  that says how deep the cut goes, so the slope the player sees is the slope the character controller
  walks on.*
- `object_param` — V2.2. *A player plants a sapling in a clearing. The record says "an oak kind, Placed,
  structure class 3". The moon's shard reads it and builds a trunk collider of that class, so a walking
  player cannot walk through the trunk.* The GROWTH STAGE is a different thing and lives in R8's sparse
  side table, because it CHANGES with time and the record must never be rewritten by the clock.
- the two `reserved` runs — R1's discipline: slack that cannot be spent by accident.

### 2.4 The sub-metre block (V2.4) shares the layout

**Ruling served.** *"smaller blocks should fit into 1m space of the bigger blocks to simplify collision
and space taking."* The metre cell stays the unit of occupancy. A sub-block is a block whose extent is a
power-of-two fraction of the cell, placed on a lattice inside the cell.

**Why one layout and one table (HR3).** A sub-block is placed, broken, meshed, collided, damaged and
persisted by the same code as a whole block. It uses the same registry, the same orientation numbering,
the same provenance rule, the same variant. A second table with a second record would be a second
machinery for one job. The only difference is the key, which carries the sub-site.

**Why eight per axis.** The catalogue's smallest extent is `Panel` at 1/8 of a cell
(`block_system_design.md:5666`, MEASURED; revision 1 cited `:5670`), snow's step is 1/8 m
(`decision_board.md:1151`), and the coordinate ruler's finest rung is 2⁻¹⁰ m
(`crates/core/src/pose.rs:476-478`), so 1/8 m sits on it exactly.
**The bit budget no longer decides this.** With twelve bytes, a 1/16 m lattice costs 12 address bits and
3 scale bits (five scales need three bits, not two — revision 1's D-2 option (c) was WRONG by one bit),
which still leaves 7 reserved bits in word A. So the choice is now a RUNTIME COST choice, not a bit
choice: 8 per axis is at most 512 sub-blocks in one cell, 16 per axis is at most 4,096. Nobody has
measured either (§9, M12). The recommendation stays 8 on cost grounds (D-2).

**What the base said, and why it is superseded.** `block_system_design.md:5503` lists "one geometry per
cell (no sub-grid)" as a YES one-way door. Its stated reason, read this revision, is *"A chisel layer
must be strictly additive; making the base cell sub-divisible is the SE1→SE2 door"* (MEASURED).
Revision 1 dismissed the door on a DIFFERENT reason (a dense sub-voxel field) that the door does not
give. The correct answer is that this design SATISFIES the door's own test: the sub-site is strictly
additive — an unchiselled cell costs zero extra bytes, sub-site `(0,0)` is exactly today's whole cell,
and no existing record changes meaning. V2.4 (owner, 2026-09-07) then overrides the door's content half.

**Cost, ESTIMATED.** A cell fully tiled by 64 quarter-scale blocks is 64 × 12 B = 768 B, against 12 B for
a whole block. A cell tiled by 512 eighth-scale blocks is 6 KiB. Sub-metre building is the heaviest
per-metre writer the store will have, and it needs a PER-CELL CAP on the placement path, not only a
per-realm byte budget (§5, and D-13).

**Where non-overlap is enforced — the key cannot do it.** A half-scale steel cube at `sub_addr` 0 covers
lattice steps 0..4. A quarter-scale nub at `sub_addr` 0 covers 0..2. Both are lattice-legal, they have
DIFFERENT keys, and they occupy the same space. A per-record decode rule cannot see a SET.
*Consequence in the game's words: a cockpit console cell holds a half-scale cube and a quarter-scale nub
inside it; the hull's shard builds two colliders in one place; a second player's boots pick one and clip
through the other, which is a seam under SL8.* So the invariant is stated in two places:
- **The placement path REFUSES an overlapping placement**, by testing the candidate's occupancy mask
  against the cell's current mask. A cell's mask is 512 bits (an 8×8×8 lattice), built from the cell's
  own records; the test is a bitwise AND. This is the writer's job and it is total.
- **The store open VALIDATES every chiselled cell's set**, once, at open, for the same 512-bit mask. It
  refuses the store on an overlap, naming the body, the chunk and the cell. A store is opened rarely and
  a chiselled cell is rare, so the cost is bounded by the number of chiselled cells, not by the world
  (M13 owes the number).
A whole-cell record and any sub-block in the same cell is the same refusal.

**Collision and occupancy.** The realm's shard treats a cell with any sub-block as occupied for
"space taking" and builds the collider from the sub-blocks' own shapes at their own scale. The collider
domain owns that. What the record guarantees: a sub-block's shape, scale and position are exact
integers, so the server and the client build the same collider and the same mesh (SL10 V1.5).

**Example.** A player builds a cockpit console. She places a whole-metre steel cube, breaks it, and
fills the cell with eight half-scale steel cubes. She then swaps two of them for two half-scale slabs
and a quarter-scale nub. The cell holds nine records, each with its own sub-site, and the placement path
refused the tenth because it would have overlapped. A second player walks up to the console and cannot
walk through it; the shard's collider is the union of the nine sub-blocks.

### 2.5 The attachment (V2.5) — this report WITHDRAWS its packed word

**Revision 1 was wrong to define a packed 8-byte attachment word.** The attachments-and-bodies domain of
this same investigation defines the attachment as a TLV row keyed by the block's address plus a face,
with a kind, a params blob, a `placed_by AccountId`, a `Fence`, and reserved tag ranges for signal
bindings and a joint's live motor state (`05_attachments_bodies.md:161-190`). Two permanent formats
cannot both shut. This report concedes, for three reasons that hold on evidence:

1. **Revision 1's own design already had two artefacts per attachment** — an 8-byte word AND a
   `block_config` TLV blob for a functional one. The TLV row is ONE artefact for the same job, which is
   the HR3-truer shape.
2. **A packed word has no room for `placed_by`.** Who stuck the gauge there is durable provenance the
   packed word cannot carry.
3. **The TLV idiom is already in code** for a built realm's body rows
   (`crates/sim/src/stub/built_store.rs`, cited by `05_attachments_bodies.md:167`), so the attachment row
   reuses a proven shape and grows by append.

**What this domain contributes to that row, and must reconcile with domain 05:**
- The attachment's `kind` comes from a registry built with the SAME MECHANISM as the block registry
  (§4.1): dense, append-only, an `ALL` array, an erroring `from_tag`, an exhaustive `def()`, and a theme
  column so V2.7 can style a gauge as well as a plate.
- **It is a SECOND TABLE, not a second mechanism.** Revision 1 recommended one table with an
  `ATTACHMENT` form class. That is now WITHDRAWN, and domain 05's separate table is adopted, because two
  typed ids (`BlockKindId` and `AttachmentKindId`) make "an attachment kind in a cell record" a
  COMPILE error instead of a runtime refusal, which is strictly stronger. D-4 records the change.
- The identity digest (§4.1) covers BOTH tables, in a fixed order, so one number states the whole
  content identity.
- **One attachment per face is a GAMEPLAY limit, not a technical one.** Domain 05 argues it because a
  HUD would sit inside a joint's gap (`05_attachments_bodies.md:165-166`). That is a good reason and it
  is the owner's call, so §10 carries it as D-14 rather than as a settled fact.

**Deletion rule, and the case revision 1 had no answer for.** An attachment dies with its host cell. Under
SL10 V1.6, base terrain has NO record, so a lamp stuck on the face of a granite cliff has a host with
nothing to delete. The rule is stated in terms of the CELL, never of the record:
- Mining a seed-decided cell WRITES an `Air` record with provenance `Terrain` and canonical zero fields
  (`block_provenance_collapse.md:240-244`). That write is what deletes the lamp, in the same
  transaction.
- An `Air` diff over a seed-decided solid cell NEVER prunes, because the seed's answer at that address
  is granite and the diff is air. If it pruned, the tunnel would close.
- Whatever sat on a removed face goes with the face, whether the host had a record before or not.
*Example: a prospector hangs a lamp on a granite cliff face on a moon, then mines the cliff cell behind
it. The moon's shard writes one `Air` record for the cell and deletes the lamp's row in the same
transaction. The tunnel stays open for ever, because the `Air` record never equals granite.*

### 2.6 The decode rule, stated once

A decoder is a total function with a typed error. It REJECTS, never defaults, on: a non-zero reserved
bit in either word; provenance 3; a kind this build does not know; a variant ≥ the kind's count; a
non-zero `orient` on a `Terrain`-form cell; the reserved orientation bit set; an orientation outside the
shape's legal set; a non-zero `object_param` on a kind whose form row lacks the OBJECT flag; a sub-site
that is not on the lattice or is non-zero with scale 0; an `Air` record with any non-zero field other
than `cell` and sub-site (the canonical form, `block_provenance_collapse.md:240-244`).

**The overlap refusal is NOT here**, because it is a property of a SET and a decoder sees one record.
§2.4 names its two homes.

Three placements, three behaviours (`block_system_design.md:2222-2235`): on the wire, a bad record tears
the connection; in a store, a bad record refuses the store and never skips the record (a skipped record
is a lost build); in generation, a bad record is unconstructable.

### 2.7 The reserved-bit ledger, as a ledger

Revision 1's table mixed a RELEASE into a column headed "Spent by", so the rows did not add up. Stated
correctly, against R1's fifteen reserved bits in one 64-bit word:

| Movement | Bits | Running reserve |
|---|---|---|
| R1's starting reserve (2026-08-03) | — | 15 |
| SPENT by R6: the second provenance bit | −1 | 14 |
| RELEASED: the unspecified 8-bit "state" leaves word A entirely | +8 | 22 |
| SPENT by V2.4: `sub_scale` (2) + `sub_addr` (9) | −11 | **11 in word A** |
| Word B is NEW, 32 bits | +32 | 32 |
| SPENT by V2.6: `variant` (6) | −6 | 26 |
| SPENT by V2.1: `density` (8) | −8 | 18 |
| SPENT by V2.2: `object_param` (8) | −8 | **10 in word B** |

**Total reserve: 11 + 10 = 21 bits.** What they may buy later, each as an epoch-gated additive change:
a `locked` bit for a claim system; a second style axis; a finer sub-lattice (which needs 3 more address
bits and 1 more scale bit — affordable in word A's eleven); a fluid rung if the pyramid's own reserve is
spent first. A fourth provenance class is NOT here: R6's rejected value 3 is that escape.

### 2.8 The one-way doors this record opens

| Door | Shuts | Cost if wrong |
|---|---|---|
| Both words' field orders and widths (§2.3) | before the first `block_wal` frame is written | a read-modify-write of every record in every saved planet, hull and station, an epoch bump, a digest change, a fleet-wide refusal |
| The record's WIDTH — twelve bytes fixed vs kind-selected (D-12) | same | a stride change re-serialises every chunk delta and every compacted chunk |
| The store key carrying a BODY id (§2.3) | same | a turret's plates and a hull's plates share addresses; every saved multi-body construction is ambiguous |
| The numbering of provenance (0/1/2), orientation (24 codes), sub-lattice order (x fastest), `CellIndex` order | same | a renumber reinterprets every saved cell; the provenance renumber also inverts the privilege comparison |
| The sub-lattice maximum (8 per axis) | same | a finer lattice needs 3 address bits and 1 scale bit from word A's eleven — affordable, but the new scale must be an APPENDED value (4), never a renumber |
| The density's meaning (signed, cell units, at the cell centre) | same; and it is domain 02's door | a resolution or sign change reinterprets every trench ever dug |
| Variant width (6 bits = 64 styles per kind) | before the first theme is authored | a kind with a 65th style needs a new kind row |
| The attachment row's key shape (`BlockAddr` + face) | before the first HUD or joint is placed | every saved gauge, joint and rail moves; and the address type must be the block's FULL address, so a sub-block index cannot be forgotten (`05_attachments_bodies.md` §2.2) |

---

## 3. Params vs types (answer 2)

### 3.1 The owner's sentence, split into its parts

*"we might have different block types for different hull types, but it can be one block type with
different params — durability, mass, and style."*

| Param | Is it per cell? | Where it lives | Why |
|---|---|---|---|
| **mass** | NO | derived per KIND at content build: `density_kg_m3 × volume_units / CELL_VOLUME_UNITS`, integer (`block_system_design.md:5625-5633`) | Two titanium plates never weigh differently. A per-cell mass would be a free number a player could set — the magic-number rule refuses it. |
| **durability** | NO | derived per KIND: `integrity_dp` from the substance's work of fracture (`:892-901`) | Same argument. What varies per cell is DAMAGE, which is the sparse side table `damage_dp` (`:1305-1312`), never the record. |
| **style** | YES | the 6-bit `variant` in word B | It is the one thing a player CHOOSES per placement that changes nothing physical. |
| **surface depth** | YES | the 8-bit `density` in word B | V2.1. It is not a player choice and not a style; it is the shape of the ground. |
| **object shape** | YES | the 8-bit `object_param` in word B | V2.2. The SERVER reads it to build a collider, so it cannot be a client-side style. |

So "one block type with different params" resolves to: one FORM (the hull plate) × N SUBSTANCES (an
aluminium alloy, structural steel, a titanium alloy), each a kind row whose mass and durability are
derived from cited physical facts by the tested function `derive_material_row` (`:865-880`). A "heavy
hull" and a "light hull" are two substances, not two numbers on a cell.

### 3.2 The per-entity facts, and where they live in code today

A hull's mass is a per-entity fact. Today the operator writes it: `BuiltFacts.mass_g: u64`
(`crates/core/src/built.rs:40`, MEASURED, corrected this revision), and the hull states it to its parent
on change under the movement contract (`owner_decisions_2026-08-26_movement.md`). The engine rating
derives from the facts (`crates/sim/src/stub/drive.rs:637-643`).

The block system changes WHO fills the row, not the row (the exact promise at `built.rs:9-10`). The
hull's shard keeps a running sum over its placed cells: on every placement or break,
`Δmass = density × volume_units(kind, sub_scale)`, `Δ(mass·r)` for the centre of mass, `Δinertia`. It
rewrites `BuiltFacts` and ships the new facts on change. A 50,000-block ship is never re-summed on a
weld (`block_system_design.md:997-1000`).

Durability per cell is a lookup: `integrity_dp(kind)` scaled by
`volume_units(kind, sub_scale) / CELL_VOLUME_UNITS`, where
`volume_units(kind, s) = volume_units(kind) >> (3 × s)`.

**The exactness of that shift is an ASSERTION, not a fact.** MEASURED this revision by dividing each
catalogue row at `block_system_design.md:5664-5686` by 512: every listed shape divides exactly (Cube
196,608 → 384; Wedge 98,304 → 192; Plate 49,152 → 96; Panel 24,576 → 48). But the base defines the
lattice so that *"every catalogue value and every future authored mesh is exactly representable"*
(`:5621-5626`), and a future authored mesh may have a volume of 1 unit, where `1 >> 9` is 0.
*Consequence in the game's words: a content author adds a thin decorative fin; every eighth-scale fin a
player places weighs nothing and contributes nothing to the pyramid, so a wall of them is invisible from
a hilltop.* So the content bake ASSERTS, per kind, that `volume_units(kind)` is a multiple of 512 for
every sub-scale the kind admits, and the build FAILS otherwise. §12 hands that assertion to the content
build, not a sentence to the pyramid domain.

**Example, CORRECTED this revision.** The base's catalogue gives `Plate` = 49,152 volume units of
196,608, that is one QUARTER of a cell (`block_system_design.md:5665`, MEASURED). Titanium at
4,430 kg/m³ therefore gives 1,107.5 kg per plate, not 4,430 kg — revision 1 used a solid cube for the
plate and the catalogue for the wedge in the same sentence, and was four times too heavy.
A hull of forty titanium plates states mass 40 × 1,107.5 kg = 44,300 kg to the station (ESTIMATED). The
pilot breaks one plate (−1,107.5 kg) and welds a half-scale titanium wedge: a `Wedge` is 98,304/196,608
= 1/2 of a cell, and a half-scale block holds 1/8 of its shape's volume, so
4,430 × (1/2) × (1/8) = 276.875 kg. The hull states 43,469.375 kg — as whole grams, 43,469,375 g — on
the change lane, once. Nothing per tick.

### 3.3 Style: how many bits, and how the client derives the details

**Six bits.** 64 variants per kind. The base's shape catalogue has 23 topologies and the largest orbit
is 24 orientations; a style axis of 64 is wider than any one kind will use, and a theme adds styles by
adding variants, not kinds. A count per kind lives on the kind's registry row (`variants: u8`), and the
decoder refuses a variant ≥ the count. **That count is deliberately NOT in the identity digest** (§4.1),
which is what makes shipping a new style free.

**Registry data per variant (never per cell).** The client's render registry — the base's Tier-B row
behind `RenderMaterialId` (`:620-621`, `:774`, `:9925-9930`) — gains a per-(kind, variant) row: the
texture set, the trim rule, the detail rule, the emissive defaults. Every float lives there; the
server's identity digest never hashes it.

**How the client derives the details when connected blocks mesh, and stores nothing.** The owner's
words: *"when same blocks of that type are connected and construct the mesh, it will render small
additional details, that are not stored on the BE."* The mechanism is the base's derived-decoration
principle (`:10243-10275`) applied to hulls instead of grass, and the base's per-block hash variation
(`:8479-8483`):

```
detail(cell) = f( kind, variant, neighbour_mask_26(cell), splitmix64(salt_of(realm_id), cell) )
```

- `neighbour_mask_26` is which of the 26 neighbours hold the same (kind, variant). It decides trim: a
  panel line where two riveted plates meet, a rounded edge where a plate meets air, a rivet cluster at a
  corner where three plates meet.
- the salt is FOLDED FROM THE `RealmId` THE CLIENT ALREADY HOLDS (`crates/client/src/net.rs:136`,
  MEASURED), so a 200 m hull does not repeat every metre and NO new data crosses any boundary. Revision
  1 wrote `realm_salt` without saying where it came from; the law verdict asked for the local
  formulation first, and this is it.
- `f` is a total, stateless function of state the client legitimately holds. It reads no pose, no tick,
  no live state.

**The rule at a chunk edge and at a realm edge — revision 1 had none.** The mask needs all 26
neighbours, and a client does not always hold them.
- **The mask STOPS at the realm boundary, always.** A hull's plate is never trimmed against a station's
  wall. This keeps the SL6 answer clean and it is also what looks right: two realms are two objects.
- **A cell whose neighbour chunk has NOT ARRIVED draws NO trim on that side, and the missing side is
  marked pending.** When the chunk lands the cell re-meshes with the rest of that chunk's arrival, in
  the same frame, so the trim appears together with the geometry it belongs to and not on its own.
*Consequence if the rule is absent, in the game's words: a pilot berths at a station; the plate at the
far side of her hull, whose neighbour chunk has not arrived, draws as an outside edge and its rivet line
SNAPS into place when the chunk lands. That is the flicker seam SL8 names, and revision 1 claimed it was
impossible by construction.* It is refused by this rule, not by construction.

**Which law covers it — this is an OPEN OWNER DECISION, not a settled one.** Revision 1 claimed SL10
V1.7 settles it. It does not. V1.7 GRANTS nothing; it RESTRICTS, and V1.1 defines the static shape as a
function of `(seed, address)`. The trim function's inputs are a player's placed records, a 26-neighbour
mask that changes whenever a player welds a plate, and a realm salt. None of those is `(seed, address)`.
SL3 pulls the same way: the realm authors HOW IT LOOKS. The case FOR the client deriving it is that the
records are already a lawful one-hop diff (SL10 V1.6) and the derivation is meshing, which is what a
renderer does. The case AGAINST is that the realm should author its own look. The owner decides: D-15.
The record survives either answer — the variant is still six bits — so the cost of a late change is
contained to the meshing path.

**What a detail may never do.** A detail never collides, is never targeted, never holds a byte of
storage, and never changes a face's airtightness or occlusion (`:11383-11392`). If a "detail" must
collide — a real rail, a real step — it is a sub-block or an attachment, not a detail.

**Seamless (SL8).** The detail set fades by the material feature ladder's `tile_max_tier` and
`normal_max_tier` per material (`:9938-9940`): a rivet line dissolves into the plate's albedo at tier 2,
never a box that pops. When a neighbour changes, the detail re-derives at the remesh the edit already
causes — an edit-rate change, never a tick-rate one.

**Example.** Two players stand at a station and look at the same hull. Both clients hold the same forty
plate records and both fold the same salt from the hull realm's id. Both derive the same rivet lines at
the same corners. Neither client asked the station's shard for a rivet. A third player watches the hull
from two kilometres; her client draws the plates at tier 2 with no rivets, and as she flies in the
rivets resolve inside the crossfade band.

---

## 4. The registry (answer 3)

### 4.1 Identity, and what the digest may cover

Four block tables plus the attachment table, all built with ONE mechanism (the proven
`entity_kind.rs` shape, `crates/core/src/entity_kind.rs:46-84`):

| Table | Id | Cap | Numbering |
|---|---|---|---|
| substance | `SubstanceId(u16)` | 65,536 (the base's 512 at `:687` is a cap, not a width) | dense, append-only, never reused |
| form | `ShapeId(u16)` | 65,536 (the base's 256 likewise) | same |
| function | `FunctionId(u16)` | 65,536 | same |
| block kind | `BlockKindId(u16)` — a row `(substance, form, function, flags, variants, sub_scale_mask, theme)` | 65,536; the build FAILS above 65,536 and WARNS at 49,152 (`:5350-5352`) | same; a row's triple never changes once allocated |
| attachment kind | `AttachmentKindId(u16)` — domain 05's table `(max_params_bytes, makes_body, legal_faces, slots, variants, theme)` | 65,536 | same |

Every `def()` is a generated exhaustive `match`. Every table has a drift tripwire test (`:2278-2310`).

**What the identity digest covers — CORRECTED this revision.** Revision 1 put the variant count, the
slot count and the sub-scale admit mask inside the digest. That BREAKS the report's own zero-migration
promise: a titanium hull plate is an OLD row inside every stored prefix, so raising its variant count
from 3 to 4 changes that row's bytes, the stored prefix digest changes, and §4.2's store-open rule
refuses every saved store.
*Consequence in the game's words: a player builds a station on a moon; the team ships a fluted style for
the hull plate; the station's shard restarts and REFUSES to open its store, and so does the moon's, and
so does every planet's. The single most common content change becomes a fleet-wide outage.*

The fix is to split the two jobs by ONE test: **does this column change what a SAVED record MEANS?**

| In the digest (it changes a saved record's meaning) | Out of the digest (it bounds what a player may place NEXT) |
|---|---|
| a block kind's `(substance, form, function)` triple | the kind's `variants` count |
| the referenced form's geometry-defining fields: volume units, face masks, collider, octant mask, mirror twin | the kind's `sub_scale_mask` |
| the form's OBJECT flag and the meaning of `object_param` for it | an attachment kind's `slots` and `legal_faces` |
| an attachment kind's identity (what its row MEANS) | an attachment kind's `max_params_bytes` |
| the `theme` column | any damage, thermal or PCU column |
| the density's resolution and sign convention (domain 02's door) | any float; any render row; a `GameScale` constant |

A saved variant is always BELOW the old count, so RAISING a count can never invalidate a saved plate.
The CURRENT build enforces the count at decode and at placement. LOWERING a count stays a refusal, which
is what §4.2 wants. Retuning anything in the right-hand column is free for ever.

The digest is `identity_prefix_digest(n)` = SHA-256 (`sha2`, `Cargo.toml:83`) over the left-hand columns
of rows `0..n` of both kind tables, in a fixed table order.

### 4.2 Compatibility: two checks in two places

| Where | Check | Refusal |
|---|---|---|
| **The handshake** (client↔gateway; shard↔shard) | EQUALITY on `identity_digest(full)` — the same shape as `coordinate_generation` (`crates/wire/src/version.rs:397-400`); between shards it rides the ALPN string beside `world_generation` (`crates/io-prod/src/trust.rs:41-43`) | a typed refusal naming both digests and both lengths (`:418-428`). Under SL10 this is what the no-drift gate demands: a differing registry is a differing chunk byte. |
| **The store open** | `(registry_len, identity_prefix_digest(registry_len))` in the block store's own header row; open succeeds iff this build's `len ≥ stored len` AND `identity_prefix_digest(stored len)` matches | refuse to open, naming both. Never a genesis, never a best-effort decode (`docs/design/DEFERRED.md:3139`, MEASURED, corrected this revision). |

**The flag-day cost of the handshake equality, stated as a cost and not a footnote.** With the corrected
digest column set, adding a STYLE no longer moves the digest, so the common content change is free on
both checks. But adding a KIND, a SUBSTANCE or a FORM does move it, and equality then refuses every
unpatched client at the gateway.
*Consequence in the game's words: the team ships one new hull SUBSTANCE on a Tuesday; every pilot in the
galaxy is refused at the gateway until she patches.* That may be the right answer under SL10, because a
client that does not know a kind cannot generate a chunk that holds it. It is a release-train
obligation, and the owner must see it: D-5 now carries the cost.

**Why the store check is not in `StoreStamp`.** The stamp compares every field for equality by law
(`crates/core/src/store_stamp.rs:24-28`), and a prefix check is a two-branch `≥`. So the block store
carries its own header row. The stamp's `layout` stays at 1 and the stamp is untouched.

**What passes without a migration:** appending a kind, a substance, a form, a function, an attachment
kind, a variant, a theme; RAISING a variant or slot count; widening a sub-scale mask; retuning any
physical or render column; adding a damage channel. **What refuses, loudly:** changing an allocated
row's triple, renumbering, reusing a retired slot, changing a form's geometry, LOWERING a variant or
slot count, narrowing a sub-scale mask, changing the density convention.

### 4.3 The decode rule is §2.6

Unknown kinds refuse the store, tear the wire, and cannot be generated. A retired kind keeps its slot
with `replaced_by: Option<BlockKindId>` and the `RETIRED` flag; placement of a retired kind is refused,
and its saved records still decode and still draw (the retired row keeps its geometry).

### 4.4 How a registry change migrates zero saved records

- **Append.** The prefix digest is stable under appends. Old stores open. Old clients are refused at the
  handshake until they patch.
- **Raise a bound.** A new style, a wider sub-scale mask, one more slot: NOT in the digest, so old
  stores open AND old clients still connect. This is the case revision 1 got wrong.
- **Retire.** The slot stays. Saved records still decode. New placements are refused.
- **Replace.** A remap is an explicit, tested pass at an epoch bump that writes NEW records through the
  ordinary WAL (`:4617-4621`). It never rewrites bytes behind the log. The pass is idempotent and
  re-shard safe.
- **Reinterpret.** Forbidden. There is no operation that changes what a saved kind number means.

**Example, CORRECTED this revision.** The team ships a "fluted" variant for every hull plate. Every
plate kind's variant count goes from 3 to 4. The count is OUTSIDE the digest, so every planet, hull and
station store on every shard opens as before, AND every client that has not patched still connects — it
simply cannot place a fluted plate, and it draws one it receives with the fallback the render registry
names for an unknown variant. Not one saved record moves. In the SAME release the team also ships a
carbon composite SUBSTANCE: that appends kind rows, the full digest changes, and every unpatched client
is refused at the gateway until it patches. Two changes, two costs, and the owner should ship them in
separate releases if a flag day is expensive.

### 4.5 Themes (V2.7): a theme is a column, inside the one mechanism

- A theme is a `ThemeId(u16)` column on every block-kind row and every attachment-kind row. Every kind
  belongs to exactly one theme; the content build asserts it.
- Kinds of different themes interleave in the append-only numbering over time. A theme is NOT a reserved
  tag range: a range would be a cap the owner has already refused twice (SL9's spirit; R1's "keep
  slack").
- Every host decodes ALL themes (SL5). Which themes a player may PLACE in a realm is realm data (a
  planet's theme allow-list), never a property of a record and never a second registry.
- Clothes, wands and weapons are ITEMS. An item registry is a separate table of a separate domain. A
  wand that places a stylised block references a kind; a kind never references an item.
- Room, ESTIMATED: the space theme is 944 rows at 40 substances × 23 shapes + 24 terrain substances
  (`block_system_design.md:5340`). A theme of the same size costs ~1,000 rows; 65 such themes fit under
  the 65,536 cap.

**Example.** A planet in the outer arm is authored with the "verdant fantasy" theme: mossy stone kinds,
carved-wood kinds, plant kinds with pass-through colliders. A player brings a titanium plate from the
space theme in her inventory. Placing it there is refused by the planet's allow-list (a realm rule), but
her hull — its own realm, its own allow-list — keeps its plates, and both clients draw both correctly,
because both decode the one registry.

---

## 5. The `block_wal` arithmetic under SL10 (answer 4)

**The rule.** Base terrain has NO record. What the seed decides is computed on both hosts (SL10 V1.1,
V1.2); only a diff from the seed is written (V1.6). Storage tracks EDITS, never world size.

**The per-planet numbers, ESTIMATED. Revision 1 kept two totals that its own scatter row disproves; they
are DELETED here and one figure replaces them.**

| Quantity | Arithmetic | Result | Source |
|---|---|---|---|
| A pre-generated starter world (what NOT storing saves) | `6 × 4096² × 272` chunks at 4 bits/cell | **2.9 PiB**; an Earth-sized body 1,551× that | `:4513-4515` |
| 100 M edits as sparse records, at TWELVE bytes | `100 M × 12 B` | **1.2 GB** (was 800 MB at 8 B) | this report, §2.1 |
| The edit pyramid, at the base's PLANNING figure of 1.1 entries/edit | `100 M × 1.1 × 8 B` | 880 MB — **SUPERSEDED, do not use** | `:1636-1640` |
| The edit pyramid, at PLANETARY SCATTER, which is the case that happens | 6.07 entries/edit at 100 M edits on a 328,454 km² starter body | **4.86 GB** | `storage_and_streaming.md:885-895` (MEASURED, read this revision) |
| The same, with the base's own fill-fold mitigation (~4 entries/edit) | `100 M × 4 × 8 B` | **3.2 GB** | `storage_and_streaming.md:905-916` |
| **The per-realm delta budget this domain recommends** | records + pyramid | **4.4 GB mitigated; 6.1 GB unmitigated** | this report |
| Chunks that flipped masked-dense — crossover RE-DERIVED for a 12-byte record | on a hull: `12n ≥ 29,791 + n` ⇒ n ≥ 2,709 cells (1.14 %). On a planet, where the density byte joins the palette index in the dense plane: `12n ≥ 29,791 + 2n` ⇒ n ≥ 2,980 cells (1.25 %) | crossover MOVES EARLIER than the base's 4,256 cells (1.8 %), because the sparse record grew and the dense form did not | this report, ESTIMATED |
| A hollow 20×20×30 building | sparse 2,928 cells × 12 B = 35,136 B; 3-D boxes 6 × 20 B = **120 B** | **293×** smaller (was 244× at 8 B / 16 B) | `storage_and_streaming.md:686-693`, re-derived |
| One felled reference oak | the base's 1,845 B was computed on an 8-byte record | **UNMEASURED at 12 bytes** — the tree domain owes the re-derivation | `block_provenance_collapse.md:802-812` |

**The recovery claim revision 1 made is WRONG, and this matters for the budget.** Revision 1 said "a
regrown forest and a refilled hole leave nothing". The base's own prune rule says otherwise, and
deliberately: *"Break a `Terrain` granite cell, place granite back: … Players would build
never-collapsing bases out of restored crust. A `Placed` cell can never equal a `Terrain` cell, so it
can never prune"* (`block_provenance_collapse.md:253-265`, MEASURED, read this revision).
- A REGROWN FOREST DOES prune: natural recovery drifts toward the generated baseline, and a recovered
  `Feature` cell equals the generator's answer (R7, R8).
- A REFILLED HOLE DOES NOT prune, ever. The put-back cell carries provenance `Placed`; the seed's answer
  is `Terrain`; the full-record comparison says they differ; the record stays for ever.
*Consequence in the game's words: a guild digs a quarry on a moon, changes its mind, and fills it back
in. The moon's store keeps every filled cell for ever, and a griefer can dig and fill all day until the
shard refuses honest edits.* So dig-and-fill needs its OWN abuse gate on the placement path — a rate
limit per account per realm, not a prune rule. D-16 puts the choice to the owner.

**What SL10 adds to the ledger.** New writers this domain introduces:

| Writer | Record | Per unit, ESTIMATED |
|---|---|---|
| a sub-block | 12 B, same table | a cell of 64 quarter-blocks = 768 B; of 512 eighth-blocks = 6 KiB |
| an attachment | one TLV row (domain 05), including `placed_by` and a params blob | a HUD ≈ 200–400 B; the DRAW LIST is transient and never stored (`:11834-11837`) |
| a growth stage (R8) | 1 B sparse side-table column | per planted thing |
| a density byte | 0 B extra (inside word B) | — |
| a variant | 0 B (inside word B) | — |

**The per-realm budget must count records per CELL, not only bytes per realm.** A player can tile 40,000
cells of a hull at eighth scale: 20.5 million sub-block records, 246 MB at 12 B — well inside a
gigabyte-sized realm budget, and a collider set no shard has been shown to build inside a tick (D-13,
M12). The budget refuses new edits with a typed error and an alarm; it never drops a record
(`:4609-4614`).

**Example.** A guild builds a spaceport on a moon: 2 M whole blocks, 40,000 chiselled console cells at
sixteen sub-blocks each, 3,000 HUDs. The record ledger is 24 MB of plates as sparse records (or ~120 KB
as boxes, because a spaceport is walls and floors), 7.7 MB of sub-blocks, ~0.9 MB of attachment rows,
and the pyramid entries above them. The moon's shard refuses the 3,001st HUD only if the guild's realm
budget is exhausted, and says so.

---

## 6. The seven format freezes of O6 (answer 5)

| # | Freeze | Stands? | Recommended answer |
|---|---|---|---|
| 1 | **Greedy 3-D box encoding** as a fourth delta arm: `anchor 18 \| extent_u 6 \| extent_v 6 \| extent_w 6 \| reserved 28` + the placement word verbatim (`storage_and_streaming.md:679-683`) | **Stands, RESIZED.** | Adopt at **20 bytes** per box (the 8-byte box header + the 12-byte record), not 16, because the record grew. A box covers WHOLE cells only — its record has sub-site `(0,0)` — and a chiselled cell is never inside a box. A box also requires a UNIFORM density byte across its cells, so a box never appears on a cut slope. Determinism by a fixed axis priority and ascending anchor order, proven by an encode/decode byte-equality proptest and an edit-order-permutation proptest (`:696-703`). |
| 2 | **An orthogonal `codec` field beside `encoding`**, always `None` today | **Stands as a DESIGN, and is NOT yet planted.** Revision 1 marked it "already planted (MEASURED)" and cited `crates/core/src/tlv.rs:46`. That constant is the WIRE envelope's codec flag (MEASURED). The block STORE's `encoding` and `codec` fields do not exist, because no block module exists (§1, row 1). | Adopt, and copy the wire envelope's proven shape: pin `BLOCK_STORE_CODEC_V1 = 0` and refuse any other value. Codec × encoding is a product, never extra encoding arms. Compression itself is deferred (D-6). |
| 3 | **Prune-on-equality as the ONE universal delta rule**, with the provenance comparison | **Stands, and SL10 makes it the law's own definition.** | Adopt. The equality compares the FULL record: kind, orient, provenance, sub-site, variant, density, object_param. A `Placed` cell never equals a `Terrain` cell, so break-and-replace cannot launder built matter into ground — AND, as §5 states, a refilled hole is therefore permanent. State that consequence in the same sentence as the rule, so nobody claims recovery twice. Apply to the pyramid, `column_sky`, `snow_depth`, `grass_cover` and the chunk record alike. |
| 4 | **Split `state_bits` into its own `(u32 local, u16 state_bits)` table** — 1.84× on the weather writer | **Stands.** | Adopt. Note for V2.6 and V2.1: the persisted STYLE is word B's `variant` and the persisted SURFACE is word B's `density`; NEITHER is in `state_bits`, which stays grass, leaf, snow, wetness and five spare. |
| 5 | **Blueprint stamps: lazy reference vs eager cell write, and the snap lattice** | **Stands, with two facts the board did not have:** R18 builds blueprints at a DOCK (`:88`), and `BlueprintId(pub u128)` already exists in code (`crates/core/src/built.rs:30`). | Recommend LAZY (a 64-byte durable stamp row) and an **8 m** snap lattice; snapping loosens freely and only tightening strands stamps. The stamp id's WIDTH is now an owner decision with a security half — see D-7. The taste half (how building feels at 8 m) is the owner's. |
| 6 | **The pyramid's storage-block granularity as a tuning field** | **Stands**; the edit-pyramid domain (03) owns the value. | One storage block per `(body, tier, chunk)`, in the same redb file as the deltas, so a realm's store moves whole on a re-shard. |
| 7 | **The `CellIndex` linearisation** `(c*62 + b)*62 + a`, `a` fastest | **The board's status is STALE.** It says the door is "ALREADY SPENT … stated only in a doc comment" (`decision_board.md:1152`, `:1169`). MEASURED today: no grid, chunk or block module exists. Nothing is spent. | Confirm it deliberately, in the record spec, as the cell order; and use the SAME order for the sub-lattice, `(z*8 + y)*8 + x`, `x` fastest. The chosen order is the better one (~2× fewer runs than radial-fastest on terrain, `storage_and_streaming.md:1223`). |

**An eighth item the board's row does not list and this domain must add:** the two-word record layout of
§2.3, the twelve-byte width, and the store key's body id. O5 froze a one-word record; V2.1, V2.2, V2.4
and the bodies domain all move it. They shut on the same day as the seven.

---

## 7. Stale claims found by this domain

| Stale claim | Where | Superseded by |
|---|---|---|
| "One geometry per cell (no sub-grid) — YES, a one-way door"; "the base cell is never subdividable" | `block_system_design.md:5503`; `decision_board.md:137-138`; `block_system_design.md:16362` | V2.4 (owner, 2026-09-07). The door's stated reason is additivity, and the sub-site design SATISFIES it: sub-site `(0,0)` is today's whole cell and no existing record changes meaning. |
| "Explicit ids rather than an iso-surface representation — YES in content terms" | `block_system_design.md:5502` | V2.1 (owner, 2026-09-07) plus `02_smooth_terrain.md:185`. The ids STAY — the density byte sits BESIDE the identity, it does not replace it. The door's content half (players build against a vocabulary) is untouched: a hull is still built of explicit shapes. |
| "The durable placement record is five bytes" and the six sites the board lists | `block_system_design.md:2039-2047`, `:2481-2484`; `docs/design/roadmap.json` P6 per `decision_board.md:2214-2216` | R1 (eight bytes) and now this report (twelve). The roadmap line is still wrong and is a one-line edit the owner must approve (D-11). |
| "Orientation is 5 bits; mirroring is a shape" | `block_system_design.md:715-724` | R1 and `:5372-5396` (6 bits). This report keeps 6 and zero-reserves the sixth bit; the reserve's PURPOSE is disputed with domain 02 (§2.2 item 1). |
| "Eight bits of material-interpreted state" in the record | `block_provenance_collapse.md:137-139`; `decision_board.md:293` | Never specified. This report replaces it with word B: `variant`, `density`, `object_param`. |
| "Fifteen reserved bits" | R1 (`block_system_design.md:52`) | Fourteen after R6; 21 across two words after this report's spend (§2.7). |
| "The `CellIndex` linearisation door is ALREADY SPENT" | `decision_board.md:1152`, `:1169` | MEASURED: no code holds it. Not spent. Confirm it deliberately. |
| "The world-generation tag already refuses a client whose generator disagrees" | `owner_decisions_2026-09-07_voxels.md` V1.3 | MEASURED: `world_generation` reaches the shard-to-shard ALPN, the durable stamp, and the ADMIN path in `crates/wire/src/admin.rs`; the CLIENT handshake struct `ProtoVersion` carries only `major`, `minor`, `coordinate_generation` (`crates/wire/src/version.rs:338-351`), and `crates/client` holds zero hits. The client half is owed by the generator domain. |
| "The cover table is HUD-shaped `(local, face, cover_ref)`" | `block_system_design.md:2137`, `:11766-11800` | V2.5 and `05_attachments_bodies.md:161-190`: it becomes a TLV row keyed by the block's full address plus a face, carrying kind, params, `placed_by` and a fence, with the `cover_ref` indirection dropped. |
| "A functional block MAY have its appearance programmed by signals through a small state index" (R5) reads as if the persisted variant and the signal-driven index are one field | `block_system_design.md:55`; `:14103-14170` | They are two: the persisted `variant` is the player's placement choice; R5's index is TRANSIENT, rides the state-light lane, and never touches the record or the pyramid. |
| "Decoration output is client-derived under a byte-identity gate" was a USER DECISION awaiting the owner (6-1) | `block_system_design.md:10243-10275` | STILL OPEN. SL10 settles the STATIC SHAPE only; the trim function's inputs are placed records and a neighbour mask, which are not `(seed, address)`. It is D-15, not a settled point — revision 1 wrongly closed it. |
| The pyramid planning figure of ~1.1 entries per edit, and the "< 1.7 GB per heavily-played planet" total | `storage_and_streaming.md:233-236` | `storage_and_streaming.md:885-895` (planetary scatter, 6.07 entries/edit) — the same document's own later section. §5 now sizes the budget at 4.4–6.1 GB. |
| "A regrown forest and a refilled hole leave nothing" | revision 1 of this report, §5 | `block_provenance_collapse.md:253-265`: a `Placed` cell can NEVER equal a `Terrain` cell, on purpose. The forest prunes; the hole does not. |
| The 16-byte greedy box (O6.1) and the masked-dense crossover at 4,256 cells | `storage_and_streaming.md:679-683`, `:194-198` | Both were derived on an 8-byte record. At 12 bytes the box is 20 B and the crossover moves to ~2,709 cells on a hull and ~2,980 on a planet (§5). |
| `BlueprintId` "needs a durable author identity that survives copying" and the stamp id is a SHA-256 | R18; `storage_and_streaming.md:721-728` | Code now has `BlueprintId(u128)` written and unread (`crates/core/src/built.rs:26-30`). The width must be reconciled, with the adversarial case (D-7). |
| "`InterShardFlow` has exactly 27 arms; 27 + 3 = 30 is the ceiling" | `decision_board.md:65-84` (O15) | Not re-counted here; the wire domain owns it. This domain adds ONE arm — `BlockEdit`, which today is a doc-comment name and not a variant — and it asks for it under SL6 (§8.1). Revision 1's "ZERO arms" was wrong. |

---

## 8. Law check

### 8.1 SL6 — the requests this domain must make, in SL6's own shape

Revision 1 answered "NONE" and was wrong. SL6 says: ask before new data crosses a realm boundary AND
before adding a wire arm; default NO. Two requests stand. A third was withdrawn by finding the local
formulation.

**Request 1 — give the `BlockEdit` arm a shape.**
- *What data:* one block edit — the body, the chunk, the cell record (12 bytes), and a `Fence`.
- *From which realm to which:* from an occupant's client through the gateway to the realm's own shard —
  which is the connection plane and not a realm crossing (SL2's 2026-08-24 clarification). Shard to
  shard ONLY if a ghost region straddles a seam during a crossing; under "the deepest containing region
  is your realm" an occupant edits its OWN realm's cells, so the shard-to-shard leg may not be needed at
  all. The wire domain must say whether it survives containment re-home.
- *Why the receiver cannot compute it:* a player's choice is not a function of the seed.
- *Cost of doing without:* no player can build anything. The requirement is the game.
- *Status:* the name exists as a doc comment only (`crates/wire/src/intershard.rs:34`,
  `crates/wire/src/lib.rs:24`, MEASURED). Giving it a shape is adding an arm.

**Request 2 — put the registry identity digest on the CLIENT handshake.**
- *What data:* 32 bytes and a row count, each way.
- *From where to where:* gateway ↔ client. Between shards it rides the existing ALPN string beside
  `world_generation` and needs nothing new.
- *Why the receiver cannot compute it:* it is the other side's build identity.
- *Cost of doing without:* a client with a different registry meshes a plate as the wrong kind, and the
  SL10 no-drift gate never sees it, because the gate compares builds and not sessions.
- *Note:* `ProtoVersion` carries `major`, `minor`, `coordinate_generation` today
  (`crates/wire/src/version.rs:338-351`, MEASURED). The generator domain owes the client-side
  world-generation tag; this digest should ride WITH it as one field, not as two.

**Request 3 — WITHDRAWN. The style hash needs no new data.** Revision 1 wrote `splitmix64(realm_salt,
cell)` without saying where the salt came from. The client already holds a `RealmId` per scene level
(`crates/client/src/net.rs:136`, MEASURED), so the salt is folded from that id. Nothing crosses.

### 8.2 The rest of the law table

| Law | This domain's position |
|---|---|
| **SL10** | The registry lives in the ONE generator crate, because the generator emits kind ids and the client must decode them identically; its identity digest folds into the no-drift gate and the world-generation tag. The style-detail function lives in the same crate (integer hash, no transcendental, fixed order — V1.4). Every record is a one-hop diff from the owning realm (V1.6). The density byte's meaning is named in the generator's pinned digest (domain 02's door). |
| **SL5 one world** | One registry mechanism, one decoder, one digest. A theme is a column. A realm's allow-list selects what may be PLACED, never what may be DECODED. No per-realm registry variant, no reduced world, no second implementation. |
| **HR3 one tooling** | One record shape and one table for whole and sub-metre blocks; one registry MECHANISM for block kinds and attachment kinds; one decoder; one digest. The two typed id tables are two TABLES, not two mechanisms, and the split makes a category error a compile error. |
| **HR1 / HR2** | ONE new `InterShardFlow` arm is requested (`BlockEdit`, §8.1), not zero. A record crosses shard→client as a TLV tag on the bulk lane inside `BulkKind::ChunkDelta`. |
| **HR4 features once, run anywhere** | The record and the registry are shard-kind agnostic; `ShardProfile` already carries `voxel`, `block_edit`, `surfaces` (`crates/sim/src/capability.rs:106-116`), so a planet-profile shard and a hull-profile shard run the same fixture. M11 owes the G-IDENTICAL run; revision 1 owed nobody. |
| **HR5** | Every decoder is a monomorphic exhaustive match with a covered refusal arm; the tripwire test sweeps the full tag space. M2 owes the branch coverage on BOTH words. |
| **SL2** | No record carries a pose. A hull's records are the hull realm's own; the station never holds them. A body's pose goes realm → gateway, which is lawful. |
| **SL3** | A realm draws itself from its own records. Whether the TRIM is authored by the realm or derived by the client is D-15, and revision 1 was wrong to close it. |
| **SL4** | Nothing on the crossing path names motion. A record is inert data. A hull, a moon and a rock cross by the same code whether or not they hold cells. |
| **SL7 / SL9** | No cost grows with a parent's child count. The mass sum is incremental on an edit. The registry sweep is a constant 65,536 rows. **Caveat, UNMEASURED:** the per-CELL cost of 512 sub-block colliders is unbounded today, which is why D-13 adds a per-cell cap and M12 owes the frame measurement. |
| **SL8 seamless** | Detail fades by the material feature ladder. The chunk-edge trim rule (§3.3) refuses the flicker seam that revision 1 claimed was impossible. M14 owes the observer's record lane, because "lane flood" is one of the eleven seam kinds. |
| **No magic numbers** | Mass and durability are derived per kind from cited facts; the variant count, slot count, sub-scale mask and theme are registry data; the sub-lattice size is one constant in the record spec, asserted by the no-drift gate. |
| **Client only renders** | The client derives trim from records it holds. Whether that is lawful rendering or unlawful authoring is D-15. It derives no pose, no velocity, no entity state. |
| **Seed and secrecy (2026-08-27)** | A record is never a function of (seed, position): it is a diff. Concealed resources stay a drop table, never a record (R11). |
| **Reuse libraries** | `sha2`, `redb`, `postcard` are workspace dependencies (`Cargo.toml:83`, `:70`, `:41`). This domain proposes NO new dependency. |

---

## 9. Measurements owed

None has run.

| # | Measurement | Method | Passes when |
|---|---|---|---|
| M1 | Both record words round-trip byte-for-byte on x86-64 and aarch64, inside the SL10 no-drift gate | encode a fixture of every field at its extremes on both targets; compare the bytes; decode and compare the structs | zero differing bytes |
| M2 | Every refusal in §2.6 is a covered branch | one `expect_err` per refusal, on the cell record and on the attachment row; `just coverage-fast` on the record crate | 100 % region+branch |
| M3 | The registry digest is stable under an append AND under a variant-count raise, and changes under a triple change | append a kind; raise a variant count from 3 to 4; assert the stored-length prefix digest is unchanged in BOTH cases; then change one row's triple and assert the store refuses with both digests in the message | all three assertions hold. **This is the measurement that would have caught revision 1's digest defect.** |
| M4 | The handshake refuses a client whose registry differs, naming both values | a process-tier gate in the S3 pattern (`crates/wire/src/version.rs:505-545`) | the refusal names both digests and lengths |
| M5 | The chiselled-cell BYTE cost | a fixture of 1,000 cells at 64 quarter-blocks each; measure sparse, masked-dense and box bytes; measure the WAL frame and the compaction time | the per-cell bytes match the 768 B estimate within 10 %; the compaction stays inside `BlockStoreTuning`'s budget |
| M6 | Palette width — RE-SCOPED | count distinct `(kind, orient, provenance, variant)` per chunk over WHOLE-CELL records ONLY, on a 50,000-block hull with 4 variants; chiselled cells and the density byte are EXCLUDED from the palette key by construction (a sub-site is a position and a density is a per-cell value; either would blow any palette) | ≤ 256 entries at the 95th percentile (`block_provenance_collapse.md:373`). Revision 1's key included the sub-site and was written to fail. |
| M7 | The style-detail function is byte-identical on both hosts | derive the detail set for the same hull on the server build and the client build on both targets; compare | zero differing bytes |
| M8 | A registry append migrates zero records | write a store under registry N; open it under registry N+1 (an append AND a variant-count raise); assert every record decodes to the same struct and no byte in the file changed | zero changed bytes, zero refusals |
| M9 | The identity digest cost at open and at handshake | time SHA-256 over the identity columns of both kind tables (~23 KiB, `block_system_design.md:851-861`) | a number in microseconds, on the reference machine; never the word "negligible" |
| M10 | The store refuses on any non-zero reserved bit | write a record with each of the 21 reserved bits set; open | refusal on every one |
| **M11** | **HR4 G-IDENTICAL** — the same fixture places, breaks, chisels and attaches on a PLANET-profile shard and on a HULL-profile shard | the forty-plate fixture, run twice through `ShardProfile` (`crates/sim/src/capability.rs:106-116`); compare the two stores byte for byte | zero differing bytes. Without this the feature does not land (HR4). |
| **M12** | **The FRAME cost of a fully chiselled cell** — the door D-2 shuts on it | build one cell at 512 eighth-scale sub-blocks; time the collider build, the greedy remesh and the upload; count vertices; repeat at 64 quarter-scale | a number, on a named thread, against the base's "edit echo → pixels ≤ 33 ms" budget (`block_system_design.md:7644-7653`). **The door must not shut before this runs.** |
| **M13** | **The overlap validation cost at store open** | a store with 40,000 chiselled cells; time the 512-bit mask validation over all of them at open | inside the shard's boot budget; and it REFUSES a store seeded with a deliberate overlap |
| **M14** | **The observer's record lane** — SL8 lane flood | six clients in a station's berth ring with twenty hulls of 50,000 blocks in reach; measure bytes per second on one observer's lane at the tier the reach ruling puts each hull at | a named byte rate, and a named tier at which records STOP and a coarse form takes over. 50,000 blocks is 600 KB of records per hull at 12 B (ESTIMATED). |
| **M15** | **The volume-shift exactness assertion** | in the content bake, assert for every kind and every sub-scale it admits that `volume_units(kind)` is a multiple of `1 << (3 × max_scale)` | the build FAILS on a thin decorative fin whose volume would shift to zero |
| **M16** | **Dig-and-fill does not recover bytes, and the abuse gate holds** | dig 100,000 cells of a moon and fill them back with the same substance; measure the store before and after | the store GROWS (it must, by law), and the rate gate refuses beyond the configured limit with a typed error, never a dropped record |

---

## 10. Open decisions for the owner, with the recommended answer

| # | Question | Options | Recommended | Why |
|---|---|---|---|---|
| D-1 | Orientation: 6 bits with a zero-reserved sixth bit, or 5 bits with mirroring as a shape? | (a) 6 + a reserved bit; (b) 5, catalogue closed under mirroring | **(a)** | One bit now against auditing every shape for a twin after players have built. Domain 02 agrees on the width and differs on what the bit is for; either reading writes the same bytes today. |
| D-2 | The sub-lattice inside a metre cell | (a) 8 per axis (1/8 m; 9 + 2 bits); (b) 4 per axis (1/4 m; 6 + 2 bits); (c) 16 per axis (1/16 m; 12 + **3** bits — revision 1 said 2 and was wrong) | **(a), and the door must NOT shut until M12 runs** | With twelve bytes the bit budget no longer decides: (c) still leaves 7 reserved bits in word A. The binding cost is RUNTIME — 512 colliders and their mesh in one cell, or 4,096. Nobody has measured either. |
| D-3 | Style: a per-cell variant, or one kind row per style? | (a) 6-bit per-cell variant; (b) style folded into the kind triple | **(a)** | (b) multiplies the 65,536-row cap by the style count (16 styles × 944 rows = 23 % of the cap for one theme) and makes a new style a new kind, hence a new digest, hence a flag day for every client. |
| D-4 | Attachments: one kind table or two? | (a) ONE table with an `ATTACHMENT` form class (revision 1); (b) TWO tables, one mechanism, one digest over both (domain 05) | **(b) — revision 1 is WITHDRAWN** | Two typed ids make "an attachment kind in a cell record" a compile error, not a runtime refusal. The columns differ (`max_params_bytes`, `makes_body`, `legal_faces` mean nothing to a plate). One digest still covers both. |
| D-5 | The registry digest at the client handshake: equality, or prefix? | (a) equality; (b) prefix | **(a), with the flag-day cost stated** | A shorter client cannot generate terrain holding new kinds, and under SL10 a differing registry is a differing chunk byte. **The cost the owner must see:** shipping a new KIND or SUBSTANCE refuses every unpatched client at the gateway. With the corrected digest column set (§4.1) a new STYLE does NOT, which removes the most common case. |
| D-6 | The substance and form caps | (a) the full u16 for every table; (b) the base's 512 / 256 | **(a)** | A cap is not a width; the pyramid entry already carries a 16-bit substance; themes need the room. |
| D-7 | `BlueprintId` and the stamp's content hash | (a) 128 bits of SHA-256, keep `u128`; (b) the full 256 bits; (c) 128 bits AND a rule that the id is never a trust boundary | **(c)** | An ACCIDENTAL collision at 128 bits is impossible in practice (ESTIMATED: at 10⁹ blueprints the birthday probability is about 1.5 × 10⁻²¹). A DELIBERATE collision costs 2⁶⁴ work, which is reachable, and blueprints are player-authored. *In the game's words: a player crafts two hull designs with the same stamp, sells the cheap one, and repairs it into the expensive one at a dock.* So keep the `u128` that code already has AND state that a dock verifies the blueprint's full bytes, never its id alone. |
| D-8 | Does a sub-block carry its own damage? | (a) yes, `damage_dp` keyed `(body, cell, sub-site)`; the crack-stage wire entry widens; (b) no, the cell's whole-block damage pool | **(a)** | A quarter-scale plate is a block with a quarter of the integrity; a shared pool would let one nub's damage break a console. The wire entry is unbuilt, so widening is free today. |
| D-9 | Blueprint stamps and the snap lattice (O6.5) | (a) lazy stamps + 8 m snap; (b) eager cell writes; (c) lazy + 4 m | **(a)** | Large savings on hulls and stations; loosening is free later, tightening strands stamps. The 8 m feel is taste and is the owner's. |
| D-10 | A theme's placement rule | (a) a per-realm allow-list of `ThemeId`, realm data; (b) global | **(a)**, built later | The registry decodes all themes today; the allow-list is a placement validation added when the second theme exists. Reserve nothing in the record. |
| D-11 | The roadmap's "5-byte dedup-to-final edits" line | (a) edit to "12-byte"; (b) leave | **(a)** | A committed DoD line that contradicts R1 and this report becomes a dispute at the exact phase it gates (`decision_board.md:2214-2216`). |
| **D-12** | **The record's width** — the biggest change in this revision | (a) TWELVE bytes fixed: word A + word B always; (b) EIGHT or twelve, selected by the kind's registry row; (c) eight bytes and V2.1's smooth terrain is refused | **(a)** | (b) saves ~400 MB per heavily-played planet (about 8 % of the total) and costs a fixed stride, so a chunk delta stops binary-searching by index and every reader branches. (c) refuses an owner requirement. **The owner must see that this door and domain 02's density door are ONE door.** |
| **D-13** | **A per-CELL record cap on the placement path** | (a) a cap, configured in `BlockStoreTuning`, refusing the N+1st sub-block in one cell with a typed error; (b) no cap, only the per-realm byte budget | **(a)** | The byte budget cannot see the frame cost. *In the game's words: a player tiles 40,000 cells of her hull at eighth scale; that is 246 MB, well inside a gigabyte budget, and 20.5 million colliders no shard has been shown to build inside a tick.* The value waits on M12. |
| **D-14** | **One attachment per face, or several slots?** | (a) ONE per face (domain 05); (b) up to eight slots per face (revision 1) | **(a)**, and it is a GAMEPLAY call | Domain 05's reason is good: a HUD would sit inside a joint's gap. But it is a limit on what a player may build, so the owner decides. The escape is cheap either way: the face byte's 250 unused values hold a second slot later, as an append. |
| **D-15** | **Who authors the connected-block trim?** — revision 1 wrongly declared this settled | (a) the CLIENT derives it from the records it holds; (b) the REALM authors it and ships it | **(a)** | SL10 V1.7 does NOT grant it: the trim's inputs are placed records and a neighbour mask, not `(seed, address)`. So this is the base's own `[USER DECISION 6-1]`, still open. The case for (a): the records are already a lawful one-hop diff, the derivation is meshing, and (b) would ship a per-face byte for pure decoration. The case for (b): SL3 says the realm authors how it looks. The record survives either answer. |
| **D-16** | **Dig-and-fill: the abuse gate** | (a) a per-account per-realm edit RATE limit; (b) let the record prune when the substance matches, and accept the laundering exploit; (c) nothing | **(a)** | (b) is refused by `block_provenance_collapse.md:253-265` — players would build never-collapsing bases out of restored crust. (c) lets a griefer exhaust a realm's delta budget. A rate limit costs nothing to honest play. |
| **D-17** | **Where the tree's shape parameter lives** | (a) word B's `object_param`, 8 bits, permanent; the GROWTH STAGE stays R8's sparse side-table byte; (b) both in the side table; (c) both in the record | **(a)** | The server reads the shape to build a collider, so it must be in the record the collider reads. The growth stage CHANGES with time, and the permanent record must never be rewritten by the clock. |
| **D-18** | **The store key carries a body id** | (a) `(body, chunk)` as the key, body 0 = the realm's own grid; (b) one grid per realm and no turrets | **(a)** | Without it, a plate on a turret and a plate on the hull one metre below share a key. It costs the record zero bits. It is a one-way door and it must shut with domain 05's. |

---

## 11. One-way doors, consolidated for this domain

| Door | Deadline | Cost if wrong |
|---|---|---|
| The record's WIDTH — twelve bytes, two words (D-12) | before the first WAL frame is written | a stride change re-serialises every chunk delta, every compacted chunk and every pyramid file |
| Both words' field orders and widths (§2.3) | same | a read-modify-write of every record in every saved world, an epoch bump, a digest change, a fleet-wide refusal |
| The store key carrying a BODY id (D-18) | same | every saved multi-body construction is ambiguous |
| The density byte's resolution, sign and sample point (domain 02's door, carried here) | same | a change reinterprets every trench ever dug |
| The `object_param`'s meaning per OBJECT form | same | every planted tree grows the wrong shape |
| The numberings: provenance 0/1/2, the 24 rotation codes, sub-lattice `x` fastest, `CellIndex` `a` fastest | same | a renumber reinterprets every saved cell; the provenance renumber inverts the privilege comparison |
| The sub-lattice maximum, 8 per axis (D-2) — **must not shut before M12 runs** | same | a finer lattice needs an APPENDED scale value plus 3 more address bits from word A's eleven; done as a renumber it reinterprets every chiselled cell |
| Dense append-only numbering in all five registry tables; the identity digest's COLUMN SET | before the first record is written | a row whose meaning changed reinterprets a player's wall; a column added to the digest later refuses every existing store; a column wrongly INSIDE it (revision 1's variant count) takes every store offline on a style change |
| The prefix rule at store open and the equality rule at the handshake (D-5) | before the first store is stamped | a store opened by a build that does not know its kinds decodes garbage as blocks |
| Style as a per-cell 6-bit variant (D-3) | before the first theme is authored | a 65th style is a new kind |
| The attachment row's key shape — the block's FULL address plus a face (D-4, D-14) | before the first HUD or joint is placed | every saved gauge, joint and rail moves; a sub-block index forgotten here cannot be added later |
| The box record's 20-byte layout (O6.1) | before the first compaction | a migration over every compacted chunk |
| Prune-on-equality over the FULL record, and its consequence that a refilled hole is permanent (O6.3, D-16) | before the compactor is written | an unbounded store, or a laundering exploit |
| `BlueprintId` = 128 bits of SHA-256, and the rule that it is not a trust boundary (D-7) | before the first blueprint is stored | every stored blueprint id is a different number under a different rule, or a dock repairs a cheap hull into an expensive one |

---

## 12. What this report hands to the other domains

- **The grid family (01):** the chunk edge (18 bits hold ≤ 64); the `CellIndex` order; the sub-lattice
  size and, before D-2 shuts, the collider and mesher FRAME cost of one cell at 512 sub-blocks (M12);
  the 512-bit occupancy mask that the placement path and the store open both use for the overlap
  refusal.
- **The smooth terrain (02):** ACCEPTED — the density byte is word B bits 25..18, `i8`, and its
  resolution, sign and sample point are named in the generator's pinned digest. The refusal "orient MUST
  be zero on a `Terrain` cell" is in §2.6. **What 02 must confirm:** that eight bits are enough at the
  cell scale, and that a box (O6.1) may only cover cells of uniform density.
- **The edit pyramid (03):** a sub-block's contribution to `fill_256` and `occ_mask` at tier 0; the
  storage-block granularity keyed `(body, tier, chunk)`; prune-on-equality applied to the pyramid; and
  the corrected scatter figure — size the pyramid at 4 to 6 entries per edit, never 1.1.
- **The attachments and bodies (05):** ADOPTED — the TLV row wins over revision 1's packed word; the
  store key carries the body id; this domain contributes the attachment kind table's mechanism, its
  theme column, and its place in the identity digest. **What 05 must confirm:** D-14 (one attachment per
  face) is the owner's, not the design's.
- **The content bake:** the assertion that `volume_units(kind)` is a multiple of `1 << (3 × max_scale)`
  for every sub-scale a kind admits (M15), so a thin fin never shifts to zero volume.
- **The wire / diff lane:** the TLV tags for cell records and box records inside `BulkKind::ChunkDelta`;
  the `BlockEdit` arm's shape (SL6 request 1); whether a ghost-region edit forward survives containment
  re-home; the registry digest on the client handshake (SL6 request 2); and the per-observer record lane
  budget (M14).
- **The generator (SL10):** the registry crate boundary; the identity digest inside the
  world-generation tag; the style-detail function's determinism rules, IF D-15 answers "the client".
- **Signals / functional blocks (P9):** the attachment row's reserved tag range for signal bindings;
  R5's transient state index on the state-light lane; joints as mechanical groups.
- **Persistence:** the block store's header row `(registry_len, prefix_digest)`; the per-realm budget
  sized at 4.4–6.1 GB, not 1.7 GB; the per-CELL cap (D-13); the dig-and-fill rate gate (D-16).

---

## 13. Revision log

Every finding from `verdicts/record_law.md` (F1–F12) and `verdicts/record_feasibility.md` (R-1–R-15),
and what revision 2 did with it.

| Finding | Verdict | What I did |
|---|---|---|
| **F1** the variant count in the digest breaks zero-migration | WRONG | ACCEPTED. §4.1 now splits the columns by one test — does it change what a SAVED record means? The variant count, the slot count and the sub-scale mask are OUT. §4.2's lists and §4.4's example are rewritten. M3 now measures the variant-count raise, so the defect cannot recur silently. |
| **F2 / R-1** no room for V2.1's surface value; domain 02 needs an 8-bit density | MISSING / MISSING | ACCEPTED, and it is the largest change. The record grows to TWELVE bytes with a 32-bit word B holding `density`. §2.1 states the reason and the cost; D-12 puts the width to the owner; §11 lists the door; §7 records the base's iso-surface door and how it is superseded without losing its content half. |
| **F3** V2.2's tree parameters have nowhere to go | MISSING | ACCEPTED. Word B carries `object_param` (8 bits) for a large object's SHAPE, read by the server for the collider. D-17 states the split from R8's growth stage, which stays a side-table byte because it changes with time. |
| **F4** the attachment escape needs 11 bits and has 10 | WRONG | ACCEPTED and then made moot: §2.5 WITHDRAWS the packed attachment word entirely (see R-3). The ten-bit claim is deleted. |
| **F5** SL6 answered "NONE" while adding an arm, a handshake field and a salt | BREAKS_LAW | ACCEPTED. §8.1 is now three explicit SL6 requests in SL6's own shape. Request 3 (the salt) is WITHDRAWN because the local formulation exists: the client already holds a `RealmId` (`crates/client/src/net.rs:136`, MEASURED). §7 corrects the "ZERO arms" claim. |
| **F6** the client-derived trim is presented as settled | UNMEASURED_AS_FACT | ACCEPTED. §3.3 no longer claims SL10 V1.7 settles it; D-15 puts it to the owner with both cases; §7 re-opens the base's `[USER DECISION 6-1]`. |
| **F7** the ledger keeps totals its own scatter row disproves | WRONG | ACCEPTED. §5 DELETES the 1.5 GB and 1.7 GB totals and states one budget: 4.4 GB mitigated, 6.1 GB unmitigated, ESTIMATED with the arithmetic named. §12 tells the pyramid domain to size against 4–6, never 1.1. |
| **F8 / R-12** the key cannot enforce non-overlap | MISSING / MISSING | ACCEPTED. §2.4 names two homes: the placement path refuses by a 512-bit occupancy mask, and the store open validates every chiselled cell's set. §2.6 says explicitly that a per-record decoder cannot see a set. M13 owes the cost. |
| **F9** no HR4 two-shard-kind measurement | MISSING | ACCEPTED. M11 added: the same fixture on a planet-profile and a hull-profile shard, stores compared byte for byte. §8.2 gains an HR4 row. |
| **F10** the observer's record lane is never measured | MISSING | ACCEPTED. M14 added, with the berth-ring fixture and the requirement to name the tier at which records stop. §8.2's SL8 row points at it. |
| **F11 and the feasibility file's §5** citation defects | WRONG (not load-bearing) | ACCEPTED. Corrected: `built.rs:40` for `mass_g`; `built.rs:9-10` for the shipyard sentence; `built.rs:37-54` for `BuiltFacts`; `block_system_design.md:52` for R1; `entity_kind.rs:46-52 / :58-69 / :73-84`; `block_system_design.md:5666` for `Panel`; `DEFERRED.md:3139`; `block_system_design.md:5502` and `:5503` for the two doors. |
| **F12** the reserved-bit table does not add up | WRONG (not load-bearing) | ACCEPTED. §2.7 is now a running ledger with the release as a positive row. Total reserve 21 bits across two words. |
| **R-2** the key has no body id; a joint makes a second grid | MISSING | ACCEPTED. The store key is `(body, chunk)`; §2.3 states it with the turret example; D-18 and §11 carry the door; §12 reconciles with domain 05. Zero record bits spent. |
| **R-3** two incompatible attachment records exist | WRONG | ACCEPTED, and revision 1 CONCEDES. §2.5 withdraws the packed word for domain 05's TLV row, with three reasons on evidence, including that revision 1 already had two artefacts per attachment. D-4's recommendation flips to two tables under one mechanism. |
| **R-4** prune-on-equality cannot both recover a refilled hole and refuse laundering | WRONG | ACCEPTED. §5 states the rule once, with the base's own words: the forest prunes, the hole never does. The recovery claim is deleted; D-16 adds the dig-and-fill rate gate; M16 measures it. |
| **R-5** the mass example is four times too heavy | WRONG | ACCEPTED. `Plate` is 49,152/196,608 = 1/4 cell (`block_system_design.md:5665`, MEASURED). Forty plates are 44,300 kg, and the worked change lands at 43,469.375 kg. §0 and §3.2 both corrected. |
| **R-6** D-2 option (c) is short one bit | WRONG | ACCEPTED. A fifth scale needs three bits, so 1/16 m costs 12 + 3 = 15. With twelve bytes the bit budget stops deciding, so D-2's reason is rewritten to a RUNTIME cost and gated on M12. |
| **R-7** a MEASURED grep does not reproduce | UNMEASURED_AS_FACT | ACCEPTED. §1 now reports the real result: `world_generation` DOES appear in `crates/wire/src/admin.rs` (six hits) and in several other crates; it does NOT appear in `crates/client`, and `ProtoVersion` carries three fields. The conclusion survives and the evidence is now true. |
| **R-8** the volume-shift exactness is true for today's twelve shapes only | UNMEASURED_AS_FACT | ACCEPTED. §3.2 states the exactness as an ASSERTION in the content bake, with the thin-fin failure named; M15 owes it; §12 hands it to the bake, not to the pyramid as a sentence. |
| **R-9** a door is recommended with its runtime cost unknown; no per-cell cap | UNMEASURED_AS_FACT | ACCEPTED. D-2 now says the door MUST NOT shut before M12 runs. M12 names the thread and the 33 ms budget. D-13 adds a per-cell record cap. §8.2's SL9 row carries the caveat. |
| **R-10** truncated SHA-256 called collision-safe without the adversarial case | UNMEASURED_AS_FACT | ACCEPTED. D-7 gains option (c) and the 2⁶⁴ deliberate-collision case, with the dock example, and the rule that the id is never a trust boundary. |
| **R-11** an attachment on a seed-decided cell has no host record | MISSING | ACCEPTED. §2.5's deletion rule is restated in terms of the CELL, with the three answers: mining a seed cell WRITES an `Air` record; that write deletes the attachment; and an `Air` diff over a solid seed cell NEVER prunes, or the tunnel closes. |
| **R-13** the derived detail has no rule at a chunk or realm edge | MISSING | ACCEPTED. §3.3 states two rules: the mask STOPS at the realm boundary always, and a cell with a missing neighbour chunk draws NO trim until the chunk lands and they re-mesh together. The "refused by construction" claim is replaced with "refused by this rule". |
| **R-14** the crossover and the palette gate assume one record per cell | WRONG | ACCEPTED. §5 re-derives the crossover for a 12-byte record (2,709 cells on a hull, 2,980 on a planet, ESTIMATED). M6 is re-scoped: the palette key is whole-cell records only and excludes the sub-site and the density. §7 records both superseded numbers. |
| **R-15** equality at the handshake is a fleet-wide flag day, uncosted | MISSING | ACCEPTED. §4.2 states the cost with the Tuesday example. D-5 carries it. The corrected digest column set (F1) removes the most common case — a new style no longer moves the digest. |
| The feasibility file's note that domain 02 reads the sixth orientation bit differently | — | KEPT WITH A DISPUTE. §2.2 item 1 records that this domain and domain 02 agree on the width and the zero rule and differ only on what a future build may do with the bit, so the frozen bytes are identical either way. |
