# Feasibility refutation — 04 the saved block record, the registry, params, styles, themes

**Date:** 2026-09-07. **Target:** `docs/investigation/2026-09-07/04_block_record_registry.md`.
**Lens:** feasibility. I try to break the report on technical grounds only.
**Verdict: REFUTED.** Five load-bearing claims are wrong. Five more are unmeasured but carry a decision.
Five cases the domain needs are absent.

**How to read this file.** MEASURED means I ran a command in this worktree today and the method is
named. ESTIMATED means I checked arithmetic against a cited source and ran no code. A number with no
mark is a defect in this file.

---

## 1. The refutations that break the report

### R-1. The record has no room for the terrain surface, and the sibling domain says it must — MISSING

The report freezes one 64-bit word with **five** reserved bits (`04:131`, `04:254`). The smooth-terrain
domain of the same investigation states a hard requirement on the same record: **"terrain density, 8
bits, `i8`, present on EVERY planet cell"** (`02_smooth_terrain.md:185`), and names it a one-way door
that shuts "before the first world is saved" (`02_smooth_terrain.md:445`).

Eight bits do not fit in five. The report never says the words `V2.1`, `smooth`, `iso-surface` or
`terrain density` (MEASURED: `grep -ci` over the report gives 0, 1, 0 and 0). It freezes the format that
the owner's V2.1 requirement depends on, and it does not know the requirement exists.

The investigation base marks the same door and the report skips it: *"Explicit ids rather than an
iso-surface representation — YES [one-way]"* (`block_system_design.md:5501`).

**In the game's words.** A player digs a trench into a hillside on a moon. The moon's shard must write,
for every cut cell, how deep the cut goes, so the surface that the player sees is the surface the
character controller stands on. The report's word can say "this cell is dirt" and "this cell is air". It
cannot say "this cell is dirt cut two-thirds of the way down". The hillside becomes a staircase.

**Fix.** Do not freeze the cell word until domain 02's density byte is either inside it or refused by the
owner. If it is inside, the word is nine bytes, or the sub-site loses bits, or terrain gets its own
record shape. State which, and re-run the §5 byte arithmetic on the new width.

### R-2. The key has no body id, and a joint makes a second grid in the same realm — MISSING

The report's key is `(chunk, cell, sub_scale, sub_addr)` (`04:136`) and, for an attachment,
`(chunk, cell, face, slot)` (`04:221`). The attachment domain states that a joint makes a **second rigid
body inside the SAME realm** and that an edit names **`(realm, body, chunk, cell, face)`**
(`05_attachments_bodies.md:30-40`), with the block address carrying a body id
(`05_attachments_bodies.md:153`).

**In the game's words.** A pilot builds a turret on her hull. The turret is a second grid that the hull's
own shard turns. She welds a plate onto the turret. Under the report's key, that plate's address is a
cell in the hull's chunk — the same address the hull's own plate one metre below already holds. Two
different plates, one key.

**Fix.** Add the body id to the address, or state that the body's grid is its own chunk space and say
where the body id lives. Either way the key is a one-way door and the report's §11 door list omits it.

### R-3. Two incompatible attachment records exist in one investigation — WRONG

The report defines an attachment as a packed 8-byte word with a **3-bit slot, eight per face**, a 6-bit
variant, a 2-bit provenance and ten reserved bits (`04:209-219`). The attachment domain defines it as a
**TLV row, ONE per face**, with a kind, a `placed_by AccountId`, and reserved tags for a joint's live
motor state (`05_attachments_bodies.md:161-190`), and argues the single slot deliberately: *"One
attachment per face keeps the key total and the refusal simple. A face that carries a joint cannot also
carry a HUD"* (`05_attachments_bodies.md:165-166`).

Both are declared permanent and one-way. They cannot both shut. The report presents its word as the
answer (D-4, `04:609`) without knowing the other exists.

**In the game's words.** A pilot sticks a shield gauge and a rotation joint on the same plate face. Under
the report's word both fit, in slots 0 and 1, and the joint's gap swallows the gauge. Under domain 05 the
second placement is refused. The player gets two different games from two frozen formats.

**Fix.** One of the two dies before the first gauge is placed. Reconcile in one sitting with domain 05,
and carry the `placed_by` field the report's word has no room for.

### R-4. Prune-on-equality cannot both return the bytes of a refilled hole and refuse laundering — WRONG

The report states both:

- §5 (`04:510`): *"a regrown forest and a refilled hole leave nothing."*
- §6 item 3 (`04:530`): *"The equality compares the FULL word: kind, orient, provenance, variant,
  sub-site. A `Placed` cell never equals a `Terrain` cell, so break-and-replace cannot launder built
  matter into ground."*

A refilled hole IS break-and-replace. The seed decided granite. The player mined it, then put granite
back. The put-back cell carries provenance `Placed`; the seed's answer is `Terrain`; the full-word
comparison says they differ; the record stays for ever.

**In the game's words.** A guild digs a quarry on a moon, changes its mind, and fills it back in. The
moon's store keeps every filled cell for ever. The §5 budget's only recovery mechanism does not recover
these bytes, and a griefer can dig and fill all day to exhaust a realm's delta budget until the shard
refuses honest edits.

**Fix.** State the rule once. Either provenance stays outside the prune comparison and laundering gets
its own refusal on the placement path, or the refilled hole is permanent, the §5 budget loses its
recovery claim, and dig-and-fill gains an abuse gate.

### R-5. The worked mass example is four times too heavy, against the report's own cited catalogue — WRONG

§3.2 (`04:327`) states *"A hull of forty titanium plates states mass 40 × 4,430 kg = 177,200 kg"*, and
the §0 headline example repeats it (`04:55`).

The report's own formula is `density_kg_m3 × volume_units / CELL_VOLUME_UNITS` (`04:299`). The base's
catalogue gives `Plate` = 49,152 volume units of 196,608, that is one quarter of a cell
(`block_system_design.md:5665`, MEASURED by `sed -n '5665,5675p'`). Titanium at 4,430 kg/m³ therefore
gives **1,107.5 kg per plate**, not 4,430 kg. Forty plates are 44,300 kg, not 177,200 kg.

The same sentence gets the wedge RIGHT: it uses `98,304 / 196,608` (`block_system_design.md:5668`,
MEASURED) for a half-cell wedge. So the report used the catalogue for one shape in the sentence and a
solid cube for the other.

**In the game's words.** The hull states its mass to the station on change, and the station applies drag
to it. A hull stated at four times its real mass flies wrong at every speed. The number is also the one
the report gives the owner to judge the design by.

**Fix.** Recompute the example. Say plainly whether "hull plate" means the `Plate` shape or a `Cube` of
hull substance; they differ by four.

---

## 2. Numbers that carry a decision and are not measured

### R-6. The D-2 finer-lattice option is short one bit — WRONG

D-2 option (c) reads *"16 per axis (1/16 m; 12 + 2 bits, 2 reserved)"* (`04:607`). The scale field
encodes whole, half, quarter and eighth — four values, two bits (`04:126`). A 1/16 m lattice is a **fifth**
scale and needs three bits. So (c) costs 12 + 3 = 15 bits, not 14, and leaves **one** reserved bit, not
two. The volume rule `volume_units(kind, s) = volume_units(kind) >> (3 × s)` (`04:323`) then shifts by up
to twelve.

This is the arithmetic on which the owner is asked to pick a permanent lattice.

### R-7. A MEASURED grep does not reproduce — UNMEASURED_AS_FACT

§1 (`04:75`) states: *"`grep -rn world_generation crates --include='*.rs'` hits `store_stamp.rs`,
`io-prod/src/{lib,trust,mesh}.rs` and nothing in `wire/` or `client/` (MEASURED)"*.

MEASURED today, same command: `crates/wire/src/admin.rs` holds **six** hits (lines 373, 375, 419, 431,
595, 597), and `crates/physics/src/worldgen/scale.rs`, `crates/node/src/orchestrator.rs`,
`crates/connection-plane/src/admin.rs` and eight files under `crates/bins` hold more. The "nothing in
`wire/`" half is false.

The report's CONCLUSION survives — `ProtoVersion` carries `major`, `minor` and `coordinate_generation`
only (`crates/wire/src/version.rs:338-351`, MEASURED), and `crates/client` holds zero hits (MEASURED). But
a MEASURED mark that does not reproduce is a defect by the report's own rule (`04:8-10`).

### R-8. The exactness handed to the edit-pyramid domain is true for today's twelve shapes only — UNMEASURED_AS_FACT

§12 (`04:641`) hands domain 03 *"its volume is `volume_units(kind) >> (3 × sub_scale)`, exact on the
1/196,608 lattice down to eighth scale"*.

MEASURED, by dividing each catalogue row at `block_system_design.md:5664-5676` by 512: every one of the
twelve listed shapes divides exactly (Plate 96, Panel 48, Wedge 192, Cube 384, and so on). But the base
defines the lattice as *"the exact volume lattice of any polyhedron whose vertices lie on the 1/32 cell
grid"* so that *"every catalogue value and every future authored mesh is exactly representable"*
(`block_system_design.md:5621-5626`). A future authored mesh may have a volume of 1 unit. `1 >> 9` is 0.

**In the game's words.** A content author adds a thin decorative fin. Every eighth-scale fin a player
places contributes zero volume to the pyramid, so a wall of them is invisible from a hilltop and weighs
nothing. The exactness must be an assertion in the bake, not a sentence in a handoff.

### R-9. A one-way door is recommended with its runtime cost explicitly unknown — UNMEASURED_AS_FACT

D-2 recommends 8 per axis, that is up to **512 sub-blocks in one metre cell** (`04:607`), and in the same
row says *"The grid domain must confirm the collider and mesher can pay for 512 sub-blocks per cell."*
§11 then lists the lattice maximum as a door that shuts before the first WAL frame (`04:627`).

The report carries **no frame budget**. It never says on which thread a chunk is meshed or a collider is
built (MEASURED: `grep -ci thread` over the report gives 0). Its measurement list (§9) sizes bytes and
compaction time (M5, `04:593`) and never a frame. So the recommendation to shut a permanent door rests on
a cost the report says somebody else must find.

There is also **no per-cell cap**. The §5 budget counts bytes per realm (`04:506-511`), not records per
cell. A player can tile 40,000 cells of a hull at eighth scale: 20.5 million sub-block colliders in one
realm, ESTIMATED at 8 bytes each = 164 MB of records, well inside a realm budget sized in gigabytes, and
a collider set no shard has been shown to build inside a tick.

**Fix.** Measure one cell at 512 sub-blocks — collider build time, greedy-mesh time, vertex count —
before the door shuts, and add a per-cell record cap to the placement refusal.

### R-10. Truncated SHA-256 is called collision-safe without the adversarial case — UNMEASURED_AS_FACT

D-7 (`04:612`) recommends `BlueprintId` = the first 128 bits of SHA-256 over the blueprint's canonical
TLV bytes, and calls 128 bits *"collision-safe for a blueprint library"*. For an ACCIDENTAL collision
that is true (ESTIMATED: at 10⁹ blueprints the birthday probability is about 1.5 × 10⁻²¹). For a
DELIBERATE collision the work is 2⁶⁴, which is reachable, and blueprints are player-authored content.

**In the game's words.** A player crafts two hull designs that carry the same stamp, sells the cheap one,
and repairs it into the expensive one at a dock. The report must either say the id is not a trust
boundary, or keep the full 256 bits.

---

## 3. Cases the domain needs and the report does not have

### R-11. An attachment on a seed-decided cell has no host record — MISSING

The deletion rule is *"An attachment dies with its host cell: breaking the plate deletes its gauges in
the same WAL transaction"* (`04:236`). Under SL10 V1.6 base terrain has **no record at all** (`04:478`).
So a lamp stuck on the face of a granite cliff has a host with nothing to delete, and mining the cliff
writes a NEW `Air` diff rather than breaking an existing row. The report never states which transaction
takes the lamp away.

This is the "mined cell under a placed block" family, and the report answers none of it: what provenance
a mined seed cell writes, whether an `Air` diff over a seed cell prunes to nothing (it must not, or the
tunnel closes), and what happens to what sat on the removed face.

### R-12. Two sub-blocks may overlap, and nothing refuses it at decode — MISSING

§2.3 promises *"a set of sub-blocks that tile without overlap"* (`04:137`), and §2.2's only lattice rule
is that the origin *"MUST be a multiple of the sub-block's own size"* (`04:130`).

That rule is not enough. A half-scale block at origin 0 covers lattice steps 0..4. A quarter-scale block
at origin 0 covers 0..2. Both are lattice-legal and they occupy the same space. §2.6's decode rules
(`04:266-272`) are all per-record; none can see a SET. So a store holding an overlapping pair opens, and
the report's promise that *"a store refuses a bad record"* does not reach it.

**In the game's words.** A cockpit console cell holds a half-scale steel cube and a quarter-scale nub
inside it. The shard builds two colliders in one space; the player's boots pick one and clip through the
other. Say where the set invariant is enforced — at placement only, or at store open too.

### R-13. The derived detail has no rule at a chunk edge or a realm edge — MISSING

§3.3 (`04:354`) derives detail from `neighbour_mask_26(cell)`, claims it is *"identical on every client"*,
and says the detail-by-box seam is *"refused by construction"* (`04:377`).

The mask needs all 26 neighbours. The report never says what happens when a neighbour lives in a chunk
the client has not yet received, or in ANOTHER REALM (MEASURED: the report contains zero occurrences of
"chunk boundary" or "neighbour chunk").

**In the game's words.** A pilot berths a hull at a station. Her client holds the hull's plates. The plate
at the hull's edge has no neighbour inside the hull realm, so its trim draws as an outside edge, which is
right. But the plate at the far side of the hull, whose neighbour chunk has not arrived, ALSO draws as an
outside edge, and its rivet line snaps into place when the chunk lands. That is the flicker seam SL8
names, and the report claims it is impossible.

If instead the mask reaches into the station's records to trim the hull's plate, that is realm data
crossing a boundary, and the report's SL6 row says *"NONE"* (`04:573`).

**Fix.** State the rule: the mask stops at the realm boundary always, and a cell whose neighbour chunk is
absent draws no trim until the chunk lands.

### R-14. The masked-dense crossover and the palette gate assume one record per cell — WRONG

§5's crossover is `8n ≥ 29,791 + n ⇒ n ≥ 4,256 cells (1.8 %)` (`04:488`). The 29,791 bytes is one BIT per
cell over a 62³ chunk (MEASURED: 62³ = 238,328; 238,328 / 8 = 29,791). One bit per cell can say "this
cell has a record". It cannot address a cell that holds 512 of them.

M6 then proposes to count *"distinct `(kind, orient, provenance, variant, sub-site)` entries per chunk"*
and pass at *"≤ 256 entries"* (`04:594`). The sub-site is a POSITION — 512 values times 4 scales — so two
hundred chiselled cells alone can produce far more than 256 distinct entries by construction. The gate is
written to fail.

The whole §5 ledger is inherited from a pre-V2.4 model where a cell held one record. The report adds
sub-blocks and does not re-derive the crossover, the mask or the palette key.

### R-15. Equality at the handshake makes every content append a fleet-wide flag day, and D-5 does not cost it — MISSING

D-5 recommends EQUALITY on the full registry digest at the client handshake (`04:610`), and the digest
covers each kind's `variants` count (`04:407`). So adding one "fluted" style to a hull plate changes the
digest and refuses **every** unpatched client. §4.4's own example says so (`04:449`) and calls it *"a
release-train question"* (`04:438`) without a cost.

**In the game's words.** The team ships one new hull style on a Tuesday. Every pilot in the galaxy is
refused at the gateway until she patches. That may be the right answer, and the owner must be shown it as
a cost, not as a footnote.

---

## 4. What I attacked and could not break — STANDS

| Claim | How I checked | Result |
|---|---|---|
| No block, grid, voxel or chunk module exists; the doors are unspent | MEASURED: `ls crates/core/src`; `grep -rn "BlockTypeId\|block_wal\|PaletteEntry\|ChunkKey\|Tier0Key" crates/` returns ONE hit, a doc comment at `crates/io-prod/src/store.rs:6` | STANDS |
| The 64-bit field layout adds up, and the ranges have no gap and no overlap | MEASURED by arithmetic: 18+16+6+2+6+2+9+5 = 64; bits 63..46, 45..30, 29..24, 23..22, 21..16, 15..14, 13..5, 4..0 tile 63..0 exactly | STANDS |
| 18 bits hold a chunk edge up to 64 | MEASURED: 64³ = 262,144 = 2¹⁸; 62³ = 238,328 fits | STANDS |
| Nine address bits reach an 8×8×8 lattice | MEASURED: 8³ = 512 = 2⁹ | STANDS |
| The TLV envelope pins a codec value and refuses any other | MEASURED: `crates/core/src/tlv.rs:46` (`CODEC_FLAGS_V1: u8 = 0`), `:193-195` (`TlvError::UnsupportedCodec`) | STANDS |
| The registry idiom is proven in code | MEASURED: `crates/core/src/entity_kind.rs:46-69` (`ALL`, `from_tag` with an error arm, `def()` exhaustive); `crates/core/src/taxonomy.rs:8-11` says it copies that idiom | STANDS |
| The handshake refuses a generation mismatch by equality and names both values | MEASURED: `crates/wire/src/version.rs:397-400`, `:418-428` | STANDS |
| The world-law generation rides the inter-shard ALPN string | MEASURED: `crates/io-prod/src/trust.rs:41-46` | STANDS |
| A hull states whole-number facts and a rating derives from them | MEASURED: `crates/core/src/built.rs:37-54`; `crates/sim/src/stub/drive.rs:637-643` (`rating_of`) | STANDS |
| `BlueprintId(pub u128)` exists, written and unread | MEASURED: `crates/core/src/built.rs:30`, and its own doc says "Nothing reads it yet" (`:27`) | STANDS |
| `ShardProfile` already carries `voxel`, `block_edit`, `surfaces`, `seats`, `functional_blocks` | MEASURED: `crates/sim/src/capability.rs:106-116` | STANDS |
| The finest coordinate rung is 2⁻¹⁰ m, so 1/8 m sits on it | MEASURED: `crates/core/src/pose.rs:476-478` | STANDS |
| No new dependency is proposed; `sha2`, `redb`, `postcard`, `noise` are workspace deps and the APIs used exist | MEASURED: `Cargo.toml:41, 70, 78, 83`; `redb::TableDefinition` in use at `crates/io-prod/src/store.rs:75` | STANDS — the report is clean on the library rule |
| The §5 storage arithmetic | ESTIMATED, re-derived: `6 × 4096² × 272` chunks × 119,164 B = 2.90 PiB; 29,791/7 = 4,255.9 so n ≥ 4,256, which is 1.79 % of 238,328; 2,928 shell cells × 8 B = 23,424 B and 23,424/96 = 244; 19,406 × 1,845 = 35.8 MB; 1.5 GB / 35.8 MB = 41.9; 16 × 944 = 15,104 = 23.0 % of 65,536; 40 × 23 + 24 = 944; 65,536 / 1,000 = 65 themes | STANDS |
| The wedge half of the mass example | ESTIMATED: 4,430 × 98,304/196,608 = 2,215; 2,215/8 = 276.875 kg | STANDS (the plate half does not — R-5) |
| Determinism of the record itself | The two words are integer only. No `libm` call, no fused multiply-add and no fast-math question arises in this domain, and the report claims no IEEE identity it does not have | STANDS |
| SL9 | No cost in this domain grows with a planet's size or a parent's child count: the registry sweep is a constant 65,536 rows, the digest is over the registry, and the store is over edits | STANDS — with R-9's caveat that the per-cell FRAME cost is unbounded and unmeasured |

---

## 5. Small citation drift, listed so a later reader is not misled

- `04:78` cites `built.rs:37-59` for `BuiltFacts`; the struct is `:37-54` (MEASURED).
- `04:313` cites `built.rs:44` for `mass_g`; it is `:40` (MEASURED).
- `04:169` cites `block_system_design.md:5670` for `Panel` at 1/8 of a cell; the row is at `:5666`
  (MEASURED).
- `04:419` cites `DEFERRED.md:3130-3134` for "never a genesis, never a best-effort decode"; the sentence
  is at `:3139` (MEASURED).
- §2.5's spend table (`04:249-254`) mixes a RELEASE of two bits into a column headed "Spent by", so the
  rows read 1 + 2 + 11 + 5 = 19 against fifteen. The final answer of five reserved bits is right
  (15 − 1 + 2 − 11 = 5); the presentation is not.
- §7 (`04:549`) dismisses the base's "one geometry per cell (no sub-grid)" door as an argument about a
  DENSE sub-voxel field. The base's stated reason at that door is different: *"A chisel layer must be
  strictly additive; making the base cell sub-divisible is the SE1→SE2 door"*
  (`block_system_design.md:5501`, MEASURED). The report happens to satisfy additivity, but it dismissed
  the door on a reason the door does not give.
- §6 item 2 (`04:529`) marks the storage codec freeze *"already planted (MEASURED)"* and cites
  `crates/core/src/tlv.rs:46`. That constant is the WIRE envelope's codec flag. The block store's
  `encoding` and `codec` fields do not exist, because no block module exists — the report's own §1 row 1.
  A MEASURED mark on the wrong artefact carries the word "Adopt".
- D-1 (`04:606`) recommends 6 orientation bits with the mirror bit as the future escape. The
  smooth-terrain domain re-validates the same field the other way: *"a future chiral shape gets its twin
  as a ROW … The sixth bit is reserved and must be zero"* (`02_smooth_terrain.md:292-300`). The two
  domains agree on the width and disagree on what the reserved bit is for. The report presents its answer
  as settled.
- `02_smooth_terrain.md:184` also requires that `orient` MUST be zero on a `Terrain` cell and be rejected
  otherwise. The report's decode rule (`04:266-272`) has no such refusal.

---

## 6. What the owner should not freeze yet

1. The cell word's width and field order, until domain 02's density byte is in it or refused (R-1).
2. The key, until the body id question with domain 05 is settled (R-2).
3. The attachment record, until it is one design and not two (R-3).
4. The sub-lattice maximum, until one cell at 512 sub-blocks has been meshed, collided and timed (R-9).
5. The prune rule, until §5 and §6.3 say the same thing (R-4).

Everything else in the report — the registry idiom, the append-only numbering, the prefix digest at store
open, the decode-refusal discipline, style as a per-cell variant, a theme as a set inside the one
registry, and the whole §5 byte ledger for whole-metre blocks — survived every check I could make.
