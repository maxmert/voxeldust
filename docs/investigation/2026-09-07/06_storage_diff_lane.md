# 06 — Persistence and streaming of everything the seed does not decide

**Domain:** the edit pyramid (the third frozen format), the durable store per realm, and the one-hop
diff lane from the owning realm to the window.
**Date:** 2026-09-07. **Status:** an investigation result for the owner. Not binding until promoted.
**Revision:** 2. Two refuters tested revision 1. This revision corrects it. See §12.
**Law read first:** CLAUDE.md (HR1–HR6, SL1–SL10), `owner_decisions_2026-09-07_voxels.md` (SL10 and
V2.1–V2.9), the rulings of 2026-09-05, 09-02, 09-01, 08-27 (three), 08-26, 08-24, `DEFERRED.md`, and the
seamless law SL8.
**Method:** every claim about what exists today cites the code. Every number is marked MEASURED (with
how) or ESTIMATED. The investigation base of 2026-08-03/04 is an input; where it is superseded, the
report says so in §9.

---

## 0. The answer in one page

1. **One edit is a tier-0 cell that changed.** The store holds only what the seed does not decide: the
   list of authored cells per chunk. The client computes the shape from `(seed, address)` (SL10 clause 1)
   and lays the diff over it (clause 6). A chunk with no edits costs zero bytes on disk and zero bytes on
   the wire, at every rung. *Example: a player digs a tunnel into a moon. The moon's shard stores the
   cells the player removed and nothing else. A second player, a week later, receives those cells as a
   diff and cuts the tunnel into the hill the client already derived.*
2. **The edit pyramid is a derived index, not a second truth.** Only tier 0 is written. Every coarser
   rung is folded from the rung below by one exact integer rule, and an entry exists only where the fold
   differs from what the generator gives at that address. The owner realm keeps it in memory, persists it
   at the checkpoint with its own watermark, and rebuilds the tail from the WAL after a crash. Because it
   is reconstructible, its format is a SOFT door (a rebuild pass), not a data-loss door. *Example: a quarry
   200 m wide. At 5 km the moon's shard ships eight 32 m cells that say "rock, half full, bottom octants
   only", and the client draws a dent in the derived hill instead of the intact hill.*
3. **The prune rule puts a HARD requirement on the generator crate.** "The fold differs from the
   generator" gives a sparse store only when the generator's COARSE answer is exactly the fold of the
   generator's own fine cells under the same address. This is not free, and the generator crate must
   supply it. See §1.3 and the ask to domain 03 in §7. Revision 1 froze this as a door and never defined
   it. That was the report's largest hole.
4. **The store is the realm store that already exists.** `StoreRole::RealmStore`
   (`crates/core/src/store_stamp.rs:64-75`) holds two families today, the berths and the body
   (`crates/sim/src/stub/built_store.rs:1-25`). The block families join that file, keyed by chunk, written
   by the owning shard alone through the existing `Store` seam (`crates/sim/src/io/mod.rs:477-495`): one
   redb transaction and one fsync per tick (`crates/io-prod/src/store.rs:372`). *Example: a hull's blocks
   live in `realm-ship-<id>.redb` (`crates/bins/src/lib.rs:2310-2333`), the same file that holds its body
   row today.* One qualification, MEASURED: `open_realm_store` returns "no store" unless `VD_REALM_STORE`
   or `VD_REALM_STORE_DIR` is set (`crates/bins/src/lib.rs:2360-2378`). The machinery exists. A deployment
   must switch it on.
5. **The diff lane is a window statement per chunk.** A realm states its chunk diffs to whoever looks in,
   exactly as it states its look today: send-on-change, reliable, one row per `(chunk, rung)`, the row
   is an ADDRESS plus a TLV bag (the VU contract: the client branches on tag presence only). Inside the
   realm the shard picks each observer's rungs from the pose it holds (SL7, the interest slice). From
   outside, a window is shared by every session behind one gateway, so the rung on the window is a
   FLOOR, not a per-person selection, and the gateway filters rows per session. See §3.3.
6. **Cross-shard edits forward to the PARENT** (DEFERRED D-39.1, `docs/design/DEFERRED.md:355-372`) on
   ONE new `InterShardFlow::BlockEdit` arm plus its ack arm, fenced, idempotent on the origin realm's own
   edit sequence, `Retained` on the durable outbox that is now on for every shard
   (`DEFERRED.md:916-950`). The child states the target point **in its own frame**. The parent adds the
   placement it authored and resolves its own chunk and cell (SL1 clauses 1, 3 and 4). The owner re-tests
   reach against the ORIGIN REALM's placement and extent, never against a person's pose (SL2).
7. **The first slice plants:** step ids 19 and 20 in one file, a `Bulk` routing class, the chunk-row
   message shapes, a rung-floor byte on the window open and on the relay request, the reliable client
   action carrier, and the two schema ids for the block families. All are free today and are a protocol
   event afterwards.

Nine asks under SL6 are listed in §7. Nothing is added by this report.

---

## 1. The edit-pyramid entry — the third frozen format

### 1.1 What one edit is

An edit is a change to one tier-0 cell in one chunk of one realm. Its durable record is the SECOND
format (the saved block record, another report's domain). This report needs only its address and the
fact that the address is a `Tier0Key` plus a `CellIndex`:

- the realm: `RealmId` (`crates/core/src/pose.rs:32-70`) — the store's file, never a field in a row;
- the chunk: a `ChunkKey` word with the tier in it, always tier 0 for a write (the grid family's format;
  the base's layout is at `block_system_design.md:2073-2085`);
- the cell: an 18-bit `CellIndex` inside the chunk (238,328 cells in a 62³ chunk, ESTIMATED from the
  chunk edge the grid domain freezes; if the edge changes, this width changes with it).

**What exists today:** none of `Tier0Key`, `ChunkKey`, `CellIndex`, `PyramidEntry`, `BlockStore`
(MEASURED: `grep -rn` over `crates/` finds only the reserved arm name `BlockEdit` at
`crates/wire/src/intershard.rs:34`, `crates/wire/src/lib.rs:24`, the `ChunkDelta` bulk kind at
`crates/wire/src/channels.rs:294`, and the capability guard `BlockEditNeedsVoxel` at
`crates/sim/src/capability.rs:98,131`).

### 1.2 What a coarse entry is

One entry per coarse cell whose summary differs from the generator's summary at the same address:

```
PyramidEntry — 8 bytes, one u64, MSB first (the base's record, re-validated)
  bits 63..46  cell       (18)  CellIndex inside the tier-L chunk
  bits 45..30  substance  (16)  the dominant substance id; the realm's EMPTY id when occ_mask == 0
  bits 29..22  occ_mask    (8)  octant o is set when child o holds anything
  bits 21..14  fill_256    (8)  mean fill of the eight children, 0..=255
  bits 13.. 0  reserved   (14)  MUST be zero; a non-zero value is refused, never decoded to Default
```

The fold, per rung, is exact integer arithmetic (`block_system_design.md:1548-1560`):

- `occ_mask` = one bit per non-empty child;
- `fill_256` = the rounded mean of the eight children's fill;
- `substance` = the child substance with the largest summed fill; ties go to the lowest id.

**Why this record serves the owner's requirements.**

| Requirement | How the entry serves it |
|---|---|
| V2.1 smooth terrain, still voxels | The entry states volume and octants, not a cube. The renderer report decides how a coarse cell carves the derived smooth surface. Open: whether a coarse entry also needs a surface-height datum (§1.7, decision D-1). |
| V2.2 a tree is one object | A tree cell's leaf summary must contribute its whole seed-derived extent to every coarse cell it spans. See §1.5. |
| V2.4 sub-metre blocks in one 1 m cell | The leaf summary reads the cell's total volume: `fill_256 = volume_units × 255 / CELL_VOLUME_UNITS`. A cell of quarter-metre detail folds like any other. Open: whether the stored record holds the sub-cells or the whole cell's volume — case 2 of §1.6. |
| V2.5 attachments take no space | Attachments never enter the pyramid (no volume). A STATIC attachment rides the tier-0 diff as its own TLV tag (§3.2). A LIVE attachment value does not ride this lane at all — case 4 of §1.6. |
| V2.6 params and client style | Style is derived on the client from substance and neighbours; the entry carries substance only. |
| V2.7 themes | A 16-bit substance id is 65,536 substances. |
| V2.9 seamless | The fly-away and fly-back gate M-15 (§6). It did not exist in revision 1: the answer was a pointer to nothing. The gate now names the quantisation bound and the pop it can cause (§1.4). |

*Example: a player builds a 20 m tower of steel on a moon. At 2 m cells the tower is 1,000 entries, each
"steel, full, all octants". At 8 m cells it is 27 entries. At 128 m cells it is ONE entry that says
"steel, `fill_256` = 1, one octant set". MEASURED arithmetic: 20³ = 8,000 m³ against 128³ = 2,097,152 m³
is 0.38 %, and 0.38 % of 255 rounds to 1; a 20 m tower fits inside one 64 m octant, so one bit is set.
Revision 1 said "2 % full, bottom octants" and was wrong by a factor of five and by three octants. The
corrected number matters, because one unit of 256 is the last unit before the tower disappears — §1.4.*

### 1.3 The prune rule, and the requirement it puts on the generator

Three rules, all from the base and re-validated:

1. **Prune on equality.** An entry is deleted the moment its summary equals `generate_summary(address)`.
   A player who digs a hole and fills it back with the same rock leaves nothing at any rung.
2. **Exit on the effective summary.** The walk-up stops at the first rung whose new summary equals the
   summary a reader would get today: the stored entry, or the generator when no entry exists. The base's
   loop compared against the stored entry only, so an unedited parent never matched and every edit
   walked every rung (`storage_and_streaming.md:833-884`). The fix is one line and is taken here.
3. **Write only from tier 0.** The write path takes a `Tier0Key`. A coarse write does not compile.

**The requirement rule 1 hides.** Rule 1 compares two numbers: the fold of the edited field, and
`generate_summary`. The store is sparse only when the two agree everywhere no player has dug. So the
generator's coarse answer must EQUAL the fold of the generator's own fine cells under the same address.
Write the two as one function of a field, `summary(field, address)`. Then:

- the pyramid entry is `summary(edited_field, address)`;
- `generate_summary(address)` is `summary(seed_field, address)`;
- the two agree on an untouched subtree **only if the generator's coarse evaluation is the exact fold of
  its own fine cells**.

The cheap way to coarsen fractal terrain is to drop octaves, and this project treats that as free
(`docs/design/window_lane.md:589-590`: *"fractal terrain coarsens free by dropping octaves"*). An
octave-dropped answer is NOT the fold of the fine cells. If the generator gives the octave-dropped
answer, nothing prunes: an untouched hill differs from its own fold at nearly every coarse cell, so the
moon's shard stores an entry for every coarse cell of every chunk it ever touches. The sparsity table
below does not survive that.

**So the design takes this requirement, and it is an ask to the generator domain (§7, A-7).** The
generator crate must expose an exact coarse summary at every legal rung whose value equals the fold of
its fine cells, and it must be cheap. The naive way is not cheap: one rung-13 address covers
8^13 = 549,755,813,888 tier-0 cells (ESTIMATED, 8^13), and at the MEASURED 12.19 ns per noise evaluation
(`scripts/noisebench`) that is about 1.9 hours for ONE coarse cell (ESTIMATED).

**The alternative, and why it is rejected.** One could prune on "no authored cell below" instead of on
equality with the generator. Then the generator's coarse cost does not matter. But the entry a client
receives is an ABSOLUTE summary, and the client draws the generator's coarse shape in the neighbouring
cell that has no entry. Wherever the two disagree, the drawn surface steps. *Example: a player digs one
hole on a moon. The coarse cell that holds the hole is drawn from the entry; the coarse cell beside it is
drawn from the generator. A cliff appears along the line between them, and it is straight, because it
follows the cell boundary.* That is the "detail-by-box" seam SL8 names. The alternative is rejected.

**The bench this creates.** M-16 in §6: for 10,000 random coarse addresses at every legal rung, compare
`generate_summary(address)` against the fold of its children, and measure the cost of both. The gate is
red on one differing entry.

Storage per edit, ESTIMATED from the base's series (`block_system_design.md:1643-1660`,
`storage_and_streaming.md:885-914`), and true only when the requirement above holds:

| Edit shape | Entries per edit | Bytes per edit |
|---|---|---|
| A quarry or a filled base (3-D cluster) | 0.14 | 1.1 |
| A wall or a floor (2-D) | 0.33 | 2.7 |
| A pipe one cell thick (1-D) | 1.0 | 8 |
| Ten thousand players leaving marks 57 m apart over a whole moon | **6.1** | **49** |

The last row is the one the base's table missed. At 100 M scattered edits it is ESTIMATED 4.9 GB of
pyramid against 0.8 GB of tier-0 delta. The per-realm byte budget must be sized against the scattered
row, not the clustered one (decision D-4).

### 1.4 The quantisation bound — when an edit stops being representable

`fill_256` is 8 bits, so a coarse cell states volume in units of 1/255 of its own volume. An edit whose
volume is under half a unit of a coarse cell rounds away, the entry equals the generator, and rule 1
DELETES it. The edit is then invisible at that rung and above.

**The bound, MEASURED by arithmetic on the fold rule.** A coarse cell of edge E holds E³ of volume. Half
a unit is E³/512. So an edit of volume V stops being representable at the first rung whose edge E obeys
E³ > 512 V.

*Example, worked through the fold one rung at a time: a player digs a quarry 20 m on a side into solid
rock on a moon, so V = 8,000 m³.*

| Coarse edge | Arithmetic | `fill_256` | Entry kept? |
|---|---|---|---|
| 64 m | 8,000 / 262,144 = 3.05 %; 255 × 0.9695 = 247.2 | 247 | yes |
| 128 m | mean of (247 + 7 × 255) / 8 = 254.0 | 254 | yes |
| 256 m | mean of (254 + 7 × 255) / 8 = 254.875 | **255** | **no — pruned** |

So the quarry survives to the 128 m rung and vanishes at the 256 m rung. `occ_mask` and `substance` do
not save it: every child still holds rock, so the mask and the substance already match the generator.

**What this means for SL8.** The step a player can see at the rung boundary is at most one unit of the
coarse cell's volume, which is a depth change of about E/255 spread over the cell. At the 128 m rung that
is about a half metre over a 128 m cell, at a distance where a 128 m cell is a few pixels. It may be under
the drawable floor the owner set on 2026-09-06 (1 px at 45° over 720 rows). It may not. **This is
UNMEASURED, and it is what gate M-15 owes.** Revision 1 claimed the picture never jumps. That was an
argument, not a measurement, and the standing rule of method forbids it.

**DISPUTED: the feasibility refuter's F-4 says the 20 m quarry "rounds from 255 to 255" at the 128 m cell
and blinks out there.** The fold arithmetic above gives 254 at 128 m and 255 at 256 m, so the vanish
happens one rung later than the refuter states. The finding itself stands and is taken; only the rung
moves. The refuter's conclusion — that revision 1's answer to V2.9 pointed at a gate that does not exist
— is correct, and M-15 and decision D-10 fix it.

### 1.5 The one-object feature rule (V2.2)

A seed-placed tree is static shape (SL10 clause 1): the client derives its trunk and canopy at every rung
with no entry anywhere. When a player fells it, the tree's cell becomes an authored cell. The base's leaf
rule summarises only that one cell. That leaves the canopy's coarse cells, twenty metres above, with no
entry, so a far client keeps drawing a canopy the server no longer has. This is a terrain-fact
divergence, which the ladder's own law forbids.

**The rule this report adds:** a feature cell's leaf summary contributes to every coarse cell the feature
SPANS, and the generator crate reports that span from the seed on both hosts. Felling a tree then dirties
every coarse cell the canopy stood in, and the walk-up writes entries there that say "empty". No format
change is needed; the update rule changes. The entry count per felled tree is UNMEASURED (§6, M-12).

### 1.6 Five cases the rule must answer

Revision 1 named none of these. Each is a real player action.

1. **A mined cell under a placed block.** A player places a steel beam on a moon, then mines the rock the
   beam stood on. Two cells are authored: one holds steel, one holds nothing. Rule 1 prunes on equality
   with the SEED, and the placed beam is not the seed, so the beam's cell never prunes and its coarse
   entries never prune. The mined cell's coarse entry prunes only if the whole subtree returns to the
   seed's volume, which the beam prevents. **The rule:** the fold reads the RESULT of the column, never
   the history. An authored empty cell and an authored full cell fold like any other two cells. Nothing
   new is needed; the report states it so that the two tags are not read as two truths.
2. **A sub-metre block on a slope.** A quarter-metre block sits in a 1 m cell whose other volume is the
   seed's rock, cut by the slope. The stored leaf record must state the WHOLE cell's resulting volume,
   never the authored sub-cells alone. **The reason is SL10 clause 4:** if the record held only the
   authored sub-cells, the drawn cell would change whenever the generator's slope changed, and a player's
   block would move. Where the SUB-CELL geometry lives is the block-record domain's second format, and
   this report asks it in §7, A-8.
3. **A tree standing on a mined edge.** A player mines the ground under a seed-placed tree. The tree's own
   cell is not authored, so no entry exists there, and the client keeps drawing a tree that hangs over the
   new hole. **The rule:** the generator crate reports which feature a tier-0 cell SUPPORTS, and an edit
   to a supporting cell dirties the supported feature's cells, which the walk-up then writes as authored
   empty. This is the same span mechanism as §1.5, read downward instead of upward. The cost is
   UNMEASURED (M-12, extended).
4. **A live attachment value — a HUD reading, a turret joint angle (V2.5).** This does NOT ride the
   chunk-diff lane. The lane is reliable, ordered and PACED (§3.6), so a moving joint would queue behind
   a 23 MB city catch-up. A live value is not a diff of the world's shape; it is live state at the tick
   rate. It belongs on the realm's own per-tick statement lane, and its carrier is the signal-bus
   domain's (the reserved `InterShardFlow::Signal`, P9). Tag 5 on the chunk row carries an attachment's
   PRESENCE and its static configuration only. Recorded as decision D-11.
5. **An edit while the author is crossing.** §4.5 covers a re-shard of the TARGET realm. It did not cover
   the AUTHOR crossing. *Example: a pilot inside a hull digs at tick T; the hull's berth is released and
   the pilot's session re-homes to the moon at tick T; the diff ships at tick T+1.* **The rule:** the ack
   is addressed to the ORIGIN REALM and its fence, never to a session. The origin shard holds the echo
   until it can address the session, and drops it, counted, when the session has left. The diff itself is
   unaffected: the pilot now stands on the moon and receives the moon's rows through the moon's own lane.
   The forward's idempotency key (§4.3) belongs to the origin realm, so a redelivery after the crossing is
   still a no-op.

### 1.7 The one-way doors of the entry

| Door | Hard or soft | Why |
|---|---|---|
| The `CellIndex` width (18 bits) and linearisation | **Hard** — shared with the saved block record and the chunk edge | It is the address of every persisted row. |
| The substance id width (16 bits) | **Hard** — shared with the block record | Renumbering reinterprets every saved cell. |
| The fold rule (mask, mean fill, dominant-by-fill, lowest-id tie-break) and the exact coarse `generate_summary` in the generator crate | **Hard on the wire, soft on disk** | Both hosts must compute the same absent-entry meaning (SL10 no-drift), so a change is a protocol event and a world-generation tag change. The disk copy rebuilds. |
| The 8-byte layout itself | **Soft** | Derived data: a layout change is a rebuild pass over every realm store (ESTIMATED 54 s of CPU per 110 M entries, `storage_and_streaming.md:1052-1055`) plus a PROTO_MINOR bump. Never a data loss. |
| A surface-height datum for smooth terrain (D-1) | **Decide before the first world is saved** | Adding a field later is a rebuild, but the RENDER contract on the wire changes. |
| A sticky "differs from the seed" bit against quantisation (D-10) | **Decide before the first world is saved** | It changes the 8-byte layout's meaning and the prune rule together. |

The base named the entry "PERSISTED, PERMANENT, ONE-WAY DOOR (row 37 / door 43)". Under the watermark
design that claim is too strong: see §9.

---

## 2. The durable store per realm

### 2.1 What exists today (MEASURED by reading the code)

- A realm's own store exists since D-MOVE-2: `StoreRole::RealmStore = 4`
  (`crates/core/src/store_stamp.rs:64-75`), opened at boot by the shard
  (`crates/bins/src/bin/shard.rs:82-100`) through `open_realm_store`
  (`crates/bins/src/lib.rs:2360-2395`), named `realm-<kind>-<number>.redb` from the `RealmId`
  (`crates/bins/src/lib.rs:2310-2333`). It returns "no store" unless `VD_REALM_STORE` or
  `VD_REALM_STORE_DIR` is set (`crates/bins/src/lib.rs:2367-2378`).
- It holds two families with one prefix byte each and TLV-framed rows: the berths (`BERTH = 1`) and the
  body (`crates/sim/src/stub/built_store.rs:20-25, 84-90`); schema ids 30 and 31 are taken
  (`built_store.rs:30-33`).
- The saved-data label refuses a file from another world before any row is read
  (`crates/core/src/store_stamp.rs:1-30`; DEFERRED D-48-S1, `docs/design/DEFERRED.md:3661-3700`). A
  per-family format version does not exist (D-48 🟥, `DEFERRED.md:3122-3160`).
- A row's TLV frame refuses a foreign SCHEMA before it decodes a field (`TlvReader::parse`,
  `crates/core/src/tlv.rs:178-192`, `SchemaMismatch`), and both families call it
  (`crates/sim/src/stub/built_store.rs:143,170`).
- The `Store` seam is `put / delete / scan(prefix) / commit / flush` (`crates/sim/src/io/mod.rs:477-495`).
  The redb backend stages puts in RAM and hands one batch per `commit` to an off-tick writer, which
  applies it in ONE write transaction and ONE fsync, all or nothing (`crates/io-prod/src/store.rs:1-41,
  340-380`). `commit` blocks on the PRIOR batch's durability, so a disk stall stalls the tick loudly, and
  a counter says so (`backpressure_stalls`, `crates/io-prod/src/store.rs:289-295`).
- The durable outbox is a different store with a different job: it mirrors RETAINED transport frames so a
  producer-less reliable send survives a crash (`crates/io-prod/src/outbox.rs`; on for every shard since
  2026-09-06, `DEFERRED.md:916-950`). Its group commit is one barrier per batch of retained rows
  (`crates/io-prod/src/mesh.rs:508-511, 2968`).

### 2.2 The design

**Keyed by chunk.** The block families are keyed by the chunk word, under one prefix byte per family in
the realm's own file. A column key is not needed: the chunk word orders `sector → tier → a → b → c`, so
one rung and one column are each one contiguous range (`block_system_design.md:2073-2085`).

| Family | Key | Value | Written when |
|---|---|---|---|
| `block_wal` | `(tier-0 chunk, batch seq)` | the batch's records, sorted, dedup-to-final | every tick that had edits |
| `chunk_delta` | tier-0 chunk | the chunk's authored cells, sparse or masked-dense, TLV-framed | at the checkpoint |
| `chunk_pyramid` | `(tier L, chunk)` | that chunk's sparse entry list, same two encodings | at the checkpoint |
| `chunk_meta` | chunk | last-ticked stamp, the pyramid watermark | at the checkpoint |
| side tables (damage, config, attachments) | the block record domain's | | |

The two encodings are chosen by exact size, never by a constant: the pyramid's crossover moves between
3,725 and 13,250 cells with the palette (`storage_and_streaming.md:1010-1033`).

**Single writer.** The owning shard is the only process that opens the file for writing. Today that is
structural by deployment (one shard per realm, node-per-realm) and by the label (a file from another
world refuses). What is missing is the owner fence: a shard whose realm fence is not strictly greater
than the file's `last_owner_fence` must refuse to open (the base's §3.9.4 rule; not in the code,
MEASURED: `grep last_owner_fence crates/` finds nothing). The fence is the mechanism; a file lock is a
courtesy, because the cloud volume is network-backed.

**A file name two realms can share — a data-loss door.** `RealmId::System(seed)` collapses across
galaxies. The type says so itself: *"a System seed collapses across galaxies … the `path` is the
collision-free disambiguator"* (`crates/core/src/realm_coord.rs:1-8`). The store's file name is built
from the `RealmId` (`crates/bins/src/lib.rs:2310-2333`), and the directory and `ChildRealmNodes` key on
the same lossy id (`crates/sim/src/stub/realm_head.rs:92-94`, deferred as D-RLM-10). *Example: two star
systems with the same seed, one in each of two galaxies, both open `realm-system-<seed>.redb`, and each
one's players overwrite the other's buildings.* Identity is another domain's. The DOOR is this domain's,
because this domain is what first writes something a player would miss. It is listed in §10 with the same
deadline as the rest: before the first world is saved.

**The type-level proof.** The writable handle is obtained only through the directory read that named
this shard the owner; its mutating methods take `&OwnedBlockStore` and a `Tier0Key`. A non-owner holds
no handle with a mutating method. This is the base's design (`block_system_design.md:4556-4571`) and
it stands.

**The write path of one edit**, in game words: *a player on a moon breaks a rock.*

1. The moon's shard validates the action (reach against the pose IT holds, occupancy, rate, nonce).
2. It stages the WAL frame (`put`) and runs the walk-up in memory.
3. At the tick's end, `commit` hands the tick's batch to the writer: one transaction, one fsync, all
   edits of the tick together. This IS the group commit for the realm store; nothing new is needed.
4. On the NEXT tick, when `commit` returns, the previous batch is durable. The shard then ships the diff
   rows for that batch (§3). A diff never precedes its durability, so a client never draws a wall the
   restarted shard does not have. Cost: one tick of latency (20 ms at 50 Hz, ESTIMATED).
5. At the checkpoint: fold the WAL into `chunk_delta`, write the pyramid blocks and the watermark, fsync,
   record the durable LSN, fsync, prune the WAL at or below it (the three-step discipline P6 already
   binds).

**What the moon does when the disk is slow — SL8 seam kind seven, the tick hitch.** Step 4 makes the edit
lane wait on a durable batch, and `commit` blocks on the prior fsync
(`crates/io-prod/src/store.rs:289-295`). Revision 1 said what good looks like and never said what happens
when it is bad. **The degrade, stated:**

- The moon keeps ticking. Poses, frames and the 20 Hz datagrams are untouched, because none of them holds
  a durable wait. A pilot who flies past the moon sees nothing at all.
- Only the EDIT lane parks. The shard stops accepting new edit batches into the WAL and counts the parked
  ticks (`backpressure_stalls` plus an edit-lane counter of its own).
- A player who is already swinging a pick sees the swing complete and the rock break LATE, by the number
  of parked ticks. The player never freezes and the client never blanks.
- Past a stated bound of parked ticks the shard REFUSES new edits with a typed reason, and the client
  says the ground is busy. A typed refusal is the doctrine; a silent drop is a defect.
- The bound is a named number in `BlockStoreTuning` (§5.2, plant 9), never a literal.

**What a heavily played planet costs**, ESTIMATED from the base's arithmetic:

| Item | Bytes | Source |
|---|---|---|
| 100 M clustered edits, tier-0 delta | 0.8 GB sparse + a few hundred MB masked-dense | `block_system_design.md:4514-4521` |
| Pyramid, clustered | 0.9 GB | `block_system_design.md:1655-1660` |
| Pyramid, scattered (one mark per 57 m) | **4.9 GB** | `storage_and_streaming.md:895-905` |
| A ten-million-block city | 41 MB (30 MB delta + 11 MB pyramid) | `storage_and_streaming.md:518-547` |
| Write amplification, one 8-byte pyramid entry in a built region | 5,508× (a whole-value redb rewrite) | `storage_and_streaming.md:1057-1086` |

The amplification is bounded by the checkpoint cadence, not by a design decision. Two remedies: the
pyramid rides the checkpoint (§1.5), and `pyramid_storage_block_cells` in the store's tuning struct lets
a dense chunk split into sub-blocks. The second is a persisted-key decision (door). Every number in the
list is UNMEASURED on this codebase (§6).

**Dormant advance.** A sleeping moon has no writer. The base's answer stands: the advance is a closed-form
fold the owner evaluates at first read of each chunk after spin-up, emitting ordinary WAL edits under its
own fence (`block_system_design.md:4573-4603`; per chunk, not realm-wide, per
`storage_and_streaming.md:1104-1123`). No second writer class.

---

## 3. The diff lane to the client

### 3.1 What the client needs, and what it already has

Under SL10 the client derives the static shape. It needs from the server exactly:

- for every chunk in its interest, at the rung it draws that chunk: the authored cells (tier 0) or the
  pyramid entries (tier L), and later their changes;
- a way to know which chunks in its interest carry a diff at all, so absence means "the seed's shape"
  and not "not yet arrived".

What exists today on the picture path (MEASURED by reading the code):

- A realm states its look to whoever holds a window on it: `ShardToGateway::WindowBody`, send-on-change
  by a digest per subject (`crates/sim/src/stub/window.rs:55-60, 386-455`), reliable, on the
  session-reply lane (`crates/wire/src/session_flow.rs:315-336`).
- The look is a TLV bag with skip-unknown tags (`crates/core/src/look.rs:1-40`, schema 20). The
  composed row the client draws is "a pose + a bag" (`SceneRow`, `crates/wire/src/channels.rs:366-395`);
  the client branches on tag presence, never on realm kind.
- A realm ships each cube of its occupants as ONE body to the sessions that hold that cube:
  `ShardToGateway::FrameFor { recipients }` (`session_flow.rs:487-497`, produced at
  `crates/sim/src/stub/frames.rs:261-290`, fanned at
  `crates/connection-plane/src/gateway/client.rs:494-512`), and tells an observer once when an occupant
  left its interest (`EntityOutOfInterest`, `Retained`, `frames.rs:300-315`).
- A subscriber states the digest of what it holds on the window open, and the shard re-sends only what
  changed (`session_flow.rs:123-152`; `window.rs:150-170`).
- **A window is opened per `(shard, scope)` and is SHARED by every session behind that gateway.** The
  gateway refuses to open a second window with the same shard and scope
  (`crates/connection-plane/src/gateway/window_lane.rs:287-296`); the shard keys it `(NodeId, WindowId)`
  — the gateway NODE, never a session (`crates/sim/src/stub/window.rs:41`). `WindowScope` names no
  person: it has `Occupants` and `Child(RealmId)` only (`crates/wire/src/session_flow.rs:582-590`).
- **A live realm is NOT windowed from outside.** There is deliberately no `Observed` scope: watching a
  live realm from outside rides the PARENT RELAY, where the parent forwards the child's own statements
  sealed and verbatim (`crates/wire/src/session_flow.rs:575-581`; `WindowRelayed` at `:344-383`;
  `RelayedStatement` at `:679-696`; `InterShardFlow::WindowRelay` at `intershard.rs:468`). The carrier's
  arity is two levels and a third is unrepresentable (`LOOK_CARRIER_ARITY`, `session_flow.rs:701-710`).
- A bulk carrier is declared but unroutable: `BulkMsg::Blob { kind: ChunkSnapshot | ChunkDelta |
  Catalog }` (`channels.rs:284-297`); `MsgClass` has no bulk arm (`crates/sim/src/io/mod.rs:51-85`);
  no crate outside `vd-wire` names `BulkMsg` (MEASURED; DEFERRED D-4, `DEFERRED.md:869`).

So the lane's SHAPE exists; the chunk row, the routing class and the interest-by-chunk do not.

### 3.2 The chunk row

A chunk diff is a window statement about one address of the stating realm:

```
ChunkRow {
    chunk: ChunkKey,        // the address in the STATING realm's OWN frame; carries the rung
    at:    UniverseTick,    // when the owner stated it (send-on-change, newest wins)
    bag:   TlvBytes,        // schema CHUNK_ROW; tags below; unknown tags skipped
}
tags:  1 = authored cells, sparse (tier 0)     2 = authored cells, masked-dense (tier 0)
       3 = pyramid entries (tier L)             4 = cells returned to the seed (a revert list)
       5 = attachment presence and static config (no volume)     6 = codec (reserved, always none)
       7 = feature state (growth stage, damage) — the block record domain's tags, reserved here
```

The row carries no pose. The address IS the position in the STATING realm's frame, and the client already
places the realm by its composed row (`SceneRow.pose`). The client keys chunks by `(realm, chunk)`,
evaluates the generator at the same address, and applies the bag. A tag it does not know is skipped: a
client that predates attachments still draws the cells. This keeps the VU contract: a row is an address
plus a bag; the client never branches on what kind of realm stated it. *Example: a station's shard states
row `(tier 0, chunk 4/17/2)` with tag 1 = twelve steel cells. The client draws twelve steel cells inside
the station's derived hull. A moon's shard states the same shape of row; the client does the same thing.*

**Note on SL1.** A realm that states its OWN chunk address to a gateway is lawful: the address lies
inside the realm's own frame and says nothing about where the realm is. What is unlawful is a CHILD that
states a PARENT-frame address, and §4.2 no longer does that.

### 3.3 Who decides which rows a client receives

Two regimes, one message, both decided where the knowledge lawfully lives.

**Inside the realm (the observer stands in it).** The shard holds the observer's pose (SL7: the parent
holds its own occupants). It computes the chunk interest per observer the way `plan_interest` computes
cube interest today: tier 0 within the near disc, rung L in its annulus (rung L out to `786 × 2^L` m at
two pixels per cell, ESTIMATED, `block_system_design_addendum_2.md` §C.2), a lead that grows with speed,
and hysteresis one step out. One row ships per `(chunk, rung, tick)` to the sessions that hold it, with
named recipients like `FrameFor`.

**What this needs that does not exist.** `plan_interest` is OBSERVER-MAJOR: it walks every observer and,
for each one, range-reads a slab of occupant CUBES (`crates/sim/src/stub/interest.rs:129-160`). Its
OUTPUT is cube-major (`InterestPlan.recipients: BTreeMap<CellKey, Vec<SessionId>>`,
`interest.rs:120-123`), which is the shape a dirty chunk needs — but the keys are occupant cubes, not
`(chunk, rung)` pairs, and building the map costs observers × slab per tick. **So the chunk lane needs a
NEW index from `(chunk, rung)` to the sessions that hold it.** Its owner is the shard's own interest
system; it is rebuilt when an observer's held set changes, never per edit; its cost is a bench (M-11).
Revision 1 presented this index as the existing pattern. It is not. The existing `recipients` map is the
SHAPE to copy, and copying it is work this report now names.

**Outside the realm (the observer looks in from another realm).** The realm never learns the observer's
pose (SL2). Three facts of the code decide the design here, and revision 1 got all three wrong.

1. A window is per `(shard, scope)`, shared by every session behind that gateway
   (`window_lane.rs:287-296`). *Example: two pilots behind one gateway watch the same moon. One sits in
   orbit; one is 20 km out and descending. They share ONE window.* A single rung on that window has no
   single correct value.
2. A live child realm is not windowed at all from outside. Its rows reach the gateway sealed inside its
   PARENT's relay (`session_flow.rs:575-581, 344-383`). So a request for a rung travels to the parent,
   and the parent must pass it down, or the child states at a rung nobody asked for.
3. The gateway composes the chain, so the gateway alone knows every session's angular size of the realm
   (`window_lane.md:578-593`, the recorded owner Q&A of 2026-08-15).

**The design that follows.** The rung on the window is a FLOOR, computed by the gateway as the FINEST
rung any session behind that window needs, and the gateway drops rows per session on top of it — the thin
per-session band filter the lane already runs (`window_lane.md:578-580`). Then:

- the realm ships at the floor, so it never ships a rung-0 city to a window whose closest looker is in
  orbit;
- the moon's egress is sized by the CLOSEST pilot behind each gateway, never by the average one. That
  cost is real, and revision 1 did not state it. It is the price of one window per gateway. The
  alternative — one window per session — pays the static roster per session instead of per gateway
  (14.2 MB and 28.4 MB/s per window at the S12 census, `crates/wire/src/session_flow.rs:136-137`).
  Neither is free;
- the floor must also ride the relay leg for a child realm. That is a new field on the relay's downward
  request, and it is ask A-4 in §7. Without it a child realm cannot know what rung its watchers need and
  must ship its finest.

**Bench M-17:** one close pilot and 127 distant ones behind ONE gateway, watching one moon that holds a
city. Measure the moon's egress and the gateway's per-session drop rate.

*Example, restated honestly: a pilot in a hull descends toward a moon. The gateway holds one window on
the moon's parent for that moon's relay, and asks for the floor its closest session needs. From orbit the
floor is rung 8 and the moon states thirty coarse rows; the client sees a speck where the city is. At
20 km the floor becomes rung 5 and the city becomes blocks. The hull lands; the pilot walks out and
becomes the moon's occupant; the moon's shard now holds the pilot's pose and serves tier 0 around the
boots.* **Whether the picture holds across that hand-over is UNMEASURED. It is gate M-13.**

**The hand-over rule the client must obey.** The decider changes when the pilot walks out of the hull:
the gateway's floor stops deciding and the moon's own per-observer interest starts. Two rules make that
survivable, and both are now stated instead of assumed:

- **The client KEEPS its `(realm, chunk)` cache across a re-home.** The cache is keyed by the realm and
  the address, and neither changes when the observer's authority changes. A client that dropped the cache
  would re-fetch a whole city at the moment of the walk-out: a lane flood and a black frame in the same
  second, two of SL8's eleven seams.
- **The client never draws a chunk at a FINER rung than the rung whose diff it holds.** See §3.4.

### 3.4 Catch-up when a chunk enters interest, and what is drawn meanwhile

1. The shard sends the chunk's current diff record at the rung the observer holds it: one row. A chunk
   with no diff at that rung sends nothing.
2. So that absence has one meaning, the shard sends a MANIFEST. **The manifest is HIERARCHICAL:** one
   digest per COARSE chunk that covers a whole column, where absence at a coarse key means "nothing below
   this key differs from the seed". The client walks down only where a coarse digest is present. A flat
   manifest, which revision 1 proposed, must list every differing key in the cube, so its cost follows
   the realm's surface instead of the player's interest.
3. **What the client draws while a manifest is outstanding.** It draws the derived shape at the COARSEST
   rung it holds, and it never draws a chunk at a finer rung than the rung whose diff it holds. So a
   chunk with no coarse digest is drawn from the seed at once, and a chunk with a coarse digest is drawn
   at the coarse rung until the finer row arrives. This rules out both named seams: nothing is undrawn
   (no black frame) and nothing sharpens before its diff (no arrival pop). Revision 1 stated neither
   half. Measured in M-13 and M-15.
4. Coarse first: the coarse rows of a column are small and arrive first. A ten-million-block city is
   ESTIMATED 5 KB to know it exists, 160 KB to collide with it, 23 MB to draw it from the middle at
   tier 0 (`storage_and_streaming.md:570-582`).
5. A subscriber states the digests it holds on re-open (the existing `static_held` pattern), so a
   reconnecting client is not re-served a city it holds.

**How big is the manifest? UNMEASURED, and the sizing case is not the city.** Bench M-18: a moon with
10,000 players' marks 57 m apart, an observer in orbit and an observer on the ground, swept at every rung.

**DISPUTED: the feasibility refuter's F-3 estimates a 7 MB rung-7 manifest on a 3,000 km moon, on the
assumption that essentially every rung-7 chunk carries a diff. The prune rule of §1.3 and the
quantisation bound of §1.4 refuse that assumption.** A rung-7 chunk has an edge of 62 × 128 ≈ 7,936 m
(ESTIMATED from the grid domain's 62-cell chunk at 1 m cells). One 1 m mark is about 1 m³ against
5.0e11 m³, which is far under the half-unit threshold E³/512, so the mark cannot move that cell's
`fill_256` and the entry PRUNES. The rung-7 manifest of a scattered moon is therefore nearly empty, not
7 MB. **The finding is still taken**, because the refuter is right that the manifest was never sized and
never benched, and right that a flat manifest's cost follows the realm and not the player. The
hierarchical form and M-18 answer it. The worst case is the MIDDLE rungs, where an edit is just large
enough to survive the prune, and M-18 must sweep them.

### 3.5 Steady state

One edit batch on chunk C in tick T produces, at tick T+1 (after durability, §2.2 step 4):

- one tier-0 row to every session holding C at tier 0, carrying only the batch's changed cells (tag 1)
  and the cells returned to the seed (tag 4);
- one rung-L row to every session or window holding C's ancestor at rung L, carrying only the pyramid
  entries that changed (tag 3), which by the walk-up is at most a handful — and often none, because the
  effective-summary exit ends the walk at rung 1 or 2 for an edit inside solid rock.

So a busy quarry is a torrent to the players in it and a single coarse cell that changes colour to a
player 50 km away (`block_system_design.md:5089-5100`). Message count per tick is bounded by dirty chunks
× holders per chunk, never by observers × edits (the coalescing rule the storage doc names D-12) — once
the new `(chunk, rung)` index of §3.3 exists.

### 3.6 Which lane class

The chunk rows must be reliable and ordered per `(realm, chunk)`: a lost row is a terrain-fact
divergence on the client's screen. They must not head-of-line-block the reliable control lane that
carries the scene deltas and the transfer cut marker. So they ride their own reliable, PACED class — the
`BulkMsg` carrier the wire already reserves, on its own stream. The manifest rides the same class.
Pacing comes from the live RTT so a 23 MB city never starves the 20 Hz datagrams (the base's rule;
SPIKE-3a MEASURED 180 MB/s of bulk on one loopback connection with sub-millisecond snapshot p99,
`memory: project_spike3a_latency_gate`; loopback proves the server side only).

A LIVE attachment value never rides this class. See case 4 of §1.6.

---

## 4. Cross-shard edits

### 4.1 The rule

An edit is applied ONLY by the owner of the target realm, and only when the origin is:

- an occupant of the target realm itself (the local path: the pilot walks on the moon; the moon's shard
  holds the pose, validates reach, and writes), or
- an occupant of a DIRECT CHILD realm of the target, whose placement the target authors (the forward
  path: the pilot stands inside a landed hull and digs through the open door).

Anything else is refused. A person cannot reach a sibling realm's surface from a metre away, because if
they were a metre away they would be inside it (containment re-homes by coordinate).

**A consequence revision 1 missed: the target is ALWAYS the parent.** The second bullet says so. So the
forward needs no directory lookup and no target-realm field. The origin shard already holds its parent's
node: `ParentRealmNode` (`crates/sim/src/stub/realm_head.rs:75-82`), resolved once by a directory head
read and re-read on the realm-recheck cadence so a parent re-home is observed. Every up-lane already
routes by it, and the reach emitter is the template (`crates/sim/src/stub/reach.rs:36-48`).

### 4.2 The forward path

*Example: a pilot inside a hull, berthed on a moon, breaks a rock beside the ramp.*

1. The client's action reaches the hull's shard, because the session's authority is the hull (the
   gateway routes input by authority; `route_input` cannot express a second target, D-39.1).
2. **The hull states the target point IN THE HULL'S OWN FRAME.** It does not name a moon chunk. It does
   not name a moon cell. It states a point in metres in its own frame, plus the verb and the substance.
3. It pushes `InterShardFlow::BlockEdit(BlockEditForward)` to `ParentRealmNode`, `Retained`, so the
   durable outbox replays it after a crash. The payload: the hull-frame points and verbs, the ORIGIN
   REALM's coord and fence, the session id for the echo, and the origin's own edit sequence number.
4. **The moon resolves the address.** The moon adds the placement it authored for the hull — the berth
   the moon is the only writer of (SL1 clause 1) — to the hull-frame point, and reads off its OWN chunk
   and cell. No other party can do this, and no other party may.
5. The moon admits the forward through the three guards every up-lane carries
   (`crates/sim/src/stub/reach.rs:78-101`): **misroute** (the origin coord's parent is this realm),
   **attestation** (the sender node is the directory's record for that child — `ChildRealmNodes`,
   `crates/sim/src/stub/realm_head.rs:83-96`), and **staleness** (the origin's fence and instant). Then
   the domain guards: the resolved cell is within `max_reach_m` of the hull's PLACEMENT plus the hull's
   EXTENT — the SL7 proxy, a realm's placement, never a person's pose (SL2); the per-session rate is
   under the cap; the edit sequence is above the origin's low-water mark (§4.3).
6. It writes locally under its own fence and acks `BlockEditAck::Applied` (or `Refused(reason)`) back to
   the hull's shard, keyed by the same origin and sequence. The hull's shard forwards the outcome to the
   gateway as an `EventMsg` for that session, or drops it counted when the session has left (§1.6 case 5).
7. The diff reaches the pilot's client through the moon's rows (§3), never through the hull.

**Why step 2 changed (SL1).** Revision 1 sent "the target realm and chunk". A chunk key is an address in
the STATING realm's frame (§3.2), and the stating realm there was the MOON. The hull could name a
moon-frame cell only by adding the berth reading the moon gave it. SL1 clause 3 refuses a placement a
child asserts about itself, and clause 4 says *"what you are told about yourself, you NEVER pass on"*.
Subtract the local offset from that chunk key and the hull's own placement falls out. The hull-frame form
obeys clause 1 exactly — *"going up the child ships its own-frame pose and the parent adds"* — and it is
simpler: the moon already holds the berth and the store, so it does the whole resolution in one place.

**The pose that no longer crosses.** The base's admission rule 2 shipped the ORIGINATING SESSION'S SERVER
POSE to the owner (`block_system_design.md:4671-4675`). That is an occupant pose crossing a realm
boundary and is refused by SL2 (clarified 2026-08-24). The hull-placement proxy replaces it; the error is
bounded by the hull's size, which is the resolution the reach test has anyway.

### 4.3 Idempotency without a transfer

Revision 1 keyed the forward on `(TransferId, step 20)` and checked it against the applied-steps journal.
That is wrong twice, MEASURED:

- `TransferId` is *"minted by the orchestrator at saga creation"* (`crates/core/src/ids.rs:94-98`). A
  block edit has no saga and no orchestrator, so nobody mints one.
- `AppliedSteps` is an in-memory `BTreeSet<(TransferId, u32)>` whose own doc owes a RETENTION BOUND —
  *"drop a transfer's steps on its terminal … so a long-lived dest shard does not accumulate one entry
  per `(transfer, step)` forever"* (`crates/sim/src/stub/crossing_receive.rs:48-53`). A transfer has a
  terminal; a stream of edits does not. A pilot who digs from inside a hull forwards a batch per tick,
  and each batch would add a permanent row to a moon shard that runs for weeks. The durable
  `applied_steps` table does not exist (MEASURED: `grep -rn applied_steps crates/` finds doc comments and
  `crates/connection-plane/src/gateway/session.rs:109` only).

**The design instead.** The forward carries the ORIGIN REALM's own monotone edit sequence, minted by the
origin shard, one counter per origin realm, carried across a restart in the origin's own store. The owner
keeps ONE low-water mark per origin realm: a forward at or below it is a no-op; a forward above it
applies and advances it. That state is one number per direct child, never one row per edit, so it is
bounded by the child count, and it drops when the child leaves the roster. The ack echoes the sequence,
so the origin knows what to stop retrying. A redelivery from the durable outbox carries the same
sequence, because the sequence is inside the retained frame.

**Bench M-19:** 200 edits per second for one hour from one hull into one moon; measure the owner's
idempotency state at the end. It must be one row.

### 4.4 The hull edits the hull

The pilot is the hull realm's occupant; the hull's shard owns `realm-ship-<id>.redb`. Local path. A hull
is a realm like any other (V4 of 2026-09-01): no ship-specific branch on this path. **HR4 makes that an
argument until it is a fixture:** *"every feature passes the identical fixture on ≥2 shard kinds
(G-IDENTICAL) or it doesn't land"* (CLAUDE.md line 110), repeated for children at
`crates/sim/src/capability.rs:85-90`. Gate M-14 runs the identical edit fixture on a moon realm and on a
ship realm and compares both stores and both row sets byte for byte.

### 4.5 The fence discipline and the refusal path

- Every forward carries the origin's fence; a deposed origin's forward is refused and counted.
- The owner's low-water mark makes a redelivery a no-op; the outbox replays exactly the retained frame.
- Every refusal is typed and acked; a silent drop is a defect (the transport doctrine).
- During a re-shard of the target realm, `BLOCK_STORE_FLUSH_STEP = 19` drains the WAL, fsyncs, writes
  the owner fence and releases the handle BEFORE the pose flush (step 7), with its own ack. On failure
  the directory CAS does not commit and the source keeps its store. Never reuse step 7
  (`crates/wire/src/intershard.rs:65` documents it as the pose flush; the base's reasoning at
  `block_system_design.md:4531-4553` stands).
- A forward that arrives at a shard that is no longer this origin's parent is refused by the misroute
  guard and counted. The origin re-reads its parent on the realm-recheck cadence and re-sends once; twice
  is a counted fault.
- The AUTHOR crossing is case 5 of §1.6.

---

## 5. The reserved wire plant — what must land in the first slice

### 5.1 What exists (MEASURED)

- `InterShardFlow` has **44 arms** (counted with `awk` over the enum at
  `crates/wire/src/intershard.rs:124-604`; the last discriminant is 43, `ReachStated`).
- **Eight arms are tombstones**, MEASURED by `grep -n "★TOMBSTONE" crates/wire/src/intershard.rs`:
  `OccupantInterest` (`:279`), `ProxySceneSet` (`:289`), `RealmCascade` (`:298`), `EntityInterest`
  (`:317`), `EntityCascade` (`:328`), `RealmObservation` (`:381`), `RealmShapeObservation` (`:400`),
  `ChildSceneSet` (`:416`). Two more tombstones live INSIDE the `Ghost` arm as `GhostFlow` variants
  (`Spawn` and `Delta`, `:993-1001`). Revision 1 said four and then listed five.
- The header prose says 16 (`intershard.rs:9`) and `lib.rs` says 24 (`crates/wire/src/lib.rs:11-12`).
- Step ids 0–6 are the route swap; 7–18 are taken (`intershard.rs:65-115`); 19 and up are free
  (MEASURED: no `_STEP: u32 = 19` or `20` exists).
- `ChannelKey` does not exist anywhere (MEASURED: `grep -rn ChannelKey crates/` is empty).
- The closed-set gate is `every_arm` + `arm_tripwire` + the durability golden pin
  (`crates/wire/tests/intershard_closed.rs:47, 623, 1172-1252`); the producer-less set is pinned at
  its current size and grows only with a deliberate edit.
- `RealmKindTag` already has `Star = 6` and `Ship = 7` (`crates/core/src/realm_path.rs:44-62`).
- The client's discrete actions ride the UNRELIABLE input datagram's `action_bits`
  (`crates/wire/src/channels.rs:262-267`), whose own comment names the reliable carrier as owed with two
  consumers (block edits, PvP fire).

### 5.2 The plant

| # | Plant | Where | Why now |
|---|---|---|---|
| 1 | `BLOCK_STORE_FLUSH_STEP = 19`, `BLOCK_EDIT_FORWARD_STEP = 20`, and the signal domain's 21–24, in ONE file with the disjointness test extended | `crates/wire/src/intershard.rs` | a step-id collision aliases two operations in a journal: silent, durable, unrecoverable |
| 2 | ONE `BlockEdit(BlockEditForward)` arm carrying HULL-FRAME points, an inner verb enum (`Place`, `Break`, `Batch`), the origin coord, the origin fence and the origin edit sequence, and ONE `BlockEditAck` arm; classes `SideEffecting` + `ProducerLessReliable`; represented in `every_arm` and in the golden pin in the same commit | same | postcard is positional; an arm's field order freezes when it ships |
| 3 | `MsgClass::Bulk` (reliable, paced), its `Reliability` and `class_to_byte` entries, appended | `crates/sim/src/io/mod.rs`, `crates/io-prod/src/outbox.rs` | the carrier class is the decoder's discriminator; nothing routes `BulkMsg` today |
| 4 | `BulkMsg::ChunkRows { realm, rows: Vec<ChunkRow> }` and `BulkMsg::ChunkManifest { realm, coarse_key, digests }` as appended variants; the `ChunkRow` bag schema id and its seven tags | `crates/wire/src/channels.rs`, `crates/core/src/look.rs` (or a sibling codec module) | the tier rides inside `ChunkKey`; no tier field on the message |
| 5 | `ShardToGateway::BulkFor { realm_fence, recipients, window: Option<WindowId>, bytes }` | `crates/wire/src/session_flow.rs` | the gateway forwards bytes it never decodes, to named sessions or to a window's holders |
| 6 | `rung_floor: u8` on `GatewayToShard::WindowOpen` ("serve this rung and coarser; 0 = every rung"), AND the same field on the relay's downward request, so a live child realm learns the floor its watchers need | same, plus `InterShardFlow::WindowRelay` | a window is shared by every session behind the gateway, so the byte is a FLOOR the gateway then filters per session; a child realm is watched through the parent relay, never through its own window |
| 7 | `ClientControlMsg::WorldAction { seq, action }` with `WorldAction::BlockEdit` and a reserved `Fire`, and `GatewayToShard::SessionAction` | `channels.rs`, `session_flow.rs` | the one reliable client action carrier, built once for two consumers (D-39.1) |
| 8 | schema ids 32 (`chunk_delta`) and 33 (`chunk_pyramid`) and the family prefix bytes in the realm store; the `last_owner_fence` row | `crates/sim/src/stub/built_store.rs` | 30 and 31 are taken; the label refuses a wrong world but not a second writer |
| 9 | `BlockStoreTuning` with every operational number named (checkpoint interval, per-realm byte budget sized to the scattered row, per-session rate cap, `max_reach_m`, `pyramid_storage_block_cells`, the parked-tick bound of §2.2) | the one tuning struct home | no magic numbers |
| 10 | Re-sync the two stale arm counts to the real number | `intershard.rs:9`, `lib.rs:11` | HR1's "one reviewed file" depends on its prose being true |

**The arm-count ceiling.** The decision board says 27 arms + 3 reserved names = 30 = the review
ceiling, with no headroom (`docs/investigation/decision_board.md:496-515, 1170-1173`). The code has
44. The "ceiling" was a design-doc number, never a law; the law is HR1's one reviewed file and its four
conformance tests, which have held through seventeen more arms. This report asks for TWO arms (the
forward and its ack). It asks for NO fetch-delta or fetch-pyramid verb between shards.

**"No chunk data crosses a realm boundary" — an OPEN question, not a settled claim.** Revision 1 stated
it as settled. One case refuses it. A hull is a realm made of blocks (V2.3). A landed hull is the moon's
built child, and the moon integrates its children's collisions (`integrates_children`,
`crates/sim/src/capability.rs:85-91`). To resolve the contact between the hull's belly and the moon's
rock the moon needs the hull's SHAPE. The look bag carries an outline only, and the ONE-RADIUS LAW forbids
more: *"a bound is a promise about space; a look is a statement about appearance … no surface, no detail,
no mesh, no second number"* (`crates/core/src/look.rs:37-42`). So either the hull's collider surface
crosses to the moon, which IS chunk data crossing a realm boundary, or the contact resolves against one
radius, which draws a hull that sinks into a hill or floats over it. **This is now an SL6 ask (A-6) and
an owner decision (D-12), not a footnote.**

---

## 6. UNMEASURED items and their benches

| # | Item | Bench | Number that must hold |
|---|---|---|---|
| M-1 | Diff-lane bytes/s and messages/tick for a busy quarry | a sim-tier fixture in the shape of `crates/sim/src/stub/tests/interest.rs`: 100 builders at 2 edits/s in one chunk cluster, 128 observers at mixed distances | messages/tick ≤ dirty chunks × holders; bytes at rung L ≈ n / 8^L of tier 0; never observers × edits |
| M-2 | Catch-up size and time for a megastructure entering interest | a synthetic realm store with 2,400 stamped buildings (the storage doc's city); a client flies in at 528 m/s on the real lane | first draw < one beat; the 20 Hz datagram p99 stays under budget while the bulk lane carries the city (the SPIKE-3a pattern) |
| M-3 | Store size for 100 M scattered edits | write 100 M marks 57 m apart, run the checkpoint | the byte budget default is sized against the measured number, not 1.1 entries/edit |
| M-4 | Write amplification of the pyramid in a built region | redb with real record sizes, one entry changed per block | decides `pyramid_storage_block_cells` |
| M-5 | Walk-up cost per edit with the effective-summary exit | bench over 13 rungs, every sibling missing, against the coarse generator M-16 defines | the base claims 5.85 µs at depth 13. That number exists only if the coarse summary costs a handful of evaluations, which is exactly what M-16 must prove |
| M-6 | Pyramid tail replay after `kill -9` | the process-tier crash proof pattern of D-6 | replay ≤ one checkpoint interval of edits; byte-identical to `verify-pyramid` |
| M-7 | `verify-pyramid` on a 100 M-edit realm | the same store as M-3 | the base's 54 s is an estimate; the test suite must run it per scenario, so it needs a ranged form |
| M-8 | No drift at coarse rungs | the SL10 gate extended: `generate_summary` for every legal tier on x86-64 and aarch64, byte for byte | zero differing bytes |
| M-9 | Click-to-pixel with durable-before-ship | a client edits; measure from the action's send to the drawn cell | the base's 73–123 ms band, plus one tick |
| M-10 | The realm store under an edit storm, and under a stalled disk | 200 edits/s for a minute on the dev cluster's disk; then repeat with an injected fsync stall | `backpressure_stalls == 0` on a healthy disk (`crates/io-prod/src/store.rs:289-295`); under the stall, poses keep shipping, only the edit lane parks, the parked-tick counter rises, and the refusal is typed (§2.2) |
| M-11 | Chunk interest cost at scale (SL9), including the NEW `(chunk, rung)` → sessions index | 128 observers, 10,000 dirty chunks in one tick | cost grows with dirty chunks × holders, never with observers × chunks; the index's rebuild cost per tick is stated |
| M-12 | Entries per felled tree, and per mined cell under a standing tree (§1.5, §1.6 case 3) | fell 19,406 trees in one disc; then mine under 1,000 standing trees | re-derive the clear-cut cost the storage doc estimated at 18.4 MB/km²; no tree is left hanging |
| **M-13** | **The regime hand-over (SL8, V2.9)** | a pilot flies a hull to a moon that holds a city, lands, and walks out; record the drawn rung per chunk per frame across the re-home | no chunk's drawn rung goes BACKWARD; no chunk is undrawn for one frame; the client's `(realm, chunk)` cache survives the re-home (zero re-fetches) |
| **M-14** | **G-IDENTICAL (HR4)** | the identical edit fixture — break a cell, walk up, checkpoint, `kill -9`, replay, ship the rows — on a MOON realm and on a SHIP realm | both stores and both row sets compare byte for byte |
| **M-15** | **The fly-away and fly-back gate (V2.9, SL8)** | fly a hull from the ground to orbit and back over a dug tunnel, a filled quarry and a built tower; sample the drawn shape at every rung boundary | each edit stays visible up to its stated quantisation rung (§1.4); the drawn step at every boundary stays under the drawable floor (1 px at 45° over 720 rows, owner 2026-09-06); the rung where each shape stops being representable is RECORDED, never discovered in play |
| **M-16** | **The coarse generator equals the fold** | 10,000 random coarse addresses at every legal rung: compare `generate_summary(a)` with the fold of its children; time both | zero differing entries; the coarse call's cost is stated and is not 8^L evaluations |
| **M-17** | **One window, many sessions** | one close pilot and 127 distant ones behind ONE gateway, all watching one moon with a city | the moon's egress at the floor; the gateway's per-session drop rate; the 20 Hz datagram p99 unmoved |
| **M-18** | **Manifest size, the scattered case** | a moon with 10,000 players' marks 57 m apart; an observer in orbit and an observer on the ground; sweep every rung | the manifest bytes per rung, and the rung where the count peaks |
| **M-19** | **The forward's idempotency state** | 200 edits/s for one hour from one hull into one moon | the owner's idempotency state is ONE row per origin realm, never one per edit |

---

## 7. SL6 — what new data must cross, and what doing without costs

Nothing here is added. Each line is an ask.

| # | Data | From → to | Why the receiver cannot compute it | Cost of doing without |
|---|---|---|---|---|
| A-1 | `InterShardFlow::BlockEdit(BlockEditForward)`: the target points **in the ORIGIN's own frame**, the verbs, the origin realm coord + fence, the session id, the origin's edit sequence | a child realm's shard → its PARENT's shard (one hop, the parent only; routed by `ParentRealmNode`) | the parent holds the store and the child's placement; the child holds the input; HR1 forbids the child writing the parent's file; SL1 forbids the child stating a parent-frame address | no edits from inside a landed hull; a pilot must step outside to dig |
| A-2 | `InterShardFlow::BlockEditAck { origin, seq, outcome }` | the parent → the child's shard | the child must tell the client the outcome and stop retrying | silent loss or endless retry |
| A-3 | `ShardToGateway::BulkFor` and `BulkMsg::ChunkRows` / `ChunkManifest` on `MsgClass::Bulk` | a realm → the gateway → the named sessions (not a realm boundary: the connection plane is not a realm, SL2 clarification 2026-08-24) | the client derives the shape but cannot derive an edit | no edit is ever visible to anybody |
| A-4 | `rung_floor: u8` on `GatewayToShard::WindowOpen` AND on the relay's downward request | the gateway → the watched realm's parent → the watched realm | the realm never learns where an outside looker is (SL2); the gateway alone composes the chain and knows every session's angular size | the realm ships its FINEST rung to every window: a rung-0 city to an observer in orbit |
| A-5 | `ClientControlMsg::WorldAction` and `GatewayToShard::SessionAction` | the client → the gateway → the session's authority shard | a block edit must never be lost; `action_bits` is latest-wins | no reliable edits; a dropped datagram is a vanished placement |
| **A-6** | **A landed child's COLLIDER SURFACE (its blocks, at a rung)** | a hull's shard → the moon's shard | the moon integrates its children's collisions but the one-radius law gives it a radius only (`crates/core/src/look.rs:37-42`) | the contact resolves against one radius: a hull sinks into a hill or floats over it. **The owner must choose — see D-12. Default NO.** |
| A-7 | An exact coarse summary at every legal rung, in the GENERATOR crate (a code ask, not a wire ask) | the generator domain (report 03) → this domain | the prune rule of §1.3 compares against it; an octave-dropped answer makes the store dense | the pyramid is not sparse: an entry per coarse cell of every touched chunk, at every rung |
| A-8 | The sub-cell geometry of a sub-metre placement, inside the BLOCK RECORD (a format ask, not a wire ask) | the block-record domain (report 04) → this domain | the fold needs the cell's resulting total volume, and it must not move when the generator changes | a player's quarter-metre block moves when the slope under it is regenerated |
| A-9 | `last_owner_fence` row and the two family schema ids in the realm store | on disk only | — | a second writer is detected by nothing but a file lock, which the cloud volume cannot promise |

Not asked, deliberately: any realm-to-realm chunk FETCH verb; any occupant pose on the forward; any
client-declared chunk residency. A `tier_floor` memory hint from the client stays a reserved idea until
M-2 says the catch-up sizes need it (decision D-6).

---

## 8. Where this domain touches others

| Question | Whose | What this domain needs |
|---|---|---|
| The chunk edge and the `CellIndex` linearisation | 01 grid family | the widths in §1.1 |
| An exact, cheap coarse summary per rung | 03 generator | A-7 — without it §1.3 collapses |
| The sub-cell geometry of a sub-metre placement | 04 block record | A-8 |
| The live attachment lane (a HUD, a joint angle) | 05 attachments + the signal domain | case 4 of §1.6, decision D-11 |
| Whether octants and fill draw a coarse quarry well enough | 08 render | D-1 and M-15 |
| Whether a landed hull's contact needs the hull's surface | 09 physics + the crossing domain | A-6 and D-12 |
| `RealmId::System(seed)` aliasing across galaxies | identity | the data-loss door in §10 |

---

## 9. Stale claims in the investigation base

| Claim | Where | What supersedes it |
|---|---|---|
| "27 arms + 3 reserved = 30, the review ceiling; no headroom left; everything further rides inside an arm" | `decision_board.md:496-515, 1170-1173` | the code has 44 arms (`intershard.rs:124-604`); the ceiling was never a law; HR1's gate is the reviewed file plus the four conformance tests |
| "the closed set is 16 arms" / "24 arms" | `intershard.rs:9`, `wire/src/lib.rs:11` | 44, of which eight arms and two `GhostFlow` variants are tombstones; the prose is owed a re-sync (plant 10) |
| "the file name is an immutable `RealmUid`; `RealmKindTag` has six arms and no `Ship`" | `block_system_design.md:4383-4415` | the realm store exists and is named by `RealmId`, which is birth-immutable for a ship (`Ship(EntityId)`, `pose.rs:40`; `bins/src/lib.rs:2310-2333`); `RealmKindTag::Ship = 7` and `Star = 6` exist (`realm_path.rs:44-62`). ⚠ `RealmId::System(seed)` collides across galaxies (`realm_coord.rs:1-8`), so ONE file serves TWO star systems — a data-loss door, §10 |
| "a realm has no store; P6 creates one" | the base's §3.9 as a whole | `StoreRole::RealmStore` with berths and body since D-MOVE-2 (`store_stamp.rs:64-75`, `built_store.rs`); the block families join THAT file, and a deployment must set `VD_REALM_STORE_DIR` (`bins/src/lib.rs:2367-2378`) |
| "the origin shard ships the server's pose for the originating session so the owner re-runs the reach test" | `block_system_design.md:4671-4675` | refused by SL2; the owner tests against the origin realm's placement and extent (§4.2) |
| "the pyramid entry is PERSISTED, PERMANENT, a ONE-WAY DOOR (row 37 / door 43)" | `block_system_design.md:1494-1497`, `decision_board.md` door table | it is derived and rebuildable from the tier-0 store; a soft door on disk, a PROTO_MINOR door on the wire (§1.7) |
| "the pyramid walk-up rides the WAL's own write transaction" | `block_system_design.md:1589-1612`, `4899-4915` | the walk-up runs in memory at commit; the persisted pyramid rides the checkpoint under its own watermark, with a bounded tail replay (§1.5; the storage doc's D-11(b) plus the watermark) |
| "the walk-up exits at rung 1 or 2 for an edit in solid rock" | `block_system_design.md:1617-1619` | false as written: the exit compared against the stored entry, which is `None` for an unedited parent (`storage_and_streaming.md:833-884`); fixed by exiting on the effective summary |
| "the pyramid uses the same 4,256-cell crossover" | `block_system_design.md:1500`, `4780` | the crossover is computed per record, between 3,725 and 13,250 with the palette (`storage_and_streaming.md:1010-1033`) |
| "the durable outbox is the owed refinement" | `crates/io-prod/src/store.rs:28, 60` (doc comments) | the outbox is on for every shard since 2026-09-06 (`DEFERRED.md:916-950`); the forward rides it |
| "D-4 owes the `MsgClass` routing arm for `EventMsg` and `BulkMsg`" | `block_system_design.md:4957-4975` | `EventMsg` landed INSIDE `ServerControlMsg::Event` on the Control lane with no new class (`DEFERRED.md:838-845`); only `Bulk` is owed |
| "the client has the generator, so only edits travel" (stated as fact) | `addendum_2.md` §B fact one | it was an assumption against the 2026-08-27 ruling S6; it is now LAWFUL for the static shape under SL10 (2026-09-07), and only for the static shape |
| "detail tiers become more tags in the same look bag" | `window_lane.md:587-597` (recorded Q&A, not a ruling) | a look bag is one digest-diffed blob per realm; chunk diffs are per address and must be their own rows (§3.2). The tier number on the window survives, as a FLOOR (plant 6) |
| "the gateway selects a tier per observer with a plain number on the window" | `window_lane.md:587-597`, and revision 1 of this report | a window is shared by every session behind that gateway (`window_lane.rs:287-296`), and a live child realm is watched through the PARENT RELAY, never through its own window (`session_flow.rs:575-581`). The number is a FLOOR plus a per-session filter (§3.3) |
| "`tier_floor` is a client-declared memory hint the server clamps for egress" | `block_system_design.md:5017-5060`, O13 | inside a realm the shard derives rungs from the pose it holds; outside, the gateway states the floor. A client hint is decision D-6, never a plant |
| "the ladder adds zero inter-shard bytes" | `block_system_design.md:4690-4696` | still OPEN, not strengthened: a landed hull's contact may need the hull's surface (A-6, D-12). Revision 1 claimed "no chunk data crosses a realm boundary, ever" as settled, and that was premature |
| "`BlockEdit` carries `{place, break, fetch-delta, fetch-pyramid}`" | `decision_board.md:504-507` (O15) | no fetch verbs; a shard never reads another realm's chunks for RENDER. Whether it needs a neighbour's surface for CONTACT is D-12 |
| "the forward is idempotent on `(TransferId, step 20)` in the applied-steps journal" | revision 1 of this report, following the base's transfer pattern | `TransferId` is minted by the orchestrator at saga creation (`ids.rs:94-98`), and `AppliedSteps` owes a retention bound it can only get from a terminal (`crossing_receive.rs:48-53`). An edit stream has no terminal. §4.3 uses the origin's own sequence and a low-water mark |

---

## 10. One-way doors (this domain)

| Door | Deadline | Cost if wrong |
|---|---|---|
| The `CellIndex` width and linearisation, the chunk edge, the substance id width (shared with the block record) | before the first world is saved | a migration over every saved planet, station and hull |
| The fold rule and the exact coarse `generate_summary` in the generator crate (A-7) | before the first world is saved | a protocol event and a world-generation tag change; every store's pyramid rebuilt; and, if the coarse call is not the fold, no store is ever sparse |
| **`RealmId::System(seed)` aliases across galaxies, and the store file is named from it** (`realm_coord.rs:1-8`, `bins/src/lib.rs:2310-2333`) | **before the first world is saved** | **two star systems share one `realm-system-<seed>.redb` and overwrite each other's buildings. DATA LOSS, and silent** |
| The family PREFIX BYTES in the realm store | before the first block is written | a wrong prefix byte silently changes which rows one `scan` returns. The SCHEMA id is NOT part of this door: `TlvReader::parse` refuses a foreign schema before it decodes a field (`crates/core/src/tlv.rs:187-192`), so a wrong schema id is caught, not silent |
| `pyramid_storage_block_cells` | before the first pyramid block is written | a persisted-key migration |
| `BLOCK_STORE_FLUSH_STEP = 19`, `BLOCK_EDIT_FORWARD_STEP = 20` | the first wire slice | a step collision is silent, durable and unrecoverable |
| The `BlockEditForward` field order (hull-frame points, never a parent chunk key) and the `ChunkRow` bag schema | when the arm ships | postcard is positional; a re-order is a flag day |
| A sticky "differs from the seed" bit against quantisation (D-10) | before the first world is saved | it changes the 8-byte layout's meaning and the prune rule together |
| The 8-byte pyramid layout itself | soft | a rebuild pass per realm plus a PROTO_MINOR bump |

---

## 11. Open decisions for the owner

| # | Question | Options | Recommended | Why |
|---|---|---|---|---|
| D-1 | Does a coarse entry carry a surface-height datum for smooth terrain, beyond octants and fill? | (a) no: 8 bytes, 14 reserved bits; (b) yes: spend 8 reserved bits on a signed height delta per coarse cell | (a) now, decided with the renderer report before the first world is saved | the entry is rebuildable, so the disk cost of a later change is a rebuild; the wire cost is a minor bump |
| D-2 | Where does the pyramid persist? | (a) inside the WAL's transaction (base); (b) at the checkpoint under a watermark, tail replayed on open | (b) | it is reconstructible; (a) pays 5,508× write amplification per entry for a crash consistency the design does not need |
| D-3 | The owner fence on the realm store | (a) a file lock; (b) `last_owner_fence` compared on open, refuse when not strictly greater | (b) | the cloud volume is network-backed; advisory locks are not reliable there |
| D-4 | The per-realm byte budget's sizing and shape | (a) 1.1 entries/edit, realm-wide refusal; (b) sized to the scattered row (~6 entries/edit), hierarchical per claim | (b) | a realm-wide refusal lets one logging operation stop everybody's editing on a moon |
| D-5 | The diff lane's carrier | (a) appended variants on the reliable Control lane; (b) a paced `Bulk` class on its own stream | (b) | a 23 MB city must not queue in front of the transfer cut marker |
| D-6 | A client memory hint (`tier_floor`) | (a) plant one byte now; (b) wait for M-2 | (b) | the shard and the gateway already decide rungs from lawful knowledge; a hint is a second decider |
| D-7 | Diffs ship durable-first or fast-first | (a) ship at tick T+1 after the batch is durable; (b) ship at tick T, accept a crash window of one batch | (a) | a client never draws a wall the restarted shard lacks; the cost is one tick |
| D-8 | Encoding of "a cell returned to the seed" on the wire | (a) a revert list tag; (b) a flag bit in the cell record | (a) | keeps the cell record identical to the stored one; the tag is skip-unknown |
| **D-9** | **Does a valuable-deposit diff ship the moment a chunk enters a client's interest?** | (a) yes, like every other diff: a changed client reads the ore before anybody surveys; (b) only after the realm records that this ACCOUNT surveyed that chunk, which needs a per-account gate on the lane and a per-account manifest | **the owner decides; this report recommends nothing** | S5.3 of 2026-08-27: *"the survey is the mechanic. What is in the ground is revealed by looking."* A deposit is live state (SL10 clause 6), so it is a diff, so the LETTER of the seed law holds — but the lane hands the mechanic away. *Example: a prospector lands on a moon; the shard ships the tier-0 diffs around the boots; a changed client reads the ore 60 m down and drills straight to it.* Option (b) makes the diff lane per-account for one tag, which breaks the "one row, many recipients" coalescing of §3.5 for that tag alone |
| **D-10** | **A sticky "differs from the seed" bit, against quantisation (§1.4)** | (a) no: an edit under half a unit of a coarse cell disappears at that rung; (b) yes: one reserved bit says "something below me differs", so a coarse cell is never pruned over an edited subtree, at the cost of an entry per edited column at every rung | (a) until M-15 measures the drawn step | if M-15 shows the step stays under the drawable floor, (a) costs nothing and (b) costs the sparsity. If M-15 shows a visible pop, (b) is the cure, and it must be decided before the first world is saved |
| **D-11** | **Which lane carries a LIVE attachment value (a HUD reading, a turret joint angle)?** | (a) the paced chunk-diff lane: it queues behind a 23 MB city; (b) the realm's own per-tick statement lane, the signal bus (P9) | (b) | a joint angle is live state at the tick rate, not a diff of the world's shape. Recorded here so the attachments domain and the signal domain do not each assume the other carries it |
| **D-12** | **Does a landed hull's collider surface cross to its parent (A-6)?** | (a) no: the contact resolves against one radius, and a hull visibly sinks into a hill or floats over it; (b) yes: the hull ships a coarse collider surface up, one hop, on change — chunk data crossing a realm boundary, which SL6 defaults to NO; (c) the parent computes nothing and the HULL resolves its own contact against the parent's terrain, which the hull can generate from the seed but needs the parent's nearby EDITS for | **the owner decides; (c) deserves the look** | (c) keeps the one-hop rule and needs only the parent's edited chunks near the berth, which is the same data the pilot's client already receives. It moves the question from "may data cross" to "who computes the contact", and 09 physics must answer whether a child may integrate its own contact with its parent |

---

## 12. Revision log

Two refuters tested revision 1: `verdicts/storage_law.md` and `verdicts/storage_feasibility.md`. Both
returned REFUTED. Every finding is answered below.

### The law refuter

| # | Finding | What I did |
|---|---|---|
| F-1 | BREAKS SL1 — the hull states a moon-frame chunk key | **ACCEPTED and fixed.** §4.2 step 2 now states a HULL-FRAME point, and step 4 makes the moon add the placement it authored. A-1's payload changed. The reasoning is written into §4.2 under "Why step 2 changed". The fix is also simpler, as the refuter said |
| F-2 | WRONG — `coordinator_of` does not exist, and the target is always the parent | **ACCEPTED and fixed.** The directory step is deleted. §4.1 states the consequence, and §4.2 step 3 routes by `ParentRealmNode` (`crates/sim/src/stub/realm_head.rs:75-82`), with the reach emitter as the template (`reach.rs:36-48`). The target-realm field is gone from A-1 |
| F-3 | WRONG citation — `window.rs:225-233` is the window pruner | **ACCEPTED and fixed, with a better citation than the one offered.** `session_flow.rs:647` is the WINDOW lane's predicate. A forward from a child is an UP-lane message, and the up-lane's predicate is the three-guard consumer at `crates/sim/src/stub/reach.rs:78-101`: misroute, attestation against `ChildRealmNodes`, staleness. §4.2 step 5 now cites it |
| F-4 | UNMEASURED AS FACT — the regime hand-over | **ACCEPTED and fixed.** The "no jump" sentence is deleted; the example now says the hand-over is UNMEASURED. Bench M-13 is added. §3.3 now states that the client KEEPS its `(realm, chunk)` cache across the re-home, and why |
| F-5 | MISSING — no G-IDENTICAL gate (HR4) | **ACCEPTED and fixed.** §4.4 names HR4 and the gate; bench M-14 runs the identical fixture on a moon realm and a ship realm, byte for byte |
| F-6 | MISSING — the survey mechanic | **ACCEPTED and fixed.** Decision D-9 is added, with the prospector example, both options and their costs, and no recommendation. This report does not decide it |
| F-7 | MISSING — a tick hitch with no continuity (SL8) | **ACCEPTED and fixed.** §2.2 carries "What the moon does when the disk is slow": poses keep shipping, only the edit lane parks, a counter says so, and past a named bound the edits are refused with a typed reason. M-10 is extended to drive an injected stall |
| F-8 | MISSING — what the client draws while a manifest is outstanding | **ACCEPTED and fixed.** §3.4 point 3 states the rule: draw the coarsest rung you hold; never draw a chunk finer than the rung whose diff you hold. Measured in M-13 and M-15 |
| F-9 | WRONG — the tombstone count | **ACCEPTED and fixed.** Eight top-level tombstone arms, listed with line numbers, plus two tombstoned `GhostFlow` variants inside the `Ghost` arm (`intershard.rs:993-1001`). Revision 1 said four and listed five |

### The feasibility refuter

| # | Finding | What I did |
|---|---|---|
| F-1 | ONE TIER PER WINDOW IS IMPOSSIBLE | **ACCEPTED, fixed, and EXTENDED.** §3.3 is rewritten: a window is per `(shard, scope)` and shared by every session behind the gateway (`window_lane.rs:287-296`). The rung is a FLOOR — the finest any session behind that window needs — plus the gateway's per-session filter, so the moon's egress is sized by the CLOSEST pilot behind each gateway. That cost is now stated, and so is the alternative's (a window per session pays the static roster per session). **I found a third fact neither the report nor the refuter had:** a live child realm is not windowed from outside at all — there is deliberately no `Observed` scope, and the child's rows ride the PARENT RELAY sealed (`session_flow.rs:575-581`, `344-383`). So the floor must ALSO ride the relay's downward request. Plant 6 and A-4 both changed |
| F-2 | THE COARSE GENERATOR IS THE WHOLE MECHANISM AND IS NEVER DEFINED | **ACCEPTED and fixed.** §1.3 now carries the requirement in full: one function `summary(field, address)`; the generator's coarse answer must be the exact fold of its own fine cells; the octave-dropped answer is not; and the naive exact answer is 8^13 evaluations, about 1.9 hours per rung-13 cell (ESTIMATED at the MEASURED 12.19 ns). It is now ask A-7 to the generator domain, bench M-16, and a door in §10. I also wrote out the alternative prune rule ("prune on no authored cell below") and why it is rejected: it draws a straight cliff along the cell boundary, which is SL8's detail-by-box seam. M-5's note now says the base's 5.85 µs exists only if M-16 passes |
| F-3 | THE CATCH-UP MANIFEST GROWS WITH THE MOON | **ACCEPTED in substance, DISPUTED in arithmetic, and fixed.** §3.4 point 2 makes the manifest HIERARCHICAL (one digest per coarse chunk covering its column), and bench M-18 sweeps the scattered case at every rung. **DISPUTED: the 7 MB rung-7 figure assumes essentially every rung-7 chunk carries a diff, and the prune rule refuses that.** A rung-7 chunk is about 7,936 m on a side; one 1 m mark is about 1 m³ against 5.0e11 m³, far below the half-unit threshold E³/512 of §1.4, so the entry prunes and the rung-7 manifest of a scattered moon is nearly empty. The worst case is the MIDDLE rungs, where an edit is just large enough to survive the prune, and M-18 must find it. The finding is taken because the manifest was never sized and never benched — only the number moves |
| F-4 | THE SEAMLESS ANSWER POINTS AT A GATE THAT IS NOT IN §6 | **ACCEPTED and fixed, with corrected arithmetic.** M-15 is written: fly a hull out and back over a tunnel, a quarry and a tower, sample at every rung boundary, record the rung where each shape stops being representable, and hold the drawn step under the owner's drawable floor. §1.4 is a new section that derives the general bound: an edit of volume V vanishes at the first coarse edge E with E³ > 512 V. Decision D-10 puts the sticky "differs from the seed" bit to the owner. **DISPUTED: the refuter's 20 m quarry does not vanish at the 128 m cell. Folded rung by rung it reads 247 at 64 m, 254 at 128 m and 255 at 256 m, so it vanishes one rung later than stated.** The finding stands; only the rung moves |
| F-5 | THE §1.2 EXAMPLE MISREADS ITS OWN RECORD | **ACCEPTED and fixed.** 20³ / 128³ = 0.38 %, so `fill_256` is 1, not 5, and a 20 m tower sits in ONE 64 m octant, not four. The example now carries the arithmetic and points at §1.4, because one unit of 256 is the last unit before the tower disappears |
| F-6 | THE IDEMPOTENCY JOURNAL HAS NO RETENTION BOUND AND NO MINTER | **ACCEPTED and fixed.** §4.3 is new. `TransferId` is minted by the orchestrator at saga creation (`ids.rs:94-98`), and an edit stream has no saga; `AppliedSteps` owes a retention bound it can only get from a terminal (`crossing_receive.rs:48-53`). The forward now carries the ORIGIN REALM's own monotone edit sequence, minted by the origin shard and carried inside the retained frame; the owner keeps ONE low-water mark per origin realm, bounded by the child count and dropped with the child. Bench M-19 |
| F-7 | THE SL9 ARGUMENT REVERSES THE CODE IT CITES | **ACCEPTED and fixed.** §3.3 now says plainly that `plan_interest` is OBSERVER-MAJOR (`interest.rs:129-160`), that the `(chunk, rung)` → sessions index does NOT exist, that the chunk lane must build one, who maintains it, and when. I kept one nuance the refuter did not name: the OUTPUT `InterestPlan.recipients` is already cube-major (`interest.rs:120-123`), so it is the shape to copy — but its keys are occupant cubes, not chunk keys, and building it costs observers × slab per tick. M-11 now benches the new index's cost too |
| F-8 | "NO CHUNK DATA CROSSES A REALM BOUNDARY, EVER" IS NOT ESTABLISHED | **ACCEPTED and fixed.** The claim is withdrawn from §5.2 and from §9. The landed-hull contact is now SL6 ask A-6 and owner decision D-12, with the one-radius law quoted (`look.rs:37-42`) and `integrates_children` cited (`capability.rs:85-91`). I added a third option the refuter did not list: the CHILD resolves its own contact against the parent's terrain, which it can generate from the seed and needs only the parent's nearby EDITS for — the same rows the pilot's client already gets. That keeps the one-hop rule and moves the question to 09 physics |
| F-9 | FIVE CASES THE DOMAIN NEVER NAMES | **ACCEPTED and fixed.** §1.6 is new and answers all five: a mined cell under a placed block; a sub-metre block on a slope (with ask A-8 to the block-record domain); a tree standing on a mined edge (the span rule read downward, M-12 extended); a live attachment value (decision D-11: the signal bus, never the paced bulk lane); and an edit while the AUTHOR is crossing (the ack is addressed to the origin realm and its fence, never to a session) |
| F-10 | TWO DOORS ARE MIS-STATED | **ACCEPTED and fixed.** The `RealmId::System(seed)` file collision is now a DATA-LOSS door in §10 with the same deadline as the rest, stated in §2.2 with the two-galaxy example and cited (`realm_coord.rs:1-8`, `bins/src/lib.rs:2310-2333`, `realm_head.rs:92-94`). The prefix-byte door is narrowed to the PREFIX BYTE alone: `TlvReader::parse` refuses a foreign schema before decoding a field (`crates/core/src/tlv.rs:187-192`), so a wrong schema id is caught |
| §1 qualification | `open_realm_store` returns "no store" unless the environment names a directory | **ACCEPTED.** Stated in §0 item 4 and in §2.1 (`bins/src/lib.rs:2367-2378`) |

### What did not change

The store half stands: the realm store, the schema ids, the block-on-prior `commit`, the one transaction
and one fsync, the outbox, the fold's freedom from float, and the fact that this report proposes no new
library. Both refuters checked those, and both let them stand.
