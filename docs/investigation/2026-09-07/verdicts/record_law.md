# Law verdict — 04 the saved block record, the registry, params, styles, themes

**Date:** 2026-09-07. **Lens:** the law refuter. **Target:**
`docs/investigation/2026-09-07/04_block_record_registry.md`.
**Result: REFUTED.** The report holds two load-bearing errors and shuts two owner doors.

**How to read this file.** Each finding names the claim, the verdict, the evidence and the fix.
MEASURED means a command ran today and the method is named. UNMEASURED means nobody measured it.
Every finding carries an example in the game's own words.

---

## 1. The summary

The report is strong on the bit arithmetic and honest about the code. I re-ran its measurements and
the code table in its §1 is true. Its law table in §8, however, is a set of arguments, and four of
those arguments fail.

- Two claims are WRONG inside the report itself. The registry chapter contradicts its own
  zero-migration promise, and the attachment record's escape hatch does not fit in the bits it names.
- Two owner requirements lose a door. V2.1 (smooth terrain built of voxels) and V2.2 (a tree as one
  object that carries its own height and structure) have no field and no hand-off.
- One law is broken. SL6 says ask before new data crosses and before a wire arm is added. The report
  answers "NONE", then adds a wire arm, a handshake field and a realm salt.
- One owner-pending decision is declared settled by an argument, not by a ruling.

Six laws stand and I say so below, so the owner does not re-open them.

---

## 2. The findings that refute the report

### F1 — The registry's own example breaks the registry's own rule. WRONG.

**The claim.** §4.4 says: *"The team ships ... a 'fluted' variant for every hull plate ... every plate
kind's variant count goes from 3 to 4. Every planet, hull and station store on every shard opens as
before, because the stored prefix still matches. ... Not one saved record moves."*

**Why it fails.** §4.1 puts the variant count inside the digest: *"It ALSO covers, new in this report:
each kind's `variants` count and `slots` count, the `ATTACHMENT` flag, the sub-scale admit mask ...,
and the theme id."* The digest is a hash over the identity columns of rows `0..n`. A titanium hull
plate is an OLD row, low in the numbering, inside every stored prefix. Raising its variant count from
3 to 4 changes that row's bytes. So `identity_prefix_digest(stored len)` changes. §4.2's store-open
rule then refuses the store.

**The consequence in the game's words.** A player builds a station on a moon. The team ships a fluted
style for the hull plate. The station's shard restarts, reads the registry digest in its store header,
and REFUSES to open the store. The moon's store, the player's hull store and every planet store refuse
the same way. The single most common content change — a new style — becomes a fleet-wide outage.

**The same defect hits three more columns.** The sub-scale admit mask and the slot count are bounds on
what a player may PLACE NEXT. They do not change what a saved record MEANS. Putting them in the digest
makes "this plate may now also be placed at quarter scale" a refusal of every saved world.

**Fix.** Split the two jobs.

- The digest covers only what changes a SAVED record's meaning: the kind's triple, the referenced
  form's geometry-defining fields, and the `ATTACHMENT` form class.
- The variant count, the slot count and the sub-scale admit mask leave the digest. The CURRENT build
  enforces them at decode and at placement. A saved variant is always below the old count, so raising
  the count can never invalidate a saved plate. Lowering it stays a refusal, which §4.2 already wants.
- Re-write §4.2's two lists and §4.4's example against the corrected column set.

---

### F2 — A door is shut on V2.1, and the base's own iso-surface door is never re-validated. MISSING.

**The claim.** §0 scopes the report to V2.4, V2.5, V2.6 and V2.7. §8's law table claims the record
serves every law.

**What is missing. MEASURED, this session:** `grep -n "V2\.1\|V2\.2\|V2\.8\|V2\.9\|smooth\|density\|isosurface"`
over the report returns four hits, and all four are about a substance's density in kilograms per cubic
metre or about the word "smooth" as a style name. The report never names V2.1, V2.2, V2.8 or V2.9.

**Why that matters.** V2.1 says: *"I'd like to try to implement the smooth realistic terrain, but still
build with voxels ... and when we place voxels, the terrain should change accordingly."* A smooth
surface that reshapes around a placed cell normally needs a per-cell surface value — an occupancy
fraction or a signed distance. The recommended record has NO such field and FIVE reserved bits. One
byte does not fit in five bits.

**The base already calls this a one-way door and the report walks past it.** `block_system_design.md`
line 5501 lists *"Explicit ids rather than an iso-surface representation | YES in content terms |
Players build against the vocabulary; you cannot take faceted hulls away later"* (MEASURED, `sed -n
'5495,5511p'`). The report's §7 supersedes the neighbouring door on the same page ("one geometry per
cell") and says nothing about this one. §12 hands the grid family the chunk edge, the cell order and
the sub-lattice size, and no surface field.

**The consequence in the game's words.** A player mines a hillside on a moon and expects the slope to
round off, as V2.1 asks. The record can only say "this cell holds granite, rotated, riveted". The
hillside stays faceted, or the grid domain must add a second per-cell record for the surface value —
which is the second machinery HR3 forbids, after the first world is written.

**Fix.** Add an open decision to §10, before the record freezes: does the smooth surface of V2.1 need
a per-cell value in the SAVED record? If the answer can be yes, the record must reserve the width now,
or the report must show, with the grid domain, that the smooth surface is entirely a function of
`(seed, address)` plus the kind, and therefore needs no stored value. Say which, and mark it
UNMEASURED until the grid domain answers.

---

### F3 — V2.2's tree carries parameters the record has nowhere to put. MISSING.

**The claim.** §3.1 rules that the record holds exactly one per-cell parameter: *"style ... the 6-bit
`variant` ... It is the one thing a player CHOOSES per placement that changes nothing physical."*

**Why it fails.** V2.2 says a tree is *"one object/block on the surface ... rendered on the client as a
tree (with different height and structure depends on the seed and height params inside this block),
but then the collisions are calculated on the server, so server somehow should know the shape."* A
tree's height and structure are per-cell parameters, they are NOT style, and the SERVER reads them to
build the collider. The 6-bit variant is already spent on style, and the report forbids a second
per-cell parameter by rule.

**Nuance, stated fairly.** A seed-placed oak needs no record at all under SL10 V1.6. The gap is
narrower than it looks. It bites on a PLANTED tree (R8's growth stage), on a tree a player edits, and
on any other large landscape part V2.2 reserves room for.

**The consequence in the game's words.** A player plants a sapling in a clearing on a moon and comes
back a month later. The moon's shard must know how tall the tree grew, so a walking player cannot walk
through the trunk. The record can say "an oak kind, provenance Placed, variant 2" and nothing about
height. The growth stage then lives in a side table that the report mentions once, for one byte, and
never connects to V2.2.

**Fix.** State where a large-object parameter lives. Either name the side table as V2.2's home and
carry it into §12's hand-off, or reserve a per-cell parameter run in the record. Add the choice to
§10 as a recommended answer, because it is a permanent door.

---

### F4 — The attachment record's escape hatch does not fit its own bits. WRONG.

**The claim.** §2.4: *"Reserve nothing for it; the ten reserved bits hold a sub-site later if needed."*

**Why it fails.** §2.2 defines a sub-site as `sub_scale` (2 bits) plus `sub_addr` (9 bits), which is
ELEVEN bits. The attachment record keeps TEN reserved bits (§2.5: *"The attachment record keeps ten
reserved bits"*). Eleven does not fit in ten.

**The consequence in the game's words.** A pilot chisels a cockpit console into quarter-metre blocks
and wants a shield gauge on the quarter-block's own face. The report's stated escape is to widen the
attachment record later with a sub-site. That escape is not available. Every saved gauge, joint and
rail then moves tables — the exact cost §11 lists for this door.

**Fix.** Say the true escape. Either the attachment record spends one more bit now, or the escape is
"a sub-scale attachment KIND on the host cell's face", which §2.4 already recommends and which needs
no bits at all. Delete the ten-bit claim.

---

### F5 — SL6 is answered "NONE" while the report adds an arm, a handshake field and a salt. BREAKS_LAW.

**The claim.** §8: *"SL6 (new data across a realm boundary) — NONE."* §7's last row: *"This domain adds
ZERO arms."*

**Three counter-facts.**

1. **The arm does not exist.** MEASURED: `grep -rn "BlockEdit" crates/` returns exactly two hits, and
   both are doc comments — `crates/wire/src/intershard.rs:34` and `crates/wire/src/lib.rs:24`, each a
   `//!` line. `BlockEdit` is a NAME, not a variant. §8's HR1 row admits an edit rides it. Giving a
   reserved name a shape IS adding an arm, and SL6's second clause says: ask before adding a wire arm,
   default NO.
2. **The client handshake gains a field.** D-5 recommends the registry digest at the client handshake,
   and §12 hands it to the wire domain. Today `ProtoVersion` carries `coordinate_generation` only
   (`crates/wire/src/version.rs:350`, MEASURED). A new field on the handshake is new data.
3. **The realm salt has no lane.** §3.3's detail function reads `splitmix64(realm_salt, cell)`. The
   report never says where a client gets a realm's salt. It is a number the moon's shard holds and the
   client must receive.

**The consequence in the game's words.** A pilot edits a hull plate inside her ship's realm. The edit
must reach the moon's shard, or the station's, if a ghost region straddles the seam. §8's HR1 row says
the wire domain "must confirm" whether that forward survives containment re-home. That confirmation is
exactly the SL6 ask, and it is written as a footnote instead of a request.

**Fix.** Replace §8's SL6 row with an explicit request, in SL6's own shape, for three items: the
`BlockEdit` arm and its payload; the registry digest field on the client handshake; the realm salt to
the client. For each, state what data, from which realm to which, why the receiver cannot compute it,
and what doing without costs. Default is NO until the owner answers.

---

### F6 — The client-derived style detail is an owner-pending decision, presented as settled. UNMEASURED_AS_FACT.

**The claim.** §3.3: *"Which law covers it. This is rendering, not derivation of state: the client
'only renders' and a detail sprite is not a placement, a velocity or an entity state (SL10 V1.7)."*
§7 marks the base's *"[USER DECISION 6-1] awaiting the owner"* as settled: *"SL10 settles the principle
for the STATIC SHAPE (V1.1–V1.3); style detail is RENDERING of received diffs (V1.7)."*

**Why it fails.** V1.7 grants nothing. It RESTRICTS: *"This is NOT a relaxation ... The exception is
exactly the static shape of the seed, because that shape is not state."* V1.1 defines the static shape
as a function of `(seed, address)`. The detail function's inputs are a player's placed records, a
26-neighbour mask that changes whenever a player welds a plate, and a realm salt. None of those is
`(seed, address)`. The report converts a pending owner decision into a settled one by an argument, and
the standing rule is: never assume, measure, or say UNMEASURED.

**SL3 pulls the same way.** SL3 says the realm authors HOW IT LOOKS. In §3.3 a shared function on the
client decides where the rivets sit. The report should show that the realm still authors the look
through its records and the shared registry, rather than assert it.

**The consequence in the game's words.** Two players stand at a station and look at one hull. If the
owner later rules that the server authors the trim, every client's meshing path changes after the
record has frozen. The record itself survives — the variant is still six bits — so the cost is
contained. That is why this is a decision to raise, not a defect to fix.

**Fix.** Add a D-entry to §10: "Does the client derive connected-block trim from the records it holds,
or does the realm author it?" Recommend the client, give the reason, and mark the answer as the
owner's. Remove the claim that SL10 settled it.

---

### F7 — The storage ledger keeps a budget its own row disproves. WRONG.

**The claim.** §5's table keeps *"Delta per heavily-played planet ... < 1.5 GB"* and *"~880 MB ⇒
< 1.7 GB with the pyramid"*, and then adds the row *"Planetary scatter ... 6.07 entries per edit at
100 M edits | 5.5× the 1.1 plan; size the budget against 4"*.

**The arithmetic. MEASURED by hand today.** At 4 pyramid entries per edit, 100 million edits cost
100e6 × 4 × 8 B = 3.2 GB of pyramid, plus 800 MB of records: about 4.0 GB. At the row's own 6.07
entries the figure is about 5.7 GB. Both are far above the 1.5 GB and 1.7 GB the same table asserts.
The base's scatter figure is real: `storage_and_streaming.md:885-895` derives it (MEASURED, read
today).

**The consequence in the game's words.** §5 then tells `BlockStoreTuning` to refuse a guild's next edit
when the realm's budget runs out. A budget set from the wrong number refuses a guild's spaceport on a
moon two and a half times too early, or never refuses at all.

**Fix.** Restate the per-realm delta budget once, against the scatter row, and delete the two
superseded totals. Mark the new figure ESTIMATED and name the arithmetic.

---

### F8 — The record key cannot enforce the non-overlap the report promises. MISSING.

**The claim.** §2.2: *"The key for dedup-to-final is `(chunk, cell, sub_scale, sub_addr)`. ... A cell
holds either one whole block or a set of sub-blocks that tile without overlap."*

**Why it fails.** A half-scale steel cube at `sub_addr` 0 and a quarter-scale steel nub at `sub_addr` 0
have DIFFERENT keys and OVERLAP in space. The key separates them, so the store accepts both. §2.6's
decode rule lists nine refusals and none of them is an overlap.

**The consequence in the game's words.** A pilot chisels a cockpit console. A corrupt store, or a
placement defect, leaves a half-scale cube and a quarter-scale nub in the same corner. The store
opens. The hull's shard then builds a collider from two solids in one place. A second player walks
into the console and the sweep test disagrees with what she sees. That is a seam under SL8, produced
by a format that promised it could not happen.

**Fix.** Say where non-overlap is enforced. It cannot be the key. Either the placement path owns it and
the store trusts its own writer, or the store-open path validates a chiselled cell's set. Name the
choice, and add the refusal to §2.6 or the check to §9's measurements.

---

### F9 — No measurement proves the feature runs on two shard kinds. MISSING.

**The claim.** §9 lists M1 to M10 and says *"Every claim above that could be a measurement is listed
here with its method."*

**Why it fails.** HR4 says a feature passes the identical fixture on at least two shard kinds
(G-IDENTICAL) or it does not land. No measurement in §9 runs one record fixture on a moon's shard and
on a hull's shard. §1 correctly notes that `ShardProfile` already carries `voxel`, `block_edit`,
`surfaces`, `seats` and `functional_blocks` (`crates/sim/src/capability.rs:106-117`, MEASURED), which
is what makes the two-kind run possible. Nobody owes the run.

**Fix.** Add M11: the same forty-plate fixture places, breaks, chisels and attaches on a
planet-profile shard and on a hull-profile shard, and the two stores compare byte for byte.

---

### F10 — The record lane is never measured, only the record store. MISSING.

**The claim.** §3.3: *"Both clients hold the same forty plate records."* §5 measures the ledger.

**Why it fails.** The trim function needs the 26-neighbour mask, so a client that draws a hull needs
every plate record of that hull. A fifty-thousand-block hull is 400 KB of records per observing
client. §9's M5 measures the WAL frame and the compaction time. M6 measures palette width. Nothing
measures the per-observer stream. "Lane flood" is one of SL8's eleven seam kinds.

**The consequence in the game's words.** Six players fly into a station's berth ring where twenty
player hulls sit. Each client wants every plate of every hull it draws. The report never says at what
distance a client stops receiving records, or what the window costs when twenty hulls are in reach.

**Fix.** Add a measurement: the bytes per second on one observer's lane, with twenty hulls of fifty
thousand blocks in reach, at the tier the reach ruling puts them at. Name the tier at which records
stop and a coarse form takes over.

---

### F11 — Small citation defects. WRONG, not load-bearing.

MEASURED today with `grep -n` and `sed -n` on the worktree:

- §3.2 cites `crates/core/src/built.rs:44` for `mass_g`. The field is at `built.rs:40`.
- §3.2 quotes *"Today an operator writes these rows; later a shipyard writes them"* as
  `built.rs:10-13`. The sentence spans `built.rs:9-10`.
- §2.1 and §2.5 cite R1 as `block_system_design.md:53`. R1 is at `block_system_design.md:52`.
- §1 cites `entity_kind.rs:46-71` for the registry idiom. `ALL` is at 46 and `from_tag` ends at 69;
  `def()` starts at 73 and falls outside the range the row describes.

**Fix.** Correct the four line numbers.

---

### F12 — The reserved-bit ledger does not add up as a table. WRONG, not load-bearing.

§2.5's table has a column headed "Spent by" and lists 1 + 2 + 11 = 14 against fifteen reserved bits,
which leaves ONE, not five. Only the prose works: the variant RELEASED two bits, so the arithmetic is
15 − 1 + 2 − 11 = 5. A one-way-door ledger must read correctly as a table.

**Fix.** Show the release as a negative row, or split the table into "released" and "spent".

---

## 3. What stands

I tried to break these and failed. The owner should not re-open them.

- **SL1 — a realm is told where it is.** STANDS. No record, no registry row and no digest names a
  realm's own placement. No chain folds an absolute. A hull's plates are addressed in the hull's own
  chunk basis, and the station never learns where the hull is from them.
- **SL2 — no occupant pose crosses.** STANDS. No record carries a pose. The attachment record carries
  a host cell, a face and a slot, which are the realm's own local address.
- **SL4 — physics and re-home are separate.** STANDS. Nothing on the crossing path names motion. A
  record is inert data. A hull, a moon and a rock cross by the same code whether or not they hold
  cells.
- **SL5 — one world.** STANDS. One registry, one decoder, one digest. A theme is a set inside the one
  registry, and every host decodes every theme. A realm's allow-list selects what a player may PLACE
  on a planet, never what a shard may DECODE. No reduced world, no second implementation, no
  config-selected variant, no placeholder mesh.
- **SL7 and SL9 — no per-child per-tick cost.** STANDS. The mass sum is incremental on an edit, not a
  re-sum of fifty thousand blocks. The trim re-derives at the remesh an edit already causes. No
  mechanism walks a parent's children, and no width is reserved for a child count.
- **The movement contract.** STANDS. The hull states mass ON CHANGE, which is the "what I AM" lane
  (`crates/core/src/built.rs:37` onward; `crates/sim/src/stub/drive.rs:639` derives the rating from
  the facts, MEASURED). No velocity crosses upward. No parent sets a speed. No re-clamp appears.
- **The seed ruling of 2026-08-27.** STANDS. A record is a diff, never a pure function of seed and
  position. §8 keeps concealed resources on a drop table.
- **The reach and visibility rulings.** STAND. Nothing walks the realm tree. Nothing draws a dormant
  realm. The report leaves the reach question to its owner, which is correct.
- **HR3 and HR5.** STAND. One record shape for a whole plate and a quarter-scale wedge. One registry
  for cells, sub-blocks and attachments. Every decoder is an exhaustive match with a refusal arm, and
  M2 owes the branch coverage.
- **The §1 code table.** STANDS, and I re-ran it. `ls crates/core/src` returns exactly the twenty-six
  files the row names. `grep -rn "BlockTypeId\|block_wal\|PaletteEntry\|ChunkKey\|Tier0Key"` over
  `crates/` returns ONE hit, a doc comment at `crates/io-prod/src/store.rs:6`. No world format exists.
  Every other citation in the table verified: `crates/core/src/tlv.rs:46` and `:193`;
  `crates/wire/src/channels.rs:288-296`; `docs/design/DEFERRED.md:869`;
  `crates/core/src/store_stamp.rs:42` and `:24-28`; `crates/wire/src/version.rs:350`, `:397-400`,
  `:418-428`; `crates/io-prod/src/trust.rs:41-43`; `crates/io-prod/src/store.rs:75`;
  `Cargo.toml:41`, `:70`, `:78`, `:83`; `crates/core/src/built.rs:30`;
  `crates/sim/src/capability.rs:106-117`; `crates/core/src/pose.rs:475-479`.
- **The bit arithmetic.** STANDS. The cell record sums to 64 bits with contiguous ranges and no gap.
  The attachment record sums to 64 the same way. 8 × 8 × 8 = 512 = 2⁹, so nine address bits are exact.
  A 1/16 m lattice needs 4096 = 2¹², so twelve bits, which leaves two reserved — as §2.3 states.
- **The storage arithmetic the report re-checked.** STANDS. I re-derived the 2.9 PiB starter world
  (6 × 4096² × 272 chunks, 62³ cells, 4 bits each), the masked-dense crossover (29,791 ÷ 7 = 4,256
  cells = 1.8 %), the clear-cut disc (19,406 × 1,845 B = 35.8 MB) and the 41.9 discs against 1.5 GB.
  All four are correct. F7 is about which budget the table then keeps, not about these rows.

---

## 4. The law conflicts the owner must rule on

Under SL6 the default is NO. Each item below is new data or a new arm, and the report must ask.

1. **The `BlockEdit` arm.** Data: a block edit — a cell, a record word, a fence. From an occupant's
   client through the gateway to the realm's shard, and shard to shard only if a ghost region
   straddles a seam. Why the receiver cannot compute it: a player's choice is not a function of the
   seed. Cost of doing without: no player can build anything.
2. **The registry identity digest on the client handshake.** Data: 32 bytes and a row count. From the
   gateway to a client, and between two shards on the ALPN string. Why the receiver cannot compute it:
   it is the other side's build identity. Cost of doing without: a client with a different registry
   meshes a plate as the wrong kind, and the no-drift gate never sees it.
3. **The realm salt for the trim hash.** Data: one 64-bit number per realm. From the realm's shard to
   a client that draws it. Why the receiver cannot compute it: the salt makes a 200 m hull stop
   repeating every metre, and it must be the same on every client. Cost of doing without: use the
   realm's own id, which the client already holds — this may need no new data at all, and the report
   must try that first.

---

## 5. What the report must do to stand

1. Fix F1. Re-write the digest column set and §4.4's example. This is the one error that would take a
   store offline.
2. Fix F4 and F7. Both are arithmetic, and each is one edit.
3. Add the two missing owner decisions: V2.1's surface value (F2) and V2.2's object parameters (F3).
   The record freezes on the same day, so the owner must see them together.
4. Replace §8's "SL6 — NONE" with the three requests in §4 above.
5. Turn §3.3's law argument into a D-entry (F6).
6. Add the two owed measurements: the two-shard-kind fixture (F9) and the observer's record lane (F10).
7. Name where non-overlap is enforced (F8), and correct the citations (F11, F12).
