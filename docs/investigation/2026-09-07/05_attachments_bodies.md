# 05 — Attachments that take no space, and the body tree a joint forces

**Domain:** V2.5 sub-blocks (a HUD on a face, a rotation joint, a rail, a piston) and the structural
consequence of a joint: a construction becomes a tree of rigid bodies.

**Status:** an investigation report. It is not binding. It serves `owner_decisions_2026-09-07_voxels.md`
V2.5, V2.6, V2.8, V2.9 and SL10, under every earlier law. The investigation base
(`docs/investigation/block_system_design.md` §2.4, §4.6, §6.9, §7.9–7.11, §7.16) is an input. Where it
disagrees with a ruling, the ruling wins, and §9 lists the claims this report found stale.

**Revision.** This is revision 2. Two refuters read revision 1 and refuted it
(`verdicts/attach_law.md`, `verdicts/attach_feasibility.md`). §13 lists every finding and what this
revision did with it. Three claims of revision 1 were false, and this revision deletes them.

**Method.** Every claim about today's code cites `file:line`, re-read in this worktree. Every number is
marked MEASURED (with the command) or ESTIMATED (with the arithmetic). Nothing was built or run.

---

## 0. The answer in one page

1. **An attachment is a row in a side table of the realm's own store, keyed by the block's address plus a
   face plus a slot byte.** It has no cell, no volume, no collider and no place in the mesh. The block
   record does not change by one bit. It crosses the wire as one more tag in the chunk's diff, one hop
   from the owning realm (SL10 clause 6). *Example: a player sticks a fuel gauge on a hull plate. The hull
   realm writes one row `(plate address, face +Y, slot 0) → {kind: Display, widget: Gauge, range 0..1,
   unit: fraction}` beside its blocks. The plate stays a plate. A crate can still be placed on top of the
   plate.*

2. **A HUD is an attachment of kind `Display`.** The foundation plants the closed widget registry, the
   attachment blob's reserved tag range for signal bindings, and the scoped channel key type. It plants no
   signal code, and it DRAWS nothing until the value lane exists. *Example: at P9 the player binds the
   gauge to the channel `fuel` in scope `Construct`. The binding is a new tag in the row's blob. A client
   that does not know the tag skips it.*

3. **A joint makes a second rigid BODY inside the SAME realm. It never makes a child realm.** The
   containing realm's shard integrates the joint, because it is that realm's physics authority (SL4) and
   because a constraint needs both bodies in one physics world, and a shard's physics world is private
   (HR1). *Example: a turret on a hull. The hull realm holds body 0 (the hull grid) and body 1 (the turret
   grid) joined by a revolute joint on the shared face. The hull's shard authors the turret's pose in the
   hull's frame every tick, and states it on the hull's own realm row. The star system still sees ONE
   child, the hull, and still receives six numbers per tick.*

4. **A block placed on the turret finds its grid through the body it was aimed at.** The edit names
   `(realm, body, chunk, cell, face)`. The turret's grid is its own lattice, carried rigidly by the turret's
   body pose. The block goes into the turret's lattice, and the client draws it at the turret's pose.

5. **A rail and a piston are the same joint with a different degree of freedom.** A piston is a prismatic
   joint along a face normal. A rail is a prismatic joint whose axis is a path of rail cells on the parent
   body's grid. One record, one closed enum `{Revolute, Prismatic, RailPath}`, one scalar of motor state.

6. **A joint's angle needs a deterministic sine and cosine that the crate CARRIES.** SL10 clause 4 forbids
   a call into the platform's `sin` and `cos`. The joint therefore reads an integer table keyed by the
   same angle grid the motor scalar counts on. Revision 1 said "no transcendental" and was WRONG.

7. **What the first formats must carry, exactly:**
   - the saved block record: **nothing new**;
   - the block ADDRESS: **a body id** (`BlockAddr { body, chunk, cell, sub }`, body 0 = the realm's own
     grid) — **a one-way door in the STORE, frozen ONCE together with the V2.4 sub-cell field**;
   - the chunk store: keyed by the same body id — **the same door**;
   - the chunk DIFF: the body id rides INSIDE the delta's TLV envelope — **additive, not a door**;
   - the edit-pyramid entry: nothing new inside the entry; its store key carries the body — **the same
     store door**;
   - the construction (the realm's own store): a **body-tree family** (bodies, joints, one scalar each) and
     an **attachment family**, both TLV-framed with append-only tags — not a door, because the store's tags
     already grow additively (`crates/sim/src/stub/built_store.rs:19-24, 79-81`);
   - the attachment key: **the block's address type plus a face byte plus a slot byte**, never a separate
     numbering — **a one-way door**.

8. **Law conflicts: THREE SL6 asks, listed in §11.** Revision 1 said "NONE" and was WRONG: it recommended
   two new wire rows without asking for them. The asks are (a) a body's pose on the realm's own row, (b) a
   HUD's value level, (c) the attachment tag inside the chunk diff. Nothing crosses the hull↔star-system
   boundary. A joint between two REALMS (a docking clamp) is a realm-level relation, designed later.

---

## 1. What exists today — the code is the only truth

**What a hull IS.** A hull is one `BuiltBody` row in its own store: a realm id, an owner, a blueprint id,
a `bound`, a `look`, a `BuiltFacts` record and a fence (`crates/core/src/built.rs:61-83`). Its facts are
mass, cross-section, drag, a maximum push and a maximum turn (`built.rs:37-53`). Its parent holds a
`Berth` for it: an offset, the bound and the look as readings, and a fence (`built.rs:91-107`). Nothing
in the row is a position, and a test enforces that (`built.rs:189-212`). The hull is drawn as one box
(`crates/bins/src/bin/build-ship.rs:52-54`). **A hull has no grid, no chunk and no block today.**

MEASURED, re-run in this worktree:

```text
$ grep -rn "SubBody\|BodyPart\|Attachment" crates --include='*.rs'
(no output)

$ grep -rni "attachment" crates --include='*.rs'
crates/wire/src/session_flow.rs:39:    /// a duplicate attach at the same fence re-sends the existing attachment.
```

So no attachment type and no body type exists in any form. Revision 1 printed a case-sensitive command
beside a lower-case hit; the command could not have produced the output shown. The conclusion is
unchanged and now stands on the command that proves it.

**How a hull moves.** The hull scales its stick by its rating and sends six whole numbers — a push and a
turn in its own frame — on `InterShardFlow::ChildDrive` (`crates/wire/src/intershard.rs:1377-1397`),
and its facts on change on `ChildFacts` (`intershard.rs:1404-1426`) with a closed set of declared states
(`intershard.rs:1437-1442`). The parent rotates the push, adds its own pull and drag, and integrates
(`crates/sim/src/stub/drive.rs:135-175`), after the routing and capability guards in `on_child_drive`
(`drive.rs:303-325`). The rating is per hull and never crosses (`drive.rs:18-33`).

**What a realm may do.** A shard's capabilities are data: `voxel: Option<VoxelGeometry>`,
`functional_blocks`, `block_edit`, `surfaces`, `seats`, `signal_relay`, `hull_host`, `self_driven`,
`integrates_children` (`crates/sim/src/capability.rs:47-91`). `block_edit` and `surfaces`/`seats` each
require a voxel realm, checked in the one constructor (`capability.rs:122-135`). The ship profile has
`voxel: Cartesian`, `surfaces`, `seats`, `self_driven` and `integrates_children`
(`capability.rs:260-274`). The planet profile has `Spherical` and `integrates_children`
(`capability.rs:244-258`). **The asteroid profile has `voxel: Cartesian` and `block_edit` and NOTHING
else** (`capability.rs:276-282`). `FrameSpace` does not exist; the doc comments name it as owed
(`capability.rs:11, 18`).

**What the store is.** A realm's own store is one flat table with a prefix byte per family and TLV-framed
rows whose tags are append-only (`crates/sim/src/stub/built_store.rs:19-24, 79-81`). The TLV framing
carries a schema id, a writer maximum tag and a required maximum tag; a reader skips unknown tags by
length and refuses a missing required tag, and a dest below the version floor REFUSES rather than decodes
to default (`crates/core/src/tlv.rs:19-23`).

**What the wire carries to the client.** A realm row is `SceneRow { realm, parent, pose, bag }`
(`crates/wire/src/channels.rs:378-391`); the bag is the realm's own look blob, tagged and skip-unknown,
and the row's stamp is explicit PER ROW (`channels.rs:384-388`). An occupant row is
`EntitySnap { entity, pose }` (`channels.rs:331-335`), drawn behind `INTERP_BUFFER_MS = 120.0`
(`channels.rs:328`). A realm sorts its occupants into cubes and ships one body per cube to the observers
that hold it (`crates/sim/src/stub/interest.rs:1-29`; DEFERRED D-9 🟩, `docs/design/DEFERRED.md:2925`).
The bulk lane exists with EXACTLY ONE arm, `BulkMsg::Blob { kind: BulkKind, bytes: Vec<u8> }`, and
`BulkKind` is `{ChunkSnapshot, ChunkDelta, Catalog}` (`channels.rs:283-296`). The chunk delta is an
opaque byte string inside that arm. The client's render seam is
`MeshPrim { vertices, color_rgba, transform: PrimTransform { translation, scale, rotation } }`, with no
engine type in it (`crates/client/src/realm_scene.rs:775-795`).

**How many arms the inter-shard wire has.** MEASURED:

```text
$ awk '/^pub enum InterShardFlow/,/^}/' crates/wire/src/intershard.rs | grep -cE "^    [A-Z]"
44
```

**44 arms** (`crates/wire/src/intershard.rs:124`). The investigation base's decision board says
*"27 arms + 3 reserved names = 30 … There is no headroom left"*
(`docs/investigation/decision_board.md:1172`). The board is stale by seventeen arms. Revision 1 relayed
the board where the standing rule says to read the code. **This report no longer argues from any arm
ceiling.** `BlockEdit` (P6), `Coupling` (P8) and `Signal` (P9) are reserved names
(`crates/wire/src/intershard.rs:34-35`), and this domain still adds no arm — for the reason in §11, not
for a ceiling.

**What physics library exists.** MEASURED: `grep -rn "rapier\|parry" Cargo.toml crates/*/Cargo.toml`
finds one comment (`Cargo.toml:96`) and no dependency. Rapier is not adopted. DEFERRED D-19 (SPIKE-6a,
rapier snapshot and determinism) is 🟥 (`docs/design/DEFERRED.md:6240`). The decision board carries the
adoption as O28 / `[USER DECISION 4-B]`. **Nothing in this report adopts rapier.** Where the base says
"rapier joint", this report says "a constraint the containing realm's shard solves", and names rapier as
one option for the owner.

**What contact machinery exists.** MEASURED: `crates/physics/src` holds `celestial`, `motion`, `taxonomy`
and `worldgen` only. There is no collider, no shape cast and no character controller in the tree; every
hit for "controller" is a flight steering loop (`crates/bins/src/flight.rs:32`,
`crates/client-harness/src/nav.rs:16`). **Standing on a moving floor does not work today, and this report
must not say it does.**

**What the transcendental rule already costs the tree.** `crates/physics/src/celestial.rs:23-25` records
that the residual `sin`/`cos`/`sqrt`/`atan2` divergence is *"STILL owed to the SPIKE-6a cross-target
build-and-diff gate"*. SL10 clause 4 forbids the libm call outright inside the generator
(`owner_decisions_2026-09-07_voxels.md` V1.4). A joint's angle is in the same class (§4.4).

**What the transfer registry can carry.** `SHIP_DEF.max_state_bytes = 8192`
(`crates/core/src/entity_kind.rs:207-214`). `DEBRIS_DEF` is `class: Transient`,
`ghost_policy: NeverTransferOnly`, `continuity: BallisticReadvance`, `loss_budget: LossBudget(4)`,
`max_state_bytes: 128` (`entity_kind.rs:225-232`). The `TransferableKind` trait with `rebind_refs` is
unbuilt (D-31 🟧, `DEFERRED.md:18-64`); the N+1-key CAS for a ship plus `ChildOf` passengers is unbuilt
(D-33 🟥, `DEFERRED.md:96-116`). The crossing plan makes the hull's exterior a directory key at the parent
and keeps the hull's own key at its own shard: *"The hull's shard keeps running. What changes hands is
the AUTHORSHIP of its placement"* (`docs/design/realm_crossing_plan_2026-09-02.md:61`); the interior world
is *"owned by the ship-shard throughout and untouched"* (`docs/design/transfer_protocol.md:376`).

**What co-hosting exists.** `CoHostedAuthority` is a per-realm map of fences that lets one process hold
more than one realm (`crates/sim/src/stub/realm_head.rs:47-48`), and the harness populates it
(`crates/harness/src/topology.rs:305-307`). This matters to §4.2: two realms CAN share one process today.

**What does not exist as code at all.** MEASURED:
`grep -rn "GridMapping\|ChunkKey\|CellIndex\|Tier0Key" crates --include='*.rs'` returns nothing, and
`AnchorGen` appears only in two doc comments (`crates/core/src/fence.rs:18`,
`crates/sim/src/capability.rs:40`). Every use of those names in this report is a name from the
investigation base or a proposal, never a claim about today's code.

---

## 2. The attachment record (question 1)

### 2.1 What an attachment is, and is not

An attachment is the fourth kind of thing in a realm, beside a cell, an entity and decoration:

| Thing | Stored? | Authoritative? | Collides? | Occupies a cell? |
|---|---|---|---|---|
| a block (a cell) | yes | yes | if its shape says so | yes |
| an entity (an occupant, a body) | yes | yes | yes | no |
| decoration (grass) | no | no | never | no |
| **an attachment** | **yes** | **yes** | **never, by itself** | **no** |

The collision rule the base states once — *a thing collides if and only if it occupies a cell or it is an
entity* (`decision_board.md` §1 "The collision rule") — stays exactly as written. An attachment is
neither, so it never collides. A joint's CHILD collides because the child is a BODY, which is the entity
arm of that rule. The HUD's glass never stops a bullet, and a piston's ROD never collides on its own; the
piston HEAD block does, because it is a block on the child body.

*Example: a player places a hull plate, then a HUD on its outer face, then a crate on the same face. The
plate occupies its cell. The HUD occupies nothing. The crate occupies the next cell out. The HUD is now
inside the crate's volume and is not drawn until the crate goes; nothing refused the crate, because
nothing could (V2.5: "it should not take space or prevent placing another voxel on top").*

### 2.2 The key: the block's address, a face, and a slot

```text
AttachmentKey = (BlockAddr, face: u8, slot: u8)
    BlockAddr = the block record's address, in full (see §6: a body id, a chunk, a cell, a sub-cell)
    face      = 0..=5 in the chunk's OWN basis, the same basis a block orientation is stored in
                (block_system_design.md §2.7.1: orientation is expressed in the chunk's basis and is
                never re-interpreted); values 6..=255 are REFUSED on decode, never decoded to a default.
    slot      = 0 in the foundation; a second attachment on one face takes slot 1.
```

Why a face:

- Every example the owner gave sits on a face: a HUD on a face; a joint between two blocks sits on the
  face they share; a piston pushes along a face normal; a rail carriage rides a face.
- A face already has a stable numbering in the record the block uses. A second numbering would drift
  from the first the day sub-metre blocks (V2.4) or a new basis land.

**Why a slot byte, and why revision 1 was wrong to leave it out.** Revision 1 said one attachment per
face, and that taking one of the face byte's 250 spare values later would be *"an append, not a
migration"*. That is FALSE. The skip-unknown rule covers TAGS inside a blob, not the bytes of a KEY
(`crates/core/src/tlv.rs:19-23`): a reader that refuses face 6 hard-errors, and a store row whose key
holds face 6 is unreadable to every earlier reader. Widening the face byte after the first attachment is
saved costs a version-floor bump on every persisted attachment in the world, and the realm's store
travels to another node on a re-home, so a rolling deploy meets it.

The cure costs one byte in a key and closes the door on the first write: the key carries a `slot` byte
from the start, always 0 today. A player who wants a gauge AND a hinge on one plate face gets slot 1
later, and no reader ever meets a key shape it does not know. The refusal rule stays: a slot the
attachment kind's registry row does not allow is refused loudly at placement, never stored.

*Example: a player wants a fuel gauge and a hinge on the same hull plate face. Under revision 1 the
second was refused forever, and lifting the refusal migrated every stored HUD in the world. Under the
slot byte the hull realm writes the hinge at slot 1 when the owner opens that door, and every older
reader still reads slot 0.*

**Sub-metre blocks (V2.4, report 04).** Report 04 gives a block a `sub_scale` and a `sub_addr` field
(`04_block_record_registry.md` §2.3), and report 01 fixes the address width at level 4 bits plus index
3×4 bits (`01_grid_family.md` §3.2). That field is part of `BlockAddr`, and an attachment on a small
block keys on it for free. The attachment key must be defined as "the block's address type plus a face
plus a slot", never as "chunk, cell, face" spelled out. This is a one-way door on the SHAPE of the key
(§7, door 2), and it cannot be closed before report 04's answer (§7, door 1's dependency).

### 2.3 What the row stores

The row's value is a TLV blob with its own schema id and append-only tags, exactly like a body row today
(`built_store.rs:19-24`):

```text
AttachmentRow (TLV, schema ATTACHMENT)
  tag 1  kind        u16 — a row in the CLOSED attachment-kind registry (Display, Joint, RailCarriage, ...)
  tag 2  params      bytes — the kind's own TLV blob, capped at the kind's `max_params_bytes`
  tag 3  placed_by   AccountId — who put it there (the provenance the block record has no room for)
  tag 4  fence       Fence — which commit last moved the row
  tags 16..=31       RESERVED for signal bindings (P9): the channel NAME + SCOPE per port, never a
                     resolved id (R19). Empty in the foundation. A reader that does not know a tag
                     skips it by length (tlv.rs:19).
  tags 32..=47       RESERVED for the joint's live state (§4.4): the motor scalar, checkpoint-carried.
```

The kind registry follows the shape of the entity-kind registry (`crates/core/src/entity_kind.rs`): a
`#[repr(u16)]` closed enum, an `ALL` array, `from_tag` that errors on an unknown tag, and a static `def()`
row per kind (`max_params_bytes`, whether the kind makes a body, which faces it may sit on, which slots
it may take). Adding a HUD widget or a joint kind is one row. Adding a kind never touches the block
record.

**What is NOT in the row:** a position (the key is the position), a velocity, a resolved channel id, a
mesh, a collider. A `cover_ref` indirection to a second table (the base's `block_cover (u32 local, u8
face, u32 cover_ref)`, `block_system_design.md:2112`) is dropped: the row holds its blob directly. One
table, one read, one row per attachment.

### 2.4 Where it lives: the same store as the blocks

The attachment family is one more prefix byte in the realm's flat store
(`built_store.rs:79-81` shows the pattern: `BERTH = 1`, `BODY = 2`). It sits in the SAME file as
the chunk records, so:

- it moves with the realm on a re-home (HR1: a realm's store is private, and the whole file travels);
- it flushes under the same fence as the block edits, so a hard kill cannot leave an attachment on a block
  that was never placed;
- the cell-change hook (§2.6) deletes or re-validates the attachment rows of a cell in the same
  transaction that changes the block. **Plant that hook in the first edit slice.** Without it every later
  table learns about removals by retrofit.

*Example: a gunner shoots a hull plate off a station. The station's shard writes the break to its edit
log, and in the same transaction removes the HUD row on that plate's face and the joint row that held a
turret to it. Nothing stays orphaned.*

### 2.5 How it crosses the wire: one tag INSIDE the chunk diff (SL10 clause 6)

SL10 says everything the seed does not decide crosses as a one-hop diff from the owning realm, and names
attachments by word (V1.6). The chunk delta the base designs is a TLV envelope carried as the opaque
`bytes` of `BulkMsg::Blob { kind: ChunkDelta }` (`crates/wire/src/channels.rs:283-296`):
`{1: sparse block records, 2: mask + packed, 3: concealed reveals, 4: pyramid entries}`
(`block_system_design.md` §3.10). **An attachment is tag 5 in that same envelope**, a sorted list of
`(cell, face, slot, AttachmentRow bytes)` for the chunk. It rides:

- the same message, so a chunk's blocks and its attachments arrive together and can never disagree by a
  tick;
- the same interest, because a chunk is shipped to the observers that hold it, and an attachment is part
  of its chunk;
- the same skip-unknown rule, so an older client draws the blocks and skips the attachments.

**The body id rides inside the SAME envelope, as tag 6 of the delta** (tag 0 is reserved for a file's own label by the store's convention, `built_store.rs:24`). It is NOT a positional field of a
new `BulkMsg` arm. Revision 1 wrote the message two ways — an opaque blob in §2.5 and
`BulkMsg::ChunkDelta { realm, body, chunk, ... }` with a "postcard is positional" warning in §6.2 — and
a format stated two ways cannot be frozen. **This revision picks the envelope.** The postcard-positional
warning applies to `BulkMsg`'s arms and not to the bytes those arms carry, so it is deleted.

The HUD's LIVE content (the value it shows) is NOT in the diff. It is a level that changes at tick rate
on a different lane (§3.3), and that lane is an SL6 ask (§11).

**No new wire arm and no new lane for the attachment itself.** The chunk-delta lane is the one P6 already
owes. An attachment adds a tag inside an envelope the same lane already carries, and a tag is additive by
construction of the TLV envelope. The ASK is that the envelope gains this tag at all (§11, ask C): the
owner reviews the bytes on the lane, not only the arm list.

### 2.6 The cell-change hook, and the three cases revision 1 did not answer

Revision 1 named an `on_block_removed` hook. A removal is not the only edit that can strand an
attachment. The hook is therefore `on_cell_changed(addr, before, after)` and it runs inside the edit
transaction:

1. **The cell is emptied.** Every attachment row on every face and slot of that address is removed. A
   joint row anchored on that address is removed too, and the detach question of §4.6 fires.
2. **The block is REPLACED in place** (a player swaps a plain hull plate for an armoured one). The cell
   is not empty, so nothing is deleted blindly. Each attachment row on the address is re-validated
   against the NEW block kind's registry row: a face the new kind does not offer, or a slot it does not
   allow, loses its row in the same transaction, and the shard states the removal in the same diff. A row
   the new kind still allows survives untouched. *Example: a player swaps a hull plate for a window. The
   window's registry row offers no outer face for a `Display`, so the fuel gauge is removed with the
   swap, in one commit, and the client that draws the chunk never sees a gauge on glass.*
3. **The block a joint's CHILD is anchored to breaks.** Revision 1 covered only the joint's PARENT block.
   The rule is symmetric: the hook runs on both anchors, and the joint row goes when either anchor goes.

---

## 3. The HUD (question 2)

### 3.1 The three parts of a HUD, and who owns each

| Part | Owner | When it exists | Lane |
|---|---|---|---|
| the ATTACHMENT (this face carries a Display of widget W, style S, range R, unit U) | the owning realm's store | placed (foundation) | the chunk diff, one hop |
| the WIDGET's static look (its frame, its dial, its glyph set) | the client's predefined set | shipped with the client | none — client-held STYLE (V2.6) |
| the VALUE and its state (Fresh / Stale / Lost) | the owning realm's shard, from the bound signal | P9 | a level on the snapshot lane, per attachment key — an SL6 ask (§11, ask B) |

The owner's words: *"a HUD on any block face that listens to a signal and shows it with a widget from a
predefined set."* The widget's art is client-held under **V2.6**, not under SL10: *"Style can be picked up
by the client, so when same blocks of that type are connected and construct the mesh, it will render small
additional details, that are not stored on the BE."* Revision 1 cited SL10 here, and that was wrong: SL10
covers what *"a fixed seed and an address decide and nothing else decides"*
(`owner_decisions_2026-09-07_voxels.md` V1.1), and a gauge's dial is a function of neither a seed nor an
address. The conclusion is unchanged; the authority is V2.6. Citing SL10 for it would invite a later
reader to stretch SL10.

The value is state, so the server states it. The client never reads a channel and never decides what a
value means; it draws the number the server gave it, on the scale the server gave it, in the dial the
server named.

### 3.2 What the foundation plants so that P9 binds additively

1. **The attachment kind `Display`** with params
   `{ widget: WidgetKind(u16), style: u8, range_lo: Fx, range_hi: Fx, unit: UnitCode(u16), label: [u8] }`
   and the closed `WidgetKind` registry shipping with ONE row, `Numeric`. The registry SHAPE is the
   expensive part; the rows are cheap (the base's §6.9 finding stands).
   **The range, the unit and the label are new in this revision.** Revision 1's params were
   `{ widget, style }` only, so a client that received the value `0.61` had to invent a scale, which is a
   magic number on the client and is the client deciding what a server value MEANS. The owning realm
   states the scale ONCE, on placement, in the chunk diff. *Example: the fuel gauge states
   `range 0..1, unit: fraction`; the thruster temperature beside it states `range 0..1200, unit: kelvin`.
   The same `Numeric` widget draws both correctly, and the client invents nothing.*
2. **The reserved binding tags (16..=31) in the attachment blob**, empty. R19's rule: a binding stores the
   player's chosen NAME and its SCOPE (`Realm | Construct | Net | Protocol`); a channel id is computed at
   load, never stored.
3. **The `ChannelKey` type in `vd-core`**: `H(scope ‖ name)` with the loud collision check (base §7.18
   item 3, ~200 lines). It stores nothing and touches no hot path. It exists so that the FIRST persisted
   binding, whenever it comes, is already scoped — R19 names this a one-way door (decision board door 49).
   The foundation persists zero bindings, so the door's real deadline is "before the first binding is
   saved"; planting the type now costs nothing and closes the door early.
4. **The value lane is NAMED and NOT BUILT**: `WidgetLevel { key: AttachmentKey, value: Fx, state:
   Fresh|Stale|Lost }` on the unreliable latest-wins snapshot lane, addressed to the observers that hold
   the attachment's chunk, stamped by the frame it rides in. It is an **SL6 ask** (§11, ask B) and it is
   declared with its consumer at P9 (the incremental-freeze rule of the wire). The foundation names it in
   a reserved comment and ships no bytes.
5. **The render seam for a face overlay**: the client library lowers an attachment to an engine-free
   primitive beside `MeshPrim` — a quad on a face with `{ widget kind, style, range, unit, label, value,
   state }`. An engine (Bevy today, Unreal in the other worktree, V2.8) rasterises it. Nothing in this
   seam names a render pass, an atlas format or a text library; the base's Bevy-specific pass details
   (`AlphaMask3d`, the prepass, `bevy_text`, §6.9) are the engine's business and are not part of the seam.
   **The foundation plants the type and its test and DRAWS NOTHING.** Revision 1 planted a primitive that
   drew the widget's static chrome with no value lane behind it. A dial with no needle, standing on a hull
   plate in the shipped game, is a placeholder drawing, and the FINAL-backend law and the
   no-placeholder-rendering rule both refuse a stand-in in the one world. The overlay draws its first
   pixel when the value lane lands at P9.

### 3.3 One decision the owner must take: value-level or draw-list

The base (§6.9) has the server compose a bounded DRAW-COMMAND list per panel (150–360 B typical,
ESTIMATED by the base) and the client rasterise it. The owner's V2.5 wording has the client draw a
predefined widget from a value.

- **A. Value-level (recommended).** The server ships `(key, value, state)`. The client draws the
  predefined widget on the server-stated range. Formatting a number into digits happens on the client.
  That is presentation of a server-stated value, the same as drawing a server-stated pose as pixels; it
  is not client-composed state.
- **B. Draw-list.** The server formats and lays out; the client is a rasteriser. Per-widget bytes rise
  (the base measures a typical widget at 150–360 B, worst case 3 KB, `block_system_design.md:14197`),
  server compose time per subscriber appears, and a closed 10-command set with a 0..4096 panel space
  becomes a wire ABI (a one-way door the base names as door 32).

**The bytes, re-derived from THIS report's key.** Revision 1 quoted 13 B per changed binding. That number
is the base's and it is for a NARROWER key: `(panel_id u32, slot u8, value Fx)` = 13 B
(`block_system_design.md:14199`). This report's key is the block's address, which is wider by design.
ESTIMATED, by adding this report's own widths: body 2 B, chunk key 8 B (the base's `chunk_meta` uses a
`u64 chunk_key`, `block_system_design.md:2113`), cell index 3 B (the base's pyramid entry spends 18 bits
on `cell`, `block_system_design.md:2156`), sub-cell 2 B (report 01 §3.2: 16 bits), face 1 B, slot 1 B =
**17 B for the key**, plus a value 4 B and a state 1 B = **ESTIMATED 22 B per changed binding**, against
the base's 13 B. Every rate the base derives from 13 B (52 kB/s naive, 6.7 kB/s after the angular floor,
`block_system_design.md:14203-14204`) is therefore understated by about seventy percent: ESTIMATED
88 kB/s naive and 11 kB/s after the floor.

- **A refinement worth measuring (A2).** The realm may hand each observer a short HANDLE for each
  attachment it is streaming to that observer, exactly as the interest plan already hands out per-cube
  bodies (`crates/sim/src/stub/interest.rs:1-29`). The steady-state row is then
  `(handle u16, value Fx, state u8)` = ESTIMATED 7 B, and the full key travels once, in the chunk diff
  that placed the attachment. The cost is one small per-observer table on the shard. §8 item 5 measures
  A against A2.

A is recommended for the foundation's seam SHAPE. B is not foreclosed: a widget kind `DrawList` whose
value is a command list is one more predefined widget on the same seam. So A→B is an append, B→A is a
wire-ABI retirement. Take A now. **The margin over B is real but smaller than revision 1 claimed**, and
it is ESTIMATED, not measured.

*Example: the fuel gauge on the hull plate. At P9 the hull's shard reads the `fuel` channel each tick,
notices the value changed from 0.62 to 0.61, and ships `(plate +Y slot 0, 0.61, Fresh)` to the three
observers who hold that chunk. Each client draws its `Gauge` at 61 % of the range the placement stated.
When the fuel sensor is shot off, the shard ships `(plate +Y slot 0, 0.61, Lost)` once, and every client
dims the dial and shows the no-signal mark; it never shows zero.*

---

## 4. The joint (question 3)

### 4.1 The two candidate shapes

A construction with a joint is two rigid bodies joined by a constraint. Two shapes are possible:

- **(A) The joint child is a BODY inside the containing realm's simulation.** The hull realm holds body 0
  (its own grid) and body 1 (the turret's grid). The hull's shard authors body 1's pose in the hull's frame
  every tick, from the joint's motor state. Both bodies live in one physics world, one store, one shard.
- **(B) The joint child is a child REALM.** The turret is its own realm: its own shard, its own directory
  keys, its own store, its own reach, its own crossing saga, and it sends a push and a turn up to the hull
  as any child does.

### 4.2 The argument from the laws

**HR1 decides it.** A constraint couples two bodies in ONE solver step: the solver reads both poses,
both masses and the joint, and writes both. A shard's physics world is private (HR1: *"a shard's
World/rapier/redb is private"*, `CLAUDE.md:103`), and the project's determinism rule says the physics
state is checkpoint-carried and *never re-simulated cross-host* (`CLAUDE.md:266-267`). So the two bodies
of a joint MUST be in one shard, which means one realm.

**How (B) fails, stated honestly.** Revision 1 said *"(B) cannot solve a constraint at all"*. That is
over-stated: two realms CAN share one process today, because `CoHostedAuthority` exists
(`crates/sim/src/stub/realm_head.rs:47-48`) and the harness populates it
(`crates/harness/src/topology.rs:305-307`). The honest form is worse for (B), not better: under (B) the
solve works only while the orchestrator happens to co-host the two realms, and the orchestrator may
re-place a realm at any time. A design that works until a demand-driven scheduler moves a shard fails
sometimes rather than always, which is the harder defect. **The conclusion is unchanged.**

**SL4 agrees.** Physics is a per-realm capability that PRODUCES a placement for each child in the realm's
frame. The hull realm already owns the physics of what is inside it. A turret is inside it. Its pose is
the hull realm's placement to author.

**SL1 and the movement contract agree.** Under (B) the turret would send a torque up and receive its
placement down, one hop. But the hull's parent — the star system — must then know nothing of the turret,
and it does not: the hull states ONE push and ONE turn (`intershard.rs:1377-1397`) and ONE mass
(`intershard.rs:1404-1426`). Under (A) that stays true with no new field: the hull sums every thruster on
every body, rotates each body's thrust by that body's joint pose into the hull frame, divides by its own
total mass, and speaks six numbers. The turn is stated as an angular ACCELERATION, so the hull divides its
torque by its own inertia, which changes as the turret turns, and the parent never needs the inertia.
**Nothing new crosses the hull↔system boundary.**

**HR2 and the crossing plan agree.** A hull crosses as a realm: the exterior key moves to the new parent,
the hull's own key and process stay (`realm_crossing_plan_2026-09-02.md:61`). Under (A) the
turret is INSIDE the hull's store and crosses with it, and the interior is *"untouched"*
(`docs/design/transfer_protocol.md:376`). Under (B) every joint is one more exterior key, one more N+1
member of the atomic bundle D-33 owes, one more saga per crossing, and the base's own warning comes true:
*"a ship whose turret arrived and whose piston boom did not"* (`block_system_design.md:14989-14992`).

**The realm-hood rule agrees.** *"You are a realm when something can be INSIDE you"* (DEFERRED D-MOVE-2,
owner 2026-08-31). Nothing is inside a turret. A gunner on a turret platform stands on a moving FLOOR of
the hull realm. The gunner's pose stays in the hull's frame (`FrameRef::ShipLocal`,
`crates/core/src/pose.rs:91`); no sub-frame is added to the wire. A person walking through a hinged door
must not cross a realm boundary to do it. **The machinery that keeps his boots on that floor does not
exist** (§1: no collider, no shape cast, no character controller), so this is a design intent and an owed
measurement (§8 item 3), never a settled fact. Revision 1 asserted it as one.

**SL9 and the reach ruling agree.** Every realm states a reach, is tested by its parent, boots on demand
and holds an exterior lease at its parent (the lease exists today: `crates/sim/src/stub/realm_head.rs:50-58`;
the reach test and the boot-on-demand are the 2026-09-02 ruling, not that code). A hinge does not deserve
a reach test, a boot and a lease.

### 4.3 The argument from cost

- **A realm is a shard.** *"The port band holds roughly five hundred"* (D-MOVE-2). ESTIMATED: a hull with
  six turrets, four doors, two pistons and a landing gear is thirteen processes under (B), and one under
  (A). At five hundred realms per node, (B) spends the whole node on forty hulls.
- **A body is a pose row.** Under (A) a body costs the realm one pose per tick to the observers that hold
  it — the same cost as one occupant (`interest.rs:1-29`) — plus the joint arithmetic. ESTIMATED: the
  kinematic pose of a body is one quaternion product, one table lookup and one add per tick. UNMEASURED;
  owed (§8 item 1).

### 4.4 The recommendation: ONE realm, a tree of bodies, kinematic by default

**Recommend (A).** A construction is a realm with a BODY TREE:

```text
BodyId(u16)            0 = the realm's own grid; it never moves in the realm's frame
Body                   { id, parent: BodyId, joint: Joint, grid: its own chunk lattice }
Joint                  { anchor_parent: (BlockAddr, face),      the face on the parent body's block
                         anchor_child:  (BlockAddr, face),      the face on the child body's block
                         dof: Dof,                              Revolute{axis} | Prismatic{axis} | RailPath{..}
                         limits: (lo, hi) on the dof's scalar,
                         state: scalar q, on an integer grid (the project's physics→control rule) }
```

- The tree is a TREE: a cycle is refused at placement by union-find, or the closing link becomes a weld
  (the base's §7.16 invariant 6 — validated, and it is the one rule that removes the "double piston" class
  of defect the base documents from shipped games).
- **Kinematic by default.** The child's pose is an exact analytic offset of the parent body's pose:
  `pose(child) = pose(parent body) ∘ T(anchor_parent) ∘ R_or_S(dof, q) ∘ T(anchor_child)⁻¹`. The motor
  drives `q`. This needs no solver, so it is buildable with what the tree has today.
- **★ The rotation needs a sine and a cosine, and the crate must CARRY them.** Revision 1 said this form
  is *"bit-deterministic (only multiply, add and one quaternion product; no transcendental)"*. That is
  FALSE. A revolute joint at angle `q` builds the quaternion `(cos(q/2), axis · sin(q/2))`, and sine and
  cosine are transcendental. SL10 clause 4 forbids exactly this: *"No call into the platform's
  transcendental functions (`sin`, `exp`, `pow` from libm); where a curve is needed, the crate carries its
  own deterministic implementation or an integer table"* (`owner_decisions_2026-09-07_voxels.md` V1.4).
  The tree already carries the same debt for orbits (`crates/physics/src/celestial.rs:23-25` records the
  residual libm divergence as owed to SPIKE-6a). **The rule for a joint:** `q` counts on an integer angle
  grid, and the joint reads the sine and the cosine from a table indexed by that grid, carried in the same
  crate on both hosts. The table's size is set by the angle grid, which is a tuning field, never a literal.
  **The gate:** the same joint at the same `q` produces a byte-identical body pose on x86-64 and on
  aarch64 (§8 item 7). Only with that table does "no new dependency, deterministic" stand.
  *Example: a gunner turns the turret to 37 degrees. The hull's shard on an x86-64 node and the same
  shard restarted on an aarch64 node read the same table row, so the floodlight the player welded to the
  barrel sits in the same place on both runs, to the last bit.*
- **A dynamic constraint is an option, not the default.** If the owner adopts a rigid-body library
  (O28 / `[4-B]`, D-19), a joint may be solved by it for a small flagged class (a loose crane). The record
  above does not change: a solver reads the same anchors, dof, limits and `q`. So the library question is
  a physics-phase question and does not block the format.
- **Contacts between a parent body and its own child are off**; the commanded motion is refused by a
  discrete shape-cast before it is applied, and the joint reports `stalled` (base §7.16 invariant 3,
  validated). Between sibling bodies the same shape-cast runs.
- **★ The sibling test is an INDEX, never a scan.** Revision 1 owed a measurement and stated no
  algorithm. SL9 says a cost that grows with the set is a defect and that finding what holds a point is a
  lookup (`CLAUDE.md:214-225`). The rule: the realm keeps a broad-phase index over the bodies' SWEPT
  boxes in its own frame; only a body whose `q` changed this tick is re-inserted, and only that body
  queries the index. A hull with five hundred pistons at rest costs zero queries. A hull with five moving
  pistons costs five queries, each bounded by the boxes that overlap.
  *Example: a hull with five hundred pistons. Four move this tick. The shard re-inserts four boxes and
  asks four queries, not two hundred and fifty thousand pair tests.*
- **★ Inertia is maintained, never recomputed whole.** The realm holds a running mass and inertia sum. A
  body whose `q` changed subtracts its old contribution and adds its new one. The cost grows with the
  bodies that MOVED, never with the bodies that exist.
- **The bound is a promise about space.** A hull's `bound` is the slot it was sold and never changes
  (`built.rs:73-76`). A joint whose child's SWEEP would leave the bound is refused at placement. So a
  parent never learns that a turret turned, because a turret can never poke outside the box the parent
  already holds. The realm's LOOK may change on a placement, and the realm states it on change (SL3; the
  reach ruling's data table already carries "up, my own look, one hop, on change",
  `owner_decisions_2026-09-02_reach.md:126-145`).
- **A body's pose ships on the realm's OWN row.** See §4.7: this is an SL6 ask and it is what ties the
  body's stamp to the realm's.

### 4.5 A block placed on the turret finds its grid

1. The client aims at a face. The gateway routes the edit to the owning realm, the hull (the base's
   §3.9.7 carrier, D-39.1 owed). The edit target is the base's
   `WorldBlockRef { realm, chunk, cell, anchor_gen, expected }` (`block_system_design.md:4654-4658`)
   **plus TWO fields, `body` and `face`** — revision 1 announced one and wrote two.
   The `anchor_gen` field's fate is open: see §4.8.
2. The hull's shard validates against the SERVER's pose: reach from the player to the face, computed
   through the body's current pose (the face's world position is `pose(body) · cell_face_centre`).
3. The new cell is `neighbor(cell, face)` in the SAME body's grid — the body's own lattice. On a hull
   that lattice is flat; on a planet it is the realm's own index space (§4.8). The turret's grid never
   rotates in its own coordinates; the body's POSE rotates it. (Revision 1 wrote
   `GridMapping::Identity` here as if it were code. MEASURED: `grep -rn "GridMapping" crates
   --include='*.rs'` returns nothing; it is a name from `block_system_design.md:2662`.)
4. The client receives the chunk diff for `(realm, body, chunk)` and draws the chunk at `pose(body)`.

So "the turret's own sub-grid rotated by the joint" is literal: the grid is fixed to the body, and the
body carries a pose the realm authors. No block record, no chunk and no pyramid entry knows the angle.

*Example: a player welds a floodlight block on the turret's barrel while the turret is aimed left. The
edit names body 1, chunk (0,0,0), cell 4114, face −Z. The hull's shard finds the reach from the player's
pose to that face through body 1's pose, admits it, writes the cell into body 1's chunk, and ships the
diff. The gunner turns the turret right; the floodlight turns with it, because it is in body 1's grid.*

### 4.6 What a joint does to damage, debris and blueprints

- **A joint's anchor block breaks ⇒ the joint row goes, and the child body's fate is OPEN.** The
  cell-change hook (§2.6) removes the joint row when EITHER anchor block goes. What becomes of the
  detached body is **not settled**, and revision 1 settled it wrongly. It said the body becomes one
  `Debris` entity. MEASURED at `crates/core/src/entity_kind.rs:225-232`: `DEBRIS_DEF` is
  `class: Transient`, `ghost_policy: NeverTransferOnly`, `loss_budget: LossBudget(4)`,
  `max_state_bytes: 128`. A turret is a grid of player-placed blocks with its own chunks. It does not fit
  in 128 bytes, and a Transient kind with a loss budget of four may be dropped — so the cited
  registration refutes the claim it was cited for. The candidates are in §10 decision 8.
  **The foundation's answer is the safe one: REFUSE the removal.** While a joint holds a block, the edit
  that would remove that block is refused loudly, and the player removes the joint first. That is a rule
  a player can learn, it strands nothing, and it does not spend an entity kind before the owner rules.
  *Example: a player tries to mine out the plate his turret hangs on. The hull's shard refuses the edit
  and says the turret holds it. He removes the turret first, and then the plate.*
  Combat cannot refuse, so P11 needs the real answer, and §10 decision 8 asks for it.
- **A blueprint carries the body tree and every attachment** (R18: blueprints carry ALL wiring). The dock
  rebuilds bodies as it rebuilds blocks. The manifest lists every binding that reaches outside the
  construct (R18's rule); the foundation stores none.
- **Docking (a clamp between two HULLS) is not a joint.** Two hulls are two realms, and a constraint
  cannot be solved reliably across a scheduler that may separate them (§4.2). A docked hull becomes a
  CHILD of the station realm — the existing re-home machinery, no new data — and is carried as any child
  is. A "docked" relation the parent authors is a P8 design and is NOT asked here.

### 4.7 ★ ONE STAMP: a body's pose rides its realm's own row

A body's pose must be drawn in the same frame as the realm that carries it, or the player sees a seam.
Revision 1 shipped a body's pose *"like an occupant's pose"*, on the unreliable snapshot lane, drawn
through `INTERP_BUFFER_MS = 120.0` (`crates/wire/src/channels.rs:328`), while the realm's own drawn
placement is a different row with its own explicit stamp (`SceneRow`, `channels.rs:378-391`). Two rows on
two stamps, each interpolated on its own schedule, IS the seam.

*Example: a gunner stands on the turret platform while the hull rolls. The hull's row arrives for tick N
and the turret's body row for tick N−1. For one frame the platform slides under the gunner's boots and
his feet sink into it. That is SL8 seam kind "jump", and the next frame is seam kind "flicker".*

**The rule this revision states, so the measurement has a mechanism to prove:** a body's pose is part of
HOW ITS REALM LOOKS, so it rides the realm's OWN row, in the tagged skip-unknown look bag that row
already carries (`channels.rs:386-390`), under the stamp that row already carries per row
(`channels.rs:384-388`). SL3 supports it exactly: *"the parent authors WHERE a realm is; the realm itself
authors HOW IT LOOKS"*. The client then draws a realm and its moving parts from ONE interpolation sample,
or draws neither — by construction, not by care. This is an SL6 ask (§11, ask A), because it is new bytes
on a reviewed lane, even though the bag is additive.

The alternative — a separate `BodyPose` row on the occupant lane — is §10 decision 2 option A, and it is
NOT recommended, because it puts the one-stamp rule back into care.

### 4.8 ★ A joint on a SPHERICAL realm, and the anchor question

Revision 1 asserted *"Rails and pistons on a planet work identically… The joint code never reads
`voxel().geometry`."* That was asserted, not shown, and the code says the geometry seam is real:
`VoxelGeometry` is *"the ONE seam where spherical planets and Cartesian ship grids differ"*
(`crates/sim/src/capability.rs:34-35`), `Spherical` is *"tangent-anchored spherical projection with
re-anchoring"* and `Cartesian` is *"flat grid, no re-anchoring"* (`capability.rs:39-40`), geometry
differences are *"confined to `FrameSpace` impls selected by `voxel().geometry`"* (`capability.rs:11`),
and `FrameSpace` *"does not exist yet"* (`capability.rs:18`).

**The lawful formulation.** A joint's axis and a rail's path are defined in the parent BODY's INDEX
space, and the body's pose comes from mapping index-space points through the realm's OWN cell-centre map
— the same map the mesher and the collider read, which is the `FrameSpace` seam. So:

- On a hull the index space is flat, and a piston's prismatic axis is a straight line in the hull frame.
- On a moon the index space is the cube-sphere's, and the same prismatic axis is the cell column, which
  is radial. A rail's polyline through cell centres curves with the surface, exactly as the rail BLOCKS
  curve, because both read one map.
- The joint code therefore reads the realm's cell-centre map and **never branches on the geometry tag**.
  The anti-vacuity assertion of the HR4 gate (§8 item 6) is corrected to that weaker and true form:
  no `match` on `voxel().geometry` in joint code; the difference lives behind the seam.
- Status: **UNMEASURED and owed at P5**, because `FrameSpace` does not exist (`capability.rs:18`).

**The anchor generation.** The feasibility refuter asks for an `AnchorGen` inside `BlockAddr`.

> **DISPUTED: "put the anchor generation inside `BlockAddr`, or a re-anchor pass rewrites every stored
> address" (attach_feasibility F6).** — Evidence: no anchor state exists on the server for the grid.
> The position lattice is `i64` cells plus an `f64` offset (`crates/core/src/pose.rs:548-551`) with a
> power-of-two fine edge (`pose.rs:255`), so every cell anywhere on a body is exact and nothing needs
> re-centring for precision. Report 01 §5.4 reaches the same conclusion from the grid side and states it
> plainly: *"No anchor state exists on the server for the grid, and no `AnchorGen` rides an address."*
> `AnchorGen` exists nowhere in code (MEASURED: two doc comments only, `crates/core/src/fence.rs:18`,
> `crates/sim/src/capability.rs:40`). The tangent anchor's only surviving reason is a `f32` physics
> island, which is BELOW the grid and is the open physics-library question (O28 / D-19 / report 01 D7).
> **So a stored rail path and a stored joint anchor are lists of cells on an i64 lattice, and no
> re-anchor rewrites them under any of the library options.** I therefore do NOT put an anchor
> generation in the key.
>
> The refuter's underlying worry is real and this revision answers it a different way: the danger was a
> stored address silently meaning a different place. It cannot, because the lattice does not move. The
> `anchor_gen` field in the base's `WorldBlockRef` (`block_system_design.md:4654-4658`) is therefore
> **an open field, not a frozen one** — §10 decision 9 asks the owner to drop it with the physics-library
> answer rather than freeze it into the edit request now.

*Example: a player builds a mine cart on a moon and walks two kilometres away. Nothing re-anchors,
because the moon's cells are counted in whole i64 steps of 2⁻¹⁰ m from the moon's own origin. The cart's
twelve rail cells still name the same twelve places, and the HUD on the cart's side is still on the cart.*

---

## 5. Rails and pistons (question 4)

One mechanism, one closed enum:

| Kind | `Dof` | Scalar `q` | The child body's pose |
|---|---|---|---|
| rotation joint (turret, manipulator, hinge, door) | `Revolute { axis: the shared face's normal }` | angle on the integer angle grid | rotate about the anchor by `q`, through the carried sine/cosine table (§4.4) |
| piston | `Prismatic { axis: the face normal in index space }` | extension | translate along the axis by `q`, within `limits` |
| rail carriage | `RailPath { path: the ordered rail cells on the parent body's grid }` | distance along the path | translate along the polyline of rail-cell centres by `q`; the carriage's facing follows the path's segment |

- A piston is a rail whose path is one straight segment. A rail is a piston whose axis is a path. Both
  are a prismatic joint. The enum keeps them as two arms because a rail's PATH is derived from BLOCKS
  (the rail cells) and a piston's axis is derived from one FACE; the derivation differs, the integration
  does not.
- **The path is derived, not stored.** The attachment stores the carriage's anchor and `q`. The path is
  recomputed from the rail cells on the parent body's grid when a rail cell changes (the cell-change hook
  again). A carriage past a break stalls at the break. This keeps the rule *"prune what the blocks
  already say"* and makes the rail's shape a fact of the blocks, which a blueprint copies for free.
- **★ Finding the carriage from the cell is a LOOKUP.** Revision 1 recomputed a path on a rail edit and
  named no way to find which carriage cared, so the naive answer was a scan of every carriage in the
  realm on every rail edit — the cost SL9 calls a defect. The rule: the realm keeps a small index from
  CHUNK to the carriages whose path touches that chunk, written when a path is computed. A rail edit
  touches one chunk, so the lookup returns the few carriages in that chunk, and a realm with a thousand
  carriages elsewhere costs nothing.
  *Example: a player mines one rail cell out of the middle of a twelve-cell track. The planet's shard
  reads the chunk index, finds the one cart that rides that chunk, recomputes its path, and the cart
  stalls at the break. No other cart in the realm is touched.*
- A chained arm (the base's "telescoping arm: N chained prismatic segments driven by one port") is N
  bodies in the tree, driven by one signal at P9. Nothing new.
- **Rails and pistons on a planet** work through the realm's own cell-centre map (§4.8), UNMEASURED and
  owed at P5. HR4's gate: the same fixture (a piston lifts a crate) on the ship profile and on the planet
  profile, with the corrected anti-vacuity assertion (no `match` on `voxel().geometry` in joint code).

*Example: a mine cart. The player lays twelve rail blocks across the planet's surface and places a
`RailCarriage` attachment on the first rail's top face. The planet's shard makes body 7 with anchor on
that face and a path through the twelve cells. A lever at P9 drives `q`; the cart moves along the
polyline, which curves with the moon's surface because the cell centres do; the ore blocks the player
stacked on the cart are in body 7's grid and ride along.*

---

## 6. What this forces into the FIRST formats (question 5)

The owner freezes three formats together (V3.2): the grid family, the saved block record, the
edit-pyramid entry. This domain's exact demands on each:

### 6.1 The saved block record: NOTHING

An attachment is a side-table row keyed by the block's address. A joint is an attachment. A piston's
HOUSING is a plain block with an orientation the record already carries (the base's §7.18 item 4: five
orientation bits, no signal or port id in the record — validated). **The record gains no bit for this
domain.** Marked: not a door, and this report asks the record for nothing.

### 6.2 The block ADDRESS and the chunk STORE keyspace: a BODY ID — ★ ONE-WAY DOOR (store only)

Two bodies in one realm each have a chunk (0,0,0). An address without a body is ambiguous the day the
first joint exists. So:

```text
BlockAddr  { body: BodyId(u16), chunk: <tier-0 chunk key>, cell: <cell index>, sub: <V2.4 sub-cell> }
CoarseAddr { body: BodyId(u16), chunk: <chunk key at tier L>, cell: <cell index> }
```

(The chunk-key and cell-index types are report 01's and report 04's to name. MEASURED: `ChunkKey`,
`CellIndex` and `Tier0Key` exist in no crate.)

- `BodyId(0)` is the realm's own grid. A planet, a moon, an area and a hull with no joint only ever use 0.
  No code branches on it (HR3): a body id is a number in a key, as a tier is a number in a key.
- The realm's chunk STORE is namespaced by body: prefix `(body, chunk key)`. The base already moved the
  realm discriminator out of the key into the table namespace (`block_system_design.md` §2.7.1); the body
  goes the same way. **This is the door**: a saved keyspace cannot gain a field.
- The edit REQUEST (`WorldBlockRef`) carries `body` and `face` (§4.5).
- The PYRAMID's STORE key is `(body, chunk key at tier L)`. The entry itself (`cell | substance |
  occ_mask | fill | reserved`, `block_system_design.md:2156`) is unchanged; a turret's coarse cells are in
  the turret's lattice and the client places them at the body's pose. Nothing in an entry knows an angle.

**Why not six reserved bits of a chunk key?** Six bits give 64 bodies. A construction with 512 actuators
is the base's own budget; 64 would be a cap discovered in play. A separate `u16` costs two bytes per
address in the store's prefix.

**★ The WIRE half is NOT a door.** Revision 1 put the body id in a positional field of a new `BulkMsg`
arm and warned that "postcard is positional". MEASURED: `BulkMsg` has one arm today,
`Blob { kind: BulkKind, bytes: Vec<u8> }` (`crates/wire/src/channels.rs:283-289`), and the delta is the
opaque `bytes`. The body id rides as a TAG inside that TLV envelope (§2.5), where an older reader skips
what it does not know (`crates/core/src/tlv.rs:19`). Nobody re-versions a message. The door table (§7)
now carries the store half and the wire half separately.

**Deadline:** the store half, before the first world is saved (P4's format freeze).
**★ Dependency:** `BlockAddr` is frozen ONCE, with the body id AND report 04's `sub_scale`/`sub_addr`
fields together (`04_block_record_registry.md` §2.3; report 01 §3.2 fixes the width at 16 bits). Revision
1 ordered the body id planted first in §12 and left the sub-cell field to another report, which is a
contradiction: a positional address cannot gain a field after the keyspace is saved. §12 now orders the
freeze once, after report 04 answers.
**Cost if wrong:** every saved chunk keyspace re-versioned, or a second address type for bodies — an HR3
fork that spreads through the mesher, the collider, the edit path and the pyramid.

### 6.3 The construction's own store: a BODY family and an ATTACHMENT family — not a door

Two new prefix bytes in the realm's flat store, TLV-framed, tags append-only:

```text
BODY family        key (body id)                              value { parent body, joint anchors, dof, limits, fence }  durable
BODY_STATE         key (body id)                              value { q on the integer angle grid, at tick }            checkpoint-carried
ATTACHMENT family  key (body, chunk, cell, sub, face, slot)   value AttachmentRow (§2.3)                                durable
```

- Durable rows (a joint's existence, an attachment's existence) are written on placement, under the edit
  log's fence. The motor scalar changes per tick when driven, so it is checkpoint-carried like damage
  (the base's §2.4.2 durability split, validated): written on realm drain and on the checkpoint cadence.
  On a hard kill a turret returns to its last checkpointed angle. That is correct: an angle is
  recoverable; a placed joint is not.
- Not a door: the store's families and tags already grow additively (`built_store.rs:19-24, 79-81`). What
  IS fixed forever is the KEY shape of the attachment family (§6.4).

### 6.4 The attachment key: the block address type, a face byte, a slot byte — ★ ONE-WAY DOOR

`AttachmentKey = (BlockAddr, face: u8, slot: u8)`, face `0..=5` in the chunk's own basis, `6..=255`
refused, slot 0 today. Defined in terms of `BlockAddr`, so the body id and report 04's sub-cell index ride
inside it by construction. **Deadline:** with the address freeze (§6.2). **Cost if wrong:** a second
numbering that must be migrated when the block address changes, and every stored HUD on the wrong face if
the face basis is not the orientation's basis. **Why the slot byte is in the key from the first write:**
§2.2 — widening a KEY costs a version floor on every persisted attachment, and the skip-unknown rule does
not cover key bytes.

### 6.5 The attachment-kind registry and the blob's reserved tags — a persistence door (P6)

The kind numbering is dense, append-only, never reused, with `from_tag` erroring on unknown (the
entity-kind pattern). Tags `16..=31` (bindings) and `32..=47` (live state) are reserved in the row's
schema. **Deadline:** before the first attachment is persisted (P6). **Cost if wrong:** a schema bump and
a version-floor bump on every persisted attachment in the world (the base's door 31, validated).

### 6.6 The wire shape for a body's pose: the realm's own look bag (P8)

§4.7 puts a body's pose in the realm's own row's bag, under that row's stamp. The bag is tagged and
skip-unknown, so the TAG is additive; the ASK is that the bytes ride at all (§11 ask A). It becomes a
door the day a client negotiates it. Deadline P8, not the foundation.

### 6.7 The capability a body tree needs — ★ CORRECTED

Revision 1 said: *"A joint needs `integrates_children`, which every voxel realm profile already has."*
**Both halves are wrong.**

MEASURED: `asteroid()` at `crates/sim/src/capability.rs:276-282` builds a voxel realm — `voxel:
Some(VoxelGeometry::Cartesian)`, `block_edit: true` — and sets no `integrates_children`. It is a realm a
player mines and builds in (V2.1), and under revision 1's gate a player could not put a piston or a rail
on it.

MEASURED: `integrates_children` is read in feature code at exactly two places, both on the child-realm
drive lane: `crates/sim/src/stub/drive.rs:321` (`on_child_drive`) and `drive.rs:378`
(`on_child_facts`), reached through `crates/sim/src/stub/register.rs:341-343`. Its own documentation says
it gates whether *a shard does the physics for what is inside it* on that lane
(`capability.rs:71-91`). **A body is not a child realm** and speaks on no lane. So the capability revision
1 named governs a different thing.

**The corrected rule:** a body tree needs the realm to have blocks and to accept edits, and nothing more.
The honest options are in §10 decision 3:

- **(a) No new capability**: a joint is placeable wherever `block_edit` is, because a joint is a placed
  block relation and its integration is the realm's own arithmetic, not a lane.
- **(b) A derived capability `mechanisms`**, set in the one constructor's lattice
  (`capability.rs:122-135`, where `functional_blocks ⇒ signal_graph` is already derived) as
  `mechanisms = block_edit`. It costs one derived field and buys a named accessor for feature code, so a
  later refusal is data rather than a code change.

**Recommend (b).** It keeps every profile correct with no per-profile edit (an asteroid gets it from
`block_edit`), and it never spends `integrates_children`.

*Example: a miner lays twelve rail blocks across an asteroid and places a rail carriage. Under (b) the
asteroid's shard admits it, because the asteroid accepts block edits, and the cart runs — with no line of
asteroid-specific code anywhere.*

### 6.8 What is explicitly NOT forced

- No new `InterShardFlow` arm. Not because the arm list is full — MEASURED 44 arms, and the board's
  ceiling is stale (§1) — but because nothing in this domain crosses a realm boundary: an attachment
  rides the chunk diff to the CLIENT, and a body never leaves its realm.
- No change to `ChildDrive` or `ChildFacts`.
- No field in the block record, the palette entry or the pyramid entry.

---

## 7. One-way doors, consolidated

| # | Door | Deadline | Cost if wrong |
|---|---|---|---|
| 1a | **`BodyId` in `BlockAddr` and in the realm's chunk STORE prefix, the edit request and the pyramid store key** (§6.2) — frozen ONCE together with report 04's sub-cell field | the format freeze: before the first world is saved (P4), and **after report 04 answers V2.4** | every saved chunk keyspace re-versioned, or a second address type for bodies (an HR3 fork through mesher, collider, edit path, pyramid) |
| 1b | ~~the `body` field in the chunk-delta MESSAGE~~ — **NOT a door**: it is a tag inside the delta's TLV envelope, and an older reader skips it (`tlv.rs:19`) | — | — |
| 2 | **The attachment key is `(BlockAddr, face u8, slot u8)`, defined by the block's address type** (§6.4) | the same freeze | a parallel numbering that drifts from the block's on the first address change; widening a key byte later costs a version floor on every persisted attachment |
| 3 | **The face numbering is the chunk's own basis, the basis block orientation uses** (§2.2) | with the orientation encoding freeze (base §2.7.1) | every stored attachment on the wrong face after a basis change |
| 4 | **The attachment-kind registry numbering and the reserved tag ranges 16..=31 / 32..=47** (§6.5) | before the first attachment is persisted (P6) | a schema bump and a version-floor bump on every persisted attachment |
| 5 | **A joint child is a BODY in the realm, never a REALM** (§4) | before P8 builds the first joint; door 1a already commits the store to it | two machineries for one thing; a body-realm needs a shard, keys, reach and a saga per crossing; retrofitting either way rewrites the store and the crossing |
| 6 | **The scoped channel key `H(scope ‖ name)`** (R19, decision-board door 49) | before the first binding is persisted (P9 first slice; plant the type in the foundation) | every saved binding re-resolves to a different channel, blueprints included |
| 7 | **The integer ANGLE GRID a joint's `q` counts on, and the carried sine/cosine table keyed by it** (§4.4) | with the body-state family, before the first joint angle is checkpointed (P8) | a re-valued grid changes every stored angle in the world, and a table swap changes every body pose — a flag day of the same class as the fine-cell edge (`crates/core/src/pose.rs:250-255`) |
| 8 | **A body's pose rides its realm's own row under that row's stamp** (§4.7, §6.6) | before a client negotiates it (P8) | a protocol-minor bump and a client migration; and a body drawn on a second stamp is an SL8 seam |

Not doors: the `Dof` enum's arms (appending a postcard enum variant is additive); the widget catalogue
rows; every budget number; kinematic-versus-solved integration (the record is the same under both); the
`mechanisms` capability (a derived bool in one constructor).

---

## 8. Measurements owed (never argue these)

1. **A hull with N bodies, N ∈ {1, 8, 64, 512}, with M of them MOVING, M ∈ {0, 4, N}:** the hull shard's
   tick time for the kinematic pose derivation, the incremental inertia update, and the swept-box index
   queries; and the bytes per tick of body rows to one observer holding the hull. On THE world, on the
   process tier (`cargo test --release`, the standing rule). ESTIMATED sub-microsecond for the poses;
   UNMEASURED.
2. **The attachment table at the base's cap (1,024 per realm):** store bytes, chunk-diff bytes for a
   chunk carrying 200 attachments, and the client's apply time. UNMEASURED; the cap itself is the base's
   number and must be re-derived.
3. **★ A gunner keeps contact with a turning platform.** No collider, no shape cast and no character
   controller exists today (§1), so this is the machinery P5 owes, not a consequence of a body tree.
   MEASURED on the process tier: a gunner stands on a turret platform turning at its rated speed, drawn
   at the interpolation buffer (`INTERP_BUFFER_MS = 120.0`, `channels.rs:328`), with the pop detector on
   his feet against the platform — zero jitter frames, zero separation. Flown in the window, release
   build. This is the SL8 seam test, and §4.7's one-stamp rule is the mechanism it proves.
4. **Click-to-pixels through a body:** an edit on a rotating turret round trip, against the P6 edit
   budget.
5. **The HUD value lane at the base's density (128 sessions, 500 covers):** bytes per client per second
   under §3.3 option A (ESTIMATED 22 B per changed binding) AND option A2 (ESTIMATED 7 B with a
   per-observer handle), against the per-client byte budget; and the compose cost on the shard.
6. **G-IDENTICAL:** the piston-lifts-a-crate fixture on the ship and the planet profile, byte-identical
   body state, with the corrected anti-vacuity assertion — no `match` on `voxel().geometry` in joint code
   (§4.8). Owed at P5 with `FrameSpace` (`capability.rs:18`).
7. **★ Cross-target joint determinism.** The same joint at the same `q` produces a byte-identical body
   pose on x86-64 and on aarch64, through the carried sine/cosine table (§4.4). This is SL10 clause 3's
   own gate shape applied to a joint. Red on one differing byte.
8. **★ The client's frame budget for a body tree.** Frame time and draw-batch count for a hull with N
   bodies, N ∈ {1, 8, 64, 512}, drawn in the window at the release build; and the thread a body's chunk
   remesh runs on. A construction with many bodies is many separate chunk-mesh transforms, and no
   revision of this report has costed the client side.

---

## 9. Claims in the investigation base this domain found stale

| Base claim | Where | What supersedes it |
|---|---|---|
| "The 30-arm ceiling on `InterShardFlow` is reached exactly. 27 arms + 3 reserved names = 30. There is no headroom left" | `decision_board.md:1172` | MEASURED: 44 arms today (`crates/wire/src/intershard.rs:124`). The board is stale by seventeen arms and must not be relayed as a constraint |
| "Panel transforms are server-authored root-absolute like every other pose (the A5 flip); the client only subtracts the server-told render origin" | §6.9, the instanced pass | SL1 as rewritten 2026-08-24: no realm's absolute exists anywhere a realm can reach it; a panel sits on a face in its realm's frame and the client places it through the composed row (`SceneRow.pose`, `channels.rs:378-391`) |
| "A cover is recorded in the parent block's instance TLV config" AND a `block_cover (u32 local, u8 face, u32 cover_ref)` side table | §6.9 vs §2.7.1 (two homes) | V2.5: an attachment sits on ANY block, including a plain plate with no config, so it is its own row keyed by the block address plus a face plus a slot (§2), never a field of a functional block's config; the `cover_ref` indirection is dropped |
| "An articulated ship is ONE transferable entity; actuator states in its 8192-byte state blob; cap 512 actuators, derived from the blob arithmetic" | §7.16.4 and decision-board P8.3 | 2026-09-01 rulings + the crossing plan: every hull is a REALM whose store moves with it and whose interior is *"untouched"* (`transfer_protocol.md:376`, `realm_crossing_plan_2026-09-02.md:61`); *"a ship built by a player is not a special case"* (visibility ruling V4). The blob binds nothing; the cap is a per-realm budget to be measured (§8 item 1) |
| "One geometry per cell (no sub-grid)" as a YES one-way door; "the base cell is never subdividable" | §4.1.5, `decision_board.md:134-138` | V2.4 and report 04 §2.3: a sparse sub-site on the same record. This report's key must therefore carry the sub-cell field (§6.2) |
| The rendering of the cover assumes Bevy (`AlphaMask3d`, the prepass, `bevy_text` vs MSDF) | §6.9 | V2.8: the render seam is engine-agnostic; those are an engine's choices behind the seam (§3.2 item 5) |
| "Rigid-body mass properties via `RigidBodyBuilder::additional_mass_properties`; rapier3d 0.34 defaults" | §4.6.2, §7.16.2 | MEASURED: no rapier or parry dependency in any `Cargo.toml` (one comment, `Cargo.toml:96`); D-19 🟥 (`DEFERRED.md:6240`); O28 / `[4-B]` open. Kinematic joints need no library; a solver is an option (§4.4) |
| Door 37: "articulated machines transfer as one ENTITY or as independent sub-assemblies — before P6" | §8.2 row 37, decision board §3 | The answer is "one REALM, bodies inside it"; the door stands, its subject changed (§7 door 5) |
| "A `DisplayBlock`… no new capability is needed: `surfaces` already exists" | §6.9 | Half true. `surfaces` exists and requires a voxel realm (`capability.rs:52, 122-135`). But a joint's gate is NOT `integrates_children` (§6.7), and an asteroid has neither `surfaces` nor `integrates_children` (`capability.rs:276-282`) |
| D-DEC-4: "per-subscriber panel addressing depends on D-9's AoI reshape" | §6.11 | D-9 is 🟩 since 2026-09-05 (`DEFERRED.md:2925`); the per-cube named-recipient body exists (`interest.rs:1-29`), so the dependency is discharged, not owed |
| "13 B per changed binding" for the HUD value lane | §7.9, `block_system_design.md:14199` | Correct for the base's `(panel_id u32, slot u8, value Fx)` key; NOT for this report's block-address key. ESTIMATED 22 B here (§3.3) |
| The hull collider "hands your ship's shape including internal voids to whoever hosts you" | decision board O28 | Unchanged as a risk, but note: the parent holds only the bound and the look as readings (`built.rs:99-103`); a body tree adds nothing the parent sees |

---

## 10. Open decisions for the owner, with a recommended answer

1. **HUD content lane: value-level, value-level with a handle, or draw-list?** Options: A, the server
   ships `(full key, value, state)` (ESTIMATED 22 B per change); A2, the server hands each observer a
   short handle and then ships `(handle, value, state)` (ESTIMATED 7 B, one small per-observer table);
   B, the server composes a draw-command list (the base measures 150–360 B per widget,
   `block_system_design.md:14197`). **Recommend A for the foundation's SHAPE, and measure A against A2
   at P9** (§8 item 5) — the owner's own words ("a widget from a predefined set"), and B stays available
   as one more widget kind on the same seam.
2. **A body's row identity on the wire:** A, a new `BodyPose { realm, body, pose }` row on the occupant
   lane; B, bodies get an `EntityId` of a new non-transferable `EntityKind::Body`; **C, the body's pose
   rides its realm's OWN row in the look bag, under that row's stamp.** **Recommend C** — it makes the
   one-stamp rule structural (§4.7) instead of remembered, it matches SL3 (a realm authors how it looks),
   and a body is not a transferable kind and must not look like one to the transfer registry.
3. **Which capability gates a body tree?** (a) none, `block_edit` is enough; (b) a derived `mechanisms`
   set from `block_edit` in the one constructor. **Recommend (b)** (§6.7). Revision 1's answer
   (`integrates_children`) is withdrawn: it is the child-realm drive lane's switch
   (`drive.rs:321, 378`) and an asteroid does not have it (`capability.rs:276-282`).
4. **Kinematic-analytic joints first, or adopt a rigid-body library now?** **Recommend analytic first**,
   WITH the carried sine/cosine table and its cross-target gate (§4.4, §8 item 7) — without the table
   the "deterministic" half of this recommendation is false. The library question stays O28 / `[4-B]` and
   is not answered here.
5. **Per-realm budgets for bodies and attachments:** the base's 512 and 1,024 are not derived from
   anything that still holds. **Recommend: measure (§8 items 1–2, 8), then set them as fields in the one
   tuning struct**, never as literals.
6. **Does a face of a sub-metre block (V2.4) carry an attachment?** **Recommend yes by construction** —
   the key is the block's address, whatever index it carries. `BlockAddr` is frozen ONCE with report 04's
   sub-cell field and this report's body id together (§6.2, door 1a).
7. **Attachments per face: one, or several?** **Recommend the KEY carries a slot byte from the first
   write, with one attachment allowed today.** Revision 1 recommended one per face and called a later
   widening an append; that was wrong (§2.2), and the widening costs a version floor on every persisted
   attachment. One byte in the key now buys the door.
8. **★ What becomes of a body whose anchor block is destroyed?** Options: (a) a new Durable entity kind
   — the entity-kind file itself calls a new kind *"first-class engineering and a crash-matrix
   multiplication"*, and 8192 bytes will not hold a grid either (`entity_kind.rs:207-222`); (b) the
   detached body becomes its own small BUILT REALM through the machinery a player-built hull already
   uses (`built.rs:61-83`), at the cost of a shard in the port band; (c) refuse the removal while a joint
   holds the block. **Recommend (c) for the foundation** (it strands nothing and spends nothing) **and
   (b) for P8/P11**, because combat cannot refuse. Revision 1 said `Debris`, and the registry refutes it:
   `DEBRIS_DEF` is Transient, 128 bytes, loss budget 4 (`entity_kind.rs:225-232`), so a player's built
   turret could be dropped.
9. **Does the edit request keep an `anchor_gen` field?** The base carries one
   (`block_system_design.md:4654-4658`). No anchor state exists on the server for the grid, and report 01
   §5.4 says none should. **Recommend: leave the field OUT of the frozen edit request and decide it with
   the physics-library answer** (O28 / D-19 / report 01 D7), because a `f32` physics island's anchor is
   below the grid and never rides a stored address (§4.8).
10. **Who may attach to a block on a realm they do not own?** The realm's owner today
    (`BuiltBody.owner`, `built.rs:68-70`). The row stores `placed_by`. The ACL is R15's domain at P9.
    **Recommend: owner-only until the ACL lands**, refused loudly.

---

## 11. Law conflicts and SL6 asks (question 6)

**Nothing new crosses a REALM-TO-REALM boundary in this design.** But SL6 has two halves, and revision 1
answered only the first. The second reads *"and before adding a wire arm. Default NO"*
(`CLAUDE.md:210-213`), and the owner reviews the bytes on a reviewed lane, not only the arm list. Revision
1 recommended two new rows and reported no conflict. **Three asks follow. None is added by the
foundation; each is owed at the phase named.**

### Ask A — a body's pose, on the realm's own row (P8)

- **The data:** for each body `b` of a realm, its pose in the realm's own frame, as a tagged entry in the
  look bag of the realm's own `SceneRow` (`crates/wire/src/channels.rs:378-391`).
- **From / to:** the owning realm's shard → the gateway → the observers that already receive that realm's
  row. Not a realm-to-realm crossing (SL2 as clarified 2026-08-24: the connection plane is not a realm).
- **Why the receiver cannot compute it:** the client only renders, and a body's angle is live state driven
  by a motor. The gateway holds no joint and no `q`.
- **The cost of doing without:** a turret is drawn welded shut. Every V2.5 mechanism — a turret, a door, a
  piston, a rail carriage — is unusable.
- **Why this shape and not a separate row:** the realm's row carries its stamp explicitly per row
  (`channels.rs:384-388`), so a body drawn from that row cannot be one frame out of step with the realm
  that carries it. A separate row on the occupant lane can, and that is an SL8 seam (§4.7).
- **Bytes:** ESTIMATED one pose per moving body per tick, to the observers that already hold the realm's
  row; §8 item 1 measures it.

### Ask B — a HUD's value level (P9)

- **The data:** `WidgetLevel { key: AttachmentKey, value: Fx, state: Fresh|Stale|Lost }` on the unreliable
  latest-wins snapshot lane, addressed to the observers that hold the attachment's chunk.
- **From / to:** the owning realm's shard → the gateway → the observers holding that chunk. The realm
  reads its OWN channel; no boundary is crossed. (A station HUD showing another SHIP's fuel is the P9
  `Signal` arm, reserved (`intershard.rs:34-35`), and is NOT asked here.)
- **Why the receiver cannot compute it:** the value is live state from a signal graph the client does not
  hold and must never hold.
- **The cost of doing without:** a HUD is a picture that never changes, which is a placeholder, which the
  no-placeholder rule refuses — so without this ask the foundation must draw nothing (§3.2 item 5, which
  is exactly what this revision does).
- **Bytes:** ESTIMATED 22 B per changed binding, or 7 B with a per-observer handle (§3.3). §8 item 5
  measures it.

### Ask C — the attachment tag inside the chunk diff (P6)

- **The data:** tag 5 of the chunk-delta TLV envelope — a sorted list of
  `(cell, face, slot, AttachmentRow bytes)` — plus tag 6, the body id.
- **From / to:** the owning realm → the client, one hop, inside the existing
  `BulkMsg::Blob { kind: ChunkDelta }` arm (`channels.rs:283-296`).
- **Why the receiver cannot compute it:** an attachment is placed by a player. SL10 clause 6 names
  attachments by word as diff data (`owner_decisions_2026-09-07_voxels.md` V1.6), so the DIRECTION is
  already approved; the ask is that these bytes ride at all.
- **The cost of doing without:** a placed HUD, hinge, piston and rail are invisible to every client.
- **No new arm and no new lane.** An older client skips the tag (`tlv.rs:19`).

### The candidates that are NOT asks

| Candidate | Direction | Verdict |
|---|---|---|
| a turret's angle, a piston's extension | hull → star system | **NO.** The hull sums its bodies into the six numbers and the mass it already states (`intershard.rs:1377-1397, 1404-1426`). The parent holds the bound the hull was sold and needs nothing more (§4.2) |
| a joint between two HULLS (docking) | hull ↔ hull | **Not a joint.** A constraint solved across two realms works only while the scheduler co-hosts them (§4.2). The local formulation: the docked hull becomes a child of the station realm through the existing re-home; a "docked" relation the parent authors is a P8 design. Not asked |
| the turret's grid contents to the parent (for its collider) | hull → parent | **NO.** The parent holds the bound and look as readings (`built.rs:101-103`); the hull collider question is the decision board's O28 and is unchanged by bodies |
| a realm's LOOK, on change, after a placement | realm → parent | **Already approved** (`owner_decisions_2026-09-02_reach.md:126-145`, "up, my own look, one hop, on change") |

---

## 12. What the foundation plants for this domain, in order

1. **Freeze `BlockAddr` ONCE** — `BodyId(u16)` together with report 04's `sub_scale`/`sub_addr` fields —
   in the realm's chunk store prefix, in the edit request and in the pyramid store key. Body 0 and
   sub-scale 0 everywhere today. **This step waits for report 04's V2.4 answer.** (Door 1a.)
2. `AttachmentKey = (BlockAddr, face u8, slot u8)`, face in the chunk's basis, `6..=255` refused, slot 0
   today. (Doors 2, 3.)
3. The attachment family in the realm store: the closed kind registry with `Display` and `Joint` rows,
   the row schema with reserved tag ranges. The `Display` params carry the widget, the style, the range,
   the unit and the label. (Door 4, §3.2 item 1.)
4. The body family and the body-state family in the realm store; body 0 implicit; `Dof` closed with
   three arms; `q` on the integer angle grid. (Door 7's grid.)
5. The carried sine/cosine table keyed by the angle grid, in the crate both hosts compile, with the
   cross-target byte-identity gate (§4.4, §8 item 7). (Door 7's table.)
6. The `on_cell_changed` hook on the edit path, with the attachment and body families registered into it,
   covering the empty case, the replace-in-place case and both anchors of a joint (§2.6).
7. The chunk-to-carriage index, so a rail edit is a lookup (§5).
8. The swept-box index over bodies, and the incremental inertia sum (§4.4).
9. The derived `mechanisms` capability in the one constructor, set from `block_edit` (§6.7).
10. `ChannelKey` (scope + name hash) in `vd-core`, with the loud collision check; no bindings stored.
    (Door 6.)
11. The engine-free face-overlay primitive beside `MeshPrim` in the client library, **and its test. It
    draws nothing until the value lane lands at P9** (§3.2 item 5).
12. Tag 5 (attachments) and tag 6 (body id) named in the chunk-delta TLV envelope's tag list.
13. The `WidgetKind` registry with one row, `Numeric`.

Everything else — the motor, the shape-cast, the value lane, the bindings, the detached-body carrier,
docking — lands with its phase (P5 physics, P6 edits, P8 ships, P9 signals, P11 combat) on top of these
records without a migration.

---

## 13. Revision log

Findings from `verdicts/attach_law.md` (L) and `verdicts/attach_feasibility.md` (E).

| Finding | Verdict | What this revision did |
|---|---|---|
| L-F1 | BREAKS_LAW — two new wire rows, "law conflicts: NONE" | **Fixed.** §11 is rewritten as three explicit SL6 asks (A a body's pose, B the HUD value level, C the attachment tag), each with the data, the sender, the receiver, why the receiver cannot compute it, and the cost of doing without. §0 item 8 no longer says "NONE" |
| L-F2 | WRONG — the 30-arm ceiling | **Fixed.** MEASURED 44 arms (`intershard.rs:124`) with the command shown in §1. Every ceiling argument is deleted; §6.8 argues "no new arm" from the local formulation. §9 records the board's stale claim |
| L-F3 / E-F2 | WRONG — `integrates_children` on every voxel profile | **Fixed.** §6.7 is rewritten: the asteroid profile has neither (`capability.rs:276-282`), and `integrates_children` gates the child-realm drive lane (`drive.rs:321, 378`). §10 decision 3 recommends a derived `mechanisms` from `block_edit` |
| L-F4 | WRONG — a spare face value is an append | **Fixed.** §2.2 admits the version-floor cost and adds a `slot` byte to the KEY from the first write. §10 decision 7 restates the recommendation |
| L-F5 | WRONG — `Debris` cannot carry a detached body | **Fixed.** §4.6 no longer settles it. §10 decision 8 opens it with three candidates; the foundation refuses the removal instead |
| L-F6 / E-F3 | WRONG — the chunk-delta message named two ways | **Fixed.** §2.5 and §6.2 both say: the body id and the attachments are TAGS inside the delta's TLV envelope. The "postcard is positional" warning is deleted. Door 1 is split into 1a (store, a door) and 1b (wire, not a door) |
| L-F7 | UNMEASURED_AS_FACT — a rail on a planet | **Fixed.** New §4.8 states the lawful formulation (index space through the realm's own cell-centre map), corrects the anti-vacuity assertion to "no `match` on `voxel().geometry`", and marks it UNMEASURED and owed at P5 (`capability.rs:18`) |
| L-F8 | MISSING — the HUD value has no range or unit | **Fixed.** §3.2 item 1 puts `range_lo`, `range_hi`, `unit` and `label` in the `Display` params, stated once on placement |
| L-F9 | MISSING — the 512-body cost shape | **Fixed.** §4.4 states the swept-box index (only a body whose `q` changed is re-inserted and queries) and the incremental inertia sum. §12 items 8 plants both |
| L-F10 | MISSING — SL10 cited for widget art | **Fixed.** §3.1 now cites V2.6 and says plainly why SL10 does not reach a dial |
| L-F11 / E lesser | WRONG — citation drift | **Fixed in part; three refuter corrections DISPUTED.** Re-read in this worktree and corrected: `BuiltBody` `built.rs:61-83`, `BuiltFacts` `built.rs:37-53`, `SHIP_DEF` `entity_kind.rs:207-214`, the capability data `capability.rs:47-91` (line 100 is inside `ProfileError`, so the voxel checks are cited as `capability.rs:122-135`), `transfer_protocol.md:376` quoted as *"untouched"*, `realm_crossing_plan_2026-09-02.md:61`, and `realm_head.rs:47-48` (co-hosting) split from `50-58` (the exterior lease only). See the three DISPUTED lines below |
| L-F12 | MISSING — the foundation draws an empty dial | **Fixed.** §3.2 item 5 and §12 item 11: plant the primitive and its test, draw nothing until P9 |
| E-F1 | WRONG — "no transcendental, therefore bit-deterministic" | **Fixed.** §4.4 states that a revolute joint needs sine and cosine, cites SL10 clause 4 and `celestial.rs:23-25`, and requires a carried integer table keyed by the angle grid. New door 7 and new measurement §8 item 7 |
| E-F4 | 13 B carried from a narrower key | **Fixed.** §3.3 re-derives ESTIMATED 22 B from this report's own key with the arithmetic shown, restates the base's rates as understated by about seventy percent, and adds option A2 (a per-observer handle, ESTIMATED 7 B) |
| E-F5 | the §1 grep cannot produce its output | **Fixed.** §1 pastes both commands and their real output |
| E-F6 | anchor generation missing from a frozen key | **DISPUTED and answered.** §4.8 keeps the anchor generation OUT of `BlockAddr`, with evidence: `pose.rs:548-551` (i64 cells, f64 offset), `pose.rs:255` (power-of-two fine edge), report 01 §5.4 ("no `AnchorGen` rides an address"), and MEASURED absence of `AnchorGen` in code. §10 decision 9 asks the owner to drop the base's `anchor_gen` field with the physics-library answer |
| E-F7 | the body pose and the realm pose share no stamp (SL8) | **Fixed.** New §4.7 states the mechanism: a body's pose rides the realm's OWN row's look bag under that row's stamp. §10 decision 2 gains option C and recommends it. §12 item 1 of the wire list, §8 item 3 keeps the measurement |
| E-F8 | the moving-floor claim is unmeasured | **Fixed.** §1 records that no collider, shape cast or character controller exists. §4.2 marks the sentence as design intent. §8 item 3 owes the measurement at P5 |
| E-F9 | door 1 cannot close before the V2.4 answer | **Fixed.** Door 1a carries the dependency, §6.2 states it, and §12 item 1 orders one freeze after report 04 answers |
| E-F10 | three unanswered cases | **Fixed.** §2.6 answers all three: the child's anchor break (the hook runs on both anchors), the replace-in-place case (re-validate against the new kind's registry row), and the rail reverse index (a chunk-to-carriage index, §5) |
| E-F11 | no client frame budget | **Fixed.** §8 item 8 |
| E lesser: `WorldBlockRef` gains two fields | accepted | §4.5 item 1 says two, `body` and `face` |
| E lesser: `GridMapping`, `ChunkKey`, `CellIndex`, `Tier0Key` written as if code | accepted | §1 records the MEASURED absence; §4.5 and §6.2 name them as base-doc names or as other reports' types |
| E lesser: "(B) cannot solve a constraint at all" over-stated | accepted | §4.2 states the honest form with `CoHostedAuthority` (`realm_head.rs:47-48`, `topology.rs:305-307`): under (B) the solve fails only sometimes, which is worse |

**Three refuter corrections this revision does NOT accept.**

- **DISPUTED: "`capability.rs:52` is `seats`, not `surfaces`" (attach_law F11).** — MEASURED,
  `grep -n "pub surfaces\|pub seats" crates/sim/src/capability.rs` prints `52:    pub surfaces: bool,`
  and `53:    pub seats: bool,`. Revision 1's citation was right and the correction is off by one. This
  revision keeps `capability.rs:52` for `surfaces`.
- **DISPUTED: "`FrameSpace` is named at 12 and 19, not 11 and 18" (attach_law F11).** — MEASURED,
  `grep -n "FrameSpace" crates/sim/src/capability.rs` prints
  `11://! are confined to \`FrameSpace\` impls selected by \`voxel().geometry\` (P4/P5).` and
  `18://! forcing a \`reanchor()\` stays owed at P5 (\`FrameSpace\` does not exist yet).` Revision 1's
  citation was right. This revision keeps 11 and 18.
- **DISPUTED: "`DEBRIS_DEF` is 226-233, not 225-232" (attach_feasibility, lesser defects).** — MEASURED,
  `grep -n "DEBRIS_DEF" crates/core/src/entity_kind.rs` prints
  `225:pub static DEBRIS_DEF: KindDef = KindDef {`, and the item closes at 232. Revision 1's range was
  right. This revision keeps `entity_kind.rs:225-232`.

*Example of why this matters: a later reader chasing the asteroid case opens `capability.rs` at the
cited line. If the line is wrong by one, the reader lands on `seats` and concludes a joint needs a seat.
A citation is a measurement like any other.*

