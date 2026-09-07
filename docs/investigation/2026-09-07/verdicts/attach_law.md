# Verdict — the law refutation of `05_attachments_bodies.md`

**Lens:** attachments and mechanical bodies.
**Subject:** `docs/investigation/2026-09-07/05_attachments_bodies.md`.
**Result: REFUTED.** The report carries load-bearing claims that are wrong, and it declares
"Law conflicts: NONE" while it proposes two new wire rows.

**Method.** I read every binding ruling first. I then read the report. I then checked every
`file:line` in the report against the code with `grep` and `sed`. I built nothing and I ran no test.
Every number below is MEASURED with the command named beside it.

---

## 1. What survives the refutation

These claims are load-bearing, and they stand.

- **A joint's child is a BODY inside the containing realm, never a child REALM (§4).** HR1 makes a
  shard's physics world private, so two bodies of one constraint must sit in one shard. SL4 gives the
  containing realm the physics of what it holds. The realm-hood rule ("you are a realm when something
  can be inside you") refuses a hinge a realm. **STANDS.**
  *Example: a turret on a hull is body 1 of the hull realm. The star system still holds one child.*
- **Nothing new crosses the hull-to-star-system boundary (§4.2).** MEASURED: `ChildDrive` carries a
  child coord, a fence, a tick, a push and a turn, and nothing else
  (`crates/wire/src/intershard.rs:1377-1400`). `ChildFacts` carries mass, cross-section, drag and the
  closed declared states (`intershard.rs:1404-1428`, `intershard.rs:1437-1442`). A turret angle fits
  in neither, and the report does not try to put it there. **STANDS.**
- **No attachment and no body exists in the tree today (§1).** MEASURED:
  `grep -rni "attachment" crates --include='*.rs'` returns exactly one line, a comment word in
  `crates/wire/src/session_flow.rs:39`. `grep -rn "SubBody\|BodyPart" crates --include='*.rs'` returns
  nothing. **STANDS.** (The report prints the grep with a capital `A`, which matches nothing. The
  claim is true; the command beside it is not the command that proves it.)
- **Rapier is not adopted (§1).** MEASURED: `grep -rn "rapier\|parry" Cargo.toml crates/*/Cargo.toml`
  returns one comment, `Cargo.toml:96`. **STANDS.**
- **A realm may state its own look upward, on change (§4.4).** The reach ruling's data table records
  "up | my own look, one hop | on change | exists"
  (`docs/design/owner_decisions_2026-09-02_reach.md:126-145`). **STANDS.**
- **The key is the block's address plus a face, so a sub-metre block (V2.4) carries an attachment for
  free (§2.2, §6.4).** This keeps V2.4's door open. **STANDS.**

---

## 2. The findings that refute the report

### F1 — BREAKS_LAW. Two new wire rows are proposed, and the report reports no conflict.

SL6 has two halves. The report answers the first ("new data across a realm boundary") and never
answers the second: *"and before adding a wire arm. Default NO."* (`CLAUDE.md:210-213`).

The report proposes two rows that do not exist today:

- `WidgetLevel { key, value, state, at }` on the unreliable snapshot lane (§3.2 item 4).
- `BodyPose { realm, body, pose }` on the occupant lane (§6.6, §10 decision 2, recommended option A).

Neither appears under law conflicts. §0 item 7 says *"Law conflicts: NONE that needs a new crossing"*
and §11 says *"No wire arm is added"*. A row is not an arm, but SL6's default NO covers the wire, not
only the arm list, and the report itself calls the body row *"a client-negotiated wire shape and
becomes a door"* (§6.6). A door the owner has not been asked about is a conflict.

*Example: the hull's shard wants to ship the turret's angle to the gateway as a `BodyPose` row beside
its occupants' rows. That row is new bytes on a reviewed lane. The owner must say yes to it, exactly
as the owner said yes to a child stating its reach.*

**Fix.** Move both rows into a `law_conflicts` section as explicit SL6 asks: the data, the sender, the
receiver, why the receiver cannot compute it, and the cost of doing without.

### F2 — WRONG. The 30-arm ceiling is not reached; the code passed it long ago.

The report states, inside the section headed *"What exists today — the code is the only truth"*:
*"The decision board records that the 30-arm ceiling is reached exactly"* (§1). §6.7 leans on it to
justify *"No new `InterShardFlow` arm"*.

MEASURED, with
`awk '/^pub enum InterShardFlow/,/^}/' crates/wire/src/intershard.rs | grep -cE "^    [A-Z]"`:
**44 arms** in `crates/wire/src/intershard.rs:124`. The decision board's own text says
*"27 arms + 3 reserved names = 30"* (`docs/investigation/decision_board.md:1172`). The board is stale
by seventeen arms. The report relayed a document where the standing rule says to read the code.

The conclusion (add no arm) may still be right. The reason given for it is false.

*Example: `ReachStated` is one of the 44. It landed after the board wrote its ceiling.*

**Fix.** Delete the ceiling argument. Argue "no new arm" from the local formulation instead: an
attachment rides the chunk diff, and a body never leaves its realm.

### F3 — WRONG. Not every voxel realm profile carries `integrates_children`.

§6.7 states: *"A joint needs `integrates_children`, which every voxel realm profile already has
(`capability.rs:244-300`)."*

MEASURED, inside the cited range: `asteroid()` at `crates/sim/src/capability.rs:276-282` builds
`CapRequest { voxel: Some(VoxelGeometry::Cartesian), block_edit: true, ..default() }`. It is a voxel
realm. It has no `integrates_children`. The cited lines refute the claim they are cited for.

The consequence is real. V2.1 says a player mines the world. An asteroid is a realm a player mines.
Under the report's own gate a player could not put a piston or a rail on an asteroid, and the flat
conclusion *"No new capability"* fails.

*Example: a miner lays twelve rail blocks across an asteroid and places a rail carriage. The
asteroid's shard refuses it, because the asteroid profile does not integrate children — a switch that
governs a child REALM speaking up a wire lane, not a body turning inside one realm.*

**Fix.** State which capability gates a body tree. Then either add it to every voxel profile as data,
or show that a body tree needs no capability at all. `integrates_children` is documented as the switch
for *"the drives of what it holds"* on the wire (`capability.rs:71-88`), which a body is not.

### F4 — WRONG. Taking a spare face value is a migration, not an append.

§2.2 says two things that cannot both be true. It says values `6..=255` *"are REFUSED on decode, never
decoded to a default"*. It then says *"The 250 unused values of the face byte leave room if the owner
later wants a second slot per face; taking one of them is an append, not a migration."*

A refusing reader hard-errors on face 6. The skip-unknown rule that makes appends free covers TAGS,
not the bytes of a key: *"A reader consumes the tags it knows and SKIPS unknown tags by `len`"*
(`crates/core/src/tlv.rs:20`), and the version floor REFUSES a blob it cannot represent
(`tlv.rs:21-23`). A stored row whose key holds face 6 is unreadable to every earlier reader, so
widening the face byte costs a version-floor bump on every persisted attachment.

The cost lands on V2.5. The owner said a sub-block goes on ANY squared voxel. One attachment per face
is therefore a firmer door than the report admits.

*Example: a player wants a fuel gauge AND a hinge on the same hull plate face. The design refuses the
second. Lifting the refusal later migrates every stored HUD in the world.*

**Fix.** Either say plainly that one-per-face is a one-way door with a version-floor cost, or make the
face a tagged field in the row's blob, where an append really is free.

### F5 — WRONG. The registered `Debris` kind cannot carry a detached body.

§4.6 states: *"the child body and everything below it become one `Debris` entity (already registered:
`entity_kind.rs:225-232`, `BallisticReadvance`)"*.

MEASURED at `crates/core/src/entity_kind.rs:225-232`: `DEBRIS_DEF` is `class: Transient`,
`ghost_policy: NeverTransferOnly`, `continuity: BallisticReadvance`, `loss_budget: LossBudget(4)`,
`max_state_bytes: 128`.

A turret is a grid of player-placed blocks with its own chunks. It does not fit in 128 bytes. It is
also player property, and a Transient kind with a loss budget of four may be dropped. The cited
registration does not support the claim; it refutes it.

*Example: a gunner shoots the plate that holds a turret. Under the report the turret becomes debris.
Under the registry that debris may be dropped, and the player's built turret is gone.*

**Fix.** Name the detached body's carrier as an OPEN decision for the owner. The candidates are a new
Durable kind (which the entity-kind file itself calls *"first-class engineering and a crash-matrix
multiplication"*), or a new realm, or a refusal to detach at all. Do not present it as settled.

### F6 — WRONG. The chunk-delta message is named two ways.

§2.5 says an attachment rides *"the same message (`BulkMsg::Blob { kind: ChunkDelta }`,
`channels.rs:285-289`)"*. §6.2 says *"The chunk DIFF message carries `body` beside `chunk`
(`BulkMsg::ChunkDelta { realm, body, chunk, epoch, encoding, payload }`). postcard is positional, so
this field must exist the first time the arm is routed."*

MEASURED: `BulkMsg` has exactly one variant today, `Blob { kind: BulkKind, bytes: Vec<u8> }`
(`crates/wire/src/channels.rs:283-289`), and `BulkKind` is `{ChunkSnapshot, ChunkDelta, Catalog}`
(`channels.rs:291-296`). Under §2.5 the body id lives inside the opaque bytes and costs the wire
nothing. Under §6.2 the body id is a positional field of a NEW variant, which is a new wire shape and
a third SL6 ask.

The owner is asked to freeze a format. A format stated two ways cannot be frozen.

**Fix.** Pick one. If the body id rides inside the TLV envelope, say so and drop the "postcard is
positional" warning. If it is a wire field, list the new variant with F1's asks.

### F7 — UNMEASURED_AS_FACT. A rail on a planet is asserted, not shown.

§5 says *"Rails and pistons on a planet work identically… The joint code never reads
`voxel().geometry`."* §8 item 6 makes that the anti-vacuity assertion of the G-IDENTICAL gate.

The code says the opposite is the standing shape. `VoxelGeometry` is documented as *"the ONE seam
where spherical planets and Cartesian ship grids differ"* (`crates/sim/src/capability.rs:34-35`);
`Spherical` is *"tangent-anchored spherical projection with re-anchoring"* (`capability.rs:39`);
`Cartesian` is *"flat grid, no re-anchoring (AnchorGen never advances)"* (`capability.rs:40`);
geometry differences are *"confined to `FrameSpace` impls selected by `voxel().geometry`"*
(`capability.rs:12`); and that `FrameSpace` *"does not exist yet"* (`capability.rs:19`).

Two questions follow, and the report answers neither:

1. A prismatic axis "along a face normal" is a straight line. On a tangent-anchored spherical grid a
   straight line in cell space is a curve in the realm's frame. Which one does the piston follow?
2. What does an anchor advance do to a stored body pose and to a stored rail path? The report carries
   `anchor_gen` only in the edit request (§4.5 item 1, copying `block_system_design.md:4654-4658`) and
   never in the body family or the joint record (§6.3).

*Example: a miner's cart rides twelve rail blocks across a moon. The moon re-anchors while the miner
walks. The cart's path is a list of cells in a frame that just moved. The report does not say what
happens to the cart.*

**Fix.** Either measure the identical fixture on a spherical realm and report the bytes, or write the
question into the open register and say the answer is owed at P5 with `FrameSpace`.

### F8 — MISSING. The HUD value carries no range and no unit.

§3.3 option A ships `(key, value, state)`. §3.2 item 1 gives the `Display` attachment the params
`{ widget: WidgetKind(u16), style: u8 }`. Neither carries a range, a unit or a label.

A client cannot draw 0.61 as "61 %" without knowing the scale. If the client assumes zero to one, that
is a magic number on the client, which the project's own convention forbids, and it is the client
deciding what a server value MEANS — presentation crossing into state.

*Example: a fuel gauge reads 0.61. A thruster temperature reads 640. The same `Numeric` widget must
draw both. Without a stated range the client invents one.*

**Fix.** Put the range, the unit code and the label into the attachment's params, where the owning
realm states them once on placement and the chunk diff carries them.

### F9 — MISSING. The cost shape of a 512-body tree is never named.

§4.4 says sibling shape-casts run between bodies, and that inertia is recomputed when a body's scalar
changes. §4.3 estimates 512 bodies. §8 item 1 owes a measurement.

A measurement is owed, but the ALGORITHM is not stated. A pairwise sibling sweep over 512 bodies is a
scan. SL7 and SL9 both say a cost that grows with the set is a defect, and that finding what holds a
point is a lookup and never a scan (`CLAUDE.md:214-225`).

*Example: a hull with five hundred pistons. Every tick each piston asks whether it would sweep into a
sibling. Written as a scan, one hull costs a quarter of a million tests per tick.*

**Fix.** State the sibling test as an index over the bodies' swept boxes. State that inertia is
recomputed from the bodies whose scalar changed, not from all of them.

### F10 — MISSING. SL10 is cited for a thing SL10 does not cover.

§3.1's table says the widget's static look ships with the client and marks the lane
*"none — static shape (SL10)"*.

SL10 defines the static shape as what *"a fixed seed and an address decide and nothing else decides"*
(`owner_decisions_2026-09-07_voxels.md` V1.1). A gauge's frame and glyphs are a function of neither a
seed nor an address. The owner's actual authority for client-held art is V2.6: *"Style can be picked
up by the client, so when same blocks of that type are connected and construct the mesh, it will
render small additional details, that are not stored on the BE."*

The conclusion is right and the law cited is wrong. A wrong citation invites a later reader to stretch
SL10 further.

**Fix.** Cite V2.6.

### F11 — WRONG. Citation drift, small but real; the brief asks for `file:line`.

MEASURED by reading each range:

| The report says | The code says |
|---|---|
| `built.rs:57-84` for `BuiltBody` | the doc opens at 56, the struct is 61-83 |
| `built.rs:33-55` for `BuiltFacts` | the struct is 37-53 |
| `entity_kind.rs:210-217` for `SHIP_DEF` | the item is 207-214; `max_state_bytes: 8192` is 213 |
| `capability.rs:52` for `surfaces` | 51 is `surfaces`, 52 is `seats` |
| `capability.rs:11, 18` for `FrameSpace` | 12 names `FrameSpace`, 19 says it does not exist |
| `transfer_protocol.md:376` quoting *"never touched"* | the line reads *"untouched"* |

Correct as cited: `intershard.rs:1377-1400`, `intershard.rs:1404-1428`, `intershard.rs:34-35`,
`entity_kind.rs:225-232`, `channels.rs:283-296`, `channels.rs:332-335`, `channels.rs:378-391`,
`tlv.rs:1-30`, `interest.rs:1-29`, `pose.rs:91`, `built.rs:190-212`, `capability.rs:244-257`,
`capability.rs:260-273`, `DEFERRED.md:2925`, `DEFERRED.md:6240`,
`realm_crossing_plan_2026-09-02.md:61`.

**Fix.** Re-anchor the drifted ranges.

### F12 — MISSING. The foundation draws a HUD that shows nothing.

§12 item 7 plants *"the engine-free face-overlay primitive… drawing the widget's static chrome; no
value lane yet."*

A dial with no needle, standing on a hull plate in the shipped game, is a placeholder drawing. The
FINAL-backend law and the no-placeholder-rendering rule both refuse a stand-in in the one world.
Planting the TYPE is right. Drawing it before its lane exists is not.

*Example: a player boards a hull at P6 and sees an empty gauge glued to a plate. It never moves,
because no lane feeds it until P9.*

**Fix.** Plant the primitive and its test. Draw nothing until the value lane lands.

---

## 3. Checks that found nothing

- **SL1.** No realm derives or states its own placement. A body is not a realm, and body 0 never moves
  in its realm's frame (§6.2). No chain folds an absolute.
- **SL2.** No occupant pose enters another realm's simulation. A gunner on a turret stays in the hull's
  frame (`crates/core/src/pose.rs:91`).
- **SL3.** The parent still holds only the bound and the look as readings. The realm draws its own
  bodies.
- **SL4.** No motion symbol reaches the crossing path. The joint record names anchors, a degree of
  freedom, limits and a scalar.
- **SL5.** No reduced world. No second implementation. The report refuses to adopt rapier and names it
  as an option for the owner (§1, §10 item 4).
- **The seed ruling.** No substance is a function of seed and position. An attachment is placed.
- **Reach and visibility.** No tree walk. No dormant realm is drawn. The report argues that a hinge
  deserves no reach test, no boot and no lease (§4.2).
- **HR3.** No match on a shard kind. A body id is a number in a key.
- **The movement contract.** No velocity crosses up. No parent sets a speed. No re-clamp.
- **V2.1-V2.9.** No door is shut, with the two qualifications in F3 (an asteroid cannot carry a joint
  under the stated gate) and F4 (one attachment per face is firmer than admitted).

---

## 4. What the report must do to stand

1. List `WidgetLevel`, `BodyPose` and the chunk-delta variant as explicit SL6 asks (F1, F6).
2. Delete the 30-arm ceiling argument and count the arms (F2).
3. Name the capability a body tree needs, and fix the asteroid case (F3).
4. Admit the version-floor cost of one attachment per face (F4).
5. Reopen the detached-body carrier as an owner decision (F5).
6. Answer, or formally owe, the spherical and re-anchoring questions (F7).
7. Put a range and a unit in the `Display` params (F8).
8. State the sibling test as an index (F9).
9. Cite V2.6, not SL10, for client-held widget art (F10).
10. Re-anchor the drifted citations (F11), and stop drawing an empty dial (F12).
