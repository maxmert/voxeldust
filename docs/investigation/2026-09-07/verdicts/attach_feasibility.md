# Feasibility refutation — 05 Attachments and mechanical bodies

**Subject:** `docs/investigation/2026-09-07/05_attachments_bodies.md`
**Lens:** technical feasibility. Every claim about the code was re-grepped in this worktree. Every
number was checked against its source.
**Verdict: REFUTED.** The report's central recommendation — one realm, a tree of bodies, kinematic
joints — survives. Four load-bearing claims under it do not. Nothing was built or run.

---

## 1. The claims that fail

### F1 — "no transcendental, therefore bit-deterministic" is WRONG

The report writes the child body's pose as
`pose(child) = pose(parent body) ∘ T(anchor_parent) ∘ R_or_S(dof, q) ∘ T(anchor_child)⁻¹`,
and says this *"needs no solver ... and it is bit-deterministic (only multiply, add and one
quaternion product; no transcendental)"* (report §4.4, lines 389–393).

A revolute joint turns the turret by the angle `q`. The rotation from an angle to a quaternion is
`(cos(q/2), axis · sin(q/2))`. Sine and cosine are transcendental. The report's own record stores `q`
as a scalar on an integer grid, so the sine and the cosine must come from somewhere.

SL10 clause 4 names this exactly: *"No call into the platform's transcendental functions (`sin`,
`exp`, `pow` from libm); where a curve is needed, the crate carries its own deterministic
implementation or an integer table"* (`docs/design/owner_decisions_2026-09-07_voxels.md:36-39`).

The claim is load-bearing. It is the whole ground of open decision 4 (*"Recommend analytic first — no
new dependency, bit-deterministic"*, report line 642) and of *"buildable with what the tree has
today"*. As written, a turret's angle is a libm call, and a libm call is the one thing SL10 forbids.

*Example: a gunner turns the turret on his hull to 37 degrees. The hull's shard on an x86-64 node and
the same shard restarted on an aarch64 node call the platform's sine and get two different last bits.
The turret's barrel sits a fraction of a millimetre apart on the two runs, and the floodlight the
player welded to the barrel sits with it.*

**Fix.** State that the joint carries its own deterministic sine and cosine — an integer table keyed
by the angle grid `q` already uses, which is the same rule SL10 clause 4 sets for the generator. Add
the gate: the same joint at the same `q` produces byte-identical body poses on x86-64 and on
aarch64. Then the "no new dependency" recommendation stands on a measurement instead of on a false
premise.

### F2 — "every voxel realm profile already has `integrates_children`" is WRONG

Report §6.7 line 570: *"A joint needs `integrates_children`, which every voxel realm profile already
has (`capability.rs:244-300`)."*

MEASURED, by reading the file: inside that very range, `asteroid()` at
`crates/sim/src/capability.rs:276-281` builds a voxel realm — `voxel: Some(VoxelGeometry::Cartesian)`
and `block_edit: true` — and does **not** set `integrates_children`. It is a realm a player may mine
and build in, and it has no such capability.

The conflation is worse than the counter-example. `integrates_children` is read at exactly one place
in the whole tree, `crates/sim/src/stub/register.rs:343`, and its doc says what it gates: *"Does THIS
shard do the physics for what is inside it"* — that is, whether the shard accepts a **child realm's**
`ChildDrive`. A body is not a child realm; the report says so itself (§4.3). So the capability the
report names governs a different lane from the one a body tree uses.

*Example: a player anchors a piston to a mined asteroid to lift ore crates. Under the report's rule
the asteroid's shard refuses it, because the asteroid profile carries no `integrates_children`. Yet
the report's §6.7 promises "no new capability".*

**Fix.** Either name the honest gate — a new capability, say `mechanisms`, derived from `block_edit`
in the lattice `ShardProfile::build` already runs (`capability.rs:120-140`) — or state that a joint
needs nothing beyond `voxel` and correct §6.7. Do not spend `integrates_children`.

### F3 — door 1's WIRE half is mis-argued, and the report contradicts itself

Two statements about the same message:

- §2.5 line 237: *"No new wire arm and no new lane"*; the attachment rides
  `BulkMsg::Blob { kind: ChunkDelta }` and adds one TLV tag, *"and a tag is additive by construction
  of the TLV envelope"*.
- §6.2 line 510: *"The chunk DIFF message carries `body` beside `chunk`
  (`BulkMsg::ChunkDelta { realm, body, chunk, epoch, encoding, payload }`). postcard is positional,
  so this field must exist the first time the arm is routed."*

Both cannot hold. MEASURED: today `BulkMsg` has exactly one arm,
`Blob { kind: BulkKind, bytes: Vec<u8> }` (`crates/wire/src/channels.rs:285-289`). The chunk delta is
an **opaque byte string** inside that arm. The report itself makes the inside of that byte string a
TLV envelope with skip-unknown tags. A field inside a skip-unknown TLV envelope is an append, not a
one-way door. And the shape the report writes in §6.2 IS a new `BulkMsg` arm with named fields, which
§2.5 says it does not add.

The STORE half of door 1 stands: a key prefix in the realm's flat table is genuinely fixed once a
world is saved. The WIRE half, as argued, is not a door at all.

*Example: the hull's shard ships a chunk of the turret's grid to a watching client. The body number
sits as a tag inside the delta blob. An older client skips the tag and draws the hull's own grid; a
newer client reads it and draws the turret's grid at the turret's pose. Nobody re-versions a message.*

**Fix.** Split door 1 in the table. Row 1a: the store's chunk keyspace prefix `(body, chunk)` — a
real door at the format freeze. Row 1b: the `body` field in the delta payload — additive, not a door,
because the payload is TLV. Delete the postcard-positional argument, which applies to `BulkMsg`'s
arms and not to the bytes they carry.

### F4 — the 13 B per binding is carried over from a much narrower key

Report §3.3 line 288 and §10 item 1: option A is *"ESTIMATED 13 B per changed binding by the base's
own arithmetic (§7.9)"*, and B is *"ESTIMATED 10–30× the bytes of A"*.

The base's 13 B is exact and it is for a different key:
`(panel_id u32, slot u8, value Fx)` = **13 B per changed binding**
(`docs/investigation/block_system_design.md:14199`).

The report's option A does not ship a `panel_id u32`. It ships
`WidgetLevel { key: AttachmentKey, value: Fx, state, at }` where
`AttachmentKey = (BlockAddr { body: u16, chunk: Tier0Key, cell: CellIndex }, face: u8)`
(report §2.2, §6.2, §3.2 item 4). ESTIMATED, by adding the report's own widths: body 2 B, a chunk key
8 B (the base's `chunk_meta` uses a `u64 chunk_key`, `block_system_design.md:2113`), a cell index 3 B
(the base's pyramid entry spends 18 bits on `cell`, `block_system_design.md:2156`), a face 1 B — 14 B
for the key alone, before the value, the state and the tick. So a changed binding is ESTIMATED
19–24 B, not 13 B, and every rate the base derives from 13 B (52 kB/s naive, 6.7 kB/s after the
angular floor, `block_system_design.md:14203-14204`) is understated by about half again.

The figure carries a decision: it is the byte argument the owner is asked to accept for option A.

*Example: the fuel gauge on the hull plate. The hull's shard cannot say "panel 41"; it must say
"body 0, chunk 12, cell 4114, face +Y", because that is the key the report froze as a one-way door.
The wider key is the price of keying on the block's address, and the report should charge it.*

**Fix.** Re-derive the per-binding bytes from this report's own key and mark it ESTIMATED with the
arithmetic shown. Then §8 item 5 measures it. Option A very likely still wins; the margin is smaller
than 10–30×.

---

## 2. The measurement that does not measure

### F5 — the §1 grep reports an output its own command cannot produce

Report §1 line 72: *"MEASURED: `grep -rn "SubBody\|BodyPart\|Attachment" crates --include='*.rs'`
finds no attachment or body symbol (one unrelated word in `crates/wire/src/session_flow.rs:39`)."*

MEASURED, re-run in this worktree: that exact command prints **nothing at all**. Line 39 of
`crates/wire/src/session_flow.rs` reads *"a duplicate attach at the same fence re-sends the existing
attachment"* — lowercase. A case-sensitive pattern `Attachment` cannot match it.

The conclusion survives and is in fact stronger: no such symbol exists in any form. The measurement
as written does not. The standing rule is that a measurement is a thing that could have failed, never
a reconstruction; a reported output must be the output.

**Fix.** Re-run the case-insensitive grep, paste what it prints, and let the reader see the four
harmless `AttachSession` hits.

---

## 3. What the domain needs and the report does not carry

### F6 — the anchor generation is missing from a key the report freezes

The report makes `AttachmentKey = (BlockAddr, face)` and the joint's stored anchors
`anchor_parent: (BlockAddr, face)` one-way doors (doors 2 and 3, and the BODY family value at §6.3
line 531). Neither carries an anchor generation.

But the base's own edit reference does: `WorldBlockRef { realm, chunk, cell, anchor_gen, expected }`
with the comment *"stale-anchor rejection, the FrameSpace fence"*
(`docs/investigation/block_system_design.md:4654-4658`). And a planet realm re-anchors: the code's
own words are *"Planet surface: tangent-anchored spherical projection with re-anchoring"* versus
*"Ship/station interior: flat grid, no re-anchoring (AnchorGen never advances)"*
(`crates/sim/src/capability.rs:38-41`). MEASURED: `AnchorGen` exists today only as a doc word
(`crates/core/src/fence.rs:18`, `crates/sim/src/capability.rs:40`), so nothing yet forces the report
to think about it — which is precisely why a freeze must.

The report's own example is the one that breaks. §5 lays twelve rail blocks *"across the planet's
surface"* and anchors a carriage to the first rail's top face.

*Example: a player builds a mine cart on a moon. The player then walks two kilometres, and the moon's
realm re-anchors its tangent grid. Every stored `(chunk, cell)` in the rail's path and in the joint's
anchors now names a different place. The cart's anchor points at empty rock, and the HUD the player
stuck on the cart's side is on somebody else's boulder. Nothing errored.*

**Fix.** Put the anchor generation inside `BlockAddr`, or state the rule that a stored address on a
`Spherical` realm is rewritten by the re-anchor pass and name the pass as a foundation item in §12.
Either way, decide it **inside** doors 1 and 2, because after the freeze it is a migration over every
saved moon.

### F7 — the body's pose and the realm's pose are never tied to one stamp (SL8)

§4.4 line 410 ships a body's pose *"like an occupant's pose"*. An occupant's pose rides the
unreliable latest-wins snapshot lane and is drawn through a 100–150 ms interpolation buffer
(`crates/wire/src/channels.rs:325-328`, `INTERP_BUFFER_MS = 120.0`). The realm's own drawn placement
is a different row, `SceneRow`, with *"the stamp explicit PER ROW"*
(`crates/wire/src/channels.rs:378-391`).

The report nowhere requires the turret's body row and the hull's realm row to carry the SAME stamp.
Two rows on two stamps, each interpolated on its own schedule, is the seam.

*Example: a gunner stands on the turret platform while the hull rolls. The hull's row arrives on tick
N and the turret's body row on tick N−1. For one frame the platform slides under the gunner's boots
and his feet sink into it. That is SL8 seam kind "jump", and the neighbouring frame is seam kind
"flicker".*

§8 item 3 owes the measurement of exactly this. It owes no mechanism. A measurement with no mechanism
behind it is a test that will go red with nothing to fix it.

**Fix.** State the rule: a body row's stamp is the stamp of its realm's row, and the client draws
both from one interpolation sample or draws neither. Add it to §12 as a foundation item, not to §8 as
a measurement.

### F8 — "the hull's physics carries the gunner as it carries someone on a moving belt" is unmeasured

§4.2 line 355 uses this sentence to defeat option (B) and to keep a gunner out of a sub-frame.

MEASURED: no such machinery exists. `crates/physics/src` holds `celestial`, `motion`, `taxonomy` and
`worldgen` only; there is no collider, no shape cast and no character controller anywhere in the
tree. Every hit for "controller" is a **flight** steering loop
(`crates/bins/src/flight.rs:32`, `crates/client-harness/src/nav.rs:16`). There is no rapier and no
parry dependency; the only mention in any `Cargo.toml` is a comment (`Cargo.toml:96`) — which the
report states correctly elsewhere (§1 line 109).

Standing on a surface that both translates and ROTATES is the hard case of a character controller,
not a free consequence of a body tree. The report states it as settled.

**Fix.** Mark the sentence as the design intent it is, and add it to §8: a gunner keeps contact with
a turret platform turning at its rated speed, MEASURED on the process tier, with the pop detector on
his feet. Name it as owed to P5, and say plainly that no contact machinery exists today.

### F9 — door 1 cannot be closed before the sub-metre answer (V2.4)

§12 item 1 orders `BodyId(u16)` into `BlockAddr` as the very first thing the foundation plants. §2.2
line 170 and open decision 6 leave the sub-metre index to another report: *"If the V2.4 design gives
a block a sub-cell index inside its 1 m cell, that index is part of `BlockAddr`."*

A positional address type cannot gain a field after the keyspace is saved. That is what makes it a
door. So `BlockAddr` must be frozen with the sub-cell index already inside it or already ruled out.
The report half-says this in decision 6 and then contradicts it by ordering the plant first in §12.

*Example: the foundation saves a moon with `BlockAddr { body, chunk, cell }`. Two slices later V2.4
lands 0.25 m blocks. Every saved chunk key on every mined moon, station and hull must be rewritten to
make room for the sub-cell index — the migration door 1 exists to prevent.*

**Fix.** Make §12 item 1 read: freeze `BlockAddr` ONCE, with the body id and the V2.4 sub-cell index
together, after the V2.4 report answers. Add the dependency to the door table.

### F10 — three cases the domain owns are not answered

1. **The child's anchor block breaks.** §4.6 covers *"a joint's PARENT block breaks"* only. If the
   turret's own base block is shot away, `anchor_child` dangles and no rule says what the body does.
2. **A block replaced in place.** `on_block_removed` (§2.4) fires on a removal. A player who swaps a
   plain hull plate for an armoured one has not removed a cell; the HUD's row survives on a block
   whose type, and possibly whose face set, changed. No rule is given.
3. **Which carriage a changed rail cell belongs to.** §5 recomputes a rail path *"when a rail cell is
   placed or removed"*. No reverse index from a cell to the carriages that ride it is designed, so
   the naive answer is a scan of every carriage in the realm on every rail edit.

*Example: a player mines one rail cell out of the middle of the twelve-cell track. The cart should
stall at the break. Today's design has no way to find the cart from the cell.*

### F11 — the client frame budget is never costed

§8 owes six measurements. Five are server-side and one is a seam test. None is a client frame cost.
A construction with many bodies is many separate chunk-mesh transforms, and the report's own
argument-from-cost imagines 512 of them (§4.3 line 369). The task's own budget questions — a planet
at 10,000 km in the window, a landing at flight speed, and which thread generates a chunk — are
answered nowhere in this report for a body's grid.

**Fix.** Add to §8: the client's frame time and draw-batch count for a hull with N bodies at N ∈
{1, 8, 64, 512}, drawn in the window at the release build, and the thread the body's chunk remesh
runs on.

---

## 4. Lesser accuracy defects (they change no conclusion)

- **`WorldBlockRef` gains two fields, not one.** §4.5 says the edit target is *"the base's shape
  (`block_system_design.md:4654-4658`) **plus `body`**"* and then writes
  `{ realm, body, chunk, cell, face, anchor_gen, expected }`. The base's shape carries no `face`
  (`block_system_design.md:4654-4658`). Two fields are added; one is announced.
- **`GridMapping::Identity` is written as if it exists.** §4.5 item 3. MEASURED: `grep -rn
  "GridMapping" crates --include='*.rs'` returns nothing. It is a base-doc type
  (`block_system_design.md:2662`), not code. The report's method promises a `file:line` for every
  claim about today's code; this one has none.
- **`ChunkKey` and its "six reserved bits" are written as if they exist.** §6.2. MEASURED: `grep -rn
  "ChunkKey\|CellIndex\|Tier0Key" crates --include='*.rs'` returns nothing.
- **Cited ranges drift.** `SHIP_DEF` is `crates/core/src/entity_kind.rs:207-214`, not 210-217 (the
  cited range runs into `NAMED_CONSTRUCTION_DEF`). `DEBRIS_DEF` is 226-233, not 225-232. The
  capability data is `crates/sim/src/capability.rs:47-91`; line 100 cited as its end is inside
  `ProfileError`. `realm_head.rs:50-58` supports the exterior lease only, not the reach and the
  boot-on-demand the sentence also claims.
- **"(B) cannot solve a constraint at all" is over-stated.** §4.2 line 328. Two realms CAN share one
  process today: `CoHostedAuthority` exists (`crates/sim/src/stub/realm_head.rs:48`) and the harness
  populates it (`crates/harness/src/topology.rs:305-307`). The honest form: under (B) the solve
  depends on a co-hosting placement the orchestrator may change at any time, which is worse than a
  refusal because it fails only sometimes. The recommendation is unchanged.

---

## 5. What stands

These were checked and hold. They are the report's real contribution.

- **A hull has no grid, no chunk and no block today.** Confirmed: `BuiltBody` is a name, an owner, a
  blueprint, two boxes, five whole numbers and a fence (`crates/core/src/built.rs:61-83`), and a test
  enforces that no field can hold a placement (`built.rs:190-212`). The hull draws as one box
  (`crates/bins/src/bin/build-ship.rs:52-54`).
- **Nothing new crosses the hull ↔ star-system boundary.** `ChildDrive` carries a push and a turn in
  the child's own frame (`crates/wire/src/intershard.rs:1377-1397`) and `ChildFacts` carries a mass,
  a cross-section, a drag coefficient and a closed set of declared states
  (`intershard.rs:1403-1427`). The parent rotates and integrates
  (`crates/sim/src/stub/drive.rs:139-183`). A turret's angle needs none of it. SL6 is respected, and
  the report correctly asks for nothing.
- **`BlockEdit`, `Coupling` and `Signal` are reserved names on the inter-shard wire**
  (`crates/wire/src/intershard.rs:34-35`), so no arm is spent.
- **The realm's own store grows additively.** One flat table, one prefix byte per family, TLV rows
  whose tags are append-only (`crates/sim/src/stub/built_store.rs:19-81`), and a reader skips unknown
  tags and refuses a missing required tag (`crates/core/src/tlv.rs:18-26`). A body family and an
  attachment family are appends, exactly as §6.3 says.
- **No rapier, no parry, and none proposed.** MEASURED: `grep -rn "rapier\|parry" Cargo.toml
  crates/*/Cargo.toml` returns one comment (`Cargo.toml:96`). D-19 is red
  (`docs/design/DEFERRED.md:6240`). The report names a solver as an owner option and adopts nothing.
  This is correct conduct.
- **HR1 is quoted exactly** (*"a shard's World/rapier/redb is private"*, `CLAUDE.md:103`) and the
  determinism rule too (*"rapier state is checkpoint-carried, never re-simulated cross-host"*,
  `CLAUDE.md:266-267`).
- **D-9 is green** (`docs/design/DEFERRED.md:2925`) and the per-cube named-recipient interest exists
  (`crates/sim/src/stub/interest.rs:1-29`), so §9's discharge of D-DEC-4 is right.
- **The collision rule is untouched.** An attachment occupies no cell and is not an entity, so it
  never collides, and a crate may still be placed on the plate that carries a gauge. This answers
  V2.5 as the owner worded it.
- **One realm, a tree of bodies, is the right answer.** The cost argument against a realm per hinge
  is sound in direction even where its numbers are estimates: thirteen shards for one hull with six
  turrets, four doors, two pistons and a landing gear, against one.

---

## 6. What must change before this report is safe to build from

1. Correct F1. Name the deterministic sine and cosine, or the recommendation in decision 4 is
   unsupported.
2. Correct F2. Name the real capability gate, and fix the asteroid case.
3. Split door 1 into its store half and its wire half (F3).
4. Re-derive the per-binding bytes from this report's own key (F4).
5. Re-run and paste the §1 grep (F5).
6. Decide the anchor generation inside the frozen key, and tie door 1 to the V2.4 answer (F6, F9).
7. Add the one-stamp rule for a body row and its realm row (F7).
8. Move the moving-floor claim from §4 into §8 as an owed measurement (F8).
9. Answer the child anchor break, the replaced block and the rail reverse index (F10).
10. Add a client frame budget to §8 (F11).
