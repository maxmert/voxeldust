# Verdict — LAW REFUTER — "Smooth terrain built of voxels"

**Target:** `docs/investigation/2026-09-07/02_smooth_terrain.md`
**Lens:** every binding law and ruling. CLAUDE.md HR1–HR6 and SL1–SL10;
`owner_decisions_2026-09-07_voxels.md` (SL10 and V2.1–V2.9); the earlier rulings; SL8; the
FINAL-backend law; the movement contract; the seed ruling of 2026-08-27; the reach and visibility
rulings.
**Method:** I read the report, then I read the law, then I ran `grep` on the code for every
`file:line` the report states. I list what I measured and what I did not.
**Date:** 2026-09-07.

## Verdict: REFUTED

The report is strong and mostly lawful. It does not fold an absolute placement, it does not send an
occupant pose across a boundary, it does not name motion on the crossing path, and it does not scan a
parent's children. But it carries **eight load-bearing claims that are wrong or that break a law**, and
five more that state an argument where the law demands a measurement. Two of the eight sit inside the
one-way doors the report asks the owner to shut, so they cost the most if they go through unchecked.

The recommendation itself — a density byte per cell, a smooth extractor, square blocks in the same grid
— survives. Fix the eight items and the report becomes a sound basis for the owner's decision.

---

## 1. BREAKS_LAW (SL6) — the report says "no new wire arm is asked for". It is not true.

**The claim.** Section 0: *"Nothing in this domain needs new data across a realm boundary (SL6)."*
Section 1, row 8: *"The terrain diff lane (SL10 clause 6) rides the reserved `BlockEdit` arm, one hop,
from the owning realm's shard. No new arm is asked for here."* Section 9 repeats it: *"SL6 — new data
across a realm boundary: NONE asked."*

**What I measured.** `BlockEdit` is reserved on `InterShardFlow`, which is the SHARD-TO-SHARD contract
(`crates/wire/src/intershard.rs:34`). The lane to the client is a different contract:
`ServerControlMsg` (`crates/wire/src/channels.rs:88`). I listed every arm it holds today: `Welcome`,
`SubscriptionOpened`, `SubscriptionClosing`, `AuthorityChanged`, `RequestCut`, `CutConfirmed`,
`TransferCosmetic`, `TransferRejected`, `Ping`, `Close`, `UniverseRate`, `OwnEntity`, `RealmRegistry`,
`RealmSceneDelta`, `Event(EventMsg)`, `StarCatalogue`, `SkyAlive`. **No arm carries a cell, a block or a
terrain edit.** A shard-to-shard arm cannot reach a client, so the terrain diff needs a new arm on the
client lane.

**Why it breaks the law.** SL6 has two halves and the report answers only one. The first half is "new
data across a realm boundary"; the SL2 clarification of 2026-08-24 does excuse that, because the
connection plane is not a realm. The second half is stated in CLAUDE.md in the same sentence: *"and
before adding a wire arm. Default NO."* The report never asks.

**Why it is load-bearing.** Slice 5 of section 11 is "mining and placing on terrain", and its whole
delivery is the diff lane. If the arm is refused, the slice has no path.

**The fix.** Make the ask, in SL6's own shape.
- WHAT: one edited cell — the chunk key in the realm's own frame, the cell index, the block identity,
  the orientation, and the density byte.
- FROM WHICH REALM TO WHICH: from the moon's own shard, through the gateway, to the client. No realm
  learns another realm's cells.
- WHY THE RECEIVER CANNOT COMPUTE IT: the seed decides the hill; it does not decide the tunnel a player
  dug last week, because a player is live state.
- THE SHAPE: an appended `EventMsg` variant on the existing `Event` arm is the cheapest lawful form.
  The arm's own doc already says P9's signal deliveries ride it as appended variants — one carrier, two
  consumers (`crates/wire/src/channels.rs:189-194`).
- COST OF DOING WITHOUT: a player digs a scoop into a dune and nobody else ever sees it.

---

## 2. WRONG — the density's SIGN contradicts itself, inside the one-way door.

**The claim.** Section 2.3: the density is *"clamped to `[−1, +1]` cell, negative inside solid (the
workspace's existing convention, `crates/core/src/geometry.rs:435`)"*, and it *"is the signed distance
from the cell's centre to the terrain surface, measured along the radial (the `h − r` of the
generator)"*. Section 7, door 1: *"signed distance in cell units, negative inside"*. Section 7, door 2:
*"The density's meaning is the generator's `h − r`"*.

**What I measured.** `Boundary::signed_distance` is negative inside
(`crates/core/src/geometry.rs:435-444`; `Shell { r } => p.length() - r`). So the convention the report
cites is real. But on a moon, a cell inside the rock has its radius `r` BELOW the terrain height `h`, so
`h − r` is POSITIVE inside solid. **The two doors state opposite signs for the same byte.**

Section 2.3's own carver rule proves which one the report means: `density = max(h_term, carver_term)`
with *"air wins"*. A `max` makes air win only when air is the POSITIVE side — which is the
negative-inside convention, not `h − r`. Section 3.3's placement rule agrees: *"density ≥ 0, i.e. the
cell centre is on the air side"*.

**Why it is load-bearing.** Section 7 states the cost in its own words: *"a change of resolution or sign
reinterprets every edit ever made."* The owner is asked to shut this door before the first world is
saved.

**The fix.** Write the datum as `r − h` (the radius minus the terrain height), not `h − r`. Pin it with
a test that could fail: a cell one metre under a moon's surface reads a negative density, and a cell one
metre above it reads a positive one.

---

## 3. WRONG — the `i8` quantum of 1/127 is not an exact count of fine cells, against the report's own rule.

**The claim.** Section 1, row 6: *"Every quantised terrain length (density step, skirt depth, controller
offsets) is an exact count of these cells."* Section 2.3: *"Quantised to `i8` (1/127 cell, about 8 mm at
tier 0)."*

**What I measured.** The fine rung's step exponent is −10 (`crates/core/src/pose.rs:475-528`,
`Tier::Fine => -10`), so a fine cell is `2⁻¹⁰ m = 0.9765625 mm`. One metre divided by 127 is
`7.874 mm`, which is `8.192` fine cells. **Not an integer.** The report's own exactness rule fails on its
own number. At a coarse rung the miss grows: a tier-L cell is `2^L` metres, so the step is `2^L/127`
metres, never an exact count at any rung.

**Why it is load-bearing.** It is door 1 in section 7 and it is permanent.

**The fix.** Quantise at **1/128 cell**, not 1/127.
- At tier 0 one density step is exactly 8 fine cells; at tier L it is exactly `2^(L+3)` fine cells.
- The `i8` then spans `[−1.0, +0.9921875]` exactly, and every step is a power of two, so nothing drifts.
- Section 5.3's pyramid mean `(sum + 4) >> 3` stays exact, unchanged.

*Example: a player mines the top of a dune; the cell's density moves by a whole number of the fine cells
the whole world already counts in, on a moon and inside a hull alike.*

---

## 4. BREAKS_LAW (HR4) — the two-shard-kind fixture for the smooth lane is empty on one kind.

**The claim.** Section 11, slice 2, gate: *"the same fixture on a Spherical and a Cartesian profile (the
Cartesian one emits nothing)"*. Section 3.1: *"On a Cartesian realm (a hull, a station) the byte is
always 'fully air' and the smooth lane emits nothing."* Section 9 states this as HR4 compliance.

**Why it breaks the law.** HR4 reads: *"every feature passes the identical fixture on ≥2 shard kinds
(G-IDENTICAL) or it doesn't land."* A fixture that asserts the feature produced NOTHING is not the
feature running on a second shard kind. It is the feature being absent there. The smooth lane — the
entire subject of this report — would then live on exactly one shard kind, which is the fork HR3 and HR4
exist to prevent.

**What I measured.** Nothing in the code refuses terrain on a Cartesian realm. `ShardProfile::build`
accepts `surfaces: true` together with `VoxelGeometry::Cartesian`
(`crates/sim/src/capability.rs:133`), and the `ship()` profile already sets exactly that pair
(`crates/sim/src/capability.rs:260-265`). The refusal is the report's own rule, not the code's.

**The fix.** Let a Cartesian realm hold terrain-form cells. The game already has the example: a
station's growing bay with a soil floor, or a hull with a dirt-filled ballast hold. Then the fixture is
truly identical — plant the same density pattern in index space on a moon and inside a station hull,
run the same extractor, and assert the same quad list on both. The extractor reads cell-centred samples
in index space, so it does not care which grid it stands on.

---

## 5. BREAKS_LAW (the FINAL-backend law, and HR3's spirit) — two collider builders chosen by a config field.

**The claim.** Section 3.2, the cube lane's collider column: *"Cartesian: the library's voxel shape if
4-B adopts it; spherical: 8-cell greedy boxes through `cell_point`"*. Section 4.2 repeats the split for
convex shapes.

**What I read.** The report inherits this from the base
(`docs/investigation/block_system_design.md:6465-6472`), which states two different collider
constructions in two columns of one table. The report marks the library line UNRATIFIED in section 10
but keeps the split.

**Why it breaks the law.** The FINAL-backend law forbids *"a second implementation selected by config"*.
`VoxelGeometry` IS a config field (`crates/sim/src/capability.rs:37-41`, set from `CapRequest`). Two
collider builders behind it are two implementations of one job, and the second one can only be tested
on one shard kind — which then fails finding 4 as well.

**The fix.** ONE cube-lane collider builder: greedy boxes in index space through `cell_point`, where
`cell_point` is the identity map on a Cartesian realm. If a library voxel shape is later adopted, it is
an OPTIMISATION that must produce the identical collider, and slice 4's digest gate is already the
measurement that proves it. *Example: a steel cube in a station's wall and a steel cube on a moon's
hillside are built into a collider by the same code, and the same digest test covers both.*

---

## 6. BREAKS_LAW (SL10 clause 5) — nothing ties the CLIENT's resident detail level to where a body stands.

**The claim.** Section 5.4: *"Collision is tier 0, by type"*, and the collider is resident inside a 64 m
physics radius. Section 5.4 again: *"The server never runs the crossfade, the skirt or the tier
selection. Those are the client's presentation of the same shape."* Section 10: *"the collider is the
extracted surface, equal to the drawn one, not a superset."*

**Why it breaks the law.** SL10 clause 5 says: *"The server computes collision on the SAME shape … What
the player sees is what the player stands on."* The report fixes the SERVER at tier 0. It leaves the
CLIENT's tier to an angular rule per chunk column (sections 5.1 and 5.2). Those two are not tied
together anywhere. A hull that decelerates hard can touch down on ground the client is still drawing at
a coarse rung, and the drawn surface then differs from the collider by the report's own derived bound —
2 cells at the switch (section 5.2). The boots float or sink. That is an arrival pop and a broken clause
in one event.

**The fix.** State the missing rule and gate it.
- The rule: every chunk inside a body's physics radius is resident at tier 0 ON THE CLIENT before
  contact is possible.
- The warming: grow that radius with the closing speed, never cap the speed — this is the owner's own
  answer of 2026-08-27, item (2).
- The gate: a landing fixture. A hull descends at its rated cruise onto a moon, touches down, and the
  character's boots never sink into the slope and never float above it, at any approach speed the suit
  or the hull states.

---

## 7. BREAKS_LAW / a door shut on V2.2 — the restated collision rule has no arm for a tree.

**The claim.** Section 2.3, item 4: *"a thing collides iff it is a square cell, an entity, or the
extracted terrain surface of tier-0 cells."*

**Why it shuts a door.** V2.2 says a tree is ONE object on the surface, drawn by the client from seed
parameters, and *"then the collisions are calculated on the server, so server somehow should know the
shape"*. Under the report's restated rule a tree is none of the three arms: it is not a square cell
(section 3.5 says so), it is not an entity (it is a seed-shaped feature of the surface, not a
transferable body), and it is not the extracted terrain surface. So the rule as written forbids a tree
from colliding. Section 4.2 makes it worse by emptying the pass-through class, and section 3.5 hands the
whole matter to the tree domain while leaving this rule standing over it.

**The fix.** Add the fourth arm, and name it here: **a seed-shaped surface object, whose collider the
realm's shard derives from the SAME generator crate** (SL10 clause 5 gives exactly that path, and it
needs no diff, because a tree's trunk and canopy are static shape). *Example: a player walks into a pine
on a moon and stops, because the moon's shard evaluated the same crate the client drew the pine from and
got the same trunk.*

---

## 8. BREAKS_LAW (the seed ruling of 2026-08-27) — the substance is a pure function of seed and position, with no carve-out.

**The claim.** Section 0, item 1: *"A terrain cell holds one substance and one density sample … The seed
decides it for every unedited cell (SL10)."* Section 11, slice 1: the generator emits *"an `i8` density +
a substance per cell"* and includes *"the strata"*.

**Why it breaks the law.** The ruling of 2026-08-27 binds the block system by name: *"a block's substance
may not be a pure function of (position, seed)"*, because *"a shipped generator inverts to its seed"*.
SL10 clause 6 restates it: *"every deposit that live state decides"* crosses as a diff. SL10 makes the
danger worse, not better — clause 2 ships the generator crate to EVERY client, so a seed-decided ore vein
is a public map the moment the first client runs it. The report never names the boundary. An implementer
reading slice 1 puts the ore in the strata.

**The fix.** State the boundary in this report, because this report owns the substance field.
- The seed decides only the COMMON structural substances: rock, dirt, sand, ice, water.
- Every VALUABLE deposit is live state and arrives on the same diff lane finding 1 asks for.
- Gate it with a test that could fail: the generator crate's substance table contains no row the
  manifest marks valuable.

*Example: two players land on the same moon and both see the same grey rock face; only the one who
surveyed it knows which seam holds the ore, because the moon's shard decided that seam and told nobody
else.*

---

## 9. MISSING — nothing says the client may derive a realm's terrain ONLY while that realm is drawn.

**What is absent.** SL3 says *"a realm that is not running cannot be drawn — which is WHY visibility is
the spin-up trigger"*. The ruling of 2026-09-01 says *"a dormant realm is never drawn by anybody"*. SL10
clause 1 permits the client to derive static shape; it does not lift either of those.

The report never states the tie. Section 5.1's example has the client evaluating a moon's seed while a
hull descends. Section 5.2's tier-refusal row leans on the realm proxy row, which comes from the
composed stream — but that is an aside, not a rule. As written, a client holding the seed can draw the
hills of a moon whose shard is asleep.

**The fix.** State it as a rule of this domain: the client derives terrain for a realm ONLY while that
realm holds a row in the composed scene (`ServerControlMsg::RealmRegistry` and `RealmSceneDelta`,
`crates/wire/src/channels.rs:148-188`), which is exactly the reach ruling's in-range set. *Example: a
moon two star systems away is not in the window, so the client evaluates nothing for it, even though it
holds the crate that could.*

---

## 10. UNMEASURED_AS_FACT — "two colliders overlap; the character stands on the higher one. Harmless."

**The claim.** Section 3.3, Case 1.

**Why it fails the standing rule.** "Harmless" is an argument. The report's own reading rule says a
number is MEASURED, ESTIMATED or UNMEASURED, and this claim carries no label at all. It is also the one
place in the design where a collider exists that nobody draws: the terrain surface hidden inside an
opaque steel foundation. SL10 clause 5 is about exactly that equality.

**The fix.** Mark it UNMEASURED and add it to section 6's bench list: a character sweep across a
foundation cut into a 30° slope, at 20 offsets, asserting no vertical impulse and no step the player can
climb. The base already carries the shape of that assertion
(`docs/investigation/block_system_design.md:6502-6505`).

---

## 11. WRONG (two code citations) — the render seam is narrower than the report says.

**The claims.** Section 1, row 10: *"`MeshPrim` is a plain vertex buffer + colour + transform"*, cited as
`crates/client/src/realm_scene.rs:1-17,787-797`, with *"the renderer does `Mesh::from(prim.vertices)`"*
cited as `crates/client-render/src/lib.rs:820-821`. Section 2.3 and section 11 slice 2 then say the
smooth lane emits *"`MeshPrim`s (widened with normals)"*.

**What I measured.**
- `Vertex` ALREADY carries a normal: `pub pos: [f32; 3]`, `pub normal: [f32; 3]`
  (`crates/client/src/realm_scene.rs:770-773`), and the existing proxy already writes one
  (`crates/client/src/realm_scene.rs:887`). **No widening is needed.**
- `crates/client-render/src/lib.rs:820-821` is the reference ground plate, a `Cuboid`. The real site is
  `mesh_from_prim` at `crates/client-render/src/lib.rs:1824-1829`.

**Why it matters.** The error runs the wrong way for the report's own case: the seam is ALREADY what V2.8
needs, which strengthens the engine-agnostic argument. But a false `file:line` in a table headed "what
exists in the code today" is the exact failure the standing rule about reading the code exists to stop.

**The fix.** Correct both citations, and delete "widened with normals" from section 1, section 2.3 and
slice 2.

---

## 12. UNMEASURED_AS_FACT (SL8) — the report says the arrival pop is visible and then defers the cure.

**The claim.** Section 5.2, the arrival-pop row: *"the bound is 2 cells = 4 px (ESTIMATED from the
derived bound). That is visible in a dither."* Then: *"Reserve it; do not build it first."* D9 recommends
geomorphing.

**Why it is a problem.** SL8 says a seam is a defect, and the standing rule is to never ship a capability
without its continuity. The report states in its own words that the pop IS visible and then puts the cure
after the slice. Slice 3's gate is the no-pop detector, so the slice cannot land red — that saves it —
but the text tells an implementer to build a known-visible pop first.

**The fix.** Fix the order, not the gate. Run M4 BEFORE slice 3 builds the ladder. If M4's residual
exceeds the detector's threshold, geomorphing is part of slice 3, not a reserve.

---

## 13. MISSING — the coarse pyramid's standing bytes are never measured.

**What is absent.** Section 5.3's example needs a distant client to hold tier-2 pyramid entries for a
whole moon so that a week-old tunnel shows as a dimple in the ridge. Section 5.2's lane-flood row admits
*"the coarse pyramid entries ride the same lane"* and marks the tolerance UNMEASURED. But M8, the only
bench for that lane, measures one player digging 10 cells per second for one minute. Nothing measures
the STANDING cost: how many bytes a client must receive and hold, at every coarse rung, for a moon with
a year of edits on it, at first sight.

**Why it matters.** SL8's lane-flood and re-state-rate tolerances are physical, and SL9's discipline says
a cost that grows must be measured on a realm that has many, never argued.

**The fix.** Add a bench. Seed a moon with N edited cells scattered over its whole surface, for N across
four orders of magnitude. Report the pyramid bytes the client must hold at each rung, and the bytes on
the wire when the moon first enters the window.

---

## 14. MISSING (low) — section 10's stale register omits one base ruling the report contradicts.

The base rules *"the base cell is never subdividable"* and *"Below the catalogue's smallest extent …
there is no cell and therefore no collider, ever"* (`docs/investigation/decision_board.md:132-142`).
Section 4.3 recommends a `SubGrid` form under V2.4. That is a direct contradiction and it belongs in
section 10's table, with V2.4 as the superseder and the sub-metre domain as the owner.

---

## What STANDS

I tried to break these and could not.

- **SL1 — a realm is told where it is; it never decides.** Nothing in this domain names a realm's own
  placement. A chunk key is an address in the realm's OWN frame, and no chain folds an absolute. The
  parent still ships only the moon's placement (section 9).
- **SL2 — no occupant pose crosses a realm boundary.** The terrain diff carries a cell, never a person.
  The report correctly refuses the hull-on-terrain contact as another domain's SL6 ask (section 3.5) and
  states what doing without costs.
- **SL3 — a realm draws itself.** The parent ships the moon's placement and nothing else; the moon's own
  cells decide how it looks. Finding 9 is the one gap, and it is an absent rule, not a broken one.
- **SL4 — physics and re-home are separate, one-way.** No orbit, gravity or thrust symbol appears on any
  crossing path in this design.
- **SL7 and SL9 — no per-child per-tick cost, no bounded child set, no scan.** The collider residency is
  a radius query around a body, never a walk of a parent's children. The tier is chosen per chunk column,
  never per child.
- **The movement contract.** No velocity crosses upward, no parent sets a speed, and the walkability
  threshold of D7 is a contact constraint on a slope, not a speed a parent states. The 2026-08-27 answer
  (4) fences a speed only where CODE states one, and nothing here does.
- **SL5 — one world.** The measurement rig in section 6 is named as a stand and scheduled for deletion
  the moment the generator crate exists. That is the right disposal, though section 6's own table still
  gates slice 1 on a bench that lives outside the crate; run the crate's own bench instead.
- **HR1 — sealed shards.** No shard reaches into another's cells.
- **The re-validated base facts.** I checked the report's re-validations against the base and they hold:
  the 23 topologies (`block_system_design.md:5661-5686`), the orientation contradiction between 5 bits
  (`:710-720`) and 6 bits (`:5379-5382`), the cube-only terrain ruling (`:7072-7080`), the iso-surface
  rejection (`:8998-9007`), the 0.885 ms strategy-C measurement (`:17414-17423`), the unit-of-matter rule
  (`:5633-5638`), and the hull-shape SL6 ask (`decision_board.md:814-817`). I also confirmed the report's
  own code facts: no terrain, chunk or mesher code exists; `FrameSpace`, `reanchor` and `AnchorGen` are
  comments only (`crates/sim/src/capability.rs:11,18`); rapier and parry are not dependencies; `noise`
  is declared at `Cargo.toml:78` and used by no crate; `SplitMix64` is the one hash
  (`crates/core/src/rng.rs:11-29`); the store stamp folds f64 constants only
  (`crates/core/src/store_stamp.rs:117-123,301-307`); and `coarsen_level` survives only in a tombstone
  (`crates/wire/src/intershard.rs:1173-1186`).

## The eight items to fix, in order

1. Make the SL6 ask for the client-facing terrain diff arm (finding 1).
2. Correct the density's sign to `r − h` before door 2 shuts (finding 2).
3. Change the quantum from 1/127 to 1/128 before door 1 shuts (finding 3).
4. State the boundary between seed-decided common substance and live-state deposit (finding 8).
5. Give the collision rule its fourth arm for a seed-shaped surface object (finding 7).
6. Tie the client's resident tier to a body's contact, and gate it with a landing fixture (finding 6).
7. Let a Cartesian realm hold terrain-form cells, so the HR4 fixture is real (finding 4).
8. Collapse the two cube-lane collider builders into one (finding 5).
