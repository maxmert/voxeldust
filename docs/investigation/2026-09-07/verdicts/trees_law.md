# Verdict — the law refuter for "Trees, vegetation and composites"

**Report under test:** `docs/investigation/2026-09-07/07_trees_composites.md`
**Lens:** the binding law — HR1–HR6, SL1–SL10, SL8 (the seamless law), the movement contract, the
2026-08-27 seed ruling, the reach and visibility rulings — and the code as the only current truth.
**Date:** 2026-09-07.

**Result: REFUTED.** Three load-bearing claims fail. The report's storage claim rests on a record
layout that the record domain does not recommend. The report's seam-free chop is a declared rule that
the two lanes in the code cannot keep. The report's falling tree holds two properties that cannot both
be true. Ten more findings follow. The rest of the design stands, and I say where.

I read every citation in the code. Each finding carries one verdict word:

- **WRONG** — the claim is false, or the report contradicts itself.
- **BREAKS_LAW** — the claim conflicts with a binding rule.
- **UNMEASURED_AS_FACT** — the report states a result that it did not measure.
- **MISSING** — the design has a hole that the report does not name.
- **STANDS** — I tried to refute the claim and I could not.

---

## 1. The findings that refute the report

### F1 — WRONG. The tree record does not fit the record the record domain recommends

The report says: *"The record is the ordinary saved block record that domain 04 freezes"*, and it lists
`cell 18 | block type 16 | orient 6 | provenance 2 | state 8 | reserved 14`
(`07_trees_composites.md:77-81`). It then spends the state byte three times: as the growth stage
(`:74`, `:250-252`), as the object SIZE (`:362`), and as a stage that runs `0..255` (`:235`).

Domain 04 recommends a different record. It has no state byte:

```
bits 21..16  variant     (6)   the style index, V2.6
bits 15..14  sub_scale   (2)   V2.4
bits 13.. 5  sub_addr    (9)   V2.4
bits  4.. 0  reserved    (5)   MUST be zero
```
(`04_block_record_registry.md:112-136`; `:247-256` records where the fifteen spare bits went.)

The eight-bit state field exists only in the investigation base
(`block_provenance_collapse.md:137-147`), which the law says is not binding and which describes a past
state. So the report attributes to domain 04 a layout that domain 04 does not hold. **"A tree record is
eight bytes and needs no field the record does not already reserve" (`:80-81`) is not supported.** A
growth stage of 0..255 needs eight bits; five bits stay reserved, and the six `variant` bits are already
spent on V2.6 style.

**Example.** A player plants a pine. Under the report the pine's record carries growth stage 0 in the
state byte. Under domain 04's record there is no state byte to carry it, so the pine's stage has
nowhere to live and the moon's shard cannot advance it.

**Fix.** Re-cost the tree record against domain 04's layout, and choose one of three by name: put the
stage in the side table the report already asks for (`:250-254`); ask domain 04 for eight bits as a
named one-way door before the record freezes; or state a stage that fits five bits. Then re-state the
storage arithmetic of §2.2 and §8 item 6, which all read "8 B per tree".

---

### F2 — BREAKS_LAW (SL8) and WRONG. The chop cannot be atomic on the lanes that exist

The report states a rule: *"an object diff and an entity spawn that share a tick are applied atomically
by the client. There is never a frame with no tree ... and never a frame with two trees"* (`:324-328`).
Three sentences later it says the topple *"animates at the entity lane's rate through the client's
100–150 ms interpolation buffer"* (`:329-330`).

The code shows two lanes with no shared order:

- The chunk diff rides the RELIABLE per-subscription bulk lane
  (`crates/wire/src/channels.rs:283-296`, `BulkMsg` and `BulkKind`).
- The entity spawn rides the UNRELIABLE 20 Hz world-state datagram
  (`crates/wire/src/channels.rs:450`).
- The client then holds every entity 100–150 ms in the interpolation buffer
  (`crates/client/src/view.rs:51`, `crates/client/src/tuning.rs:10`, `CLAUDE.md:261`).

So the stump arrives on a reliable stream and is drawn at once, and the falling crown arrives on a
datagram and is drawn a tenth of a second later. **That is the black-frame seam and the flicker seam
the report claims to prevent, and the report measures neither.** Declaring the rule does not build the
mechanism.

**Example.** A player chops an oak. The stump appears. For about three frames the oak's crown is gone
from the sky. Then the crown appears, already leaning.

**Fix.** Make the object diff carry the tick it belongs to, and make the client apply it at the instant
the interpolator shows, not on arrival. Add a measurement to §9: chop an oak 1 000 times, and assert
that no frame shows a stump without a crown and no frame shows two crowns.

---

### F3 — WRONG. A falling tree cannot both collide and land where the seed says

The report recommends that a falling tree collides while it falls (`:305-308`, and open decision 3 at
`:573`), and it keeps the base's closed-form topple, quoting *"a final pose that does not depend on the
path"* and *"live and dormant paths writing identical bytes"* (`:299-301`).

Both cannot hold. If the crown meets a parked hull, either the crown stops — and its final pose now
depends on the path, which destroys the property that makes the dormant moon and the live moon write
the same bytes (`block_provenance_collapse.md:508-515`) — or the crown does not stop, and it passes
through the hull, which is the phantom miss the report's own divergence contract forbids (`:116`).

The report also misreads why the base refused the collider. It says the base chose no collider
*"because its collider was thousands of voxels (§4.5 line 503)"* (`:306-307`). Line 503 reads
*"No physics body, no collider, no float"* and gives no size reason; the base's reason is the
closed-form determinism of the schedule (`block_provenance_collapse.md:500-515`).

**Example.** A pilot parks a hull under an oak and chops the trunk. The crown falls onto the hull. The
report does not say whether the crown rests on the hull or sinks through it, and the two answers give
two different worlds when nobody watches.

**Fix.** State which property the design keeps. Either the topple stays scripted and the collider is a
one-way pusher that never changes the tree's own pose — then say so, and say what a hull under a falling
crown feels — or the pose reacts, and §4.5's identical-bytes property is retired by name.

---

### F4 — BREAKS_LAW (SL8). An unbounded phantom hit is accepted for art-pack kinds

SL8 says a seam is a defect, and that tolerances are PHYSICAL. The report accepts, with no bound:
*"a thin branch inside a metre capsule is a phantom hit (allowed)"* (`:407`). A player who stops one
metre short of anything drawn has walked into nothing. That is a jump seam under any physical
tolerance, and open decision 2 recommends shipping route (b) for some kinds (`:572`).

The gate the report owes tests membership, not distance: *"every tube vertex of the reference mesh lies
inside its capsule"* (`:536`). A vertex one metre inside its capsule passes that gate.

**Example.** A pilot walks toward a birch from an art pack. She stops a metre before the bark, in open
air, with nothing between her and the trunk.

**Fix.** State the physical bound: the collider surface lies within N centimetres of the drawn surface
everywhere a player, a suit or a landing gear can touch it. Make measurement 3 assert that distance,
and refuse an archetype table that cannot meet it.

---

### F5 — WRONG. The collider comparison is not like for like, and the section names two capsule sizes

The cost table (`:148-157`) puts these two rows side by side:

| The report's row | What it actually measures |
|---|---|
| "collision bytes per disc — ~0.2 MB" | a 64 m disc, 12 868 m², 129 trees |
| "the base's number for the same disc — 6.9–22.5 MB" | a 64 m physics CLUSTER, nine surface chunk columns, 34 596 m², 346 trees (`collidable_decoration.md:1004-1025`) |

It is not the same disc. The areas differ by 2.7 times and so do the tree counts. The report's own
footnote says so, while the row header says the opposite. Per tree the base is 20–65 KB and the report
is 1.5 KB — a gain of 13 to 43 times, not the 35 to 112 times the raw rows suggest.

The same section states two capsule sizes: *"16 capsules × ~100 B"* (`:152`) and *"a capsule holds a
10 m limb in 28 bytes"* (`:160-161`). They differ by 3.5 times.

The conclusion — the collider budget becomes a rounding error — survives. The arithmetic that supports
it does not.

**Fix.** Normalise per tree or per square metre, pick one capsule size, and re-state §1.6 and stale
claim 7 (`:506-507`).

---

### F6 — WRONG. Two units for one length, and the report freezes that unit as a one-way door

§1.3 puts a skeleton's segments on *"a 1/1024 m integer lattice"* (`:98-99`). §1.4 says *"Lengths and
radii are integer millimetres"* (`:131`). A millimetre and 1/1024 m are different units. §10 then makes
*"the skeleton format (segment lattice units, radius units...)"* a door that must shut before the first
client links the crate (`:560`), so the ambiguity sits inside the frozen part.

Domain 04 records that the coordinate ruler's finest rung is 2⁻¹⁰ m (`crates/core/src/pose.rs:475-479`,
cited at `04_block_record_registry.md:190`), so 1/1024 m is the unit that sits on the ruler.

**Example.** A birch's first limb starts 1 800 units up the trunk. On one reading that is 1.800 m; on
the other it is 1.758 m. The moon's capsule and the client's tube then disagree by 42 mm.

**Fix.** State 1/1024 m once, in §1.3, and delete the millimetre wording.

---

### F7 — WRONG and UNMEASURED_AS_FACT. A stored timestamp is denied and then used, and the growth tick has no budget

§2.4 says: *"no TIMESTAMP is stored (growth is memoryless)"* (`:238-239`). §2.3 says the shard *"draws
the elapsed stages from a binomial over the chunk's last-ticked stamp"* (`:220-221`). A per-chunk
last-ticked stamp is a stored timestamp. The "stores nothing" claim is false as written, and the byte
that carries the stamp is costed nowhere.

The slow tick also has no cost model. §2.3 and §2.4 make the realm's shard step every diverged cell
toward the generator's baseline, which means it evaluates the generator at every such cell on every
slow tick. The ten measurements of §9 (`:530-550`) contain none for it.

**Example.** A moon holds a million cells that players cut or planted. Every slow tick the moon's shard
evaluates the baseline for all of them. Nobody has measured that cost, and no budget refuses it.

**Fix.** Name the stamp, say which domain stores it, count its bytes, and add a measurement: the slow
tick's p99 on a moon with a million diverged cells.

---

### F8 — WRONG. A planted tree is not permanent, under the report's own instance-seed rule

§2.3 says: *"A planted tree never equals the generator's baseline at that cell (the baseline says air,
or another kind), so its record never prunes. It is permanent, as R8 says"* (`:224-226`).

The parenthetical assumes what it needs. Open decision 1 recommends deriving the instance seed from the
cell address (`:571`), and §1.2 gives a planted tree the same provenance `Feature` as a seed tree
(`:75`). So a player who cuts the seed oak and replants an oak on the same cell grows a tree that is
byte-identical to the baseline once it matures, and the prune-on-equality rule (`:235-236`) deletes the
record.

The claim in §2.4 that *"the provenance clause keeps the prune safe: a `Placed` cell can never equal a
`Feature` baseline"* (`:240-241`) protects nothing here, because the report puts planted trees on the
`Feature` side.

**Example.** A logger fells an oak beside a landing pad and plants an oak back in the same hole. Weeks
later the oak's record deletes itself. In the world nothing changed, but the report promised the record
was permanent, and R8's stored growth stage was the reason.

**Fix.** Choose. Write a planted tree `Placed` and accept that a replanted forest never prunes, or drop
the permanence claim and say that a replanted tree becomes a seed tree again. State which, because §2.4
leans on it.

---

### F9 — BREAKS_LAW (SL6). "None found" reads only half of SL6

SL6 says: *"ASK BEFORE NEW DATA CROSSES A REALM BOUNDARY, **and before adding a wire arm.** Default
NO."* §12 answers the first half only and reports **None found** (`:582-592`). The design adds at least
three wire-visible things and requests none of them:

1. A new entity kind for the falling group. The report names it and picks tag 14 (`:302-303`). The kind
   tag lives in `EntityId` bits 120..128 and the table is exhaustive
   (`crates/core/src/entity_kind.rs:27-41`).
2. The falling group's parameter blob — kind, instance seed, stage, size, cut height (`:297-299`).
   Entity blobs are TLV-framed, and decode-to-Default is banned for Durable kinds (`CLAUDE.md:265-267`).
3. The object diff at every rung (`:261-265`). Domain 04 already places cell, attachment and box TLV
   tags inside `BulkKind::ChunkDelta` (`04_block_record_registry.md:645-647`), so object rows are new
   tags on that lane, not free.

`ObjectDraw` (`:419-426`) is the client library's own output and is not a wire arm. That one is fine.

**Fix.** List the three under a law_conflicts heading, with the data, the lane and the cost of doing
without, so the owner can answer them. SL6's default is NO.

---

### F10 — MISSING. Nothing ties the derived forest to a LIVE realm

SL3 says a realm that is not running cannot be drawn, and that is why visibility is the spin-up trigger.
SL10 now lets the client derive a moon's static shape on its own. The report never states the rule that
joins the two.

Without it a client draws a dormant moon's seed forest from its own copy of the generator, and a dormant
moon ships no object diff, so every tree that players felled stands again. That is the arrival-pop the
report says it prevents (`:481`). §2.5 comes close — *"a client that holds a chunk at any rung also
holds that chunk's object diff"* (`:259-261`) — but it never says that a live realm is what ships that
diff.

**Example.** A pilot coasts past a moon that nobody woke. The client draws 214 oaks from the seed. Two
of them fell last week, and the pilot sees them standing.

**Fix.** State the rule: a client draws a realm's derived objects only where it holds that realm's
object diff, and only a live realm ships one. Add it to measurement 9.

---

### F11 — MISSING. The seed ruling is never applied to the object KIND draw

The 2026-08-27 ruling binds the block system before it starts: *"a block's substance may not be a pure
function of (position, seed)"*, because a fixed seed plus a position is a public treasure map.

§2.1 step 1 draws *"an anchor cell, a kind, a size class"* purely from
`SplitMix64(child_seed(realm seed, FEATURE_SALT, region))` (`:188-189`), and §3.3 turns kinds into drops
through the kind's yield table (`:312-316`). So a rare kind that drops a valuable substance is a pure
function of seed and position. §12 does not check it.

**Example.** V2.7 adds a crystal spire to a themed planet, and the spire drops a valuable crystal. Every
spire in the universe then stands at a seed-derived address, and one player publishes the list.

**Fix.** State the fence beside the instance-seed door in §10: a seed-drawn object kind may drop COMMON
substance only, and anything valuable is live state on the record that arrives as a diff. Then a spire
may stand where the seed says, and what it holds is decided by the world, not by the seed.

---

### F12 — WRONG. The file count is wrong, and it is a MEASURED claim

The report says `crates/core/src/` holds *"24 files"* (`:22-24`). MEASURED on 2026-09-07 in this
worktree with `ls crates/core/src | wc -l`: **26**. The claim that none of them is a voxel module is
true; I checked every name.

**Fix.** Say 26.

---

### F13 — WRONG. The nesting-fence citation points at the ruler, not at the fence

The report cites *"the nesting fence refuses a child that pokes out of its parent
(`crates/core/src/geometry.rs:350-368`)"* (`:172-173`). Lines 350-368 hold
`Boundary::circumscribed_extent` and the doc comment of the exact-reach helper. The fence is built from
that pair elsewhere (`crates/core/src/geometry.rs:4328`). The idea is right; the line range is not.

**Fix.** Cite `circumscribed_extent` as the reach helper, and say the boot-time fence is built from it.

---

## 2. What I tried to refute and could not (STANDS)

- **SL1 — a realm is told where it is, and never decides it.** A tree is not a realm; it is part of the
  moon's own look. No chain folds an absolute anywhere in the report. **One wording note, not a
  finding:** §5.2 calls the anchor's place *"the anchor cell's composed position, from the server or
  derived"* (`:425`). The word "composed" belongs to the gateway. The next sentence says *"The client
  never derives a pose"* (`:437`), so the intent is lawful; the words should read "in the realm's own
  frame".
- **SL2 — no occupant pose crosses a boundary.** The note on a hull that lands in a forest (`:590-592`)
  is right: the planet's shard collides the hull with the planet's capsules, and the hull's own realm
  learns nothing about the trees.
- **SL3 — a realm draws itself.** The trees belong to the moon's own look, and no parent describes them.
  (The dormant case is the hole; see F10.)
- **SL4 — the crossing path never names motion.** No orbit, gravity or thrust symbol appears on a
  crossing path here. The falling group crosses as an ordinary entity.
- **SL5 — one world.** §8 item 10 and open decision 8 refuse the base's test-only three-boulder
  generator by name, and plant a real object through the shipped placement path (`:514-517`, `:578`).
  That is right and well argued.
- **SL7 and SL9 — no per-child per-tick cost, no scan, no bounded child set.** An object's dirty set is
  its own anchor cell (`:377-379`), and object shapes sit in a per-chunk list keyed by the anchor cell
  (`:170-171`). Nothing walks a realm's children. (The growth tick's cost is a separate hole; see F7.)
- **HR3 and HR4 — no match on shard kind.** The up-face rule makes a potted tree in a hull the same
  record, the same expansion and the same capsules (`:174-177`), and measurement 10 owes the fixture on
  a planet and in a hull (`:548-550`).
- **The movement contract.** No velocity crosses upward. A falling tree governs nothing itself, so the
  containing realm owning its speed is exactly the 2026-08-27 answer on the ungoverned rock. No parent
  states a speed to a child, and no re-clamp appears.
- **SL10 V1.2 and V1.4 — one generator crate, integer draws, no libm.** §1.4 matches the code it cites:
  `SplitMix64` and `child_seed` are integer-only (`crates/core/src/rng.rs:12-27`, `:66-72`). §1.5's
  skeleton gate is a real measurement that could fail, not an argument.
- **The report's code citations.** I checked every one, and they are true except F12 and F13: the
  discovery-permanence law (`crates/physics/src/worldgen.rs:17-36`), the seven entity kinds
  (`crates/core/src/entity_kind.rs:33-41`, with tag 14 free in the 10..20 transient band), the bulk lane
  (`crates/wire/src/channels.rs:283-296`), `BlockEdit` reserved and not built
  (`crates/wire/src/intershard.rs:34-35`), the render seam
  (`crates/client/src/realm_scene.rs:769-793`), the shape-free mesh build
  (`crates/client-render/src/lib.rs:1828-1846`), `Boundary`
  (`crates/core/src/geometry.rs:338-348`), `TAG_LOOK` (`crates/core/src/look.rs:31`), D-38 with
  `FrameSpace` and `reanchor()` still absent (`docs/design/DEFERRED.md:298-353`), `noise = "=0.9.0"`
  declared and unused (`Cargo.toml:78`), and no `rapier`, `parry` or `noise` package in `Cargo.lock`
  (MEASURED by grep; count 0).

---

## 3. What the report must do to stand

1. Re-cost the tree record against domain 04's actual layout, and say where the growth stage lives (F1).
2. Build the atomic chop, do not declare it, and measure it (F2).
3. Choose between a colliding fall and a path-independent landing (F3).
4. State the physical bound on a phantom hit, and gate it (F4).
5. Normalise the collider comparison, and pick one capsule size (F5).
6. State the skeleton's unit once (F6).
7. Own the chunk stamp, and budget the growth tick (F7).
8. Fix the planted-tree permanence claim (F8).
9. List the three wire additions as explicit SL6 requests (F9).
10. Tie the derived forest to a live realm (F10).
11. Fence the seed-drawn kind against the 2026-08-27 seed ruling (F11).
12. Correct the file count and the fence citation (F12, F13).
