# Verdict — the completeness of `00_proposed_voxel_foundation.md`

**Date:** 2026-09-07. **Role:** the completeness critic.
**Subject:** `docs/investigation/2026-09-07/00_proposed_voxel_foundation.md` (1 046 lines).
**Read against:** the ten revised domain reports `01`–`10` in the same directory, `CLAUDE.md`,
`docs/design/owner_decisions_2026-09-07_voxels.md`, and the code in this worktree.

**The verdict in one line.** The synthesis is NOT refuted. Its design holds. Its formats hold. But it
drops, flattens or contradicts twenty-two items that its own source reports carry, and four of those
are format-level. The owner cannot freeze the three formats on this document as it stands, because two
starred format doors shut before the measurements that decide them, one gate contradicts a frozen
format, and one number the design rests on is marked MEASURED when its own source says UNMEASURED.

**How to read a verdict.**

- **UNMEASURED_AS_FACT** — the synthesis marks a number MEASURED, and the source report says nobody ran it.
- **CONTRADICTS_REPORT** — the synthesis states something a source report refutes, or two reports
  disagree and the synthesis picks one silently.
- **WRONG** — the synthesis contradicts itself, or contradicts the code.
- **MISSING** — a door, a decision, a case, a gate or a measurement exists in a report and not here.
- **BREAKS_LAW** — a hard rule or a standing law refuses it.
- **STANDS** — I tested it and it holds.

**How to read a number.** MEASURED means a program ran and I say which. ESTIMATED means arithmetic on a
cited number. UNMEASURED means nobody ran it.

---

## 1. The refusals — a wrong answer here costs a migration or a seam

### F1. The write amplification is called MEASURED and nobody ran it

**Verdict: UNMEASURED_AS_FACT.**

The synthesis says *"the base's shape pays a MEASURED 5 508× write amplification per entry"*
(`00_proposed_voxel_foundation.md:304`) and repeats it in the register (`:684`, V52).

Its source says the opposite. Report 06 lists the figure in a table of inherited numbers and then
states, five lines below the table: *"Every number in the list is UNMEASURED on this codebase"*
(`06_storage_diff_lane.md:391,396`). The 5 508× comes from an older document
(`storage_and_streaming.md:1057-1086`), not from a run on this tree.

**The cost.** The number carries V52. V52 decides where the edit pyramid lives, and slice 9 builds the
checkpoint, the watermark and the tail replay on that answer. The owner would freeze a store shape on a
number that is an inherited estimate, against the standing rule that a claim of this weight must be a
measurement that could have failed.

**The fix.** Mark it ESTIMATED. Keep the recommendation (the checkpoint) and state its real ground: the
pyramid is derived and rebuildable, so it does not need the write-ahead log's crash guarantee. Add the
run to §7 as a new row, beside U-26.

*Example: a moon holds a quarry. Every mined cell writes one tier-0 row and folds up thirteen rungs. The
document tells the owner that the write-ahead route costs 5 508 writes per rung entry, as if a bench
printed it. No bench printed it.*

---

### F2. A felled tree costs eight bytes, and the frozen record is twelve

**Verdict: WRONG (the synthesis contradicts its own Format B).**

- §1 says *"Felling is an 8-byte removal diff"* (`:81`).
- Slice 14's gate says *"fell 1 000 seed trees and assert 8 B ± 0 of permanent record each"* (`:562`).
- Format B says *"BlockRecord — 12 bytes: word A (u64) then word B (u32)"* (`:169`), and ★V5
  recommends *"twelve bytes fixed"* (`:627`).

The eight-byte figure comes from report 07 (`07_trees_composites.md:331,863`), which was written
against the older eight-byte record. Report 04 then widened the record to twelve
(`04_block_record_registry.md:107-140`, D-12 at `:836`) and the synthesis adopted twelve. The tree
number did not follow.

**The cost.** Slice 14's gate is written to fail on the shipped format. A gate that must fail is worse
than no gate, because somebody will "fix" it by loosening the assertion instead of by reading the
format.

**The fix.** Restate the gate as *"12 B ± 0 of permanent record per felled tree, and 0 B of side row"*,
and correct §1. Then re-derive every tree byte figure that rests on eight: report 07's clear-cut disc
of 19 406 trees becomes 233 KB, not 155 KB (ESTIMATED, arithmetic on 12 B).

*Example: a player fells a thousand oaks on a moon. The moon writes one air record per stump. The gate
counts 12 000 bytes and the document expects 8 000.*

---

### F3. The formats hold no NEGATIVE entry, and two reports ask for one

**Verdict: MISSING (a format-level hole).**

Report 10 names it as a case the base never asks: *"the diff must carry a **deletion tombstone against
derived shape** … the diff format needs a NEGATIVE entry, and the diff is one of the three formats V3.2
freezes"* (`10_law_audit_stale_register.md:459-464`). Report 06 asks the same question as a decision:
*"D-8 — Encoding of 'a cell returned to the seed' on the wire — (a) a revert list tag; (b) a flag bit in
the cell record"*, recommending (a) (`06_storage_diff_lane.md:890`).

The synthesis carries neither. Format B has provenance values `0 Terrain, 1 Feature, 2 Placed` and no
removal class (`:181`). Its decode rule mentions *"an `Air` record with any other non-zero field"*
(`:206`), so an `Air` KIND appears to be the removal, but no section says so, and report 06's D-8 does
not appear anywhere in the register V1–V100.

**The cost.** The client derives a hill and an oak from the seed. Every removal must beat the derivation
for all time. If the encoding is decided after the first store is written, every tunnel and every stump
in the world re-encodes.

**The fix.** Add the removal encoding to Format B as a named field value and to Format C as a named
fold, and add report 06's D-8 to band A as a starred question. State whether a removal is an `Air` kind
record, a provenance value, or a separate revert tag on the wire.

*Example: a player cuts down an oak on a moon. A week later her client re-derives that chunk from the
seed. The oak must stay down, and the byte that keeps it down has no home in this document.*

---

### F4. A sub-metre block on a smooth slope has no seat

**Verdict: MISSING (a format-level hole, and a case).**

Report 10 raises it as a new contradiction and says the answer decides a saved field: *"On a smooth
slope the surface does not follow a cell face, so a block placed on a sub-cell LATTICE either floats
above the ground or sinks into it. The two candidate answers are two different saved formats"*
(`10_law_audit_stale_register.md:441-452`). Its recommendation is a lattice offset for the ADDRESS plus
a *"client-and-server-shared SEATING rule that drops the block onto the derived-plus-diff surface inside
its cell, so the seat is DERIVED and never stored"*.

The synthesis has the address (`sub_scale` and `sub_addr`, `:184-185`) and V39 (a sub-metre block may
sit in a planet cell, `:669`). It has no seating rule. The word "seat" does not appear in the document.

**The cost.** V2.4 is an owner requirement. Without the seating rule a quarter-metre lamp post on a
hillside floats or sinks, and the seat is either stored (which re-binds when the generator version
moves) or invented twice — once by the moon's shard and once by the client — which SL10 V1.2 forbids by
name.

**The fix.** Add the seating rule to slice 6 (the extractor and the composition order), because it is
part of the composition and it must live inside the one generator crate. Add it to Format B's door table
as *"the sub-site is a LATTICE OFFSET and the seat is DERIVED"*.

*Example: a player sets a 25 cm lamp post on a moon's hillside. The lamp's cell is a lattice address.
Both the client and the moon's shard drop the lamp onto the same slope by the same rule, so the boots
and the picture agree.*

---

### F5. Two starred format doors shut before the measurements that decide them

**Verdict: WRONG (a self-contradicting sequence).**

- ★V7 (the sub-block step and the address width) says *"the door must not shut before that frame cost is
  measured"* (`:629`). Its measurement is U-35, and §7 files U-35 under *"Before slice 9–12"* (`:912`).
  The sub-lattice width is a band-A answer, and slice 0 collects band A (`:338-343`). So the width
  freezes at slice 0 and the number arrives at slice 12.
- ★V20 (the sticky "differs from the seed" bit) says *"(A) until the fly-away gate measures the drawn
  step"* (`:642`). Its measurement is U-33 (`:910`), and the fly-away gate is slice 10 (`:498-500`).
  Format C's door table says the bit must be decided *"before the first world is saved"*, and slice 9
  writes the first store.

**The cost.** The owner must answer a permanent question at slice 0 with a stated condition the schedule
cannot meet. The document does not say what happens when the number arrives and disagrees.

**The fix.** Do one of two things for each, and say which:

1. Pull the measurement forward into a bench spike that runs before slice 0's sitting. U-35 needs a
   collider and a mesher, so this is a spike, not a slice.
2. Freeze the WIDTH now and mark the STEP and the bit as PROVISIONAL, with a stated re-open rule: what
   the owner may still change after slice 12, and what the change costs.

*Example: a player tiles forty thousand cells of her hull at eighth scale. Nobody has measured what one
such cell costs the shard's tick. The document asks the owner to freeze the address width for that cell
today.*

---

### F6. Slice 17's acceptance gate cannot go red

**Verdict: MISSING (the tolerances are dropped).**

The synthesis's slice 17 lists the twelve seam kinds and then says *"Measurement: one number per seam
kind, each with a PHYSICAL tolerance, recorded as a baseline that a later slice must not regress"*
(`:604`). It states no tolerance. A baseline is a record of what happened. A first run against a
baseline cannot fail.

Report 08 states a physical tolerance for every kind, and marks each one ESTIMATED where nobody
measured it (`08_render_seam_seamless.md:525-600` and on). Four the synthesis drops:

- Jump: *"a drawn chunk's screen position across the swap moves by less than one pixel"*.
- Black frame: *"zero frames in which a resident region has no rung drawn"*.
- Brightness pop: *"the frame's mean luminance changes by less than one eighth of a stop between
  successive frames outside a physical cause"*.
- Arrival pop: *"at the hand-over distance the coarse rung's silhouette differs from the sphere by less
  than one pixel"*.

**The cost.** V2.9 and SL8 are the owner's hardest requirement. The one slice that proves them ships
with no pass condition.

**The fix.** Copy report 08's twelve tolerances into slice 17, each marked MEASURED or ESTIMATED, and
name the run that turns each ESTIMATED one into a threshold.

*Example: a pilot flies from orbit to a walk on the ground. The gate records twelve numbers and passes,
whatever the numbers are.*

---

### F7. "Edit echo to pixels ≤ 33 ms" is a budget from a different mesher

**Verdict: UNMEASURED_AS_FACT.**

Slice 10 says *"Measurement: edit echo to pixels ≤ 33 ms"* (`:503`). Slice 12 repeats it as *"against
the 33 ms edit-to-pixel budget"* (`:535`), and V71 rests on it (`:703`).

Report 09 refuses the figure in its own words: the 33 ms is *"a BUDGET … and it was a budget for
remeshing CUBE faces. Under V2.1 the client extracts a smooth surface from a density field, which is a
different and heavier job"* (`09_physics_controller.md:589`). The same section measures the network half
of the path at ~101 ms ESTIMATED and says click-to-pixel is *"quotable only after S19"*. The synthesis's
own U-54 says *"no click-to-pixel figure may be quoted to the owner before this runs"* (`:936`).

**The cost.** The document quotes a click-to-pixel threshold it forbids itself to quote. Two slices gate
on it.

**The fix.** Split the number. Name the CLIENT re-extract budget as the gate, and let U-54 fill it. Name
the ~101 ms echo path as ESTIMATED and separate. State no whole-path threshold before U-54 runs.

*Example: a miner breaks a rock beside a hull's ramp. The document promises she sees the hole inside 33
milliseconds. The message alone takes about a hundred.*

---

### F8. Slice 7 links a motion crate into the client

**Verdict: WRONG (it contradicts the code and the slice's own gate).**

Slice 7 says it lands *"`vd-physics`'s successor moved from `[dev-dependencies]` to `[dependencies]` in
`crates/client/Cargo.toml`"* (`:448`).

The code refuses this in its own words. `crates/client/Cargo.toml:20-22` reads: *"DEV-ONLY: scene tests
build THE world; the shipped client never names a motion (SL4)."* And `crates/physics/src/motion.rs`
exists (MEASURED by `ls crates/physics/src/`), so `vd-physics` IS a motion crate. Slice 7's own gate
demands *"a structural control that FAILS when the client-linked crate's surface grows a placement or an
orbit symbol"* (`:459-461`), which `vd-physics` fails on the first day.

The crate the client must link is `vd-terrain`, which slice 5 creates as a NEW crate over a `vd-seed`
leaf (`:412-416`). `vd-terrain` is not `vd-physics`'s successor: slice 5 states that `vd-physics` keeps
the forest and READS the body radius from `vd-terrain` (`:420`).

**The cost.** As written, slice 7 puts orbits and thrust in the shipped client binary. That is SL4 and
the client-only-renders law in one line.

**The fix.** Rewrite the line: *"`vd-terrain` added to `[dependencies]` in `crates/client/Cargo.toml`;
`vd-physics` STAYS a dev-dependency, and the crate-isolation test gains a row that refuses a client
dependency on it."*

*Example: a pilot's client draws a moon's hills from the seed. The same binary must not hold the code
that moves the moon.*

---

### F9. Nothing puts the SHARD's chunk build off the tick thread

**Verdict: MISSING (a named seam with no slice).**

Report 09 names the rule and the seam: *"A descent demands tens of chunk builds in one tick at an
ESTIMATED 0.5 ms each. If the build runs on the tick thread that is a TICK HITCH, which SL8 lists among
the eleven seams"*, and then *"The generator evaluation, the diff application and the extraction run on
a worker pool, off the tick thread"* (`09_physics_controller.md:668-680`). It names an already-present
pool, `bevy_tasks` (VERIFIED: `Cargo.lock:1570`), and says that rayon is a new library and needs the
owner's word.

The synthesis answers only the CLIENT side. V44 asks *"Who evaluates chunks on the client"* (`:671`) and
V45 picks the client's pool mechanism (`:672`). Slice 11 lands the collider and names no thread
(`:507-522`). Slice 16's measurement names *"the shard's stated physics budget"* and the document never
states that budget (`:594`).

**The cost.** The tick hitch is one of the eleven seam kinds. On the server it has no owner, no
mechanism and no gate.

**The fix.** Add to slice 11: the worker-pool seam on the shard; the rule that a chunk enters the physics
world only at a tick boundary from a COMPLETED build; and the rule that a body is never stepped past the
frontier of built chunks. Add a register row for the SERVER pool with `bevy_tasks` as a named option,
because it is already in the lock file and rayon is not.

*Example: a pilot dives a hull at a mesa at 250 m/s. The moon's shard builds the mesa's chunks a hundred
metres ahead of the nose. If it builds them on the tick thread, every player in that realm stutters.*

---

### F10. The SL9 sleep rule for parked hulls disappears

**Verdict: MISSING.**

Report 09 §3.4 states the rule and calls its gate *"the one that can fail"*: *"A child realm's shell
body SLEEPS the moment its authored velocity and its contact set stop changing… The per-tick cost must
therefore grow with the number of MOVING children, never with the number of parked ones"*
(`09_physics_controller.md:474-487`).

The synthesis keeps only the measurement (U-44, `:926`). Slice 11 lands *"one collider per chunk per
realm, reference-counted"* (`:509`) and never names a child realm's shell body, sleeping, or the broad
phase.

**The cost.** SL9 says a cost that grows with the child count is a defect, and that somebody must
measure it on a realm that has many. Six hundred parked hulls at a spaceport would be stepped every tick.

**The fix.** Add the sleep rule to slice 11's *lands*, and promote report 09's S12 to slice 11's GATE:
1 / 100 / 600 parked hulls in one planet realm, with the per-tick core time demanded not to grow.

*Example: six hundred hulls sit parked at a spaceport on a moon. A pilot fires one hull's thrusters.
That hull wakes, and the hulls its shell touches wake with it. The other five hundred and ninety stay
asleep.*

---

### F11. The shell-swap pop rule and its settle cap are dropped

**Verdict: MISSING.**

Report 09 withdrew its own first answer here and wrote a rule: *"A block is one metre, a one-metre lift
in one tick is a JUMP, and SL8 forbids shipping a capability without its continuity… the parent replaces
the collider and lets the SOLVER resolve the new resting contact, with rapier's penetration-correction
velocity CAPPED so the change is a settle and not a jump… The cap is a field of the realm's own record,
not a constant"* (`09_physics_controller.md:562-572`). Its gate is S14.

The synthesis carries none of it. The phrase "shell swap" does not appear. Slice 12 places blocks and
names no consequence for a hull that a crew member stands on.

**The cost.** Two losses. First, a jump seam with no gate. Second, a RECORD FIELD — the realm's settle
cap — that the no-magic-numbers rule requires and that no format in §2 reserves.

**The fix.** Add the settle cap to the realm record's field list, add the rule to slice 12, and add S14
as a slice-12 gate: weld a plate onto a parked hull's belly with a crew member on the ramp, one hundred
repeats, and demand that no tick moves the hull vertically by more than the stated cap.

*Example: a mechanic welds a plate onto the belly of a parked hull. A crewmate stands on the ramp. The
hull settles onto its new plate. It must not jump a metre.*

---

### F12. Five format-level one-way doors are absent from §2

**Verdict: MISSING.**

§2 is the section the owner freezes. These doors appear in the reports with the same deadline as the
three formats, and §2 does not carry them.

| Door | Source | The deadline the report gives |
|---|---|---|
| The tree SKELETON format — a segment lattice in 1/1024 m, radii in 1/1024 m, canopy volumes, inside the generator tag | `07_trees_composites.md:900` | before the first client links the crate (slice 7) |
| The BOX record's 20-byte layout (the compacted form of a run of cells) | `04_block_record_registry.md:851` | before the first compaction (slice 9) |
| The realm store's FAMILY PREFIX BYTES | `06_storage_diff_lane.md:872` | before the first block is written |
| The scoped channel key `H(scope ‖ name)` for a signal binding | `05_attachments_bodies.md:900` | before the first binding is persisted; the type is planted in the foundation |
| The integer ANGLE GRID a joint's angle counts on, and the carried sine and cosine table keyed by it | `05_attachments_bodies.md:901` | before the first joint angle is checkpointed |

A sixth door is flattened rather than dropped. Report 05's door 3 says *"The face numbering is the
chunk's own basis, the basis block orientation uses"* (`05_attachments_bodies.md:896`). §2.2's
attachment door names the face byte and never ties its numbering to the block orientation basis
(`:229`).

**The cost.** The skeleton format is the sharpest. The client links the crate at slice 7 and trees land
at slice 14, so the format would freeze inside a shipped static library before anybody writes it down.
The box record is the second sharpest: it is the on-disk form the compactor writes, and slice 9 builds
the compactor.

**The fix.** Add the five rows to §2's door tables, each under the format it belongs to. Put the
skeleton under Format B, because it is what `object_param` means.

---

### F13. The generator's arithmetic profile is a FOURTH freeze, and §2 does not say so

**Verdict: MISSING.**

Report 10 calls this *"the one a ruling OPENS"*, with the deadline *"Before the first world any player
builds on is saved"* and the retrofit cost *"The whole edit corpus"*
(`10_law_audit_stale_register.md:258-260`). Report 03 states the same door in its own words: *"The
generator's output is frozen the moment the first diff is saved"* (`03_generator_sl10.md:767`), and it
names two more doors inside it — THE EXTRACTOR'S PLACEMENT and THE ONE OWNER OF A BODY'S RADIUS.

The synthesis puts the whole thing in band B as decision V21 (`:648`), which its own preamble marks as
*"before the first generator or collider code"*. §2 never names it as a freeze.

**The cost.** The owner reads §2 as the list of things that must never move. The item with the largest
blast radius in the whole investigation is not in that list. One added octave moves the ground under
every placed block and under every saved diff.

**The fix.** Add a fourth sub-section, §2.4 — the generator's world identity: the crate version, the
arithmetic profile, the noise source, the octave table, the cave lattice step, the position hash, the
warp constants, the extractor's rule and the composition order. State the append-only octave rule as its
door, and state that the golden literals are the moment it shuts.

*Example: a player builds a landing pad on a hill on a moon. Six months later the generator gains one
octave. The hill is 40 cm higher and the pad floats.*

---

### F14. Slice 5 pins the golden set before trees exist

**Verdict: WRONG (a sequence defect).**

§1 says the one generator crate holds *"the seed-placed feature anchors and their geometry"* (`:53`).
Slice 5 lands *"the chunk digest; the golden self-check"* and its measurement pins *"~832 chunk digests
… equal byte for byte"* (`:412-433`). Slice 9 writes the first block store (`:479`). Slice 10 writes the
first player edits (`:492`). Slice 14 lands trees (`:555`).

A tree is a record on a cell (`:556`), so a chunk digest taken at slice 5 differs from the same chunk's
digest at slice 14. Every diff saved at slices 9 and 10 is stored against the older shape.

**The cost.** The world-generation tag bumps after a store exists, and the fail-safe on a stamp mismatch
DISCARDS the file (`crates/core/src/store_stamp.rs:182-192`, cited by the synthesis at `:653`).

**The fix.** State the rule in §3's preamble: nobody writes a store that must survive before the last
slice that changes the seed's shape. Either move the seed-placed feature anchors into slice 5, or mark
the stores of slices 9 and 10 as throw-away until slice 14 lands.

*Example: a tester digs a tunnel on a moon at slice 10. Slice 14 plants the moon's forests. That chunk
now hashes differently, so the store the tester built is refused and discarded.*

---

## 2. Open decisions the synthesis drops or answers silently

**Verdict: MISSING for every row.** The register V1–V100 claims to be *"Deduplicated across all ten
reports"* (`:618`). These rows sit in a report and not in the register.

| The dropped question | Source | Why it matters |
|---|---|---|
| The encoding of "a cell returned to the seed" on the wire — a revert tag or a record flag | `06_storage_diff_lane.md:890` (D-8) | see F3; it is a format field |
| A built realm's slot is a BOX — sell slots as `Aabb`, or keep `Shell` slots and inscribe a cube | `01_grid_family.md:895` (D5) | it decides a hull's grid DOMAIN, which slice 2 needs |
| The substance and form registry CAPS — the full `u16`, or the base's 512 / 256 | `04_block_record_registry.md:825` (D-6) | themes need the room, and slice 3 builds the tables |
| ONE attachment per face, or several slots? | `04_block_record_registry.md:833` (D-14) | report 04 calls it *"a GAMEPLAY call"* and *"the owner decides"*. §2.2 puts a slot byte in the key (`:224`), so the design answers "several" without asking |
| Who may attach to a block on a realm they do not own? | `05_attachments_bodies.md:1015` (item 10) | report 05 recommends owner-only until the access list lands, refused loudly. The synthesis refuses nothing |
| The mark rule and the cosmetic-local tier | `08_render_seam_seamless.md:833` (item 5) | two of the eight open rendering answers the owner still owes |
| The residency split — the server holds realms at the dot angle, the client library holds chunks at the drawable angle | `08_render_seam_seamless.md:834` (item 6) | V46 answers who owns the TIER rule, never who owns RESIDENCY |
| Felt acceleration DOWN to a child, so a walker inside a thrusting hull has a "down" | `10_law_audit_stale_register.md:337` (ask 1) | it is the core promise "walk inside a ship while it flies". It has no slice, no ask and no refusal |
| Buoyancy volume and centre of buoyancy UP on change | `10_law_audit_stale_register.md:339` (ask 3) | R-6 carries the shell, the centre of mass and the inertia, and drops buoyancy |

Two more rows are answered without the question being shown.

- **V2.6's reading.** The owner said *"it can be one block type with different params — durability,
  mass, and style"*. §2.2 answers *"Mass and durability are per KIND, derived from the substance row"*
  (`:211`) and gives style a 6-bit variant. That is a defensible reading, and it is the OPPOSITE of "one
  block type with different params" read literally. Report 10 lists it as an open contradiction:
  *"Per-instance block parameters (V2.6). A table row or a per-block record?"*
  (`10_law_audit_stale_register.md:361`). The register does not ask it.
- **V78 and V80.** Report 07 says of the falling tree *"Neither option is clean… Measurement 14
  decides"* (`07_trees_composites.md:919`) and of the object horizon *"the crossover is visible and the
  crossfade must be measured, not assumed"* (`07_trees_composites.md:927`). The synthesis recommends
  both answers on the ESTIMATED numbers (`:710`, `:712`) and never says the recommendation waits on the
  run.

**The fix.** Add the nine dropped rows to the register. Mark V78 and V80 as PENDING A MEASUREMENT rather
than recommended. Add report 10's V2.6 question as a starred band-A row, because the record width
depends on the answer.

---

## 3. Slices whose gate cannot go red, or whose measurement is not a number

**Verdict: WRONG** against §3's own promise — *"its GATE (a test that can go red) and its MEASUREMENT (a
number with a pass threshold)"* (`:326`).

| Slice | The defect |
|---|---|
| 1 | *"the suite's wall time does not grow"* (`:360`). No number, no tolerance, no noise band. A suite's wall time always moves |
| 3 | *"the digest's cost at open and at handshake, in microseconds"* (`:391`). A reported number with no pass threshold |
| 5 | *"pass ≤ 1.20 ms per tier-0 chunk"* (`:432`). The threshold comes from the base (`block_system_design.md:17461-17475`) for a BLOCKY field with no extractor. Slice 6 then adds the extractor's ESTIMATED 0.3–1.5 ms to the same chunk, and the synthesis's own §8 item 27 refuses the bench the 0.885 ms came from. The threshold is inherited, never derived |
| 6 | *"extraction milliseconds, vertices, quads and bytes per chunk … ESTIMATED 0.3–1.5 ms, UNMEASURED"* (`:445`). No threshold |
| 7 | *"the binary size and cold-build cost … chunks per second on ONE core and on eight"* (`:463`). No threshold on any of the four |
| 8 | *"the maximum and 99th-percentile vertical disagreement … in pixels"* (`:477`). The text says four pixels is visible; it never states the pass number |
| 13 | *"the shard's tick time … and the bytes per tick to one observer"* (`:552`). No threshold |
| 16 | *"no tick exceeds the shard's stated physics budget"* (`:594`). The document never states that budget |
| 17 | see F6 |

**The fix.** Give every measurement a pass number, or mark it as a REPORTED baseline that gates nothing.
Re-derive slice 5's throughput threshold for the gap field plus the extractor. Name the shard's physics
budget once, in the one tuning struct, so slice 16 can cite it.

---

## 4. Cases with no home

**Verdict: MISSING.** I tested the seven cases the task names. Four hold. Three do not.

| The case | The state |
|---|---|
| A mined cell under a placed block | **COVERED.** V74 (`:706`): no settling in v1, and the block hangs. Support is a placement-time check |
| A sub-metre block on a slope | **MISSING.** See F4 |
| A tree on a mined edge | **PARTLY MISSING.** Slice 14 lands *"the support rule"* as three words (`:558`), and no gate tests it. Report 01 owes it as measurement 13 (`01_grid_family.md:861`), and report 03 answers it in §4.6. The felling tombstone is F3 |
| A HUD on a rotating turret | **COVERED IN SHAPE, UNGATED.** V40 (the mount), V62 (one stamp) and the part-transform table in slice 7 (`:453`) place it. Report 05's measurement 4, *"Click-to-pixels through a body: an edit on a rotating turret round trip"* (`05_attachments_bodies.md:921`), is dropped from §7 |
| An edit during a crossing | **HALF COVERED.** Slice 10 gates the SUBSCRIPTION flip and slice 15 gates the AUTHOR crossing (`:500`, `:568`). Report 10's case 1 asks a different question: the OWNING REALM crosses while a player digs inside it — *"Who owns the diff for those ticks? How does the diff's fence order against the transfer saga?"* — and it asks for *"a fixture that digs during a hand-over"* (`10_law_audit_stale_register.md:445-450`). Slice 15 plants `BLOCK_STORE_FLUSH_STEP` in the re-shard drain (`:574`) and never gates it |
| A landing on a moving moon | **COVERED, AND THE ANSWER IS BURIED.** Report 09 corrects the whole picture: a planet's bound is its sphere of influence, so *"There is NO crossing at the ground"* — the hull became the planet's child hundreds of thousands of kilometres up (`09_physics_controller.md:450-465`). That finding deletes a seam nobody must build, and the synthesis never states it. Slice 11's landing fixture reads as if a crossing happens at touchdown |
| Two players editing one cell in one tick | **COVERED.** Report 09 §4.1 answers it with the occupancy query and the two-stage receipt (`09_physics_controller.md:529-550`). The synthesis carries the validation list in slice 10 (`:494`) without the refusal-path shape |

**The fix.** Add the dig-during-a-hand-over fixture to slice 15's gate. Add the tree-support case to
slice 14's gate. Add report 09's sphere-of-influence finding to §1 or to slice 11, because it deletes a
seam and the owner should know that it is deleted.

---

## 5. UNMEASURED rows the synthesis drops

**Verdict: MISSING.** §7 claims to hold *"Every number the design depends on that has no measurement"*
(`:857`). A report owes each row below, and §7 does not hold it.

| The owed measurement | Source | What it decides |
|---|---|---|
| The C-ABI copy cost per chunk. The payload is ~715 KB packed, not 476 KB | `03_generator_sl10.md:659` (U9) | the Unreal interface of V2.8, and slice 7's seam |
| The k3d image architecture on this host | `03_generator_sl10.md:660` (U10) | whether the dev cluster ALREADY runs two architectures, which is the no-drift gate's first real leg |
| `verify-pyramid` on a 100 M-edit realm; the base's 54 s is an estimate, and the suite runs it per scenario | `06_storage_diff_lane.md:783` (M-7) | slice 9's gate is written against `verify-pyramid` |
| The realm store under an edit storm, and under a stalled disk | `06_storage_diff_lane.md:786` (M-10) | the typed refusal and the parked-tick counter: what every builder on a moon meets when the disk stalls |
| Palette width over WHOLE-CELL records on a 50 000-block hull | `04_block_record_registry.md:801` (M6) | the chunk delta's compaction crossover |
| The style-detail function is byte-identical on both hosts | `04_block_record_registry.md:802` (M7) | V2.6's client-derived detail; see §6 |
| The content bake assertion that `volume_units(kind)` is a multiple of `1 << (3 × max_scale)` | `04_block_record_registry.md:812` (M15) | a thin decorative fin whose volume shifts to zero at eighth scale |
| Rapier snapshot and restore across targets | `09_physics_controller.md:812` (S8) | how a checkpoint travels; Category C forbids cross-host re-simulation |
| The interpolation-buffer falsifier on today's client | `09_physics_controller.md:814` (S10) | report 09 runs it *"before any latency figure is quoted"*, and the synthesis quotes latency figures |
| What the pinned parry actually holds | `09_physics_controller.md:819` (S15) | V27's comparison keeps an UNMEASURED column |
| One integrator, no bend at a crossing — a hull under constant drive crossing from a stub realm into a rapier realm | `09_physics_controller.md:822` (S18) | a jump seam at the shell of every realm that gains physics |
| The window-versus-capture pixel diff at one tick | `08_render_seam_seamless.md:872` (item 6) | HR6 says the harness captures what the player sees. Nobody measured whether the two paths draw the same image |
| The atmosphere's cost, ESTIMATED 0.70 ms | `08_render_seam_seamless.md:876` (item 10) | the client's frame budget on a landing |
| The residual sky error under the `+Y` flat-ray approximation | `08_render_seam_seamless.md:879` (item 13) | V87's atmosphere hand-off rests on it |
| Resident chunks of a 100 km view WHILE MOVING — the ≤ 7 500 chunk cap | `08_render_seam_seamless.md:862` (bench 3) | the client's memory cap and the crossfade band fraction |

**The fix.** Add the fifteen rows to §7, each under the slice that needs it.

---

## 6. Requirements V2.1–V2.9 with a thin place

**Verdict: MISSING for V2.1's second half and for V2.6's second half. The rest hold.**

| Requirement | Where the design holds it | The state |
|---|---|---|
| V2.1 smooth terrain, mine and place | §2.2's density byte, slice 6's extractor, slice 10's edits | **HALF.** The MINING half is complete (V65, the yield). The PLACING half — *"when we place voxels, the terrain should change accordingly"* — has no deliverable. Slice 10 lands *"the gap under a placed square block"* (`:494`), and that is a SQUARE block cut into a slope. No slice states what happens when a player places a TERRAIN-form cell: what density the placed cell carries, whether a player may raise ground above the seed's surface, and whether V67 (*"no natural substance placed as a square block for the first slice"*, `:698`) refuses it. **The section that should hold it is slice 10 and Format B** |
| V2.2 trees as one object | §1, slice 14, V17, V76–V83 | **HOLDS**, except F2, F3 and the skeleton door of F12 |
| V2.3 ~20 square shapes | Slice 3's catalogue, slice 12's lanes | **HOLDS** |
| V2.4 sub-metre blocks | §2.1's sub-site, slice 12, V7, V39 | **HOLDS in the address. MISSING the seat (F4). The door order is wrong (F5)** |
| V2.5 attachments with no volume | §2.2's attachment row, slice 13, V12, V13, V40–V42 | **HOLDS** |
| V2.6 params and client-side style | §2.2's variant, V72 (the client derives the trim) | **THIN.** V72 answers WHO derives. No slice lands the derivation, no gate covers it, and report 04's M7 (the style function is byte-identical on both hosts) is dropped. **The section that should hold it is slice 12 and §7.** See also the V2.6 reading question in §2 |
| V2.7 themes | Slice 3's theme column, V93 | **HOLDS** |
| V2.8 an engine-agnostic seam | Slice 7's `ChunkGeometry`, V47, V48, and the R and S marking in §3 | **HOLDS**, except the C-ABI cost (§5) and F8 |
| V2.9 seamless | Slice 17, V96's twelfth seam kind | **HOLDS in shape. It fails in the gate (F6)** |

---

## 7. Two smaller contradictions

### F15. The tier field is four bits here and five bits in report 08

**Verdict: CONTRADICTS_REPORT (unreconciled).**

§2.1 says *"`tier` is 4 bits"* (`:137`), which matches report 01 (`01_grid_family.md:690`). Report 08's
door table says *"The tier field on the chunk address (five bits) … Before the first pyramid is
persisted"* (`08_render_seam_seamless.md:991`). Four bits hold sixteen rungs and the design uses
thirteen, so four is probably right. The synthesis picks it and never names the disagreement, and report
08 owns a door on the other number.

**The fix.** State the width once. Name report 08's row as superseded. Say in Format A's door table that
the address freeze already covers the tier width.

### F16. Report 02's door still says the carver combines by `max`

**Verdict: CONTRADICTS_REPORT (harmless in effect, sharp in wording).**

§2.2's door says the density is *"combined with the carver by a comparison, never `max`"* (`:216`).
Report 02's door says *"combined with the carver by `max`"* (`02_smooth_terrain.md:625`). The two agree
on the OPERATION and disagree on how a programmer writes it, because V34 removes `f64::max` from the
fenced float type (`:661`). The frozen bytes are identical either way.

**The fix.** One sentence in §6 (disputed items) that says so, and a correction to report 02's door text.

---

## 8. What stands

I tested these and they hold. A critique that lists only faults hides what the owner may rely on.

1. **Every code claim I re-ran is true.** `InterShardFlow` holds 44 arms (MEASURED by `awk` over
   `crates/wire/src/intershard.rs:124` onward). `INTERP_BUFFER_MS = 120.0`
   (`crates/wire/src/channels.rs:328`). The `Store` trait holds exactly five methods — `put`, `delete`,
   `scan`, `commit`, `flush` — with no point read and no bounded range
   (`crates/sim/src/io/mod.rs:477-495`). `noise` and `rapier` hold zero `Cargo.lock` entries, and
   `noise = "=0.9.0"` sits unused at `Cargo.toml:78` (MEASURED by `grep -c`). `crates/client/src/` holds
   zero `std::thread` hits and zero `rayon` hits (MEASURED). The drawable floor is 45° over 720 rows
   (`crates/core/src/geometry.rs:1156-1173`).
2. **The three formats are internally consistent where they touch.** The `CellIndex` width, the
   substance width and the sub-site width agree across Format A, Format B and Format C.
3. **The prune rule and its consequence are honest.** §2.3 states the requirement the rule puts on the
   generator, states that the naive fold costs ESTIMATED 1.9 hours for one coarse cell, and books it as
   R-7 and U-14 instead of hiding it.
4. **The SL6 table is in SL6's own shape.** Every row states the data, the direction, why the receiver
   cannot compute it, and the cost of doing without. R-17 correctly refuses a sibling's placement as
   REFUSED-UNTIL-ASKED instead of treating SL2's prose as built law, and it cites the passing test that
   pins the absence (`crates/wire/tests/intershard_closed.rs:485`).
5. **The stale-text register is the strongest section.** It names 34 refused claims, it separates
   "refused by a ruling" from "refused by the code", and it also names what SURVIVES — the float
   newtype, the drag term, the pop detector and the ~20 shapes. Its warning that the flight cap is STILL
   IN THE CODE (`crates/sim/src/stub/dot.rs:457-474`) is exactly the shape the owner asked for.
6. **The extractor's move into the shared crate** (§6 item 12) is the right call, and the document says
   so plainly. Without it an Unreal client owns a second ground, which SL10 V1.2 forbids by name.

---

## 9. The shortest path to a document the owner can freeze

In order:

1. Fix F2 and F1. One is a gate that must fail. One is a number that lies about its own provenance.
2. Answer F3 and F4 — the two format holes. Both are cheap now and permanent later.
3. Resolve F5. Say what is PROVISIONAL and what the re-open costs.
4. Fix F8. It is one line, and it is a law breach as written.
5. Add F12's five doors and F13's fourth freeze to §2.
6. Give slice 17 its twelve tolerances (F6), and give every other slice a pass number (§3).
7. Add the nine dropped register rows (§2) and the fifteen dropped UNMEASURED rows (§5).
8. Add the three missing cases (§4), V2.1's fill rule and V2.6's style slice (§6).
