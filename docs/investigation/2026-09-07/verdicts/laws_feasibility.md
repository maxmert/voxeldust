# Verdict — the feasibility refutation of report 10 (the law audit and the stale-text register)

**Date:** 2026-09-07. **Lens:** cross-cutting law audit and stale-text register.
**Target:** `docs/investigation/2026-09-07/10_law_audit_stale_register.md`.
**Result: REFUTED.** One load-bearing claim is wrong. Three load-bearing items are missing.

I re-measured every code fact in the report's table 1 by `grep` and by reading the file on this tree.
Most of them stand. The report fails on what it leaves out, and on one sentence about collision.

---

## 1. What stands (re-measured, 2026-09-07)

I confirm these report claims. I state them so the synthesis can trust them.

- `InterShardFlow` holds **44 arms** (MEASURED: a Python brace walk over the enum block,
  `crates/wire/src/intershard.rs:124-604`). Door 38's thirty-arm ceiling is passed.
- The approval-citation gate is REAL, and the report is right to name it as the successor to the
  count: `crates/wire/src/version.rs:800`, `every_ledger_minor_from_9_up_carries_an_owner_citation`.
- `ChannelKey` does not exist (MEASURED: `grep -rn ChannelKey crates` returns nothing).
- The saga step ids run 7..18 and 19 is free (`crates/wire/src/intershard.rs:65-115`).
- `pin_abs` and `anchor_epoch` are removed at wire minor 8 (`crates/wire/src/version.rs:67-79`).
- The transient re-clamp is deleted (`crates/sim/src/stub/transient.rs:231-236`).
- The shard no longer states a marker; the arm stays a reserved tombstone
  (`crates/sim/src/stub/window.rs:412`, `crates/wire/src/session_flow.rs:625-640`).
- Neither `rapier3d` nor `noise` appears in `Cargo.lock` (MEASURED: `grep` returns nothing on a
  7,851-line lock file). `noise = "=0.9.0"` is declared at `Cargo.toml:78` and no crate uses it.
  **The report adopts no library. It offers both to the owner. That is correct.**
- M7's one scalar distance is BUILT: `RealmInterest` carries it
  (`crates/wire/src/intershard.rs:1319-1342`). The report does not measure this, but the claim holds.

**Example, in the game's words.** A hull leaves System 7 and joins the galaxy. The galaxy's shard no
longer draws a marker for it, and the hull's own look is the only picture of it. The report says this
and the code agrees.

---

## 2. The refutations

### 2.1 WRONG — the server's collision surface is the derived shape PLUS the diff, never the generator alone

The report writes, for O28: *"the server collides on the smooth shape the client draws (a heightfield
or trimesh from the generator, not voxel cubes)"*. Its §4 item 5 repeats it: *"the collider comes from
the same extractor (SL10 V1.5)"*.

The ruling does not say that. V1.5 states the server evaluates the same crate. V1.6 states that every
edit crosses as a one-hop DIFF and that **the client applies the diff over the shape it derived**. The
surface the client DRAWS is therefore `generator(seed, address)` composed with the diff. A collider
built from the generator alone is a different surface.

**The failure, in the game's words.** A player digs a tunnel into a moon. The tunnel is live state, so
it arrives as a diff and the client cuts it into the derived hill. The moon's shard collides on the
generator's hill, which has no tunnel. The player walks into the tunnel on screen and stands on solid
rock in the simulation. That is a jump, and a jump is a seam (SL8).

**Fix.** Register one row: the collision surface is the derived shape composed with the owning realm's
diff, and the composition happens before the collider is built, on both hosts. Add the measurement the
row owes: the client's drawn surface and the shard's collider surface must agree on a chunk that
carries a diff, not only on a virgin chunk.

### 2.2 MISSING — SL10 creates a one-way door the report does not register

The report's door table moves ten doors. It opens none. SL10 opens the largest door in the file, and
the report does not name it.

Once a player builds on ground the client derived, the generator crate's version and the target's
arithmetic profile become **world identity**. A later change to the crate — one added octave, one
rounding fix — moves the ground under every placed block and under every saved diff, because a diff is
stored against the derived shape (the report's own O6 item 3 says prune-on-equality compares against
the static shape). There is no rollback. The world cannot be re-rolled, because G1 already forbids a
knob that moves a star anybody has seen.

**Example, in the game's words.** A player builds a landing pad on a hill on a moon. Six months later
the generator gains one octave. The hill is 40 cm higher. The landing pad now floats, or the hill eats
it. Nobody can undo it, because every player's moon moves at the same time.

**Fix.** Add a door row: *"the generator crate's arithmetic profile and version"*, deadline **before
the first world any player builds on is saved**, retrofit cost **the whole edit corpus** (every diff
must be re-based against a new shape, or the old shape must be kept forever as a second generator,
which SL5 forbids). The recommended answer belongs beside it: a frozen arithmetic profile plus an
append-only octave rule (a new octave may only add detail below the existing surface's tolerance),
measured, not argued.

### 2.3 MISSING — the door table carries no deadline and no retrofit cost

MEASURED on the report: three occurrences of "deadline", none inside the door table; zero occurrences
of "retrofit". The base gave its doors deadlines ("before P4"). The report removes or moves ten of
them and gives back neither a deadline nor a cost.

A door row without a deadline cannot be scheduled, and a door row without a retrofit cost cannot be
traded against a slice. Door 12 is the sharp one: the report says *"CHANGED by V2.4 — re-design as a
strictly additive finer layer"*, and never says by when, nor what it costs to add the finer layer after
the saved block record is frozen. V3.2 makes the owner freeze that record. The two must be decided in
the same sitting.

**Fix.** Give every moved door three fields: the new state, the deadline (which format freeze or which
slice), and what it costs to change afterwards.

### 2.4 MISSING — no structural fence for the determinism rules, and the base's own surviving fence goes unlisted

The report's SL10 row restates V1.4 (integer hashing, fixed order, no fast-math, no FMA, no libm
transcendentals) and names the byte-for-byte gate as the answer. **A gate is a detector, not an
exclusion.** The gate compares the chunks it generates. It cannot exclude a target-dependent
divergence at an address it never generated.

This project's own law says how to do it properly. SL1 clause 5 requires a fence enforced by a crate
or module rule with an observed-failing control, *"never by care"*. The conventions section already
uses that shape for I/O (clippy `disallowed-methods`).

Worse, the report is a stale-text register, and it misses the base text that SURVIVES and matters most:
`docs/investigation/block_system_design.md:3758-3765` proposes exactly the fence — a newtype `Gf(f64)`
that exposes `+ − × ÷`, `sqrt`, `floor`, `abs`, `min`, `max` and nothing else, so a transcendental is
unreachable on the generation path by construction. The report lists dozens of dead base mechanisms
and does not list this live one.

Three arithmetic items the report also leaves unregistered:

- **The float remainder.** `%` on `f64` lowers to the platform's `fmod`. V1.4 names `sin`, `exp` and
  `pow` and does not name it. A `Gf` newtype excludes it; a prose rule does not.
- **`powi`.** It expands to repeated multiplication and is safe, but nothing in the report says so, and
  a reader who takes "no `pow`" literally will strike a lawful operation.
- **`mul_add`.** The base states it is bit-exact and keeps it (`block_system_design.md:3748-3751`).
  V1.4 forbids fused multiply-add CONTRACTION, which is a different thing (contraction is the compiler
  fusing `a*b+c` without being asked). The report's flat *"no FMA"* strikes `mul_add` without
  registering that the base and the ruling disagree. That is precisely the kind of row this register
  exists to carry.

**Example, in the game's words.** The moon's shard and the client both evaluate the one crate for the
chunk under the player's boots. One of them was built for a machine whose `fmod` rounds differently.
The gate generated ten thousand chunks and never reached that branch. The player's boots sink one
centimetre into a slope on some machines and not on others.

**Fix.** Register the `Gf` newtype as the surviving base mechanism and as the fence V1.4 needs. Add the
three arithmetic rows above. State the gate as the control that must be OBSERVED FAILING (delete one
octave, watch it go red), not as the exclusion.

### 2.5 UNMEASURED AS FACT — SL9: one O(children) fold survives, and the report does not register it

The report's SL9 row says *"any fold that visits every child is a defect"*, then cites one measurement
(20 µs, 5 candidates, 279,380 children) and calls R8 item 1 closed. A fold that visits every child is
still in the tree today.

MEASURED, by reading the code: `rebuild_static_rows_for` at `crates/sim/src/stub/regions.rs:876` walks
`children_of[parent]` in full. It is called on every adoption (`regions.rs:752`) and every release
(`regions.rs:855`), for every child that is not a `RealmId::Ship`. On the galaxy that is 279,380 rows.
`docs/design/DEFERRED.md` records the measurement: the galaxy shard's slowest tick was **60.0 ms
against a 20 ms budget**, right after a hand-over, and the fix exempted ships only — *"A planet's
adoption still rebuilds its parent's layer."*

This is not a distant concern for the voxel foundation. The base's §3.6 mints a realm when a player
places a block on a construction anchor. Minting a realm is an adoption on the parent. If the minted
realm is not a `Ship`, every such placement pays an O(children) copy on its parent.

**Example, in the game's words.** A player places the first block of a station on an anchor in a busy
star system. The star system copies its whole static row vector to admit the new realm. In the galaxy,
with 279,380 children, that same event cost 60 ms — three times the tick budget.

**Fix.** Add a fact row for the surviving fold with its file:line and its measured cost, and add a
consequence row: any voxel mechanism that mints a realm inherits it. State the measurement owed — mint
a realm on a wide parent and read the pace line — before the anchor design is promoted.

### 2.6 MISSING — five cases the domain needs and the report never names

MEASURED on the report by `grep`: zero occurrences of "thread", "frame budget", "slope", "turret",
"retrofit"; "crossing" appears twice, neither about an edit.

1. **An edit during a crossing.** A player digs inside a hull while the hull hands over from the star
   system to the galaxy. Who owns the diff for those ticks, and how does the diff's fence order against
   the transfer saga? The report never asks. The transfer machinery is the project's most proven part,
   and the voxel diff is a new payload riding beside it.
2. **A sub-metre block on a smooth slope.** V2.1 makes the terrain surface smooth. V2.4 asks for
   sub-metre blocks that seat inside the 1 m cell. The report treats the two requirements in separate
   rows and never registers that they collide: on a smooth slope the surface does not follow a cell
   face, so a block on a sub-cell lattice either floats above the ground or sinks into it. This belongs
   in §4 as an eleventh open contradiction, because it decides the saved record's field for a sub-cell
   seat (a lattice offset, or a surface-relative seat — two different formats).
3. **A HUD on a rotating turret.** V2.5's joint *"turns what is built on it"*. The report calls a joint
   a no-volume attachment and stops. A turning sub-assembly is a MOVING FRAME inside one realm: the
   cells built on it no longer sit at fixed addresses, which breaks the pyramid's address and breaks
   SL10's function of seed and address for anything riding it. That is what the `FrameSpace` seam is
   for, and D-38 rules that seam to be terrain's FIRST slice. The report cites D-38 for O29 and never
   connects it to V2.5. Door 37 (articulated machines) sits beside it, unlinked.
4. **A tree on a mined edge.** A tree is one seed-placed record whose geometry the client derives
   (V2.2). A player fells it. The felling is live state, so the diff must carry a **deletion tombstone
   against derived shape**, and the client must not re-grow the tree when it re-derives the chunk. The
   report never states that the diff format needs a negative entry. The diff is one of the three
   formats V3.2 freezes, so the omission reaches a freeze.
5. **The client's frame budget, and the thread.** SL10 moves generation cost onto the CLIENT for the
   first time. The report names the tick-hitch seam only for the server-side residency governor (O32).
   It never states that the tick-hitch seam is now a CLIENT seam, never states a frame budget, and
   never asks which thread evaluates the crate. The one relevant base measurement it quotes
   (0.885 ms per chunk on an M4 Pro) stays *"a measurement of THAT bench"* and never becomes a
   requirement. A hull landing at flight speed crosses chunk after chunk; a planet 10,000 km away in
   the window needs coarse chunks over its whole visible face.

**Example, in the game's words.** A hull descends toward a moon at flight speed. The client must derive
the hills ahead in time. If that work runs on the frame thread, the picture stutters, and a stutter is
the tick-hitch seam. Nothing in the report says where the work runs.

### 2.7 WRONG (small) — the 150,000-row figure belongs to a live law, not to the base

The report's fact row 61 says *"the base's '150,000 rows per player per tick' fear is historical"*.
MEASURED: that phrase appears nowhere in `docs/investigation/`. It appears at `CLAUDE.md:158`, inside
SL1's own reasoning for the 2026-08-24 reversal.

A register that marks text stale must not mark a live law's own justification "historical". SL1 rests
on it. **Fix:** delete the clause, or restate it as *"SL1's measured justification, which stands"*.

### 2.8 WRONG (small) — the grandparent's gravity field is refused by SL6, not by SL1 clause 4

§4 item 7 says *"the grandparent's 24 B gravity field is REFUSED outright (SL1 one hop)"*. SL1 clause 4
governs PLACEMENTS: what you are told about yourself you never pass on, and you are never told your
parent's placement. A uniform external acceleration is not a placement. The refusal ground is SL6 (new
data crossing a boundary, default NO) and SL2. The report's §5 list cites both, so the fix is one word.
Naming the wrong law weakens the refusal, because the owner can lawfully approve an SL6 ask and cannot
lawfully approve a breach of SL1.

### 2.9 WRONG (small) — the doc-comment count in fact row 43

The report says `FrameSpace`, `SphericalSpace`, `reanchor()` and `AnchorGen` are named by *"two doc
comments"* in two files. MEASURED: at least six mentions in three files —
`crates/core/src/fence.rs:18`, `crates/sim/src/capability.rs:11,18,35,40`, and
`crates/sim/src/stub/transient.rs:239` (which the report misses, and which is the one that names the
seed-derived analytic gravity P5 owes). **The load-bearing half stands: none of the four exists as
code.** The count does not.

---

## 3. What the report does right, and should keep

- It marks its unknowns honestly. §4 item 10 says plainly that it could not find the generator tag
  V1.3 names, and marks it UNMEASURED. That is the correct posture.
- It refuses O57's test-only generator by name (SL5). Correct and important.
- It proposes no library. `rapier3d` and `noise` stay owner options. Correct.
- Its do-not-promote list in §5 is the most useful part of the file. The three WRONG rows above and the
  missing collision-plus-diff row should join it.

---

## 4. The verdict

**REFUTED.** The register is broadly accurate about the past and incomplete about the future. Its code
facts survive re-measurement. Its one wrong load-bearing sentence (the collision surface) would let a
player fall through their own tunnel. Its three load-bearing omissions — the generator-version door,
the missing structural fence for the determinism rules, and the surviving O(children) fold — each
reach a format the owner is about to freeze.

**Example, to close.** A player digs a tunnel into a moon and builds a landing pad on the hill above
it. Under the report as written: the moon's shard collides on a hill with no tunnel; the generator may
change next year and move the pad; nothing but care stops a transcendental entering the generator; and
the anchor that minted the pad's realm copied 279,380 rows on a wide parent. Four rows fix all four,
and all four rows are cheap to write now and expensive to write after a freeze.
