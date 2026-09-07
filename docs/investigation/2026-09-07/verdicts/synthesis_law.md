# Verdict — the law critic on `00_proposed_voxel_foundation.md`

**Date:** 2026-09-07. **Role:** the law critic. **Target:** the synthesis at
`docs/investigation/2026-09-07/00_proposed_voxel_foundation.md`.

**Method.** I read the binding law first: `CLAUDE.md` (HR1–HR6, SL1–SL10), the newest ruling
`docs/design/owner_decisions_2026-09-07_voxels.md`, then the suit, the reach, the visibility radius, the
seed and secrecy, the movement answers, the galaxy shape, the movement contract, and the 2026-08-24
ruling. I then tried to break the synthesis against each of them. I verified every code citation this
verdict leans on by reading the file at the cited line.

**Result: PARTIALLY REFUTED.** The core of the design survives. One generator crate on both hosts, the
form-selected lane, the twelve-byte record, the derived pyramid and the engine-free chunk seam all hold
against the law. Twenty findings stand against it. Three of them break a law in the FIRST slices and must
be answered before the owner shuts any format: the sequence never asks the owner the SL6 questions it
itself wrote (F1), the client grows its own residency lead from a speed it computes (F2), and five slices
land a feature with no two-shard-kind gate (F3).

**How to read a number below.** MEASURED means a program ran and I say which. ESTIMATED means arithmetic
on a cited number. UNMEASURED means nobody has run it.

---

## The findings

### F1 — BREAKS SL6. The sequence never asks the owner the nineteen questions it wrote.

**Where.** §3 slice 0, §3 slice 4, §5.

SL6 says the default is NO, and the synthesis agrees: *"Default NO. Nothing below is added by this
document"* (§5). Slice 4 then LANDS seven of those requests — the block-edit forward and its ack, the
Bulk class with its chunk rows, the rung floor on a window, the world action, the client's world
handshake, and `TAG_SURFACE`. Slice 0 is the one sitting that collects owner answers, and its gate reads
*"every field in §2 has an owner-stated value"*. §2 is the three FORMATS. §5 is not in slice 0's gate,
and no other slice collects it.

So the sequence, as written, builds a wire plant whose every item is refused by default.

*Example.* A pilot inside a hull berthed on a moon breaks a rock beside the ramp. That edit rides
`InterShardFlow::BlockEdit` from the hull's shard to the moon's shard. That arm is R-1. Nobody asked the
owner whether it may exist, yet slice 4 cuts it into the one reviewed wire file
(`crates/wire/src/intershard.rs:124`).

**Fix.** Slice 0's gate gains a second half: the owner answers every row of §5 by name, and the answer is
written into `docs/design/`. Slice 4 lands ONLY the arms the owner approved, and a planted arm with no
recorded approval is a defect the closed-set test names.

### F2 — BREAKS SL10 V1.7. The client derives a velocity to size its own resident band.

**Where.** §1 "How the landing stays seamless"; §3 slice 8; §4 V88; §5 R-18.

SL10 V1.7 is plain: *"The client never derives a pose, a velocity, or any state of any entity or realm."*
The synthesis writes the residency rule as *"a lead grown from the closing speed and the interpolation
buffer"*, and V88 answers the "how" with *"the residency rule reads the delivered track's spread"*. The
spread of a delivered track over time IS a velocity. V88 then names the alternative — a warm radius the
realm states (R-18) — and recommends NOT asking for it.

The reach ruling shows where that arithmetic belongs. *"A fast looker grows the reach by closing speed ×
boot time"* (`docs/design/owner_decisions_2026-09-02_reach.md:184`) is the SERVER's sum, and the shard's
own interest lead is already server-side: *"the interest lead is the closing speed times this buffer plus
one tick"* (`crates/wire/src/channels.rs:322-328`). Both are computed where the poses are authored.

*Example.* A hull dives at a moon at its rated cruise. Under V88 the client watches the moon's delivered
rows, subtracts two of them, calls the difference a closing speed, and decides how much ground to build
ahead of the boots. That is the client computing motion, which is the exact class of drift the
client-only-renders law exists to stop.

**Fix.** Size the client's band from data the server states and the client only reads: the realm's own
reach, the interpolation buffer, and a lead the OWNING shard states beside them. If no delivered datum is
enough, R-18 is a real ask and must go to the owner as one. V88 must stop being presented as the local
formulation that avoids an ask.

### F3 — BREAKS HR4. Five slices land a feature with no identical fixture on two shard kinds.

**Where.** §3 slices 9, 10, 11, 14, 16.

HR4 says *"every feature passes the identical fixture on ≥2 shard kinds (G-IDENTICAL) or it doesn't
land"* (`CLAUDE.md:110-111`). Only slice 12 and slice 13 name such a pair. Slice 9 (the block store),
slice 10 (mining and placing), slice 11 (the terrain collider), slice 14 (trees) and slice 16 (the
character on the surface) each land a feature and each names a single-kind gate.

The profiles exist as data today: `crates/sim/src/capability.rs:246,262,278,290` gives the planet
`VoxelGeometry::Spherical` and the ship, the station and the asteroid `VoxelGeometry::Cartesian`. So the
pair is available and cheap.

This also promotes V38 out of band B. V38 asks whether a Cartesian realm may hold terrain-form cells. If
the answer is NO, the smooth lane can never run on two kinds, and slice 10 cannot satisfy HR4 at all.
V38 therefore BLOCKS slice 10, not the generator.

*Example.* A miner digs the same trench with the same tool in two places: on a moon, and in a station's
soil bay. One fixture, two profiles, two stores compared byte for byte. That is the run HR4 asks for and
no slice schedules it.

**Fix.** Each of the five slices names its pair and its identical fixture body. V38 moves to band A and
is answered before slice 10 is planned.

### F4 — BREAKS SL6 AND SL3. R-4 asks for a rung floor after the document already found the local rule.

**Where.** §5 R-4; §4 V89; §3 slice 4.

SL6 says *"Find the local formulation first; there usually is one."* The document FINDS it, in V89:
*"the realm derives fine interest from the poses it authors and publishes down to the rung drawable at
its own bound"*. It then applies that rule to R-19 (an interest radius) and refuses the ask — and does
not apply it to R-4 (a rung floor), which is the same question with a coarser answer.

R-4 also collides with SL3, which the request never names. SL3 gives the realm its own look *"at a detail
level it chooses"* (`CLAUDE.md:177-181`), and the reach ruling repeats it: a realm may state a smaller
reach than its size implies, *"the realm's choice, never a parent's clamp"*
(`docs/design/owner_decisions_2026-09-02_reach.md:181-183`). A rung floor pushed down from the gateway is
a clamp on the realm's own drawing.

*Example.* A pilot in orbit looks at a mining moon. Under R-4 the gateway tells the moon "do not send
finer than eight-metre cells". Under V89 the moon works it out alone: it knows its own bound, it knows
the drawable floor, and it publishes the rung its own size can show.

**Fix.** Apply V89's local rule to R-4 and drop the ask. If a floor is still wanted for a byte budget,
re-argue it against SL3 by name and state what the realm gives up.

### F5 — BREAKS THE SUIT RULING. Two gates need a character that lands nine slices later.

**Where.** §3 slice 8, §3 slice 11, §3 slice 16.

Slice 11's gate reads *"a character walks off the ramp"*, and slice 8's measurement runs the pop detector
*"at walking speed"*. The character lands in slice 16. The document's own U-48 says *"no collider, no
shape cast and no character controller exists today"*.

The suit ruling binds what a gate may do: *"The gates fly the shipped path: berth a hull, board it, push
— or wear a suit whose rating states the speed the leg needs"*, and *"never a walking dot"*
(`docs/design/owner_decisions_2026-09-05_suit.md` S4). The only walker available at slice 8 is the dot
with its boot-time ramp, which the same ruling calls wrong and defers (S6, D-MOVE-3). That ramp is still
in the code (`crates/sim/src/stub/dot.rs:457`; `crates/core/src/flight.rs:115`).

**Fix.** Slice 8's and slice 11's gates fly a HULL only. The walking legs move into slice 16, and slice
17 keeps the whole flight. The owner's order stands: the foundation, then blocks and terrain, then the
character, then the suit.

### F6 — MISSING. An edit changes what a realm IS, and nothing carries the change up.

**Where.** §1, §3 slice 10, §3 slice 12; the movement contract; the reach ruling.

The movement contract's upward lane carries *"mass, cross-section, drag coefficient … ON CHANGE ONLY —
when the hull is rebuilt, never per tick"* (`docs/design/owner_decisions_2026-08-26_movement.md:42`). The
reach ruling adds a second on-change statement: *"A realm's reach comes from its own LOOK, not only its
size … The realm states its own reach"*
(`docs/design/owner_decisions_2026-09-02_reach.md:79-81`, and the lane at `:135`).

Every block a player places changes all four numbers. No slice derives them from the block field, none
re-states them, and none says what happens when a builder places ten blocks a second — which turns an
"on change" lane into a per-tick lane, which the contract refuses by name.

*Example.* An engineer welds a 500-metre mast onto a station. The station's mass grows, its cross-section
grows, its look radius grows, and so its reach grows. Until the station re-states its reach, its parent
planet keeps testing the old number, and a pilot who should already see the mast sees nothing. That is
the sprite-cull seam, caused by a build.

**Fix.** A named slice derives the mass, the cross-section, the drag coefficient and the look radius from
the realm's own block field, with a coalescing rule (state on a settle, never per placement) and a gate
that builds 1 000 blocks and counts the upward statements.

### F7 — MISSING. `BodyId` sits in a persisted key and nothing says how one is allocated.

**Where.** §2.1, §4 V11, §4 V40.

`BlockAddr` carries `body_id: BodyId(u16)`, the store key is `(body, chunk)`, and the diff frame's header
carries it too. V11 makes it a one-way door: *"every saved multi-body construction is ambiguous"* if it
is wrong. Yet no format says who allocates a body id, whether the numbering is dense, whether an id is
reused after a turret is cut off, or how it survives a store reopen.

*Example.* A gunner bolts a turret to a hull's spine. The turret's plates are filed under body 1. The
gunner cuts the turret off and welds a crane in its place. If the crane also takes body 1, the turret's
old chunks answer the crane's key, and a saved hull loads with plates that belong to a machine that is
gone.

**Fix.** Freeze the allocation rule WITH the address in slice 0: monotone per realm, never reused,
persisted in the realm's own store, with body 0 reserved for the realm's own grid.

### F8 — MISSING. Format C's reserved bits are not budgeted against the canopy fold slice 14 needs.

**Where.** §2.3, §5 R-15, §3 slice 14.

The pyramid entry keeps fourteen reserved bits and names two candidate spends: a surface-height delta
(8 bits) and a sticky "differs from the seed" bit. R-15 then asks for a THIRD spend — *"per-coarse-cell
canopy occupancy, mean height, mean colour"* — and marks it **NO** (trees cannot ship without it). Three
spends do not fit in fourteen bits.

*Example.* A logger fells a hundred oaks around a landing pad. A pilot at four kilometres reads the
coarse rungs. If the canopy fold has no room in the entry, the far view keeps drawing the forest that is
gone, which is the arrival-pop seam the slice-14 gate exists to catch.

**Fix.** Cost all three spends together before the freeze, in one table, and say which two survive. The
entry is rebuildable on disk but a protocol event on the wire, so the WIRE half of this door is hard.

### F9 — MISSING. The quad-to-triangle rule is not frozen, and it decides what the player stands on.

**Where.** §1 "Physics on the same surface"; §3 slice 6; §3 slice 11.

SL10 V1.5 says *"What the player sees is what the player stands on."* Slice 6 pins *"a pinned quad list
for a fixture chunk"*. Slice 11 builds *"a triangle mesh from the SAME extractor output the client drew
with"*. A quad on a saddle splits into two triangles two ways, and the two ways differ by up to half a
cell in the middle. The client's renderer and the shard's collider each pick a diagonal, and nothing in
any format makes them pick the same one.

*Example.* A prospector stands on a smooth ridge. The client split the ridge quad one way and drew a
crest. The moon's shard split it the other way and collided a hollow. The boots sink into the crest the
player can see.

**Fix.** The triangulation rule lives inside the generator crate beside the extractor, and slice 6 pins a
TRIANGLE list, not a quad list.

### F10 — MISSING. Nothing keeps client-derived trim off the collided surface.

**Where.** §4 V72; V2.6; SL10 V1.5.

The owner granted the client its style: *"Style can be picked up by the client, so when same blocks of
that type are connected and construct the mesh, it will render small additional details, that are not
stored on the BE"* (V2.6). The synthesis answers WHO derives the trim (V72, the client) and never states
the rule that keeps the trim harmless.

*Example.* A shipwright plates a hull. The client bevels every welded seam and adds a rim to each corner
plate. The moon's shard collides the plain plates. A crewmate walks along the hull's spine and the boots
pass through the rim the crewmate can see.

**Fix.** State the rule in the render seam: derived decoration never changes the extractor's output,
never changes a collider, and never moves a vertex the collision mesh shares. Slice 7's gate asserts it
by comparing the geometry the seam hands out with the geometry the collider reads.

### F11 — BREAKS THE NO-MAGIC-NUMBERS RULE. The object horizon is a constant.

**Where.** §4 V80.

V80 recommends *"~800 m"* for the object horizon and admits the cost in the same row: *"a 20 m oak is 22
pixels at 800 m, so the crossover is visible"*. A drawing distance is not an operational parameter. The
visibility ruling makes a drawing distance a consequence of what a thing LOOKS like, and the reach ruling
repeats it: one number, derived from the look, never a constant chosen for a kind.

*Example.* A themed world (V2.7) grows a crystal spire eighty metres tall. At 800 metres the spire still
covers ninety pixels and it drops into a flat fold. Beside it a one-metre shrub is meshed in full detail
at 790 metres, where it covers one pixel.

**Fix.** Derive the horizon per object kind and per growth stage, from the object's own size and the
drawable floor (`crates/core/src/geometry.rs:1156-1173`) — the same rule that already decides a realm's
reach. Keep U-53's crossfade measurement.

### F12 — MISSING. The chunk order and the crossfade band are not placed in the engine-free library.

**Where.** §3 slice 7, §3 slice 8; §4 V44, V46; V2.8.

V46 puts the TIER rule in the client library, so the detail-by-box tolerance stays gateable on any engine.
Slice 8 then lands three more rules — *"coarse-before-fine ordering"*, *"the dither crossfade band"*, and
the rule that the client derives only for a realm with a row in the composed scene — and never says which
side of the seam they live on. V44 puts the worker pool in the render binary.

If the order and the band live in the render binary, an Unreal client writes its own, and SL8's
continuity forks with the engine. V2.8 asks the opposite.

*Example.* A hull descends onto a moon. On the Bevy client the coarse ground arrives first and the fine
cells dissolve in over four frames. On the Unreal client the plugin's own queue delivers fine cells
first, so the ground appears in patches. The same seam gate passes on one client and fails on the other,
and nobody can tell which is the world's fault.

**Fix.** The library owns the ORDER, the tier and the crossfade BAND (which chunk next, at what blend
weight). The engine owns the thread that runs the work and the shader that paints the blend. Slice 7's
seam contract states the split before either encoder is written.

### F13 — MISSING. A swept collider demand has no bound, and a clamp is refused.

**Where.** §3 slice 11; §7 U-42; the 2026-08-27 movement answers.

Slice 11 demands colliders over *"each body's SWEPT SEGMENT for the tick, fattened by the body's bounding
radius plus reach"*, which is right, because the owner deleted the ceiling from the flight path. U-42
measures 30, 250 and 7 000 metres per second. The ruling states the other end of the range: *"the
galaxy's ceiling is ~170 million times the speed of light"*
(`docs/design/owner_decisions_2026-08-27_movement_answers.md`), and nothing on the flight path may clamp,
refuse or stop a speed.

So the design owes an answer for the tick whose swept segment asks for more chunks than the shard can
build. The answer cannot be a clamp, because a re-clamp is a jump and a jump is a seam.

*Example.* A pilot switches off the hull's own safety block and dives at a moon at ten kilometres a
second. In one 50 ms tick the hull travels 500 metres. The shard must know the ground along that whole
line before it can say whether the hull hit the ridge.

**Fix.** Use SL10's own grant. The swept test asks the generator crate for the surface height along the
segment analytically, cell by cell, with no chunk built and no collider allocated; a contact found that
way then builds the ONE chunk it needs. Add the rule to slice 11 and extend U-42 to the speed at which a
hull crosses a whole chunk in one tick.

### F14 — MISSING. The registry digest makes every new theme a fleet-wide flag day.

**Where.** §2.2, §4 V14, §4 V15; V2.7; V2.9.

V15 recommends EQUALITY of the registry digest at the client handshake, and states the cost for one new
substance. V2.7 promises themes: *"Stylised blocks, plants, clothes and weapons for other settings on
other planets."* Under equality, every theme that appends a kind refuses every unpatched client at the
gateway. The register never states that cost against V2.7, and never against the seamless law. A pilot
who is refused at the gateway meets the hardest seam the game has.

**Fix.** State the cost against V2.7 in the register. Offer the owner the alternative the same document
already uses for the STORE (V14): a prefix digest for the client, plus a per-chunk refusal when a chunk
names a kind the client does not hold. Then a themed planet refuses only the pilots who fly to it.

### F15 — MISSING. A realm's block store must travel with the realm, and the cut is not sized.

**Where.** §3 slice 9, §3 slice 15; HR1, HR2.

Slice 9 puts the block families in `StoreRole::RealmStore` (`crates/core/src/store_stamp.rs:64-75`),
which today holds a berth row and a body. HR1 makes that store private and it travels with the realm.
Slice 15 adds `BLOCK_STORE_FLUSH_STEP` to the re-shard drain and stops there. Nothing sizes the move.

*Example.* A fifty-thousand-block hull flies from one star system to another, and its realm re-homes to
another node. The berth row moves in milliseconds. The hull's block store is measured in gigabytes, and
the transfer's cut marker waits behind it.

**Fix.** Slice 15 states the rule — the store's bulk moves BEFORE the cut and the cut carries only the
tail — and adds the measurement: the cut duration for a 50 000-block hull against the transfer's own
budget, and the tick hitch a watching client sees.

### F16 — MISSING. HR5 never gets a tier for the new crates.

**Where.** §3 slice 5, §3 slice 7; HR5.

Slice 5 creates `vd-terrain` and `vd-seed`, and slice 7 links them into `vd-client`. Both slices name HR5
among their laws and neither gate contains a coverage run. HR5 sets Tier-A crates at 100 % region and
branch and documents the monomorphization trap. The `Gf` newtype wraps every arithmetic operation the
generator uses, so its surface is exactly the shape that discipline is about.

**Fix.** Each slice that creates a crate states that crate's tier and adds `just coverage-fast` to its
gate, or records the exemption in `coverage-exemptions.toml` with the reason.

### F17 — UNMEASURED STATED AS MEASURED. The write amplification and the quarry rung.

**Where.** §2.3, §4 V52, §7 U-26.

The document defines its own words: *"MEASURED means a program ran and this document says which."* It
then writes *"a MEASURED 5 508× write amplification per entry"* twice and names no program, while U-26
lists the same quantity as unmeasured (*"Write amplification of one pyramid entry in a built region"*).
The same section writes *"MEASURED by arithmetic on the fold rule"* for the 20-metre quarry, and V20
repeats it. Arithmetic on a cited number is ESTIMATED by the document's own rule.

**Fix.** Re-label both as ESTIMATED and show the arithmetic, or name the program that ran. The
conclusions do not change; the mark does.

### F18 — MISSING. Seed-decided common ore needs the owner to relax S5.1 by name.

**Where.** §4 V36, V37; the 2026-08-27 seed ruling.

The seed ruling's binding sentence is absolute: *"A block's SUBSTANCE may not be a pure function of
(position, seed)"* (`docs/design/owner_decisions_2026-08-27_seed_and_secrecy.md` S5.1). SL10 V1.1 speaks
of *"the seed-decided common materials"*, which relaxes it for common stock. V36 then recommends bulk
PLUS common ore — iron, copper and coal. The recommendation may well be right. It is not the synthesis's
to make quietly, because it turns an absolute sentence into a graded one.

*Example.* A prospector lands on a moon to survey a copper seam. If copper is seed-decided, a second
player computes every copper seam on every moon in the galaxy from the client he already runs, and prints
the map. The survey instrument still exists and nobody needs it.

**Fix.** V36 states in the register that option (B) asks the owner to relax S5.1 by name, and it carries
the ruling's own test: *"if this were printed on a public wiki tomorrow, would the game still work?"*

### F19 — MISSING. Three slices adopt an SL6 request without naming SL6.

**Where.** §3 slice 10, slice 13, slice 14; §5 R-11, R-13, R-14, R-15.

Slice 13 ships a turret's angle to observers and measures *"the bytes per tick to one observer"* — that
is R-11, marked **NO** (the design cannot live without it). Slice 14 ships the felling diff, the canopy
fold and a falling crown — R-13, R-15 and R-14. Neither slice lists SL6 among its laws, and slice 4's
wire plant carries none of those tags. Slice 4 is the only slice that names SL6 at all.

**Fix.** Every slice that puts a new byte on a lane names SL6 and cites the R-number it consumes. Slice 4
either plants the tags, or states that they arrive later behind the skip-unknown rule
(`crates/core/src/tlv.rs:19-23`), which covers tags and never key bytes.

### F20 — MISSING. Slice 7 draws terrain before slice 8 gives it a detail rule.

**Where.** §3 slice 7, §3 slice 8; SL8.

Slice 7 links the crate into the client and builds a mesh from `ChunkGeometry`. Slice 8 lands the tier
rule and the crossfade. Between them the client draws one rung of ground with a hard edge where the
resident band stops. Nobody ships that state to a player, so it is not a defect against SL8 by itself. It
is a gate gap: slice 7 has no seam measurement, so the first pictures of the ground are judged by eye.

**Fix.** Slice 7 states that it draws ONE rung inside one chunk band, and its gate asserts that no second
rung is drawn. The pop detector arrives with slice 8, as written.

---

## What I tried to refute and could not

These three are the strongest attacks I had. Each fails, and I record why, so the next reader does not
repeat them.

**A. "The client deriving the ground breaks SL3, because a realm that is not running cannot be drawn."**
It does not. SL3's sentence is about a realm DRAWING ITSELF, and slice 8 keeps the gate: the client
derives only for a realm that holds a row in the composed scene
(`crates/wire/src/channels.rs:158,179`). A dormant moon has no row, so nobody draws it. The 2026-09-01
visibility ruling stays intact. *Example: a moon nobody has woken is a name in a parent's placement list
and no ground at all, exactly as before.*

**B. "The address breaks SL1, because a cell names a position."** It does not. `CellAddr` names a realm,
a face, a rung and three integer indices, all inside the realm's own frame, and `k` counts metres from
the body's own floor radius — its SIZE, never its place. Nothing in the address can be folded toward a
root, so SL1 clause 4 holds. *Example: the cell under a landing pad on a moon reads the same whether the
moon sits in its own star system or nowhere at all.*

**C. "Two mesh lanes are a second implementation, which the FINAL-backend law refuses."** They are not.
That law refuses a second world and a second implementation selected by CONFIG. The lane here is selected
by the CELL's form, which is data a player wrote, and both lanes end in one collider builder. *Example: a
player sets a steel foundation into a hillside; the dirt goes through the smooth lane, the foundation
through the square lane, and one builder makes both colliders.*

---

## What must happen before slice 0 shuts a format

1. The owner answers §5, not only §4 band A (F1).
2. V38 moves into band A, because HR4 depends on its answer (F3).
3. The body-id allocation rule joins the address freeze (F7).
4. The pyramid's three candidate bit spends are costed together (F8).
5. The triangulation rule joins the extractor's freeze (F9).
