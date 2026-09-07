# Law verdict — 09 Physics, collision and the controller on a planet

**Date:** 2026-09-07. **Lens:** the law refuter.
**Report under test:** `docs/investigation/2026-09-07/09_physics_controller.md`.
**Verdict: REFUTED.** Three load-bearing claims are wrong or break a law. Four required things are
missing. The report's foundation — a trimesh terrain collider from the one generator, colliders only
around bodies, gravity as a function of position — SURVIVES. The frame ruling, the tolerance claim and
the child-count cost do not.

**How to read this file.** MEASURED means somebody ran a command or read a line, and this file says
which. ESTIMATED means arithmetic on stated inputs.

---

## 1. REFUTED — the body-fixed realm frame (§2.5, §6 decision 7) breaks the realm's own children

**The claim.** *"the planet realm's own frame is BODY-FIXED"* (line 226-229), and *"Fictitious forces
in the body-fixed frame: centrifugal `ω²·r` is 0.034 m/s² at Earth's equator … The controller ignores
it in v1"* (line 230-232). Decision 7 recommends yes (line 425-426).

**Why it is wrong.** A planet realm's frame is not only the frame the boots stand in. It is the frame
in which the planet AUTHORS every child's placement, out to the planet's own bound. The code says what
that bound is: a planet's `shape` is its gravitational sphere of influence
(`crates/physics/src/worldgen/generate.rs:1795-1797`, `planet_soi(...)`), and a MOON is a child realm
of a planet (`generate.rs:1792-1793`, *"A MOON IS A PLANET (the ruling, literally): same kind, parent =
a planet"*). So the report measured the fictitious force at ONE place — the equator — and then applied
the answer to a realm that reaches a thousand times further.

The numbers, all ESTIMATED from Earth's rotation rate (7.2921 × 10⁻⁵ rad/s) and the code's own shapes:

| Place inside the planet realm | Centrifugal `ω²·r` |
|---|---|
| The equator, 6 371 km (the report's own case) | 0.034 m/s² |
| The Moon's orbit, 384 000 km | 2.04 m/s² |
| The edge of the planet's bound, ~924 000 km | **4.91 m/s²** |

At the edge of the realm the force the report ignores is HALF the pull it carefully models. Coriolis
(`2·ω×v`) never appears in the report at all; for a hull at 7 km/s it is ESTIMATED at 1.0 m/s².

**The report's own example refutes it.** Line 235-238 says: *"A moon orbits a planet at 1 km/s. … The
planet's shard moves the moon's placement 20 m per tick."* In a body-fixed frame the spin ALONE sweeps
that moon's authored placement **560 m per tick** (ESTIMATED: 7.2921e-5 × 3.84e8 × 0.02 s). The example
and the decision cannot both be true.

**The consequence in the game's words.** A pilot flies a hull out from a moon toward the edge of the
planet's realm. Under decision 7 the planet authors his placement in a frame that turns once a day, so
the planet must add a sideways push of about five metres per second per second that no engine made and
no player asked for — or the hull flies wrong. The swept containment test reads the LINE from the last
tick to this tick (`crates/core/src/geometry.rs:1527`, `region_verdict`); a moon whose authored
placement jumps 560 m a tick from spin alone makes that line long enough to cross bands it never
physically crossed.

**Fix.** Separate the two frames and say so. The COLLIDER frame is body-fixed, because the terrain must
not move under a boot. The AUTHORING frame — the frame the planet writes its children's placements in —
stays non-rotating, and the planet's spin lives in the placement the planet's own parent authors for it,
exactly as the report says at line 227-228. Then a hull near the bound feels only gravity, a moon's
placement moves the 20 m the example claims, and Coriolis never appears anywhere. Put BOTH frames in
decision 7 and name which one the rapier world uses.

---

## 2. REFUTED — "the geometric error is ZERO by construction" (§1.5) is an argument, and it is false on a branch the report leaves open

**The claim.** *"The vertices are FINE-lattice integers on both hosts, so the geometric error between
the triangle the client draws and the triangle the server collides is **zero** by construction"*
(line 133-135).

**Why it is refuted.** Three reasons.

1. **The standing rule.** No-drift is a MEASUREMENT, never an argument (SL10 rule 3). "By construction"
   is an argument. The report knows this and writes the words anyway.
2. **The gate does not test the claim.** Row S2 (line 467) compares *"the server-built trimesh
   vertex/index bytes"* with *"the client-built mesh bytes"*. That measures the GENERATOR's output on two
   builds. It never measures what the renderer DRAWS. The sentence claims identity of the drawn triangle
   and the collided triangle; no row measures that.
3. **It is false on the f32 branch.** Decision 1 (line 411-412) leaves rapier3d (f32) open. A FINE cell
   is 1/1024 m (MEASURED constant, `crates/core/src/pose.rs:255`). At an Earth radius the vertex index
   is 6.52 × 10⁹ FINE units, which needs 32.6 bits (ESTIMATED). An f32 mantissa holds 24. The spacing
   of f32 at 6.371 × 10⁶ m is ESTIMATED at 0.76 m — about 780 FINE cells. So on the f32 branch the
   error is not zero; it is nearly a metre. The same arithmetic applies to any client whose renderer is
   f32, which every GPU pipeline is, and the report states no client anchoring rule at all — while V2.8
   says the design may not assume Bevy.

**In the game's words.** A player walks up a hill on a moon. If the client draws the hill from f32
vertices taken at the moon's true radius, the drawn slope and the collided slope disagree by most of a
metre, and the boots float or sink. The report promises one millimetre.

**Fix.** Say that the identity holds for the GENERATOR's integer output and nothing else. Then add the
missing rule: the client re-expresses the chunk's FINE integers relative to a chunk-local origin before
it hands them to a renderer, and the server does the same before it hands them to rapier. Add a
measurement row S2b: drop a capsule on the SERVER's collider and read the same point off the CLIENT's
uploaded vertex buffer, on x86-64 and aarch64, and on the f32 branch as well as f64.

---

## 3. MISSING — the owner's tree requirement (V2.2) has no collider

**The requirement, in the owner's words:** *"they should be placed as one object/block on the surface,
… but then the collisions are calculated on the server, so server somehow should know the shape"*
(V2.2). That sentence is ABOUT COLLISION. It belongs to this report's domain and to no other.

**What the report says.** Nothing. The word "tree" appears once in the report, and it means the source
tree (line 21). MEASURED: `grep -n "tree" docs/investigation/2026-09-07/09_physics_controller.md`
returns one line.

**Why it is a hole, not an omission.** §1.3 opens with *"A cell is either a terrain-field cell or a
built cell. It is never both"* (line 83). A seed-placed pine is neither. It is one object on one surface
cell whose trunk and canopy stand metres above that cell. It has no arm in the report's collider
taxonomy (trimesh terrain, parry `Voxels` for a hull, `Compound` for the ~20 shapes, boxes for
sub-metre blocks, no collider for an attachment). It has no row in the chunk cost model (§1.4, line
121-124), and no row in the eleven measurements (§7).

**In the game's words.** A player lands a hull in a forest on a moon. The client draws a hundred pines
from the moon's seed. The moon's shard holds a trimesh of the ground and nothing else, so the hull
settles through every trunk, and a player walks through a pine he can see.

**Fix.** Add a fourth collider arm: a seed-placed feature states its own collision shape from the same
generator crate — a trunk capsule plus a canopy shape from the feature's seed parameters — built by the
same chunk ring, cached with the chunk, evicted with the chunk. Add its bytes to §1.4's cost model and
a row to §7 that measures a forest chunk against a bare chunk.

---

## 4. BREAKS HR4 — no second shard kind, no G-IDENTICAL fixture, and the seam the report deletes is HR4's owed half

**MEASURED:** the strings `HR2`, `HR3`, `HR4`, `HR5`, `HR6` and `G-IDENTICAL` appear nowhere in the
report except the boilerplate line 7.

HR4 says a feature *"passes the identical fixture on ≥2 shard kinds (G-IDENTICAL) or it doesn't land."*
The terrain trimesh lane lives on a planet realm only. The report never names its second shard kind and
never names its fixture.

Worse, the code names the seam that carries HR4 for exactly this feature, and the report proposes
deleting it without saying so:

- `crates/sim/src/capability.rs:10-11`: *"Geometry differences are confined to `FrameSpace` impls
  selected by `voxel().geometry` (P4/P5)."*
- `crates/sim/src/capability.rs:13-18`: *"KNOWN LIMIT (DEFERRED [[D-38]]): this is HR4's STRUCTURAL
  half … The variant forcing a `reanchor()` stays owed at P5 (`FrameSpace` does not exist yet)."*
- `crates/sim/src/capability.rs:34-42`: `VoxelGeometry::Spherical` / `Cartesian` — *"the ONE seam where
  spherical planets and Cartesian ship grids differ."*

The report's §1.3 forks collider construction by that very grid family — parry `Voxels` on a hull's flat
grid, warped convex parts on a planet's cube-sphere grid — and then §6 decision 1 recommends f64 because
*"The physics side then needs no anchor and no reanchor at all"* (line 155-156). The report is right that
f64 deletes the re-anchoring. It never says what then holds the two grid families as ONE machinery, and
it never says that the thing it deletes is the half of HR4 the code has ledgered as owed.

**In the game's words.** A player builds a wall on a moon and the same wall inside his hull. Under HR4
the same fixture must place both, and only the realm's stated geometry may differ. The report does not
name that fixture, so nothing proves the two paths are one.

**Fix.** Name the second shard kind (a hull realm) and the G-IDENTICAL fixture (one identical
placement-and-walk body run on a planet realm and a hull realm). State whether `FrameSpace` survives as
the grid seam after the anchor half dies, or what replaces it, and cite `capability.rs:13-18` when you
do.

---

## 5. BREAKS SL9 — an unbounded child count is never costed or measured

**What the report serves.** §1.4 and row S5 cost the CHUNK colliders against the number of OCCUPANTS and
the planet's radius. That half is sound and the measurement can fail, which is right.

**What it never serves.** §3.2 puts every landed child realm's exterior shell into the parent's rapier
world as a rigid body, and §4.1 says *"The bodies it can see are the realm's own occupants and the
exterior shells of child realms it holds"* (line 322-323). SL9 says a parent may hold *"six hundred ships
and stations"* and forbids *"a per-tick walk of all of them"*. A rapier world steps every body it holds,
every tick, and holds every convex decomposition in memory. The report states no sleeping rule; the word
"sleep" does not appear (MEASURED by grep). Row S5 measures 1, 10 and 100 OCCUPANTS. No row measures a
realm with many CHILDREN.

**In the game's words.** Six hundred hulls sit parked at a spaceport on a moon. Each states its exterior
shell to the moon on the slow lane. The moon now holds six hundred convex decompositions and steps six
hundred bodies every tick, for six hundred hulls with their engines off.

**Fix.** State the rule: a child realm's shell body sleeps the moment its authored velocity and its
contact set stop changing, and it wakes on a contact or a stated drive. A sleeping body is not stepped.
Add a row S12: 1 / 100 / 600 parked hulls in one realm, report bytes and core time per tick, and demand
that the per-tick cost does not grow with the parked count.

---

## 6. BREAKS SL8 — a one-block lift pop is accepted and its tolerance is deferred

§4.2 line 338-341: *"the hull rests on its old shell for one to a few ticks; the change is one block, so
the pop is at most one block of lift — below the SL8 tolerance the P6 slice must state, and measured
there."*

A block is one metre. A hull that lifts by one metre in one tick is a JUMP, which is the first seam kind
in the taxonomy. SL8 says a tolerance is PHYSICAL, and no physical tolerance admits a metre. The report
declares the pop below a tolerance that does not exist yet, which is the shape SL8 forbids: never ship a
capability without its continuity.

**In the game's words.** A player welds one plate onto the belly of his parked hull. The hull he is
standing next to jumps a metre into the air, and so does he if he was on the ramp.

**Fix.** State the continuity now, not at P6: the parent interpolates the shell swap over the ticks the
rebuild costs, or the hull states the new shell BEFORE the edit applies so the two shells overlap for
one tick. Then measure the lift and demand it stays under one FINE cell per tick, the same bar row S4
already sets for a chunk boundary.

---

## 7. MISSING — SL1 clause 6 against decision 3: what a surface realm does when its reading goes stale

Decision 3 (line 416-418) and §6.3 item 3 let a spaceport Area read the placement its parent stated and
use it as an address into the generator. The report reads SL1 clause 5 correctly: a terrain sampler is
not the placement, containment or crossing machinery, so the clause does not fence it.

Clause 6 is the one the report does not answer: *"A STALE reading is REFUSED, never used."* If the
area's stamped placement goes past its bound, the area may not use it — and the area's colliders were
built from it.

**In the game's words.** A hull sits on the pad of a spaceport Area on a moon. The moon's shard stumbles
and the area's stamped placement goes stale. Under clause 6 the area refuses the reading, so it cannot
say which cells are under it, so the pad has no collider, and the hull falls through the world the
client is still drawing.

**Fix.** State the degrade: the area KEEPS the colliders it already built from the last fresh reading
and stops building new ones, and it says so. A collider is a thing already made, not a fresh statement.
Write that sentence into decision 3 so the owner rules on it.

There is a second, smaller gap in the same decision. The area builds its colliders from the SEED only.
The planet owns the edits (decision 3's own last clause). A tunnel another player dug under the pad
last week never reaches the area, so the area's pad is solid where the client draws a hole. State that
the planet ships the area the DIFF for the cells under it, one hop, and record it as an SL6 ask.

---

## 8. WRONG numbers and over-claims

| Claim | Verdict | Evidence |
|---|---|---|
| *"handed the hull to the planet a kilometre up"* (line 399-400) | WRONG | A planet realm's bound is its gravitational sphere of influence (`crates/physics/src/worldgen/generate.rs:1795-1797`), not its surface. For an Earth-like planet the hand-over happens ESTIMATED ~924 000 km up, not one kilometre. The error is generous — the planet gets far more time to build colliders than the example claims — but the owner is told the wrong thing about when a crossing happens. |
| *"at orbital speed near a small moon, 140 m per tick"* (line 380) | WRONG | 140 m per 20 ms is 7 000 m/s. That is low-orbit speed at an Earth-sized body. Orbital speed near a 500 m asteroid is ESTIMATED well under 1 m/s. The sentence names the wrong body for the number. |
| *"the code ticks at 50 Hz"* and *"every 'at 20 Hz' figure is wrong by 2.5× against the 50 Hz code"* (line 30, 480) | OVER-CLAIM | `tick_hz: 50` is a field of the DEV cluster constant (`crates/bins/src/lib.rs:412-414`, the struct is named `DEV`). `crates/core/src/kinematics.rs:148-149` says the opposite of a fixed rate: *"`tick_hz` is a per-shard knob passed in (a global `TICKS_PER_SECOND` const…"* — and `crates/wire/src/channels.rs:785` carries a 20 Hz `UniverseRate`. There is no single shipped tick rate to divide by. |
| parry `Voxels` at *"~1 B per voxel, seam classification built in"* (line 89-90) and *"Parry has no signed-distance-field shape"* (line 66) | UNMEASURED AS FACT | Neither parry nor rapier is in `Cargo.lock` (MEASURED: `grep -n 'name = "rapier\|name = "parry' Cargo.lock` returns nothing — the report's own §0 says so). Both claims come from the stale base, and both decide a collider lane. Add them to §7 as a row: read the pinned parry version's shape list and state it. |
| *"reach"* used for three different things | AMBIGUOUS | Line 109 (`reach + body extent`, a physics ring), lines 318 and 336 (an arm's reach for a placement), lines 272 and 443 (the realm reach of the 2026-09-02 ruling, on the slow lane). REACH is a law word. One word, one meaning. |

---

## 9. Citation drift in §0 — every substantive claim STANDS; six line numbers are off

I re-ran every §0 grep. Every claim in the table is TRUE. Six citations point a few lines away from
the thing they name. Fix them so a reader who follows one does not think the claim moved:

| Cited | Actual | MEASURED by |
|---|---|---|
| `built.rs:37-59` (`BuiltFacts`) | 37-54 | `sed -n '37,54p' crates/core/src/built.rs` |
| `drive.rs:139-183` (`advance_driven`) | 139-182 | `grep -n "pub fn advance_driven" crates/sim/src/stub/drive.rs` |
| `ghost.rs:52` (`GhostColliderRegistration`) | 55 (52 is inside its doc comment) | `grep -n "pub struct GhostColliderRegistration"` |
| `motion.rs:52-55` (the Kepler arm) | 49-52 | `sed -n '49,52p' crates/physics/src/motion.rs` |
| *"5 doc-comment mentions"* of `FrameSpace` etc. | 6 lines, in 3 files | `grep -rn "FrameSpace\|SphericalSpace\|SurfaceAnchor\|AnchorGen" crates --include='*.rs'` |
| `drive.rs:353-355` (*"the parent holds the last drive"*) | 349-350; 353-355 is the doc comment of `on_child_facts` | `sed -n '345,355p' crates/sim/src/stub/drive.rs` |

The two claims I checked hardest both hold. No planet spins: the Kepler arm returns
`FramePlacement::moving`, and that constructor writes `orientation: DQuat::IDENTITY` and
`angular_velocity: DVec3::ZERO` (`crates/core/src/frame.rs:92-101`). The re-clamp is deleted, and the
code says why in the owner's own words (`crates/sim/src/stub/transient.rs:230-236`).

---

## 10. What the report gets right, and must keep

These survived every test I could put to them. They are not findings; they are the parts the next
draft must not lose.

1. **SL1 stays whole.** No chain folds an absolute. A planet realm is centred on itself, so the vector
   from its own origin to a boot IS the radial (line 167-169), and no placement is needed to know
   "down". Nothing in the report passes on a reading.
2. **SL2 stays whole.** No occupant pose crosses for collision. A crew member inside a parked hull is
   protected from a planet-side block by the HULL'S SHELL, never by the crew member's pose (line
   453-455). That is the right shape, and the report is right to record it so nobody reopens it.
3. **SL4 stays whole.** The crossing path is untouched: containment stays the swept integer test
   (`crates/core/src/geometry.rs:1527`) and the index still answers a segment
   (`crates/core/src/child_index.rs:171`). Gravity became a function of position INSIDE the parent's
   integrator, where physics belongs, and never on the crossing path.
4. **The movement contract stays whole.** No velocity crosses upward. The parent never chooses a speed:
   the report keeps `advance_driven`'s shape (`crates/sim/src/stub/drive.rs:139-182`) and adds only
   contact. The collider ring GROWS WITH SPEED and never caps it (line 388-389), which is M-B stated
   correctly.
5. **SL5 stays whole.** The measurements run on the starter world and on real bodies. No reduced world
   and no second generator appear anywhere.
6. **The seed ruling stays whole.** Nothing in the report makes a valuable substance a pure function of
   position and seed. Terrain SHAPE is the static shape SL10 allows.
7. **The two SL6 asks are stated the way SL6 demands** — data, direction, rate, why the receiver cannot
   compute it, and the cost of doing without (line 441-448). The exterior-only rule (line 272-274) is
   the right minimum: a parent may touch a hull only from outside, so no interior void crosses.
8. **No new library is adopted silently.** §1.6 says plainly that adding rapier is the owner's word
   (line 148-149).

---

## 11. The one thing the report should have put to the owner and did not

Decision 4 (line 419) recommends the placement preview and calls it *"a drawing, not state"*. The
convention list says *"NO client-side prediction"* (CLAUDE.md, Non-negotiable conventions), and SL10
rule 7 says the client never derives state. A preview draws a block the server has not confirmed. The
report notes it *"looks like prediction and is not"* (line 363) but leaves it out of §6.3, the law
conflicts list.

**In the game's words.** A player holds a plate against a hull's belly and sees a ghost of it before the
hull's shard says yes. If the shard refuses, the ghost vanishes and nothing else moves — no rollback,
no jump. That is very likely lawful. It is still a client drawing something no shard has stated, and it
belongs in the law conflicts list where the owner will read it.

**Fix.** Move it into §6.3 as item 5, with the same four fields the other asks carry.
