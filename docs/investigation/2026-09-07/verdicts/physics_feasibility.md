# Feasibility refutation — 09 Physics, collision and the controller on a planet

**Date:** 2026-09-07. **Lens:** technical feasibility.
**Target:** `docs/investigation/2026-09-07/09_physics_controller.md`.
**Verdict: REFUTED.** Four load-bearing claims are wrong or break a law. Five more carry a decision on
an unmeasured number. Four requirements of the domain are absent.

Every number below is marked. MEASURED means I read the line or the ruling and name it. ESTIMATED
means arithmetic on stated inputs, and the inputs are stated.

---

## 1. The §0 code table is honest. It STANDS, with three small corrections

I re-ran the greps. The table is true.

- No physics engine is in the tree. `grep -n 'name = "rapier\|name = "parry' Cargo.lock` returns
  nothing (MEASURED). The one mention is the profile comment at `Cargo.toml:96` (MEASURED).
- `advance_driven` is at `crates/sim/src/stub/drive.rs:139-183`, and `Ambient` holds one pull vector
  and one density at `drive.rs:112-118` (MEASURED).
- The swept verdict covers `Shell` and `Aabb` and skips `Obb` at
  `crates/core/src/geometry.rs:1544-1551` (MEASURED). The exclusion list at `geometry.rs:8-27` says
  the same (MEASURED).
- `FINE_CELL_EDGE_M = 1/1024` at `crates/core/src/pose.rs:255` (MEASURED).
- The subjective-time step is at `crates/sim/src/stub/placement.rs:60`, exactly as cited (MEASURED).

Three corrections, none load-bearing:

1. `governed_ceiling_for_frame` is called at `crates/sim/src/stub/dot.rs:465`, not `:463` (MEASURED).
2. `GhostColliderRegistration` is declared at `crates/sim/src/stub/ghost.rs:55`, not `:52`, and about
   twenty sites reference it, not one (MEASURED). The report's substance is right: the type holds no
   collision behaviour, and `ghost.rs:9-10` says so.
3. `crates/physics/src/celestial.rs:151` is the field `central_mass` of a child's orbital elements —
   the PARENT's mass, not the planet's own (MEASURED). A planet's own mass is generator data
   elsewhere. The claim survives; the citation points at the wrong thing.

**Example.** A moon's shard can still read its own mass from the seed and compute its own pull. The
report's conclusion holds. Its evidence line does not.

---

## 2. REFUTED — "zero by construction" rests on a rule the owner never wrote

**The claim.** §1.5: *"The vertices are FINE-lattice integers on both hosts, so the geometric error
between the triangle the client draws and the triangle the server collides is zero by construction."*
§1.2 attributes the source rule to **SL10 rule 4**: *"The extractor emits vertices on the FINE lattice
and a fixed index order (SL10 rule 4)."*

**The evidence.** I read SL10 rule 4 (`docs/design/owner_decisions_2026-09-07_voxels.md:36-39`,
MEASURED). It says five things: integer hashing for every random draw, fixed evaluation order, no
fast-math flag, no fused-multiply-add contraction, and no call into the platform's transcendental
functions. **It says nothing about a vertex lattice. It says nothing about an index order.** The same
text appears in `CLAUDE.md:227-236` (MEASURED) and names neither.

**Why this is load-bearing.** The report's headline (§9: *"the boot stands on the drawn triangle to
within one millimetre by construction"*), its tolerance (§1.5), and two of its eleven gates (S2 and S3)
all rest on the FINE-lattice vertex rule. The rule may be the right design. It is the report's own
proposal, and the report presents it as an owner ruling already in force.

**The fix.** Move the vertex-lattice rule and the index-order rule out of the SL10 citation and into
§6.1 as a NEW open decision with its own one-way door. State how a smooth extractor produces a
lattice-integer vertex: a surface crossing between two density samples lands between two lattice
points, so the extractor must snap it, and the snap changes the surface the boot stands on. State that
error. Today the report states zero.

**Example.** A player stands on a hillside on the starter moon. The client drew the slope from the
seed. The moon's shard built the same slope. The report promises the two triangles are the same bytes.
Nothing in the owner's ruling makes them so, and no line of the report says how the extractor rounds
the crossing point.

---

## 3. REFUTED — decision 7 makes the whole realm a spinning frame, and the integrator has no spin terms

**The claim.** §6.1 decision 7 and §2.5: *"the planet realm's own frame is BODY-FIXED"*, and
*"centrifugal `ω²·r` is 0.034 m/s² at Earth's equator (ESTIMATED, 0.3 % of g). The controller ignores
it in v1."*

**The evidence.**

1. A planet realm's bound is its own gravitational SOI:
   `crates/physics/src/worldgen/generate.rs:1796-1798` writes `Boundary::Shell { r: planet_soi(...) }`
   (MEASURED). The realm's frame therefore reaches to the SOI, not to the surface.
2. `advance_driven` (`drive.rs:139-183`) sums exactly three terms: the rotated push, the realm's
   pull, and drag (MEASURED). There is no Coriolis term. There is no centrifugal term. There is no
   Euler term. `Ambient` (`drive.rs:112-118`) carries no angular velocity (MEASURED).
3. The report never writes the word "Coriolis" (MEASURED: `grep -ci coriolis` returns 0).

**The arithmetic the report did not do.** For an Earth-class planet, ω = 7.292 × 10⁻⁵ rad/s and the
Hill-class SOI is about 1.5 × 10⁹ m. Then ω²·r = 8.0 m/s² (ESTIMATED, from those two inputs). That is
about 0.8 g, not 0.3 % of g. The report measured the fictitious force at the ONE radius inside the
realm where it is smallest against gravity, and then made "ignore it" a realm-wide rule.

For a hull flying at 250 m/s inside the realm, the Coriolis term 2ωv is 0.036 m/s² (ESTIMATED). Over
one minute of a descent that is about 65 m of sideways error (ESTIMATED). The hull does not land where
the pilot aimed.

**The unbounded case.** A hull in a circular orbit inside the planet realm is a normal thing in this
game. Integrated in a spinning frame with no fictitious terms, its ground track is wrong by the whole
rotation of the planet. A hull parked over one continent drifts around the world in a day.

**What is right in the code.** `transfer_frame` DOES carry the `ω × r` term at a crossing
(`crates/core/src/frame.rs:64-66` declares `angular_velocity`; `frame.rs:243-244` adds
`from.angular_velocity.cross(lever)`) (MEASURED). So the hull's velocity is converted correctly at the
shell and then integrated wrongly for every tick after it. That is worse than a uniform error: the
hull is right at the boundary and drifts inside it, so the defect shows as a mismatch between what the
star system saw and what the planet reports back on the way out — a jump at the ruler switch, which
SL8 names a seam.

**The fix.** One of three, and the owner must pick:

1. Split the frames. The COLLIDER frame is body-fixed. The realm's DYNAMICS frame is inertial (spin
   axis fixed, no rotation), and only the terrain collider carries the body's rotation as a moving
   static set.
2. Keep one body-fixed frame and add the fictitious terms to `advance_driven`: centrifugal
   `ω × (ω × r)`, Coriolis `2ω × v`, and the Euler term. `Ambient` then gains an angular velocity, and
   §2.1's line *"that is the only change to the parent-side integrator"* is false.
3. Bound the realm's body-fixed frame to a radius where the terms are below the tolerance, and hand
   anything above it to a second, inertial frame. That is a new seam and needs its own ruling.

The report recommends decision 7 with none of these.

---

## 4. REFUTED — the fast-flight collider ring is not "bounded", and its escape hatch does not exist on a moon

**The claim.** §5: *"At 7 km/s the same hull demands ~700 builds per second ≈ 350 ms per second:
heavy, bounded, and only while a body flies at orbital speed inside the surface ring, which the realm's
air (drag, M2a) ends quickly."*

**The arithmetic.** The ring radius is `|v| · dt · k_lead` (§1.4 rule 1). At 7,000 m/s, dt = 0.02 s and
a modest `k_lead` of 2 ticks, the radius is 280 m — about 8.75 chunks at a 32 m chunk edge. The swept
tube is therefore about 18 chunks wide across the flight path. The tube advances 7,000 / 32 = 219
chunk lengths per second. New surface chunks per second ≈ 18 × 1 × 219 ≈ **3,900** on perfectly flat
ground, and more over a ridge, because a vertical face puts several surface chunks in one column
(ESTIMATED, method stated). At the report's own 0.5 ms per build that is about **2 seconds of one core
per second of flight** (ESTIMATED). The report says 350 ms. It is optimistic by about 5×, and the
honest floor already exceeds one core.

The same method at 250 m/s over a mesa — the report's own example — gives roughly 100–160 builds per
second, not 24 (ESTIMATED), because the tube's cross-section is two-dimensional and a cliff face puts
three to five surface chunks in one column.

**The escape hatch fails.** The report ends the case with the realm's air. `Ambient.density_kgpm3` is
zero in space and the drag term is then exactly zero (`drive.rs:117-118`, MEASURED). **A moon has no
air.** Low orbit over an airless moon is about 1.7 km/s and never decays. The report's own worked
examples in §1.2 and §2.5 both put the player on a moon.

**The rule has no back-pressure.** §5 states: *"A body must never be stepped into a chunk whose
collider is not built. If the build is late, the ring was too small."* Growing the ring at a fixed
speed makes the build rate WORSE, not better, because the tube gets wider. So the stated remedy makes
the stated defect larger, and M-B forbids capping the speed. There is no third option in the report.

**The fix.** State the real answer and put it to the owner: build colliders only where the swept tube
meets the surface (an altitude test the report's ring formula does not have), and state what happens
when the build rate still exceeds the shard's budget. A hull skimming an airless moon must not tunnel
through a crater rim.

---

## 5. REFUTED — the trimesh size and the "no-drift" gate are written for the flavour the report rejects

**The claim.** §1.2 costs a chunk collider as *"f32 vertices 12 KB, indices 24 KB, BVH ~100 KB ⇒ ~140
KB"* and a walker's ring at *"0.6–1.2 MB"*. §1.6 recommends **rapier3d-f64**.

**Two problems.**

1. **The numbers are for f32.** With f64 vertices the vertex block doubles to 24 KB and the tree's
   bounding boxes double with it, so the per-chunk figure is about 240 KB and the walker's ring about
   1.2–2.4 MB (ESTIMATED, by doubling the report's own inputs). The S1 pass of *"≤ 256 KB per chunk"*
   then has no margin at all. A gate with no margin fails on the first rough chunk.
2. **f32 vertices cannot hold a planet's surface in the realm frame.** At a 6,371 km radius the f32
   step is 6.371 × 10⁶ × 2⁻²³ = 0.76 m (ESTIMATED). So the "A. Trimesh" column's word **"Exact"** is
   true only in a chunk-local frame, which the report never states. Under the rejected f32 flavour the
   whole recommendation collapses into the anchor seam the report says f64 deletes.

**The gate cannot be run as written.** S2 asks that *"the server-built trimesh vertex/index bytes equal
the client-built mesh bytes"*. The server holds a `TriMesh` for the solver. The client holds a vertex
buffer for a renderer, in whatever type the engine wants, and V2.8 says the engine may be Unreal. Two
different structures never share bytes. The gate must compare the GENERATOR crate's canonical output on
both hosts — an integer form — and then each host builds its own structure from it. As written, S2 is
red for a reason that has nothing to do with drift.

---

## 6. UNMEASURED AS FACT — five numbers that carry a decision

1. **The skin of one FINE cell (0.977 mm), §1.5 and gate S3.** The report sets the character skin and
   the penetration gate to one FINE cell without stating the solver's own allowed penetration. Every
   contact solver keeps a small slop to stay stable, and it is of the same order as a millimetre. The
   gate may therefore be red for the engine's default and not for the design. The report also cites the
   base's 15.6 mm skin as *"derived from f32 quantisation at a 4,096 m anchor"*; the f32 step at
   4,096 m is 0.49 mm (ESTIMATED), which is 32× smaller, so the base number is not that derivation.
   **Fix:** state the solver's slop as an input, then set the gate above it.
2. **The 33 ms client re-extract, §4.3.** The base's 33 ms is a BUDGET (`block_system_design.md:6193`
   states *"§5's budget is 'edit echo → pixels ≤ 33 ms'"*, MEASURED), and it was a budget for remeshing
   CUBE faces. Under V2.1 the client extracts a smooth surface from a density field, which is a
   different and heavier job. The report puts the budget into a sum and reports **~145 ms click to
   pixel**. A target used as a cost is not a measurement.
3. **The 0.2–0.5 ms trimesh build, §1.2**, carries §5's whole cost model and S6's pass. It is marked
   UNMEASURED, and then every downstream sentence reads as if it were settled.
4. **The 8-chunk walker ring, §1.4's example.** *"About the same eight chunk colliders"* has no stated
   chunk edge and no stated reach. It is the sentence a reader will quote to the owner.
5. **The 51 ms "Downlink + gateway" row, §4.3.** Half of an 80 ms round trip is 40 ms, and the gateway
   hop is already charged 11 ms one row above. The row is unexplained and the total depends on it.

---

## 7. MISSING — four things the domain needs and the report does not name

1. **Trees and large vegetation have no collider (V2.2).** The owner wrote: *"the collisions are
   calculated on the server, so server somehow should know the shape."* The word "tree" appears once in
   the whole report, in the phrase "in the tree" about the source tree (MEASURED). A tree's collider
   lives in the planet realm's rapier world beside the terrain trimesh; it must be demanded by the same
   ring, cached by the same rule, and rebuilt when a player mines the cell under its root. The
   collider-cache rule (§1.4, *"one collider per chunk per realm"*) accounts for the terrain and for
   placed blocks, and for nothing else. **Example:** a player fells a tree on the rim of a pit another
   player dug. Whose collider changes, and who rebuilds it? The report does not say.
2. **No thread and no per-tick budget.** The word "thread" does not appear (MEASURED). §1.4 says *"the
   shard evaluates the generator ... extracts, builds"*. A shard ticks at 50 Hz, so it has 20 ms. A
   descent demands tens of chunk builds in one tick at 0.5 ms each. If the build runs on the tick
   thread, that is a tick hitch, which SL8 lists as one of the eleven seams. If it runs on a worker,
   the report owes the rule that keeps §5's *"never step a body into an unbuilt chunk"* true across
   threads. Neither is stated. This is the frame-budget question, unanswered.
3. **The rotating joint and the turret (V2.5) break §2.5's own rule.** The owner asked for *"a rotation
   joint between two blocks that turns what is built on it by a signal (a manipulator, a
   remote-controlled turret)"*. §2.5 states the rule that makes the design work: *"The terrain must be
   static in the frame the collider lives in."* A turret is a set of blocks that MOVES in the hull's own
   frame, and a HUD attached to it (§1.3: an attachment has no collider) rides along. Worse, a turning
   turret changes the hull's EXTERIOR shell, and §3.2 asks the owner for that shell on the reliable
   slow lane *"on change"*. A turret that turns changes it every tick. That is a lane flood, which SL8
   also lists. The report does not name the case.
4. **Support removed after placement.** §1.3 says a cell is either terrain or built, never both, and
   §4.1 lists "support" among the placement checks. Nothing states what happens when a player mines the
   terrain cell UNDER a placed block. Does the built cell fall, float, or refuse the mine? This is the
   named "mined cell under a placed block" case and it decides whether a structure needs a settling
   simulation at all.

Two smaller gaps, recorded:

- **Two integrators for one hull.** §3.1 says the planet *"integrates the hull exactly as
  `advance_driven` does today (push, pull, drag), plus the contact forces rapier resolves"*. Rapier
  integrates its own rigid bodies. The report never says whether `advance_driven` stays and feeds
  rapier a force, or whether rapier replaces it. If it is replaced, one hull uses two different
  integrators either side of a crossing, and its path bends at the shell — a seam.
- **The walk has no migration.** Today an occupant's step happens per input datagram
  (`dot.rs:238` → `dot.rs:442`, MEASURED), not per tick. A rapier controller wants a fixed step. The
  report defers this to the suit ruling's S6 and never states the seam between a per-datagram walk and
  a fixed-tick solver.

---

## 8. What STANDS

These claims survive the check and should not be reopened.

- **The trimesh is the right shape for smooth terrain, and a heightfield is not.** A cube-sphere face
  is not a height function and a mined tunnel is not a height function. Correct.
- **The collider cache obeys SL9.** Colliders exist only around dynamic bodies, one per chunk per
  realm, reference-counted, evicted after a cooldown (§1.4). No term reads the planet's radius. The
  cost model is stated so it can be measured, and S5 measures it on a 500 m asteroid and an
  Earth-sized body. This is the report's best section.
- **Gravity as `G·M/r²` from the realm's authored mass** is right, and `surface_gravity_mps2` at
  `crates/physics/src/taxonomy.rs:750` already computes the surface value (MEASURED).
- **Up is the radial in the realm's own frame**, and no placement is needed to know down. Correct
  under SL1.
- **Slope and step thresholds as character data, not constants**, is right and follows the
  no-magic-numbers rule.
- **The two SL6 asks are correctly framed.** The hull's exterior shell and its inertia are stated with
  the data, the direction, the rate, why the receiver cannot compute them, and the cost of doing
  without. The owner can rule on them as written. The exterior-only restriction is a good answer to
  the hull-scan leak. §7's rule 3 caveat applies: a turning turret changes the shell.
- **The stale-claim table (§8)** is accurate where I checked it. The base's `Obb` hole is still open
  (`geometry.rs:8-18`, MEASURED) and the report keeps it ledgered rather than claiming it closed.
- **rapier3d and rapier3d-f64 are named as an owner choice, not adopted.** §1.6 says plainly that
  adding the crate is the owner's word. That is the right handling of a new dependency.

One SL9 gap inside a section that otherwise passes: **no gate measures a planet realm with many child
realms.** SL9 says a parent's child count is unbounded, and §4.1 puts every child realm's exterior
shell into the planet's rapier world as a body. Six hundred parked hulls on a planet is a lawful world.
S5 measures occupants and does not measure children. Add a row.

---

## 9. The verdict, said plainly

The report's shape is right: one collider surface, taken from the same generator the client draws with,
built only where a body stands, with gravity from the realm's own mass. Four things must change before
the owner rules on it.

1. Stop calling the FINE-lattice vertex rule an owner ruling. It is a proposal. Give it a door.
2. Withdraw decision 7 as written, or add the spinning-frame terms to the integrator. A realm that
   reaches to its SOI cannot spin silently.
3. Redo the fast-flight cost with the tube's real cross-section, and answer the airless moon.
4. Recompute the collider sizes for f64, and rewrite gate S2 so it compares the generator's canonical
   output rather than two hosts' private buffers.

Then add the tree, the thread, the turret and the mined support.
