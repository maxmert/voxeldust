# Galaxy cell-lattice design — settled 2026-08-05 (8-agent workflow, 7 blocking findings resolved)

> Owner rulings this design serves are recorded in `docs/design/DEFERRED.md` entry **D-SCALE-1**.

> This file is the WORKING DESIGN behind those rulings — the arithmetic, the topology and the build steps.


## Owner-facing summary
WHAT WAS WRONG, AND WHAT FIXED IT

The whole "cells nearly cannot exist" tension came from one mistake: assuming that how precisely a body can move depends on how far it is from the origin of the space it is in. That is only true because of how a position is currently stored — as three plain decimal numbers of metres. Store it that way at star-system distances and the numbers snap to a grid about two millimetres wide. A walking step is twenty millimetres, so walking picks up a permanent two-percent speed error, and anything slower than about five centimetres per second stops moving altogether while still reporting to the client that it is moving. That would silently break the ruling that all mechanics — manoeuvring, repositioning, space-walking — must work out in deep space.

The cure is already half-built in the codebase and was simply never switched on: store a position as a whole number of millimetre-sized steps plus a leftover under a millimetre, and roll the leftover into the whole number every tick. Then every tick only ever adds a two-centimetre step to a sub-millimetre leftover. The error per tick is around a billionth of a billionth of a metre, and it is the same everywhere in the universe. Precision stops depending on where you are.

That single change dissolves the tension. The size of a cell is no longer decided by arithmetic — it is decided by what has to fit inside one.

THE SETTLED NUMBERS

A cell is a cube about 58.8 astronomical units from centre to face — 117.6 across. A true-size Sun-like system, out to the Kuiper cliff at 50 units, fits inside one with about five units to spare, at true internal distances with nothing compressed. Anything bigger than that — a wide binary, for instance — is not capped: the cells form a ladder where each rung is twice the last, and oversized content simply hangs from a higher rung. There is no permanent limit on how large a single structure can be.

Systems are sprinkled one per six-by-six-by-six block of cells, so most cells are empty. Neighbouring systems sit about 706 astronomical units apart — about four hundred times closer than reality, which matches the target. From inside one system the nearest star is a dot two and a half arc-seconds wide, far too small to resolve, but as bright as a full Moon. That is a consequence of shortening the gaps this much and you should look at it before brightness is fixed; brightness stays a live knob as you ruled. In the rare closest case two systems can be 110 units apart, where the neighbouring sun is roughly thirty times full-Moon brightness. Crossing one gap swings a star three gaps away through about eighteen degrees, so the sky has real depth. Crossing one gap without warp at a hundred metres per second takes thirty-three thousand years, so repositioning and re-engaging warp is the only way to travel, exactly as ordered.

The whole chart at the recommended size holds about six hundred thousand systems and is roughly one light-year across. That small diameter is the direct price of shortening the gaps four hundredfold; it is worth seeing the number before it is fixed.

WHAT ELSE HAD TO CHANGE

Three things in the review were right and are now built into the plan.

First, warp speed cannot be derived from a single tick. Handing a traveller from one cell to the next is a fifteen-message conversation with two disk syncs, so the real speed limits are several times lower than first claimed. The chosen cruise — one gap per minute — still sits comfortably inside them, but a ship must now slow down automatically if the space ahead has not finished starting up, rather than flying on into nothing.

Second, deciding which cells to keep running by how big they look on screen does not work for a stack of nested cubes, because a child is always exactly half its parent and so is always "close enough" — every cell everywhere would stay running. Cells are now kept alive by travel time ahead of you, not by apparent size. The same change also guarantees an authority boundary is never drawn: a realm now separately carries "the thing you can see" and "the volume I am in charge of", and cells and system boundaries have the first of these empty. One rule, no special cases, and the boundary is invisible because there is nothing there to draw.

Third, when you cross a face, both cells briefly claim you. The old rule broke that tie by comparing internal names, which meant the crossing happened at a slightly different place depending on which direction you were going. It now breaks the tie by which cell you are actually deeper inside, so a crossing happens at the same place in both directions.

Also fixed: a naming scheme that would have made a cell and its own ancestors share an identity; a hand-off that showed the ship jumping backwards by most of an astronomical unit at warp; and a background chatter rate that would have sent about a million and a half messages a second at a thousand players.

THE ONE-LINE ANSWER ON COORDINATES

The millimetre-grained position type stays and finally becomes real — it is the fix. The light-year-grained one goes: a light-year does not even fit in the counter it was specified with, and converting between the two loses precision at exactly the seam it existed to protect. Its slot on the wire is reserved rather than deleted so nothing else has to change.

## The settled sizing (with the arithmetic)
SETTLED SIZING — THE TENSION IS DISSOLVED BY REPRESENTATION, NOT BY SIZE.

A. WHY THE OLD RULE WAS WRONG (blocking finding upheld)
f64 spacing in binade [2^e, 2^(e+1)) is 2^(e-52), constant across the binade.
 - Brief's "walk needs magnitude under 9e12 m": at 9e12 the spacing is 2^-9 = 1.953125 mm, NOT ~1 cm. The 1 cm point is ~4.5e13 m. The stated bound was ~5x too strict, and as stated it CROSSED the "cell must hold a system" bound, which is why nothing satisfied both.
 - But the real defect is worse and is not about smoothness. The shipped integrator adds the step into a raw metre triple (pose.pos = map_offset(|o| o + step)). At |p| = 2^43 every result snaps to a 1.953125 mm grid:
     v = 1.00 m/s, 50 ticks -> 0.976562 m: a PERMANENT -2.34% speed bias, direction-dependent (a closed square does not close).
     v <= 4.8828125 cm/s (= half-ulp x 50 Hz) -> step < ulp/2 -> position NEVER changes, while vel = step/dt keeps reporting the commanded speed to the client.
   That falsifies ruling 5 (all mechanics in deep space) and the draft's own corner-dwell acceptance test.

B. THE CURE (reverses the draft's "declare the pose cell dormant")
Position becomes integer lattice cell (edge 2^-10 m = 0.9765625 mm) + sub-cell remainder, with the remainder carried into the integer EVERY tick. The machinery exists and has zero production callers: LatticePos::normalize is exact at a power-of-two edge (a pure exponent shift), and the integrator's own comment already calls the per-tick normalize "a PURE ADDITION here ... NOT a clobber-and-replace".
Per-tick rounding then = ulp(~20 mm) ~ 3.5e-18 m; after 1e9 ticks ~ 3.5e-9 m. Anywhere in the universe.
=> Motion precision is a property of the STORAGE, not of the coordinate magnitude. Cell size is therefore chosen by CONTENT.

C. CHOSEN CELL
 leaf half-extent h = 2^43 m = 8,796,093,022,208 m = 58.7983 AU
 pitch P = 2h = 2^44 m = 1.75921860e13 m = 117.5965 AU = 0.0018595 ly
 Powers of two are load-bearing: "which cell is this point in" is an exact divide (exponent-only), and octant offsets (+/-h) are exactly representable, so composing an origin down any depth is exact.
 Why this rung: it is the smallest that holds a TRUE-SCALE Sun-like system. With seeded jitter <= h/16 = 3.674 AU, guaranteed clear radius = h - h/16 = 8.246337e12 m = 55.123 AU. Kuiper cliff 50 AU = 7.479894e12 m fits with 5.12 AU spare; Pluto aphelion 49.3 AU fits; Neptune-scale 30 AU fits with 25 AU spare.
 NOTE: the fence tests GEOMETRY only; the containment band is a membership device and is NOT subtracted from the fit. That is what dissolves the undefined-band-fraction objection: the 50 AU fit does not depend on any band fraction.
 NO structure ceiling: content wider than 55 AU nests one or more rungs UP the same ladder (each rung doubles). A 5,000 AU wide binary is a child of a cell 7 rungs up (half-extent 7,526 AU). The 51.4 AU permanent ceiling the review flagged does not exist under this rule.

D. PRECISION INSIDE A CELL (for DISTANCES, not motion)
 axis max 8.796093e12 and corner h*sqrt(3) = 1.5235280e13 are in the SAME binade -> spacing 1.953125 mm at both.
 (The draft's "last rung where corner and axis share a quantum" is FALSE — that holds at every power of two. Verified k = 38..49. 2^43 is chosen for the 50 AU fit, nothing else.)
 Distance/containment answers good to ~2 mm at leaf scale; band floor set at 1024 x local quantum = 2.0 m so hysteresis is never rounding noise (this is the derived f_floor the review demanded: it is a multiple of the quantum, not a fraction of extent).
 Millimetre lattice capacity: corner = 1.5601e16 cells vs the +/-4.6117e18 domain clamp. 0.34% of range. Fits with 3 orders to spare.

E. HYSTERESIS AT A SHARED FACE (with the geometric tiebreak of step 7)
 Cell A centred 0, cell B centred 2h. sd_A = x - h, sd_B = h - x.
 Acquire B at x >= h + inset; release A at x > h + outset. For inset < outset both are members on [h+inset, h+outset]; the most-negative-distance tiebreak picks B, so the commit lands at exactly x = h + inset. Returning, the commit lands at x = h - inset. Symmetric, direction-independent, hysteresis width 2*inset.
 outset >= inset is required ONLY to prevent a no-man's-land where neither cell claims you (which would drop you to the parent). Set outset = 2*inset.
 inset = max(2^10 * local_quantum, K * v_rel * T_handoff), K = 2. At leaf and cruise: 2 * 1.759219e12 * 0.3 = 1.0555e12 m = 7.06 AU = 12% of h. Harmless: cells are empty and a deeper container always wins.
 System-SOI band is sized from APPROACH speed, not cruise (a ship must drop out of warp to enter — see the governor in step 11). At 0.01c approach: 1.8e6 m, i.e. 1.2e-5 AU — invisible against the 5.12 AU fit margin.

F. STAR FIELD — STRATIFIED (independent-chance occupancy REJECTED)
 One system per 6x6x6 block of leaf cells.
 mean spacing G = 6P = 1.0555312e14 m = 705.58 AU = 0.011157 ly
 compression vs the codebase's own 4.0e16 m nearest-star constant: 379x ; vs a 5 ly disc spacing: 448x
 Sun (1.392e9 m) at G: 2.720 arcsec (eye resolves ~60) -> a dot. Apparent magnitude -12.50 == FULL MOON as a point source. The owner must see this before brightness is fixed (ruling 4).
 closest possible pair (adjacent cells across a block face, both jittered inward): P - 2*(h/16) = 1.649267e13 m = 110.25 AU -> 17.41 arcsec, magnitude -16.53 (~30x full Moon). Rare, spectacular.
 REJECTED alternative (independent 1/216 per cell): mean nearest neighbour is 0.55396 * n^(-1/3) = 5.847e13 m, NOT 6P — the draft's published gap/compression/arcsec figures were 1.8x wrong under it — and 11.4% of systems would get a neighbour at the 110 AU minimum. Stratified makes the published numbers true and the blinding case rare.
 parallax: one gap of travel swings a star 3 gaps out through atan(1/3) = 18.43 deg.
 sub-warp: one gap at 100 m/s = 1.0555e12 s = 33,448 years (ruling 5 satisfied by construction).

G. CHART SIZE (config)
 D rungs below the root; leaves 8^D; systems 8^D/216; root half-extent 2^(43+D).
 D=8 : 1.6777e7 leaves, 77,672 systems, 0.4760 ly ACROSS
 D=9 : 1.3422e8 leaves, 621,378 systems, 0.9521 ly ACROSS   <- recommended
 D=10: 1.0737e9 leaves, 4,971,027 systems, 1.9041 ly ACROSS
 (Diameters. The draft quoted half-extents as diameters, understating by 2x.)

H. SPEED CEILINGS — RE-DERIVED FROM THE HAND-OFF, NOT FROM ONE TICK
 T_handoff floor = 15 tick-hops x 20 ms + 2 durable checkpoints + 2 network RTT = 0.30 s BUDGET (must be MEASURED before the cruise is fixed).
 one crossing per K hand-offs, K = 3:        v <= P/(K*T) = 1.955e13 m/s
 destination warm with 1 cell of lead, T_boot = 2.0 s: v <= P/(T_boot + T) = 7.649e12 m/s = 25,514 c   <- binding
 chosen cruise (one gap in 60 s) = 1.759219e12 m/s = 5,868 c -> 4.35x under the binding ceiling.
 DELETED as wrong: "anti-tunnel v <= h*f = 4.398e14" (that is a cell-skip bound, not anti-tunnelling, and it assumes a one-tick hand-off) and "band <= h/8 -> 91,687 c".

I. SHARD SCOPE (64-wide membership bitset)
 leaf:         1 self + (D+1) ancestors + 26 lattice siblings + 1 system child = 38 at D=9
 intermediate: 1 + (D+1) + 26 + 8 children = 45 at D=9
 depth ceiling with intermediates running: 64 - 1 - 26 - 8 = 29 ancestors -> D <= 28 (NOT ~35).
 This holds ONLY because a lattice sibling is DERIVED from my own identity and needs no parent entry in the set (step 7). Listing siblings as ordinary regions drags in their ancestor closure: 42 minimum, 91 maximum at D=8 — over the fence. That objection is upheld and is resolved by derivation, not by counting.

J. THE GENERAL LAW (why precision never binds again)
 A level-k cell has half-extent 2^(43+k). Under the lattice representation the integration error is ~1e-18 m at EVERY level, so there is no ladder-dependent precision bound at all. What varies with level is only the resolution of DISTANCE answers (2^(k-9) mm), and the band floor scales with it (1024 x quantum), so hysteresis stays meaningful at every rung by construction.

## Verdict on the coordinate types
THE LIGHT-YEAR-GRAINED COORDINATE TIER GOES. THE MILLIMETRE-GRAINED ONE STAYS AND BECOMES LIVE — it is the fix, not a dormant option. This REVERSES the draft's step 9 in both halves.

Goes: the light-year tier and the single unnamed galaxy frame that is its only user. Reasons, both fatal and both already true in source: one light-year is 9.688e18 millimetre cells, past the signed-64-bit range, so the two tiers cannot carry between each other by integers at all; and the cross-tier conversion folds through floating-point metres, so it is inexact at exactly the seam it exists to protect. It also has no producer anywhere and the fixture that was supposed to activate it does not exist. Its wire discriminant is RESERVED, not deleted — deleting it would renumber two later arms of a frozen append-only enum, which is a protocol break for zero gain.

Stays and becomes mandatory: the millimetre lattice (integer cell + sub-cell remainder). It is the answer to the blocking finding — with it, integration error is ~1e-18 m per tick anywhere in the universe; without it, motion below 4.9 cm/s does not happen at all at cell scale. It must be produced, normalized every tick, and honoured by every server-side distance seam, not merely carried on the wire.

One honest limit, stated so nobody banks on more than it gives: the millimetre lattice makes STORAGE and INTEGRATION exact, and makes SHORT relative displacements exact. It does not make a 1e13 m DISTANCE more precise than 2 mm, because no floating-point metre triple can hold such a number better — the integer cell difference exceeds exact conversion beyond 8.796e12 m. That is fine and is not a defect: an error that scales with distance is invisible, and nothing needs sub-millimetre knowledge of a 58 AU separation. The rule to hold to is that MOTION is integrated in lattice units and DISTANCE is answered relative to the nearest meaningful origin (the viewer, or the containing realm), never as a universe-wide absolute.

Separately, the integer that used to be proposed for the pose now also lives in the realm's NAME: a cell's packed level-and-index word is exact integer arithmetic, and neighbour, parent and child are integer operations on it. That is why crossing a face is exact forever (the coordinate shifts by exactly one pitch, a power of two, in the same binade) with no drift no matter how many cells you cross.

## Build steps


### 1. Switch on the millimetre lattice as the live position representation
**Change:** Integrate into (integer cell + sub-cell remainder) and carry the remainder into the cell every tick: keep the existing add, then normalize at the fine tier, driven by the existing quantization decision point rather than a new flag. Make every SERVER-side seam cell-aware instead of reducing to metres first: the containment signed-distance subtract, the swept-segment previous position, the AoI/demand observer and child positions, the ghost anchor, the login spawn resolve, and the cross-frame re-expression that currently forces the destination cell to zero. Keep the client seams that are already cell-correct.

**Acceptance:** A commanded 1 cm/s at a cell corner produces 1 cm/s within 0.1% over 100k ticks (today: exactly zero motion). A closed square walked at 1 m/s at the corner returns to its start within 1 mm. Existing walk/visual fixtures byte-identical (every cell still zero there). Property test: integrate N random steps, assert cell*edge+offset equals the exact sum to 1e-9 m.


### 2. Retire the light-year tier; append a Copy cell frame
**Change:** Stop using the light-year tier and the single unnamed galaxy frame; RESERVE their wire discriminants rather than deleting them (the enum is frozen append-only and renumbering would break two later arms). Append a fixed-size, Copy frame arm carrying (chart root seed, packed level+index word). Do not put a heap lineage inside the frame: it is embedded in the per-entity per-tick pose lane and must stay Copy/Ord/Hash.

**Acceptance:** Wire round-trip tests prove old bytes still decode and the reserved discriminant is never emitted. The pose struct remains Copy and its encoded size is unchanged for existing arms. No production path constructs a light-year-tier position.


### 3. Cell as an ordinary realm kind with a level-tagged, losslessly invertible identity
**Change:** Append Cell to the kind tag registry, the profile-kind map and the capability profile table (all lookup tables, additive). Do NOT ride the star-system stand-in: the render-origin pin currently keys off 'deepest star-system ancestor' and a cell would steal it. Pack a cell index as 19 signed bits per axis plus a 5-bit level (62 of 64 bits, 2 spare): the level is REQUIRED because arithmetic-shift-right has fixed points at 0 and -1, so without it a cell at the all-zero index and every one of its ancestors collapse to one identity. Ancestors are recovered by shifting, so the packed word IS the full lineage, not a lossy short id. Fold the kind tag into the per-realm content stream so a dense cell index cannot collide with an avalanched seed of another kind. One Cell capability profile carrying the union (hull hosting plus signal relay) — capability must not vary by level.

**Acceptance:** Exhaustive test: for every level 0..19 and a spread of indices, pack/unpack round-trips and no two (level,index) pairs collide. A leaf plus its full ancestor chain plus 26 siblings all have distinct identities (today the all-zero chain aliases). Boot succeeds on a lattice forest that previously tripped the duplicate-realm guard; the client no longer rejects the streamed scene as duplicated.


### 4. Make every placement parent-relative and delete the universe-wide absolute
**Change:** Static region centres become offsets in the PARENT's frame (today a satellite sits at an absolute that only works in one flat frame). Retire the seed-folded absolute chain and the neighbourhood frame placement that positions every frame against the ambient root. Replace the render pin with 'the realm you are authoritatively in' — no kind test — and have the SERVER compose each drawn realm's position relative to that pin by walking up to the common ancestor and back down, each step an exact integer index times a power-of-two pitch, collapsing to metres only at the end.

**Acceptance:** No function returns a universe-root absolute. A round-trip test composes a leaf-to-leaf relative position across the chart and matches the exact rational answer to 1 mm. Client is pure passthrough: it applies no composition of its own. Existing round-trip ride-gap test still passes.


### 5. Make the nesting fence real and tiling-capable
**Change:** The geometric child-fits-parent test currently returns 'not comparable' whenever child and parent frames differ, which is EVERY pair in every shipped forest — it has never once run in production. Compare the child's placement expressed in the parent's frame (which is what a placement is after step 4). For box-in-box test per axis (|offset|+half <= parent half) instead of the sphere-conservative corner-vs-face pair: an exact octant subdivision scores reach 1.732h against a limit of h and fails by construction today. Take the band inset OUT of the fence — the fence is geometry, the band is membership. Fix the two stale statements that still claim this check is unimplemented.

**Acceptance:** A test proves the fence now FIRES on a cross-frame forest (it cannot today). An exact octree subdivision passes; a child poking one millimetre outside its parent fails boot loudly. A true-scale 50 AU system inside a 58.8 AU cell with maximum jitter passes; 56 AU fails.


### 6. Computed child index: the lazy generator
**Change:** Add ONE data-driven child-index rule alongside the boundary enum in the core crate: Enumerated(list) or Lattice{half_extent, level}. Lattice answers 'which child contains this local point' by an exact power-of-two divide and 'what is my neighbour across face f' by an integer increment with carry. Rewrite the deepest-containing-child resolver on top of it and stop materialising the whole forest per call. Settle the two-partition-functions conflict explicitly: the exact divide NOMINATES (it is authoritative on cold start, when there is no prior membership, and after a spun-down cell comes back); the hysteresis band GATES transitions only. Without this the union of acquirable regions is the cells shrunk by the inset, leaving an unclaimable slab around every face.

**Acceptance:** Containment resolves in constant time with no sibling enumerated and no forest materialised (assert allocation-free). A cold-start occupant standing exactly on a face is assigned a leaf, not the parent. A spun-down/spun-up cell under a stationary occupant restores the same leaf. Fuzz: 1e6 random points, the divide-nominated cell always equals the brute-force scan answer.


### 7. Geometric tie-break at equal depth; derived siblings in scope
**Change:** The container fold breaks equal-depth ties by realm id ascending — with siblings in scope for the first time, that makes the commit point of every crossing depend on which name sorts lower, so the same physical face commits at inset in one direction and outset in the other. Break equal-depth ties by MOST-NEGATIVE signed distance, with identity only as a final determinism tiebreak. Extend the scope rule with 'plus the 26 lattice siblings when my parent's child rule is a lattice', and make derived siblings exempt from parent-resolution (their parent is computable) — otherwise their ancestor closure is 42 to 91 regions against a 64 fence.

**Acceptance:** Pinned scope-count test: 38 regions at a leaf and 45 at an intermediate at depth 9; boot fails loudly above 64; depth 28 passes and 29 fails. Crossing a face in +x and then -x commits at the same coordinate to within a millimetre. A corner where 8 cells meet asserts WHICH cell wins, deterministically, from every approach direction.


### 8. Derived containment band, per realm, with live velocity
**Change:** Replace the flat 1 m / 2 m scale-blind band (built with zero velocity, so its safety widening is inert everywhere) with inset = max(1024 x local coordinate quantum, K x v_rel x T_handoff), outset = 2 x inset, K = 2 as today. Note the second term is the HAND-OFF time, not one tick — a tick-sized band is 50x too thin. Thread the live relative speed and the measured hand-off through, as the playground override already does for the demo path.

**Acceptance:** At leaf scale the floor is 2.0 m and at cruise the term is 1.06e12 m. Assert the band always exceeds one representable step at its realm's extent (today at galaxy extents the whole band is five orders below one step, i.e. membership decided by rounding). A traveller at 1.5x the design cruise is refused at boot rather than tunnelling.


### 9. Split 'what you can see' from 'what I am in charge of'
**Change:** Give a realm an optional VISUAL BODY (radius, and emissive luminosity for stars) distinct from its authority boundary. Both the render filter and the visibility term of the interest rule read the visual body; cells and system authority boundaries have none. Replace the fixed 200 m renderable-extent cut with the angular rule applied to the visual body. The interest radius becomes max(visibility term, travel-time lead). This single datum resolves two blocking findings at once: the authority boundary is never drawn without any kind test in the renderer (ruling 3), and the angular rule stops keeping every cell alive.

**Acceptance:** A cell and a system boundary never appear in the streamed scene, and the renderer contains no kind test that gates drawing. A star 706 AU away is drawn as a point of the configured brightness. Regression: existing visual-scale scenes render identically.


### 10. Travel-time demand, multi-level lookahead, throttled cadence
**Change:** The interest radius is a multiple of the CHILD's own size, and an octree fixes the parent:child ratio at 2, so with the current 14.3x factor every child of every live cell is permanently in range — teardown never fires and roughly 8 cells per level per occupant stay live. Give cells a travel-time lead instead: lead >= v x (boot p99 + settle + hand-off) x safety, evaluated per traveller velocity. Extend the occupant-interest relay past the immediate parent by addressing ancestors through their COMPUTED identity, so intermediate cells stay addressing-only unless content is placed at that level. Throttle keep-alive and empty demand to a fraction of the demand time-to-live instead of one message per child per tick (today ~1500 messages per second per traveller, ~1.5e6 at a thousand). Shard the demand plane by lattice subtree instead of one address for the whole chart.

**Acceptance:** One traveller at cruise keeps live exactly its leaf, the leaf ahead, and its occupied ancestors — not 8 per level. Teardown fires behind. Message rate per traveller measured and under a stated budget. A thousand simulated travellers stay inside the stated per-orchestrator budget or the plane splits.


### 11. Continuous hand-off and a warp governor
**Change:** The destination adopts the pose sampled at flush, several hops before the hand-off completes, while the source keeps integrating — a backwards jump of 2 to 4 ticks, invisible at walking speed but 0.5 to 0.9 AU at cruise, every ten seconds. Ship (pose, velocity, flush tick) and have the destination adopt the closed-form extrapolation to the adopt tick, with the source freezing integration at flush so both agree exactly. Separately, an unresolvable destination currently just increments a counter while the traveller flies on at full speed in a frame it has left, and the destination frozen into the request goes stale. Make it backpressure: clamp speed to the lease frontier (you cannot warp faster than space is chartered ahead of you), and re-target after commit if the container changed in flight.

**Acceptance:** Crossing at cruise shows zero position discontinuity (assert < 1 mm, not < 1 AU). With the destination held cold, the traveller decelerates and holds at the face instead of overshooting; when the lease lands it resumes with no jump. A traveller forced past the ceiling never accumulates more than one cell of authority lag.


### 12. Content: stratified occupancy, jitter, true-scale systems
**Change:** Cell content closed-form from the seed: stratified one system per 6x6x6 block, cell chosen and centre jittered within h/16 by the same stream; system authority radius derived from its own outermost apoapsis plus that body's sphere of influence, capped at h minus jitter. Delete the uniform AU-to-render scale factor (even the 'realistic' preset inflates every orbit 2.5x) and the hardcoded system/galaxy radius literals. Content wider than a leaf nests at the shallowest rung that fits it. Make the dead galaxy configuration fields (system counts, type mix) live here or remove them.

**Acceptance:** A generated system's internal distances match real proportions to 1e-12 relative (no compression inside a system). Sampled over 1e6 blocks: mean spacing 705.6 AU, minimum 110.2 AU, no two systems overlap, no visible grid in a rendered star field. A 5,000 AU binary generates at the correct rung and passes the fence.


### 13. Gate: crossing matrix, precision proof, load proof
**Change:** Real-process scenarios on the shipped path: leaf-to-leaf across a face; across an edge and a corner (three carries at once); a carry that rolls several rungs up the ladder; cell into a system boundary and back out; a slow space-walk parked exactly on a cell corner for 100k ticks; a cold-start login in deep space; log out and back in at a corner. Load: the implied crossing rate (one gap per 60 s per traveller is a cell every ~10 s, so a thousand travellers is ~100 hand-offs per second, ~1500 saga messages and ~200 durable syncs per second) PLUS the demand plane, plus a scope/memory test at depth 9.

**Acceptance:** All scenarios green in the process-tier gate with the dev-control features on. The corner dwell asserts the position quantum stays at 1.953125 mm, the commanded speed is delivered, and no membership flap occurs. The load test sustains the stated rates with p99 hand-off inside the budget used to derive the speed ceilings — and if the measured hand-off exceeds the 0.3 s budget, the cruise speed is lowered from the measurement, not the other way round.


## In-game validation required
MUST BE SEEN WORKING IN THE ACTUAL GAME BEFORE ANY OF THIS IS CALLED DONE (the standing rule is that acceptance drives the exact shipped path, and nothing is committed as working until you confirm it in game).

1. THE SLOW SPACE-WALK, PARKED ON A CELL CORNER. Out in empty space, at the worst point of a cell, use the finest translation you have — around a centimetre a second — and confirm you actually move, that you stop where you aimed, and that walking a square brings you back to the same spot. This is the direct proof of the whole sizing decision. Today it produces exactly zero motion while the client is still told you are moving, so this is the single most important thing to look at.

2. CROSSING A FACE AT WARP. Fly across a cell boundary at cruise and confirm nothing pops, freezes, stutters or jumps backwards, and that the star field does not shift. The current hand-off would slide you back most of an astronomical unit every ten seconds.

3. CROSSING A CORNER. Aim at the point where eight cells meet and cross it. Three boundary changes land in the same tick; this is where an index-arithmetic bug hides and where the tie-break rule is exercised hardest.

4. DEEP SPACE TO PLANET SURFACE IN ONE CONTINUOUS FLIGHT. Start between systems, warp one gap, drop out, fly in past the outer planets at true distances, land. No loading, no toggle, no change of apparent scale as you cross the system boundary. The system boundary and the cell boundaries must be completely invisible while you do it.

5. WARP INTO COLD SPACE. Head somewhere nobody has been. Confirm the ship throttles itself back rather than sailing on into a region that has not started up yet, and that it resumes smoothly the moment the space ahead is ready.

6. THE SKY. Look at it and decide: the nearest star is a two-and-a-half arc-second dot as bright as a full Moon; the rare closest neighbour is thirty times that. Travel one gap and watch the near stars swing about eighteen degrees against the far ones. Confirm the star field does not read as a grid. Brightness is a live knob at this point, exactly as you ruled — this is the moment to set it.

7. TWO PLAYERS ACROSS A BOUNDARY. Stand either side of a cell face and confirm you see each other, that you collide, and that walking through each other's boundary hands you over cleanly with no rubber-band.

8. LOG OUT AND BACK IN, IN DEEP SPACE AND ON A CORNER. You must come back exactly where you left, in the same cell, with no drift and no snap.

9. A THOUSAND-TRAVELLER RUN. Not a feel test but a watch-the-numbers test: hand-offs per second, background chatter per second, how many shards are actually running per traveller, and whether the hand-off time stays inside the budget the warp speed was derived from. If the measured hand-off is slower than budgeted, the cruise speed comes down — the speed is derived from the measurement, never the reverse.

## UNCHECKED — gaps named honestly rather than implied covered
LENSES THAT RETURNED BROKEN: two of three. The sizing lens and the generic lens both returned BROKEN; the crossing-cost lens returned sound-with-changes. Every blocking finding from all three is resolved above (none declined): sizing's integrator-accumulation finding is UPHELD and drove the central reversal; crossing-cost's three (speed ceilings from one tick, interest saturation plus one-level lookahead, overshoot during boot) are UPHELD and folded into steps 8, 9, 10 and 11; generic's three (cross-level identity collision, sibling ancestor closure over the 64 fence, angular rule drawing cells) are UPHELD and folded into steps 3, 7 and 9. Two blocking framings were CORRECTED rather than accepted: the boot fence does not currently constrain anything (it is skipped on every cross-frame pair, i.e. every pair in every shipped forest), so cell sizing was never fighting an enforced constraint; and the "system must be at least 1e13 m" input is an illustrative literal, not a law.

WHAT NOBODY LOOKED AT — NAMED, NOT HIDDEN:

1. NOTHING WAS EXECUTED. This was a read-only settlement. No build, no test, no coverage run, no game session. Every number here is derivation plus source reading, not measurement.

2. THE HAND-OFF TIME IS A BUDGET, NOT A MEASUREMENT. The 0.3 s floor is 15 message hops at 20 ms plus two durable syncs plus two round trips, counted from the state machine, never timed. Both the warp cruise and the band thickness descend from it. It must be measured before the cruise speed is fixed.

3. BOOT LATENCY IS UNMEASURED AT THIS SCALE. The 2 s figure comes from a fixture value, and the shipped development setting has the forward-looking term at zero (reactive only). The one-cell-of-lead conclusion depends on both.

4. PERSISTENCE AND CRASH BEHAVIOUR WERE NOT EXAMINED BY ANY LENS. What a durable checkpoint of a lattice position looks like, whether a re-anchored position survives a reload unchanged (it should — re-anchoring is idempotent, but that was reasoned, not tested), and what happens if a cell shard dies mid-crossing.

5. THE 100% COVERAGE RULE WAS NOT COSTED. Making the lattice live turns a large body of currently test-only code into product code, and the generic-code monomorphization gotcha applies. Nobody estimated that work.

6. THE CLIENT AT TRUE SCALE IS UNEXAMINED. Depth buffering, star point rendering, exposure and dynamic range for a full-Moon-bright point source, and whether the scene stream stays sane when the drawn set spans 13 orders of magnitude. Only the server-side composition was settled.

7. THE ORCHESTRATOR AS A SINGLE POINT WAS NOT DESIGNED, ONLY FLAGGED. Splitting the demand plane by lattice subtree is asserted in step 10 with no mechanism behind it.

8. TRANSIENT ENTITIES, SIGNALS AND VOXEL CONTENT WERE OUT OF SCOPE. Every lens looked at occupant motion and realm containment. Nothing checked how a batched transient hand-off, the signal bus, or block/terrain generation behaves across a cell face.

9. TWO NUMBERS REMAIN OWNER CHOICES, NOT ENGINEERING ONES: chart depth (recommended 9, which is 621,378 systems in a chart 0.95 light-years across — note that diameter, it is the direct price of shortening the gaps 400-fold) and star brightness, which stays a live knob as ruled and whose real range is magnitude -12.5 typical to -16.5 in the rare close pair.