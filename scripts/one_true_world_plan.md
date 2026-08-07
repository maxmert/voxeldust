# ONE WORLD AT TRUE SCALE — plan + confirmed breakages

Produced 2026-08-06 by a 41-agent audit (19 breakages confirmed after adversarial refutation, 16 refuted).
Owner ruling that provoked it: remove the world config, ONE final world, TRUE astronomical scale, final
game machinery — not a POC. See memory feedback_final_backend_not_poc.

---

# ONE WORLD AT TRUE SCALE — THE PLAN

---

## 1. THE HONEST HEADLINE

**No. It is not one change.** Deleting the four world sizes is the small half — about eight lines of real product code. The world we would then be running does not start, cannot be seen, cannot be drawn, and cannot be travelled. Four things have to exist first, and one of them is cheaper than it looks.

**What silently breaks the moment the world is astronomical:**

- **Positions get rewritten without a word.** Every frame in the game today counts position in roughly-one-millimetre steps on a whole-number grid. That grid's total reach is 0.95 light-years — and once a position crosses between two machines a safety clamp halves it to 0.48. The nearest neighbouring star sits at 4.23 light-years. So a player who walks out of the second star system is silently moved 3.28 light-years back toward the origin on the next packet they send — including a packet that only turned their head. No error, no log, no counter. Beyond that point they can move inward but never outward again.
- **The world refuses to start.** The rule that decides when a body becomes visible carries a fixed 0.25-metre safety pad. Above a body about 315 trillion metres across, one representable step in the arithmetic is larger than that pad, so the pad vanishes, the visible-in and visible-out radii come out equal, the rule is rejected as invalid, and start-up aborts. The galaxy is 5×10²⁰ metres. One representable step there is 1,048,576 metres against a 0.25-metre pad.
- **You see nothing.** The visibility rule says a body appears when it looks 8 degrees wide. Earth seen from 1 AU looks 17.6 arcseconds wide — the threshold is 28,800 arcseconds, so it is **1,640 times too coarse**. Our own sun seen from the next star over is 0.007 arcseconds — **4 million times too coarse**. And there is no setting that fixes both ends: making a planet visible from its own star needs about 1.15 degrees, which still leaves the neighbouring star system 40 times short; reaching the neighbouring star needs about 103 arcseconds, at which point every planet in your own system is permanently switched on. **This is not tuning. The distance at which a thing must be running and the distance at which it must be seen differ by three to four orders of magnitude, and today one number does both jobs.**
- **The screen goes to scaffolding.** The client throws away any body wider than 200 metres — that is every body at true scale, including a 5-kilometre station. It also refuses to draw past 6 kilometres. And because it throws them all away, the placeholder debug floor and eight coloured pillars from the early prototype are *not* removed, so the player gets a plausible-looking empty room instead of an obvious failure.
- **Looking around stops working.** Everything drawn is positioned in single precision relative to the player's own star. That format holds one metre of accuracy only within 16,777 kilometres of its origin. At 1 AU the drawing grid is 16,384 metres: two players standing one metre apart are drawn at the same point, the ground under the camera snaps in kilometre steps, and the camera itself only moves once every 546 seconds of walking, then jumps 8 kilometres. Separately, the camera works out where it is aiming by subtracting two points one metre apart — that becomes visibly jerky about 10 kilometres from the origin, shakes by 476 pixels while standing on an Earth-sized planet, and past about 16,800 kilometres it stops responding to the mouse entirely and silently stares along a fixed direction.
- **Nothing is reachable.** At today's 15 metres per second, one AU takes 316 years, the outer edge of a star system 21,100 years, the nearest star 84.5 million years. Even at the speed of light the nearest star is 4.23 years.

**What it does *not* force:**

- **Ships are not a precondition.** A ship in this design is a place, not a vehicle with a speed; the player's travel speed is a single configured number that is not capped anywhere. A "fast craft" today is a bigger number.
- **Warp is not a precondition for *proving* the machinery.** The proportion between "visible from here" and "arrived" is a fixed 13.3 to 1 no matter how big the world or how fast the traveller. Raising the configured speed keeps every proof identical in shape: a true-scale approach to a planet takes 2,206 ticks at exactly today's fidelity, comfortably inside the 3,000-tick budget those tests already have. **Warp is a precondition for a human being able to play it**, not for the tests.
- **A second coarser grid may not be needed at all.** The grid's reach and its fineness are set by one number — the size of a whole-number cell. Today that cell is 0.98 millimetres, which is why the reach is only 0.95 light-years. A cell of about **4.29 million kilometres** gives a grid that reaches **22 times past the edge of the observable universe** and still resolves **0.95 micrometres anywhere inside it**. A cell of 4.4 trillion metres gives millimetre resolution across 23,000 universe radii. One grid can hold everything. That turns the hardest-looking item on the list into a single deliberate constant plus a loud refusal when something exceeds it.

---

## 2. THE ORDERED SLICES

Each one lands alone, and each one has a measurement that fails before it and passes after.

**A — Make the coordinate cliff loud.** *(new machinery, small)*
Today a position past the grid's reach is silently rewritten. Make it a refusal that names the offending value, and make start-up reject any world whose geometry exceeds the grid's reach.
*Proof:* feed the movement step a position 4.23 light-years out and assert a named refusal. Today it returns a position 3.28 light-years away and says nothing.

**B — Set the grid to hold the world.** *(one constant + the guard from A)*
Pick the cell size (owner decision D1) and widen the safety clamp to match.
*Proof:* store a position at 4.23 light-years, at 52,850 light-years, and at the edge of the observable universe; send it between two machines; run one movement tick; read it back. It must return within one grid step. In the same test, a 1-millimetre step must still move you, at every one of those magnitudes.

**C — Make the visibility hysteresis proportional.** *(pure fix, tiny)*
Replace the fixed 0.25-metre safety pad with a fraction of the visibility radius. A working proportional version already exists in one of the presets being deleted — 20% wider out than in.
*Proof:* build the rule at every body size from 1 metre to 10²⁷ metres and assert it constructs, with the out-radius exceeding the in-radius by a fixed fraction. Today it fails above 315 trillion metres.

**D — ONE world builder, generated lazily from the seed.** *(the big new machinery)*
Today there are two. The gateway builds the client's world from a hardcoded list of seven bodies; the shards build a different world with orbiting planets from the same seed. That is why the client is told a star system is 40 metres across while the shards fly planets out to 152 metres. The replacement must generate only the children of the node being examined, must accept a grid position rather than a bare metre triple, and must be able to emit a real star census — today's cannot emit more than 2 stars, 1 planet, 1 station and 1 area, and the star-count, mass-distribution and planet-formation settings it carries are read by nothing.
*Proof:* (a) same seed, gateway and shards report identical size and position for every body — today they differ by 3.8×; (b) resolving a login touches only the children of each node it descends through, provable by a counter; (c) the tree respects the 61-direct-children limit by construction at a stated subdivision depth.

**E — Travel speed and approach braking.** *(configuration + a small controller change)*
The travel speed is already unbounded and already feeds every distance-based rule. What is missing is a brake: the test pilot always commands full throttle. The throttle axis already exists on the wire and already accepts fractions.
*Proof:* an end-to-end run that flies from a star to a planet at true scale and stops at it, inside the existing tick budget. Without this, every end-to-end proof after D is dead.

**F — Delete the four worlds.** *(pure deletion)*
The environment switch, six of the seven world presets, and everything that exists only to shrink a real star system into a 150-metre window.
*Proof:* the world's size appears in exactly one place; the switch appears nowhere; the whole suite passes.

**G — Repair the start-up containment check.** *(new machinery, medium)*
The check that a body sits inside its declared parent has **never fired on real data** — every parent and child are described in different frames of reference, so it declines on all of them. Simply switching it on rejects today's world on a false alarm and still cannot judge an orbiting child.
*Proof:* a test that inflates a planet beyond its star system and asserts start-up refuses. It must fail today.

**H — Split "drawn" from "running".** *(new machinery, medium)*
One decision currently serves both. A distant body needs a third state: a server-authored point of light with a position and a brightness, and no running process behind it.
*Proof:* standing at a star, the client is told about every planet in the system and about the neighbouring systems, while the number of live processes stays bounded and named.

**I — Visibility on brightness and drawn size, in arcseconds.** *(new machinery)*
Key what you can see on the body's real radius, not on its sphere of influence — those differ by 141 times for an Earth. Key what must be running on reachability.
*Proof:* from Earth's orbit, the count and apparent sizes of the visible bodies match an astronomy table within a stated tolerance.

**J — Move the drawing origin to the player.** *(server change, medium)*
Today the origin is the player's star. It must be a point that follows the player. Also stop deriving the camera's aim by subtracting two nearby points — the exact aim direction is already known.
*Proof:* two players one metre apart, one AU from their star, are drawn one metre apart (today: the same point). The camera's drawn aim matches the commanded aim to under a tenth of a pixel at 10¹² metres (today: it stares along a fixed axis).

**K — Drawable size and view distance become per-body values from the server.** *(deletion of two constants + plumbing)*
*Proof:* at true scale a screenshot contains a real body. Today the same screenshot contains a debug floor and eight pillars.

**L — A real sky.** *(new machinery)*
The current starfield is 2,800 small spheres invented by the client and glued to the camera; it goes completely black about 87,000 kilometres from the origin. Replace it with the real far catalogue, drawn from viewing direction only.
*Proof:* the sky is unchanged in appearance at 1 AU, at 65 AU, and at 4 light-years from the origin, and the stars are the ones the world builder made.

**M — Warp.** *(new machinery, the largest single feature)*
Warp is a big per-tick displacement on the existing machinery, not a separate mode. It needs one thing nobody has built: an **arrival phase**. The "warm the destination up before you get there" mechanism gives a fixed 13.3 body-radii of lead — at 100 times light speed you cross a planet's entire warm-up radius in under two ticks, so there is nothing left to warm up with. Warp needs a drop-out corridor that decelerates before approach.
*Proof:* a player flies between two star systems, the destination is fully live before arrival, and the count of live processes behind them falls back to the baseline.

---

## 3. WHAT THE OWNER MUST DECIDE FIRST

**D1 — How fine does a position need to be?** This one number picks everything else. Say the smallest thing that must land exactly where it was put: a millimetre, a centimetre, a voxel edge. A cell of 4.29 million kilometres buys 0.95 micrometres everywhere and reaches 22 universe radii. A cell 1,000 times bigger buys 1 millimetre and reaches 23,000. Either way it is one grid, not a ladder of two, and that removes an entire class of conversion bugs. **The decision to make is: one grid at a bigger cell, or two rungs.** My recommendation is one.

**D2 — How far apart are neighbouring star systems, really?** The earlier ruling was: systems keep their true size, only the gaps between them compress, to roughly 100 trillion metres — about 400 times closer than reality's 4.23 light-years. Confirm or overturn. It changes every travel time by 400 times and it decides whether the grid ever needs to reach past a light-year at all.

**D3 — How long should a journey take?** Give two numbers: minutes to cross a star system, and minutes to reach the next star. One AU in ten minutes is 0.83 times light speed. The next star at the compressed gap is 56 minutes at 100× light speed, 5.6 minutes at 1000×. These numbers size the warm-up lead, the tick rate, and whether warp needs the drop-out corridor.

**D4 — What is a distant body when it isn't running?** Confirm the three states: running, a server-authored point of light, absent. Everything in the visibility work depends on the middle one existing.

**D5 — The shape of the galaxy's tree.** No place may have more than about 61 direct children, and scanning more than that every tick is unaffordable regardless of the limit. Seven nested levels of unnamed sectors hold 2.2 trillion star systems with that limit untouched. Confirm nested sectors are acceptable — they are invisible in play, they are just how the world is addressed.

**D6 — Saved players.** Existing saved positions are metre coordinates in a 180-metre world and are meaningless at true scale. Wipe or migrate.

---

## 4. THE BIGGEST RISK, AND THE CHEAPEST WAY TO RETIRE IT

**The risk is not that the world cannot be made big. It is that it can be made big and turn out to be nothing to look at.** At true proportions space is overwhelmingly empty: fly for an hour and the sky does not change. The current 180-metre demo hides this by inflating every planet's zone of influence to 35% of the gap to the next orbit — 24.5 times what physics gives an Earth. Take that inflation away and even a physically perfect world puts every terrestrial planet 7 to 18 times outside its own star's visibility radius. That ratio does not depend on the size of the world at all; it depends only on mass. **Slices H, I, L and M are all bets on an answer to a question nobody has looked at yet: what does a real-proportion sky actually feel like.**

**The cheapest experiment — one afternoon, client-side, throwaway.**
Before touching the coordinate grid or the world builder, keep today's running world exactly as it is and hand-feed the client a fixed list of about 40 bodies at real distances, real radii and a brightness each, drawn from viewing direction only. Then fly the camera between two of them.

That answers, in one sitting:
- Does a real-proportion sky read as a place, or as a black screen with dots?
- What angular threshold makes the number of visible bodies match a real night sky?
- Does an approach read as motion at all, or does the destination just pop in over the last fraction of a percent of the trip? (Measured today, with the star backdrop glued to the camera, 99.64% of an interstellar trip has literally zero visual change.)
- How far can the camera get from the drawing origin before the picture falls apart? Predicted: visible jerk at ~10 kilometres, mouse-look dead at ~16,800 kilometres, sky gone at ~87,000 kilometres.

If the answer is "a true-proportion sky is boring", that is a design ruling you want **before** rebuilding the coordinate base, not after.

**Second experiment, ten minutes, run it the same day:** take the existing true-scale geometry and simply try to build the world with the visibility rule switched on. Prediction: it aborts at start-up on the galaxy, because the 0.25-metre safety pad is smaller than one representable step at that size. Confirming that costs one test and settles slice C immediately.

---

## 5. PURE DELETION VERSUS REAL REPLACEMENT

### Pure deletion — cheap

- The environment switch that picks a world, and the three-way choice behind it. **One production read.**
- Six of the seven world presets. Two of them already have no callers at all.
- Everything that exists only to shrink a real star system into a 150-metre window: the astronomical-unit-to-metres compressor, the invented central mass, the derived influence radii, the target orbital period. About five helpers and eight constants, one purpose between them.
- The two fixture writers and the launcher scripts that emit two different world files.
- The tests that assert the compressed world is correctly compressed — they die with the feature they test.
- A second, entirely unread copy of the drawable-size limit that the world config already carries.

**Real size:** of 88 references to a world preset, **77 are inside tests that this work rewrites anyway**, three are inside the presets being deleted, and only **eight are production lines across four files**.

### Must be replaced — the metre world was carrying real weight

- **Two world builders.** One hardcoded seven-body list for the client, one seeded generator with orbiting planets for the shards. Same seed, two worlds — 40 metres against 152. One lazy seeded builder replaces both, and it must be able to emit a real census; today's cannot exceed 2 stars, 1 planet, 1 station, 1 area.
- **The login "where am I" descent.** It walks a fully materialised list and takes a bare metre triple. It must generate children on demand and take a grid position.
- **The start-up containment check.** It has never once judged real data. It needs both positions expressed in the same frame, real shape-inside-shape containment rather than a bounding-sphere estimate, and for an orbiting child, its farthest orbital point rather than a static centre.
- **The visibility rule.** One number doing two jobs, working today only because the demo inflates influence radii 24.5-fold.
- **The visibility hysteresis.** A fixed 0.25 metres must become a fraction.
- **Everything the client uses to decide what to draw:** the 200-metre size limit, the 6-kilometre view distance, the drawing origin at the star, the camera's aim arithmetic, and the invented starfield. Five metre-world constants, none of them told by the server.
- **The test fixtures.** About 40 sites across 14 files hardcode the demo's coordinates — a planet at 20 metres, a station at −25, a second system at 130 — and roughly 13 end-to-end binaries depend on them.

### Two warnings for whoever starts

**Do not promote the existing "true-scale" preset.** It looks like the starting point and it is not. Its astronomical-unit conversion is 2.5 times the real value, so its innermost planet lands at 1.0 AU instead of 0.4 and its outermost at 8.35 instead of 3.34. Its area sits **one full astronomical unit outside** the planet that claims to contain it. Its station sits inside the sun's surface. It is illustrative junk. **Write the builder; do not inherit the constants.**

**Do not trust the picture gates.** Two of the three screenshot proofs fly hand-authored 12-to-60-metre boxes and never touch the world builder at all — they would stay green with a client that cannot draw a single real body. Exactly one gate renders the real streamed world and asserts a planet is drawn and moving; **that one is the canary, and it will fail loud the moment the world goes true-scale.** Keep it in front.

---

# CONFIRMED BREAKAGES (survived an adversarial refutation attempt)

## [blocker] `LatticePos::normalize` saturates SILENTLY at 2^53 m = 0.952 ly, and the integrator calls it every tick — so at true scale a player's first movement input after crossing into interstellar space teleports them.

**At true scale:** Universe and Galaxy stand in as `RealmId::System(0)`/`System(1)` (worldgen.rs:91-92) ⇒ `FrameRef::SystemSpace` ⇒ Tier::Fine. A player in the between-systems gap sits at ~1e16-4e16 m in the Galaxy frame. `transfer_frame_resolved` writes that as cell-0 full metres (frame.rs:161); the very next movement tick normalizes it and pins them at exactly 0.952 ly from the galaxy origin. They keep walking; the position never changes again. Same event for any pose beyond 0.952 ly in any Fine frame.

**Evidence:** crates/core/src/pose.rs:301-320 — `carry = (offset/edge).floor()`, then per-axis `self.cell.x.saturating_add(carry.x as i64)`; Rust's f64→i64 `as` cast saturates (the code says so at :310, and its own test `normalize_saturates_instead_of_panicking` at :970-979 pins the behaviour). The integrator calls it unconditionally: crates/sim/src/stub.rs:2551-2555 `pos.map_offset(|o| o + step).normalize(dot.pose.frame.tier())`. Arithmetic (simulated in IEEE-754 binary64): at Tier::Fine, edge 2^-10, offset 9.1e15 m → carry 9.3184e18 > i64::MAX 9.2234e18 → cell = i64::MAX, offset = 0, represented = 9.0072e15 m, LOSS 9.28e13 m (1.02%). offset 4.0e16 m (= CANONICAL_SYSTEM_B_OFFSET_M, worldgen.rs:986, 4.23 ly) → represented 9.0072e15 m, LOSS 3.099e16 m (77.5%). offset 5.0e20 m (= CANONICAL_GALAXY_R_M, worldgen.rs:982) → represented 9.0072e15 m, LOSS 4.9999e20 m (99.998%). No Result, no log, no counter anywhere on the path.

**Fix shape:** Two independent things: (1) make the cliff LOUD — normalize must return a typed out-of-domain error or the boot fence must reject any forest whose geometry exceeds `2^53 * edge` for its frame's tier; (2) give the frames above a star system a real COARSE tier so they never reach the cliff (see the next item). Do not raise `FINE_CELL_EDGE_M` as the only fix — that buys range but leaves the silent-saturation class intact.

**Refutation attempt failed because:** NOT REFUTED. I re-derived the arithmetic independently and then EXECUTED a byte-exact replication of `normalize` (rustc -O, same expressions, `saturating_add` + Rust's saturating f64→i64 `as` cast). Every number the prior agent gave reproduces exactly.

MEASURED (executed, not argued):
- Cliff is exactly 2^53 m: `(i64::MAX as f64) * 2^-10 == 9.007199254740992e15 == 2^53`. In ly: 9007199254740992 / 9460730472580800 = 0.9520617124487123. Their "2^53 m = 0.952 ly" is exact.
- offset 9.1e15 → cell i64::MAX, offset 0.0, represented 9.007199254740992e15, LOSS 9.2800745259008e13 (1.0198%). Matches.
- offset 4.0e16 (= CANONICAL_SYSTEM_B_OFFSET_M) → LOSS 3.099280074525901e16 (77.4820%). Matches.
- offset 5.0e20 (= CANONICAL_GALAXY_R_M) → LOSS 4.9999099280074526e20 (99.9982%). Matches.
- Below the cliff the fold is EXACT: normalize(0, 1.0e13) → cell 10240000000000000, offset 0.0, zero loss. So this is a razor cliff, not a gradient — exact everywhere up to 2^53 m, then total collapse.

CODE CHAIN — verified by reading, all four links hold:
1. worldgen.rs:85-95 — `UNIVERSE = RealmId::System(0)`, `GALAXY = RealmId::System(1)`.
2. pose.rs:133-146 `frame_for_realm` → `FrameRef::SystemSpace`; pose.rs:112-121 `tier()` → `Tier::Fine`. No production `GalaxySpace` producer exists.
3. frame.rs:161-165 `transfer_frame_resolved` writes `pos: LatticePos::local(new_pos)` — cell-0, full metres. `generate_walk_forest` (worldgen.rs:657-663) places SYSTEM_B at `at_x(sa.system_b_offset_m)` under GALAXY, so an occupant leaving System B's SOI is at 4.0e16 m in the Galaxy frame IMMEDIATELY.
4. stub.rs:2551-2555 — `dot.pose.pos = dot.pose.pos.map_offset(|o| o + step).normalize(dot.pose.frame.tier())`, unconditional, no Result, no counter.

Reachability is the part worth stressing: this does NOT require 19 Myr of walking from System A (that leg is (9.007e15 − 1e13)/15 m/s = 6.0e14 s = 19.0 Myr). It requires walking out of the SECOND star system's SOI, which every forest contains at 4.23 ly. Zero travel needed.

FOUR CORRECTIONS TO THE CLAIM — three make it WORSE, one is an over-claim:
(a) WORSE — trigger is not "the first movement input". `integrate` (stub.rs:2524-2559) runs normalize unconditionally after the finite/dup-seq gates; `local_axes_from_movement` with zero movement yields zero step but normalize still fires. ANY accepted input datagram — a look-only mouse move — triggers it.
(b) WORSE — the claim understates the ceiling for anything that crosses the wire. `StampedPose::sanitized` (pose.rs:490-496) clamps cell to ±CELL_DOMAIN_MAX = i64::MAX/2 → 4.503599627370496e15 m = 0.47603 ly, and it IS applied server-side at crossing ingress (stub.rs:3571), ghost refresh (:3220), re-home ingress (:3165) and client ingress (view.rs:99, realm_view.rs:81). So a saturated pose is HALVED AGAIN at the next hop.
(c) WORSE — the safety story documented at pose.rs:396-402 is false. It says `re_anchor` is "THE single quantization decision point — the integrator, frame entry, and the generator call it, all `Inert` through P3/P4". Grep over all of crates/ + tests/: `re_anchor`/`CellAnchor` has ZERO callers outside pose.rs's own unit tests. The integrator calls `.normalize()` directly. There is no inert gate on this path.
(d) OVER-CLAIM — "they keep walking; the position never changes again" is half wrong, and the freeze is mis-attributed. Measured: after pinning, outward ticks produce zero displacement, but INWARD ticks still decrement the cell — it is a one-way wall, not a freeze. And I measured the control `4.0e16_f64 + 0.75 == 4.0e16` → TRUE, so at that magnitude raw f64 eats the step regardless of saturation. The unique, catastrophic contribution of the saturation is the TELEPORT (3.099e16 m = 3.28 ly), plus the destruction of the very fold that makes movement exact at distance below the cliff.

WHAT I COULD NOT REFUTE IT WITH: no magnitude bound exists anywhere on the offset (`finite_or_zero` only kills NaN/Inf); `convert_tier` same-tier is identity and no COARSE pose is ever produced; nothing rejects, logs, or counts.

SCOPE HONESTY — two qualifiers that do NOT refute it:
- Zero impact at HEAD. The live world's galaxy radius is 180 m, 5e13× below the cliff. This is unreachable today.
- It is NOT an independent finding. It is exactly H1 of the removal surface (Universe/Galaxy aliased to System(0)/System(1) ⇒ FINE tier ⇒ ±0.952 ly), with the same root cause and the same fix (a Universe/Galaxy frame arm returning COARSE + a live `convert_tier`). It is the sharpest concrete statement of the blocker already ranked #1 in the removal order, not a second workstream.

Severity: blocker for the stated goal (one world at true astronomical scale) — it is on the shipped path, silent, and catastrophic, and it gates the coordinate base the whole true-scale world sits on. It is not a blocker for anything shipping today.

SHARPEST CONSEQUENCE (one line): At true scale, any input packet — even a mouse look — from a player more than 0.952 ly from their frame's origin (which is anyone who steps out of the second star system, 4.23 ly out in the Galaxy frame) silently relocates them 3.28 light-years toward the origin and leaves an impassable outward wall behind them, because the per-tick fold that makes movement exact at any distance saturates to i64::MAX instead of erroring.

---

## [blocker] Nothing in the codebase can ever produce a COARSE-tier pose. `RealmId` has no Galaxy/Universe arm, so `frame_for_realm` — the one total forward map — cannot emit `GalaxySpace`, the only frame whose tier is Coarse.

**At true scale:** Every realm in the world — including a Galaxy of radius 5e20 m and a Universe of 8.8e26 m — lands on the FINE 2^-10 m lattice, whose entire i64 span is 9.0072e15 m. The galaxy is 55,511x wider than its own coordinate system's representable range, and the universe 9.8e10x. Every position above the star-system level saturates on contact (previous item). The COARSE rung, which comfortably holds both (52,850 ly and 9.30e10 ly against an i64 span of 9.2234e18 ly), is unreachable.

**Evidence:** crates/core/src/pose.rs:112-121 `FrameRef::tier()`: `GalaxySpace => Tier::Coarse`, all five other arms => `Tier::Fine`. crates/core/src/pose.rs:133-147 `frame_for_realm` matches `RealmId` {Planet, System, Ship, Station, Area} — every arm returns a Fine frame; there is no Galaxy variant to match. `FrameRef::realm()` (pose.rs:78) returns `None` for GalaxySpace, so no realm can own one. Grep for `GalaxySpace` across the whole tree: the only non-test occurrences are two `map_or(FrameRef::GalaxySpace, …)` empty-forest defaults at crates/sim/src/stub.rs:1074 and :1179, whose own comments (:1083, :3776-3778, :12015) state the detector short-circuits before reaching them and that no live shard uses GalaxySpace.

**Fix shape:** `RealmId` needs Galaxy/Universe arms (or the frame→tier map needs to key on something other than a variant that no realm can name), `frame_for_realm` needs the matching arms, and the region generator must place galaxy-level bodies in a Coarse frame. This is a precondition for the true-scale world, not a follow-up — the ordering is: tier/frame map, THEN generator, THEN config removal.

**Refutation attempt failed because:** VERIFIED, NOT REFUTED.

STRUCTURAL CLAIM CONFIRMED BY READING: pose.rs:112-121 FrameRef::tier() maps GalaxySpace=>Coarse and all five other arms=>Fine; pose.rs:133-147 frame_for_realm matches RealmId {System,Planet,Ship,Station,Area}, every arm a Fine frame, no Galaxy variant to match; pose.rs:78 realm() returns None for GalaxySpace. Grep confirms the only non-test GalaxySpace uses outside pose.rs are the two empty-forest map_or defaults at stub.rs:1074/:1179.

I closed two holes the prior agent did not check; both STRENGTHEN the finding. (1) to_regions (worldgen.rs:218) is the one region-frame constructor and it calls frame_for_realm(b.realm, b.parent); UNIVERSE=RealmId::System(0) and GALAXY=RealmId::System(1) (worldgen.rs:91-92), so the Universe and Galaxy regions are handed SystemSpace{0}/SystemSpace{1} => Fine. shard.rs:132-136 sources the shard's own frame from that same forest or falls back to frame_for_realm - Fine-only either way. (2) The Galaxy concept EXISTS one layer up and is destroyed at exactly the wrong seam: RealmKindTag has Universe/Galaxy arms (level_of, worldgen.rs:293-295) but RealmLevel::to_realm_id (realm_path.rs:80-89) collapses them to System(0)/System(1). The coord layer knows it is a galaxy; the pose layer - the layer that selects the coordinate unit - cannot.

ARITHMETIC REDONE FROM SCRATCH, all six figures correct: FINE span i64::MAX x 2^-10 = 9.007199e15 m = 0.952062 ly (claimed 9.0072e15); galaxy 5.0e20 / FINE span = 55511.2x (claimed 55511x); universe 8.8e26 / FINE span = 9.76996e10x (claimed 9.8e10x); COARSE span = 9.223372e18 ly (claimed 9.2234e18); galaxy = 52850 ly; universe = 9.30161e10 ly. Saturation traced through normalize (pose.rs:301-320) at CANONICAL_SYSTEM_B_OFFSET_M=4.0e16: carry=4.096e19 > i64::MAX, the f64->i64 cast saturates, cell=i64::MAX, offset = 4e16 - 4.096e19*2^-10 = 0.0 exactly, represented 9.007199e15 m - 3.099e16 m = 3.276 ly deleted silently.

TWO REFINEMENTS (neither refutes). (a) The failure splits by object class and the claim over-attributes saturation: compose (pose.rs:351-362) and LatticePos::local never normalize, so realm centres folded by fold_origin / realm_abs_center (gateway.rs:1827, hardcoded Tier::Fine) ride at cell=0 with the full f64 - they DEGRADE rather than saturate (f64 ULP is 65536 m at 5.0e20 m and 1.374e11 m ~ 0.92 AU at 8.8e26 m). Only occupants saturate, via the one production normalize caller I could find: the movement integrator, stub.rs:2551-2555, .normalize(dot.pose.frame.tier()), run unconditionally on every applied input tick. re_anchor/CellAnchor has ZERO production callers, so it is not the gate - the integrator normalizes directly. (b) There is no single-tier escape hatch, which makes it worse: widening FINE_CELL_EDGE_M instead of activating Coarse needs an edge >= 5.0e20/i64::MAX = 54.21 m to span the galaxy and 9.541e7 m (95400 km) to span the observable universe; at a 1 m edge the span is only 974.9 ly, still 54x short of the galaxy radius. Millimetre motion and galaxy extent are provably not co-representable in one i64 tier, so the two-tier design is necessary and the rung that makes it work is the unreachable one.

SEVERITY: blocker. It is inert at today's 180 m world (nothing goes near 0.95 ly), and the code's own docs concede the state (pose.rs:110 "Through P3 only SystemSpace is live"; worldgen.rs:89 "Universe, Galaxy get one at P4+"). But under the ruling to build the one true-astronomical-scale world, a documented P10 deferral sitting on the critical path is exactly a blocker: it is a precondition, and no amount of preset deletion or generator unification reaches around it.

SHARPEST CONSEQUENCE: Every frame in the world resolves to the FINE tier, so the whole universe is confined to a +/-0.95 light-year integer lattice - the first movement tick of any occupant more than 0.95 ly from its realm's origin silently rewrites its position to exactly 0.95 ly with a zero residual, and every realm centre beyond that rides as raw f64 with 65 km of resolution at the galaxy radius.

---

## [major] Every cross-frame transfer collapses the integer lattice back to a single f64 metre triple and writes the result at cell ZERO — and the exact cell anchor that IS computed for it is silently discarded.

**At true scale:** Every realm crossing — the single most frequent authoritative operation in a demand-driven world, and the whole mechanism of warp — quantizes the player's position to ULP(distance-from-frame-origin). Inside a star system that is 2 mm (harmless). At the galaxy level a crossing displaces you by up to 65 km, and the cell-0 result it writes is exactly the un-normalized full-magnitude offset that the next integrator tick saturates. The two defects compound: the crossing creates the value that the saturation destroys.

**Evidence:** crates/core/src/frame.rs:139-163: `let local = pose.pos.offset() + pose.pos.cell().as_dvec3() * pose.frame.tier().cell_edge_m();` then `world_pos`, `rel`, `new_pos` are all plain f64 DVec3 arithmetic, then `pos: LatticePos::local(new_pos)` (frame.rs:161) — cell forced to ZERO. `FramePlacement::origin_cell` (frame.rs:36) is declared and IS written with a real value at crates/sim/src/stub.rs:1101 (`origin_cell: frame_pos.cell() - root_pos.cell()`), but grep for `origin_cell` returns only frame.rs:35/36/53/66 (doc + decl + two ctors), frame.rs:495/533 (tests) and stub.rs:1101 — it appears NOWHERE in `transfer_frame_resolved`. Quantization imposed by the fold-to-metres, by magnitude: 30.5 µm at 1 AU, 1.95 mm at 1e13 m (66.85 AU), 8 m at 4.0e16 m (4.23 ly), 65,536 m at 5.0e20 m.

**Fix shape:** Read `from.origin_cell`/`dest.origin_cell` and carry the transform in exact integer cells plus a bounded residual, writing the dest pose with a real cell (`LatticePos::at`, not `::local`). The field and its producer already exist — only the consumer is missing. Guard it with a round-trip proptest at astronomical magnitudes, not at 180 m where cell-0 and cell-anchored are indistinguishable.

**Refutation attempt failed because:** The base code fact is REAL and I confirmed it; three of the four severity-bearing legs are WRONG.

CONFIRMED BY READING
- `crates/core/src/frame.rs:145` folds the lattice into one f64 metre triple (`pose.pos.offset() + pose.pos.cell().as_dvec3() * tier().cell_edge_m()`), and `:161` writes `pos: LatticePos::local(new_pos)` — cell forced to ZERO. Both exactly as stated.
- `origin_cell` genuinely appears nowhere in `transfer_frame_resolved` (grep: frame.rs:35/36/53/66/141, tests 495/533, stub.rs:1101 — that is all).
- The path is live: `rebind_pose_to_dest` at saga_runtime.rs:919/979, stub.rs:3383 (with real `LocalFrames`), stub.rs:4205; and `region_signed_distance` (geometry.rs:1018) routes the CONTAINMENT decision through the same `transfer_frame`. Non-zero cells are live today — `integrate` normalizes unconditionally every tick (stub.rs:2551-2555).
- The four ULP numbers are arithmetically right. I recomputed: ULP(1.496e11)=3.0518e-5, ULP(1e13)=1.9531e-3, ULP(4e16)=8, ULP(5e20)=65536.

REFUTED LEG 1 — "the exact cell anchor that IS computed for it is silently discarded". FALSE. `origin_cell` at stub.rs:1101 is `frame_pos.cell() - root_pos.cell()`, and both operands come from `frame_abs_map` (stub.rs:1030-1044) → `fold_origin` (worldgen.rs:863-872). `fold_origin` seeds `LatticePos::local(DVec3::ZERO)` (cell ZERO by construction) and composes only `LatticePos::local(link_pos)` terms (also cell ZERO), and `compose` (pose.rs:358-361) adds cells with NO normalize — by the types, the result is `I64Vec3::ZERO` for every input, today and at true scale. The discarded anchor is identically zero; discarding it discards nothing. The real gap is upstream in `fold_origin`, which is a different finding.

REFUTED LEG 2 — "at the galaxy level a crossing displaces you by up to 65 km". UNREACHABLE. The 8 m figure needs a pose at 4.0e16 m and the 65 km figure needs 5.0e20 m. At FINE (2^-10 m cells) the i64 cell overflows past 9.007e15 m = 0.952 ly, so neither magnitude can exist as a FINE `LatticePos` at all. The only tier that could carry them is COARSE, selected solely by `FrameRef::GalaxySpace` (pose.rs:114), and `frame_for_realm` (pose.rs:133-146) never returns it — grep finds no production `GalaxySpace` producer outside the empty-forest fallback (stub.rs:1074, :1179). I MEASURED the worst-case fold error over 20k sub-cell residuals inside the tier's own domain: 1.1e-13 m at 180 m, 1.5e-5 m at 1 AU, 1.95e-3 m at 1e13 m (canonical system SOI), 0.25 m at the wire-sanitize limit 0.476 ly, 2 m at the FINE ceiling 0.952 ly. Two metres is this defect's CEILING, and only at a distance no realm reaches.

REFUTED LEG 3 — "the two defects compound: the crossing creates the value that the saturation destroys". FALSE. `normalize` saturates only when `offset/edge > i64::MAX`, i.e. offset > 0.952 ly. A pose at that magnitude could not have been riding as cell+offset either (the cell overflows), so the crossing does not create it; and the integrator normalizes unconditionally every tick, so such a pose saturates on its first tick with or without any crossing. Measured: at 0.476 ly the fold→cell-0→re-normalize round trip does not saturate and costs 0.25 m. Saturation is H1 (the frame→tier map), not an amplification of this.

REFUTED LEG 4 (severity) — the cell-0 write is transient, not sticky: I measured the fold→cell-0→next-integrator-tick round trip returns the IDENTICAL cell at 1 AU and lands one cell (1.7 mm) off at 1e13 m. And fixing this alone would not fix galaxy scale anyway, because `FramePlacement.origin` is a plain `DVec3` of metres (frame.rs:38) — at 5e20 m the placement itself is only 65 km precise, so the whole placement type must go lattice-native. That is the same H1/H2 tier work, not an additive finding.

WHAT SURVIVES, SIZED: at true scale a crossing rounds the position once, by ≤ 244 µm for any planet-SOI crossing (planets at 1–8.35 AU) — BELOW the 977 µm FINE cell edge, i.e. lossless relative to the lattice's own quantum — and ≤ 1.95 mm at the system's 66.85 AU outer shell. Against a 3 m containment dead-zone (worldgen.rs:46/48) and a 0.75 m per-tick movement step, that is 3+ orders below significance, one-shot, immediately re-anchored, non-accumulating. It is already ledgered in the code comment at frame.rs:141-144 as owed with the P4/P5 re-centering.

UNMEASURED: nothing was compiled or run in-repo; the fold/normalize/saturation arithmetic was reproduced in IEEE-754 doubles outside the tree, which is the same semantics but not the shipped binary.

---

## [blocker] The live AoI band CANNOT BE CONSTRUCTED for a galaxy- or universe-sized realm — the generator panics. `AoiConfig::for_velocity_safe` sets `spin_up = extent·f` and `tear_down = max(extent·f, spin_up + need)` where `need = |v_occ+v_child|·dt·(K_SAFETY+extra)`. Because `spin_up_factor == tear_down_factor` (the single visibility factor), the whole hysteresis gap IS `need` — a fixed metre amount, scale-invariant. Above a certain extent `spin_up + need` rounds back to `spin_up` in f64, `tear_down > spin_up` is false, the ctor returns `InvalidEdges`, and `to_regions` `.expect()`s it.

**At true scale:** Every shard boot and every gateway login-registry build calls the generator. With one world and a live demand band, the process aborts at boot with an `.expect` panic — not a degraded world, no world at all. The Universe and Galaxy realms are exactly the two that must exist in any single-world forest.

**Evidence:** crates/core/src/geometry.rs:817-840 (ctor), crates/core/src/worldgen.rs:210-214 (`.expect("aoi band edges are valid by construction")`). MEASURED: `realm_regions_for_config(0, canonical+5 planets+visual interest)` → `panicked at worldgen.rs:214: aoi band edges are valid by construction: InvalidEdges`. MEASURED by bisection on the real ctor, largest constructible extent = 2.884218e14 m at the shipped occupant speed 2 m/s (worldgen.rs:1117); 2.579838e18 m at v_child = 29.78 km/s; 8.255481e19 m at v = 1e6 m/s. CANONICAL_GALAXY_R_M = 5e20 (worldgen.rs:982) is 1.73e6× over the first and still 6.1× over the third; CANONICAL_UNIVERSE_R_M = 8.8e26 is 3.05e12× over. Arithmetic: need = 2.0·0.05·(2.0+0.5) = 0.25 m; ULP(5e20·14.3007 = 7.15e21) = 2^20 = 1.049e6 m ≫ 0.25 ⇒ the addition is a no-op.

**Fix shape:** The AoI dead-zone must be RELATIVE (a fraction of `spin_up`, or a per-realm datum derived from the realm's own scale and the traversal speed at that scale), not an absolute metre pad added to an astronomical radius. Equivalently: `spin_up_factor != tear_down_factor` so the geometric gap does not collapse. Either way the ctor stops depending on a 0.25 m addition surviving at 7e21 m.

**Refutation attempt failed because:** ## VERDICT: not refuted — the mechanism and arithmetic are correct. Severity downgraded blocker → major.

### What I verified independently (all recomputed, not read)

`visibility_factor(0.139626) = 1/tan(0.069813) = 14.300701209730468` — matches.

`InterestConfig::build` (worldgen.rs:1181-1195) passes `v_rel = occupant_v_max_mps + v_child`, and `AoiConfig::for_velocity_safe` (geometry.rs:817-840) computes `need = |v_rel|·dt·(K_SAFETY + extra)`. At the shipped visual values (`VISUAL_OCCUPANT_V_MAX_MPS = 2.0` worldgen.rs:1117, `AOI_TICK_DT_S = 0.05` :1107, `K_SAFETY = 2.0` geometry.rs:31, extra `0.5`): `2.0·0.05·2.5 = 0.25` m **exactly** (the f64 product rounds to exactly 0.25 — I checked, the excess is 1.39e-17, below the 2.78e-17 half-ULP).

Because `spin_up_factor == tear_down_factor` in both visual presets (worldgen.rs:1401-1402, :1432-1433), `base_extent·tear_down_factor` is the *same expression* as `spin_up`, so `tear_down = max(spin_up, spin_up + 0.25)` and the entire gap is that 0.25 m. `spin_up + 0.25 == spin_up` whenever `0.25 < ½·ULP(spin_up)`, i.e. `ULP > 0.5`, i.e. `spin_up ≥ 2^52`. Hard threshold: `2^52 / 14.300701 = 3.1492e14 m`. Their galaxy check: `5e20·14.3007 = 7.1504e21`, which lies in `[2^72, 2^73)`, so `ULP = 2^20 = 1 048 576 m ≫ 0.25` — their ULP claim is exactly right. `CANONICAL_GALAXY_R_M = 5e20` (worldgen.rs:982) and `CANONICAL_UNIVERSE_R_M = 8.8e26` (:981) both fail; `CANONICAL_SYSTEM_SOI_R_M = 1e13` (:984) and the 9e8 planet SOI both pass. So the panic is on the Universe and Galaxy realms specifically, as claimed. `to_regions` (worldgen.rs:206-215) `.expect`s it at :214 for **every** body unconditionally, and both the shard boot path and the gateway login-registry path funnel through `to_regions`. My bisections at v = 29 780 m/s and v = 1e6 m/s returned `2.579838e18` and `8.255481e19` — identical to theirs.

### One inaccuracy in their report (does not change the conclusion)

"Largest constructible extent = 2.884218e14 m at 2 m/s" is not a well-defined quantity: the predicate is **not monotonic** in extent. Inside the binade `[2^51/f, 2^52/f) = [1.5746e14, 3.1492e14)` the ULP is exactly 0.5 and `need` is exactly half an ULP, so round-to-even makes validity flip value-to-value (I get `ok` at 2.9e14 but `Err` at 1.6e14, 1.9e14, 3.1e14). Bisecting an ill-posed predicate gave me `1.844634e14`; they got `2.884218e14`. The defensible number is the **monotone** cliff at `3.1492e14 m`.

### Why this is major, not a blocker

1. **It is not intrinsic to the constructor or to true scale — it is intrinsic to the choice `spin_up_factor == tear_down_factor`, which only 2 of the 7 presets make.** The codebase already ships the counterexample: `walk_demand` uses 1.2 / 1.8 (worldgen.rs:1125-1126), making the gap `extent × 0.6` — scale-*proportional*, so it can never round away. I checked it at both offending extents: at 5e20 → spin 6.00e20 / tear 9.00e20; at 8.8e26 → spin 1.056e27 / tear 1.584e27. Both valid, nowhere near f64 overflow. Restoring any proportional geometric hysteresis removes the cliff at every extent simultaneously — a preset-level edit, no data model, wire, or schema change.

2. **The two realms that trip it are the two whose bands the demand loop never consults as roots.** `region.aoi` is read only via `child_placements(config.realm, …)` (sim/src/stub.rs:5633, and `.in_range` at :5654) — i.e. a region's band is evaluated only when that region is a *direct child of the realm a shard hosts*. The Universe has `parent: None` (worldgen.rs:457-461), so no shard ever evaluates its band. The panic is a construction-time guard firing on data that is dead for Universe and near-dead for Galaxy. Giving the ambient root an inert band is an equally local, equally legitimate fix.

3. **It fails loud, deterministically, at boot, with a named error, and the sibling degenerate case is already unit-tested and documented.** `equal_visibility_factor_with_zero_v_rel_is_err_not_panic` (worldgen.rs:2623-2641) and the `visual_demand` doc block (worldgen.rs:1424-1428) already state in terms that band validity under equal factors rests *entirely* on the velocity pad and that "`to_regions`'s `.expect` would PANIC at boot on one". This finding is a quantitative extension of a documented tripwire — `v_rel = 0` breaks it, and so does `v_rel·dt·2.5 < ½·ULP(extent·f)` — not a newly discovered wall. Contrast the genuine blocker class in the same surface: the FINE lattice silently rewriting a 4.23 ly position to 0.95 ly with no error and no log. That is unrecoverable data corruption; this is a crash with a message.

4. **Today's headroom is 3.15e5×, and the AoI rule must be redesigned at true scale for an independent reason 2-3 orders larger.** The largest live-band extent today is not 180 m but `UNIVERSE_R_M = 1e9` (worldgen.rs:53; `visual_geometry` inherits it from `walk_scale`), giving `spin_up = 1.43e10`, `ULP = 1.9e-6` — 3.15e5× below the cliff. Meanwhile θ_min = 8° = 28 800 arcsec against a 1e13 m SOI subtending 103 arcsec at 4.23 ly is the far bigger problem, and whoever fixes those factors discharges this in passing.

### The honest counter-point that argues for fixing it soon (not for calling it a blocker)

Reducing θ_min to anything physically sane *raises* the factor and drags the cliff **down into star-system extents**: at θ_min = 1000 arcsec the factor is 412.5 and the hard cliff is 1.09e13 m = 73 AU — already below `CANONICAL_SYSTEM_SOI_R_M`; at 100 arcsec the factor is 4125 and the cliff is 7.3 AU. So the equal-factor design cannot survive the θ_min correction that true scale forces, regardless of the galaxy realm. That makes it a required item on the AoI redesign, sequenced with it — which is where a major belongs.

### Sharpest statement of the consequence

One world at true scale panics at boot in `to_regions` on the Universe and Galaxy realms — but only because the two equal-factor visibility presets make their whole hysteresis gap a fixed 0.25 m velocity pad, which drops below one f64 ULP once `extent × 14.3007 ≥ 2^52` (extent ≥ 3.1492e14 m); `walk_demand`'s proportional 1.2/1.8 factors already in-tree are valid at 8.8e26 m, so this is a preset-level defect on the AoI redesign path, not an architectural wall.

---

## [blocker] At true scale NOTHING is ever in Area of Interest, so nothing is ever spun up and nothing is ever drawn. `spin_up_r_m = finite_extent · cot(θ_min/2)` with θ_min = 8°. Every planet's own orbit is 11.6×–97× further from its star than the planet's visibility radius; the nearest sibling system is 280× further than a system's visibility radius.

**At true scale:** You spawn in a star system and see empty space. No planet realm is ever demanded, so no planet shard ever launches; and because the render set is literally the same membership as the demand set (crates/sim/src/stub.rs:5654-5675 — `next_in` drives both `push_demand` and `dot_visible`), the client is never even told a planet exists. The 180 m demo hides this because there the SAME rule gives spin_up 59.5 m against a 4.16 m planet SOI inside a 150 m system — visibility radius is comfortably larger than the whole system.

**Evidence:** crates/core/src/worldgen.rs:139 (θ_min = 0.139626 rad), :342-344 (`1/tan(θ/2)`), crates/core/src/geometry.rs:826. MEASURED factor = 14.300701209730. MEASURED on the real canonical forest (n_planets=5): planet SOI 9e8 m → spin_up 1.2871e10 m = 0.0860 AU; the five emitted orbits are sma = 1.0000/1.7000/2.8900/4.9130/8.3521 AU ⇒ orbit/spin_up = 11.62× / 19.76× / 33.59× / 57.11× / 97.08×. System SOI 1e13 m → spin_up 1.4301e14 m = 955.93 AU; CANONICAL_SYSTEM_B_OFFSET_M = 4.0e16 m = 267,380 AU ⇒ 279.7× short. Using the DRAWN body radius instead of the SOI, Earth (6.371e6 m) → spin_up 9.111e7 m; 1 AU / 9.111e7 = 1642× short.

**Fix shape:** Two separate decisions, currently one: (1) what the occupant can SEE, keyed on the drawn body radius and a threshold in arcseconds, not on the SOI at 8°; (2) what must be a LIVE shard, keyed on reachability/interaction, not on visibility. The SOI-vs-body-radius mismatch alone is 141× (9e8 / 6.371e6).

**Refutation attempt failed because:** CONFIRMED — I re-derived every number from source and all of them reproduce exactly. Mechanism verified by reading, not comments: worldgen.rs:211-214 stamps aoi = interest.build(shape.finite_extent(), v_child); a planet's shape is Shell{r: planet_soi_r_m} (worldgen.rs:479) and Shell::finite_extent is the radius (geometry.rs:300-305); geometry.rs:821 sets spin_up = base_extent * spin_up_factor, and visual_scale/visual_demand set spin_up_factor = tear_down_factor = cot(theta/2) (worldgen.rs:1399, :1430) — so spin_up_r_m = extent * cot(theta_min/2) as claimed. stub.rs:5652-5686 confirms ONE in_range result drives both push_demand and dot_visible/proxy_now, so render membership IS demand membership. gateway.rs:1793-1802 ships only the ancestor shell-chain at minor>=6 and its own doc says that split is "a FACTORING, not a second visibility mechanism"; child_shape is the only RealmShape producer to the client. The F7 predictive lead (stub.rs:5867-5878) shortens distance by |v|*horizon_s only (~1 s * 15 m/s = 15 m against 1.5e11 m) — no rescue.

MEASURED (python, from constants read in source): cot(0.139626/2) = 14.300701209730468 (matches to all digits). Canonical planet SOI 9e8 m -> spin_up 1.28706e10 m = 0.08603 AU. Emitted orbits (orbital_axis_au(n,0.4,1.7) * au_to_render 3.74e11) = 1.0000/1.7000/2.8900/4.9130/8.3521 AU -> orbit/spin_up = 11.62/19.76/33.59/57.11/97.08. System SOI 1e13 -> spin_up 1.43007e14 m = 955.94 AU; CANONICAL_SYSTEM_B_OFFSET_M 4.0e16 = 267,383 AU -> 279.71x. Drawn Earth radius 6.371e6 -> spin_up 9.111e7 m -> 1 AU is 1641.95x. Every figure the prior agent gave is correct.

TWO CORRECTIONS, BOTH MAKING IT WORSE, NOT BETTER.
(1) This is NOT primarily a metre-scale artifact. For a physical Hill sphere the ratio orbit/spin_up = (3M/m)^(1/3)/cot(theta/2) is SCALE-INVARIANT — it depends only on the mass ratio and theta. Real solar-system masses: Mercury 18.35x, Earth 6.99x, Mars 14.70x, Neptune 2.71x, Jupiter 1.02x. So even a physically-correct true-scale preset (real Hill spheres instead of canonical's flat 9e8 m) still culls every terrestrial planet from its own star by 7-18x. The 180 m demo works only because visual_geometry MANUFACTURES the planet SOI as 35% of the inter-orbit gap (worldgen.rs:373-380), giving SOI/orbit = 0.2450 versus physics' 0.0100 for Earth — a 24.5x inflation. That inflation, not the metres, is what hides the defect. Correcting canonical's separate 2.5x au_to_render defect does not help either: real orbits 0.4-3.34 AU against the flat 9e8 m SOI still give 4.65x-38.83x.
(2) canonical() sets interest: InterestConfig::inert() (worldgen.rs:1345), so spin_up_factor = 0 -> AoiConfig::inert() -> spin_up_r_m = 0 -> literally nothing is demanded or drawn at true scale at HEAD. There is no configuration in the tree today that yields a visible true-scale world; the two routes to the empty sky are the inert band and the cot rule.

NO SINGLE-KNOB FIX EXISTS, which is why this is structural and not tuning. To make Earth's Hill sphere subtend theta_min at 1 AU needs cot >= 100 (theta <= 1.146 deg); at that theta a 1e13 m system SOI reaches 9.997e14 m = 0.1057 ly — still 40x short of the 4.23 ly neighbour. To reach 4.23 ly needs cot >= 4000 (theta ~ 103 arcsec); at THAT theta a 1.5e9 m Hill sphere reaches 5.99e12 m = 40 AU, so every planet in the system is permanently in AoI and every planet shard is permanently live. The planet tier and the system tier pull the same constant in opposite directions by ~40x.

SEVERITY blocker, upheld. Under the owner's ruling (one final world, true astronomical sizes, full game machinery) the shipped visibility rule yields a sky containing only the static cosmetic starfield (client-render/src/lib.rs:84-91, seed-fixed, unrelated to the real forest). Nothing to see, nothing to fly to, nothing spun up — the world is not merely degraded, it is unobservable and untestable, and no value of theta_min recovers it. Confidence is high on the arithmetic and the code path (both read and recomputed); what remains UNMEASURED is only the runtime confirmation, whose measurement would be: build the canonical geometry with a live interest band, run one tick of the demand/render pass with an occupant at the system origin, and assert dot_visible and the demand outbox are both empty.

SHARPEST STATEMENT OF THE CONSEQUENCE: at real astronomical proportions the radius at which a body must be simulated and the radius at which it must be seen differ by three to four orders of magnitude, and the code derives both from one band on one angular threshold — so the true-scale sky is empty, by construction, at every setting of that threshold.

---

## [blocker] Angular size is the wrong criterion for an astronomical sky, and no single θ_min can serve both the render set and the lifecycle set. They are the same predicate today.

**At true scale:** Either the sky is empty (large θ, nothing visible) or ~10⁷ realms are demanded at once (small θ). Each demanded realm is an OS process under the RLM spawner. A real naked-eye sky is ~9,000 point-lights every one of which is sub-milliarcsecond — brightness, not angular size, is what makes them visible. The standing law "stars always visible, warp = physical fly-by" cannot be met by any setting of this one knob.

**Evidence:** crates/sim/src/stub.rs:5654-5687 — `region.aoi.in_range(...)` → `next_in` feeds BOTH `push_demand` (lifecycle) and `dot_visible`/`proxy_now` (render). MEASURED angular sizes: shipped θ_min = 8° = 28,799.9 arcsec. Earth from 1 AU = 17.57 arcsec (6.10e-4× threshold). The MOON from EARTH = 1864.5 arcsec (0.065× threshold — 15× too small). The Sun from 4.23 ly = 0.0072 arcsec (2.49e-7× threshold). Arithmetic on the other side: to make a 1e13 m system SOI visible at 1000 ly needs θ = 2.114e-6 rad = 0.436 arcsec; at solar-neighbourhood density (≈0.00254 systems/ly³, from ~370 systems within 10 pc — an EXTERNAL estimate, not from this repo) that is 4/3·π·1000³·0.00254 = 1.07e7 systems inside the spin-up radius simultaneously. At 4.23 ly it is 0.81 systems; at today's 0.0151 ly it is 3.7e-8.

**Fix shape:** Split the predicate. A far realm must be REPRESENTABLE (a server-authored point of light with a pose and a brightness) without being LIVE. That is a third state between 'spun up' and 'absent' which the RLM does not have. Note the current night sky is a client-local fake (`STARFIELD_SEED`, crates/client-render/src/lib.rs:91, drawn on a 2000 m sphere that follows the camera, unrelated to the universe seed) — under the server-authoritative law it has to become real server output.

**Refutation attempt failed because:** NOT REFUTED. The core diagnosis is confirmed in code, and every number they gave is right. But two of their three consequence statements are wrong in mechanism, which lowers this from blocker to major.

WHAT I CONFIRMED BY READING

1. One predicate, two consumers — TRUE, verbatim. `crates/sim/src/stub.rs:5654` computes `let now_in = region.aoi.in_range(state.was_in, dist);` → `aoi_transition` (`:5898`) → `(verb, next)`. The verb drives `push_demand` (lifecycle, `:5705`); `let next_in = next.is_some_and(|s| s.was_in)` (`:5673`) gates `dot_visible` (`:5675`) and `proxy_now` (`:5684`). Both read the same band on the same tick. `RealmRegion` (`crates/core/src/geometry.rs`) carries exactly ONE `aoi: AoiConfig` field — there is no second band to split render off onto.

2. θ really does drive both. `visual_scale` (`worldgen.rs:1398`) and `visual_demand` (`:1429`) both set `spin_up_factor = tear_down_factor = visibility_factor(VISIBILITY_THETA_MIN_RAD)`. `visibility_factor` (`:342`) = `1/tan(θ/2)`. I get 14.3007012 — the doc's "≈14.301" is right. `InterestConfig::build` (`:1181`) multiplies it by `b.shape.finite_extent()`, and for `Shell{r}` that is `r` (`geometry.rs:2095`), i.e. the SOI radius, not the drawn body radius. So the rule is exactly "angular DIAMETER ≥ θ_min" — self-consistent, and keyed on the wrong extent.

3. The AoI delta is the ONLY way a non-ancestor realm reaches the client. Client scene = `from_shapes` on the gateway's ancestor-chain registry + `with_delta` (`client/src/realm_scene.rs:275`), and the sole producer of `RealmSceneDelta` in the sim is the `next_in` gate above plus `on_proxy_scene_set` (`stub.rs:6036`), which reflects the same set.

MY ARITHMETIC (independent, python3, not theirs)
- θ = 0.139626 rad = 7.99998° = 28,799.93 arcsec; cot(θ/2) = 14.3007012
- Earth from 1 AU: 2·atan(6.371e6/1.496e11) = 8.5174e-5 rad = 17.5686 arcsec = 6.100e-4 × θ ✓
- Moon from Earth: 2·atan(1.7374e6/3.844e8) = 9.0397e-3 rad = 1864.53 arcsec = 0.0647 × θ (15.4× too small) ✓
- Sun from 4.23 ly: 1.3927e9/4.0019e16 = 3.480e-8 rad = 0.007178 arcsec = 2.492e-7 × θ ✓
- θ for a 1e13 m SOI at 1000 ly: 2·atan(1e13/9.4607e18) = 2.1140e-6 rad = 0.4360 arcsec ✓
- Census at 0.00254/ly³: 1000 ly → 1.064e7; 4.23 ly → 0.805; 0.01512 ly → 3.67e-8 ✓ (their density also checks: 370/[4/3·π·32.616³] = 0.002546/ly³)
Every figure reproduces. No error found.

WHERE THEY ARE WRONG

(a) "~10⁷ realms demanded at once, each an OS process" is UNREACHABLE. The AoI loop iterates `regions.child_placements(config.realm, …)` (`stub.rs:5632`) — only the shard's own direct children inside its `RealmRegions`, and `guard_region_count` (`geometry.rs:1149`) hard-refuses boot at `regions.len() > MAX_REGIONS = 64` (`stub.rs:845`, enforced `bins/src/bin/shard.rs:260`). A shard's entire demand candidate set is ~61 children by construction. A small θ therefore produces `RegionNestError::TooManyRegions` — a Galaxy shard that will not start — not a process storm. (`ProcLaunchBackend`, `bins/src/proc_launch.rs:95`, does fork a real `vd-shard` per realm, so the process claim is right in kind, wrong in count.)

(b) "The sky is empty" overstates. `crates/client-render/src/lib.rs:441-521` already draws an always-on ambient starfield — 1600 uniform + 1200 Milky-Way-band unlit emissive points from a fixed `STARFIELD_SEED`, camera-followed with identity rotation — entirely outside the AoI system. Its own doc (`:85-90`) already names the intended two-tier split: "the far, UNREACHABLE stars are a permanent night-sky backdrop… at the real-astronomy / galaxy-catalog slice (warp) this seed becomes SERVER-provided, the NEAR reachable systems promote to real realm content you fly to, and the far field stays a backdrop." So brightness-as-visibility is already the mechanism for the far field; what is missing is the promotion, not the sky.

(c) Not wrong but worth noting: `RealmRegion.aoi` is already stored PER REALM, so per-body lifecycle radii need no new plumbing. Only the derivation is one global factor. The genuinely missing thing is a SECOND predicate for render.

WHY MAJOR, NOT BLOCKER
- Nothing is silently wrong today; at 180 m the rule is correct and gated.
- It is strictly downstream of two harder true-scale stops that fire first and that no θ can address: the FINE-tier i64 range (`pose.rs:163`, 2⁻¹⁰ m cell ⇒ ±0.952 ly representable, silently saturating), and MAX_REGIONS = 64.
- The fix is a required redesign (split the predicate; key render on drawn radius + a magnitude model, key lifecycle on SOI + census density), not a tuning change — but it is schedulable after the coordinate tier, and half of it (the far-field backdrop) already exists with the seam documented.

SHARPEST CONSEQUENCE
Recomputed: with the shipped θ, a canonical 1e13 m system SOI becomes a live, drawable realm only at 1e13 × 14.3007 = 1.4301e14 m = 955.9 AU = 0.01512 ly, while the nearest sibling system sits at 4.23 ly = 267,400 AU — and the backdrop that fills the gap is camera-locked and parallax-free (`follow_starfield`, `client-render/src/lib.rs:525`, position-only, identity rotation). So 99.64% of an interstellar trip has literally zero visual change, and the destination pops into existence over the last 0.36%.

---

## [blocker] The client renderable-extent filter drops 100% of a true-scale world, and the camera far plane is 6 km.

**At true scale:** Black screen. Not a degraded view — every realm hits `continue` before it becomes a `RealmBox`. Even with the filter removed, one perspective camera with near≈0.1 and far spanning a star system has no usable depth buffer.

**Evidence:** crates/core/src/worldgen.rs:105 `MAX_RENDERABLE_EXTENT_M = 200.0`; applied at crates/client/src/realm_scene.rs:245 (`from_shapes`) and :285 (`with_delta`) — `continue`, silently skipped. MEASURED on the real forests: true-scale canonical (5 planets) → 0 of 8 realms survive the filter; today's visual → 7 of 8 survive. Smallest true-scale extent = 9.0e8 m, i.e. 4.5e6× over the 200 m limit. crates/client-render/src/lib.rs:97 `STAR_FAR_PLANE = 6000.0`, applied at :326; a planet 1 AU from the pin is 2.5e7× beyond it. Mesh translation AND scale are cast to f32 at crates/client/src/realm_scene.rs:604 and :610 — f32 relative precision 6e-8 gives 8,960 m absolute at 1 AU and 596 km at 1e13 m.

**Fix shape:** The render extent must be server-told per realm, not a `vd-core` constant. Note `ScaleConfig::render_extent_m` (worldgen.rs:1004) already exists as the per-world field and has ZERO non-test consumers (verified by grep: only worldgen.rs:1241/1300 write it and 1917/1929/2648/3093/3095 assert on it) — the client filters on the hardcoded const instead. Beyond that, drawing across 10+ orders of magnitude needs camera-relative positions and a multi-pass/log-depth scheme that does not exist.

**Refutation attempt failed because:** CORE CONFIRMED, EVIDENCE PARTLY WRONG, SEVERITY OVERSTATED.

WHAT SURVIVES (re-derived, not taken on trust):
- worldgen.rs:105 `MAX_RENDERABLE_EXTENT_M = 200.0`. geometry.rs:300-305 `finite_extent()` = `r` for Shell, `half.max_element()` for boxes. realm_scene.rs:245 and :285 both do `if extent > MAX_RENDERABLE_EXTENT_M { continue; }` — on `from_shapes` (the streamed VU path) and `with_delta` (the AoI path), i.e. the shipped client, not a dev fixture.
- Canonical constants read at worldgen.rs:981-991: universe 8.8e26, galaxy 5.0e20, system SOI 1.0e13, planet SOI 9.0e8, area half 1.0e4, station half 5.0e3. generate_system_forest (worldgen.rs:449-486) gives every planet `Shell{ r: pl.planet_soi_r_m }` verbatim. So the ratios are: 9.0e8/200 = 4.5e6x (their figure — correct for the system forest); the true minimum over the whole canonical config is station_half 5.0e3/200 = 25x (walk topology, which they did not measure). Either way EVERY realm is dropped. The filter claim is not refutable.
- ScaleConfig::render_extent_m exists (worldgen.rs:1004) and is set to CANONICAL_RENDER_EXTENT_M = 1.5e11 for canonical, but grep across the whole tree shows it is read by NOTHING outside worldgen.rs's own tests (:1917, :1929, :2648, :3093, :3095). There is no existing mechanism that makes the filter scale-aware.

WHAT IS WRONG:
1. "No usable depth buffer" is FALSE. Bevy 0.18's PerspectiveProjection::get_clip_from_view is `Mat4::perspective_infinite_reverse_rh(fov, aspect_ratio, near)` (bevy_camera-0.18.1/src/projection.rs:339) — `far` is NOT in the clip matrix at all. There is no far clip plane in the depth math. Reverse-Z with an infinite far gives z_ndc = near/z_view; ULP(d) = d*2^-24, so dz = (near/d^2)*(d*2^-24) = z*2^-24 — CONSTANT RELATIVE precision 6.0e-8 at every distance, independent of far. At 1 AU that is ~8.9 km of depth resolution on an object 1.496e11 m away, i.e. 1e-4 of a pixel. Depth is a non-issue; the sentence that made this sound architectural is factually incorrect.
2. The f32 evidence is IRRELEVANT. realm_scene.rs:604/:612 cast `draw_center`, which the function's own contract (:589-596) states is "the box's centre ALREADY reduced against the server-told render origin by the caller, through the ONE `world_pos` chokepoint" — a pin-relative delta, not an absolute. Their 8,960 m at 1 AU understates the true ULP (2^14 = 16,384 m), but at 1280 px / 45 deg fov one pixel subtends 6.13e-4 rad = 9.2e7 m at 1 AU, so 16,384 m is 1.8e-4 pixels. Invisible. The prior REMOVAL SURFACE said exactly this ("the f32 cast is not the problem"); this finding contradicts its own predecessor without saying so.
3. "Black screen" is FALSE as stated. despawn_reference_scaffold (client-render/src/lib.rs:~348-359) despawns the P1.5 ground plate + 8 landmark pillars ONLY `if !boxes.0.is_empty()`. At true scale the box set is empty, so the scaffolding STAYS, alongside the starfield sphere (STAR_SPHERE_RADIUS 2000, inside the 6000 far). The player sees a plausible-looking debug world with zero real content and no error — arguably worse than black, but not what was claimed.
4. The far plane is a CPU-CULL constant, not a projection wall, and they cited the wrong camera for the gate. STAR_FAR_PLANE = 6000.0 (lib.rs:97) is applied ONLY to the WINDOWED camera (:325-327). The headless capture camera (:1008-1017) spawns `Camera3d::default()` with NO projection override, so it runs Bevy's default far = 1000.0 (bevy_camera projection.rs:422). `far` reaches rendering solely through compute_frustum -> Frustum::from_clip_from_world_custom_far(..., self.far()) (projection.rs:72-77, primitives.rs:308-318) — a CPU frustum half-space consumed by check_visibility. So beyond-far objects ARE culled (the effect holds), but the mechanism is per-entity CPU culling, defeatable by NoFrustumCulling, fixed by one constant.

SEVERITY: MAJOR, not blocker. It is real, it is on the shipped path, and true scale cannot ship without it. But once the depth-buffer leg is removed there is no architectural wall here: no data loss, no silent corruption, no cap on the world, 100% detectable on the first run. The mechanical half is two constants becoming server-told per-realm (render_extent_m is already carried in ScaleConfig and merely unplumbed). The genuinely hard half — what you actually DRAW when the nearest realm is a 1e13 m sphere-of-influence shell that is not a body — is the SAME design question as the AoI/drawn-radius finding (H4) and pending task #181 ("draw a real star body at each system centre"), not an independent blocker. Contrast a true blocker like the FINE-tier 0.95 ly saturation, which rewrites positions silently with no error.

SHARPEST TRUE STATEMENT: flip the world to true scale and the client's `finite_extent() > 200 m -> continue` drops 100% of the realms the server ships, leaving the P1.5 debug ground plate, eight landmark pillars and a starfield on screen — a plausible-looking empty world, logged nowhere.

---

## [blocker] `MAX_REGIONS = 64` is a hard cap on how many direct children any realm may have — the shard REFUSES TO BOOT past it.

**At true scale:** A Galaxy realm hosting a real star census cannot boot. The config already carries the intent to exceed this — `CANONICAL_SYSTEM_COUNT_HI = 8` (worldgen.rs:995) is inside the cap, but a Milky Way is ~6e10 systems. The demand loop is also O(observers × direct children) per tick (stub.rs:5633-5696), so the cap is the only thing keeping that bounded.

**Evidence:** crates/sim/src/stub.rs:845 `pub const MAX_REGIONS: usize = 64` (the membership bitset is a `u64`, stub.rs:774-791); enforced at crates/bins/src/bin/shard.rs:260 via `guard_regions_nest(&regions, MAX_REGIONS)` → `RegionNestError::TooManyRegions` → "refusing to boot". A shard's region set is ancestors ∪ direct children (crates/core/src/worldgen.rs:774-791). For a Galaxy shard the ancestry is {Galaxy, Universe} = 2 ⇒ at most 62 star systems in the entire galaxy.

**Fix shape:** Direct-child membership must stop being a fixed-width bitset over a materialised child list, and the AoI scan must go through a spatial index over lazily generated children (the ledgered D-45). Until then no realm can hold more than ~61 children regardless of scale.

**Refutation attempt failed because:** CONFIRMED AS FACT, REFUTED AS A BLOCKER. Every mechanical claim checks out: MAX_REGIONS = 64 backs a u64 bitset (crates/sim/src/stub.rs:845, RegionMembership at :775-793); guard_region_count (crates/core/src/geometry.rs:1149) rejects regions.len() > max as the FIRST check in guard_regions_nest; crates/bins/src/bin/shard.rs:258-262 makes that boot-fatal in the production shard bin with the literal "refusing to boot". The region set is ancestors(held) INTERSECT-FREE-UNION direct-children(held) (neighbourhood_scope, crates/core/src/worldgen.rs:770-790; ancestor_realms at :794 includes self), so held={Galaxy} gives [Galaxy, Universe]=2 and children <= 62 — the arithmetic is correct. I also checked the one thing that could have dissolved it: whether the region set is demand-live rather than boot-static. It is boot-static — RealmRegions is planted once at shard.rs:265 and never mutated in production (grep over every RealmRegions::new / resource_mut site finds only unit tests and crates/harness/src/topology.rs:1517). The demand loop is indeed O(observers x direct children) per tick (child_placements, stub.rs:1181-1185, consumed in the nested loop at stub.rs:5633+).

WHAT IS WRONG IS THE SEVERITY AND THE FRAMING. The cap constrains the BRANCHING FACTOR of the realm tree, not the size of the world, and nothing in the traversal hardwires Galaxy->System adjacency: ancestor_realms, neighbourhood_scope, region_depth and container_coord_at's deepest_containing_child are all depth-generic. With k nested sector levels the reachable system count is the product of (62-i) for i=1..k: k=5 -> 7.14e8, k=6 -> 3.998e10, k=7 -> 2.199e12. A Milky Way (~1e11 systems) fits at k=7 with MAX_REGIONS COMPLETELY UNCHANGED. Total tree depth would be Universe + Galaxy + 7 sectors + System + Planet + Area = 12, inside the client's MAX_NEST_DEPTH = 16 (crates/client/src/realm_scene.rs:127). The intermediate levels need new RealmKindTag / RealmId arms, and both enums are documented APPEND-only (crates/core/src/realm_path.rs:49-58; crates/core/src/pose.rs:28-34 "APPENDED (discriminant 3/4) so the wire stays additive") — additive wire growth, and the same class of work the frame->tier map already owes.

Moreover, widening the bitset would not buy anything. The per-tick AoI scan is a linear distance test per (observer, direct child) and the containment detector is O(entities x regions). At 20 Hz, 62 children is 1,240 tests/s/observer; a flat 1e11 would be 2e12 tests/s/observer. The flat-galaxy shape is unrunnable independent of the bitset width, so the subdivision is mandatory for the demand loop regardless. That inverts the finding's last sentence: the subdivision is what bounds the demand loop; the cap merely refuses to boot a shape that would never have run. The finding itself concedes the config's own intent (CANONICAL_SYSTEM_COUNT_HI = 8, worldgen.rs:995; WALK_SYSTEM_COUNT = 2) is inside the cap.

WHAT SURVIVES, and why this is not "not-an-issue": the cap is a real, enforced, boot-fatal constraint that is a genuine design input — it dictates <= ~61 direct children per realm, so the first generator someone writes that emits a flat child list (galaxy -> systems, or planet -> thousands of Area districts) fails loud at boot. A designer must know the number before choosing the subdivision. Two honest caveats: (a) the cap applies to a BOOT-STATIC seed-derived set, so anything whose child count varies at runtime — P8 ship realms, which RealmId::Ship names but which are excluded from seed forests today since worldgen::level_of returns None for Ship (worldgen.rs:285,300) — would hit both the 64 cap and the boot-static-ness; that is a P8 shape question, not established here and not what the finding claimed; (b) I ran nothing — the boot-refusal is established by reading guard_region_count plus shard.rs:260, the rest is arithmetic.

CORRECTED ONE-LINE CONSEQUENCE: MAX_REGIONS caps a realm's DIRECT children at ~61, so any flat child list fails loud at boot — it constrains the world tree's branching factor, not the world's size; seven nested sector levels hold 2.2e12 systems with the 64 left exactly as it is.

---

## [major] The AoI ACQUIRE edge has no velocity safety. `width_safe_for` only widens `tear_down`, but `in_range` acquires only within `spin_up` — so at speed an occupant can pass a realm entirely between two ticks and it is never demanded.

**At true scale:** At any speed that makes interstellar distance traversable, the AoI loop is a point sampler whose stride exceeds the entire sphere it is testing against. You fly through a star system and it never spins up — the same tunnelling class as containment, on the lifecycle side.

**Evidence:** crates/core/src/geometry.rs:880-884 — `hold = d <= tear_down; acquire = d <= spin_up; (was_in & hold) | acquire`. If `was_in` is false the only path in is `spin_up`. crates/core/src/geometry.rs:828 inflates ONLY `tear_down` by `need`, and geometry.rs:871-873 checks only the gap. MEASURED band at planet SOI 9e8: v_occ = 1.496e11 m/s ⇒ spin_up 1.287063e10, tear_down 3.157063e10, dead-zone 1.87e10 m, per-tick travel 7.48e9 m. Arithmetic on the sample gap at dt = 0.05 s: planet AoI diameter 2.574e10 m, system AoI diameter 2.860e14 m. At 1 AU/s you cross 0.29 planet-AoI-diameters per tick (still sampled); at 1 ly/s (9.46e15 m/s — the speed that makes 4.23 ly a 4.2 s trip) you cross 18,380 planet-AoI-diameters and 1.65 SYSTEM-AoI-diameters per tick.

**Fix shape:** The acquire test must be a swept-segment test, or the dead-zone must be widened on BOTH edges (`spin_up` too), or the tick rate must scale with speed. `Boundary::swept` / `segment_shell_crossing` already exist for exactly this.

**Refutation attempt failed because:** The underlying property is real — AoI acquire is a two-point sampler with stride v·dt — but the finding is misstated, mis-ranked, and computed on numbers that cannot occur and are slated for replacement. Corrected to MINOR (a design note for the AoI redesign / warp, not a true-scale blocker).

WRONG #1 — "the ACQUIRE edge has no velocity safety" is false about the code. The acquire-side velocity term lives in the distance function, not the band: crates/sim/src/stub.rs:5867-5878 `occupant_child_dist` returns `min(|d|, |d − v·horizon_s|)`, horizon_s = boot_ticks_p99·tick_dt_s — the documented F7 predictive horizon (stub.rs:146-149, shard.rs:164-167). `min` can only lower the distance, so it only ever widens acquire. The agent read AoiConfig in isolation. (It did miss something real in the other direction: boot_ticks_p99 defaults to 0 in every production path — lib.rs:421, shard.rs:167, gateway.rs:88, orchestrator.rs:244, absent from deploy/k3d — so that term is currently switched off.)

WRONG #2 — the containment analogy is backwards; AoI acquire is the LAST thing in that family to break, by 14.3×. spin_up = 14.301 × finite_extent, and finite_extent is the MAX half-extent (geometry.rs:300-305), so every region's circumradius ≤ √3 × finite_extent (Shell: exactly 1×). The AoI sphere contains the region's own volume by ≥ 8.26× in radius, unconditionally. Therefore any tick at which containment can fire is a tick at which acquire fires — you can never be inside a realm whose AoI never acquired. Containment tunnels at 14.3× LOWER speed. The "fly through a system and it never spins up" case is exactly the case where you also never entered it: a missed render/warm-up, not an authority, physics, or collision break. bins/src/lib.rs:406-410 states this design intent verbatim ("the spin-up-ahead is a natural consequence of visibility-radius ≫ crossing-radius, not a predictive horizon").

WRONG #3 — the binding limit is the boot lead time, and it binds 43·t_boot times earlier, at every scale. Tunnelling needs v > 2FE/dt; booting in time needs (F−1)E/v ≥ t_boot ⇒ v ≤ (F−1)E/t_boot. Ratio = 2F·t_boot/((F−1)·dt) = 28.602·t_boot/0.66505 = 43·t_boot(s) — the extent E cancels, so it is scale-free. At canonical planet SOI: sampling fails above 5.148e11 m/s (3.44 AU/s, 1717c); lead time fails above 1.197e10 m/s at t_boot=1s (43× lower) or 2.39e9 m/s at 5s (215× lower). Anything fast enough to tunnel the acquire sphere arrived hundreds of ticks before its shard could exist even with a perfect swept test.

WRONG #4 — the speeds quoted cannot be produced. stub.rs:2524-2557: the integrator's only velocity source is move_speed_mps · tick_dt_s · time_multiplier with axes clamped to [−1,1], and shard.rs:121 threads that same move_speed·time_multiplier into visual_demand(occupant_v_max, tick_dt). "1 AU/s"/"1 ly/s" are a config change that also re-authors the band.

WRONG #5 — computed on radii that are already non-functional and whose mandatory fix raises the threshold. At canonical, planet spin_up = 0.086 AU while planets orbit at 1–8.35 AU: no planet is in AoI at ANY speed, including zero. θ_min = 8° must be re-derived. Redoing spin_up on drawn radius (Earth 6.371e6 m): at θ_min = 1′, cot(30″)=6875 ⇒ spin_up 0.293 AU ⇒ threshold 11.7 AU/s (3.4× higher than their 3.44); at 1″, cot(0.5″)=412,530 ⇒ spin_up 17.57 AU ⇒ threshold 703 AU/s (204× higher).

NOT REACHABLE TODAY by ~400×: live demo planet SOI = 0.35·0.4·0.7·[(150−4)/(0.4·1.7^4 + 0.098)] = 4.1607 m; spin_up = 4.1607 × 14.3007 = 59.5 m (reproduces the code comment at lib.rs:407 exactly). Per-tick travel 15 m/s × 0.02 s = 0.3 m ⇒ 397 samples on a central pass, 56 even at 0.99 impact parameter.

WHAT SURVIVES (why not not-an-issue): acquire is genuinely a point sampler, and the predictive horizon shifts the sample lattice without densifying it — the union of {k·dt} and {k·dt + h} still has max gap dt (best case dt/2 when h mod dt = dt/2). If a future design buys lead time with the horizon h instead of the radius ratio F — the natural move once F is re-derived from drawn radius rather than SOI — the radius ratio stops covering acquire and the sampler becomes the binding constraint. That is one line in the AoI redesign and a check on warp (task #183), not a true-scale blocker.

Sharpest corrected statement: acquire is a point sampler, but its sphere is ≥ 8.26× every region's circumradius, so it can only miss realms you also never entered — and it misses them only above a speed 43·t_boot times faster than the shard could ever boot in time.

---

## [major] Nothing in the world is reachable at any speed the code can express; warp is not a feature at true scale, it is the only connectivity.

**At true scale:** With one true-scale world and no warp, the player can never leave the volume they log into — and every AoI/band/tick guarantee above is stated against occupant speeds (2–15 m/s) that are 10+ orders of magnitude below what the world requires.

**Evidence:** crates/bins/src/lib.rs:411 `move_speed: 15.0`; deploy/k3d/50-shard.yaml:55 `VD_SPEED="2.0"`. Arithmetic at 15 m/s: 1 AU = 9.97e9 s = 316 yr; the canonical system SOI 1e13 m = 6.67e11 s = 21,100 yr; the 4.23 ly to the sibling system = 2.67e15 s = 84.5 Myr. At 0.1c the nearest star is still 42 years. Task #183 ("Warp: implement + prove it end to end") is [pending].

**Fix shape:** Warp has to land WITH the scale change, not after it, because the AoI acquire edge, the band widths and the tick rate are all sized against occupant speed and all of them break at the same time.

**Refutation attempt failed because:** ARITHMETIC: verified independently, all correct. 1 AU/15 m/s = 9.973e9 s = 316.04 yr; 1e13 m/15 = 6.667e11 s = 21,125 yr; 4.0e16 m/15 = 2.667e15 s = 84.50 Myr; 4.0e16/0.1c = 42.28 yr; and at FULL c, 4.0e16 m is still 4.23 yr. The core observation survives.

BUT THREE OF THE FOUR LOAD-BEARING CLAIMS ARE REFUTED BY READING:

(1) "no speed the code can express" — FALSE. `move_speed_mps` is a plain unclamped f64 on StubConfig (crates/sim/src/stub.rs:82), parsed from VD_SPEED via `env.parse::<f64>` with no bound (crates/bins/src/bin/shard.rs:85), and consumed at crates/sim/src/stub.rs:2536-2538 as `orient * axes * (move_speed_mps * tick_dt_s * time_multiplier)` — no clamp, no validation, plus a SECOND unclamped multiplier (time_multiplier). I found no boot assert cross-checking speed against the band. The two cited constants (DEV.move_speed 15.0, VD_SPEED "2.0") are dev-fixture values, not machinery; they prove nothing about capability.

(2) "every AoI/band/tick guarantee is stated against 2-15 m/s" — FALSE, AND BACKWARDS. Those guarantees take v as a PARAMETER and scale with EXTENT: ContainmentBand::for_containment_velocity_safe sets outset = max(outset_min, v*dt*(K_SAFETY+extra) - inset) (crates/core/src/geometry.rs:741-755); AoiConfig::for_velocity_safe sets tear_down = max(base*factor, spin_up + v*dt*(K+extra)) (:816-840); override_containment_band sizes edges as 10%/20% FRACTIONS of the region's own extent, explicitly scale-invariant (crates/bins/src/lib.rs:2763-2788). My arithmetic: spin_up = extent * cot(theta/2) = extent * 14.3007. Today, walk planet SOI 10 m -> 143 m, at 15 m/s = 9.53 s of boot lead. Canonical planet SOI 9e8 m -> 1.287e10 m, AT LIGHT SPEED = 42.93 s of lead — 4.5x MORE than today. Per-tick step at c with dt=0.02 s = 5.996e6 m = 0.67% of a planet SOI radius, so containment sampling is three orders from tunnelling. The lifecycle machinery scales; it is not what breaks. (The one real adjacent gap is DIFFERENT from what the finding says: the seed-forest band is built at v_rel=0.0, dt=1.0 — BandConfig::build, crates/core/src/worldgen.rs:1092-1101 — a fixed 1 m/2 m band whose width_safe_for fails above 3.0/(0.02*2) = 75 m/s at 50 Hz. That is a fixed-metre-band defect, unrelated to warp, and it costs nothing at true-scale extents because acquire only needs one sample at d <= -1 m inside a 9e8 m sphere.)

(3) "hidden by the 180-metre world" — FALSE. Warp is the most heavily DESIGNED unbuilt feature in the repo: docs/design/PLAN.md:180 is a full P10 row with acceptance criteria; PLAN.md:88 puts AWAIT_PROVISION in the transfer FSM for warp; identity_persistence.md:145-148,344,360 types SystemWarp and explicitly stages it last; connection_plane.md:190,211,316 designs the warp bootstrap, the Failed->retry->abort path and invariant B2; crates/wire/src/channels.rs:94,132,140,151,159 already reserves the warp re-anchor and TransferCosmetic{warp_progress} lanes; crates/core/src/pose.rs:365 names the SOI/warp tier crossing; CLAUDE.md's roadmap lists P10 warp; task #183 is open. Nothing was hidden — and I confirmed no WarpTo/DevTeleport exists in code (grep over crates/ finds none; crates/devproto/src/dispatch.rs:28 names them as future privileged commands), so the "unbuilt" half is accurate, just not a discovery.

WHAT SURVIVES: the single irreducible line — even at c, 4.0e16 m is 4.23 years, and 1 AU at any sublight walk speed is centuries. That is arithmetic about the world, not a property of the code, and no config change touches it. Its one decision-relevant consequence, which the finding buried: the true-scale ruling promotes warp from "P10, scheduled last, after terrain/physics/blocks/checkpoints/ships/signals" to "a precondition for the world being traversable at all". That reorders the roadmap.

SEVERITY: not a defect. No code path is wrong, no guarantee is violated, and the cited evidence is two dev-fixture numbers. Its actionable content is a roadmap-ordering note on an already-designed, already-ledgered, already-tasked feature, with a mechanism attached that I showed is wrong. Downgrade major -> minor.

SHARPEST CORRECT STATEMENT: at true scale no propulsion connects anything — 4.23 ly is 4.23 years even at light speed — so warp stops being P10-last and becomes a precondition for traversability; but the band/AoI machinery it is blamed for is velocity-parameterised and extent-proportional and in fact gets MORE margin at true scale (42.9 s of shard-boot lead at c vs 9.53 s at 15 m/s today), so nothing in it breaks.

---

## [blocker] The first-person camera's ORIENTATION is recovered from the f32 difference of two positions one metre apart. It breaks below planetary scale, and it breaks as a SHAKE, not a bias.

**At true scale:** The render origin is the player's star (see next item), so |eye| is the player's distance from their star: 1.496e11 m at 1 AU. `target.as_vec3() - translation` is exactly zero, `try_into::<Dir3>` fails, and Bevy silently substitutes world -Z. Mouse-look does nothing, the camera stares along a fixed world axis, and no error is logged anywhere. This already fails at 100 km from the origin and is unusable at 1000 km — 8 orders of magnitude below one AU. It is the same failure CLASS as the 'planet edges jitter' shake already chased (task #176), one level up.

**Evidence:** crates/client-render/src/lib.rs:662-665 — `let eye = camera.cam.eye(own_pos); let target = eye + camera.cam.forward(); Transform::from_translation(eye.as_vec3()).looking_at(target.as_vec3(), ...)`. `forward()` is a UNIT vector, so target-eye is a 1 m lever on an f32 magnitude |eye|. Bevy then does `look_at(target,up) -> look_to(target - self.translation, up)` (bevy_transform-0.18.1/src/components/transform.rs:459-460) and `look_to` does `-direction.try_into().unwrap_or(Dir3::NEG_Z)` (transform.rs:472) — a ZERO direction SILENTLY becomes 'look down world -Z'. MEASURED (scratchpad/scale2.rs, scratchpad/jitter.rs), aim error vs |eye|, in 1280x720 / 45deg-vfov pixels (1 px = 1.0908e-3 rad): |eye|=1e5 m -> 0.3 px; 1e6 -> 24 px; 4.19e6 -> 151 px; 8.389e6 (=2^23, where f32 ULP first reaches 1 m) -> 248 px; >=1e8 m -> target-eye is EXACTLY the zero vector, camera locks to -Z. SECOND MEASUREMENT (walking 0.3 m/tick with the mouse perfectly still): peak-to-peak swing 0.36 px at 1e4 m, 2.6 px at 1e5, 38 px at 1e6, 134 px at Earth's radius 6.371e6, 234 px at 1e7 — and only 3-4 distinct camera directions across 200 ticks. The view SNAPS between a handful of orientations as you walk.

**Fix shape:** Never derive the camera rotation from a position difference. `FollowCamera` already holds yaw/pitch and `forward()` returns an exact f64 unit vector — build the quaternion directly (`Transform { translation: eye_reduced.as_vec3(), rotation: Quat::from_f64_basis(right, up, -forward), .. }`) so the orientation never passes through a large-magnitude subtraction. Add a tripwire test asserting the drawn forward matches `cam.forward()` to <0.1 px at |eye| = 1e12 m.

**Refutation attempt failed because:** I could not refute it. I re-derived every number independently and mine are WORSE than theirs at every magnitude but one.

MECHANISM — verified link by link, not taken on trust:
- crates/client-render/src/lib.rs:662-665 is exactly as quoted. `camera.cam.forward()` (crates/client-harness/src/camera.rs:53 -> kinematics::forward_in_frame) returns a UNIT DVec3, so the lever really is 1 m.
- It IS the played camera. `sync_world` is registered in the windowed `run()` Update set (lib.rs:300). The only other writer of the FollowCam transform, `frame_scene_camera` (lib.rs:741), is registered ONLY in the headless `run_capture` (lib.rs:973). Nothing overrides line 665 in the interactive client.
- bevy_transform-0.18.1/src/components/transform.rs:459-460 `look_at -> look_to(target - self.translation, up)`; :472 `-direction.try_into().unwrap_or(Dir3::NEG_Z)`. bevy_math-0.18.1/src/direction.rs:544-549 -> :403 `Dir3::new` errors on a zero vector. So back = -NEG_Z = +Z, right = Y x Z = X, rotation = IDENTITY, camera looks down world -Z. Silent: no log, no panic.
- |eye| really is the distance to the player's STAR. gateway.rs:1836-1847 `render_pin` -> worldgen.rs:943 `pin_realm_of` -> :930 `is_system_level` picks the deepest System ancestor; client/src/view.rs:248 `world_pos` subtracts pin_abs and returns full metres with no clamp.

MY MEASUREMENT (3000 random eye directions x look directions, 45 deg vfov / 720 px = 1.0908e-3 rad/px — their px scale is right):
|eye|=1e4 -> mean 0.24 px, max 0.88; 1e5 -> mean 2.31, max 7.41; 1e6 -> mean 20.95, max 67.8; 6.371e6 -> mean 148.0, max 572; 1e7 -> mean 239, max 1236. Their 1e6 / 6.371e6 / 1e7 figures (24 / 151 / 234) match my MEANS within 3%. Their 1e5 figure of 0.3 px is the one number that is wrong — it is ~8x too small; the true mean is 2.3 px. That error is in their own disfavour.

THEIR ONE REAL OVERSTATEMENT: ">=1e8 m -> target-eye is EXACTLY the zero vector". I measure the zero-direction rate as 47% at 1e8, 89% at 1e9, 99.8% at 1 AU, 100% at >=1e13. So between 1e8 and 1e11 it is not a clean lock — the camera FLIPS between a coarse quantised direction and hard world -Z depending on which way you look. That is uglier than the lock they describe, not milder. At 1 AU (1.496e11) it is 99.8% zero: effectively the permanent -Z lock they claim.

WALK TEST re-run with a GENERIC yaw (their axis-aligned setup would have shown nothing — with forward = -Z and a walk along +x the quantisation is constant and p2p is exactly 0): 0.3 m/tick, mouse still, 200 ticks. p2p 0.16 px at 1e3, 1.33 at 1e4, 8.21 at 1e5, 132 at 1e6, 476 at Earth's radius 6.371e6, 928 at 1e7 — with only 6-8 DISTINCT camera orientations across the 200 ticks (2 at 1e9, 0 at 1 AU: all 200 ticks zero). Their p2p numbers are 3-4x conservative; the "snaps between a handful of orientations" characterisation is confirmed.

WHY I STILL DEMOTE "blocker" TO "major" — I am not disputing the physics, only the engineering weight:
1. The fix is ONE LINE with zero design dependency. `Transform::looking_to` exists (transform.rs:197) and `forward()` is ALREADY a unit f64 vector, so `.looking_to(camera.cam.forward().as_vec3(), up)` has no cancellation to lose and is exact at any |eye|. No wire change, no server change, no architecture. It gates no other work — unlike the frame/tier map, which everything downstream of true scale depends on.
2. It is latent today: at the 180 m world I measure max 0.01 px. Nothing is broken now.

WHY IT IS NOT MERELY "minor" EITHER — and this is the part their write-up UNDERSELLS: it is NOT subsumed by the known "client render space cannot hold 1 AU" item. Even after the obvious re-pin from the star to the PLANET centre, a player standing on an Earth-sized planet is 6.371e6 m from that origin, and I measure 476 px peak-to-peak while walking with the mouse perfectly still. The defect only disappears if the render origin lands within roughly 1 km of the player — which no design in this repo proposes. So it survives the fix everyone assumes will cover it, and it must be fixed on its own. Nothing in docs/design/DEFERRED.md ledgers it.

SHARPEST STATEMENT OF THE CONSEQUENCE: the first-person camera recovers its aim from an f32 subtraction of two points 1 m apart, so once the player is more than ~10 km from the render origin the view snaps between a handful of discrete orientations as you walk, and past ~1e8 m mouse-look stops working entirely — the camera silently falls back to staring down world -Z, with no error anywhere.

---

## [blocker] The server-told render origin is the player's STAR SYSTEM, and every stage after `world_pos` is f32. At one AU nothing in front of the camera projects at all.

**At true scale:** Pinned at the star, a player standing on a planet at 1 AU has R/d_near = 1.496e11 / 1 = 1.5e11, against a budget of 1.83e4 — over by 8.2 million times. The ground under their feet, their own avatar and every other player collapse onto a 16 km lattice and land behind the near plane. The canonical system SOI is 1e13 m, so the overshoot inside ONE star system reaches 5.5e8x. Nothing coherent renders anywhere except within ~18 km of the star itself.

**Evidence:** `pin_realm_of` (crates/core/src/worldgen.rs:930-951) selects the deepest `RealmId::System(_)`; `render_pin` (crates/connection-plane/src/gateway.rs:1835-1848) ships that realm's absolute as `pin_abs`; the client carries it (crates/client/src/view.rs:234-236) and subtracts it in exact f64 (view.rs:248-250, `LatticePos::delta_m`). The f64 subtraction is correct. What follows is not: `world.as_vec3()` into `Transform.translation` (client-render/src/lib.rs:627, 640, 664), `center.x as f32` (client/src/realm_scene.rs:604), and Bevy's `clip_from_world: mat4x4<f32>` (bevy_render-0.18.1/src/view/view.wgsl:17) x `world_from_local: mat4x4<f32>` (bevy_pbr-0.18.1/src/render/mesh_functions.wgsl:61). MEASURED f32 ULP: 16384 m at 1 AU (1.496e11), 1.049e6 m at the canonical system SOI (1e13). MEASURED drawn-pixel error, f32 pipeline vs f64 truth, with a PERFECT camera basis (scratchpad/scale2.rs): at |world| = 1.496e11 a point 1 m ahead and a point 100 m ahead both get w<=0 (projected BEHIND the eye, gone); 1e4 m ahead is off by 2398 px; 1e6 m ahead by 9.1 px. At |world| = 1e13 even a point 1e6 m ahead is off by 443 px. DERIVED RULE (measured, section D): sub-pixel requires |render origin| < d_near x 2^24 x 1.0908e-3 = 18,300 x d_near. d_near = 1 m -> 18.3 km. d_near = 1 km -> 18,300 km. d_near = 1000 km -> 0.12 AU.

**Fix shape:** The render origin must track the PLAYER, not the star. Ship a per-tick origin (the player's own quantized FINE cell, or the cell rounded to a coarse power-of-two so it changes rarely and predictably) on the reliable lane, and let the client re-pin without a visible jump — camera and content already subtract the same origin (view.rs:243-246), so a re-pin is invisible by construction. Keep the drawn magnitude bounded by an explicit budget: assert `|world_pos| < 18300 * near_plane` in a client tripwire. This is a change to `render_pin`, not to the client's subtraction.

**Refutation attempt failed because:** NOT REFUTED — verified by reading and by independent arithmetic; the claim understates the failure in two places, though it misdiagnoses the cause.

CONFIRMED BY READING: pin_realm_of (crates/core/src/worldgen.rs:943-951) selects the deepest RealmId::System(_); a planet is RealmId::Planet, so a player on the ground is pinned at the star. render_pin (crates/connection-plane/src/gateway.rs:1835-1848) ships that realm's tick-0 absolute as pin_abs; the client subtracts it in exact f64 lattice arithmetic (crates/client/src/view.rs:248-250). Everything after is f32: world.as_vec3() at client-render/src/lib.rs:626, 640, 664, 759 and center.x as f32 at client/src/realm_scene.rs:604. Bevy 0.18.1 has no camera-relative path — world_from_local: mat4x4<f32> (bevy_pbr-0.18.1/src/render/mesh_functions.wgsl:54) and clip_from_world: Mat4 = projection * view_from_world in f32 (bevy_render-0.18.1/src/view/mod.rs:541, 979). Information is destroyed at the as_vec3() cast, BEFORE any matrix, so the projection details are irrelevant to the conclusion. No client rescale exists: au_to_render_m has zero consumers outside worldgen.rs. CANONICAL_PLANET_OFFSET_M = 1.496e11 (worldgen.rs:986), CANONICAL_SYSTEM_SOI_R_M = 1.0e13 (:984).

MY OWN MEASUREMENTS (numpy float32, independent of theirs): f32 ULP = 1.53e-5 m at 180 m; 0.5 m at 6.371e6 m; 16384 m at 1.496e11; 1.049e6 m at 1e13 — their two ULPs reproduce exactly. Collapse test f32(R+d)-f32(R): at 1 AU a point 1 m ahead and 100 m ahead both give EXACTLY 0.0 m; at 1e13 even 1e4 m ahead gives 0.0. Their "w <= 0" is understated — the world-space delta is identically zero, so w == 0 exactly under Bevy's infinite reverse-z (w = -z_view). Their derived budget reproduces exactly: 2^24 * (45 deg / 720 px) = 18301 * d_near; 1 AU over by 8.17e6x, 1e13 over by 5.46e8x.

TWO THINGS THE CLAIM MISSED, BOTH SHARPER:
(1) client-render/src/lib.rs:664-665 does Transform::from_translation(eye.as_vec3()).looking_at(target.as_vec3(), ...) where target = eye + forward and forward is a UNIT vector. Measured: f32(R+1) == f32(R) from |R| >= 2^24 = 1.678e7 m (16 777 km, ~2.6 Earth radii). Above that the direction is Vec3::ZERO and Transform::look_to (bevy_transform-0.18.1/src/components/transform.rs:471-472) silently falls back to Dir3::NEG_Z — the camera stops obeying look input entirely. That threshold is 8 900x SMALLER than 1 AU: it is crossed by a merely planetary world, not an astronomical one.
(2) The pin is not the cause and re-pinning is not the fix. Pin at the planet instead and |R| = 6.371e6 m -> measured f32 ULP 0.5 m, so a player standing on the ground still jitters on a half-metre lattice. The real defect is WHERE the f64->f32 boundary sits: world space (world.as_vec3()) instead of eye space. Converting camera-relative (subtract the f64 eye before the cast) removes the |R| dependence entirely and needs no server change — camera.cam.eye(own_pos) already returns a DVec3.

SCOPE CHECK: this is independent of the 200 m MAX_RENDERABLE_EXTENT_M realm-box filter — sync_world draws entity dots with no extent filter, so the player's own avatar, every other player and the camera break at true scale even if no realm box is ever drawn.

---

## [blocker] `MAX_RENDERABLE_EXTENT_M = 200.0` silently skips every realm bigger than 200 m. At true scale that is every realm, including the planet you are standing on.

**At true scale:** `RealmScene` comes back empty for every realm except a station. `sync_realm_boxes` spawns nothing, `despawn_reference_scaffold` (client-render/src/lib.rs:349-360) never fires because `boxes.0.is_empty()`, so the windowed client shows the 250 m stub ground plate and eight landmark pillars floating in a star system. The user sees the P1.5 scaffolding, not the world.

**Evidence:** crates/core/src/worldgen.rs:105 defines it; crates/client/src/realm_scene.rs:245 (`from_shapes`) and :285 (`with_delta`) `continue` past any shape whose `finite_extent()` exceeds it — no counter, no log. MEASURED ratios to the 200 m cut: canonical planet SOI 9.0e8 m -> 4.5e6x over; canonical system SOI 1.0e13 -> 5.0e10x; Earth's radius 6.371e6 -> 3.19e4x; canonical galaxy radius 5.0e20 -> 2.5e18x. The ONLY canonical body under the cut is a 50 m station half-extent. There is a second, unread copy of the same number as `ScaleConfig::render_extent_m` (worldgen.rs:1004, set at :1241/:1300) that no filter consults.

**Fix shape:** Delete the constant and make 'is this drawable, and at what radius' a per-realm SERVER-TOLD field on `RealmShape`. Note this is entangled with the AoI rule: today `finite_extent()` is the SPHERE-OF-INFLUENCE radius, not the drawn body radius (9e8 m SOI vs 6.37e6 m for an Earth — a 141x overstatement), so 'what to draw' and 'how big to draw it' must be separated in the same change.

**Refutation attempt failed because:** NOT REFUTED — the mechanism is real and on the shipped path — but two of the prior agent's specifics are wrong, one in the finding's favour and one against calling it a blocker.

CONFIRMED BY READING:
- `MAX_RENDERABLE_EXTENT_M: f64 = 200.0` at crates/core/src/worldgen.rs:105.
- The skip is a bare `continue` with no counter and no log at crates/client/src/realm_scene.rs:245 (`from_shapes`) and :285 (`with_delta`). `finite_extent()` (crates/core/src/geometry.rs:300-305) is `Shell{r} => r`, `Aabb/Obb => half.max_element()`.
- THE PATH IS LIVE, not a fixture path: crates/client/src/net.rs:248 applies `from_shapes` to `ServerControlMsg::RealmRegistry`, and :267 applies `with_delta` to `RealmSceneDelta`. That is the shipped streamed scene.
- NO server-side pre-filter exists. `realm_registry_for_home` (crates/connection-plane/src/gateway.rs:1779-1813) maps every neighbourhood region to a `RealmShape` and ships it; the only filter there is the minor-6 direct-child shrink. Grep for `finite_extent`/`renderable` in gateway.rs returns nothing.
- The second copy is genuinely unread: `ScaleConfig::render_extent_m` (worldgen.rs:1004) is set at :1241 and :1300 and appears nowhere else outside test asserts (:1917, :1929, :2648, :3093, :3095). Confirmed by grep across all crates.
- The scaffold claim checks out: `despawn_reference_scaffold` (crates/client-render/src/lib.rs:349-360) early-returns on `boxes.0.is_empty()`. The scaffold is a 500 m plate (`GROUND_HALF = 250.0` at :71, `Cuboid::new(GROUND_HALF*2.0, 0.2, GROUND_HALF*2.0)` at :381) and exactly 8 pillars (`LANDMARK_COLORS: [Color; 8]` at :74).

MY ARITHMETIC (recomputed, matches theirs on 4 of 5): 9.0e8/200 = 4.5e6x; 1.0e13/200 = 5.0e10x; 6.371e6/200 = 3.186e4x; 5.0e20/200 = 2.5e18x. Also 8.8e26/200 = 4.4e24x, 4.0e16/200 = 2.0e14x, 1.496e11/200 = 7.48e8x.

ERROR 1 — THEIR NUMBER IS WRONG, AND THE FINDING IS STRONGER THAN STATED. There is NO canonical body under the cut. worldgen.rs:988 `CANONICAL_STATION_HALF_M = 5.0e3` (5 km, 25x over) and :990 `CANONICAL_AREA_HALF_M = 1.0e4` (10 km, 50x over) — `generate_walk_forest` (worldgen.rs:663,671) lowers both through `boxed(half)` = `Aabb{half: splat(half)}`, so `finite_extent` = 5000 and 10000. Their "50 m station half-extent" is off by 100x. Every canonical realm is skipped, not "every realm except a station."

ERROR 2 — THE CONSTANT IS NOT INDEPENDENTLY LOAD-BEARING, so "blocker" over-attributes. Delete the filter entirely and the screen is still empty:
(a) `STAR_FAR_PLANE = 6000.0` m (crates/client-render/src/lib.rs:97, applied at :326) clips everything: 1 AU is 2.49e7x the far plane, the canonical system SOI 1.67e9x, even the planet SOI radius 1.5e5x.
(b) A realm box is the SOI SHELL, not a body. `generate_walk_forest` gives Planet A `shell(pl.planet_soi_r_m)` (worldgen.rs:655). Standing on a canonical planet you are 6.371e6 m INSIDE a 9.0e8 m sphere. Drawing it would not show you "the planet you are standing on" — that phrase in the finding is wrong; there is no body geometry anywhere in the client until P4 terrain. The box renderer is the VU "boxes, not meshes" scaffolding by construction.
So this is one of at least three coupled client-render constants, and fixing it alone changes nothing visible. It is a line item in "the render contract must become server-told and scale-aware," not a gate of its own.

WHAT MAKES IT WORSE THAN THEY SAID (silence): the visual gate would stay green. crates/bins/tests/render_boxes_smoke.rs:66 builds its own hand-written `one_box_boundaries()` metre-scale fixture and projects it via `RealmScene::from_boundaries` (:214) — it never reads `UniverseConfig`. Flip the world to true scale and that gate passes while the real client shows nothing.

SEVERITY major, not blocker: it corrupts no server state and gates no backend machinery (the owner's ruling is about the backend); it cannot be fixed in isolation; and it sits strictly behind the real hard stops (the FINE-tier ±0.95 ly saturation at pose.rs, and the 8-degree AoI threshold that would ship an almost-empty neighbourhood anyway). It is real, must be tracked, and must not be described as the thing standing between here and a true-scale world.

---

## [major] The night sky collapses to a single point. 'Stars always visible' fails at 0.07 AU from the render origin.

**At true scale:** Past ~8400 km from the render origin the individual star meshes degenerate; past ~0.07 AU (1e10 m) they merge into a few hundred; at 1 AU all 2800 render at one point, which sits at the camera position and is clipped by the 0.1 m near plane. The sky goes black — a direct violation of the standing 'stars always visible' rule, at a distance ~20x smaller than the nearest planet.

**Evidence:** The starfield is 2800 real meshes on a 2000 m sphere (`STAR_SPHERE_RADIUS = 2000.0`, crates/client-render/src/lib.rs:95; `STAR_COUNT_UNIFORM 1600` + `STAR_COUNT_BAND 1200`, :99-100), parented to a `StarfieldRoot` that is snapped to the camera every frame (`follow_starfield`, :519-530). So each star's f32 GlobalTransform is `camera_position_f32 + dir*2000`. MEASURED distinct f32 positions of the 2800 stars (scratchpad/scale2.rs, camera off-axis on all three components): 2800 at R_cam <= 1e9 m; 270 at 1e10; 26 at 3.436e10; **1 at 1.496e11**. Separately, star SIZE is `STAR_SIZE_MIN 0.7` .. `STAR_SIZE_MAX 2.4` m (:104-105) — one f32 ULP first exceeds 0.7 m at R_cam = 2^23 = 8.389e6 m (8389 km), so the stars stop having a size well before they stop having distinct positions.

**Fix shape:** A starfield must not be world-positioned geometry. Render it from the view DIRECTION only (a skybox / procedural sky shader, or a mesh drawn with an identity translation in a pre-pass with depth writes off) so its coordinates never carry the camera's magnitude. This also removes 2800 entities from the transform propagation and culling loops.

**Refutation attempt failed because:** NOT REFUTED — I reproduced the endpoint independently and it is worse than filed, but the framing and the threshold are both wrong.

WHAT I MEASURED (my own script, scratchpad/star_ulp.py — exact SplitMix64 reproduction of generate_starfield, seed 0x5644535441525300, 2800 stars, camera off-axis at (0.372,0.514,0.773)·R). Distinct f32 GlobalTransform translations:
  R=1e2 m: 2800 | 4.19e6: 2800 | 8.39e6: 2800 | 1.68e7: 2800 | 6.71e7: 2797 | 1e9: 2543 | 1e10: 397 | 3.44e10: 36 | 1.496e11 (1 AU): 1 | 1e13: 1.
Their 1-at-1-AU endpoint reproduces exactly. Their intermediate counts (270 at 1e10, 26 at 3.44e10) differ from mine (397, 36) only because they used a different camera direction — same order, immaterial.

MECHANISM CONFIRMED: Bevy 0.18 has no camera-relative rendering — I grepped bevy_render-0.18.1 for camera_relative/relative_to_camera: nothing; bevy_pbr-0.18.1 mesh_functions.wgsl:54-56 is a plain `world_from_local * vertex_position` on an absolute f32 mat3x4. GlobalTransform propagation gives star_translation = round_f32(cam + dir*2000), one rounding per axis. Confirmed.

CORRECTION 1 — the threshold is off by 115x–1700x, TOO LENIENT. A star is a sphere MESH, not a point. Its vertices are `M*local + T` in f32; once half the camera's ULP exceeds the star radius, every vertex rounds to the same world point → zero-area triangles → no fragments. MEASURED: at R=6.71e7 m only 452 of 2800 stars still have a radius exceeding half the camera's largest-axis ULP; at R=1e9 m, ZERO do. Full blackout lands at R ≈ 8.7e7 m = 87,000 km = 5.8e-4 AU — 115x closer than their "merge at 0.07 AU" and ~1700x closer than their "black at 1 AU" headline. Their own 8389 km note brackets the other side and is ~10x early (ULP=1 m still rasterizes a 0.7 m sphere). Their MECHANISM for the size half is also wrong: Transform scale is stored exactly in the affine's linear part regardless of translation magnitude (parent matrix3 is identity, so scale never rounds); what kills the size is the world-space vertex sum in the shader.

CORRECTION 2 — it is not a starfield defect. The starfield is the ONE piece of scene content that is already camera-relative by construction (follow_starfield snaps the root to the camera every frame, lib.rs:525-534). 100% of its error is inherited from the CAMERA's own absolute f32 translation: `Transform::from_translation(eye.as_vec3())` (lib.rs:664), where eye is metres from the server-told render pin, and pin_realm_of (core/src/worldgen.rs:943, doc :936-942) deliberately pins "the player's OWN star system". At true scale that origin is 1 AU away BY DESIGN. So any fix that re-centres the render origin near the viewer — which the true-scale client needs anyway — restores the sky for free, with zero changes to client-render. The item has no independent remediation; it should be re-filed against pin_realm_of/pin_abs (a SERVER-authored value), not against STAR_SPHERE_RADIUS.

CORRECTION 3 — the sharper consequence from the same measurement: at 1 AU the camera's own f32 translation lands on a 4096/8192/8192 m per-axis grid. At the dev walk speed of 15 m/s the camera does not move AT ALL for 546 seconds, then jumps 8 km. At the 65-AU canonical system SOI (1e13 m) the grid is 262 km and the wait is 9.7 hours. The 0.5 m player dot (DOT_RADIUS, lib.rs:68) degenerates before the 0.7 m stars do. The black sky is the least interesting symptom of this root cause.

CORRECTION 4 — this refutes the removal surface's own H3 line "The f32 cast is not the problem; the 200 m filter and the 6 km far plane are." That dismissal is backwards: the 200 m filter and the far plane are two constants that become server-told values; the f32 render-origin distance is the binding limit and it bites at 87,000 km from the pin. (Aside, adjacent and worth its own check: Bevy's perspective is infinite-reverse-Z — `perspective_infinite_reverse_rh(fov, aspect, near)`, bevy_camera projection.rs:339 — so `far` only feeds frustum culling, not the clip matrix.)

NOT CAUGHT BY ANY GATE: setup_starfield/follow_starfield are Windowed-only (registered in setup_scene and the windowed Update set, lib.rs:305, :339); the headless Capture path has no starfield at all. No automated test can ever observe this.

SEVERITY: major stands (not blocker). It is real, it is permanent rather than conditional at true scale, and it invalidates a prior dismissal — but it is a client-render precondition with a single known fix, it blocks no backend design, and the true-scale server can land and be proven without it.

SHARPEST STATEMENT OF THE CONSEQUENCE: the render origin is the player's own star (pin_realm_of), so at true scale every f32 world position — the camera included — sits ~1 AU out on an 8 km quantization grid; past ~87,000 km from that star the whole scene, sky and player alike, collapses to zero-area geometry and the camera stops moving for 9 minutes at a time.

FILES: /Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/client-render/src/lib.rs (:68 DOT_RADIUS, :95 STAR_SPHERE_RADIUS, :104-105 sizes, :515 star local translation, :525-534 follow_starfield, :664 camera translation), /Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/core/src/worldgen.rs:943 (pin_realm_of), /Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/connection-plane/src/gateway.rs:1835 (render_pin), /Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/client/src/view.rs:248 (world_pos). Measurement script: /private/tmp/claude-501/-Users-maxim-Projects-my-voxeldust--claude-worktrees-new-system/b4812749-1619-48c5-b5be-bb4924a578c8/scratchpad/star_ulp.py

---

## [major] f32 swallows orbital motion: a planet freezes for half a second, then jumps 16 km. The interpolation math itself is scale-clean — the f32 stage is what breaks.

**At true scale:** Planets stutter: frozen for 0.55 s (Earth) to 5 s (Jupiter), then teleporting one f32 quantum. The 120 ms interpolation buffer, the 12-deep track and the shared-cursor fix (slice 6 S4) are all doing their job in f64 and are then discarded by a single `as f32`. This is the same shake the project already fixed once, re-entering through a different door.

**Evidence:** Realm placements are interpolated by the same primitive as entities (`RealmView::realm_pose` -> `EntityTrack::sample`, crates/client/src/realm_view.rs:112-117; crates/client/src/interp.rs:227-273) at the SAME render cursor the entities use (client-render/src/lib.rs:689-691 vs :609-612), which is correct. MEASURED (scratchpad/scale.rs) at 20 Hz: Earth, v = 29,784.8 m/s, moves 1489.2 m per frame; at |pos| = 1.496e11 the f32 ULP is 16384 m, so **11.00 consecutive frames of orbital motion fall inside one f32 step**. Jupiter, v = 13,070 m/s at 7.785e11 m: 653.5 m/frame vs a 65,536 m ULP = **100.3 frames swallowed** (5.0 s frozen, then a 65.5 km jump). Moon about Earth at 3.844e8 m: 51.1 m/frame vs a 32 m ULP = 0.63 frames — marginal, already visible. The SAME numbers in f64: 2.05e-8 frames for Earth (f64 ULP 3.05e-5 m at 1 AU). The interp cell rebase `prev.offset() + (prev.cell() - cell).as_dvec3() * edge` (interp.rs:250) is exact for |Δcell| <= 2^53, i.e. 8.8e12 m per tick at FINE — never approached.

**Fix shape:** Same root as the render-origin item: keep the drawn magnitude small. Once the origin follows the player, a planet 1 AU away draws at 1.496e11 relative to nothing — it draws at whatever its distance from the PLAYER is, and at 1 AU away a 16 km error is 1.1e-7 rad, invisible. No change to interp.rs is needed.

**Refutation attempt failed because:** NOT REFUTED — but the finding's stated consequence is wrong, and the real failure at the same line is worse. Severity corrected major -> blocker.

WHAT I CONFIRMED BY READING
- The path is as described. Realm placements are interpolated in f64 (crates/client/src/realm_view.rs:112-117 -> crates/client/src/interp.rs:227-273), reduced against the server-told pin in exact lattice arithmetic (crates/client/src/view.rs:249 world_pos -> LatticePos::delta_m; crates/client/src/realm_scene.rs:96 draw_center), then cast to single precision. I enumerated every cast: crates/client/src/realm_scene.rs:604 (realm boxes), crates/client-render/src/lib.rs:626 and :640 (entity dots), :664 (first-person camera eye), :759 (capture camera). There is no camera-relative rebase anywhere; Bevy consumes f32 world space.
- The pin is the player's own STAR SYSTEM, nothing closer: crates/core/src/worldgen.rs:943 pin_realm_of returns the deepest System-level ancestor, and Planet/Station/Area/Ship are explicitly not system-level (worldgen.rs:929). crates/connection-plane/src/gateway.rs:1835 render_pin ships that realm's absolute; the client only adopts it (crates/client/src/net.rs:247, :266). So |drawn value| = |position - own star|.

THEIR ARITHMETIC — I reproduced it, and it is right
Earth: 29784.8 m/s -> 1489.2 m per 20 Hz frame; 1.496e11 lies in [2^37,2^38) so the f32 ULP is 2^14 = 16384 m; 16384/1489.2 = 11.00 frames = 0.550 s. Jupiter: 7.785e11 in [2^39,2^40) -> ULP 2^16 = 65536 m; 653.5 m/frame -> 100.28 frames = 5.01 s. Both correct.

WHERE THEY ARE WRONG (three things)
1. THE CLAIMED EFFECT DOES NOT EXIST. f32 is a RELATIVE-precision format: ULP/value <= 2^-23 always. A one-ULP jump at distance d subtends ULP/d <= 1.192e-7 rad = 0.0246 arcsec REGARDLESS of d. Bevy's default 45-degree vertical FOV over 720 px is 225 arcsec/px. Earth's 16 km jump = 1.0e-4 px; Jupiter's 65.5 km jump = 7.7e-5 px. Apparent size shrinks with distance at exactly the rate the ULP grows — they cancel. A distant planet watched from the pin does not stutter; it is sub-pixel by ~4 orders of magnitude. The whole "frozen 0.55 s then teleporting" picture is refuted for the observer the finding implies.
2. THE MOON NUMBER USES AN ORIGIN THE CODE NEVER USES. They took the moon's ULP from |pos| = 3.844e8 (its distance from Earth). The drawn value is its distance from the STAR, ~1.496e11, so its ULP is 16384 m, not 32 m. Their "0.63 frames, marginal, already visible" is arithmetic against a pin pin_realm_of cannot return.
3. IT IS NOT "the same shake re-entering". The shake (#176) was a relative-TIME error (realm boxes read at a different cursor from entities), fixed by slice 6 S4. This is a relative-SPACE error, and framed as they framed it it has no relative component at all.

WHY IT IS STILL REAL — AND A BLOCKER
The harm is not the observer at the origin looking out; it is the observer who is NOT at the origin, which is the normal case, because the pin is the star while the player is out at a planet. f32 world space holds 1 m of absolute precision only within 2^24 = 16 777 km of the origin (0.01 m only within 2^17 = 131 km). 1 AU is 8917x beyond that radius. MEASURED (round-trip through f32, the same rounding as `as f32`) at |draw_center| = 1.496e11:
  - two points 1.0 m apart collapse to the SAME f32 (149599993856.0 for both) — two players standing next to each other draw at one point;
  - a true 10 000 m separation draws as 16 384 m (64% error);
  - a surface point 6 371 000 m off the planet centre draws 6 373 376 m off — and camera, dot and box each cross their rounding boundary on their own schedule, so that error is a moving +/-16 km sawtooth, not a fixed bias.
All of that runs through shipped code today (sync_world's dots, the camera eye, to_render_prims), not future terrain.

CORRECTED CONSEQUENCE, one line: the render origin is the player's own star and f32 world space keeps 1 m only within 16 777 km of it, so at true scale everything a player can actually stand on is drawn on a 16 384 m grid — two players 1 m apart collapse to one point and the ground under the camera snaps by kilometres; it is not a distant-planet stutter (that is 1e-4 px), it is the near field being destroyed.

WHY BLOCKER, NOT MAJOR: without a viewer-local origin the client draws nothing correctly outside a 16 777 km ball around a star centre (0.0001 AU) — unplayable everywhere a player would actually be. It also directly contradicts the removal surface's H3 ("the f32 cast is not the problem; the 200 m filter and the 6 km far plane are"), so a scale rework scoped to H3 alone (raise MAX_RENDERABLE_EXTENT_M, raise STAR_FAR_PLANE) would ship exactly this. The fix is a server-authoritative seam change — pin_realm_of must pin to the deepest containing realm, or the camera must rebase per frame — not a client tweak.

UNMEASURED / caveat: no true-scale world exists to run, so the above is f32 rounding measured directly rather than observed in the client. Settling measurement: a Tier-A unit test on crates/client/src/realm_scene.rs:602 to_render_prims — build two RealmBoxes 1 m apart with draw_center magnitude 1.496e11 and assert their PrimTransform.translation differ (it will fail); and a second asserting a box 10 km from the camera survives the cast within 1 m.

---

## [blocker] The boot fence that would catch a generator placing a realm outside its parent NEVER FIRES on a real forest — every parent/child pair has different frames, so the check declines on all of them. Evidenced by the fact that the current world boots at all.

**At true scale:** A lazy seed generator emitting a real star census will make placement mistakes — that is what a boot fence is for. Today that fence is decorative on production data while being 100%-covered by fixtures that don't resemble production data. At true scale the mistakes it would catch (a moon placed outside its planet, a station outside its system) surface instead as a player who is 'inside' a realm they are physically outside, which is exactly the falsehood the fence's own doc comment says it exists to prevent (geometry.rs:1075-1086).

**Evidence:** `child_fits_in_parent` (crates/core/src/geometry.rs:1088-1097) opens with `if child.frame != parent.frame { return None; }`. `to_regions` (worldgen.rs:218) sets each region's frame to `frame_for_realm(b.realm, b.parent)`, and `frame_for_realm` (crates/core/src/pose.rs:132-146) derives the frame from the realm's OWN id: System(s)->SystemSpace{s}, Planet(p)->PlanetCentered{p}, Station->StationLocal, Area->AreaLocal. Distinct realms therefore always have distinct frames, and `guard_unique_realms` forbids duplicates — so no parent/child pair in ANY generated forest is ever co-framed.
THE MEASUREMENT THAT ALREADY RAN: at walk scale, Area A sits at x=25 with half-extent 3, under Planet A at x=20 with r=10. reach = |25-20| + |(3,3,3)| = 5 + 5.196 = 10.196 m; limit = parent inscribed 10.0 minus its inset 1.0 = 9.0 m. reach > limit, so if the frames matched, `guard_regions_nest` (called at crates/bins/src/bin/shard.rs:260) would return ChildEscapesParent and NO shard would boot. Every gate is green, so the check is demonstrably declining. The unit tests that exercise it pass only because `test_region` (geometry.rs:1504-1515) hardcodes `frame: SystemSpace { system_seed: 0 }` on every region.

**Fix shape:** Decide what `RealmRegion.center` is expressed in — today it is a parent-frame offset while `.frame` names the child's own frame, and those are different things. Once that is one answer, the fit check compares commensurable numbers and fires. Land a test that asserts the REAL forest is judged (e.g. inflate Planet A's radius in `realm_regions_for` and assert `guard_regions_nest` rejects it) — that test must FAIL at HEAD.

**Refutation attempt failed because:** CANNOT REFUTE — I measured it and the claim is correct to six decimal places, but its severity and its implied remedy are both wrong.

WHAT I MEASURED (scratch crate outside the repo, path-dep on vd-core, `cargo run`):
- `guard_regions_nest(&realm_regions_for(0), 64)` = `Ok(())`. The shipped boot fence accepts the shipped world.
- Across all four world configs (walk, visual_scale, visual_demand, canonical) there are 6/7/7/6 parent-child pairs and **0 co-framed pairs**. `child_fits_in_parent` returns `None` on 100% of them. The fence is a measured no-op on every forest the product generates.
- With the frame gate removed, walk's `Area(7)` under `Planet(7)` computes reach=10.196152, limit=9.000000 → rejects. The author's 5 + sqrt(27) = 10.196 and 10.0 − 1.0 = 9.0 are exactly right (`circumscribed_extent` for Aabb is `half.length()`, geometry.rs:250; `inscribed_extent` for Shell is `r`, :260; `band.inset()` = CONTAINMENT_INSET_M = 1.0, worldgen.rs:46).
- Injectivity holds: `frame_for_realm` (pose.rs:133-146) is a one-field lift per RealmId variant, so distinct realms always yield distinct frames; `guard_unique_realms` forbids duplicates and `guard_chains_reach_root` forbids self-parenting. No generated forest can ever contain a co-framed parent/child pair. `test_region` (geometry.rs:1504) hardcodes `SystemSpace{system_seed:0}` on every fixture region, which is why the unit tests exercise a branch production never reaches.

WHAT THE AUTHOR MISSED, IN THEIR FAVOUR: the `canonical()` preset — the one the brief calls the starting point for true scale — already contains exactly the bug the fence claims to catch. Its Area sits at x=1e5 m while its parent Planet sits at x=1.496e11 m (1 AU). Measured reach = 1.4959991e11 m against a limit of 8.99999999e8 m: the Area is **one astronomical unit outside the planet that claims to contain it**, and `guard_regions_nest` returns `Ok(())`. The two presets disagree about whether an Area offset is absolute or planet-relative, and nothing catches it. This is not a hypothetical future generator mistake; it is in the tree today.

WHERE THE AUTHOR IS WRONG — THE REMEDY, NOT THE FINDING:
1. Un-gating the frame check is NOT a fix. `region_center_of` (worldgen.rs:180-185) returns ZERO for every `Orbital` body. Measured: all five visual planets report `child_center=(0,0,0)` and reach = their own SOI radius (4.160705), never their orbital placement (SMA up to ~146 m). A true-scale generator emits movers, so even with the gate deleted the fence would compare 0 against 0 and validate a radius, never a placement. Catching "a moon outside its planet" requires reading `OrbitalElements` (apoapsis + child extent vs parent inscribed − inset) — the check does not exist in any form.
2. Un-gating naively would refuse to boot today's world. And it would be partly right: exact geometry gives Area A's far corner at sqrt(8²+3²+3²) = 9.0554 m from Planet A's centre, which is inside the r=10 sphere but still 5.5 cm past the inset-reduced 9.0 limit. The fence's spherical bound also overstates reach by 1.14 m (10.196 vs 9.055), so it is both over-conservative AND detecting a real 5.5 cm violation.
3. Partial mitigant the author did not mention: `visual_geometry_respects_the_far_plane_and_soi_non_overlap` (worldgen.rs:3088-3097) does assert `outer_sma + planet_soi < system_soi` — the placement check, but only for the visual preset, only as a hand-written unit test over that preset's derived helpers, and with no inset term. Nothing analogous exists for walk or canonical, which is why canonical's 1-AU escape is invisible.

WHY NOT A BLOCKER: it blocks nothing. The true-scale world can be generated, booted and played with this fence inert — that is precisely what happens today. It is a missing safety net that has been advertised as present, not a mechanism that makes true scale impossible or wrong. Contrast the genuine blockers in the same brief: the FINE-tier ±0.95 ly cap silently rewrites positions, the 200 m render filter draws an empty scene, MAX_REGIONS=64 refuses to boot. Those stop the world existing. This one only stops you finding out it is broken.

Severity major, not blocker: a boot fence documented as load-bearing (geometry.rs:1075-1086), 100% covered by fixtures that hardcode a frame no production region has, and measured to decline on every production forest — while the tree's own true-scale preset contains an undetected 1-AU placement error.

---

## [major] `canonical()`'s Area realm is placed 166x outside the planet it declares as its parent — an unreachable orphan region, and the fence that would catch it is the inert one above.

**At true scale:** `container_coord_at` (worldgen.rs:537-561) descends by containment: a point at 1.0e5 m from the star resolves to System A, never enters Planet A, and therefore never tests Planet A's children. Area A becomes a region no position in the universe resolves to — it exists in the registry, gets a shard spun up for it by any lineage-driven demand, and holds nothing. This preset has no production caller today, so the defect is inert — but the brief says a true-scale preset is what we are building, and these are the exact constants it inherits.

**Evidence:** `generate_walk_forest` (crates/core/src/worldgen.rs:670-677) places Area A at `at_x(sa.area_offset_m)` with `parent: Some(PLANET_A)`, and Planet A at `at_x(sa.planet_offset_m)` — both offsets are measured from the SYSTEM origin on the same +X axis. At walk scale that is deliberate: area 25, planet 20 with r=10, so 25 is inside the planet's SOI. At canonical (worldgen.rs:1332-1339) `area_offset_m = CANONICAL_AREA_OFFSET_M = 1.0e5` while `planet_offset_m = CANONICAL_PLANET_OFFSET_M = 1.496e11`. Separation = 1.496e11 - 1.0e5 = 1.49599e11 m against a planet SOI of 9.0e8 m => 166.2x outside.
The Station is fine by luck: CANONICAL_STATION_OFFSET_M = 4.0e8 under a system of SOI 1.0e13. The Area is not.
Because `guard_children_fit_parents` declines on frame mismatch (see the separate finding), boot accepts it.

**Fix shape:** Whatever replaces canonical() must express a child's placement in its PARENT's frame, not the system's, so the numbers stop being a coincidence of walk scale. Then the (repaired) fit fence catches this class at boot.

**Refutation attempt failed because:** The observation is arithmetically correct and I reproduced it independently, but the severity claim ("major for true scale") rests on a premise that is false, and the stated effect contains two errors.

CONFIRMED BY MY OWN READING + ARITHMETIC:
1. Placement. generate_walk_forest (crates/core/src/worldgen.rs:648-678) places Planet A at at_x(sa.planet_offset_m) and Area A at at_x(sa.area_offset_m) with parent Some(PLANET_A). region_center_of (worldgen.rs:180) lowers both to LatticePos::local(offset), and region_signed_distance_resolved (crates/core/src/geometry.rs:1024-1040) compares p.pos.delta_m(region.center) after an identity reframe — so both offsets live in ONE flat space rooted at the Universe origin (Universe/Galaxy/System A are all at the origin, so "system frame" and "universe frame" coincide). Verified against the walk fixture: area 25 lies inside planet span [10,30].
2. Numbers. 1.496e11 - 1.0e5 = 1.495999e11 m separation; planet SOI 9.0e8 m; ratio = 166.222x (their 166.2 is right). Planet A spans [1.487e11, 1.505e11]; Area A box spans [9.0e4, 1.1e5] — disjoint by 1.487e11 m. child_fits_in_parent would compute reach 1.4959991e11 vs limit (9.0e8 - 1.0 inset) = 8.99999999e8, a 166.22x overshoot.
3. The fence really is inert — and MORE broadly than they said. child_fits_in_parent (geometry.rs:1091) returns None on child.frame != parent.frame, and each region's frame is its OWN frame_for_realm(realm, parent). In generate_walk_forest EVERY parent/child pair is cross-frame: Universe SystemSpace{0} / Galaxy SystemSpace{1} / System A SystemSpace{7} / Planet A PlanetCentered{7} / System B SystemSpace{8} / Station A StationLocal{7} / Area A AreaLocal{7,7}. guard_children_fit_parents therefore fires on NO pair at ANY config, not merely this one.

ERRORS IN THE CLAIMED EFFECT:
4. "gets a shard spun up for it by any lineage-driven demand" is impossible on two independent grounds. canonical() sets interest: InterestConfig::inert() (worldgen.rs:1345; spin_up_factor 0.0 => is_live() false), so there is no AoI demand under this preset at all; and Area A is a LEAF, so it is nobody's ancestor and the ancestor-closure/KeepAlive path never names it either.
5. "container_coord_at ... never tests Planet A's children" is true of the LOGIN descent only. The live containment engine is NOT a parent-filtered descent: evaluate_subject (crates/sim/src/stub.rs:3860-3877) scans EVERY region in the shard's set flatly and folds via geometry::container (depth argmax, geometry.rs:993). A Planet-A-hosting shard's neighbourhood is ancestors u direct children = {Universe, Galaxy, System A, Planet A, Area A}, so Area A IS a live candidate there. Their conclusion survives, but by positional accident rather than structure: no subject can be at |x| ~ 1e5 m while owned by Planet A — continuous motion exits the SOI at 1.487e11 m and re-homes to System A (whose neighbourhood excludes Area A), and login/spawn resolves through container_coord_at, which places such a point in System A.

WHY IT IS NOT MAJOR — the load-bearing premise is false:
6. Their only route to severity is "the brief says a true-scale preset is what we are building, and these are the exact constants it inherits." Grep across the whole tree: area_offset_m / station_offset_m / planet_offset_m / system_b_offset_m are read by EXACTLY ONE function — generate_walk_forest (worldgen.rs:656, 663, 670, 677). Nothing else reads them. generate_walk_forest is the hardcoded 7-body demo topology (2 systems, 1 planet, 1 station, 1 area) that the one-world ruling deletes; a true-scale generator places children from the parent's seed, the shape generate_system_forest / Placement::Orbital (worldgen.rs:449-490) already has. These are demo FIXTURE placements, not a scale ladder a true-scale preset inherits. The constants it does inherit are the radii (CANONICAL_SYSTEM_SOI_R_M / CANONICAL_PLANET_SOI_R_M / CANONICAL_GALAXY_R_M), and those are already covered by the separate +/-0.95 ly FINE-lattice finding.
7. Corroboration that the canonical satellite block is placeholder junk rather than a ladder with one bad entry: CANONICAL_STATION_OFFSET_M = 4.0e8 puts Station A INSIDE the solar photosphere (R_sun = 6.96e8 m). The Station is not "fine by luck"; it is equally illustrative and only escapes geometric conflict because no star body exists as a region at canonical (n_planets: 0).
8. canonical() production callers: ZERO — verified by grep over crates/ and tests/. The only non-test reference is seed_derived (worldgen.rs:1470), itself with no production caller. No test calls guard_regions_nest on it either (worldgen.rs:2302 iterates the forest but only asserts AoI inertness).

NET: the finding is factually real (arithmetic right, orphan real, fence genuinely inert) but its realized effect is nil, its stated mechanism of harm is wrong, and its route to true-scale relevance does not exist. It belongs in the removal plan as a one-line caution — "do not promote canonical(); write the generator" — not as a work item.

SHARPEST STATEMENT OF THE CONSEQUENCE: in a preset nothing selects, built by a generator the ruling deletes, from four fixture literals only that generator reads, the Area sits 166x outside its declared parent and the boot fence that should catch it is structurally inert for every pair in that forest — so the only real cost is that anyone who "goes true-scale" by swapping walk_scale() for canonical() instead of writing the generator ships an unreachable orphan realm silently.

---

## [major] The render contract's own invariant test encodes 'the galaxy must be smaller than the render cull extent' — structurally unsatisfiable at true scale by 9 orders of magnitude. At canonical the client would draw nothing at all.

**At true scale:** The design intent behind that invariant chain — draw the Galaxy as a containing box so an entity in the between-systems gap is visibly still inside a realm and never orphaned (worldgen.rs:56-60) — cannot survive true scale: the between-systems space is 4 light-years across and is not a box you draw. Whatever replaces it is a new render contract, not a constant swap.

**Evidence:** `visual_geometry_respects_the_far_plane_and_soi_non_overlap` (crates/core/src/worldgen.rs:3087-3108) asserts, in order: `system_soi_r_m < galaxy_r_m`, `galaxy_r_m < render_extent_m`, `planet_soi < system_soi`, `render_extent_m >= system_soi_r_m`.
At canonical: galaxy_r_m = 5.0e20, render_extent_m = 1.5e11 => the second assert is false by 3.3e9x. The fourth: 1.5e11 >= 1.0e13 is also false.
The live filter is `MAX_RENDERABLE_EXTENT_M = 200.0` (worldgen.rs:105), applied at crates/client/src/realm_scene.rs:245 (`from_shapes`) and :285 (`with_delta`): any realm with `finite_extent() > 200` is silently `continue`d. Canonical's SMALLEST body is the planet SOI at 9.0e8 m — 4.5 million times the filter. Every realm is skipped; RealmScene is empty; `fit_camera_to_scene` (crates/client-harness/src/camera.rs:184) returns None and the capture falls back to the follow camera. Separately, the GPU far plane is fixed at `STAR_FAR_PLANE = 6000.0` m (crates/client-render/src/lib.rs:97, applied at :326) and `frame_scene_camera` (:741-761) only sets the camera TRANSFORM, never the projection.

**Fix shape:** Make the cull extent and the far plane server-told per realm rather than constants in vd-core / client-render, and replace the 'galaxy is a drawn containing box' invariant with whatever the owner's lattice-of-cell-realms model actually implies. Until that lands, no true-scale render gate can be written at all.

**Refutation attempt failed because:** NOT REFUTED IN SUBSTANCE, but the finding as written is wrong in its headline and in two of its three evidence numbers, and its severity is overstated.

=== WHAT I CONFIRMED BY READING ===

The live cull is real. `MAX_RENDERABLE_EXTENT_M: f64 = 200.0` is a hard `pub const` in vd-core (crates/core/src/worldgen.rs:105), imported by the client (realm_scene.rs:22) and applied at exactly two sites: `from_shapes` (realm_scene.rs:245) and `with_delta` (realm_scene.rs:285), both `if ... > MAX_RENDERABLE_EXTENT_M { continue; }`. `from_regions` (realm_scene.rs:202-215) lowers RealmRegion→RealmShape and delegates to `from_shapes`, so the boot-file path and the streamed RealmRegistry path cull identically. `finite_extent()` (geometry.rs:300-305) is `r` for a Shell and `half.max_element()` for a box. At any true-astronomical geometry every body exceeds 200 m, so RealmScene empties. `scene_bounds` returns None when no box was seen, so `fit_camera_to_scene` (client-harness/src/camera.rs:170-200) returns None and `frame_scene_camera` (client-render/src/lib.rs:741-761) hits its `else { return; }` and keeps the follow camera. `frame_scene_camera` sets only `Transform`, never `Projection`. Nothing server-side applies the cull — the server ships all shapes including the 1e9 Universe. So the mechanism is genuine and it is not already handled: a constant swap does not rescue it either, because at 1e13 m box extents the f32 vertex/depth pipeline is unusable and the far plane is a hardcoded f32 metre.

=== WHERE THE FINDING IS WRONG ===

(1) THE HEADLINE IS A CATEGORY ERROR. `visual_geometry_respects_the_far_plane_and_soi_non_overlap` (worldgen.rs:3086-3108) opens with `let config = UniverseConfig::visual_scale();`. It never touches `canonical()`. It is not "the render contract's own invariant test" — it is a preset-consistency pin on the visual preset, and it is deleted along with that preset by the very ruling in question. Nothing "at true scale" is required to satisfy it, so "structurally unsatisfiable at true scale by 9 orders of magnitude" is vacuous. Worse, the field it asserts on — `ScaleConfig::render_extent_m` — has ZERO non-test readers. Exhaustive grep across the tree returns exactly: the field definition (worldgen.rs:1004), two setters (:1241 walk, :1300 canonical), and four asserts all inside `mod tests` (:1917, :1929, :3093, :3095). The live cull is the unrelated `MAX_RENDERABLE_EXTENT_M` const, which no test ties to canonical at all. The finding leads with a test-only relation between a test-only field and calls it the contract.

(2) THE "SMALLEST BODY" ARITHMETIC IS WRONG BY FIVE ORDERS. Feed `canonical()` to `generate_walk_forest` (worldgen.rs:618-679 — the generator the gateway actually uses) and the bodies are Universe 8.8e26, Galaxy 5.0e20, System A/B 1.0e13, Planet A 9.0e8, Station A half = `CANONICAL_STATION_HALF_M` = 5.0e3 (worldgen.rs:988), Area A half = `CANONICAL_AREA_HALF_M` = 1.0e4 (:989). The smallest is the STATION at 5 km — 5.0e3/200 = 25x the filter, not 4.5 million x. Feed it to `generate_system_forest` instead and `canonical()` sets `n_planets = 0` (:1328), so there is no planet at all and the smallest body is the System SOI at 1.0e13 = 5e10x. The 9.0e8 planet SOI they cite is the wrong body under either reading. Their conclusion (everything culls) survives; their evidence does not.

(3) "THE CLIENT WOULD DRAW NOTHING AT ALL" IS FALSE. `sync_world` (client-render/src/lib.rs:600-645) spawns and moves dots from `snap.rendered(now_s)` and `snap.world_pos(pose)` — the snapshot lane, entirely independent of RealmScene. The starfield is a separate always-on entity set. And `despawn_reference_scaffold` (lib.rs:349-360) begins `if boxes.0.is_empty() { return; }`, so with every realm culled the P1.5 ground plate (GROUND_HALF 250 m) and the eight landmark pillars are NOT despawned — they stay. The actual failure signature is a plausible-looking metre-scale stub world with dots walking around in it, which is a worse (silent) symptom than blank, but it is not what they claimed.

(4) THE FAR PLANE CITATION IS INCOMPLETE. `STAR_FAR_PLANE = 6000.0` (lib.rs:97) is applied only at lib.rs:325-327, on the WINDOWED FollowCam. `setup_capture` (lib.rs:1006-1016) — the headless agent-eyes camera — spawns `Camera3d::default()` with no `Projection` component at all, i.e. Bevy's default 1000 m far. Only one `PerspectiveProjection` exists in the whole crate.

=== SEVERITY ===

Real, but minor as a finding against the one-world work, for four reasons. (a) It is entirely CLIENT-side; the ruling is explicitly "THE BACKEND SERVER". No server state, wire bytes, persistence, containment, or transfer is affected — the server composes correct absolutes and ships them; only the box-drawing layer discards them. (b) The layer that breaks is self-declared placeholder: worldgen.rs:50 "P3 WALK-SCALE geometry (placeholders)", client-render "Pure render scaffolding — replaced wholesale by real terrain at P4, never extended", and the VU arc is explicitly "boxes not meshes". The finding reduces to "the placeholder renderer is a placeholder", and its replacement is already P4 scope. (c) It fails LOUD on the first run — an obviously wrong picture — not silently like the FINE-lattice saturation (pose.rs:312-319 saturating_add rewriting a 4.23 ly position to 0.95 ly with no error), which is the genuinely blocker-class item in the same surface. (d) It is not a precondition and does not constrain the backend design; even the finding's own ordering places it at step 4, after the tier map, the cell anchor, and the generator.

The one genuinely load-bearing residue the finding did NOT make its claim: `MAX_RENDERABLE_EXTENT_M` living in vd-core and being applied client-side is a standing-law violation on its own terms — under the locked streaming contract the server composes and the client never decides what is drawable. That is a one-const relocation to a server-told per-realm field, and it should be ledgered as such rather than as a render-contract redesign.

---

## [major] (b) The three GPU pixel gates survive a world change UNTOUCHED — because they fly through authored playground boxes, not the real world. That is good news for the port and bad news for what they prove.

**At true scale:** Deleting the metre-scale worlds costs these three gates nothing — they keep running and keep passing. The problem is that the ONLY pixel evidence in the whole suite is against hand-built 12-to-60-metre boxes, so the moment the real world is 1e13 m the pixel proof and the shipped world share no code path at all. render_boxes_smoke would still be green with a client that cannot draw a single real realm.

**Evidence:** `render_boxes_smoke` (crates/bins/tests/render_boxes_smoke.rs:62-64) builds its own scene: one box, `BOX_CENTER = (500,0,0)`, `BOX_HALF = (60,60,60)`, `RealmId::Station(4242)` — written to a boxes.json and loaded via `--realm-boxes`. `render_crossing_smoke` uses `vd_bins::crossing_playground` (crates/bins/src/lib.rs:1642-1646): `BOX_A_CENTER = 0`, `BOX_B_CENTER = 50`, `BOX_HALF = 12`, plus an authored `RealmBoundary` shell (`TRIGGER_R_SOI = 10.0`) injected via VD_DEVCLUSTER_BOUNDARIES. `render_smoke` uses NO scene at all — its content floor is met by the P1.5 reference scaffold (ground plate `GROUND_HALF = 250.0`, landmark pillars at radius 25, client-render/src/lib.rs:71-73), which `despawn_reference_scaffold` (:349) deliberately keeps on the capture path precisely so this gate keeps passing.
None of the three references worldgen, realm_regions_for, or UniverseConfig (verified by grep: worldgen_refs = 0 for all three).

**Fix shape:** Keep them (they are honest machinery proofs of the readback path) but add one that renders the REAL streamed world at the real scale and asserts a body is pixel-visible. That gate cannot be written until the render contract above is server-told — which is the dependency worth surfacing now rather than discovering after the config is deleted.

**Refutation attempt failed because:** The factual base is correct and I verified it independently, including a mechanism the finding did not name. The three pixel gates really are insulated from worldgen at the SCENE level, and the reason is `maybe_announce_realm_registry` (gateway.rs:1855-1863) returning early unless `cfg.armed`, with `armed = demand_armed` (bin/gateway.rs:97). A static `devcluster up` is unarmed, so no RealmRegistry/RealmSceneDelta is ever sent, so the boot `boxes.json` scene is never replaced (the replace is real and wholesale — client/src/net.rs:248). grep confirms 0 worldgen/realm_regions_for/UniverseConfig references in all three test files; render_boxes_smoke.rs:62-64 is BOX_CENTER(500,0,0)/BOX_HALF 60/Station(4242); render_crossing_smoke.rs:287 uses crossing_playground::scene(&DEV); render_smoke.rs passes no --realm-boxes.

BUT the claimed EFFECT is wrong, and that is what drops the severity. "render_boxes_smoke would still be green with a client that cannot draw a single real realm" is true of that one binary and FALSE of the suite. `just gate` (justfile:238) includes `rlm-demand-login`, whose binary boots an ARMED demand cluster at visual_demand scale and launches a real client with NO --realm-boxes (rlm_demand_login.rs:124-146 — the arg list has no scene file), so its entire world arrives over the stream. Two tests there read DevState.realm_boxes, which is `render_snapshot().scene_now(now_s)` (client/src/net.rs:536-537) — the same overlaid scene the renderer draws, built by from_shapes/with_delta, which is exactly where MAX_RENDERABLE_EXTENT_M is applied (realm_scene.rs:246, :286). `a_demand_login_draws_moving_planets_not_a_frozen_scene` asserts >=1 `Planet(` box is DRAWN and its centre moves >1e-3 m over 3 s; `a_parked_ship_keeps_the_planets_orbiting_at_full_speed` re-asserts it over 28 s.

Arithmetic, from constants read (worldgen.rs:120-130; taxonomy.rs:516-517 orbital_axis_au = a0*ratio^n):
outer_axis_au = 0.4*1.7^4 = 3.34084 AU; soi_au = 0.35*0.4*0.7 = 0.098 AU;
au_to_render_m = (150-4)/3.43884 = 42.456 m/AU; planet SOI = 0.098*42.456 = 4.16 m — 48x UNDER the 200 m filter, which is why it is drawn today.
At true scale the extent filter is the FIRST wall: canonical planet SOI 9.0e8 m is 4.5e6x over 200 m; system SOI 1.0e13 m is 5.0e10x; even a bare Earth radius 6.371e6 m is 31 855x over. Every realm is skipped, realm_boxes comes back empty, and that gate fails inside its 30 s budget with the message it was written for ("(a) NO planet drawn").

Second correction: "survive UNTOUCHED" holds for only one of the three. render_crossing_smoke's shard takes the VD_REALM_BOUNDARIES override, which REPLACES the seed forest entirely (bin/shard.rs:229-245) — fully insulated. render_smoke and render_boxes_smoke take the else arm (:246-252): their shards plant the worldgen forest and their logins resolve home through container_coord_at. Their pixels are insulated; their boot and login are not. Whether they still pass at true scale is UNKNOWN (measurement: run `just render-smoke` against a true-scale forest and check the client reaches universe_tick >= 1).

What survives as real: no GPU-readback gate ever renders a worldgen realm, so the specifically pixel-tier risks — STAR_FAR_PLANE = 6000.0 m (client-render/src/lib.rs:97), mesh/material spawn in sync_realm_boxes, depth precision, the f32 cast — stay unproven past 6 km. That is a genuine gap, but it is the SECOND wall (reachable only after the extent filter is fixed) and it is the already-ledgered VU pixel work, not a new discovery. It describes no defect in shipped code and blocks nothing in the port. Hence minor, not major.

Sharpest corrected consequence: no GPU gate ever renders a worldgen realm, so the far-plane/mesh half of "can the client draw the real world" stays unproven — but the suite is not blind, because `just gate`'s rlm-demand-login already fails loud, and first, when the drawn scene contains no real realm.

---

