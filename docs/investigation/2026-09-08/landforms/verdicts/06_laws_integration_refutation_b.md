# Refutation B of 06 — laws, integration and the slice plan

**Date:** 2026-09-08. **Target:** `docs/investigation/2026-09-08/landforms/06_laws_integration.md` (revision 2).
**Lens:** believability, cost and the owner. Would the result look like the reference picture? Would a
geologist or a climatologist accept it? Does it fit the 8 ms budget and the boot? Does it answer what
the owner asked? Is it written so the owner learns the terms?

Every number below is re-derived from the code or from a stated arithmetic. Where I could not measure,
I write UNMEASURED and name the bench.

---

## Blockers

### B-1. The cost model funds the flow routing TWICE for a hundred erosion passes. The rivers then do not fit the ground.

§5 states the build as two stage families:

```
   STREAMING (uplift, isostasy, thermal creep, STREAM POWER):  P = 100 passes
   ORDERED   (priority flood, flow accumulation):              2 sweeps, TOTAL
```

Two things are wrong, and they cannot both be repaired.

1. **Stream power is not a streaming stage.** `E = K · A^m · S^n` reads the RECEIVER cell — the
   downstream neighbour along the flow direction. That is a pointer chase along the drainage tree,
   not a three-row stencil. §5 puts it in the family whose cheapness rests on a stencil staying in
   cache, and the whole 0.06–0.16 s figure comes from that placement.
2. **The routing must re-run.** After 100 passes the surface is not the surface the flow directions
   were built on. Divides migrate, a basin captures its neighbour, a filled hollow drains. If the
   routing runs twice in total, the final field carries a river graph built on the FIRST surface. The
   report's own slice 8c gate — *"no river runs uphill (the stage falls downstream over 100 000
   sampled segments)"* (§6.5) — is then a gate the design cannot pass, because nothing kept the graph
   and the ground in step.

The arithmetic of the fork. If the routing runs once per pass, the ordered work is `100 × 0.25 s =
25 s` at §5's own (already low, see D-5) rate. If it runs every tenth pass, 2.5 s. If it runs twice,
the rivers are stale.

*In the game's words.* A pilot lands in a valley on the home planet. The valley was cut by 100 passes
of water. The river the client draws was routed across the hill that stood there before the first pass.
The water runs up one shoulder and stops in the air.

**What is owed:** state the re-routing period as a stated constant of the recipe (it is also a
determinism input, §3.1 rule 6), and re-price the ordered family by that count. Until then §0 item 2,
§5's total and decision L1 all rest on a model that does not describe the algorithm.

### B-2. The slice order puts the erosion BEFORE the climate, so the erosion is rainfall-blind — and no ice ever cuts anything.

§6.1 fixes: 8b erodes, 8d gives the climate. The physical dependency runs the other way and closes a
loop:

```
   relief  ->  wind and orographic lift  ->  rainfall  ->  discharge  ->  erosion  ->  relief
     ^                                                                                   |
     +-----------------------------------------------------------------------------------+
```

- **Discharge, not area.** In the stream power law `A` is drainage AREA used as a PROXY for discharge.
  The proxy holds inside one climate. Across a rain shadow it fails: a leeward basin of the same area
  carries a small fraction of the windward basin's water. With the climate landing after the erosion,
  a desert basin and a rainforest basin on the home planet are cut to the same depth by the same rule.
  §6.6's own gate (*"a ridge has a wet side and a dry side"*) then measures a paint colour on ground
  that was carved as if both sides were wet. A geomorphologist reads that in one frame.
- **No glacial erosion at all.** The erosion set is uplift, isostasy, thermal creep and stream power
  (§6.4). The reference picture's subject is a SNOW-CAPPED range. Above the snow line on Earth the
  sculptor is ice, not water: cirques, arêtes, U-shaped troughs, hanging valleys. Fluvial incision
  alone gives V-notches all the way to the summit. §6.9 maps *"Snow caps on the ridges only"* to the
  lapse rate at 8d — a colour — and maps *"carved valleys"* to the stream power law. The one landform
  family the reference picture's top third is made of is missing from the plan, and it is missing
  BECAUSE the snow line is not known when the erosion runs.

**What is owed:** either a first climate pass on the un-eroded macro relief (cheap: it is a closed form
over 1.57 M cells), so the erosion can read a precipitation field and a snow line, or an explicit
statement that the arc ships rainfall-blind erosion and no glacial landforms, with the picture the
owner will get. This is a slice-order decision and it belongs in §9 as its own row.

### B-3. Road B pipes floats whose cross-host bit-equality gate the code itself DEFERS into the crate whose whole purpose is proven determinism.

§4.2 recommends road B: the body facts ride the surface tag, snapped, *"exactly what the radius does
today, and the snap's harmlessness is MEASURED"*.

The code says something the report does not quote. `crates/physics/src/taxonomy.rs:21-24`:

> *"Determinism note: the transcendental steps here (`ln`/`sqrt`/`powf` in the luminosity, frost-line
> and Rayleigh closed forms) are the SAME libm class as `celestial.rs` and are evaluated at BOOT /
> seed time … **the cross-host bit-equality gate is SPIKE-6a (step-2/P4)**, the same deferral the
> module docs there carry."*

So every fact road B carries — gravity from a drawn mass, `insolation_rel`, `t_eq_k`, the atmosphere,
the Bond albedo — is produced by an OPENLY UNPROVEN chain. `crates/terrain/src/body.rs:85-87` states
the crate's own invariant against exactly this: *"a body is DRAWN from a seed by
`BodyDefinition::from_seed` and never assembled from numbers computed elsewhere, so no float from an
unfenced crate can enter the recipe as a body."*

Two further facts make it sharp, and neither is in the report:

- **The client recomputes the home body locally, on its own chip.** `crates/bins/src/lib.rs:1047-1052`
  (`world_identity`) calls `home_body` → `home_system_boot` → the taxonomy, and
  `crates/bins/src/bin/client.rs:118` calls it at boot. So the client and the gateway each run the
  unfenced draw. If a snapped fact straddles its grid boundary on one chip, the two MEASURED halves
  differ and the client is refused at login — not a cosmetic drift, a refusal.
- **What was measured is not what is claimed.** `crates/terrain/src/home.rs` (the test at :47-79)
  measures that the RADIUS survives a 1 000-ulp move. The radius's snap unit is one top-rung cell,
  2 048 m of radius — about 6 × 10⁻⁴ relative. A gravity snapped at 1/64 m/s² is 3 × 10⁻³ relative on
  the home planet; a cosine snapped at 1/1024 is 10⁻³. Those are the same family, so the straddle
  probability is tiny — but it is a correctness-by-luck argument, and the report presents it as a
  measurement. And `home.rs` itself shows the cure the project already chose for the radius: it PINS
  the bits (`HOME_PLANET_RADIUS_BITS`) rather than trusting the forest's float.

**What is owed:** say plainly that road B imports SPIKE-6a's deferral, and choose one of: (a) the facts
are stated by the realm and NEVER recomputed by a receiver, with the home body's facts pinned as bits
in `home.rs` the way the radius is; or (b) SPIKE-6a lands before slice 8a. §9 L2 must carry this, and
it does not.

---

## Defects

### D-1. `relief = draw × min(strength bound, accretion bound)` still exceeds the bound. The section's own complaint survives its own cure.

§3.4 correctly finds the defect: today `relief_cap.clamp(200, 12_000) * (HALF + draw_unit)`
(`crates/terrain/src/body.rs:147-149`) multiplies the factor OUTSIDE the clamp, so the drawn relief
(MEASURED 14 305 m) exceeds the cap (12 000 m). It then writes the cure as

> *"`relief = draw × min(strength bound, accretion bound)`, with the draw INSIDE, so a bound is a
> bound."*

and never changes the draw's range. With `draw ∈ [0.5, 1.5)` unchanged, the home planet gives
`1.192 × 13 403 = 15 977 m` against a bound of 13 403 m. The formula moves the multiplication but does
not make a bound a bound.

Either the draw's range becomes `[0, 1]` (or `[0.5, 1.0]`), which is a change to EVERY body's relief
and a one-way door §8 does not list, or the defect is re-imported. §6.3 lands the change with no
statement of the range at all.

### D-2. The macro map cannot make the reference picture. The layer that can is described in one sentence and costed at 80 ns.

The owner's reference is a ~50 km vista. At `E = 9` the home planet's macro cell is 10 280 m, so the
whole frame is **4.9 macro cells wide** — the report says so itself in §3.6's crease argument. Across a
1 280-pixel frame, one macro cell is about 260 pixels. Nothing a player looks at in that frame is
resolved by the erosion.

Yet §0, §6.4 and §6.9 attribute the picture's content to the macro map:

| §6.9 row | Attributed to | What actually draws it at 50 km |
|---|---|---|
| Ridges you can navigate by ("orienters") | *"tectonic uplift, isostasy and erosion on the macro map"*, slice 8b | a 5-sample trend plus 9 octaves of noise |
| Carved valleys with floors and shoulders | *"the stream power law, then the closed-form valley"*, 8b and 8c | the closed-form valley alone |

The closed-form layer is the whole picture, and this report gives it one rule (§3.6: *"a valley is a
shaping FUNCTION applied to the height that already exists"*), one cost row (80 ns per column, sharing
a line with the river query) and no law audit of its own.

The practical consequence is a decision risk the owner will pay. §6.4's picture gate says *"a 50 km
vista at a coarse rung from a ridge. **The owner judges 'orienters' here.**"* — at slice 8b, BEFORE the
closed-form valley lands at 8c. The owner will judge the direction on a picture that shows a gentle
trend under the same noise he already rejected.

**What is owed:** move the "orienters" judgement to 8c, or state that 8b's picture is a trend check and
not the quality gate.

### D-3. The weather FIELD's bytes are never computed, and at the resolution §6.7's own picture needs, one statement is about 400 KB.

§3.14 corrects revision 1's six scalars to *"a FIELD stated ONCE for the realm, as coefficients over
the body's own surface (the same face grid the macro map uses, at a very coarse `M`)"*, and §6.7
promises the picture *"a storm on the horizon over a desert, with clear sky elsewhere in the same
frame"*.

The arithmetic that decides whether those two sentences can both be true is missing. A storm front the
pilot sees on the horizon is a ~50 km feature. On the home planet that needs
`M ≈ n / 50 000 = 5 263 360 / 50 000 ≈ 105`, so `M = 128` and `6 × 128² = 98 304` cells. At even 4
bytes of coefficient per cell that is **393 KB per statement**, restated as the weather moves, against
`SELF_LOOK_BUDGET_BYTES = 1200` for the whole self-look bag (`crates/core/src/look.rs:78`). At *"a very
coarse `M`"* — say `M = 8`, 384 cells — one cell spans 658 km on the home planet, and every storm is
658 km wide.

So the field, as described, either misses the lane's budget by 300× or cannot draw the promised
picture. The lawful third road is a small set of SEEDED PARAMETERS the client evaluates as a procedural
field — which the report gestures at (*"the client evaluates the field per place and paints it. That is
rendering, not deriving"*) but never distinguishes from stored coefficients, and never prices.

U-L8 owes the bytes. It owes them BEFORE §9 L17 asks the owner what *"simulate"* means, because the
answer to L17 changes with whether a field can cross at all.

### D-4. The flow-accumulation area weight uses a LINEAR width where an AREA is needed, and it points at the wrong place on the cube.

§1's last row and §3.1 rule 7(c) state:

> *"the bend makes an angular cell about 6.5 % narrower at a corner than at a face centre (ESTIMATED
> from `crates/seed/src/bend.rs:28-30`: the derivative falls from `k₁ = 0.7854` at the centre to about
> 0.735 at the corner)"*

Re-derived from the code. `direction(face, a, b) = normalize(n + W(a)u + W(b)v)`
(`crates/seed/src/bend.rs:251-259`), `W(a) = a(K1 + a²(K2 + a²K3))`, `K1 = π/4`, `K2 = 0.15`,
`K3 = 1 − K1 − K2 = 0.0646`. With `s² = 1 + W(a)² + W(b)²`:

```
   cell width  along a:  |dr/da| = W'(a)·sqrt(1 + W(b)^2) / s^2
   cell AREA:            |dr/da x dr/db| = W'(a)·W'(b) / s^3
```

| Place on a face | width, relative | AREA, relative |
|---|---|---|
| centre (0, 0) | 1.000 | 1.000 |
| face-edge midpoint (1, 0) | 0.992 | **0.702** |
| cube corner (1, 1) | **0.935** | 0.758 |

Three corrections follow.

1. **6.5 % is the corner's LINEAR width, not its area.** The report says *"Flow accumulation counts
   AREA, so it must weight by it"* and then quotes the linear number. The area weight at a corner is
   24 % below the centre, not 6.5 % — wrong by about 3.7×.
2. **The extreme is on a FACE EDGE, not at a cube corner.** The smallest cell area (70 % of the centre)
   sits at the midpoint of a face edge. Rule 7 frames the whole seam risk on the eight cube corners
   and ruling A5's corner pictures; the largest area distortion runs along the TWELVE face edges —
   which is exactly where a river must cross.
3. **The stated derivation is wrong even though its number is not.** `W'(a) = K1 + a²(3K2 + a²·5K3)`
   RISES from 0.7854 at the centre to **1.5584** at `a = 1`; it does not *"fall … to about 0.735"*.
   0.735 is the metric width, a different quantity. And `bend.rs:28-30` is the `K3`/`INVERSE_STEPS`
   text, not a derivative.

U-L15 owes the measurement, which is right. The stated estimate must be corrected before it is used,
because a 4× wrong area weight bends every river toward the face centres — the exact failure rule 7
exists to prevent.

### D-5. The priority flood is modelled as one random access per cell. A binary heap over 1.57 M cells is not that.

§5: *"ORDERED stages (priority flood, flow accumulation): 2 sweeps, ~1 random access per cell at 80 ns
= 0.25 s."*

A priority flood pushes and pops every cell once through a heap of up to 1 572 864 entries. §5's own
memory table sizes that heap at 8 B per cell = 12.6 MB.

```
   heap depth          log2(1 572 864) = 20.6 levels
   top 18 levels       2^18 x 8 B = 2 MB          - L3, mostly hits
   bottom ~3 levels    ~ 11 MB of the array       - misses
   per pop             ~ 3-5 misses sifting down
   per popped cell     + 8 neighbour reads of a 6.3 MB elevation array, in HEIGHT order,
                         so spatially random: ~ 3 distinct lines, ~ 3 misses
   ------------------------------------------------------------------
   ESTIMATED per cell  6-8 misses x 80 ns  =  0.5-0.65 us
   flood alone         1 572 864 x 0.5-0.65 us  =  0.8-1.0 s
```

Flow accumulation adds a height sort (1.57 M entries) and a sweep that chases receivers, ESTIMATED a
further 0.3–0.5 s. So the ordered family is **ESTIMATED 1.1–1.5 s, not 0.25 s** — 4× to 6× low, before
B-1's re-routing count multiplies it.

The report labels the figure ESTIMATED and owes M-L1, which is honest. But §0 item 2, §5's headline
*"TOTAL ESTIMATED 0.31 – 0.41 s, ONE core"*, §4.1's *"Under a second on one core"* and decision L1's
recommendation all rest on it. A recommendation built on a 5× low estimate is a recommendation the
measurement will overturn.

### D-6. Several bodies cost TIME as well as memory, and the "few hundred kilobytes" moon is an assumption, not a measurement.

§3.10: *"ESTIMATED worst case at `E = 9`: three planets at 18.9 MB plus eleven moons at a few hundred
kilobytes each ≈ 60 MB."*

Under the report's own rule `M = 2^min(E, top)`, a body reaches `M = 512` — the full 1 572 864 cells,
the full 18.9 MB and the full build — as soon as its top rung reaches 9. Re-derived from
`crates/seed/src/ladder.rs:22-23,48-57,76`:

```
   top >= 9   <=>   n > 3968 x 256 = 1 015 808   <=>   radius > 1 015 808 x 2/pi = 646 683 m
```

**Any body over about 647 km of radius costs the full map.** Our Moon is 1 737 km. Ganymede is
2 634 km. `docs/investigation/2026-09-07/slice_02_geometry_seam.md:275` records that the home system
holds 10 planets and 11 moons, and it does NOT record their radii — so *"a few hundred kilobytes each"*
has no source. If three of the eleven moons are Luna-sized, the worst case is `6 × 18.9 = 113 MB`, not
60 MB.

And the time is never added at all. §3.1 rule 2 makes the build SINGLE-THREADED by law, so the builds
queue one behind another. Six large bodies at §5's (low) 0.4 s is 2.4 s of one core; at D-5's corrected
rate, 8–9 s. §5 lists the cost as *"once per body per session"* and never multiplies by the bodies in
the band. U-L13 owes the census; the report should not state a worst case before it has one.

### D-7. `E` is not the same kind of constant as `SHORT_WAVE_M`, and the no-magic-number audit passes it on an analogy that does not hold.

§4.3: *"`E` is a stated RESOLUTION, and §3.4's own rule says a stated resolution is lawful; … this
report's audit calls `SHORT_WAVE_M = 30` lawful for exactly this reason."*

The two are different in kind.

- `SHORT_WAVE_M = 30 m` is METRIC and identical on every body. It states a physical fact: the finest
  wave in the world is 30 m, everywhere.
- `E` is ANGULAR. Its metric consequence is `n / 2^E` metres, which varies with the body:

| Body | macro cell |
|---|---|
| a 50 km moon | 2 454 m |
| a 100 km moon | 2 454 m |
| a 1 000 km moon | 3 068 m |
| the home planet | 10 280 m |
| the largest body the address allows | 131 072 m |

A 53× spread. Drainage density — how close together the streams run — is a METRIC fact set by rainfall
and rock (§1 teaches it as 1 to 5 km on Earth). So under `E` the drainage skeleton is resolved four
times more finely on a 1 000 km moon than on the home planet, and a larger Earth-like planet gets a
COARSER skeleton than a smaller one with the same climate. Two Earth-like planets will not look equally
believable, and the closed-form layer must bridge a four times larger gap on the bigger one.

The report says the metric size varies and *"says so plainly"*, which is good. It does not draw the
consequence, and it does not withdraw the `SHORT_WAVE_M` analogy that carries the lawfulness argument.
`E` is a COST knob traded against fidelity, which is a different lawful category and should be named as
one.

### D-8. The relief's split between the macro map and the surviving octaves is never named — and it decides exactly what the owner is worried about.

Today the octave amplitudes are scaled so their SUM is the relief
(`crates/terrain/src/body.rs:180-184`), which is what makes `relief_bound_m` exact. §3.8 rules that
`relief_bound_m` becomes `macro bound + octave sum` and stays exact. So the octave sum must SHRINK by
whatever the macro map takes.

By how much is stated nowhere. §3.7 says only *"the macro map takes amplitude away from them"*.

That single ratio decides:

- how much relief the player sees at 62 m, which is the owner's stated worry (*"the surface will not be
  interesting enough"*);
- how much the rung-to-rung disagreement shrinks (§3.7's claimed improvement);
- whether the 8c closed-form valley has room to cut, or is cutting into ground the octaves already
  roughened.

It has no decision row in §9, no measurement in §10, no owner in §6. It is the most consequential
number in the direction and it is invisible.

### D-9. No plate tectonics, no bimodal hypsometry — so no continental shelf, no linear range, and an ocean fraction that has to be tuned by hand.

§1 teaches hypsometry well: *"Earth's is two-humped: ocean floor, then continents."* The design then
never produces two humps. Earth's two humps come from TWO CRUST TYPES of different density floating at
different levels — isostasy applied to composition. §1's isostasy row says only *"It bounds a range's
height"*, and §6.4 lands isostasy as a height bound.

Two believability consequences a geologist would name at once.

1. **Coasts.** A sea level poured onto a single-humped, eroded-noise surface gives a fractal coastline
   with no continental shelf, no abyssal plain and no clear land/sea separation. The land/sea split
   also becomes hypersensitive to the water draw — which is precisely why §9 L11 has to set the water
   inventory's range from a TARGET ocean fraction rather than from physics. The report is admirably
   honest about that (*"it is dressed as physics if it is not said plainly"*), but it treats the tuning
   as the answer instead of as the symptom.
2. **Ranges.** §1 describes tectonic uplift as *"a few slow patterns from the seed that say where the
   ground rises."* Slow noise gives BLOBBY highlands. The reference picture shows a LINEAR range with a
   consistent crest line and a consistent aspect — the shape a convergent margin makes. Nothing in the
   direction produces linearity, and §6.9 nevertheless maps *"ridges you can navigate by"* to it.

An "orienter" is a landform a pilot recognises from far away and steers by. A blob is not one. This is
the owner's own worry and it deserves a named mechanism — a plate or margin field, or an explicit
statement that ranges will be blobby in this arc.

### D-10. Seasons are absent, so the owner's word "trajectory" is answered only nominally.

The owner asked for biomes dependent on *"position, spin, trajectory, size and gravity"*.

The report's climate is a STATIC ANNUAL MEAN (§3.14: *"CLIMATE (static, in the recipe) — the long-term
average"*). In an annual mean, eccentricity's whole contribution is the factor `(1 − e²)^(−1/2)`: at
`e = 0.17` that is 1.5 %, and at Earth's `e = 0.017` it is 0.014 %. Worse, it is already inside the
datum the report proposes to ship: `insolation_rel = L / d²`
(`crates/physics/src/taxonomy.rs:931-932`), with `d` the body's own orbit.

So the visible consequences of spin tilt and eccentricity — a snow line that moves, a summer, a winter,
a perihelion heat wave — are all TIME-VARYING. SL10 clause 1 forbids time inside the recipe, and
§3.14's live layer carries only *"today's cloud, rain and wind"* plus a decaying offset. No layer of the
design carries a season.

§4.2's snap table nevertheless justifies eccentricity's snap grid with *"A strongly eccentric orbit is
the strongest seasonal signal a planet can have"* — a true statement about a thing the design cannot
show.

**What is owed:** either name the season as a live field on the same lane as the weather (it is a
closed form on the universe tick, dormant-safe, and it costs the shard nothing), or tell the owner that
"trajectory" moves a body's mean temperature by a fraction of a percent and nothing else.

### D-11. The rain shadow from "the macro map's gradient" is a local stripe. A continental interior dries by depletion along the wind, which is another ordered pass and is not budgeted.

§6.6 lands *"orographic lift and the rain shadow from the macro map's gradient"*, and §3.6 costs the
whole climate evaluation at **20 ns per column** — a closed form.

A gradient term gives the windward face more rain and the leeward face less. It cannot give the
Taklamakan or the Great Basin, because those are dry from CUMULATIVE depletion: air crossing three
ranges in a row has lost its water at the first. That needs an upwind integration over the macro grid —
a march along the wind, cell by cell, carrying a moisture budget. That is:

- another ORDERED pass (so §3.1 rule 2's fixed order and rule 6's stated step count apply to it, and
  neither §3.1 nor §6.6 names it);
- another traversal of 1.57 M cells (so it belongs in §5's cost, and it is absent);
- circular with B-2's loop, because the wind is set by the relief the erosion is still changing.

§6.6's gate — *"a ridge has a wet side and a dry side"* — passes on the weak local form, so the gate
cannot tell the two designs apart. §1's promise, *"Why one side of a range is forest and the other is
desert"*, is the weak form. The owner will read it as the strong one.

### D-12. The boot canary's re-basing narrows what the login gate proves, and the report calls the change strictly stronger.

`crates/terrain/src/tag.rs:30-34` states the gate's meaning: *"the DECLARED half, a constant, and the
MEASURED half, the home body's eight golden chunks **evaluated by this binary on this chip**."*

§3.9 rule 1 keeps the eight chunks but re-bases them: *"they are evaluated at addresses whose macro
dependency is a SINGLE macro cell, **read from a stated pinned value rather than from a built map**."*

After that change the login gate proves the kernels agree on a stated neighbourhood and the chunk
arithmetic agrees on a pinned input. It no longer proves that this binary on this chip BUILDS the same
macro map — the map is 1.57 M cells of accumulated arithmetic, and none of it is in the fold. A host
whose whole-body build drifts logs in successfully and stands its player's boots off the ground, which
is the failure §4.1's own table names.

The kernel canary is a good addition and I do not dispute it. The claim I dispute is *"This is STRONGER
than moving the canary to a smaller body"* stated without naming what the re-basing gives up. §9 L14
puts three options to the owner and describes none of them as narrowing the gate. A fourth option
exists and is not offered: keep the kernel canary AND fold a digest of a stated small SLICE of the real
map (one face row, say), which costs milliseconds and keeps the build itself inside the gate.

---

## Weaknesses

### W-1. HR5: nothing is assigned to cover the whole-body DRIVER, which lives in a Tier-A crate.

§3.16 answers HR5 with the kernel test and §3.3 puts the whole-body run in
`crates/bins/tests/home_body_pin.rs`. The kernels are pure functions over a neighbourhood; the DRIVER —
the pass loop, the face walk, the seam conversion, the heap's own management, the allocation and free —
sits in `vd-terrain`, which is Tier-A at 100 % region and branch. A test in `bins` is outside the crate,
and HR5's own recorded discipline says *"cover a crate's generics completely in its own unit tests"*.

The report may already have the answer without knowing it: under its new rule a 2.5 km rock takes
`M = 1` (6 macro cells) and a 50 km moon takes `M = 32` (6 144 cells), so a real, tiny, in-world body
can drive the driver in the crate's own tests in milliseconds. §3.3 rejects the small-body cure for the
KERNEL test, correctly; it does not notice that the small body is exactly right for the DRIVER.

### W-2. `BodyDefinition` is `Copy`. A resident macro map cannot live in it.

`crates/terrain/src/body.rs:88` is `#[derive(Clone, Copy, Debug, PartialEq)]`. Everything downstream
takes a body by value on that basis. §3.8 and §4.1 say the macro map is resident per body and read per
column, and never say where it lives. The client is already safe (`ChunkJob.body` is
`Arc<BodyDefinition>`, `crates/client/src/chunks.rs:71`), but the crate's own API shape changes: the
body stops being `Copy`, and every by-value use in `chunk.rs`, `height.rs` and `digest.rs` changes with
it. It is small work, and it belongs in slice 8b's landing list, which does not mention it.

### W-3. The home planet's pinned literals grow from two to about thirteen, and every later tuning of a draw that does not exist yet becomes a world epoch.

`crates/terrain/src/home.rs` states the home planet as TWO literals — the seed and the radius's exact
bits — *"so the client — which links no motion crate — can fold its DECLARED world tag"*, cross-pinned
by `crates/bins/tests/home_body_pin.rs`.

Road B adds eleven facts. Three of them (spin, obliquity, water inventory) do not exist anywhere yet
(§2, MEASURED by grep). So the pinned set grows to about thirteen literals, each cross-pinned against a
forest draw that has not been written. Every later tuning of those three draws — and a first draw law is
always tuned — re-records the golden tables, bumps `GENERATOR_VERSION` and, after slice 9, opens a world
epoch.

§6.3's gate half-covers it (*"a cross-pin that the forest's numbers and the recipe's snapped numbers
agree"*). §4.2 does not name the cost, and §8's door table does not list "the spin, obliquity and water
draw laws" as a door — which they are, and a heavily-tuned one.

### W-4. The gravity snap's justification is off by 15×, and on the home planet the fact it justifies does not bind at all.

§4.2's table: *"Surface gravity … 1/64 m/s² … The relief bound moves by 0.02 % per step."*

On the home planet `g = 5.152 m/s²` (§3.4's own arithmetic), so one snap step is
`0.015625 / 5.152 = 0.30 %` of `g`. The strength bound is `h_max = σ / (ρ g)`, inverse in `g`, so it
moves **0.30 %** per step — 15× the stated figure. And §3.4's own conclusion is that on the home planet
the ACCRETION bound binds, not the strength bound, so gravity moves the relief bound by **0 %** there.
Neither reading gives 0.02 %, and no derivation is offered.

### W-5. The obliquity snap's justification is about 8× low.

§4.2: *"A cosine step of 1/1024 moves an ice cap's edge by about a kilometre on the home planet."*

Re-derived. At `ε = 23.4°`, `dε = d(cos ε) / sin ε = (1/1024) / 0.397 = 0.00246 rad`. On a radius of
3 350 759 m that is `0.00246 × 3 350 759 = 8 246 m`, about **8 km**, not one. The report gives no
derivation, so I cannot tell which quantity it meant. Eight kilometres is still far under the drawable
floor at any orbital distance, so the CHOICE survives; the number does not.

### W-6. The sample box holds 4 096 columns, not 4 268.

§3.6: *"The sample box is 35 % more than the bare chunk (1.46 → 1.98 ms, MEASURED), so a chunk samples
about 4 268 columns, not 3 844."*

The 35 % is real (`slice_06_extractor.md:355`). The column count that follows from it is not. The same
slice states the box's shape and its column cost directly: *"64 × 64 (the halo columns included: 6.5 %
more columns)"* (`:118`) and M6-5, *"The halo's cost: 64 × 64 columns against 62 × 62 … about 6.5 %
more"* (`:233`). So the box is `64 × 64 = 4 096` columns. `3 844 × 1.11 = 4 267` is the CELL ratio
(`64³ / 62³ = 1.10`, the same line's *"ten per cent more cells"*) applied to a COLUMN count.

The error is 4 % and it is conservative, so no conclusion moves. It is recorded because §3.6 presents it
as following from a measurement, and it does not.

### W-7. The 8 ms table omits three named slice-8c costs.

§3.6's added-work table holds three rows: the macro read, the nearest-river query, the climate. §6.5
lands, per column: the closed-form tributary network below the macro cell, the valley shaping function,
the alluvium on a valley floor, the stratigraphic-column lookup at a fixed radius, and differential
erosion on it. Only the river query is costed.

The margin the report defends is 1.29 ms on the worst chunk. The uncosted items are per-column work in
the same family as the ones costed at 40 to 80 ns each. Four more rows at 40 ns over 4 096 columns is
0.66 ms — half the margin. §10's U-L3 owes the measurement for three of the rows; it does not name the
other five.

---

## Notes

### N-1. "Fields" are cultural, like roads.

§6.9 correctly rules *"Roads | built, not seeded. Live state. | out of this arc"*. The row above it maps
*"Forest as a mass, **fields** below"* to *"the biome, the vegetation density, the canopy fold"*. The
reference picture's fields are CULTIVATED — bounded, rectangular, hedged. A biome gives grassland, not
a field. Fields are the same class as roads and belong in the same row.

### N-2. The static snow line is permanent snow.

§6.6 lands *"the snow line and the tree line as temperatures"* in the static recipe. A static snow line
is a PERMANENT snow line, which on Earth stands far higher than the seasonal one. Nothing is wrong with
that choice; it should be stated, because the reference picture's caps are (probably) seasonal and the
owner will compare.

---

## Checked, and sound

These I tried to break and could not.

- The ladder re-derivation. `n_ideal = radius_m * FRAC_PI_2` (`crates/seed/src/ladder.rs:76`), home
  `n_ideal ≈ 5 263 986`, `top_rung_for` gives 11, `snap` gives `n = 2 570 × 2 048 = 5 263 360`. Every
  row of §4.3's body table re-derives exactly, including the 1 000 km moon (`n = 1 570 816`, top 9,
  3 068 m), the 100 km moon (157 056, top 6, 2 454 m), the 50 km moon (78 528, top 5, 2 454 m) and the
  largest legal body (67 108 864, top 15, 131 072 m).
- `M = 2^min(E, top)` tiles exactly. `n` is a multiple of `2^top` by `snap`, and `M ≤ 2^top` is a power
  of two, so `n / M` is an integer at every body. At `E = 9` the home planet gives 10 280 m and
  `6 × 512² = 1 572 864` cells, 18.9 MB at 12 B.
- The small-body deletion argument. `g = (4/3)πGρR = 0.0839 m/s²` at 100 km and `ρ = 3 000`;
  `h_max = 200×10⁶ / (2 700 × 0.0839) = 883 km`, 8.8 × the body's radius; `Ladder::for_radius` returns
  `None` when `crust >= surface` (`ladder.rs:88-91`). A naive strength-only bound really would delete
  every small round body. The Vesta comparison is apt.
- The home planet's own numbers: `g = 5.152`, `h_max = 14 379 m`, accretion `13 403 m`, so the
  accretion bound binds; Mars's density moves `g` to 3.68 and `h_max` to 20.1 km. All re-derive.
- The 200 m floor is redundant: `0.004 × 50 000 = 200`.
- The boot-payer correction. `world_identity` has exactly two callers,
  `crates/bins/src/bin/gateway.rs:193` and `crates/bins/src/bin/client.rs:118`; the shard and the
  orchestrator fold `vd_bins::world_generation()` only. Revision 2's §3.9 is right and revision 1 was
  wrong.
- The float fence's contents. `crates/terrain/clippy.toml` bans every transcendental, `powf`, `powi`,
  `mul_add`, `min` and `max`, and `Gf` offers `+ − × ÷ √`, `floor`, `trunc`, `abs`, `lesser`,
  `greater`, `clamp` (`gf.rs:56-148`) — so `m = 1/2, n = 1` really does fit, and the published
  `m/n ≈ 0.35…0.6` range really does contain it. Rules 4, 5 and 6 (no angle, a vendored polynomial, a
  stated pass count instead of a convergence test) are each correct and each necessary.
- The look-tag law. `crates/core/src/look.rs:70-73` forbids the measured half in a look by name, and
  `:53-54` documents `TAG_SURFACE` as once-per-realm-on-change. Withdrawing L-2 and refusing weather on
  `TAG_SURFACE` are both right.
- The registry facts. `substance.rs:66-79` holds density, work of fracture and melting point and NO
  yield strength; `registry/digest.rs:1-9` folds *"every physical column"*. Appending the column really
  does move the digest and refuse every earlier store.
- The strata fact. `StrataTable::at(&self, biome, depth_m)` (`strata.rs:189`) reads a DEPTH, so no band
  can outcrop. The stratigraphic column at a fixed radius is the right cure and it is what the
  reference picture's pillars need.
- `POLE_AXIS` does not need to move. `height.rs:30-35` already says the parent authors the tilt, and
  the annual-mean insolation profile really is a second-order polynomial in `dir·pole` whose
  coefficient is a function of the obliquity's cosine (North's `S(x) = 1 + S₂P₂(x)`). Deleting §8's
  door is correct, and it is the report's best single call.
- The SL1 clause-4 argument's premise. `body_params` is documented as called *"with
  (illuminator_distance = its own orbit)"* (`taxonomy.rs:905-908`), so `insolation_rel` really does
  invert to the orbit's size and not to a placement. The argument stands on its own terms; L16 still
  belongs with the owner.
- Five legs, not four: `justfile:744-747` (debug, release, `target-cpu=native`) and `:756-758`
  (aarch64 Linux, emulated x86-64).
- The measured anchors: 3.30 ms and 6.11 ms per chunk, 710 µs and 266 µs per column pass, 13.5 ms of
  self-check, 14 305 m of relief, one column in a hundred under water — every one traces to
  `slice_05_generator.md:368-386` and `slice_06_extractor.md:344-350`.
- The residency arithmetic: one pixel at 5.83 million km (`6 701 518 m × 870`), 100 pixels at
  58 300 km, 32 minutes and 54 hours at 30 km/s. All three re-derive.
- The rung-11 read estimate: `62 × 2 048 = 127 km`, `(127 000 / 10 280 + 1)² ≈ 178` macro cells.
- The peak-memory sum: `18.9 + 6.3 + 1.6 + 12.6 + 6.3 = 45.7 MB`.
- Refusing ore in rivers (§3.2, L5) is right and it is the correct reading of the seed ruling. So is
  the observation that good land is ROOM, and room is live.
- Refusing revision 1's "small body is cheap" escapes, the 4 B / 12 B contradiction, the `sqrt(4π/6)`
  formula, the per-chunk worker seam and the "no saved record changes" claim: every one of those
  corrections is right, and §12 is an honest ledger.
