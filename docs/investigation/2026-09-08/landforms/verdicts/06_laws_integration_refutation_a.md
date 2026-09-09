# Refutation A of `06_laws_integration.md` (revision 2)

**Date:** 2026-09-08. **Refuter:** A. **Lens:** the laws and the code. Is every claim about the
repository true? Is every law gate passed? Does every number have a source?

**Method.** I read the document in full. I then read every file it cites, at the lines it cites, in
the working tree. I re-derived every arithmetic claim. Where the document quotes a source, I read the
source. Where the document states a rate, I checked it against the measurement it names.

**Summary.** Revision 2 is a large improvement. Every line citation into `body.rs`, `height.rs`,
`strata.rs`, `look.rs`, `tag.rs`, `chunks.rs`, `substance.rs`, `registry/digest.rs` and the `justfile`
is now correct. The whole ladder re-derivation in §4.3 is correct, and I proved the tiling claim the
document only asserts. The caller set in §3.9 is correct.

Two claims still break. **The proposed octave replacement turns a shipped Tier-A test red and stops
the ladder coarsening above rung 8** (blocker 1). **The relief fix does not fix what it says it
fixes: the draw still multiplies outside the bound** (blocker 2). Nine defects follow, including a
quotation that does not exist in the ruling it cites, a column count that contradicts the measurement
it cites, and a per-octave rate derived by dividing a total when the document's own two data points
give the marginal rate.

| Severity | Count |
|---|---|
| Blocker | 2 |
| Defect | 9 |
| Weakness | 8 |
| Note | 5 |

---

## BLOCKER 1 — The octave replacement turns a shipped test red and stops the ladder coarsening

**Where:** §3.4 (the audit table's "coarsest wave" row), §3.6 ("9 octaves instead of 14"), §3.7.

§3.4 replaces the coarsest wavelength with the macro cell: *"On the home planet that moves the start
from 400 km to 10 280 m: 9 octaves instead of 14."* §3.6 and §5 spend that saving.

The code refuses it.

- `BodyDefinition::octaves_at` keeps `octave_count − rung` octaves, **floored at one**
  (`crates/terrain/src/body.rs:251-261`).
- The home planet has **12 rungs** (`Ladder::for_radius` sets `rungs = top + 1`,
  `crates/seed/src/ladder.rs:94`; the top rung is 11, re-derived below).
- With 9 octaves the keep count saturates at rung 9. Rungs 9, 10 and 11 all keep **one** octave.
- The shipped Tier-A test asserts the opposite:

```
crates/terrain/src/body.rs:355-365
fn every_rung_sums_strictly_fewer_octaves_than_the_rung_below_it() {
    while rung < m.ladder.rungs {
        assert!(here < below, "rung {rung}: {here} vs {below}");
```

At rung 9, `here = 1` and `below = 1`. The assertion fails. The crate's own doc comment calls this
test the EXACT half of gate M-16 (`body.rs:340-342`: *"the coarse answer at every rung sums strictly
fewer octaves than the rung below it … this is the gate"*).

**The document cites that very test as proof the ladder still works** (§3.7: *"a unit test proves
that every rung sums strictly fewer octaves than the rung below it"*), then proposes the change that
breaks it, and never notices. §3.7's *"rung 11 (three octaves)"* and *"rung 11 drops eleven octaves
and saves a MEASURED 444 µs"* both describe the OLD table, not the proposed one.

**The arithmetic of the conflict.** Let `C` be the macro cell in metres. The octave count is
`ceil(log2(C ÷ SHORT_WAVE_M))` with `SHORT_WAVE_M = 30` (`body.rs:25,169`). For the count to stay at
or above the rung count on the home planet:

```
  count >= rungs = 12    =>   C / 30 >= 2^11   =>   C >= 61 440 m
  C = n / M = 5 263 360 / M                    =>   M <= 85.7   =>   M <= 64 = 2^6
  so  E <= 6,  giving 6 x 64^2 = 24 576 macro cells of 82 240 m
```

`E = 9` (the recommendation in §0 item 1 and decision L4) needs `E <= 6` to keep the ladder's own
gate. Those are 64 times apart in cell count and 8 times apart in linear resolution.

**What is owed.** Either state that the octave table keeps its coarse octaves and the macro map is
ADDED on top — in which case the 254 µs saving in §3.6 and §5 disappears and the long wavelengths are
counted twice — or state the new octave floor rule and the replacement for the M-16 gate, and re-open
`E`. The document must not present §3.7 as "nothing regresses" while its own §3.4 removes the octaves
the ladder coarsens with.

---

## BLOCKER 2 — "a bound is a bound" is false: the draw still multiplies outside the bound

**Where:** §3.4, "THE TWO BOUNDS ON RELIEF"; decision L3; §8's first door row.

The document correctly identifies the defect (finding A-10 in its own §13): today's flow is
`share → clamp → × [0.5, 1.5)`, so the drawn relief exceeds the "cap". Verified in the code:

```
crates/terrain/src/body.rs:147-149
let relief_share = radius_m * Gf::from_f64(0.004);
let relief_cap   = relief_share.clamp(Gf::from_i64(200), Gf::from_i64(12_000));
let relief_m     = relief_cap * (Gf::HALF + draw_unit(&mut octave_rng));
```

MEASURED cap 12 000 m, MEASURED drawn relief 14 305 m
(`docs/investigation/2026-09-07/slice_05_generator.md:368,388`). The finding is right.

**The stated cure is the same algebra.** §3.4: *"The rule: `relief = draw × min(strength bound,
accretion bound)`, with the draw INSIDE, so a bound is a bound."*

Today's code is `draw × min(max(share, 200), 12 000)`. The proposal is `draw × min(bound₁, bound₂)`.
The draw is in the same place. The document never restates the draw's range, which is `[0.5, 1.5)`
(`Gf::HALF + draw_unit`, `body.rs:149`). So under its own rule:

```
  accretion bound (home)      = 0.004 x 3 350 759 = 13 403 m   (the document's own number)
  strength bound  (home)      = 14 379 m                        (the document's own number)
  min                         = 13 403 m
  relief = draw x 13 403      with draw < 1.5  =>  up to 20 105 m
```

20 105 m stands above BOTH bounds. The defect the document says it fixed survives verbatim, and the
sentence *"so a bound is a bound"* is false as written.

**Also wrong in the same paragraph:** *"Against today's drawn 14 305 m that is a change of up to
51 %."* The range is 6 701 m to 20 105 m against 14 305 m, that is **−53 % to +41 %**. The low side
is the one that matters, because it is the side that flattens the home planet.

**What is owed.** State the draw's new range (`[0, 1]` makes a bound a bound; `[0.5, 1.0]` keeps
variety and a bound), and re-derive the home planet's relief under it. That number, not the bound, is
what the owner is choosing.

---

## DEFECT 3 — A quotation that does not exist carries the plan's central ordering claim

**Where:** §6.1, "The rule that fixes the order".

The document writes:

> **The store lands at slice 9** (`owner_decisions_2026-09-07_voxels.md:263`: *"no store exists yet
> (the first is slice 9)"*)

The ruling's lines 262-263 read:

```
docs/design/owner_decisions_2026-09-07_voxels.md:262-263
... so the family is per-body data (ruling V5) and a look cannot be confused with a bound; no caller
exists yet (the first is slice 9), and the round-versus-lump threshold is the body definition's ...
```

`grep -n "no store exists yet" docs/design/owner_decisions_2026-09-07_voxels.md` returns nothing. The
word is **caller**, not **store**, and the sentence is about the grid derivation's first caller, not
about a store at all.

The conclusion happens to hold — line 385 of the same ruling says *"bind the store, slice 9"* — but
the document's most consequential planning claim (the free window, decision L6, the whole §8 door
table) is carried by a quote that was manufactured. A refuter caught revision 1 asserting slice 14
with no source (finding B-W5); the answer replaced an unsourced claim with an invented one.

---

## DEFECT 4 — The no-drift gate is presented as sufficient; the owner has already ruled it is not

**Where:** §3.1, "The gate, stated so that it can go red"; §6.4's gate; §11's M-L2.

§3.1 states the SL10 gate as five legs and says it can go red. The document calls the iterative
erosion the biggest new drift risk the crate will ever hold, then rests it on that gate.

The owner has already ruled on that gate's x86-64 half:

```
docs/design/owner_decisions_2026-09-07_voxels.md:375
| S5-5 the x86-64 leg | Yes: emulation as a smoke test now; a real machine before "no drift" is
called satisfied. |
```

The `justfile` says the same in the recipe the document cites:

```
justfile:753-755
# ... and under x86-64 EMULATION (a smoke test: it can find a drift, never prove its absence
# — a real x86-64 machine is owed, DEFERRED D-TERRAIN-1). Not in `gate`: it needs Docker.
```

So of the five legs, three are the same host in three build modes, one is aarch64 Linux, and the
fifth is an emulator the owner has ruled non-probative. The document never names S5-5, never names
`D-TERRAIN-1`, and never says that a 100-pass accumulation over 1.57 million cells is exactly the
computation an emulator's rounding is least likely to reproduce. M-L2 is described as deciding
*"whether an iterative layer is lawful at all"* — it cannot decide that on the legs that exist.

**What is owed.** Name `D-TERRAIN-1` as a precondition of slice 8b, or state plainly that the
landform arc lands with no-drift UNSATISFIED by the owner's own definition.

---

## DEFECT 5 — "4 268 columns" contradicts the measurement it cites

**Where:** §3.6, "Note on the halo"; every row of §3.6's added-work table; §5.

The document writes:

> The sample box is 35 % more than the bare chunk (1.46 → 1.98 ms, MEASURED), so a chunk samples
> about 4 268 columns, not 3 844. The estimates below use 4 268.

The cited source says the opposite about columns:

```
docs/investigation/2026-09-07/slice_06_extractor.md:355
- The halo costs 35 % over the bare chunk at rung 0 (1.46 -> 1.98 ms): ten per cent more cells,
  six per cent more columns, the partner lattices, and the loss of the skips on the halo layers.
```

And the geometry is stated exactly:

```
docs/investigation/2026-09-07/slice_06_extractor.md:97
So the extractor works on 64 x 64 x 64 samples: the chunk plus a one-cell halo on every side.
```

The sample box holds `64 × 64 = 4 096` columns. `4 096 ÷ 3 844 = 1.0656` — the six per cent the source
states. The document took the **ten per cent more CELLS** figure and applied it to COLUMNS:
`3 844 × 1.11 = 4 268`.

Consequences: every added cost in §3.6 and §5 is 4.2 % high (the total falls from 0.60 ms to
0.574 ms). The direction is conservative, so nothing breaks — but the document explicitly says *"the
estimates below use 4 268"*, and it uses **3 844** four paragraphs later for the octave saving. Two
column counts in one section, and neither is the measured one.

---

## DEFECT 6 — The per-octave rate divides a total where the document's own data give the margin

**Where:** §3.6, "What landforms REMOVE"; §5's `- 254 us` row; §3.4's octave row.

The document states *"the MEASURED rate of 13.2 ns per octave sample per column"*. That number is
`710 µs ÷ 14 octaves ÷ 3 844 columns = 13.19 ns`, which assumes the column pass has **zero fixed
cost**.

The cited table gives two points, and they refute the assumption:

```
docs/investigation/2026-09-07/slice_05_generator.md:373-380
| 0 (1 m)     | 710 us, 14 octaves | ...
| 11 (2 048 m)| 266 us,  3 octaves | ...
```

```
  marginal cost per octave = (710 - 266) / (14 - 3) = 444 / 11 = 40.4 us per octave
  fixed cost               = 710 - 14 x 40.4        = 145 us
  per octave per column    = 40 400 ns / 3 844      = 10.5 ns, not 13.2 ns
```

The saving for five dropped octaves is therefore **202 µs, not 254 µs** — the document overstates by
26 % the only credit the landform layers earn against the 8 ms budget. §3.7's other number
(*"rung 11 drops eleven octaves and saves a MEASURED 444 µs"*) uses the two data points directly and
is correct, so the document holds both the right method and the wrong one, two sections apart.

(If blocker 1 forces the octave table to keep its coarse octaves, the saving is zero and this row
disappears.)

---

## DEFECT 7 — The C1 read and the boot canary contradict each other

**Where:** §3.6, "Why C1 and not bilinear"; §3.9, cure rule 1.

§3.6 costs the macro read as a C1 kernel: *"16 loads in a bicubic-class kernel"*. A 16-tap kernel
reads a 4 × 4 footprint of macro cells.

§3.9 rule 1 then says:

> The eight golden chunks stay, and they are evaluated at addresses whose macro dependency is a
> SINGLE macro cell, read from a stated pinned value rather than from a built map.

No column can depend on a single macro cell under a 16-tap kernel. A rung-0 chunk spans 62 m; even at
a face centre, well inside one 10 280 m macro cell, the bicubic footprint reaches the 4 × 4
neighbourhood around it. The address the document promises does not exist.

Two further problems in the same rule:

1. **A pinned macro value is a stand-in on the shipped path.** The world identity's MEASURED half
   exists to prove that this binary, on this chip, computes the world's bytes. If the eight chunks
   are fed a pinned value the world never uses, the gateway and the client compare a computation the
   world never performs. That is the "test exactly production" law inverted at the one gate whose job
   is to refuse a differing build.
2. **A kernel canary over a stated small neighbourhood cannot see accumulation drift.** The failure
   mode §3.1 rule 6 warns about is a single ulp changing a comparison after many passes over
   1.57 million cells. A five-cell neighbourhood run once exercises the arithmetic, never the
   accumulation. §3.9 calls the kernel canary *"STRONGER than moving the canary to a smaller body"*
   and presents it as settled; it is stronger in one dimension and weaker in the one that matters.

---

## DEFECT 8 — The snap grids are justified perceptually where the law demands identity

**Where:** §4.2, "The facts, the snap grids … and the fence", the "Why that coarse" column.

Every row justifies its snap unit by how far one step moves the WORLD:

- surface gravity, 1/64 m/s²: *"The relief bound moves by 0.02 % per step, far under the drawable
  floor"*
- equilibrium temperature, 1/16 K: *"It moves the snow line by a few metres"*
- obliquity cosine, 1/1024: *"moves an ice cap's edge by about a kilometre"*

Format D's tolerance is ZERO (ruling S5-2, `owner_decisions_2026-09-07_voxels.md:372`), which the
document itself quotes in §3.5. 0.02 % of a 13 403 m relief is **2.7 m** — thousands of moved bytes,
not "under the floor". A drawable floor is the wrong ruler for a byte-identity law.

**The right test is not applied, and the quantity it needs is never named.** A snap is safe when the
snap unit is far larger than the cross-host SPREAD of the fact, so both hosts land on the same
integer. That spread comes from the code that draws the facts:

```
crates/physics/  has NO clippy.toml — there is no float fence in the forest at all
crates/physics/src/worldgen/generate.rs:571   -scale_length * (u1.ln() + u2.ln())
crates/physics/src/worldgen/generate.rs:598   (r / r_max).max(1.0e-6).ln() / shape.pitch_tan
crates/physics/src/worldgen/generate.rs:639   DVec3::new(planar * azimuth.cos(), z, planar * azimuth.sin())
crates/physics/src/worldgen/generate.rs:1880  .powf(1.0 / ICE_MR_SEGMENTS[0].2)
```

`ln`, `cos`, `sin` and `powf` are libm calls whose results differ between targets. Their spread is
UNMEASURED, and the document does not say so.

**The precedent the document leans on measures the wrong quantity.** §2 and §4.2 call the radius snap
*"MEASURED, not argued"* and cite `crates/terrain/src/home.rs:49-77`. That test moves the radius by
1 000 ulps and asserts the same body. It measures the SNAP'S TOLERANCE. It never measures the
PERTURBATION the snap must absorb, because nobody has run the forest on two architectures and
compared. The document's own standing rule — "never assume, measure" — is broken by calling a
measurement of the wrong quantity a measurement.

§6.3's gate copies the same 1 000 ulps as a stated constant with no source. It should be derived from
a measured spread, per fact.

---

## DEFECT 9 — The accretion bound is the old magic number renamed, and the document's own witness refutes it

**Where:** §3.4, the audit table's first row and "THE TWO BOUNDS ON RELIEF"; decision L3.

The audit table calls the relief share `0.004` **MAGIC**. The cure keeps it:

> The accretion bound stays a share of the radius, which is lawful: it is a shape ratio, not a
> stand-in for a fact the world holds.

No source. No seed derivation. No physical driver, although the physical driver is already in the
repository and unused: `escape_velocity_mps` (`crates/physics/src/taxonomy.rs:755-758`) is what sets
how far impact ejecta travels and therefore how lumpy a small body stays.

**And the document's own geological witness contradicts the bound it defends.** §3.4 argues:

> A geologist agrees: Vesta carries about 20 km of relief on a 260 km radius, not 880 km …

```
  Vesta, as the document states it :  20 000 m / 260 000 m = 7.7 % of the radius
  the accretion bound, as stated   :   0.004 x 260 000     = 1 040 m
  ratio                            :   19 times too smooth
```

The sentence written to prove the accretion bound is right proves it is wrong by a factor of
nineteen. Real bodies carry MORE relative relief as they get smaller — a 50 km moon at `0.004 R`
gets 200 m of relief on a 50 km body, which is a billiard ball. A share of the radius has the
physics backwards, and it is Earth-calibrated: `0.004 × 3 350 759 = 13 403 m` is roughly Earth's
land-to-trench range, which is why nobody noticed.

**What is owed.** Either derive the accretion bound from a fact the world holds (escape velocity,
impact history, the drawn mass) with a citation, or say plainly that it is a stated shape ratio the
owner picks — the way §3.4 admits the water inventory's range is a quality choice. Do not call it
lawful because it is a ratio, while calling `0.004` magic in the row above.

---

## DEFECT 10 — The erosion's cost model omits the priority flood's heap

**Where:** §5, the "ORDERED stages" block; decisions L1 and L4; §4.1's verdict.

§5 costs the ordered stages as:

```
   ORDERED stages (priority flood, flow accumulation): 2 sweeps, ~1 random access per cell at 80 ns
     = 2 x 1 572 864 x 80 ns = 0.25 s
```

A priority flood is a heap, not a sweep. The document's own scratch table on the next line allots
*"a priority heap (up to 8 B/cell, 12.6 MB)"* — so it knows the heap is there and then costs it as
one access.

```
  heap entries                 1 572 864
  sift depth  log2(1 572 864)  ~ 21 levels
  levels resident in L2/L3     ~ the top 17 (up to ~1 MB of the 12.6 MB heap)
  levels that miss             ~ 4 per sift
  operations per cell          2 (one push, one pop)
  cost = 1 572 864 x 2 x 4 x 80 ns  =  1.0 s
```

One second for the flood alone, against a whole-build figure of **0.31 to 0.41 s** that is the sole
basis for recommending derive-over-ship (L1) and `E = 9` (L4). §5's own preamble scolds revision 1
for costing an erosion cell at a noise sample's rate — *"a category error"* — and then costs a heap
at a sweep's rate.

This does not refute the recommendation; it refutes the margin the recommendation is stated with.
M-L1 must run before L1 and L4 are answered, which §11 already says. §4.1's verdict — *"Recommended.
Under a second on one core"* — must not be written until it has.

---

## DEFECT 11 — HR4 is passed by naming gates that do not test the work

**Where:** §3.16, the HR4 bullet.

> **HR4, features once, run anywhere.** The landform work sits BELOW the `GridMapping` seam … Its
> gates are the seam's own (`G-MAPPING-TABLE`, `G-MAPPING-ROUNDTRIP`) plus the golden tables and the
> no-drift legs.

CLAUDE.md states those two gates as gates on the seam ITSELF: *"the seam ITSELF passes
G-MAPPING-TABLE (exhaustive at N = 62, the cube net) and G-MAPPING-ROUNDTRIP (at the largest legal
N), because no fixture can give equal results on a bent grid and a flat one (ruling V6 A1)."* An
exhaustive address round-trip at N = 62 proves the mapping is a bijection. It says nothing about
whether a drainage network is correct, whether a river crosses a seam, or whether a corner cell
drains.

So the landform work is given no HR4 gate of its own; it inherits the gates of the thing it sits on.
Ruling V6 A1's reason — a bent grid and a flat one cannot give equal results — bites HARDER on a
neighbour algorithm than on a height sample, which is the point §3.1 rule 7 makes well. The HR4
paragraph should carry rule 7's own gates (the eight corners, the twelve seams, the area weight) as
the landform layer's HR4 evidence, not the seam's bijection tests.

---

## WEAKNESS 12 — The relief budget split is never stated, and it answers the owner's stated worry

**Where:** §3.8's body-definition row; §3.4's octave row; absent everywhere else.

Today the octave amplitudes are scaled so their SUM is the drawn relief:

```
crates/terrain/src/body.rs:181-185
octaves[o].amplitude_m = octaves[o].amplitude_m * relief_m / weight_sum;
```

That scaling is what makes `relief_bound_m` exact and what sizes the band
(`body.rs:233-235`). §3.8 correctly says the band must grow by a macro-relief term. It never says
which of two things happens:

- **The macro relief is ADDED.** Total relief grows; the near field is unchanged; the band grows;
  `relief_bound_m` becomes `macro bound + octave sum`, which is what §3.8 writes.
- **The macro relief is TAKEN FROM the octave budget.** Total relief is preserved; the fine octaves
  shrink; close-range ground gets SMOOTHER than today.

The owner's complaint is *"the surface will not be interesting enough."* The second option makes it
worse and would not be visible in an orbital picture. This is the decision that answers him, it is
the decision that sets everything §3.8 promises to keep exact, and it is not in §9's table.

---

## WEAKNESS 13 — The macro map is finest where it is least useful and coarsest where the picture is

**Where:** §4.3's table and its "What the macro cell's metric size means, honestly" paragraph.

Under `M = 2^min(E, top)` the metric macro cell is `n / M`.

```
  body                       n            top   M     macro cell    rivers?  (per §0 item 7)
  100 km airless moon        157 056       6     64    2 454 m       NO
  50 km airless moon          78 528       5     32    2 454 m       NO
  home planet              5 263 360      11    512   10 280 m       yes
  an Earth-sized planet   10 006 000      12    512   19 543 m       yes
```

(The Earth row: `n_ideal = 6 371 000 × π/2 = 10 007 543`, top rung 12, `M = 512`.)

Every airless body that gets no river network at all gets the world's FINEST drainage skeleton, and
the resolution gets worse as a planet gets more Earth-like. The document states the variation
(*"The metric size varies per body, and this report says so plainly"*) but never names the inversion,
never names its consequence for the reference picture, and never considers the obvious alternative:
fix the macro cell in METRES (a stated resolution, exactly as `SHORT_WAVE_M = 30` is) and let `M`
follow from the body, capped by memory. That alternative is not in §9 L4's options.

---

## WEAKNESS 14 — At `E = 9` the macro map puts about five samples in the owner's picture

**Where:** §4.3; decision L4; U-L6.

```
  reference vista            ~50 000 m across
  home macro cell at E = 9    10 280 m
  macro cells in frame          4.9
```

The document says so itself: *"the macro map cannot draw every stream, and it is not meant to: it
carries the DRAINAGE SKELETON … the ridges and valleys the player SEES are the closed-form layer
keyed to that skeleton."* So `E = 9` buys 18.9 MB of residency, 0.3–1.0 s of build per body and a
whole new cross-host drift surface in exchange for five samples in the frame the owner pointed at.
`E = 8` puts 2.5 samples there for a quarter of the cost.

U-L6 correctly defers the judgement to pictures. Recommending 9 in §0, §4.3 and L4 before those
pictures exist is not deferring it. The recommendation should be "measure M-L1 and take the
pictures", with no number attached.

---

## WEAKNESS 15 — A spectral gap between the macro map and the octaves

**Where:** §3.4's octave row; §3.6's C1 kernel.

A grid of spacing `d` carries no feature below `2d`. A C1-interpolated macro map at 10 280 m
therefore carries wavelengths of about 20 560 m and longer. §3.4 starts the octave table AT
10 280 m. The band between 10 280 m and 20 560 m is carried by neither layer, and it is attenuated
in the shape.

On a 50 km vista that band is the scale of one whole ridge-and-valley system — the "orienter" scale
the owner asked for by name. The octave table should start at about twice the macro cell, which
changes the octave count again (`log2(20 560 ÷ 30) = 9.4`, so 10 octaves, not 9) and shrinks the
saving of §3.6 a third time.

---

## WEAKNESS 16 — No glacial process, in the one feature of the reference the owner pointed at

**Where:** §6.9's "Snow caps on the ridges only" row; §1's glossary; §6.4 and §6.5's *Lands* lists.

The erosion set is fluvial (stream power) plus thermal creep plus impact relaxation. §6.9 maps the
reference's snow-capped ridges to *"the lapse rate and the snow line"*, which are 8d's climate — that
is, to **paint**.

Above the snow line there is no liquid water, so the stream power law does not apply there at all.
The report's own physics therefore leaves the high ground shaped by uplift noise and creep, and then
colours it white. The reference's skyline is glacially sculpted: cirques, arêtes, horns, U-shaped
valleys with truncated spurs, hanging valleys with waterfalls. A V-shaped fluvial valley with a snow
cap on it is the picture a geologist would laugh at, and it is exactly the composition of the
reference frame.

§1 has no glossary row for a glacier, a cirque, a moraine or an equilibrium line altitude. §6.6's
*Lands* list has the snow line as a temperature and nothing else.

**What is owed.** Either land a cheap glacial pass on the macro map (an ice mask from the climate,
then a widening and over-deepening of the valleys inside it — one more streaming stage), or state
plainly to the owner that alpine skylines will be fluvial shapes with white paint, and let him rule.

---

## WEAKNESS 17 — "Asserted at every rung" describes a sampled test, not an invariant

**Where:** §2's band row; §3.8's body-definition row.

> `relief_bound_m` … is an EXACT bound asserted at every rung by `crates/terrain/src/height.rs:92`
> and `:102`.

Both lines sit inside `#[cfg(test)] mod tests` (`crates/terrain/src/height.rs:67` opens it), in a
test that walks 400 sampled directions on the home planet only. Nothing in the shipped code asserts
the bound.

The bound IS exact by construction — the noise is in `[−1, 1]` and the amplitudes sum to the relief
(`body.rs:181-185`) — so the claim of exactness is right. But §3.8 rests the ladder's coarse-answer
promise on the word "asserted", and after the macro map is added the exactness is no longer a
construction: it becomes a CLAMP the document proposes (*"the erosion is clamped to it. A test
asserts the clamp never binds on the home planet"*). A clamp that never binds on one body is not an
exact bound on every body.

---

## WEAKNESS 18 — HR5 covers the kernels and leaves the driver

**Where:** §3.3's third bullet; §3.16's HR5 bullet.

The plan is: the kernels are pure functions over stated neighbourhoods with unit tests, and *"the
whole-body run stays a gate, not a unit test"* in `crates/bins/tests/home_body_pin.rs`.

`vd-terrain` is Tier-A at 100 % region and branch (`just coverage-fast`). The DRIVER — the pass loop,
the ordered sweep, the allocation of the 45 MB of scratch, the residency, the refusal arms, the
`None` paths of a body with no water — lives in the Tier-A crate too, and the plan exercises it only
from a `bins` integration test. CLAUDE.md's HR5 discipline (c) says an integration test exercises the
full surface or none, and (a) says branching lives in monomorphic helpers.

The document never says which test binary covers the driver, nor what a 0.3–1.0 s whole-body build
does to the coverage run's wall time when it must run in the crate's own tests to count. This is the
same gap refuter B found in revision 1 (finding B-D9); the answer moved it from the body to the
kernels without closing it.

---

## WEAKNESS 19 — The SL6 ask L-4 states no data shape and no cost

**Where:** §7, row L-4; §3.14.

SL6's own words in CLAUDE.md: *"State what data, from which realm to which, why the receiver cannot
compute it from what it legitimately holds, and what doing without costs."* Default NO.

L-4 asks for *"The weather FIELD's coefficients, on a NEW weather tag with its own rate"* and
recommends **yes**. It states no count, no width, no rate. Its sibling L-1 states 22 bytes. §3.14
says the field is *"coefficients over the body's own surface (the same face grid the macro map uses,
at a very coarse `M`)"*.

```
  at M = 8 :  6 x 8^2   =   384 cells
  at one byte per cell   =   384 bytes, stated continuously
  the self-look bag's whole budget = 1 200 bytes (crates/core/src/look.rs:78)
```

384 bytes of a 1 200-byte bag, restated at some unstated rate, for one realm, is a shape the owner
cannot rule on. U-L8 says the bytes are unmeasured — so the row should say **ASK LATER, after
U-L8**, not **ASK: yes**.

---

## NOTE 20 — A superseded cell-pass figure is quoted as current

§2's budget row and §3.6 quote *"the cell pass 579 µs"* from slice 5. The newer measurement in the
document's other source supersedes it:

```
docs/investigation/2026-09-07/slice_06_extractor.md:361
- The cell pass at rung 0 went from 578 us (slice 5) to 716 us with the site plumbing.
```

§5 says those bare figures are used in no sub-row, so nothing downstream breaks. §2 still states the
old number as a present fact.

---

## NOTE 21 — Two different 12-byte records, one section apart

§3.8's first row says the **12-byte cell record** (ruling V6 B-1..B-13) does not change. §4.3 then
proposes a **12-byte macro cell** (height 4, flow direction 1, accumulation 3, water surface 2,
uplift class 1, hardness class 1). They are different objects — one is stored world data, the other
is derived residency — and they share a width with no sentence separating them.

---

## NOTE 22 — The flow accumulation's encoding is unstated

§4.3 gives the macro cell three bytes of flow accumulation. §3.1 rule 7(c) requires the accumulation
to be AREA-weighted by the bend's own factor, which is not an integer. Three bytes of what unit,
rounded how, and is the rounding stable through the ordered sweep on both hosts? The record is
proposed without it, and it is a byte of Format D's zero tolerance.

---

## NOTE 23 — "A small body costs strictly less" is not strictly true

§3.10: *"a SMALL body costs strictly less, because its top rung is lower."*

`M = 2^min(E, top)` saturates at `E`. `top ≥ 9` needs `chunks ≥ 257`, that is
`n > 256 × 3 968 = 1 015 808`, that is a radius above `1 015 808 × 2/π = 646 700 m`. So **every body
above about 647 km of radius pays the identical 18.9 MB**. A 2 000 km moon and the home planet are
equal, not ordered. The flat bound the paragraph states two sentences earlier is the correct claim;
"strictly less" is not.

---

## NOTE 24 — An airless body gets no craters

§0 item 7 gives a dead moon *"uplift, impact relaxation and thermal creep"*. Relaxation acts on
craters; nothing in the plan PLACES one. A 100 km moon with smooth uplift noise, creep and no crater
field is not a believable airless body, and ruling A5 (*"the generator always places a landform at
all eight cube corners"*, `owner_decisions_2026-09-07_voxels.md:184`) applies to it as much as to the
home planet. §6.4's *Lands* list has no crater field.

---

## Checked, sound

Each of these I re-derived or read at the cited line, and each holds.

1. **§4.3's whole ladder re-derivation.** `n_ideal = radius_m * FRAC_PI_2` (`crates/seed/src/ladder.rs:76`),
   `snap` (`:59-66`), `top_rung_for` (`:48-57`), `CHUNK_EDGE = 62`, `TOP_RUNG_CHUNKS = 64` (`:21-23`),
   `N_MAX = 1 << 26`, `RUNG_MAX = 15` (`:25-27`). I re-derived every row independently: home
   5 263 360 / top 11 / M 512 / 10 280 m; 1 000 km moon 1 570 816 / 9; 100 km moon 157 056 / 6;
   50 km moon 78 528 / 5; 2.5 km rock 3 927 / 0 / six cells; the largest body 67 108 864 / 15 /
   131 072 m. All correct. Revision 1's `sqrt(4π/6)` really was wrong and really did poison
   everything downstream.
2. **The tiling proof.** The document asserts `n / M` is an exact integer. I proved it: after
   `snap`, `n ≤ 3 968 × 2^t` and `n` is a multiple of `2^t`, so the recomputed `top_rung_for(n)` can
   only fall, never rise; `M = 2^min(E, top) ≤ 2^top ≤ 2^t` divides `n`. Refuter B's finding B-2 is
   correctly and completely closed.
3. **§3.9's caller set.** `world_identity` (`crates/bins/src/lib.rs:1047-1053`) is called exactly
   twice in production: `crates/bins/src/bin/gateway.rs:193` and `crates/bins/src/bin/client.rs:118`.
   `shard.rs:279` and `orchestrator.rs:86,596` fold `world_generation()` (`lib.rs:970-979`), a
   constant. The gateway half of the finding is real and is the sharp half, as the document says.
4. **The stratigraphic finding.** `StrataTable::at(&self, biome, depth_m)` really is at
   `crates/terrain/src/strata.rs:189`, and it really is read at a depth below the surface
   (`crates/terrain/src/chunk.rs:637`), so no band can outcrop across a slope. No caprock, no mesa,
   no pillar. This is the document's best new finding and it is correct.
5. **The float fence's contents.** `crates/terrain/clippy.toml` bans `sin`, `cos`, `asin`, `atan2`,
   `to_radians`, `exp`, `ln`, `powf`, `powi`, `mul_add`, `min` and `max` — and allows `sqrt`. So
   rule 1 (`m = 1/2` needs only `√`), rule 4 (no angle) and rule 5 (a vendored polynomial) are all
   correctly derived from the code, not from memory. `dir[POLE_AXIS].abs()` at `height.rs:45` really
   is the pattern already in use.
6. **The seam and corner rule (§3.1 rule 7).** `three_faces_meet_at_every_cube_corner` really is at
   `crates/seed/src/seam.rs:309-320`; ruling A5 really is at
   `owner_decisions_2026-09-07_voxels.md:184`; `bend.rs:24-26` really gives `k₁ = π/4`. This is the
   most valuable addition in revision 2.
7. **The strength-bound arithmetic.** 100 km moon: `g = (4/3)πGρR = 0.0839 m/s²`,
   `h_max = 200e6 / (2 700 × 0.0839) = 883 km`, 8.8 × the radius. Home: `g = 5.152`,
   `h_max = 14 379 m`; at Mars's density `g = 3.684`, `h_max = 20.1 km`. All four re-derived and
   correct, and correctly labelled ASSUMED.
8. **`Ladder::for_radius` returns `None` when `crust >= surface`** (`ladder.rs:88-91`), so refuter
   B's "a naive slice 8a deletes every small body" really is a live failure mode, and §6.3's gate is
   the right one.
9. **The registry digest really does fold every physical column** (`crates/core/src/registry/digest.rs:9`),
   so appending a yield strength really does move it. `substance.rs:74-77` really holds density and
   work of fracture and no yield strength.
10. **`look.rs`'s own law.** `SurfaceStmt.generator` is documented *"Never the measured half"*
    (`:70-73`), `TAG_SURFACE` is *"ONCE PER REALM ON CHANGE, carried retained, never on a
    keep-alive"* (`:53-54`), `SELF_LOOK_BUDGET_BYTES = 1200` (`:78`). The withdrawal of L-2 and the
    move of weather to its own tag are both correctly grounded.
11. **The worker seam's shape.** `ChunkWorkers` takes a `ChunkJob` and cancels by
    `(RealmId, ChunkKey)` (`crates/client/src/chunks.rs:85-93`); `InlineWorkers` *"runs on the
    calling thread at `submit`"* (`:95-97`). The new body job kind really is owed.
12. **The five legs.** `terrain-pin` really runs three host legs (`justfile:744-747`);
    `terrain-legs` really runs two Docker legs (`justfile:756-758`).
13. **§4.1's reach arithmetic.** At the drawable floor (1 m subtends 1 px at ~870 m), a 6 701 518 m
    planet is 1 px at 5.83 × 10⁹ m and 100 px at 5.83 × 10⁷ m; at 30 km/s that is 54 hours and
    32 minutes. Correct.
14. **§3.7's rung-11 footprint.** `62 × 2 048 = 126 976 m`, `(126 976 ÷ 10 280 + 1)² = 178`. Correct.
15. **§5's streaming arithmetic.** `100 × 2 × 4 B × 1 572 864 = 1.26 GB`; at 8–20 GB/s that is
    0.06–0.16 s. Correct (the ordered half is defect 10).
16. **The seed ruling's application (§3.2).** Refusing placer ore in a river bed is right, and the
    "good land is room, and room is live" reading of the galaxy ruling is a correct use of it.
17. **§3.14's three-layer split and the decaying-offset wake rule**, and §3.15's static landscape.
    Both are correct against SL10 clause 1 and the dormant-world design.
18. **§12.** Every measurement it quotes from slices 5 and 6 I checked at its line and found exact:
    710 µs / 14 octaves, 266 µs / 3 octaves, 13.5 ms self-check, 1.98 + 1.32 ms plain chunk,
    4.38 + 1.73 ms worst named chunk, 8 ms budget, relief 14 305 m, sea 5 297 m under, last chunk
    (84 892, 7). §12 remains the report's strongest section.
