# Refutation A — domain 07, `07_measurements_costs.md` (revision 2)

**Date:** 2026-09-08. **Refuter:** A. **Lens:** the laws and the code. Is every claim about the
repository true? Does every law gate pass? Does every number carry a source?

**Method.** I read the document in full. I then read `crates/terrain`, `crates/seed`,
`crates/bins/examples/terrain_cost.rs`, `crates/core/src/geometry.rs`, the `justfile`, the ruling
`owner_decisions_2026-09-07_voxels.md`, and the sibling documents 01 and 03. I replayed four of the
document's own numbers in `python3`: the ladder snap, the octave amplitudes, the cube-sphere area
metric, and the divisor families of several bodies. Every arithmetic statement below is shown.

I found **4 blockers, 9 defects, 9 weaknesses and 10 notes**. The document is honest about what it
does not know, and it is far stronger than a first revision. Its failures are of two kinds: it states
one rule and then breaks it, and it prices a kernel that is not the kernel its sibling domains design.

---

## Blockers

### B1 — THE COMPOSITION RULE IS NEVER STATED, AND THE SIBLING DOMAIN STATES THE OPPOSITE ONE

The document never says how the macro map's height JOINS the column pass's octave sum. §5.3 says "The
base height is the existing octave field, evaluated once per macro cell", so a macro cell holds
`octaves + erosion`. §7.1 then charges "the macro height sample" as an ADDITION to the sample box that
already sums all fourteen octaves:

```
   TODAY, a surface chunk           3.30 ms   (the box sums 14 octaves per column)
   ADDED: the macro height sample   0.18 ms   (a bicubic of a field that ALREADY holds octaves 1..k)
```

If the chunk adds the macro sample to its own octave sum, then octaves 1 to k are counted **twice**,
and the home planet's long-wave relief doubles from 14 304 m to about 26 000 m. If the chunk instead
REPLACES its coarse octaves with the macro sample, then §7.1's arithmetic is wrong in the other
direction (a saving is owed, not a cost), §9.2's "which level a rung reads" table changes meaning, and
the crossfade the ladder already runs on octaves must be re-designed.

The word "residual" appears exactly once in the document (`07_measurements_costs.md:903`), inside the
DELETED background-build paragraph. The rule is nowhere.

The sibling domain has already decided it the other way:
`03_erosion_rivers.md:4` item 4 — *"`height_m` gains three terms: the eroded macro field `Z`, **which
REPLACES the coarse octaves**"*. Domain 07 is the cost document for that design and it prices the
other one.

**Why it is a blocker.** Every number in §7.1, §7.2, §9.1 and §9.2 depends on which of the two is
true, and so does the ladder gate (V8/V9) and SL8. A cost document that does not know what it is
costing cannot be read.

### B2 — THE ARRIVAL COST COUNTS ONE PATCH PER LEVEL; A DESCENT NEEDS TENS

§5.5 states `FIRST ARRIVAL = 2.28 s more`, which is three patches, one per level, at 0.76 s each.
§7.4 then prices a patch build only as a HORIZONTAL crossing rate.

A vista is not one patch. §9.2 says rungs 0 to 7 read level 3, and §8.3 says a level-3 cell must be
finer than one drawn pixel inside 236 km. So every square metre of ground within 236 km of the pilot
needs a level-3 patch, and a level-3 patch covers 256 x 257 m = 65.8 km on a side, an area of
4.33 x 10^9 m².

The home planet's radius is 3 350 759 m (MEASURED, `terrain_cost`; reproduced from
`Ladder::for_radius`). The geometric horizon from altitude `h` is `sqrt(2 R h)`.

```
   altitude   horizon     level-3 ground radius            level-3 patches   one-core build   resident
   --------   --------    ------------------------------   ---------------   --------------   --------
     1.7 m      4.6 km    4.6 km (the horizon binds)                     1          0.76 s     0.46 MB
     5 km      183 km     183 km (the horizon binds)                    24         18.2 s      11.0 MB
    20 km      366 km     235 km = sqrt(236^2 - 20^2)                   40         30.4 s      18.4 MB
   100 km      819 km     214 km = sqrt(236^2 - 100^2)                  33         25.1 s      15.1 MB
```

DERIVED: `pi * r^2 / 4.33e9`, then `x 0.76 s` and `x 0.459 MB`.

So a pilot at 20 km — an ordinary point on a descent, and the flight the slice-7 gate already flies —
needs **30 s of one-core build and 18.4 MB resident**. The document's own ceilings are
`BOOT_CEILING = 2 s on one core` and `READ_CEILING = 8 MB per body`. The recommended shape misses the
first by fifteen times and the second by more than two.

The level-2 ring adds to it. At 100 km altitude the ground from 214 km to the horizon at 819 km needs
level 2 (329 km patches): `pi(819^2 - 214^2) km^2 / 329^2 km^2 = 18` patches, another 13.7 s.

§7.4's kill row and M-L15 do not catch this, because both measure a HORIZONTAL crossing. §7.1 mentions
"the descent's arrival rate" only for CHUNKS.

**The cure the document owes** is a per-level patch RESIDENCY count as a function of altitude, a
memory ceiling that is a total and not per body, and either a much smaller patch or an admission that
the pyramid needs a fourth global level.

### B3 — A LEVEL-3 PATCH DOES NOT TILE THE LEVEL-2 PARENT GRID

§1's pyramid uses the ratios 8, 8, 5:

```
   level 0     64 cells per face edge     82 240 m
   level 1    512                         10 280 m     x8
   level 2  4 096                          1 285 m     x8
   level 3 20 480                            257 m     x5
```

A patch keeps 256 cells and builds 384 (§5.5). Divide by the level-3-to-level-2 ratio of five:

```
   256 / 5 = 51.2      the kept square is 51.2 parent cells across
   384 / 5 = 76.8      the built square is 76.8 parent cells across
    64 / 5 = 12.8      the halo is 12.8 parent cells wide
```

None is whole. Three clauses of §5.6 then cannot all hold on the same patch:

- **C2 / E9 (mean-pinning)** shifts each parent cell's block of children so the block's mean equals the
  parent's height. A parent block that lies partly outside the patch has no mean to shift. §12 E9's
  coverage note calls this "the mean-pinning arm where a parent block is partly outside the patch" and
  treats it as a test fixture. It is not a fixture; it is the whole boundary, and it sits exactly where
  C4 demands ZERO quanta of disagreement.
- **C4 (the shared edge is exact)** pins the boundary column to the parent's bicubic value with no
  erosion applied. C2 then adds a uniform shift to the whole block that contains that column. After the
  shift the column no longer equals the parent's bicubic value. The two clauses contradict each other
  wherever a parent block straddles a tile line — which, at 51.2 parent cells per patch, is four patch
  edges in five.
- **§9.2 (a coarse rung reads level 2 and sees the mean of level 3)** is the property C2 buys. It is
  false on every straddling block.

M-L4's pass bound is `MAX difference = ZERO quanta`. By the document's own arithmetic that bound cannot
be met.

**The arithmetic that would fix it** is a patch whose kept edge is a whole multiple of every ratio in
the pyramid: 256 works for 8, and 320 or 240 works for 5 and 8 together (`320 = 5 * 64`,
`240 = 5 * 48`). Neither is chosen and neither is priced.

### B4 — THE KERNEL PRICED IS NOT THE KERNEL THE DESIGN NEEDS, AND THE SIBLING'S GRID IS NEVER PRICED

§4.1's 160 ns buys nine items: receivers and slope, donor counting, basin labelling, priority flood,
the stack build, flow accumulation, incision, hillslope diffusion, mean-pinning and quantisation.

`03_erosion_rivers.md` — the domain that designs the erosion this document prices — requires more, by
name: **isostatic rebound on a pyramid** (`:55`, `:671`, `ISOSTASY_EVERY = 5`), **talus relaxation**
(`TALUS_PASSES = 8`, `:704`), **an ice depth proxy of about twenty passes** (`:732`), **a coast mark**,
**a lake table** (`:62`), **a climate re-solve every ten passes** (`CLIMATE_EVERY = 10`), and a
**quantised-logarithm discharge byte**. None of these appears in §4.1's op count, in §5.4's pass table,
in §5.2's 29-byte build record or in §5.2's 3-byte read record.

Worse, the two documents recommend different grids and the cost document never prices the other one:

```
   domain 03: ONE GLOBAL macro lattice, 640 nodes per face edge, 8 224 m per node,
              2 457 600 nodes, PASSES = 40, 9.83 MB           (03_erosion_rivers.md:55-63)
   domain 07: 6 x 64^2 global at 82 240 m, I = 128, plus patches, 4.67 MB
```

Price domain 03's grid with domain 07's own kernel:

```
   2 457 600 nodes x 40 passes x 160 ns  =  15.7 s on one core     (8x BOOT_CEILING)
   9.83 MB                                                          (over READ_CEILING = 8 MB)
   and under §5.3's own law I = 2M = 1 280 passes:  503 s
```

So domain 07's model KILLS domain 03's recommendation, at eight times its own boot ceiling, and says
nothing. A cost document exists to price the design. This one prices a design nobody proposed.

---

## Defects

### D1 — THE DRAWABLE FLOOR IS 1.1506 mrad IN THE CODE, NOT 1.0908

§8.3: *"The reach ruling's own floor is 1 px at 45 degrees over 720 rows = **1.0908 mrad** (owner
2026-09-06)."* The code computes it differently (`crates/core/src/geometry.rs:1164-1173`):

```rust
pub fn drawable_theta_min_rad() -> f64 {
    2.0 * (REFERENCE_VIEW_FOV_Y_RAD * 0.5).tan() / REFERENCE_VIEW_ROWS_PX
}
```

```
   the code:      2 * tan(22.5 deg) / 720  =  0.8284271 / 720  =  0.00115059 rad
   the document:      (pi/4)         / 720  =  0.7853982 / 720  =  0.00109083 rad
   error: the document's floor is 5.2 % too small
```

The code's value is the one the owner's own worked example reproduces: a one-metre extent is drawable
to `1 / 0.00115059 = 869 m`, which is the "~870 m" the reach ruling states. The document's value gives
917 m.

Every number in §8.3 and §13's M-L14 is 5.2 % wrong in the unsafe direction (it claims a level is
needed FARTHER away than it is):

```
   level      document      corrected
   -------    ----------    ----------
   0          75 400 km     71 480 km
   1           9 424 km      8 933 km
   2           1 178 km      1 117 km
   3             236 km        223 km
   relief     13 100 km     12 424 km
   M-L14's pixel at the 50 km vista:  54.5 m -> 57.5 m
```

This is a measurements document, and it attributes a wrong number to the owner's ruling.

### D2 — §3's CAVE-DENSE ROW MIXES TWO SOURCE ROWS, AND §7.1's HEADROOM RESTS ON THE RESULT

§3: *"A cave-dense chunk (rung 2, the seam chunk): box 4.38 + extraction 1.73 (**23 084 vertices**) |
6.11 ms, extraction **75 ns per vertex**"*.

The source table (`slice_06_extractor.md:348-353`) has two cave-dense rows:

```
   +X rung 2, chunk (3, 5), cave-dense               2.83 ms   2.39 ms   23 084 vertices
   +X rung 2, the last chunk (21 223, 7): a seam     4.38 ms   1.73 ms    8 234 vertices
```

The document takes the times from the second row and the vertex count from the first.

```
   the document:  1.73 ms / 23 084 =  75 ns per vertex
   the truth:     1.73 ms /  8 234 = 210 ns per vertex
                  2.39 ms / 23 084 = 104 ns per vertex
```

§3 then builds the curve that §7.1 uses: *"234 ns at 5 650, 75 ns at 23 084, 104 ns at 250 046 ... it
settles near 100 ns"*. The true curve is 234, 210, 104, 104 — it falls much later, so the extraction
growth is DEARER at the counts §7.1 actually uses:

```
   §7.1 surface chunk at 1.5x vertices:  8 475 x 180 ns = 1.53 ms   (document, +0.21)
                     at the true curve:  8 475 x 210 ns = 1.78 ms   (+0.46)
   NEW surface chunk:  3.30 + 0.40 + 0.46 = 4.16 ms, not 4.08 ms
```

And the cave-dense figure moves the other way, because its true base is 8 234 vertices, not 23 084.
Both headline numbers of §1 come from this row. §18 C13 already says the growth factor is the whole
headroom; it must be re-derived from the right rows.

### D3 — THE NYQUIST RULE IS MIS-APPLIED THREE TIMES, AND IT IS NOT THE RULE `octaves_at` IMPLEMENTS

§5.3: *"The rule, **which is the rule `octaves_at` already implements**, generalised from a rung index
to a cell size in metres: A level evaluates the octaves whose wavelength is at least TWICE its cell
size (Nyquist). Level 0 (82 240 m) is octaves 1 to 3. The climate grid (20 560 m) is octaves 1 to 5.
Level 3 (257 m) is octaves 1 to 11."*

**The claim about the code is false.** `crates/terrain/src/body.rs:253-258`:

```rust
pub fn octaves_at(&self, rung: u8) -> &[Octave] {
    let count = usize::from(self.octave_count);
    let keep = count.saturating_sub(usize::from(rung));
    ...
}
```

That drops exactly ONE octave per rung. At rung `r` the cell is `2^r` m and the finest kept octave has
a wavelength of `48.83 * 2^r` m, so the crate keeps octaves down to about **49 times** the cell size,
not twice it. The document's rule is 24 times more permissive than the crate's, and it is a different
rule, not a generalisation of it.

**The three applications are also wrong**, against the document's own definition. The home planet's
octave wavelengths are 400, 200, 100, 50, 25, 12.5, 6.25, 3.125, 1.5625 km, then 781.25, 390.6, 195.3,
97.7, 48.8 m (fourteen; MEASURED by replaying `body.rs:150-180` — `long_wave_m` clamps to the
400 km cap, and the loop is `while wave_m > SHORT_WAVE_M` with `SHORT_WAVE_M = 30`):

```
   level 0,  cell 82 240 m -> needs lambda >= 164 480 m -> octaves 1..2   (doc says 1..3; 100 km < 164 km)
   climate,  cell 20 560 m -> needs lambda >=  41 120 m -> octaves 1..4   (doc says 1..5;  25 km <  41 km)
   level 3,  cell    257 m -> needs lambda >=     514 m -> octaves 1..10  (doc says 1..11; 391 m < 514 m)
```

Off by one in all three. The consequence is not only cost: under the document's own rule level 3
ALIASES the 391 m octave, which is a 3.86 m ripple sampled on a 257 m grid.

### D4 — THE PATCH ITERATION COUNT BREAKS THE DOCUMENT'S OWN ITERATION LAW

§5.3 states the law and calls it the fact revision 1 missed:

> *"The iteration count is not a constant of the world; it scales with the grid's DIAMETER in cells,
> because the erosion signal travels from an outlet to a divide one cell per iteration. Writing
> `I = c * M` with `c` a recipe constant of order 2."*

§5.5 then applies `I = 2M` to level 0 (`M = 64`, `I = 128`) and applies **`I = 32`** to a patch of 384
built cells. Under the document's own law a patch needs `I = 768`.

```
   the document:  147 456 cells x  32 x 160 ns  =  0.76 s per patch
   its own law:   147 456 cells x 768 x 160 ns  = 18.1 s per patch     (24x)
   FIRST ARRIVAL: 2.28 s  ->  54.4 s
```

The escape §5.5 offers is the refinement argument: mean-pinning holds the long wavelengths, so only
the detail inside a parent cell has to grade, over five to eight child cells. That argument is
plausible and it is honestly marked ESTIMATED. But the document does not RECONCILE it with the law it
states two sections earlier, and the law is the one thing revision 2 added to kill shape A. A document
may not use a law to kill an alternative and then exempt its own recommendation from it without saying
why the exemption is lawful.

### D5 — "ANY WHOLE DIVISOR" DOES NOT GIVE A USABLE PYRAMID ON EVERY BODY (SL5)

§5.1 fact 2: *"**Family 2 removes the problem entirely**: every body's `N` is a whole number and has
divisors far below its top rung."* §8.4: *"The finer levels are `M * 8`, `M * 8`, `M * 5` (or whatever
whole ratios `N`'s factors allow)."*

`N = q * 2^top`, and `q` is `round(n_ideal / 2^top)` (`crates/seed/src/ladder.rs:60-65`), so `q` is an
arbitrary integer. The divisor set is therefore arbitrary. I factored the bodies of
`01_grid_family.md:302-309`:

```
   body          N            factorisation           the x8, x8, x5 ladder from M = 64
   -----------   ----------   ---------------------   ---------------------------------------
   home planet    5 263 360   2^12 * 5 * 257          64, 512, 4 096, 20 480      WORKS
   moon (200km)     314 112   2^8 * 3 * 409           64, then 512 does NOT divide (2^9 > 2^8)
   Luna           2 728 960   2^10 * 5 * 13 * 41      64, 512, then 4 096 does NOT divide
   Earth-size    10 006 528   2^12 * 7 * 349          64, 512, 4 096, then no factor 5 exists
   Ceres            742 912   2^9 * 1451              64, 512, then only 1451 and 2 902
   asteroid (500m)      785   5 * 157                 the ONLY divisors are 5 and 157
```

On the 200 km moon the second level of the recommended pyramid does not exist. On a 500 m asteroid the
whole macro map is a global grid of `6 x 5^2 = 150` cells or `6 x 157^2 = 147 894` cells, with nothing
between. SL5 says ONE world, and every body in it is that world.

The document never enumerates the world's bodies against its own rule. §8.4's parenthesis — "or
whatever whole ratios `N`'s factors allow" — is where the hard part is, and it is one clause long.

**The measurement it owes:** run the chooser over every body kind of `01_grid_family.md`'s table and
print the ladder each one gets. It is a `python3` script, and it takes minutes.

### D6 — THE COARSE-ANSWER LAW IS PROVEN ON THE BUILD RECORD AND CONSUMED FROM THE READ RECORD

E9's control (§12): *"A test that for every parent cell the children's mean equals the parent **to the
millimetre**"*. The build record holds `height, i32 millimetres` (§5.2). The **read** record holds
`height, i16 metres` (§5.2), and the read record is what a chunk samples (§7.2).

Quantising to whole metres destroys the mean relation the test proves:

```
   five level-3 children, exact heights:  1 200.6, 1 200.6, 1 200.4, 1 200.4, 1 200.4 m
   their exact mean, and the parent:      1 200.48 m
   after i16-metre rounding:  children 1 201, 1 201, 1 200, 1 200, 1 200 -> mean 1 200.4
                              parent   1 200
   the mean relation is broken by up to +/- 0.5 m per parent cell
```

§9.2 rests the whole "no pop across a level boundary" claim on that relation ("bounded by
construction, not by a crossfade"). The control tests a record nobody reads.

§18 C15 notices half of this (the slope quantum) and calls it a check that "deserves" running. It is
not a check on a detail; it is the record on which the SL8 argument stands.

### D7 — THE DOCUMENT'S OWN RATE TABLE FIRES ITS OWN KILL ROW, AND THE TEXT SAYS THE OPPOSITE

§7.4's table:

```
   speed        one patch every     build share of one core
   3 000 m/s    21.9 seconds        3.5 %
   30 000 m/s    2.2 seconds        35 %
```

§14's kill row: *"A patch built on a shard worker | KILL at M-L15's build share over **10 %** of one
core"*. §13's M-L15 pass bound: *"the build share of one core under 10 %"*.

35 % is over 10 %. §7.4 nevertheless concludes: *"**At every speed** the lead a build needs is far
inside the patch the pilot is leaving, so a one-patch lead along her own address chain is enough."*
The lead in METRES is indeed inside the patch. The CPU SHARE is not, and the share is what the kill row
reads. The two paragraphs disagree, and the reader is told the good one.

The share is worse than the table shows: a pilot crossing a patch boundary enters a new ROW of
patches, so at least three new level-3 patches are needed, not one. That is 105 % of one core at
30 000 m/s.

### D8 — THE ARRIVAL CASE HAS NO LEAD, SO THE COLLIDER'S SHAPE DOES NOT EXIST

§14's fallback: *"the shard pre-builds along the occupant's own address chain, one patch ahead (§7.4
shows the lead is always inside the patch)"*.

A lead exists only when there is a previous position. It does not exist for:

- a login (the pilot's stored position is anywhere on the body);
- a transfer into the planet's realm from a hull;
- a warp arrival;
- a shard restart after a kill-9, which the project's own chaos matrix runs.

In each of those the pilot lands where no patch is built. V1 clause 5 says the shard computes collision
on the same shape. Until the patch exists there is no shape, so there is no collider. Under SL8 that is
a seam, and under the character controller it is a fall through the world.

The document has a place for this and does not use it: §8.1 answers the CLIENT's cold start with "a
REFUSAL plus a STAGING BY DISTANCE". The SHARD's cold start has no answer at all.

### D9 — THE OP COUNTS FOR THE TWO PASSES REVISION 2 ADDED ARE OPTIMISTIC AND INTERNALLY INCONSISTENT

§4.1 charges:

```
   BASIN LABELLING (pointer jumping over the forest)    ~4 jumps, irregular            20 ns
   PRIORITY FLOOD (a heap over the pit cells)           ~19 compares, irregular        55 ns
```

Three problems.

1. **The flood is not "over the pit cells".** A priority flood pushes the whole boundary and then pops
   every cell of the grid exactly once, raising each to its flood level. That is the published
   algorithm, and it is why the document's own §5.4 lists it as a PASS. Charging it only for pits
   under-counts by the ratio of pits to cells.
2. **The heap costs more compares.** A binary heap over `n` items costs about `log2(n)` compares per
   push and per pop. For a patch, `n = 147 456`, `log2 n = 17.2`, so a push-and-pop pair is about 34
   compares, not 19. For a global 6 x 64^2 grid it is about 30.
3. **Pointer jumping cannot use a fixed round count.** Pointer jumping resolves a chain of length `D`
   in `ceil(log2 D)` rounds. A basin on a 64-cell face can be thousands of cells long, so `D` is in the
   thousands and the count is 11 to 13, not 4. §5.4 and E1 both require the round count to be **a
   recipe constant**. A constant that is too small leaves cells with the wrong basin label, which
   changes the answer, not only the cost. A constant that is large enough is 3 times §4.1's charge.

§4.1 says 160 ns "is the number M-L1 must replace", which is the right posture. But §1 says
*"doubling the estimate is the conservative repair"*, and by these three items it is not conservative.
The honest statement is that the kernel is UNBOUNDED above until M-L1 runs, and every table in §5.3 and
§5.5 inherits that.

---

## Weaknesses

### W1 — MEAN-PINNING CONSERVES MASS, SO THE EROSION REMOVES NOTHING; ISOSTASY IS DEFINED AND NEVER USED

C2 forces each parent block's mean back to the parent's height after every iteration. The net effect
over a patch is exactly zero change of volume: whatever the rivers cut out, the shift puts back.
Erosion is named for **denudation** — the removal of mass. Under C2 the fine levels do not denude; they
redistribute inside a 1 285 m box.

Two consequences a geomorphologist would name at once:

- A gorge 300 m deep must be paid for by raising its own shoulders. §2's glossary states it as a
  feature: *"Level 3 puts a 300 m valley and a 300 m shoulder inside it, and their mean is still
  1 200 m."* That is not what a river does to a landscape.
- **Isostasy** is defined in §2 (*"Take mass off the top and the crust rises"*) and then appears in NO
  pass of §5.4, NO row of §4.1's op count, NO byte of §5.2's build record and NO measurement in §13.
  It is a word in a glossary. `03_erosion_rivers.md` requires it as a real pass (`ISOSTASY_EVERY = 5`).

### W2 — §2's TECTONIC-UPLIFT EXAMPLE IS WRONG BY ABOUT FOUR ORDERS OF MAGNITUDE

§2: *"Where two plates converge, the macro map raises the ground **by a few millimetres each
iteration**, and after **a hundred iterations** a range stands there."*

```
   3 mm x 100 iterations = 0.3 m
```

That is a kerb, not a range. The home planet's relief is 14 304 m. For uplift and incision to reach a
steady state inside 128 iterations, the uplift must be of order `relief / I = 112 m` per iteration.
The example is off by about 4 x 10^4.

The number matters, because §5.3 folds the uplift pass into the cost as "about 20 ns per cell ...
under 0.4 %, folded in as a rounding". If uplift is real work at a real rate, the plate field, the
convergence test and the uplift rate all need their own op counts.

### W3 — THE STREAM-POWER EXAMPLE PROMISES DEPOSITION THE CHOSEN LAW CANNOT PRODUCE

§2: *"A flat cell with the same catchment **lays a floodplain** instead."*

E5 fixes the law as `K * sqrt(A) * S` — a **detachment-limited** stream power law. That law only
removes rock; its rate goes to zero as the slope goes to zero. It never DEPOSITS. A floodplain, an
alluvial fan, a delta and a valley fill all need a transport-limited term (a sediment flux carried
downstream and dropped where the capacity falls). No such term appears in §4.1, §5.2 or §5.4.

The reference picture's valley floor is a floodplain with a river on it. The document prices a
mechanism that cannot make one, and its own glossary says it can.

### W4 — §8.3's RULE READS RELIEF ONLY, SO A PLANET LOSES ITS CONTINENTS BEFORE IT LOSES ITS BUMPS

§8.3 and D-C12: a body's macro map is built only when *"the body's own RELIEF subtends more than one
drawn pixel"* — 13 100 km for the home planet, and at that distance the planet is still 469 pixels
across.

The macro map carries the CLIMATE, and the climate decides the biome, and the biome decides the colour
(§10.1). Colour is legible at one pixel; relief is not. Under this rule a planet 20 000 km away is a
uniform ball 300 pixels wide, with no deserts, no ice caps and no forest belts, and it grows them when
it crosses 13 100 km. That is an arrival pop, and the seam taxonomy names it.

The rule needs a second arm on the climate's own contrast, or the climate must leave the macro map's
staging rule.

### W5 — `BOOT_CEILING = 2 s` HAS NO SOURCE

§5.3 and §8.4 both justify it as *"a defensible ceiling is 2 s on one core, because **a pod is not
promised more** (§4.3)"*. §4.3 is about the absence of a thread pool in `vd-terrain`. It contains no
statement about a pod, a CPU request, a Kubernetes limit or a shard's budget. I found no other source
in the document.

In a document whose §1 says *"No number in this document is a result until §13's table has a green row
beside it"*, the two constants that decide the world's resolution must carry a source. One of them
does not. D-C4 correctly puts both to the owner, so the cure is one sentence: state it as UNSOURCED and
owner-owed.

### W6 — THE WORLD'S SHAPE BECOMES A FUNCTION OF A BENCHMARK

§8.4: the chooser takes the finest `M` with `6 * M^2 * 2M * KERNEL_NS <= BOOT_CEILING`, runs ONCE
OFFLINE, and rides `GENERATOR_VERSION`.

That is drift-safe (no host evaluates it at run time), and §15 claims the no-magic-numbers gate on it.
But `KERNEL_NS` is a measurement on one machine, and `BOOT_CEILING` is an operational number. So the
resolution of every planet's landforms — how big a valley the home planet gets — is decided by how fast
one Mac ran one benchmark in 2026. §8.4 says as much: *"If M-L1 measures 50 ns, `M = 128` passes and
the pyramid starts one level finer."*

The no-magic-numbers rule says world params are seed-derived and entity props are per entity. A
resolution chosen by a benchmark is neither. The honest framing is that this is an operational
parameter that MOVES THE WORLD, so it must be frozen once, recorded in the tag, and never re-derived —
which is close to what §8.4 says, but §15 claims the gate is passed rather than that a tension is
managed.

### W7 — §12 CALLS THE KERNEL "STRAIGHT-LINE CODE OVER ARRAYS" AFTER ADDING A HEAP AND A POINTER CHASE

§12's coverage note: *"The erosion kernel is straight-line monomorphic code over arrays, which is the
easy case for 100 % region and branch coverage."*

Revision 2's own §4.1 added a **priority flood driven by a heap** and a **basin labelling by pointer
jumping**. Neither is straight-line: a heap has sift-up and sift-down loops with data-dependent
branches, and pointer jumping has a per-round convergence structure. The four named fixtures (a closed
pit, an all-sea grid, the face edge and corner, a partial parent block) do not obviously cover a heap's
arms.

HR5 is 100 % of REGIONS and BRANCH SIDES. The estimate "one afternoon of tests per pass" is the only
number, and it now covers two passes that are the hard case, not the easy one.

### W8 — THE CLIMATE STAYS GLOBAL AT 20 560 m, WHICH CANNOT RESOLVE THE PICTURE'S OWN BIOME EDGE

§10.1 keeps the climate on one global 6 x 256^2 grid, *"finer than the erosion's level 0 because a rain
shadow needs a range resolved"*. A cell is 20 560 m.

The reference picture shows a forest mass that ENDS at a ridge, and the owner's task asks for
believable biomes. A rain shadow at 20 km resolution puts the forest-to-desert edge inside one cell, so
the boundary the pilot walks over is a bicubic ramp about 40 km wide. §7.2 says the lapse rate is
applied to the column's own height, which recovers the SNOW line sharply, but moisture is the quantity
that draws the forest edge, and it is not refined.

No level of the pyramid refines the climate, and §13 has no measurement of the biome-edge width. It is
the cheapest believability item in the domain and it is the one left global.

### W9 — THE PATCH CACHE IS NOWHERE DESIGNED, AND IT IS THE CRATE'S FIRST MUTABLE STATE

§2 defines a patch as *"computed on demand from its own ADDRESS, then **cached**"*. §17 item 8 asks
whether the macro pyramid belongs in `vd-terrain` at all, because *"it is a per-body field with a
build phase and a cache, which is a different shape from the pure per-address functions the crate
holds today"*.

`vd-terrain` today is a pure library with one dependency (`Cargo.toml`, verified). A cache brings
mutable state, a residency policy, an eviction order and a lifetime, into the Tier-A crate that must
hit 100 % branch coverage and must link as a `staticlib` into another engine. None of that is designed
or priced here, and B2 shows the residency is the number that decides whether the shape fits at all.

---

## Notes

- **N1.** §9.1 mixes chunk counts inside one block: `12.3 s` is the VISIBLE HALF (5 292 chunks),
  `2.03 GB` is the WHOLE BODY (10 584 chunks x 4 055 vertices x 47.24 B). Both are right; the block
  reads as one calculation.
- **N2.** §8.1's table prints a level-3 patch as `0.26 MB` in a column headed "shipped raw, **2 B**".
  At 2 B a 256^2 patch is 0.131 MB; 0.26 MB is 4 B. The other rows in that column are consistent at 2 B.
- **N3.** §5.5 prices shape B's raw ship at `1.25 MB` (the 3 B read record), while §8.1 prices the same
  two grids at `0.84 MB` (2 B). The kill is unaffected; the numbers do not reconcile.
- **N4.** §10.1 derives the climate grid's base height from `69 ns per column`, which §4.1 obtained
  from the rung-11 column pass with **three** octaves. §10.1 evaluates **five** (its own Nyquist rule)
  or **four** (the corrected rule of D3). The 27 ms is an under-count.
- **N5.** §5.5 prices shape A at `I = 100`, while §5.3's own law gives `I = 2M = 40 960`. §5.3's table
  leaves the cell blank, which is honest; §5.5 and §1 print 11.2 hours as "the honest figure" without
  noting that the document's own law makes it 4 580 hours. The kill stands either way.
- **N6.** §10.2 cites *"the client MAY NOT derive it (**SL10 c.3**)"*. Clause 3 of V1 is "NO DRIFT is a
  MEASUREMENT". The clause meant is 7 ("the client never derives a pose, a velocity, or any state") or
  1 ("never a function of time"). §11's use of clause 3 is correct.
- **N7.** §4.2's band diagram ends the octave table at 30 m and labels the band under 30 m "nothing".
  `SHORT_WAVE_M = 30` is the loop's BOUND (`while wave_m > 30`), so the finest octave the home planet
  draws is 48.8 m. The band from 49 m to 30 m carries nothing either, and the table's own row says
  "800 m - 49 m".
- **N8.** §5.5 calls the climate "5 passes"; §10.1 lists six rows (base height, insolation, lapse rate,
  wind, orographic moisture, ocean distance).
- **N9.** §8.4's chooser *"walks the WHOLE DIVISORS of `N`, **coarsest first**, and takes the **finest**
  `M`"*. Those two halves describe opposite scans. The intended rule is clear; the sentence is not.
- **N10.** Terms used before they are explained, against §2's own promise: **pointer jumping** (§4.1,
  §5.4), **bicubic** (§5.6, §6.1, §7.2 — bilinear is contrasted with it but neither is defined),
  **Nyquist** (§5.3), **topological order** and **donor** (§5.4), **stencil** (§4.1, §5.4). §2 explains
  twelve harder terms well, so the omission reads as an oversight.

---

## Checked, and sound

These I tried to break and could not.

1. **The ladder arithmetic.** I re-ran `Ladder::for_radius(3 351 154, ...)` in `python3`: `n_ideal =
   5 263 980`, `top_rung_for = 11`, `q = 2 570`, `N = 5 263 360`, `rungs = 12`, radius
   `3 350 759.045 m`. Every figure the document prints matches.
2. **`N = 2^12 * 5 * 257`**, and the rebuttal of revision 1 (`/1 024 = 5 140`, `/2 048 = 2 570`,
   `/4 096 = 1 285`) is exact. The four pyramid levels 64, 512, 4 096, 20 480 all divide `N`.
3. **The cube-sphere metric replay.** I re-derived it from `K1 = pi/4`, `K2 = 0.15`,
   `K3 = 1 - K1 - K2` (`crates/seed/src/bend.rs:24-30`) with the area element
   `W'(a)W'(b)/(1+W(a)^2+W(b)^2)^{3/2}`: max 0.616850, min 0.432940, **ratio 1.4248**, six-face integral
   12.5663712 against `4 pi = 12.5663706`. The document's 1.425 and its 1.2 x 10^-6 agreement both
   hold. Rule E8 is correctly motivated and correctly measured.
4. **The octave amplitude replay.** From `k_rough = 0.4683` (which the ratio 3 562 / 7 606 implies) the
   sum of all fourteen amplitudes is 14 304.6 m, matching the relief `terrain_cost` prints; octave 10
   is 8.26 m and octaves 10 to 14 sum to 15.19 m. §4.2's "15.2 m across the five octaves under 800 m"
   is right, and so is its conclusion that this domain does not fix the cliff band.
5. **`vd-terrain` has exactly one dependency** (`vd-seed`) and ships as `crate-type = ["lib",
   "staticlib"]`. §4.3's deletion of the fourteen-thread column, and D-C11, are correct and correctly
   argued.
6. **§10.1's climate blocker is real.** `crates/physics/src/worldgen/generate.rs` uses `ln` (`:571`,
   `:598`, `:633`), `powi` (`:1727`) and `powf` (`:1880`), so it is unfenced, and `vd-terrain` may not
   depend on it. D-C9 is the right shape.
7. **The float fence bans `powf`, `powi`, `min`, `max` and `mul_add` by name**
   (`crates/terrain/clippy.toml`), and allows `sqrt`. E5's `m = 1/2, n = 1` is therefore lawful, and
   the `terrain-fence-control` control exists as described.
8. **`terrain-legs` is NOT in `just gate`.** `justfile:408` lists `terrain-pin`,
   `terrain-link-scan` and `terrain-fence-control`; `terrain-legs` is at `:756` with its own comment
   saying "Not in `gate`: it needs Docker". §11's retraction is correct. `terrain-pin` does run three
   builds (debug, release, `target-cpu=native`).
9. **The river carve stops at rung 5.** `tubes_carve_at` is `cell_m <= tube_radius_m * 2`
   (`carve.rs:36-38`); a 15 m radius gives 30 m, so rung 4 (16 m) carves and rung 5 (32 m) does not.
10. **`tube_region_m = 512`** (`body.rs:226`), a constant, so §5.1's family-2 precedent is real: the
    crate already indexes a whole-metre lattice by division and floor and passes the golden legs.
11. **Every line citation I checked is right**: `CHUNK_EDGE` at `ladder.rs:21`, `TOP_RUNG_CHUNKS` at
    `:23`, `RUNG_MAX` at `:27`, `POLE_AXIS` at `height.rs:35`, `biome_at` at `:40`, `HALO` at
    `lattice.rs:48`, `EXTRACT_BUDGET_US = 8_000.0` at `terrain_cost.rs:41`, `GENERATOR_VERSION` at
    `tag.rs:19`, the three world-identity call sites, and `compose.rs`'s rank 1 / rank 2.
12. **The cell is 2 bytes today** (`Stratum` plus `gap: i8`, `chunk.rs:63-69`), so §15's record row is
    accurate and correctly marks the 12-byte record as PROPOSED.
13. **The seam table exists and is gated exhaustively** (`crates/seed/src/seam.rs`: 24 directed edges,
    generated from the face basis, pinned by a digest, "an exhaustive test on a small face proves it for
    every body (ruling V6 A1, `G-MAPPING-TABLE`)"). E7's claim that it reuses that mapping is true, and
    the seven-neighbour count at a cube corner is right (3 on the own face, 2 across each of the two
    edges).
14. **`6 x 3 968^2 = 94 470 144`** is right, and so is the rung-11 body arithmetic
    (`ceil(2 570/62) = 42`, `6 x 42^2 = 10 584`, half 5 292, `x 2.33 ms = 12.3 s`).
15. **The owner's budget quote is exact** against `owner_decisions_2026-09-07_voxels.md:442-443`.
16. **The shape-A kill** (`6 x 20 480^2 = 2 516 582 400` cells, `x 100 x 160 ns = 40 265 s`, 7.55 GB at
    3 B, 73 GB at 29 B) is arithmetically right, and it is the right thing to kill.
