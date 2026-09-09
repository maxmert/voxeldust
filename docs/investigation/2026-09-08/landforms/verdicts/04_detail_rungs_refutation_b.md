# REFUTATION B (round 2) — DOMAIN 04, `04_detail_rungs.md` revision 1

**Lens:** believability, cost, the boot, and the owner's own sentence.
**Date:** 2026-09-08. **Method:** every citation re-opened in the source tree; every number re-derived;
DOMAIN 03's current text read against DOMAIN 04's contract for it.
**Round 1 is not repeated.** The 46 answers in §14 were checked and they hold, with one exception
recorded below (F12). Every finding here is NEW to revision 1.

Revision 1 is a large step up. It states the budget as FAILED, it derives `Lip(T)`, it withdraws three
false proposals and it names its estimates. The findings below are what survives that improvement.

---

## BLOCKERS

### F1 — The gravity relief ceiling REFUSES every body smaller than about 313 km, and the home planet's band grows by half

§4.3 replaces the relief law with `relief_ceiling_m = crustal_strength_Pa / (rock_density × surface_gravity)`.
§11's no-magic-numbers row deletes the `200..12 000` m clamp with it. §5.2 item 1 then feeds that ceiling
to the ladder band, "a per-body number the seed and the body's facts decide".

For a body of uniform density the surface gravity is `g = (4/3)·π·G·ρ·R`, so the ceiling is

```text
   ceiling(R) = S / (ρ · (4/3)πGρR) = 3S / (4πGρ² R)      it GROWS as the body SHRINKS
```

COMPUTED at `S = 200 MPa` and `ρ = 2 700 kg/m³`:

| Body | Radius | Surface gravity | Relief ceiling | Verdict |
|---|---|---|---|---|
| The home planet | 3 351 km | 3.44 m/s² | 21.5 km | accepted |
| A 500 km moon | 500 km | 0.377 m/s² | **196 km** | accepted, and absurd |
| The break-even body | **313 km** | 0.236 m/s² | **313 km** | the ladder refuses it |
| A 100 km moon | 100 km | 0.0755 m/s² | **981 km** | the ladder refuses it |
| §4.3's own 10 km asteroid | 5 km | 3.77 × 10⁻³ m/s² | **19 600 km** | the ladder refuses it |

`Ladder::for_radius` returns `None` when `crust >= surface` (`crates/seed/src/ladder.rs:84-86`), and
`crust_m = relief + strata + caves + 64` (`body.rs:234`). So under D4-13 **every body under about
313 km of radius has no ladder at all** — no moon, no asteroid, no small ice body. The threshold moves
to 384 km at the top of the proposed strength range (300 MPa), so the seed decides whether a given moon
exists.

This refutes §4.3's own worked example by name. §4.3 says a 10 km asteroid "gets `n = 2` octaves ... and
it is one world with no variant (SL5)". That asteroid cannot be built. The octave law was made total and
the relief law was made partial in the same section.

**It also costs the home planet.** Today `cells_in_band(0) = 29 105` and one radial column holds 470
chunks (MEASURED, `slice_05_generator.md:387-389`). With the relief at 21.5 km instead of 14.3 km the
band grows to about 43 000 cells, so a radial column holds about **700 chunks, 49 % more**, and the
column's measured 417 ms grows with it. §9 prices no part of this.

**What is missing is a size term.** Real relief is bounded by the body as well as by the rock: nothing
stands 39 % of its own radius high. The deleted `0.4 % of the radius` was crude, but it was the term
that made the law total. The recommendation must be `relief = min(strength ÷ (ρ·g), a share of the
radius)`, and the share must itself have a source. Until then D4-13 and D4-14 may not be taken.

### F2 — The erosion anchor `S_e = 2^(rungs+1)` cannot tile a cube face, and DOMAIN 03 says so by name

§3.2 requirement 1 and §4.3 anchor the whole octave table to `S_e = 2^(rungs+1) m = 8 192 m` on the home
planet, and §3.2 calls it "a RUNG of the ladder, never a number of metres".

Two facts refute it.

1. **8 192 m is not a rung.** The home planet has 12 rungs, 0 to 11, with cells `1 << rung`
   (`ladder.rs:44-46`), so the coarsest cell is 2 048 m. `2^(rungs+1)` is two rungs ABOVE the top rung,
   which does not exist.
2. **8 192 rung-0 cells do not divide the face.** `N = 5 263 360 = 2¹² · 5 · 257` (COMPUTED; `n` is
   snapped only to a multiple of `2^top = 2 048`, `ladder.rs:59-65,82`). So `N / 8 192 = 642.5`. The
   node lattice leaves a HALF cell at every face edge. Two faces then meet with two different node
   spacings, the bicubic stencil reads different nodes on the two sides, and the result is a height
   crease along a cube edge — the exact failure §3.1 requirement 2 forbids by name.

**DOMAIN 03 already found this and chose otherwise.** Its §1 item 1 reads: *"640 nodes per face edge,
8 224 m per node, 2 457 600 nodes ... 640 divides `N` exactly, 642 does not"*. So the real spacing is
**8 224 m**, not 8 192 m, and the real node count is 2 457 600, not the 2 472 984 §3.2 quotes.

Everything DOMAIN 04 builds on the wrong anchor falls with it:

- `λ₀ = 2·S_e = 16 448 m`, not 16 384 m. Halving eleven times reaches 16.06 m, not 16 m.
- `n = log2(16 448 / 16) + 1 = 11.006`, which is not a whole number. **The identity `n = rungs − 1`
  fails**, and that identity is the whole argument in §4.3 and §11 that the landed test
  `every_rung_sums_strictly_fewer_octaves_than_the_rung_below_it` (`body.rs:355-365`) survives.
- **`t` is no longer dyadic.** §4.2's no-drift argument rests on the offset inside a node cell having a
  power of two below the line. The real spacing is 8 224 rung-0 cells, and `8 224 = 2⁵ · 257`, so `t` is
  an odd multiple of `1/16 448`. That denominator is not a power of two, the Catmull-Rom weights are not
  exact, and §4.2's central sentence — "multiplying by it rounds the same way on every machine" — is
  withdrawn by arithmetic.
- D4-3 asks the owner to approve a world-epoch change on the wrong number.

**Example in the game's words.** The pilot walks along the cube edge on the home planet. On the +X side
the erosion node under his boots sits 8 192 m from its neighbour; on the +Y side the last node cell is
only 4 096 m wide. The two faces hand the bicubic different neighbours, and the ground steps where he
crosses.

### F3 — The terrace is DISCONTINUOUS, so `Lip(T) = 1.413` is not the constant, and the 1.26-pixel headline does not follow

§4.5 states two rules that cannot both hold.

```text
   the band index    floor((r - datum_m) / band_thickness) mod m     tops are ONE thickness apart
   the terrace       h + q·H·(nearest_band_top - h) · f(|h - top| / band_thickness)
   the falloff       f(u) = 1 - u²(3 - 2u),   f(0) = 1,  f(1) = 0
```

If the band tops are one thickness apart, then `|h − nearest top| ≤ thickness/2`, so **`u` never passes
0.5**. Two consequences follow, both fatal to §5.2.

1. **The quoted derivative is measured outside the reachable range.** The document's own expression is
   `1 − q·H·(1 − 9u² + 8u³)`, whose extreme is at `u = 0.75` and gives 1.413 at `q = 0.6` (re-derived,
   correct). At the largest reachable `u = 0.5` the value is `1 + 0.25·q·H = 1.15`. So the derived
   constant is taken at a point the function cannot reach.
2. **The switch between two tops is a JUMP, not a slope.** Just below the midpoint the nearest top is the
   one below, and the pull is `q·H·(−t/2)·f(0.5)`; just above it, `q·H·(+t/2)·f(0.5)`. With
   `f(0.5) = 0.5`, `q = 0.6`, `H = 1` and `t = 40 m` the height JUMPS by

```text
   2 · q · H · (t/2) · f(0.5) = 0.6 × 40 × 0.5 = 12 m
```

So `Lip(T)` is **unbounded**, not 1.413. §5.2's bound `|h(0) − h(L)| ≤ Lip(T)·Σ|a_i| + |R_0 − R_L|` is
void wherever the coarse rung and rung 0 fall on opposite sides of a mid-band radius — which is common,
because the octaves are exactly what moves them. At rung 1 the alpine octave bound is 0.22 m; the terrace
can turn that into 12 m, which is **6 cells of the rung-1 cell (2 m)**, six pixels by the document's own
conversion, at 1 740 m from the pilot. SL8 fails visibly, and §0's headline sentence — "the worst rung
change on the whole ladder moves the surface by 1.26 pixels" — is not supported.

**The cure exists and must be stated:** the falloff must reach zero before the next top (`u` measured
against half the spacing, or the terrace applied only at HARD band tops that are spaced two thicknesses
apart). Either changes `Lip(T)`, §5.2 and every pixel figure, so it is a decision, not an edit.

**The terrace also has no rung rule.** Every other term got one: the octaves get `C` cells per
wavelength (§4.3), the channel gets `C` cells of width (§4.6), the undercut gets `C` cells of size
(§6.2). The terrace has bands of 8 to 40 m and it runs at "every rung" (§0's fold diagram), including
rung 6 with 64 m cells and rung 11 with 2 048 m cells. A 40 m bench on a 2 048 m cell is 0.02 samples per
band: it aliases, and it disagrees with rung 0 by its own step size at every coarse rung.

### F4 — §3.2 and §11 state DOMAIN 03's solve wrongly, so the SL6 row is false and the client's boot is unpriced

§3.2 opens: *"DOMAIN 03 runs the erosion ONCE per body on a coarse grid, **derived on both hosts**"*.
§11's SL6 row concludes: *"This domain adds NO new crossing ... the erosion correction is derived once
on both hosts"*. §11's SL10 row adds: *"there is no stored map on one side and a derived map on the
other, which would be a port and is forbidden"*.

DOMAIN 03's own text says the opposite, three times:

- *"A macro solve, per body, cached, **computed on the server**"* (§1 item 1).
- *"**The body's physical facts are SHIPPED, never recomputed on the client.** The client links no motion
  crate (ruling S7-1), so it cannot compute the gravity, the surface temperature, the obliquity or the
  spin the solve needs. The owning realm states them ONCE, on change, as quantised integers beside the
  surface statement ... **That is an SL6 ask** (§12 D3)"* (§1 item 7).
- *"derived from facts the owning realm states rather than from arithmetic no client can run"* (§1
  closing sentence).

So the law rows in §11 rest on a false statement about the neighbour this domain reads. Three
consequences the owner must be given:

1. **SL6 is not PASSED.** The lane DOMAIN 04 depends on carries new data across a realm boundary, and
   DOMAIN 03 records it as an ask, still open.
2. **The client's boot is unpriced.** The correction is a **9.83 MB artifact per body** (DOMAIN 03 §1
   item 2). Before the client draws its first chunk of the home planet it must hold that artifact, and
   the pilot who flies to a second planet must hold a second one. §5.4 prices the vista's mesh at about
   810 MB and the vista's time at 1.0 s on 14 threads, and prices the artifact at nothing.
3. **If the artifact were derived on the client instead**, the cost is a full erosion solve — a priority
   flood and a downstream sweep over 2 457 600 nodes with about 125 MB of transient state (DOMAIN 03
   §4) — before the first chunk. Neither path is costed here, and the document must say which one it
   assumes.

### F5 — Below the erosion grid there are still NO LINES, and that is the whole vista the player walks in

§0 states the real cause of "no orienters" correctly, and it is the best sentence in the document: *"an
isotropic sum of band-limited noise has no lines ... no ridge crest, no coastline, no scarp, no basin
rim and no valley axis, at any scale, because every direction is statistically the same."*

Then §4 fixes the AMPLITUDE of that isotropic sum and leaves it isotropic. Count what carries a line at
each scale on the home planet:

```text
   scale            what carries a LINE under this proposal
   ---------------  ---------------------------------------------------------------
   > 8 224 m        the macro stack (D02) and the erosion solve (D03): ridges, scarps, basins   YES
   8 224 m          the D8 channel graph: ONE valley axis per node                              YES
   8 224 m - 128 m  six damped ISOTROPIC octaves, plus a terrace at fixed radii                 NO
   128 m - 16 m     three damped ISOTROPIC octaves                                              NO
   16 m - 1 m       one biome octave at 8 m, isotropic                                          NO
```

The reference picture's carved valleys, spurs, gullies and ridge lines sit between about 100 m and 3 km.
That whole band is isotropic noise with a damping term. The pilot standing on the ridge sees the trunk
valley the solve cut, and between it and his boots he sees the same crumpled sheet the owner already
rejected, only steeper.

**DOMAIN 03 says so in numbers, and DOMAIN 04 does not read it.** DOMAIN 03 §1 item 3: *"the recipe
spends ... 47 m at 3 km, 23 m at 1.6 km. The reference picture needs **300–600 m** there."* DOMAIN 04's
own §4.8 alpine column gives 113.957 m at 2 048 m and 59.548 m at 1 024 m, so about **85 m at 1.6 km** —
still three to seven times short of what the sibling domain says the picture needs, and the sibling's
cure (a slope spectrum peaked at the valley spacing) is not in DOMAIN 04's table.

**And DOMAIN 03 already proposes the missing machinery**: *"a detail term — scree, ridged crests,
flow-warped gullies — whose slope comes from an ADDRESS-DEFINED stencil"* (§1 item 4). DOMAIN 04 proposes
damped isotropic fBm for the same band. Two domains, one spectral band, two different laws, and neither
document says which one wins. §3.2 requirement 5 flags only the smaller overlap (DOMAIN 02's L4 against
DOMAIN 03's carve).

A geologist reads a real mountain belt by its DRAINAGE DENSITY — a channel every 100 to 500 m, spurs
between them, a dendritic pattern that repeats down to the metre. This proposal has one channel every
8.2 km and nothing between. That is the finding the owner's sentence *"the surface will not be
interesting enough"* actually names, and revision 1 answers it for the amplitude only, exactly as it
admits for the 1–16 m band in §4.7 but not for the 100 m–3 km band.

---

## DEFECTS

### F6 — D4-13 needs a NEW DATUM to reach the client, and the local formulation is not looked for

§4.3 says the surface gravity "becomes an INPUT to `BodyDefinition::from_seed` ... exactly as the look
radius is an input today".

The two are not alike. The client builds the body from the seed and the look shell it ALREADY receives:

```rust
// crates/client/src/chunks.rs:275-279
(FrameRef::PlanetCentered { planet_seed }, Boundary::Shell { r }) => {
    BodyDefinition::from_seed(planet_seed, *r)
}
```

Gravity is not in that pair. It comes from a mass the physics forest draws through `powf`
(`taxonomy.rs:640-644,653-655`), which the fence forbids and which the client cannot run (ruling S7-1:
the client links no motion crate). So D4-13 REQUIRES a new field on the surface statement — an SL6 ask
and a wire change — and §11's "This domain adds NO new crossing" is wrong for the document's own
recommendation.

Two things are owed. First, say the ask plainly, and note that DOMAIN 03 already opened the same lane
(its §12 D3), so ONE ask should carry both. Second, look for the local formulation SL6 asks for first:
for a rocky planet the forest derives the mass FROM the radius through the mass-radius law, so the
gravity is a function of the look radius alone. A fenced integer table over that law would need nothing
to cross at all. The document does not consider it.

### F7 — The periodic radial stack unbounds `max_depth_m()` and KILLS the below-skip; §4.5 claims the opposite

§4.5 item 3 says the sediment band leaves `StrataTable`, so `max_depth_m()` becomes `topsoil + subsoil`
and the crust falls by 20 to 80 m.

The code's contract for that number is *"The deepest metre the strata change at; **below it a cell is
bedrock without a lookup**"* (`strata.rs:165-169`). A stack that REPEATS in radius changes substance at
every band boundary, forever downward. So `max_depth_m()` has no value at all under the proposal; it does
not shrink.

The consequence is measured, not argued. The below-skip writes a whole chunk without a cell pass only
when *"every cell is bedrock"* (`chunk.rs:14-17`). Today one radial column at rung 0 holds 470 chunks and
**463 of them are skipped**, and the column still costs 417 ms (MEASURED,
`slice_05_generator.md:387-389`). Under a periodic stack no deep chunk is uniform, so all 463 must run
the cell pass: 463 × 238 328 cells. At the measured surface cell-pass cost of 579 µs per chunk
(`slice_05_generator.md:381`) that is about **270 ms more per radial column**, a 65 % rise, and §9 carries
none of it. It is also a change to a landed invariant that a test asserts
(`chunk.rs:14-17`, "A test generates a skipped chunk both ways and compares cell for cell").

### F8 — D4-14 moves the ladder band; DOMAIN 03 has already DELETED the same proposal

§12 D4-14 says: *"The ladder BAND moves, twice ... **Accept it now.** Both are free today."*

DOMAIN 03 §1 item 10 says: *"**The band does not move.** `Z` replaces the coarse octaves inside the SAME
envelope ... The channel's cut fits inside the 64 m of slack the crust term already carries
(`crates/terrain/src/body.rs:234`). Revision 1's proposal to grow the crust is **DELETED**: it would have
shifted every cell index on every body."*

The two domains hand the owner opposite recommendations on the same world-epoch decision, and neither
says so. DOMAIN 04's §14 reports the neighbours as reconciled. The owner must be told there is one
question here, not two, and that it decides whether every stored edit on every planet re-addresses.

### F9 — The "44.3° without the damping" figure does not reproduce, and the RMS method is never stated

§4.8 and D4-5 rest on one comparison: 33.8° of added RMS slope with the damping, 44.3° without it, and
the geologist's reading that 44° puts the median above the angle of repose.

The 33.8° reproduces. Each row's "max slope" is `atan(2π·a/λ)` (checked: 600 m at 16 384 m gives 13.0°;
358.658 m at 8 192 m gives 15.4°). Adding the tangents in quadrature and halving the sum of squares —
the sinusoid's RMS — gives `atan(sqrt(0.8735/2)) = 33.5°`, which is §4.8's number.

The 44.3° does not. Removing the damping means the cascade keeps its stated gains, 0.62 above the 128 m
break and 0.42 below it, from `a₀ = 600 m`. That table is 600, 372, 230.6, 143.0, 88.6, 54.9, 34.1, 21.1,
8.87, 3.73, 1.56 m. By the SAME method the added RMS is **56.7°**, not 44.3°.

So one of three things is true, and the document must say which: the undamped table is not the stated
gains, the RMS method is not the one that reproduces 33.8°, or 44.3° is a leftover. The direction of the
argument survives either way — the damping is load-bearing — but a number the owner is asked to accept a
world epoch on must reproduce from its own stated inputs. **The RMS method itself is nowhere in the
document**; a reader cannot check the headline without guessing it.

### F10 — The +0.60 ms undercut row uses the cost the document forbids itself to use

§9.1 strikes the cave unit cost and marks it **NOT AVAILABLE**, and §6.2 says of the undercut term: *"it
costs what the caves cost, and that is the ONE number this document may not take from `2.83 − 1.98`"*.

§9.2 then writes `the undercut lattice at stride 4 ... +0.60 ESTIMATED` with no other source, and the
whole budget verdict turns on it: 9.60 ms with it, 9.00 ms with the first lever, which is the row that
deletes it. A number the document declares unavailable cannot then carry the budget row. Either M4-11
runs first, or the row must read UNKNOWN and the total must be given as a range.

The same objection is weaker but real for `+0.82` (the channel carve) and `+0.29` (analytic derivatives,
"+80 % on the 9 that remain"): neither the 80 % nor the 0.82 has a stated derivation, and together they
are 1.11 ms of a 1.60 ms overrun.

### F11 — §9.2 counts the columns two ways

The octave unit cost is stated per 3 844 columns (§9.1, `62 × 62`), and the macro stack row uses
`180 ns × 3 844`. The slope-correction row uses `20 ns × 4 096`. The halo makes the real count
`64 × 64 = 4 096` and the halo costs a measured +35 % (`slice_06_extractor.md:354`). Using 3 844 for the
macro stack understates the largest added row by 6.6 %, which is 0.05 ms at the low end and 0.09 ms at
the high end. Small, but the budget is short by 1.60 ms and every row must be counted the same way.

### F12 — The pixel gate is really "the bound in cells", and 870 m contradicts the code

§5.2's conversion uses two numbers from the same 720-row reference: a switch distance of `2^L × 870 m`
and a pixel of `0.785 / 720 = 1.091` mrad. They cancel almost exactly, so every figure in the table is
`bound ÷ cell_m(L)` to within 5 %:

```text
   rung 6:  73.60 / 64       = 1.15    the table says 1.21
   rung 9:  611.03 / 512     = 1.19    the table says 1.26
   rung 11: 1 965.62 / 2 048 = 0.96    the table says 1.01
```

That is GOOD news and the document should say it: the gate is **resolution-independent**, because a
finer screen moves the switch distance out by the same factor it shrinks the pixel by. As written, a
reader assumes the 1.26 px becomes 2.5 px at 1440 rows, and refuses the design for the wrong reason.

The 870 m itself is wrong against the code. The drawable factor is `cot(θ/2) = 1 738` with
`drawable_theta_min_rad() = 1.1506 × 10⁻³` (`crates/core/src/geometry.rs:1156-1173`), and a 1 m cell is
drawable to about 1 505 m. The previous investigation's document 08 corrected this by name — *"B4 the
870 m radius is a person's number | WRONG"* (`08_render_seam_seamless.md:1082`). Using the person's
number here is conservative by 1.7×, so no conclusion moves, but a document that gates a world epoch may
not quote a figure its own investigation has already retracted.

---

## WEAKNESSES

### F13 — The relief ceiling is a one-parameter fit, presented as a validation

§4.3 says: *"Earth at 9.81 m/s² gives 7.6 km against Everest's 8.8 km; Mars at 3.71 m/s² gives 20.0 km
against Olympus Mons' 21.9 km ... That is a law with a measurement behind it."*

Both numbers come from choosing `S = 200 MPa`, and `S` is then declared a seed draw over 100 to 300 MPa
(§4.3). At the ends of that range the home planet's ceiling is 10.7 km and 32.2 km — a factor of three.
So the fit is one free parameter matched to two points, and the parameter is afterwards allowed to move
three-fold. The RATIO does test something (2.64 predicted against 2.49 observed, using 21.9/8.8), and
that sentence should carry the claim alone.

A geologist would add two things the document does not. Everest's height is set by the balance of
tectonic uplift against glacial and fluvial erosion, not by basal rock failure; the strength bound is an
upper limit that Earth does not reach. Olympus Mons is a volcanic construct on a thick, cold lithosphere.
Naming both as evidence for one crustal-strength law will read as a stretch to anyone who knows them.
State it as an ORDER-OF-MAGNITUDE ceiling, which is what it is.

### F14 — "At most nine channel segments" holds only while a chunk is small against an erosion cell

§3.2 requirement 2 and §4.6 property 1 state the nine-segment gather as a general property, and §9.2's
second lever prices the whole budget rescue on it (`−0.62 ms`).

It is true at rung 0: a chunk is 62 m and an erosion cell is 8 224 m. It stops being true where a chunk
approaches a cell. A chunk is `62 × 2^L` metres, so it passes 8 224 m at **rung 7** (7 936 m) and spans
15.5 erosion cells at rung 11 (126 976 m). §4.6 property 3 keeps the carve alive at a rung while the
channel's own width is at least `C` cells, so a 2 km valley is still carved at rung 6 and a wider one
above it. Two things are owed: the window must be stated as a function of the rung, and the gather must
be proved chunk-independent, or a halo column gathered by a neighbouring chunk will find a different set
and the two meshes will crack — the property the tube carver earns by gathering per REGION
(`carve.rs:127-146`), not per chunk.

### F15 — The owner's biome sentence is answered for gravity only, and `biome_at` is frozen while its inputs are handed away

The owner asked for biomes dependent on *"the planet position, spin, trajectory, size and gravity"*.

This document takes gravity, and uses it for the relief ceiling only — not for a biome. §3.1 requirement
4 then declares `height::biome_at` to be THE biome and forbids a second source, which is right, and hands
what it READS to DOMAIN 02 and DOMAIN 05. But it never states what that function must BECOME, and today
it reads latitude about a pole axis fixed at `+Z` with no obliquity (`height.rs:29-35`), a height over the
sea, and two slow noises (`height.rs:39-66`). Position, spin, trajectory and size touch none of it.

The risk is concrete: §3.1 requirement 4 freezes an interface around a function whose signature —
`biome_at(body, dir, surface_m)` — has no room for insolation, obliquity, an upwind ridge or a distance
to a coast. Requirement 4 should say what the signature must grow to, or DOMAIN 05 will have to break it.

### F16 — Weather is named, listed and not simulated

The owner said *"we also should simulate the weather"* in the same sentence as the picture. §3.3 answers
with a table of what the shape OWES a weather model, a wind LAW from latitude and spin, and one sentence:
*"Today's storm is LIVE STATE, and if the game ever wants one it crosses as a one-hop diff from the
owning realm, which is a new lane and needs the owner's word."*

That is the right law reading, and it is not an answer. Nothing states what a storm costs, what it would
carry, how often, or whether the static climate alone can carry the picture (haze, cloud, a snow line
that moves). §11 marks SL6 PASSED partly BECAUSE no weather crosses. The owner should be told plainly, in
§0, that this domain gives the weather its inputs and builds none of it, and which domain owes the
answer and when.

---

## NOTES

### F17 — The budget is 8 ms per CHUNK, not per rung-0 chunk

§0 and §9 say *"8 ms of worker time per rung-0 chunk (ruling V10)"*. The ruling reads: *"the 4 ms worker
budget per chunk held for a surface chunk (3.3 ms) and not for a cave-dense one (6.1 ms); the owner set
it at 8 ms"* (`owner_decisions_2026-09-07_voxels.md:441-442`). It is per chunk at any rung. This helps
the document — building §9.2 on the measured rung-2 chunk is then legitimate — and the wording should be
corrected so nobody later strikes the table for mixing rungs.

### F18 — The 0.935 m corner cell is right; the shown working does not produce it

§4.3 derives `W'(1) = K1 + 3K2 + 5K3 = 1.5584` (checked against `bend.rs:22-30`) and then states a
0.935 m cell at a cube corner. The reader cannot get there: `1/1.5584 = 0.642`. The missing step is the
normalisation of `(a, b, 1)`, whose tangent scale at a corner is `sqrt(1 − 1/3)/sqrt(3) = 0.4714` against
1 at the face centre, so the corner cell is `1.5584 × 0.4714 / 0.7854 = 0.935` of a centre cell. The
number is correct. Show the step.

### F19 — The reference picture's forest and fields have no biome to hang on

§11's V4 row says *"`biome_at`, which both hosts already compute, is what the server's scatter skeleton
reads to choose an asset"*. The enum has four members: Desert, Grassland, Tundra, Highland
(`strata.rs`, `Biome`). The reference picture's forest MASS and its cultivated fields are neither
grassland nor highland, and the picture's snow-capped ridge is Tundra topsoil (`Stratum::Snow`,
`strata.rs:19`) only where the temperature test happens to fire. Say which domain owes the biome set the
picture needs, or the art assets will have nothing to key on.

---

## Checked, sound

- Every code citation in revision 1 was re-opened. All of them read as the document says:
  `body.rs:146-149,155,169,176,180-184,196-201,221,228,233-236,252-258,355-365,368-379`;
  `height.rs:17-27,29-35,39-66`; `chunk.rs:19-23,139-142,183,207-213,611-614,630,633-635,641-649`;
  `extract.rs:37,62-69,182-187`; `seat.rs:32-62`; `strata.rs:18,166-178,188-209`;
  `carve.rs:160-175`; `taxonomy.rs:750-752`; `ladder.rs:87-97,132-139`; `bend.rs:22-30`.
- The three withdrawals are correct and were re-verified: base terrain mints no record; the `object`
  byte is the tree's; `seat_eighths` reads one neighbour and already answers an arch; `greater` can only
  open air (`chunk.rs:641-649`), so no addition term can ride the cave machinery.
- The gradient fix is right: `frequency = radius/wave` and `dir` is a unit vector, so the derivative
  needs `1/wave_m` and a tangent projection.
- The extractor argument is right: `crossing` is a ratio of two magnitudes (`extract.rs:182-187`), so a
  per-column factor divides out on a radial edge and moves no vertex; the clamp on the lateral neighbour
  is the real defect.
- The home planet's own numbers reproduce: `N = 5 263 360` gives 84 892.9 chunks per face edge at rung 0
  and 21 223.2 at rung 2, matching the chunk names in `slice_06_extractor.md`; the Chen-Kipping mass is
  0.0969 Earth masses and the surface gravity 3.44 m/s².
- The unit-cost regression reproduces: `(710 − 266)/11 = 40.4 µs` per octave and `144 µs` fixed.
- The terrace derivative expression `1 − q·H·(1 − 9u² + 8u³)` reproduces 1.275 / 1.413 / 1.550 at
  `u = 0.75`. Only its reachability is refuted (F3).
- Marking the budget row FAILED, and saying the last 0.4 ms turns on an unmeasured vertex count, is the
  honest reading and it should stay in §0 exactly as written.
- §4.7's statement of the metre-scale limit — about 8 m is the finest height feature 1 m cells carry, so
  boulders and talus are objects and not octaves — is the strongest new paragraph in the revision.
