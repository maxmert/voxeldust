# Refutation A of DOMAIN 03 — erosion, rivers, coasts and ice

**Target:** `docs/investigation/2026-09-08/landforms/03_erosion_rivers.md` (revision 2).
**Date:** 2026-09-08. **Lens:** THE LAWS AND THE CODE. Is every claim about the repository true? Is
every law gate passed? Is every number sourced?
**Method:** I read the document in full, then the code it cites, then the rulings it names. I ran no
`cargo` command. I recomputed the document's arithmetic in Python from the code's own recipe.

**Verdict:** revision 2 is a large improvement on revision 1. Its citations are now mostly accurate.
But it fails five law gates, and one of its own headline numbers is measurably wrong on the body it
names. The five blockers are new; none of them repeats refutation A or B on revision 1.

---

## Blockers

### A2-B1 — Ruling S7-2 refuses recommendation D3, and the document never cites S7-2

**Where:** §0.7, §5.2, §10, §12 D3, §11 (the SL10 rows).

The document's largest structural change is D3: *"the owning realm STATES the body facts (28 bytes)
and SHIPS the artifact"*, argued from rulings S7-1 and S7-3. It never names S7-2, which sits on the
same page of the same ruling and refuses it:

> **S7-2 the world hello** — *"The client states both halves at login, computed on the home planet's
> literals; a client that cannot compute them does not log in."*
> (`docs/design/owner_decisions_2026-09-07_voxels.md:498`)

The code implements exactly that. `crates/client/src/net.rs:1192-1198` builds
`WorldIdentity::of(HOME_UNIVERSE_SEED, home_planet())`; `crates/terrain/src/tag.rs:43-50` makes the
MEASURED half `golden_self_check(home)`; `crates/terrain/src/digest.rs:22-26` generates eight home
planet chunks with `generate(body, key)`; `crates/client/src/net.rs:146-151` states the pair to the
gateway right after `Welcome`.

The chain does not close:

```
   login  ─►  the client must generate 8 home-planet chunks
          ─►  those chunks now hold Z
          ─►  Z comes from the artifact
          ─►  the artifact arrives from the realm AFTER login (D3)
```

§5.2's one sentence — *"The generator's world identity digest must FOLD the stated facts"* — hides the
whole problem: folding a fact the client does not have does not give the client the fact. §12 D11's
cure (*"its artifact digest belongs beside"* `home.rs:16-21`) does not close it either: a digest is
not the bytes, and `generate` needs the bytes.

The `TAG_SURFACE` doc comment says the same from the other side
(`crates/sim/src/stub/tests/window_lane.rs:2557-2561`): *"a client that holds the same recipe DERIVES
the hills from the seed"*. D3 reverses that sentence without saying so.

**What is owed:** either the artifact is derived on both hosts (which §10 refuses, for reasons that
are themselves sound), or S7-2 is put to the owner as a ruling to change, with the new login shape
written out. The document does neither, so D3 as written cannot be built.

---

### A2-B2 — §5.4's C1 seam rests on a premise the cited code contradicts: no macro node lies on a cube edge

**Where:** §4.1 (the node's direction), §5.4 rules 1, 2 and 4, §14 M5.

§4.1 defines a macro node's direction as
`direction(face, face_param(i, n_macro), face_param(j, n_macro))`, *"the same call the column pass
makes (`crates/terrain/src/chunk.rs:133-137`)"*. That call is correct — and it is a **cell centre**:

> `crates/seed/src/ladder.rs:142-145`
> `/// The face parameter of a cell's CENTRE: (2i + 1)/n_l − 1, in (−1, 1).`

The interval is **open**. The outermost node sits at `a = 1 − 1/n_macro`, half a node inside the cube
edge. COMPUTED on the home planet: 4 112 m of arc inside the edge.

§5.4 then states the opposite and builds its whole seam argument on it:

1. *"Node planes sit ON the cube edges … the edge nodes of two partner faces are the same points on
   the sphere."* **False.** They are two distinct points 8 224 m apart, straddling the edge — which is
   what `across()` encodes (`crates/seed/src/seam.rs:183-203`: the partner's index is `n_l − 1` or
   `0`, an adjacent CELL, never a shared one).
2. *"An edge node's value is OWNED by one face … the field is single-valued on every cube edge by
   construction."* There is no shared node to own. The construction does not exist.
3. Rule 4's C1 argument evaluates `W'(1)` and `|p|` *"at the edge"* — a place where the lattice has no
   sample. Two half-offset grids do not mirror into each other, so the symmetry argument is void
   whether or not `W'` is symmetric there.

`corner_param` (`crates/seed/src/ladder.rs:148-151`, `2i/n_l − 1`, in the **closed** `[−1, 1]`) does
put samples on the edge. But then §4.1's integer lookup `(i·2^L) / (N / n_macro)` names the wrong
node — it names the cell, and a corner lattice is offset from it by half a node. The two sections
need two different conventions and the document uses one word for both.

This is the same class of error the document charges revision 1 with (B-B3) and, at 63 000 km of cube
edge (the document's own COMPUTED figure, which I checked: `2π·3 350 759/4 = 5 264 km × 12`), it has
the same consequence. §14 M5's G-MACRO-EDGE would go red on the first run, with no design behind it
to fix.

---

### A2-B3 — the slope spectrum multiplies the ladder's coarse-answer bound by up to ten, and the only ladder gate named is a tautology that grows with it

**Where:** §4.2, §7.1, §9, §11 (the ladder row), §14 M8.

I recomputed `crates/terrain/src/body.rs:143-183` for the home planet's real literals
(`HOME_PLANET_SEED = 7 701 581 858 760 374 086`, radius bits `0x41499139_1e692dfa`), then applied
§4.2's own spectrum law at `o_peak = 7`, `σ = 1.4`, normalised to the same relief. Both columns sum
to 14 305 m.

| rung `L` | `dropped_bound_m(L)` today | with §4.2's spectrum | ratio |
|---|---|---|---|
| 4 | 6.9 m | 25.9 m | 3.8 |
| 6 | 32.8 m | 263.4 m | **8.0** |
| 7 | 70.3 m | 713.2 m | **10.1** |
| 8 | 150.6 m | 1 435.8 m | **9.5** |
| 10 | 687.9 m | 3 466.9 m | 5.0 |

`dropped_bound_m` is the ladder's stated coarse-answer bound (ruling V8/V9) and the direct input to
the arrival-pop budget. `docs/investigation/2026-09-07/02_smooth_terrain.md:477` already estimates the
arrival pop at **4 px at `k_rough = 0.5`**, calls it *"visible"*, and says *"M4 RUNS BEFORE SLICE 3
BUILDS THE LADDER"*. A tenfold increase in the dropped amplitude at rung 7 takes that estimate to tens
of pixels. SL8 calls one such step a defect.

The document does not mention this once. §7.1 says only that *"`Z` contributes exactly ZERO to
`dropped_bound_m`"* — true, and beside the point — and that `C` and `T` grow it by *"the deepest
channel dropped at that rung plus the detail's amplitude"*, which at 32 m (§4.12's
`MAX_CHANNEL_DEPTH_M = 64`, §6.1's Amazon at 32 m deep) is forty times smaller than the term it omits.

Worse, the gate the document names cannot catch it. `crates/terrain/src/height.rs:98-101` asserts
`(hl − h0).abs() <= dropped_bound_m(rung)`, and `dropped_bound_m`
(`crates/terrain/src/body.rs:274-276`) is itself the sum of the dropped amplitudes. Grow the
amplitudes and the bound grows with them: the test stays green while the picture pops. §14 M8 names
*"the pop at a rung transition where a river arrives, and at a pyramid-level change of `Z`"* — not the
spectrum, which is the term that moved.

§9 rule 4's escape — *"The existing crossfade (ruling V10) already covers the octaves and the
caves"* — is also false about the code. `grep -rn "crossfade" crates/` finds it in no source file;
ruling V10 and `00_proposed_voxel_foundation.md:719-721` put the crossfade in **slice 8**, which is
not built.

---

### A2-B4 — the macro lattice under-samples the octaves it declares "coarse", by a factor of two, and cannot see the slopes it is asked to cap

**Where:** §4.2 (the octave selector and the overshoot answer), §4.8, §7.3 rule 2.

§4.2: *"The solve starts from the sum of every octave whose WAVELENGTH exceeds the macro node size …
octaves 0 to 5 (400 km down to 12.5 km) are coarse."*

The macro node is 8 224 m, so its Nyquist wavelength is **16 448 m**. Octave 5 at 12 500 m is sampled
**below Nyquist** and folds. Octave 4 at 25 000 m gets three samples per wavelength.

Under §4.2's own spectrum, octave 5 carries **908.6 m** of amplitude on the home planet (my
computation). Aliasing 908 m of relief onto an 8 224 m lattice produces a long-wave beat locked to the
lattice — a regular 8 km pattern over every continent, which is the eleven-seam taxonomy's
*"detail-by-box"* row at continental scale. No gate in §14 looks for it. The rule the document needs
is *"wavelength exceeds TWICE the node size"*, which moves octaves 4 and 5 out of the solve, changes
`A_coarse` from 12 869 m to 10 838 m, and changes what the envelope renormalisation clamps. That is a
design change, not an edit.

The mirror of the same error is the overshoot answer: *"The overshoot is the talus rule's job … why
the design has a talus relaxation on the macro lattice (§4.8)."* The overshoot lives at octaves 6, 7
and 8 — 6.25 km, 3.13 km and 1.56 km, carrying 723 m, 450 m and 181 m with per-octave slopes 0.73,
0.91 and 0.73 against a loose-rock talus tangent of 0.70. **Every one of those scales is invisible to
an 8 224 m lattice.** The macro talus pass cannot see, let alone cap, the slopes the spectrum creates.
The remaining cure — §7.3 rule 2, *"subtract a term that pulls the surface toward the local mean"* —
has no stated form, no stated amplitude and no stated bound, while it must remove roughly a third of
the relief the spectrum just added. That is a hand-wave exactly where the hard part is.

---

### A2-B5 — the artifact cannot answer §8.1's water rule: nothing maps a node to a lake

**Where:** §0.9, §4.4 (the lake table), §4.13, §6.1, §8.1, §8.5.

§6.1 itemises the artifact's four bytes per node exactly: *"the receiver direction, 3 bits; the
surface facies, 3 bits; the ice mask and the coastal band, 1 bit each — the eight bits pack into ONE
byte; the discharge, a separate byte"*, plus the `i16` height. There is **no lake identifier**.

§8.1 then requires one:

> `water_level(column) = … the spill level of the lake the column is in  if the column is inside a lake`

and §4.4 makes the lake table *"a small map from lake id to spill level"*. With three facies bits a
column can learn that it is in **a** lake; nothing tells it **which** lake, so nothing gives it the
spill level. A planet has many lakes at many heights, so the answer is not constant. Item 9 of the
one-page recommendation — *"Water gets a level per column"* — is not derivable from the artifact the
same document specifies.

The cheapest fix is a per-node level: an `i16` spill level beside `Z`, taking the artifact from 4 to
6 bytes per node. COMPUTED: `2 457 600 × 6 = 14.75 MB`, not 9.83 MB. That number appears eight times
in the document, including in the one-sentence version, in §12 D3's cost line and in §15.

The same hole runs through §8.5, which promises a live weather layer *"the ELA per node"* and *"the
precipitation climatology"* from the artifact. §4.13 keeps neither: §10's memory table has
`precipitation (2)` as **solve-only** state, thrown away with the rest. Two of the four fields §8.5
offers the weather domain do not exist after the solve ends.

---

## Defects

### A2-D1 — the headline spectrum numbers are not the home planet's, and the "constant slope" diagnosis is false for it

**Where:** §0.3, §1 (rows 2 and 3), §4.2, §16 B-B4.

The document says, labelled COMPUTED and attributed to the home planet:

* *"the recipe spends 6 000 m of amplitude at a 400 km wavelength and 47 m at 3 km, 23 m at 1.6 km"*;
* *"at `k = 0.5` the per-octave SLOPE `2π·a/λ` is a constant 0.094 at every wavelength … the surface is
  self-similar and uniformly gentle, at every scale, everywhere. That is exactly the *"no orienters and
  no details"* picture."*

I recomputed `BodyDefinition::from_seed` for the home planet's own literals
(`crates/terrain/src/home.rs:16-21`), reimplementing `SplitMix64` (`crates/seed/src/rng.rs:22-28`),
`child_seed` (`:70-74`) and `draw_unit` (`crates/terrain/src/body.rs:113-115`):

| quantity | the document | the home planet |
|---|---|---|
| relief | 12 000 m (assumed) | **14 304.89 m** |
| `k_rough` | 0.50 (assumed) | **0.468 338** |
| amplitude at 400 km | 6 000 m | **7 605.6 m** |
| amplitude at 6.25 km | 94 m | **80.3 m** |
| amplitude at 3.13 km | **47 m** | **37.6 m** |
| amplitude at 1.56 km | **23 m** | **17.6 m** |
| per-octave slope | *"a constant 0.094"* | **falls 0.1195 → 0.0510**, a factor of 2.34 |

Two consequences.

1. Every number in §0.3, §1 and §4.2's first table carries the label COMPUTED while describing a body
   that does not exist. The recipe draws `relief = clamp(0.004·R, 200, 12 000)·(0.5 + u)` and
   `k_rough = 0.45 + 0.10·u` (`crates/terrain/src/body.rs:147-155`); `k = 0.5` exactly is a draw of
   probability zero. The document's own preamble promises every number is MEASURED, COMPUTED,
   ESTIMATED or UNMEASURED, and the standing rule is that a claim about the world is a measurement,
   never an argument.
2. The **diagnosis** is wrong in shape, not only in digits. The home planet's spectrum is not
   self-similar; it is progressively **smoother** toward fine scale, because `k = 0.468 < 0.5`. The
   picture is not *"uniformly gentle at every scale"* — it is a big smooth swell that loses what
   little texture it has as you look closer. That is a different complaint and a different cure at the
   fine end.

The conclusion survives (37.6 m is further from the 300–600 m target than 47 m), which is why this is
a defect and not a blocker. Its evidence does not.

### A2-D2 — the discharge byte does not fit the range the document's own §4.5 states

**Where:** §4.5, §6.1, §16 "Findings KEPT".

§4.5 bounds `Q` for the `u64` check with a maximum precipitation of **10 000 mm/yr**: *"the largest
possible `Q` on the home planet is `1.41 × 10^14 m² × 10 000 mm = 1.4 × 10^18`"*.

§6.1 then sizes the log byte with a maximum of **1 000 mm/yr**: *"the ceiling is the whole surface at
1 000 mm, `1.41 × 10^17`"*, floor `6.76 × 10^7`, ratio `2^31`, *"It fits in `2^32` with one octave to
spare"*.

COMPUTED at §4.5's own ceiling: `1.41 × 10^18 / 6.76 × 10^7 = 2.09 × 10^10 = 2^34.3`. The byte spans
256 steps at 1/8 octave = `2^32`. **It does not fit, by 2.3 octaves.** Earth's wettest station
(Mawsynram, about 11 900 mm/yr) says 10 000 is the honest ceiling, not 1 000.

This is the arithmetic on which §16 explicitly KEEPS half of refutation A's finding A-D2. The kept
half is wrong under the document's own other section.

### A2-D3 — `PASSES` and `Δt·K0` are degenerate, so §12 D9's "measure it" is not a measurement

**Where:** §4.6, §4.11, §5.5, §12 D9, §16 A-W5.

§4.6's implicit update is `z_new = (z + Δt·K·√Q/L · z_r) / (1 + Δt·K·√Q/L)`, run `PASSES` times with
`U = 0`. Far from the base level the total cut depends on `PASSES` and `Δt` only through their
product. §5.5 states that *"`K0` and `Δt` appear only as the product `Δt·K0`, so ONE constant is
free"* — and then leaves that free constant unconstrained while §12 D9 promises to *"MEASURE `PASSES`
against the per-basin hypsometric integral and freeze"*.

Any hypsometric integral can be hit at any `PASSES` by moving `Δt·K0`. The pair is one degree of
freedom presented as a measured number beside a free constant. The document's answer to A-W5 — *"a
number somebody liked would set the relief the player sees, which the no-magic-numbers rule
forbids"* — therefore does not land: the taste knob is still there, renamed.

The physics behind it is the same problem. §4.2 says *"the body's coarse octaves ARE the uplift"*.
Uplift `U` in the stream power law is a **rate**, in metres per year; the coarse octaves are an
**initial height**. With `U = 0` the equation has no steady state: every pass is a monotone decay
toward the base level, slowed by the isostatic rebound by a factor `1/(1 − 0.83) ≈ 5.9` and never
stopped. There is no mature equilibrium landscape, only a frozen moment on a decay curve, and which
moment is exactly the free constant above.

### A2-D4 — the second extractor run is charged at a fifth of the first, with no basis, and it is the number that decides D15

**Where:** §7.4, §8.3, §10, §12 D15, §14 M2.

§8.3's water surface is *"the SAME extractor a second time over the signed field
`min(water_level − r, r − rock_surface(column))`"* — a full 64³ field build plus a full surface-nets
pass over the same box. §7.4 charges it **0.3–1.0 ms**, against a MEASURED 6.1 ms for the whole
cave-dense chunk (`crates/bins/examples/terrain_cost.rs:37-45`; ruling V10,
`owner_decisions_2026-09-07_voxels.md:441-442`) in which the extraction is one of two halves.

A second run of the same extractor over the same box cannot plausibly cost a fifth of a run that
already includes one. If the extraction half of 6.1 ms is 2–3 ms, the honest row is 2–3 ms and the
worst case is 9–10 ms, not 8.25 ms — over the owner's budget by 25 %, not by 3 %. §12 D15's
recommendation (*"(a), with M2 before the slice lands"*) and its named dial (`SUBDIV`, worth perhaps
0.2 ms) are both sized against the low figure. `SUBDIV` cannot buy 2 ms.

### A2-D5 — §11's HR5 row answers a cost question and calls it a coverage answer

**Where:** §11 (the HR5 row), §14 M7.

The row reads: *"The solve holds data-dependent branches: a pit, a lake, a seam, a corner, a no-sea
planet, a dry planet, a node with no receiver, a glaciated node, a coastal node. COMPUTED: a 200 km
moon of THE world has 6 144 macro nodes and a 50 km body has 384, so a full solve runs inside a unit
test in milliseconds even instrumented."*

That shows the fixture is **affordable**. HR5 asks whether every one of those arms is **reachable** on
a real body, which under SL5 must be a body of THE world and not an invented one. §14 M7 says exactly
that the reach is unknown: *"UNMEASURED: whether THE world holds a body that is both small and wet
enough to reach the fluvial arms."* A law-gate table must state what it is gating. §11's SL8 row does
this correctly (*"UNMEASURED until M8 and M11"*); the HR5 row should read the same way.

### A2-D6 — §4.1's `n_macro` rule contradicts itself on a small body

**Where:** §4.1, §14 M7, §11 (HR4).

The rule: *"`n_macro` is the divisor of `N` whose node size `N / n_macro` is closest to
`MACRO_CELL_TARGET_M` … clamped to at least 8."*

The clamp can produce a non-divisor, which destroys the one property the rule exists for (*"it makes
the per-chunk lookup INTEGER"*). COMPUTED for a 5 km body of THE world: `n_ideal = 7 854`,
`top_rung_for = 1` (`crates/seed/src/ladder.rs:47-56`), `snap` gives `N = 7 854 = 2·3·7·11·17`. The
divisor closest to 8 192 m is `n_macro = 1`; clamping it to 8 gives a number that does not divide
7 854. The next divisor at or above 8 is 11, node size 714 m — a hundredfold departure from the
target the rule names. The rule as written does not say which of the three answers it means, and HR4
says the same code runs on every body.

The document also asserts *"A rung-0 chunk is 62 m across and sits inside ONE macro node"* (§6.4).
COMPUTED: `8 224 / 62 = 132.6`, so a chunk edge is not commensurate with a node and chunks do straddle
node boundaries. The 5 × 5 gather covers it, so the consequence is nil, but the sentence is false.

### A2-D7 — §7.3 rule 2 claims the reference picture's rock pillars and describes the rule that erases them

**Where:** §7.3 rule 2, §4.2.

The rule: *"Where the slope exceeds the talus tangent, subtract a term that pulls the surface toward
the local mean … That is the reference picture's rock pillars and their debris."*

A rock pillar, a tor or a hoodoo is precisely a column of rock that stands **steeper than the talus
angle and does not collapse**, because it is jointed competent rock and not loose debris. A rule that
relaxes everything above the talus tangent toward the local mean removes every pillar in the world by
construction; the scree fan it leaves is the debris of the pillar that is no longer there.

The landform the reference picture shows needs differential resistance — a lithology field or a joint
field — which §12 D14 asks DOMAIN 02 for. The document should say the pillars wait for D14, not claim
them from a rule that deletes them.

### A2-D8 — the pyramid's top level is 4.8 KB, not 19 KB

**Where:** §0.2, §4.13, §9 rule 2, §10, §12 D3.

§4.13: *"a coarse PYRAMID of `Z` (levels 320…20) … +1.6 MB; top level 19 KB"*.

COMPUTED. The pyramid halves 640 → 320 → 160 → 80 → 40 → 20. `Z` is an `i16`, so level 20 holds
`6 × 20² × 2 = 4 800 B = 4.8 KB`, and level 40 holds `6 × 40² × 2 = 19 200 B = 19.2 KB`. The 19 KB
figure is level **40**, not the stated top of 20. (The `+1.6 MB` total is right:
`2 × 6 × (320² + 160² + 80² + 40² + 20²) = 1.637 MB`.) The wrong number is repeated in five places,
including the recommendation the owner would act on in §12 D3.

### A2-D9 — the ELA is computed and never reaches the ground

**Where:** §4.9, §7.3, §11 (the record row), §12 D8.

The reference picture's most obvious single feature is a **snow-capped ridge**. §4.9 derives the ELA
carefully — the two lapse rates, the dry-climate offset, the glacial line — and then uses it only to
carve U-valleys, cirques and fjords. Nothing in the document puts snow on the ground above it.

The code puts snow on the ground by biome, not by height: `crates/terrain/src/strata.rs:190-196` gives
`Stratum::Snow` for `Biome::Tundra` and nothing else, and `biome_at`
(`crates/terrain/src/height.rs:40-66`) reads latitude, height over the sea and two noises, never the
ELA. §11's record row promises *"Nothing changes"*, and §12 D8 keeps the facies out of the record. So
after this whole design lands, a 6 000 m equatorial ridge on the home planet still carries
`Stratum::Gravel`, and the white cap must come from client style alone.

That may be the right answer under ruling V4 (art assets, client style). The document never says so,
and the owner's acceptance picture turns on it.

### A2-D10 — the talus pass moves the same excess to up to eight neighbours

**Where:** §4.8.

*"compute the drop to each of the eight neighbours; where the drop exceeds `L_ij · tan(θ)`, move half
the excess to the lower neighbour."*

Read against the loop, a node with eight lower neighbours ships half of **each** excess, so up to four
times its own excess leaves the node in one pass. Mass is not conserved and the pass can overshoot and
oscillate. The standard form distributes the excess **proportionally among** the lower neighbours,
with the total capped. `TALUS_PASSES = 8` is then recommended with no stability statement of any kind.
§5.1's row for the talus pass answers only determinism (*"a second buffer, one floor on apply"*),
which is a different property from convergence.

---

## Weaknesses

### A2-W1 — "the cell box does not grow" implies a restructure of `SampleBox` that is never named

§7.3 rule 1: *"The column ring must grow from one cell to two — COMPUTED: 64² = 4 096 columns to
66² = 4 356, +6.3 % of the box's columns … The CELL box does not grow."*

The arithmetic is right and the cost is honestly stated. The structure is not. `HALO = 1`
(`crates/terrain/src/lattice.rs:48-49`) sizes `BOX_EDGE = CHUNK_EDGE + 2` and therefore
`BOX_CELLS = BOX_EDGE³`, and `SampleBox::column_index` indexes the column arrays by the same
`BOX_EDGE`. Growing only the column ring means two ring depths, two index functions and a changed
`sites` / `dirs` / `surfaces` layout that `extract` also reads. That is a refactor of the sample box,
not a `+6.3 %` line item, and it belongs in §12 as a decision.

### A2-W2 — the priority flood's `ε` writes a false drainage direction across flat ground

§4.4 raises each flooded node by `ε = 1/16 m`. §4.4 itself computes the consequence for a lake
(*"would tilt a 100-node lake by 6.25 m"*) and cures it with the spill level. It does not compute the
consequence for a **flat drainage path**: 10 000 nodes of flat continental interior accumulate
`10 000 / 16 = 625 m` of artificial gradient in `z_flood`, and `z_flood` is what §4.4's table says the
D8 receiver pass reads. On flat ground the artificial gradient dominates the real one, and the river's
direction becomes a function of the heap's pop order — deterministic, by the `(z, index)` tie-break,
and geologically arbitrary. On a continental plain the rivers would run in index-order stripes.

### A2-W3 — the solve is transport-free, so every depositional landform in the reference picture is cosmetic

The stream power law as written (§4.6) is **detachment-limited**: rock is removed and vanishes. There
is no sediment budget anywhere in the design. Yet the document draws, from nothing:

* the **delta** (§4.10) — *"fans it over a radius that grows as `sqrt(Q)`"*;
* the **floodplain** (§7.2) — *"the flat strip beside a river that the river BUILT out of its own
  sediment"* (§2's own definition);
* the **freeboard** — a stated constant.

An **alluvial fan**, the landform where a mountain stream meets a plain and the one that most often
carries the settlements in a range like the reference picture's, is not named anywhere. Volume is not
conserved: the eroded rock does not fill the sea, so the hypsometric curve M6 measures is missing its
depositional half. §8.4's postponed list names aeolian, karst and mass wasting; it does not name
sediment transport, which is the largest omission of the three.

### A2-W4 — no volcanic landform is built or postponed

An earth-like planet's second landform family after fluvial is volcanic: cones, calderas, lava plains,
volcanic necks and plateaus. `crates/terrain/src/strata.rs:136-142` draws Basalt, Gabbro and Andesite
as bedrocks, so the world already asserts a volcanic history it never shows. §8.4 postpones aeolian,
karst and mass wasting by name and is silent about volcanism. The owner should be told, since the
reference picture's rock pillars are as likely volcanic necks as fluvial remnants.

### A2-W5 — the climate recomputation has no cost row

§4.11 recomputes the precipitation field every `CLIMATE_EVERY = 10` passes, four times over 2 457 600
nodes. §10's cost table has rows for the receivers, the flood, the accumulation, the sweep, the
rebound, the talus and the ice. It has **no row for the climate**. If DOMAIN 02's orographic lift is
an upwind integration along a wind streamline rather than a per-node closed form, it is not
`O(nodes)`, and it could dominate the 12–20 s headline that §12 D3 and D11 both hang on. §14 M1 must
bench it and does not name it.

### A2-W6 — one word, two meanings: "facies"

§6.1 makes the facies three bits **per macro node** in the artifact. §7.3 rule 2 and §4.10 set the
facies **per column** at the fine pass (*"set the facies to `Gravel`"*, *"the facies is `Sand` over
`Clay`"*), and §12 D8 rules that the facies is *"derived per column and never stored"*. Those are two
different fields with one name, at two resolutions 8 224 m apart. Worse, the fine-pass names are
`Stratum` values from `crates/terrain/src/strata.rs` (Gravel, Sand, Clay), not members of §6.1's
eight-value facies list (upland, floodplain, delta, beach, scree, glacier, lake bed, sea bed). The
art-asset skeleton (ruling V4) is the consumer and it must be told which field it reads.

### A2-W7 — the meander band is narrower than stated, by a factor of four

§6.3: *"at 514 m sub-segments the subdivision can express a meander for rivers between about 47 m and
745 m wide."* The upper bound is right (`11 × 745 = 8 195 m`, one macro segment). The lower is not: at
`w = 47 m` the meander wavelength is `11 × 47 = 517 m`, which is **one** sub-segment, and one sample
cannot draw a wavelength. Two samples per wavelength gives `w ≥ 94 m`; four, which is the least that
reads as a curve rather than a zigzag, gives `w ≥ 187 m`. The served band is about 190–745 m, a factor
of four, not a factor of sixteen. §13 open question 9 is then wider than it says.

### A2-W8 — the 9.83 MB ship has no lane

§12 D3 recommends the realm *"SHIPS the artifact, coarse pyramid first, full field for the body you
are at"*, and argues its lawfulness from `SurfaceStmt`, whose whole self-look bag has a **1 200-byte**
budget (`crates/core/src/look.rs:76-79`, `SELF_LOOK_BUDGET_BYTES = 1200`, asserted at `:502-507`). The
28 bytes of facts fit. The 9.83 MB (14.75 MB after A2-B5) does not, and the document never names the
lane that carries it, its budget, its retention or its back-pressure. SL6 asks for *"what data, from
which realm to which"*; D3 answers for the 28 bytes and asks the owner to approve the 9.83 MB in the
same breath. Those are two asks and the second is the expensive one. §13 open question 5 notices the
storage half and not the transport half.

### A2-W9 — the macro node's area is a midpoint approximation described as exact

§4.1: *"The area is closed-form from the polynomial `W'(a) = k₁ + 3k₂a² + 5k₃a⁴` … then FLOORED to a
whole square metre and summed as a `u64`, so accumulation is exact integer arithmetic with no rounding
order at all."*

The **sum** is exact and order-free — that half is right and it is a good rule. The **area** is not
closed-form: the density `W'(a)·W'(b) / |n + W(a)u + W(b)v|³` must be integrated over the node's
parameter cell, and evaluating it at the node centre is the midpoint rule. The error is small (the
document's own table shows the density varies by 24 % across a whole face, so across one node of
1/640 of a face it is parts per thousand), but the word "exact" belongs to the summation, not to the
area.

---

## Notes

* **A2-N1** — `crates/terrain/src/body.rs:196` is cited three times (§1, §4.6, §8.4) for the sediment
  list. The list is at **`:194`**; `:196` is `topsoil_m`. Revision 2 corrected revision 1's citation to
  a line two rows off the right one.
* **A2-N2** — `crates/seed/src/bend.rs:56-58` is cited for the derivative `W'`. It is computed at
  **`:60`** (`let df = K1 + a2 * (3.0*K2 + a2*(5.0*K3));`). `bend.rs:24-30` is cited for the
  constants; `K1` is at `:23`.
* **A2-N3** — §1 cites `crates/bins/examples/terrain_cost.rs:37-45` for *"the column pass at rung 0
  about 1.5 ms per 62 × 62 columns"*. Those lines hold `EXTRACT_BUDGET_US` and
  `WORST_CASE_MEASURED_US` only. The 1.5 ms figure has no source in the file cited.
* **A2-N4** — `crates/terrain/src/chunk.rs:47` is cited for the gap byte; `GAP_STEPS_PER_CELL` is at
  `:46`. `chunk.rs:95-108` is cited for `ColumnField`; it is at `:94-107`.
* **A2-N5** — §4.1 cites `crates/terrain/src/extract.rs:13-18` for a corner node having seven
  neighbours. The extractor's fact there is that a corner **ring** has three members, not four — a
  different statement about a different object. The count of seven is right; the citation does not
  carry it.
* **A2-N6** — §7.4's *"one 4 × 4 node gather, then one Catmull-Rom per column | < 0.05 ms"* is
  ESTIMATED. COMPUTED at 16 multiply-adds per column over 4 096 columns and 1 ns per operation, the
  floor is about 0.065 ms before any cache miss into a 9.83 MB table. M2 covers it.
* **A2-N7** — §0's *"Everything the player stands close to is closed-form"* and §7's *"the closed-form
  pass"* use "closed-form" for a rule that reads a 9.83 MB table. One word, two meanings.
* **A2-N8** — the seed ruling and the floodplain. A seed-derived floodplain map is a public map of the
  only ground flat enough to build a settlement on (§2, §7.2). The 2026-08-27 ruling permits it —
  *"an expensive gate buys ROOM, not treasure … which is live state"* — so the design passes. The
  document should say so in the §11 seed row rather than answering only for ore and placers, because
  the buildable-land map is the one a reader will ask about.
* **A2-N9** — §4.12's *"Setting `MAX_CHANNEL_DEPTH_M = 64` keeps that inside `crust_m` with nothing to
  spare"* is conservative-wrong in the safe direction. `crust_m = relief + strata.max_depth_m() +
  caves.max_depth_m + 64` (`crates/terrain/src/body.rs:234`) adds the strata depth **and** the cave
  depth, while only the deeper of the two is needed below the surface. COMPUTED: a further 23–92 m of
  unused crust on every body. The channel fits with more room than the document claims — but §7.3
  rule 2 also cuts **downward** and no bound is stated for it, so the accounting is still incomplete.

---

## Checked, sound

* `N = 5 263 360`, ladder radius 3 350 759.045 m, 12 rungs, 14 octaves, surface area 141.09 Mkm²,
  node area 67 634 176 m² at `n_macro = 640` — all reproduce exactly from
  `crates/seed/src/ladder.rs:70-96` and `crates/terrain/src/home.rs:16-21`.
* `640` divides `N = 2^12 · 5 · 257` and `8 224 m` is the divisor node size closest to 8 192 m (the
  neighbours are 5 140 m at `n = 1 024` and 10 240 m at `n = 514`). The divisor table in §4.1 is
  correct row by row, including the node counts and the megabytes.
* The bend area densities: centre 0.616 850, edge midpoint 0.432 739 (ratio 0.702), cube corner
  0.467 391 (ratio 0.758). I recomputed all three from `K1`, `K2`, `K3`
  (`crates/seed/src/bend.rs:23-30`). The sign correction against revision 1 is right.
* The 63 000 km of cube edge: `2π · 3 350 759 / 4 = 5 264 km`, twelve edges.
* `u64` cannot overflow on `Q` at 10 000 mm (`1.41 × 10^18` against `1.8 × 10^19`), and the D8
  cross-multiplication cannot overflow `i64` (`576 000 × 16 000 = 9.2 × 10^9`).
* `Z` as an `i16` in metres holds the envelope: `|Z| ≤ A_coarse ≤ relief < 18 000 < 32 767`.
* The hydraulic geometry table reproduces exactly at `a_w = 3.9`, `a_d = 0.4`, a runoff of 0.3 m/yr
  and 31 556 952 s/yr, including the Amazon row (57 034 m³/s, 931 m, 32.0 m), and the 10.2 km meander
  wavelength at 11 × width.
* The drainage density arithmetic: `1 / 8.224 km = 0.122 km/km²` against a literature 0.5–5, and a
  67.6 km² node far past a 0.1–1 km² channel head. Deleting revision 1's threshold is right.
* The `2^(1/16) − 1 = 4.43 %` half-step error on the log byte, and 2.2 % on a width.
* The solve's memory: 39 B × 2 457 600 = 95.8 MB, plus a 19.7 MB heap and a 9.8 MB stack = 125 MB.
* The pyramid's total of 1.637 MB, and the level-4 smoothing node size of 131 584 m being nearest a
  150 km flexural wavelength (9 600 nodes).
* Every float named in §5.1 is inside the grant at `crates/seed/src/bend.rs:9-14`; `Gf::sqrt` is at
  `crates/terrain/src/gf.rs:102` and `Gf::to_i64_floor` at `:96`, both cited correctly.
* Any dyadic exponent is reachable by repeated square root — the correction to revision 1's A-W9 is
  right, and `m = 1/2` needs one `sqrt`.
* The Braun and Willett implicit form is unconditionally stable at `n = 1`, needs the receiver updated
  first, and §4.6 orders the sweep outlets-upward correctly. The weighted average cannot push a node
  below its receiver.
* The priority flood's `(height, node index)` key is a total order over integers, so ties and repeated
  pushes cannot make the result depend on pop order.
* `fluid_at` (`crates/terrain/src/chunk.rs:205-213`) really does give every body an ocean today;
  `sea_radius_m` is drawn for every body (`crates/terrain/src/body.rs:186-190`) and it is the OFFSET
  that is floored. §8.1's first line is a real fix.
* `is_rock(gap) = gap < 0` (`crates/terrain/src/extract.rs:71-75`) really does leave the top of the
  water unmeshed, and §8.3's `min(water_level − r, r − rock_surface)` really is the intersection that
  ends the sheet at the shoreline. The refusal of a bespoke water mesher on HR3 is right.
* All five bedrocks (`crates/terrain/src/strata.rs:145-151`) are hard igneous or metamorphic, and the
  strata stop 23–92 m down (`crates/terrain/src/body.rs:196-198`). §12 D14's asks are well founded and
  the strath-terrace replacement in §7.3 rule 4 is the right answer for now.
* `crates/physics/src/taxonomy.rs:21-24` really does defer the cross-host bit-equality gate to
  SPIKE-6a; `:874-888` really does record `D-TAX-1`; `:891-909` really does say surface gravity is not
  stored. `t_eq_k` really is the equilibrium temperature, and reading it as a surface temperature
  would have given the home planet no rivers. The grey-slab correction and `τ = 0.835` are right.
* `crates/terrain/src/home.rs:44-77` really is a SNAP with a `10^12` margin (snap unit
  `2^11 · 2/π = 1 303.8 m` against a `10^-9 m` drift), and §5.2 is right to abandon revision 1's
  analogy.
* `foreign_lattices` (`crates/terrain/src/chunk.rs:523-561`) really is per-face, `value_at` (`:502-520`)
  really is `trilinear8` and therefore C0, and the continuity comment (`:358-365`) really claims a
  CHUNK edge only. §1's corrected row is accurate; only §5.4's replacement fails (A2-B2).
* The `SampleBox` halo really does pay an extra `height_m` per halo column
  (`crates/terrain/src/lattice.rs:262-272`), so a core column's slope at a one-cell halo really is
  free. §16's KEPT answer to B-D4 is correct.
* §7.3 rule 1's insistence that the slope come from an ADDRESS-DEFINED stencil is right and important:
  `height_m` is a pure function of one direction (`crates/terrain/src/height.rs:17-27`), and a
  stencil-dependent term would crack every chunk boundary.
* The single-threaded rule (§5.3) is the right call, and §14 M4 is the right gate to hold any future
  parallel decomposition to.
* §4.12's deletion of revision 1's "grow `crust_m`" is right: `floor_m` is at
  `crates/seed/src/ladder.rs:95` and every radial index `k` is measured from it (`:132-139`), so
  growing the crust would have shifted every cell on every body.
