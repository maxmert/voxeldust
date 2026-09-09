# Refutation A — domain 05, climate, biomes and weather

**Date:** 2026-09-08. **Target:** `docs/investigation/2026-09-08/landforms/05_climate_biomes_weather.md`
(revision 1). **Lens:** the laws and the code. Is every claim about the repository true, is every law
gate passed, is every number sourced?

**Method.** I read the target in full, then the files it cites, then the rulings it names. I recomputed
every arithmetic step. Every finding below names a file and a line, or the arithmetic.

**Score.** 3 blockers, 8 defects, 14 weaknesses, 3 notes. The document's physics is competent and its
weather split is right. Its two load-bearing engineering claims — the cost number that refuses the
per-column march, and the lane the charter rides — are both wrong, and one measured fact about the home
planet destroys its precipitation model on its own.

---

## Blockers

### B1. The per-octave cost is wrong by 2.1×, and it is the number that refuses the march

**Where.** §1 (the cost row), §5.1 (the grid build table), §5.4 (the refusal), §5.7 (the budget table).

The document states: *"the column pass about 1.5 ms per 62×62 columns at 14 octaves"*, sourced to
*"ruling V10; `crates/bins/examples/terrain_cost.rs`"*. Ruling V10 states no such number
(`docs/design/owner_decisions_2026-09-07_voxels.md:397-445` names 3.3 ms, 6.1 ms and 8 ms only). The
MEASURED number is in the landed slice-5 record:

> `docs/investigation/2026-09-07/slice_05_generator.md:372-374` — rung 0, **710 µs**, 14 octaves.
> Repeated in ruling V9: *"the coarse answer falls from 710 µs at rung 0 to 266 µs at the top rung"*
> (`owner_decisions_2026-09-07_voxels.md:382-383`).

`column_field` walks `CHUNK_EDGE × CHUNK_EDGE = 3 844` columns (`crates/terrain/src/chunk.rs:174-192`),
so the measured per-octave cost is `710 µs / (3 844 × 14) = 13.19 ns`, not 27.9 ns. The document's
figure is 2.11 times too high. 1.5 ms is not a measurement of anything: it is the OLD ESTIMATE of the
EXTRACTOR (`docs/investigation/2026-09-07/02_smooth_terrain.md:132`, "ESTIMATED 0.3–1.5 ms per surface
chunk"), carried across to a different pass.

**What it breaks.** §5.4 refuses the per-column upwind march with a number:

| Item | The document | Recomputed at 13.19 ns |
|---|---|---|
| a per-column march, 16 steps × 3 octaves | **5.1 ms — "It does not fit"** | **2.43 ms** |
| the surface chunk's headroom | 4.7 ms | 4.7 ms |
| the verdict | refused | **it fits** |
| the grid's coarse-height pass | 8.2 ms | 3.89 ms |
| the grid build, once per body | about 30 ms | about 25.5 ms |

The document says *"The grid exists precisely because of this number."* The number is wrong, so the
grid's justification is gone. The grid may still be right (a 16-step march does not deepen a shadow
across a whole range), but C-4 must be re-argued on the picture, not on a budget that does not bind.

**The fix.** Re-derive every ESTIMATED cost from 13.19 ns, restate the source as
`slice_05_generator.md:374`, and re-open C-4.

---

### B2. The charter rides a lane the owner closed by name, and the server needs no lane at all

**Where.** §0.1, §3.2, §3.4, §11 (the law-gate table), C-1.

§3.4 puts the charter on the parent's per-child row: *"the parent … states the child's row: placement +
look + recipe tag + CHARTER"*, and §3.2 justifies it as *"the charter rides beside two data that already
ride. It is a widening, not a new lane."* Both halves are false.

**(a) The recipe tag does not ride the parent's row.** `TAG_SURFACE` is the realm's OWN statement about
ITSELF: *"Stated by the realm about ITSELF (it rides `BodyStmt::SelfLook`, which only a RUNNING realm
may send …)"* (`crates/core/src/look.rs:50-56`). Ruling V6 row R-8 approves *"the realm's own surface
tag in its look bag"* (`owner_decisions_2026-09-07_voxels.md:314`). There is no precedent on the
parent's row.

**(b) The parent's per-child bag may carry ONE number, by a verbatim owner ruling.**

> `crates/core/src/look.rs:41-47`: *"THE ONE-RADIUS LAW (owner's ruling, verbatim) … **The parent's
> point-of-light bag may never carry more than this single radius — no surface, no detail, no mesh, no
> second number**"*.

Seventeen whole numbers on that bag is a direct breach. **SL3** says the same thing:
*"A parent's per-child message carries a PLACEMENT and nothing else, and must shrink toward that, never
grow."* §11's law-gate table lists sixteen laws and **omits SL3 entirely** — the one law the proposal
actually breaks.

**(c) On the server the ask is unnecessary — SL6 requires the local formulation first.** The code
already answers it:

> `crates/physics/src/worldgen/body.rs:47-51`: *"Never lowered onto `RealmRegion`, never on the wire:
> every process derives the whole forest from the seed at boot (**the complete SL6 answer — a planet
> shard computes its own mass, gravity, temperature and atmosphere locally, from the seed, with no
> message**)."*

A planet's own shard already holds its taxon. It needs no charter from its parent; it passes its own
charter into `BodyDefinition::from_seed` exactly as it passes the look radius today. The only host that
cannot derive it is the CLIENT (`crates/client/Cargo.toml:24-26`, `vd-physics` is dev-only — verified,
the document is right about that). So the real ask is one leg, realm → its own client, on the bag that
already carries the surface statement — a bag with a 1 200-byte budget
(`look.rs:78-80`, `SELF_LOOK_BUDGET_BYTES`) that 60 bytes fits inside.

**The fix.** Delete the parent→child leg. Put the charter in `SurfaceStmt` beside `frame` and
`generator` (`crates/core/src/look.rs:68-75`), which is the realm's own statement about what it IS —
which is also what the 2026-09-05 suit ruling actually blesses. Then add SL3 and HR1 rows to §11.

---

### B3. The home planet has almost no sea, and the whole precipitation model is driven by the sea

**Where.** §5.4, §5.1, §6, §9, §10. The document never mentions this fact.

MEASURED, in the landed slice-5 record and repeated in ruling V9:

> `docs/investigation/2026-09-07/slice_05_generator.md:388-392` — *"the home planet's sea level is
> 5 297 m under the ladder radius against a relief of 14 305 m, so the sea covers **about one column in
> a hundred** (0 to 5 of 300 sampled columns per rung)"*. Also
> `owner_decisions_2026-09-07_voxels.md:387-390`.

The document's precipitation law is
`P = supply(T_source, ocean_fraction_upwind) × cell_factor × orographic × continental(distance_to_sea)`
(§5.4). Two of its four terms are functions of a sea that is 1 % of the surface. On this body every
column's upwind fetch is land and every column's distance to the sea is thousands of kilometres, so the
supply term is near zero everywhere and the continental term multiplies it down again. The classifier
then reads about 0 mm of rain at every temperature, and §6.1's chart returns DESERT or COLD DESERT for
the entire planet. No forest, no wetland, no river, no beach — none of the reference picture.

This is INDEPENDENT of §10 (the airless-rock finding). Even after C-11 re-picks the home body, the sea
radius stays a seed draw between −40 % and +30 % of the relief (`crates/terrain/src/body.rs:186-190`,
verified), so an ocean is a coin flip and a 1 %-sea world is an ordinary outcome. The document's §9
table lists *"an ocean world"* as one case but never asks what THE home body's own sea does, though the
ruling states it as a fact for the owner.

**The fix.** Add a measurement (extend U1) that prints the sea fraction and the resulting biome
histogram for the candidate home body, and state honestly whether the sea draw must become a
hypsometric law rather than an offset. Until then the vista is not reachable on any body the recipe
draws by chance.

---

## Defects

### D1. §5.3's circulation needs an arcsine, which §4 refuses by name

§5.3 states `cells_per_hemisphere = clamp(round(90 deg / phi_H), 1, 6)` and then claims *"Every input is
a charter integer or a body constant. **The square root is inside the fence.**"* The square root is; the
ARCSINE is not. The formula produces `sin(phi_H)`, and to divide 90° by `phi_H` you must take
`asin`. Worse, §5.3's next paragraph reads the cell index by comparing `|x|` (a sine of latitude)
against *"the cell edges"* — the edges of `k` equal-width cells in DEGREES are `sin(k · phi_H)`, another
transcendental per edge. `crates/terrain/src/gf.rs:20-36` makes both a compile error, with three
`compile_fail` controls (verified). §4's own technique 4 refuses ad-hoc series. The document therefore
breaks its own rule three sections after stating it.

**The fix.** The cell count and the cell edges (in sine-of-latitude) are functions of the body alone.
They belong in the charter as integers, like `s2_insolation_q12`. Say so.

### D2. `WorldIdentity` is the wrong container for a per-body charter

§3.2: *"The charter is folded into the world identity. The measured half of the world tag already folds
a golden set of chunk digests. The charter joins it. A client with a different charter is refused at the
handshake."*

`WorldIdentity` is ONE pair for the whole world: `declared = fnv(GENERATOR_VERSION, universe_seed)` and
`measured = golden_self_check(home)` — eight chunks of the HOME body
(`crates/terrain/src/tag.rs:22-50`, `crates/terrain/src/digest.rs:1-4`). A charter is per body, arrives
per realm, and arrives long after login. A moon whose charter differs between two hosts cannot be caught
by a pair the client stated before it saw the moon. Folding a per-body datum into a per-world tag would
also make the tag change every time any body's charter changed, which reopens a world epoch for a moon.

**The fix.** The charter belongs in the PER-BODY digest (the chunk digest's key material, or a per-realm
refusal counter beside the existing foreign-recipe refusal, S7-3), never in the world tag.

### D3. The ladder gate is claimed for a discrete field with a continuous bound

§11, the ladder row: *"The biome at rung L is the biome of the coarsened hill, which is the coarse answer
of rung 0 within the same dropped-amplitude bound `dropped_bound_m` already proves
(`body.rs:264-270`)."*

`dropped_bound_m` (actually `crates/terrain/src/body.rs:274-276`; 264-270 is `relief_bound_m`) bounds a
HEIGHT difference in metres. A biome, a snow line, a sea-ice edge and a soil row are CLASSIFICATIONS with
thresholds. A bounded input error gives an UNBOUNDED output error at a threshold: one column that is
20 m under the snow line at rung 0 and 20 m over it at rung 4 flips from Grassland to Ice, and the
picture shows a white patch that appears when you fly away. That is precisely the SL8 defect ruling V10
forbids (*"no visible jump between detail levels"*). The document asserts the gate instead of designing
it.

**The fix.** State the real gate: the fraction of columns whose biome differs between rung L and rung 0,
measured over the golden set, with a stated ceiling; plus a blend width in the classifier wide enough to
absorb `dropped_bound_m(L)` at the rung being drawn.

### D4. `e = 0.2` cannot occur in this world

§5.5 and §9 use `e = 0.2` (*"2.25 times the light at periapsis … a whole extra season"*) as a world case.
The generator hard-caps eccentricity:

> `crates/physics/src/worldgen/config.rs:61-70` — `ECC_SIGMA = 0.03`, `ECC_CAP_SIGMAS = 4.0`, so
> `ecc_cap = 0.12`, and the doc-comment says *"the generator also hard-caps at `ecc_cap`"*.

The largest possible periapsis/apoapsis light ratio is `((1.12)/(0.88))² = 1.62`, not 2.25. The
arithmetic the document does is right; the world it describes does not exist. §9's "high-eccentricity
world" row must be deleted or re-based at 0.12.

### D5. The cube-sphere march is hand-waved at exactly the hard part

§5.1: *"The generator already solves exactly this problem for chunks at a face edge:
`crates/terrain/src/lattice.rs` holds `site_of`, the partner-face cell and the corner phantom. The
grid's march uses that same machinery."*

`site_of(body, key, a, b)` is keyed by a `ChunkKey` and resolves a HALO cell one step past a chunk's own
face edge (`crates/terrain/src/lattice.rs:121-160`). It is not a general "walk N cells along a 3-D
direction across faces" primitive. A 16-step upwind march at 41 km steps on a 128-per-face grid crosses
faces at arbitrary angles and can pass a cube CORNER, where the fourth cell of a 2×2 ring does not exist
(`lattice.rs:14-20`, the phantom column). Three further things the document does not state:

- the WIND VECTOR must ROTATE at every face seam (a face's tangent frame is not its neighbour's); the
  document gives no rule and no test for that rotation;
- a chamfer distance transform gives an exact distance only on an open raster under a total scan order;
  a CLOSED cube surface has no total order, so *"two chamfer passes"* (§5.1, §5.4) does not converge, and
  the residual error is exactly a visible band at a seam;
- the moisture relaxation's *"4 sweeps"* is deterministic only once the traversal order over the six
  faces is pinned. The document never pins it, yet U6 asks the no-drift gate to fold the grid's digest.

**The fix.** Name the walk primitive that does not exist yet, state the seam rotation rule, replace the
two-pass chamfer with an iterated pass whose count is fixed and whose order is stated, and add a numeric
seam test (which U11 half-asks for, on pictures).

### D6. The biome set is not defined, and the five-bit decision decides nothing

§6.2 lists seventeen names (`Ocean, SeaIce, Beach, Ice, Tundra, Taiga, TemperateForest,
TemperateRainforest, TropicalRainforest, Grassland, Savanna, Shrubland, Desert, ColdDesert, Wetland,
Alpine, Regolith` — I counted 17). The prose in the same paragraph says *"Fourteen chart biomes plus two
overrides"* (16). C-5 says *"Sixteen biomes in five bits"*. §6.1's chart draws eleven, one of which
(`WOODLAND`) appears in no list. Four different counts for one set.

The bit width is also a non-decision: the same section states *"The five bits live in the generator's own
enum and in the client's painter, not on the wire and not in the store."* A Rust enum has no bit width.
C-5 asks the owner to approve a field that does not exist.

### D7. §8.1 and §11 disagree about who authors the sub-solar point

- §8.1: *"THE ALMANAC … Computed by the OWNING realm and shipped in a small row."*
- §11 (SL1 row): *"The one placement-shaped datum, the sub-solar point, is the PARENT's to author, and
  the parent authors it."*

Both cannot be true. The sub-solar direction is a function of where the star sits relative to the body —
the body's own placement in the parent's frame, which SL1 clause 3 forbids the child to derive and clause
4 forbids it to pass on. The document never picks one, and it also never notices that the client already
receives the star's row in the same window: `S7-7` rules *"One directional light from the brightest
luminous row in the window"* (`owner_decisions_2026-09-07_voxels.md:502`), and `home_orbit`'s comment
already computes the star direction this way (`crates/bins/src/lib.rs:1004-1007`). The almanac's
sub-solar row may be data that already crosses.

### D8. The charter's determinism is probabilistic, and nothing names the authority

§3.2 argues *"thirteen orders of magnitude of margin"* between a `powf` drift (10⁻¹⁶) and the charter's
quantum (10⁻³). Then U5 concedes: *"A body that lands within a quantum of a boundary is a named defect,
not a surprise."* SL10 clause 4 is a MEASUREMENT that could have failed, not a probability. Three gaps:

- The climate changes the STRATA, the snow line and sea ice, so it changes the CELL, hence the chunk
  digest and the COLLIDER (§11 admits this in the collider row). A one-quantum disagreement is a
  different world shape, not a different colour.
- `worldgen/body.rs:47-51` says EVERY process derives the whole forest at boot. A gateway on aarch64 and
  a shard on x86-64 therefore each compute a charter for the same body. The document never says whose
  charter is authoritative when the shipped one and the derived one disagree.
- U5's test perturbs by 1 000 ulps over 1 000 bodies and asserts nothing moves. A world holds about
  150 000 systems × 9 planets. The measurement's population is three orders of magnitude short of the
  world's, and the document does not say so.

---

## Weaknesses

### W1. "The resolution is not a magic number" is hollow on every planet-sized body

§5.1 derives `target_cell_m = long_wave_m / 8` and calls it derived because `long_wave_m` is a seed draw.
Verified in `crates/terrain/src/body.rs:150-153`: `long_share = 0.25 + u·0.25`, clamped to
`[20 000, LONG_WAVE_CAP_M]`. On the home planet `radius × 0.25 = 837 690 m` and `× 0.5 = 1 675 380 m`,
so the clamp bites at **400 000 m for every draw** — the document says so itself. Any body over
1 600 km radius always clamps. So on every planet the "derived" wavelength is the CONSTANT
`LONG_WAVE_CAP_M`, and the grid resolution is a function of the radius only. The claim survives on small
moons and nowhere the vista lives.

### W2. A 41 km cell cannot make the picture's contrast, and C-4 drops the term that can

The reference picture's windward-forest / leeward-scrub contrast is across ONE ridge, a few kilometres
wide. The grid's cell is 41.12 km and its coarse height uses THREE octaves — wavelengths 400 km, 200 km
and 100 km (`body.rs:164-179`, halving). Those are continental swells, not ridges. §5.4 offers the local
form (one extra coarse evaluation at the upwind point, 0.32 ms per chunk by the document's own number,
0.15 ms recomputed at 13.19 ns) which DOES resolve a ridge, and then C-4 recommends the grid ALONE. The
answer is almost certainly both terms. §5.4 also labels the local-vs-grid question "C-4" while §13's C-4
asks grid-vs-closed-form: two questions, one number.

### W3. Isostasy is missing, and the measured relief makes the snow line absurd

The brief names isostasy. The word appears nowhere in the document, and gravity enters it only through
`c_p`. Yet the MEASURED relief of the home planet is **14 304 m** at a surface gravity of about
3.3 m/s² (`slice_05_generator.md:368`; the document's own §10 estimate for g). At the document's own
6.5 K/km lapse rate that is **93 K** of cooling between the sea and the peaks, so the permanent snow line
(§5.6) sits below most of the land and the planet is white. On the earth-like candidate the recipe draws
6–18 km of relief (`0.004 × 6 515 500 = 26 062`, clamped to 12 000, × [0.5, 1.5)), against Earth's ~9 km
at 1 g. A geologist would ask why a 0.5-g body carries Earth's relief and why an Earth-gravity body
carries twice it: isostasy makes maximum relief fall roughly as 1/g. The document invokes gravity for the
lapse rate and never for the law that actually sets a mountain's height.

### W4. C-11's cost is understated

§10 calls the re-pick *"the two-line fix"*. What actually moves: the golden table (2 592 digests,
`owner_decisions_2026-09-07_voxels.md:378-380`), the boot self-check (13.5 ms MEASURED, same place), the
world tag's measured half (`tag.rs:41-50`), and `home.rs`'s two literals with their pin test. The
candidate body is 6 515.5 km, so `n = r·π/2 ≈ 10 233 000` and the ladder gives **13 rungs**, not 12 —
a different rung count, a different octave count, a different golden shape, and a re-measured cost table.
It is on the ladder (`N_MAX = 1 << 26`, `crates/seed/src/ladder.rs:25`), so the fix is viable, but it is
not two lines and the document should say what it re-measures.

### W5. HR5 at 100 % collides with SL5's one-body test rule

`crates/terrain/src/home.rs:5` states: *"Every unit test of this crate runs on this body, never on an
invented one."* §11's HR5 row claims the classifier is covered because it is *"a table walk"* and
*"an ordered chain of comparisons"*. Seventeen biomes and six overrides cannot all occur on one body:
Regolith needs an airless body, SeaIce needs a frozen ocean, the terminator ring needs a locked world.
The document never says how those branches get covered without an invented body. (The answer probably
exists — a charter is DATA, so a test may build a charter for another REAL body of the world — but the
document owes it, and it is the exact HR5 gotcha CLAUDE.md warns about.)

### W6. The vegetation anchor has no home in the format

§7.2 emits `(cell, kind, param)` anchors and §6.2 says *"A record exists only for a cell a player
changed."* But the record's provenance field has THREE values — *"Terrain, biome feature, or placed"*
(`docs/investigation/2026-09-07/topic_00_format_sitting.md:54`) — and ruling V4 step 9(c) KEEPS *"the
modular record (one object on one cell, seed and stage parameters, a server-side skeleton for
collision)"*. V4 step 9(d) derives the COLLIDER from that skeleton, so an anchor is part of the SHAPE.
The document never says whether an anchor is folded into the chunk digest (`digest_of` folds
`stratum.code()` and `gap` only, `crates/terrain/src/digest.rs:36-42`), whether it is stored, or whether
it is refused. If a tree collides, its anchor must be in the digest or the two hosts disagree about where
a player can stand.

### W7. The dormant almanac has no carrier

§8.4 rests on *"Any process that holds the tick and the charter can state the season."* A dormant realm
sends no self-look — `crates/core/src/look.rs:52-54`: *"it rides `BodyStmt::SelfLook`, which only a
RUNNING realm may send — so a dormant realm is still never drawn"* — and the parent's marker bag may
carry only the one radius (B2). So no client ever holds a dormant body's charter. The section's promise
is fine as a SERVER fact and empty as a client fact, and the document does not separate the two.

### W8. The new soils may be exactly the indicator material S5-6 forbids

§7.1 adds laterite, peat and dune sand as biome-derived, publicly derivable rows. Ruling V4 15.3:
*"Ideally this info is not public"* — common ore is NOT public by default. Ruling S5-6:
*"bulk stock only from the seed; every ore is live state; **no indicator material**"*
(`owner_decisions_2026-09-07_voxels.md:369`; the same words in `crates/terrain/src/strata.rs:1-3`).
Laterite is the world's bauxite and nickel proxy; peat is fuel. A climate map anybody can compute that
names where laterite lies is an indicator material by definition. §11's seed-ruling row asserts
*"It decides no VALUE"* without testing its own new rows against S5-6.

### W9. §8.3 contradicts §3.4

§8.3: *"**Nothing crosses realm to realm.** A moon's weather is the moon's."* §3.4 crosses a
seventeen-number charter from a parent to its direct child. Both are in the same document. Even after B2
is fixed the sentence needs qualifying: the charter derives from the STAR's luminosity and the body's
ORBIT, which are the parent's facts.

### W10. The client's grid build is unbudgeted and ungated

§5.1: the grid *"runs on a worker before the first chunk of that body is built, on the server and on the
client alike"*, at 25–30 ms and 1.18 MB per body. There is no budget line, no gate, and no SL8
measurement for the moment a pilot approaches a body and the client must finish a grid before it may draw
one chunk. Ruling V10's crossfade and slice 7's chunk lane both assume a chunk can start now. Name the
budget or make the grid incremental.

### W11. The shoreline comparison mixes two units

§10 compares an insolation of 6.2 S⊕ against a threshold of 0.187. `cosmic_shoreline_retains(xuv_rel,
v_esc)` takes XUV irradiation and says so by name: *"`xuv_rel` is the XUV irradiation relative to
Earth's ([`xuv_rel_of`] — **NOT bolometric insolation**)"* (`crates/physics/src/taxonomy.rs:791-794`).
For a G star the enhancement may be 1.0 (G is the normalising class), in which case the arithmetic
survives — but the document never names `xuv_rel_of`, never states the G ratio, and presents a bolometric
number where the law takes an XUV one. U1 must print both.

### W12. §10's insolation chain assumes an unverified rung

The chain is `0.748 × 1.7⁴ = 6.2 S⊕`, which is right only if the home body sits at ladder rung 0
(verified: `ORBITAL_RATIO = 1.7`, `config.rs:58`; planets are pushed in rung order,
`generate.rs:1269`; insolation falls as `1/a²`, so rung 0 is `1.7^(4−0)` times rung 2's 0.748). But
`home_body` takes the first planet **the ladder ACCEPTS** — `find_map` over `BodyDefinition::from_seed`,
which returns `None` for a refused radius (`crates/bins/src/lib.rs:991-1002`). The document says
*"takes the FIRST planet row of the home system … which is the innermost"* and drops the filter. If rung
0 is refused, home is rung 1 at 2.16 S⊕ and every number in §10 shifts. U1 must print the rung.

### W13. C-2 invalidates a landed pin, and the tilt's carrier is left implicit

`crates/bins/tests/home_body_pin.rs:45-68` pins `POLE_AXIS = 2` **because it is the orbits' own axis**,
with the refuter's finding 4 named as the reason, and asserts an uninclined orbit stays at `z = 0` at
three instants. Obliquity breaks that identity by construction. The document never says the pin must be
rewritten, and never says WHERE the tilt lives: `StampedPose` already carries an `orient` quaternion
(`crates/core/src/pose.rs:919`, and the rotation law at 865-905), so the parent CAN author a tilted
orientation. That is the answer; the document leaves it implicit.

### W14. Citation drift

Several `file:line` citations are a few lines off, and two mislead:

| The document says | The code says |
|---|---|
| `height.rs:52` — *"`x` … the sine of the latitude … the code already computes it"* | line 45 computes `dir[POLE_AXIS].abs()`. It is `|sin lat|`, not `sin lat`. Harmless for `P2(x)`, wrong for a seasonal term that needs the hemisphere's sign. |
| `body.rs:264-270` for `dropped_bound_m` | 264-270 is `relief_bound_m`; `dropped_bound_m` is 274-276. |
| `body.rs:196-209` for the biome field's parameters | 196-201 is the strata table; `BiomeField` is 204-215. |
| `strata.rs:52-54` for the appended `Empty` | 52-54 are Sandstone/Limestone/Shale; `Empty` is line 61. |
| `strata.rs:180-198` for "four biomes exist" | that range is `StrataTable::at`'s first arm. |
| §1: the face edge of 5 263 360 is *"MEASURED by its own test"* (`home.rs:19-42`) | that test asserts 12 rungs, 14 octaves and the radius within 1 m. It never asserts the edge count. (The number itself is right: `n = 3 350 759.045 × π/2` snapped to `2570 × 2048 = 5 263 360`; I recomputed it.) |

---

## Notes

### N1. Jargon §2 promised to explain and then did not

§2 explains 22 words. §4, §5 and §7 then introduce, unexplained: *ulp*, *the Tetens form*, *a Legendre
coefficient* and *`P2`*, *the thermal Rossby number*, *an energy-balance climate model*, *a chamfer
pass*, *degree-days*, *a hyper-arid index*, *aspect* (used in §7.2 before its one-line gloss). The owner
asked to learn the terms; each of these needs its plain sentence and its example in the game's words.

### N2. §5.7's conclusion quotes the easy chunk only

*"the climate costs about 3 % of the headroom on a surface chunk"*. On a cave-dense chunk (6.1 ms
MEASURED) the headroom is 1.9 ms and the climate is 8 % of it. Quote both, as ruling V10 does.

### N3. The grid's 12 bytes per cell is borrowed from the wrong twelve

§5.1's memory column uses *"12 bytes"* per climate grid cell. Twelve is the CELL RECORD's width
(`topic_00_format_sitting.md:36-48`), which has nothing to do with a climate cell. The grid holds a coarse
height, a sea mask, a distance to the sea, a moisture and a wind; the document never lists the fields or
their widths, so 1.18 MB is a coincidence, not a computation.

---

## Checked, sound

- **The fence analysis (§4) is correct and correctly shaped.** `crates/terrain/src/gf.rs:1-40, 96-140`
  offers add, subtract, multiply, divide, negate, square root, floor, truncate, absolute value and
  comparison, with three `compile_fail` controls, no `mul_add`, no `min`/`max`, no transcendental. The
  three techniques (per-body constant, fitted polynomial, table-and-blend) are the right answers, and
  refusing a value-dependent Newton iteration on HR5 and drift grounds is exactly right.
- **The transcendental citations are true.** `generate.rs:571` (`u1.ln()`), `:598` (`.ln()`), `:633`
  (`.ln()`), `:807` (`pitch_rad.tan()`), `taxonomy.rs:616, 672, 684, 734` (all `powf`). Verified line by
  line.
- **`crates/terrain` depends on `vd-seed` and nothing else** (its `Cargo.toml`), and `vd-client` links
  `vd-terrain`/`vd-seed` with `vd-physics` as a dev dependency only. Both claims are true.
- **`POLE_AXIS = +Z` with no obliquity and no day length** (`height.rs:30-35`), cross-pinned in
  `home_body_pin.rs`, and `crates/bins/src/lib.rs:1005-1007` does say *"while the planet does not
  spin"*. True.
- **D-TAX-1 is real and honest in the code** (`taxonomy.rs:875-879`), and the house style for a fitted
  constant (`SHORELINE_COEFF = 6.0229`, Zahnle & Catling 2017, recomputed by a golden) is the right model
  to copy.
- **The ladder arithmetic reproduces exactly.** `n = 5 263 360`, radius `3 350 759.045`, top rung 11,
  12 rungs; `5 263 360/64 = 82 240 > 50 000` and `/128 = 41 120 ≤ 50 000`, so C = 128; 6 × 128² = 98 304
  samples. Every step recomputed and correct.
- **§10's body arithmetic reproduces.** `g = GM/R² = 3.30 m/s²` at 0.093 M⊕ and 3 350 759 m;
  `v_esc = 4 703 m/s`; `6.0229 × (4693/11180)⁴ = 0.187`. Chen–Kipping at `R ∝ M^0.279` gives 0.10 M⊕ for
  0.526 R⊕, close to the document's 0.093 and to Mars. Sound as an ESTIMATE.
- **The climate arithmetic checks.** `9.81/1004 = 9.77 K/km`; `6.5/9.77 = 0.665`;
  `((1.0167)/(0.9833))² = 1.069`; `((1.1)/(0.9))² = 1.494`; Held–Hou at `dH = 0.388` gives Earth exactly
  30° (a reverse fit, and the document says so); Earth's `s2 ≈ −0.48` is the published value; 1 401 steps
  of 0.1 K covers −80 °C to +60 °C.
- **§7.2's counts check.** A rung-0 chunk face is 62 × 62 = 3 844 m²; at 0.06 stems/m² that is 231
  anchors; 231 × 8 B = 1.8 KB against 405 KB of geometry.
- **★ §8.2, "a season must never invalidate a chunk", is the right law and is argued correctly.** The
  seed decides the shape and the permanent cover; the almanac decides the paint; lying snow a player can
  dig is a live diff on the same lane an edit uses. C-8 should be approved as written.
- **★ §8.4's wake-at-zero construction is the right no-seam shape.** The live layer contributing exactly
  zero at the waking tick is the correct SL8 answer, and U10 is the right measurement to owe.
- **The three-layer split (static / almanac / live) is sound** and it does map onto SL10 clause 7 and
  SL2 correctly, apart from D7's contradiction about who authors the sub-solar point.
- **The obliquity diagram (§5.2) is physically right.** A tilt does not rotate the annual-mean bands;
  it enters through `s2` and the seasonal swing, and `s2` changes sign near 54°. Correct, and correctly
  reconciled with V12's *"the recipe lives in the planet's own frame"*.
- **Q1–Q7 are the right open questions,** and Q6 (a terraformed ridge casts no new rain shadow) is an
  honest, correctly-reasoned limit rather than a hidden one.
