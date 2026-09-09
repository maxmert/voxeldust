# 06 — The landforms direction: the law audit, the integration, and the slice plan

**Date:** 2026-09-08. **Revision 3** (after refutations A and B of revision 2; §13 answers every finding).
**Domain:** every law the landforms work must pass, how it joins the code that already runs, and the
order in which it is built.
**Status:** an investigation report for the owner. It decides nothing by itself.
**Binding law read first:** CLAUDE.md (HR1–HR6, SL1–SL10), `docs/design/owner_decisions_2026-09-07_voxels.md`
(V1 SL10, V2.1–V2.9, V4, V5, V6 A/B/C/D/E, V8, V9, V10, V11, V12),
`docs/design/owner_decisions_2026-08-27_seed_and_secrecy.md`, `docs/design/owner_decisions_2026-09-02_reach.md`,
`docs/design/owner_decisions_2026-08-26_movement.md`, `docs/design/owner_decisions_2026-09-05_suit.md`,
and SL8 (a seam is a defect).

**The owner's task, in his words:** *"Make sure that we reach that quality on the picture for
earth-like planets (biomes can be different of course, should be dependent on the planet position,
spin, trajectory, size and gravity, etc.). It should be very believable, as we also should simulate
the weather."* And of the three pictures of slice 7: *"no orienters and no details at all … The only
thing I'm worried about is that the surface will not be interesting enough."*

**What this report is, and what it is not.** The sibling reports of this date design the MECHANISMS:
DOMAIN 02 the planet's layout, DOMAIN 03 the erosion and the rivers, DOMAIN 04 the detail rungs,
DOMAIN 05 the climate and the weather. This report is the law gate and the plan. It answers three
questions. Does the direction break a law? Where does it join the code that runs today? In what order
is it built, and what is measured before what?

Every number is marked MEASURED (with its source), COMPUTED (with its arithmetic, from the code) or
ESTIMATED (a model, not a result). Where a number does not exist, the text says UNMEASURED and names
the bench that would produce it.

**What revision 3 changed. Two refuters broke revision 2 in three places, and the direction turned.**

1. **The octave replacement was a defect, and it is deleted.** Revision 2 proposed to start the octave
   table at the macro cell, "9 octaves instead of 14". Refuter A proved that this turns a shipped
   Tier-A test red and stops the ladder coarsening above rung 9. The octave COUNT does not change at
   all. DOMAIN 03's slope spectrum re-weights the same fourteen octaves inside the same envelope, and
   the macro field REPLACES the coarse octaves' contribution at evaluation time. There is no octave
   saving, and this report no longer spends one.
2. **The build is not under a second. It is ESTIMATED 12 to 40 s.** Both refuters showed that revision
   2 costed a priority-flood heap at a sweep's rate, and refuter B showed that the flow routing must
   re-run as the ground changes. DOMAIN 03 re-priced the solve and reached the same range. **So the
   recommendation REVERSES: the owning realm SOLVES ONCE and SHIPS the artifact, coarse first; the
   client derives nothing of the macro field.** §4.1 and decision L1.
3. **A snapped float from an unfenced crate is not a safe input, and the cure already exists in the
   tree.** `crates/physics` carries no float fence, and `crates/physics/src/taxonomy.rs:21-24` states
   plainly that the cross-host bit-equality gate for its transcendentals is DEFERRED to SPIKE-6a.
   **So the body facts cross as INTEGERS the AUTHORING realm quantises — a BODY CHARTER — never as
   floats a receiver snaps.** §4.2.

Nine further defects are fixed: the manufactured quotation in §6.1, the emulated x86-64 leg the owner
already ruled non-probative, the column count, the area weight at a face edge, the relief draw that
still stood above its own bound, the shape bound that made a 100 km moon a billiard ball, the boot
canary's pinned stand-in, HR4's borrowed gates and the weather field's unpriced bytes. Two absences are
now named to the owner rather than passed over: **no ice sculpts anything** and **no crust dichotomy
makes a coastline**.

---

## 0. The recommendation, first

**The direction passes every law, and it passes ONLY with the rulings in §9.** Each one is small. Each
one is cheap today and expensive after slice 9.

1. **THE OWNING REALM SOLVES, AND SHIPS. THE CLIENT DERIVES NOTHING OF THE MACRO FIELD.** SL10 permits
   the client to derive it, and the cost refuses. ESTIMATED 12 to 20 s of one core, plausibly up to
   40 s, with 125 MB of transient memory (DOMAIN 03 §10, re-derived in §5). A player who logs in
   standing on the home planet cannot wait for it, and there is no proxy for the ground under his
   boots. The artifact is 9.83 MB with a 1.6 MB pyramid whose top level is 19 KB, so the approach ships
   coarse and the full field arrives only for the body the player is at. **Revision 2 recommended the
   opposite, on a cost model that was about thirty times low.**
2. **THE BODY FACTS CROSS AS A CHARTER OF INTEGERS, AUTHORED BY THE PARENT.** A believable climate needs
   gravity, mean insolation, equilibrium temperature, the atmosphere, the spin period, the obliquity,
   the orbit's eccentricity and the star's colour. The generator crate may name no motion crate, and the
   motion crate's floats have no drift gate (`crates/physics/src/taxonomy.rs:21-24`, SPIKE-6a). So the
   PARENT evaluates them once with full `libm`, QUANTISES each to a stated integer grid, and states the
   integers. The recipe reads integers. **No float from an unfenced crate ever enters the recipe**,
   which is the invariant `crates/terrain/src/body.rs:85-87` already states in its own words. The home
   body's charter is PINNED in `crates/terrain/src/home.rs` beside the radius bits, because the client's
   binary folds the world identity before it connects to anything.
3. **THE MACRO LATTICE IS SIZED IN METRES, NOT IN RUNGS.** Revision 2 sized it as `2^E` cells per face
   edge. That gives a 2 454 m skeleton on an airless moon that has no rivers and a 19 544 m skeleton on
   an Earth-sized planet that does — the resolution runs backwards. DOMAIN 03's rule is adopted:
   `n_macro` is the divisor of `N` whose node size is closest to a stated `MACRO_CELL_TARGET_M`
   (recommended 8 192 m), clamped to at least 8. COMPUTED on the home planet: 640 nodes per face edge,
   8 224 m per node, 2 457 600 nodes. Every body gets about the same METRIC skeleton, and the division
   is exact, so nothing is ragged at a seam.
4. **THE RELIEF BECOMES TWO BOUNDS, THE DRAW GOES INSIDE, AND THE SMALL-BODY BOUND IS CALIBRATED ON A
   SMALL BODY.** `relief = draw × min(strength bound, shape bound)` with `draw ∈ [0.5, 1.0]`, so a bound
   is a bound. The strength bound is `σ ÷ (ρ_crust × g)`. The shape bound is a share of the radius set
   from an OBSERVED small body, not from Earth: Vesta carries about 20 km of relief on a 260 km radius,
   that is 0.077 R. COMPUTED: the two bounds cross at a radius of about 939 km, so a large body is
   strength-limited and a small body is shape-limited, which is the right physics in both directions.
   Today's `0.004` is Earth's own ratio in disguise and it would leave a 100 km moon with 400 m of
   relief.
5. **THE OCTAVE TABLE DOES NOT SHRINK.** The macro field replaces the coarse octaves' CONTRIBUTION
   inside the same envelope (DOMAIN 03 §4.12), and the slope spectrum re-weights the fourteen octaves so
   the band the eye reads at 1 to 6 km carries five to ten times what it carries today. The ladder's
   M-16 gate stays green and untouched, and no cost is saved by dropping octaves.
6. **THE LANDFORM SLICES GO BETWEEN SLICE 8 AND SLICE 9.** They change the shape at every rung. Format
   D's tolerance is ZERO (ruling S5-2). The store lands at slice 9
   (`docs/design/owner_decisions_2026-09-07_voxels.md:385`, *"bind the store, slice 9"*), so before
   slice 9 the change is free.
7. **LIVE WEATHER IS NOT SHAPE, AND IT HAS THREE LAYERS, NOT TWO.** The static CLIMATE is in the recipe.
   The ALMANAC — the season, the sub-solar point, the day's phase — is a CLOSED FORM ON THE UNIVERSE
   TICK, computed by the owning realm and shipped in a small row. The LIVE weather is the realm's own
   state. The almanac is what makes the owner's word *"trajectory"* visible: in an annual mean an
   eccentricity of 0.17 moves the temperature by 1.5 %, and a season moves a snow line by kilometres.
8. **AN AIRLESS BODY GETS NO RIVERS, AND IT GETS CRATERS INSTEAD.** Hydraulic erosion, a river graph,
   lakes and coasts are built ONLY for a body whose charter holds an atmosphere and liquid water. A dead
   moon gets a crater field, impact relaxation and thermal creep. A 100 km moon with smooth noise and no
   crater is not a believable airless body, and revision 2 gave it exactly that.

**Two absences the owner must rule on, not two details.**

- **NOTHING IN REVISION 2 SCULPTED ICE.** The reference picture's top third is a glacial skyline:
  cirques, arêtes, U-shaped troughs, hanging valleys. Above the snow line there is no liquid water, so
  the stream power law does not act there at all. Revision 2 mapped *"snow-capped ridges"* to the lapse
  rate, which is PAINT on a fluvial V-notch. DOMAIN 03's revision 2 has since added an ice pass; this
  report carries it into the plan and puts its cost to the owner (§9, L20).
- **NOTHING IN THE PLAN MAKES A CONTINENT.** Earth's two-humped hypsometry comes from two crust types of
  different density floating at different levels. Without it there is no continental shelf, no abyssal
  plain, and the land-sea split is hypersensitive to the water draw — which is exactly why the ocean
  fraction has to be tuned by hand (§9, L11). The tuning is the symptom, not the cure (§9, L21).

**The boot finding, in its final shape.** The DECLARED half of the world tag folds a `u32` constant and
every process pays nanoseconds (`crates/bins/src/lib.rs:970-979`). The MEASURED half evaluates EIGHT
CHUNKS of the home planet and has exactly two production callers: `crates/bins/src/bin/gateway.rs:193`
and `crates/bins/src/bin/client.rs:118`; MEASURED 13.5 ms
(`docs/investigation/2026-09-07/slice_05_generator.md:386`). So the GATEWAY — which owns no planet,
draws nothing and decodes no chunk — self-checks the home planet at boot, and so does every client,
before either connects to anything.

Once the height field reads a macro artifact, neither of them can do that by solving: the solve is 12 to
40 s. **The cure: the eight golden chunks are evaluated on the SHIPPED PATH with the home artifact's OWN
values for the nodes they read, committed as literals beside `HOME_PLANET_RADIUS_BITS`.** That is not a
stand-in; it is the real artifact's real bytes, pinned, exactly as the home planet's radius is pinned
today. The whole artifact is checked by its own digest when it ARRIVES on the bulk lane, and a kernel
canary folds the solve's own arithmetic in microseconds. Nobody solves at boot, and a structural control
proves the gateway holds no artifact.

---

## 1. The words, explained once

The owner asked to learn the terms. Each row gives the industry word, its plain meaning, and what it is
in the game's own words.

| Term | In plain words | In the game |
|---|---|---|
| **Hypsometry** | How much of a world's surface stands at each height. Earth's is two-humped: ocean floor, then continents. | The home planet's sea stands 5 297 m under the ladder radius and covers about one column in a hundred (MEASURED, slice 5). Earth's ocean covers 71 %. The hypsometry is what makes that number what it is. |
| **Crust dichotomy** | Two kinds of crust of different density. Light crust floats high and makes continents; heavy crust floats low and makes ocean floor. This is what gives Earth TWO humps. | The plan does not have it. Without it the home planet has one hump, no continental shelf, and a coastline that moves a long way when the water draw moves a little (§9, L21). |
| **Tectonic uplift** | The slow push that raises rock. It makes mountains. | DOMAIN 03 states there is NO separate uplift field: the coarse octaves ARE the seed's statement of where the land stands high, and the solve redistributes what the seed drew. |
| **Isostasy** | Crust floats on the mantle like a raft. Pile rock on it and the raft sinks; take rock away and it rises. | The flexural rebound pass on the macro lattice. It is why a valley the water cut does not simply keep sinking, and why the home planet's ranges stop growing. |
| **Base level** | The lowest height a river can cut to. Usually the sea. | The sea radius the seed draws. Every valley on the home planet ends there. |
| **Drainage basin** | All the ground whose rain runs to one river mouth. | The land a pilot flies over between two ridges. One basin is one orienter. |
| **Flow accumulation** | For each node, how much ground drains through it. A large accumulation means a river. | The number that decides where the pilot sees water. It is summed as WHOLE SQUARE METRES in a `u64`, so the sum has no rounding order at all. |
| **Discharge** | The water that actually passes, which is the rain that fell on the basin, not the basin's area. | Across a rain shadow the two differ by a large factor. The erosion must read discharge, or a desert basin is cut as deep as a rainforest basin of the same size (§6.1). |
| **D8 flow direction** | Each node sends its water to ONE of its eight neighbours: the steepest downhill one. | The arrow stored in a macro node. On a cube-sphere a node at a face edge has neighbours on ANOTHER face, and a node at a cube corner has seven neighbours, not eight (§3.1 rule 7). |
| **Priority flood** | Work inward from the sea, always taking the lowest rim node next, and raise every hollow to the height of the lowest point of its rim. Then no node is left with nowhere to send its water, and what stands under the raised surface is a lake. | The lake in the crater the hull lands beside. It is a HEAP over every node of the body, and it is the solve's single most expensive step (§5). |
| **Stream power law** | Rock is cut away in proportion to the water that passes and to the slope. Written `E = K · Q^m · S^n`. | The rule that turns a noise hill into a valley with a floor and shoulders. |
| **Erodibility (`K`)** | How easily a rock is cut. Granite is low, shale is high. | Per body, from the bedrock the registry names on that body's row (`crates/terrain/src/strata.rs`). Not a new constant. |
| **Hydraulic erosion** | Water cutting rock and carrying the pieces downstream. | It makes valleys. It is built ONLY for a body with water (§0 item 8). |
| **Thermal erosion (creep), and the talus angle** | Loose material slides down a slope that is too steep to hold it. The angle at which it stops is the talus angle. | It makes the scree at the foot of a cliff, and it stops a wall standing at 89°. A dead moon gets this, a crater field and nothing else. |
| **Glacier, cirque, arête, U-shaped trough** | Ice flows and grinds. It carves a bowl at its head (a cirque), leaves a knife ridge between two bowls (an arête), and widens a valley from a V into a U. | The reference picture's skyline is made of these. Water cannot make them, because above the snow line the water is ice. Without an ice pass the skyline is a fluvial V-notch with white paint on it (§9, L20). |
| **Equilibrium line altitude** | The height on a glacier where a year's snowfall exactly balances a year's melt. Above it ice grows; below it ice wastes. | The height at which the ice pass starts acting on the home planet's ridges. It comes from the climate, so the climate must run BEFORE the ice (§6.1). |
| **Differential erosion** | Hard rock stands out; soft rock wears back. | The cliff band, the mesa and the rock pillar in the reference picture. It needs the strata at a fixed RADIUS, not at a fixed depth below the surface (§3.4). |
| **Caprock** | A hard layer over a soft one. It protects what is under it and leaves a table or a pillar. | What the reference picture's pillars are made of, and what today's strata table cannot produce. |
| **Stratigraphic column** | The rock layers of a place, listed by their HEIGHT, not by their depth under today's surface. | What slice 8c must add so a cliff face shows bands. Today `StrataTable::at(biome, depth_m)` reads a depth (`crates/terrain/src/strata.rs:189`), so every layer follows the surface and no band ever outcrops. |
| **Drainage density** | How close together the streams are. On Earth, one to five kilometres. | Why an 8 224 m macro node alone cannot draw every stream, and why the small streams are closed-form detail keyed to the skeleton. |
| **Slope spectrum** | How much of a landscape's roughness sits at each wavelength. Real eroded ground is NOT self-similar: its roughness PEAKS at the valley spacing. | The answer to *"not interesting enough"*. COMPUTED by DOMAIN 03 from `crates/terrain/src/body.rs:157-183`: today the recipe spends 6 000 m of amplitude on one 400 km wave and 47 m at 3 km. The spectrum keeps the same total and moves the amplitude into the band the eye reads. |
| **Insolation** | How much starlight a body receives per square metre over a year. | The home planet's `insolation_rel` in the taxonomy row today (`crates/physics/src/taxonomy.rs:932`). |
| **Albedo** | The share of light a surface throws back. Snow is high, forest is low. | Already drawn per body (`bond_albedo`). It feeds the temperature. |
| **Obliquity** | The tilt of a body's spin axis against its orbit. Earth's is 23.4°. It makes the seasons and it sets how strong the pole-to-equator difference is. | It does NOT exist in the code. It enters the charter as a quantised COSINE, and it does NOT move `POLE_AXIS` (§4.2). |
| **Eccentricity** | How far an orbit is from a circle. | The owner's word "trajectory". In an ANNUAL MEAN it changes the temperature by 1.5 % at `e = 0.17`. What a player sees of it is the SEASON, which is the almanac's, not the recipe's (§3.14). |
| **Escape velocity** | The speed a thrown stone needs to leave a body for good. `sqrt(2GM/R)`. | Already derived per body (`crates/physics/src/taxonomy.rs:755-758`). It is the physical driver of how lumpy a small body stays, because it decides how far impact ejecta travels. |
| **Lapse rate** | How fast air cools with height. About 6.5 °C per kilometre on Earth. | Why the ridge in the reference picture is snow-capped and the valley below it is green. |
| **Hadley cell** | The large loop of air that rises at the equator and falls near 30° of latitude. It puts the deserts there. | Why the home planet's deserts should sit in bands and not in blobs of noise. Under the fence the band is a test on a direction component, never on an angle: 30° is `|dir·pole| = 0.5`. |
| **Coriolis** | The bend a spinning world puts into moving air. It sets the prevailing wind. | The Coriolis parameter is `2 × spin rate × dir·pole` — a multiplication, with no trigonometry, because `dir·pole` is already the sine of the latitude. |
| **Orographic lift** | Air pushed up a mountain. It cools, and it rains. | The wet side of the ridge. |
| **Rain shadow, and upwind depletion** | The dry side, behind the ridge. A CONTINENTAL interior is dry for a different reason: air that crossed three ranges lost its water at the first. That needs a march ALONG the wind, not a local slope test. | DOMAIN 05 MEASURED the difference: a per-column upwind march costs 5.1 ms of the 8 ms chunk budget and is REFUSED; the march runs once on a coarse climate grid instead, and each column reads it (§3.6). |
| **Clausius–Clapeyron** | How much water air can hold before it rains. It rises steeply with temperature — an exponential curve. | The float fence bans `exp`, so this curve ships as a VENDORED polynomial or a table with a stated maximum error (§3.1 rule 5). |
| **Whittaker classification** | Naming a biome from two numbers: temperature and rainfall. | It replaces today's four biomes with sixteen the client's art kit reads. Five bits, and the cell record does not change. |
| **Almanac** | A table of where the sun stands and what the season is, for each day of the year. | The closed form on the universe tick that the owning realm ships: the season's phase, the sub-solar point, the day's phase. It is what makes spin, tilt and eccentricity VISIBLE. |
| **Face seam** | The join between two of the six faces of the cube-sphere. Across it the two grids' axes swap, and one may run backwards. | `crates/seed/src/seam.rs` holds the table of all 24 directed edges. A river must cross it, and the macro height must be single-valued on it or the pilot sees a cliff 5 264 km long. |
| **Area weight** | A node's true surface area. On a cube-sphere it is not the same everywhere. | COMPUTED from the bend `W(a) = k₁a + k₂a³ + k₃a⁵` (`crates/seed/src/bend.rs:43`) with the area density `W'(a)·W'(b) ÷ \|n + W(a)u + W(b)v\|³`: a node at a FACE-EDGE MIDPOINT is **0.702** of a face-centre node, and a node at a cube CORNER is **0.758**. Flow accumulation counts AREA, so it must weight by this, or every river drifts toward the face centres. |
| **Catmull-Rom interpolation** | A way to read a value between four grid points so that the RESULT and its SLOPE are both smooth. | How a column reads the macro height. A simpler two-point read is smooth in value but not in slope, and its slope jump would show as a straight crease every 8 224 m across a 50 km vista. |

---

## 2. What the code holds today (the only current truth)

Every line citation below was re-checked against the working tree on 2026-09-08. The rows marked ★ are
NEW in revision 3, and each one closes a refuter's finding.

| Fact | Where | What it means for landforms |
|---|---|---|
| The recipe takes exactly TWO inputs: a seed and a look radius. | `crates/terrain/src/body.rs:140` (`from_seed(seed, look_radius_m)`) | Gravity, insolation, spin, obliquity and eccentricity have no way in. §4.2 opens one. |
| The radius is an input from OUTSIDE the fence, made harmless by a SNAP. | `crates/terrain/src/home.rs:49-77`: a radius moved by 1 000 ulps gives the same body; a whole snap unit changes it | This measures the SNAP'S TOLERANCE. **It does NOT measure the cross-host SPREAD the snap must absorb**, and revision 2 called it a measurement of the wrong thing. §4.2. |
| ★ The home planet is stated as TWO PINNED LITERALS: the seed and the radius's exact bits. | `crates/terrain/src/home.rs:14-22` (`HOME_UNIVERSE_SEED`, `HOME_PLANET_SEED`, `HOME_PLANET_RADIUS_BITS`), cross-pinned by `crates/bins/tests/home_body_pin.rs` | **The project has already chosen the cure for an unfenced float: pin the bits.** The charter follows it (§4.2). |
| ★ `crates/physics` carries NO float fence, and the taxonomy's own doc comment defers its drift gate. | no `crates/physics/clippy.toml`; `crates/physics/src/taxonomy.rs:21-24`: *"the cross-host bit-equality gate is SPIKE-6a (step-2/P4)"*; `worldgen/generate.rs` uses `ln`, `cos`, `sin`, `powf` | Every fact the climate needs is produced by an openly unproven chain. **A receiver must never re-derive one.** |
| ★ The recipe's own invariant forbids exactly what revision 2 proposed. | `crates/terrain/src/body.rs:85-87`: *"a body is DRAWN from a seed … and never assembled from numbers computed elsewhere, so no float from an unfenced crate can enter the recipe as a body"* | The charter obeys it: integers cross, not floats. |
| ★ The CLIENT BINARY runs the forest's unfenced draw at boot, although the client LIBRARY links no motion crate. | `crates/bins/src/bin/client.rs:117-118` calls `vd_bins::world_identity` → `home_body` → `home_system_boot`; `crates/client/Cargo.toml:24-26` has `vd-physics` as a DEV dependency only | The home body's charter must be PINNED, or a straddle on one chip refuses that client at login. |
| The ladder's edge count is `N = R · π/2`, snapped to a multiple of `2^top`. | `crates/seed/src/ladder.rs:76` (`let n_ideal = radius_m * FRAC_PI_2;`), `:59-66` (`snap`) | COMPUTED: home `n_ideal = 5 263 980`, `n = 5 263 360`, top rung 11, twelve rungs. Confirmed independently by `docs/investigation/2026-09-07/slice_06_extractor.md:340` (the last rung-0 chunk is 84 892, and `84 893 × 62 = 5 263 366`). |
| The top rung is chosen by a CHUNK COUNT, not by a length. | `crates/seed/src/ladder.rs:21-23` (`CHUNK_EDGE = 62`, `TOP_RUNG_CHUNKS = 64`), `:48-57` (`top_rung_for`) | The top rung always holds between about 1 985 and 3 968 cells per face edge, on a moon and on a giant alike. **This is why a rung-derived macro grid gives the same cost on every body.** §4.3. |
| ★ `octaves_at` keeps `count − rung` octaves, FLOORED AT ONE, and a shipped Tier-A test asserts that every rung keeps strictly fewer than the rung below it. | `crates/terrain/src/body.rs:251-261`; the test `every_rung_sums_strictly_fewer_octaves_than_the_rung_below_it`, `crates/terrain/src/body.rs:355-365`; the doc comment at `:340-342` calls it *"the gate"* (M-16) | **The home planet has 12 rungs and 14 octaves. Cut the octave table below 12 and the gate goes red at rung 10.** Revision 2 proposed 9 octaves and did not notice. The octave count does not change. §3.4. |
| The relief is a share of the radius with two hard caps, times a seed factor in `[0.5, 1.5)`. | `crates/terrain/src/body.rs:147-149`: `0.004`, `clamp(200, 12_000)`, `Gf::HALF + draw_unit` | The factor multiplies OUTSIDE the cap, so the drawn relief EXCEEDS the cap. MEASURED on the home planet: cap 12 000 m, drawn relief 14 305 m (`slice_05_generator.md:368`). **Moving the multiplication without changing the draw's range does not fix it.** §3.4. |
| The octave amplitudes are scaled so their SUM is the relief, which is what makes the band's bound exact. | `crates/terrain/src/body.rs:180-185` | The slope spectrum re-weights them and keeps the same sum, so the envelope, `floor_m` and `band_m` are all untouched. |
| The height field is the ladder radius plus a sum of octaves; a coarse rung drops the fine octaves. | `crates/terrain/src/height.rs:17-27`; `crates/terrain/src/body.rs:253-261` | It is pure noise. It has no valleys, no drainage and no landmarks. This is the owner's *"no orienters and no details at all"*. |
| The band that holds the surface is derived from the drawn relief, and `relief_bound_m` is an EXACT bound. | `crates/terrain/src/body.rs:232-236`, `:263-268` | ★ The assertions at `crates/terrain/src/height.rs:92` and `:102` sit inside `#[cfg(test)]` and walk 400 sampled directions of ONE body. **The bound is exact by CONSTRUCTION, not by assertion**, and the macro field must keep it that way (§3.8). |
| ★ `crates/terrain/src/body.rs:234-235` already carries 64 m of spare crust and 64 m of spare room above. | the same lines | The channel's cut fits inside it. **Growing `crust_m` would lower `floor_m` and shift EVERY cell's radial index on EVERY body** (`crates/seed/src/ladder.rs:95,132-134`), so the band must not grow. |
| ★ `BodyDefinition` is `Copy`. | `crates/terrain/src/body.rs:88` | A resident macro artifact cannot live inside it. The client already holds `Arc<BodyDefinition>` (`crates/client/src/chunks.rs:71`); the crate's own API must stop being `Copy`, which touches `chunk.rs`, `height.rs` and `digest.rs`. Small work, and it belongs in the landing list. |
| The biome reads a pole component, height over the sea and two slow noises. Four biomes. | `crates/terrain/src/height.rs:40-64`; `crates/terrain/src/strata.rs` | No rain shadow, no lapse rate, no Hadley band, no obliquity. Note `dir[POLE_AXIS].abs()` at `height.rs:45` is a COSINE-like component, never an angle — the pattern the fence forces. |
| The strata are a layer cake measured DOWNWARD FROM THE SURFACE. | `crates/terrain/src/strata.rs:189` (`at(&self, biome, depth_m)`), read at `crates/terrain/src/chunk.rs:637` | A layer that follows the surface can never outcrop as a band across a slope. **No caprock, so no mesa and no rock pillar** — which the reference picture shows. §3.4. |
| `POLE_AXIS` is `+Z`, the body's own axis; an obliquity is named as a later slice. | `crates/terrain/src/height.rs:30-35` | The pole stays where it is. The PARENT tilts the body; the recipe needs only a scalar (§4.2). |
| The float fence bans every trigonometric, exponential and logarithmic function, `powf`, `powi`, `mul_add`, `min` and `max`. | `crates/terrain/clippy.toml`; the compile-fail control `just terrain-fence-control`; the link scan `just terrain-link-scan` | **The climate cannot name an angle or an exponential.** §3.1 rules 4 and 5. |
| `Gf` offers `+ − × ÷ √`, `floor`, `trunc`, `abs`, `lesser`, `greater`, `clamp`. | `crates/terrain/src/gf.rs:56-148` | Everything a landform layer needs, and nothing more. `m = 1/2` needs only `√`. |
| The taxonomy already derives mass, radius, insolation, equilibrium temperature, Bond albedo, an atmosphere, surface gravity and ESCAPE VELOCITY per body. | `crates/physics/src/taxonomy.rs:895-908`, `:881-891`, `:750`, `:755-758`, `:762`, `:932` | Every fact the climate needs EXISTS. It lives in a crate the recipe may not name, and it is computed by unfenced floats. |
| A spin period, an obliquity and a water inventory exist NOWHERE. | MEASURED by grep over `crates/physics/src/worldgen/*.rs` and `taxonomy.rs` | Three new seed draws are owed. Their owner is an owner decision (§9, L7). |
| The registry holds density, work of fracture and melting point per substance, from cited tables. | `crates/core/src/registry/substance.rs:66-79` | The strength bound needs the YIELD STRENGTH, which is a FOURTH column. Work of fracture is not it. **Exactly ONE of the strength bound's three terms is already a cited registry row: the crust density.** |
| The registry's identity digest folds EVERY PHYSICAL COLUMN, not only the keys. | `crates/core/src/registry/digest.rs:1-9` | **Appending a yield-strength column MOVES the digest and refuses every store written before it.** Free before slice 9; a world epoch after it. |
| The surface tag carries a frame and the DECLARED generator tag, and the comment forbids the measured half by name. | `crates/core/src/look.rs:50-73`: *"Never the measured half — a chip difference refuses a lane, never a look"* | **A macro-artifact digest may NOT ride the look.** §7 keeps that ask withdrawn. |
| The whole self-look bag — outline, luma and surface together — must fit one conservative datagram. | `crates/core/src/look.rs:78`: `SELF_LOOK_BUDGET_BYTES = 1200` | The charter must be counted against it. ESTIMATED 24 bytes packed (§7). |
| The surface tag is stated ONCE PER REALM ON CHANGE, carried retained, never on a keep-alive. | `crates/core/src/look.rs:53-54` | **Weather changes continuously, so weather may NOT ride the surface tag.** It needs its own lane with its own rate. §7. |
| The MEASURED half of the world identity evaluates 8 chunks of the home planet, in the gateway and the client only. | `crates/terrain/src/tag.rs:10,30-34,44-49`; `crates/terrain/src/digest.rs:181`; the callers `crates/bins/src/bin/gateway.rs:193`, `crates/bins/src/bin/client.rs:118`; MEASURED 13.5 ms | The tag's own words: *"the home body's eight golden chunks evaluated by this binary on this chip"*. §3.9. |
| The golden tables pin 3 240 cell rows and 1 082 triangle rows of the home planet, on FIVE legs. | `just terrain-pin` runs three host legs (`justfile:744-747`); `just terrain-legs` runs two Docker legs (`justfile:756-758`) | ★ **The fifth leg is EMULATED x86-64, and the owner has already ruled it non-probative**: S5-5 (`owner_decisions_2026-09-07_voxels.md:375`) says *"emulation as a smoke test now; a real machine before 'no drift' is called satisfied"*, and `justfile:753-755` says the same and names `D-TERRAIN-1`. §3.1. |
| The budget is 8 ms of worker time per rung-0 chunk. MEASURED: a plain surface chunk 3.30 ms (1.98 sample box + 1.32 extraction), the worst named chunk 6.11 ms (4.38 + 1.73), the column pass 710 µs at 14 octaves and 266 µs at 3. | ruling V10; `docs/investigation/2026-09-07/slice_06_extractor.md:344-350`; `slice_05_generator.md:372-378` | ★ The sample box holds `64 × 64 = 4 096` columns, not 4 268 (`slice_06_extractor.md:97,118,233`: *"six per cent more columns"*, and the ten per cent is CELLS). ★ The cell pass is now 716 µs, not 578 (`slice_06_extractor.md:361`). |
| The client links the recipe and runs chunk work on an injected PER-CHUNK worker seam whose Tier-A implementation runs INLINE. | `crates/client/src/chunks.rs:84-104`; MEASURED link cost +51 280 bytes, +12.7 s of cold build | Under the SHIP recommendation the client needs no whole-body job at all, which deletes a piece of work revision 2 owed. |
| The client draws ONE rung per realm behind a dev flag, which slice 8 must delete. | `D-TERRAIN-3` 🟥, `docs/design/DEFERRED.md:7775` | The owner refused a surviving flag. §6.1 keeps slice 8 first for that reason alone. |
| A planet does not spin, and the picture gate states the avatar's facing by hand. | `crates/bins/src/lib.rs:1005-1008`; `D-TERRAIN-4` 🟥 | Day and night, and therefore the almanac and the weather, need the spin the parent authors. |

---

## 3. The law audit, law by law

### 3.1 SL10 — one generator, two hosts, no drift

**The direction passes, and it brings EIGHT drift risks that need rules.** Revision 1 named three,
revision 2 seven; the eighth comes from refuter B.

- A macro artifact, a river graph, a climate field and the closed-form detail are all functions of
  `(seed, charter, address)`. None of them reads a tick, a pose or any live state. Clause 1 is kept.
- ONE crate computes them, and both hosts link it (clause 2). Nothing is ported.
- The server computes collision on the same shape (clause 5), because the collider reads the extractor,
  the extractor reads the density, the density reads the height, and the height reads the artifact.
- Everything the seed does not decide still crosses as a one-hop diff (clause 6).
- **Under the SHIP recommendation the client does not COMPUTE the macro artifact at all**, so the
  artifact's drift risk narrows to server against server, plus the build-time legs. The risks below
  still bind, because the shard that solves must agree with the shard that re-solves after a store is
  lost, and because the owner may yet choose the derive road.

**THE RISK: an iterative computation can drift where a closed form cannot.** Today every value is a
straight-line expression over one column. A solve is a loop over a whole body, and eight things inside
it can move a byte.

1. **A power function.** The stream power law `E = K · Q^m · S^n` is usually written with `m ≈ 0.5` and
   `n = 1`. At exactly `m = 1/2` and `n = 1` it needs only `+ − × ÷ √`, and V1.4 allows all five.
   **Rule: the exponents are `m = 1/2` and `n = 1`.** The published range for real catchments is
   `m/n ≈ 0.35 … 0.6`, so this is inside the observed range, not a distortion made for the machine.
2. **A reduction whose order depends on the thread count.** If eight workers sum a basin's water in the
   order they finish, two hosts disagree. **Rule: every reduction runs in one fixed order, stated by
   the address, never by the schedule. THE WHOLE SOLVE IS SINGLE-THREADED.** DOMAIN 03 removes most of
   the risk at the root by summing areas as WHOLE SQUARE METRES in a `u64`, where the order does not
   matter at all.
3. **A sort with an unstable tie.** The flood and the accumulation walk the nodes by height. Two nodes
   at the same height must break the tie by their integer address, never by the sort's internals.
   **Rule: the comparator is total, the height is an integer (1/16 m in an `i32`), and the address is
   the last key.**
4. **AN ANGLE.** The fence bans `sin`, `cos`, `asin`, `atan2` and `to_radians`
   (`crates/terrain/clippy.toml`). **Rule: no angle ever crosses the fence and none is ever formed. A
   direction, a direction COMPONENT or a COSINE crosses instead.** `dir[POLE_AXIS]` is already the sine
   of the latitude (`crates/terrain/src/height.rs:45` uses it today); the Hadley band's 30° is the test
   `|dir·pole| ≥ 0.5`; the Coriolis parameter is `2 × spin rate × dir·pole`; the annual-mean insolation
   profile is a second-order polynomial in `dir·pole` whose one coefficient is a function of the
   obliquity's COSINE. The obliquity therefore enters the charter as a quantised cosine (§4.2).
5. **AN EXPONENTIAL OR A LOGARITHM.** Saturation vapour pressure and the atmosphere's pressure with
   height are exponentials, and `exp` and `ln` are banned. **Rule: every such curve ships as a VENDORED
   POLYNOMIAL in Horner form or a stated table, with a STATED range and a STATED maximum relative
   error, and a unit test that asserts the error against a table of reference values.** A fit is a
   function of its input alone, so it is byte-identical by construction.
6. **A CONVERGENCE TEST.** "Erode until the landscape reaches grade" compares a float against a
   threshold. If anything else drifts by one ulp, one host runs one more pass than the other, and the
   two answers differ by a whole pass. **Rule: every pass count is a STATED CONSTANT of the recipe,
   never a convergence test.** This is not an excuse for a magic number; determinism REQUIRES it. The
   constants are chosen from a convergence study made once, off the shipped path, and recorded (§9,
   L13). DOMAIN 03's recommended set: `PASSES = 40`, `CLIMATE_EVERY = 10`, `FLOOD_EVERY = 10`,
   `ISOSTASY_EVERY = 5`, `TALUS_PASSES = 8`.
7. **THE FACE SEAM AND THE CUBE CORNER.** The macro lattice lives on a cube-sphere. A D8 flow
   direction, a priority flood and a flow accumulation are NEIGHBOUR algorithms, and on this lattice
   the neighbours cross faces. `crates/seed/src/seam.rs` holds the table of all 24 directed edges with
   the partner face, the partner side and the reversal flag, and its own test
   (`crates/seed/src/seam.rs:309-320`, `three_faces_meet_at_every_cube_corner`) proves a corner node's
   neighbourhood is degenerate: **three faces meet, so a corner node has SEVEN neighbours, not eight.**
   **Rules:** (a) a flow direction is stored as a NEIGHBOUR INDEX in the node's own face frame, and the
   seam table converts it when it crosses; (b) a corner node's neighbour set is stated by the corner
   table, and a test asserts it is seven; (c) flow accumulation counts AREA, weighted by the bend's own
   area factor; (d) ruling A5 (*"the generator always places a landform at all eight cube corners"*,
   `owner_decisions_2026-09-07_voxels.md:184`) is a GATE on the artifact: a picture at each of the eight
   corners, and a test that no corner node is a sink.
   **The area weight, CORRECTED.** Revision 2 wrote *"about 6.5 % narrower at a corner"* and derived it
   from the wrong quantity. COMPUTED here from `crates/seed/src/bend.rs:24-30,43,251-259`, with
   `W'(a) = k₁ + 3k₂a² + 5k₃a⁴` and the area density `W'(a)·W'(b) ÷ s³`, `s = |n + W(a)u + W(b)v|`:

   | place on a face | area density | ratio to the face centre | linear width, ratio |
   |---|---|---|---|
   | centre `(0, 0)` | 0.616 850 | 1.000 | 1.000 |
   | **face-edge midpoint `(1, 0)`** | 0.432 739 | **0.702** | 0.992 |
   | cube corner `(1, 1)` | 0.467 391 | 0.758 | 0.935 |

   Three things follow, and each was wrong in revision 2. The area error is **30 %**, not 6.5 %; the
   6.5 % figure was the corner's LINEAR width. The EXTREME sits at the midpoint of a FACE EDGE, not at
   a cube corner — which is exactly where a river must cross. And `W'` RISES from 0.785 at the centre
   to 1.558 at `a = 1`; it does not fall. A 30 % area error bends every basin toward the face centres,
   which is the failure rule 7 exists to prevent.
8. **THE FLOW ROUTING GOES STALE. (new, from refuter B)** After forty stream-power passes the ground is
   not the ground the flow directions were built on. Divides migrate, a basin captures its neighbour, a
   filled hollow drains. If the routing runs once, the river graph describes a hill that no longer
   exists, and the pilot sees water running up a shoulder and stopping in the air. **Rule: the
   re-routing period is a STATED CONSTANT (`FLOOD_EVERY`), it is part of the world tag, and the cost
   model multiplies by it.** Revision 2's cost model ran the routing twice in total and priced it as a
   sweep. §5 prices it properly, and the answer changed the recommendation.

**The gate, stated so that it can go red — and stated honestly.** The artifact's digest for the home
planet is equal in debug, in release, under `-C target-cpu=native` (that is `just terrain-pin`, three
legs), on aarch64 Linux and under EMULATED x86-64 (that is `just terrain-legs`, two legs).

**Of those five legs, three are one host in three build modes, one is aarch64 Linux, and the fifth is an
emulator the owner has already ruled non-probative.** S5-5
(`docs/design/owner_decisions_2026-09-07_voxels.md:375`): *"emulation as a smoke test now; a real
machine before 'no drift' is called satisfied."* The `justfile` says the same and names the entry:
*"a smoke test: it can find a drift, never prove its absence — a real x86-64 machine is owed, DEFERRED
D-TERRAIN-1"* (`justfile:753-755`). A forty-pass accumulation over 2.46 million nodes is precisely the
computation an emulator's rounding is least likely to reproduce faithfully. **So this report states
plainly: either `D-TERRAIN-1` (a real x86-64 machine) lands as a PRECONDITION of the erosion slice, or
the landform arc ships with "no drift" UNSATISFIED by the owner's own definition.** That is an owner
decision (§9, L22), not a detail for the measurement table.

*Example.* A pilot follows a river north on face `+X` of the home planet. Where the face ends the river
must continue on face `+Z`, with the two grids' axes swapped. The seam table says which node that is.
Without rule 7 the river ends in a wall, and the priority flood fills the whole basin into a lake.

### 3.2 The seed ruling — shape may be published, value may not

`owner_decisions_2026-08-27_seed_and_secrecy.md`: anything a fixed seed alone decides must be safe to
publish; anything valuable must depend on live state. Ruling S5-6 already binds the recipe: bulk stock
only, every ore is live state, no indicator material.

**The audit finds ONE new temptation and refuses it.** A river carries placer deposits in the real
world, and a river bed is the obvious place to put value. That would make a seed-derived map of riches
— a treasure map, refused by name. **Rule: a river, a lake, a delta and a beach are SHAPE. They carry
no ore, no gem and no marker of one.** The graph may still decide where SAND, GRAVEL and CLAY lie,
because those are bulk stock. A prospector reads a river because a river is where a person digs, not
because the seed put value there.

**And one thing the ruling makes legal.** Fertile land, a sheltered harbour and a mountain pass ARE
seed-derived, and to publish them is safe. The ruling's own words: *"An expensive gate buys ROOM, not
treasure … a new galaxy is worth reaching because nobody has taken it yet, which is live state."* Good
land is room. Its value is that nobody has built there, and that is live.

### 3.3 SL5 — one world, no scale knob, no variant

- The macro lattice's size is a stated METRIC target plus the body's own `N` (§4.3). There is no
  resolution setting, no preset per body class and no test world.
- **The fast-test cure of revision 1 is still WITHDRAWN, and revision 2's replacement is now split in
  two.** Refuter B found the missing half.
  - **THE KERNELS ARE TESTED ON STATED NEIGHBOURHOODS.** The priority flood, the flow router, the
    stream-power sweep and the talus pass are pure functions over a stated NEIGHBOURHOOD, addressed
    exactly as a body's own nodes are addressed. A unit test hands the shipped kernel a small stated
    neighbourhood — a face centre, a face seam, a cube corner, a pit, a plateau, a node with no downhill
    neighbour — and asserts the answer. The kernel under test IS the shipped kernel.
  - **THE DRIVER IS TESTED ON A REAL SMALL BODY OF THE WORLD.** The pass loop, the face walk, the seam
    conversion, the heap's own management, the allocation and the free are NOT kernels, and a
    neighbourhood test never reaches them. Under §4.3's rule a 50 km body of THE world gets
    `n_macro = 8`, that is 384 nodes; a 2.5 km rock gets the clamp. **So the crate's own unit tests
    solve a REAL body of THE world in milliseconds.** That is not a variant and not a test world: it is
    the smallest round body the world already holds.
- **The whole-body run on the home planet stays a gate, not a unit test.** It runs in
  `crates/bins/tests/home_body_pin.rs`, which already cross-pins the home body.

### 3.4 No magic numbers

The rule: world parameters are seed-derived; entity properties are per entity; operational parameters
live in one config struct. A draw's RANGE is part of a recipe and is lawful. A STATED RESOLUTION is also
lawful; the code already holds four (`OCTAVES`, `SHORT_WAVE_M`, `CHUNK_EDGE`, `TOP_RUNG_CHUNKS`). A
range that STANDS IN for a physical fact the world already holds is a magic number. Here is the audit of
`body.rs`.

| Constant today | Where | Verdict | What it becomes |
|---|---|---|---|
| relief share `0.004`, caps `200 m` and `12 000 m`, factor `[0.5, 1.5)` | `body.rs:147-149` | **MAGIC, and MIS-SHAPED** | See "the two bounds" below. |
| coarsest wave `0.25–0.5 × radius`, clamped to `20 km … 400 km` | `body.rs:151-153` | **KEPT, and re-weighted** | ★ **Revision 2 proposed to replace it with the macro cell. That is WITHDRAWN** (§3.7). The octave TABLE is unchanged; the slope spectrum re-weights the amplitudes, and the macro field replaces the coarse octaves' CONTRIBUTION inside the same envelope. |
| roughness `k_rough ∈ [0.45, 0.55)` | `body.rs:155` | **REPLACED by the slope spectrum** | A single ratio makes a self-similar surface whose per-octave slope is a constant 0.094 at every scale (COMPUTED by DOMAIN 03). Real eroded ground peaks at the valley spacing. The spectrum's peak and width are physical facts of the body, and they are measured against a real elevation model. |
| short wave `30 m` | `body.rs:25` | Lawful (a stated resolution) | Kept. Better stated AS the finest rung's own size. |
| sea offset `−0.4 … +0.3 × relief` | `body.rs:188-190` | **MAGIC** | The sea level follows from a WATER INVENTORY poured onto the hypsometry the solve produces. The inventory is a seed draw. **Its range is NOT physics: it is set from the ocean fraction the owner asks for** (§9, L11), and this report states that plainly rather than dressing it as a derivation. |
| biome wavelengths `30–120 km` and `20–80 km` | `body.rs:205-213` | **MAGIC** | The climate's scales follow from the body: a Hadley band's width follows from the spin and the radius; a rain shadow's length follows from the ridge and the wind. |
| highland threshold `0.55 × relief` | `body.rs:214` | **MAGIC** | A snow line and a tree line are temperatures, and a temperature at height follows from the lapse rate. |
| strata thicknesses `1–4 m`, `2–8 m`, `20–80 m`, read at a DEPTH | `body.rs:195-201`, `strata.rs:189` | **MAGIC, and INSUFFICIENT** | See "the stratigraphic column" below. |
| the cave numbers | `body.rs:219-229` | Lawful | Untouched, except that a cave mouth should prefer a valley wall. |

**THE TWO BOUNDS ON RELIEF, and the draw that must go inside them.**

Revision 2 named the defect correctly and then did not fix it. Both refuters caught the same thing.

- Today's flow is `share → clamp → × [0.5, 1.5)`. The factor multiplies OUTSIDE the clamp, so the drawn
  relief EXCEEDS the "cap": MEASURED 14 305 m against a cap of 12 000 m.
- Revision 2 wrote `relief = draw × min(bound₁, bound₂)` and called it *"a bound is a bound"*, **and
  never restated the draw's range**. With `draw ∈ [0.5, 1.5)` unchanged the home planet reaches
  `1.5 × 13 403 = 20 105 m`, which stands above both bounds. The algebra had not moved.
- **The rule, stated completely: `relief = draw × min(strength bound, shape bound)` with
  `draw ∈ [0.5, 1.0]`.** Now a bound is a bound, and the world keeps a two-to-one variety between a
  flat body and a rugged one. **This changes EVERY body's relief and it is a one-way door** (§8, §9 L3).
- **The strength bound** is `h_max = σ ÷ (ρ_crust × g)`: how tall a column of rock stands before it
  yields under its own weight. It binds on a LARGE body, where `g` is large.
- **The shape bound** is a share of the radius. It binds on a SMALL body, where the strength bound is
  enormous — COMPUTED for a 100 km moon at `ρ_bulk = 3 000 kg/m³`: `g = 0.0839 m/s²`, so
  `h_max = 883 km`, **8.8 times the moon's own radius**. `crates/terrain/src/body.rs:232-236` would then
  derive a crust deeper than the body, and `Ladder::for_radius` returns `None` when `crust >= surface`
  (`crates/seed/src/ladder.rs:88-91`). **A strength-only slice 8a DELETES every small round body from
  the world.**
- **The shape bound's VALUE is calibrated on a small body, not on Earth. (corrected)** Refuter A is
  right that revision 2 kept `0.004` and defended it with a witness that refutes it: Vesta carries about
  20 km of relief on a 260 km radius, which is **0.077 R**, nineteen times what `0.004` allows. And
  `0.004 × 3 350 759 = 13 403 m` is Earth's own land-to-trench range, which is why nobody noticed.
  **Rule: the shape bound is `0.077 R`, cited to the observed relief of a small differentiated body,
  and the strength bound then binds on every large body by itself.** COMPUTED with `σ = 200 MPa`
  (ASSUMED, UNCITED), `ρ_crust = 2 700 kg/m³` (`substance.rs`, granite), and each row's OWN bulk
  density, named in the row, because `g = (4/3)πGρ_bulk R`:

  | body (bulk density) | `g`, m/s² | strength bound | shape bound `0.077 R` | which binds | relief at `draw ∈ [0.5, 1.0]` |
  |---|---|---|---|---|---|
  | Earth, 6 371 km (5 514) | 9.82 | 7 543 m | 490 567 m | strength | 3 772 – 7 543 m |
  | the home planet, 3 351 km (3 900, ASSUMED) | 3.65 | 20 276 m | 258 008 m | strength | 10 138 – 20 276 m |
  | a 1 000 km moon (3 300) | 0.92 | 80 293 m | 77 000 m | shape (just) | 38 500 – 77 000 m |
  | Vesta, 260 km (3 456) | 0.25 | 294 879 m | 20 020 m | shape | 10 010 – 20 020 m |
  | a 100 km moon (3 000) | 0.084 | 883 223 m | 7 700 m | shape | 3 850 – 7 700 m |

  **COMPUTED at `ρ_bulk = 3 900`: the two bounds cross at a radius of about 939 km.** Above it a body is strength-limited;
  below it a body is shape-limited. That is the right physics in both directions, and it is what makes
  the pair a derivation rather than two knobs. Earth's own 7 543 m falling out of the strength bound is
  a sanity check, not a proof: `σ = 200 MPa` is still ASSUMED and UNCITED (U-L10).
- **Three numbers are UNMEASURED and must not be presented as physics.** `ρ_bulk` must be READ from the
  body's own drawn mass and radius, not assumed: **M-L8 measures the home planet's own drawn mass and
  density** by extending `crates/bins/examples/terrain_cost.rs` to print the home body's taxonomy row.
  `σ = 200 MPa` needs a citation per bedrock. `ρ_crust` must be read from the registry row.
- **The yield-strength column moves the registry digest.** `crates/core/src/registry/digest.rs:1-9`
  folds *"every physical column"*. Appending it refuses every store written before it: free before
  slice 9, a world epoch after it.

**THE STRATIGRAPHIC COLUMN.** The reference picture's cliffs and rock pillars need a HARD band over a
SOFT one that outcrops across a slope. Today `StrataTable::at(biome, depth_m)` reads a depth below the
surface (`crates/terrain/src/strata.rs:189`, read at `crates/terrain/src/chunk.rs:637`), so every layer
follows the surface and no band can ever outcrop. **Rule: the bedrock's layering is a function of the
RADIUS, not of the depth.** The topsoil and the subsoil stay depth-following, because soil really does
follow the surface; the sediments and the bedrock become a stratigraphic column at a fixed radius, so a
valley wall cuts across them. Differential erosion then reads the hardness of the rock the solve is
CUTTING, and a hard band leaves a mesa. **This is what funds the picture's pillars, and revision 1
funded them nowhere.** It lands in slice 8c.

### 3.5 The world identity, the version and the golden tables

Ruling S5-2: **the tolerance is ZERO.** One moved rung-0 byte bumps the version and opens a world
epoch. Every landform slice moves bytes. So:

- Each landform slice bumps `GENERATOR_VERSION` (`crates/terrain/src/tag.rs:19`), re-records
  `golden_home.txt` (3 240 rows) and `golden_home_mesh.txt` (1 082 rows), and re-runs `just terrain-pin`
  (three legs) and `just terrain-legs` (two legs).
- **Every constant of the solve is part of the DECLARED tag**: the pass counts, the re-routing period,
  `MACRO_CELL_TARGET_M`, the spectrum's peak and width, the two relief bounds' constants, the draw's
  range. Change one and every stored edit on every planet sits on ground that moved.
- **The registry digest changes too, at slice 8a**, because the yield-strength column is a physical
  column (§3.4). Every store written before it refuses to open.
- **The artifact must be inside the identity — but NOT inside a LOOK.** `crates/core/src/look.rs:70-73`
  states the law: *"the DECLARED generator tag … Never the measured half — a chip difference refuses a
  lane, never a look."* The artifact's own digest rides the BULK lane it arrives on. §3.9.
- **This is the whole reason the landform slices come before the store.** After a real world is saved
  the same change costs every player's world.

### 3.6 The 8 ms per-chunk budget — and the one case that does not fit

MEASURED today, release, on this Mac (`docs/investigation/2026-09-07/slice_06_extractor.md:344-350`): a
plain rung-0 surface chunk costs 3.30 ms of worker time (1.98 ms sample box plus 1.32 ms extraction);
the worst named chunk, a rung-2 cave-dense seam chunk, costs 6.11 ms (4.38 + 1.73). The budget is 8 ms.

**Two corrections to revision 2's arithmetic, both from the refuters.**

- **The sample box holds 4 096 columns, not 4 268.** `slice_06_extractor.md:97` states the box as
  `64 × 64 × 64`; `:118` and `:233` state the column cost as *"6.5 % more columns"*; the *"ten per
  cent"* on `:355` is CELLS. Revision 2 applied the cell ratio to a column count.
- **The marginal per-octave rate is 10.5 ns, not 13.2 ns.** COMPUTED from the two published points
  (710 µs at 14 octaves, 266 µs at 3): the margin is `444 ÷ 11 = 40.4 µs` per octave over 3 844 bare
  columns, that is 10.5 ns per octave per column, and the column pass carries a 145 µs fixed cost.
  Revision 2 divided the total and assumed no fixed cost. **The number no longer matters for the
  budget, because §3.7 deletes the octave saving entirely.**

**What landforms add per rung-0 chunk**, taken from the sibling domains' own tables so the three reports
do not each carry a different number (DOMAIN 03 §7.4, DOMAIN 05 §5.7):

| Added work | Where it runs | ESTIMATED |
|---|---|---|
| `Z`, the macro height: one 4 × 4 node gather, then one Catmull-Rom per column | column pass | < 0.05 ms |
| the two-cell column ring for the slope | column pass | 0.10 ms |
| the channel and floodplain carve, up to six macro segments, prefiltered | column pass | 0.25 – 0.75 ms |
| the water level per column | column pass | < 0.05 ms |
| scree, ridged crests, terraces and gullies | column pass | 0.15 – 0.25 ms |
| the climate: one coarse-grid lookup and blend per column, plus the classifier | column pass | 0.15 ms |
| the WATER SURFACE mesh, on the chunks the water range crosses only | a second extractor run | 0.30 – 1.00 ms |
| **total, a chunk with a river and no water surface** (the six column rows) | | **+0.75 to +1.35 ms** |
| **total, a chunk with a river AND a water surface** (all seven rows) | | **+1.05 to +2.35 ms** |

**The rows sum to the totals**, which is not a small point: refuter A caught revision 2 printing a total
its own rows did not reach.

**The budget, judged honestly:**

```
   plain surface chunk   3.30 + 1.35            =  4.65 ms   of 8 ms      PASSES
   worst named chunk, dry
     (cave-dense + river) 6.11 + 1.35           =  7.46 ms   of 8 ms      PASSES, margin 0.54 ms
   worst named chunk, wet
     (cave-dense + river + a meshed water sheet)
                          6.11 + 2.35           =  8.46 ms   of 8 ms      OVER BUDGET by 0.46 ms
```

**So the budget is met everywhere except one coincidence: a cave-dense chunk that also holds a meshed
river surface.** Revision 2 reported a comfortable 1.29 ms of margin because it costed three terms and
omitted five. This report states the over-budget case instead of hiding it, names it as the case M-L3
must bench first, and names the levers: DOMAIN 03's `SUBDIV` falls from 4 to 3, or the water sheet is
meshed at a coarser rung than the rock. **The gate is unchanged: the bench asserts the costliest named
chunk against 8 ms. It is now expected to go RED on that one case until a lever is pulled, which is what
a gate is for.**

**Why a C1 read and not the cheap one.** A two-point read is smooth in value and NOT in slope: its slope
jumps at every node boundary, so the shape carries a visible crease every 8 224 m. The reference picture
is a 50 km vista — about six macro nodes across the frame — so six creases would be in shot. Catmull-Rom
costs a 4 × 4 gather and removes the crease. The seam risk is listed in §3.11.

**The rule that protects the whole budget.** The closed-form valley must not need a second noise stack
at every octave. **A valley is a shaping FUNCTION applied to the height that already exists, never a new
stack of octaves.**

### 3.7 The ladder — rung L must be cheaper than rung 0, and its answer must be bounded

MEASURED today: the column pass falls from 710 µs at rung 0 (14 octaves) to 266 µs at rung 11 (three
octaves), and a shipped Tier-A test proves that every rung sums strictly fewer octaves than the rung
below it (`crates/terrain/src/body.rs:355-365`; the doc comment at `:340-342` calls it the M-16 gate).

**REVISION 2'S OCTAVE RULE IS WITHDRAWN. It would have turned that gate red.** COMPUTED, and this is
refuter A's blocker: `octaves_at` floors the kept count at one (`body.rs:251-261`); the home planet has
12 rungs; with only 9 octaves the count saturates at rung 9, so rungs 9, 10 and 11 all keep ONE octave,
and the assertion `here < below` fails at rung 10. To keep the gate under revision 2's rule the macro
cell would have to be at least `30 × 2^11 = 61 440 m`, which resolves no drainage at all.

**The rule that replaces it, from DOMAIN 03 §4.2 and §4.12.**

- **The octave table is untouched: fourteen octaves on the home planet, one per halving from the
  coarsest wave down to 30 m.** M-16 stays green, and no code that reads `octaves_at` changes.
- **The macro field `Z` REPLACES the coarse octaves' contribution, inside the same envelope.** The
  height becomes `radius + Z(dir) + Σ(the FINE octaves at or coarser than the rung) + the carve + the
  detail`. `|Z| ≤ A_coarse` after a stated renormalisation, and `|Σ fine| ≤ A_fine`, and
  `A_coarse + A_fine = relief_bound_m(0)` exactly, because `body.rs:180-185` normalises them. **So the
  bound stays EXACT and the band does not move.**
- **The slope spectrum re-weights the fourteen amplitudes and keeps their sum.** COMPUTED by DOMAIN 03
  on the home planet: the 6.25 km octave rises from 94 m to 606 m, the 3.13 km octave from 47 m to
  377 m, the 1.56 km octave from 23 m to 151 m, and the 400 km wave still carries 3 713 m. **This, not
  the macro lattice, is what answers *"the surface will not be interesting enough"*.**
- **A coarse rung reads a COARSE LEVEL of the artifact's pyramid, not the full field.** The pyramid is
  1.6 MB beside a 9.83 MB artifact, and its top level is 19 KB. At rung 11 a chunk spans
  `62 × 2 048 = 127 km`; reading level 0 would touch ESTIMATED `(127 000 ÷ 8 224 + 1)² ≈ 260` nodes,
  and reading a matched pyramid level touches a handful. **So the macro read gets CHEAPER with the
  rung, exactly as the octave drop does**, and revision 2's claim that the read costs the same at every
  rung is corrected in the right direction.

**The valley detail is still a danger.** A valley 300 m wide is four cells at the 64 m rung and half a
cell at the 512 m rung. If it simply disappears at that rung, the player sees a valley switch off. That
is a detail-by-box seam, and SL8 refuses it. **Rule: a valley's depth is a continuous function of the
rung's cell size, so it fades and never vanishes.** The measurement is slice 8's own: the maximum and
99th-percentile vertical disagreement between rung L and rung L+1, in pixels at the switch distance,
passing at p99 ≤ 1 pixel and max ≤ 2 pixels. The landform slices re-run it.

### 3.8 The record, the pyramid, the store, the BODY DEFINITION and the collider

| Item | Verdict |
|---|---|
| **The 12-byte CELL record** (ruling V6 B-1..B-13) | **NO CHANGE.** Everything landforms add is DERIVED. Only edits are stored, and an edit's record is what it is today. A biome is a function of the address, so it never needs a byte. **This is a different object from the macro NODE, which is 4 bytes of height plus two bytes of state; the two share no format and no lane, and revision 2 gave both a 12-byte width one section apart.** |
| **The density byte** | Unchanged. A valley reaches the surface through the height field, and the density stays the signed radial gap at the cell centre. |
| **The biome/object param** | Unchanged in width. Its VALUE space grows (four biomes become sixteen, five bits). That is a registry append, and §3.4 notes that a registry append moves the registry digest. |
| **The 8-byte pyramid entry** (ruling S5-3: deltas from the recipe's coarse answer) | **NO CHANGE.** The coarse answer changes; a delta from it stays a delta. |
| **THE BODY DEFINITION AND THE BAND** | **The band does NOT grow, and revision 2 said it would.** `Z` REPLACES the coarse octaves inside the same envelope, so `relief_bound_m` keeps its present meaning and stays exact. The mechanism is DOMAIN 03's ENVELOPE RENORMALISATION: after the solve, take `zmax = max|Z|` as an integer maximum (exact, order-free); if `zmax > A_coarse`, multiply every `Z` by `A_coarse ÷ zmax` with one stated rounding. **Growing `crust_m` instead is REFUSED**: it lowers `floor_m` and shifts EVERY cell's radial index on EVERY body (`crates/seed/src/ladder.rs:95,132-134`). The channel's cut fits inside the 64 m of spare crust `body.rs:234` already carries. |
| **The exactness, stated precisely** | Refuter A is right that today's assertions live in `#[cfg(test)]` and sample 400 directions of one body. The bound is exact by CONSTRUCTION, not by assertion. After the renormalisation it stays a construction, and the gate asserts `|Z| ≤ A_coarse` at EVERY node — an exhaustive integer test over the artifact, not a sample — plus the golden chunks inside the band. |
| **`BodyDefinition` stops being `Copy`** | `crates/terrain/src/body.rs:88` derives `Copy` today, so a resident artifact cannot live in it. The client already holds `Arc<BodyDefinition>` (`crates/client/src/chunks.rs:71`). The crate's by-value uses in `chunk.rs`, `height.rs` and `digest.rs` change with it. Small, named work in slice 8b. |
| **The chunk margin** | `crates/terrain/src/digest.rs:88-95` gives the surface chunk span *"one chunk of margin each way"* and warns that a slope steeper than the margin breaks it. **Slice 8c lands cliffs.** The margin must be re-derived from the eroded field, and slice 8's residency band re-measured. |
| **The store** (slice 9) | The artifact is DERIVED, not world data. **The owning shard CACHES it, keyed by the world tag and the body seed**, so a realm the demand loop spins down and up does not solve again. That is now a requirement, not an option, because the solve is 12 to 40 s. |
| **The collider** (slice 11) | The same shape, by construction. The shard holds its planet's artifact, 9.83 MB. The analytic swept query of slice 11 stays O(1) per cell, because a resident artifact is a lookup. |

### 3.9 The boot cost — the Step 23 defect, in its true shape

**The defect's shape.** The DECLARED half of the world tag is folded by every process and costs
nanoseconds (`crates/bins/src/lib.rs:970-979`). The MEASURED half evaluates eight chunks of the home
planet and is folded by the GATEWAY (`crates/bins/src/bin/gateway.rs:193`) and the CLIENT
(`crates/bins/src/bin/client.rs:118`) only; MEASURED 13.5 ms. The shard and the orchestrator pay
nothing. Revision 1 named four payers and cited the wrong function.

The sharp half survives. **The gateway owns no planet, draws nothing and decodes no chunk, and it
self-checks the home planet at boot.** Once a chunk's height reads a macro artifact, that self-check
would need the artifact — ESTIMATED 12 to 40 s of solve, in the gateway and in every client, before
either connects to anything. That is the boot-plant defect of Step 23 repeating
(`project_boot_plant_fix`: every process built the 3.5-million-region forest at boot, and the gateway
grew to 5.3 GB).

**The cure, in four rules. Rules 1 and 2 replace what revision 2 wrote.**

1. **THE EIGHT GOLDEN CHUNKS ARE EVALUATED ON THE SHIPPED PATH, WITH THE HOME ARTIFACT'S OWN VALUES
   PINNED AS LITERALS.** Refuter A is right that revision 2's *"a single macro cell, read from a stated
   pinned value"* is impossible under a 4 × 4 Catmull-Rom stencil, and right that a value the world
   never uses would invert the "test exactly production" law at the one gate meant to enforce it. The
   cure is neither: **pin the REAL artifact's real nodes** — at most `8 × 16 = 128` nodes, ESTIMATED
   under 400 bytes of literals — beside `HOME_PLANET_RADIUS_BITS`, re-recorded with the golden tables
   whenever the solve changes. Under the SHIP road (§4.1) `Z` is an INPUT to the client's chunk
   arithmetic exactly as the look radius is, so pinning it pins an input, not a stand-in.
2. **THE WHOLE ARTIFACT IS CHECKED BY ITS OWN DIGEST WHEN IT ARRIVES, ON THE BULK LANE.** Refuter B is
   right that a kernel canary cannot see accumulation drift: a five-node neighbourhood run once
   exercises the arithmetic and never the accumulation. The artifact carries its digest with it, the
   receiver refuses a mismatch, and the SOLVE's own cross-host equality is proven where it belongs — by
   `just terrain-pin` and `just terrain-legs` at build time, on a real x86-64 machine (§3.1, L22).
3. **THE KERNEL CANARY IS KEPT, AND IT IS NOT CLAIMED TO BE MORE THAN IT IS.** The flood, the router and
   the sweep run over a stated small neighbourhood at stated addresses and their digest is folded into
   the world identity. ESTIMATED cost: microseconds. It catches a chip that computes the kernels
   differently at LOGIN, which is early and cheap. It does not catch an accumulation drift, and this
   report no longer says it is *"STRONGER"* without saying what it gives up.
4. **NOBODY SOLVES AT BOOT, AND NOBODY SOLVES IN THE GATEWAY, EVER.** A structural control makes the
   defect fail: a test that the gateway's resident artifact set is empty after a flight, and a test that
   no solve runs before a realm is asked for a chunk.

*Example.* Two hundred shards boot in the cluster. One of them owns the home planet, and it loads its
cached artifact from its own store. The gateway pays the kernel canary in microseconds, holds no
artifact, and re-emits the bulk bytes it never decodes, as it does today. A client that logs in standing
on the home planet receives the artifact's coarse pyramid first and the full field for the body it is
at, and it never solves anything.

### 3.10 SL9 — the cost grows with the observer, never with the number of children

- The macro artifact is ONE array per body, sized by a stated metric target and the body's own `N`,
  never by the number of children. No per-child walk exists anywhere in this direction.
- The river graph is a bounded list per macro node, so the per-column query is O(1)-ish, never a scan.
  **The bound must be MEASURED, not stated:** revision 1 wrote "at most eight segments" with no source.
  Slice 8c measures the true distribution and states a cap WITH a refusal test (§9, L15).
- The chunk work stays per observer, as today.
- **The honest tension, restated correctly.** The artifact's cost is a function of the BODY, not of what
  the observer looks at. A player who lands on one hillside pays for the whole body's solve, because a
  river's shape depends on ground the player never sees. Under the SHIP road the player pays it as
  BYTES, not as seconds, and the shard pays it as seconds ONCE EVER.
- **"A small body costs strictly less" is not strictly true, and the claim is corrected.** Under
  §4.3's metric rule the artifact's node count rises with the SURFACE AREA of the body, so a 50 km moon
  holds 384 nodes and the home planet holds 2 457 600. That IS ordered. But the cost per body rises as
  `R²` with no ceiling: COMPUTED, an Earth-sized planet holds about 8.9 million nodes, four times the
  home planet's, so about 40 MB kept, 500 MB transient and 50 to 160 s of solve. **`MACRO_CELL_TARGET_M`
  therefore needs a stated MEMORY CEILING for large bodies, and a body above it gets a coarser node**
  (§9, L4). Revision 2's flat `2^E` bound hid this by making every large body identical and every small
  body wrong.
- **A client holds MORE THAN ONE artifact, and a shard may solve MORE THAN ONE body.** The home system
  holds 10 planets and 11 moons (MEASURED,
  `docs/investigation/2026-09-07/slice_02_geometry_seam.md:275`), and that source does NOT record their
  radii. Revision 2 wrote *"eleven moons at a few hundred kilobytes each"* with no source; our own Moon
  is 1 737 km and Ganymede is 2 634 km, so several may be full-sized. **Under the SHIP road the client's
  worst case is bytes and residency. COMPUTED under the metric rule, a Luna-sized moon of 1 737 km gets
  `N = 2 727 936` and `n_macro = 333` (which divides it exactly), that is 665 334 nodes and **2.66 MB** —
  not a full-sized artifact, which is the improvement the metric rule brings over the angular one. A
  plausible band of one planet and three large moons is then ESTIMATED `9.83 + 3 × 2.66 ≈ 18 MB` plus
  the pyramids, UNMEASURED (U-L13). **Under the derive road the same band would be four solves of 3 to
  40 s each, SINGLE-THREADED, queued one behind another.** The residency rule must be written for many,
  and it must state which artifact is freed first.

### 3.11 SL8 — seamless

Seven seam risks, each with its rule and its detector.

| Risk | The rule | The detector |
|---|---|---|
| A valley or a river appears when a rung changes | the depth fades with the cell size (§3.7) | slice 8's pop detector, re-run |
| A crease every 8 224 m from a slope-discontinuous read of `Z` | the read is Catmull-Rom, C1 in value and slope (§3.6) | a 50 km vista picture at a slant, and a second-derivative scan across a node boundary |
| **A SCARP ON A CUBE EDGE** | `Z` is single-valued on every cube edge: the edge node's value is owned by the face with the smaller index, and both faces read the owner's value | **G-MACRO-EDGE**: walk all twelve cube edges and assert `Z` agrees exactly and the slope agrees inside a stated bound. COMPUTED: a cube edge on the home planet is `2π × 3 350 759 ÷ 4 = 5 264 km`, and there are twelve — **63 000 km of possible cliff** |
| The artifact arrives after a chunk is drawn, so the ground moves under the boots | **no chunk is built before the artifact it reads is complete for that node's neighbourhood** | a gate: a flight that draws before the artifact is ready must fail |
| A coastline pops as the water level resolves | the sea and the lakes are part of the artifact, so they are ready with it | the same gate |
| A body's realm wakes and stalls | **the artifact is CACHED in the shard's store; a wake loads it and never solves it** | a measurement: wake to first chunk, against the reach's own lead |
| Weather pops when a dormant realm wakes | the almanac is a closed form on the tick, so it is continuous by construction; the live offset DECAYS with a stated time constant and is advanced by the closed-form decay over the sleep in one step | a gate: sleep a planet for an hour of world time, wake it, and no cloud jumps |

### 3.12 SL1, SL2, SL3, SL7 — who says what to whom

- **SL1 clauses 1 to 3.** The recipe names no position of any realm. It works in the body's OWN frame, at
  an address. The charter it receives holds facts about WHAT THE BODY IS, never about where it is.
  **The recipe must never read an instantaneous distance to the star**, because that is a placement and
  it changes with time. The season is the almanac's (§3.14); the shape is not.
- **SL1 clause 4, THE ONE-HOP QUESTION.** The charter changes this question's shape for the better, and
  revision 2 asked it in the harder form.
  - Gravity, the spin period, the obliquity and the water inventory follow from the BODY's own seed,
    mass and radius. They are the body's own facts. No hop. Clean.
  - Mean insolation, equilibrium temperature and eccentricity follow from the STAR's luminosity and the
    ORBIT, which the SYSTEM authors. **Under the charter the SYSTEM — the parent — computes and states
    them.** So the datum crosses ONE hop, from the realm that authored it to the realm it is about, and
    the body then carries its charter in its own look the way it carries its radius. Nothing is "passed
    on" about the body's own placement, because none of these facts IS a placement.
  - `insolation_rel = L/d²` (`crates/physics/src/taxonomy.rs:931-932`, with `d` documented as the body's
    own orbit at `:905-908`) inverts to the orbit's SEMI-MAJOR AXIS, a constant of the orbit. It does not
    change with the tick and it does not say where the planet is. Telling a client the orbit's SIZE is
    the same class of statement as telling it the body's radius, which slice 4 already ships.
  - **A mitigation if the owner wants belt and braces:** ship the MEAN SURFACE TEMPERATURE instead of the
    insolation and the equilibrium temperature. It is what the climate actually needs, and it does not
    invert to a distance without the albedo and the greenhouse term as well.
  - **This is an owner decision (§9, L16), not a conclusion of this report.**
- **SL2.** No occupant pose crosses. Weather is a property of the REALM, stated by the realm, and it
  names no occupant.
- **SL3.** A realm draws itself. Weather is part of how a planet LOOKS, so it rides the realm's own look
  — but NOT the surface tag, which is a once-on-change retained statement
  (`crates/core/src/look.rs:53-54`). It needs its own lane with its own rate (§7).
- **SL7.** A planet's realm serves its own observers. The weather statement goes to the observers inside
  the realm's reach, on the realm's own lane, ONE statement for the realm.

### 3.13 V4 — vegetation are art assets, placed by a server skeleton

The climate field is what makes this work. The chain is: the climate gives a biome; the biome gives a
vegetation density and a species mix; the server's skeleton places objects at that density; the client
blends the art kit. **The landform work owes the biome and the density. It owes no mesh and it draws no
primitive.** Slice 14 consumes it. The canopy fold of the far view (rulings C-4 and R-15) reads the same
density, so a forest to the horizon is the biome painted at a coarse rung.

**And one row of the reference picture is not vegetation at all.** The picture's FIELDS are cultivated:
bounded, rectangular, hedged. A biome gives grassland, not a field. **Fields are the same class as
roads: built, not seeded, and out of this arc.** Revision 2 mapped them to the biome (§6.9, corrected).

### 3.14 The weather split — what is shape, what is almanac, what is live, and what is force

Weather is the one part of this direction that moves with time. It has THREE layers, not two, and the
line between them is the law. Revision 2 had two, and refuter B showed that the missing one is exactly
the layer the owner's word *"trajectory"* lives in.

```
   CLIMATE (static, in the recipe)     the long-term average: temperature, rainfall, wind bearing.
        |                              f(seed, charter, address). No clock. It decides the biome, the
        |                              snow line, the trees. Both hosts compute it.
        v
   ALMANAC (closed form on the tick,   the season's phase, the sub-solar point, the day's phase, the
   computed by the OWNING realm)       nominal temperature now, the nominal cloud cover.
        |                              f(seed, charter, universe tick). A SMALL ROW, shipped.
        |                              This is where spin, obliquity and eccentricity become VISIBLE.
        v
   LIVE WEATHER (the realm's own       storms, fronts, gusts, rain now, lying snow, river stage.
   state, on its shard)                A one-hop diff to that realm's own occupants.
        |
        +--- as LOOK ----->  the client paints it. That is rendering, not deriving.
        |
        +--- as FORCE ---->  the realm's own physics adds wind to its medium. The parent already
                             computes drag from the child's stated mass and cross-section. Nothing new
                             crosses upward, and the movement contract is untouched.
```

**THE SEASON IS THE ANSWER TO "TRAJECTORY", AND REVISION 2 HAD NO LAYER FOR IT.** COMPUTED: in an annual
mean, eccentricity's whole contribution is the factor `(1 − e²)^(−1/2)` — 1.5 % at `e = 0.17`, 0.014 %
at Earth's `e = 0.017` — and it is already inside `insolation_rel`. So a static climate cannot show a
tilt or an orbit at all. What a player SEES is the season: a snow line that moves, a summer, a winter, a
perihelion heat wave. The almanac is a closed form on the universe tick, so it is dormant-safe by
construction, it costs the shard almost nothing, and it breaks no law: V12 already put the LIGHT on the
ship-it side.

**THE LIVE FIELD'S BYTES, PRICED.** Revision 2 said the weather is *"a FIELD stated ONCE for the realm,
as coefficients over the body's own surface … at a very coarse `M`"* and never priced it. COMPUTED: a
storm front a pilot sees on the horizon is a ~50 km feature; on the home planet that needs about
`N ÷ 50 000 ≈ 105` cells per face edge, so `6 × 128² = 98 304` cells, and at 4 bytes each that is
**393 KB per statement** against `SELF_LOOK_BUDGET_BYTES = 1 200` for the whole self-look bag. At a
genuinely coarse grid — 8 cells per face edge, 384 cells — one cell spans 658 km and every storm is
658 km wide. **So a stored coefficient grid either misses the lane by 300 times or cannot draw the
promised picture.** The lawful third road is a SMALL SET OF SEEDED PARAMETERS the client evaluates as a
procedural field: a handful of moving cyclone centres with radii, strengths and a phase, from which the
client derives cloud, rain and wind per place. That is rendering from stated numbers, and it is what
L-4 must ask for — **after U-L8 measures it**, not before.

**The dormant-wake rule.** A live part that is not simulated while dormant MUST jump on wake. The cure:
**the live offset decays toward zero with a stated time constant, and on wake it is advanced by the
closed-form decay over the whole sleep, in one step.** The almanac needs no such rule, because it is a
closed form on the tick.

**And the plain words the owner is owed.** The owner said *"we also should simulate the weather."* **A
closed form plus a decaying offset is NOT a simulation:** there is no pressure field, no front and no
advection. It is a believable, cheap, dormant-safe animation of a climate. A real simulation — a coarse
global circulation on the climate grid, stepped by the planet's shard — is possible, it is live state,
it breaks no law, and it costs a per-tick pass over that grid. **This report does not choose. §9 L17
puts it to the owner in those words, and it must be answered AFTER U-L8, because a field that cannot
cross changes the question.**

**The rule that keeps it lawful either way:** anything that decides a COLLISION or a HIT is the server's
and is never derived on the client; anything that only PAINTS is the realm's look, and the client may
animate it from stated numbers. Rain that darkens the sky is look. Wind that pushes a hull is a force
the shard applies. The statement carries a phase, so two clients side by side paint the same cloud in
the same place.

**SL10 is not touched**, because none of this enters the generator crate.

### 3.15 The dormant world, and whether the land changes with time

`scripts/dormant_world_simulation_design.md` asks that the world advance believably while 99 % of it is
off. Landforms answer by NOT advancing: **the landscape is static.** Erosion is a generation-time
process that produces the shape. It is not a simulation that runs during play. SL10 clause 1 forces
this, and it is also what the player wants: a river that moved while you were away is a seam.

**What DOES advance while a realm sleeps is the almanac**, because it is a closed form on the universe
tick. A planet that slept through half its year wakes with the right season, with no state carried and
nothing to catch up.

**What the owner may want later, and what it would cost.** Long-term change — a delta growing, a cliff
retreating — would be LIVE state, a diff from the owning realm, and a pyramid entry like any edit. It
breaks no law and it is not in this arc. Recorded as an open decision (§9, L10).

### 3.16 HR3, HR4, HR5, HR6

- **HR3, one tooling.** The recipe is one crate, and the landform layers are functions inside it. No
  shard-kind match appears anywhere. **The water gate of §0 item 8 is per-body DATA, never a kind
  branch:** the recipe asks the charter whether the body holds an atmosphere and liquid water, and a
  body that does not simply has no hydraulic layer. A hull and a station have no round body definition
  at all. A body whose `N` is tiny gets `n_macro = 8` by the clamp — the same rule with no special case.
- **HR4, features once, run anywhere. (corrected)** Refuter A is right that revision 2 borrowed the
  seam's gates and gave the landform work none of its own. `G-MAPPING-TABLE` and `G-MAPPING-ROUNDTRIP`
  are gates on the `GridMapping` seam ITSELF: an exhaustive address round-trip proves the mapping is a
  bijection, and it says nothing about whether a drainage network is correct. **Ruling V6 A1's reason —
  a bent grid and a flat one cannot give equal results — bites HARDER on a neighbour algorithm than on a
  height sample.** So the landform layer carries its OWN gates of that kind, and they are §3.1 rule 7's:
  **G-MACRO-CORNER** (every one of the eight cube corners has exactly seven neighbours and a receiver
  among them), **G-MACRO-EDGE** (every one of the twelve cube edges is single-valued and slope-continuous
  in `Z`), **G-MACRO-AREA** (the accumulation's area weight equals the bend's own area density, tested
  exhaustively over a face like `G-MAPPING-TABLE`), and **G-MACRO-SEAM** (solve a small real body, solve
  it again with the faces visited in reverse order, compare every byte).
- **HR5, 100 % region and branch in Tier-A. (corrected)** `vd-terrain` and `vd-seed` are Tier-A. Refuter
  A and refuter B both found the same hole: revision 2 covered the KERNELS and left the DRIVER, which
  also lives in the Tier-A crate. §3.3 states the two-part plan: the kernels by neighbourhood tests, and
  **the driver by solving a REAL small body of THE world inside the crate's own unit tests** — 384 nodes
  at `n_macro = 8`, milliseconds, no variant and no test world. The whole-body run on the home planet
  stays a gate in `bins`. The branches an iterative loop brings — the first pass, an empty basin, a node
  with no downhill neighbour, a lake that overflows, a face seam, a seven-neighbour corner, a body with
  no water — are all reachable from those two. The generic-code discipline is unchanged: a generic
  function stays a branchless shim.
- **HR6, agent-operable end to end.** The pictures are taken by `vdctl` through the harness, as slice
  7's three were. Each landform slice owes named pictures (§6).

---

## 4. The three integration decisions everything hangs on

### 4.1 Derive the macro artifact, or ship it? — THE ANSWER REVERSED

Revision 2 recommended "derive on both hosts" on a cost model of 0.31 to 0.41 s. Both refuters
dismantled that model, and DOMAIN 03 re-priced the solve independently and reached the same range.

**What the model got wrong.** It costed a priority flood — a heap over every node of the body — as
*"~1 random access per cell"*, in the same section that scolded revision 1 for costing an erosion cell
at a noise sample's rate. And it ran the flow routing TWICE IN TOTAL, which leaves the river graph
describing a hill that forty erosion passes have since removed.

| | **A. Both hosts derive it** | **B. The owning realm SHIPS it, coarse first (RECOMMENDED)** |
|---|---|---|
| Law | SL10 grants it: it is `f(seed, charter)`. | Lawful too. The star field is the precedent (shipped once, reach R1); ruling V1 makes anything the seed does not decide a one-hop diff, and this is a stronger case, not a weaker one. |
| Cost to the client | **ESTIMATED 12 to 20 s of ONE core, plausibly up to 40 s, per body**, with 125 MB transient | ESTIMATED 9.83 MB of bulk for the body the player is at, and 19 KB for the approach; 9.83 MB held |
| Cost to the server | the same solve, once per body | the same solve, ONCE EVER per body, cached in the shard's store; plus the fan-out, where the gateway re-emits one shared buffer per message (the slice 10 rule) |
| The player standing on the ground at login | **there is no proxy for the ground under your boots.** The client cannot mesh the chunk it is standing in until the artifact exists | the artifact arrives on the bulk lane while the approach is drawn from the pyramid |
| The client's own facts | the client cannot compute the charter at all (§4.2), so it could not solve even if it had the time | it needs no facts: it reads the artifact |
| Failure mode | a drift stands the boots off the ground; the no-drift legs are the defence, and the fifth leg is an emulator (§3.1) | a stale or truncated blob stands the boots off the ground; the defence is the artifact's own digest and a refusal |
| New wire | none | a new bulk payload kind: an SL6 row, a manifest, a resume path |
| Verdict | **REFUSED on cost, not on law.** | **RECOMMENDED.** |

**The measurement that could still change it** (M-L1, §11): the solve's wall time and peak memory on ONE
worker, for the home planet, a 100 km moon and an Earth-sized body, at two or three values of
`MACRO_CELL_TARGET_M`. If the solve turns out to be one second rather than twenty, road A returns as a
lawful option. **It must run before L1 is answered.** Revision 2 wrote its verdict — *"Under a second on
one core"* — before any measurement existed, and that is the habit this report is correcting.

**What the reversal DELETES from the plan.** Revision 2 owed a new whole-body JOB KIND on the client's
per-chunk worker seam (`crates/client/src/chunks.rs:84-104`, whose Tier-A implementation runs inline).
Under road B the client runs no whole-body job, so that work disappears. What arrives instead is a bulk
consumer, which the client already has for chunks.

**The distance the deadline is measured against is still UNMEASURED (U-L4).** The deadline is not "the
planet is 100 pixels wide"; it is "the client is about to build a chunk", which slice 8's residency band
sets and which nothing today states. For scale only: COMPUTED at the drawable floor (1 m subtends
1 pixel at about 870 m), the home planet is one pixel at 5.83 million km and 100 pixels at 58 300 km, so
at 30 km/s the 100-pixel mark is 32 minutes away and the one-pixel mark 54 hours. A 9.83 MB transfer
fits either with room. The risk is warp, and slice 8's residency band already reads a server-stated lead
rather than a derived speed.

### 4.2 How the physical facts reach the recipe — THE BODY CHARTER

The recipe cannot compute gravity or insolation: they need the mass, the star and the orbit, and the
crate may not name the motion crate. Revision 2 proposed that the facts ride the surface tag as FLOATS
which the RECIPE snaps. **Refuter B showed why that is unsound, and refuter A reached the same place from
the law's side.**

- `crates/physics` has **no float fence at all** — there is no `crates/physics/clippy.toml` — and
  `crates/physics/src/worldgen/generate.rs` uses `ln`, `cos`, `sin` and `powf`, whose results differ
  between targets.
- The taxonomy's own doc comment says so plainly: *"the cross-host bit-equality gate is SPIKE-6a
  (step-2/P4)"* (`crates/physics/src/taxonomy.rs:21-24`). **So road B as revision 2 wrote it imports a
  DEFERRED gate into the crate whose whole purpose is proven determinism.**
- The recipe's own invariant already forbids it: *"a body is DRAWN from a seed … and never assembled
  from numbers computed elsewhere, so no float from an unfenced crate can enter the recipe as a body"*
  (`crates/terrain/src/body.rs:85-87`).
- And the 1 000-ulp test at `crates/terrain/src/home.rs:49-77` measures the SNAP'S TOLERANCE, not the
  PERTURBATION a snap must absorb. Nobody has run the forest on two architectures and compared. Calling
  it a measurement of the right thing was revision 2's own "never assume, measure" rule broken.

**THE CURE, WHICH THE TREE ALREADY CHOSE ONCE: THE PARENT QUANTISES, AND INTEGERS CROSS.**

```
   the SYSTEM (the parent)                        the BODY's realm            every observer
   ------------------------                       -----------------           --------------
   draws the taxonomy row with full libm
   evaluates g, insolation, T_eq, the
     atmosphere, the spin, the obliquity's
     cosine, e, the star class
   QUANTISES each to a stated integer grid  --->  holds its charter    --->   read the same integers
   states the integers, once, on change           states it in its look       off the realm's look
                                                                              and derive the same
   NO FLOAT EVER CROSSES.                                                     climate, byte for byte
```

- **The quantisation happens ONCE, in the authoring realm.** Both hosts then read the SAME INTEGER.
  There is no straddle risk on a receiver, because no receiver rounds anything.
- **A straddle can still happen in the AUTHOR**, if the same system is drawn on two different machines
  and one float lands either side of a grid line. That risk is real and it is UNMEASURED. **M-L10
  measures it**: draw the home system on both target legs and compare every charter integer. The snap
  grid must be far larger than the measured spread, and that is what makes a snap safe — not how far a
  step moves the world.
- **Revision 2 justified every snap grid by how far one step moves the WORLD, which is the wrong ruler
  for a byte-identity law.** Format D's tolerance is ZERO. Two of those justifications were also wrong
  arithmetic: the gravity row said *"the relief bound moves by 0.02 % per step"* when one 1/64 step is
  0.30 % of the home planet's `g` (and the strength bound is inverse in `g`, so 0.30 %); the obliquity
  row said *"about a kilometre"* when `d(ε) = (1/1024) ÷ sin 23.4° = 0.00246 rad`, which on a 3 350 759 m
  radius is **8.2 km**. Both CHOICES survive; neither NUMBER did.
- **The home body's charter is PINNED in `crates/terrain/src/home.rs`, beside the radius bits.** The
  client's BINARY folds the world identity before it connects to anything
  (`crates/bins/src/bin/client.rs:117-118` → `world_identity` → `home_body` → the forest), so for the
  home body there is no author to state the charter yet. `home.rs` states two literals today; it will
  state about thirteen. `crates/bins/tests/home_body_pin.rs` cross-pins them against the forest's own
  draw, as it does for the radius.

**The facts, the proposed quantum, and the fence.** The owner sets the grids at the slice's discussion.

| Fact | Source today | Proposed quantum | Crosses as | Note |
|---|---|---|---|---|
| Surface gravity | `taxonomy.rs:750`, from mass and radius | 1/64 m/s² | an integer | One step is 0.30 % of `g` on the home planet, and the strength bound moves by the same 0.30 % — where that bound binds at all |
| Bulk density | the taxonomy's mass and radius | 1 kg/m³ | an integer | It sets `g` and it must be READ, never assumed (§3.4) |
| Escape velocity | `taxonomy.rs:755-758` | 1 m/s | an integer | The physical driver of a small body's lumpiness, and of how far ejecta throws a crater's rim |
| Mean insolation | `BodyTaxon::insolation_rel`, `taxonomy.rs:932` | 1/256 of Earth's | an integer | About 0.3 K per step |
| Equilibrium temperature | `BodyTaxon::t_eq_k` | 1/16 K | an integer | It moves the snow line by a few metres |
| Atmosphere scale height and mean molecular weight | `taxonomy.rs:881-891` | 1 m, 1/256 | integers | They set the lapse rate and the haze |
| **Whether the body holds liquid water** | derivable from the atmosphere and `t_eq_k` | a flag | a flag | The water gate of §0 item 8 |
| Spin period | **does not exist** | 1 s | an integer | It sets the Coriolis bend, the day, and the almanac |
| **Obliquity, as its COSINE** | **does not exist** | 1/1024 | **a cosine, NEVER an angle** | `sin` and `cos` are banned (§3.1 rule 4). COMPUTED: one step is 0.00246 rad, that is 8.2 km of ice-cap edge on the home planet |
| **Orbital eccentricity** (the owner's *"trajectory"*) | `crates/physics/src/worldgen/generate.rs` draws it | 1/1024 | an integer | It is visible through the ALMANAC, not through the annual mean (§3.14) |
| **The star's spectral class** | the taxonomy's star row | the class itself | a small integer | It sets the light's colour, and so what a plant looks like |
| Water inventory | **does not exist** | 1/1024 of the crust volume | an integer | It sets the sea level, and so the ocean fraction. Its RANGE is set from the owner's target ocean fraction, not from physics (§3.4) |

**`POLE_AXIS` DOES NOT MOVE.** Obliquity is the angle between the spin axis and the ORBIT's normal — a
relation between two frames, and **the parent already owns that relation** (SL1 clause 1). Two things are
wanted, and they separate: WHERE the ice caps are (at the pole, in the body's own frame, which
`POLE_AXIS = +Z` already answers — the parent tilts the whole body and the caps tilt with it, free, with
no shape change and no epoch), and HOW STRONG the pole-to-equator gradient is (a SCALAR: the obliquity's
cosine, one charter integer). **So the recipe needs the scalar and never needs `POLE_AXIS` to move.**

**Who draws the three missing facts is an owner decision (§9, L7).** The recommendation: the FOREST draws
the spin, the obliquity and the water inventory, because the forest authors the body's rotation on the
placement row and one fact must have one drawer. **The cost of that choice, named:** those three draw
laws do not exist yet, a first draw law is always tuned, and every tuning re-records the golden tables
and bumps the version. **The three draw laws are therefore a ONE-WAY DOOR of their own** (§8), and they
should be written and tuned BEFORE slice 9, not after.

### 4.3 Where the macro lattice's resolution comes from

**Revision 2's rule is WITHDRAWN, for a reason neither refuter had to argue: it makes the resolution run
backwards.** The rule was `M = 2^min(E, top)` cells per face edge. COMPUTED from
`crates/seed/src/ladder.rs:22-23,48-57,76`, at `E = 9`:

| body | `N` | top rung | `M` | metric node | rivers? |
|---|---|---|---|---|---|
| a 50 km airless moon | 78 528 | 5 | 32 | **2 454 m** | NO |
| a 100 km airless moon | 157 056 | 6 | 64 | **2 454 m** | NO |
| a 1 000 km moon | 1 570 816 | 9 | 512 | 3 068 m | maybe |
| the home planet | 5 263 360 | 11 | 512 | 10 280 m | yes |
| an Earth-sized planet | 10 006 528 | 12 | 512 | **19 544 m** | yes |

**Every airless body that gets no river network at all gets the world's FINEST drainage skeleton, and
the resolution gets WORSE as a planet gets more Earth-like.** A 53-fold metric spread also breaks the
lawfulness argument revision 2 leaned on: `SHORT_WAVE_M = 30 m` is METRIC and identical on every body,
and `E` is ANGULAR and identical on none. They are not the same kind of constant. `E` is a COST KNOB
traded against fidelity, and it should have been named as one.

**THE REPLACEMENT RULE, adopted from DOMAIN 03 §4.1.**

```
  MACRO_CELL_TARGET_M   a stated METRIC resolution of the world     (recommended 8 192 m)
  n_macro  =  the divisor of N whose node size N / n_macro is closest to MACRO_CELL_TARGET_M,
              ties to the smaller divisor, clamped to at least 8, and clamped above by a stated
              MEMORY CEILING for very large bodies
  nodes    =  6 · n_macro²
  the node's size in rung-0 cells  =  N / n_macro,  an EXACT integer, so the lattice tiles a face
              and a rung-L cell finds its node by one integer division, with no float comparison
```

COMPUTED for the home planet (`N = 5 263 360 = 2^12 · 5 · 257`): `n_macro = 640`, node **8 224 m**,
**2 457 600 nodes**, 9.83 MB kept. COMPUTED elsewhere: a 200 km moon gets 32 and 6 144 nodes; a 50 km
body gets the clamp at 8 and 384 nodes — which is what makes §3.3's driver test affordable.

**Why this is lawful.** `MACRO_CELL_TARGET_M` is a stated METRIC resolution, exactly like
`SHORT_WAVE_M = 30`: it says the drainage skeleton of THE world is resolved at about eight kilometres,
everywhere, on every body. It stands in for no physical fact the world holds. The MEMORY CEILING is a
cost knob and is named as one.

**What it costs, so the owner can choose the target.** ESTIMATED from §5's model:

| `MACRO_CELL_TARGET_M` | home `n_macro` | home nodes | kept | transient | ESTIMATED solve, one core |
|---|---|---|---|---|---|
| 16 448 m | 320 | 614 400 | 2.5 MB | 31 MB | 3 – 5 s |
| **8 224 m** | **640** | **2 457 600** | **9.83 MB** | **125 MB** | **12 – 20 s, up to 40 s** |
| 5 140 m | 1 024 | 6 291 456 | 25.2 MB | 320 MB | 31 – 51 s |

**And the honest statement about what the skeleton is for.** At 8 224 m the home planet's macro node is
about six nodes across the owner's 50 km reference frame, and one node is about 210 pixels wide in a
1 280-pixel frame. **Nothing the player looks at in that frame is resolved by the solve.** The solve
carries the DRAINAGE SKELETON and the hardness and ice classes; **what the player SEES is the slope
spectrum plus the closed-form carve keyed to that skeleton** (§3.7). Revision 2 attributed the picture's
ridges and valleys to the macro map and then judged them on a picture taken before the closed-form layer
existed. §6 moves that judgement to the slice that lands it.

---

## 5. The cost model, gathered

**Built from an operation model, and rebuilt twice.** Revision 1 costed an erosion cell at a noise
sample's rate, which is a category error: a noise sample is a register-bound polynomial on data in
cache; a solve step gathers eight neighbours out of a multi-megabyte array. Revision 2 fixed that and
then made a second one: it costed a priority-flood HEAP at one random access per node, and it ran the
flow routing twice in total instead of every `FLOOD_EVERY` passes. This is the third model. It agrees
with DOMAIN 03's independent one, and it is still ESTIMATED.

```
   ONCE per body, EVER (home planet, n_macro = 640, 2 457 600 nodes)
   -----------------------------------------------------------------
   D8 receivers      2.46 M nodes x 8 neighbours through the seam table, integer     0.1 s
   ONE priority      2.46 M pushes and pops; heap depth log2(2.46M) = 21 levels;
     flood             a 20 MB working set, larger than L2; ESTIMATED 6-8 misses
                       per node at 80 ns                                          1.0 - 1.5 s
   the stack and     2 x 2.46 M, integer                                            0.1 s
     the accumulation
   ONE stream-power  2.46 M, one sqrt, one divide, two random loads              0.15 - 0.25 s
     sweep
   ONE flexural      restrict, smooth 9 600 nodes, prolong                          < 0.05 s
     rebound
   -----------------------------------------------------------------------------------------
   THE SCHEDULE      PASSES = 40 sweeps, FLOOD_EVERY = 10 (so 4 floods),
                     ISOSTASY_EVERY = 5 (so 8 rebounds)                            11 - 17 s
   talus (8), ice (~20), the coast mark, the envelope renormalisation                1 - 3 s
   -----------------------------------------------------------------------------------------
   THE WHOLE SOLVE                                 ESTIMATED 12 - 20 s, plausibly up to 40 s

   MEMORY DURING THE SOLVE (COMPUTED from what is held at once)
     z_terrain 4 + z_flood 4 + receiver 1 + area 8 + Q 8 + lake id 4 + precipitation 2
     + ice 4 + the talus second buffer 4  =  39 B/node  =  95.8 MB
     + the heap (up to 8 B/node, 19.7 MB) + the flood's stack (4 B/node, 9.8 MB)   = 125 MB
   MEMORY KEPT                                 9.83 MB artifact + 1.6 MB pyramid
```

**Why the flood dominates.** It is the only step that is neither a stencil nor a straight walk. Every
node is pushed and popped through a heap whose lower levels do not fit in cache, and the neighbours it
reads are visited in HEIGHT order, which is spatially random. Revision 2 gave this step the same rate as
a streaming pass and got 0.25 s where the honest range is 1.0 to 1.5 s — and it ran it twice instead of
four times.

```
   PER CHUNK, on a worker (rung 0)      MEASURED and ESTIMATED, from S3.6
   -----------------------------------------------------------------------
   MEASURED plain surface chunk       3 300 us   |  worst named chunk   6 110 us
   + landforms, dry (the six rows)   +1 350 us   |  + landforms, dry   +1 350 us
   ---------------------------------------------|---------------------------------
   plain, dry            ESTIMATED    4 650 us   |  worst, dry          7 460 us   of 8 000
   + a meshed water sheet            +1 000 us   |  + a water sheet     8 460 us   OVER BUDGET
   -----------------------------------------------------------------------
   NO OCTAVE SAVING IS CREDITED. Revision 2 credited 254 us for dropping five octaves.
   The octave table does not shrink (S3.7), so the saving does not exist.
```

**Three costs the owner should see named.**

1. **A body's FIRST solve** is ESTIMATED 12 to 40 s of one core. Under the SHIP road it is paid ONCE
   EVER, on the shard that owns the body, cached in that shard's store, and never on a client.
   **Whether it is paid at world creation instead of at first demand is an owner question** (§9, L23):
   a cold first boot of the home planet's realm answers no chunk for that long, and there is no proxy.
2. **A client's first visit** is 9.83 MB of bulk for the body it is at, and 19 KB for the approach. It
   is a transfer, not a stall.
3. **Memory** is 9.83 MB kept per resident large body plus 1.6 MB of pyramid, and 125 MB transient on
   the SOLVING shard only. A gateway holds none. The home system's worst case is UNMEASURED (U-L13),
   because the source that records 10 planets and 11 moons does not record their radii, and the cost
   rises as `R²`.

---

## 6. The slice plan

### 6.1 The rule that fixes the order — and the loop that fixes it differently

Format D's tolerance is ZERO (ruling S5-2, `owner_decisions_2026-09-07_voxels.md:372`). Every landform
slice moves rung-0 bytes, and slice 8a also moves the REGISTRY digest (§3.4). **The free window closes
when the first world a player owns is saved. The store lands at slice 9**
(`docs/design/owner_decisions_2026-09-07_voxels.md:385`: *"S5-3 (the deltas) and S5-6 (the seed law)
bind the store, slice 9, and the block record"*), so slice 9 is the earliest that can happen.

**A correction the owner should see, because it is the kind that matters.** Revision 2 carried this same
conclusion under a quotation that does not exist: it cited line 263 as *"no store exists yet (the first
is slice 9)"*. The ruling's word there is **caller**, not **store**, and the sentence is about the grid
derivation's first caller. Refuter A caught it. The conclusion holds on line 385; the quotation was
manufactured, and a manufactured quotation carrying the plan's central ordering claim is worse than the
unsourced claim it replaced.

**Why not before slice 8.** `D-TERRAIN-3` is red (`docs/design/DEFERRED.md:7775`), and the owner said a
flag must not survive: *"I don't want it to survive."* Slice 8 deletes it. Slice 8's machinery — the
tier rule, the crossfade, the residency band — does not depend on WHAT the field is, only on the field
having rungs. So slice 8 goes first, and the landform slices re-run its detectors.

**THE ORDER INSIDE THE ARC CHANGED, because the physics is a LOOP.** Revision 2 put the erosion at 8b
and the climate at 8d. Refuter B is right that the dependency runs both ways:

```
   relief  ->  wind and orographic lift  ->  rainfall  ->  discharge  ->  erosion  ->  relief
     ^                                          |                                        |
     |                                          v                                        |
     |                                    snow line  ->  ICE  ->  cirques and troughs  ->-+
     +----------------------------------------------------------------------------------+
```

Two consequences, and both are visible in one frame.

- **Discharge, not area.** In the stream power law `A` is drainage AREA used as a PROXY for discharge.
  The proxy holds inside one climate and fails across a rain shadow: a leeward basin of the same area
  carries a small fraction of the windward basin's water. With the climate landing AFTER the erosion, a
  desert basin and a rainforest basin on the home planet are cut to the same depth. The gate *"a ridge
  has a wet side and a dry side"* would then measure a paint colour on ground carved as if both sides
  were wet.
- **No ice.** Above the snow line there is no liquid water, so stream power does not act. Without a snow
  line at solve time the high ground is uplift noise plus creep, painted white.

**The cure, and it is cheap.** The climate is a CLOSED FORM over the macro lattice: 2.46 M nodes, no
heap, no ordering. DOMAIN 05 ESTIMATED a whole coarse climate grid at 30 ms once. **So the climate runs
INSIDE the solve's schedule, every `CLIMATE_EVERY` passes, on the relief as it then stands** — which is
how real landscape evolution models break the same loop. The slice order becomes: the climate's MODEL
lands with the solve (8b), and the biome, the vegetation and the art-facing classification land after it
(8d). Nothing moves twice.

```
 landed  ->  slice 8         slice 8a      slice 8b            slice 8c      slice 8d   ->  slice 9 ...
 (0..7)      the ladder,     the charter,  the macro solve:    rivers,       biome, snow      the store
             the crossfade,  the two       climate IN the      lakes,        line, tree
             D-TERRAIN-3     relief        schedule, flood,    coasts, the   line, soil,
             deleted         bounds,       accumulation,       carve, the    vegetation
                             the slope     stream power,       stratigraphic density, the
                             spectrum      isostasy, talus,    column, the   art-facing
                                           ICE, craters,       cliffs        table
                                           the seam and
                                           the corner                        slice 18: the almanac
                                                                             and live weather
```

### 6.2 What changes in slice 8's scope

Slice 8 keeps its scope and gains three sentences.

- **Nothing is added to what it builds.** The tier rule, the coarse-before-fine order, the dither
  crossfade, the residency band from server-stated data, and the deletion of `D-TERRAIN-3`.
- **Its detectors are written to be RE-RUN.** The pop detector, the arrival-rate gate and the
  rung-disagreement measurement each become a recipe that a later slice invokes with one command.
- **Two new preconditions are written into it.** (a) The client must not build a chunk of a realm before
  that realm's whole-body artifact has arrived for that neighbourhood; from slice 8b that is the macro
  artifact. (b) The residency band must STATE the distance at which a body's whole-body data must be
  complete, because that distance is the artifact's transfer deadline and nothing today names it (U-L4).

### 6.3 Slice 8a — the charter, the two relief bounds, and the slope spectrum

*Lands:* the body charter as quantised integers, authored by the parent; the quantum grids; the widened
`TAG_SURFACE`; the client's use of it; the home body's charter PINNED in `home.rs` and cross-pinned in
`home_body_pin.rs`; the yield-strength column on the registry's substance rows; the relief as
`draw × min(strength bound, shape bound)` with `draw ∈ [0.5, 1.0]` and the shape bound at `0.077 R`; the
water-gate flag; the sea level from a water inventory poured onto the hypsometry; the SLOPE SPECTRUM
replacing `k_rough`; the deletion of the `200 / 12 000` clamp pair and the sea's `0.7 / −0.4`.
`POLE_AXIS` is UNTOUCHED and the octave COUNT is untouched.
*Measurements first:* **M-L8**, the home planet's own drawn mass and density (extend `terrain_cost` to
print the taxonomy row), because §3.4's illustration assumes them; **M-L10**, the charter's cross-host
spread, so each quantum is far larger than a measured perturbation; the relief bounds for the home
planet, a 100 km moon, a 1 000 km moon and the largest body, so the owner sees that no body is deleted;
the home planet's ocean fraction under a water inventory, swept over the draw's range.
*Gate:* a quantum test on EVERY charter fact (a fact moved inside its quantum gives the same body; one
quantum changes it); a cross-pin that the forest's numbers and the recipe's integers agree; **a test
that `Ladder::for_radius` still accepts the smallest round body in the world**; the shipped M-16 test
still green (the octave count is unchanged); the golden tables re-recorded; `just terrain-pin` (three
legs) and `just terrain-legs` (two legs) green; the registry digest's move recorded and the
store-refusal path exercised; Tier-A at 100 %.
*Pictures:* a 50 km vista under the SLOPE SPECTRUM alone, with no solve. **This is the first honest test
of *"not interesting enough"*, and it is cheap.**
*From the owner:* the SL6 row for the charter (L-1); who draws the spin, the obliquity and the water
inventory (L7); the yield strength's cited source; the target ocean fraction (L11); the draw's new range
(L3); the one-hop route for the orbit-derived facts (L16).

### 6.4 Slice 8b — the macro solve: climate in the schedule, water, ice, craters, the seam and the corner

*Lands:* the macro lattice `n_macro` as a divisor of `N` near `MACRO_CELL_TARGET_M`; the closed-form
climate over the lattice, run every `CLIMATE_EVERY` passes; D8 routing with an exact integer tie-break;
priority-flood depression filling with two named surfaces; flow accumulation in WHOLE SQUARE METRES
weighted by the bend's own area density; the implicit stream-power sweep at `m = 1/2, n = 1` driven by
DISCHARGE; flexural isostasy on a pyramid; talus relaxation; **the ICE pass (mask from the equilibrium
line, trunk over-deepening, cirque marking)**; **the crater field and impact relaxation for an airless
body**; the coast mark; the seam-crossing flow direction and the seven-neighbour corner; the fixed
reduction order and the single-threaded solve; the stated pass counts; the ENVELOPE RENORMALISATION and
the exact `relief_bound_m`; the artifact's own digest and its bulk payload kind; the shard's cache of
the artifact; `BodyDefinition` stops being `Copy`; the kernel canary in the world identity; the
structural control that the gateway holds no artifact.
*Measurements first (M-L1, M-L2):* solve wall time and peak memory on ONE worker, for a 100 km moon, the
home planet and an Earth-sized body, at two or three values of `MACRO_CELL_TARGET_M`; the artifact's
digest equal across the target legs; **`D-TERRAIN-1`, a real x86-64 machine, or a stated acceptance that
"no drift" is unsatisfied** (§9, L22); the boot self-check still under 20 ms with the kernel canary.
*Gate:* every node has a downhill path to a base level (**no node drains to nowhere, tested at all eight
cube corners and along every one of the twelve face seams** — ruling A5); **G-MACRO-EDGE, G-MACRO-CORNER,
G-MACRO-AREA and G-MACRO-SEAM** (§3.16); `|Z| ≤ A_coarse` at EVERY node after the renormalisation; the
gateway's resident artifact set empty after a flight; the 8 ms chunk budget re-measured, with the
cave-dense-plus-water case named as the one expected to be tight; a body with no water gets craters and
no river.
*Pictures:* the whole body from orbit, coloured by height; the same from 200 km; **a picture at a cube
corner and one across a face seam**; a glacial skyline against a fluvial one, side by side.
*From the owner:* derive or ship (L1); `MACRO_CELL_TARGET_M` and the memory ceiling (L4); the pass
counts' convergence evidence (L13); the ice pass (L20); the real x86-64 leg (L22); solve at world
creation or at first demand (L23).

### 6.5 Slice 8c — rivers, lakes, coasts, the carved valley and the stratigraphic column

*Lands:* the river graph from the accumulation; lake surfaces from the flood fill; the closed-form
tributary network below the macro node; the valley as a shaping function of the existing height, keyed
to the graph; water cells in the valley and the lake, with the water sheet CLIPPED to where water
stands; the coast where land meets sea; **the stratigraphic column at a fixed radius, and differential
erosion on it, so a caprock leaves a mesa and a pillar**; the alluvium on a valley floor; the
rung-continuous valley depth; the re-derived chunk margin for cliffs.
*Measurements first:* the per-column river query against the 8 ms budget; **the true distribution of
river segments per macro node, so the query's bound is measured, not stated**; the rung-to-rung
disagreement of the valley field; **the cave-dense-plus-water-sheet chunk, which §3.6 estimates at
8.46 ms against a budget of 8 ms**.
*Gate:* no river runs uphill (the stage falls downstream over 100 000 sampled segments, seams included);
a lake surface is level to within one quantum; slice 8's pop detector re-run at 1.4, 240 and 528 m/s,
passing at p99 ≤ 1 pixel and max ≤ 2 pixels; a water cell never sits above its own stage; the segment
cap carries a refusal test, never a silent truncation; the 8 ms budget met on every named chunk after a
lever is pulled.
*Pictures:* a river valley from the ground at rung 0; the same river to the horizon; a lake; a coast; **a
cliff with visible rock bands; a mesa; a rock pillar**; **THE 50 km VISTA THE OWNER JUDGES "ORIENTERS"
ON.** Revision 2 put that judgement at 8b, before the closed-form layer that draws everything in the
frame existed. **The crease question and the cave-lattice step, which slice 7's three pictures could not
judge, are judged here** (ruling V10 left both open).
*From the owner:* the crease answer (surface nets round a cliff by half a cell of the rung drawn; dual
contouring keeps the edge — a reserved upgrade, free until a world is saved); confirmation that rivers
carry no ore (L5); the stratigraphic column (L19).

### 6.6 Slice 8d — the biome, the snow line, the soil, and what the vegetation reads

*Lands:* the Whittaker biome table with an aridity term and a short ordered chain of overrides (no
atmosphere, sea, beach, permanent snow, steep slope, wetland) — sixteen biomes in five bits, and the
cell record does not change; the snow line and the tree line as temperatures; the soil law per biome;
sea ice where the annual mean is below freezing; the strata that follow the biome and the landform; the
vegetation density and species mix that slice 14's skeleton reads; the star class's effect on the light
and so on the plant colour. **`POLE_AXIS` is untouched. The climate MODEL landed at 8b; this slice is
what the art kit reads.**
*Measurements first:* the classifier's cost per column; the vendored saturation-vapour polynomial's
maximum error against a reference table; the biome census over the home planet.
*Gate:* a ridge has a wet side and a dry side, measured on sampled transects **and measured as a
DISCHARGE difference, not only as a colour**; the snow line rises toward the equator; every biome the
registry names appears somewhere; the classifier byte-identical on all legs; **the link scan finds no
transcendental symbol**; Tier-A at 100 % through the kernel and small-body tests.
*Pictures:* the snow-capped ridge; the forest mass at 5 km and at 50 km; a desert behind a range; the
same latitude band on the wet side and on the dry side.
*From the owner:* the biome list; whether the home planet's biome census is the world he wants; whether
a static snow line is what he expects, because **a static snow line is a PERMANENT snow line, which
stands far higher than the seasonal one the reference picture probably shows** (the almanac is what
moves it).

### 6.7 Slice 18 — the almanac and live weather (after 8d; not on the shape's critical path)

*Lands:* the ALMANAC as a closed form on the universe tick — the season's phase, the sub-solar point,
the day's phase, the nominal temperature and cloud — computed by the owning realm and shipped in a small
row; the LIVE weather as a small set of seeded PARAMETERS the client evaluates as a procedural field
(cyclone centres, radii, strengths, a phase), on its OWN lane, never on `TAG_SURFACE`; the decaying live
offset with its stated time constant; the client's cloud, rain and haze; wind in the realm's medium, so
the parent's existing drag reads it.
*Laws:* SL3 (the realm states its own look), SL7 (its own observers), SL10 (nothing enters the recipe),
the movement contract (nothing new crosses upward), the dormant-world rule (the offset's decay is
advanced in one step on wake; the almanac needs no rule at all).
*Measurements FIRST, and this is the change from revision 2:* **U-L8, the bytes and the rate**, because
COMPUTED a stored coefficient grid fine enough for a 50 km storm front is 393 KB per statement against a
1 200-byte self-look bag (§3.14). **L-4 is not asked until U-L8 answers.**
*Gate:* sleep a planet for an hour of world time, wake it, and no cloud jumps; a planet that slept
through half its year wakes in the right season; two clients side by side paint the same cloud in the
same place; the statement's bytes per observer per minute against the lane's budget.
*Pictures:* a cloud deck from orbit; rain on the ground; a storm on the horizon over a desert, with clear
sky elsewhere in the same frame; the same ridge in summer and in winter.
*From the owner:* whether weather pushes a ship or only paints the sky (L8); whether *"simulate"* means a
real circulation (L17, after U-L8); the statement's rate.

### 6.8 What each slice owes the ones after it

| Slice | Gives to |
|---|---|
| 8a, the charter and the spectrum | every later slice; slice 16's gravity function; the client's haze; the water gate; **and the first cheap answer to "not interesting enough"** |
| 8b, the macro solve | 8c (the graph), 8d (the climate field), 11 (the collider reads the same shape), 14 (where a forest can stand) |
| 8c, the rivers and the stratigraphic column | 10 (a player digs a canal), 14 (a riverside species), the picture's cliffs, **the owner's "orienters" judgement** |
| 8d, the biome | 14 (the vegetation density and mix), 18 (the weather's baseline), the client's sky |
| 18, the almanac and the weather | the client's look; the realm's medium; the season that makes spin and tilt visible |

### 6.9 The reference picture, mapped to a slice

The owner's reference is a hand-authored vista: snow-capped ridges, carved valleys, cliffs and pillars, a
forest as a mass, fields, a river, roads, blue haze, clouds, a character for scale, detail to the horizon
over 50 km.

| What the picture shows | What produces it | Slice |
|---|---|---|
| Ridges you can navigate by ("orienters") | the SLOPE SPECTRUM first, then the drainage skeleton and the carve keyed to it. **Not the macro solve alone: one macro node is 210 pixels wide in this frame** | 8a, then 8b and 8c |
| Carved valleys with floors and shoulders | the stream power law on the skeleton, then the closed-form valley | 8b, 8c |
| A river running to the horizon | the river graph and its clipped water sheet | 8c |
| **A glacially sculpted skyline** — cirques, arêtes, U-shaped troughs | **the ICE pass on the macro lattice.** Fluvial incision alone gives a V-notch to the summit, and revision 2 mapped this row to the lapse rate, which is paint | **8b — newly named; §9 L20** |
| **Cliffs and rock pillars** | **the stratigraphic column at a fixed radius, plus differential erosion, plus the crease answer** | **8c — newly funded; revision 1 mapped these to a strata table that cannot produce them** |
| Snow caps on the ridges only | the lapse rate and the snow line (permanent), the almanac (seasonal) | 8d, 18 |
| Forest as a mass | the biome, the vegetation density, the canopy fold | 8d, then 14 |
| **Fields** | **built, not seeded. Live state. The same class as roads** | **out of this arc — corrected** |
| Blue haze with distance | the atmosphere's scale height, on the client | 8a gives the number; the client paints it |
| Clouds, and weather that differs across the frame | the live weather parameters | 18 |
| A character for scale | the character on the surface | 16 |
| Detail to the horizon, no visible jump | the ladder, the crossfade, the rung-continuous valley, the C1 macro read, the pyramid | 8, re-run after 8b and 8c |
| Roads | built, not seeded. Live state. | out of this arc |

---

## 7. The SL6 asks — new data crossing a realm boundary

The default is NO. Each row states the data, the direction, why the receiver cannot compute it, and what
it costs to do without it.

| # | Data | From → to | Why it cannot be computed locally | Cost of refusing | Recommendation |
|---|---|---|---|---|---|
| **L-1** | **THE BODY CHARTER: quantised INTEGERS** (gravity, bulk density, escape velocity, mean insolation, equilibrium temperature, the atmosphere's two numbers, the water flag, the spin period, the obliquity's cosine, the eccentricity, the star class, the water inventory), inside the widened `TAG_SURFACE`. ESTIMATED 24 bytes packed, against `SELF_LOOK_BUDGET_BYTES = 1 200` shared with the outline and the luma (`crates/core/src/look.rs:78`). | the body's realm → its observers' clients | The client links no motion crate in its library and must not re-derive an unfenced float in its binary (§4.2). | The climate cannot depend on the planet's size, gravity, spin, tilt or orbit — which is exactly what the owner asked for. | **ASK: yes.** A protocol-minor bump; the floor does not move. The byte count must be MEASURED against the budget before it lands. **Integers, never floats.** |
| **L-2** | ~~The macro-artifact digest, in the same tag~~ | — | — | — | **WITHDRAWN, and it stays withdrawn.** `crates/core/src/look.rs:70-73` forbids the measured half in a look by name: *"a chip difference refuses a lane, never a look."* The artifact carries its digest on the BULK lane it arrives on (L-3), and the recipe VERSION already rides the tag. |
| **L-3** | **THE MACRO ARTIFACT ITSELF**, as bulk rows on the existing bulk class: the 19 KB pyramid top for the approach, then the 9.83 MB field for the body the player is at, each carrying its own digest | the body's realm → the client | **The client cannot solve it: ESTIMATED 12 to 40 s of one core, and it cannot compute the charter the solve needs.** | No macro field on the client at all, which means no rivers, no coasts and no eroded shape. | **ASK: yes. This is now the PRIMARY ask, not the fallback.** It needs a payload kind, a manifest and a resume path. M-L1 could still return it to a derive, and the row says so. |
| **L-4** | The LIVE weather, as a small set of seeded PARAMETERS on a NEW lane with its own rate | the body's realm → its observers | Weather is live state, and the client may never derive live state. | No weather, which the owner asked for by name. | **ASK LATER, after U-L8.** COMPUTED: a stored coefficient grid fine enough for a 50 km storm front is 393 KB per statement against a 1 200-byte bag (§3.14). Until the bytes are measured the owner cannot rule. It may NOT ride `TAG_SURFACE`, which is *"ONCE PER REALM ON CHANGE, carried retained, never on a keep-alive"* (`crates/core/src/look.rs:53-54`). |
| **L-5** | A weather field composed PER OBSERVER | the realm → each observer | — | — | **NO.** That is the composed-per-observer shape the reach ruling killed. **It is NOT the same as a per-PLACE field:** one statement for the realm, which every observer evaluates, costs the realm nothing per observer. |
| **L-6** | **THE ALMANAC ROW**: the season's phase, the sub-solar direction and the day's phase, as a few integers on the realm's own lane, restated at a slow stated rate | the body's realm → its observers | It is a function of the universe tick and the charter. The client could derive it from the charter — but V12 already put the LIGHT on the ship-it side, and one authority for one datum is the rule. | Spin, tilt and eccentricity stay invisible: no season, no moving snow line, no day. The owner's word *"trajectory"* is then answered by a 1.5 % change in an annual mean. | **ASK: yes**, at slice 18, and it is small. |

---

## 8. The one-way doors

| Door | Free until | Cost after |
|---|---|---|
| The relief becoming two bounds, with the draw INSIDE and a `[0.5, 1.0]` range | the first saved world | every player's world, at every rung |
| **The slope spectrum replacing `k_rough`** | the first saved world | the same, and it changes the near ground everywhere |
| The yield-strength column on the registry | the first saved store | the registry digest moves and every store refuses to open |
| `MACRO_CELL_TARGET_M` and the memory ceiling | the first saved world | the same |
| The solve's pass counts and re-routing period | the first saved world | the same; they are part of the declared tag |
| Rivers, lakes, coasts and the ice pass | the first saved world | the same |
| The stratigraphic column at a fixed radius | the first saved world | every cliff and every mine face changes |
| **The spin, obliquity and water-inventory DRAW LAWS** | the first saved world | a first draw law is always tuned, and each tuning re-records the golden tables and bumps the version. **Write and tune them before slice 9** |
| ~~Obliquity replacing `POLE_AXIS`~~ | — | **DELETED.** The pole does not move; the parent tilts the body and the recipe reads a scalar (§4.2). |
| The crease answer (surface nets or dual contouring) | the first saved world (ruling S6-1 reserved it) | a re-mesh of every chunk and a version bump |
| The widened `TAG_SURFACE` (the charter) | its first consumer, which is slice 8a itself | a protocol-minor bump per change afterwards |
| The artifact's bulk payload format | its first consumer, slice 8b | a payload version and a refusal path |
| The new weather lane | its first consumer, slice 18 | the same |
| Rivers carrying value | it never opens | a seed-derived treasure map, refused by the seed ruling |
| Live landscape change over time | any time, as live state | a new diff family and a new pyramid consumer |

---

## 9. What you decide

| # | Decision | Options | Recommended | Why |
|---|---|---|---|---|
| **L1** | **REVERSED.** Does the client DERIVE the macro artifact, or does the owning realm SHIP it? | (A) derive on both hosts; (B) **ship, coarse first** | **(B)**, and M-L1 runs before it is final | ESTIMATED 12–40 s of one core per body, and the client cannot compute the charter the solve needs. Revision 2 recommended (A) on a model about thirty times low. |
| **L2** | Do the body facts cross as INTEGERS the PARENT quantises, or as floats the recipe snaps? | (A) **integers, a charter**; (B) floats, snapped by the receiver; (C) the recipe draws everything | **(A)** | `crates/physics` has no float fence and its drift gate is DEFERRED (SPIKE-6a). The recipe's own invariant forbids (B) in its own words. The home body's charter is pinned in `home.rs`, as the radius already is. |
| **L3** | The relief: `draw × min(strength bound, shape bound)` with `draw ∈ [0.5, 1.0]`, and the shape bound at `0.077 R` (Vesta)? | (A) yes; (B) keep `0.004 / 200 / 12 000 × [0.5, 1.5)` | **(A)** | Without the draw's new range a bound is not a bound. Without the shape bound every small round body is DELETED from the world. With `0.004` a 100 km moon gets 400 m of relief, and Vesta — the report's own witness — carries nineteen times that. |
| **L4** | **CHANGED SHAPE.** `MACRO_CELL_TARGET_M`, and the memory ceiling for a large body | 16 448 m (2.5 MB, 3–5 s); **8 224 m (9.83 MB, 12–20 s)**; 5 140 m (25.2 MB, 31–51 s) | **8 224 m, confirmed ON PICTURES at 8c**, with a ceiling stated for bodies above Earth's size | Revision 2's `2^E` rule is withdrawn: it gave a 2 454 m skeleton to an airless moon with no rivers and 19 544 m to an Earth-sized planet with rivers. A METRIC target gives every body the same skeleton. |
| **L5** | Do rivers, beaches and deltas carry ore? | (A) no; (B) yes | **(A) no** | A seed-derived map of value is a treasure map, refused by name. Sand, gravel and clay are bulk stock and stay lawful. |
| **L6** | Where do the landform slices sit? | (A) before slice 8; (B) **between 8 and 9**; (C) after the store | **(B)** | (A) keeps a flag the owner refused; (C) costs a world epoch and every saved store. |
| **L7** | Who draws the spin period, the obliquity and the water inventory? | (A) the forest, and the recipe receives them quantised; (B) the recipe | **(A)** | The forest authors the body's rotation on the placement row, and one fact must have one drawer. **Write and TUNE the three draw laws before slice 9** (§8). |
| **L8** | Does weather push a ship, or only paint the sky? | (A) look only; (B) look and force | **(B)**, with the force added at slice 18 and measured | Wind is already expressible in the realm's medium, and drag already reads a medium. |
| **L9** | May a realm CACHE its solved artifact in its own store? | (A) no, always solve; (B) **yes**, keyed by the world tag and the body seed | **(B), and it is now REQUIRED, not optional** | A 12–40 s solve on every spin-up of a realm the demand loop cycles is not a design, it is a stall. |
| **L10** | Does the landscape change over game time? | (A) no, static; (B) yes, as live state | **(A) for this arc** | SL10 forbids time inside the recipe. (B) stays possible later as a diff family. |
| **L11** | The home planet's ocean fraction | today about one column in a hundred (MEASURED); Earth is 71 % | **the owner picks a target, and the water inventory's draw range is set to it** | This is a QUALITY choice and it is dressed as physics if it is not said plainly. **It is also a SYMPTOM: without a crust dichotomy (L21) the split is hypersensitive to the draw.** |
| **L12** | The crease: does a cliff keep its edge? | (A) surface nets, rounded by half a cell of the rung drawn; (B) dual contouring | **decide at slice 8c's pictures**, while it is still free | Ruling V10 reserved it; slice 7's three pictures held no cliff. |
| **L13** | The solve's pass counts are STATED constants, never a convergence test. | (A) yes, with a recorded convergence study; (B) a convergence test | **(A)** | One drifted ulp makes one host run one more pass than the other, and the two worlds differ (§3.1 rule 6). The recommended set is `PASSES = 40`, `CLIMATE_EVERY = 10`, `FLOOD_EVERY = 10`, `ISOSTASY_EVERY = 5`, `TALUS_PASSES = 8`. |
| **L14** | **CHANGED.** What does the boot gate check? | (A) the eight golden chunks on the SHIPPED PATH with the home artifact's OWN nodes pinned as literals, plus a kernel canary, plus the artifact's own digest at arrival; (B) a whole small body; (C) the home planet's whole solve | **(A)** | (C) is 12–40 s in the gateway and in every client. Revision 2 proposed a pinned value the world never uses, which inverts "test exactly production" at the one gate meant to enforce it; pinning the REAL artifact's real nodes does not. |
| **L15** | The river query's per-macro-node segment bound | (A) MEASURE the distribution, then state a cap with a refusal test; (B) state "at most eight" | **(A)** | Revision 1 stated eight with no source, and the 8 ms budget rests on it. |
| **L16** | Which realm states the ORBIT-derived facts (insolation, equilibrium temperature, eccentricity)? | (A) the SYSTEM computes and quantises them, the body holds its charter and states it; (B) the SYSTEM states them direct to the observer; (C) ship a mean surface temperature instead | **(A)**, and the owner should rule on SL1 clause 4 explicitly | Under the charter the datum crosses ONE hop from the realm that authored it, and none of these facts is a placement. (B) splits one body's facts across two lanes. |
| **L17** | Does *"simulate the weather"* mean a real circulation? | (A) an almanac plus a decaying live offset (a believable animation); (B) a coarse global circulation stepped on the climate grid by the body's shard | **(A) for this arc; ask again after U-L8** | (A) is free when a realm sleeps and costs nothing per observer. (B) is live state, breaks no law, and costs a per-tick pass over the grid. **The answer changes if a field cannot cross the lane at all.** |
| **L18** | May slices 9 to 13 keep a throw-away store? | (A) yes, no player world exists until slice 14; (B) no, the window closes at slice 9 | **state it, and plan for (B)** | If a real world can be saved at slice 9, the free window is shorter than the plan assumed. |
| **L19** | Do the strata become a stratigraphic column at a fixed RADIUS? | (A) yes, for the sediments and the bedrock; the soil stays depth-following; (B) keep the depth-following cake | **(A)** | Without it a hard band can never outcrop, so there is no caprock, no mesa and no rock pillar — and the reference picture shows all three. |
| **L20** | **NEW. Does the arc land an ICE pass?** | (A) yes, an ice mask from the climate and a widening and over-deepening of the trunk valleys inside it — one more streaming stage in the schedule; (B) no: alpine skylines are fluvial shapes with white paint | **(A)** | The reference picture's top third is a glacial skyline. Above the snow line there is no liquid water, so stream power does not act there at all. This is the single largest believability gap the two refutations found in the picture the owner pointed at. |
| **L21** | **NEW. Does the arc land a CRUST DICHOTOMY (two crust types, two densities)?** | (A) yes: it is what makes a two-humped hypsometry, a continental shelf and a stable coastline; (B) no, and the ocean fraction stays a hand-set number | **the owner rules; this report recommends asking DOMAIN 02 to price (A) before 8b** | Earth's two humps come from two crust types floating at different levels. Without it the land-sea split is hypersensitive to the water draw, and L11's hand-tuning is the symptom of the missing mechanism, not its cure. |
| **L22** | **NEW. Does `D-TERRAIN-1` — a REAL x86-64 machine — block the erosion slice?** | (A) yes, it lands first; (B) no, and the arc ships with "no drift" UNSATISFIED by the owner's own S5-5 definition | **(A)** | S5-5: *"emulation as a smoke test now; a real machine before 'no drift' is called satisfied."* A forty-pass accumulation over 2.46 M nodes is exactly what an emulator is least likely to reproduce. |
| **L23** | **NEW. Is a body's first solve paid at WORLD CREATION or at FIRST DEMAND?** | (A) at first demand, on the owning shard, cached; (B) offline at world creation for the home system's bodies, shipped with the world | **(A) with (B) for the home system** | A cold first boot of the home planet's realm answers no chunk for 12–40 s, and the home planet is where every gate and every new player starts. |

---

## 10. Open questions, and what is UNMEASURED

| # | Unknown | How it would be measured | Which slice needs it |
|---|---|---|---|
| U-L1 | **The solve's wall time and peak memory on one worker**, for a 100 km moon, the home planet and an Earth-sized body, at two or three values of `MACRO_CELL_TARGET_M` | a bench beside `crates/bins/examples/terrain_cost.rs`, release, idle machine, three runs | 8b; **L1, L4 and L23 all depend on it** |
| U-L2 | Whether a solved artifact is byte-identical on every target leg | the same bench, digests compared under `terrain-pin`'s three host legs and `terrain-legs`'s two — **and on a REAL x86-64 machine (L22)** | 8b |
| U-L3 | The per-column cost of the macro read, the carve, the water level, the detail and the climate, against the 8 ms budget — **and specifically the cave-dense chunk that also holds a meshed water sheet, ESTIMATED at 8.46 ms** | extend `terrain_cost` with the named chunks it already uses, at both ends of the ladder | 8b, 8c, 8d |
| U-L4 | **The residency band's own distance** — at what range must a body's artifact be complete? — and a body realm's wake-to-first-chunk time against it | slice 8's residency band must STATE it; then the wake probe with a planet realm | 8, 8b; L9 depends on it |
| U-L5 | The rung-to-rung disagreement of the eroded field, in pixels at the switch distance | slice 8's detector, re-run | 8b, 8c |
| U-L6 | Whether the slope spectrum plus an 8 224 m skeleton gives landmarks a pilot can navigate by at 50 km | the pictures, judged by the owner — **at 8a for the spectrum alone, and at 8c for the whole chain** | 8a, 8c; L4 depends on it |
| U-L7 | The home planet's biome census under the new climate | a census bench over sampled columns, like slice 5's sea-cover census | 8d; L11 depends on it |
| U-L8 | **The bytes and the rate of a live weather statement**, and whether a seeded-parameter field can draw a 50 km storm front inside the lane's budget | the load harness of slice 10, extended | 18; **L-4 and L17 both wait on it** |
| U-L9 | The client's first-visit behaviour: the artifact's transfer time and the frames dropped while it arrives | the window flight, release, with the frame timer | 8b |
| U-L10 | Whether a cited yield strength exists for every bedrock the registry names | a literature pass, cited per row, like the density and fracture columns | 8a |
| U-L11 | Whether the realm's look shell should include the macro relief, so no peak stands outside the realm's own stated look | the sprite-cull detector at the reach edge | 8a |
| U-L12 | The true distribution of river segments per macro node | a census over the home planet's artifact | 8c; L15 depends on it |
| U-L13 | **The client's and the shard's worst-case artifact residency in the home system** (10 planets, 11 moons, radii NOT recorded anywhere) | the window flight, in low orbit, with the resident-set probe; plus a census of the home system's radii | 8b; §3.10 |
| U-L14 | The maximum error of the vendored saturation-vapour polynomial over the world's temperature range | a unit test against a reference table, error reported | 8d |
| U-L15 | **The bend's true area weight over a whole face** (COMPUTED here: 0.702 at a face-edge midpoint, 0.758 at a cube corner, from `crates/seed/src/bend.rs:24-30,43`) | an exhaustive evaluation of the area density over a face, like `G-MAPPING-TABLE` — this is gate G-MACRO-AREA | 8b |
| U-L16 | The home planet's own drawn mass and bulk density — §3.4 assumes 3 900 kg/m³ | extend `terrain_cost` to print the home body's taxonomy row | 8a; L3 depends on it |
| U-L17 | The convergence behaviour of the solve, so each stated pass count is chosen from evidence | a one-off study off the shipped path, plotting relief change against pass count | 8b; L13 depends on it |
| U-L18 | **The CROSS-HOST SPREAD of every charter fact** — the quantity a quantum must absorb, which nobody has ever measured | draw the home system on both target legs, print every taxonomy float's bits, and compare | 8a; **L2's soundness depends on it** |
| U-L19 | **What an ice pass costs, and what it changes in a picture** | DOMAIN 03's ice pass, benched, and a skyline picture with it on and off | 8b; L20 depends on it |
| U-L20 | **What a crust dichotomy costs, and whether it gives a two-humped hypsometry** | DOMAIN 02 prices it; the hypsometric curve is measured against a real elevation model | before 8b; L21 depends on it |

---

## 11. Measurements owed, in order

1. **M-L1** the solve's time and memory (U-L1). **It decides L1, L4 and L23, and nothing else starts
   before it.** Revision 2 wrote its verdict before this measurement existed, and the verdict was wrong.
2. **M-L2** the target legs, including a REAL x86-64 machine (U-L2, L22). It decides whether an
   iterative layer is lawful at all, and it cannot decide that on an emulator.
3. **M-L10** the charter's cross-host spread (U-L18). It decides whether the quanta of L2 are safe, and
   it comes BEFORE slice 8a lands.
4. **M-L8** the home planet's own mass and density (U-L16). It decides whether L3's illustration is
   physics or a guess.
5. **M-L3** the per-column added cost against the 8 ms budget, with the cave-dense-plus-water case first
   (U-L3).
6. **M-L4** the residency band's distance and the wake-to-first-chunk time (U-L4). It decides L9.
7. **M-L5** the ocean-fraction sweep (U-L7 and L11). The owner judges it before the sea level freezes.
8. **M-L6** the rung disagreement, re-run after 8b and after 8c (U-L5).
9. **M-L9** the river-segment census (U-L12). It decides L15 and closes the 8 ms estimate.
10. **M-L11** the weather statement's bytes (U-L8). **L-4 is not asked before it, and L17 is not
    answered before it.**
11. **M-L7** the pictures, in the order of §6 — the slope spectrum alone at 8a first, because it is the
    cheapest test of the owner's own worry. They are the only measurement of "believable" the design has.

---

## 12. Stale claims corrected

| Claim, and where it is stated | Correction |
|---|---|
| "There is NO atmosphere on a planet today" (the task brief) | `crates/physics/src/taxonomy.rs:881-891` derives an `Atmosphere` per body. What does NOT exist is a surface pressure (`D-TAX-1`), a spin period and an obliquity. |
| "No surface gravity field on a planet today" (the task brief) | `crates/physics/src/taxonomy.rs:750` derives surface gravity from mass and radius, and `:755-758` derives escape velocity. What is missing is a lawful way for the RECIPE to read them. |
| "The column pass at rung 0 is about 1.5 ms per 62 × 62 columns" (the task brief) | MEASURED 710 µs at rung 0 with 14 octaves (`docs/investigation/2026-09-07/slice_05_generator.md:372`). |
| "The far view is the near hill without the small octaves" (`crates/terrain/src/height.rs:1-4`) | True today. From slice 8b the far view is the macro field plus the surviving octaves, read at a matched pyramid level. |
| "An obliquity the parent authors per body is a later slice" (`crates/terrain/src/height.rs:31-35`) | Named here as slice 8a's charter, and it does NOT move `POLE_AXIS` (§4.2). |
| Ruling D-2, "the generator crate owns a body's radius and the forest reads it" | Slice 5 did NOT do this. The radius stays an input, and the LADDER SNAP plus the PINNED BITS make it harmless. |
| "The cell pass at rung 0 is 578 µs" (`slice_05_generator.md`) | Superseded: **716 µs** with the site plumbing (`docs/investigation/2026-09-07/slice_06_extractor.md:361`). Revision 2 quoted the old number as a present fact. |
| **Revision 1: `N = R · sqrt(4π/6)`** | **WRONG.** `crates/seed/src/ladder.rs:76` is `n_ideal = radius_m * FRAC_PI_2`. Home `N = 5 263 360`. |
| **Revision 2: "the octave table starts at the macro cell — 9 octaves instead of 14"** | **WRONG, and it would turn a shipped Tier-A gate RED.** `octaves_at` floors the kept count at one (`body.rs:251-261`); the home planet has 12 rungs; the M-16 test (`body.rs:355-365`) fails at rung 10 with fewer than 12 octaves. **The octave count does not change.** §3.7. |
| **Revision 2: "a saving of 254 µs at 13.2 ns per octave per column"** | **WRONG twice.** The saving does not exist (the table does not shrink), and the rate is a MARGINAL 10.5 ns, COMPUTED from the two published points: `(710 − 266) ÷ 11 ÷ 3 844`. Revision 2 divided a total and assumed no fixed cost. |
| **Revision 2: "the macro grid is `2^min(E, top)` cells per face edge"** | **WITHDRAWN.** It gives a 2 454 m skeleton to an airless moon and 19 544 m to an Earth-sized planet — the resolution runs backwards — and `E` is an ANGULAR cost knob, not a metric resolution like `SHORT_WAVE_M`. Replaced by a metric target that divides `N` (§4.3). |
| **Revision 2: "TOTAL ESTIMATED 0.31 – 0.41 s, ONE core" and "Under a second on one core"** | **WRONG by about thirty times.** The priority flood is a heap, not a sweep, and the routing must re-run every `FLOOD_EVERY` passes. ESTIMATED 12 to 20 s, plausibly 40 s (§5). **The recommendation reversed because of it.** |
| **Revision 2: "the client already has a worker seam" and "a new body job kind is owed"** | **MOOT.** Under the SHIP road the client runs no whole-body job at all. |
| **Revision 2: `relief = draw × min(bound₁, bound₂)`, "so a bound is a bound"** | **FALSE as written.** The draw's range was never restated; at `[0.5, 1.5)` the home planet reaches 20 105 m above a 13 403 m bound. The range becomes `[0.5, 1.0]` (§3.4). |
| **Revision 2: "the accretion bound stays a share of the radius, which is lawful"** | **WRONG, and its own witness refuted it.** `0.004` is Earth's ratio; Vesta carries 0.077 R. The shape bound is now cited to an observed small body, and COMPUTED it crosses the strength bound at about 939 km of radius. |
| **Revision 2: "the bend makes a cell about 6.5 % narrower at a corner … the derivative falls to 0.735"** | **WRONG three ways.** The AREA error is 30 %, not 6.5 %; the extreme sits at a FACE-EDGE MIDPOINT (0.702), not at a cube corner (0.758); and `W'` RISES from 0.785 to 1.558. COMPUTED in §3.1 rule 7. |
| **Revision 2: "the sample box holds about 4 268 columns"** | **WRONG.** `64 × 64 = 4 096` (`slice_06_extractor.md:97,118,233`). Revision 2 applied the CELL ratio to a COLUMN count. |
| **Revision 2: "the boot canary reads a stated pinned value; this is STRONGER"** | **HALF WRONG.** No address depends on a single macro node under a 4 × 4 stencil, and a kernel canary cannot see accumulation drift. The cure is the REAL artifact's own nodes pinned as literals, plus the artifact's digest at arrival (§3.9). |
| **Revision 2: "the store lands at slice 9 (`owner_decisions_2026-09-07_voxels.md:263`: 'no store exists yet …')"** | **THE QUOTATION DOES NOT EXIST.** The ruling's word at :263 is *"caller"*, about the grid derivation. The conclusion holds on **line 385**, *"bind the store, slice 9"*. |
| **Revision 2: "five legs … the gate can go red"** | **INCOMPLETE.** The fifth leg is EMULATED x86-64, which S5-5 (`:375`) and `justfile:753-755` both call a smoke test, with a real machine owed as `D-TERRAIN-1`. §3.1 states it and L22 asks it. |
| **Revision 2: "the body facts ride the tag, snapped by the recipe"** | **UNSOUND.** `crates/physics` has no float fence and its drift gate is deferred (SPIKE-6a, `taxonomy.rs:21-24`); the recipe's own invariant (`body.rs:85-87`) forbids assembling a body from numbers computed elsewhere. Integers cross now (§4.2). |
| **Revision 2: the gravity snap "moves the relief bound by 0.02 % per step" and the obliquity cosine "by about a kilometre"** | **WRONG.** COMPUTED: 0.30 % and 8.2 km. The choices survive; the numbers did not, and the justification used the wrong ruler (a drawable floor against a zero-tolerance format). |
| **Revision 2: the weather as "coefficients over the body's own surface at a very coarse M"** | **UNPRICED, and it does not fit.** COMPUTED 393 KB per statement for a 50 km front against a 1 200-byte bag. The lawful road is seeded PARAMETERS, and the ask waits on U-L8. |
| **Revision 2: HR4 passed by naming `G-MAPPING-TABLE` and `G-MAPPING-ROUNDTRIP`** | **THOSE ARE THE SEAM'S GATES, not the landform layer's.** §3.16 now states four gates of the landform layer's own: G-MACRO-EDGE, G-MACRO-CORNER, G-MACRO-AREA and G-MACRO-SEAM. |
| **Revision 2: HR5 answered by the kernel test alone** | **INCOMPLETE.** The DRIVER lives in the Tier-A crate too. It is covered by solving a REAL 50 km body of THE world (384 nodes) inside the crate's own unit tests (§3.3). |
| **Revision 2: "a small body costs strictly less"** | **Now true under the metric rule, and it was not under the angular one.** But the cost rises as `R²` with no ceiling, so a ceiling is stated (L4). |
| **Revision 2: "eleven moons at a few hundred kilobytes each ≈ 60 MB"** | **UNSOURCED.** `slice_02_geometry_seam.md:275` records the counts, not the radii. Our Moon is 1 737 km. U-L13 owes the census. |
| **Revision 2: the weather has two layers** | **INCOMPLETE.** It has three. Without the ALMANAC the owner's word *"trajectory"* moves an annual mean by 1.5 % and nothing else moves at all (§3.14). |
| **Revision 2: "fields below" mapped to the biome** | **WRONG CLASS.** A field is cultivated, bounded and hedged. It is the same class as a road: built, live state, out of this arc. |

---

## 13. Refutation answers

Forty-eight findings from refutations A and B of revision 2. Every one is answered. **"FIXED" means the
section was rewritten, not annotated.** Where a refuter is wrong, the row says so with evidence. Where
the answer belongs to somebody else, the row says to whom.

### Refutation A

| # | Finding | Severity | Answer |
|---|---|---|---|
| A-1 | The octave replacement turns the shipped M-16 test red and stops the ladder coarsening above rung 9 | BLOCKER | **FIXED, and the refuter is right.** Re-checked in the code: `octaves_at` floors at one (`body.rs:251-261`), the home planet has 12 rungs, and `every_rung_sums_strictly_fewer_octaves_than_the_rung_below_it` (`body.rs:355-365`) fails at rung 10 with fewer than 12 octaves. **The octave table is untouched.** §3.4's row, §3.6 and §3.7 are rewritten: `Z` replaces the coarse octaves' CONTRIBUTION inside the same envelope, and DOMAIN 03's slope spectrum re-weights the same fourteen amplitudes. The 254 µs saving is deleted from §3.6 and §5. |
| A-2 | "A bound is a bound" is false: the draw still multiplies outside the bound, and the change is −53 % to +41 %, not 51 % | BLOCKER | **FIXED.** §3.4 states the draw's range as part of the rule: `draw ∈ [0.5, 1.0]`. §9 L3 asks it, §8 lists it as a door, and §6.3 lands it with the range named. The refuter's arithmetic is adopted. |
| A-3 | §6.1's quotation *"no store exists yet (the first is slice 9)"* does not exist in the ruling | DEFECT | **FIXED, and this is the finding the report is most grateful for.** Verified: `grep` finds no such line; `:263` reads *"no caller exists yet"*, about the grid derivation. §6.1 now cites **line 385**, *"bind the store, slice 9"*, and says plainly that a manufactured quotation is worse than the unsourced claim it replaced. §12 records it. |
| A-4 | The five-leg gate is presented as sufficient; S5-5 and `D-TERRAIN-1` already rule the x86-64 leg non-probative | DEFECT | **FIXED, and escalated to a decision.** §3.1's gate paragraph now quotes S5-5 (`:375`) and `justfile:753-755`, and §9 **L22** asks whether a real x86-64 machine blocks the erosion slice. **One clarification:** the aarch64-Linux leg is a genuinely different OS and libm from the host, and the debug/release/native legs catch optimisation-driven drift, so the existing legs can FIND a drift class — they cannot PROVE its absence on x86-64, which is the refuter's point and it stands. |
| A-5 | "4 268 columns" contradicts the source: the box holds 4 096 | DEFECT | **FIXED.** §2 and §3.6 state 4 096, cited to `slice_06_extractor.md:97,118,233`. The report no longer applies a CELL ratio to a COLUMN count. |
| A-6 | The per-octave rate divides a total where the two data points give the margin | DEFECT | **FIXED.** COMPUTED marginal rate 10.5 ns per octave per column with a 145 µs fixed cost, in §3.6 and §12. **The number is now moot**, because A-1 deletes the saving it was used for. |
| A-7 | The C1 read (16 taps) and the boot canary's "single macro cell" contradict each other; a pinned value is a stand-in; a kernel canary cannot see accumulation drift | DEFECT | **FIXED on all three.** §3.9 rule 1 pins **the REAL artifact's own nodes** — ESTIMATED under 400 bytes — beside `HOME_PLANET_RADIUS_BITS`. **One disagreement, stated:** under the SHIP road `Z` is an INPUT to the client's chunk arithmetic exactly as the look radius is, so pinning the real artifact's real nodes tests production, in the same sense the project already accepts for the radius. The refuter's other two points are adopted in full: rule 2 checks the whole artifact by its own digest at arrival, and rule 3 no longer calls the canary "stronger". |
| A-8 | The snap grids are justified perceptually where the law demands byte identity; the perturbation the snap must absorb is never named and is UNMEASURED | DEFECT | **FIXED, and it changed the road.** §4.2 replaces the snapped float with the BODY CHARTER: the PARENT quantises, integers cross, and no receiver rounds anything. **U-L18 / M-L10** measure the cross-host spread of every charter fact, which is the quantity the refuter names and nobody has ever measured. §12 records that the 1 000-ulp test measures the snap's tolerance, not the perturbation. |
| A-9 | The accretion bound is `0.004` renamed, and the document's own Vesta witness refutes it by nineteen times | DEFECT | **FIXED, and the refuter is right.** §3.4 sets the shape bound at **0.077 R**, cited to Vesta's observed relief, and COMPUTED shows the two bounds cross at about 939 km — so a large body is strength-limited and a small body is shape-limited. Escape velocity is added to the charter as the physical driver the refuter named. |
| A-10 | The cost model omits the priority flood's heap; 1.0 s for the flood alone | DEFECT | **FIXED, and it reversed the recommendation.** §5 is rebuilt with the heap at 6–8 misses per node and the routing re-run every `FLOOD_EVERY` passes: ESTIMATED 12–20 s, up to 40 s. §4.1 recommends SHIP, not derive, and §4.1 no longer writes a verdict before M-L1 runs. |
| A-11 | HR4 is passed by naming the seam's own gates, which do not test a drainage network | DEFECT | **FIXED.** §3.16 states four gates of the landform layer's own: **G-MACRO-EDGE, G-MACRO-CORNER, G-MACRO-AREA, G-MACRO-SEAM**, each of the `G-MAPPING-TABLE` kind ruling V6 A1 requires. |
| A-12 | The relief budget split — added, or taken from the octaves — is never stated, and it answers the owner's own worry | WEAKNESS | **FIXED, with a named mechanism and a named owner.** The relief is NOT added: `Z` replaces the coarse octaves inside the same envelope, and the SLOPE SPECTRUM moves amplitude into the 1–6 km band the eye reads (COMPUTED by DOMAIN 03: 47 m → 377 m at 3.13 km). §3.7, §6.3, and a picture gate at 8a that tests it before any solve exists. |
| A-13 | The macro grid is finest where it is least useful; the metric alternative is not in the options | WEAKNESS | **FIXED by adopting the refuter's alternative.** §4.3's rule is now a METRIC target (`MACRO_CELL_TARGET_M`, recommended 8 192 m) whose divisor of `N` gives `n_macro`. §4.3 shows the inversion the old rule produced. |
| A-14 | At `E = 9` the macro map puts about five samples in the owner's frame; recommending a number before the pictures is not deferring | WEAKNESS | **FIXED in part; KEPT in part, with the reason stated.** §4.3 and §6.9 now say plainly that one macro node is about 210 pixels wide in the reference frame and that **nothing the player looks at there is resolved by the solve** — the spectrum and the carve draw it. §6.5 moves the owner's "orienters" judgement to 8c. **KEPT:** a recommended target (8 224 m) stays, because DOMAIN 03 states one, because the owner asked for a recommendation, and because M-L1 and the 8c pictures both gate it. |
| A-15 | A spectral gap between the macro map and the octave table | WEAKNESS | **DISSOLVED by A-1's fix.** The octave table is not truncated, so no band is carried by neither layer. §3.7. |
| A-16 | No glacial process, in the one feature of the reference the owner pointed at | WEAKNESS | **FIXED, and promoted to a decision.** §1 gains glacier, cirque, arête, U-shaped trough and equilibrium line altitude; §6.4 lands the ice pass; §6.9 maps the skyline to it; **§9 L20** asks the owner. This is one of the two absences named in §0. |
| A-17 | "Asserted at every rung" describes a `#[cfg(test)]` sampled test, and a clamp that never binds on one body is not an exact bound | WEAKNESS | **FIXED.** §2 and §3.8 say the bound is exact by CONSTRUCTION, not by assertion, and name the assertions' `#[cfg(test)]` home. The clamp is replaced by DOMAIN 03's ENVELOPE RENORMALISATION, whose gate asserts `|Z| ≤ A_coarse` at EVERY node — exhaustive integer work, not a sample. |
| A-18 | HR5 covers the kernels and leaves the driver, which lives in the Tier-A crate | WEAKNESS | **FIXED**, with refuter B's own suggestion: §3.3 and §3.16 cover the DRIVER by solving a REAL 50 km body of THE world (`n_macro = 8`, 384 nodes) inside the crate's own unit tests. No variant, no test world, milliseconds. |
| A-19 | The SL6 ask L-4 states no data shape and no cost, and recommends yes | WEAKNESS | **FIXED.** §7 L-4 is now **ASK LATER, after U-L8**, and §3.14 prices the two candidate shapes: 393 KB per statement for a 50 km front, or 658 km storms at a coarse grid. §11 puts M-L11 before both L-4 and L17. |
| A-20 | The cell-pass figure 579 µs is superseded by 716 µs | NOTE | **FIXED.** §2 and §12 state 716 µs, cited to `slice_06_extractor.md:361`. |
| A-21 | Two different 12-byte records, one section apart | NOTE | **FIXED.** §3.8's first row separates them by name, and the macro NODE is now 4 bytes of height plus two state bytes (DOMAIN 03 §4.13), not 12. |
| A-22 | The flow accumulation's encoding and rounding are unstated | NOTE | **FIXED.** §1 and §3.1 rule 2 state it: the accumulation sums WHOLE SQUARE METRES in a `u64`, so there is no rounding order at all; the kept byte is a quantised logarithm of that exact sum, rounded once, at the end, deterministically. |
| A-23 | "A small body costs strictly less" is not strictly true | NOTE | **FIXED, and the claim's basis changed.** Under the metric rule the node count follows the body's surface area, so it IS ordered — and it rises as `R²` with no ceiling, which §3.10 states and §9 L4 asks a ceiling for. |
| A-24 | An airless body gets no craters | NOTE | **FIXED.** §0 item 8 and §6.4 land a crater field and impact relaxation for a body with no water, and ruling A5's corner gate applies to it too. |

### Refutation B

| # | Finding | Severity | Answer |
|---|---|---|---|
| B-1 | Stream power is not a streaming stage, and the flow routing must re-run; the model funds it twice | BLOCKER | **FIXED.** §3.1 gains **rule 8** (the re-routing period is a stated constant, part of the world tag), and §5 prices four floods and forty sweeps under DOMAIN 03's schedule. The pilot's example — water running up a shoulder — is the failure the rule prevents. |
| B-2 | The slice order erodes before the climate, so the erosion is rainfall-blind and no ice cuts anything | BLOCKER | **FIXED, and the plan's order changed.** §6.1 shows the loop, and the climate now runs INSIDE the solve's schedule every `CLIMATE_EVERY` passes — it is a closed form over 2.46 M nodes, ESTIMATED 30 ms, so it is affordable there. The stream power law is driven by DISCHARGE, not by area. 8d becomes what the ART KIT reads, not the climate model. The ice half is A-16 and L20. |
| B-3 | Road B imports SPIKE-6a's deferred bit-equality gate into the crate whose purpose is proven determinism | BLOCKER | **FIXED, by taking the refuter's own option (a).** §4.2 is rewritten as the BODY CHARTER: the parent quantises, integers cross, the recipe reads no float from an unfenced crate, and the home body's charter is PINNED in `home.rs` — the cure the tree already chose for the radius. §2 gains four rows of evidence, including that the CLIENT BINARY runs the forest's draw at boot even though the client LIBRARY does not link it. |
| B-4 | `relief = draw × min(…)` still exceeds the bound | DEFECT | **FIXED.** Same as A-2: the draw's range is now `[0.5, 1.0]`, stated in the rule, in the door table and in L3. |
| B-5 | The macro map cannot make the reference picture; the layer that can gets one sentence; and the "orienters" judgement happens before it lands | DEFECT | **FIXED.** §4.3 and §6.9 state the honest division of labour (one node is ~210 pixels wide in the frame; the spectrum and the carve draw everything in it), §6.5 moves the judgement to 8c, and §6.3 adds a **spectrum-only picture at 8a** so the owner's worry is tested at the cheapest possible point. |
| B-6 | The weather field's bytes are never computed; at the needed resolution one statement is ~393 KB | DEFECT | **FIXED.** §3.14 computes both ends and names the lawful third road (seeded parameters the client evaluates). §7 L-4 becomes ASK LATER; §11 puts M-L11 ahead of it and ahead of L17. |
| B-7 | The area weight uses a linear width where an area is needed, and points at the wrong place on the cube | DEFECT | **FIXED, and re-derived here independently.** COMPUTED with `W'(a) = k₁ + 3k₂a² + 5k₃a⁴` and the area density `W'(a)W'(b)/s³`: centre 1.000, **face-edge midpoint 0.702**, cube corner 0.758; and `W'` RISES from 0.785 to 1.558. §1's glossary row and §3.1 rule 7 both carry the table. U-L15 becomes gate G-MACRO-AREA. |
| B-8 | The priority flood is modelled as one random access per cell | DEFECT | **FIXED.** Same as A-10: §5 is rebuilt and the recommendation reversed. |
| B-9 | Several bodies cost TIME as well as memory, and the "few hundred kilobytes" moon has no source | DEFECT | **FIXED.** §3.10 states that the source records the counts and NOT the radii, that our Moon is 1 737 km, and that under a single-threaded solve the builds QUEUE. **The metric rule improves the arithmetic the refuter used:** COMPUTED, a 1 737 km moon gets `n_macro = 333` (which divides its `N = 2 727 936` exactly), 665 334 nodes and 2.66 MB — not a full-sized artifact. U-L13 owes the census. |
| B-10 | `E` is not the same kind of constant as `SHORT_WAVE_M`; the lawfulness argument does not hold | DEFECT | **FIXED, and the constant is replaced.** §4.3 withdraws the analogy by name and adopts a METRIC target, which IS the same kind of constant as `SHORT_WAVE_M`. The memory ceiling is named as a cost knob. |
| B-11 | The relief's split between the macro field and the octaves is never named | DEFECT | **FIXED.** Same as A-12: the slope spectrum, with COMPUTED numbers, an owner, a picture gate at 8a and a door row. |
| B-12 | No plate tectonics, no bimodal hypsometry: no shelf, no linear ranges, and an ocean fraction tuned by hand | DEFECT | **OWED to the owner and to DOMAIN 02, and named in §0 as one of the two absences.** §1 gains a crust-dichotomy row; §9 **L21** asks it and says plainly that L11's hand-tuning is the SYMPTOM of the missing mechanism, not its cure; U-L20 asks DOMAIN 02 to price it before 8b. This report cannot close it alone: the mechanism belongs to the planet-layout domain. |
| B-13 | Seasons are absent, so the owner's word "trajectory" is answered only nominally | DEFECT | **FIXED, and the refuter's arithmetic is adopted.** §3.14 adds the **ALMANAC** as a third layer — a closed form on the universe tick, shipped by the owning realm — and states that in an annual mean eccentricity moves the temperature by 1.5 % at `e = 0.17` and 0.014 % at Earth's. §7 L-6 asks the row; §6.7 lands it; §3.15 notes it is dormant-safe by construction. |
| B-14 | The rain shadow "from the macro map's gradient" is a local stripe; a continental interior needs an upwind march, which is another ordered pass and is not budgeted | DEFECT | **FIXED.** §1's glossary separates the two; §6.4 puts the climate's march on the COARSE lattice inside the solve's schedule; §3.6 carries DOMAIN 05's MEASURED refusal of the per-column march (5.1 ms of an 8 ms budget) and its 0.15 ms grid read; §6.6's gate now measures a DISCHARGE difference, not only a colour. |
| B-15 | The boot canary's re-basing narrows what the login gate proves, and the report calls it strictly stronger | DEFECT | **FIXED, and the refuter's fourth option is taken further.** §3.9 rule 1 pins the REAL artifact's own nodes (not a stated slice of a value the world never uses), rule 2 checks the WHOLE artifact by its digest at arrival, and rule 3 states what the kernel canary does not catch instead of claiming it is stronger. |
| B-16 | HR5: nothing covers the whole-body driver, and a small real body would | WEAKNESS | **FIXED, using the refuter's own observation.** §3.3 and §3.16: a REAL 50 km body of THE world, 384 nodes, in the crate's own unit tests. |
| B-17 | `BodyDefinition` is `Copy`, so a resident artifact cannot live in it | WEAKNESS | **FIXED.** §2 and §3.8 carry the row, and §6.4's landing list names the change. |
| B-18 | The home planet's pinned literals grow from two to about thirteen, and the three missing draw laws become a heavily tuned door | WEAKNESS | **FIXED.** §4.2 states the pinning and its cost; §8 adds *"the spin, obliquity and water-inventory DRAW LAWS"* as a door with the instruction to write and tune them before slice 9; §9 L7 carries it. |
| B-19 | The gravity snap's justification is off by 15×, and on the home planet the bound it justifies does not bind | WEAKNESS | **FIXED.** §4.2 states 0.30 % per step and says the strength bound is inverse in `g`. §3.4's table shows which bound binds on which body, so the reader can see where gravity matters. |
| B-20 | The obliquity snap's justification is about 8× low | WEAKNESS | **FIXED.** COMPUTED 8.2 km, in §4.2's table. The choice survives; the number did not. |
| B-21 | The sample box holds 4 096 columns | WEAKNESS | **FIXED.** Same as A-5. |
| B-22 | The 8 ms table omits five named per-column costs | WEAKNESS | **FIXED, and it changed the verdict.** §3.6's table now carries all seven terms from DOMAIN 03 §7.4 and DOMAIN 05 §5.7, and states that the cave-dense chunk that also holds a meshed water sheet is **ESTIMATED 8.46 ms against a budget of 8 ms — over budget** — with the levers named. Revision 2 claimed 1.29 ms of margin. |
| B-23 | "Fields" are cultural, like roads | NOTE | **FIXED.** §3.13 and §6.9 move fields out of the arc, beside roads. |
| B-24 | A static snow line is a PERMANENT snow line, and the owner will compare | NOTE | **FIXED.** §6.6 states it to the owner and names the almanac as what moves it seasonally. |

### Checked and sound, by both refuters

**KEPT, unchanged.** Both refuters re-derived these from the code and could not break them, and this
report leans on them: the whole ladder re-derivation and every row of its body table; the tiling proof
(`n / n_macro` exact); §3.9's caller set (`world_identity` has exactly two production callers); the
stratigraphic finding (`StrataTable::at` reads a DEPTH, so no band can outcrop, so no caprock, no mesa,
no pillar); the float fence's contents and rules 1, 4, 5 and 6 derived from it; the seam and corner rule;
the strength-bound arithmetic and the small-body deletion (`Ladder::for_radius` returns `None` when
`crust >= surface`); the registry facts (one cited term, and the digest folds every physical column);
`look.rs`'s own law and the withdrawal of L-2; the five legs; the reach arithmetic; `POLE_AXIS` staying
put and §8's door being deleted; the SL1 clause-4 premise (`insolation_rel` inverts to the orbit's size,
not to a placement); refusing ore in rivers; the static landscape; and §12 as an honest ledger, which
both refuters called the report's strongest section and which revision 3 has again extended.

