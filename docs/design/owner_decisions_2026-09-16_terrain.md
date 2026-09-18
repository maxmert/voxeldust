# Owner decisions, 2026-09-16 — the terrain arc before slice 9, the ocean, and time

Written after Step 17 (`601a1e1`), at the owner's terrain discussion. Newest ruling file; it wins
over every earlier design document, including `docs/investigation/2026-09-08/landforms/*` and the
V13 order in `owner_decisions_2026-09-07_voxels.md`.

The reference for an earth-like planet, and for the home planet in particular, is the owner's
`crimson desert.jpeg` (read as numbers in `docs/investigation/2026-09-08/landforms/01_reference_target.md`
§2). The owner: *"not the whole planet should look like on the picture — of course we should have
different weather zones; depending on all factors there should be different terrain, flora and
fauna."* The reference is ONE zone of ONE planet, produced by the chain charter → climate → biome →
column; it is never the planet's look.

## T1. THE WHOLE OF 8 BEFORE 9 (owner: *"we need to work on the full 8 before 9; move all the topics before that"*)

**RULED.** The terrain arc lands whole before the block store. The order:

1. **M1b** — the measurement: the skyline march on 8a's proposed layer heights, the predicted break
   count published BEFORE 8a starts. One afternoon, no code. If the count does not move, 8a does not
   start and the arc is re-thought.
2. **8a the spectrum** — the slope spectrum anchored on the angle of repose; the per-column
   downward-only roughness factor; the ridged, flow-aligned middle band; the cap-rock bench; the relief
   law and the sea. Judged by the owner in the pictures.
3. **8b the charter** — the missing draws (spin under tidal locking, the damped tilt, the water
   inventory, the optical depth, the elastic thickness) and the orbit half stated by the star system
   to its planet one hop down (A1, A2, approved in V13).
4. **8s the sky and the shadow** — aerial perspective from the charter's scale height, a sky dome, a
   cascaded shadow map. ★ MOVED INTO 8 by the owner's decision (T4): it is the same exception the light
   (8L) got — without it no vista can be judged.
5. **8c the solve** — the macro layout (plates, isostasy, orogeny), the drainage, the priority flood,
   discharge and stream power, isostasy, talus whose angle reads the aridity, ice, craters, the coast;
   the climate inside the schedule (insolation, the lapse rate, the Hadley edge, orographic lift, the
   rain shadow). Once per body; the artifact kept.
6. **8d water and rock** — rivers, lakes, the clipped water sheet as a REAL SURFACE at the sea radius,
   the channel and floodplain carve with its two-rung fade, the stratigraphic column at a fixed radius,
   the rock map, and THE AUTHORED OVERRIDE (a stored layer of authored shape changes over the seed's
   shape, crossing as a diff).
7. **8e the biome and the soil** — the Whittaker classifier at a derived climate rung, nineteen biomes
   in five bits, the soil law, the snow line and the tree line as temperatures, sea ice; the paint
   table as the first step (the seam for textures).
8. **8f the third dimension** — the 3-D removal term in a thin band at the surface: arches, overhangs,
   pillars, cave mouths in cliffs; placed rock objects.
9. **8o the ocean** — see T2.
10. **18's state model** — the almanac derived from the orbit (no new lane) and the weather as a list
    of systems on its own lane (T3). Nothing drawn yet beyond what 8s needs.
11. Then **9, 10, 11** (the block store, the diff lane, the collider — the collider is built ONCE on
    the final carved shape), then **14 the freeze**, then the look that remains (paint and textures,
    clouds and rain on screen, the forest, the character).

**The dependency that flips.** V13 put 8c and 8d after the collider "so the collider agrees with the
shape". With the shape first the collider agrees with it from its first day and never follows a
change. The cost is time: the arc is estimated at 22–28 weeks; 9, 10 and 11 move behind it. During
that time the ground may move freely — nothing is built on it — which is the freedom the freeze takes
away at 14.

**The eight-millisecond budget (L25).** MEASURED on the walk's path at Step 17: one rung-1 chunk
builds in 111–112 ms, fourteen times the budget, before any roughness. 8a makes such chunks common.
8a measures its densest chunk FIRST, and the card's stand-down gets a second trigger — a chunk over
budget — judged on the same flights. The budget itself stays the owner's call when 8a's numbers are in.

**The reference as a number.** The picture's "8–12 silhouette breaks per 60°" was counted by eye; our
zero was computed by a rule. Before 8a's gate is set, the reference's skyline is traced once by hand
and the same break rule run on it, so 8a has a calibrated band, not a ratchet.

**The pictures.** The five frozen references are red since the ladder extension (Step 17) and 8a moves
them again. The owner looks ONCE after 8a; the far stand's candidate is judged then too.

**The cave in the crossfade (from Step 17).** A feature fades with the rung whose cells match its size,
so a cave mouth closes only when it is under a pixel; the fixtures bound a vertex by the field's WHOLE
sink, caves included. One picture stand with a cave in frame joins 8a's protocol.

## T2. THE OCEAN — added later, with the seams left now (owner: *"I'd really like to give an ability to build real ships and sail"*)

**RULED.** The ocean is its own piece inside 8, **8o**, after 8d, with its own discussion. A sailing
ship is a hull realm inside the planet, like a hull on the ground or in orbit: one machinery, one
shipyard, the same crossings. What 8 must hold so the ocean stays open:

- the sea level and the water inventory in the CHARTER, stored (8b);
- the bathymetry KEPT in the solve's artifact, never discarded as "under water" (8c);
- the water sheet in 8d drawn as a real surface at the sea radius with the coast clipped, not a colour;
- 8o: the surface as a SHAPE (a sphere at the sea radius: a floor for a walker, a fluid for a hull),
  and the TIDES from the almanac (the moon's position is closed form from the seed and the tick);
- buoyancy and the water's drag are slice 11's per-realm ambient forces — the hull already states its
  mass, cross-section and drag coefficient on change (the movement contract), so buoyancy needs no new
  datum; a sail is a block that turns the weather's wind into thrust (the block and signal slices);
- waves: as look, a shader on the delivered tick after the freeze; as physics, a closed-form height
  field from the seed, the address and the tick, which both hosts compute (a per-realm physics
  function like an orbit — never part of the static shape);
- currents: NOT planned (they are the ocean heat transport the climate model lacks); priced before
  promised.

**Example.** A harbour town's quay stands on a coast the solve made, with 8 m of water beside it and a
shelf falling to 6 000 m offshore. A ship built at that quay floats by the planet's physics, feels the
storm's wind on its sails, and crosses into the next realm exactly as a space hull does.

## T3. TIME (owner: *"we don't need to implement time for the terrain (no erosion). Agree with your proposal of weather systems."*)

**RULED.** The static shape is a function of the seed, the address and the charter, NEVER of time
(SL10). Everything that moves falls into three classes:

1. **Closed form from the universe tick** — both hosts compute it from the seed and the tick; no
   state, no lane. The sun, the seasons, the day, the moon, the tides, and THE ALMANAC: the snow line
   this month, a river's stage in the melt. Example: the client draws the river 2 m higher in the melt
   because the almanac says so for this tick, and the server puts the walker's feet at the same level.
2. **Live state stepped by the shard, shipped on a lane** — THE WEATHER: a list of weather systems
   (position, size, strength), about 4 ms a step per awake planet, shipped as a small list on its own
   lane. Clouds, rain, wind and their shadows come from that list on the client; the wind reaches a
   hull as a force in the parent's physics through the movement contract. Dormant planets advance their
   weather by the dormant-world rule, not by stepping.
3. **Never** — the land's shape in time. NO EROSION RUNS IN THE GAME. After the freeze the only shape
   change is the authored override (8d), a stored layer that an author or a tool writes, crossing as a
   diff like any edit. Example: a meteor strike is a placed change written by a tool, never a
   computation the recipe re-runs.

A river's visible FLOW is look in class 1: the water's surface moves with the tick, deterministic on
every client; its channel and its stage are the shape and the almanac.

## T4. THE LOOK LINE, AMENDED

The V13 addendum ("foundation first; the look after the freeze") stands, with two exceptions the owner
funds because they make the shape judgeable: the light (8L, landed) and the sky (8s, T1 item 4). The
paint table lands in 8e as the seam; textures, clouds and rain on screen, the forest (14) and the
character (16) stay after the freeze.

## T5. WHAT THE ARC DOES NOT DELIVER (restated, so nobody is surprised)

No fields, roads or walls (live state); no moving glacier; no meander on a great river; no volcanic,
aeolian or karst family until priced (L21); no live landscape change (T3); no cold coastal upwelling,
no ice-albedo feedback, no ocean heat transport and no monsoon (the climate model refuses them by
construction); fauna is live state in a shard and belongs after the physics and block slices, with the
biome field as its habitat map.

## Status

Step 17 (`601a1e1`) is the base. Owed from it: the hull legs under the reveal rule; the owner's look at
the moved pictures; the third-person boarding swing (the 60 m boom releases 4 211 chunks at the swap;
the flights fly the pilot view); the pre-merge gate before a merge. M1b is the next step.

## T6. THE 8a ASKS, ANSWERED (owner, 2026-09-16, on `slice_8a_design.md` §7)

| ask | the owner | what it means for the build |
|---|---|---|
| 1 the relief law | *"agree"* — it moves to 8b | 8a's spectrum does not read it; 8b's charter (gravity, density) caps the relief |
| 2 the sea's interim | *"not based on the percentage, but based on the physical laws or some models; let's see what numbers it will produce"* | NO interim ocean fraction. The sea level comes from a WATER INVENTORY computed by a physical model in the charter (8b: the body's mass, its volatiles from formation, insolation and escape → a water volume), then the level by bisection of the volume over the shape. 8b's water inventory therefore lands BEFORE 8a's pictures are judged; the bisection mechanism is built once, on the volume. The numbers it produces are reported, and the owner judges them. |
| 3 the top rungs | *"let's try that and change if quality will not be good or we can improve"* | keep ONE aliased octave at the rungs past the table, the ratio printed, 8c named as the fix; re-judged in the far picture |
| 4 the frozen patch's pair | *"agree"* | at the freeze (14) F1 stores (the recipe's version, the octave count) |
| 5 the skyline gate | *"I would also not use any limiting number, but rely on physics and models more and see how that will unfold"* | P1 is a REPORT, not a band: the march prints the counts at every threshold and the largest rise beside the reference trace (`11_reference_trace.md`), for the owner's eye with the pictures. No floor, no target, no refusal. The frozen pictures stay the regression guard. |

**The order, refined by ask 2:** 8a stages 1–5 (the spectrum, the ridge, the roughness placeholder, the bench, the wavelength rule) → 8b's charter draws (the relief law, the water inventory by a physical model, the orbit half A1/A2, the remaining draws) → the sea by bisection of the volume → the version bump, the pins, the pictures, the cave stand (the owner looks once).

**The chunk budget (L25), first reading (2026-09-16, `12_chunk_budget_base.md` when published):** the 111–112 ms chunk on the walk's path spends ~80 % of its time on PARENT MESHES (the eight coarse chunks its morph targets read), not on the cell field or the extraction (~6 ms together). If the quiet-machine run confirms it, the dense chunk is a lookup cost, not a roughness cost, and the budget question changes shape.

## T7. THE RUNG DISAGREEMENT UNDER THE RIDGE — three derived rules (owner, 2026-09-17: *"Agree with all 3 ... all 3 will increase the visual quality and strengthen the existing machinery"*)

MEASURED after 8a stage 2: the rung disagreement's worst p99 is 1.54 px against the 1 px line (0.43 before 8a, 0.98 after the spectrum), at the rung pair that drops the crest octave (3 125 m) — a crest has a kink, and a kink needs finer sampling than a round wave. **RULED**, as one step between stage 3 and stage 4, all three as DERIVED rules with no tuned number, judged by the same measurement (worst p99 ≤ 1 px on every rung pair):

1. **The sampling rule per octave kind.** A smooth octave survives on a rung while its wavelength covers the stated cells (49 today); a RIDGED octave needs twice that, from the kink's harmonics. It survives one rung longer and is dropped where the crest stands under a pixel.
2. **The crossfade width from the residual disagreement.** For each rung pair the ladder bounds the dropped octave's disagreement in pixels at the switch distance; where the bound exceeds one pixel the band widens by that ratio. Inert after rule 1; wakes by itself if a later term (the terrace, the macro field) pushes a pair over.
3. **The switch distance floor from the same bound.** A rung's switch distance is the larger of "one cell one pixel" (today's rule) and "the dropped octave's disagreement under one pixel". Inert after rule 1; protects the picture where the band alone would stretch too far.

The pixel is the tolerance; it is physical (a crossfade step over a pixel is a seam), never a taste number.

## T8. THE DROWNING — the sea's mechanism now, the pictures judged dry (owner, 2026-09-18: *"(a)"*)

MEASURED (8b stage 5): the physical water model gives the home planet 2 735 928 089 km³, two Earth
oceans, and on today's one-humped ground that is a 98–99.98 % sea; the control (Earth's own ocean on
the same shape: 96 %) clears the water model — the missing term is the crustal dichotomy, 8c's
two-crust isostasy. **RULED (a):** 8b stage 6 builds the sea's MECHANISM (the bisection over the
shape against `water_km3`, the level stored as `sea_offset_mm` in the charter, the liquid rule for
an ice world), and 8a's pictures are judged WITH THE SEA OFF — the dry shape — until 8c gives the
ground its second hump; the coast is judged after 8c. A picture with a sea level nothing draws is a
lie, so the dry judgement is stated as dry.

Also on record from stages 3–5, OPEN: the thermostat (ask 2, ASSUMED yes), the crust density ratio
(ask 3, ASSUMED), the gas giant's ground (the relief law read on a body with no solid surface:
2 168 km — the natural rule is no ground where the charter clears the solid-surface flag), and the
home planet's surface pressure, which DREW 1 239 Pa from its stated band (ask 6) — a derivation
from the inventory and the escape physics would replace the draw.

## T9. ★ NO PHYSICAL NUMBER IS DRAWN — EVERY PHYSICAL FACT IS COMPUTED (owner, 2026-09-18: *"we don't randomly generate any physic numbers or physic laws — always calculate"*)

**A STANDING RULE, the same weight as the hard rules.** A physical fact of a body — its pressure, its
temperature, its water, its spin, its crust, its relief cap — is COMPUTED from the body's other facts
by a published law with a stated calibration body, never DRAWN from a band. A draw is allowed only
for what physics does not decide from the facts in hand: the seed's own identity choices (which
noise, which face, where a hill stands), a history the facts cannot recover (a tilt's prior, an
impact), or a stated scatter AROUND a computed value in a stated band, named as such. A band that
stands in for a missing law is a defect: the law is owed.

**The first application: the surface pressure.** MEASURED at 8b stage 4: the pressure was drawn
(the design's ask 6, a log-uniform band over four decades) and the home planet drew 1 239 Pa — a
Mars-thin air under which its two oceans boil (the boiling point at that pressure is 279 K against
a 288 K surface). Not a wrong pick: a draw with no cause. **RULED:** the pressure is DERIVED —
a rocky body's outgassed air scales with its mass (one calibration body, Earth), the pressure is
that air's weight over the surface (`p = g·M_air/(4πR²)`), retained by the census's own shoreline
verdict: `p = 1 bar · (M/M⊕) · (g/g⊕) · (R⊕/R)²`, plus the thermostat's CO₂ partial pressure when
that law is written in the same terms. The home planet, Earth's twin in mass, gravity and radius,
reads about 1 bar BY CONSTRUCTION, not by choice. The scatter of real planets around the law
(Venus, Mars) is NOT modelled; if ever wanted it is a stated modulation around the derived value.

**The loop closed:** once the pressure is a fact, the census's earth-like predicate gains the
liquid-water clause (the surface temperature and the pressure against the triple point and the
boiling curve), so a home planet is earth-like WITH A SEA by construction.

**Audit owed:** every other draw the charter or the forest makes on a physical quantity is listed
and either derived or named as history/scatter with its reason (the tilt's prior is history; the
planet masses are the forest's identity draws; the ocean fraction was refused already by T6 ask 2).
