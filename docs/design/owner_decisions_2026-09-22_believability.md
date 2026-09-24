# Owner decisions, 2026-09-22 — believability before 8e, and the gates that define it

Written after the owner flew the home planet at 100–150 km (three screenshots: lakes in every hollow,
scratches along the grid, plains that read as dunes) and asked whether to proceed with the whole of
slice 8 or to stop and reach believability first. Newest ruling file; it wins over every earlier design
document. It reads with `owner_decisions_2026-09-21_water.md` (W1–W11) and the research report
`docs/investigation/2026-09-22/lakes_and_landscape_models.md`.

## B1. STOP AND REACH BELIEVABILITY INSIDE 8d, BEFORE 8e STARTS (owner: *"Agreed, add the ruling and the gates"*)

The terrain arc does not march on to 8e, 8f and 8o while the solved field is not believable. Every later
slice reads this field: the biomes of 8e read the water, the slope and the soil the solve leaves; the ocean
of 8o reads the bathymetry it leaves; the block store of 9 stores edits against this ground; the freeze of
14 locks it. A fault in the field is repaired once now, and against every player's edits later. Paint hides
nothing: the faults the owner saw are the model's, not the look's.

## B2. THE ORDER OF WORK, ALL INSIDE 8d

1. The lakes from a water budget (recommendation 1 of the research; W11): the flood fill is a scratch
   surface; a lake is water that stands, from Fill-Spill-Merge over the depression hierarchy with a
   published closed-basin balance; a hollow the rain cannot fill is a dry basin.
2. The summer ice line with the ice run inside the passes (recommendation 2), and the crater retention
   age on a wet body (recommendation 4).
3. The flat receivers ranked by Euclidean distance (recommendation 3), then the scratch measurement on a
   cone and by rotation. If the grid still shows, the hexagonal mesh is discussed as a new world (SL5).
4. The river lines with the width law, the carve and the floodplain — at the coarse nodes first, then
   into the fine rungs. This is the cure for the dunes: the fine relief under 8 km is noise today, and
   noise reads as dunes from the air because nothing drains it; real ground at that scale is shaped by
   drainage — ridges and valleys that join.
5. The lake side word in the coast mask (recommendation 5), so the far view draws what the model has.

Each step ends with the census's numbers and ONE picture on each of the three stands: the belt, the
coast, and a plain. 8e starts when the owner says the three stands look like a planet.

## B3. THE GATES — BELIEVABILITY AS NUMBERS, WITH A CALIBRATION BODY EACH

Every gate is a line of `lake_census` (or a sibling instrument) and a red on any of them stops the arc.
The numbers are Earth's, from the research report's §1 and §6; a gate is never tuned to pass.

| gate | the number | the body and the source |
|---|---|---|
| G-LAKE-SHARE | lakes ≤ 4 % of the land; only where the budget fills a hollow | Earth: 3.7 % of the non-glaciated land (Verpoorter et al. 2014) |
| G-LAKE-COUNT | lakes wider than one node of order 10³, not 10⁴; sizes on a power law | HydroLAKES (Messager et al. 2016); Cael & Seekell 2017 |
| G-BASIN | every depression ends dry, partial or spilling by its water supply; the inventory balances (sea + lakes = the charter's water) | Barnes, Callaghan & Wickert 2021 |
| G-ICE | the ice covers about 10 % of the land on an Earth-like charter (about 30 % at a glacial maximum) | NSIDC; Egholm et al. 2009 |
| G-BUZZSAW | no peak stands more than about 1 500 m above its local snowline | Egholm et al. 2009 |
| G-CRATER | the crater count within an order of magnitude of 190 on an Earth-like charter; the airless moon unchanged | Earth Impact Database; Osinski et al. 2022 |
| G-GRID | the long-axis histogram of the lakes and valleys is flat against the grid's directions; Hyväluoma's rotation test scores near one at every angle | Hyväluoma 2017; Tarboton 1997 |
| G-HYPSO | the land's height distribution shows Earth's two humps and Earth's share above 1 000 m | Earth's hypsometric curve (the solve's own G-AGE integrals) |
| G-RIVER | every river reaches the sea or a dry basin; a valley's floor descends monotonically along its trunk | the receiver tree's own reading |
| G-LOOK | the owner's look on the three stands: the belt, the coast, a plain | the owner |

## B4. WHAT THIS DOES NOT CHANGE

The laws already ruled stand: T9 (no drawn physical number), the freeboard law (W8), the shore laws (W6,
W10), the rock map (W4/W5), the one generator on two hosts (SL10). The block store, the ocean and the
freeze keep their order after the arc. The coverage gate (HR5) is still owed for everything since the
last pass and runs when the owner's window is closed.

**Example.** After step 4 a pilot over the belt sees valleys that join into a river, the river reaching the
sea through a gorge, and hills between the valleys that are ridges, not ripples. After step 2 the snow lies
only on the range's top. After step 1 the plain has no lake, because the rain never filled its hollows; the
wet north has one lake per valley, up to its outlet.

## B5. THE HOME PLANET'S TILT STAYS; THE ICE GATE TESTS THE LAW, NOT THE WORLD (owner: *"Let's wait till 8e lands and then we can decide"*)

After B2 step 2 the ice covers 4.06 % of the home planet's land against Earth's ~10 %, and the cause is the
world's own tilt of 55°, a seed identity choice (T9): the polar summer is the planet's hottest place, so no
cap lasts the year. A hand-set tilt for the home planet would be a seam — one world edited by hand — and is
REFUSED. The world is what the seed made; only the laws are adjusted, for every world at once.

Therefore: (1) G-ICE moves to a unit test that hands the ice law an Earth-like charter (tilt 23.4°, Earth's
insolation) and demands about 10 % of the land under ice; the home planet's own share is a READING, not a
gate. (2) The decision whether to want caps on the home planet waits for 8e, whose two-column seasonal
balance (a continental interior with its own hotter summer and colder winter) may put winter ice on the
high interiors under a 55° tilt on its own. (3) If caps are wanted after 8e, the lawful road is the seed
search choosing another home by a tilt bound — a reset of every stand, berth and pin — never an edit.

**Example.** On this planet a pilot flying north from the belt meets long summers of melt and long dark
winters of frost, and never a year-round cap. On an Earth-tilted world the same flight meets the ice where
the summer stops melting it. Both are the one law reading two worlds.

## B6. THE PLAN OF 2026-09-24, THE REFERENCE AS THE VISUAL TARGET, AND THE LIVING STATUS FILE (owner: *"In general I agree with your proposals, but please make sure that visually we are moving towards this target picture"*)

Written after the owner flew the home planet again and named three faults: *"the land look like dunes everywhere"*,
*"shores are squared like teeth"*, *"water positions for lake and rivers is not believable"*. A read-only workflow
(`wf_1780e367-7e9`, nine agents, one skeptic per fault) traced all three to one root: the solve knows the ground only
on nodes 8 192 m apart, and under the node the ground is noise and the water is a whole node square. The findings,
with their lines of code, are in `docs/investigation/2026-09-24/terrain_believability_status.md`.

1. **THE PLAN, AGREED IN GENERAL.** Inside 8d, in this order: (1) measure first, with no change to the world, and
   add the census lines that can see the dunes and the teeth; (2) repair the lake budget — each hollow fed from its
   whole watershed, a full hollow's overflow passed on, the discharge re-accumulated after the final route, a small
   lake kept at pyramid level 1; (3) the survey's R2 — one eroded level at 1 024 m shipped in the artifact, with the
   height retargeting and the multi-scale breach, calibrated on Earth through Hack's law — and R4's second half, the
   noise only below the solved level; (4) one shoreline from one field on the R2 ground, the lake's edge under W6's
   law; (5) G-GRID again, and the D8 decision only if it is still red; (6) the lakes' outlets, the deltas, and R3;
   (7) G-LOOK. The numbers each step states (the level, the bytes, the version, the bake) are the recommendation the
   owner agreed to in general; a step that must depart from one asks first.
2. **THE REFERENCE PICTURE IS THE VISUAL TARGET.** Every step's report ends with: the three stands' pictures (the
   belt, the coast, a plain); one picture on a FOURTH stand framed like the reference (an eye ~400 m over its local
   ground near the belt's range, looking across valleys toward the range 30–60 km away, ~55° of view, a low sun
   across the view); the skyline numbers on that stand by the rule of `11_reference_trace.md`; and the reference
   checklist (§9 of the status file) brought up to date. A step that turns a gate green but moves that picture AWAY
   from the reference says so. The picture itself stays outside the repo (the repo is public and it is a photograph
   of a commercial game); `01_reference_target.md` §2 and `11_reference_trace.md` hold it as numbers.
3. **THE DRESSING IS REGISTERED, NOT FORGOTTEN** (owner: *"make sure that we don't forget about those"*). What the
   reference shows and this arc does not give — the snow on the tops, the ground's colour and the beach, the forest
   as a mass, the cliffs and mesas and bedded faces, the long far shadows, the water's look, the clouds — is entry
   D-TERRAIN-8 in `docs/design/DEFERRED.md`, each with the slice that owes it. They start after B1's word, as B1
   says; the far shadow's timing is still the owner's question (a depth cue that shows the shape, on the `perf`
   tree's side).
4. **THE LIVING STATUS FILE** (owner: *"Make sure that we update it with the changes we did, so next read will not
   mislead. We might also remove it after all is fixed and done"*). `docs/investigation/2026-09-24/
   terrain_believability_status.md` is updated IN THE SAME CHANGE as any work on its items: the state flipped, each
   moved number corrected, one line in its change log. When the code and the file disagree, the file is the defect.
   It is deleted when every step of item 1 is done and the owner gave G-LOOK; the rulings and DEFERRED keep the
   history.

**Example.** After step 3 a pilot on the reference stand looks across the belt and sees ranked ridges with valleys
that join, and the skyline count at 0.25° rises from zero toward the reference's 2.5–5.5 per 60°. The report shows
that picture beside the reference, the new numbers, and the checklist's rows that moved; the status file's plan row
3 turns ✅ with its commit.

### B6.5 THE OWNER'S TWO ANSWERS (2026-09-24)

1. **The bench is busy** (owner: *"No, it's testing at the moment"*). The `perf` tree measures now, so this tree
   runs no build, no test and no instrument until it is idle (one measuring bench). Step 1's instruments are
   WRITTEN meanwhile and built and run when the bench is free.
2. **The pictures are judged with long shadows** (owner: *"Ideally with long shadows"*). The far shadow moves out
   of the dressing's "after G-LOOK" into a PREREQUISITE of the judged pictures: the ground at 100–150 km casts its
   shadow kilometres long, because a shadow shows the shape and paints nothing over it. It is the `perf` tree's
   work (the drawing side). Until it lands, a picture is still taken at each step and is marked "no far shadow";
   the reference stand's picture is taken again when the shadow lands.

**Example.** At step 1 the reference stand's "before" picture shows the belt in soft light only, and its caption
says "no far shadow". When the far shadow lands, the same stand under the same low sun shows each ridge's shadow
across the next valley, and that picture becomes the "before" the owner judges.

**B6.5 AMENDED, the same day** (owner: *"We can go without long shadows for now"*). The far shadow is NOT a
prerequisite of the judged pictures. The owner judges the shape as it is drawn today, and each picture's caption still
says "no far shadow". The far shadow stays registered in D-TERRAIN-8 (the dressing), wanted, for the `perf` tree when it
has room; it is not dropped.
