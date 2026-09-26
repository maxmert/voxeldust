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

## B7. STEP 2'S TWO WORDS, AND TWO TRACKS (2026-09-24; owner: *"I agree with your 2 recommendations, if it will lead us to believable terrain. Agree with 2 tracks"*)

1. **An inland pocket is a closed basin, not the sea.** The census (step 1) found 148 pockets under the sea's level with
   no path to the ocean — 3.2 million km², the largest 1.35 million km² and 11.8 km deep — each drawn as sea at the
   sea's level. From step 2 on, each is a closed basin like any other: its water from its budget (a lake at its own
   level where the rain fills it, a partial lake or a dry basin where it does not), and the sea's level is solved over
   the CONNECTED ocean only. One law, no special case (T9).
2. **The discharge row carries the RUNOFF**, accumulated over the final receivers — the water that reaches a river,
   the budget's own water, and the row R2 seeds its rivers from. The rain would make a desert river as large as a wet
   valley's.
3. **Two tracks at once:** step 2 is built; R2's detailed design is written meanwhile (read-only work, no load on the
   bench), so step 3 starts from a reviewed design the day step 2 lands.

**Example.** A hollow at the foot of the belt's wet range, fed by the range's own receivers, fills to its sill and
spills a river toward the sea; a pocket 2 km under the sea's level in a dry interior stands as a salt pan with a small
lake at its bottom, at its own level, never as a square of ocean.

### B7 BUILT (2026-09-25; plan step 2; NOT committed — the owner's word owed)

`GENERATOR_VERSION` 20, `ARTIFACT_VERSION` 8 (`PROTO_MINOR` unchanged). The census lines are in
`docs/investigation/2026-09-24/step2_census_gen20.txt`; the living status file §2, §4, §6, §7 holds the detail.

1. **The inland pocket is a closed basin (item 1).** The ocean is the largest connected water under the sea's level
   (`land::ocean_mask`); the sea is solved over it alone at the first land, the freeboard, the running sea, the final sea
   and the lakes' fixed point (`land::sea_level_connected`); only the ocean seeds the flood, carries the sea's load and
   facies, and warms the climate's air. An inland dry column under the sea's level reads `SIDE_INLAND`, and the shore law
   leaves it alone. MEASURED: 0 pocket nodes carry the sea's bit (59 470 were drawn as sea); 236 pockets, 3.3 million
   km², by their budget DRY 181, PARTIAL 45, SPILLING 10; the sea 3 591 m (3 566 m).
2. **The discharge row carries the runoff (item 2)**, accumulated over the receivers of the last routing. MEASURED:
   W10-DISCHARGE-TREE 0 (1 304 002).
3. **The budget of W11 is Fill–Spill–Merge's** (B6 item 1 step 2): each hollow takes its whole watershed down the ground,
   and a full hollow's overflow runs on to the sibling, the parent's ring or the sea; nothing is lost (the books exact).
   MEASURED: lakes 3.51 % of the land (0.068 %), 3 883 patches (506), bodies DRY 9 599 / PARTIAL 1 549 / SPILLING 5 839
   (19 511 / 506 / 0). The small lake keeps its level at pyramid level 1: 0 lakes change at the rung 9 → 10 swap (427 of
   506), 0 lip columns (3 753).
4. **What it exposed, MEASURED, for the owner's word:** G-RIVER-DESCENT RED — 3 430 trunk steps rise inside filled
   hollows, because the shipped receivers follow the routing fill across a parent's dry ring while the budget pours the
   water into the sibling (proposal: receivers that follow the budget's water, before step 3); G-GRID worse, 1.484
   against 1.284 (1.306 outside the filled hollows: the new flats over the pockets lean on the grid); G-RUNG-HOLD RED,
   19 columns at 9 → 10 up to 6 090 m (plan risk R7, question 5); at the level swaps 14 → 15 and 15 → 16, 3 698 and 558
   lake columns fold away under the wet-majority rule of W15, one pixel each at the swap distance.
5. **The review of step 2 (2026-09-25; NOT committed):** a pocket's dry floor is its own coast-mask word (word 2), so no
   floor is lifted into a mesa at any rung — MEASURED G-POCKET-HELD 0 of 60 383 columns, G-RUNG-HOLD 5 columns at
   9 → 10 up to 731 m (was 19 up to 6 090 m); a far chunk waits for the coast mask; a floor under the sea's level is no
   beach. STILL FOR THE OWNER'S WORD: the discharge row carries to the sea the 1 607 km³/yr the budget holds
   (question 3 B), MEASURED by the census's new line ROW-OUT-OF-CLOSED-HOLLOWS.

**Example.** The hollow at the foot of the belt's wet range now fills from the range's own runoff and spills; the
largest pocket, 11.9 km under the sea's level, is a dry basin at its own depth by its budget, and from 450 km a one-node tarn
in a pass is still water at its own level.

## B8. AFTER STEP 2: THE EDGES BEFORE R2, THE RIVERS FOLLOW THE WATER, R2's BENCH FIRST (2026-09-25; the owner chose each recommended option)

Step 2's pictures (`docs/investigation/2026-09-24/pictures/*_step2.png`) show the lakes in the right amount, but each
small lake is a flat blue square one node (8 km) wide — the node-square edge (status file §6 T4) that step 2 made
visible by making the lakes exist. The owner's four answers:

1. **THE EDGES COME BEFORE R2** (amends B6's order, steps 3 and 4 change places): the next build step is ONE SHORELINE
   FROM ONE FIELD — the sea's shore and the lake's edge where the ground crosses the water's level, never a node square,
   the lake's level held by its dry neighbours (W6's owed law) — on today's ground. The law does not depend on which
   ground it reads; R2 later gives it the 1 km valleys, and the pictures are judged again then. Until R2 the edges
   follow the fine noise, so a shore may look dune-shaped: stated, accepted.
2. **THE RIVERS FOLLOW THE WATER** (plan question 3, choice B, and the receivers question; amends B7 item 2): the
   shipped receivers and the discharge row follow the budget's own water — a full hollow's river runs over its spill
   into the sibling (Fill–Spill–Merge's flow directions), and a closed body that never fills passes nothing on. BEFORE
   R2, because R2 seeds its rivers from them. Cures G-RIVER-DESCENT (3 430 rising steps at generator 20) and the 1 607
   km³/yr the row carries to the sea out of hollows the budget holds (ROW-OUT-OF-CLOSED-HOLLOWS).
3. **R2's SLICE 0 FIRST**: the measured bench (R1's loop as the control, one change at a time, every number read
   through the shipped kernel) and a HILLSHADE of the four stand tiles at the owner's 100–150 km view, for the owner's
   look, BEFORE any bake service. A planet's coarser ground stands until its bake is done; never a temporary ground
   (T3, `r2_design.md` §16).
4. **COMMIT STEP 2 NOW** as generator 20's checkpoint; the square lakes stay until the edge step.

**Example.** On the plain stand forty lakes stand as blue tiles on the dunes. After the edge step each one is the
water inside the ground's own hollow at the lake's level — a ragged outline that follows the ground — and the river
that leaves a full one crosses its lowest rim notch into the next hollow, never back across a dry floor. Then R2's
bench shows the owner the same ground eroded at 1 km, before anything bakes a planet.

### B8 item 2 BUILT (2026-09-25; plan step 2b; NOT committed — ★ STOPPED on the plan's own stop condition)

`GENERATOR_VERSION` 21, `ARTIFACT_VERSION` 9 (`PROTO_MINOR` unchanged). The census lines are in
`docs/investigation/2026-09-24/step2b_census.txt`; the living status file §2, §4, §7 W10 and W11, §11 item 10 hold the
detail.

1. **The shipped receivers follow the budget's water** (`lakes::water_tree`): the budget's descent on dry ground, the
   flat step across a spilling lake to its spill (Rho8's split, a named scatter, T9), the spill over its saddle into the
   side the budget pours into, and an END under a closed lake's water or at a dry hollow's pit. MEASURED:
   G-RIVER-DESCENT 0 (3 430).
2. **The row is the budget's own water** (`MacroSolve::follow_the_water`): the runoff over those receivers, less at
   each spill what its lake hands the air. MEASURED: the row into the sea equals the cascade's to the unit (3 851.69
   km³/yr); 0 of 2 571 closed hollows leak (4 544 of 5 322); every closed body's ends carry what it holds.
3. **To be exact the budget had to follow the ground** (the plan's questions 1 and 2, choice A): a full hollow's
   overflow takes the saddle's steepest open side and runs down the ground toward the sea into the lower hollow it
   reaches (Fill–Spill–Merge's own link). MEASURED: the tree over generator 20's own budget leaves the row 188.2 km³/yr
   short of the cascade's; with questions 1 and 2 alone (variant V3) the row is exact. Beside them, NOT for the row
   (corrected by the review, 2026-09-25): a flat is swept from its drains (question 3, choice A — for item 1's flat
   directions, and it closes 8 795 leaves of zero depth that held 462.7 km³/yr on flats that drain), and one law the
   plan did not name — a node at its hollow's spill height costs that hollow nothing.
4. **The solved ground did not move** (0 of 8 871 936 heights; the sea's metre, the ocean share and the identity's
   measured half unchanged); the water words, the facies and the coast mask moved.
5. ★ **THE STOP: G-LAKE-SHARE 4.828 %** against the 4 % ceiling (3.510 % at generator 20; the attribution 4.162 % →
   4.741 % → 4.828 % in status §7 W11). The owner's word is owed before the commit (status §11 item 10: A keep and rule
   on the ceiling, recommended; B the tree over generator 20's budget; C look for a missing law first; D questions 1
   and 2 without the flat sweep and the brim, 4.162 %, added by the review).
6. ★ **THE SECOND STOP, found by the review (2026-09-25): the plan's flat law is NOT met.** The directions over a filled
   flat and across a spilling lake lean on the lattice: G-GRID-VALLEYS-BY-CLASS the flats 1.470 and the spilling lakes
   1.553 against ~1.006, G-LAKE-WALK 14.07° against test T9's 12°, both RED (both were printed NO-BAND before the
   review). Like for like the flat sweep made G-GRID-VALLEYS worse (1.093 without it, 1.205 with it). The owner's word
   is owed with item 5 (status §11 item 11).

**Example.** The hollow at the foot of the belt's wet range fills and spills east: its river crosses the lake straight
to the notch, runs down into the dry sibling hollow and ends at that hollow's salt lake. The row at the notch is the
range's runoff less what the full lake hands the air, and nothing of it reaches the sea.

## B9. STEP 2b's TWO STOPS ANSWERED, AND THE COMMIT (2026-09-25; the owner chose each recommended option)

After step 2b's census, its review and its pictures (status §8b: the four stands as at step 2; two new stands show
faults hidden until step 2 — the second pocket's floor with regular bump rows and a straight wall, and the high lake
from 1 000 km drawn as stripes; the lake depths measured, the largest lake 75 m deep at its median):

1. **THE LAKE CEILING: KEEP STEP 2b AND JUDGE LATER** (status §11 item 10, choice A). All of step 2b stays; G-LAKE-SHARE
   stays RED at 4.828 % against ruling B3's 4 %. The owner judges whether 4.8 % reads too wet on the pictures AFTER the
   shoreline step (plan step 4), when the lakes are no longer node squares; the ceiling is read again then.
2. **THE FLAT DIRECTIONS: COMMIT NOW, CURE BEFORE R2** (status §11 item 11, a new choice between A and B). Step 2b is
   committed with G-GRID-VALLEYS-BY-CLASS (flats 1.470, spilling lakes 1.553) and G-LAKE-WALK (14.07°) RED. The cure
   (Barnes, Lehman & Mulla's two combined fields, or a target that does not collapse onto one nearest drain) lands
   BEFORE R2 carves any valley, so no R2 valley follows the lattice; it moves the budget, so G-LAKE-SHARE is read again
   with it.
3. **COMMIT STEP 2b** (generator 21, artifact 9) with its census, the lake depths, the seven pictures and the documents.

**Example.** A pilot 150 km over the plain sees the same square lakes as after step 2 — the shoreline step changes
that. Before R2 cuts the plain's valleys at 1 km, the water across a filled flat stops leaning on the lattice, so R2's
rivers leave a flat toward its real drain, never along a grid diagonal.

## B10. THE SHORELINE BENCH'S VERDICT, THE WASHBOARD, AND THE ORDER (2026-09-26; the owner chose each recommended option)

The shoreline bench (plan step 4 slice 0; status §8c; `docs/investigation/2026-09-25/shoreline_bench*`) measured the drawn
water of four rules on eight stands, with its self-checks green and no world change:
- today's rule walls 84–97 % of the shore pairs at rung 7, and draws 130 979 columns of the high lake as land (ties `s = l`);
- **O1** (the ground's own crossing, a coarser cell from the node-scale line) jumps 3–9 px at almost every swap — a W16 breach;
- **N1** (the node-scale side as one field) is steady from rung 0 to rung 9 (p99 ≤ 1.4 px), walls 0–2 % of the pairs and draws
  no false land — but at rung 3 it flattens the ground around every lake into a textureless disc with node-aligned edges, keeps a
  notched staircase on a steep coast, and, like every rule, jumps about 9 px at the far swaps 9 → 10 and 10 → 11;
- **THE WASHBOARD**: the lakes and valleys themselves follow the lattice's eight directions (chains of lakes on exact diagonals,
  parallel 45° ridges one node apart with bumps along their crests, the high lake's stripes) — D8's own bias on smooth ground
  (Fairfield & Leymarie 1991), which no shore rule cures and R2 would carve along.

The owner's four answers:
1. **ONE MORE BENCH ROUND FOR THE SHORE (N2)** before any kernel form: keep N1's steady line; keep the ground's walking-scale
   texture near the water (fade only what can carry a column across it, with a reach from the ground, never the node square);
   clamp the node words to a band around the water before the read (the steep coast's notches); solve the far swaps (the bench's
   slice 0b). No game code changes until the numbers and the pictures pass.
2. **MEASURE TWO CURES FOR THE WASHBOARD BEFORE R2**: (a) a seeded, repeatable spread of each node's drain direction between the
   two lattice neighbours that bracket its steepest descent (Rho8, Fairfield & Leymarie 1991 — a NAMED scatter under T9, one
   receiver a node; the earlier refusal "a random tie-break breaks SL10" does not hold for a seeded hash, which step 2b already
   uses on the flats), and (b) D∞ (Tarboton 1997 — a continuous direction split between two neighbours, no fitted number). Each is
   measured on the existing grid-bias instruments (Tarboton's cone, Hyväluoma's rotation, the fourfold amplitude, G-GRID) and on
   the bench's pictures before the owner chooses. The refused cures stay refused: MFD with a fitted weight (T9), a hexagonal
   mesh (a new world, SL5). Step 2c (the flats) is folded into this work.
3. **THE ORDER: THE WASHBOARD FIRST, THEN THE SHORE (N2)**, because the washboard's cure moves every lake and valley and the
   shore bench must judge the final ones; then both are built in the game, and ALL stands are flown and compared with BOTH
   reference frames (the owner, 2026-09-25: *"After those fixes please make sure that you've also checked all visuals"*).
4. **COMMIT THE BENCH** (an instrument; no world change).

**Example.** Over the plain a chain of ponds runs north-east, one node wide, straight as a ruler for two hundred kilometres. After
the washboard's cure the same water finds its own way down the plain, and the ponds lie where the ground's hollows are, not on
the lattice's diagonal; then N2 draws each pond's edge where its ground meets its water, with the ground's texture to the shore.

## B11. THE WASHBOARD IS THE DEPOSIT'S; A SLOPED BASIN FILL, MEASURED FIRST; D∞ WITH IT (2026-09-26; the owner chose each recommended option)

The washboard bench (status §8d; `docs/investigation/2026-09-26/washboard_bench/`), its copy of the solve proven exact, measured
B10's two routing cures and found the washboard's cause:
- seeded Rho8 helps nothing; **D∞ improves the valleys** (Tarboton's cone error four to six times smaller, G-GRID's valleys
  1.205 → 1.116); neither removes the 45° ridges;
- a stage trace of the second pocket's floor shows the ridges born in the FIRST erosion pass (the fine texture 25 m → 405 m)
  and growing to 1 004 m at exactly 45° by pass 40: **the DEPOSIT lays each pass's cut in one-node lines along the flat's
  straight receiver paths**, because it fills the first lake node the flux reaches up to its spill;
- a LEVEL basin fill (the hollow fills from its lowest nodes to one level) removes the ridges (21 m) but makes dead-flat plains
  that flood whole — one lake per basin, straight node-line edges, the largest lake 12 562 → 43 776 nodes, the share 5.0 %.

The owner's three answers:
1. **A SLOPED BASIN FILL, MEASURED FIRST**: the sediment builds out from where the rivers enter and the floor falls gently to the
   basin's lowest part, which alone keeps the water — its slope from a PUBLISHED law with a stated calibration (T9, no drawn
   number), built in the bench and shown on the same basin pictures before any world change. Ruling W7's "the cut is laid in the
   first hollow downstream" stands; only how the hollow takes it changes.
2. **D∞ IS ADOPTED FOR THE EROSION PASSES, IN THE SAME CHANGE AS THE DEPOSIT CURE** (the world moves once); the shipped river
   data keeps one receiver a node (the water's tree of step 2b).
3. **COMMIT THE WASHBOARD BENCH** (an instrument; no world change).

**Example.** From 150 km over the second pocket a pilot sees today rows of beaded ridges at 45°. After the cure its floor falls
gently from the streams at its rim to a lake in its lowest part, and the valleys above it run where the ground leads them, not
along the lattice's eight directions.

## B12. THE BASINS PARKED UNTIL AFTER R2; THE SHORE (N2) AND R2 NEXT (2026-09-26; the owner chose the recommended option)

The sloped fill (ruling B11) was measured (status §8d): Paola, Heller & Angevine 1992's fans with the skip bit keep the pocket
floor clean (13.6 m RMS, no 45° line), survive the next sweep (99.6 %) and halve the ground's grid facing (15.6 % → 7.8 %);
but the pictures barely change at rung 7, the lakes merge into giants (45 340 nodes; share 5.1–5.2 %), and two faults further
up show through: the sweep cuts with the RAIN's discharge while the fan law reads the RUNOFF (5 545 of 12 362 inlets brought
sediment and no water), and the solve makes closed pockets up to 11.9 km deep and 1.57 million km² that no Earth basin
resembles. The owner's two answers:

1. **PARK THE BASINS; DO THE SHORE (N2) AND R2 NEXT.** Today's deposit stays in the world for now; the washboard bench, the
   level fill and the fan law stay ready in `vd_bins::router_variants`. R2's design takes up WHICH WATER DRIVES THE CUT (plan
   question 4) and measures D∞ inside its own erosion (B11 item 2's adoption moves into R2's world change). The basins —
   the pocket depths, the fill, the lakes' extent — are taken up again after R2, on R2's field.
2. **COMMIT THE FAN MEASUREMENT** (an instrument; no world change).

**Example.** A pilot 150 km over the second pocket still sees today's 45° ridges until R2 lands; the next pictures she is shown
are the shore's (N2): the plain's lakes as round ponds with their ground's texture to the water, and the steep coast without
its notched staircase.

## B13. N2 IS THE SHORE; THE 30 m BAND KEPT; BUILD IT, THEN FLY (2026-09-26; the owner chose every recommended option)

The shore round N2 was run in the shoreline bench (status §8e; an instrument, no world change). N2 is N1's side and N1's
node-scale blend plus four measured changes: a FADE PER OCTAVE (each fine octave climbs from nought at the line over its own
reach, 15/8 × its RMS height over its RMS slope, measured over the sphere — 38 m for the 49 m wave, 20 km for the 25 km wave),
faded by EVERY nearby water through a distance that cannot jump; a 30 m BAND on the node words before the held read; and THE
SIDE HOLD (a side-dry column stands over the sea by W6's share of the sea's own field, relaxed by the sea's presence, and in full
where it carries the sea as its standing water). The bench found WHY N1 draws flat discs: the ridged fine octaves stand on a
positive mean of 571.8 m at share one, so the land by every water stands `m × 572` m over the ground the solve placed the water
against. MEASURED: the deciding number passes (1.40 px, p99 0.35 px, against N1's 8.9 px); the columns whose drawn water
disagrees with the side fall from N1's 768 to 7; no lake appears or flickers; no wall at rung 7 on the plain, the gentle coast and
the high lake; the ground keeps a slope of 0.017–0.080 within 1 km of the shore where N1 keeps 0.007–0.089 with flat discs.

The owner's four answers:
1. **KEEP THE 30 m BAND.** Its stated price: a steep coast's line sits midway between its wet and dry nodes, up to half a node
   seaward of where the node-scale ground meets the sea (drawn as land at the sea's level there), and a lake deeper than 30 m is
   drawn by its node's footprint (the plain at rung 9: 16 of N1's 20 lakes).
2. **ACCEPT THE FILLED TROUGHS UNTIL R2 AND R4.** Fading the long 12–25 km waves toward nought near the water also fills their
   troughs (p99 81–97 m over the raw ground at rung 7; a lowland 12 km from the gentle coast rises 190 m). R2's 1 km level
   resolves those waves, and R4 (the noise only under the solved level) removes the lift; then the fade shrinks by itself.
3. **BUILD N2, THEN FLY.** Step 4 is built: N2 as the integer shore law in the recipe, on the server and on the client (one
   generator, two hosts, no drift — SL10), re-measured by the bench against its f64 reference; then the flight of ALL stands
   against BOTH reference pictures. NARROW WATER at the far swaps (an inlet or a lake about one far column wide flickers under
   the majority fold) stays RED for slice 0b (the per-water fold with a crossing byte — new data, the SL6 question).
4. **COMMIT THE ROUND** (the bench's N2, the census, the 50 pictures, the status file, the deferred registry).

**Example.** A pilot lands on the gentle coast. Today she sees a blue square; under N1 a smooth grey plate 4 km wide on a
straight line; under N2 the ground's swells run down to a smooth beach, and the coast stays put as the hull climbs to 1 400 km.

## B14. THE BUILD'S FIRST FOUR QUESTIONS, ANSWERED ON MEASURED NUMBERS (2026-09-26)

Step 4's build design (`docs/investigation/2026-09-26/n2_build_design.md`, written and attacked by workflow
`wf_b476b3a6-7c8`; 29 review findings answered) found that the near rungs can draw N2 from the rows the client already
holds, and that from rung R₁ = 10 up the client holds no fine row. Its slice S0 measured the far data first (§10 of it).
The owner's four answers:

1. **SEND THE SHORE BAND (SL6: YES).** Once, with the planet's head: the height of every fine node within 30 m of a water
   it is read for, and every lake node's level. MEASURED: 202 431 band nodes, 0.588 MB as a delta-coded list, and 93 225
   lake nodes, 0.178 MB — about 0.77 MB in all (the coast mask that ships today is 2.2 MB). So every rung reads the same
   words, and the far swaps the owner saw measured are the swaps that ship.
2. **THE REACH IS A LAW, AND THE SPECTRUM'S CONFLICT IS REGISTERED.** Each fine octave's reach near water is
   `C_kind × its wavelength`, with two constants measured over unit octaves of the recipe's own noise: C_smooth =
   0.7723, C_ridged = 0.9107 (the law against the per-octave statistic: −3.1 % to +3.7 %; with the law every N2 verdict
   stands and the deciding number reads 1.459 px). THE CONFLICT: the spectrum states each fine octave as a sine at the
   angle of repose (`body.rs:50-66`: an RMS slope of 0.70), but the noise realizes 0.139 per axis at share one — about a
   fifth — so its reach is about three times the sine law's. The spectrum's own law bears on the dunes and goes to
   DEFERRED for R2 and 8e.
3. **SEND THE FAR FOLD'S COUNTS TOO** (not the recommended option; the owner prefers no empty far globe). MEASURED: 111 089
   mixed node cells on the home planet; deriving them would leave a far globe without coasts for about 3 s on two workers;
   shipping their counts costs 0.636 MB more with the head (three 2-byte counts a mixed cell, and the level word the one
   naming rule needs). W15.1 item 5 ("the counts are derived and never shipped") is AMENDED for the samples' counts.
4. **KEEP THE POCKET EXEMPTION AS A STATED RULE.** A POCKET-word column is exempt from the sea's side hold, and the
   standing water's floor applies where the sea governs. MEASURED: on the home planet it changes no line (no pocket lake
   stands within three nodes of the sea); the variant that relaxes the floor by the sea's presence is REFUTED (at rungs
   10–12 it put the sea over 84 / 20 / 1 dry columns on the steep coast). N2's pull on the second pocket's floor (up to
   3.3 km near its lakes) is the 30 m band's price, accepted in B13; the pictures show cleaner lake outlines than N1's.

**Example.** A pilot climbs out of the gentle coast to 1 400 km. Under this ruling her client holds, beside the coast mask,
0.77 MB of shore heights and lake levels and 0.64 MB of far-fold counts, all sent with the planet's head: at 704 km the
coast does not move, and the far globe shows its coasts the moment the head's parts land.

**CORRECTION OF TWO MEASURED SIZES (2026-09-26, after this ruling; S0 item 5; the owner's re-confirmation asked).** The
ruling's numbers were measured by S0's first instrument, and two of them were low. (1) The band's rule left out every node
wet for a water and one metre (or less) under its level: its word is `W − 1`, not the band's edge. The one definition
(`shore_rules::band_needs`) holds 217 742 band nodes, **0.631 MB** (was 202 431, 0.588 MB); with the lake levels about
**0.81 MB** (was 0.77). The bench proves the missing nodes mattered: under the first rule N2 draws another far column in
1 466–10 476 of 1 397 760 columns per window. (2) The far counts' shipped unit is the NODE SQUARE (a rung-13 column holds a
quarter of each of four node cells), not the node cell: 192 271 squares, **1.100 MB** (was 111 089 cells, 0.636 MB). With
the head the planet ships about 1.91 MB beside the 2.2 MB mask.

## B15. S0's LAST QUESTIONS, ANSWERED ON MEASURED NUMBERS (2026-09-26)

Slice S0 of step 4's build design measured items 4, 5 and 6 (`docs/investigation/2026-09-26/n2_build_design.md` §10): the
one naming rule, the far windows at rungs 13–18 under the sample count pyramid, and the cube's seam and corner. The owner's
four answers:

1. **CURE THE FAR LAKE FLICKER FIRST** (Q7; NOT the recommended option). MEASURED: the one naming rule changes nothing at
   rungs 8–13 (0 of 16 748 census lines; the two rules name the same water on all 10 524 872 folded columns), and at rungs
   13–18 it keeps the water's area steadier than today's rule — but a big lake's ragged edge flickers in one-column fragments
   (LAKE-FLICKER at rung 14: 260 of 693 bodies against today's 53 of 714). Before S1, the bench measures a fold that does not
   flicker on nested footprints; the one naming rule is not built until then.
2. **BOTH FAR CURES JOIN N2** (Q13, Q14). The WET TIE: a column whose side is wet stands at least one length unit under its
   water (from rung 14 a column's centre is its node cell's middle, where a straight coast reads exactly nought, and a sea
   column 1 067 m deep was drawn as dry land at the sea's level). The LONE HOLD: a column with no water in its own block
   whose side is dry stands over its standing water by `k × band`, the default rule's own slot. MEASURED with both: 0–1
   mismatched columns at every far rung, WATER-STEP 0, 2 of 16 804 near lines moved (one a cure).
3. **B14 ITEMS 1 AND 3 STAND AT THE CORRECTED SIZES**: the shore band with the lake levels about 0.81 MB, the far counts per
   NODE SQUARE 1.10 MB; about 1.91 MB with the planet's head, beside the 2.2 MB coast mask.
4. **THE SMALLER QUESTIONS ACCEPTED AS RECOMMENDED**: Q4 (the three re-worded ruling items: a side is the held field's
   sign; a far cell's side is its samples' wet share; rungs 8–13 fold samples), Q5 (k = ¼, K = 4), Q6 (F = R₁ − 3), Q8
   (no side data: the ground's own sign), Q9 (the beach: today's rule on N2's governing water), Q10 (one switch at every
   rung after the band ships), Q12 (the corner phantom rule until the corner blend; MEASURED: ONE cube corner of the home
   planet is coastal).

**Example.** A pilot climbs over the high lakes at 5 882 m. Under the one naming rule a piece of the lake's edge shows at
rung 14, hides at rung 15 and shows again at rung 16 (64 %, 40 % and 59 % of its columns wet). Under this ruling that edge
is cured on the bench first, and only a fold that holds the edge still across the far swaps goes into the build.

## B16. NO FLICKER AND NO SHAPE CHANGE BETWEEN RUNGS — AT ALL (2026-09-26)

Asked to choose among three measured cures of the far lake flicker (the box, the tent, no lone column; the build design
§10 "B15 ITEM 1"), the owner answered: *"We need seamless experience - no flicker at all please. We need to find the best
solution for it. No flicker or changing of the shapes between rungs at all"*.

1. **NONE OF THE THREE FOLDS IS TAKEN.** Each one decides a far column's water by a threshold on counts, and a threshold
   on nested footprints lets a small feature change at a swap: the box lets one-column fragments show, hide and show; the
   tent and the lone rule remove the fragments but make water vanish or appear at a swap where every finer column agreed.
2. **THE BAR IS W16 IN FULL:** at every swap no flicker and no change of the water's or the land's shape. Find THE BEST
   SOLUTION — the fold alone may not be the place — measure it on the bench against that bar, and bring it to the owner
   before S1.

**Example.** A pilot climbs over the high lakes from 7 500 km to 60 000 km. Under this ruling the lakes' edges do not
shimmer at any of the four swaps on the way, and no lake changes its outline when the ring behind it swaps.
