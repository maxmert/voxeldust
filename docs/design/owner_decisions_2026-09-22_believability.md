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

## B17. THE SKIN'S SHADER CHOOSES ITS OWN LEVEL; THE RIM IS SIZED AT 86.4° AND JUDGED IN FLIGHT (2026-09-27)

Asked on slice K0's numbers (`docs/investigation/2026-09-26/far_water_seamless_design.md` §9; the censuses and panels
beside it), the owner answered four questions:

1. **Q9 — THE SHADER CHOOSES THE SKIN'S LEVEL ITSELF** (owner: "Shader's own level"). The skin's shader reads each probe at
   the level of P_max / η — the Vulkan specification's sketch, η the footprint's ratio capped at 16 — and takes its own
   probes (N = ⌈η⌉ trilinear reads at an explicit level), with the design's half-cell windows (a 22-texel margin, five
   levels). MEASURED: all 268 front lines and both climbs at 0.000 R8 steps; 17–79 MB of windows per stand over the
   client's own chunk set, plus an 11.3 MB far page; the same picture on every GPU. The GPU's own anisotropic filter is
   NOT used for the skin: its level rule is the vendor's, and under the GL extension's sketch the same windows pulse up to
   54 steps at the swaps (the quarter-cell cure costs 3.6 times the memory).
2. **Q8 — THE RIM IS SIZED FOR θ_rim = 86.4° NOW AND JUDGED IN THE FLIGHT** (owner: "86.4° now, judge in flight"). The
   margins stay the design's. Steps of up to 81 at a front can show in the last rows of a low pilot's skyline (88°–89°).
   Wider margins come only if the flight shows the rim (MEASURED prices at the high lake: 88° at twice the margin,
   122 MB; 89° at four times, 277 MB; nothing closes 90°).
3. **Q1–Q7 STAY OPEN** (owner: "Discuss some first"): the owner discusses some of them before answering.
4. **K0 IS COMMITTED** (owner: "Commit it").

**Example.** A pilot 150 km over the high lakes looks across the chain of small lakes. The skin's shader chooses each
probe's level itself, so a GPU from any vendor draws the same colours, and the lakes' windows cost about 79 MB. The pilot
drops to 20 km and looks at the skyline: in its last rows a faint line can show where the 9 → 10 ring meets the next ring;
the flight decides whether the margins grow.

## B18. THE LID AND ITS DRY WORD, THE NEAR SLIDES, THE HAND-OVER, SLICE 0b AND THE RAMP (2026-09-27)

The owner first asked for plain explanations (*"Q2 - I don't want anything, that will not be believable experience -
when we will have long shadows all flat will not cast anything. So I'd have heights at any rung"*; *"Q3 - don't
understand what that means"*; *"Q4 - … will it be more believable experience?"*; *"Q5 - no jumps or flicker between
rungs"*; *"Q6 - only if that will not hurt realistic experience"*; *"Q7 - why we have performance degradation?"*). The
answers were MEASURED where a number was owed (LID-RELIEF, the design §9.6), and the owner then ruled:

1. **Q2 — THE LID, YES.** From rung 14 a far column's height is the mean of what a pilot sees over it. It IS a height at
   every rung: MEASURED on the four stands, a dry column keeps its own ground (within 6 m at rungs 16–18; where rungs
   14–15 differ by up to 697 m it is today's rule that moves a dry column near a lake off its ground), and the relief —
   the mean height step between neighbouring columns, which the light and the far shadows read — is today's within
   1–3 % at every far rung. Only a column that is part water and part land stands between the two instead of jumping.
2. **Q3 — THE DRY WORD SHIPS (SL6 yes).** Each part-wet level-1 node's dry-node mean height, 25 053 numbers, 48.9 KB,
   with the planet's head in the band's arm: the lid stands on the land's own height at a coast (without it: 47 m
   typical, 1 750 m at the worst node).
3. **Q4 — R2 NOW; THE SLIDES JUDGED IN THE FLIGHT.** A near shore slides with its ground across each band (at most
   1.46 px for the sea, 3.29 and 4.79 px for lakes) instead of popping at the front; the exact nested line (design B)
   is designed only if the flight shows the slides.
4. **Q5 — THE HAND-OVER BY THE PIXEL'S SIZE, YES** (the owner's condition: no jump or flicker between rungs — met: no
   rung decides it; MEASURED a p99 change of 13–19 % over the whole 56–111 km band, gradual).
5. **Q6 — SLICE 0b RETIRES** (the owner's condition: no loss of realism — met: the skin draws narrow water by its share
   at every distance, AREA-NEST exactly 0 at every far swap); B13 item 3's crossing-byte question is withdrawn.
6. **Q7 — THE ARRIVAL RAMP IS 0.5 s.** A late chunk (streaming: at 150 km over the plain the client wants 3 846 chunks,
   a rung-9 chunk takes a median of 11 ms on one thread) enters on its stand-in's state and rises into place over half
   a second.
7. **Q1 — THE FAR LOOK, YES** (asked last, on the globes and the climb strip): past 111 km a coast is a soft edge about
   one pixel wide, and water thinner than a pixel is a pale tint that follows the pixel's size. Design D is ruled whole;
   NEXT: slice R1 (the renderer's own pops) on the owner's word.

**Example.** A pilot 20 000 km over the steep coast sees the far mountains at their heights, so their long shadows fall
as the ground's own steps say; the coast's columns blend the sea and the land at their true heights; and when the pilot
dives, each late chunk rises into place over half a second instead of jumping.

## B19. THE WALK ON THE SOLVED GROUND: THE ARTIFACT DELIVERY, THE STAND, THE BERTH'S FACING AND THE WINDOW FRAME'S PARTS (2026-09-27)

The owner's order was *"The walk's band gap first"*, then R2. The walk's gap was the world's first solves landing in the
moving-eye gate's legs (DEFERRED "THE WALK'S BAND GOES INCOMPLETE AGAIN"), and the cure found four more defects. Each
answer below is the owner's choice between measured options:

1. **THE GATEWAY'S ARTIFACT DELIVERY IS FIXED NOW** (owner: "Fix now"): one byte budget a beat per client, a re-drive after
   a shed send, and a session forgets a realm's artifact records when its scene delta removes the realm.
2. **THE GATE SOLVES THE HOME PLANET ITSELF** (owner: "Solve in the test") and seats every stand and the hull's clearance
   on the solved ground.
3. **THE STAND** (owner: "Near equator, star 35°", then *"Move the stand to a 15° star first"*): the walkable day-side
   land on the circle where the star stands 15° up.
4. **A BERTH STATES ITS FACING** (owner: "Berths get a facing"; the commit after the full gate passes): the attitude the
   parent parked its child in, an optional stored field.
5. **A TICK'S PICTURE IS BUILT ONLY FROM COMPLETE FRAMES** (owner: "Complete frames only"): the far planets blinked out of
   the picture because the gateway built a tick's picture from the first part of a two-part window frame.
6. **SL6 YES — THE WINDOW FRAME'S PARTS** (owner: "New trailing arm"): *"A new ShardToGateway arm at index 17: the same
   window frame plus its part number and part count; minor 35, the floor stays 34 (additive). The shard sends the new
   arm; the gateway still reads the old arm as a whole one-part frame. The gateway builds a tick's picture only from a
   stamp whose parts are all here."* Built as `ShardToGateway::WindowFramePart`; the window frame has its own header
   budget (MEASURED: the older frame's widest header was 187 bytes against the 184 it was cut under; the part's is 197).

**Example.** A walker on the home planet stands still and looks up. The star system ships its nine planets in two parts
each tick, and the three largest planets ride the second. When that part is late, the gateway keeps the last whole
tick's picture, so the three planets stay in the sky and the client never forgets their artifacts.

## B20. THE FIRST LOOK IN THE WINDOW: THE COAST FIRST, AND HALF THE CORES (2026-09-27)

The owner looked in the window at the 15° stand and said: *"The quality and speed is quite terrible at the moment"*, and
*"I just moved the observer — I haven't even fly in the ship fast"*. MEASURED against the solved ground at the positions of
the owner's two pictures: the observer stood 104 m over the sea near the coast, the camera 117–135 m over it (the observer
keeps the radius it started at; nothing holds it to the ground yet). From there the coast drew a hole between the water
and the land (the sky under the land), the skirts as dark walls with holes, and the land's undersides; a full press of W
moved the observer at up to about 200 m/s (124 m/s measured over 5 s); the window built its ground on 3 workers of 14
cores. No gate had looked at a coast from over the sea. The owner ruled:

1. **THE COAST FIRST** (owner: "The coast first"): first a gate that stands about 100 m over the sea and looks at the coast,
   and that FAILS on today's picture; then the hole and the walls are cured until it passes; then the owner looks again in
   the window. The observer's walk on the drawn ground comes after.
2. **THE TERRAIN'S SHARE IS HALF THE CORES** (owner: "Half: 7 of 14"; amends F6's quarter): `worker_share` takes half of
   the cores, never fewer than two — seven on the owner's Mac, four on an eight-core machine.

**Example.** A walker at the stand presses W and drifts out over the bay, 100 m over the water. Today the coast behind
the water shows the sky under the land and dark walls; the new gate stands exactly there and counts those pixels, and
the owner looks again only when it reads zero.

## B21. STOP THE PATCHES: DESIGN THE ONE-VOLUME MODEL FIRST (2026-09-27)

After the first look and the measured 208 m coast wall (B20), the owner said: *"I wonder how all that will work when we
will edit the terrain using different tools? Even today without any editing we have so many issues."* and *"I feel that
you don't build a generic solution, but some parts that you try to fit together."* The answer given: yes — the ground and
the water come from several separate sources today (the 8 km solve, the fine octaves per column, the per-distance copies
that are means, the sea and lake levels with a sea/land word per 8 km node), and each special rule (the quarter hold,
the side words, the band, the three sheet cuts, the far skin and lid) moves a disagreement instead of removing it; an
edit near water would have to satisfy every one of them. The owner asked: *"for the one-volume model - what exactly
that means? Can we generate terrain, but then 'fill' with water, or air, or other substances? And don't try to
connect"*, and ruled:

1. **STOP ADDING TERRAIN RULES; N2 IS NOT BUILT** (B13's build is parked, not retired, until the design below is decided).
2. **DESIGN THE ONE-VOLUME MODEL FIRST:** one volume of cells per body, each cell a material (rock kinds, sand, soil,
   water, air, vacuum, ice, lava, gas, later blocks) with a fill amount for smooth surfaces; generation in passes — the
   SOLID pass, then the FILL pass (the sea, each lake to its level, air or vacuum, other substances); edits change the
   same cells (the one-hop diff, SL10; live water, W2); every distance is a copy made from the same volume by one rule; a
   shore is where water cells touch ground cells. "Don't try to connect" is read as: do not fit the new model onto
   today's special shore and water rules; an existing part stays only if it fits unchanged.
3. **THE METHOD:** a multi-agent design with adversarial reviews (owner: "Multi-agent with reviews"; workflow
   `wf_a9730eb2-4b9`: three readers, three competing designs — procedural first, stored truth first, learned from shipped
   games — two adversarial reviewers, one final document, `docs/investigation/2026-09-27/one_volume_design.md`). The far
   coast question (exact agreement at every distance against a small smooth change) goes to the owner with a price on
   each side.

**Example.** A player digs a trench from the sea into the land. Today the land's 8 km node keeps its LAND word, the water
sheet is drawn from the generator's water, and the shore law lifts or sinks the dug ground — no water in the trench. In
the one-volume model the dug cells become empty, the fill rule runs again on them, the shard moves water in, and a pilot
50 km up sees a thin line of water because the far copy is made from the same cells.

## B22. THE ONE-VOLUME DESIGN'S ANSWERS AND THE NEW ORDER OF WORK (2026-09-28)

The design (`docs/investigation/2026-09-27/one_volume_design.md`, workflows `wf_a9730eb2-4b9` and `wf_3932de59-28b`)
went to the owner with its questions. The owner added four requirements while it ran — *"please make sure that we are
extremely performant!!!"*, *"We can comment out all current special shores and sea models"*, *"while performant we should
not build seams. Also there should not be jumps or flicker between rungs"*, *"we SHOULD NOT break ground laws and rules of
course"* — and then ruled, after a discussion of the design and the approach (*"my current machine is very performant,
and even today the game is slow ... we need to work on optimisations"*; *"should we disable the old water algos, work
on the terrain (with biomes), fully believable, optimised, and only then work on the filling, as in theory those 2 are
separate things now and hardly affect each other?"*):

1. **THE ORDER OF WORK** (owner: "Confirm the order"; supersedes B20 item 1's "the coast first"):
   1. switch off the old water and shore rules (the world changes once: the coast walls go, the dry sea floor shows);
   2. SPEED — the builder and the main thread, no world change; 60 frames a second at FULL SCREEN, no stall;
   3. THE GROUND with biomes: believable (the finer erosion level, the valleys, the dunes, the cliffs), fast, and with
      no flicker at any distance; judged in the window against the reference picture;
   4. THE WATER FILL: the one-volume water (design option (b)), no flicker at any distance;
   5. THE EDIT SYSTEM as its own step ("it's not only about the terrain edit").
   The planet's water CALCULATION (the solve's rivers, sea level and lakes, which shape the ground) stays in the ground
   work; only the DRAWING of water is off until step 4.
2. **WHILE THE WATER IS OFF, NO WATER IS DRAWN AT ALL** (owner: "No water at all"): the dry sea floor shows (the sea is
   78 % of the home planet); coasts are judged after step 4.
3. **THE FRAME TARGET: 60 FRAMES A SECOND AT FULL SCREEN** (3 456 × 2 234; owner: "60 at full screen"), not at today's
   window size.
4. **THE DESIGN'S QUESTIONS:** Q1 THE FAR COAST — option (b), exact from about 450 km out and smooth nearer; Q2 — MEASURE
   the smaller hand-over, with the goal *"no flickers at ANY distance"*; Q3 still water in hollows and caves — ACCEPTED;
   Q4 dug holes under W2 — ACCEPTED; Q5 the small hills' average — DECIDED IN THE GROUND STEP (the owner: *"We will need
   to work on the hills and terrain again, so it might change again"*; the zero-average law was recommended); Q6 the new
   data — ACCEPTED; Q7 — the trench waits for the edit system, which is its own step. Q8–Q23 stand at their
   recommendations as defaults, each asked again with its numbers when it is due.

**Example.** A pilot flies over the home planet after step 1: the belt's hills stand as before, the old coast walls are
gone, and where the sea was the dry sea floor falls away into deep basins; no water is drawn anywhere until step 4 fills
every open cell under its water's level.

## B23. NO CAPS, THE THIRD REFERENCE PICTURE, AND LAWS THAT HOLD ON EVERY BODY (2026-09-28)

After the plan for step 3 (the ground: the small hills' average, the flat directions, the finer eroded level R2, the
1–5 km valleys R3, the biomes, the cliffs, the owner's look) the owner said: *"Ideally we don't do any caps and see
what's happening, this is the reference [a picture] We need it very believable, but fully generated, not custom built,
as all those laws should be applied to other planets and moons"*. Read as:

1. **NO CAPS: THE LAWS DECIDE, AND WE LOOK.** Nothing limits what the laws produce in order to reach a look.
   - The gates' bands (B3: G-LAKE-SHARE's 4 %, G-ICE's ~10 %, G-CRATER's ~190, G-BUZZSAW's 1 500 m) become READINGS that
     every step prints. A reading outside its band is reported, and it never stops a step by itself.
   - Every clamp, ceiling and floor inside an operator (for example R2's carve ceiling and its clamped stream power, the
     placeholder roughness) is listed by the ground step's design. Each one goes, or the law it stood in for replaces it
     (T9: a band that stands in for a missing law is a defect).
   - A limit that IS a law stays, named with its law and its calibration body: the angle of repose, the height that a
     crust can hold under its gravity, the stability bound of a time step.
   - The polar and snow caps come only from the climate law (B5); nobody sets a cap by hand.
   - The judge is the owner's look (G-LOOK) on the pictures and in the window.
2. **THE THIRD REFERENCE PICTURE.** It is kept outside the public repo like the first two
   (`/Users/maxim/Downloads/Everything-we-Know-About-Crimson-Desert-Featured.webp`, a photograph of a commercial game) and
   described in words in the living status file §9.
3. **FULLY GENERATED, NOT CUSTOM BUILT.** Every landform comes from the generator's laws, from the seed and the body's
   charter. Nothing is placed by hand, and nothing is tuned for one planet. The same laws make every planet and moon, so
   each ground step is judged on the home planet AND on other real bodies of THE world (another planet, a moon), never on a
   test-only variant (SL5). The towns, castles and fields in the picture are not terrain: players build them.

**Example.** Nobody models the picture's red rock towers. A layered sandstone with a hard cap over soft beds erodes into
mesas, buttes and pillars by the same law that rounds a granite hill. On an airless moon the same code makes no towers,
because no water or wind cuts its rock, and its craters stay sharp.

## B24. BUILD THE TEMPORAL PATH (2026-09-29)

The full-screen frame on the GPU's own clock is 32.25 ms on foot, and it is paid per pixel (DEFERRED "THE SPEED STEP").
The owner asked whether one-pass MSAA suits a Star-Citizen-like game ("everything will be way more complex, with
constructions, clouds, weather, hundreds of players"), took the answer (no: temporal reconstruction is the foundation),
chose *"Design the temporal path"*, asked whether the CPU cores can help and whether weather, clouds, water and waves need
the path (answered: the CPU helps in its own lane — deciding what the GPU must not draw, building, simulating — and those
features enter through the design's contract, §4.6), and then ruled: *"Ok, let's build temporal pass now."* Read as:

1. **D1 YES: BUILD THE TEMPORAL PATH** by `docs/investigation/2026-09-28/temporal_path_design.md`. MSAA stays the default
   until the owner's edge look (D5); the retirement of MSAA is D14.
2. **D2, D3 AND D4 AS THE DESIGN RECOMMENDS** (the owner read them with their recommendations and gave the word to build;
   each stays open to the owner's change): the order T0 (measure, no pixel change) → T1 (the motion) → T2 (the first
   picture, at the sizes that ship) → T3 (the one geometry pass, only after D5 = yes); our own reconstruction on FSR 2.2's
   published structure (MIT), which runs on every GPU, with Apple's scaler only as a later challenger and only on D12
   (its new library); the ground in one pass that writes its surface, its depth and its motion, at T3.
3. **EVERY LOOK STAYS THE OWNER'S** (D5–D16), each judged on pictures at its slice; a window opens only with the owner's
   word at that moment.

**Example.** A pilot on foot sees no change after T0 and T1: T0 adds the clocks and the meters, T1 gives every drawn thing
its motion behind a switch. The first picture change comes at T2, where the owner compares the rebuilt horizon at full,
two-thirds and half size against today's MSAA edge before anything larger is built.

## B25. THE AIR RESTS ON THE SEA AND REACHES EVERY GROUND (2026-09-30)

The owner flew the hull low over the dry sea floor and saw the haze vanish (*"something is wrong with the athmosphere
level … Not sure if it's normal behaviour"*). MEASURED (2026-09-30, headless, `examples/air_ground_stand.rs`): the air
model's ground was one sphere at the ladder's datum; the solve's sea stands 3 591 m over it and the dry sea floor reaches
8 276 m under it (7.7 km under the sea only 20 km from the owner's stand). An eye under that sphere was lifted onto it,
and every ray under the eye's level ended there: at 2 135 m under the sphere the picture showed a hard line at the eye's
level, the ground over it hazed (RGB 71, 65, 68) and the ground under it bare (62, 50, 36); 2 002 m over the sphere on the
same line of sight the haze was even. The temporal path on and off gave the same numbers. Two more defects of the same law
were measured: the air's words held at the datum, so over the sea and the owner's coast the blue air was 61 % and the haze
3 % of the charter's words; and the haze's scattering and absorption were swapped (Bevy 0.18.1's own Earth medium swaps
them; Bevy 0.16.1's `Atmosphere::EARTH` and Bruneton's calibration scatter 3.996 parts in 4.440). The owner approved all
four laws and chose to build them before the rest of T2:

1. **THE AIR RESTS ON THE SEA.** The charter's surface words (its pressure is the air's weight, calibrated on Earth's at
   its sea level) hold at the SEA LEVEL on a body with a sea — the level the owning realm states in its charter — and at
   the ladder's datum on a dry body.
2. **THE AIR REACHES EVERY GROUND.** No eye and no drawn ground is ever under the air. The molecules grow denser lower
   down by the pressure law, with no cap; under the sea level the haze keeps its sea-level value (the haze is the ledgered
   placeholder, and its source law is 8c/8e/8o's).
3. **THE PLANET'S SHADOW IN THE AIR IS THE SEA'S HORIZON:** the air over the coast goes dark when the sun sets behind the
   sea (before, the sphere 3.6 km lower kept it lit until the sun stood 1.9° under the sea's horizon).
4. **THE HAZE SCATTERS** as the published calibration says: nine tenths of its extinction scatters, one tenth absorbs.

**BUILT AS (2026-09-30):** the law (`vd_client::sky`) puts the air's ground sphere at the charter's sea (laws 1 and 3 in
one number) and states the air under it; the vendored air (`vendor/bevy_pbr/PATCH.md`, the sixth change) marches each
pixel from the TRUE eye to its drawn ground, under the sphere where the ground stands there, with the molecules grown by
the pressure law and the haze held; a pixel's SKY ray also runs under the sphere and ends only where it leaves the air or
at the FLOOR (the datum less the body's relief bound: no ground stands lower). MEASURED on the way: a sky ray still ended
at the sea's sphere drew that sphere's plain ground between the sea's horizon and the far dry floor's (0.3° to 2.8° under
the eye's level at the owner's stand), a dark band from brightness 128 down to 99; with the sky ray through the air the
rows run 128.5 → 126.5 → 126.7 from the sky into the far floor, and the two drawing paths agree within 0.2 levels. Law 2's first wording moved the model's ground sphere down to the ladder's floor; it
was not built so. MEASURED: the home planet's ladder floor stands 262 144 m under the datum (one whole top-rung cell), where
the pressure law gives the air `e^37` times the sea level's density — no table holds it. A sphere at the LOWEST GROUND
instead (the finest node, 11 867 m under the sea) would put air inside the rock into the engine's light tables, with an
optical depth of 0.43 (green) and 1.04 (blue) between the sea level and that sphere, and that air would light the sky
from below. UNMEASURED: neither form was drawn.

**Example.** The hull hangs 2 km over the dry sea floor, 20 km off the owner's coast. Before: a hard line at the eye's
level, bare ground under it. After: the far floor and the rising coast take the haze of every metre of air between, and
the coast's air is the charter's own, so ridges 50 km off wear a pale veil.

## B26. NO TERM GROWS UNDER THE SEA'S LEVEL (2026-09-30)

The owner, from 7 800 km over the home planet with the water off (B22), the dry sea floors washed pale: *"That how it looks
from above - the athmosphere in low places might destroy the whole look. Can we not increse dencity below the sea
level?"* — this AMENDS B25 law 2: under the sea level EVERY term of the air HOLDS its sea-level value (the molecules no
longer grow by the pressure law; the haze held already). Laws 1, 3 and 4 and law 2's reach (no eye and no drawn ground is
under the air) stand.

**BUILT AS (2026-09-30):** the law's growing coefficient (`vd_client::sky::Below::scattering_per_m`) is nought; the
vendored march keeps its form (the sixth change states the coefficient the game gives).

**MEASURED (the same pose and light, 7 793 km over the ground, headless):** the basins moved 0–5 levels (one basin
(102, 112, 136) → (99, 107, 131), another unchanged), the land not at all — the basins STAY pale, because a dry basin 7 km
deep holds about as much air as the whole column over the sea level even at the sea level's density (the molecules'
optical depth about 0.09 under the sea against about 0.1 over it, the haze's about 0.03 against 0.005), so from orbit a
basin looks through about twice the air the land does. The planned cure is the water fill (B22 step 4): the basins hold
water, and the air ends on the sea's surface.

**Example.** The pilot hangs 7 800 km over the dry ocean east of the dunes: the basin's floor is a little less blue than
before, still paler than the tan land beside it, until the water fill puts the sea back.

## B27. THE EYE'S RULE IS THE SEAM LAW; THE WALK'S STEPS AND THE CRACK FIRST (2026-10-01)

The owner, on the eye's reading of the four-frame flight (DEFERRED, the T2 block): *"Yes, as it stands"* — and for what
comes next: *"1,2 first please, then rest of T2 gates"*.

1. **THE EYE'S RULE IS THE SEAM LAW of the flight gate, AS IT STANDS** (`vd_client_harness::eye`): the DISPLAY picture,
   read under the engine's screen dither; a pop is the JUMP between two still sides of a step (four frames a pair, the
   step the middle two); a compact pop counts beyond the eight-bit rounding as a PATCH, a faint wide pop as a WINDOW at
   least a degree wide; judged against Blackwell's thresholds as Crumey fitted them, at the natural seat (the picture
   fills the camera's 45° on the eye), on the reference display (white 200 cd/m², black 0.2 cd/m²), SEEN at level one.
   Every flight leg — the walk, 240 m/s, 528 m/s, the turn — must hold NO pop the eye sees; the landing (the login,
   which may hide behind a loading screen) is REPORTED. The level rule (one hundredth of a border's pixels plus one a
   pair past the floor) retires as a law; its numbers stay as readings.
2. **THE ORDER:** (1) the walk's stepping eye (at 1.4 m/s the drawn eye stands still for up to 5 ticks, then moves
   4–7 cm at once) and (2) the flickering crack (a single dark pixel on the ground for a frame or two) — each measured
   to its cause, then cured; THEN the rest of T2's gates.

**Example.** The hull flies at 240 m/s over the dunes and a rung-2 band under it steps two levels brighter and stays: a
patch of 0.15 square degrees at 3 % on a 40 cd/m² field, level ten — the gate goes red and names the rung. A lone
pixel one level darker for one frame stays under the rounding and is no pop.


## B28. THE TILE REACH CURED NOW; THE PICTURE GATE WAITS FOR THE SOLVE; FIND THE HISTORY'S AMPLIFIER (2026-10-01)

Three findings of the T2 gates' first run (DEFERRED, the T2 block: G-DETERMINISM and the seam stand), each answered with
the recommended option:

1. **THE TILE REACH IS CURED NOW, BEFORE THE REST OF T2** (owner: "Cure it now"). The shard and the gateway measured the
   pilot's height over the planet's RADIUS, the client's ladder over the SOLVED ground (since 2026-09-24); at the picture
   gate's seam stand the client waited for tiles the shard never sent, and a 3 km step with sky in it stood while the pilot
   stood there. One reach rule on both hosts (ruling W17), from the artifact each one holds; no new data crosses. (Built:
   the shard reads the ground its artifact states; the gateway, which relays tiles undecoded, serves the whole ring.)
2. **THE PICTURE GATE WAITS FOR THE SOLVE** (owner: "Wait for the solve"): every stand shows the solved ground, as the
   moving-eye gate does (ruling B19 items 2 and 3). The four exact references change; the new pictures come to the owner
   beside the old ones, and nothing is frozen before the owner accepts them.
3. **G-DETERMINISM: FIND THE AMPLIFIER** (owner: "Find the amplifier"): why a world a few ticks apart moves a converged
   pixel by up to 4 levels while the single frames agree within one; a still scene should not amplify so small a change.

**Example.** The pilot stands on a cube's edge, 3 283 m over the dry sea floor. The shard names the tiles out to the
ring the client asks from that floor, and the ground under the far chunks builds; the picture gate takes that picture
only once the planet's artifact is held, so the stand shows the solved world the players see.

## B29. THE SOLVED PICTURES ACCEPTED; THE AMPLIFIER'S MOST ROBUST CURE; THE WINDOW ASKED AGAIN (2026-10-01)

The owner's answers to the four questions of B28's report:

1. **THE SOLVED-GROUND PICTURES ARE ACCEPTED AND FROZEN** (owner: "Accept and freeze"). The picture gate on the solved
   ground, the default path (the temporal path off), froze every stand into the owner's folder
   (`docs/investigation/2026-09-07/pictures/exact/`, 2026-10-01 16:00–16:08), the far stand's reference with them.
   The recipe-ground references of 2026-09-24 are retired.
2. **THE AMPLIFIER'S CURE: "the most robust and production ready solution"** (owner's question). The answer, MEASURED
   by the shader's replay on the captured frames: FSR 2.2's own history rectification — the history CLAMPED axis by
   axis in YCoCg into the deviation box cut to the samples' own range (`RectifyHistory`), the 16-bit history kept (FSR
   2.2 ships it). It is the box's nearest point, so two colours never move apart; two inputs 1e-5 apart end within one
   level at every clock, and the picture at one jitter phase repeats exactly every cycle. The 32-bit history only
   removes one source (the rounding) and keeps the clip's 177-fold growth of any real input difference. BUILT, and
   MEASURED in two flights: every stand within one level (the hill 7 → 1). THE OWNER'S LOOK GIVEN (2026-10-01, the
   window at the hill stand: scale 1.0 "Good", then flying hull 10 and hull 11 at 0.67: "All good!").
3. **THE TERRAIN CRATE'S COVERAGE** (owner: "We will work on the terrain believability now, do you recommend to write
   those tests still?"): answered — not as a separate pass now; each believability change leaves the files it touches
   at 100 % in the same change, and what remains closes when the arc ends. The owner's word stands owed.
4. **THE WINDOW** (owner: "Ask me then"): the looks D5, D6 and D7 are asked again when the cure stands.

**Example.** The pilot stands on the hill at dusk. Before the cure, two flights a few ticks apart drew the dune's sunlit
slope up to 7 levels apart, because the history drifted along its colour box; with the clamp the history sits at the
box's nearest point in both flights, and the two pictures of the slope agree within one level.

## B30. ONE COMMIT NOW; THE TIMING WITH DOCKER STOPPED; G-GHOST AND THE VIEW SWITCH'S CROSSING MOVE TO T4 (2026-10-01)

The owner's answers to the three questions after the look of B29:

1. **ONE COMMIT NOW** (owner: "Commit now"): the work of B28 and B29 lands as one commit on the `game` branch, with no
   push. T2's last flights follow in a later commit.
2. **THE TIMING READINGS** (owner: "Docker stopped"): the window's own cost reading and G-NO-STALL are measured while
   Docker is stopped. The window opens only on the owner's word at that moment.
3. **THE CUT OF T2** (owner: "Yes, move them"): G-GHOST waits for T4 with D9, because its bound is D9's; the crossing
   reading of the view switch (V, the chase camera) moves to T4 too. The crossing readings of a boarding and of a
   departure stay in T2.

**Example.** The pilot walks into hull 11 on the hill and takes the hull's frame: T2 reads the frames before and after
that swap. The pilot then presses V and the camera moves back behind the hull: T4 reads that switch, at the frame rate
that ships.

## B31. THE BOARDING'S THREE CURES; THE SWAP'S TOLERANCE; THE SKY'S LIGHT JUDGED IN THE WINDOW (2026-10-02)

The owner's answers after the boarding's seams were measured (the owner first: "Cure both now"):

1. **THE THREE CURES, BUILT AND MEASURED:** the planet's window rows carry each child's attitude (the hull's box no
   longer turns at the swap: 20.1 → 0.06 levels); every drawn row reads the camera's own moment (the hull's box no
   longer vanishes for one frame); the vendored sky table's march starts at the eye's own place in the world's axes —
   on foot the ground had NO sky light (94.615 levels with it on and at nought), and now it takes it (the vendored
   eighth change). On the temporal path no pop the eye sees on the walk or the hull leg.
2. **THE SWAP'S TOLERANCE** (owner: "We need to have very seamless experience, believable, without any visual flickering
   or jumps. So if half-level provides that, then it's ok"): the swap pair's mean step may stand over the widest pair
   beside it by at most half a level (the eight-bit rounding of one display level), and its share of pixels past 3
   levels by at most one in ten thousand; the eye's rule (B27) still judges every leg for what the eye sees. Built as
   the moving-eye gate's verdict (G-CROSSING).
3. **THE SKY'S LIGHT ON FOOT IS JUDGED IN THE WINDOW** (owner: "Look in the window"): the ground's shade is bluer and
   brighter now; the solved references freeze again only on the owner's word after the look.
4. **THE COMMIT FOLLOWS THE LOOK** (owner: "After the look").

**Example.** The pilot stands on the ground stand, the sun 15° up: before the cure the red ruler ball's shaded side took
the haze alone and read dark grey (27, 23, 28); now the sky lights it a dark red, the ground's shade turns bluer, and at
the boarding the ground keeps that light through the swap instead of switching it on one frame later.

## B32. THE SKY'S LIGHT ON FOOT ACCEPTED; FROZEN AND COMMITTED (2026-10-02)

The owner looked in the window at the ground stand, the ground under the sky's light (B31 item 3): **"It looks great!!!
freeze and commit please!"**

1. **THE NEW LOOK IS ACCEPTED:** the ground on foot takes the sky's light (its shade bluer and brighter); the solved
   references (`docs/investigation/2026-09-07/pictures/exact/`) are frozen again on it, every stand, the default path.
2. **ONE COMMIT** of the boarding's three cures, the gates' changes and the new references, on `game`, with no push.

**Example.** The pilot stands on the ground stand and looks at a dune's lee side: its shade is the sky's cool light on the
sand, as the window showed the owner, and the picture gate now holds exactly that picture as its reference.

## B33. T2 CLOSES WITH ONE WINDOW SESSION; ITS READINGS MOVE TO T4; THEN THE TERRAIN (2026-10-02)

The owner asked: "When you're done, will we continue to making terrain believable?" — and chose, of three ways to close T2:
**"Readings to T4"**.

1. **TO T4:** the departure's crossing (its own trigger), the sky's lag on the descent and the climb, the T0 meters at two
   sizes, the still records' own grid, a tool's jump as a reset — readings with no look of the owner's in them.
2. **T2 CLOSES AFTER ONE WINDOW SESSION** with the owner, Docker stopped: D5 (the edge) at half size, still and turning;
   D6 (the stars' light: today's sprite, or the law's exact light); the window's frame cost at full screen; G-NO-STALL
   (the worst frame over a start, a full-screen switch and a boarding, and no pipeline compiled after the pre-roll).
3. **THEN THE TERRAIN STEP'S DISCUSSION**, its first item the waves the owner saw in flight.

**Example.** The owner sits in the window at the ground stand at half size and turns: the dune crests' edges are judged
there (D5); at dusk the stars are judged (D6); then the talk turns to the dunes rising ahead of a fast low hull.

## B34. T2 CLOSES: THE TWO LOOKS GIVEN, THE READINGS RED AND MOVED TO T4, THE WINDOW CRASH CURED (2026-10-02)

The window session of B33, flown by the owner at the ground stand and at the dusk stand (`VD_TEMPORAL=1
VD_RENDER_SCALE=0.5`, Docker's machine stopped):

1. **D5 GIVEN** (the edge at half size, still and turning; the owner: **"Looks amazing"**).
2. **D6 GIVEN FOR TODAY'S SPRITE** (the stars at dusk as the temporal path draws them: today's star sprite after the
   reconstruction, through the air table's transmittance, about 2–4 % dimmer than the default path's marched light, no
   star lost; the owner: **"Looks amazing!"**). The other choice, the law's exact light over each pixel, is not built and
   not needed. G-STARS's light line reads against this light from now on (the −2.1 % still and −3.7 % turning against
   the marched default are the accepted change).
3. **THE READINGS, MEASURED AND RED** (a poller read the client's own frame row five times a second):
   - the frame at full screen (3 456 × 2 168, the first attempt only; the owner: "I simply forgot" the switch later):
     23.1 ms at the median, about 43 frames a second, against B22's 60;
   - G-NO-STALL: a frame over 33 ms in 18–36 % of the seconds at the ground stand and in every second at the dusk
     stand; 146 ms at the
     dusk boarding, with one pipeline compiled in the same moment; 94 ms at the full-screen switch; three pipelines
     compiled after the loading at each of the two later stands (none in the first attempt);
   - three causes UNMEASURED: the AirPlay stream to a second display (the system's encoder runs all the time), the
     poller's own five state reads a second, and the cluster on the same machine.
4. **THE READINGS GO TO T4, A CLEAN READING FIRST** (owner: "To T4, clean reading first"): T4 starts with a reading with
   AirPlay off, a lighter meter and the cluster's load known; the cures follow (every pipeline warmed before play, the
   boarding's frame, 60 frames a second at full screen).
5. **THE WINDOW CRASH, FOUND IN THE SESSION AND CURED:** a window that shrank in one frame under the render picture of
   the frame before stopped the client (wgpu: a copy of 720 rows from a picture of 678 rows). The reconstruction's frame
   decision read the render size of the frame before, because it stood beside the render scale with no command sync
   point between them. Now it runs after the render scale; a test reads the engine's schedule and is red on the old
   order. The owner's check by hand reached 756 rows, where the old order could not fail either.
6. **ONE COMMIT AFTER A CHECK** (owner: "One commit after a check"): the session's record, the frame meter and the crash
   cure, on `game`, with no push.
7. **THE WAVES WAIT FOR THE PROPER TERRAIN** (owner: *"I'd prefer not to work on the waves cure before we actually build
   the proper terrain - as then we might apply different approaches"*).

**Example.** The pilot drags the window's bottom edge up in one quick move, from 1 440 rows to 678: before the cure the
client stopped; now the reconstruction copies the 678 rows that this frame drew.

## B35. THE GROUND STEP'S PLAN: THE PICTURE SETS THE QUALITY, THE WEATHER BEFORE THE FREEZE, A REALM CALCULATES AND DRAWS ITSELF (2026-10-02)

The owner showed the third reference picture again (*"we want to achieve this believability"*), asked for the plan in
detail, corrected its frame (*"Make sure that it's not only about the bioms or things we have in the picture, I gave it to
you so you understand the quality we're trying to achieve"*), asked where the clouds and the weather stand, and ruled:
**"Ok, agree with the plan then. Please get into the context the ground rules and laws before designing the plan.
Remember that realm calculate and draw itself"**. Read as:

1. **THE PICTURE SETS THE QUALITY, NOT A LIST OF THINGS.** Seven qualities hold everywhere, at every height (on foot, in
   low flight, high up and from orbit), on every body:
   - structure at every scale, from ranges of 100 km to stones underfoot, with no scale left as plain noise;
   - the land makes sense: rain on any point reaches a sea or a closed basin along valleys that join;
   - each form shows the process that made it;
   - the surface follows the shape (bare rock on steep ground, soil in hollows, the tree line, the snow line);
   - variety in one view, with natural borders between kinds of country;
   - the light and the air (shadows at every distance, light in crevices and under trees, rich materials, haze, clouds);
   - detail to the horizon with no flicker, at 60 frames a second at full screen.
2. **THE CATALOGUE: EVERY KIND OF COUNTRY THE BODY'S OWN FACTS CALL FOR**, each at that quality: rivers and erosion, ice,
   wind (real dunes), coasts, fire (volcanoes), rock control, crust movement, impacts, slopes; every biome the climate
   makes; other bodies (a dry world, an icy moon, an airless moon) from the same laws. T5's "until priced" stands: the
   design prices each family before any of them is built.
3. **THE CHECKS:** a stated rule chooses the stands (one for each mix of biome, relief and rock kind, plus random
   stands) on the home planet and on two more bodies; each quality has numbers that can go red; the owner's look judges
   against the three reference pictures' quality.
4. **THE ORDER:** phase 0, ONE DESIGN WITH REVIEWS (the laws with their calibration, every cap that goes (B23), the frame
   cost of each phase, the bodies, the checks, the quality bar and the catalogue) → 1 the shape's foundations (the small
   hills' average, the flat directions) → 2 R2 → 3 R3 and the slopes → 4 the processes → 5 the biomes and the ground's
   materials → 6 the cover (plants, stones, debris) → THE LIGHT AND THE SURFACE'S LOOK (the far shadows join the ground
   step; amends B6.5) → 7 the owner's look (G-LOOK) → the waves (B34 item 7) → the water fill (B22 step 4) → ONE WEATHER
   STEP → the edit system (B22 step 5).
5. **ONE WEATHER STEP, AFTER THE WATER FILL AND BEFORE THE EDIT SYSTEM:** the almanac, the weather list and its lane
   (T3), the clouds and their shadows, rain and snow on screen, the wind on hulls and trees, lying snow after a storm,
   wet ground after rain. The ground step's design keeps frame time free for the clouds from phase 0. **AMENDS T1 items
   10–11 and T4:** the clouds, the rain on screen and the forest come BEFORE the freeze.
6. **A REALM CALCULATES AND DRAWS ITSELF** (SL3, the owner's reminder): every product of the ground step and the weather
   belongs to the realm it describes. The planet's own shard calculates its solve, its finer levels and its weather
   (both hosts for the static shape, SL10), and the client draws them as that realm's own look. A parent positions its
   children and never calculates or draws them; a moon calculates and draws itself.
7. **THE LAWS FIRST:** the design starts after the binding rules and laws are read into the work's context.

**Example.** A pilot crosses the home planet: a desert of star dunes, then a canyon cut in red beds, then a glacier valley
under a snowy range, every one of them at the picture's quality and every one made by the same laws. Above them the
planet's own clouds move by its own weather list, and their shadows run over the ground. On an airless moon the same
code makes only craters and regolith under a black sky, calculated and drawn by the moon's own realm.

## B36. THE GROUND STEP'S DESIGN: THE FIRST FIVE ANSWERS (2026-10-02)

The phase 0 design of B35 is written (`docs/investigation/2026-10-02/ground_step_design.md`; workflow `wf_883767e3-3f5`:
four angles, one synthesis, three adversarial reviews with 70 findings, all answered; read-only). Of its 18 questions,
five were put to the owner as the ones that change a ruling or need the owner's number. The owner answered:

1. **Q3 — THE CIRCULATION THAT FOLLOWS THE SUN: ALLOWED** (owner: "allow"). Under the home planet's 55° tilt the law's
   circulation is the seasonal cell that crosses the equator and follows the summer sun (Lindzen & Hou 1988). **AMENDS T5**
   ("no monsoon"): a circulation the law produces is the law. No stand, biome class, dune type or rain-shadow stand rests on
   the climate until the corrected climate runs (the design's M0-9).
2. **Q7 — THE FIRST VISIT: ONLY BY LAW, NO MANUAL LAW CHANGES** (owner: "only by law, no manual law changes"). Read as: no
   registered exception to B16 for a planet nobody visited; the 1 km ground arrives by the laws already ruled (the bake as
   the realm's own warm-up along the paths it authors, the interest grown with the closing speed of 2026-08-27, never a
   cap on the speed), and if the measured bake (the design's M2-5) cannot win the race, the answer is another LAW (a faster
   bake, a different split of the work), brought to the owner with its numbers — never a stand-in and never a hand-made
   exception.
3. **Q10 — THE TARGET MACHINE IS THE OWNER'S CURRENT MAC** (owner: "current mac, i can't tell tbh"): an Apple M4 Pro, 14
   CPU cores (10 performance, 4 efficiency), 48 GB of memory, its 20-core GPU — until the owner names another machine.
   The terrain's share stays half of the cores (B20). Amends F6's "the Mac is the measuring instrument, not the target".
4. **Q14 — THE FOREST AND THE ART: DISCUSSED WHEN ITS TIME COMES** (owner: "we will need to discuss when time comes"). The
   design's forest bench with stand-in trees (M0-15) waits for that discussion too.
5. **Q17 — THE PLANET'S SUN: (a), THE ORBIT STATED ONCE** (owner, after a plain explanation: "q17-(a)"; earlier: *"we
   should not have any seams - only laws"*). The star system states the planet's orbit elements and spin law ONCE, on
   change, stamped (SL1 clause 2); the planet evaluates its own sun from the universe tick by Kepler's law (T3 class 1).
   **A NAMED EXCEPTION TO SL1 CLAUSE 3, fenced:** the planet uses the result only for its own sun (the weather, the melt,
   the light); the placement, containment and crossing machinery never reads it (SL1 clause 5); nothing states it on any
   lane. The other roads were refused: the sun's direction sent at every weather step is a per-child row SL9 forbids, and
   no sun is a missing law.

6. **Q1 — THE FAR VIEW'S TWO RULES AND THE SMALL HILLS' AVERAGE: YES, AS RECOMMENDED** (owner: "Yes, as recommended";
   answers B22 Q5). RULE M, THE AVERAGE: at every rung the drawn height, the material shares, the shade and the canopy
   cover equal the mean of the finer rung over the cell (a zero-mean detail may drop at a swap; any other term stays as
   its cell mean), with G-MEAN, G-CLOSURE, G-SHADE-NEST and G-FAR-EXACT. RULE S, THE OUTLINE: a compact form taller than it
   is wide (a pillar, a tower, a needle, a fin, a sea stack, a volcanic neck) is a placed form of the static shape — both
   hosts compute it from its law, the cells draw it while it is at least two cells wide, its own coarser meshes draw the
   same outline farther out, and under one display pixel it is a tint of its true share — with G-SILHOUETTE (at most one
   display pixel at the hand-over). THE ZERO-MEAN LAW IS A GATE FROM NOW ON, and today's ridged hills are NOT centred:
   phase 2 replaces them, so one world change and one re-freeze are saved.

7. **Q2 — EARTH, THE MOON AND MARS AS THE MEASURE: YES, IN PHASE 0** (owner: "Yes, in phase 0"). Public elevation data
   (SRTM and GMTED2010 for Earth, open lidar tiles for the small hollows, LOLA for the Moon, MOLA for Mars) and public
   maps of climate zones and rock kinds, read by OUR OWN instruments into the checks' reference numbers (the design's
   M0-4). The data stays OUTSIDE the repository and only numbers are recorded; each licence is checked first; no request
   carries the owner's name, e-mail or address; a source that needs an account waits for the owner's word.

8. **Q16 — THE STANDS AND THE BODIES: YES, AS RECOMMENDED** (owner: "Yes, as recommended"). A stated rule picks the
   stands ONCE in phase 0 and pins them (later phases add stands, never move one): class stands (climate zone × relief ×
   rock kind, at the middle of the class's largest area), family stands, line and border stands, random stands (24 on the
   home planet, 20 on land and 4 on the sea floor; 8 on the moon; 12 on the third body), and today's pinned stands. Four
   views (on foot, the reference view from a VALLEY stand toward the ranges — the old summit pose kept as a stand —, 150
   km up, orbit) and two suns on every reference view (12°–18° and about 40°–60° up). Three bodies in every phase's
   pictures: the home planet, the home moon, and a third body chosen by the charter rule. **AMENDS B5 item 1:** G-ICE
   tests the ice law on EARTH'S REAL heights and climate (Q2's data), never on an Earth-like test body (B23 item 3).

9. **THE THIRD BODY: THE DESERT WORLD** (owner: "The desert world"; phase 0, bench M0-5). The first census of the home
   system's 24 round bodies (9 planets, 15 moons) found the rule's pick to be a 25 407 km rocky world (5.3 g, 28.7 bar,
   456 K, no water) whose lattice reaches the edge ceiling: 25.2 million nodes of 19 456 m, against the home planet's
   8 192 m (MEASURED: the node at a face's centre, the target met exactly). THE RULE GAINS ONE CONDITION: the third
   body's node is the one the 8 192 m TARGET sets — its lattice's edge is the divisor rule's answer with no memory
   ceiling — as on the home planet, so every check reads it at the same scales. The rule then picks `Planet(17782101822166660494)`: 7 393 km, 12.65 m/s² (1.3 g),
   1.68 bar, 350 K (77 °C), no water, 11.9 million nodes of 8 192 m. Found on the way: an icy moon with thin air,
   `Planet(61143963952882988)` (1 368 km, 555 Pa, water ice, 55 K) — the real icy body the design's F46 asks for before
   any icy-moon law is built.

THE REMAINING QUESTIONS ARE ASKED WHEN THEIR ANSWER CHANGES THE WORK, one by one with a plain explanation (owner: "do
we need to answer all questions now? If yes, let's do one by one, with proper explanations"): Q1, Q2 and Q16 before
phases 0 and 1; Q4, Q5, Q6 and Q8 when phase 1 ends; Q9, Q11, Q12, Q13, Q15 and Q18 at their own phases, with their
measured numbers.

**Example.** On the home planet the summer sun stands over the north in one half of the year and over the south in the
other, and the rain belt follows it: a valley that is wet in its summer is dry in its winter, and nobody wrote that in by
hand. A pilot who comes to a planet nobody visited meets ground made only by its laws; if the 1 km ground could not be
ready in time, that is a defect to solve with a law, not a rule to switch off.

## B37. THE CAUSE OF THE LOW MOUNTAINS, AND WHERE ITS CURES GO (2026-10-02)

The owner asked whether the home planet is "not good enough" (*"No hills, no mountains, no divercity?"*), chose "Earth
first", and then "Find the cause". Phase 0 measured (`docs/investigation/2026-10-02/phase0_earth_relief.txt`,
`docs/investigation/2026-10-02/phase0_relief_cause.txt`; DEFERRED D-TERRAIN-10):
- at the 8.2 km grid Earth holds no hills either (the scale, not the world); away from its closed basins the home planet
  holds one fifth of Earth's mountains (0.47 % of the land against 2.39 %) and about half of Earth's relief;
- NO cap binds; THE BELTS BARELY RISE (the largest uplift over the age 769 m, because a belt's room is the relief less
  the planet's TALLEST platform: 2 323 m); THE EROSION TURNS THE BELTS INTO LOWLANDS (it removes 8.4 times their uplift
  over the system's whole age; they end 478 m over the sea with no mountains); the erosion's TIME is the lever, but a
  shorter time alone keeps crater rims and plate steps, not ranges;
- THE REBOUND LEAVES GRID STEPS (one lift per 131 km block: 74 m at a block's edge against 46 m inside; the control 0.96).

The owner ruled (*"Design now, build in phase 2"*; *"Commit now"*):
1. **THE BELTS' LAW AND THE LANDSCAPE'S TIME ARE DESIGNED NOW** (phase 0), with benches and Earth's measured numbers as
   the reference, for the owner's review: a belt rises from its plates' convergence over its own active time, limited at
   each node only by the crust's strength (the law-limit that stays); each landscape erodes for the time since its ground
   was last raised, not for the system's whole age.
2. **THEY ARE BUILT IN PHASE 2's MACRO CHANGE, BEFORE THE FIRST BAKE OF THE 1 km GROUND**, so the world changes once.
3. **THE REBOUND AS A SMOOTH FLEXURE GOES WITH THE FLAT DIRECTIONS IN PHASE 1** (a grid-direction defect).
4. **THE BELTS' ROOM IS A CAP** (B23) and goes with the belts' law; the design's caps list carries it as row 53.
5. One commit now, on `game`, no push.

**Example.** Today a pilot who follows a folded belt of the home planet flies over a lowland 478 m over the sea, with
flat plateaus a kilometre higher beside it. After phase 2 the same belt rises by the convergence of its two plates over
its own active time, and its rivers have worked on it only since it rose: the pilot sees a young range cut by valleys.

## B38. ONE BENCH; NEW LAWS ONLY WHERE THEY ADD REALISM; NO CAPS AND NO ARTIFICIAL CHANGES (2026-10-02)

Two adversarial reviews of the belts' design (`docs/investigation/2026-10-02/belts_and_time_design.md`, revision 2) found
the cure larger than B37's two laws: the limits the new belts would hit, the erosion law at the 8.2 km grid, the
platforms' height, the margins, dry bodies and the time axis. Asked how to continue, the owner ruled:
*"Ok to build one bench, but the main laws and rules should not be broken. Also we should not build the sim, physic laws
should decide, but they whould also be right ones."* — and, asked which reading is right: *"If you want to add a new
physic law, it's ok if it will add more realism, but we should not implement caps or artificaial changes."* Read as:

1. **ONE BENCH IS BUILT**: a copy of the macro solve in which each item of the design's §3 can be switched IN THE COPY
   only; its control reproduces the pinned artifact digest; nothing in the game changes.
2. **THE BINDING LAWS AND RULES STAY UNBROKEN** in the bench and in what it proposes (HR1–HR6, SL1–SL10, B16, B21, B23,
   T9; no test seam in the game).
3. **A NEW PHYSICAL LAW IS WELCOME WHERE IT ADDS REALISM**, and it must be the RIGHT law: published, applied as its source
   states it, with its own calibration body.
4. **NO CAPS AND NO ARTIFICIAL CHANGES**: no clamp, no tuning knob, no hand-made adjustment that makes a number match.
   Earth's numbers are READINGS to compare against (B23), never a target a constant is turned toward. A diagnostic run
   (such as the cause bench's erosion a tenth as strong) is a measurement, never a candidate.

**Example.** The bench may test whether a belt's relief follows the relief law measured on Earth at the 10 km scale,
because that law is published and calibrated on Earth's own ranges. It may not narrow the platforms' crust scatter until
the home planet's hypsometry matches Earth's: that would be a knob turned toward a reading.

## B39. THE WORLD CONVERGES IN TIME: A BINDING GATE (2026-10-03)

The belts' bench (ruling B38; `docs/investigation/2026-10-02/phase0_belts_bench.txt`, the design §10) found today's solve
NOT converged in time: the same laws on 40 / 80 / 160 even passes give mountains 0.47 / 0.11 / 0.02 % of the land away
from closed basins and land over 1 000 m 63 / 55 / 43 %; on the geometric axis 9.60 % at 40 passes and 6.12 % at 80. The
world's relief is partly a numerical error of 40 long passes, and accurate integration grades today's laws nearly flat.
The owner ruled (*"Yes, a binding gate"*; *"Commit now"*):

1. **THE WORLD'S LOOK MUST NOT DEPEND ON THE PASS COUNT.** A binding gate: the solve's readings at N and 2N passes agree
   within a stated tolerance, and the solve uses as many passes (or as fine a step rule) as the gate needs. The solve's
   own rule (03 §4.6, "the passes are a resolution, not a dial") becomes a measured gate.
2. **IT COMES WITH THE NEW LAWS** (phase 2's macro change, ruling B37 item 2), because converged today's laws grade the
   planet flat: the belts at Earth's rates, each landscape's own time, the erosion constant's calibration, then the gate.
3. **THE COST IS ACCEPTED**: a slower first solve (MEASURED: 160 passes took 311 s against 150 s for 40).
4. One commit now, on `game`, no push.

**Example.** Today a pilot sees a crater rim stand high on the home planet only because the solve's long first steps
under-eroded it. After phase 2 the solve takes as many steps as the gate needs, so the rim stands as high as its own age
and its rock allow, and a second solve with twice the steps shows the pilot the same rim.

## B40. THE BELT WIDENS UNDER A FORCE BALANCE; THE PLATES' MEAN SPEED IS 50 mm/yr (2026-10-03)

The belts' bench (`docs/investigation/2026-10-02/phase0_belts_bench.txt`) showed that a belt rising at Earth's rates takes
in more rock than today's erosion removes, and that a per-node collapse cannot carry the excess away; and that the world's
plate speed constant ("Earth's mean plate speed, 50 mm/yr", `craters.rs:133-138`) multiplies a uniform share whose mean is
½, so the mean is 25 mm/yr and the ridges open at 21 mm/yr against Earth's 53. The owner ruled:

1. **THE FORCE BALANCE, AND THE BELT WIDENS** (owner: *"Force balance, belt widens"*; the design's Q-B2 (b); AMENDS B37
   item 1's "limited at each node only by the crust's strength"): a belt's MEAN height stops where the plates' force can
   hold it (Molnar & Lyon-Caen 1988; England & Houseman 1986; Earth's numbers give about 5 km, Tibet's measured mean is
   5 023 m), and the extra rock WIDENS the belt into a plateau; the width grows only by the influx less the erosion's
   efflux; at the ceiling the profile is a plateau. Peaks keep the strength law (Everest). The force is a law with its
   own calibration body (stated in the build), never a knob.
2. **A PLATE'S MEAN SPEED IS EARTH'S 50 mm/yr** (owner: *"Mean plate speed = 50 mm/yr"*; the design's Q-B1): the
   constant is corrected to its own stated meaning, checked against Earth's mean ridge opening of 53.4 mm/yr, and the
   craters' clock and the belts read the ONE corrected speed. A world change in phase 2's macro change.

**Example.** A belt of the home planet that reaches about 5 km over its lowland stops rising in the mean; each million
years after that its plateau grows wider by the rock its plates push in less the rock its rivers carry away, and its
peaks stay under Everest's strength. Its plates close at a speed whose mean is Earth's, so its active time follows a
plates' turnover near Earth's own.

## B41. THE GATE'S CAUSE FIRST, THE ANDEAN LAW'S RESEARCH BESIDE IT; PROPER LAWS, NEVER EARTH (2026-10-03)

The belts' bench, runs 9 and 10 (`docs/investigation/2026-10-02/phase0_belts_bench.txt`; the design §10 and §12): the
crust's partition at a collision (P2: the crust above the decoupling level stays, Replumaz et al. 2010) and the
collision's slowing (S2: `e^(−t/45 Myr)`, Clark 2012) cut the collisions' rock to a fifth and the plateau land by almost
half; the Andean margins now raise 81 % of the belts' rock; THE GATE (B39) FAILS ON THE LAND'S HEIGHT — 214 → 414 passes
move the mountains by 0.3–1.0 point but the land over 1 000 m by 8–10 points under both erosion laws. The owner ruled
(*"Cause + Andean research"*; *"Commit now"*):

1. **THE GATE'S CAUSE FIRST**: measure where a finer step erodes more (by river size, height and rock), find the cause,
   cure it; the erosion-law decision (F49) waits for it. **BESIDE IT** a research agent reads the sources for the Andean
   margins' law (the rate from the plates' own motion, the height from the megathrust's force); it puts no load on the
   machine. Each result comes back with its numbers.
2. **THE TWO BELT LAWS** — the owner: *"It's ok, but from where do we have India in our home world? Are you trying to
   build Earth? Or we simply trying to get proper laws that we would apply to the home world."* THE ANSWER GIVEN: proper
   laws, never Earth. A law reads only the body's own facts (its plates, its crust, its speeds, its lithosphere); an
   Earth place only measures the one constant a law cannot compute yet — its CALIBRATION BODY (rulings T9, B38), as
   Everest fixes the strength law's constant and Tibet the force balance's. India–Asia is the only collision whose crust
   budget is measured, so it fixes the decoupling depth; Tibet fixes the slowing time. These two constants stay Earth's
   until the world states a heat law per body (the decoupling level is a depth set by the crust's temperature; the
   slowing time is the mantle lithosphere's viscosity), then become each body's own — an interim, registered
   (D-TERRAIN-10). **The owner's word on this answer is owed before the two laws are recorded as adopted.**
3. One commit now, on `game`, no push.

**Example.** The home planet holds no India. When two of its own continents meet, the belt keeps the crust above the
decoupling level — 20 km down in the home planet's own lithosphere — and the rest sinks into its mantle; the pilot sees
the home planet's own plateau, its width set by its own plates' speeds. India's only role is the one measured number
behind the 20 km.

## B42. THE ROUTING'S MECHANISM FIRST; THE THREE DEFECTS CURED IN THE COPY; THE ANDEAN MARGINS PARKED (2026-10-03)

The gate's cause, MEASURED (runs 11–13; `docs/investigation/2026-10-02/belts_and_time_design.md` §14): the pass ledger
shows the rivers' cut is the term that moves; the sweep's own step is not the cause (eight sweeps a pass on one routing
move the cut 5 %); THE NUMBER OF ROUTINGS IS (1 / 2 / 4 / 8 routings a pass: the cut 588 / 802 / 991 / 1 103 × 10¹⁵ m³,
the land over 1 000 m 35.7 / 21.0 / 8.4 / 0.9 %, the land's mean height 874 / 540 / 269 / 101 m — today's laws, routed
often, grade the planet flat, as B39 said); and the HISTORY does not settle even at one routing every 3 Myr, so each
routing seems to unlock about the same cut whatever the time between routings. The Andean research (§13): no published
rate law; the megathrust's force law fits Earth's margins but needs three facts the world lacks; no window law. The
owner ruled:

1. **FIND THE MECHANISM** (*"Find the mechanism"*): split each pass's cut by its source — the nodes whose drain changed,
   the lakes that drained, the flats that got a new outlet — so the measurement names which candidate unlocks the cut
   at each routing; then the cure is designed with its numbers and brought to the owner.
2. **THE THREE DEFECTS CURED IN THE COPY** (*"Cure them in the copy"*): the ice's cut stepped in time so a glacier
   shrinks as its bed falls, with no per-call thickness cap (ruling B23: a cap standing in for a law goes); the ice's cut
   booked for the rebound; the rebound run by time, not by pass count. Each cure measured against its control; the game
   changes only in phase 2.
3. **THE ANDEAN MARGINS PARKED** (*"Park, keep the share"*): today's shortening share (Norabuena et al. 1998) stays as a
   REGISTERED INTERIM (D-TERRAIN-10) until the gate converges, because the erosion decides how high these belts stand;
   the research (§13) is kept for the return.
4. One commit now, on `game`, no push. (B41 item 2's answer — proper laws, never Earth — still waits for the owner's
   word.)

**Example.** A pilot over the home planet's plains sees them stand 874 m over the sea on average when the solve looks
for the rivers' paths once a pass, and 101 m when it looks eight times. After the mechanism is found and cured, a
second solve with twice the looks shows the pilot the same plains.

## B43. THE BASE CURE AND THE JOINT LAW GO INTO PHASE 2; THE DEFECTS, THEN THE OLD PLAINS (2026-10-03)

The mechanism, FOUND (runs 14–15; `docs/investigation/2026-10-02/belts_and_time_design.md` §14): the game's sweep reads
a receiver as a lake wherever its flood level stands over its ground (`vd_terrain::solve::MacroSolve::routing_base`) —
true of every node the sweep has just cut — so the implicit sweep cuts toward the receivers' heights at the routing, and
a lowered node's news climbs one node per routing. Cured in the copy (a receiver is a lake only where the routing found
one), the routing converges at first order; today's laws then grade the planet flat; the full law set with the solve's
own erosion drowns the old plains (land 6.8 % of the globe), with the joint law it keeps them (land 17.5–19 %, the land
over 1 000 m 20–21 % against Earth's 20.8 %, the mountains about six times Earth's), and its gate nearly passes. The
owner ruled:

1. **THE BASE CURE GOES INTO THE GAME'S SOLVE IN PHASE 2** (*"Yes, in phase 2"*): the sweep reads a receiver as a lake
   only where the routing found one — a defect's cure with no knob; it lands with phase 2's new laws (ruling B39),
   never alone, because alone it grades the home planet flat.
2. **F49 RULED: THE JOINT LAW** (*"The joint law, F2"*): phase 2's erosion is Adams et al. 2020's joint calibration,
   `E = K_lp · Q · S^2.2` (`K_lp = 2.2 × 10⁻⁹ m⁻²`, `m = 1`, `n = 2.2`; calibration body: 142 basins of Bhutan and
   central-eastern Nepal), its rock shares read against the crystalline basement; its authors' caveat (n not universal)
   is stated with it.
3. **NEXT** (*"Defects, then old plains"*): the three defect cures (the ice with no per-call cap, the ice's cut in the
   rebound, the rebound by time) measured on the cured base with the joint law; then the research on how Earth's old
   continents keep their height over billions of years (ruling B37's landscape time), because with a correct sweep the
   old plains still sink to about 110 m.
4. One commit now, on `game`, no push.

**Example.** A pilot crossing the home planet's oldest plain sees, after phase 2, the height its rivers can really leave
it: the joint law cuts a steep valley fast and a gentle plain slowly, and the sweep sees each lowered valley floor at once,
so a second solve with twice the steps shows the pilot the same plain.

## B44. THE ICE CURE GOES INTO PHASE 2; THE OLD PLAINS RESEARCHED, THE REBOUND'S SPACING MEASURED BESIDE IT (2026-10-03)

The three defect cures, MEASURED (runs 16–17; the belts' design §14): the stepped ice first read every bed it had just cut
as water (the sweep's stale-flood rule again) and chopped every glacier — fixed in the copy (water only where the
routing found a lake); then the ice with no cap cuts more than the capped ice (4.8 against 3.5 × 10¹⁵ m³ on the land),
deepens the deepest valleys and keeps the gate's agreement; the ice's cut feeds the rebound only where a rebound follows
it; and THE REBOUND'S SPACING HAS NOT CONVERGED (the land over 1 000 m about 20 / 25.6 / 28.8 % with a rebound every
125 / 25 / 12.5 Myr). The owner ruled:

1. **THE ICE CURE GOES INTO PHASE 2'S SOLVE** (*"Yes, in phase 2"*): the ice stepped in time with no per-call thickness
   cap (ruling B23), a node water only where the routing found a lake, the ice's cut booked for the rebound — with phase
   2's laws, like the base cure.
2. **NEXT** (*"Old plains + rebound beside"*): the research on how Earth's old continents keep their height over billions
   of years (a read-only agent: the freeboard, the rock cycle, the supercontinent cycle, dynamic topography), and BESIDE
   it the measurement of where the rebound's spacing settles; both come back with their numbers.
3. One commit now, on `game`, no push.

**Example.** A glacier on a young range of the home planet now cuts its valley for the whole glacial epoch, step by step,
and stops when its catchment falls under the snow line — never at its own thickness each call; the crust under the cut
valley floats back as the rivers' cuts do.

## B45. THE EXACT REBOUND GOES INTO PHASE 2; THE LANDSCAPE'S TIME BENCHED NEXT (2026-10-03)

The rebound's spacing, MEASURED (run 19; the belts' design §14): the game's rebound floors four times (−0.125 m a node per
rebound); without the floors the land over 1 000 m settles from a spacing of 12.5 Myr (28.7–30.0 % down to 1.6 Myr) and
the gate passes with a matched rebound (29.09 / 29.06 %). The old plains, RESEARCHED (§15): Earth's old plains stand at a
mean of 485 m, the home planet's at 105–114 m before its belts rise; they do not erode too fast, they are low because
nothing raises them again for 4.8 Gyr (30 % of Earth's continental area was last raised under 0.35 Ga ago, Artemieva
2006). The owner ruled:

1. **THE EXACT REBOUND GOES INTO PHASE 2'S SOLVE** (*"Yes: exact, ≤ 12.5 Myr"*): the rebound without its floors, each
   node's fraction of a sixteenth carried, at a spacing of 12.5 Myr or finer — a resolution cure with no knob.
2. **BENCH THE LANDSCAPE'S TIME** (*"Bench the landscape's time"*; ruling B37 item 1): each plate's plains erode only since
   its ground was last raised; a reworking law gives each plate its raise time, with Earth's measured continental ages
   (Artemieva 2006) as the calibration body, replacing the plates' uniform age share (a lawful one-time draw of an
   unrecoverable history, ruling T9); read the old plains against Earth's 485 m.
3. One commit now, on `game`, no push.

**Example.** A plate of the home planet whose ground was last raised 300 Myr ago keeps young, high plains; a plate last
raised 2 Gyr ago has worn its plains low; a pilot crossing from one to the other sees the step in their ages, as on Earth
between a young platform and an old shield.

## B46. THE LANDSCAPE'S TIME GOES INTO PHASE 2 WITH THE LAKES' CURE; THE LAKES NEXT (2026-10-04)

The landscape's time, BENCHED (runs 20–21; the belts' design §15): each plate's ground erodes only since its raise under
the reworking law (`λ` = 1.019 per Gyr, Artemieva 2006); the old plains rise from 310 m (median 93 m) to 586–680 m (median
392–467 m) against Earth's 485 m (349 m); the mountains fall from 19.9 to 11.6 %; the land turns to tablelands (66–69 %
against Earth's 45.5 %); and the gate fails by the lakes (1 → 2 routings a pass move the land over 1 000 m 28.7 → 32.8 %,
all in the nodes held by a lake: a lake's level is set at a routing and fixed while the sweeps cut around it). The owner
ruled:

1. **THE LANDSCAPE'S TIME GOES INTO PHASE 2** (*"Yes, with the lakes' cure"*), together with the lakes' cure, because the
   gate must pass first.
2. **CURE THE LAKES** (*"Cure the lakes"*): a lake's level follows its outlet inside the sweep — when the sweep cuts the
   outlet, the lake's level falls with it — so the result stops depending on how often the solve routes; then the gate
   is measured again with the landscape's time.
3. One commit now, on `game`, no push.

**Example.** A river cuts the rim of an old crater lake on the home planet; in the same step the lake's surface falls
with its outlet, and the streams that feed the lake cut down to the new shore, instead of waiting for the next routing to
learn that the lake is lower.

## B47. THE LAKES' CURE GOES INTO PHASE 2; THE GATE'S TOLERANCE STATED; THE MOUNTAINS NEXT (2026-10-04)

The lakes' cure, MEASURED (run 22; the belts' design §15): a lake's level follows its outlet inside the sweep; the routing
dependence is gone (1 / 2 / 4 routings a pass: the land over 1 000 m 27.8 / 27.3 / 26.5 %, the old plains 564 / 554 / 553 m)
and the whole law set passes from 214 to 414 passes within about 2 points on every share and 5 % on the old plains'
height. The converged world: land over 1 000 m about 27–29 % (Earth 20.8 %), mountains about 11–13 % (Earth 2.39 %), the
old plains about 553–584 m (Earth 485 m), plains / tablelands about 21 / 67 % (Earth 52 / 45.5 %). The owner ruled:

1. **THE LAKES' CURE GOES INTO PHASE 2'S SOLVE** (*"Yes, in phase 2"*): a lake's level follows its outlet inside the
   sweep; the deposit fills each lake up to its level now — with the base cure, the ice cure, the exact rebound and the
   landscape's time.
2. **THE GATE'S TOLERANCE** (ruling B39 item 1, stated — *"2 points, 10 %"*): at N and 2N passes, and at twice the
   routings, every share (mountains, land over 1 000 m, plains, tablelands) agrees within 2 points and every height (the
   old plains' mean and median, the relief at 50 / 90 / 99 %) within 10 %. The current law set passes.
3. **NEXT: THE MOUNTAINS** (*"The mountains"*): measure which cause carries the 11–13 % (the belts, the edges of the
   widened plateaus, the craters), then research each one's law; the parked Andean margins are part of it.
4. One commit now, on `game`, no push.

**Example.** A second solve of the home planet with twice the steps and twice the routings shows a pilot the same old
plains within a tenth of their height and the same share of mountains within two points — so what the pilot sees is the
laws' answer, and the next work is to make that answer's mountains as rare as Earth's.

## B48. THE MOUNTAINS MEASURED; THE BELTS' SHAPE RESEARCHED NEXT (2026-10-04)

The mountains by cause, MEASURED (run 23 and the Earth instrument's run 4; the belts' design §16): the belts and their
plateaus carry 88 % of the home planet's mountains — they cover about Earth's share of the land (17.7 % against Earth's
orogens' and arcs' 16 %, CRUST 2.0) but are 49–68 % mountains against Earth's 8–12 %, and stand 1–1.8 km higher; the old
plains match Earth's (516 / 318 m against 485 / 349 m); the craters are minor (398 stamped, 1.2 % of the mountains — the
earlier "43 % of the land near a crater" counted craters a raise had erased, a readout flaw now fixed). The owner ruled:

1. **RESEARCH THE BELTS' SHAPE** (*"Research the belts' shape"*): why Earth's orogens at 8 km are only about 12 %
   mountains and 1.8 km high — the belt's cross-section (the critical-taper wedge, Davis, Suppe & Dahlen 1983), a
   plateau's low inner relief (Tibet), the forelands, and the Andean margins' height law (the parked A2) — compared with
   the bench's belt today (a smooth bump over two flexural parameters, a force-balance plateau with steep edges); then
   bench the laws found.
2. One commit now, on `game`, no push.

**Example.** On Earth a pilot crossing the Himalaya and Tibet flies over a steep front, then a high plateau whose inside
reads as tablelands at 8 km; on the home planet the same crossing reads mountains across half the belt. The research
asks which law makes a belt's inside a plateau or a foreland rather than a field of peaks.

## B49. THE BELT AS A CRITICAL WEDGE WITH A PLATEAU, BENCHED NEXT (2026-10-04)

The belts' shape, RESEARCHED and their slopes MEASURED (the belts' design §17; run 24): Earth's young belts read a median
steepest slope of 3.2 % at 8 km, their mountains in one narrow steep band a flank, their insides plateaus (Tibet's interior
96.5 % tablelands); the home belts read 7.7–10.8 %, about 3.4 times steeper in every height band; the joint law's steady
state explains the size (a node draining its own 67 km² under ~1 mm/yr of uplift stands near 10 %) — the ±240 km bump lifts
the whole belt at once, and the widened plateaus are cut by rivers that reach the sea. The owner ruled:

1. **BENCH THE WEDGE AND THE PLATEAU** (*"Wedge and plateau"*): replace the ±240 km bump with a belt that grows like a
   critical wedge (Roe et al. 2008) — the front moves out as the belt takes in rock, its surface slope set by the critical
   taper (Davis, Suppe & Dahlen 1983, calibrated on Taiwan), its crest rising to the force-balance ceiling, and a plateau
   beyond — so the uplift sits at the front, as on Earth; in the same run, read where the plateaus' rivers end (the sea or
   a closed basin).
2. One commit now, on `game`, no push.

**Example.** A collision on the home planet grows a ramp of about 2° toward the incoming plate and a steeper flank toward
the other; when its crest reaches the force balance's height, the extra rock widens a flat plateau behind it, and a pilot
crossing it climbs one front, flies over the plateau, and descends one flank.

## B50. THE CRITICAL WEDGE GOES INTO PHASE 2; A LEVEL PLATEAU NEXT (2026-10-04)

The belt as a critical wedge, BENCHED (run 25; the belts' design §17): each margin's cross-section a taper wedge sized by its
rock (Roe et al. 2008; Davis, Suppe & Dahlen 1983's taper calibrated on Taiwan), its crest to the force-balance height, a
plateau behind it — the mountains fall 11.7 → 5.4–6.2 % (Earth 2.39 %), the ramps near Earth's (the collision belts 20–30 %
mountains, the Andean 13–16 %), and the gate passes; the plateaus carry about 64 % of the mountains left, because each
plateau node stands at its old platform's height plus the force height, so the plateau copies the platform's tilt and
about 90 % of its land drains to the sea. The owner ruled:

1. **THE CRITICAL WEDGE GOES INTO PHASE 2'S BELTS' LAW** (*"Yes, in phase 2"*), replacing the ±240 km bump and the
   force balance's widening.
2. **NEXT: A LEVEL PLATEAU** (*"A level plateau"*): the plateau's surface stands at one level along its margin — the lower
   crust flows until the plateau is level (Clark & Royden 2000) — never copying each node's old platform; then read whether
   the dry plateaus' lakes close under the solve's own water budget, so inward drainage follows from the existing lake law
   with no new constant.
3. One commit now, on `game`, no push.

**Example.** The plateau behind a home-planet collision stands at one height from end to end, as Tibet does; where the air
over it is dry, its streams end in closed lakes instead of cutting to the sea, and a pilot flies over a flat high plain.

## B51. THE LEVEL PLATEAU AND THE WATER'S OWN PATHS GO INTO PHASE 2; THE ANDEAN MARGINS' LAW NEXT (2026-10-04)

A level plateau and the water's own paths, BENCHED (runs 26–30; the belts' design §17): one level a convergent pair takes
the mountains 6.20 → 5.45 % (5.24 % at 414 passes) and the gate passes; the joint law reads the RAIN (Adams et al. 2020:
`Q = A·R`, "the product of drainage area and mean upstream rainfall"), so the budget's runoff is not its water (the old
plains would stand at 1.1–1.2 km); the law's own rain on the water's own paths — a closed lake ends its rivers, the lakes
are the budget's standing water, a closed basin keeps its rock and fills from its floor — changes the mountains little
(5.80 %; 5.56 % at 414), raises the old plains (median 356 m against Earth's 349 m, mean 593 m against 485 m) and the land
over 1 000 m (25 → 32 %), widens the closed basins that reach under the sea (2.7 → 7–8 % of the land) and slows the solve
by about 30 %; inward drainage does not follow from the lake law on the wedge's smooth plateaus (Earth's closed plateau
basins are contractional: ranges rise across the rivers' paths in a dry climate, Sobel et al. 2003). FOUND: 87 % of the
plateau land stands on Andean margins (8.8 % of the land, about half the mountains left; the central Andes' plateau is about
0.44 % of Earth's land) — the registered Andean interim, whose return waited for the gate's convergence. The owner ruled:

1. **THE LEVEL PLATEAU GOES INTO PHASE 2'S BELTS' LAW** (*"Yes, in phase 2"*), with the wedge: each convergent pair's
   plateau stands at one level, the area-weighted mean of its wedge nodes' platform plus force height (Clark & Royden 2000).
2. **PHASE 2'S EROSION FOLLOWS THE WATER'S OWN PATHS** (*"Yes, in phase 2"*): after every routing the solve's lakes' budget
   and water tree give the receivers; the discharge is the joint law's own rain on the area that truly drains to each
   node (a closed lake ends its rivers); the lakes in the passes are the budget's standing water (the sweep skips only
   water and cuts a closed hollow's dry rim; the deposit fills a lake only up to its water); and the rock that reaches a
   closed hollow stays in it, filling it from its floor to one level, never over its spill. Ruling B8 (the rivers follow
   the water) now holds in the passes as at the solve's end.
3. **NEXT: THE ANDEAN MARGINS' LAW** (*"The Andean margins' law"*): research, then bench, a law for the Andean margins'
   rock and ceiling in place of the registered interim (Norabuena et al. 1998's share at every ocean-under-continent
   margin for its whole window, the collisions' force as its ceiling).
4. One commit now, on `game`, no push.

**Example.** A home-planet river that runs off a dry range into a closed salt lake no longer cuts the gorge below the
lake's rim as if its water went on; the rock it carries stays in the lake's basin and fills it. A pilot who flies inland
from a coast where an ocean plate dives under the continent will cross the plateau the margin's own law builds — on most
of Earth's such coasts, one range and no plateau.

## B52. THE BUILDING STARTS NOW; THE ANDEAN LAW IS RESEARCHED BESIDE IT (2026-10-04)

After runs 9–30 of phase 0's belts' bench, the owner asked: *"You are doing investigations for quite a lot, when the
implementation part will start?"* Every ruled law of phase 2 has no open question left except the Andean margins' law,
which has a registered interim. The owner ruled (*"Build now, Andean beside"*; amends B51 item 3's order):

1. **THE "BEFORE" RECORD FIRST**: today's numbers on the pinned stands, headless (no window), as the baseline every later
   phase's census compares with. The other phase-0 items (the climate bench M0-9, the rest of M0-4, the window session)
   move into the phases that use them.
2. **THEN PHASE 1** (the ground step design §5.1): the flat directions (Barnes, Lehman & Mulla 2014's two fields, one
   potential) and the smooth rebound (B37 item 3), with the G-MEAN gate.
3. **THEN PHASE 2'S MACRO CHANGE** with every ruled law (B39's convergence gate; B43 the cured base and the joint law; B44
   the ice cure; B45 the exact rebound; B46–B47 the landscape's time and the lakes' cure; B42/B41 the crust partition and
   the collision's slowing; B50 the critical wedge; B51 the level plateau and the water's own paths), each in the
   generator's own integer form under the float fence (SL10 clause 4: the copy's powers, exponentials, logarithms and
   tangents get integer forms), one change at a time with its census.
4. **THE ANDEAN MARGINS' LAW IS RESEARCHED BESIDE THE BUILD** by a read-only agent; its bench and its build join phase 2
   as one change. Until then the registered interim stands.

**Example.** The pilot's next landing on the home planet comes after phase 1 (the flats drain without the grid's lean);
after phase 2 the ranges are critical wedges, the plateaus level, and a dry range's rivers end in closed salt basins.

## B53. BUILD PHASE 2, THEN ONE LOOK AT THE END RESULT (2026-10-04)

Phase 1 was built and checked (generator 23: the flats' potential, the smooth rebound; the census
`docs/investigation/2026-10-04/phase1_census.txt`), and its pictures were offered for the owner's look. The owner:
*"It's very hard to judge at the moment - I didn't see the mountain region or anything else. If we still need to build
in the phase 2, let's do it and then look at the end result please."* So:

1. **PHASE 2 IS BUILT NEXT**, every ruled law in the generator's own form (B52 item 3), one change at a time with its
   census; phase 1 stays as built under it.
2. **THE OWNER'S LOOK COMES AT THE END RESULT**, with stands that hold a MOUNTAIN REGION and the other kinds of country
   in frame — never only the plains around today's five picture stands. The frozen picture references stay until that
   look; phase 1 and phase 2 are judged together.
3. No commit until the owner asks for one.

**Example.** The next pictures the owner sees stand over a collision belt's wedge, an Andean plateau, the old plains and
a coast of the solved home planet, each before and after, instead of five views of the same plain.

## B54. THE GATE'S SENSITIVITY: THE CURE FIRST, THEN A NOISE RULE; 214 PASSES UNTIL THE CURE (2026-10-05)

Phase 2's census and the gate's cause were measured (the census document's gate section; DEFERRED D-TERRAIN-10): a defect
was found and cured in the game's solve (17 % of all the rock the rivers cut vanished past the closed hollows' spills; now
a hollow full of rock passes the rest on and the rock is kept exactly); with the rock kept the land's heights agree from
414 to 814 passes but not from 214 to 414; and the plains' share carries its own noise of about 2.6 points at one
resolution (the chaos test: the axis's last pass 4 % longer), over B47's 2-point tolerance — the young plateaus'
drainage layout is sensitive. The owner chose:

1. **THE CURE FIRST**: find whether the sensitivity is numerical (the suspect: the routing rations how fast a valley
   captures flat plateau ground — the solve without the water's own paths converges), and cure it if it is.
2. **THEN A NOISE RULE**: each reading's own noise measured at one resolution; where it exceeds the tolerance, N and 2N
   passes agree within that noise.
3. **214 PASSES STAY** until the cure (a cure may make the heights converge on the solve's own axis).

**Example.** Plate 10's plateau, raised 223 Myr ago, is dissected by one valley system in one run and by a valley a node
away in another; the cure must make how fast the valleys capture the plateau a fact of the law, not of the routing's
spacing.

## B55. THE BAND FROM THE ARTIFACT'S HEAD (SL6 YES); THE LEVEL HANDOVER AND THE FAR CULL MEASURED NEXT (2026-10-05)

The skyline's walls, cracks and lids were measured on the pictures' probe: phase 2's summits rise over the band's top
(+9.3 km on the home planet; the solved field reaches +12.0 km), where a chunk's halo reads air and its core reads rock.
The client holds the artifact's head and its coarse levels whole but the fine heights only in the tiles near the eye, so
no host can scan the whole field but the shard. After plain explanations (the owner: *"If we go with 2, will it be the
same on all machines? … will it produce 'waves' while ship is flying?"*; *"What is the AAA MMO games approach?"*; *"Will
the quality significantly improve, if we will go with more detailed?"*), the owner: *"Agree, go"*:

1. **THE BAND'S TWO ENDS IN THE ARTIFACT'S HEAD (SL6 YES)**: the shard states the solved ground's lowest and highest
   point once; every host — the shard, the gateway, the client — reads them from the head and none computes its own, so
   the band is one value on every machine and never changes while a ship flies (it changes once, when the artifact
   lands, as the solved ground already does).
2. **THEN MEASURED, PLANET-WIDE, HEADLESS**: the level-of-detail handover's pops on steep ground (the slope law the solved
   ranges break by up to six times) and the far summits the culling drops.
3. **PER-TILE OR PER-POINT RANGES** (the AAA terrain pattern: bounds travel with the data) only where that measurement
   shows a defect, with their measured price.

**Example.** A summit of the belt 12 km over the home planet's radius is built whole — no wall at its chunk's edge, no
flat lid — and every machine builds the same chunks, because each reads the same two heights from the head.

## B56. THE SOLVE'S AXIS IS 414 PASSES (2026-10-05)

B54's cure was looked for and measured (the census's gate section items 12–14): a second defect in the rock's way was found
and CURED — a full hollow whose side's water ran back into it sent the rest round for ever (a bench run spun 90 minutes);
now the rest passes each hollow once and no rock is lost (six runs: no way refused). The sensitivity itself is NOT
NUMERICAL: the chaos twins' spread on the plains share is 1.4 / 1.6 / 5.0 points at 214 / 414 / 814 passes (plate 10's
young plateau carries it), so no finer axis cures it. Under the noise rule (B54 item 2: the tolerance the larger of B47's
and the reading's own twin spread; the axes compared by their pairs' means) the gate FAILS 214 → 414 on the heights beyond
their noise (land over 1 km −3.0 points, the old plains' mean −11 % and median −16.5 %, plate 5 −24 %) and PASSES 414 → 814
on every reading (every share within 0.4 points, every height within 4.5 %). Asked which axis, with the price (every first
solve about 1.6 times as long: the home planet about 22 → 35 minutes, the desert world about 35 → 57), the owner chose
*"414 steps (Recommended)"*:

1. **THE SOLVE'S OWN AXIS IS 414 PASSES** — the mixed axis twice as fine as today's (the last pass 0.25 Myr, the longest
   12.5 Myr, the climate's spacing in years kept); B54 item 3's 214 passes retire. B39's gate stands: 414 against 814.
2. **THEN THE WORLD MOVES ONCE**: the game's schedule, the generator's version and the artifact key's generation to 25,
   the pins re-recorded; then the census and the pictures on the three bodies, and the owner's one look (B53).

**Example.** A pilot who lands on the home planet's old plains finds them at the height the laws give, not 70 m higher
because the solve took long steps; and a second solve with twice the steps shows the same plains, while the rivers on
plate 10's young plateau may run a valley over, as real young rivers do.

## B57. PHASES 1 AND 2 LOOKED AT; R2 NEXT, ITS SLICE 0 FIRST (2026-10-06)

Generator 25 was built, checked (the release pin gate, the bench's control, a cluster's shard on the same digest) and shown
on the 32 pinned stands of the three bodies beside generator 22 (the owner's page; the census's generator-25 table). The
pictures showed the large forms moved — the ground under the fixed eyes rose by up to 12.5 km where the belts and plateaus
stand, a range now on the desert world's skyline — and the near ground still dominated by the dune-like noise under the
8 km grid (B6's root), the views from 150 km mostly haze, the moon unmoved. Asked for the look and the next step, with the
recommendation R2 (one eroded 1 km level sent to players), the owner: *"Yes, proceed"*:

1. **THE LOOK AT PHASES 1 AND 2 IS GIVEN** (B53's one look at the end result); the world stands at generator 25. This is not
   G-LOOK: the final look comes at the ground step's end (B35).
2. **R2 IS NEXT, IN THE GROUND STEP DESIGN'S ORDER** (`docs/investigation/2026-10-02/ground_step_design.md` §5.2): ITS
   SLICE 0 FIRST — the bench (M2-1: R1's loop as the control; the new operators without caps under B23; the bake's rate;
   the fine relief under G-MEAN) and a shaded relief at 100–150 km on the three bodies for the owner's look (B8 item 3),
   before any bake service, any world change or any data crossing a realm. The questions the design holds for R2 (Q5's
   parts, Q6 the far hand-over, Q8 the prebake, the SL6 asks A1 and A2) come to the owner with their measured numbers at
   the slice that needs them.
3. No commit until the owner asks for one (B53 item 3 stands).

**Example.** The next picture the owner sees is a shaded relief of a 524 km tile of the home planet's belt from the bench:
valleys a few kilometres apart that join toward the sea, where today's pictures show rounded dunes. Nothing in the game
changes until the owner rules on what that picture and its numbers show.

## B58. THE MACRO ROUTING'S GRID LEAN CURED FIRST, BEFORE R2's BAKE (2026-10-06)

R2's slice 0 measured on the erosion bench (the owner's page https://claude.ai/artifact/LnTGvq541tsrib68TvFGyd): the 1 km
joint law passes Hack's law on the belt tile (n 0.29 → 0.52); on a smooth cone with no grid in it, D8 runs the water down
the grid's rays (the valleys' lean 52 %, none 22 %), D∞ gathers its valleys into an eight-armed star (its dispersion), and
Rho8 cuts valleys at every angle (24 %); and the generator-25 macro ground carries the grid's mark on every pinned tile —
straight trunks with side valleys at 45°, coasts stepping along the node rows, regular bands on the belt tile — which any
1 km ground keeps, since the 1 km level holds each 8 km node's mean. Asked what comes first, the owner: *"I would take the
recommendation which will remove the strait or 45 percent angled rivers (or staicased), as it hurts believability"*:

1. **THE MACRO ROUTING'S GRID LEAN IS CURED FIRST.** Rho8 (Fairfield & Leymarie 1991: one receiver of the D∞ facet, chosen
   with the facet's angle as its chance by the node's own hash — the method phase 1 already uses on the flats) in the macro
   solve's water routing and in its erosion sweep; measured in the bench copy first (the macro ground's shaded relief and
   grid lean on the pinned tiles and on the cone, the census, the convergence gate at 414 against 814 passes), then built
   as generator 26 (a new solve, new pins, the gate again). AMENDS B11's D∞ for the macro passes: the cone shows D∞'s star.
2. **THE TARGET IS THE OWNER'S LOOK**: no straight rivers, no rivers at 45°, no staircased rivers or coasts. The readings
   (the valleys' grid lean, G-GRID by class) report it; the shaded relief shows it; a cause the routing does not remove
   (the stepped coasts may have their own) is measured and brought back with its numbers.
3. **THEN R2's 1 km BAKE** on the clean macro ground; slice 0 continues (the steady state, the duration).

**Example.** Today a river on the home planet's old plains runs 60 km due south in a straight line, with side rivers joining
it at exactly 45°. After the cure the same river meanders with the ground's own slope, and its side rivers join it at the
angles the land gives them; the 1 km valleys then cut into that river's own bends.

## B59. THE FIRST LAND'S RELIEF AND MARGINS BENCHED; THE WORLD MOVES ONCE (2026-10-06)

B58 item 2's causes were measured and brought back (the owner's page https://claude.ai/artifact/LnTGvq541tsrib68TvFGyd,
sections 5–6; the status file's 2026-10-06 entries): the descent's Rho8 takes the rivers on a lattice line 32.7 → 22.5 %
(none 22.2 %) and the straight runs of 82 km or more 10.5 → 2.5 %, and the two marks that remain have ONE ROOT — the first
land. 87 % of its nodes stand on an EXACT FLAT (deep in a continent the crust share is exactly one, so the Airy height is
one number for the whole plate), and it holds no relief between the 8 km node and the affinity noise's shortest wave
(375 km). The bench's history of the Andean tile shows the first rivers born straight on that smooth ground and kept by
their own valleys for 1.4 Gyr; a ±12 cm tie-breaking diagnostic does not move them. The stepped coasts are the plates'
steps one node wide (21.9 % of the boundary coastline stepped against 0.4 % inside a plate; each node's cell mean 8.4 %;
diagnostic margins of ±40 / ±80 km 3.2 / 1.4 %). A read-only research agent
(`docs/investigation/2026-10-06/platform_margin_research.md`) found NO published law for a platform's relief when its
erosion starts (the best-founded choice a named crust-thickness scatter, Earth's stable continents its calibration), the
margin laws by kind, and that the per-plate crust scatter [0.7, 1.4] names no source and is about 5.6 times Earth's
stable continents' spread. The owner chose every recommendation:

1. **BENCH THE FORMS OF THE FIRST RELIEF, THEN THE OWNER CHOOSES FROM THE PICTURES**: (a) a named crust-thickness scatter
   inside each plate through the model's Airy isostasy; (b) the same scatter through flexure (the body's own elastic
   thickness and gravity); (c) the older landscape's relief by Earth's measured topography spectrum. Each is read for the
   exact-flat share, the rivers' lean and straight runs, the Andean tile's picture, the old plains' relief against
   Earth's readings (a check, never a target) and the convergence gate.
2. **THE MARGINS BY KIND, BY THEIR PUBLISHED LAWS**: each node holds its cell's mean of the plate terms; a rifted margin's
   width by the necking instability (Fletcher & Hallet 1983, as Chenin et al. 2018 apply it); an Andean coast over the
   crossing of the plate interface and the upper plate's Moho (Ruff & Tichelaar 1996; the dip 23 ± 8°, Heuret et al.
   2011); a transform margin at Earth's measured necking of about 50 km (Mercier de Lépinay et al. 2016), a named
   calibration until a law exists.
3. **THE PER-PLATE CRUST SPREAD**: the bench also measures a SOURCED spread (Earth's stable continents') beside today's
   [0.7, 1.4], and shows the steps where two continental plates meet.
4. **THE WORLD MOVES ONCE**: generator 26 = the descent's Rho8 + the chosen relief + the margins, with new pins and the
   gate (B39, B56), then R2's 1 km bake on the clean ground (B58 item 3).

**Example.** Today the plate under the Andean tile starts as one flat at 3 489 m, and its first rivers run due west in
straight lines that its plateau keeps 1.4 Gyr later. With a first relief the same plate starts with low swells and hollows
a few hundred metres high; its first rivers wind between them, and the plateau rises over valleys that bend. Its coast no
longer steps one node at a time: the continent's crust thins toward the trench over the forearc's own width.

## B60. THE FIRST RELIEF IS THE OLDER LANDSCAPE'S, BY EARTH'S SPECTRUM (2026-10-06)

B59's bench round was measured (the owner's page, section 7; the status file's 2026-10-06 entry): every world solved over
5 Gyr with Rho8, the margins by kind, each node's cell mean and the sourced spread, and one of three first reliefs, each also
at twice the passes. The readings: rivers on a grid line (a) 19.2 %, (b) 23.3 %, (c) 19.4 % (none 22.2 %); straight runs of
82 km or more from 1.0 / 4.0 / 1.1 % of the river nodes (Rho8 alone 2.5 %); the old plains' relief at 110 / 1 200 km 190 /
425, 141 / 228, 175 / 269 m (Earth's stable continents 131 / 328 m, a check); the gate (414 → 828 passes) closest to passing
for (a), two readings over the tolerance for (c); on the Andean plateau (a) removes the bands and (c) keeps some. The
recommendation was (a); the owner chose **(c)**:

1. **THE FIRST RELIEF IS THE OLDER LANDSCAPE'S RELIEF** — a surface field with the continents' published spectrum
   (β = 2.09, Gagnon, Lovejoy & Schertzer 2006), its size set by Earth's stable continents' height difference of 131 m at
   110 km (a named scatter for an unrecoverable history, ruling T9), weighted by each node's crust share. It joins
   generator 26 with the descent's Rho8, the margins by kind (each node's cell mean; the Andean wedge rising from the law's
   coast, the trench's inner wall) and the sourced spread (B59 item 4).
2. **THE GATE FOR THIS WORLD** (B39, B56): the 414 and 828-pass solves with the full margins, each with its twin for B54's
   noise rule, before the world moves.
3. **THE SECOND CLIFF IS MEASURED NEXT** (B58 item 2): a coast 200–345 km inside a plate, smooth on the first land, ends after
   the solve in a cliff about 3.5 km tall; today's world has it too. Its cause and its numbers come back before generator 26.

**Example.** A pilot over the home planet's old plains sees low rises and shallow basins a few hundred metres apart in
height over a hundred kilometres — the older landscape the plate carried when its erosion began — and rivers that wind
between them toward the sea, where today the same plains start as one exact flat.

## B61. THE BELTS' LAW READS THE CRUST SHARE; THE GATE'S CAUSE FIRST (2026-10-06)

The owner asked whether world (c) floods the plains (*"Wait, in C - the plain will be covered by water?"*): measured, the
first (c) run kept only 18.4 % land because the bench added the older landscape's relief after the freeboard law had solved
the continents' thickness — a defect in the bench, corrected (the law's bisection now reads the relief; (c) solves to
24.6 % land, the plains and mountains stands dry land again). B60 item 3's cliff was found: the belts' law raised its wedge
only where a node's crust share is at least one half, so the Andean plateau stopped dead at the continent's own half-share
line in a cliff about 3 km tall (also in today's world). Benched with the law reading the share continuously, both coasts of
the Andean tile draw one smooth line, the coast slope's 99th percentile falls 19° → 10°, and the twin runs tighten; the gate
(414 → 828 passes under the noise rule) passes every share, the relief and the old plains, and misses two plate means
(plate 5 −10.2 %, plate 13 −12.3 %). The owner chose both recommendations:

1. **THE BELTS' LAW READS THE CRUST SHARE CONTINUOUSLY** in generator 26: a node of crust share `s` takes in `s` of the
   margin's rock, and the room's profiles blend the continental and the oceanic profile by `s` — the cell's mean of the
   law's own variable, in place of the yes-or-no test at one half.
2. **THE GATE'S CAUSE FIRST**: the bench measures why plates 5 and 13 — old, low plains — stand 21–34 m lower on the finer
   axis (which nodes, which process) and brings the cure or a stated finding back before the world moves.

**Example.** Plate 1's continent ends southward over a hundred kilometres where its crust thins. Today the Andean plateau
stands 4 km high to the line where the crust is half continental and drops 3 km in one node; with the law reading the share,
the plateau's edge falls over that same hundred kilometres as the continent thins, and the coast below it is one line.

## B62. EIGHT SWEEPS A PASS; THE CLOSED BASIN'S CAPTURE CURED INSIDE THE SWEEP (2026-10-07)

B61 item 2's cause was measured (the owner's page, section 10; the status file's 2026-10-07 entries). Two numerical causes
carry the gate's miss on the old, low plains of plates 5 and 13: (1) THE SWEEP'S STEP — the joint law's one implicit step
per sweep under-erodes decaying high ground; with 8 sweeps a pass the 414-pass world reaches the 828-pass value on plate 5
(185 against 188 m) for about 15 % more time; (2) THE CLOSED BASINS' CAPTURE WAITS FOR THE NEXT ROUTING — with the sweep
step equal, the bench's ledger by plate shows only the rivers' cut differing (2.2 % more on plate 5, 6.5 % on plate 13 on
the finer axis); the finer axis keeps less land in closed basins from 1.5 Gyr ago on, basins open 17 % more often and 2.2 %
more rock reaches the sea, because a basin can open only when the water is routed again, once a pass. The owner chose the
recommendation:

1. **EIGHT SWEEPS A PASS** (`SUB_SWEEPS` 2 → 8; a resolution under B39, B41 item 1, B45 — the rebound's spacing finer).
2. **THE CAPTURE CURED INSIDE THE SWEEP**: as B47 made a lake's level follow its outlet inside the sweep, a closed basin's
   state follows its rim inside the sweep — when a river cuts the basin's way out below what the basin holds, the basin
   passes its water's rock on at once, not at the next routing. Designed and benched first; then the gate again (414
   against 828 passes, with the twins).

**Example.** A river on plate 13 cuts back into the rim of a dry hollow the older landscape left. Today the hollow keeps
filling with its own rock until the next routing, up to 12.5 Myr later; after the cure the hollow's rock follows the river
out in the same sweep in which the rim falls below it, on either axis.

## B63. PLATE 13's CAUSE FIRST; A SINKING FOREARC IS DEEP SEDIMENT; THE WINDOW NOW AS A PREVIEW (2026-10-07)

B62's gate was measured (the status file's 2026-10-07 entries): with 8 sweeps a pass and the capture inside the sweep,
thirteen of fourteen readings pass under the noise rule — plate 5's mean among them (+4.2 %; it failed at −10.2 % before)
— and PLATE 13's MEAN FAILS: 261.5 → 230.5 m from 414 to 828 passes (−11.9 % against the 10 % tolerance; its twins 2.7 %
apart), against −12.3 % before the cure, its loss on its belts and plateaus. Generator 26 was built in the game (not
committed, the pins not moved): its first land equals the benched world's byte for byte. A finding came with it: the
Andean trench's inner wall makes the continent's outer margin sink, and the province rule named sinking continental crust
"rift basalt" — 12 609 nodes of the home planet (generator 25: none), so a forearc strip took basalt's hardness, and the
rock map's own test ("a rift with no rifting") went red. The owner chose every recommendation:

1. **PLATE 13's CAUSE FIRST**: the bench's ledger by plate measures which process differs on plate 13 between the two
   axes (as it did for plate 5), and the cure comes back with its numbers; the world's pins wait for the gate (B39, B60
   item 2).
2. **A SINKING FOREARC IS DEEP SEDIMENT**: continental crust that sinks where plates push together fills with sediment,
   like the trench beside it; rift basalt stays only where plates pull apart. A small world change in generator 26,
   measured in the next gate run.
3. **THE WINDOW NOW, AS A PREVIEW OF GENERATOR 26**: the generator-25 cluster stops, the release binaries are built with
   generator 26, the home planet's shard solves its new artifact (about 55 minutes), and then the owner is asked to open
   the window (the window is never opened without that word). Bench runs for plate 13 run beside it; no frame timing is
   read that day.

**Example.** On Chile's coast the land between the trench and the coastal hills is a slope of mud and sand that the rivers
carry down; under this ruling the home planet's Andean forearc wears like that sediment, and a pilot finds basalt only in
the rift valleys where its plates pull apart.

## B64. THE FIRST SOLVE ON MORE CORES, THE SAME BYTES (2026-10-07)

The owner asked whether every planet solves for about an hour when a player first reaches a new system (yes, today: once
per planet in the world's life, the planets of a system in parallel, one core each; the first visitor sees the seed's
own ground replaced when the artifact lands — a seam), and then: *"Can we give more cores for passes to build the realm
for the first time?"* The answer given: the passes stay one after another (each reads the ground the one before left),
but the work inside a pass can run on many cores — the first land, the climate, the rebound and the routing's receivers by
node, the sweeps by river basin, the priority flood and the lakes' hierarchy by published parallel methods (Barnes 2017)
— an ESTIMATE of 2.5× on 10 cores for the easy steps (55 → about 21 minutes for the home planet), 4–5× with the hard
ones. The owner chose:

1. **BUILD THE PARALLEL FIRST SOLVE NOW, MEASURE AFTER**: in the game's solve, beside plate 13's running bench runs;
   its speed measured once they end (one job at a time for timings).
2. **THE SAME BYTES ON ANY NUMBER OF CORES**: every thread writes only its own nodes and every sum keeps one order, so
   the artifact's digest does not move — a pure speed change, proven by the pin gate; no world change.
3. **RUST'S OWN THREADS**: the standard library's scoped threads, no new dependency; the generator keeps its rule (the
   recipe, the seed leaf, nothing else).

**Example.** A pilot warps toward an unvisited star. Its planets start their first solves on a share of the host's cores;
the home-planet-sized world that took 55 minutes on one core takes about 20 on ten, with exactly the same rivers and coasts.

## B65. FOUR ROUTINGS A PASS; THE GATE READS THE PLATES' OLD PLAINS (2026-10-07)

B63 item 1's cause was measured (the status file's 2026-10-07 entries): with the sweep's step equal, the rivers cut 7.1 %
more on plate 13 and 10.3 % more on plate 5 on the axis with twice the routings, every other process equal; and with the
routings counted apart from the passes (a bench switch, the water routed R times a pass), PLATE 13 FOLLOWS THE ROUTING COUNT
ALONE — at 1 656 routings over the age 414 and 828 passes give its mean 214 and 215 m, its cut 36.10 and 36.19 million
km³ — and converges (each doubling adds less: +7.7 %, then +3.5 %); plate 5's old plains converge more slowly (345 → about
317 → about 299 m). Each routing finds the basins and the water's ways afresh, so more routings let the rivers capture the
old plains' hollows sooner. And a plate's whole-land mean swings with the dry basins under the sea level (plate 5's reads
61 to 365 m under one law). The owner chose both recommendations:

1. **FOUR ROUTINGS A PASS**: the solve routes the water four times a pass, each routing (with its budget) followed by two
   of the pass's eight sweeps; the uplift, the talus, the ice and the peak law stay once a pass. A world change in
   generator 26; its cost about +64 minutes a home-planet first solve on one core, which the parallel solve (B64) cuts.
   Then the gate with twins: 414 against 828 passes, both routed four times.
2. **THE GATE READS EACH TRACKED PLATE'S OLD PLAINS' MEAN** (beside the shares and the relief), never its whole-land
   mean, whose dry basins under the sea level are the water fill's later lakes and read apart.

**Example.** A river on plate 13 works headward toward a hollow in the old plains. Routed once a pass, the solve learns
that the river has reached the hollow's rim only at the next pass, up to 12.5 million years later; routed four times, it
learns within a quarter of that, and the hollow's rock follows the river out, as it would on a real old plain.

## B66. PLATE 5's CAUSE FIRST (2026-10-08)

B65's gate was measured (414 against 828 passes, both routed four times, each with its twin; the plates read by their old
plains): every share, the old plains' mean and median, the relief and plates 1, 10 and 13 pass — PLATE 13 CURED (−3.2 %)
— and PLATE 5's OLD PLAINS FAIL: 300.5 → 257.5 m (−14.3 % against 10 %; its twins 1.7 / 4.3 % apart, so a numerical
effect, not noise). Its old plains by the routings over the age: 345 m (414), 305–329 (828), 294–303 (1 656), 252–263 (3 312)
— the last step the largest, so plate 5 does not converge as plate 13 did; the finer axis also halves the sweep's step and
the uplift's steps. The owner chose the recommendation:

1. **PLATE 5's CAUSE FIRST**: 414 passes routed eight times (the routing count alone against 828 × 4) and the ledger by
   plate on both axes routed four times (which process, where, when); then the cure with its numbers. The pins wait.

**Example.** Plate 5 is the home planet's oldest continent, its plains worn for 2.6 billion years. If its plains stand 40 m
lower only because the solve took finer steps, a pilot's view of them depends on the solve's resolution, not on the laws;
the cause decides whether finer routings, finer sweeps or a law's own form cures it.

## B67. THE WEATHER STEP DESIGNED NOW, BUILT IN ITS PLACE (2026-10-08)

The owner asked, while plate 5's solves ran: *"Is there any way how we can parallelize work? Maybe we can work on something
else in the other session? Clouds?"* Three facts were stated: B35 puts the clouds in the one weather step, after G-LOOK, the
waves and the water fill, so building them now changes a ruled order; today's terrain work is uncommitted with its pins not
final, so no clean base exists for a second worktree, and cloud code touches the sky and drawing files that change holds;
and the machine runs the solves, so a client build slows them and no timing reading holds while they run. The owner chose
both recommendations:

1. **THE WEATHER STEP IS DESIGNED NOW, AS DOCUMENTS ONLY** — the almanac, the weather list, clouds and their shadows, rain and
   snow on screen, wind and lying snow, as ONE design with reviews (B35's pattern: readers, one synthesis, adversarial
   reviews, then the owner's questions), joining the parts that exist (the climate design's §8, the ground step design's
   asks A9–A12 and questions Q17–Q18). No code, no world change; the BUILD keeps its B35 place.
2. **BY AGENTS FROM THIS SESSION**: read-only on the code, writing notes and the draft into the design folder; the session
   reviews each draft against the rulings and brings the owner's questions.

**Example.** While the solve works out why plate 5's old plains sink 40 m on the finer axis, the readers collect what the
rulings already say about a storm over the home planet's plains — who steps it (the planet's own shard), how it reaches a
pilot (the weather list on its own lane), and what it may never do (touch the land's shape) — so the day the water fill is
done, the weather step starts from a reviewed design.

## B68. PLATE 5's CAUSE FOUND; BOTH CURES BENCHED FIRST (2026-10-08)

B66's runs measured plate 5's cause (the bench's ledger by plate and pass, the field dumps with each node's cause and
where its water ends, and two splitting runs): the routing count alone carries about 11 m of the 40 m; only the rivers'
cut and the rebound differ between the axes; the plains sink on the same ground (not by a change of class), most in the
high plains beside the plate's dry basins. TWO NUMERICAL PARTS: (1) THE LONG HISTORY — at 50 Myr ago the calm plains read
323 m (414 passes × 4 routings), 287 m (828 passes, coarse sweeps), 276 m (414 passes, 16 sweeps and 8 routings a pass)
and 274 m (828 passes): the finer river step with the finer routing reaches the finer axis within 3 m; (2) THE END — the
sea and the climate are re-solved every 20 passes, so near the present the sea stood still for about 225 Myr while the
young belts rose (95 % of plate 5's uplift in the last 27 Myr); one re-solve then raised the sea 53–60 m (the land 30.2 →
24.2 % of the globe), 4.8 million km² of plate 5 left the connected ocean, and that re-solve fell 17 Myr ago on one axis
and 8.6 Myr ago on the other, so the plains' last recovery (+13 to +42 m) depended on the axis. The owner chose both
recommendations:

1. **BENCH BOTH CURES**: (a) THE RIVER'S STEP — 16 sweeps and 8 routings a pass, twice today's; (b) THE SEA AND THE CLIMATE
   RE-SOLVED AT EVERY PASS, so the land's change reaches the sea when it happens, on any axis. Then the gate: 414 against
   828 passes with both cures, each with its twin; beside it the sea's cure alone on both axes. Price: about twice the
   first solve (to be measured cleanly); the runs take hours.
2. **THE BENCHES FIRST**: the coverage gate stops and starts again by itself when they end.

**Example.** Plate 5's young range closes a strait, and an inland sea the size of the Mediterranean is cut off. Today the
solve notices it only at its next look at the sea, sometimes 9 Myr later, and the plains beside the new dry basin read
30 m higher or lower for that reason alone; re-solved every pass, the sea follows the range as it rises.

## B69. THE CHEAP CURE; GENERATOR 26 CLOSES BY A FIXED LIST (2026-10-09)

B68's two gates were measured (414 against 828 passes, each with its twin, under the noise rule): THE CHEAP CURE — the sea and
the climate re-solved at every pass, today's river step — PASSES on every reading (plate 5's old plains −9.1 %, inside even
the plain 10 %; the plains share +0.78 pt); BOTH CURES (plus 16 sweeps and 8 routings a pass) FAIL on the plains share
(+2.28 pt) and the tablelands (−2.86 pt), by about the size of the noise. The noise grew with the sea every pass: one twin of
a pair can end in another state (plate 5 269 / 304 m) through a late event — the young belts cut a large sea off plate 5 in
the last 5–10 Myr. The owner chose:

1. **THE CHEAP CURE** in generator 26: the climate and the running sea re-solved at every pass (`CLIMATE_EVERY` 20 → 1);
   the river step stays 8 sweeps and 4 routings a pass.
2. **THE NOISE RULE AS RULED** (B54, B56): each reading's tolerance is the larger of B47's and its own twin spread, one twin
   an axis; the late event's sensitivity is registered as a reading.
3. The owner asked: *"we are trying to solve this for couple of weeks now, we're not proceeding with the development of the
   game for a while now. Is there any guarantee that when we do this, we're done and we can proceed with other topics
   according to our plan?"* — answered: no blanket guarantee for the later steps, each of which has its own gate; for the
   macro solve the cause hunt is over, and GENERATOR 26 CLOSES BY A FIXED LIST — (1) the cure in the game's solve with its
   documents and tests; (2) a clean timing of a first solve, one job at a time; (3) the final pins (the game's own solve:
   the golden field, the digest, the sea, the identity) and the release pin gate; (4) the coverage gate green; (5) the
   census and the pictures on the three bodies for the owner's one look. Only a defect a gate finds, or the owner's look,
   reopens it; anything else that appears is REGISTERED and does not stop the move to R2's bake unless it breaks a hard rule
   or a ruling.

**Example.** A pilot over plate 5's old plains sees them at about 265 m whether the world was solved in 414 passes or in
828: the sea now follows the young range as it rises, pass by pass, instead of jumping 53 m at one late look.

## B70. AFTER GENERATOR 26: BEVY 0.20, EVERY BENEFIT, NOTHING BROKEN (2026-10-09)

The owner: *"When you're done and we ok to proceed, we need to migrate to bevy 0.20 and make sure that we are using all
benefits that bevy 0.20 is provide, while not breaking anything we have today."*

1. **THE ORDER**: when generator 26 closes by B69's list and the owner gives the word to proceed, the next work is the move
   from Bevy 0.18 to Bevy 0.20 (amends B35's order: it comes before R2's bake). It LIFTS the owner's upgrade gate of
   2026-08-27 ("no upgrade until a ship realm is flown in a window" — met: hulls 9, 10 and 11 flown in the window, B29 and
   B63) and sets the target at 0.20.
2. **THE SCOPE**: every crate that names Bevy — `bevy_ecs` is the ECS of every shard (sim, node, connection-plane, harness),
   the client and its renderer (`bevy`, `bevy_egui`), the vendored `bevy_pbr` with its eight atmosphere changes, and every
   crate the move pulls along (wgpu, naga, glam).
3. **EVERY BENEFIT**: each feature Bevy 0.19 and 0.20 offer that serves the game is listed with its price, and the owner
   chooses which to adopt (the owner's standing rule: no new library or feature adopted alone).
4. **NOTHING BROKEN**: the whole suite, the Tier-A coverage, the process gates, the pin gates, the frozen pictures (within
   their tolerance, or the owner's look) and a clean frame reading before and after.
5. **ITS DESIGN STARTS NOW, AS DOCUMENTS ONLY**, by read-only agents of this session (B67's pattern), while generator 26
   closes; the code changes only after the owner's word.

**Example.** A pilot flies hull 10 from the home planet's plains to orbit on Bevy 0.20 exactly as on 0.18: the same picture
within the frozen references' tolerance, the same crossings, the same frame cost or better — and the new version's features
the owner chose, such as a better shadow or a faster frame, on top.

## B71. THE FROZEN PICTURES FOLLOW QUALITY (2026-10-09)

The owner: *"If pictures quality improves in comparison to frozen ones, let's use new ones."*

1. **A CHANGE THAT IMPROVES THE PICTURES RE-FREEZES THEM**: each stand's new picture is shown beside its frozen reference;
   when it holds no new defect by the gates (no hole, no seam, no pop or flicker by the eye's rule, B27) and its quality is
   better, the new picture becomes the reference; the owner sees the pairs. This amends B28/B29/B32's "the references change
   only on the owner's acceptance" for an improvement; the owner's look still judges any picture the gates cannot.
2. **A WORSE PICTURE IS A DEFECT**, never a reason to re-freeze: its cause is found and cured, then the pair is compared
   again.
3. It applies first to generator 26's pictures (the world moved; the references are from generator 22) and to Bevy 0.20's
   quiet pixel changes (linear bloom, the white-furnace fixes, the sky table's mapping; B70).

**Example.** On Bevy 0.20 the dusk stand's sky gains its new table mapping: the horizon band is smoother and nothing pops on
the flight leg, so the new dusk picture replaces the frozen one, and the owner sees both side by side.

## B72. LESS WAITING: THE FASTER SOLVE AND A PARALLEL FEATURE TRACK (2026-10-09)

The owner: *"I want to improve the way we implement things. We are in testing mode for several weeks already without any
new features implementing and most of the time we are waiting. Any suggestions to improve that, while don't compromize on
quality?"* The causes were stated as measured: one home-planet solve 1.5–4 h on the bench and four of them a gate; about
three quarters of a solve on one core (the flood, the lakes' hierarchy, the deposit); the coverage gate 6–8 h, apart from
the solves; many steps waiting on an answer; T1 putting every other feature behind the whole terrain arc. Six changes were
offered; the owner chose two:

1. **THE FASTER SOLVE**: the flood, the lakes' hierarchy and the deposit on all cores (B64's owed items; the published
   parallel priority-flood), guarded by the same-bytes gate — a pure speed change, the world's bytes unchanged. It is built
   beside generator 26's close: the game's own solve records the pins while the code is written, and the release pin gate
   then runs on the faster solve and proves both the pins and the same bytes; the clean timing measures before and after.
2. **A PARALLEL FEATURE TRACK — AMENDS T1**: a feature that does not depend on the terrain's final look may move in its own
   worktree and session, with its own gates, beside the terrain work; the owner names the feature.
3. NOT ADOPTED: standing rules for routine decisions (each decision still comes to the owner), the bench's checkpoints and a
   nightly-only coverage gate, and (not now) a rented many-core machine.

**Example.** The release pin gate that seals generator 26 solves the home planet on all cores in a fraction of today's
hour, and gives the same digest as the one-core solve; meanwhile the block store's design moves in its own session.

## B73. THE FASTER SOLVE STOPS HERE; THE ROUTING'S UNREAD WORK IS SKIPPED FOR NOW (2026-10-09)

Generator 26's close (B69's list) measured: the pins recorded by the game's own solve on one core (the digest
[0x6f92_ab73_2b8f_e622, 0xab3a_2d52_064b_b265], the sea 3 660 m, the ocean 76.04 %), and the release pin gate GREEN on all 14
cores with the faster solve — the same bytes — in 2 179 s (one core: 11 383 s while the machine paged; not a clean timing).
On 14 cores half the wall time is still on one core (the budget's merge sweep about 15 %, the receiver order 6 %, the
routing's flat walk 5 %). FOUND: in a WET body the routing's flood, flat walk, order and discharge run four times a pass and
the budget's own water tree replaces all of them before anything reads them; only the routing's ocean label is read. The
owner chose:

1. **THE ROUTING'S UNREAD WORK IS SKIPPED, FOR NOW** (owner: "Skip it for now"): in a wet body's passes the routing does only
   what the rest of the pass reads — the ocean labelled at the sea's level, an ocean node's flood level its own ground — and
   the budget's tree does the rest; the bytes stay the same (the release pin gate proves it). The census keeps the routing's
   readings from the solve's final routing. "For now": the full per-pass routing may return if a later need reads it.
2. **THE FASTER SOLVE STOPS HERE** (owner: "Stop here"): generator 26 closes by B69's list — the coverage gate once, the clean
   timing, the census and the pictures; the steps still on one core are REGISTERED (the budget's sweep split by landmass
   with one renumbering pass, the receiver order, the flat walk, the tree deal, the water tree's spill walk).
3. **DOCKER STOPPED** (owner: "Yes, stop it") for the coverage gate and the clean timing (its VM held 16.7 GB and pushed the
   machine into heavy paging on 2026-10-09).

**Example.** The home planet's shard solves its world: each pass it labels the ocean, lets the budget's water find the lakes
and the rivers, and erodes — it no longer floods the whole planet four times a pass for a tree nobody reads, and the
artifact it stores is the same, byte for byte.

## B74. FEWER, FASTER TEST RUNS — ASK FIRST; THE VISUAL TESTS OFTEN (2026-10-10)

The owner, after a coverage gate of about 15 hours (most of it whole moon solves inside the tests, run under coverage
instrumentation): *"We indeed need to find a new strategy for this — we are waiting for tens of hours instead of building
the game."* Of five offered changes the owner chose:

1. **THE FASTER COVERAGE BUILD**: the instrumented tests with light optimisation — ADOPTED, after a proof on the terrain
   crate that it reports the same misses as today's build (optimisation can merge or drop counted regions).
2. **THE SHARED TEST SOLVES**: tests that need a solved moon share one solve per kind and test binary, on all cores (test
   code only) — ADOPTED.
3. **TESTS ONLY AFTER A SUBSTANTIAL CHANGE, AND ASK FIRST** (owner: *"run those tests only after substantial changes (like
   we finished a big implementation or improvement), and I prefer being asked before we are running any tests with
   information why you think we should run it"*): no test run — a suite, a gate, coverage — starts without the owner's
   word, and each ask states why it is needed now. This replaces the habit of running the suite after every edit.
4. **THE VISUAL TESTS MORE OFTEN** (owner: *"The only tests that I want to run a bit more often (but ideally in parallel if
   possible) is the tests to make sure that we don't break visuals (glitches, flickering, etc)"*): the picture and flight
   gates that catch a glitch, a crack or a flicker run more often, beside other work where the machine allows.
5. **NOT ADOPTED NOW**: the parallel feature track ("Not now"), per-crate coverage, a rented machine.

**Example.** After the faster coverage build lands, the agent asks: "the routing change is finished; may I run the terrain
suite (about 3 minutes) and the coverage gate (about 3 hours) to prove the same bytes and 100 % coverage?" — and runs them
only on the owner's yes; the flight gate that watches the coast for flicker runs beside the work after each change to the
drawing path.

## B75. LOOK FIRST: GENERATOR 26 CLOSES ON THE WINDOW LOOK; R2 AND THE SURFACE TOGETHER; BEVY 0.20 LATER (2026-10-10)

The owner looked in the window on generator 26's solved ground (the stand at 16.3° S, hull 12 berthed beside it), and
said: *"it looked good, but it still VERY far from believable terrain … Please don't run heavy tests. I'm interested in fast
proceeding with the game development, not waiting for several weeks till gates are fully passed."* The least believable parts,
in the owner's own choice: THE SHAPES (noise bumps, no valleys that join), THE SURFACE (one look everywhere: no rock on the
steep ground, no soil, sand or snow), and NO WATER AND NO COVER. Of three orders the owner chose "Look first":

1. **GENERATOR 26 CLOSES ON TODAY'S WINDOW LOOK.** B69's last item (the census and the pictures on three bodies) is
   REGISTERED as not done; the clean timing stays skipped (B69 item 6, the owner's "Skip the timing").
2. **THE MOVE TO BEVY 0.20 WAITS** (amends B70 item 1, "before R2"): its design (4.5–6.5 weeks, 90–160 machine hours of gates,
   no change to the ground) stays ready; it starts on the owner's later word.
3. **R2 STARTS NOW, BY ITS SMALL FIRST SLICE**: one region eroded at 1 km on the bench and a shaded picture of it for the
   owner's look, before any bake service, world change or data crossing.
4. **A FIRST SURFACE LOOK BY LAW, BESIDE R2** (amends B35's order, which put the materials after R3 and the processes): rock
   on the steep ground, soil in the hollows, snow where the climate law puts it — from the ground the client already holds and
   by laws the ground step's design names; nothing placed by hand, no new data across a realm boundary without the SL6 ask.
5. **CHECKS BY LOOK**: the owner's window look and headless pictures; the heavy gates only after a large change and on the
   owner's word (B74).
6. **THE PARALLEL FEATURE TRACK: NOT NOW** (B74 item 5 stands).

**Example.** Next the owner sees a shaded picture of a 150 km piece of the home planet's belt eroded at 1 km — valleys that
join and run to the sea in place of noise bumps — and, in the window, grey rock on a belt's steep flank, soil in the valley
under it, and snow on the crest where the climate law puts it.

## B76. THE PICTURES DECIDE FIRST, ALSO DURING THE BUILD (2026-10-10)

The owner, on R2's bench pictures after the hillslope law removed the grid's two-cell stripes at 256 m: *"There are still a
lot of strait lines. Not believable in my opinion. Are you sure we are using good algorithms?"* — then, on the proposed
order: *"Order is ok, we rely on the visuals first, even when we are building the game."*

1. **THE VISUALS DECIDE FIRST**: a ground step is judged by the owner's look at its pictures before its gates; a step that
   passes its numbers and does not look believable is not done. This holds on the bench AND during the build into the game.
2. **R2's ORDER**: the literature report on straight and grid-bound channels → the three suspected causes benched one at a
   time against today's pictures (the start from the game's own fine relief, not a smooth upsample of the 8 km field;
   routing that splits the water over many neighbours; the finer cell near the viewer) → the pictures side by side with the
   straight-line share as a number → R2's build into the game only on the owner's word that the pictures are believable.

**Example.** The belt's flank at 256 m shows no grid stripes but parallel straight gullies: that picture is not done. The
next picture starts from the flank's own fine relief and splits the water, and the owner compares the two before any bake
service is built.

## B77. LANDFORMS BY LAW BELOW 8 KM; THE MACRO WORLD'S WATER AND RISE MEASURED BESIDE (2026-10-11)

The owner, on the balance's readings (the land almost still — a present erosion rate of 0.00005 mm/yr at the median — and
dry — 1.5 % of the land over 1 000 mm of rain): *"Today's reading is enough, and it's not good enough. The terrain should be
very believable"*, with FOUR NEW REFERENCE PICTURES (kept outside the repository, `~/Downloads/voxeldust_references_2026-10-11/`:
a forested valley with a lake under a jagged massif with rock towers; a canyon with a stream between rock walls; a dry grass
plain with red layered mesas under a snowy range; an aerial view of snowy peaks, forested valleys, a red rock massif and
lakes in fog). Of the offered methods the owner chose:

1. **LANDFORMS BY LAW** (AMENDS B57/B58's R2 bake and the ground step design's R2/R3 as the way the forms under 8 km are
   made): the macro world still decides WHERE each kind of country is (the belts, the plains, the coasts, the climate, the
   large rivers); below 8 km THE ONE GENERATOR makes the forms directly, at each point, on both hosts and with no bake, from
   published laws — eroded mountains from the stream law's analytic solution on a drainage network built first, cliffs,
   mesas, buttes and towers from the rock beds' hardness and the cliff law, valley floors and river corridors from the
   discharge; the laws give the numbers (relief, valley spacing, cliff height); nothing is placed by hand (B23), the far
   view follows the rules of the average and the outline, and no form flickers (B16). The grid erosion benches stay as
   measuring tools.
2. **FIRST A DESIGN WITH REVIEWS** (the four references split into landform families, each with its law, its generator
   function and its checks), THEN ONE FAMILY ON THE BENCH with pictures beside the references, in days (B76: the pictures
   decide).
3. **THE MACRO WORLD MEASURED BESIDE** (the owner: "Measure it beside"): a read-only study of why the land gets so little
   rain and why the belts barely rise today, against the laws' sources; the owner then rules whether the cure is in the macro
   laws. The shape work does not wait for it.

**Example.** A pilot over the belt sees a sandstone plateau: the generator reads the beds' hardness at each height and the
cliff law, so the hard cap stands as mesas and red towers over soft slopes; where the climate law gives rain, the river cuts a
gorge with flat meadows on its floor — the same functions on the desert world and on the moon, with their own facts.

## B78. THE RAIN LAW'S CURE WAITS FOR THE LANDFORMS' FIRST PICTURES (2026-10-11)

The read-only study of B77 item 3 (`docs/investigation/2026-10-11/macro_rain_rise_study.md`) found the land's dryness NOT
the law's honest answer: a calibration defect (the rain belt's share never renormalised, so the law cannot give back its
own calibration body) and a missing law (the seasonal circulation of a 55° tilt, allowed by B36 Q3 and never built), plus a
wrong height law; and the stillness mainly the absence of uplift outside the belts and the scale effect of the joint law
read on 8 km node falls (its lawful cure is the law at its own scale, B77). Asked when to cure the rain (a world change:
generator 27, a new first solve, ~3.5–5.5 engineer-weeks by the design's estimate), the owner chose **"Later"**:

1. **THE LANDFORMS FIRST** (B77): today's rain and today's world stand while the landforms' design and their first bench
   family are made; the rain's cure is discussed again after their first pictures.
2. **REGISTERED, NOT FORGOTTEN**: the rain belt's unnormalised share, the missing seasonal circulation, the height law and
   the rain and lift clamps (B23) stay registered defects until that discussion.
3. The present uplift map (no world change) may be measured for the landforms when their design needs it.

**Example.** The belt's windward ramp still reads 96 mm of rain a year while the first landform family is benched; its
pictures show the forms the laws make on that dry ground, and the rain's cure comes back to the owner after that look.

## B79. THE LANDFORMS DESIGN'S FIRST FOUR ANSWERS; BENCH STEP 1 ON EARTH STARTS (2026-10-11)

The landforms design, revision 2 (`docs/investigation/2026-10-11/landforms_by_law_design.md`, three adversarial reviews
answered in its §7), asked ten questions. The four that gate bench step 1, answered as recommended:

1. **Q1 — EARTH'S GROUND AS A MEASURING STICK: YES.** A bench program runs the same generator code on Earth's own 8 km
   heights and rain (Taiwan, Oregon, a Colorado Plateau tile with its published rock layers) beside Earth's real ground;
   the data stays outside the repository; nothing in the game reads it; no constant is set from it (B36 Q2, B38 item 4).
2. **Q3 — THE LAW'S OWN WATER SHIPS (SL6 YES):** one byte per node of the joint law's water (rain × area over the solve's
   tree), 8.9 MB stored for the home planet, a tile +10 % (40 960 → 45 056 B), with one new first solve when it is built;
   the bench measures it against the shipped steepness and against today's runoff byte first. (The shipped discharge byte
   is the budget's runoff, not the law's `Q`.)
3. **Q10 — THE RIVER TREE IS A LAWFUL INPUT:** the ladder reads each row's receiver slot (its "no receiver" included) and
   never a sea or lake bit or a water level; the drainage network is the ground's own, so it is no water word in the solid
   pass (B21, one-volume §2.1).
4. **Q4 — NOTHING RETIRES BEFORE THE OWNER'S LOOK:** today's fine octaves, ridged band, roughness factor and cap-rock bench
   stay until a stand passes the owner's look; then they retire only on ground the ladder reaches; the moon, the desert
   world and the dry sea floor keep today's detail until their own families exist.

Q2 (the far view past rung 9), Q5 (the tall forms' outlines), Q6 (the steep-face hand-over), Q7 (the start ground), Q8
(the diagnostic under Earth's rain) and Q9 (the beds) come at their steps. **BENCH STEP 1 STARTS**: the method on Earth,
then the owner's look (B76).

**Example.** The bench builds Taiwan's Central Range from Earth's own 8 km means and rain with the same ladder the game will
run, and sets its shaded and painted pictures beside Taiwan's real 30 m ground with the straight-valley share of each.

## B80. THE STREAM LAW ON IRREGULAR POINTS: THE LOOK GIVEN AT MAP SCALE (2026-10-11)

After weeks of grid methods that kept straight valleys, 45° directions, stripes and combs (the owner: "Clearly you are
doing something wrong"; "We are on one step for several weeks now without ANY progress"), the owner named eleven open
sources to take from. Checked (code, licence, pictures): fastlem (T. Tada, MPL-2.0) runs the stream law's steady state on
IRREGULAR points; spacesim (MIT) holds a lawful seasonal climate for any tilt; the rest are ideas only or nothing (the
status file, 2026-10-11). The owner: "Please use whatever you can please, show me the pictures for judgement". The bench
linked fastlem unmodified for its point graph and ran the joint law's own steady state on it (`S = (U/(K_lp·Q))^(1/2.2)`,
the 35° threshold law, each node's uplift solved so the fine mean equals the node's own) on Taiwan, Oregon and the home
belt. THE OWNER ON THE PAIRS: **"I like it WAY BETTER! What's next?"**

1. **THE ROOT NAMED:** the regular grid under every erosion method below 8 km; the cure is the law on irregular points.
2. **THE DIRECTION BELOW 8 KM** is the joint law's steady state on an irregular point graph, driven by the macro world's
   node means (the macro says how high, the law makes the form); the earlier ladder and grid bakes stop.
3. **THE ORDER PROPOSED** (the owner's word on it owed; work starts on item 1): (1) the bench's visible defects (the 8 km
   squares, the tree that does not settle) and a view at eye height framed like the references; (2) finer ground — the
   same law inside each point down to 2–4 m; (3) a short design for the way into the game (where the solve runs, what
   crosses — SL6 —, how two regions meet, the integer form — SL10 —), the owner's answers; (4) the build on the belt
   stand, the owner's look in the window; (5) the rain (B78) from spacesim's seasonal structure, a world change.

**Example.** The belt's 8 km nodes stand 2.5–3.2 km over the sea; the law on 590 000 points 128 m apart cuts that plateau
into branching valleys at every angle, and each node's made ground keeps the node's own mean height.

## B81. THE FINE GROUND'S WAY INTO THE GAME: THE RECOMMENDATIONS ACCEPTED, NO SEAMS, GENERIC (2026-10-11)

The design `docs/investigation/2026-10-11/fine_ground_way_in_design.md` (revision 2: three adversarial reviews answered)
asked eight questions. The owner: **"Let's accept recommendations if they are not breaking ground rules and laws and if
we don't build seams — all our solutions should be generic and reusable."** Earlier in the step: "Current rungs and all
mechanic works quite good now (but not very performant). If we can simply change the terrain generation algos, that
would be good enough. If we can optimize without compromising quality, that would be even better."

1. **The recommendations ACCEPTED, as amendments named:**
   - **Q1:** below 8 km the ground is the TILE SOLVE — each tile solved once per world version and stored. This AMENDS
     B77's "no bake: forms per point". Under the finest level the ground is that level's smooth curve until a law level
     (2–4 m) exists; no noise is added there.
   - **Q2:** the planet's own shard solves, stores and ships its tiles (SL3, B35).
   - **Q3: SL6 YES** — the fine tiles cross from the planet's shard to its own clients through the gateway (one new bulk
     part variant; the client acknowledges held tiles).
   - **Q4:** while the realm is alive, its idle share solves L1 around its own arrival places. Idle work never keeps a
     realm alive.
   - **Q5:** on a never-solved place the coarser rung stands until the tiles are live.
   - **Q6:** a tile live under a standing body moves the body with the ground over the 0.5 s ramp.
   - **Q7:** the fine step bound and the band's bound are MEASURED first (P-SWITCH, P-BAND), then chosen.
   - **Q8:** F2 AMENDED for a new fine law — it may move unbuilt ground; built ground never moves (F1).
2. **NO BREAK OF A HARD RULE OR A STANDING LAW** (review A found none). The three amendments above are the owner's own
   rulings, named.
3. **NO SEAMS.** Every place where two pieces of ground meet gets ONE generic treatment, and G-SEAM is a BLOCKING gate on
   the bench: a seam seen means no build. Revision 3 removes three seam sources revision 2 still had:
   - the far levels' straight tile edge — the far levels are made from the stitched ground;
   - the edge of the rising region — the rise fades smoothly in the map, so the law fades the gullies;
   - the coast against the sea floor's ground — the solve's coast points take the sea floor's own drawn heights.
4. **GENERIC AND REUSABLE.** ONE tile machinery (levels in rung cells; solve, store, go live, ship, read, stitch) for
   every level, every body and every landform family; the fluvial law is its first family. Craters, dunes and the sea
   floor plug into the same machinery later.
5. **NEXT: B0, the bench only, no game change.** The decisive test first: two neighbour tiles solved apart and stitched,
   beside ONE solve of both together; then the speed under the fence, the point sets, the bytes, the bounds.

**Example.** Two L2 tiles on the home belt meet along a node's edge; the bench solves them apart and stitches them by the
cut, and solves the same 16 km square once whole; the owner sees the two pictures side by side, and the build waits
until no seam shows.

## B82 (2026-10-11) — the threshold by the level's scale; the straight valleys belong to the 8 km world

The owner, on B0 step 3's first readings (the 128 m tile 65 s → 5.2 s; the design's linear hold refuted; one threshold at
both levels gives 52 % of the 128 m ground over 45°; a 40 km straight valley on the belt stand from an 8 km river along a
node column): *"Agree, let's try"*.

1. **THE THRESHOLD SLOPE BY THE LEVEL'S SCALE.** At 128 m the threshold is tan 35° (0.70: Burbank et al. 1996 read the
   threshold hillslopes at ~32–35° on 90 m relief); at 16 m it is S_c = 1.25 (Roering et al. 1999, Oregon, read at the
   hillslope's own scale). A slope averaged over a 128 m link is gentler than over a 16 m one. AMENDS the design's A6
   ("one value at both levels"), refuted by measurement (52 % of the points over 45° at 1.25 against 0.18 % at 0.70).
2. **THE STRAIGHT VALLEYS ARE A DEFECT OF THE 8 km WORLD**, cured there beside the rain's cure (generator 27, B78): where
   an 8 km river runs along a node column, each node's own mean keeps the fine valley in that column. The seeded
   displacement of the macro reads stays OFF (it bends the line, it does not cure it).
3. **THE 128 m MEAN HOLD IS B80's ROUNDS** (the coarse start, three rounds, then the additive hold); the design's one
   linear solve (§3.3, B-B4) is refuted by measurement (6 min a tile, the worst node 1.25 km off, cliffs). AMENDS §3.3.
4. **NEXT, as stated:** why some tiles settle slowly, the cost under the fence (P-COST), then step 4.

**Example.** On the home belt a 128 m tile stands at most at 35° over each link, and its 16 m tiles reach 51° on their own
short links; the straight north–south valley the 8 km river draws along one node column stays until generator 27 cures the
river in the 8 km world.

## B83 (2026-10-11) — B0 step 4's choices; the 16 m level must refine the 128 m ground

The owner, on B0 steps 3–4's numbers (the 128 m tile 3.3–6.6 s on one core; the Rice coder within the size gate; no
added chunk at 128 m; the 16 m tiles stepping up to 974 m at the hand-over to the 128 m ground): *"Agree with the fix if
it's performant, seamless and will be equal on BE and client on all devices. Please continue with your plan"*.

1. **THE POINT SET IS THE HASHED POISSON-DISK** (Matérn's type II: four hashed candidates a cell, kept unless a higher
   hashed priority stands within cell/√π) — no lean, the straight share of B80's relaxed set, the same points in every
   tile.
2. **THE 128 m TILE SERVES RUNGS 7–12** from its own mean pyramid (amends the design's 7–9; every step from rung 7 to the
   node measured under the coarser cell's own one-pixel distance).
3. **THE BAND (B55) WIDENS BY THE LAW'S BOUND**: ±0.70 × the node's half-diagonal (±4.1 km on the home planet) over the
   rows — it holds before any tile is solved and never changes in flight.
4. **THE TILE CODER IS THE ADAPTIVE RICE CODE** (JPEG-LS's; measured within the size gate, every level decoded exactly).
5. **THE 16 m FIX, ON THREE CONDITIONS — PERFORMANT, SEAMLESS, AND EQUAL ON THE SERVER AND THE CLIENT ON EVERY DEVICE:** the
   16 m solve under the 128 m ground's own drainage and its mean held over each 128 m cell, benched with the rung-step
   reading (the hand-over step under a pixel at its switch), the seam test and the time, and built only under the fence
   (the same bytes on every host: P-SAME, G-NO-DRIFT). Before B3; B2 (the 128 m level) does not wait.
6. **NEXT: STEP 5** — the plain and the belt at 128 m for the owner's look; the 16 m fix benched beside.

**Example.** A walker on the home belt sees the 128 m valleys from 100 km away; as the walker comes nearer, the 16 m
gullies appear INSIDE those same valleys — the valley floor never jumps to another place when the finer ground arrives.

## B84 (2026-10-11) — the 128 m level is the law alone

The owner, on step 5's pictures and the recommendation to drop the exact mean hold at 128 m: *"Proceed"*.

1. **THE 128 m LEVEL IS THE LAW ALONE — NO ADDITIVE HOLD** (amends B82 item 3). MEASURED: the law leaves no pit; the
   additive hold dug hollows up to 678–1 245 m deep under their spill (4 % of the points over 1 m); four other hold forms
   failed (a fill-and-hold loop never converged, the slope hold steepened 10–22 % of the points over 45°, an inner-ring hold
   made it worse). Without the hold each node's mean differs from its row by at most 1 233 m — 0.15 of a pixel where the
   rows take over (7 510 km) — every rung step stays under its coarser cell's pixel distance, the band stays inside
   ±4.1 km, and the seam between tiles stays invisible.
2. **G-MEAN BECOMES "THE SWAP TO THE COARSER LEVEL UNDER A PIXEL AT ITS SWITCH"** (amends the design's "exactly zero"), the
   same rule at every level: W16's invisible rung boundary is the law, the exact mean was only one way to it.
3. **NEXT: THE 16 m FIX** (B83 item 5), judged by the same rule at its own hand-over (rung 6 → 7: under a pixel at 117 km),
   plus the seam test and the time, under the fence.

**Example.** A pilot 7 500 km over the home belt sees the 8 km rows; diving, the 128 m tile takes over and the massif's
mean may sit a kilometre off its row — 0.15 of a pixel at that distance, so nothing on the screen moves.

## B85 (2026-10-11) — the planet's shard ships the finished tile; the corner rule

The owner, after a plain explanation (tiles, the cut, the corner problem, who joins the tiles): *"Agree with your proposal
if we don't sacrifice quality"*.

1. **THE SHARD SHIPS THE FINISHED TILE** — the tile's square already joined to its neighbours by the cut, with its coarser
   copies (2 × 2 … 32 × 32) — in place of the raw tile with half its buffer and the cut paths (the design's §4.4 amended).
   The client only reads heights; every host draws one surface; a tile carries 36 % fewer samples. The price: a tile is
   finished only once its eight neighbours are solved (on never-visited ground the first fine ground ~16 s in place of
   ~7 s; the coarser rung stands meanwhile, B81 Q5). No rule is broken: SL3 (the planet draws itself, the joining too),
   SL10 (one generator; the client reads the heights it receives), SL6 (the fine tiles' message approved in B81 Q3, the
   same message), W16 and no seams (one join, done once), B64.
2. **THE CORNER RULE** (a repair, not a choice): each tile edge's cut runs in TWO HALVES — from each of the edge's corner
   points to the edge's MIDPOINT — each corner point a pure function of its four tiles (where the four agree best within
   the cut's band), each midpoint a pure function of the edge's two tiles; so every cell of a finished tile depends only on
   the tile and its eight neighbours, and every host and every tile computes the same cut.
3. **THE CONDITION — NO QUALITY LOST:** the finished tiles are judged against the bench's own stitch at a four-tile corner
   (the slope break across every cut against the inside, and the pictures) before the build takes them.

**Example.** Four 128 m tiles of the home belt meet at one corner; the point where their ground agrees best is chosen
from those four alone; the two cuts on each side run from that point to their edges' midpoints; the finished tile ships
with its corners already joined, and every client draws the same ridge there.

**THE CONDITION MET (measured 2026-10-11, `erosion_bench --fine-corner 3,23,69`):** sixteen generator-solved belt tiles
(10 s on the M4 Pro's cores), the middle four finished by the corner rule (`fine::stitch::tile_cuts` + `finished_tile`,
4 ms a tile) against the bench's two-step stitch of the same four raw tiles, the cut's band ±56 cells (a node less the
blend, so no blend reads past a raster):

| reading | the corner rule | the bench's stitch |
|---|---|---|
| a line's largest slope break across the cuts, median / 90th / 99th | 1.165 / 1.444 / 1.669 | 1.166 / 1.445 / 1.669 |
| the same inside a tile | 1.176 / 1.421 / 1.527 | 1.177 / 1.420 / 1.527 |
| at the four-tile corner, 99th (a tile's middle 1.278) | 1.304 | 1.307 |
| the step across the finished tiles' straight edges, median / 99th / largest | 9.17 / 52.42 / 78.58 m | 9.25 / 50.17 / 78.58 m on the same lines |

The two surfaces differ on 4.9 % of the cells (where the two choose another cut through ground the raw tiles disagree on);
the maps side by side show no line at the corner in either. The finished tile through the coder: 303 KB (1.16 bytes a
sample) against the raw tile's 463 KB; its five mean levels (rungs 8–12) 115 KB; every level decoded back exactly.

## B86 (2026-10-12) — the final ground from the first frame; the loading screen over the running game

The owner, on the flats of B2's first window look (the 8 km stand-ins at their own height over the solved valleys) and
on the plain answers (the three cures A/B/C, the client's own solve, a loading screen, the bandwidth a player needs):
*"Yes please, add those. I also agree with the loading screen, but please no teleport. We need to put player at the
starting position immediately, so whole loading machinery can kick in, but show the loading screen over the gameplay in
the meanwhile."*

**Why.** Every temporary ground (the 8 km rows, or a cheaper solve on the client) differs from the final ground, so its
swap is a jump — a seam. The client's own solve is refused for the speed: each player's machine would repeat 7–8 s of
work a tile on the cores that make the picture, against ~10 ms to decode the shard's tile.

1. **THE FINAL GROUND FROM THE FIRST FRAME.** A player sees only the final ground, at the detail its distance reads: a
   finished tile's mean levels far, its full square near; each finer level refines the coarser one under a pixel at its
   switch (B84 item 2). In the normal case no temporary ground is drawn.
2. **THE ADVANCE CALCULATION.** The planet's own shard solves every fine tile of its body once per world version (the
   input word) and stores it — the same worker, the same rows (SL3, HR3). The order: the tiles its occupants and occupied
   children name; then its arrival places (B81 Q4); then the rest of the body, on the solve's share of the cores (F6).
   ESTIMATED: ~138 000 L1 tiles on the home planet, ~14 h on 7 cores; 15–60 GB stored a planet (the sea tiles unmeasured).
   OPEN at its build: how a planet nobody visits completes its solve, while idle work never keeps a realm alive (SL7,
   B81 Q4).
3. **FAR COPIES FIRST.** A tile's levels ship apart: a far tile sends only the levels its distance reads, the full square
   goes only to the near ring, and an arriving finer level refines in place. ESTIMATED ~10–15 MB around a stand in place of
   ~128 MB.
4. **SENT AHEAD WITH THE SPEED.** An occupant's reach grows with its closing speed (the movement answers of 2026-08-27,
   item 2: grow the interest radius with the closing speed, never cap the speed); the order by the time to reach (the
   way-in design §4.1).
5. **THE PLAYER'S DISK CACHE.** The client keeps every finished level it received on its own disk, keyed by the world
   version's input word and the tile; a new world version reads none of the old. On arrival the client reads its disk
   first, and the shard never sends a piece the client holds: the client's acknowledgement (B81 Q3, approved — "the
   client acknowledges held tiles") names every tile it holds, from the wire or from its disk. The cache's size is the
   player's setting (an operational number in the client's config; its default asked at the build).
6. **THE LOADING SCREEN OVER THE RUNNING GAME, AT LOGIN ONLY. NO TELEPORT, NO SPECIAL LOGIN PATH** (the login builds no
   world): the player is placed at the starting position at once; every machine (the area of interest, the solve, the
   ship, the chunks) runs exactly as in play; the client draws a loading screen OVER the running game until the ground in
   its view is complete (every wanted chunk on its final ground, no stand-in), then the screen goes. The client decides
   "complete" from what it already holds (its tile book and its chunks) — no new data crosses. (The owner's word of
   2026-09-30 already allowed "the loading screen before all meshes are loaded" at login.)
7. **THE SAFETY NET (cure B).** Where a place is truly not solved yet (a new world version, an arrival that beat the
   queue), the 8 km ground fills ONLY the empty place and never covers final ground that has arrived. AMENDS the own-height
   stand-in of 2026-09-30 for the fine level: a coarse stand-in stands at its own height only where nothing finer has
   arrived. Its price — a step at the border between solved ground and the patch while a place loads (up to ~1 km on the
   belt) — is a defect that items 2 and 4 keep out of the flight (the seam law stays asserted on every flight leg).
8. **THE BANDWIDTH (ESTIMATES, measured when items 3–5 are built):** the login ~10–15 MB (about 4 s at 25 Mbit/s); travel
   ~34 KB a km of new ground (28 km/s low over new ground ≈ 8 Mbit/s); a player's link about 10 Mbit/s at the least,
   25 Mbit/s or more comfortable. The shard's pace (8 parts of 32 KiB a tick, ~5 MiB/s a session) stays a stated pace.
9. **THE BUILD ORDER** (tests on the owner's word, B74): (1) the face-edge cut — every tile finishes, at a face's edge and
   at the cube's corner (owed by B2); (2) the safety net; then the owner's look (the flats gone); (3) far copies first;
   (4) the send-ahead; (5) the loading screen; (6) the advance calculation; (7) the disk cache.

**Example.** A player logs in on the home belt. The player stands at the start position at once, and the loading screen
covers the running game for a few seconds while the planet's shard ships the near tiles whole and the far tiles as small
copies. The screen goes; the player boards a hull and flies off fast; the far ridges are already the real ridges, coarse,
and they sharpen as the hull comes near, with no flat lid and no jump. The next evening the same player logs in again:
the tiles come from the player's own disk, and the screen goes almost at once.

## B87 (2026-10-12) — the fine ground reaches every viewer by distance, whatever realm the viewer is in

The owner, after the finding that a pilot aboard a hull over the planet received no fine tile (the planet's shard shipped
them only to its own direct occupants): *"Detailed pieces for players in a hull or any other realm that sees the planet —
please. It should be based on the distance from the rung, not in which realm I'm. And it's true for ANY realm."*

1. **ONE RULE BY DISTANCE.** What fine ground a viewer receives depends only on its distance from the ground, by the
   rung's own reach (each fine level out to the edge the client's ladder asks that level's rung at, bounded by the ground
   the eye can see) — never on which realm the viewer stands in: on foot on the planet, aboard a hull over it, in a
   station in orbit, or passing by.
2. **ANY REALM.** The same rule for every realm that draws fine ground (every body where the level applies), and for
   every viewer of it.
3. **ONE RULE, ONE CALL** (`vd_terrain::artifact::fine_needs_of_head`, read off the artifact head's three numbers):
   - every viewer OUTSIDE the realm: the gateway, which composes each session's picture (SL2's clarification: it is not a
     realm), states the viewer's VIEW of the realm — a direction and a distance from its centre, never a pose entering
     its simulation — on the existing one tile path (`ArtifactWant { view }`); the realm's shard names the fine tiles
     that view reads, solves what is missing (the view is the realm's reason, SL3: a realm draws itself, whoever looks),
     and answers them one level a record, coarse first, to the gateway's cache for the realm; the gateway serves each
     session the levels its own distance reads;
   - the realm's OWN occupants: the realm's shard, which alone holds their poses (the gateway cannot see a pose inside a
     realm), serves them by the same call at their poses, one level a record, coarse first.
   No new data class crosses and no wire arm is added (the view, the realm audience and the fine part exist).

**Example.** A pilot aboard a hull 10 m over the home belt and a walker on the ground beside it receive the same fine
tiles, the near ones whole and the far ones as their coarse copies; a pilot in orbit 2 000 km up receives the far copies
of the tiles under her view, and their finer levels as she descends.

## B88 (2026-10-12) — one path for the ground's pieces, through the gateway, for every viewer in every realm

The owner, on the report of B87's two deliveries (the realm's own occupants served by the realm's shard, every other
viewer through the gateway): *"Two ways to deliver the same pieces: no, it should be one way only — through gateway!"*;
and: *"How then picture is drawn? I can be in the hull that stands on the planet. I need all being drawn using exactly
the same machinery, despite of which realm I currently belong."* AMENDS B87 item 3.

1. **ONE PATH.** Every tile a session draws — the artifact's tiles and the fine levels alike — comes from the gateway's
   cache for the realm, chosen by the one call from the session's view of the realm. No shard ships a tile to a session
   directly; the realm's own occupants are served like every other viewer.
2. **ONE EYE, THE PICTURE'S OWN INPUTS.** The picture is drawn from two inputs: each realm's centre and facing relative
   to the realm the player stands in (the rows the gateway composes), and the player's own position inside that realm
   (the realm's own snapshot). The view of every drawn realm comes from the same two, by one formula: the eye in the
   frame of the realm the player stands in, carried into the drawn realm by its row. That eye is the avatar's own
   position for the realm the player stands in — the gateway reads it off the snapshot that realm sends the player's
   client (SL2's clarification: the connection plane is not a realm; a shard handing its own occupants to it is no
   crossing), once a beat, and the view goes back to that same realm alone — and the centre of the player's realm for
   every other realm (SL7: an occupied realm stands in for whoever is inside it; SL2: no occupant pose enters another
   realm). The proxy's error is the player's realm's own size; the fine reach carries one fine tile (64 km on the home
   planet) past its edge.
3. **ONE RULE.** The occupant-only width of the tile rule (the interest side) is gone; every viewer is named by the reach
   rule alone. The gateway's record of tiles relayed straight to a session is gone with the second path; the
   named-session relay stays for what a realm owes named sessions (slice 9's rows).
   No new data class crosses and no wire arm is added.

**Example.** A pilot sits in a hull berthed on the home belt: her eye is the hull's centre carried into the planet's
frame by the hull's row. She walks out onto the belt: her eye is now her avatar's position, as the planet's snapshot to
her client states it. Both views go to the planet's shard the same way, the shard answers both into the gateway's cache
the same way, and the same call serves her the same pieces from the same spot — there is no seam at the hull's door.

## B89 (2026-10-12) — the speed step: all five items, judged in the window

The owner, after the window look on the one delivery path (*"It looks great now"*) and the measured cost (14 526 pieces
of ground, 48 million vertices, 2.6 GB of meshes, a frame every 46 ms in the headless picture): *"Ok, do all 5 from the
speed work please. When you do the changes, please check performance, but not run full gates or anything else — we can
judge visually in the window please. I don't want to spend week on this."*

1. **ALL FIVE ITEMS**, in the order the first measurement shows: the measurement (the frame split, the count of
   visible pieces); a frame cap at the screen's refresh and the accepted render scale; smaller vertices; one draw for
   many pieces; asking only for what can be seen (the planet's horizon and the nearer hills' horizon, around the eye
   in every direction); fewer vertices on flat ground (our own simplification — no new library).
2. **CHECKS BY PERFORMANCE AND BY LOOK** — headless readings of the cost, and the owner's look in the window (asked
   each time); no full gate, no suite, no coverage run in this step.
3. **SHORT** — days, not a week.

## B90 (2026-10-12) — the most optimized version stays: the new detail rule and the reconstruction at 0.67 by default

The owner, after four window sessions on the speed step (the new detail rule with the reconstruction on and off, flying
fast and low and then at 20 km/s): *"No, it's the same for all on and off. So if the off (on any tier) performing better,
we can keep it. Here what happens — I'm flying super fast now. I guess we will not be able to do anything with that at the
moment, as we have loading limits. So let's take the most optimized version please."*

1. **THE NEW DETAIL RULE STAYS** (B89 item: the fine ground's rungs hand over by their curve's step): the level
   differences the owner saw are the same with every setting; at 20 km/s they are the LOADING LIMIT — the fine tiles land
   after the eye, so a chunk stands on the rough ground first (B86's safety net) and its edge shows against the finished
   ground. The cure is B86 item 4 (sent ahead with the speed), not a ladder change.
2. **THE RECONSTRUCTION AT 0.67 IS THE WINDOW'S DEFAULT** (amends B24's "MSAA stays the default"): MEASURED a frame every
   16.8 ms (60 a second) against 32 ms at full size with 4× MSAA; `VD_TEMPORAL=0` turns it off for a run. The headless
   capture keeps its reference defaults (its pictures change only on the owner's word, B71).
3. **THE FREEZES STAY OPEN, MEASURED**: ~2 frames a second 50–150 ms late with every setting, the terrain's own work in
   them ~5 ms and the network thread on time — the drawing side's; found by a profiler next.
