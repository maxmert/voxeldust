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
