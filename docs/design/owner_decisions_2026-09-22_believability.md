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
