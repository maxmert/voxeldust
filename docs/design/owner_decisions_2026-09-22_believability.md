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
