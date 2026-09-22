# Slice 8d — water and rock

For the owner's discussion, before any 8d code starts (ruling V4: no slice starts before its own
discussion). It obeys ruling T1's line for 8d, ruling T2 (what 8 keeps open for the ocean), ruling T3
(no erosion in the game), ruling T9 (no physical number is drawn), ruling F1 (the frozen patch) and
law SL10 (one generator, two hosts, no drift).

---

## 1. What 8d is for

Today a pilot flies over a solved planet and sees hills, a shore and a flat sea. No water runs
anywhere: the solve knows where every river goes, and the ground does not show it. The rock is one
sediment over one bedrock for the whole planet, picked by depth under the grass, so a hillside has no
bands and a hill can grow no cliff of its own. After 8d the same pilot sees a river in a real channel
with banks and a flat floodplain beside it, a lake with a shore and an outlet where the river starts
again, a gorge where the river cut through a hard bed, cliffs and treads where the beds cut the
hillside, and a mine that hits limestone in one province and basalt in another. And the owner can
place a named canyon by hand, over the seed's own shape, without moving one block that anybody built.

---

## 2. What exists already

Read from the code, not from the design documents.

| The piece | State | Where it lives |
|---|---|---|
| The routing: every node picks its steepest lower neighbour; the pits are filled to a spill level; the flats are resolved by a distance field | built | `crates/terrain/src/solve.rs` |
| The discharge: the rain accumulated downstream, kept as a log class | built, and **nothing reads it back** — 8d is its first reader | `crates/terrain/src/artifact.rs` |
| The stored row per node: height, water level, receiver and kind bits, discharge, temperature, rain, aridity | built | `crates/terrain/src/artifact.rs` |
| The four water bits: sea, lake, coast, ice | built; **only the coast bit is read** by shipped code | `crates/terrain/src/solve.rs`, `crates/terrain/src/chunk.rs` |
| A lake's spill level, stored per node and folded into the far view's pyramid | built | `crates/terrain/src/artifact.rs` |
| The clipped water sheet as a real surface at the water's own level; the shore as the land's own crossing of it | built | `crates/terrain/src/position.rs` |
| One column rule for the core and the halo: the direction, the height, the biome, the water | built | `crates/terrain/src/chunk.rs` |
| The beach: a column inside the coast band that stands above its water takes the desert's strata | built | `crates/terrain/src/chunk.rs` |
| The cap-rock bench: the surface pulled toward bed tops at FIXED RADII, one body-wide strength, faded out over ONE rung | built as a SHAPE only | `crates/recipe/src/terrace.rs` |
| The substance of a cell, by DEPTH under the surface | built, and wrong for a cliff | `crates/terrain/src/strata.rs` |
| The sediment each basin deposited, in cubic metres | built, and **nothing reads it** — 8d is its first reader | `crates/terrain/src/solve.rs` |
| The caves, and the point-to-line distance the channel carve needs | built | `crates/terrain/src/carve.rs` |
| The composition order: the shape, then cell edits, then blocks | built | `crates/terrain/src/compose.rs` |
| The store's row families for the block diff and its pyramid | planted, no writer yet | `crates/sim/src/stub/built_store.rs` |
| A river as a surface; the channel carve; the floodplain; a per-bed hardness; a rock map; the authored override | **missing** | — |

| The number that binds | Value | Source |
|---|---|---|
| A solve node's spacing | exactly 8 192 m; 8 871 936 nodes on the home planet | the solve bench, 2026-09-19 |
| A stored node row | 9 bytes | `crates/terrain/src/artifact.rs` |
| A shipped tile | 64 by 64 nodes, 36 KB | `crates/terrain/src/artifact.rs` |
| The home planet's artifact | 85.8 MB, six pyramid levels; the solve 57–71 s of one core | the solve bench, 2026-09-19 |
| The bed spacing of the bench | twice the body's sediment band, drawn in 20–80 m | `crates/terrain/src/body.rs` |
| The registry's substances | 60, of which eleven are bedrock and ten are ore | `crates/core/src/registry/substance.rs` |
| The chunk budget, re-read | 80 % of a measured 111 ms chunk is the eight PARENT meshes its morph reads, not its own work | the chunk budget base, 2026-09-16 |
| The reference's own rock | beds 1–20 m thick; cliffs 20–200 m of vertical; a mesa 1–20 km across and 100–600 m tall; valleys 200–800 m deep, 1–5 km apart | the reference target |

---

## 3. The pieces of 8d

### 3.1 Rivers as real surfaces

**The mechanism.** The solve already stores, for every node, which neighbour its water leaves to and
how much water passes. A chain of those arrows IS a river. 8d reads the chain and makes a line of
points in the body's own metres, then gives the line a width and a depth from the stored discharge
class by a published law. Nothing new is stored: a river is an index a host builds from the tiles it
already holds.

**The width law, and its calibration body.** The width and the depth come from the hydraulic geometry
of Leopold and Maddock (1953), calibrated on EARTH. The width grows as the square root of the
discharge; the depth grows as the discharge to the power two fifths. Gravity enters from physics and
not from a fit: a bed's shear is the water's weight on its slope, so at a fixed critical shear the
depth falls as gravity rises and the width rises with it. A low-gravity moon therefore gets deep,
narrow rivers, with no new constant. The two Earth coefficients are named as calibrations and ride
the world tag. The power of two fifths is read from a committed table of 256 entries indexed by the
stored discharge class, because the float fence forbids the power function; the class's own step
costs 2.5 % on a width.

| Drainage area | Discharge | Width | Depth |
|---|---|---|---|
| 100 km² | 1.0 m³/s | 3.8 m | 0.4 m |
| 10 000 km² | 95 m³/s | 38 m | 2.5 m |
| 1 000 000 km² | 9 506 m³/s | 380 m | 15.6 m |
| 6 000 000 km² | 57 034 m³/s | 931 m | 32.0 m |

(Computed on Earth's gravity at a runoff of 0.3 m a year; source: the rivers investigation.)

**A straight 8 km line reads as a canal.** Each line piece is split four times, into pieces of about
514 m, and each new middle point is moved sideways by a seed-drawn offset. The key folds the two node
identities in a fixed order, so the chunk on either side of the line draws the same bend. The
offset's size starts at a multiple of the channel's own width, because a real river's bend length is
about eleven times its width — a physical fact, not a taste number. Below about 190 m of width there
are too few samples to draw a bend, so a small stream takes its bends from the valley it runs in.

**In the game.** The pilot lands beside the great river of the home planet's largest basin. It is
931 m wide and 32 m deep near the coast, and it bends over kilometres, not over metres.

**The datum.** Nothing new crosses a boundary. The arrows and the discharge already ride the artifact
tile the owner approved on 2026-09-19.

**The cost.** The sample box gathers the channel pieces of the twenty-five nodes around the chunk
once, then prunes them against the box's own sphere. The whole landform column work is estimated at
0.45 ms a sample box of 4 096 columns, of which the channel distance and the carve blend are 55 ns a
column; the gate refuses a stored distance field if the search passes 150 ns a column.

**The gate.** A histogram over the home planet: the drawn channel length per square kilometre against
Earth's published 0.5 to 5; the exponent of Hack's law; the branching ratio against the published 3
to 5. The picture stand at a river bend, which the picture set lists as owed.

### 3.2 The channel and the floodplain carve

**The mechanism.** The carve is a term the column pass SUBTRACTS from the raw surface. Inside the
channel the bed is a shallow parabola under the water. Over the bank the ground blends back to the
raw surface with the same quintic fade the noise already carries, so no crease stands. Beyond the
bank the ground blends toward the water level plus a freeboard: that flat strip is the floodplain,
and it is the only ground a settlement can stand on without terracing. The floodplain's width and its
freeboard grow with the basin's own deposited sediment, which the solve already computes in cubic
metres and which nothing reads today. The carve runs last and never lifts the bed, so a channel stays
falling all the way down.

**How it reads the rung.** The test is the VALLEY's width, not the water's. Testing the water's own
width deletes every headwater stream at the first rung, which was the earlier proposal's defect. A
valley is carved while its width covers a stated number of the rung's cells, and over the TWO rungs
above that the carve's strength falls to zero (ruling T1 names the two-rung fade). The cap-rock bench
fades over ONE rung, because a tread is a smaller thing than a valley. Both rules are ASSERTED today
and 8d owes their derivation; the fade is named as the one unsafe term of the whole arc.

**In the game.** The pilot descends toward the delta. At 40 km the ground is a smooth plain. As the
rungs change the valley deepens into that plain over two ring swaps, and no edge of it ever pops. The
far view gets nothing from the carve, and the pyramid folds the uncarved ground, as it does today.

**The gate.** The rung-disagreement measurement, re-run with the carve on: the worst disagreement in
the ninety-ninth percentile stays under one pixel on every rung pair (ruling T7). The standing
height-bound test cannot see a channel, because it samples 400 spread directions and a channel covers
about a thousandth of the surface; 8d adds a fixture that samples ALONG a known channel, and a pop
measurement over five separately faded terms at a walk and at 240 m/s. The picture stands: a river
valley, and a gorge where the river crosses a hard bed.

### 3.3 Lakes and their shores

**What exists.** A lake's level is stored on every node it covers, and the far view's pyramid carries
one water word per coarse node, so a highland lake no longer vanishes from orbit. The sheet draws one
flat quad per block of eight cells by eight where the corners agree on one level, and one quad per
cell where they disagree — which is exactly a lake's edge. Ruling T1's "clipped water sheet as a real
surface" is therefore already standing.

**What is missing, read in the code.** The lake has no SHORE: the coast band is measured against the
SEA alone, so a lake never sets the coast bit and never gets the beach. The lake bit is written and
read by nothing. The pyramid's water word carries no kind bits at all, so the far view can mark no
shore. The lake's OUTLET is not drawn: the river should stop at the near shore and start again at the
far one. Nothing asserts the level is flat across a whole lake. And the water inventory still spends
its whole volume on the sea, so every lake the flood fills holds water that was already spent — a
stated double count, owed since the initial land landed.

**The mechanism.** One machinery, not a second one. The coast band reads "a wet node beside a dry one"
and does not care which water it is, so the same band and the same beach serve a lake. A river's
water level inside a lake is the lake's own level, constant, and the river resumes at the node where
the descent resumes. Water AS CELLS — a surface a walker stands on, a second extracted mesh — is NOT
taken here: it was priced at 10.6 ms a chunk against an 8 ms budget, and it is the ocean slice's own
question.

**In the game.** The pilot follows the river upstream into a mountain lake. The water is level all the
way across, there is a gravel shore she can stand on, and the far shore is where the river starts
again.

**The datum.** Nothing new: both bits are already on the stored row.

**The gate.** An assertion over every channel node of the home planet: outside a lake the water level
strictly falls along the chain; inside a lake it is constant. The lakes' volume is subtracted from
the inventory and the sea's level is re-read. A new picture stand at a lake shore, from the ground and
from orbit — the orbit half is UNMEASURED by eye today.

### 3.4 The stratigraphic column at a fixed radius

**The defect today.** A cell's substance is chosen by its depth under the surface: a topsoil, a
subsoil, a sediment band, then the body's one bedrock forever, over four biomes. A depth band drapes
over the hill, so the order is always soft over hard, and a mesa and a hoodoo are a hard cap over a
soft layer. Today's table makes both impossible under any erosion rule. The deepest boundary sits 23
to 92 m under the grass, so a canyon wall would carry ledges in its top 92 m and nothing below. The
cap-rock bench already stands at FIXED RADII and already grows treads and risers; its own note names
what it lacks — one body-wide strength, and no mesa whose lid is a different rock.

**The mechanism 8d adds.** Each bed gets its own HARDNESS, from a fenced hash of the bed index and
the seed, read as a fraction. The bench then pulls hard where a hard bed stands and not at all where
a soft one does. The SUBSTANCE reads the bed index too — but only inside the veneer, the depth the
strata already change over. Below that depth every cell stays the body's bedrock with no lookup, so
the cheap skip that writes 463 of a radial column's 470 chunks with no cell pass keeps working, and
its landed test keeps passing. The soil's thickness is stripped on the SOLVE's slope, never on the
fine octaves' slope, so a cell's substance is the same at every rung and no band is owed.

**In the game.** The pilot walks up a hillside. Every hundred metres or so of altitude the ground
flattens into a tread and then rises in a riser. The riser is sandstone and the tread is shale, all
the way along the hillside, at the same height.

**The cost, and the warning.** The bed index inside the veneer is a subtract, a multiply, a floor and
a mask: estimated 0.48 ms a surface chunk at 2 ns a cell. The bench AMPLIFIES every variation under
it, so the relief bound, the dropped bound and the column bound are each multiplied by its own slope
factor, which is 1.28 to 1.55 over the strengths on the table.

**The gate.** The chunk-skip test stays green. A histogram of substance against radius along one
hillside: the same bed stands at the same radius at both ends. The picture stands: a cliff face and a
mesa.

### 3.5 The rock map

**The mechanism.** A hillside of flat beds reads as Monument Valley and never as the Alps, so the beds
need a REGION and a TILT. The solve already knows, per node, which crust it stands on and which belt
it is in. 8d turns that into one word per node: a PROVINCE (crystalline basement, folded belt, flat
shelf, rift basalt, deep sediment). The province picks which of the registry's eleven bedrocks the
beds are drawn from. The TILT is derived, not stored: the bed dip is one dot product per column over
the stored heights of the neighbouring nodes. A folded belt then grows ridges of upturned rock and a
shelf grows flat mesas.

**The seed law is honoured.** The map says which ROCK and which COLOUR, which is a map of LOOK. Bulk
stock alone comes from the seed. Every ore stays live state and no indicator material exists (ruling
S5-6). A published ore map is a treasure map, and the seed ruling forbids it. A river bar, a beach and
a delta carry no ore either.

**In the game.** A miner digs under the shelf province and hits limestone, then shale, then granite. A
miner digging the same depth in the rift province hits basalt all the way. Neither can read where the
copper is from the seed, because no copper is in the seed.

**The datum — an SL6 ASK.** The province is ONE more word on the stored node row, which grows from
nine bytes to ten. The local formulation was tried first: a client could recompute the province from
the crust and belt fields — but those are the solve's transient working state and are thrown away, so
recomputing them means re-running the solve, which is 57 to 71 s. The word therefore rides the
artifact tile that ALREADY crosses, on the row the owner approved on 2026-09-19. No new lane and no
new payload kind. Cost on the home planet: about 8.9 MB on an 85.8 MB artifact, and one version bump.
The standing red item "the mountains read soft" already asks for a per-node slope or uplift word
before 8d starts; if that word lands first, the province packs beside it and the row grows once, not
twice.

**The gate.** A map picture of the provinces over one face; a mine pin that digs a column in each
province and asserts the substances it passes.

### 3.6 The authored override

**What it is for.** The owner will want a named canyon, a crater, or a mountain that a story needs.
The seed cannot be asked for it, and an edit at one-metre cells cannot build it.

**The mechanism, and the one that is refused.** The change is NOT put inside the recipe below the
octaves. That would stop the shape being a function of the seed and the address on both hosts (SL10),
and it would break the world's self-check, which is computed from the seed's shape alone. The change
is added ON TOP of the generator's answer, after the fold, one hop — exactly what ruling S5-3 says
the pyramid holds.

**A row.** A coarse cell address at a stated rung; a height change in whole metres; a falloff radius
over which the change fades to nothing; and, when the author wants it, a substance for the ground it
makes. The rows sit in the realm's own store, in the block store's pyramid family that is already
planted, and they reach a client on the diff lane like any placed block.

**How a host applies it.** The composition order gains one rank between the generated shape and the
cell edits: the authored layer. A host builds the seed's shape, adds the authored layer, then applies
the player's cell edits over the result — so a player who fills an authored canyon's floor fills it.
The pyramid folds the layer's average change exactly as it folds a block tower's, and an entry whose
change is zero is deleted.

**How it relates to the frozen patch.** They compose and never fight. The frozen patch says WHICH
octave count an area is derived with; the override says WHAT to add on top of that. A house built
under version five keeps its hill, and an authored canyon cut beside it is added over the same hill.

**What is 8d's, and what is slice 9's.** 8d owns the READ: the row's shape, the falloff, the
composition rank, the pyramid fold, and the gate that the world identity does not move. Slice 9 owns
the STORE — the row family and its writer — and slice 10 owns the lane. 8d ships one pinned authored
layer on the home planet so the read path is measured against a picture. The layer's ARRIVAL is a
named open seam: a change of hundreds of metres must not land in one frame.

**In the game.** The owner authors a 900 m canyon across the shelf province. The realm stores four
rows. Every client that draws the realm gets them with the realm's other diffs, and the canyon has
the terrain's own rock inside it, because the seed's shape was folded first.

**The gate.** The world identity's golden table does not move when an authored layer exists: the layer
is live state and never enters the identity. The picture stand at the authored canyon, from the rim
and from orbit.

---

## 4. The order of work, and what the owner looks at

| Step | What it builds | The owner's picture | The measured gate |
|---|---|---|---|
| 1 | The rock map word on the row | a province map over one face | the mine pin; the artifact's new size |
| 2 | The per-bed hardness and the substance at a fixed radius | a cliff face and a mesa | the chunk-skip test; substance against radius; the amplified bounds |
| 3 | The channel lines and the width law | a river from 2 km up | drainage density, Hack's law, the branching ratio; the column search under 150 ns |
| 4 | The carve and the floodplain | a river valley and a gorge | the rung disagreement under one pixel; the along-channel fixture; the pop at a walk and at 240 m/s |
| 5 | Lakes, shores and outlets | a lake shore from the ground and from orbit | the flat-level and falling-level assertions; the inventory's lake subtraction |
| 6 | The authored override's read path | an authored canyon | the identity's golden table does not move |

Every step also runs the identical fixture the hard rules demand: the carve and the bench on a hull's
soil bay, which holds no seed artifact, make no channel and the same treads as on the planet. Each
step stops at its gate. The owner looks once per step. A red gate does not go to a picture. Two of the
six pictures — the lake shore and the gorge — do not exist in the picture set yet and 8d adds them.

---

## 5. The asks for the owner

| # | The question | The options | Recommended, and why |
|---|---|---|---|
| 1 | Which law gives a river its width and its depth, and on which body is it calibrated? | (a) Leopold and Maddock's hydraulic geometry, calibrated on EARTH, with gravity from bed shear; (b) a width drawn per river; (c) a width from the channel's own slope and a critical shear | **(a).** Ruling T9 demands a published law with a stated calibration body. (b) is a draw with no cause and the same ruling refuses it. (c) needs a grain size that nothing computes yet. |
| 2 | Between a half and four fifths of the valleys a player can SEE carry no water, because a valley the eye reads is narrower than one solve node. Build the small stream? | (a) a table-driven stream in any valley whose floor converges, carrying no discharge and stored nowhere; (b) accept dry valleys; (c) make the solve's lattice finer | **(a).** On Earth a mountainside with twenty valleys has twenty streams, and a dry valley is rare. (c) multiplies the solve's minute and its 85.8 MB by four at each halving. |
| 3 | Is a lake's surface the sea's own mechanism, or its own? | (a) the same water word, the same sheet, the same coast band and beach; (b) a separate lake surface | **(a).** The word and the fold already exist and already carry a lake. A second mechanism is a second cost model for one job. |
| 4 | At what resolution does the rock map state a province? | (a) one word per solve node, 8 192 m; (b) per climate node, about 32 km; (c) per chunk | **(a).** A province boundary is a geological boundary tens of kilometres wide, so 8 km draws it. (c) would be per-cell state with no source. |
| 5 | What is a row of the authored override, and who may write it? | (a) a coarse cell, a height change, a falloff radius, an optional substance, written by an author's tool through the realm that owns the ground; (b) the same, and a player may write one; (c) a free-form mesh | **(a).** A player's shape change is a cell edit and already has a path; an authored landform is the owner's tool. (c) has no pyramid fold and no falloff. |
| 6 | The province word is new data on a lane that already crosses. Approve it? | (a) one more word on the artifact tile row; (b) recompute on the client, which means re-running the whole solve; (c) no rock map, one palette for the whole world forever | **(a).** The local formulation was tried and costs a re-solve. It adds no lane and no payload kind. |
| 7 | A player dams the seed's river. What happens? | (a) the dam holds nothing: the water still runs where the seed's water runs, and the local change wins inside its own width and nothing further; (b) a local re-solve; (c) refuse the build | **(a).** A re-solve is a minute on a player's action. This is the open ruling L20, and a player will find it. |
| 8 | 8d's cliff pictures are the first cliffs this world has. The mesh extractor's open question — a rounded crease or a sharp one — is free to decide until the world is frozen. Decide it here? | (a) yes: judge both on the same cliff and mesa pictures and pick; (b) keep the rounded one and revisit after the freeze, at the cost of re-addressing | **(a).** The question was reserved for 8d's pictures by name, and a bench lip is exactly the crease it asks about. After the freeze the change is no longer free. |
| 9 | One ocean world of the home system draws 39 km of relief, which the water sheet will now draw. Accept, or cap? | (a) accept: it is the relief law's own arm; (b) cap a small water world's relief | **(a) until the picture.** The ledger names it and decides it nowhere. A blocker rides with it: the stored sea level is a sixteen-bit metre count, and two water worlds already saturate it and state a lying level, so that word must widen before any water-world picture is judged. |

---

## 6. What 8d does not deliver

No erosion in the game, and no landscape that changes with time (ruling T3): the authored override is
the only shape change after the freeze. No moving river, and no re-solve after a player changes the
ground; a player who digs under a lake does not drain it. No large bend on a great river: the solve's
arrows cannot express a 10 km meander, so a great river in this world runs straighter than the
Mississippi. No waves, no buoyancy, no depth colour, and no water surface a walker can stand on —
those are the ocean slice and the physics slice. No flooded cave: a carved cell is air and knows
nothing of the water level. No free-standing rock pillar: a pillar needs a joint field, and the third
dimension slice owns it. No ore map, ever, from the seed. No fields, roads or walls; no fauna; no
volcanic, wind-blown or cave-dissolved rock family until each is priced (ruling T5).
