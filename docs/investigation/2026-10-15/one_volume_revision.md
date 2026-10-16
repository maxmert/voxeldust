# The one-volume model on today's ground — revision 2 (ruling B99 step 3)

*Written 2026-10-15 on branch `mc/terrain-fine-ground-next-step` at `aeee643` (GENERATOR_VERSION 26,
`crates/terrain/src/tag.rs:348`; ARTIFACT_KEY_GENERATION 27, `tag.rs:380`; ARTIFACT_VERSION 9,
`crates/terrain/src/artifact.rs:96`; FINE_VERSION 3 and L2_VERSION 2, `crates/terrain/src/fine/tile.rs:37`, `:43`;
PROTO_MINOR 39, floor 36, `crates/wire/src/version.rs:378`, `:409`). It revises
`docs/investigation/2026-09-27/one_volume_design.md` (the ORIGINAL, written at generator 21) and does not edit it.
Revision 2 answers two adversarial reviews: A, the code evidence (4 blocking, 10 major, 14 minor), and B, the rulings
and the soundness (3 blocking, 7 major, 14 minor). §7 holds one ledger row for each finding. This document builds
nothing, measures nothing and runs nothing.*

**Marks.** READ = read in the code at `path:line`. MEASURED = a run printed it; the source is named. ARITH = hand
arithmetic, the sum shown. ESTIMATE = arithmetic on an input nobody measured. INFERENCE = a conclusion from the code
that no run checked. DOC = a number that a ruling states, not measured again here. UNVERIFIED = a published constant that
must be read in its source before use.

**Binding.** B99 (the base first; this is its step 3). B21 (one volume; the solid pass, then the fill pass; "don't try to
connect"). B22 (the order; Q1 option (b); Q2 measure; Q3, Q4, Q6 accepted; Q8–Q23 at their defaults). W16 laws 1–4 and
B16 (one shape at every distance; a channel may not be a thing a rung decides to draw). B79 item 3 (the river tree is a
lawful input; no water word in the solid pass). B81 (one tile machinery; item 3 the coast join; Q8 unbuilt ground may
move). B84 items 2–3 (the swap under a pixel at its switch; rung 6 → 7 at 117 km). B85, B86 (items 1, 3, 5, 7), B87,
B88 (item 3 the named-session relay). B94 item 1 (the fine level's base level, FINE_VERSION 4). B23, T9, B38 (no caps;
every number a law). B76, B93 item 2 (the pictures decide, from the side). W2, W11 (live water; a lake stands by its
budget). F1–F3. V4, V5 (Empty and Air; caves). S1–S5 (seed and secrecy). HR1–HR6, SL1–SL10, the seamless rule.

---

## 0. For the owner

### 0.1 What the model is, on today's ground

The planet is one volume of small cells, made in two passes.

1. **The ground pass** makes the solid cells from the ground that the planet's server already calculates. Each cell
   gets a material by a law: the rock of the bed it cuts, soil, loose stones under a cliff, glacier ice.
2. **The fill pass** gives every open cell one substance: the sea, a lake, a river, air, or nothing on a body with no
   air.

The game draws the top of the ground and the top of the water as one surface. A dig changes the same cells.

**Inside each detail level, every far copy is the exact average of the near copy.** Where one detail level hands the
picture to the next, the shore can slide a little. We measure that slide before you judge it.

**Example.** A walker stands on a beach of the home planet. The cells under their boots are sand over the bed's rock.
The open cells in front of them, under the sea's level, hold sea water. They dig a trench to the water, and the sea
flows in.

### 0.2 What changed since 28 September

- The ground below 8 km is calculated by the server at 128 m and at 16 m, stored, and sent through the gateway. The 4 m
  level comes after the rain's cure. Under the finest level the ground is a smooth curve with no random noise.
- No water is drawn. The old shore rules are deleted.
- The game seals every cave: it draws the rock over a cave as solid rock.
- Nothing is dug, placed, flooded or collided yet. Only a parked hull stands on the ground.

### 0.3 What the reviews and the code showed

1. **The server's 128 m and 16 m ground has no hollow for a closed lake.** Its rivers end only in the sea or at the
   tile's edge. A lake needs a floor under its level.
2. **The same calculation treats a dry basin under the sea's level as sea**, so the floor of such a basin has no small
   valleys at all.
3. **The game already holds the 8 km water facts** (which water each 8 km point ends in, and at what level). The server
   must still send the small-scale water map, because only the server holds the ground calculation's own drainage, and
   because a far copy needs the finest map to make its average.
4. **The plan of revision 1 had three holes:** rivers only at the 4 m level (they would appear and vanish as a pilot
   climbs), caves that the game seals while the server floods them, and a built area that froze its water (a wall of
   water after a world change).

### 0.4 What this revision proposes

**The ground pass (step 4):**
- The server's ground calculation ends its rivers where the 8 km water ends: in the sea, in each closed lake, in each
  dry basin. It reads only the 8 km drainage, never a water level that it did not read before. So each closed lake and
  each dry basin gets its own floor.
- The materials by law: the bed's rock (built), soil from how fast the ground wears down, glacier ice where the climate
  keeps ice (a smooth line, no 8 km squares), loose stones under cliffs. No ore ever comes from the map and the seed.
- **Caves stay solid rock on the server and in the game until a dig breaks into one.** Then the server opens it, and
  every computer draws and collides the same open cave.

**The fill pass (step 4):**
- Each tile carries a water map: which cells hold the sea, a lake or a river. Water fills only the low cells that join
  their water under its level, so no wall of water and no hanging pond can stand.
- **Rivers at every detail level by one law:** the river's width and depth come from its water flow. A river narrower
  than a cell is a tint of that cell. A river wider than a cell is a channel in the ground, filled to its banks. A pilot
  sees the same river from orbit (a faint line), from 50 km and on foot.

**Edits, built areas and standing (steps 5 and 6):**
- A built area keeps its own ground, stored as a small copy. Its water always comes from the newest world. A large
  world change makes a slope at the area's edge no steeper than the angle real hillsides stand at, and the far view
  shows the mound.
- One ground answer on the server serves the parked hulls, the walkers, the collisions and the floods.

### 0.5 What it costs (estimates)

- The server calculates the stored tiles again once: about one to five hours of its calculation threads around today's
  stands (ESTIMATE).
- Each tile grows by about a tenth to a sixth with the water map, as the tiles are packed today. A small change to the
  packing makes uniform areas almost free.
- The far view of a planet grows by about 3 MB (the share of water in each far cell).
- The game's memory while it builds the ground: one more byte for each cell. The drawn points need no new bytes.
- Every judged stand, coast and lake is judged again.

### 0.6 What stays as you ruled (reminders)

The far coast is option (b) of 28 September; the far map holds the mean top and the water's share (accepted then). Dug
holes fill by the server's bounded flood with its two refusals. Built ground never moves. Tests and gates run only on
your word.

### 0.7 The questions

Each question gives the recommendation, then its price. §6 gives the options in full.

- **Q1 — Where the ground ends its rivers.** At the 8 km water's own ends (the sea, closed lakes, dry basins).
  *Recommended.* Price: lakes that overflow their rim do not get a floor; we count them on the new rain's world first.
  If they are many, the other option reads each lake's level in the ground calculation, which changes your ruling of
  11 October that the ground reads no water level.
- **Q2 — The water map crosses with the tile.** *Recommended.* Price: about a tenth to a sixth more data a tile.
- **Q3 — Rivers at every level by one law, before the base freezes.** *Recommended.* Price: a bench and a step check of
  their own. Also: a basin dug beside a river fills to its lowest joint with the river; a dig that would leave a step of
  more than one cell is refused.
- **Q4 — How deep the soil is.** By the published soil law from how fast the ground wears down; about 100 bytes more
  for each 16 m tile. *Recommended.* Price: the dry world gives thick soil almost everywhere until the rain's cure.
- **Q5 — Glacier ice.** Ice in the top of the ground above the climate's smooth ice line, its thickness by the published
  ice law. *Recommended.* Price: the glacier tongues below the line come later.
- **Q6 — How a built area keeps its ground.** It stores a small copy of its own ground; its water always comes from the
  newest world; its edge slopes at most at the hillsides' own angle. Any dig makes such an area. *Recommended.* Price:
  a few kilobytes for each hectare built; after a large change, a mound or a hollow around the area.
- **Q7 — A dig reaches every viewer by distance**, as the tiles do. *Recommended.* This changes your ruling of
  12 October, which kept these rows for the sessions that the planet names.
- **Q8 — Caves stay solid until a dig breaks in.** *Recommended.* Also: please confirm that a cave under land stays
  dry until a groundwater law exists.
- **Q9 — The shore's slide between detail levels** is judged by the ground's own line (under a pixel). *Recommended*,
  measured first.
- **Q10 — The far map above the tiles** comes from the 8 km water now, and from the tiles' own average once the server
  has calculated a whole planet. *Recommended.* Price: one change of the far view, in one frame, per planet.

---

## 1. What still holds from the original design

| # | Item | Original | Why it holds |
|---|---|---|---|
| H1 | Two passes: solid, then fill | §0.1, §2 | B21 item 2. |
| H2 | ONE surface: the top `t = min(s, f)` | §1.1 | B20. The extractor reads one sign field (READ `crates/terrain/src/extract.rs:79-83`, `:334-348`). |
| H3 | The working cell: substance, `s`, derived `t`; `t` never stored or shipped | §1.1 | V6 B-3. The cell is two bytes today (READ `crates/terrain/src/chunk.rs:96-102`); Format B has no type yet (READ `crates/core/src/registry/mod.rs:51-58`). |
| H4 | One mesh; the wet share per vertex; a quad faces not-gas → gas | §1.2 | The renderer has no water material (READ: only comments name water in `crates/client-render`). The wet share fits the vertex's two spare bytes (READ `crates/client/src/chunks.rs:148-151`, `:186-192`). |
| H5 | The bed is drawn only where an eye can see it | §1.3 | So a far copy needs the top, never the bed (§3.6). |
| H6 | The flat merge under the chord bound, with the morph condition | §1.4 | Its code was deleted with the sheet; the arithmetic is written again. |
| H7 | A hollow opens its cell; the fill decides | §2.1 | Replaced for generated caves by §3.3; it holds for opened cells. |
| H8 | The medium: Air with air, Empty without | §2.5 | V4 row 13; today every body holds Air (READ `crates/terrain/src/chunk.rs:446`); the flag exists (READ `crates/core/src/look.rs:167-168`). |
| H9 | Salt or fresh by law; tides and waves never in the cells | §2.5 | T3, T9, SL10. |
| H10 | S6-6's order; the static fill in rank 1; live water rank 2 | §3.1 | READ `crates/terrain/src/compose.rs:1-17` (tests call it only). |
| H11 | Only the owning shard's flood makes water | §3.2 | READ `crates/core/src/registry/kind.rs:116-121`. |
| H12 | The bounded flood on the composition worker, `block_edit`, refusals not walls | §3.3 | W2, B22 Q4. |
| H13 | An Area: its own rows only; a hull: liquids from rows only | §3.5, §5.5 | R-16 NO, V5. |
| H14 | Composition off the tick; the edit tick | §3.6 | No shard builds cells today (READ `crates/bins/src/ground.rs:1-5`). |
| H15 | THE RULE of the far fold: the mean, rounded once, from the finest cells | §4.3 | Rule M (B36 item 6). Its mechanism changes (§3.6). |
| H16 | Airless moon: vacuum; gas giant: no solid pass | §5.3–§5.4 | Unchanged. |
| H17 | Collision reads the composed rung-0 cells | §6.2 | SL10, W10 item 4. |
| H18 | One world epoch for a change of meaning; refuse another version by name | §6.3 | `accept_head` checks no version today (READ `crates/client/src/artifact_book.rs`, `fn accept_head`). |
| H19 | The dig's data: brush, two refusals, fluid levels tag, wet coverage, edit-tick counts | §6.4 A4–A9 | B22 Q6 (e) accepted; Q22's default is the brush. |
| H20 | Every gate names a failing control | §9 | The method stands. |

**Example (H2).** At rung 0 on the beach, a cell 20 m seaward has its centre 0.3 m over the sea floor and 1.2 m under the
sea's level: `s = +38`, `f = −128`, `t = −128`, liquid. The cell under the walker's boots: `s = −26`, `t = −26`, solid.

---

## 2. What today's ground changed

### 2.1 The ground's source

- **The 8 km artifact.** One row of ten bytes for each 8 192 m node: height, water word, receiver slot with the facies
  bits, discharge, three climate bytes, rock province (READ `crates/terrain/src/artifact.rs:102-157`). The facies' SEA bit
  is the CONNECTED ocean only (READ `crates/terrain/src/solve.rs:2910-2925`). "No receiver" marks where the water ENDS:
  the ocean, a closed lake's water, a dry pit (READ `artifact.rs:88-91`); every node under a closed lake's water has no
  receiver (READ `crates/terrain/src/lakes.rs:1607-1612`). The home planet has 1 216 nodes on a face edge, 8 871 936 in
  all (ARITH).
- **The client holds those rows whole** wherever it draws the fine ground (READ `crates/bins/src/artifact_source.rs:244-254`;
  `crates/client/src/artifact_book.rs:873-912`). A row's side (land, sea, lake, pocket) is a function of the row and the
  head's sea (READ `artifact.rs:300-324`, `side_of_row`). So the coast mask is not needed by anybody.
- **The fine tiles.** L1 (128 m, rung 7): 8 × 8 nodes a tile, five mean levels, rungs 7–12. L2 (16 m, rung 4): one node
  a tile, two mean levels, rungs 4–6; rungs 0–3 read its curve (READ `crates/terrain/src/fine/level.rs:6-11`, `:56-94`).
  L3 (4 m) is bench-only (READ `crates/terrain/src/fine/l3.rs:1-3`).
- **The law.** The joint law's steady state on hashed points under tan 35° (READ `crates/terrain/src/fine/l1.rs:1-20`;
  the rounds integrate at `:456-468` with `round_slopes` at `:437-454`). The 16 m ground is today's 128 m curve plus the
  law's detail finer than 128 m (READ `crates/terrain/src/fine/refine.rs:453-554`; `l2.rs:1-35` item 4).
- **The raster.** The 128 m raster samples the point surface by barycentric weights and rounds to a quarter metre (READ
  `l1.rs:604-640`, `:631`). The finished tile is joined by the corner rule from its 3 × 3 raw block (READ
  `crates/bins/src/fine_source.rs:640-656`), with a blend of ±8 cells across each cut (READ `crates/terrain/src/fine/stitch.rs:34`, `:88-104`).
- **Where the fine levels apply:** only on a body whose artifact states a sea (READ `crates/terrain/src/fine/read.rs:83-88`).

### 2.2 The rungs and the hand-overs

- A chunk is 62³ cells; a rung-r cell is 2^r m; 19 rungs (READ `crates/seed/src/ladder.rs:16-26`, `:49-51`).
- A chunk at rung r reads level `k = max(r − level rung, 0)` through a 16-sample Catmull-Rom; the 16 m book first (READ
  `read.rs:215-238`, `:355-369`).
- **A missing tile.** The terrain read answers `Missing` (READ `read.rs:18-19`; `chunk.rs:352`), but the client builds a
  ROUGH chunk on the 8 km field while a 128 m tile is missing (B86 item 7), and a COARSE chunk at rungs 0–6 on the 128 m
  ground while a 16 m tile is missing, and builds both again when the tiles land (READ `artifact_book.rs:143-206`;
  `crates/client/src/chunks.rs:171-180`). So the near ground — and later its water — changes in time under the eye.
- **Above rung 12** a chunk reads the 8 km PYRAMID, level 1 (16 km nodes) at rungs 10–14; the rows only to rung 9 (READ
  `artifact.rs:816-843`). So rung 12 → 13 goes from the tile's 4 096 m means to 16 384 m means.
- **The node fit.** MEASURED with today's method (B93, drainage rebuilt every second round): the node means at most
  349–417 m off their rows, 46–97 m on average, on three berth tiles (status file, 2026-10-13, the B92 walls entry). B84's
  1 233 m was FINE_VERSION 1's number; its cause there was MEASURED as 8 km slopes steeper than the 35° threshold (status
  file, 2026-10-11, B0 step 2b).

### 2.3 Where the water stands today

- No cell holds water (READ `chunk.rs:430-437`, `:955-957`; `crates/recipe/src/cell.rs:357-369`). No water is drawn (READ
  `chunks.rs:1556-1557`).
- The solve's water stays: the sea, Fill–Spill–Merge and Langbein's budget, the water's own tree (READ `lakes.rs:14-59`;
  `solve.rs:1256-1265`).
- **The fine solve's outlets:** every point whose base stands at or under the sea's level (READ `l1.rs:292`; at the 16 m
  level on the DRAWN 128 m ground, `refine.rs:196-200`), the 8 km rivers' exits at the rim at the lower row (READ
  `l1.rs:324-346`), else the rim's lowest point (READ `l1.rs:347-363`). An outlet keeps its base through every round
  (READ `l1.rs:427-429`, `:462-464`, `:594-599`). The fine module reads no lake level (READ: no `lake` or `water_m` in
  `crates/terrain/src/fine/`; `l1.rs:307` reads the receiver slot only).
- **No pit on the points.** Every pit spills by the priority flood (READ `crates/terrain/src/fine/tree.rs:89-131`), and the
  final heights integrate up the last tree with a slope over zero (READ `l1.rs:456-468`). MEASURED "no pit" on the 128 m
  POINTS on the bench (the way-in design, line 155); never on a finished raster, after the stitch, or at 16 m.

### 2.4 Substances, edits, collision

- The cell kernel: topsoil and subsoil by depth and a NOISE biome, then the bed's rock at a fixed radius, then bedrock
  (READ `cell.rs:403-452`; the biome rows `chunk.rs:452-457`, the snow row `:455`). The soil's strip is owed (READ
  `cell.rs:427-432`). `Stratum` has 21 codes and no salt water, regolith or ore (READ `crates/terrain/src/strata.rs:13-42`).
- The ICE facies comes from the climate's equilibrium line and the ice flux (READ `solve.rs:3287-3295`; `climate.rs:71-72`,
  `:568-572`). No cell is ever ice.
- No edit runs end to end: the gateway counts and drops a world action (READ
  `crates/connection-plane/src/gateway/client.rs:236-243`); the block store's families have no writer (READ
  `crates/sim/src/stub/built_store.rs:31-39`).
- No walker collides (READ `crates/sim/src/stub/dot.rs:495-549`). A parked hull stands on the shard's height read (READ
  `crates/sim/src/stub/stand.rs:33-45`; `crates/bins/src/ground.rs:66-102`). The pull is zero (READ
  `crates/sim/src/stub/placement.rs:69-86`). No shard builds cells.
- **The client seals the caves:** a hollow with rock over it in its own column is drawn as rock; the code registers that
  "a dug opening (the edit system) will need the hollow's reach to the open air instead" (READ `chunks.rs:1290-1308`).

### 2.5 The original's parts: what is gone, what is dead, what is live

| Part | State today | In this revision |
|---|---|---|
| The shore law, the water sheet, its client parts | DELETED (READ: no hit; `chunks.rs:1556`) | — |
| The coast mask, `CoastCounts`, `coast_side` | Built, stored and folded into the digest; never shipped (READ `artifact.rs:199-239`, `:375`, `:1709-1724`, `:1819-1827`; `crates/bins/src/artifact_store.rs:144-162`; `artifact_source.rs:199-203`). The client's receive path and the gateway's coast cache stay in the code (READ `artifact_book.rs:603-623`, `:800-870`; `crates/connection-plane/src/gateway/artifact.rs:124`, `:193-225`) | Delete at generator 27's own artifact move, so no extra re-solve (§3.9). `Side` and `side_of_row` stay: a function of a row. |
| The pyramid's water words | Built, stored, shipped EMPTY (READ `artifact_source.rs:132-141`) and READ LIVE by the client's snow line, which therefore sees a dry planet (READ `crates/client/src/surface.rs:64-91`; the registered defect `docs/design/DEFERRED.md:12139`) | Replaced by the far map's share (§3.6) and the head's climate words (§3.8 V4). |
| `field_step_m` | LIVE: a ground hand-over law that sizes the tile ring on the shard, the gateway and the client (READ `artifact_source.rs:166-168`; `gateway/artifact.rs:475`; `crates/client/src/ladder_view.rs:233-239`) | Stays. Revision 1 listed it as dead by error. |
| The ridged centring μ, the level map with the descent step, the rim pass, `L_ref`, K | Obsolete where fine levels apply: no octave under the fine ground (READ `chunk.rs:351`) | Their purposes return in §3.4–§3.6 in today's form. |

**Example.** A walker on the coast today sees the summer snow line of a dry planet: the client reads empty water words
and finds no sea. The far map's share in §3.6 gives the client the planet's real sea.

---

## 3. Step 4: the volume

### 3.1 The cell grid and the solid pass

- The cells are the ladder's own; rung 0 is 1 m. Each fine level stands on the same face chart and nests in the cells
  by powers of two (READ `read.rs:376-386`). The finest tile is a surface level, never a cell size.
- The solid pass is today's kernel: `s = (r − h)` in 1/128 of a cell, `h` the finest held level's curve (READ
  `cell.rs:502-508`). No noise under the finest level (B81 Q1).
- **A tile edge.** The finished tiles are joined by the corner rule (B85); a column's 16 samples come from whichever tile
  holds them (READ `read.rs:241-276`). Two chunks at an edge read the same samples.
- **A rung swap.** The solid pass adds no rung rule. The morph, the sink, the fade and the near-piece bound stay (READ
  `chunks.rs:58-86`, `:1474-1524`; `ladder_view.rs:305-331`).
- **THE LEVEL A WORLD VERSION DEFINES.** At each place the contract names one finest level: L2 today, L3 after B99 step 2.
  A ROUGH or COARSE chunk is a stated provisional stand-in (B86 item 7). No host collides an occupant on a stand-in: the
  shard solves the tiles under its occupants first (B86 item 2), and an occupant whose tile is missing keeps its pose
  until the tile lands (§4.6).

**Example.** Two rung-0 chunks meet on hull 19's valley floor across a 16 m tile edge. Both read the same sixteen samples
at the edge, so the ground has no step there.

### 3.2 Where the ground ends its rivers (Q1)

**The proposal, after review B's M-1: THE RECEIVER-END FORM. The solid pass reads the receiver slot, never a lake level.**

1. **An END node** is a node with no receiver: the ocean, a closed lake's water, a dry pit (READ `artifact.rs:88-91`;
   `lakes.rs:1607-1612`).
2. **The outlets** are the points of END nodes whose smooth base stands at or under the higher of the node's own row
   (its eroded ground, a solid word) and the sea's level (B81 item 3's coast join, already in the solve). The 8 km rivers'
   exits at the rim stay; B94 item 1's lawful exit depth joins them in the same world move (below).
3. **Today's sea rule is kept only inside END nodes.** So the land and the pocket nodes under the sea's level are no
   longer outlets: they drain to their own end.

**What it cures.**
- F-1: a pocket's non-end nodes drain to the pocket's pit node, so its floor gets small valleys; the pit node's own floor
  is a smooth playa under its row.
- F-2: a closed lake's nodes are END nodes, so the basin keeps a floor under the lake's row, and the fill pass fills it.
- B79 item 3 holds: no sea bit, lake bit or new water level enters the solid pass.

**What it does not cure (the price).**
- **A spilling lake** passes its water on, so its nodes have receivers. The ground drains through it, and it gets no
  floor. Under this form a spilling lake is not drawn at any distance (§3.6 keeps the far map consistent). W11 MEASURED
  2 spilling of 74 080 bodies at generator 14 (DOC); generator 27's wetter world may hold more. G-OUTLETS counts them
  and their area before the owner chooses.
- **Straight lines.** The outlet set of an end node stops at the node's edge where the neighbour is not an end node. A
  faint slope line along that edge is possible (INFERENCE); G-GRID and the side views read it.
- **The other option (Q1 b):** every standing water's points under its level are outlets, read from the water word,
  bounded by the smooth base's own contour, never by node squares. It cures spilling lakes too. It AMENDS B79 item 3.

**One world move with B94 item 1.** B94 item 1 rules the river's exit at its bed as FINE_VERSION 4, not built (READ
`tile.rs:37`; the exit at the lower row, `l1.rs:332`). Its measurement says it adds almost nothing on plains (status file,
2026-10-13, B94 step 1). This revision's outlets, B94 item 1's exits, the river carve (§3.5) and the water map (§3.4) move
the fine tiles ONCE: FINE_VERSION 4 and L2_VERSION 3, after generator 27 (B99's order).

**Example.** A closed salt lake covers three 8 km nodes on the belt's plateau at 2 340 m. Today the 128 m ground drains
the basin over its rim, so the basin has no floor under 2 340 m (INFERENCE: G-OUTLETS reads, for each lake node, the sign
of its mean against its row and the lowest fine crossing of the rim against the level). Under the receiver-end form the
three nodes are ends; the basin keeps its floor; the fill pass fills it to 2 340 m.

### 3.3 Caves: one shape on both hosts (Q8)

**The proposal: A GENERATED CAVE IS LATENT. Both hosts take every generated hollow as solid rock until a dig reaches it.**

- The cell kernel computes the cell as if no hollow were there: the bed's rock by depth (READ `cell.rs:436-452`). This
  is what the client draws today (the seal), now written into the contract for the server too. The client's column seal
  retires; the 41 % of triangles that it saved stay saved (B22 step 2, DOC).
- **The opening.** When a dig removes a cell that the generator's cave field marks hollow (READ `cell.rs:519-530`), the
  shard runs THE OPENING FLOOD: the same bounded kernel as W2's flood, through the connected hollow cells, writing each as
  an opened row (rank 2, the medium). Then, if the cave touches water under its level, W2's flood fills it. Both hosts
  draw, collide and flood the same composed cells, so G-NO-DRIFT runs on the composed cells with no exception.
- **The bound.** A cave larger than the flood's total bound would open in part and show a rock face inside it. G-CAVE-SIZE
  measures the home planet's largest connected hollow first; the bound in the shard's tuning struct stands over it.
- **Natural cave mouths** (V5 (c)) do not exist today: caves carve only in a band under the surface (READ
  `chunks.rs:1293-1297`). A cave law with mouths needs a static open set per cave, designed with that law.
- **Water in caves.** A latent cave holds no water. B22 accepted the original's Q3; this revision reads it as the
  recommendation (a): a published groundwater law first, so an opened cave under land stays dry unless a dig joins it to
  water. The owner confirms this reading (Q8).

**Example.** A miner cuts sideways 3 m under the floor of the beach trench. Twelve metres under the beach the generator
holds a tube. The miner's pick breaks into it; the shard opens the tube's cells as rows in the same tick and floods them
from the trench, because the tube lies under the sea's level. Every client draws the open, flooded tube in the same
frame, and the miner's boots touch water that they can see.

### 3.4 The water map: which cells hold water (Q2)

**THE IDENTITY comes from the tile's own drainage; THE WET TEST comes from the drawn ground; CONNECTION is computed, never
assumed.**

1. **Identity (the shard).** Each raw raster cell takes the outlet that its nearest point's flow chain reaches in the
   tile's own solve: an end node's outlet (its water: the sea, a closed lake, or dry), or a rim exit (the water that the
   exit's 8 km link ends in, by the 8 km receiver chain). The finished cell takes the identity of the raster its height
   came from in the cut.
2. **Level.** The sea: 0 over the sea's datum. A closed lake: its water word less the sea. A dry end: none. A river:
   §3.5.
3. **Connection (the shard).** For each water `w` in the tile, a search from `w`'s outlet cells through cells whose drawn
   ground stands under `λ_w`. A reached cell of identity `w` is WET. An unreached cell of identity `w` under `λ_w` stays
   DRY: no hanging pond. A reached cell of another identity is a CONTACT: two waters join under the higher level (G-DIVIDE
   counts it).
4. **Footprint: B85's 3 × 3 raw block is kept.** The search runs on the tile's finished square and on the tile's OWN raw
   margin outside it (the raw raster reaches 64 cells past the tile, READ `level.rs:6-11`). A margin cell whose chain
   leaves the raw raster is a seed at its own height with its identity: its water runs down from there. The price: a path
   that leaves the square through a neighbour's blend band reads the raw heights there, not the finished ones (B85
   MEASURED 4.9 % of cells differ). G-SEAM-WATER counts disagreements at the cuts. The other form reads the neighbours'
   FINISHED squares, which needs the 5 × 5 raw block: 25 solves before a coast tile ships, not 9 (each 13–49 s on the
   shard, status file 2026-10-13). Offered only if G-SEAM-WATER is not zero, and then only for tiles with water.
5. **The 16 m level refines.** A 16 m cell takes the identity of the 128 m water at its place (the finished 128 m block
   that the 16 m solve already needs, READ `fine_source.rs:437-472`), and the search runs again on the 16 m drawn ground.
   The same waters, a finer shore.
6. **The spill.** If G-DIVIDE finds contacts (a lake's fine rim cut under its level), the lawful cure is Fill–Spill–Merge's
   own law: a still water stands at most at its lowest outlet (W11, B47). Across tiles that needs one fine level per
   lake, computed by the shard over the lake's tiles: a small table per realm, an SL6 ask brought only with the counts.

**The column (both hosts).** A column takes the identity of the raster sample nearest its centre, so one column has one
water. It is wet where its own curve height stands under that water's level. So the shore follows the smooth ground,
never a square. A river's surface never drops under the level of the water it ends in (§3.5), so a river mouth meets the
sea with no step.

**Why the shard computes it (SL6, review A B-1).** The client holds the 8 km water tree, so revision 1's reason ("the client
holds neither") was false. The true reasons: the identity reads the tile's own point drainage, from a solve that the client
never runs (B86: refused for the speed, 3–13 s a tile); a raster-only identity on the client would need the neighbours'
finished squares, which a client at its ring's edge does not hold; and the far copies need the finest map to fold (§3.6).

**Example.** On the beach the column 3 m behind the last wet sample reads that sample's water, the sea. Its own curve
stands 0.2 m under the sea's level, so it is wet; the next column stands 0.1 m over it and is dry. A hollow 200 m inland
stands 1 m under the sea's level, but its search never reaches it over the dune's crest: it stays dry.

### 3.5 Rivers at every level by one law (Q3)

**The law exists in the code:** Leopold & Maddock 1953's downstream hydraulic geometry calibrated on Earth, `w = 3.9·√Q`
and `d = 0.4·Q^0.4` at Earth's gravity with the gravity factor, as integer tables by the discharge class (READ
`crates/terrain/src/river.rs:1-30`). W4 item 1 approved it; W16 retired only the DRAWN stamp.

**ONE FORM AT EVERY LEVEL: THE CARVE AND THE FILL READ ONE LAW.**
1. **The discharge.** Each level's own tree carries `Q` (the rain over the area, and the 8 km rivers' inflow at the rim,
   as the solve already sums it: READ `l1.rs:280-345`). At the 8 km level `Q` is the row's discharge byte.
2. **A river wider than the level's cell** is carved into that level's raster along its channel, `w` wide and `d` deep,
   and the fill fills it TO ITS BANKS. So the drawn TOP does not change where the carve appears: the water's top is the
   bank. The 16 m channel lies inside the 128 m channel's cell, as B83's example asks of valleys.
3. **A river narrower than the cell** is a SHARE of that cell (its area over the cell's area): a tint, never geometry.
4. **The far copies** fold the share and the mean top like any water (§3.6). Above rung 12 the pyramid's share comes from
   the rows' discharge along each node's receiver link (`w·L/A`).
5. **No channel-head knob.** A small `Q` gives a tiny `w`, so its share is near nought and fades into the ground's colour.
   ARITH from `w = 3.9·√Q`: a channel is carved at the 128 m level from about 1 077 m³/s, at the 16 m level from about
   17 m³/s, at the 4 m level from about 1 m³/s.
6. **The floodplain:** Leopold & Wolman 1960's meander belt (2.7 channel widths, READ `river.rs:15-17`): the cells in the
   belt from the bed up to the bank hold alluvium (§3.7).
7. **Its own gate:** G-RIVER-RUNG at the 4 m → 16 m, 16 m → 128 m and 12 → 13 hand-overs: the share and the mean top step
   under the eye's rule (B27). W16 law 4 holds: no rung decides to draw a channel; every level shows the same river from
   the same law, as a tint or as a filled channel.

**Example.** The belt's trunk carries 9 500 m³/s: 391 m wide and 16 m deep (READ `river.rs:25-27`). From orbit it is a
faint line, 391 m of each 16 km node. At 50 km the 128 m tiles carve it and fill it to its banks. On foot the 16 m and 4 m
levels show its banks and its small side creeks; a creek of 3 m³/s is a tint at the 16 m level and a filled channel at
the 4 m level.

**The live flood beside a river (review B M-6).** A basin dug open to a river stands at the level of its LOWEST joint
with the river, as every still water stands at its outlet (W11). A dig whose joints differ by more than one cell's step is
refused with a reason, like two waters (B22 Q4), until W3's flowing water. This reads W2 rule 2 ("at that one level") for
a sloped river and is part of Q3.

### 3.6 The far view of water (option (b) on today's ground)

- **The mean levels carry the MEAN TOP in place of the mean ground, and the WET SHARE.** The top is `max(ground, λ)` with
  `λ` per cell from its identity (none on dry cells, so a dry pocket under the sea's level counts no water). The bed under
  far water is never drawn (H5), so the mean ground is not needed there. This is B22 Q6 (c)'s accepted form. Every level
  is computed from the finest raster and rounded once (H15): G-FAR-EXACT.
- **The 16 m → 128 m hand-over (rung 6 → 7, at 117 km, B84 item 3).** The 128 m level's own water meets the fold of the
  16 m water. A pixel there is about 135 m (ARITH: 117 000 × 0.0011506). The shore slides by about the detail over the
  shore's slope; the detail per cell after B97 is UNMEASURED (B96's 2.2 m median is superseded). G-SLIDE measures it at
  117 km; Q9 states the line.
- **The arrival of a 16 m tile** moves the near shore in time, as it moves the ground (§2.2). The 0.5 s ramp fades it
  (READ `crates/client/src/arrival.rs:26-27`); G-ARRIVE-WATER measures it. It belongs to B86 item 7's provisional price.
- **Rung 12 → 13.** The pyramid's level 1 cell is 4 × 4 of the tile's rung-12 cells. Before every tile under it is solved,
  its top and share come from the 8 km curve under each node's water and the river law (a registered interim). A share
  difference is a COLOUR step (W16 law 3), so G-SWAP reads it under the eye's rule. Q10: once the advance calculation
  (B86 item 2) holds a whole body, its pyramid becomes the exact fold of the tiles, in one frame (as the original's
  G-EPOCH).
- **Consistency with Q1 (a).** The interim far map counts only the waters that the near view can show: the sea, the
  closed lakes, the rivers. A spilling lake is in neither, so no far lake vanishes on arrival.

**Example.** A pilot climbs from the coastal plain. Up to 117 km every ring is the exact average of the finer one, and a
pond fades one way into a tint. At 117 km the shore may slide by a measured amount while the band passes. At the rung-13
switch the far map takes over; until the planet is whole in the server's store, that is the 8 km average, with its colour
step measured.

### 3.7 The substance of each cell by law

The law gives the substance at a point; rung 0 is authoritative (collision and mining); the look folds the shares later
(rule M; B99 parks the paint).

| Substance | The law | Inputs | Who computes | Crossing |
|---|---|---|---|---|
| Bed rock | The bed the cell cuts at its own radius (BUILT, READ `cell.rs:403-414`) | radius, province byte | both | none (W4 item 4) |
| Scree | The rock's powder at the foot of a face; first form by the rock share's lower margin (the colours design §3.4) | the drawn slope | both | none; a T9 interim to register |
| Soil | Q4 (a): the soil production function (Heimsath et al. 1997, UNVERIFIED constants): depth `h0·ln(P0/E)`, bare rock where `E ≥ P0`, `E` the lowering rate that the 16 m level reads back, one median per node (READ `l2.rs:117-139`), read by the Gaussian of logarithms (READ `l2.rs:140-155`). On eroding slopes only; one site's constants, so checked against Earth's soil depths (B79 Q1) | the 49 node medians of the tile's flow domain | both, from shipped medians | V2 (§3.8) |
| Alluvium | The meander belt of each carved river (§3.5 item 6) | the river's `Q` and channel | both, from the water map | inside V1 |
| Regolith | The loose top of a body with no air and no water; depth by an impact-gardening law, owed | the charter's flags | both | none; a T9 interim to register |
| Glacier ice | Q5: ice in the top of the ground where the fine ground stands over the climate's equilibrium line, read as a SMOOTH field over the nodes (no 8 km squares). Thickness: the basal shear form `τ_y/(ρ·g·sin α)` with the surface slope averaged over several ice thicknesses, and Nye's ice-cap form `√(2·τ_y·x/(ρ·g))` on flat ground (Nye 1952; τ_y about 100 kPa, UNVERIFIED; calibration body: Earth's glaciers); the bench reads which form holds where. The tongues below the line are owed | the nodes' climate bytes and the head's climate words (V4) | both | V4 |
| Snow | Paint only (K12, F17) | — | client | none |
| Water | The fill (§3.4, §3.5); a closed lake's water is salt (Langbein 1961) | the water map | shard → both | V1 |
| Air, Empty | The medium by the air flag (H8) | charter | both | none |
| Ore and every valuable | NEVER in the volume pass (§3.8 G-SECRECY) | — | — | — |

- `Stratum` grows by appending: salt water, regolith (no code moves). The noise biome's topsoil retires (T9).
- The province's 8 km squares stay a registered interim until the outcrop rule (the colours design §3.3).

**Example.** A miner digs into the mountain beside hull 19. The first metre is thin soil, because that slope wears down
fast. Under it is granite; forty metres higher, slate. On the pass above, the ice line crosses the slope as a smooth curve,
and the miner cuts through ice before rock.

### 3.8 What crosses, and why (SL6)

Each row states what, from which realm to which, why the receiver cannot compute it from what it holds, and the price.

| Ask | What, from where to where | Why the receiver cannot compute it | Price |
|---|---|---|---|
| V1 (Q2) | The water map in each finished fine tile: a water code a cell (dry, the sea, a lake of the tile's table, a river), a sparse river list (cell, depth, share), and on each mean level the mean top in place of the mean ground and the wet share; the planet's realm → every viewer through the gateway's cache (the approved tile path, B81 Q3) | The identity reads the tile's own point drainage (a solve the client never runs, B86); a raster identity needs neighbours a client lacks at its ring's edge; a far copy needs the finest map to fold (B86 item 3) | §3.9: about 10–16 % more bytes a tile today; a record form 3 that the gateway must learn (READ `gateway/artifact.rs:347-358`, `tile.rs:245-250`); a new protocol minor |
| V2 (Q4 a) | The 49 node medians of the lowering rate that each 16 m tile's solve used, in the finished 16 m tile | They come from a 128 m flow over 7 × 7 nodes (about 200 000 points, a triangulation in floats) that the client never runs (B86); SL10's no-drift gate is measured on the integer recipe, not on client floats | about 100 B a 16 m tile (ARITH 49 × 2 B); an integer Gaussian read in the recipe |
| V3 (reminder) | The pyramid's mean top and wet share; planet → viewers | It needs every row of the body | B22 Q6 (c) accepted it; +2.96 MB (ARITH: one share byte for each of 2 956 590 words), the top replaces today's heights |
| V4 | The realm's climate words in its head, so the client's climate law equals the shard's (the snow line and the ice line) | The client's climate reads the sea's share, which it cannot know exactly; today it reads a dry planet (§2.5) | a few words once; the parked colours design's Q3, moved here because ice is a substance |
| V5 (reminder) | The acknowledgement names a tile's level | — | registered (D-TERRAIN-12) |
| V6 (Q6) | A built area's frozen ground: a small raster of its height and substance inputs, as rows of the block store's diff | It is live state (a moment's ground), owned by the realm | a few KB a hectare (§4.3); a new row tag (R-13 approved new tags in the diff; this one is asked) |
| — | Opened cave cells, live water rows, dig rows | Live state | H19 (B22 Q6 (e)) and §3.3; no new arm |

**Example.** A pilot hovers 100 m over the bay with the bay's tiles held. Their client holds the rows that say "the water
here ends in the ocean", but not the 128 m drainage that decides which low cell behind the dune joins the bay. The
planet's shard sends that answer inside the tile it already sends.

### 3.9 Costs (estimates)

| Cost | Number | Mark |
|---|---|---|
| The water code plane | at least one bit a sample with today's coder (no run mode, READ `crates/terrain/src/fine/coder.rs:51-67`, `:94-122`): 262 144 samples ≈ 32 KB a tile at either level | ARITH |
| The wet share on the mean levels | 87 296 samples at the 128 m level (65 536 + 16 384 + 4 096 + 1 024 + 256) ≈ 11 KB; 81 920 at the 16 m level ≈ 10 KB | ARITH |
| The mean top in place of the mean ground | about the same bytes on land; fewer on the sea (a flat top) | ESTIMATE |
| The river list | in proportion to the channel length; small | ESTIMATE |
| Together | about 43–50 KB a tile: 10–16 % of a 303–420 KB 128 m tile (MEASURED 303 KB B85; `tile.rs:13-15` states about 420 KB); with JPEG-LS's run mode (a change in the same coder family, with FINE_VERSION 4) the uniform planes cost near nothing | ESTIMATE |
| The re-solve of FINE_VERSION 4 | around today's stands: 1 317 finished 128 m tiles in one login (MEASURED, `docs/design/DEFERRED.md:12303`) at 13–49 s each on the shard, and 1 196 16 m tiles named at two stands (621 + 575, status file 2026-10-14) at 9–41 s, on 7 threads: about 1–4.5 hours | ESTIMATE |
| The dead coast data | deleted at generator 27's artifact move, which re-solves every store anyway (B99 step 1), so it costs no extra solve; deleted alone it would change every digest and re-solve the home planet (2 245 s) and every far planet (hours) | READ `artifact.rs:1819-1827` |
| The working cell's top byte | +256 KiB a 64³ box while it builds | ARITH |
| A drawn vertex | +0 B: the wet share and its target take the surface word's two spare bytes | READ `chunks.rs:148-151` |
| The patch rows | about 2.5 KB a hectare at 4 m (625 samples × 4 B) | ARITH |

### 3.10 The checks

Visual checks often; heavy gates only on the owner's word (B74). The owner's look decides, from the side (B76, B93 item
2), on the home planet, the moon and the desert world (B23 item 3).

| Gate | Checks | Failing control |
|---|---|---|
| G-OUTLETS (bench, first) | Today: pocket points read as sea; each lake node's mean against its row and the rim's lowest fine crossing against its level; the count and area of spilling lakes (on generator 27's world) | — (a reading) |
| G-FACE | Water faces against open cells: level meets, tile cuts, patch edges, caves, rough chunks | The connection search switched off |
| G-DIVIDE | Contacts of two waters under the higher level | Today's outlets |
| G-SEAM-WATER | Wet/dry disagreements across a cut | The search on raw squares only |
| G-PIT-RASTER | Cells of a finished raster (128 m, after the stitch; 16 m) under their water's level that connect nowhere | — (a reading) |
| G-COAST-100 (B20) | 100 m over the sea at the 15° stand: sky between water and land, skirts, undersides | The extractor on `s` in place of `t` |
| G-WATERTIGHT, G-FLAT | Wet top edges shared; open water on its level | A merge block over the chord bound |
| G-FAR-EXACT | A parent's top and share against its children's mean | A parent read at one point |
| G-SLIDE (117 km), G-SWAP (rung 13), G-ARRIVE-WATER, G-POND-FLICKER | Shore motion and colour steps at the hand-overs and at a tile's arrival | The water map computed per rung |
| G-RIVER-RUNG | A river's share and top at 4 → 16 m, 16 → 128 m and 12 → 13 | The river carved without the fill |
| G-NO-DRIFT | Composed cells (substance, `s`, `t`) and the mesh, shard against client | The tampered table |
| G-CAVE-SIZE | The largest connected generated hollow | — (a reading) |
| G-SECRECY | No substance the volume pass can write carries the registry's survey-only property (a property to add; ores first) | A charter that names a survey-only substance |
| G-GRID | No lattice line in the ground at end-node edges | Outlets clipped to node squares with no contour |
| G-SUBSTANCE (bench) | Soil and ice depths against Earth's readings (a measuring stick) | — |
| HR5 | 100 % region and branch on every touched line | An uncovered branch |

### 3.11 The order of step 4

1. The bench, no world change: G-OUTLETS, G-PIT-RASTER and G-CAVE-SIZE on today's tiles.
2. After generator 27 (B99 step 1): the receiver-end outlets, B94 item 1's exits, the river carve and the water map in a
   bench copy; G-DIVIDE, G-SEAM-WATER, G-FACE, side views of a lake shore, a pocket, a coast and a river.
3. The owner's look and word on Q1–Q3 and Q8–Q10.
4. The one world move: FINE_VERSION 4, L2_VERSION 3, record form 3, the protocol minor; the latent caves; the cell's top
   byte; the extractor on `t`; the flat merge; a plain water material; the far map; the dead coast data gone with
   generator 27's artifact move.
5. The substances with Q4 and Q5; the climate words (V4).

---

## 4. Steps 5 and 6: edits, built areas, standing, collision, delivery

### 4.1 The edit interface

- An edit is a one-hop diff from the owning realm (SL10): rank-2 rows (a dug cell `Empty`, a placed cell, an opened cave
  cell, live water) and rank 3–5 rows (blocks, attachments), in S6-6's order (H10).
- **An edit never changes a tile.** The tiles stay a function of the world version; the far copies and the disk cache
  stay valid (B86 item 5 holds).
- The far view of an edit is S5-3's fold: the mean change of `s` and the top's shares (H19).
- The dig input first (V4 row 14), then the route: the gateway forwards a world action to the authority shard (R-5 YES).
- **Q7: the rows reach every viewer by distance**, as the tiles do (B87, B88 item 1). B88 item 3 kept the named-session
  relay "for slice 9's rows"; Q7 asks to change that. Its price: the gateway caches each realm's rows with a bound in its
  tuning struct and evicts by the client's acknowledgement, and an edit tick swaps in one frame for every viewer.
- The composition worker composes rung-0 cells on the shard, off the tick, from the shard's own books (READ
  `ground.rs:66-102`), the water map and the rows.
- **The flood's frontier is durable.** A flood that runs over several ticks (B22 Q4) stores its frontier with its rows. If
  the realm sleeps, the flood resumes at its next boot. So no liveness rule is added to SL7.

### 4.2 Built areas: the frozen patch on today's ground (Q6)

**F1's PURPOSE stands; its MECHANISM changes** (F1 counted octaves; the ground below 8 km is now a stored solve, B81 Q1).
After review B's B-3:

1. **Freeze the SOLID only.** At the first rank-2 or rank-3 row in an area, the realm stores the area's own drawn ground:
   the finest level's heights over its footprint in ABSOLUTE metres over the ladder's radius (never over the sea, which a
   world change can move), with the substance inputs (the soil depth, the ice depth, the province). These are V6's rows.
2. **The water is never frozen.** The static water outside comes from the newest world. Inside the area and its band, the
   shard's live flood decides the water at each world change (the same kernel, §4.1), so a moved sea re-levels the
   trench's water and no wall of water stands.
3. **The band's law: no slope in the band steeper than the threshold that the ground itself stands at** (tan 35°; B82,
   B97; a limit that IS a law, B23 item 1). The band's ground is the newest ground held within `tan 35° × distance` of the
   area's edge height: a talus apron, at least `|move| / 0.70` wide (ARITH: 60 m → 86 m; 100 m → 143 m; 300 m → 429 m).
4. **The far view shows it.** The area's frozen ground less the newest ground folds into the coarse entries exactly as an
   edit (S5-3), so a pilot far away sees the mound or the hollow before they land (W16 law 2).
5. **The list before a world version ships** (G-PATCH-MOVE): each area's largest move, its band's width, a river or a sea
   that now meets it. The owner sees the list.
6. **Any rank-2 or rank-3 row makes an area.** Edit rows are absolute cells, so a later ground change would bury or lift
   them (review B m-13).

The other form (Q6 b): the area names the world words it was built on, and the server keeps the old tiles under it. Its
price: an old 128 m and 16 m tile per area and word on the server and to nearby viewers, two books on the client, and an
amendment of B86 item 5 (the disk cache reads old words). Q6 (a) needs none of that.

### 4.3 The trench from the sea, step by step, across a large world change

A miner on the beach at the 15° coast stand digs a trench 2 m wide, 5 m deep and 100 m long, from the waterline inland.

1. The client sends the dig (a brush, Q22's default). The gateway routes it to the planet's shard (HR1); no pose crosses
   (SL2).
2. The shard checks the miner's reach and asks its child index whether an Area owns the cells (one lookup, SL9). Its
   composition worker composes the rung-0 cells from its books, the water map and the rows.
3. The dug cells become `Empty` rows with the brush's gap. If a cut breaks into a latent cave, the opening flood runs
   first (§3.3).
4. W2's flood starts at the cells that the water map fills with the sea and wets every dug cell under the sea's level that
   connects; a sealed pit beside it stays dry. Each wet row states its level (the fluid levels tag).
5. One durable commit under the owner fence; the area freezes its beach (§4.2); the coarse entries fold the wet coverage.
6. The rows reach every viewer by distance (Q7). Each client composes, fills, extracts and swaps in one frame.
7. The walker beside the miner sees water in the trench; a pilot 50 km up sees at most a faint tint (the trench is 1/29
   of a pixel, ARITH from the original). Collision reads the same cells.
8. **Generator 28 moves the world: the sea settles 4 m lower, and the coast plain beside the beach sinks 60 m.**
   - The beach keeps its frozen ground. The band falls from the beach's edge to the new plain at most at 35°: an apron at
     least 86 m wide.
   - The sea outside comes from the newest world, 4 m lower. In the area, the shard's flood re-levels the trench's water
     to the new sea in the first ticks: the top 4 m of the trench drain; no wall.
   - The far view shows the beach as a low mound over the new plain, folded like an edit.
9. **The same change with a 300 m move:** the apron is at least 429 m wide; the area stands on a mound; G-PATCH-MOVE lists
   it for the owner before the world version ships.
10. **The sea rises 4 m instead:** the newest sea covers the frozen beach's open cells under its new level, through the
    same flood, and the list shows that the beach now stands in water.

### 4.4 The landing pad by the river, step by step, across a large world change

A builder sets a steel pad on the bank of the river under the belt's pass.

1. The bank's cells are soil over the bed's rock. The river (§3.5) is a carved channel filled to its banks; its bank
   cells beside the pad hold alluvium.
2. The pad's blocks are rank-3 rows; a cell keeps its gap byte under a block (READ `compose.rs:13-16`). The first block
   freezes the area's ground (§4.2).
3. One corner juts into the channel: its blocks replace the water cells that they fill; the river is not re-routed (W2
   rule 1).
4. A hull lands on the pad: the stand reads the composed ground, the pad's top (§4.6).
5. The pad's rows reach a pilot 30 km away by distance (Q7); there the pad and the river are tints of the coarse entries
   and of the water map's shares.
6. **Generator 28 moves the river 200 m sideways and lowers the valley floor by 60 m.**
   - The pad's frozen bank stays; an apron of at least 86 m falls to the new floor.
   - The old channel inside the area holds no water from the newest world, so it stays a dry hollow; a hollow under a
     water's level that joins nothing stays dry (§3.4).
   - The new river outside comes from the newest world. Where its new course meets the area, its water stands against
     the frozen ground as against a dam (W2 rule 1), and it continues beyond the area.
   - G-PATCH-MOVE lists "a river meets this area" for the owner.
7. A mooring basin later dug open to the river along a 0.4 m fall stands at its lowest joint (§3.5); a basin open along a
   1.5 m fall is refused with a reason until W3.

### 4.5 Standing, collision and delivery (step 6)

**ONE GROUND QUERY on the shard:** `ground at (direction, rung 0)` → the surface radius, the substance at the surface, the
water's level — from the composed rung-0 truth (the finest level the world version defines, the water map, the rows, the
frozen areas). The stand, the walker, the collider and the flood read it. Today's `ShardGround` is its first half (READ
`ground.rs:66-102`).

| Item (B99 step 6) | Interface | Note |
|---|---|---|
| The walker stood | The walker stands on the query along the realm's down | Needs the planet's radial gravity function first (READ `placement.rs:69-86`) |
| A landed hull stood again | The stand re-plants a child that stops near the ground | D-TERRAIN-11 |
| The first statement on the fine ground | The tiles under the arrival places held from boot | B81 Q4 |
| A parked hull never follows a dropped tile | The tiles under parked children never dropped | D-TERRAIN-11, MEASURED 432 m |
| No collision on a stand-in | An occupant whose finest tile is missing keeps its pose until the tile lands; the shard solves its occupants' tiles first | B86 item 2 |
| The collider | A mesh per rung-0 chunk from the composed cells (latent caves solid), on the composition worker, within the occupants' reach | rapier; the same cells the client draws |
| The advance calculation, the disk cache, one library | Unchanged in kind: the water map rides the same records | B86 items 6–7, D-TERRAIN-13 |
| The acknowledgement with a level | V5 | registered |

**Example.** A pilot walks out of hull 17 onto the day-side hills. The shard's query answers the 16 m curve under their
boots, soil, and no water. They walk to a creek: the query answers water 0.4 m deep, and the collider and the flood read
the same answer.

### 4.6 The order of steps 5 and 6

1. The ground query (heights first) and the walker stood, beside step 4.
2. Step 5's detailed discussion (V4 row 14): the dig input, the store's writer, Q6, F3's open items.
3. The minimal edit path for one realm: route, rows, commit, the frozen area, the floods (dig, cave, re-level), delivery
   (Q7), the edit tick.
4. The collider on the composed cells; then G-IDENTICAL on the planet and an Area (HR4).

---

## 5. The rules check

| Rule | How revision 2 keeps it |
|---|---|
| HR1 | The volume, the water map and the edits stay in the planet's shard and store; rows cross only to the gateway. |
| HR2 | No entity crosses; cells and live water are realm state. |
| HR3 | One tile machinery; ONE flood kernel for a dig beside water, a cave opening and a re-level; `block_edit`, never a shard kind. |
| HR4 | The fill and the floods sit above `GridMapping`; G-IDENTICAL on the planet and an Area. |
| HR5 | 100 % on every new kernel; `fine::l3` before any game path. |
| HR6 | `vdctl` presses the dig input; the probe gets the wet kinds. |
| SL1, SL4 | No placement changes; no crossing path reads the volume. |
| SL2 | Floods and rows stay in the owning realm. |
| SL3 | The planet calculates its own water, substances and caves. |
| SL5 | One world; the bench runs THE world's bodies. |
| SL6 | V1, V2, V4, V6 are asked with their true reasons and prices; V3, V5 are reminders. Near water facts are not claimed impossible for the client: the reasons are B86's speed, the footprint and the far fold. |
| SL7 | Unchanged: a flood's frontier is durable, so no liveness reason is added. |
| SL9 | The Area check is one lookup. |
| SL10 | Latent caves make the drawn, collided and flooded shapes one; G-NO-DRIFT on composed cells. |
| W16, B16 | Rivers, lakes and the sea are in every level by one law; folds exact inside a level; hand-overs measured (Q9, Q10). |
| B79 item 3 | Kept under Q1 (a); amended only under Q1 (b), on the owner's word. |
| B21 | The solid reads only the drainage; the fill reads the water; no shore rule. |
| B23, T9 | The band and the ice use laws, not caps; interims are listed for the owner's word at the freeze (§6.1). |
| S1–S5 | G-SECRECY on a survey-only property; ores only as live state. |
| F1, B81 Q8 | Built ground never moves; its water follows the newest world. |

---

## 6. Risks and questions

### 6.1 Risks, and the interims that enter the base

- **R1** The world move changes lake shores, basins and coasts; references re-freeze only on a better look (B71).
- **R2** Spilling lakes get no floor under Q1 (a) (§3.2).
- **R3** The 6 → 7 shore slide may pass a pixel at 117 km on flat shores (Q9).
- **R4** G-SEAM-WATER may not read zero; then the 5 × 5 form for water tiles (§3.4).
- **R5** A body with water but no sea has no fine level (READ `read.rs:83-88`); a census of THE world's bodies answers it.
- **R6** The dry world's slow rates give thick soil until generator 27 (ARITH with UNVERIFIED `h0` 0.5 m and `P0`
  77 m/Myr: about 1–4 m for `E` from 8.6 down to 0.05 m/Myr).
- **R7** A large world change leaves mounds around built areas (§4.2); the owner sees the list.
- **R8** A cave larger than the flood's bound opens in part (§3.3).
- **R9** No gravity function yet: the walker's stand and the flood's "down" wait for the planet's radial function.
- **R10** Water and ice on a dry body may become valuable: their wiki test is owed with the economy (S5 item 4).
- **Interims that enter the base** (each to be registered in DEFERRED, and the owner's word at the freeze, B99 item 7):
  scree's first form; regolith's law; the glacier tongues; the province's 8 km squares; the 8 km far map before a body
  is whole; the groundwater law.

### 6.2 The questions for the owner

**Q1 — Where the ground ends its rivers (§3.2).**
- (a) The receiver-end form. *Recommended.* Price: spilling lakes get no floor and are not drawn; possible slope lines at
  end-node edges; FINE_VERSION 4 with B94 item 1.
- (b) Every standing water an outlet under its level, by the smooth contour. Price: AMENDS B79 item 3 (a water level in the
  solid pass).

**Q2 — SL6 V1: the water map crosses inside the finished tile (§3.4, §3.8).**
- (a) Yes. *Recommended.* Price: about 10–16 % more bytes a tile today, near zero with a run mode; record form 3.
- (b) The client computes near water from the rows and the rasters. Price: the neighbours' finished squares (absent at the
  ring's edge), a raster identity unlike the shard's, and still no far fold.

**Q3 — Rivers at every level by one law, before the freeze (§3.5).**
- (a) Yes, with the lowest-joint rule for a dug basin. *Recommended.* Price: a bench, G-RIVER-RUNG, the river carve in the
  world move.
- (b) Rivers after the freeze. Price: the contract's water changes after the base freezes.

**Q4 — Soil depth (§3.7).**
- (a) The soil production law from the shipped node rates (V2). *Recommended.* Price: about 100 B a 16 m tile; thick soil
  until generator 27; the constants read first.
- (b) A depth from the slope. Price: a stated choice, not a law (T9).

**Q5 — Glacier ice (§3.7).**
- (a) Over the smooth ice line, thickness by the basal shear and ice-cap forms. *Recommended.* Price: the tongues later;
  V4.
- (b) No ice in the base. Price: a miner on a glacier cuts rock.

**Q6 — Built areas (§4.2).**
- (a) The area stores its own ground; the water is never frozen; the band at most 35°; any edit makes an area.
  *Recommended.* Price: a few KB a hectare; mounds after a large change. AMENDS F1's mechanism (its purpose stands).
- (b) The area names its world words. Price: old tiles kept and sent; AMENDS B86 item 5.

**Q7 — A dig's rows reach every viewer by distance (§4.1).** (a) Yes. *Recommended.* AMENDS B88 item 3. (b) The named
sessions only. Price: a pilot in a hull may miss a trench that a walker beside them sees.

**Q8 — Latent caves (§3.3).** (a) Solid on both hosts until a dig reaches them. *Recommended.* (b) No caves in the base
until the cave law with mouths. (c) Both hosts compute each cave's reach to the air at generation. Price of (c): a search
per cave on every host. Also: confirm the reading of B22 Q3 — a groundwater law first, so caves under land stay dry.

**Q9 — The shore's slide at the 6 → 7 hand-over.** (a) The ground's own line, under a pixel at the switch (B84 item 2),
measured first. *Recommended.* (b) B18 Q4's accepted sizes, judged in flight. If (a) fails, a design that carries the
finer shore into the coarser level comes back with its price.

**Q10 — The far map above rung 12 (§3.6).** (a) The 8 km average now, the tiles' exact average once a body is whole, in
one frame. *Recommended.* Price: one far-view change per body. (b) The 8 km average for ever. Price: the colour step at
rung 13 stays.

**Example.** If the owner answers (a) to every question, a pilot who descends over the belt's plateau sees the closed
salt lake as a tint from orbit, then a lake whose curved shore stays at every ring, the river that runs into it, and on
the bank a landing pad that keeps its ground — with its water always the newest world's.

---

## 7. Review ledger (revision 1 → 2)

Verdicts: ACCEPTED (the finding is right; the design changed), PARTLY (part right; the rest rebutted with evidence),
REBUTTED (wrong, with evidence). Each reviewer claim on code was read again in the code before its verdict.

### Review A — the code evidence

| Id | Verdict | What changed, or why not |
|---|---|---|
| A-B1 | PARTLY | Right: the client holds the 8 km water tree (READ `artifact_source.rs:244-254`, `artifact_book.rs:873-912`, `artifact.rs:300-324`); F-3 withdrawn. Not right that the client can compute the same map: the identity reads the tile's point drainage, and a raster form needs neighbours a client lacks at its ring's edge. V1 now states those reasons and the far fold (§3.4, §3.8); Q2 (b) offers the client form with its price. |
| A-B2 | PARTLY | Right for the raster, the stitch and the 16 m form; "nothing more is needed" removed. The connection is now computed by a search, and G-PIT-RASTER and G-FACE count the residue (§3.4). Kept: on the points, every height integrates up the last tree with a slope over zero (READ `l1.rs:456-468`), used only for the margin seeds. |
| A-B3 | ACCEPTED | The map now searches the tile's own finished square and its own raw margin: the 3 × 3 raw bound is kept; the residue is G-SEAM-WATER; the 5 × 5 form is priced (25 solves, 13–49 s each) as a fallback (§3.4). |
| A-B4 | ACCEPTED | `field_step_m` is live (READ `artifact_source.rs:168`, `gateway/artifact.rs:475`); moved to "stays" (§2.5). |
| A-M1 | ACCEPTED | The 1 233 m attribution withdrawn; today's fit 349–417 m stated; the example and G-OUTLETS read both outcomes (§2.2, §3.2). |
| A-M2 | ACCEPTED | F-1's consequence restated (a smooth floor with no small valleys); `refine.rs:200` reads the drawn 128 m ground (§2.3, §3.2). |
| A-M3 | ACCEPTED | ROUGH and COARSE chunks stated; the shore's change in time and G-ARRIVE-WATER added (§2.2, §3.6). |
| A-M4 | ACCEPTED | The 2.2 m input withdrawn (UNMEASURED); rung 13 reads pyramid level 1; the colour step is G-SWAP; Q9, Q10 (§3.6). |
| A-M5 | ACCEPTED | Costs redone: two planes, the mean top in place of the mean ground, the 16 m tile priced, 349 440 corrected, a range on 303–420 KB (§3.9). |
| A-M6 | ACCEPTED | V2 is now 49 node medians a 16 m tile (about 100 B), with the true reason (§3.8). |
| A-M7 | ACCEPTED | Counts and times re-cited (1–4.5 h); the dead data deleted at generator 27's move to avoid an extra re-solve (§3.9). |
| A-M8 | ACCEPTED | Record form 3, the gateway's label taught, a new protocol minor (§3.8, §3.11). |
| A-M9 | ACCEPTED | B94 item 1 named; one world move owns FINE_VERSION 4 (§3.2). |
| A-M10 | ACCEPTED | The snow line's live reader and its defect named; the far share and V4 cure it (§2.5, §3.8). |
| A-m1 | ACCEPTED | Lines corrected (`chunk.rs:455`, `read.rs:18-19`, `:355-369`, `artifact.rs:103-107`). |
| A-m2 | ACCEPTED | The sink, `fine_ship.rs:133`, the solve's state, `ladder.rs:49-51` cited as found. |
| A-m3 | ACCEPTED | The rounds cited at `l1.rs:456-468`, `:437-454`. |
| A-m4 | ACCEPTED | The blend is ±8 cells (a half-width). |
| A-m5 | ACCEPTED | "Only comments name water in client-render". |
| A-m6 | ACCEPTED | The vertex is 32 B; the wet share takes the spare bytes (+0 B). |
| A-m7 | ACCEPTED | "No shard builds cells" stated; the caves exist in the generator's kernel only (§2.4, §3.3). |
| A-m8 | ACCEPTED | The client's coast receive path and the gateway's coast cache added to the deletion (§2.5). |
| A-m9 | ACCEPTED | The coast mask goes; `side_of_row` stays (§2.5). |
| A-m10 | ACCEPTED | One identity a column, from the nearest sample (§3.4). |
| A-m11 | ACCEPTED | `FineField` forwards the water words; the solve reads the slot only (§2.3). |
| A-m12 | ACCEPTED | `read.rs:41-42`'s stale "rung 13 reads the node's row" noted with `level_for` (§2.2). |
| A-m13 | ACCEPTED | Above rung 12: the pyramid alone (§2.2). |
| A-m14 | ACCEPTED | H18 holds (confirmation; no change). |

### Review B — the rulings and the soundness

| Id | Verdict | What changed, or why not |
|---|---|---|
| B-B1 | ACCEPTED | Rivers at every level by one law: carve and fill to the banks where wider than the cell, a share where narrower, the far share from the rows' discharge; G-RIVER-RUNG at three hand-overs (§3.5). |
| B-B2 | ACCEPTED | Latent caves: solid on both hosts until a dig reaches them; one opening flood; G-NO-DRIFT on composed cells (§3.3, Q8). |
| B-B3 | ACCEPTED | Freeze the solid only, in absolute metres; the water from the newest world and the live re-level; the band by the 35° law; the far fold; the large-move walks (§4.2–§4.4). |
| B-M1 | ACCEPTED | The receiver-end form is the recommendation; Q1 (b) names the B79 item 3 amendment; the side word is not revived; node squares avoided or measured (§3.2). |
| B-M2 | ACCEPTED | Every ask states its true reason and price; the alluvium inside V1; coastal and 16 m prices (§3.8, §3.9). |
| B-M3 | ACCEPTED | 117 km stated; the 12 → 13 interim named with Q10; the arrival's water change; §0.1 corrected (§0.1, §3.6). |
| B-M4 | ACCEPTED | One identity a column; connection by search; river mouths continuous; caves latent; G-FACE (§3.4). |
| B-M5 | ACCEPTED | The ice by a smooth line, two published thickness forms, no node squares (§3.7, Q5). |
| B-M6 | ACCEPTED | The lowest-joint rule and the refusal over a one-cell step, inside Q3 (§3.5). |
| B-M7 | ACCEPTED | The area stores its substance inputs; Q6 (a) needs no B86 item 5 amendment, (b) names it; the per-hectare price (§4.2, §3.9). |
| B-m1 | ACCEPTED | B94 item 1 named; one move (§3.2). |
| B-m2 | ACCEPTED | The far map is the accepted form: top and share, the top in place of the heights (§3.6, V3). |
| B-m3 | ACCEPTED | Q8 asks the owner to confirm B22 Q3's reading. |
| B-m4 | ACCEPTED | §0.7 and Q7 say that B88 item 3 chose the other way; the cache's bound and eviction priced. |
| B-m5 | ACCEPTED | Calibration limits, the eroding-slope scope and the smooth read stated; R6 corrected. |
| B-m6 | ACCEPTED | The interims listed for registration and the owner's word at the freeze (§6.1); the floodplain now uses the meander belt law. |
| B-m7 | ACCEPTED | G-SECRECY keyed on a survey-only registry property; the wiki test for water and ice on dry bodies owed (R10). |
| B-m8 | PARTLY | The half-flood is real. Rebutted cure: no liveness rule is needed; the frontier is durable and resumes at boot, so SL7 is not amended (§4.1). |
| B-m9 | ACCEPTED | A dry end's outlets are its points under its row: a smooth playa, not a funnel; side views (§3.2). |
| B-m10 | ACCEPTED | The top uses each cell's own water; a dry identity counts no water (§3.6). |
| B-m11 | ACCEPTED | R5 and a census of the bodies. |
| B-m12 | ACCEPTED | The level a world version defines; no collision on a stand-in (§3.1, §4.5). |
| B-m13 | ACCEPTED | Any rank-2 or rank-3 row makes an area (§4.2). |
| B-m14 | ACCEPTED | §0 rewritten in plain words; its missing prices added. |

**The count.** 52 findings: ACCEPTED 49, PARTLY 3 (A-B1, A-B2, B-m8), REBUTTED 0.
