# The ground at 2–4 m (build step B5, level L3) — design, revision 2

Status: a design for the owner's answers. Nothing here is built. Revision 1 (2026-10-14) had two adversarial reviews — A
the rules and rulings (no blocker, 9 major, 11 minor), B the code, the cost and the seams (1 blocker, 6 major, 8 minor);
both kept outside the repo in `/tmp/vd-b5/reviews/`. Revision 2 answers every finding (§13). Sources: the two research
reports (`/tmp/vd-b5/research/laws_2to4m.md`, `/tmp/vd-b5/research/earth_small_scales.md`), the code at `c5b83b7`, the
way-in design (`docs/investigation/2026-10-11/fine_ground_way_in_design.md` §9 step B5) and rulings B80–B97.

Marks: MEASURED = a tool printed it; READ = read in the code; ESTIMATE = arithmetic, not run; UNVERIFIED = a source's claim
not checked.

## 0. The idea, and its honest price

Today the ground under 16 m is a smooth Catmull-Rom curve through the 16 m points. Earth's lidar says that is nearly right
in height but not in slope: between 2 m and 16 m Earth holds a detail of 0.7–1.2 m RMS that carries 11–18 % of the slope
the light sees (MEASURED). We would add ONE more level to the one tile machinery: L3 at the rung-2 cell (4 m on the home
planet), solved by the same law as L1 and L2 (B80, B97), in B96's form (today's drawn 16 m ground plus the law's detail
finer than 16 m), reaching every viewer by distance (B87) through the gateway (B88).

**THE PRICE (ESTIMATE from the measured 16 m solve and the code's own reach rule; review B M-1, M-2):** an L3 tile holds
about 560 000 points — the same as a 16 m tile — so it costs 4.5–15 s on a calm core and 9–41 s on the shard in a running
game. The code's reach names about 58 L3 tiles around a viewer for rung 2 alone (6.8 km), about 150 for rungs 2–3
(11.5 km). So the 4 m ring around a NEW place takes about 1–6 minutes (rung 2 alone) or 3–15 minutes (rungs 2–3) of the
solve's cores in a running game, and the 16 m ground stands near the eye meanwhile. Revision 1 implied ~20 s. **Before
any build, the bench measures whether a lever brings this down, and the owner decides whether the look is worth the
price (§11 Q1).**

**Example.** A walker steps off hull 19 onto the valley floor. With L3, within a few kilometres the slopes carry the small
chutes and the sharper crests the 16 m curve smooths away. At a berth solved ahead, that ground is there from the first
frame; at a place nobody visited, the 16 m ground stands for minutes and then gains its 4 m detail.

## 1. The rules this design keeps

| Rule | How |
|---|---|
| SL3, B35 | the planet's shard solves, stores and ships its L3 tiles; the 16 m flow record (§5) is the shard's own derived data, never shipped |
| SL10, F7, B64 | B97's solve one level down, under `Gf`, fixed-order sums; the stored tile is integers in the level's unit (§9) |
| SL6 | the L3 records ride the existing fine part with its level byte (data approved in B81 Q3); the acknowledgement's level needs a NEW appended wire arm — an SL6 ask (§11 Q5) |
| SL7, SL9 | the L3 reasons are the existing fine reasons by distance; no walk over children |
| B23, B38, T9 | the law is B97's; every new number is a STATED CHOICE listed in §6, none hidden; the threshold at 4 m is the owner's choice (§11 Q3) |
| B39 | the L3 settle law is stated (§6) and the bench reads N against 2N |
| B81 Q1 | no noise: the detail comes from the law; where the law adds nothing the curve stands |
| W16, B84 | B96's form; the swap L3 → L2 read by its LARGEST offset against one pixel at its switch, and the whole hand-over step against the ladder's tolerance (§8) |
| B81, B85 | the same tile, cut, corner rule; ONE machinery by level index, no special book (§10); G-SEAM at a face edge and a cube corner |
| B86, B87, B88 | the provisional ground is stated and priced (§7), with B86 item 1's exception as an explicit owner question |
| B81 Q6 | the shard's collision ground reads the finest held level, L3 included (§10) |
| B91 | the same fine work threads, below the window |
| B93, B76, B79 Q4 | from the side and at eye height beside Earth; nothing retires before the owner's look (the terrace, §11 Q4) |
| B74, HR5, HR4 | tests on the owner's word; 100 % coverage of the new branches; P-SAME on 1 and 4 threads and on aarch64 and x86-64 |

## 2. What Earth holds between 2 m and 16 m (MEASURED, US Geological Survey 3DEP 1 m lidar)

| Reading | Oregon hills | San Gabriel, gentler | San Gabriel, steep |
|---|---|---|---|
| RMS of the 2 m residual under the 16 m Catmull-Rom read-back | 0.71 m | 0.95 m | 1.22 m |
| 99th percentile / largest | 2.45 / 9.7 m | 3.11 / 12.2 m | 4.19 / 27.7 m |
| its share of the 2 m slope variance | 11–18 % (all three) | | |
| 4 m links over 35° / 45° / 63° | 9.4 / 0.52 / 0.002 % | — | 21.6 / 5.01 / 0.36 % |

- WHERE: 65–73 % on hillslopes and ridges (area under 100 m²; ridges 0.6–0.8 m over the curve, hollows 0.7–0.8 m under it);
  channels (area ≥ 10³ m², 4 % of the ground) cut 0.7–2.2 m under the curve, 11–16 %; rock steps only on the steep tile
  (the 9 % of the ground over 45° at 16 m carries 32 %).
- THE SLOPE–AREA STRUCTURE: the mountains hold a threshold plateau at a median 0.67–0.78 (tan 35° = 0.70) out to ~10³ m²,
  the break to valleys at 0.7–1.4 × 10⁴ m²; the hills roll over at 56 m² with no sharp head.
- 4 m against 2 m: a 4 m level holds ~90 % of the 2 m residual's HEIGHT variance but only ~70 % of its SLOPE share (review
  A A-8), and fewer steep links (over 63° on the steep tile 0.36 % at 4 m against 0.54 % at 2 m).

## 3. What the laws say (the literature report; its UNVERIFIED items stay marked there)

- Soil ground is smooth under the hillslope length (Perron, Kirchner & Dietrich 2008): smooth is lawful; today's curve is
  wrong in SHAPE (too round at crests and hollows), not in smoothness.
- The sharp things at 1–5 m: channel beds and banks, rock ledges, talus (~37°, DiBiase, Heimsath & Whipple 2012),
  stochastic micro-relief (needs facts the world lacks; stays ABSENT).
- Creep is not the answer at today's rates (Perron's length ~270 m on the mountain, ~580 m on the hills with Oregon's `D`):
  a creep law at 4 m would erase the finer channels — B97's failure one level down.
- No published method shows a lawful 1–5 m height field in eye-height pictures; Earth's lidar is the measuring stick.
- Above the channel head the joint law (a river law) does not apply as its source states (review A A-2). On steep ground
  the THRESHOLD law holds there (Burbank 1996; Earth's mountains: the threshold plateau); on gentle ground the law's own
  slope may stand under the threshold — the bench reads the law's slope–area plot beside Earth's, and the channel-head law
  (`A·S² ≥ C`, Montgomery & Dietrich 1988/1992; `C` ≈ 500 m² at 1–5 m cells, Tarolli et al. 2009 — a NEW CONSTANT with its
  calibration body) is the fallback if the law draws channels on hillslopes where Earth has none.

## 4. The level and its tiles

| | L3 at rung 2 (4 m) — RECOMMENDED for the bench | L3 at rung 1 (2 m) |
|---|---|---|
| Earth's 2 m detail it holds | ~90 % in height, ~70 % in slope | all |
| a tile (512 × 512 cells) | 2 048 m, a quarter node | 1 024 m, an eighth of a node |
| points a tile with its 128-cell buffer | ~560 000 (READ: the same cells as an L2 tile) | ~560 000 |
| a tile | 4.5–15 s calm; 9–41 s in a running game | the same a tile |
| tiles around a viewer (the code's reach: switch × 1.1 × 1.17 + a chunk + a tile, a square walk) | rung 2 alone (6.8 km): ~58; rungs 2–3 (11.5 km): ~150 | rungs 1–3: ~4× more |
| the ring in a running game (7 background threads) | rung 2 alone: 1.2–5.7 min; rungs 2–3: 3.2–14.6 min | ~4× more |

The recommendation for the bench: rung 2 ALONE first (rung 3 at 4–8 m a pixel shows little of the 8 m-mean detail, and
costs ~2.6× the tiles — review B M-2); rungs 0–1 read the L3 raster by the curve.

**Tiles in cells (review B m-1):** `LevelGeom.ts` already counts cells (9 at L1 and L2). The node shift becomes signed (L3
−2, a 2 m level −3) so `from_shift` and `from_side` still pair; L3's `margin_shift` is 5 (2 048 >> 5 = 64 cells). The tile
key stays `(face, tx, ty)` in tiles. At L3 a face edge holds exactly 4 864 tiles. The integer ranges and the planar
Delaunay predicates hold at 4 m and 2 m with room (READ + ESTIMATE, review B m-2b).

## 5. The 16 m flow (review B's BLOCKER, B-1)

An L3 tile needs the water and the rate on the 16 m ground around it, as an L2 tile needs them on the 128 m ground. A
literal copy of L2's pattern (a flow region of 7 × 7 macro nodes) is ~12.8 million 16 m points a tile — it cannot run.
The design:

1. **ONE 16 m FLOW RECORD PER L2 TILE**, built once on the shard after the L2 tile is FINISHED (its stitched 16 m ground):
   the 16 m points of the L2 tile's domain (the node and its 2 km buffer) at the drawn 16 m ground; their tree; the water
   through each point with the inflow at the domain's rim taken from the 128 m flow the L2 solve already builds (the 8 km
   rivers enter that flow, so a main valley carries its true water); each 128 m cell's rate read back by the joint law
   over its 16 m links under the threshold (the median a cell), and each 16 m point's rate by a Gaussian of the
   logarithms over the 128 m cells (σ half a 128 m spacing, reach 2.5 σ — the same read L2 makes one level up).
2. **Kept beside the L2 record on the shard, never shipped** (the shard's own derived data, SL3); all 16 L3 tiles inside
   the L2 tile read it, so two L3 neighbours inside one L2 tile see the same exits. Neighbours across an L2 tile's edge
   read two records whose domains overlap by the 2 km buffer; the cut stitches them and G-SEAM measures the join.
3. **Cost (ESTIMATE):** one tree and one accumulation over ~590 000 points, plus the rate read: well under a second, once
   per L2 tile — instead of once per L3 tile.
4. **Stated choices** (B38: each named with its reason, none a tuning knob): the region (the L2 domain), the rate unit
   (the 128 m cell), σ and its reach (L2's own read, one level down).

## 6. The law and every stated choice

1. **The steady form** on the 4 m points: each point over its one steepest lower neighbour by `(U/(K·Q))^(1/2.2)` under
   the threshold (§11 Q3), the drainage built again each pass, the half step from the second pass (B97).
2. **B96's form:** the 4 m ground is today's drawn 16 m ground plus the law's raster less its own 16 m version.
3. **The rim:** a 16 m link of the flow record that crosses the L3 domain's rim is its exit (on the drawn 16 m ground
   along the link) or its inflow (its water handed to the nearest 4 m point); the sea's points are outlets.
4. **THE STATED CHOICES:**
   - the buffer: 128 cells each side (512 m), the raster's reach 64 cells — the same cell counts as L1 and L2;
   - the settle law: no 16 m cell's mean of the law moves more than ONE STORED UNIT (§9) within the 128 m level's bound
     (180) — 1 m would be about the whole 4 m signal (review A A-4); the bench reads N against 2N (B39) (a settle law on
     the DRAWN detail was measured at L2 and refuted: the detail moves as much as the cell means);
   - the threshold: §11 Q3; the channel-head law only as the measured fallback (§3).
5. **Where L3 applies:** where L2 applies (a body with the macro solve's water: §3.6 of the way-in design); a dry moon has
   no L3 until its own family exists.
6. **Not in this step** (later families, each its own bench and look): flat creek beds and banks, rock ledges by the cliff
   law at the body's beds, talus, the crest's convexity, the rock-cover share. Never creep at 4 m at today's rates.

## 7. The provisional ground (review B M-5, review A A-7)

- **Today's state at L2** (READ): a chunk on the coarser ground until its tiles land, built again then (a pop in time, in
  view); the loading screen waits for the 128 m ground only; the shard's collision ground takes a tile when the shard
  holds it, the client when the gateway delivers it (a short window where the boots stand off the drawn ground).
- **With L3 at a new place** that 16 m ground stands near the eye for the ring's minutes (§0). Ruling B86 item 1 says
  "the final ground from the first frame (no temporary ground in the normal case)": at a new place the provisional 4 m
  state IS the normal case. Four answers to choose (§11 Q2): (a) the order — the nearest L3 tiles before the outer L2 ring,
  aligned on the shard and the gateway (their orders already differ, review B m-7); (b) the loading screen waits for the
  L3 tiles under the eye (a login a few tens of seconds longer, ESTIMATE); (c) a landing tile MORPHS in over the arrival
  ramp (0.5 s, B18 Q7, B81 Q6) instead of popping; (d) the advance calculation (B86 item 6) pre-solves L3 at the berths
  and arrival places (local, never body-wide: the whole land would be ~10 TB).

## 8. The near pieces and the hand-overs (review B M-6, m-5)

- **The near-piece rule generalised:** every rung up to the coarsest rung a level serves is bounded by the step of the
  level it reads (L3's cell for rungs 0–1, L3's raster against its mean for 2 → 3, the L3 → L2 swap for 3 → 4). ESTIMATE:
  rungs 0–1 hand over at 968 m / 1.94 km (today 428 m / 1.63 km) — about five times today's rung-0 pieces (~150 → ~770 on
  flat ground before culling), rungs 0–3 together about a fifth more. The frame cost is measured on the bench's login.
- **The swap L3 → L2:** its LARGEST mean offset against one pixel at its switch (G-MEAN, B84) — Earth's largest 4 m
  residual is 8.9–23.7 m, 2.2–5.9 pixels at 3.5 km (review A A-1), so the bench must measure the law's, not assume.
- **The whole hand-over step** against the ladder's tolerance (one cell of the coarser rung); today's 3 → 4 step on the 16 m
  ground already reaches 8.58 m at its largest at hull 19 (MEASURED). `fine_steps` extends to L3 and reads the DRAWN
  column too (the terrace escapes today's probe, review B m-4).

## 9. The stored unit, the wire, the store

- **The unit (review B M-4, review A A-12):** every level stores quarter metres today — a slope step of 6.25 % (3.6°)
  between two 4 m cells, a staircase risk on gentle ground under a low sun. The bench pictures a gentle slope at eye height
  with the L3 raster at quarter metres and at a sixteenth of a metre (the same slope step as 16 m at quarter metres); if
  the steps show, the unit becomes per level (the descriptor and the record's form; ~2–3 more bits a sample in the Rice
  code, ESTIMATE).
- **The wire (review B m-3, review A A-10/A-11):** the level byte 3 rides `FineTilePart` unchanged; every reader of today
  refuses or ignores it, so the gateway serves L3 only to clients whose minor knows it (a new `PROTO_MINOR`, as the fine
  records were gated). The acknowledgement: `FineHeld` is a frozen appended arm whose fourth byte is the pyramid level, not
  the fine level; the cure is a NEW appended arm naming the fine level (one byte a tile, client to gateway) — an SL6 ask
  (§11 Q5). Without it one shed send re-sends the whole L3 ring (~150 × ~0.3 MB).
- **The store:** ~0.3 MB a tile; local by distance; never a body-wide bake.

## 10. The code changes (review B m-2: the honest size)

| Place | Change |
|---|---|
| `fine/level.rs` | `FineLevel::L3`; a signed node shift; `margin_shift` 5; the unit per level if Q-unit says so |
| `fine/l2.rs` → a shared solve | the steady form, the rim, B96's form and the raster written ONCE for "a level under the next coarser level" (key in tiles, buffer in cells, the plane shift by one rule); L2 calls it with the 128 m flow, L3 with the 16 m flow record; L2's bytes unchanged (`L2_VERSION` stays; the 16 m test and a byte comparison prove it) |
| a new flow record (§5) | built after an L2 tile finishes; kept on the shard |
| `artifact.rs` `fine_needs_of_head` | the walk in the level's own tiles (the tile under the eye at the level's cell count, the square by true chords, the face edge by `chart_cell`, the order by the tile's centre cell) — one function for the shard and the gateway |
| `bins/src/fine_source.rs` | an array of levels, not two; L3's holder (`tx >> 2` in L2) and its flow record; the order (§7) |
| `fine/read.rs` `FineField`, `bins/src/ground.rs`, client `artifact_book.rs`, `chunks.rs` | a CHAIN of books by level index, finest first (no third special book); the shard's ground on the finest held level |
| client `ladder_view.rs` | the generalised near-piece rule (§8) |
| `fine/tile.rs` | L3's input word folds `L3_VERSION` after L2's |
| wire / gateway | the minor gate for L3; the new acknowledgement arm if Q5 says yes; the gateway's order aligned with the shard's |
| benches | `planet_stand l3`; `fine_steps` and `fine_detail` by level |

## 11. The questions for the owner (recommendations first)

- **Q1 — the price (§0, §4):** bench L3 at 4 m for rung 2 alone, with the shared 16 m flow measured (the second lever, a
  settle law on the drawn detail, was REFUTED on the calm machine at L2 the same evening: the drawn detail moves as much as
  the cell means), then decide with the measured price beside the pictures. RECOMMENDED. The
  alternatives: rungs 2–3 (~2.6× the tiles), or 2 m (~4× the tiles), or no L3 now (the curve stays; the next families —
  ledges, beds, talus — come first).
- **Q2 — the provisional ground (§7):** RECOMMENDED (a) the nearest L3 tiles first, (c) a landing tile morphs in over the
  arrival ramp, (d) L3 pre-solved at berths and arrival places; (b) the loading screen does NOT wait for L3. This states an
  exception to B86 item 1 for L3 at a new place: the 16 m ground stands until the 4 m detail morphs in.
- **Q3 — the threshold at 4 m:** tan 35° unchanged (one law at every level; Earth's 2 m plateau median 0.67–0.78 holds it)
  RECOMMENDED; the alternative ~37° (DiBiase, Heimsath & Whipple 2012, debris wedges read at 1 m) only if the bench shows a
  mismatch with Earth at 4 m.
- **Q4 — the bench terrace on the fine ground:** the column reader applies the slice-8a cap-rock bench to the held fine
  ground (`height.rs` `field_column`), its strength from the ladder's tolerance, not a law (READ). It escapes the near-piece
  bound and the steps probe (review B m-4). RECOMMENDED: the bench pictures the near ground with and without it and
  `fine_steps` reads the drawn column; the owner decides; its lawful successor is the cliff law (a later family).
- **Q5 — SL6, the acknowledgement's level:** a new appended wire arm naming the fine level (one byte a tile, client to
  gateway). RECOMMENDED yes, built with L3.
- **Q-unit (asked with the bench's pictures):** a finer stored unit at 4 m if quarter metres draw steps.

## 12. The bench (no game change; the pictures decide)

1. `planet_stand l3`: the 128 m and 16 m tiles around the target solved and finished as the shard does; the 16 m flow
   record; the L3 tiles in the middle (2 × 2), through a bench copy of the shared solve.
2. Stands: hull 17's hills, hull 19's mountain tile, planet #3's hills (B23), one tile on a face edge and one at a cube
   corner (G-SEAM).
3. Readings: the 4 m link census beside Earth's 4 m numbers; the residual against the 16 m curve (RMS, 99th, largest)
   beside Earth's; the law's slope–area plot beside Earth's; the joins after the stitch; the swap's largest offset against
   one pixel and the hand-over steps against the tolerance; passes and the settle at N and 2N; the time a tile and the
   flow record's time; the same bytes on 1 and 4 threads.
4. Pictures at eye height and low: today's curve, the L3 ground, Earth's 2 m triptychs; the gentle slope at quarter and
   sixteenth metres; the near ground with and without the terrace.
5. The owner's look and the measured price; then the build (§10) on the owner's word.

## 13. The reviews' findings and their answers

| Finding | Answer in revision 2 |
|---|---|
| B B-1 (BLOCKER) the 16 m flow undefined | §5: one flow record per L2 tile, the 128 m flow's rim water, the 128 m cell rate |
| B M-1 / A A-8 the cost a tile ×4 low; 90 % is height | §0, §2, §4 corrected |
| B M-2 the tile count ×3–4 low; minutes | §0, §4 from the code's reach; Q1 |
| B M-3 the needs' walk counts nodes | §10: the walk in the level's own tiles |
| B M-4 / A A-12 the quarter-metre unit | §9; the bench pictures; Q-unit |
| B M-5 / A A-7 the provisional ground; B86 item 1 | §7; Q2 states the exception |
| B M-6 the near-piece rule; five times the rung-0 pieces | §8 |
| A A-1 the reach's edge inside rung 3; the largest offset | §4 (rung 2 alone first), §8 (G-MEAN by the largest) |
| A A-2 "no new constant"; the law above the head | §3, §6: the head law a named fallback with its constant |
| A A-3 the threshold at 4 m | Q3 |
| A A-4 / B m-6 the settle law | §6 item 4 |
| A A-5 the shard's ground | §10 |
| A A-6 one machinery | §10: a chain of books by level index; one shared solve |
| A A-9 the water at 4 m | §5 |
| A A-10, A-11 / B m-3 the wire | §9; Q5 a new appended arm; the minor gate |
| B m-1, m-2, m-2b | §4, §10 |
| B m-4 the terrace escapes the bound | §8, Q4 |
| B m-5 the hand-over's statistic | §8 |
| B m-7 the two orders | §7 (a), §10 |
| B m-8 face edges and corners | §12 item 2 |
| A minor: HR4/HR5/P-SAME; the moon; the status and registry entries | §1, §6 item 5; the status file and DEFERRED carry this design |
