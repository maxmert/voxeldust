# The ground's colours by law (ruling B98) — design, revision 2

Status: ★ PARKED BY RULING B99 (2026-10-15): the terrain base comes first; this design's SUBSTANCE half (which material
stands at a point) moves into the base's volume step, its PAINT half waits for the look after the base. Nothing here is built. Revision 1 (2026-10-15) had two adversarial reviews — A the
rules and rulings (2 blockers, 13 major, 12 minor), B the code, the cost and the seams (2 blockers, 9 major, 11 minor);
both kept outside the repo (`/tmp/vd-b5/colours/review_A_rules.md`, `review_B_code.md`). Revision 2 answers every blocker
and major finding (§11) and REORDERS the work into three phases, because two rulings and one defect stand in front of the
colours that read the climate. Sources: `/tmp/vd-b5/colours/laws_and_data.md` (the published laws, 656 measured spectra),
`/tmp/vd-b5/colours/code_survey.md` (the code map), the code at `0548325`, the ground step design
(`docs/investigation/2026-10-02/ground_step_design.md`: F1, C12, K2, K12, rule M and its gates), DEFERRED "THE FIRST
SURFACE LOOK".

Marks: MEASURED = a tool printed it; READ = read in the code; ESTIMATE = arithmetic; UNVERIFIED = a source's claim not
checked.

## 0. The idea, and the three phases

The ground's colour at a point is **Σ share × paint, in linear RGB**. Laws give the shares from the world's facts;
measured spectra give one paint per material, the same on every body; the shares — never the colours, never the inputs —
are what a coarser rung folds (rule M, B36 item 6), so the far look is the mean of the near look.

The reviews showed that the colours split into three phases:

- **C0 — COLOURS THAT DO NOT READ THE CLIMATE** (buildable after three answers, §9): the material table from spectra;
  the rock by its kind (the bed the surface cuts) as a face and as scree (the powder row); the airless body's regolith;
  snow and glacier ice; the star's light; the shares computed ONCE in the generator and carried by the vertex (F1); rule M
  for the rock share. Two defects are cured first: the client's summer reads a dry planet (§4), and the rock law cannot
  show bare rock on the ground the law draws (§3.2).
- **C1 — THE CORRECTED CLIMATE** (a world change, the owner's word): ruling B36 item 1 — *"No stand, biome class, dune
  type or rain-shadow stand rests on the climate until the corrected climate runs"* (the ground step design's M0-9) — and
  B78's registered rain defects (the rain belt's share never renormalised, the 55° tilt's seasonal circulation missing).
  The cure is generator 27, a new first solve.
- **C2 — PLANTS AND SOILS BY CLIMATE** (after C1): the plant classes and covers, the soils' climate pigments, the season,
  and the far fold of the shares (an SL6 ask).

**MEASURED on today's rows** (`examples/cover_census.rs`, the home planet's 2 082 629 land nodes; the rain read at each
class's middle by the generator's own `decode_class`): the rain p10 / median / p90 101 / 375 / 847 mm a year; 38.4 % of
the land under 250 mm, 35.2 % at 250–650, 25.0 % at 650–1 000, 1.4 % over 1 000; Berdugo's aridity past the plant-cover
collapse (0.80) on 24.2 %; the canopy free to close (≥ 650 mm, Sankaran) on 26.4 %; the annual mean 10–20 °C on 74.8 %.
Read only as a forecast of C2 on today's climate — B36 item 1 forbids resting any class on it.

**Example.** After C0, on hull 19's mountain the flanks show the colour of the bed the slope cuts — dark where slate,
pinkish-grey where granite — with paler scree at their feet and clean white snow over the shard's own summer line; on the
home moon the same table paints grey regolith. Green slopes wait for C1 and C2.

## 1. The rules this design keeps

| Rule | How |
|---|---|
| F7 item 3, SL10, the ground design's F1/C12/K12 | THE SHARES ARE COMPUTED ONCE, in the one generator (`vd-terrain`), at the vertex, from integer inputs; the vertex carries them; the shader ONLY blends shares into paint and never derives a threshold (revision 1's per-pixel shares were a port — refused) |
| B36 item 1 | no plant class, soil class or biome rests on today's climate: C2 waits for C1 |
| T9, B23, B38, B41 | every share a named law with its calibration body; Earth's data a measuring stick and a calibration body, never a target; every stated choice named (§10) |
| B23 item 3, B36 Q16 | every body: the home planet, the home moon, the desert world, an icy body; two suns; the 150 km view |
| SL3, SL6 | the realm states what only it knows (its snow line, §4; later the far fold); each new datum is an ask |
| W16, B16, B84, B36 item 6 | rule M with its gates (§7): G-CLOSURE, G-FAR-EXACT, G-FAR-MATERIAL, G-SHADE-NEST, G-PAINT-AGREES, G-BIOME-RUNG (C2) |
| B58, B76, B80 | no 8 km squares or lattice lines in the colour: the rock kind from the bed at the vertex, never the nearest node's category; G-GRID on the colour field |
| B71 | the "warm ground" picture test (READ, `bins/tests/terrain_pictures.rs:1390`) is replaced by the colour gates; the references re-freeze only on the owner's look |
| B76 | the owner judges the table that ships (§2: one table for the bench and the build) |
| B36 Q14 | the art questions are asked (§8) |
| B74 | tests and heavy runs on the owner's word |

## 2. The paint: one material table from measured spectra

- **Method (MEASURED by the research script):** each spectrum → the CIE 1931 observer by the analytic Wyman–Sloan–Shirley
  fit (ΔE ≤ 0.43 against the table; the CC BY-SA table never enters the game) → linear sRGB.
- **The rows** (linear albedo, medians): basalt 0.10 (powder 0.20), gabbro 0.13, andesite 0.21, granite 0.27 (pinkish),
  sandstone 0.22 (red where iron-coated), limestone 0.33, shale 0.17, quartzite 0.26, slate 0.13, marble 0.36, gneiss
  0.31, schist 0.13 — each with its POWDER row (scree, talus, regolith); clean snow 0.95–0.99 (bluer as grains grow,
  Gardner & Sharp 2010), firn, glacier ice. **A REGOLITH row is owed** (lunar and Martian regolith spectra; today's
  interim 8 paints it as soil). The spread inside one kind is large (granite 0.16–0.45): one median per kind first.
- **Today's table is off:** rock 0.30 is 2–3 times too bright for basalt or slate; snow 0.85 reads dirty.
- **ONE TABLE for the bench and the build** (B76): computed from **USGS splib07 (CC0)**. The research script could not
  download it (the page refused a script) — §9 Q4. ECOSTRESS ("© Caltech, all rights reserved") stays a cross-check
  outside the repo, never the shipped table.
- **The star (§5)** sets the light's colour; the paints stay albedos.

## 3. The shares C0 computes

### 3.1 Where and how (F1)
ONE Tier-A function in `vd_terrain::surface` takes the vertex's integer inputs (its slope moments, its height over the
datum, its latitude's sine, the bed the surface cuts there, the realm's snow line) and returns K share bytes. The client's
geometry step calls it for the vertex AND for its morph target; `ParentMesh` keeps its vertices' share bytes so the target
is the barycentric blend of the parent's shares (review B M2) — the hand-over is then exact for every share. The shader
blends shares into paint and does nothing else.

### 3.2 The rock: a law at the drawn scale (review B M4, review A M4)
- **The defect, READ:** the paint reads Roering's `S_c` = 1.25 (the soil mantle's limit at the hillslope's own scale),
  but the solved 128 m and 16 m ground stands under tan 35° = 0.70 (B82, B97). So the drawn rock share `(S/1.25)²` stays
  under ~0.31: flanks only tint grey; no bare face shows (the status file saw "NO ROCK" on 2026-10-10).
- **The cure (rule M, exact):** the EXPECTED share under the slope spread the drawn cell hides,
  `E[(S/S_c)²] = (|ḡ|² + tr Σ)/S_c²` — exact for any slope distribution and linear under averaging, and the clamp at 1
  almost never binds because the fine law caps the slope at 0.70 (review B M3). The spread `tr Σ` per rung:
  (1) a client-side MOMENT PYRAMID — the mean of |∇h|² per mean level, built once per tile from its raster when it lands
  (nothing crosses); (2) zero under the finest level (rungs 0–3 draw the curve with nothing dropped); (3) far copies and
  rows: the moment shipped with each mean level — an SL6 ask with a tile-form change and a fine-version bump (C2's far
  fold; until then the far rungs use the held levels' moments, the gap stated); (4) on a body with no sea, the octave
  table's `slope_ref` (the law bound that exists there).
- **Whether that shows rock where the eye expects it** is a measurement on the bench (§6). If the sub-16 m spread is too
  small (the 16 m ground has no detail under it), the remaining lawful road is a threshold-hillslope rock law at the drawn
  scale (Heimsath et al. 1997: bare rock where the lowering passes the maximum soil production) — but it needs the lowering
  rate, which the dry world's slow read-back rate distorts (B97). §9 Q2.

### 3.3 The rock's kind (review A M9, review B M7)
The kind is the BED THE SURFACE CUTS at the vertex — the sample box already holds the stratum per cell (READ,
`client/src/chunks.rs:1349-1369`) — never the nearest 8 km node's province category, which draws 8 km squares (the ground
design's F35, G-BORDER). Beds cut by the slope give outcrop bands that follow the contours. The province borders stay the
registered interim until the outcrop rule (the ground design's C14). The home moon's sedimentary beds are a registered
T9 defect: C0 paints them as the table says and names the defect in the caption.

### 3.4 Scree
The powder row on the ground below a face, by the rock share's lower margin as a first form (a talus law later) — a
stated choice (§10).

### 3.5 Snow and ice
The realm's own summer freezing line (§4), as an EXPECTED share from an integer table of the height spread under the
vertex (no per-vertex 0/1 step, no per-pixel test — the edge's pop at a swap goes); the ICE facies bit for glacier ice;
the snow's holding angle (Bernhardt & Schulz 2010) in place of the soil's.

## 4. A defect cured first: the client's summer reads a DRY planet (review B B2)
- **READ:** since B22 the shard ships the pyramid with no water words (`bins/src/artifact_source.rs:8-12`, `:136-141`);
  the client's `wet_share_of` zips the heights with that empty vector (`client/src/surface.rs:80-91`), so its climate
  sees NO sea: the season's heat store is a rock skin, not an ocean's mixed layer — its seasonal amplitude about 9 times
  the shard's (ESTIMATE). The MEASURED client line (2.3 km over the sea at the equator, 17.7 km at the poles) is a dry
  planet's; the law's own polar line is ~7 km (ESTIMATE, UNMEASURED). The shard's ICE facies and the client's snow
  disagree.
- **The cure (SL3: the realm draws itself):** the realm states its own snow line — the three words of `SnowLine`
  (`terrain/src/surface.rs:80-111`) — in its head, once: an SL6 ask (§9 Q3). The alternatives: one word (the climate
  law's wet share) in the head; or the level-1 water words again (~4.4 MB, the dearest).
- **Registered** against DEFERRED's "THE FIRST SURFACE LOOK" interim 5; measured in the client's log before and after.

## 5. The star's light (review A M11, review B M8)
- The light's colour is the engine's white today (READ, `client-render/src/terrain.rs:4296-4307`). C0 sets it from the
  star's own spectrum: a Planck curve at the star's effective temperature through the same colour fit, kept at the same
  luminance. The charter states the class (a temperature band); the star's own temperature is a seed fact of the star —
  §9 Q5 (read the class's band centre as a named interim, or state the temperature).
- ONE convention: the paints stay albedos; the light carries the star's colour; the eye's adaptation (the exposure and
  the white point) is the ground design's own law, not an art choice (review A M8).
- **The sky changes with it:** the vendored air and the sun's disc read the light's colour (review B M8, the vendored
  `functions.wgsl`), so a coloured light changes the sky the owner froze in B32 — the references re-freeze only on the
  owner's look (B71).
- The research measured one RGB light under G, F and A stars as exact enough (ΔE p95 0.97 at 5 772 K); K and M stars
  need the light's own colour per class (ΔE p95 up to 8.5 at 2 700 K).

## 6. The C0 bench (buildable after §9's three answers)
- The table from splib07; the shares per vertex through the one function, own and target, the parent's shares as the
  target; σ from the held rasters' moment pyramid (σ = 0 where none, said in each caption); the vertex widened for the
  bench and its cost read (§8).
- Stands (B36 Q16): hull 19's valley and hull 17's hills (eye height, low, crest), the home moon, the desert world
  (`Planet(17782101822166660494)`: no sea, so no fine tile — its caption says so), the 150 km view, two suns (a high and a
  low star). One picture across the 9 → 10 band that SHOWS the colour's drift (no far fold yet), with its number under
  B27's eye rule.
- Readings: the gates of §7; the rock share's distribution against the drawn slope; G-GRID on the colour field; the
  brightness step at each swap; the frame cost and the vertex count before and after.
- Then the owner's look; then the build; then the 4 m level again with colours (B98 item 3).

## 7. The gates (rule M, the ground design's own)
G-CLOSURE (the shares sum to one), G-FAR-EXACT (a coarser rung's shares equal the mean of the finer), G-FAR-MATERIAL (the
far colour equals the mean near colour within the eye's step), G-SHADE-NEST (the shade nests across a swap), G-PAINT-AGREES
(the paint the volume's topsoil cell names agrees with the share — the noise biome's topsoil, a T9 defect, is registered,
review A M12), G-GRID (no lattice direction in the colour field); G-BIOME-RUNG in C2.

## 8. Cost (review B M1, M5)
- Each share needs an OWN and a TARGET byte: K shares cost 2K bytes a vertex. C0's K is about 4 (rock face, scree, snow,
  ice; soil the remainder): +8 B → 40 B a packed vertex. At B89's 18.8 M vertices (counted before the 16 m level; today's
  count UNMEASURED) that is ~+150 MB of GPU memory and ~25 % more upload frames (ESTIMATE). Measured on the bench.
- The far fold (C2): 6–8 share planes, 17.7–23.7 MB a planet raw, ~8–11 s more far view at the gateway's pace — with its
  wire form (a field append, or a new arm), its cut and its arrival rule (a ramp per chunk) stated then.

## 9. The questions for the owner (recommendations first)

- **Q1 — the order (B36 item 1):** C0 now (colours that do not read the climate) → C1 the corrected climate (generator 27,
  a world change: the home planet and the far planets solve again, the fine tiles solve again) → C2 plants and soils by
  climate. RECOMMENDED. The alternatives: amend B36 item 1 to bench C2 on today's climate (pictures only); or bench the
  plant laws on Earth's real climate as the measuring stick (B79 Q1) beside C1.
- **Q2 — the rock at the drawn scale:** the expected share under the hidden slope spread (exact by rule M; the moment
  pyramid on the client, nothing crosses for near ground) RECOMMENDED, measured on the bench; a threshold-hillslope rock law
  only if the bench shows too little rock.
- **Q3 — SL6, the snow line:** the realm states its three snow-line words in its head (the client's summer then matches
  the shard's) RECOMMENDED; or one wet-share word; or the water words again (~4.4 MB).
- **Q4 — the data:** the owner downloads USGS splib07 (CC0) in a browser (the page refused a script), and one table serves
  the bench and the build RECOMMENDED; or the ECOSTRESS-derived numbers with credit (the licence says "all rights
  reserved": the owner's call).
- **Q5 — the star's temperature:** the class band's centre as a named interim RECOMMENDED for C0; the star's own effective
  temperature stated by the star system (a seed fact; SL6 when it crosses) later.
- **Q6 — the art (B36 Q14):** what the art step owns — the micro-texture inside a material (its mean held on the paint),
  the tree and grass assets (V4), the fine BRDF shape — and when: RECOMMENDED after C2, with the forest.
- **Q7 — life on a body (review A M6, B41):** whether a body carries plants at all is not a fact the charter states. A
  seed identity choice (T9 allows identity draws), or a law of the charter (water, air, temperature, age)? To answer
  before C2.

## 10. Stated choices (B38: each named)
The share function's integer form; K and the bytes a share; the moment pyramid's levels; scree on the rock share's lower
margin (first form); the snow's expected share table; the class band's centre for the star (interim); the province borders
(interim until C14).

## 11. The reviews' findings and their answers

| Finding | Answer in revision 2 |
|---|---|
| A X1 / B B1 (BLOCKER) shares per pixel = a port (F7 item 3, F1, C12, K12) | §1, §3.1: one generator function at the vertex; the shader only blends |
| A X2 (BLOCKER) biome classes on today's climate (B36 item 1) | §0, Q1: C2 waits for C1 |
| B B2 (BLOCKER) the client's summer reads a dry planet | §4, Q3 |
| B M4 / A M4 no bare rock under tan 35° | §3.2, Q2 |
| A M1 / B M5, M6 the far fold's size, wire, arrival; folding inputs breaks rule M | §8, C2; fold shares only |
| A M2 / B M3 rule M beyond the rock; σ's source; the gates | §3.2, §3.5, §7 |
| A M3 Kubelka–Munk is not linear | C2: soils fold their end-member AREA shares (linear); KM only inside one material's own paint |
| A M5 every body; the regolith row; the stands | §2, §6 |
| A M6 the plant laws; life on a body | C2; Q7 |
| A M7 soil brightness | C2 (the powder rows are not soils) |
| A M8 the art questions; exposure | Q6; §5 |
| A M9 / B M7 8 km squares, lattice lines | §3.3, §7 G-GRID |
| A M10 the dry sea floor | C0 paints the sea floor by its bed (no plants, no climate soil); C2 excludes the sea floor; the water fill later |
| A M11 / B M8 the star's light | §5, Q5 |
| A M12 the topsoil cell from noise | §7 G-PAINT-AGREES, registered |
| A M13 two tables | §2: one table |
| B M1 the vertex cost | §8 |
| B M2 the target share | §3.1 |
| B M9 no gates | §7 |
