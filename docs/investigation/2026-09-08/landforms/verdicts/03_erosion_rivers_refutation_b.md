# Refutation B (round 2) — DOMAIN 03, erosion, rivers, coasts and ice

**Target:** `docs/investigation/2026-09-08/landforms/03_erosion_rivers.md`, revision 2.
**Lens:** believability, cost and the owner. Would the vista look like the reference picture? Would a
geologist accept it? Does it fit the 8 ms chunk and the boot? Does it answer what the owner asked?
**Method:** read-only. I re-opened every code citation and re-computed every number the document states.
The code citations are good; revision 2 fixed them. The ARITHMETIC OF THE NEW DESIGN is where it breaks.

**Verdict in one line.** The headline change — THE SLOPE SPECTRUM (§4.2) — is normalised in a way that
makes the whole home planet stand at 53°, it applies the same roughness to a plain and to a mountain
range, and it multiplies the ladder's rung-to-rung error by five. The second headline change — the realm
SHIPS the artifact (§12 D3) — has no lane, reverses the reason SL10 exists, and moves the cross-host
drift onto the server, where a rescheduled realm re-addresses every stored edit.

---

## Blockers

### B1 — The slope spectrum's own normalisation forces a 53° planet, and the two cures named cannot reach the scale that makes it

§4.2 states the law in three lines: `s(o)` is a bump in the octave index, `a(o) = s(o)·λ(o)/2π`, **then
scale every `a(o)` so that `Σ a(o) = relief_m`.** That last line is not free. It FIXES `s_peak`.

COMPUTED, home planet, `relief = 12 000 m`, `o_peak = 7`, `σ = 1.4`, 14 octaves from 400 km:

* `Σ s(o)·λ(o)/2π = 12 000` gives `s_peak = 0.7588`. The document's own table (0.759 at 3.13 km) agrees,
  so this is the design, not a misreading.
* The per-octave slopes are then 0.058, 0.077, 0.107, 0.154, 0.237, 0.383, 0.609, **0.759**, 0.609,
  0.383, 0.237, 0.154, 0.107, 0.077.
* The root-mean-square surface slope is `sqrt(Σ s(o)²) = 1.345`, which is **53.4°**.
* The FINE octaves alone (`o = 6…13`, the ones that survive after `Z` replaces the coarse six) give
  `1.250`, which is **51.3°**.

The document sees the overshoot and says: *"The overshoot is the talus rule's job."* It names two cures.
Neither can act at the scale that makes the overshoot.

* **§4.8, the talus relaxation, runs on the MACRO lattice — 8 224 m nodes.** The overshoot is made by
  the octaves at 6.25 km, 3.13 km and 1.56 km. Two of the three are at or below one macro node. A
  relaxation on an 8 224 m lattice cannot remove a 1.56 km ridge; it does not see it.
* **§7.3 rule 2, "scree by slope", reads a stencil of the FOUR NEIGHBOURING CELLS** at that rung — one
  metre at rung 0. A one-metre stencil measures the slope of the 3.13 km octave correctly (0.76) and
  then *"pulls the surface toward the local mean"*. Pulling a 1 m neighbourhood toward its own mean
  removes nothing at 3 km; it only rounds the metre-scale bumps. To take 0.76 out of a 3.13 km octave
  you must remove the octave.

So the design states a spectrum, computes that it is over the angle of repose everywhere, and hands the
problem to two passes that cannot see it. **This is the largest change in revision 2 and it does not
close.** §14 M9 is offered as the settlement, but M9 CALIBRATES `s_peak` — and `s_peak` is not free,
because `Σ a(o) = relief_m` already determined it. §4.2 also says `s_peak` is *"a stated share of the
talus tangent the bedrock gives"*, which is a THIRD determination of the same number. Three rules set one
value and the document does not reconcile them.

**The fork the document must state and does not:** either `Σ a(o) < relief_m` (lawful — the band is
derived from `relief_m`, so spending less is safe and the envelope still holds), or the relief budget
itself must fall. The document commits, in writing, to the branch that gives a 53° planet.

*In the game:* the pilot lands on the home planet and cannot walk. Every square metre, on the plain and
on the ridge, stands at about 53°. The collider is the same shape, so he slides.

### B2 — The spectrum is GLOBAL. The reference picture's fields, plains and valley floors cannot exist

The recipe holds ONE octave table per body (`crates/terrain/src/body.rs:158-184`), and §4.2 redistributes
it globally. Every column of the home planet therefore gets the same roughness at every scale. Real land
does not work that way, and the reference picture is half made of the exception: flat fields beside the
river, a broad valley floor, a plain running to the forest edge, and a mountain range behind it.

The document's only flat ground is the floodplain (§7.2), and §6.1 COMPUTES the drainage density of the
macro graph at **0.122 km of channel per km²**. So flat ground exists on about a hundredth of the
surface, in strips beside one river per 8 224 m node. Everywhere else the planet is at the spectrum's
slope.

Two consequences the document never states:

* **There is nowhere to build.** §2's own gloss on the floodplain says it is *"the only ground flat
  enough to build a settlement on without terracing"*. With a globally uniform 51° fine spectrum that
  sentence is the whole story of the planet, and the owner's world is a building game.
* **A geologist reads it immediately.** Earth's mean land slope is a few degrees; even the Himalaya's
  hillslopes sit near 30–35°. A planet at a uniform 51° is a talus heap, not a landscape.

The missing piece is a **roughness field** — a slowly varying, seed-derived modulation of `s_peak` (or of
`o_peak`) across the body, so a craton is smooth and an orogen is rough. That is exactly the thing the
owner asked for when he said the biomes must depend on the planet's facts. The document has no such
field, does not name one as owed, and §12 has no decision row for it.

### B3 — The spectrum multiplies the ladder's rung-to-rung error by five, and §11 claims the gate on `Z` alone

`dropped_bound_m(rung) = relief_bound_m(0) − relief_bound_m(rung)`
(`crates/terrain/src/body.rs:274-276`), and `octaves_at` drops the FINEST octaves BY COUNT
(`crates/terrain/src/body.rs:253-258`). The bound is therefore the SUM OF THE DROPPED AMPLITUDES — and
the slope spectrum moves amplitude INTO the octaves that get dropped first.

COMPUTED, home planet, 14 octaves, `relief = 12 000 m`:

| rung | cell | dropped bound TODAY | dropped bound UNDER §4.2 | ratio |
|---|---|---|---|---|
| 5 | 32 m | 22.7 m | 69.4 m | 3.1× |
| 6 | 64 m | 46.1 m | **220.9 m** | 4.8× |
| 7 | 128 m | 93.0 m | 598.3 m | 6.4× |
| 8 | 256 m | 186.8 m | **1 204.5 m** | 6.4× |

§11's ladder row says: *"`Z` is identical at every rung and contributes zero to `dropped_bound_m`."*
That is true and it is not the question. The question is what the OCTAVES do, and the answer is that the
far view of a ridge moves by 221 m instead of 46 m when the client changes from a 32 m rung to a 64 m
rung. SL8 calls a visible jump a defect, and the eleven-seam taxonomy names "detail-by-box" for exactly
this. §14 M8 measures only *"the pop at a rung transition where a river arrives"* — the river, not the
octaves.

The vista the document sells is the one this breaks. At 50 km, with 720 rows at 45°, one pixel subtends
COMPUTED 54 m, so the far ridge is drawn at a 32–64 m rung. §3's promise — *"the spurs on the far side
stand 300 m tall instead of 47 m"* — is a promise about octaves 7 and 8, and octave 8 is the first one
the ladder throws away at that rung.

**The design must state a new coarsening rule** (drop by WAVELENGTH against the cell, the way §4.2
already demands for the macro solve), or state and budget the new bound. It does neither.

### B4 — "The realm SHIPS the artifact" has no lane, and it reverses the reason SL10 exists

§12 D3 recommends (b) and calls it *"one SL6 ask on a statement that already exists"*. Two things are
wrong with that sentence.

* **9.83 MB does not ride `SurfaceStmt`.** `crates/core/src/look.rs:79` states
  `SELF_LOOK_BUDGET_BYTES = 1200`, *"one conservative datagram"*, and the whole self-look bag — outline,
  luma and surface together — must fit inside it. The 28 bytes of facts fit. The artifact is **8 192
  times the whole bag.** The document's own §13 open question 5 admits it: *"neither holds a place for a
  per-body derived blob today"*. So D3's recommendation is a new bulk transport: chunked, retained,
  resumable, invalidated by the world tag, on the reach path, for every body a player approaches. That is
  a subsystem, and it is presented to the owner as an ask beside an existing field.
* **It reverses SL10.** Ruling V1 exists so the client DERIVES the static shape instead of receiving it —
  one generator crate, both hosts, no drift, no bytes on the wire. The erosion artifact is static shape
  that the seed alone decides. Making it a ship is the largest architectural reversal in the document,
  and §11's SL10 row does not mention it; it only says the shape is a function of the seed and the stated
  facts.

**The cost nobody computed.** A system with eight planets and twenty moons is 28 bodies. At 9.83 MB each
that is 275 MB per player who tours it, and no machinery that exists caches it. The document computes the
SOLVE time (12–40 s) and never computes the SHIP time. UNMEASURED, and it is the number D3 turns on.

### B5 — The shipped facts move the drift from the client to the server, where a rescheduled realm re-addresses every stored edit

§5.2 makes the argument correctly and then stops one step short. Its own words: *"A surface temperature
quantised to 1/16 K has NO margin: two hosts whose `powf` differs by one ulp near a quantum boundary land
on opposite sides of it, the ELA moves 10 m, the ice pass carves a different valley, and the artifact
differs."*

That argument does not depend on the host being a CLIENT. It depends on the arithmetic being
transcendental and the output being quantised. `crates/physics/src/taxonomy.rs:21-24` records the
deferral for exactly this code, and the project runs shards on BOTH targets — an M4 host and x86-64 k3d
pods — and reschedules a realm from one pod to another as a shipped feature.

So under D3(b): the home planet's realm boots on an aarch64 host, computes `t_eq_k · (1+0.75τ)^(1/4)`,
quantises to 1/16 K, solves, and ships. The demand loop later spins it down; it boots on an x86-64 pod,
recomputes the same expression, and lands one quantum away. The ELA moves, the ice pass carves a
different valley, the artifact digest changes, and **every stored edit on that planet sits on ground that
moved** — the exact event §5.5 and §4.12 exist to prevent.

The same hazard exists across SERVER BUILDS: nothing pins the facts, and §5.5's world-tag list does not
contain them. §5.2 says the digest must fold the stated facts, which DETECTS the change; it does not
prevent it, and detection after the fact means refusing a planet a player has built on.

**What the document owes:** the facts must be PINNED literals per body, in the
`crates/terrain/src/home.rs:16-22` shape (a stated value plus a test that the forest still draws it), not
recomputed at each boot. §12 D6 asks for six drawn facts and never says who computes them, when, or what
pins them.

---

## Defects

### D1 — The 8 ms budget is checked against a baseline this design deletes

§7.4, §10 and §11 all charge the new terms against the MEASURED 6.1 ms cave-dense chunk
(`crates/bins/examples/terrain_cost.rs:38-41`). That measurement was taken on TODAY'S recipe. §4.2
changes the recipe. A measurement of a surface the design replaces is not a baseline.

Two mechanisms move it, both COMPUTED from §4.2's own numbers:

* **More chunks hold the surface.** The mean surface slope rises from 0.353 to 1.345, so the surface
  crosses `sqrt(1 + s²)` = 1.06 → 1.60 chunk columns per column of ground: about **50 % more surface
  chunks per unit area**. The `AboveSurface`/`BelowSurface` skips (`crates/terrain/src/chunk.rs:216-247`)
  fire on fewer chunks, and each non-skipped chunk pays the full box.
* **The extractor works harder per chunk.** The same file states a MEASURED synthetic worst case of
  **26 ms** for a 3-D checkerboard (`terrain_cost.rs:43-45`), called *"a BOUND no chunk of THE world
  reaches"*. Roughening the world by a factor of five in the 1–6 km band moves THE world toward that
  bound, and nobody has measured how far.

So the honest statement is not *"6.1 + 1.15 = 7.25 ms of 8 ms"*. It is: the baseline is UNMEASURED under
the new recipe, and §14 M2 must bench the four cases **after** §4.2 lands, not before.

### D2 — The discharge byte's range fails on the document's own two ceilings

§4.5: *"the largest possible `Q` on the home planet is `1.41 × 10^14 m² × 10 000 mm = 1.4 × 10^18`"*.
§6.1: *"the ceiling is the whole surface at 1 000 mm, `1.41 × 10^17`"*. The same document, two ceilings,
a factor of ten apart.

COMPUTED with §6.1's own floor of `6.76 × 10^7`:

* against the §6.1 ceiling: `2^30.96` — fits, as claimed;
* against the §4.5 ceiling: **`2^34.28`** — does not fit in `2^32`, and there is no octave to spare.

The floor is also soft in two ways the document does not name. It uses `8 224² = 67 634 176 m²`, which is
the node area at a FACE CENTRE; §4.1's own table says a corner node is 0.702 of that, and COMPUTED the
MEAN node area on the home planet is `1.411 × 10^14 / 2 457 600 = 5.74 × 10^7 m²`, 15 % lower. And it
assumes a minimum precipitation of 1 mm/yr; nothing in this document or in DOMAIN 02 states that `P` has
a floor, and `P = 0` over a hyper-arid basin gives `Q = 0`, which a logarithm cannot hold at all.

The fix is small — 256 steps over 35 octaves is 4.9 % half-step error on `Q` and 2.4 % on a width, still
fine. But the "KEPT" answer to refutation A's A-D2 rests on the arithmetic, and the arithmetic is wrong.

### D3 — A gorge at every lake inlet: the sweep's receiver term is undefined at the lake edge

§4.4's table says the stream-power sweep reads `z_terrain` and SKIPS a node whose `z_flood > z_terrain`.
It says nothing about a node that is NOT in a lake whose RECEIVER is.

Follow it through. Outside a lake `z_flood = z_terrain`. A river reaches the lake shore: its receiver is
the first lake node, whose `z_flood` is the lake surface but whose `z_terrain` is the LAKE FLOOR, which
the flood deliberately left as a hole (*"Skipping it leaves the hole exactly as the seed drew it, which is
what a lake IS"*). The sweep computes `S = (z_i − z_receiver)/L` from `z_terrain`, so `S` is the drop to
the lake FLOOR, not to the lake SURFACE. On a deep lake that is hundreds of metres over 8 224 m, and the
implicit update pulls the inlet down toward the floor on every pass.

*In the game:* the pilot follows the river down to the mountain lake §7.2 promises, and the last 8 km
before the shore is a gorge the depth of the lake. Forty passes make it worse each time.

The cure is one sentence the document does not have: **a river's base level at a lake is the lake's SPILL
LEVEL**, so the receiver term must read the spill level for a lake node, exactly as §7.2 already does for
`z_w`. §4.4's table needs a fourth row and §7.2's gate needs a fourth statement.

### D4 — The pyramid's top level is 4.8 KB, not 19 KB, and D3's cost argument quotes the wrong one

§4.13: *"a coarse PYRAMID of `Z` (levels 320…20) | `i16` | +1.6 MB; top level 19 KB"*. COMPUTED at
`6 · n² · 2` bytes:

| level | nodes per edge | bytes |
|---|---|---|
| 320 | 320 | 1 228 800 |
| 160 | 160 | 307 200 |
| 80 | 80 | 76 800 |
| 40 | 40 | **19 200** |
| 20 | 20 | **4 800** |
| total | | 1 636 800 = 1.64 MB ✓ |

So 19 KB is level 40 and the stated top level 20 is 4.8 KB. The two statements cannot both be true. The
number appears four times (§0.2, §4.13, §10, §12 D3) and it is the number D3's recommendation leans on
(*"19 KB covers the approach"*). Either the pyramid stops at 40, or the figure is 4.8 KB.

### D5 — The design REMOVES cliffs, and no rule in it MAKES one; the "rock pillars" claim is false

The reference picture's strongest features are a vertical cliff band and free-standing rock pillars.
§7.3 rule 2 claims them: *"That is the reference picture's rock pillars and their debris."*

It is not. Read the two rules the design actually has:

* §4.8 talus: *"where the drop exceeds `L_ij · tan(θ)` … move half the excess to the lower neighbour"* —
  this REMOVES every slope above 35°. It is a cliff DESTROYER.
* §7.3 rule 2 scree: *"subtract a term that pulls the surface toward the local mean"* — this rounds the
  slope down. It is also a cliff destroyer.

Nothing anywhere adds a vertical face. A sum of band-limited noises is continuous and bounded, so a 90°
face cannot appear by itself. Real cliffs come from three things, and the design has none of them at the
right scale: a resistant BED (the strata stop 23–92 m down, and §7.3 rule 4 admits it), glacial
oversteepening (present, but only as a macro trunk cut at 8 224 m), and river undercutting (present, but
only inside the channel's own width).

A rock pillar is harder still: it needs differential erosion of a bedded rock, which §12 D14 says the
world does not draw. Claiming the scree rule produces one is a believability claim with no mechanism
behind it, and the vista test (M3) will show it.

### D6 — Twenty side valleys, and every one of them is dry

§6.1 sells the picture: *"the pilot on the ridge sees ONE macro river and, between him and it, twenty
side valleys the closed-form pass drew, all running the same way."* It also states, plainly, that *"water
is drawn only where the macro graph says"* (§7.3 rule 5).

Put the two together. A 50 km vista on the home planet spans about six macro nodes, so it holds about six
channel links. Everything else — the twenty side valleys the slope spectrum and the gully warp draw —
carries no water at all. COMPUTED, the drawn drainage density is 0.122 km/km²; the valley density the eye
reads is 0.33–0.67 km/km². **Between a half and four fifths of the visible valleys are dry.**

On Earth, a mountainside with twenty valleys has twenty streams. A dry valley is a specific and rare
landform — a karst valley, a wadi, a relict glacial channel. Making it the normal case is the kind of
thing a player feels before he can name it, and it is the opposite of the reference picture, where the
water is the thing the eye follows.

The document raises the density as a NUMBER in open question 8 and never states the VISIBLE consequence.
A sub-macro stream — even a closed-form one with no discharge, drawn where the gully field's flow
converges — is the missing piece, and it is not designed.

### D7 — The divisor rule's "integer lookup" does not apply at the only entry point the generator has

§4.1 justifies `n_macro | N` with: *"it makes the per-chunk lookup INTEGER: a rung-`L` cell index `i` maps
to its macro node by `(i · 2^L) / (N / n_macro)`, an exact integer division with no float comparison
anywhere."*

But the generator's entry point is `height_m(body, dir, rung)`
(`crates/terrain/src/height.rs:16-27`), and §7.1 keeps that signature. `dir` is a unit direction, not a
cell index. To find the macro node from `dir` you must INVERT the bend — `unbend`
(`crates/seed/src/bend.rs:52-65`), four Newton steps with a division each, on two axes — and then floor.
Every caller that is not the column pass goes this way: the halo columns in `sample_box`
(`crates/terrain/src/lattice.rs:246-280`), `vertex_position_m`, the digest's golden chunks, and every
test.

Two costs the document does not charge:

* **the unbend itself**, ESTIMATED 50–100 ns per column for two axes, so **0.2–0.4 ms per rung-0 chunk**
  on 3 844 core columns — comparable to the whole 0.55–1.15 ms the design claims for everything;
* **a signature change if it is avoided instead** — passing the address into `height_m` breaks the very
  property §7.3 rule 1 relies on (*"a pure function of ONE direction"*) and touches `extract`, `digest`,
  `position` and `carve`. §8.3's *"nothing in the cell record changes"* is true; *"nothing in the
  generator's shape changes"* is not stated and is not true.

### D8 — Catmull-Rom is undefined at a cube corner, and G-MACRO-CORNER does not test it

§5.4 rule 3 gathers a 4-wide Catmull-Rom stencil through the seam table. At a cube CORNER only three
faces meet, so the 4 × 4 stencil for a node one or two nodes from the corner reaches into a quadrant that
does not exist — the same fact §4.1 already states for the D8 stencil (*"a corner node has SEVEN
neighbours, not eight"*).

§5.4's gate list holds G-MACRO-CORNER, and it asserts only *"the node has exactly seven neighbours and
its receiver is one of them"* — the D8 stencil, not the interpolation stencil. So `Z`'s value near each of
the eight cube corners is undefined by the design and untested by the gate. Eight corners on the home
planet, each covering of the order of two macro nodes — COMPUTED, about 2 200 km² of surface with no
stated rule, on a planet the player can walk.

### D9 — "There is no soft rock in the world" is overstated, and it drives an unnecessary ask

§4.6 and §12 D14 both rest on: *"all five are hard igneous or metamorphic rock … There is no soft rock in
the world today to key a wide wave-cut shelf or a soft-rock badland on."*

The five BEDROCKS are indeed hard — Granite, Basalt, Gabbro, Andesite, Quartzite
(`crates/terrain/src/strata.rs:145-151`), confirmed. But the body also draws ONE SEDIMENT of three —
**Sandstone, Limestone or Shale** (`crates/terrain/src/body.rs:194,199`) — and it is **20–80 m thick**
(`crates/terrain/src/body.rs:198`). Those are precisely the soft rocks the two named landforms need, and
both landforms live inside that thickness:

* a wave-cut platform is 10–500 m WIDE and only metres DEEP — well inside 20–80 m;
* a badland is a soft-sediment landform whose relief is tens of metres, not hundreds.

So the shelf and the badland CAN be keyed today, on the sediment the world already draws. §12 D14's
lithology ask is needed only for the deep case (a 1 km canyon wall), which §7.3 rule 4 already routes
around with strath terraces. Asking another domain for a field you do not need is a cost the owner pays
in a slice he did not have to buy.

### D10 — §11 claims the HR5 gate passes while §14 M7 marks the enabling fact UNMEASURED

§11's HR5 row states the gate as passed: *"COMPUTED: a 200 km moon of THE world has 6 144 macro nodes …
so a full solve runs inside a unit test in milliseconds even instrumented."*

§14 M7 states the opposite about the thing that matters: *"UNMEASURED: whether THE world holds a body
that is both small and wet enough to reach the fluvial arms."*

HR5 is 100 % REGION AND BRANCH in a Tier-A crate. The arms this design adds are data-dependent — a pit, a
lake, a no-sea planet, a dry planet, a corner node, a seam-crossing river, a delta, a fjord, a glaciated
node. A dry 200 km moon reaches none of the fluvial ones. So the gate's pass depends on a body nobody has
found, and the document names no fork if it does not exist. The two lawful branches are (a) run the home
planet under instrumentation — COMPUTED 12–40 s per solve before instrumentation, which is minutes per
test binary — or (b) invent a body, which SL5 forbids. Neither is named.

Claiming a hard rule's gate is passed on an UNMEASURED premise is the pattern the "never assume, measure"
standing rule exists to stop.

### D11 — A 12–40 s single-threaded solve inside a shard, with nothing said about the tick loop

§10 states the boot cost honestly as a wait: *"A cold first boot of the home planet's realm answers no
chunk request for 12–40 s"*. It never says WHERE the solve runs.

Every shard is a `bevy_ecs` tick loop. §5.3 rule 6 forbids parallelising the solve. So a naive
implementation blocks the tick for up to 40 s. The project has already met this failure once — a shard
whose ticks doubled wedged the realm and tripped the gateway's readiness machinery. A blocked tick is not
a slow boot; it is a realm that stops answering the lane, and the peer book reads that as a lost
connection.

Moving the solve off the tick thread is the obvious answer, and it has its own consequence the document
must state: the tick at which the artifact becomes available is then a function of wall-clock scheduling,
so the shard must REFUSE chunk requests until it lands rather than answer them from a partial field.
Neither sentence is in the document.

---

## Weaknesses

### W1 — Two of the owner's five named inputs never reach the shape

The owner asked for biomes dependent on *"the planet position, spin, trajectory, size and gravity"*.

* **Size and gravity** reach the shape: `f_gravity` in `K` (§4.6), the hydraulic geometry's `g` factors
  (§6.1), `h_wave` (§4.10). Good.
* **Position** reaches it through `t_eq_k`, which is a function of the insolation. Good.
* **Spin** is one of the six facts §12 D6 asks for and one of the 28 bytes §5.2 ships — and **nothing in
  this document consumes it.** It enters only through DOMAIN 02's precipitation, which is not written.
* **Trajectory** — the orbit's eccentricity, which drives the seasonal swing and therefore the ELA's
  range — is **never mentioned once**, although `crates/physics/src/worldgen/generate.rs` draws `ecc` for
  every planet.

So the document asks the owner to approve a large cross-crate change — six new body facts in
`vd-physics` — while leaving one of the six unused and one of his five words unaddressed.

### W2 — The first slice is not buildable, and no minimal slice is named

§12 holds fifteen decisions. Five of them are prerequisites in OTHER crates or domains: D3 (a new bulk
lane), D6 (six facts in `vd-physics`), D12, D13 and D14 (three fields from DOMAIN 02 and the weather
domain). §4.5's fallback (`P = 1`) covers one of them, and §4.6's placeholder covers part of another.

Nothing in the document says what CAN be built first, in what order, with what still stubbed. Compare the
format the owner is used to: `docs/investigation/2026-09-07/slice_07_client_link.md` §11 is "how it is
built". §11 here is the law gates instead, and there is no build order at all. A document that decides
nothing by itself still owes the owner the shape of the first commit.

### W3 — The 64 m under the crust is consumed to zero, and nobody says what it was for

§4.12: *"Setting `MAX_CHANNEL_DEPTH_M = 64` keeps that inside `crust_m` with nothing to spare and nothing
to change."* Confirmed against `crates/terrain/src/body.rs:234`:
`crust_m = relief_whole + strata.max_depth_m() + caves.max_depth_m + 64`.

The document treats the 64 as SLACK. The code's own comment names the purpose of the 64 ABOVE (*"room to
stand on the highest peak"*) and not of the 64 BELOW. A design that spends an undocumented margin to
exactly zero should say what it believes the margin is for. One candidate the document does not consider:
the extractor needs a rock cell BELOW the deepest cave for the mesh to close, and `below_surface_cell`
(`crates/terrain/src/chunk.rs:225-231`) is the fill that depends on it.

### W4 — "Real eroded topography is NOT self-similar; its roughness PEAKS at the valley spacing" carries no source

That one sentence in §4.2 is the physical justification for the whole slope spectrum, and it is the only
claim in the document with no citation, no computation and no measurement. The published picture is a
spectral BREAK at the hillslope-to-valley transition, not a PEAK: above the break, measured topographic
spectra follow a power law over decades. A peak and a break give different amplitude ladders, and the
design's `σ = 1.4` bump is a peak.

The document is otherwise scrupulous about labelling. Here it presents geomorphology as settled to
justify its largest change. §14 M9 is the right measurement; it should be named as the thing that decides
whether the LAW is a bump or a break, not only what the bump's parameters are.

### W5 — The solve's headline is probably still low, and D11 turns on it

§10 charges one priority flood at 1.0–1.5 s for 2.46 M nodes on a 20 MB working set. That is 0.40–0.61 µs
per node for one push and one pop through a 21-level binary heap. ESTIMATED with a per-miss model — five
to eight cache misses per heap operation at about 80 ns each — the figure is 1.3 µs per node, so
**3.2 s per flood** and 12.8 s for the four floods alone. The headline then sits nearer 21–25 s than
12–20 s.

The document already marks M1 as blocking D3 and D11, which is right. The weakness is that the range
offered (12–40 s) is presented as bracketing the answer when its lower half rests on a per-operation
budget that a heap on a 20 MB working set does not meet.

### W6 — The water surface's second mesh is charged in time and not in memory or draw calls

§8.3 runs the extractor a second time per water-crossing chunk. §7.4 charges 0.3–1.0 ms. Ruling V10's
MEASURED client figure is 405 KB of geometry per chunk on one thread. A second mesh is a second vertex
buffer, a second index buffer, a second material and a second draw call per chunk, and coastal and lake
chunks are exactly the chunks a player looks at. None of that is stated, and the 8 ms budget covers
worker time only.

### W7 — A macro node's area is quoted at its largest value in three arguments

§6.1 uses `8 224² = 67 634 176 m²` for the channel-head argument and for the discharge floor, and §4.1's
own table says the area density varies by 24–30 % across a face. COMPUTED, the MEAN node area on the home
planet is `1.411 × 10^14 / 2 457 600 = 5.74 × 10^7 m²` — 15 % below the quoted figure — and a corner node
is lower still. The conclusions survive; the practice of quoting the face-centre value as *"one macro
node is 67.6 km²"* does not, in a document that is otherwise careful to state a number's source.

### W8 — "Facies" is used for two different things

§6.1 defines the facies as 3 bits over eight DEPOSITIONAL ENVIRONMENTS: upland, floodplain, delta, beach,
scree, glacier, lake bed, sea bed. §7.3 rule 2 then says *"set the facies to `Gravel`"* and §4.10 says
*"the facies is `Sand`"*. `Gravel` and `Sand` are `Stratum` values
(`crates/terrain/src/strata.rs:186-200`), not members of that list. One word, two meanings, in a document
whose §2 exists to fix one meaning per word.

---

## Notes

* **N1 — "a gentle, featureless spectrum" mis-describes the code.** COMPUTED at `k = 0.5`: the per-octave
  slope is 0.094, and over 14 octaves the RMS surface slope is 0.353, which is 19.4°. That is not gentle;
  Earth's mean land slope is a few degrees. The real defect is SELF-SIMILARITY — no characteristic scale,
  so no orienters — which the same paragraph names correctly two sentences later. The word "gentle"
  points the reader at the wrong cure, and it is the word that made the design reach for MORE amplitude
  rather than for a roughness FIELD (B2).
* **N2 — §3's vista claim cannot be checked because no rung is named.** *"The spectrum decided that the
  spurs on the far side stand 300 m tall instead of 47 m"* is a statement about rung 0. At 50 km the
  client draws a 32–64 m rung (COMPUTED: one pixel subtends 54 m at 720 rows and 45°), and B3 shows that
  is where the new bound bites. The vista's arithmetic should be stated at the rung the vista is drawn
  at.

---

## Checked, sound

Each of these I re-derived or re-read against the code and found correct. They are the document's
strongest parts, and no time should be spent re-arguing them.

* The area density table in §4.1: recomputed from `K1 = π/4`, `K2 = 0.15`, `K3 = 1 − K1 − K2`
  (`crates/seed/src/bend.rs:23-30`) with `W'(a)·W'(b)/|n + W(a)u + W(b)v|³` — centre 0.616850, edge
  midpoint 0.432739 (0.702), corner 0.467391 (0.758). Exact agreement, and the sign correction against
  revision 1 is right.
* `n_macro = 640` divides `N = 5 263 360 = 2^12 · 5 · 257`; `642` does not; node 8 224 m; 2 457 600
  nodes; 4 bytes each = 9.83 MB. All confirmed.
* The §4.2 amplitude tables: I reproduced every row (3 713 m at 400 km, 606 m at 6.25 km, 377 m at
  3.13 km, sum exactly 12 000 m). The arithmetic is right; only the CONSEQUENCE is missing (B1).
* The C1-across-a-cube-edge symmetry argument in §5.4 rule 4 holds: `W'(1) = K1 + 3K2 + 5K3 = 1.5584` is
  the same from both faces and the along-edge parameter matches, so a C1 interpolation in the face
  parameter with a shared edge value is C1 in metres. Gating it with G-MACRO-EDGE anyway is the right
  call.
* The integer D8 cross-multiplication, the `(height, node index)` heap key, the `u64` discharge sum, the
  second-buffer talus, the integer 2 × 2 restriction, and the single-thread rule. Each removes a real way
  a generator stops being deterministic.
* The dyadic-exponent point (§4.6): any `m` of the form `k/2^j` is reachable by repeated `sqrt`, and
  `crates/terrain/src/gf.rs:102` grants it.
* The five bedrocks, their hardness, and the 23–92 m strata depth: confirmed at
  `crates/terrain/src/strata.rs:145-151` and `crates/terrain/src/body.rs:196-201`.
* `fluid_at` gives EVERY body an ocean today (`crates/terrain/src/chunk.rs:206-213`), and §8.1's first
  line is the right cure.
* The water clip `min(water_level − r, r − rock_surface)` (§8.3) is the correct intersection, and it
  reuses the extractor rather than writing a second one (HR3).
* Deleting the 10-node watercourse threshold (§6.1): the arithmetic is right — one link per node at
  8.224 km gives 0.122 km/km², already four to forty times below the literature range.
* Deleting revision 1's proposal to grow `crust_m` (§4.12): correct and important. `floor_m`
  (`crates/seed/src/ladder.rs:95`) and every radial index measure from it, so growing the crust
  re-addresses every cell on every body.
* `Z` is rung-independent and contributes zero to `dropped_bound_m` (§7.1). True — and it is the octaves,
  not `Z`, that break the bound (B3).
* §7.3 rule 1's insistence that the slope stencil be ADDRESS-defined rather than array-defined. That is
  the difference between a shape that is a function of the address and a cracked chunk boundary.
* §7.1 and §14 M10: the existing 400-sample lattice test
  (`crates/terrain/src/height.rs:82-105`) genuinely cannot see a channel, and a channel-following fixture
  is the right replacement.
