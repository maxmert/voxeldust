# 00 — CRITIC A: completeness and believability

**Date:** 2026-09-08. **Subject:** `docs/investigation/2026-09-08/landforms/00_proposed_landforms.md`
(the synthesis), read against its eight domain documents `01`–`08` of the same date, against the code,
and against the owner's own words.

**The owner's words this review holds the proposal to.** *"Make sure that we reach that quality on the
picture for earth-like planets (biomes can be different of course, should be dependent on the planet
position, spin, trajectory, size and gravity, etc.). It should be very believable, as we also should
simulate the weather."* And: *"no orienters and no details at all … The only thing I'm worried about is
that the surface will not be interesting enough."*

**The method.** I read the synthesis in full. I skimmed the eight domains and read every section they name
as a limit, an omission or a handover. I checked eighteen of the synthesis's own claims against the source
files it cites. I re-did five of its arithmetic steps. I asked four questions of every part: what would a
geologist say, what would a hydrologist say, what would a climatologist say, and what would a level artist
say.

**The verdict in one line.** The proposal's mechanism is sound, its self-criticism is unusually honest,
and **it will not produce the reference picture**, because five of the eleven things in that picture have
no slice and no owner, and because the arc never predicts the one number the whole investigation was
started to move.

**How to read the severities.**

- **blocker** — the arc can finish and still fail the owner's sentence.
- **defect** — a claim is wrong, or a decision the owner must take is missing from the register.
- **weakness** — the design is thinner than the picture needs, and the gap is not stated.
- **note** — an omission or an inconsistency the owner should see before he rules.

---

## THE FINDINGS

### F1 — blocker — The fourth measured cause of "no details at all" has no slice, no owner and no cost

**Detail.** §1 of the synthesis names four causes. Three are terrain. The fourth is the light. I confirmed
both halves of it in the code:

```
crates/client-render/src/terrain.rs:56    const FILL_SHARE: f32 = 0.08;
crates/client-render/src/terrain.rs:459   shadows_enabled: false,     // the sun
crates/client-render/src/terrain.rs:473   shadows_enabled: false,     // the fill light
crates/client-render/src/lib.rs:848       shadows_enabled: false,
crates/bins/tests/terrain_pictures.rs:43  const SUN_ELEVATION_DEG: f64 = 25.0;
```

The picture the owner judged was shot into the light, with no shadow, with a fill light on the shadow
side, on ground whose median slope is 1.36°. **The arc then leaves that cause where it found it.** §1.5
item 6 says *"No slice owns the sky's rendering, and no slice owns a shadow pass."* §6.5 repeats it for
the haze. Q6 repeats it a third time. §6.4 drops the two-light picture *"until a shadow pass exists"*.

So the arc can land the charter, the solve, the rivers, the biome and the third dimension — 19 to 24
weeks — and shoot the flagship 1.8 m vista into a flat light under a black sky, and the owner will say
the same sentence again.

**The internal contradiction the owner will trip on.** §6.2's slice-18 row promises *"the client's cloud,
rain and haze"*. §1.5, §6.5 and Q6 each say no slice owns the sky. One of the two statements is wrong,
and the synthesis does not say which.

**What a level artist says.** Relief is read by shadow. On a ridge lit from behind the camera, a 300 m
spur and a 3 m bump look the same. The reference picture's whole legibility comes from a low raking sun,
cast shadows, and aerial perspective separating four depth planes.

**The cheap half is hours, not a slice.** Move the star off the camera's own azimuth in the picture
harness, and switch `shadows_enabled` on for the terrain's directional light. That belongs in slice 8p,
before one metre of new landform is built, because otherwise every judged picture in the arc carries a
known defect in its instrument.

**Recommendation.** Give the sky and the shadow an owner and a slice, or get an explicit owner ruling that
every arc picture is judged unlit and hazeless. Put the harness's own two lighting fixes in 8p.

---

### F2 — blocker — No slice paints the ground, so the biome slice cannot produce its own pictures

**Detail.** MEASURED by `08` and confirmed by grep: the client draws the whole terrain with ONE material,
`base_color: Color::srgb(0.55, 0.50, 0.42)` (`crates/client-render/src/terrain.rs:362`), and the word
"biome" occurs nowhere in `crates/client-render/src` or `crates/client/src`. `08` §184 states the
consequence: verdict **V7, "the picture shows the biome the stamp names", CANNOT RUN.**

The synthesis's slice 8e delivers nineteen biomes, a soil law, a snow line, a tree line and sea ice —
all of them server-side column properties. Its named pictures are *"the snow-capped ridge; forest at 5 km
and at 50 km; a desert behind a range; the same latitude wet and dry."* **Every one of those four is a
COLOUR.** With one brown material the snow cap is brown, the desert is brown, and the wet and the dry
side of the range are the same brown.

I searched the whole synthesis for the words `paint`, `material` and `colour`. They appear only where the
picture probe replaces a colour classifier. **No row of §6.2, no row of §7 and no line of §10 gives the
client's paint table an owner, a slice or a week.**

The same hole swallows the water. `03` §8.3 designs a meshed, clipped, budgeted water sheet. The client
has no water material either — no fog, no sky dome, no water shader. So the river the pilot follows is
brown rock in the shape of a river.

**Recommendation.** Put the paint table — a biome-to-material map plus a water material — inside slice
8e's scope with its own cost, and make V7 a landing condition of that slice. Until it lands, 8e's four
named pictures cannot be judged and should not be promised.

---

### F3 — defect — The arc never PREDICTS the number the whole investigation exists to move

**Detail.** The investigation opens with a converged, named, four-station measurement: from stations
S2-a…S2-d, over 360 bearings, marched to 360 km, the largest rise over the local trend is **0.054°–0.067°**
and the skyline holds **zero breaks per 60° at 0.10° and above** (`01` §1.4a). That is excellent work. It
is an instrument that fails today and could pass tomorrow.

Then §1.3's own table says, for the same row:

| Skyline breaks per 60° | Today 0 | After **UNMEASURED** |

**The instrument was never pointed at the proposal.** The skyline march is a pure function of the octave
amplitudes and the macro height. The proposed slope spectrum (`03` §4.2.3) is a CLOSED-FORM change to
those amplitudes — it needs no solve, no artifact, no charter and no lane. The table of new amplitudes
is already printed (319 m at 6.25 km against 80 m today; 199 m at 3.13 km against 38 m). Re-running the
same march on those numbers is one afternoon of the same replay that produced the 0.054°.

**What the owner is being asked to fund without it.** Nineteen to twenty-four weeks, on the promise that
the surface becomes interesting, with no stated expectation for how interesting. And the arc's own kill
criteria cannot catch the failure: slice 8a could land, the march could still read 0.1 breaks per 60°,
and the arc would already be three slices deep.

**Q3 makes it worse, honestly.** The reference's *"8–12 breaks per 60°"* was counted by eye, so the gate
lands as a ratchet with **no floor at all**. So today the arc has neither a target nor a prediction — only
a rule that says the number may not get worse than zero.

**Recommendation.** Before slice 8a starts: (a) re-run `01` §1.4a's march at the four named stations with
the proposed spectrum, and publish the predicted break count and largest rise; (b) run the same march over
a synthetic macro field with the proposed macro relief, to predict what the solve adds; (c) run U16 — the
reference's own count under our rule — so the ratchet can become a floor. Item (a) alone converts the
whole arc from an argument into a prediction that can fail.

---

### F4 — defect — The free-coarsening bound is stated for one term of three, and the synthesis contradicts itself about `Z`

**Detail.** §4.1 claims *"A coarse rung's answer is a PREFIX of the fine rung's answer, and the coarse rung
states its own error exactly."* The prefix property itself is sound and provable, and `04` §5.1 names the
four ways to break it. That half is good.

The BOUND is not exact. `04` §5.2 writes it as three terms:

```
   |h(dir,0) − h(dir,L)|  ≤  Lip(T)·r(dir)·Σ A_i        the octaves      DERIVED
                          +  the terrace's own fade step  the terrace     asserted, not derived
                          +  |R_0(dir) − R_L(dir)|        the carve       asserted, not derived
```

Only the first term has arithmetic behind it (`bound/cell = 3.27 × s_m`). The other two are called
"statable". The synthesis concedes the second half of the problem in its own §4.4: *"The only pop left is
the dropped channel carve, and `M-L14` measures it."* **A term a measurement chases is not a term the rung
states exactly.**

**And `Z` contradicts itself across three sections.**

- §4.4: *"The macro field `Z` is rung-independent, so there is no level boundary and no crossfade to
  design for it."*
- §2.1 [E]: `s0 = MACRO(dir) + Z(dir)` *"read through a Catmull-Rom at a matched pyramid level"*.
- §9.1 `M-L14`: measure the pop of *"a pyramid-level change of `Z`"*.

If a pyramid level is chosen by drawing distance (`03` §9 rule 2), then `Z` has a level boundary, and
§4.4's sentence is false. The right statement is: `Z` is rung-independent in its DEFINITION and
level-dependent in its READING, and the reading's step is one of the three pop terms.

**Recommendation.** Rewrite §4.1 and §4.4 as: one derived term, two named terms owed a derivation or a
measurement, and a `Z` reading that has a level step. Grow the property test at
`crates/terrain/src/height.rs:80-105` a leg per term, not one leg for the octaves.

---

### F5 — defect — The synthesis's "what we do NOT deliver" list omits four landform families its own domain named

**Detail.** §1.5 states six omissions. `03` §8.4 states more, and the four the synthesis dropped are the
ones a player photographs:

| Omitted family | What the world already asserts | Source |
|---|---|---|
| **VOLCANIC** — no cone, no caldera, no lava plain, no neck, no volcanic plateau | the body draws **Basalt, Gabbro and Andesite** as bedrocks | `crates/terrain/src/strata.rs:136-142`, `03` §8.4 |
| **AEOLIAN** — no dune, no yardang, no ventifact | the world has `Biome::Desert` and a `Sand` topsoil, and the owner asked for weather by name | `crates/terrain/src/strata.rs:192`, `03` §8.4 |
| **KARST** — no sinkhole, no doline, no dry limestone valley | the world draws `Limestone` and already has caves | `crates/terrain/src/body.rs:194`, `03` §8.4 |
| **Mass wasting beyond talus** — no landslide scar, no rock-avalanche deposit | — | `03` §8.4 |

`03` says it plainly and the synthesis did not carry it: *"the world already asserts a volcanic history it
never shows"*, and volcanic is *"the second landform family after fluvial on an earth-like planet"*.

**What a geologist says.** He gets one look at a planet-wide vista and asks where the volcanoes are. On a
body with basalt bedrock and no volcano, the rock is a lie.

**What the owner will do.** He asked for *"biomes can be different of course"*. He will fly to one of the
nineteen biomes' deserts and find a sand-coloured grassland with no dune in it.

**Recommendation.** §1.5 grows from six items to ten. The register gains one row: is a volcanic family in
this arc, in a later arc, or never? A cone and a caldera are a closed-form radial term on the plate field —
they are far cheaper than the ice pass the arc already buys.

---

### F6 — defect — The climate never reaches the SHAPE, so an arid range and a humid range are the same geometry

**Detail.** `05` §7.3 states this as *"the deepest believability gap in this document"*:

> *"In this model a desert is a grassland with a sand-coloured topsoil byte."*

And it names the cure as a **binding cross-domain requirement** (`05` §12): the climate must feed the
erosion domain's rate laws.

I checked whether the synthesis carries it. It does not. The roughness field — the one term that decides
how rough the ground is per column — reads the macro slope alone:

```
   r_raw(site) = |∇Z| / SLOPE_REF                         03 §4.2.4
```

No aridity, no precipitation, no vegetation cover. `03`'s talus pass uses one angle of repose for the whole
body. So a range under 2 000 mm of rain a year and a range under 50 mm have identical slopes, identical
crests and identical hillslope form.

**What a geologist says.** He tells an arid landscape from a humid one in one photograph at any distance.
Arid: sharp scarps, mesas, alluvial fans, no soil mantle, angular crests. Humid: convex rounded hilltops,
soil-mantled slopes, no bare rock below the tree line. This is the most reliable read in landscape
photography, and the model has no term for it.

**In the game's words.** The pilot flies from the wet side of the range to the dry side. `05`'s three-scale
precipitation makes the vegetation change and the topsoil byte change. The ROCK under it does not change
shape at all, so the two sides are the same mountain with two paint jobs.

**Recommendation.** One register row: does the fine spectrum's roughness factor and the talus angle read
the climate grid's aridity? The climate grid is already read per column for the biome, so the added cost is
one lookup and one multiply — the cheapest believability in the whole arc, and it is not in the register.

---

### F7 — defect — The component the reference picture is mostly made of is missing from every cost line

**Detail.** The reference picture is forest. Q5 of the synthesis admits it: *"No document in this
investigation prices the skeleton — they price the biome it reads."*

So four numbers the owner will read as decisive are all rock-only:

| The number | What it excludes |
|---|---|
| a rung-0 surface chunk, 3.30 → **4.10–4.67 ms** of 8 ms | the vegetation skeleton, the anchor digest |
| the cave-dense chunk, 6.11 → **7.42–7.47 ms** of 8 ms | the same |
| the cave-dense chunk with a water sheet, **8.46–10.6 ms, OVER budget** | the same |
| the owner's 50 km vista, **1.0–1.1 GB of mesh** | every tree instance in the frame |

The worst chunk is already over the budget by the synthesis's own estimate, and the term that will grow it
further is unpriced. **D29 asks the owner to decide the budget question on numbers that leave out the
forest.**

The anchor digest is in the same class: §3.1 proposes a THIRD digest beside the cell and the mesh digests,
because *"a tree a player stands on is part of the shape"*, and nowhere prices it.

**Recommendation.** Price a vegetation skeleton on one chunk before D29 goes to the owner. State every
vista figure as "rock and water only". If the skeleton pushes the worst chunk further over, that is D29's
real input.

---

### F8 — defect — The arc's own acceptance pictures need slices the arc does not contain, and the estimate does not say so

**Detail.** §6.5 maps the reference picture to slices, honestly, row by row. Read the rows together and
count what the arc actually contains:

| In the reference picture | Owner |
|---|---|
| ridges, valleys, cliffs, a river, snow caps, a coast | **inside the arc** (8a–8f) |
| forest as a mass | **slice 14** (ruling S6-8: *"Not this slice (slice 14)"*) |
| a character for scale | **slice 16** |
| blue haze, sky | **nobody** |
| shadow | **nobody** (F1) |
| ground colour | **nobody** (F2) |
| fields, roads, walls | **nobody**; §1.5 item 1 and Q13 |

§10's estimate prices 8p, 8a, 8b, 8c, 8d, 8e, 8f and 18. It prices nothing between 9 and 17. So at the end
of the 19–24 weeks the flagship picture — the pilot's eye at 1.8 m — holds shaped rock, water and a white
cap that is painted brown, under a black sky, in flat light, with no tree and no figure.

**That frame will not answer the owner's question, and the owner is not told.**

**Recommendation.** Add one table to §10: what the flagship frame holds at the end of each slice. It is
five lines and it is the most useful thing in the document, because it tells the owner which slice first
produces a picture worth his judgement.

---

### F9 — weakness — Spin, obliquity and four more facts are DRAWN, and nothing makes the draw physical

**Detail.** D17 gives six missing facts to the forest as new seed draws: spin, obliquity, water inventory,
optical depth, effective elastic thickness and the glacial offsets. I confirmed the premise: a grep over
`crates/physics/src/worldgen/*.rs` finds no rotation period and no obliquity anywhere; the only "spin" in
that directory is the area-of-interest spin-up factor.

The owner's word was *"spin"*, and he meant a fact about the planet. A free draw makes it a lottery ticket.
The one physical constraint that exists is named in `05` §3.3 — **tidal locking**: a planet close enough to
its star turns once per orbit. `05` §9 even designs the tidally locked world (a burning day side, a frozen
night side, a habitable terminator ring). **The synthesis's D17 carries neither the constraint nor the
world.**

**What a climatologist says.** Give me a planet at 0.05 AU with a six-hour day and I will tell you it does
not exist. Give me 60° of obliquity on a circular orbit and I will ask which process produced it. A draw
with no conditioning produces bodies a reader refuses.

**In the game's words.** The pilot approaches the close-in planet of the home system. Its charter says the
day is nine hours. Its star should stand still in its sky forever.

**Recommendation.** D17 gains its conditioning: the locking test from the semi-major axis and the star's
mass sets the spin where it binds; the obliquity's draw is damped for close-in bodies. Pin the home
planet's drawn values in `home_body_pin.rs`, the way `POLE_AXIS` is pinned today.

---

### F10 — weakness — The three climate facts the arc rests on are a one-point fit, an unmeasured unit and three refused families

**Detail.** `05` is admirably honest, and the synthesis carries none of the three into its register.

1. **The Hadley fit misses Mars by a whole cell.** `05` §5.3: `dH = 0.388` is *"a REVERSE FIT to Earth
   alone"*; it gives Mars two cells where one is observed. Mars is the closest analogue to the moons of the
   home system. U8 is owed.
2. **The precipitation has no unit.** `05` §5.4: `supply` is in pascals, the classifier's axis is
   millimetres per year, and *"nothing converted one into the other"*. `K_mm_per_year` is unmeasured (U15).
   Without it *"the whole map's realism rests on a number nobody wrote down"*.
3. **Three climate families are refused by construction.** No cold coastal upwelling, so no Atacama and no
   Namib (`05` §5.4). No ice-albedo feedback, so every polar cap is a circle of latitude (`05` §5.6). And
   no ocean heat transport and no monsoon appear in any of the eight documents — I searched for both.

**What a climatologist says.** A one-anchor fit is a coincidence, not a law. A precipitation field with no
unit cannot be compared with the Whittaker chart it feeds. And a world with no ocean heat transport runs
its poles too cold and its equator too hot, which moves every biome boundary the picture shows.

**Recommendation.** Carry U8 and U15 into §9.1 with pass bounds. State the three refused families in §1.5,
beside the six omissions already there, so the owner learns them from the synthesis and not from a domain
document.

---

### F11 — weakness — Every cliff on a planet shows the same rock, and every bed is horizontal

**Detail.** Two facts multiply, and the synthesis states neither together.

- MEASURED (`crates/terrain/src/body.rs:194-201`, read): a body draws **one** sediment of three and **one**
  bedrock of five, one topsoil thickness, one subsoil thickness and one sediment thickness — for the whole
  planet.
- D34 refuses a rock map, so nothing varies that palette by place.
- D10 makes the strata a column at a fixed RADIUS, which is right and which cures the mesa. But the beds
  stay HORIZONTAL everywhere, including inside a fold belt.

**What a geologist says.** Rock type follows tectonic setting: a craton shows old crystalline basement, an
orogen shows folded and steeply dipping rock, a passive margin shows flat-lying carbonate, a rift shows
basalt. Horizontal bedding everywhere is a plateau landscape. **A range built of flat beds reads as
Monument Valley, never as the Alps** — and the reference picture is an alpine range.

**What a level artist says.** One sediment and one bedrock means one palette for the whole world. The
reference picture's identity is red rock against green forest against white snow. Ours will be one rock
against one forest against one snow, on every planet, forever.

**The seed ruling does not require this.** The 2026-08-27 ruling forbids a seed-derived map of VALUE. A map
of ROCK TYPE is a map of shape and colour, not of riches. D34's own option (b) — *"a rock map whose
deposits do not correlate with it"* — is the lawful form, and D34 rejects it with an argument about
prospecting that applies to the ore and not to the rock.

**Recommendation.** Split D34 into two rows: an ORE map (refuse, as now) and a ROCK map (decide). Add a bed
DIP term keyed to the orogen field — it is one dot product per column and it turns a plateau into a range.

---

### F12 — weakness — No measurement asks whether the new ground can be walked, landed on or built on

**Detail.** The arc's whole amplitude question was decided by a playability argument. `03` §4.2.2 deletes
the normalisation because refutation B computed a **53.4°** RMS surface slope: *"A player could not walk,
and the collider is the same shape, so he would slide."* The cure anchors the fine octaves at
`TALUS_RMS × tan(θ_loose) = 0.70`, which is **35° RMS at the roughest place on the body**.

So playability decided the design, and then no measurement was written for it. `M9`'s pass bound in §9.1
reads *"inside stated bands"* — **and the bands are not stated anywhere in the synthesis.** A band that is
not stated cannot fail, which is exactly the defect §6.3 and D40 catch in the picture instrument.

Three quantities the owner's game needs and nobody measures:

1. the fraction of the surface steeper than the character controller's own limit;
2. the flat area per square kilometre on which a pilot can set a hull down;
3. the fraction of the surface a player can build on without terracing — in a building game.

`03` §4.2.4 argues (2) and (3) qualitatively with the roughness field and the craton's 2.4°. An argument is
not a measurement, by this project's own standing rule.

**Recommendation.** Give `M9` three stated bands, and add one measurement: the walkable, landable and
buildable fractions at three roughness settings, over the same named chunk set the cost bench already uses.

---

### F13 — weakness — The almanac moves the snow and never moves the water

**Detail.** `05` §8.2 and §8.3 derive a moving snow line and lying snow from the almanac, and the argument
is good: a season may never invalidate a chunk, and 12 MB of per-cell snow diffs per square kilometre is
refused correctly.

The river gets nothing. Its water surface `z_w` comes from the static artifact (`03` §7.2) and never
changes. So on the home planet the snow line climbs the range in spring and the river below it stays the
same width, the same depth and the same colour all year.

**What a hydrologist says.** Discharge is the most seasonal thing in a landscape. A snow-fed river doubles
or triples in the melt. A dry-season channel shows its bars. The plan ships a season and freezes the water
inside it.

**The cure is cheap and it is already the same machinery.** The lying-snow rule derives a cover from the
almanac. A per-basin STAGE offset — one number, derived from the same almanac and the basin's own snow
fraction, applied to `z_w` and to the channel width — is the same shape of rule and costs one lookup.

**Recommendation.** One register row: does the almanac reach the water surface? If no, state in the
acceptance document that the rivers of this world do not have seasons.

---

### F14 — note — The weather the shard actually simulates never reaches a cost table or a kill criterion

**Detail.** The synthesis describes the live layer as *"a SHORT LIST OF WEATHER SYSTEMS, ~16 bytes each,
≤ 64 per body, ~1 KB on change"*, and it prices that list. **The list is the OUTPUT.** `05` §8.5 designs the
thing that produces it: an anomaly field of 10 bytes per climate-grid cell, stepped by semi-Lagrangian
advection with a relaxation and a source term.

```
   Earth-like candidate, C = 256:   393 216 cells x 10 B  =  3.93 MB per body of SHARD state
   the current home body, C = 128:   98 304 cells x 10 B  =  0.98 MB per body
   one weather step:                 about 4 ms          (ESTIMATED)
```

Neither number appears in §1.4's cost table, in §3.1's row (which reads UNMEASURED), or in §9.2's kill
criteria. A shard that holds several awake bodies multiplies it. Nothing states the ceiling, and nothing
states what a shard does when it is over.

**Recommendation.** Add a §1.4 row (bytes per body times awake bodies, and the step's share of a core) and
a kill criterion beside the others.

---

### F15 — note — No authoring path reaches the synthesis, and the reference picture is hand-authored

**Detail.** `04` §8 designs the authoring path properly, and it is one of the better pieces of the whole
set: **kind A**, an authored delta on a coarse pyramid rung, applied ON TOP of the fold, live state, never
in the identity (ruling S5-3); **kind B**, an authored landform held in the generator as a pure function of
the address, evaluated inside the fold so the octaves roughen it, a world-epoch change that enters the
identity. The separating rule is one sentence and it is right.

**The synthesis carries none of it.** I searched for `authored`, `landmark`, `override` and `edit pyramid`.
There is no register row, no slice, no cost and no measurement.

**Why it matters here.** The reference picture is a hand-authored AAA open world. Every ridge in it was
placed by a person. Our answer is that the seed and the water produce the ridges instead — which is the
right answer for a whole galaxy — but the owner will want to place ONE named canyon, and the mechanism that
lets him is designed and then dropped on the floor between two documents.

**Recommendation.** One register row for the two authored kinds, with `04` §8's law quoted, so the owner
can rule on whether the arc keeps the door open.

---

## THE FOUR PROFESSIONALS, IN ONE TABLE

| Who | The first objection | Where it lands |
|---|---|---|
| **A geologist** | Where are the volcanoes, on a planet whose bedrock is basalt? And why is every bed flat inside a fold belt? | F5, F11 |
| **A hydrologist** | Sediment leaves a basin and never arrives anywhere, the sea never fills, and a snow-fed river has no season. | F13, and `03` §4.11's own stated limit |
| **A climatologist** | A one-anchor Hadley fit, a precipitation field with no unit, no ocean heat transport, no monsoon, no ice-albedo feedback. | F10 |
| **A level artist** | The frame is flat-lit, one colour, treeless and skyless — and shape without shadow is not read at all. | F1, F2, F8 |

---

## DOES IT REACH THE OWNER'S TWO SENTENCES?

**"Biomes … dependent on the planet position, spin, trajectory, size and gravity."**
**Partly, and the missing half is the shape, not the label.** The chain is real: the charter's integers
(D14–D16), the insolation and the equilibrium temperature from the orbit (A2), the Hadley edge from the
spin (D19–D20), the relief ceiling from mass and radius (D6), the snow line from the lapse rate, and
verdict V13 tests it over the world's own bodies. That is a genuine answer. But **the biome changes the
paint and never the ground** (F6): an arid range and a humid range have the same slopes, the same crests
and the same hillslopes. And the ground has no paint at all today (F2). So the owner's sentence is answered
in the classifier and not yet in the picture.

**"We also should simulate the weather."**
**The simulation is designed; the drawing of it is not.** `05` §8.5 is a real weather model — anomalies,
advection, relaxation, forcing, a shipped list of systems, a checkpoint, a dormant-world rule. The
statement's shape (one row per realm, never one per observer) is the right shape and D25 defends it well.
But the client has no sky, no cloud, no fog and no rain (F1), and no verdict asks whether the weather LOOKS
like weather — V15, V16 and V17 ask only whether it is shipped, whether it moves, and whether the wind
agrees with the spin. **So the arc ends with a storm no picture can show.**

---

## CHECKED, AND SOUND

These I tested against the source, or re-did the arithmetic for, and each one holds.

1. **The three terrain causes reproduce in the code.** `SHORT_WAVE_M = 30` and the halving loop
   (`crates/terrain/src/body.rs:25,157-179`); the amplitudes re-normalised so their sum is the relief
   (`body.rs:180-184`); `octaves_at` drops octaves BY COUNT (`body.rs:251-259`); `dropped_bound_m` is the
   sum of the dropped amplitudes (`body.rs:266-277`); and the property test asserts that bound and nothing
   else (`crates/terrain/src/height.rs:80-105`). The diagnosis is not an argument.
2. **The relief law is exactly as described** — 0.4 % of the radius, clamped to 200…12 000 m, times a draw
   in `[0.5, 1.5)` (`body.rs:145-148`). D6's replacement (`draw × min(strength bound, shape bound)`) is a
   real improvement, and its Vesta-derived shape bound stops it from deleting every small round body.
3. **`POLE_AXIS = +Z` with the obliquity owed** (`height.rs:29-34`), and `biome_at` reads a latitude, a
   height and two slow noises into four biomes (`height.rs:37-64`). **D18 is right**: obliquity is a
   relation between two frames, the parent owns that relation, and carrying the cosine as a scalar keeps
   the `home_body_pin` intact. That is the cheapest correct answer available.
4. **D14's premise holds.** `crates/seed/clippy.toml` and `crates/terrain/clippy.toml` exist;
   **`crates/physics/clippy.toml` does not.** Two shards on two architectures can quantise a taxonomy float
   either side of a grid line. Authoring the charter once as integers is the right call, and the
   rescheduled-realm argument (a stored edit re-addressed) is the stronger half.
5. **The lighting evidence is real**, at every line cited (F1 quotes them). The synthesis found a defect in
   its own instrument, which is the hardest kind to find.
6. **The horizon arithmetic reproduces exactly.** From an eye at 1.8 m on `R = 3 350 759 m`, a point at
   50 km needs `(50 000 − sqrt(2Rh))² / 2R = 323 m` of prominence, and at 100 km it needs 1 390 m. **The
   flagship camera at 1.8 m is the right choice**, and `08`'s own correction from 373 m to 1.8 m is the
   single best decision in the picture domain: a flat world must FAIL the flagship rather than be flown
   over.
7. **The artifact's arithmetic reproduces.** 6 faces × 640² = 2 457 600 nodes; times 6 bytes = 14 745 600 B
   = 14.75 MB; and `(pi/2)·R/640 = 8 226 m` per node, which is the stated 8 224 m. §3.3's self-correction
   from 9.83 MB to 14.75 MB — against the author's own earlier figure — is the right way to carry a number.
8. **The lying-snow arithmetic reproduces**: 1 km² of 1 m cells times 12 bytes = 12 MB, so deriving the snow
   rather than storing it is not a preference, it is the only affordable answer.
9. **Four refusals are argued from a law and a number, not from taste**: D2 (no fine global bake — 3.6 hours
   and 131 GB), D25 (no per-observer weather window — the composed-per-observer shape the reach ruling
   killed by name), D33 (no placer ore — the seed ruling by name), and D1 (derive, because ruling S7-2 puts
   the client's first eight chunks BEFORE any realm speaks, so no lane exists at that moment).
10. **The picture instrument's self-criticism is the strongest part of the set.** A tolerance of
    `relief_bound_m` is 14 304.9 m — a 277 px band against a 69.4 px signal — so it **passed the very
    picture the work exists to refuse**. Putting 8p before 8a follows from that, and it is correct.
11. **The slice order is right.** 8a — the slope spectrum alone, with no ask, no lane, no artifact and no
    other domain — is the cheapest honest test of *"not interesting enough"*, and it is buildable today.
    F3's prediction should run first, but the ORDER is right.
12. **The eleven determinism rules are the right list for an iterative field on a bent grid**, and D26 is
    right: a forty-pass accumulation over 2.46 M nodes is exactly what an emulator is least likely to
    reproduce, so a real x86-64 machine must land before *"no drift"* is called satisfied.
13. **§5.4's refusal to fold the whole solve into the boot self-check is right**, and it names the measured
    defect it would repeat (5.3 GB at boot). The gateway owns no planet and must draw none.
14. **D27 is the right question to ask first.** The voxel home planet's look radius is 3 350 759 m
    (`crates/terrain/src/home.rs:22`) and the census's earth-like body is 6 515.5 km. If they are two
    different bodies, every picture the owner has judged was taken on the wrong world, and it is one bench
    to find out.

---

## WHAT I WOULD DO IN THE FIRST WEEK

1. **Run F3's prediction.** One afternoon, no new code path: the four-station march on the proposed
   amplitudes. It either says the arc works, or it saves nineteen weeks.
2. **Run U1** — is the home planet the right body? One bench.
3. **Switch the shadow on and move the sun off the camera's azimuth** in the picture harness, and re-shoot
   the three pictures the owner has already judged. That costs hours, and it separates *"the terrain is
   flat"* from *"the light hides everything"*.

Only then is the register worth answering.
