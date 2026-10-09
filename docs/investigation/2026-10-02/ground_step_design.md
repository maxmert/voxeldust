# The ground step (ruling B35), phase 0: the design — FINAL

> **What this is.** Phase 0 of ruling B35 (2026-10-02): ONE design for the whole ground step. It holds the laws with
> their calibration bodies, every cap that goes (B23), the frame cost of each phase, the bodies, the checks, the quality
> bar and the catalogue of every kind of country (B35 items 2–4).
>
> **How it was made.** Four design angles (the shape; the processes and the rock; the climate, the biomes and the
> cover; the light, the frame and the weather) → one synthesis (the draft) → three adversarial reviews (the laws, the
> quality, the speed: 70 findings, 5 of them blockers) → this final pass. The final pass read the code again wherever a
> finding disputed a claim about the code, answered every finding (the Appendix lists each one with its answer), and
> changed the design wherever a finding held.
>
> **Date.** 2026-10-02.
>
> **The tree.** Branch `game` at `ae39b32` with its uncommitted changes: `GENERATOR_VERSION` 22
> (`crates/terrain/src/tag.rs:250`), `ARTIFACT_VERSION` 9 (`crates/terrain/src/artifact.rs:95`), `PROTO_MINOR` 35 with
> its floor at 34 (`crates/wire/src/version.rs:357`, `:385`). The work was read-only: this file is the only file the work
> wrote, and no build, test or instrument ran.
>
> **The marks** (§11 defines them):
> - `path:line` — read in today's code;
> - MEASURED — a tool printed it in an earlier run; the record is named;
> - DOC — from a ruling, a design document or a code comment, not measured again;
> - ARITH — this panel's arithmetic on marked inputs;
> - ESTIMATE — a judgement, or a published value that nobody read at first hand in this work (the source is named);
> - UNMEASURED — nobody printed it; §9.2 names the bench that measures it.
>
> **The labels** (one label for one meaning). F1–F70 name the review findings (the Appendix). K1–K18 name the families
> of the catalogue (§3.2). "Ruling F1" to "ruling F9" name the owner's rulings of 2026-09-12. V-foot, V-ref, V-high and
> V-orbit name the four framings (§2.2). FR1–FR8 name the frame rules (§7.7). R1 is the erosion bench, R2 the eroded
> 1 km level (A0), R3 the 1–5 km valleys. P1–P16 name the contract (§3.1), Q1–Q18 the owner questions (§10), A1–A14
> the SL6 asks (§4.4), C1–C28 the conflicts (§1.2), M0-1 to MX-1 the benches (§9.2), G-… and B-… the checks (§2.1).
>
> **The reference pictures** stay outside the public repository. This design reads them as a bar of quality, never as a
> list of things (B35 item 1). The living status file §9 describes all three frames in words.

---

## 0. FOR THE OWNER

### 0.1 What we build

- The three reference pictures set the quality. They are not a list of things to copy.
- Seven qualities hold everywhere: on foot, in low flight, high up and from orbit, on every body.
  1. The ground has structure at every scale, from ranges of 100 km down to the stones under the feet. No scale is
     plain noise.
  2. The land makes sense. Rain that falls on any point flows along valleys that join, to a sea or to a closed basin.
  3. Each form shows the process that made it.
  4. The surface follows the shape: rock on steep crests, soil in hollows, a tree line, a snow line.
  5. One view holds several kinds of country, with natural borders between them.
  6. The light and the air are right: shadows at every distance, light in crevices and under trees, rich materials,
     haze, clouds.
  7. The detail goes to the horizon. Nothing flickers. The game draws 60 frames a second at full screen.
- The laws make every form from the seed and from the body's own facts. Nobody places a form by hand. The same laws
  make every planet and every moon.

**Example.** A pilot flies a hull from the belt's foothills up to orbit. On foot, she walks over a round hilltop and
down into a gully. At 400 m, she sees side valleys that join a river valley, with forest in the hollows. At 150 km, she
sees ranges and valleys, not ripples. From orbit, she sees snow only where the climate puts it. No part of the ground
jumps at any height.

### 0.2 Two rules under the far view

- The game draws far ground with fewer and larger cells. Each cell size is one detail step.
- **Rule 1, the average.** Each coarser detail step shows the average of the finer step. The rule applies to the
  height of the ground, to the share of each material and to the light on the ground.
- **Rule 2, the outline.** A form that is taller than it is wide (a rock pillar, a tower, a thin fin of rock) keeps its
  outline. Both machines make it by the same law. Near the eye, the cells draw it. Farther away, coarser meshes of the
  same form draw it. When it is less than one screen pixel tall, it becomes a tint.
- Why two rules: on ground that is less steep than 45°, a thing that is narrower than one cell is also lower than one
  cell, so the average loses nothing that the eye can see. A pillar is different: it stays several pixels tall after it
  is narrower than one cell.
- Numbers check both rules in every phase. A number goes red when a rule breaks.
- Today one part of the ground breaks rule 1. The sharp small hills stand on average up to 572 m over the ground that
  the server solved (measured). Over the belt the lift is 287 m (measured). Phase 2 replaces those hills. From then on,
  every small form must have a zero average, and a number checks it.

**Example.** A red rock pillar, 30 m wide and 100 m tall, stands in a canyon. A pilot climbs away from it. With rule 1
alone, the far detail step at 28 km would show the pillar as a bump 22 m tall, while her eye still sees it 3 pixels
tall: a change of shape (arithmetic). With rule 2, the pillar keeps its outline until it is less than one pixel tall,
and then it is a faint red tint. At no distance does the pillar appear, go away or change its shape in one frame.

### 0.3 Each realm calculates and draws itself

- The planet's own server calculates the planet's products:
  - the large-scale solved ground, once for each planet;
  - the finer eroded ground at 1 km (we call it the bake), once for each region;
  - the far map and the climate;
  - later, the weather and its memory (lying snow, wet ground).
- One generator makes the fixed shape under 1 km and the ground's materials. The server and each client run the same
  generator and get the same bytes. The server will use the shape for collision. The client uses it for the picture.
- The client draws all these products as the planet's own look.
- A parent only positions its children. The parent never calculates them and never draws them. A moon calculates and
  draws itself.
- The bake follows the planet's own visitors. The planet first bakes the ground under the hulls and the people that it
  holds, along the paths that it gives them. The planet's own server bakes the home planet before play, with the whole
  machine.

**Example.** A hull flies low over the home moon. The moon's server solved the moon's large craters and baked its 1 km
ground. The moon's server and the pilot's client both make the small craters under the hull with the same generator,
and they get the same bytes. The planet's server never calculates the moon. The moon's parent only positions the moon.

### 0.4 The order of work, and what you see after each phase

1. **Phase 0: this design and its measurements.** Nothing in the world changes.
   - We measure today's world on every stand, and we pin the stands.
   - We measure real ground on Earth, the Moon and Mars from public elevation data.
   - We list the facts of every body in the home system. A rule then picks the third body.
   - We measure today's frame at full screen cleanly, the cost of a forest with stand-in trees, and the memory.
   - You see tables of numbers, the list of bodies and the pinned stands.
2. **Phase 1: the flat directions.** Water on flat ground takes its direction from a published method, not from the
   grid. The rule "every small form has a zero average" becomes a check. You see the lakes and the flat valley floors
   again. The pictures change a little.
3. **Phase 2: the climate laws and the 1 km ground.**
   - First come the large-scale changes, one at a time: the water that runs off cuts the valleys; the flow divides in
     all directions; the seasons and the rain belts follow the planet's own tilt.
   - Then the planet's server erodes its ground again at 1 km.
   - First you see a shaded relief picture from the bench. Then you look in the window.
   - From 150 km you see valleys about 3 to 8 km apart that join toward the sea (a prediction). The ripples 3.2 km
     apart go away (measured today).
4. **Before phase 3: speed.** The builder speed-ups, today's frame cures, and the design of the geometry step with its
   price. You decide where the geometry step goes.
5. **Phase 3: the valleys of 1 to 5 km, and the slopes.** Side valleys, gullies, round hilltops, scree aprons, small
   craters.
6. **Phase 4: the processes.** Hard and soft beds, cliffs, mesas and pillars; ice; wind with real dunes; volcanoes;
   rock that dissolves; faults; coasts; the sea floor; frost; overhangs and arches. We build one family at a time, each
   with its own numbers and pictures.
7. **Phase 5: the climate zones and the ground's materials.** Soil, rock, scree, sand, river gravel, snow and ice, from
   the laws. The same generator computes them on both machines.
8. **Phase 6: the cover.** Trees, bushes, grass, boulders and stones from art assets.
9. **The light and the surface's look.** Shadows at every distance, light in crevices and under trees, textures, the
   eye's adjustment to bright and dark.
10. **Phase 7: your look** at all stands on the three bodies, and in the window.
11. **Then:** the waves; the water fill; one weather step; the edit system.

**Example.** After phase 2, the pilot hovers 150 km over the belt and sees long valleys that join and run to the sea.
After phase 4, she flies to a dry plateau and sees mesas with hard caps, and red pillars that keep their outline while
she climbs. After phase 6, she sees a forest edge in a valley that stops under the tree line.

### 0.5 What it costs

- **The server.**
  - The old bake method costs 14 to 25 processor-core hours for the home planet (arithmetic). The new method has no
    caps and a different solver. Nobody has measured its speed. The bench measures it before you rule.
  - Each later change of the large-scale ground bakes it again.
  - On a planet that nobody visited, the pilot first sees the 8 km ground. The 1 km ground then arrives in front of
    her: about 8 to 15 minutes for one view on an average machine, by the old method (arithmetic of the R2 review).
    This is a change of shape in front of the pilot. We ask you to accept it as a known exception, and we measure the
    fastest approach that meets no unbaked ground.
- **The download** (at today's pace of about 2 MB a second):
  - about 10 MB on a 150 km approach, about 5 seconds, against 0.4 MB today (arithmetic);
  - the exact far hand-over: 2 to 4 MB more at 150 km, and up to about 7 MB high up (arithmetic);
  - the far shares of the materials: about 49 to 65 MB before compression at the worst height, 23 to 31 seconds
    (arithmetic). We measure the compressed size first.
- **The frame.**
  - Today's only full-screen reading is 23.1 ms (measured, but not a clean reading). 60 frames a second needs 16.7 ms.
  - On that reading, the planned new parts give 29 to 35 ms a frame (arithmetic).
  - The full game fits only if the speed work cuts today's content to about 7.4 ms (estimate). Nobody has priced that
    cut yet. The clouds' 2.0 ms and the weather's 0.8 ms stay inside the plan.
  - Below half size, the processor already limits the frame (measured). So the ground step may add very little work
    for the processor.
- **The ground's builders.** At 528 m/s low over the belt, about 230 pieces of ground a second arrive late today
  (measured). Each new kind of detail makes each piece slower to build: 1.4 to 6.5 ms more (estimate). The builder
  speed-ups therefore come before phase 3.
- **The work.** All families together: about 66 to 105 engineer-weeks, plus 8 to 15 for the speed work (estimate).
  This is a relative size for the order, not a calendar.
- **The art and the memory.** Textures, tree kits and boulders: about 0.6 to 1.2 GB (estimate). The material shares on
  the ground add 0.2 to 0.45 GB (arithmetic). Bought assets need a private store, because the repository is public.

**Example.** A pilot flies at 528 m/s, 50 m over a forest. Each new piece of ground must be built before she sees it,
and each tree adds work to each frame. If the forest costs more than its line, the forest phase stops and you choose
one of these: fewer near trees, a nearer distance where the trees become a canopy surface, or a smaller render size
after your look at it.

### 0.6 How we check the work

- A rule picks the stands. We run the rule once in phase 0 and pin every stand. Later phases add stands and never move
  one.
  - one stand for each mix of climate zone, kind of relief and kind of rock;
  - three stands for each family of processes;
  - stands at the tree line, at the snow line, on the two sides of a rain shadow, and on borders between kinds of
    country;
  - random stands, on land and on the sea floor;
  - today's pinned stands.
- Three bodies: the home planet, the home moon, and a third planet that a second rule picks. Every phase shows
  pictures of all three.
- Four views on each stand: on foot, the reference view from 400 m, 150 km up, and from orbit. Each view has two suns:
  a low sun (as in the first reference) and a high sun (as in the third reference).
- Each quality has numbers that can go red. A red number is reported. It does not stop a step by itself (your ruling:
  no caps). Your look judges.

**Example.** The rule finds a dry, high region of layered rock on the home planet and pins a stand at its middle. The
bench prints there the cliff heights against the law's limit and the share of bare rock on each slope. You then see the
reference view from that stand under a 15° sun and under a 50° sun.

### 0.7 What the panel found in the code

1. No drawn pixel reads the climate, the rock or the climate zone today. The whole ground has one colour.
2. The climate zone that the generator computes today changes with the detail step. The single colour hides this
   fault. Phase 5 cures it before colours come.
3. The kind of rock changes at the edges of 8 km squares. In nature the border follows the valleys, where the ground
   cuts the layers. Phase 4 cures this.
4. The client draws a closed hollow under rock as rock. An overhang would therefore go from the picture, while the
   server's own cells hold the opening (the collider is not built yet). Phase 4 cures this together with the overhangs.
5. A shadow near 3.5 km can switch on or off when the pilot only turns her head. The code shows this, but nobody has
   seen it yet. A bench checks it first.
6. The wind cells and the rain belts read a number fitted to Earth's tilt. The home planet's tilt is 55°. So the year's
   warmth is almost equal at the equator and at the poles, and the law moves the rain belt with the summer sun. Today's
   code gives one fixed cell, deserts near the poles and an east wind everywhere. We withdraw those predictions until
   the corrected climate runs.
7. The home planet is dry: 374 mm of rain a year over the land (measured), against Earth's 750 mm.
8. The moon has sedimentary beds today, but no water ever ran on it. This breaks your rule that no physical fact is
   drawn.
9. The ice in the solve cuts rock, but the cut rock goes nowhere. No moraine forms.
10. Today's rule for bare rock is two slope lines. On wet ground every slope over 25° would be bare rock, but forested
    slopes on Earth hold soil to about 50°.
11. The detail steps are sized for a picture 720 pixels tall. At your full screen, one step is about 3 screen pixels.
12. The graphics card builds no piece of solved ground today, and it is off by default.

**Example.** Today a pilot who lands on the moon walks on shale and sandstone in beds. After phase 4, the moon's crust
law gives it no beds. She walks on rock that impacts crushed, and its cliffs stand only as tall as that weaker rock
allows.

### 0.8 The questions for you

Each question has a recommendation. §10 gives the details.

1. **The two far-view rules and the small hills' average.** The average for height, colour and light; the outline for
   tall thin forms; "every small form has a zero average" as a check from now on. We do not build the centring of
   today's sharp small hills, because phase 2 replaces those hills. **Recommendation:** yes.
2. **Real ground for the calibration.** Public elevation data of Earth, the Moon and Mars. The data stays outside the
   repository, we record only numbers, and we check each licence first. No request carries your name or your address.
   A source that needs an account waits for your word. **Recommendation:** yes, in phase 0.
3. **The home planet's wind and rain follow the summer sun.** With a 55° tilt, the law moves the rain belt with the
   seasons. Your ruling of 16 September excludes "the monsoon", and this circulation is of that kind.
   **Recommendation:** allow it; we run the climate census again before any stand rests on it.
4. **Phase 2's large-scale change.** The water that runs off cuts the valleys; the flow divides in all directions; the
   seasons and the rain belts come from their laws with no caps; we measure each change alone before the first bake.
   **Recommendation:** yes.
5. **The 1 km ground under your newer rulings.** No caps; small forms never make closed pits; the rock of each bed sets
   how fast it wears; the bench and a shaded relief picture first; its speed measured first. **Recommendation:** yes.
6. **The far hand-over at 890 km.** Exact (2 to 7 MB more data) or a small smooth change. **Recommendation:** exact.
7. **The first visit, and who orders the bake.** The planet bakes the ground under its own visitors first. A viewer
   outside the planet asks for ground by tile number, never by position. Where the bake is too slow, the 8 km ground
   stands and refines later: a known exception. **Recommendation:** yes.
8. **The prebake.** The home planet's own server bakes it, with the whole machine, before play. No separate tool.
   **Recommendation:** yes.
9. **The size of a detail step at full screen.** Keep today's ruled size (a picture 720 pixels tall) and let the
   eye's rule judge at full screen, or make the steps 3 times finer (5 to 10 times the pieces of ground).
   **Recommendation:** keep it; we measure the eye's rule at full screen in phase 0.
10. **The frame and the builders.** A target split that fits 16.7 ms; a running total at the end of each phase; the
    speed work before phase 3; the forest's cost measured in phase 0; the card stays off in this plan; please name the
    average machine. **Recommendation:** yes.
11. **The valleys of 1 to 5 km.** We compare three methods on the bench against Earth, and then we choose.
    **Recommendation:** yes.
12. **The families at their prices:** dunes, volcanoes and rock that dissolves (held since 16 September), and the new
    families of the sea floor and of frost. **Recommendation:** yes, one at a time in phase 4.
13. **The ground's materials from one source.** The generator computes them on both machines; the screen only blends
    them; the far shares go with the planet's data. **Recommendation:** yes, with the sizes measured first.
14. **The forest and the art.** The generator makes each tree's collision shape, and the art fits that shape; no
    generated stand-in shapes; free stand-in trees for the bench of phase 0; a private store for bought assets; your
    budget. **Recommendation:** yes.
15. **The shadows.** Shadows at every distance from the same ground as the picture; a fade at the end of the near
    shadows; a bench proves the turning fault first. **Recommendation:** yes, in the light phase.
16. **The stands.** The rule of 0.6, pinned once; the third body by its rule; two suns; the ice law tested on Earth's
    real data. **Recommendation:** yes.
17. **The weather's connections.** The weather list on its own lane, as you ruled on 16 September; a sleeping planet
    starts its weather again from its climate; the planet knows its sun from its orbit, which its parent states once.
    **Recommendation:** yes; we build them in the weather step.
18. **The scope of your final look.** The final look judges the ground and the light without clouds and with the dry
    sea floor. A second look after the weather step judges the clouds and the water. The status file stays until then.
    **Recommendation:** yes.

**Example.** Suppose you answer "exact" to question 6. A pilot who climbs from 800 km to 1 000 km over the belt then
sees the valleys fold into the far ground, with no change of shape at the 890 km step. If you answer "smooth", the
ground at that step can move by about a tenth of a far cell. The old bench measured 226 m on a cell of 2 km. We measure
it again first.

---

## 1. The binding laws, and the conflicts that the code and the newest ruling resolve

### 1.1 What each law requires of this design

| Law | What it requires here |
|---|---|
| HR1 sealed shards | The bake, the weather stepper and the weather memory run inside the body's own shard, and so does the prebake (no separate tool; Appendix F18). Their products cross only as appended bulk arms through the gateway, which relays them without decoding them. |
| HR3 one tooling | One generator (`vd-recipe` + `vd-terrain`). One bake function, run only by the body's own shard. One set of ground pipelines, warmed before play: nothing compiles in flight (F65). A new shard feature is a `ShardProfile` capability (`crates/sim/src/capability.rs:116`), never a `match` on a shard kind. |
| HR4 features once | Every term is a function of (seed, address, charter, the realm's shipped rows or A0). No switch on a body kind (§3.1 P7). "Up" comes from the realm's gravity function, never from the sphere's radial alone (V4 15.2; P12). Each new shard feature (the bake service, the weather stepper, the weather memory) passes a G-IDENTICAL fixture on two shard kinds — the spherical and the Cartesian voxel geometry with the same capability (`capability.rs:36-42`). A piece that reads the cube-sphere lattice itself sits below the seam and carries G-MAPPING coverage (F10). |
| HR5 100 % | `vd-recipe` and `vd-terrain` stay Tier-A. Generic functions are branchless shims. Each change leaves the files it touches at 100 % region and branch (B29 item 3). |
| HR6 + owner 2026-10-01 | Every stand and framing is a vdctl run. New probe kinds are owed (cover, cloud, weather); today the probe knows none, ruler and terrain (`crates/bins/tests/terrain_pictures.rs:61`). A bench's control lives in the harness (synthetic frames in Tier-A tests), never as a switch, hook or branch in the game (F15). |
| SL1 | The client takes the sun from the composed picture. The planet's own shard has no lawful source for its own sun today; the weather step asks for one (A12, Q17), and no placement, containment or crossing code may read it (clause 5). |
| SL2 | No occupant pose orders the planet's work. The bake follows the planet's own occupants and occupied children at the placements it authors (SL7); a viewer outside the planet asks by tile number (A11, Q7; F4). |
| SL3 + B35 item 6 | The realm that a product describes calculates it; the client draws it as that realm's look; a parent positions its children and never calculates or draws them (§4.3). |
| SL5 one world | The stands stand on the world's own bodies. The Earth, Moon and Mars data are instruments, never a world. No Earth-like test body: G-ICE reads Earth's real data (B23 item 3 is newer than B5; F45). |
| SL6 | Every new datum and every new arm is an ask with its size and its arrival time at the gateway's pace (§4.4). The default is NO. |
| SL7 | The bake serves the live realm: its own occupants first, then the tiles that viewers name. A planet that nobody sees bakes nothing. |
| SL9 | No per-child row on a per-tick or per-weather-step lane: the planet's sun comes from orbit elements stated once (A12), never from a direction stated again at every step (F5). |
| SL10 (V1, V13 L12) | Every static-shape term, the ground's materials (F1) and each tree's collider skeleton (F7) run in fenced integers on both hosts under the no-drift gates (terrain-pin, gpu-drift, artifact-pin, G-AMP-PIN on both targets, the anchor digest, the skeleton pin). Each new law states its deterministic form: an integer table, the crate's own function, or the recipe's exact reciprocal (F22). |
| Conventions | No inline operational literal: the gateway's pace `ARTIFACT_BYTES_PER_BEAT` (`crates/connection-plane/src/gateway/artifact.rs:50`) moves into the transport tuning. Integer kernels. No client prediction: the client draws the weather one statement back. |
| T3 time | The static shape never reads the tick. The wave-cut platform reads the tidal range from the charter (A14), not from the almanac. Snow has three layers: perennial ice and firn in the static shape; the almanac's seasonal snow line as paint with no collision; lying snow as the weather memory's live state (F17). The weather list rides its own lane (F6). A dormant planet does not step its weather (F14). |
| T5 | The aeolian, volcanic and karst families wait for their prices (§3.3). The seasonal circulation that the home planet's 55° tilt calls for is the kind of circulation that T5 names "the monsoon": asked (Q3; F29). |
| T7 | Each hand-over step stays under one cell of the rung that takes over. A compact tall form is judged by its own height step in pixels (rule S, §3.1; F28). |
| T9 | Every physical number comes from a law with a calibration body: K₀'s water term, the moon's crust cohesion, the meander law, the circulation's contrast (§3.2). |
| B23 no caps | §6 lists every clamp with its fate. A gate's band is a reading. A limit that IS a law stays, named. |
| Rulings F1/F2 | A frozen patch's identity names the A0 bake it stands on, or A0 freezes at slice 14: asked at the freeze (finding F9). |
| Rulings F6, B20 | The client's terrain workers take half the cores (`crates/client-render/src/terrain.rs:4960-4963`); the server's bake takes one budget for each host. Every budget here is the instrument's (the owner's Mac); the average machine's numbers are owed by the owner (Q10). |
| Rulings F7/F8 | Fenced integers from one source for the shard's CPU, the client's CPU and the card. The card builds no solved chunk today (`terrain.rs:1582-1588`) and is off by default (`terrain.rs:439`): asked (Q10; F54). |
| Ruling F9 | The bounded ask. The coarser rung stands while a finer one waits. A hole and a pop stay defects. |
| V1, V4 | Trees, grass and decoration are art assets; no procedural primitive mesh is ever drawn for a tree (V4 step 9 (a)). The collider comes from the generator's skeleton and the picture from the kit (V4 step 9 (c)–(d); F7). |
| V5 | Nothing underground exists until a surface crossing lies in a band; a cave seen through its mouth draws with no pop. |
| V10 | No impostors: the full asset, then the asset's own coarser meshes, then the canopy fold — three stages, no generated fourth (F8). Surface nets now; dual contouring decided on pictures, benched at rung 0 and at rungs 5–9 (F28). The 8 ms chunk budget. |
| V13 L24, V14 | The paint table comes first; textures replace it on the same seam. The seam carries the stratum per vertex and the biome per column, for ALL ground (F1, F2). D8-1's pixel (45° over 720 rows) stays unless the owner changes it (Q9; F32). |
| S5-6 | The seed decides bulk stock only. A province is static composition, so every new substance passes S5.4's wiki test, and anything valuable is live state (F16). |
| W16, W17, B16, B18 | A finer rung refines a coarser one and never contradicts it. A rung boundary is invisible in shape, colour and brightness. A hand-over's step includes the field's own fold. A sub-pixel feature is a tint. The arrival ramp is 0.5 s. |
| B6, B8 item 3 | Each step ends with the stands' numbers and pictures, and the living status file changes in the same change. R2's bench and hillshade come before any bake service. Ground is never temporary. |
| B9, B12 | The flat directions come before R2; the basins come after R2. |
| B21, B22 | One volume. No water word in the solid pass, so the Ocean, SeaIce and Beach classes leave the solid pass (F17). The ground, then the fill, then the edits. 60 frames a second at full screen. |
| B23 item 3, B35 item 3 | Every phase's pictures include the home moon and the third body, not only the home planet (F13). |
| B27, B31 | The eye's rule judges every flight leg on the display picture. G-CROSSING allows half a level. |
| B34 item 7 | The waves wait for the proper terrain. The builder levers that §5.3 puts before phase 3 are B22 step 2's owed speed items, not the waves' cure (F54). |
| Reach rulings | A realm states its reach from its own look, size and brightness both. A parent never draws a child. A finer drawable floor would move every realm's reach (Q9). |
| Owner 2026-09-25 | No outside request carries the user's identity (Q2; F26). |

### 1.2 The conflicts, and how the code and the newest ruling resolve them

Rows C1–C20 were found by the synthesis; rows C21–C28 were raised by the reviews. A row marked CHANGED differs from the
draft.

| # | Between | The conflict | Decided by | The result |
|---|---|---|---|---|
| C1 | cover vs light | Which climate rung reads the biome | T1 item 7; W16; B16; F1 | The recipe computes the class from fields that every rung reads alike (A0's height, the rows' climate bytes). The lines (snow, trees, bare rock) are expectations under the dropped spread, also computed by the recipe per column. No drawn rung decides a class. |
| C2 | cover vs light | The frame sum, 20.7 or 21.7 ms | F53 (CHANGED) | §7 starts from the only full-screen reading and from a target split whose lines sum to 16.7 ms. |
| C3 | processes vs B35 order | Families before R2's first bake, or in phase 4 | B35 item 4; B12; F33 (CHANGED) | The process families stay in phase 4 with ONE grouped re-bake. The climate laws are not a process family: they move into phase 2's macro change and are measured alone before R2's first bake. R2's first bake reads each bed's own rock through today's province map. |
| C4 | r2 §8.4 vs B8 item 3 | A stand-in that morphs into A0 | B8 item 3; T3; r2 skeptic 1; F3, F56 (CHANGED) | An unbaked part WAITS and rung 13 stands; no stand-in morph. The bake is the realm's warm-up. Where the bake cannot win the race, the refinement in front of the pilot is a registered transient, asked (Q7). |
| C5 | r2 §8.2 vs B19; ground vs weather | Which arm at which minor | B19 took minor 35 (`version.rs:357`); `BulkMsg` uses discriminants 0–6 (`crates/wire/src/channels.rs:395-480`); T3 (CHANGED) | The A0 arm takes discriminant 7 at the next free minor when it lands. The weather list gets its own appended arms at their own minor (F6). |
| C6 | shape vs cover vs one-volume | The far fold under R2 | B22 Q1 (b), whose far half is one rule for ground and water (`docs/investigation/2026-09-27/one_volume_design.md` §4.3); B16; F25 (CHANGED citation) | `L_ref` = rung 10, A0's own grid. Rungs 11–12 read A0's own 2 km and 4 km means (A2). Rung 13 reads the rows, which are A0's node means by the mean hold. Rungs 14–18 read the far map. Under R2 the choice of `L_ref` and its data are new, so Q6 asks. |
| C7 | all four angles | Four stand rules | B35 item 3; F42 | One layered rule, run once in phase 0 and pinned (§2.2). |
| C8 | shape, processes, light | The third body | B35 item 3; B23 item 3 | A charter-distance rule, after bench M0-5 prints the charters. The home moon is the second body. |
| C9 | shape vs processes | R3 as a closed form or as erosion | SL10; ruling F7; V10's 8 ms; F34 (CHANGED) | A three-way bench against Earth decides (Q11). No recommendation before it. |
| C10 | shape vs processes | Repose as the only slope limit | B23; F30; F50 (CHANGED) | Repose for debris only (scree under a cliff). The soil-mantled limit S_c for soil slopes. Culmann's critical height with each rock's own cohesion for intact rock. |
| C11 | processes vs code | The province byte's bits | ARITH; `crates/recipe/src/cell.rs:50-57` | Phase 4 decides on a measurement: (a) ≤ 8 kinds a body + 5 bits; (b) 16 rows + strike 3 bits + 1 fold bit; (c) a second byte. Default (b). |
| C12 | cover, light, V14 | Materials per pixel against V14's seam | V14; ruling F7 item 3; F1, F2 (REVERSED) | V14's seam stands for all ground. The recipe computes each column's materials in integers on both hosts; the vertex carries them; the shader only blends them. No shader derives a threshold. |
| C13 | cover vs V4 | Asset-baked skeletons against "the skeleton selects the art" | V1 item 1; V4 step 9 (c)–(d); F7 (CHANGED) | The generator's skeleton decides the collider; the art fits the skeleton (Q14). |
| C14 | processes vs cover | The province warp | F35 (REPLACED) | The outcrop rule: a rock border is where the ground cuts the cover's base surface. The seeded warp goes. |
| C15 | shape vs r2 §7.4 | The per-octave lattice offset | F48 (CHANGED) | It rides phase 2's fine relief, and only if bench M0-2 measures a lean. |
| C16 | processes vs T7 | Survival constants | T7 rule 1; B22 Q2 | Each term's constant is measured for its own profile. |
| C17 | light vs shape | The shape's frame line | No measurement; F61 | 0.5–2.8 ms (ESTIMATE). Rule FR5 reads each shape phase's vertex line on today's pass and on T3's one pass, and carries the larger. |
| C18 | r2 D11 vs B23 | The ladder clip on the fine relief | B23 item 1; T7 rules 2–3 | Refused. |
| C19 | 05 §8.8 vs light | Who receives the weather list | SL3; reach rulings; T3; F6 (CHANGED) | Every session whose picture holds the planet, on the planet's own lane through the gateway. No realm-to-realm crossing. |
| C20 | shape vs code | Runoff for the cut, rain for denudation | Two laws; F49 | Not a conflict. The stream-power cut reads the runoff, with K₀'s water term normalised to Earth's land runoff (`crates/terrain/src/solve.rs:83-88`, `:1097`); the denudation reads the rain, for the reason the code states (`crates/terrain/src/craters.rs:139-157`). |
| C21 | the climate law vs T5 | The seasonal circulation of a 55° tilt is "the monsoon" | T5; T9; F29 | Asked (Q3). |
| C22 | light vs the accepted shadow ladder | "Shadows from the drawn surface" against casters one rung coarser | DEFERRED.md:8166-8172 (accepted by the owner); `terrain.rs:303-313`; F57 | Law L2 restated: shadows come from the same field, the casters one rung coarser, and a gate holds the edge in display pixels (§5.8). |
| C23 | the ladder vs full screen | One ladder pixel is about 3 display pixels | V14 D8-1; B22 item 3; F32, F63 | Asked (Q9). Every swap gate reads display pixels too. |
| C24 | the bake order vs SL2 | A viewer's eye orders the planet's bake | SL2; SL7; F4 | The planet's own occupants first; outside viewers by tile number (Q7). |
| C25 | the planet's sun vs SL1 | T3 class 1's almanac against SL1 clause 3 | T3; SL1; SL9; F5 | Asked at the weather step (Q17). The ground step needs no server sun. |
| C26 | the ridged centring vs R2 | A world change for a term that phase 2 retires | B22 Q5; r2 §7.4; F48 | The zero-mean law becomes a gate now; no centring is built (Q1). |
| C27 | the cell mean vs B16 | A pillar flattens while it stays pixels tall | B16; V10; T7 rule 3; F28 | Rule S (§3.1): a compact tall form keeps its outline. |
| C28 | G-LOOK vs the fill and the weather | G-LOOK comes before clouds and water | B35 item 4; B6 item 4; F47 | Asked (Q18). |

---

## 2. The quality bar: checks that can go red, and the stands

Every reading below is a READING (B23). A red reading is reported and never stops a step by itself. The owner's look
judges. Three exceptions are budgets, not believability bands: the frame, the build rate and the memory (§7); a red
budget stops its phase and goes to the owner (rule FR1).

**The pixel is named in every gate** (F63). A *ladder pixel* is 2·tan 22.5°/720 = 1.15 mrad (`crates/core/src/geometry.rs:1160-1171`;
`crates/client/src/ladder_view.rs:199-210`); T7's line and the ladder's switch distances use it. A *display pixel* at the
owner's full screen (2 168 rows drawn, the engine's default 45°, `crates/client-render/src/lib.rs:1485`) is 0.382 mrad, so
one ladder pixel is 3.0 display pixels (ARITH). A *render pixel* is the display pixel ÷ the render scale (1.5 render pixels
a ladder pixel at s = 0.5, ARITH). The eye's rule (B27) judges the display picture; every swap gate prints both units.

### 2.1 The checks

| Quality | Reading | Instrument | Today (mark) | Band and its source | Phase |
|---|---|---|---|---|---|
| 1 structure | G-SPECTRUM: 2-D spectrum per stand at rungs 0, 3, 5, 7, 10; band slopes, the largest bump, the rollover | NEW (Perron, Kirchner & Dietrich 2008's method, ESTIMATE source) | a bump at 3.125 km (DOC status §5 D3); hill spacing 3.2 / 3.2 / 4.2 km (MEASURED gen 19) | the Earth stand for each relief class (UNMEASURED, M0-4) | 0, 2, 3 |
| 1 | The noise share: per band, the share of the variance that no law explains | NEW | ~100 % under 16 km (ARITH) | under half at every band over 50 m (ESTIMATE; B35 "no scale left as plain noise") | 2, 3 |
| 1 | Skyline REPORT at V-ref: breaks at 0.25° and 0.50°, the largest rise, the range, the relief 30–60 km away | `reference_stand`, skyline march | 0; 0; 0.096°; 0.90°; 1 612 m (MEASURED gen 20) | 2.5–5.5; 0–3.3; 0.43–1.07°; 4.87–7.49°; 1.5–3 km (DOC 11 §0, 01 §2); a report (T6 ask 5) | every phase |
| 1, 7 | G-SILHOUETTE: the skyline's angle step at each rung swap, in display pixels, on the pillar, tower and mesa stands | NEW (F28) | no tall form exists today | ≤ 1 display pixel (B16; T7) | 4 |
| 2 land makes sense | G-HACK n and c at rungs 3, 5, 7 | `shipped_drainage` | n 0.224 / 0.205 / 0.251; c 2.63 / 2.46 / 2.95 (MEASURED gen 19, belt / coast / plain) | n 0.45–0.7, c 1–6 (DOC B3) | 2, 3 |
| 2 | Closed basins ≥ 10 m per 100 km²; dry ground in a basin ≥ 1 m | `shipped_drainage` | 2.84 / 2.55 / 2.00; 35.6 / 41.3 / 9.8 % (MEASURED gen 19) | the Earth stand (UNMEASURED) | 2, 3 |
| 2 | G-DRAIN-DRAWN: closed pits ≥ 1 m deep per 100 km² at rungs 0–3; random drops reach a budget terminal | NEW (F31) | UNMEASURED | Earth lidar stands (UNMEASURED, M0-4); only process forms (kettles, dolines, playas) may hold pits | 2, 3 |
| 2 | G-RIVER; on A0: G-RIVER-FOLLOW, G-BREACH-RESIDUAL, G-SILL | `lake_census`; r2 §10.4 | 0 rising steps (MEASURED gen 21) | 0 | 1, 2 |
| 2 | G-HYPSO: humps, integral, share over 1 000 m | `lake_census` | −9 241 / +9 m; 0.590; 63.11 % (MEASURED gen 21) | Earth's curve (DOC B3); the share UNMEASURED | 1, 2, 4 |
| 2 | G-GRID: flatness by class, Tarboton, fourfold, G-LAKE-WALK; also on R3's own field and on rock borders | `grid_bias`, `lake_census` | valleys 1.205; flats 1.470; lakes 1.553; walk 14.07° (MEASURED gen 21) | ~1.006 (95th 1.012, DOC W13); 12° (DOC) | 1, 2, 3, 4 |
| 2, 7 | G-MEAN: the signed mean of every fine term over each ring and over each 1 km cell | NEW in `rung_swap` (F23, F48, F59) | 14.0–224.2 m at m = 1 for the ridged term (ARITH §3.1) | 0 within its error | from 2 |
| 3 process | G-FLINT θ | r2 §10.4 | UNMEASURED | 0.35–0.6 (DOC r2) | 2, 3 |
| 3 | G-HILLSLOPE (hilltop curvature against E; R* against E*) | NEW | UNMEASURED | Roering, Perron & Kirchner 2007 (ESTIMATE source) | 3 |
| 3 | G-JOIN: junction angles by aridity | NEW | UNMEASURED | Seybold, Rothman & Kirchner 2017 (ESTIMATE source) | 3 |
| 3 | G-CRATER, G-SFD, G-DD (0.2 fresh), G-RETAIN; G-RAYS (optical maturity and ray age on airless bodies) | extends G-CRATER | 1 668 macro craters (MEASURED gen 20) | Neukum, Gault, Hartmann (DOC `craters.rs:84-147`); Lucey et al. 2000 (ESTIMATE source) | 2–4 |
| 3 | G-BENCH, G-CLIFF-H (no face over H_c for its own rock), G-SLOPE-BIMODAL, G-CONTACT | NEW | UNMEASURED | the laws of §3.2 K3 | 4 |
| 3 | G-TALUS, G-SLIDE, G-FAN, G-DELTA (exact), G-MORAINE (exact), G-TROUGH-B, G-CIRQUE, G-COAST-FORM, G-DUNE-λ / TYPE, G-MEANDER, G-SEAFLOOR (depth against age; abyssal-hill RMS), G-FROST (block-field share between the tree line and the snow line) | NEW | UNMEASURED | the law of each family (§3.2) | 4 |
| 3 | G-ICE; G-BUZZSAW | census | 3.74 % of land; 1 296 m (MEASURED gen 21) | the ice law against EARTH'S REAL hypsometry and climate (M0-4; F45); 1 500 m (DOC B3) | 2, 4 |
| 4 surface follows shape | Slope histograms for each (biome, relief, rock) class | `slope_histogram`, extended | 1.6–2.0° over the planet before 8a (DOC) | the Earth stand by class (UNMEASURED) | 2–5 |
| 4 | G-TREELINE; G-SNOWLINE (both aspects); G-SNOW-SLOPE (snow share by slope class); G-BARE-SLOPE (bare-rock share by slope class, from the soil law); G-WOODY | NEW (F30, F38) | UNMEASURED | Paulsen & Körner 2014; Ohmura 1992; Bernhardt & Schulz 2010; Heimsath et al. 1997; Sankaran 2005 (ESTIMATE sources); the Earth stands by class | 5, 6 |
| 4 | G-BIOME-RUNG (`assert_eq` at every rung); G-PAINT-AGREES = gpu-drift of the recipe's two compilations; G-TREE-FIT (the drawn trunk against the generator's collider) | NEW (F1, F7) | the class moves with the rung today (`crates/recipe/src/plan.rs:214`, `crates/recipe/src/height.rs:423`) | exact; exact; within the import tolerance | 5, 6 |
| 5 variety | G-VARIETY: classes within 60 km of the V-ref eye | NEW (stands census) | UNMEASURED | Earth's own distribution by relief class, from global geology and Köppen maps (M0-4; F43); the third reference stays the owner's look, never a band | 4, 5 |
| 5 | G-BORDER: sinuosity (1 km against 8 km); G-BORDER-CONTOUR: how closely a rock border follows the contour lines; G-GRID on province and biome borders | NEW (F35) | rock borders are 8 km squares (`crates/terrain/src/artifact.rs:1211-1237`) | Earth (UNMEASURED) | 4, 5 |
| 5 | P7a desert belt; P7c rain-shadow pair | NEW | UNMEASURED | the corrected circulation's prediction (§3.2 K11; Q3) | 2, 5 |
| 6 light, air | G-SHADE-NEST: the shade step at each swap on the EYE'S WINDOWS (≥ 1°, display pixels) and on ring means, at 5°, 15° and 45° suns | `ring_step` + windows (F70) | UNMEASURED | half a level (B31) | light |
| 6 | G-SHADOW-SWAP: a side-lit shadow edge's move at every swap pair, in display pixels, at 5° and 12° suns; G-SHADOW-HANDOVER: the overlap of the near cascades and the far march | NEW (F57) | UNMEASURED; ARITH: the measured 0.64-ladder-pixel p99 moves a side-lit edge 3.0 ladder pixels at 12° and 7.3 at 5° | no pop by the eye's rule (B27) | light |
| 6 | G-TURN-SHADE: a 360° turn under a 5° sun with ridges 3–5 km away | NEW (bench M0-12) | the fault is in the code (`vendor/bevy_pbr/src/render/shadows.wgsl:117-126`, `:192-198`); UNMEASURED in pictures | no pop by the eye's rule | 0 (bench), light |
| 6 | G-FAR-SHADOW-STILL; G-CANOPY-NEST; G-FOREST-FOLD with a silhouette gate at the crown-to-fold hand-over; G-ADAPT; G-TILING | NEW (F58) | — | still; half a level; ≤ 1 display pixel; unseen; no autocorrelation peak | 6, light |
| 7 no flicker, 60 fps | Swap p99, height and shade, for each rung pair, in ladder AND display pixels | `rung_disagreement`, `rung_swap` | 0.64 ladder px worst p99 (MEASURED, DOC `crates/terrain/src/body.rs:193`) = 1.9 display px (ARITH) | ≤ 1 ladder px (T7); the display reading is judged by the eye's rule (Q9) | all |
| 7 | The eye's rule on every leg; G-CROSSING | `terrain_moving_eye` | no pop on the walk, 240 m/s, 528 m/s or the turn; ONE pop on the 1.4 m/s hull leg (level 1.02), red, moved to T4 (MEASURED, `docs/design/DEFERRED.md:11608-11611`) | B27; half a level (B31) | all |
| 7 | G-CLOSURE (the shares at rung L against rung 0, on ring means AND eye windows); G-FAR-EXACT (a parent is the mean of its children); G-OPEN (the drawn mesh against the recipe's own rung-0 cells, which the collider will read) | NEW (F19, F70) | — | 1 % absolute; 1 m and 1/255; 0 | 2, 4, 5 |
| 7 | G-FRAME-TOTAL: today's measured base + every phase's measured line + the reserved lines G and H, p99 on moving legs, GPU and CPU | window meter (F53) | 23.1 ms median at full screen (MEASURED B34, not clean) | ≤ 16.7 ms p99 and ≤ 1 frame in 1 000 over it (DOC one_volume B-FRAME); a gate from phase 3 (§7 FR1) | every phase |
| 7 | B-BUILD (ms a chunk, weighted by the ask, at 7 and at 4 workers); B-RATE (late chunks a second on the low legs at 240 and 528 m/s, 50–500 m up) | `chunk_budget`, arrival census (F54) | about 230 late chunks a second at the owner's speed (MEASURED, DEFERRED "THE WAVES") | not worse than the phase-0 baseline; B-BUILD ≤ 6.5 ms (DOC one_volume §10A.2) | every shape phase |
| 7 | G-NO-STALL with the pipeline meter's count after the loading; B-MEM (GPU, CPU, gateway); B-LOGIN (arrival time of every new part) | window meter; `crates/client-render/src/pipeline_meter.rs` (F62, F65, F66) | 3 pipelines compiled after the loading; 146 ms at the dusk boarding (MEASURED B34); graphics memory 2.5–2.9 GB on foot (MEASURED, status log 2026-09-28) | 0 after the pre-roll; each phase's line (§7); every planet whole ≤ 30 s (DOC one_volume B-LOGIN) | every phase |
| no drift | terrain-pin, gpu-drift, artifact-pin, G-AMP-PIN on x86-64 AND aarch64, the anchor digest, the tree skeleton pin | pins | gpu-drift 5 979 / 5 979 boxes (MEASURED gen 21) | equal | all |

### 2.2 The stands rule (one rule in five layers, run ONCE and pinned)

**The pin (F42).** Bench M0-6 runs the rule once in phase 0, on generator 22, and pins every pick by its direction, as
the coast and the plain are pinned today (`crates/bins/src/drainage_gates.rs:66-80`: *"every step is compared on the SAME
stands"*, ruling B6). Later phases print the rule's new picks beside the pins and ADD stands; they never move one.

1. **The class stands.** The class is (b, r, p):
   - **b, the biome.** Until phase 5, a Whittaker proxy from the row's temperature and rain bytes
     (`crates/terrain/src/artifact.rs:107-130`). From phase 5, the 19 ruled classes (T1 item 7). An airless body has no
     biome class.
   - **r, the relief.** Hammond's classes (Hammond 1964; Karagulle et al. 2017; ESTIMATE sources): the share of gentle
     slope (< 8 %) and the local relief in a ~10 km window. The window reads the rows' 3 × 3 nodes (24.6 km, ARITH)
     before R2 and A0 after it. Two more classes: a closed-basin floor and the sea floor.
   - **p, the rock.** The province (`crates/terrain/src/strata.rs:166-177`); phase 4 adds kinds.
   - **Context bits:** coast, ice, lake (`crates/terrain/src/solve.rs:151-154`).
   - **The pick.** A class qualifies with at least one view's area: π × 30² ≈ 2 830 km² ≈ 42 nodes (ARITH). The stand
     is on the class's largest patch, at the node farthest from the patch's edge by the summed chord (the flats' own
     "middle" rule, `crates/terrain/src/lakes.rs:407-421`). A tie goes to the smaller index.
2. **The family stands** (one set for each family of §3.2): where the family's facts are strongest, a node at its
   threshold, one random node of its domain.
3. **The line and border stands:**
   - the tree line; the snow line, one slope facing each way; a desert scree slope; the two sides of a rain shadow;
   - the forest at 1, 5 and 50 km; the band where the trees become the canopy fold;
   - **border stands** (F42): on each body, the three nodes with the most classes within 60 km.
4. **The random stands** (F42), from a hash of (seed, salt, i), split by the surface: on the home planet 20 on land and 4
   on the sea floor for the numbers (ESTIMATE: the count that the medians of the readings need), pictures on a subset;
   on the home moon 8; on the third body 12, split by its land share. Four uniform stands would put 0.85 on the home
   planet's land (ARITH: 4 × 0.2124).
5. **The pinned stands of today:** belt, coast, plain, pockets 1 and 2, the high lake
   (`crates/bins/src/drainage_gates.rs:64-100`); the reference stand (`reference_stand`'s pose, status §9); the 15° stand
   of the moving-eye gate (B19 item 2). The rule's own current picks print beside them (`rule_picks`,
   `drainage_gates.rs:172`).

**The framings.**
- V-foot: on foot, 1.6 m up (`EYE_HEIGHT_M`, `crates/terrain/src/artifact.rs:970`), the nose toward the steepest ground
  within 5 km.
- V-ref: the reference framing (B6 item 2): ~400 m over the local ground, looking across the valleys to the highest ground
  30–60 km away, ~55° of view. The reference look comes from the class stands in valleys: the reference looks from a
  lower point toward taller ranges, and today's pose stands on a summit and looks down (status §9 ⚠). The old pose stays
  pinned as a summit stand.
- V-high: 150 km up, or a quarter of the radius if that is less.
- V-orbit: orbit at 1.2 radii: 7 645 km at the home planet, ~424 km at the moon (ARITH).
- **Two suns** (F44): every V-ref class stand carries the first reference's low sun (12°–18° up, 100°–140° off the nose,
  V14 D8-7) AND the third reference's high sun (ESTIMATE from the picture: 40°–60°; the stamp states each). The light
  stands add 5°, 45° and dusk. A body with a bright neighbour adds one night stand.
- **The flights:** a descent from V-orbit to V-foot; traverses at 240 and 528 m/s, 50–500 m up; a turn under a low sun; a forest leg;
  a climb at dusk through the far swaps. The eye's rule (B27) judges each flight.

**The bodies.**
- **B1, the home planet:** `Planet(4030111653607004909)` (`crates/terrain/src/home.rs:51`).
- **B2, the home moon:** `Planet(2918819812335288845)` (`home.rs:75`): look radius 353 km, g = 0.330 m/s² (`home.rs:79`),
  no air, no water, tidally locked, 27 744 nodes (DOC `home.rs` comment), its solve 0.3 s (DOC).
- **B3, by rule:** the solid body with the largest charter distance from B1 over log water, pressure, temperature and
  gravity, a dry world with air preferred. The candidates (DOC `slice_8c_bench.md` §1): rocky planets of 4 589 km and
  7 393 km, an ocean planet of 4 840 km. The gas giants are refused (no solid surface). Their charters are UNMEASURED, so
  bench M0-5 prints them first.
- **Every phase's pictures include class stands on B2 and B3** (F13; B23 item 3). The icy-moon laws are built only
  together with a stand on a real icy body of THE world, if M0-5 finds one (F46).

**Counts.** Numbers on every stand at every step, headless: ~2.6 s for each rung-7 square (ESTIMATE). Pictures on a
subset chosen with the owner: ~36 on the home planet, ~12 on each other body (ESTIMATE). Occupied classes: 20–60 of the
315 possible (ESTIMATE). Owed tooling: a spawn on another body (`land_stand` serves only the home planet,
`crates/bins/examples/land_stand.rs:1-6`).

---

## 3. The catalogue

### 3.1 The two far-view rules, and the contract every family obeys

**RULE M, THE MEAN (W16 law 2 made into numbers).** At every rung, the drawn product equals the mean of the finer rung's
product over the cell, within T7's line. It applies to four products: the height, the material shares, the shade (the
light nests) and the canopy cover. A term obeys rule M in one of two ways:
- **(a)** a zero-mean detail may drop at a swap: the fine relief under A0, R3's components with their hold, every
  kernel term with a zero mean over its cell;
- **(b)** any other term stays as its CELL MEAN where the rung cannot resolve it (a cone, a crater, a sub-pixel lake as a
  tint, B18 Q1's rule), and the bake folds its volume per A0 cell into A0's coarse means and the far map, so it does not
  vanish at the 10 → 11 swap (F23).

**RULE M's LIMIT.** On ground whose slopes stay under 45°, a feature narrower than one cell is also lower than one cell,
so it stands under one ladder pixel at the rung's switch distance, and the mean loses nothing that the eye sees. A form
that is taller than it is wide breaks this (F28). ARITH: a pillar 30 m wide and 100 m tall stands 6.3 ladder pixels tall
at rung 4's switch distance (13.9 km) and 3.1 at rung 5's (27.8 km), while rule M draws it at rung 6 as a bump of
30²/64² × 100 m = 22 m, 0.7 ladder pixel at 27.8 km. That is a change of shape between rungs, which B16 forbids. The
step is 100 − 22 = 78 m, so T7 rule 3 alone would move the switch out to 78 m ÷ 1.15 mrad ≈ 68 km in pillar country, and
rung 4's ring would grow from 455 km² to ~14 380 km², about 30 times (ARITH, F28).

**RULE S, THE OUTLINE (B16 for compact tall forms).** A compact form whose height exceeds its width — a pillar, a tower,
a needle, a fin, a narrow butte, a sea stack, a volcanic neck — is a PLACED FORM of the static shape:
1. both hosts compute it from its law (§3.2 K3 law 9), so collision reads the same shape (SL10);
2. the cells draw it while it is at least two cells wide at the drawn rung;
3. from that rung out, its own coarser meshes, made from the same form description on the client, draw it, and the
   cells under it hold the base ground only, so nothing is counted twice; this is terrain, not decoration, so V4 step
   9 (a) does not bind it;
4. when its height falls under one display pixel, it becomes a tint of its true share (rule M (b));
5. the hand-over from the cells to the form's meshes lies inside the rung's crossfade band, and G-SILHOUETTE gates it
   (≤ 1 display pixel).
The crease of a far mesa's rim is the same question at a larger size: dual contouring is benched at rung 0 and at
rungs 5–9 (M4-9), and the owner decides it on pictures (V10).

**TODAY'S LIFT, AND WHY NO CENTRING IS BUILT.** The ridged fine term `((1 − |n|) << 1) − 1` has a positive mean
(`crates/recipe/src/height.rs:361-368`):
- MEASURED: 571.8 m at share one at rungs 0–5, 513 m at rung 7, 224 m at rung 9 (DOC `docs/design/DEFERRED.md:10362-10366`).
- ARITH: one constant explains the three readings: μ = 571.8 / 1 025.3 = 0.558; 0.558 × 920.3 = 513.5 m;
  0.558 × 401.8 = 224.2 m. Each ridged swap therefore drops a whole ring by μ·a: 14.0, 44.6, 111.0, 178.3 and 224.2 m at
  m = 1 from 5 → 6 to 9 → 10 (W16's fault D, rings of different shade).
- Under a field the ridged fold retires (r2 §7.4), every drawn body holds a field (a client builds nothing from the recipe
  alone, `crates/core/src/look.rs:82-88`), and an unbaked part waits instead of drawing a stand-in (C4). So the ridged
  term has no future past phase 2, and a centring would cost one world change and one re-freeze for nothing (F48). The
  zero-mean law becomes a gate now (G-MEAN), and phase 2's fine relief must pass it by construction.

**THE CONTRACT (P1–P16)** for every term of every family:
- **P1.** A term is a function of the site: the seed, the address, the charter, the realm's shipped rows or A0. It never
  reads the eye, the tick or the rung, except for the fade weight and the prefilter's cell (SL10; T3).
- **P2.** The mass lives in the coarsest level that resolves it. A finer term has a zero mean over its cell, or it keeps
  its volume and the bake folds that volume (rule M (b)).
- **P3.** A survival constant MEASURED for the term's own profile (T7 rule 1's method). The hand-over step stays under
  one cell of the next rung, or T7 rules 2–3 act.
- **P4.** A compact feature is drawn as its cell mean where the rung cannot resolve it, unless rule S applies.
- **P5.** Fenced integers and branch-free masks, no division in the kernel; one source for the shard, the client's CPU
  and the card.
- **P6.** Gathered once for each box, never once for each column.
- **P7.** No switch on the body kind: the facts decide.
- **P8.** A census reading that can go red, on the pinned stands of §2.2, on all three bodies.
- **P9.** A late item enters on its stand-in over the 0.5 s ramp (B18 Q7).
- **P10.** The temporal contract (temporal §4.6): it states its motion, marks itself reactive, or draws after the
  reconstruction.
- **P11.** No pipeline compiles in flight: each phase lists its new pipelines and warms them before play (F65).
- **P12.** "Up" comes from the realm's gravity function, in every term, in the cover and in the canopy fold (V4 15.2;
  F27).
- **P13.** Each law states its deterministic form before its phase: an integer table, the crate's own function
  (`crates/terrain/src/gf.rs:177-225` holds its own `exp`, `ln`, `powf`), or the recipe's exact reciprocal; never a
  platform transcendental (`crates/terrain/clippy.toml:9-40`; V1 item 4; F22).
- **P14.** Each new drawn thing states its rule at every distance: rule M (a), rule M (b) or rule S, and its share in the
  material closure and in the canopy fold (F24).
- **P15.** No new minimum: a fine term keeps its gradient under the drained slope of the ground under it, so only a
  process (ice, karst, wind, impact) makes a closed hollow (F31).
- **P16.** Each term prints its cost on the builder (B-BUILD) and its vertex line (FR5) before it lands (F54, F61).

### 3.2 The families (B35 item 2), with their laws, their layers and their checks

**K1 RIVERS AND EROSION.**
- **Today.** Implicit stream power (Braun & Willett 2013), m = ½, n = 1, K₀ = 2 × 10⁻⁶ /yr with the drainage AREA
  (Earth; Stock & Montgomery 1999; Whipple & Tucker 1999; `crates/terrain/src/solve.rs:79-88`), on a one-receiver tree.
  The passes cut with the RAIN times the area (`solve.rs:1093-1104`). The deposit fills the first lake node to its spill,
  and that deposit makes the washboard: 1 004 m ridges at 45° (MEASURED B11). The drawn rivers are retired (W16.1).
- **Added.** R2 (§5.2); R3 (§5.4); floodplains over the meander belt by Leopold & Wolman 1960's amplitude law
  A = 2.7·w^1.1 with A and w in FEET, ≈ 3.04·w^1.1 in metres, 5.5 w at w = 391 m (ARITH; ESTIMATE until read) — the
  code uses 2.7 w (`crates/terrain/src/river.rs:15-16`, `:40-43`) and changes; ALLUVIUM (gravel and sand bars) as a
  material on channels and floodplains, read from A0's discharge (F49); fans with A_f = c·A_d^n, n ≈ 0.9 (Bull 1964);
  strath terraces paced by the glacial epoch (`solve.rs:142`); deltas by the deposit machinery up to the base level.
  None of these is a shore rule.
- **K₀'s water (F49).** K₀ is calibrated on the drainage area; the code reads rain × area. When the cut reads the runoff
  (one water, Q4), the effective area falls by runoff ÷ rain (75 ÷ 374 over the home planet's land, MEASURED,
  `crates/terrain/src/craters.rs:146-152`), and the cut weakens by √(374/75) ≈ 2.2 (ARITH) unless K₀'s water term is
  normalised to Earth's land runoff (the calibration basins' own water; read at first hand before M2-2).
- **Layers.** Deltas and the budget in the SOLVE. Floodplains and fans ≥ 2 km in A0. Fans and terraces of 0.1–2 km in
  R3. Small terraces in the KERNEL, only as a pure function of R3's field, never a stamp (W16).
- **Other bodies.** The moon: no runoff, so no fluvial pass (r2 §4.3). A dry world: no fluvial form unless the owner
  rules a wet past (K15). An icy moon: none.

**K2 SLOPES AND SOIL.**
- **Laws:**
  - the nonlinear hillslope q = K·S / (1 − (S/S_c)²) (Roering, Kirchner & Dietrich 1999; S_c ≈ 1.25, the Oregon Coast
    Range: the limit of soil-mantled, forested slopes; ESTIMATE until read);
  - R*(E*) (Roering, Perron & Kirchner 2007); channel heads A_c·S² = const (Montgomery & Dietrich 1988, 1992);
  - soil production P = P₀·e^(−h/h₀) (Heimsath et al. 1997): the soil depth h = h₀·ln(P₀/E), where E is the LOCAL
    lowering rate of the drawn surface — the hillslope law's E = −∇·q on R3's field and on the fine relief. The node's
    denudation (`craters.rs:330-346`) is Earth's OUTCROP rate over one 67 km² node: the wrong quantity and the wrong
    scale for soil, so it stays the crater-retention clock only (F30);
  - bare rock where E > P₀, or where the slope nears S_c (the threshold hillslope);
  - repose (tan 0.47 wet to tan 0.70 dry; `solve.rs:104-108`, `:1644-1650`) ONLY for debris: scree under a cliff and
    talus aprons. The apron is the plane from a cliff's foot at repose, its volume the face's retreat;
  - slide runout by the Heim ratio (Scheidegger 1973); block sizes from the joint spacing (Palmström 2005).
- **One rule for soil and bare rock:** this law (F21, F30). K12's materials read it, and no slope threshold stands in
  for it.
- **Layers.** Slides ≥ 1 km in the SOLVE and A0 (mass kept). Hillslope forms in R3. Aprons and the soil law in the
  KERNEL. Blocks as OBJECTS.

**Example.** A pilot lands in a wet valley. The 30° forested spur above her keeps its soil (S = 0.58, under S_c), so it
shows forest on soil to about 40°, rock only on its sharp crest, and deep soil in the hollow below.

**K3 ROCK CONTROL** (the forms of the third reference: mesas, buttes, pillars, the dark ridge, pale boulders).
1. **Erodibility** K = K₀·(σ_T,shale / σ_T,rock)² (Sklar & Dietrich 2001; `crates/terrain/src/strata.rs:179-227`), read at
   the BED of the eroding cell; R2 and R3 read it.
2. **The cliff law for intact rock:** H_c(β) = 4c·sin β·cos φ / (ρg·[1 − cos(β − φ)]) (Culmann 1866; Schmidt &
   Montgomery 1995), with c and φ for EACH rock (Selby 1993; ESTIMATE until read). ARITH at c = 1 MPa, φ = 35°,
   ρ = 2 600 kg/m³: a vertical face stands to 301 m on the home planet. The moon's crust is impact-crushed breccia and
   megaregolith with far less cohesion, from its own porosity or impact-history law (ESTIMATE source; F50), so the
   draft's "30 times taller on the moon" (intact rock) is withdrawn. A limit that IS a law (B23).
3. **Bench strength** from the hardness contrast 1 − q = (σ_soft/σ_hard)²; it replaces the ladder-solved 0.560
   (`crates/terrain/src/body.rs:1210-1236`). ARITH: sandstone over shale q = 0.64; quartzite asks 0.96 and the fold-over
   limit stops it at 0.94 (`body.rs:384-396`).
4. **The cap share** from the sedimentary record (Blatt 1970); it replaces the ASSUMED half
   (`crates/recipe/src/terrace.rs:131`).
5. **Structure.** The bed index reads r − U_s(site), so beds tilt on a belt's flank (ARITH ~1.9° on Earth's numbers).
   Folds at Biot's wavelength (Biot 1961) for the COMPETENT STACK, kilometres thick, never one 138 m bed (F50).
6. **Joints:** two orthogonal sets spaced ~ the bed thickness (Bai & Pollard 2000; Engelder 1985); cooling columns
   (DeGraff & Aydin 1993): the second reference's columnar pillar.
7. **THE OUTCROP RULE** (F35; it replaces the draft's warped contacts). The base of each sedimentary cover is a 3-D
   surface: its thickness from the solve's deposit and the basin's subsidence (a law; McKenzie 1978 for stretched
   basins, ESTIMATE source), its dip from law 5. The rock at a point reads the point's radius against that surface: over
   it, the cover's beds; under it, the basement. So the border on the ground is where the ground cuts the surface: the
   basement shows in the canyon floors, and the cover stays on the ridges as mesas and buttes, at every rung, and the
   far shares fold it exactly. The seeded warp goes. Data: A6 (the cover's thickness, one byte a node) becomes
   necessary.
8. **The deep stack:** a bed's rock shows at every depth of the cover, not only in today's 23–92 m band
   (`crates/recipe/src/cell.rs:425-452`).
9. **PLACED FORMS** (rule S): where a hard cap over soft beds is cut through, pillars, towers, needles and fins stand;
   the outline from the joint spacing (law 6), the height from the bed stack and the cap's retreat, the face limited by
   law 2.
- **Layers.** SOLVE: provinces, the cover's thickness, the dip. A0: mesas and cuestas ≥ 2 km. R3: buttes and fins.
  KERNEL: benches, joint cuts, banding. PLACED FORM: pillars and towers. VOLUME: undercuts. OBJECT: fallen blocks.
- **Other bodies.** The moon gets no beds. Today it draws sediment provinces by the same rule as every body
  (`crates/terrain/src/land.rs:506-528`) on the 0.40 continental share of every body (`land.rs:88-91`): a T9 defect. A
  dry world keeps its caps long. On an icy moon, ice is the rock.

**K4 ICE.**
- **Today.** Egholm 2009's mass balance, Nye's thickness, and a cut only where the bed is warm (`solve.rs:115-141`).
  G-ICE reads 3.74 % of the land (MEASURED gen 21).
- **Gaps.** The cut is capped at the ice's thickness (`.lesser(thickness)`, `solve.rs:1523-1524`); the cut never enters
  the deposit, so no moraine and no rebound follow; the troughs, cirques and horns are narrower than a node.
- **Laws.** Troughs z ∝ |x|^b, b ≈ 1.5–2.5 (Harbor 1992), as a reading; cirques (MacGregor et al. 2009); arêtes and
  horns through the cliff law; moraines from the ice's sediment flux, the volume exact.
- **Layers.** SOLVE: the sediment and the rebound, with the cap removed. A0: one more gather (ice covers ~0.8 % of the
  globe, ARITH 3.74 % × 21.24 %). R3: cirques.
- **Other bodies.** The moon has no ice (`solve.rs:1763-1768`).

**K5 WIND (real dunes)** (revised, F39).
- **Switch:** air; an aridity of semi-arid or drier; a SAND SUPPLY — the solve's sediment in dry closed basins, on dry
  floodplains and on coasts (the cover class, A6); a wind over the threshold.
- **Laws** (all ESTIMATE until read):
  - the threshold u*_t = A·√(((ρ_s − ρ)/ρ)·g·d) (Bagnold 1941);
  - the smallest dune λ ≈ 20·l_sat with l_sat ∝ (ρ_s/ρ_air)·d (Claudin & Andreotti 2006): ~17 m on the home planet
    (ARITH, ρ_air = 1.443 kg/m³);
  - GIANT DUNES (star dunes and draa, 50–300 m tall, 1–3 km apart) spaced by the depth of the atmosphere's boundary
    layer (Andreotti et al. 2009);
  - the dune type from the SPREAD of the wind's directions over the year against the supply (Wasson & Hyde 1983; Rubin &
    Hunter 1987). The spread comes from the seasonal circulation (K11 A3; Q3). Today's climate states one fixed direction
    for each node (`crates/terrain/src/climate.rs:651-681`), which gives transverse dunes only;
  - slip faces at repose; yardangs 3–10:1 (Ward & Greeley 1984);
  - the wind's SPEED law is owed (research first).
- **The home planet.** The draft's predictions (one cell, an east wind everywhere, linear and star dunes in mid
  latitudes) rest on today's Earth-fitted circulation and are withdrawn until bench M0-9 runs the corrected law.
- **Layers.** Dunes ≥ 1 km apart in A0 or R3, as deposits that keep their volume: a sand sea is a deposit, so it shows its
  relief from high flight. Small dunes in the KERNEL, one anisotropic wave for each type with a zero mean about the
  deposit's surface, the lee face at repose, stopping by P3.

**K6 COASTS.** Forms of the ground only. The water comes in the fill step, and the coast is judged after it (B22 item 2).
- **Laws:** the storm wave scaled by the air and the gravity (`solve.rs:1655-1668`; 11.8 m on the home planet, ARITH);
  cliff retreat ∝ ln(F_wave/F_rock) (Sunamura 1992); the platform's width from the TIDAL RANGE that the charter states
  (A14), never from the almanac (T3; F17); Dean's beach profile h = A·x^(2/3) (Dean 1991); barrier coasts and spits by
  long-shore drift (K18). All ESTIMATE.
- The term sits inside the erosion operators at the body's base level and reads no water or side word for a column, so
  it is not a shore rule (B21). The Beach material comes from the solve's coastal deposit, never from a column's height
  against the sea (F17).

**K7 FIRE (volcanoes).**
- **Laws:** arc volcanoes over the slab at ~100 km depth (Syracuse & Abers 2006); arc spacing by the lithosphere (de
  Bremond d'Ars et al. 1995); edifice height by the relief law (`body.rs:415-451`; Olympus Mons as the check); a lava
  dome by the plastic profile (Nye 1952; Blake 1990) — not Hulme 1974, a lava-flow paper (F50); shields from the erupted
  volume; cinder cones H = 0.18 W (Wood 1980). All ESTIMATE. The hotspot count needs a law (owed).
- **Layers.** SOLVE: edifices ≥ 16 km, stamped before the water works, so rivers cut radial valleys; new provinces. A0:
  2–16 km. KERNEL: cones and maars as point processes whose volume the bake folds per A0 cell (rule M (b)); a neck taller
  than wide is a placed form (rule S). The moon (T_e 1 094 km, `home.rs:96`) has none.

**K8 CRUST MOVEMENT.**
- **Today.** Voronoi plates (`land.rs:7-11`). A transform makes no form (`land.rs:474-476`). From orbit the owner saw
  straight coasts ~1 000 km long that meet at sharp angles (DOC status log 2026-10-01; cause UNMEASURED).
- **Laws:** folds (K3 law 5); normal faults spaced ~ the brittle crust (Vendeville et al. 1987); throw γL (Cowie &
  Scholz 1992); scarp diffusion (Hanks 1984); a transform as a soft strip in the hardness field. All ESTIMATE.
- **Readings in phase 0:** the coastline's divider dimension at 100–1 000 km (Mandelbrot 1967) and the belts' curvature
  (Frank 1968). The cure belongs to this family in phase 4.

**K9 IMPACTS.**
- **Today.** Every crater law is built: Neukum production, slope −2, 5 % saturation, air screening, Pike shapes with
  the 1/g transition, ejecta, retention clocks (`craters.rs:84-147`). The floor is 2 nodes (16.4 km, `craters.rs:104`).
  A bowl is clipped at the envelope (`solve.rs:1746-1751`).
- **Added.** Craters of 1–16 km in A0, stamped before R2's rivers. Craters under 1 km in the KERNEL from the same
  production law, gathered per box and kept by volume (P4). Diffusive degradation with κ ≈ 5.5 m²/Myr, which is Fassett &
  Thomson 2014's value for lunar-mare craters near 1 km; its use at other sizes is an extrapolation and is stated as
  such (F50). Regolith depth (Shoemaker 1969). RAYS AND OPTICAL MATURITY as a law of the crater family on every airless
  body (Lucey et al. 2000), not an option: they are the main structure of an airless body seen from orbit (F50).
- **ARITH.** On the moon ~30 craters touch a rung-0 box. On the home planet the air screens craters under 250 m, and a
  1 km crater lasts ~17 Myr, so only ~1 500 craters ≥ 1 km exist on all the land.

**K10 KARST AND CAVES.** Denudation D = q·c_eq(pCO₂, T)/ρ (White 1984; Dreybrodt 1988); dolines at joint crossings;
poljes as the solve's own dry closed basins on limestone; caves by the same rate and the water table; lava tubes in
volcanic provinces; TOWER KARST in humid limestone (F51). All ESTIMATE. They replace the drawn cave band
(`body.rs:1151-1176`): a world change for caves.

**K11 THE CLIMATE ON THE GROUND** (revised; F29, F33, F37).
- **Today.** Per 8 192 m node: North 1975 / Budyko–Sellers T̄; the lapse Γ = 0.666·g/c_p; Held & Hou cells with
  ΔH = 0.388, *"a REVERSE FIT to Earth's 30°"* (`climate.rs:102-104`), capped at 6 (`:106`); the count reads the gravity,
  the scale height, the spin and the radius, and never the body's own contrast (`climate.rs:321-347`); in an even cell the
  surface wind blows from the east at every latitude (`climate.rs:651-681`); the Tetens supply × the belt × Smith's lift,
  clamped to [0.1, 4] (`climate.rs:127-130`); the rain clamped to [1, 10 000] mm/yr (`solve.rs:56-60`;
  `climate.rs:792-795`); PET, the Turc–Pike runoff, UNEP aridity, Ohmura's ELA; ONE seasonal column for ocean and land
  (its stated limit, `climate.rs:445-447`); no daily term. Three bytes ship; no client code reads them.
- **The home planet (ARITH on `crates/terrain/src/home.rs:99-119`).** The tilt is 55.1° (cos 586/1 024). The annual
  insolation's P₂ coefficient s₂ = −(5/16)·(3cos²ε − 1) ≈ +0.005, against Earth's −0.48, so the annual contrast from the
  equator to the pole is about zero (the code's own T₂ = +0.21 K). Held & Hou's annual-mean cell needs that contrast, so
  the law's circulation is the SEASONAL cell that follows the sun across the equator (Lindzen & Hou 1988; for high tilt,
  Ferreira et al. 2014; ESTIMATE until read). Today's code reads Earth's ΔH and gives sin²φ_H = 1.91 > 1: one fixed cell
  (ARITH). So the draft's figures — one cell, ~990 mm of rain at the equator and ~90 mm at the pole, polar deserts, a tree
  line at ~1.33 km, ELA 1.69–4.39 km — are PRODUCTS OF TODAY'S CODE, not of the law. They are withdrawn until bench M0-9
  runs the corrected law (F29). What stays: Γ = 6.29 K/km, the air's density 1.443 kg/m³, the 3.0-day day (259 597 s,
  `home.rs:113`), the 374 mm of rain over today's land (MEASURED, `craters.rs:146-152`).
- **Added** (each replaces a cap, B23). A1–A3 and the circulation are SOLVE changes: they move into PHASE 2's macro
  change, measured alone before R2's first bake, so R2's valleys are cut by uncapped water (F33):
  - **A1** a two-mode seasonal balance with each node's heat capacity (North, Short & Mengel 1983); +1 B a node (A7).
  - **A2** a moisture budget along the wind (Smith 1979's condensation, W ≥ rain); it replaces the lift and rain clamps.
  - **A3 THE CIRCULATION FROM THE BODY'S OWN CONTRAST:** ΔH from the body's own s₂ (annual) and s₁ (seasonal) and its
    radiative response; the cell edges at each season by Held & Hou and Lindzen & Hou; the surface wind's sign change
    inside a cell at φ_H/√3 (Held & Hou 1980: the ground's net torque on the air is zero); no cell cap, an extrapolation
    past one cell named as such. The solstice cell is the kind of circulation that T5 names "the monsoon" (Q3).
  - **A4** the sun on each slope as the local term (Lee 1962; Swift 1976; Priestley–Taylor); orographic rain falls
    10–20 km downwind (Smith & Barstad 2004). Phase 5.
  - **A5** the tree line and the snow line as temperatures: a growing-season mean of 6.4 °C over ≥ 94 days (Paulsen &
    Körner 2014); the ELA moved by A4. Phase 5.
  - **A6** the class reads A0's height (C1). Phase 5.
  - **A7 THE DAILY SWING** (F37): a closed form of the day length and the surface's heat store (T3 class 1). The surface's
    daily amplitude grows as the square root of the period (ESTIMATE: ~1.7 times Earth's at a 72 h day), so night frost
    reaches ground that a seasonal model alone calls frost-free. Phase 2 with A1.

**K12 BIOMES, SOIL AND MATERIALS** (revised; F1, F2, F17, F21, F30, F38).
- **ONE SOURCE (F1).** The recipe computes, for each column and in fenced integers: the class (19 biomes in 5 bits,
  T1 item 7); the line quantities (the temperature at the column, the snow-line and tree-line heights, the soil depth,
  the repose tangent); and the expected material shares under the variance that the drawn rung drops (the closure's
  tables: integer CDFs and tails, P13). The shard's CPU, the client's CPU and later the card (rust-gpu, ruling F8 decision 5)
  run that one function; the far fold calls it too. The recipe's row mask grows to 32 rows (`crates/recipe/src/cell.rs:43-48`).
- **The class.** Regolith (no air); Ice (perennial: the solve's ice and the ELA law); Wetland (ln(a/tan β), Beven &
  Kirkby 1979); Alpine/Tundra over the tree line; then Whittaker 1975 on T̄ (at A0's height) against P, with Köppen's
  seasonal limits (Kottek et al. 2006). Today: four classes from two noises and literal thresholds
  (`crates/recipe/src/height.rs:395-455`; `body.rs:1132-1150`). Ocean and SeaIce are WATER classes and move to the fill
  pass; Beach is the solve's coastal deposit (F17; B22; one_volume §2.1: *"No water word enters the solid pass"*).
- **Cover inside a class**, continuous: Liebig's minimum of water (Sankaran et al. 2005), temperature and soil; the
  plant-type envelopes (Sitch et al. 2003); Beer–Lambert under the canopy (Monsi & Saeki 1953). So a border is an ecotone.
- **Materials** (one rule each):
  - soil and bare rock by K2's soil law (no slope threshold); scree and talus under cliffs by the repose law;
  - sand by K5's supply; alluvium by A0's discharge (K1);
  - snow in THREE LAYERS (F17): perennial ice and firn in the static shape (the climate law); the seasonal snow line as
    the almanac's paint, with no collision; lying snow after a storm as the weather memory's live state (§8.3). The static
    snow share follows Bernhardt & Schulz 2010 (snow does not stay on slopes steeper than its holding angle) and
    Winstral, Elder & Davis 2002 (the wind moves snow from ridges into lee hollows and gullies). Both are static functions
    of the slope, the curvature and the static wind, so every rung reads them alike (F38);
  - glacier ice as a surface material until the fill step gives it a thickness (an interim, §9.4).
- **THE SEAM (V14, kept; F2).** The stratum per vertex and the biome per column, for all ground. The vertex carries the
  column's expected shares (4–8 B); the shader only blends them across the triangle. No shader derives a threshold.
- **Registry.** New strata append after Marble 20 (`crates/terrain/src/strata.rs:40-41`); laterite, podzol, regolith and
  alluvium need rows. Each new substance passes S5.4's wiki test; anything valuable — salt and potash in evaporites
  (Salt is already code 17, `strata.rs:33`), bauxite in laterite, placers in fans — is live state (F16; S5.1; V4 15.3).

**K13 THE COVER** (revised; F7, F8, F24, F58).
- **The record (V4).** One object on one cell; its seed = hash(realm seed, cell, kind). THE SKELETON — the trunk and
  main crown capsules — comes from the generator's law (allometry from the seed and the stage parameters) and IS the
  collider (V1 item 1; V4 step 9 (c)–(d); F7). The art fits the skeleton: a variant is chosen and scaled to it now, or
  modules are placed on it (V4 (d)) next (Q14). G-TREE-FIT holds the drawn trunk to the collider. The art never changes
  the static shape, so the private store of bought assets stays outside the generator and its no-drift gates.
- **The placement.** A stratified grid with an ordered (blue-noise) acceptance against the cover share, so the fold
  equals the expectation; the kind by the plant-type shares; the size by allometry; the riparian density from the
  discharge row; clearings from the laws (wet meadows, bare rock, avalanche tracks, Lied & Bakkehøi 1980).
- **SL10.** Acceptance reads the column's rung-0 truth at every distance, so the SET of trees never changes with
  distance. The anchor digest and the skeleton pin join the golden table.
- **THE SEAT (F58).** Each drawn instance sits on the drawn, morphed ground in the vertex stage (a look), so a trunk
  never floats or sinks with a rung's sink (ARITH: up to ~14 m at rung 3 otherwise, T7's one-cell steps 2 + 4 + 8 m). The
  future collider stands on the rung-0 truth.
- **THE DISTANCE CHAIN (V10: three stages, F8).** The full asset → the asset's own coarser meshes, the last one authored
  by the artist (no generated crown hull) → the canopy fold. THE FOLD IS ONE FIELD (F58): the cover share and the stand's
  top height with a gap fraction; its silhouette at the crown envelope (a skyline of crowns is a maximum, not a mean),
  its colour by the shares; never built from the tree list, so it can stand in for a late list. A silhouette gate at the
  crown-to-fold hand-over (≤ 1 display pixel).
- **GRASS** is client-only and never collides (V4 step 9, row R12). Its far share enters the material closure as a grass-cover share,
  so the ground beyond the grass ring keeps the same green (F24). **BOULDERS** collide, and both hosts place them (aprons,
  tors and core stones (Linton 1955), cobbles, erratics). Their far share enters the closure as rock cover; their own
  coarser meshes draw them to under one display pixel, then the tint (F24).
- **PRIORITY (F58).** The cover's jobs sit in the build budget by the time to need, never behind it.
- **SHADOWS (F58).** About 0.29–0.43 M trees stand inside the cascades' 3.5 km at full screen (ARITH: a 67° sector,
  7.2 km², 400–600 stems a hectare). They cast through coarser casters, as the ground's shadow ladder does, priced inside
  line D (§7).
- **ARITH.** At 400–600 stems a hectare a full-screen view holds ~93 000–140 000 trees within 2 km and ~1.8–2.7 M within
  8.7 km: the horizon and the culling decide the frame.

**K14 THE LIGHT** (§5.8): the light sum with V_body, V_near, V_far, G_unres, V_cloud, A_shape, A_contact, A_canopy,
A_unres and E_bounce; law L2 restated (C22); photometric laws Oren–Nayar 1994, Hapke 1981/2012, Wiscombe & Warren 1980,
Kuusk 1991, PROSPECT (Jacquemoud & Baret 1990), Lekner & Dorf 1988, Planck (DOC as papers).

**K15 OTHER BODIES (the same laws)** (revised, F46).
- Each body's expected forms are written only after bench M0-5 prints its charter.
- **The airless moon:** craters at every size with rays and maturity, regolith, aprons, contraction scarps (Watters 2010,
  law owed); no fluvial, aeolian or ice forms and no soil; Hapke light, black shadows, planetshine.
- **A dry world with air:** dunes and yardangs from its own supply. Fans, playas, salt pans and water-cut mesas only
  with a ruled WET PAST: a history (T9 allows one) with its own law and price, for the owner to rule.
- **An icy moon:** ice as rock by a composition law from the bulk density (calibrated on Ganymede, Callisto and Europa),
  creep (Glen's law, owed); built only together with a stand on a real icy body of THE world.
- **A locked world with air:** the zonal temperature law (`climate.rs:409-413`) cannot make a day side and a night side.
  The owed law is a two-hemisphere balance (ESTIMATE 1–2 engineer-weeks), built when such a body is a stand.

**K16 THE SEA FLOOR** (NEW; F36). The sea floor is 78 % of the home planet (DOC B22 item 2). It is drawn dry until the
fill, and G-LOOK comes before the fill (Q18), so its structure is part of the ground step.
- **Laws:** the depth from the plate's age (Parsons & Sclater 1977), the age from the solve's own plate clock — the
  distance to the plate's boundary over its speed (`crates/terrain/src/craters.rs:311-327`); abyssal hills from the
  spreading rate (Goff & Jordan 1988); seamounts (Wessel 2001); a sediment drape by age; submarine canyons on the
  continental slope. All ESTIMATE until read.
- **Layers.** SOLVE: depth against age, seamounts ≥ 16 km. A0: canyons ≥ 2 km. KERNEL: abyssal hills with a zero mean.

**K17 FROST** (NEW; F37). The home planet's land stands at a mean of +1 546 m (MEASURED gen 21, status §4), so a large
share of it (UNMEASURED) may lie between the tree line and the snow line.
- **Laws:** frost cracking in the −3 to −8 °C window (Hales & Roering 2007); block fields; solifluction lobes
  (Matsuoka 2001); rock glaciers under shaded faces; patterned ground. The switch: the band between the tree line and the
  snow line, with the daily swing (K11 A7). All ESTIMATE.
- **Layers.** KERNEL: lobes and patterned ground. OBJECT: blocks. R3: rock glaciers.

**K18 THE OTHER KINDS OF COUNTRY** (NEW; F51). Each with the facts that switch it on and its layer; bench M0-5 states
whether a body of the home system calls for it:
- **badlands:** soft beds and a dry climate, where the hillslope Péclet spacing falls to tens of metres (Perron, Kirchner
  & Dietrich 2009), under R3's floor: a KERNEL channel term;
- **inselbergs and domes:** jointed granite with a deep weathering front (K3 law 6): R3 and PLACED FORM;
- **loess mantles:** downwind of glacial outwash and deserts (the dust supply and the wind): KERNEL and the cover class;
- **barrier coasts and spits:** long-shore drift on gentle coasts with a sand supply: A0 and R3;
- **tower karst:** humid limestone (K10): PLACED FORM.

### 3.3 The prices (T5 and B35 item 2: each family priced before it is built)

Every cell is an ESTIMATE unless marked; each names the bench that replaces it before its phase builds (F12, F52). The
SIZE column is engineer-weeks: a relative price for the order, not a calendar. This project's measured pace is faster
(the temporal path's T0–T2 ran from 2026-09-29 to 2026-10-02, DOC status log), so the calendar is UNMEASURED.

| Family | Phase | Server (once) | Client per chunk | Bytes across a boundary | Frame | Size (eng.-weeks) | Bench |
|---|---|---|---|---|---|---|---|
| Rivers: R2 | 2 | UNMEASURED for the new operators; the old loop's ARITH 14–25 core-h home (r2 §12) as a reference only (F60) | the fine relief under P15: ±0.2 ms | A1 ~9 MB per 150 km approach + A2 2–4 MB (ARITH) | ±10 % vertices near the eye: ±0.7 ms (ARITH on the 6.9 ms vertex work) | 5–8 | M2-1, M2-5 |
| Climate A1–A3, A7, the circulation | 2 | +1.5–8 s a solve | — | A7 +8.87 MB a planet (ARITH) | — | 3.5–5.5 | M0-9, M2-2 |
| Valleys 1–5 km (R3) | 3 | 0 (a); patches (b) 0.3–1.3 cores at speed (ARITH, F34); (c′) ~1 core-min a view (ARITH) | +1.2–4.1 ms; ×2 at rung 8 with the hold | 0 (a, b); (c′) ~1 MB a view, 3.3 GB store (ARITH) | +10–40 % triangles near the eye | 4–7 (with the three-way bench) | M3-1 |
| Slopes, aprons, the soil law | 3–5 | slides < 2 s | ~10 ns a column | 0 | small | 2–3 | M3-1, M5-1 |
| Rock control (outcrop rule, cliff law, joints, folds) | 4 | in the grouped re-bake | ≤ 0.3 ms | A4, A5, A6 | +5–15 % triangles on layered stands | 3–4 | M4-1–M4-4, M4-10 |
| Placed forms (rule S) | 4 | — | a mesh chain per form, a few hundred triangles | 0 | < 0.1 ms | 2–3 | M4-9 |
| Ice (sediment, rebound, moraines, troughs, cirques) | 4 | < 5 s; R2 +< 1 % | — | 0 | — | 1.5–3 | M4-6 |
| Wind (dunes; giant dunes in A0 and R3) | 4 | dune volumes < 10 s | ~0.16 ms in dune fields (ARITH) | 0 (A6 carries the supply) | +10–20 % triangles in dune fields | 2–3 | M4-6 |
| Coasts (forms) | 4 | +2–5 % of the operators on coastal tiles | — | A14 (one charter word) | — | 1–2 | M4-6 |
| Volcanoes | 4 | stamps < 2 s | 0–30 ns a column in fields | new province codes (A4) | small | 1.2–1.7 | M4-6 |
| Crust forms | 4 | — | ~20 ns a column in faulted provinces | 0 | small | 2–4 | M0-7, M4-6 |
| Impacts < 16 km, rays, maturity | 2–4 | R2 moon ~1.2 × 10⁵ craters: seconds | moon rung 0: 1.2 ms without bins, ~0.1 ms with them | 0 | +5–10 % triangles on the moon | 1.5–2.5 | M3-2 |
| Karst, caves, tower karst | 4 | ~7 s (ARITH on an ESTIMATE rate) | 20–40 ns a column | 0 | small | 1–1.5 | M4-6 |
| Sea floor (NEW) | 4 | < 10 s | ~20 ns a column | 0 (the age from the plate clock) | small | 1.5–3 | M4-11 |
| Frost (NEW) | 4–5 | — | 10–30 ns a column | 0 | small | 1–2 | M4-12 |
| Other kinds (badlands, inselbergs, loess, barriers) | 4 | — | 10–40 ns a column each | 0 | small | 3–6 | M0-5, M4-6 |
| 3-D removals with the seal and breach cures | 4 | — | 0.1–1 ms a cliff chunk | 0 | removal faces UNMEASURED | 3–5 | M4-5 |
| Climate A4–A6 (per column) | 5 | — | +20–50 ns a column | 0 | in line C | 1–2 | M4-7 |
| Classifier, soil and snow laws | 5 | — | +0.1–0.3 ms | 0 | — | 2–3 | M5-1 |
| Materials (one source, the vertex shares, the far shares, the paint table) | 5 | the far fold +30–60 core-s | +4–8 B a vertex: +224–448 MB at the walk (ARITH) | A8: far map 17.7–23.6 MB raw a planet + ring parts 31–41 MB at the worst eye (ARITH) | +0.3–0.6 ms GPU | 5–6.5 | M5-1–M5-3 |
| The AAA surface | light | — | — | 0 | +0.8–2.0 ms; 120–475 MB of textures | 4–6 + assets | M5-2 |
| Forest, grass, rocks | 6 | — | scatter ~9 ms per km² per worker | 0 | +2.6–7.3 ms GPU (M0-15 measures it in phase 0) | 11–15 + assets | M0-15, M6-3 |
| The light phase (far march, sky-view at a chunk's write, fade, bias, AO, bounce, adaptation) | light | — | the sky-view factor +0.1–0.4 ms a chunk on the builder | A11 (tile numbers) | +0.3–0.8 ms | 5–8 | ML-1–ML-4 |

**ARITH on these estimates.** The families sum to about 66–105 engineer-weeks. The speed items that §5.3 puts before
phase 3 add about 8–15 (T4's cures 1–2, the builder levers 3–5, T3 and the geometry step 4–8). The water fill, the
weather step and the edit system are not in these sums.

---

## 4. The architecture

### 4.1 The layers

| Layer | Cell | Calculated by (realm, host) | Makes | Its limit |
|---|---|---|---|---|
| SOLVE | 8 192 m nodes (`crates/terrain/src/macro_lattice.rs:48`) | the body's own shard, once a body. Home: 205 s first solve (MEASURED, `docs/design/DEFERRED.md:9137-9139`). Moon: 0.3 s (DOC) | plates, isostasy, belts, rivers ≥ 16 km, the budget's lakes, the ice extent, craters ≥ 16.4 km, the climate rows, the provinces. Phase 2 adds the climate laws; phase 4 adds the cover's thickness, edifices, karst, deltas, moraines, slides and the sea floor's depth by age | nothing under 2 nodes |
| A0 (R2) | 1 024 m, plus its 2 km and 4 km means | the body's own shard, a bake per row tile (524 km) | valleys, troughs, mesas, escarpments ≥ 2 km; craters 1–16 km; floodplains; fans; giant dunes; submarine canyons | nothing under 2 km |
| R3 | features ~0.25–2 km (A0 owns ≥ 2 km, F59) | both hosts (SL10); the mechanism by the bench | tributaries, gullies, hillslope forms, buttes, cirques, fans, rock glaciers | ~the hillslope length |
| KERNEL | the rung's cell | both hosts, per column | the fine relief (P15), the bench, joints, banding, small dunes, scarps, aprons, small craters, cones, dolines, abyssal hills, frost lobes; the column's class and material shares | no iteration, no overhang |
| PLACED FORM (rule S) | the form's own outline | both hosts compute it; the client draws its coarser meshes | pillars, towers, needles, fins, sea stacks, volcanic necks | nothing narrower than its joint spacing |
| VOLUME | the rung's cell, in the surface band | both hosts, per cell | undercuts, arches, sea notches, cave mouths | removal only |
| OBJECT | an art asset over a generated skeleton | both hosts place it and its collider; the client draws its art | trees, boulders, blocks | collides above the step height |
| LOOK | the pixel | the client | textures, light, the blending of the vertex shares | never a shape, never a material decision |

### 4.2 The ladder under R2 and the one-volume model

| Rungs (home) | Cell | Ring (ARITH: 869 m × 2^L) | Z from | Detail on top | Materials from | Shipped as | The outer swap |
|---|---|---|---|---|---|---|---|
| 0–5 | 1–32 m | to 27.8 km | A0 | the fine relief (P15: smooth, zero mean, under 2 048 m); R3; the bench; phase-4 terms; placed forms by their cells; removals at 0–3 | the recipe per column, carried by the vertex | A0 parts + row tiles | one fine-relief octave or R3 component, under one cell (T7) |
| 6–10 | 64 m–1 024 m | 27.8–890 km | A0 | R3 to ~rung 8; compact features as cell means; placed forms by their own meshes | the recipe per column | A0 parts + row tiles | only R3's components, by their own survival |
| 11–12 | 2–4 km | 890–3 560 km | A0's own 2 km and 4 km means (the far fold) | — | the shard's folded shares (A8 ring parts) | A0 coarse means and share parts by ring (A2, A8) | exact |
| 13 | 8 km | 3 560–7 120 km | the rows = A0's node means (the mean hold) | — | A8 ring parts | row tiles to the horizon (one_volume's ask A2) | exact (G-FOLD reads the breach's residue) |
| 14–18 | 16–262 km | > 7 120 km | the far map (mean tops, the wet share, the material shares) | — | the far map | the far map, coarsest first | exact |

- **`L_ref` = rung 10**, named by the FIELD (C6).
- **The far HEIGHT needs no bake.** A0's node means equal the rows by the mean hold (r2 §4.5), so the far map's mean tops
  fold the rows exactly, to the rounding. Only the wet share and the material shares need A0 (F55; §4.4 A8; Q13).
- **The far shares at a swap** are sampled by the shader's own level (B17 Q9's rule), or measured at the swaps (F23).
- **Deleted** with the one-volume model: `field_step_m` (`crates/terrain/src/artifact.rs:942-960`) and the 9 → 10 field
  swap.
- **On an unbaked tile:** rungs 0–12 wait and rung 13 stands (C4); the refinement that follows is the first-visit
  transient (Q7).

### 4.3 Who calculates, who draws (SL3, B35 item 6)

| Product | Calculated by | Host | Drawn by |
|---|---|---|---|
| The solve, the climate rows, the provinces, the cover's thickness, the far map | the body's own shard | its shard | the client, as the body's look |
| A0 and its coarse means; the bake's order | the body's own shard: its own occupants and occupied children first, then the tiles that viewers name (Q7) | its shard, the prebake too (Q8) | the client |
| R3, the kernel terms, the placed forms, the removals, the class, the soil, the material shares, the cover's anchors and skeletons | the body's static shape (seed, address, charter, shipped rows and A0) | both hosts (SL10) | the client |
| The far material shares and their ring parts | the body's own shard, folding the same recipe function | its shard | the client |
| The blending of the shares, textures, shadows, AO, bounce, eclipses, planetshine; the tree art | no realm: the look of what is drawn | the client | the client |
| Grass and decoration | — | the client only (R12) | the client |
| The weather field, its list, its memory | the planet's own shard | its shard | the client: clouds, rain, snow, wet ground |
| The wind's force on a hull | the planet's shard (the parent's physics of its child) | its shard | — |
| The reach | the body, from the brightest look its charter allows | its shard | — |
| A moon's everything | the moon's own shard | the moon's shard and both hosts | the client, as the moon's look |
| The star system | the placement and the attitude ONLY (and, if Q17 says yes, the planet's orbit elements once) | the system's shard | never draws a child |

### 4.4 What crosses a realm boundary (SL6: default NO; each item is an ask with its size and its arrival time)

Arrival times are ARITH at the gateway's pace, 1 MiB a half-second beat for each client, about 2.1 MB/s
(`crates/connection-plane/src/gateway/artifact.rs:41-50`).

| # | Data | From → to | Size and arrival | Why the receiver cannot compute it | Without it | State |
|---|---|---|---|---|---|---|
| A1 | A0 parts, 128² cells, 32 KiB each | the body's shard → the gateway → its viewers' clients: `BulkMsg::ArtifactAmp`, discriminant 7, the next free minor | ~9 MB per 150 km approach, ~4.3 s; ~4.7 MB at the reference stand, ~2.2 s; 1.06 GiB a planet in the store (ARITH r2 §12) | the CPU fallback costs ~81 core-min a view (DOC r2 §12) | no valleys under 16 km | ASK (Q5) |
| A2 | A0's 2 km and 4 km means for rungs 11–12 | the same arm | 2–4 MB at 150 km; ≤ 7.3 MB at the worst eye (1.85 M + 1.81 M cells × 2 B), ≤ 3.5 s | the client holds no A0 past 890 km | a shape step at 10 → 11 | ASK (Q6) |
| A3 | the row's water word as the level map | rows | 0 B | — | — | ruled (B22 Q6) |
| A4 | new province codes (arc, plume, flood basalt, basin, evaporite) | rows | 0 B; the recipe's rows 8 → 16 | the solve's crust fields are transient | no volcanic or red-bed provinces | ASK (phase 4) |
| A5 | strike and dip or fold class | rows | 0 B, but see C11 | the plates are a float draw; a port moves every plate | horizontal beds everywhere | ASK + the bit budget (phase 4) |
| A6 | the cover's thickness, a log class | rows | +1 B a node = 8.87 MB a planet; ≤ 1.49 MB at the worst eye (ARITH) | the deposit and the subsidence are the solve's state | no outcrop rule: rock borders on 8 km squares (F35); no sand supply for dunes (F39) | ASK (phase 4): NEEDED by K3 law 7 |
| A7 | the seasonal amplitude | rows | +1 B a node = 8.87 MB a planet (ARITH) | the global harmonic solve | no continental winters | ASK (Q4, phase 2) |
| A8 | the far material shares (rungs 14–18) and ring parts (rungs 11–13) | the far map; the A0 arm | K = the material count less one = 6–8 B a cell (bare rock, scree, soil, grass, canopy, sand, alluvium, snow and firn, ice; F55): far map 17.7–23.6 MB raw a planet (2 954 424 cells); ring parts 31–41 MB at the worst eye (5.15 M cells); together 23–31 s at the worst eye (all ARITH); compressed UNMEASURED | the shares fold rung-10 cells, which the client lacks far away | no far snow, forest or rock colour, or a colour step at 890 km | ASK (Q13), sizes measured first (M5-3). A lever: the mean albedo and roughness (4 B) where the light is linear |
| A9 | the weather list | the planet's shard → the gateway → every session whose picture holds the planet: a want and a statement, appended arms in the artifact's realm-audience pattern | N × 16 B a statement, one each weather step (10 s, DOC 05 §8.7) and at most one a second (DOC 05 C-20); N by the law (§8.2) | live state | the almanac only | ASK at the weather step (Q17); no realm-to-realm crossing |
| A10 | the weather memory (snow water, wetness) | by digest on the same realm audience | 192–768 KiB (ARITH on 05's grid); sparse size UNMEASURED | live history | no lying snow, no wet ground | ASK at the weather step |
| A11 | the TILE-NUMBER want: the tiles that a client's own ladder names, for the bake's order, the serve and the light's sunward strip | the client → the gateway → the body's shard: an appended arm | ~4 B a tile; a few hundred bytes a request (ARITH: ~50 tiles) | — (it replaces the eye's view as an order; no pose crosses) | the eye's view orders the bake (F4); no sunward strip (F5) | ASK (Q7, Q15) |
| A12 | the planet's orbit elements and spin law | the star system → the planet, ONCE on change (an extension of V13's A2) | a few dozen bytes | SL1 clause 3: the planet may not derive its own placement | no season or day in the weather's forcing and melt | ASK at the weather step (Q17), with the owner's ruling on T3 class 1 against SL1 clause 3 |
| A13 | R3's option (c′): a 256 m baked level on land | the A0 arm | ~1 MB a view; 3.3 GB a planet in the store (ARITH, F34) | the bake | — | ASK only if the bench chooses it (Q11) |
| A14 | the tidal range | the charter (`TAG_CHARTER`, `crates/core/src/look.rs:81`) | one word | the planet computes it from its own children's stated masses and the orbits it authors, plus the star's term from V13 A2's facts; the client cannot | no wave-cut platform width | ASK (phase 4) |
| — | Row growth | — | A6 and A7 take the row from 10 to 12 B (`ROW_BYTES`, `crates/terrain/src/artifact.rs:101`) and the tile from 40 960 to 49 152 B; an `ARTIFACT_VERSION` bump | — | — | stated |

**The worst eye, all new data together** (ARITH, F66): A0 ~9 MB + A2 ≤ 7.3 MB + A8 49–65 MB = 65–81 MB, about 31–39 s,
over B-LOGIN's 30 s. So the gateway serves by the time to need (lever E12): the far map coarsest first, and ring parts
only for the rungs that the session draws. M5-3 measures the compressed sizes before Q13's build.

---

## 5. The phases, in order

Every phase ends with: the census numbers on the pinned stands of all three bodies; the stands' pictures, captioned "no
far shadow" until the light phase; the living status file updated in the same change; the DEFERRED.md rows flipped as
they land and the new interims registered (§9.4); the frame, build and memory lines (§7). One job at a time runs on the
machine.

### 5.0 Phase 0: this design and its benches (no world change)

- **Scope.** This document with its reviews. The benches M0-1 to M0-17 (§9.2). The stands rule run once and pinned.
- **Owner decisions:** Q1–Q18 (§10.1).
- **Data.** None crosses. The Earth, Moon and Mars tiles stay outside the repository (Q2).
- **Output:** the readings of today's generator-22 world on all pinned stands (the shore law off, so the numbers can move;
  UNMEASURED); the bodies' charter table; the stands census; the climate census of today's code beside the corrected
  circulation's bench; the 1 000 km coast reading; the clean full-screen frame, the CPU split and the memory (T4's
  reading); the turning-shadow bench; the forest bench with stand-in trees; the eye's rule at full screen on today's
  swaps; the baseline of B-BUILD and B-RATE on the low fast legs.

### 5.1 Phase 1: the flat directions (B35; B9 item 2; B22 Q5)

- **Scope.**
  - (a) Barnes, Lehman & Mulla 2014's two fields combined as one potential, in metres of chord; the step follows −∇Φ
    through Rho8's seeded split (the 2:1 weighting ESTIMATE until read). Measured one change at a time against the status file's variant V3
    and today (status §11 item 11).
  - (b) The zero-mean law adopted as a gate (G-MEAN) that every fine term must pass from phase 2. No ridged centring is
    built (§3.1; F48; Q1).
- **Code.** `crates/terrain/src/lakes.rs:362-560` (`flats`, `spread`, `flat_step`, `split`); the routing's flats in
  `crates/terrain/src/solve.rs` as a bench variant only.
- **Data.** None crosses. The macro rows move with the flat law, so the artifact digest changes: a re-solve (205 s first
  solve, MEASURED) and the pins re-recorded.
- **Costs.** Server: +1 Dijkstra (< 2 s of 205 s, ESTIMATE). Client: 0. Frame: ~0 (ESTIMATE: no drawn vertex count moves;
  FR5 prints it).
- **Checks.** G-GRID-BY-CLASS, G-LAKE-WALK, G-LAKE-SHARE (a reading; 4.828 % today), G-RIVER, G-BASIN; Tarboton's cone
  must not move (W13.3).
- **Pictures and the owner's look.** The pinned stands of all three bodies and V-ref. The frozen references change only on
  the owner's word (the B29/B32 pattern).
- **Risks.** The lake share and the budget move with the flat law.

### 5.2 Phase 2: the climate laws, then R2 under the later rulings

- **The order** (B8 item 3; r2 §16; F33):
  1. **Slice 0, the bench (M2-1).** R1's loop as the control; one change at a time; the new operators (the implicit
     stream power on D∞'s graph, the transport law, the steady state); the fine relief under P15 and G-MEAN; S beside A0
     on every line; every line read through the shipped kernel at 128 m columns; THE BAKE'S RATE: ns a cell an
     iteration, the iterations to G-DOSE, one tile's wall time on one core (F60). Then a HILLSHADE at 100–150 km for the
     owner's look, on all three bodies.
  2. **The macro change, one change at a time (M2-2):** D∞ in the sweep; one water (the runoff) with K₀'s water
     normalised (F49); the climate laws A1–A3, A7 and the circulation (Q3, Q4); each alone, then together: the land share,
     the sea, G-HYPSO, the solve's time, and the climate census on the corrected law.
  3. **The bake service:** the warm-up order (Q7), one budget for each host, the prebake by the shard itself (Q8), idle
     during every frame reading.
  4. **The kernel, the ladder (§4.2) and the fine relief.**
  5. **The ship and the client** (A1, A2; A11 if Q7 says yes).
  6. **The gates**, including G-GRID on A0 (status step 5) and the 10 → 11 height AND shade step (r2 skeptic 23).
- **The changes to `r2_design.md`** (the draft's 13, updated by the reviews):
  1. the one-volume model: A0 is Z; no water word in the kernel; the terminals read the level map;
  2. no caps (§6 rows 10–15, 48);
  3. D∞ in the bake and in the macro sweep; the shipped tree keeps one receiver;
  4. one water — the runoff, with K₀'s water normalised to Earth's runoff — for the cut at both levels, measured alone
     (it moves the land share; W7's history reached 1.4 % land);
  5. the bench first, with the new operators' rate measured (F60);
  6. no temporary ground (C4); the bake as the realm's warm-up (Q7);
  7. the far view exact at every swap (§4.2, C6);
  8. SL3: each body's own shard bakes, the moon's too; the prebake by the shard (Q8);
  9. the arm at the next free minor;
  10. the only field swap at 10 → 11, exact;
  11. one bake budget for each host;
  12. **the fine relief** (the word "skin" retires for it; F25): smooth, zero mean (G-MEAN), under 2 048 m, its gradient
      under A0's drained slope everywhere (P15) and so zero on channels and floodplains; below the hillslope length the
      forms come from the hillslope law (R3), not from noise; no ladder clip; the placeholder retired under a field; the
      Earth stand's spectrum is a READING, never a target (F31);
  13. r2 §1's stale lines corrected (§9.3).
  - Added: the erodibility reads the bed's own rock at the cell's height from the first bake (C3).
- **The operators under B23:**
  - routing by D∞ (Tarboton 1997), its facet angle from an integer table (P13);
  - the implicit stream power on D∞'s graph with n = 1, m = ½ and the macro's K₀. For n = 1 it is exact in one ordered
    sweep (ARITH): z′ᵢ = (zᵢ + ΔtK√Aᵢ·Σⱼ wᵢⱼz′ⱼ/Lᵢⱼ) / (1 + ΔtK√Aᵢ·Σⱼ wᵢⱼ/Lᵢⱼ). No a_max, s_max, slope hold, spe_max or
    incision floor (`crates/recipe/src/amplify.rs:233-261`, `:286-288`);
  - the sill floor and the base level stay as laws (G-SILL);
  - the talus by the macro's repose (debris);
  - deposition by a transport law with a calibration body — Davy & Lague 2009 (ξ–q) or Yuan et al. 2019 (ESTIMATE
    sources) — which answers r2 skeptic 3;
  - the dose: steady state with the macro field (~7.7 Myr response, DOC r2 §4.4), tested by G-DOSE (2T moves < 5 % of
    the cut);
  - D_max removed. The iterations to steady state are UNMEASURED, so the bake's cost can rise or fall (F60).
- **Code.** `crates/recipe/src/amplify.rs` wired; a new `crates/terrain/src/amp.rs` (`bake_tile`);
  `crates/terrain/src/artifact.rs:810-833` → a `FieldLadder`; `crates/terrain/src/solve.rs:1093-1211` (runoff, D∞); a
  `BakeWorker` in the body's shard (r2 §6.4) as a `ShardProfile` capability with its G-IDENTICAL fixture (F10);
  `crates/wire/src/channels.rs`, `version.rs`; the gateway's byte pace into the transport tuning; the client's book with
  byte bounds; the gateway's cache bounded ONCE for each gateway, not for each realm (F62).
- **Data.** A1, A2, A7 (and A11 if ruled).
- **Costs.**
  - The bake: UNMEASURED for the new operators. The old loop's ARITH (14–25 core-hours home; 1.8 h on 14 cores; r2 §12)
    stands only as a reference until M2-1 prints the new rate (F60).
  - A first view of an unbaked planet: 8–15 min at a quarter of an 8-core host by the old loop (DOC r2 skeptic 16);
    UNMEASURED for the new one (M2-5).
  - The macro re-solve with D∞, the runoff and the climate laws: UNMEASURED.
  - Frame: ±10 % vertices near the eye (ESTIMATE), ±0.7 ms (ARITH on the 6.9 ms vertex work); B-BUILD +0.2 ms a chunk
    (ESTIMATE).
- **Checks.** G-HACK, basins, G-DRAIN-DRAWN, G-SPECTRUM, G-MEAN, G-FOLD, G-SILL, G-AMP-HALO, G-DOSE, G-AMP-PIN on both
  targets, G-AMP-ORDER and G-AMP-PATH, the 10 → 11 height and shade step, G-FAR-EXACT, G-GRID on A0, the skyline report,
  P7a and P7c, B-BUILD, B-RATE, the frame total (a reading until §7 FR1 makes it a gate).
- **The owner's look.** Twice: the hillshade, then the window, on all three bodies.
- **Risks.** The washboard lives in the rows (1 004 m at 45°, MEASURED B11) until the basins step after R2 cures it. The
  macro change moves the sea, the deserts and the ice. A0 may grow its own bump at 2–4 km. Every macro change means a
  re-bake. Dry interiors stay smooth (r2 risk 12).

### 5.3 Before phase 3: the speed items (a proposal; the owner decides where they go, D15)

D15 leaves the geometry step's place to the owner: *"a step of its own, before or inside the ground step"*
(`docs/investigation/2026-09-28/temporal_path_design.md:1533-1535`). This design proposes, before phase 3:
- **T4's cures** on the clean reading of phase 0: every pipeline warmed before play, the boarding's frame, the
  full-screen switch (B34 item 4).
- **The builder levers E1–E6 and E10** (one_volume §10A.3; owed at `docs/design/DEFERRED.md:11739-11741`): no world
  change. ARITH (one_volume §10A.3–10A.4): 790–1 330 chunks a second on 7 workers and 520–760 on 4, against an ask of
  525–1 030 at 528 m/s and 537 m.
- **T3, the one geometry pass, and the geometry step's own design with a priced cut** (occlusion culling, the vertex
  bytes): about 6.9 ms of the ground's 8.44 ms pass at s = 0.5 is vertex work (ARITH on MEASURED,
  `docs/design/DEFERRED.md:10922-10925`), and the model's line A needs about four fifths of the fixed price removed
  (`temporal_path_design.md:1043`).
- **T5 steps 1–3** (the cheaper air), which line A's model assumes.
- **Why before phase 3.** Phase 3 adds +1.2–4.1 ms a chunk and +10–40 % triangles near the eye (ESTIMATE). Read on a
  pipeline that changes later, its lines would be wrong (F61); without the levers the late chunks grow (F54).
- **Not the waves' cure.** The build-ahead along the motion and the early bound stay after G-LOOK (B34 item 7). The
  levers are B22 step 2's owed speed items.

### 5.4 Phase 3: the valleys of 1–5 km, the slopes, the small craters

- **Scope.** Valleys 1–5 km apart and 200–800 m deep; sub-ridges and gullies of 200–500 m; hillslope forms; talus aprons;
  craters under the node on every body (DOC 01 §2, §3.4; status §9).
- **Laws.** Valley spacing by the Péclet balance (Perron, Kirchner & Dietrich 2009); channel heads (Montgomery &
  Dietrich); Horton's ratios and Hack's law; Flint's profile S = k_s·A^−θ, θ = ½; the hillslope law (K2); the erosion rate
  E = K·k_s from A0's channels; every rock read of K3. ESTIMATE sources.
- **The mechanism (Q11; F34).** Bench M3-1 compares three options against Earth, on the same stands, on the CPU path after
  the levers (F54):
  - (a) a closed-form network under A0 on both hosts (Gaillard et al. 2019's Dendry structure, ESTIMATE): no data,
    0.3–1 µs a column (ESTIMATE);
  - (b) patch erosion at 128 m with A0's discharge as the boundary at the patch edges: 2.8 s for each 16 km patch on one
    core (ARITH), so 0.3–1.3 cores at 528 m/s (ARITH, F34), and the server pays it again for collision;
  - (c′) a second baked level at 256 m on LAND only: 3.3 GB a planet in the store, ~1 MB a view to send, about one
    core-minute a view by the old loop's rate (ARITH, F34; A13).
  - The draft's option (c), a 128 m level over the whole globe (72.7 GB), stays rejected. No option is recommended before
    G-HACK, G-JOIN, G-HILLSLOPE, G-SPECTRUM, the hold's residual and the cost a chunk read all three.
- **The hold (F59).** R3 has a zero mean over each A0 cell BY CONSTRUCTION (each carve balanced inside its cell), or an
  exact integral; never a 16-sample mean, which misses by 17–67 m a cell (ARITH on an ESTIMATE spread) and steps at every
  cell edge. G-MEAN prints the residual at R3's swap, and G-GRID reads R3's own field.
- **The bands (F59).** A0 owns ≥ 2 km (its Nyquist is 2 048 m, r2 §7.4). R3 owns < 2 km, at the depth that its swap rung
  allows (T7). The 1–2 km band is R3's.
- **Survival (P3).** Example: a 150 m deep valley 1 km apart moves rung 6's switch from 55.6 to 65 km (ARITH).
- **Costs.** Client +1.2–4.1 ms a chunk (ESTIMATE) on today's column work, which is UNMEASURED for generator 22 (the
  839 ns a column was generator 17's, DOC one_volume §10A.1.1 H2). Builds of 11–22 ms on a moving eye (DOC one_volume §0.8)
  against V10's 8 ms: THE MAIN RISK; B-BUILD and B-RATE are this phase's red lines. Triangles near the eye +10–40 %
  (ESTIMATE), read on both pipelines (FR5).
- **The card.** Integer kernels in `vd-recipe`; `gpu-drift` extends to the new kernels. The card builds no solved chunk
  today, so the CPU path must meet the budget alone (Q10).
- **Checks.** G-HILLSLOPE, G-JOIN, G-FLINT, G-SPECTRUM at rungs 0–5, G-MEAN, G-DRAIN-DRAWN, G-SFD on the moon, the
  skyline, B-BUILD, B-RATE, the frame total (a gate from here, §7 FR1).

### 5.5 Phase 4: the processes

- **The order inside the phase:**
  1. ONE grouped macro change, measured alone, then ONE re-bake of A0: the provinces with the cover's thickness (the
     outcrop rule, A6), the contacts and the dip; volcanic edifices; karst denudation; glacial sediment and rebound with
     the cap removed; deltas; large slides; the composition law of the check bodies, the moon's crust included; the sea
     floor's depth by age; the envelope's draw and band term removed (§6).
  2. Then the kernel families, one at a time, each with its census and its stands: rock control (joints, banding, the
     cliff law); the placed forms (rule S, with the silhouette bench M4-9); slopes; dunes (the giant ones in A0 and R3);
     cones, necks and maars; dolines and tower karst; scarps; small craters with rays and maturity; coastal forms; the sea
     floor's abyssal hills; frost; the other kinds (K18).
  3. The 3-D removals.
- **The 3-D removals.** Capsules placed for each region by laws: an undercut at a soft bed under a hard one; an arch where
  two undercuts meet in a fin; a sea notch at the base level; a dead-end cave mouth where a conduit meets a face — all with
  the tube carver (`crates/recipe/src/cell.rs:173-265`). Two defects are cured in the same change:
  - THE SEAL: the client draws every closed hollow under rock in its column as rock (`crates/client/src/chunks.rs:1219-1279`,
    used at `:1316-1318`); it must skip cells inside an OPEN primitive.
  - THE LATERAL BREACH: a cavern carves by the depth under its OWN column (`cell.rs:353-355`; the band 8–30 m drawn,
    `crates/terrain/src/body.rs:1151-1176`), so on a tall face the client would draw closed rock where the shard's cells
    hold an opening. The cavern must read the lowest surface within its reach.
  - Stopping: drawn true while the feature is at least 2 cells wide; at its last rung it shrinks to ≤ 2 cells, gated at
    p99 ≤ 1 ladder pixel (F24); the far fold never sees removals. Underside vertices morph and take the coarser shade.
- **The crease.** Dual contouring benched at rung 0 AND at rungs 5–9, outside the game path (M4-9; F28). ARITH: a 0.5 m
  bevel is 43 ladder pixels (131 display pixels) at 10 m, and one display pixel at ~1.3 km. Decided on the pictures at the mesa and cliff stands (V10).
- **Rock objects.** Shape above the step height (0.5 m ESTIMATE, the owner's number); cover below it.
- **Costs.** §3.3. Frame +5–15 % triangles on layered stands (ESTIMATE).
- **Checks.** §2.1 rows 3 and 5; G-SILHOUETTE; G-OPEN = 0 against the recipe's rung-0 cells; G-TERM-STEP for each family;
  G-BORDER-CONTOUR; G-SEAFLOOR; G-FROST; B-BUILD; B-RATE; the frame total.
- **Risks.** The largest world change of the step; the bit budget (C11); cliff faces and placed forms raise triangles;
  the chunk budget.

### 5.6 Phase 5: the biomes and the ground's materials

- **5a, identity.** The climate read per column (A4–A6), the 19-class classifier, the soil law (K2), the snow laws
  (K12), all in the recipe in fenced integers (F1). A `GENERATOR_VERSION` bump with new pins. The drawn 1–4 / 2–8 /
  20–80 m depths go (`body.rs:1123-1126`).
- **5b, the seam (V14, kept).** The recipe computes each column's expected shares under the dropped spread
  (Σ = the realized spread of the dropped octaves, from a per-kind constant measured like B14's C constants (M0-14),
  × m²; pinned and derived again by a test, like G-MU); the vertex carries them (4–8 B); the shader only blends them by
  the vertex weights and the morph (`crates/client-render/src/ladder_fade.wgsl:205-220`). No shader derives a threshold
  (F1).
- **5c, the paint table (V13 L24).** Each material's albedo, roughness and specular level from published spectra
  (ECOSTRESS/ASTER and USGS splib07, ESTIMATE source plan) under CIE 1931 and the star's own light. Each later texture's
  mean must match its paint value within one 8-bit level.
- **5d, the far material fold (A8):** shares, never colours, folded by the shard from the same recipe function; ring
  parts for rungs 11–13. The source of an unbaked body's far shares (Q13; F55): the home system's bodies are prebaked;
  any other body shows the rows' fold until its whole bake lands, then a ruled ramp.
- **Costs.** §3.3. GPU memory +224–448 MB for the vertex shares at the walk (ARITH).
- **Checks.** G-BIOME-RUNG, G-PAINT-AGREES (two compilations of one source), G-CLOSURE (1 %, ring means and eye
  windows), G-FAR-MATERIAL (1/255), G-TREELINE, G-SNOWLINE, G-SNOW-SLOPE, G-BARE-SLOPE, G-BORDER, the eye's rule on
  material swaps, B-MEM.
- **Risks.** The closure's accuracy (M5-1); the far shares' size and arrival (M5-3).

### 5.7 Phase 6: the cover

- **The order.** 6a the scatter, the skeletons and the anchor digest → 6b trees → 6c grass → 6d rocks and debris
  (avalanche tracks optional).
- **Preconditions.** The geometry step (§5.3) if the owner put it before; the forest's line from bench M0-15 (phase 0);
  the tree horizon set on pictures. As a start the horizon sits where a 20 m tree is 8 display pixels tall: 6.5 km
  (ARITH: 20 m ÷ (8 × 0.382 mrad)).
- **Rules.** Nothing per tree on the main thread or as an entity: a chunk's cover is ONE buffer that a worker builds,
  culled and drawn indirectly on the GPU. A late list stands on the canopy fold (one field, §3.2 K13) and crossfades in
  over the ramp. Hashed alpha (Wyman & McGuire 2017), because alpha-to-coverage needs MSAA. The leaf floor law: nothing
  thinner than a render pixel is drawn as geometry. Each instance sits on the drawn, morphed ground. The trees cast
  through coarser casters. The cover's jobs sit in the build budget by the time to need.
- **Checks.** G-CANOPY-NEST, G-FOREST-FOLD with its silhouette gate, G-WOODY, G-TREE-FIT, the forest leg under the eye's
  rule, foliage shimmer under the reconstruction, B-RENDER-CPU, B-MEM, B-BUILD, the frame total.

### 5.8 The light and the surface's look (B35 item 4; amends B6.5)

- **Five laws:**
  - L1, the light nests (Oren–Nayar and Smith shadowing over the dropped slope variance that the body states for each
    rung);
  - **L2, RESTATED (F57):** a shadow comes from the same field as the drawn surface; the casters stand at the drawn rung
    or one rung coarser, as the owner's accepted shadow ladder does (`crates/client-render/src/terrain.rs:303-313`;
    `docs/design/DEFERRED.md:8166-8172`: 29 → 56 frames a second at the ground stand, MEASURED then); G-SHADOW-SWAP
    holds the shadow edge's error in display pixels. Casting from the drawn surface itself would multiply today's 1.83 ms
    casting pass by up to about four (ARITH), so it is priced, not assumed;
  - L3, no shade reads where the eye LOOKS;
  - L4, every photometric number comes from a law;
  - L5, the look goes into pixels: the vertex work does not shrink with s. MEASURED: of the ground pass's 8.44 ms at
    s = 0.5, ~6.9 ms is vertex work (ARITH, `DEFERRED.md:10922-10925`).
- **The far shadow, H1.**
  - Keep the cascades to ~3.5 km.
  - Beyond them, march toward the sun over a height pyramid that each drawn chunk writes once, from the SAME morphed
    heights and band weights as the drawn ground; a late chunk's write ramps in (F57). The far map serves rungs 14–18.
  - **Cache horizon ANGLES, not visibility (F57).** V_far is the share of the sun's disc over the cached horizon angle in
    the sun's azimuth, so it follows the sun's elevation every frame with no step. Only the azimuth's change runs the march
    again, with a crossfade over its period. ARITH: a quarter-disc step (0.12°) takes ~1.5 min at 0.083° a minute
    (`crates/terrain/src/home.rs:113`); a cached visibility would step by 0.25 inside a penumbra 198 m wide at 12° and
    1 126 m wide at 5° (h = 1 km), against B31's half a level.
  - The sunward strip: the client names the sunward tiles by number (A11); no direction reaches the planet (F5).
  - H2 (an analytic march on the GPU) is refused: a second generator outside rust-gpu breaks SL10 and L2.
- **Cures in the near shadow.**
  - The cascade is chosen by VIEW DEPTH and returns lit past the last one with no fade
    (`vendor/bevy_pbr/src/render/shadows.wgsl:117-126`, `:192-198`). Cure: a radial fade that ends inside the last
    cascade's far bound — a ninth vendored change, or the ground's own light function — after bench M0-12 (Q15).
  - The snap grid moves with the eye (T6).
  - The bias cap tan = 8, which stops growing under a 7.13° sun (`terrain.rs:117-122`, `:4005`), becomes a
    receiver-plane bias (Isidoro 2006, DOC).
- **The sky-view factor (F67)** comes from the same pyramid, computed for each texel when a chunk writes it (N = 8–16
  directions), so its cost moves to the chunk's arrival: +0.1–0.4 ms a chunk on the builder (ESTIMATE), counted in
  B-BUILD. The other home, each render pixel each frame, costs about N times the far march: 2.4–12.8 ms (ARITH on line E's
  0.3–0.8 ms), and is refused.
- **Also in this phase:** contact AO after MSAA retires (the temporal design §4.6: at s < 1 its passes read the render
  size's viewport, patch seven's rule); one bounce; Beer–Lambert under the canopy; the AAA surface; the eye's adaptation (Pattanaik et al. 2000, DOC); the
  wetness w and the snow share s as material inputs from day one.
- **Costs.** +0.3–0.8 ms (ESTIMATE). Textures in line C.
- **Checks.** G-SHADE-NEST on eye windows, G-SHADOW-SWAP, G-SHADOW-HANDOVER, G-TURN-SHADE, G-FAR-SHADOW-STILL, G-ADAPT,
  G-TILING, the low-sun legs.

### 5.9 Phase 7: the owner's look (G-LOOK)

All pinned stands of the three bodies, all framings with both suns, the flights and the window, against the three
references' quality. G-LOOK's scope and the status file's life are asked (Q18): the recommendation is that G-LOOK judges
qualities 1–5, 7 and the light without clouds and with the dry sea floor, and that a second look after the weather step
judges the clouds and the water before the status file is deleted (amends B6 item 4).

### 5.10 After the ground step: what it leaves ready

- **The waves (B34 item 7).** The ground step does not cure them, and it must not make the cure harder: every shape phase
  prints B-BUILD and B-RATE on the low legs at 240 and 528 m/s, and a phase that raises the late chunks over its line
  stops (§7 FR1). The earlier draft's sentence "no ground item lengthens a chunk's critical path" was false and is
  withdrawn: phases 3–5 add about 1.4–6.5 ms to each chunk (ARITH on ESTIMATE, F54).
- **The water fill (B22 step 4).** The solid pass has no water word; the solve's water calculation stays; the far map
  keeps the wet-share slot; the material model reads w; Ocean and SeaIce wait there.
- **The weather step:** §8.
- **The edit system (B22 step 5).** V14's per-vertex stratum is the seam for edits too; removals stay dead ends until the
  edit step; the authored override (T1 item 6) is not built (status §3) and stays owed.

---

## 6. The caps list (B23): every clamp, ceiling and floor on this step's path

| # | Cap | Where | Fate |
|---|---|---|---|
| 1 | The relief DRAW in [0.5, 1.0) of the law's cap | `crates/terrain/src/body.rs:950-966` | GOES (T9): the cap is the law's value; uplift and erosion decide the relief. Phase 4 |
| 2 | The envelope: every node clamped to ±min(drawn relief, the ladder band) | `crates/terrain/src/solve.rs:1588-1604`, `:1973-1978` | The band term GOES. The crust-strength cap STAYS as a law-limit (Everest calibration, 243.1 MPa, `body.rs:429`). Hits counted for each pass. On the moon the band binds today (14 487 m drawn against a 12 385 m band, DOC) |
| 3 | The crater clip; the uplift clamps | `solve.rs:1746-1756`; `crates/terrain/src/land.rs:1056` | The same law-limit as row 2 |
| 4 | The placeholder roughness, floor `M_MIN` 0.06, 20–120 km | `body.rs:310-325`, `:1104-1111`; `crates/recipe/src/height.rs:141-160` | GOES under a field (a band for a law, T9). Phase 2 |
| 5 | The slope share's min(1, ·) and its per-axis clamp | `crates/terrain/src/artifact.rs:1827-1839` | GOES with row 4 |
| 6 | m ≤ 1; the band sized at m = 1 | `height.rs:109-111`; `body.rs:1271-1290` | REPLACED by the drainage law P15 (the fine relief's gradient under A0's drained slope); the band follows that law's own maximum (F31) |
| 7 | The spectrum anchored at RMS slope 0.70 | `body.rs:79-85`, `:806-818` | REPLACED: the fine relief's amplitude by P15 and the hillslope law; the Earth stand's spectrum is a READING (F31). Phase 2 |
| 8 | fine_scale = min(1, relief / fine_sum) | `body.rs:1035` | Becomes the relief law's bound on the fine relief; it never binds on the home planet |
| 9 | The coarsest wave in [20, 400] km; k_rough drawn | `body.rs:967-972` | Inert under a field; retires when every solid body holds one |
| 10 | R2's ladder clip on the fine relief | r2 §7.4 | REFUSED (T7 rules 2–3) |
| 11 | R2's a_max, s_max, slope power ≤ 1, spe_max | `crates/recipe/src/amplify.rs:233-261` | GO (the implicit scheme on D∞) |
| 12 | R2's incision floor | `amplify.rs:282-288` | GOES |
| 13 | R2's deposit constants 1.0 and 0.1 | `amplify.rs:338-356` | A transport law replaces them (Davy & Lague 2009 or Yuan et al. 2019, ESTIMATE sources) |
| 14 | R2's D_max 768 m | r2 §4.7, §7.3 | GOES |
| 15 | R2's dose N = 128 | r2 §4.4 | Steady state with the macro field; G-DOSE |
| 16 | Repose 35° dry / 25° wet | `solve.rs:104-108`, `:1644-1650` | STAYS for DEBRIS only (scree, talus). Soil slopes: S_c (K2). Intact rock: the cliff law (K3 law 2) |
| 17 | Crater floor 2 nodes (16.4 km) | `crates/terrain/src/craters.rs:104` | STAYS as the lattice's resolution; smaller craters come in A0 and the kernel |
| 18 | Saturation 5 % | `craters.rs:92` | STAYS (Hartmann 1984, lunar highlands) |
| 19 | Crater ≤ the body's radius | `craters.rs:106` | STAYS (geometry; the largest lunar basin) |
| 20 | Rain [1, 10 000] mm/yr | `solve.rs:56-60`; `crates/terrain/src/climate.rs:794` | REPLACED by the moisture budget A2, in PHASE 2 (F33); P ≥ 0 is physics |
| 21 | Lift [0.1, 4.0] | `climate.rs:129-130`, `:406` | REPLACED by A2, in phase 2 |
| 22 | Hadley cells ≤ 6 | `climate.rs:106`, `:333-345` | GOES (A3), in phase 2 |
| 23 | Temperature class 173–428 K | `artifact.rs:107-108` | A per-body offset from the charter (representation only) |
| 24 | Rain byte top at 211 (10 000 mm) | `artifact.rs:123-129` | Follows row 20; the byte holds ~63 000 mm at 255 (ARITH) |
| 25 | One seasonal column for ocean and land | `climate.rs:445-470` | REPLACED by A1, in phase 2 |
| 26 | Freeboard factor [0.5, 3.0] | `land.rs:136-147` | REPLACED (F21): the range becomes the crust law's own bounds — the thinnest crust that floats a platform and the thickest that the crust's strength holds (the 243.1 MPa law-limit) — named as law-limits; a body outside them is a water world or a desert world by that law. Phase 4's grouped change; until then the ends are ASSUMED and their hits are printed |
| 27 | Plates ≤ 40 | `land.rs:115`, `:262` | Becomes a reading; the plate law (R / T_e) decides |
| 28 | Trench, arc, ridge and rift shares; a transform makes no form | `land.rs:128-135`, `:474-476` | Laws owed (K7, K8) |
| 29 | Continental share 0.40 on every body | `land.rs:88-91` | Law owed (crust differentiation). On the moon it makes sediment provinces |
| 30 | Macro edge ceiling 2 048 | `crates/terrain/src/macro_lattice.rs:56` | STAYS as a named cost knob; it never clips the land |
| 31 | The ice cut ≤ the ice's thickness | `solve.rs:1523-1524` | GOES; a time-step stability bound only if a scheme needs one |
| 32 | The bench strength solved from the ladder (0.560) | `body.rs:1210-1236` | REPLACED by the hardness contrast (K3 law 3), measured (M4-3) |
| 33 | The strength ceiling 1 − M_MIN | `body.rs:384-396` | STAYS as the terrace's fold-over limit; past it the cliff law and the volume act |
| 34 | Cap share ½ ASSUMED | `crates/recipe/src/terrace.rs:131` | Law owed (K3 law 4) |
| 35 | Soil and sediment depths drawn (1–4, 2–8, 20–80 m) | `body.rs:1123-1126` | The soil law (K2); the stratigraphy |
| 36 | Bedrock drawn uniformly from 5 silicates | `body.rs:1127-1129` | The composition law |
| 37 | Biome noises, the literal thresholds, highland at 0.55 × relief | `body.rs:1132-1150`; `height.rs:399-455` | The classifier (phase 5, K12) |
| 38 | Cave band and parameters drawn | `body.rs:1151-1176` | Karst and lava-tube laws (K10) |
| 39 | Crust density share 0.50774 ASSUMED | `body.rs:440` | The owner's word owed (T8 ask 3); stays named |
| 40 | Shape bound 0.077 R (Vesta) | `body.rs:451` | STAYS (a law with its calibration body) |
| 41 | Shadow reach rung 2 (3 476 m) | `crates/client-render/src/terrain.rs:110-116` | The far field (the light phase) |
| 42 | Bias tan cap 8 | `terrain.rs:117-122`, `:4005` | A receiver-plane bias |
| 43 | Cascade end by view depth, no fade | `vendor/bevy_pbr/src/render/shadows.wgsl:117-126`, `:192-198` | A radial fade |
| 44 | Sun lux = π / exposure; a fixed exposure | `terrain.rs:126-136` | The eye's adaptation law |
| 45 | One ground colour, roughness 0.95 | `terrain.rs:2325-2326`, `:2393-2394` | The material laws (K12) |
| 46 | The air's ground albedo = the Bond albedo; the aerosol placeholder | D-TERRAIN-6 | The drawn materials' mean albedo; the source law |
| 47 | 64 weather systems; snow ≤ one cell | 05 §8.5, §8.3 | The count law (§8.2); snow as live memory and a fill substance (§8.4) |
| 48 | R2's m = min(1, \|∇A0\| / TAN_REPOSE), "on average" | r2 §7.4 | REPLACED by P15, which holds everywhere, not on average (F31) |
| 49 | Held & Hou's ΔH = 0.388, fitted to Earth, read on every body | `climate.rs:102-104`, `:321-347` | REPLACED by the body's own contrast (A3), in phase 2 (F29) |
| 50 | The meander belt at 2.7 widths | `crates/terrain/src/river.rs:15-16`, `:40-43` | REPLACED by Leopold & Wolman's full amplitude law (F49) |
| 51 | The province as the nearest node's word (a stand-in for a law) | `artifact.rs:1211-1237` | REPLACED by the outcrop rule (K3 law 7; F35), phase 4 |
| 52 | The tree horizon and the canopy-fold distance | not yet in code | Set on pictures and the forest bench (M0-15), never a typed cap |
| 53 | ADDED 2026-10-02 (the cause bench, `docs/investigation/2026-10-02/phase0_relief_cause.txt`): THE BELTS' ROOM — every belt's amplitude is the relief less the TALLEST platform on the whole planet, times the convergence's and the plate age's shares | `crates/terrain/src/land.rs:1004`, `:1014-1016` | MEASURED on the home planet: room 2 323 m, the largest uplift 769 m, no land node rises 1 000 m. Replaced by the belts' law (convergence over the belt's own active time, limited per node by the crust's strength), designed in phase 0 and built in phase 2's macro change (ruling B37) |
| 54 | ADDED 2026-10-08 (found by the coverage gate, HR5): THE ERODIBILITY'S CEILING — a province's share of `K₀` held at shale's 256 | `crates/terrain/src/strata.rs` (`Province::erodibility_q8`) | GONE 2026-10-08: it never bound (no province holds a rock softer than shale, so every province read 256 or less without it); a province of a softer rock would read more, by Sklar & Dietrich's law itself |

**Kept on purpose:**
- The render-scale floor s ≥ 720 / rows (`crates/client/src/reconstruction.rs:220-232`), a law of the pixel. The valve
  goes under 0.5 only after the owner's look at that scale (F20).
- The shadow switch-off, which reads the body's own relief bound (`terrain.rs:4007-4021`).
- The reconstruction's floors 1/64 and 1/16, the box 1.25 and the sharpen 0.5 (`reconstruction.rs:84-98`): the owner's
  D7 and D9 look constants, marked "proposed until the owner rules D7 and D9" in the code, not physical caps.
- Representation bounds: integer narrowing; the [0, 1] share maps (`climate.rs:434`, `:647`).

---

## 7. The frame, the builders and the memory (full screen, the p99 on moving legs)

### 7.1 Today (MEASURED)

- The full-screen window at s = 0.5 with the temporal path on: **23.1 ms median, about 43 frames a second** (B34 item 3;
  NOT clean: the AirPlay stream, the poller, the cluster on the same machine).
- G-NO-STALL red: frames over 33 ms in 18–36 % of the seconds; 146 ms at the dusk boarding; 3 pipelines compiled after
  the loading (B34 item 3).
- Per pass at s = 0.5 (window hidden, passes steady): the ground 8.44 ms, of which ~6.9 ms is vertex work (ARITH); the air
  1.95; the shadow casting 1.83; bloom 1.10 (`docs/design/DEFERRED.md:10922-10925`).
- The scale ladder 22.2 / 17.0 / 15.5 / 12.7 / 10.5 ms at s = 1 / 0.75 / 0.67 / 0.5 / 0.33 is a SPAN, and a span is not
  a frame's cost when frames overlap (`DEFERRED.md:10934-10944`).
- The frame interval: 15.9 ms at s = 0.5 with the client's CPU at 107 %; 12.4 ms at s = 0.33 with the CPU at 121 %. Below
  half size the client's CPU is the limit (`DEFERRED.md:10944-10946`).
- The main thread on foot: the release walk 1.3 ms and the request walk 1.4 ms a frame, against B-MAIN's 1.0 ms
  (`DEFERRED.md:10728-10730`).
- The default path today is MSAA at s = 1: a 30.9–32.9 ms span (`DEFERRED.md:10938`).
- On foot: 14 261 chunks drawn and 81 M triangles (MEASURED, status log 2026-09-28).

### 7.2 The honest sum (ARITH, F53)

- The draft's planning lines add 5.5 ms (B–E: shape, materials, cover, light) and 6.5 ms (F–K: screen-space light, clouds,
  weather, water, constructions, margin).
- On today's only full-screen reading: 23.1 + 5.5 + 6.5 = **35.1 ms**; with the temporal design's unpriced 5.8 ms cut,
  **29.3 ms**.
- On the model's base (16.0 − 5.8 = 10.2 ms): the draft's 22.2 ms (20.3–30.2).
- So the gap to 16.7 ms lies between 5.5 and about 18 ms, not at 5.5. The model's line A is a NEED, *"not a priced gain"*
  (`docs/investigation/2026-09-28/temporal_path_design.md:1043`), and the geometry step that must deliver most of it is
  "its own design" (`:1286`).

### 7.3 The target split (a proposal for Q10; GPU, s = 0.5, full screen, p99 on moving legs)

| Line | Target (ms) | Lever | Basis |
|---|---|---|---|
| A. Today's content after the speed work | 7.4 | T3's one pass and the geometry step (occlusion culling, the vertex bytes); T5's cheaper air; fewer full-screen display passes | It needs −8.6 ms against the model's 16.0 (DOC) and −15.7 ms against the 23.1 reading (ARITH). UNPRICED: the geometry step's design prices it before phase 3 (§5.3) |
| B. Shape, phases 1–4 (vertices) | 0.6 | FR5 on both pipelines; rule S instead of ring growth; T7's ring rule measured | ESTIMATE 0.5–2.8 (C17) |
| C. Materials and the AAA surface | 1.0 | the shares carried by the vertex (no per-pixel classification); two texture layers; the paint level | ESTIMATE 0.75–2.6 |
| D. The cover and its casting | 2.0 | the tree horizon on pictures; GPU culling; coarser casters; the canopy fold | ESTIMATE 2.0–7.3; measured in phase 0 by M0-15 (F40) |
| E. The light phase | 0.4 | cached horizon angles; the sky-view factor at a chunk's write | ESTIMATE 0.3–0.8 |
| F. Screen-space light | 0.5 | AO at the render size | DOC 0.7 (temporal §10.4), target 0.5 |
| G. Clouds — RESERVED (B35 item 5) | 2.0 | — | ESTIMATE from another GPU and size: Nubis³ 2.1 ms (PS5, 60 Hz), Frostbite 1.6 ms at half size (`temporal_path_design.md:1046`); scaled to the instrument's render pixels in phase 0 (M0-11) and read again in the light phase (F69) |
| H. Weather and particles — RESERVED | 0.8 | — | proposal (DOC temporal §10.4) |
| I. The water fill and the ocean | 0.8 | — | proposal (1.0 → 0.8) |
| J. Constructions, hulls, players | 0.7 | — | proposal (1.5 → 0.7) |
| K. Spike margin | 0.5 | — | a frame over 16.7 ms shows for 25 ms (DOC) |
| **Sum** | **16.7** | | ARITH |

The split fits only if line A falls to about 7.4 ms. Nobody has priced that cut. The owner rules the split (Q10) on the
numbers of M0-11 (the clean base), M0-15 (the forest) and the geometry step's priced design, never on this table alone.

### 7.4 The CPU (F53)

The CPU already limits the frame below half size: 12.4 ms at s = 0.33 with the CPU at 121 % (MEASURED). About 4.3 ms
(ARITH: 16.7 − 12.4) remain for everything the ground step, the weather and the game to come add on the main and render
threads. Proposal:
- the terrain's main-thread work ≤ 1.0 ms mean and ≤ 2.0 ms p99 (B-MAIN; today 2.7 ms on foot);
- the ground step's whole CPU line (main and render threads) ≤ 1.5 ms: no entity and no main-thread work per tree, the
  cover culled on the GPU;
- the weather ≤ 0.5 ms;
- the terrain workers run on their own cores, budgeted by B-BUILD and B-RATE.

### 7.5 The builders (F54)

- **B-BUILD:** the ask-weighted wall time of one build, parents included, at 7 workers and at 4 (the average machine's
  share, B20): ≤ 6.5 ms (DOC one_volume §10A.2).
- **B-RATE:** the late chunks a second on the low legs (240 and 528 m/s, 50–500 m up) may not rise over the phase-0
  baseline; today about 230 a second at the owner's speed (MEASURED, `DEFERRED.md:11612-11626`).
- **ARITH (F54):** phases 3–5 add +1.4–6.5 ms a chunk (ESTIMATE: R3 +1.2–4.1, the classifier and soil +0.1–0.3, the
  climate per column +0.08–0.2, karst +0.08–0.16, crust +0.08, volcanoes 0–0.12, rock control ≤ 0.3, slopes +0.04;
  dunes and cliffs where they stand). After the levers (790–1 330 a second on 7 workers) the rate falls to about 460–970
  a second against an ask of 525–1 030 at 528 m/s and 537 m. So each shape phase's B-BUILD line can go red, and Q11's
  mechanism is chosen on the CPU path after the levers.
- **The card.** It is off by default (`crates/client-render/src/terrain.rs:439`) and builds no solved chunk
  (`terrain.rs:1582-1588`); its quarter of a frame (`crates/client/src/card_budget.rs:34`) has no room in a frame that is
  already over. This plan gives the card no slice. Ruling F7 asks for the GPU "to the maximum", so Q10 asks whether that
  stands (one_volume E9 and P12 hold its measured harms).

### 7.6 The memory (F62)

Today's graphics memory on foot: 2.5–2.9 GB (MEASURED, status log 2026-09-28). Planned additions: the vertex shares
+224–448 MB (ARITH); textures 120–475 MB; trees and rocks 0.5–0.7 GB; the A0 book 128 MiB (ESTIMATE); the far-shadow
pyramid and the cover's buffers UNMEASURED. The gateway's cache is bounded ONCE for each gateway, never for each realm
(r2 §8.3 proposed 512 MiB per realm). Each phase reports the GPU, the CPU and the gateway memory. The average machine's
memory is owed by the owner (ruling F6; Q10).

### 7.7 The rules

- **FR1.** Each phase states its lines (GPU, CPU, B-BUILD, B-RATE, memory) from a bench before it builds. A reading over a
  line stops the phase and goes to the owner. THE RUNNING TOTAL (G-FRAME-TOTAL) — today's measured base, plus each
  landed phase's line, plus the reserved lines G and H — is printed at the end of every phase. In phases 1 and 2 it may not
  grow by more than the phase's line; from phase 3 (after §5.3's speed items) it is a gate at 16.7 ms p99, on the GPU and
  on the CPU (F53).
- **FR2.** The ground step never spends lines G and H (B35 item 5).
- **FR3.** The geometry step's design, T3 and the builder levers come before phase 3 (§5.3; the owner's call under D15).
  The geometry step itself lands before phase 6 at the latest.
- **FR4.** The valve is dynamic resolution between 0.5 and 1. A scale under 0.5 (down to the 0.33 floor) only after the
  owner's look at it (F20). The valve cannot rescue the CPU, which is the limit below 0.5.
- **FR5.** Each shape phase prints its vertices a chunk and the ground pass's vertex ms, on today's forward pass AND on
  T3's one pass once it exists; the larger enters the running total (F61).
- **FR6.** No entity and no main-thread work per tree; the terrain's main-thread cost ≤ 1.0 ms mean and ≤ 2.0 ms p99
  (B-MAIN).
- **FR7.** Every new pipeline is listed and warmed before play; the pipeline meter's count after the loading is part of
  G-NO-STALL (`crates/client-render/src/pipeline_meter.rs`; F65).
- **FR8.** Every budget here is the instrument's (the owner's M4 Pro, one_volume §10A). The average machine (ruling F6) is owed by
  the owner, and every line is ruled again for it when it is named (one_volume §10A.4).
- **Preconditions.** The s = 0.5 budget assumes the temporal path that the owner judged at half size (B34 item 1); D8
  (the scale) and D14 (one path) are still open in the temporal design.

---

## 8. The weather step's interfaces (decided now, built after the water fill)

### 8.1 The almanac (T3 class 1)

- A closed form from the tick and the charter: the day 259 597 s, cos ε 586/1 024, the year 34 727 239 s
  (`crates/terrain/src/home.rs:113-116`).
- **The client** takes the season's phase and the sun from the placement that the composed picture carries. Paint only,
  no chunk rebuild.
- **The planet's own shard** needs the sun's place in its own sky for the weather's forcing and for the melt. Today it has
  no lawful source (F5): the want carries only the eye (`crates/wire/src/session_flow.rs:200-213`). Proposal (A12): the
  star system states the planet's orbit elements and spin law ONCE on change, stamped (SL1 clause 2), and the planet
  evaluates the almanac from the tick. That is the planet computing its own orbital phase, which SL1 clause 3 forbids
  unless the owner rules T3 class 1 as its named exception; the weather module may name no placement, containment or
  crossing code (SL1 clause 5). The alternative — the parent states the sun's direction at every weather step — is a
  per-child row on a 10 s lane, which SL9 forbids at scale (Q17).

### 8.2 The list and its lane (T3 class 2; F6)

- The planet's shard steps the field. It is Category C state, carried by checkpoints. The client never simulates it.
- Rows of 16 B: centre, radius, intensity, phase, drift, kind (DOC 05 §8.5).
- **The COUNT from a law**, not 05's cap of 64: ∝ area / L_R² (Gill 1982, ESTIMATE), calibrated on Earth's 30–60 systems
  (DOC 05). ARITH: L_R ∝ 1/f, and the home planet's spin is 0.33 of Earth's, so the law gives ~3–7 large systems, not 38.
  Bench MW-1 reads it.
- **THE LANE (T3: "on its own lane").** A want and a statement, appended arms in the artifact's realm-audience pattern:
  the gateway asks the planet's shard for its weather; the shard states the list once each weather step (10 s, DOC 05
  §8.7) and on change at most once a second (DOC 05 C-20); the gateway keeps the latest list for every session whose
  picture holds the planet.
  - Crossings: ONE, the planet's shard → the gateway (the connection plane is not a realm, SL2's clarification). NO
    realm-to-realm crossing: nothing rides the sealed relay through the star system. A moon's list goes from the moon's
    shard to the gateway directly.
  - Bytes: N × 16 B a statement (ARITH: ~0.1 KB at N = 7). An SL6 ask (A9).
  - Why not the self-look bag: it rides the window lane's body statement, send-on-change and reliable, and the parent
    relays it one hop up, sealed, to every outside observer (`session_flow.rs:339-354`;
    `crates/wire/src/intershard.rs:415-475`, `:1331-1360`). A 10 s list there would cross the planet → star system
    boundary at every step. `TAG_ARTIFACT` is no precedent: the surface, charter and artifact tags are stated once on
    change and kept (`crates/core/src/look.rs:50-56`, `:82-88`).
- **The stale rule (SL1 clause 6).** A list older than two weather steps plus one is refused. The client then draws the
  almanac alone (the climate's mean cloud cover), never a frozen storm.
- One interval back, no prediction. A new system starts at zero strength.
- **The receivers:** every session whose picture holds the planet (C19).

### 8.3 The dormant planet (T3 class 2's last sentence; F14)

A dormant planet does not step its weather. At the wake, the live layer starts at ZERO (05 §8.4): the sky at the waking
tick is exactly the almanac's, so the wake makes no seam, and storms grow from there. The memory at the wake starts from
the almanac's seasonal snow and wetness, a closed form of the date and the climate: last month's particular storm snow is
gone, and this month's seasonal snow lies where the climate puts it. A planet with an occupant does not sleep, so a storm
that a player stands in never vanishes (the visibility ruling).

### 8.4 The memory (lying snow, wet ground)

- The planet's shard keeps the snow water and the surface wetness on its grid: melt by the degree-day law with the daily
  swing (K11 A7); drying by Penman–Monteith FAO-56 (DOC).
- It states the map by digest and serves it on the realm audience (A10).
- Both hosts read the same map: the client paints it, and the server seats feet on the same snow.
- Deep snow becomes a fill substance of the one-volume model (05's one-cell cap goes).
- The three snow layers (F17): the static shape holds only perennial ice and firn; the almanac paints the seasonal snow
  line with no collision; the memory holds lying snow as live state.

### 8.5 The clouds

The planet's own look, from the static climate, the almanac and the interpolated list. V_cloud multiplies V_sun. The
environment map includes the clouds. The points pass dims the sun and the stars through the sky-share channel. A cloud
pixel writes the motion of its weighted depth. Far away, the mean cover joins the far fold. Line G.

### 8.6 Rain and snow on screen

Reactive, or drawn in the points pass. A top-down depth hides them under roofs and inside hulls (a hull draws its own
inside, SL3). Line H.

### 8.7 The wind

On a hull: a force in the planet's own physics through the drag words that the hull already states; no new data. In
trees: a closed-form sway from the static wind and the list, with exact motion (P10).

### 8.8 What the ground step leaves ready

1. Lines G and H free (§7.7 FR2).
2. V_cloud = 1 in the light sum, and a cloud input to the environment map.
3. s and w as material inputs, with s in the far fold.
4. The sky-share channel.
5. The air volume (T5 step 4) before the weather.
6. The eye's adaptation.
7. Probe kinds for clouds and weather.
8. The pipelines warmed (FR7).
9. The reach read from the brightest look the charter allows, so weather never moves a reach statement.
10. A second air for a far body with air, priced in the weather step (UNMEASURED).
11. The weather stepper and the memory as `ShardProfile` capabilities with their G-IDENTICAL fixture: an O'Neill
    cylinder with soil and air (V4 15.2) steps its weather with the same feature as the home planet (F10).

---

## 9. Risks, the unmeasured items and their benches, the stale record, and the interims

### 9.1 The risks, largest first

1. **The frame does not fit.** The gap to 16.7 ms lies between 5.5 and about 18 ms (ARITH, §7.2). Line A needs a cut that
   nobody has priced, and the CPU already limits the frame below half size (MEASURED).
2. **The builders.** Phases 3–5 add +1.4–6.5 ms a chunk (ESTIMATE) onto builds of 11–22 ms on a moving eye (DOC) against
   V10's 8 ms. Today about 230 chunks a second arrive late at the owner's speed (MEASURED). The card has no slice.
3. **The home planet's climate changes** under the corrected circulation: the deserts, the ice, the tree line, the dunes
   and the stands move (F29). M0-9 benches it first.
4. **R2's predicted cure is unmeasured with this loop** (r2 skeptic 5), and the new operators' bake rate is unknown
   (F60).
5. **The macro change** (the runoff, D∞, the climate laws, phase 4's group) moves the land share and the sea. W7 once
   reached 1.4 % land (DOC).
6. **The first visit.** Minutes of 8 km ground on an unbaked planet that then refines in front of the pilot; an approach
   faster than 1.3–2.4 km/s meets unbaked tiles by the old loop's rate (ARITH, F56). The home planet is prebaked.
7. **The far material shares** need the whole body's bake, and they are large: 49–65 MB raw at the worst eye, and a login
   over 30 s with everything new (ARITH; F55, F66).
8. **The washboard** stays in the rows until the basins step after R2.
9. **Every macro change re-bakes** (1.8 h wall on 14 cores by the old loop; the new loop UNMEASURED).
10. **The ladder's pixel.** One step is ~3 display pixels; the eye's rule at full screen may see swaps that T7's line
    passes (F32).
11. **New machinery:** rule S's placed forms and the outcrop rule (F28, F35).
12. **The province byte's bit budget** (C11) and the row growth (A6, A7).
13. **The art:** licences, the public repository, the budget, 0.6–1.2 GB of memory.
14. **The cascade-end pop** may be visible already (in the code; UNMEASURED).
15. **Overhangs** are drawn closed while the shard's cells hold them open, until the seal cure lands with them.
16. **Dry interiors stay smooth:** 374 mm of rain over the land (MEASURED), so the runoff cuts little; the corrected
    circulation may change that.
17. **The lift stays in the pictures until phase 2** (no centring is built): up to 572 m over the solved ground.

### 9.2 Every unmeasured item, and the bench that measures it

| # | Unmeasured | Bench | Phase |
|---|---|---|---|
| M0-1 | G-MEAN on today's fine terms (each ring, each 1 km cell); μ on two lattices, as a reading only | `rung_swap` with a signed column; one_volume M1 | 0 |
| M0-2 | D9: the fine field's fourfold lean (decides C15) | a new mode of `crates/bins/src/grid_bias.rs` | 0 |
| M0-3 | Barnes' potential: G-GRID-BY-CLASS, G-LAKE-WALK, G-LAKE-SHARE, the washboard's trace | `lake_census` + `router_variants` | 0 |
| M0-4 | Earth, Moon and Mars bands: spectrum, Hack, basins, pits ≥ 1 m per 100 km² on lidar, slopes by class, the bare-rock and snow shares by slope class, junction angles, hypsometry, the class count within 60 km by relief class, the ice line on Earth's real climate and hypsometry, the realized-to-stated slope ratio by kind | a new stand bench on outside tiles (Q2) | 0 |
| M0-5 | The home system's charters; the third body; which bodies have air, water and ice; which kinds of country each calls for (K18) | `home_body_facts` extended, `system_census` | 0 |
| M0-6 | The stands census (classes, picks, counts, border stands), RUN ONCE AND PINNED | new | 0 |
| M0-7 | The 1 000 km coast's divider dimension; the belts' straightness | a divider bench on the rows | 0 |
| M0-8 | The dunes' numbers at generator 22 (the shore law off) | `shipped_drainage` | 0 |
| M0-9 | The climate census of today's code (cells, biome shares, tree line, ELA, the reference stand's ice bit) BESIDE a variant bench of the corrected circulation (the body's own contrast, the seasonal cell, the wind's sign change) | a census line + a variant solve bench (not the world) | 0 |
| M0-10 | The moon's beds and bench today | `province_map` on the moon | 0 |
| M0-11 | T4's clean full-screen frame (AirPlay off, a light meter, the cluster's load known): GPU per pass, CPU per thread, the pipeline count after loading, GPU / CPU / gateway memory; line G scaled to the instrument's render pixels | T4's window meter | 0 |
| M0-12 | The cascade-end pop on a turn: 360° at 90°/s, a 5° sun, ridges 3–5 km away | B-TURN; its control a synthetic frame pair in a Tier-A harness test, never a game switch (F15) | 0 |
| M0-13 | The light step at a swap, on eye windows ≥ 1°, at 5°, 15° and 45° suns | `ring_step` + windows; its control in the harness | 0 |
| M0-14 | The realized spread of height and slope for each octave kind (Σ's constants) | unit octaves (B14's C method) | 0 |
| M0-15 | B-FOREST, B-MEM, B-RENDER-CPU: the forest's GPU and CPU cost at each horizon, culling, casting, memory, the render thread's per-entity cost | a standalone bench binary with CC0 stand-in trees (Q14), never a switch in the game | 0 |
| M0-16 | The eye's rule at full screen on today's swaps: every swap step in display pixels on every leg | `terrain_moving_eye` at full screen | 0 |
| M0-17 | B-BUILD and B-RATE baselines on the low fast legs (240 and 528 m/s; 50–500 m) at 7 and at 4 workers | the arrival census, `chunk_budget` | 0 |
| M2-1 | R2's slice 0 with the new operators: Hack, basins, pits, spectrum, G-MEAN, G-FOLD, G-SILL, G-AMP-HALO, G-DOSE, one level against three, S beside A0; THE RATE (ns a cell an iteration, the iterations to G-DOSE, one tile's wall time on one core) beside R1's loop | `erosion_bench`, re-seeded on today's artifact | 2 |
| M2-2 | The macro change one at a time — D∞; the runoff with K₀ normalised; A1–A3, A7 and the circulation: the land share, the sea, G-HYPSO, the solve's time, the climate census | `lake_census`, `shipped_drainage` on variant solves | 2 |
| M2-3 | The 10 → 11 height and shade step; the exact path's sizes and arrival times | `rung_swap` extended; a size census | 2 |
| M2-4 | The envelope's hits for each pass | the `SolveReport.envelope` census line | 2 |
| M2-5 | The bake's real rate in the service; the first-visit time for one view at the average share; the fastest approach that meets no unbaked tile | the service's timing | 2 |
| M2-6 | The memory of the bake, the client's book and the gateway's cache | an allocation census | 2 |
| M3-1 | R3's options (a), (b), (c′) against Earth: G-HACK, G-JOIN, G-HILLSLOPE, G-SPECTRUM, the hold's residual, G-GRID on R3's field, the CPU cost a chunk after the levers | a new `r3_bench` | 3 |
| M3-2 | Small craters on the moon: cost and SFD with and without bins | a crater bench on B2 | 3 |
| M4-1 | R2's cost with the bed hardness; G-BENCH; G-SLOPE-BIMODAL; the skyline | `erosion_bench`, `slope_histogram`, the skyline | 4 |
| M4-2 | The cohesion values (intact rocks; the moon's crushed crust); face heights against H_c | read the sources; a face census | 4 |
| M4-3 | The bench strength from the contrast: steps for each rung pair, band growth | `rung_swap`, `rung_disagreement`, the chunk count | 4 |
| M4-4 | The outcrop rule's borders: G-BORDER-CONTOUR and the lean | a `grid_bias`-style histogram on rock borders | 4 |
| M4-5 | The removals: cost, lateral breaches, G-OPEN, swap steps | `terrain_cost`, `chunk_budget`, `rung_swap`, the eye's rule | 4 |
| M4-6 | Dunes (giant and small), volcanoes, karst, glacial, coasts, crust, the other kinds: cost, G-DUNE-λ / TYPE, the arc line, the G-HYPSO effect, G-MORAINE exact | M-DUNE-1, M-VOLC-1, M-KARST-1, M-GLAC-1 | 4 |
| M4-7 | A1 against Earth's seasonal amplitude; A2 against the Cascades ratio; A4–A6 per column | the laws run on Earth's REAL inputs (M0-4's data) against Earth's observed climate — an instrument, never a world (F45) | 2, 4 |
| M4-8 | The cell-mean rule's largest pixel step; G-TERM-STEP; the bake's fold of compact terms at 10 → 11 | M-PREFILTER-1, M-STEP-1 | 4 |
| M4-9 | Rule S: the placed forms' hand-over (G-SILHOUETTE); dual contouring at rungs 0 and 5–9 | a silhouette bench; its control a synthetic pillar in a harness test | 4 |
| M4-10 | The cover's thickness law candidates (A6) | a solve bench | 4 |
| M4-11 | The sea floor: depth against age, abyssal-hill RMS against Earth | M0-4's Earth data | 4 |
| M4-12 | The frost band's area on the home planet; the block-field share | a census | 4 |
| M5-1 | The closure's accuracy on ring means and eye windows | a new `material_closure` | 5 |
| M5-2 | The GPU cost of the blending and of 1, 2 and 4 texture layers at s = 0.5 | the per-pass clock | 5 |
| M5-3 | The vertex shares' bytes; the far shares and ring parts compressed; the far fold's bake; arrival times | an allocation count; the far-map bake | 5 |
| M6-3 | The cover's late lists, the fold's silhouette gate, the trees' casting, in the game path | B-FOREST in the game | 6 |
| ML-1 | The far field's cost and stability; the cached horizon angles; the sunward strip by tile numbers at 1° and 5° suns | B-FARSHADOW; G-FAR-SHADOW-STILL | light |
| ML-2 | Contact AO; the shadow read's filter; bias speckles; G-ADAPT | U7, U8, U13 and U12 of the light angle | light |
| ML-3 | G-SHADOW-SWAP and G-SHADOW-HANDOVER | `ring_step` side-lit + the eye's rule | light |
| ML-4 | The sky-view factor at a chunk's write: its cost on B-BUILD | `chunk_budget` | light |
| MW-1 | The list's size by the law; the memory's sparse size; the second air (`VD_SKY_PICK`, `crates/client-render/src/sky.rs:42-46`, only on the owner's word, F15) | a weather census | weather |
| MX-1 | Every phase: B-BUILD, B-RATE, the late chunks a second on the low legs, G-NO-STALL with the pipeline count, the memory, G-FRAME-TOTAL | `terrain_moving_eye` low legs; the window meter | every phase |

### 9.3 The stale record (fix each document when its step touches it)

- Status §3 and D-TERRAIN-8 cite `client-render/src/terrain.rs:109`, `:3339-3353`, `:1974`, `:2034`, `:2073`. Today the
  shadow reach is at `:110-116`, the switch-off at `:4007-4021`, the colour at `:2325` and `:2393`; no water is drawn (B22).
- Status §3 cites `recipe/src/terrace.rs:152` for the cap share; the code is `CAP_THRESHOLD` at `:131`.
- D-TERRAIN-8 cites the biome at `height.rs:501-529`; those lines are test code, and the biome is at `:423`.
- r2 §1 fact 2 says "five words"; the column reads three (`crates/recipe/src/plan.rs:138-148`).
- r2 §8.2 says minor 35, which B19 took. r2 §6.4's prebake tool is replaced by the shard's own prebake (Q8). r2 §7.4's "on
  average" drainage argument is replaced by P15 (F31). r2 §12's bake rates belong to the retired explicit loop (F60).
- one_volume §2.1 cites `octave_term` at `height.rs:460-467`; it is at `:361-368`.
- 05 §7.1 appends strata at codes 19–23, but 19 and 20 are Slate and Marble (`strata.rs:40-41`). 05 §8.8 puts the weather
  list on the self-look bag; T3 (newer) gives it its own lane (F6). 05 §5.4's local rain term is replaced by A4.
- `body.rs:86-93` says 8 224 m / 32 896 m; the code is 32 768 m (`:94-96`).
- `body.rs:1033-1034` says "a relief of 16 454 m"; the law gives ≤ 8 846 m (ARITH), and the solve measured an envelope of
  8 276 m (`solve.rs:1753`).
- `height.rs:52-55` says "five words carry the octave, three are padding"; since the `fine` word was added, it is six and
  two.
- `river.rs:15-16` names Leopold & Wolman's meander belt as 2.7 widths; the published amplitude law is 2.7·w^1.1 in feet
  (ESTIMATE until read; F49).
- `body.rs:195-202` says that a typed 4 "makes that step many cells high"; one_volume §4.6's arithmetic says the opposite;
  the comment marks its own number UNMEASURED.
- **This design's own draft:** "no pop seen on the T2 legs" — the record holds one pop on the 1.4 m/s hull leg (level
  1.02), red, moved to T4 (`DEFERRED.md:11608-11611`); "below s = 0.5 the CPU is the limit: 15.9 ms at 107 %" — 15.9 ms is
  the s = 0.5 reading, and the CPU-limited reading is 12.4 ms at s = 0.33 (`DEFERRED.md:10944-10946`); "~3.4 ms of column
  work" is generator 17's 839 ns a column (one_volume §10A.1.1 H2: "Today's number: UNMEASURED"); the citation of W2 for
  the rung law and the reference to a §7.4 that did not exist (F25, F64).

### 9.4 The interims, to register in DEFERRED.md (this pass writes no other file; the change that builds or rules each one registers it)

| Interim | WHAT proper solution is missing | WHERE | WHEN |
|---|---|---|---|
| The first-visit transient (8 km ground that refines later) | a bake that always wins the race, or the owner's ruled transient (Q7) | the bake service; the client's book | phase 2 |
| An unbaked body's far shares from the rows' fold, then a ramp | the body's whole bake, or the ruled ramp (Q13) | the far map; the client | phase 5 |
| Glacier ice as a surface material | ice with its thickness as a substance of the one volume | the paint table; the fill | phase 5 → the fill step |
| The card parked | the card as a budgeted builder of solved chunks (ruling F9 item 2), only on the owner's word after P12 | `crates/client-render/src/terrain.rs:1582-1588` | the speed work |
| The render-scale valve under 0.5 | the owner's look at that scale | `crates/client/src/reconstruction.rs` | T4 |
| The moon's sedimentary provinces | the composition law | `crates/terrain/src/land.rs:506-528` | phase 4 |
| The ridged fold's lift in the pictures | the fine relief under P15 and G-MEAN | `crates/recipe/src/height.rs:361-368` | phase 2 |
| The climate's Earth-fitted ΔH and its clamps | A1–A3 and the circulation | `crates/terrain/src/climate.rs:102-106`, `:129-130` | phase 2 |
| No far shadow (pictures captioned) | the far march | the light phase | the light phase |
| The province as the nearest node's word | the outcrop rule | `crates/terrain/src/artifact.rs:1211-1237` | phase 4 |
| The frozen patch's identity | (the recipe version, the octave count, the A0 bake tag, the macro digest), or A0 frozen at slice 14 | ruling F1 | slice 14 |
| The planet's own sun | A12 under the owner's ruling (Q17) | the weather step | the weather step |

---

## 10. The owner questions

### 10.1 Asked now (each changes phase 0, 1 or 2, the frame, or an interface decided now)

1. **Q1 — THE FAR VIEW'S TWO RULES AND THE SMALL HILLS' AVERAGE** (B22 Q5; §3.1).
   - (a) Rule M for the height, the material shares, the shade and the canopy, with G-MEAN, G-CLOSURE, G-SHADE-NEST and
     G-FAR-EXACT.
   - (b) Rule S for compact tall forms (placed forms with their own coarser meshes, then a tint), with G-SILHOUETTE. The
     other road, T7 rule 3 alone, grows rung 4's ring about 30 times in pillar country (ARITH).
   - (c) The zero-mean law as a gate now, and NO ridged centring: the ridged fold retires under a field in phase 2, and
     every drawn body holds a field, so one world change and one re-freeze are saved. The lattice offset rides phase 2's
     fine relief only if M0-2 measures a lean.
   - **Recommendation:** yes to (a), (b) and (c).
2. **Q2 — EARTH, MOON AND MARS FOR THE CALIBRATION** (M0-4). SRTM and GMTED2010 (open lidar tiles for the pits under
   30 m), LOLA, MOLA. The data stays outside the repository; only numbers are recorded; each licence is checked first. No
   request carries the user's name or address (a neutral User-Agent); a source that needs an account waits for the
   owner's word (F26). New outside data, not a library.
   - **Recommendation:** yes, in phase 0.
3. **Q3 — THE CIRCULATION THAT FOLLOWS THE SUN** (F29; T5; T9). The home planet's 55° tilt leaves almost no yearly
   contrast from the equator to the pole (ARITH s₂ ≈ +0.005), so the law's circulation is the seasonal cell that crosses
   the equator and follows the sun (Lindzen & Hou 1988). T5 says the climate model refuses "the monsoon" by construction,
   and this circulation is of that kind. Choices: (a) the seasonal circulation as the law, with the wind's sign change
   inside a cell; (b) Earth's fitted contrast on every body — a number that stands in for a missing law (a T9 defect).
   Price: inside phase 2's climate work (3.5–5.5 engineer-weeks, ESTIMATE); the census moves the deserts, the ice, the
   dunes and the stands.
   - **Recommendation:** (a). M0-9 benches it in phase 0 beside today's code, and no stand, biome class, dune type or
     rain-shadow stand rests on the climate until it runs.
4. **Q4 — PHASE 2's MACRO CHANGE** (F33; F49; r2 Q9). D∞ in the sweep; one water (the runoff) for the cut at both levels,
   with K₀'s water term normalised to Earth's land runoff; A1–A3, A7 and Q3's circulation; each measured alone before
   R2's first bake. A7 is +1 B a node (8.87 MB a planet; an SL6 ask).
   - **Recommendation:** yes.
5. **Q5 — R2 UNDER THE RULINGS** (§5.2). The 13 changes; the operators with no caps; the fine relief under P15 (no new
   closed pit, zero on channels, the hillslope law below its length); each bed's own rock in the erodibility from the
   first bake; the bench and the hillshade before any service; the new operators' rate measured first (M2-1), because the
   old loop's 14–25 core-hours do not apply (F60). A1 is an SL6 ask with its arm at the next free minor.
   - **Recommendation:** yes.
6. **Q6 — THE FAR HAND-OVER AT 890 km** (A2; C6). Exact: A0's own 2 km and 4 km means for rungs 11–12 (2–4 MB at
   150 km, ≤ 7.3 MB at the worst eye, ≤ 3.5 s), and later A8's ring parts. Or a bounded smooth step measured at 10 → 11,
   which B16 forbids without the owner's amendment.
   - **Recommendation:** exact.
7. **Q7 — THE FIRST VISIT, AND WHO ORDERS THE BAKE** (F3, F4, F56; A11).
   - (a) The bake is the realm's warm-up: from the moment an occupant or a child is inside the planet's realm, the planet
     bakes the tiles along the path it authors, by the time to need, with the interest radius grown by the closing speed
     (the movement answer of 2026-08-27); the tile under an occupant first.
   - (b) A viewer outside the planet: the client's own ladder names the tiles by number in an appended want arm (A11), so
     no pose crosses. Until the owner rules, a view-based want orders only the serving of baked parts, never the bake.
   - (c) Where the bake cannot win the race, the 8 km ground stands and refines in front of the pilot: a registered
     transient (r2 Q6), entered in DEFERRED.md. ARITH: a tile is named ~1 145 km out, so by the old loop's 8–15 minutes a
     view an approach faster than 1.3–2.4 km/s meets unbaked tiles; M2-5 measures the new number.
   - (d) The future collider (slice 11) collides on the composed rung-0 cells; whether an occupant may stand on an unbaked
     tile is asked again at that slice.
   - **Recommendation:** yes to (a)–(d), with the tile-number want.
8. **Q8 — THE PREBAKE** (F18; B35 item 6; HR1). The home planet's own shard runs the bake with the whole host before
   play: the same binary and function, its own store, G-AMP-PIN on x86-64 and aarch64. Never a separate tool that writes
   rows no shard computed.
   - **Recommendation:** yes.
9. **Q9 — THE LADDER'S PIXEL AT FULL SCREEN** (F32, F63; V14 D8-1). One ladder cell at its switch distance is one pixel
   of a 720-row view; at the owner's full screen it is ~3 display pixels, so T7's "under one cell" lets a step of 2–3
   display pixels through and the far ground is softer than the screen. Choices: (a) keep D8-1, read every swap in display
   pixels too, and let the eye's rule (B27) judge at full screen; (b) a drawable floor at the display pixel: every ring
   ~3 times farther out, ~5–10 times the chunks (ESTIMATE), and every realm's reach moves.
   - **Recommendation:** (a), with M0-16 in phase 0. If the eye's rule sees swaps at full screen, (b) comes back with its
     measured price.
10. **Q10 — THE FRAME, THE BUILDERS AND THE MEMORY** (§7; F11, F20, F40, F53, F54, F61, F62).
    - The target split of §7.3 (16.7 ms p99 at s = 0.5) as the plan, with the honest gap (5.5–18 ms) stated.
    - The running total at the end of every phase, a gate from phase 3.
    - The CPU line: the ground step ≤ 1.5 ms on the main and render threads.
    - T4's clean reading in phase 0; the builder levers, T3 and the geometry step's priced design before phase 3 (D15:
      the owner's call).
    - The forest bench (M0-15) in phase 0.
    - B-BUILD and B-RATE lines in every shape phase.
    - The card: no slice in this plan, because it builds no solved chunk today. Does that stand under ruling F7 item 2?
    - The render scale under 0.5 only after the owner's look.
    - The average machine (cores, memory, GPU class), named by the owner (ruling F6).
    - **Recommendation:** yes; keep the card parked until one_volume's P12 shows no harm; please name the average
      machine.
11. **Q11 — R3's MECHANISM** (F34; A13). A three-way bench against Earth: (a) a closed form; (b) patch erosion; (c′) a
    256 m baked level on land (3.3 GB a planet, ~1 MB a view; an SL6 ask if chosen). The choice follows M3-1.
    - **Recommendation:** approve the bench; no pick before it.
12. **Q12 — THE FAMILIES AT THEIR PRICES** (T5; §3.3). The aeolian family (now with giant dunes and the seasonal spread),
    the volcanic and the karst families (held by T5), and the new ones: the sea floor, frost, and the other kinds
    (badlands, inselbergs, loess, barrier coasts, tower karst) where M0-5 shows that a body calls for them.
    - **Recommendation:** yes, in phase 4, one at a time, each priced again on its bench before it builds.
13. **Q13 — THE GROUND'S MATERIALS FROM ONE SOURCE** (F1, F2, F55; A8). The recipe computes each column's class, lines
    and expected shares in integers on both hosts (the card through rust-gpu later); the vertex carries the shares (V14's
    seam kept; +224–448 MB at the walk, ARITH); the shader only blends them. The far shares (A8): K = the material count
    less one, 6–8 B a cell; the far map 17.7–23.6 MB raw a planet; the ring parts 31–41 MB at the worst eye (ARITH);
    compressed sizes measured first. An unbaked body's far shares: the home system's bodies are prebaked; any other body
    shows the rows' fold until its whole bake lands, then a ruled ramp (a registered transient).
    - **Recommendation:** yes.
14. **Q14 — THE FOREST AND THE ART** (F7, F8; V1; V4). The skeleton (the trunk and crown capsules) from the generator's
    law is the collider, and the art fits it: (a) variants chosen and scaled to the skeleton now, modules (V4 (d)) next;
    or (b) modules first. No generated crown hull: the last coarser mesh is the artist's. CC0 stand-in trees for the
    phase-0 forest bench (the owner's word on those assets). A private store for bought assets. The owner's budget for
    trees.
    - **Recommendation:** (a); CC0 stand-ins for M0-15; the owner's budget.
15. **Q15 — THE SHADOWS** (F57; §5.8; A11). L2 restated with the accepted shadow ladder; the far march over cached horizon
    angles, with the sunward strip named by tile numbers; the cascade fade (a ninth vendored change, or the ground's own
    light function) after M0-12 confirms the pop; G-SHADOW-SWAP and G-SHADOW-HANDOVER. Also asked (F47): should the far
    march come before phase 2's window look? Recommendation for that part: keep B35's order, because the phase-2
    hillshade shows the relief with its own shading.
    - **Recommendation:** yes, in the light phase.
16. **Q16 — THE STANDS AND THE BODIES** (F13, F42, F44, F45; §2.2). The rule run once and pinned; border stands; random
    stands split between land and sea floor; the third body by the charter rule; two suns on every V-ref stand; the moon and
    the third body in every phase's pictures; the reference framing from valley stands, the old pose kept as a summit
    stand; G-ICE on Earth's real hypsometry and climate, not on an Earth-like test body (B23 item 3 over B5) — or the
    owner restates B5.
    - **Recommendation:** yes.
17. **Q17 — THE WEATHER'S CONNECTIONS** (decided now, built in the weather step; §8; A9, A10, A12).
    - (a) Its own lane: an appended want and statement on the realm audience; one crossing, the planet → the gateway; no
      realm-to-realm crossing; the stale rule.
    - (b) The dormant planet: the live layer starts at zero at the wake; the memory starts from the almanac's seasonal
      snow.
    - (c) The memory by digest. (d) The count by the law.
    - (e) The planet's own sun: the star system states the orbit elements and the spin law once (A12), and the planet
      evaluates the almanac. The owner rules T3 class 1 as SL1 clause 3's named exception, fenced off the placement
      machinery.
    - **Recommendation:** yes to (a)–(e).
18. **Q18 — G-LOOK's SCOPE AND THE STATUS FILE** (F47; F36). G-LOOK judges qualities 1–5, 7 and the light without clouds
    and with the dry sea floor (the sea-floor family lands in phase 4, before G-LOOK); a second look after the weather
    step judges the clouds and the water; the status file stays until that second look (amends B6 item 4).
    - **Recommendation:** yes.

### 10.2 Asked at the phase, with its numbers (each default stated)

- **Phase 2:** Barnes' law on the routing's flats, built with the basins after R2, unless the bench removes the washboard
  with no flooding; the envelope's draw and band term removed (the highest belts can rise ~550 m, ARITH); the bake's share,
  a quarter of each host at the lowest priority (B20's half is the client's), plus the prebake by the shard; re-bakes
  accepted.
- **Before phase 3:** the target machine (Q10).
- **Phase 4:** the cohesion sources (the intact rocks; the moon's crushed crust); the bench strength, the cap share and
  the bed thickness from laws (today's printed as ASSUMED until then); the bit budget of C11 (default (b)); the outcrop
  rule's data A6; the new provinces and the composition law before the moon is a check stand; the removals with the seal
  and breach cures; dual contouring at rungs 0 and 5–9 as a bench; caves by law; moraines and rebound with the ice cap
  removed; the wind-speed law (research first); rule M and rule S on every body; the coastal wave term inside the
  operators, judged after the fill; the tidal word A14; a wet past for a dry body (a history, with its price); the
  icy-moon laws only with a stand on a real icy body.
- **Phase 5:** geology names for soils (C-19); glaciers as a surface material until the fill, captioned; the snow laws'
  calibration on M0-4.
- **Phase 6:** the step height for rock objects, 0.5 m (ESTIMATE; the owner's number); avalanche tracks yes (cheap and
  static); fire mosaics later.
- **The light phase:** the shadows of folded hills fade with their hills (build B only if the eye's rule sees the fade);
  the eye's adaptation law (the frozen references then change once); G-SHADE-NEST at half a level on eye windows.
- **The weather step:** the second air; the reach from the brightest look; `VD_SKY_PICK` only on the owner's word (F15).
- **The freeze (slice 14)** (F9): the frozen patch's identity becomes (the recipe version, the octave count, the A0 bake
  tag, the macro digest), with the shard keeping a patch's old A0 parts and serving them (an SL6 row) and the client
  refusing a mismatched bake (V1 item 3); or the owner rules that A0 and every family freeze at slice 14 and only
  appended octaves change after it.

---

## 11. The marks

| Mark | Meaning |
|---|---|
| `path:line` | Read in today's code at `ae39b32` plus its uncommitted changes. The final pass re-read every line that a finding disputed or that a changed claim rests on: the versions; the octave term; the biome and the stratum reads (`crates/recipe/src/plan.rs:214`, `height.rs:423`, `cell.rs:436-452`); the province read (`artifact.rs:1211-1237`); the slope share (`artifact.rs:1813-1839`); the climate's circulation, wind, clamps and seasonal limit; the solve's constants, rain cut, ice cut, envelope and repose; the crater constants, denudation and plate clock; the meander constant; the float fence; the ladder pixel and the window's projection; the survival rule; the shadow ladder, the bias, the card switch, the card's job filter, the worker share, the colour, the switch-off; the vendored cascade selection; the sinks, the seal and the morph; the reconstruction's floor and constants; the gateway's pace; the bulk arms, the want, the window body and relay arms, the look tags; the shard kinds; the artifact boot; the terrain edges of sim, node and physics; the pinned stands; the pipeline meter; the home facts. Other lines come from the angle reports and the draft. |
| MEASURED | A tool printed it in an earlier run. The record is named (a ruling, `DEFERRED.md` with its line, the status file, or a code comment). Nothing ran for this design. |
| DOC | From a ruling, a design document or a code comment; not measured again. |
| ARITH | Arithmetic of this panel on cited inputs. The inputs carry their own marks. |
| ESTIMATE | A judgement, or a published value not read at first hand in this work. The source is named, and it is read before a gate uses it. |
| UNMEASURED | Nobody printed it. §9.2 names its bench. |

---

## APPENDIX — every review finding and its answer

**The count.** 70 findings from three reviewers (laws F1–F27, quality F28–F52, speed F53–F70): 5 blockers, 31 majors,
34 minors. **ACCEPTED: 70** — 69 in full, and F25 with one of its claims narrowed (that part is refuted with evidence).
**REFUTED in full: 0.** F68 is accepted and becomes moot for now, because F48's answer removes the change it priced.
Where a finding disputed a claim about the code, the final pass read the code again; the lines are named in each answer.

| Reviewer | Blockers | Majors | Minors | Accepted | Refuted |
|---|---|---|---|---|---|
| laws (F1–F27) | 1 | 11 | 15 | 27 (F25 narrowed in part) | 0 |
| quality (F28–F52) | 2 | 13 | 10 | 25 | 0 |
| speed (F53–F70) | 2 | 7 | 9 | 18 | 0 |

**F1 · laws · BLOCKER · ACCEPTED.** The draft's per-pixel rule w = F(fields, h, n, Σ) derived the recipe's own
thresholds again in a shader, which is a port (ruling F7 item 3), and its far fold would have been a third copy. Verified: the
recipe decides the biome on the final surface (`crates/recipe/src/plan.rs:214`, `crates/recipe/src/height.rs:423`), and the
stratum reads that biome (`crates/recipe/src/cell.rs:436-452`). Change: the recipe computes the class, the line quantities,
the soil law and the expected shares per column in fenced integers on both hosts (the card through rust-gpu later); the
far fold calls the same function; the vertex carries the shares; the shader only blends them. M5-4's float classifier is
deleted; G-PAINT-AGREES compares two compilations of one source; §4.3's row reads "the body's static shape, both hosts".
(§1.2 C12; §3.2 K12; §4.3; §5.6.)

**F2 · laws · major · ACCEPTED.** V14's seam stands for ALL ground: the stratum per vertex and the biome per column, with
the vertex also carrying the column's expected shares. So no amendment of V14 or B21 is needed, and Q13 asks none.
(C12; K12; §5.6.)

**F3 · laws · major · ACCEPTED.** The bake is now the realm's warm-up (the occupant's tile first; the interest radius
grown by the closing speed). The collider rule on an unbaked tile is asked at the collider slice (no collider exists
today, F19). The baked/unbaked edge and the later refinement are a registered transient (r2 Q6), asked in Q7 and listed in
§9.4. §0.5's false sentence ("it never changes shape later") is replaced by the true one. (C4; §4.2; §5.2; Q7.)

**F4 · laws · major · ACCEPTED.** The bake's order reads the planet's own occupants and occupied children at the placements
it authors (SL7). An outside viewer's want names tiles by number (A11), so no eye orders the planet's work; until the owner
rules, a view-based want orders only the serving of baked parts. Verified: `ArtifactWant` carries the eye's direction and
distance (`crates/wire/src/session_flow.rs:190-213`). (§1.1 SL2, SL7; C24; §4.3; Q7.)

**F5 · laws · major · ACCEPTED.** A11 is now an ask: the client names the sunward tiles by number, so no direction reaches
the planet. The planet's own sun for the weather's forcing and the melt is A12: orbit elements stated once by the parent,
the almanac evaluated by the planet, under the owner's ruling on T3 class 1 against SL1 clause 3; the per-step alternative
is refused by SL9. Verified: the want holds the eye only (`session_flow.rs:200-213`). (§4.4 A11, A12; §8.1; C25; Q15;
Q17.)

**F6 · laws · major · ACCEPTED.** The weather list leaves the self-look bag and gets its own lane (T3): appended want and
statement arms on the realm audience; one crossing (the planet → the gateway); no realm-to-realm crossing; the cadence,
the bytes and the stale rule stated. Verified: the window body is send-on-change, reliable and re-driven
(`session_flow.rs:339-354`); the parent relays it sealed (`crates/wire/src/intershard.rs:415-475`, `:1331-1360`); the
surface, charter and artifact tags are stated once on change (`crates/core/src/look.rs:50-56`, `:82-88`). (§8.2; A9; C5;
C19; Q17.)

**F7 · laws · major · ACCEPTED.** The collider comes from the generator's skeleton (V1 item 1; V4 step 9 (c)–(d)); the art
fits the skeleton; G-TREE-FIT and a skeleton pin join the gates; the private store never touches the static shape. Q14
tells the owner. (K13; C13; Q14.)

**F8 · laws · major · ACCEPTED.** The generated crown hull is deleted. The asset's own last coarser mesh, authored by the
artist, is V10's middle stage. (K13; Q14.)

**F9 · laws · major · ACCEPTED.** A frozen patch's identity must name the A0 bake tag and the macro digest, or A0 and
every family freeze at slice 14; asked at the freeze (§10.2) and listed as an interim (§9.4). (§1.1 rulings F1/F2.)

**F10 · laws · major · ACCEPTED.** The bake service, the weather stepper and the weather memory are `ShardProfile`
capabilities with a G-IDENTICAL fixture on two shard kinds; a piece that reads the cube-sphere lattice sits below the
seam with G-MAPPING coverage. Verified: `NodeKind`, `VoxelGeometry` and `ShardProfile` (`crates/sim/src/capability.rs:25-42`,
`:116`). (§1.1 HR3, HR4; §5.2; §8.8.)

**F11 · laws · major · ACCEPTED.** §7 now gives a target split whose lines sum to 16.7 ms, each with its lever and basis;
it states the honest gap (5.5–18 ms), adds the CPU line and the running total, and names the instrument; the average
machine is asked (Q10). (§7.)

**F12 · laws · major · ACCEPTED.** Every family and every phase carries an ESTIMATE with its basis and the bench that
replaces it (§3.3; §5.1–§5.2 costs); the locked world with air is priced (K15). (§3.3.)

**F13 · laws · minor · ACCEPTED.** Every phase's pictures include class stands on the home moon and on the third body.
(§2.2; the §5 preamble; Q16.)

**F14 · laws · minor · ACCEPTED.** The dormant rule: the live layer starts at zero at the wake (05 §8.4), and the memory
starts from the almanac's seasonal snow. (§8.3; Q17.)

**F15 · laws · minor · ACCEPTED.** Every bench control lives in the harness (synthetic frames in Tier-A tests); no new
game switch; `VD_SKY_PICK` only on the owner's word. Verified: it is a measurement knob in the renderer
(`crates/client-render/src/sky.rs:42-46`). (§1.1 HR6; §9.2 M0-12, M0-13, MW-1.)

**F16 · laws · minor · ACCEPTED.** A province is static composition; each new substance passes S5.4's wiki test; anything
valuable is live state. Verified: Salt is already code 17 (`crates/terrain/src/strata.rs:33`). (§1.1 S5-6; K12.)

**F17 · laws · minor · ACCEPTED.** The tidal range from the charter (A14); snow in three layers; the Ocean and SeaIce
classes to the fill pass; Beach from the coastal deposit. (§1.1 T3; K6; K12; §8.4.)

**F18 · laws · minor · ACCEPTED.** The prebake runs as the planet's own shard (the same binary, the whole host before play,
its own store), with G-AMP-PIN on both targets; asked (Q8). Verified: a shard solves its own artifact at its first boot
(`crates/bins/src/artifact_boot.rs:1-16`). (§1.1 HR1; §5.2; Q8.)

**F19 · laws · minor · ACCEPTED.** No terrain collider exists: `crates/sim/Cargo.toml`, `crates/node/Cargo.toml` and
`crates/physics/Cargo.toml` hold no `vd-terrain` dependency (read). G-OPEN compares the drawn mesh with the recipe's rung-0
cells; §0.7 item 4 is reworded. (§0.7; §2.1; §5.5.)

**F20 · laws · minor · ACCEPTED.** The valve stays between 0.5 and 1; under 0.5 only after the owner's look. (§6 "kept on
purpose"; §7.7 FR4; §9.4.)

**F21 · laws · minor · ACCEPTED.** Row 26 has a fate (the crust law's own bounds, as named law-limits); the soil law is the
one rule for soil and bare rock, and repose serves debris only. Verified: the factor is "clamped at its ends"
(`crates/terrain/src/land.rs:136-147`). (§6 row 26; K2; K12.)

**F22 · laws · minor · ACCEPTED.** P13: each law states its deterministic form before its phase (integer tables for D∞'s
facet angle and Culmann's sines and cosines, the crate's own functions, the recipe's exact reciprocal). Verified: the
fence (`crates/terrain/clippy.toml:9-40`) and the crate's own `exp`, `ln` and `powf` (`crates/terrain/src/gf.rs:177-225`).
(§3.1 P13; §5.2.)

**F23 · laws · minor · ACCEPTED.** Rule M (b): the bake folds each compact term's volume per A0 cell into A0's coarse means
and the far map; the far shares are sampled by the shader's own level (B17 Q9) or measured at the swaps. (§3.1; §4.2;
M4-8.)

**F24 · laws · minor · ACCEPTED.** Grass and boulders have far rules (their shares in the closure; the boulders' own
meshes to under a display pixel, then the tint); a removal's last step is gated at p99 ≤ 1 ladder pixel. (K13; P14; §5.5.)

**F25 · laws · minor · ACCEPTED, one claim narrowed.** Accepted: W2 is removed from the rung row; the reference to a §7.4
that did not exist is gone; the false §5.9 sentence is withdrawn (§5.10); "skin" no longer names R2's fine relief; the new
interims are listed (§9.4). Narrowed (refuted in part): B22 Q1 (b) does reach the ground's far half — one_volume §4.3's far
fold is *"One rule — the mean — at every far rung, for ground and water alike"*, and option (b) reads *"Above `L_ref` ...
the far fold"* (§4.5). What is new under R2 is the choice of `L_ref` and the data it costs, so the citation now says that
and Q6 asks. (§1.1; C6; Q6.)

**F26 · laws · minor · ACCEPTED.** Q2 states the identity rule (a neutral User-Agent) and the owner's word for any
account-bound source. (Q2.)

**F27 · laws · minor · ACCEPTED.** P12: "up" comes from the realm's gravity function in every term, in the cover and in
the canopy fold. (§3.1 P12; §1.1 HR4.)

**F28 · quality · BLOCKER · ACCEPTED.** Rule S is added: a compact tall form is a placed form that both hosts compute and
the client draws by its own coarser meshes to under one display pixel, then as a tint; G-SILHOUETTE gates the hand-over;
dual contouring is benched at rungs 0 and 5–9; the ring-growth road is priced (about 30 times rung 4's ring). Verified: the
ladder pixel (`crates/client/src/ladder_view.rs:199-210`, test `:2832`; `crates/core/src/geometry.rs:1160-1171`) and the
hand-over line (`crates/terrain/src/body.rs:146-150`). (§0.2; §3.1; K3 law 9; §4.1; C27; Q1.)

**F29 · quality · BLOCKER · ACCEPTED.** Verified: ΔH = 0.388 is *"a REVERSE FIT to Earth's 30°"*
(`crates/terrain/src/climate.rs:102-104`); the count reads the gravity, the scale height, the spin and the radius, never the
body's contrast (`:321-347`); an even cell's wind blows from the east at every latitude (`:651-681`); the tilt's cosine is
586/1 024 (`crates/terrain/src/home.rs:114`). A3 derives the circulation from the body's own contrast (the seasonal cell,
the wind's sign change at φ_H/√3); the climate figures are marked as products of today's code and withdrawn until M0-9;
the solstice cell against T5's "no monsoon" is asked (Q3). (§0.7 item 6; K5; K11; C21; Q3.)

**F30 · quality · major · ACCEPTED.** Verified: the wet repose tangent is 0.47, *"calibration: vegetated hillslopes'
threshold"* (`crates/terrain/src/solve.rs:104-108`), and the denudation is Earth's outcrop rate for each node
(`crates/terrain/src/craters.rs:139-145`, `:330-346`). The soil law now reads the local lowering rate of the drawn surface;
S_c ≈ 1.25 bounds soil-mantled slopes; repose serves debris only; G-BARE-SLOPE reads against Earth. (K2; K12; C10; §6
row 16.)

**F31 · quality · major · ACCEPTED.** P15: the fine relief's gradient stays under A0's drained slope everywhere (zero on
channels and floodplains); below the hillslope length the hillslope law makes the forms; G-DRAIN-DRAWN counts pits ≥ 1 m
per 100 km² at rungs 0–3 against Earth lidar; the spectrum is a reading. (§3.1 P15; §5.2 change 12; §6 rows 6, 7, 48.)

**F32 · quality · major · ACCEPTED.** The ladder's pixel at the target is stated (3.0 display pixels); every swap gate reads
display pixels too; the owner gets the two priced choices (Q9), with M0-16 in phase 0. Verified: *"one pixel high at the
reference view"* (`ladder_view.rs:4-5`), the pixel (`:199-210`), and the window's default 45° projection
(`crates/client-render/src/lib.rs:1485`). (§2 preamble; C23; Q9.)

**F33 · quality · major · ACCEPTED.** A1–A3, A7 and the circulation move into phase 2's macro change, measured alone before
R2's first bake; phase 4 keeps the process families. Verified: the rain clamp (`solve.rs:56-60`), `CELLS_MAX`
(`climate.rs:106`) and the lift clamp (`climate.rs:129-130`, `:406`). (C3; K11; §5.2; §6 rows 20–22, 25; Q4.)

**F34 · quality · major · ACCEPTED.** R3 gets a three-way bench — (a), (b) and (c′) — against Earth, with no recommendation
before it; (c′)'s size is priced (3.3 GB a planet, ~1 MB a view, ARITH). (§5.4; A13; C9; Q11.)

**F35 · quality · major · ACCEPTED.** The outcrop rule replaces the warped contacts (K3 law 7); A6 becomes necessary;
G-BORDER-CONTOUR reads it. Verified: the province is the nearest node's word (`crates/terrain/src/artifact.rs:1211-1237`),
and the bed's rock is read at a fixed radius (`cell.rs:380-425`). (K3; C14; §4.4 A6; §6 row 51.)

**F36 · quality · major · ACCEPTED.** A sea-floor family (K16) with its laws, layers and price, built on the solve's own
plate clock (`craters.rs:311-327`); G-LOOK's scope over the dry sea floor is asked (Q18). (K16; §3.3; Q12; Q18.)

**F37 · quality · major · ACCEPTED.** A frost family (K17); the daily swing A7 as a closed form; "no frost at sea level" is
withdrawn as a product of the seasonal model alone. Verified: the climate holds a seasonal model and no daily term
(`climate.rs:420-470`). (K11 A7; K17; Q12.)

**F38 · quality · major · ACCEPTED.** The snow share follows the slope (Bernhardt & Schulz 2010) and the wind (Winstral,
Elder & Davis 2002) as static functions; G-SNOW-SLOPE reads it. (K12; §2.1.)

**F39 · quality · major · ACCEPTED.** The dune type comes from the seasonal spread of the wind; giant dunes are spaced by
the boundary layer; the supply comes from the solve's sediment (A6); big dunes sit in A0 and R3 as deposits; the family is
priced again. Verified: one fixed wind direction for each node (`climate.rs:651-681`). (K5; §3.3.)

**F40 · quality · major · ACCEPTED.** The forest bench (M0-15: B-FOREST, B-MEM, B-RENDER-CPU) moves to phase 0 as a
standalone bench with CC0 stand-ins on the owner's word; Q10 rules the split on its numbers. (§5.0; §7.3 line D; Q10;
Q14.)

**F41 · quality · major · ACCEPTED.** The false sentence is withdrawn; every shape phase has a B-RATE line on the low legs
that can go red; the builder levers come before R3, as B22 step 2's owed items and not as the waves' cure. Verified: about
230 late arrivals a second (`docs/design/DEFERRED.md:11612-11626`). (§5.3; §5.10; §7.5.)

**F42 · quality · major · ACCEPTED.** The rule runs once in phase 0 and pins every pick; border stands are added; the
random stands are split between land and the sea floor. Verified: the coast and the plain are pinned by direction for
exactly this reason (`crates/bins/src/drainage_gates.rs:66-80`). (§2.2; Q16.)

**F43 · quality · minor · ACCEPTED.** G-VARIETY's band comes from Earth's own distribution by relief class (M0-4); the
picture stays the owner's look. (§2.1.)

**F44 · quality · minor · ACCEPTED.** Two suns on every V-ref class stand. The final pass viewed the third reference (outside
the repository): the trees' shadows are short, so its sun is high (ESTIMATE 40°–60°). (§0.6; §2.2.)

**F45 · quality · minor · ACCEPTED.** G-ICE reads Earth's real hypsometry and climate (M0-4), or a real body of THE world;
otherwise the owner restates B5 under B23. (§1.1 SL5; §2.1; M4-7; Q16.)

**F46 · quality · minor · ACCEPTED.** Each body's expected forms are written after M0-5; a wet past is a ruled history; the
icy-moon laws come only with a stand on a real icy body. Verified: rain only where air and water exist
(`climate.rs:730`). (K1; K15; §2.2.)

**F47 · quality · minor · ACCEPTED.** Asked (Q18): G-LOOK's scope, a second look after the weather step, and the status
file kept until then. The far march before phase 2's look is offered as a question (Q15), not as a change of order.
(§5.9; Q15; Q18.)

**F48 · quality · minor · ACCEPTED.** No ridged centring is built; the zero-mean law becomes a gate (G-MEAN) that phase 2's
fine relief passes; asked in Q1. (§3.1; §5.1; C26; Q1.)

**F49 · quality · minor · ACCEPTED.** Verified: 2.7 widths (`crates/terrain/src/river.rs:15-16`, `:40-43`), K₀ calibrated
on the drainage area (`solve.rs:83-88`), the rain times the area (`solve.rs:1097`), 374 mm of rain and 75 mm of runoff
(`craters.rs:146-152`). Change: Leopold & Wolman's full amplitude law, an alluvium material, and K₀'s water normalisation
before M2-2. (K1; K12; §6 row 50; Q4.)

**F50 · quality · minor · ACCEPTED.** The moon's crust gets its own cohesion law (the "30 times taller" is withdrawn); folds
follow the competent stack; Nye and Blake for domes, not Hulme; κ's size range is stated; rays and maturity are a law of
the crater family. (§0.7 example; K3; K7; K9.)

**F51 · quality · minor · ACCEPTED.** K18 lists badlands, inselbergs and domes, loess mantles, barrier coasts and spits,
and tower karst, with their switches and layers; M0-5 states which bodies call for them; they are priced. (K18; §3.3.)

**F52 · quality · minor · ACCEPTED.** As F12: every family is priced as a range, with its bench. (§3.3; Q12.)

**F53 · speed · BLOCKER · ACCEPTED.** §7 is re-based: the honest sum on the only reading (29–35 ms), the target split, a
CPU column, the running total (a gate from phase 3), T4's cures, T5 and the geometry step's priced design before phase 3,
the instrument named and the average machine asked, D8 and D14 as preconditions. Verified: line A is *"the NEED ... not a
priced gain"* (`docs/investigation/2026-09-28/temporal_path_design.md:1043`) and the geometry step is *"its own design"*
(`:1286`); the main-thread walk (`DEFERRED.md:10728-10730`), the default path (`:10938`) and the CPU limit
(`:10944-10946`). (§7; Q10.)

**F54 · speed · BLOCKER · ACCEPTED.** The false sentence is withdrawn; B-BUILD and B-RATE lines in every phase; the builder
levers before phase 3; Q11 decided on the CPU path after the levers; the owner is told that the card gets no slice.
Verified: the card is off by default (`crates/client-render/src/terrain.rs:439`, `env_armed`), takes only jobs without an
artifact (`:1582-1588`), and holds a quarter of a frame (`crates/client/src/card_budget.rs:34`). (§5.3; §5.10; §7.5; Q10.)

**F55 · speed · major · ACCEPTED.** The far height needs no bake (the mean hold); the far shares do. Their source for an
unbaked body is asked (Q13): prebake the home system's bodies, and elsewhere the rows' fold and a ruled ramp. K is the
material count less one (6–8 B), and A8 is priced again. (§4.2; §4.4 A8; §9.4; Q13.)

**F56 · speed · major · ACCEPTED.** The first-visit sentence is corrected; the bake is the warm-up; the fastest clean
approach is measured (M2-5); the transient is asked and listed; the collider question is asked at its slice. Verified: r2's
own numbers (§12; §16 finding 16; the A0 ask's reach, `docs/investigation/2026-09-24/r2_design.md:374`). (§0.5; §5.2; Q7.)

**F57 · speed · major · ACCEPTED.** L2 is restated with the shadow ladder; horizon angles are cached instead of visibility;
G-SHADOW-SWAP and G-SHADOW-HANDOVER are added; the pyramid reads the morphed heights and ramps a late chunk's write.
Verified: the ladder (`terrain.rs:217-222`, `:303-313`), the owner's accepted look (`DEFERRED.md:8166-8172`), and the
50-level step of an unsunk caster (`terrain.rs:2381-2385`). (§5.8; C22; Q15.)

**F58 · speed · major · ACCEPTED.** Trees sit on the drawn, morphed ground; the canopy fold is one field with its silhouette
at the crown envelope; the cover's jobs go by the time to need; the trees' casting is priced in line D. Verified: the morph,
the sink and the ramp (`crates/client-render/src/ladder_fade.wgsl:175-222`; `crates/client/src/chunks.rs:60-77`). (K13;
§5.7; §7.3.)

**F59 · speed · major · ACCEPTED.** R3's hold is exact by construction or an exact integral; the bands are split (A0
≥ 2 km, R3 < 2 km); the residual and G-GRID on R3 join M3-1. (§4.1; §5.4.)

**F60 · speed · major · ACCEPTED.** Every bake number is now UNMEASURED for the new operators; M2-1 prints the rate before
the service; the bake stays idle during every frame reading. (§0.5; §3.3; §5.2; Q5.)

**F61 · speed · major · ACCEPTED.** T3 and the geometry step's priced design are proposed before phase 3 (the owner's call
under D15, `temporal_path_design.md:1533-1535`); FR5 reads both pipelines and carries the larger. (§5.3; §7.7 FR3, FR5; Q10.)

**F62 · speed · minor · ACCEPTED.** A memory line in every phase (GPU, CPU, gateway); today's baseline in M0-11; the
gateway's cache bounded once for each gateway; the machine's memory asked. (§7.6; Q10.)

**F63 · speed · minor · ACCEPTED.** The pixel is named in every gate and rule (ladder, display, render); the tree horizon
is computed again in display pixels (6.5 km). (§2 preamble; §5.7.)

**F64 · speed · minor · ACCEPTED.** All five lines are corrected (§9.3; §2.1; §7.1), and generator 22's column work is
marked UNMEASURED. Verified: `DEFERRED.md:10940-10946`, `:11608-11611`; one_volume §10A.1.1 H2.

**F65 · speed · minor · ACCEPTED.** P11 and FR7: each phase lists and warms its pipelines, and the pipeline meter's count
after the loading joins G-NO-STALL. Verified: the meter counts ready and waiting pipelines
(`crates/client-render/src/pipeline_meter.rs:1-30`). (§3.1; §7.7.)

**F66 · speed · minor · ACCEPTED.** Every size in §4.4 carries its arrival time; B-LOGIN joins the gates; the worst eye's
total (65–81 MB, 31–39 s, ARITH) is stated and served by the time to need. Verified: 1 MiB a half-second beat for each
client (`crates/connection-plane/src/gateway/artifact.rs:41-50`). (§4.4; §2.1.)

**F67 · speed · minor · ACCEPTED.** The sky-view factor is computed per texel at a chunk's write (+0.1–0.4 ms a chunk on
the builder, ESTIMATE); the per-pixel home is priced (2.4–12.8 ms, ARITH) and refused. (§5.8; ML-4.)

**F68 · speed · minor · ACCEPTED (moot for now).** With F48 no centring is built, so no crest grows. If a body ever draws
the ridged fold again, M0-1 prices which T7 rule acts (one more rung of life, or rings out) with the chunk count and the
column cost. Verified: the survival rule (`body.rs:143-200`). (§3.1; §9.2 M0-1.)

**F69 · speed · minor · ACCEPTED.** Line G is marked an ESTIMATE from another GPU and another size, scaled to the
instrument's render pixels in M0-11 and read again in the light phase. Verified: *"Nubis³ 2.1 ms (PS5, 60 Hz); Frostbite
1.6 ms at half size"* (`temporal_path_design.md:1046`). (§7.3.)

**F70 · speed · minor · ACCEPTED.** G-CLOSURE and G-SHADE-NEST read the eye's windows (≥ 1°, display pixels) at each swap,
beside the ring means. (§2.1; M0-13; M5-1.)
