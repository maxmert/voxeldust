# Owner decisions, 2026-09-12 — the frozen patch, the world identity's tolerance, and the GPU question

Written after Step 8 (`d4d8381`). Newest ruling file; it wins over every earlier design document.

## F1. THE FROZEN PATCH — a built area keeps the shape it was built on (owner: *"let's do it"*)

**The question.** Format D (the world identity) is an APPEND-ONLY octave list. A new octave improves
the terrain of a world that players already live in. The owner's idea: *"we don't change world parts
where something was built."*

**RULED.** The mechanism is approved:

1. The octave list stays append-only. World version *k* is the first *k* octaves of the list.
2. When a player builds the first thing in an area, the owning realm records THE OCTAVE COUNT of that
   moment for a PATCH of cells around the built thing. The count is a small integer stored with the
   realm and shipped as a row of the block store's diff (slice 9), like any placed block.
3. The client (and the server, for collision) derives a frozen patch's static shape with the patch's
   recorded count and the rest of the world with the newest count. Both hosts run the one generator
   (SL10); the count is one more integer input beside the seed, the address and the charter.
4. At a patch's edge a BLEND BAND crosses from the old shape to the new one, so no ridge stands around
   a built thing.

**Example.** A player builds a house on a hill under version 5. Version 6 adds an octave that makes
cliffs. The hill under the house keeps its version 5 shape forever; the unbuilt valley next to it gets
the cliffs; the band between them slopes from one to the other over a few cells.

**What it protects.** Nothing a player built ever floats or is buried by a later version of the
world. The unbuilt world may still improve.

## F2. THE TOLERANCE OF AN ADDED OCTAVE (the number Format D owed)

**RULED, as recommended, unless the owner states another number.** An added octave may move the ground
by at most what the blend band hides: ten cells of the band times a slope of one in twenty — HALF A
METRE at the one-metre rung. The generator's gate measures it: the largest surface move of the new
octave over the home planet's sampled columns is red above the tolerance.

Superseded: the earlier recommendation of one density step (about eight millimetres), which forbade
every visible change of existing terrain anywhere; with the frozen patch a built area never moves, so
the tolerance protects only the band's look.

## F3. STILL OPEN — for the slice 9 discussion (no slice starts before its own discussion, ruling V4)

- The patch's EXTENT: how many cells around a built thing are frozen (a fixed radius; the chunk; the
  chunk plus one ring).
- Whether a patch's count may ever ADVANCE (an empty patch — every built thing removed — thaws to the
  newest count; a lived-in patch never).
- The band's width and its shape (a linear slope over N cells, or the generator's own smooth step).
- The storage row (the count as a row keyed by the patch, in the block store's diff lane).

## F4. THE GPU QUESTION (owner: *"maybe worth to do it on gpu right away?"*) — ANSWERED, NOT RULED

The owner asked whether GPU generation avoids the big-number problem of vast coordinates. The answer
given, from the GPU spike (`slice_08_gpu_spike.md`, ruling V18):

- The vast coordinates are INTEGERS (the seed, the address, the charter) and the 64-bit integer hash
  agrees on the GPU on 3 936 256 corners. That part is safe on either host.
- The recipe's FLOATS are 64-bit on the CPU. Metal has NO 64-bit floats. The GPU can run only a 32-bit
  shadow of the noise, and that shadow agreed with the CPU ONLY with fused multiply-add contraction
  switched off, which wgpu exposes no switch for. So the GPU does not avoid the precision problem; it
  makes it worse, and SL10's no-drift gate would have to measure every target byte for byte.
- On the height columns the GPU equalled fourteen CPU cores: no speed gain measured.
- The wall today (§24.4): at fourteen workers the workers are busy about half the time; the harvest
  cap and the ask's timing bind, not the build. A faster builder would not move those numbers.

Ruling V18 stands: the recipe stays 64-bit on the CPU; the GPU is tried only after the disk cache
and a flight that still shows the workers as the wall, and only after a second discussion.

## F5. STILL OWED BY THE OWNER

- The wall's defaults (§24.4): fourteen workers with the harvest cap at 96 (a whole band at 528 m/s,
  30 frames a second on the Mac) or ten workers (38 frames a second, a gap on one frame in three at
  528 m/s). Recommended: fourteen and 96.
- R-18, the server-stated warm lead (SL6): the measurement it waited for exists (48–103 gap frames a
  minute at 528 m/s with the delivery buffer's 63 m lead). Reopen or keep refused.

## F6. THE AVERAGE MACHINE — the terrain gets a SHARE of the cores, never all of them (owner)

**Owner, verbatim:** *"The problem of using 14 cores — not every machine has it, plus my laptop fan
is trying to take off. We are building a game that can be run on an average machine. Plus in game
those cores will need to do a bunch of other stuff — we don't have any game built yet."*

**RULED by that statement.**

1. The game targets an AVERAGE machine. The Mac's fourteen cores are the measuring instrument, not
   the target. A result that needs fourteen workers is not a result.
2. The terrain workers get a SHARE of the cores. The rest belong to the game that is not built yet
   (physics, the ship's systems, audio, the UI, the network). The pool never takes one thread per core.
3. "The best possible detail at speed" (V15) is read on the average machine: the rung the share of
   workers can deliver WHOLE, never a hole, with the finer rung fading in as it arrives.

**OWED by the owner:** the target machine in numbers (cores, memory, the GPU class) and the terrain's
core share (recommended: a quarter of the cores, at least two; on an eight-core machine, two
workers).

**OWED by the code (the next measurements and levers, in order):** the flights at the share's count
on this machine; the disk cache (V15 item 3); a wanted set BOUNDED BY THROUGHPUT — at speed the
finest ring is wanted only when the workers can deliver it inside the lead, otherwise the next
rung is drawn whole (a new rule; the owner's word under §16.3); the far-rung voxel renderer (D8-8);
R-18 the server-stated warm lead.

## F7. THE RECIPE GOES INTEGER-ONLY, AND THE GPU IS USED TO THE MAXIMUM (owner)

**Owner, verbatim:** *"Then we go integer only in the recipe, and we utilise gpu to the maximum."*

**RULED.**

1. THE RECIPE IS INTEGER-ONLY. Every step of the static shape — the hash, the noise, the octave sum,
   the face bend, the density — is computed in fixed-point integers. No float enters the recipe on
   either host. Integers give the same bytes on every CPU and every GPU; the no-drift gate (SL10)
   stays a measurement, and it now has a reason to pass by construction.
2. THE GPU DOES THE CLIENT'S CHUNK WORK TO THE MAXIMUM: the generation, the mesh extraction, and the
   ladder's per-vertex work, so a GPU-built chunk never crosses the bus as an upload and the CPU's
   cores stay with the game (F6). The server computes the same integers on the CPU for collision.
3. ONE SOURCE, TWO TARGETS (SL10 "a port is forbidden" is untouched): the generator's integer core is
   one Rust crate compiled for the CPU and for the GPU. A hand-written shader copy of the recipe is a
   port and is refused. The build path is part of the design.
4. THE PICTURES ARE JUDGED AGAIN: an integer recipe is a new world shape. The frozen exact pictures
   are re-frozen on the owner's look after the recipe lands, as on 2026-09-12.
5. Ruling V18's "GPU only after a second discussion" is satisfied by this ruling; its "the recipe
   stays 64-bit on the CPU" is superseded.

**THE ORDER.** (a) The bench: one integer noise on the CPU and on the GPU, byte for byte, and its
cost per column against today's float recipe — one day, a number before a design. (b) The design
discussion of the integer recipe and the GPU path (the fixed-point format and its range over a planet's
radius, the build path, the extraction on the GPU, the gate on every target), with the owner, before
code. (c) The build. Slice 9 (the block store) and the frozen patch (F1) follow the new recipe,
because they build on its output.

**Step (a) DONE, 2026-09-12 (`slice_08_integer_bench.md`):** 0 of 3 936 256 columns differ between
the CPU and the GPU; within 1.06 mm of today's float world; one core +18 %; the GPU 127 ms with the
transfer. Step (b), the design discussion, is next.

## F8. THE INTEGER RECIPE'S DESIGN — THE FIVE DECISIONS, RULED (owner: *"Ok, agreed. Agree with your other recommendations, please implement"*)

The design document `slice_08_integer_recipe_design.md` (step (b)) is accepted with its five
recommendations:

1. **The direction carries 40 fraction bits** in a 64-bit word (MEASURED, bench part 3: within
   0.02 mm of the float bend on every column; 0.29 ms of a chunk's build on one core; two-word
   products from 32-bit halves on both hosts). The owner's question — *"why 40 and not 64?"* —
   answered: every number IS a 64-bit integer; 40 is the binary point's position, and a product
   doubles the fraction bits, so 64 would leave no word for the whole part.
2. **The fallback for a GPU without 64-bit integers is the CPU path of the same source** (every CPU
   has 64-bit integers; about 1 % of desktop GPUs lack them in shaders); limb emulation is refused.
3. **The runtime self-check is the GPU's gate**: the client builds the eight golden chunks on its
   own GPU at start and compares them with the CPU's; a GPU that differs gets the CPU path and a
   line in the log.
4. **The order**: the CPU integer recipe → the pictures re-frozen on the owner's look → the GPU
   steps one at a time (the cell field, the extraction, the position, the morph) → slice 9.
5. **The toolchain is rust-gpu** (one crate, two compilations, a pinned nightly from a `build.rs`),
   with its WGSL transpile route in reserve; DX12 is unproven until measured on a Windows machine.

Step (c), the build, starts with the CPU integer recipe.

**Step (c) in progress, 2026-09-12.** C1 DONE: `crates/recipe` (vd-recipe) — the fenced integer
`Gi` (wrapping ops, masked shifts, no `/`, `%`, negation or abs; four compile-fail doctests), the
two-word product, the integer root and the exact reciprocal, the bend at 40 fraction bits with the
cell-count reciprocal at a whole word (a 2⁻⁵⁶ reciprocal left the face's edge cells 3 mm off — the
crate's own test caught it), the noise at 28, the octave sum; 19 unit tests + 4 doctests; lint
clean; the coverage gate PASS at 100 % of lines and branch sides; registered in the workspace,
the Tier-A list and the link scan. C2 DONE: `vd-seed` steps its hash through the recipe's
`splitmix_step` (one implementation), `child_seed` through `hash1` (the pinned vectors unchanged),
and offers `bend::direction_q` beside the float bend, tested within eight units of 2⁻⁴⁰ against
it on every face at the centre, the edges and a corner. C3 (vd-terrain on the recipe, `Gf` deleted,
pins and golden tables re-recorded) is in progress.

**C3 LANDED, 2026-09-13.** `vd-terrain` computes the whole static shape in the recipe's kernels
(the integer charter on the body, the direction, the height, the biome, the density, the caves, the
vertex position, the column bound); `Gf` keeps two callers (the once-per-body draw and the metre
doors at the seam); `GENERATOR_VERSION` is 2; the golden tables and the pins re-recorded by the
recorder. MEASURED against the float world: 1.4992 mm widest, 0.2002 mm mean on 3 936 256 columns.
An adversarial review found ten items — one red law test (the crate-isolation law still named the
seed as the root; now the recipe is), the float fence missing on the recipe crate, `/` and `%` on
the per-cell cavern lattice (now a shift and a mask; the lint `integer_division` and
`modulo_arithmetic` now DENIED in the seed and the terrain, every remaining CPU-only site carrying
its reason), a poison word from the cell-count reciprocal for a face under three cells (clamped),
a false "exact" claim on the metre door above 2⁵³ (corrected: about four nanometres of the home
planet's radius), the two rounding conventions undocumented (documented), a wrong cast in an
example, the crate missing from the workspace dependencies — all fixed and re-gated (lint clean,
the three pin legs, coverage PASS). The frozen pictures wait on the owner's look: in report mode
the near stands moved (ground 46 274 pixels / widest step 28; hill 60 022 / 49; seam 67 579 / 21),
the far stands did not (aloft 5 573 / 3; orbit 6 967 / 2) — a fine even speckle of one or two
shade levels over the ground, no shape moved.

**F6 LANDED (2026-09-13):** the terrain workers' default is the machine's SHARE — a quarter of the
cores, at least two (`vd_client_render::terrain::worker_share`; on this Mac three, on an eight-core
machine two); the knob `VD_TERRAIN_WORKERS` overrides it for a measurement. The §24.4 ceiling
numbers were taken at the whole machine and stay the instrument's, not the target's. The picture
and moving-eye harnesses launch their client at the WHOLE machine's count unless the knob names
another (MEASURED: at the share of three the ground stand settled at tick 2 389 against a capture
tick of 1 200), so the stands stay bit-exact and the knob is how the share itself is measured.

**Step 10 COMMITTED (e5689f6):** the one source on the GPU — 0 of 3 936 256 columns differ through
the recipe crate compiled to SPIR-V and loaded into Metal. The module's cost (274–352 ms for the
columns against the transcription's 56–127 ms) is the open lever; reading the octaves in place
changed nothing, so the cost sits in the compiled module itself (the next probe reads naga's MSL).

**F8 decision 3 BUILT (2026-09-13):** the client compiles the recipe's GPU shell at build time and
measures its own GPU at start against the CPU on the eight golden chunks' columns — PASSED on the
Apple M4 Pro (30752 columns, 16000 µs); a GPU without 64-bit integers or with a differing
word gets the CPU path and a log line.

**G1 LANDED (2026-09-13) — the cell field on the GPU.** The recipe's per-cell kernel
(`vd_recipe::cell`) runs on the card through the shell's `cell_field` entry point, and vd-terrain's
own cell pass calls the same function, so the shard's collision and the card's picture are one
arithmetic. MEASURED: **0 cells differ** between the CPU and the GPU — 0 of 2 097 152 over the
eight golden chunks and 0 of 268 435 456 over the bench's square of 1 024 chunks, which is 0 of
270 532 608 cells over 1 032 boxes in all — and no byte of the world moved (the three pin legs
green). The runtime self-check now runs both kernels at start: 30 752
columns and 2 097 152 cells. **G1 alone does not pay yet:** the card's own share is 1.79 ms a box,
but the host still spends 2.14 ms preparing the box's plan and a megabyte crosses back, so the whole
path is 3.93 ms a box against 1.24 ms on this machine's three-worker terrain share. The client's
builder therefore stays on the CPU workers until G2 removes the readback. Two GPU kernel faults were
found and cured on the way — a loop's value read after the loop, and an accumulator a branch
assigns, both one step stale on the card — and both are now rules in the design's §1b, with two
PROBE entry points as the instrument that names the next one in a line. A third measurement came
free: the carvers' hollow now compares SQUARED distances and pays the integer root only inside a
carver, which takes the cave-dense chunk at the eight-metre rung from 48.7 ms to 10.8 ms — 4.5
times cheaper than before this step, with no byte of the world moved.

**F6 MEASURED at the share (2026-09-13, three workers on this Mac):** walk and the slow hull whole
at 49 frames/s; 240 m/s: 551 frames with a gap (13 urgent at the worst); 528 m/s: every frame
with a gap (974 urgent at the worst, the queue at 2 187, 195 chunks/s built against the ask of
about 400). The average machine cannot hold the finest ring at speed on its CPU share; the GPU
chain (the columns, the cave grid, the cells and the extraction all on the card) and the
throughput-bounded wanted set are the two levers, both owed.

**G2-A LANDED AND MEASURED (2026-09-13) — the PLAN on the card, and the number that stops G2.** The
two passes that stood in front of the cell field moved onto the card: the COLUMN pass (a column's
direction, its surface and its biome, from its site alone) and the NODE pass (the cavern field at a
lattice node). A box's request now carries its key and its charter; the host keeps only the
TOPOLOGY — which face each column belongs to across a seam, which carvers reach the box, where the
lattices stand. `vd-terrain`'s own column pass and node lattice CALL the same recipe kernels, so
there is one arithmetic and not two. MEASURED on 1 080 boxes: **0 of 283 115 520 cells and 0 of
4 423 680 column directions differ** between the card and the host — and the bench grew a THIRD set
for this step, THE SEAMS (the four corner chunks of every face at two rungs), because neither the
golden chunks nor the square ever stands at a face's edge: all 48 of those boxes hold BOTH a
partner face's columns and a corner phantom, which are exactly the two arms the column kernel newly
carries. The three pin legs, the lint, the fence control and the link
scan are unchanged, so no byte of the world moved. **The host's share of a box fell about 38
times — 2.14 ms to about 0.06 ms — and the whole GPU path from 3.93 ms a box to about 2.0 ms.**

★ **AND THE MEASUREMENT THAT STOPS G2.** G2 (the extraction on the card) was to pay by removing the
megabyte of cells from the bus. The same part measured that megabyte by running the three passes
with nothing copied home, FIVE times: **218 ms, −17 ms, 15 ms, 106 ms and 109 ms of a whole path of
about 2 070 ms — readings that STRADDLE ZERO, so the largest is a TENTH of the path and the
measurement cannot separate the rest from noise.** ★ **AND THE CONCLUSION HOLDS AT THAT TENTH:** hand
the whole of the largest reading to G2 and the card's share is still about 1.8 ms a box against the
three-worker share's 1.26 ms — a tenth of 2.0 ms does not close 2.0 against 1.26. So the bus was
never the cost. The wall is the card's own arithmetic: about 1.9 ms a box, about 7 ns a cell,
against one CPU core's 14 ns. The card is worth about TWO of this machine's cores on a kernel built
from 64-bit integer multiplies (Metal has no instruction for one), and ruling F6's terrain share is
THREE cores, which do the same box in about 1.26 ms. G2 would add card work, not remove it.

★ **THE DIGITS ARE SOFTER THAN THEY LOOK.** Two GPU passes over the SAME 1 024 boxes in one run read
1 821 ms and 2 074 ms — 13 % apart. Every timing above is therefore quoted to two figures and the
gap is stated as ABOUT A THIRD, never as a percentage to three. The ruling stands on a gap far wider
than the spread; the third digit does not stand at all.

**SO THE CHAIN IS BUILT, PROVEN BYTE FOR BYTE, AND NOT WIRED INTO THE CLIENT.** What it would buy is
not speed but CORES, and that is the owner's call, not a measurement's:

| | chunks a second | the CPU it costs | the card it costs |
|---|---|---|---|
| the CPU workers (today, the client's path) | about 790 boxes | 3 cores of 14 | nothing |
| the GPU chain (built, measured, unwired) | about 490 boxes | about 0.03 of ONE core | about all of it at full rate |

Roughly three fifths of the throughput for under one percent of the CPU — and then the card is
busy and cannot draw the picture. **OWED BY THE OWNER:** whether F6's "the cores belong to the game
that is not built yet" is worth a third of the chunk rate and most of the card. What would change
the reading without a ruling: a hot path in 32-bit words (the noise's lattice point and fade already
fit; the bend and the radius do not), or the average machine of F6 itself, whose share is TWO
workers — there the card and the share are even.

## F9. THE BOUNDED ASK, AND THE CARD AS A BUDGETED BUILDER (owner, 2026-09-13: *"Agree with your proposal, please implement"*)

**The measurement that raised it (Step 13, F6's baseline):** on the average machine's share the
eye at 528 m/s asks for about 400 chunks a second and three CPU workers build 195; the card does
the recipe's 64-bit integer work at the speed of about two cores and, building alone, cannot draw.
Neither alone reaches the ask; together they can, if the card keeps time to draw.

**RULED.**

1. **THE THROUGHPUT-BOUNDED ASK** (the rule §16.3 left to the owner's word): the client measures
   its builders' throughput as it goes and asks for the finest ring only as far ahead as the
   builders can deliver it before the ground reaches the screen; beyond that it asks for the next
   rung, which stands whole, and the ladder's crossfade blends the finer rung in as it lands. The
   bound never binds where the builders deliver the whole ask (a strong machine, a walk, a slow
   approach); it binds only at speed on a small share. A hole and a pop stay defects: the bound
   trades detail the eye cannot see at speed for the holes and late pops the unbounded ask makes.
   The owner's question — *"will the seamless experience degrade?"* — answered: the detail at
   speed drops on a weak machine; the seamlessness does not; completeness first (V15 read under
   SL8).
2. **THE CARD AS A SECOND BUILDER WITH A TIME BUDGET**: the GPU chain (built and proven in Step
   13) is wired into the client beside the CPU share, taking chunks from the same wanted list,
   under a fixed slice of each frame (a knob; a quarter by default) so the card keeps the rest to
   draw; the self-check decides at start whether the card may build at all; the CPU share is the
   default and the fallback.
3. **THE MESH EXTRACTION ON THE CARD (G2) IS SKIPPED** until a measurement changes the numbers
   (a card with fast 64-bit integers, or a 32-bit hot path).
4. **THE ORDER**: the bounded ask first (it helps every machine), measured on the three-worker
   flight (the pop detector's widest step must not rise; the band's completeness at the DRAWN rung
   must reach every frame); then the card as a budgeted builder, its budget a knob, the same flight
   the judge; then slice 9.


**F9 ITEM 1 — THE THROUGHPUT-BOUNDED ASK — BUILT AND FLOWN (2026-09-13).** The client measures its
builders' CAPACITY (the worker count over the mean wall time of a build, smoothed over ten seconds —
a capacity, never the chunks they happened to finish, which on a walk is the ask and not the
ceiling) and the eye's own speed (the PEAK of the lead's metres over the interpolation buffer's
seconds — two DELIVERED poses, SL10 clause 7), and gives every rung a DELIVERABLE HORIZON: its
EFFECTIVE SWITCH DISTANCE (`vd_client::ladder_view::AskBound`). The descent's split, the rung's
territory, the crossfade's bands and the sink's ramp all read that ONE number, so a column inside
the horizon is still asked at the tier rule's rung, one beyond it is asked at the NEXT rung whose
territory now reaches in, and the handover is the ladder's own crossfade and never a cut. The
horizon SLIDES (a quarter of its own length a second, after the cure below) instead of jumping,
because a jumped band moves every
chunk in it in one frame. The bound never binds where the builders cover the ask — a still stand, a
walk, a strong machine, an empty ring — and `VD_TERRAIN_BOUND=0` switches it off.

**THE JUDGE, two flights of one binary at the average machine's three workers** (§25.5):

| the 528 m/s leg | bound OFF | bound ON |
|---|---|---|
| the band's worst gap, at the DRAWN rung | 981 urgent chunks | **449** |
| the queue's peak | 2 199 | **1 611** |
| the pop detector's widest step | 48 levels | **47** |
| frames a second | 43.5 | **39.1** |

✅ The band's completeness at the drawn rung improves by more than half at 528 m/s, and ✅ the pop
detector's widest step does not rise (both readings sit inside the leg's own 47-to-54 range over
seven earlier flights). 🟥 **THE FRAMES FELL, 43.5 to 39.1**, three times measured — **and the cause
was then MEASURED and CURED (2026-09-14, §25.7 and §25.8).** The flight now times each piece the
bound added: the throughput read, the bound's arithmetic and the materials' rewrite cost 0.000 to
0.003 ms a frame, and **every millisecond was the wanted set's DESCENT** — 9.7 ms a frame unbounded
against 18.1 bounded, because the horizon SLID every frame and a slid horizon forced a descent for
every body in the window. The cure separates the two rates: the drawn bands still slide every frame
(a jumped band is a pop), the DESCENT re-runs only when the horizon leaves the ring it last asked
for, and it asks that ring WIDER by the same fraction — but only at the rungs the bound actually
moved, because widening a rung that never slides costs every column of the widening. **RE-FLOWN, the
same pair on one binary: 45.0 frames a second with the bound on against 43.8 with it off, the
descent 8.565 ms against 9.805 — the bounded ask is now FASTER than the unbounded one — with the
worst gap 401 against 974 and the queue 1 768 against 2 189.** 🟨 The pop detector reads 43 against
that pair's OFF value of 29 on the 528 m/s leg, where four legs of five improved; 29 is the lowest
of the eight readings that leg has produced (54, 50, 48, 48, 47, 43, 35, 29) and a second pair is
owed. 🟥 The 240 m/s leg still shows 540 frames with a gap (14 urgent
chunks at the worst against the unbounded pair's 11 on 449), but the bound is INERT there — 0 of
1 227 samples — so that residue is the ask's TIMING, which §24.4 named and no bound cures. The still stands are untouched (the picture gate: 0 to 18 content pixels
differing at a widest step of one, inside the gate's own noise), so the frozen pictures stand.

★★ **AND AN ADVERSARIAL REVIEW THEN FOUND THAT THE FEATURE NEVER BOUND ABOVE 54 FRAMES A SECOND**
(2026-09-14, cured; §25.9). The horizon's hysteresis gated the PER-FRAME STEP instead of the
distance from the target, and a step is proportional to the frame's own seconds: below 18.7 ms a
frame every step was refused and the horizon never left the tier rule's own radii at all. On a fast
machine the bounded ask did nothing, and the flights that judged it ran at 39 to 47 frames a second
— under the cliff by luck. The gate now reads the DISTANCE, so the horizon arrives in the same wall
time at any frame rate. **THE CURE THAT MATTERS IS WHERE THE CODE LIVES:** the pace had grown in the
Tier-B render crate, which no test reaches. It is now `vd_client::ask_pace` in the Tier-A library at
100 % coverage, and the render crate only wires it. Three more defects fell out of the move — the
speed's "peak hold" was a decay that still read 194 m/s a second after a hull stopped dead from 528;
the descent's slack was the bracket alone, where the two tolerances together need 1.170 against an
asked edge of 1.10, so it is now DERIVED and asserted at compile time; and a reading after a long
idle threw the capacity's ten-second window away. Each is now a unit test.

★ **THE SECOND PAIR, FLOWN ON THE CURED CODE (2026-09-14, §25.10): EVERY BAR OF F9 IS GREEN.** One
binary, three workers, the bound ON against OFF at 528 m/s — **frames 45.6 against 43.3** (the
bounded ask is FASTER than the unbounded one), the worst gap at the DRAWN rung **207 against 972**,
the queue **1 512 against 2 194**, the descent **7.695 ms a frame against 10.345**. The TURNING leg
holds the band on EVERY frame with the bound on, against 102 urgent chunks at the worst without it.
And the pop detector, which one pair had left open, now IMPROVES on four legs of five: 528 m/s 37
against 40, 240 m/s 40 against 47, turning 44 against 48, the walk 17 against 18. The one leg that
reads higher is the slow hull, whose unbounded run entered the leg still filling the ring from the
walk (a 3 710-chunk gap and a 6 519-deep queue). The bound does not widen the pop.

★ **F9 ITEM 2, THE CARD AS A BUDGETED SECOND BUILDER — BUILT, REVIEWED, RE-FLOWN, AND SHIPPED AS
THE DEFAULT (2026-09-14; `slice_08_ladder_discussion.md` §26).**

**THE TWO SEAMS WERE MEASURED FIRST, as the ruling's order asks** (`just gpu-seam`). A worker thread
that submits the recipe's box chain to the RENDERER'S OWN device and waits for it costs the renderer
nothing: **52.2 frames a second while it builds against 51.5 while it is quiet, the SAME 20.9 ms
worst single frame, and 0 stalls over 9 936 boxes.** Neither seam fails.

★★ **AND THE MEASUREMENT THAT CHANGED THE WHOLE PICTURE: A BOX COSTS THE CARD 0.08 ms, NOT 1.8 ms.**
The first writing timed the box with the host's wall clock around the submit, because no timestamp
query existed; the device's own clock says the three compute passes take an eightieth of a
millisecond. **The 1.8 ms is the ROUND TRIP** — the submit, the wait, the map and the read back. So
the budget had been rationing the wrong seconds, and the bench's old reading of "the card is worth
about two of this machine's cores" was a reading of LATENCY, not of the card's arithmetic.

**THE BUILDER.** The card takes chunks from the SAME wanted list by the SAME priority as the CPU
workers, LEAVING THEM THE SINGLE MOST URGENT ONE; it samples the box on the card in one stage
(plan, upload, submit, wait, read back — over pooled buffers and bind groups that are grown and
never rebuilt) and hands the raw bytes to a SECOND stage that decodes them and runs the SAME
geometry step and the same harvest. Its budget rations THE CARD'S OWN measured seconds
(`vd_client::card_budget::CardBudget`, Tier-A at 100 % coverage), at the rate the anti-banking cap
really allows. A dispatch that fails DETACHES the card: its capacity goes to zero, the job goes back
to the queue, the CPU share carries the ladder and the client goes on drawing.

**THE JUDGE — five flights of one binary at the average machine's three workers, the 528 m/s leg.**

| | the card OFF | quarter, capacity SUMMED | quarter, NOT summed, skip 1 | skip 0 | the WHOLE frame |
|---|---|---|---|---|---|
| frames a second | 45.8 | 43.3 | **45.7** | 45.8 | 46.0 |
| the band's worst gap | 125 | 442 | **87** | 113 | 84 |
| the queue's peak | 1 442 | 1 900 | **529** | 615 | 486 |
| the pop's widest step | 44 | 38 | 46 | — | 41 |
| the engine harvested | 204 chunks/s | 250 | **256** | — | 257 |
| the card built | — | 84 boxes/s | **95 boxes/s (36 %)** | 91 | 95 |

★ **TWO CAUSES, ISOLATED BY THE TWO ABLATIONS, AND BOTH REAL.**
1. **THE BOUND SPENDS THE CARD'S CHUNKS ON A WIDER RING.** Summed into the bounded ask, the card's
   capacity takes the builders' reading from 177–219 chunks a second to 222–445 and pulls the finest
   ring's horizon from 405 m to 569 m — and the band goes MORE incomplete, not less. The capacity is
   measured and stated, and NOT summed (`VD_TERRAIN_GPU_BOUND=1` sums it).
2. **HEAD-OF-LINE BLOCKING IS REAL, AND SMALLER.** A card that takes the single most urgent request
   reads 113 urgent chunks at 528 m/s and 65 at 240 m/s; one that leaves it to the workers reads 87
   and 10. The card holds a chunk for a round trip where a worker finishes it in arithmetic. ★ And
   LEAVING MORE is worse: a card that skips the whole worker count reads 97 at 528 m/s, 58 turning,
   and 3 394 on the slow hull's leg against a band that held — it stands down wherever the CPU share
   keeps up, so those legs read what no card reads. **The shipped rule is ONE.**

✅ **THE BARS.** The worst gap falls 125 → **87** (Step 14's bar was 207); the queue 1 442 → **529**
(the bar was 1 512); and THE SLOW HULL'S BAND NOW HOLDS ON EVERY FRAME, against 3 714 urgent chunks
and a 6 523-deep queue without the card. 🟨 Two readings are not strictly better and both sit inside
this leg's own spread: the frames 45.7 against 45.8 (the leg's eight readings span 43.3 to 45.8) and
the pop's widest step 46 against 44 (they span 29 to 54). 🟥 The turning leg's gap rises 14 → 50.

★ **AND THE PICTURE GATE MEASURED THE OTHER HALF: THE CARD COSTS A STILL STAND.** With the card
building, the hill stand's terrain settles at tick **2 408** against **2 081** with no card — past
that stand's own capture tick, so the gate is RED. It is the same round trip that pays at speed: a
builder that holds a chunk for 1.8 ms of submit-and-wait helps a queue of hundreds and hurts a queue
of three, and the tail of a settle is a queue of three.

**SO THE CARD SHIPS AS A KNOB (`VD_TERRAIN_GPU=1`), NOT AS THE DEFAULT** — not because it fails, but
because it is now measured BOTH WAYS and a shipped default must be right for both. The shipped path
(the card off) is green on every gate. ⚠ **OWED BY THE OWNER:** whether a third of the 528 m/s leg's
gap and two thirds of its queue are worth eighteen seconds of a still stand's settle. **OWED BY THE
CODE:** the card standing down where the queue is short (the cure the measurement points at, and
unmeasured); the turning leg; TWO BOXES IN FLIGHT, now that the round trip is known to be twenty
times the card's own arithmetic; and whether a DAMPED sum of the card's capacity beats no sum.