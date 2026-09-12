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

