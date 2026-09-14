# The integer recipe on the GPU — the design for the discussion (ruling F7 step (b), 2026-09-12)

**What this is.** The owner ruled (F7): the recipe goes integer-only, and the GPU does the client's
chunk work to the maximum, one source compiled for both targets; (F6): the terrain gets a share of
the cores, never all of them, on an average machine. Step (a), the bench, measured that an integer
recipe is identical on the CPU and the GPU and keeps today's world within a millimetre
(`slice_08_integer_bench.md`). This document is step (b): the design, each part with a
recommendation and the decision it needs from the owner. Nothing here is built.

**The laws it stands under.** SL10 (one generator, two hosts, no drift MEASURED on every target;
the server computes collision on the same shape); SL5 (one world); the seed ruling (the static
shape is safe to publish); F6; F7. The vocabulary: a *column* is one cell's direction from the
body's centre; a *chunk* is 62 × 62 × 62 cells with a one-cell halo; the *rung* is the detail
level, a coarser rung drops the finest octaves; the *gap byte* is the signed distance to the
surface in 1/128 of a cell; the *extractor* turns gap bytes into a mesh; the *morph* is the
metre each vertex sinks toward the coarser rung's surface during a crossfade.

## 1. The integer recipe, stage by stage

Every stage below runs on 64-bit integers with a fixed number of fraction bits, every product
shifted back once, every intermediate under 2⁶³, and — after part 2 of the bench — **no division
operator anywhere** (naga's Metal back end cannot compile one; a recipe without `/` runs on every
back end and needs no argument about it).

| stage | today | the integer form | fraction bits | measured |
|---|---|---|---|---|
| the face parameter `(2i+1)/n_l − 1` | one float division | `(2i+1) × inv_n >> 16`, `inv_n = floor(2⁵⁶ / n_l)` computed ONCE per body on the CPU and stored in the body's integers (the charter) | 40 | identical |
| the bend `a(k₁ + a²(k₂ + a²k₃))` | `k₁ = π/4` in f64 | `k₁ = round(π/4 · 2⁴⁰)`, `k₂ = round(0.15 · 2⁴⁰)`, `k₃ = 2⁴⁰ − k₁ − k₂` (the sum is exactly one, a face edge lands on the cube edge exactly); the products in two words | 40 | identical |
| the basis `n + W(a)u + W(b)v` | floats | integers; the axes are −1, 0, 1 | 40 | identical |
| the normalise `v / |v|` | one sqrt, three divisions | the square sum at 80 bits in two words; a 30-bit seed (an integer square root of the top word, its reciprocal landed exactly by two compare loops); ONE Newton step of the reciprocal square root with the residual at 120 bits; three two-word multiplies | 40 | identical; within 0.02 mm of the float bend; 77 ns a column |
| the lattice point `dir × frequency` | one float multiply | `dir × f_int + (dir × f_frac >> 28)`, the frequency stored as an integer part and a 28-bit fraction | 28 | identical, the frequency must be this exact |
| the noise: hash, gradients, fade, blend | `Gf` floats | as the bench: the hash unchanged, the fade and the blend at 28 bits | 28 | identical; within 0.13 mm mean |
| the octave sum | float adds | summed at 28 bits with the amplitude at 1/32 768 m, floored ONCE | 28 | identical; widest 1.06 mm |
| the value noise (caves) | one division by 2⁵³ | a shift | 28 | not benched; a shift cannot differ |
| the tube distance (caves) | a division and a sqrt | the squared distance compared against squared thresholds where only the sign matters; one integer square root where the metres matter | 28 | not benched |
| the radius and the density | `r − h` in floats, floored to 1/128 cell | the radius in 1/128 m (33 bits at the largest legal body, 30 on the home planet), the surface in the same unit, one subtraction, one floor to the gap byte | 7 (the gap step) | by construction |
| the vertex position (`position.rs`) | a trilinear blend of cell centres, sorted by site | the cell centres are integers (radius steps × direction); the blend weights are the extractor's own quanta (1/256 cell); one integer sum in site order | 30 + 8 | to bench with the extraction |
| the ladder | drops octaves | unchanged | — | by construction |

**The world's shape changes** — by less than the picture gate's own noise on the columns (a
millimetre), and by two hundredths of a millimetre laterally at 40 bits. The pins
(`NOISE_PIN`, `VALUE_PIN`), the golden tables and the frozen exact pictures are re-recorded once,
on the owner's look (F7 item 4). Example: the hill stand's crease stays where it is to the eye; the
exact picture behind it moves by a pixel level or two at the crease, and the new picture becomes
the reference.

**Decision 1 (the owner): the direction's fraction bits — MEASURED at 30 and at 40 (bench part
3).** 30 bits: 5.9 mm steps on the home planet, 40 mm on the largest legal body (radius
42 700 km); 44 ns a column. 40 bits: 0.006 mm and 0.04 mm; within 0.02 mm of the float bend on
every column; 77 ns a column, 0.29 ms of a chunk's 5 ms build on one core (0.17 ms at 30 bits,
0.02 ms for the float bend), nothing on the GPU; a two-word product in four places, spelled out
from 32-bit halves on both hosts. Recommended: **40** — a hundred times finer than the surface
the gap byte stores, on every body, for two percent of a chunk's build.

## 2. What runs on the GPU, in which order ("to the maximum")

A chunk's work today, on a CPU worker (§16.5, §17): the columns (0.4–0.6 ms of a 5 ms build), the
cell field (the 62³ gap bytes with the caves — the bulk), the extraction (surface nets, pure
integer), the vertex position, the morph metre (a ray per vertex against the parent's mesh), the
normals, the upload (0.07 ms of the main thread per chunk, plus the bytes over the bus).

The GPU form: ONE compute pass per chunk (or per batch of chunks) that produces the chunk's mesh
IN A GPU BUFFER, so nothing crosses the bus but the request (a chunk key) and the answer's size.

| step | what moves to the GPU | why this order |
|---|---|---|
| G1 ✅ **DONE 2026-09-13** | the CELL FIELD: the substance and the gap of every cell of a 64³ box, one invocation a cell, through the recipe's own `cell::cell_word` (the shell's `cell_field` entry point) | the bulk of the build; MEASURED byte for byte against the CPU's `sample_box` — **0 of 270 532 608 cells differ**: 0 of 2 097 152 over the eight golden chunks and 0 of 268 435 456 over the bench's square of 1 024 chunks (`slice_08_integer_bench.md` part 5) |
| G2-A ✅ **DONE 2026-09-13** | the PLAN: the COLUMN pass (a column's direction, surface and biome, from its site) and the NODE pass (the cavern field at a lattice node), so a box's request carries its key and its charter and the host keeps only the TOPOLOGY | the host's own share was 2.14 ms of a 3.93 ms box, almost all of it arithmetic; MEASURED byte for byte over **1 080 boxes** — the eight golden chunks, the square's 1 024, AND THE 48 SEAM BOXES this step added — **0 of 283 115 520 cells and 0 of 4 423 680 column directions differ**; the host's plan fell to **about 0.06 ms a box** (bench part 6) |
| G2 🟥 **STOPPED, MEASURED** | the extraction: surface nets on the box, into a vertex list and an index list | pure integer already; the data-dependent counts need a prefix sum (one workgroup pass), a standard shape. ★ **ITS OWN LEVER IS GONE:** G2 was to pay by removing the megabyte of cells from the bus, and part 6 MEASURED that megabyte over five runs — the readings straddle zero and the LARGEST is a tenth of the path. The card's own arithmetic is the wall (about 1.9 ms a box against the three-worker share's 1.26 ms), and even taking the largest readback reading the card's share would still stand about a third above that share; extracting on the card ADDS card work besides. See part 6's own reading |
| F9-2 ✅ **DONE 2026-09-14** | THE CLIENT'S BUILDER: the card takes chunks from the SAME wanted list by the SAME priority as the CPU workers — leaving the single most urgent one to them — samples the box on the card in ONE stage and hands the raw bytes to a SECOND stage that runs the SAME geometry step (`CardSeam` in `vd_client_render::terrain`; the pooled `BoxGear` in `gpu_check`; the budget's arithmetic in `vd_client::card_budget`, Tier-A) | the two client seams were MEASURED first (`just gpu-seam`): a worker thread submitting to the renderer's own device and waiting for it costs the renderer NOTHING — 52.2 frames a second against 51.5 quiet, the same 20.9 ms worst frame, 0 stalls over 9 936 boxes. ★ AND THE CARD'S OWN CLOCK SAYS THE BOX COSTS **0.08 ms**, not the 1.8 ms the host's wall clock reads: that 1.8 ms is the ROUND TRIP, and the earlier "the card is worth two cores" was a reading of latency. THE JUDGE, five flights of one binary at three workers, the 528 m/s leg: with the card's capacity NOT summed into the bounded ask and the card skipping the most urgent request, the worst gap falls **125 → 87**, the queue **1 442 → 529**, the frames hold 45.8 → 45.7 — and summing the capacity instead reads 442 and 1 900. ★ AND THE PICTURE GATE MEASURED THE OTHER HALF: the card DELAYS A STILL STAND'S SETTLE (the hill stand settles at tick 2 408 against 2 081 with no card, past its own capture tick), because the round trip that pays on a queue of hundreds is latency on a queue of three. So the card ships as a KNOB (`VD_TERRAIN_GPU=1`), not as the default — `slice_08_ladder_discussion.md` §26 |
| G3 | the vertex position and the normals | integers; per vertex; the site-order sum is a fixed order, so no atomics |
| G4 | the morph metre and the morph normal | per vertex, a ray against the parent's mesh, which is ALREADY in a GPU buffer once G1–G3 run for the parent |
| G5 | the ladder's per-vertex work | already on the GPU (the shaders); unchanged |

What stays on the CPU on the client: the wanted set, the queue, the ladder's bookkeeping, the
digest of a chunk (read back only for the gate, below), and the whole path as a FALLBACK on a GPU
without 64-bit integers (the same source, F6's share of the cores).

★ **AND THE WANTED SET IS NOW BOUNDED BY THROUGHPUT (ruling F9 item 1, LANDED 2026-09-13).** Whatever
builds the chunks, the client no longer asks for more of them than its builders can deliver before
the ground reaches the screen. It measures its builders' CAPACITY (the worker count over the mean
wall time of a build, smoothed over three seconds) and its own speed (the lead's metres over the
interpolation buffer's seconds — two DELIVERED poses, SL10 clause 7), and gives every rung a
DELIVERABLE HORIZON: the rung's effective switch distance
(`vd_client::ladder_view::AskBound`). Inside a rung's horizon the tier rule's own rung is asked;
beyond it the NEXT rung, whose territory now reaches in to the horizon, stands whole — and because
the crossfade's bands and the sink's ramp read the SAME effective switch distances, the handover is
the ladder's own crossfade and never a cut. The horizon SLIDES toward what the measurement asks for
(half its own length a second) instead of jumping, because a jumped band is a pop. The bound never
binds where the builders cover the ask: a still stand, a walk, a strong machine and an empty ring
all read the tier rule's own radii, which is why the frozen pictures are untouched.
`VD_TERRAIN_BOUND=0` switches it off for the comparison flight. The design and both flights:
`slice_08_ladder_discussion.md` §25.

★ **G2-A's MEASUREMENT, AND WHAT IT SAYS ABOUT THE REST OF THE ORDER (2026-09-13, bench part 6).**
The plan moved to the card and the host's share fell about 38 times — 2.14 ms a box to about 0.06 ms —
with 0 of 283 115 520 cells and 0 of 4 423 680 column directions differing over 1 080 boxes (the 48
seam boxes included) and the three pin legs green. The whole GPU path is now about **2.0 ms a box**
against 3.93 ms at G1. But the SAME part measured the readback directly, by running the three passes
with nothing copied home, five times: **218 ms, −17 ms, 15 ms, 106 ms and 109 ms of a whole path of
about 2 070 ms — readings that straddle zero, so the largest is a TENTH of the path and the
measurement cannot separate the rest from noise.** **AND THE CONCLUSION HOLDS AT THAT TENTH:** take
the largest reading at face value and the card's share is still about 1.8 ms a box against the
three-worker share's 1.26 ms. So the bus was never the cost, and G2's own lever cannot close the
gap on this machine. The wall is the card's arithmetic: about 1.9 ms a box, about 7 ns a cell,
against one CPU core's 14 ns — the card is worth about two of this machine's cores on a kernel
built from 64-bit integer multiplies, which Metal has no instruction for. The terrain's share is
THREE cores, and three cores win.

★ **THE DIGITS ARE SOFTER THAN THEY LOOK.** Two GPU passes over the SAME 1 024 boxes in one run
read 1 821 ms and 2 074 ms — 13 % apart — and the readback readings span 218 ms. So every timing
here is quoted to two figures and the gap is stated as ABOUT A THIRD, not as 37 %. The direction of
the ruling stands on a gap far wider than the spread; the third digit does not stand at all.

What the chain does buy is the CPU: about 490 chunks a second for about 0.03 of one core, against the workers' about 790
for three cores — and then the card is busy and cannot draw. That is ruling F6's trade stated in
numbers, and it is an OWNER decision, not a measurement that says "go". **So G2, G3 and G4 do not
start, and the client's builder stays on the CPU workers.** What would change the reading: a kernel
whose hot arithmetic is 32-bit (the noise's lattice point and fade already fit 32 bits; the bend and
the radius do not), or a machine whose CPU share is two workers rather than three — on the average
machine of F6 the card and the share are even.

**G1's own measurement, and what it says about the order (2026-09-13).** The card computes a box
in 1.79 ms with the upload and the 1 MB readback; the host still spends 2.14 ms on the box's PLAN
(the column pass, the cavern nodes, the carvers, the lattice's topology), so the whole path costs
3.93 ms a box against the terrain share's own 1.24 ms on three workers. **G1 alone does not pay.**
It is a correctness landing and the ground G2 stands on: G2 removes the readback (the mesh never
leaves the card), and the columns and the nodes moving up removes the plan. So the client's chunk
builder still runs on the CPU workers, and the GPU path is not wired into the worker pool yet.
**The wiring, when its measurement earns it:** split `vd_client::chunks::geometry_with` into a
SAMPLE step and a GEOMETRY step (it calls `sample_box` at its first line and never again), give
`ThreadedWorkers::start` an optional box source holding the render device, the queue and the
`cell_field` pipeline (all three are `Send + Sync`), and let each worker call
`vd_terrain::gpu::plan` → the card → `BoxPlan::box_of` → the same geometry step. The two seams
that need care and that this step did NOT prove: a worker thread submitting to the device the
renderer draws with, and `device.poll(wait)` from a worker waiting on the renderer's own
submissions.

What stays on the CPU on the server: everything — collision needs the gap bytes in the shard's
memory, and a server has no GPU. The same source, the same bytes.

**Decision 2 (the owner): the fallback policy for a GPU without 64-bit integers.** The bench
needs `SHADER_INT64`, which about 99 % of desktop and laptop GPUs in use offer (§4). (a) The CPU
path of the same source, with F6's share of the cores — the average machine of ten years ago
still plays, at the CPU's rate. (b) A 32-bit-pair emulation of the 64-bit integers in the GPU
source — every machine gets the GPU path, at sixteen to twenty operations per multiply, and the
Rust side must use the same pair type to stay one source. Recommended: **(a)**; (b) is refused.

## 3. The no-drift gate on the GPU — measured per machine, at runtime

SL10 says no drift is a MEASUREMENT on every target. A GPU is a target the build machine never
sees. So the client MEASURES ITS OWN GPU before trusting it: at start, it builds the eight golden
self-check chunks (`GOLDEN_SELF_CHECK_KEYS`, the same ones the handshake's world identity folds)
on the GPU, reads their digests back, and compares them with the CPU's. Equal: the GPU path is
on. Different: the GPU path is off for this machine, the CPU path runs, and the client says so in
its stamp (a measured refusal, never a silent one). Example: a laptop whose driver miscompiles a
64-bit shift builds the eight chunks wrong at start, and the player gets the CPU path and a line
in the log — never a hill the server disagrees with. The build-time gates stay: the golden tables
on every CPU target, the link scan, the fence control.

**Decision 3 (the owner): the runtime self-check as the GPU's gate.** Recommended: yes, and its
cost at start is eight chunks, under a tenth of a second.

★ **GROWN TO THE CELL FIELD (2026-09-13, step G1):** the check now runs BOTH kernels on the eight
golden chunks — the octave sum over their 30 752 columns, and the CELL FIELD over their 2 097 152
cells — and the resource and the log line carry both counts. One weakness, stated rather than
hidden: the tube carvers reach NONE of the eight golden boxes (MEASURED — the bench counts the
boxes holding a carver of a real radius: 0 of the eight, 178 of the square's 1 024), so the runtime
check does not exercise the carver kernel. The bench's square does, it is the build-time gate that
found the carvers' fault, and it now goes RED if that count ever falls to zero.

★ **A SECOND WEAKNESS OF THE SAME SHAPE, NAMED BY STEP G2-A:** neither the eight golden chunks nor
the square stands at a FACE'S EDGE, so neither box ever holds a column of a PARTNER face or a CORNER
PHANTOM — exactly the two arms the column kernel newly carries. The bench grew a THIRD set for it
(part 6, "the seams": the four corner chunks of every face at rung 0 and at the coarsest rung), and
that part goes red if no box of it crosses. The RUNTIME check still folds the golden chunks alone,
because that set is the world identity's and not this step's to change; a card that miscompiled the
phantom's normalise would be caught by the build-time gate and not at start. **OWED:** whether the
runtime self-check should carry a seam chunk of its own — an owner question, because the golden set
is what the handshake folds.

**BUILT and MEASURED, 2026-09-13 (the column kernel; the cell field joined it the same day).** The client's build script (`crates/client-render/build.rs`)
compiles the recipe's GPU shell to SPIR-V with cargo-gpu into the build's own output directory
(no binary in the tree); `vd_client_render::gpu_check` runs that module on the columns of the
eight golden chunks at start, through Bevy's own render device, and compares every word with the
CPU's call of the same function; the verdict is a resource and a log line. In the picture gate's
client on the Apple M4 Pro: `GPU RECIPE SELF-CHECK PASSED`, 30752 columns, 16000 µs. A
GPU without 64-bit integers is reported as such and the CPU path runs; a differing word is
counted and named. The kernel is the column height today; the cell field (G1) replaces it with
the golden chunks' whole digests when it lands.

## 4. The build path — one source, two targets (from the research of 2026-09-12)

**The facts** (each read in the pinned sources: naga 27.0.3, wgpu 27.0.1, Bevy 0.18.1, the rust-gpu
repository; the dates are the sources' own).

- **rust-gpu** (`rustc_codegen_spirv`, `spirv-builder`, `spirv-std`) compiles a Rust crate to
  SPIR-V. The newest release is `spirv-builder 0.10.0-alpha.1` (2026-04-17), pinned to one exact
  nightly (`nightly-2026-04-11`; `main` pins 2026-05-22); the previous release was 2023. It is a
  community project since Embark handed it over (2024-08-12), active on `main`. A crate can be
  BOTH a plain dependency (the server, stable 1.94.1) and a GPU shader (`#![cfg_attr(target_arch =
  "spirv", no_std)]`); the repository's own compute example does exactly this, with a doctest that
  runs the kernel on the CPU. `cargo-gpu` (now inside the rust-gpu repository) installs the
  nightly and the back end from a `build.rs`, so the workspace stays on stable — the same shape as
  the coverage gate's pinned nightly. 64-bit integers are supported (`Capability::Int64`, declared
  at build). Gotchas read in the source: a panic is a silent wrong value (an endless loop the
  back ends then bound); `checked_mul` is unsupported; debug builds insert overflow checks, so
  every operation must be `wrapping_*`; no integer square root intrinsic; macOS is "secondary"
  support in the book (the SPIR-V goes through naga to Metal, not through MoltenVK, so that row
  does not bind us).
- **wgpu 27 ingests SPIR-V through naga**: `ShaderSource::SpirV` (feature `spirv`) → naga's
  SPIR-V front end (which lists `Int64` among its capabilities) → MSL (`long`), HLSL (`…L`
  literals), SPIR-V. `Features::SHADER_INT64` sets naga's capability; Metal requires family
  Apple3 or Metal 3 and MSL 2.3; DX12 requires shader model 6.0 with `Int64ShaderOps` (DXC).
  Bevy 0.18 loads `.spv` through `shader_format_spirv`; `naga_oil` cannot preprocess SPIR-V (no
  `#import`, no shader defs), which the generator does not need. rust-gpu also offers a
  `spirv-unknown-naga-wgsl` target that transpiles to WGSL in-process — a second delivery route.
- **naga's 64-bit defects, dated**: wgpu #6081 (open since 2024-08-05) — a compute shader with
  `SHADER_INT64` loses the DX12 device; #7109 (open) — 64-bit negation and `abs` are emitted
  unguarded in HLSL (the MSL writer guards them); the SPIR-V front end maps logical and
  arithmetic right shifts to one operator and picks by the operand's signedness (Rust emits a
  logical shift only on unsigned types, so they agree — a coincidence, not a guarantee); no
  widening multiply or add-with-carry exists in any naga front or back end; and the 64-bit `/`
  guard on Metal is the ambiguity the bench met.
- **64-bit integers on shipping GPUs**: Vulkan (gpuinfo, 2026-09-12): Windows 95.6 %, Linux
  95.8 %, macOS 95.2 % of device reports, zero NVIDIA misses; Metal: every Apple-silicon Mac
  (family Apple3 is the bar; M1 is Apple7); DX12 (the info database): 99.3 % of 1 508 reports. The
  named misses: Intel Ivy Bridge and Haswell iGPUs, NVIDIA Fermi, an Adreno on Windows-on-Arm,
  SwiftShader, and most Android parts. **About 99 % of desktop and laptop GPUs in use.**
- **The alternatives**: CubeCL (Rust → WGSL/CUDA, a CPU runtime through an LLVM JIT; 64-bit
  integers undocumented; no bit-identity gate); the same WGSL run on the server through a
  software Vulkan driver (lavapipe: reports `shaderInt64`; but core WGSL has no 64-bit integers,
  only naga's extension, and the server would carry a Vulkan loader and LLVM); naga's WGSL → Rust
  back end (early, "expect incorrect behaviors"); Slang (mature, but the recipe leaves Rust);
  krnl (stale); Rust CUDA (CUDA only). No production WGSL interpreter for the CPU exists.
- **Limb emulation** (32-bit pairs) for GPUs without 64-bit integers: WGSL has no widening
  multiply, so a 32 × 32 → 64 multiply is 16–20 operations and a division a loop of hundreds; and
  the Rust side must hold the same pair type to stay one source.
- **The integer divergences to fence**: a shift by 64 or more (WGSL masks, SPIR-V is poison,
  HLSL masks, Rust panics); division and remainder (WGSL defines `x/0 = x`, SPIR-V undefined,
  Rust panics); wrapping (the GPU wraps, Rust panics in debug); `abs(MIN)` and `-MIN` (WGSL
  identity, SPIR-V unspecified, naga's HLSL guard missing at 64 bits). An arithmetic right shift
  of a negative value agrees everywhere.

**The design's answer.**

1. **rust-gpu, one crate, two compilations.** The recipe's integer core moves to a `no_std`
   crate (`vd-recipe`, below `vd-seed`) that the server and the client's CPU path take as a plain
   dependency, and the client's `build.rs` compiles to SPIR-V through `cargo-gpu` with its pinned
   nightly. The same functions, the same tests (the doctest pattern proves both compilations in
   one place). The hand-written WGSL of the bench is deleted when this lands.
1b. **THE KERNEL RULES THE GPU COMPILER SET (MEASURED, 2026-09-13):** an index loop, never a
   slice iterator; no runtime-length slice of a local array (a fixed-size table form instead); no
   8-bit integer (the tables are 32-bit words); no 128-bit word (the once-per-body reciprocal is
   gated off the GPU target); and NO CONST ARRAY INDEXED AT RUNTIME — the compiler copies the whole
   table into private memory per use (the gradient table cost the module eight times its due; a
   `match` over the draws is the form both hosts like).
   ★ **THREE MORE, from step G1 (MEASURED, 2026-09-13; `slice_08_integer_bench.md` part 5):**
   (a) **A LOOP MAY CARRY A VALUE OUT ONLY BY ADDING TO IT.** An accumulator a BRANCH assigns, or
   any value read after a loop whose body rewrites it, comes back ONE STEP STALE: rust-gpu carries
   it in a word spilled at the TOP of the body. `isqrt` read 0 for `isqrt(1)` in three different
   loop shapes; the carvers' `best = greater(best, open)` read the hollow back as zero. The cures:
   thirty-two steps written out for the root, `best += if open > best { open − best } else { ZERO }`
   for the carvers. The octave sum's `h += …` was right all along, which is what named the shape.
   (b) **NO INDEX LOOP OVER A FIXED, TINY COUNT** — the tube's three axes are written out, because
   a loop over them makes the two offset triples local arrays the shader indexes at run time.
   (c) **NO DERIVED `PartialOrd`** on a fenced word: a derived `a < b` goes through `partial_cmp`,
   whose `Ordering` is an EIGHT-BIT word, and the back end refuses the module ("`i8` type used
   without `OpCapability Int8`"). `Gi` spells its four comparisons out.
   ★ **AND A COST RULE THE SAME STEP MEASURED:** writing a loop out makes it pay its full count
   every time, so a kernel that runs it per cell must be GUARDED by the cheap test that says
   whether the answer matters. The carvers' hollow now compares SQUARED distances and pays the
   integer root only where a point stands inside a carver: the cave-dense chunk at the eight-metre
   rung fell from 73.7 ms to 7.8 ms of cell pass — 4.5 times cheaper than before this step — with
   no byte of the world moved.
   ★ **A THIRD INSTANCE OF (a), from step G2-A (MEASURED, 2026-09-13; bench part 6):** the same
   rule caught `bend::recip_sqrt`, whose square sum was a `while` loop over three components whose
   body RE-BOUND a two-word accumulator (`(s_hi, s_lo) = add_wide(…)`). On the card the third
   component's square was missing, so every direction the column pass wrote was 1.22 times too long
   and all 32 768 columns of the eight golden boxes differed. The three steps are written out now.
   The new reading: the accumulator need not be one word — a TUPLE the body re-binds goes stale the
   same way, and "assign" covers a destructuring assignment. ★ **AND HOW IT WAS FOUND IN ONE RUN:**
   the bench now compares the column pass's own DIRECTIONS, not only the cells they end up in. A
   direction that differs in its last bit can still pack the same cell byte, so an OUTPUT the kernel
   writes is worth comparing even when a later stage already is.
   ★ **AND ONE MORE COMPILER REFUSAL (step G2-A):** an 8-bit integer is refused as a FUNCTION
   PARAMETER too, not only as a table word. `bend::direction(face: u8, …)` compiled fine while only
   `relief_columns` ran; the moment the column pass called it the back end refused the module
   ("`u8` type used without `OpCapability Int8`"). Every face index on the recipe's path is a 32-bit
   word now.
   ★ **AND THE INSTRUMENT:** two PROBE entry points (`isqrt_probe`, `hollow_probe`) run ONE kernel
   over a list of words. A box of a quarter of a million cells can only say that something
   differs; a probe says which function does. Both faults above were found with them in minutes,
   and the bench runs them before part 5 so the next one is too.
2. **A fenced integer type, the way `Gf` fences floats**: `Gi(i64)` whose operators are only
   `wrapping_add/sub/mul`, `>>` and `<<` with the amount masked to 0..63 in the shared source,
   `&`, `|`, `^`, the compares, and the two loops (the square root, the exact landing). No `/`, no
   `%`, no `abs`, no unary minus on the type (a subtraction from zero, guarded from `MIN` by the
   format's headroom), no `checked_*`, no panic reachable. That one type removes every divergence
   above by construction; the clippy fence and the link scan keep it so, as today.
3. **SPIR-V → Bevy's `shader_format_spirv`**, `Capability::Int64` declared; the WGSL transpile
   target kept in reserve if the SPIR-V route fights on one back end.
4. **DX12 is unproven** until measured: the two open defects say a DX12 device may be lost or a
   64-bit negation unguarded. The byte-for-byte leg runs on Metal, Vulkan AND DX12 as three
   separate legs of `terrain-pin`; DX12 red means the CPU path on DX12 until naga is fixed (or a
   naga fix contributed), never a different recipe.
5. **The fallback for the 1 %** is the CPU path of the same source (Decision 2, (a)); limbs are
   refused: they double the arithmetic and rewrite the source.
6. **lavapipe as the server-side cross-check, not the server**: a Linux container with no GPU can
   run the client's own SPIR-V against the CPU recipe byte for byte in the gate — the no-drift
   measurement on a target the build machine can reach.

**Decision 5 (the owner): the toolchain.** rust-gpu is a community project at an alpha release,
tracked by a pinned git revision and a second nightly. It is the only option that keeps the recipe
in Rust, in our crate, under our tests. Recommended: **take it**, with the reserve route named
above; the alternative is Slang, which is mature and moves the recipe out of Rust.

**MEASURED, 2026-09-13 (bench part 4):** the recipe crate compiled to SPIR-V by cargo-gpu through
the shell `crates/recipe-gpu`, loaded by naga into Metal with `Int64`, gives 0 differing columns
of 3 936 256 against the CPU's call of the same function. The build path is real on this machine;
DX12 and Vulkan stay unmeasured until a Windows or Linux machine runs the same module.

## 5. The pictures, the pins and the gates

- The exact pictures are re-frozen once after the CPU integer recipe lands, on the owner's look,
  before any GPU step: the GPU steps are then judged against the INTEGER pictures with the
  one-level tolerance (V18), and a GPU-built chunk that differs from the CPU-built one by a byte is
  a red gate, not a picture question.
- The pins: `NOISE_PIN` and `VALUE_PIN` become the integer noise's; the golden tables are
  re-recorded; `GENERATOR_VERSION` advances, so an old client is refused at the handshake by name.
- The float fence stays: `Gf` loses its callers stage by stage and is deleted when the last one
  goes; the clippy fence and the link scan keep the crate float-free after that.

## 6. The order of work

1. **The integer recipe on the CPU** (`vd-seed`, `vd-terrain`): the stages of §1, in the table's
   order, each stage landing with its unit tests, the pins re-recorded at the end, the pictures
   re-frozen on the owner's look. The server is done at this step; the client's CPU path too.
2. **The one-source build** (§4's answer): the generator's integer core compiled for the GPU;
   G1 (the cell field) as the first GPU step, with the runtime self-check gate (§3); measured on
   the moving eye at F6's share of the cores.
3. **G2–G4**: the extraction, the position, the morph — each measured on the moving eye.
4. Then **slice 9** (the block store, the frozen patch) on the integer recipe's output.

**Decision 4 (the owner): the order.** Recommended as above; the alternative — the GPU steps
before the pictures are re-frozen — judges every GPU step against a moving reference and is
refused for that reason.

## 7. The five decisions, in one place

| # | decision | recommended |
|---|---|---|
| 1 | the direction's fraction bits | 40 (measured: within 0.02 mm of the float bend, +0.1 ms a chunk over 30 bits) |
| 2 | the fallback for a GPU without 64-bit integers | the CPU path of the same source; no limb emulation |
| 3 | the runtime self-check as the GPU's gate (eight golden chunks at start) | yes |
| 4 | the order: the CPU integer recipe, the pictures re-frozen, then G1–G4, then slice 9 | yes |
| 5 | the toolchain: rust-gpu (one crate, two compilations, a pinned nightly in a `build.rs`) with the WGSL transpile route in reserve; Slang the alternative | rust-gpu |
