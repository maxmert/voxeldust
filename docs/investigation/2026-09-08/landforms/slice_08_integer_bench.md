# The integer bench — the recipe's noise in fixed point on the CPU and on the GPU (ruling F7 step (a), 2026-09-12)

**The question.** The owner ruled the recipe integer-only and the GPU used to the maximum
(`owner_decisions_2026-09-12_frozen_patches.md` F7). Before a design: does an integer recipe give
the same bytes on the CPU and on the GPU, how close does it sit to today's float world, and what
does it cost? `crates/bins/examples/integer_bench.rs`; run
`cargo run --release -p vd-bins --features render --example integer_bench`.

**What runs.** The height field's octave sum — the recipe's gradient noise with the same corner
hash, the same sixteen gradients, the same quintic fade and the same blend order — in FIXED POINT
on 64-bit integers, on the same 3 936 256 column directions the GPU spike used (1 024 rung-0 chunks
on face +X of the home planet, 14 octaves). Both hosts read the same integer inputs and write the
relief as a 64-bit integer. The GPU side is a WGSL transcription, an instrument for this
measurement only; the product's GPU path is one source compiled for both targets (F7 item 3).

## The result (Apple M4 Pro, Metal, wgpu 27)

| | |
|---|---|
| IDENTITY: columns whose relief differs between the CPU and the GPU | **0 of 3 936 256** |
| PRECISION: the fixed-point relief against the float recipe's, widest | **1.06 mm** |
| PRECISION: mean | **0.13 mm** |
| columns further than one gap step (7.8 mm) from the float recipe | **0** |
| COST: the integer path, one CPU core | 698 ms (177 ns a column) |
| COST: the float recipe, one CPU core | 591 ms (150 ns a column) |
| COST: the GPU, with the upload and the readback | 127 ms |

Read: an integer recipe is identical on both hosts by construction and measured so on four million
columns; it reproduces today's world within a seventh of a gap step, so the pictures move by less
than the picture gate's own noise; on one CPU core it costs 18 % more than the float recipe; the
GPU does the four million columns 5.5 times faster than one core with the transfer included (the
spike's 32-bit float shadow took 56 ms — 64-bit integer multiplies cost more on this GPU than
32-bit floats, and the identity they buy is the point).

## The fixed-point format the bench settled on (the design's starting point, not its ruling)

| quantity | format | why |
|---|---|---|
| a direction component | 2⁻³⁰ in i64 | \|d\| ≤ 1; 31 bits |
| an octave's frequency | an integer part + a fraction at 2⁻²⁸ | up to 2.1 × 10⁵ cells per unit direction on the home planet; the lattice point is two products (`d × int`, exact; `d × frac >> 28`) summed at 30 fraction bits |
| the lattice point | 28 fraction bits | the noise's precision: 24 bits left the coarsest octave's 8 km of amplitude a 5 mm error |
| the fade, the dot, the blend | 28 fraction bits, each product shifted back once | every intermediate under 2⁶⁰ |
| an amplitude | 1/128 m with 8 more fraction bits (1/32 768 m) | an amplitude rounded to whole gap steps cost up to 3.9 mm an octave |
| the octave sum | summed at full precision, floored ONCE to the gap step | a floor per octave lost up to one step each, 57 mm over fourteen |

**Four runs to get there — each a lesson for the design.** (1) A frequency rounded to 2⁻⁸ moved
the coarsest lattice point by a ten-thousandth of a cell, which eight kilometres of amplitude
turned into 2.7 m: the frequency must be exact to about 2⁻²⁸. (2) Flooring each octave's product to
whole gap steps lost up to one step per octave (57 mm mean over fourteen): floor once. (3) 24
fraction bits in the noise left 5 mm at the coarsest octave: 28 are needed. (4) An amplitude in
whole gap steps cost 3.9 mm an octave: carry eight more bits. With all four, the widest
difference is 1.06 mm.

## Part 2 — the bend in integers (the direction itself, not borrowed from the float bend)

The same 3 936 256 columns' `(face, i, j)`; the face parameter `(2i + 1)/n_l − 1`, the quintic bend
with `k₁ = round(π/4 · 2³⁰)`, `k₂ = round(0.15 · 2³⁰)`, `k₃ = 2³⁰ − k₁ − k₂` (the three sum to one
exactly, so a face edge still lands on the cube edge exactly), the basis, and the normalise as an
INTEGER SQUARE ROOT (bit by bit) and three divisions — all at 30 fraction bits, on the CPU and in
a WGSL transcription on the GPU.

Two forms were measured: the divisions spelled out as restoring long division (shifts, compares
and subtracts), and the DIVISION-FREE form the design takes — the face parameter by a reciprocal
computed once per body (`floor(2⁵⁶ / n_l)`, a multiply and a shift per column), the normalise by
an integer Newton reciprocal of the length (six fixed steps from the seed 1.0, then two compare
loops that land it exactly on `floor(2⁶⁰ / len)`), and the last product rounded to nearest.

| | the divisions as loops | division-free |
|---|---|---|
| direction components that differ between the CPU and the GPU | **0 of 11 808 768** | **0 of 11 808 768** |
| the integer direction against the float bend's, laterally on the surface: widest | 13.3 mm | 15.4 mm |
| the same, mean | 5.1 mm | 5.4 mm (one step of 2⁻³⁰ is 5.9 mm on the home planet) |
| the integer bend, one CPU core | 1 273 ms (323 ns a column) | 187 ms (48 ns a column) |
| the float bend, one CPU core | 21 ms (5 ns a column) | 19 ms (5 ns a column) |
| the GPU, with the transfer | 111 ms | 84 ms |

**A FINDING FOR THE DESIGN — no 64-bit division operator on the Metal path today.** naga 27's
Metal back end guards a 64-bit `/` against overflow with a `select` call that Metal's compiler
rejects as ambiguous (`call to 'select' is ambiguous`: the `int` and the `long` overloads), so a
shader with an `i64` division does not build on macOS (MEASURED here; the first Metal error the
bench met). The bench spells its four divisions out as restoring long division (shifts, compares
and subtracts only), which builds and agrees bit for bit — at 323 ns a column (four 64-step loops
and a 32-step root; the float bend is 5 ns). The division-free form costs 48 ns a column, ten
times less, and lands within one direction step of the float bend, the same as the loops. (A
Newton reciprocal WITHOUT the exact landing read three steps low — 17 mm mean — because each
step truncates; the two compare loops cure it, and the last product is rounded, not floored.)

Read: the bend is lawful in integers, identical on both hosts, division-free at 48 ns a column
(ten times the float bend, 60 columns' worth in a chunk's 3 844 — under 0.2 ms of a 5 ms build),
and within one step of 5.9 mm of the float bend at 30 fraction bits; 32 fraction bits (1.5 mm
steps) fit if the square sum is carried unsigned. Integer division has ONE answer wherever it is
computed (no drift is possible), so a native `/` on the CPU beside a loop on a GPU would not be a
port in the law's sense — but a recipe with no division needs no such argument, and runs on every
back end naga has.

## Part 3 — the direction at 40 fraction bits (the owner: millimetres must stay; performance must not suffer)

The owner asked whether the 30-bit direction (5.9 mm steps on the home planet, 40 mm on the
largest legal body) gives up precision, and ruled that performance must not suffer. Part 3
measures the direction at 40 fraction bits: the two-word (128-bit) product spelled out from
32-bit halves on both hosts (`mul_wide`, `shr_wide`, `mul_shr`), the bend's polynomial at 40 bits,
the square sum at 80 bits in two words, and the reciprocal square root by ONE Newton step from the
30-bit path's exact reciprocal (an error of 2⁻³⁰ squares to 2⁻⁶⁰, past 40 bits), the residual
carried at 120 bits. No division anywhere.

| | 30 bits (part 2) | 40 bits (part 3) | the float bend |
|---|---|---|---|
| direction components that differ between the CPU and the GPU | 0 of 11 808 768 | **0 of 11 808 768** | — |
| against the float bend, laterally on the surface: mean | 5.4 mm | **0.0095 mm** | — |
| the same, widest | 15.4 mm | **0.020 mm** | — |
| one step of the direction on the home planet | 5.9 mm | 0.0058 mm | — |
| one step on the largest legal body (42 700 km) | 40 mm | 0.039 mm | — |
| one CPU core, a column | 44–48 ns | 77 ns | 5 ns |
| one CPU core, a chunk's 3 844 columns | 0.17 ms | **0.29 ms** | 0.02 ms |
| the GPU, the four million columns with the transfer | 31–84 ms | 107 ms | — |

Read: at 40 bits the integer direction sits within two hundredths of a millimetre of the float
bend on every column — a hundred times finer than the gap byte (7.8 mm) and the vertex grid
(3.9 mm) that store and draw the surface — on the largest legal body too. Its cost is 0.29 ms of
a chunk's build on one CPU core against 0.17 ms at 30 bits and 0.02 ms for the float bend: a
tenth of a millisecond more per chunk, two percent of a five-millisecond build, and nothing on
the GPU. The bench's 30-bit path stays as the seed of the 40-bit one (the reciprocal square root's
first guess), so the 40-bit path is the 30-bit path plus one Newton step in two words.

## What the bench decides, and what it leaves to the design discussion

Decided by measurement: the integer recipe is lawful under SL10 on both hosts (identical bytes),
it keeps today's shape (within a millimetre), and it is affordable (one core: +18 %; the GPU: 5.5
cores' worth on the columns, transfer included).

Left to the design (F7 step (b)), each a real decision:

1. **The bend and the direction.** Parts 2 and 3 measured it: identical on both hosts at 30 and
   at 40 fraction bits, division-free; 40 bits sit within 0.02 mm of the float bend for 0.29 ms
   of a chunk's build on one core. The design takes 40.
2. **The caves.** `value3` (a division by 2⁵³, a shift in fixed point) and the tube distance
   (`segment_distance_m`: a division and a square root) — an integer square root again.
3. **The density.** `quantise_gap` already floors once at 1/128 cell; the integer radius in
   1/128 m needs 33 bits at the largest legal body (radius 4.27 × 10⁷ m), so the radial subtraction
   stays in i64 and squares (the bounding spheres) need care or i128 on the CPU only.
4. **The vertex position** (`position.rs`: a trilinear blend of cell centres, sorted by site) and
   the client's morph — the extractor is already pure integer; these two are the float stages a
   GPU extraction must carry.
5. **The GPU build path.** One source compiled for both targets — `rust-gpu` (SPIR-V → naga) is
   the candidate named by the spike, UNMEASURED; the 64-bit integer feature (`SHADER_INT64`) is
   required and must be gated at the handshake (a GPU without it falls back to the CPU path of the
   same source).
6. **What "to the maximum" covers on the client**: the columns (this bench), the cell field, the
   extraction, the morph, and the ladder's per-vertex work — in which order, and what the CPU
   share (F6) keeps.
7. **The pins and the pictures**: `NOISE_PIN`, `VALUE_PIN`, the golden tables and the frozen
   exact pictures are re-recorded once, on the owner's look (F7 item 4).

## The build's own measurement (F7 step (c), 2026-09-12)

The CPU integer recipe landed: `vd-terrain` computes the static shape in `vd-recipe`'s kernels, and
the one float left draws a body from its seed (the charter) and states metres at the seam.

**MEASURED, the whole surface against the whole float surface** (not the relief alone, as part 1 was):
over the 3 936 256 columns of this bench's own square (face +X, rung 0, chunks (3..35, 5..37)), the
integer surface radius stands **1.4992 mm** from the float recipe's at the widest (cell (face +X,
1033, 486)) and **0.2002 mm** on average. The instrument was a throwaway example (`float_vs_integer`)
holding the float draw, the float bend and Perlin's noise in `f64`; it is deleted with the float
recipe it measured, and this line is the record.

The widest is bigger than part 1's 1.06 mm because the whole surface carries three more roundings
than the relief alone: the radius rounded once into the charter, the 40-bit bend, and the frequency
and amplitude rounded once each into the charter.

`GENERATOR_VERSION` is 2; `NOISE_PIN` and `VALUE_PIN` are the integer noise's words; the golden
tables are re-recorded. THE FROZEN PICTURES ARE NOT: they wait on the owner's look (F7 item 4).

## Part 4 — THE ONE SOURCE on the GPU (F8 decision 5, step (c)(2)-A, 2026-09-13)

No hand-written shader: the recipe crate itself, compiled to SPIR-V by cargo-gpu (rust-gpu
0.10.0-alpha.1 at revision 7fa56ad6, its pinned nightly 2026-05-22, the `Int64` capability) through
the thin shell `crates/recipe-gpu` (one entry point, `relief_columns`, that reads its bindings and
calls `vd_recipe::height::relief_of_table` — the function the server links), loaded through wgpu's
SPIR-V front end (naga: SPIR-V → MSL) on the Apple M4 Pro, run on the same 3 936 256 columns over
the home planet's own fourteen octaves, and compared with the CPU's call of the same function.

| | |
|---|---|
| columns that differ between the CPU and the GPU through the one source | **0 of 3 936 256** |
| the CPU, one core | 719 ms |
| the GPU, the first pass (the pipeline's own compilation included) | 285 ms |
| the GPU, the second pass (the upload and the readback included) | 348 ms |

**What the GPU compiler refused on the way, and what changed — each a rule for the kernels.**
(1) A `for` over a slice and a runtime-length slice of a stack array: the compiler cannot convert
pointers to integers, so the octave sum is an index loop and the shell calls a fixed-size-table
form (`relief_of_table`). (2) The 8-bit integer type of the gradient and basis tables: the GPU
carries none without a capability of its own, so both tables are 32-bit words (−1, 0 and 1 read
the same). (3) The 128-bit division of the cell-count reciprocal: gated off the GPU target
(`#[cfg(not(target_arch = "spirv"))]`); the charter carries the word and the GPU never computes it.
The stable workspace declares the `spirv` target arch for the cfg check and EXCLUDES the shell
crate (cargo-gpu builds it with its own toolchain). The build: `just recipe-gpu`.

Read: SL10's "one generator, two hosts, no drift" now holds on the GPU by the same source, not by a
transcription — the first time the GPU ran the shipped function. The GPU's cost through this path
is the next measurement (the transcription ran the same columns in 56 ms; the one-source module's
second pass is the number above), and the client's runtime self-check (F8 decision 3) and the
cell-field step (G1) grow from this entry point.

