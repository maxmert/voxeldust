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

## What the bench decides, and what it leaves to the design discussion

Decided by measurement: the integer recipe is lawful under SL10 on both hosts (identical bytes),
it keeps today's shape (within a millimetre), and it is affordable (one core: +18 %; the GPU: 5.5
cores' worth on the columns, transfer included).

Left to the design (F7 step (b)), each a real decision:

1. **The bend and the direction.** The bench takes the column directions from today's float bend
   (`vd_seed::bend`: `W(a) = a(k₁ + a²(k₂ + a²k₃))` with `k₁ = π/4`, then a normalise with a square
   root and three divisions) and rounds them to 2⁻³⁰. An integer recipe must produce the direction
   itself: a rational `k₁`, an integer square root (the normalise), and the face parameter as a
   pair of integers. The bend's own identity (the same point on both hosts) is then by construction.
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
