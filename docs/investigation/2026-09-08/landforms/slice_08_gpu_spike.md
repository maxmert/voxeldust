# Slice 8 — THE GPU SPIKE, MEASURED (2026-09-10, ruling V17 item 3)

**The question.** Can the world's recipe run on the GPU byte for byte (SL10: one generator, two
hosts, no drift)? The recipe computes with 64-bit floats (`Gf`), which the Mac's GPU does not have
(`SHADER_F64` is Vulkan-only in wgpu 27, and 16 to 64 times slower where it exists), so the spike
asked the two questions that remain, on this machine (Apple M4 Pro, Metal, wgpu 27.0.1, naga
27.0.3), with `crates/bins/examples/gpu_spike.rs`:

1. THE INTEGER HALF: the corner hash (`SplitMix64` on 64-bit integers, `SHADER_INT64`, which Metal
   offers natively) in a compute shader against the CPU.
2. THE 32-BIT QUESTION: a SHADOW of the recipe's noise in 32-bit floats — `noise3` with every
   float a 32-bit one, integer lattice addressing, the same operations in the same order, the
   height's octave sum — on the CPU and in a compute shader, over every column of 1 024 rung-0
   chunks of the home planet (3 936 256 columns, 14 octaves each), byte for byte.

The shadow changes nothing in the world; it is an instrument outside the fence.

## The measurement

| Part | Setting | Columns / corners | Differ | Widest |
|---|---|---|---|---|
| 1, the integer hash | any | 3 936 256 corners | **0** | — |
| 2, the 32-bit shadow | Metal's default (fast math on, contraction on) | 3 936 256 columns | 3 855 023 (98 %) | 76 ulps, 4.6 mm |
| 2 | fast math OFF (`fastMathEnabled = false`), contraction on | 3 936 256 | 3 851 236 (98 %) | 67 ulps, 4.1 mm |
| 2 | `#pragma METAL fp contract(off)`, fast math on | 3 936 256 | **0** | 0 |
| 2 | `#pragma METAL fp contract(off)`, fast math off | 3 936 256 | **0** | 0 |

The fast-math switch and the pragma were measured through a scratch copy of the Metal backend
(`wgpu-hal` 27.0.4 with two environment switches, patched into the workspace for the measurement
only and removed after it; the shipped tree carries no patch). The pragma is prepended to the MSL
text naga emits; wgpu exposes no option for it.

**What it means, in plain words.** A "fused multiply-add" is one instruction that computes
`a × b + c` with one rounding instead of two. The Metal compiler inserts it wherever it sees a
multiply feeding an add, by default AND with fast math off. That single habit is the whole
difference: with contraction off, four million columns of relief agree with the CPU to the last
bit, fast math on or off. The integer hash never differed.

**Example.** The relief under the ground stand's first column is 902.597 168 m on the CPU. With
Metal's default the GPU says 902.597 412 m (four ulps up); with fast math off, 902.598 145 m
(sixteen ulps up); with contraction off, 902.597 168 m — the same bits.

**The payoff, measured on the same columns.** GPU 56 ms with the upload and the readback;
the 32-bit shadow on ONE CPU core 664 ms; the real 64-bit recipe on one core 593 ms (the 64-bit
`Gf` path is FASTER than the 32-bit shadow on this CPU). The 14 workers would do the columns in
about 45 ms. On the height columns alone the GPU is twelve times one core and about equal to the
fourteen: the columns are 0.4 to 0.6 ms of a 5 ms chunk build, so the height field is not where
GPU generation would pay; the cell field, the extraction and the morph would all have to move.

## What the result decides, and what it leaves to the owner

1. **A 32-bit recipe CAN cross to the Mac's GPU without drift**, provided contraction is off in
   the shader compiler. On Vulkan and DX12 the equivalent is SPIR-V's `NoContraction` decoration
   and HLSL's `precise`; naga emits neither today. Every shipped target needs its own measurement
   (SL10: no drift is measured on every target, never argued).
2. **The recipe's number format is the owner's decision (V17's A or B).** Today: 64-bit, fence,
   CPU only. GPU generation means the recipe moves to 32-bit floats plus integer lattice
   addressing on BOTH hosts (the server in 32-bit on the CPU, the same operation order), which
   moves every hill a little and regenerates the pin; SL5 allows it before anything ships.
3. **SL10 says "a port is FORBIDDEN"** — and a WGSL compute shader is a second implementation of
   the noise, written by hand. The spike's shader is exactly such a port. Three ways to honour
   the law, for the owner:
   - (a) ONE source, two targets: compile the Rust generator crate to SPIR-V (`rust-gpu`) and let
     naga translate it. A separate nightly toolchain, a `no_std` recipe, and the contraction
     control still needed — UNMEASURED whether the crate compiles that way.
   - (b) A port under a GATE: the WGSL copy is allowed only because a byte-for-byte gate compares
     it with the Rust recipe on every shipped target on every build (this spike is that gate's
     seed). The law's word would change from "forbidden" to "forbidden unless gated".
   - (c) No GPU generation: the recipe stays 64-bit on the CPU; the levers are the parent cache,
     job order, fewer builds and the far-rung voxel renderer (D8-8).
4. **What it does not buy.** The still stand's 29 frames a second is the draw count; the 2.5 GB
   on the GPU is the drawn bytes. Both belong to the lossy packing (V17 item 2, the tolerance
   decision) and to the far-rung voxel renderer, not to where the chunk is born.

## The instrument

`cargo run --release -p vd-bins --features render --example gpu_spike` prints the adapter, the
feature flags, both parts' counts and times, and the first differing column with both bit
patterns. It stops at part 1 with exit 1 if one hash differs. The contraction and fast-math
switches live only in the scratch backend used for this measurement.
