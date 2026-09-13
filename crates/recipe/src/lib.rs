//! ★ `vd-recipe` — THE INTEGER RECIPE (ruling F7, `owner_decisions_2026-09-12_frozen_patches.md`;
//! the design in `docs/investigation/2026-09-08/landforms/slice_08_integer_recipe_design.md`).
//!
//! The world's static shape is a function of the seed and the address (SL10). Two hosts compute it —
//! the server for collision, the client for the picture — and the bytes must be the same on every
//! CPU and every GPU. Floats cannot promise that (the GPU spike measured a Metal shader that agreed
//! with the CPU only with a compiler switch nobody controls; `slice_08_gpu_spike.md`). Integers can:
//! an add, a multiply, a shift and a compare on 64-bit words give one answer everywhere. So every
//! stage of the recipe lives here on 64-bit integers with a fixed number of fraction bits, and this
//! crate is ONE SOURCE compiled for the server's CPU, the client's CPU and — through `rust-gpu` —
//! the client's GPU (F8 decision 5). Nothing here is a port of anything.
//!
//! **The fence.** [`gi::Gi`] is the only arithmetic type the kernels use: its operators wrap, its
//! shifts mask their amount, and it has NO division, NO remainder, NO negation and NO absolute value,
//! because those four are where the CPU, the SPIR-V and the WGSL specifications disagree (a shift by
//! the word's width, a division by zero, the negation of the most negative word). The crate denies
//! float arithmetic and integer division outright, so a bare `/` between two words is a red lint
//! before it is a red gate. Every operation that may wrap is written `wrapping_*` at the one place
//! the primitive is touched (`gi.rs`, `wide.rs`), never in a kernel.
//!
//! **The formats** (MEASURED, `slice_08_integer_bench.md`): a direction at 40 fraction bits
//! ([`bend`]: within 0.02 mm of the float bend on every column, two-word products); the noise's
//! lattice point at 28 ([`noise`]: within 0.13 mm mean of the float recipe over four million
//! columns); the octave sum at 28 with the amplitude at 1/32 768 m, floored ONCE to the gap step
//! ([`height`]).
//!
//! **Example.** The moon's shard asks which cell holds the pilot's boots, and the client asks where
//! cell (face 2, 1181, 77) is and how high the ground stands there. Both go through this crate's
//! bend and octave sum, and the boots and the drawn hill name the same cell and the same height on
//! an Apple chip, an Intel chip and the GPU beside either.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::float_arithmetic)]
#![deny(clippy::integer_division)]
#![deny(clippy::modulo_arithmetic)]

pub mod bend;
pub mod cell;
pub mod gi;
pub mod height;
pub mod noise;
pub mod plan;
pub mod rng;
pub mod root;
pub mod wide;

pub use gi::Gi;
