//! ★ `vd-seed` — THE LEAF two hosts must compute identically (the voxel foundation, slice 5;
//! ruling V9 S5-1, ruling V6 Part D).
//!
//! The generator crate (`vd-terrain`) turns `(seed, address)` into the world's static shape, and the
//! server and every client link the SAME crate (SL10). The generator needs four things from the rest
//! of the tree, and each of them must be ONE implementation, never two: the integer hash every draw
//! comes from, the content digest, the face bend that turns an address into a direction on the
//! sphere, and the ladder arithmetic that turns a cell index into a radius. They live here, in a crate
//! that depends on nothing that computes with floats, so a client on another engine links the
//! generator, this leaf and serde's derives (for the face byte) — no `glam`, no `libm` edge, no I/O.
//!
//! `vd-core` depends on this crate and re-exports every module at the path it always had
//! (`vd_core::rng`, `vd_core::digest`, `vd_core::grid::Face`), so no caller moved.
//!
//! **The float fence applies here** (`clippy.toml` beside this file): every operation on this crate's
//! float path is one IEEE-754 fixes on every target. A transcendental in this crate is a red lint.
//!
//! **Example.** The moon's shard asks "which cell holds the pilot's boots?" and the client asks "what
//! direction is cell (face 2, 1181, 77)?". Both go through the bend in this crate, so the boots and the
//! drawn hill name the same cell on an Apple chip and an Intel chip.

// ★ NO `/` AND NO `%` ON THE RECIPE'S PATH (ruling F7, `owner_decisions_2026-09-12_frozen_patches.md`;
// the design's §1: "no division operator anywhere"). naga's Metal back end cannot compile a 64-bit
// division, and WGSL, SPIR-V and Rust disagree about a division by zero, so the arithmetic a GPU kernel
// will run carries NEITHER operator: a power of two is a shift and a mask, and anything else is a
// reciprocal drawn ONCE per body (or once per tube carver) and multiplied. The two lints below make that
// a compile error rather than a review finding. A site that is CPU-ONLY today and integer-exact on every
// host by definition says so with `#[allow(clippy::integer_division, reason = …)]` and names WHY; the
// extractor (the GPU's step G2, not yet ported) is the one named exception.
#![deny(clippy::integer_division)]
#![deny(clippy::modulo_arithmetic)]

pub mod bend;
pub mod digest;
pub mod ladder;
pub mod rng;
pub mod seam;
