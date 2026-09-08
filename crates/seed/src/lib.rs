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

pub mod bend;
pub mod digest;
pub mod ladder;
pub mod rng;
pub mod seam;
