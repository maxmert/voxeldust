//! ★ `vd-terrain` — THE ONE GENERATOR (SL10; the voxel foundation, slice 5; ruling V9).
//!
//! The world is too big to store, so the game stores one number — the seed — and a recipe. The recipe
//! turns the seed and an address into the shape of the ground at that address: the substance of a cell
//! and its GAP, how far the ground surface is from the cell's centre along the line from the body's
//! centre. This crate IS the recipe. The server compiles it to collide; every client compiles it to
//! draw; a port to another language is forbidden, because two recipes drift.
//!
//! What the recipe decides: only what never changes — the body from its seed, the hills, which common
//! rock is where, where a cave runs, where the water stands. What it never decides: a hole a player
//! dug, a wall a player built, where the copper is. Those are state; the owning realm stores them and
//! ships them as a diff.
//!
//! ★ **THE RECIPE IS INTEGER-ONLY** (ruling F7, `owner_decisions_2026-09-12_frozen_patches.md`). Every
//! step of the static shape — the hash, the noise, the octave sum, the face bend, the density, the
//! vertex position — is a 64-bit integer in a fixed-point format, in the [`vd_recipe`] kernels this
//! crate is built on. Integers give the same bytes on every CPU and every GPU, so the no-drift gate
//! (SL10) passes by construction and not by luck. The recipe's unit of length is the GAP STEP, 1/128 m
//! ([`units`]); metres exist only at the seam a host outside the recipe reads.
//!
//! **The one float left** is the DRAW of a body from its seed ([`body::BodyDefinition::from_seed`]):
//! the shares, caps and weights the seed states are read as fenced floats ([`Gf`], the operations
//! IEEE-754 fixes on every target) and ROUNDED ONCE into the body's integer charter, on the CPU, once
//! per body. A kernel never sees a float. A lint (`clippy.toml`) and a link scan (`just
//! terrain-link-scan`) keep the platform's transcendentals out; the golden gate
//! (`tests/terrain_pin.rs`) is the measurement: the same bytes on every build and every chip, red on
//! one differing byte.
//!
//! **Example.** A pilot flies to the home moon. The client computes the hills from the seed while the
//! hull is 100 km out. Nothing about the hills crosses the network. The moon's shard computes the same
//! hills to know where the boots land.

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

pub mod artifact;
pub mod body;
pub mod carve;
pub mod chunk;
pub mod climate;
pub mod compose;
pub mod craters;
pub mod digest;
pub mod extract;
pub mod gf;
pub mod gpu;
pub mod height;
pub mod home;
pub mod lakes;
pub mod land;
pub mod lattice;
pub mod macro_lattice;
pub mod noise;
pub mod position;
pub mod river;
pub mod seat;
pub mod solve;
pub mod strata;
pub mod tag;
pub mod units;

pub use body::{BodyDefinition, BodyFacts, OCTAVES};
pub use chunk::{CHUNK_CELLS, CHUNK_EDGE, Cell, ChunkKey, ChunkLattice};
pub use compose::{EditRow, compose};
pub use digest::{
    ChunkDigest, GOLDEN_SELF_CHECK_KEYS, chunk_digest, golden_self_check, mesh_digest,
};
pub use extract::{ChunkMesh, VERTEX_QUANTUM, extract};
pub use gf::Gf;
pub use lattice::{SampleBox, sample_box};
pub use position::{vertex_position, vertex_position_m};
pub use seat::seat_eighths;
pub use strata::{Biome, Stratum};
pub use tag::{GENERATOR_VERSION, WorldIdentity, declared_world_tag};
pub use units::{STEPS_PER_M, metres_of_q28, metres_of_steps};

/// ★ THE LINT CONTROL (SL1 clause 5: a structural fence has a control that is seen failing). Built
/// ONLY under the `fence-control` feature, which `just terrain-fence-control` turns on and expects
/// clippy to go RED on — a sine, a fused multiply-add and a max, the three kinds of call the fence
/// exists to refuse. A release build never contains it.
#[cfg(feature = "fence-control")]
pub mod fence_control {
    /// Three calls the fence refuses; the recipe expects three red lints.
    #[must_use]
    pub fn control(x: f64) -> f64 {
        x.sin() + x.mul_add(x, x) + x.max(1.0)
    }
}
