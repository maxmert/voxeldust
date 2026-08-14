//! # vd-physics — THE motion crate (SL4, the placement arc S5)
//!
//! *"Physics and re-home are separate machinery, one-way. Physics produces A PLACEMENT;
//! re-home/containment CONSUMES placements and may never ask HOW a thing moves."*
//!
//! This crate is the one place in the tree that knows how anything moves: the closed-form
//! celestial math (Category A), the [`Motion`](motion::Motion) discriminant, and the seed
//! universe generator (everything that reads the seed stream or mints a body — the split
//! rule; what stayed in `vd-core` only reads an already-built `&[RealmRegion]`).
//!
//! THE FENCE: the crossing/containment path (`vd-core`, `vd-wire`, `vd-sim`, the client)
//! carries NO Cargo edge to this crate — `tests/tests/crate_isolation.rs` asserts it — so
//! an orbit symbol on the crossing path is an unresolved-crate compile error, not a review
//! finding. Motion reaches the simulation ONLY as data: authored placement rows in
//! `vd_core::placement` books, and the opaque [`MotionFn`](vd_core::placement::MotionFn)
//! closures the boot composition injects (the same seam discipline as `sim::io`).
//!
//! Coverage: Tier-A — 100% region + branch (HR5).

pub mod celestial;
pub mod motion;
pub mod taxonomy;
pub mod worldgen;
