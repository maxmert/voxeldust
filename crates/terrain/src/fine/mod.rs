//! ★ THE FINE GROUND (B1 of the way-in design, revision 4: `docs/investigation/2026-10-11/
//! fine_ground_way_in_design.md`; rulings B80–B84).
//!
//! Below the 8 km macro nodes the planet's own shard solves its ground in TILES at fine LEVELS, each level a rung's cell
//! and a law: L1 at the rung-7 cell (128 m on the home planet), L2 at the rung-4 cell (16 m). A tile is a pure function
//! of its inputs (the seed, the charter, the artifact, the present rise, the fine version), so the result never depends
//! on which host solved it, when, or on how many cores (B64). This module is the generator's part of it, one source for
//! the shard and every client (SL10): the points, the point graph, the solve, the mean pyramid, the coder and the read.
//!
//! **Example.** A hull lands on the home planet's belt; the planet's shard solves the 128 m tiles under its path on its
//! share of the cores, stores them, and ships them; every client reads the same integer heights from the same bytes.

pub mod coder;
pub mod delaunay;
pub mod finish;
pub mod geometry;
pub mod l1;
pub mod l2;
pub mod level;
pub mod points;
pub mod read;
pub mod stitch;
pub mod tile;
pub mod tree;
