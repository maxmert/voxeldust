//! ★ THE SEAM TABLE — MOVED to the `vd-seed` leaf in the voxel foundation's slice 6 (ruling V10
//! S6-2), because the generator gathers a chunk's halo across a face edge with the SAME table the grid
//! uses to step across it, and one table is the only way two hosts agree. Re-exported here at the
//! path the grid always used.

pub use vd_seed::seam::*;
