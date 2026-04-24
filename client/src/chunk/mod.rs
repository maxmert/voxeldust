//! Generic, shard-agnostic chunk streaming + meshing pipeline.
//!
//! Primary + every secondary pump through the same path: decode the
//! server's palette+data blob into `ChunkStorage`, keep it in the
//! `ChunkStorageCache` for raycast (Phase 14), greedy-mesh it with
//! neighbour context, spawn / replace a child mesh under the shard's
//! `ChunkSource` root entity.
//!
//! Per-shard-type vertex-transform hook (`ShardTypePlugin::mesh_vertex_transform`
//! — Phase 8+ for PLANET sphere-projection) is consulted during mesh
//! construction. Until a plugin overrides it the transform is identity
//! (flat cubic chunks — correct for SHIP).

pub mod cache;
pub mod material;
pub mod plugin;
pub mod stream;

pub use cache::{ChunkKey, ChunkStorageCache};
pub use material::{ChunkMaterial, ChunkMaterialCache};
pub use plugin::{ChunkStreamPlugin, ChunkStreamSet};
pub use stream::ChunkIndex;
