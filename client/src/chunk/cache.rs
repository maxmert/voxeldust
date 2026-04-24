//! Retained per-chunk block storage keyed by `(ShardKey, chunk_index)`.
//!
//! Retained so:
//! - Block raycast (Phase 14) tests block occupancy along the ray.
//! - Neighbour-aware greedy meshing reads adjacent chunks to suppress
//!   seam-faces at chunk boundaries.
//! - Delta application (`ChunkDelta` with sparse `BlockMod` entries)
//!   mutates the retained storage then re-meshes.

use std::collections::HashMap;

use bevy::prelude::*;

use voxeldust_core::block::chunk_storage::ChunkStorage;

use crate::shard::ShardKey;

pub type ChunkKey = (ShardKey, IVec3);

#[derive(Resource, Default)]
pub struct ChunkStorageCache {
    pub entries: HashMap<ChunkKey, ChunkStorage>,
}

impl ChunkStorageCache {
    pub fn get(&self, key: ShardKey, chunk_index: IVec3) -> Option<&ChunkStorage> {
        self.entries.get(&(key, chunk_index))
    }

    pub fn get_mut(
        &mut self,
        key: ShardKey,
        chunk_index: IVec3,
    ) -> Option<&mut ChunkStorage> {
        self.entries.get_mut(&(key, chunk_index))
    }

    /// Populate a neighbour array for greedy meshing.
    ///
    /// The order must match `voxeldust_core::block::chunk_mesher::FACE_OFFSETS`
    /// (±X, ±Y, ±Z). Missing neighbours (not yet streamed) return `None`;
    /// greedy mesher treats them as air.
    pub fn neighbours<'a>(
        &'a self,
        key: ShardKey,
        chunk_index: IVec3,
    ) -> [Option<&'a ChunkStorage>; 6] {
        // Mesher convention: [+X, -X, +Y, -Y, +Z, -Z] — matches
        // `voxeldust_core::block::chunk_mesher::FACE_NORMALS` ordering.
        [
            self.get(key, chunk_index + IVec3::X),
            self.get(key, chunk_index - IVec3::X),
            self.get(key, chunk_index + IVec3::Y),
            self.get(key, chunk_index - IVec3::Y),
            self.get(key, chunk_index + IVec3::Z),
            self.get(key, chunk_index - IVec3::Z),
        ]
    }
}
