//! Client-side chunk cache — stores block data received from multiple shard
//! connections simultaneously, with per-source lifecycle management.
//!
//! ## Multi-Source Architecture
//!
//! The client can be connected to multiple shards at once (ship + planet during
//! transitions, or ship + system for exterior rendering). Each connection is a
//! **chunk source** identified by `ChunkSourceId`. Chunks from different sources
//! coexist in the cache — when the player exits a ship, the ship's chunks remain
//! visible as exterior geometry while planet chunks load alongside them.
//!
//! Sources are added when a shard connection is established and removed when the
//! connection is fully terminated AND the chunks are no longer needed for rendering.
//!
//! ## Lifecycle
//!
//! 1. Shard connection established → `add_source(source_id)`
//! 2. `ChunkSnapshot` received → `insert_snapshot(source_id, ...)` → stored, marked dirty
//! 3. `ChunkDelta` received → `apply_delta(source_id, ...)` → edits applied, marked dirty
//! 4. Render loop calls `drain_dirty()` → gets `(source_id, chunk_key)` pairs to remesh
//! 5. Shard disconnected → `remove_source(source_id)` → all chunks from that source
//!    are marked for GPU buffer cleanup

use std::collections::{HashMap, HashSet};

use glam::IVec3;

use super::block_id::BlockId;
use super::chunk_storage::ChunkStorage;
use super::serialization;

/// Identifies the source of chunk data (which shard connection provided it).
///
/// Typically the shard's connection index or a monotonically increasing ID
/// assigned when the connection is established. NOT the shard_id itself,
/// because the same shard might reconnect with different data after a restart.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkSourceId(pub u32);

/// Composite key for a chunk in the multi-source cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkKey {
    pub source: ChunkSourceId,
    pub chunk: IVec3,
}

/// Client-side chunk storage with multi-source support and dirty tracking.
pub struct ClientChunkCache {
    /// Chunk data keyed by (source, chunk_position).
    chunks: HashMap<ChunkKey, ChunkStorage>,
    /// Chunks that need remeshing. Includes the source so the renderer knows
    /// which GPU buffer set to update.
    dirty: HashSet<ChunkKey>,
    /// Per-chunk sequence numbers for stale-delta rejection.
    sequences: HashMap<ChunkKey, u64>,
    /// Active sources. When a source is removed, all its chunks are cleaned up.
    active_sources: HashSet<ChunkSourceId>,
    /// Sources that were removed this frame — renderer should clean up their GPU buffers.
    removed_sources: Vec<ChunkSourceId>,
    /// Monotonically increasing counter for generating unique source IDs.
    next_source_id: u32,
}

impl ClientChunkCache {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            dirty: HashSet::new(),
            sequences: HashMap::new(),
            active_sources: HashSet::new(),
            removed_sources: Vec::new(),
            next_source_id: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Source lifecycle
    // -----------------------------------------------------------------------

    /// Register a new chunk source (shard connection). Returns the assigned ID.
    pub fn add_source(&mut self) -> ChunkSourceId {
        let id = ChunkSourceId(self.next_source_id);
        self.next_source_id += 1;
        self.active_sources.insert(id);
        id
    }

    /// Remove a chunk source and all its data. The source's chunks are not
    /// immediately deleted — instead, the source is added to `removed_sources`
    /// so the renderer can clean up GPU buffers on the next frame.
    pub fn remove_source(&mut self, source: ChunkSourceId) {
        if !self.active_sources.remove(&source) {
            return;
        }
        // Remove all chunks belonging to this source.
        self.chunks.retain(|key, _| key.source != source);
        self.dirty.retain(|key| key.source != source);
        self.sequences.retain(|key, _| key.source != source);
        self.removed_sources.push(source);
    }

    /// Check if a source is currently active.
    pub fn is_source_active(&self, source: ChunkSourceId) -> bool {
        self.active_sources.contains(&source)
    }

    /// Drain the list of sources removed since the last call.
    /// The renderer uses this to clean up GPU buffers for removed sources.
    pub fn drain_removed_sources(&mut self) -> Vec<ChunkSourceId> {
        std::mem::take(&mut self.removed_sources)
    }

    // -----------------------------------------------------------------------
    // Chunk data operations
    // -----------------------------------------------------------------------

    /// Insert a full chunk snapshot from a specific source.
    pub fn insert_snapshot(
        &mut self,
        source: ChunkSourceId,
        chunk_pos: IVec3,
        seq: u64,
        compressed_data: &[u8],
    ) -> Result<(), String> {
        if !self.active_sources.contains(&source) {
            return Err(format!("source {:?} is not active", source));
        }

        let chunk = serialization::deserialize_chunk(compressed_data)
            .map_err(|e| format!("chunk {chunk_pos} deserialize failed: {e}"))?;

        let key = ChunkKey { source, chunk: chunk_pos };
        self.chunks.insert(key, chunk);
        self.sequences.insert(key, seq);
        self.dirty.insert(key);
        self.mark_neighbors_dirty(source, chunk_pos);

        Ok(())
    }

    /// Apply incremental block edits from a specific source.
    pub fn apply_delta(
        &mut self,
        source: ChunkSourceId,
        chunk_pos: IVec3,
        seq: u64,
        edits: &[(u8, u8, u8, BlockId)],
    ) -> bool {
        let key = ChunkKey { source, chunk: chunk_pos };

        // Reject stale deltas.
        if let Some(&last_seq) = self.sequences.get(&key) {
            if seq <= last_seq {
                return false;
            }
        }

        let chunk = match self.chunks.get_mut(&key) {
            Some(c) => c,
            None => return false,
        };

        for &(bx, by, bz, block_type) in edits {
            chunk.set_block(bx, by, bz, block_type);
        }

        self.sequences.insert(key, seq);
        self.dirty.insert(key);
        if edits.iter().any(|&(bx, by, bz, _)| is_boundary_block(bx, by, bz)) {
            self.mark_neighbors_dirty(source, chunk_pos);
        }

        true
    }

    /// Get a chunk by source and position.
    pub fn get_chunk(&self, source: ChunkSourceId, chunk_pos: IVec3) -> Option<&ChunkStorage> {
        self.chunks.get(&ChunkKey { source, chunk: chunk_pos })
    }

    /// Get a chunk's 6 face neighbors within the same source.
    pub fn get_neighbors(&self, source: ChunkSourceId, chunk_pos: IVec3) -> [Option<&ChunkStorage>; 6] {
        [
            self.get_chunk(source, chunk_pos + IVec3::X),
            self.get_chunk(source, chunk_pos - IVec3::X),
            self.get_chunk(source, chunk_pos + IVec3::Y),
            self.get_chunk(source, chunk_pos - IVec3::Y),
            self.get_chunk(source, chunk_pos + IVec3::Z),
            self.get_chunk(source, chunk_pos - IVec3::Z),
        ]
    }

    // -----------------------------------------------------------------------
    // Dirty tracking
    // -----------------------------------------------------------------------

    /// Drain all dirty chunk keys. Returns `(source, chunk_pos)` pairs.
    pub fn drain_dirty(&mut self) -> Vec<ChunkKey> {
        self.dirty.drain().collect()
    }

    /// Whether any chunks need remeshing.
    pub fn has_dirty(&self) -> bool {
        !self.dirty.is_empty()
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Total number of chunks across all sources.
    pub fn total_chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Number of chunks from a specific source.
    pub fn source_chunk_count(&self, source: ChunkSourceId) -> usize {
        self.chunks.keys().filter(|k| k.source == source).count()
    }

    /// All chunk keys for a specific source.
    pub fn source_chunk_keys(&self, source: ChunkSourceId) -> impl Iterator<Item = IVec3> + '_ {
        self.chunks.keys()
            .filter(move |k| k.source == source)
            .map(|k| k.chunk)
    }

    /// Remove all data (all sources, all chunks). Use only on full disconnect.
    pub fn clear_all(&mut self) {
        for source in self.active_sources.drain() {
            self.removed_sources.push(source);
        }
        self.chunks.clear();
        self.dirty.clear();
        self.sequences.clear();
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    fn mark_neighbors_dirty(&mut self, source: ChunkSourceId, chunk_pos: IVec3) {
        for offset in [IVec3::X, -IVec3::X, IVec3::Y, -IVec3::Y, IVec3::Z, -IVec3::Z] {
            let neighbor_key = ChunkKey { source, chunk: chunk_pos + offset };
            if self.chunks.contains_key(&neighbor_key) {
                self.dirty.insert(neighbor_key);
            }
        }
    }
}

fn is_boundary_block(bx: u8, by: u8, bz: u8) -> bool {
    bx == 0 || bx == 61 || by == 0 || by == 61 || bz == 0 || bz == 61
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{BlockId, ChunkStorage, serialize_chunk};

    fn make_test_chunk() -> Vec<u8> {
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(5, 5, 5, BlockId::STONE);
        chunk.set_block(10, 10, 10, BlockId::HULL_STANDARD);
        serialize_chunk(&chunk)
    }

    #[test]
    fn source_lifecycle() {
        let mut cache = ClientChunkCache::new();
        let src = cache.add_source();
        assert!(cache.is_source_active(src));

        cache.remove_source(src);
        assert!(!cache.is_source_active(src));

        let removed = cache.drain_removed_sources();
        assert_eq!(removed, vec![src]);
    }

    #[test]
    fn insert_and_retrieve_with_source() {
        let mut cache = ClientChunkCache::new();
        let src = cache.add_source();
        let data = make_test_chunk();

        cache.insert_snapshot(src, IVec3::ZERO, 1, &data).unwrap();
        assert_eq!(cache.total_chunk_count(), 1);

        let chunk = cache.get_chunk(src, IVec3::ZERO).unwrap();
        assert_eq!(chunk.get_block(5, 5, 5), BlockId::STONE);
    }

    #[test]
    fn multiple_sources_coexist() {
        let mut cache = ClientChunkCache::new();
        let ship_src = cache.add_source();
        let planet_src = cache.add_source();

        let data = make_test_chunk();
        cache.insert_snapshot(ship_src, IVec3::ZERO, 1, &data).unwrap();
        cache.insert_snapshot(planet_src, IVec3::ZERO, 1, &data).unwrap();

        // Same chunk position, different sources — both exist.
        assert_eq!(cache.total_chunk_count(), 2);
        assert!(cache.get_chunk(ship_src, IVec3::ZERO).is_some());
        assert!(cache.get_chunk(planet_src, IVec3::ZERO).is_some());
    }

    #[test]
    fn remove_source_keeps_other_sources() {
        let mut cache = ClientChunkCache::new();
        let ship_src = cache.add_source();
        let planet_src = cache.add_source();

        let data = make_test_chunk();
        cache.insert_snapshot(ship_src, IVec3::ZERO, 1, &data).unwrap();
        cache.insert_snapshot(planet_src, IVec3::ZERO, 1, &data).unwrap();

        // Remove ship source — planet chunks remain.
        cache.remove_source(ship_src);
        assert_eq!(cache.total_chunk_count(), 1);
        assert!(cache.get_chunk(ship_src, IVec3::ZERO).is_none());
        assert!(cache.get_chunk(planet_src, IVec3::ZERO).is_some());
    }

    #[test]
    fn dirty_includes_source() {
        let mut cache = ClientChunkCache::new();
        let src = cache.add_source();
        let data = make_test_chunk();

        cache.insert_snapshot(src, IVec3::ZERO, 1, &data).unwrap();
        let dirty = cache.drain_dirty();
        assert!(dirty.iter().any(|k| k.source == src && k.chunk == IVec3::ZERO));
    }

    #[test]
    fn apply_delta_with_source() {
        let mut cache = ClientChunkCache::new();
        let src = cache.add_source();
        let data = make_test_chunk();
        cache.insert_snapshot(src, IVec3::ZERO, 1, &data).unwrap();
        cache.drain_dirty();

        let edits = vec![(5, 5, 5, BlockId::DIRT)];
        assert!(cache.apply_delta(src, IVec3::ZERO, 2, &edits));

        let chunk = cache.get_chunk(src, IVec3::ZERO).unwrap();
        assert_eq!(chunk.get_block(5, 5, 5), BlockId::DIRT);
    }

    #[test]
    fn reject_stale_delta() {
        let mut cache = ClientChunkCache::new();
        let src = cache.add_source();
        let data = make_test_chunk();
        cache.insert_snapshot(src, IVec3::ZERO, 5, &data).unwrap();
        cache.drain_dirty();

        let edits = vec![(5, 5, 5, BlockId::DIRT)];
        assert!(!cache.apply_delta(src, IVec3::ZERO, 3, &edits));
        assert!(!cache.apply_delta(src, IVec3::ZERO, 5, &edits));
        assert!(cache.apply_delta(src, IVec3::ZERO, 6, &edits));
    }

    #[test]
    fn insert_to_inactive_source_fails() {
        let mut cache = ClientChunkCache::new();
        let src = cache.add_source();
        cache.remove_source(src);

        let data = make_test_chunk();
        assert!(cache.insert_snapshot(src, IVec3::ZERO, 1, &data).is_err());
    }

    #[test]
    fn neighbor_dirty_within_same_source() {
        let mut cache = ClientChunkCache::new();
        let src = cache.add_source();
        let data = make_test_chunk();

        cache.insert_snapshot(src, IVec3::ZERO, 1, &data).unwrap();
        cache.insert_snapshot(src, IVec3::X, 1, &data).unwrap();
        cache.drain_dirty();

        // New snapshot for (0,0,0) should mark neighbor (1,0,0) dirty too.
        cache.insert_snapshot(src, IVec3::ZERO, 2, &data).unwrap();
        let dirty = cache.drain_dirty();
        assert!(dirty.iter().any(|k| k.chunk == IVec3::X && k.source == src));
    }

    #[test]
    fn clear_all_removes_everything() {
        let mut cache = ClientChunkCache::new();
        let src1 = cache.add_source();
        let src2 = cache.add_source();
        let data = make_test_chunk();

        cache.insert_snapshot(src1, IVec3::ZERO, 1, &data).unwrap();
        cache.insert_snapshot(src2, IVec3::X, 1, &data).unwrap();

        cache.clear_all();
        assert_eq!(cache.total_chunk_count(), 0);
        assert!(!cache.is_source_active(src1));
        assert!(!cache.is_source_active(src2));

        let removed = cache.drain_removed_sources();
        assert_eq!(removed.len(), 2);
    }
}
